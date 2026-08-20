//! Discrete distributional bar parametrization and emission head.
//!
//! A 5-minute aggregate carries exactly five degrees of freedom once the previous
//! close is known: the log return, the log range, the close and open positions
//! inside that range, and the log volume relative to a causal volume EMA. This
//! module maps `PackedBar` onto those five numbers ([`encode_dof`] /
//! [`decode_dof`]), discretizes each of them onto an equal-mass quantile support
//! fitted on training data ([`BarSupports`]), and predicts the bar as a chain of
//! five categoricals conditioned on a latent ([`BarEmissionHead`]).
//!
//! The chain order is [`BAR_CHAIN`] = `r -> s -> u -> v -> w`. `BAR_CHAIN[0]` is the
//! only factor with no prefix, hence the only one predicted from strictly past
//! information, so the traded degree of freedom leads; the range, the intra-bar shape
//! and the volume close the factorization behind it.
//!
//! This is deliberately independent of [`crate::torch::value::hl_gauss`]: that one
//! is the critic's value distribution over a symlog/±5σ support tuned for GAE
//! returns, whereas these supports are empirical per-DOF quantiles of the bar
//! observables and are persisted next to the checkpoint.

use std::fmt;
use std::path::Path;
use std::str::FromStr;

use anyhow::{anyhow, bail, ensure, Context, Result};
use rayon::slice::ParallelSliceMut;
use serde::{Deserialize, Serialize};
use shared::bars::PackedBar;
use tch::nn::Init;
use tch::{nn, Device, Kind, Tensor};

/// Number of free degrees of freedom in a bar given the previous close.
pub const BAR_DOF: usize = 5;
/// Categorical resolution of every per-DOF factor.
pub const NUM_BAR_BINS: i64 = 128;
/// Gaussian label-smoothing width, as a multiple of the local bin width CAPPED at the
/// per-DOF typical bin width (see [`BarSupports::smooth_sigma`]).
///
/// The cap is the whole content of the constant. Bins here are EQUAL-MASS quantile bins,
/// so their widths in return space span three orders of magnitude within one DOF — for the
/// live 300s supports, DOF `r` runs from 0.26 bps in the centre to 725 bps in the outermost
/// bin, a factor of 2743. An uncapped `0.75 * local_width` is therefore not one smoother
/// but 128 of them, and the tail ones are catastrophic: measured model-free on
/// `long_data/bars/bar_supports.300.json`, the uncapped kernel adds 0.21 bps of label
/// standard deviation in the centre and 373 bps in the outermost bins, it puts 56% of the
/// label mass further than six typical widths from the observation, and it biases the
/// label's implied mean by up to 465 bps — outward, not inward, in the second and
/// second-from-last bins. 99.5% of the total added label variance came from 4 of the 128
/// bins. Capping at the typical width leaves the centre bit-identical (bins at or below
/// the median keep their own width) and collapses the tail kernels to the containing bin,
/// which cuts the mean added label variance by 5546x on `r` and 4635x on `s`.
pub const BAR_LABEL_SIGMA_RATIO: f64 = 0.75;
/// Composite-midpoint nodes per continuous bin used by
/// [`BarSupports::scoring_floor`]. `H(t(x))` varies smoothly inside a bin, so 16
/// nodes pin the integral to well under a milli-nat while keeping the whole
/// evaluation at `NUM_BAR_BINS * 16` rows.
pub const SMOOTHING_FLOOR_NODES: usize = 16;
/// Width of the per-DOF conditioning embedding in the intra-bar chain.
pub const BAR_PREFIX_EMBED_DIM: i64 = 32;
/// Number of DOF that can ever appear as a chain prefix (everything but the last).
pub const BAR_PREFIX_SLOTS: usize = BAR_DOF - 1;
/// Span of the causal volume EMA that anchors the `w` degree of freedom.
pub const BAR_VOLUME_EMA_SPAN: f64 = 20.0;

// ---------------------------------------------------------------------------
// Scoring rule
// ---------------------------------------------------------------------------

/// Which scoring rule turns the predicted categorical chain into nats.
///
/// The three modes are NOT comparable to one another in absolute nats. They differ by
/// additive constants that depend on the binning, so a `density` figure is tens of nats
/// below a `smoothed` one on the identical model. Every artifact therefore records the
/// mode, the lineage hash covers it, and `pretrain-compare` refuses to pair two runs that
/// disagree.
///
/// * [`Self::Smoothed`] inherits the critic's HL-Gauss setting: the target is a Gaussian at
///   `BAR_LABEL_SIGMA_RATIO` local bin widths, discretized over the bins. That regularizes
///   against a NOISY target, which is what a bootstrapped value estimate is — and what a
///   bar observation is not. Kept because the campaign's earlier runs were scored under it.
/// * [`Self::Hard`] is the one-hot cross entropy on the containing bin: proper for the
///   discretized law, with no artificial floor, but its scale still moves with
///   [`NUM_BAR_BINS`] because finer bins mean a smaller per-bin probability.
/// * [`Self::Density`] is the proper log-likelihood of the MIXED measure we actually
///   observe: a probability MASS `P(atom)` on an atom, and a DENSITY `P_b / width_b` inside
///   a continuous bin. It has no floor and, up to discretization error, no dependence on
///   the bin count at all, which is what makes [`NUM_BAR_BINS`] ablatable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BarScoring {
    Smoothed,
    Hard,
    #[default]
    Density,
}

impl BarScoring {
    /// Every mode, in the order the banner and the reference tables list them.
    pub const ALL: [BarScoring; 3] = [Self::Smoothed, Self::Hard, Self::Density];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Smoothed => "smoothed",
            Self::Hard => "hard",
            Self::Density => "density",
        }
    }

    /// True when the observation's MEASURE enters the score, i.e. the continuous part is a
    /// log density rather than a log probability. Only [`Self::Density`] is.
    pub fn is_density(self) -> bool {
        matches!(self, Self::Density)
    }

    /// True when the target is a Gaussian smear rather than the containing bin, i.e. when
    /// the rule is proper for the SMOOTHED law and pays an unreachable floor.
    pub fn is_smoothed(self) -> bool {
        matches!(self, Self::Smoothed)
    }
}

impl fmt::Display for BarScoring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for BarScoring {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "smoothed" => Ok(Self::Smoothed),
            "hard" => Ok(Self::Hard),
            "density" => Ok(Self::Density),
            other => Err(format!(
                "unknown bar scoring mode {other:?}; expected one of smoothed, hard, density"
            )),
        }
    }
}

/// DOF slot indices. The tensor layout is always `[r, s, u, v, w]`.
pub const DOF_R: usize = 0;
pub const DOF_S: usize = 1;
pub const DOF_U: usize = 2;
pub const DOF_V: usize = 3;
pub const DOF_W: usize = 4;

/// Short names in tensor order, for report series naming.
pub const BAR_DOF_NAMES: [&str; BAR_DOF] = ["r", "s", "u", "v", "w"];

/// Autoregressive factorization order, as DOF slot indices.
pub const BAR_CHAIN: [usize; BAR_DOF] = [DOF_R, DOF_S, DOF_U, DOF_V, DOF_W];

/// Position of each DOF slot within [`BAR_CHAIN`]; a head may condition on every
/// prefix slot strictly below its own chain position.
const CHAIN_POS: [usize; BAR_DOF] = chain_positions();

/// `encode_dof` pins `u = v = 0.5` on a flat bar (`s == 0`, 14.76% of bars), and the fitted
/// support mandates that atom for both. Only a head that has already seen `s` can place
/// mass there, so `s` MUST precede both.
const _: () = assert!(
    CHAIN_POS[DOF_S] < CHAIN_POS[DOF_U] && CHAIN_POS[DOF_S] < CHAIN_POS[DOF_V],
    "u and v are degenerate at 0.5 whenever s == 0, so the chain must place s before both"
);

/// DOF slot occupying each prefix slot, i.e. `BAR_CHAIN` without the final entry.
const PREFIX_SLOT_DOF: [i64; BAR_PREFIX_SLOTS] = [
    BAR_CHAIN[0] as i64,
    BAR_CHAIN[1] as i64,
    BAR_CHAIN[2] as i64,
    BAR_CHAIN[3] as i64,
];

const BAR_PREFIX_WIDTH: i64 = BAR_PREFIX_SLOTS as i64 * BAR_PREFIX_EMBED_DIM;

/// Parameter-name fragments used by this module. The emission head is an output
/// projection, so the pretrain optimizer routes it to AdamW exactly like
/// `value_proj` / `next_return_head`.
pub const BAR_EMISSION_ADAMW_NAME_SUBSTRINGS: [&str; 2] = ["bar_dof_head", "bar_prefix_embed"];

const SQRT_2: f64 = std::f64::consts::SQRT_2;
/// Prices below this are treated as corrupt and clamped away from zero/negatives.
const PRICE_FLOOR: f64 = 1e-6;
/// Ceiling that keeps every reconstructed price inside f32 range.
const PRICE_CEIL: f64 = 1e30;
const VOLUME_FLOOR: f64 = 1e-3;
/// Hard bound on every log-space DOF, so corrupt inputs stay finite.
const LOG_LIMIT: f64 = 30.0;

const fn chain_positions() -> [usize; BAR_DOF] {
    let mut pos = [0usize; BAR_DOF];
    let mut c = 0;
    while c < BAR_DOF {
        pos[BAR_CHAIN[c]] = c;
        c += 1;
    }
    pos
}

// ---------------------------------------------------------------------------
// Contract B: bar parametrization
// ---------------------------------------------------------------------------

/// The five free degrees of freedom of a bar, in tensor order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarDof {
    /// `ln(close / prev_close)`.
    pub r: f32,
    /// `ln(high / low)`, non-negative.
    pub s: f32,
    /// Close position inside the log range, in `[0, 1]`.
    pub u: f32,
    /// Open position inside the log range, in `[0, 1]`.
    pub v: f32,
    /// `ln(volume / ema_volume)`.
    pub w: f32,
}

impl Default for BarDof {
    fn default() -> Self {
        Self {
            r: 0.0,
            s: 0.0,
            u: 0.5,
            v: 0.5,
            w: 0.0,
        }
    }
}

impl BarDof {
    pub fn to_array(self) -> [f32; BAR_DOF] {
        [self.r, self.s, self.u, self.v, self.w]
    }

    pub fn from_array(values: [f32; BAR_DOF]) -> Self {
        Self {
            r: values[DOF_R],
            s: values[DOF_S],
            u: values[DOF_U],
            v: values[DOF_V],
            w: values[DOF_W],
        }
    }

    pub fn is_finite(&self) -> bool {
        self.to_array().iter().all(|x| x.is_finite())
    }
}

fn positive_finite(x: f32) -> Option<f64> {
    let v = x as f64;
    (v.is_finite() && v > 0.0).then_some(v)
}

fn safe_price(x: f32, fallback: f64) -> f64 {
    positive_finite(x).unwrap_or(fallback).clamp(PRICE_FLOOR, PRICE_CEIL)
}

fn finite_or(x: f32, fallback: f64) -> f64 {
    let v = x as f64;
    if v.is_finite() {
        v
    } else {
        fallback
    }
}

/// Map a bar onto its five degrees of freedom relative to `prev_close` and the
/// causal volume reference `ema_volume`.
///
/// Corrupt inputs never produce NaN: non-positive or non-finite prices fall back
/// to the first usable price on the bar, the range is widened to contain open and
/// close so `u`/`v` stay in `[0, 1]`, a flat bar yields `s = 0` and `u = v = 0.5`,
/// and every log-space value is clamped to `±30`.
pub fn encode_dof(prev_close: f32, bar: &PackedBar, ema_volume: f32) -> BarDof {
    // `PackedBar` is `repr(C, packed)`; bind by value, never by reference.
    let (raw_open, raw_high, raw_low, raw_close) = (bar.open, bar.high, bar.low, bar.close);
    let anchor = positive_finite(raw_close)
        .or_else(|| positive_finite(prev_close))
        .or_else(|| positive_finite(raw_open))
        .or_else(|| positive_finite(raw_high))
        .or_else(|| positive_finite(raw_low))
        .unwrap_or(1.0)
        .clamp(PRICE_FLOOR, PRICE_CEIL);

    let close = safe_price(raw_close, anchor);
    let open = safe_price(raw_open, close);
    let high_in = safe_price(raw_high, close);
    let low_in = safe_price(raw_low, close);
    let high = high_in.max(low_in).max(open).max(close);
    let low = low_in.min(high_in).min(open).min(close);
    let prev = safe_price(prev_close, close);

    let (ln_open, ln_high, ln_low, ln_close) = (open.ln(), high.ln(), low.ln(), close.ln());
    let range = ln_high - ln_low;
    let s = range.clamp(0.0, LOG_LIMIT);
    let r = (ln_close - prev.ln()).clamp(-LOG_LIMIT, LOG_LIMIT);
    // Divide by the true range, not the clamped `s`, so a pathologically wide bar
    // keeps its intra-bar positions instead of saturating both to 1.
    let (u, v) = if range > 0.0 {
        (
            ((ln_close - ln_low) / range).clamp(0.0, 1.0),
            ((ln_open - ln_low) / range).clamp(0.0, 1.0),
        )
    } else {
        (0.5, 0.5)
    };

    let volume = positive_finite(bar.volume).unwrap_or(VOLUME_FLOOR).max(VOLUME_FLOOR);
    let reference = positive_finite(ema_volume).unwrap_or(volume).max(VOLUME_FLOOR);
    let w = (volume / reference).ln().clamp(-LOG_LIMIT, LOG_LIMIT);

    BarDof {
        r: r as f32,
        s: s as f32,
        u: u as f32,
        v: v as f32,
        w: w as f32,
    }
}

/// Reconstruct a bar from its five degrees of freedom.
///
/// `low <= min(open, close) <= max(open, close) <= high` holds by construction:
/// the four log prices are built as `ln_low + {0, v, u, 1} * s` with `s >= 0` and
/// `u, v` clamped to `[0, 1]`, and both `exp` and the `f32` narrowing are
/// monotone. `vwap` is the geometric mean of the four reconstructed prices (in
/// range by the same monotonicity) and `trades` is zero: neither is a modelled
/// degree of freedom. `ts_ms` is left at zero for the caller to stamp.
pub fn decode_dof(prev_close: f32, dof: &BarDof, ema_volume: f32) -> PackedBar {
    let prev = safe_price(prev_close, 1.0);
    let r = finite_or(dof.r, 0.0).clamp(-LOG_LIMIT, LOG_LIMIT);
    let s = finite_or(dof.s, 0.0).clamp(0.0, LOG_LIMIT);
    let u = finite_or(dof.u, 0.5).clamp(0.0, 1.0);
    let v = finite_or(dof.v, 0.5).clamp(0.0, 1.0);
    let w = finite_or(dof.w, 0.0).clamp(-LOG_LIMIT, LOG_LIMIT);

    let ln_close = prev.ln() + r;
    let ln_low = ln_close - u * s;
    let ln_high = ln_low + s;
    let ln_open = ln_low + v * s;
    let price = |ln_p: f64| ln_p.exp().clamp(PRICE_FLOOR, PRICE_CEIL) as f32;

    let reference = positive_finite(ema_volume).unwrap_or(VOLUME_FLOOR).max(VOLUME_FLOOR);
    let volume = (reference * w.exp()).clamp(0.0, PRICE_CEIL) as f32;

    PackedBar {
        ts_ms: 0,
        open: price(ln_open),
        high: price(ln_high),
        low: price(ln_low),
        close: price(ln_close),
        volume,
        vwap: price(0.25 * (ln_open + ln_high + ln_low + ln_close)),
        trades: 0,
    }
}

/// Causal EMA of bar volume, the reference for the `w` degree of freedom.
///
/// The reference for bar `t` is the EMA over bars `< t`, so `w` never sees its own
/// volume. Before any observation the reference is the bar's own volume, making
/// `w = 0` on the first bar of a series.
#[derive(Clone, Copy, Debug)]
pub struct VolumeEma {
    alpha: f64,
    value: f64,
    initialized: bool,
}

impl VolumeEma {
    pub fn new(span: f64) -> Self {
        assert!(span >= 1.0, "volume EMA span must be at least 1");
        Self {
            alpha: 2.0 / (span + 1.0),
            value: 0.0,
            initialized: false,
        }
    }

    /// Volume reference to encode a bar carrying `volume`.
    pub fn reference_for(&self, volume: f32) -> f32 {
        if self.initialized {
            self.value as f32
        } else {
            positive_finite(volume).unwrap_or(VOLUME_FLOOR) as f32
        }
    }

    pub fn observe(&mut self, volume: f32) {
        let v = positive_finite(volume).unwrap_or(VOLUME_FLOOR);
        if self.initialized {
            self.value += self.alpha * (v - self.value);
        } else {
            self.value = v;
            self.initialized = true;
        }
    }
}

impl Default for VolumeEma {
    fn default() -> Self {
        Self::new(BAR_VOLUME_EMA_SPAN)
    }
}

/// Encode a contiguous bar series, carrying the previous close and the causal
/// volume EMA forward. The first bar has no predecessor and is skipped, so the
/// result aligns with `bars[1..]`.
pub fn encode_series(bars: &[PackedBar]) -> Vec<BarDof> {
    if bars.len() < 2 {
        return Vec::new();
    }
    let mut ema = VolumeEma::default();
    let first_volume = bars[0].volume;
    ema.observe(first_volume);
    let mut out = Vec::with_capacity(bars.len() - 1);
    let mut prev_close = bars[0].close;
    for bar in &bars[1..] {
        let volume = bar.volume;
        out.push(encode_dof(prev_close, bar, ema.reference_for(volume)));
        ema.observe(volume);
        prev_close = bar.close;
    }
    out
}

// ---------------------------------------------------------------------------
// Contract C: atom + equal-mass-continuous mixture supports
// ---------------------------------------------------------------------------

/// An exact value that carries a point mass and therefore owns a dedicated,
/// zero-width bin.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BarAtom {
    pub value: f64,
    pub bin: usize,
    /// Empirical share of the fitted sample sitting exactly on `value`. Read from
    /// the fitted bin histogram, so the atom table and the marginal reference can
    /// never disagree.
    pub mass: f64,
}

/// One DOF's bin geometry, borrowed out of a [`BarSupports`] without its tensors.
///
/// Exists for exactly one reason: [`BarSupports`] owns [`Tensor`] fields, so it is neither `Send`
/// nor `Sync` and no parallel fold can borrow it, while the host-side binning rule needs nothing
/// but three slices of plain numbers. [`BarSupports::bin_of`] is DEFINED as this type's `bin_of`,
/// so there is one rule and not two, and a streaming audit over the whole corpus places bars in
/// exactly the bins a training step would.
#[derive(Clone, Copy, Debug)]
pub struct DofBinner<'a> {
    lo: &'a [f64],
    hi: &'a [f64],
    atoms: &'a [BarAtom],
}

impl DofBinner<'_> {
    /// Exact matches on an atom take the atom bin; every other value takes the last bin whose
    /// lower bound it reaches, after clamping onto the support. The value is narrowed to `f32`
    /// first, because the support is an `f32` object and the tensor twin compares in that
    /// precision.
    pub fn bin_of(&self, value: f64) -> usize {
        let bins = self.lo.len();
        let clamped = ((value as f32) as f64).clamp(self.lo[0], self.hi[bins - 1]);
        if let Some(atom) = self.atoms.iter().find(|a| a.value == clamped) {
            return atom.bin;
        }
        let count = self.lo.partition_point(|&bound| bound <= clamped);
        count.saturating_sub(1).min(bins - 1)
    }
}

/// Everything about the corpus and the fit that decides what these bins MEAN.
///
/// The supports define the model's output space and therefore the `nll_bar` scale, so two
/// runs whose supports were fitted on different data are not comparable no matter how
/// carefully everything else is pinned. A bin-count check cannot see that; this can.
/// Deliberately reused across an ablation campaign — a frozen support is the right call for
/// comparability — but the freeze has to be a recorded decision, not a filesystem accident.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BarSupportsProvenance {
    /// `BarCorpus::identity_fingerprint()` of the corpus the fit sampled from.
    pub corpus_fingerprint: String,
    /// `(train|val, val|test)` split instants in epoch millis. The fit draws from the train
    /// region only, so these bound what the supports were allowed to see.
    pub split_bounds: (i64, i64),
    /// Training DOF actually drawn for the fit.
    pub sample_count: usize,
    /// UTC ISO-8601 instant the fit completed.
    pub fitted_utc: String,
}

#[derive(Serialize, Deserialize)]
struct BarSupportsJson {
    format_version: u32,
    num_bins: i64,
    dof_names: Vec<String>,
    /// `BAR_DOF` rows of `num_bins` bin lower bounds.
    lo: Vec<Vec<f64>>,
    /// `BAR_DOF` rows of `num_bins` bin upper bounds; `hi[j] == lo[j + 1]`.
    hi: Vec<Vec<f64>>,
    /// `BAR_DOF` rows of `num_bins` empirical bin probabilities, measured on the
    /// fit sample with the same rule `bin_of` applies.
    masses: Vec<Vec<f64>>,
    /// `BAR_DOF` rows of `num_bins`: the MEAN SMOOTHED TARGET `q* = E_x[t(x)]` over
    /// the fit sample. `H(q*)` is exactly the soft-target loss an optimal marginal
    /// head achieves, which is the meaningful reference line for `nll_bar`.
    smoothed_marginal: Vec<Vec<f64>>,
    /// Absent in [`BAR_SUPPORTS_LEGACY_VERSION`] files, which is itself the signal that the
    /// artifact predates provenance tracking and cannot be verified.
    #[serde(default)]
    provenance: Option<BarSupportsProvenance>,
    /// `BAR_DOF` rows of `num_bins`: `E[x | x in bin]` on the fit sample, over RAW
    /// unclamped observations. Absent below [`BAR_SUPPORTS_MOMENTS_VERSION`], which is
    /// itself the signal that a reader must not substitute midpoints or bounds for them.
    #[serde(default)]
    bin_means: Option<Vec<Vec<f64>>>,
    /// `BAR_DOF` rows of `num_bins`: `E[x^2 | x in bin]`, same sample and same rule.
    #[serde(default)]
    bin_second_moments: Option<Vec<Vec<f64>>>,
}

impl BarSupportsJson {
    /// The ONLY place [`BAR_SUPPORTS_FORMAT_VERSION`] is stamped, and it takes the fitted
    /// moments as a REQUIRED argument. The version and the content that version promises
    /// therefore cannot come apart: this constructor is unnameable without moments in hand,
    /// so no code path can produce a v5 value whose moments are absent.
    fn v5(supports: &BarSupports, moments: &BarBinMoments) -> Self {
        Self {
            format_version: BAR_SUPPORTS_FORMAT_VERSION,
            num_bins: NUM_BAR_BINS,
            dof_names: BAR_DOF_NAMES.iter().map(|s| (*s).to_owned()).collect(),
            lo: supports.lo.iter().cloned().collect(),
            hi: supports.hi.iter().cloned().collect(),
            masses: supports.masses.iter().cloned().collect(),
            smoothed_marginal: supports.smoothed_marginal.iter().cloned().collect(),
            provenance: supports.provenance.clone(),
            bin_means: Some(moments.mean.iter().cloned().collect()),
            bin_second_moments: Some(moments.second.iter().cloned().collect()),
        }
    }
}

/// Current persisted schema. v4 adds [`BarSupportsProvenance`]; v5 adds the fitted per-bin
/// conditional moments that [`BarSupports::bin_means`] exposes.
pub(crate) const BAR_SUPPORTS_FORMAT_VERSION: u32 = 5;
/// First schema carrying fitted per-bin moments. Below it `bin_means_measured()` is false and
/// no consumer may invent a substitute — that is the whole point of versioning them separately.
pub(crate) const BAR_SUPPORTS_MOMENTS_VERSION: u32 = 5;
/// Still readable, and deliberately so: the campaign's live supports were written under v3/v4 and
/// refitting them would move the `nll_bar` scale mid-campaign. They load with no provenance
/// (v3) and no fitted moments (v3, v4), which the caller must then accept explicitly rather
/// than by default.
const BAR_SUPPORTS_LEGACY_VERSION: u32 = 3;
/// Every schema this build accepts. An unlisted version is refused outright rather than
/// coerced, because a support whose geometry we cannot name is not a support.
const BAR_SUPPORTS_READABLE_VERSIONS: [u32; 3] = [5, 4, BAR_SUPPORTS_LEGACY_VERSION];

/// `format_version` of the artifact at `path`, read WITHOUT building a support.
///
/// A loaded [`BarSupports`] carries the CONTENT of its file and deliberately not its version
/// number, because every consumer must branch on what the support HAS
/// ([`BarSupports::bin_means_measured`]) rather than on a number it claims. A refusal message is
/// the one place the number itself is the useful fact, so it is read from disk on demand.
pub(crate) fn bar_supports_format_version(path: &Path) -> Result<u32> {
    #[derive(Deserialize)]
    struct VersionOnly {
        format_version: u32,
    }
    let body = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let probe: VersionOnly = serde_json::from_slice(&body)
        .with_context(|| format!("parsing the format version of {}", path.display()))?;
    Ok(probe.format_version)
}

/// Continuous rows used to estimate the Gaussian half of the mean smoothed target.
///
/// The atom half is exact regardless of this cap. For the continuous half the error in
/// `H(q*)` behaves like `ln(NUM_BAR_BINS) / sqrt(n)`, so 512k rows pins the reference
/// to roughly `7e-3` nats — under 0.3% of the ~2.6 nat headroom between the uniform
/// and marginal baselines, and far below any difference we would act on. Bounding it
/// keeps `fit` from running a 40M-row erf pass.
const MARGINAL_ESTIMATE_ROWS: usize = 512_000;

/// Exact values holding at least this share of the sample are promoted to atoms.
pub const BAR_ATOM_MASS_THRESHOLD: f64 = 0.005;
/// Upper bound on atoms per DOF, so continuous bins never drop below 120.
pub const MAX_BAR_ATOMS: usize = 8;

/// Tail fraction excluded when choosing the OUTER support bounds.
///
/// Genuine bad ticks are vanishingly rare (35 bars in 255M exceed a 4x move) so they
/// cannot shift an equal-mass bin's mass, but they do define the outermost EDGE, and
/// a bin whose center decodes to a 7000x bar makes `expectation`, `sample` and the
/// candle reports nonsense. Bounding the support at these quantiles keeps the
/// outermost bins as open-ended catch-alls: out-of-range observations still bin into
/// them and still score, so a real 40% move remains representable and is never
/// dropped or winsorized. Only what the edge bin DECODES to changes.
pub const BAR_SUPPORT_CLIP_QUANTILE: f64 = 1e-4;

/// Atoms every fit reserves whether or not the sample happens to contain them, so
/// the support schema is identical across datasets and ablations and held-out
/// `nll_bar` stays comparable. `s == 0` is the flat bar; `u`/`v` in `{0, 0.5, 1}`
/// are the flat bar and the close/open sitting exactly on a bar extreme.
fn mandated_atoms(dof: usize) -> &'static [f32] {
    match dof {
        DOF_S => &[0.0],
        DOF_U | DOF_V => &[0.0, 0.5, 1.0],
        _ => &[],
    }
}

/// Per-DOF support: a value-ordered tiling of `NUM_BAR_BINS` bins over
/// `[lo[0], hi[NUM_BAR_BINS - 1]]`, where `hi[j] == lo[j + 1]` exactly.
///
/// A bin with `hi == lo` is an ATOM bin holding the point mass at that value. A
/// bin with `hi > lo` is a continuous bin; within each inter-atom segment the
/// continuous bins carry equal empirical mass, and their outer edges are pinned to
/// the enclosing atoms so the tiling has no gaps.
///
/// The geometry alone makes almost everything atom-aware for free: the Gaussian
/// kernel integrates to exactly zero over a zero-width bin, so atoms never receive
/// smoothing spill; `lo + (hi - lo) * u` reproduces an atom's exact value; and an
/// atom bin's center *is* the atom, so `expectation` and the CRPS atom identity are
/// unchanged. Only [`Self::encode_targets`] (exact one-hot on an atom observation)
/// and [`bar_pit_from_logits`] (randomized PIT across an atom's probability
/// interval) branch on atom-ness.
#[derive(Debug)]
pub struct BarSupports {
    lo: [Vec<f64>; BAR_DOF],
    hi: [Vec<f64>; BAR_DOF],
    centers: [Vec<f64>; BAR_DOF],
    widths: [Vec<f64>; BAR_DOF],
    atoms: [Vec<BarAtom>; BAR_DOF],
    /// Empirical bin probabilities from the fit sample.
    masses: [Vec<f64>; BAR_DOF],
    /// Mean smoothed target `q* = E_x[t(x)]` from the fit sample.
    smoothed_marginal: [Vec<f64>; BAR_DOF],
    /// Fitted per-bin conditional moments, or `None` on a pre-v5 artifact.
    ///
    /// SEPARATE from [`Self::centers`] on purpose, and the two must never be merged.
    /// `centers` decodes the catch-all bins to their outer BOUNDS, which is correct for
    /// sampling, candle rendering and the CRPS grid — it refuses to invent an extreme
    /// nobody observed. It is wrong by a factor of about 3.1 for any MOMENT: read as a
    /// representative value it inflates the marginal variance of `r` by 5.25x over the
    /// realized law, with 92.4% of the predicted second moment coming from 1.45% of the
    /// mass. Moment consumers read these; sampling consumers read `centers`.
    bin_moments: Option<BarBinMoments>,
    device: Device,
    /// `[BAR_DOF, NUM_BAR_BINS]`, non-decreasing; drives the continuous lookup.
    lo_t: Tensor,
    /// `[BAR_DOF, NUM_BAR_BINS]`
    hi_t: Tensor,
    /// `[BAR_DOF, NUM_BAR_BINS]`
    centers_t: Tensor,
    /// `[BAR_DOF, MAX_BAR_ATOMS]`, NaN-padded so padding matches nothing.
    atom_value_t: Tensor,
    /// `[BAR_DOF, MAX_BAR_ATOMS]` i64 bin index of each atom; padding is zero.
    atom_bin_t: Tensor,
    /// `[BAR_DOF * NUM_BAR_BINS]` flat lookups.
    lo_flat: Tensor,
    hi_flat: Tensor,
    centers_flat: Tensor,
    widths_flat: Tensor,
    /// `[BAR_DOF]`, `dof * NUM_BAR_BINS`.
    bin_offsets: Tensor,
    /// `[BAR_DOF, 1]`, the narrowest continuous bin per DOF; floors the kernel width so
    /// an atom observation can never produce a degenerate sigma.
    min_width_t: Tensor,
    /// `[BAR_DOF, 1]`, the MEDIAN continuous bin width per DOF: the resolution at which
    /// the mass actually sits, since equal-mass bins put half the observations in bins at
    /// or below it. Ceilings the label-smoothing kernel so the width is bounded in RETURN
    /// space rather than tracking a tail bin three orders of magnitude wider than the
    /// centre. See [`BAR_LABEL_SIGMA_RATIO`] for the measured cost of not capping.
    ///
    /// A DOF with no continuous bin at all leaves this `+inf`, which is inert: every
    /// observation of such a DOF is an atom and takes the exact one-hot branch.
    cap_width_t: Tensor,
    /// Where these bins came from. `None` for a legacy artifact or a freshly fitted
    /// support the caller has not stamped yet; never inferred, because a guessed
    /// provenance is worse than an absent one.
    provenance: Option<BarSupportsProvenance>,
}

/// Fitted `E[x | bin]` and `E[x^2 | bin]` per DOF, measured on the fit sample.
///
/// Both are accumulated over RAW, UNCLAMPED observations, binned by the same rule
/// [`BarSupports::bin_of`] applies. That is what makes the mean unbiased: `bin_of` clamps,
/// so the outermost bin's probability is the probability that `x` lands ANYWHERE beyond the
/// support bound, and the representative that makes `sum_b p_b m_b` an unbiased estimate of
/// `E[x]` is therefore the untruncated `E[x | x beyond bound]`. A mean measured on clamped
/// values would be pulled toward the bound and reproduce the very bias being removed, and
/// `m_b` may legitimately fall outside `[lo_b, hi_b]` for the two catch-alls.
///
/// CAVEAT ON THE SECOND MOMENT, which a consumer must not launder into a population
/// constant: `r` has a measured tail exponent near 1.8, so `E[x^2]` DOES NOT EXIST in the
/// population and the outer entries are sample statistics that grow with sample size and
/// with wherever the support clip was placed. They are strictly better than decoding a
/// second moment off the bounds — that overstates by 9.6x on the lever arm alone — but a
/// variance built from them is a statement about this sample's truncation, not a converged
/// quantity. The mean has no such problem: first moments converge at exponent 1.8, which is
/// what makes the Mincer-Zarnowitz MEAN slope a well-posed calibration target and the
/// variance slope not one.
// No `Clone`: `Tensor` has none, and `to_device` is the only copy anyone needs.
#[derive(Debug)]
struct BarBinMoments {
    mean: [Vec<f64>; BAR_DOF],
    second: [Vec<f64>; BAR_DOF],
    /// `[BAR_DOF, NUM_BAR_BINS]` device copies for the tensor path.
    mean_t: Tensor,
    second_t: Tensor,
}

impl BarBinMoments {
    fn new(mean: [Vec<f64>; BAR_DOF], second: [Vec<f64>; BAR_DOF], device: Device) -> Self {
        // Narrowed to f32 for the tensor path and re-widened on the host side, exactly as
        // the bounds are, so a host lookup and a device lookup agree bit for bit and a JSON
        // round trip changes nothing.
        let flat = |rows: &[Vec<f64>; BAR_DOF]| -> Vec<f32> {
            rows.iter()
                .flat_map(|row| row.iter().map(|&x| x as f32))
                .collect()
        };
        let shape = [BAR_DOF as i64, NUM_BAR_BINS];
        let mean_t = Tensor::from_slice(&flat(&mean))
            .view(shape)
            .to_device(device);
        let second_t = Tensor::from_slice(&flat(&second))
            .view(shape)
            .to_device(device);
        let narrow = |rows: [Vec<f64>; BAR_DOF]| -> [Vec<f64>; BAR_DOF] {
            rows.map(|row| row.into_iter().map(|x| x as f32 as f64).collect())
        };
        Self {
            mean: narrow(mean),
            second: narrow(second),
            mean_t,
            second_t,
        }
    }

    fn to_device(&self, device: Device) -> Self {
        Self {
            mean: self.mean.clone(),
            second: self.second.clone(),
            mean_t: self.mean_t.to_device(device),
            second_t: self.second_t.to_device(device),
        }
    }
}

/// Which per-bin representative a FIRST-MOMENT decode reads.
///
/// The two conventions are NOT interchangeable and the difference is not small: on the live
/// `r` support the two catch-all bins hold 1.4474% of the marginal mass yet control 41.00% of
/// the absolute first moment, 92.38% of the central second moment and 84.67% of the reachable
/// mean span under [`Self::Edge`]. Under [`Self::Fitted`] the same three shares are what the
/// corpus actually realized there. So which convention a consumer reads is a first-order
/// property of every predicted mean, not a rounding choice, and it is therefore named,
/// selected explicitly, and recorded rather than inferred.
///
/// [`Self::Edge`] IS THE DEFAULT AND STAYS THE DEFAULT. Every historical number in this tree —
/// the Mincer-Zarnowitz mean slopes, the Kelly bets, the horizon frontier, the skill deciles —
/// was measured under it, and flipping the default would silently move every one of them
/// without moving the artifact they were computed from. A consumer that wants the fitted
/// decode asks for it by name.
///
/// EVERY PRODUCTION CONSUMER OF THE DECODE, and exactly what switching each one to
/// [`Self::Fitted`] would change. Enumerated here rather than left to a grep because a partial
/// switch is worse than none UNLESS the split is deliberate and written down: two consumers on
/// two conventions otherwise produce two incomparable predicted means with no error anywhere.
/// Located by symbol; the line numbers move.
///
/// The split is deliberate, and it is drawn in exactly one place — between the OBJECTIVE and
/// every MEASUREMENT. Item (2) is the sole objective-side consumer and the sole consumer read
/// under [`Self::Fitted`]. Items (1) and (3) through (7) are measurement-side and stay on
/// [`Self::Edge`], so every number this tree has ever reported remains comparable with the one
/// before it. FIRST-MOMENT consumers, all of which WOULD move if switched:
/// 1. [`BarSupports::expectation`] — `sum_b p_b centers_b` over `[..., BAR_DOF, NUM_BAR_BINS]`
///    logits, the generic predicted mean. Switching moves the predicted mean of EVERY DOF by
///    `p_0 (m_0 - lo_0) + p_127 (m_127 - hi_127)`, which on `r` is the whole 3.1x catch-all
///    re-pricing; it is the single highest-leverage switch and everything else inherits from it.
/// 2. `train::growth::GrowthSupport::new` — maps the decode through `exp_m1` into the per-bin
///    SIMPLE returns of the expected-log-growth term. SWITCHED, and the only one: it requests
///    [`Self::Fitted`] by name and ERRORS on a support without measured moments rather than
///    degrading. Under [`Self::Edge`] it priced `r`'s two open-ended bins at
///    `exp_m1(-883.32 bps) = -8.4543%` and `exp_m1(+880.38 bps) = +9.2030%` against fitted
///    conditional means of `-2.7794%` and `+2.9014%` — 3.04x and 3.17x too much on 1.4474% of
///    the mass that carries 92.38% of the central second moment, so the cheapest route to
///    expected log growth was to move mass into two bins that overpaid threefold. An objective
///    may not pay for an outcome the corpus never realized; a measurement must keep its
///    convention. That is the whole of the split. This consumer reaches NO solver: the growth
///    term's `f_raw = mu/var` is saturated at `train::trade_bench::LEVERAGE_CAP` and its log
///    argument is guarded off the support BOUNDS rather than off the decode, so the ruin-point
///    and bracket consequence described under (6) does not arise here — it belongs to (6)
///    alone, which is unchanged.
/// 3. `train::horizon` (the frontier's one-bar decode) — the predicted mean the break-even cost
///    curve is built on. Switching lowers every predicted `|mu|`, so break-even cost falls and
///    the frontier shifts DOWN; the model-versus-baseline ORDERING is preserved only if the
///    baselines are re-decoded in the same pass, which is why this one must not be switched
///    alone.
/// 4. `train::horizon` (the k-bar aggregate decode) — same tensor, same effect, aggregated over
///    the holding horizon, so the shift compounds with `k`.
/// 5. `train::skill::SkillCutpoints::with_support_geometry` — the decode defines the CONFIDENCE
///    DECILE cutpoints. Switching compresses the outer deciles, so the same bars land in
///    different deciles: the skill profile's x-axis moves and no decile is comparable across the
///    switch. Deciles are the one place a decode change is invisible in the y-values and total
///    in the x-binning.
/// 6. `train::trade_bench::bin_returns` — the `exp_m1` per-bin simple returns the Kelly solve
///    and every bench policy price their bets with. Same ruin-point consequence as (2).
/// 7. `train::trade_bench` (the per-chunk `centers` row on the window paths) — feeds
///    `predicted_mean`, `predicted_var` and hence the Mincer-Zarnowitz mean slope. This is the
///    consumer the whole decode investigation is about: the slope is
///    `Cov(mu, r) / Var(mu)`, so shrinking the catch-all decode shrinks `Var(mu)` far faster
///    than `Cov(mu, r)` — 92.38% of `Var(mu)` sits in those two bins — and the slope RISES.
///
/// NON-first-moment consumers of the SAME array, which must NOT be switched:
/// 8. [`BarSupports`]'s sampling path (`sample_flat`) — decodes a DRAWN bin to a value. The
///    bound is correct here: a sample from the catch-all should not be an extreme nobody
///    observed, and the conditional MEAN of a catch-all is not a plausible draw from it.
/// 9. [`bar_crps_from_logits`] — treats the bins as atoms at their decode values. CRPS is a
///    distributional score, not a first moment, and its atom grid is a property of the
///    geometry.
///
/// A `#[cfg(test)]`-only site in `train::skill` also reads `centers(DOF_R)` to rebuild `mu` for
/// a test; it is NOT a production consumer and is listed only because it has been miscounted as
/// one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MeanDecode {
    /// Bin MIDPOINTS, with the two outermost bins pinned to the support BOUNDS.
    ///
    /// Correct for sampling, candle rendering and the CRPS grid: it refuses to invent an
    /// extreme nobody observed, and `lo + (hi - lo) * u` reproduces an atom exactly. Wrong
    /// for a moment, because a catch-all's bound is not its conditional mean — it is the
    /// nearest value the bin contains, and the bin extends to infinity.
    #[default]
    Edge,
    /// The FITTED conditional mean `E[x | x in bin]`, measured on the fit sample over RAW
    /// unclamped observations.
    ///
    /// The unbiased choice for any first moment: `bin_of` clamps, so the outermost bin's
    /// probability is the probability that `x` lands ANYWHERE beyond the bound, and the
    /// representative that makes `sum_b p_b d_b` unbiased for `E[x]` is the untruncated
    /// `E[x | x beyond bound]`. Available only on a support carrying measured moments; on a
    /// pre-v5 artifact asking for it is an ERROR rather than a silent fall back to
    /// [`Self::Edge`], because a geometric stand-in presented as a measurement is exactly the
    /// failure this enum exists to make unrepresentable.
    Fitted,
}

impl fmt::Display for MeanDecode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Edge => "edge",
            Self::Fitted => "fitted",
        })
    }
}

impl BarSupports {
    /// Fit the mixture support from encoded bars. Each DOF is fitted independently;
    /// non-finite values are dropped. One column is materialized at a time, so the
    /// transient cost is `4 * samples.len()` bytes.
    pub fn fit(samples: &[BarDof]) -> Self {
        assert!(
            !samples.is_empty(),
            "bar supports need at least one sample to fit"
        );
        let mut lo_rows: Vec<Vec<f64>> = Vec::with_capacity(BAR_DOF);
        let mut hi_rows: Vec<Vec<f64>> = Vec::with_capacity(BAR_DOF);
        for dof in 0..BAR_DOF {
            let mut column: Vec<f32> = samples
                .iter()
                .map(|s| s.to_array()[dof])
                .filter(|x| x.is_finite())
                // Collapse -0.0 onto 0.0 so a signed zero cannot split an atom.
                .map(|x| if x == 0.0 { 0.0 } else { x })
                .collect();
            assert!(
                !column.is_empty(),
                "DOF {} has no finite samples to fit a support",
                BAR_DOF_NAMES[dof]
            );
            column.par_sort_unstable_by(f32::total_cmp);
            let (lo, hi) = fit_dof_support(&column, mandated_atoms(dof));
            lo_rows.push(lo);
            hi_rows.push(hi);
        }
        let mut lo_iter = lo_rows.into_iter();
        let mut hi_iter = hi_rows.into_iter();
        let lo: [Vec<f64>; BAR_DOF] =
            std::array::from_fn(|_| lo_iter.next().expect("one row per DOF"));
        let hi: [Vec<f64>; BAR_DOF] =
            std::array::from_fn(|_| hi_iter.next().expect("one row per DOF"));

        // The geometry is enough to bin and to smooth, so measure both statistics
        // through a provisional support. That way the histogram uses exactly
        // `bin_of` and the marginal uses exactly `encode_targets`, with no second
        // implementation of either rule to drift out of sync.
        let uniform = || -> [Vec<f64>; BAR_DOF] {
            std::array::from_fn(|_| vec![1.0 / NUM_BAR_BINS as f64; NUM_BAR_BINS as usize])
        };
        let geometry = Self::from_bins(lo.clone(), hi.clone(), uniform(), uniform(), Device::Cpu)
            .expect("fitted supports are well formed");
        let masses = geometry.measure_bin_masses(samples);
        let smoothed_marginal = geometry.measure_smoothed_marginal(samples, &masses);
        Self::from_bins(lo, hi, masses, smoothed_marginal, Device::Cpu)
            .expect("fitted supports are well formed")
            .with_measured_bin_moments(samples)
    }

    /// Empirical bin probabilities of `samples`, using exactly [`Self::bin_of`].
    fn measure_bin_masses(&self, samples: &[BarDof]) -> [Vec<f64>; BAR_DOF] {
        let bins = NUM_BAR_BINS as usize;
        let mut counts: [Vec<f64>; BAR_DOF] = std::array::from_fn(|_| vec![0.0; bins]);
        let mut seen = [0usize; BAR_DOF];
        for sample in samples {
            let values = sample.to_array();
            for dof in 0..BAR_DOF {
                if values[dof].is_finite() {
                    counts[dof][self.bin_of(dof, values[dof] as f64)] += 1.0;
                    seen[dof] += 1;
                }
            }
        }
        std::array::from_fn(|dof| {
            let total = seen[dof].max(1) as f64;
            counts[dof].iter().map(|c| c / total).collect()
        })
    }

    /// Mean smoothed target `q* = E_x[t(x)]` over `samples`.
    ///
    /// Split so the expensive half is bounded and the exact half stays exact. An atom
    /// observation's target is a one-hot on its own bin, so atoms contribute their
    /// EXACT histogram mass with no kernel evaluation at all — which matters because
    /// atoms carry up to half the mass on `u`/`v`. Only the continuous rows need the
    /// Gaussian, and those are estimated from a deterministic stride of at most
    /// [`MARGINAL_ESTIMATE_ROWS`] rows evaluated in chunks.
    fn measure_smoothed_marginal(
        &self,
        samples: &[BarDof],
        masses: &[Vec<f64>; BAR_DOF],
    ) -> [Vec<f64>; BAR_DOF] {
        let bins = NUM_BAR_BINS as usize;
        let stride = samples.len().div_ceil(MARGINAL_ESTIMATE_ROWS).max(1);
        let rows: Vec<BarDof> = samples
            .iter()
            .step_by(stride)
            .copied()
            .filter(BarDof::is_finite)
            .collect();

        // Exact atom contribution, straight from the full-sample histogram.
        let mut marginal: [Vec<f64>; BAR_DOF] = std::array::from_fn(|_| vec![0.0; bins]);
        let mut atom_mass = [0.0f64; BAR_DOF];
        for dof in 0..BAR_DOF {
            for atom in &self.atoms[dof] {
                marginal[dof][atom.bin] = masses[dof][atom.bin];
                atom_mass[dof] += masses[dof][atom.bin];
            }
        }
        if rows.is_empty() {
            for dof in 0..BAR_DOF {
                let spread = (1.0 - atom_mass[dof]).max(0.0) / bins as f64;
                for bin in 0..bins {
                    marginal[dof][bin] += spread;
                }
            }
            return normalize_rows(marginal);
        }

        // Continuous contribution: the mean Gaussian target over continuous rows only.
        let mut total = Tensor::zeros([BAR_DOF as i64, NUM_BAR_BINS], (Kind::Float, Device::Cpu));
        let mut counted = Tensor::zeros([BAR_DOF as i64], (Kind::Float, Device::Cpu));
        const CHUNK: usize = 1 << 14;
        for chunk in rows.chunks(CHUNK) {
            let flat: Vec<f32> = chunk.iter().flat_map(|d| d.to_array()).collect();
            let values = Tensor::from_slice(&flat).view([chunk.len() as i64, BAR_DOF as i64]);
            let clamped = self.prepare(&values);
            let (index, _, is_atom) = self.locate(&clamped);
            let continuous = is_atom.neg() + 1.0;
            total += (self.smooth(&clamped, &index, BAR_LABEL_SIGMA_RATIO) * &continuous)
                .sum_dim_intlist([0i64].as_slice(), false, Kind::Float);
            counted += continuous
                .squeeze_dim(-1)
                .sum_dim_intlist([0i64].as_slice(), false, Kind::Float);
        }
        for dof in 0..BAR_DOF {
            let seen = counted.double_value(&[dof as i64]);
            if seen <= 0.0 {
                continue;
            }
            let weight = (1.0 - atom_mass[dof]).max(0.0) / seen;
            let row = total.get(dof as i64);
            for bin in 0..bins {
                marginal[dof][bin] += weight * row.double_value(&[bin as i64]);
            }
        }
        normalize_rows(marginal)
    }

    /// Accumulate `E[x | bin]` and `E[x^2 | bin]` over the whole fit sample.
    ///
    /// One pass, exact, no subsampling: unlike the smoothed marginal this needs no kernel
    /// evaluation, so the full sample is affordable and the outer bins — which hold under
    /// 1% of the mass and dominate both moments — need every row they can get.
    ///
    /// RAW values, binned by [`Self::bin_of`], which clamps. See [`BarBinMoments`]: that
    /// combination is what makes `sum_b p_b m_b` unbiased for `E[x]`, and it lets the two
    /// catch-alls take a representative outside their own bounds.
    fn measure_bin_moments(&self, samples: &[BarDof]) -> BarBinMoments {
        let bins = NUM_BAR_BINS as usize;
        let mut sum: [Vec<f64>; BAR_DOF] = std::array::from_fn(|_| vec![0.0; bins]);
        let mut sum_sq: [Vec<f64>; BAR_DOF] = std::array::from_fn(|_| vec![0.0; bins]);
        let mut count: [Vec<f64>; BAR_DOF] = std::array::from_fn(|_| vec![0.0; bins]);
        for sample in samples.iter().filter(|d| d.is_finite()) {
            let values = sample.to_array();
            for dof in 0..BAR_DOF {
                let x = values[dof] as f64;
                let bin = self.bin_of(dof, x);
                sum[dof][bin] += x;
                sum_sq[dof][bin] += x * x;
                count[dof][bin] += 1.0;
            }
        }
        // An unobserved bin falls back to its center, the only value the geometry alone
        // justifies. Equal-mass bins make this essentially unreachable on a real corpus —
        // 4M rows over 128 bins — and where it does happen the bin carries no mass, so it
        // cannot move a moment. It is NOT a silent stand-in for a measurement: an
        // all-fallback support is impossible, because `fit` refuses an empty sample.
        let mean: [Vec<f64>; BAR_DOF] = std::array::from_fn(|dof| {
            (0..bins)
                .map(|bin| {
                    let n = count[dof][bin];
                    if n > 0.0 {
                        sum[dof][bin] / n
                    } else {
                        self.centers[dof][bin]
                    }
                })
                .collect()
        });
        let second: [Vec<f64>; BAR_DOF] = std::array::from_fn(|dof| {
            (0..bins)
                .map(|bin| {
                    let n = count[dof][bin];
                    if n > 0.0 {
                        // Clamped below `mean^2` so a consumer can never read a negative
                        // within-bin variance out of rounding on a near-degenerate bin.
                        (sum_sq[dof][bin] / n).max(mean[dof][bin] * mean[dof][bin])
                    } else {
                        self.centers[dof][bin] * self.centers[dof][bin]
                    }
                })
                .collect()
        });
        BarBinMoments::new(mean, second, self.device)
    }

    fn from_bins(
        lo: [Vec<f64>; BAR_DOF],
        hi: [Vec<f64>; BAR_DOF],
        masses: [Vec<f64>; BAR_DOF],
        smoothed_marginal: [Vec<f64>; BAR_DOF],
        device: Device,
    ) -> Result<Self> {
        let bins = NUM_BAR_BINS as usize;
        // The tensor path evaluates the support in f32, so canonicalize every bound
        // to an f32-exact value and validate the geometry in that precision. Host
        // lookups, tensor lookups and the JSON artifact then agree bit for bit.
        let lo: [Vec<f64>; BAR_DOF] =
            lo.map(|row| row.into_iter().map(|x| x as f32 as f64).collect());
        let hi: [Vec<f64>; BAR_DOF] =
            hi.map(|row| row.into_iter().map(|x| x as f32 as f64).collect());

        for dof in 0..BAR_DOF {
            let (low, high) = (&lo[dof], &hi[dof]);
            let name = BAR_DOF_NAMES[dof];
            if low.len() != bins || high.len() != bins {
                bail!(
                    "DOF {name} support has {} lower and {} upper bounds, expected {bins} of each",
                    low.len(),
                    high.len()
                );
            }
            for j in 0..bins {
                if !low[j].is_finite() || !high[j].is_finite() || high[j] < low[j] {
                    bail!("DOF {name} bin {j} spans [{}, {}]", low[j], high[j]);
                }
                if j + 1 < bins && high[j] != low[j + 1] {
                    bail!(
                        "DOF {name} bins {j} and {} do not tile: {} != {}",
                        j + 1,
                        high[j],
                        low[j + 1]
                    );
                }
            }
            if !high.iter().zip(low.iter()).any(|(h, l)| h > l) {
                bail!("DOF {name} support has no continuous bin");
            }
            for (what, row) in [("bin masses", &masses[dof]), ("smoothed marginal", &smoothed_marginal[dof])] {
                if row.len() != bins {
                    bail!("DOF {name} {what} has {} entries, expected {bins}", row.len());
                }
                if let Some(bad) = row.iter().position(|p| !p.is_finite() || *p < 0.0) {
                    bail!("DOF {name} {what} entry {bad} is {}", row[bad]);
                }
                let sum: f64 = row.iter().sum();
                if (sum - 1.0).abs() > 1e-6 {
                    bail!("DOF {name} {what} sums to {sum}, expected 1");
                }
            }
        }

        let centers: [Vec<f64>; BAR_DOF] = std::array::from_fn(|dof| {
            let mut row: Vec<f64> = (0..bins)
                .map(|j| (0.5 * (lo[dof][j] + hi[dof][j])) as f32 as f64)
                .collect();
            // The outermost bins are catch-alls for everything past the clipped
            // support bounds, so they decode to the bound itself rather than to an
            // interior midpoint that would invent an extreme nobody observed. Atom
            // bins already satisfy this (lo == hi == center), so this is a no-op there.
            row[0] = lo[dof][0];
            row[bins - 1] = hi[dof][bins - 1];
            row
        });
        let widths: [Vec<f64>; BAR_DOF] = std::array::from_fn(|dof| {
            (0..bins)
                .map(|j| (hi[dof][j] - lo[dof][j]) as f32 as f64)
                .collect()
        });
        // Atom bins are recovered from the geometry and their mass is read straight
        // out of the histogram, so a reloaded support cannot disagree with its own
        // bounds and the atom table cannot disagree with the marginal.
        let atoms: [Vec<BarAtom>; BAR_DOF] = std::array::from_fn(|dof| {
            (0..bins)
                .filter(|&j| widths[dof][j] == 0.0)
                .map(|j| BarAtom {
                    value: lo[dof][j],
                    bin: j,
                    mass: masses[dof][j],
                })
                .collect()
        });
        for dof in 0..BAR_DOF {
            if atoms[dof].len() > MAX_BAR_ATOMS {
                bail!(
                    "DOF {} has {} atom bins, at most {MAX_BAR_ATOMS} are supported",
                    BAR_DOF_NAMES[dof],
                    atoms[dof].len()
                );
            }
        }

        let flat32 = |rows: &[Vec<f64>; BAR_DOF]| -> Vec<f32> {
            rows.iter()
                .flat_map(|row| row.iter().map(|&x| x as f32))
                .collect()
        };
        let dof_count = BAR_DOF as i64;
        let lo_flat = Tensor::from_slice(&flat32(&lo)).to_device(device);
        let hi_flat = Tensor::from_slice(&flat32(&hi)).to_device(device);
        let centers_flat = Tensor::from_slice(&flat32(&centers)).to_device(device);
        let widths_flat = Tensor::from_slice(&flat32(&widths)).to_device(device);
        let lo_t = lo_flat.view([dof_count, NUM_BAR_BINS]);
        let hi_t = hi_flat.view([dof_count, NUM_BAR_BINS]);
        let centers_t = centers_flat.view([dof_count, NUM_BAR_BINS]);

        let mut atom_values = vec![f32::NAN; BAR_DOF * MAX_BAR_ATOMS];
        let mut atom_bins = vec![0i64; BAR_DOF * MAX_BAR_ATOMS];
        for dof in 0..BAR_DOF {
            for (slot, atom) in atoms[dof].iter().enumerate() {
                atom_values[dof * MAX_BAR_ATOMS + slot] = atom.value as f32;
                atom_bins[dof * MAX_BAR_ATOMS + slot] = atom.bin as i64;
            }
        }
        let atom_value_t = Tensor::from_slice(&atom_values)
            .view([dof_count, MAX_BAR_ATOMS as i64])
            .to_device(device);
        let atom_bin_t = Tensor::from_slice(&atom_bins)
            .view([dof_count, MAX_BAR_ATOMS as i64])
            .to_device(device);

        let min_widths: Vec<f32> = (0..BAR_DOF)
            .map(|dof| {
                widths[dof]
                    .iter()
                    .copied()
                    .filter(|&w| w > 0.0)
                    .fold(f64::INFINITY, f64::min) as f32
            })
            .collect();
        let min_width_t = Tensor::from_slice(&min_widths)
            .view([dof_count, 1])
            .to_device(device);
        // The MEDIAN continuous width, taken on the lower of the two central order
        // statistics so an even count needs no interpolation between two widths that may
        // differ by orders of magnitude. `+inf` when a DOF is all atoms, which is inert
        // because every such observation takes the exact one-hot branch.
        let cap_widths: Vec<f32> = (0..BAR_DOF)
            .map(|dof| {
                let mut continuous: Vec<f64> =
                    widths[dof].iter().copied().filter(|&w| w > 0.0).collect();
                if continuous.is_empty() {
                    return f32::INFINITY;
                }
                continuous.sort_unstable_by(f64::total_cmp);
                continuous[(continuous.len() - 1) / 2] as f32
            })
            .collect();
        let cap_width_t = Tensor::from_slice(&cap_widths)
            .view([dof_count, 1])
            .to_device(device);
        let bin_offsets =
            (Tensor::arange(dof_count, (Kind::Int64, device)) * NUM_BAR_BINS).contiguous();

        Ok(Self {
            lo,
            hi,
            centers,
            widths,
            atoms,
            masses,
            smoothed_marginal,
            device,
            lo_t,
            hi_t,
            centers_t,
            atom_value_t,
            atom_bin_t,
            lo_flat,
            hi_flat,
            centers_flat,
            widths_flat,
            bin_offsets,
            min_width_t,
            cap_width_t,
            provenance: None,
            // Fitted moments are attached by `fit` or by `load`, never invented here:
            // `from_bins` knows the geometry but has never seen an observation.
            bin_moments: None,
        })
    }

    /// Copy of these supports with every cached tensor resident on `device`.
    /// Build one per training device so the hot path performs no host transfers.
    pub fn to_device(&self, device: Device) -> Self {
        let mut moved = Self::from_bins(
            self.lo.clone(),
            self.hi.clone(),
            self.masses.clone(),
            self.smoothed_marginal.clone(),
            device,
        )
        .expect("existing supports stay well formed");
        moved.provenance = self.provenance.clone();
        moved.bin_moments = self
            .bin_moments
            .as_ref()
            .map(|moments| moments.to_device(device));
        moved
    }

    /// Stamp the corpus and fit these bins came from. Called once, right after
    /// [`Self::fit`], before the artifact is persisted.
    pub fn with_provenance(mut self, provenance: BarSupportsProvenance) -> Self {
        self.provenance = Some(provenance);
        self
    }

    /// Attach fitted per-bin conditional moments measured on `samples`.
    ///
    /// Called by [`Self::fit`] once the geometry exists, because binning an observation
    /// requires the bins. Separate from the geometry fit so a caller can never end up with
    /// moments measured against different bounds than the ones they are indexed by.
    ///
    /// PUBLIC so a LOADED support can be upgraded in place from v4 to v5 without refitting:
    /// this touches `bin_moments` and NOTHING else, so `lo`, `hi`, `centers`, `widths`,
    /// `atoms`, `masses` and `smoothed_marginal` are bit-preserved and `nll_bar` stays on the
    /// same scale. Refitting the geometry to obtain moments would invalidate every persisted
    /// report in the tree; measuring moments against the geometry already on disk invalidates
    /// nothing. The caller is responsible for handing over the SAME sample the geometry was
    /// fitted on — otherwise `masses` and `bin_means` describe different populations. Use
    /// [`Self::with_verified_bin_moments`] to have that checked rather than assumed.
    pub fn with_measured_bin_moments(mut self, samples: &[BarDof]) -> Self {
        let moments = self.measure_bin_moments(samples);
        self.bin_moments = Some(moments);
        self
    }

    /// [`Self::with_measured_bin_moments`] with the SAMPLE IDENTIFIED against this support's
    /// own persisted histogram, rather than taken on trust.
    ///
    /// Recomputes the empirical bin masses of `samples` through exactly [`Self::bin_of`] and
    /// refuses the upgrade unless every entry reproduces the PERSISTED `masses` row to within
    /// `tolerance`. That single check covers both ways a v4 -> v5 upgrade can go quietly wrong:
    /// a different SAMPLE (wrong budget, wrong seed, a corpus that grew, the wrong split
    /// bound) and a different BINNING RULE (a `bin_of`, atom-detection or clip change since the
    /// artifact was written). Either produces moments indexed by bins whose mass they do not
    /// match, and neither is visible in the result — `bin_means` would simply be a measurement
    /// of some other population, correctly computed.
    ///
    /// This is deliberately stronger than comparing source against the commit the artifact was
    /// written at: that proves the code matched, this proves the DATA did. Returns the per-DOF
    /// worst absolute mass deviation so the caller can report how much slack was actually used.
    pub fn with_verified_bin_moments(
        self,
        samples: &[BarDof],
        tolerance: f64,
    ) -> Result<(Self, [f64; BAR_DOF])> {
        ensure!(
            tolerance >= 0.0 && tolerance.is_finite(),
            "the mass-agreement tolerance must be a finite non-negative probability, got \
             {tolerance}"
        );
        ensure!(
            !samples.is_empty(),
            "measuring per-bin moments needs at least one sample"
        );
        let recomputed = self.measure_bin_masses(samples);
        let mut worst = [0.0f64; BAR_DOF];
        for dof in 0..BAR_DOF {
            let (mut bin_of_worst, mut deviation) = (0usize, 0.0f64);
            for bin in 0..NUM_BAR_BINS as usize {
                let gap = (recomputed[dof][bin] - self.masses[dof][bin]).abs();
                if gap > deviation {
                    deviation = gap;
                    bin_of_worst = bin;
                }
            }
            worst[dof] = deviation;
            ensure!(
                deviation <= tolerance,
                "DOF {}: re-measuring the bin masses of the {} supplied samples reproduces the \
                 persisted histogram only to {deviation:.3e} (bin {bin_of_worst}: measured \
                 {:.9} against persisted {:.9}), past the tolerance of {tolerance:.3e}. The \
                 sample or the binning rule is not the one this support was fitted with, so \
                 per-bin moments measured on it would describe a different population than the \
                 masses they sit beside; refusing the upgrade",
                BAR_DOF_NAMES[dof],
                samples.len(),
                recomputed[dof][bin_of_worst],
                self.masses[dof][bin_of_worst],
            );
        }
        Ok((self.with_measured_bin_moments(samples), worst))
    }

    /// `E[x | bin]` per DOF, or `None` on a pre-v5 artifact.
    ///
    /// The correct decode for any FIRST MOMENT: `expectation`, the Mincer-Zarnowitz mean
    /// slope, and the Kelly mean. NOT [`Self::centers`], which decodes the two catch-alls
    /// to their outer bounds and overstates them by about 3.1x.
    pub fn bin_means(&self, dof: usize) -> Option<&[f64]> {
        self.bin_moments
            .as_ref()
            .map(|moments| moments.mean[dof].as_slice())
    }

    /// `E[x^2 | bin]` per DOF, or `None` on a pre-v5 artifact.
    ///
    /// Needed ALONGSIDE [`Self::bin_means`] by any second-moment consumer: a single
    /// representative per bin cannot carry within-bin dispersion, which is 12.0% of the
    /// true second moment of `r` and 98.6% of that sits in the two catch-alls. Using only
    /// the means understates the marginal variance by 12%; using the bounds overstates it
    /// by 5.25x. Read [`BarBinMoments`] on why the outer entries are not converged.
    pub fn bin_second_moments(&self, dof: usize) -> Option<&[f64]> {
        self.bin_moments
            .as_ref()
            .map(|moments| moments.second[dof].as_slice())
    }

    /// Whether this support carries FITTED per-bin moments.
    ///
    /// The gate exists so a pre-v5 artifact cannot present bin midpoints or bounds as
    /// measured conditional means. A consumer that needs moments refuses to start rather
    /// than substituting a geometric stand-in for a measurement.
    pub fn bin_means_measured(&self) -> bool {
        self.bin_moments.is_some()
    }

    /// `[BAR_DOF, NUM_BAR_BINS]` fitted means and second moments on this support's device.
    pub fn bin_moment_tensors(&self) -> Option<(&Tensor, &Tensor)> {
        self.bin_moments
            .as_ref()
            .map(|moments| (&moments.mean_t, &moments.second_t))
    }

    /// The per-bin representative `d_b` this DOF decodes to under `convention`.
    ///
    /// The ONE accessor every first-moment consumer should route through, so the convention in
    /// force is an argument at the call site rather than a property of whichever artifact
    /// happened to load. [`MeanDecode::Fitted`] is an ERROR on a support without measured
    /// moments; it never degrades to [`MeanDecode::Edge`].
    pub fn mean_decode(&self, dof: usize, convention: MeanDecode) -> Result<&[f64]> {
        match convention {
            MeanDecode::Edge => Ok(self.centers[dof].as_slice()),
            MeanDecode::Fitted => self.bin_means(dof).ok_or_else(|| {
                anyhow!(
                    "DOF {} was asked for the FITTED mean decode but this support carries no \
                     measured per-bin moments (pre-v{BAR_SUPPORTS_MOMENTS_VERSION} artifact); \
                     refit the moments or ask for {} explicitly",
                    BAR_DOF_NAMES[dof],
                    MeanDecode::Edge
                )
            }),
        }
    }

    /// `[BAR_DOF, NUM_BAR_BINS]` device twin of [`Self::mean_decode`], for the tensor path.
    ///
    /// Same refusal on a pre-v5 artifact, for the same reason: the batched decode is where a
    /// silent substitution would be least visible.
    pub fn mean_decode_tensor(&self, convention: MeanDecode) -> Result<&Tensor> {
        match convention {
            MeanDecode::Edge => Ok(&self.centers_t),
            MeanDecode::Fitted => self.bin_moment_tensors().map(|(mean, _)| mean).ok_or_else(|| {
                anyhow!(
                    "the FITTED mean decode was requested but this support carries no measured \
                     per-bin moments (pre-v{BAR_SUPPORTS_MOMENTS_VERSION} artifact)"
                )
            }),
        }
    }

    /// Largest `|E[x]|` any distribution over this DOF's bins can possibly produce under
    /// `convention`.
    ///
    /// `E[x] = sum_b p_b d_b` is a convex combination of the per-bin decode values, so it
    /// is bounded by `max_b |d_b|` — exactly, with no distributional assumption. A predicted
    /// mean above this is not a confident forecast, it is arithmetically impossible, and the
    /// only ways to produce one are a bug or a decode the caller did not intend.
    ///
    /// NAMES ITS CONVENTION AND REFUSES WHEN IT CANNOT HONOUR IT. This used to read
    /// `bin_means(dof).unwrap_or_else(|| centers)`, which returned the 883.32 bps EDGE ceiling
    /// on any support without measured moments while its own documentation said it would
    /// return the fitted one — so after the fitted decode was believed to have landed, the
    /// pre-fix number kept coming back with nothing anywhere saying so. An absent measurement
    /// must not read as a real one, so the fallback is gone: the caller states the convention
    /// and an unavailable convention is an error.
    pub fn representable_mean_ceiling(&self, dof: usize, convention: MeanDecode) -> Result<f64> {
        Ok(self
            .mean_decode(dof, convention)?
            .iter()
            .fold(0.0f64, |worst, x| worst.max(x.abs())))
    }

    /// The same ceiling with the two catch-all bins excluded.
    ///
    /// A predicted mean above this REQUIRES catch-all mass: it cannot be produced by any
    /// distribution supported on the interior bins. On the live `r` support the interior
    /// ceiling is 136.48 bps against an all-bin EDGE ceiling of 883.32, so the gap between the
    /// two is entirely the two catch-alls, and comparing a predicted mean against both
    /// separates "confident about an interior move" from "leaning on the tails".
    ///
    /// Note the asymmetry between the two ceilings, which is why the convention has to be
    /// stated here too even though the interior bins are midpoints under both: the FITTED
    /// decode of an interior bin is its conditional mean, not its midpoint, so the interior
    /// ceiling moves slightly with the convention while the all-bin ceiling moves by a factor
    /// of three.
    ///
    /// NOT the analogous statement for a predicted SD. The hard interior ceiling on a
    /// standard deviation is the same 136.48 bps — put the mass on the two extreme interior
    /// bins — and the marginal-weighted interior RMS of 30.68 bps that has been quoted
    /// beside it is a typical value, not a bound. A predicted sd above 30.68 is unremarkable;
    /// a predicted mean above 136.48 is not.
    pub fn interior_mean_ceiling(&self, dof: usize, convention: MeanDecode) -> Result<f64> {
        let decode = self.mean_decode(dof, convention)?;
        let last = decode.len().saturating_sub(1);
        Ok(decode
            .iter()
            .enumerate()
            .filter(|(bin, _)| *bin != 0 && *bin != last)
            .fold(0.0f64, |worst, (_, x)| worst.max(x.abs())))
    }

    /// Smallest variance any distribution over this DOF's bins can have while producing the
    /// predicted mean `mean`. Zero outside the decode range, where no mixing is needed.
    ///
    /// TEST-ONLY, DELIBERATELY. This is exact and tight — minimising `E[x^2] - mean^2`
    /// subject to `E[x] = mean` over a finite support is attained by two-point mass on the
    /// ADJACENT decode values bracketing `mean`, giving `(mean - d_j) * (d_(j+1) - mean)` —
    /// but it is NOT exposed to production and must not become a rejection test on predicted
    /// moments. It exists so the representational gap it measures is recorded and checked
    /// rather than re-derived, because it was twice re-derived WRONGLY during the
    /// investigation that produced it.
    ///
    /// WHY THE FIXED-PAIR FORM IS WRONG, recorded so it is not re-derived a third time.
    /// Using the fixed pair `(interior_ceiling, outer_bound)` for every mean, and inverting
    /// it to bound the MEAN given a standard deviation, fails twice: `sd = sqrt(w (1-w)) *
    /// (b - a)` has TWO roots in `w`, so `sd = 76.72 bps` admits `w = 0.010752` with mean
    /// 144.48 bps AND `w = 0.989248` with mean 872.38 bps — and the second is the maximum.
    /// Reasoning about that relation as if it were monotone yields a bound that rejects
    /// precisely the bars it should permit: a bar with ALL mass in the outermost bin has
    /// `|mean| = 880.38 bps` and variance ZERO, the most extreme forecast the support can
    /// express by design. This function returns 0 there, as it must.
    ///
    /// On the live `r` support the decode jumps 136.48 -> 880.38 bps with nothing between,
    /// so a mean strictly inside that gap forces `sd >= 251 bps` at 234 bps and `>= 372 bps`
    /// at the midpoint, while BOTH ENDPOINTS ARE FREE. That 6.4x representational gap is a
    /// property of the equal-mass binning, not of the decode convention, and it survives the
    /// move to fitted conditional means — narrower, at 131.98 -> 283.62 bps, but still there.
    /// Its consequence is that the model can predict an 8.8% expected five-minute move with
    /// ZERO predicted uncertainty, on which `E[ln(1 + f R)]` is monotone with no interior
    /// maximum — so nothing in the DISTRIBUTION bounds `f*`.
    ///
    /// What bounds it is `trade_bench::MAX_LEVERAGE`, applied to the solver bracket as
    /// `cap.min(MAX_LEVERAGE)`: the bracket is the TIGHTER of the declared ceiling and the
    /// support's own ruin point, so such a bar solves to the ceiling instead of diverging.
    /// The support's ruin point is `1 / max |R|` over its negative decodes — `11.83x` today
    /// off `centers_t[0]`, about `36x` once the catch-alls decode to their fitted conditional
    /// means. Because the bracket takes the min, that refit loosens the OPERATIVE bound only
    /// from `11.83x` to the ceiling, NOT to `36x`. The invariant that matters is therefore
    /// that the ceiling stay strictly BELOW the post-refit ruin point. `12.0` sits DELIBERATELY
    /// just above today's `11.83x` so that landing it moved no measured quantity (see the
    /// rationale on `trade_bench::MAX_LEVERAGE` itself): today's operative ruin bound is still
    /// the support edge, and the ceiling starts binding only once the decode moves.
    ///
    /// TAKES ITS CONVENTION for the same reason the two ceilings now do: this used to read
    /// `bin_means(dof).unwrap_or_else(|| centers)`, so on a support without measured moments it
    /// silently answered off the EDGE decode while its documentation discussed fitted means.
    #[cfg(test)]
    fn min_variance_for_mean(&self, dof: usize, mean: f64, convention: MeanDecode) -> Result<f64> {
        let decode = self.mean_decode(dof, convention)?;
        // The decode is non-decreasing for `centers` by construction and for fitted means
        // because each is confined to its own ascending bin — except the two catch-alls,
        // whose means may sit outside their bounds. Scanning for the tightest bracketing
        // pair rather than assuming order keeps the bound valid either way.
        let (mut below, mut above) = (f64::NEG_INFINITY, f64::INFINITY);
        for &d in decode {
            if d <= mean && d > below {
                below = d;
            }
            if d >= mean && d < above {
                above = d;
            }
        }
        if !below.is_finite() || !above.is_finite() {
            // `mean` lies outside the decode range entirely, so no distribution over these
            // bins attains it and there is nothing to bound. Reporting zero refuses to
            // manufacture a constraint from a mean this support cannot produce at all;
            // `representable_mean_ceiling` is the check for that case.
            return Ok(0.0);
        }
        Ok(((mean - below) * (above - mean)).max(0.0))
    }

    /// Recorded provenance, or `None` for a legacy artifact. An absent provenance is
    /// not "unchanged": it means the file cannot be checked at all.
    pub fn provenance(&self) -> Option<&BarSupportsProvenance> {
        self.provenance.as_ref()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn num_bins(&self) -> i64 {
        NUM_BAR_BINS
    }

    /// Bin lower bounds; `lo(dof)[j + 1] == hi(dof)[j]`.
    pub fn lower_bounds(&self, dof: usize) -> &[f64] {
        &self.lo[dof]
    }

    pub fn upper_bounds(&self, dof: usize) -> &[f64] {
        &self.hi[dof]
    }

    pub fn centers(&self, dof: usize) -> &[f64] {
        &self.centers[dof]
    }

    /// Per-bin widths; exactly zero for atom bins.
    pub fn widths(&self, dof: usize) -> &[f64] {
        &self.widths[dof]
    }

    /// The label-smoothing kernel width actually used for an observation landing in
    /// `bin`, in RETURN units: `BAR_LABEL_SIGMA_RATIO * clamp(width, min, median)`.
    ///
    /// The `f64` twin of the tensor path in [`Self::smooth`], and the thing to print when
    /// a soft-target run's calibration is in question — an equal-mass binning makes the
    /// uncapped `0.75 * width` vary by three orders of magnitude across one DOF.
    /// `+inf` only for a DOF with no continuous bin, whose observations are all atoms.
    pub fn smooth_sigma(&self, dof: usize, bin: usize) -> f64 {
        let continuous: Vec<f64> = self.widths[dof]
            .iter()
            .copied()
            .filter(|&w| w > 0.0)
            .collect();
        if continuous.is_empty() {
            return f64::INFINITY;
        }
        let min = continuous.iter().copied().fold(f64::INFINITY, f64::min);
        let mut sorted = continuous;
        sorted.sort_unstable_by(f64::total_cmp);
        let cap = sorted[(sorted.len() - 1) / 2];
        BAR_LABEL_SIGMA_RATIO * self.widths[dof][bin].max(min).min(cap)
    }

    /// The per-DOF ceiling on [`Self::smooth_sigma`], i.e. the widest kernel any
    /// observation of `dof` can receive. `+inf` for an all-atom DOF.
    pub fn smooth_sigma_cap(&self, dof: usize) -> f64 {
        let mut sorted: Vec<f64> = self.widths[dof]
            .iter()
            .copied()
            .filter(|&w| w > 0.0)
            .collect();
        if sorted.is_empty() {
            return f64::INFINITY;
        }
        sorted.sort_unstable_by(f64::total_cmp);
        BAR_LABEL_SIGMA_RATIO * sorted[(sorted.len() - 1) / 2]
    }

    /// Atom bins in ascending value order, with the empirical mass observed at fit
    /// time. Report these alongside `nll_bar`: a large atom mass on `u`/`v` means a
    /// large share of the corpus is illiquid filler.
    pub fn atoms(&self, dof: usize) -> &[BarAtom] {
        &self.atoms[dof]
    }

    /// Empirical probability of each bin, measured on the fit sample with exactly
    /// the rule [`Self::bin_of`] applies.
    pub fn bin_masses(&self, dof: usize) -> &[f64] {
        &self.masses[dof]
    }

    /// The mean smoothed target `q* = E_x[t(x)]` over the fit sample. This is the
    /// prediction an optimal MARGINAL head converges to, because the soft-target
    /// cross entropy is minimized over fixed predictions exactly at the mean target.
    pub fn smoothed_marginal(&self, dof: usize) -> &[f64] {
        &self.smoothed_marginal[dof]
    }

    /// The per-DOF row an optimal MARGINAL head converges to under `scoring`.
    ///
    /// Under [`BarScoring::Smoothed`] that is the mean smoothed target `q*`, because the
    /// soft-target cross entropy over fixed predictions is minimized exactly at the mean
    /// target. Under [`BarScoring::Hard`] and [`BarScoring::Density`] the target is a
    /// one-hot on the containing bin, so the optimum is the bin HISTOGRAM: the two rules
    /// differ only by the additive measure term, which no prediction can move.
    pub fn reference_row(&self, dof: usize, scoring: BarScoring) -> &[f64] {
        if scoring.is_smoothed() {
            &self.smoothed_marginal[dof]
        } else {
            &self.masses[dof]
        }
    }

    /// Per-DOF `E_x[ln width(bin(x))]` over the fit sample: the additive term that turns a
    /// categorical log-loss into a log-DENSITY loss.
    ///
    /// Atoms carry a probability MASS rather than a density, so their zero-width bins
    /// contribute exactly nothing. It is strongly NEGATIVE — a `128`-bin equal-mass tiling
    /// of five-minute log returns has bins of order `1e-4` wide — which is why a `density`
    /// figure sits tens of nats below a `hard` one on the identical model, and why the two
    /// must never be compared.
    pub fn log_measure_dof(&self) -> [f64; BAR_DOF] {
        std::array::from_fn(|dof| {
            (0..NUM_BAR_BINS as usize)
                .filter(|&bin| self.widths[dof][bin] > 0.0)
                .map(|bin| self.masses[dof][bin] * self.widths[dof][bin].ln())
                .sum()
        })
    }

    /// Nats per bar of [`Self::log_measure_dof`].
    pub fn log_measure_bar(&self) -> f64 {
        self.log_measure_dof().iter().sum()
    }

    /// The measure term added to a reference line under `scoring`: zero for the two
    /// discrete rules, [`Self::log_measure_dof`] for the density rule.
    fn measure_term(&self, scoring: BarScoring) -> [f64; BAR_DOF] {
        if scoring.is_density() {
            self.log_measure_dof()
        } else {
            [0.0; BAR_DOF]
        }
    }

    /// Per-DOF nats an optimal marginal head achieves under `scoring`.
    ///
    /// This is the loss under the SAME objective the model is trained and reported on, so
    /// it is directly comparable to a per-DOF `nll` term: for a fixed prediction `q` the
    /// loss is the cross entropy `H(q*, q)` plus a prediction-independent measure term,
    /// which is minimized at `q = q*`.
    ///
    /// Under [`BarScoring::Smoothed`] this is close to but NOT ordered against the entropy
    /// of the hard bin histogram — that is the optimum of a different objective, and
    /// smoothing both widens interior labels and piles truncated mass onto the edge bins,
    /// so either can come out on top. Under the other two modes it IS that entropy, plus
    /// the measure term for the density rule.
    pub fn marginal_nll_dof(&self, scoring: BarScoring) -> [f64; BAR_DOF] {
        let measure = self.measure_term(scoring);
        std::array::from_fn(|dof| {
            let entropy: f64 = self
                .reference_row(dof, scoring)
                .iter()
                .filter(|p| **p > 0.0)
                .map(|p| -p * p.ln())
                .sum();
            entropy + measure[dof]
        })
    }

    /// Nats per bar an optimal marginal head achieves, i.e. the sum over the five
    /// chain factors of [`Self::marginal_nll_dof`].
    ///
    /// This is the reference line that matters for model selection: beating
    /// [`Self::uniform_nll_bar`] only proves the unconditional marginals were
    /// learned, which is trivial. Beating this is the first evidence of conditional
    /// structure. Derived per corpus, so it moves with the symbol set and session mix.
    pub fn marginal_nll_bar(&self, scoring: BarScoring) -> f64 {
        self.marginal_nll_dof(scoring).iter().sum()
    }

    /// Bin holding the flat-bar value of `u`/`v`, i.e. the `0.5` atom that `encode_dof`
    /// emits whenever `high == low`. `None` for a DOF that has no such atom.
    fn flat_shape_bin(&self, dof: usize) -> Option<usize> {
        if dof != DOF_U && dof != DOF_V {
            return None;
        }
        self.atoms[dof]
            .iter()
            .find(|atom| atom.value == 0.5)
            .map(|atom| atom.bin)
    }

    /// Per-DOF marginal reference with the ENCODING TAUTOLOGY removed.
    ///
    /// `encode_dof` sets `u = v = 0.5` whenever the bar is flat, i.e. whenever `s == 0`,
    /// so `s == 0` implies the `u` and `v` outcomes exactly, and the chain puts `s` ahead of
    /// both. A model that has learned nothing but that one bit therefore collects
    /// `I(u; 1{s=0}) + I(v; 1{s=0})` nats for free and the unconditional
    /// marginal credits it as skill. Here the `u` and `v` references are the entropies
    /// CONDITIONED on a non-flat bar: the `0.5` atom is dropped and the rest renormalized.
    /// `r`, `s` and `w` are unchanged, because nothing in the encoding determines them.
    ///
    /// The measure term is conditioned the same way, so the density rule's conditional
    /// reference stays the log-likelihood of the same conditioned law.
    ///
    /// This is the honest yardstick for [`crate::torch::train::pretrain`]'s
    /// `nll_bar_conditional`, which scores `u` and `v` only on bars with `s != 0`.
    pub fn marginal_nll_dof_conditional(&self, scoring: BarScoring) -> [f64; BAR_DOF] {
        let density = scoring.is_density();
        std::array::from_fn(|dof| {
            let row = self.reference_row(dof, scoring);
            let measure = |live: f64, skip: Option<usize>| -> f64 {
                if !density {
                    return 0.0;
                }
                (0..NUM_BAR_BINS as usize)
                    .filter(|bin| Some(*bin) != skip && self.widths[dof][*bin] > 0.0)
                    .map(|bin| row[bin] / live * self.widths[dof][bin].ln())
                    .sum()
            };
            let Some(flat_bin) = self.flat_shape_bin(dof) else {
                let entropy: f64 = row.iter().filter(|p| **p > 0.0).map(|p| -p * p.ln()).sum();
                return entropy + measure(1.0, None);
            };
            let live = 1.0 - row[flat_bin];
            if live <= 0.0 {
                return 0.0;
            }
            let entropy: f64 = row
                .iter()
                .enumerate()
                .filter(|(bin, p)| *bin != flat_bin && **p > 0.0)
                .map(|(_, p)| {
                    let q = p / live;
                    -q * q.ln()
                })
                .sum();
            entropy + measure(live, Some(flat_bin))
        })
    }

    /// Nats per bar of the conditional reference, i.e. the sum of
    /// [`Self::marginal_nll_dof_conditional`].
    pub fn marginal_nll_bar_conditional(&self, scoring: BarScoring) -> f64 {
        self.marginal_nll_dof_conditional(scoring).iter().sum()
    }

    /// Nats per bar a model gains from the `s == 0 => u = v = 0.5` identity alone.
    ///
    /// `I(u; 1{s=0}) = H(u) - (1 - m) * H(u | s != 0)`, where `m` is the flat-bar mass, and
    /// likewise for `v`; the `s == 0` branch contributes zero because the outcome is
    /// deterministic there. On the live 300s supports this is ~0.690 nats, which is ~19% of
    /// the gain over the calibrated marginal a trained model currently reports. It is
    /// arithmetic, not market structure, so it belongs on the baseline chart as its own line
    /// rather than inside the headline number.
    ///
    /// The flat bin is an ATOM, so it carries no width and the density rule's measure term
    /// cancels exactly between the two references: this is a property of the encoding, not
    /// of the scoring rule, and `hard` and `density` report it identically.
    pub fn encoding_identity_nats(&self, scoring: BarScoring) -> f64 {
        let unconditional = self.marginal_nll_dof(scoring);
        let conditional = self.marginal_nll_dof_conditional(scoring);
        [DOF_U, DOF_V]
            .into_iter()
            .filter_map(|dof| {
                let flat_bin = self.flat_shape_bin(dof)?;
                let live = 1.0 - self.reference_row(dof, scoring)[flat_bin];
                Some(unconditional[dof] - live * conditional[dof])
            })
            .sum()
    }

    /// Nats per bar of the "marginal plus the free encoding identity" baseline: what a head
    /// that learned the unconditional bin masses AND the flat-bar identity — and nothing
    /// else — scores. Any claimed conditional structure has to clear THIS, not
    /// [`Self::marginal_nll_bar`].
    pub fn marginal_plus_identity_nll_bar(&self, scoring: BarScoring) -> f64 {
        self.marginal_nll_bar(scoring) - self.encoding_identity_nats(scoring)
    }

    /// `BAR_DOF * ln(NUM_BAR_BINS)`: the CATEGORICAL loss of a uniform head, i.e. the
    /// value a zero-initialized [`BarEmissionHead`] starts at under the two discrete
    /// rules. The mode-aware line is [`Self::uniform_nll_bar`].
    pub fn uniform_categorical_nll_bar() -> f64 {
        BAR_DOF as f64 * (NUM_BAR_BINS as f64).ln()
    }

    /// Per-DOF nats a UNIFORM-over-bins head pays under `scoring`, which is exactly where
    /// the zero-initialized emission head starts.
    ///
    /// `ln(NUM_BAR_BINS)` for both discrete rules — the smoothed target still sums to one,
    /// so its cross entropy against a uniform prediction is the same — plus the measure
    /// term for the density rule.
    pub fn uniform_nll_dof(&self, scoring: BarScoring) -> [f64; BAR_DOF] {
        let measure = self.measure_term(scoring);
        std::array::from_fn(|dof| (NUM_BAR_BINS as f64).ln() + measure[dof])
    }

    /// Nats per bar of [`Self::uniform_nll_dof`].
    pub fn uniform_nll_bar(&self, scoring: BarScoring) -> f64 {
        self.uniform_nll_dof(scoring).iter().sum()
    }

    /// Share of the fit sample sitting exactly on an atom of this DOF.
    ///
    /// Report it: `u`/`v` carry ~47% atom mass on a real equity corpus, so a corpus
    /// whose illiquid-filler share drifts moves `nll_bar` without anything having
    /// been learned or forgotten.
    pub fn atom_mass(&self, dof: usize) -> f64 {
        self.atoms[dof].iter().map(|a| a.mass).sum()
    }

    /// The marginal reference of [`Self::marginal_nll_dof`], split the same way
    /// [`bar_nll_decomposition`] splits the model's loss, so the two can be charted
    /// against each other. `class + shape == marginal_nll_dof(scoring)` per DOF, exactly.
    ///
    /// The class half is the entropy of the degeneracy indicator — which atom the
    /// bar sits on, or "somewhere continuous" — and on a real corpus it is worth
    /// ~3.0 of the 21.7 marginal nats, concentrated in `u`/`v`. Predicting
    /// degeneracy is largely a logical consequence of the chain prefix (`s == 0`
    /// forces `u == v == 0.5`), so a head that learned only that would post a gain
    /// the undivided number cannot distinguish from intra-bar shape.
    ///
    /// The density rule's measure term belongs entirely to `shape`: choosing WHICH atom or
    /// whether the bar is continuous at all is a discrete decision that carries no width,
    /// and only the placement inside the continuous part is a density.
    pub fn marginal_nll_parts(&self, scoring: BarScoring) -> BarNllSplit {
        let mut split = BarNllSplit::default();
        let density = scoring.is_density();
        for dof in 0..BAR_DOF {
            let q = self.reference_row(dof, scoring);
            let mut continuous_mass = 0.0;
            for bin in 0..NUM_BAR_BINS as usize {
                if self.widths[dof][bin] == 0.0 {
                    if q[bin] > 0.0 {
                        split.class[dof] -= q[bin] * q[bin].ln();
                    }
                } else {
                    continuous_mass += q[bin];
                }
            }
            if continuous_mass > 0.0 {
                split.class[dof] -= continuous_mass * continuous_mass.ln();
                for bin in 0..NUM_BAR_BINS as usize {
                    if self.widths[dof][bin] > 0.0 && q[bin] > 0.0 {
                        split.shape[dof] -= q[bin] * (q[bin] / continuous_mass).ln();
                    }
                }
            }
            if density {
                for bin in 0..NUM_BAR_BINS as usize {
                    if self.widths[dof][bin] > 0.0 {
                        split.shape[dof] += self.masses[dof][bin] * self.widths[dof][bin].ln();
                    }
                }
            }
        }
        split
    }

    /// Per-DOF nats every prediction pays under `scoring` no matter how good it is.
    ///
    /// Zero for [`BarScoring::Hard`] and [`BarScoring::Density`]: both score the bin the
    /// observation actually landed in, and an oracle that knew it exactly would pay nothing.
    /// That is the whole reason `density` is the default — the number below is what
    /// smoothing costs.
    ///
    /// For [`BarScoring::Smoothed`] this is `E_x[H(t(x))]`. Gaussian label smoothing makes
    /// the soft-target cross entropy a proper scoring rule for the SMOOTHED law `T(P)`, not
    /// for `P`. The identified optimum is the smoothed conditional, so a head that knew the
    /// next bar exactly would still pay the entropy of its own target on every continuous
    /// row. It is asymmetric: the marginal reference of [`Self::marginal_nll_bar`] pays
    /// essentially none of it, because averaging targets over the corpus washes the kernel
    /// out. Draw it under `nll_bar` — the reachable range is
    /// `marginal_nll_bar(mode) - scoring_floor_bar(mode)`.
    ///
    /// Atom rows get an exact one-hot and contribute zero, so the floor is the
    /// continuous mass times the mean continuous-row target entropy, and the per-DOF
    /// floors differ by 2x — which is why the per-DOF `nll` series are not comparable
    /// with one another even though they share an axis.
    ///
    /// Derived from the fitted support alone, never hardcoded: within a bin the
    /// equal-mass construction makes the observation uniform, so the expectation is
    /// the bin masses against a composite-midpoint integral of `H(t(x))` across each
    /// continuous bin. The integrand is smooth away from the bin edges, so the
    /// [`SMOOTHING_FLOOR_NODES`]-node rule is converged to well under a milli-nat.
    pub fn scoring_floor(&self, scoring: BarScoring) -> [f64; BAR_DOF] {
        if !scoring.is_smoothed() {
            return [0.0; BAR_DOF];
        }
        let bins = NUM_BAR_BINS as usize;
        let nodes = SMOOTHING_FLOOR_NODES;
        let rows = bins * nodes;
        let mut values = vec![0f32; rows * BAR_DOF];
        let mut weights = vec![0f64; rows * BAR_DOF];
        for dof in 0..BAR_DOF {
            for bin in 0..bins {
                let (lo, width) = (self.lo[dof][bin], self.widths[dof][bin]);
                if width <= 0.0 {
                    continue;
                }
                let share = self.masses[dof][bin] / nodes as f64;
                for node in 0..nodes {
                    let at = (bin * nodes + node) * BAR_DOF + dof;
                    values[at] = (lo + width * (node as f64 + 0.5) / nodes as f64) as f32;
                    weights[at] = share;
                }
            }
        }

        let probe = Tensor::from_slice(&values).view([rows as i64, BAR_DOF as i64]);
        let clamped = self.prepare(&probe);
        let (index, _, _) = self.locate(&clamped);
        let target = self.smooth(&clamped, &index, BAR_LABEL_SIGMA_RATIO);
        let log_target = target.clamp_min(1e-30).log();
        let entropy =
            -(&target * &log_target).sum_dim_intlist([-1].as_slice(), false, Kind::Double);
        let weight = Tensor::from_slice(&weights)
            .view([rows as i64, BAR_DOF as i64])
            .to_device(entropy.device());
        let weighted = entropy * weight;
        let floor = weighted.sum_dim_intlist([0i64].as_slice(), false, Kind::Double);
        std::array::from_fn(|dof| floor.double_value(&[dof as i64]))
    }

    /// Nats per bar of the unreachable floor, i.e. the sum over the five chain
    /// factors of [`Self::scoring_floor`].
    pub fn scoring_floor_bar(&self, scoring: BarScoring) -> f64 {
        self.scoring_floor(scoring).iter().sum()
    }

    /// Host-side bin lookup. Exact matches on an atom take the atom bin; every
    /// other value takes the last bin whose lower bound it reaches, after clamping
    /// onto the support. The value is narrowed to `f32` first, because the support
    /// is an `f32` object and the tensor twin compares in that precision.
    pub fn bin_of(&self, dof: usize, value: f64) -> usize {
        self.binner(dof).bin_of(value)
    }

    /// One DOF's binning rule, detached from the tensors.
    ///
    /// [`BarSupports`] owns [`Tensor`] fields and is therefore neither `Send` nor `Sync`, so a
    /// rayon fold that has to place bars in bins cannot borrow it. The rule itself needs nothing
    /// but three slices of `f64`, and this hands them over by reference with the SAME lookup
    /// [`Self::bin_of`] performs — because [`Self::bin_of`] is now defined as this.
    pub fn binner(&self, dof: usize) -> DofBinner<'_> {
        DofBinner {
            lo: &self.lo[dof],
            hi: &self.hi[dof],
            atoms: &self.atoms[dof],
        }
    }

    /// Persist the fitted bin bounds as JSON next to the checkpoint. Decimal
    /// round-tripping is faithful to within an ulp of `f64`, and `from_bins`
    /// re-narrows to `f32`, so a reloaded support is bit-identical where evaluated.
    ///
    /// REFUSES, before touching the filesystem, a support carrying no fitted moments.
    /// [`BAR_SUPPORTS_FORMAT_VERSION`] is the only schema this build writes and its invariant
    /// is that the moments are present, so such a support has no valid representation on disk.
    /// Stamping v4 instead is NOT the alternative: the file would then be indistinguishable
    /// from an honestly fitted pre-moments artifact and the caller's belief that it holds a
    /// measurement would go unrecorded. Reaching this means the caller loaded a pre-v5 support
    /// and is trying to hand it on as a checkpoint's own geometry, which
    /// [`crate::torch::train::pretrain`] refuses at startup so it cannot surface here.
    pub fn save(&self, path: &Path) -> Result<()> {
        let Some(moments) = self.bin_moments.as_ref() else {
            bail!(
                "refusing to write bar supports {}: these in-memory supports carry no fitted \
                 per-bin moments, so no valid version {BAR_SUPPORTS_FORMAT_VERSION} artifact can \
                 be written from them. They were loaded from a pre-v{BAR_SUPPORTS_MOMENTS_VERSION} \
                 file; measure moments onto that exact geometry with the `bar-supports-moments` \
                 subcommand and point this run at the result",
                path.display()
            );
        };
        let json = BarSupportsJson::v5(self, moments);
        let body = serde_json::to_vec_pretty(&json).context("serializing bar supports")?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
        }
        std::fs::write(path, body).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let body = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let json: BarSupportsJson =
            serde_json::from_slice(&body).with_context(|| format!("parsing {}", path.display()))?;
        if !BAR_SUPPORTS_READABLE_VERSIONS.contains(&json.format_version) {
            bail!(
                "bar supports {} has format version {}, this build reads {:?}",
                path.display(),
                json.format_version,
                BAR_SUPPORTS_READABLE_VERSIONS
            );
        }
        // A file at or above the moments version MUST carry them. Without this a v5 written
        // by a build that failed to populate them would load with `bin_means_measured()`
        // false and be indistinguishable from an honest v4 — the version would claim a
        // measurement the file does not contain.
        if json.format_version >= BAR_SUPPORTS_MOMENTS_VERSION
            && (json.bin_means.is_none() || json.bin_second_moments.is_none())
        {
            bail!(
                "bar supports {} declares version {} but is missing fitted per-bin moments",
                path.display(),
                json.format_version
            );
        }
        if json.num_bins != NUM_BAR_BINS {
            bail!(
                "bar supports {} has {} bins, this build uses {}",
                path.display(),
                json.num_bins,
                NUM_BAR_BINS
            );
        }
        for (what, len) in [
            ("lo", json.lo.len()),
            ("hi", json.hi.len()),
            ("masses", json.masses.len()),
            ("smoothed_marginal", json.smoothed_marginal.len()),
        ] {
            if len != BAR_DOF {
                bail!(
                    "bar supports {} has {len} {what} rows, expected {BAR_DOF}",
                    path.display()
                );
            }
        }
        let mut lo = json.lo.into_iter();
        let mut hi = json.hi.into_iter();
        let mut masses = json.masses.into_iter();
        let mut smoothed = json.smoothed_marginal.into_iter();
        let provenance = json.provenance;
        let mut supports = Self::from_bins(
            std::array::from_fn(|_| lo.next().expect("length checked above")),
            std::array::from_fn(|_| hi.next().expect("length checked above")),
            std::array::from_fn(|_| masses.next().expect("length checked above")),
            std::array::from_fn(|_| smoothed.next().expect("length checked above")),
            Device::Cpu,
        )?;
        // Moments are attached only when BOTH rows are present and correctly shaped. A
        // half-present pair is refused rather than half-attached, so `bin_means_measured()`
        // is never true for a support whose second moments are absent.
        if let (Some(mean), Some(second)) = (json.bin_means, json.bin_second_moments) {
            let bins = NUM_BAR_BINS as usize;
            for (what, rows) in [("bin_means", &mean), ("bin_second_moments", &second)] {
                if rows.len() != BAR_DOF || rows.iter().any(|row| row.len() != bins) {
                    bail!(
                        "bar supports {} has malformed {what}: expected {BAR_DOF} rows of {bins}",
                        path.display()
                    );
                }
                if let Some(dof) = rows.iter().position(|row| row.iter().any(|x| !x.is_finite())) {
                    bail!(
                        "bar supports {} has a non-finite {what} entry on DOF {}",
                        path.display(),
                        BAR_DOF_NAMES[dof]
                    );
                }
            }
            let mut mean = mean.into_iter();
            let mut second = second.into_iter();
            supports.bin_moments = Some(BarBinMoments::new(
                std::array::from_fn(|_| mean.next().expect("length checked above")),
                std::array::from_fn(|_| second.next().expect("length checked above")),
                Device::Cpu,
            ));
        }
        supports.provenance = provenance;
        Ok(supports)
    }

    /// Bin indices `[..., BAR_DOF]` (Int64, on `dof`'s device) for DOF values
    /// `[..., BAR_DOF]`, clamped onto the support. Tensor twin of [`Self::bin_of`],
    /// sharing its clamping and atom conventions exactly.
    pub fn bin_ids(&self, dof: &Tensor) -> Tensor {
        let lead = leading_dims(dof, BAR_DOF as i64, "target dof");
        let clamped = self.prepare(dof);
        let (index, _, _) = self.locate(&clamped);
        index
            .squeeze_dim(-1)
            .reshape(with_tail(&lead, &[BAR_DOF as i64]))
    }

    /// Bin index, in-bin position and atom mask of `[N, BAR_DOF, 1]` clamped
    /// values. The continuous lookup is the last bin whose lower bound the value
    /// reaches; an exact atom match overrides it.
    fn locate(&self, clamped: &Tensor) -> (Tensor, Tensor, Tensor) {
        let device = clamped.device();
        let continuous = clamped
            .ge_tensor(&self.lo_t.to_device(device))
            .sum_dim_intlist([-1].as_slice(), true, Kind::Int64)
            - 1;
        let continuous = continuous.clamp(0, NUM_BAR_BINS - 1);
        // NaN padding never compares equal, so unused atom slots cannot match.
        let hit = clamped.eq_tensor(&self.atom_value_t.to_device(device));
        let is_atom = hit
            .any_dim(-1, true)
            .to_kind(Kind::Int64);
        let atom_index = (hit.to_kind(Kind::Int64) * self.atom_bin_t.to_device(device))
            .sum_dim_intlist([-1].as_slice(), true, Kind::Int64);
        let index = &is_atom * atom_index + (1 - &is_atom) * continuous;

        let lo = self.gather_bin(&self.lo_flat, &index);
        let width = self.gather_bin(&self.widths_flat, &index);
        let atom_mask = is_atom.to_kind(Kind::Float);
        // An atom bin has zero width, so add the mask to the denominator instead of
        // clamping it: continuous bins keep their exact width (no distortion even for
        // a one-ulp bin) and atom rows divide by one, yielding position zero rather
        // than 0/0. Callers that care about an atom's position use their own draw.
        let position = ((clamped - lo) / (width + &atom_mask)).clamp(0.0, 1.0);
        (index, position, atom_mask)
    }

    /// Per-element lookup into a `[BAR_DOF * NUM_BAR_BINS]` table given a
    /// `[N, BAR_DOF, 1]` bin index.
    fn gather_bin(&self, table: &Tensor, index: &Tensor) -> Tensor {
        let device = index.device();
        let flat = (&index.squeeze_dim(-1) + &self.bin_offsets.to_device(device)).reshape([-1]);
        table
            .to_device(device)
            .index_select(0, &flat)
            .reshape(index.size().as_slice())
    }

    /// Clamped `[N, BAR_DOF, 1]` view of `[..., BAR_DOF]` DOF values.
    fn prepare(&self, dof: &Tensor) -> Tensor {
        let flat = dof
            .detach()
            .to_kind(Kind::Float)
            .reshape([-1, BAR_DOF as i64, 1]);
        let device = flat.device();
        let low = self.lo_t.to_device(device).narrow(1, 0, 1);
        let high = self.hi_t.to_device(device).narrow(1, NUM_BAR_BINS - 1, 1);
        flat.clamp_tensor(Some(&low), Some(&high))
    }

    /// Smoothed target distributions: `[..., BAR_DOF]` values become
    /// `[..., BAR_DOF, NUM_BAR_BINS]` rows summing to one.
    ///
    /// This is the [`BarScoring::Smoothed`] rule specifically; [`Self::targets`] is the
    /// mode-aware entry point every objective and metric goes through.
    ///
    /// An observation sitting exactly on an atom gets that atom's bin as an exact
    /// one-hot. Every other observation gets the discretized
    /// `N(x, (0.75 * min(local_bin_width, typical_bin_width))^2)`, integrated across each
    /// bin's bounds and
    /// renormalized. Atom bins are zero-width, so the integral hands them exactly
    /// zero mass without a special case, while the kernel still spans an atom's
    /// location into the continuous bins beyond it. Values outside the support clamp
    /// onto the edge bins.
    pub fn encode_targets(&self, dof: &Tensor) -> Tensor {
        self.targets(dof, BarScoring::Smoothed).into_targets()
    }

    /// The scoring rule in force, materialized against `[..., BAR_DOF]` observations.
    ///
    /// One call produces everything a loss or a metric needs, so the objective and the
    /// reported number cannot disagree about which rule is in force: they take the same
    /// [`BarTargets`].
    pub fn targets(&self, dof: &Tensor, scoring: BarScoring) -> BarTargets {
        self.targets_with_sigma(dof, scoring, BAR_LABEL_SIGMA_RATIO)
    }

    /// [`Self::targets`] with an explicit smoothing width, in multiples of the local bin
    /// width.
    ///
    /// The public path always uses [`BAR_LABEL_SIGMA_RATIO`]. The parameter exists because
    /// the smoothed rule's limit as `sigma -> 0` IS the hard rule, which is the property
    /// that pins the two modes against each other.
    fn targets_with_sigma(
        &self,
        dof: &Tensor,
        scoring: BarScoring,
        sigma_ratio: f64,
    ) -> BarTargets {
        let lead = leading_dims(dof, BAR_DOF as i64, "target dof");
        let clamped = self.prepare(dof);
        let (index, _, is_atom) = self.locate(&clamped);
        let target_shape = with_tail(&lead, &[BAR_DOF as i64, NUM_BAR_BINS]);
        let targets = if scoring.is_smoothed() {
            let smoothed = self.smooth(&clamped, &index, sigma_ratio);
            let one_hot = Tensor::zeros_like(&smoothed).scatter_value(-1, &index, 1.0);
            let continuous = is_atom.neg() + 1.0;
            (&is_atom * one_hot + continuous * smoothed).reshape(target_shape)
        } else {
            Tensor::zeros(
                [
                    index.size()[0],
                    BAR_DOF as i64,
                    NUM_BAR_BINS,
                ],
                (Kind::Float, clamped.device()),
            )
            .scatter_value(-1, &index, 1.0)
            .reshape(target_shape)
        };
        let log_measure = scoring.is_density().then(|| {
            // An atom bin has zero width and carries a probability MASS, so it takes no
            // correction at all. Adding the mask to the width — rather than clamping it —
            // leaves every continuous width exact (no distortion even for a one-ulp bin)
            // and turns an atom's `ln(0)` into `ln(1) == 0` without a branch.
            (self.gather_bin(&self.widths_flat, &index) + &is_atom)
                .log()
                .squeeze_dim(-1)
                .reshape(with_tail(&lead, &[BAR_DOF as i64]))
        });
        BarTargets {
            targets,
            log_measure,
            scoring,
        }
    }

    /// The Gaussian part of the target: `[N, BAR_DOF, NUM_BAR_BINS]` rows summing to
    /// one, before the atom one-hot override. Atom bins receive exactly zero because
    /// their bounds coincide, so the kernel integrates to nothing over them.
    ///
    /// The width is `sigma_ratio * clamp(local_width, min_width, cap_width)`. The lower
    /// clamp keeps an atom observation from producing a degenerate sigma; the upper clamp
    /// is what makes this ONE smoother instead of one per bin, and is the fix documented
    /// on [`BAR_LABEL_SIGMA_RATIO`]. Both clamps are no-ops on a uniform binning, so this
    /// is exactly the textbook HL-Gauss kernel whenever the bins are equal-WIDTH; the
    /// clamps only bite on the equal-MASS binning this head actually uses.
    fn smooth(&self, clamped: &Tensor, index: &Tensor, sigma_ratio: f64) -> Tensor {
        let device = clamped.device();
        let sigma = self
            .gather_bin(&self.widths_flat, index)
            .maximum(&self.min_width_t.to_device(device))
            .minimum(&self.cap_width_t.to_device(device))
            * sigma_ratio;
        let scale = sigma * SQRT_2;
        let upper = ((self.hi_t.to_device(device) - clamped) / &scale).erf();
        let lower = ((self.lo_t.to_device(device) - clamped) / &scale).erf();
        let mass = (upper - lower).clamp_min(0.0);
        &mass
            / mass
                .sum_dim_intlist([-1].as_slice(), true, Kind::Float)
                .clamp_min(1e-30)
    }

    /// `E[value]` under `[..., BAR_DOF, NUM_BAR_BINS]` logits, using bin centers.
    /// An atom bin's center is the atom itself.
    pub fn expectation(&self, logits: &Tensor) -> Tensor {
        let _ = factor_dims(logits, "logits");
        let probs = logits.softmax(-1, Kind::Float);
        let centers = self.centers_t.to_device(logits.device());
        (probs * centers).sum_dim_intlist([-1].as_slice(), false, Kind::Float)
    }

    /// Sample DOF values from `[..., BAR_DOF, NUM_BAR_BINS]` logits, returning
    /// `[..., BAR_DOF]`. A positive `temperature` samples a bin and then a uniform
    /// point inside it, which reproduces an atom's exact value because its bin has
    /// zero width; `temperature <= 0` takes the argmax bin center.
    pub fn sample(&self, logits: &Tensor, temperature: f64) -> Tensor {
        let lead = factor_dims(logits, "logits");
        let rows = lead.iter().product::<i64>();
        let flat = logits.reshape([-1, NUM_BAR_BINS]);
        let dof_ids = Tensor::arange(BAR_DOF as i64, (Kind::Int64, logits.device())).repeat([rows]);
        self.sample_flat(&flat, &dof_ids, temperature)
            .0
            .reshape(with_tail(&lead, &[BAR_DOF as i64]))
    }

    /// Sample a single DOF from `[..., NUM_BAR_BINS]` logits, returning `[...]`.
    pub fn sample_dof(&self, dof: usize, logits: &Tensor, temperature: f64) -> Tensor {
        self.sample_dof_binned(dof, logits, temperature).0
    }

    /// Sample a single DOF from `[..., NUM_BAR_BINS]` logits, returning the drawn
    /// value and the `[0, NUM_BAR_BINS)` bin it came from, both shaped `[...]`.
    ///
    /// The bin is handed back rather than recovered from the value because the two
    /// are not the same map: a continuous draw is `lo + (hi - lo) * u`, which can
    /// round onto the shared edge `hi == lo[j + 1]` and re-bin one bin high. The
    /// ancestral chain in [`BarEmissionHead::sample`] conditions on this bin, so
    /// the round trip has to be avoided, not merely made unlikely.
    pub fn sample_dof_binned(
        &self,
        dof: usize,
        logits: &Tensor,
        temperature: f64,
    ) -> (Tensor, Tensor) {
        assert!(dof < BAR_DOF, "DOF index {dof} out of range");
        let lead = leading_dims(logits, NUM_BAR_BINS, "logits");
        let flat = logits.reshape([-1, NUM_BAR_BINS]);
        let dof_ids = Tensor::full([flat.size()[0]], dof as i64, (Kind::Int64, logits.device()));
        let (value, index) = self.sample_flat(&flat, &dof_ids, temperature);
        (
            value.reshape(lead.as_slice()),
            index.reshape(lead.as_slice()),
        )
    }

    /// `(value, per-DOF bin index)` for flattened `[N, NUM_BAR_BINS]` logits whose
    /// row `i` belongs to DOF `dof_ids[i]`.
    fn sample_flat(&self, flat: &Tensor, dof_ids: &Tensor, temperature: f64) -> (Tensor, Tensor) {
        tch::no_grad(|| {
            let device = flat.device();
            let index = if temperature > 0.0 {
                (flat / temperature)
                    .softmax(-1, Kind::Float)
                    .multinomial(1, true)
                    .reshape([-1])
            } else {
                flat.argmax(-1, false)
            };
            let bin = dof_ids * NUM_BAR_BINS + &index;
            if temperature <= 0.0 {
                return (
                    self.centers_flat.to_device(device).index_select(0, &bin),
                    index,
                );
            }
            let lo = self.lo_flat.to_device(device).index_select(0, &bin);
            let hi = self.hi_flat.to_device(device).index_select(0, &bin);
            let uniform = Tensor::rand(lo.size().as_slice(), (Kind::Float, device));
            (&lo + (hi - &lo) * uniform, index)
        })
    }
}

/// Fit one DOF: promote atoms, then tile the gaps between them with equal-mass
/// continuous bins whose outer edges are pinned to the enclosing atoms.
fn fit_dof_support(sorted: &[f32], mandated: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let bins = NUM_BAR_BINS as usize;
    let atom_values = detect_atoms(sorted, mandated);

    let continuous: Vec<f32> = sorted
        .iter()
        .copied()
        .filter(|x| !atom_values.iter().any(|a| a == x))
        .collect();

    // Pinned points delimit the segments. The atoms always pin; the empirical
    // extremes pin only where they fall OUTSIDE the atom range, since an atom
    // already bounds the support there. Pinning an interior extreme would carve off
    // an observation-free stretch and waste bins on it.
    let mut pins: Vec<f32> = atom_values.clone();
    let (data_lo, data_hi) = clipped_range(sorted);
    match (atom_values.first().copied(), atom_values.last().copied()) {
        (Some(lowest), Some(highest)) => {
            if f32::total_cmp(&data_lo, &lowest).is_lt() {
                pins.push(data_lo);
            }
            if f32::total_cmp(&data_hi, &highest).is_gt() {
                pins.push(data_hi);
            }
        }
        _ => {
            pins.push(data_lo);
            pins.push(data_hi);
        }
    }
    pins.sort_unstable_by(f32::total_cmp);
    pins.dedup_by(|a, b| a == b);
    if pins.len() == 1 {
        // Every observation is the same value: manufacture a support around it so
        // the remaining bins have somewhere to live.
        let anchor = pins[0];
        let pad = anchor.abs().max(1.0) * 1e-3;
        pins.push(anchor + pad);
    }

    let segments: Vec<Segment> = pins
        .windows(2)
        .map(|pair| {
            let (lo, hi) = (pair[0], pair[1]);
            let start = continuous.partition_point(|x| f32::total_cmp(x, &lo).is_le());
            let end = continuous.partition_point(|x| f32::total_cmp(x, &hi).is_lt());
            Segment::new(lo, hi, &continuous[start..end], is_atom(&atom_values, lo), is_atom(&atom_values, hi))
        })
        .collect();

    let allocation = allocate_bins(&segments, bins - atom_values.len());
    let mut lo_bounds = Vec::with_capacity(bins);
    let mut hi_bounds = Vec::with_capacity(bins);
    for (i, &pin) in pins.iter().enumerate() {
        if is_atom(&atom_values, pin) {
            lo_bounds.push(pin as f64);
            hi_bounds.push(pin as f64);
        }
        if let Some(segment) = segments.get(i) {
            for pair in segment.edges(allocation[i]).windows(2) {
                lo_bounds.push(pair[0]);
                hi_bounds.push(pair[1]);
            }
        }
    }
    assert_eq!(
        lo_bounds.len(),
        bins,
        "support fit produced {} bins instead of {bins}",
        lo_bounds.len()
    );
    (lo_bounds, hi_bounds)
}

/// Renormalize each DOF row to sum to one, guarding a degenerate row.
fn normalize_rows(mut rows: [Vec<f64>; BAR_DOF]) -> [Vec<f64>; BAR_DOF] {
    for row in rows.iter_mut() {
        let total: f64 = row.iter().sum();
        if total > 0.0 {
            for p in row.iter_mut() {
                *p /= total;
            }
        } else {
            row.fill(1.0 / NUM_BAR_BINS as f64);
        }
    }
    rows
}

/// Outer support bounds: the `BAR_SUPPORT_CLIP_QUANTILE` and its complement, rather
/// than the observed extremes. `sorted` must be ascending.
fn clipped_range(sorted: &[f32]) -> (f32, f32) {
    let last = sorted.len() - 1;
    let at = |q: f64| sorted[((q * last as f64).round() as usize).min(last)];
    (
        at(BAR_SUPPORT_CLIP_QUANTILE),
        at(1.0 - BAR_SUPPORT_CLIP_QUANTILE),
    )
}

fn is_atom(atoms: &[f32], value: f32) -> bool {
    atoms.iter().any(|a| *a == value)
}

/// Exact values carrying at least [`BAR_ATOM_MASS_THRESHOLD`] of the sample, unioned
/// with the mandated set, sorted ascending and capped at [`MAX_BAR_ATOMS`].
fn detect_atoms(sorted: &[f32], mandated: &[f32]) -> Vec<f32> {
    let n = sorted.len();
    let mut detected: Vec<(f32, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        let value = sorted[i];
        let mut j = i + 1;
        while j < n && sorted[j] == value {
            j += 1;
        }
        if (j - i) as f64 / n as f64 >= BAR_ATOM_MASS_THRESHOLD && !mandated.contains(&value) {
            detected.push((value, j - i));
        }
        i = j;
    }
    detected.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(f32::total_cmp(&a.0, &b.0)));

    let mut atoms: Vec<f32> = mandated.to_vec();
    for (value, _) in detected {
        if atoms.len() >= MAX_BAR_ATOMS {
            break;
        }
        atoms.push(value);
    }
    atoms.truncate(MAX_BAR_ATOMS);
    atoms.sort_unstable_by(f32::total_cmp);
    atoms.dedup_by(|a, b| a == b);
    atoms
}

/// One inter-atom stretch of the value axis, with its observations.
struct Segment {
    lo: f32,
    hi: f32,
    /// Distinct observed values strictly inside `(lo, hi)`, ascending.
    distinct: Vec<f32>,
    /// Cumulative observation counts aligned with `distinct`.
    cumulative: Vec<usize>,
    count: usize,
    lo_is_atom: bool,
    hi_is_atom: bool,
}

impl Segment {
    fn new(lo: f32, hi: f32, values: &[f32], lo_is_atom: bool, hi_is_atom: bool) -> Self {
        let mut distinct = Vec::new();
        let mut cumulative = Vec::new();
        let mut i = 0;
        while i < values.len() {
            let value = values[i];
            let mut j = i + 1;
            while j < values.len() && values[j] == value {
                j += 1;
            }
            distinct.push(value);
            cumulative.push(j);
            i = j;
        }
        Self {
            lo,
            hi,
            distinct,
            cumulative,
            count: values.len(),
            lo_is_atom,
            hi_is_atom,
        }
    }

    /// `bins + 1` strictly increasing, `f32`-exact edges pinned at `lo` and `hi`.
    ///
    /// Interior edges are chosen from DISTINCT observed values at equal-mass
    /// positions, which makes strict `f32` increase true by construction rather than
    /// by nudging. When the segment holds fewer distinct values than it has bins,
    /// the shortfall is made up by bisecting the widest remaining gaps in `f32` bit
    /// space, which is exact and cannot collide.
    fn edges(&self, bins: usize) -> Vec<f64> {
        assert!(bins >= 1, "a segment needs at least one bin");
        let distinct = self.distinct.len();
        let first = usize::from(!self.lo_is_atom);
        let last = distinct as isize - 1 - isize::from(!self.hi_is_atom);

        let mut edges: Vec<f32> = Vec::with_capacity(bins + 1);
        edges.push(self.lo);
        if self.count > 0 {
            let mut previous = first as isize - 1;
            for k in 1..bins {
                let target = (k as f64 / bins as f64 * self.count as f64).ceil() as usize;
                let candidate = self.cumulative.partition_point(|&c| c < target) as isize;
                let index = candidate.max(previous + 1).max(first as isize);
                if index > last {
                    break;
                }
                edges.push(self.distinct[index as usize]);
                previous = index;
            }
        }
        edges.push(self.hi);
        while edges.len() < bins + 1 {
            let (at, span) = widest_key_gap(&edges);
            assert!(
                span >= 2,
                "segment [{}, {}] cannot host {bins} bins",
                self.lo,
                self.hi
            );
            edges.insert(at + 1, f32_midpoint(edges[at], edges[at + 1]));
        }
        edges.into_iter().map(f64::from).collect()
    }
}

/// Give every segment one bin, then distribute the rest by observation share using
/// largest remainder. Segments are at most `MAX_BAR_ATOMS + 1` while continuous
/// bins are at least `NUM_BAR_BINS - MAX_BAR_ATOMS`, so the floor always fits.
fn allocate_bins(segments: &[Segment], continuous_bins: usize) -> Vec<usize> {
    let count = segments.len();
    assert!(
        count >= 1 && continuous_bins >= count,
        "{continuous_bins} continuous bins cannot cover {count} segments"
    );
    let mut allocation = vec![1usize; count];
    let mut extra = continuous_bins - count;
    if extra == 0 {
        return allocation;
    }
    let total: usize = segments.iter().map(|s| s.count).sum();
    if total == 0 {
        allocation[0] += extra;
        return allocation;
    }
    let shares: Vec<f64> = segments
        .iter()
        .map(|s| s.count as f64 / total as f64 * extra as f64)
        .collect();
    let mut order: Vec<usize> = (0..count).collect();
    for (i, share) in shares.iter().enumerate() {
        let whole = share.floor() as usize;
        allocation[i] += whole;
        extra -= whole;
    }
    order.sort_by(|&a, &b| {
        let remainder = |i: usize| shares[i] - shares[i].floor();
        remainder(b)
            .total_cmp(&remainder(a))
            .then(segments[b].count.cmp(&segments[a].count))
    });
    for &i in order.iter() {
        if extra == 0 {
            break;
        }
        allocation[i] += 1;
        extra -= 1;
    }
    allocation
}

/// Monotone total-order key for finite `f32`, so bit arithmetic can bisect.
fn f32_key(x: f32) -> u32 {
    let bits = x.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

fn f32_from_key(key: u32) -> f32 {
    if key & 0x8000_0000 != 0 {
        f32::from_bits(key & 0x7FFF_FFFF)
    } else {
        f32::from_bits(!key)
    }
}

fn f32_midpoint(a: f32, b: f32) -> f32 {
    f32_from_key(((f32_key(a) as u64 + f32_key(b) as u64) / 2) as u32)
}

/// Index of the edge pair with the most representable `f32` values between them,
/// plus that count.
fn widest_key_gap(edges: &[f32]) -> (usize, u64) {
    let mut best = (0usize, 0u64);
    for (i, pair) in edges.windows(2).enumerate() {
        let span = u64::from(f32_key(pair[1])) - u64::from(f32_key(pair[0]));
        if span > best.1 {
            best = (i, span);
        }
    }
    best
}

/// Counter-based uniforms in `[0, 1)` keyed by `(seed, element index)`, so a
/// randomized statistic is bit-reproducible for a fixed seed without touching the
/// global torch generator.
fn counter_uniforms(seed: u64, count: usize) -> Vec<f32> {
    (0..count as u64)
        .map(|i| {
            let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i.wrapping_add(1)));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

fn leading_dims(t: &Tensor, last: i64, what: &str) -> Vec<i64> {
    let size = t.size();
    assert!(
        size.last().copied() == Some(last),
        "{what} must have trailing dimension {last}, got {size:?}"
    );
    size[..size.len() - 1].to_vec()
}

/// Leading dimensions of a `[..., BAR_DOF, NUM_BAR_BINS]` factor tensor.
fn factor_dims(t: &Tensor, what: &str) -> Vec<i64> {
    let size = t.size();
    assert!(
        size.len() >= 2
            && size[size.len() - 1] == NUM_BAR_BINS
            && size[size.len() - 2] == BAR_DOF as i64,
        "{what} must be shaped [..., {}, {}], got {size:?}",
        BAR_DOF,
        NUM_BAR_BINS
    );
    size[..size.len() - 2].to_vec()
}

fn with_tail(lead: &[i64], tail: &[i64]) -> Vec<i64> {
    let mut shape = Vec::with_capacity(lead.len() + tail.len());
    shape.extend_from_slice(lead);
    shape.extend_from_slice(tail);
    shape
}

// ---------------------------------------------------------------------------
// Contract C: emission head
// ---------------------------------------------------------------------------

/// Intra-bar autoregressive emission head, factorized in [`BAR_CHAIN`] order:
/// `p(bar|h) = p(r|h) p(s|h,r) p(u|h,r,s) p(v|h,r,s,u) p(w|h,r,s,u,v)`.
///
/// One `Linear(latent_dim + BAR_PREFIX_SLOTS * BAR_PREFIX_EMBED_DIM -> NUM_BAR_BINS)`
/// per DOF, plus one `[NUM_BAR_BINS, BAR_PREFIX_EMBED_DIM]` embedding table per
/// prefix slot. A constant `[BAR_DOF, BAR_PREFIX_SLOTS, 1]` mask zeroes the
/// embeddings of the slots a head may not see, which lets all five factors be
/// evaluated in a single batched pass instead of a loop over the chain.
///
/// The chain conditions on the prefix DOF's BIN, never on its raw value. An affine
/// map of the value (`x * w + b`) is exactly rank one in `x`, so the whole head
/// would collapse to `logit_bin = alpha_bin * x + beta_bin` and the embedding width
/// would buy nothing over `Linear(1 -> NUM_BAR_BINS)`. That form cannot express the
/// hard identities [`encode_dof`] manufactures — `s == 0` implies `u == v == 0.5`
/// exactly, and flat bars are ~11% of a real corpus — because a ramp in `s` has to
/// approximate a step at `s == 0` while staying sane over the whole range; measured
/// against a coarse lookup table on the same teacher-forced prefix, the affine head
/// lost 0.12 nats on `u` and 0.10 on `v`. A bin lookup is bounded, nonlinear, and
/// represents that identity exactly. It also makes the support clamp structural:
/// bins come from [`BarSupports::bin_ids`], which clamps onto
/// `[lo[0], hi[NUM_BAR_BINS - 1]]`, so the head can never be fitted on a prefix
/// value that rollout cannot produce.
///
/// The head weights are zero-initialized in the modded-nanogpt style, so training
/// starts from exactly uniform categoricals (`nll = BAR_DOF * ln(NUM_BAR_BINS)`).
#[derive(Debug)]
pub struct BarEmissionHead {
    heads: Vec<nn::Linear>,
    /// `[BAR_PREFIX_SLOTS * NUM_BAR_BINS, BAR_PREFIX_EMBED_DIM]`: the four slot
    /// tables laid end to end, so one `embedding` gathers every slot at once.
    /// Slot `s` owns rows `[s * NUM_BAR_BINS, (s + 1) * NUM_BAR_BINS)`.
    prefix_embed: Tensor,
    latent_dim: i64,
    /// `[BAR_DOF, BAR_PREFIX_SLOTS, 1]`, constant, not a VarStore variable.
    prefix_mask: Tensor,
    /// `[1, BAR_PREFIX_SLOTS]` constant `slot * NUM_BAR_BINS`, the row base of
    /// each slot's table inside [`Self::prefix_embed`].
    prefix_row_base: Tensor,
    /// `[BAR_PREFIX_SLOTS]` constant, the DOF slot occupying each prefix slot.
    prefix_slot_dof: Tensor,
}

/// Inverse-CDF draw of one bin per row from `[rows, NUM_BAR_BINS]` probabilities, using
/// [`counter_uniforms`] rather than the global torch RNG.
///
/// The evaluation path needs draws that are reproducible from a seed alone: `multinomial`
/// advances a process-wide generator, so two runs of the same checkpoint would disagree and
/// a marginalized held-out number would not be a fixed quantity. `counter_uniforms` is keyed
/// by (seed, element), so row `i` of every call draws its own stream.
fn sample_bin_by_cdf(probs: &Tensor, seed: u64) -> Tensor {
    let rows = probs.size()[0];
    let uniforms = Tensor::from_slice(&counter_uniforms(seed, rows as usize))
        .to_device(probs.device())
        .unsqueeze(-1);
    // `cdf[.., NUM_BAR_BINS - 1]` is 1.0 only up to f32 rounding, so a uniform above it
    // would index one past the end; the clamp is load-bearing, not defensive.
    probs
        .cumsum(-1, Kind::Float)
        .lt_tensor(&uniforms)
        .sum_dim_intlist([-1].as_slice(), false, Kind::Int64)
        .clamp(0, NUM_BAR_BINS - 1)
}

/// Independent stream per (draw, chain position) of a forecast mixture.
fn prefix_stream_seed(seed: u64, draw: usize, position: usize) -> u64 {
    let mut z = seed
        .wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(draw as u64 + 1))
        .wrapping_add(0xBF58_476D_1CE4_E5B9u64.wrapping_mul(position as u64 + 1));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl BarEmissionHead {
    pub fn new(vs: &nn::Path, latent_dim: i64) -> Self {
        assert!(latent_dim > 0, "bar emission head needs a positive latent dim");
        let in_features = latent_dim + BAR_PREFIX_WIDTH;
        let heads = (0..BAR_DOF)
            .map(|dof| {
                nn::linear(
                    vs / format!("bar_dof_head_{}", BAR_DOF_NAMES[dof]),
                    in_features,
                    NUM_BAR_BINS,
                    nn::LinearConfig {
                        ws_init: Init::Const(0.0),
                        bs_init: Some(Init::Const(0.0)),
                        bias: true,
                    },
                )
            })
            .collect();
        // Unit per-component scale, matching the RMS of the latent half of the head
        // input, so neither half of the concatenation dominates the learning rate.
        // The heads are zero-init, so the table sees no gradient until they move.
        let prefix_embed = vs.var(
            "bar_prefix_embed",
            &[BAR_PREFIX_SLOTS as i64 * NUM_BAR_BINS, BAR_PREFIX_EMBED_DIM],
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );

        let mut mask = vec![0f32; BAR_DOF * BAR_PREFIX_SLOTS];
        for dof in 0..BAR_DOF {
            for slot in 0..BAR_PREFIX_SLOTS {
                if slot < CHAIN_POS[dof] {
                    mask[dof * BAR_PREFIX_SLOTS + slot] = 1.0;
                }
            }
        }
        let device = vs.device();
        let prefix_mask = Tensor::from_slice(&mask)
            .view([BAR_DOF as i64, BAR_PREFIX_SLOTS as i64, 1])
            .to_device(device);
        let row_base: Vec<i64> = (0..BAR_PREFIX_SLOTS as i64)
            .map(|slot| slot * NUM_BAR_BINS)
            .collect();
        let prefix_row_base = Tensor::from_slice(&row_base)
            .view([1, BAR_PREFIX_SLOTS as i64])
            .to_device(device);
        let prefix_slot_dof = Tensor::from_slice(&PREFIX_SLOT_DOF).to_device(device);

        Self {
            heads,
            prefix_embed,
            latent_dim,
            prefix_mask,
            prefix_row_base,
            prefix_slot_dof,
        }
    }

    pub fn latent_dim(&self) -> i64 {
        self.latent_dim
    }

    fn latent_weights(&self, detach: bool) -> Tensor {
        stack_maybe_detached(
            self.heads
                .iter()
                .map(|h| h.ws.narrow(1, 0, self.latent_dim)),
            detach,
        )
    }

    fn prefix_weights(&self, detach: bool) -> Tensor {
        stack_maybe_detached(
            self.heads
                .iter()
                .map(|h| h.ws.narrow(1, self.latent_dim, BAR_PREFIX_WIDTH)),
            detach,
        )
    }

    fn biases(&self, detach: bool) -> Tensor {
        stack_maybe_detached(
            self.heads
                .iter()
                .map(|h| h.bs.as_ref().expect("head bias").shallow_clone()),
            detach,
        )
    }

    /// `[rows, BAR_PREFIX_SLOTS, BAR_PREFIX_EMBED_DIM]` slot embeddings for
    /// `[rows, BAR_PREFIX_SLOTS]` prefix bin ids. One gather over the four tables
    /// laid end to end: no GEMM, and the result is bounded by the table itself.
    fn prefix_lookup(&self, prefix_bins: &Tensor, detach: bool) -> Tensor {
        let device = prefix_bins.device();
        let table = if detach {
            self.prefix_embed.detach()
        } else {
            self.prefix_embed.shallow_clone()
        };
        let flat = (prefix_bins + self.prefix_row_base.to_device(device)).reshape([-1]);
        Tensor::embedding(&table, &flat, -1, false, false).view([
            -1,
            BAR_PREFIX_SLOTS as i64,
            BAR_PREFIX_EMBED_DIM,
        ])
    }

    /// Teacher-forced logits `[..., BAR_DOF, NUM_BAR_BINS]` for latents
    /// `[..., latent_dim]` and the ground-truth bar's `[..., BAR_DOF]` bin ids.
    ///
    /// The bins MUST come from [`BarSupports::bin_ids`] (or
    /// [`crate::torch::world_model::BarSupportSet::bin_ids`] when resolutions are
    /// mixed): that is what pins the prefix onto the fitted support and what makes
    /// an exact atom land on its own zero-width bin rather than a neighbour.
    pub fn logits(&self, h: &Tensor, target_bins: &Tensor) -> Tensor {
        self.forward_logits(h, target_bins, false)
    }

    /// Same factorization with every head parameter detached, so gradients reach
    /// only `h`. This is the predicted-latent branch of the dynamics KL term.
    pub fn logits_frozen(&self, h: &Tensor, target_bins: &Tensor) -> Tensor {
        self.forward_logits(h, target_bins, true)
    }

    fn forward_logits(&self, h: &Tensor, target_bins: &Tensor, detach: bool) -> Tensor {
        let lead = leading_dims(h, self.latent_dim, "latent");
        let bin_lead = leading_dims(target_bins, BAR_DOF as i64, "target bins");
        assert_eq!(
            lead, bin_lead,
            "latent and target bins must share leading dimensions"
        );
        assert_eq!(
            target_bins.kind(),
            Kind::Int64,
            "target bins must be the i64 output of BarSupports::bin_ids, not raw DOF values"
        );
        let device = h.device();
        let rows = lead.iter().product::<i64>();
        let h_flat = h.to_kind(Kind::Float).reshape([-1, self.latent_dim]);
        let prefix_bins = target_bins
            .reshape([-1, BAR_DOF as i64])
            .index_select(1, &self.prefix_slot_dof.to_device(device));

        let embedded = self.prefix_lookup(&prefix_bins, detach);
        let masked = (embedded.unsqueeze(1) * self.prefix_mask.to_device(device)).reshape([
            rows,
            BAR_DOF as i64,
            BAR_PREFIX_WIDTH,
        ]);

        let latent_part = Tensor::einsum(
            "nl,kol->nko",
            &[&h_flat, &self.latent_weights(detach)],
            None::<&[i64]>,
        );
        let prefix_part = Tensor::einsum(
            "nkp,kop->nko",
            &[&masked, &self.prefix_weights(detach)],
            None::<&[i64]>,
        );
        (latent_part + prefix_part + self.biases(detach).unsqueeze(0))
            .reshape(with_tail(&lead, &[BAR_DOF as i64, NUM_BAR_BINS]))
    }

    /// Ancestral sample of a bar's DOF from latents `[..., latent_dim]`, returning
    /// `[..., BAR_DOF]`. Sequential over the five chain factors (inherent to the
    /// factorization) and fully vectorized over every leading dimension.
    ///
    /// Each step conditions the rest of the chain on the BIN it drew, not on the
    /// value it decoded to, so the rollout prefix is exactly the quantity the
    /// teacher-forced path was fitted on. Re-binning the drawn value would round-trip
    /// through `lo + (hi - lo) * u`, which can land on a bin boundary and shift the
    /// conditioning by one bin.
    pub fn sample(&self, h: &Tensor, supports: &BarSupports, temperature: f64) -> Tensor {
        tch::no_grad(|| {
            let lead = leading_dims(h, self.latent_dim, "latent");
            let device = h.device();
            let rows = lead.iter().product::<i64>();
            let h_flat = h.to_kind(Kind::Float).reshape([-1, self.latent_dim]);

            let base = Tensor::einsum(
                "nl,kol->nko",
                &[&h_flat, &self.latent_weights(false)],
                None::<&[i64]>,
            ) + self.biases(false).unsqueeze(0);
            let prefix_w_all = self.prefix_weights(false);
            let mask = self.prefix_mask.to_device(device);

            // Unvisited slots hold bin 0; the mask zeroes their embedding, so the
            // seed value cannot reach any logit.
            let mut slot_bins: Vec<Tensor> = (0..BAR_PREFIX_SLOTS)
                .map(|_| Tensor::zeros([rows], (Kind::Int64, device)))
                .collect();
            let mut sampled: Vec<Option<Tensor>> = (0..BAR_DOF).map(|_| None).collect();

            for (position, &dof) in BAR_CHAIN.iter().enumerate() {
                let prefix_bins = Tensor::stack(&slot_bins, 1);
                let embedded = self.prefix_lookup(&prefix_bins, false);
                let masked =
                    (embedded * mask.select(0, dof as i64)).reshape([rows, BAR_PREFIX_WIDTH]);
                let logits = base.select(1, dof as i64)
                    + masked.linear(&prefix_w_all.select(0, dof as i64), None::<Tensor>);
                let (value, bin) = supports.sample_dof_binned(dof, &logits, temperature);
                if position < BAR_PREFIX_SLOTS {
                    slot_bins[position] = bin;
                }
                sampled[dof] = Some(value);
            }

            let values: Vec<Tensor> = sampled
                .into_iter()
                .map(|v| v.expect("every DOF sampled"))
                .collect();
            Tensor::stack(&values, -1).reshape(with_tail(&lead, &[BAR_DOF as i64]))
        })
    }

    /// Log of the MARGINALIZED predictive law of every factor: `[..., BAR_DOF,
    /// NUM_BAR_BINS]` rows holding `log( (1/S) * sum_s p(dof | h, prefix_s) )`, where each
    /// `prefix_s` is an ancestral draw of the SAME bar's preceding chain factors from the
    /// head's own law rather than their realized values.
    ///
    /// This is the difference between forecasting and within-bar accounting. [`Self::logits`]
    /// teacher-forces the prefix on the realized bar, so only [`BAR_CHAIN`]`[0]` — `r`, the
    /// factor that determines P&L — is predicted from strictly past information; `s`, `u`, `v`
    /// and `w` are each scored already knowing the realized factors ahead of them in the chain.
    /// The teacher-forced sum is still the proper joint log-likelihood of the bar, but its
    /// per-factor terms are not forecasts, and the sum of the marginals returned here is:
    /// each factor conditions only on the past. By subadditivity the marginal sum is >= the
    /// joint, with equality exactly when the chain factors are conditionally independent
    /// given `h`, so their difference measures how much of the reported per-factor skill is
    /// same-bar accounting.
    ///
    /// The estimate is a plain Monte-Carlo average of `draws` prefixes, drawn by inverse CDF
    /// from [`counter_uniforms`] and therefore reproducible from `seed` alone — no global RNG
    /// is touched. `-log` of an average is convex, so the returned law scores with an upward
    /// bias of order `1 / draws`; callers report the group standard error beside the number
    /// rather than pretending it is exact. Factor `BAR_CHAIN[0]` has no prefix, so its row is
    /// bit-identical to its teacher-forced row and its marginal is exact by construction.
    ///
    /// The latent GEMM is hoisted out of the draw loop: only the prefix embedding lookup and
    /// its `[BAR_PREFIX_WIDTH, NUM_BAR_BINS]` projection are repeated per draw.
    pub fn forecast_log_probs(
        &self,
        h: &Tensor,
        draws: usize,
        seed: u64,
    ) -> Tensor {
        assert!(draws > 0, "the forecast mixture needs at least one draw");
        tch::no_grad(|| {
            let lead = leading_dims(h, self.latent_dim, "latent");
            let device = h.device();
            let rows = lead.iter().product::<i64>();
            let h_flat = h.to_kind(Kind::Float).reshape([-1, self.latent_dim]);
            let base = Tensor::einsum(
                "nl,kol->nko",
                &[&h_flat, &self.latent_weights(false)],
                None::<&[i64]>,
            ) + self.biases(false).unsqueeze(0);
            let prefix_w_all = self.prefix_weights(false);
            let mask = self.prefix_mask.to_device(device);

            let mut total: Option<Tensor> = None;
            for draw in 0..draws {
                // Unvisited slots hold bin 0; the mask zeroes their embedding, so the seed
                // value cannot reach any logit.
                let mut slot_bins: Vec<Tensor> = (0..BAR_PREFIX_SLOTS)
                    .map(|_| Tensor::zeros([rows], (Kind::Int64, device)))
                    .collect();
                let mut per_dof: Vec<Option<Tensor>> = (0..BAR_DOF).map(|_| None).collect();
                for (position, &dof) in BAR_CHAIN.iter().enumerate() {
                    let prefix_bins = Tensor::stack(&slot_bins, 1);
                    let embedded = self.prefix_lookup(&prefix_bins, false);
                    let masked =
                        (embedded * mask.select(0, dof as i64)).reshape([rows, BAR_PREFIX_WIDTH]);
                    let logits = base.select(1, dof as i64)
                        + masked.linear(&prefix_w_all.select(0, dof as i64), None::<Tensor>);
                    let probs = logits.softmax(-1, Kind::Float);
                    if position < BAR_PREFIX_SLOTS {
                        slot_bins[position] =
                            sample_bin_by_cdf(&probs, prefix_stream_seed(seed, draw, position));
                    }
                    per_dof[dof] = Some(probs);
                }
                let stacked = Tensor::stack(
                    &per_dof
                        .into_iter()
                        .map(|p| p.expect("every DOF has a predictive row"))
                        .collect::<Vec<_>>(),
                    1,
                );
                total = Some(match total {
                    Some(acc) => acc + stacked,
                    None => stacked,
                });
            }
            let mixture = total.expect("at least one draw") / draws as f64;
            // Softmax output is strictly positive, so the mixture is too; the floor only
            // guards f32 underflow on a factor the head has driven to a point mass.
            mixture
                .clamp_min(f32::MIN_POSITIVE as f64)
                .log()
                .reshape(with_tail(&lead, &[BAR_DOF as i64, NUM_BAR_BINS]))
        })
    }

    /// Mean NLL in nats per bar plus the per-DOF `[BAR_DOF]` breakdown, under `scoring`.
    ///
    /// The rule is a parameter and has no default here on purpose: the three modes differ
    /// by additive constants, so a convenience method that picked one would let a caller
    /// report a number in a rule its objective never used.
    ///
    /// This and [`Self::crps`] / [`Self::pit`] each run their own teacher-forced
    /// forward. To report all three in one step, call [`Self::logits`] once and
    /// feed the result to [`bar_nll_from_logits`], [`bar_crps_from_logits`] and
    /// [`bar_pit_from_logits`]. Likewise, build a device-resident
    /// [`BarSupports::to_device`] copy once instead of letting every call copy the
    /// cached edges across the PCIe bus.
    pub fn nll(
        &self,
        h: &Tensor,
        target_dof: &Tensor,
        supports: &BarSupports,
        scoring: BarScoring,
    ) -> (Tensor, Tensor) {
        let logits = self.logits(h, &supports.bin_ids(target_dof));
        bar_nll_from_logits(&logits, &supports.targets(target_dof, scoring))
    }

    /// Per-DOF `[BAR_DOF]` CRPS of the predictive categoricals, for calibration
    /// reporting.
    pub fn crps(&self, h: &Tensor, target_dof: &Tensor, supports: &BarSupports) -> Tensor {
        let logits = self.logits(h, &supports.bin_ids(target_dof));
        bar_crps_from_logits(&logits, target_dof, supports)
    }

    /// Per-sample PIT values in `[0, 1]`, shaped like `target_dof`, for histogram
    /// reporting. `seed` makes the randomized draws at atoms reproducible.
    pub fn pit(
        &self,
        h: &Tensor,
        target_dof: &Tensor,
        supports: &BarSupports,
        seed: u64,
    ) -> Tensor {
        let logits = self.logits(h, &supports.bin_ids(target_dof));
        bar_pit_from_logits(&logits, target_dof, supports, seed)
    }
}

fn stack_maybe_detached(parts: impl Iterator<Item = Tensor>, detach: bool) -> Tensor {
    let collected: Vec<Tensor> = if detach {
        parts.map(|t| t.detach()).collect()
    } else {
        parts.collect()
    };
    Tensor::stack(&collected, 0)
}

/// The scoring rule of [`BarScoring`], materialized against one batch of observations:
/// the target distribution over bins plus the additive measure correction.
///
/// Carrying both together is what makes the objective and the reported metric agree by
/// construction. A caller that took only the target rows would silently score
/// [`BarScoring::Density`] as [`BarScoring::Hard`].
#[derive(Debug)]
pub struct BarTargets {
    /// `[..., BAR_DOF, NUM_BAR_BINS]` rows summing to one.
    targets: Tensor,
    /// `[..., BAR_DOF]` additive nats: `+ln(width_b)` for a continuous observation under
    /// [`BarScoring::Density`], and zero on an atom, whose factor is a probability MASS.
    /// `None` for the discrete rules, where it would be an all-zero tensor.
    log_measure: Option<Tensor>,
    scoring: BarScoring,
}

impl BarTargets {
    /// The `[..., BAR_DOF, NUM_BAR_BINS]` target rows.
    pub fn targets(&self) -> &Tensor {
        &self.targets
    }

    /// The additive measure term, `None` under the two discrete rules.
    pub fn log_measure(&self) -> Option<&Tensor> {
        self.log_measure.as_ref()
    }

    pub fn scoring(&self) -> BarScoring {
        self.scoring
    }

    /// Consume into the target rows alone. Only correct where the measure term is known to
    /// be absent, i.e. for a deliberately mode-specific caller.
    pub fn into_targets(self) -> Tensor {
        debug_assert!(
            self.log_measure.is_none(),
            "discarding the measure term of a density-scored target"
        );
        self.targets
    }

    /// `[..., BAR_DOF]` measure term, or a broadcastable zero.
    fn measure_or_zero(&self, like: &Tensor) -> Tensor {
        match &self.log_measure {
            Some(measure) => measure.shallow_clone(),
            None => Tensor::zeros([], (Kind::Float, like.device())),
        }
    }
}

/// Per-factor nats of `[..., BAR_DOF, NUM_BAR_BINS]` logits under the scoring rule
/// `targets` was built with, reduced over the bin axis ONLY: the result is
/// `[..., BAR_DOF]`.
///
/// [`bar_nll_from_logits`] averages this over every leading axis and throws the individual
/// values away, which is exactly what makes a held-out mean have no measurable dispersion.
/// Selection needs the per-window vector — to block-bootstrap a confidence interval, and to
/// pair two runs window by window — and the conditional metric needs the per-BAR values so
/// the `u`/`v` terms can be masked to non-flat bars. Both come off this tensor.
pub fn bar_nll_terms(logits: &Tensor, targets: &BarTargets) -> Tensor {
    let _ = factor_dims(logits, "logits");
    assert_eq!(
        logits.size(),
        targets.targets.size(),
        "logits and targets must have identical shapes"
    );
    let log_probs = logits.log_softmax(-1, Kind::Float);
    let cross_entropy =
        -(&targets.targets * log_probs).sum_dim_intlist([-1].as_slice(), false, Kind::Float);
    cross_entropy + targets.measure_or_zero(logits)
}

/// Nats per bar of `[..., BAR_DOF, NUM_BAR_BINS]` logits under the scoring rule `targets`
/// was built with. Returns `(mean nats per bar, per-DOF nats)` where the scalar is the sum
/// over the five chain factors.
pub fn bar_nll_from_logits(logits: &Tensor, targets: &BarTargets) -> (Tensor, Tensor) {
    let per_dof = bar_nll_terms(logits, targets)
        .reshape([-1, BAR_DOF as i64])
        .mean_dim([0i64].as_slice(), false, Kind::Float);
    (per_dof.sum(Kind::Float), per_dof)
}

/// Per-DOF split of a bar NLL into the DEGENERACY class and the intra-continuous
/// SHAPE, in nats. `class[dof] + shape[dof]` is that DOF's total NLL.
///
/// The two halves answer different questions and, measured on a real corpus, are
/// the same size — the class term alone carries 3.03 of the 21.69 marginal nats,
/// which is as large as the entire gain a trained head currently reports. Charted
/// as one number they are indistinguishable, and they should not be: getting the
/// degeneracy class right is largely a logical consequence of the chain prefix
/// (`s == 0` forces `u == v == 0.5`), whereas placing the bar inside the continuous
/// part is the forecasting problem.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BarNllSplit {
    /// `-log p(which atom, or "continuous")`, per DOF.
    pub class: [f64; BAR_DOF],
    /// `-log p(bin | continuous)` weighted by the target's continuous mass, per DOF.
    pub shape: [f64; BAR_DOF],
}

impl BarNllSplit {
    pub fn total(&self) -> [f64; BAR_DOF] {
        std::array::from_fn(|dof| self.class[dof] + self.shape[dof])
    }

    pub fn class_bar(&self) -> f64 {
        self.class.iter().sum()
    }

    pub fn shape_bar(&self) -> f64 {
        self.shape.iter().sum()
    }
}

/// Batch-side twin of [`BarNllSplit`]: three `[BAR_DOF]` tensors of mean nats.
#[derive(Debug)]
pub struct BarNllParts {
    /// `[BAR_DOF]` mean nats spent on the degeneracy class.
    pub class: Tensor,
    /// `[BAR_DOF]` mean nats spent inside the continuous part.
    pub shape: Tensor,
    /// `[BAR_DOF]` mean total, identical to [`bar_nll_from_logits`]'s per-DOF term.
    pub total: Tensor,
}

/// Split the per-factor NLL of `[..., BAR_DOF, NUM_BAR_BINS]` logits into
/// [`BarNllParts`], using the support's zero-width bins as the atom set.
///
/// Writing `A` for the atom bins and `C` for the continuous ones, `p_C = sum_C p`
/// and `t_C = sum_C t`, the identity is
/// `-sum t log p  =  [-sum_A t log p - t_C log p_C] + [-sum_C t log(p / p_C)]`,
/// which is exact for every row: the class term is the cross entropy of the
/// `|A| + 1`-way degeneracy indicator and the shape term is the `t_C`-weighted
/// cross entropy of the within-continuous law. Atom observations get an exact
/// one-hot under every scoring rule, so they contribute `t_C = 0` and spend everything on
/// the class term; continuous observations get `t_C = 1` because the Gaussian integrates to
/// zero over a zero-width bin.
///
/// Under [`BarScoring::Density`] the measure term joins `shape`: which atom the bar sits on,
/// or whether it is continuous at all, is a discrete decision that carries no width, and
/// only the placement inside the continuous part is a density.
///
/// Chart these against [`BarSupports::marginal_nll_parts`], which decomposes the
/// matching marginal reference the same way.
pub fn bar_nll_decomposition(
    logits: &Tensor,
    targets: &BarTargets,
    supports: &BarSupports,
) -> BarNllParts {
    let _ = factor_dims(logits, "logits");
    let soft_targets = &targets.targets;
    assert_eq!(
        logits.size(),
        soft_targets.size(),
        "logits and targets must have identical shapes"
    );
    let device = logits.device();
    // Atom bins ARE the zero-width bins, and the width table is already resident on
    // the training device, so the mask costs no host transfer and cannot drift out
    // of sync with the geometry `locate` bins against.
    let is_atom = supports
        .widths_flat
        .to_device(device)
        .view([BAR_DOF as i64, NUM_BAR_BINS])
        .eq(0.0);
    let atom = is_atom.to_kind(Kind::Float);
    let continuous = is_atom.logical_not().to_kind(Kind::Float);

    let log_probs = logits.log_softmax(-1, Kind::Float);
    let log_p_cont = log_probs
        .masked_fill(&is_atom, f64::NEG_INFINITY)
        .logsumexp([-1].as_slice(), true);
    let sum_last = |t: Tensor| t.sum_dim_intlist([-1].as_slice(), true, Kind::Float);
    let target_cont = sum_last(soft_targets * &continuous);
    let atom_term = sum_last(soft_targets * &log_probs * &atom);
    let cont_term = sum_last(soft_targets * &log_probs * &continuous);

    let measure = targets.measure_or_zero(logits);
    let class = -(atom_term + &target_cont * &log_p_cont).squeeze_dim(-1);
    let shape = -(cont_term - target_cont * log_p_cont).squeeze_dim(-1) + &measure;
    let total = -sum_last(soft_targets * &log_probs).squeeze_dim(-1) + measure;

    let mean_per_dof = |t: Tensor| {
        t.reshape([-1, BAR_DOF as i64])
            .mean_dim([0i64].as_slice(), false, Kind::Float)
    };
    BarNllParts {
        class: mean_per_dof(class),
        shape: mean_per_dof(shape),
        total: mean_per_dof(total),
    }
}

/// `KL(sg[softmax(target_logits)] || softmax(pred_logits))` over the per-DOF
/// categoricals. Returns `(mean nats per bar, per-DOF nats)`; the target branch is
/// detached, matching the stop-gradient in the dynamics objective.
pub fn bar_categorical_kl(target_logits: &Tensor, pred_logits: &Tensor) -> (Tensor, Tensor) {
    let _ = factor_dims(target_logits, "target logits");
    assert_eq!(
        target_logits.size(),
        pred_logits.size(),
        "target and predicted logits must have identical shapes"
    );
    let target_log = target_logits.detach().log_softmax(-1, Kind::Float);
    let target = target_log.exp();
    let pred_log = pred_logits.log_softmax(-1, Kind::Float);
    let kl = (&target * (target_log - pred_log)).sum_dim_intlist(
        [-1].as_slice(),
        false,
        Kind::Float,
    );
    let per_dof = kl
        .reshape([-1, BAR_DOF as i64])
        .mean_dim([0i64].as_slice(), false, Kind::Float);
    (per_dof.sum(Kind::Float), per_dof)
}

/// Per-DOF `[BAR_DOF]` CRPS of the categorical predictive distributions, treating
/// the bins as atoms at their centers. Uses the `O(NUM_BAR_BINS)` identity
/// `CRPS = E|X - y| - integral F (1 - F)`, so it is exactly zero for a point mass
/// on the observed value.
pub fn bar_crps_from_logits(
    logits: &Tensor,
    target_dof: &Tensor,
    supports: &BarSupports,
) -> Tensor {
    let _ = factor_dims(logits, "logits");
    let device = logits.device();
    let probs = logits.softmax(-1, Kind::Float);
    let centers = supports.centers_t.to_device(device);
    let target = target_dof
        .detach()
        .to_kind(Kind::Float)
        .reshape([-1, BAR_DOF as i64, 1]);
    let probs = probs.reshape([-1, BAR_DOF as i64, NUM_BAR_BINS]);

    let absolute = (&probs * (&centers - &target).abs()).sum_dim_intlist(
        [-1].as_slice(),
        false,
        Kind::Float,
    );
    let cdf = probs.cumsum(-1, Kind::Float).narrow(-1, 0, NUM_BAR_BINS - 1);
    let gaps = centers.narrow(1, 1, NUM_BAR_BINS - 1) - centers.narrow(1, 0, NUM_BAR_BINS - 1);
    let complement = cdf.neg() + 1.0;
    let spread =
        (complement * cdf * gaps).sum_dim_intlist([-1].as_slice(), false, Kind::Float);
    (absolute - spread)
        .clamp_min(0.0)
        .mean_dim([0i64].as_slice(), false, Kind::Float)
}

/// Probability integral transform of each observation under its predictive
/// categorical, shaped like `target_dof`.
///
/// Inside a continuous bin the predictive density is read as uniform, which is the
/// continuous extension of the equal-mass discretization. At an ATOM the PIT is
/// randomized: the observation is placed uniformly across that atom's probability
/// interval `[F(atom-), F(atom)]`, which is what keeps the histogram uniform for a
/// well-specified head instead of spiking at the atom's cumulative probability.
///
/// The draws come from a counter-based generator keyed by `(seed, flat element
/// index)`, so a fixed `seed` gives a bit-reproducible histogram across runs
/// without perturbing the global torch generator.
pub fn bar_pit_from_logits(
    logits: &Tensor,
    target_dof: &Tensor,
    supports: &BarSupports,
    seed: u64,
) -> Tensor {
    let lead = leading_dims(target_dof, BAR_DOF as i64, "target dof");
    let _ = factor_dims(logits, "logits");
    let probs = logits
        .detach()
        .softmax(-1, Kind::Float)
        .reshape([-1, BAR_DOF as i64, NUM_BAR_BINS]);
    let clamped = supports.prepare(target_dof);
    let (index, position, is_atom) = supports.locate(&clamped);
    let draws = counter_uniforms(seed, (index.numel()) as usize);
    let uniform = Tensor::from_slice(&draws)
        .to_device(index.device())
        .reshape(index.size().as_slice());
    let continuous = is_atom.neg() + 1.0;
    let offset = &is_atom * uniform + continuous * position;
    let inclusive = probs.cumsum(-1, Kind::Float).gather(-1, &index, false);
    let mass = probs.gather(-1, &index, false);
    ((inclusive - &mass) + mass * offset)
        .clamp(0.0, 1.0)
        .reshape(with_tail(&lead, &[BAR_DOF as i64]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::test_rng;
    use tch::nn::OptimizerConfig;

    /// Deterministic xorshift64* stream, so tests never depend on a global RNG.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let u1 = self.uniform().max(1e-12);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    fn bar(open: f32, high: f32, low: f32, close: f32, volume: f32) -> PackedBar {
        PackedBar {
            ts_ms: 1_700_000_000_000,
            open,
            high,
            low,
            close,
            volume,
            vwap: 0.5 * (high + low),
            trades: 42,
        }
    }

    fn relative(actual: f32, expected: f32) -> f64 {
        ((actual as f64 - expected as f64) / (expected as f64).abs().max(1e-12)).abs()
    }

    fn synthetic_dof(rng: &mut Rng) -> BarDof {
        let s = (0.002 * (1.0 + 0.5 * rng.normal()).abs()).min(0.2);
        BarDof {
            r: (0.0009 * rng.normal()) as f32,
            s: s as f32,
            u: rng.uniform() as f32,
            v: rng.uniform() as f32,
            w: (0.4 * rng.normal()) as f32,
        }
    }

    fn synthetic_supports(count: usize, seed: u64) -> BarSupports {
        let mut rng = Rng::new(seed);
        let samples: Vec<BarDof> = (0..count).map(|_| synthetic_dof(&mut rng)).collect();
        BarSupports::fit(&samples)
    }

    /// A fixture whose EQUAL-MASS bins have the pathological width spread the live corpus
    /// actually has. `synthetic_dof` draws `r` Gaussian, which gives quantile bins spanning
    /// only a couple of decades and would let a tail-blind smoother pass unnoticed; the live
    /// 300s supports span 2743x on `r` because the real tail exponent is about 1.8.
    ///
    /// Drawn as a signed Pareto with `alpha = 1.8`, which is the exponent measured off
    /// `long_data/bars/bar_supports.300.json`'s own persisted quantile grid, plus the atom
    /// at zero that an unchanged price produces.
    fn heavy_tailed_supports(count: usize, seed: u64) -> BarSupports {
        BarSupports::fit(&heavy_tailed_samples(count, seed))
    }

    /// The samples behind [`heavy_tailed_supports`], for tests that must compare a fitted
    /// support against the very sample it was fitted on.
    fn heavy_tailed_samples(count: usize, seed: u64) -> Vec<BarDof> {
        let mut rng = Rng::new(seed);
        let pareto = |rng: &mut Rng, scale: f64| {
            let u = rng.uniform().max(1e-12);
            let magnitude = scale * u.powf(-1.0 / 1.8);
            if rng.uniform() < 0.5 {
                -magnitude
            } else {
                magnitude
            }
        };
        (0..count)
            .map(|i| BarDof {
                // Every ninth bar closes unchanged, which is the atom the real `r` carries.
                r: if i % 9 == 0 {
                    0.0
                } else {
                    pareto(&mut rng, 3e-5) as f32
                },
                s: pareto(&mut rng, 2e-5).abs() as f32,
                u: rng.uniform() as f32,
                v: rng.uniform() as f32,
                w: pareto(&mut rng, 1e-2) as f32,
            })
            .collect()
    }

    /// THE identity the decode fix rests on: `sum_b p_b E[x | b] == E[x]`, exactly, and the
    /// same for the second moment. A per-bin representative satisfies it if and only if it is
    /// the fitted conditional mean, so this is what separates the fix from every geometric
    /// stand-in — and the second half of the test proves the bin CENTERS do not satisfy it,
    /// which is the bug being repaired rather than a restatement of the fix.
    ///
    /// Deliberately on the heavy-tailed fixture: the whole error lives in the two catch-all
    /// bins, so a light-tailed sample would pass under either decode and prove nothing.
    #[test]
    fn fitted_bin_moments_reproduce_the_sample_moments_and_centers_do_not() {
        let _torch_rng_guard = test_rng::shared();
        let samples = heavy_tailed_samples(200_000, 0xB1A5);
        let supports = BarSupports::fit(&samples);
        assert!(
            supports.bin_means_measured(),
            "a freshly fitted support must carry measured moments"
        );

        for dof in 0..BAR_DOF {
            let rows: Vec<f64> = samples
                .iter()
                .filter(|d| d.is_finite())
                .map(|d| d.to_array()[dof] as f64)
                .collect();
            let n = rows.len() as f64;
            let (truth_mean, truth_second) = (
                rows.iter().sum::<f64>() / n,
                rows.iter().map(|x| x * x).sum::<f64>() / n,
            );
            let masses = supports.bin_masses(dof);
            let means = supports.bin_means(dof).expect("fitted means");
            let seconds = supports.bin_second_moments(dof).expect("fitted second moments");
            let mixed_mean: f64 = masses.iter().zip(means).map(|(p, m)| p * m).sum();
            let mixed_second: f64 = masses.iter().zip(seconds).map(|(p, s)| p * s).sum();

            // Tolerance is relative to the scale of the quantity itself and covers only the
            // f32 narrowing of the persisted moments — nothing statistical, because the
            // identity is algebraic on the very sample that was fitted.
            let mean_scale = truth_second.sqrt().max(1e-12);
            assert!(
                (mixed_mean - truth_mean).abs() < 1e-5 * mean_scale,
                "DOF {}: fitted means give E[x] = {mixed_mean:.6e}, the sample has {truth_mean:.6e}",
                BAR_DOF_NAMES[dof]
            );
            assert!(
                (mixed_second - truth_second).abs() < 1e-5 * truth_second.max(1e-24),
                "DOF {}: fitted moments give E[x^2] = {mixed_second:.6e}, the sample has \
                 {truth_second:.6e}",
                BAR_DOF_NAMES[dof]
            );
        }

        // The centers cannot do this, and `r` is where it hurts: the two catch-alls decode
        // to their outer bounds, so the same mixture badly overstates the second moment. If
        // this half ever passes, the centers have silently become conditional means and the
        // first half of the test has stopped discriminating.
        let masses = supports.bin_masses(DOF_R);
        let centers = supports.centers(DOF_R);
        let center_second: f64 = masses.iter().zip(centers).map(|(p, c)| p * c * c).sum();
        let rows: Vec<f64> = samples
            .iter()
            .filter(|d| d.is_finite())
            .map(|d| d.r as f64)
            .collect();
        let truth_second = rows.iter().map(|x| x * x).sum::<f64>() / rows.len() as f64;
        assert!(
            center_second > 2.0 * truth_second,
            "the center decode is supposed to overstate E[r^2]; it gave {center_second:.6e} \
             against a true {truth_second:.6e}, so this fixture no longer exercises the bug"
        );
    }

    /// The bracketing-pair variance bound is a genuine lower bound on every representable
    /// distribution AND is attained, which is what makes it exact rather than merely
    /// conservative. Both halves are asserted, because a bound that is never tight would
    /// pass the first half vacuously.
    ///
    /// It also pins the two facts that were twice re-derived wrongly: a mean sitting exactly
    /// ON a decode value is free, so the most extreme forecast the support can express costs
    /// no predicted uncertainty at all; and a mean strictly inside the catch-all gap is
    /// expensive. Those two together are why no bound on the predicted MEAN can be asserted.
    #[test]
    fn the_bracketing_variance_bound_is_a_bound_and_is_attained() {
        let _torch_rng_guard = test_rng::shared();
        let supports = heavy_tailed_supports(60_000, 0xBEEF);
        let mut rng = Rng::new(7);
        let bins = NUM_BAR_BINS as usize;
        for dof in 0..BAR_DOF {
            let decode: Vec<f64> = supports.bin_means(dof).expect("fitted means").to_vec();

            // A mean landing exactly on a decode value needs no mixing, so the bound is zero
            // there — including the outermost bin, the case every mean-side bound rejects.
            for &at in &decode {
                assert_eq!(
                    supports
                        .min_variance_for_mean(dof, at, MeanDecode::Fitted)
                        .expect("fitted decode"),
                    0.0,
                    "DOF {}: a mean sitting on the decode value {at:.6e} must be free",
                    BAR_DOF_NAMES[dof]
                );
            }

            // Lower bound: no random distribution over the bins may undercut it.
            for _ in 0..64 {
                let weights: Vec<f64> = (0..bins).map(|_| rng.uniform().powf(6.0)).collect();
                let total: f64 = weights.iter().sum();
                if total <= 0.0 {
                    continue;
                }
                let probs: Vec<f64> = weights.iter().map(|w| w / total).collect();
                let mean: f64 = probs.iter().zip(&decode).map(|(p, d)| p * d).sum();
                let variance: f64 = probs
                    .iter()
                    .zip(&decode)
                    .map(|(p, d)| p * (d - mean) * (d - mean))
                    .sum();
                let bound = supports
                    .min_variance_for_mean(dof, mean, MeanDecode::Fitted)
                    .expect("fitted decode");
                assert!(
                    variance >= bound * (1.0 - 1e-9) - 1e-30,
                    "DOF {}: a distribution with mean {mean:.6e} has variance {variance:.6e}, \
                     under the {bound:.6e} the bracketing pair forces",
                    BAR_DOF_NAMES[dof]
                );
            }

            // Attained: the two-point mixture on a bracketing pair matches the bound exactly.
            let mut sorted = decode.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite decode"));
            for pair in sorted.windows(2) {
                let (below, above) = (pair[0], pair[1]);
                if above - below <= 0.0 {
                    continue;
                }
                for w in [0.1, 0.5, 0.9] {
                    let mean = below + w * (above - below);
                    let variance = w * (1.0 - w) * (above - below) * (above - below);
                    let bound = supports
                        .min_variance_for_mean(dof, mean, MeanDecode::Fitted)
                        .expect("fitted decode");
                    let scale = variance.max(1e-30);
                    assert!(
                        (variance - bound).abs() < 1e-6 * scale,
                        "DOF {}: the two-point mixture at w={w} has variance {variance:.6e} but \
                         the bound says {bound:.6e}; the bound is not tight",
                        BAR_DOF_NAMES[dof]
                    );
                }
            }
        }

        // And the gap is expensive in the middle while free at both ends, which is the whole
        // reason a mean-side assertion is unsafe on this geometry.
        let means = supports.bin_means(DOF_R).expect("fitted means").to_vec();
        let (interior, outer) = (
            supports
                .interior_mean_ceiling(DOF_R, MeanDecode::Fitted)
                .expect("a fitted support offers the fitted decode"),
            supports
                .representable_mean_ceiling(DOF_R, MeanDecode::Fitted)
                .expect("a fitted support offers the fitted decode"),
        );
        let midpoint = 0.5 * (interior + outer);
        assert!(
            supports
                .min_variance_for_mean(DOF_R, midpoint, MeanDecode::Fitted)
                .expect("fitted decode")
                .sqrt()
                > 0.25 * (outer - interior),
            "a mean in the middle of the catch-all gap must force a large sd"
        );
        let extreme = means
            .iter()
            .copied()
            .fold(0.0f64, |worst, m| if m.abs() > worst.abs() { m } else { worst });
        assert_eq!(
            supports
                .min_variance_for_mean(DOF_R, extreme, MeanDecode::Fitted)
                .expect("fitted decode"),
            0.0,
            "the most extreme representable mean must cost no predicted uncertainty"
        );
    }

    /// `|E[x]|` can never exceed the largest decode magnitude, and cannot exceed the
    /// INTERIOR one without catch-all mass. The first is what makes an impossible predicted
    /// mean detectable; the second is what distinguishes "confident about an interior move"
    /// from "leaning on the catch-alls".
    #[test]
    fn the_mean_ceiling_bounds_every_representable_expectation() {
        let _torch_rng_guard = test_rng::shared();
        let supports = heavy_tailed_supports(60_000, 0xC0FF);
        let mut rng = Rng::new(31);
        let bins = NUM_BAR_BINS as usize;
        for dof in 0..BAR_DOF {
            let means = supports.bin_means(dof).expect("fitted means");
            let (all, interior) = (
                supports
                    .representable_mean_ceiling(dof, MeanDecode::Fitted)
                    .expect("fitted decode"),
                supports
                    .interior_mean_ceiling(dof, MeanDecode::Fitted)
                    .expect("fitted decode"),
            );
            assert!(
                interior <= all,
                "DOF {}: the interior ceiling {interior:.6e} exceeds the all-bin one {all:.6e}",
                BAR_DOF_NAMES[dof]
            );
            for trial in 0..64 {
                // Half the draws put no mass on the catch-alls at all, which is the only
                // case the interior ceiling claims to bound.
                let interior_only = trial % 2 == 0;
                let weights: Vec<f64> = (0..bins)
                    .map(|bin| {
                        if interior_only && (bin == 0 || bin == bins - 1) {
                            0.0
                        } else {
                            rng.uniform().powf(4.0)
                        }
                    })
                    .collect();
                let total: f64 = weights.iter().sum();
                if total <= 0.0 {
                    continue;
                }
                let mean: f64 = weights
                    .iter()
                    .zip(means)
                    .map(|(w, m)| w / total * m)
                    .sum();
                let bound = if interior_only { interior } else { all };
                assert!(
                    mean.abs() <= bound * (1.0 + 1e-9) + 1e-18,
                    "DOF {}: a distribution over {} bins produced |E[x]| = {:.6e}, past its \
                     ceiling of {bound:.6e}",
                    BAR_DOF_NAMES[dof],
                    if interior_only { "interior" } else { "all" },
                    mean.abs()
                );
            }
        }

        // And the gap is real, not a formality: on a heavy tail the catch-alls carry a
        // representative far outside the interior range, which is exactly why a predicted
        // mean above the interior ceiling is evidence about WHERE the mass sits.
        assert!(
            supports
                .representable_mean_ceiling(DOF_R, MeanDecode::Fitted)
                .expect("fitted decode")
                > 1.5
                    * supports
                        .interior_mean_ceiling(DOF_R, MeanDecode::Fitted)
                        .expect("fitted decode"),
            "this fixture no longer separates the two ceilings"
        );
    }

    /// A pre-v5 artifact reports no fitted moments rather than presenting its geometry as a
    /// measurement, and a file CLAIMING v5 without them is refused outright. The refusal is
    /// the load-bearing half: without it a v5 written by a build that failed to populate the
    /// moments would be indistinguishable from an honest v4.
    #[test]
    fn legacy_supports_report_no_fitted_moments_and_a_lying_v5_is_refused() {
        let _torch_rng_guard = test_rng::shared();
        let dir = std::env::temp_dir()
            .join(format!("trading_bot_0_supports_moments_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("bar_supports.300.json");

        let fitted = synthetic_supports(20_000, 0x5EED);
        fitted.save(&path).expect("save");
        let reloaded = BarSupports::load(&path).expect("load");
        assert!(reloaded.bin_means_measured(), "a v5 round trip keeps its moments");
        for dof in 0..BAR_DOF {
            let (before, after) = (
                fitted.bin_means(dof).expect("fitted"),
                reloaded.bin_means(dof).expect("reloaded"),
            );
            assert_eq!(before, after, "DOF {} means changed on reload", BAR_DOF_NAMES[dof]);
        }
        // Moving them to a device must not lose them either — the training path only ever
        // sees a `to_device` copy.
        assert!(reloaded.to_device(Device::Cpu).bin_means_measured());

        let mut raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("parse");

        // v4: provenance but no moments. Loads, and says so.
        let object = raw.as_object_mut().expect("object");
        object.insert("format_version".to_owned(), serde_json::json!(4));
        object.remove("bin_means");
        object.remove("bin_second_moments");
        std::fs::write(&path, serde_json::to_vec(&raw).expect("serialize")).expect("write");
        let legacy = BarSupports::load(&path).expect("v4 load");
        assert!(
            !legacy.bin_means_measured(),
            "a v4 artifact must not claim measured moments"
        );
        assert_eq!(legacy.bin_means(DOF_R), None);
        assert_eq!(legacy.bin_second_moments(DOF_R), None);
        // THE LOUD-ABSENCE CONTRACT, and the regression test for the removed
        // `unwrap_or_else(|| centers)`. The EDGE convention is always available, so the
        // ceiling still answers under it. The FITTED convention is NOT available here and
        // asking for it is an ERROR — it does not quietly hand back the edge decode. That
        // silent substitution is exactly how a v4 artifact kept returning the 883.32 bps
        // pre-fix ceiling after the fitted decode was believed to have landed, with nothing
        // anywhere in the output saying which decode had actually been read.
        assert!(
            legacy
                .representable_mean_ceiling(DOF_R, MeanDecode::Edge)
                .expect("the edge decode is always available")
                > 0.0
        );
        for convention in [MeanDecode::Edge, MeanDecode::Fitted] {
            let all = legacy.representable_mean_ceiling(DOF_R, convention);
            let interior = legacy.interior_mean_ceiling(DOF_R, convention);
            let decode = legacy.mean_decode(DOF_R, convention);
            match convention {
                MeanDecode::Edge => {
                    assert!(all.is_ok() && interior.is_ok() && decode.is_ok());
                }
                MeanDecode::Fitted => {
                    for (what, outcome) in [
                        ("representable_mean_ceiling", all.is_err()),
                        ("interior_mean_ceiling", interior.is_err()),
                        ("mean_decode", decode.is_err()),
                    ] {
                        assert!(
                            outcome,
                            "{what} answered for the FITTED decode on a support carrying no \
                             measured moments; an absent measurement must not read as a real one"
                        );
                    }
                }
            }
        }
        assert!(
            legacy.mean_decode_tensor(MeanDecode::Fitted).is_err(),
            "the batched decode must refuse the fitted convention too; it is where a silent \
             substitution would be least visible"
        );

        // A file claiming v5 with the moments stripped is a lie about its own contents.
        raw["format_version"] = serde_json::json!(BAR_SUPPORTS_MOMENTS_VERSION);
        std::fs::write(&path, serde_json::to_vec(&raw).expect("serialize")).expect("write");
        assert!(
            BarSupports::load(&path).is_err(),
            "a v5 file missing its fitted moments must be refused, not loaded as a legacy one"
        );

        // So is a half-present pair, which would otherwise attach means with no second
        // moments and let `bin_means_measured()` be true for an incomplete support.
        raw["bin_means"] = serde_json::json!(vec![vec![0.0f64; NUM_BAR_BINS as usize]; BAR_DOF]);
        std::fs::write(&path, serde_json::to_vec(&raw).expect("serialize")).expect("write");
        assert!(BarSupports::load(&path).is_err(), "a half-present moment pair must be refused");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// THE PRODUCTION FAILURE, moved to the write boundary. A run whose supports came from the
    /// live pre-v5 corpus artifact used to write a checkpoint sidecar stamped
    /// `format_version: 5` with no moments in it, and then fail its own reload check — after
    /// 1000 steps and its first promotion, holding a file the loader was right to refuse. The
    /// write must fail INSTEAD, and nothing may reach disk.
    #[test]
    fn a_support_carrying_no_fitted_moments_cannot_be_written_at_all() {
        let _torch_rng_guard = test_rng::shared();
        let dir = std::env::temp_dir()
            .join(format!("trading_bot_0_supports_unwritable_{}", uuid::Uuid::new_v4()));
        let corpus_artifact = dir.join("bar_supports.300.json");
        synthetic_supports(20_000, 0x5EED).save(&corpus_artifact).expect("v5 save");

        // Exactly the shape of the live artifact: v4 geometry, no moments.
        let mut raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&corpus_artifact).expect("read"))
                .expect("parse");
        let object = raw.as_object_mut().expect("object");
        object.insert("format_version".to_owned(), serde_json::json!(4));
        object.remove("bin_means");
        object.remove("bin_second_moments");
        std::fs::write(&corpus_artifact, serde_json::to_vec(&raw).expect("serialize"))
            .expect("write");
        let loaded = BarSupports::load(&corpus_artifact).expect("a v4 artifact still loads");
        assert!(!loaded.bin_means_measured());

        // The sidecar the first promotion would have written beside the weights.
        let sidecar = dir.join("weights").join("pretrain_best_diag896.supports.300.json");
        let err = loaded.save(&sidecar).expect_err(
            "a support with no fitted moments must be unwritable: the only schema this build \
             writes promises them, so every file it could produce here is one the loader refuses",
        );
        let message = format!("{err:#}");
        assert!(
            message.contains(&sidecar.display().to_string())
                && message.contains("no fitted per-bin moments"),
            "the refusal must name the artifact it declined to write and why: {message}"
        );
        assert!(
            !sidecar.exists() && !sidecar.parent().expect("parent").exists(),
            "the refusal must precede every filesystem effect; an unloadable artifact must \
             never exist, not even briefly"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn dof_tensor(samples: &[BarDof]) -> Tensor {
        let flat: Vec<f32> = samples.iter().flat_map(|d| d.to_array()).collect();
        Tensor::from_slice(&flat).view([samples.len() as i64, BAR_DOF as i64])
    }

    /// The support must be a gapless, value-ordered tiling in `f32`, with strictly
    /// positive continuous widths and exactly-zero atom widths.
    fn assert_tiling(supports: &BarSupports, dof: usize) {
        let bins = NUM_BAR_BINS as usize;
        let (lo, hi) = (supports.lower_bounds(dof), supports.upper_bounds(dof));
        assert_eq!(lo.len(), bins);
        assert_eq!(hi.len(), bins);
        let mut continuous = 0usize;
        for j in 0..bins {
            assert!(
                lo[j].is_finite() && hi[j].is_finite(),
                "DOF {} bin {j} bound is not finite",
                BAR_DOF_NAMES[dof]
            );
            let width = supports.widths(dof)[j];
            assert_eq!(width, (hi[j] - lo[j]) as f32 as f64);
            if width > 0.0 {
                continuous += 1;
                assert!(
                    (hi[j] as f32) > lo[j] as f32,
                    "DOF {} bin {j} collapsed in f32",
                    BAR_DOF_NAMES[dof]
                );
            } else {
                assert_eq!(lo[j], hi[j], "DOF {} bin {j} has negative width", BAR_DOF_NAMES[dof]);
            }
            if j + 1 < bins {
                assert_eq!(
                    hi[j], lo[j + 1],
                    "DOF {} leaves a gap between bins {j} and {}",
                    BAR_DOF_NAMES[dof],
                    j + 1
                );
            }
        }
        assert_eq!(
            bins - continuous,
            supports.atoms(dof).len(),
            "DOF {} atom count disagrees with the zero-width bins",
            BAR_DOF_NAMES[dof]
        );
        for atom in supports.atoms(dof) {
            assert_eq!(lo[atom.bin], atom.value);
            assert_eq!(hi[atom.bin], atom.value);
        }
    }

    /// Atom-heavy sample shaped like extended-hours data: flat bars, and closes or
    /// opens sitting exactly on a bar extreme.
    fn atom_heavy_samples(count: usize, seed: u64) -> Vec<BarDof> {
        let mut rng = Rng::new(seed);
        (0..count)
            .map(|i| {
                let mut dof = synthetic_dof(&mut rng);
                match i % 5 {
                    0 => {
                        dof.s = 0.0;
                        dof.u = 0.5;
                        dof.v = 0.5;
                    }
                    1 => {
                        dof.u = 0.0;
                        dof.v = 1.0;
                    }
                    2 => {
                        dof.u = 1.0;
                        dof.v = 0.0;
                    }
                    3 => dof.w = 0.0,
                    _ => {}
                }
                dof
            })
            .collect()
    }

    /// The conditional reference must remove EXACTLY the deterministic mass and nothing
    /// else. `encode_dof` forces `u = v = 0.5` on a flat bar, so the flat-bar rate `m` is the
    /// whole free lunch, and the exact decomposition `H(p) = H_b(m) + (1 - m) * H(r)` says
    /// the identity gain must come out as the binary entropy of `m` — derived here
    /// independently of how `encoding_identity_nats` computes it.
    ///
    /// Checked under every scoring rule. The flat bin is an ATOM and therefore carries no
    /// width, so the density rule's measure term cancels exactly between the two references
    /// and the identity gain must come out mode-independent.
    #[test]
    fn conditional_marginal_removes_exactly_the_flat_bar_mass() {
        let supports = BarSupports::fit(&atom_heavy_samples(40_000, 0xC0FF_EE01));
        let mut identity_by_mode = Vec::new();
        for scoring in BarScoring::ALL {
            let unconditional = supports.marginal_nll_dof(scoring);
            let conditional = supports.marginal_nll_dof_conditional(scoring);

            let mut expected_identity = 0.0;
            for dof in [DOF_U, DOF_V] {
                let flat = supports
                    .atoms(dof)
                    .iter()
                    .find(|atom| atom.value == 0.5)
                    .expect("the flat-bar shape atom is mandated");
                let m = supports.reference_row(dof, scoring)[flat.bin];
                assert!(
                    (0.15..0.25).contains(&m),
                    "one bar in five is flat, so the 0.5 atom should hold ~0.2, got {m} \
                     under {scoring}"
                );
                let live = 1.0 - m;
                let binary = -m * m.ln() - live * live.ln();
                let recomposed = binary + live * conditional[dof];
                assert!(
                    (recomposed - unconditional[dof]).abs() < 1e-9,
                    "DOF {} conditional entropy is not the exact residual under {scoring}: \
                     {recomposed} != {}",
                    BAR_DOF_NAMES[dof],
                    unconditional[dof]
                );
                expected_identity += binary;
            }

            // Nothing in the encoding determines r, s or w, so their references are
            // untouched — including the density rule's measure term.
            for dof in [DOF_R, DOF_S, DOF_W] {
                assert_eq!(
                    conditional[dof], unconditional[dof],
                    "DOF {} under {scoring}",
                    BAR_DOF_NAMES[dof]
                );
            }
            assert!(
                (supports.encoding_identity_nats(scoring) - expected_identity).abs() < 1e-9,
                "identity gain {} != binary entropy sum {expected_identity} under {scoring}",
                supports.encoding_identity_nats(scoring)
            );
            assert!(
                (supports.marginal_plus_identity_nll_bar(scoring)
                    - (supports.marginal_nll_bar(scoring) - expected_identity))
                    .abs()
                    < 1e-12
            );
            assert!(
                supports.marginal_plus_identity_nll_bar(scoring)
                    < supports.marginal_nll_bar(scoring)
            );
            identity_by_mode.push(expected_identity);
        }
        // The free lunch is a property of the ENCODING, not of the scoring rule: the flat
        // bin is an ATOM, so no width enters either side of the difference and the density
        // rule's measure term cancels exactly. `hard` and `density` share the bin histogram
        // as their reference row and must therefore agree BIT for bit.
        let [smoothed, hard, density] = <[f64; 3]>::try_from(identity_by_mode.as_slice())
            .expect("one identity gain per scoring rule");
        assert_eq!(hard, density, "{identity_by_mode:?}");
        // `smoothed` reads its flat-bin mass off the renormalized mean smoothed target
        // rather than off the histogram, and `normalize_rows` divides by a sum that is one
        // only to `from_bins`'s 1e-6 validation, so it agrees to that precision and no
        // further. Anything larger would mean the identity depends on the rule.
        assert!(
            (smoothed - hard).abs() < 1e-6,
            "the encoding identity moved with the scoring rule: {identity_by_mode:?}"
        );
    }

    /// Provenance survives the JSON round trip, and an artifact written before provenance
    /// existed still loads — reporting `None` rather than a fabricated match.
    #[test]
    fn supports_provenance_round_trips_and_legacy_files_report_none() {
        let dir = std::env::temp_dir().join(format!(
            "trading_bot_0_supports_prov_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("bar_supports.300.json");

        let provenance = BarSupportsProvenance {
            corpus_fingerprint: "a".repeat(64),
            split_bounds: (1_700_000_000_000, 1_710_000_000_000),
            sample_count: 4_000_000,
            fitted_utc: "2026-08-15T00:00:00Z".to_owned(),
        };
        let fitted =
            synthetic_supports(20_000, 0x9A9A).with_provenance(provenance.clone());
        fitted.save(&path).expect("save");
        let reloaded = BarSupports::load(&path).expect("load");
        assert_eq!(reloaded.provenance(), Some(&provenance));

        // A v3 artifact carries no provenance field at all and must still load.
        let mut raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("parse");
        raw["format_version"] = serde_json::json!(BAR_SUPPORTS_LEGACY_VERSION);
        raw.as_object_mut().expect("object").remove("provenance");
        std::fs::write(&path, serde_json::to_vec(&raw).expect("serialize")).expect("write");
        let legacy = BarSupports::load(&path).expect("legacy load");
        assert_eq!(legacy.provenance(), None);

        // Anything else is refused outright.
        raw["format_version"] = serde_json::json!(2);
        std::fs::write(&path, serde_json::to_vec(&raw).expect("serialize")).expect("write");
        assert!(BarSupports::load(&path).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dof_round_trip_is_exact() {
        let cases = [
            (191.32_f32, bar(191.40, 192.05, 190.88, 191.77, 812_344.0), 640_000.0_f32),
            (4.21, bar(4.19, 4.44, 4.05, 4.40, 1_912.0), 3_000.0),
            (1_284.5, bar(1_284.5, 1_284.5, 1_270.0, 1_275.25, 88.0), 91.5),
            (0.0421, bar(0.0430, 0.0455, 0.0412, 0.0418, 55_000.0), 61_233.0),
        ];
        for (prev_close, original, ema) in cases {
            let dof = encode_dof(prev_close, &original, ema);
            assert!(dof.is_finite(), "dof must be finite: {dof:?}");
            assert!((0.0..=1.0).contains(&dof.u) && (0.0..=1.0).contains(&dof.v));
            assert!(dof.s >= 0.0);
            let back = decode_dof(prev_close, &dof, ema);
            for (field, actual, expected) in [
                ("open", back.open, original.open),
                ("high", back.high, original.high),
                ("low", back.low, original.low),
                ("close", back.close, original.close),
                ("volume", back.volume, original.volume),
            ] {
                assert!(
                    relative(actual, expected) < 1e-5,
                    "{field} round trip: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn decode_preserves_ohlc_ordering() {
        let mut rng = Rng::new(0xC0FFEE);
        let pathological = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -1.0,
            0.0,
            1e30,
            -1e30,
        ];
        for i in 0..20_000 {
            let dof = if i % 7 == 0 {
                let pick = |rng: &mut Rng| pathological[(rng.next_u64() % 7) as usize];
                BarDof {
                    r: pick(&mut rng),
                    s: pick(&mut rng),
                    u: pick(&mut rng),
                    v: pick(&mut rng),
                    w: pick(&mut rng),
                }
            } else {
                BarDof {
                    r: (0.5 * rng.normal()) as f32,
                    s: (2.0 * rng.uniform() - 0.5) as f32,
                    u: (1.4 * rng.uniform() - 0.2) as f32,
                    v: (1.4 * rng.uniform() - 0.2) as f32,
                    w: (3.0 * rng.normal()) as f32,
                }
            };
            let prev_close = if i % 11 == 0 {
                pathological[(rng.next_u64() % 7) as usize]
            } else {
                (0.01 + 900.0 * rng.uniform()) as f32
            };
            let ema = if i % 13 == 0 {
                pathological[(rng.next_u64() % 7) as usize]
            } else {
                (1.0 + 1e6 * rng.uniform()) as f32
            };
            let out = decode_dof(prev_close, &dof, ema);
            let (o, h, l, c, vol, wap) = (
                out.open, out.high, out.low, out.close, out.volume, out.vwap,
            );
            for (field, value) in [
                ("open", o),
                ("high", h),
                ("low", l),
                ("close", c),
                ("volume", vol),
                ("vwap", wap),
            ] {
                assert!(value.is_finite(), "{field} must be finite, got {value}");
            }
            assert!(o > 0.0 && h > 0.0 && l > 0.0 && c > 0.0 && vol >= 0.0);
            assert!(l <= o.min(c), "low {l} above min(open,close) {}", o.min(c));
            assert!(h >= o.max(c), "high {h} below max(open,close) {}", o.max(c));
            assert!(l <= wap && wap <= h, "vwap {wap} outside [{l}, {h}]");
        }
    }

    #[test]
    fn flat_bar_maps_to_mid_positions() {
        let flat = bar(150.0, 150.0, 150.0, 150.0, 1_000.0);
        let dof = encode_dof(149.5, &flat, 900.0);
        assert_eq!(dof.s, 0.0);
        assert_eq!(dof.u, 0.5);
        assert_eq!(dof.v, 0.5);
        let back = decode_dof(149.5, &dof, 900.0);
        for value in [back.open, back.high, back.low, back.close] {
            assert!(relative(value, 150.0) < 1e-5, "flat bar price {value}");
        }
        assert!(relative(back.volume, 1_000.0) < 1e-5);

        // A degenerate range on a corrupt bar must not divide by zero either.
        let corrupt = bar(f32::NAN, 0.0, -3.0, 88.0, 0.0);
        let dof = encode_dof(f32::NAN, &corrupt, f32::NAN);
        assert!(dof.is_finite(), "{dof:?}");
        assert_eq!(dof.s, 0.0);
        assert_eq!(dof.u, 0.5);
    }

    /// Continuous bins carry equal empirical mass within each inter-atom segment,
    /// and atom bins carry exactly the mass recorded at fit time. Asserted on both a
    /// tie-free sample and an atom-heavy one.
    #[test]
    fn fitted_supports_are_equal_mass() {
        for (label, samples) in [
            ("continuous", {
                let mut rng = Rng::new(0x5EED);
                (0..40_000).map(|_| synthetic_dof(&mut rng)).collect()
            }),
            ("atom heavy", atom_heavy_samples(40_000, 0x5EEE)),
        ] {
            let count: usize = samples.len();
            let supports = BarSupports::fit(&samples);
            for dof in 0..BAR_DOF {
                assert_tiling(&supports, dof);
                let mut counts = vec![0usize; NUM_BAR_BINS as usize];
                for sample in &samples {
                    counts[supports.bin_of(dof, sample.to_array()[dof] as f64)] += 1;
                }
                assert_eq!(counts.iter().sum::<usize>(), count);

                // Atom bins hold exactly the mass the fit recorded.
                for atom in supports.atoms(dof) {
                    let observed = counts[atom.bin] as f64 / count as f64;
                    assert!(
                        (observed - atom.mass).abs() < 1e-9,
                        "{label} DOF {}: atom {} holds {observed} but recorded {}",
                        BAR_DOF_NAMES[dof],
                        atom.value,
                        atom.mass
                    );
                }

                // Continuous bins are equal-mass inside each maximal run between
                // atoms. Runs shorter than 8 bins exist only because every segment is
                // guaranteed at least one bin, so they carry no equal-mass claim.
                //
                // The band is a scale-free 5 standard deviations of the multinomial
                // count, not a fixed ratio: a fixed +-20% is under 3 sigma once a bin
                // holds only a couple of hundred samples, which flakes, while 5 sigma
                // still catches any SYSTEMATIC error (a collapsed or empty bin lands
                // tens of sigma out).
                for run in continuous_runs(&supports, dof) {
                    if run.len() < 8 {
                        continue;
                    }
                    let total: usize = run.iter().map(|&bin| counts[bin]).sum();
                    let share = total as f64 / run.len() as f64;
                    assert!(share > 0.0, "{label} DOF {} run is empty", BAR_DOF_NAMES[dof]);
                    let tolerance = 5.0 * share.sqrt().max(1.0);
                    for &bin in &run {
                        let deviation = counts[bin] as f64 - share;
                        assert!(
                            deviation.abs() <= tolerance,
                            "{label} DOF {} bin {bin} holds {} samples, {:.1} off its \
                             segment's equal-mass share of {share:.1} (tolerance {tolerance:.1})",
                            BAR_DOF_NAMES[dof],
                            counts[bin],
                            deviation
                        );
                    }
                }
            }
        }
    }

    /// Maximal runs of consecutive continuous (positive-width) bins.
    fn continuous_runs(supports: &BarSupports, dof: usize) -> Vec<Vec<usize>> {
        let mut runs: Vec<Vec<usize>> = Vec::new();
        let mut current: Vec<usize> = Vec::new();
        for (bin, &width) in supports.widths(dof).iter().enumerate() {
            if width > 0.0 {
                current.push(bin);
            } else if !current.is_empty() {
                runs.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            runs.push(current);
        }
        runs
    }

    /// Real bars pile mass on discrete `s`/`u`/`v` values. Each of those must own a
    /// dedicated zero-width bin, the tiling must stay exact in the `f32` precision
    /// the tensor path uses, and every lookup path must agree.
    #[test]
    fn atoms_own_dedicated_bins_and_every_lookup_agrees() {
        let samples = atom_heavy_samples(40_000, 0x7A11ED);
        let supports = BarSupports::fit(&samples);

        for dof in 0..BAR_DOF {
            assert_tiling(&supports, dof);
        }
        // The mandated atoms are present on exactly the DOF that can carry them.
        for (dof, mandated) in [(DOF_S, &[0.0f32][..]), (DOF_U, &[0.0, 0.5, 1.0]), (DOF_V, &[0.0, 0.5, 1.0])] {
            for &value in mandated {
                assert!(
                    supports.atoms(dof).iter().any(|a| a.value == value as f64),
                    "DOF {} is missing its mandated atom {value}",
                    BAR_DOF_NAMES[dof]
                );
            }
        }
        // 20% of the sample is a flat bar, so s == 0 and u == v == 0.5 each hold
        // about a fifth of the mass, and u == 0 / u == 1 another fifth each.
        let atom_mass = |dof: usize, value: f64| {
            supports.atoms(dof)
                .iter()
                .find(|a| a.value == value)
                .map(|a| a.mass)
                .unwrap_or(0.0)
        };
        assert!((atom_mass(DOF_S, 0.0) - 0.2).abs() < 0.01, "{}", atom_mass(DOF_S, 0.0));
        assert!((atom_mass(DOF_U, 0.5) - 0.2).abs() < 0.01, "{}", atom_mass(DOF_U, 0.5));
        assert!((atom_mass(DOF_U, 0.0) - 0.2).abs() < 0.01, "{}", atom_mass(DOF_U, 0.0));
        assert!((atom_mass(DOF_U, 1.0) - 0.2).abs() < 0.01, "{}", atom_mass(DOF_U, 1.0));

        // `bin_of` (host), `bin_ids`/`locate` (tensor) and `encode_targets` must all
        // name the same bin for every observed value, atoms included.
        let probe = &samples[..4_096];
        let values = dof_tensor(probe);
        let (index, position, is_atom) = supports.locate(&supports.prepare(&values));
        let ids = supports.bin_ids(&values);
        let encoded = supports.encode_targets(&values);
        for (row, sample) in probe.iter().enumerate() {
            for dof in 0..BAR_DOF {
                let value = sample.to_array()[dof];
                let host = supports.bin_of(dof, value as f64) as i64;
                let located = index.int64_value(&[row as i64, dof as i64, 0]);
                let public = ids.int64_value(&[row as i64, dof as i64]);
                assert_eq!(
                    host, located,
                    "DOF {} value {value} binned {host} on the host and {located} in locate",
                    BAR_DOF_NAMES[dof]
                );
                assert_eq!(host, public, "bin_ids disagrees with bin_of");
            }
        }
        assert!(position.min().double_value(&[]) >= 0.0);
        assert!(position.max().double_value(&[]) <= 1.0);
        let sums = encoded.sum_dim_intlist([-1].as_slice(), false, Kind::Double);
        assert!((sums - 1.0).abs().max().double_value(&[]) < 1e-5);

        // An atom observation is labelled as an exact point mass, and no observation
        // ever puts mass on an atom bin it did not land on.
        let containing = encoded.gather(-1, &index, false).squeeze_dim(-1);
        let atom_rows = is_atom.squeeze_dim(-1);
        let atom_count = atom_rows.sum(Kind::Double).double_value(&[]);
        assert!(atom_count > 0.0, "the atom-heavy sample produced no atom rows");
        let atom_mass_on_target = (&containing * &atom_rows).sum(Kind::Double).double_value(&[]);
        assert!(
            (atom_mass_on_target - atom_count).abs() < 1e-4,
            "atom labels are not exact one-hots: {atom_mass_on_target} over {atom_count} rows"
        );
        let atom_bins: Vec<i64> = supports.atoms(DOF_U).iter().map(|a| a.bin as i64).collect();
        let spill = encoded
            .select(1, DOF_U as i64)
            .index_select(1, &Tensor::from_slice(&atom_bins))
            .sum_dim_intlist([-1].as_slice(), false, Kind::Double);
        let continuous_rows = atom_rows.select(1, DOF_U as i64).neg() + 1.0;
        let leaked = (spill * continuous_rows).max().double_value(&[]);
        assert!(
            leaked < 1e-12,
            "a continuous observation leaked {leaked} onto u's atom bins"
        );

        // Away from atoms the containing bin still holds a share close to the peak.
        let peak = encoded.max_dim(-1, false).0;
        let ratio = (containing / peak).min().double_value(&[]);
        assert!(
            ratio > 0.5,
            "smoothed mass on the containing bin fell to {ratio} of the peak"
        );

        // A degenerate DOF (one repeated value) must still fit a usable support.
        let constant = vec![BarDof::default(); 1_000];
        let degenerate = BarSupports::fit(&constant);
        for dof in 0..BAR_DOF {
            assert_tiling(&degenerate, dof);
        }
        let encoded = degenerate.encode_targets(&dof_tensor(&constant[..8]));
        let sums = encoded.sum_dim_intlist([-1].as_slice(), false, Kind::Double);
        assert!((sums - 1.0).abs().max().double_value(&[]) < 1e-5);
    }

    /// The marginal reference must be exactly the loss an optimal MARGINAL head
    /// achieves under the soft-target objective we report, must sit strictly between
    /// the hard-histogram entropy and the uniform baseline, and must survive
    /// persistence so a report can read it off a checkpoint's supports file.
    #[test]
    fn marginal_reference_is_the_loss_a_marginal_head_achieves() {
        // 40k rows keeps the whole sample under `MARGINAL_ESTIMATE_ROWS`, so the
        // fitted `q*` averages over exactly these rows and the independent
        // recomputation below must agree to floating-point precision.
        let samples = atom_heavy_samples(40_000, 0x4D41_5247);
        let supports = BarSupports::fit(&samples);

        // `q*` must literally be the mean smoothed target, not some other nearby
        // distribution. Recomputed here through `encode_targets` in one f64 reduction,
        // whereas the fit splits atoms out exactly and accumulates the continuous half
        // in f32 across chunks, so they agree to roughly f32 epsilon rather than
        // bit-exactly. 1e-6 is still three orders of magnitude tighter than the
        // ~1.5e-3 gap a hard-histogram substitution would open, which is the mutation
        // this assertion exists to catch.
        let mean_target = supports
            .encode_targets(&dof_tensor(&samples))
            .to_kind(Kind::Double)
            .mean_dim([0i64].as_slice(), false, Kind::Double);
        for dof in 0..BAR_DOF {
            for bin in 0..NUM_BAR_BINS {
                let want = mean_target.double_value(&[dof as i64, bin]);
                let got = supports.smoothed_marginal(dof)[bin as usize];
                assert!(
                    (got - want).abs() < 1e-6,
                    "DOF {} bin {bin}: q* is {got} but the mean smoothed target is {want}",
                    BAR_DOF_NAMES[dof]
                );
            }
        }

        // `q*` is the mean SMOOTHED target, so this whole test is a statement about
        // `BarScoring::Smoothed`; the other two rules take the bin histogram as their
        // reference row and are pinned by `the_reference_row_is_the_argmin_of_its_rule`.
        let scoring = BarScoring::Smoothed;
        let uniform = supports.uniform_nll_bar(scoring);
        let marginal = supports.marginal_nll_bar(scoring);
        let per_dof = supports.marginal_nll_dof(scoring);
        assert!((uniform - BAR_DOF as f64 * (NUM_BAR_BINS as f64).ln()).abs() < 1e-12);
        assert!((marginal - per_dof.iter().sum::<f64>()).abs() < 1e-12);
        assert!(marginal < uniform - 0.5, "marginal {marginal} must beat uniform {uniform}");

        // No ordering is claimed against the hard-histogram entropy: that is the
        // optimum of a different objective and the two land within ~0.002 nats.

        // The claim under test: a constant q* head reproduces exactly this number
        // through the real reporting path.
        let rows = 8_192usize;
        let target = dof_tensor(&samples[..rows]);
        // A head whose prediction is a fixed per-DOF distribution, broadcast over rows.
        let constant = |per_dof: [&[f64]; BAR_DOF]| {
            let mut flat = Vec::with_capacity(BAR_DOF * NUM_BAR_BINS as usize);
            for row in per_dof {
                flat.extend(row.iter().map(|p| (p.max(1e-30) as f32).ln()));
            }
            Tensor::from_slice(&flat)
                .view([1, BAR_DOF as i64, NUM_BAR_BINS])
                .expand([rows as i64, BAR_DOF as i64, NUM_BAR_BINS], false)
                .contiguous()
        };
        let q_star: [&[f64]; BAR_DOF] = std::array::from_fn(|d| supports.smoothed_marginal(d));
        let histogram: [&[f64]; BAR_DOF] = std::array::from_fn(|d| supports.bin_masses(d));
        let encoded = supports.targets(&target, scoring);
        let (achieved, achieved_dof) = bar_nll_from_logits(&constant(q_star), &encoded);
        let achieved = achieved.double_value(&[]);
        assert!(
            (achieved - marginal).abs() < 0.02,
            "a q* head achieved {achieved} but the reference claims {marginal}"
        );
        for dof in 0..BAR_DOF {
            let got = achieved_dof.double_value(&[dof as i64]);
            assert!(
                (got - per_dof[dof]).abs() < 0.02,
                "DOF {} achieved {got} vs reference {}",
                BAR_DOF_NAMES[dof],
                per_dof[dof]
            );
        }
        // The property that makes q* a valid reference: it is the ARGMIN over fixed
        // predictions, so no other constant head can beat it. Checked against the hard
        // histogram, a uniform head, and a deliberately skewed q*.
        let skewed: [Vec<f64>; BAR_DOF] = std::array::from_fn(|dof| {
            let row = supports.smoothed_marginal(dof);
            let bumped: Vec<f64> = row
                .iter()
                .enumerate()
                .map(|(bin, p)| {
                    (p + if bin < NUM_BAR_BINS as usize / 2 { 1e-3 } else { -1e-3 }).max(0.0)
                })
                .collect();
            let total: f64 = bumped.iter().sum();
            bumped.into_iter().map(|p| p / total).collect()
        });
        let uniform_row = vec![1.0 / NUM_BAR_BINS as f64; NUM_BAR_BINS as usize];
        for (label, rows_in) in [
            ("hard histogram", histogram),
            ("uniform", std::array::from_fn(|_| uniform_row.as_slice())),
            ("skewed q*", std::array::from_fn(|d| skewed[d].as_slice())),
        ] {
            let (other, _) = bar_nll_from_logits(&constant(rows_in), &encoded);
            assert!(
                other.double_value(&[]) >= achieved - 1e-4,
                "{label} scored {}, beating the supposed optimum {achieved}",
                other.double_value(&[])
            );
        }

        for dof in 0..BAR_DOF {
            for atom in supports.atoms(dof) {
                assert_eq!(atom.mass, supports.bin_masses(dof)[atom.bin]);
            }
        }

        let dir = std::env::temp_dir().join(format!("bar_dist_marginal_{}", std::process::id()));
        let path = dir.join("supports.json");
        supports.save(&path).expect("save");
        let loaded = BarSupports::load(&path).expect("load");
        assert!((loaded.marginal_nll_bar(scoring) - marginal).abs() < 1e-9);
        for dof in 0..BAR_DOF {
            for (a, b) in supports.bin_masses(dof).iter().zip(loaded.bin_masses(dof)) {
                assert!((a - b).abs() < 1e-12);
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Bad ticks must not define the support bounds, but must still be representable
    /// and must still score: the outermost bins are open-ended catch-alls.
    #[test]
    fn extreme_ticks_are_clipped_from_the_bounds_but_still_score() {
        let _torch_rng_guard = test_rng::shared();
        let mut rng = Rng::new(0xC11C_1A9E);
        let bulk = 100_000usize;
        let mut samples: Vec<BarDof> = (0..bulk).map(|_| synthetic_dof(&mut rng)).collect();
        // A handful of 7000x bars, the shape the spliced ticker series produced.
        for i in 0..8 {
            samples.push(BarDof {
                r: if i % 2 == 0 { 8.85 } else { -8.87 },
                s: 8.85,
                u: 0.5,
                v: 0.5,
                w: if i % 2 == 0 { 8.3 } else { -10.6 },
            });
        }
        let supports = BarSupports::fit(&samples);
        for dof in 0..BAR_DOF {
            assert_tiling(&supports, dof);
        }

        // r is clipped well inside the outliers: 8 rows in 100k is 8e-5, below the
        // 1e-4 clip, so no outlier survives as a bound.
        let r_lo = supports.lower_bounds(DOF_R)[0];
        let r_hi = *supports.upper_bounds(DOF_R).last().expect("bounds");
        assert!(
            r_lo > -1.0 && r_hi < 1.0,
            "r support [{r_lo}, {r_hi}] still contains the bad ticks"
        );
        let s_hi = *supports.upper_bounds(DOF_S).last().expect("bounds");
        assert!(s_hi < 1.0, "s support upper bound {s_hi} still at the bad tick");

        // Outer bin centers decode to the clipped bound, not to an invented extreme.
        for dof in 0..BAR_DOF {
            let centers = supports.centers(dof);
            assert_eq!(centers[0], supports.lower_bounds(dof)[0]);
            assert_eq!(
                centers[NUM_BAR_BINS as usize - 1],
                supports.upper_bounds(dof)[NUM_BAR_BINS as usize - 1]
            );
            assert!(centers.windows(2).all(|p| p[1] > p[0]), "DOF {} centers", BAR_DOF_NAMES[dof]);
        }

        // The outliers still bin (into the catch-all bins) and still score finitely.
        let extremes = dof_tensor(&samples[bulk..]);
        let ids = supports.bin_ids(&extremes);
        for row in 0..8i64 {
            let r_bin = ids.int64_value(&[row, DOF_R as i64]);
            assert!(
                r_bin == 0 || r_bin == NUM_BAR_BINS - 1,
                "an out-of-range r landed in interior bin {r_bin}"
            );
            assert_eq!(r_bin, supports.bin_of(DOF_R, samples[bulk + row as usize].r as f64) as i64);
        }
        let targets = supports.targets(&extremes, BarScoring::Smoothed);
        let encoded = targets.targets().shallow_clone();
        let sums = encoded.sum_dim_intlist([-1].as_slice(), false, Kind::Double);
        assert!((sums - 1.0).abs().max().double_value(&[]) < 1e-5);
        let uniform = Tensor::zeros(encoded.size().as_slice(), (Kind::Float, Device::Cpu));
        let (nll, _) = bar_nll_from_logits(&uniform, &targets);
        assert!(
            (nll.double_value(&[]) - supports.uniform_nll_bar(BarScoring::Smoothed)).abs() < 1e-3,
            "a clipped observation must still score exactly like any other"
        );

        // Sampling and decoding can never produce the absurd move any more.
        let sharp = encoded.clamp_min(1e-30).log() * 4.0;
        let drawn = supports.sample(&sharp, 1.0);
        assert!(drawn.select(1, DOF_R as i64).abs().max().double_value(&[]) < 1.0);
        assert!(supports.expectation(&sharp).select(1, DOF_R as i64).abs().max().double_value(&[]) < 1.0);
    }

    /// A head that predicts the empirical distribution exactly attains its entropy,
    /// which is the property that makes `nll_bar` comparable across ablations. With
    /// a 30% atom the entropy is strictly below `ln(NUM_BAR_BINS)`, and the atom
    /// contributes `-0.3 * ln(0.3)` rather than being smeared over degenerate bins.
    #[test]
    fn atom_mass_is_recovered_exactly() {
        let mut rng = Rng::new(0x0A70_4321);
        let count = 100_000usize;
        let samples: Vec<BarDof> = (0..count)
            .map(|i| {
                let mut dof = synthetic_dof(&mut rng);
                if i % 10 < 3 {
                    dof.u = 0.5;
                }
                dof
            })
            .collect();
        let supports = BarSupports::fit(&samples);
        let atom = supports
            .atoms(DOF_U)
            .iter()
            .find(|a| a.value == 0.5)
            .copied()
            .expect("u = 0.5 must be an atom");
        assert!(
            (atom.mass - 0.3).abs() < 0.005,
            "u = 0.5 recovered mass {}",
            atom.mass
        );
        assert_eq!(supports.widths(DOF_U)[atom.bin], 0.0);

        // Empirical bin distribution of u, then a head that predicts exactly it.
        let mut counts = vec![0f64; NUM_BAR_BINS as usize];
        for sample in &samples {
            counts[supports.bin_of(DOF_U, sample.u as f64)] += 1.0;
        }
        assert!(
            (counts[atom.bin] / count as f64 - 0.3).abs() < 0.005,
            "the atom bin holds {} of the sample",
            counts[atom.bin] / count as f64
        );
        let probs: Vec<f32> = counts.iter().map(|c| (c / count as f64) as f32).collect();
        let entropy: f64 = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -(p as f64) * (p as f64).ln())
            .sum();
        assert!(
            entropy < (NUM_BAR_BINS as f64).ln() - 0.1,
            "a 30% atom must cost less than a uniform categorical: {entropy}"
        );

        let rows = 4_096usize;
        let target = dof_tensor(&samples[..rows]);
        let logits = empirical_logits(&supports, &samples, rows);
        let (_, per_dof) = bar_nll_from_logits(
            &logits,
            &supports.targets(&target, BarScoring::Smoothed),
        );
        let achieved = per_dof.double_value(&[DOF_U as i64]);
        // The soft targets for continuous observations spread mass over neighbouring
        // bins, which can only raise the cross entropy above the hard-label entropy.
        assert!(
            achieved >= entropy - 0.02,
            "u cross entropy {achieved} undercut the empirical entropy {entropy}"
        );
        assert!(
            achieved < (NUM_BAR_BINS as f64).ln(),
            "u cross entropy {achieved} must beat the uniform baseline"
        );
    }

    /// `pit` must be bit-reproducible for a fixed seed and must actually randomize
    /// across the atom's probability interval, otherwise the histogram spikes.
    #[test]
    fn randomized_pit_is_seeded_and_spreads_atom_mass() {
        let samples = atom_heavy_samples(60_000, 0x5EEDED);
        let supports = BarSupports::fit(&samples);
        let rows = 20_000usize;
        let target = dof_tensor(&samples[..rows]);
        let logits = empirical_logits(&supports, &samples, rows);

        let a = bar_pit_from_logits(&logits, &target, &supports, 42);
        let b = bar_pit_from_logits(&logits, &target, &supports, 42);
        let c = bar_pit_from_logits(&logits, &target, &supports, 43);
        assert_eq!(
            (a - &b).abs().max().double_value(&[]),
            0.0,
            "a fixed seed must reproduce the histogram bit for bit"
        );
        assert!(
            (&b - &c).abs().max().double_value(&[]) > 0.0,
            "a different seed must move the randomized draws"
        );
        assert!(b.min().double_value(&[]) >= 0.0 && b.max().double_value(&[]) <= 1.0);

        // u = 0.5 holds ~20% of the mass. Without randomization every one of those
        // observations would land on a single PIT value; with it they must spread
        // across the atom's whole probability interval.
        let u_pit = b.select(1, DOF_U as i64);
        let is_half = target
            .select(1, DOF_U as i64)
            .eq(0.5)
            .to_kind(Kind::Float);
        let hits = is_half.sum(Kind::Double).double_value(&[]);
        assert!(hits > 1_000.0, "expected many u = 0.5 rows, got {hits}");
        let selected = u_pit.masked_select(&is_half.to_kind(Kind::Bool));
        let spread = selected.max().double_value(&[]) - selected.min().double_value(&[]);
        assert!(
            spread > 0.1,
            "randomized PIT collapsed the u = 0.5 atom into a spike of width {spread}"
        );
    }

    #[test]
    fn smoothed_targets_are_normalized_and_local() {
        let supports = synthetic_supports(50_000, 0xA11CE);
        let mut rng = Rng::new(7);
        let mut samples: Vec<BarDof> = (0..256).map(|_| synthetic_dof(&mut rng)).collect();
        // Force out-of-support values to exercise the edge clamp.
        samples[0] = BarDof {
            r: -10.0,
            s: 50.0,
            u: -3.0,
            v: 4.0,
            w: -80.0,
        };
        let values = dof_tensor(&samples);
        let encoded = supports.encode_targets(&values);
        assert_eq!(
            encoded.size(),
            vec![samples.len() as i64, BAR_DOF as i64, NUM_BAR_BINS]
        );
        let sums = encoded.sum_dim_intlist([-1].as_slice(), false, Kind::Double);
        let deviation = (sums - 1.0).abs().max().double_value(&[]);
        assert!(deviation < 1e-5, "target rows must sum to 1, off by {deviation}");
        assert!(encoded.min().double_value(&[]) >= 0.0);

        // The mode must sit on the bin containing the value, and the mass must be
        // concentrated on a handful of bins around it (sigma = 0.75 bin widths).
        let argmax = encoded.argmax(-1, false);
        for (row, sample) in samples.iter().enumerate().skip(1) {
            for dof in 0..BAR_DOF {
                let expected = supports.bin_of(dof, sample.to_array()[dof] as f64) as i64;
                let actual = argmax.int64_value(&[row as i64, dof as i64]);
                assert!(
                    (actual - expected).abs() <= 1,
                    "DOF {} mode {actual} far from bin {expected}",
                    BAR_DOF_NAMES[dof]
                );
            }
        }
        let peak = encoded.max_dim(-1, false).0;
        assert!(
            peak.min().double_value(&[]) > 0.2,
            "smoothing must stay local, weakest peak {}",
            peak.min().double_value(&[])
        );
    }

    /// Label smoothing must be a perturbation at ONE return-space scale, not one scale per
    /// bin.
    ///
    /// The bins are equal-MASS quantile bins, so `0.75 * local_bin_width` is not a width —
    /// it is 128 different widths spanning three orders of magnitude, and the tail ones are
    /// wider than the entire central 95% of the support. That made the soft label deposit
    /// most of its mass nowhere near the observation, inflate the label's implied variance
    /// by 5546x in the mean, and bias the label's implied MEAN outward by hundreds of basis
    /// points in the second and second-from-last bins. `BarSupports::smooth` therefore caps
    /// the width at the per-DOF median, and this test is the statement of what the cap buys.
    ///
    /// Three assertions, all in RETURN units against the per-DOF CONSTANT
    /// [`BarSupports::smooth_sigma_cap`], because a bound stated relative to the local
    /// sigma is vacuous exactly where the bug was — the sigma blew up with the bin.
    #[test]
    fn label_smoothing_is_local_in_return_space() {
        let _torch_rng_guard = test_rng::shared();
        let supports = heavy_tailed_supports(200_000, 0x5EED_1A7E);

        // Probe every bin at five interior positions, edges included, so a kernel that
        // misbehaves only near a bin boundary cannot hide between midpoints.
        const POSITIONS: [f64; 5] = [0.02, 0.25, 0.5, 0.75, 0.98];
        let bins = NUM_BAR_BINS as usize;
        let rows = bins * POSITIONS.len();
        let mut probes = vec![0f32; rows * BAR_DOF];
        for dof in 0..BAR_DOF {
            let (lo, hi) = (supports.lower_bounds(dof), supports.upper_bounds(dof));
            for bin in 0..bins {
                for (slot, position) in POSITIONS.iter().enumerate() {
                    let row = bin * POSITIONS.len() + slot;
                    probes[row * BAR_DOF + dof] =
                        (lo[bin] + (hi[bin] - lo[bin]) * position) as f32;
                }
            }
        }
        let probe_t = Tensor::from_slice(&probes).view([rows as i64, BAR_DOF as i64]);

        // Which DOF actually exhibit the pathology. The cap can only bite where the
        // widest bin dwarfs the median one, and that is a property of the DOF's law, not
        // of the fix: `u` and `v` are near-uniform on [0, 1], so their equal-mass bins are
        // already near-equal-WIDTH above the median and the cap is close to inert on them.
        // Measured on the live 300s supports the same way: `r` spans 1234x from median to
        // widest and `s` 2540x, while `u` spans 1.75x and `v` 1.76x. So the correctness
        // assertions below run on all five, and the pre-fix COMPARISON runs only where
        // there is a bug to compare against.
        let pathological: [bool; BAR_DOF] = std::array::from_fn(|dof| {
            let widest = supports
                .widths(dof)
                .iter()
                .copied()
                .fold(0.0f64, f64::max);
            widest / (supports.smooth_sigma_cap(dof) / BAR_LABEL_SIGMA_RATIO) > 8.0
        });
        assert!(
            pathological.iter().filter(|&&p| p).count() >= 3,
            "only {} of {BAR_DOF} DOF have a width spread the cap can bite on, so this \
             fixture does not reproduce the live geometry and the comparison below is not \
             evidence: {pathological:?}",
            pathological.iter().filter(|&&p| p).count()
        );

        let encoded = supports.encode_targets(&probe_t);
        // The counterfactual: the SAME geometry with the cap lifted, which is exactly the
        // pre-fix kernel. Without this the test could not distinguish the fix from a
        // fixture whose bins happen to be uniform.
        let mut uncapped = supports.to_device(Device::Cpu);
        uncapped.cap_width_t =
            Tensor::full([BAR_DOF as i64, 1], f64::INFINITY, (Kind::Float, Device::Cpu));
        let before = uncapped.encode_targets(&probe_t);

        for dof in 0..BAR_DOF {
            let (lo, hi) = (supports.lower_bounds(dof), supports.upper_bounds(dof));
            let centers = supports.centers(dof);
            let sigma_cap = supports.smooth_sigma_cap(dof);
            let window = 6.0 * sigma_cap;
            let mut worst_far = 0.0f64;
            let mut worst_far_before = 0.0f64;
            let mut worst_bias = 0.0f64;
            let mut worst_bias_before = 0.0f64;
            for row in 0..rows {
                let x = probes[row * BAR_DOF + dof] as f64;
                let containing = supports.bin_of(dof, x);
                // Mass on bins that do not intersect `[x - window, x + window]`, plus the
                // width of the widest bin that DOES. The containing bin always intersects,
                // so `far` is purely spilled mass.
                let (mut far, mut far_before) = (0.0f64, 0.0f64);
                let (mut mean, mut second) = (0.0f64, 0.0f64);
                let mut mean_before = 0.0f64;
                let mut reachable_width = 0.0f64;
                for bin in 0..bins {
                    let mass = encoded.double_value(&[row as i64, dof as i64, bin as i64]);
                    let mass_before =
                        before.double_value(&[row as i64, dof as i64, bin as i64]);
                    mean += mass * centers[bin];
                    second += mass * centers[bin] * centers[bin];
                    mean_before += mass_before * centers[bin];
                    if lo[bin] > x + window || hi[bin] < x - window {
                        far += mass;
                        far_before += mass_before;
                    } else {
                        reachable_width = reachable_width.max(supports.widths(dof)[bin]);
                    }
                }
                // Tolerance is PER PROBE: the kernel window plus the widest bin the kernel
                // can REACH. The containing bin's width alone is not enough — a 2.7e-7
                // wide bin can sit next to one 80x wider, and a kernel straddling that
                // edge splits mass between two centers far apart in return space. That
                // displacement is the CENTER GRID's resolution, not the smoother's spread,
                // and it is present in the hard target too. A DOF-wide max width would go
                // the other way and let the 725 bps outermost bin excuse an arbitrarily bad
                // label in the 0.26 bps centre, which is the confusion the bug lived in.
                let tolerance = window + reachable_width;
                let sd = (second - mean * mean).max(0.0).sqrt();
                let bias = (mean - centers[containing]).abs();
                assert!(
                    sd < tolerance,
                    "DOF {} at x={x:.6e} (bin {containing}, reachable width \
                     {reachable_width:.3e}): the label's \
                     implied return-space sd is {sd:.3e}, past the {tolerance:.3e} the \
                     binning itself justifies",
                    BAR_DOF_NAMES[dof]
                );
                // Against the HARD target's implied mean, which is the containing bin's
                // center: the smoother must not move the label's mean by more than the
                // resolution already present in the binning.
                assert!(
                    bias < tolerance,
                    "DOF {} at x={x:.6e} (bin {containing}, reachable width \
                     {reachable_width:.3e}): the label's \
                     implied mean sits {bias:.3e} off the hard target's, past the \
                     {tolerance:.3e} the binning itself justifies",
                    BAR_DOF_NAMES[dof]
                );
                worst_far = worst_far.max(far);
                worst_far_before = worst_far_before.max(far_before);
                worst_bias = worst_bias.max(bias);
                worst_bias_before =
                    worst_bias_before.max((mean_before - centers[containing]).abs());
            }
            assert!(
                worst_far < 1e-6,
                "DOF {}: {worst_far:.3e} of the label mass landed further than 6 sigma \
                 ({window:.3e}) from the observation",
                BAR_DOF_NAMES[dof]
            );
            if !pathological[dof] {
                continue;
            }
            // Both halves asserted, so neither can pass vacuously: the capped kernel is
            // local and the uncapped one provably is not.
            assert!(
                worst_far_before > 0.05,
                "DOF {}: the UNCAPPED kernel spilled only {worst_far_before:.3e} past 6 \
                 sigma despite an 8x+ width spread, so the comparison is not evidence that \
                 the cap fixed anything",
                BAR_DOF_NAMES[dof]
            );
            assert!(
                worst_far_before > worst_far * 1e3,
                "DOF {}: capping cut the spill only from {worst_far_before:.3e} to \
                 {worst_far:.3e}",
                BAR_DOF_NAMES[dof]
            );
            // Item 4 of the audit: the uncapped kernel's damage was not only dispersion.
            // Because a narrow bin's neighbour can be hundreds of times wider, the
            // uncapped kernel leaked mass onto a distant center and pushed the label's
            // implied mean OUTWARD, away from the observation — measured at up to 465 bps
            // on the live `r` support. Capping must strictly reduce that, or the fix
            // addressed dispersion and left the mean bias in place.
            assert!(
                worst_bias_before > worst_bias * 4.0,
                "DOF {}: capping moved the worst label mean bias only from \
                 {worst_bias_before:.3e} to {worst_bias:.3e}; the mean axis is unfixed",
                BAR_DOF_NAMES[dof]
            );
        }
    }

    /// A uniform head pays `ln(NUM_BAR_BINS)` per factor under both DISCRETE rules, and
    /// exactly that plus the observation's own log bin width under the density rule. The
    /// second half is what makes the measure term auditable: it is an additive constant of
    /// the observation, never a property of the prediction.
    #[test]
    fn uniform_logits_cost_log_bins_per_dof() {
        let _torch_rng_guard = test_rng::shared();
        let supports = synthetic_supports(50_000, 0xB0B);
        let mut rng = Rng::new(11);
        let samples: Vec<BarDof> = (0..512).map(|_| synthetic_dof(&mut rng)).collect();
        let values = dof_tensor(&samples);
        let expected = BAR_DOF as f64 * (NUM_BAR_BINS as f64).ln();
        for scoring in [BarScoring::Smoothed, BarScoring::Hard] {
            let targets = supports.targets(&values, scoring);
            let uniform =
                Tensor::zeros(targets.targets().size().as_slice(), (Kind::Float, Device::Cpu));
            let (mean, per_dof) = bar_nll_from_logits(&uniform, &targets);
            assert!(
                (mean.double_value(&[]) - expected).abs() < 1e-3,
                "uniform NLL {} vs {expected} under {scoring}",
                mean.double_value(&[])
            );
            for dof in 0..BAR_DOF {
                let value = per_dof.double_value(&[dof as i64]);
                assert!(
                    (value - (NUM_BAR_BINS as f64).ln()).abs() < 1e-3,
                    "DOF {dof} {value} under {scoring}"
                );
            }
        }

        // The density rule differs from the hard rule by exactly the mean log bin width of
        // THESE observations, which no prediction can move.
        let density = supports.targets(&values, BarScoring::Density);
        let uniform =
            Tensor::zeros(density.targets().size().as_slice(), (Kind::Float, Device::Cpu));
        let (mean, _) = bar_nll_from_logits(&uniform, &density);
        let measure = density
            .log_measure()
            .expect("the density rule carries a measure term")
            .to_kind(Kind::Double)
            .sum_dim_intlist([-1].as_slice(), false, Kind::Double)
            .mean(Kind::Double)
            .double_value(&[]);
        assert!(
            (mean.double_value(&[]) - (expected + measure)).abs() < 1e-3,
            "density uniform NLL {} != {expected} + measure {measure}",
            mean.double_value(&[])
        );

        // A freshly built head is zero-initialized, so it starts exactly uniform.
        let vs = nn::VarStore::new(Device::Cpu);
        let head = BarEmissionHead::new(&vs.root(), 48);
        let h = Tensor::randn([4, 128, 48], (Kind::Float, Device::Cpu));
        let mut rng = Rng::new(12);
        let batch: Vec<BarDof> = (0..4 * 128).map(|_| synthetic_dof(&mut rng)).collect();
        let batch_dof = dof_tensor(&batch).view([4, 128, BAR_DOF as i64]);
        let (mean, _) = head.nll(&h, &batch_dof, &supports, BarScoring::Hard);
        assert!(
            (mean.double_value(&[]) - expected).abs() < 1e-3,
            "fresh head NLL {} vs {expected}",
            mean.double_value(&[])
        );
    }

    #[test]
    fn matching_the_smoothed_target_lowers_nll() {
        let supports = synthetic_supports(50_000, 0xD00D);
        let mut rng = Rng::new(13);
        let samples: Vec<BarDof> = (0..512).map(|_| synthetic_dof(&mut rng)).collect();
        let targets = supports.targets(&dof_tensor(&samples), BarScoring::Smoothed);
        let rows = targets.targets().shallow_clone();
        let uniform = Tensor::zeros(rows.size().as_slice(), (Kind::Float, Device::Cpu));
        let (uniform_nll, _) = bar_nll_from_logits(&uniform, &targets);
        let (matched_nll, matched_per_dof) =
            bar_nll_from_logits(&rows.clamp_min(1e-30).log(), &targets);

        let uniform_nll = uniform_nll.double_value(&[]);
        let matched_nll = matched_nll.double_value(&[]);
        assert!(
            matched_nll < uniform_nll - 1e-3,
            "matched NLL {matched_nll} must beat uniform {uniform_nll}"
        );
        // Matching the target attains its entropy exactly.
        let entropy = -(&rows * rows.clamp_min(1e-30).log()).sum_dim_intlist(
            [-1].as_slice(),
            false,
            Kind::Double,
        );
        let entropy_per_dof = entropy
            .reshape([-1, BAR_DOF as i64])
            .mean_dim([0i64].as_slice(), false, Kind::Double);
        for dof in 0..BAR_DOF {
            let got = matched_per_dof.double_value(&[dof as i64]);
            let want = entropy_per_dof.double_value(&[dof as i64]);
            assert!((got - want).abs() < 1e-4, "DOF {dof}: {got} vs entropy {want}");
        }
    }

    #[test]
    fn crps_vanishes_on_a_point_mass_at_truth() {
        let supports = synthetic_supports(50_000, 0xFEED);
        let rows = 300i64;
        let mut rng = Rng::new(17);
        let mut values = vec![0f32; (rows * BAR_DOF as i64) as usize];
        let mut logits = vec![0f32; (rows * BAR_DOF as i64 * NUM_BAR_BINS) as usize];
        for row in 0..rows as usize {
            for dof in 0..BAR_DOF {
                let bin = (rng.next_u64() % NUM_BAR_BINS as u64) as usize;
                values[row * BAR_DOF + dof] = supports.centers(dof)[bin] as f32;
                logits[(row * BAR_DOF + dof) * NUM_BAR_BINS as usize + bin] = 40.0;
            }
        }
        let target = Tensor::from_slice(&values).view([rows, BAR_DOF as i64]);
        let logits = Tensor::from_slice(&logits).view([rows, BAR_DOF as i64, NUM_BAR_BINS]);
        let crps = bar_crps_from_logits(&logits, &target, &supports);
        assert_eq!(crps.size(), vec![BAR_DOF as i64]);
        for dof in 0..BAR_DOF {
            let value = crps.double_value(&[dof as i64]);
            assert!(value < 1e-4, "DOF {dof} CRPS {value} should vanish");
        }

        // A point mass cannot detect a `narrow` misalignment in the
        // `integral F (1 - F)` term, because that term is identically zero for any
        // offset. Pin it against a closed-form evaluation of the defining integral
        // `integral (F(x) - 1{x >= y})^2 dx` on a non-degenerate predictive: the
        // CDF is flat at `F_j` on `[c_j, c_j+1)`, so each segment is exact.
        fn reference_crps(centers: &[f64], probs: &[f64], y: f64) -> f64 {
            let mut cumulative = 0.0;
            let mut total = (centers[0] - y).max(0.0);
            for j in 0..centers.len() - 1 {
                cumulative += probs[j];
                let (lo, hi) = (centers[j], centers[j + 1]);
                total += if y <= lo {
                    (cumulative - 1.0).powi(2) * (hi - lo)
                } else if y >= hi {
                    cumulative.powi(2) * (hi - lo)
                } else {
                    cumulative.powi(2) * (y - lo) + (cumulative - 1.0).powi(2) * (hi - y)
                };
            }
            total + (y - centers[centers.len() - 1]).max(0.0)
        }

        let uniform = Tensor::zeros(logits.size().as_slice(), (Kind::Float, Device::Cpu));
        let uniform_crps = bar_crps_from_logits(&uniform, &target, &supports);
        let flat_probs = vec![1.0 / NUM_BAR_BINS as f64; NUM_BAR_BINS as usize];
        for dof in 0..BAR_DOF {
            let got = uniform_crps.double_value(&[dof as i64]);
            assert!(
                got > crps.double_value(&[dof as i64]),
                "uniform CRPS must exceed the point mass for DOF {dof}"
            );
            let want: f64 = (0..rows as usize)
                .map(|row| {
                    reference_crps(
                        supports.centers(dof),
                        &flat_probs,
                        values[row * BAR_DOF + dof] as f64,
                    )
                })
                .sum::<f64>()
                / rows as f64;
            assert!(
                (got - want).abs() < 1e-4 * want.abs(),
                "DOF {dof} uniform CRPS {got} vs closed form {want}"
            );
        }
    }

    /// Build logits that reproduce a sample's empirical bin distribution exactly,
    /// i.e. the correctly specified head for that law, broadcast over `rows`.
    fn empirical_logits(supports: &BarSupports, samples: &[BarDof], rows: usize) -> Tensor {
        let bins = NUM_BAR_BINS as usize;
        let mut probs = vec![0f32; BAR_DOF * bins];
        for sample in samples {
            for dof in 0..BAR_DOF {
                probs[dof * bins + supports.bin_of(dof, sample.to_array()[dof] as f64)] += 1.0;
            }
        }
        let logits: Vec<f32> = probs
            .iter()
            .map(|&c| (c / samples.len() as f32).max(1e-30).ln())
            .collect();
        Tensor::from_slice(&logits)
            .view([1, BAR_DOF as i64, NUM_BAR_BINS])
            .expand([rows as i64, BAR_DOF as i64, NUM_BAR_BINS], false)
            .contiguous()
    }

    #[test]
    fn pit_of_a_well_specified_head_is_uniform() {
        // With atom bins the support is no longer uniform in probability, so the
        // correctly specified head is the one predicting the empirical bin
        // distribution. Atoms are handled by the randomized PIT: without it, the
        // ~20% of rows sitting on u = v = 0.5 would all collapse onto one PIT value.
        let fitting = atom_heavy_samples(40_000, 0x1234);
        let supports = BarSupports::fit(&fitting);
        // 40k rows puts the per-bucket standard error near 0.0015, so the band below
        // is ~4 sigma while the logit tensor stays around 100 MB.
        let rows = 40_000usize;
        let held_out = atom_heavy_samples(rows, 0x9876);
        let target = dof_tensor(&held_out);
        let logits = empirical_logits(&supports, &fitting, rows);
        let pit = bar_pit_from_logits(&logits, &target, &supports, 0xBEEF);
        assert_eq!(pit.size(), vec![rows as i64, BAR_DOF as i64]);
        assert!(pit.min().double_value(&[]) >= 0.0 && pit.max().double_value(&[]) <= 1.0);

        let buckets = 10i64;
        let histogram = (&pit * buckets as f64)
            .floor()
            .clamp(0.0, (buckets - 1) as f64)
            .to_kind(Kind::Int64);
        for dof in 0..BAR_DOF as i64 {
            let column = histogram.select(1, dof);
            for bucket in 0..buckets {
                let share = column
                    .eq(bucket)
                    .to_kind(Kind::Double)
                    .mean(Kind::Double)
                    .double_value(&[]);
                assert!(
                    // 4x tighter than one bin width (1/128 = 0.0078), so a
                    // one-bin CDF misalignment fails, at ~4 sigma of headroom.
                    (0.094..=0.106).contains(&share),
                    "DOF {} PIT bucket {bucket} share {share:.4} is not uniform",
                    BAR_DOF_NAMES[dof as usize]
                );
            }
            let mean = pit
                .select(1, dof)
                .to_kind(Kind::Double)
                .mean(Kind::Double)
                .double_value(&[]);
            assert!((mean - 0.5).abs() < 0.005, "DOF {dof} PIT mean {mean}");
        }
    }

    #[test]
    fn expectation_and_sampling_stay_on_the_support() {
        let _torch_rng_guard = test_rng::shared();
        let supports = synthetic_supports(50_000, 0x2468);
        let mut rng = Rng::new(19);
        let samples: Vec<BarDof> = (0..1_024).map(|_| synthetic_dof(&mut rng)).collect();
        let target = dof_tensor(&samples);
        let sharp = supports.encode_targets(&target).clamp_min(1e-30).log() * 4.0;

        let expected = supports.expectation(&sharp);
        assert_eq!(expected.size(), vec![samples.len() as i64, BAR_DOF as i64]);
        let drawn = supports.sample(&sharp, 1.0);
        assert_eq!(drawn.size(), expected.size());
        let greedy = supports.sample(&sharp, 0.0);
        for dof in 0..BAR_DOF {
            let lo = supports.lower_bounds(dof)[0];
            let hi = *supports.upper_bounds(dof).last().expect("bounds");
            for column in [&expected, &drawn, &greedy] {
                let slice = column.select(1, dof as i64).to_kind(Kind::Double);
                assert!(slice.min().double_value(&[]) >= lo - 1e-6);
                assert!(slice.max().double_value(&[]) <= hi + 1e-6);
            }
            // A sharp head must recover the value it was built from.
            let error = (expected.select(1, dof as i64) - target.select(1, dof as i64))
                .abs()
                .mean(Kind::Double)
                .double_value(&[]);
            let width = supports.widths(dof).iter().sum::<f64>() / NUM_BAR_BINS as f64;
            assert!(
                error < 4.0 * width,
                "DOF {} expectation off by {error} against mean bin width {width}",
                BAR_DOF_NAMES[dof]
            );
        }
    }

    #[test]
    fn head_respects_the_chain_factorization() {
        let _torch_rng_guard = test_rng::shared();
        let supports = synthetic_supports(20_000, 0x3690);
        let vs = nn::VarStore::new(Device::Cpu);
        let head = BarEmissionHead::new(&vs.root(), 32);
        // Break the zero init so the prefix conditioning has an effect.
        tch::no_grad(|| {
            for variable in vs.trainable_variables() {
                let mut variable = variable;
                let _ = variable.normal_(0.0, 0.4);
            }
        });

        let mut rng = Rng::new(23);
        let samples: Vec<BarDof> = (0..64).map(|_| synthetic_dof(&mut rng)).collect();
        let h = Tensor::randn([64, 32], (Kind::Float, Device::Cpu));
        let base_dof = dof_tensor(&samples);
        let base_bins = supports.bin_ids(&base_dof);
        let base = head.logits(&h, &base_bins);
        assert_eq!(base.size(), vec![64, BAR_DOF as i64, NUM_BAR_BINS]);

        // Perturbing one DOF may only move the factors that come after it.
        for (position, &dof) in BAR_CHAIN.iter().enumerate() {
            let mut perturbed = samples.clone();
            for sample in &mut perturbed {
                let mut array = sample.to_array();
                array[dof] += 0.37;
                *sample = BarDof::from_array(array);
            }
            let moved = head.logits(&h, &supports.bin_ids(&dof_tensor(&perturbed)));
            let delta = (&moved - &base).abs().amax([0i64, 2].as_slice(), false);
            for (other_position, &other) in BAR_CHAIN.iter().enumerate() {
                let change = delta.double_value(&[other as i64]);
                if other_position > position {
                    assert!(
                        change > 1e-5,
                        "{} must condition on {}",
                        BAR_DOF_NAMES[other],
                        BAR_DOF_NAMES[dof]
                    );
                } else {
                    assert!(
                        change < 1e-6,
                        "{} must not see {}",
                        BAR_DOF_NAMES[other],
                        BAR_DOF_NAMES[dof]
                    );
                }
            }
        }

        // Frozen weights keep the same values but stop parameter gradients.
        let frozen = head.logits_frozen(&h, &base_bins);
        assert!((frozen - &base).abs().max().double_value(&[]) < 1e-6);

        let drawn = head.sample(&h, &supports, 1.0);
        assert_eq!(drawn.size(), vec![64, BAR_DOF as i64]);
        assert!(drawn.isfinite().all().int64_value(&[]) == 1);
        let (mean, per_dof) = head.nll(&h, &base_dof, &supports, BarScoring::Density);
        assert!(mean.double_value(&[]).is_finite());
        assert_eq!(per_dof.size(), vec![BAR_DOF as i64]);
        assert_eq!(
            head.pit(&h, &base_dof, &supports, 7).size(),
            vec![64, BAR_DOF as i64]
        );
        assert_eq!(head.crps(&h, &base_dof, &supports).size(), vec![BAR_DOF as i64]);

        // The training loop bins the whole `[B, T + 1, BAR_DOF]` window once and hands
        // the head a NARROWED, non-contiguous view of it. That view must produce the
        // same logits as its contiguous copy, or the hoist silently trains on a
        // reinterpreted stride.
        let window = supports.bin_ids(&base_dof.view([4, 16, BAR_DOF as i64]));
        let view = window.narrow(1, 1, 12);
        assert!(!view.is_contiguous(), "the probe view is contiguous after all");
        let h_window = Tensor::randn([4, 12, 32], (Kind::Float, Device::Cpu));
        assert_eq!(
            f64::try_from(
                (head.logits(&h_window, &view) - head.logits(&h_window, &view.contiguous()))
                    .abs()
                    .max()
            )
            .expect("view gap"),
            0.0,
            "a narrowed bin view produced different logits from its contiguous copy"
        );
    }

    /// `forward_logits` and `sample` build the prefix through two separate code paths:
    /// the batched one masks `[rows, 1, SLOTS, DIM]` against `[BAR_DOF, SLOTS, 1]` and
    /// contracts with `einsum`, the sequential one masks `[rows, SLOTS, DIM]` against
    /// one DOF's mask row and contracts with `linear`. Both then flatten (slot, dim)
    /// into the `BAR_PREFIX_WIDTH` block that `ws.narrow(1, latent_dim, ..)` reads. A
    /// transposition or a slot swap in either one would leave the teacher-forced
    /// training loss exactly correct while every ancestral draw — direction accuracy,
    /// `BarWorldModel::imagine`, the planner's whole forecast — came from the wrong
    /// conditional, with no failing test and no loss regression.
    ///
    /// At temperature zero the chain is a deterministic argmax and each step decodes to
    /// its bin's center, so re-binning the draw recovers exactly the prefix the chain
    /// conditioned on. Replaying that prefix through the batched head must therefore
    /// reproduce the drawn bin for every factor. This also pins the round trip
    /// `rollout_beliefs` depends on: sampled value -> `bin_ids` -> trunk/head.
    #[test]
    fn ancestral_sampling_agrees_with_the_batched_head() {
        let _torch_rng_guard = test_rng::exclusive();
        let _ = tch::manual_seed(0x5A57);
        let supports = synthetic_supports(20_000, 0x5A57);
        let vs = nn::VarStore::new(Device::Cpu);
        let head = BarEmissionHead::new(&vs.root(), 20);
        tch::no_grad(|| {
            for variable in vs.trainable_variables() {
                let mut variable = variable;
                let _ = variable.normal_(0.0, 0.5);
            }
        });

        let h = Tensor::randn([48, 20], (Kind::Float, Device::Cpu));
        let drawn = head.sample(&h, &supports, 0.0);
        let drawn_bins = supports.bin_ids(&drawn);
        let replayed = head.logits(&h, &drawn_bins).argmax(-1, false);
        assert_eq!(
            Vec::<i64>::try_from(drawn_bins.reshape([-1]).contiguous()).expect("drawn bins"),
            Vec::<i64>::try_from(replayed.reshape([-1]).contiguous()).expect("replayed bins"),
            "the sequential chain and the batched head disagree on the same prefix"
        );
        // The draw has to be non-degenerate for the comparison to mean anything: a head
        // that always emitted bin 0 would pass trivially.
        let distinct: std::collections::HashSet<i64> =
            Vec::<i64>::try_from(drawn_bins.reshape([-1]).contiguous())
                .expect("drawn bins")
                .into_iter()
                .collect();
        assert!(distinct.len() > 4, "the argmax chain collapsed onto {distinct:?}");
    }

    #[test]
    fn nll_gradients_reach_the_latent_and_the_head() {
        let _torch_rng_guard = test_rng::shared();
        let supports = synthetic_supports(20_000, 0x4812);
        let vs = nn::VarStore::new(Device::Cpu);
        let head = BarEmissionHead::new(&vs.root(), 24);
        let mut rng = Rng::new(29);
        let samples: Vec<BarDof> = (0..32).map(|_| synthetic_dof(&mut rng)).collect();
        let target = dof_tensor(&samples);
        let h = Tensor::randn([32, 24], (Kind::Float, Device::Cpu)).set_requires_grad(true);

        let grad_sum = |t: &Tensor| {
            let grad = t.grad();
            if grad.defined() {
                grad.abs().sum(Kind::Double).double_value(&[])
            } else {
                0.0
            }
        };

        let (loss, _) = head.nll(&h, &target, &supports, BarScoring::Density);
        loss.backward();
        let head_weight = &head.heads[DOF_S].ws;
        assert!(
            grad_sum(head_weight) > 0.0,
            "head weights must receive gradient"
        );
        // Zero-init heads mean the prefix table cannot have moved yet: the gradient
        // into an embedding IS the head's prefix weight block, which is exactly zero.
        assert_eq!(grad_sum(&head.prefix_embed), 0.0);
        tch::no_grad(|| {
            for linear in &head.heads {
                let mut ws = linear.ws.shallow_clone();
                let _ = ws.normal_(0.0, 0.2);
            }
        });
        let (loss, _) = head.nll(&h, &target, &supports, BarScoring::Density);
        loss.backward();
        assert!(
            grad_sum(&head.prefix_embed) > 0.0,
            "the bin-embedding table must receive gradient once the heads are awake"
        );

        // The frozen branch must leave every head parameter untouched, the embedding
        // table included — it is the branch the dynamics KL runs through, and a leak
        // there would train the head on its own predicted latent.
        let before = (grad_sum(head_weight), grad_sum(&head.prefix_embed));
        let target_bins = supports.bin_ids(&target);
        let frozen_logits = head.logits_frozen(&h, &target_bins);
        let (kl, _) = bar_categorical_kl(&head.logits(&h, &target_bins).detach(), &frozen_logits);
        kl.backward();
        let after = (grad_sum(head_weight), grad_sum(&head.prefix_embed));
        assert!(
            (after.0 - before.0).abs() < 1e-9 && (after.1 - before.1).abs() < 1e-9,
            "frozen logits leaked gradient into head parameters: {before:?} -> {after:?}"
        );
        assert!(h.grad().defined());
    }

    #[test]
    fn supports_survive_a_json_round_trip() {
        let supports = synthetic_supports(20_000, 0x1010);
        let dir = std::env::temp_dir().join(format!("bar_dist_{}", std::process::id()));
        let path = dir.join("supports.json");
        supports.save(&path).expect("save supports");
        let loaded = BarSupports::load(&path).expect("load supports");
        // JSON decimal parsing is round-trip faithful to within an ulp, which is
        // far below the f32 precision the supports are actually evaluated in.
        for dof in 0..BAR_DOF {
            assert_eq!(supports.lower_bounds(dof).len(), loaded.lower_bounds(dof).len());
            for (a, b) in supports
                .lower_bounds(dof)
                .iter()
                .chain(supports.upper_bounds(dof))
                .zip(loaded.lower_bounds(dof).iter().chain(loaded.upper_bounds(dof)))
            {
                assert!(
                    (a - b).abs() <= 1e-12 * a.abs().max(1e-12),
                    "edge {a} reloaded as {b}"
                );
            }
        }
        let mut rng = Rng::new(31);
        let samples: Vec<BarDof> = (0..64).map(|_| synthetic_dof(&mut rng)).collect();
        let values = dof_tensor(&samples);
        let delta = (supports.encode_targets(&values) - loaded.encode_targets(&values))
            .abs()
            .max()
            .double_value(&[]);
        assert!(delta < 1e-6, "reloaded supports encode differently by {delta}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn encode_series_uses_a_causal_volume_reference() {
        let bars: Vec<PackedBar> = (0..64)
            .map(|i| {
                let base = 100.0 + i as f32 * 0.1;
                bar(base, base + 0.4, base - 0.3, base + 0.2, 1_000.0 + i as f32 * 25.0)
            })
            .collect();
        let dof = encode_series(&bars);
        assert_eq!(dof.len(), bars.len() - 1);
        assert!(dof.iter().all(|d| d.is_finite()));

        // Volume rises monotonically, so a trailing EMA reference keeps w positive.
        assert!(dof.iter().skip(1).all(|d| d.w > 0.0));

        let mut ema = VolumeEma::default();
        ema.observe(bars[0].volume);
        let expected = encode_dof(bars[0].close, &bars[1], ema.reference_for(bars[1].volume));
        assert_eq!(dof[0], expected);
        // The reference for bar 1 is bar 0's volume alone.
        assert_eq!(ema.reference_for(bars[1].volume), bars[0].volume);
    }

    /// A corpus in which the flat bar is common and carries the exact encoder
    /// identity `s == 0  =>  u == v == 0.5`, with `r`/`w` unconstrained so the
    /// identity is genuinely a function of `s` alone.
    fn flat_bar_corpus(count: usize, seed: u64) -> Vec<BarDof> {
        let mut rng = Rng::new(seed);
        (0..count)
            .map(|_| {
                let r = (0.004 * rng.normal()) as f32;
                let w = (0.4 * rng.normal()) as f32;
                if rng.uniform() < 0.35 {
                    BarDof { r, s: 0.0, u: 0.5, v: 0.5, w }
                } else {
                    BarDof {
                        r,
                        s: (0.003 * (1.0 + 0.5 * rng.normal()).abs()).min(0.2) as f32,
                        u: rng.uniform() as f32,
                        v: rng.uniform() as f32,
                        w,
                    }
                }
            })
            .collect()
    }

    /// The whole point of indexing the prefix by BIN: `P(u = 0.5 | s = 0) = 1` is a
    /// hard logical identity of [`encode_dof`], and a rank-1 affine map of the raw
    /// `s` value has to approximate a step function at `s == 0` to express it. A bin
    /// lookup represents it exactly, and a few hundred steps are enough to find it.
    #[test]
    fn bin_prefix_learns_the_flat_bar_identity() {
        // The embedding table is drawn from the global torch generator, and the verdict
        // is a threshold on the fitted probability, so pin the draw.
        let _torch_rng_guard = test_rng::exclusive();
        let _ = tch::manual_seed(0xF1A7);
        let corpus = flat_bar_corpus(40_000, 0xF1A7);
        let supports = BarSupports::fit(&corpus);
        let flat_s = supports
            .atoms(DOF_S)
            .iter()
            .find(|a| a.value == 0.0)
            .expect("s == 0 atom")
            .bin;
        let half: Vec<usize> = [DOF_U, DOF_V]
            .into_iter()
            .map(|dof| {
                supports
                    .atoms(dof)
                    .iter()
                    .find(|a| a.value == 0.5)
                    .unwrap_or_else(|| panic!("{} 0.5 atom", BAR_DOF_NAMES[dof]))
                    .bin
            })
            .collect();

        // A constant latent: the head can only learn this from the chain prefix.
        let vs = nn::VarStore::new(Device::Cpu);
        let head = BarEmissionHead::new(&vs.root(), 4);
        let mut optimizer = nn::Adam::default()
            .build(&vs, 0.05)
            .expect("adam");

        let batch = &corpus[..1024];
        let target = dof_tensor(batch);
        let bins = supports.bin_ids(&target);
        let soft = supports.targets(&target, BarScoring::Smoothed);
        let h = Tensor::zeros([batch.len() as i64, 4], (Kind::Float, Device::Cpu));
        for _ in 0..400 {
            let (loss, _) = bar_nll_from_logits(&head.logits(&h, &bins), &soft);
            optimizer.backward_step(&loss);
        }

        // Probe flat bars at several distinct r bins: the identity must hold for
        // every one of them, because it is a function of s alone.
        let probes: Vec<BarDof> = [-0.02f32, -0.004, 0.0, 0.004, 0.02]
            .into_iter()
            .map(|r| BarDof { r, s: 0.0, u: 0.5, v: 0.5, w: 0.0 })
            .collect();
        let probe_bins = supports.bin_ids(&dof_tensor(&probes));
        assert!(
            (0..probes.len() as i64)
                .all(|row| probe_bins.int64_value(&[row, DOF_S as i64]) == flat_s as i64),
            "the probe rows did not land on the s == 0 atom bin"
        );
        let probs = tch::no_grad(|| {
            head.logits(
                &Tensor::zeros([probes.len() as i64, 4], (Kind::Float, Device::Cpu)),
                &probe_bins,
            )
            .softmax(-1, Kind::Double)
        });
        for (slot, &dof) in [DOF_U, DOF_V].iter().enumerate() {
            let mut worst = 1.0f64;
            for row in 0..probes.len() as i64 {
                let mass = probs.double_value(&[row, dof as i64, half[slot] as i64]);
                worst = worst.min(mass);
                assert!(
                    mass > 0.99,
                    "P({} = 0.5 | s = 0) is {mass} on probe row {row}, the identity was not learned",
                    BAR_DOF_NAMES[dof]
                );
            }
            println!("P({} = 0.5 | s = 0) >= {worst:.6}", BAR_DOF_NAMES[dof]);
        }

        // And it did not collapse to always predicting the atom: a live bar must
        // still spread its shape mass.
        let live = [BarDof { r: 0.001, s: 0.004, u: 0.31, v: 0.77, w: 0.0 }];
        let live_probs = tch::no_grad(|| {
            head.logits(
                &Tensor::zeros([1, 4], (Kind::Float, Device::Cpu)),
                &supports.bin_ids(&dof_tensor(&live)),
            )
            .softmax(-1, Kind::Double)
        });
        for (slot, &dof) in [DOF_U, DOF_V].iter().enumerate() {
            let mass = live_probs.double_value(&[0, dof as i64, half[slot] as i64]);
            assert!(
                mass < 0.2,
                "P({} = 0.5 | s != 0) is {mass}; the head collapsed onto the atom",
                BAR_DOF_NAMES[dof]
            );
        }
    }

    /// The prefix cannot be fitted on a value rollout can never produce: it enters
    /// the head as a bin id, and [`BarSupports::bin_ids`] clamps onto
    /// `[lo[0], hi[NUM_BAR_BINS - 1]]` exactly as [`BarSupports::bin_of`] does. This
    /// is what makes the old separate-clamp fix unnecessary rather than merely
    /// redundant — there is no longer a value path into the head at all.
    #[test]
    fn prefix_path_and_bin_ids_agree_on_clamping() {
        let _torch_rng_guard = test_rng::shared();
        let supports = synthetic_supports(20_000, 0xC1A3);
        let vs = nn::VarStore::new(Device::Cpu);
        let head = BarEmissionHead::new(&vs.root(), 16);
        tch::no_grad(|| {
            for variable in vs.trainable_variables() {
                let mut variable = variable;
                let _ = variable.normal_(0.0, 0.5);
            }
        });

        // 14x past the s edge and 20x past the r edge, the scale the live corpus
        // actually produces, plus the encode clamp itself.
        let edge = |dof: usize, high: bool| -> f32 {
            if high {
                supports.upper_bounds(dof)[NUM_BAR_BINS as usize - 1] as f32
            } else {
                supports.lower_bounds(dof)[0] as f32
            }
        };
        let wild = vec![
            BarDof { r: -LOG_LIMIT as f32, s: 14.4 * edge(DOF_S, true), u: 1.0, v: 0.0, w: 9.0 },
            BarDof { r: 20.4 * edge(DOF_R, true), s: 1.369, u: 0.0, v: 1.0, w: -9.0 },
        ];
        let clamped: Vec<BarDof> = wild
            .iter()
            .map(|d| {
                let mut array = d.to_array();
                for (dof, value) in array.iter_mut().enumerate() {
                    *value = value.clamp(edge(dof, false), edge(dof, true));
                }
                BarDof::from_array(array)
            })
            .collect();

        let wild_bins = supports.bin_ids(&dof_tensor(&wild));
        let clamped_bins = supports.bin_ids(&dof_tensor(&clamped));
        assert_eq!(
            Vec::<i64>::try_from(wild_bins.reshape([-1]).contiguous()).expect("wild bins"),
            Vec::<i64>::try_from(clamped_bins.reshape([-1]).contiguous()).expect("clamped bins"),
            "an out-of-support prefix binned somewhere the support edge does not"
        );
        // ... and the host lookup agrees, so nothing in the chain can drift.
        for (row, sample) in wild.iter().enumerate() {
            for (dof, value) in sample.to_array().into_iter().enumerate() {
                assert_eq!(
                    wild_bins.int64_value(&[row as i64, dof as i64]) as usize,
                    supports.bin_of(dof, value as f64)
                );
            }
        }

        let h = Tensor::randn([2, 16], (Kind::Float, Device::Cpu));
        let wild_logits = head.logits(&h, &wild_bins);
        let clamped_logits = head.logits(&h, &clamped_bins);
        let gap = f64::try_from((&wild_logits - &clamped_logits).abs().max()).expect("gap");
        assert_eq!(
            gap, 0.0,
            "the head extrapolated past the support edge by {gap} nats of logit"
        );

        // The equality above is only informative if the head reads the prefix at all:
        // a head that ignored its bins entirely would satisfy it trivially. Move the
        // `s` prefix one bin off the edge and the factors AFTER `s` in the chain must
        // move, while `s` itself must not.
        let inside: Vec<BarDof> = clamped
            .iter()
            .map(|d| {
                let mut array = d.to_array();
                let bin = supports.bin_of(DOF_S, array[DOF_S] as f64);
                let shifted = bin.saturating_sub(1).min(NUM_BAR_BINS as usize - 2);
                array[DOF_S] = supports.centers(DOF_S)[shifted] as f32;
                BarDof::from_array(array)
            })
            .collect();
        let inside_bins = supports.bin_ids(&dof_tensor(&inside));
        assert_ne!(
            inside_bins.int64_value(&[0, DOF_S as i64]),
            wild_bins.int64_value(&[0, DOF_S as i64]),
            "the probe did not actually change the s bin"
        );
        let moved = head.logits(&h, &inside_bins);
        let delta = (&moved - &wild_logits).abs().amax([0i64, 2].as_slice(), false);
        for (position, &dof) in BAR_CHAIN.iter().enumerate() {
            let change = delta.double_value(&[dof as i64]);
            if position > CHAIN_POS[DOF_S] {
                assert!(
                    change > 1e-5,
                    "{} did not react to the s prefix bin, so the clamp assertion above \
                     proves nothing",
                    BAR_DOF_NAMES[dof]
                );
            } else {
                assert!(change < 1e-6, "{} must not see s", BAR_DOF_NAMES[dof]);
            }
        }
    }

    /// The label-smoothing floor is a quadrature over the fitted support; check it
    /// against a Monte-Carlo integration that draws from the same fitted law and
    /// routes every draw through the PUBLIC [`BarSupports::targets`], atom one-hot
    /// branch included. The other two rules score the bin the observation landed in,
    /// so their floor is exactly zero and an oracle pays nothing.
    #[test]
    fn smoothing_floor_matches_an_independent_integration() {
        let supports = synthetic_supports(60_000, 0x5F10);
        let floor = supports.scoring_floor(BarScoring::Smoothed);
        assert!(
            (supports.scoring_floor_bar(BarScoring::Smoothed) - floor.iter().sum::<f64>()).abs()
                < 1e-12
        );
        for scoring in [BarScoring::Hard, BarScoring::Density] {
            assert_eq!(supports.scoring_floor(scoring), [0.0; BAR_DOF], "{scoring}");
            assert_eq!(supports.scoring_floor_bar(scoring), 0.0, "{scoring}");
        }

        // 20k draws puts the Monte-Carlo standard error near 1e-3 nats, two orders
        // below the tolerance, while keeping the target tensor under 60 MB.
        const DRAWS: usize = 20_000;
        let mut rng = Rng::new(0xF100);
        let mut flat = Vec::with_capacity(DRAWS * BAR_DOF);
        for _ in 0..DRAWS {
            for dof in 0..BAR_DOF {
                // Inverse-CDF draw from the fitted bin masses, then uniform inside
                // the bin. An atom bin has zero width, so it reproduces its value.
                let mut target = rng.uniform();
                let mut bin = NUM_BAR_BINS as usize - 1;
                for (index, mass) in supports.bin_masses(dof).iter().enumerate() {
                    target -= mass;
                    if target <= 0.0 {
                        bin = index;
                        break;
                    }
                }
                let (lo, hi) = (
                    supports.lower_bounds(dof)[bin],
                    supports.upper_bounds(dof)[bin],
                );
                flat.push((lo + (hi - lo) * rng.uniform()) as f32);
            }
        }
        let draws = Tensor::from_slice(&flat).view([DRAWS as i64, BAR_DOF as i64]);
        let targets = supports
            .targets(&draws, BarScoring::Smoothed)
            .into_targets();
        let entropy = -(&targets * targets.clamp_min(1e-30).log()).sum_dim_intlist(
            [-1].as_slice(),
            false,
            Kind::Double,
        );
        let sampled = entropy.mean_dim([0i64].as_slice(), false, Kind::Double);
        for dof in 0..BAR_DOF {
            let mine = floor[dof];
            let theirs = sampled.double_value(&[dof as i64]);
            assert!(
                (mine - theirs).abs() < 0.02,
                "{} smoothing floor {mine} vs sampled {theirs}",
                BAR_DOF_NAMES[dof]
            );
            assert!(mine > 0.0 && mine < (NUM_BAR_BINS as f64).ln());
        }
    }

    /// The degeneracy/shape split has to be exact, not approximate: it is a
    /// regrouping of the same cross entropy, so the halves must reconstruct the
    /// total for the model AND for the marginal reference it is charted against,
    /// under every scoring rule.
    #[test]
    fn nll_decomposition_reconstructs_the_total() {
        let _torch_rng_guard = test_rng::shared();
        let supports = synthetic_supports(40_000, 0xD3C0);
        let vs = nn::VarStore::new(Device::Cpu);
        let head = BarEmissionHead::new(&vs.root(), 12);
        tch::no_grad(|| {
            for variable in vs.trainable_variables() {
                let mut variable = variable;
                let _ = variable.normal_(0.0, 0.3);
            }
        });
        let mut rng = Rng::new(0xD3C1);
        let samples: Vec<BarDof> = (0..256).map(|_| synthetic_dof(&mut rng)).collect();
        let target = dof_tensor(&samples);
        let h = Tensor::randn([256, 12], (Kind::Float, Device::Cpu));
        let logits = head.logits(&h, &supports.bin_ids(&target));

        for scoring in BarScoring::ALL {
            let marginal = supports.marginal_nll_parts(scoring);
            let whole = supports.marginal_nll_dof(scoring);
            for dof in 0..BAR_DOF {
                assert!(
                    (marginal.total()[dof] - whole[dof]).abs() < 1e-9,
                    "{} marginal split {} + {} != {} under {scoring}",
                    BAR_DOF_NAMES[dof],
                    marginal.class[dof],
                    marginal.shape[dof],
                    whole[dof]
                );
                // The smoothed marginal sums to one only to within `from_bins`'s 1e-6
                // validation, so `-m ln m` at `m ~ 1` carries that much signed noise. The
                // class half never carries a width, so it is non-negative under every
                // rule; only the density rule's SHAPE half may go negative, and that is
                // the whole point of it being a log density.
                assert!(
                    marginal.class[dof] >= -1e-6,
                    "{} class {} went negative under {scoring}",
                    BAR_DOF_NAMES[dof],
                    marginal.class[dof]
                );
                if !scoring.is_density() {
                    assert!(marginal.shape[dof] >= -1e-6);
                }
            }
            // `w` has no atoms in this corpus, so its degeneracy indicator is
            // deterministic and its class term is zero up to that same noise.
            assert_eq!(supports.atoms(DOF_W).len(), 0);
            assert!(marginal.class[DOF_W].abs() < 1e-6);

            let soft = supports.targets(&target, scoring);
            let parts = bar_nll_decomposition(&logits, &soft, &supports);
            let (_, per_dof) = bar_nll_from_logits(&logits, &soft);
            for dof in 0..BAR_DOF {
                let class = parts.class.double_value(&[dof as i64]);
                let shape = parts.shape.double_value(&[dof as i64]);
                let total = parts.total.double_value(&[dof as i64]);
                assert!(
                    (total - per_dof.double_value(&[dof as i64])).abs() < 1e-4,
                    "{} decomposition total drifted from bar_nll_from_logits under {scoring}",
                    BAR_DOF_NAMES[dof]
                );
                assert!(
                    (class + shape - total).abs() < 1e-3,
                    "{} split {class} + {shape} != {total} under {scoring}",
                    BAR_DOF_NAMES[dof]
                );
                assert!(class >= -1e-6);
                if !scoring.is_density() {
                    assert!(shape >= -1e-6);
                }
            }
        }
    }

    /// `H(p) + sum_b p_b ln w_b` — the density rule's marginal reference — written out for
    /// an ARBITRARY bin count over the analytic law's clipped quantile range, so "double
    /// the bins" is expressible even though [`NUM_BAR_BINS`] is a compile-time constant.
    ///
    /// Mirrors `fit_dof_support` exactly: the outer edges sit at the
    /// [`BAR_SUPPORT_CLIP_QUANTILE`] quantiles and the interior edges are equal-mass
    /// positions of the law conditioned onto that range.
    fn density_marginal_reference(bins: usize, quantile: impl Fn(f64) -> f64) -> f64 {
        let clip = BAR_SUPPORT_CLIP_QUANTILE;
        let edge = |j: usize| quantile(clip + (1.0 - 2.0 * clip) * j as f64 / bins as f64);
        let p = 1.0 / bins as f64;
        -p.ln() + (0..bins).map(|j| p * (edge(j + 1) - edge(j)).ln()).sum::<f64>()
    }

    /// The property that makes `density` proper for a continuous law and `hard` not: its
    /// value does not move when the discretization is refined, because the measure term
    /// cancels the `ln(bins)` that the categorical picks up.
    ///
    /// [`NUM_BAR_BINS`] is fixed at compile time, so the doubling happens in
    /// [`density_marginal_reference`], which is the same formula the production code
    /// evaluates. The link back to production is the first assertion: the fitted
    /// 128-bin support reproduces that formula on a sample from the same law.
    #[test]
    fn the_density_rule_is_invariant_to_the_bin_count() {
        // Exponential(1) on `r`: quantile `-ln(1 - u)`, differential entropy exactly 1 nat.
        let quantile = |u: f64| -(1.0 - u).ln();
        const ROWS: usize = 200_000;
        let mut rng = Rng::new(0xB1_0000);
        let samples: Vec<BarDof> = (0..ROWS)
            .map(|_| {
                let mut dof = synthetic_dof(&mut rng);
                dof.r = quantile(rng.uniform()) as f32;
                dof
            })
            .collect();
        let supports = BarSupports::fit(&samples);
        assert!(
            supports.atoms(DOF_R).is_empty(),
            "a continuous law must not manufacture atoms on r"
        );

        let fitted = supports.marginal_nll_dof(BarScoring::Density)[DOF_R];
        let coarse = density_marginal_reference(NUM_BAR_BINS as usize, quantile);
        let fine = density_marginal_reference(2 * NUM_BAR_BINS as usize, quantile);
        assert!(
            (fitted - coarse).abs() < 0.03,
            "the fitted 128-bin density reference {fitted} does not match the analytic \
             {coarse}"
        );
        // The claim: doubling the bins moves the density figure by discretization error
        // only. The hard rule moves by ln 2 = 0.693 by construction, which is what makes
        // it unusable for a bin-count ablation.
        let drift = (fine - coarse).abs();
        assert!(
            drift < 0.02,
            "doubling the bins moved the density reference by {drift} nats \
             ({coarse} -> {fine})"
        );
        assert!(
            drift < 0.05 * std::f64::consts::LN_2,
            "the density drift {drift} is not small beside the hard rule's ln 2 shift"
        );
        // Both bin counts sit at the analytic differential entropy of Exponential(1).
        for (label, value) in [("128", coarse), ("256", fine)] {
            assert!(
                (value - 1.0).abs() < 0.05,
                "{label}-bin density reference {value} is not the 1-nat differential entropy"
            );
        }
        // The contrast, measured rather than asserted from theory: the hard rule's
        // reference IS ln(bins) up to the histogram's own entropy deficit.
        let hard = supports.marginal_nll_dof(BarScoring::Hard)[DOF_R];
        assert!(
            (hard - (NUM_BAR_BINS as f64).ln()).abs() < 0.05,
            "the hard reference {hard} is not ln(bins) = {}",
            (NUM_BAR_BINS as f64).ln()
        );
    }

    /// The hard rule is the `sigma -> 0` limit of the smoothed one. Probed at every bin's
    /// center, which is strictly interior for a continuous bin and the atom itself for an
    /// atom bin, so the limit is an exact one-hot rather than an edge split.
    #[test]
    fn hard_is_the_zero_sigma_limit_of_smoothed() {
        let _torch_rng_guard = test_rng::shared();
        let supports = synthetic_supports(50_000, 0x51_6D_A0);
        let mut flat = Vec::with_capacity(NUM_BAR_BINS as usize * BAR_DOF);
        for bin in 0..NUM_BAR_BINS as usize {
            for dof in 0..BAR_DOF {
                flat.push(supports.centers(dof)[bin] as f32);
            }
        }
        let probes = Tensor::from_slice(&flat).view([NUM_BAR_BINS, BAR_DOF as i64]);
        let hard = supports.targets(&probes, BarScoring::Hard);
        let logits = Tensor::randn(
            [NUM_BAR_BINS, BAR_DOF as i64, NUM_BAR_BINS],
            (Kind::Float, Device::Cpu),
        );
        let (hard_nll, _) = bar_nll_from_logits(&logits, &hard);
        let hard_nll = hard_nll.double_value(&[]);

        let mut previous = f64::INFINITY;
        for sigma_ratio in [BAR_LABEL_SIGMA_RATIO, 0.1, 1e-3] {
            let smoothed = supports.targets_with_sigma(&probes, BarScoring::Smoothed, sigma_ratio);
            let gap = (smoothed.targets() - hard.targets())
                .abs()
                .max()
                .double_value(&[]);
            assert!(
                gap < previous,
                "shrinking sigma to {sigma_ratio} did not tighten the gap to the one-hot: \
                 {gap} >= {previous}"
            );
            previous = gap;
            let (nll, _) = bar_nll_from_logits(&logits, &smoothed);
            if sigma_ratio <= 1e-3 {
                assert!(
                    gap < 1e-5,
                    "at sigma {sigma_ratio} the smoothed target is still {gap} off the one-hot"
                );
                assert!(
                    (nll.double_value(&[]) - hard_nll).abs() < 1e-4,
                    "at sigma {sigma_ratio} the smoothed NLL {} != the hard NLL {hard_nll}",
                    nll.double_value(&[])
                );
            }
        }
        // At the production width the two rules must NOT coincide, or the test above would
        // be vacuous and `scoring_floor` would be measuring nothing.
        assert!(
            supports.scoring_floor_bar(BarScoring::Smoothed) > 1.0,
            "the production smoothing width costs no measurable floor"
        );
    }

    /// On an analytic MIXED law the density rule must score an atom as a probability MASS
    /// and a continuous observation as a DENSITY. Those are different units, and the whole
    /// reason the rule needs the support's widths.
    #[test]
    fn the_atom_path_is_a_mass_and_the_continuous_path_a_density() {
        // r ~ 0.25 * delta(0) + 0.75 * Uniform(-0.05, 0.05).
        const ATOM_MASS: f64 = 0.25;
        const SPAN: f64 = 0.1;
        const ROWS: usize = 200_000;
        let mut rng = Rng::new(0xA70_D1);
        let samples: Vec<BarDof> = (0..ROWS)
            .map(|i| {
                let mut dof = synthetic_dof(&mut rng);
                dof.r = if i % 4 == 0 {
                    0.0
                } else {
                    (-0.5 * SPAN + SPAN * rng.uniform()) as f32
                };
                dof
            })
            .collect();
        let supports = BarSupports::fit(&samples);
        let atoms = supports.atoms(DOF_R);
        assert_eq!(atoms.len(), 1, "the 25% point mass must be promoted to an atom");
        assert_eq!(atoms[0].value, 0.0);
        assert!((atoms[0].mass - ATOM_MASS).abs() < 5e-3, "{}", atoms[0].mass);

        // The oracle head: a fixed prediction equal to the fitted marginal row.
        let rows = 40_000usize;
        let target = dof_tensor(&samples[..rows]);
        let mut logit_flat = Vec::with_capacity(BAR_DOF * NUM_BAR_BINS as usize);
        for dof in 0..BAR_DOF {
            logit_flat.extend(
                supports
                    .bin_masses(dof)
                    .iter()
                    .map(|p| (p.max(1e-30) as f32).ln()),
            );
        }
        let logits = Tensor::from_slice(&logit_flat)
            .view([1, BAR_DOF as i64, NUM_BAR_BINS])
            .expand([rows as i64, BAR_DOF as i64, NUM_BAR_BINS], false)
            .contiguous();
        let terms = bar_nll_terms(&logits, &supports.targets(&target, BarScoring::Density))
            .select(-1, DOF_R as i64);
        let on_atom = target.select(-1, DOF_R as i64).eq(0.0);
        let atom_nats = terms.masked_select(&on_atom).mean(Kind::Double).double_value(&[]);
        let continuous_nats = terms
            .masked_select(&on_atom.logical_not())
            .mean(Kind::Double)
            .double_value(&[]);

        // A MASS: `-ln P(atom)`, with no width anywhere in it.
        let expected_atom = -ATOM_MASS.ln();
        assert!(
            (atom_nats - expected_atom).abs() < 0.02,
            "atom rows scored {atom_nats}, expected the mass -ln({ATOM_MASS}) = {expected_atom}"
        );
        // A DENSITY: `-ln((1 - m) / span)`, negative because the span is under one.
        let expected_continuous = -((1.0 - ATOM_MASS) / SPAN).ln();
        assert!(
            expected_continuous < 0.0,
            "the fixture must make the continuous log density negative"
        );
        assert!(
            (continuous_nats - expected_continuous).abs() < 0.03,
            "continuous rows scored {continuous_nats}, expected the log density \
             {expected_continuous}"
        );
        // And the reference line is the analytic mixed-measure entropy of the same law.
        let expected_total = ATOM_MASS * expected_atom + (1.0 - ATOM_MASS) * expected_continuous;
        let reported = supports.marginal_nll_dof(BarScoring::Density)[DOF_R];
        assert!(
            (reported - expected_total).abs() < 0.03,
            "the density reference {reported} is not the analytic {expected_total}"
        );

        // The hard rule charges the same atom the same MASS — atoms carry no width under
        // any rule — but charges the continuous rows a bin probability instead, which is
        // where the two rules part company.
        let hard_terms = bar_nll_terms(&logits, &supports.targets(&target, BarScoring::Hard))
            .select(-1, DOF_R as i64);
        let hard_atom = hard_terms
            .masked_select(&on_atom)
            .mean(Kind::Double)
            .double_value(&[]);
        assert!(
            (hard_atom - atom_nats).abs() < 1e-4,
            "the atom path must be width-free: hard {hard_atom} vs density {atom_nats}"
        );
        let hard_continuous = hard_terms
            .masked_select(&on_atom.logical_not())
            .mean(Kind::Double)
            .double_value(&[]);
        assert!(
            hard_continuous > 0.0 && hard_continuous - continuous_nats > 1.0,
            "the continuous path must differ by the log measure: hard {hard_continuous} vs \
             density {continuous_nats}"
        );
    }

    /// OBJ-FCST-001. The marginalized forecast law is the teacher-forced law EXACTLY when the
    /// chain factors are conditionally independent, and strictly worse when they are not.
    ///
    /// This is the property that makes the forecast number interpretable. Independence is
    /// imposed surgically by zeroing the prefix embedding table: every other weight stays
    /// random, so the latent path is untouched and the only thing removed is the same-bar
    /// dependence. Under independence `p(dof | h, prefix_s)` does not depend on `s`, the
    /// mixture collapses onto its single component, and the analytic answer IS the
    /// teacher-forced one — the estimator must recover it to Monte-Carlo-free precision,
    /// because there is nothing left to average over.
    #[test]
    fn the_marginalized_forecast_recovers_teacher_forcing_only_under_independence() {
        let _torch_rng_guard = test_rng::exclusive();
        let _ = tch::manual_seed(0x0FEC);
        let supports = synthetic_supports(40_000, 0x0FEC);
        let rows = 256i64;
        let latent = 12i64;
        let vs = nn::VarStore::new(Device::Cpu);
        let head = BarEmissionHead::new(&vs.root(), latent);
        // Every head is zero-initialized, which is itself the degenerate independent case, so
        // the dependent arm has to give the weights something to depend on.
        tch::no_grad(|| {
            for variable in vs.trainable_variables() {
                let mut variable = variable;
                let _ = variable.normal_(0.0, 0.4);
            }
        });
        let h = Tensor::randn([rows, latent], (Kind::Float, Device::Cpu));
        // Targets drawn from the head's OWN law, so the comparison is not dominated by a
        // mismatch between the head and the data.
        let target = head.sample(&h, &supports, 1.0);
        let bins = supports.bin_ids(&target);
        let targets = supports.targets(&target, BarScoring::Density);
        let per_dof = |terms: &Tensor| -> [f64; BAR_DOF] {
            let mean = terms
                .reshape([-1, BAR_DOF as i64])
                .mean_dim([0i64].as_slice(), false, Kind::Float);
            let mut out = [0.0f64; BAR_DOF];
            for dof in 0..BAR_DOF {
                out[dof] = mean.double_value(&[dof as i64]);
            }
            out
        };

        let teacher = per_dof(&bar_nll_terms(&head.logits(&h, &bins), &targets));
        // `forecast_log_probs` returns normalized log-probabilities, and `log_softmax` of a
        // normalized log-probability row is that row, so the same scorer applies unchanged.
        let forecast = per_dof(&bar_nll_terms(
            &head.forecast_log_probs(&h, 128, 0xE7A1_5E7D),
            &targets,
        ));
        let first = BAR_CHAIN[0];
        assert!(
            (forecast[first] - teacher[first]).abs() < 1e-5,
            "{} has no prefix, so its marginal must be exact: {} vs {}",
            BAR_DOF_NAMES[first],
            forecast[first],
            teacher[first]
        );
        let dependent_inflation: f64 =
            forecast.iter().sum::<f64>() - teacher.iter().sum::<f64>();
        assert!(
            dependent_inflation > 1e-2,
            "with a dependent chain the marginalized law must be strictly worse than the \
             teacher-forced one; inflation was only {dependent_inflation} nats/bar \
             (forecast {forecast:?} vs teacher {teacher:?})"
        );

        // Independence: the prefix embedding is the ONLY route from one factor of a bar to
        // another, so zeroing it makes the chain conditionally independent given `h`.
        tch::no_grad(|| {
            let mut table = head.prefix_embed.shallow_clone();
            let _ = table.zero_();
        });
        let teacher_indep = per_dof(&bar_nll_terms(&head.logits(&h, &bins), &targets));
        let forecast_indep = per_dof(&bar_nll_terms(
            &head.forecast_log_probs(&h, 128, 0xE7A1_5E7D),
            &targets,
        ));
        for dof in 0..BAR_DOF {
            assert!(
                (forecast_indep[dof] - teacher_indep[dof]).abs() < 1e-5,
                "{} must coincide under independence: forecast {} vs teacher {}",
                BAR_DOF_NAMES[dof],
                forecast_indep[dof],
                teacher_indep[dof]
            );
        }
        let independent_inflation: f64 =
            forecast_indep.iter().sum::<f64>() - teacher_indep.iter().sum::<f64>();
        assert!(
            independent_inflation.abs() < 5e-5,
            "independent chains leave nothing to marginalize, so the inflation must be zero, \
             not {independent_inflation}"
        );
        // The whole point of the pair: the estimator is not silently returning the
        // teacher-forced number in both regimes.
        assert!(
            dependent_inflation > 100.0 * independent_inflation.abs().max(1e-6),
            "the two regimes must be distinguishable: dependent {dependent_inflation} vs \
             independent {independent_inflation}"
        );
        println!(
            "teacher-forcing inflation on the synthetic fixture: {dependent_inflation:.4} \
             nats/bar dependent, {independent_inflation:.2e} independent"
        );
    }
}
