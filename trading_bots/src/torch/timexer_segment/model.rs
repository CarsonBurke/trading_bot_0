use anyhow::{ensure, Result};
use fused_kernels::{loss_geometry, qk_norm_rope, relu_square};
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};
use tch::{nn, Device, Kind, Tensor};

use super::{
    calibration::FrozenGain,
    corpus::Batch,
    features::{Feature, FeatureSet},
    target_basis::{self, BasisTransform, BasisWeight, TargetBasis},
};
use crate::torch::model::rope::RotaryEmbedding;

pub const CHANNELS: i64 = 4;
/// Per-bar log-return variance floor: one basis point of volatility per five-minute bar.
pub const RETURN_VARIANCE_FLOOR: f64 = 1e-8;
/// Ridge weight of the β=1 prior in bars: β_k is the OLS slope of the ticker's bar returns on
/// the market steps over `[0, t_k]`, shrunk toward 1 with the weight of `BETA_PRIOR_BARS` average
/// market-step squares, so origins with little history sit near 1 and β converges to the OLS
/// slope as history accumulates (half-way at 256 valid pairs).
pub const BETA_PRIOR_BARS: f64 = 256.0;
const COVARIATE_WIDTH: i64 = 256;
const HEAD_HIDDEN: i64 = 1024;
/// Four candle coordinates plus one log-scale per OHLC channel for every future bar.
const OUTPUTS_PER_BAR: i64 = 2 * CHANNELS;
/// μP-style output multiplier `1/√fan_in` on the zero-initialised head: Adam moves every weight
/// by ~lr per step, so an unscaled 1024-wide head would jump O(lr·fan_in) ≈ 8 per output.
const HEAD_OUTPUT_SCALE: f64 = 1.0 / 32.0;
/// tanh soft cap on the log predictive scale around the `½·ln h` prior: `s/σ√h ∈ [e⁻⁴, e⁴]`.
const LOG_SCALE_CAP: f64 = 4.0;
/// RMSNorm epsilon - see [`rms_norm`] for why it is explicit rather than the reference's
/// dtype-dependent default.
const NORM_EPS: f64 = 1e-6;
/// Residual-stream scale at init, per SUB-BLOCK. modded-nanogpt `train_gpt.py:1336-1338`:
/// "sqrt(1.1) per sublayer so cumulative per-layer scaling is 1.1", a raw (not sigmoid)
/// learnable scalar. The stack therefore multiplies the residual stream by 1.1^layers before
/// the final norm, which the final norm divides straight back out - see
/// `the_stack_is_the_identity_on_the_residual_stream_at_init`.
const RESID_LAMBDA_INIT: f64 = 1.048_808_848_170_151_6; // √1.1, `world_model.rs:112-113`
/// Sub-block output scale at init. modded-nanogpt `train_gpt.py:1334`:
/// `self.post_lambdas = nn.Parameter(torch.ones(num_layers, 2))`.
const POST_LAMBDA_INIT: f64 = 1.0;
/// Embedding re-injection scale at init. modded-nanogpt `train_gpt.py:1389`
/// (`bs_init[0, 10] = 0.0  # x0_lambda[10] (init 0)`) and `train_gpt_medium.py:1078`
/// (`self.x0_lambdas = nn.Parameter(torch.zeros(2*num_layers))`). Zero, so the injection
/// starts as an exact no-op and the stack starts as the identity map.
const X0_LAMBDA_INIT: f64 = 0.0;
/// Logit initialising every U-net skip gate, from modded-nanogpt: the skips entered the
/// reference at raw weight 1.0 (`records/track_1_short/2024-11-10_UNetDoubleLr/
/// c87bb826-797b-4f37-98c7-d3a5dad2de74.txt:244`) and were re-parameterised to
/// `σ(logit)` with this init in `records/track_1_short/2025-11-18_RefineSkip/
/// 00f4e1e6-0044-4a08-b88a-3b7ec0624081.txt:953-954`, which the current
/// `train_gpt.py:1346` still carries verbatim (`-1.5 * torch.ones(1),  # skip_lambda ->
/// σ(-1.5) ≈ 0.18`). σ(-1.5) = 0.18243: eight layers whose decoder half adds an encoder
/// output at weight 1 would double the early residual stream before a single gradient step,
/// and the sigmoid keeps the gate in (0, 1) however far the logit travels.
const SKIP_LOGIT_INIT: f64 = -1.5;
/// Value-residual mixing weight at init, from modded-nanogpt's learnable form
/// (`records/track_1_short/2024-11-06_ShortcutsTweaks/43f60c4f-0448-4de7-83d9-643ca26f61e7.txt`
/// `:168`, `self.lamb = nn.Parameter(torch.tensor(0.5))`): a RAW per-layer scalar - no sigmoid,
/// not shared - carrying the fixed 0.5 of the original value residual (arXiv:2410.17897) as its
/// starting point.
const VALUE_LAMBDA_INIT: f64 = 0.5;

/// Whether the per-layer embedding re-injection EXISTS.
///
/// [`Self::Disabled`] is not a lambda pinned at zero: the `lambdas.x0` bank is never registered
/// in the varstore, no `addcmul` against the embedding is issued, and the checkpoint carries a
/// different FORMAT stamp because its parameter set is smaller (`runner::format_stamp`). The
/// mode is a diagnostic: the shortcut hands every layer the patch embedding directly, which is
/// the cheapest possible route to fitting the training period's own drift, so a run that
/// generalizes worse WITH the shortcut than without it is evidence about the shortcut rather
/// than about the eight layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum X0Lambdas {
    /// One learned scale per layer, init 0, as modded-nanogpt has it.
    #[default]
    Enabled,
    /// No bank, no injection, `layers` fewer trained scalars.
    Disabled,
}

impl X0Lambdas {
    pub fn enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

/// How the training objective weights the `pred_len` forecast horizons.
///
/// The head emits every horizon from every causal token and the objective used to average all
/// of them with equal weight. Jobs 5190-5193 measured what that costs: on the held-out sample
/// between step 2k and 5k the market-neutral MSE ratio at h = 1 and h = 8 kept IMPROVING
/// (0.9636 -> 0.9602, 0.9458 -> 0.9256) while h = 64 and h = 192 rotted (0.9886 -> 1.0688,
/// 1.0257 -> 1.0854), calibration stayed nominal, and the mis-scaling cross term inverted along
/// the horizon axis. One trunk plus one equal-weighted loss means the long horizons'
/// memorization of the training period dominates both the gradient and the scalar that selects
/// checkpoints. This knob is the dial that decides how much of the objective the unlearnable
/// end of the horizon axis is allowed to own.
///
/// Every mode is normalized to mean 1 over ALL `pred_len` horizons (`Σ w = pred_len`), so
/// [`Self::Uniform`] is the weight vector `1` exactly and reproduces the pre-knob loss
/// bit-for-bit. The loss additionally divides by `Σ w·mask·CHANNELS` rather than by
/// `Σ mask·CHANNELS`, which makes the reported NLL a weighted MEAN of per-element NLL - still
/// nats per bar, still comparable across modes, and invariant to the overall scale of `w`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HorizonLoss {
    /// Today's objective: every horizon weighted 1. The control arm.
    #[default]
    Uniform,
    /// `1/√h`, normalized. The gentler of the two decays: h = 192 still carries 1/13.9 of
    /// h = 1's weight, so the long end contributes but cannot dominate.
    InverseSqrt,
    /// `1/h`, normalized. The aggressive decay: h = 192 carries 1/192 of h = 1's weight. The
    /// correct decay rate is the open question and these two bracket it.
    Inverse,
    /// Train ONLY horizons `1..=K`; every horizon above `K` gets weight exactly 0, so no
    /// gradient reaches the head rows that are exclusive to it. The forecasts are still
    /// EMITTED for all `pred_len` horizons - the loss is masked, the head is not shrunk - so
    /// every per-horizon diagnostic keeps reading the untrained end and the run pays the full
    /// head cost. See [`ModelConfig::step_cost`] for what that costs.
    Cutoff(i64),
}

impl HorizonLoss {
    /// The effective per-horizon weight vector, index `i` being horizon `i + 1`, normalized to
    /// mean 1 over all `pred_len` horizons. [`Self::Uniform`] returns exactly `1.0` in every
    /// slot: `pred_len` ones sum exactly in fp64 and the division by `pred_len` is exact, so
    /// the control arm's loss is bit-identical to the unweighted one.
    pub fn weights(self, pred_len: i64) -> Vec<f64> {
        assert!(pred_len > 0, "weights need a positive horizon");
        let raw: Vec<f64> = (1..=pred_len)
            .map(|step| match self {
                Self::Uniform => 1.0,
                Self::InverseSqrt => (step as f64).sqrt().recip(),
                Self::Inverse => (step as f64).recip(),
                Self::Cutoff(cut) => f64::from(u8::from(step <= cut)),
            })
            .collect();
        let mean = raw.iter().sum::<f64>() / pred_len as f64;
        raw.into_iter().map(|weight| weight / mean).collect()
    }

    /// The horizons that carry nonzero weight, for the cost accounting: `Cutoff` still runs the
    /// head over all `pred_len` of them.
    pub fn trained_horizons(self, pred_len: i64) -> i64 {
        match self {
            Self::Cutoff(cut) => cut.min(pred_len),
            _ => pred_len,
        }
    }
}

impl fmt::Display for HorizonLoss {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uniform => formatter.write_str("uniform"),
            Self::InverseSqrt => formatter.write_str("inv-sqrt"),
            Self::Inverse => formatter.write_str("inv"),
            Self::Cutoff(cut) => write!(formatter, "cutoff:{cut}"),
        }
    }
}

impl FromStr for HorizonLoss {
    type Err = anyhow::Error;
    /// `uniform` | `inv-sqrt` | `inv` | `cutoff:K`. `K` is validated to be positive HERE and
    /// against `pred_len` in [`ModelConfig::validate`], which runs before the corpus loads.
    fn from_str(spec: &str) -> Result<Self> {
        match spec {
            "uniform" => return Ok(Self::Uniform),
            "inv-sqrt" => return Ok(Self::InverseSqrt),
            "inv" => return Ok(Self::Inverse),
            _ => {}
        }
        let cut = spec.strip_prefix("cutoff:").ok_or_else(|| {
            anyhow::anyhow!(
                "unknown horizon loss {spec:?}; expected uniform, inv-sqrt, inv or cutoff:K"
            )
        })?;
        let cut: i64 = cut
            .parse()
            .map_err(|_| anyhow::anyhow!("cutoff:K needs an integer K, got {cut:?}"))?;
        ensure!(
            cut >= 1,
            "cutoff:{cut} trains no horizon at all; K must be at least 1"
        );
        Ok(Self::Cutoff(cut))
    }
}

impl Serialize for HorizonLoss {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for HorizonLoss {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let spec = String::deserialize(deserializer)?;
        spec.parse().map_err(serde::de::Error::custom)
    }
}

/// How the conditional MEAN is parameterized along the forecast horizon.
///
/// [`HorizonLoss`] changes what the objective WEIGHTS; this changes what the head can EXPRESS,
/// and it is the only knob here that removes parameters. The measurement it answers is the one
/// in [`HorizonLoss`]'s own doc: between step 2k and 5k the held-out market-neutral MSE ratio
/// kept improving at h = 1 and h = 8 while h = 64 and h = 192 rotted past persistence, and the
/// mis-scaling cross term INVERTED along the horizon axis (h = 8: -0.108 -> -0.004, h = 64:
/// -0.007 -> -0.073) while coverage stayed nominal. A mean whose amplitude grows away from its
/// target only at the long end, with a calibrated scale, is a mean that is fitting the training
/// period's own realized path. The 192 free means per channel are the degrees of freedom that
/// let it: each long horizon has its own row of head weights and nothing ties it to its
/// neighbours.
///
/// # The basis
///
/// Under [`Self::Basis`] horizons `1..=free` keep an independent mean each - that is the band
/// where out-of-period skill is DEMONSTRATED, so it is left alone - and the `pred_len - free`
/// horizons above it are expressed as
///
/// ```text
/// mean(channel, h) = Σ_b coefficient(channel, b) · φ_b(h - free),   h > free
/// φ_b(u)           = exp(-u / τ_b) / rms_b,   τ_b = free · (pred_len/free)^(b/(B-1))
/// rms_b            = √( mean_u exp(-2u / τ_b)² )      (unit RMS over the restricted band)
/// ```
///
/// so `functions` coefficients per channel replace `pred_len - free` free means. The choice was
/// fixed BEFORE any validation number was read, and it is this and not something adaptive for
/// four reasons:
///
/// - **It is the shape a term structure of expected return actually has.** The close coordinate
///   is in units of the h-step persistence deviation σ√h, so it is a signal-to-noise ratio per
///   horizon, not a price. A predictable component with half-life `T` contributes an expected
///   cumulative return whose σ√h-normalized profile decays like `exp(-h/T)`. A sum of `B`
///   exponentials with geometrically spaced timescales is exactly a discretized mixture over
///   half-lives, and by Bernstein's theorem every completely monotone decay - power laws
///   included - is such a mixture, so the span covers the plausible term structures rather than
///   one hand-picked curve.
/// - **It cannot express per-horizon idiosyncrasy.** A nonzero real exponential sum
///   `Σ_b c_b e^{-u/τ_b}` with distinct `τ_b` has at most `B - 1` zeros in `u` (the
///   Descartes rule for exponential sums), so ANY mean this span can emit changes sign at most
///   `functions - 1` times across the whole restricted band. A per-horizon wiggle needs one
///   sign change per horizon. That is the entire restriction, and
///   `the_basis_represents_a_term_structure_and_refuses_per_horizon_idiosyncrasy` is the test
///   that holds it to it rather than the claim.
/// - **Zero coefficients are exactly persistence.** The expansion is linear and homogeneous:
///   no intercept function, no additive constant. `coefficient = 0` gives `mean = 0` at every
///   restricted horizon, which [`decode_joint`] maps to the origin close - so the zero-init
///   head still starts at the persistence forecast, bit for bit, exactly as it does under
///   [`Self::Free`].
/// - **The timescale grid spans the band that is left to explain.** The fastest is `free`
///   itself: anything decaying faster than the free band's own width is already representable
///   there. The slowest is `pred_len`, which over the restricted band decays from 0.995 to
///   0.383 - a drift, and deliberately not a constant, since a flat level in σ√h units is a
///   cumulative return growing like √h forever. Unit-RMS scaling makes every coefficient read
///   in the same units (σ√h of mean at the tail) without touching the span or the monotone
///   decay of any function.
///
/// The log predictive scales are NOT restricted: they stay one free parameter per horizon per
/// channel. Coverage at 5k was 0.678/0.923 against nominal 0.683/0.950, so the distribution's
/// shape is not the measured failure and restricting it would trade a working part of the model
/// for nothing.
///
/// The one cost of insisting on monotone functions rather than an orthonormal span: the
/// columns overlap heavily near `h = free + 1`, so the coefficient Gram matrix is
/// ill-conditioned and reaching the last few digits of a fitted term structure needs large
/// cancelling coefficients. Orthonormalizing would fix that and lose the per-function
/// monotonicity, and it buys nothing here - the shipped fp32 head emits a hyperbolic term
/// structure to 0.75% relative
/// (`the_basis_represents_a_term_structure_and_refuses_per_horizon_idiosyncrasy`), which is
/// three orders of magnitude below the amplitude error being diagnosed, and AdamW's per-
/// parameter normalization makes it indifferent to the column scaling anyway.
///
/// The likelihood is unchanged and still consumes all `pred_len` means: the head EMITS the
/// dense `[2·CHANNELS, pred_len]` block in both modes ([`CausalPatchModel::head`] folds the
/// expansion onto the output weight), so no dense-target traffic disappears. See
/// [`ModelConfig::step_cost`] for what the fold costs and what it does not save.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HorizonMean {
    /// One independent mean per horizon per channel: `pred_len · 2·CHANNELS` head outputs. The
    /// control arm, and bit-for-bit the pre-knob head.
    #[default]
    Free,
    /// Independent means for horizons `1..=free`; `functions` basis coefficients per channel
    /// for everything above. The free means those coefficients replace do not exist - they are
    /// absent from the varstore, not masked, not frozen.
    Basis { free: i64, functions: i64 },
    /// Per-BAR increment means - `CHANNELS · pred_len` of them, full rank, the cumulative
    /// forecast recovered by a cumsum at the decode - plus `scales` state-dependent log-scale
    /// coefficients per channel over a smooth basis in `ln h`.
    ///
    /// This is not a rank restriction on the mean: the difference operator is square and
    /// invertible, so the emitted 192-horizon forecast keeps every degree of freedom it has
    /// today. What changes is the SPACE THE LOSS IS COMPUTED IN. The `h = 192` cumulative
    /// target is a 192-bar overlapping sum, so the 2.46 M rows carrying it rest on ~504
    /// independent draws; the `j = 192` increment target is a single non-overlapping calendar
    /// bar and rests on all of them. Supervising increments moves the long-horizon evidence
    /// without moving what the model can represent.
    ///
    /// `scales` is the one real restriction, and it is in the UNCERTAINTY channel: today's head
    /// emits 192 state-dependent log scales per channel and this emits `scales` coefficients
    /// over [`Self::scale_basis`]. `scales = pred_len` recovers today's freedom exactly, which
    /// is what makes the mean-side and scale-side changes separable knobs rather than one
    /// bundled arm.
    Increment { scales: i64 },
}

impl HorizonMean {
    /// Horizons that keep an independent mean. All of them under [`Self::Free`], and all of
    /// them under [`Self::Increment`] too - a difference is square and invertible, so nothing
    /// about the mean's freedom is reduced there.
    pub fn free_horizons(self, pred_len: i64) -> i64 {
        match self {
            Self::Free | Self::Increment { .. } => pred_len,
            Self::Basis { free, .. } => free,
        }
    }

    /// The decay timescales in bars, fastest first: `free · (pred_len/free)^(b/(B-1))`, a
    /// geometric grid from the free band's own width to the whole forecast horizon. Empty under
    /// [`Self::Free`]. A single function takes the SLOWEST end - one function is a drift, and
    /// the fast decays are what the free band already covers.
    pub fn timescales(self, pred_len: i64) -> Vec<f64> {
        let Self::Basis { free, functions } = self else {
            return Vec::new();
        };
        let (fastest, slowest) = (free as f64, pred_len as f64);
        (0..functions)
            .map(|index| {
                if functions == 1 {
                    slowest
                } else {
                    fastest * (slowest / fastest).powf(index as f64 / (functions - 1) as f64)
                }
            })
            .collect()
    }

    /// The `[pred_len - free, functions]` row-major basis, unit RMS per column. `None` under
    /// [`Self::Free`]. fp64 throughout: it is built once per model, and the fp32 cast is the
    /// last thing that happens to it.
    pub fn basis(self, pred_len: i64) -> Option<Vec<f64>> {
        let Self::Basis { free, functions } = self else {
            return None;
        };
        let restricted = pred_len - free;
        assert!(
            restricted > 0 && functions > 0,
            "an unvalidated horizon mean reached the basis: {self} against pred_len {pred_len}"
        );
        let mut matrix = vec![0.0; (restricted * functions) as usize];
        for (index, tau) in self.timescales(pred_len).into_iter().enumerate() {
            let column: Vec<f64> = (1..=restricted)
                .map(|step| (-(step as f64) / tau).exp())
                .collect();
            let rms = (column.iter().map(|value| value * value).sum::<f64>()
                / restricted as f64)
                .sqrt();
            for (step, value) in column.into_iter().enumerate() {
                matrix[step * functions as usize + index] = value / rms;
            }
        }
        Some(matrix)
    }

    /// The head's output width: what the projection PARAMETERIZES, not what the model emits.
    /// Under [`Self::Basis`] the means cost `CHANNELS·(free + functions)` and the untouched log
    /// scales `CHANNELS·pred_len`. Under [`Self::Increment`] the means cost `CHANNELS·pred_len`
    /// - unreduced - and the log scales `CHANNELS·scales`, which is where the whole width
    /// saving is: at `scales = 8` the projection emits 800 instead of 1536.
    pub fn head_outputs(self, pred_len: i64) -> i64 {
        match self {
            Self::Free => pred_len * OUTPUTS_PER_BAR,
            Self::Basis { free, functions } => CHANNELS * (free + functions + pred_len),
            Self::Increment { scales } => CHANNELS * (pred_len + scales),
        }
    }

    /// The `[scales, pred_len]` row-major log-scale basis `φ_k(h) = cos(k·π·ln h / ln pred_len)`,
    /// `None` outside [`Self::Increment`]. fp64 until the single fp32 cast at construction.
    ///
    /// A cosine family in `ln h` and not in `h`, because dispersion structure over a 192-bar
    /// horizon is a phenomenon of ORDERS OF MAGNITUDE - the interesting variation between one
    /// bar and eight is the same size as the variation between 24 and 192, and a basis linear
    /// in `h` spends almost all of its resolution on the flat far end. `φ_0 ≡ 1` is the level,
    /// `φ_1` is monotone in `ln h` and is exactly the slope mode a mis-strengthened aggregation
    /// needs, and the family is a DCT-II grid on `ln h`, so `scales = pred_len` spans the full
    /// 192-dimensional space and recovers today's per-horizon freedom rather than approximating
    /// it.
    pub fn scale_basis(self, pred_len: i64) -> Option<Vec<f64>> {
        let Self::Increment { scales } = self else {
            return None;
        };
        assert!(
            scales > 0 && pred_len > 0,
            "an unvalidated horizon mean reached the scale basis: {self} against pred_len \
             {pred_len}"
        );
        let span = (pred_len as f64).ln().max(f64::MIN_POSITIVE);
        let mut matrix = vec![0.0; (scales * pred_len) as usize];
        for function in 0..scales as usize {
            for step in 0..pred_len as usize {
                let position = ((step + 1) as f64).ln() / span;
                matrix[function * pred_len as usize + step] =
                    (function as f64 * std::f64::consts::PI * position).cos();
            }
        }
        Some(matrix)
    }

    /// The `[2·CHANNELS·pred_len, head_outputs]` row-major expansion `Ψ`, mapping the head's
    /// parameters to the dense channel-major block the loss consumes. `None` under
    /// [`Self::Free`], which needs no expansion at all.
    ///
    /// Rows are the dense block's own layout (`row = channel·pred_len + h`, channel-major, the
    /// four coordinates then the four log scales). A free-mean row and a log-scale row are
    /// one-hot, so the fold copies them bit-exactly; a restricted row carries that horizon's
    /// `functions` basis values.
    pub fn expansion(self, pred_len: i64) -> Option<Vec<f32>> {
        let Self::Basis { free, functions } = self else {
            return None;
        };
        let basis = self.basis(pred_len)?;
        let outputs = self.head_outputs(pred_len);
        let stride = outputs as usize;
        let mut matrix = vec![0f32; (OUTPUTS_PER_BAR * pred_len) as usize * stride];
        let per_channel = free + functions;
        for channel in 0..CHANNELS {
            let base = (channel * per_channel) as usize;
            for step in 0..free {
                let row = (channel * pred_len + step) as usize;
                matrix[row * stride + base + step as usize] = 1.0;
            }
            for step in 0..pred_len - free {
                let row = (channel * pred_len + free + step) as usize;
                for function in 0..functions {
                    matrix[row * stride + base + (free + function) as usize] =
                        basis[(step * functions + function) as usize] as f32;
                }
            }
        }
        let scales = CHANNELS * per_channel;
        for channel in 0..CHANNELS {
            for step in 0..pred_len {
                let row = ((CHANNELS + channel) * pred_len + step) as usize;
                matrix[row * stride + (scales + channel * pred_len + step) as usize] = 1.0;
            }
        }
        Some(matrix)
    }

    /// The kebab fragment the checkpoint FORMAT carries. The parameter SET differs between
    /// modes and between `(free, functions)` pairs, so this is part of the stamp, not a note.
    pub fn stamp(self) -> String {
        match self {
            Self::Free => "free".to_owned(),
            Self::Basis { free, functions } => format!("basis-{free}-{functions}"),
            Self::Increment { scales } => format!("increment-{scales}"),
        }
    }
}

impl fmt::Display for HorizonMean {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Free => formatter.write_str("free"),
            Self::Basis { free, functions } => write!(formatter, "basis:{free}:{functions}"),
            Self::Increment { scales } => write!(formatter, "increment:{scales}"),
        }
    }
}

impl FromStr for HorizonMean {
    type Err = anyhow::Error;
    /// `free` | `basis:S:B` | `increment:K`. Positivity is validated HERE and the bounds against
    /// `pred_len` in [`ModelConfig::validate`], which runs before the corpus loads.
    fn from_str(spec: &str) -> Result<Self> {
        if spec == "free" {
            return Ok(Self::Free);
        }
        if let Some(scales) = spec.strip_prefix("increment:") {
            let scales: i64 = scales.parse().map_err(|_| {
                anyhow::anyhow!("increment:K needs an integer K, got {scales:?}")
            })?;
            ensure!(
                scales >= 1,
                "increment:{scales} leaves the log scale no state dependence at all, not even a \
                 level; K must be at least 1"
            );
            return Ok(Self::Increment { scales });
        }
        let restricted = spec.strip_prefix("basis:").ok_or_else(|| {
            anyhow::anyhow!("unknown horizon mean {spec:?}; expected free, basis:S:B or increment:K")
        })?;
        let (free, functions) = restricted
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("basis:S:B needs both S and B, got {restricted:?}"))?;
        let parse = |name: &str, value: &str| -> Result<i64> {
            value
                .parse()
                .map_err(|_| anyhow::anyhow!("basis:S:B needs an integer {name}, got {value:?}"))
        };
        let (free, functions) = (parse("S", free)?, parse("B", functions)?);
        ensure!(
            free >= 1,
            "basis:{free}:{functions} restricts every horizon including h = 1, where skill is \
             demonstrated; S must be at least 1"
        );
        ensure!(
            functions >= 1,
            "basis:{free}:{functions} leaves the restricted horizons no freedom at all, which \
             is `--horizon-loss cutoff:{free}` with a dead head, not a structured mean; B must \
             be at least 1"
        );
        Ok(Self::Basis { free, functions })
    }
}

impl Serialize for HorizonMean {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for HorizonMean {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let spec = String::deserialize(deserializer)?;
        spec.parse().map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, clap::Args)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    #[arg(long, default_value_t = 6000)]
    pub seq_len: i64,
    #[arg(long, default_value_t = 192)]
    pub pred_len: i64,
    #[arg(long, default_value_t = 16)]
    pub patch_len: i64,
    #[arg(long, default_value_t = 8)]
    pub layers: usize,
    #[arg(long, default_value_t = 512)]
    pub d_model: i64,
    #[arg(long, default_value_t = 8)]
    pub heads: i64,
    #[arg(long, default_value_t = 2048)]
    pub ffn: i64,
    #[arg(long, default_value_t = 0.0)]
    pub dropout: f64,
    /// Origins with fewer valid history bars are excluded from the loss.
    #[arg(long, default_value_t = 256)]
    pub min_history: i64,
    /// Comma-separated exogenous variates: time-of-day, day-of-week, session-gap, volume,
    /// market, spy, dispersion, cross-section-z; `all` or `none`.
    #[arg(long, default_value_t = FeatureSet::ALL)]
    pub features: FeatureSet,
    /// Whether the per-layer embedding re-injection exists; `disabled` removes the parameters
    /// and the kernel, and changes the checkpoint FORMAT stamp.
    #[arg(long, value_enum, default_value_t = X0Lambdas::Enabled)]
    pub x0_lambdas: X0Lambdas,
    /// How the objective weights the horizon axis: `uniform`, `inv-sqrt`, `inv` or `cutoff:K`.
    /// This is the OBJECTIVE, not the parameter set - every mode trains the same tensors and
    /// emits the same 192 forecasts.
    #[arg(long, default_value_t = HorizonLoss::Uniform)]
    pub horizon_loss: HorizonLoss,
    /// How the conditional mean is parameterized along the horizon axis: `free`,
    /// `basis:S:B` for independent means up to `S` and `B` fixed decaying basis functions per
    /// channel above it, or `increment:K` for full-rank per-BAR means with `K` state-dependent
    /// log-scale coefficients per channel. This IS the parameter set - `basis` deletes mean
    /// rows and `increment` deletes log-scale rows - so it moves the checkpoint FORMAT stamp.
    #[arg(long, default_value_t = HorizonMean::Free)]
    pub horizon_mean: HorizonMean,
    /// `λ` for the training-time amplitude prior, `0` (the control) to disable it entirely.
    ///
    /// This penalizes the ENERGY of the predicted mean function on the σ-scaled close
    /// coordinate the decoder already materializes - see
    /// [`CausalPatchModel::amplitude_prior`]. It is NOT a parameter, NOT a multiplier on the
    /// output and NOT a horizon reweighting: a fixed output multiplier is pure
    /// reparameterization that the upstream layers undo within a few hundred steps, which is
    /// exactly why `--horizon-mean basis:8:8` failed as an amplitude intervention. A penalty
    /// on output VALUES cannot be undone that way, because growing an upstream weight grows
    /// the penalty proportionally.
    ///
    /// Skipped from the manifest at `0`, so a control checkpoint written before this knob
    /// existed serializes - and therefore digests - byte-identically.
    #[arg(long, default_value_t = 0.0)]
    #[serde(default, skip_serializing_if = "no_amplitude_prior")]
    pub amplitude_prior: f64,
    /// Which orthonormal map the objective measures horizon error in: `cumulative` (the
    /// identity, and the control), `haar` or `dct`.
    ///
    /// This is neither the objective's WEIGHTING nor its parameter set - it is the METRIC. `W`
    /// is full rank and orthonormal, hence a bijection, so every forecast function the head
    /// could emit before it can emit after, at the same head shape and the same parameter
    /// count. That is the whole difference from `--horizon-mean basis:8:8`, which restricted
    /// the RANK of the emitted mean and destroyed the correlation gain.
    ///
    /// Under a non-identity basis the head's four log-scale rows are reinterpreted as
    /// per-COEFFICIENT scales, so the Gaussian stays diagonal in the space it is modelled in
    /// and horizon-space `σ_h` becomes DERIVED - see
    /// [`target_basis::BasisTransform::horizon_log_scale`]. `cumulative` takes the fused loss
    /// path untouched and is bit-for-bit the pre-knob objective.
    #[arg(long, default_value_t = TargetBasis::Cumulative)]
    #[serde(default, skip_serializing_if = "is_cumulative_basis")]
    pub target_basis: TargetBasis,
    /// How the objective weights the COEFFICIENT axis: `uniform` (equal weight per unit of
    /// target variance once whitened) or `snr` (by measured `ρ̂²` on the [70%,80%) partition).
    /// Mean 1 either way, so the loss stays nats per bar against the 2.3947663 anchor.
    #[arg(long, default_value_t = BasisWeight::Uniform)]
    #[serde(default, skip_serializing_if = "is_uniform_basis_weight")]
    pub basis_weight: BasisWeight,
    /// The authenticated per-coefficient statistics artifact: whitening scales fitted on
    /// TRAINING origins and, when `--basis-weight snr`, `ρ̂` fitted on the [70%,80%) partition.
    /// Absent means the analytic persistence prior `diag(W·C·Wᵀ)` and no measured whitening.
    #[arg(long)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub basis_stats: Option<std::path::PathBuf>,
}

/// `--amplitude-prior 0` is the control, and a control manifest must serialize exactly as it
/// did before the knob existed or its authenticating digest moves and every checkpoint written
/// before today stops loading.
fn no_amplitude_prior(lambda: &f64) -> bool {
    *lambda == 0.
}

/// The control basis, skipped from the manifest for the same reason `--amplitude-prior 0` is:
/// a control checkpoint must serialize, and therefore digest, exactly as it did before the knob.
fn is_cumulative_basis(basis: &TargetBasis) -> bool {
    basis.is_identity()
}

fn is_uniform_basis_weight(weight: &BasisWeight) -> bool {
    *weight == BasisWeight::Uniform
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            seq_len: 6000,
            pred_len: 192,
            patch_len: 16,
            layers: 8,
            d_model: 512,
            heads: 8,
            ffn: 2048,
            dropout: 0.0,
            min_history: 256,
            features: FeatureSet::ALL,
            x0_lambdas: X0Lambdas::Enabled,
            horizon_loss: HorizonLoss::Uniform,
            horizon_mean: HorizonMean::Free,
            amplitude_prior: 0.0,
            target_basis: TargetBasis::Cumulative,
            basis_weight: BasisWeight::Uniform,
            basis_stats: None,
        }
    }
}

/// Arithmetic and traffic cost of one training step, forward plus backward.
///
/// `traffic_bytes` is a LOWER BOUND: it charges every materialized activation one read and one
/// write per pass, so the ratio of the measured step time to `traffic_bytes / peak bandwidth` is
/// the multiplier that elementwise re-reads and gradient plumbing add on top. That ratio, not
/// the bound itself, is what the fusion work moves.
#[derive(Debug, Clone, Copy)]
pub struct StepCost {
    pub matmul_flops: f64,
    pub traffic_bytes: f64,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.seq_len > 0 && self.pred_len > 0,
            "sequence lengths must be positive"
        );
        ensure!(
            self.patch_len > 0 && self.seq_len % self.patch_len == 0,
            "patches must cover the entire context exactly"
        );
        ensure!(
            self.heads > 0 && self.d_model > 0 && self.d_model % self.heads == 0,
            "model width must be divisible by attention heads"
        );
        ensure!(
            (self.d_model / self.heads) % 2 == 0,
            "rotary attention needs an even head dimension"
        );
        ensure!(
            self.layers > 0 && self.ffn > 0,
            "layer count and feedforward width must be positive"
        );
        ensure!(
            self.layers % 2 == 0,
            "the U-net skip stack pairs each encoder layer with one decoder layer, so the layer count must be even"
        );
        ensure!(
            self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout),
            "dropout must be in [0, 1)"
        );
        ensure!(
            self.amplitude_prior.is_finite() && self.amplitude_prior >= 0.,
            "--amplitude-prior must be finite and non-negative, got {}; a negative penalty \
             REWARDS forecast energy and diverges",
            self.amplitude_prior
        );
        ensure!(
            (2..=self.seq_len).contains(&self.min_history),
            "min_history must lie in [2, seq_len]"
        );
        if let HorizonLoss::Cutoff(cut) = self.horizon_loss {
            ensure!(
                (1..=self.pred_len).contains(&cut),
                "--horizon-loss cutoff:{cut} is outside the 1..={} forecast horizon",
                self.pred_len
            );
        }
        if let HorizonMean::Basis { free, functions } = self.horizon_mean {
            ensure!(
                (1..self.pred_len).contains(&free),
                "--horizon-mean basis:{free}:{functions} has no restricted horizon to structure: \
                 S must leave at least one of the {} forecast horizons above it",
                self.pred_len
            );
            ensure!(
                (1..=self.pred_len - free).contains(&functions),
                "--horizon-mean basis:{free}:{functions} asks for more basis functions than the \
                 {} horizons they replace, which restricts nothing and only adds parameters",
                self.pred_len - free
            );
        }
        if let HorizonMean::Increment { scales } = self.horizon_mean {
            ensure!(
                (1..=self.pred_len).contains(&scales),
                "--horizon-mean increment:{scales} asks for more log-scale coefficients than the \
                 {} horizons they parameterize; K = {} already spans the space exactly and \
                 recovers the per-horizon head",
                self.pred_len,
                self.pred_len
            );
        }
        if !self.target_basis.is_identity() {
            // A per-HORIZON weight vector is meaningless once the objective's rows are
            // coefficients: `w_h` would multiply coefficient `h`, which is not horizon `h`.
            // The coefficient weight is `--basis-weight`, and stacking the two would produce a
            // loss nobody could attribute.
            ensure!(
                self.horizon_loss == HorizonLoss::Uniform,
                "--target-basis {} rotates the objective's rows into coefficients, so \
                 --horizon-loss {} would weight coefficient k by horizon k's weight, which is \
                 not the same axis; weight the coefficient axis with --basis-weight instead",
                self.target_basis,
                self.horizon_loss
            );
            // Two different refusals wearing one `ensure!`. `basis` restricts the RANK of the
            // emitted mean, and composing a rank restriction with a full-rank rotation makes
            // any effect attributable to either. `increment` is not a rank restriction at all -
            // the difference operator `D` is square and invertible - but `W·D` is neither
            // orthonormal nor a difference, so a measurement through the composition is a
            // measurement of neither idea. One knob per axis or neither.
            ensure!(
                self.horizon_mean == HorizonMean::Free,
                "--target-basis {} is a full-rank orthonormal rotation and --horizon-mean {} \
                 reparametrizes the same horizon axis; their composition is neither orthonormal \
                 nor a difference, so running both makes neither attributable",
                self.target_basis,
                self.horizon_mean
            );
            ensure!(
                self.pred_len >= 3,
                "--target-basis {} needs at least three horizons to rotate, got {}",
                self.target_basis,
                self.pred_len
            );
        }
        if self.basis_weight == BasisWeight::Snr {
            ensure!(
                self.basis_stats.is_some(),
                "--basis-weight snr needs --basis-stats to name the artifact carrying the \
                 measured ρ̂ per coefficient; it is fitted on the [70%,80%) partition"
            );
        }
        Ok(())
    }

    pub fn origins(&self) -> i64 {
        self.seq_len / self.patch_len
    }

    /// `(encoder source, decoder destination)` for every U-net skip, deepest encoder layer
    /// first: at eight layers this is `3->4, 2->5, 1->6, 0->7`. The order IS the stack
    /// discipline - encoder layers push in index order, decoder layers pop - and the `k`-th
    /// pair is gated by `skip_weights[k]`, matching the reference's `skip_weights[i - n]`
    /// indexing (`records/track_1_short/2024-11-10_UNetDoubleLr/
    /// c87bb826-797b-4f37-98c7-d3a5dad2de74.txt:266-270`).
    pub fn skip_pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let encoder = self.layers / 2;
        (0..self.layers - encoder).map(move |k| (encoder - 1 - k, encoder + k))
    }

    /// Arithmetic and traffic cost of one training step at `batch`, from shapes alone.
    pub fn step_cost(&self, batch: usize) -> StepCost {
        let (rows, origins, horizon) = (batch as f64, self.origins() as f64, self.pred_len as f64);
        let tokens = rows * origins;
        let (width, ffn) = (self.d_model as f64, self.ffn as f64);
        let aux = self.features.channels() as f64;
        let known = self
            .features
            .channel_mask(Feature::known_future)
            .iter()
            .filter(|known| **known)
            .count() as f64;
        let covariate_width = if known > 0. { COVARIATE_WIDTH as f64 } else { 0. };
        let head_input = width + covariate_width;
        // What the head EMITS, in both modes: the expansion is folded onto the output weight,
        // so the token-space GEMM and every activation below are shape-identical under
        // `basis:S:B`. What it PARAMETERIZES is `head_parameters`, and the gap between the two
        // is the whole restriction.
        let head_outputs = horizon * OUTPUTS_PER_BAR as f64;
        let head_parameters = self.horizon_mean.head_outputs(self.pred_len) as f64;
        let gemm = |rows: f64, reduce: f64, out: f64| 2. * rows * reduce * out;
        let mut flops = gemm(
            tokens,
            self.patch_len as f64 * (CHANNELS as f64 + aux),
            width,
        );
        // Causal SDPA: `QKᵀ` and `AV` are each `2·rows·origins²·d_model` dense and half of that
        // under the mask, so the pair costs one dense GEMM's worth.
        flops += self.layers as f64
            * (gemm(tokens, width, 3. * width)
                + gemm(tokens, width, width)
                + 2. * gemm(tokens, width, ffn)
                + 2. * rows * origins * origins * width);
        flops += gemm(tokens, horizon * known, covariate_width)
            + gemm(tokens, head_input, HEAD_HIDDEN as f64)
            + gemm(tokens, HEAD_HIDDEN as f64, head_outputs);
        // Every activation the step materializes, written once and read once. Backward pays it
        // twice more (grad-in, grad-out), hence the factor three on both totals.
        let bf16 = |elements: f64| 2. * elements;
        let fp32 = |elements: f64| 4. * elements;
        let mut bytes = bf16(tokens * self.patch_len as f64 * (CHANNELS as f64 + aux));
        // Per layer: two pre-norms (2·width), the packed projection (3·width), the OUTPUT of the
        // fused QK-norm-plus-rotation (2·width - the only tensor `fused_kernels::qk_norm_rope`
        // materializes; the composed form charged 2·width for the normalized `q‖k` block it had
        // to hand on and 8·width for the rotation's two full-width products, its two
        // half-crossing sums and the buffer that interleaved them), the attention output, its
        // projection and the two residual `addcmul`s (6·width), the x0 injection's second
        // `addcmul` (1·width, and exactly zero when the injection does not exist), and the
        // feedforward PAIR (2·ffn - the up projection and
        // `fused_kernels::relu_square`'s single output; the `relu`-then-`square` composition
        // materialized three, GELU two). The rotation term was missing entirely before this
        // accounting was measured per kernel class, and at 8·width of 22 the bound it produced
        // was 40% low.
        //
        // Against the LayerNorm/GELU form (19·width + 2·ffn) the residual recipe now adds ONE
        // width-unit per layer, the x0 `addcmul`, and nothing else: the QK norm it introduced no
        // longer materializes anything, `square`'s extra `[tokens, ffn]` tensor is gone, and so
        // are six of the rotation's eight width-units. The post-lambdas cost nothing here
        // because they ride the projection weights ([`scaled_linear`]), and the residual scales
        // cost nothing because `addcmul` folds them into the add that was already there.
        //
        // The fp32 `rstd` a materializing RMSNorm writes is not charged here and never was;
        // the fused kernel writes none at all, recomputing the normalization in its backward.
        let x0_injection = if self.x0_lambdas.enabled() { width } else { 0. };
        bytes += self.layers as f64
            * bf16(
                tokens
                    * (2. * width
                        + 3. * width
                        + 2. * width
                        + 6. * width
                        + x0_injection
                        + 2. * ffn),
            );
        // The patch embedding's own norm: `x0 = rms_norm(patch(tokens))`, once per step.
        bytes += bf16(tokens * width);
        // The U-net skip: ONE fused `addcmul` per decoder layer (see `unet_stack`), so each
        // pair materializes a single `width` activation, not the two a separate `gate * skip`
        // product and add would. The encoder half's outputs are free - they are the very
        // tensors the next layer's norm already retains, so the stack itself adds no bytes.
        bytes += (self.layers / 2) as f64 * bf16(tokens * width);
        bytes += bf16(tokens * (width + horizon * known + covariate_width + head_input))
            + bf16(tokens * 2. * HEAD_HIDDEN as f64)
            + bf16(tokens * head_outputs);
        // The head geometry, the NLL and their backward are counted directly rather than
        // scaled: 74 reads-or-writes of the `[rows, origins', 1, pred_len]` fp32 channel
        // space, enumerated op by op from `CausalPatchModel::losses`.
        //
        // FORWARD, 51 units. The mask fold is 2 (read the mask, write `mask·w`; the
        // `[1, 1, 1, pred_len]` weight vector rides in L2). `fused_kernels::loss_geometry`
        // reads 9 - the whole bf16 head space is 8 half-units, the fp32 targets 4, the folded
        // mask 1 - and writes 13, the mean coordinate plus the twelve fp32 vectors the
        // reductions consume. The twelve `dot`s read 24. The `½·ln h` prior and the two
        // denominators reduce the mask 3 more times.
        //
        // BACKWARD, 13 units. ONE kernel: 9 read (head, targets, folded mask) and 4 written
        // (the dense bf16 head gradient). It recomputes the whole geometry from `head` instead
        // of reading anything the forward saved, which is why the forward retains no
        // full-size fp32 tensor and the backward is a fifth of the forward rather than double.
        //
        // Plus 10 for building the targets and the mask, which carry no gradient.
        //
        // The composition this replaced was 278 units - 121 forward and 144 backward over 65
        // and 70 separate kernels - at 1613 and 1470 GB/s measured, already 82-90% of this
        // card's streaming roof. The 3.8x reduction here is entirely deleted passes; not one
        // kernel in the old chain was running slowly.
        //
        // `HorizonLoss::Cutoff(K)` does NOT reduce this term or any FLOP above it. The head
        // still emits all `pred_len` horizons and the geometry still runs over all of them, so
        // a `cutoff:K` arm pays the full head cost and trains `K/pred_len` of it: at K = 32 of
        // 192 that is 5/6 of this term and 5/6 of the head-output GEMM spent on horizons the
        // objective weights at zero. Masking the loss rather than narrowing the head is the
        // deliberate trade - every per-horizon diagnostic keeps reading the untrained end,
        // which is the entire point of the arm.
        let head_loss = fp32(tokens * horizon) * 74.;
        // Value residual: `layers - 1` decoder layers each `lerp` their own value against layer
        // 0's. Ten passes over one `[tokens, d_model]` bf16 activation per such layer - three
        // forward (two reads and one write) and seven backward (`grad_self` and `grad_end` at a
        // read plus a write each, and the three-operand reduction that produces the lambda
        // gradient) - plus the in-place adds that accumulate those contributions into layer 0's
        // value gradient, three passes each. Counted here rather than folded into the per-layer
        // width units because the mix is not a plain write-once-read-once materialization.
        let decoders = (self.layers - 1).max(0) as f64;
        let value_residual =
            bf16(tokens * width) * (10. * decoders + 3. * (decoders - 1.).max(0.));
        // The structured mean's own arithmetic and traffic, and the honest side of the trade.
        //
        // `CausalPatchModel::head` expands the parameters in WEIGHT space: one
        // `[2·CHANNELS·pred_len, head_parameters] × [head_parameters, HEAD_HIDDEN]` GEMM per
        // step, forward plus its weight gradient - two passes, not three, because the
        // expansion is a constant and takes no gradient of its own. At 192/8/8 that is
        // 5.23 GFLOP against the step's 16.98 TFLOP, and the folded weight and its gradient
        // are 3.15 MB each, charged here at four materializations (write and read, forward and
        // backward).
        //
        // What it deliberately does NOT do is shrink the token-space head GEMM, which still
        // emits all `2·CHANNELS·pred_len` outputs from all `HEAD_HIDDEN` hidden units. Doing
        // the expansion in ACTIVATION space instead would cut that GEMM by
        // `3·2·tokens·HEAD_HIDDEN·704` = 0.415 TFLOP at 192/8/8, but it would have to
        // materialize the restricted block and then concatenate it with the free band and the
        // log scales - `+0.9 GB/step` at best, and only after `Head` stops being one tensor
        // and the fused loss stops reading one `split` node. This model is within a factor of
        // two of both roofs, so 2.4% of the arithmetic is not worth 0.6% of the traffic plus a
        // second code path through the loss; more to the point, an ablation arm wants the
        // control's cost profile, not a cheaper one.
        let expansion = if matches!(self.horizon_mean, HorizonMean::Basis { .. }) {
            2. * gemm(head_outputs, head_parameters, HEAD_HIDDEN as f64)
        } else {
            0.
        };
        let expansion_bytes = if matches!(self.horizon_mean, HorizonMean::Basis { .. }) {
            4. * bf16(head_outputs * HEAD_HIDDEN as f64)
        } else {
            0.
        };
        // The training-time amplitude prior, and exactly zero when `λ = 0` - the control's
        // cost bound must not move because a knob it does not use exists. Twenty-two passes
        // over the same `[rows, origins', 1, pred_len]` fp32 close slice: eight forward (the
        // masked multiply's two reads and one write, that product's reduction read, the
        // square-multiply's two reads and one write, and its reduction read) and fourteen
        // backward (one grad slice out of each of the two reductions at a read and a write
        // each, the first moment's broadcast, the two accumulations into the masked product
        // and into the close coordinate, and the mask multiply that routes one back through
        // the other). At 256×375×192 that is 1.62 GB against the step's 143.4 GB - 1.1%, and
        // the largest single reason this prior is not free. It is also the natural second
        // fusion target: the two reductions read a tensor the fused loss already has in
        // registers.
        let amplitude_prior = if self.amplitude_prior > 0. {
            fp32(tokens * horizon) * 22.
        } else {
            0.
        };
        StepCost {
            matmul_flops: 3. * flops + expansion,
            traffic_bytes: 3. * 2. * bytes
                + head_loss
                + value_residual
                + expansion_bytes
                + amplitude_prior,
        }
    }
}

fn projection(path: nn::Path, input: i64, output: i64, bias: bool) -> nn::Linear {
    // Match torch.nn.Linear, including its fan-in initialization.
    let bound = 1.0 / (input as f64).sqrt();
    let init = nn::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    nn::linear(
        path,
        input,
        output,
        nn::LinearConfig {
            ws_init: init,
            bs_init: Some(init),
            bias,
        },
    )
}

/// A projection whose weight AND bias start at exactly zero, so the branch it terminates
/// contributes nothing at step 0.
///
/// modded-nanogpt zero-initialises the MLP DOWN projection (`train_gpt_medium.py:1006`
/// `self.c_proj.zero_()`, `train_gpt.py:1308` `self.mlp_bank[:, 1, :, :].zero_()`, both "zero
/// init suggested by @Grad62304977"). It does NOT zero-initialise the attention output
/// projection, and this comment used to say it did: `train_gpt.py:1293` fills the whole real
/// `vo_bank` - every V and every O - from a uniform, and only the groups padded out to
/// `world_size` are zeroed at `:1294`; `:1161` then uses O straight from that bank.
/// `CastedLinearT`, whose `reset_parameters` is `nn.init.zeros_(self.weight)`
/// (`train_gpt.py:973-975`), is the LM HEAD class (`:1221`), a different module entirely.
///
/// So OUR attention O is a deliberate divergence, not a port: this function is what
/// `Block::new` calls for it, so it starts at exactly zero while the reference's starts
/// uniform. Zero O means no attention branch reaches the residual stream at step 0, which
/// delays the QKV projections' first gradient by however long the O rows take to leave zero.
/// It is recorded as an untried mechanistic port in
/// `research/worker_reports/timexer_nanogpt_ledger.md` (N01) and is NOT changed here.
///
/// Bias-free, like every projection in the reference (`train_gpt.py:1103` and `:1161` call
/// `F.linear` with a weight only; `train_gpt_medium.py:995-996` stores bare weight
/// parameters). Dropping the block biases removes four AdamW parameters per layer from the
/// force-AdamW list and four fp32->bf16 parameter casts per layer, and it is what makes the
/// zero-init actually mean "this branch is off": a nonzero bias would leave a constant
/// per-channel offset on the residual stream at step 0.
fn zeroed_projection(path: nn::Path, input: i64, output: i64) -> nn::Linear {
    nn::linear(
        path,
        input,
        output,
        nn::LinearConfig {
            ws_init: nn::Init::Const(0.0),
            bs_init: None,
            bias: false,
        },
    )
}

/// A bias-free hidden projection at the reference's 2D init scale,
/// `uniform(±√3·0.5·fan_in^-½)` (std `0.5·fan_in^-½`, i.e. 0.866 of the `1/√fan_in` bound
/// `torch.nn.Linear` uses): `train_gpt.py:1288-1291` and `:1304-1307`, "improved init scale by
/// @YouJiacheng and @srashedll", `train_gpt_medium.py:1002-1005`, and in-repo
/// `world_model.rs:2903-2909` (`uniform_init`). Bias-free for the reasons in
/// [`zeroed_projection`].
fn hidden_projection(path: nn::Path, input: i64, output: i64) -> nn::Linear {
    let bound = 3f64.sqrt() * 0.5 / (input as f64).sqrt();
    nn::linear(
        path,
        input,
        output,
        nn::LinearConfig {
            ws_init: nn::Init::Uniform {
                lo: -bound,
                up: bound,
            },
            bs_init: None,
            bias: false,
        },
    )
}

fn linear(input: &Tensor, layer: &nn::Linear) -> Tensor {
    input.linear(
        &layer.ws.to_kind(input.kind()),
        layer.bs.as_ref().map(|bias| bias.to_kind(input.kind())),
    )
}

/// [`linear`] with a scalar folded onto the weight and bias copies the GEMM already needs.
///
/// modded-nanogpt folds its own sub-block scalars onto the projection this way rather than
/// scaling the activation: `train_gpt.py:1161`,
/// `y = F.linear(y, sa_lambdas[1] * qkvo_w[self.dim * 3:].type_as(y))` with the comment
/// "sa_lambdas[1] pre-multiplied to O @shenberg". The product and its gradient are identical
/// either way; a 512×512 weight is 1/96 000 of the `[rows, origins, d_model]` activation it
/// would otherwise multiply, so folding the post-lambda here costs no traffic at all.
fn scaled_linear(input: &Tensor, layer: &nn::Linear, scale: &Tensor) -> Tensor {
    input.linear(
        &(layer.ws.to_kind(input.kind()) * scale),
        layer
            .bs
            .as_ref()
            .map(|bias| bias.to_kind(input.kind()) * scale),
    )
}

/// `x * rsqrt(mean(x²) + eps)` over the last dimension, with NO learnable gain and no bias.
///
/// modded-nanogpt's only normalization, `train_gpt.py:952-953` and
/// `train_gpt_medium.py:839-840`:
///
/// ```python
/// def norm(x: Tensor):
///     return F.rms_norm(x, (x.size(-1),))
/// ```
///
/// `F.rms_norm` with `weight=None` is parameter-free, so this replaces LayerNorm's per-block
/// gain and bias with nothing at all: no 1024-element parameter cast per norm, no mean pass,
/// no gain/bias gradients, and - the reason it matters here - no scalar tensors that could be
/// misrouted into NorMuon's 2D-hidden group. The block's own scale freedom lives in the
/// residual lambdas instead (see [`BlockLambdas`]).
///
/// `eps` is passed explicitly, at the in-repo value: `world_model.rs:109-110`
/// (`BAR_NORM_EPS = 1e-6`, "The norm carries no learnable gain anywhere in this model"), which
/// is also what the reference passes where it passes one at all (`train_gpt.py:1079`).
///
/// `eps=None` does NOT resolve to the input dtype's epsilon. `_fused_rms_norm` takes its
/// default from the fp32 ACCUMULATE type, `FLT_EPSILON = 1.1920929e-7`, so
/// `finfo(bfloat16).eps = 7.8125e-3` never reaches the kernel and no consequence of it is a
/// consequence of the default. This comment used to quote one - a "0.4% shrink" (which is what
/// `1 - √(1/(1 + 7.8125e-3))` is at unit RMS) - and that number described nothing this code
/// does; the same counterfactual at a collapsed RMS reaches 95%, which is how far a
/// dtype-epsilon story can be pushed while staying arithmetically consistent and physically
/// vacuous.
///
/// What passing `1e-6` instead of the default actually changes, as `1 - √(ms/(ms + eps))`:
/// at unit RMS, 5.0e-7 against 6.0e-8, a 4.4e-7 relative gap and far below bf16's 2^-8
/// resolution - indistinguishable, as the reference's own omission implies. On a head block
/// whose RMS has collapsed to 4.2e-3, mean square 1.76e-5, `1e-6` is 5.7% of the mean square:
/// it shrinks by 2.72% where the fp32 default shrinks by 0.34%. That 2.4% is the entire
/// behavioural content of the explicit epsilon, and it is why it is passed - `1e-6` is the
/// floor the rest of the repo normalizes against, and it only bites where the shell has
/// already collapsed.
///
/// `_fused_rms_norm`, NOT `rms_norm`: `rms_norm`'s composite body dispatches to
/// `_fused_rms_norm`, but `rms_norm` itself registers as a math kernel on CUDA, so calling it
/// directly costs an extra dispatch layer and hides the real kernel from the profiler - see
/// `train/pretrain_profile.rs:95-99`, which names this exact trap, and `model/rmsnorm.rs:17`,
/// which is the in-repo form. The second return is the fp32 `rstd`, which the RMSNorm class
/// in [`CausalPatchModel::kernel_classes`] charges for.
fn rms_norm(input: &Tensor) -> Tensor {
    let width = *input
        .size()
        .last()
        .expect("rms_norm needs at least one dimension");
    input
        .internal_fused_rms_norm([width], None::<&Tensor>, Some(NORM_EPS))
        .0
}

/// The residual-mixing scalars one [`Block`] consumes, as 0-dim views of the model-level
/// lambda vectors ([`CausalPatchModel::resid_lambdas`] and friends). Borrowed rather than
/// owned so the whole stack pays one cast and one `unbind` per step, exactly as the reference
/// does (`train_gpt.py:1509-1512`: `self.resid_lambdas[:, 0].bfloat16().unbind(0)`).
struct BlockLambdas<'a> {
    /// Residual-stream scale, `[attention, feedforward]`. `train_gpt.py:1338`.
    resid: [&'a Tensor; 2],
    /// Sub-block output scale, `[attention, feedforward]`. `train_gpt.py:1334`.
    post: [&'a Tensor; 2],
    /// The embedding re-injection as ONE thing: `(x0, λ0)`, the normalized patch embedding and
    /// this layer's scale, applied on the attention residual only (`train_gpt.py:1638` with the
    /// init-0 gate of `:1389`). `None` under [`X0Lambdas::Disabled`] - the embedding is then
    /// simply not an input to the block, so there is no zero-multiplied tensor in the graph.
    x0: Option<(&'a Tensor, &'a Tensor)>,
}

struct Block {
    qkv: nn::Linear,
    output: nn::Linear,
    first: nn::Linear,
    second: nn::Linear,
    /// Value-residual mixing weight, `[1]` fp32, `None` in the SOURCE layer.
    ///
    /// modded-nanogpt's value residual (`records/track_1_short/2024-11-06_ShortcutsTweaks/`
    /// `README.md:16-41`, code at `43f60c4f-0448-4de7-83d9-643ca26f61e7.txt:168,177`) gives every
    /// block a raw scalar `lamb`, initialized to 0.5, and mixes
    /// `v = (1 - lamb)·v + lamb·v_1` against layer 0's value. Layer 0 receives `v1=None` and
    /// sets `v1 = v`, so its own mix is the identity and its `lamb` gets an exactly zero
    /// gradient - a dead parameter. Here that dead scalar simply does not exist, which is also
    /// what makes layer 0 bit-identical to the model without this path.
    value_lambda: Option<Tensor>,
    heads: i64,
    width: i64,
    dropout: f64,
}

impl Block {
    /// `layer` selects the value-residual role: layer 0 publishes its value, every later layer
    /// mixes against it.
    fn new(path: nn::Path, config: &ModelConfig, layer: usize) -> Self {
        let width = config.d_model;
        Self {
            qkv: hidden_projection(&path / "qkv", width, 3 * width),
            output: zeroed_projection(&path / "output", width, width),
            first: hidden_projection(&path / "first", width, config.ffn),
            second: zeroed_projection(&path / "second", config.ffn, width),
            value_lambda: (layer > 0)
                .then(|| path.var("value_lambda", &[1], nn::Init::Const(VALUE_LAMBDA_INIT))),
            heads: config.heads,
            width,
            dropout: config.dropout,
        }
    }

    /// One pre-norm block: `x = λr·x + λp·O(attn(rms(x))) + λ0·x0`, then
    /// `x = λr·x + λp·W2(relu(W1(rms(x)))²)`.
    ///
    /// Pre-norm, matching the reference: `train_gpt_medium.py:1020-1023`
    /// (`x = x + self.attn(norm(x))`, `x = x + self.mlp(norm(x))`) and `train_gpt.py:1598`
    /// (`attn_in_normed = norm(cache.get(7, x))`) / `:1643` (`normed = norm(x)`). The residual
    /// lambda scheme only makes sense on a pre-norm stack - it rescales the raw residual
    /// stream, which a post-norm block would immediately normalize away - and this model was
    /// already pre-norm, so nothing had to be converted.
    ///
    /// The lambda lines are `train_gpt.py:1638` and `:1665`:
    ///
    /// ```python
    /// x = resid_lambdas_attn[i] * x + post_lambdas_attn[i] * attn_out + x0 * x0_gates[i]
    /// x = resid_lambdas_mlp[i] * x + post_lambdas_mlp[i] * ReLUSqrdMLP(normed, *mlp_args)
    /// ```
    ///
    /// with the post-lambdas folded onto the projection weights ([`scaled_linear`]) and the
    /// remaining two products issued as `addcmul`, so each residual line is ONE kernel over
    /// the residual stream instead of a multiply and an add. The attention line costs two when
    /// [`BlockLambdas::x0`] is present and one when it is not; the feedforward line costs
    /// exactly what the old bare `state + ff` cost.
    ///
    /// `first_value` is the SOURCE layer's head-shaped value, `None` in the source layer itself.
    /// Returns the block output beside the value it published, which is `Some` exactly in the
    /// source layer.
    fn forward(
        &self,
        input: &Tensor,
        first_value: Option<&Tensor>,
        lambdas: &BlockLambdas<'_>,
        rotation: (&Tensor, &Tensor),
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        assert_eq!(
            self.value_lambda.is_some(),
            first_value.is_some(),
            "every non-source layer must receive the source layer's value"
        );
        let (batch, length, _) = input.size3().unwrap();
        let head_dim = self.width / self.heads;
        // ONE `split_with_sizes`, not two `narrow`s. ATen's backward for `narrow` is
        // `zeros_like(input)` plus a copy into the slice, so splitting the packed projection
        // with narrows would zero-fill 295 MB twice per layer at batch 256 and then reduce
        // them; `split_with_sizes` records one node that scatters both gradients into a single
        // buffer. Forward is views either way, and `q‖k` stays one tensor so the rotation can
        // run as full-width products over it.
        let packed = linear(&rms_norm(input), &self.qkv)
            .split_with_sizes([2 * self.width, self.width], -1);
        // QK-norm BEFORE the rotation, which is the reference's order:
        // `train_gpt.py:1106` `q, k = norm(q), norm(k)  # QK norm @Grad62304977` and only then
        // `:1109` `q, k = yarn.rotary(q), yarn.rotary(k)`; identically
        // `train_gpt_medium.py:968` before `:970`, and in-repo `world_model.rs:1396-1397`
        // feeding `:1722-1724`. RoPE rotates each `(r, r+half)` pair, so it preserves the
        // head-dim norm exactly and the two orders agree in exact arithmetic - but not in
        // bf16, and not in what the rotation's inputs look like: normalizing first is what
        // keeps the rotary products O(1), which is the entire point of QK-norm. The V path is
        // deliberately NOT normalized (`packed[1]` goes straight to the mix and then to SDPA),
        // matching the reference, where `norm` touches only `q, k` - so both operands of the
        // value-residual mix below are unnormalized, as they are upstream.
        //
        // ONE kernel in each direction for the whole normalize-then-rotate pair:
        // `fused_kernels::qk_norm_rope` reads the RAW packed `q‖k` block, normalizes each
        // `head_dim` row of the free `[batch, length, 2·heads, head_dim]` view (column
        // `t·heads·head_dim + h·head_dim + d` already carries tensor `t`, head `h`, dim `d`),
        // rotates it against the untiled `[origins, head_dim/2]` rows and writes the
        // `[batch, length, 2, heads, head_dim]` buffer. The normalized block is never
        // materialized and no `rstd` is written or read back - the backward recomputes the
        // normalization from the raw block, which is the cheap half of a memory-bound kernel.
        // Bit-identical to `_fused_rms_norm`-then-`fused_kernels::rope` in both directions.
        let rotated =
            qk_norm_rope(&packed[0], rotation.0, rotation.1, self.heads).split(1, 2);
        // Value residual. `packed[1]` is the layer's own value, head-shaped by a VIEW (splitting
        // the last dimension is always expressible as a stride, so this costs nothing); the mix
        // is one `lerp` - a single read of each operand and one write - rather than the
        // reference's literal `(1-λ)·v + λ·v_1`, which is three kernels and twice the traffic
        // for the same value. `lerp` also makes the endpoints EXACT: ATen selects
        // `self + w·(end - self)` for `|w| < 0.5` and `end - (end - self)·(1 - w)` otherwise, so
        // λ = 0 returns this layer's value bit-for-bit and λ = 1 returns the source layer's.
        // The lambda is cast to the activation dtype first: an fp32 `[1]` operand would promote
        // the whole bf16 mix to fp32 and double every byte it moves.
        let value = packed[1].reshape([batch, length, self.heads, head_dim]);
        let value = match (&self.value_lambda, first_value) {
            (Some(lambda), Some(first)) => value.lerp_tensor(first, &lambda.to_kind(value.kind())),
            _ => value,
        };
        let query_key = |tensor: &Tensor| tensor.squeeze_dim(2).transpose(1, 2);
        // Every attention input is a strided VIEW of the projection or of the rotation buffer,
        // with `head_dim` contiguous: SDPA needs only unit stride on the last dimension, so
        // materializing contiguous q/k/v would be three 197 MB copies a layer for nothing.
        let attended = Tensor::scaled_dot_product_attention(
            &query_key(&rotated[0]),
            &query_key(&rotated[1]),
            &value.transpose(1, 2),
            None::<&Tensor>,
            if train { self.dropout } else { 0.0 },
            true,
            None,
            false,
        )
        // Structurally required, not ours: SDPA emits `[batch, heads, length, head_dim]` and
        // the output projection contracts over `heads·head_dim`, so the head and length axes
        // must be swapped before they can be merged, and no stride expresses that merge.
        .transpose(1, 2)
        .reshape([batch, length, self.width]);
        let residual = scaled_linear(&attended, &self.output, lambdas.post[0])
            .dropout(self.dropout, train)
            .addcmul(input, lambdas.resid[0]);
        let state = match lambdas.x0 {
            Some((x0, lambda)) => residual.addcmul(x0, lambda),
            None => residual,
        };
        // ReLU² instead of GELU: `train_gpt_medium.py:1010`,
        // `x = F.relu(x).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than
        // GELU`, and the fused `relu(x @ W1.T)^2 @ W2.T` kernel of `train_gpt.py:46`. The
        // reference applies NO output-scale correction for the change of activation: the
        // down projection is zero-initialised and every consumer of the residual stream is
        // RMS-normalized, so the activation's second moment is absorbed rather than
        // compensated. Its hidden width is `4 * model_dim` (`train_gpt.py:1299`), which is
        // exactly our fixed 2048 at `d_model` 512, so there is no width mismatch to correct
        // either. ONE kernel in each direction: `fused_kernels::relu_square` is the same
        // fusion the reference has as a Triton kernel, bit-identical to `relu().square()`
        // down to the NaN conventions, and it removes the second full-width pass over the
        // `[tokens, ffn]` hidden activation - the largest single traffic cost of this recipe.
        let ff = scaled_linear(
            &relu_square(&linear(&rms_norm(&state), &self.first)).dropout(self.dropout, train),
            &self.second,
            lambdas.post[1],
        )
        .dropout(self.dropout, train);
        // Only the source layer publishes; a later layer's value is already the mix.
        (
            ff.addcmul(&state, lambdas.resid[1]),
            self.value_lambda.is_none().then(|| value.shallow_clone()),
        )
    }

    /// This block's learned mixing scalars, as `(kind, device tensor)`. The KIND only - the
    /// layer index is the caller's, so a block never has to know where in the stack it sits.
    /// Tensors, not `f64`: [`CausalPatchModel::recipe_scalars`] copies the whole stack in one
    /// host transfer, and the chart is written once per report interval, never per step.
    fn recipe_scalars(&self) -> Vec<(&'static str, &Tensor)> {
        self.value_lambda
            .iter()
            .map(|lambda| ("value lambda", lambda))
            .collect()
    }
}

/// The U-net encoder/decoder layer loop: the first half of the layers push their output onto a
/// stack, the second half pop one and fold it into the residual stream BEFORE their own compute,
/// which pairs the deepest unconsumed encoder layer with the shallowest decoder layer -
/// `3->4, 2->5, 1->6, 0->7` at eight layers.
///
/// Straight from modded-nanogpt, `records/track_1_short/2024-11-10_UNetDoubleLr/
/// c87bb826-797b-4f37-98c7-d3a5dad2de74.txt:257-270` (@brendanh0gan): encoder outputs are
/// appended, then `self.transformer.h[encoder_layers + i](x + skip_weights[i] * pop(), ...)`.
/// The current `train_gpt.py:1591-1595` keeps the same shape - `x = x + gate * cache[3]` ahead
/// of layer 6's compute - having pruned the twelve-layer U down to the single pair that paid
/// for itself there; we keep the full U because eight layers is not twelve and because the
/// pruning was a wall-clock decision on a 3-minute run, not a loss result.
///
/// `gates` are the POST-sigmoid gates, one per decoder layer, already in the residual stream's
/// dtype: the reference casts its residual scalars to bf16 for exactly this multiply
/// (`train_gpt.py:1509-1512`), and an fp32 gate would promote a 94 MB bf16 activation to fp32.
///
/// The fold is one `addcmul`, not a `gate * skip` product followed by an add: `addcmul`'s
/// backward saves only the two factors - both already retained - so the fused form materializes
/// one `[tokens, width]` tensor per pair instead of two and never keeps the product alive.
///
/// `layer` is the per-index compute so this loop can be exercised with distinguishable stub
/// layers; the model's forward passes the real blocks.
fn unet_stack<F>(mut state: Tensor, layers: usize, gates: &[Tensor], mut layer: F) -> Tensor
where
    F: FnMut(usize, &Tensor) -> Tensor,
{
    let encoder = layers / 2;
    assert_eq!(
        gates.len(),
        layers - encoder,
        "one skip gate per decoder layer"
    );
    let mut skips: Vec<Tensor> = Vec::with_capacity(encoder);
    for index in 0..layers {
        if index >= encoder {
            let skip = skips
                .pop()
                .expect("every decoder layer consumes one encoder output");
            state = state.addcmul(&skip, &gates[index - encoder]);
        }
        state = layer(index, &state);
        if index < encoder {
            // A reference, not a copy: this tensor is the next layer's input either way, so the
            // stack costs four pointers and no bytes.
            skips.push(state.shallow_clone());
        }
    }
    state
}

/// Causal per-origin statistics, `[batch, origins]` in fp32. Origin `k` is the last bar of
/// patch `k`; every value uses only bars at or before it.
pub struct Statistics {
    /// Expanding population std of valid close-to-close log returns, variance floored.
    pub sigma: Tensor,
    /// Expanding mean relative range `(high - low) / low`; falls back to σ when flat.
    pub range: Tensor,
    /// Origin close log-price relative to the row anchor.
    pub log_close: Tensor,
    /// Cumulative market log return at the origin, relative to the row's last context bar.
    pub market: Tensor,
    /// Causal ridge slope of the ticker's returns on the market steps, shrunk toward 1.
    pub beta: Tensor,
    /// 1 where at least `min_history` valid bars precede the origin inclusive.
    pub mask: Tensor,
}

impl Statistics {
    pub fn last(&self) -> Self {
        let last = self.sigma.size()[1] - 1;
        Self {
            sigma: self.sigma.narrow(1, last, 1),
            range: self.range.narrow(1, last, 1),
            log_close: self.log_close.narrow(1, last, 1),
            market: self.market.narrow(1, last, 1),
            beta: self.beta.narrow(1, last, 1),
            mask: self.mask.narrow(1, last, 1),
        }
    }
}

fn per_bar(statistic: &Tensor) -> Tensor {
    statistic.unsqueeze(-1).unsqueeze(-1)
}

/// Raw dense-head output, channel-major bf16 `[rows, origins', 2·CHANNELS, pred_len]`: the four
/// candle coordinates then the four log predictive scales, with the μP output multiplier already
/// folded into the head weight.
///
/// Channel-major is load-bearing, not cosmetic. Every consumer of this tensor addresses ONE
/// channel at a time; in the natural `[.., pred_len, 2·CHANNELS]` layout each such access is a
/// stride-8 gather that pulls a 32-byte sector per two useful bytes, and the dense head emits
/// `256 · 375 · 192 · 8` = 147 M elements per step. Here a channel slice is `pred_len`
/// contiguous elements, which is what lets the fused loss address channel `c` of bar `h` as
/// `token·2·CHANNELS·pred_len + c·pred_len + h` and stay coalesced across a warp.
///
/// Nothing splits it any more: the loss reads all eight channels in one kernel and writes one
/// dense gradient, where the `split` this type used to expose cost a `cat` over eight slices in
/// every backward.
pub struct Head(Tensor);

/// Head outputs per (origin, future bar) in fp32, channel-major `[.., CHANNELS, pred_len]`:
/// candle coordinates and the log predictive scale per OHLC channel in σ units, the latter
/// already carrying the `½·ln h` random-walk prior. Materialized for the evaluation decoders
/// only; the training loss consumes [`Head`] directly.
pub struct Output {
    pub coordinates: Tensor,
    pub log_scale: Tensor,
}

/// One class of backbone kernel, as [`CausalPatchModel::kernel_classes`] hands it to
/// `benchmark --profile`: the ops themselves plus the bytes and arithmetic they cannot avoid.
pub struct KernelClass<'a> {
    pub name: &'static str,
    /// Shape and dtype of every tensor `run` consumes. The CALLER allocates them, so allocation
    /// and the first-touch page faults stay outside the timed region; entry 0 is the
    /// differentiable input backward is measured against, and an empty list means the class
    /// carries no gradient at all.
    pub inputs: Vec<(Vec<i64>, Kind)>,
    /// Bytes ONE forward pass must move: every input read once, every output written once. A
    /// floor, so `forward_bytes / measured seconds` is a floor on the class's bandwidth.
    pub forward_bytes: f64,
    /// Arithmetic ONE forward pass must do.
    pub forward_flops: f64,
    /// Leaf parameters whose gradients this class's backward must also produce. Without them
    /// `Tensor::run_backward` prunes the graph to the paths reaching the activation input and
    /// the weight-gradient GEMM - half of a projection's backward - never runs, so a GEMM class
    /// would measure as twice as efficient as it is.
    pub parameters: Vec<Tensor>,
    pub run: Box<dyn Fn(&[Tensor]) -> Tensor + 'a>,
}

/// Decoder-only causal transformer over non-overlapping OHLC+covariate patches with a dense
/// heteroscedastic multi-bar head at every origin.
pub struct CausalPatchModel {
    config: ModelConfig,
    patch: nn::Linear,
    /// Rotary `cos`/`sin` for the fixed position grid, `[origins, head_dim/2]` bf16 and
    /// UNTILED: row `t`, column `r` is the pair-`r` angle at origin `t`. 24 KiB apiece that the
    /// whole stack reads out of L2, because `fused_kernels::rope` indexes these rows directly -
    /// the composed rotation needed them broadcast to a `[1, origins, 2·d_model]` pair of
    /// 768 KiB tiles.
    rotation: (Tensor, Tensor),
    blocks: Vec<Block>,
    /// Per-sub-block residual-stream scale, `[2·layers]` fp32, laid out `[attn_0, ffn_0,
    /// attn_1, ...]`. The reference packs the same numbers as a `[layers, 2]` parameter
    /// (`train_gpt.py:1338`); ONE-dimensional here on purpose - a 2-D parameter is what
    /// NorMuon's router looks for (`optim/muon.rs:1316`), and a 2-D lambda bank would be one
    /// dropped name-allowlist entry away from being orthogonalized as if it were a weight
    /// matrix. 1-D cannot be misrouted by construction.
    resid_lambdas: Tensor,
    /// Per-sub-block branch-output scale, `[2·layers]` fp32, same layout. Folded onto the
    /// output projection weights rather than the activations - see [`scaled_linear`].
    post_lambdas: Tensor,
    /// Per-layer embedding re-injection scale, `[layers]` fp32; `None` - unregistered, not
    /// zeroed - under [`X0Lambdas::Disabled`].
    x0_lambdas: Option<Tensor>,
    /// LOGITS of the U-net skip gates, `[layers/2]` fp32, one per decoder layer in
    /// [`ModelConfig::skip_pairs`] order. The gate the residual stream sees is `σ(logit)`, so
    /// it can never leave `(0, 1)` and starts at σ([`SKIP_LOGIT_INIT`]) = 0.18243. Stored on the
    /// root path, 1-D and outside the `block_*` allowlist, which is what routes it to AdamW.
    skip_weights: Tensor,
    covariates: Option<nn::Linear>,
    head_hidden: nn::Linear,
    head_output: nn::Linear,
    known_index: Tensor,
    sigma_scale: Tensor,
    unit_scale: Tensor,
    /// `√h` per future bar, `[1, 1, 1, pred_len]` fp32: the unit the close coordinate is
    /// measured in, and the ONE tensor a post-hoc mean calibration touches - see
    /// [`Self::fold_mean_gain`].
    horizon_scale: Tensor,
    half_log_horizon: Tensor,
    /// `1/h` per future bar, `[1, 1, 1, pred_len]`: `exp(-2·½·ln h)` pulled out of the
    /// per-element NLL so the `½·ln h` prior never enters a full-size kernel.
    inverse_horizon: Tensor,
    /// The objective's per-horizon weight, `[1, 1, 1, pred_len]` fp32, mean 1 over the whole
    /// axis - see [`HorizonLoss`]. A CONSTANT buffer, never a varstore variable: it must not
    /// be trained, and a fixed tensor is what makes it safe inside the captured CUDA graph.
    horizon_weight: Tensor,
    /// `Ψ`, the structured mean's expansion, `[2·CHANNELS·pred_len, head_outputs]` fp32;
    /// `None` - and no kernel at all - under [`HorizonMean::Free`]. A CONSTANT buffer for the
    /// same two reasons as [`Self::horizon_weight`]: the basis is fixed by construction, and a
    /// fixed tensor at a fixed address is what makes it safe inside the captured CUDA graph.
    /// [`Self::head`] folds it onto the output weight, so it never touches an activation.
    mean_expansion: Option<Tensor>,
    /// `Φ`, the log-scale basis `[scales, pred_len]` fp32, `None` outside
    /// [`HorizonMean::Increment`]. Unlike [`Self::mean_expansion`] this canNOT be folded onto
    /// the output weight: folding it would make the token-space GEMM emit the full
    /// `2·CHANNELS·pred_len` again and give back the entire width saving the mode exists for.
    /// It runs in activation space instead, where it is `[rows·origins·CHANNELS, scales] ×
    /// [scales, pred_len]` - 1.2 GFLOP at `scales = 8`, against the 0.45 TFLOP/step the narrower
    /// output projection saves.
    scale_expansion: Option<Tensor>,
    /// The three per-horizon constants the loss reads, in INCREMENT space: `√1`, `½·ln 1` and
    /// `1/1`, i.e. ones, zeros and ones. `None` outside [`HorizonMean::Increment`].
    ///
    /// They are ones and zeros because a one-bar move in σ units has scale exactly 1, so the
    /// random-walk prior that `√h`, `½·ln h` and `1/h` carry is the identity on a single bar.
    /// Materialized as buffers rather than folded away because the fused loss takes them as
    /// tensor operands and a captured CUDA graph needs a fixed address, not a literal.
    increment_geometry: Option<(Tensor, Tensor, Tensor)>,
    /// `1/LOG_SCALE_CAP`, shaped `[1, 1, 1, 1]` rather than 0-dim so that multiplying a bf16
    /// log-scale channel by it promotes the result to fp32 in a single kernel.
    log_scale_gain: Tensor,
    /// The applied amplitude calibration of the emitted mean, `None` on an uncalibrated model.
    /// See [`Self::set_mean_gain`]; read only by [`Self::gained`], which is the last step of
    /// [`Self::decode`] and is not on the training path.
    mean_gain: Option<MeanGain>,
    /// The orthonormal horizon reparametrization, `None` - and no new kernel, no new buffer and
    /// no new branch in any hot path - under [`TargetBasis::Cumulative`]. That `None` is what
    /// makes the control arm bit-for-bit the pre-knob objective rather than merely equal to it:
    /// the fused loss call below is reached by exactly the code it was reached by before.
    basis: Option<BasisTransform>,
}

/// A frozen amplitude calibration, resident: the two curves as the decode's operands, beside
/// the manifest record they came from so a report can state what was applied without
/// round-tripping a device tensor.
struct MeanGain {
    frozen: FrozenGain,
    /// `[1, pred_len]` fp32, so it broadcasts against a `[.., 1, pred_len]` channel slice at
    /// any rank without changing it.
    anchor: Tensor,
    offset: Tensor,
}

impl CausalPatchModel {
    pub fn new(path: &nn::Path, config: &ModelConfig) -> Self {
        config
            .validate()
            .expect("invalid causal patch model configuration");
        // Loading and authenticating the statistics artifact belongs here, beside the config
        // validation it completes: a mispaired artifact is a different objective, and
        // discovering that after the first optimizer step would waste the lease.
        let basis = (!config.target_basis.is_identity())
            .then(|| {
                let statistics = config
                    .basis_stats
                    .as_deref()
                    .map(target_basis::BasisStatistics::load)
                    .transpose()
                    .expect("unreadable --basis-stats artifact");
                BasisTransform::new(
                    config.target_basis,
                    config.basis_weight,
                    config.pred_len,
                    statistics.as_ref(),
                    path.device(),
                )
                .expect("invalid target basis")
            });
        let device = path.device();
        let head_dim = config.d_model / config.heads;
        let aux_channels = config.features.channels() as i64;
        let known: Vec<i64> = config
            .features
            .channel_mask(Feature::known_future)
            .iter()
            .enumerate()
            .filter_map(|(index, known)| known.then_some(index as i64))
            .collect();
        let sigma_scale: Vec<f32> = config
            .features
            .channel_mask(Feature::sigma_scaled)
            .iter()
            .map(|&scaled| f32::from(u8::from(scaled)))
            .collect();
        let covariates = (!known.is_empty()).then(|| {
            projection(
                path / "covariates",
                config.pred_len * known.len() as i64,
                COVARIATE_WIDTH,
                true,
            )
        });
        let head_input = config.d_model + covariates.as_ref().map_or(0, |_| COVARIATE_WIDTH);
        let horizon = (Tensor::arange(config.pred_len, (Kind::Float, device)) + 1.0)
            .reshape([1, 1, 1, config.pred_len]);
        let sigma_scale = Tensor::from_slice(&sigma_scale).to_device(device);
        // The dense causal-patch model attends over a FIXED position grid, so the rotation rows
        // are constant: building them once removes four small kernels per attention tensor per
        // layer (64 launches per forward), and `fused_kernels::rope` consumes them in exactly
        // this untiled form, so the broadcast tiles the composed rotation needed are never
        // built at all.
        let rope = RotaryEmbedding::new(config.origins(), head_dim, head_dim, device);
        let rotation = rope.cached_rotation(
            &Tensor::arange(config.origins(), (Kind::Int64, device)),
            Kind::BFloat16,
        );
        Self {
            patch: projection(
                path / "patch",
                config.patch_len * (CHANNELS + aux_channels),
                config.d_model,
                true,
            ),
            rotation,
            blocks: (0..config.layers)
                .map(|index| Block::new(path / format!("block_{index}"), config, index))
                .collect(),
            // `lambdas.*`, NOT `block_*`: the NorMuon allowlist is `["block_"]`
            // (`compute.rs:115`) and the routing assertion at `compute.rs:141-152` demands
            // that every NorMuon tensor be a 2-D `block_*.weight`. These are 1-D and outside
            // the allowlist, so they land in AdamW on both counts, and the `lambda` fragment
            // in `adamw_no_weight_decay_name_substrings` keeps weight decay off them - the
            // reference gives all three `wd_mul = 0` (`train_gpt.py:2035-2036`,
            // `train_gpt_medium.py:1081`).
            resid_lambdas: (path / "lambdas").var(
                "resid",
                &[2 * config.layers as i64],
                nn::Init::Const(RESID_LAMBDA_INIT),
            ),
            post_lambdas: (path / "lambdas").var(
                "post",
                &[2 * config.layers as i64],
                nn::Init::Const(POST_LAMBDA_INIT),
            ),
            x0_lambdas: config.x0_lambdas.enabled().then(|| {
                (path / "lambdas").var(
                    "x0",
                    &[config.layers as i64],
                    nn::Init::Const(X0_LAMBDA_INIT),
                )
            }),
            skip_weights: path.var(
                "skip_weights",
                &[(config.layers / 2) as i64],
                nn::Init::Const(SKIP_LOGIT_INIT),
            ),
            covariates,
            head_hidden: projection(path / "head" / "hidden", head_input, HEAD_HIDDEN, true),
            head_output: nn::linear(
                path / "head" / "output",
                HEAD_HIDDEN,
                config.horizon_mean.head_outputs(config.pred_len),
                nn::LinearConfig {
                    ws_init: nn::Init::Const(0.0),
                    bs_init: Some(nn::Init::Const(0.0)),
                    bias: true,
                },
            ),
            known_index: Tensor::from_slice(&known).to_device(device),
            unit_scale: 1.0 - &sigma_scale,
            sigma_scale,
            horizon_scale: horizon.sqrt(),
            half_log_horizon: horizon.log() * 0.5,
            inverse_horizon: horizon.reciprocal(),
            horizon_weight: Tensor::from_slice(
                &config
                    .horizon_loss
                    .weights(config.pred_len)
                    .into_iter()
                    .map(|weight| weight as f32)
                    .collect::<Vec<f32>>(),
            )
            .reshape([1, 1, 1, config.pred_len])
            .to_device(device),
            mean_expansion: config.horizon_mean.expansion(config.pred_len).map(|values| {
                Tensor::from_slice(&values)
                    .reshape([
                        OUTPUTS_PER_BAR * config.pred_len,
                        config.horizon_mean.head_outputs(config.pred_len),
                    ])
                    .to_device(device)
            }),
            scale_expansion: config.horizon_mean.scale_basis(config.pred_len).map(|values| {
                let values: Vec<f32> = values.into_iter().map(|value| value as f32).collect();
                Tensor::from_slice(&values)
                    .reshape([-1, config.pred_len])
                    .to_device(device)
            }),
            increment_geometry: matches!(config.horizon_mean, HorizonMean::Increment { .. })
                .then(|| {
                    let shape = [1, 1, 1, config.pred_len];
                    (
                        Tensor::ones(shape, (Kind::Float, device)),
                        Tensor::zeros(shape, (Kind::Float, device)),
                        Tensor::ones(shape, (Kind::Float, device)),
                    )
                }),
            log_scale_gain: Tensor::full(
                [1, 1, 1, 1],
                1.0 / LOG_SCALE_CAP,
                (Kind::Float, device),
            ),
            mean_gain: None,
            basis,
            config: config.clone(),
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The orthonormal horizon reparametrization in force, `None` under `cumulative`.
    pub fn basis(&self) -> Option<&BasisTransform> {
        self.basis.as_ref()
    }

    pub fn horizon_scale(&self) -> &Tensor {
        &self.horizon_scale
    }

    pub fn half_log_horizon(&self) -> &Tensor {
        &self.half_log_horizon
    }

    /// Apply a frozen amplitude calibration to the emitted MEAN.
    ///
    /// This is the ONLY thing an amplitude calibration does to the model, and it is the whole
    /// of it. A decoded candle has exactly two amplitude degrees of freedom - see
    /// [`decode_joint`] - so the transform is
    ///
    /// ```text
    /// f_c(h) -> g_anchor(h)·close(h) + g_offset(h)·(f_c(h) - close(h))
    /// ```
    ///
    /// with both gains strictly positive. Three consequences, all of them the point:
    ///
    /// - the close channel is rescaled exactly, `close -> g_anchor·close`, and the other three
    ///   ride it, so the amplitude the long-horizon MSE is destroyed by is corrected in every
    ///   channel at once;
    /// - every intrabar offset keeps its sign and its size relative to the anchor, so
    ///   `high ≥ max(open, close) ≥ min(open, close) ≥ low` survives for exactly the reason it
    ///   survives uncalibrated. Independent per-CHANNEL gains would not have this property;
    /// - the predictive scales are untouched: they come from `half_log_horizon` and the
    ///   log-scale channels, which this does not read. The NLL is therefore the un-gained
    ///   model's NLL, which is what keeps checkpoint selection comparable across the whole
    ///   experiment - a mean gain with no matching σ refit makes NLL worse even where it makes
    ///   MSE better.
    ///
    /// The TRAINING objective cannot see this. The loss chains read `horizon_scale`,
    /// `inverse_horizon` and `half_log_horizon`, none of which this touches, and
    /// [`Self::losses`] refuses to run on a calibrated model rather than relying on that
    /// reading. The checkpoint is untouched too: the two curves are buffers derived from the
    /// manifest, not varstore variables, so no saved tensor and no training bit moves.
    ///
    /// Setting REPLACES any previous curve rather than composing with it, so a re-fit at a
    /// later step cannot silently square its own shrinkage.
    pub fn set_mean_gain(&mut self, gain: &FrozenGain) -> Result<()> {
        let pred_len = self.config.pred_len;
        gain.validate(pred_len as usize)?;
        let device = self.horizon_scale.device();
        let curve = |values: &[f64]| {
            Tensor::from_slice(&values.iter().map(|value| *value as f32).collect::<Vec<f32>>())
                .reshape([1, pred_len])
                .to_device(device)
        };
        self.mean_gain = Some(MeanGain {
            anchor: curve(&gain.anchor),
            offset: curve(&gain.offset),
            frozen: gain.clone(),
        });
        Ok(())
    }

    /// The applied amplitude calibration, or `None` on an uncalibrated model.
    pub fn mean_gain(&self) -> Option<&FrozenGain> {
        self.mean_gain.as_ref().map(|gain| &gain.frozen)
    }

    /// The two-coordinate rescale, the identity on an uncalibrated model. Three elementwise
    /// kernels on a `[rows, origins, CHANNELS, pred_len]` decode, which is a rounding error
    /// beside the transformer that produced it, and none of them on the training path.
    fn gained(&self, decoded: Tensor) -> Tensor {
        let Some(gain) = &self.mean_gain else {
            return decoded;
        };
        let anchor = decoded.narrow(-2, CHANNELS - 1, 1);
        let offsets = &decoded - &anchor;
        anchor * &gain.anchor + offsets * &gain.offset
    }

    /// The objective's per-horizon weight buffer, `[1, 1, 1, pred_len]` fp32, for the reference
    /// [`gaussian_nll`] and the evaluation path.
    pub fn horizon_weight_buffer(&self) -> &Tensor {
        &self.horizon_weight
    }

    /// The objective's per-horizon weight vector on the host, index `i` being horizon `i + 1`.
    /// Read back from the buffer the loss actually multiplies by, not recomputed, so the chart
    /// and the selection scalar cannot disagree with the gradient.
    pub fn horizon_weights(&self) -> Vec<f64> {
        Vec::<f64>::try_from(self.horizon_weight.to_kind(Kind::Double).flatten(0, -1))
            .expect("the horizon weight buffer is a dense fp32 vector")
    }

    pub fn statistics(&self, batch: &Batch) -> Statistics {
        let c = &self.config;
        let (context, patch, origins) = (c.seq_len, c.patch_len, c.origins());
        let log_prices = batch.log_prices.narrow(1, 0, context);
        let valid = batch.valid.narrow(1, 0, context);
        let rows = log_prices.size()[0];
        let close = log_prices.select(2, 3);
        let pair = valid.narrow(1, 1, context - 1) * valid.narrow(1, 0, context - 1);
        let returns = (close.narrow(1, 1, context - 1) - close.narrow(1, 0, context - 1)) * &pair;
        let lead = Tensor::zeros([rows, 1], (Kind::Float, log_prices.device()));
        let at_origin = |series: &Tensor| series.reshape([rows, origins, patch]).select(2, patch - 1);
        let cumulative = |series: &Tensor| {
            at_origin(&Tensor::cat(&[&lead, series], 1).cumsum(1, Kind::Float))
        };
        let pairs = cumulative(&pair).clamp_min(1.0);
        let mean = cumulative(&returns) / &pairs;
        let variance = cumulative(&returns.square()) / &pairs - mean.square();
        let sigma = (variance.clamp_min(0.0) + RETURN_VARIANCE_FLOOR).sqrt();
        let market = batch.market_cum.narrow(1, 0, context);
        let steps = (market.narrow(1, 1, context - 1) - market.narrow(1, 0, context - 1)) * &pair;
        let market_squares = cumulative(&steps.square());
        let ridge = (&market_squares / &pairs).clamp_min(RETURN_VARIANCE_FLOOR) * BETA_PRIOR_BARS;
        let beta = (cumulative(&(&returns * &steps)) + &ridge) / (market_squares + ridge);
        let bars = at_origin(&valid.cumsum(1, Kind::Float));
        let spread = ((log_prices.select(2, 1) - log_prices.select(2, 2)).exp() - 1.0) * &valid;
        let range = at_origin(&spread.cumsum(1, Kind::Float)) / bars.clamp_min(1.0);
        Statistics {
            range: range.where_self(&range.gt(0.0), &sigma),
            sigma,
            log_close: at_origin(&close),
            market: at_origin(&market),
            beta,
            mask: bars.ge(c.min_history as f64).to_kind(Kind::Float),
        }
    }

    /// Windows covering bars `t_k + 1 ..= t_k + pred_len` of a `[batch, seq_len + pred_len, ...]`
    /// series for every origin or the final origin only, channel-major:
    /// `[batch, origins', channels, pred_len]` for a 3-D series and `[batch, origins', pred_len]`
    /// for a 2-D one. Channel-major is what `unfold` produces natively and what every consumer
    /// slices along, so no transpose survives here.
    fn future_windows(&self, series: &Tensor, last_only: bool) -> Tensor {
        let c = &self.config;
        if last_only {
            let window = series.narrow(1, c.seq_len, c.pred_len).unsqueeze(1);
            return if series.dim() == 2 {
                window
            } else {
                window.transpose(2, 3)
            };
        }
        series
            .narrow(1, c.patch_len, c.seq_len + c.pred_len - c.patch_len)
            .unfold(1, c.pred_len, c.patch_len)
    }

    /// `stats` must be the full statistics; `last_only` restricts the head to the final origin.
    pub fn forward(&self, batch: &Batch, stats: &Statistics, train: bool, last_only: bool) -> Head {
        self.head(batch, &self.backbone(batch, stats, train, last_only), last_only)
    }

    /// The σ-normalised patch tokens the backbone embeds, bf16
    /// `[rows, origins, patch_len·(CHANNELS + aux)]`.
    ///
    /// The normalisation itself must be fp32: subtracting the origin close from a log price is a
    /// cancellation, and σ is a small fp32 statistic. But the CAST happens per part, BEFORE the
    /// concatenation, not after it. Casting after made the concatenation write and re-read a
    /// 98 MB fp32 tensor at batch 256 for a 49 MB bf16 result; casting first is bit-identical
    /// (an elementwise cast commutes with concatenation exactly) and moves 197 MB less.
    ///
    /// Nothing here carries a gradient - prices, auxiliaries and statistics are all data - so
    /// none of these fp32 intermediates is retained past the cast.
    pub fn tokens(&self, batch: &Batch, stats: &Statistics) -> Tensor {
        let c = &self.config;
        let (context, patch, horizon, origins) = (c.seq_len, c.patch_len, c.pred_len, c.origins());
        let aux_channels = c.features.channels() as i64;
        let rows = batch.log_prices.size()[0];
        assert_eq!(batch.log_prices.size(), [rows, context + horizon, CHANNELS]);
        assert_eq!(batch.aux.size(), [rows, context + horizon, aux_channels]);
        assert_eq!(stats.sigma.size(), [rows, origins]);
        let inv_sigma = per_bar(&stats.sigma.reciprocal());
        let prices = ((batch
            .log_prices
            .narrow(1, 0, context)
            .reshape([rows, origins, patch, CHANNELS])
            - per_bar(&stats.log_close))
            * &inv_sigma)
            .to_kind(Kind::BFloat16);
        let aux = (batch
            .aux
            .narrow(1, 0, context)
            .reshape([rows, origins, patch, aux_channels])
            * (&self.sigma_scale * inv_sigma + &self.unit_scale))
            .to_kind(Kind::BFloat16);
        Tensor::cat(&[prices, aux], 3).reshape([
            rows,
            origins,
            patch * (CHANNELS + aux_channels),
        ])
    }

    /// Patch embedding, the causal transformer stack and the final norm, narrowed to the scored
    /// origins. Separated from [`Self::head`] so the step timing can attribute the two phases
    /// without a synchronization inside the composed forward.
    ///
    /// `x0` is the normalized patch embedding, which is what the reference re-injects:
    /// `train_gpt.py:1549` `x = x0 = norm(x[None])` (and `train_gpt_medium.py:1158`), so the
    /// embedding is normalized ONCE here rather than only inside block 0's pre-norm. Every
    /// consumer of the residual stream is a gainless RMSNorm, so this changes nothing about
    /// block 0's input; it makes `x0` a unit-RMS tensor that the per-layer `x0_lambda` can
    /// mix in on a known scale.
    pub fn backbone(
        &self,
        batch: &Batch,
        stats: &Statistics,
        train: bool,
        last_only: bool,
    ) -> Tensor {
        let origins = self.config.origins();
        let x0 = rms_norm(
            &linear(&self.tokens(batch, stats), &self.patch).dropout(self.config.dropout, train),
        );
        // ONE cast and ONE `unbind` for the whole stack, as the reference does
        // (`train_gpt.py:1509-1512`, `self.resid_lambdas[:, 0].bfloat16().unbind(0)`). The cast
        // is mandatory, not cosmetic: this model runs without autocast, so a DIMENSIONED fp32
        // scalar against a bf16 activation would promote the whole result to fp32 and double
        // the residual stream's traffic (`docs/timexer_segment.md:16`). `unbind` yields 0-dim
        // views, which broadcast against anything.
        let kind = x0.kind();
        let resid = self.resid_lambdas.to_kind(kind).unbind(0);
        let post = self.post_lambdas.to_kind(kind).unbind(0);
        let x0_lambdas = self
            .x0_lambdas
            .as_ref()
            .map(|bank| bank.to_kind(kind).unbind(0));
        // The U-net gates ride the same one-cast rule: POST-sigmoid, in the activation dtype.
        // One `unbind`, not `layers/2` `select`s - `select`'s backward is `zeros_like` plus a
        // copy, `unbind`'s is a single `stack`.
        let gates = self.skip_weights.sigmoid().to_kind(kind).unbind(0);
        // Layer 0's raw value, published once and mixed into every deeper layer's V. `None`
        // until layer 0 returns it, which is also why layer 0 owns no `value_lambda`.
        let mut first_value: Option<Tensor> = None;
        let state = unet_stack(
            x0.shallow_clone(),
            self.blocks.len(),
            &gates,
            |index, state| {
                let lambdas = BlockLambdas {
                    resid: [&resid[2 * index], &resid[2 * index + 1]],
                    post: [&post[2 * index], &post[2 * index + 1]],
                    x0: x0_lambdas
                        .as_ref()
                        .map(|lambdas| (&x0, &lambdas[index])),
                };
                let (next, published) = self.blocks[index].forward(
                    state,
                    first_value.as_ref(),
                    &lambdas,
                    (&self.rotation.0, &self.rotation.1),
                    train,
                );
                if let Some(value) = published {
                    first_value = Some(value);
                }
                next
            },
        );
        let state = rms_norm(&state);
        if last_only {
            state.narrow(1, origins - 1, 1)
        } else {
            state
        }
    }

    /// Every learned mixing scalar the backbone holds, as `(label, post-parameterization
    /// value)`, for the shared `timexer_segment_recipe_scalars` report base.
    ///
    /// Values are what the forward pass actually applies: the residual, post, x0 and value
    /// lambdas are stored raw and applied raw (`train_gpt.py:1334`, `:1338`,
    /// `train_gpt_medium.py:1078`, and the value residual's
    /// `records/track_1_short/2024-11-06_ShortcutsTweaks/…txt:168`), so they report raw; the
    /// U-net skip is stored as a LOGIT and applied as `σ(logit)`, so it reports the sigmoid -
    /// the question a reader asks of this panel is whether a gate moved off its 0.18 init, and
    /// a logit does not answer it.
    ///
    /// ONE host transfer for the whole set: every part is concatenated on device and copied
    /// once. Called per report interval, never per step - it synchronizes - and deliberately
    /// not `.item()` per scalar, which would be one synchronization each.
    pub fn recipe_scalars(&self) -> Vec<(String, f64)> {
        let (names, values) = self.recipe_scalar_parts();
        if names.is_empty() {
            return Vec::new();
        }
        let flat = Tensor::cat(&values, 0)
            .to_kind(Kind::Double)
            .to_device(Device::Cpu);
        assert_eq!(
            names.len() as i64,
            flat.numel() as i64,
            "every recipe scalar name must have exactly one value"
        );
        let values = Vec::<f64>::try_from(flat).expect("recipe scalars are a 1-D fp64 vector");
        names.into_iter().zip(values).collect()
    }

    /// The contributions to [`Self::recipe_scalars`], names and device-side values kept apart so
    /// that adding a family of scalars costs one `push` here and no host round-trip of its own.
    /// The names must be pushed in the same order as the tensors they label, which
    /// [`Self::recipe_scalars`]'s length assertion is there to catch.
    fn recipe_scalar_parts(&self) -> (Vec<String>, Vec<Tensor>) {
        let layers = self.config.layers;
        let mut names = Vec::new();
        let mut values = Vec::new();
        // Per sub-block, in the banks' own `[attn_0, ffn_0, attn_1, …]` layout.
        names.extend((0..layers).flat_map(|layer| {
            [
                format!("residual lambda L{layer} attn"),
                format!("residual lambda L{layer} ffn"),
            ]
        }));
        values.push(self.resid_lambdas.shallow_clone());
        names.extend((0..layers).flat_map(|layer| {
            [
                format!("post lambda L{layer} attn"),
                format!("post lambda L{layer} ffn"),
            ]
        }));
        values.push(self.post_lambdas.shallow_clone());
        // Absent entirely under `X0Lambdas::Disabled`: the chart's series are the model's own
        // scalars, so a mode with no injection must not report a flat zero line for one.
        if let Some(bank) = &self.x0_lambdas {
            names.extend((0..layers).map(|layer| format!("x0 lambda L{layer}")));
            values.push(bank.shallow_clone());
        }
        // POST-sigmoid, in `skip_pairs` order - the same order the gates are unbound in.
        names.extend(
            self.config
                .skip_pairs()
                .map(|(source, destination)| format!("skip weight {source}->{destination}")),
        );
        values.push(self.skip_weights.sigmoid());
        // The value residual, per layer that owns one: layer 0 is the source and has none.
        for (layer, block) in self.blocks.iter().enumerate() {
            for (kind, scalar) in block.recipe_scalars() {
                names.push(format!("{kind} L{layer}"));
                values.push(scalar.reshape([1]));
            }
        }
        (names, values)
    }

    /// The backbone's kernel classes at the real per-layer shapes: what `benchmark --profile`
    /// times with CUDA events, one entry per class of kernel the step issues.
    ///
    /// Every closure calls the SAME function the forward path calls - `Block::rotate`,
    /// `rms_norm`, `linear`, `Self::tokens` - so a class cannot drift from the model
    /// without the compiler noticing. The last entry is the composed layer, so the sum of the
    /// parts can be compared against the whole and the attribution error stated rather than
    /// assumed.
    pub fn kernel_classes<'a>(
        &'a self,
        batch: &'a Batch,
        stats: &'a Statistics,
        train: bool,
    ) -> Vec<KernelClass<'a>> {
        let c = &self.config;
        let (origins, width, ffn) = (c.origins(), c.d_model, c.ffn);
        let (heads, head_dim) = (c.heads, c.d_model / c.heads);
        let rows = batch.log_prices.size()[0];
        let block = &self.blocks[0];
        let rotation = (&self.rotation.0, &self.rotation.1);
        let tokens = (rows * origins) as f64;
        let (bf16, fp32) = (|n: f64| 2. * n, |n: f64| 4. * n);
        // One `[tokens, d_model]` bf16 activation, one `[tokens, ffn]` one, and the fp32
        // `rstd` a gainless RMSNorm writes - one reduction output per row, where the LayerNorm
        // this replaced wrote a `mean`/`rstd` PAIR.
        let state = bf16(tokens * width as f64);
        let hidden = bf16(tokens * ffn as f64);
        let rstd = |rows_of: f64| fp32(rows_of);
        let gemm = |reduce: i64, out: i64| 2. * tokens * reduce as f64 * out as f64;
        // fp32 master read plus bf16 copy written, per parameter the class casts.
        let cast = |elements: i64| fp32(elements as f64) + bf16(elements as f64);
        let activation = |last: i64| (vec![rows, origins, last], Kind::BFloat16);
        let projected = |layer: &nn::Linear| {
            let mut params = vec![layer.ws.shallow_clone()];
            params.extend(layer.bs.iter().map(Tensor::shallow_clone));
            params
        };
        let mut classes = vec![
            KernelClass {
                name: "RMSNorm",
                inputs: vec![activation(width)],
                // No parameter cast (gainless) and one reduction output instead of two: the
                // LayerNorm this replaced charged `2·state + fp32(2·tokens) + cast(2·width)`.
                forward_bytes: 2. * state + rstd(tokens),
                // `x²` summed, one `rsqrt`, one multiply: 3 per element against LayerNorm's 8
                // (mean, centred square, rsqrt, scale, gain, bias).
                forward_flops: 3. * tokens * width as f64,
                parameters: Vec::new(),
                run: Box::new(move |input| rms_norm(&input[0])),
            },
            KernelClass {
                name: "QK norm + rotary",
                // The RAW packed `q‖k` block: ONE kernel normalizes each of its `2·heads` rows
                // of `head_dim` per token AND rotates them, so this charges `4·state` - read the
                // block, write the rotated buffer. Nothing else crosses HBM: the normalized
                // block is never materialized, no `rstd` is written (the backward recomputes the
                // normalization from the raw block), and the untiled `[origins, head_dim/2]`
                // rotation rows are 24 KiB of L2. The composition charged `4·state +
                // fp32(2·heads·tokens)` for the norm and another `4·state` for the rotation, on
                // top of the `18·state` the pre-kernel composed rotation cost.
                inputs: vec![activation(2 * width)],
                forward_bytes: 4. * state,
                // The norm's three per element over `2·width` plus the rotation's six per
                // `width`: one product and one sum for each of the two half-crossing terms.
                forward_flops: 3. * tokens * 2. * width as f64 + 6. * tokens * width as f64,
                parameters: Vec::new(),
                run: Box::new(move |input| qk_norm_rope(&input[0], rotation.0, rotation.1, heads)),
            },
            KernelClass {
                name: "QKV projection",
                inputs: vec![activation(width)],
                forward_bytes: 4. * state + cast(3 * width * width),
                forward_flops: gemm(width, 3 * width),
                parameters: projected(&block.qkv),
                run: Box::new(move |input| linear(&input[0], &block.qkv)),
            },
            KernelClass {
                name: "causal SDPA",
                // The rotation buffer and the projection, exactly as `Block::forward` holds
                // them: q/k stride over 2·d_model, v over 3·d_model, both unit on head_dim.
                inputs: vec![activation(2 * width), activation(3 * width)],
                forward_bytes: 4. * state + fp32((rows * heads * origins) as f64),
                // `QKᵀ` and `AV` are each a dense `2·rows·origins²·d_model` and half of that
                // under the causal mask, so the pair costs one dense GEMM's worth.
                forward_flops: 2. * (rows * origins * origins * width) as f64,
                parameters: Vec::new(),
                run: Box::new(move |input| {
                    let rotated = input[0]
                        .reshape([rows, origins, 2, heads, head_dim])
                        .split(1, 2);
                    let query_key = |tensor: &Tensor| tensor.squeeze_dim(2).transpose(1, 2);
                    Tensor::scaled_dot_product_attention(
                        &query_key(&rotated[0]),
                        &query_key(&rotated[1]),
                        &input[1]
                            .narrow(-1, 2 * width, width)
                            .reshape([rows, origins, heads, head_dim])
                            .transpose(1, 2),
                        None::<&Tensor>,
                        if train { block.dropout } else { 0.0 },
                        true,
                        None,
                        false,
                    )
                }),
            },
            KernelClass {
                name: "attention output flatten",
                inputs: vec![(vec![rows, heads, origins, head_dim], Kind::BFloat16)],
                forward_bytes: 2. * state,
                forward_flops: 0.,
                parameters: Vec::new(),
                run: Box::new(move |input| {
                    input[0].transpose(1, 2).reshape([rows, origins, width])
                }),
            },
            KernelClass {
                name: "attention output projection",
                inputs: vec![activation(width)],
                // The post-lambda is folded onto the weight COPY the cast already makes, so
                // the extra traffic is one 512×512 bf16 read plus one write - 1/96 000 of the
                // activation it would otherwise scale (see [`scaled_linear`]).
                forward_bytes: 2. * state
                    + cast(width * width)
                    + bf16(2. * (width * width) as f64),
                forward_flops: gemm(width, width),
                parameters: projected(&block.output),
                run: Box::new(move |input| {
                    scaled_linear(
                        &input[0],
                        &block.output,
                        &self.post_lambdas.get(0).to_kind(input[0].kind()),
                    )
                }),
            },
            KernelClass {
                name: "residual addcmul",
                // `out + x·λ` in ONE kernel: two reads and a write, exactly what the bare
                // `out + x` it replaces cost. The residual scale is free; only the x0
                // injection is a second invocation.
                inputs: vec![activation(width), activation(width)],
                forward_bytes: 3. * state,
                forward_flops: 2. * tokens * width as f64,
                parameters: Vec::new(),
                run: Box::new(move |input| {
                    input[0].addcmul(&input[1], &self.resid_lambdas.get(0).to_kind(input[0].kind()))
                }),
            },
            KernelClass {
                name: "FFN up projection",
                inputs: vec![activation(width)],
                forward_bytes: state + hidden + cast(ffn * width),
                forward_flops: gemm(width, ffn),
                parameters: projected(&block.first),
                run: Box::new(move |input| linear(&input[0], &block.first)),
            },
            KernelClass {
                name: "ReLU^2",
                inputs: vec![activation(ffn)],
                // ONE kernel: `fused_kernels::relu_square` writes its result in a single pass,
                // so this charges `2·hidden` - what GELU charged - instead of the `4·hidden`
                // the `relu`-then-`square` composition charged, and the extra materialized
                // `[tokens, ffn]` tensor that was the single largest traffic cost of the
                // residual recipe does not exist. Same fusion the reference has as a Triton
                // kernel (`train_gpt.py:46-48`, `relu(x @ W1.T)^2 @ W2.T`), and ours is
                // bit-identical to the composition it replaces in both directions.
                forward_bytes: 2. * hidden,
                forward_flops: 2. * tokens * ffn as f64,
                parameters: Vec::new(),
                run: Box::new(move |input| relu_square(&input[0])),
            },
            KernelClass {
                name: "FFN down projection",
                inputs: vec![activation(ffn)],
                forward_bytes: hidden + state + cast(width * ffn) + bf16(2. * (width * ffn) as f64),
                forward_flops: gemm(ffn, width),
                parameters: projected(&block.second),
                run: Box::new(move |input| {
                    scaled_linear(
                        &input[0],
                        &block.second,
                        &self.post_lambdas.get(1).to_kind(input[0].kind()),
                    )
                }),
            },
        ];
        // The composed layer charges every class it invokes more than once again: the pre-norm
        // runs twice (before attention, before the feedforward) and the residual `addcmul`
        // twice for the residual scales, plus once more for the x0 injection on the attention
        // line when that injection exists. The QK norm runs once.
        let charged = |name: &'static str| -> (f64, f64) {
            let class = classes
                .iter()
                .find(|class| class.name == name)
                .expect("kernel class present in the list above");
            (class.forward_bytes, class.forward_flops)
        };
        let (norm_bytes, norm_flops) = charged("RMSNorm");
        let (addcmul_bytes, addcmul_flops) = charged("residual addcmul");
        let x0_addcmuls = if self.config.x0_lambdas.enabled() { 1. } else { 0. };
        let layer_bytes: f64 = classes.iter().map(|class| class.forward_bytes).sum::<f64>()
            + norm_bytes
            + (1. + x0_addcmuls) * addcmul_bytes;
        let layer_flops: f64 = classes.iter().map(|class| class.forward_flops).sum::<f64>()
            + norm_flops
            + (1. + x0_addcmuls) * addcmul_flops;
        classes.push(KernelClass {
            name: "composed layer",
            // Entry 1 is `x0`, the normalized patch embedding the block re-injects - present
            // only when the injection is.
            inputs: if self.config.x0_lambdas.enabled() {
                vec![activation(width), activation(width)]
            } else {
                vec![activation(width)]
            },
            forward_bytes: layer_bytes,
            forward_flops: layer_flops,
            // No norm parameters any more: the RMSNorm is gainless and the projections are
            // bias-free, so a block's leaves are four matrices plus the residual and post
            // lambdas, plus the x0 bank where it exists.
            parameters: projected(&block.qkv)
                .into_iter()
                .chain(projected(&block.output))
                .chain(projected(&block.first))
                .chain(projected(&block.second))
                .chain([
                    self.resid_lambdas.shallow_clone(),
                    self.post_lambdas.shallow_clone(),
                ])
                .chain(self.x0_lambdas.iter().map(Tensor::shallow_clone))
                .collect(),
            run: Box::new(move |input| {
                let kind = input[0].kind();
                let (resid, post) = (
                    self.resid_lambdas.to_kind(kind),
                    self.post_lambdas.to_kind(kind),
                );
                let x0 = self
                    .x0_lambdas
                    .as_ref()
                    .map(|bank| bank.to_kind(kind).get(0));
                let (resid_attn, resid_ffn) = (resid.get(0), resid.get(1));
                let (post_attn, post_ffn) = (post.get(0), post.get(1));
                let lambdas = BlockLambdas {
                    resid: [&resid_attn, &resid_ffn],
                    post: [&post_attn, &post_ffn],
                    x0: x0.as_ref().map(|lambda| (&input[1], lambda)),
                };
                block.forward(&input[0], None, &lambdas, rotation, train).0
            }),
        });
        // The value-residual mix, at the shape a decoder layer runs it: one `lerp` over two
        // head-shaped values. Its own class rather than part of the composed layer because the
        // source layer does not run it.
        if let Some(decoder) = self.blocks.get(1) {
            let lambda = decoder
                .value_lambda
                .as_ref()
                .expect("layer 1 mixes against layer 0's value");
            classes.push(KernelClass {
                name: "value residual mix",
                inputs: vec![activation(width), activation(width)],
                forward_bytes: 3. * state,
                forward_flops: 3. * tokens * width as f64,
                parameters: vec![lambda.shallow_clone()],
                run: Box::new(move |input| {
                    let head_shaped =
                        |value: &Tensor| value.reshape([rows, origins, heads, head_dim]);
                    head_shaped(&input[0]).lerp_tensor(
                        &head_shaped(&input[1]),
                        &lambda.to_kind(input[0].kind()),
                    )
                }),
            });
        }
        classes.push(KernelClass {
            name: "patch embedding tokens",
            // No gradient: prices, auxiliaries and statistics are all data, so nothing here is
            // retained past the cast and there is no backward pass to charge.
            inputs: Vec::new(),
            forward_bytes: {
                let patch = c.patch_len as f64;
                let aux_channels = c.features.channels() as f64;
                let prices = fp32(tokens * patch * CHANNELS as f64);
                let auxiliaries = fp32(tokens * patch * aux_channels);
                // Prices: subtract the origin close, scale by 1/σ (two fp32 passes, 2 each).
                // Auxiliaries: one fp32 pass. Per-origin auxiliary scale: two small fp32 ops.
                // Then one bf16 cast each (read fp32, write half) and the bf16 concatenation.
                4. * prices
                    + 2. * auxiliaries
                    + 4. * fp32(tokens * aux_channels)
                    + 1.5 * (prices + auxiliaries)
                    + (prices + auxiliaries)
            },
            parameters: Vec::new(),
            forward_flops: tokens * c.patch_len as f64 * (2. * CHANNELS as f64 + c.features.channels() as f64),
            run: Box::new(move |_| self.tokens(batch, stats)),
        });
        // The two forms this work replaced, measured in the SAME process against the same
        // device peaks so the before/after table is a comparison rather than two runs. They are
        // bit-identical to their replacements (`the_packed_rotation_and_split_match_the_per_
        // tensor_reference_bit_for_bit` pins that), so only their cost differs.
        let rope = RotaryEmbedding::new(origins, head_dim, head_dim, self.rotation.0.device());
        let (cosine, sine) = rope.cached_rotation(
            &Tensor::arange(origins, (Kind::Int64, self.rotation.0.device())),
            Kind::BFloat16,
        );
        classes.push(KernelClass {
            name: "reference rotary (per tensor)",
            inputs: vec![activation(2 * width)],
            forward_bytes: 18. * state,
            forward_flops: 6. * tokens * width as f64,
            parameters: Vec::new(),
            run: Box::new(move |input| {
                let parts = input[0].split(width, -1);
                let heads_of = |part: &Tensor| {
                    part.reshape([rows, origins, heads, head_dim]).transpose(1, 2)
                };
                let query = rope.apply_cached(&heads_of(&parts[0]), &cosine, &sine);
                let key = rope.apply_cached(&heads_of(&parts[1]), &cosine, &sine);
                Tensor::stack(&[query, key], 2)
            }),
        });
        classes.push(KernelClass {
            name: "reference embedding (cast after cat)",
            inputs: Vec::new(),
            forward_bytes: classes[classes.len() - 2].forward_bytes,
            forward_flops: classes[classes.len() - 2].forward_flops,
            parameters: Vec::new(),
            run: Box::new(move |_| {
                let (context, patch) = (c.seq_len, c.patch_len);
                let aux_channels = c.features.channels() as i64;
                let inv_sigma = per_bar(&stats.sigma.reciprocal());
                let prices = (batch
                    .log_prices
                    .narrow(1, 0, context)
                    .reshape([rows, origins, patch, CHANNELS])
                    - per_bar(&stats.log_close))
                    * &inv_sigma;
                let auxiliaries = batch
                    .aux
                    .narrow(1, 0, context)
                    .reshape([rows, origins, patch, aux_channels])
                    * (&self.sigma_scale * inv_sigma + &self.unit_scale);
                Tensor::cat(&[prices, auxiliaries], 3)
                    .reshape([rows, origins, patch * (CHANNELS + aux_channels)])
                    .to_kind(Kind::BFloat16)
            }),
        });
        // The head and the loss, which the backbone-only list above never charged at all. The
        // dense head emits `tokens · 2·CHANNELS · pred_len` = 147 M elements per step and the
        // loss walks that space channel by channel, so a step-level attribution that stops at
        // the last block cannot say where its own milliseconds went.
        let horizon = c.pred_len;
        let outputs = OUTPUTS_PER_BAR * horizon;
        // One `[rows, origins', 1, pred_len]` channel slice, the unit the loss works in.
        let slice = tokens * horizon as f64;
        let head_space = slice * OUTPUTS_PER_BAR as f64;
        let target_space = slice * CHANNELS as f64;
        if let Some(covariates) = &self.covariates {
            let known = covariates.ws.size()[1];
            classes.push(KernelClass {
                name: "head known-future covariate projection",
                inputs: vec![(vec![rows, origins, known], Kind::BFloat16)],
                forward_bytes: bf16(tokens * known as f64)
                    + bf16(tokens * COVARIATE_WIDTH as f64)
                    + cast(COVARIATE_WIDTH * known),
                forward_flops: 2. * tokens * known as f64 * COVARIATE_WIDTH as f64,
                parameters: projected(covariates),
                run: Box::new(move |input| linear(&input[0], covariates)),
            });
        }
        let head_input = self.head_hidden.ws.size()[1];
        classes.push(KernelClass {
            name: "head hidden projection and GELU",
            inputs: vec![(vec![rows, origins, head_input], Kind::BFloat16)],
            // The GEMM writes its `[tokens, 1024]` output and the GELU reads it and writes
            // another: two materialized activations, not one.
            forward_bytes: bf16(tokens * head_input as f64)
                + 3. * bf16(tokens * HEAD_HIDDEN as f64)
                + cast(HEAD_HIDDEN * head_input),
            forward_flops: gemm(head_input, HEAD_HIDDEN) + 8. * tokens * HEAD_HIDDEN as f64,
            parameters: projected(&self.head_hidden),
            run: Box::new(move |input| linear(&input[0], &self.head_hidden).gelu("none")),
        });
        classes.push(KernelClass {
            name: "head output projection",
            inputs: vec![(vec![rows, origins, HEAD_HIDDEN], Kind::BFloat16)],
            // The `[1536, 1024]` weight cast and the horizon-mean expansion's own two small
            // matmuls are parameter-space work; the activation cost is the hidden state in and
            // the whole 147 M-element head space out.
            forward_bytes: bf16(tokens * HEAD_HIDDEN as f64)
                + bf16(head_space)
                + cast(outputs * HEAD_HIDDEN),
            forward_flops: gemm(HEAD_HIDDEN, outputs),
            parameters: {
                let mut params = vec![self.head_output.ws.shallow_clone()];
                params.extend(self.head_output.bs.iter().map(Tensor::shallow_clone));
                params
            },
            run: Box::new(move |input| {
                let (weight, bias) = self.head_output_weights(input[0].kind());
                input[0]
                    .linear(&weight, bias.as_ref())
                    .reshape([rows, -1, OUTPUTS_PER_BAR, horizon])
            }),
        });
        // Registered only when the prior actually runs, so an unpenalized arm's class list and
        // its `step_cost` describe the same kernels. The mask is built outside the timed
        // closure exactly as the fused loss class below builds its targets.
        if c.amplitude_prior > 0. {
            let amplitude_mask = self.targets(batch, stats, false).1;
            classes.push(KernelClass {
                name: "amplitude prior mean energy",
                inputs: vec![(vec![rows, origins, 1, horizon], Kind::Float)],
                // Eight fp32 passes over the close mean slice: the masked multiply reads two
                // and writes one, its reduction reads one, the square-multiply reads two and
                // writes one, and its reduction reads one. The two `[pred_len]` outputs are
                // 768 B and do not round.
                forward_bytes: 8. * fp32(slice),
                // One masked multiply, one square multiply and two summations.
                forward_flops: 4. * slice,
                parameters: Vec::new(),
                run: Box::new(move |input| {
                    self.amplitude_prior(&input[0], &amplitude_mask)
                        .expect("the amplitude prior class is registered only at nonzero λ")
                }),
            });
        }
        classes.push(KernelClass {
            name: "targets, market drift and validity mask",
            // No gradient: prices, the market path, the validity flags and the statistics are
            // all data. This is pure forward traffic that the backward never revisits.
            inputs: Vec::new(),
            // `(future - log_close) / sigma - drift` is three fp32 passes over the target
            // space, the drift two over the `[.., 1, pred_len]` market space, and the mask one
            // more; each pass reads and writes.
            forward_bytes: 6. * fp32(target_space) + 6. * fp32(slice),
            forward_flops: 3. * target_space + 4. * slice,
            parameters: Vec::new(),
            run: Box::new(move |_| self.targets(batch, stats, false).0),
        });
        let (targets, mask) = self.targets(batch, stats, false);
        classes.push(KernelClass {
            name: "fused loss geometry and NLL",
            inputs: vec![(vec![rows, origins, OUTPUTS_PER_BAR, horizon], Kind::BFloat16)],
            // ENUMERATED, not a floor, because after the fusion there is nothing left to
            // guess: one mask fold (read the mask, write `mask·w`), the fused kernel's reads
            // (the whole bf16 head space, the fp32 targets, the folded mask) and writes (the
            // mean coordinate plus the twelve fp32 vectors the reductions consume), the twelve
            // `dot`s at two vectors each, and the three `[pred_len]`-or-scalar mask reductions
            // the `½·ln h` prior and the two denominators need. 51 channel-slice units in all.
            //
            // The composition this replaced was 65 kernels and 136 such units, and the old
            // figure here charged 43 of them - which is why it reported 783 GB/s for a chain
            // that was actually streaming 1613 GB/s. A floor in the denominator of a roofline
            // fraction understates the kernel and hides the fact that the win available was
            // never bandwidth but pass count.
            forward_bytes: 2. * fp32(slice)
                + bf16(head_space)
                + fp32(target_space)
                + fp32(slice)
                + 13. * fp32(slice)
                + 24. * fp32(slice)
                + 3. * fp32(slice),
            forward_flops: 24. * slice + CHANNELS as f64 * 8. * slice,
            parameters: Vec::new(),
            run: Box::new(move |input| {
                self.losses(&Head(input[0].shallow_clone()), stats, &targets, &mask)
                    .nll
            }),
        });
        classes
    }

    /// The dense per-origin head: known-future covariates, the two head GEMMs and the raw
    /// channel-major coordinate/log-scale space. `state` comes from [`Self::backbone`] under the
    /// same `last_only` scope.
    pub fn head(&self, batch: &Batch, state: &Tensor, last_only: bool) -> Head {
        let horizon = self.config.pred_len;
        let rows = state.size()[0];
        let head_input = match &self.covariates {
            Some(covariates) => {
                // bf16 BEFORE the gather: `flatten` over the unfolded window dimensions is a
                // copy either way, and casting first halves what it writes. The transpose keeps
                // the historical `(bar, channel)` feature order of the covariate projection.
                let known = self
                    .future_windows(
                        &batch
                            .aux
                            .index_select(2, &self.known_index)
                            .to_kind(Kind::BFloat16),
                        last_only,
                    )
                    .transpose(2, 3)
                    .flatten(2, 3);
                Tensor::cat(&[state, &linear(&known, covariates)], 2)
            }
            None => state.shallow_clone(),
        };
        let hidden = linear(&head_input, &self.head_hidden).gelu("none");
        let (weight, bias) = self.head_output_weights(hidden.kind());
        let emitted = hidden.linear(&weight, bias.as_ref());
        let Some(expansion) = &self.scale_expansion else {
            return Head(emitted.reshape([rows, -1, OUTPUTS_PER_BAR, horizon]));
        };
        // `increment`: the projection emits `CHANNELS·pred_len` increment means followed by
        // `CHANNELS·scales` log-scale coefficients, and this is the ONE place the two are put
        // back into the dense `[2·CHANNELS, pred_len]` block every consumer below expects. The
        // means need no expansion at all - they are already per-bar and full rank, and the
        // cumsum that turns them into a horizon forecast happens at the decode, not here,
        // because the loss is computed on the increments themselves.
        let means = horizon * CHANNELS;
        let coefficients = emitted
            .narrow(-1, means, emitted.size()[emitted.dim() - 1] - means)
            .reshape([rows, -1, CHANNELS, expansion.size()[0]]);
        Head(Tensor::cat(
            &[
                emitted.narrow(-1, 0, means).reshape([rows, -1, CHANNELS, horizon]),
                coefficients
                    .to_kind(expansion.kind())
                    .matmul(expansion)
                    .to_kind(emitted.kind()),
            ],
            2,
        ))
    }

    /// The head's output weight and bias in `kind`.
    ///
    /// The μP output multiplier rides on the weight copy the GEMM already needs: scaling a
    /// 1024×1536 weight instead of a 147 M-element output is the same product and the same
    /// gradient (`HEAD_OUTPUT_SCALE` is a power of two, so the cast commutes exactly) for
    /// 1/96 000 of the traffic.
    ///
    /// [`HorizonMean::Basis`] rides the same weight copy, for the same reason one order of
    /// magnitude further: `Ψ · W` is `[1536, 832] × [832, 1024]`, 2.6 GFLOP against the
    /// 0.5 TFLOP the token-space head GEMM would have to redo if the expansion happened in
    /// activation space - and it emits the identical dense block, so [`Head`], the fused loss
    /// and every per-horizon diagnostic are shape-identical in both modes. The one-hot rows of
    /// `Ψ` (the free band and every log scale) sum one weight against 831 exact zeros, so they
    /// arrive bit-identical to the parameter itself.
    ///
    /// `Free` takes the `None` branch and runs the pre-knob GEMM unchanged, which is what makes
    /// the control arm bit-for-bit the old head rather than merely equal to it.
    ///
    /// Its own method so `kernel_classes` can time exactly this rather than a paraphrase of it.
    fn head_output_weights(&self, kind: Kind) -> (Tensor, Option<Tensor>) {
        let weight = self.head_output.ws.to_kind(kind) * HEAD_OUTPUT_SCALE;
        let bias = self
            .head_output
            .bs
            .as_ref()
            .map(|bias| bias.to_kind(kind) * HEAD_OUTPUT_SCALE);
        match &self.mean_expansion {
            None => (weight, bias),
            Some(expansion) => {
                let expansion = expansion.to_kind(kind);
                let folded_bias = bias.map(|bias| expansion.matmul(&bias));
                (expansion.matmul(&weight), folded_bias)
            }
        }
    }

    /// fp32 channel-major coordinates and log predictive scales for the evaluation decoders.
    /// The training loss never calls this: it would materialize the whole 590 MB fp32 space.
    ///
    /// `log_scale` is always in HORIZON space, which under a non-identity [`TargetBasis`] means
    /// it is DERIVED rather than emitted: the head's rows are per-coefficient scales `τ_k`, the
    /// implied horizon covariance is `Wᵀ diag(τ²) W`, and `σ_h = √Σ_hh`. Deriving it HERE, at
    /// the one place that materializes the evaluation output, is what keeps every existing
    /// per-horizon report - calibration coverage, the scale charts, the trading family -
    /// reading a per-horizon scale that still MEANS a per-horizon scale. A report that read the
    /// raw head rows under a rotation would be charting coefficients on a horizon axis.
    pub fn output(&self, head: &Head) -> Output {
        let capped = |prior: &Tensor| {
            (head.0.narrow(2, CHANNELS, CHANNELS).to_kind(Kind::Float) / LOG_SCALE_CAP).tanh()
                * LOG_SCALE_CAP
                + prior
        };
        if self.increment_geometry.is_some() {
            // The head's rows are PER-BAR here, so the horizon-space scale is DERIVED, exactly
            // as it is under a rotation and for the same reason: every per-horizon report -
            // calibration coverage, the scale charts, the trading family - must keep reading a
            // quantity that still MEANS a per-horizon scale.
            //
            // `σ_h² = Σ_{j≤h} s_j² · A_h`, and this evaluates it at `A_h ≡ 1`, the SERIAL
            // INDEPENDENCE assumption. That assumption is MEASURED FALSE - `V_h/D_h` is ~.59 on
            // training and swings to ~.65 on a recent window against ~.40 on held-out full - so
            // this mode makes no calibration claim until an aggregation is supplied. It is
            // stated here rather than hidden because the alternative is a frozen `A_h`, and a
            // frozen first-moment-like multiplier biases every interval by an amount nobody
            // bounded, where this one is wrong in a named direction by a measured factor.
            let per_bar_log = (head.0.narrow(2, CHANNELS, CHANNELS).to_kind(Kind::Float)
                / LOG_SCALE_CAP)
                .tanh()
                * LOG_SCALE_CAP;
            return Output {
                coordinates: head.0.narrow(2, 0, CHANNELS).to_kind(Kind::Float),
                log_scale: (per_bar_log * 2.).exp().cumsum(-1, Kind::Float).log() * 0.5,
            };
        }
        Output {
            coordinates: head.0.narrow(2, 0, CHANNELS).to_kind(Kind::Float),
            log_scale: match &self.basis {
                // Bit-for-bit the pre-knob expression. `W = I` makes the derivation the
                // identity mathematically, but `½·ln(exp(2·ls))` is not the identity on the
                // bits, so the control arm must not evaluate it.
                None => capped(&self.half_log_horizon),
                Some(basis) => basis.horizon_log_scale(&capped(basis.half_log_prior())),
            },
        }
    }

    /// Market log return from each origin to bars `t_k+1..=t_k+pred_len`, scaled by the origin's
    /// causal β and in units of its σ, `[batch, origins', 1, pred_len]`; `stats` narrowed like in
    /// [`Self::targets`]. Adding it back to the targets recovers the ticker's own σ-scaled return.
    pub fn market_drift(&self, batch: &Batch, stats: &Statistics, last_only: bool) -> Tensor {
        let future = self.future_windows(&batch.market_cum, last_only);
        assert_eq!(future.size()[1], stats.sigma.size()[1]);
        ((future - stats.market.unsqueeze(-1)) * (&stats.beta / &stats.sigma).unsqueeze(-1))
            .unsqueeze(2)
    }

    /// Market-neutral σ-scaled log-return targets relative to each origin close and their
    /// validity mask, `[batch, origins', 4, pred_len]` and `[batch, origins', 1, pred_len]`: the
    /// ticker's log return minus β_k times the cumulative market log return over the same bars,
    /// divided by the ticker's own causal σ (the ticker's exposure to the market drift over the
    /// horizon is removed from the mean; σ stays the causal ticker scale so the persistence
    /// prior and decoder are unchanged).
    /// `stats` must already be narrowed to the scored origins (`Statistics::last` when
    /// `last_only`).
    pub fn targets(&self, batch: &Batch, stats: &Statistics, last_only: bool) -> (Tensor, Tensor) {
        let future = self.future_windows(&batch.log_prices, last_only);
        assert_eq!(future.size()[1], stats.sigma.size()[1]);
        let targets = (future - per_bar(&stats.log_close)) / per_bar(&stats.sigma)
            - self.market_drift(batch, stats, last_only);
        let mask =
            (self.future_windows(&batch.valid, last_only) * stats.mask.unsqueeze(-1)).unsqueeze(2);
        (targets, mask)
    }

    /// Point forecast in σ units for `output` produced under the same `last_only` scope.
    ///
    /// The applied amplitude calibration ([`Self::set_mean_gain`]) is the last step, in
    /// DECODED space, so it lands identically under both mean parameterizations: the increment
    /// branch below accumulates its per-bar candles first and the gain then rescales the
    /// cumulative anchor, which is the quantity the amplitude was measured on.
    pub fn decode(&self, output: &Output, stats: &Statistics) -> Tensor {
        let (sigma, range) = (per_bar(&stats.sigma), per_bar(&stats.range));
        let Some((unit, _, _)) = &self.increment_geometry else {
            return self.gained(decode_joint(
                &output.coordinates,
                &sigma,
                &range,
                &self.horizon_scale,
            ));
        };
        // Per-bar candles first - `decode_joint` at unit horizon scale, since a one-bar move in
        // σ units has scale exactly 1 - then the cumulative forecast. Only the CLOSE channel
        // accumulates; open, high and low at bar `j` are that bar's own extremes measured from
        // bar `j-1`'s close, so they ride the running close rather than summing. This is the
        // exact inverse of [`Self::increment_pair`], which is what makes the emitted
        // 192-horizon forecast identical in shape and meaning to the control's.
        let increments = decode_joint(&output.coordinates, &sigma, &range, unit);
        let horizon = increments.size()[increments.dim() - 1];
        let close = increments
            .narrow(-2, CHANNELS - 1, 1)
            .cumsum(-1, Kind::Float);
        let previous = Tensor::cat(
            &[
                close.narrow(-1, 0, 1).zeros_like(),
                close.narrow(-1, 0, horizon - 1),
            ],
            -1,
        );
        self.gained(Tensor::cat(
            &[
                increments.narrow(-2, 0, CHANNELS - 1) + previous,
                close,
            ],
            -2,
        ))
    }

    /// The training-time amplitude prior: a penalty on the ENERGY of the predicted mean
    /// FUNCTION, `None` at `λ = 0`.
    ///
    /// ```text
    /// R = (λ/(2·H)) · Σ_h w_h · (1/N_h) · Σ_b mask_bh·(m_bh - m̄_h)²,  m̄_h = Σ_b mask·m / N_h
    /// ∂R/∂m_bh = (λ/(H·N_h)) · w_h · mask_bh · (m_bh - m̄_h)
    /// ```
    ///
    /// `m` is `close`, the σ-scaled close mean coordinate [`Self::losses`] already
    /// materializes - i.e. AFTER the `√h` decode factor - and `N_h` is that horizon's valid
    /// bar count, so with an all-valid mask this is exactly `(λ/(2·B·H))·Σ_b Σ_h w_h(m - m̄)²`
    /// and its pre-registered gradient. `w_h` is the objective's OWN horizon weight buffer, so
    /// the prior cannot silently re-weight the horizon axis behind the loss's back.
    ///
    /// # Why a penalty on values, and not a multiplier
    ///
    /// A fixed output multiplier is pure reparameterization: the head's last GEMM absorbs it
    /// in a few hundred steps and the amplitude comes back. That is exactly what happened to
    /// `--horizon-mean basis:8:8`, which restricted the horizon SHAPE to a low-rank span while
    /// leaving its coefficients unbounded, so smooth over-amplitude survived intact (its
    /// `C` at h=192 was `-0.0373` against the control's `-0.0385`). Penalizing the emitted
    /// VALUES cannot be undone that way: scaling an upstream weight by `k` scales `R` by `k²`.
    ///
    /// The batch mean is removed so the prior attacks amplitude and not level - the whole
    /// level opportunity at these horizons is `ȳ²/E[y²] ≤ 4.6e-5` of the persistence MSE, and
    /// a penalty that fought over it would be spending the gradient on nothing.
    ///
    /// # What it does to the equilibrium, and why one λ suffices for 192 horizons
    ///
    /// The NLL's mean gradient carries the precision factor `1/s²` with `s ≈ σ√h`, while this
    /// penalty carries no `h` at all. For a linear predictor `m = a·g` the stationary point of
    /// `E[(m-y)²]/(8·B·H·s²) + R` is `a = a*/(1 + 4λs²)`, so ONE scalar λ produces the
    /// horizon-increasing shrinkage `1/(1 + 4λh)` - flat at the short end where the measured
    /// `β̂` is already at or above 1, and strong at the long end where it is `0.26`. That
    /// coincidence is the reason this is a single knob rather than a curve. HYPOTHESIS: the
    /// optimizer reaches that stationary point; the arm in the report is what tests it.
    ///
    /// # Cost
    ///
    /// At `rows = 256`, `origins = 375`, `H = 192`, one channel: 18.4 M elements, 73.7 MB per
    /// fp32 pass. One masked multiply, one square-multiply and two `[H]` reductions forward,
    /// their transposes in backward: ≈ 0.11 GFLOP and ≈ 1.0 GB of traffic against the step's
    /// 16.98 TFLOP and 143.4 GB - 6e-6 of the arithmetic and 0.7% of the traffic, so a
    /// predicted `+0.5` to `+0.9` ms on a 167 ms step. One 73.7 MB intermediate is retained
    /// for the backward, so peak memory rises by that and not by zero.
    pub fn amplitude_prior(&self, close: &Tensor, mask: &Tensor) -> Option<Tensor> {
        let lambda = self.config.amplitude_prior;
        if lambda == 0. {
            return None;
        }
        // Reduce over rows, origins and the singleton channel: the surviving axis is the
        // horizon, which is the axis the penalty is weighted and normalized along.
        let batch = [0i64, 1, 2];
        let counts = mask
            .sum_dim_intlist(batch.as_slice(), false, Kind::Float)
            .clamp_min(1.0)
            .detach();
        let masked = close * mask;
        let first = masked.sum_dim_intlist(batch.as_slice(), false, Kind::Float);
        let second = (&masked * close).sum_dim_intlist(batch.as_slice(), false, Kind::Float);
        // `Σ mask·(m - m̄)² = Σ mask·m² - (Σ mask·m)²/N`, so the whole penalty is two `[H]`
        // reductions and the autograd of that identity is the pre-registered gradient exactly.
        let dispersion = (second - first.square() / &counts) / &counts;
        Some(
            dispersion.dot(&self.horizon_weight.reshape([-1]))
                * (lambda / (2. * self.config.pred_len as f64)),
        )
    }

    /// Masked Gaussian NLL of the dense head: ONE elementwise kernel, twelve ATen `dot`s, and
    /// ONE backward kernel.
    ///
    /// `targets` is `[rows, origins', CHANNELS, pred_len]` and `mask` `[rows, origins', 1,
    /// pred_len]`, both from [`Self::targets`]. Mathematically this is exactly
    /// `gaussian_nll(decode_joint(coordinates, ..), log_scale, targets, mask)`, and
    /// numerically it is bit-for-bit the ATen composition it replaced
    /// ([`fused_kernels::reference::loss_geometry`]) - see
    /// `fused_loss_geometry_is_bit_identical_at_the_production_shape`.
    ///
    /// The algebra that makes the chain cheap in the first place is unchanged and still
    /// load-bearing:
    ///
    /// - `1/σ` is applied to the three candle offsets rather than to their three differences, so
    ///   `low` is shared by `high` and `open` instead of recomputed;
    /// - `exp(-2·ls)` is factored as `exp(-2·CAP·tanh(u))·(1/h)`, which removes the `+ ½·ln h`
    ///   add over the full space, and the `Σ mask·½·ln h` it leaves behind is a gradient-free
    ///   constant reduced over `[pred_len]`;
    /// - the mask multiply is folded into the per-element precision weight, and the reductions
    ///   are `dot`s, so nothing writes a full-size masked copy of the NLL or of the error;
    /// - the per-horizon objective weight is folded into the mask ONCE, at
    ///   `[rows, origins', 1, pred_len]`, so the weighting never touches the channel space.
    ///
    /// What the fusion adds on top of that is the deletion of the intermediates. The
    /// composition ran 65 kernels forward and 70 backward over the 18.4 M-element channel
    /// slice - 136 fp32 slice-passes forward and 179 backward, measured at 1613 and 1470 GB/s,
    /// i.e. already at 82-90% of this card's 1.79 TB/s streaming roof. Nothing there was slow;
    /// there were simply 135 passes. This runs 17 forward and 1 backward, 51 and 13 slice-units
    /// (see [`ModelConfig::step_cost`]), and retains no full-size fp32 tensor at all: the
    /// backward recomputes the geometry from `head` rather than reading twelve saved vectors.
    ///
    /// The NLL is `Σ w·mask·nll / Σ w·mask·CHANNELS` - a weighted MEAN, hence still nats per
    /// bar and invariant to the scale of `w`. The MSE keeps the UNWEIGHTED denominator and the
    /// raw mask: it is a diagnostic that every report and every arm has to be able to compare,
    /// so its definition does not move with the objective.
    pub fn losses(
        &self,
        head: &Head,
        stats: &Statistics,
        targets: &Tensor,
        mask: &Tensor,
    ) -> Losses {
        // The amplitude calibration is a SCORING transform on the decoded mean and no loss
        // chain here reads its buffers, so a calibrated model would train against its own
        // un-gained output. That state is unreachable by construction - the run fits its gain
        // at evaluation and stamps it into the manifest instead of applying it - and this is
        // what keeps it unreachable by accident.
        assert!(
            self.mean_gain.is_none(),
            "a calibrated model must not be trained: the applied mean gain is a scoring \
             transform and the training objective does not carry it"
        );
        // The ONE branch the basis knob adds to this path, above everything the fused chain
        // does. Under `cumulative` - the control - `self.basis` is `None` and the code below is
        // reached exactly as it was before the knob existed, so the control arm is bit-for-bit
        // the pre-knob objective rather than merely equal to it.
        if let Some(basis) = &self.basis {
            return self.basis_losses(basis, head, stats, targets, mask);
        }
        // The increment branch, and the whole arm is in the two substitutions it makes: the
        // targets become per-BAR differences, and the three per-horizon constants become the
        // one-bar ones. Everything else - the fused geometry, all twelve reductions, the
        // amplitude prior - is the same code on the same shapes, which is what makes the
        // difference between the arms attributable to the objective's SPACE and to nothing
        // else.
        if let Some((scale, half_log, inverse)) = &self.increment_geometry {
            let (targets, mask) = Self::increment_pair(targets, mask);
            let (sigma, range, weighted_mask) = self.loss_operands(stats, &mask);
            let geometry = loss_geometry(
                &head.0,
                &targets,
                &weighted_mask,
                &mask,
                &sigma,
                &range,
                scale,
                inverse,
                &self.log_scale_gain,
                LOG_SCALE_CAP,
            );
            return self.reduce_geometry(&geometry, &mask, &weighted_mask, half_log);
        }
        let (sigma, range, weighted_mask) = self.loss_operands(stats, mask);
        let geometry = loss_geometry(
            &head.0,
            targets,
            &weighted_mask,
            mask,
            &sigma,
            &range,
            &self.horizon_scale,
            &self.inverse_horizon,
            &self.log_scale_gain,
            LOG_SCALE_CAP,
        );
        self.reduce_geometry(&geometry, mask, &weighted_mask, &self.half_log_horizon)
    }

    /// The SAME objective with the composed-ATen geometry instead of the fused kernel, for
    /// the paired benchmark arm and for the equivalence tests.
    ///
    /// This is not a switch and nothing on the training path can reach it: [`Self::losses`]
    /// carries no flag, no fallback and no branch on a kernel choice, and this function is
    /// called only by `benchmark.rs`'s paired arm and by tests. It exists because a
    /// cross-RUN step-time comparison is not defensible on a shared card - the 5452
    /// measurement fell 16.3% on the machine's own measured GEMM peak, and a step that is
    /// part arithmetic-bound, part bandwidth-bound and part launch-bound cannot be
    /// normalized by one scalar. Running both chains in ONE process, alternated, makes
    /// contention common to both arms so it cancels in the difference.
    ///
    /// The only difference from [`Self::losses`] is which geometry provider is called: the
    /// operand preparation and all twelve reductions are literally the same code. That is
    /// what makes the paired difference a measurement of the fusion rather than of two
    /// separately-written objectives, and `fused_kernels`' own bit-exactness suite is what
    /// makes the two arms' loss VALUES identical rather than merely close.
    pub fn composed_losses(
        &self,
        head: &Head,
        stats: &Statistics,
        targets: &Tensor,
        mask: &Tensor,
    ) -> Losses {
        assert!(
            self.basis.is_none(),
            "the composed reference arm is defined on the dense head; a target basis has its \
             own chain in `basis_losses`"
        );
        let (sigma, range, weighted_mask) = self.loss_operands(stats, mask);
        let geometry = fused_kernels::reference::loss_geometry(
            &head.0,
            targets,
            &weighted_mask,
            mask,
            &sigma,
            &range,
            &self.horizon_scale,
            &self.inverse_horizon,
            &self.log_scale_gain,
            LOG_SCALE_CAP,
        );
        self.reduce_geometry(&geometry, mask, &weighted_mask, &self.half_log_horizon)
    }

    /// σ, ρ and the once-folded per-horizon objective weight, at mask width. `uniform` is
    /// `w = 1` exactly, so the fold is the identity on the bits.
    fn loss_operands(&self, stats: &Statistics, mask: &Tensor) -> (Tensor, Tensor, Tensor) {
        let smallest = f64::from(f32::MIN_POSITIVE);
        (
            per_bar(&stats.sigma).clamp_min(smallest),
            per_bar(&stats.range).clamp_min(smallest),
            mask * &self.horizon_weight,
        )
    }

    /// The twelve `dot`s and the gradient-free prior: everything the fusion deliberately did
    /// NOT absorb, because a reduction's summation tree is cuBLAS's and reassociating it
    /// moves the loss value.
    fn reduce_geometry(
        &self,
        geometry: &fused_kernels::LossGeometry,
        mask: &Tensor,
        weighted_mask: &Tensor,
        half_log: &Tensor,
    ) -> Losses {
        // `Some` only at nonzero λ, and the arithmetic is two `[pred_len]` reductions over the
        // mean coordinate the fused op already had to produce - see [`Self::amplitude_prior`].
        let amplitude = self.amplitude_prior(&geometry.close, mask);
        // `Σ w·mask·½·ln h` over channels: gradient-free, so it never belongs in a full-size
        // kernel. `mask` is `[rows, origins', 1, pred_len]`, so this reduces to `[pred_len]`.
        let prior = weighted_mask
            .sum_dim_intlist([0i64, 1, 2].as_slice(), false, Kind::Float)
            .dot(&half_log.reshape([-1]))
            * CHANNELS;
        let objective_count = (weighted_mask.sum(Kind::Float) * CHANNELS).clamp_min(1.0);
        let count = (mask.sum(Kind::Float) * CHANNELS).clamp_min(1.0);
        // The prior is already normalized per (origin, horizon), so it is added to the
        // NORMALIZED objective rather than folded into the numerator: λ then means the same
        // thing whatever share of the batch's bars happened to be valid, and the reported NLL
        // stays comparable to the control's only insofar as it genuinely includes the penalty.
        let nll = (geometry.terms.sum(Kind::Float) + prior) / objective_count;
        Losses {
            nll: match amplitude {
                Some(penalty) => nll + penalty,
                None => nll,
            },
            mse: geometry.squares.sum(Kind::Float) / count,
        }
    }

/// Per-BAR targets and their validity, from the cumulative pair. `targets` is
/// `[rows, origins', CHANNELS, pred_len]` relative to the ORIGIN close; the increment at bar `j`
/// is the same four coordinates relative to bar `j-1`'s CLOSE, which is `decode_joint`'s fourth
/// channel and the only one that is a level rather than an extreme of its own bar.
///
/// Bar 0 needs no shift: its previous close IS the origin close, which is exactly zero in these
/// σ-scaled targets. Its validity is `stats.mask`, already folded into `mask`.
///
/// A bar is valid as an increment only if it AND its predecessor were observed, so the mask is
/// the pointwise product of the two - one bar of validity is lost at every gap, and that is a
/// real cost of the parameterization rather than an accounting choice.
fn increment_pair(targets: &Tensor, mask: &Tensor) -> (Tensor, Tensor) {
    let horizon = targets.size()[targets.dim() - 1];
    let shift = |source: &Tensor, first: Tensor| {
        Tensor::cat(&[first, source.narrow(-1, 0, horizon - 1)], -1)
    };
    let close = targets.narrow(-2, CHANNELS - 1, 1);
    let previous = shift(&close, close.narrow(-1, 0, 1).zeros_like());
    let carried = shift(mask, mask.narrow(-1, 0, 1).ones_like());
    (targets - previous, mask * carried)
}

    /// The objective in COEFFICIENT space: [`Self::losses`] under a non-identity
    /// [`TargetBasis`].
    ///
    /// ```text
    /// r = y - ŷ            (horizon space, ŷ from decode_joint, a valid candle)
    /// e = W·r              (one [tokens·CHANNELS, H] × [H, H] fp32 GEMM)
    /// ls_k = CAP·tanh(u_k/CAP) + ½·ln(prior variance of coefficient k)
    /// NLL = Σ_k w_k·rowmask·(½·(e_k·exp(-ls_k))² + ls_k) / Σ_k w_k·rowmask·CHANNELS
    /// ```
    ///
    /// Three things are load-bearing and none of them is a stylistic choice.
    ///
    /// **The residual is rotated, not the two moments separately.** `W(y - ŷ) = Wy - Wŷ` by
    /// linearity, so one GEMM buys what two would, and under `W = I` the expression is
    /// bit-for-bit [`nll_elements`] - which is what makes `cumulative` a provable identity
    /// rather than an approximate one.
    ///
    /// **The mean is decoded in HORIZON space first.** The head's four mean coordinates go
    /// through [`decode_joint`] unchanged, so `high ≥ max(open, close) ≥ min(open, close) ≥ low`
    /// still holds bar by bar and the emitted forecast is still a valid candle. Reinterpreting
    /// the mean rows as coefficients directly would have destroyed that invariant, and the
    /// rotation of a valid candle is the same bijection either way.
    ///
    /// **The mask is the row COMPLETENESS indicator, not the per-bar mask.** A coefficient is a
    /// weighted sum over the whole horizon window, so a row whose window is partly unobserved
    /// has no defined coefficient vector; zero-filling its tail would inject a false zero
    /// return into all 192 coefficients. The corpus mask is a prefix mask and held-out targets
    /// are complete by contract, so only the training remainder is dropped -
    /// [`super::reports::write_target_basis`] reports what share that was.
    ///
    /// The MSE is unchanged in definition: horizon space, raw mask, unweighted denominator. It
    /// is the diagnostic every arm has to be comparable on, so it does not move with the metric.
    ///
    /// Cost, in the currency that is scarce. The rotation is 28.3 GFLOP forward at the
    /// production shape (96,000 origins × 4 channels × 192² × 2), 0.167% of the step's
    /// 16.98 TFLOP, and one read-write pair over the 295 MB coefficient space. What it costs is
    /// the composed fp32 chain around it: this is `fused_kernels::reference::loss_geometry`'s
    /// traffic profile, ~150 slice-passes forward over the 73.7 MB channel slice against the
    /// fused path's 51, so ~+13 ms on a 168 ms step. Fusing it is worth doing only if the arm
    /// is worth keeping.
    pub fn basis_losses(
        &self,
        basis: &BasisTransform,
        head: &Head,
        stats: &Statistics,
        targets: &Tensor,
        mask: &Tensor,
    ) -> Losses {
        let smallest = f64::from(f32::MIN_POSITIVE);
        let sigma = per_bar(&stats.sigma).clamp_min(smallest);
        let range = per_bar(&stats.range).clamp_min(smallest);
        let prediction = decode_joint(
            &head.0.narrow(2, 0, CHANNELS).to_kind(Kind::Float),
            &sigma,
            &range,
            &self.horizon_scale,
        );
        // The same expression [`Self::output`] uses, so the two cannot drift and the identity
        // basis reproduces the evaluation path's log scale on the bits.
        let log_scale = (head.0.narrow(2, CHANNELS, CHANNELS).to_kind(Kind::Float)
            / LOG_SCALE_CAP)
            .tanh()
            * LOG_SCALE_CAP
            + basis.half_log_prior();
        let error = targets - &prediction;
        // The mask is applied BEFORE the rotation, not only after it. On a complete window the
        // mask is exactly 1 and this multiply is the identity on the bits, so the `cumulative`
        // identity survives; on an incomplete one it is what makes an unobserved bar contribute
        // exact zero rather than `0 · large`. The unobserved tail of a short row is built from
        // zero-filled prices, so its target is a finite but arbitrary number of order
        // `ln(anchor)/σ`; relying on the row weight alone to annihilate it after a 192-term
        // rotation would be relying on a product that has already lost precision.
        let residual = basis.rotate(&(&error * mask));
        let weighted = mask.amin([-1i64].as_slice(), true) * basis.weight();
        let objective_count = (weighted.sum(Kind::Float) * CHANNELS).clamp_min(1.0);
        let count = (mask.sum(Kind::Float) * CHANNELS).clamp_min(1.0);
        let terms = (residual * (-&log_scale).exp()).square() * 0.5 + &log_scale;
        let nll = (terms * weighted).sum(Kind::Float) / objective_count;
        // The close coordinate the penalty reduces is decode_joint's fourth channel, which is
        // the same tensor the fused path hands back as `geometry.close`.
        let amplitude = self.amplitude_prior(&prediction.narrow(-2, CHANNELS - 1, 1), mask);
        Losses {
            nll: match amplitude {
                Some(penalty) => nll + penalty,
                None => nll,
            },
            mse: tch::no_grad(|| (error.square() * mask).sum(Kind::Float) / count),
        }
    }
}

/// Maps candle coordinates `[.., 4, pred_len]` to σ-scaled log returns relative to the origin
/// close; `sigma` and `range` broadcast as `[.., 1, 1]`. The close coordinate is in units of the
/// h-step persistence deviation σ√h, the range is a softplus multiple of the mean relative range,
/// and open/close positions are sigmoids inside it. Every channel is the close plus a monotone
/// offset, so `high >= max(open, close)` and `low <= min(open, close)` survive rounding.
pub fn decode_joint(
    coordinates: &Tensor,
    sigma: &Tensor,
    range: &Tensor,
    horizon_scale: &Tensor,
) -> Tensor {
    let smallest = f64::from(f32::MIN_POSITIVE);
    let sigma = sigma.clamp_min(smallest);
    let close = coordinates.narrow(-2, 0, 1) * horizon_scale;
    let relative_range =
        range.clamp_min(smallest) * coordinates.narrow(-2, 1, 1).softplus() / std::f64::consts::LN_2;
    let close_offset = (coordinates.narrow(-2, 2, 1).sigmoid() * &relative_range).log1p();
    let open_offset = (coordinates.narrow(-2, 3, 1).sigmoid() * &relative_range).log1p();
    let full = relative_range.log1p();
    let low = &close - &close_offset / &sigma;
    let high = &close + (full - &close_offset) / &sigma;
    let open = &close + (open_offset - close_offset) / sigma;
    Tensor::cat(&[open, high, low, close], -2)
}

/// Prices from σ-scaled log returns; fp64 intermediates clamped into the fp32 range.
pub fn decode_prices(scaled: &Tensor, anchor: &Tensor, sigma: &Tensor) -> Tensor {
    let smallest = f64::from(f32::MIN_POSITIVE);
    let largest = f64::from(f32::MAX);
    let log_prices = (scaled.to_kind(Kind::Double) * sigma.to_kind(Kind::Double))
        .clamp(smallest.ln(), largest.ln());
    (anchor.to_kind(Kind::Double).clamp_min(smallest) * log_prices.exp())
        .clamp(smallest, largest)
        .to_kind(Kind::Float)
}

/// Per-element Gaussian negative log-likelihood `½·((y - ŷ)/s)² + ln s` with `s = exp(log_scale)`.
pub fn nll_elements(prediction: &Tensor, log_scale: &Tensor, target: &Tensor) -> Tensor {
    ((target - prediction) * (-log_scale).exp()).square() * 0.5 + log_scale
}

pub struct Losses {
    pub nll: Tensor,
    pub mse: Tensor,
}

/// Masked means over valid (origin, channel, bar) triples; `mse` carries no gradient. The
/// reference form of [`CausalPatchModel::losses`], kept for the evaluation path and for the
/// equivalence test that pins the fused one.
///
/// `horizon_weight` broadcasts over the last dimension - `[1, 1, 1, pred_len]`, from
/// [`CausalPatchModel::horizon_weight_buffer`]. The NLL is the `w`-weighted mean, the MSE the
/// unweighted one, exactly as the fused form has it.
pub fn gaussian_nll(
    prediction: &Tensor,
    log_scale: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    horizon_weight: &Tensor,
) -> Losses {
    let weighted_mask = mask * horizon_weight;
    let objective_count = (weighted_mask.sum(Kind::Float) * CHANNELS).clamp_min(1.0);
    let count = (mask.sum(Kind::Float) * CHANNELS).clamp_min(1.0);
    let nll = (nll_elements(prediction, log_scale, target) * weighted_mask).sum(Kind::Float)
        / objective_count;
    let mse = tch::no_grad(|| ((target - prediction).square() * mask).sum(Kind::Float) / count);
    Losses { nll, mse }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The composed reference the fused kernels are checked against: normalize the packed
    // block, then rotate it. `Block::forward` runs ONE kernel for the pair, and these tests
    // are what pins the two to the same bytes.
    use fused_kernels::rope as fused_rope;
    use tch::Device;

    fn small_config() -> ModelConfig {
        ModelConfig {
            seq_len: 64,
            pred_len: 8,
            patch_len: 16,
            layers: 2,
            d_model: 32,
            heads: 4,
            ffn: 64,
            dropout: 0.0,
            min_history: 16,
            features: FeatureSet::ALL,
            x0_lambdas: X0Lambdas::Enabled,
            horizon_loss: HorizonLoss::Uniform,
            horizon_mean: HorizonMean::Free,
            amplitude_prior: 0.0,
            target_basis: TargetBasis::Cumulative,
            basis_weight: BasisWeight::Uniform,
            basis_stats: None,
        }
    }

    /// Random-walk rows in the corpus layout; `valid_future[i]` observed horizon bars per row.
    fn synthetic(config: &ModelConfig, valid_future: &[i64]) -> Batch {
        let rows = valid_future.len() as i64;
        let length = config.seq_len + config.pred_len;
        let aux_channels = config.features.channels() as i64;
        let close = Tensor::randn([rows, length, 1], (Kind::Float, Device::Cpu)).cumsum(1, Kind::Float) * 0.002;
        let close = &close - close.narrow(1, config.seq_len - 1, 1);
        let open = &close + Tensor::randn([rows, length, 1], (Kind::Float, Device::Cpu)) * 0.0005;
        let high = close.maximum(&open) + Tensor::rand([rows, length, 1], (Kind::Float, Device::Cpu)) * 0.001;
        let low = close.minimum(&open) - Tensor::rand([rows, length, 1], (Kind::Float, Device::Cpu)) * 0.001;
        let log_prices = Tensor::cat(&[open, high, low, close], 2);
        let valid = Tensor::ones([rows, length], (Kind::Float, Device::Cpu));
        for (row, &count) in valid_future.iter().enumerate() {
            let _ = valid
                .narrow(0, row as i64, 1)
                .narrow(1, config.seq_len + count, config.pred_len - count)
                .fill_(0.0);
        }
        let aux = Tensor::randn([rows, length, aux_channels], (Kind::Float, Device::Cpu));
        let market = Tensor::randn([rows, length], (Kind::Float, Device::Cpu)).cumsum(1, Kind::Float) * 0.001;
        let market = &market - market.narrow(1, config.seq_len - 1, 1);
        let anchor = Tensor::arange(rows, (Kind::Float, Device::Cpu)) + 100.0;
        let packed = Tensor::cat(
            &[
                log_prices.flatten(1, 2),
                valid,
                aux.flatten(1, 2),
                market,
                anchor.unsqueeze(1),
            ],
            1,
        );
        Batch::from_packed(
            packed,
            config.seq_len as usize,
            config.pred_len as usize,
            aux_channels as usize,
            valid_future.iter().sum::<i64>() as usize,
        )
    }

    #[test]
    fn requires_complete_uniform_patches_and_bounded_history() {
        let mut config = ModelConfig::default();
        config.validate().unwrap();
        config.seq_len = 97;
        assert!(config.validate().is_err());
        config.seq_len = 96;
        config.dropout = f64::NAN;
        assert!(config.validate().is_err());
        config.dropout = 0.0;
        config.min_history = 97;
        assert!(config.validate().is_err());
        config.min_history = 1;
        assert!(config.validate().is_err());
        config.min_history = 96;
        config.heads = 512 / 3;
        assert!(config.validate().is_err());
    }

    /// Channel-major `[.., 4, bars]`, matching what the model emits.
    fn assert_valid_candles(prices: &Tensor) {
        let open = prices.narrow(-2, 0, 1);
        let high = prices.narrow(-2, 1, 1);
        let low = prices.narrow(-2, 2, 1);
        let close = prices.narrow(-2, 3, 1);
        assert_eq!(prices.isfinite().all().int64_value(&[]), 1);
        assert_eq!(prices.gt(0.0).all().int64_value(&[]), 1);
        assert_eq!(
            high.ge_tensor(&open.maximum(&close)).all().int64_value(&[]),
            1
        );
        assert_eq!(
            low.le_tensor(&open.minimum(&close)).all().int64_value(&[]),
            1
        );
    }

    fn horizon(pred_len: i64) -> Tensor {
        (Tensor::arange(pred_len, (Kind::Float, Device::Cpu)) + 1.0)
            .sqrt()
            .reshape([1, 1, pred_len])
    }

    #[test]
    fn zero_coordinates_decode_to_persistence_and_keep_price_gradients() {
        let sigma = Tensor::from_slice(&[0.002f32, 0.05]).reshape([2, 1, 1]);
        let range = Tensor::from_slice(&[0.004f32, 0.3]).reshape([2, 1, 1]);
        let anchor = Tensor::from_slice(&[100.0f32, 0.000001]).reshape([2, 1, 1]);
        let coordinates =
            Tensor::zeros([2, 4, 3], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let scaled = decode_joint(&coordinates, &sigma, &range, &horizon(3));
        let prices = decode_prices(&scaled, &anchor, &sigma);
        assert_valid_candles(&prices);
        for step in 0..3 {
            assert_eq!(scaled.double_value(&[0, 3, step]), 0.0);
            assert_eq!(scaled.double_value(&[0, 0, step]), 0.0);
            assert_eq!(prices.double_value(&[0, 3, step]), 100.0);
            assert_eq!(prices.double_value(&[0, 0, step]), 100.0);
            assert_eq!(prices.double_value(&[1, 3, step]), f64::from(0.000001f32));
        }
        let low = prices.double_value(&[0, 2, 0]);
        let high = prices.double_value(&[0, 1, 0]);
        assert!(((high - low) / low - 0.004).abs() < 1e-6);
        let target = Tensor::from_slice(&[0.5f32, 1.0, -0.5, 0.25])
            .reshape([1, 4, 1])
            .expand_as(&scaled);
        (&scaled - target).square().mean(Kind::Float).backward();
        assert_eq!(coordinates.grad().isfinite().all().int64_value(&[]), 1);
        for coordinate in 0..4 {
            assert!(
                coordinates
                    .grad()
                    .narrow(1, coordinate, 1)
                    .abs()
                    .sum(Kind::Float)
                    .double_value(&[])
                    > 0.0
            );
        }
    }

    #[test]
    fn close_coordinate_is_measured_in_horizon_persistence_sigmas() {
        let sigma = Tensor::from_slice(&[0.01f32]).reshape([1, 1, 1]);
        let range = Tensor::from_slice(&[0.002f32]).reshape([1, 1, 1]);
        let coordinates = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0])
            .reshape([1, 4, 1])
            .expand([1, 4, 4], true);
        let scaled = decode_joint(&coordinates, &sigma, &range, &horizon(4));
        let prices = decode_prices(&scaled, &Tensor::from_slice(&[50.0f32]).reshape([1, 1, 1]), &sigma);
        for step in 0..4 {
            let steps = (step + 1) as f64;
            assert!((scaled.double_value(&[0, 3, step]) - steps.sqrt()).abs() < 1e-6);
            let expected = 50.0 * (0.01 * steps.sqrt()).exp();
            assert!((prices.double_value(&[0, 3, step]) - expected).abs() < 1e-4 * expected);
        }
    }

    #[test]
    fn joint_decoder_remains_valid_at_flat_and_saturated_position_limits() {
        let mut coordinates = Vec::new();
        for close in [-1e6f32, 0.0, 1e6] {
            for range in [-1000.0, 0.0, 1e6] {
                for close_position in [-1000.0, 0.0, 1000.0] {
                    for open_position in [-1000.0, 0.0, 1000.0] {
                        coordinates.extend([close, range, close_position, open_position]);
                    }
                }
            }
        }
        let count = coordinates.len() as i64 / 4;
        let coordinates = Tensor::from_slice(&coordinates)
            .reshape([1, count, 4])
            .transpose(1, 2)
            .contiguous()
            .set_requires_grad(true);
        let sigma = Tensor::from_slice(&[0.003f32]).reshape([1, 1, 1]);
        let range = Tensor::from_slice(&[0.001f32]).reshape([1, 1, 1]);
        let anchor = Tensor::from_slice(&[100.0f32]).reshape([1, 1, 1]);
        let scaled = decode_joint(&coordinates, &sigma, &range, &horizon(count));
        assert_valid_candles(&decode_prices(&scaled, &anchor, &sigma));
        scaled.mean(Kind::Float).backward();
        assert_eq!(coordinates.grad().isfinite().all().int64_value(&[]), 1);
        let flat = decode_prices(
            &decode_joint(
                &Tensor::from_slice(&[0.0f32, -1000.0, 0.0, 0.0]).reshape([1, 4, 1]),
                &sigma,
                &range,
                &horizon(1),
            ),
            &anchor,
            &sigma,
        );
        assert_eq!(flat.max().double_value(&[]), flat.min().double_value(&[]));
        assert_eq!(flat.max().double_value(&[]), 100.0);
        for extreme in [3e38f32, f32::MIN_POSITIVE] {
            let anchor = Tensor::from_slice(&[extreme]).reshape([1, 1, 1]);
            let coordinates = Tensor::from_slice(&[0.0f32, 100.0, -100.0, 0.0])
                .reshape([1, 4, 1])
                .set_requires_grad(true);
            let scaled = decode_joint(&coordinates, &sigma, &Tensor::from_slice(&[1.0f32]).reshape([1, 1, 1]), &horizon(1));
            let prices = decode_prices(&scaled, &anchor, &sigma);
            assert_valid_candles(&prices);
            assert_eq!(prices.double_value(&[0, 3, 0]), f64::from(extreme));
            scaled.mean(Kind::Float).backward();
            assert_eq!(coordinates.grad().isfinite().all().int64_value(&[]), 1);
        }
    }

    #[test]
    fn causal_sigma_matches_an_independent_expanding_std_over_masked_bars() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(7);
        let config = small_config();
        let batch = synthetic(&config, &[8, 3]);
        let _ = batch.valid.narrow(0, 1, 1).narrow(1, 2, 18).fill_(0.0);
        let _ = batch.valid.narrow(0, 1, 1).narrow(1, 40, 1).fill_(0.0);
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let stats = model.statistics(&batch);
        assert_eq!(stats.sigma.size(), [2, 4]);
        let closes: Vec<f32> = Vec::try_from(batch.log_prices.select(2, 3).flatten(0, 1)).unwrap();
        let valid: Vec<f32> = Vec::try_from(batch.valid.flatten(0, 1)).unwrap();
        let length = (config.seq_len + config.pred_len) as usize;
        for row in 0..2 {
            for origin in 0..4 {
                let end = 16 * (origin + 1);
                let base = row * length;
                let mut returns = Vec::new();
                let mut bars = 0;
                for t in 0..end {
                    if valid[base + t] > 0.0 {
                        bars += 1;
                    }
                    if t > 0 && valid[base + t] > 0.0 && valid[base + t - 1] > 0.0 {
                        returns.push(f64::from(closes[base + t]) - f64::from(closes[base + t - 1]));
                    }
                }
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance =
                    returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
                let expected = (variance + RETURN_VARIANCE_FLOOR).sqrt();
                let actual = stats.sigma.double_value(&[row as i64, origin as i64]);
                assert!((actual - expected).abs() <= 1e-5 * expected, "{actual} vs {expected}");
                assert_eq!(
                    stats.mask.double_value(&[row as i64, origin as i64]),
                    f64::from(u8::from(bars >= 16))
                );
                assert_eq!(
                    stats.log_close.double_value(&[row as i64, origin as i64]),
                    f64::from(closes[base + end - 1])
                );
            }
        }
        assert_eq!(stats.mask.double_value(&[1, 0]), 0.0);
        assert_eq!(stats.mask.double_value(&[1, 1]), 0.0);
        assert_eq!(stats.mask.double_value(&[1, 2]), 1.0);
        assert_eq!(stats.mask.double_value(&[0, 0]), 1.0);
    }

    #[test]
    fn targets_read_only_bars_after_each_origin() {
        let _rng = crate::torch::test_rng::shared();
        let config = small_config();
        let mut batch = synthetic(&config, &[8, 5]);
        let length = config.seq_len + config.pred_len;
        // Encode the bar index in every channel: y[k, h] must be (index - t_k) / σ_k with index = t_k + h.
        let indexed = (Tensor::arange(length, (Kind::Float, Device::Cpu)) * 1e-3)
            .reshape([1, length, 1])
            .expand([2, length, CHANNELS], true);
        batch.log_prices.copy_(&indexed);
        let _ = batch.market_cum.zero_();
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let stats = model.statistics(&batch);
        let (targets, mask) = model.targets(&batch, &stats, false);
        assert_eq!(targets.size(), [2, 4, 4, 8]);
        assert_eq!(mask.size(), [2, 4, 1, 8]);
        for origin in 0..4 {
            let sigma = stats.sigma.double_value(&[0, origin]);
            for step in 0..8 {
                for channel in 0..4 {
                    let expected = (step + 1) as f64 * 1e-3 / sigma;
                    let actual = targets.double_value(&[0, origin, channel, step]);
                    assert!((actual - expected).abs() <= 1e-3 * expected, "{actual} vs {expected}");
                }
                let bar = 16 * (origin + 1) + step;
                let expected_mask = if bar < config.seq_len + 5 { 1.0 } else { 0.0 };
                assert_eq!(mask.double_value(&[1, origin, 0, step]), expected_mask);
                assert_eq!(mask.double_value(&[0, origin, 0, step]), 1.0);
            }
        }
        let (last, last_mask) = model.targets(&batch, &stats.last(), true);
        assert!(last.equal(&targets.narrow(1, 3, 1)));
        assert!(last_mask.equal(&mask.narrow(1, 3, 1)));
        let windows = model.future_windows(&batch.log_prices, false).copy();
        let _ = batch.log_prices.narrow(1, 0, 32).fill_(5.0);
        let perturbed = model.future_windows(&batch.log_prices, false);
        assert!(
            perturbed.narrow(1, 1, 3).equal(&windows.narrow(1, 1, 3)),
            "windows of origins 1..3 must not read bars at or before their origin"
        );
        assert!(!perturbed.narrow(1, 0, 1).equal(&windows.narrow(1, 0, 1)));
    }

    #[test]
    fn a_ticker_tracking_the_market_has_zero_close_target() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(5);
        let config = small_config();
        let mut batch = synthetic(&config, &[8, 8]);
        let length = config.seq_len + config.pred_len;
        // Market path with drift and gaps; the ticker's close follows it bar for bar.
        let market = (Tensor::randn([2, length], (Kind::Float, Device::Cpu)) * 0.003 + 0.002)
            .cumsum(1, Kind::Float);
        let market = &market - market.narrow(1, config.seq_len - 1, 1);
        batch.market_cum.copy_(&market);
        batch
            .log_prices
            .copy_(&market.unsqueeze(-1).expand([2, length, CHANNELS], true));
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let stats = model.statistics(&batch);
        assert!(stats.market.narrow(1, 3, 1).abs().max().double_value(&[]) == 0.0);
        for origin in 0..4 {
            let expected = market.double_value(&[0, 16 * (origin + 1) - 1]);
            assert_eq!(stats.market.double_value(&[0, origin]), expected);
        }
        let (targets, _) = model.targets(&batch, &stats, false);
        assert!(targets.abs().max().double_value(&[]) < 1e-3, "{}", targets.abs().max());
        let drift = model.market_drift(&batch, &stats, false);
        assert_eq!(drift.size(), [2, 4, 1, 8]);
        assert!(drift.abs().max().double_value(&[]) > 0.0);
        let raw = (model.future_windows(&batch.log_prices, false) - per_bar(&stats.log_close))
            / per_bar(&stats.sigma);
        assert!((raw - &drift - &targets).abs().max().double_value(&[]) < 1e-5);
        let (last, _) = model.targets(&batch, &stats.last(), true);
        assert!(last.abs().max().double_value(&[]) < 1e-3);
    }

    #[test]
    fn a_half_beta_ticker_converges_to_a_zero_close_target_with_history() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(7);
        let config = ModelConfig {
            seq_len: 8192,
            ..small_config()
        };
        let origins = config.origins();
        let mut batch = synthetic(&config, &[8]);
        let length = config.seq_len + config.pred_len;
        let market = (Tensor::randn([1, length], (Kind::Float, Device::Cpu)) * 0.003 + 0.001)
            .cumsum(1, Kind::Float);
        let market = &market - market.narrow(1, config.seq_len - 1, 1);
        batch.market_cum.copy_(&market);
        batch
            .log_prices
            .copy_(&(&market * 0.5).unsqueeze(-1).expand([1, length, CHANNELS], true));
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let stats = model.statistics(&batch);
        // Exact proportionality makes the ridge slope closed-form: (0.5 n + 256) / (n + 256).
        for origin in [0, 15, origins - 1] {
            let pairs = (16 * (origin + 1) - 1) as f64;
            let expected = (0.5 * pairs + BETA_PRIOR_BARS) / (pairs + BETA_PRIOR_BARS);
            let beta = stats.beta.double_value(&[0, origin]);
            assert!((beta - expected).abs() < 1e-4, "origin {origin}: {beta} vs {expected}");
        }
        assert!(stats.beta.double_value(&[0, 0]) > 0.97);
        let raw = (model.future_windows(&batch.log_prices, false) - per_bar(&stats.log_close))
            / per_bar(&stats.sigma);
        let (targets, _) = model.targets(&batch, &stats, false);
        let residual = |origin: i64| {
            targets.select(1, origin).abs().max().double_value(&[])
                / raw.select(1, origin).abs().max().double_value(&[])
        };
        // Early origins still carry most of the market; the final origin keeps ~6% of it.
        assert!(residual(0) > 0.5, "{}", residual(0));
        assert!(residual(origins - 1) < 0.07, "{}", residual(origins - 1));
        let (last, _) = model.targets(&batch, &stats.last(), true);
        assert!(last.equal(&targets.narrow(1, origins - 1, 1)));
    }

    #[test]
    fn head_covariates_ignore_history_channels_beyond_the_context() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(11);
        let config = small_config();
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        tch::no_grad(|| {
            let _ = model.head_output.ws.shallow_clone().uniform_(-0.1, 0.1);
        });
        let batch = synthetic(&config, &[8, 8]);
        let stats = model.statistics(&batch);
        let reference = model.output(&model.forward(&batch, &stats, false, false));
        let history_channels: Vec<i64> = config
            .features
            .channel_mask(Feature::known_future)
            .iter()
            .enumerate()
            .filter_map(|(index, known)| (!known).then_some(index as i64))
            .collect();
        assert_eq!(history_channels, [6, 7, 8, 9, 10, 11]);
        let _ = batch
            .aux
            .narrow(1, config.seq_len, config.pred_len)
            .narrow(2, 6, 6)
            .fill_(7.0);
        let unchanged = model.output(&model.forward(&batch, &stats, false, false));
        assert!(unchanged.coordinates.equal(&reference.coordinates));
        assert!(unchanged.log_scale.equal(&reference.log_scale));
        let _ = batch
            .aux
            .narrow(1, config.seq_len, config.pred_len)
            .narrow(2, 0, 1)
            .fill_(7.0);
        let changed = model.output(&model.forward(&batch, &stats, false, false));
        assert!(!changed.coordinates.equal(&reference.coordinates));
    }

    /// The whole contract of an applied mean calibration, on a live head: it rescales the close
    /// anchor exactly, rescales every intrabar offset by the second curve, keeps every candle
    /// valid, cannot change a within-timestamp rank, and does not touch one byte of the
    /// checkpoint.
    #[test]
    fn an_applied_mean_gain_rescales_anchor_and_offsets_without_touching_geometry_or_weights() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(11);
        let config = small_config();
        let store = nn::VarStore::new(Device::Cpu);
        let mut model = CausalPatchModel::new(&store.root(), &config);
        live_head(&model);
        let batch = synthetic(&config, &[8, 8, 8, 8, 8, 8]);
        let stats = model.statistics(&batch);
        let last = stats.last();
        let anchored = |tensor: &Tensor| tensor.reshape([-1, 1, 1, 1]);
        let decode = |model: &CausalPatchModel| {
            let output = model.output(&model.forward(&batch, &stats, false, true));
            model.decode(&output, &last)
        };
        let baseline = decode(&model);
        let before = decode_prices(&baseline, &anchored(&batch.anchor), &anchored(&last.sigma));
        assert_valid_candles(&before);
        // The measured two-sided shape: the amplifications the short end asks for (3.662 at
        // h=1 on the step-2000 checkpoint), then the middle band, then the hard long-end shrink
        // and one horizon at the smallest gain a fit would ever emit. A gain above 1 is exactly
        // as safe as one below it here. The offset curve is deliberately DIFFERENT from the
        // anchor's at every horizon, including one horizon where the two straddle 1, because a
        // shared curve would let a bug that ignores one of them pass.
        let frozen = FrozenGain {
            estimator: "test".to_owned(),
            blocks: calibration_blocks(),
            anchor: vec![3.662, 2.7309, 2.2125, 1.0283, 0.85, 0.41, 0.32, 0.01],
            offset: vec![0.5, 1.4, 0.9, 2.0, 1.0, 0.25, 3.0, 1.7],
            // The signed measurement the curves were smoothed from: it rides with the frozen
            // gain but nothing in the model reads it, which is what this fixture pins.
            measured_anchor: vec![
                Some(3.6),
                Some(2.7),
                Some(2.2),
                Some(1.0),
                Some(0.9),
                Some(0.4),
                Some(-0.3),
                None,
            ],
        };
        assert_eq!(frozen.anchor.len(), config.pred_len as usize);
        let weights = |store: &nn::VarStore| {
            let mut named: Vec<(String, Vec<f64>)> = store
                .variables()
                .into_iter()
                .map(|(name, tensor)| {
                    (
                        name,
                        Vec::<f64>::try_from(tensor.to_kind(Kind::Double).flatten(0, -1)).unwrap(),
                    )
                })
                .collect();
            named.sort_by(|a, b| a.0.cmp(&b.0));
            named
        };
        let saved = weights(&store);
        model.set_mean_gain(&frozen).unwrap();
        assert_eq!(model.mean_gain().unwrap(), &frozen);
        // The checkpoint is bit-identical: the two curves are derived buffers, not variables,
        // so nothing a `VarStore::save` would write has moved.
        assert!(saved == weights(&store), "the gain changed a saved tensor");
        let calibrated = decode(&model);
        let after = decode_prices(&calibrated, &anchored(&batch.anchor), &anchored(&last.sigma));
        assert_valid_candles(&after);
        for bar in 0..config.pred_len {
            let (anchor, offset) = (
                frozen.anchor[bar as usize],
                frozen.offset[bar as usize],
            );
            let close = baseline.narrow(-2, CHANNELS - 1, 1).narrow(-1, bar, 1);
            let scaled_close = calibrated.narrow(-2, CHANNELS - 1, 1).narrow(-1, bar, 1);
            let tolerance = 1e-5 * (1. + close.abs().max().double_value(&[]));
            assert!(
                (&scaled_close - &close * anchor)
                    .abs()
                    .max()
                    .double_value(&[])
                    <= tolerance,
                "the close channel at h={} is not the anchor gain times the uncalibrated close",
                bar + 1
            );
            // Each channel is `anchor·close + offset·(channel - close)`, which is the whole
            // transform: the offsets keep their sign, so the geometry above survives.
            let expected = &close * anchor
                + (baseline.narrow(-1, bar, 1) - &close) * offset;
            assert!(
                (calibrated.narrow(-1, bar, 1) - expected)
                    .abs()
                    .max()
                    .double_value(&[])
                    <= tolerance,
                "a channel at h={} did not move by the two-coordinate rescale",
                bar + 1
            );
            // The mechanism behind IC invariance: a positive gain cannot reorder the rows of
            // one horizon, so no within-timestamp rank statistic can move.
            let order = |tensor: &Tensor| {
                tensor
                    .narrow(-2, CHANNELS - 1, 1)
                    .narrow(-1, bar, 1)
                    .reshape([-1])
                    .argsort(0, false)
            };
            assert!(
                order(&baseline).equal(&order(&calibrated)),
                "the row order at h={} changed under a positive gain",
                bar + 1
            );
        }
        // Setting REPLACES rather than composes, so a refit cannot square its own shrinkage.
        model.set_mean_gain(&frozen).unwrap();
        let again = decode(&model);
        assert!((again - &calibrated).abs().max().double_value(&[]) == 0.);
        // And never a wrong-length curve or a sign flip.
        let mut fresh = CausalPatchModel::new(&nn::VarStore::new(Device::Cpu).root(), &config);
        let mut short = frozen.clone();
        short.anchor.truncate(4);
        assert!(fresh.set_mean_gain(&short).is_err());
        let mut flipped = frozen.clone();
        flipped.offset[2] = -1.;
        assert!(fresh.set_mean_gain(&flipped).is_err());
        let mut nonfinite = frozen.clone();
        nonfinite.anchor[1] = f64::NAN;
        assert!(fresh.set_mean_gain(&nonfinite).is_err());
        assert!(fresh.mean_gain().is_none());
    }

    /// A dated pair of blocks for the fixtures above; the numbers are provenance only - nothing
    /// in the model reads them.
    fn calibration_blocks() -> super::super::calibration::Blocks {
        super::super::calibration::Blocks {
            calibration_first_origin_ms: 1,
            calibration_last_origin_ms: 2,
            calibration_last_target_ms: 3,
            calibration_origins: 4,
            evaluation_first_origin_ms: 5,
            evaluation_last_origin_ms: 6,
            evaluation_origins: 7,
            purge_gap_ms: 2,
        }
    }

    /// The prior's VALUE and its GRADIENT against the pre-registered algebra, on a masked
    /// population, plus the fact that `λ = 0` emits nothing at all.
    ///
    /// Both halves matter and neither implies the other: a penalty with the right value and
    /// the wrong normalization would still train, just at a λ that means something else at
    /// every batch size, and a penalty whose gradient is not `(λ/(H·N_h))·w_h·mask·(m - m̄_h)`
    /// is a different intervention from the one that was pre-registered.
    #[test]
    fn the_amplitude_prior_is_the_pre_registered_mean_energy_and_its_gradient() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(23);
        let mut config = small_config();
        // Not `uniform`: a weighting the prior silently ignored would pass under `w ≡ 1`.
        config.horizon_loss = HorizonLoss::InverseSqrt;
        config.amplitude_prior = 0.7;
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let (rows, horizons) = (6i64, config.pred_len);
        let cpu = (Kind::Float, Device::Cpu);
        let close = Tensor::randn([rows, 1, 1, horizons], cpu).set_requires_grad(true);
        // A ragged mask, including one horizon with a single valid bar - whose within-horizon
        // dispersion is exactly zero and must contribute exactly zero rather than a NaN.
        let mut flags = vec![1f32; (rows * horizons) as usize];
        for row in 0..rows {
            for bar in 0..horizons {
                if bar == horizons - 1 && row > 0 {
                    flags[(row * horizons + bar) as usize] = 0.;
                }
            }
        }
        flags[horizons as usize] = 0.;
        let mask = Tensor::from_slice(&flags).reshape([rows, 1, 1, horizons]);
        let penalty = model
            .amplitude_prior(&close, &mask)
            .expect("a nonzero λ must emit a penalty");
        let host = |tensor: &Tensor| {
            Vec::<f64>::try_from(tensor.to_kind(Kind::Double).flatten(0, -1)).unwrap()
        };
        let (values, flags, weights) = (host(&close), host(&mask), model.horizon_weights());
        let width = horizons as usize;
        let mut expected = 0.;
        let mut expected_gradient = vec![0.; values.len()];
        for bar in 0..width {
            let count: f64 = (0..rows as usize).map(|row| flags[row * width + bar]).sum();
            let scale = count.max(1.);
            let mean: f64 = (0..rows as usize)
                .map(|row| flags[row * width + bar] * values[row * width + bar])
                .sum::<f64>()
                / scale;
            for row in 0..rows as usize {
                let index = row * width + bar;
                let deviation = values[index] - mean;
                expected += config.amplitude_prior * weights[bar] * flags[index] * deviation
                    * deviation
                    / (2. * width as f64 * scale);
                expected_gradient[index] = config.amplitude_prior * weights[bar] * flags[index]
                    * deviation
                    / (width as f64 * scale);
            }
        }
        let measured = penalty.double_value(&[]);
        // fp32 against an f64 reference. `Σ mask·m² - (Σ mask·m)²/N` is a moment form, so it
        // would cancel if the per-horizon mean dominated the dispersion; on the measured
        // population it does not come close - at h=192 the mean forecast is `-0.105` against a
        // `Var(f)` of about `9`, so the subtracted term is under 0.2% of the first and fp32
        // relative error stays at the 1e-7 level this asserts.
        assert!(
            (measured - expected).abs() <= 1e-6 * (1. + expected.abs()),
            "the prior evaluated to {measured} against the pre-registered {expected}"
        );
        penalty.backward();
        for (index, gradient) in host(&close.grad()).into_iter().enumerate() {
            assert!(
                (gradient - expected_gradient[index]).abs()
                    <= 1e-6 * (1. + expected_gradient[index].abs()),
                "∂R/∂m at flat index {index} is {gradient} against the pre-registered {}",
                expected_gradient[index]
            );
        }
        // The control emits no penalty at all, which is what makes an unpenalized arm's loss
        // bit-identical to the pre-knob one rather than merely numerically close.
        let mut control = small_config();
        control.amplitude_prior = 0.;
        let unpenalized = CausalPatchModel::new(&nn::VarStore::new(Device::Cpu).root(), &control);
        assert!(unpenalized.amplitude_prior(&close, &mask).is_none());
    }

    /// A control arm's `ModelConfig` MUST serialize to the same BYTES it did before the knob
    /// existed, because [`super::runner`]'s manifest digest is a SHA-256 over
    /// `serde_json::to_vec` of the whole manifest: one extra key, or one reordered key, and
    /// every checkpoint written before today fails its own authentication and stops loading.
    /// The literal below is the real `model` object out of
    /// `training/runs/timexer-control-4k/weights/best/manifest.json`, the checkpoint the
    /// amplitude calibration is paired to.
    #[test]
    fn a_control_config_serializes_exactly_as_it_did_before_the_amplitude_prior_existed() {
        const PRE_KNOB: &str = r#"{"seq_len":6000,"pred_len":192,"patch_len":16,"layers":8,"d_model":512,"heads":8,"ffn":2048,"dropout":0.0,"min_history":256,"features":{"time_of_day":true,"day_of_week":true,"session_gap":true,"volume":true,"market":true,"spy":true},"x0_lambdas":"disabled","horizon_loss":"uniform","horizon_mean":"free"}"#;
        let control: ModelConfig = serde_json::from_str(PRE_KNOB).unwrap();
        assert_eq!(control.amplitude_prior, 0.);
        assert_eq!(serde_json::to_string(&control).unwrap(), PRE_KNOB);
        // And a penalized arm is a DIFFERENT config that says so in its own manifest, rather
        // than an arm that looks like the control with a hidden objective term.
        let penalized = ModelConfig {
            amplitude_prior: 0.0035,
            ..control
        };
        let written = serde_json::to_string(&penalized).unwrap();
        assert_eq!(
            written,
            PRE_KNOB.replace(
                r#""horizon_mean":"free"}"#,
                r#""horizon_mean":"free","amplitude_prior":0.0035}"#
            )
        );
        assert_eq!(
            serde_json::from_str::<ModelConfig>(&written)
                .unwrap()
                .amplitude_prior,
            0.0035
        );
    }
    #[test]
    fn zero_head_forecasts_persistence_with_root_horizon_scale_at_every_origin() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(3);
        let config = small_config();
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let batch = synthetic(&config, &[8, 5]);
        let stats = model.statistics(&batch);
        let head = model.forward(&batch, &stats, false, false);
        let output = model.output(&head);
        let scaled = model.decode(&output, &stats);
        assert_eq!(scaled.size(), [2, 4, 4, 8]);
        assert_eq!(scaled.narrow(2, 3, 1).abs().max().double_value(&[]), 0.0);
        assert_eq!(scaled.narrow(2, 0, 1).abs().max().double_value(&[]), 0.0);
        let expected_scale = model.half_log_horizon().expand_as(&output.log_scale);
        assert!(output.log_scale.equal(&expected_scale));
        let anchor = batch.anchor.reshape([2, 1, 1, 1]) * stats.log_close.unsqueeze(-1).unsqueeze(-1).exp();
        let prices = decode_prices(&scaled, &anchor, &per_bar(&stats.sigma));
        assert!((prices.narrow(2, 3, 1) - &anchor).abs().max().double_value(&[]) < 1e-3);
        let (targets, mask) = model.targets(&batch, &stats, false);
        let losses = gaussian_nll(
            &scaled,
            &output.log_scale,
            &targets,
            &mask,
            model.horizon_weight_buffer(),
        );
        let nll = losses.nll.double_value(&[]);
        let mut by_hand = 0.0;
        let mut squared = 0.0;
        let mut count = 0.0;
        for row in 0..2 {
            for origin in 0..4 {
                let sigma = stats.sigma.double_value(&[row, origin]);
                let range = stats.range.double_value(&[row, origin]);
                let half = (0.5 * range).ln_1p();
                let candle = [0.0, (range.ln_1p() - half) / sigma, -half / sigma, 0.0];
                for step in 0..8 {
                    let weight = mask.double_value(&[row, origin, 0, step]);
                    for (channel, reference) in candle.iter().enumerate() {
                        let y = targets.double_value(&[row, origin, channel as i64, step]);
                        let h = (step + 1) as f64;
                        let residual = y - reference;
                        by_hand += weight * (0.5 * residual * residual / h + 0.5 * h.ln());
                        squared += weight * residual * residual;
                        count += weight;
                        assert!(
                            (scaled.double_value(&[row, origin, channel as i64, step]) - reference).abs()
                                < 1e-5,
                            "persistence candle mismatch at channel {channel}"
                        );
                    }
                }
            }
        }
        assert!((nll - by_hand / count).abs() < 1e-5, "{nll} vs {}", by_hand / count);
        assert!((losses.mse.double_value(&[]) - squared / count).abs() < 1e-5);
        assert!(mask.sum(Kind::Float).double_value(&[]) < 2.0 * 4.0 * 8.0);
        // The fused training loss must reproduce the reference chain it replaced.
        let fused = model.losses(&head, &stats, &targets, &mask);
        assert!(
            (fused.nll.double_value(&[]) - nll).abs() <= 1e-6 * nll.abs().max(1e-6),
            "fused {} vs reference {nll}",
            fused.nll.double_value(&[])
        );
        assert!(
            (fused.mse.double_value(&[]) - losses.mse.double_value(&[])).abs() < 1e-6
        );
    }

    /// The fused training loss against the decode+NLL chain it replaced, with a NONZERO head so
    /// every branch of the candle geometry and of the log-scale cap carries signal: the loss, the
    /// no-gradient MSE, and every parameter gradient.
    ///
    /// Run for EVERY [`HorizonLoss`] mode. The reference chain applies the weight to the mask
    /// and normalizes by `Σ w·mask·CHANNELS`, so this is also the definitional pin on what the
    /// weighted objective - and therefore the selection scalar computed the same way on the
    /// held-out split - means.
    fn fused_matches_reference(horizon_loss: HorizonLoss) {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(17);
        let config = ModelConfig {
            horizon_loss,
            ..small_config()
        };
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        // The residual-branch output projections are zero at init (that is what makes the
        // stack the identity, see `the_stack_is_the_identity_on_the_residual_stream_at_init`),
        // so at init the QKV and FFN-up matrices and the post-lambdas legitimately receive
        // no gradient. This test is about the fused loss reaching every parameter, so it
        // needs a live network.
        live_head(&model);
        let batch = synthetic(&config, &[8, 5]);
        let stats = model.statistics(&batch);
        let (targets, mask) = model.targets(&batch, &stats, false);
        let head = model.forward(&batch, &stats, false, false);
        let output = model.output(&head);
        let reference = gaussian_nll(
            &model.decode(&output, &stats),
            &output.log_scale,
            &targets,
            &mask,
            model.horizon_weight_buffer(),
        );
        let fused = model.losses(&head, &stats, &targets, &mask);
        let expected = reference.nll.double_value(&[]);
        assert!(expected.is_finite() && expected.abs() > 1e-3, "{expected}");
        let relative = (fused.nll.double_value(&[]) - expected).abs() / expected.abs();
        assert!(relative <= 1e-4, "NLL relative error {relative}");
        let expected_mse = reference.mse.double_value(&[]);
        assert!(
            (fused.mse.double_value(&[]) - expected_mse).abs() <= 1e-4 * expected_mse.abs(),
            "MSE {} vs {expected_mse}",
            fused.mse.double_value(&[])
        );
        let parameters = store.trainable_variables();
        let reference_grads = Tensor::run_backward(&[&reference.nll], &parameters, true, false);
        let fused_grads = Tensor::run_backward(&[&fused.nll], &parameters, false, false);
        let mut touched = 0;
        let mut worst = 0.;
        for (index, (left, right)) in reference_grads.iter().zip(&fused_grads).enumerate() {
            let scale = left.abs().max().double_value(&[]);
            assert!(scale.is_finite(), "nonfinite reference gradient {index}");
            if scale > 0.0 {
                touched += 1;
            }
            let error = (left - right).abs().max().double_value(&[]) / scale.max(1e-8);
            worst = f64::max(worst, error);
            assert!(error <= 1e-4, "parameter {index} gradient relative error {error}");
        }
        assert_eq!(touched, parameters.len(), "a parameter received no gradient");
        println!(
            "fused vs reference at {horizon_loss}: NLL {relative:.3e} relative, MSE {:.3e} \
             relative, worst parameter gradient {worst:.3e} relative over {touched} parameters",
            (fused.mse.double_value(&[]) - expected_mse).abs() / expected_mse.abs()
        );
    }

    /// The paired benchmark arm's composed chain is THE objective, not a paraphrase of it.
    ///
    /// This is the test that keeps `benchmark.rs`'s paired measurement meaningful. The two
    /// functions differ in exactly one line - which geometry provider they call - and share
    /// their operand preparation and all twelve reductions, so a paired difference between
    /// them is a measurement of the fusion. If someone later changes `losses`'s σ clamp, its
    /// horizon-weight fold or its denominators without changing `composed_losses`, the
    /// difference would silently become a measurement of two different objectives, and this
    /// fails the moment that happens: off CUDA both providers ARE the reference chain, so
    /// any inequality here is prep or tail drift and nothing else.
    #[test]
    fn the_composed_benchmark_arm_is_the_same_objective_as_the_fused_one() {
        // A concurrent `manual_seed` rewinds the global stream this test reads - see
        // `torch::test_rng` - and a rewind here would compare two DIFFERENT batches.
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(19);
        let config = small_config();
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        live_head(&model);
        let batch = synthetic(&config, &[8, 5]);
        let stats = model.statistics(&batch);
        let (targets, mask) = model.targets(&batch, &stats, false);
        let head = model.forward(&batch, &stats, false, false);
        let fused = model.losses(&head, &stats, &targets, &mask);
        let composed = model.composed_losses(&head, &stats, &targets, &mask);
        let value = fused.nll.double_value(&[]);
        assert!(value.is_finite() && value.abs() > 1e-3, "{value}");
        assert_eq!(
            value.to_bits(),
            composed.nll.double_value(&[]).to_bits(),
            "the composed arm's NLL is {} against the fused {value}",
            composed.nll.double_value(&[])
        );
        assert_eq!(
            fused.mse.double_value(&[]).to_bits(),
            composed.mse.double_value(&[]).to_bits(),
            "the composed arm's diagnostic MSE is {} against the fused {}",
            composed.mse.double_value(&[]),
            fused.mse.double_value(&[])
        );
        // And the gradient, because the benchmark times a backward: the paired arms must
        // drive the same one.
        let parameters = store.trainable_variables();
        let left = Tensor::run_backward(&[&fused.nll], &parameters, true, false);
        let right = Tensor::run_backward(&[&composed.nll], &parameters, false, false);
        for (index, (fused_grad, composed_grad)) in left.iter().zip(&right).enumerate() {
            assert!(
                fused_grad.equal(composed_grad),
                "parameter {index} receives a different gradient from the two arms"
            );
        }
    }

    #[test]
    fn fused_loss_matches_the_reference_decode_and_nll_including_gradients() {
        let _rng = crate::torch::test_rng::exclusive();
        for horizon_loss in [
            HorizonLoss::Uniform,
            HorizonLoss::InverseSqrt,
            HorizonLoss::Inverse,
            HorizonLoss::Cutoff(3),
            HorizonLoss::Cutoff(small_config().pred_len),
        ] {
            fused_matches_reference(horizon_loss);
        }
    }

    /// Every profiled kernel class runs at the input shape it declares, and the classes that
    /// stand in for a piece of the real forward produce that piece's exact shape.
    ///
    /// The profile chart's rows are worth reading only if each one is the op it names. A class
    /// that declared the wrong width - `d_model` instead of `d_model + COVARIATE_WIDTH` for the
    /// head's hidden projection, say - would still time a GEMM, just not the one the step runs,
    /// and the chart would report a fiction with nothing anywhere raising an error.
    #[test]
    fn every_kernel_class_runs_at_the_shape_it_declares() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(7);
        let config = small_config();
        let batch = synthetic(&config, &[6, 8]);
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let stats = model.statistics(&batch);
        let head = model.forward(&batch, &stats, false, false);
        let targets = model.targets(&batch, &stats, false).0;
        for class in model.kernel_classes(&batch, &stats, false) {
            let inputs: Vec<Tensor> = class
                .inputs
                .iter()
                .map(|(shape, kind)| {
                    Tensor::randn(shape.as_slice(), (Kind::Float, Device::Cpu)).to_kind(*kind)
                })
                .collect();
            let output = (class.run)(&inputs);
            assert!(
                output.isfinite().all().int64_value(&[]) != 0,
                "class {} produced a nonfinite output",
                class.name
            );
            match class.name {
                "head output projection" => assert_eq!(output.size(), head.0.size()),
                "targets, market drift and validity mask" => {
                    assert_eq!(output.size(), targets.size());
                }
                // One masked mean over the whole space: a class that reduced per channel or
                // per row would time a different amount of reduction traffic.
                "fused loss geometry and NLL" => assert_eq!(output.size(), Vec::<i64>::new()),
                _ => {}
            }
        }
    }

    /// A live head, so the loss value and the gradients are nontrivial.
    fn live_head(model: &CausalPatchModel) {
        tch::no_grad(|| {
            let _ = model.head_output.ws.shallow_clone().uniform_(-0.5, 0.5);
            let _ = model
                .head_output
                .bs
                .as_ref()
                .unwrap()
                .shallow_clone()
                .uniform_(-0.5, 0.5);
            for block in &model.blocks {
                let _ = block.output.ws.shallow_clone().uniform_(-0.1, 0.1);
                let _ = block.second.ws.shallow_clone().uniform_(-0.1, 0.1);
            }
        });
    }

    /// Every mode's normalization: mean 1 over ALL `pred_len` horizons, so `Σ w = pred_len`.
    /// That is what keeps the reported NLL a weighted MEAN in nats per bar rather than a sum
    /// whose scale moves with the mode, and it is why `uniform` is the vector `1` exactly.
    #[test]
    fn every_horizon_loss_mode_normalizes_to_mean_one_over_the_whole_axis() {
        for pred_len in [8i64, 192] {
            for mode in [
                HorizonLoss::Uniform,
                HorizonLoss::InverseSqrt,
                HorizonLoss::Inverse,
                HorizonLoss::Cutoff(1),
                HorizonLoss::Cutoff(pred_len / 6),
                HorizonLoss::Cutoff(pred_len),
            ] {
                let weights = mode.weights(pred_len);
                assert_eq!(weights.len(), pred_len as usize, "{mode}");
                let sum: f64 = weights.iter().sum();
                assert!(
                    (sum - pred_len as f64).abs() <= 1e-9 * pred_len as f64,
                    "{mode} at pred_len {pred_len} sums to {sum}, not {pred_len}"
                );
                assert!(weights.iter().all(|w| w.is_finite() && *w >= 0.), "{mode}");
                // The decays are strictly monotone; the cutoff is a step, never negative.
                match mode {
                    HorizonLoss::Uniform => {
                        assert_eq!(weights, vec![1.0; pred_len as usize], "uniform is not 1")
                    }
                    HorizonLoss::InverseSqrt | HorizonLoss::Inverse => assert!(
                        weights.windows(2).all(|pair| pair[0] > pair[1]),
                        "{mode} is not strictly decreasing"
                    ),
                    HorizonLoss::Cutoff(cut) => {
                        let trained = weights.iter().filter(|w| **w > 0.).count() as i64;
                        assert_eq!(trained, cut, "cutoff:{cut} trains {trained} horizons");
                        assert!(
                            weights[..cut as usize]
                                .iter()
                                .all(|w| (*w - pred_len as f64 / cut as f64).abs() < 1e-9),
                            "cutoff:{cut} does not weight its trained horizons uniformly"
                        );
                    }
                }
            }
        }
        // The two decays BRACKET the rate rather than being two spellings of one. Both carry
        // mean 1, so they must cross exactly once: `inv` concentrates strictly more of the
        // objective on the short horizons that jobs 5190-5193 showed still improving, and
        // strictly less on the long ones that rot. The ratio `w_inv / w_inv-sqrt` is `√h`
        // scaled, hence strictly decreasing, which is what "one crossing" means here.
        let (fast, slow) = (
            HorizonLoss::Inverse.weights(192),
            HorizonLoss::InverseSqrt.weights(192),
        );
        let ratio: Vec<f64> = fast.iter().zip(&slow).map(|(f, s)| f / s).collect();
        assert!(
            ratio.windows(2).all(|pair| pair[0] > pair[1]),
            "the two decays do not order monotonically along the horizon axis"
        );
        assert!(
            ratio[0] > 1.0 && *ratio.last().unwrap() < 1.0,
            "the two decays must cross: ratio {} at h = 1, {} at h = 192",
            ratio[0],
            ratio.last().unwrap()
        );
        let head = |weights: &[f64]| weights[..8].iter().sum::<f64>();
        let tail = |weights: &[f64]| weights[64..].iter().sum::<f64>();
        assert!(
            head(&fast) > head(&slow) && tail(&fast) < tail(&slow),
            "1/h must move objective weight from the long end to the short end"
        );
    }

    /// `uniform` is the control arm, so it has to reproduce the objective it replaced EXACTLY,
    /// not merely closely: the jobs 5190-5193 curves are only comparable to a `uniform` arm if
    /// the number means the same thing. The proof is structural, which is stronger than a
    /// tolerance: every tensor the pre-knob loss read was `mask`, the weighted loss reads
    /// `mask · w` and divides by `Σ w·mask·CHANNELS` instead of `Σ mask·CHANNELS`, and at
    /// `w ≡ 1` both are the identity ON THE BITS - so every downstream kernel receives byte
    /// for byte what it received before and emits byte for byte what it emitted before.
    #[test]
    fn uniform_horizon_loss_reproduces_the_previous_unweighted_objective() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(23);
        let config = small_config();
        assert_eq!(
            config.horizon_loss,
            HorizonLoss::Uniform,
            "the default must stay the control"
        );
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        live_head(&model);
        let batch = synthetic(&config, &[8, 5]);
        let stats = model.statistics(&batch);
        let (targets, mask) = model.targets(&batch, &stats, false);
        // The fold and the denominator, both bit-exact identities at w = 1.
        assert!(
            (&mask * model.horizon_weight_buffer()).equal(&mask),
            "the uniform fold is not the identity on the mask"
        );
        let count = (mask.sum(Kind::Float) * CHANNELS).clamp_min(1.0);
        let weighted_count =
            ((&mask * model.horizon_weight_buffer()).sum(Kind::Float) * CHANNELS).clamp_min(1.0);
        assert!(
            weighted_count.equal(&count),
            "the uniform denominator is not the unweighted one"
        );
        // And the value against the pre-knob expression written out verbatim.
        let head = model.forward(&batch, &stats, false, false);
        let output = model.output(&head);
        let previous = ((nll_elements(
            &model.decode(&output, &stats),
            &output.log_scale,
            &targets,
        ) * &mask)
            .sum(Kind::Float)
            / count)
            .double_value(&[]);
        let fused = model.losses(&head, &stats, &targets, &mask).nll.double_value(&[]);
        assert!(previous.is_finite() && previous.abs() > 1e-3, "{previous}");
        let relative = (fused - previous).abs() / previous.abs();
        // The fused reassociation's own documented tolerance against the reference chain, not
        // the weighting's: the weighting contributes exactly zero to this difference.
        assert!(relative <= 1e-4, "uniform NLL {fused} vs {previous}");
    }

    /// `--target-basis cumulative` with uniform coefficient weights is a BIT-EXACT identity on
    /// the objective. This is the gate on the whole knob: if the control arm's loss moves by an
    /// ulp, no step-matched comparison against the `inv-sqrt` control means anything.
    ///
    /// The identity is established at two levels, and the distinction is not pedantry.
    ///
    /// **The control arm does not execute this code.** `basis` is `None` under `cumulative`, so
    /// [`Self::losses`] falls through to the same fused call it made before the knob existed
    /// and `uniform_horizon_loss_reproduces_the_previous_unweighted_objective` still pins its
    /// value. That is a structural identity, not a numerical one, and it is the one that
    /// matters for the control.
    ///
    /// **The coefficient path itself reproduces [`gaussian_nll`] on the bits, term for term.**
    /// Every input to the reduction is bit-equal: the decoded mean, the rotation (a matmul by
    /// the identity sums one value against exact zeros), the coefficient prior against
    /// `½·ln h`, the per-element NLL against [`nll_elements`], the mask fold against
    /// `mask·w`, and the denominator. The reduced SCALAR differs by exactly one fp32 ulp, and
    /// the reason is measured here rather than waved at: `gaussian_nll`'s numerator tensor is
    /// NON-contiguous (its log-scale term is a strided view of the head), so ATen's cascade
    /// sum visits it in a different order than it visits this path's contiguous numerator.
    /// Forcing the reference contiguous makes the scalars bit-equal too, which localizes the
    /// ulp to the reference composition's strides and not to the reparametrization.
    #[test]
    fn cumulative_target_basis_is_a_bit_exact_identity_on_the_objective() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(31);
        let config = small_config();
        assert_eq!(
            config.target_basis,
            TargetBasis::Cumulative,
            "the default must stay the control"
        );
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        assert!(
            model.basis().is_none(),
            "the control arm must not construct a transform at all"
        );
        live_head(&model);
        // Complete windows: the identity claim is about the metric, and the row-completeness
        // mask is a separate contract with its own test below.
        let batch = synthetic(&config, &[config.pred_len, config.pred_len]);
        let stats = model.statistics(&batch);
        let (targets, mask) = model.targets(&batch, &stats, false);
        let head = model.forward(&batch, &stats, false, false);
        let output = model.output(&head);
        let identity = BasisTransform::new(
            TargetBasis::Cumulative,
            BasisWeight::Uniform,
            config.pred_len,
            None,
            Device::Cpu,
        )
        .unwrap();
        assert!(
            identity.half_log_prior().equal(model.half_log_horizon()),
            "the identity coefficient prior is not bit-for-bit ½·ln h"
        );
        // Every input to the reduction, bit for bit. The implementation masks the residual
        // before rotating, and on a complete window the mask is exactly 1, so that multiply is
        // the identity on the bits and this mirrors it.
        let prediction = model.decode(&output, &stats);
        let error = &targets - &prediction;
        let masked = &error * &mask;
        assert!(
            identity.rotate(&masked).equal(&masked),
            "the identity rotation is not the identity on real residuals"
        );
        let reference_terms = nll_elements(&prediction, &output.log_scale, &targets);
        let terms = (identity.rotate(&masked) * (-&output.log_scale).exp()).square() * 0.5
            + &output.log_scale;
        let weighted = mask.amin([-1i64].as_slice(), true) * identity.weight();
        let weighted_mask = &mask * model.horizon_weight_buffer();
        assert!(
            weighted.equal(&weighted_mask),
            "the coefficient weight fold is not bit-for-bit the horizon mask fold"
        );
        assert!(
            (&terms * &weighted).equal(&(&reference_terms * &weighted_mask)),
            "the weighted numerator is not bit-for-bit the reference's"
        );
        // And the scalar, once the reference's strides are taken out of the comparison.
        let count = (weighted_mask.sum(Kind::Float) * CHANNELS).clamp_min(1.0);
        let contiguous_reference =
            (&reference_terms * &weighted_mask).contiguous().sum(Kind::Float) / &count;
        let rotated = model.basis_losses(&identity, &head, &stats, &targets, &mask);
        assert!(
            rotated.nll.equal(&contiguous_reference),
            "the identity basis NLL is {} not {}",
            rotated.nll.double_value(&[]),
            contiguous_reference.double_value(&[])
        );
        // The MSE is a diagnostic every arm must be comparable on, so it is bit-identical to
        // the reference outright - no stride caveat, because both numerators are contiguous.
        let reference = gaussian_nll(
            &prediction,
            &output.log_scale,
            &targets,
            &mask,
            model.horizon_weight_buffer(),
        );
        assert!(
            rotated.mse.equal(&reference.mse),
            "the identity basis MSE is {} not {}",
            rotated.mse.double_value(&[]),
            reference.mse.double_value(&[])
        );
        // The remaining scalar difference against the strided reference is ONE ulp, stated as a
        // number so a regression that grows it is visible.
        let ulp = (rotated.nll.double_value(&[]) - reference.nll.double_value(&[])).abs()
            / reference.nll.double_value(&[]).abs();
        assert!(
            ulp <= f64::from(f32::EPSILON),
            "the identity basis NLL differs from the strided reference by {ulp:e} relative, \
             more than one fp32 ulp; that is no longer a reduction-order difference"
        );
    }

    /// The shipped rotation is a ROTATION in the arithmetic it ships in: `‖W·r‖ = ‖r‖` on real
    /// fp32 residuals at the production horizon length. This is the operational form of
    /// orthonormality - the Gram matrix test in [`super::target_basis`] proves the matrix, this
    /// proves the GEMM that consumes it - and it is why the reparametrization restricts nothing.
    #[test]
    fn every_basis_preserves_the_residual_norm_at_the_production_horizon() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(37);
        let residual = Tensor::randn([16, 4, 192], (Kind::Float, Device::Cpu)) * 3.0;
        let energy = residual.square().sum_dim_intlist([-1i64].as_slice(), false, Kind::Double);
        for basis in [TargetBasis::Cumulative, TargetBasis::Haar, TargetBasis::Dct] {
            let transform =
                BasisTransform::new(basis, BasisWeight::Uniform, 192, None, Device::Cpu).unwrap();
            let rotated = transform
                .rotate(&residual)
                .square()
                .sum_dim_intlist([-1i64].as_slice(), false, Kind::Double);
            let worst = ((&rotated - &energy).abs() / &energy).max().double_value(&[]);
            assert!(
                worst < 1e-5,
                "{basis} moved the residual energy by {worst:e} relative; the map is not a \
                 rotation in fp32 and therefore not a reparametrization"
            );
        }
    }

    /// A rotated objective must not restrict the head, which is the ONE way this could repeat
    /// `--horizon-mean basis:8:8`'s failure. Every head row - all `2·CHANNELS·pred_len` of them,
    /// mean rows and log-scale rows alike - receives nonzero gradient under a full-rank
    /// rotation, because `W` is invertible and every coefficient reads every horizon.
    #[test]
    fn a_rotated_objective_leaves_no_head_row_untrained() {
        let _rng = crate::torch::test_rng::exclusive();
        for basis in [TargetBasis::Haar, TargetBasis::Dct] {
            tch::manual_seed(41);
            let config = ModelConfig {
                target_basis: basis,
                ..small_config()
            };
            let store = nn::VarStore::new(Device::Cpu);
            let model = CausalPatchModel::new(&store.root(), &config);
            assert_eq!(model.basis().map(BasisTransform::basis), Some(basis));
            live_head(&model);
            let batch = synthetic(&config, &[config.pred_len, config.pred_len]);
            let stats = model.statistics(&batch);
            let (targets, mask) = model.targets(&batch, &stats, false);
            let head = model.forward(&batch, &stats, false, false);
            let losses = model.losses(&head, &stats, &targets, &mask);
            assert!(
                losses.nll.double_value(&[]).is_finite(),
                "{basis} produced a nonfinite NLL"
            );
            let gradient = Tensor::run_backward(
                &[&losses.nll],
                &[&model.head_output.ws],
                false,
                false,
            )
            .remove(0);
            let dead = gradient
                .abs()
                .reshape([gradient.size()[0], -1])
                .sum_dim_intlist([1i64].as_slice(), false, Kind::Double)
                .eq(0.0)
                .sum(Kind::Int64)
                .int64_value(&[]);
            assert_eq!(
                dead, 0,
                "{basis} left {dead} of {} head rows with exactly zero gradient",
                gradient.size()[0]
            );
        }
    }

    /// A coefficient is a weighted sum over the WHOLE horizon window, so an origin whose window
    /// is only partly observed carries no coefficient vector and must contribute NOTHING to the
    /// rotated objective.
    ///
    /// The contract is a two-sided invariance, and both sides are load-bearing:
    ///
    /// - overwriting an UNOBSERVED entry with garbage may not move the objective by one bit -
    ///   which is what "dropped, not zero-filled" means, and what a `0 · large` product after a
    ///   192-term rotation would not survive;
    /// - overwriting an OBSERVED entry MUST move it, so the test cannot pass by ignoring
    ///   everything.
    ///
    /// The masked population is identified from the mask itself rather than assumed. Origins
    /// sit at patch boundaries INSIDE the context, so with `pred_len` short futures only the
    /// last origins of a short row read unobserved bars at all; a test that poisoned a fixed
    /// horizon band across every origin would be poisoning observed data and asserting the
    /// wrong thing.
    #[test]
    fn the_rotated_objective_ignores_every_incomplete_horizon_window() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(43);
        let config = ModelConfig {
            target_basis: TargetBasis::Dct,
            ..small_config()
        };
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        live_head(&model);
        // Row 0 complete, row 1 observed for only three of `pred_len` future bars.
        let batch = synthetic(&config, &[config.pred_len, 3]);
        let stats = model.statistics(&batch);
        let (targets, mask) = model.targets(&batch, &stats, false);
        // The masked population has to exist, or the test proves nothing.
        let unobserved = mask
            .eq(0.0)
            .logical_and(&mask.amax([-1i64].as_slice(), true).gt(0.0));
        assert!(
            unobserved.sum(Kind::Int64).int64_value(&[]) > 0,
            "this batch has no unobserved future bar on a scored origin, so the invariance is \
             untested"
        );
        let head = model.forward(&batch, &stats, false, false);
        let before = model.losses(&head, &stats, &targets, &mask);
        assert!(before.nll.double_value(&[]).is_finite());
        // Poison exactly the unobserved entries, wherever the mask says they are.
        let poisoned = targets.where_self(&unobserved.logical_not(), &Tensor::from(1e6f32));
        let after = model.losses(&head, &stats, &poisoned, &mask);
        assert!(
            after.nll.equal(&before.nll),
            "an unobserved bar moved the objective: {} vs {}",
            after.nll.double_value(&[]),
            before.nll.double_value(&[])
        );
        // And the converse: an OBSERVED bar must move it, so the invariance above is not
        // vacuous. The last origin of the complete row is observed at every horizon.
        let observed = targets.shallow_clone();
        let _ = observed
            .narrow(0, 0, 1)
            .narrow(1, config.origins() - 1, 1)
            .fill_(1e3);
        let moved = model.losses(&head, &stats, &observed, &mask);
        assert!(
            !moved.nll.equal(&before.nll),
            "poisoning an OBSERVED bar left the objective unchanged, so the rotated loss is \
             reading nothing"
        );
        // The rotated objective's denominator counts complete windows only, so the reported
        // number stays a per-element mean - nats per bar - and stays comparable to the
        // persistence anchor. Every scored origin except the incomplete ones is counted.
        let complete = mask
            .amin([-1i64].as_slice(), true)
            .sum(Kind::Double)
            .double_value(&[]);
        let scored = mask
            .amax([-1i64].as_slice(), true)
            .sum(Kind::Double)
            .double_value(&[]);
        assert!(
            complete > 0. && complete < scored,
            "the completeness mask must drop some but not all scored origins, got {complete} \
             of {scored}"
        );
    }

    /// The knobs that would make an arm unattributable are refused, not silently composed.
    #[test]
    fn a_rotated_basis_refuses_a_horizon_weighting_and_a_rank_restriction() {
        let error = ModelConfig {
            target_basis: TargetBasis::Dct,
            horizon_loss: HorizonLoss::InverseSqrt,
            ..small_config()
        }
        .validate()
        .unwrap_err()
        .to_string();
        assert!(error.contains("not the same axis"), "{error}");
        let error = ModelConfig {
            target_basis: TargetBasis::Haar,
            horizon_mean: HorizonMean::Basis {
                free: 2,
                functions: 2,
            },
            ..small_config()
        }
        .validate()
        .unwrap_err()
        .to_string();
        // The refusal is identified by the two knobs it names, not by its prose: a sibling
        // rewrote this message today and a wording assertion would have failed while the
        // behaviour was correct.
        assert!(
            error.contains("--target-basis") && error.contains("--horizon-mean"),
            "{error}"
        );
        let error = ModelConfig {
            basis_weight: BasisWeight::Snr,
            ..small_config()
        }
        .validate()
        .unwrap_err()
        .to_string();
        assert!(error.contains("--basis-stats"), "{error}");
        // And the control composes with everything, exactly as it did before the knob.
        ModelConfig {
            horizon_loss: HorizonLoss::InverseSqrt,
            ..small_config()
        }
        .validate()
        .unwrap();
    }

    /// `cutoff:K` must genuinely stop the gradient, not merely shrink it. The head weight is
    /// channel-major (`row = channel·pred_len + h`, see [`CausalPatchModel::head`]), so the
    /// rows exclusive to horizons above `K` are exactly those with `row % pred_len >= K`, and
    /// their gradient has to be the number 0, not a small number. Forecasts are still emitted
    /// for every horizon, which is what keeps the per-horizon diagnostics readable on the
    /// untrained end.
    #[test]
    fn a_horizon_cutoff_leaves_exactly_zero_gradient_above_it_and_still_forecasts_every_horizon() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(29);
        let cut = 3;
        let config = ModelConfig {
            horizon_loss: HorizonLoss::Cutoff(cut),
            ..small_config()
        };
        let pred_len = config.pred_len;
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        live_head(&model);
        let batch = synthetic(&config, &[8, 5]);
        let stats = model.statistics(&batch);
        let (targets, mask) = model.targets(&batch, &stats, false);
        let head = model.forward(&batch, &stats, false, false);
        // Every horizon is still emitted: the head shape is untouched by the mode.
        assert_eq!(
            model.output(&head).coordinates.size()[3],
            pred_len,
            "the cutoff must mask the loss, never shrink the head"
        );
        let losses = model.losses(&head, &stats, &targets, &mask);
        assert!(losses.nll.double_value(&[]).is_finite());
        let bias = model.head_output.bs.as_ref().unwrap();
        let grads = Tensor::run_backward(
            &[&losses.nll],
            &[&model.head_output.ws, bias, &model.patch.ws],
            false,
            false,
        );
        for (name, gradient) in [("weight", &grads[0]), ("bias", &grads[1])] {
            let rows = gradient.size()[0];
            assert_eq!(rows, pred_len * OUTPUTS_PER_BAR);
            let horizon = Tensor::arange(rows, (Kind::Int64, Device::Cpu)).remainder(pred_len);
            let magnitude = gradient
                .abs()
                .reshape([rows, -1])
                .sum_dim_intlist([1i64].as_slice(), false, Kind::Double);
            assert_eq!(
                magnitude.masked_select(&horizon.ge(cut)).max().double_value(&[]),
                0.0,
                "head {name} rows above cutoff:{cut} received gradient"
            );
            assert!(
                magnitude.masked_select(&horizon.lt(cut)).min().double_value(&[]) > 0.0,
                "head {name} rows at or below cutoff:{cut} received no gradient"
            );
        }
        // The trunk still trains: the cutoff removes horizons, not layers.
        assert!(grads[2].abs().max().double_value(&[]) > 0.0);
    }

    /// The spec parser is the only gate before a GPU-hours run starts, so it rejects at parse
    /// time. `cutoff:0` and garbage die in `FromStr`; a cutoff past the horizon needs
    /// `pred_len`, so it dies in [`ModelConfig::validate`] - the first statement of
    /// `runner::train`, long before the corpus loads.
    #[test]
    fn horizon_loss_specs_round_trip_and_invalid_ones_are_refused_before_any_run() {
        for mode in [
            HorizonLoss::Uniform,
            HorizonLoss::InverseSqrt,
            HorizonLoss::Inverse,
            HorizonLoss::Cutoff(1),
            HorizonLoss::Cutoff(32),
            HorizonLoss::Cutoff(192),
        ] {
            assert_eq!(mode.to_string().parse::<HorizonLoss>().unwrap(), mode);
            assert_eq!(
                serde_json::from_str::<HorizonLoss>(&serde_json::to_string(&mode).unwrap())
                    .unwrap(),
                mode
            );
        }
        assert_eq!("cutoff:32".parse::<HorizonLoss>().unwrap(), HorizonLoss::Cutoff(32));
        for garbage in [
            "cutoff:0",
            "cutoff:-4",
            "cutoff:",
            "cutoff:1.5",
            "cutoff",
            "uniform ",
            "Uniform",
            "inv_sqrt",
            "inverse",
            "",
            "1/h",
        ] {
            assert!(
                garbage.parse::<HorizonLoss>().is_err(),
                "{garbage:?} parsed as a horizon loss"
            );
        }
        // Past the horizon: accepted by the parser, refused by the configuration.
        let config = ModelConfig {
            horizon_loss: HorizonLoss::Cutoff(193),
            ..ModelConfig::default()
        };
        assert_eq!(config.pred_len, 192);
        let error = config.validate().unwrap_err().to_string();
        assert!(
            error.contains("cutoff:193") && error.contains("1..=192"),
            "{error}"
        );
        assert!(ModelConfig {
            horizon_loss: HorizonLoss::Cutoff(192),
            ..ModelConfig::default()
        }
        .validate()
        .is_ok());
    }

    /// `--horizon-mean free` is the CONTROL: every measurement the structured mean will be
    /// compared against was taken by the head this branch has to reproduce, so "equal to
    /// within tolerance" is not good enough - it has to be the same bits. And the fold's
    /// one-hot rows have to be exact too, because that is what makes the free band and every
    /// log scale under `basis:S:B` the same numbers the dense head would have produced: a
    /// row of `Ψ` that sums one weight against `head_outputs - 1` exact zeros must return the
    /// weight itself, not a rounding of it.
    #[test]
    fn free_horizon_mean_is_the_dense_head_and_the_fold_copies_one_hot_rows_exactly() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(31);
        let dense = ModelConfig {
            features: FeatureSet::NONE,
            ..small_config()
        };
        assert_eq!(dense.horizon_mean, HorizonMean::Free);
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &dense);
        assert!(
            model.mean_expansion.is_none(),
            "the control arm must not carry an expansion at all"
        );
        tch::no_grad(|| {
            let _ = model.head_output.ws.shallow_clone().uniform_(-0.3, 0.3);
            let _ = model
                .head_output
                .bs
                .as_ref()
                .expect("the head output is biased")
                .shallow_clone()
                .uniform_(-0.3, 0.3);
        });
        let batch = synthetic(&dense, &[8, 6]);
        let stats = model.statistics(&batch);
        let state = model.backbone(&batch, &stats, false, false);
        let head = model.head(&batch, &state, false);
        // The pre-knob body, transcribed. `FeatureSet::NONE` has no known-future channel, so
        // the covariate branch is absent and the head input IS the backbone state, which is
        // what lets the reference be the two lines that matter rather than a second copy of
        // the covariate gather.
        let hidden = linear(&state, &model.head_hidden).gelu("none");
        let reference = hidden
            .linear(
                &(model.head_output.ws.to_kind(hidden.kind()) * HEAD_OUTPUT_SCALE),
                model
                    .head_output
                    .bs
                    .as_ref()
                    .map(|bias| bias.to_kind(hidden.kind()) * HEAD_OUTPUT_SCALE),
            )
            .reshape([2, -1, OUTPUTS_PER_BAR, dense.pred_len]);
        assert!(
            head.0.equal(&reference),
            "the free head is no longer the dense GEMM it was before the knob"
        );
        // Now the fold itself, on the full-covariate configuration, against an expansion that
        // is nothing but one-hot rows: the identity. Same weights, same batch, same bits.
        let covariate_config = small_config();
        let folded_store = nn::VarStore::new(Device::Cpu);
        let mut folded = CausalPatchModel::new(&folded_store.root(), &covariate_config);
        let plain_store = nn::VarStore::new(Device::Cpu);
        let plain = CausalPatchModel::new(&plain_store.root(), &covariate_config);
        folded.mean_expansion = Some(Tensor::eye(
            OUTPUTS_PER_BAR * covariate_config.pred_len,
            (Kind::Float, Device::Cpu),
        ));
        tch::no_grad(|| {
            for store in [&plain_store, &folded_store] {
                tch::manual_seed(37);
                for mut variable in store.trainable_variables() {
                    let _ = variable.uniform_(-0.2, 0.2);
                }
            }
        });
        let batch = synthetic(&covariate_config, &[8, 8]);
        let stats = plain.statistics(&batch);
        let expected = plain.forward(&batch, &stats, false, false);
        let through_fold = folded.forward(&batch, &stats, false, false);
        assert!(
            through_fold.0.equal(&expected.0),
            "folding a one-hot expansion onto the head weight is not a copy"
        );
    }

    /// The point of the restriction is that the free means above the band ARE NOT THERE. A
    /// masked or frozen parameter is a different experiment: it still occupies optimizer
    /// state, it still counts against weight decay, and one missing mask makes it trainable
    /// again. So this checks the varstore itself - same entries, `704 · (HEAD_HIDDEN + 1)`
    /// fewer numbers - and then checks that no surviving parameter is exclusive to one long
    /// horizon, which is the property the deleted rows used to violate.
    #[test]
    fn basis_means_delete_the_long_horizon_parameters_rather_than_masking_them() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(41);
        let (free, functions) = (8i64, 8i64);
        let control = ModelConfig {
            pred_len: 192,
            ..small_config()
        };
        let structured = ModelConfig {
            horizon_mean: HorizonMean::Basis { free, functions },
            ..control.clone()
        };
        let control_store = nn::VarStore::new(Device::Cpu);
        let _control_model = CausalPatchModel::new(&control_store.root(), &control);
        let structured_store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&structured_store.root(), &structured);
        let names = |store: &nn::VarStore| {
            let mut names: Vec<String> = store.variables().into_keys().collect();
            names.sort();
            names
        };
        assert_eq!(
            names(&control_store),
            names(&structured_store),
            "the structured mean must remove parameters from a tensor, not tensors from the model"
        );
        let numbers = |store: &nn::VarStore| {
            store
                .variables()
                .values()
                .map(|variable| variable.numel())
                .sum::<usize>()
        };
        let (dense_rows, rows) = (
            OUTPUTS_PER_BAR * control.pred_len,
            structured.horizon_mean.head_outputs(control.pred_len),
        );
        assert_eq!((dense_rows, rows), (1536, 832));
        assert_eq!(model.head_output.ws.size(), [rows, HEAD_HIDDEN]);
        assert_eq!(
            model
                .head_output
                .bs
                .as_ref()
                .expect("the head output is biased")
                .size(),
            [rows]
        );
        assert_eq!(
            numbers(&control_store) - numbers(&structured_store),
            (dense_rows - rows) as usize * (HEAD_HIDDEN as usize + 1),
            "the parameter drop is the deleted rows and their biases, exactly"
        );
        // One restricted horizon's gradient. In the dense head this reaches exactly one mean
        // row per coordinate channel and nothing else, which is precisely the freedom that
        // memorized the training period. Here it must reach every coefficient row of every
        // coordinate channel and NOT ONE row of the free band.
        let batch = synthetic(&structured, &[192, 192]);
        let stats = model.statistics(&batch);
        let head = model.forward(&batch, &stats, true, false);
        let probe = 100;
        assert!(probe > free && probe < control.pred_len);
        head.0
            .narrow(2, 0, CHANNELS)
            .narrow(3, probe, 1)
            .sum(Kind::Float)
            .backward();
        let gradient = model.head_output.ws.grad().abs().sum_dim_intlist(
            [1i64].as_slice(),
            false,
            Kind::Double,
        );
        let per_channel = free + functions;
        for channel in 0..CHANNELS {
            for slot in 0..per_channel {
                let touched = gradient.double_value(&[channel * per_channel + slot]) > 0.0;
                assert_eq!(
                    touched,
                    slot >= free,
                    "channel {channel} slot {slot}: a restricted horizon must move the {functions} \
                     coefficients and none of the {free} free means"
                );
            }
        }
        assert_eq!(
            gradient
                .narrow(0, CHANNELS * per_channel, CHANNELS * control.pred_len)
                .gt(0.0)
                .sum(Kind::Int64)
                .int64_value(&[]),
            0,
            "a mean gradient must not reach the log-scale rows"
        );
        // And the cost side of the same change, at the shipped batch: the token-space GEMMs
        // and every activation are untouched, and the fold's own arithmetic and traffic are
        // the only movement. These literals are what `timexer_basis_means.md` quotes.
        let (dense_cost, basis_cost) = (
            ModelConfig {
                horizon_mean: HorizonMean::Free,
                ..ModelConfig::default()
            }
            .step_cost(256),
            ModelConfig {
                horizon_mean: HorizonMean::Basis { free, functions },
                ..ModelConfig::default()
            }
            .step_cost(256),
        );
        assert_eq!(
            basis_cost.matmul_flops - dense_cost.matmul_flops,
            2. * 2. * 1536. * 832. * 1024.
        );
        assert_eq!(
            basis_cost.traffic_bytes - dense_cost.traffic_bytes,
            4. * 2. * 1536. * 1024.
        );
        println!(
            "basis:8:8 at batch 256: {:.5} TFLOP and {:.4} GB against the dense head's {:.5} \
             TFLOP and {:.4} GB; head output weight {rows}x{HEAD_HIDDEN} against \
             {dense_rows}x{HEAD_HIDDEN}, {} parameters gone",
            basis_cost.matmul_flops / 1e12,
            basis_cost.traffic_bytes / 1e9,
            dense_cost.matmul_flops / 1e12,
            dense_cost.traffic_bytes / 1e9,
            numbers(&control_store) - numbers(&structured_store)
        );
    }

    /// Zero coefficients must be EXACTLY the persistence forecast, not approximately it: the
    /// head is zero-initialised, so this is the forecast the first step departs from, and the
    /// whole mis-scaling diagnostic is measured as a departure from it. A basis with an
    /// intercept, or an expansion that added anything to the restricted rows, would show up
    /// here as a nonzero coordinate.
    #[test]
    fn zero_basis_coefficients_are_exactly_persistence_above_the_free_band() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(43);
        let (free, functions) = (3i64, 2i64);
        let config = ModelConfig {
            horizon_mean: HorizonMean::Basis { free, functions },
            ..small_config()
        };
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let per_channel = free + functions;
        tch::no_grad(|| {
            let _ = model.head_output.ws.shallow_clone().uniform_(-0.5, 0.5);
            let bias = model
                .head_output
                .bs
                .as_ref()
                .expect("the head output is biased");
            let _ = bias.shallow_clone().uniform_(-0.5, 0.5);
            // Every coefficient to zero; the free band and the log scales keep their values,
            // so a restricted horizon that is not exactly persistence can only come from the
            // basis.
            for channel in 0..CHANNELS {
                let _ = model
                    .head_output
                    .ws
                    .narrow(0, channel * per_channel + free, functions)
                    .fill_(0.0);
                let _ = bias.narrow(0, channel * per_channel + free, functions).fill_(0.0);
            }
        });
        let batch = synthetic(&config, &[8, 8]);
        let stats = model.statistics(&batch);
        let output = model.output(&model.forward(&batch, &stats, false, false));
        let restricted = config.pred_len - free;
        let above = output.coordinates.narrow(3, free, restricted);
        assert_eq!(
            above.abs().max().double_value(&[]),
            0.0,
            "a zero coefficient must give a zero coordinate, bit for bit"
        );
        let scaled = model.decode(&output, &stats);
        for channel in [0, 3] {
            assert_eq!(
                scaled
                    .narrow(2, channel, 1)
                    .narrow(3, free, restricted)
                    .abs()
                    .max()
                    .double_value(&[]),
                0.0,
                "the open and close means above the free band must be the origin close exactly"
            );
        }
        assert!(
            output
                .coordinates
                .narrow(3, 0, free)
                .abs()
                .max()
                .double_value(&[])
                > 0.0,
            "the free band must still be free, or this test proves nothing"
        );
        assert!(
            output.log_scale.std(false).double_value(&[]) > 0.0,
            "the log scales are deliberately unrestricted and must still vary per horizon"
        );
    }

    /// THE restriction, as a measurement rather than a claim. A term structure of expected
    /// return - here a hyperbolic decay, which is completely monotone and so a genuine
    /// exponential mixture, but NOT one of the eight functions - has to survive the
    /// projection. A per-horizon alternating pattern, which needs one sign change per horizon
    /// against a span whose members change sign at most `functions - 1` times, has to be
    /// destroyed by it. Both are pushed through the real head: the fitted coefficients are
    /// written into the head bias, so what is measured is what the model can emit.
    #[test]
    fn the_basis_represents_a_term_structure_and_refuses_per_horizon_idiosyncrasy() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(47);
        let (free, functions) = (8i64, 8i64);
        let config = ModelConfig {
            pred_len: 192,
            horizon_mean: HorizonMean::Basis { free, functions },
            ..small_config()
        };
        let restricted = config.pred_len - free;
        let basis = Tensor::from_slice(
            &config
                .horizon_mean
                .basis(config.pred_len)
                .expect("the basis mode has a basis"),
        )
        .reshape([restricted, functions])
        .to_kind(Kind::Double);
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let batch = synthetic(&config, &[8]);
        let stats = model.statistics(&batch);
        let bias = model
            .head_output
            .bs
            .as_ref()
            .expect("the head output is biased")
            .shallow_clone();
        // The head weight stays zero, so the emitted block is the folded bias alone and the
        // coefficients are read straight off it. `HEAD_OUTPUT_SCALE` folds onto the bias too,
        // hence the pre-division.
        let emit = |coefficients: &Tensor| {
            tch::no_grad(|| {
                let _ = bias.shallow_clone().fill_(0.0);
                let _ = bias
                    .narrow(0, free, functions)
                    .copy_(&(coefficients.to_kind(Kind::Float) / HEAD_OUTPUT_SCALE));
            });
            model
                .output(&model.forward(&batch, &stats, false, false))
                .coordinates
                .narrow(2, 0, 1)
                .narrow(3, free, restricted)
                .reshape([-1, restricted])
                .narrow(0, 0, 1)
                .reshape([restricted])
                .to_kind(Kind::Double)
        };
        let steps = (Tensor::arange(restricted, (Kind::Double, Device::Cpu)) + 1.0).reshape([restricted]);
        let cases = [
            // A term structure: signal-to-noise decaying hyperbolically over 24 bars. Bernstein
            // says a completely monotone decay is a positive mixture of exponentials, so the
            // geometric timescale grid should reach it closely.
            ("hyperbolic term structure", (&steps / 24.0 + 1.0).reciprocal() * 0.8, 0.05),
            // Per-horizon idiosyncrasy: one sign change per horizon.
            (
                "per-horizon alternating",
                (&steps * std::f64::consts::PI).cos() * 0.8,
                0.95,
            ),
        ];
        let relative = |curve: &Tensor, target: &Tensor| {
            ((curve - target).square().sum(Kind::Double).sqrt()
                / target.square().sum(Kind::Double).sqrt())
            .double_value(&[])
        };
        for (name, target, bound) in cases {
            let (coefficients, ..) =
                basis.linalg_lstsq(&target.reshape([restricted, 1]), None, "gelsd");
            let coefficients = coefficients.reshape([functions]);
            // Two numbers, because they answer two different questions. `span` is the best the
            // SPAN can do, in fp64: the restriction itself, independent of any head. `head` is
            // what the model actually emits once those coefficients are written into the head
            // bias and expanded in fp32, and it is what the tolerance is stated on. The gap
            // between them is the fp32 rounding of an fp64 least-squares solution that reaches
            // its last digits through large cancelling coefficients - the Gram matrix of eight
            // exponentials is ill-conditioned by construction, which is a property of this
            // parameterization and not of its span. On a target the span cannot reach, that
            // same cancellation can push the emitted curve past 1, further from the target than
            // emitting nothing at all.
            let span = relative(&basis.matmul(&coefficients), &target);
            let head = relative(&emit(&coefficients), &target);
            println!("{name}: best relative L2 residual, span {span:.4}, through the head {head:.4}");
            if bound < 0.5 {
                assert!(
                    span < bound,
                    "{name} must be representable to {bound}, the span reached {span}"
                );
                assert!(
                    head < bound,
                    "{name} must be EMITTABLE to {bound}, the head reached {head} against the \
                     span's own {span}"
                );
            } else {
                assert!(
                    span > bound,
                    "{name} must NOT be representable - that is the restriction - but the span \
                     fitted it to {span}"
                );
                assert!(
                    head > bound,
                    "the head emitted {head} for a target the span fits no better than {span}"
                );
            }
        }
        // The mechanism behind the contrast, stated where it can be checked: every basis
        // function is positive and strictly decreasing, so the span is smooth and monotone
        // by construction rather than by fitting.
        let column = |index: i64| basis.narrow(1, index, 1).reshape([restricted]);
        for index in 0..functions {
            let values = column(index);
            let difference = values.narrow(0, 1, restricted - 1) - values.narrow(0, 0, restricted - 1);
            assert_eq!(values.gt(0.0).all().int64_value(&[]), 1);
            assert_eq!(difference.lt(0.0).all().int64_value(&[]), 1);
        }
        let timescales = config.horizon_mean.timescales(config.pred_len);
        assert_eq!(timescales.len(), functions as usize);
        assert!((timescales[0] - free as f64).abs() < 1e-9);
        assert!((timescales[functions as usize - 1] - config.pred_len as f64).abs() < 1e-9);
        for pair in timescales.windows(2) {
            assert!(pair[1] > pair[0], "the timescale grid must be strictly increasing");
        }
    }

    #[test]
    fn horizon_mean_specs_round_trip_and_invalid_ones_are_refused_before_any_run() {
        for mode in [
            HorizonMean::Free,
            HorizonMean::Basis {
                free: 8,
                functions: 8,
            },
            HorizonMean::Basis {
                free: 1,
                functions: 1,
            },
        ] {
            assert_eq!(mode.to_string().parse::<HorizonMean>().unwrap(), mode);
            assert_eq!(
                serde_json::from_str::<HorizonMean>(&serde_json::to_string(&mode).unwrap())
                    .unwrap(),
                mode
            );
        }
        assert_eq!(
            "basis:8:8".parse::<HorizonMean>().unwrap(),
            HorizonMean::Basis {
                free: 8,
                functions: 8
            }
        );
        assert_eq!(HorizonMean::default(), HorizonMean::Free);
        assert_eq!(
            HorizonMean::Basis {
                free: 8,
                functions: 8
            }
            .stamp(),
            "basis-8-8"
        );
        for garbage in [
            "basis:0:8",
            "basis:8:0",
            "basis:-8:8",
            "basis:8:-8",
            "basis:8",
            "basis:8:",
            "basis::8",
            "basis:8:8:8",
            "basis:8.5:8",
            "basis",
            "basis:",
            "free ",
            "Free",
            "dense",
            "",
        ] {
            assert!(
                garbage.parse::<HorizonMean>().is_err(),
                "{garbage:?} parsed as a horizon mean"
            );
        }
        // Accepted by the parser, refused by the configuration - before the corpus loads.
        let past_the_horizon = ModelConfig {
            horizon_mean: HorizonMean::Basis {
                free: 192,
                functions: 1,
            },
            ..ModelConfig::default()
        };
        assert_eq!(past_the_horizon.pred_len, 192);
        let error = past_the_horizon.validate().unwrap_err().to_string();
        assert!(
            error.contains("basis:192:1") && error.contains("no restricted horizon"),
            "{error}"
        );
        let too_many = ModelConfig {
            horizon_mean: HorizonMean::Basis {
                free: 8,
                functions: 185,
            },
            ..ModelConfig::default()
        };
        let error = too_many.validate().unwrap_err().to_string();
        assert!(
            error.contains("basis:8:185") && error.contains("184"),
            "{error}"
        );
        assert!(ModelConfig {
            horizon_mean: HorizonMean::Basis {
                free: 191,
                functions: 1
            },
            ..ModelConfig::default()
        }
        .validate()
        .is_ok());
        assert!(ModelConfig {
            horizon_mean: HorizonMean::Basis {
                free: 8,
                functions: 184
            },
            ..ModelConfig::default()
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn cuda_fresh_model_trains_every_branch() {
        let _rng = crate::torch::test_rng::exclusive();
        if std::env::var("TIMEXER_SEGMENT_GPU_TEST").as_deref() != Ok("1") {
            return;
        }
        assert!(tch::Cuda::is_available());
        crate::torch::cuda::cfg::configure_cuda();
        crate::torch::cuda::cfg::enable_tf32_matmul().unwrap();
        let device = Device::Cuda(0);
        let _backward = crate::torch::cuda::cfg::disable_autograd_multithreading();
        tch::manual_seed(20260905);
        tch::Cuda::manual_seed_all(20260905);
        let config = ModelConfig::default();
        let store = nn::VarStore::new(device);
        let model = CausalPatchModel::new(&store.root(), &config);
        let batch = synthetic(&config, &[192, 100]).to_device(device);
        let stats = model.statistics(&batch);
        use tch::nn::OptimizerConfig;
        let mut optimizer = tch::nn::Sgd::default().build(&store, 0.1).unwrap();
        for _ in 0..2 {
            let head = model.forward(&batch, &stats, true, false);
            let (targets, mask) = model.targets(&batch, &stats, false);
            let losses = model.losses(&head, &stats, &targets, &mask);
            assert!(losses.nll.double_value(&[]).is_finite());
            optimizer.zero_grad();
            losses.nll.backward();
            optimizer.step();
        }
        for (name, parameter) in store.variables() {
            let gradient = parameter.grad();
            assert!(gradient.defined(), "missing gradient: {name}");
            assert_eq!(parameter.kind(), Kind::Float, "master dtype: {name}");
            assert_eq!(gradient.isfinite().all().int64_value(&[]), 1, "nonfinite gradient: {name}");
            assert!(
                gradient.abs().sum(Kind::Float).double_value(&[]) > 0.0,
                "a branch received no gradient: {name}"
            );
        }
    }

    /// A small config with the real head geometry: `head_dim` even, rotary over the whole head.
    fn narrow_config() -> ModelConfig {
        ModelConfig {
            seq_len: 96,
            pred_len: 8,
            patch_len: 8,
            layers: 2,
            d_model: 32,
            heads: 4,
            ffn: 64,
            min_history: 8,
            ..Default::default()
        }
    }

    /// The packed full-width rotation, the single `split_with_sizes` and the strided attention
    /// views must produce EXACTLY the tensors the per-tensor half-width form produced - not
    /// within a tolerance. The whole point of the rewrite is that it is a regrouping of
    /// kernels, so any nonzero difference means an operand moved, and the composed block
    /// output and every parameter gradient must be bit-identical too.
    #[test]
    fn the_packed_rotation_and_split_match_the_per_tensor_reference_bit_for_bit() {
        let _rng = crate::torch::test_rng::exclusive();
        let config = narrow_config();
        let store = nn::VarStore::new(Device::Cpu);
        let mut block = Block::new(store.root() / "block", &config, 0);
        let (width, heads) = (config.d_model, config.heads);
        let head_dim = width / heads;
        let (rows, length) = (2, config.origins());
        let rope = RotaryEmbedding::new(length, head_dim, head_dim, Device::Cpu);
        let (cosine, sine) = rope.cached_rotation(
            &Tensor::arange(length, (Kind::Int64, Device::Cpu)),
            Kind::BFloat16,
        );
        let input = (Tensor::randn([rows, length, width], (Kind::Float, Device::Cpu)) * 0.5)
            .to_kind(Kind::BFloat16)
            .set_requires_grad(true);
        let x0 = (Tensor::randn([rows, length, width], (Kind::Float, Device::Cpu)) * 0.5)
            .to_kind(Kind::BFloat16)
            .set_requires_grad(true);
        // Distinct, non-unit lambdas and non-zero residual-branch weights: at the real init the
        // output projections are zero and `x0_lambda` is zero, which would make the whole
        // feedforward half of this comparison `0 == 0` and leave four parameter gradients with
        // no signal at all.
        let lambdas = store.root() / "lambdas";
        let mut resid = lambdas.var("resid", &[2], nn::Init::Const(0.0));
        let mut post = lambdas.var("post", &[2], nn::Init::Const(0.0));
        let mut x0_lambda = lambdas.var("x0", &[1], nn::Init::Const(0.0));
        tch::no_grad(|| {
            resid.copy_(&Tensor::from_slice(&[1.05_f32, 0.9]));
            post.copy_(&Tensor::from_slice(&[0.8_f32, 1.3]));
            x0_lambda.copy_(&Tensor::from_slice(&[0.25_f32]));
            for weight in [&mut block.output.ws, &mut block.second.ws] {
                let noise =
                    Tensor::randn(weight.size(), (Kind::Float, Device::Cpu)) * 0.05;
                weight.copy_(&noise);
            }
        });
        let (resid_bf16, post_bf16, x0_bf16) = (
            resid.to_kind(Kind::BFloat16),
            post.to_kind(Kind::BFloat16),
            x0_lambda.to_kind(Kind::BFloat16),
        );
        let (resid_parts, post_parts) = (resid_bf16.unbind(0), post_bf16.unbind(0));
        let x0_scale = x0_bf16.get(0);
        let block_lambdas = BlockLambdas {
            resid: [&resid_parts[0], &resid_parts[1]],
            post: [&post_parts[0], &post_parts[1]],
            x0: Some((&x0, &x0_scale)),
        };
        let projection = linear(&rms_norm(&input), &block.qkv);
        let parts = projection.split(width, -1);
        let per_head = |part: &Tensor| {
            part.reshape([rows, length, heads, head_dim])
                .transpose(1, 2)
        };
        // The per-tensor reference for QK-norm: normalize q and k SEPARATELY over `head_dim`,
        // then rotate. Row-wise normalization is independent per row, so normalizing the packed
        // `q‖k` block in one kernel must reproduce this bit for bit.
        let normed_head = |part: &Tensor| {
            rms_norm(&part.reshape([rows, length, heads, head_dim])).transpose(1, 2)
        };
        let reference = [
            rope.apply_cached(&normed_head(&parts[0]), &cosine, &sine),
            rope.apply_cached(&normed_head(&parts[1]), &cosine, &sine),
            per_head(&parts[2]),
        ];
        let packed = projection.split_with_sizes([2 * width, width], -1);
        let normed_packed = rms_norm(&packed[0].reshape([rows, length, 2 * heads, head_dim]))
            .reshape([rows, length, 2 * width]);
        let rotated = fused_rope(&normed_packed, &cosine, &sine, heads).split(1, 2);
        let query_key = |tensor: &Tensor| tensor.squeeze_dim(2).transpose(1, 2);
        let fused = [
            query_key(&rotated[0]),
            query_key(&rotated[1]),
            per_head(&packed[1]),
        ];
        for (name, expected, actual) in [
            ("query", &reference[0], &fused[0]),
            ("key", &reference[1], &fused[1]),
            ("value", &reference[2], &fused[2]),
        ] {
            assert!(
                expected.abs().max().double_value(&[]) > 0.0,
                "{name} carries no signal, so equality proves nothing"
            );
            assert_eq!(
                (expected - actual).abs().max().double_value(&[]),
                0.0,
                "{name} is not bit-identical to the per-tensor rotation"
            );
        }
        // Rotation is not the identity: the kernel must actually rotate, or the test above
        // would pass on a pair of untouched projections.
        assert!(
            (&reference[0] - per_head(&parts[0])).abs().max().double_value(&[]) > 0.0,
            "the rotary rows left the query unchanged"
        );
        let attended = Tensor::scaled_dot_product_attention(
            &reference[0],
            &reference[1],
            &reference[2],
            None::<&Tensor>,
            0.0,
            true,
            None,
            false,
        )
        .transpose(1, 2)
        .reshape([rows, length, width]);
        let state = scaled_linear(&attended, &block.output, block_lambdas.post[0])
            .addcmul(&input, block_lambdas.resid[0])
            .addcmul(&x0, &x0_scale);
        let expected = scaled_linear(
            &linear(&rms_norm(&state), &block.first).relu().square(),
            &block.second,
            block_lambdas.post[1],
        )
        .addcmul(&state, block_lambdas.resid[1]);
        let (actual, published) =
            block.forward(&input, None, &block_lambdas, (&cosine, &sine), false);
        // The SOURCE layer publishes exactly its own value, and mixes nothing into it.
        assert!(block.value_lambda.is_none(), "layer 0 owns no lambda");
        assert_eq!(
            (published.expect("layer 0 publishes its value").transpose(1, 2) - &fused[2])
                .abs()
                .max()
                .double_value(&[]),
            0.0,
            "the published value is not layer 0's own value"
        );
        assert_eq!(
            (&expected - &actual).abs().max().double_value(&[]),
            0.0,
            "the composed block output is not bit-identical"
        );
        let mut targets = vec![input.shallow_clone(), x0.shallow_clone()];
        targets.extend(store.trainable_variables());
        let expected_grads = Tensor::run_backward(&[&expected], &targets, true, false);
        let actual_grads = Tensor::run_backward(&[&actual], &targets, false, false);
        for (index, (left, right)) in expected_grads.iter().zip(&actual_grads).enumerate() {
            let scale = left.abs().max().double_value(&[]);
            assert!(scale > 0.0, "gradient {index} carries no signal");
            assert_eq!(
                (left - right).abs().max().double_value(&[]),
                0.0,
                "gradient {index} is not bit-identical"
            );
        }
        println!(
            "packed rotation vs per-tensor reference: 0.0e0 on q/k/v, on the composed block \
             output and on all {} gradients",
            targets.len()
        );
    }

    /// Casting each part to bf16 BEFORE the concatenation must give exactly the tensor that
    /// concatenating in fp32 and casting after gave: an elementwise cast commutes with
    /// concatenation, and that is the whole argument for moving 197 MB less per step.
    #[test]
    fn the_patch_tokens_cast_before_concatenation_are_bit_identical() {
        let _rng = crate::torch::test_rng::exclusive();
        let config = narrow_config();
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let batch = synthetic(&config, &[8, 5]);
        let stats = model.statistics(&batch);
        let (context, patch, origins) = (config.seq_len, config.patch_len, config.origins());
        let aux_channels = config.features.channels() as i64;
        let rows = batch.log_prices.size()[0];
        let inv_sigma = per_bar(&stats.sigma.reciprocal());
        let prices = (batch
            .log_prices
            .narrow(1, 0, context)
            .reshape([rows, origins, patch, CHANNELS])
            - per_bar(&stats.log_close))
            * &inv_sigma;
        let auxiliaries = batch
            .aux
            .narrow(1, 0, context)
            .reshape([rows, origins, patch, aux_channels])
            * (&model.sigma_scale * inv_sigma + &model.unit_scale);
        let expected = Tensor::cat(&[prices, auxiliaries], 3)
            .reshape([rows, origins, patch * (CHANNELS + aux_channels)])
            .to_kind(Kind::BFloat16);
        let actual = model.tokens(&batch, &stats);
        assert_eq!(actual.kind(), Kind::BFloat16);
        assert!(expected.abs().max().double_value(&[]) > 0.0);
        assert_eq!(
            (&expected - &actual).abs().max().double_value(&[]),
            0.0,
            "casting before the concatenation changed the tokens"
        );
    }

    /// RMSNorm must be `x·rsqrt(mean(x²)+eps)` over the last axis with NO gain and NO mean
    /// subtraction, computed against a scalar reference rather than against another ATen call.
    /// The mean subtraction is the part that is easy to leave in by reaching for LayerNorm with
    /// `elementwise_affine=False`: on a tensor with a nonzero row mean the two differ.
    #[test]
    fn rms_norm_matches_a_scalar_reference_and_keeps_the_row_mean() {
        let _rng = crate::torch::test_rng::exclusive();
        // A deliberate nonzero row mean, so subtracting it would change the answer.
        let rows: [[f64; 4]; 2] = [[1.0, 2.0, 3.0, 4.0], [-0.5, 0.25, 8.0, -2.0]];
        let flat: Vec<f32> = rows
            .iter()
            .flat_map(|row| row.iter().map(|&value| value as f32))
            .collect();
        let input = Tensor::from_slice(&flat).reshape([2, 4]);
        let actual = rms_norm(&input);
        for (row_index, row) in rows.iter().enumerate() {
            let mean_square = row.iter().map(|value| value * value).sum::<f64>() / 4.0;
            let scale = 1.0 / (mean_square + NORM_EPS).sqrt();
            for (column, value) in row.iter().enumerate() {
                let expected = value * scale;
                let got = actual.double_value(&[row_index as i64, column as i64]);
                assert!(
                    (got - expected).abs() < 1e-6,
                    "rms_norm[{row_index}][{column}] = {got}, expected {expected}"
                );
            }
            // A LayerNorm would have centred the row; RMSNorm must not.
            let row_mean: f64 = (0..4)
                .map(|column| actual.double_value(&[row_index as i64, column]))
                .sum::<f64>()
                / 4.0;
            let expected_mean = row.iter().sum::<f64>() / 4.0 * scale;
            assert!(
                (row_mean - expected_mean).abs() < 1e-6,
                "rms_norm centred row {row_index}: mean {row_mean}, expected {expected_mean}"
            );
        }
        // Gainless: the norm registers no variables, so there is nothing for an optimizer to
        // route. This is what makes `every_causal_patch_parameter_lands_in_its_intended_\
        // optimizer_group`'s NorMuon list exactly four matrices per block.
        let store = nn::VarStore::new(Device::Cpu);
        let before = store.len();
        let _ = rms_norm(&input.to_device(Device::Cpu));
        assert_eq!(store.len(), before, "rms_norm must register no parameters");
    }

    /// The FFN activation must be `relu(x)²`, not GELU and not `relu(x²)`. The two wrong forms
    /// differ exactly where it matters: GELU is smooth and nonzero for small negatives,
    /// `relu(x²)` is `x²` everywhere and never zero.
    #[test]
    fn relu_squared_matches_its_definition_and_is_not_gelu() {
        let points: [f64; 7] = [-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0];
        let flat: Vec<f32> = points.iter().map(|&value| value as f32).collect();
        let input = Tensor::from_slice(&flat);
        let actual = input.relu().square();
        for (index, &value) in points.iter().enumerate() {
            let expected = value.max(0.0).powi(2);
            let got = actual.double_value(&[index as i64]);
            assert!(
                (got - expected).abs() < 1e-6,
                "relu² at {value} = {got}, expected {expected}"
            );
        }
        // Nonzero separation from both plausible mistakes, on the negative half-line.
        let gelu = input.gelu("none");
        assert!(
            (&actual - &gelu).abs().max().double_value(&[]) > 1e-3,
            "relu² is indistinguishable from GELU on this range, so the test proves nothing"
        );
        assert!(
            (&actual - &input.square().relu())
                .abs()
                .max()
                .double_value(&[])
                > 1e-3,
            "relu² must not equal relu(x²)"
        );
    }

    /// QK-norm is applied per HEAD over `head_dim`, to q and k only, and BEFORE the rotation -
    /// modded-nanogpt `train_gpt.py:1106` then `:1109`. Checked against a hand-computed case
    /// small enough to write out: 1 token, 2 heads, `head_dim` 2, and a rotation angle of 0 at
    /// position 0 so the expected values are the normalized projections themselves.
    #[test]
    fn qk_norm_is_per_head_before_the_rotation_and_leaves_the_value_path_alone() {
        let _rng = crate::torch::test_rng::exclusive();
        let (heads, head_dim) = (2_i64, 2_i64);
        let width = heads * head_dim;
        // q‖k for one token: head 0 = (3, 4) with RMS √12.5, head 1 = (1, 0) with RMS √0.5,
        // then the same two heads again for k.
        let packed_rows: [f64; 8] = [3.0, 4.0, 1.0, 0.0, -2.0, 0.0, 0.5, 0.5];
        let flat: Vec<f32> = packed_rows.iter().map(|&value| value as f32).collect();
        let packed = Tensor::from_slice(&flat).reshape([1, 1, 2 * width]);
        let normed = rms_norm(&packed.reshape([1, 1, 2 * heads, head_dim]))
            .reshape([1, 1, 2 * width]);
        for head in 0..4_i64 {
            let pair = [
                packed_rows[(head * head_dim) as usize],
                packed_rows[(head * head_dim + 1) as usize],
            ];
            let scale = 1.0 / ((pair[0] * pair[0] + pair[1] * pair[1]) / 2.0 + NORM_EPS).sqrt();
            for lane in 0..head_dim {
                let expected = pair[lane as usize] * scale;
                let got = normed.double_value(&[0, 0, head * head_dim + lane]);
                assert!(
                    (got - expected).abs() < 1e-6,
                    "QK norm head {head} lane {lane} = {got}, expected {expected}"
                );
            }
        }
        // Normalizing over the whole packed axis instead of per head would give a DIFFERENT
        // answer, so the reshape is load-bearing and not decoration.
        assert!(
            (&normed - &rms_norm(&packed)).abs().max().double_value(&[]) > 1e-3,
            "per-head and whole-row normalization agree here, so this test proves nothing"
        );
        // Order: the block normalizes and THEN rotates. At position 0 the rotation is the
        // identity (cos 1, sin 0), so `rotate(norm(x))` must equal `norm(x)` exactly, while
        // `norm(rotate(x))` would too - the order is only observable off position 0. Use two
        // positions and compare the block's own path against the reference order.
        let config = ModelConfig {
            seq_len: 32,
            pred_len: 4,
            patch_len: 8,
            layers: 1,
            d_model: width,
            heads,
            ffn: 8,
            min_history: 4,
            ..Default::default()
        };
        let length = config.origins();
        let rope = RotaryEmbedding::new(length, head_dim, head_dim, Device::Cpu);
        let (cosine, sine) = rope.cached_rotation(
            &Tensor::arange(length, (Kind::Int64, Device::Cpu)),
            Kind::Float,
        );
        let projection = Tensor::randn([1, length, 2 * width], (Kind::Float, Device::Cpu));
        let norm_then_rotate = fused_rope(
            &rms_norm(&projection.reshape([1, length, 2 * heads, head_dim]))
                .reshape([1, length, 2 * width]),
            &cosine,
            &sine,
            heads,
        );
        let rotate_then_norm = rms_norm(
            &fused_rope(&projection, &cosine, &sine, heads)
                .reshape([1, length, 2 * heads, head_dim]),
        )
        .reshape([1, length, 2, heads, head_dim]);
        // RoPE preserves each head's norm, so in exact arithmetic the two orders agree; the
        // point of the assertion is that they agree to fp32 rounding and NOT further, which is
        // why the reference's order (and ours) is the one that keeps the rotation's inputs at
        // unit RMS.
        let gap = (&norm_then_rotate - &rotate_then_norm)
            .abs()
            .max()
            .double_value(&[]);
        assert!(
            gap < 1e-5,
            "rotation is not norm-preserving per head (gap {gap}), so QK-norm before RoPE is \
             not the reference's scheme"
        );
        let rotated_rms = norm_then_rotate
            .square()
            .mean_dim(-1, false, Kind::Float)
            .sqrt();
        assert!(
            (rotated_rms - 1.0).abs().max().double_value(&[]) < 1e-3,
            "the rotary inputs are not unit-RMS per head, which is the point of QK-norm"
        );
    }

    /// At initialization the whole stack must be the identity on the residual stream: every
    /// residual branch is zero (zero-init output projections, no biases) and `x0_lambda` is 0,
    /// so the only thing the eight layers do is multiply the stream by `√1.1` sixteen times -
    /// which the gainless final RMSNorm divides straight back out. `backbone` at init must
    /// therefore return the twice-normalized patch embedding and nothing else.
    ///
    /// The residual stream is bf16 (`docs/timexer_segment.md:16`), so sixteen multiplications
    /// by a bf16 `√1.1` are not exact: they round the stream by up to ~1 ULP per layer, which
    /// the final norm cannot undo because it only removes the SCALE. The test therefore pins
    /// two things - the direction to fp32 precision, and bit-exactness once the residual scale
    /// is set to 1, which isolates the rounding as the only deviation.
    ///
    /// modded-nanogpt: `train_gpt.py:1334` (post lambdas = 1), `:1338` (resid = √1.1),
    /// `:1389` (x0 lambda init 0), `:1308`/`:973-975` (zero-init branch outputs), `:1682`
    /// (final `norm`).
    #[test]
    fn the_stack_is_the_identity_on_the_residual_stream_at_init() {
        let _rng = crate::torch::test_rng::exclusive();
        let config = ModelConfig {
            layers: 8,
            ..narrow_config()
        };
        let store = nn::VarStore::new(Device::Cpu);
        let mut model = CausalPatchModel::new(&store.root(), &config);
        let batch = synthetic(&config, &[8, 5]);
        let stats = model.statistics(&batch);
        // The init values themselves, read back through the accessor the report base uses. The
        // accessor also carries the U-net gates and the value lambdas, which have their own
        // tests; this pins the three families the residual recipe owns.
        let scalars: Vec<(String, f64)> = model
            .recipe_scalars()
            .into_iter()
            .filter(|(label, _)| {
                label.starts_with("residual lambda")
                    || label.starts_with("post lambda")
                    || label.starts_with("x0 lambda")
            })
            .collect();
        assert_eq!(scalars.len(), 5 * config.layers);
        for (label, value) in &scalars {
            let reference = if label.starts_with("residual lambda") {
                RESID_LAMBDA_INIT
            } else if label.starts_with("post lambda") {
                POST_LAMBDA_INIT
            } else {
                X0_LAMBDA_INIT
            };
            // The parameters are stored fp32; the constants are f64 literals of the same value,
            // so the readback is exact only to single precision.
            assert!(
                (value - reference).abs() < 1e-7,
                "{label} = {value}, expected {reference}"
            );
        }
        assert!(scalars
            .iter()
            .any(|(label, _)| label == "residual lambda L7 ffn"));
        assert!(scalars.iter().any(|(label, _)| label == "post lambda L3 attn"));
        assert!(scalars.iter().any(|(label, _)| label == "x0 lambda L0"));
        // What the identity map returns: `x0 = rms_norm(embed)` through the final `rms_norm`.
        let x0 = rms_norm(&linear(&model.tokens(&batch, &stats), &model.patch));
        let expected = rms_norm(&x0);
        let actual = model.backbone(&batch, &stats, false, false);
        let scale = expected.abs().max().double_value(&[]);
        assert!(
            scale > 0.0,
            "the embedding carries no signal, so equality proves nothing"
        );
        // Direction: unchanged. The stack cannot have mixed anything into the stream. The bar
        // is bf16-limited, not arbitrary - the `1 - cos` a live branch produces is checked
        // against it below.
        let cosine = |left: &Tensor, right: &Tensor| {
            let (left, right) = (left.to_kind(Kind::Float), right.to_kind(Kind::Float));
            let norm = |tensor: &Tensor| tensor.square().sum(Kind::Float).double_value(&[]);
            (&left * &right).sum(Kind::Float).double_value(&[])
                / (norm(&left) * norm(&right)).sqrt()
        };
        let aligned = cosine(&expected, &actual);
        assert!(
            (1.0 - aligned).abs() < 1e-4,
            "the stack rotated the residual stream at init: cosine {aligned}"
        );
        // Magnitude: within bf16 rounding of sixteen `√1.1` multiplications (2^-8 per step).
        let gap = (&expected - &actual).abs().max().double_value(&[]) / scale;
        assert!(
            gap < 2. * (2. * config.layers as f64) * f64::powi(2., -9),
            "the stack is not the identity at init: max relative deviation {gap}"
        );
        // With the residual scale set to exactly 1 and the U-net gates closed there is nothing
        // left to round, and the identity must hold bit for bit. The gates have to be closed
        // for this line and only this line: `σ(-1.5) = 0.182` is a DELIBERATE nonzero addition
        // at init, and since every layer's output is collinear with `x0` here, a skip multiplies
        // the stream by `1.182` - a pure scale the final norm removes, but only to bf16
        // precision (measured 1.6e-2 relative, which is bf16's own ULP at this magnitude, not a
        // mixing error). The cosine and magnitude bounds above were taken with the gates OPEN,
        // so the U-net's contribution is still covered here; that it is a scale and nothing
        // more is what `closed_skip_gates_leave_the_backbone_bit_identical_to_a_plain_stack`
        // pins.
        tch::no_grad(|| {
            let _ = model.resid_lambdas.fill_(1.0);
            let _ = model.skip_weights.fill_(f64::NEG_INFINITY);
        });
        assert_eq!(
            (&expected - &model.backbone(&batch, &stats, false, false))
                .abs()
                .max()
                .double_value(&[]),
            0.0,
            "at unit residual scale the stack must be the identity exactly"
        );
        // The identity is not an accident of a dead network: the branches are live as soon as
        // one output projection is nonzero, and then the direction moves by orders of magnitude
        // more than the bf16 bar above.
        tch::no_grad(|| {
            let weight = &mut model.blocks[0].output.ws;
            let noise = Tensor::randn(weight.size(), (Kind::Float, Device::Cpu)) * 0.1;
            weight.copy_(&noise);
        });
        let moved = model.backbone(&batch, &stats, false, false);
        assert!(
            (&expected - &moved).abs().max().double_value(&[]) / scale > 1e-3,
            "a nonzero attention output projection did not change the backbone output, so the \
             identity above says nothing about the branches"
        );
        let disturbed = 1.0 - cosine(&expected, &moved);
        assert!(
            disturbed > 10. * (1.0 - aligned).abs(),
            "a live attention branch moved the direction by {disturbed}, no more than the \
             identity's own bf16 noise - the cosine bound above is vacuous"
        );
    }

    /// `--x0-lambdas disabled` must REMOVE the shortcut, not hold it at zero.
    ///
    /// Two things have to be true at once, and only together do they say the mode is a real
    /// path: the parameter is gone from the varstore (so the optimizer has nothing to route,
    /// the state dict has one fewer tensor and the chart has `layers` fewer series), and the
    /// forward pass is bit-identical to the learned mode AT ITS INIT, where `λ0 = 0` makes the
    /// injection an exact no-op. If the second failed, the mode would be a different model
    /// rather than the same model without a shortcut; if the first failed, it would be a flag
    /// guarding a zero-multiplied tensor that still costs traffic and still trains.
    #[test]
    fn disabling_the_x0_injection_removes_its_parameters_and_its_kernel() {
        let _rng = crate::torch::test_rng::exclusive();
        let learned = eight_layer_config();
        let none = ModelConfig {
            x0_lambdas: X0Lambdas::Disabled,
            ..learned.clone()
        };
        // Same seed on both sides: every shared parameter draws from the same stream, and the
        // x0 bank is a constant init that consumes none of it, so the two stores differ in
        // exactly one tensor and in nothing else.
        let build = |config: &ModelConfig| {
            tch::manual_seed(20260906);
            let store = nn::VarStore::new(Device::Cpu);
            let model = CausalPatchModel::new(&store.root(), config);
            (store, model)
        };
        let (learned_store, mut learned_model) = build(&learned);
        let (none_store, none_model) = build(&none);
        assert!(
            learned_store.variables().contains_key("lambdas.x0"),
            "the learned mode must register the bank"
        );
        assert!(
            !none_store.variables().contains_key("lambdas.x0"),
            "the disabled mode registered a bank it must not have"
        );
        assert_eq!(
            none_store.variables().len() + 1,
            learned_store.variables().len(),
            "disabling the injection must drop exactly one varstore entry"
        );
        assert_eq!(
            none_store.trainable_variables().len() + 1,
            learned_store.trainable_variables().len(),
            "the dropped entry must be a TRAINED parameter, not a buffer"
        );
        // Every scalar the chart reports, per mode: `layers` fewer series and not one of them
        // an x0 label.
        let labels = |model: &CausalPatchModel| -> Vec<String> {
            model
                .recipe_scalars()
                .into_iter()
                .map(|(label, _)| label)
                .collect()
        };
        let (learned_labels, none_labels) = (labels(&learned_model), labels(&none_model));
        assert_eq!(
            none_labels.len() + learned.layers,
            learned_labels.len(),
            "one x0 series per layer must disappear"
        );
        assert!(
            !none_labels.iter().any(|label| label.starts_with("x0 lambda"))
                && learned_labels
                    .iter()
                    .filter(|label| label.starts_with("x0 lambda"))
                    .count()
                    == learned.layers,
            "{none_labels:?}"
        );
        // And the traffic bound: exactly one `addcmul` over the residual stream per layer,
        // forward and backward, is what the mode removes.
        let (with, without) = (learned.step_cost(4), none.step_cost(4));
        assert_eq!(
            with.matmul_flops, without.matmul_flops,
            "the injection is elementwise; it must not move the GEMM count"
        );
        assert!(
            with.traffic_bytes > without.traffic_bytes,
            "the removed `addcmul` must show up in the traffic bound"
        );
        let batch = synthetic(&learned, &[8, 5]);
        let stats = learned_model.statistics(&batch);
        let learned_output = learned_model.backbone(&batch, &stats, false, false);
        let none_output = none_model.backbone(&batch, &stats, false, false);
        assert!(
            learned_output.abs().max().double_value(&[]) > 0.0,
            "a zero backbone would make the comparison vacuous"
        );
        assert_eq!(
            (&learned_output - &none_output).abs().max().double_value(&[]),
            0.0,
            "at `λ0 = 0` the two modes must agree bit for bit: the disabled path has to be the \
             same model with the shortcut removed, not a different one"
        );
        // Live the shortcut, and only the learned mode can follow it.
        tch::no_grad(|| {
            let _ = learned_model
                .x0_lambdas
                .as_mut()
                .expect("the learned mode owns the bank")
                .fill_(0.25);
        });
        let injected = learned_model.backbone(&batch, &stats, false, false);
        assert!(
            (&injected - &none_output).abs().max().double_value(&[])
                / none_output.abs().max().double_value(&[])
                > 1e-3,
            "a nonzero x0 lambda did not change the learned mode's output, so the bit-identity \
             above says nothing about the injection"
        );
    }

    fn eight_layer_config() -> ModelConfig {
        ModelConfig {
            layers: 8,
            ..small_config()
        }
    }

    #[test]
    fn skip_gates_start_at_the_reference_sigmoid_of_minus_three_halves() {
        let config = eight_layer_config();
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let stored = store
            .variables()
            .remove("skip_weights")
            .expect("the skip gate logits must be a named varstore parameter");
        assert_eq!(stored.size(), [4]);
        assert!(stored.requires_grad());
        assert_eq!(
            (&stored - SKIP_LOGIT_INIT).abs().max().double_value(&[]),
            0.0,
            "the stored parameter is the LOGIT, so it must start at -1.5, not at 0.18"
        );
        // σ(-1.5), the reference's own annotation of this init.
        let expected = 1.0 / (1.0 + 1.5f64.exp());
        assert!((expected - 0.182_425_523_806_356_35).abs() < 1e-15);
        let reported: Vec<(String, f64)> = model
            .recipe_scalars()
            .into_iter()
            .filter(|(name, _)| name.starts_with("skip weight"))
            .collect();
        // The pairing order, which is also the order the gates are unbound in: the shared
        // accessor also carries the residual recipe's lambdas and the value residual's.
        assert_eq!(
            reported
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            [
                "skip weight 3->4",
                "skip weight 2->5",
                "skip weight 1->6",
                "skip weight 0->7"
            ]
        );
        for (name, value) in &reported {
            assert!(
                (value - expected).abs() < 1e-6,
                "{name} reported {value}, not the post-sigmoid init {expected}"
            );
        }
    }

    /// Closed gates must leave the backbone bit-identical to the same stack without the U-net,
    /// and open gates must not.
    ///
    /// The branch weights have to be randomized first, and that is not cosmetic: the residual
    /// recipe zero-initializes both output projections, so at init every layer's output is a
    /// scalar multiple of `x0`, a skip adds a multiple of `x0` to a multiple of `x0`, and the
    /// gainless final RMSNorm divides the scale straight back out - an OPEN gate would then be
    /// bit-identical too and this test would prove nothing about the skip. With live branches
    /// the encoder outputs are no longer collinear with the stream and the gate is observable.
    #[test]
    fn closed_skip_gates_leave_the_backbone_bit_identical_to_a_plain_stack() {
        let config = eight_layer_config();
        let store = nn::VarStore::new(Device::Cpu);
        let mut model = CausalPatchModel::new(&store.root(), &config);
        tch::no_grad(|| {
            for block in &mut model.blocks {
                for weight in [&mut block.output.ws, &mut block.second.ws] {
                    let noise =
                        Tensor::randn(weight.size(), (weight.kind(), weight.device())) * 0.1;
                    weight.copy_(&noise);
                }
            }
        });
        let batch = synthetic(&config, &[8, 8, 8]);
        let stats = model.statistics(&batch);
        let opened = model.backbone(&batch, &stats, false, false);
        // σ(-∞) = 0 exactly: the only way to close a sigmoid gate, and the only setting under
        // which the U-net stack is required to vanish.
        tch::no_grad(|| {
            let _ = model.skip_weights.fill_(f64::NEG_INFINITY);
        });
        let actual = model.backbone(&batch, &stats, false, false);
        // The plain stack, spelled out: the same per-layer compute `backbone` runs, with the
        // U-net fold and nothing else removed. Written as a loop over `Block::forward` rather
        // than re-deriving the block math, so this test stays about the SKIP - the residual
        // lambdas, the x0 injection and the value residual all still have to be threaded
        // exactly as the backbone threads them, or the comparison fails for the wrong reason.
        let x0 = rms_norm(&linear(&model.tokens(&batch, &stats), &model.patch));
        let kind = x0.kind();
        let resid = model.resid_lambdas.to_kind(kind).unbind(0);
        let post = model.post_lambdas.to_kind(kind).unbind(0);
        let x0_lambdas = model
            .x0_lambdas
            .as_ref()
            .expect("this model runs the learned injection")
            .to_kind(kind)
            .unbind(0);
        let mut state = x0.shallow_clone();
        let mut first_value: Option<Tensor> = None;
        for (index, block) in model.blocks.iter().enumerate() {
            let lambdas = BlockLambdas {
                resid: [&resid[2 * index], &resid[2 * index + 1]],
                post: [&post[2 * index], &post[2 * index + 1]],
                x0: Some((&x0, &x0_lambdas[index])),
            };
            let (next, published) = block.forward(
                &state,
                first_value.as_ref(),
                &lambdas,
                (&model.rotation.0, &model.rotation.1),
                false,
            );
            state = next;
            if let Some(value) = published {
                first_value = Some(value);
            }
        }
        let expected = rms_norm(&state);
        assert_eq!(actual.size(), expected.size());
        assert!(
            expected.abs().max().double_value(&[]) > 0.0,
            "a zero backbone output would make this comparison vacuous"
        );
        assert!(
            actual.equal(&expected),
            "closed skip gates must not perturb a single bit of the backbone"
        );
        assert!(
            !opened.equal(&expected),
            "the skip gates at their σ(-1.5) init did not move the backbone, so closing them \
             proves nothing"
        );
    }

    #[test]
    fn the_skip_stack_pairs_the_deepest_encoder_layer_with_the_shallowest_decoder_layer() {
        let layers = 8usize;
        let options = (Kind::Double, Device::Cpu);
        let width = layers as i64;
        // Layer `i` adds a one-hot tag, so channel `i` of the result is exactly the coefficient
        // with which layer `i`'s output reached the output - which is what a pairing IS. The
        // gates are distinct primes so no two pairings can coincide by commutativity.
        let tag = |index: usize| {
            let value = Tensor::zeros([width], options);
            let _ = value.narrow(0, index as i64, 1).fill_(1.0);
            value
        };
        let gates: Vec<Tensor> = [2.0, 3.0, 5.0, 7.0]
            .into_iter()
            .map(|gate| Tensor::scalar_tensor(gate, options))
            .collect();
        let actual = unet_stack(
            Tensor::zeros([width], options),
            layers,
            &gates,
            |index, state| state + tag(index),
        );

        // The four encoder outputs, straight-line.
        let encoder_0 = Tensor::zeros([width], options) + tag(0);
        let encoder_1 = &encoder_0 + tag(1);
        let encoder_2 = &encoder_1 + tag(2);
        let encoder_3 = &encoder_2 + tag(3);
        // Straight-line, no loop: the pairing the reference specifies, one line per skip.
        let layer_4 = encoder_3.addcmul(&encoder_3, &gates[0]) + tag(4); // 3 -> 4
        let layer_5 = layer_4.addcmul(&encoder_2, &gates[1]) + tag(5); // 2 -> 5
        let layer_6 = layer_5.addcmul(&encoder_1, &gates[2]) + tag(6); // 1 -> 6
        let expected = layer_6.addcmul(&encoder_0, &gates[3]) + tag(7); // 0 -> 7
        assert!(
            actual.equal(&expected),
            "expected 3->4, 2->5, 1->6, 0->7; got {actual:?} against {expected:?}"
        );

        // Negative control: the queue order a `remove(0)` instead of a `pop` would produce. If
        // this matched, the assertion above would be proving nothing.
        let queue_4 = encoder_3.addcmul(&encoder_0, &gates[0]) + tag(4); // 0 -> 4
        let queue_5 = queue_4.addcmul(&encoder_1, &gates[1]) + tag(5); // 1 -> 5
        let queue_6 = queue_5.addcmul(&encoder_2, &gates[2]) + tag(6); // 2 -> 6
        let queue_7 = queue_6.addcmul(&encoder_3, &gates[3]) + tag(7); // 3 -> 7
        assert!(
            !actual.equal(&queue_7),
            "the tags and gates cannot distinguish stack order from queue order"
        );
    }

    #[test]
    fn an_odd_layer_count_cannot_split_into_encoder_and_decoder_halves() {
        let config = ModelConfig {
            layers: 7,
            ..small_config()
        };
        let error = config.validate().expect_err("7 layers must be rejected");
        assert!(error.to_string().contains("even"), "{error}");
    }

    /// The mixing weight is modded-nanogpt's raw per-layer scalar at 0.5, it exists on every
    /// layer EXCEPT the source, and the optimizer sees it as a 1-D non-hidden parameter - which
    /// is what keeps it on AdamW instead of NorMuon.
    #[test]
    fn the_value_lambda_is_a_raw_half_on_every_layer_but_the_source() {
        let store = nn::VarStore::new(Device::Cpu);
        let config = ModelConfig {
            layers: 4,
            ..small_config()
        };
        let model = CausalPatchModel::new(&store.root(), &config);
        assert!(
            model.blocks[0].value_lambda.is_none(),
            "the source layer must not carry a dead mixing weight"
        );
        let named: Vec<_> = store
            .variables()
            .into_iter()
            .filter(|(name, _)| name.contains("value_lambda"))
            .collect();
        let mut names: Vec<_> = named.iter().map(|(name, _)| name.clone()).collect();
        names.sort();
        assert_eq!(
            names,
            [
                "block_1.value_lambda",
                "block_2.value_lambda",
                "block_3.value_lambda"
            ]
        );
        for (name, lambda) in &named {
            assert_eq!(lambda.size(), [1], "{name} must stay 1-D");
            assert_eq!(lambda.kind(), Kind::Float, "{name} must be an fp32 master");
            assert!(
                lambda.requires_grad(),
                "{name} must be trainable, not a buffer"
            );
            // The optimizer routes `block_*` 2-D `.weight` tensors to NorMuon and everything
            // else to AdamW (`compute.rs::polar_express`), so a 1-D lambda not named `.weight`
            // is on AdamW by construction.
            assert!(!(name.ends_with(".weight") && lambda.dim() == 2));
            assert_eq!(lambda.double_value(&[0]), VALUE_LAMBDA_INIT);
        }
        // The shared accessor also carries the residual recipe's and the U-net's scalars; the
        // value residual's contribution is exactly one raw entry per non-source layer.
        assert_eq!(
            model
                .recipe_scalars()
                .into_iter()
                .filter(|(name, _)| name.starts_with("value lambda"))
                .collect::<Vec<_>>(),
            vec![
                ("value lambda L1".to_string(), 0.5),
                ("value lambda L2".to_string(), 0.5),
                ("value lambda L3".to_string(), 0.5),
            ]
        );
    }

    /// A decoder layer at λ = 0 must reproduce the model WITHOUT a value residual bit for bit -
    /// output and every gradient - and at λ = 1 must attend with the source layer's value
    /// instead of its own, also bit for bit. Both endpoints are exact because `lerp` selects
    /// `self + λ·(end - self)` below |λ| = 0.5 and `end - (end - self)·(1 - λ)` above it.
    #[test]
    fn the_value_mix_is_exactly_the_own_value_at_zero_and_the_source_value_at_one() {
        let _rng = crate::torch::test_rng::exclusive();
        let config = narrow_config();
        let store = nn::VarStore::new(Device::Cpu);
        let mut block = Block::new(store.root() / "block", &config, 1);
        // The recipe zero-initializes both branch output projections, so an untouched block
        // returns its input whatever V it attended with and BOTH endpoints would agree at 0.
        // Live branches are what makes the value the output depends on.
        tch::no_grad(|| {
            for weight in [&mut block.output.ws, &mut block.second.ws] {
                let noise = Tensor::randn(weight.size(), (Kind::Float, Device::Cpu)) * 0.05;
                weight.copy_(&noise);
            }
        });
        let mut lambda = block
            .value_lambda
            .as_ref()
            .expect("a decoder layer carries a lambda")
            .shallow_clone();
        let (width, heads) = (config.d_model, config.heads);
        let head_dim = width / heads;
        let (rows, length) = (2, config.origins());
        let rope = RotaryEmbedding::new(length, head_dim, head_dim, Device::Cpu);
        let (cosine, sine) = rope.cached_rotation(
            &Tensor::arange(length, (Kind::Int64, Device::Cpu)),
            Kind::BFloat16,
        );
        let input = (Tensor::randn([rows, length, width], (Kind::Float, Device::Cpu)) * 0.5)
            .to_kind(Kind::BFloat16)
            .set_requires_grad(true);
        let first = (Tensor::randn(
            [rows, length, heads, head_dim],
            (Kind::Float, Device::Cpu),
        ) * 0.5)
            .to_kind(Kind::BFloat16);
        let x0 = (Tensor::randn([rows, length, width], (Kind::Float, Device::Cpu)) * 0.5)
            .to_kind(Kind::BFloat16);
        let (resid_parts, post_parts) = (
            Tensor::from_slice(&[RESID_LAMBDA_INIT, 0.9])
                .to_kind(Kind::BFloat16)
                .unbind(0),
            Tensor::from_slice(&[POST_LAMBDA_INIT, 1.1])
                .to_kind(Kind::BFloat16)
                .unbind(0),
        );
        let x0_scale = Tensor::from_slice(&[0.25]).to_kind(Kind::BFloat16).get(0);
        let block_lambdas = BlockLambdas {
            resid: [&resid_parts[0], &resid_parts[1]],
            post: [&post_parts[0], &post_parts[1]],
            x0: Some((&x0, &x0_scale)),
        };
        // The model WITHOUT the residual: the same block math `Block::forward` runs - gainless
        // pre-norm, QK-norm before the rotation, ReLU² feedforward, folded post-lambdas and
        // `addcmul` residual lines - reading either its own value or a replacement standing in
        // for it. Only the V operand differs from the production path.
        let unmixed = |replacement: Option<&Tensor>| {
            let projection = linear(&rms_norm(&input), &block.qkv);
            let packed = projection.split_with_sizes([2 * width, width], -1);
            let normed = rms_norm(&packed[0].reshape([rows, length, 2 * heads, head_dim]))
                .reshape([rows, length, 2 * width]);
            let rotated = fused_rope(&normed, &cosine, &sine, heads).split(1, 2);
            let query_key = |tensor: &Tensor| tensor.squeeze_dim(2).transpose(1, 2);
            let value = replacement
                .map(Tensor::shallow_clone)
                .unwrap_or_else(|| packed[1].reshape([rows, length, heads, head_dim]));
            let attended = Tensor::scaled_dot_product_attention(
                &query_key(&rotated[0]),
                &query_key(&rotated[1]),
                &value.transpose(1, 2),
                None::<&Tensor>,
                0.0,
                true,
                None,
                false,
            )
            .transpose(1, 2)
            .reshape([rows, length, width]);
            let state = scaled_linear(&attended, &block.output, block_lambdas.post[0])
                .addcmul(&input, block_lambdas.resid[0])
                .addcmul(&x0, &x0_scale);
            scaled_linear(
                &linear(&rms_norm(&state), &block.first).relu().square(),
                &block.second,
                block_lambdas.post[1],
            )
            .addcmul(&state, block_lambdas.resid[1])
        };
        let own_value = {
            let projection = linear(&rms_norm(&input), &block.qkv);
            projection
                .split_with_sizes([2 * width, width], -1)[1]
                .reshape([rows, length, heads, head_dim])
        };
        assert!(
            (&own_value - &first).abs().max().double_value(&[]) > 0.0,
            "the two values coincide, so neither endpoint proves anything"
        );
        assert!(
            (&unmixed(None) - &unmixed(Some(&first)))
                .abs()
                .max()
                .double_value(&[])
                > 0.0,
            "the block output does not depend on V at all, so the endpoints are vacuous"
        );
        for (name, weight, expected) in [
            ("zero", 0.0, unmixed(None)),
            ("one", 1.0, unmixed(Some(&first))),
        ] {
            tch::no_grad(|| {
                let _ = lambda.fill_(weight);
            });
            let (actual, published) = block.forward(
                &input,
                Some(&first),
                &block_lambdas,
                (&cosine, &sine),
                false,
            );
            assert!(
                published.is_none(),
                "only the source layer may publish a value"
            );
            assert!(expected.abs().max().double_value(&[]) > 0.0);
            assert_eq!(
                (&expected - &actual).abs().max().double_value(&[]),
                0.0,
                "lambda = {name} is not the exact endpoint"
            );
            let mut targets = vec![input.shallow_clone()];
            targets.extend(
                store
                    .trainable_variables()
                    .into_iter()
                    .filter(|tensor| tensor.dim() == 2),
            );
            let expected_grads = Tensor::run_backward(&[&expected], &targets, true, false);
            let actual_grads = Tensor::run_backward(&[&actual], &targets, true, false);
            for (index, (left, right)) in expected_grads.iter().zip(&actual_grads).enumerate() {
                assert!(
                    left.abs().max().double_value(&[]) > 0.0,
                    "gradient {index} carries no signal"
                );
                assert_eq!(
                    (left - right).abs().max().double_value(&[]),
                    0.0,
                    "gradient {index} moved at lambda = {name}"
                );
            }
        }
        // The mix must still TEACH the lambda at λ = 0, or the residual could never turn on:
        // ∂/∂λ = Σ g·(v_1 - v) is nonzero even where the forward is the identity.
        tch::no_grad(|| {
            let _ = lambda.fill_(0.0);
        });
        let (output, _) = block.forward(
            &input,
            Some(&first),
            &block_lambdas,
            (&cosine, &sine),
            false,
        );
        let grad = Tensor::run_backward(&[&output], &[lambda.shallow_clone()], false, false);
        assert!(
            grad[0].abs().double_value(&[0]) > 0.0,
            "lambda is unlearnable at its identity point"
        );
    }

    /// End to end: with λ = 1 every decoder layer must be attending with LAYER 0's value, so
    /// destroying a decoder layer's own value projection cannot move the output - while
    /// destroying layer 0's must. At λ = 0 the dependence is exactly the other way round.
    #[test]
    fn every_decoder_layer_attends_with_the_source_layers_value_at_lambda_one() {
        let _rng = crate::torch::test_rng::exclusive();
        // FOUR layers, not three: the U-net needs an even count, and four gives one encoder
        // pair plus two decoder layers that mix.
        let config = ModelConfig {
            layers: 4,
            ..small_config()
        };
        let store = nn::VarStore::new(Device::Cpu);
        let mut model = CausalPatchModel::new(&store.root(), &config);
        // Zero-init branch projections would make the backbone independent of V entirely, so
        // "breaking a value projection does not move the output" would hold for the wrong
        // reason. Live branches make V observable.
        tch::no_grad(|| {
            for block in &mut model.blocks {
                for weight in [&mut block.output.ws, &mut block.second.ws] {
                    let noise =
                        Tensor::randn(weight.size(), (weight.kind(), weight.device())) * 0.05;
                    weight.copy_(&noise);
                }
            }
        });
        let batch = synthetic(&config, &[8, 5]);
        let stats = model.statistics(&batch);
        let width = config.d_model;
        let set_lambdas = |weight: f64| {
            tch::no_grad(|| {
                for block in &model.blocks {
                    if let Some(lambda) = &block.value_lambda {
                        let _ = lambda.shallow_clone().fill_(weight);
                    }
                }
            })
        };
        // The value rows of a layer's packed `q‖k‖v` projection. The recipe leaves the four
        // block projections bias-free, so the weight rows are the whole of the V path.
        let break_value = |layer: usize| {
            tch::no_grad(|| {
                let qkv = &model.blocks[layer].qkv;
                assert!(qkv.bs.is_none(), "the block projections are bias-free");
                let _ = qkv.ws.narrow(0, 2 * width, width).fill_(0.0);
            })
        };
        let output = || {
            model
                .backbone(&batch, &stats, false, false)
                .to_kind(Kind::Float)
        };
        let moved = |before: &Tensor, after: &Tensor| {
            (before - after).abs().max().double_value(&[]) > 0.0
        };
        set_lambdas(1.0);
        let reference = output();
        assert!(reference.abs().max().double_value(&[]) > 0.0);
        break_value(1);
        assert!(
            !moved(&reference, &output()),
            "layer 1 still reads its own value at lambda = 1"
        );
        break_value(2);
        assert!(
            !moved(&reference, &output()),
            "layer 2 still reads its own value at lambda = 1"
        );
        // The same broken decoder projections must matter again at lambda = 0, which they only
        // can if the mix - not the plumbing - is what silenced them above. Checked BEFORE layer
        // 0 is broken: with every value projection dead the two settings agree trivially.
        set_lambdas(0.0);
        assert!(
            moved(&reference, &output()),
            "lambda = 0 did not restore each layer's own value"
        );
        set_lambdas(1.0);
        assert!(
            !moved(&reference, &output()),
            "the stack did not return to the source layer's value"
        );
        // Layer 0 is the source, so at lambda = 1 the whole stack depends on it alone.
        break_value(0);
        assert!(
            moved(&reference, &output()),
            "the stack ignored the source layer's value"
        );
    }
}
