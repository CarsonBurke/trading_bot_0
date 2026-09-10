//! A rank-preserving orthonormal reparametrization of the forecast horizon axis.
//!
//! # What is wrong with the objective's horizon geometry
//!
//! The 192 targets are cumulative sums of ONE five-minute market-neutral return series, so they
//! are near-duplicates: `corr(y_h, y_{h+1}) = √(h/(h+1))` under the persistence prior, which is
//! `0.9974` at h = 191. The count-normalized Gaussian NLL gives each of them an equal gradient
//! share, so roughly 160 of the 192 shares are spent on rows that are almost the same row. The
//! `1/h` precision prior that already rides in `half_log_horizon` is VARIANCE normalization -
//! it makes the rows unit-scale - not predictive-SNR weighting, and unit scale is not equal
//! information.
//!
//! # Why this is not `--horizon-mean basis:8:8`
//!
//! That knob restricted the emitted mean to eight exponential columns and was refuted: the
//! correlation gain collapsed (`D` 0.00481 -> 0.00116) while the mis-scaling penalty barely
//! moved (`C` -0.0385 -> -0.0373). It was a RANK restriction on the predicted function - an
//! amplitude/shape prior - and low rank is not an amplitude prior.
//!
//! This restricts nothing. `W` here is a full-rank orthonormal map on the horizon axis, hence a
//! bijection: every forecast function the head could emit before it can emit after, at the same
//! parameter count and the same head shape. What changes is only the METRIC the loss measures
//! error in, which is exactly the thing the evidence says is mis-specified.
//!
//! # Where the scale lives, and why that is forced
//!
//! An orthonormal rotation of a DIAGONAL covariance is not diagonal. Rotating the first moment
//! while keeping a per-horizon diagonal `σ` would leave the likelihood inconsistent with the
//! geometry of its own residual, and the restriction the rotation removed from the mean would
//! reappear in the second moment. So the head's four log-scale rows are REINTERPRETED as
//! per-COEFFICIENT log scales - the same 192 rows, the same head, no resize - and the NLL is
//! diagonal in the space it is actually modelled in.
//!
//! The consequence is free and was previously priced as its own project: a diagonal scale in an
//! orthonormal coefficient basis IS a structured full covariance over horizons,
//! `Σ = Wᵀ diag(τ²) W`. Horizon-space `σ_h` therefore becomes DERIVED rather than emitted,
//! `σ_h² = Σ_k W_kh²·τ_k²` ([`BasisTransform::horizon_log_scale`]), and every existing
//! per-horizon report reads the derived quantity through the one place that materializes it.
//!
//! # 192 is not a power of two
//!
//! `192 = 3·2^6`. The Haar cascade halves 192 -> 96 -> 48 -> 24 -> 12 -> 6 -> 3 and then meets
//! an odd length. It is NOT padded and NOT projected: at length three the cascade applies an
//! explicit orthonormal 3x3 completion (`[1,1,1]/√3`, `[1,0,-1]/√2`, `[1,-2,1]/√6`), which is a
//! rotation of the same three scaling coefficients rather than an approximation of them. A
//! composition of orthonormal maps is orthonormal, so the shipped 192x192 matrix is orthonormal
//! by construction and [`orthonormality_defect`] measures it in fp32 at the shipped shape.
//! DCT-II has no length constraint at all and is the other shipped option.
//!
//! # Cost
//!
//! One dense `[tokens·CHANNELS, pred_len] × [pred_len, pred_len]` fp32 GEMM per rotation:
//! 96,000 origins x 4 channels x 192 x 192 x 2 = 28.3 GFLOP forward against the step's
//! 16.98 TFLOP, i.e. 0.167%. A fast transform is NOT worth it: Haar is O(n) and DCT-II
//! O(n log n), so both would delete arithmetic that is already free while replacing ONE
//! read-write pair over the 295 MB coefficient space with `log2(192) ≈ 7.6` of them. The step
//! is bandwidth-bound at 90-96% of the streaming roof, so the dense GEMM is strictly cheaper in
//! the currency that is actually scarce.

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{fmt, fs, path::Path, str::FromStr};
use tch::{Device, Kind, Tensor};

use super::{corpus::Corpus, model::CausalPatchModel};

/// The statistics artifact's format stamp. It names the basis family, the space the whitening
/// scales were measured in and the partition they were measured on, because all three change
/// what a stored vector MEANS: a vector fitted in the DCT basis is not interchangeable with one
/// fitted in the Haar basis, and one fitted on validation origins is leakage rather than a
/// statistic.
pub const BASIS_STATISTICS_FORMAT: &str =
    "timexer-segment-target-basis-v1-orthonormal-horizon-rotation-training-whitened";

/// How far `WᵀW` may sit from the identity, in fp32, at the shipped shape, before construction
/// refuses. Measured rather than hoped: [`orthonormality_defect`] accumulates the Gram matrix in
/// fp32 exactly as the GEMM will, and the shipped 192-point maps come in an order of magnitude
/// under this. `192·2^-24 ≈ 1.1e-5` is the worst plain error bound for a 192-term fp32 dot
/// product of unit-norm rows, so this threshold is that bound with a factor of two of headroom -
/// tight enough that a genuinely non-orthonormal matrix (a padded Haar, a forgotten `√2`)
/// cannot pass, loose enough that no correct one fails.
pub const ORTHONORMALITY_TOLERANCE: f64 = 2.5e-5;

/// Which orthonormal map the objective measures error in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TargetBasis {
    /// The identity. Today's objective: the coefficients ARE the 192 cumulative horizons. The
    /// control arm, and the one mode for which the model takes the fused loss path untouched.
    #[default]
    Cumulative,
    /// Dyadic Haar over the horizon axis with an orthonormal 3-point completion at the
    /// non-dyadic tail. Localized in horizon: coefficient 0 is the whole-window mean, the
    /// finest details are single-bar differences. The natural basis if the predictable
    /// structure is a small number of horizon SEGMENTS.
    Haar,
    /// Orthonormal DCT-II over the horizon axis. Localized in frequency, no length constraint.
    /// The natural basis if the predictable structure is smooth in the horizon.
    Dct,
}

impl TargetBasis {
    /// Whether this basis leaves the objective exactly where it is. The one mode that must be
    /// bit-for-bit the pre-knob loss, which it achieves by not entering any of this code.
    pub fn is_identity(self) -> bool {
        matches!(self, Self::Cumulative)
    }

    /// Row-major `[pred_len, pred_len]` fp64: row `k` is coefficient `k`'s weights over the
    /// horizon axis, `z_k = Σ_h W_kh·y_h`. Row 0 is the lowest-frequency coefficient in every
    /// basis, so a low-index concentration means the same thing in all three.
    pub fn matrix(self, pred_len: i64) -> Result<Vec<f64>> {
        ensure!(pred_len > 0, "a horizon basis needs a positive pred_len");
        let n = pred_len as usize;
        match self {
            Self::Cumulative => {
                let mut rows = vec![0.0; n * n];
                for k in 0..n {
                    rows[k * n + k] = 1.0;
                }
                Ok(rows)
            }
            Self::Haar => haar_matrix(n),
            Self::Dct => Ok(dct_matrix(n)),
        }
    }
}

impl fmt::Display for TargetBasis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Cumulative => "cumulative",
            Self::Haar => "haar",
            Self::Dct => "dct",
        })
    }
}

impl FromStr for TargetBasis {
    type Err = anyhow::Error;
    fn from_str(spec: &str) -> Result<Self> {
        match spec {
            "cumulative" => Ok(Self::Cumulative),
            "haar" => Ok(Self::Haar),
            "dct" => Ok(Self::Dct),
            _ => Err(anyhow::anyhow!(
                "unknown target basis {spec:?}; expected cumulative, haar or dct"
            )),
        }
    }
}

impl Serialize for TargetBasis {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for TargetBasis {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// How the objective weights the COEFFICIENT axis, mean 1 over all `pred_len` coefficients so
/// the loss stays a weighted mean of per-element NLL - nats per bar, comparable to the
/// 2.3947663 persistence anchor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BasisWeight {
    /// Equal weight per coefficient. Under whitening each coefficient is unit scale, so this is
    /// equal weight per unit of TARGET variance rather than per near-duplicate row.
    #[default]
    Uniform,
    /// Weight by measured predictive SNR, `ρ̂_k²` on the [70%,80%) calibration partition,
    /// floored at that estimate's own null level `1/n_k` so that no coefficient is ever fully
    /// untrained: a zero weight is `--horizon-loss cutoff` in disguise, and cutoff is rejected.
    Snr,
}

impl fmt::Display for BasisWeight {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Uniform => "uniform",
            Self::Snr => "snr",
        })
    }
}

impl FromStr for BasisWeight {
    type Err = anyhow::Error;
    fn from_str(spec: &str) -> Result<Self> {
        match spec {
            "uniform" => Ok(Self::Uniform),
            "snr" => Ok(Self::Snr),
            _ => Err(anyhow::anyhow!(
                "unknown basis weight {spec:?}; expected uniform or snr"
            )),
        }
    }
}

impl Serialize for BasisWeight {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for BasisWeight {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// Which population a data-derived vector was measured on. Fitting and scoring on the same data
/// is leakage, so the partition is not a comment: [`BasisStatistics::fit`] REFUSES anything but
/// the partition each vector is allowed to come from, and the refusal is what the leakage test
/// exercises.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FitPartition {
    /// `[0%,70%)` training origins. The only legal source of the whitening scales: they are a
    /// property of the TARGETS, so no model forward and no held-out bar is involved.
    Training,
    /// `[70%,80%)`. Never touched by checkpoint selection, which reads `validation_refs` only.
    /// The only legal source of the SNR weights, which do need a model forward.
    Calibration,
    /// `[80%,90%)`. Selection reads this. Nothing may be fitted on it.
    Validation,
    /// `[90%,100%)`. Terminal, locked.
    Test,
}

impl fmt::Display for FitPartition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Training => "training",
            Self::Calibration => "held-out sample calibration [70%,80%)",
            Self::Validation => "held-out validation",
            Self::Test => "held-out test",
        })
    }
}

/// Per-coefficient fp64 moment accumulator, fed batch by batch.
///
/// Two populations in one pass, deliberately different in what they pool over:
///
/// - the WHITENING moments pool over all four candle channels, because the scale prior they
///   replace (`half_log_horizon`) is a `[1, 1, 1, pred_len]` buffer broadcast over channels and
///   a per-channel replacement would silently change what the tanh cap is centred on;
/// - the `ρ`/`β` moments are CLOSE-channel only, matching
///   [`super::calibration`]'s close-anchor convention, because the measured amplitude defect
///   (`β = 0.26594` at h = 192) is a close-channel quantity and the other three channels are
///   the close plus a bounded intrabar offset.
#[derive(Clone, Debug)]
pub struct CoefficientMoments {
    /// Coefficient count; index 0 is the lowest-frequency coefficient.
    coefficients: usize,
    /// `Σ mask` over all four channels, per coefficient.
    pooled_count: Vec<f64>,
    /// `Σ mask·z²` over all four channels, per coefficient.
    pooled_square: Vec<f64>,
    /// Close-channel `Σ mask`, `Σ z`, `Σ z²`, `Σ m`, `Σ m²`, `Σ m·z`.
    count: Vec<f64>,
    target_sum: Vec<f64>,
    target_square: Vec<f64>,
    mean_sum: Vec<f64>,
    mean_square: Vec<f64>,
    cross: Vec<f64>,
    /// `(row, origin)` pairs with at least one observed future bar, and the subset whose whole
    /// horizon window is observed. Their ratio is the price the rotation charges: a partial
    /// window has no coefficient vector, so it is dropped rather than zero-filled.
    any_windows: f64,
    complete_windows: f64,
}

impl CoefficientMoments {
    pub fn new(coefficients: usize) -> Self {
        Self {
            coefficients,
            pooled_count: vec![0.; coefficients],
            pooled_square: vec![0.; coefficients],
            any_windows: 0.,
            complete_windows: 0.,
            count: vec![0.; coefficients],
            target_sum: vec![0.; coefficients],
            target_square: vec![0.; coefficients],
            mean_sum: vec![0.; coefficients],
            mean_square: vec![0.; coefficients],
            cross: vec![0.; coefficients],
        }
    }

    pub fn coefficients(&self) -> usize {
        self.coefficients
    }

    /// The whitening population, from TARGETS alone: `targets` is
    /// `[rows, origins, CHANNELS, pred_len]` and `mask` `[rows, origins, 1, pred_len]`, exactly
    /// as [`super::model::CausalPatchModel::targets`] returns them. No head, no forward pass,
    /// hence nothing that could depend on a checkpoint.
    pub fn accumulate_targets(
        &mut self,
        transform: &BasisTransform,
        targets: &Tensor,
        mask: &Tensor,
    ) {
        let (coefficients, rows) = self.rotate_pair(transform, targets, mask);
        let square = (&coefficients * &coefficients * &rows).sum_dim_intlist(
            [0i64, 1, 2].as_slice(),
            false,
            Kind::Double,
        );
        // The completeness mask does not depend on the coefficient, so its reduction is ONE
        // number: the complete (row, origin) pairs, times the four channels they each carry.
        // Broadcasting it over the coefficient axis is what keeps the denominator a count of
        // scored elements rather than of masked bars.
        let complete = rows.sum(Kind::Double).double_value(&[]);
        add_into(&mut self.pooled_square, &square);
        for slot in &mut self.pooled_count {
            *slot += complete * f64::from(super::model::CHANNELS as i32);
        }
        self.complete_windows += complete;
        self.any_windows += mask
            .amax([-1i64].as_slice(), true)
            .sum(Kind::Double)
            .double_value(&[]);
    }

    /// The `ρ`/`β` population, close channel only. `prediction` is the DECODED conditional mean
    /// in the same σ-scaled horizon space as `targets`, i.e.
    /// [`super::model::decode_joint`]'s output, so both sides are rotated by the same map and
    /// the resulting `β_k` is directly comparable to the per-horizon `β_h` the calibration fits.
    pub fn accumulate_pair(
        &mut self,
        transform: &BasisTransform,
        targets: &Tensor,
        prediction: &Tensor,
        mask: &Tensor,
    ) {
        self.accumulate_targets(transform, targets, mask);
        let close = super::model::CHANNELS - 1;
        let target = transform.rotate(&targets.narrow(2, close, 1).to_kind(Kind::Float));
        let mean = transform.rotate(&prediction.narrow(2, close, 1).to_kind(Kind::Float));
        let rows = row_mask(mask);
        let reduce = |tensor: Tensor| {
            tensor.sum_dim_intlist([0i64, 1, 2].as_slice(), false, Kind::Double)
        };
        for slot in &mut self.count {
            *slot += rows.sum(Kind::Double).double_value(&[]);
        }
        add_into(&mut self.target_sum, &reduce(&target * &rows));
        add_into(&mut self.target_square, &reduce(&target * &target * &rows));
        add_into(&mut self.mean_sum, &reduce(&mean * &rows));
        add_into(&mut self.mean_square, &reduce(&mean * &mean * &rows));
        add_into(&mut self.cross, &reduce(&target * &mean * &rows));
    }

    /// Rotate the target block and produce the broadcastable row mask beside it.
    fn rotate_pair(
        &self,
        transform: &BasisTransform,
        targets: &Tensor,
        mask: &Tensor,
    ) -> (Tensor, Tensor) {
        (
            transform.rotate(&targets.to_kind(Kind::Float)),
            row_mask(mask),
        )
    }

    /// Share of `(row, origin)` pairs with any observed future bar whose window was COMPLETE,
    /// hence rotatable. `NaN` before anything is accumulated, never a flattering 1.0.
    pub fn complete_share(&self) -> f64 {
        match self.any_windows > 0. {
            true => self.complete_windows / self.any_windows,
            false => f64::NAN,
        }
    }

    /// `rms(z_k)` pooled over channels: the per-coefficient whitening scale. `NaN` where no bar
    /// was observed, never a silent 1.0 - an unmeasured coefficient must render unmeasured.
    pub fn whitening(&self) -> Vec<f64> {
        self.pooled_count
            .iter()
            .zip(&self.pooled_square)
            .map(|(count, square)| {
                if *count > 0. && *square > 0. {
                    (square / count).sqrt()
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// Close-channel `ρ̂_k`, the within-population Pearson correlation of coefficient `k`'s
    /// forecast and target. `NaN` where either side is constant.
    pub fn correlation(&self) -> Vec<f64> {
        (0..self.coefficients)
            .map(|k| {
                let n = self.count[k];
                if n < 2. {
                    return f64::NAN;
                }
                let cov = self.cross[k] / n - (self.target_sum[k] / n) * (self.mean_sum[k] / n);
                let var_y = self.target_square[k] / n - (self.target_sum[k] / n).powi(2);
                let var_f = self.mean_square[k] / n - (self.mean_sum[k] / n).powi(2);
                if var_y > 0. && var_f > 0. {
                    cov / (var_y * var_f).sqrt()
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// Close-channel `β̂_k = Cov(f_k, y_k)/Var(f_k)`, the least-squares amplitude gain on the
    /// demeaned forecast coefficient. `1.0` is correct amplitude; below 1 is over-amplification.
    /// In an orthonormal basis this is the DIAGONAL of the amplitude defect, which is the whole
    /// reason to look at it here.
    pub fn amplitude_gain(&self) -> Vec<f64> {
        (0..self.coefficients)
            .map(|k| {
                let n = self.count[k];
                if n < 2. {
                    return f64::NAN;
                }
                let cov = self.cross[k] / n - (self.target_sum[k] / n) * (self.mean_sum[k] / n);
                let var_f = self.mean_square[k] / n - (self.mean_sum[k] / n).powi(2);
                if var_f > 0. {
                    cov / var_f
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// Close-channel bars behind each coefficient's `ρ̂` and `β̂`.
    pub fn bars(&self) -> Vec<f64> {
        self.count.clone()
    }
}

/// `Σ mask` reduced over the horizon axis is NOT the right mask for a rotation. A coefficient is
/// a weighted sum over the WHOLE horizon window, so a row whose window is only partly observed
/// has no defined coefficient vector at all: zero-filling the unobserved tail would inject a
/// false zero return into every one of the 192 coefficients. The corpus mask is a prefix mask
/// (`validation-disjoint-complete-targets`; only the training remainder is partial), so the
/// minimum over the horizon axis is exactly the indicator that the row's window is complete.
fn row_mask(mask: &Tensor) -> Tensor {
    mask.amin([-1i64].as_slice(), true).to_kind(Kind::Float)
}

fn add_into(destination: &mut [f64], values: &Tensor) {
    let host = Vec::<f64>::try_from(values.reshape([-1]))
        .expect("a coefficient reduction is a dense fp64 vector");
    assert_eq!(
        host.len(),
        destination.len(),
        "a coefficient reduction must have one entry per coefficient"
    );
    for (slot, value) in destination.iter_mut().zip(host) {
        *slot += value;
    }
}

/// Frozen, authenticated, data-derived per-coefficient vectors.
///
/// # Where these live and how they are authenticated
///
/// One JSON artifact beside the run's weights, named in the manifest by its `sha256` so a
/// checkpoint and a statistics file cannot be mispaired: [`Self::authenticate`] refuses on
/// format, on basis, on `pred_len` and on digest, and names which one failed rather than
/// reporting a generic mismatch. The vectors are NOT varstore variables and NOT trained - they
/// are frozen before step 1 and never move, which is what makes scoring deterministic and what
/// makes them safe inside a captured CUDA graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BasisStatistics {
    pub format: String,
    /// The basis these were measured in. A DCT vector is meaningless in the Haar basis.
    pub basis: TargetBasis,
    pub pred_len: i64,
    /// Where the whitening scales came from. MUST be [`FitPartition::Training`].
    pub whitening_partition: FitPartition,
    /// Origin-horizon-channel triples behind the whitening scales.
    pub whitening_bars: f64,
    /// Share of training `(row, origin)` pairs whose horizon window was complete, hence
    /// rotatable. This is the objective's DROPPED share under a non-identity basis and it is
    /// stored rather than recomputed so a report cannot state a different number than the fit
    /// measured.
    pub whitening_complete_share: f64,
    /// `rms(z_k)` per coefficient on training origins, pooled over the four candle channels.
    pub whitening: Vec<f64>,
    /// Where the SNR weights came from. MUST be [`FitPartition::Calibration`]. `None` when the
    /// run is not weighting by SNR, so a uniform-weight artifact does not have to invent one.
    pub snr_partition: Option<FitPartition>,
    /// Close-channel `ρ̂_k` on the calibration partition, retained raw so the report can show
    /// what was measured beside what was frozen.
    pub correlation: Vec<f64>,
    /// Close-channel `β̂_k` on the same population: the amplitude defect, diagonalized.
    pub amplitude_gain: Vec<f64>,
    /// Bars behind each `ρ̂_k`, which sets its null level `1/n_k`.
    pub bars: Vec<f64>,
    pub sha256: String,
}

impl BasisStatistics {
    /// Fit from moments, refusing any partition a vector is not allowed to come from.
    ///
    /// `whitening` must be measured on training origins and `snr` - when present - on the
    /// [70%,80%) calibration partition. Both refusals are hard: passing
    /// [`FitPartition::Validation`] is the leakage the test exercises.
    pub fn fit(
        basis: TargetBasis,
        pred_len: i64,
        whitening_partition: FitPartition,
        whitening: &CoefficientMoments,
        snr: Option<(FitPartition, &CoefficientMoments)>,
    ) -> Result<Self> {
        ensure!(
            whitening_partition == FitPartition::Training,
            "the whitening scales are fitted on {whitening_partition} - fitting and scoring on \
             the same data is leakage, and only training origins may set them"
        );
        ensure!(
            whitening.coefficients() == pred_len as usize,
            "the whitening moments carry {} coefficients, not the {pred_len} the basis has",
            whitening.coefficients()
        );
        let scales = whitening.whitening();
        ensure!(
            scales.iter().all(|scale| scale.is_finite() && *scale > 0.),
            "a whitening scale is not finite and positive; every coefficient must have been \
             observed on training origins before it can be whitened"
        );
        let (snr_partition, correlation, amplitude_gain, bars) = match snr {
            None => (
                None,
                vec![f64::NAN; pred_len as usize],
                vec![f64::NAN; pred_len as usize],
                vec![0.; pred_len as usize],
            ),
            Some((partition, moments)) => {
                ensure!(
                    partition == FitPartition::Calibration,
                    "the SNR weights are fitted on {partition} - only the [70%,80%) partition, \
                     which checkpoint selection never reads, may set them"
                );
                ensure!(
                    moments.coefficients() == pred_len as usize,
                    "the SNR moments carry {} coefficients, not the {pred_len} the basis has",
                    moments.coefficients()
                );
                (
                    Some(partition),
                    moments.correlation(),
                    moments.amplitude_gain(),
                    moments.bars(),
                )
            }
        };
        let mut statistics = Self {
            format: BASIS_STATISTICS_FORMAT.to_owned(),
            basis,
            pred_len,
            whitening_partition,
            whitening_bars: whitening.pooled_count.iter().sum(),
            whitening_complete_share: whitening.complete_share(),
            whitening: scales,
            snr_partition,
            correlation,
            amplitude_gain,
            bars,
            sha256: String::new(),
        };
        statistics.sha256 = statistics.digest()?;
        Ok(statistics)
    }

    /// `w_k ∝ max(ρ̂_k², 1/n_k)`, normalized to mean 1.
    ///
    /// The floor is the estimator's OWN null level: `E[ρ̂²] = 1/n` under no signal, so a
    /// coefficient whose measured SNR is at or below its noise floor is weighted by that floor
    /// rather than by zero. A zero weight would be `--horizon-loss cutoff` on the coefficient
    /// axis - it deletes gradient from a whole subspace - and cutoff is a rejected arm.
    pub fn snr_weights(&self) -> Result<Vec<f64>> {
        ensure!(
            self.snr_partition.is_some(),
            "this artifact carries no SNR measurement, so --basis-weight snr has nothing to \
             weight by; fit it on the [70%,80%) partition first"
        );
        let raw: Vec<f64> = self
            .correlation
            .iter()
            .zip(&self.bars)
            .map(|(rho, bars)| {
                let floor = if *bars >= 2. { bars.recip() } else { 1. };
                match rho.is_finite() {
                    true => (rho * rho).max(floor),
                    false => floor,
                }
            })
            .collect();
        let mean = raw.iter().sum::<f64>() / raw.len() as f64;
        ensure!(
            mean.is_finite() && mean > 0.,
            "the measured SNR weights have no positive mean, so they cannot be normalized"
        );
        Ok(raw.into_iter().map(|weight| weight / mean).collect())
    }

    /// Refuse a statistics artifact that is not this run's.
    pub fn authenticate(&self, basis: TargetBasis, pred_len: i64) -> Result<()> {
        ensure!(
            self.format == BASIS_STATISTICS_FORMAT,
            "basis statistics format is {:?}, not {BASIS_STATISTICS_FORMAT:?}",
            self.format
        );
        ensure!(
            self.basis == basis,
            "basis statistics were fitted in the {} basis and this run uses {basis}; the \
             coefficients are different quantities and the vectors are not interchangeable",
            self.basis
        );
        ensure!(
            self.pred_len == pred_len,
            "basis statistics cover {} coefficients, this run has {pred_len}",
            self.pred_len
        );
        ensure!(
            self.whitening.len() == pred_len as usize,
            "basis statistics carry {} whitening scales for {pred_len} coefficients",
            self.whitening.len()
        );
        let digest = self.digest()?;
        ensure!(
            digest == self.sha256,
            "basis statistics digest is {} but the content hashes to {digest}",
            self.sha256
        );
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading basis statistics {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("parsing basis statistics {}", path.display()))
    }

    pub fn store(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(self)?)
            .with_context(|| format!("writing basis statistics {}", path.display()))?;
        Ok(())
    }

    fn digest(&self) -> Result<String> {
        let mut copy = self.clone();
        copy.sha256.clear();
        Ok(
            ring::digest::digest(&ring::digest::SHA256, &serde_json::to_vec(&copy)?)
                .as_ref()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        )
    }
}

/// The device-resident form: the rotation, the coefficient prior, and the coefficient weights.
///
/// Every tensor here is a CONSTANT buffer, never a varstore variable, for the reason
/// `horizon_weight` is: it must not be trained, and a fixed tensor at a fixed address is what
/// makes it safe inside the captured CUDA graph.
pub struct BasisTransform {
    basis: TargetBasis,
    /// `Wᵀ`, `[pred_len, pred_len]` fp32, so `y.matmul(forward)` contracts the horizon axis and
    /// emits coefficients: entry `k` is `Σ_h y_h·W_kh`.
    forward: Tensor,
    /// `W∘W`, `[pred_len, pred_len]` fp32, so `τ².matmul(squares)` emits the DIAGONAL of
    /// `Wᵀ diag(τ²) W` - the derived horizon-space variance.
    squares: Tensor,
    /// `½·ln(prior variance of coefficient k)`, `[1, 1, 1, pred_len]` fp32: what the tanh soft
    /// cap on the log predictive scale is centred on. Under `cumulative` with no whitening this
    /// is `½·ln h` computed from the identical fp32 input by the identical two ops, so it is
    /// bit-for-bit `half_log_horizon`.
    half_log_prior: Tensor,
    /// The objective's per-coefficient weight, `[1, 1, 1, pred_len]` fp32, mean 1.
    weight: Tensor,
    weight_host: Vec<f64>,
    prior_variance: Vec<f64>,
    whitened: bool,
    /// `max |WᵀW - I|` in fp32 at the shipped shape.
    defect: f64,
}

impl BasisTransform {
    /// Build the transform. `statistics` is `None` for the analytic persistence prior and
    /// `Some` for measured whitening; when it is `Some` it is authenticated first, so a
    /// mispaired artifact is a refusal rather than a silently different objective.
    pub fn new(
        basis: TargetBasis,
        weight: BasisWeight,
        pred_len: i64,
        statistics: Option<&BasisStatistics>,
        device: Device,
    ) -> Result<Self> {
        let rows = basis.matrix(pred_len)?;
        let n = pred_len as usize;
        let defect = orthonormality_defect(&rows, n);
        ensure!(
            defect <= ORTHONORMALITY_TOLERANCE,
            "the {basis} map at pred_len {pred_len} is not orthonormal in fp32: \
             max |WᵀW - I| = {defect:e} exceeds {ORTHONORMALITY_TOLERANCE:e}. A rotation that \
             is not orthonormal is not a reparametrization - it restricts the target space, \
             which is exactly the failure --horizon-mean basis:8:8 already demonstrated"
        );
        if let Some(statistics) = statistics {
            statistics.authenticate(basis, pred_len)?;
        }
        let prior_variance = prior_variance(&rows, n);
        // The prior variance the coefficient's log scale is centred on: `diag(W·C·Wᵀ)` with
        // `C_hg = min(h, g)` the persistence covariance of the cumulative target. Under the
        // identity this is exactly `h`.
        let center: Vec<f32> = match statistics {
            None => prior_variance.iter().map(|v| *v as f32).collect(),
            Some(statistics) => statistics
                .whitening
                .iter()
                .map(|scale| (scale * scale) as f32)
                .collect(),
        };
        let weight_host = match weight {
            BasisWeight::Uniform => vec![1.0; n],
            BasisWeight::Snr => statistics
                .context(
                    "--basis-weight snr needs a statistics artifact carrying a measured ρ̂ per \
                     coefficient; --basis-stats names it",
                )?
                .snr_weights()?,
        };
        let total: f64 = weight_host.iter().sum();
        ensure!(
            (total - n as f64).abs() <= 1e-9 * n as f64,
            "the per-coefficient weight vector sums to {total}, not the {n} its mean-1 \
             normalization requires; the loss would stop being nats per bar"
        );
        let host = |values: &[f32]| {
            Tensor::from_slice(values)
                .reshape([1, 1, 1, pred_len])
                .to_device(device)
        };
        let dense = |values: &[f64], transpose: bool| {
            let mut flat = vec![0f32; n * n];
            for k in 0..n {
                for h in 0..n {
                    let value = values[k * n + h] as f32;
                    flat[if transpose { h * n + k } else { k * n + h }] = value;
                }
            }
            Tensor::from_slice(&flat)
                .reshape([pred_len, pred_len])
                .to_device(device)
        };
        let squares: Vec<f64> = rows.iter().map(|value| value * value).collect();
        Ok(Self {
            basis,
            forward: dense(&rows, true),
            squares: dense(&squares, false),
            half_log_prior: host(&center).log() * 0.5,
            weight: host(
                &weight_host
                    .iter()
                    .map(|weight| *weight as f32)
                    .collect::<Vec<f32>>(),
            ),
            weight_host,
            prior_variance,
            whitened: statistics.is_some(),
            defect,
        })
    }

    pub fn basis(&self) -> TargetBasis {
        self.basis
    }

    /// `max |WᵀW - I|` in fp32 at the shipped shape: the measured orthonormality.
    pub fn orthonormality_defect(&self) -> f64 {
        self.defect
    }

    pub fn whitened(&self) -> bool {
        self.whitened
    }

    /// `diag(W·C·Wᵀ)` per coefficient, `C_hg = min(h, g)`: what a pure persistence random walk
    /// puts into each coefficient. `h` at every horizon under `cumulative`.
    pub fn prior_variance(&self) -> &[f64] {
        &self.prior_variance
    }

    pub fn weights(&self) -> &[f64] {
        &self.weight_host
    }

    /// `[1, 1, 1, pred_len]` mean-1 per-coefficient objective weight.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// `[1, 1, 1, pred_len]` prior centre of the log predictive scale, per coefficient.
    pub fn half_log_prior(&self) -> &Tensor {
        &self.half_log_prior
    }

    /// Rotate the horizon axis of a `[.., pred_len]` fp32 tensor into coefficients. ONE GEMM;
    /// the contraction is over the last dimension, so any leading shape passes through.
    pub fn rotate(&self, tensor: &Tensor) -> Tensor {
        tensor.matmul(&self.forward)
    }

    /// The DERIVED horizon-space log predictive scale.
    ///
    /// A diagonal scale in an orthonormal coefficient basis is a structured FULL covariance over
    /// horizons, `Σ = Wᵀ diag(τ²) W`, so the per-horizon `σ_h` every existing report reads is
    /// `√Σ_hh = √(Σ_k W_kh²·τ_k²)`. It is derived, not emitted. Under `cumulative` this is the
    /// identity on `log_scale`, and the model short-circuits to exactly that so the control
    /// arm's calibration charts do not move by an ulp.
    pub fn horizon_log_scale(&self, log_scale: &Tensor) -> Tensor {
        (log_scale * 2.0).exp().matmul(&self.squares).log() * 0.5
    }
}

/// `max |WᵀW - I|` accumulated in fp32, which is the arithmetic the GEMM will actually use.
///
/// fp64 would report the mathematics; this reports the shipped map. The rows are cast to fp32
/// first and the dot products accumulate in fp32, so the number below is an upper bound on how
/// far the rotation the kernel performs is from a bijection.
pub fn orthonormality_defect(rows: &[f64], n: usize) -> f64 {
    assert_eq!(rows.len(), n * n, "a basis matrix is square");
    let narrow: Vec<f32> = rows.iter().map(|value| *value as f32).collect();
    let mut worst = 0.0f64;
    for k in 0..n {
        for l in 0..n {
            let mut sum = 0.0f32;
            for h in 0..n {
                sum += narrow[k * n + h] * narrow[l * n + h];
            }
            let expected = f32::from(u8::from(k == l));
            worst = worst.max(f64::from((sum - expected).abs()));
        }
    }
    worst
}

/// `diag(W·C·Wᵀ)` with `C_hg = min(h, g)`, the persistence covariance of a cumulative target:
/// `C = L·Lᵀ` with `L` the lower-triangular ones matrix, so the diagonal is the squared row norm
/// of `W·L`, i.e. `Σ_j (Σ_{h≥j} W_kh)²`. Suffix sums, `O(n²)`.
pub fn prior_variance(rows: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(rows.len(), n * n, "a basis matrix is square");
    (0..n)
        .map(|k| {
            let mut suffix = 0.0;
            let mut total = 0.0;
            for h in (0..n).rev() {
                suffix += rows[k * n + h];
                total += suffix * suffix;
            }
            total
        })
        .collect()
}

/// The orthonormal Haar cascade at any length ≥ 1.
///
/// Even lengths pair adjacent scaling functions into `(sum, difference)/√2`. An ODD length
/// pairs the first `m - 3` and applies the explicit orthonormal 3x3 completion to the last
/// three; that is what handles `192 = 3·2^6` without padding and without projection, and it
/// generalizes to any length rather than special-casing 192. Length 1 terminates and becomes
/// coefficient 0.
///
/// Row order: coefficient 0 is the whole-window mean, then the coarsest details, down to the
/// finest. So "low-index" means "low-frequency" in every basis this module ships.
fn haar_matrix(n: usize) -> Result<Vec<f64>> {
    let root2 = 2.0f64.sqrt();
    let mut active: Vec<Vec<f64>> = (0..n)
        .map(|h| {
            let mut row = vec![0.0; n];
            row[h] = 1.0;
            row
        })
        .collect();
    // Details, coarsest level LAST, so reversing the accumulation puts the coarsest first.
    let mut levels: Vec<Vec<Vec<f64>>> = Vec::new();
    while active.len() > 1 {
        let m = active.len();
        let pairs = if m % 2 == 0 { m / 2 } else { (m - 3) / 2 };
        let mut next = Vec::with_capacity(pairs + usize::from(m % 2 == 1));
        let mut details = Vec::with_capacity(m - pairs - usize::from(m % 2 == 1));
        for index in 0..pairs {
            let (left, right) = (&active[2 * index], &active[2 * index + 1]);
            next.push(combine(&[(left, 1.0 / root2), (right, 1.0 / root2)], n));
            details.push(combine(&[(left, 1.0 / root2), (right, -1.0 / root2)], n));
        }
        if m % 2 == 1 {
            let (a, b, c) = (&active[m - 3], &active[m - 2], &active[m - 1]);
            let third = 3.0f64.sqrt().recip();
            let sixth = 6.0f64.sqrt().recip();
            next.push(combine(&[(a, third), (b, third), (c, third)], n));
            details.push(combine(&[(a, 1.0 / root2), (c, -1.0 / root2)], n));
            details.push(combine(&[(a, sixth), (b, -2.0 * sixth), (c, sixth)], n));
        }
        ensure!(
            next.len() < m,
            "the Haar cascade did not shorten at length {m}; length {n} is not supported"
        );
        levels.push(details);
        active = next;
    }
    let mut rows = Vec::with_capacity(n * n);
    rows.extend(active.pop().unwrap_or_else(|| vec![1.0]));
    for details in levels.into_iter().rev() {
        for row in details {
            rows.extend(row);
        }
    }
    ensure!(
        rows.len() == n * n,
        "the Haar cascade produced {} rows for length {n}",
        rows.len() / n.max(1)
    );
    Ok(rows)
}

fn combine(parts: &[(&Vec<f64>, f64)], n: usize) -> Vec<f64> {
    let mut row = vec![0.0; n];
    for (vector, scale) in parts {
        for (slot, value) in row.iter_mut().zip(vector.iter()) {
            *slot += scale * value;
        }
    }
    row
}

/// Orthonormal DCT-II: `W_kh = α_k·cos(π·(h + ½)·k/n)`, `α_0 = √(1/n)`, `α_k = √(2/n)`.
fn dct_matrix(n: usize) -> Vec<f64> {
    let mut rows = vec![0.0; n * n];
    for k in 0..n {
        let alpha = match k {
            0 => (1.0 / n as f64).sqrt(),
            _ => (2.0 / n as f64).sqrt(),
        };
        for h in 0..n {
            rows[k * n + h] = alpha
                * (std::f64::consts::PI * (h as f64 + 0.5) * k as f64 / n as f64).cos();
        }
    }
    rows
}

/// Fit the artifact off the corpus, each vector from the ONE partition it is allowed to come
/// from.
///
/// The whitening scales walk TRAINING origins and read targets only - no head, no forward pass,
/// nothing that could depend on a checkpoint - so the same corpus produces the same scales for
/// every arm and every seed. The `ρ̂`/`β̂` measurement walks the [70%,80%) partition, which
/// checkpoint selection never reads, and does need a forward, so it is skipped entirely at
/// `calibration_batches = 0`. The partition tags are written HERE, beside the reference list
/// each one came from, and [`BasisStatistics::fit`] refuses any other pairing.
///
/// Both walks are strided picks over the partition's own origin order, so they span it rather
/// than sampling its first rows, and they are deterministic: no RNG, no shuffle.
pub fn fit_statistics(
    corpus: &Corpus,
    model: &CausalPatchModel,
    basis: TargetBasis,
    device: Device,
    batch_rows: usize,
    training_batches: usize,
    calibration_batches: usize,
) -> Result<BasisStatistics> {
    let pred_len = model.config().pred_len;
    ensure!(
        batch_rows > 0 && training_batches > 0,
        "the whitening fit needs at least one batch of at least one row"
    );
    // The rotation is all this needs from the transform, and the rotation does not depend on
    // the prior or the weights - so the analytic prior and uniform weights are used here even
    // when the run being prepared will use neither.
    let transform = BasisTransform::new(basis, BasisWeight::Uniform, pred_len, None, device)?;
    let mut whitening = CoefficientMoments::new(pred_len as usize);
    let mut snr = CoefficientMoments::new(pred_len as usize);
    let walk = |refs: &[super::corpus::WindowRef], batches: usize| -> Vec<Vec<_>> {
        if refs.is_empty() || batches == 0 {
            return Vec::new();
        }
        let wanted = (batches * batch_rows).min(refs.len());
        let stride = (refs.len() / wanted).max(1);
        let picked: Vec<_> = refs.iter().copied().step_by(stride).take(wanted).collect();
        picked
            .chunks(batch_rows)
            .filter(|chunk| chunk.len() == batch_rows)
            .map(<[_]>::to_vec)
            .collect()
    };
    for chunk in walk(&corpus.train_refs, training_batches) {
        let batch = corpus.batch(&chunk, device)?;
        tch::no_grad(|| {
            let stats = model.statistics(&batch);
            let (targets, mask) = model.targets(&batch, &stats, false);
            whitening.accumulate_targets(&transform, &targets, &mask);
        });
    }
    let mut measured = false;
    for chunk in walk(&corpus.calibration_refs, calibration_batches) {
        let batch = corpus.batch(&chunk, device)?;
        tch::no_grad(|| {
            let stats = model.statistics(&batch);
            let (targets, mask) = model.targets(&batch, &stats, false);
            let head = model.forward(&batch, &stats, false, false);
            let prediction = model.decode(&model.output(&head), &stats);
            snr.accumulate_pair(&transform, &targets, &prediction, &mask);
        });
        measured = true;
    }
    BasisStatistics::fit(
        basis,
        pred_len,
        FitPartition::Training,
        &whitening,
        measured.then_some((FitPartition::Calibration, &snr)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every shipped map is orthonormal in fp32 AT THE SHIPPED SHAPE, and the measured defect
    /// is recorded here rather than asserted loosely: this is the number the report states.
    #[test]
    fn shipped_maps_are_orthonormal_in_fp32_at_the_production_shape() {
        for basis in [TargetBasis::Cumulative, TargetBasis::Haar, TargetBasis::Dct] {
            let rows = basis.matrix(192).unwrap();
            let defect = orthonormality_defect(&rows, 192);
            assert!(
                defect <= ORTHONORMALITY_TOLERANCE,
                "{basis} defect {defect:e} exceeds {ORTHONORMALITY_TOLERANCE:e}"
            );
            // fp64 is the mathematics: a composition of exact rotations, so it must be an
            // order of magnitude tighter than the fp32 arithmetic that ships.
            let mut exact = 0.0f64;
            for k in 0..192usize {
                for l in 0..192usize {
                    let dot: f64 = (0..192).map(|h| rows[k * 192 + h] * rows[l * 192 + h]).sum();
                    exact = exact.max((dot - f64::from(u8::from(k == l))).abs());
                }
            }
            assert!(exact < 1e-12, "{basis} is not orthonormal in fp64: {exact:e}");
        }
    }

    /// 192 = 3·2^6, so the dyadic cascade meets an odd length. Nothing is padded: the map is
    /// square, full rank, and its row space is the whole horizon space.
    #[test]
    fn haar_handles_the_non_dyadic_length_without_padding_or_projection() {
        let rows = haar_matrix(192).unwrap();
        assert_eq!(rows.len(), 192 * 192);
        // Coefficient 0 is the whole-window mean: every entry 1/√192.
        let dc = 1.0 / (192.0f64).sqrt();
        for h in 0..192 {
            assert!((rows[h] - dc).abs() < 1e-12, "row 0 entry {h} is {}", rows[h]);
        }
        // Full rank, proven by orthonormality: `WᵀW = I` means `W` is invertible, so the map
        // is a bijection and restricts nothing. Also check the cascade at other odd tails.
        for length in [1usize, 2, 3, 5, 6, 7, 12, 96, 191, 192] {
            let rows = haar_matrix(length).unwrap();
            let defect = orthonormality_defect(&rows, length);
            assert!(defect <= ORTHONORMALITY_TOLERANCE, "length {length}: {defect:e}");
        }
    }

    /// The persistence prior in coefficient space reduces to exactly `h` under the identity,
    /// which is what makes `cumulative`'s `half_log_prior` bit-for-bit `half_log_horizon`.
    #[test]
    fn identity_prior_variance_is_the_horizon_itself() {
        let rows = TargetBasis::Cumulative.matrix(192).unwrap();
        let prior = prior_variance(&rows, 192);
        for (index, value) in prior.iter().enumerate() {
            assert_eq!(*value, (index + 1) as f64, "coefficient {index}");
        }
        // Total variance is basis-invariant: `tr(W·C·Wᵀ) = tr(C) = Σ h`.
        let total: f64 = prior.iter().sum();
        for basis in [TargetBasis::Haar, TargetBasis::Dct] {
            let rotated: f64 = prior_variance(&basis.matrix(192).unwrap(), 192).iter().sum();
            assert!(
                (rotated - total).abs() < 1e-6 * total,
                "{basis} moved the total prior variance: {rotated} vs {total}"
            );
        }
    }

    /// The Haar and DCT priors CONCENTRATE the persistence variance: the whole point is that a
    /// handful of low-frequency coefficients carry what 192 near-duplicate rows carried.
    #[test]
    fn rotation_concentrates_the_persistence_variance() {
        let total: f64 = (1..=192).map(f64::from).sum();
        for basis in [TargetBasis::Haar, TargetBasis::Dct] {
            let prior = prior_variance(&basis.matrix(192).unwrap(), 192);
            let leading: f64 = prior[..8].iter().sum();
            assert!(
                leading / total > 0.5,
                "{basis} puts only {:.3} of the persistence variance in its first eight \
                 coefficients; a basis that does not concentrate it cannot reweight it",
                leading / total
            );
        }
    }

    fn moments(coefficients: usize, scale: f64) -> CoefficientMoments {
        let mut moments = CoefficientMoments::new(coefficients);
        moments.pooled_count = vec![1000.; coefficients];
        moments.pooled_square = vec![1000. * scale * scale; coefficients];
        moments.count = vec![1000.; coefficients];
        moments.target_square = vec![1000.; coefficients];
        moments.mean_square = vec![1000. * 4.; coefficients];
        moments.cross = vec![1000. * 0.5; coefficients];
        moments
    }

    /// Fitting whitening on anything but training origins is leakage and is REFUSED, and the
    /// refusal names the partition rather than reporting a generic error.
    #[test]
    fn whitening_refuses_every_partition_but_training() {
        let good = moments(8, 2.0);
        for partition in [
            FitPartition::Calibration,
            FitPartition::Validation,
            FitPartition::Test,
        ] {
            let error = BasisStatistics::fit(TargetBasis::Dct, 8, partition, &good, None)
                .unwrap_err()
                .to_string();
            assert!(error.contains("leakage"), "{partition}: {error}");
            assert!(error.contains(&partition.to_string()), "{partition}: {error}");
        }
        let fitted =
            BasisStatistics::fit(TargetBasis::Dct, 8, FitPartition::Training, &good, None).unwrap();
        assert_eq!(fitted.whitening_partition, FitPartition::Training);
        assert!(fitted.whitening.iter().all(|scale| (scale - 2.0).abs() < 1e-12));
        // The SNR side is refused on the SELECTION partition specifically, which is the one a
        // careless fit would reach for.
        let error = BasisStatistics::fit(
            TargetBasis::Dct,
            8,
            FitPartition::Training,
            &good,
            Some((FitPartition::Validation, &good)),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("[70%,80%)"), "{error}");
    }

    /// A statistics artifact is refused unless it is THIS run's: format, basis, `pred_len` and
    /// digest each fail on their own, and each names itself.
    #[test]
    fn statistics_authenticate_on_format_basis_shape_and_digest() {
        let good = moments(8, 1.5);
        let fitted =
            BasisStatistics::fit(TargetBasis::Dct, 8, FitPartition::Training, &good, None).unwrap();
        fitted.authenticate(TargetBasis::Dct, 8).unwrap();
        let error = fitted.authenticate(TargetBasis::Haar, 8).unwrap_err().to_string();
        assert!(error.contains("not interchangeable"), "{error}");
        let error = fitted.authenticate(TargetBasis::Dct, 16).unwrap_err().to_string();
        assert!(error.contains("16"), "{error}");
        let mut tampered = fitted.clone();
        tampered.whitening[3] *= 1.01;
        let error = tampered.authenticate(TargetBasis::Dct, 8).unwrap_err().to_string();
        assert!(error.contains("digest"), "{error}");
        let mut stale = fitted.clone();
        stale.format = "timexer-segment-target-basis-v0".into();
        stale.sha256 = stale.digest().unwrap();
        let error = stale.authenticate(TargetBasis::Dct, 8).unwrap_err().to_string();
        assert!(error.contains("format"), "{error}");
    }

    /// The SNR weights are mean 1, never zero, and floored at each coefficient's own null level.
    #[test]
    fn snr_weights_are_mean_one_and_never_delete_a_coefficient() {
        let mut fitted = BasisStatistics::fit(
            TargetBasis::Dct,
            4,
            FitPartition::Training,
            &moments(4, 1.0),
            Some((FitPartition::Calibration, &moments(4, 1.0))),
        )
        .unwrap();
        // One coefficient with real signal, one at zero, one unmeasured, one negative.
        fitted.correlation = vec![0.2, 0.0, f64::NAN, -0.1];
        fitted.bars = vec![10_000., 10_000., 0., 10_000.];
        fitted.sha256 = fitted.digest().unwrap();
        let weights = fitted.snr_weights().unwrap();
        let total: f64 = weights.iter().sum();
        assert!((total - 4.0).abs() < 1e-12, "{weights:?} sums to {total}");
        assert!(weights.iter().all(|weight| *weight > 0.), "{weights:?}");
        // ρ² = .04 against a 1e-4 floor: the measured coefficient must dominate, and the sign
        // of ρ must not matter (rank order is what a correlation weight is about).
        assert!(weights[0] > 10. * weights[1], "{weights:?}");
        assert!((weights[3] / weights[0] - 0.25).abs() < 1e-9, "{weights:?}");
        // The unmeasured coefficient falls back to weight 1 before normalization, not to 0.
        assert!(weights[2] > weights[1], "{weights:?}");
        // Uniform weights need no artifact at all; SNR weights without one are refused.
        let bare =
            BasisStatistics::fit(TargetBasis::Dct, 4, FitPartition::Training, &moments(4, 1.), None)
                .unwrap();
        assert!(bare.snr_weights().unwrap_err().to_string().contains("no SNR measurement"));
    }

    /// A non-orthonormal map is refused at construction. The failure mode this guards is exactly
    /// the one that killed `basis:8:8`: a map that is not a bijection restricts the target space.
    #[test]
    fn construction_refuses_a_map_that_is_not_a_bijection() {
        let defect = orthonormality_defect(&[1.0, 0.0, 0.0, 0.5], 2);
        assert!(defect > ORTHONORMALITY_TOLERANCE, "{defect}");
        // And the real maps pass at the production shape through the constructor.
        for basis in [TargetBasis::Cumulative, TargetBasis::Haar, TargetBasis::Dct] {
            let transform =
                BasisTransform::new(basis, BasisWeight::Uniform, 192, None, Device::Cpu).unwrap();
            assert!(transform.orthonormality_defect() <= ORTHONORMALITY_TOLERANCE);
            assert_eq!(transform.weights().len(), 192);
        }
    }

    /// `cumulative` with uniform weights is the IDENTITY on the objective's inputs: the
    /// rotation is bit-exact on real data and the coefficient prior is bit-for-bit the `½·ln h`
    /// buffer the pre-knob loss used. This is the gate on the whole knob.
    #[test]
    fn cumulative_rotation_and_prior_are_bit_exact_identities() {
        let _rng = crate::torch::test_rng::shared();
        let transform =
            BasisTransform::new(TargetBasis::Cumulative, BasisWeight::Uniform, 192, None, Device::Cpu)
                .unwrap();
        let values = Tensor::randn([3, 5, 4, 192], (Kind::Float, Device::Cpu)) * 7.0;
        let rotated = transform.rotate(&values);
        assert!(
            rotated.equal(&values),
            "the identity rotation moved {} of {} elements",
            rotated.not_equal_tensor(&values).sum(Kind::Int64).int64_value(&[]),
            values.numel()
        );
        // The pre-knob prior, built exactly as `CausalPatchModel::new` builds it.
        let horizon = (Tensor::arange(192, (Kind::Float, Device::Cpu)) + 1.0).reshape([1, 1, 1, 192]);
        assert!(
            transform.half_log_prior().equal(&(horizon.log() * 0.5)),
            "the cumulative coefficient prior is not bit-for-bit ½·ln h"
        );
        // And the derived horizon scale is the emitted one, to fp32 rounding: `W = I` makes
        // `Σ = diag(τ²)`, so this is the round trip `½·ln(exp(2·ls))`.
        let log_scale = Tensor::randn([2, 3, 4, 192], (Kind::Float, Device::Cpu)) * 0.5;
        let derived = transform.horizon_log_scale(&log_scale);
        let worst = (&derived - &log_scale).abs().max().double_value(&[]);
        assert!(worst < 1e-5, "the identity covariance round trip is off by {worst:e}");
    }

    /// The derived horizon variance IS the diagonal of `Wᵀ diag(τ²) W`, checked against an
    /// independent host computation. This is the claim every per-horizon calibration chart now
    /// rests on, so it is proven rather than asserted in prose.
    #[test]
    fn derived_horizon_variance_is_the_covariance_diagonal() {
        let _rng = crate::torch::test_rng::shared();
        let n = 24usize;
        for basis in [TargetBasis::Haar, TargetBasis::Dct] {
            let rows = basis.matrix(n as i64).unwrap();
            let transform =
                BasisTransform::new(basis, BasisWeight::Uniform, n as i64, None, Device::Cpu)
                    .unwrap();
            let log_scale = Tensor::randn([1, 1, 1, n as i64], (Kind::Float, Device::Cpu)) * 0.7;
            let tau: Vec<f64> = Vec::<f64>::try_from(
                (&log_scale * 2.0).exp().to_kind(Kind::Double).reshape([-1]),
            )
            .unwrap();
            let derived: Vec<f64> = Vec::<f64>::try_from(
                (transform.horizon_log_scale(&log_scale) * 2.0)
                    .exp()
                    .to_kind(Kind::Double)
                    .reshape([-1]),
            )
            .unwrap();
            for h in 0..n {
                let expected: f64 = (0..n).map(|k| rows[k * n + h].powi(2) * tau[k]).sum();
                assert!(
                    (derived[h] - expected).abs() <= 1e-4 * expected,
                    "{basis} horizon {h}: derived {} vs {expected}",
                    derived[h]
                );
            }
        }
    }

    /// Whitening is measured from TARGETS alone - no head, no checkpoint - and the accumulator
    /// reproduces a hand-computed fp64 rms. The row mask is the completeness indicator, so a
    /// partially observed window contributes nothing rather than a zero-filled coefficient.
    #[test]
    fn target_moments_measure_the_rms_and_drop_incomplete_windows() {
        let transform =
            BasisTransform::new(TargetBasis::Dct, BasisWeight::Uniform, 4, None, Device::Cpu)
                .unwrap();
        let targets = Tensor::from_slice(&[
            1.0f32, 0.0, 0.0, 0.0, // row 0, channel 0
            2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
            // row 1: every channel 100, and a window that is NOT complete
            100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0,
        ])
        .reshape([2, 1, 4, 4]);
        let mask = Tensor::from_slice(&[1.0f32, 1., 1., 1., 1., 1., 0., 0.]).reshape([2, 1, 1, 4]);
        let mut moments = CoefficientMoments::new(4);
        moments.accumulate_targets(&transform, &targets, &mask);
        let scales = moments.whitening();
        // Only row 0 counts. Its four channels carry 1, 2, 2, 2 at horizon 0, and the DCT-II
        // coefficient of a spike at h = 0 is `α_k·cos(π·k/(2·4))`.
        let rows = TargetBasis::Dct.matrix(4).unwrap();
        for k in 0..4 {
            let w = rows[k * 4];
            let expected = (((1.0f64 * w).powi(2) + 3.0 * (2.0f64 * w).powi(2)) / 4.0).sqrt();
            assert!(
                (scales[k] - expected).abs() <= 1e-6 * expected.max(1e-12),
                "coefficient {k}: {} vs {expected}",
                scales[k]
            );
        }
    }

    /// `ρ̂` and `β̂` per coefficient recover a planted amplitude defect exactly. A forecast that
    /// is `g` times its target has `β̂ = 1/g` and `ρ̂ = 1`, which is the reading rule the new
    /// report chart states.
    #[test]
    fn coefficient_moments_recover_a_planted_amplitude_defect() {
        let _rng = crate::torch::test_rng::shared();
        let transform =
            BasisTransform::new(TargetBasis::Haar, BasisWeight::Uniform, 12, None, Device::Cpu)
                .unwrap();
        let targets = Tensor::randn([64, 3, 4, 12], (Kind::Float, Device::Cpu));
        let prediction = &targets * 3.8;
        let mask = Tensor::ones([64, 3, 1, 12], (Kind::Float, Device::Cpu));
        let mut moments = CoefficientMoments::new(12);
        moments.accumulate_pair(&transform, &targets, &prediction, &mask);
        for (k, (rho, beta)) in moments
            .correlation()
            .into_iter()
            .zip(moments.amplitude_gain())
            .enumerate()
        {
            assert!((rho - 1.0).abs() < 1e-4, "coefficient {k} ρ̂ = {rho}");
            assert!((beta - 1.0 / 3.8).abs() < 1e-4, "coefficient {k} β̂ = {beta}");
        }
    }
}
