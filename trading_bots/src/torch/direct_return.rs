//! Six-horizon cumulative-return categorical law.
//!
//! Each decision belief plus its causal adjusted-daily context predicts six independent
//! 128-bin categoricals in the fixed [`DIRECT_RETURN_HORIZONS`] order. Targets are cumulative
//! future close log returns; the joint five-DOF bar emission remains a separate law.

use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tch::{nn, nn::Module, Device, Kind, Tensor};

use crate::torch::{
    adjusted_daily::ADJUSTED_DAILY_CONTEXT_FEATURES,
    bar_dist::{BAR_DOF, DOF_R},
    dataset::{TimeRange, DIRECT_RETURN_HORIZONS},
    world_model::BAR_MODEL_DIM,
};

pub const DIRECT_RETURN_COUNT: usize = DIRECT_RETURN_HORIZONS.len();
pub const NUM_DIRECT_RETURN_BINS: i64 = 128;
pub const DIRECT_RETURN_SUPPORTS_FORMAT_VERSION: u32 = 2;
pub const DIRECT_RETURN_HEAD_INPUT_DIM: i64 =
    BAR_MODEL_DIM + ADJUSTED_DAILY_CONTEXT_FEATURES as i64;
/// The projection name deliberately contains the existing bar-emission AdamW
/// classifier substring (`bar_dof_head`). This keeps this output projection in
/// the established emission group without introducing a second optimizer rule.
pub const DIRECT_RETURN_HEAD_NAME: &str = "bar_dof_head_direct_return";

const BINS: usize = NUM_DIRECT_RETURN_BINS as usize;
const BOUNDARIES: usize = BINS - 1;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
/// Whether the v8 direct-return head consumes its causal adjusted-daily features or the
/// scientifically matched all-zero control. Both modes retain the identical 518-wide
/// projection and differ only in the six context values concatenated to each belief.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectReturnContextMode {
    #[default]
    Enabled,
    Masked,
}

impl DirectReturnContextMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Enabled => "enabled",
            Self::Masked => "masked",
        }
    }
}

/// Everything that identifies the rows used to fit a direct-return support.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectReturnSupportsProvenance {
    pub corpus_fingerprint: String,
    pub fit: TimeRange,
    pub fold_plan_hash: String,
    pub fold_index: u32,
    pub admitted_universe_digest: String,
    pub train_seed: u64,
    /// Campaign-constant, fold-specific sampling seed; independent of weight initialization.
    pub support_sample_seed: u64,
    pub row_count: usize,
}

impl DirectReturnSupportsProvenance {
    fn validate(&self) -> Result<()> {
        self.fit.validate()?;
        ensure!(
            !self.corpus_fingerprint.trim().is_empty(),
            "direct-return support corpus fingerprint is empty"
        );
        ensure!(
            !self.fold_plan_hash.trim().is_empty(),
            "direct-return support fold-plan hash is empty"
        );
        ensure!(
            !self.admitted_universe_digest.trim().is_empty(),
            "direct-return support admitted-universe digest is empty"
        );
        ensure!(
            self.row_count > 0,
            "direct-return support provenance has zero rows"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DirectReturnSupportsJson {
    format_version: u32,
    horizons: [usize; DIRECT_RETURN_COUNT],
    num_bins: i64,
    /// IEEE-754 payloads keep support thresholds and fitted moments bit-exact across JSON
    /// implementations. One-ULP drift can otherwise move an observed tie across a boundary.
    boundary_bits: [Vec<u64>; DIRECT_RETURN_COUNT],
    /// Explicit owner of the exact zero atom under the same strict-threshold rule.
    zero_bins: [usize; DIRECT_RETURN_COUNT],
    bin_mean_bits: [Vec<u64>; DIRECT_RETURN_COUNT],
    bin_second_moment_bits: [Vec<u64>; DIRECT_RETURN_COUNT],
    /// Per-bin fitted moments of the cumulative SIMPLE return `expm1(log_return)`.
    bin_simple_mean_bits: [Vec<u64>; DIRECT_RETURN_COUNT],
    bin_simple_second_moment_bits: [Vec<u64>; DIRECT_RETURN_COUNT],
    /// Exact fitted count law. Counts retain the empirical law independently of floating-point
    /// division; probability bits are persisted too so the cached categorical reference is
    /// bit-identical after reload.
    bin_counts: [Vec<u64>; DIRECT_RETURN_COUNT],
    bin_probability_bits: [Vec<u64>; DIRECT_RETURN_COUNT],
    provenance: DirectReturnSupportsProvenance,
}

/// Train-fitted equal-mass supports for the six fixed cumulative-return horizons.
///
/// Bounds and moments stay as deterministic `f64` host data for persistence and
/// audits. Device tensors are cached once, so binning and analytic moments never
/// copy through the host on the hot path. Quantile ties intentionally create empty
/// bins: all identical values belong to one class instead of being assigned by row
/// order. Empty-bin moments copy the nearest observed bin, a deterministic finite
/// completion that affects only probability assigned to a class absent from fit.
#[derive(Debug)]
pub struct DirectReturnSupports {
    boundaries: [Vec<f64>; DIRECT_RETURN_COUNT],
    zero_bins: [usize; DIRECT_RETURN_COUNT],
    bin_means: [Vec<f64>; DIRECT_RETURN_COUNT],
    bin_second_moments: [Vec<f64>; DIRECT_RETURN_COUNT],
    bin_simple_means: [Vec<f64>; DIRECT_RETURN_COUNT],
    bin_simple_second_moments: [Vec<f64>; DIRECT_RETURN_COUNT],
    bin_counts: [Vec<u64>; DIRECT_RETURN_COUNT],
    bin_probabilities: [Vec<f64>; DIRECT_RETURN_COUNT],
    provenance: DirectReturnSupportsProvenance,
    boundaries_t: Tensor,
    bin_means_t: Tensor,
    bin_second_moments_t: Tensor,
    bin_simple_means_t: Tensor,
    bin_simple_second_moments_t: Tensor,
    log_bin_probabilities_t: Tensor,
}

impl Serialize for DirectReturnSupports {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.as_json().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for DirectReturnSupports {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let json = DirectReturnSupportsJson::deserialize(deserializer)?;
        Self::from_json(json, Device::Cpu).map_err(serde::de::Error::custom)
    }
}

impl DirectReturnSupports {
    /// Fit all horizons independently from complete rows. No row or horizon is
    /// dropped or reordered: one non-finite target rejects the whole fit.
    pub fn fit(
        rows: &[[f64; DIRECT_RETURN_COUNT]],
        provenance: DirectReturnSupportsProvenance,
    ) -> Result<Self> {
        ensure!(
            !rows.is_empty(),
            "direct-return supports need at least one row"
        );
        provenance.validate()?;
        ensure!(
            provenance.row_count == rows.len(),
            "direct-return support provenance says {} rows but fit received {}",
            provenance.row_count,
            rows.len()
        );
        for (row, values) in rows.iter().enumerate() {
            for (horizon, value) in values.iter().enumerate() {
                ensure!(
                    value.is_finite(),
                    "direct-return fit row {row}, horizon {} is not finite: {value}",
                    DIRECT_RETURN_HORIZONS[horizon]
                );
                ensure!(
                    (value * value).is_finite(),
                    "direct-return fit row {row}, horizon {} has a non-finite square: {value}",
                    DIRECT_RETURN_HORIZONS[horizon]
                );
                let simple = value.exp_m1();
                ensure!(
                    simple.is_finite() && (simple * simple).is_finite(),
                    "direct-return fit row {row}, horizon {} has non-finite simple-return moments: {value}",
                    DIRECT_RETURN_HORIZONS[horizon]
                );
            }
        }

        let mut boundaries: [Vec<f64>; DIRECT_RETURN_COUNT] = std::array::from_fn(|_| Vec::new());
        let mut zero_bins = [0usize; DIRECT_RETURN_COUNT];
        let mut bin_means: [Vec<f64>; DIRECT_RETURN_COUNT] = std::array::from_fn(|_| Vec::new());
        let mut bin_second_moments: [Vec<f64>; DIRECT_RETURN_COUNT] =
            std::array::from_fn(|_| Vec::new());
        let mut bin_simple_means: [Vec<f64>; DIRECT_RETURN_COUNT] =
            std::array::from_fn(|_| Vec::new());
        let mut bin_simple_second_moments: [Vec<f64>; DIRECT_RETURN_COUNT] =
            std::array::from_fn(|_| Vec::new());
        let mut bin_counts: [Vec<u64>; DIRECT_RETURN_COUNT] = std::array::from_fn(|_| Vec::new());
        let mut bin_probabilities: [Vec<f64>; DIRECT_RETURN_COUNT] =
            std::array::from_fn(|_| Vec::new());

        for horizon in 0..DIRECT_RETURN_COUNT {
            let mut sorted: Vec<f64> = rows.iter().map(|row| row[horizon]).collect();
            sorted.sort_unstable_by(f64::total_cmp);
            let row_boundaries = equal_mass_boundaries(&sorted);
            zero_bins[horizon] = bin_from_boundaries(&row_boundaries, 0.0);

            let mut counts = vec![0usize; BINS];
            let mut sums = vec![0.0f64; BINS];
            let mut squares = vec![0.0f64; BINS];
            let mut simple_sums = vec![0.0f64; BINS];
            let mut simple_squares = vec![0.0f64; BINS];
            for &value in &sorted {
                let bin = bin_from_boundaries(&row_boundaries, value);
                counts[bin] += 1;
                sums[bin] += value;
                squares[bin] += value * value;
                let simple = value.exp_m1();
                simple_sums[bin] += simple;
                simple_squares[bin] += simple * simple;
            }
            let (means, seconds) = fitted_moments(&counts, &sums, &squares);
            let (simple_means, simple_seconds) =
                fitted_moments(&counts, &simple_sums, &simple_squares);
            boundaries[horizon] = row_boundaries;
            bin_means[horizon] = means;
            bin_second_moments[horizon] = seconds;
            bin_simple_means[horizon] = simple_means;
            bin_simple_second_moments[horizon] = simple_seconds;
            bin_counts[horizon] = counts.iter().map(|&count| count as u64).collect();
            bin_probabilities[horizon] = counts
                .iter()
                .map(|&count| count as f64 / rows.len() as f64)
                .collect();
        }

        Self::from_json(
            DirectReturnSupportsJson {
                format_version: DIRECT_RETURN_SUPPORTS_FORMAT_VERSION,
                horizons: DIRECT_RETURN_HORIZONS,
                num_bins: NUM_DIRECT_RETURN_BINS,
                boundary_bits: encode_f64_bits(&boundaries),
                zero_bins,
                bin_mean_bits: encode_f64_bits(&bin_means),
                bin_second_moment_bits: encode_f64_bits(&bin_second_moments),
                bin_simple_mean_bits: encode_f64_bits(&bin_simple_means),
                bin_simple_second_moment_bits: encode_f64_bits(&bin_simple_second_moments),
                bin_counts,
                bin_probability_bits: encode_f64_bits(&bin_probabilities),
                provenance,
            },
            Device::Cpu,
        )
    }

    fn as_json(&self) -> DirectReturnSupportsJson {
        DirectReturnSupportsJson {
            format_version: DIRECT_RETURN_SUPPORTS_FORMAT_VERSION,
            horizons: DIRECT_RETURN_HORIZONS,
            num_bins: NUM_DIRECT_RETURN_BINS,
            boundary_bits: encode_f64_bits(&self.boundaries),
            zero_bins: self.zero_bins,
            bin_mean_bits: encode_f64_bits(&self.bin_means),
            bin_second_moment_bits: encode_f64_bits(&self.bin_second_moments),
            bin_simple_mean_bits: encode_f64_bits(&self.bin_simple_means),
            bin_simple_second_moment_bits: encode_f64_bits(&self.bin_simple_second_moments),
            bin_counts: self.bin_counts.clone(),
            bin_probability_bits: encode_f64_bits(&self.bin_probabilities),
            provenance: self.provenance.clone(),
        }
    }

    fn from_json(json: DirectReturnSupportsJson, device: Device) -> Result<Self> {
        ensure!(
            json.format_version == DIRECT_RETURN_SUPPORTS_FORMAT_VERSION,
            "direct-return supports have format version {}, this build requires {}",
            json.format_version,
            DIRECT_RETURN_SUPPORTS_FORMAT_VERSION
        );
        ensure!(
            json.horizons == DIRECT_RETURN_HORIZONS,
            "direct-return support horizons {:?} differ from canonical {:?}",
            json.horizons,
            DIRECT_RETURN_HORIZONS
        );
        ensure!(
            json.num_bins == NUM_DIRECT_RETURN_BINS,
            "direct-return supports have {} bins, this build requires {}",
            json.num_bins,
            NUM_DIRECT_RETURN_BINS
        );
        json.provenance.validate()?;
        let boundaries = decode_f64_bits(&json.boundary_bits);
        let bin_means = decode_f64_bits(&json.bin_mean_bits);
        let bin_second_moments = decode_f64_bits(&json.bin_second_moment_bits);
        let bin_simple_means = decode_f64_bits(&json.bin_simple_mean_bits);
        let bin_simple_second_moments = decode_f64_bits(&json.bin_simple_second_moment_bits);
        let bin_probabilities = decode_f64_bits(&json.bin_probability_bits);

        for horizon in 0..DIRECT_RETURN_COUNT {
            let name = DIRECT_RETURN_HORIZONS[horizon];
            let row = &boundaries[horizon];
            ensure!(
                row.len() == BOUNDARIES,
                "H{name} support has {} boundaries, expected {BOUNDARIES}",
                row.len()
            );
            for (index, value) in row.iter().enumerate() {
                ensure!(value.is_finite(), "H{name} boundary {index} is not finite");
                if index > 0 {
                    ensure!(
                        row[index - 1] <= *value,
                        "H{name} boundaries are not monotone at {index}"
                    );
                }
            }
            ensure!(
                json.zero_bins[horizon] == bin_from_boundaries(row, 0.0),
                "H{name} zero atom claims bin {}, expected {}",
                json.zero_bins[horizon],
                bin_from_boundaries(row, 0.0)
            );
            for (what, moments) in [
                ("log means", &bin_means[horizon]),
                ("log second moments", &bin_second_moments[horizon]),
                ("simple-return means", &bin_simple_means[horizon]),
                (
                    "simple-return second moments",
                    &bin_simple_second_moments[horizon],
                ),
            ] {
                ensure!(
                    moments.len() == BINS,
                    "H{name} bin {what} has {} entries, expected {BINS}",
                    moments.len()
                );
                ensure!(
                    moments.iter().all(|value| value.is_finite()),
                    "H{name} bin {what} contains a non-finite entry"
                );
            }
            ensure!(
                json.bin_counts[horizon].len() == BINS,
                "H{name} fitted count law has {} entries, expected {BINS}",
                json.bin_counts[horizon].len()
            );
            ensure!(
                bin_probabilities[horizon].len() == BINS,
                "H{name} fitted probability law has {} entries, expected {BINS}",
                bin_probabilities[horizon].len()
            );
            let count_total = json.bin_counts[horizon]
                .iter()
                .try_fold(0u64, |total, &count| total.checked_add(count));
            ensure!(
                count_total == Some(json.provenance.row_count as u64),
                "H{name} fitted counts do not sum to the {} fit rows",
                json.provenance.row_count
            );
            let mut probability_total = 0.0f64;
            for bin in 0..BINS {
                let probability = bin_probabilities[horizon][bin];
                ensure!(
                    probability.is_finite() && probability >= 0.0,
                    "H{name} bin {bin} fitted probability is not finite and nonnegative: {probability}"
                );
                let expected =
                    json.bin_counts[horizon][bin] as f64 / json.provenance.row_count as f64;
                ensure!(
                    probability.to_bits() == expected.to_bits(),
                    "H{name} bin {bin} probability is not the bit-exact fitted count frequency"
                );
                probability_total += probability;
            }
            ensure!(
                (probability_total - 1.0).abs() <= 256.0 * f64::EPSILON,
                "H{name} fitted probabilities sum to {probability_total}, expected one"
            );
            for (unit, means, seconds) in [
                (
                    "log-return",
                    &bin_means[horizon],
                    &bin_second_moments[horizon],
                ),
                (
                    "simple-return",
                    &bin_simple_means[horizon],
                    &bin_simple_second_moments[horizon],
                ),
            ] {
                for bin in 0..BINS {
                    let mean = means[bin];
                    let second = seconds[bin];
                    ensure!(
                        second >= 0.0,
                        "H{name} {unit} bin {bin} has negative raw second moment {second}"
                    );
                    let tolerance = 1e-12 * (1.0 + second.abs() + mean.abs() * mean.abs());
                    ensure!(
                        second + tolerance >= mean * mean,
                        "H{name} {unit} bin {bin} has second moment {second} below mean square {}",
                        mean * mean
                    );
                }
            }
        }

        let flatten = |rows: &[Vec<f64>; DIRECT_RETURN_COUNT]| -> Vec<f64> {
            rows.iter().flat_map(|row| row.iter().copied()).collect()
        };
        let boundaries_t = Tensor::from_slice(&flatten(&boundaries))
            .view([DIRECT_RETURN_COUNT as i64, BOUNDARIES as i64])
            .to_device(device);
        let bin_means_t = Tensor::from_slice(&flatten(&bin_means))
            .view([DIRECT_RETURN_COUNT as i64, NUM_DIRECT_RETURN_BINS])
            .to_device(device);
        let bin_second_moments_t = Tensor::from_slice(&flatten(&bin_second_moments))
            .view([DIRECT_RETURN_COUNT as i64, NUM_DIRECT_RETURN_BINS])
            .to_device(device);
        let bin_simple_means_t = Tensor::from_slice(&flatten(&bin_simple_means))
            .view([DIRECT_RETURN_COUNT as i64, NUM_DIRECT_RETURN_BINS])
            .to_device(device);
        let bin_simple_second_moments_t = Tensor::from_slice(&flatten(&bin_simple_second_moments))
            .view([DIRECT_RETURN_COUNT as i64, NUM_DIRECT_RETURN_BINS])
            .to_device(device);
        let log_bin_probabilities_t = Tensor::from_slice(&flatten(&bin_probabilities))
            .view([DIRECT_RETURN_COUNT as i64, NUM_DIRECT_RETURN_BINS])
            .to_device(device)
            .log();

        Ok(Self {
            boundaries,
            zero_bins: json.zero_bins,
            bin_means,
            bin_second_moments,
            bin_simple_means,
            bin_simple_second_moments,
            bin_counts: json.bin_counts,
            bin_probabilities,
            provenance: json.provenance,
            boundaries_t,
            bin_means_t,
            bin_second_moments_t,
            bin_simple_means_t,
            bin_simple_second_moments_t,
            log_bin_probabilities_t,
        })
    }

    pub fn provenance(&self) -> &DirectReturnSupportsProvenance {
        &self.provenance
    }

    pub fn boundaries(&self, horizon: usize) -> &[f64] {
        &self.boundaries[horizon]
    }

    pub fn bin_means(&self, horizon: usize) -> &[f64] {
        &self.bin_means[horizon]
    }

    pub fn bin_second_moments(&self, horizon: usize) -> &[f64] {
        &self.bin_second_moments[horizon]
    }
    pub fn bin_simple_means(&self, horizon: usize) -> &[f64] {
        &self.bin_simple_means[horizon]
    }

    pub fn bin_simple_second_moments(&self, horizon: usize) -> &[f64] {
        &self.bin_simple_second_moments[horizon]
    }
    pub fn bin_counts(&self, horizon: usize) -> &[u64] {
        &self.bin_counts[horizon]
    }

    pub fn bin_probabilities(&self, horizon: usize) -> &[f64] {
        &self.bin_probabilities[horizon]
    }

    pub fn zero_bin(&self, horizon: usize) -> usize {
        self.zero_bins[horizon]
    }

    pub fn to_device(&self, device: Device) -> Self {
        Self::from_json(self.as_json(), device).expect("validated supports remain valid")
    }

    /// Host twin of [`Self::bin_ids`]. Exact ties, including signed zero, are
    /// assigned together using the number of thresholds strictly below the value.
    pub fn bin_of(&self, horizon: usize, value: f64) -> Result<usize> {
        ensure!(
            horizon < DIRECT_RETURN_COUNT,
            "direct-return horizon index {horizon} is out of range"
        );
        ensure!(
            value.is_finite(),
            "cannot bin non-finite direct return {value}"
        );
        Ok(bin_from_boundaries(&self.boundaries[horizon], value))
    }

    /// `[..., 6]` cumulative returns to `[..., 6]` Int64 class ids.
    pub fn bin_ids(&self, values: &Tensor) -> Tensor {
        let size = values.size();
        assert_eq!(
            size.last().copied(),
            Some(DIRECT_RETURN_COUNT as i64),
            "direct-return values must end in six horizons"
        );
        values
            .to_kind(Kind::Double)
            .unsqueeze(-1)
            .gt_tensor(&self.boundaries_t.to_device(values.device()))
            .sum_dim_intlist([-1].as_slice(), false, Kind::Int64)
    }

    /// Analytic `(E[x], Var[x])` for `[..., 6, 128]` logits using fitted
    /// within-bin first and second moments.
    pub fn expectation_variance(&self, logits: &Tensor) -> (Tensor, Tensor) {
        assert_direct_logits(logits);
        let probs = logits.softmax(-1, Kind::Float);
        let device = logits.device();
        let kind = probs.kind();
        let means = self.bin_means_t.to_device(device).to_kind(kind);
        let seconds = self.bin_second_moments_t.to_device(device).to_kind(kind);
        let mean = (&probs * means).sum_dim_intlist([-1].as_slice(), false, kind);
        let second = (probs * seconds).sum_dim_intlist([-1].as_slice(), false, kind);
        let variance = (second - &mean * &mean).clamp_min(0.0);
        (mean, variance)
    }
    /// Exact categorical `(E[R], E[R²])` for cumulative simple return
    /// `R = expm1(log_return)`, using fitted within-bin raw moments. This deliberately does
    /// not use `expm1(E[log_return])`, which is biased under within-bin dispersion.
    pub fn simple_expectation_second_moment(&self, logits: &Tensor) -> (Tensor, Tensor) {
        assert_direct_logits(logits);
        let probs = logits.softmax(-1, Kind::Float);
        let device = logits.device();
        let kind = probs.kind();
        let means = self.bin_simple_means_t.to_device(device).to_kind(kind);
        let seconds = self
            .bin_simple_second_moments_t
            .to_device(device)
            .to_kind(kind);
        let mean = (&probs * means).sum_dim_intlist([-1].as_slice(), false, kind);
        let second = (probs * seconds).sum_dim_intlist([-1].as_slice(), false, kind);
        (mean, second)
    }
    /// Per-horizon sufficient statistics for the actual train-fit categorical marginal.
    ///
    /// Zero-probability fitted bins cache as `-inf`. Invalid target positions are replaced
    /// before summation rather than multiplied by zero, avoiding the undefined `0 * inf`.
    /// The caller divides these sums by the returned valid counts so chunks are weighted by
    /// target rows, never by batch count.
    pub fn marginal_ce_sums(&self, target_bins: &Tensor, valid: &Tensor) -> (Tensor, Tensor) {
        assert_eq!(
            target_bins.kind(),
            Kind::Int64,
            "direct-return target bins must be Int64"
        );
        assert_eq!(
            valid.kind(),
            Kind::Bool,
            "direct-return valid mask must be Bool"
        );
        assert_eq!(
            target_bins.size(),
            valid.size(),
            "direct-return marginal targets and valid mask must have identical shapes"
        );
        assert_eq!(
            target_bins.size().last().copied(),
            Some(DIRECT_RETURN_COUNT as i64),
            "direct-return marginal targets must end in six horizons"
        );
        let flat_bins = target_bins.reshape([-1, DIRECT_RETURN_COUNT as i64]);
        let flat_valid = valid.reshape([-1, DIRECT_RETURN_COUNT as i64]);
        let rows = flat_bins.size()[0];
        let log_probabilities = self
            .log_bin_probabilities_t
            .to_device(target_bins.device())
            .unsqueeze(0)
            .expand(
                [rows, DIRECT_RETURN_COUNT as i64, NUM_DIRECT_RETURN_BINS],
                true,
            );
        let terms = -log_probabilities
            .gather(-1, &flat_bins.unsqueeze(-1), false)
            .squeeze_dim(-1);
        let terms = terms.masked_fill(&flat_valid.logical_not(), 0.0);
        let sums =
            terms
                .to_kind(Kind::Double)
                .sum_dim_intlist([0i64].as_slice(), false, Kind::Double);
        let counts = flat_valid.to_kind(Kind::Double).sum_dim_intlist(
            [0i64].as_slice(),
            false,
            Kind::Double,
        );
        (sums, counts)
    }

    /// Atomically replace a JSON support artifact and durably publish its rename.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty());
        if let Some(parent) = parent {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let temporary = temporary_path(path);
        let write_result = (|| -> Result<()> {
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .with_context(|| format!("creating {}", temporary.display()))?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer_pretty(&mut writer, self)
                .context("serializing direct-return supports")?;
            writer.write_all(b"\n")?;
            writer.flush()?;
            writer.get_ref().sync_all()?;
            drop(writer);
            fs::rename(&temporary, path).with_context(|| {
                format!(
                    "renaming direct-return supports {} onto {}",
                    temporary.display(),
                    path.display()
                )
            })?;
            if let Some(parent) = parent {
                File::open(parent)?.sync_all()?;
            }
            Ok(())
        })();
        if write_result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        write_result
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let reader = BufReader::new(
            File::open(path).with_context(|| format!("opening {}", path.display()))?,
        );
        serde_json::from_reader(reader).with_context(|| format!("parsing {}", path.display()))
    }
}

/// Final six-horizon diagnostics on one pinned fold-validation population.
#[derive(Clone, Copy, Debug)]
pub struct ValidationStats {
    pub model_ce: [f64; DIRECT_RETURN_COUNT],
    pub marginal_ce: [f64; DIRECT_RETURN_COUNT],
    pub gain: [f64; DIRECT_RETURN_COUNT],
    pub valid_rows: [u64; DIRECT_RETURN_COUNT],
    pub coverage: [f64; DIRECT_RETURN_COUNT],
    pub mean_bias: [f64; DIRECT_RETURN_COUNT],
    pub mean_rmse: [f64; DIRECT_RETURN_COUNT],
    pub predicted_variance_mean: [f64; DIRECT_RETURN_COUNT],
    pub squared_error_mean: [f64; DIRECT_RETURN_COUNT],
    pub directional_accuracy: [f64; DIRECT_RETURN_COUNT],
    pub directional_rows: [u64; DIRECT_RETURN_COUNT],
}

impl ValidationStats {
    pub fn nan() -> Self {
        Self {
            model_ce: [f64::NAN; DIRECT_RETURN_COUNT],
            marginal_ce: [f64::NAN; DIRECT_RETURN_COUNT],
            gain: [f64::NAN; DIRECT_RETURN_COUNT],
            valid_rows: [0; DIRECT_RETURN_COUNT],
            coverage: [f64::NAN; DIRECT_RETURN_COUNT],
            mean_bias: [f64::NAN; DIRECT_RETURN_COUNT],
            mean_rmse: [f64::NAN; DIRECT_RETURN_COUNT],
            predicted_variance_mean: [f64::NAN; DIRECT_RETURN_COUNT],
            squared_error_mean: [f64::NAN; DIRECT_RETURN_COUNT],
            directional_accuracy: [f64::NAN; DIRECT_RETURN_COUNT],
            directional_rows: [0; DIRECT_RETURN_COUNT],
        }
    }
}

/// Additive host sufficient statistics. Every device transfer used to construct this value is
/// bounded by the six horizons, regardless of validation-window count or sequence length.
#[derive(Clone, Copy, Debug, Default)]
pub struct ValidationSums {
    model_ce_sum: [f64; DIRECT_RETURN_COUNT],
    marginal_ce_sum: [f64; DIRECT_RETURN_COUNT],
    error_sum: [f64; DIRECT_RETURN_COUNT],
    squared_error_sum: [f64; DIRECT_RETURN_COUNT],
    predicted_variance_sum: [f64; DIRECT_RETURN_COUNT],
    directional_correct: [f64; DIRECT_RETURN_COUNT],
    valid_rows: [u64; DIRECT_RETURN_COUNT],
    directional_rows: [u64; DIRECT_RETURN_COUNT],
    eligible_rows: u64,
}

impl ValidationSums {
    pub fn absorb(&mut self, other: Self) {
        for horizon in 0..DIRECT_RETURN_COUNT {
            self.model_ce_sum[horizon] += other.model_ce_sum[horizon];
            self.marginal_ce_sum[horizon] += other.marginal_ce_sum[horizon];
            self.error_sum[horizon] += other.error_sum[horizon];
            self.squared_error_sum[horizon] += other.squared_error_sum[horizon];
            self.predicted_variance_sum[horizon] += other.predicted_variance_sum[horizon];
            self.directional_correct[horizon] += other.directional_correct[horizon];
            self.valid_rows[horizon] += other.valid_rows[horizon];
            self.directional_rows[horizon] += other.directional_rows[horizon];
        }
        self.eligible_rows += other.eligible_rows;
    }

    pub fn finish(self) -> ValidationStats {
        let mut stats = ValidationStats::nan();
        for horizon in 0..DIRECT_RETURN_COUNT {
            let valid = self.valid_rows[horizon];
            stats.valid_rows[horizon] = valid;
            stats.directional_rows[horizon] = self.directional_rows[horizon];
            if self.eligible_rows > 0 {
                stats.coverage[horizon] = valid as f64 / self.eligible_rows as f64;
            }
            if valid > 0 {
                let scale = 1.0 / valid as f64;
                stats.model_ce[horizon] = self.model_ce_sum[horizon] * scale;
                stats.marginal_ce[horizon] = self.marginal_ce_sum[horizon] * scale;
                stats.gain[horizon] = stats.marginal_ce[horizon] - stats.model_ce[horizon];
                stats.mean_bias[horizon] = self.error_sum[horizon] * scale;
                stats.squared_error_mean[horizon] = self.squared_error_sum[horizon] * scale;
                stats.mean_rmse[horizon] = stats.squared_error_mean[horizon].sqrt();
                stats.predicted_variance_mean[horizon] =
                    self.predicted_variance_sum[horizon] * scale;
            }
            let directional = self.directional_rows[horizon];
            if directional > 0 {
                stats.directional_accuracy[horizon] =
                    self.directional_correct[horizon] / directional as f64;
            }
        }
        stats
    }
}

/// Reduce one validation chunk to six-horizon host sufficient statistics.
pub fn direct_return_validation_sums(
    supports: &DirectReturnSupports,
    logits: &Tensor,
    realized: &Tensor,
    target_bins: &Tensor,
    valid: &Tensor,
) -> ValidationSums {
    assert_direct_logits(logits);
    let logits_size = logits.size();
    let expected = &logits_size[..logits_size.len() - 1];
    assert_eq!(
        realized.size().as_slice(),
        expected,
        "direct-return realizations must match logits without the bin axis"
    );
    assert_eq!(
        target_bins.size().as_slice(),
        expected,
        "direct-return bins must match logits without the bin axis"
    );
    assert_eq!(
        valid.size().as_slice(),
        expected,
        "direct-return valid mask must match logits without the bin axis"
    );
    let flat_valid = valid.reshape([-1, DIRECT_RETURN_COUNT as i64]);
    let model_terms = -logits
        .log_softmax(-1, Kind::Float)
        .gather(-1, &target_bins.unsqueeze(-1), false)
        .squeeze_dim(-1)
        .reshape([-1, DIRECT_RETURN_COUNT as i64])
        .masked_fill(&flat_valid.logical_not(), 0.0);
    let model_ce_sum = horizon_values(&model_terms.to_kind(Kind::Double).sum_dim_intlist(
        [0i64].as_slice(),
        false,
        Kind::Double,
    ));
    let (marginal_ce_sum_t, valid_rows_t) = supports.marginal_ce_sums(target_bins, valid);
    let marginal_ce_sum = horizon_values(&marginal_ce_sum_t);
    let valid_rows_f = horizon_values(&valid_rows_t);

    let (predicted_mean, predicted_variance) = supports.expectation_variance(logits);
    let realized = realized
        .reshape([-1, DIRECT_RETURN_COUNT as i64])
        .to_kind(Kind::Double);
    let predicted_mean = predicted_mean
        .reshape([-1, DIRECT_RETURN_COUNT as i64])
        .to_kind(Kind::Double);
    let predicted_variance = predicted_variance
        .reshape([-1, DIRECT_RETURN_COUNT as i64])
        .to_kind(Kind::Double);
    let error = &predicted_mean - &realized;
    let invalid = flat_valid.logical_not();
    let error_sum = horizon_values(&error.masked_fill(&invalid, 0.0).sum_dim_intlist(
        [0i64].as_slice(),
        false,
        Kind::Double,
    ));
    let squared_error_sum =
        horizon_values(&error.square().masked_fill(&invalid, 0.0).sum_dim_intlist(
            [0i64].as_slice(),
            false,
            Kind::Double,
        ));
    let predicted_variance_sum = horizon_values(
        &predicted_variance
            .masked_fill(&invalid, 0.0)
            .sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
    );

    let directional_valid = flat_valid.logical_and(&realized.not_equal(0.0));
    let directional_hits = predicted_mean
        .sign()
        .eq_tensor(&realized.sign())
        .logical_and(&directional_valid);
    let directional_correct =
        horizon_values(&directional_hits.to_kind(Kind::Double).sum_dim_intlist(
            [0i64].as_slice(),
            false,
            Kind::Double,
        ));
    let directional_rows_f =
        horizon_values(&directional_valid.to_kind(Kind::Double).sum_dim_intlist(
            [0i64].as_slice(),
            false,
            Kind::Double,
        ));

    ValidationSums {
        model_ce_sum,
        marginal_ce_sum,
        error_sum,
        squared_error_sum,
        predicted_variance_sum,
        directional_correct,
        valid_rows: valid_rows_f.map(|value| value as u64),
        directional_rows: directional_rows_f.map(|value| value as u64),
        eligible_rows: u64::try_from(flat_valid.size()[0])
            .expect("direct-return validation row count fits u64"),
    }
}

fn horizon_values(tensor: &Tensor) -> [f64; DIRECT_RETURN_COUNT] {
    let values = Vec::<f64>::try_from(tensor.reshape([-1]))
        .expect("six-horizon sufficient statistic is convertible");
    assert_eq!(values.len(), DIRECT_RETURN_COUNT);
    std::array::from_fn(|horizon| values[horizon])
}

fn encode_f64_bits(rows: &[Vec<f64>; DIRECT_RETURN_COUNT]) -> [Vec<u64>; DIRECT_RETURN_COUNT] {
    std::array::from_fn(|horizon| rows[horizon].iter().map(|value| value.to_bits()).collect())
}

fn decode_f64_bits(rows: &[Vec<u64>; DIRECT_RETURN_COUNT]) -> [Vec<f64>; DIRECT_RETURN_COUNT] {
    std::array::from_fn(|horizon| {
        rows[horizon]
            .iter()
            .map(|bits| f64::from_bits(*bits))
            .collect()
    })
}

fn temporary_path(path: &Path) -> PathBuf {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("direct-return-supports.json");
    path.with_file_name(format!(".{name}.{}.{}.tmp", std::process::id(), sequence))
}

fn equal_mass_boundaries(sorted: &[f64]) -> Vec<f64> {
    debug_assert!(!sorted.is_empty());
    (1..BINS)
        .map(|cut| {
            let right =
                ((sorted.len() as u128 * cut as u128 + BINS as u128 - 1) / BINS as u128) as usize;
            let right = right.clamp(1, sorted.len());
            let left_value = sorted[right - 1];
            if right == sorted.len() || sorted[right] == left_value {
                left_value
            } else {
                // Overflow-safe midpoint. If adjacent f64 values have no interior
                // representable value, choosing the lower one still separates them
                // under the strict-threshold lookup.
                left_value + (sorted[right] - left_value) * 0.5
            }
        })
        .collect()
}

fn bin_from_boundaries(boundaries: &[f64], value: f64) -> usize {
    boundaries.partition_point(|boundary| *boundary < value)
}

fn fitted_moments(counts: &[usize], sums: &[f64], squares: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let observed: Vec<usize> = counts
        .iter()
        .enumerate()
        .filter_map(|(bin, count)| (*count > 0).then_some(bin))
        .collect();
    debug_assert!(!observed.is_empty());
    let mut means = vec![0.0; BINS];
    let mut seconds = vec![0.0; BINS];
    for bin in 0..BINS {
        let source = if counts[bin] > 0 {
            bin
        } else {
            *observed
                .iter()
                .min_by_key(|observed| observed.abs_diff(bin))
                .expect("at least one observed bin")
        };
        means[bin] = sums[source] / counts[source] as f64;
        seconds[bin] = squares[source] / counts[source] as f64;
    }
    (means, seconds)
}

/// Independent six-horizon readout from a decision belief and the exact point-in-time
/// adjusted-daily context attached to that decision. No future bar or target-prefix input enters.
#[derive(Debug)]
pub struct DirectReturnHead {
    projection: nn::Linear,
}

impl DirectReturnHead {
    pub fn new(vs: &nn::Path) -> Self {
        let projection = nn::linear(
            vs / DIRECT_RETURN_HEAD_NAME,
            DIRECT_RETURN_HEAD_INPUT_DIM,
            DIRECT_RETURN_COUNT as i64 * NUM_DIRECT_RETURN_BINS,
            nn::LinearConfig {
                ws_init: nn::Init::Const(0.0),
                bs_init: Some(nn::Init::Const(0.0)),
                bias: true,
            },
        );
        Self { projection }
    }

    /// `belief [..., BAR_MODEL_DIM]` plus `adjusted_daily
    /// [..., ADJUSTED_DAILY_CONTEXT_FEATURES]` to `[..., 6, 128]` logits.
    ///
    /// `context_mode` is part of the checkpoint contract. The masked arm still executes this
    /// exact v8 projection, but replaces only the six adjusted-daily inputs with exact zeros.
    pub fn logits(
        &self,
        belief: &Tensor,
        adjusted_daily: &Tensor,
        context_mode: DirectReturnContextMode,
    ) -> Tensor {
        let mut shape = belief.size();
        assert_eq!(
            shape.last().copied(),
            Some(BAR_MODEL_DIM),
            "direct-return beliefs must end in BAR_MODEL_DIM"
        );
        let mut context_shape = adjusted_daily.size();
        assert_eq!(
            context_shape.pop(),
            Some(ADJUSTED_DAILY_CONTEXT_FEATURES as i64),
            "direct-return adjusted-daily context has the wrong feature width"
        );
        assert_eq!(
            context_shape,
            shape[..shape.len() - 1],
            "direct-return belief and adjusted-daily leading dimensions differ"
        );
        shape.pop();
        shape.extend([DIRECT_RETURN_COUNT as i64, NUM_DIRECT_RETURN_BINS]);
        // Allocate the zero control in both arms. That keeps capacity probing and runtime
        // tensor geometry matched; only the six values selected for concatenation differ.
        let zero_context = Tensor::zeros_like(adjusted_daily);
        let effective_context = match context_mode {
            DirectReturnContextMode::Enabled => adjusted_daily,
            DirectReturnContextMode::Masked => &zero_context,
        };
        let input = Tensor::cat(&[belief, effective_context], -1);
        self.projection.forward(&input).reshape(&shape)
    }

    pub fn loss(
        &self,
        belief: &Tensor,
        adjusted_daily: &Tensor,
        context_mode: DirectReturnContextMode,
        target_bins: &Tensor,
    ) -> (Tensor, Tensor) {
        direct_return_ce_from_logits(
            &self.logits(belief, adjusted_daily, context_mode),
            target_bins,
        )
    }

    pub fn moments(
        &self,
        belief: &Tensor,
        adjusted_daily: &Tensor,
        context_mode: DirectReturnContextMode,
        supports: &DirectReturnSupports,
    ) -> (Tensor, Tensor) {
        supports.expectation_variance(&self.logits(belief, adjusted_daily, context_mode))
    }
}

/// Proper hard categorical cross entropy. Returns `(mean over rows and horizons,
/// six per-horizon means)` without materializing one-hot targets.
pub fn direct_return_ce_from_logits(logits: &Tensor, target_bins: &Tensor) -> (Tensor, Tensor) {
    assert_direct_logits(logits);
    let logits_size = logits.size();
    let expected = &logits_size[..logits_size.len() - 1];
    assert_eq!(
        target_bins.size().as_slice(),
        expected,
        "direct-return target bins must match logits without the bin axis"
    );
    assert_eq!(
        target_bins.kind(),
        Kind::Int64,
        "direct-return target bins must be Int64"
    );
    let terms = -logits
        .log_softmax(-1, Kind::Float)
        .gather(-1, &target_bins.unsqueeze(-1), false)
        .squeeze_dim(-1);
    let per_horizon = terms.reshape([-1, DIRECT_RETURN_COUNT as i64]).mean_dim(
        [0i64].as_slice(),
        false,
        Kind::Float,
    );
    let mean = per_horizon.mean(Kind::Float);
    (mean, per_horizon)
}

fn assert_direct_logits(logits: &Tensor) {
    let size = logits.size();
    assert!(
        size.len() >= 2,
        "direct-return logits need horizon and bin axes"
    );
    assert_eq!(
        &size[size.len() - 2..],
        &[DIRECT_RETURN_COUNT as i64, NUM_DIRECT_RETURN_BINS],
        "direct-return logits must end in [6, 128]"
    );
}

/// Cumulative targets and the exact positions at which every horizon is complete.
#[derive(Debug)]
pub struct DirectReturnTargets {
    /// `[..., sequence, 6]` cumulative future log returns.
    pub values: Tensor,
    /// `[..., sequence, 6]` Bool. Invalid target values are zeroed.
    pub valid: Tensor,
}

/// Construct targets assuming every supplied sequence row is real (not padding).
pub fn direct_return_targets(dof: &Tensor) -> DirectReturnTargets {
    let mut valid_shape = dof.size();
    assert!(
        valid_shape.len() >= 2,
        "teacher-forced DOF needs sequence and DOF axes"
    );
    assert_eq!(
        valid_shape.last().copied(),
        Some(BAR_DOF as i64),
        "teacher-forced DOF must end in BAR_DOF"
    );
    valid_shape.pop();
    let row_valid = Tensor::ones(&valid_shape, (Kind::Bool, dof.device()));
    direct_return_targets_with_mask(dof, &row_valid)
}

/// Construct targets with an explicit `[..., sequence]` real-row mask. Belief
/// position `t`, horizon `h`, is `sum(DOF_R[t+1..=t+h])`; it is valid only when
/// the decision row and every one of those future rows is real and in range.
pub fn direct_return_targets_with_mask(dof: &Tensor, row_valid: &Tensor) -> DirectReturnTargets {
    let size = dof.size();
    assert!(
        size.len() >= 2,
        "teacher-forced DOF needs sequence and DOF axes"
    );
    assert_eq!(
        size[size.len() - 1],
        BAR_DOF as i64,
        "teacher-forced DOF must end in BAR_DOF"
    );
    assert!(
        size[size.len() - 2] > 0,
        "teacher-forced DOF sequence is empty"
    );
    assert_eq!(
        row_valid.size().as_slice(),
        &size[..size.len() - 1],
        "row-valid mask must match teacher-forced DOF without its DOF axis"
    );
    assert_eq!(row_valid.kind(), Kind::Bool, "row-valid mask must be Bool");
    let length = size[size.len() - 2];
    let rows = size[..size.len() - 2].iter().product::<i64>();
    let r = dof.select(-1, DOF_R as i64).reshape([rows, length]);
    let row_valid = row_valid
        .to_device(dof.device())
        .to_kind(Kind::Int64)
        .reshape([rows, length]);

    let zero_values = Tensor::zeros([rows, 1], (r.kind(), r.device()));
    let prefix = Tensor::cat(&[zero_values, r.cumsum(1, r.kind())], 1);
    let zero_counts = Tensor::zeros([rows, 1], (Kind::Int64, r.device()));
    let valid_prefix = Tensor::cat(&[zero_counts, row_valid.cumsum(1, Kind::Int64)], 1);

    let positions = Tensor::arange(length, (Kind::Int64, r.device())).view([length, 1]);
    let horizons = Tensor::from_slice(
        &DIRECT_RETURN_HORIZONS
            .iter()
            .map(|horizon| *horizon as i64)
            .collect::<Vec<_>>(),
    )
    .to_device(r.device())
    .view([1, DIRECT_RETURN_COUNT as i64]);
    let end = &positions + 1 + &horizons;
    let complete = end.le(length);
    let end_clamped = end.clamp_max(length);
    let base = (&positions + 1).expand([length, DIRECT_RETURN_COUNT as i64], true);
    let gather_shape = [rows, length, DIRECT_RETURN_COUNT as i64];
    let end_index = end_clamped.unsqueeze(0).expand(gather_shape, true);
    let base_index = base.unsqueeze(0).expand(gather_shape, true);

    let prefix = prefix
        .unsqueeze(-1)
        .expand([rows, length + 1, DIRECT_RETURN_COUNT as i64], true);
    let values = prefix.gather(1, &end_index, false) - prefix.gather(1, &base_index, false);
    let valid_prefix = valid_prefix
        .unsqueeze(-1)
        .expand([rows, length + 1, DIRECT_RETURN_COUNT as i64], true);
    let valid_count =
        valid_prefix.gather(1, &end_index, false) - valid_prefix.gather(1, &base_index, false);
    let decision_valid = row_valid
        .to_kind(Kind::Bool)
        .unsqueeze(-1)
        .expand(gather_shape, true);
    let valid = complete
        .unsqueeze(0)
        .expand(gather_shape, true)
        .logical_and(&decision_valid)
        .logical_and(&valid_count.eq_tensor(&horizons.unsqueeze(0)));
    let values = values * valid.to_kind(r.kind());

    let mut output_shape = size[..size.len() - 2].to_vec();
    output_shape.extend([length, DIRECT_RETURN_COUNT as i64]);
    DirectReturnTargets {
        values: values.reshape(&output_shape),
        valid: valid.reshape(&output_shape),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance(rows: usize) -> DirectReturnSupportsProvenance {
        DirectReturnSupportsProvenance {
            corpus_fingerprint: "corpus-sha256".to_owned(),
            fit: TimeRange::new(1_000, 9_000),
            fold_plan_hash: "fold-plan-sha256".to_owned(),
            fold_index: 2,
            admitted_universe_digest: "universe-sha256".to_owned(),
            train_seed: 42,
            support_sample_seed: 7,
            row_count: rows,
        }
    }

    fn support_rows() -> Vec<[f64; DIRECT_RETURN_COUNT]> {
        (0..512)
            .map(|row| {
                std::array::from_fn(|horizon| {
                    if row < 96 {
                        0.0
                    } else if row == 96 {
                        -50.0 - horizon as f64
                    } else if row == 511 {
                        70.0 + horizon as f64
                    } else {
                        ((row % 37) as f64 - 18.0) * (horizon as f64 + 1.0) / 1_000.0
                    }
                })
            })
            .collect()
    }

    #[test]
    fn cumulative_targets_align_every_horizon_and_reject_partial_tails() {
        let length = 104i64;
        let mut data = vec![0.0f32; length as usize * BAR_DOF];
        for t in 0..length as usize {
            data[t * BAR_DOF + DOF_R] = (t + 1) as f32;
        }
        let dof = Tensor::from_slice(&data).view([1, length, BAR_DOF as i64]);
        let targets = direct_return_targets(&dof);
        for (slot, horizon) in DIRECT_RETURN_HORIZONS.iter().copied().enumerate() {
            let expected = (horizon * (horizon + 3) / 2) as f64;
            assert_eq!(targets.values.double_value(&[0, 0, slot as i64]), expected);
            assert_eq!(
                targets
                    .valid
                    .to_kind(Kind::Int64)
                    .sum(Kind::Int64)
                    .int64_value(&[]),
                DIRECT_RETURN_HORIZONS
                    .iter()
                    .map(|h| length - *h as i64)
                    .sum::<i64>()
            );
            assert!(
                targets
                    .valid
                    .int64_value(&[0, length - horizon as i64 - 1, slot as i64])
                    != 0
            );
            assert!(
                targets
                    .valid
                    .int64_value(&[0, length - horizon as i64, slot as i64])
                    == 0
            );
        }
    }

    #[test]
    fn cumulative_targets_have_no_s_u_v_w_dependency_and_respect_padding() {
        let length = 110i64;
        let base = Tensor::zeros([2, length, BAR_DOF as i64], (Kind::Float, Device::Cpu));
        let changed = base.copy();
        for dof in 1..BAR_DOF as i64 {
            let _ = changed.narrow(-1, dof, 1).fill_(17.0 * dof as f64);
        }
        let valid = Tensor::ones([2, length], (Kind::Bool, Device::Cpu));
        let _ = valid.narrow(1, 105, 5).fill_(0);
        let a = direct_return_targets_with_mask(&base, &valid);
        let b = direct_return_targets_with_mask(&changed, &valid);
        assert_eq!((a.values - b.values).abs().max().double_value(&[]), 0.0);
        assert_eq!(a.valid.int64_value(&[0, 5, 5]), 0, "H100 crosses padding");
        assert_ne!(a.valid.int64_value(&[0, 5, 4]), 0, "H78 remains complete");
    }

    #[test]
    fn support_fit_keeps_ties_zero_atoms_and_open_tails_deterministic() {
        let rows = support_rows();
        let supports = DirectReturnSupports::fit(&rows, provenance(rows.len())).unwrap();
        for horizon in 0..DIRECT_RETURN_COUNT {
            assert!(supports
                .boundaries(horizon)
                .windows(2)
                .all(|pair| pair[0] <= pair[1]));
            let zero = supports.bin_of(horizon, 0.0).unwrap();
            assert_eq!(zero, supports.zero_bin(horizon));
            assert!(rows
                .iter()
                .filter(|row| row[horizon] == 0.0)
                .all(|row| { supports.bin_of(horizon, row[horizon]).unwrap() == zero }));
            assert_eq!(supports.bin_of(horizon, -1e100).unwrap(), 0);
            assert_eq!(supports.bin_of(horizon, 1e100).unwrap(), BINS - 1);
            assert!(supports
                .bin_means(horizon)
                .iter()
                .all(|value| value.is_finite()));
            assert!(supports
                .bin_second_moments(horizon)
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0));
        }
    }

    #[test]
    fn json_round_trip_preserves_provenance_and_rejects_schema_drift() {
        let rows = support_rows();
        let supports = DirectReturnSupports::fit(&rows, provenance(rows.len())).unwrap();
        let dir = std::env::temp_dir().join(format!(
            "direct-return-supports-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let path = dir.join("supports.json");
        supports.save(&path).unwrap();
        let loaded = DirectReturnSupports::load(&path).unwrap();
        assert_eq!(loaded.provenance(), supports.provenance());
        for horizon in 0..DIRECT_RETURN_COUNT {
            for bin in 0..BOUNDARIES {
                let actual = loaded.boundaries[horizon][bin];
                let expected = supports.boundaries[horizon][bin];
                assert!(
                    actual == expected,
                    "boundary round-trip changed H{} bin {bin}: actual={actual:e} ({:016x}), expected={expected:e} ({:016x})",
                    DIRECT_RETURN_HORIZONS[horizon],
                    actual.to_bits(),
                    expected.to_bits(),
                );
            }
            for bin in 0..BINS {
                let actual = loaded.bin_means[horizon][bin];
                let expected = supports.bin_means[horizon][bin];
                assert!(
                    (actual - expected).abs()
                        <= 2.0 * f64::EPSILON * expected.abs().max(1.0),
                    "mean round-trip changed H{} bin {bin}: actual={actual:e} ({:016x}), expected={expected:e} ({:016x})",
                    DIRECT_RETURN_HORIZONS[horizon],
                    actual.to_bits(),
                    expected.to_bits(),
                );
                let actual = loaded.bin_second_moments[horizon][bin];
                let expected = supports.bin_second_moments[horizon][bin];
                assert!(
                    (actual - expected).abs() <= 2.0 * f64::EPSILON * expected.abs().max(1.0),
                    "second-moment round-trip changed H{} bin {bin}",
                    DIRECT_RETURN_HORIZONS[horizon],
                );
                assert_eq!(
                    loaded.bin_simple_means[horizon][bin].to_bits(),
                    supports.bin_simple_means[horizon][bin].to_bits(),
                    "simple-return mean round-trip changed H{} bin {bin}",
                    DIRECT_RETURN_HORIZONS[horizon],
                );
                assert_eq!(
                    loaded.bin_simple_second_moments[horizon][bin].to_bits(),
                    supports.bin_simple_second_moments[horizon][bin].to_bits(),
                    "simple-return second moment round-trip changed H{} bin {bin}",
                    DIRECT_RETURN_HORIZONS[horizon],
                );
                assert_eq!(
                    loaded.bin_probabilities[horizon][bin].to_bits(),
                    supports.bin_probabilities[horizon][bin].to_bits(),
                );
                assert_eq!(
                    loaded.bin_counts[horizon][bin],
                    supports.bin_counts[horizon][bin],
                );
            }
        }
        let mut json = serde_json::to_value(&supports).unwrap();
        assert_eq!(
            json["format_version"],
            serde_json::json!(DIRECT_RETURN_SUPPORTS_FORMAT_VERSION)
        );
        json["bin_probability_bits"][0][0] = serde_json::json!(f64::NAN.to_bits());
        fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(
            DirectReturnSupports::load(&path).is_err(),
            "non-finite fitted probabilities must be rejected"
        );

        let mut json = serde_json::to_value(&supports).unwrap();
        json["bin_counts"][0][0] = serde_json::json!(supports.bin_counts(0)[0].saturating_add(1));
        fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(
            DirectReturnSupports::load(&path).is_err(),
            "a count law that does not sum to fit rows must be rejected"
        );

        let mut json = serde_json::to_value(&supports).unwrap();
        json["horizons"][0] = serde_json::json!(2);
        fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(DirectReturnSupports::load(&path).is_err());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn tensor_and_host_binning_are_identical() {
        let rows = support_rows();
        let supports = DirectReturnSupports::fit(&rows, provenance(rows.len())).unwrap();
        let values: Vec<f64> = (-12..12)
            .flat_map(|step| {
                (0..DIRECT_RETURN_COUNT).map(move |horizon| {
                    if step == 0 {
                        0.0
                    } else {
                        step as f64 * (horizon + 1) as f64 / 300.0
                    }
                })
            })
            .collect();
        let tensor = Tensor::from_slice(&values).view([-1, DIRECT_RETURN_COUNT as i64]);
        let ids = supports.bin_ids(&tensor);
        for row in 0..ids.size()[0] {
            for horizon in 0..DIRECT_RETURN_COUNT {
                let value = tensor.double_value(&[row, horizon as i64]);
                assert_eq!(
                    ids.int64_value(&[row, horizon as i64]) as usize,
                    supports.bin_of(horizon, value).unwrap()
                );
            }
        }
    }

    #[test]
    fn zero_init_is_uniform_and_scores_log_bin_count() {
        let vs = nn::VarStore::new(Device::Cpu);
        let head = DirectReturnHead::new(&vs.root());
        let belief = Tensor::randn([3, 5, BAR_MODEL_DIM], (Kind::Float, Device::Cpu));
        let adjusted_daily = Tensor::zeros(
            [3, 5, ADJUSTED_DAILY_CONTEXT_FEATURES as i64],
            (Kind::Float, Device::Cpu),
        );
        let logits = head.logits(&belief, &adjusted_daily, DirectReturnContextMode::Enabled);
        assert_eq!(
            logits.size(),
            [3, 5, DIRECT_RETURN_COUNT as i64, NUM_DIRECT_RETURN_BINS]
        );
        assert_eq!(logits.abs().max().double_value(&[]), 0.0);
        let targets = Tensor::zeros(
            [3, 5, DIRECT_RETURN_COUNT as i64],
            (Kind::Int64, Device::Cpu),
        );
        let (mean, per_horizon) = direct_return_ce_from_logits(&logits, &targets);
        let expected = (NUM_DIRECT_RETURN_BINS as f64).ln();
        assert!((mean.double_value(&[]) - expected).abs() < 1e-6);
        for horizon in 0..DIRECT_RETURN_COUNT {
            assert!((per_horizon.double_value(&[horizon as i64]) - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn masked_context_zeros_only_the_daily_features_of_the_same_projection() {
        let vs = nn::VarStore::new(Device::Cpu);
        let mut head = DirectReturnHead::new(&vs.root());
        tch::no_grad(|| {
            let _ = head.projection.ws.zero_();
            let _ = head
                .projection
                .ws
                .narrow(0, 0, 1)
                .narrow(1, BAR_MODEL_DIM, 1)
                .fill_(2.0);
            let _ = head
                .projection
                .bs
                .as_mut()
                .expect("direct-return projection has a bias")
                .zero_();
        });
        let belief = Tensor::ones([2, BAR_MODEL_DIM], (Kind::Float, Device::Cpu));
        let adjusted_daily = Tensor::from_slice(&[3.0f32, -1.0, 2.0, 4.0, 5.0, 6.0])
            .view([1, ADJUSTED_DAILY_CONTEXT_FEATURES as i64])
            .expand([2, ADJUSTED_DAILY_CONTEXT_FEATURES as i64], true);
        let enabled = head.logits(&belief, &adjusted_daily, DirectReturnContextMode::Enabled);
        let masked = head.logits(&belief, &adjusted_daily, DirectReturnContextMode::Masked);
        let explicit_zero = head.logits(
            &belief,
            &Tensor::zeros_like(&adjusted_daily),
            DirectReturnContextMode::Enabled,
        );
        assert_eq!(
            (&masked - explicit_zero).abs().max().double_value(&[]),
            0.0,
            "masked mode must be exactly the enabled head fed six zeros"
        );
        assert_eq!(enabled.double_value(&[0, 0, 0]), 6.0);
        assert_eq!(masked.double_value(&[0, 0, 0]), 0.0);
        assert!(
            (&enabled - &masked).abs().max().double_value(&[]) > 0.0,
            "nonzero causal context must move logits only in the enabled arm"
        );
    }

    #[test]
    fn analytic_moments_match_explicit_probability_sum() {
        let rows = support_rows();
        let supports = DirectReturnSupports::fit(&rows, provenance(rows.len())).unwrap();
        let logits = Tensor::arange(NUM_DIRECT_RETURN_BINS, (Kind::Float, Device::Cpu))
            .view([1, 1, NUM_DIRECT_RETURN_BINS])
            .expand(
                [2, DIRECT_RETURN_COUNT as i64, NUM_DIRECT_RETURN_BINS],
                true,
            )
            / 17.0;
        let (mean, variance) = supports.expectation_variance(&logits);
        let probs = logits.softmax(-1, Kind::Double);
        for row in 0..2 {
            for horizon in 0..DIRECT_RETURN_COUNT {
                let mut expected_mean = 0.0;
                let mut expected_second = 0.0;
                for bin in 0..BINS {
                    let p = probs.double_value(&[row, horizon as i64, bin as i64]);
                    expected_mean += p * supports.bin_means(horizon)[bin];
                    expected_second += p * supports.bin_second_moments(horizon)[bin];
                }
                assert!((mean.double_value(&[row, horizon as i64]) - expected_mean).abs() < 1e-5);
                assert!(
                    (variance.double_value(&[row, horizon as i64])
                        - (expected_second - expected_mean * expected_mean).max(0.0))
                    .abs()
                        < 1e-4
                );
            }
        }
    }

    #[test]
    fn tied_extreme_supports_keep_the_actual_marginal_and_exact_simple_moments() {
        let rows: Vec<[f64; DIRECT_RETURN_COUNT]> = (0..256)
            .map(|row| {
                let value = match row {
                    0 => -20.0,
                    255 => 20.0,
                    _ => 0.0,
                };
                [value; DIRECT_RETURN_COUNT]
            })
            .collect();
        let supports = DirectReturnSupports::fit(&rows, provenance(rows.len())).unwrap();
        for horizon in 0..DIRECT_RETURN_COUNT {
            let zero = supports.zero_bin(horizon);
            assert_eq!(supports.bin_counts(horizon).iter().sum::<u64>(), 256);
            let zero_count = supports.bin_counts(horizon)[zero];
            assert!(zero_count >= 254);
            assert_eq!(
                supports.bin_probabilities(horizon)[zero].to_bits(),
                (zero_count as f64 / 256.0).to_bits()
            );
            assert!(
                supports
                    .bin_probabilities(horizon)
                    .iter()
                    .any(|probability| *probability == 0.0),
                "quantile ties must retain their empty fitted bins"
            );
        }

        let target_bins = Tensor::from_slice(
            &supports
                .zero_bins
                .iter()
                .map(|&bin| bin as i64)
                .collect::<Vec<_>>(),
        )
        .view([1, DIRECT_RETURN_COUNT as i64]);
        let valid = Tensor::ones([1, DIRECT_RETURN_COUNT as i64], (Kind::Bool, Device::Cpu));
        let (ce_sum, count) = supports.marginal_ce_sums(&target_bins, &valid);
        for horizon in 0..DIRECT_RETURN_COUNT {
            let expected = -supports.bin_probabilities(horizon)[supports.zero_bin(horizon)].ln();
            assert!((ce_sum.double_value(&[horizon as i64]) - expected).abs() < 1e-12);
            assert_eq!(count.double_value(&[horizon as i64]), 1.0);
        }

        let empty_bins = Tensor::from_slice(
            &(0..DIRECT_RETURN_COUNT)
                .map(|horizon| {
                    supports
                        .bin_counts(horizon)
                        .iter()
                        .position(|count| *count == 0)
                        .expect("tied fit has an empty bin") as i64
                })
                .collect::<Vec<_>>(),
        )
        .view([1, DIRECT_RETURN_COUNT as i64]);
        let invalid = Tensor::zeros([1, DIRECT_RETURN_COUNT as i64], (Kind::Bool, Device::Cpu));
        let (ignored, ignored_count) = supports.marginal_ce_sums(&empty_bins, &invalid);
        assert_eq!(ignored.abs().max().double_value(&[]), 0.0);
        assert_eq!(ignored_count.abs().max().double_value(&[]), 0.0);

        let logits = Tensor::zeros(
            [1, DIRECT_RETURN_COUNT as i64, NUM_DIRECT_RETURN_BINS],
            (Kind::Float, Device::Cpu),
        );
        let (simple_mean, simple_second) = supports.simple_expectation_second_moment(&logits);
        for horizon in 0..DIRECT_RETURN_COUNT {
            let explicit_mean =
                supports.bin_simple_means(horizon).iter().sum::<f64>() / BINS as f64;
            let explicit_second = supports
                .bin_simple_second_moments(horizon)
                .iter()
                .sum::<f64>()
                / BINS as f64;
            assert!(
                (simple_mean.double_value(&[0, horizon as i64]) - explicit_mean).abs()
                    <= 1e-5 * explicit_mean.abs().max(1.0)
            );
            assert!(
                (simple_second.double_value(&[0, horizon as i64]) - explicit_second).abs()
                    <= 1e-5 * explicit_second.abs().max(1.0)
            );
        }
    }

    #[test]
    fn validation_sums_weight_heldout_rows_and_exclude_the_h100_tail() {
        let rows = support_rows();
        let supports = DirectReturnSupports::fit(&rows, provenance(rows.len())).unwrap();
        let sequence = 101i64;
        let dof = Tensor::zeros([1, sequence, BAR_DOF as i64], (Kind::Float, Device::Cpu));
        let row_valid = Tensor::ones([1, sequence], (Kind::Bool, Device::Cpu));
        let targets = direct_return_targets_with_mask(&dof, &row_valid);
        let realized = targets.values.narrow(1, 0, sequence - 1);
        let valid = targets.valid.narrow(1, 0, sequence - 1);
        let bins = supports.bin_ids(&realized);
        let logits = Tensor::zeros(
            [
                1,
                sequence - 1,
                DIRECT_RETURN_COUNT as i64,
                NUM_DIRECT_RETURN_BINS,
            ],
            (Kind::Float, Device::Cpu),
        );
        let tail =
            direct_return_validation_sums(&supports, &logits, &realized, &bins, &valid).finish();
        assert_eq!(tail.valid_rows[0], 100);
        assert_eq!(tail.valid_rows[5], 1, "only one H100 target is complete");
        assert_eq!(tail.coverage[5], 0.01);

        let chunk = |rows: i64, target_logit: f64| {
            let realized = Tensor::zeros(
                [rows, DIRECT_RETURN_COUNT as i64],
                (Kind::Float, Device::Cpu),
            );
            let bins = supports.bin_ids(&realized);
            let valid = Tensor::ones(
                [rows, DIRECT_RETURN_COUNT as i64],
                (Kind::Bool, Device::Cpu),
            );
            let logits = Tensor::zeros(
                [rows, DIRECT_RETURN_COUNT as i64, NUM_DIRECT_RETURN_BINS],
                (Kind::Float, Device::Cpu),
            );
            for horizon in 0..DIRECT_RETURN_COUNT as i64 {
                let bin = bins.int64_value(&[0, horizon]);
                let _ = logits
                    .narrow(1, horizon, 1)
                    .narrow(2, bin, 1)
                    .fill_(target_logit);
            }
            direct_return_validation_sums(&supports, &logits, &realized, &bins, &valid)
        };
        let one = chunk(1, 0.0).finish();
        let three = chunk(3, 2.0).finish();
        let mut pooled = ValidationSums::default();
        pooled.absorb(chunk(1, 0.0));
        pooled.absorb(chunk(3, 2.0));
        let pooled = pooled.finish();
        for horizon in 0..DIRECT_RETURN_COUNT {
            let expected = (one.model_ce[horizon] + 3.0 * three.model_ce[horizon]) / 4.0;
            assert!((pooled.model_ce[horizon] - expected).abs() < 1e-10);
            assert_eq!(pooled.valid_rows[horizon], 4);
            assert!(
                (pooled.gain[horizon] - (pooled.marginal_ce[horizon] - pooled.model_ce[horizon]))
                    .abs()
                    < 1e-12
            );
        }
    }

    #[test]
    fn validation_calibration_uses_fitted_log_moments_and_nonflat_direction_rows() {
        let rows = support_rows();
        let supports = DirectReturnSupports::fit(&rows, provenance(rows.len())).unwrap();
        let realized_values = [
            0.0, 0.01, -0.02, 0.03, -0.04, 0.05, 0.02, -0.01, 0.04, -0.03, 0.06, -0.05,
        ];
        let realized = Tensor::from_slice(&realized_values).view([2, DIRECT_RETURN_COUNT as i64]);
        let valid = Tensor::ones([2, DIRECT_RETURN_COUNT as i64], (Kind::Bool, Device::Cpu));
        let bins = supports.bin_ids(&realized);
        let logits = Tensor::arange(NUM_DIRECT_RETURN_BINS, (Kind::Float, Device::Cpu))
            .view([1, 1, NUM_DIRECT_RETURN_BINS])
            .expand(
                [2, DIRECT_RETURN_COUNT as i64, NUM_DIRECT_RETURN_BINS],
                true,
            )
            / 29.0;
        let (predicted_mean, predicted_variance) = supports.expectation_variance(&logits);
        let stats =
            direct_return_validation_sums(&supports, &logits, &realized, &bins, &valid).finish();
        for horizon in 0..DIRECT_RETURN_COUNT {
            let mut error_sum = 0.0;
            let mut square_sum = 0.0;
            let mut variance_sum = 0.0;
            let mut directional_hits = 0u64;
            let mut directional_rows = 0u64;
            for row in 0..2 {
                let actual = realized.double_value(&[row, horizon as i64]);
                let predicted = predicted_mean.double_value(&[row, horizon as i64]);
                let error = predicted - actual;
                error_sum += error;
                square_sum += error * error;
                variance_sum += predicted_variance.double_value(&[row, horizon as i64]);
                if actual != 0.0 {
                    directional_rows += 1;
                    directional_hits += if predicted.signum() == actual.signum() {
                        1
                    } else {
                        0
                    };
                }
            }
            assert!((stats.mean_bias[horizon] - error_sum / 2.0).abs() < 1e-9);
            assert!((stats.squared_error_mean[horizon] - square_sum / 2.0).abs() < 1e-9);
            assert!((stats.mean_rmse[horizon] - (square_sum / 2.0).sqrt()).abs() < 1e-9);
            assert!((stats.predicted_variance_mean[horizon] - variance_sum / 2.0).abs() < 1e-9);
            assert_eq!(stats.directional_rows[horizon], directional_rows);
            assert_eq!(
                stats.directional_accuracy[horizon],
                directional_hits as f64 / directional_rows as f64
            );
        }
        assert_eq!(
            stats.directional_rows[0], 1,
            "exactly flat realizations alone are excluded from direction"
        );
    }
}
