use tch::{nn, Kind, Tensor};

/// Number of bins for the critic value distribution.
pub const NUM_BINS: i64 = 255;

/// Symlog-space bounds of the critic value support (legacy symlog scheme used by
/// the direct-PPO `train/` pipeline).
pub const SYMLOG_SUPPORT_MIN: f64 = -3.0;
pub const SYMLOG_SUPPORT_MAX: f64 = 3.0;

const SQRT_2: f64 = std::f64::consts::SQRT_2;
pub(crate) const DIRECT_SIGMA_RATIO: f64 = 0.5;

/// Half-width, in standard deviations, of the standardized-space support for the
/// running-stats scheme. The support spans `[-K, K]` in z-units. K=5 leaves
/// headroom for the heavy-tailed GAE returns (|z| up to 5σ) without clamping,
/// while still leaving each bin narrow (255 bins over 10σ).
pub const STANDARDIZED_STD_RANGE: f64 = 5.0;

/// Sigma-to-bin ratio for the standardized scheme, matching the reference
/// `DEFAULT_SIGMA_TO_BIN_RATIO = 2.0`. With NUM_BINS=255 over `[-5, 5]` the bin
/// width is `10/255 ≈ 0.0392` z-units and `sigma = 2.0 * bin_width ≈ 0.0784`, so
/// a typical normalized return (|z| ~ 0.1–3) has its Gaussian target spread over
/// roughly `6σ / bin_width ≈ 12` bins — never collapsing onto a single center bin.
pub const STANDARDIZED_SIGMA_RATIO: f64 = 2.0;

/// Variance floor guarding `std -> 0`. The reference clamps variance at `1e-5`
/// (a `~3.16e-3` std floor), which is tuned to its ~100-scale targets. Our GAE
/// returns are ~1e-3 with std ~1e-3, so that floor would over-inflate std and
/// re-compress the normalized targets toward the center bin. We floor variance
/// at `1e-12` (std floor `1e-6`) so the guard only binds on genuinely degenerate
/// (near-constant) batches while true return spread normalizes to unit scale.
const VAR_EPS: f64 = 1e-12;

fn symexp_tensor(x: &Tensor) -> Tensor {
    x.sign() * (x.abs().exp() - 1.0)
}

/// Symmetric logarithm: sign(x) * ln(|x| + 1). Inverse of symexp.
#[allow(unused)]
pub fn symlog(x: f64) -> f64 {
    x.signum() * (x.abs() + 1.0).ln()
}

pub(crate) fn symlog_tensor(x: &Tensor) -> Tensor {
    x.sign() * (x.abs() + 1.0).log()
}

/// The scalar-to-support transform used for encoding targets and un-transforming
/// decoded expected values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueScheme {
    /// Support lives in symlog space; decode reverses via symexp. No running
    /// normalization. Used by the direct-PPO `train/` pipeline.
    Symlog,
    /// Support lives in standardized (z-score) space defined by running mean/std
    /// of the target values; decode un-normalizes back to raw units. Mirrors the
    /// reference `HLGaussLossFromRunningStats`. Used by the belief planner.
    Standardized,
}

/// Shared intermediates derived from raw scalar values prior to transform
/// encoding or range analysis: float-cast values, their flattened view, and the
/// device/kind-aligned support with its edge scalars.
struct PreparedValues {
    values: Tensor,
    flat_values: Tensor,
    support: Tensor,
    min_support: Tensor,
    max_support: Tensor,
}

/// Histogram bins for critic targets and decoding.
pub struct HlGaussBins {
    support: Tensor,
    centers: Tensor,
    sigma: f64,
    scheme: ValueScheme,
    /// Running statistics of the target values (Standardized scheme only). These
    /// are non-trainable buffers registered in the planner VarStore so decode is
    /// reproduced exactly across save/resume/infer. Welford aggregate:
    /// `running_mean`, `running_m2` (sum of squared deviations), `running_count`.
    running_mean: Tensor,
    running_m2: Tensor,
    running_count: Tensor,
}

impl HlGaussBins {
    pub fn new(log_min: f64, log_max: f64, num_bins: i64, device: tch::Device) -> Self {
        Self::new_with_sigma_ratio(log_min, log_max, num_bins, DIRECT_SIGMA_RATIO, device)
    }

    pub fn new_with_sigma_ratio(
        log_min: f64,
        log_max: f64,
        num_bins: i64,
        sigma_ratio: f64,
        device: tch::Device,
    ) -> Self {
        assert!(num_bins > 1, "hl-gauss bin count must be greater than one");
        assert!(
            log_min < log_max,
            "hl-gauss support must be strictly increasing"
        );
        assert!(
            sigma_ratio.is_finite() && sigma_ratio > 0.0,
            "hl-gauss sigma ratio must be positive and finite"
        );
        let support = Tensor::linspace(log_min, log_max, num_bins + 1, (Kind::Float, device));
        let centers = (&support.narrow(0, 0, num_bins) + &support.narrow(0, 1, num_bins)) * 0.5;
        let bin_width = (log_max - log_min) / num_bins as f64;
        let sigma = sigma_ratio * bin_width;
        let (running_mean, running_m2, running_count) = standalone_running_stats(device);
        Self {
            support,
            centers,
            sigma,
            scheme: ValueScheme::Symlog,
            running_mean,
            running_m2,
            running_count,
        }
    }

    pub fn default_for(device: tch::Device) -> Self {
        Self::new(SYMLOG_SUPPORT_MIN, SYMLOG_SUPPORT_MAX, NUM_BINS, device)
    }

    /// Build the standardized running-stats scheme with its running-stats buffers
    /// registered as NON-TRAINABLE variables in `path`'s VarStore, so they are
    /// serialized alongside the planner weights and restored on resume/infer.
    /// Support is linear over `[-STANDARDIZED_STD_RANGE, STANDARDIZED_STD_RANGE]`
    /// in z-units. Must be constructed BEFORE the VarStore is loaded so the
    /// buffers exist to receive the persisted stats.
    pub fn planner(path: &nn::Path, device: tch::Device) -> Self {
        let running_mean = path.zeros_no_train("running_mean", &[1]);
        let running_m2 = path.zeros_no_train("running_m2", &[1]);
        let running_count = path.zeros_no_train("running_count", &[1]);
        Self::standardized_with_stats(running_mean, running_m2, running_count, device)
    }

    /// Standardized scheme with standalone (non-persistent) running-stats tensors,
    /// for tests and non-checkpointed use.
    pub fn default_standardized_for(device: tch::Device) -> Self {
        let (running_mean, running_m2, running_count) = standalone_running_stats(device);
        Self::standardized_with_stats(running_mean, running_m2, running_count, device)
    }

    fn standardized_with_stats(
        running_mean: Tensor,
        running_m2: Tensor,
        running_count: Tensor,
        device: tch::Device,
    ) -> Self {
        let support = Tensor::linspace(
            -STANDARDIZED_STD_RANGE,
            STANDARDIZED_STD_RANGE,
            NUM_BINS + 1,
            (Kind::Float, device),
        );
        let centers = (&support.narrow(0, 0, NUM_BINS) + &support.narrow(0, 1, NUM_BINS)) * 0.5;
        let bin_width = 2.0 * STANDARDIZED_STD_RANGE / NUM_BINS as f64;
        let sigma = STANDARDIZED_SIGMA_RATIO * bin_width;
        Self {
            support,
            centers,
            sigma,
            scheme: ValueScheme::Standardized,
            running_mean,
            running_m2,
            running_count,
        }
    }

    pub fn num_bins(&self) -> i64 {
        self.centers.size()[0]
    }

    /// Running mean and std of the target values as `[1]` float tensors on
    /// `device`. Cold start (count == 0) yields mean=0, std=1 so encode/decode
    /// reduce to identity before any target is observed.
    fn running_mean_std(&self, device: tch::Device) -> (Tensor, Tensor) {
        let count = self.running_count.to_device(device).to_kind(Kind::Double);
        let has = count.ge(1.0).to_kind(Kind::Double);
        let n = count.clamp_min(1.0);
        let m2 = self.running_m2.to_device(device).to_kind(Kind::Double);
        // var = has ? m2/n : 1.0
        let var: Tensor = &has * (&m2 / &n) + (1.0 - &has);
        let std = var.clamp_min(VAR_EPS).sqrt().to_kind(Kind::Float);
        let mean = self.running_mean.to_device(device).to_kind(Kind::Float);
        (mean, std)
    }

    /// Update the Welford running statistics from a batch of raw target values.
    /// Uses Chan's parallel merge over ALL elements (not per-batch means) so the
    /// std reflects the true spread of the returns. Buffers are updated in place
    /// under `no_grad`, so the persisted VarStore variables carry the new stats.
    pub fn update_running_stats(&self, targets: &Tensor) {
        assert!(
            self.scheme == ValueScheme::Standardized,
            "running-stats update is only valid for the standardized scheme"
        );
        tch::no_grad(|| {
            let device = self.running_mean.device();
            let flat = targets.reshape([-1]).to_device(device).to_kind(Kind::Double);
            let batch_n = flat.numel() as f64;
            if batch_n == 0.0 {
                return;
            }
            let batch_mean = flat.mean(Kind::Double);
            let batch_m2 = (&flat - &batch_mean).square().sum(Kind::Double);

            let n = self.running_count.double_value(&[]);
            let mean = self.running_mean.to_kind(Kind::Double);
            let m2 = self.running_m2.to_kind(Kind::Double);
            let new_n = n + batch_n;
            let delta = &batch_mean - &mean;
            let mean_new = &mean + &delta * (batch_n / new_n);
            let m2_new = &m2 + &batch_m2 + delta.square() * (n * batch_n / new_n);

            // Update the registered buffers in place via a shallow clone (shared
            // storage), so the VarStore-persisted variables carry the new stats
            // without needing a mutable borrow of `self`.
            self.running_mean
                .shallow_clone()
                .copy_(&mean_new.to_kind(Kind::Float));
            self.running_m2
                .shallow_clone()
                .copy_(&m2_new.to_kind(Kind::Float));
            self.running_count.shallow_clone().copy_(
                &Tensor::from(new_n)
                    .to_kind(Kind::Float)
                    .reshape([1])
                    .to_device(device),
            );
        });
    }

    /// Current running mean/std as host scalars (for diagnostics/tests).
    pub fn running_mean_std_scalars(&self) -> (f64, f64) {
        let (mean, std) = self.running_mean_std(tch::Device::Cpu);
        (mean.double_value(&[0]), std.double_value(&[0]))
    }

    fn prepare(&self, values: &Tensor) -> PreparedValues {
        let values = values.to_kind(Kind::Float);
        let flat_values = values.reshape([-1]);
        let support = self.support.to_device(values.device()).to_kind(Kind::Float);
        let min_support = support.get(0);
        let max_support = support.get(support.size()[0] - 1);
        PreparedValues {
            values,
            flat_values,
            support,
            min_support,
            max_support,
        }
    }

    /// Map raw scalar targets into support space: symlog for the legacy scheme,
    /// z-score standardization for the running-stats scheme.
    fn transform_targets(&self, flat_values: &Tensor) -> Tensor {
        match self.scheme {
            ValueScheme::Symlog => symlog_tensor(flat_values),
            ValueScheme::Standardized => {
                let (mean, std) = self.running_mean_std(flat_values.device());
                (flat_values - &mean) / &std
            }
        }
    }

    pub fn range_stats(&self, values: &Tensor) -> Tensor {
        let p = self.prepare(values);
        let transformed = self.transform_targets(&p.flat_values);
        let below_frac = transformed
            .lt_tensor(&p.min_support)
            .to_kind(Kind::Float)
            .mean(Kind::Float);
        let above_frac = transformed
            .gt_tensor(&p.max_support)
            .to_kind(Kind::Float)
            .mean(Kind::Float);
        let (edge_lo, edge_hi) = match self.scheme {
            ValueScheme::Symlog => {
                (symexp_tensor(&p.min_support), symexp_tensor(&p.max_support))
            }
            ValueScheme::Standardized => {
                let (mean, std) = self.running_mean_std(values.device());
                (
                    &p.min_support * &std + &mean,
                    &p.max_support * &std + &mean,
                )
            }
        };
        Tensor::stack(
            &[
                p.flat_values.min(),
                p.flat_values.max(),
                edge_lo,
                edge_hi,
                below_frac,
                above_frac,
            ],
            0,
        )
    }

    fn encode_with_clamp(&self, values: &Tensor, clamp_to_range: bool) -> Tensor {
        let PreparedValues {
            values,
            flat_values,
            support,
            min_support,
            max_support,
        } = self.prepare(values);
        let mut t = self.transform_targets(&flat_values);
        if clamp_to_range {
            t = t.clamp_tensor(Some(&min_support), Some(&max_support));
        }
        let scaled = (&support - &t.unsqueeze(-1)) / (self.sigma * SQRT_2);
        let cdf = scaled.erf();
        let bin_probs =
            cdf.narrow(-1, 1, support.size()[0] - 1) - cdf.narrow(-1, 0, support.size()[0] - 1);
        let z = (cdf.narrow(-1, support.size()[0] - 1, 1) - cdf.narrow(-1, 0, 1)).clamp_min(1e-10);
        let encoded = &bin_probs / &z;

        let mut out_shape = values.size();
        out_shape.push(self.centers.size()[0]);
        encoded.reshape(out_shape)
    }

    /// Encode scalar values [... ] into normalized hl-gauss target distributions
    /// [..., NUM_BINS] in support space (symlog or standardized per scheme).
    pub fn encode(&self, values: &Tensor) -> Tensor {
        self.encode_with_clamp(values, true)
    }

    /// Encode using the default `hl-gauss-pytorch` semantics: transform the
    /// target, do not clamp to support, then truncate/renormalize by the support.
    pub fn encode_unclamped(&self, values: &Tensor) -> Tensor {
        self.encode_with_clamp(values, false)
    }

    /// Compute E[symexp(z)] for probabilities/logits over symlog-space centers.
    /// Only meaningful for the symlog scheme.
    pub fn bins_to_expected_scalar_value(
        &self,
        logits_or_probs: &Tensor,
        normalize: bool,
    ) -> Tensor {
        let weights = if normalize {
            logits_or_probs.softmax(-1, Kind::Float)
        } else {
            logits_or_probs.shallow_clone()
        }
        .to_kind(Kind::Double);
        let centers = self
            .centers
            .to_device(logits_or_probs.device())
            .to_kind(Kind::Double);
        (weights * symexp_tensor(&centers))
            .sum_dim_intlist([-1].as_slice(), false, Kind::Double)
            .to_kind(Kind::Float)
    }

    pub fn decode_mean_and_variance(&self, logits: &Tensor) -> (Tensor, Tensor) {
        let probs = logits.softmax(-1, Kind::Float).to_kind(Kind::Double);
        let centers = self
            .centers
            .to_device(logits.device())
            .to_kind(Kind::Double);
        let raw_centers = symexp_tensor(&centers);
        let mean = (&probs * &raw_centers).sum_dim_intlist([-1].as_slice(), false, Kind::Double);
        let diff = raw_centers - mean.unsqueeze(-1);
        let variance = (probs * diff.pow_tensor_scalar(2.0)).sum_dim_intlist(
            [-1].as_slice(),
            false,
            Kind::Double,
        );
        (mean.to_kind(Kind::Float), variance.to_kind(Kind::Float))
    }

    pub fn decode_reference_mean_and_variance(&self, logits: &Tensor) -> (Tensor, Tensor) {
        let probs = logits.softmax(-1, Kind::Float).to_kind(Kind::Double);
        let centers = self
            .centers
            .to_device(logits.device())
            .to_kind(Kind::Double);
        let symlog_mean = (&probs * &centers).sum_dim_intlist([-1].as_slice(), false, Kind::Double);
        let mean = symexp_tensor(&symlog_mean);
        let raw_centers = symexp_tensor(&centers);
        let diff = raw_centers - mean.unsqueeze(-1);
        let variance = (probs * diff.pow_tensor_scalar(2.0)).sum_dim_intlist(
            [-1].as_slice(),
            false,
            Kind::Double,
        );
        (mean.to_kind(Kind::Float), variance.to_kind(Kind::Float))
    }

    /// Decode logits [batch, NUM_BINS] to raw scalar expected values [batch].
    /// Symlog: `E[symexp(centers)]`. Standardized: `E[centers] * std + mean`,
    /// un-normalizing the expected z-score back to raw return units.
    pub fn decode(&self, logits: &Tensor) -> Tensor {
        match self.scheme {
            ValueScheme::Symlog => self.bins_to_expected_scalar_value(logits, true),
            ValueScheme::Standardized => {
                let probs = logits.softmax(-1, Kind::Float).to_kind(Kind::Double);
                let centers = self
                    .centers
                    .to_device(logits.device())
                    .to_kind(Kind::Double);
                let expected_z =
                    (probs * centers).sum_dim_intlist([-1].as_slice(), false, Kind::Double);
                let (mean, std) = self.running_mean_std(logits.device());
                (expected_z * std.to_kind(Kind::Double) + mean.to_kind(Kind::Double))
                    .to_kind(Kind::Float)
            }
        }
    }
}

fn standalone_running_stats(device: tch::Device) -> (Tensor, Tensor, Tensor) {
    (
        Tensor::zeros([1], (Kind::Float, device)),
        Tensor::zeros([1], (Kind::Float, device)),
        Tensor::zeros([1], (Kind::Float, device)),
    )
}

#[cfg(test)]
mod tests {
    use tch::{nn, Device, Kind, Tensor};

    use super::{symexp_tensor, symlog, HlGaussBins, NUM_BINS};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn direct_decode(bins: &HlGaussBins, encoded: &Tensor) -> Tensor {
        bins.bins_to_expected_scalar_value(encoded, false)
    }

    fn symexp_scalar(x: f64) -> f64 {
        x.signum() * (x.abs().exp() - 1.0)
    }

    #[test]
    fn symlog_properties() {
        assert!(approx_eq(symlog(0.0), 0.0, 1e-10));
        assert!(symlog(1.0) > 0.0);
        assert!(symlog(-1.0) < 0.0);
        assert!(approx_eq(symlog(1.0), -symlog(-1.0), 1e-10));
        assert!(approx_eq(symlog(1.0), (2.0f64).ln(), 1e-10));
    }

    #[test]
    fn encode_shape() {
        let bins = HlGaussBins::new(-5.0, 5.0, 31, tch::Device::Cpu);
        let values = Tensor::zeros([4], (Kind::Float, tch::Device::Cpu));
        let encoded = bins.encode(&values);
        assert_eq!(encoded.size(), vec![4, 31]);
    }

    #[test]
    fn decode_shape() {
        let bins = HlGaussBins::new(-5.0, 5.0, 31, tch::Device::Cpu);
        let logits = Tensor::zeros([4, 31], (Kind::Float, tch::Device::Cpu));
        let decoded = bins.decode(&logits);
        assert_eq!(decoded.size(), vec![4]);
    }

    #[test]
    fn decode_mean_and_variance_shapes_are_nonnegative() {
        let bins = HlGaussBins::new_with_sigma_ratio(-5.0, 5.0, 31, 2.0, tch::Device::Cpu);
        let logits = Tensor::zeros([2, 3, 31], (Kind::Float, tch::Device::Cpu));
        let (mean, variance) = bins.decode_mean_and_variance(&logits);
        assert_eq!(mean.size(), vec![2, 3]);
        assert_eq!(variance.size(), vec![2, 3]);
        assert!(variance.min().double_value(&[]) >= 0.0);
    }

    #[test]
    fn reference_decode_applies_inverse_after_expected_transformed_value() {
        let bins = HlGaussBins::new(-3.0, 3.0, 21, tch::Device::Cpu);
        let logits = Tensor::full([1, 21], -100.0, (Kind::Float, tch::Device::Cpu));
        let _ = logits.get(0).get(10).fill_(0.0);
        let _ = logits.get(0).get(20).fill_(0.0);
        let expected_raw = bins.decode_mean_and_variance(&logits).0.double_value(&[0]);
        let reference = bins
            .decode_reference_mean_and_variance(&logits)
            .0
            .double_value(&[0]);

        assert!(
            (expected_raw - reference).abs() > 1e-3,
            "broad distributions should distinguish E[symexp(z)] from symexp(E[z])"
        );
    }

    #[test]
    fn distribution_sums_to_one() {
        let bins = HlGaussBins::default_for(tch::Device::Cpu);
        let values = Tensor::from_slice(&[0.0f32, 1.5, -3.7, 50.0]);
        let encoded = bins.encode(&values);

        for i in 0..4 {
            let row_sum = encoded.get(i).sum(Kind::Float).double_value(&[]);
            assert!(
                approx_eq(row_sum, 1.0, 1e-5),
                "row {i} sums to {row_sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn weights_are_nonnegative() {
        let bins = HlGaussBins::default_for(tch::Device::Cpu);
        let values = Tensor::randn([64], (Kind::Float, tch::Device::Cpu)) * 50.0;
        let encoded = bins.encode(&values);
        let min_val = encoded.min().double_value(&[]);
        assert!(
            min_val >= -1e-7,
            "encoded contains negative weight: {min_val}"
        );
    }

    #[test]
    fn hl_gauss_spreads_mass_over_multiple_bins() {
        let bins = HlGaussBins::default_for(tch::Device::Cpu);
        let values = Tensor::from_slice(&[0.25f32]);
        let encoded = bins.encode(&values);
        let nonzero_count = encoded.get(0).gt(1e-6).sum(Kind::Float).int64_value(&[]);
        assert!(
            nonzero_count > 2,
            "hl-gauss target should spread mass beyond two bins, got {nonzero_count}"
        );
    }

    #[test]
    fn roundtrip_is_reasonable_inside_support() {
        let bins = HlGaussBins::new(-5.0, 5.0, 51, tch::Device::Cpu);
        let values = Tensor::from_slice(&[-10.0f32, -3.0, -0.5, 0.0, 0.5, 3.0, 10.0]);
        let encoded = bins.encode(&values);
        let decoded = direct_decode(&bins, &encoded);

        let max_diff = (&decoded - &values).abs().max().double_value(&[]);
        assert!(
            max_diff < 0.15,
            "hl-gauss roundtrip max error {max_diff} exceeds tolerance"
        );
    }

    #[test]
    fn roundtrip_via_log_decode_is_reasonable() {
        let bins = HlGaussBins::default_for(tch::Device::Cpu);
        let values = Tensor::from_slice(&[0.0f32, 1.5, -1.5, 0.5, -0.5]);
        let encoded = bins.encode(&values);
        let logits = encoded.clamp_min(1e-30).log();
        let decoded = bins.decode(&logits);

        let max_diff = (&decoded - &values).abs().max().double_value(&[]);
        assert!(
            max_diff < 0.05,
            "log-roundtrip max error {max_diff} exceeds tolerance"
        );
    }

    #[test]
    fn bins_to_expected_scalar_value_matches_normalize_flag() {
        let bins = HlGaussBins::new(-3.0, 3.0, 21, tch::Device::Cpu);
        let values = Tensor::from_slice(&[-1.5f32, 0.0, 2.25]);
        let encoded = bins.encode(&values);

        let direct = bins.bins_to_expected_scalar_value(&encoded, false);
        let logits = encoded.clamp_min(1e-30).log();
        let normalized = bins.bins_to_expected_scalar_value(&logits, true);

        let max_diff = (&direct - &normalized).abs().max().double_value(&[]);
        assert!(
            max_diff < 1e-5,
            "normalize flag mismatch, max diff {max_diff}"
        );
    }

    #[test]
    fn decode_consumes_expected_raw_scalar_not_transformed_mean() {
        let bins = HlGaussBins::new(-3.0, 3.0, 21, tch::Device::Cpu);
        let logits = Tensor::full([1, 21], -100.0, (Kind::Float, tch::Device::Cpu));
        let _ = logits.get(0).get(10).fill_(0.0);
        let _ = logits.get(0).get(20).fill_(0.0);

        let decoded = bins.decode(&logits).double_value(&[0]);
        let transformed_mean_decode = symexp_scalar((0.0 + (3.0 - 3.0 / 21.0)) * 0.5);

        assert!(
            decoded > transformed_mean_decode,
            "expected raw decode should preserve high-bin mass: expected > {transformed_mean_decode}, got {decoded}"
        );
    }

    #[test]
    fn cross_entropy_of_matching_distribution_is_lower() {
        let bins = HlGaussBins::default_for(tch::Device::Cpu);
        let value = Tensor::from_slice(&[0.5f32]);
        let target = bins.encode(&value);

        let matching_logits = target.clamp_min(1e-30).log();
        let wrong_value = Tensor::from_slice(&[-1.5f32]);
        let wrong_target = bins.encode(&wrong_value);
        let wrong_logits = wrong_target.clamp_min(1e-30).log();

        let ce_match = -(&target * matching_logits.log_softmax(-1, Kind::Float))
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
            .double_value(&[0]);
        let ce_wrong = -(&target * wrong_logits.log_softmax(-1, Kind::Float))
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
            .double_value(&[0]);

        assert!(
            ce_match < ce_wrong,
            "matching distribution should have lower CE ({ce_match}) than wrong ({ce_wrong})"
        );
    }

    #[test]
    fn bins_are_monotonically_increasing() {
        let bins = HlGaussBins::default_for(tch::Device::Cpu);
        let bin_values = symexp_tensor(&bins.centers);
        let n = bin_values.size()[0];
        for i in 1..n {
            let prev = bin_values.get(i - 1).double_value(&[]);
            let curr = bin_values.get(i).double_value(&[]);
            assert!(
                curr > prev,
                "bin {i} ({curr}) not greater than bin {} ({prev})",
                i - 1
            );
        }
    }

    #[test]
    fn bins_are_symmetric_around_zero() {
        let bins = HlGaussBins::default_for(tch::Device::Cpu);
        let bin_values = symexp_tensor(&bins.centers);
        let n = bin_values.size()[0];
        let first = bin_values.get(0).double_value(&[]);
        let last = bin_values.get(n - 1).double_value(&[]);
        let center = bin_values.get(n / 2).double_value(&[]);
        assert!(
            approx_eq(first, -last, 1e-4),
            "bins not symmetric: first={first}, last={last}"
        );
        assert!(
            approx_eq(center, 0.0, 1e-6),
            "center bin should be 0.0, got {center}"
        );
    }

    #[test]
    fn out_of_support_targets_clamp_to_edge_targets() {
        let bins = HlGaussBins::new(-3.0, 3.0, 21, tch::Device::Cpu);
        let far_positive = Tensor::from_slice(&[9999.0f32]);
        let max_edge = Tensor::from_slice(&[symexp_scalar(3.0) as f32]);
        let far_negative = Tensor::from_slice(&[-9999.0f32]);
        let min_edge = Tensor::from_slice(&[symexp_scalar(-3.0) as f32]);

        let pos_diff = (&bins.encode(&far_positive) - &bins.encode(&max_edge))
            .abs()
            .max()
            .double_value(&[]);
        let neg_diff = (&bins.encode(&far_negative) - &bins.encode(&min_edge))
            .abs()
            .max()
            .double_value(&[]);

        assert!(pos_diff < 1e-6, "positive clamp mismatch: {pos_diff}");
        assert!(neg_diff < 1e-6, "negative clamp mismatch: {neg_diff}");
    }

    #[test]
    fn unclamped_encoding_keeps_out_of_range_targets_outside_support() {
        let bins = HlGaussBins::new(-3.0, 3.0, 21, tch::Device::Cpu);
        let far_positive = Tensor::from_slice(&[9999.0f32]);
        let max_edge = Tensor::from_slice(&[symexp_scalar(3.0) as f32]);

        let clamped = bins.encode(&far_positive);
        let unclamped = bins.encode_unclamped(&far_positive);
        let max_edge_encoded = bins.encode(&max_edge);
        let clamped_diff = (&clamped - &max_edge_encoded).abs().max().double_value(&[]);
        let unclamped_diff = (&unclamped - &max_edge_encoded)
            .abs()
            .max()
            .double_value(&[]);

        assert!(clamped_diff < 1e-6, "clamped path mismatch: {clamped_diff}");
        assert!(
            unclamped_diff > 1e-4,
            "unclamped path should not move the target center to the boundary"
        );
    }

    // ---- standardized running-stats scheme ----

    #[test]
    fn standardized_cold_start_is_identity() {
        let bins = HlGaussBins::default_standardized_for(Device::Cpu);
        let (mean, std) = bins.running_mean_std_scalars();
        assert!(approx_eq(mean, 0.0, 1e-9), "cold-start mean {mean}");
        assert!(approx_eq(std, 1.0, 1e-9), "cold-start std {std}");
        // With mean=0/std=1, decode of a symmetric distribution is ~0.
        let logits = Tensor::zeros([2, NUM_BINS], (Kind::Float, Device::Cpu));
        let decoded = bins.decode(&logits);
        assert!(decoded.abs().max().double_value(&[]) < 1e-4);
    }

    #[test]
    fn standardized_small_returns_span_many_bins_after_stats() {
        // GAE-scale returns: ~1e-3 spread. Before running stats the raw z-scores
        // would be ~1e-3 and collapse onto the center bin. After updating stats,
        // targets must standardize to O(1) and vary per sample across many bins.
        let bins = HlGaussBins::default_standardized_for(Device::Cpu);
        let returns = Tensor::from_slice(&[
            -2.0e-3f32, -1.0e-3, -3.0e-4, 0.0, 4.0e-4, 1.1e-3, 2.3e-3, 3.0e-3,
        ]);
        bins.update_running_stats(&returns);

        let (mean, std) = bins.running_mean_std_scalars();
        assert!(std > 1e-4 && std < 1e-1, "return std standardized to {std}");

        let encoded = bins.encode(&returns);
        // Each sample must spread mass over many bins (not a single center bin).
        for i in 0..returns.size()[0] {
            let nonzero = encoded.get(i).gt(1e-6).sum(Kind::Float).int64_value(&[]);
            assert!(
                nonzero > 4,
                "sample {i} spread over only {nonzero} bins after standardization"
            );
        }
        // Peak (argmax) bins must differ across samples -> per-sample-varying
        // targets rather than everything collapsing to the center bin.
        let argmax = encoded.argmax(-1, false);
        let unique = argmax.unique_dim(0, true, false, false).0.size()[0];
        assert!(
            unique >= 5,
            "standardized targets collapsed: only {unique} distinct peak bins"
        );
        let center = (NUM_BINS - 1) / 2;
        let spread = (&argmax - center).abs().max().int64_value(&[]);
        assert!(spread > 3, "peak bins clustered at center (max offset {spread})");
        let _ = mean;
    }

    #[test]
    fn standardized_decode_unnormalizes_to_raw_units() {
        // A distribution peaked near z=+1 should decode to ~ mean + 1*std in raw
        // units. Verify decode inverts the standardize transform used by encode.
        let bins = HlGaussBins::default_standardized_for(Device::Cpu);
        let returns = Tensor::from_slice(&[
            1.0e-3f32, 2.0e-3, -1.0e-3, 5.0e-4, -2.0e-3, 3.0e-4, 1.5e-3, -5.0e-4,
        ]);
        bins.update_running_stats(&returns);
        let (mean, std) = bins.running_mean_std_scalars();

        // Encode a raw target, log it to logits, decode -> should recover raw.
        let target = Tensor::from_slice(&[mean as f32 + std as f32]);
        let encoded = bins.encode(&target);
        let logits = encoded.clamp_min(1e-30).log();
        let decoded = bins.decode(&logits).double_value(&[0]);
        assert!(
            (decoded - (mean + std)).abs() < 5e-2 * std.max(1e-3),
            "decode {decoded} did not recover raw target {}",
            mean + std
        );
    }

    #[test]
    fn standardized_welford_matches_batch_statistics() {
        let bins = HlGaussBins::default_standardized_for(Device::Cpu);
        let a = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4]);
        let b = Tensor::from_slice(&[-0.5f32, 0.0, 0.25]);
        bins.update_running_stats(&a);
        bins.update_running_stats(&b);
        let (mean, std) = bins.running_mean_std_scalars();

        let all = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4, -0.5, 0.0, 0.25]);
        let ref_mean = all.mean(Kind::Double).double_value(&[]);
        let ref_var = (&all - all.mean(Kind::Double))
            .square()
            .mean(Kind::Double)
            .double_value(&[]);
        assert!(approx_eq(mean, ref_mean, 1e-5), "mean {mean} vs {ref_mean}");
        assert!(
            approx_eq(std, ref_var.sqrt(), 1e-5),
            "std {std} vs {}",
            ref_var.sqrt()
        );
    }

    #[test]
    fn standardized_running_stats_survive_save_load_roundtrip() {
        // The running stats are part of the value function's definition; they
        // must persist with the VarStore weights and restore exactly on load.
        let dir = std::env::temp_dir().join(format!(
            "hl_gauss_roundtrip_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stats.ot");

        let saved_stats = {
            let vs = nn::VarStore::new(Device::Cpu);
            let bins = HlGaussBins::planner(&(vs.root() / "value_running_stats"), Device::Cpu);
            let returns = Tensor::from_slice(&[
                -2.0e-3f32, -1.0e-3, 3.0e-4, 1.1e-3, 2.3e-3, 4.0e-4, -5.0e-4, 9.0e-4,
            ]);
            bins.update_running_stats(&returns);
            vs.save(&path).unwrap();
            bins.running_mean_std_scalars()
        };

        let mut loaded_vs = nn::VarStore::new(Device::Cpu);
        let loaded_bins =
            HlGaussBins::planner(&(loaded_vs.root() / "value_running_stats"), Device::Cpu);
        loaded_vs.load(&path).unwrap();
        let loaded_stats = loaded_bins.running_mean_std_scalars();

        assert!(
            approx_eq(saved_stats.0, loaded_stats.0, 1e-6),
            "mean not preserved: {} vs {}",
            saved_stats.0,
            loaded_stats.0
        );
        assert!(
            approx_eq(saved_stats.1, loaded_stats.1, 1e-6),
            "std not preserved: {} vs {}",
            saved_stats.1,
            loaded_stats.1
        );

        // A decode with the loaded stats must equal a decode with the saved bins.
        let logits = Tensor::randn([3, NUM_BINS], (Kind::Float, Device::Cpu));
        let reloaded_decode = loaded_bins.decode(&logits);
        assert!(reloaded_decode.isfinite().all().int64_value(&[]) != 0);

        std::fs::remove_dir_all(&dir).ok();
    }
}
