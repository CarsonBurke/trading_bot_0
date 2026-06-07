use tch::{Kind, Tensor};

/// Number of bins for the critic value distribution.
pub const NUM_BINS: i64 = 255;

/// Symlog-space bounds of the critic value support.
pub const SYMLOG_SUPPORT_MIN: f64 = -3.0;
pub const SYMLOG_SUPPORT_MAX: f64 = 3.0;

const SQRT_2: f64 = std::f64::consts::SQRT_2;
const SIGMA_RATIO: f64 = 0.5;

/// Symmetric exponential: sign(x) * (exp(|x|) - 1)
fn symexp(x: f64) -> f64 {
    x.signum() * (x.abs().exp() - 1.0)
}

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

/// Histogram bins for critic targets and decoding.
pub struct HlGaussBins {
    /// Bin centers in raw value space, kept for debug/tests.
    pub(crate) bin_values: Tensor,
    support: Tensor,
    centers: Tensor,
    sigma: f64,
}

impl HlGaussBins {
    pub fn new(log_min: f64, log_max: f64, num_bins: i64, device: tch::Device) -> Self {
        let support = Tensor::linspace(log_min, log_max, num_bins + 1, (Kind::Float, device));
        let centers = (&support.narrow(0, 0, num_bins) + &support.narrow(0, 1, num_bins)) * 0.5;
        let bin_values = symexp_tensor(&centers);
        let bin_width = (log_max - log_min) / num_bins as f64;
        let sigma = SIGMA_RATIO * bin_width;
        Self {
            bin_values,
            support,
            centers,
            sigma,
        }
    }

    pub fn default_for(device: tch::Device) -> Self {
        Self::new(SYMLOG_SUPPORT_MIN, SYMLOG_SUPPORT_MAX, NUM_BINS, device)
    }

    pub fn range_stats(&self, values: &Tensor) -> Tensor {
        let values = values.to_kind(Kind::Float);
        let flat_values = values.reshape([-1]);
        let support = self.support.to_device(values.device()).to_kind(Kind::Float);
        let min_support = support.get(0);
        let max_support = support.get(support.size()[0] - 1);
        let symlog_values = symlog_tensor(&flat_values);
        let below_frac = symlog_values
            .lt_tensor(&min_support)
            .to_kind(Kind::Float)
            .mean(Kind::Float);
        let above_frac = symlog_values
            .gt_tensor(&max_support)
            .to_kind(Kind::Float)
            .mean(Kind::Float);
        Tensor::stack(
            &[
                flat_values.min(),
                flat_values.max(),
                symexp_tensor(&min_support),
                symexp_tensor(&max_support),
                below_frac,
                above_frac,
            ],
            0,
        )
    }

    /// Encode scalar values [... ] into normalized hl-gauss target distributions
    /// [..., NUM_BINS] in symlog space.
    pub fn encode(&self, values: &Tensor) -> Tensor {
        let values = values.to_kind(Kind::Float);
        let flat_values = values.reshape([-1]);
        let support = self.support.to_device(values.device()).to_kind(Kind::Float);
        let min_support = support.get(0);
        let max_support = support.get(support.size()[0] - 1);
        let t = symlog_tensor(&flat_values).clamp_tensor(Some(&min_support), Some(&max_support));
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

    /// Compute the expected scalar value. Probabilities/logits live on symlog-space
    /// centers; the expectation is mapped back with symexp.
    pub fn bins_to_scalar_value(&self, logits_or_probs: &Tensor, normalize: bool) -> Tensor {
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
        let symlog_value =
            (weights * centers).sum_dim_intlist([-1].as_slice(), false, Kind::Double);
        symexp_tensor(&symlog_value).to_kind(Kind::Float)
    }

    /// Decode logits [batch, NUM_BINS] to scalar values [batch].
    pub fn decode(&self, logits: &Tensor) -> Tensor {
        self.bins_to_scalar_value(logits, true)
    }
}

#[cfg(test)]
mod tests {
    use tch::{Kind, Tensor};

    use super::{symlog, HlGaussBins};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn direct_decode(bins: &HlGaussBins, encoded: &Tensor) -> Tensor {
        bins.bins_to_scalar_value(encoded, false)
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
    fn bins_to_scalar_value_matches_normalize_flag() {
        let bins = HlGaussBins::new(-3.0, 3.0, 21, tch::Device::Cpu);
        let values = Tensor::from_slice(&[-1.5f32, 0.0, 2.25]);
        let encoded = bins.encode(&values);

        let direct = bins.bins_to_scalar_value(&encoded, false);
        let logits = encoded.clamp_min(1e-30).log();
        let normalized = bins.bins_to_scalar_value(&logits, true);

        let max_diff = (&direct - &normalized).abs().max().double_value(&[]);
        assert!(
            max_diff < 1e-5,
            "normalize flag mismatch, max diff {max_diff}"
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
        let n = bins.bin_values.size()[0];
        for i in 1..n {
            let prev = bins.bin_values.get(i - 1).double_value(&[]);
            let curr = bins.bin_values.get(i).double_value(&[]);
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
        let n = bins.bin_values.size()[0];
        let first = bins.bin_values.get(0).double_value(&[]);
        let last = bins.bin_values.get(n - 1).double_value(&[]);
        let center = bins.bin_values.get(n / 2).double_value(&[]);
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
}
