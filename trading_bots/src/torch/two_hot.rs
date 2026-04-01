use tch::{Kind, Tensor};

/// Number of bins for the two-hot value distribution.
pub const NUM_BINS: i64 = 255;

/// Symmetric exponential: sign(x) * (exp(|x|) - 1)
fn symexp(x: f64) -> f64 {
    x.signum() * (x.abs().exp() - 1.0)
}

/// Symmetric logarithm: sign(x) * ln(|x| + 1). Inverse of symexp.
#[allow(unused)]
pub fn symlog(x: f64) -> f64 {
    x.signum() * (x.abs() + 1.0).ln()
}

/// Precomputed bin boundaries in symexp space for two-hot encoding/decoding.
pub struct TwoHotBins {
    /// Bin boundary values after symexp transform, length NUM_BINS
    pub(crate) bin_values: Tensor,
}

impl TwoHotBins {
    pub fn new(log_min: f64, log_max: f64, num_bins: i64, device: tch::Device) -> Self {
        let step = (log_max - log_min) / (num_bins - 1) as f64;
        let values_vec: Vec<f64> = (0..num_bins)
            .map(|i| symexp(log_min + step * i as f64))
            .collect();
        let bin_values = Tensor::from_slice(&values_vec)
            .to_kind(Kind::Float)
            .to_device(device);
        Self { bin_values }
    }

    pub fn default_for(device: tch::Device) -> Self {
        Self::new(-6.0, 6.0, NUM_BINS, device)
    }

    /// Encode scalar values [... ] into two-hot distributions [..., NUM_BINS].
    /// Keeps the operation on-device so clipped-value gradients can flow.
    pub fn encode(&self, values: &Tensor) -> Tensor {
        let values = values.to_kind(Kind::Float);
        let flat_values = values.reshape([-1]);
        let bins = self.bin_values.to_device(values.device());
        let num_bins = bins.size()[0];

        let min_bin = bins.get(0);
        let max_bin = bins.get(num_bins - 1);
        let clamped = flat_values.clamp_tensor(Some(&min_bin), Some(&max_bin));

        let indices = clamped.bucketize(&bins, false, false);
        let left_indices = (&indices - 1).clamp(0, num_bins - 2);
        let right_indices = &left_indices + 1;

        let left_values = bins.gather(0, &left_indices, false);
        let right_values = bins.gather(0, &right_indices, false);
        let total_distance = &right_values - &left_values;

        let right_weight = (&clamped - &left_values) / &total_distance;
        let left_weight = right_weight.ones_like() - &right_weight;

        let encoded = Tensor::zeros(
            [flat_values.size()[0], num_bins],
            (Kind::Float, values.device()),
        )
        .scatter(1, &left_indices.unsqueeze(1), &left_weight.unsqueeze(1))
        .scatter(1, &right_indices.unsqueeze(1), &right_weight.unsqueeze(1));

        let mut out_shape = values.size();
        out_shape.push(num_bins);
        encoded.reshape(out_shape)
    }

    /// Match dreamer4's bins_to_scalar_value: optionally normalize logits first,
    /// then compute the expected scalar value in symexp space.
    pub fn bins_to_scalar_value(&self, logits_or_probs: &Tensor, normalize: bool) -> Tensor {
        let weights = if normalize {
            logits_or_probs.softmax(-1, Kind::Float)
        } else {
            logits_or_probs.shallow_clone()
        }
        .to_kind(Kind::Double);
        let bins = self
            .bin_values
            .to_device(logits_or_probs.device())
            .to_kind(Kind::Double);
        (weights * bins)
            .sum_dim_intlist([-1].as_slice(), false, Kind::Double)
            .to_kind(Kind::Float)
    }

    /// Decode logits [batch, NUM_BINS] to scalar values [batch].
    pub fn decode(&self, logits: &Tensor) -> Tensor {
        self.bins_to_scalar_value(logits, true)
    }
}
