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
    bin_values: Tensor,
    /// Raw f64 bin values for CPU-side encode
    values_vec: Vec<f64>,
}

impl TwoHotBins {
    pub fn new(log_min: f64, log_max: f64, num_bins: i64, device: tch::Device) -> Self {
        let step = (log_max - log_min) / (num_bins - 1) as f64;
        let values_vec: Vec<f64> = (0..num_bins)
            .map(|i| symexp(log_min + step * i as f64))
            .collect();
        let bin_values =
            Tensor::from_slice(&values_vec).to_kind(Kind::Float).to_device(device);
        Self {
            bin_values,
            values_vec,
        }
    }

    pub fn default_for(device: tch::Device) -> Self {
        Self::new(-20.0, 20.0, NUM_BINS, device)
    }

    /// Encode scalar values [batch] into two-hot distributions [batch, NUM_BINS].
    /// Done on CPU for simplicity (called infrequently relative to forward pass).
    pub fn encode(&self, values: &Tensor) -> Tensor {
        let device = values.device();
        let values_f64 = values.to_kind(Kind::Double).to_device(tch::Device::Cpu);
        let scalars: Vec<f64> = Vec::try_from(&values_f64).unwrap();
        let num_bins = self.values_vec.len();
        let batch = scalars.len();
        let min_val = self.values_vec[0];
        let max_val = self.values_vec[num_bins - 1];

        let mut encoded = vec![0.0f32; batch * num_bins];

        for (b, &val) in scalars.iter().enumerate() {
            let val = val.clamp(min_val, max_val);

            // Binary search for right bin boundary
            let right = match self
                .values_vec
                .binary_search_by(|v| v.partial_cmp(&val).unwrap())
            {
                Ok(i) => i.min(num_bins - 1),
                Err(i) => i.min(num_bins - 1),
            };
            let left = if right == 0 { 0 } else { right - 1 };

            if left == right {
                encoded[b * num_bins + left] = 1.0;
            } else {
                let left_val = self.values_vec[left];
                let right_val = self.values_vec[right];
                let total = right_val - left_val;

                if total.abs() < 1e-10 {
                    encoded[b * num_bins + left] = 1.0;
                } else {
                    let right_weight = ((val - left_val) / total) as f32;
                    let left_weight = 1.0 - right_weight;
                    encoded[b * num_bins + left] = left_weight;
                    encoded[b * num_bins + right] = right_weight;
                }
            }
        }

        Tensor::from_slice(&encoded)
            .reshape([batch as i64, num_bins as i64])
            .to_device(device)
    }

    /// Decode logits [batch, NUM_BINS] to scalar values [batch].
    /// Accumulates in f64 to avoid f32 summation bias over large symexp bins.
    pub fn decode(&self, logits: &Tensor) -> Tensor {
        let probs = logits.softmax(-1, Kind::Float).to_kind(Kind::Double);
        let bins = self.bin_values.unsqueeze(0).to_kind(Kind::Double);
        (probs * bins).sum_dim_intlist([-1].as_slice(), false, Kind::Double).to_kind(Kind::Float)
    }
}
