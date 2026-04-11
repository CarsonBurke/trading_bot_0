use tch::{Kind, Tensor};

/// Number of bins for the critic value distribution.
pub const NUM_BINS: i64 = 255;

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

fn symlog_tensor(x: &Tensor) -> Tensor {
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
        Self::new(-6.0, 6.0, NUM_BINS, device)
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
        let bin_probs = cdf.narrow(-1, 1, support.size()[0] - 1) - cdf.narrow(-1, 0, support.size()[0] - 1);
        let z = (cdf.narrow(-1, support.size()[0] - 1, 1) - cdf.narrow(-1, 0, 1))
            .clamp_min(1e-10);
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
        let symlog_value = (weights * centers).sum_dim_intlist([-1].as_slice(), false, Kind::Double);
        symexp_tensor(&symlog_value).to_kind(Kind::Float)
    }

    /// Decode logits [batch, NUM_BINS] to scalar values [batch].
    pub fn decode(&self, logits: &Tensor) -> Tensor {
        self.bins_to_scalar_value(logits, true)
    }
}
