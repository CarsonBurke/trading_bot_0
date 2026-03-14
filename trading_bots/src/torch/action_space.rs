use tch::{Kind, Tensor};

pub fn gaussian_log_prob(diff: &Tensor, std: &Tensor, log_2pi: f64) -> Tensor {
    let var = std.pow_tensor_scalar(2);
    let log_std = std.log();
    let mahal = diff.pow_tensor_scalar(2) / &var;
    let per_dim: Tensor = -(mahal + log_2pi) * 0.5 - &log_std;
    per_dim.sum_dim_intlist([-1].as_slice(), false, Kind::Float)
}

pub fn gaussian_entropy(std: &Tensor, log_2pi: f64) -> Tensor {
    (std.log() + 0.5 * (1.0 + log_2pi)).sum_dim_intlist([-1].as_slice(), false, Kind::Float)
}

pub fn implicit_cash_weights(latent: &Tensor) -> Tensor {
    let batch = latent.size()[0];
    let cash_baseline = Tensor::zeros([batch, 1], (latent.kind(), latent.device()));
    let augmented = Tensor::cat(&[latent, &cash_baseline], 1);
    augmented
        .softmax(-1, Kind::Float)
        .narrow(1, 0, latent.size()[1])
}

pub fn implicit_cash_log_det(latent: &Tensor) -> Tensor {
    let batch = latent.size()[0];
    let cash_baseline = Tensor::zeros([batch, 1], (latent.kind(), latent.device()));
    let augmented = Tensor::cat(&[latent, &cash_baseline], 1);
    augmented
        .log_softmax(-1, Kind::Float)
        .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
}

pub fn transformed_action_log_prob(
    latent: &Tensor,
    mean: &Tensor,
    std: &Tensor,
    log_2pi: f64,
) -> Tensor {
    let gaussian = gaussian_log_prob(&(latent - mean), std, log_2pi);
    gaussian - implicit_cash_log_det(latent)
}

#[cfg(test)]
mod tests {
    use super::{implicit_cash_log_det, implicit_cash_weights};
    use tch::{Kind, Tensor};

    #[test]
    fn implicit_cash_weights_leave_cash_as_residual() {
        let latent = Tensor::from_slice(&[0.0f32, 1.0]).view([1, 2]);
        let ticker_weights = implicit_cash_weights(&latent);
        let ticker_sum = ticker_weights.sum(Kind::Float).double_value(&[]);
        let cash_weight = 1.0 - ticker_sum;

        assert!(ticker_sum > 0.0);
        assert!(cash_weight > 0.0);
        assert!((ticker_sum + cash_weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn implicit_cash_log_det_matches_two_class_case() {
        let latent = Tensor::zeros([1, 1], (Kind::Float, tch::Device::Cpu));
        let log_det = implicit_cash_log_det(&latent).double_value(&[]);
        let expected = 2.0f64 * 0.5f64.ln();

        assert!((log_det - expected).abs() < 1e-6);
    }
}
