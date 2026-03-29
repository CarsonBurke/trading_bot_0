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

pub fn sigmoid_target_weight(latent: &Tensor) -> Tensor {
    latent.sigmoid()
}

pub fn sigmoid_log_det(latent: &Tensor) -> Tensor {
    (latent.log_sigmoid() + (-latent).log_sigmoid()).sum_dim_intlist(
        [-1].as_slice(),
        false,
        Kind::Float,
    )
}

pub fn transformed_action_log_prob(
    latent: &Tensor,
    mean: &Tensor,
    std: &Tensor,
    log_2pi: f64,
) -> Tensor {
    let gaussian = gaussian_log_prob(&(latent - mean), std, log_2pi);
    gaussian - sigmoid_log_det(latent)
}

#[cfg(test)]
mod tests {
    use super::{sigmoid_log_det, sigmoid_target_weight};
    use tch::{Kind, Tensor};

    #[test]
    fn sigmoid_target_weight_maps_zero_to_half() {
        let latent = Tensor::zeros([1, 1], (Kind::Float, tch::Device::Cpu));
        let weight = sigmoid_target_weight(&latent).double_value(&[]);

        assert!((weight - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_log_det_matches_half_allocation_case() {
        let latent = Tensor::zeros([1, 1], (Kind::Float, tch::Device::Cpu));
        let log_det = sigmoid_log_det(&latent).double_value(&[]);
        let expected = 0.25f64.ln();

        assert!((log_det - expected).abs() < 1e-6);
    }
}
