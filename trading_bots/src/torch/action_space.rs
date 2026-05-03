use tch::{Kind, Tensor};

pub const SQUASHED_GAUSSIAN_STD_MIN: f64 = 1e-3;
const LOG_2: f64 = 0.6931471805599453;

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

pub fn gaussian_std_from_log_var(log_var: &Tensor) -> Tensor {
    (log_var * 0.5).exp() + SQUASHED_GAUSSIAN_STD_MIN
}

pub fn tanh_target_weight(latent: &Tensor) -> Tensor {
    (latent.tanh() + 1.0) * 0.5
}

pub fn sample_squashed_gaussian_action(mean: &Tensor, std: &Tensor) -> (Tensor, Tensor) {
    let latent = mean + mean.randn_like() * std;
    let target_weight = tanh_target_weight(&latent);
    (latent, target_weight)
}

pub fn tanh_unit_log_det(latent: &Tensor) -> Tensor {
    let log_one_minus_tanh_sq = ((latent * -2.0).softplus() + latent - LOG_2) * -2.0;
    log_one_minus_tanh_sq.sum_dim_intlist([-1].as_slice(), false, Kind::Float)
}

pub fn squashed_gaussian_log_prob(
    latent: &Tensor,
    mean: &Tensor,
    std: &Tensor,
    log_2pi: f64,
) -> Tensor {
    gaussian_log_prob(&(latent - mean), std, log_2pi) - tanh_unit_log_det(latent)
}

#[cfg(test)]
mod tests {
    use super::{
        gaussian_std_from_log_var, squashed_gaussian_log_prob, tanh_target_weight,
        tanh_unit_log_det,
    };
    use tch::{Kind, Tensor};

    #[test]
    fn tanh_target_weight_maps_zero_to_half() {
        let latent = Tensor::zeros([1, 1], (Kind::Float, tch::Device::Cpu));
        let weight = tanh_target_weight(&latent).double_value(&[]);

        assert!((weight - 0.5).abs() < 1e-6);
    }

    #[test]
    fn tanh_unit_log_det_drops_ratio_invariant_affine_constant() {
        let latent = Tensor::zeros([1, 1], (Kind::Float, tch::Device::Cpu));
        let log_det = tanh_unit_log_det(&latent).double_value(&[0]);

        assert!(log_det.abs() < 1e-6);
    }

    #[test]
    fn gaussian_std_from_log_var_uses_dreamer_style_variance() {
        let log_var = Tensor::from_slice(&[-100.0f32, 0.0, 1.0]).view([1, 3]);
        let std = gaussian_std_from_log_var(&log_var);

        assert!((std.double_value(&[0, 0]) - 1e-3).abs() < 1e-6);
        assert!((std.double_value(&[0, 1]) - (1.0 + 1e-3)).abs() < 1e-6);
        assert!((std.double_value(&[0, 2]) - (0.5f64.exp() + 1e-3)).abs() < 1e-6);
    }

    #[test]
    fn squashed_gaussian_log_prob_matches_manual_sum() {
        let latent = Tensor::from_slice(&[0.2f32, -0.4]).view([1, 2]);
        let mean = Tensor::from_slice(&[0.1f32, -0.1]).view([1, 2]);
        let std = Tensor::from_slice(&[0.9f32, 1.2]).view([1, 2]);
        let total = squashed_gaussian_log_prob(&latent, &mean, &std, 1.8378770664093453);
        let expected = [(0.2f64, 0.1f64, 0.9f64), (-0.4f64, -0.1f64, 1.2f64)]
            .into_iter()
            .map(|(z, mu, sigma)| {
                let diff = z - mu;
                let gaussian = -0.5 * ((diff / sigma).powi(2) + (2.0 * core::f64::consts::PI).ln())
                    - sigma.ln();
                let squash_correction = 2.0 * (2.0f64.ln() - z - (-2.0 * z).exp().ln_1p());
                gaussian - squash_correction
            })
            .sum::<f64>();

        assert!((total.double_value(&[0]) - expected).abs() < 1e-6);
    }
}
