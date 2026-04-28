use tch::{Kind, Tensor};

pub const BETA_SAMPLE_EPS: f64 = 1e-7;
pub const BETA_ALPHA_MAX: f64 = 20.0;

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

pub fn gaussian_kl_old_new_from_log_var(
    old_mean: &Tensor,
    old_log_var: &Tensor,
    new_mean: &Tensor,
    new_log_var: &Tensor,
) -> Tensor {
    let old_var = old_log_var.exp();
    let new_var = new_log_var.exp();
    let mean_delta_sq = (old_mean - new_mean).pow_tensor_scalar(2);
    let per_dim: Tensor =
        (new_log_var - old_log_var + (old_var + mean_delta_sq) / new_var - 1.0) * 0.5;
    per_dim.sum_dim_intlist([-1].as_slice(), false, Kind::Float)
}

pub fn sigmoid_target_weight(latent: &Tensor) -> Tensor {
    latent.sigmoid()
}

pub fn beta_concentration_from_head(head: &Tensor) -> Tensor {
    1.0 + BETA_ALPHA_MAX * (head.softplus() / BETA_ALPHA_MAX).tanh()
}

pub fn beta_mean(alpha: &Tensor, beta: &Tensor) -> Tensor {
    alpha / (alpha + beta)
}

pub fn beta_variance(alpha: &Tensor, beta: &Tensor) -> Tensor {
    let sum = alpha + beta;
    alpha * beta / (sum.pow_tensor_scalar(2.0) * (sum + 1.0))
}

pub fn beta_std(alpha: &Tensor, beta: &Tensor) -> Tensor {
    beta_variance(alpha, beta).sqrt()
}

pub fn sample_beta_action(alpha: &Tensor, beta: &Tensor) -> Tensor {
    let x = alpha.internal_standard_gamma();
    let y = beta.internal_standard_gamma();
    (&x / (&x + &y)).clamp(BETA_SAMPLE_EPS, 1.0 - BETA_SAMPLE_EPS)
}

fn beta_log_norm(alpha: &Tensor, beta: &Tensor) -> Tensor {
    alpha.lgamma() + beta.lgamma() - (alpha + beta).lgamma()
}

pub fn beta_log_prob_per_dim(action: &Tensor, alpha: &Tensor, beta: &Tensor) -> Tensor {
    let z = action.clamp(BETA_SAMPLE_EPS, 1.0 - BETA_SAMPLE_EPS);
    let one_minus_z = z.neg() + 1.0;
    (alpha - 1.0) * z.log() + (beta - 1.0) * one_minus_z.log() - beta_log_norm(alpha, beta)
}

pub fn beta_log_prob(action: &Tensor, alpha: &Tensor, beta: &Tensor) -> Tensor {
    beta_log_prob_per_dim(action, alpha, beta).sum_dim_intlist([-1].as_slice(), false, Kind::Float)
}

pub fn beta_entropy(alpha: &Tensor, beta: &Tensor) -> Tensor {
    let sum = alpha + beta;
    let per_dim = beta_log_norm(alpha, beta)
        - (alpha - 1.0) * alpha.digamma()
        - (beta - 1.0) * beta.digamma()
        + (&sum - 2.0) * sum.digamma();
    per_dim.sum_dim_intlist([-1].as_slice(), false, Kind::Float)
}

pub fn beta_kl_old_new(
    old_alpha: &Tensor,
    old_beta: &Tensor,
    new_alpha: &Tensor,
    new_beta: &Tensor,
) -> Tensor {
    let old_sum = old_alpha + old_beta;
    let new_sum = new_alpha + new_beta;
    let per_dim = beta_log_norm(new_alpha, new_beta) - beta_log_norm(old_alpha, old_beta)
        + (old_alpha - new_alpha) * old_alpha.digamma()
        + (old_beta - new_beta) * old_beta.digamma()
        + (new_sum - &old_sum) * old_sum.digamma();
    per_dim.sum_dim_intlist([-1].as_slice(), false, Kind::Float)
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

pub fn transformed_action_log_prob_per_dim(
    latent: &Tensor,
    mean: &Tensor,
    std: &Tensor,
    log_2pi: f64,
) -> Tensor {
    let diff = latent - mean;
    let var = std.pow_tensor_scalar(2);
    let log_std = std.log();
    let mahal = diff.pow_tensor_scalar(2) / &var;
    let gaussian = -(mahal + log_2pi) * 0.5 - &log_std;
    let log_det = latent.log_sigmoid() + (-latent).log_sigmoid();
    gaussian - log_det
}

pub fn transformed_action_log_prob_entropy_and_var(
    latent: &Tensor,
    mean: &Tensor,
    std: &Tensor,
    log_2pi: f64,
) -> (Tensor, Tensor, Tensor) {
    let diff = latent - mean;
    let var = std.pow_tensor_scalar(2);
    let log_std = std.log();
    let mahal = diff.pow_tensor_scalar(2) / &var;
    let gaussian =
        (-(mahal + log_2pi) * 0.5 - &log_std).sum_dim_intlist([-1].as_slice(), false, Kind::Float);
    let log_prob = gaussian - sigmoid_log_det(latent);
    let entropy =
        (&log_std + 0.5 * (1.0 + log_2pi)).sum_dim_intlist([-1].as_slice(), false, Kind::Float);
    (log_prob, entropy, var)
}

#[cfg(test)]
mod tests {
    use super::{
        beta_concentration_from_head, beta_kl_old_new, beta_log_prob,
        gaussian_kl_old_new_from_log_var, sigmoid_log_det, sigmoid_target_weight,
        transformed_action_log_prob, transformed_action_log_prob_per_dim, BETA_ALPHA_MAX,
    };
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

    #[test]
    fn gaussian_kl_old_new_is_zero_for_matching_distribution() {
        let mean = Tensor::from_slice(&[0.25f32, -0.5]).view([1, 2]);
        let log_var = Tensor::from_slice(&[0.1f32, -0.2]).view([1, 2]);
        let kl = gaussian_kl_old_new_from_log_var(&mean, &log_var, &mean, &log_var);

        assert!(kl.double_value(&[0]).abs() < 1e-6);
    }

    #[test]
    fn gaussian_kl_old_new_penalizes_new_variance_collapse() {
        let old_mean = Tensor::zeros([1, 1], (Kind::Float, tch::Device::Cpu));
        let new_mean = Tensor::zeros([1, 1], (Kind::Float, tch::Device::Cpu));
        let old_log_var = Tensor::zeros([1, 1], (Kind::Float, tch::Device::Cpu));
        let new_log_var = Tensor::from_slice(&[-4.0f32]).view([1, 1]);
        let kl = gaussian_kl_old_new_from_log_var(&old_mean, &old_log_var, &new_mean, &new_log_var);

        assert!(kl.double_value(&[0]) > 20.0);
    }

    #[test]
    fn transformed_action_log_prob_per_dim_sums_to_total_log_prob() {
        let latent = Tensor::from_slice(&[0.2f32, -0.4]).view([1, 2]);
        let mean = Tensor::from_slice(&[0.1f32, -0.1]).view([1, 2]);
        let std = Tensor::from_slice(&[0.9f32, 1.2]).view([1, 2]);
        let total = transformed_action_log_prob(&latent, &mean, &std, 1.8378770664093453);
        let per_dim = transformed_action_log_prob_per_dim(&latent, &mean, &std, 1.8378770664093453)
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float);

        assert!((total.double_value(&[0]) - per_dim.double_value(&[0])).abs() < 1e-6);
    }

    #[test]
    fn beta_concentration_is_soft_capped() {
        let head = Tensor::from_slice(&[-100.0f32, 0.0, 100.0]).view([1, 3]);
        let concentration = beta_concentration_from_head(&head);

        assert!(concentration.min().double_value(&[]) >= 1.0);
        assert!(concentration.max().double_value(&[]) <= 1.0 + BETA_ALPHA_MAX + 1e-5);
        assert!((concentration.double_value(&[0, 1]) - 1.693).abs() < 1e-3);
    }

    #[test]
    fn beta_log_prob_matches_uniform_density() {
        let action = Tensor::from_slice(&[0.2f32, 0.8]).view([1, 2]);
        let alpha = Tensor::ones([1, 2], (Kind::Float, tch::Device::Cpu));
        let beta = Tensor::ones([1, 2], (Kind::Float, tch::Device::Cpu));
        let log_prob = beta_log_prob(&action, &alpha, &beta);

        assert!(log_prob.double_value(&[0]).abs() < 1e-6);
    }

    #[test]
    fn beta_kl_old_new_is_zero_for_matching_distribution() {
        let alpha = Tensor::from_slice(&[1.5f32, 3.0]).view([1, 2]);
        let beta = Tensor::from_slice(&[2.0f32, 4.0]).view([1, 2]);
        let kl = beta_kl_old_new(&alpha, &beta, &alpha, &beta);

        assert!(kl.double_value(&[0]).abs() < 1e-5);
    }
}
