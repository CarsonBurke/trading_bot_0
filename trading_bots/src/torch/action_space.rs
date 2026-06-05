use tch::{Kind, Tensor};

pub const BETA_SAMPLE_EPS: f64 = 1e-6;

pub fn beta_concentration(raw: &Tensor) -> Tensor {
    raw.clamp(-20.0, 8.0).exp() + 1.0
}

pub fn sample_beta_action(alpha: &Tensor, beta: &Tensor) -> Tensor {
    let ga = alpha.internal_standard_gamma();
    let gb = beta.internal_standard_gamma();
    (&ga / (&ga + &gb)).clamp(BETA_SAMPLE_EPS, 1.0 - BETA_SAMPLE_EPS)
}

fn beta_log_norm(alpha: &Tensor, beta: &Tensor) -> Tensor {
    alpha.lgamma() + beta.lgamma() - (alpha + beta).lgamma()
}

pub fn beta_log_prob(x: &Tensor, alpha: &Tensor, beta: &Tensor) -> Tensor {
    let x = x.clamp(BETA_SAMPLE_EPS, 1.0 - BETA_SAMPLE_EPS);
    let one_minus_x: Tensor = 1.0 - &x;
    let log_norm = beta_log_norm(alpha, beta);
    let per_dim = (alpha - 1.0) * x.log() + (beta - 1.0) * one_minus_x.log() - log_norm;
    per_dim.sum_dim_intlist([-1].as_slice(), false, Kind::Float)
}

pub fn beta_entropy(alpha: &Tensor, beta: &Tensor) -> Tensor {
    let log_norm = beta_log_norm(alpha, beta);
    let per_dim = log_norm - (alpha - 1.0) * alpha.digamma() - (beta - 1.0) * beta.digamma()
        + (alpha + beta - 2.0) * (alpha + beta).digamma();
    per_dim.sum_dim_intlist([-1].as_slice(), false, Kind::Float)
}

pub fn beta_mean(alpha: &Tensor, beta: &Tensor) -> Tensor {
    alpha / (alpha + beta)
}

#[cfg(test)]
mod tests {
    use super::{beta_entropy, beta_log_prob, beta_mean};
    use tch::Tensor;

    fn lgamma(x: f64) -> f64 {
        // ln Gamma via integer/half-integer closed forms used in the tests.
        // For the chosen test points all arguments are integers, so use factorials.
        let n = x.round() as i64;
        assert!((x - n as f64).abs() < 1e-12 && n >= 1);
        (1..n).map(|k| (k as f64).ln()).sum()
    }

    #[test]
    fn beta_log_prob_matches_hand_computed_scalar() {
        // alpha=2, beta=2, x=0.5 -> pdf = 1.5 -> log_pdf = ln(1.5) ≈ 0.405465
        let alpha = Tensor::from_slice(&[2.0f32]).view([1, 1]);
        let beta = Tensor::from_slice(&[2.0f32]).view([1, 1]);
        let x = Tensor::from_slice(&[0.5f32]).view([1, 1]);
        let lp = beta_log_prob(&x, &alpha, &beta).double_value(&[0]);
        assert!((lp - 1.5f64.ln()).abs() < 1e-5);
    }

    #[test]
    fn beta_entropy_matches_closed_form() {
        // Closed form for alpha=beta=2: digamma(2)=1-gamma, digamma(4)=11/6-gamma.
        let alpha = Tensor::from_slice(&[2.0f32]).view([1, 1]);
        let beta = Tensor::from_slice(&[2.0f32]).view([1, 1]);
        let ent = beta_entropy(&alpha, &beta).double_value(&[0]);
        let log_norm = lgamma(2.0) + lgamma(2.0) - lgamma(4.0);
        let psi2 = digamma_int(2);
        let psi4 = digamma_int(4);
        let expected = log_norm - 1.0 * psi2 - 1.0 * psi2 + 2.0 * psi4;
        assert!((ent - expected).abs() < 1e-5);
    }

    #[test]
    fn beta_mean_is_alpha_over_alpha_plus_beta() {
        let alpha = Tensor::from_slice(&[3.0f32, 1.0]).view([1, 2]);
        let beta = Tensor::from_slice(&[1.0f32, 3.0]).view([1, 2]);
        let mean = beta_mean(&alpha, &beta);
        assert!((mean.double_value(&[0, 0]) - 0.75).abs() < 1e-6);
        assert!((mean.double_value(&[0, 1]) - 0.25).abs() < 1e-6);
    }

    fn digamma_int(n: i64) -> f64 {
        // psi(n) = -gamma + sum_{k=1}^{n-1} 1/k for positive integer n.
        const EULER_MASCHERONI: f64 = 0.5772156649015329;
        let harmonic: f64 = (1..n).map(|k| 1.0 / k as f64).sum();
        -EULER_MASCHERONI + harmonic
    }
}
