use tch::{Kind, Tensor};

use crate::torch::value::hl_gauss::HlGaussBins;

pub(crate) fn hl_gauss_value_loss(
    hl_gauss: &HlGaussBins,
    value_logits: &Tensor,
    returns: &Tensor,
) -> Tensor {
    let return_bins = hl_gauss.encode(returns);
    let log_probs = value_logits.log_softmax(-1, Kind::Float);
    -(&return_bins * &log_probs).sum_dim_intlist([-1].as_slice(), false, Kind::Float)
}

#[cfg(test)]
mod tests {
    use tch::{Kind, Tensor};

    use super::hl_gauss_value_loss;
    use crate::torch::value::hl_gauss::HlGaussBins;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn hl_gauss_value_loss_matches_cross_entropy() {
        let hl_gauss = HlGaussBins::new(-3.0, 3.0, 21, tch::Device::Cpu);
        let logits = Tensor::zeros([1, 21], (Kind::Float, tch::Device::Cpu));
        let _ = logits.get(0).get(10).fill_(0.3);
        let _ = logits.get(0).get(11).fill_(0.1);

        let returns = Tensor::from_slice(&[0.15f32]);

        let loss = hl_gauss_value_loss(&hl_gauss, &logits, &returns);

        let return_bins = hl_gauss.encode(&returns);
        let reference = -(&return_bins * logits.log_softmax(-1, Kind::Float)).sum_dim_intlist(
            [-1].as_slice(),
            false,
            Kind::Float,
        );

        let actual = loss.double_value(&[0]);
        let expected = reference.double_value(&[0]);

        assert!(
            approx_eq(actual, expected, 1e-6),
            "value loss mismatch: expected {expected}, got {actual}"
        );
    }
}
