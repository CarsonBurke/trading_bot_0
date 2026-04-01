#[cfg(test)]
mod tests {
    use tch::{Kind, Tensor};

    use crate::torch::ppo::two_hot_value_loss_terms;
    use crate::torch::two_hot::TwoHotBins;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn clipped_value_loss_matches_dreamer4_log_softmax_branch() {
        let two_hot = TwoHotBins::new(-3.0, 3.0, 21, tch::Device::Cpu);
        let logits = Tensor::zeros([1, 21], (Kind::Float, tch::Device::Cpu));
        let _ = logits.get(0).get(10).fill_(0.3);
        let _ = logits.get(0).get(11).fill_(0.1);

        let old_values = Tensor::zeros([1], (Kind::Float, tch::Device::Cpu));
        let returns = Tensor::from_slice(&[0.15f32]);

        let (_, clipped_loss) = two_hot_value_loss_terms(&two_hot, &logits, &old_values, &returns);

        let decoded = two_hot.bins_to_scalar_value(&logits, true);
        assert!(
            decoded.abs().double_value(&[0]) < 0.3,
            "test setup must avoid activating value clipping"
        );

        let return_bins = two_hot.encode(&returns);
        let clipped_value_bins = two_hot.encode(&decoded);

        let reference = -(&return_bins * clipped_value_bins.log_softmax(-1, Kind::Float))
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float);
        let non_reference = -(&return_bins * clipped_value_bins.clamp_min(1e-8).log())
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float);

        let actual = clipped_loss.double_value(&[0]);
        let expected = reference.double_value(&[0]);
        let wrong = non_reference.double_value(&[0]);

        assert!(
            approx_eq(actual, expected, 1e-6),
            "reference clipped loss mismatch: expected {expected}, got {actual}"
        );
        assert!(
            !approx_eq(actual, wrong, 1e-4),
            "clipped loss unexpectedly matched non-reference log branch: {actual} vs {wrong}"
        );
    }
}
