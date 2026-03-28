#[cfg(test)]
mod tests {
    use tch::{Kind, Tensor};

    use crate::torch::two_hot::{symlog, TwoHotBins};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Direct expected value from a two-hot distribution (no softmax).
    /// This matches dreamer4's bins_to_scalar_value with normalize=False.
    fn direct_decode(bins: &TwoHotBins, encoded: &Tensor) -> Tensor {
        bins.bins_to_scalar_value(encoded, false)
    }

    // --- symlog / symexp inverse ---

    #[test]
    fn symlog_symexp_inverse() {
        let bins = TwoHotBins::new(-5.0, 5.0, 51, tch::Device::Cpu);
        for &v in &[-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0] {
            let t = Tensor::from_slice(&[v]).to_kind(Kind::Float);
            let encoded = bins.encode(&t);
            let decoded = direct_decode(&bins, &encoded).double_value(&[0]);
            assert!(
                approx_eq(decoded, v, 0.05),
                "roundtrip failed for {v}: got {decoded}"
            );
        }
    }

    // --- Roundtrip: encode → direct_decode recovers original values ---

    #[test]
    fn roundtrip_default_bins() {
        let bins = TwoHotBins::default_for(tch::Device::Cpu);
        let values: Vec<f64> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 10.0, -10.0, 100.0, -100.0];
        for &v in &values {
            let t = Tensor::from_slice(&[v]).to_kind(Kind::Float);
            let encoded = bins.encode(&t);
            let decoded = direct_decode(&bins, &encoded).double_value(&[0]);
            let clamped = v.clamp(bins_min(&bins), bins_max(&bins));
            assert!(
                approx_eq(decoded, clamped, 1e-3),
                "roundtrip failed for {v} (clamped={clamped}): got {decoded}"
            );
        }
    }

    #[test]
    fn roundtrip_small_range() {
        // Matches dreamer4 test: range (-3, 3), 20 bins
        let bins = TwoHotBins::new(-3.0, 3.0, 20, tch::Device::Cpu);
        let values = Tensor::randn([10], (Kind::Float, tch::Device::Cpu));
        let encoded = bins.encode(&values);
        let decoded = direct_decode(&bins, &encoded);

        let diff = (&values - &decoded).abs();
        let max_diff = diff.max().double_value(&[]);
        assert!(
            max_diff < 1e-5,
            "roundtrip max error {max_diff} exceeds tolerance"
        );
    }

    #[test]
    fn roundtrip_batch() {
        let bins = TwoHotBins::default_for(tch::Device::Cpu);
        let values = Tensor::from_slice(&[0.0f32, 5.0, -5.0, 0.1, -0.1]);
        let encoded = bins.encode(&values);
        let decoded = direct_decode(&bins, &encoded);

        for i in 0..5 {
            let orig = values.double_value(&[i]);
            let dec = decoded.double_value(&[i]);
            assert!(
                approx_eq(dec, orig, 1e-3),
                "batch roundtrip failed at index {i}: {orig} -> {dec}"
            );
        }
    }

    // --- Roundtrip through decode (with softmax) via log-space ---
    // decode() applies softmax, so we pass log(encoded) to undo it:
    // softmax(log(p)) = p when p is a valid distribution.

    #[test]
    fn roundtrip_via_log_decode() {
        let bins = TwoHotBins::default_for(tch::Device::Cpu);
        let values = Tensor::from_slice(&[0.0f32, 3.0, -3.0, 0.5]);
        let encoded = bins.encode(&values);
        // Convert probs → logits so decode's softmax recovers the original probs
        let logits = encoded.clamp_min(1e-30).log();
        let decoded = bins.decode(&logits);

        for i in 0..4 {
            let orig = values.double_value(&[i]);
            let dec = decoded.double_value(&[i]);
            assert!(
                approx_eq(dec, orig, 1e-2),
                "log-roundtrip failed at index {i}: {orig} -> {dec}"
            );
        }
    }

    // --- Distribution properties ---

    #[test]
    fn distribution_sums_to_one() {
        let bins = TwoHotBins::default_for(tch::Device::Cpu);
        let values = Tensor::from_slice(&[0.0f32, 1.5, -3.7, 50.0]);
        let encoded = bins.encode(&values);

        for i in 0..4 {
            let row_sum = encoded.get(i).sum(Kind::Float).double_value(&[]);
            assert!(
                approx_eq(row_sum, 1.0, 1e-6),
                "row {i} sums to {row_sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn two_hot_has_at_most_two_nonzero() {
        let bins = TwoHotBins::default_for(tch::Device::Cpu);
        let values = Tensor::from_slice(&[0.0f32, 2.5, -7.0, 0.001]);
        let encoded = bins.encode(&values);

        for i in 0..4 {
            let row = encoded.get(i);
            let nonzero_count = row.gt(1e-7).sum(Kind::Float).int64_value(&[]);
            assert!(
                nonzero_count <= 2,
                "row {i} has {nonzero_count} nonzero entries, expected <= 2"
            );
            assert!(nonzero_count >= 1, "row {i} has no nonzero entries");
        }
    }

    #[test]
    fn weights_are_nonnegative() {
        let bins = TwoHotBins::default_for(tch::Device::Cpu);
        let values = Tensor::randn([100], (Kind::Float, tch::Device::Cpu)) * 50.0;
        let encoded = bins.encode(&values);

        let min_val = encoded.min().double_value(&[]);
        assert!(
            min_val >= -1e-7,
            "encoded contains negative weight: {min_val}"
        );
    }

    // --- Edge cases ---

    #[test]
    fn exact_bin_center_gives_single_hot() {
        let bins = TwoHotBins::new(-5.0, 5.0, 11, tch::Device::Cpu);
        // Bin at index 5 is symexp(0.0) = 0.0
        let t = Tensor::from_slice(&[0.0f32]);
        let encoded = bins.encode(&t);
        let row = encoded.get(0);
        let max_weight = row.max().double_value(&[]);
        assert!(
            approx_eq(max_weight, 1.0, 1e-5),
            "exact bin center should have weight ~1.0, got {max_weight}"
        );
    }

    #[test]
    fn min_max_boundary_values() {
        let bins = TwoHotBins::new(-3.0, 3.0, 21, tch::Device::Cpu);
        let min_val = bins_min(&bins);
        let max_val = bins_max(&bins);

        let t_min = Tensor::from_slice(&[min_val as f32]);
        let dec_min = direct_decode(&bins, &bins.encode(&t_min)).double_value(&[0]);
        assert!(
            approx_eq(dec_min, min_val, 1e-3),
            "min boundary: expected {min_val}, got {dec_min}"
        );

        let t_max = Tensor::from_slice(&[max_val as f32]);
        let dec_max = direct_decode(&bins, &bins.encode(&t_max)).double_value(&[0]);
        assert!(
            approx_eq(dec_max, max_val, 1e-3),
            "max boundary: expected {max_val}, got {dec_max}"
        );
    }

    #[test]
    fn values_beyond_range_get_clamped() {
        let bins = TwoHotBins::new(-3.0, 3.0, 21, tch::Device::Cpu);
        let min_val = bins_min(&bins);
        let max_val = bins_max(&bins);

        let far_positive = Tensor::from_slice(&[9999.0f32]);
        let dec = direct_decode(&bins, &bins.encode(&far_positive)).double_value(&[0]);
        assert!(
            approx_eq(dec, max_val, 1e-3),
            "far positive should clamp to max {max_val}, got {dec}"
        );

        let far_negative = Tensor::from_slice(&[-9999.0f32]);
        let dec = direct_decode(&bins, &bins.encode(&far_negative)).double_value(&[0]);
        assert!(
            approx_eq(dec, min_val, 1e-3),
            "far negative should clamp to min {min_val}, got {dec}"
        );
    }

    // --- Decode with logits (the actual network use case) ---

    #[test]
    fn decode_uniform_logits_gives_mean_of_bins() {
        let bins = TwoHotBins::new(-5.0, 5.0, 51, tch::Device::Cpu);
        let logits = Tensor::zeros([1, 51], (Kind::Float, tch::Device::Cpu));
        let decoded = bins.decode(&logits).double_value(&[0]);
        // symexp bins are symmetric around 0, so mean should be ~0
        assert!(
            approx_eq(decoded, 0.0, 0.1),
            "uniform logits should decode to ~0 for symmetric bins, got {decoded}"
        );
    }

    #[test]
    fn decode_one_hot_logit_gives_bin_value() {
        let bins = TwoHotBins::new(-3.0, 3.0, 21, tch::Device::Cpu);
        // Large logit at center bin → softmax concentrates there → symexp(0) = 0
        let logits = Tensor::full([1, 21], -100.0, (Kind::Float, tch::Device::Cpu));
        let _ = logits.get(0).get(10).fill_(100.0);
        let decoded = bins.decode(&logits).double_value(&[0]);
        assert!(
            approx_eq(decoded, 0.0, 1e-4),
            "one-hot logit at center bin should decode to 0.0, got {decoded}"
        );
    }

    // --- Cross-entropy loss sanity (used in PPO value loss) ---

    #[test]
    fn cross_entropy_of_matching_distribution_is_minimal() {
        let bins = TwoHotBins::default_for(tch::Device::Cpu);
        let value = Tensor::from_slice(&[2.5f32]);
        let target = bins.encode(&value);

        // Logits that produce the same distribution after softmax
        let matching_logits = target.clamp_min(1e-30).log();
        // Logits that produce a different distribution (shifted)
        let wrong_value = Tensor::from_slice(&[10.0f32]);
        let wrong_target = bins.encode(&wrong_value);
        let wrong_logits = wrong_target.clamp_min(1e-30).log();

        let ce_match = -(&target * matching_logits.log_softmax(-1, Kind::Float))
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
            .double_value(&[0]);
        let ce_wrong = -(&target * wrong_logits.log_softmax(-1, Kind::Float))
            .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
            .double_value(&[0]);

        assert!(
            ce_match < ce_wrong,
            "matching distribution should have lower CE ({ce_match}) than wrong ({ce_wrong})"
        );
    }

    // --- Shape correctness ---

    #[test]
    fn encode_shape() {
        let bins = TwoHotBins::new(-5.0, 5.0, 31, tch::Device::Cpu);
        let values = Tensor::zeros([4], (Kind::Float, tch::Device::Cpu));
        let encoded = bins.encode(&values);
        assert_eq!(encoded.size(), vec![4, 31]);
    }

    #[test]
    fn decode_shape() {
        let bins = TwoHotBins::new(-5.0, 5.0, 31, tch::Device::Cpu);
        let logits = Tensor::zeros([4, 31], (Kind::Float, tch::Device::Cpu));
        let decoded = bins.decode(&logits);
        assert_eq!(decoded.size(), vec![4]);
    }

    #[test]
    fn bins_to_scalar_value_matches_reference_normalize_flag() {
        let bins = TwoHotBins::new(-3.0, 3.0, 21, tch::Device::Cpu);
        let values = Tensor::from_slice(&[-1.5f32, 0.0, 2.25]);
        let encoded = bins.encode(&values);

        let direct = bins.bins_to_scalar_value(&encoded, false);
        for i in 0..3 {
            let actual = direct.double_value(&[i]);
            let expected = values.double_value(&[i]);
            assert!(
                approx_eq(actual, expected, 1e-4),
                "normalize=false mismatch at {i}: expected {expected}, got {actual}"
            );
        }

        let logits = encoded.clamp_min(1e-30).log();
        let normalized = bins.bins_to_scalar_value(&logits, true);
        for i in 0..3 {
            let actual = normalized.double_value(&[i]);
            let expected = values.double_value(&[i]);
            assert!(
                approx_eq(actual, expected, 1e-4),
                "normalize=true mismatch at {i}: expected {expected}, got {actual}"
            );
        }
    }

    // --- symlog properties ---

    #[test]
    fn symlog_properties() {
        assert!(approx_eq(symlog(0.0), 0.0, 1e-10));
        assert!(symlog(1.0) > 0.0);
        assert!(symlog(-1.0) < 0.0);
        assert!(approx_eq(symlog(1.0), -symlog(-1.0), 1e-10)); // odd function
        assert!(approx_eq(symlog(1.0), (2.0f64).ln(), 1e-10)); // ln(|1| + 1) = ln(2)
    }

    // --- Bin construction ---

    #[test]
    fn bins_are_monotonically_increasing() {
        let bins = TwoHotBins::default_for(tch::Device::Cpu);
        let n = bins.bin_values.size()[0];
        for i in 1..n {
            let prev = bins.bin_values.get(i - 1).double_value(&[]);
            let curr = bins.bin_values.get(i).double_value(&[]);
            assert!(
                curr > prev,
                "bin {i} ({curr}) not greater than bin {} ({prev})",
                i - 1
            );
        }
    }

    #[test]
    fn bins_are_symmetric_around_zero() {
        let bins = TwoHotBins::default_for(tch::Device::Cpu);
        let n = bins.bin_values.size()[0];
        let first = bins.bin_values.get(0).double_value(&[]);
        let last = bins.bin_values.get(n - 1).double_value(&[]);
        assert!(
            approx_eq(first, -last, 1e-4),
            "bins not symmetric: first={first}, last={last}"
        );
        // Center bin should be ~0 (symexp(0) = 0)
        let center = bins.bin_values.get(n / 2).double_value(&[]);
        assert!(
            approx_eq(center, 0.0, 1e-6),
            "center bin should be 0.0, got {center}"
        );
    }

    // --- Helpers ---

    fn bins_min(bins: &TwoHotBins) -> f64 {
        bins.bin_values.get(0).double_value(&[])
    }

    fn bins_max(bins: &TwoHotBins) -> f64 {
        let n = bins.bin_values.size()[0];
        bins.bin_values.get(n - 1).double_value(&[])
    }
}
