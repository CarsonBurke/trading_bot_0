use std::{collections::BTreeSet, sync::LazyLock};

use tch::{Kind, Tensor};

const MIN_PRICE: f64 = f32::from_bits(1) as f64;
const MAX_PRICE: f64 = f32::MAX as f64;
const ORDER_EDGES: [(usize, usize); 4] = [(2, 0), (2, 3), (0, 1), (3, 1)];

/// Keep physical prices for display and validity checks: tiny positive projected
/// prices can disappear in a price -> standardized -> price round trip.
pub struct Projection {
    pub standardized: Tensor,
    pub prices: Tensor,
}

fn active_partitions() -> &'static [[usize; 4]] {
    static PARTITIONS: LazyLock<Vec<[usize; 4]>> = LazyLock::new(|| {
        let mut partitions = BTreeSet::new();
        for active in 0..1 << ORDER_EDGES.len() {
            let mut groups = [0, 1, 2, 3];
            for (edge, &(left, right)) in ORDER_EDGES.iter().enumerate() {
                if active & (1 << edge) != 0 {
                    let from = groups[right];
                    let to = groups[left];
                    for group in &mut groups {
                        if *group == from {
                            *group = to;
                        }
                    }
                }
            }
            let canonical = std::array::from_fn(|index| {
                (0..4)
                    .find(|&other| groups[other] == groups[index])
                    .unwrap()
            });
            partitions.insert(canonical);
        }
        partitions.into_iter().collect()
    });
    &PARTITIONS
}

fn feasible(prices: &Tensor) -> Tensor {
    let open = prices.select(-1, 0);
    let high = prices.select(-1, 1);
    let low = prices.select(-1, 2);
    let close = prices.select(-1, 3);
    low.le_tensor(&open)
        .logical_and(&low.le_tensor(&close))
        .logical_and(&open.le_tensor(&high))
        .logical_and(&close.le_tensor(&high))
        .logical_and(&low.ge(MIN_PRICE))
        .logical_and(&high.le(MAX_PRICE))
}

/// Weighted Euclidean projection onto valid OHLC in physical prices. `scaling`
/// contains train-only means/stds as [B, 2, 4]; inputs are [B, P, 4]. Weights
/// 1/std² make the distance exactly the existing standardized OHLC MSE metric.
/// Inputs must be finite, with positive finite training standard deviations.
///
/// Each optimal face pools connected active order constraints. Enumerating those
/// partitions, solving their weighted means, and retaining the closest feasible
/// solution gives the weighted projection. Common price bounds are included in
/// each face solution; every finite positive F32 target lies inside these bounds.
/// Only one [B, P, 4] candidate is live at a time, without a horizon/candidate stack.
pub fn project(predictions: &Tensor, scaling: &Tensor) -> Projection {
    let shape = predictions.size();
    assert_eq!(shape.len(), 3, "predictions must have shape [B, P, 4]");
    assert_eq!(shape[2], 4, "OHLC requires four channels");
    assert_eq!(scaling.size(), [shape[0], 2, 4]);
    assert_eq!(predictions.device(), scaling.device());
    let predictions = predictions.to_kind(Kind::Double);
    let scaling = scaling.to_kind(Kind::Double);
    let means = scaling.select(1, 0).unsqueeze(1);
    let stds = scaling.select(1, 1).unsqueeze(1);
    let prices = &predictions * &stds + &means;
    let weights = stds.reciprocal().square();
    let masks: Vec<f64> = active_partitions()
        .iter()
        .flat_map(|groups| {
            (0..4).flat_map(move |row| {
                (0..4).map(move |column| {
                    if groups[row] == groups[column] {
                        1.
                    } else {
                        0.
                    }
                })
            })
        })
        .collect();
    let masks = Tensor::from_slice(&masks)
        .reshape([active_partitions().len() as i64, 4, 4])
        .to_device(predictions.device());
    let pooled = (&prices * &weights).sum_dim_intlist(&[-1i64][..], true, Kind::Double)
        / weights.sum_dim_intlist(&[-1i64][..], true, Kind::Double);
    let mut best = pooled.clamp(MIN_PRICE, MAX_PRICE).expand_as(&prices);
    let distance = |candidate: &Tensor| {
        ((candidate - &prices) / &stds)
            .square()
            .sum_dim_intlist(&[-1i64][..], false, Kind::Double)
    };
    let mut best_distance = distance(&best);
    for index in 0..active_partitions().len() {
        let weighted = masks.get(index as i64).unsqueeze(0) * &weights;
        let averaging = &weighted / weighted.sum_dim_intlist(&[-1i64][..], true, Kind::Double);
        let candidate = prices
            .matmul(&averaging.transpose(1, 2))
            .clamp(MIN_PRICE, MAX_PRICE);
        let candidate_distance = distance(&candidate);
        let take = feasible(&candidate).logical_and(&candidate_distance.lt_tensor(&best_distance));
        best = candidate.where_self(&take.unsqueeze(-1), &best);
        best_distance = candidate_distance.where_self(&take, &best_distance);
    }
    let standardized = (&best - &means) / &stds;
    let already_feasible = feasible(&prices).unsqueeze(-1);
    Projection {
        standardized: predictions.where_self(&already_feasible, &standardized),
        prices: prices.where_self(&already_feasible, &best),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    fn scales(stds: [f64; 4], means: [f64; 4]) -> Tensor {
        Tensor::from_slice(&[means, stds].concat()).reshape([1, 2, 4])
    }

    #[test]
    fn feasible_candles_are_unchanged_and_projection_is_idempotent() {
        let scaling = scales([2., 3., 4., 5.], [90., 100., 80., 95.]);
        let prices =
            Tensor::from_slice(&[101., 103., 99., 102., 98., 98., 98., 98.]).reshape([1, 2, 4]);
        let normalized =
            (&prices - scaling.select(1, 0).unsqueeze(1)) / scaling.select(1, 1).unsqueeze(1);
        let first = project(&normalized, &scaling);
        assert!(first.standardized.equal(&normalized));
        assert!(first.prices.allclose(&prices, 1e-14, 1e-14, false));
        let second = project(&first.standardized, &scaling);
        assert!(second.standardized.equal(&first.standardized));
        assert!(feasible(&second.prices).all().int64_value(&[]) != 0);
    }

    #[test]
    fn unequal_weights_pool_violations_in_the_scored_metric() {
        let scaling = scales([1., 0.1, 1., 1.], [0.; 4]);
        let original_prices = Tensor::from_slice(&[1., 1., 1., 3.]).reshape([1, 1, 4]);
        let normalized = &original_prices / scaling.select(1, 1).unsqueeze(1);
        let projected = project(&normalized, &scaling);
        let expected = 103. / 101.;
        assert!((projected.prices.double_value(&[0, 0, 1]) - expected).abs() < 1e-12);
        assert!((projected.prices.double_value(&[0, 0, 3]) - expected).abs() < 1e-12);
        assert_eq!(projected.prices.double_value(&[0, 0, 0]), 1.);
        assert_eq!(projected.prices.double_value(&[0, 0, 2]), 1.);
        let truth = Tensor::ones([1, 1, 4], (Kind::Double, Device::Cpu));
        let weights = scaling.select(1, 1).unsqueeze(1).reciprocal().square();
        let error = |prediction: &Tensor| {
            ((prediction - &truth).square() * &weights)
                .sum(Kind::Double)
                .double_value(&[])
        };
        assert!(error(&projected.prices) < error(&original_prices));
        assert!(
            error(&Tensor::from_slice(&[1., 2., 1., 2.]).reshape([1, 1, 4]))
                > error(&original_prices)
        );
    }

    #[test]
    fn valid_target_mse_never_increases_across_deterministic_cases() {
        let values: Vec<f64> = (0..256 * 4)
            .map(|i| ((i * 137 + 29) % 997) as f64 / 31. - 10.)
            .collect();
        let predictions = Tensor::from_slice(&values).reshape([1, 256, 4]);
        let scaling = scales([0.3, 3., 0.7, 8.], [11., 17., 5., 13.]);
        let projected = project(&predictions, &scaling);
        assert!(feasible(&projected.prices).all().int64_value(&[]) != 0);
        for base in [0.01, 1., 10., 100.] {
            let targets =
                Tensor::from_slice(&[base + 1., base + 4., base, base + 2.]).reshape([1, 1, 4]);
            let normalized_targets =
                (targets - scaling.select(1, 0).unsqueeze(1)) / scaling.select(1, 1).unsqueeze(1);
            let before = (&predictions - &normalized_targets)
                .square()
                .sum_dim_intlist(&[-1i64][..], false, Kind::Double);
            let after = (&projected.standardized - &normalized_targets)
                .square()
                .sum_dim_intlist(&[-1i64][..], false, Kind::Double);
            assert!((after - before).max().double_value(&[]) < 1e-9);
        }
    }

    #[test]
    fn extreme_f32_inputs_produce_finite_representable_prices() {
        let scaling = scales([f32::MAX as f64; 4], [0.; 4]);
        let predictions = Tensor::full([1, 2, 4], f32::MAX as f64, (Kind::Float, Device::Cpu));
        let projected = project(&predictions, &scaling);
        assert!(projected.standardized.isfinite().all().int64_value(&[]) != 0);
        assert!(projected.prices.eq(MAX_PRICE).all().int64_value(&[]) != 0);
        assert!(
            projected
                .prices
                .to_kind(Kind::Float)
                .isfinite()
                .all()
                .int64_value(&[])
                != 0
        );
    }

    #[test]
    fn negative_prices_project_to_a_positive_representable_floor() {
        let scaling = scales([0.5, 3., 2., 7.], [100., 104., 96., 102.]);
        let predictions = Tensor::full([1, 5, 4], -1e6, (Kind::Double, Device::Cpu));
        let projected = project(&predictions, &scaling);
        assert!(projected.prices.eq(MIN_PRICE).all().int64_value(&[]) != 0);
        assert!(
            projected
                .prices
                .to_kind(Kind::Float)
                .gt(0.)
                .all()
                .int64_value(&[])
                != 0
        );
        assert!(projected.standardized.isfinite().all().int64_value(&[]) != 0);
    }
}
