use tch::{Device, Kind, Tensor};

pub const SIGREG_PROJECTIONS: i64 = 1_024;
pub const SIGREG_KNOTS: i64 = 17;
pub const SIGREG_MAX_VIEWS: i64 = 256;
pub const SIGREG_MAX_T: f64 = 3.0;
pub const DEFAULT_SIGREG_LAMBDA: f64 = 0.09;

/// LeWM SIGReg over `[views, independent_samples, embedding_dim]`.
pub fn sigreg_loss(embeddings: &Tensor) -> Tensor {
    validate_embeddings(embeddings);
    let dim = embeddings.size()[2];
    let directions = Tensor::randn(
        [dim, SIGREG_PROJECTIONS],
        (Kind::Float, embeddings.device()),
    );
    let directions = &directions / directions.norm_scalaropt_dim(2, [0i64].as_slice(), true);
    sigreg_loss_with_directions(embeddings, &directions)
}

pub(crate) fn sigreg_loss_with_directions(embeddings: &Tensor, directions: &Tensor) -> Tensor {
    validate_embeddings(embeddings);
    assert_eq!(
        directions.dim(),
        2,
        "SIGReg directions must be [dim,projections]"
    );
    assert_eq!(
        directions.size()[0],
        embeddings.size()[2],
        "SIGReg direction dimension mismatch"
    );
    let samples = embeddings.size()[1];
    let knots = Tensor::linspace(
        0.0,
        SIGREG_MAX_T,
        SIGREG_KNOTS,
        (Kind::Float, embeddings.device()),
    );
    let dt = SIGREG_MAX_T / (SIGREG_KNOTS - 1) as f64;
    let coefficients = Tensor::full([SIGREG_KNOTS], 2.0 * dt, (Kind::Float, embeddings.device()));
    let _ = coefficients.narrow(0, 0, 1).fill_(dt);
    let _ = coefficients.narrow(0, SIGREG_KNOTS - 1, 1).fill_(dt);
    let normal_ecf = (-knots.square() * 0.5).exp();
    let coefficients = coefficients * &normal_ecf;
    let projected = embeddings.to_kind(Kind::Float).matmul(
        &directions
            .to_device(embeddings.device())
            .to_kind(Kind::Float),
    );
    let phases = projected.unsqueeze(-1) * knots.view([1, 1, 1, SIGREG_KNOTS]);
    let cos_error = phases.cos().mean_dim([1i64].as_slice(), false, Kind::Float)
        - normal_ecf.view([1, 1, SIGREG_KNOTS]);
    let sin_error = phases.sin().mean_dim([1i64].as_slice(), false, Kind::Float);
    let integrated = ((cos_error.square() + sin_error.square())
        * coefficients.view([1, 1, SIGREG_KNOTS]))
    .sum_dim_intlist([-1i64].as_slice(), false, Kind::Float);
    integrated.mean(Kind::Float) * samples as f64
}

/// Select temporal positions as semantic views while preserving batch rows as samples.
pub fn sample_temporal_views(tokens: &Tensor, train: bool) -> Tensor {
    assert_eq!(
        tokens.dim(),
        4,
        "temporal SIGReg input must be [batch,tickers,time,dim]"
    );
    let total_positions = tokens.size()[2];
    let indices = temporal_view_indices(
        total_positions,
        SIGREG_MAX_VIEWS.min(total_positions),
        train,
        tokens.device(),
    );
    let selected = tokens.index_select(2, &indices);
    let samples = selected.size()[0] * selected.size()[1];
    selected.permute([2, 0, 1, 3]).contiguous().reshape([
        indices.size()[0],
        samples,
        selected.size()[3],
    ])
}

pub(crate) fn temporal_view_indices(
    total_positions: i64,
    views: i64,
    train: bool,
    device: Device,
) -> Tensor {
    assert!(total_positions > 0, "temporal positions must be non-empty");
    assert!(views > 0 && views <= total_positions);
    let last = total_positions - 1;
    if views == 1 {
        return Tensor::from_slice(&[last]).to_device(device);
    }
    if train {
        let prefix = Tensor::randperm(last, (Kind::Int64, device)).narrow(0, 0, views - 1);
        return Tensor::cat(
            &[&prefix, &Tensor::from_slice(&[last]).to_device(device)],
            0,
        );
    }
    let indices = (0..views)
        .map(|index| index * last / (views - 1))
        .collect::<Vec<_>>();
    Tensor::from_slice(&indices).to_device(device)
}

fn validate_embeddings(embeddings: &Tensor) {
    assert_eq!(
        embeddings.dim(),
        3,
        "SIGReg input must be [views,samples,dim]"
    );
    let size = embeddings.size();
    assert!(size[0] > 0 && size[1] > 0 && size[2] > 0);
}

#[cfg(test)]
mod tests {
    use super::{sigreg_loss_with_directions, temporal_view_indices, SIGREG_KNOTS, SIGREG_MAX_T};
    use crate::torch::test_rng;
    use tch::{Device, Kind, Tensor};

    fn scalar_sigreg(
        values: &[f32],
        views: usize,
        samples: usize,
        dim: usize,
        dirs: &[f32],
        projections: usize,
    ) -> f64 {
        let dt = SIGREG_MAX_T / (SIGREG_KNOTS - 1) as f64;
        let mut total = 0.0;
        for view in 0..views {
            for projection in 0..projections {
                let mut integrated = 0.0;
                for knot in 0..SIGREG_KNOTS as usize {
                    let t = SIGREG_MAX_T * knot as f64 / (SIGREG_KNOTS - 1) as f64;
                    let phi = (-0.5 * t * t).exp();
                    let trap = if knot == 0 || knot + 1 == SIGREG_KNOTS as usize {
                        dt
                    } else {
                        2.0 * dt
                    };
                    let mut cosine = 0.0;
                    let mut sine = 0.0;
                    for sample in 0..samples {
                        let mut projected = 0.0;
                        for feature in 0..dim {
                            let value = values[(view * samples + sample) * dim + feature] as f64;
                            projected += value * dirs[feature * projections + projection] as f64;
                        }
                        cosine += (t * projected).cos();
                        sine += (t * projected).sin();
                    }
                    cosine /= samples as f64;
                    sine /= samples as f64;
                    integrated += trap * phi * ((cosine - phi).powi(2) + sine.powi(2));
                }
                total += integrated;
            }
        }
        samples as f64 * total / (views * projections) as f64
    }

    #[test]
    fn tensor_kernel_matches_the_lewm_reference_fixture() {
        let values = [
            0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 1.0,
        ];
        let diagonal = std::f32::consts::FRAC_1_SQRT_2;
        let directions = [1.0f32, 0.0, diagonal, 0.0, 1.0, diagonal];
        let tensor = Tensor::from_slice(&values).view([2, 3, 2]);
        let dirs = Tensor::from_slice(&directions).view([2, 3]);
        let actual = sigreg_loss_with_directions(&tensor, &dirs).double_value(&[]);
        let scalar = scalar_sigreg(&values, 2, 3, 2, &directions, 3);
        let reference = 0.535_642_385_5;
        assert!((actual - reference).abs() < 1e-6, "actual={actual}");
        assert!(
            (actual - scalar).abs() < 1e-5,
            "actual={actual} scalar={scalar}"
        );
    }

    #[test]
    fn statistic_uses_the_actual_instantaneous_sample_count() {
        let values = Tensor::from_slice(&[0.25f32, -0.75]).view([1, 2, 1]);
        let duplicated = Tensor::cat(&[&values, &values], 1);
        let directions = Tensor::ones([1, 1], (Kind::Float, Device::Cpu));
        let n2 = sigreg_loss_with_directions(&values, &directions).double_value(&[]);
        let n4 = sigreg_loss_with_directions(&duplicated, &directions).double_value(&[]);
        assert!((n4 - 2.0 * n2).abs() < 1e-6);
    }

    #[test]
    fn sigreg_has_finite_nonzero_gradients() {
        let _torch_rng_guard = test_rng::shared();
        let values = Tensor::randn([3, 5, 4], (Kind::Float, Device::Cpu));
        let _ = values.set_requires_grad(true);
        let directions = Tensor::randn([4, 7], (Kind::Float, Device::Cpu));
        let directions = &directions / directions.norm_scalaropt_dim(2, [0i64].as_slice(), true);
        sigreg_loss_with_directions(&values, &directions).backward();
        let gradient = values.grad();
        assert!(gradient.isfinite().all().int64_value(&[]) != 0);
        assert!(gradient.abs().sum(Kind::Float).double_value(&[]) > 0.0);
    }

    #[test]
    fn temporal_selection_includes_final_position_once() {
        let _torch_rng_guard = test_rng::shared();
        for train in [false, true] {
            let indices = temporal_view_indices(19, 8, train, Device::Cpu);
            let values = Vec::<i64>::try_from(indices).unwrap();
            assert_eq!(values.iter().filter(|&&index| index == 18).count(), 1);
            assert_eq!(values.len(), 8);
        }
    }

    #[test]
    fn prediction_and_target_branches_remain_attached() {
        let prediction = Tensor::from_slice(&[0.2f32, -0.1, 0.8]).set_requires_grad(true);
        let target = Tensor::from_slice(&[0.4f32, 0.3, -0.2]).set_requires_grad(true);
        (&prediction - &target)
            .square()
            .mean(Kind::Float)
            .backward();
        assert!(prediction.grad().abs().sum(Kind::Float).double_value(&[]) > 0.0);
        assert!(target.grad().abs().sum(Kind::Float).double_value(&[]) > 0.0);
    }
}
