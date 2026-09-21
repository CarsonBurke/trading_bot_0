//! Fixed causal conditional-mean instruments and diagonal-free batch-row moments.
//! Time is a collection of separate populations, never a second independent-sample axis.
//! The final objective equally weights nonempty horizons, then their eligible source means.
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tch::{Device, Kind, Tensor};

use super::{
    corpus::Batch,
    model::{CausalPatchModel, Head, HorizonMean, ModelConfig, Statistics},
};

pub(super) const HORIZONS: [i64; super::reports::DECISION_HORIZONS.len()] = {
    let mut horizons = [0; super::reports::DECISION_HORIZONS.len()];
    let mut index = 0;
    while index < horizons.len() {
        horizons[index] = super::reports::DECISION_HORIZONS[index] as i64;
        index += 1;
    }
    horizons
};
const LAGS: [i64; 8] = [1, 8, 16, 32, 64, 128, 512, 2048];
const SUMMARIES: i64 = 32;
const FOURIER_PAIRS: i64 = 64;
pub(super) const INSTRUMENT_WIDTH: i64 = 1 + SUMMARIES + 2 * FOURIER_PAIRS;

/// S=||sum(mask*e*phi)||², D=sum(||mask*e*phi||²), N=sum(mask).
/// U/V/diagonal average the SAME N>=2 source populations; V is NOT U+diagonal,
/// because its denominator is N² rather than N(N-1). Rows/pairs are sums over sources.
pub(super) const DIAGNOSTIC_LABELS: [&str; 7] = [
    "U (S-D)/(N*(N-1)); mean sources N>=2",
    "V S/N^2; mean sources N>=2",
    "Removed diagonal D/(N*(N-1)); mean sources N>=2",
    "Close MSE in sigma*sqrt(h) units; valid-row mean",
    "Residual target-prediction in sigma*sqrt(h) units; valid-row mean",
    "Valid rows sum N over sources",
    "Ordered distinct row pairs sum N*(N-1) over sources",
];

pub(super) struct DecisionValues {
    /// Actual close forecast and observed close target, both in source sigma*sqrt(h) units.
    pub prediction: Tensor,
    pub target: Tensor,
    /// Original close-score validity; no horizon decimation or stricter JEPA validity.
    pub mask: Tensor,
}

pub(super) struct MomentOutput {
    pub moment_loss: Tensor,
    pub decision_mse: Tensor,
    /// Detached [diagnostic, horizon], in DIAGNOSTIC_LABELS and HORIZONS order.
    pub diagnostics: Tensor,
}

/// Device-resident, parameter-free operands shared by training and frozen witnesses.
/// Construction consumes only a private host RNG, never the global Torch/loader RNG.
pub(super) struct TemporalMoments {
    context: i64,
    origins: i64,
    source: Tensor,
    source_end: Tensor,
    lookback: Tensor,
    available: Tensor,
    lag: Tensor,
    sqrt_lag: Tensor,
    horizons: Tensor,
    sqrt_horizon: Tensor,
    frequencies: Tensor,
}

impl TemporalMoments {
    pub fn new(config: &ModelConfig, device: Device) -> Self {
        assert!(
            config.pred_len >= HORIZONS[HORIZONS.len() - 1],
            "temporal moments require all seven decision horizons"
        );
        assert!(
            config.target_basis.is_identity()
                && !matches!(config.horizon_mean, HorizonMean::Increment { .. }),
            "temporal moments require cumulative, non-increment means"
        );
        assert!(
            !config.future_calendar,
            "temporal moments require causal future-calendar=false"
        );
        let source: Vec<i64> = (0..config.origins())
            .map(|k| (k + 1) * config.patch_len - 1)
            .collect();
        let lookback: Vec<i64> = source
            .iter()
            .flat_map(|&t| LAGS.map(|lag| (t - lag).max(0)))
            .collect();
        let available: Vec<f32> = source
            .iter()
            .flat_map(|&t| LAGS.map(|lag| if t >= lag { 1. } else { 0. }))
            .collect();
        let lag = Tensor::from_slice(&LAGS)
            .to_device(device)
            .to_kind(Kind::Float)
            .reshape([1, 1, LAGS.len() as i64]);
        let mut rng = ChaCha8Rng::seed_from_u64(20260920);
        let mut frequencies = vec![0f32; (SUMMARIES * FOURIER_PAIRS) as usize];
        for (band, bandwidth) in [0.5, 1., 2., 4.].into_iter().enumerate() {
            for projection in 0..16 {
                let p = band * 16 + projection;
                for d in (0..SUMMARIES as usize).step_by(2) {
                    let radius = (-2. * (1. - rng.random::<f64>()).ln()).sqrt();
                    let (sin, cos) = (std::f64::consts::TAU * rng.random::<f64>()).sin_cos();
                    let scale = bandwidth / (SUMMARIES as f64).sqrt();
                    frequencies[d * FOURIER_PAIRS as usize + p] = (radius * cos * scale) as f32;
                    frequencies[(d + 1) * FOURIER_PAIRS as usize + p] =
                        (radius * sin * scale) as f32;
                }
            }
        }
        Self {
            context: config.seq_len,
            origins: config.origins(),
            source_end: Tensor::from_slice(&source.iter().map(|t| t + 1).collect::<Vec<_>>())
                .to_device(device),
            source: Tensor::from_slice(&source).to_device(device),
            lookback: Tensor::from_slice(&lookback).to_device(device),
            available: Tensor::from_slice(&available)
                .reshape([1, config.origins(), LAGS.len() as i64])
                .to_device(device),
            sqrt_lag: lag.sqrt(),
            lag,
            horizons: Tensor::from_slice(&HORIZONS.map(|h| h - 1)).to_device(device),
            sqrt_horizon: Tensor::from_slice(&HORIZONS.map(|h| (h as f32).sqrt()))
                .to_device(device),
            frequencies: Tensor::from_slice(&frequencies)
                .reshape([SUMMARIES, FOURIER_PAIRS])
                .to_device(device),
        }
    }

    /// [B,O,32], lag-major (neutral return, market return, realized scale, availability).
    /// A lag L uses closes at t-L and t and exactly L intervening close steps. Missing
    /// lookbacks produce zero summaries, not dropped forecast populations. Context narrowing
    /// occurs before any difference or prefix sum, and every gather stops at its own source.
    fn summaries(&self, batch: &Batch, stats: &Statistics) -> Tensor {
        assert_eq!(
            stats.sigma.size()[1],
            self.origins,
            "instruments need full-origin statistics"
        );
        let close = batch.log_prices.narrow(1, 0, self.context).select(2, 3);
        let market = batch.market_cum.narrow(1, 0, self.context);
        let valid = batch.valid.narrow(1, 0, self.context);
        let rows = close.size()[0];
        let shape = [rows, self.origins, LAGS.len() as i64];
        let lagged = |values: &Tensor| values.index_select(1, &self.lookback).reshape(shape);
        let at_source = |values: &Tensor| values.index_select(1, &self.source).unsqueeze(-1);
        let lead = Tensor::zeros([rows, 1], (Kind::Float, close.device()));
        let valid_prefix = Tensor::cat(&[&lead, &valid.cumsum(1, Kind::Float)], 1);
        let count =
            valid_prefix.index_select(1, &self.source_end).unsqueeze(-1) - lagged(&valid_prefix);
        let complete = count.eq_tensor(&(&self.lag + 1.)).to_kind(Kind::Float) * &self.available;
        let pair = valid.narrow(1, 1, self.context - 1) * valid.narrow(1, 0, self.context - 1);
        let step = close.narrow(1, 1, self.context - 1) - close.narrow(1, 0, self.context - 1);
        let step = step.where_self(&pair.gt(0.), &step.zeros_like());
        let square_prefix = Tensor::cat(&[&lead, &step.square()], 1).cumsum(1, Kind::Float);
        let mean_square = (at_source(&square_prefix) - lagged(&square_prefix)) / &self.lag;
        let sigma = stats.sigma.unsqueeze(-1);
        let denominator = &sigma * &self.sqrt_lag;
        let market_return = at_source(&market) - lagged(&market);
        let neutral_return =
            at_source(&close) - lagged(&close) - stats.beta.unsqueeze(-1) * &market_return;
        let neutral = (neutral_return / &denominator).tanh();
        let market = (market_return / denominator).tanh();
        // tanh(0.5*log(r))=(r-1)/(r+1), including the r=0 limit -1.
        // This avoids log(0), arbitrary volatility floors, and an unnecessary log/tanh pair.
        let ratio = mean_square.clamp_min(0.) / sigma.square();
        let realized = (&ratio - 1.) / (&ratio + 1.);
        let keep = complete.gt(0.);
        let zero = complete.zeros_like();
        Tensor::stack(
            &[
                neutral.where_self(&keep, &zero),
                market.where_self(&keep, &zero),
                realized.where_self(&keep, &zero),
                complete,
            ],
            -1,
        )
        .flatten(2, 3)
    }

    /// Fixed [B,O,161] instruments with squared norm <=1: constant energy 1/3,
    /// bounded linear energy <=1/3, and 64 Fourier pairs of total energy exactly 1/3.
    pub fn instruments(&self, batch: &Batch, stats: &Statistics) -> Tensor {
        tch::no_grad(|| {
            let summaries = self.summaries(batch, stats);
            let phase = summaries.matmul(&self.frequencies);
            let constant = Tensor::full(
                [summaries.size()[0], self.origins, 1],
                1. / 3f64.sqrt(),
                (Kind::Float, summaries.device()),
            );
            Tensor::cat(
                &[
                    constant,
                    summaries / (3. * SUMMARIES as f64).sqrt(),
                    phase.sin() / (3. * FOURIER_PAIRS as f64).sqrt(),
                    phase.cos() / (3. * FOURIER_PAIRS as f64).sqrt(),
                ],
                -1,
            )
        })
    }

    /// [B,O',7], supporting either the dense training head or final-origin evaluation head.
    pub fn select(
        &self,
        model: &CausalPatchModel,
        head: &Head,
        targets: &Tensor,
        mask: &Tensor,
    ) -> DecisionValues {
        DecisionValues {
            prediction: model.selected_normalized_close(head, &self.horizons),
            target: targets
                .select(2, 3)
                .index_select(-1, &self.horizons)
                .to_kind(Kind::Float)
                / &self.sqrt_horizon,
            mask: mask
                .select(2, 0)
                .index_select(-1, &self.horizons)
                .to_kind(Kind::Float),
        }
    }
    /// Equal-horizon logistic loss for the observable sign decision. Zero targets are
    /// deliberately excluded because their direction is undefined; the close forecast remains
    /// in source-sigma*sqrt(h) units so this adds no hidden per-horizon calibration.
    pub fn decision_sign_loss(&self, decision: &DecisionValues) -> Tensor {
        let finite = decision.target.isfinite().to_kind(Kind::Float);
        let active = &decision.mask * finite * decision.target.ne(0.).to_kind(Kind::Float);
        let target = decision
            .target
            .where_self(&active.gt(0.), &decision.target.zeros_like());
        let margin = &decision.prediction * target.sign();
        let rows = active.sum_dim_intlist([0i64].as_slice(), false, Kind::Float);
        let horizon_loss = ((-&margin).softplus() * &active).sum_dim_intlist(
            [0i64].as_slice(),
            false,
            Kind::Float,
        ) / rows.clamp_min(1.);
        let present = rows.gt(0.).to_kind(Kind::Float);
        (&horizon_loss * &present).sum(Kind::Float) / present.sum(Kind::Float).clamp_min(1.)
    }

    /// Each (source,horizon) population contains batch rows only. Two BMMs avoid materializing
    /// [B,O,H,F]: [O,H,B]@[O,B,F] gives the summed moment, and its diagonal needs only
    /// [O,H,B]@[O,B,1]. N<2 populations are absent, not zero-valued observations.
    pub fn objective(&self, decision: &DecisionValues, instruments: &Tensor) -> MomentOutput {
        let mask = &decision.mask;
        let residual = &decision.target - &decision.prediction;
        let residual = residual.where_self(&mask.gt(0.), &residual.zeros_like());
        let weighted = &residual * mask;
        let by_source = weighted.permute([1, 2, 0]);
        let phi = instruments.permute([1, 0, 2]);
        let summed =
            by_source
                .bmm(&phi)
                .square()
                .sum_dim_intlist([-1i64].as_slice(), false, Kind::Float);
        let energy = phi
            .square()
            .sum_dim_intlist([-1i64].as_slice(), true, Kind::Float);
        let diagonal = by_source.square().bmm(&energy).squeeze_dim(-1);
        let count = mask.sum_dim_intlist([0i64].as_slice(), false, Kind::Float);
        let pairs = &count * (&count - 1.);
        let eligible = count.ge(2.).to_kind(Kind::Float);
        let u = (&summed - &diagonal) / pairs.clamp_min(1.) * &eligible;
        let populations_h = eligible.sum_dim_intlist([0i64].as_slice(), false, Kind::Float);
        let u_h =
            u.sum_dim_intlist([0i64].as_slice(), false, Kind::Float) / populations_h.clamp_min(1.);
        let paired_h = populations_h.gt(0.).to_kind(Kind::Float);
        // Equal horizon weighting, even when long horizons have fewer eligible sources.
        let moment_loss =
            (&u_h * &paired_h).sum(Kind::Float) / paired_h.sum(Kind::Float).clamp_min(1.);
        let rows_h = count.sum_dim_intlist([0i64].as_slice(), false, Kind::Float);
        let mse_h =
            (residual.square() * mask).sum_dim_intlist([0i64, 1].as_slice(), false, Kind::Float)
                / rows_h.clamp_min(1.);
        let present_h = rows_h.gt(0.).to_kind(Kind::Float);
        let decision_mse =
            (&mse_h * &present_h).sum(Kind::Float) / present_h.sum(Kind::Float).clamp_min(1.);
        let diagnostics = tch::no_grad(|| {
            let population_mean = |values: Tensor| {
                (values * &eligible).sum_dim_intlist([0i64].as_slice(), false, Kind::Float)
                    / populations_h.clamp_min(1.)
            };
            Tensor::stack(
                &[
                    u_h.detach(),
                    population_mean(summed.detach() / count.square().clamp_min(1.)),
                    population_mean(diagonal.detach() / pairs.clamp_min(1.)),
                    mse_h.detach(),
                    weighted
                        .detach()
                        .sum_dim_intlist([0i64, 1].as_slice(), false, Kind::Float)
                        / rows_h.clamp_min(1.),
                    rows_h,
                    pairs.sum_dim_intlist([0i64].as_slice(), false, Kind::Float),
                ],
                0,
            )
        });
        MomentOutput {
            moment_loss,
            decision_mse,
            diagnostics,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> TemporalMoments {
        TemporalMoments::new(
            &ModelConfig {
                future_calendar: false,
                ..ModelConfig::default()
            },
            Device::Cpu,
        )
    }

    #[test]
    fn horizon_means_are_equal_weighted_despite_unequal_sources_and_singleton_horizons() {
        let values = DecisionValues {
            prediction: Tensor::zeros([2, 3, 7], (Kind::Float, Device::Cpu))
                .set_requires_grad(true),
            target: Tensor::zeros([2, 3, 7], (Kind::Float, Device::Cpu)),
            mask: Tensor::zeros([2, 3, 7], (Kind::Float, Device::Cpu)),
        };
        let _ = values.target.select(2, 0).fill_(1.);
        let _ = values.mask.select(2, 0).fill_(1.);
        let _ = values.target.select(2, 1).narrow(1, 0, 1).fill_(3.);
        let _ = values.mask.select(2, 1).narrow(1, 0, 1).fill_(1.);
        // A horizon with one observed row enters direct MSE, but cannot enter a pair moment.
        let _ = values
            .mask
            .select(0, 0)
            .select(0, 0)
            .narrow(0, 2, 1)
            .fill_(1.);
        let output = geometry().objective(
            &values,
            &Tensor::ones([2, 3, 1], (Kind::Float, Device::Cpu)),
        );
        assert_eq!(
            output.moment_loss.double_value(&[]),
            5.,
            "three U=1 sources must not outweigh one U=9 source"
        );
        assert!((output.decision_mse.double_value(&[]) - 10. / 3.).abs() < 1e-6);
        output.moment_loss.backward();
        assert_eq!(
            values
                .prediction
                .grad()
                .select(2, 2)
                .abs()
                .max()
                .double_value(&[]),
            0.,
            "a singleton horizon contributed a fake pair"
        );
    }

    #[test]
    fn decision_sign_loss_excludes_zero_targets_and_pushes_correct_margin() {
        let prediction =
            Tensor::zeros([1, 1, 7], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let target = Tensor::zeros([1, 1, 7], (Kind::Float, Device::Cpu));
        let mask = Tensor::ones([1, 1, 7], (Kind::Float, Device::Cpu));
        let _ = target.select(2, 0).fill_(1.);
        let _ = target.select(2, 1).fill_(-1.);
        let loss = geometry().decision_sign_loss(&DecisionValues {
            prediction: prediction.shallow_clone(),
            target,
            mask,
        });
        assert!((loss.double_value(&[]) - (2f64.ln())).abs() < 1e-6);
        loss.backward();
        let grad = prediction.grad();
        assert!(grad.double_value(&[0, 0, 0]) < 0.);
        assert!(grad.double_value(&[0, 0, 1]) > 0.);
        assert_eq!(grad.double_value(&[0, 0, 2]), 0.);
    }

    #[test]
    fn missing_populations_match_explicit_distinct_row_pairs_not_flattened_time() {
        let geometry = geometry();
        let (batch, origins, horizons, width) = (4usize, 3usize, 7usize, 3usize);
        let mut target = vec![f32::NAN; batch * origins * horizons];
        let mut mask = vec![0f32; target.len()];
        let mut features = vec![0f32; batch * origins * width];
        let value = |b: usize, o: usize, h: usize| b as f64 - 1. + o as f64 * 0.3 + h as f64 * 0.05;
        for b in 0..batch {
            for o in 0..origins {
                features[(b * origins + o) * width..(b * origins + o + 1) * width]
                    .copy_from_slice(&[0.5, b as f32 * 0.1, o as f32 * 0.05]);
                for h in 0..horizons {
                    if b < (o + h) % (batch + 1) {
                        let i = (b * origins + o) * horizons + h;
                        target[i] = value(b, o, h) as f32;
                        mask[i] = 1.;
                    }
                }
            }
        }
        let prediction = Tensor::zeros(
            [batch as i64, origins as i64, horizons as i64],
            (Kind::Float, Device::Cpu),
        )
        .set_requires_grad(true);
        let selected = DecisionValues {
            prediction,
            target: Tensor::from_slice(&target).reshape([
                batch as i64,
                origins as i64,
                horizons as i64,
            ]),
            mask: Tensor::from_slice(&mask).reshape([
                batch as i64,
                origins as i64,
                horizons as i64,
            ]),
        };
        let phi =
            Tensor::from_slice(&features).reshape([batch as i64, origins as i64, width as i64]);
        let output = geometry.objective(&selected, &phi);
        let (mut expected_loss, mut expected_mse) = (0., 0.);
        for h in 0..horizons {
            let mut diagnostic = [0f64; 7];
            let mut eligible = 0.;
            for o in 0..origins {
                let n = (o + h) % (batch + 1);
                diagnostic[5] += n as f64;
                diagnostic[6] += (n * n.saturating_sub(1)) as f64;
                let mut diagonal = 0.;
                let mut off_diagonal = 0.;
                for b in 0..n {
                    let e = value(b, o, h);
                    diagnostic[3] += e * e;
                    diagnostic[4] += e;
                    for other in 0..n {
                        let dot: f64 = (0..width)
                            .map(|f| {
                                features[(b * origins + o) * width + f] as f64
                                    * features[(other * origins + o) * width + f] as f64
                            })
                            .sum();
                        if b == other {
                            diagonal += e * e * dot;
                        } else {
                            off_diagonal += e * value(other, o, h) * dot;
                        }
                    }
                }
                if n >= 2 {
                    let pairs = (n * (n - 1)) as f64;
                    diagnostic[0] += off_diagonal / pairs;
                    diagnostic[1] += (diagonal + off_diagonal) / (n * n) as f64;
                    diagnostic[2] += diagonal / pairs;
                    eligible += 1.;
                }
            }
            for item in diagnostic.iter_mut().take(3) {
                *item /= eligible;
            }
            expected_loss += diagnostic[0] / horizons as f64;
            diagnostic[3] /= diagnostic[5];
            diagnostic[4] /= diagnostic[5];
            expected_mse += diagnostic[3] / horizons as f64;
            for (row, expected) in diagnostic.into_iter().enumerate() {
                let actual = output.diagnostics.double_value(&[row as i64, h as i64]);
                assert!(
                    (actual - expected).abs() < 2e-6,
                    "diagnostic {row}, horizon {h}: {actual} != {expected}"
                );
            }
        }
        assert!((output.moment_loss.double_value(&[]) - expected_loss).abs() < 2e-6);
        assert!((output.decision_mse.double_value(&[]) - expected_mse).abs() < 2e-6);
        assert!(!output.diagnostics.requires_grad());
        output.moment_loss.backward();
        assert_eq!(
            selected.prediction.grad().isfinite().all().int64_value(&[]),
            1
        );
        assert_eq!(
            (selected.prediction.grad() * selected.mask.eq(0.).to_kind(Kind::Float))
                .abs()
                .max()
                .double_value(&[]),
            0.
        );
        let absent = DecisionValues {
            prediction: Tensor::zeros([1, 2, 7], (Kind::Float, Device::Cpu))
                .set_requires_grad(true),
            target: Tensor::full([1, 2, 7], f64::NAN, (Kind::Float, Device::Cpu)),
            mask: Tensor::zeros([1, 2, 7], (Kind::Float, Device::Cpu)),
        };
        let empty = geometry.objective(
            &absent,
            &Tensor::ones([1, 2, 1], (Kind::Float, Device::Cpu)),
        );
        assert_eq!(empty.moment_loss.double_value(&[]), 0.);
        assert_eq!(empty.decision_mse.double_value(&[]), 0.);
        empty.moment_loss.backward();
        assert_eq!(absent.prediction.grad().abs().max().double_value(&[]), 0.);
    }

    #[test]
    fn diagonal_removal_cancels_independent_target_noise_and_detects_predictable_signal() {
        let geometry = geometry();
        let phi = Tensor::ones([2, 1, 1], (Kind::Float, Device::Cpu));
        let (mut u, mut v, mut mse) = (0., 0., 0.);
        // Enumerate the full independent Rademacher noise population, not a stochastic test.
        for first in [-1f32, 1.] {
            for second in [-1f32, 1.] {
                let values = DecisionValues {
                    prediction: Tensor::zeros([2, 1, 7], (Kind::Float, Device::Cpu)),
                    target: Tensor::from_slice(&[first, second])
                        .reshape([2, 1, 1])
                        .expand([2, 1, 7], true),
                    mask: Tensor::ones([2, 1, 7], (Kind::Float, Device::Cpu)),
                };
                let output = geometry.objective(&values, &phi);
                let realized_u = output.moment_loss.double_value(&[]);
                assert_eq!(
                    realized_u,
                    (first * second) as f64,
                    "negative U must not be clamped"
                );
                u += realized_u / 4.;
                v += output.diagnostics.double_value(&[1, 0]) / 4.;
                mse += output.decision_mse.double_value(&[]) / 4.;
            }
        }
        assert_eq!(u, 0.);
        assert_eq!(v, 0.5, "squared batch mean retains noise energy/N");
        assert_eq!(mse, 1.);
        let x = Tensor::from_slice(&[-1f32, 1.]).reshape([2, 1, 1]);
        let coefficient = Tensor::from(0f32).set_requires_grad(true);
        let values = DecisionValues {
            prediction: (&x * &coefficient).expand([2, 1, 7], true),
            target: x.expand([2, 1, 7], true),
            mask: Tensor::ones([2, 1, 7], (Kind::Float, Device::Cpu)),
        };
        let signal = geometry.objective(&values, &x);
        assert_eq!(signal.moment_loss.double_value(&[]), 1.);
        signal.moment_loss.backward();
        assert!(
            (coefficient.grad().double_value(&[]) + 2.).abs() < 1e-6,
            "conditional moment gradient did not remove predictable close signal"
        );
        let corrected = DecisionValues {
            prediction: values.target.shallow_clone(),
            ..values
        };
        let optimum = geometry.objective(&corrected, &x);
        assert_eq!(optimum.moment_loss.double_value(&[]), 0.);
        assert_eq!(optimum.decision_mse.double_value(&[]), 0.);
    }

    #[test]
    fn causal_summaries_use_source_beta_scale_and_complete_observed_lookbacks() {
        let config = ModelConfig {
            seq_len: 64,
            min_history: 16,
            future_calendar: false,
            ..ModelConfig::default()
        };
        let geometry = TemporalMoments::new(&config, Device::Cpu);
        let length = config.seq_len + config.pred_len;
        let aux_width = config.features.channels() as i64;
        let time = Tensor::arange(length, (Kind::Float, Device::Cpu));
        let close = time.square() * 0.0001;
        let batch = Batch::from_packed(
            Tensor::cat(
                &[
                    close.unsqueeze(-1).expand([length, 4], true).flatten(0, 1),
                    Tensor::ones([length], (Kind::Float, Device::Cpu)),
                    Tensor::zeros([length * aux_width], (Kind::Float, Device::Cpu)),
                    &time * 0.002,
                    Tensor::ones([1], (Kind::Float, Device::Cpu)),
                ],
                0,
            )
            .unsqueeze(0),
            config.seq_len as usize,
            config.pred_len as usize,
            aux_width as usize,
            config.pred_len as usize,
        );
        let stats = Statistics {
            sigma: Tensor::full([1, 4], 0.01, (Kind::Float, Device::Cpu)),
            beta: Tensor::full([1, 4], 0.5, (Kind::Float, Device::Cpu)),
            range: Tensor::zeros([1, 4], (Kind::Float, Device::Cpu)),
            log_close: Tensor::zeros([1, 4], (Kind::Float, Device::Cpu)),
            market: Tensor::zeros([1, 4], (Kind::Float, Device::Cpu)),
            mask: Tensor::ones([1, 4], (Kind::Float, Device::Cpu)),
        };
        let before = geometry.summaries(&batch, &stats);
        let neutral = ((0.0001f64 * (15f64.powi(2) - 7f64.powi(2)) - 0.5 * 8. * 0.002)
            / (0.01 * 8f64.sqrt()))
        .tanh();
        let market = (8. * 0.002 / (0.01 * 8f64.sqrt())).tanh();
        let mean_square: f64 = (8..=15)
            .map(|t| ((2 * t - 1) as f64 * 0.0001).powi(2))
            .sum::<f64>()
            / 8.;
        let realized = (0.5 * (mean_square / 0.01f64.powi(2)).ln()).tanh();
        for (summary, expected) in [neutral, market, realized, 1.].into_iter().enumerate() {
            assert!((before.double_value(&[0, 0, 4 + summary as i64]) - expected).abs() < 1e-6);
        }
        assert_eq!(
            before
                .narrow(2, 8, 24)
                .select(1, 0)
                .abs()
                .max()
                .double_value(&[]),
            0.,
            "unavailable long lookbacks were not zeroed"
        );
        let _ = batch.valid.narrow(1, 10, 1).fill_(0.);
        let after = geometry.summaries(&batch, &stats);
        assert!(
            (before.narrow(2, 0, 4).select(1, 0) - after.narrow(2, 0, 4).select(1, 0))
                .abs()
                .max()
                .double_value(&[])
                < 1e-6,
            "a hole before the one-step lookback changed it"
        );
        assert_eq!(
            after
                .narrow(2, 4, 4)
                .select(1, 0)
                .abs()
                .max()
                .double_value(&[]),
            0.,
            "an incomplete lookback was treated as observed"
        );
    }
}
