use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

use super::data::HORIZONS;

pub const BINS: usize = 128;
const SEGMENTS: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Supports {
    pub horizons: Vec<HorizonSupport>,
    pub training_origin_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HorizonSupport {
    pub horizon: usize,
    pub edges: Vec<f64>,
    pub marginal: Vec<f64>,
    pub bins: Vec<BinLaw>,
}

/// Interior quantiles interpolate fitted training quantiles; tails remain open.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BinLaw {
    pub mean: f64,
    pub second_moment: f64,
    pub pair_distance: f64,
    pub law: ConditionalLaw,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConditionalLaw {
    LowerExponential { edge: f64, scale: f64 },
    Quantiles(Vec<f64>),
    UpperExponential { edge: f64, scale: f64 },
}

impl Supports {
    pub fn fit(targets: &[[f64; 6]]) -> Result<Self> {
        ensure!(
            targets.len() >= BINS * 2,
            "at least 256 training origins are required to fit return supports"
        );
        let horizons = HORIZONS
            .iter()
            .enumerate()
            .map(|(h, &horizon)| {
                HorizonSupport::fit(horizon, targets.iter().map(|row| row[h]).collect())
            })
            .collect::<Result<Vec<_>>>()?;
        let result = Self {
            horizons,
            training_origin_count: targets.len(),
        };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.training_origin_count >= BINS * 2 && self.horizons.len() == HORIZONS.len(),
            "invalid support dimensions"
        );
        for (support, &horizon) in self.horizons.iter().zip(&HORIZONS) {
            ensure!(support.horizon == horizon, "support horizon mismatch");
            support.validate()?;
        }
        Ok(())
    }
}

fn empirical_quantile(sorted: &[f64], q: f64) -> f64 {
    let index = q * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower as f64)
}

impl HorizonSupport {
    fn fit(horizon: usize, mut values: Vec<f64>) -> Result<Self> {
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite training return"
        );
        values.sort_unstable_by(f64::total_cmp);
        ensure!(
            values[0] < values[values.len() - 1],
            "constant training targets cannot fit open tails"
        );
        let edges: Vec<_> = (1..BINS)
            .map(|b| empirical_quantile(&values, b as f64 / BINS as f64))
            .collect();
        let mut members = vec![Vec::new(); BINS];
        for &value in &values {
            members[edges.partition_point(|&edge| edge <= value)].push(value);
        }
        let mut bins = Vec::with_capacity(BINS);
        for b in 0..BINS {
            let sample = &members[b];
            let law = if b == 0 || b == BINS - 1 {
                let edge = if b == 0 { edges[0] } else { edges[BINS - 2] };
                let excess = sample.iter().map(|v| (v - edge).abs()).sum::<f64>();
                // A tied edge can leave a tail empty. A train-fitted global scale keeps its
                // law proper while its empirical marginal remains zero.
                let scale = if excess > 0.0 {
                    excess / sample.len() as f64
                } else {
                    (values[values.len() - 1] - values[0]) / BINS as f64
                };
                if b == 0 {
                    ConditionalLaw::LowerExponential { edge, scale }
                } else {
                    ConditionalLaw::UpperExponential { edge, scale }
                }
            } else {
                let lower = edges[b - 1];
                let upper = edges[b];
                let mut knots = vec![lower];
                for j in 1..SEGMENTS {
                    knots.push(if sample.is_empty() {
                        lower + (upper - lower) * j as f64 / SEGMENTS as f64
                    } else {
                        empirical_quantile(sample, j as f64 / SEGMENTS as f64)
                    });
                }
                knots.push(upper);
                ConditionalLaw::Quantiles(knots)
            };
            bins.push(BinLaw::new(law));
        }
        Ok(Self {
            horizon,
            edges,
            marginal: members
                .iter()
                .map(|row| row.len() as f64 / values.len() as f64)
                .collect(),
            bins,
        })
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.edges.len() == BINS - 1 && self.bins.len() == BINS && self.marginal.len() == BINS,
            "invalid categorical support size"
        );
        ensure!(
            self.edges.iter().all(|v| v.is_finite()) && self.edges.windows(2).all(|w| w[0] <= w[1]),
            "unordered or nonfinite support cuts"
        );
        ensure!(
            self.marginal.iter().all(|p| p.is_finite() && *p >= 0.0)
                && (self.marginal.iter().sum::<f64>() - 1.0).abs() < 1e-10,
            "invalid fitted marginal"
        );
        for (i, bin) in self.bins.iter().enumerate() {
            match &bin.law {
                ConditionalLaw::LowerExponential { edge, scale } => ensure!(
                    i == 0 && *edge == self.edges[0] && scale.is_finite() && *scale > 0.0,
                    "invalid lower tail"
                ),
                ConditionalLaw::UpperExponential { edge, scale } => ensure!(
                    i == BINS - 1
                        && *edge == self.edges[BINS - 2]
                        && scale.is_finite()
                        && *scale > 0.0,
                    "invalid upper tail"
                ),
                ConditionalLaw::Quantiles(knots) => ensure!(
                    i > 0
                        && i < BINS - 1
                        && knots.len() == SEGMENTS + 1
                        && knots[0] == self.edges[i - 1]
                        && knots[SEGMENTS] == self.edges[i]
                        && knots.iter().all(|v| v.is_finite())
                        && knots.windows(2).all(|w| w[0] <= w[1]),
                    "invalid interior quantiles"
                ),
            }
            let expected = BinLaw::new(bin.law.clone());
            ensure!(
                bin.mean.is_finite()
                    && bin.second_moment.is_finite()
                    && bin.pair_distance.is_finite()
                    && (bin.mean - expected.mean).abs() < 1e-10
                    && (bin.second_moment - expected.second_moment).abs() < 1e-10
                    && (bin.pair_distance - expected.pair_distance).abs() < 1e-10,
                "support moments disagree with fitted law"
            );
        }
        Ok(())
    }

    pub fn bin(&self, value: f64) -> usize {
        self.edges.partition_point(|&edge| edge <= value)
    }

    pub fn mean(&self, probabilities: &[f64]) -> f64 {
        probabilities
            .iter()
            .zip(&self.bins)
            .map(|(p, b)| p * b.mean)
            .sum()
    }

    pub fn cdf(&self, probabilities: &[f64], value: f64) -> f64 {
        self.cdf_interval(probabilities, value).1
    }

    pub fn cdf_interval(&self, probabilities: &[f64], value: f64) -> (f64, f64) {
        probabilities
            .iter()
            .zip(&self.bins)
            .fold((0.0, 0.0), |(lo, hi), (p, b)| {
                let (left, right) = b.cdf_interval(value);
                (lo + p * left, hi + p * right)
            })
    }

    pub fn quantile(&self, probabilities: &[f64], q: f64) -> f64 {
        let mut cumulative = 0.0;
        for (p, bin) in probabilities.iter().zip(&self.bins) {
            if *p > 0.0 && cumulative + p >= q {
                return bin.quantile(((q - cumulative) / p).clamp(0.0, 1.0));
            }
            cumulative += p;
        }
        f64::INFINITY
    }

    /// Evaluate sorted quantiles in one pass over the categorical masses.
    pub fn quantiles<const N: usize>(&self, probabilities: &[f64], qs: [f64; N]) -> [f64; N] {
        debug_assert!(qs.windows(2).all(|w| w[0] <= w[1]));
        let mut result = [f64::INFINITY; N];
        let (mut index, mut cumulative) = (0, 0.0);
        for (result, q) in result.iter_mut().zip(qs) {
            while index < probabilities.len().min(self.bins.len()) {
                let p = probabilities[index];
                if p > 0.0 && cumulative + p >= q {
                    *result = self.bins[index].quantile(((q - cumulative) / p).clamp(0.0, 1.0));
                    break;
                }
                cumulative += p;
                index += 1;
            }
        }
        result
    }

    pub fn crps(&self, probabilities: &[f64], value: f64) -> f64 {
        self.mean_and_crps(probabilities, value).1
    }

    pub fn mean_and_crps(&self, probabilities: &[f64], value: f64) -> (f64, f64) {
        let mut absolute = 0.0;
        let mut half_pair = 0.0;
        let (mut mass, mut moment) = (0.0, 0.0);
        for (p, bin) in probabilities.iter().zip(&self.bins) {
            absolute += p * bin.absolute(value);
            half_pair += 0.5 * p * p * bin.pair_distance + p * (mass * bin.mean - moment);
            mass += p;
            moment += p * bin.mean;
        }
        (moment, (absolute - half_pair).max(0.0))
    }
}

impl BinLaw {
    fn new(law: ConditionalLaw) -> Self {
        let (mean, second_moment, pair_distance) = match &law {
            ConditionalLaw::LowerExponential { edge, scale } => (
                edge - scale,
                edge * edge - 2.0 * edge * scale + 2.0 * scale * scale,
                *scale,
            ),
            ConditionalLaw::UpperExponential { edge, scale } => (
                edge + scale,
                edge * edge + 2.0 * edge * scale + 2.0 * scale * scale,
                *scale,
            ),
            ConditionalLaw::Quantiles(knots) => {
                let n = (knots.len() - 1) as f64;
                let (mut mean, mut second, mut pair) = (0.0, 0.0, 0.0);
                for (j, w) in knots.windows(2).enumerate() {
                    let (a, d, u, du) = (w[0], w[1] - w[0], j as f64 / n, 1.0 / n);
                    mean += (a + 0.5 * d) * du;
                    second += (a * a + a * d + d * d / 3.0) * du;
                    pair += 2.0
                        * du
                        * ((2.0 * u - 1.0) * (a + d / 2.0) + 2.0 * du * (a / 2.0 + d / 3.0));
                }
                (mean, second, pair.max(0.0))
            }
        };
        Self {
            mean,
            second_moment,
            pair_distance,
            law,
        }
    }

    fn quantile(&self, q: f64) -> f64 {
        match &self.law {
            ConditionalLaw::LowerExponential { edge, scale } => edge + scale * q.ln(),
            ConditionalLaw::UpperExponential { edge, scale } => edge - scale * (-q).ln_1p(),
            ConditionalLaw::Quantiles(knots) => empirical_quantile(knots, q),
        }
    }

    fn cdf_interval(&self, value: f64) -> (f64, f64) {
        match &self.law {
            ConditionalLaw::LowerExponential { edge, scale } => {
                let p = ((value - edge) / scale).exp().min(1.0);
                (p, p)
            }
            ConditionalLaw::UpperExponential { edge, scale } => {
                let p = -(-((value - edge) / scale).max(0.0)).exp_m1();
                (p, p)
            }
            ConditionalLaw::Quantiles(knots) => {
                if value < knots[0] {
                    return (0.0, 0.0);
                }
                if value > knots[knots.len() - 1] {
                    return (1.0, 1.0);
                }
                let (mut left, mut right) = (0.0, 0.0);
                for w in knots.windows(2) {
                    if w[0] == w[1] {
                        left += f64::from(value > w[0]);
                        right += f64::from(value >= w[0]);
                    } else {
                        let p = ((value - w[0]) / (w[1] - w[0])).clamp(0.0, 1.0);
                        left += p;
                        right += p;
                    }
                }
                let n = (knots.len() - 1) as f64;
                (left / n, right / n)
            }
        }
    }

    fn absolute(&self, value: f64) -> f64 {
        match &self.law {
            ConditionalLaw::LowerExponential { edge, scale } => {
                if value >= *edge {
                    value - self.mean
                } else {
                    self.mean - value + 2.0 * scale * ((value - edge) / scale).exp()
                }
            }
            ConditionalLaw::UpperExponential { edge, scale } => {
                if value <= *edge {
                    self.mean - value
                } else {
                    value - self.mean + 2.0 * scale * (-(value - edge) / scale).exp()
                }
            }
            ConditionalLaw::Quantiles(knots) => {
                if value <= knots[0] {
                    return self.mean - value;
                }
                if value >= knots[knots.len() - 1] {
                    return value - self.mean;
                }
                knots
                    .windows(2)
                    .map(|w| {
                        if value <= w[0] {
                            (w[0] + w[1]) / 2.0 - value
                        } else if value >= w[1] {
                            value - (w[0] + w[1]) / 2.0
                        } else {
                            ((value - w[0]).powi(2) + (w[1] - value).powi(2))
                                / (2.0 * (w[1] - w[0]))
                        }
                    })
                    .sum::<f64>()
                    / (knots.len() - 1) as f64
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_cdf(bin: &BinLaw, value: f64) -> (f64, f64) {
        let ConditionalLaw::Quantiles(knots) = &bin.law else {
            return bin.cdf_interval(value);
        };
        let (mut left, mut right) = (0.0, 0.0);
        for w in knots.windows(2) {
            if w[0] == w[1] {
                left += f64::from(value > w[0]);
                right += f64::from(value >= w[0]);
            } else {
                let p = ((value - w[0]) / (w[1] - w[0])).clamp(0.0, 1.0);
                left += p;
                right += p;
            }
        }
        let count = (knots.len() - 1) as f64;
        (left / count, right / count)
    }

    fn reference_absolute(bin: &BinLaw, value: f64) -> f64 {
        let ConditionalLaw::Quantiles(knots) = &bin.law else {
            return bin.absolute(value);
        };
        knots
            .windows(2)
            .map(|w| {
                if value <= w[0] {
                    (w[0] + w[1]) / 2.0 - value
                } else if value >= w[1] {
                    value - (w[0] + w[1]) / 2.0
                } else {
                    ((value - w[0]).powi(2) + (w[1] - value).powi(2)) / (2.0 * (w[1] - w[0]))
                }
            })
            .sum::<f64>()
            / (knots.len() - 1) as f64
    }

    #[test]
    fn optimized_scores_preserve_segment_laws_and_quantile_boundaries() {
        let targets: Vec<_> = (0..4096)
            .map(|i| {
                [if i % 3 == 0 {
                    0.0
                } else {
                    i as f64 / 2048.0 - 1.0
                }; 6]
            })
            .collect();
        let supports = Supports::fit(&targets).unwrap();
        let support = &supports.horizons[0];
        let qs = [0.0, (1.0 - 0.9) / 2.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0];
        for pattern in 0..4 {
            let mut p: Vec<f64> = (0..BINS)
                .map(|i| match pattern {
                    0 => 1.0,
                    1 => f64::from(i % 7 == 0),
                    2 => f64::from(i == 63),
                    _ => ((i * 37 % 131) as f64).exp(),
                })
                .collect();
            let total = p.iter().sum::<f64>();
            for value in &mut p {
                *value /= total;
            }
            let quantiles = support.quantiles(&p, qs);
            for (q, value) in qs.into_iter().zip(quantiles) {
                assert_eq!(value, support.quantile(&p, q));
            }
            for value in [-100.0, -1.0, -0.1, 0.0, 0.3, 1.0, 100.0]
                .into_iter()
                .chain(support.edges.iter().copied())
            {
                let (mut absolute, mut half_pair, mut mass, mut moment) = (0.0, 0.0, 0.0, 0.0);
                let (mut lo, mut hi) = (0.0, 0.0);
                for (&probability, bin) in p.iter().zip(&support.bins) {
                    absolute += probability * reference_absolute(bin, value);
                    half_pair += 0.5 * probability * probability * bin.pair_distance
                        + probability * (mass * bin.mean - moment);
                    mass += probability;
                    moment += probability * bin.mean;
                    let (left, right) = reference_cdf(bin, value);
                    lo += probability * left;
                    hi += probability * right;
                }
                let (mean, crps) = support.mean_and_crps(&p, value);
                assert_eq!(mean, support.mean(&p));
                assert!((crps - (absolute - half_pair).max(0.0)).abs() < 1e-12);
                assert_eq!(support.cdf_interval(&p, value), (lo, hi));
            }
        }
    }

    #[test]
    fn open_tails_and_moments_are_one_consistent_law() {
        let targets: Vec<_> = (0..4096)
            .map(|i| [((i as f64 + 0.5) / 4096.0).ln(); 6])
            .collect();
        let supports = Supports::fit(&targets).unwrap();
        let s = &supports.horizons[0];
        let p = &s.marginal;
        assert!(s.quantile(p, 1e-12) < s.edges[0]);
        assert!(s.quantile(p, 1.0 - 1e-12) > s.edges[BINS - 2]);
        for q in [0.001, 0.05, 0.25, 0.5, 0.9, 0.999] {
            assert!((s.cdf(p, s.quantile(p, q)) - q).abs() < 1e-10);
        }
        let numerical_mean = (0..100_000)
            .map(|i| s.quantile(p, (i as f64 + 0.5) / 100_000.0))
            .sum::<f64>()
            / 100_000.0;
        assert!((numerical_mean - s.mean(p)).abs() < 1e-4);
        let y = -0.7;
        let numerical_crps = (0..20_000)
            .map(|i| {
                let q = (i as f64 + 0.5) / 20_000.0;
                let error = y - s.quantile(p, q);
                2.0 * error * (q - f64::from(error < 0.0))
            })
            .sum::<f64>()
            / 20_000.0;
        assert!((numerical_crps - s.crps(p, y)).abs() < 1e-4);
    }

    #[test]
    fn repeated_training_quantiles_preserve_probability_atoms() {
        let targets: Vec<_> = (0..4096)
            .map(|i| {
                [if i % 3 == 0 {
                    0.0
                } else {
                    i as f64 / 4096.0 - 0.5
                }; 6]
            })
            .collect();
        let supports = Supports::fit(&targets).unwrap();
        let s = &supports.horizons[0];
        let (left, right) = s.cdf_interval(&s.marginal, 0.0);
        assert!(right - left > 0.25);
        assert!(s.crps(&s.marginal, 0.0).is_finite());
        let mut corrupted = supports;
        corrupted.horizons[0].bins[0].mean += 1.0;
        assert!(corrupted.validate().is_err());
    }
}
