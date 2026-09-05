use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use tch::{Cuda, Device, Kind};

use super::data::{Dataset, Split, HORIZONS};
use super::model::ForecastModel;
use super::support::HorizonSupport;

pub const EVIDENCE_SCHEMA: u32 = 1;
pub const SCORING_CONTRACT: &str = "cumulative-return-hard-nll-v1;standardized-crps;raw-return-rmse-mae;categorical-randomized-pit20;calendar-week-bootstrap-v1";
pub const PIT_BUCKETS: usize = 20;
pub use super::FROZEN_SEEDS;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OriginScores {
    pub origin: usize,
    pub timestamp: i64,
    pub target: [f64; 6],
    pub nll: Option<[f64; 6]>,
    pub crps: Option<[f64; 6]>,
    pub squared_error: [f64; 6],
    pub standardized_squared_error: [f64; 6],
    pub absolute_error: [f64; 6],
    pub brier: Option<[f64; 6]>,
    pub pit: Option<[f64; 6]>,
    pub coverage: Option<[[f64; 3]; 6]>,
    pub tails: Option<[[f64; 2]; 6]>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HorizonMetrics {
    pub horizon: usize,
    pub nll: Option<f64>,
    pub crps: Option<f64>,
    pub rmse: f64,
    pub standardized_rmse: f64,
    pub mae: f64,
    pub brier: Option<f64>,
    pub pit_tv: Option<f64>,
    pub pit_histogram: Option<[f64; PIT_BUCKETS]>,
    pub coverage: Option<[f64; 3]>,
    pub tails: Option<[f64; 2]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Evaluation {
    pub schema: u32,
    pub scoring_contract: String,
    pub training_protocol_sha256: Option<String>,
    pub selected_epoch: usize,
    pub selected_step: usize,
    pub identity: String,
    pub split: String,
    pub model_kind: String,
    pub training_seed: u64,
    pub checkpoint_sha256: Option<String>,
    pub frozen: bool,
    pub origins: Vec<OriginScores>,
    pub horizons: Vec<HorizonMetrics>,
    pub latency_p95_ms: Option<f64>,
    pub latency_samples: usize,
    pub peak_device_memory_bytes: Option<u64>,
    pub device: String,
    pub precision: String,
    pub evaluation_batch_size: usize,
}

impl Evaluation {
    fn new(dataset: &Dataset, split: Split, model_kind: String) -> Result<Self> {
        Ok(Self {
            schema: EVIDENCE_SCHEMA,
            scoring_contract: SCORING_CONTRACT.to_owned(),
            training_protocol_sha256: None,
            selected_epoch: 0,
            selected_step: 0,
            identity: super::checkpoint::DataContract::from_dataset(dataset).sha256()?,
            split: format!("{split:?}"),
            model_kind,
            training_seed: 0,
            checkpoint_sha256: None,
            frozen: false,
            origins: Vec::new(),
            horizons: Vec::new(),
            latency_p95_ms: None,
            latency_samples: 0,
            peak_device_memory_bytes: None,
            device: String::new(),
            precision: String::new(),
            evaluation_batch_size: 0,
        })
    }

    pub fn authenticate_checkpoint(
        &mut self,
        manifest: &super::checkpoint::Manifest,
    ) -> Result<()> {
        ensure!(
            manifest.data.sha256()? == self.identity,
            "checkpoint and evaluation data contracts differ"
        );
        ensure!(
            format!("{:?}", manifest.model_kind) == self.model_kind,
            "checkpoint and evaluation model kinds differ"
        );
        self.checkpoint_sha256 = Some(manifest.weights_sha256.clone());
        self.frozen = manifest.frozen;
        self.training_seed = manifest.seed;
        self.selected_epoch = manifest.epoch;
        self.selected_step = manifest.step;
        let protocol = serde_json::to_vec(&(
            manifest.selection_epochs,
            manifest.planned_epochs,
            manifest.batch_size,
            manifest.learning_rate,
            &manifest.optimizer,
            &manifest.objective,
            manifest.data.supports.training_origin_count,
        ))?;
        let digest = ring::digest::digest(&ring::digest::SHA256, &protocol);
        self.training_protocol_sha256 =
            Some(digest.as_ref().iter().map(|b| format!("{b:02x}")).collect());
        Ok(())
    }

    pub fn selection_loss(&self) -> f64 {
        self.horizons
            .iter()
            .map(|h| h.nll.unwrap_or(h.standardized_rmse * h.standardized_rmse))
            .sum::<f64>()
            / self.horizons.len() as f64
    }

    pub fn summarize(&mut self) -> Result<()> {
        ensure!(!self.origins.is_empty(), "evaluation split has no origins");
        let count = self.origins.len() as f64;
        let probabilistic = self.origins[0].nll.is_some();
        for row in &self.origins {
            ensure!(
                row.target.iter().all(|value| value.is_finite())
                    && row
                        .squared_error
                        .iter()
                        .chain(&row.standardized_squared_error)
                        .chain(&row.absolute_error)
                        .all(|value| value.is_finite() && *value >= 0.0),
                "invalid point forecast evidence"
            );
            ensure!(
                [
                    row.nll.is_some(),
                    row.crps.is_some(),
                    row.brier.is_some(),
                    row.pit.is_some(),
                    row.coverage.is_some(),
                    row.tails.is_some()
                ]
                .iter()
                .all(|present| *present == probabilistic),
                "inconsistent probabilistic evidence"
            );
            if probabilistic {
                ensure!(
                    row.nll
                        .unwrap()
                        .iter()
                        .chain(row.crps.unwrap().iter())
                        .all(|value| value.is_finite() && *value >= -1e-12),
                    "invalid likelihood/CRPS evidence"
                );
                ensure!(
                    row.brier
                        .unwrap()
                        .iter()
                        .chain(row.pit.unwrap().iter())
                        .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
                    "invalid probability score evidence"
                );
                ensure!(
                    row.coverage
                        .unwrap()
                        .iter()
                        .flatten()
                        .chain(row.tails.unwrap().iter().flatten())
                        .all(|value| *value == 0.0 || *value == 1.0),
                    "invalid interval indicator evidence"
                );
            }
        }
        self.horizons = HORIZONS
            .iter()
            .enumerate()
            .map(|(h, &horizon)| {
                let mean = |get: fn(&OriginScores, usize) -> f64| {
                    self.origins.iter().map(|row| get(row, h)).sum::<f64>() / count
                };
                let mut metric = HorizonMetrics {
                    horizon,
                    rmse: mean(|row, h| row.squared_error[h]).sqrt(),
                    standardized_rmse: mean(|row, h| row.standardized_squared_error[h]).sqrt(),
                    mae: mean(|row, h| row.absolute_error[h]),
                    ..Default::default()
                };
                if probabilistic {
                    metric.nll = Some(mean(|row, h| row.nll.unwrap()[h]));
                    metric.crps = Some(mean(|row, h| row.crps.unwrap()[h]));
                    metric.brier = Some(mean(|row, h| row.brier.unwrap()[h]));
                    let mut histogram = [0.0; PIT_BUCKETS];
                    let mut coverage = [0.0; 3];
                    let mut tails = [0.0; 2];
                    for row in &self.origins {
                        let bucket = ((row.pit.unwrap()[h] * PIT_BUCKETS as f64) as usize)
                            .min(PIT_BUCKETS - 1);
                        histogram[bucket] += 1.0 / count;
                        for (i, value) in coverage.iter_mut().enumerate() {
                            *value += row.coverage.unwrap()[h][i] / count;
                        }
                        for (i, value) in tails.iter_mut().enumerate() {
                            *value += row.tails.unwrap()[h][i] / count;
                        }
                    }
                    metric.pit_tv = Some(
                        histogram
                            .iter()
                            .map(|p| (p - 1.0 / PIT_BUCKETS as f64).abs())
                            .sum::<f64>()
                            / 2.0,
                    );
                    metric.pit_histogram = Some(histogram);
                    metric.coverage = Some(coverage);
                    metric.tails = Some(tails);
                }
                metric
            })
            .collect();
        Ok(())
    }
}

pub fn evaluate_model(
    dataset: &Dataset,
    model: &ForecastModel,
    device: Device,
    split: Split,
    batch_size: usize,
    seed: u64,
) -> Result<Evaluation> {
    ensure!(batch_size > 0, "evaluation batch size must be positive");
    let Device::Cuda(index) = device else {
        bail!("model evaluation requires CUDA");
    };
    let origins = dataset.origins(split);
    ensure!(!origins.is_empty(), "evaluation split has no origins");
    let mut evaluation = Evaluation::new(dataset, split, format!("{:?}", model.kind()))?;
    evaluation.training_seed = seed;
    evaluation.evaluation_batch_size = batch_size;
    evaluation.precision = "bfloat16".to_owned();
    evaluation.device = Python::attach(|py| -> PyResult<String> {
        py.import("torch")?
            .getattr("cuda")?
            .call_method1("get_device_name", (index,))?
            .extract()
    })?;
    let _guard = tch::no_grad_guard();
    let warmup = dataset.batch(&origins[..1], device)?;
    for _ in 0..5 {
        let _ = model.forward(
            &warmup.endogenous,
            &warmup.exogenous,
            &warmup.validity,
            &warmup.future_clock,
            false,
        );
    }
    Cuda::synchronize(index as i64);
    cuda_memory(index, true)?;
    let count = origins.len().min(128);
    let mut latencies = Vec::with_capacity(count);
    for i in 0..count {
        let origin = origins[i * origins.len() / count];
        let batch = dataset.batch(&[origin], device)?;
        Cuda::synchronize(index as i64);
        let start = Instant::now();
        let _output = model.forward(
            &batch.endogenous,
            &batch.exogenous,
            &batch.validity,
            &batch.future_clock,
            false,
        );
        Cuda::synchronize(index as i64);
        latencies.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    latencies.sort_by(f64::total_cmp);
    evaluation.latency_p95_ms = Some(latencies[(count as f64 * 0.95).ceil() as usize - 1]);
    evaluation.latency_samples = count;
    for rows in origins.chunks(batch_size) {
        let batch = dataset.batch(rows, device)?;
        let output = model.forward(
            &batch.endogenous,
            &batch.exogenous,
            &batch.validity,
            &batch.future_clock,
            false,
        );
        let point = output.size().len() == 2;
        if point {
            ensure!(
                output.size() == [rows.len() as i64, 6],
                "invalid point prediction shape"
            );
            let predictions = Vec::<f64>::try_from(
                output
                    .to_device(Device::Cpu)
                    .to_kind(Kind::Double)
                    .view([-1]),
            )?;
            for (i, &origin) in rows.iter().enumerate() {
                evaluation.origins.push(score_point(
                    dataset,
                    origin,
                    &predictions[i * 6..(i + 1) * 6],
                )?);
            }
        } else {
            ensure!(
                output.size() == [rows.len() as i64, 6, 128],
                "invalid categorical prediction shape"
            );
            let log_probabilities = Vec::<f64>::try_from(
                output
                    .log_softmax(-1, Kind::Double)
                    .to_device(Device::Cpu)
                    .view([-1]),
            )?;
            for (i, &origin) in rows.iter().enumerate() {
                let logs = &log_probabilities[i * 768..(i + 1) * 768];
                let probabilities: Vec<_> = logs.iter().map(|logp| logp.exp()).collect();
                evaluation.origins.push(score_probabilities(
                    dataset,
                    origin,
                    &probabilities,
                    Some(logs),
                )?);
            }
        }
    }
    Cuda::synchronize(index as i64);
    evaluation.peak_device_memory_bytes = Some(cuda_memory(index, false)?);
    evaluation.summarize()?;
    Ok(evaluation)
}

fn cuda_memory(index: usize, reset: bool) -> Result<u64> {
    Python::attach(|py| -> PyResult<u64> {
        let cuda = py.import("torch")?.getattr("cuda")?;
        if reset {
            cuda.call_method1("reset_peak_memory_stats", (index,))?;
        }
        cuda.call_method1("max_memory_allocated", (index,))?
            .extract()
    })
    .context("read PyTorch CUDA allocator peak memory")
}

pub fn evaluate_unconditional(dataset: &Dataset, split: Split) -> Result<Evaluation> {
    let mut evaluation = Evaluation::new(dataset, split, "UnconditionalJeffreys".to_owned())?;
    let count = dataset.origins(Split::Train).len() as f64;
    let probabilities: Vec<f64> = dataset
        .supports
        .horizons
        .iter()
        .flat_map(|support| {
            support
                .marginal
                .iter()
                .map(|p| (p * count + 0.5) / (count + 64.0))
        })
        .collect();
    for &origin in dataset.origins(split) {
        evaluation
            .origins
            .push(score_probabilities(dataset, origin, &probabilities, None)?);
    }
    evaluation.summarize()?;
    Ok(evaluation)
}

fn score_point(dataset: &Dataset, origin: usize, prediction: &[f64]) -> Result<OriginScores> {
    ensure!(
        prediction.len() == 6 && prediction.iter().all(|v| v.is_finite()),
        "nonfinite forecast"
    );
    let target = dataset.target(origin);
    let mut squared_error = [0.0; 6];
    let mut standardized_squared_error = [0.0; 6];
    let mut absolute_error = [0.0; 6];
    for h in 0..6 {
        let residual = prediction[h] - target[h];
        standardized_squared_error[h] = residual.powi(2);
        let raw_residual = residual * dataset.sigma(origin) * (HORIZONS[h] as f64).sqrt();
        squared_error[h] = raw_residual.powi(2);
        absolute_error[h] = raw_residual.abs();
    }
    Ok(OriginScores {
        origin,
        timestamp: dataset.timestamp(origin),
        target,
        nll: None,
        crps: None,
        squared_error,
        standardized_squared_error,
        absolute_error,
        brier: None,
        pit: None,
        coverage: None,
        tails: None,
    })
}

fn score_probabilities(
    dataset: &Dataset,
    origin: usize,
    probabilities: &[f64],
    log_probabilities: Option<&[f64]>,
) -> Result<OriginScores> {
    ensure!(
        probabilities.len() == 768,
        "expected six 128-bin distributions"
    );
    let mut row = score_point(dataset, origin, &[0.0; 6])?;
    let mut nll = [0.0; 6];
    let mut crps = [0.0; 6];
    let mut brier = [0.0; 6];
    let mut pit = [0.0; 6];
    let mut coverage = [[0.0; 3]; 6];
    let mut tails = [[0.0; 2]; 6];
    for h in 0..6 {
        let support = &dataset.supports.horizons[h];
        let probability = &probabilities[h * 128..(h + 1) * 128];
        ensure!(
            probability.iter().all(|v| v.is_finite() && *v >= 0.0)
                && (probability.iter().sum::<f64>() - 1.0).abs() < 1e-6,
            "invalid probability distribution"
        );
        let target = row.target[h];
        let bin = support.bin(target);
        nll[h] = match log_probabilities {
            Some(logs) => -logs[h * 128 + bin],
            None => -probability[bin].ln(),
        };
        ensure!(
            nll[h].is_finite(),
            "zero target probability or nonfinite categorical NLL"
        );
        let (mean, score) = support.mean_and_crps(probability, target);
        crps[h] = score;
        row.standardized_squared_error[h] = (mean - target).powi(2);
        let raw_residual = (mean - target) * dataset.sigma(origin) * (HORIZONS[h] as f64).sqrt();
        row.squared_error[h] = raw_residual.powi(2);
        row.absolute_error[h] = raw_residual.abs();
        let positive = 1.0 - support.cdf(probability, 0.0);
        brier[h] = (positive - f64::from(target > 0.0)).powi(2);
        let below: f64 = probability[..bin].iter().sum();
        pit[h] = below + randomized_uniform(row.timestamp, h) * probability[bin];
        let quantiles = support.quantiles(
            probability,
            [
                (1.0 - 0.9) / 2.0,
                0.05,
                (1.0 - 0.8) / 2.0,
                0.25,
                0.75,
                (1.0 + 0.8) / 2.0,
                (1.0 + 0.9) / 2.0,
                0.95,
            ],
        );
        for (i, (low, high)) in [(3, 4), (2, 5), (0, 6)].into_iter().enumerate() {
            coverage[h][i] = f64::from(target >= quantiles[low] && target <= quantiles[high]);
        }
        tails[h] = [
            f64::from(target < quantiles[1]),
            f64::from(target > quantiles[7]),
        ];
    }
    row.nll = Some(nll);
    row.crps = Some(crps);
    row.brier = Some(brier);
    row.pit = Some(pit);
    row.coverage = Some(coverage);
    row.tails = Some(tails);
    Ok(row)
}

fn randomized_uniform(timestamp: i64, horizon: usize) -> f64 {
    let mut value =
        (timestamp as u64) ^ (horizon as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ 0x243f6a8885a308d3;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^= value >> 31;
    ((value >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HarStudentT {
    pub identity: String,
    coefficients: [[f64; 4]; 6],
    variance_correction: [f64; 6],
    degrees_of_freedom: [f64; 6],
}

impl HarStudentT {
    pub fn fit(dataset: &Dataset) -> Result<Self> {
        let origins = dataset.origins(Split::Train);
        ensure!(
            origins.len() >= 32,
            "HAR fit needs at least 32 training origins"
        );
        let mut gram = [[0.0; 4]; 4];
        let mut rhs = [[0.0; 4]; 6];
        let mut examples = Vec::with_capacity(origins.len());
        for &origin in origins {
            let x = har_features(dataset, origin)?;
            let mut response = [0.0; 6];
            for (h, &horizon) in HORIZONS.iter().enumerate() {
                let returns = dataset.history_returns(origin + horizon);
                let variance = returns[returns.len() - horizon..]
                    .iter()
                    .map(|r| r * r)
                    .sum::<f64>()
                    / horizon as f64;
                response[h] = variance.max(1e-20).ln();
                for i in 0..4 {
                    rhs[h][i] += x[i] * response[h];
                }
            }
            for i in 0..4 {
                for j in 0..4 {
                    gram[i][j] += x[i] * x[j];
                }
            }
            examples.push((origin, x, response));
        }
        let mut baseline = Self {
            identity: super::checkpoint::DataContract::from_dataset(dataset).sha256()?,
            coefficients: [[0.0; 4]; 6],
            variance_correction: [0.0; 6],
            degrees_of_freedom: [0.0; 6],
        };
        for h in 0..6 {
            baseline.coefficients[h] = solve_normal_equations(gram, rhs[h])?;
            baseline.variance_correction[h] = examples
                .iter()
                .map(|(_, x, y)| (y[h] - dot(x, &baseline.coefficients[h])).exp())
                .sum::<f64>()
                / examples.len() as f64;
            let mut best = (f64::INFINITY, 0.0);
            for df in [3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0, 30.0, 60.0] {
                let mut nll = 0.0;
                for (origin, x, _) in &examples {
                    let variance = dot(x, &baseline.coefficients[h]).exp()
                        * baseline.variance_correction[h]
                        / dataset.sigma(*origin).powi(2);
                    let scale = (variance * (df - 2.0) / df).sqrt();
                    nll -= student_log_density(dataset.target(*origin)[h] / scale, df) - scale.ln();
                }
                if nll < best.0 {
                    best = (nll, df);
                }
            }
            ensure!(best.0.is_finite(), "nonfinite HAR Student-t fit");
            baseline.degrees_of_freedom[h] = best.1;
        }
        Ok(baseline)
    }

    pub fn evaluate(&self, dataset: &Dataset, split: Split) -> Result<Evaluation> {
        ensure!(
            self.identity == super::checkpoint::DataContract::from_dataset(dataset).sha256()?,
            "HAR fit and evaluation dataset differ"
        );
        let mut evaluation = Evaluation::new(dataset, split, "HarStudentT".to_owned())?;
        for &origin in dataset.origins(split) {
            let x = har_features(dataset, origin)?;
            let mut probabilities = Vec::with_capacity(768);
            for h in 0..6 {
                let df = self.degrees_of_freedom[h];
                let variance = dot(&x, &self.coefficients[h]).exp() * self.variance_correction[h]
                    / dataset.sigma(origin).powi(2);
                let scale = (variance * (df - 2.0) / df).sqrt();
                ensure!(
                    scale.is_finite() && scale > 0.0,
                    "nonfinite HAR predictive scale"
                );
                probabilities.extend(student_masses(&dataset.supports.horizons[h], scale, df));
            }
            evaluation
                .origins
                .push(score_probabilities(dataset, origin, &probabilities, None)?);
        }
        evaluation.summarize()?;
        Ok(evaluation)
    }
}

fn har_features(dataset: &Dataset, origin: usize) -> Result<[f64; 4]> {
    let features = dataset.har_features(origin);
    ensure!(
        features.iter().all(|x| x.is_finite()),
        "nonfinite HAR features"
    );
    Ok(features)
}

fn dot(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn solve_normal_equations(mut gram: [[f64; 4]; 4], mut rhs: [f64; 4]) -> Result<[f64; 4]> {
    let ridge = (1..4).map(|i| gram[i][i]).sum::<f64>() * 1e-10;
    for i in 1..4 {
        gram[i][i] += ridge;
    }
    for pivot in 0..4 {
        let row = (pivot..4)
            .max_by(|&a, &b| gram[a][pivot].abs().total_cmp(&gram[b][pivot].abs()))
            .unwrap();
        gram.swap(pivot, row);
        rhs.swap(pivot, row);
        ensure!(
            gram[pivot][pivot].abs() > 1e-20,
            "singular HAR training features"
        );
        let scale = gram[pivot][pivot];
        for j in pivot..4 {
            gram[pivot][j] /= scale;
        }
        rhs[pivot] /= scale;
        for i in 0..4 {
            if i == pivot {
                continue;
            }
            let scale = gram[i][pivot];
            for j in pivot..4 {
                gram[i][j] -= scale * gram[pivot][j];
            }
            rhs[i] -= scale * rhs[pivot];
        }
    }
    Ok(rhs)
}

fn student_masses(support: &HorizonSupport, scale: f64, df: f64) -> Vec<f64> {
    let mut masses = Vec::with_capacity(128);
    for bin in 0..128 {
        let low = if bin == 0 {
            f64::NEG_INFINITY
        } else {
            support.edges[bin - 1] / scale
        };
        let high = if bin == 127 {
            f64::INFINITY
        } else {
            support.edges[bin] / scale
        };
        let mass = if low >= 0.0 {
            student_survival(low, df) - student_survival(high, df)
        } else {
            student_cdf(high, df) - student_cdf(low, df)
        };
        masses.push(mass);
    }
    masses
}

fn student_log_density(x: f64, df: f64) -> f64 {
    log_gamma((df + 1.0) / 2.0)
        - log_gamma(df / 2.0)
        - 0.5 * (df * std::f64::consts::PI).ln()
        - (df + 1.0) / 2.0 * (x * x / df).ln_1p()
}

fn student_survival(x: f64, df: f64) -> f64 {
    if x.is_infinite() {
        return if x > 0.0 { 0.0 } else { 1.0 };
    }
    let tail = 0.5 * regularized_beta(df / (df + x * x), df / 2.0, 0.5);
    if x >= 0.0 {
        tail
    } else {
        1.0 - tail
    }
}

fn student_cdf(x: f64, df: f64) -> f64 {
    if x == 0.0 {
        return 0.5;
    }
    let tail = 0.5 * regularized_beta(df / (df + x * x), df / 2.0, 0.5);
    if x < 0.0 {
        tail
    } else {
        1.0 - tail
    }
}

fn log_gamma(z: f64) -> f64 {
    let coefficients = [
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - log_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut x = 0.9999999999998099;
    for (i, coefficient) in coefficients.iter().enumerate() {
        x += coefficient / (z + i as f64 + 1.0);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

fn regularized_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front =
        (log_gamma(a + b) - log_gamma(a) - log_gamma(b) + a * x.ln() + b * (-x).ln_1p()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_fraction(x, a, b) / a
    } else {
        1.0 - front * beta_fraction(1.0 - x, b, a) / b
    }
}

fn beta_fraction(x: f64, a: f64, b: f64) -> f64 {
    let nonzero = |value: f64| {
        if value.abs() < 1e-300 {
            1e-300_f64.copysign(value)
        } else {
            value
        }
    };
    let mut c = 1.0;
    let mut d = 1.0 / nonzero(1.0 - (a + b) * x / (a + 1.0));
    let mut result = d;
    for m in 1..=256 {
        let m = m as f64;
        for coefficient in [
            m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0)),
        ] {
            d = 1.0 / nonzero(1.0 + coefficient * d);
            c = nonzero(1.0 + coefficient / c);
            let delta = d * c;
            result *= delta;
        }
        if (d * c - 1.0).abs() < 3e-14 {
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn student_cdf_matches_cauchy_and_known_t_quantile() {
        for x in [-10.0, -1.0, 0.0, 0.5, 2.0, 10.0] {
            assert!((student_cdf(x, 1.0) - (0.5 + x.atan() / std::f64::consts::PI)).abs() < 1e-12);
            assert!((student_cdf(x, 5.0) + student_cdf(-x, 5.0) - 1.0).abs() < 1e-12);
        }
        assert!((student_cdf(2.570581835636305, 5.0) - 0.975).abs() < 1e-10);
    }

    #[test]
    fn pit_randomization_is_repeatable_and_non_degenerate() {
        let values: Vec<_> = (0..10000)
            .map(|i| randomized_uniform(1_700_000_000 + i * 300, 3))
            .collect();
        assert!(values.iter().all(|v| *v > 0.0 && *v < 1.0));
        assert!((values.iter().sum::<f64>() / values.len() as f64 - 0.5).abs() < 0.01);
        assert_eq!(randomized_uniform(42, 3), randomized_uniform(42, 3));
        assert_ne!(randomized_uniform(42, 3), randomized_uniform(42, 4));
    }
}
