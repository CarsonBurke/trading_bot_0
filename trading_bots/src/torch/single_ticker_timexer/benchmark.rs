use std::{path::PathBuf, time::Instant};

use anyhow::{ensure, Result};
use clap::Args;
use pyo3::prelude::*;
use tch::{nn, Cuda, Device, Kind, Tensor};

use super::{
    data::{Batch, Dataset, Split},
    model::{ForecastModel, ModelKind},
    reports, runner,
};

#[derive(Clone, Debug, Args)]
pub struct BenchmarkArgs {
    #[arg(long)]
    pub ticker: Option<String>,
    #[arg(long, default_value_os_t = crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 128)]
    pub steps: usize,
    #[arg(long, default_value_t = 16)]
    pub warmup: usize,
}

fn tensors(batch: &Batch) -> [&Tensor; 7] {
    [
        &batch.endogenous, &batch.exogenous, &batch.validity, &batch.future_clock,
        &batch.targets, &batch.bins, &batch.weights,
    ]
}

fn synchronize(device: Device) {
    if let Device::Cuda(index) = device {
        Cuda::synchronize(index as i64);
    }
}

pub fn run(args: BenchmarkArgs) -> Result<()> {
    ensure!(args.batch_size > 0 && args.steps >= 32 && args.warmup >= 8,
        "benchmark requires a positive batch size, at least 32 measured steps and 8 warmups");
    ensure!(!args.output.exists(), "benchmark output already exists");
    let ticker = runner::configured_ticker(args.ticker.as_deref())?;
    let device = runner::cuda_device()?;
    let _backward = crate::torch::cuda::cfg::disable_autograd_multithreading();
    let mut dataset = Dataset::load(&args.data_dir, &ticker)?;
    let origins = dataset.origins(Split::Train).to_vec();
    ensure!(origins.len() >= args.batch_size, "insufficient training origins");
    let rows: Vec<Vec<usize>> = (0..args.steps + args.warmup).map(|i| {
        (0..args.batch_size).map(|j| origins[(i * 7919 + j * 997) % origins.len()]).collect()
    }).collect();
    let initial_memory = cuda_memory(false)?;
    let prepared = Instant::now();
    dataset.prepare(device);
    synchronize(device);
    let preparation_ms = prepared.elapsed().as_secs_f64() * 1000.;
    let cache_memory = cuda_memory(false)?.saturating_sub(initial_memory);
    for selected in [&rows[0][..], &rows[1][..], &rows[2][..1]] {
        let reference = dataset.batch_reference(selected, device)?;
        let optimized = dataset.batch(selected, device)?;
        for (a, b) in tensors(&reference).into_iter().zip(tensors(&optimized)) {
            ensure!(a.equal(b), "cached batch differs from reference");
        }
    }
    let mut measurements = Vec::new();
    // Alternate order to reduce clock/temperature bias. Both paths use the repaired BF16 model.
    for trial in 0..3 {
        for cached in if trial % 2 == 0 { [false, true] } else { [true, false] } {
            for selected in &rows[..args.warmup] {
                let _ = batch(&dataset, selected, device, cached)?;
            }
            synchronize(device);
            let started = Instant::now();
            for selected in &rows[args.warmup..] {
                let _ = batch(&dataset, selected, device, cached)?;
            }
            synchronize(device);
            let ms = started.elapsed().as_secs_f64() * 1000. / args.steps as f64;
            measurements.push((format!("{} batch ms", label(cached)), ms));
        }
        let mut states = Vec::new();
        for cached in if trial % 2 == 0 { [false, true] } else { [true, false] } {
            tch::manual_seed(super::FROZEN_SEEDS[0] as i64);
            Cuda::manual_seed_all(super::FROZEN_SEEDS[0]);
            let store = nn::VarStore::new(device);
            let model = ForecastModel::new(&store.root(), ModelKind::SingleTickerTimeXer);
            let mut optimizer = runner::optimizer(&store, 0.0003)?;
            let mut loss_sum = Tensor::zeros([], (Kind::Double, device));
            let mut started = Instant::now();
            for (i, selected) in rows.iter().enumerate() {
                if i == args.warmup {
                    synchronize(device);
                    cuda_memory(true)?;
                    started = Instant::now();
                }
                let batch = batch(&dataset, selected, device, cached)?;
                optimizer.zero_grad();
                let logits = model.forward(&batch.endogenous, &batch.exogenous,
                    &batch.validity, &batch.future_clock, true);
                let loss = runner::hard_loss(&logits, &batch.bins, &batch.weights);
                if cached {
                    runner::check_objective(&loss)?;
                } else {
                    ensure!(loss.double_value(&[]).is_finite(), "nonfinite reference objective");
                }
                loss.backward();
                optimizer.step();
                loss_sum += loss.detach().to_kind(Kind::Double) * selected.len() as f64;
            }
            synchronize(device);
            let ms = started.elapsed().as_secs_f64() * 1000. / args.steps as f64;
            ensure!(loss_sum.double_value(&[]).is_finite(), "nonfinite benchmark objective");
            measurements.push((format!("{} peak CUDA bytes including cache", label(cached)), cuda_memory(false)? as f64));
            measurements.push((format!("{} training step ms", label(cached)), ms));
            measurements.push((format!("{} training origins per second", label(cached)),
                args.batch_size as f64 * 1000. / ms));
            let mut parameters: Vec<_> = store.variables().into_iter().collect();
            parameters.sort_by(|a, b| a.0.cmp(&b.0));
            let parameters: Vec<_> = parameters.into_iter().map(|(_, value)|
                value.detach().to_device(Device::Cpu)).collect();
            states.push(parameters);
        }
        let mut max_difference = 0.0_f64;
        for (a, b) in states[0].iter().zip(&states[1]) {
            let difference = (a - b).abs().max().double_value(&[]);
            ensure!(difference.is_finite(), "nonfinite updated benchmark parameters");
            max_difference = max_difference.max(difference);
        }
        measurements.push(("parameter maximum absolute difference".to_owned(), max_difference));
        ensure!(max_difference <= 1e-6, "optimized updates differ from reference: {max_difference}");
    }
    measurements.push(("cache preparation ms".to_owned(), preparation_ms));
    measurements.push(("resident cache bytes".to_owned(), cache_memory as f64));
    reports::write_benchmark(&args.output, &ticker, args.batch_size, args.steps, &measurements)?;
    println!("TimeXer performance and equivalence reports: {}", args.output.display());
    Ok(())
}

fn label(cached: bool) -> &'static str {
    if cached { "optimized" } else { "reference" }
}

fn batch(dataset: &Dataset, rows: &[usize], device: Device, cached: bool) -> Result<Batch> {
    if cached { dataset.batch(rows, device) } else { dataset.batch_reference(rows, device) }
}

fn cuda_memory(reset: bool) -> Result<u64> {
    Python::attach(|py| -> Result<u64> {
        let cuda = py.import("torch")?.getattr("cuda")?;
        if reset {
            cuda.call_method1("reset_peak_memory_stats", (0,))?;
        }
        Ok(cuda.call_method1("max_memory_allocated", (0,))?.extract()?)
    })
}
