use std::{
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use anyhow::{ensure, Context, Result};
use clap::Args;
use nvml_wrapper::Nvml;
use pyo3::prelude::*;
use shared::report::{write_report, Report, ReportKind, ReportSeries, ScaleKind};
use tch::{nn, Cuda, Kind, Tensor};

use super::{
    compute::{Engine, OptimizerKind},
    model::{ModelConfig, SegmentModel, CHANNELS},
};

#[derive(Clone, Debug, Args)]
pub struct BenchmarkArgs {
    #[command(flatten)]
    pub model: ModelConfig,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 20)]
    pub steps: usize,
    #[arg(long, default_value_t = 5)]
    pub warmup: usize,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub fused: bool,
    #[arg(long, value_enum, default_value_t = OptimizerKind::PolarExpress)]
    pub optimizer: OptimizerKind,
    #[arg(long)]
    pub learning_rate: Option<f64>,
    #[arg(long)]
    pub profile: bool,
    /// Verify native/fused Adam update equivalence independently of the timed optimizer.
    #[arg(long)]
    pub verify_optimizer: bool,
}

pub fn cuda_memory(reset: bool) -> Result<(u64, u64)> {
    Python::attach(|py| -> Result<_> {
        let cuda = py.import("torch")?.getattr("cuda")?;
        cuda.call_method0("init")?;
        if reset {
            cuda.call_method1("reset_peak_memory_stats", (0,))?;
        }
        Ok((
            cuda.call_method1("max_memory_allocated", (0,))?.extract()?,
            cuda.call_method1("max_memory_reserved", (0,))?.extract()?,
        ))
    })
}

/// NVML identifies the same physical device as CUDA, including CUDA_VISIBLE_DEVICES remapping.
pub struct HardwareSampler {
    stop: mpsc::Sender<()>,
    thread: Option<thread::JoinHandle<Result<Vec<[f64; 5]>>>>,
}

impl HardwareSampler {
    pub fn start() -> Result<Self> {
        let uuid = Python::attach(|py| -> Result<String> {
            Ok(py
                .import("torch")?
                .getattr("cuda")?
                .call_method1("get_device_properties", (0,))?
                .getattr("uuid")?
                .str()?
                .to_str()?
                .to_owned())
        })?;
        let nvml = Nvml::init()?;
        nvml.device_by_uuid(uuid.as_str())
            .context("map CUDA device UUID to NVML")?;
        let (stop, receive) = mpsc::channel();
        let thread = thread::spawn(move || -> Result<_> {
            let device = nvml.device_by_uuid(uuid.as_str())?;
            let started = Instant::now();
            let mut values = Vec::new();
            loop {
                let use_rates = device.utilization_rates()?;
                let memory = device.memory_info()?;
                values.push([
                    started.elapsed().as_secs_f64() * 1000.,
                    use_rates.gpu as f64,
                    use_rates.memory as f64,
                    memory.used as f64 / 1048576.,
                    device.power_usage()? as f64 / 1000.,
                ]);
                if receive.recv_timeout(Duration::from_millis(100)).is_ok() {
                    break;
                }
            }
            Ok(values)
        });
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }

    pub fn finish(mut self) -> Result<Vec<[f64; 5]>> {
        let _ = self.stop.send(());
        self.thread
            .take()
            .unwrap()
            .join()
            .map_err(|_| anyhow::anyhow!("hardware sampler panicked"))?
    }
}

impl Drop for HardwareSampler {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

fn chart(
    output: &Path,
    base: &str,
    title: &str,
    unit: &str,
    steps: Vec<u64>,
    series: Vec<ReportSeries>,
) -> Result<()> {
    write_report(
        output.join(format!("{base}.report.bin")),
        &Report {
            title: title.to_owned(),
            x_label: Some("measured step or elapsed milliseconds".to_owned()),
            y_label: Some(unit.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines { steps, series },
        },
    )?;
    Ok(())
}

pub fn write_hardware(output: &Path, title: &str, samples: &[[f64; 5]]) -> Result<()> {
    let (uuid, total_memory) = Python::attach(|py| -> Result<(String, u64)> {
        let device = py
            .import("torch")?
            .getattr("cuda")?
            .call_method1("get_device_properties", (0,))?;
        Ok((
            device.getattr("uuid")?.str()?.to_str()?.to_owned(),
            device.getattr("total_memory")?.extract()?,
        ))
    })?;
    let nvml = Nvml::init()?;
    let power_limit = nvml
        .device_by_uuid(uuid.as_str())?
        .power_management_limit()? as f64
        / 1000.;
    let memory_mib = total_memory as f64 / 1048576.;
    let peak_memory = samples.iter().map(|row| row[3]).fold(0., f64::max);
    let peak_power = samples.iter().map(|row| row[4]).fold(0., f64::max);
    chart(output, "timexer_segment_hardware", &format!("{title} | peak {peak_memory:.0}/{memory_mib:.0} MiB, {peak_power:.0}/{power_limit:.0} W"),
        "percent of device capacity", samples.iter().enumerate().map(|(i, row)| (row[0] as u64).max(i as u64)).collect(),
        [(1, "GPU busy", 1.), (2, "memory controller busy", 1.), (3, "VRAM used", 100. / memory_mib),
            (4, "power limit used", 100. / power_limit)].into_iter().map(|(column, label, scale)| ReportSeries {
                label: label.to_owned(), values: samples.iter().map(|row| (row[column] * scale) as f32).collect(),
            }).collect())
}

fn timed<T>(operation: impl FnOnce() -> Result<T>) -> Result<(T, f64)> {
    Cuda::synchronize(0);
    let started = Instant::now();
    let result = operation()?;
    Cuda::synchronize(0);
    Ok((result, started.elapsed().as_secs_f64() * 1000.))
}

fn verify_optimizer(
    config: &ModelConfig,
    input: &Tensor,
    target: &Tensor,
    price_scaling: &Tensor,
    geometry_context: &Tensor,
    auxiliary: Option<&Tensor>,
) -> Result<f64> {
    let mut config = config.clone();
    config.dropout = 0.;
    let reference_store = nn::VarStore::new(input.device());
    let reference_model = SegmentModel::new(&reference_store.root(), &config);
    let mut fused_store = nn::VarStore::new(input.device());
    let _fused_model = SegmentModel::new(&fused_store.root(), &config);
    fused_store.copy(&reference_store)?;
    let mut reference = Engine::new(&reference_store, 0.0001, false, OptimizerKind::Adam, 0)?;
    let mut fused = Engine::new(&fused_store, 0.0001, true, OptimizerKind::Adam, 0)?;
    let source = reference_store.variables();
    let destination = fused_store.variables();
    for _ in 0..3 {
        reference.zero_grad()?;
        let loss = Engine::forward_loss(
            &reference_model,
            input,
            target,
            price_scaling,
            geometry_context,
            auxiliary,
            None,
        );
        loss.backward();
        Python::attach(|py| -> Result<()> {
            py.import("torch")?;
            for (name, tensor) in &source {
                let gradient = tensor.grad();
                ensure!(
                    gradient.defined(),
                    "missing optimizer validation gradient: {name}"
                );
                crate::torch::fa4::tensor_object(py, &destination[name])?
                    .setattr("grad", crate::torch::fa4::tensor_object(py, &gradient)?)?;
            }
            Ok(())
        })?;
        reference.optimizer_step()?;
        fused.optimizer_step()?;
    }
    let difference = Tensor::stack(
        &source
            .iter()
            .map(|(name, tensor)| (tensor - &destination[name]).abs().max())
            .collect::<Vec<_>>(),
        0,
    )
    .max()
    .double_value(&[]);
    ensure!(
        difference.is_finite() && difference <= 1e-6,
        "fused Adam differs from native Adam: {difference}"
    );
    Ok(difference)
}

pub fn run(args: BenchmarkArgs) -> Result<()> {
    args.model.validate()?;
    ensure!(
        args.batch_size > 0 && args.steps >= 10 && args.warmup >= 3,
        "benchmark needs batch > 0, steps >= 10, warmup >= 3"
    );
    ensure!(
        args.optimizer != OptimizerKind::PolarExpress || args.warmup >= 4,
        "Polar Express benchmark needs at least four warmups to complete optimizer graph capture"
    );
    ensure!(!args.output.exists(), "benchmark output already exists");
    let device = crate::torch::single_ticker_timexer::runner::cuda_device()?;
    let _backward = crate::torch::cuda::cfg::disable_autograd_multithreading();
    tch::manual_seed(20260905);
    Cuda::manual_seed_all(20260905);
    cuda_memory(true)?;
    let length = args.model.seq_len + args.model.pred_len;
    let rows = length + args.batch_size as i64 * 8;
    // Synthetic resident history isolates kernel throughput; it is never accuracy evidence.
    let close =
        Tensor::randn([rows, 1], (Kind::Float, device)).cumsum(0, Kind::Float) * 0.002 + 100.0;
    let open = &close + Tensor::randn([rows, 1], (Kind::Float, device)) * 0.01;
    let high = close.maximum(&open) + Tensor::rand([rows, 1], (Kind::Float, device)) * 0.02;
    let low = close.minimum(&open) - Tensor::rand([rows, 1], (Kind::Float, device)) * 0.02;
    let raw_history = Tensor::cat(&[open, high, low, close], 1);
    let means = Tensor::from_slice(&[99.9f32, 100.1, 99.8, 100.0])
        .to_device(device)
        .reshape([1, 4]);
    let stds = Tensor::from_slice(&[2.0f32, 2.1, 1.9, 2.05])
        .to_device(device)
        .reshape([1, 4]);
    let history = (&raw_history - &means) / &stds;
    let price_scaling =
        Tensor::cat(&[means, stds], 0)
            .unsqueeze(0)
            .repeat([args.batch_size as i64, 1, 1]);
    let auxiliary_history = args.model.volume_features.then(|| {
        Tensor::cat(
            &[
                Tensor::randn([rows, 1], (Kind::Float, device)),
                Tensor::ones([rows, 1], (Kind::Float, device)),
            ],
            1,
        )
    });
    let offsets = Tensor::arange(length, (Kind::Int64, device)).unsqueeze(0);
    let origin_indices =
        Tensor::arange(args.batch_size as i64, (Kind::Int64, device)).unsqueeze(1) * 8;
    let indices = (&origin_indices + offsets).flatten(0, -1);
    let raw_context = raw_history
        .index_select(0, &indices)
        .reshape([args.batch_size as i64, length, CHANNELS])
        .narrow(1, 0, args.model.seq_len);
    let raw_close = raw_context.narrow(2, 3, 1);
    let mean_close = raw_close.mean_dim([1i64].as_slice(), false, Kind::Float);
    let close_scale =
        (raw_close.var_dim([1i64].as_slice(), false, false) + 1e-5 * 2.05f64.powi(2)).sqrt();
    let relative_range = ((raw_context.narrow(2, 1, 1) - raw_context.narrow(2, 2, 1))
        / raw_context.narrow(2, 2, 1))
    .mean_dim([1i64].as_slice(), false, Kind::Float);
    let geometry_context = Tensor::cat(&[mean_close, close_scale, relative_range], 1);
    drop(raw_context);
    drop(raw_history);
    let gather = || {
        let window =
            history
                .index_select(0, &indices)
                .reshape([args.batch_size as i64, length, CHANNELS]);
        let auxiliary = auxiliary_history.as_ref().map(|values| {
            values
                .index_select(0, &indices)
                .reshape([args.batch_size as i64, length, 2])
                .narrow(1, 0, args.model.seq_len)
        });
        (
            window.narrow(1, 0, args.model.seq_len),
            window.narrow(1, args.model.seq_len, args.model.pred_len),
            auxiliary,
        )
    };
    let mut equivalence = None;
    if args.verify_optimizer {
        let (input, target, auxiliary) = gather();
        equivalence = Some(verify_optimizer(
            &args.model,
            &input,
            &target,
            &price_scaling,
            &geometry_context,
            auxiliary.as_ref(),
        )?);
    }
    let store = nn::VarStore::new(device);
    let model = SegmentModel::new(&store.root(), &args.model);
    let mut engine = Engine::new(
        &store,
        args.learning_rate
            .unwrap_or(args.optimizer.default_learning_rate()),
        args.fused,
        args.optimizer,
        0,
    )?;
    for _ in 0..args.warmup {
        let (input, target, auxiliary) = gather();
        let _ = engine.step(
            &model,
            &input,
            &target,
            &price_scaling,
            &geometry_context,
            auxiliary.as_ref(),
            None,
        )?;
    }
    Cuda::synchronize(0);
    cuda_memory(true)?;
    let sampler = HardwareSampler::start()?;
    let started = Instant::now();
    let mut losses = Vec::new();
    for _ in 0..args.steps {
        let (input, target, auxiliary) = gather();
        losses.push(engine.step(
            &model,
            &input,
            &target,
            &price_scaling,
            &geometry_context,
            auxiliary.as_ref(),
            None,
        )?);
    }
    Cuda::synchronize(0);
    let step_ms = started.elapsed().as_secs_f64() * 1000. / args.steps as f64;
    let hardware = sampler.finish()?;
    let (peak, reserved) = cuda_memory(false)?;
    ensure!(peak > 0, "CUDA allocator instrumentation returned zero");
    ensure!(
        Tensor::stack(&losses, 0).isfinite().all().int64_value(&[]) != 0,
        "nonfinite benchmark objective"
    );
    let title = format!(
        "TimeXer synthetic kernel benchmark | context {} | batch {} | {}",
        args.model.seq_len,
        args.batch_size,
        match args.optimizer {
            OptimizerKind::PolarExpress => "Polar Express 5 + AdamW",
            OptimizerKind::Adam if args.fused => "fused Adam",
            OptimizerKind::Adam => "native Adam",
        }
    );
    let mut summary = vec![
        ("end-to-end step milliseconds", step_ms),
        (
            "origins per second",
            args.batch_size as f64 * 1000. / step_ms,
        ),
        ("peak allocator MiB", peak as f64 / 1048576.),
        ("peak reserved MiB", reserved as f64 / 1048576.),
        (
            "parameter count",
            store
                .trainable_variables()
                .iter()
                .map(Tensor::numel)
                .sum::<usize>() as f64,
        ),
    ];
    if let Some(captured) = engine.optimizer_graph_captured() {
        summary.push((
            "Polar Express optimizer graph captured",
            if captured { 1. } else { 0. },
        ));
    }
    if let Some(difference) = equivalence {
        summary.push(("fused Adam maximum absolute update difference", difference));
    }
    chart(
        &args.output,
        "timexer_segment_benchmark",
        &title,
        "named units",
        vec![args.batch_size as u64],
        summary
            .into_iter()
            .map(|(label, value)| ReportSeries {
                label: label.to_owned(),
                values: vec![value as f32],
            })
            .collect(),
    )?;
    write_hardware(&args.output, &title, &hardware)?;
    if args.profile {
        let mut phase_values: Vec<[f64; 5]> = Vec::new();
        for _ in 0..3 {
            let ((input, target, auxiliary), gather_ms) = timed(|| Ok(gather()))?;
            let (_, zero_ms) = timed(|| engine.zero_grad())?;
            let (loss, forward_ms) = timed(|| {
                Ok(Engine::forward_loss(
                    &model,
                    &input,
                    &target,
                    &price_scaling,
                    &geometry_context,
                    auxiliary.as_ref(),
                    None,
                ))
            })?;
            let (_, backward_ms) = timed(|| {
                loss.backward();
                Ok(())
            })?;
            let (_, adam_ms) = timed(|| engine.optimizer_step())?;
            phase_values.push([gather_ms, zero_ms, forward_ms, backward_ms, adam_ms]);
        }
        chart(
            &args.output,
            "timexer_segment_benchmark_phases",
            &format!("{title} | synchronized phase attribution, not throughput"),
            "milliseconds",
            (1..=3).collect(),
            [
                "input gather",
                "zero gradients",
                "forward and loss",
                "backward",
                "optimizer update",
            ]
            .into_iter()
            .enumerate()
            .map(|(column, label)| ReportSeries {
                label: label.to_owned(),
                values: phase_values
                    .iter()
                    .map(|values| values[column] as f32)
                    .collect(),
            })
            .collect(),
        )?;
    }
    println!("TimeXer performance reports: {}", args.output.display());
    Ok(())
}
