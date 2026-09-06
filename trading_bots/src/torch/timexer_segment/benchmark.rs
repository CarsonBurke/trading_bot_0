use std::{
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use anyhow::{ensure, Context, Result};
use clap::Args;
use nvml_wrapper::Nvml;
use pyo3::{prelude::*, types::PyDict};
use shared::report::{write_report, Report, ReportKind, ReportSeries, ScaleKind};
use tch::{nn, Cuda, Device, Kind, Tensor};

use super::{
    compute::{Engine, OptimizerKind},
    corpus::Batch,
    model::{CausalPatchModel, ModelConfig, CHANNELS},
};

/// Steps the capture audit compares between the eager and the captured engine. Twenty
/// consecutive steps, because the comparison is over the FINAL parameters: the optimizer
/// integrates every step's gradient into them, so one differing gradient element anywhere in
/// the window shows up, and twenty steps of NorMuon is long enough for it to grow.
const CAPTURE_AUDIT_STEPS: usize = 20;

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
    /// Measure the forward+backward CUDA-graph capture: memory budget, step-time delta, and
    /// bit-equality of the objective and of the parameters against the eager step. Runs two
    /// engines in sequence before the main benchmark, so it doubles the run's wall clock.
    #[arg(long)]
    pub capture_audit: bool,
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
    let value = operation()?;
    Cuda::synchronize(0);
    Ok((value, started.elapsed().as_secs_f64() * 1000.))
}

/// Bytes the caching allocator currently holds LIVE, from the same torch that runs the kernels.
/// [`cuda_memory`] reports peaks, which cannot answer "what did autograd retain from this one
/// call"; the live figure, differenced across a call whose output is still held, can - and that
/// is the only honest way to find an activation silently saved in fp32.
fn cuda_allocated() -> Result<f64> {
    Python::attach(|py| -> Result<f64> {
        let cuda = py.import("torch")?.getattr("cuda")?;
        cuda.call_method0("init")?;
        Ok(cuda.call_method1("memory_allocated", (0,))?.extract::<u64>()? as f64)
    })
}

/// Device milliseconds per execution of `operation`, measured with CUDA events.
///
/// Not [`timed`]: the classes below run 0.05-4 ms each, the same order as the host round-trip a
/// `synchronize` costs, so an `Instant` would be measuring the harness. `torch-sys` binds no
/// `cudaEvent_t`, so the events come from the same libtorch through its Python API.
///
/// The events are recorded on `default_stream`, EXPLICITLY, not on `current_stream`: this runs
/// on a tokio worker thread, PyTorch's current stream is thread-local, and the optimizer's
/// graph capture installs a side stream of its own, so "whatever stream this thread holds" is
/// not a safe way to name the stream the kernels are on. `tch` launches on the default stream
/// of the current device.
///
/// `elapsed_time` measures from the receiver to its argument, so it is the START event that is
/// asked; the reverse returns a negative interval.
///
/// Two untimed rounds first, so cuBLAS heuristics, workspace allocation and the allocator's
/// first touch are not in the window.
fn event_timed<T>(rounds: usize, mut operation: impl FnMut() -> T) -> Result<f64> {
    ensure!(rounds > 0, "event timing needs at least one round");
    // The GIL is held ONLY for the four event calls. `operation` must run without it: the
    // autograd engine refuses to be entered by a thread holding the GIL, so timing a backward
    // inside a `Python::attach` block aborts.
    let (start, stop, stream) = Python::attach(|py| -> Result<_> {
        let cuda = py.import("torch")?.getattr("cuda")?;
        let timing = PyDict::new(py);
        timing.set_item("enable_timing", true)?;
        Ok((
            cuda.call_method("Event", (), Some(&timing))?.unbind(),
            cuda.call_method("Event", (), Some(&timing))?.unbind(),
            cuda.call_method1("default_stream", (0,))?.unbind(),
        ))
    })?;
    let record = |event: &Py<PyAny>| -> Result<()> {
        Python::attach(|py| -> Result<()> {
            event
                .bind(py)
                .call_method1("record", (stream.bind(py),))?;
            Ok(())
        })
    };
    for _ in 0..2 {
        drop(operation());
    }
    Cuda::synchronize(0);
    record(&start)?;
    for _ in 0..rounds {
        drop(operation());
    }
    record(&stop)?;
    Cuda::synchronize(0);
    let elapsed = Python::attach(|py| -> Result<f64> {
        Ok(start
            .bind(py)
            .call_method1("elapsed_time", (stop.bind(py),))?
            .extract()?)
    })?;
    ensure!(
        elapsed > 0.,
        "CUDA events measured {elapsed} ms over {rounds} rounds, which is not a device interval"
    );
    Ok(elapsed / rounds as f64)
}

/// One measured backbone kernel class.
struct KernelRow {
    name: &'static str,
    forward_ms: f64,
    backward_ms: f64,
    /// The FORWARD pass alone: bytes every input must be read and every output written once,
    /// and the arithmetic that pass must do. Both are exactly enumerable from the shapes, so
    /// `forward_bytes / forward_ms` is a measurement, not a convention.
    ///
    /// Backward is deliberately NOT charged bytes. Its traffic is per-op ATen internals, not a
    /// multiple of the forward: `add`'s backward moves nothing (autograd hands the same gradient
    /// to both inputs), a `reshape`'s backward is one more view, and a GEMM's is two more GEMMs.
    /// `step_cost`'s uniform "twice more" rule is right for the step as a whole and wrong per
    /// class - it reported the residual add at 3540 GB/s, above this card's spec bandwidth.
    forward_bytes: f64,
    forward_flops: f64,
    /// Live allocator bytes the class's forward left behind with its output still held: the
    /// output plus everything autograd saved for backward.
    retained_bytes: f64,
    /// The output tensor alone. `retained - output` is what backward saved, and a LayerNorm or a
    /// cast that quietly keeps an fp32 copy of a `[tokens, d_model]` activation shows up here as
    /// 196 MB that the bf16 accounting does not explain.
    output_bytes: f64,
}

/// Per-kernel-class device time, retained activations and roofline fractions for one backbone
/// layer at the real configuration.
///
/// Each class allocates its inputs OUTSIDE the timed window, records the live-allocator delta of
/// one forward, then times `rounds` forwards and `rounds` forward+backwards; the difference is
/// the backward. `Tensor::run_backward` is given the class's leaf parameters as well as its
/// activation input, so the weight-gradient GEMM is not pruned out of the measurement.
fn kernel_profile(
    model: &CausalPatchModel,
    batch: &Batch,
    rounds: usize,
) -> Result<Vec<KernelRow>> {
    let device = batch.log_prices.device();
    let stats = model.statistics(batch);
    let mut rows = Vec::new();
    for class in model.kernel_classes(batch, &stats, true) {
        let inputs: Vec<Tensor> = class
            .inputs
            .iter()
            .map(|(shape, kind)| {
                Tensor::randn(shape.as_slice(), (Kind::Float, device))
                    .to_kind(*kind)
                    .set_requires_grad(true)
            })
            .collect();
        Cuda::synchronize(0);
        let before = cuda_allocated()?;
        let output = (class.run)(&inputs);
        Cuda::synchronize(0);
        let retained_bytes = cuda_allocated()? - before;
        let output_bytes =
            output.numel() as f64 * output.kind().elt_size_in_bytes() as f64;
        drop(output);
        let forward_ms = event_timed(rounds, || (class.run)(&inputs))?;
        let differentiable = !inputs.is_empty();
        let backward_ms = if differentiable {
            let targets: Vec<&Tensor> = std::iter::once(&inputs[0])
                .chain(class.parameters.iter())
                .collect();
            let both = event_timed(rounds, || {
                let output = (class.run)(&inputs);
                Tensor::run_backward(&[&output], &targets, false, false);
            })?;
            (both - forward_ms).max(0.)
        } else {
            0.
        };
        rows.push(KernelRow {
            name: class.name,
            forward_ms,
            backward_ms,
            forward_bytes: class.forward_bytes,
            forward_flops: class.forward_flops,
            retained_bytes,
            output_bytes,
        });
    }
    Ok(rows)
}

/// Device ceilings, MEASURED rather than looked up: a large square bf16 GEMM for arithmetic and
/// a large device-to-device copy for bandwidth. A table of nominal peaks would be one more thing
/// to keep true across cards; these two kernels are what this card actually sustains, so the
/// achieved fractions the profile reports are honest denominators.
///
/// BEST of `rounds`, not the mean, and timed with CUDA events rather than the host clock. A peak
/// is a maximum: one contended round must not halve the denominator every other number in the
/// profile is divided by. The mean-of-ten host-timed form this replaces reported 630 GB/s and
/// 94 TFLOPS on a card whose own residual add measured 1569 GB/s and whose own QKV projection
/// measured 104 TFLOPS in the same process - denominators smaller than their numerators.
///
/// Takes the device rather than resolving one: `cuda_device` is entry-point work
/// (`single_ticker_timexer/runner.rs:127` runs `configure_threads`, and
/// `tch::set_num_interop_threads` ABORTS once torch has done parallel work), and this runs
/// after the whole timing window.
fn device_peaks(device: Device) -> Result<(f64, f64)> {
    let side = 8192i64;
    let left = Tensor::randn([side, side], (Kind::BFloat16, device));
    let right = Tensor::randn([side, side], (Kind::BFloat16, device));
    let source = Tensor::zeros([1 << 28], (Kind::Float, device));
    let mut sink = Tensor::zeros([1 << 28], (Kind::Float, device));
    let rounds = 8;
    let mut matmul_ms = f64::MAX;
    let mut copy_ms = f64::MAX;
    for _ in 0..rounds {
        matmul_ms = matmul_ms.min(event_timed(1, || left.matmul(&right))?);
        copy_ms = copy_ms.min(event_timed(1, || sink.copy_(&source))?);
    }
    let flops = 2. * (side as f64).powi(3);
    let bytes = 2. * 4. * (1u64 << 28) as f64;
    ensure!(
        matmul_ms > 0. && copy_ms > 0. && matmul_ms < f64::MAX && copy_ms < f64::MAX,
        "device peak probes did not register"
    );
    Ok((
        flops / (matmul_ms / 1000.) / 1e12,
        bytes / (copy_ms / 1000.) / 1e9,
    ))
}

fn verify_optimizer(config: &ModelConfig, batch: &Batch) -> Result<f64> {
    let mut config = config.clone();
    config.dropout = 0.;
    let device = batch.log_prices.device();
    let reference_store = nn::VarStore::new(device);
    let reference_model = CausalPatchModel::new(&reference_store.root(), &config);
    let mut fused_store = nn::VarStore::new(device);
    let _fused_model = CausalPatchModel::new(&fused_store.root(), &config);
    fused_store.copy(&reference_store)?;
    let mut reference = Engine::new(&reference_store, 0.0001, false, OptimizerKind::Adam, 0)?;
    let mut fused = Engine::new(&fused_store, 0.0001, true, OptimizerKind::Adam, 0)?;
    let source = reference_store.variables();
    let destination = fused_store.variables();
    for _ in 0..3 {
        reference.zero_grad()?;
        Engine::forward_loss(&reference_model, batch, true)
            .nll
            .backward();
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
    let rows = args.batch_size as i64;
    let stride = 8;
    let history_len = length + rows * stride;
    // Synthetic resident log-price history isolates kernel throughput; it is never accuracy evidence.
    let close = Tensor::randn([history_len, 1], (Kind::Float, device)).cumsum(0, Kind::Float)
        * 0.002
        + 100.0f64.ln();
    let open = &close + Tensor::randn([history_len, 1], (Kind::Float, device)) * 0.0001;
    let high = close.maximum(&open) + Tensor::rand([history_len, 1], (Kind::Float, device)) * 0.0002;
    let low = close.minimum(&open) - Tensor::rand([history_len, 1], (Kind::Float, device)) * 0.0002;
    let log_history = Tensor::cat(&[open, high, low, close], 1);
    let aux_channels = args.model.features.channels() as i64;
    let auxiliary_history = Tensor::cat(
        &[
            Tensor::randn([history_len, 1], (Kind::Float, device)),
            Tensor::ones([history_len, 1], (Kind::Float, device)),
        ],
        1,
    )
    .repeat([1, aux_channels / 2]);
    let offsets = Tensor::arange(length, (Kind::Int64, device)).unsqueeze(0);
    let origin_indices = Tensor::arange(rows, (Kind::Int64, device)).unsqueeze(1) * stride;
    let indices = (&origin_indices + offsets).flatten(0, -1);
    let valid = Tensor::ones([rows, length], (Kind::Float, device));
    let gather = || {
        let window = log_history
            .index_select(0, &indices)
            .reshape([rows, length, CHANNELS]);
        let anchor_log = window.narrow(1, args.model.seq_len - 1, 1).narrow(2, 3, 1);
        let packed = Tensor::cat(
            &[
                (&window - &anchor_log).flatten(1, 2),
                valid.shallow_clone(),
                auxiliary_history
                    .index_select(0, &indices)
                    .reshape([rows, length * aux_channels]),
                Tensor::zeros([rows, length], (Kind::Float, device)),
                anchor_log.reshape([rows, 1]).exp(),
            ],
            1,
        );
        Batch::from_packed(
            packed,
            args.model.seq_len as usize,
            args.model.pred_len as usize,
            aux_channels as usize,
            (rows * args.model.pred_len) as usize,
        )
    };
    let mut equivalence = None;
    if args.verify_optimizer {
        equivalence = Some(verify_optimizer(&args.model, &gather())?);
    }
    // Before the main store, model and engine exist: the audit builds two engines of its
    // own in sequence, and the captured one reserves a private mempool about the size of
    // the eager step's whole working set, so nothing else may be resident.
    let capture = if args.capture_audit {
        Some(super::compute::audit_step_capture(
            &args.model,
            &gather(),
            args.learning_rate
                .unwrap_or(args.optimizer.default_learning_rate()),
            args.fused,
            args.optimizer,
            CAPTURE_AUDIT_STEPS,
        )?)
    } else {
        None
    };
    let store = nn::VarStore::new(device);
    let model = CausalPatchModel::new(&store.root(), &args.model);
    let mut engine = Engine::new(
        &store,
        args.learning_rate
            .unwrap_or(args.optimizer.default_learning_rate()),
        args.fused,
        args.optimizer,
        0,
    )?;
    for _ in 0..args.warmup {
        let _ = engine.step(&model, &gather())?;
    }
    Cuda::synchronize(0);
    cuda_memory(true)?;
    let sampler = HardwareSampler::start()?;
    let started = Instant::now();
    let mut losses = Vec::new();
    for _ in 0..args.steps {
        losses.push(engine.step(&model, &gather())?.nll);
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
        "CausalPatch synthetic kernel benchmark | context {} | batch {} | {}",
        args.model.seq_len,
        args.batch_size,
        match args.optimizer {
            OptimizerKind::PolarExpress => "Polar Express 5 + AdamW",
            OptimizerKind::Adam if args.fused => "fused Adam",
            OptimizerKind::Adam => "native Adam",
        }
    );
    let cost = args.model.step_cost(args.batch_size);
    let achieved_tflops = cost.matmul_flops / (step_ms / 1000.) / 1e12;
    let achieved_gbs = cost.traffic_bytes / (step_ms / 1000.) / 1e9;
    let (peak_tflops, peak_gbs) = device_peaks(device)?;
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
        ("step matmul TFLOP", cost.matmul_flops / 1e12),
        ("achieved matmul TFLOPS", achieved_tflops),
        ("measured device bf16 GEMM TFLOPS", peak_tflops),
        ("achieved fraction of GEMM peak", achieved_tflops / peak_tflops),
        ("step activation traffic GB (analytic lower bound)", cost.traffic_bytes / 1e9),
        ("achieved HBM GB/s (analytic lower bound)", achieved_gbs),
        ("measured device copy GB/s", peak_gbs),
        ("achieved fraction of copy peak", achieved_gbs / peak_gbs),
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
    if let Some(audit) = capture {
        // Named units, so each row states its own: MiB for the byte budget, milliseconds for
        // the two step times, dimensionless relative difference for the two equalities.
        summary.extend([
            ("captured step private mempool MiB", audit.budget.pool_reservation_mib),
            ("device total MiB", audit.budget.device_total_mib),
            ("allocator reserved before warmup MiB", audit.budget.before_warmup.reserved_mib),
            ("allocator reserved after warmup MiB", audit.budget.after_warmup.reserved_mib),
            ("allocator reserved after empty_cache MiB", audit.budget.after_empty_cache.reserved_mib),
            ("allocator reserved at capture end MiB", audit.budget.at_capture_end.reserved_mib),
            ("allocator live at capture start MiB", audit.budget.at_capture_start.allocated_mib),
            ("eager step peak allocator MiB", audit.eager_peak_allocated_mib),
            ("eager step milliseconds", audit.eager_step_ms),
            ("captured replay step milliseconds", audit.replay_step_ms),
            (
                "captured replay step time as a fraction of eager",
                audit.replay_step_ms / audit.eager_step_ms,
            ),
            ("capture vs eager objective maximum relative difference", audit.loss_max_relative),
            ("capture vs eager parameter maximum relative difference", audit.parameter_max_relative),
            ("capture audit compared steps", audit.compared_steps as f64),
            (
                "training step with one host objective read per step, milliseconds",
                audit.host_read_step_ms,
            ),
            (
                "training step uploaded from pinned host memory, milliseconds",
                audit.pinned_upload_step_ms,
            ),
            (
                "training step uploaded from pageable host memory, milliseconds",
                audit.pageable_upload_step_ms,
            ),
        ]);
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
        let mut phase_values: Vec<[f64; 7]> = Vec::new();
        for _ in 0..3 {
            let (batch, gather_ms) = timed(|| Ok(gather()))?;
            let (_, phases) = engine.timed_step(&model, &batch)?;
            phase_values.push([
                gather_ms,
                phases[0],
                phases[1],
                phases[2],
                phases[3],
                phases[4],
                phases[5],
            ]);
        }
        chart(
            &args.output,
            "timexer_segment_benchmark_phases",
            &format!(
                "{title} | synchronized phase attribution, not throughput | {:.1} achieved TFLOPS of {peak_tflops:.0} measured GEMM peak, {achieved_gbs:.0} GB/s of {peak_gbs:.0} measured copy peak",
                achieved_tflops
            ),
            "milliseconds",
            (1..=3).collect(),
            [
                "input gather (synthetic, stands in for host batch)",
                "batch upload into the resident device batch",
                "forward backbone",
                "forward head and loss",
                "backward",
                "captured forward+backward replay (NaN when not captured)",
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
        // Per-kernel-class attribution. The phase chart above says how much of the step is
        // backbone; this says WHICH kernels inside it, and against which roof each one is
        // stuck. The classes are per-layer, so the step-level column scales by `layers`.
        let kernels = kernel_profile(&model, &gather(), 8)?;
        let layers = args.model.layers as f64;
        let scaled = |row: &KernelRow| {
            let name: &str = row.name;
            let repeats = match name {
                // Two pre-norms per layer and three residual `addcmul`s (the residual scale
                // on both sub-blocks plus the x0 injection), one U-net fold on half the
                // layers, and one value-residual mix on every layer but the source; the
                // embedding and the two pre-change reference forms run once for the whole step.
                "RMSNorm" => 2. * layers,
                "residual addcmul" => 3. * layers,
                "value residual mix" => layers - 1.,
                "patch embedding tokens" | "reference embedding (cast after cat)" => 1.,
                _ => layers,
            };
            // Rates are FORWARD bytes and FORWARD flops over the FORWARD time: both numerator
            // and denominator are then exactly what one measured pass did.
            let forward_seconds = row.forward_ms / 1000.;
            (
                repeats * row.forward_ms,
                repeats * row.backward_ms,
                row.forward_bytes / forward_seconds / 1e9,
                row.forward_flops / forward_seconds / 1e12,
            )
        };
        println!(
            "{:<28} {:>9} {:>9} {:>8} {:>9} {:>7} {:>9} {:>7} {:>9} {:>9}",
            "kernel class",
            "fwd ms",
            "bwd ms",
            "step ms",
            "fwd GB",
            "GB/s",
            "% copy",
            "TFLOPS",
            "% GEMM",
            "saved MiB"
        );
        for row in &kernels {
            let (forward, backward, gbs, tflops) = scaled(row);
            println!(
                "{:<28} {:>9.3} {:>9.3} {:>8.1} {:>9.2} {:>7.0} {:>9.1} {:>7.1} {:>9.1} {:>9.1}",
                row.name,
                row.forward_ms,
                row.backward_ms,
                forward + backward,
                row.forward_bytes / 1e9,
                gbs,
                100. * gbs / peak_gbs,
                tflops,
                100. * tflops / peak_tflops,
                (row.retained_bytes - row.output_bytes) / 1048576.
            );
        }
        // Sum of the parts against the whole. The composed layer, the once-per-step embedding
        // and the two pre-change reference forms are not parts of a layer, and neither is the
        // value-residual mix - the composed layer is layer 0, which is the mix's SOURCE and
        // pays no mix. The pre-norm runs twice per layer and the residual `addcmul` three
        // times.
        let parts: f64 = kernels
            .iter()
            .filter(|row| {
                !row.name.starts_with("reference ")
                    && row.name != "composed layer"
                    && row.name != "patch embedding tokens"
                    && row.name != "value residual mix"
            })
            .map(|row| {
                let each = row.forward_ms + row.backward_ms;
                match row.name {
                    "RMSNorm" => 2. * each,
                    "residual addcmul" => 3. * each,
                    _ => each,
                }
            })
            .sum();
        let composed = kernels
            .iter()
            .find(|row| row.name == "composed layer")
            .map(|row| row.forward_ms + row.backward_ms)
            .unwrap_or_default();
        println!(
            "attribution: sum of classes {parts:.3} ms/layer vs composed layer {composed:.3} ms/layer ({:+.1}%)",
            100. * (parts - composed) / composed
        );
        let steps = vec![args.batch_size as u64];
        let per_class = |suffix: &str, value: &dyn Fn(&KernelRow) -> f64| {
            kernels
                .iter()
                .map(|row| ReportSeries {
                    label: format!("{}{suffix}", row.name),
                    values: vec![value(row) as f32],
                })
                .collect::<Vec<_>>()
        };
        chart(
            &args.output,
            "timexer_segment_benchmark_kernels",
            &format!("{title} | one backbone layer, CUDA-event device time per kernel class, forward and backward separated"),
            "milliseconds",
            steps.clone(),
            per_class(": forward", &|row| row.forward_ms)
                .into_iter()
                .chain(per_class(": backward", &|row| row.backward_ms))
                .collect(),
        )?;
        chart(
            &args.output,
            "timexer_segment_benchmark_kernel_roofline",
            &format!("{title} | per kernel class, against {peak_gbs:.0} GB/s measured copy peak and {peak_tflops:.0} TFLOPS measured bf16 GEMM peak"),
            "percent of the measured device peak",
            steps.clone(),
            per_class(": HBM bandwidth", &|row| 100. * scaled(row).2 / peak_gbs)
                .into_iter()
                .chain(per_class(": arithmetic", &|row| {
                    100. * scaled(row).3 / peak_tflops
                }))
                .collect(),
        )?;
        chart(
            &args.output,
            "timexer_segment_benchmark_kernel_activations",
            &format!("{title} | live allocator delta across one forward per class: a class saving more than its bf16 output is holding something extra"),
            "mebibytes",
            steps,
            per_class(": saved for backward", &|row| {
                (row.retained_bytes - row.output_bytes) / 1048576.
            })
            .into_iter()
            .chain(per_class(": class output", &|row| {
                row.output_bytes / 1048576.
            }))
            .collect(),
        )?;
    }
    println!("CausalPatch performance reports: {}", args.output.display());
    Ok(())
}
