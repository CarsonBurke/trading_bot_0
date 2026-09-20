use std::{
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

use anyhow::{ensure, Context, Result};
use clap::Args;
use nvml_wrapper::Nvml;
use pyo3::{prelude::*, types::PyDict};
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use shared::report::{write_report, Report, ReportKind, ReportSeries, ScaleKind};
use tch::{nn, Cuda, Device, Kind, Tensor};

use super::{
    compute::{Engine, OptimizerKind, RecipeKnobs, CAPTURE_AFTER_STEPS},
    corpus::{Batch, Corpus},
    features::FeatureSet,
    model::{CausalPatchModel, KernelClass, ModelConfig, CHANNELS},
    runner::Prefetcher,
};

/// Steps the capture audit compares between the eager and the captured engine. Twenty
/// consecutive steps, because the comparison is over the FINAL parameters: the optimizer
/// integrates every step's gradient into them, so one differing gradient element anywhere in
/// the window shows up, and twenty steps of NorMuon is long enough for it to grow.
const CAPTURE_AUDIT_STEPS: usize = 20;

/// The measured bf16 GEMM peak of an UNCONTENDED card, job 5399: 230.99 TFLOPS. It is a
/// measurement of this machine, not a spec number, and it exists so a run can say how far
/// from quiet it was. Any cross-run step-time claim needs both runs near this figure; job
/// 5452 sat at 193.32, 16.3% down, which is exactly why the paired arm exists.
const QUIET_CARD_GEMM_TFLOPS: f64 = 230.99;

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
    #[arg(long, default_value_t = 8)]
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
    /// Measure the fused loss chain against the composed-ATen chain PAIRED: both arms in one
    /// process, alternated A-B-A-B, differencing inside the run. A cross-run step-time
    /// comparison is not defensible on a shared card - job 5452's measured bf16 GEMM peak was
    /// 16.3% below job 5399's, and one scalar cannot normalize a step that is part
    /// arithmetic-bound, part bandwidth-bound and part launch-bound. Pairing puts the
    /// contention in BOTH arms so it cancels in the difference, and the A-A spread across
    /// repeats is the comparison's own error bound rather than an assumption.
    #[arg(long)]
    pub paired_loss: bool,
    /// Alternations of the paired arm. Each one times both chains once, so the A-A spread is
    /// measured over this many fused arms.
    #[arg(long, default_value_t = 5)]
    pub paired_repeats: usize,
    /// Corpus directory for the third timed arm: the REAL host loader, prefetched exactly as
    /// `runner::train` prefetches it. Without it that arm reports NaN and the loader's
    /// contribution to the step is not measured at all.
    #[arg(long)]
    pub corpus: Option<PathBuf>,
    /// Shared eligible-origin history for the corpus arm; must cover `--seq-len`.
    #[arg(long, default_value_t = 6000)]
    pub common_context: usize,
    /// Minimum cross-section a shared grid slot needs to define a market step, for the corpus arm.
    #[arg(long, default_value_t = 2000)]
    pub market_min_cross_section: usize,
}

#[derive(Clone, Debug, Args)]
pub struct LoaderAuditArgs {
    #[arg(long, default_value_os_t = crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub ticker: Vec<String>,
    #[arg(long, default_value_t = 6000)]
    pub seq_len: usize,
    #[arg(long, default_value_t = 192)]
    pub pred_len: usize,
    #[arg(long, default_value_t = 6000)]
    pub common_context: usize,
    /// Comma-separated exogenous variates: time-of-day, day-of-week, session-gap, volume,
    /// market, spy, dispersion, cross-section-z, cross-section-rank, relative-volume, range-z;
    /// `all` or `none`.
    #[arg(long, default_value_t = FeatureSet::ALL)]
    pub features: FeatureSet,
    #[arg(long, default_value_t = 2000)]
    pub market_min_cross_section: usize,
    /// Row counts to audit, so one corpus load prices the whole batch-size scaling.
    #[arg(long, value_delimiter = ',', default_value = "64,128,256")]
    pub batch_size: Vec<usize>,
    #[arg(long, default_value_t = 5)]
    pub rounds: usize,
}

/// Component attribution of host batch assembly, on the real corpus, CPU only.
///
/// No device, deliberately. `Corpus::load` and `Corpus::host_batch` need none, and the whole
/// question is what the host work costs in host time; running it under a GPU lease would put
/// the measurement behind the queue it exists to explain. The pinned allocation the training
/// path uses is therefore NOT exercised here - a pageable `Tensor::empty` of the same size
/// stands in - and the audit says so in its own output.
pub fn loader_audit(args: LoaderAuditArgs) -> Result<()> {
    ensure!(
        args.rounds > 0 && !args.batch_size.is_empty(),
        "the loader audit needs at least one round and one batch size"
    );
    let corpus = Corpus::load(
        &args.data_dir,
        &args.ticker,
        args.seq_len,
        args.pred_len,
        args.common_context,
        &args.features,
        args.market_min_cross_section,
        0,
    )?;
    let mut rng = ChaCha8Rng::seed_from_u64(20260907);
    let mut refs = corpus.train_refs.clone();
    refs.shuffle(&mut rng);
    println!(
        "CausalPatch host loader audit (CPU only, pageable stand-in for the pinned block): {} tickers, {} training rows, context {}, horizon {}, {} auxiliary channels, {} rounds per rung",
        corpus.contract.tickers.len(),
        refs.len(),
        args.seq_len,
        args.pred_len,
        args.features.channels(),
        args.rounds
    );
    for &rows in &args.batch_size {
        ensure!(
            rows > 0 && rows <= refs.len(),
            "audited batch size {rows} exceeds the {} available training rows",
            refs.len()
        );
        let phases = corpus.audit_host_batch(&refs[..rows], args.rounds)?;
        let removed: f64 = phases
            .iter()
            .filter(|phase| phase.name.starts_with("removed:"))
            .map(|phase| phase.ms)
            .sum();
        let after = phases
            .iter()
            .find(|phase| phase.name.starts_with("production"))
            .map(|phase| phase.ms)
            .context("the audit ladder produced no production total")?;
        println!("\nbatch {rows}, milliseconds per batch:");
        for phase in &phases {
            println!("  {:<78} {:>9.3}", phase.name, phase.ms);
        }
        println!(
            "  {:<78} {:>9.3}\n  {:<78} {:>9.3}\n  {:<78} {:>9.2}x",
            "before: production total plus both removed rungs",
            after + removed,
            "after: production total",
            after,
            "speedup of host batch assembly",
            (after + removed) / after
        );
    }
    Ok(())
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

/// Per-phase milliseconds of one SAMPLED batch of a non-training accumulation pass.
///
/// Its own base rather than rows in `timexer_segment_timing`: see the registry entry in
/// `shared::report`. The four series partition the sampled batch's wall clock exactly - the
/// device is drained before the sample and each span is closed by a synchronization - so they
/// may be read as a stacked budget, and the loader-wait row is the only one prefetch depth can
/// hide. The drained run-ahead is deliberately NOT a fifth series: it is a different question
/// in the same unit, and it rides in the title instead.
pub fn write_eval_phases(output: &Path, title: &str, trace: &[[f64; 6]]) -> Result<()> {
    let labels = [
        "host loader wait (batch assembly the prefetch failed to hide)",
        "H2D upload of the packed rows",
        "forward: statistics, backbone, last-origin head",
        "accumulate: nine fp64 per-timestamp scatter sums",
    ];
    write_report(
        output.join("timexer_segment_eval_phases.report.bin"),
        &Report {
            title: title.to_owned(),
            x_label: Some("batch ordinal within the pass, every stride-th batch sampled".to_owned()),
            y_label: Some(
                "milliseconds of one sampled batch; the four sum to that batch's wall clock, lower is faster".to_owned(),
            ),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: trace.iter().map(|row| row[0] as u64).collect(),
                series: labels
                    .iter()
                    .enumerate()
                    .map(|(column, label)| ReportSeries {
                        label: (*label).to_owned(),
                        values: trace.iter().map(|row| row[column + 1] as f32).collect(),
                    })
                    .collect(),
            },
        },
    )?;
    Ok(())
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
    let classes = model.kernel_classes(batch, &stats, true);
    let measure = |class: &KernelClass<'_>, name: &'static str| -> Result<KernelRow> {
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
        let output_bytes = output.numel() as f64 * output.kind().elt_size_in_bytes() as f64;
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
        Ok(KernelRow {
            name,
            forward_ms,
            backward_ms,
            forward_bytes: class.forward_bytes,
            forward_flops: class.forward_flops,
            retained_bytes,
            output_bytes,
        })
    };
    let mut rows = Vec::new();
    for class in &classes {
        rows.push(measure(class, class.name)?);
    }
    let attention = classes
        .iter()
        .find(|class| class.name == SDPA_CLASS)
        .context("the kernel class list must contain the attention class")?;
    for (row, forced) in SDPA_BACKENDS.iter().enumerate() {
        let mut only = [false; SDPA_BACKENDS.len()];
        only[row] = true;
        set_sdpa_backends(&only)?;
        rows.push(match silently(|| measure(attention, forced.row)) {
            Some(measured) => measured?,
            // The backend cannot run this shape at all: `causal SDPA (flash only)` reading NaN
            // is the answer to "is flash even eligible at 375 tokens", not a failure to measure.
            None => KernelRow {
                name: forced.row,
                forward_ms: f64::NAN,
                backward_ms: f64::NAN,
                forward_bytes: attention.forward_bytes,
                forward_flops: attention.forward_flops,
                retained_bytes: 0.,
                output_bytes: 0.,
            },
        });
    }
    set_sdpa_backends(&[true; SDPA_BACKENDS.len()])?;
    Ok(rows)
}

const SDPA_CLASS: &str = "causal SDPA";

/// One scaled-dot-product-attention backend: the `torch.backends.cuda` setter that admits it and
/// the kernel row that reports it alone.
struct SdpaBackend {
    setter: &'static str,
    row: &'static str,
}

/// Every backend the dispatcher can choose between, in no significant order: the point of forcing
/// them one at a time is that the unforced `causal SDPA` row above must equal ONE of them, which
/// identifies what training actually gets by timing rather than by trusting a precedence table
/// that changes between torch releases. The alternatives are the actionable part - a faster one
/// at `origins` = 375 costs a global flag to adopt.
const SDPA_BACKENDS: [SdpaBackend; 4] = [
    SdpaBackend {
        setter: "enable_flash_sdp",
        row: "causal SDPA (flash only)",
    },
    SdpaBackend {
        setter: "enable_mem_efficient_sdp",
        row: "causal SDPA (mem-efficient only)",
    },
    SdpaBackend {
        setter: "enable_cudnn_sdp",
        row: "causal SDPA (cuDNN only)",
    },
    SdpaBackend {
        setter: "enable_math_sdp",
        row: "causal SDPA (math only)",
    },
];

/// Admit exactly the backends flagged `true`. These are process-global, so every caller restores
/// the all-enabled state it found.
fn set_sdpa_backends(admitted: &[bool; SDPA_BACKENDS.len()]) -> Result<()> {
    Python::attach(|py| -> Result<()> {
        let backends = py.import("torch")?.getattr("backends")?.getattr("cuda")?;
        for (backend, on) in SDPA_BACKENDS.iter().zip(admitted) {
            backends.call_method1(backend.setter, (*on,))?;
        }
        Ok(())
    })
}

/// `operation` with libtorch's panic message suppressed: `None` when it raised.
///
/// An ineligible SDPA backend is a normal outcome of asking which backends this shape admits, so
/// its stack trace is noise rather than news. Nothing else in this file catches anything.
fn silently<T>(operation: impl FnOnce() -> T) -> Option<T> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)).ok();
    std::panic::set_hook(previous);
    outcome
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
pub(super) fn device_peaks(device: Device) -> Result<(f64, f64)> {
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
    let knobs = RecipeKnobs::reference(config.x0_lambdas);
    let mut reference = Engine::new(&reference_store, 0.0001, knobs, false, OptimizerKind::Adam)?;
    let mut fused = Engine::new(&fused_store, 0.0001, knobs, true, OptimizerKind::Adam)?;
    let source = reference_store.variables();
    let destination = fused_store.variables();
    for _ in 0..3 {
        reference.zero_grad()?;
        Engine::forward_loss(&reference_model, batch, true, None)
            .objective
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

/// Milliseconds per step with the batch arriving from the REAL corpus loader.
///
/// The loop is the training loop's order - receive, step, request the next - because that
/// order is what decides whether the loader's service time overlaps device execution. The
/// engine, the model and the capture are the caller's, already warmed and already armed, so
/// the only difference from the pinned-host arm is where the bytes came from.
fn loader_arm(
    directory: &Path,
    args: &BenchmarkArgs,
    model: &CausalPatchModel,
    engine: &mut Engine,
    device: Device,
) -> Result<f64> {
    let mut corpus = Corpus::load(
        directory,
        &[],
        args.model.seq_len as usize,
        args.model.pred_len as usize,
        args.common_context,
        &args.model.features,
        args.market_min_cross_section,
        0,
    )?;
    corpus.prepare(device);
    let corpus = Arc::new(corpus);
    let mut refs = corpus.train_refs.clone();
    refs.shuffle(&mut ChaCha8Rng::seed_from_u64(20260907));
    let needed = (args.steps + args.warmup + 1) * args.batch_size;
    ensure!(
        refs.len() >= needed,
        "the corpus arm needs {needed} training rows for {} warmup and {} timed steps at batch {}, and the corpus has {}",
        args.warmup,
        args.steps,
        args.batch_size,
        refs.len()
    );
    let loader = Prefetcher::new(Arc::clone(&corpus));
    let mut served = 0usize;
    let mut serve = |loader: &Prefetcher, served: &mut usize| -> Result<()> {
        let start = *served * args.batch_size;
        *served += 1;
        loader.request(&refs[start..start + args.batch_size])
    };
    serve(&loader, &mut served)?;
    // Warmed on real batches first: the pinned host block is a new size class for libtorch's
    // caching host allocator, and its first `cudaHostAlloc` is not what a steady-state step
    // pays.
    for _ in 0..args.warmup {
        let (host, _) = loader.receive()?;
        let _ = engine.step(model, &host)?;
        serve(&loader, &mut served)?;
    }
    Cuda::synchronize(0);
    let started = Instant::now();
    for _ in 0..args.steps {
        let (host, _) = loader.receive()?;
        let _ = engine.step(model, &host)?;
        serve(&loader, &mut served)?;
    }
    Cuda::synchronize(0);
    Ok(started.elapsed().as_secs_f64() * 1000. / args.steps as f64)
}

/// One paired measurement of the loss chain: both forms alternated inside ONE process.
struct PairedLoss {
    fused_ms: f64,
    composed_ms: f64,
    fused_drift_ms: f64,
    composed_drift_ms: f64,
    repeats: usize,
    per_repeat: Vec<(f64, f64)>,
    /// What scheduling this arm actually needs. Job 5461 died on a 188 MiB allocation under
    /// foreign tenants, and a job whose requirement is unknown cannot be scheduled against a
    /// card whose headroom is unknown - so the arm reports its own peak rather than leaving
    /// both sides of that comparison to guesswork.
    peak_allocated_mib: f64,
    peak_reserved_mib: f64,
}

/// The fused loss chain against the composed-ATen chain it replaced, PAIRED.
///
/// WHY PAIRED. Job 5452 measured the fused class at 4.096 ms against job 5399's 15.192 -
/// inside the pre-registered band - but its whole-step number was useless, because the
/// harness's own `measured device bf16 GEMM TFLOPS` had fallen 16.3% under foreign tenants
/// and every arithmetic-bound class fell with it while the bandwidth-saturated ones did not
/// move at all. So contention taxed SM share and not DRAM, and no single scalar can
/// normalize a step that is part arithmetic-bound, part bandwidth-bound and part
/// launch-bound. Alternating the two chains inside one process makes whatever the machine is
/// doing common to both arms, so it cancels in the DIFFERENCE, and the spread across the
/// repeats of the same arm bounds the comparison's own error instead of being assumed away.
///
/// WHAT IS AND IS NOT IN THE WINDOW. Forward through the trunk and the head, the loss, and
/// the full backward - eager, and eager in BOTH arms, which matters: if one arm could be
/// captured and the other could not, the difference would measure capture rather than
/// fusion. The optimizer is outside the window because it is identical in both arms and
/// three times the size of the difference being measured. The batch, the statistics and the
/// targets are built once and shared, so no arm pays for the harness's own assembly.
///
/// The two arms must produce the SAME loss, and this refuses to report if they do not: the
/// chains are bit-identical by `fused_kernels`' own suite, so an inequality here means the
/// benchmark is comparing two different objectives and its delta means nothing.
fn paired_loss_arm(
    model: &CausalPatchModel,
    store: &nn::VarStore,
    batch: &Batch,
    rounds: usize,
    repeats: usize,
) -> Result<PairedLoss> {
    ensure!(rounds > 0 && repeats > 0, "the paired arm needs a round and a repeat");
    // Warmup's cached blocks go back to the driver first, and the peak counters start here,
    // so the number this reports is THIS arm's requirement and not the whole run's history.
    crate::torch::cuda::empty_cache();
    cuda_memory(true)?;
    let stats = model.statistics(batch);
    let (targets, mask) = model.targets(batch, &stats, false);
    let pass = |composed: bool| {
        let head = model.forward(batch, &stats, true, false);
        let losses = match composed {
            true => model.composed_losses(&head, &stats, &targets, &mask),
            false => model.losses(&head, &stats, &targets, &mask),
        };
        losses.objective.backward();
        losses.nll.double_value(&[])
    };
    let clear = || {
        for mut variable in store.trainable_variables() {
            variable.zero_grad();
        }
    };
    // The objective equality, checked before either arm is timed and with the gradient
    // buffers cleared afterwards so neither arm inherits the other's accumulation.
    let (fused_nll, composed_nll) = (pass(false), pass(true));
    clear();
    ensure!(
        fused_nll.to_bits() == composed_nll.to_bits(),
        "the paired arms disagree on the objective ({fused_nll} fused, {composed_nll} \
         composed); the two chains are bit-identical by construction, so this is the \
         benchmark comparing two different losses and its delta would be meaningless"
    );
    // Warm both arms outside the timed window: first touch allocates, and the composed chain
    // allocates about fifty full-size fp32 tensors the fused one never asks for.
    for _ in 0..2 {
        let _ = pass(false);
        let _ = pass(true);
        clear();
    }
    let mut per_repeat = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let fused = event_timed(rounds, || pass(false))?;
        clear();
        let composed = event_timed(rounds, || pass(true))?;
        clear();
        per_repeat.push((fused, composed));
    }
    let spread = |values: &mut dyn Iterator<Item = f64>| -> f64 {
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for value in values {
            low = low.min(value);
            high = high.max(value);
        }
        high - low
    };
    let mean = |values: &mut dyn Iterator<Item = f64>| -> f64 {
        let (sum, count) = values.fold((0., 0.), |(sum, count), value| (sum + value, count + 1.));
        sum / count
    };
    // Read AFTER the timed window and before anything else allocates: the two arms are
    // sequential, so this is the composed arm's own peak plus the shared inputs, not the sum
    // of both arms.
    let (peak, reserved) = cuda_memory(false)?;
    Ok(PairedLoss {
        fused_ms: mean(&mut per_repeat.iter().map(|pair| pair.0)),
        composed_ms: mean(&mut per_repeat.iter().map(|pair| pair.1)),
        fused_drift_ms: spread(&mut per_repeat.iter().map(|pair| pair.0)),
        composed_drift_ms: spread(&mut per_repeat.iter().map(|pair| pair.1)),
        repeats,
        per_repeat,
        peak_allocated_mib: peak as f64 / 1048576.,
        peak_reserved_mib: reserved as f64 / 1048576.,
    })
}

pub fn run(args: BenchmarkArgs) -> Result<()> {
    args.model.validate()?;
    // Warmup must EXCEED `CAPTURE_AFTER_STEPS`, because the benchmark now arms the
    // forward+backward capture on exactly the step the trainer arms it on, after that many
    // warmup steps, and the arming step itself is not a timed step.
    ensure!(
        args.batch_size > 0 && args.steps >= 10 && args.warmup > CAPTURE_AFTER_STEPS,
        "benchmark needs batch > 0, steps >= 10, warmup > {CAPTURE_AFTER_STEPS}"
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
        RecipeKnobs::reference(args.model.x0_lambdas),
        args.fused,
        args.optimizer,
    )?;
    // ONE batch, materialized once, shared by every arm. Calling `gather()` inside the timed
    // loop - which this benchmark did until now - charged every step for 114 MB of device
    // `index_select` and `cat` that the training path never runs, so the reported step time
    // described the harness's own synthetic assembly as well as the model.
    let device_batch = gather();
    // What the training loader hands the engine: the same numbers in PINNED host memory, so
    // the upload is a real asynchronous H2D of the packed row block rather than a D2D copy.
    let pinned_batch = device_batch.host_copy(Some(device));
    // The eager comparison needs its own activation set; run before the training graph
    // reserves a private pool that cannot be reused by eager allocations.
    let paired = match args.paired_loss {
        true => Some(paired_loss_arm(
            &model,
            &store,
            &device_batch,
            args.steps,
            args.paired_repeats,
        )?),
        false => None,
    };
    // Warmup and capture in exactly training's order: `CAPTURE_AFTER_STEPS` steps on the
    // capture stream's warmup body, then the forward+backward capture, then the rest of the
    // warmup on the replay. Before this, `run` never called `arm_step_graph` at all, so its
    // timed loop measured an EAGER forward and backward with only the optimizer captured -
    // a configuration the trainer has not used since `runner.rs` started arming the step.
    for _ in 0..CAPTURE_AFTER_STEPS {
        let _ = engine.step(&model, &device_batch)?;
    }
    ensure!(
        engine.capture_ready(),
        "this device cannot capture the forward and backward, so the benchmark cannot measure \
         the configuration the trainer runs"
    );
    let _ = engine.arm_step_graph(&model, &device_batch)?;
    for _ in CAPTURE_AFTER_STEPS + 1..args.warmup {
        let _ = engine.step(&model, &device_batch)?;
    }
    let mut arm = |engine: &mut Engine, source: &Batch| -> Result<(f64, Tensor)> {
        Cuda::synchronize(0);
        let started = Instant::now();
        let mut losses = Vec::with_capacity(args.steps);
        for _ in 0..args.steps {
            losses.push(engine.step(&model, source)?.nll);
        }
        Cuda::synchronize(0);
        let ms = started.elapsed().as_secs_f64() * 1000. / args.steps as f64;
        Ok((ms, Tensor::stack(&losses, 0)))
    };
    cuda_memory(true)?;
    let sampler = HardwareSampler::start()?;
    let (device_step_ms, losses) = arm(&mut engine, &device_batch)?;
    let hardware = sampler.finish()?;
    let (peak, reserved) = cuda_memory(false)?;
    let (pinned_step_ms, _) = arm(&mut engine, &pinned_batch)?;
    let loader_step_ms = match &args.corpus {
        Some(directory) => loader_arm(directory, &args, &model, &mut engine, device)?,
        None => f64::NAN,
    };
    // The production step: what a training step costs when its batch arrives the way the
    // trainer's batches arrive. The device-fed arm above is the same kernels without the
    // host transfer, and the difference between them IS the upload.
    let step_ms = pinned_step_ms;
    ensure!(peak > 0, "CUDA allocator instrumentation returned zero");
    ensure!(
        losses.isfinite().all().int64_value(&[]) != 0,
        "nonfinite benchmark objective"
    );
    let title = format!(
        "CausalPatch captured-step benchmark | context {} | batch {} | {}",
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
        // The production configuration: captured forward+backward, captured optimizer, batch
        // uploaded from pinned host memory. The two arms beside it bracket the upload.
        ("captured step from a pinned host batch, milliseconds", pinned_step_ms),
        ("captured step from a device-resident batch, milliseconds", device_step_ms),
        (
            "packed row block upload, milliseconds (pinned-host arm minus device-resident arm)",
            pinned_step_ms - device_step_ms,
        ),
        (
            "captured step fed by the real corpus loader, milliseconds (NaN without --corpus)",
            loader_step_ms,
        ),
        (
            "host loader cost, milliseconds (corpus arm minus pinned-host arm; NaN without --corpus)",
            loader_step_ms - pinned_step_ms,
        ),
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
        // The GATE on any UNPAIRED step-time comparison. Job 5399 measured 230.99 TFLOPS on
        // a quiet card and job 5452 measured 193.32 under foreign tenants; a cross-run step
        // comparison between those two is not admissible in either direction. Nothing
        // enforces this automatically because the harness cannot know which run it is being
        // compared against - but the fraction is now on the chart, so the check is one
        // subtraction rather than an archaeology exercise.
        (
            "measured bf16 GEMM TFLOPS as a fraction of the quiet-card reference (job 5399)",
            peak_tflops / QUIET_CARD_GEMM_TFLOPS,
        ),
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
    // The paired comparison as SERIES, not as a log line: the two arms, their difference,
    // and the same-arm spread that bounds it, plus every repeat individually so a reader can
    // see whether the drift was a trend or a single disturbed alternation.
    if let Some(paired) = &paired {
        let mean_delta = paired.composed_ms - paired.fused_ms;
        summary.extend([
            ("paired loss chain, fused forward+backward, milliseconds", paired.fused_ms),
            ("paired loss chain, composed-ATen forward+backward, milliseconds", paired.composed_ms),
            ("paired loss chain delta, milliseconds (composed minus fused)", mean_delta),
            (
                "paired loss chain, fused arm spread across repeats, milliseconds",
                paired.fused_drift_ms,
            ),
            (
                "paired loss chain, composed arm spread across repeats, milliseconds",
                paired.composed_drift_ms,
            ),
            (
                "paired loss chain delta as a multiple of the fused arm's own spread",
                mean_delta / paired.fused_drift_ms.max(f64::MIN_POSITIVE),
            ),
            ("paired loss chain alternations", paired.repeats as f64),
            // Scheduling information, deliberately beside the timing rows: this arm's peak is
            // strictly HIGHER than either production step, because the composed chain
            // materializes the intermediates the fusion exists to delete. It is the most
            // OOM-prone job in the batch while being the one that most needs to run, so its
            // requirement is measured rather than assumed.
            ("paired loss chain peak allocator MiB", paired.peak_allocated_mib),
            ("paired loss chain peak reserved MiB", paired.peak_reserved_mib),
        ]);
    }
    // Owned labels, so the per-repeat rows are built where the series are and nothing has to
    // manufacture a `'static` string for them.
    let mut series: Vec<ReportSeries> = summary
        .into_iter()
        .map(|(label, value)| ReportSeries {
            label: label.to_owned(),
            values: vec![value as f32],
        })
        .collect();
    if let Some(paired) = &paired {
        for (index, (fused, composed)) in paired.per_repeat.iter().enumerate() {
            series.push(ReportSeries {
                label: format!("paired loss chain fused arm {}, milliseconds", index + 1),
                values: vec![*fused as f32],
            });
            series.push(ReportSeries {
                label: format!("paired loss chain composed arm {}, milliseconds", index + 1),
                values: vec![*composed as f32],
            });
        }
    }
    chart(
        &args.output,
        "timexer_segment_benchmark",
        &title,
        "named units",
        vec![args.batch_size as u64],
        series,
    )?;
    write_hardware(&args.output, &title, &hardware)?;
    if args.profile {
        // Fed from the PINNED host batch, so column 0 is the real asynchronous H2D of the
        // packed row block, which is the upload the trainer pays. The synthetic device
        // `gather()` that used to occupy a column of this chart is harness work, not a stand
        // -in for the host batch: the host batch is built on the CPU from mapped bar files,
        // and `audit-timexer-segment-loader` is what prices it.
        let mut phase_values: Vec<[f64; 6]> = Vec::new();
        for _ in 0..3 {
            let (_, phases) = engine.timed_step(&model, &pinned_batch)?;
            phase_values.push(phases);
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
                "H2D of the packed row block into the resident device batch",
                "forward backbone (NaN once captured: one replay is one launch)",
                "forward head and loss (NaN once captured)",
                "backward (NaN once captured)",
                "captured forward+backward replay",
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
        // Room for the per-class inputs beside the capture's private mempool: the pool holds
        // the whole step's working set for the life of the graph and cannot be released, so
        // the classes get what the allocator is merely caching on top of it.
        crate::torch::cuda::empty_cache();
        let kernels = kernel_profile(&model, &device_batch, 8)?;
        let layers = args.model.layers as f64;
        // Two residual `addcmul`s per layer for the residual scales, plus one for the x0
        // injection when the run has one.
        let addcmuls = if args.model.x0_lambdas.enabled() { 3. } else { 2. };
        let scaled = |row: &KernelRow| {
            let name: &str = row.name;
            let repeats = match name {
                // Two pre-norms per layer and two or three residual `addcmul`s, one U-net fold
                // on half the layers, and one value-residual mix on every layer but the source;
                // the embedding and the two pre-change reference forms run once for the whole
                // step.
                "RMSNorm" => 2. * layers,
                "residual addcmul" => addcmuls * layers,
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
        // value residual mix - the composed layer is layer 0, which is the mix's SOURCE and
        // pays no mix. The pre-norm runs twice per layer and the residual `addcmul` twice or
        // three times, matching `CausalPatchModel::kernel_classes`'s composed layer.
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
                    "residual addcmul" => addcmuls * each,
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
