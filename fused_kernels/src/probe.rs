//! Microbenchmark: each fused kernel against the composed-ATen form it replaces, at the
//! real training shapes, plus the device's streaming roof measured in the same window.
//!
//! Three methodological choices, each forced by a measurement that came out wrong first.
//!
//! * **CUDA events, not a host clock.** The repository already shipped a profile whose
//!   host-timed "measured copy peak" was 630 GB/s while its own kernels measured 1569 - a
//!   roof below the things under it, halving every roofline fraction in the table.
//! * **Best of N, not the mean.** A peak is a maximum. This device has foreign non-mlq
//!   tenants; one of them landing inside one batch must not become the answer.
//! * **Interleaved, not sequential.** Best-of-N is not enough on its own: measuring the
//!   roof once and then the kernels afterwards produced 659 GB/s for the roof and 1181 GB/s
//!   for a kernel in the same job, because a neighbour occupied the device for tens of
//!   seconds - long enough to swallow every batch of one measurement and none of another.
//!   Every quantity here is therefore measured in every pass, so contention lands on all of
//!   them together and the ratios between them survive it.
//!
//! Milliseconds saved compares composed against fused through the same autograd harness, so
//! the harness cancels out of the difference. Achieved GB/s times the fused kernels ALONE,
//! through [`crate::raw`], because at the `[96000, 2048]` FFN shape the objective's own
//! reduction moves more bytes than the kernel does.

use tch::{Cuda, Device, Kind, Tensor};

use crate::{qk_norm_rope, raw, reference, relu_square, rope, stream_copy, Timer};

/// Timed executions per batch, batches per measurement, untimed warm-up rounds.
const ROUNDS: usize = 20;
const BATCHES: usize = 6;
const WARMUP: usize = 3;

/// One kernel measured both ways. Times are one layer's worth of work per execution.
///
/// Two rows may share a `fused_*` side against different baselines: the QK-norm fusion is
/// measured both against the whole composed-ATen sequence and against `_fused_rms_norm`
/// plus the already-landed rotary kernel, because the second is what the call site actually
/// looks like by the time this op lands and is therefore the honest marginal saving.
pub struct Comparison {
    pub name: &'static str,
    /// Forward alone, autograd off: the kernels and nothing else, both forms.
    pub composed_forward_ms: f64,
    pub fused_forward_ms: f64,
    /// Backward through the autograd harness. The harness is identical on both sides, so
    /// the DIFFERENCE is exact even though neither figure is the kernel alone.
    pub composed_backward_ms: f64,
    pub fused_backward_ms: f64,
    /// The fused backward kernel called directly, harness-free. The backward bandwidth is
    /// computed from this.
    pub fused_backward_kernel_ms: f64,
    /// Bytes the FUSED kernel must move: every input read once, every output written once,
    /// enumerated from the shapes. The composed form's traffic is a per-op ATen internal,
    /// so it stays measured milliseconds rather than a byte count nobody can audit.
    pub fused_forward_bytes: f64,
    pub fused_backward_bytes: f64,
    /// Invocations per step, so a per-execution saving becomes a step saving.
    pub layers: f64,
    /// Whether this row's saving belongs in the step total. `QK norm + rotary` measures the
    /// fused op against the WHOLE composed sequence, so its saving already contains the
    /// `packed rotary` row's; only the marginal row is additive on top of it, and double
    /// counting a 26 ms kernel is exactly the sort of arithmetic a table invites.
    pub additive: bool,
}

impl Comparison {
    pub fn fused_forward_gbs(&self) -> f64 {
        self.fused_forward_bytes / (self.fused_forward_ms / 1000.0) / 1e9
    }

    pub fn fused_backward_gbs(&self) -> f64 {
        self.fused_backward_bytes / (self.fused_backward_kernel_ms / 1000.0) / 1e9
    }

    /// Milliseconds the step loses to the composed form and would not lose to the kernel.
    pub fn step_ms_saved(&self) -> f64 {
        self.layers
            * ((self.composed_forward_ms - self.fused_forward_ms)
                + (self.composed_backward_ms - self.fused_backward_ms))
    }
}

/// The whole measurement, plus the roof it is a fraction of.
pub struct Measurement {
    pub comparisons: Vec<Comparison>,
    /// One read plus one write of a buffer the size of the FFN hidden, through a 128-bit
    /// vectorized copy kernel: the same launch geometry and access width as the kernels
    /// measured against it, so a percentage of it is an efficiency statement.
    pub streaming_roof_gbs: f64,
}

/// What the composition retains per layer and this fusion does not, measured from the
/// caching allocator rather than counted from shapes.
pub struct ActivationSaving {
    pub layers: i64,
    /// Allocator peak across a `layers`-deep chain, with every layer's rotated output kept
    /// alive exactly as attention keeps it.
    pub composed_peak_mib: f64,
    pub fused_peak_mib: f64,
    /// Bytes still live at the end of the chain: what autograd retains for its backward.
    pub composed_live_mib: f64,
    pub fused_live_mib: f64,
}

impl ActivationSaving {
    pub fn peak_saved_mib(&self) -> f64 {
        self.composed_peak_mib - self.fused_peak_mib
    }

    pub fn retained_saved_mib(&self) -> f64 {
        self.composed_live_mib - self.fused_live_mib
    }
}

/// Best per-task batch mean over [`BATCHES`] interleaved passes, CUDA-event timed.
fn interleaved(tasks: &mut [Box<dyn FnMut() + '_>]) -> Vec<f64> {
    let mut timer = Timer::new();
    let mut best = vec![f64::INFINITY; tasks.len()];
    for task in tasks.iter_mut() {
        for _ in 0..WARMUP {
            task();
        }
    }
    for _ in 0..BATCHES {
        for (index, task) in tasks.iter_mut().enumerate() {
            Cuda::synchronize(0);
            timer.start();
            for _ in 0..ROUNDS {
                task();
            }
            best[index] = best[index].min(timer.stop() / ROUNDS as f64);
        }
    }
    best
}

/// Every kernel at the real training geometry: `rows * origins` tokens, `d_model` wide,
/// `heads` heads, `ffn` hidden, over `layers` backbone layers.
pub fn measure(
    device: Device,
    rows: i64,
    origins: i64,
    d_model: i64,
    heads: i64,
    ffn: i64,
    layers: i64,
) -> Measurement {
    let tokens = rows * origins;
    let head_dim = d_model / heads;
    let half = head_dim / 2;
    let element = 2.0;

    // ReLU^2 over the FFN hidden activation.
    let hidden = Tensor::randn([tokens, ffn], (Kind::Float, device))
        .to_kind(Kind::BFloat16)
        .set_requires_grad(true);
    let hidden_detached = hidden.detach();
    let hidden_grad =
        Tensor::randn([tokens, ffn], (Kind::Float, device)).to_kind(Kind::BFloat16);
    let hidden_elements = (tokens * ffn) as f64;

    // Packed q‖k rotary. The input is a strided view of the QKV projection, exactly as the
    // model hands it over.
    let inverse = (Tensor::arange(half, (Kind::Float, device)) * (1.0 / half as f64)
        * -(10000.0_f64.ln()))
    .exp();
    let angles =
        Tensor::arange(origins, (Kind::Float, device)).unsqueeze(1) * inverse.unsqueeze(0);
    let cosine = angles.cos().to_kind(Kind::BFloat16);
    let sine = angles.sin().to_kind(Kind::BFloat16);
    let (cosine_tile, sine_tile) = reference::rotation_tiles(&cosine, &sine, heads);
    let projection = Tensor::randn([rows, origins, 3 * d_model], (Kind::Float, device))
        .to_kind(Kind::BFloat16)
        .set_requires_grad(true);
    let projection_detached = projection.detach();
    let rotated_grad =
        Tensor::randn([rows, origins, 2, heads, head_dim], (Kind::Float, device))
            .to_kind(Kind::BFloat16);
    let rope_elements = (tokens * 2 * d_model) as f64;
    let rotation_bytes = 2.0 * (origins * half) as f64 * element;
    let packed = |tensor: &Tensor| {
        tensor.split_with_sizes([2 * d_model, d_model], -1)[0].shallow_clone()
    };

    let objective = |output: Tensor, leaf: &Tensor| {
        let _ = Tensor::run_backward(&[output.sum(Kind::Float)], &[leaf], false, false);
    };
    // Order matters only in that every task is in the same list: they are timed one after
    // another inside every pass, so a neighbour's burst is shared out over all of them.
    let mut tasks: Vec<Box<dyn FnMut()>> = vec![
        Box::new(|| drop(stream_copy(&hidden_detached))),
        Box::new(|| drop(tch::no_grad(|| reference::relu_square(&hidden_detached)))),
        Box::new(|| drop(tch::no_grad(|| relu_square(&hidden_detached)))),
        Box::new(|| drop(reference::relu_square(&hidden))),
        Box::new(|| objective(reference::relu_square(&hidden), &hidden)),
        Box::new(|| drop(relu_square(&hidden))),
        Box::new(|| objective(relu_square(&hidden), &hidden)),
        Box::new(|| drop(raw::relu_square_backward(&hidden_detached, &hidden_grad))),
        Box::new(|| {
            drop(tch::no_grad(|| {
                reference::rope_tiled(
                    &packed(&projection_detached),
                    &cosine_tile,
                    &sine_tile,
                    heads,
                )
            }))
        }),
        Box::new(|| {
            drop(tch::no_grad(|| {
                rope(&packed(&projection_detached), &cosine, &sine, heads)
            }))
        }),
        Box::new(|| {
            drop(reference::rope_tiled(
                &packed(&projection),
                &cosine_tile,
                &sine_tile,
                heads,
            ))
        }),
        Box::new(|| {
            objective(
                reference::rope_tiled(&packed(&projection), &cosine_tile, &sine_tile, heads),
                &projection,
            )
        }),
        Box::new(|| drop(rope(&packed(&projection), &cosine, &sine, heads))),
        Box::new(|| {
            objective(
                rope(&packed(&projection), &cosine, &sine, heads),
                &projection,
            )
        }),
        Box::new(|| drop(raw::rope_backward(&rotated_grad, &cosine, &sine, heads))),
        // QK-norm + rotary. `packed(&projection)` is the RAW block: the whole point is that
        // the normalized one is never built.
        Box::new(|| {
            drop(tch::no_grad(|| {
                reference::qk_norm_rope(&packed(&projection_detached), &cosine, &sine, heads)
            }))
        }),
        Box::new(|| {
            drop(tch::no_grad(|| {
                reference::qk_norm_fused_rope(
                    &packed(&projection_detached),
                    &cosine,
                    &sine,
                    heads,
                )
            }))
        }),
        Box::new(|| {
            drop(tch::no_grad(|| {
                qk_norm_rope(&packed(&projection_detached), &cosine, &sine, heads)
            }))
        }),
        Box::new(|| {
            drop(reference::qk_norm_rope(
                &packed(&projection),
                &cosine,
                &sine,
                heads,
            ))
        }),
        Box::new(|| {
            objective(
                reference::qk_norm_rope(&packed(&projection), &cosine, &sine, heads),
                &projection,
            )
        }),
        Box::new(|| {
            drop(reference::qk_norm_fused_rope(
                &packed(&projection),
                &cosine,
                &sine,
                heads,
            ))
        }),
        Box::new(|| {
            objective(
                reference::qk_norm_fused_rope(&packed(&projection), &cosine, &sine, heads),
                &projection,
            )
        }),
        Box::new(|| drop(qk_norm_rope(&packed(&projection), &cosine, &sine, heads))),
        Box::new(|| {
            objective(
                qk_norm_rope(&packed(&projection), &cosine, &sine, heads),
                &projection,
            )
        }),
        Box::new(|| {
            drop(raw::qk_norm_rope_backward(
                &rotated_grad,
                &packed(&projection_detached),
                &cosine,
                &sine,
                heads,
                crate::QK_NORM_ROUNDING,
            ))
        }),
    ];
    let timings = interleaved(&mut tasks);

    Measurement {
        comparisons: vec![
            Comparison {
                name: "ReLU^2",
                composed_forward_ms: timings[1],
                fused_forward_ms: timings[2],
                // `full` minus `forward with recording on`: what remains is the backward
                // plus the objective, and the objective is the same on both sides.
                composed_backward_ms: timings[4] - timings[3],
                fused_backward_ms: timings[6] - timings[5],
                fused_backward_kernel_ms: timings[7],
                // Read x, write y.
                fused_forward_bytes: 2.0 * hidden_elements * element,
                // Read x, read grad, write dx.
                fused_backward_bytes: 3.0 * hidden_elements * element,
                layers: layers as f64,
                additive: true,
            },
            Comparison {
                name: "packed rotary",
                composed_forward_ms: timings[8],
                fused_forward_ms: timings[9],
                composed_backward_ms: timings[11] - timings[10],
                fused_backward_ms: timings[13] - timings[12],
                fused_backward_kernel_ms: timings[14],
                // Read q‖k, write the rotated buffer. The cos/sin rows are 24 KiB each and
                // L2-resident across the launch; they are counted and round to nothing.
                fused_forward_bytes: 2.0 * rope_elements * element + rotation_bytes,
                fused_backward_bytes: 2.0 * rope_elements * element + rotation_bytes,
                layers: layers as f64,
                additive: true,
            },
            Comparison {
                name: "QK norm + rotary",
                composed_forward_ms: timings[15],
                fused_forward_ms: timings[17],
                composed_backward_ms: timings[19] - timings[18],
                fused_backward_ms: timings[23] - timings[22],
                fused_backward_kernel_ms: timings[24],
                // Read the raw q‖k, write the rotated buffer. `rstd` is neither written nor
                // read back, and the normalized block never exists.
                fused_forward_bytes: 2.0 * rope_elements * element + rotation_bytes,
                // Read the upstream gradient, read the raw q‖k, write the gradient. The
                // third pass is what buys the retained `rstd` and the normalized block.
                fused_backward_bytes: 3.0 * rope_elements * element + rotation_bytes,
                layers: layers as f64,
                additive: false,
            },
            Comparison {
                name: "QK norm marginal",
                composed_forward_ms: timings[16],
                fused_forward_ms: timings[17],
                composed_backward_ms: timings[21] - timings[20],
                fused_backward_ms: timings[23] - timings[22],
                fused_backward_kernel_ms: timings[24],
                fused_forward_bytes: 2.0 * rope_elements * element + rotation_bytes,
                fused_backward_bytes: 3.0 * rope_elements * element + rotation_bytes,
                layers: layers as f64,
                additive: true,
            },
        ],
        streaming_roof_gbs: 2.0 * hidden_elements * element / (timings[0] / 1000.0) / 1e9,
    }
}

/// Allocator bytes, from the same torch that runs the kernels. `max_memory_allocated` is
/// the peak of the LIVE set, which is what a graph capture's private mempool has to fit on
/// top of; `memory_allocated` is what is still live, i.e. what autograd retained.
fn allocator_mib(reset: bool) -> (f64, f64) {
    use pyo3::types::PyAnyMethods;
    pyo3::Python::attach(|python| {
        let cuda = python
            .import("torch")
            .expect("torch")
            .getattr("cuda")
            .expect("torch.cuda");
        cuda.call_method0("init").expect("cuda init");
        if reset {
            cuda.call_method0("empty_cache").expect("empty_cache");
            cuda.call_method1("reset_peak_memory_stats", (0,))
                .expect("reset_peak_memory_stats");
        }
        let read = |name: &str| {
            cuda.call_method1(name, (0,))
                .expect("allocator statistic")
                .extract::<u64>()
                .expect("allocator statistic is an integer") as f64
                / 1048576.0
        };
        (read("max_memory_allocated"), read("memory_allocated"))
    })
}

/// The activation cost of the composition that this fusion does not pay, MEASURED: build a
/// `layers`-deep chain of the QK-norm-then-rotate step, keep every layer's rotated output
/// and its projection alive exactly as attention and the QKV backward keep them, and read
/// the allocator.
///
/// Both arms are identical apart from the op, and each starts from an emptied cache with the
/// peak counter reset, so the difference is the composition's own footprint: per layer, the
/// contiguous copy ATen makes of the STRIDED `q‖k` view, the normalized block that copy
/// feeds, and the fp32 `[tokens, 2·heads]` `rstd` that `_fused_rms_norm` retains for its
/// backward. The first two are transient and bound the PEAK; the third is retained for the
/// whole step and is what the `layers`-fold difference in live bytes is made of.
///
/// The projection is drawn DIRECTLY in bf16. Drawing it in fp32 and casting made the first
/// version of this measurement useless: an fp32 `[rows, origins, 3·d_model]` scratch is
/// 562.5 MiB at batch 256, three times the transient the measurement is trying to see, so
/// both peaks landed on the scratch and reported only the retained difference.
pub fn activation_saving(
    device: Device,
    rows: i64,
    origins: i64,
    d_model: i64,
    heads: i64,
    layers: i64,
) -> ActivationSaving {
    let half = d_model / heads / 2;
    let inverse = (Tensor::arange(half, (Kind::Float, device)) * (1.0 / half as f64)
        * -(10000.0_f64.ln()))
    .exp();
    let angles =
        Tensor::arange(origins, (Kind::Float, device)).unsqueeze(1) * inverse.unsqueeze(0);
    let cosine = angles.cos().to_kind(Kind::BFloat16);
    let sine = angles.sin().to_kind(Kind::BFloat16);

    let chain = |fused: bool| {
        let (_, before) = allocator_mib(true);
        // Held to the end of the arm: the rotated buffer is what SDPA saves for its
        // backward, and the projection is what the QKV linear's backward and the value path
        // hold. Everything else a layer allocates is the op's own business.
        let mut retained: Vec<Tensor> = Vec::new();
        for _ in 0..layers {
            let projection = Tensor::randn(
                [rows, origins, 3 * d_model],
                (Kind::BFloat16, device),
            )
            .set_requires_grad(true);
            let packed = projection.split_with_sizes([2 * d_model, d_model], -1);
            let rotated = if fused {
                qk_norm_rope(&packed[0], &cosine, &sine, heads)
            } else {
                reference::qk_norm_fused_rope(&packed[0], &cosine, &sine, heads)
            };
            retained.push(rotated);
            retained.push(projection);
        }
        Cuda::synchronize(0);
        let (peak, live) = allocator_mib(false);
        drop(retained);
        (peak - before, live - before)
    };

    // The composed arm first, then the fused one, each behind its own `empty_cache`, so
    // neither inherits the other's fragmentation.
    let (composed_peak_mib, composed_live_mib) = chain(false);
    let (fused_peak_mib, fused_live_mib) = chain(true);
    ActivationSaving {
        layers,
        composed_peak_mib,
        fused_peak_mib,
        composed_live_mib,
        fused_live_mib,
    }
}
