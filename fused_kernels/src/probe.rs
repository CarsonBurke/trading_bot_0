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

use crate::{raw, reference, relu_square, rope, stream_copy, Timer};

/// Timed executions per batch, batches per measurement, untimed warm-up rounds.
const ROUNDS: usize = 20;
const BATCHES: usize = 6;
const WARMUP: usize = 3;

/// One kernel measured both ways. Times are one layer's worth of work per execution.
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

/// Both kernels at the real training geometry: `rows * origins` tokens, `d_model` wide,
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
            },
        ],
        streaming_roof_gbs: 2.0 * hidden_elements * element / (timings[0] / 1000.0) / 1e9,
    }
}
