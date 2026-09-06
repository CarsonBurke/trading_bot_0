//! Fused bf16 kernels for the CausalPatch backbone, and the composed-ATen references they
//! are bit-identical to.
//!
//! Two fusions, both chosen by measurement rather than taste:
//!
//! * [`relu_square`] replaces `x.relu().square()`. The composition makes four full passes
//!   over the `[tokens, ffn]` hidden activation in forward and six in backward; this makes
//!   two and three. At `tokens = 96_000`, `ffn = 2048` that is 15.73 GB/step of traffic
//!   removed over eight layers, 83% of the recipe change's whole traffic delta.
//! * [`rope`] replaces the packed `q‖k` rotation, which ATen has no operator for at all:
//!   the composition is two full-width products plus a half-crossing sum, seven passes
//!   over the `[tokens, 2·d_model]` block in forward, against this kernel's two.
//!
//! Both are differentiable from Rust with no further plumbing: the returned tensor carries
//! a real `grad_fn`, so [`tch::Tensor::backward`] and [`tch::Tensor::run_backward`] drive
//! the fused backward kernels. See `csrc/bridge.cpp` for the mechanism and why it is sound.
//!
//! Both are CUDA-graph-capturable: no host synchronization, no device-to-host read, and a
//! grid that is a pure function of the launch arguments. The only allocation is the output,
//! taken from the caching allocator exactly like any ATen op's, which is what a capture's
//! private mempool is for.

pub mod probe;

use std::ffi::{c_char, CStr};

use tch::Tensor;
use torch_sys::C_tensor;

extern "C" {
    fn fk_last_error() -> *const c_char;
    fn fk_relu_square(input: *const C_tensor) -> *mut C_tensor;
    fn fk_rope(
        input: *const C_tensor,
        cosine: *const C_tensor,
        sine: *const C_tensor,
        heads: i64,
    ) -> *mut C_tensor;
    fn fk_relu_square_backward_raw(input: *const C_tensor, grad: *const C_tensor)
        -> *mut C_tensor;
    fn fk_rope_backward_raw(
        grad: *const C_tensor,
        cosine: *const C_tensor,
        sine: *const C_tensor,
        heads: i64,
    ) -> *mut C_tensor;
    fn fk_stream_copy_tensor(input: *const C_tensor) -> *mut C_tensor;
    fn fk_timer_new() -> *mut std::ffi::c_void;
    fn fk_timer_start(timer: *mut std::ffi::c_void) -> i32;
    fn fk_timer_stop(timer: *mut std::ffi::c_void) -> f64;
    fn fk_timer_free(timer: *mut std::ffi::c_void);
}

/// Every failure these kernels can report is a shape, dtype, stride or device mismatch -
/// a programming error in the call site, not a runtime condition - so they panic like
/// every other `tch` tensor operation instead of colouring the backbone's forward with a
/// `Result` it has no way to handle.
fn finish(raw: *mut C_tensor, operation: &str) -> Tensor {
    if raw.is_null() {
        let message = unsafe { fk_last_error() };
        let message = if message.is_null() {
            "no message".to_owned()
        } else {
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        };
        panic!("fused {operation} failed: {message}");
    }
    unsafe { Tensor::from_ptr(raw) }
}

/// `relu(x)^2` in one pass, with a one-pass backward.
///
/// `input` must be a contiguous bf16 CUDA tensor of any shape; the output has the same
/// shape. Bit-identical to [`reference::relu_square`] in both directions, including the
/// NaN conventions, which are not the obvious ones: `relu` propagates NaN, so the forward
/// does, and the composition's gradient mask is `threshold_backward`'s
/// `NOT (relu(x) <= 0)`, TRUE at NaN, so the gradient propagates NaN as well.
///
/// Backward saves the INPUT, which costs no traffic (it already exists, as the
/// up-projection's output) and no extra memory over the composition (whose `square` saves
/// `relu`'s output, one tensor of the same size, while `addmm`'s backward never wants its
/// own output, so `x` is what the composition frees). The mask `NOT (relu(x) <= 0)` equals
/// `NOT (x <= 0)` and `relu(x) == x` wherever that mask holds, so the input recovers both
/// factors of `2·relu(x)·grad` exactly.
pub fn relu_square(input: &Tensor) -> Tensor {
    finish(unsafe { fk_relu_square(input.as_ptr()) }, "relu_square")
}

/// The packed `q‖k` rotary embedding in one pass, with a one-pass backward.
///
/// `input` is `[batch, length, 2·heads·head_dim]` bf16 on CUDA, the `q‖k` block of the QKV
/// projection - a strided view is fine and is the point, as long as its columns are dense
/// and its batch stride is exactly `length` rows. `cosine` and `sine` are the UNTILED
/// `[length, head_dim/2]` bf16 rotation rows, 24 KiB each at the real geometry: the
/// `[1, length, 2·d_model]` broadcast tiles the composition needs are not built at all.
///
/// The output is `[batch, length, 2, heads, head_dim]`, contiguous, which is exactly what
/// the composed form produced.
///
/// The rotation is treated as a constant, so backward saves no activation whatsoever -
/// the transpose of a rotation needs only the rotation.
pub fn rope(input: &Tensor, cosine: &Tensor, sine: &Tensor, heads: i64) -> Tensor {
    finish(
        unsafe { fk_rope(input.as_ptr(), cosine.as_ptr(), sine.as_ptr(), heads) },
        "rope",
    )
}

/// The gradient kernels called directly, with no autograd node around them.
///
/// These are NOT differentiable and are not the model path - [`relu_square`] and [`rope`]
/// already run them through autograd. They exist because timing a bandwidth-bound kernel
/// through `run_backward` charges it whatever objective the harness needed to reach it,
/// which at the `[96000, 2048]` FFN shape is several milliseconds of reduction and cast
/// traffic that says nothing about the kernel.
pub mod raw {
    use super::{
        fk_relu_square_backward_raw, fk_rope_backward_raw, finish, Tensor,
    };

    /// `x <= 0 ? 0 : 2·grad·x`, from the saved forward input.
    pub fn relu_square_backward(input: &Tensor, grad: &Tensor) -> Tensor {
        finish(
            unsafe { fk_relu_square_backward_raw(input.as_ptr(), grad.as_ptr()) },
            "relu_square backward",
        )
    }

    /// The transpose of the packed rotation: `grad` is `[batch, length, 2, heads, head_dim]`
    /// and the result is the dense `[batch, length, 2·heads·head_dim]` gradient.
    pub fn rope_backward(grad: &Tensor, cosine: &Tensor, sine: &Tensor, heads: i64) -> Tensor {
        finish(
            unsafe { fk_rope_backward_raw(grad.as_ptr(), cosine.as_ptr(), sine.as_ptr(), heads) },
            "rope backward",
        )
    }
}

/// A bf16 clone through a 128-bit vectorized copy kernel: this device's streaming roof,
/// measured with the same launch geometry and access width as the kernels measured against
/// it, so a percentage of it is an efficiency statement.
///
/// It exists because ATen's `copy_` is the wrong roof. On this card a same-dtype
/// device-to-device `copy_` lowers to `cudaMemcpyAsync`, which measures far below what a
/// vectorized copy kernel reaches - so a profile that used `copy_` as its bandwidth peak
/// reported kernels at over 100% of it, and every roofline fraction in that profile was
/// understated by the same factor. Both figures are reported by the probe.
pub fn stream_copy(input: &Tensor) -> Tensor {
    finish(
        unsafe { fk_stream_copy_tensor(input.as_ptr()) },
        "stream_copy",
    )
}

/// A CUDA-event interval on the stream the kernels launch on.
///
/// Host-clock timing is not good enough for this comparison and the repository has the scar
/// to prove it: a host-timed mean reported this card's copy peak at 630 GB/s while its own
/// kernels measured 1569 GB/s, which halved every roofline fraction in that profile.
/// [`Timer::stop`] synchronizes, so this is a measurement tool and never appears on the
/// model path.
pub struct Timer(*mut std::ffi::c_void);

impl Timer {
    pub fn new() -> Self {
        let raw = unsafe { fk_timer_new() };
        assert!(!raw.is_null(), "could not create CUDA events");
        Self(raw)
    }

    pub fn start(&mut self) {
        let status = unsafe { fk_timer_start(self.0) };
        assert_eq!(status, 0, "could not record the start event");
    }

    /// Milliseconds of device time since [`Timer::start`].
    pub fn stop(&mut self) -> f64 {
        let milliseconds = unsafe { fk_timer_stop(self.0) };
        assert!(milliseconds >= 0.0, "could not read the event interval");
        milliseconds
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        unsafe { fk_timer_free(self.0) }
    }
}

/// The composed-ATen forms the kernels replace, kept as the numerical reference and as the
/// exact statement of what a call site is giving up.
///
/// These are not dead code: they are what the equality tests compare against, and what the
/// microbenchmark times the kernels against.
pub mod reference {
    use tch::Tensor;

    /// `x.relu().square()`, four passes forward and six backward.
    pub fn relu_square(input: &Tensor) -> Tensor {
        input.relu().square()
    }

    /// Half-width rotary rows broadcast across the packed `q‖k` block, transcribed from
    /// `CausalPatchModel`'s `rotation_tiles`: column
    /// `t·heads·head_dim + h·head_dim + s·half + r` carries row `r`.
    ///
    /// The final `reshape` of an `expand` cannot be a view, so each tile is a real
    /// `[1, length, 2·d_model]` bf16 tensor - 768 KiB at the real geometry. The model
    /// builds them ONCE, in its constructor, which is why the microbenchmark hoists them
    /// out of the timed loop as well: charging their construction per step would flatter
    /// the fused kernel for no reason. The fused kernel does not need them at all.
    pub fn rotation_tiles(cosine: &Tensor, sine: &Tensor, heads: i64) -> (Tensor, Tensor) {
        let (length, half) = cosine
            .size2()
            .expect("rotary rows are [length, head_dim/2]");
        let tile = |rows: &Tensor| {
            rows.reshape([1, length, 1, 1, 1, half])
                .expand([1, length, 2, heads, 2, half], false)
                .reshape([1, length, 4 * heads * half])
        };
        (tile(cosine), tile(sine))
    }

    /// The packed rotation as `Block::rotate` composes it: two full-width products over the
    /// pre-built tiles and a half-crossing sum over their halves.
    pub fn rope_tiled(
        input: &Tensor,
        cosine_tile: &Tensor,
        sine_tile: &Tensor,
        heads: i64,
    ) -> Tensor {
        let (batch, length, columns) = input
            .size3()
            .expect("packed q‖k is [batch, length, 2·d_model]");
        let head_dim = columns / (2 * heads);
        let half = head_dim / 2;
        let cosine_product = input * cosine_tile;
        let sine_product = input * sine_tile;
        let halves = |product: &Tensor| {
            let parts = product
                .reshape([batch, length, 2, heads, 2, half])
                .split(1, 4);
            (parts[0].squeeze_dim(4), parts[1].squeeze_dim(4))
        };
        let (cos_low, cos_high) = halves(&cosine_product);
        let (sin_low, sin_high) = halves(&sine_product);
        Tensor::stack(&[&cos_low - &sin_high, &cos_high + &sin_low], 4)
            .reshape([batch, length, 2, heads, head_dim])
    }

    /// [`rope_tiled`] with the tiles built inline, which is the whole composed form the
    /// fused kernel replaces.
    pub fn rope(input: &Tensor, cosine: &Tensor, sine: &Tensor, heads: i64) -> Tensor {
        let (cosine_tile, sine_tile) = rotation_tiles(cosine, sine, heads);
        rope_tiled(input, &cosine_tile, &sine_tile, heads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    /// Rotary rows at the real geometry, built the way the model builds them: half-width
    /// `[length, head_dim/2]` bf16.
    fn rotation(length: i64, half: i64, device: Device) -> (Tensor, Tensor) {
        let exponents = Tensor::arange(half, (Kind::Float, device)) * (1.0 / half as f64);
        let inverse = (exponents * -(10000.0_f64.ln())).exp();
        let angles =
            Tensor::arange(length, (Kind::Float, device)).unsqueeze(1) * inverse.unsqueeze(0);
        (
            angles.cos().to_kind(Kind::BFloat16),
            angles.sin().to_kind(Kind::BFloat16),
        )
    }

    fn bf16_randn(shape: &[i64], device: Device) -> Tensor {
        Tensor::randn(shape, (Kind::Float, device)).to_kind(Kind::BFloat16)
    }

    /// Bit-for-bit equality of two bf16 tensors.
    ///
    /// bf16 widens to f32 losslessly, so f32 equality IS bit equality except for the two
    /// values f32 `==` cannot separate: NaN, which must match NaN, and the sign of zero,
    /// which the `signbit` clause pins. "Close enough" is the wrong predicate here - the
    /// whole point of these kernels is that a call site can adopt them without moving a
    /// training curve by one ulp.
    fn identical(left: &Tensor, right: &Tensor) -> bool {
        assert_eq!(left.size(), right.size(), "shapes differ");
        let left = left.reshape(-1).to_kind(Kind::Float);
        let right = right.reshape(-1).to_kind(Kind::Float);
        let nan = left.isnan().logical_and(&right.isnan());
        let value = left.eq_tensor(&right).logical_or(&nan);
        let sign = left
            .signbit()
            .eq_tensor(&right.signbit())
            .logical_or(&left.isnan());
        value
            .logical_and(&sign)
            .all()
            .int64_value(&[])
            == 1
    }

    fn max_absolute(left: &Tensor, right: &Tensor) -> f64 {
        (left.to_kind(Kind::Float) - right.to_kind(Kind::Float))
            .abs()
            .max()
            .double_value(&[])
    }

    /// The gradient of the composition IS `2·x·grad` with a SINGLE inexact rounding:
    /// `grad * 2` only decrements a bf16 exponent, and a product of two 8-bit mantissas is
    /// exact in fp32. That is the whole reason the fused backward can be bit-identical
    /// rather than merely close, and it is the claim the CUDA test's bit-equality would
    /// otherwise be silently relying on.
    #[test]
    fn composed_relu_square_gradient_is_one_rounded_product() {
        let device = Device::Cpu;
        let input = bf16_randn(&[64, 128], device).set_requires_grad(true);
        let upstream = bf16_randn(&[64, 128], device);
        let output = reference::relu_square(&input);
        let gradient = Tensor::run_backward(
            &[(&output * &upstream).sum(Kind::Float)],
            &[&input],
            false,
            false,
        );
        let raw = input.detach().to_kind(Kind::Float);
        // `masked_fill`, not a multiply by a 0/1 mask: `0.0 * negative` is `-0.0`, and the
        // reference writes a literal `+0.0` there, which `identical` distinguishes.
        let expected = (&raw * 2.0 * upstream.to_kind(Kind::Float))
            .to_kind(Kind::BFloat16)
            .masked_fill(&raw.le(0.0), 0.0);
        assert!(
            identical(&gradient[0], &expected),
            "the composition's gradient is not the single-rounding form the kernel implements"
        );
    }

    /// The gradient mask is `NOT (x <= 0)`, not `x > 0`, and the difference is only visible
    /// at NaN - where the composition PROPAGATES a NaN gradient, because
    /// `threshold_backward` is `self <= threshold ? 0 : grad` and `NaN <= 0` is false. The
    /// obvious `x > 0 ? 2·g·x : 0` kernel silently disagrees with the reference exactly
    /// where a numerical audit would look; this test is what forced the kernel to the
    /// measured form.
    #[test]
    fn composed_relu_square_gradient_mask_is_not_x_greater_than_zero() {
        let input = Tensor::from_slice(&[f32::NAN, -1.0, 0.0, 2.0])
            .to_kind(Kind::BFloat16)
            .set_requires_grad(true);
        let output = reference::relu_square(&input);
        assert_eq!(
            output.isnan().int64_value(&[0]),
            1,
            "the forward must propagate NaN"
        );
        let gradient = Tensor::run_backward(&[output.sum(Kind::Float)], &[&input], false, false);
        let values = gradient[0].to_kind(Kind::Float);
        assert_eq!(
            values.isnan().int64_value(&[0]),
            1,
            "the composition propagates NaN through the gradient mask"
        );
        assert_eq!(values.double_value(&[1]), 0.0);
        assert_eq!(values.double_value(&[2]), 0.0, "x == 0 is masked off");
        assert_eq!(values.double_value(&[3]), 4.0);
    }

    /// The packed layout the kernel's addressing depends on: the packed rotation equals the
    /// per-tensor half-width rotation applied separately to q and to k. Without this, every
    /// equality test against the packed reference would still pass while both forms rotated
    /// the wrong pairs.
    #[test]
    fn packed_rotation_equals_the_per_tensor_rotation() {
        let (heads, head_dim, length, batch) = (4i64, 16i64, 7i64, 2i64);
        let half = head_dim / 2;
        let width = heads * head_dim;
        let (cosine, sine) = rotation(length, half, Device::Cpu);
        let packed = bf16_randn(&[batch, length, 2 * width], Device::Cpu);
        let fused = reference::rope(&packed, &cosine, &sine, heads);

        let per_tensor = |tensor: &Tensor| {
            let parts = tensor
                .reshape([batch, length, heads, head_dim])
                .split(half, -1);
            let (low, high) = (&parts[0], &parts[1]);
            let rows = |r: &Tensor| r.reshape([1, length, 1, half]);
            Tensor::cat(
                &[
                    low * rows(&cosine) - high * rows(&sine),
                    high * rows(&cosine) + low * rows(&sine),
                ],
                -1,
            )
        };
        let split = packed.split(width, -1);
        let expected = Tensor::stack(&[per_tensor(&split[0]), per_tensor(&split[1])], 2);
        assert_eq!(fused.size(), expected.size());
        assert!(
            identical(&fused, &expected),
            "the packed rotation does not agree with the per-tensor rotation it replaced"
        );
    }

    /// Skipped, not failed, without a device - the same gate the repository's other CUDA
    /// tests use, so `cargo test` runs off the training box.
    fn cuda() -> Option<Device> {
        tch::Cuda::is_available().then_some(Device::Cuda(0))
    }

    #[test]
    fn fused_relu_square_is_bit_identical_including_gradients() {
        let Some(device) = cuda() else { return };
        // The second shape is deliberately not a multiple of eight, so the vectorized body
        // and the scalar path are both covered by the same assertion.
        for shape in [vec![4096i64, 2048i64], vec![1023i64]] {
            let values = bf16_randn(&shape, device);
            // Force the interesting cases into the sample rather than hoping for them.
            let mut zeros = values.narrow(0, 0, 1);
            let _ = zeros.fill_(0.0);
            let mut nans = values.narrow(0, 1, 1);
            let _ = nans.fill_(f64::NAN);
            let mut negatives = values.narrow(0, 2, 1);
            let _ = negatives.fill_(-3.5);

            let input = values.detach().copy().set_requires_grad(true);
            let composed_input = values.detach().copy().set_requires_grad(true);
            let fused = relu_square(&input);
            let composed = reference::relu_square(&composed_input);
            assert!(
                identical(&fused, &composed),
                "fused relu^2 forward differs from the composition at {shape:?}"
            );

            let upstream = bf16_randn(&shape, device);
            let fused_grad = Tensor::run_backward(
                &[(&fused * &upstream).sum(Kind::Float)],
                &[&input],
                false,
                false,
            );
            let composed_grad = Tensor::run_backward(
                &[(&composed * &upstream).sum(Kind::Float)],
                &[&composed_input],
                false,
                false,
            );
            assert!(
                identical(&fused_grad[0], &composed_grad[0]),
                "fused relu^2 backward differs from the composition at {shape:?}, max |delta| {}",
                max_absolute(&fused_grad[0], &composed_grad[0])
            );
        }
    }

    #[test]
    fn fused_rope_is_bit_identical_including_gradients() {
        let Some(device) = cuda() else { return };
        let (heads, head_dim, length, batch) = (8i64, 64i64, 375i64, 3i64);
        let half = head_dim / 2;
        let width = heads * head_dim;
        let (cosine, sine) = rotation(length, half, device);

        // The REAL call site's input: a `split_with_sizes` view of the `[.., 3·d_model]`
        // QKV projection, so the kernel is exercised on a strided view and the gradient
        // has to land in the right columns of a wider buffer.
        let source = bf16_randn(&[batch, length, 3 * width], device);
        let projection = source.detach().copy().set_requires_grad(true);
        let composed_projection = source.detach().copy().set_requires_grad(true);
        let packed = projection.split_with_sizes([2 * width, width], -1);
        let composed_packed = composed_projection.split_with_sizes([2 * width, width], -1);
        assert!(
            !packed[0].is_contiguous(),
            "the q‖k block should be a strided view of the projection"
        );

        let fused = rope(&packed[0], &cosine, &sine, heads);
        let composed = reference::rope(&composed_packed[0], &cosine, &sine, heads);
        assert_eq!(fused.size(), composed.size());
        assert!(
            identical(&fused, &composed),
            "fused rope forward differs from the composition, max |delta| {}",
            max_absolute(&fused, &composed)
        );

        let upstream = bf16_randn(&fused.size(), device);
        let fused_grad = Tensor::run_backward(
            &[(&fused * &upstream).sum(Kind::Float)],
            &[&projection],
            false,
            false,
        );
        let composed_grad = Tensor::run_backward(
            &[(&composed * &upstream).sum(Kind::Float)],
            &[&composed_projection],
            false,
            false,
        );
        assert!(
            identical(&fused_grad[0], &composed_grad[0]),
            "fused rope backward differs from the composition, max |delta| {}",
            max_absolute(&fused_grad[0], &composed_grad[0])
        );
    }

    /// Graph-capture compatibility, proven rather than argued: capture a forward AND a
    /// backward through both kernels, overwrite the input buffers in place, replay, and
    /// require the replayed gradients to be what a fresh eager evaluation produces on the
    /// new bytes. A host synchronization inside a kernel, an allocation outside the
    /// capture's private pool, or a launch geometry that depended on anything but the
    /// arguments would fail the capture or produce a stale replay.
    #[test]
    fn both_kernels_capture_and_replay_inside_a_cuda_graph() {
        let Some(device) = cuda() else { return };
        if !unsafe { torch_sys::at_cuda_graph_is_available() } {
            return;
        }
        let (heads, head_dim, length, batch) = (8i64, 64i64, 32i64, 2i64);
        let half = head_dim / 2;
        let width = heads * head_dim;
        let (cosine, sine) = rotation(length, half, device);
        let hidden = bf16_randn(&[batch * length, 512], device).set_requires_grad(true);
        let packed = bf16_randn(&[batch, length, 2 * width], device).set_requires_grad(true);

        let step = || {
            let activation = relu_square(&hidden);
            let rotated = rope(&packed, &cosine, &sine, heads);
            let objective = activation.sum(Kind::Float) + rotated.sum(Kind::Float);
            let gradients = Tensor::run_backward(&[&objective], &[&hidden, &packed], false, false);
            (
                gradients[0].shallow_clone(),
                gradients[1].shallow_clone(),
                objective,
            )
        };

        // Warm up outside the window: module load and the allocator's first touch are not
        // capturable work.
        drop(step());
        tch::Cuda::synchronize(0);

        let graph = unsafe { torch_sys::at_cuda_graph_new() };
        assert!(!graph.is_null(), "torch-sys returned a null CUDA graph");
        let captured;
        unsafe {
            torch_sys::at_cuda_graph_stream_begin(graph, 0);
            torch_sys::at_cuda_graph_capture_begin(graph, 0);
            captured = step();
            torch_sys::at_cuda_graph_capture_end(graph);
            torch_sys::at_cuda_graph_stream_end(graph);
        }
        assert!(
            unsafe { torch_sys::get_and_reset_last_err() }.is_null(),
            "capturing the fused kernels reported a libtorch error"
        );

        // New data into the SAME buffers: that is the only way a replay sees new inputs.
        let fresh_hidden = bf16_randn(&hidden.size(), device);
        let fresh_packed = bf16_randn(&packed.size(), device);
        tch::no_grad(|| {
            hidden.detach().copy_(&fresh_hidden);
            packed.detach().copy_(&fresh_packed);
        });
        unsafe {
            torch_sys::at_cuda_graph_stream_begin(graph, 0);
            torch_sys::at_cuda_graph_replay(graph, 0);
            torch_sys::at_cuda_graph_stream_end(graph);
        }
        tch::Cuda::synchronize(0);
        assert!(
            unsafe { torch_sys::get_and_reset_last_err() }.is_null(),
            "replaying the fused kernels reported a libtorch error"
        );

        // Against the COMPOSITION on the new bytes, eagerly: a stale replay cannot pass,
        // and neither can a replay that reproduced the kernels' own bug.
        let hidden_leaf = fresh_hidden.detach().copy().set_requires_grad(true);
        let packed_leaf = fresh_packed.detach().copy().set_requires_grad(true);
        let eager = {
            let activation = reference::relu_square(&hidden_leaf);
            let rotated = reference::rope(&packed_leaf, &cosine, &sine, heads);
            let objective = activation.sum(Kind::Float) + rotated.sum(Kind::Float);
            let gradients =
                Tensor::run_backward(&[&objective], &[&hidden_leaf, &packed_leaf], false, false);
            (
                gradients[0].shallow_clone(),
                gradients[1].shallow_clone(),
                objective,
            )
        };
        assert!(
            identical(&captured.0, &eager.0),
            "the replayed relu^2 gradient is not the composition's gradient of the new input"
        );
        assert!(
            identical(&captured.1, &eager.1),
            "the replayed rope gradient is not the composition's gradient of the new input"
        );
        assert_eq!(
            captured.2.double_value(&[]),
            eager.2.double_value(&[]),
            "the replayed objective does not match an eager evaluation on the same bytes"
        );
        unsafe { torch_sys::at_cuda_graph_free(graph) };
    }
}
