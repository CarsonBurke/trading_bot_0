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
    fn fk_qk_norm_rope(
        input: *const C_tensor,
        cosine: *const C_tensor,
        sine: *const C_tensor,
        heads: i64,
        rounding: i64,
    ) -> *mut C_tensor;
    fn fk_qk_norm_rope_backward_raw(
        grad: *const C_tensor,
        input: *const C_tensor,
        cosine: *const C_tensor,
        sine: *const C_tensor,
        heads: i64,
        rounding: i64,
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
///
/// Off CUDA this IS [`reference::relu_square`]: there is no kernel to run, the reference
/// is bit-identical by the tests below, and it lives here rather than at the call site so
/// the backbone has exactly one spelling of the op. Training asserts CUDA long before it
/// reaches this; the CPU path exists for the model's own equality tests.
pub fn relu_square(input: &Tensor) -> Tensor {
    if !input.device().is_cuda() {
        return reference::relu_square(input);
    }
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
///
/// Off CUDA this is [`reference::rope`], for the reason given on [`relu_square`].
pub fn rope(input: &Tensor, cosine: &Tensor, sine: &Tensor, heads: i64) -> Tensor {
    if !input.device().is_cuda() {
        return reference::rope(input, cosine, sine, heads);
    }
    finish(
        unsafe { fk_rope(input.as_ptr(), cosine.as_ptr(), sine.as_ptr(), heads) },
        "rope",
    )
}

/// The RMSNorm epsilon the QK-norm fusion is defined at, matching the backbone's `NORM_EPS`
/// and `world_model.rs`'s `BAR_NORM_EPS`. It is a constant and not an argument for the same
/// reason the kernel bakes it in: `_fused_rms_norm` resolves `eps = None` to the accumulate
/// type's epsilon, `FLT_EPSILON = 1.19e-7` for a bf16 input, so a call site that chooses its
/// own epsilon is a call site that can silently stop matching the composition. At unit scale
/// that default is bit-indistinguishable from this one; on a collapsed head block it is not,
/// which `the_norm_epsilon_is_observable_where_it_is_load_bearing` shows.
pub const NORM_EPS: f64 = 1e-6;

/// The fp32 contraction forms ATen's own build emitted, as a three-bit selector: bit 0 the
/// squared accumulation of the RMS reduction, bit 1 the gradient statistic's accumulation,
/// bit 2 the gradient's `f -= (x·rstd)·stats`. A set bit means that operation contracts
/// into a single `fma`.
///
/// This is MEASURED, not chosen: `nvcc` contracts `a*b + c` by default, so which form a
/// kernel implements is a property of the compiler that built it, it changes the last bit of
/// a reduction over 64 values, and one flipped bit in `rstd` moves roughly one output
/// element in 10^5. The measurement
/// (`qk_norm_rope_rounding_is_the_measured_pair`) found that bit 1 must be SET and bit 2
/// must be CLEAR, and that bit 0 cannot be measured at all because it is provably inert: the
/// accumulated value is a product of two bf16 mantissas, which is exact in fp32, so
/// `fmaf(v, v, a)` and `a + v*v` round identically. This ships the form with no gratuitous
/// contraction in the two places where contraction is observable and where ATen has one.
pub const QK_NORM_ROUNDING: i64 = 2;

/// Per-head QK-normalization AND the packed `q‖k` rotary in one pass, with a one-pass
/// backward.
///
/// `input` is the RAW, UN-normalized `q‖k` block: `[batch, length, 2·heads·head_dim]` bf16
/// on CUDA, exactly what `split_with_sizes` hands back from the QKV projection, strided
/// view included. `cosine` and `sine` are the untiled `[length, head_dim/2]` rotation rows,
/// the same tensors [`rope`] takes. The output is the contiguous
/// `[batch, length, 2, heads, head_dim]` buffer, so the call site keeps its `.split(1, 2)`.
///
/// Each head block of `head_dim` columns is simultaneously one gainless, biasless RMS
/// normalization group (epsilon [`NORM_EPS`]) and one rotary block, which is what makes the
/// fusion exact rather than approximate. `V` is neither normalized nor rotated and is not
/// part of this block: the packed layout is `q‖k` only, `[2·heads, head_dim]` per token with
/// column `t·heads·head_dim + h·head_dim + d`, and the value block is the SECOND output of
/// the same split.
///
/// Bit-identical to [`reference::qk_norm_rope`] in both directions. The normalized block -
/// one `[tokens, 2·d_model]` bf16 tensor, plus ATen's contiguous copy of the raw block, plus
/// the fp32 `rstd` the composition retains for its backward - is never materialized: the
/// backward recomputes `rstd` from the raw block it already has to stream.
///
/// Off CUDA this is [`reference::qk_norm_rope`], for the reason given on [`relu_square`],
/// and it is dtype-agnostic there: the model's CPU tests reach this in fp32 as well as
/// bf16, so nothing on the reference path assumes bf16.
pub fn qk_norm_rope(input: &Tensor, cosine: &Tensor, sine: &Tensor, heads: i64) -> Tensor {
    if !input.device().is_cuda() {
        return reference::qk_norm_rope(input, cosine, sine, heads);
    }
    finish(
        unsafe {
            fk_qk_norm_rope(
                input.as_ptr(),
                cosine.as_ptr(),
                sine.as_ptr(),
                heads,
                QK_NORM_ROUNDING,
            )
        },
        "qk_norm_rope",
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
        fk_qk_norm_rope, fk_qk_norm_rope_backward_raw, fk_relu_square_backward_raw,
        fk_rope_backward_raw, finish, Tensor,
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

    /// The fused QK-norm rotary backward, with the rounding selector exposed: `grad` is
    /// `[batch, length, 2, heads, head_dim]`, `input` the RAW packed block the forward saw,
    /// and the result is the dense `[batch, length, 2·heads·head_dim]` gradient.
    ///
    /// `rounding` is a parameter here and nowhere else. The model path is
    /// [`super::qk_norm_rope`], which passes [`super::QK_NORM_ROUNDING`]; this entry point
    /// exists so the discovery test can prove that constant is the only bit-identical one.
    pub fn qk_norm_rope_backward(
        grad: &Tensor,
        input: &Tensor,
        cosine: &Tensor,
        sine: &Tensor,
        heads: i64,
        rounding: i64,
    ) -> Tensor {
        finish(
            unsafe {
                fk_qk_norm_rope_backward_raw(
                    grad.as_ptr(),
                    input.as_ptr(),
                    cosine.as_ptr(),
                    sine.as_ptr(),
                    heads,
                    rounding,
                )
            },
            "qk_norm_rope backward",
        )
    }

    /// The fused QK-norm rotary forward with the rounding selector exposed, differentiable
    /// like [`super::qk_norm_rope`] but not the model path, for the same reason.
    pub fn qk_norm_rope(
        input: &Tensor,
        cosine: &Tensor,
        sine: &Tensor,
        heads: i64,
        rounding: i64,
    ) -> Tensor {
        finish(
            unsafe {
                fk_qk_norm_rope(
                    input.as_ptr(),
                    cosine.as_ptr(),
                    sine.as_ptr(),
                    heads,
                    rounding,
                )
            },
            "qk_norm_rope",
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

    /// The gainless, biasless per-head RMS normalization the recipe applies to the packed
    /// `q‖k` block BEFORE the rotation, transcribed from `Block::forward`:
    /// `rms_norm(packed.reshape([b, t, 2·heads, head_dim])).reshape([b, t, 2·width])`.
    ///
    /// `_fused_rms_norm`, not `rms_norm`: the latter registers as a math composite on CUDA
    /// and hides the kernel this has to be bit-identical to. Both reshapes are views even
    /// when `input` is a strided slice of the projection, so the only tensor this
    /// materializes is the normalized block itself - plus the contiguous copy ATen makes of
    /// the strided input, and the fp32 `[tokens, 2·heads]` `rstd` it retains for backward.
    pub fn qk_norm(input: &Tensor, heads: i64) -> Tensor {
        let (batch, length, columns) = input
            .size3()
            .expect("packed q‖k is [batch, length, 2·d_model]");
        let head_dim = columns / (2 * heads);
        input
            .reshape([batch, length, 2 * heads, head_dim])
            .internal_fused_rms_norm([head_dim], None::<&Tensor>, Some(super::NORM_EPS))
            .0
            .reshape([batch, length, columns])
    }

    /// The whole composed form the fused QK-norm rotary replaces: normalize per head, then
    /// rotate the packed block. This is the sequence `Block::forward` runs today.
    pub fn qk_norm_rope(input: &Tensor, cosine: &Tensor, sine: &Tensor, heads: i64) -> Tensor {
        rope(&qk_norm(input, heads), cosine, sine, heads)
    }

    /// The same composition with the rotation already fused: `_fused_rms_norm` followed by
    /// [`super::rope`]. This is the state of the call site AFTER the packed rotary lands and
    /// BEFORE this fusion does, so it is the honest baseline for what fusing the norm buys
    /// on top of it. Bit-identical to [`qk_norm_rope`], since the rotary kernel is.
    pub fn qk_norm_fused_rope(
        input: &Tensor,
        cosine: &Tensor,
        sine: &Tensor,
        heads: i64,
    ) -> Tensor {
        super::rope(&qk_norm(input, heads), cosine, sine, heads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};
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
        differing(left, right) == 0
    }

    /// How many elements of two bf16 tensors are not bit-for-bit equal. This, not
    /// `max |delta|`, is the diagnostic that survives a NaN in the input: `NaN - NaN` is
    /// NaN, so a subtraction-based summary of a tensor that legitimately contains NaN
    /// reports NaN whether the kernel is right or wrong.
    fn differing(left: &Tensor, right: &Tensor) -> i64 {
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
            .logical_not()
            .sum(Kind::Int64)
            .int64_value(&[])
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

    /// The layout claim the fused kernel's normalization groups depend on: normalizing the
    /// packed block viewed as `[.., 2·heads, head_dim]` IS normalizing q and k separately,
    /// per head, per token. Without this an equality test against the packed reference
    /// would pass while both forms normalized across head boundaries.
    #[test]
    fn packed_qk_norm_equals_the_per_tensor_per_head_norm() {
        let (heads, head_dim, length, batch) = (4i64, 16i64, 7i64, 2i64);
        let width = heads * head_dim;
        let packed = bf16_randn(&[batch, length, 2 * width], Device::Cpu);
        let fused = reference::qk_norm(&packed, heads);

        let per_tensor = |tensor: &Tensor| {
            tensor
                .reshape([batch, length, heads, head_dim])
                .internal_fused_rms_norm([head_dim], None::<&Tensor>, Some(NORM_EPS))
                .0
                .reshape([batch, length, width])
        };
        let split = packed.split(width, -1);
        let expected = Tensor::cat(&[per_tensor(&split[0]), per_tensor(&split[1])], -1);
        assert!(
            identical(&fused, &expected),
            "the packed QK norm does not agree with per-head norms of q and k"
        );
    }

    /// The OFF-CUDA path of the public op, in fp32, forward and backward. This is the path
    /// ten of the model's CPU tests take through `Block::forward`, two of them in fp32, so
    /// it is a real contract and not a courtesy: if `qk_norm_rope` raised off CUDA, or
    /// assumed bf16, or returned the wrong axis order, those tests would be what discovered
    /// it. The comparison is against an INDEPENDENT per-tensor composition - split `q‖k`,
    /// normalize each per head, rotate each pair - so it pins the layout too, and it runs
    /// the gradient because the model's CPU tests differentiate through here.
    #[test]
    fn the_off_cuda_path_is_the_per_tensor_composition_in_fp32() {
        let (heads, head_dim, length, batch) = (4i64, 16i64, 7i64, 2i64);
        let (half, width) = (head_dim / 2, heads * head_dim);
        let (cosine, sine) = rotation(length, half, Device::Cpu);
        let (cosine, sine) = (cosine.to_kind(Kind::Float), sine.to_kind(Kind::Float));

        let sample = Tensor::randn([batch, length, 2 * width], (Kind::Float, Device::Cpu));
        let dispatched_input = sample.detach().copy().set_requires_grad(true);
        let dispatched = qk_norm_rope(&dispatched_input, &cosine, &sine, heads);
        assert_eq!(
            dispatched.size(),
            vec![batch, length, 2, heads, head_dim],
            "the off-CUDA path must produce the same axis order the kernel does"
        );
        assert_eq!(dispatched.kind(), Kind::Float, "fp32 in, fp32 out");

        let expected_input = sample.detach().copy().set_requires_grad(true);
        let per_tensor = |tensor: Tensor| {
            let normed = tensor
                .reshape([batch, length, heads, head_dim])
                .internal_fused_rms_norm([head_dim], None::<&Tensor>, Some(NORM_EPS))
                .0;
            let parts = normed.split(half, -1);
            let rows = |r: &Tensor| r.reshape([1, length, 1, half]);
            Tensor::cat(
                &[
                    &parts[0] * rows(&cosine) - &parts[1] * rows(&sine),
                    &parts[1] * rows(&cosine) + &parts[0] * rows(&sine),
                ],
                -1,
            )
        };
        let split = expected_input.split(width, -1);
        let expected = Tensor::stack(
            &[
                per_tensor(split[0].shallow_clone()),
                per_tensor(split[1].shallow_clone()),
            ],
            2,
        );
        assert!(
            max_absolute(&dispatched, &expected) < 1e-5,
            "the off-CUDA path disagrees with the per-tensor composition, max |delta| {}",
            max_absolute(&dispatched, &expected)
        );

        let upstream = Tensor::randn(dispatched.size(), (Kind::Float, Device::Cpu));
        let gradient = Tensor::run_backward(
            &[(&dispatched * &upstream).sum(Kind::Float)],
            &[&dispatched_input],
            false,
            false,
        );
        let reference_gradient = Tensor::run_backward(
            &[(&expected * &upstream).sum(Kind::Float)],
            &[&expected_input],
            false,
            false,
        );
        assert!(
            max_absolute(&gradient[0], &reference_gradient[0]) < 1e-5,
            "the off-CUDA gradient disagrees with the per-tensor composition's, max |delta| {}",
            max_absolute(&gradient[0], &reference_gradient[0])
        );
    }

    /// The epsilon is 1e-6 and it MATTERS - but not for the reason the recipe report gives.
    /// `_fused_rms_norm_cuda` resolves `eps = None` to the ACCUMULATE type's epsilon, which
    /// for a bf16 input is `FLT_EPSILON = 1.19e-7` and not `finfo(bf16).eps = 7.8e-3`; at
    /// unit scale that default and 1e-6 produce bit-identical bf16 output, which is why the
    /// first version of this test failed. The difference appears where the epsilon is
    /// load-bearing: a head block whose RMS is near the epsilon itself, which is exactly the
    /// state QK-norm has to survive (a dead head, or a token whose projection collapsed).
    /// The kernel bakes 1e-6 in, so this pins that the choice is observable rather than
    /// letting a default silently stand in for the backbone's.
    #[test]
    fn the_norm_epsilon_is_observable_where_it_is_load_bearing() {
        let values = bf16_randn(&[4, 6, 64], Device::Cpu) * 1e-4;
        let chosen = values
            .internal_fused_rms_norm([64], None::<&Tensor>, Some(NORM_EPS))
            .0;
        let defaulted = values.internal_fused_rms_norm([64], None::<&Tensor>, None).0;
        assert!(
            !identical(&chosen, &defaulted),
            "eps=1e-6 and the resolved default agree even on a near-zero block, so this \
             convention is untested"
        );
    }

    /// The device a CUDA test runs on, together with the process-wide claim on it that lets
    /// `cargo test -p fused_kernels` be green without a `--test-threads=1` incantation.
    ///
    /// WHY A LOCK, and why not a private stream. `CUDAGraph::capture_begin` captures in
    /// `cudaStreamCaptureModeGlobal`, the only mode libtorch's C++ API exposes, and that
    /// mode rejects any "unsafe" CUDA action anywhere in the PROCESS for as long as the
    /// capture is open. Cargo runs tests on one thread each, so a sibling CUDA test doing a
    /// host-to-device copy dies inside `CachingHostAllocator` with
    /// `cudaErrorStreamCaptureUnsupported` and takes the capture down with it
    /// (`cudaErrorStreamCaptureInvalidated`). Capturing on a side stream does NOT help: the
    /// restriction is on the process, not on the stream. The capture test passing under
    /// `--test-threads=1` was a scheduling accident, and `WireKernels` caught it on the
    /// merged tip (job 5185: 6 of 14 failed).
    ///
    /// So the device is handed out under a mutex, and it is handed out ONLY under the mutex:
    /// [`CudaClaim`] is the sole way for a test in this module to name a CUDA device, which
    /// is what stops the next CUDA test anyone adds from forgetting to serialize. Bind the
    /// claim to a `let` for the whole test body - `*cuda()?` in a temporary would release it
    /// at the end of that statement.
    struct CudaClaim {
        /// Never read: its Drop is the whole point.
        _claim: MutexGuard<'static, ()>,
        device: Device,
    }

    impl std::ops::Deref for CudaClaim {
        type Target = Device;

        fn deref(&self) -> &Device {
            &self.device
        }
    }

    static CUDA_DEVICE: Mutex<()> = Mutex::new(());

    /// Skipped, not failed, without a device - the same gate the repository's other CUDA
    /// tests use, so `cargo test` runs off the training box.
    fn cuda() -> Option<CudaClaim> {
        tch::Cuda::is_available().then(|| CudaClaim {
            // A failing CUDA test poisons this mutex. The tests after it should report their
            // own result rather than a poison error about someone else's failure.
            _claim: CUDA_DEVICE.lock().unwrap_or_else(|error| error.into_inner()),
            device: Device::Cuda(0),
        })
    }

    #[test]
    fn fused_relu_square_is_bit_identical_including_gradients() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
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
        let Some(claim) = cuda() else { return };
        let device = *claim;
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
    /// backward through every kernel, overwrite the input buffers in place, replay, and
    /// require the replayed gradients to be what a fresh eager evaluation produces on the
    /// new bytes. A host synchronization inside a kernel, an allocation outside the
    /// capture's private pool, or a launch geometry that depended on anything but the
    /// arguments would fail the capture or produce a stale replay.
    ///
    /// ONE capture test for all three kernels, not one each: `cargo test` runs tests in
    /// parallel threads and two concurrent captures on one device fail each other, so the
    /// suite gets exactly one capture window.
    #[test]
    fn every_kernel_captures_and_replays_inside_a_cuda_graph() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        if !unsafe { torch_sys::at_cuda_graph_is_available() } {
            return;
        }
        let (heads, head_dim, length, batch) = (8i64, 64i64, 32i64, 2i64);
        let half = head_dim / 2;
        let width = heads * head_dim;
        let (cosine, sine) = rotation(length, half, device);
        let hidden = bf16_randn(&[batch * length, 512], device).set_requires_grad(true);
        let packed = bf16_randn(&[batch, length, 2 * width], device).set_requires_grad(true);
        let raw_packed = bf16_randn(&[batch, length, 2 * width], device).set_requires_grad(true);

        let step = || {
            let activation = relu_square(&hidden);
            let rotated = rope(&packed, &cosine, &sine, heads);
            let normed = qk_norm_rope(&raw_packed, &cosine, &sine, heads);
            let objective =
                activation.sum(Kind::Float) + rotated.sum(Kind::Float) + normed.sum(Kind::Float);
            let gradients = Tensor::run_backward(
                &[&objective],
                &[&hidden, &packed, &raw_packed],
                false,
                false,
            );
            (
                gradients[0].shallow_clone(),
                gradients[1].shallow_clone(),
                gradients[2].shallow_clone(),
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
        let fresh_raw = bf16_randn(&raw_packed.size(), device);
        tch::no_grad(|| {
            hidden.detach().copy_(&fresh_hidden);
            packed.detach().copy_(&fresh_packed);
            raw_packed.detach().copy_(&fresh_raw);
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
        let raw_leaf = fresh_raw.detach().copy().set_requires_grad(true);
        let eager = {
            let activation = reference::relu_square(&hidden_leaf);
            let rotated = reference::rope(&packed_leaf, &cosine, &sine, heads);
            let normed = reference::qk_norm_rope(&raw_leaf, &cosine, &sine, heads);
            let objective =
                activation.sum(Kind::Float) + rotated.sum(Kind::Float) + normed.sum(Kind::Float);
            let gradients = Tensor::run_backward(
                &[&objective],
                &[&hidden_leaf, &packed_leaf, &raw_leaf],
                false,
                false,
            );
            (
                gradients[0].shallow_clone(),
                gradients[1].shallow_clone(),
                gradients[2].shallow_clone(),
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
        assert!(
            identical(&captured.2, &eager.2),
            "the replayed qk_norm_rope gradient is not the composition's gradient of the new input"
        );
        assert_eq!(
            captured.3.double_value(&[]),
            eager.3.double_value(&[]),
            "the replayed objective does not match an eager evaluation on the same bytes"
        );
        unsafe { torch_sys::at_cuda_graph_free(graph) };
    }

    /// A packed input with the interesting values forced in rather than hoped for. The
    /// special tokens go along the LENGTH axis, not the batch axis, because the batch here
    /// is two or three rows and every one of them has to stay a normal sample.
    ///
    /// Token 0 is exactly zero, so `rstd` is `rsqrt(eps)` for every head block of it and a
    /// naive `1/rms` would divide by zero; token 1 is all NaN; token 2 is all negative; and
    /// token 3 carries a SINGLE NaN element, which has to poison exactly its own head block
    /// through the reduction and no other.
    fn qk_sample(shape: &[i64], device: Device) -> Tensor {
        let values = bf16_randn(shape, device);
        let _ = values.narrow(1, 0, 1).fill_(0.0);
        let _ = values.narrow(1, 1, 1).fill_(f64::NAN);
        let _ = values.narrow(1, 2, 1).fill_(-3.5);
        let _ = values.narrow(1, 3, 1).narrow(-1, 5, 1).fill_(f64::NAN);
        values
    }

    /// Forward and backward of the fused op against the composition, on the REAL call
    /// site's input: a `split_with_sizes` view of the `[.., 3·d_model]` QKV projection, so
    /// the kernel normalizes and rotates out of a strided buffer and the gradient has to
    /// land in the right columns of a wider one.
    #[test]
    fn fused_qk_norm_rope_is_bit_identical_including_gradients() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        let (heads, head_dim, length, batch) = (8i64, 64i64, 375i64, 3i64);
        let half = head_dim / 2;
        let width = heads * head_dim;
        let (cosine, sine) = rotation(length, half, device);

        let source = qk_sample(&[batch, length, 3 * width], device);
        let projection = source.detach().copy().set_requires_grad(true);
        let composed_projection = source.detach().copy().set_requires_grad(true);
        let packed = projection.split_with_sizes([2 * width, width], -1);
        let composed_packed = composed_projection.split_with_sizes([2 * width, width], -1);
        assert!(
            !packed[0].is_contiguous(),
            "the q‖k block should be a strided view of the projection"
        );

        let fused = qk_norm_rope(&packed[0], &cosine, &sine, heads);
        let composed = reference::qk_norm_rope(&composed_packed[0], &cosine, &sine, heads);
        assert_eq!(fused.size(), composed.size());
        assert!(
            identical(&fused, &composed),
            "fused qk_norm_rope forward differs from the composition, {} elements differ",
            differing(&fused, &composed)
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
            "fused qk_norm_rope backward differs from the composition, {} elements differ",
            differing(&fused_grad[0], &composed_grad[0])
        );
        // The V block is neither normalized nor rotated, so its gradient columns must be
        // untouched zeros - the op must not have written outside the q‖k half.
        let value_gradient = fused_grad[0].narrow(-1, 2 * width, width);
        assert_eq!(
            value_gradient.abs().sum(Kind::Float).double_value(&[]),
            0.0,
            "the fused op wrote into the value block's gradient columns"
        );
    }

    /// The composition and the fusion agree with the ALREADY-FUSED rotation too, which is
    /// the baseline this op actually replaces once the packed rotary has landed.
    #[test]
    fn fused_qk_norm_rope_matches_the_norm_plus_fused_rotary_pair() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        let (heads, head_dim, length, batch) = (8i64, 64i64, 375i64, 2i64);
        let (cosine, sine) = rotation(length, head_dim / 2, device);
        let packed = qk_sample(&[batch, length, 2 * heads * head_dim], device);
        assert!(identical(
            &qk_norm_rope(&packed, &cosine, &sine, heads),
            &reference::qk_norm_fused_rope(&packed, &cosine, &sine, heads)
        ));
    }

    /// The geometry the 128-bit path refuses. `head_dim = 12` gives `half = 6`, so the
    /// vector path cannot be taken and the single-thread emulation of ATen's reduction tree
    /// runs instead - while ATen itself still runs its VECTORIZED kernel, because it makes
    /// that choice on its own contiguous copy. Without this the emulated tree would be
    /// dead, unproven code the moment a head dimension changed.
    #[test]
    fn fused_qk_norm_rope_is_bit_identical_on_the_emulated_reduction_path() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        let (heads, head_dim, length, batch) = (3i64, 12i64, 17i64, 2i64);
        let (cosine, sine) = rotation(length, head_dim / 2, device);
        let packed = qk_sample(&[batch, length, 2 * heads * head_dim], device);
        let input = packed.detach().copy().set_requires_grad(true);
        let composed_input = packed.detach().copy().set_requires_grad(true);

        let fused = qk_norm_rope(&input, &cosine, &sine, heads);
        let composed = reference::qk_norm_rope(&composed_input, &cosine, &sine, heads);
        assert!(
            identical(&fused, &composed),
            "the emulated reduction path is not bit-identical, {} elements differ",
            differing(&fused, &composed)
        );

        let upstream = bf16_randn(&fused.size(), device);
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
            "the emulated reduction path's gradient is not bit-identical, {} elements differ",
            differing(&fused_grad[0], &composed_grad[0])
        );
    }

    /// [`QK_NORM_ROUNDING`] is the MEASURED contraction form, and this is the measurement:
    /// all eight forms run against the composition, and the set that reproduces it bit for
    /// bit is asserted whole. It is `{2, 3}`, a pair and not a singleton, because bit 0 is
    /// inert - the value it accumulates is a product of two bf16 mantissas and is therefore
    /// exact in fp32, so contracting that multiply-add changes nothing. Bits 1 and 2 are not
    /// inert and only one of their four combinations is ATen's.
    ///
    /// The forms differ only in whether an fp32 multiply-accumulate contracts into an `fma`,
    /// which is invisible in any tolerance-based comparison and moves roughly one output
    /// element in 10^5 - about 1500 per layer at the training shape. If a toolchain change
    /// ever moves the answer, this fails with the new set in the message instead of the
    /// kernel quietly drifting off the reference.
    #[test]
    fn qk_norm_rope_rounding_is_the_measured_pair() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        let (heads, head_dim, length, batch) = (8i64, 64i64, 375i64, 2i64);
        let (cosine, sine) = rotation(length, head_dim / 2, device);
        let width = heads * head_dim;
        let sample = qk_sample(&[batch, length, 2 * width], device);
        let upstream = bf16_randn(&[batch, length, 2, heads, head_dim], device);

        let composed_input = sample.detach().copy().set_requires_grad(true);
        let composed = reference::qk_norm_rope(&composed_input, &cosine, &sine, heads);
        let composed_grad = Tensor::run_backward(
            &[(&composed * &upstream).sum(Kind::Float)],
            &[&composed_input],
            false,
            false,
        );

        let mut matching = Vec::new();
        let mut diagnostic = String::new();
        for rounding in 0..8 {
            let input = sample.detach().copy().set_requires_grad(true);
            let fused = raw::qk_norm_rope(&input, &cosine, &sine, heads, rounding);
            let gradient = Tensor::run_backward(
                &[(&fused * &upstream).sum(Kind::Float)],
                &[&input],
                false,
                false,
            );
            let forward_ok = identical(&fused, &composed);
            let backward_ok = identical(&gradient[0], &composed_grad[0]);
            if forward_ok && backward_ok {
                matching.push(rounding);
            }
            diagnostic.push_str(&format!(
                "\n  rounding {rounding}: forward {forward_ok} ({} elements differ), backward {backward_ok} ({} elements differ)",
                differing(&fused, &composed),
                differing(&gradient[0], &composed_grad[0])
            ));
        }
        assert_eq!(
            matching,
            vec![QK_NORM_ROUNDING, QK_NORM_ROUNDING | 1],
            "exactly the measured contraction pair should reproduce ATen's reduction:{diagnostic}"
        );
    }

    /// ATen's `rstd` is the reduction TREE the kernel reproduces, proven without the kernel:
    /// squares summed four at a time in order, then halved pairwise - `p[i] + p[i+8]`, then
    /// `+4`, `+2`, `+1`, which is `WARP_SHFL_DOWN` at offsets 16..1 with the top level
    /// landing on structural zeros - then divided by `head_dim` and `rsqrt`-ed.
    ///
    /// This is the claim the kernel's whole addressing scheme is built on, and it is worth
    /// pinning separately from the kernel: a naive `sum(-1)` reference would disagree with
    /// ATen here, and a test that only compared kernel against composition could not say
    /// whether it was the tree or the arithmetic that was wrong.
    #[test]
    fn atens_rms_statistic_is_the_reduction_tree_the_kernel_reproduces() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        let (rows, head_dim) = (4096i64, 64i64);
        let values = bf16_randn(&[rows, head_dim], device);
        let statistic = values
            .internal_fused_rms_norm([head_dim], None::<&Tensor>, Some(NORM_EPS))
            .1
            .reshape(-1);

        // bf16 widens to fp32 losslessly and a product of two 8-bit mantissas is exact, so
        // `square()` here is ATen's `val * val` to the bit.
        let chunks = values
            .to_kind(Kind::Float)
            .square()
            .reshape([rows, head_dim / 4, 4]);
        // `((a+b)+c)+d`, not `(a+b)+(c+d)`: the partial is accumulated one element at a
        // time from zero.
        let mut tree = &(&chunks.select(2, 0) + &chunks.select(2, 1)) + &chunks.select(2, 2);
        tree = &tree + &chunks.select(2, 3);
        let mut width = head_dim / 4;
        while width > 1 {
            width /= 2;
            tree = &tree.narrow(1, 0, width) + &tree.narrow(1, width, width);
        }
        let expected = (&tree.reshape(-1) / head_dim as f64 + NORM_EPS).rsqrt();
        assert!(
            identical(&statistic, &expected),
            "ATen's rstd is not the four-element-partial binary tree the kernel implements, max |delta| {}",
            max_absolute(&statistic, &expected)
        );
    }
}
