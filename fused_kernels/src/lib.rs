//! Fused bf16 kernels for the CausalPatch backbone, and the composed-ATen references they
//! are bit-identical to.
//!
//! Four fusions, all chosen by measurement rather than taste:
//!
//! * [`relu_square`] replaces `x.relu().square()`. The composition makes four full passes
//!   over the `[tokens, ffn]` hidden activation in forward and six in backward; this makes
//!   two and three. At `tokens = 96_000`, `ffn = 2048` that is 15.73 GB/step of traffic
//!   removed over eight layers, 83% of the recipe change's whole traffic delta.
//! * [`rope`] replaces the packed `q‖k` rotation, which ATen has no operator for at all:
//!   the composition is two full-width products plus a half-crossing sum, seven passes
//!   over the `[tokens, 2·d_model]` block in forward, against this kernel's two.
//! * [`qk_norm_rope`] folds the per-head RMS normalization into that rotation, so the
//!   normalized block and its `rstd` are never materialized at all.
//! * [`loss_geometry`] replaces the candle geometry and the Gaussian NLL: forty-eight
//!   elementwise ops over the 147 M-element head space, 65 kernels forward and 70 backward,
//!   become one kernel each side. The twelve `dot`s are deliberately NOT fused - see the
//!   function - so the objective keeps its bits while 3.8x of its traffic disappears.
//!
//! All are differentiable from Rust with no further plumbing: the returned tensor carries
//! a real `grad_fn`, so [`tch::Tensor::backward`] and [`tch::Tensor::run_backward`] drive
//! the fused backward kernels. See `csrc/bridge.cpp` for the mechanism and why it is sound.
//!
//! All are CUDA-graph-capturable: no host synchronization, no device-to-host read, and a
//! grid that is a pure function of the launch arguments. The only allocations are the
//! outputs, taken from the caching allocator exactly like any ATen op's, which is what a
//! capture's private mempool is for.

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
    fn fk_relu_square_backward_raw(input: *const C_tensor, grad: *const C_tensor) -> *mut C_tensor;
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
    fn fk_loss_geometry(
        head: *const C_tensor,
        targets: *const C_tensor,
        weighted_mask: *const C_tensor,
        mask: *const C_tensor,
        sigma: *const C_tensor,
        range: *const C_tensor,
        horizon_scale: *const C_tensor,
        inverse_horizon: *const C_tensor,
        log_scale_gain: *const C_tensor,
        cap: f64,
        ln2: f64,
        rounding: i64,
        decoupled: i64,
        outputs: *mut *mut C_tensor,
    ) -> i32;
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
        panic!("fused {operation} failed: {}", last_error());
    }
    unsafe { Tensor::from_ptr(raw) }
}

/// The bridge's thread-local failure message, cleared by the next call on this thread.
fn last_error() -> String {
    let message = unsafe { fk_last_error() };
    if message.is_null() {
        "no message".to_owned()
    } else {
        unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned()
    }
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

/// The candle channel count the fused loss is defined on: one close, one range and two
/// positions inside it, each with its own log predictive scale.
pub const LOSS_CHANNELS: i64 = 4;

/// The fp32 forms ATen's own build emitted for the loss chain, as a five-bit selector.
///
/// MEASURED, not chosen, exactly like [`QK_NORM_ROUNDING`], and for the same reason: each
/// field is a place where the composition's last bit depends on how a kernel happened to be
/// written rather than on the mathematics. The value below is `2`, and what that means field
/// by field is the measurement `loss_geometry_rounding_is_the_measured_form` performs:
///
/// * bit 0, CLEAR - `div` by the host scalar `ln 2` is a multiply by its fp32 RECIPROCAL.
///   ATen's CUDA `div_true` special-cases a CPU scalar divisor and removes the operand, so
///   `x·(1/ln2)` is the reference and `x/ln2` is a different tensor. Note this is the
///   opposite of the CPU build, where the same expression is a true division - which is why
///   an off-device probe cannot answer the question and this sweep can.
/// * bit 1, SET - `tanh_backward`'s `1 - y·y` arrives CONTRACTED into one `fma`.
/// * bit 2, CLEAR - `sigmoid_backward` is `(g·(1-y))·y`, not `(g·y)·(1-y)`.
/// * bits 3-4, ZERO - `softplus_backward` is `(g·z)/(z+1)`; `1` selects `g·(z/(z+1))` and
///   `2` selects `(g/(z+1))·z`, and `3` is the same form as `0`.
///
/// Each field moves the last bit of a gradient over 147 M elements, and each was previously
/// a guess in this repository. The sweep is the artefact: any future fused kernel over this
/// chain can read the four answers off this constant instead of re-deriving them.
pub const LOSS_GEOMETRY_ROUNDING: i64 = 2;

/// What the fused candle geometry and Gaussian NLL hands back.
///
/// `terms` and `squares` are the REDUCED scalars - the twelve `dot`s are still ATen's, so
/// their summation trees are the reference's - and `close` is the full
/// `[rows, origins, 1, pred_len]` σ-scaled mean coordinate, the one full-size intermediate
/// anything outside the loss has a use for.
pub struct LossGeometry {
    /// `[rows, origins, 1, pred_len]` fp32: `coordinate_0 · horizon_scale`, differentiable.
    pub close: Tensor,
    /// `[2·LOSS_CHANNELS]` fp32 under the full coupling: per channel `½·dot(square, weight)`
    /// then `cap·dot(log_scale, weighted_mask)`, in that order. Their sum is the objective's
    /// unnormalized numerator. `[3·LOSS_CHANNELS]` under
    /// [`decoupled_loss_geometry`], which splits the quadratic in two - see its reference.
    pub terms: Tensor,
    /// `[LOSS_CHANNELS]` fp32, gradient-free: `dot(square, mask)`, the diagnostic MSE's
    /// numerator per channel.
    pub squares: Tensor,
    /// `[2·LOSS_CHANNELS]` fp32, gradient-free, `Some` ONLY when `terms` is not itself the
    /// NLL's numerator: the decoupled `terms` double-count the squared error under two
    /// different weightings, so their VALUE is not a likelihood. These are the full
    /// coupling's eight terms, detached, so the reported NLL stays the same number every
    /// other arm reports.
    pub nll_terms: Option<Tensor>,
}

/// The candle geometry and the masked Gaussian NLL element chain in ONE pass, with a
/// one-pass backward.
///
/// `head` is the dense `[rows, origins, 2·LOSS_CHANNELS, pred_len]` bf16 head output -
/// four candle coordinates then four log scales - `targets` the fp32
/// `[rows, origins, LOSS_CHANNELS, pred_len]` σ-scaled log returns, and `mask` /
/// `weighted_mask` the fp32 `[rows, origins, 1, pred_len]` validity flags, raw and folded
/// with the objective's per-horizon weight. `sigma` and `range` are the per-origin
/// statistics, already clamped, and `horizon_scale` / `inverse_horizon` / `log_scale_gain`
/// the resident per-horizon buffers.
///
/// WHAT IS FUSED AND WHAT IS NOT. The forty-eight elementwise ops of the composition - the
/// bf16 widening, the softplus, three `log1p`s, two sigmoids, four `tanh`s, four `exp`s, the
/// mask folds, the residuals and their squares - become one kernel. The twelve `dot`s do
/// NOT: a reduction's summation tree is cuBLAS's and cannot be reproduced, so the kernel
/// materializes exactly the twelve fp32 vectors they consume and ATen reduces them, which is
/// what keeps the loss VALUE bit-identical rather than merely close. The backward is one
/// kernel too: it recomputes the chain from `head` instead of reading the twelve retained
/// tensors plus the geometry's five, so the whole chain retains nothing beyond its inputs.
///
/// Bit-identical to [`reference::loss_geometry`] in both directions when nothing consumes
/// `close`. When something does, its gradient is a THIRD addend on the mean coordinate and
/// this kernel adds it after the two the loss itself contributes; the composition's autograd
/// engine would order it by node creation instead, so that one channel agrees to a rounding
/// rather than to the bit. Nothing in the incumbent objective takes that path.
///
/// Off CUDA this IS [`reference::loss_geometry`], for the reason given on [`relu_square`].
#[allow(clippy::too_many_arguments)]
pub fn loss_geometry(
    head: &Tensor,
    targets: &Tensor,
    weighted_mask: &Tensor,
    mask: &Tensor,
    sigma: &Tensor,
    range: &Tensor,
    horizon_scale: &Tensor,
    inverse_horizon: &Tensor,
    log_scale_gain: &Tensor,
    cap: f64,
) -> LossGeometry {
    loss_geometry_with_rounding(
        head,
        targets,
        weighted_mask,
        mask,
        sigma,
        range,
        horizon_scale,
        inverse_horizon,
        log_scale_gain,
        cap,
        LOSS_GEOMETRY_ROUNDING,
    )
}
/// The decoupled mean/scale objective, fused with the same candle geometry as
/// [`loss_geometry`]. Operands and output layout match
/// [`reference::decoupled_loss_geometry`]: per channel, `terms` contains the fixed-precision
/// mean quadratic, the detached-residual scale quadratic, then the log-scale term.
///
/// `nll_terms` contains the detached true Gaussian NLL terms, NOT the sum of the two
/// quadratics. A caller reporting the NLL while training the split objective must preserve
/// its surrogate-gradient construction. The mean gradient never uses the learned scale;
/// the scale gradient is unchanged from the coupled NLL. `close` remains differentiable.
///
/// CUDA shares the existing geometry and backward implementation. One extra workspace row
/// holds the horizon-fixed precision for the mean's four ATen `dot`s; the true NLL reuses
/// the already-reduced scale/log terms. No activation copy or host read is needed. The
/// composed reference remains the off-CUDA implementation, as for [`loss_geometry`].
#[allow(clippy::too_many_arguments)]
pub fn decoupled_loss_geometry(
    head: &Tensor,
    targets: &Tensor,
    weighted_mask: &Tensor,
    mask: &Tensor,
    sigma: &Tensor,
    range: &Tensor,
    horizon_scale: &Tensor,
    inverse_horizon: &Tensor,
    log_scale_gain: &Tensor,
    cap: f64,
) -> LossGeometry {
    loss_geometry_impl(
        head,
        targets,
        weighted_mask,
        mask,
        sigma,
        range,
        horizon_scale,
        inverse_horizon,
        log_scale_gain,
        cap,
        LOSS_GEOMETRY_ROUNDING,
        true,
    )
}

/// [`loss_geometry`] with the rounding selector open, so the discovery test can prove which
/// of the thirty-two forms ATen's build emitted and that the others disagree. The model path
/// calls [`loss_geometry`].
#[allow(clippy::too_many_arguments)]
pub fn loss_geometry_with_rounding(
    head: &Tensor,
    targets: &Tensor,
    weighted_mask: &Tensor,
    mask: &Tensor,
    sigma: &Tensor,
    range: &Tensor,
    horizon_scale: &Tensor,
    inverse_horizon: &Tensor,
    log_scale_gain: &Tensor,
    cap: f64,
    rounding: i64,
) -> LossGeometry {
    loss_geometry_impl(
        head,
        targets,
        weighted_mask,
        mask,
        sigma,
        range,
        horizon_scale,
        inverse_horizon,
        log_scale_gain,
        cap,
        rounding,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn loss_geometry_impl(
    head: &Tensor,
    targets: &Tensor,
    weighted_mask: &Tensor,
    mask: &Tensor,
    sigma: &Tensor,
    range: &Tensor,
    horizon_scale: &Tensor,
    inverse_horizon: &Tensor,
    log_scale_gain: &Tensor,
    cap: f64,
    rounding: i64,
    decoupled: bool,
) -> LossGeometry {
    if !head.device().is_cuda() {
        let composed = if decoupled {
            reference::decoupled_loss_geometry
        } else {
            reference::loss_geometry
        };
        return composed(
            head,
            targets,
            weighted_mask,
            mask,
            sigma,
            range,
            horizon_scale,
            inverse_horizon,
            log_scale_gain,
            cap,
        );
    }
    let mut outputs = [std::ptr::null_mut::<C_tensor>(); 4];
    let status = unsafe {
        fk_loss_geometry(
            head.as_ptr(),
            targets.as_ptr(),
            weighted_mask.as_ptr(),
            mask.as_ptr(),
            sigma.as_ptr(),
            range.as_ptr(),
            horizon_scale.as_ptr(),
            inverse_horizon.as_ptr(),
            log_scale_gain.as_ptr(),
            cap,
            std::f64::consts::LN_2,
            rounding,
            i64::from(decoupled),
            outputs.as_mut_ptr(),
        )
    };
    if status != 0 {
        panic!("fused loss_geometry failed: {}", last_error());
    }
    let [close, terms, squares, nll_terms] = outputs;
    LossGeometry {
        close: unsafe { Tensor::from_ptr(close) },
        terms: unsafe { Tensor::from_ptr(terms) },
        squares: unsafe { Tensor::from_ptr(squares) },
        nll_terms: decoupled.then(|| unsafe { Tensor::from_ptr(nll_terms) }),
    }
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
        finish, fk_qk_norm_rope, fk_qk_norm_rope_backward_raw, fk_relu_square_backward_raw,
        fk_rope_backward_raw, Tensor,
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

    /// The composed candle geometry and Gaussian NLL, transcribed op for op from what
    /// `CausalPatchModel::losses` ran before the fusion landed.
    ///
    /// Every line here is one full-size fp32 kernel and every one of them is what the fused
    /// kernel deletes. It is also the numerical definition of the objective: the ORDER of
    /// the twelve `dot`s, the `·½` and `·cap` scalars on the terms, the `no_grad` on the
    /// diagnostic squares and the fact that the mask fold happens once at mask width are
    /// all load-bearing, and a reference that paraphrased any of them would pin nothing.
    ///
    /// `close` is returned as well as consumed, which the pre-fusion form did not do - the
    /// amplitude prior reduces it. Handing back the same tensor the loss already built
    /// costs nothing and keeps the two definitions from drifting.
    #[allow(clippy::too_many_arguments)]
    pub fn loss_geometry(
        head: &Tensor,
        targets: &Tensor,
        weighted_mask: &Tensor,
        mask: &Tensor,
        sigma: &Tensor,
        range: &Tensor,
        horizon_scale: &Tensor,
        inverse_horizon: &Tensor,
        log_scale_gain: &Tensor,
        cap: f64,
    ) -> super::LossGeometry {
        let (channels, close, predictions) = candles(head, sigma, range, horizon_scale);
        let last = super::LOSS_CHANNELS as usize;
        let flat = |tensor: &Tensor| tensor.reshape([-1]);
        let mask_flat = flat(mask);
        let weighted_flat = flat(weighted_mask);
        let precision = weighted_mask * inverse_horizon;
        let mut terms = Vec::with_capacity(2 * last);
        let mut squares = Vec::with_capacity(last);
        for (channel, prediction) in predictions.into_iter().enumerate() {
            let scale = (&channels[last + channel] * log_scale_gain).tanh();
            let weight = (&scale * (-2.0 * cap)).exp() * &precision;
            let square = (targets.narrow(2, channel as i64, 1) - prediction).square();
            terms.push(flat(&square).dot(&flat(&weight)) * 0.5);
            terms.push(flat(&scale).dot(&weighted_flat) * cap);
            squares.push(tch::no_grad(|| flat(&square).dot(&mask_flat)));
        }
        super::LossGeometry {
            close,
            terms: Tensor::stack(&terms, 0),
            squares: Tensor::stack(&squares, 0),
            nll_terms: None,
        }
    }

    /// The four decoded candle coordinates in channel order and the σ-scaled close they are
    /// all offsets from, lifted out of [`loss_geometry`] verbatim.
    ///
    /// It is lifted rather than copied because [`decoupled_loss_geometry`] has to reproduce
    /// the full coupling's NLL VALUE on the bits, and it can only do that if the residuals
    /// entering both compositions are the same bits. A second transcription of five
    /// elementwise ops would be a second place for that to stop being true.
    fn candles(
        head: &Tensor,
        sigma: &Tensor,
        range: &Tensor,
        horizon_scale: &Tensor,
    ) -> (Vec<Tensor>, Tensor, [Tensor; 4]) {
        let channels = head.split(1, 2);
        let coordinate = |index: usize| channels[index].to_kind(tch::Kind::Float);
        let close = &channels[0] * horizon_scale;
        let relative_range = coordinate(1).softplus() * range / std::f64::consts::LN_2;
        let low = &close - (coordinate(2).sigmoid() * &relative_range).log1p() / sigma;
        let high = &low + relative_range.log1p() / sigma;
        let open = &low + (coordinate(3).sigmoid() * &relative_range).log1p() / sigma;
        let shared = close.shallow_clone();
        (channels, close, [open, high, low, shared])
    }

    /// The same geometry with the squared-error term SPLIT, so the mean and the log scale
    /// each receive the gradient they should instead of one term serving both.
    ///
    /// The defect this removes is in the Gaussian NLL itself, not in the implementation of
    /// it. With `u = cap·scale + ½·ln h` the quadratic is `½·r²·exp(-2·u)`, so the gradient
    /// reaching the MEAN is weighted by the model's own predicted precision, with no detach
    /// anywhere. On an origin the trunk has memorized the residual shrinks, the head answers
    /// with a smaller scale, and the mean's gradient weight RISES - a super-linear reward for
    /// memorization, largest exactly where memorization is cheapest (long horizons, whose
    /// overlapping windows are almost the same window). It is also why two checkpoints whose
    /// `h = 1` IC differed by 2.3x scored the same held-out NLL to four decimals: the head
    /// can trade mean accuracy against scale accuracy at constant likelihood.
    ///
    /// The split, per channel:
    ///
    /// ```text
    /// mean  : ½·dot(r²,         w·m·(1/h))                 gradient -> mean only
    /// scale : ½·dot(detach(r²), w·m·(1/h)·exp(-2·cap·s))   gradient -> scale only
    /// log   : cap·dot(s, w·m)                              gradient -> scale, unchanged
    /// ```
    ///
    /// Three properties make this the right split rather than a convenient one. The
    /// stationary point in `s` is IDENTICAL to the original NLL's - detaching `r²` removes no
    /// `s`-dependence, so `∂/∂s` of the last two terms is the original's exactly, and the
    /// fitted scale still means what it meant. The mean's per-horizon weight is `1/h`, which
    /// is precisely the horizon-fixed factor the original already carried once `exp(-2·ls)`
    /// was factored as `exp(-2·cap·tanh)·(1/h)`, so no horizon reweighting is smuggled in and
    /// `--horizon-loss` remains the only knob on that axis. And the mean now sees plain
    /// weighted MSE, which is what a conditional-mean estimator is supposed to minimize.
    ///
    /// This is deliberately NOT Seitzer et al. 2022's β-NLL (arXiv:2203.09168). Their
    /// multiplicative `detach(σ^{2β})·NLL` also rescales the LOG term: at β = 1 with the
    /// `½·ln h` prior hoisted out, that factor is `detach(exp(2·cap·s))`, spanning `e^-8` to
    /// `e^+8` element by element, and using the unfactored `σ²` instead makes it `h`, up to
    /// 192x at the far end. Either one reweights the scale objective far harder than it
    /// repairs the mean's. The additive split reweights nothing.
    ///
    /// What it costs: `terms` is no longer a likelihood. Its two quadratic rows count the
    /// same squared error under two different weightings, so the training loss VALUE is not
    /// the NLL and would not be comparable to any other arm's. `nll_terms` is therefore
    /// returned as well - the full coupling's own eight terms, detached - and it is bit-equal
    /// to what [`loss_geometry`] would have produced on the same inputs, because
    /// `detach(r²)·weight` and `r²·weight` are the same numbers reduced by the same `dot`.
    #[allow(clippy::too_many_arguments)]
    pub fn decoupled_loss_geometry(
        head: &Tensor,
        targets: &Tensor,
        weighted_mask: &Tensor,
        mask: &Tensor,
        sigma: &Tensor,
        range: &Tensor,
        horizon_scale: &Tensor,
        inverse_horizon: &Tensor,
        log_scale_gain: &Tensor,
        cap: f64,
    ) -> super::LossGeometry {
        let (channels, close, predictions) = candles(head, sigma, range, horizon_scale);
        let last = super::LOSS_CHANNELS as usize;
        let flat = |tensor: &Tensor| tensor.reshape([-1]);
        let mask_flat = flat(mask);
        let weighted_flat = flat(weighted_mask);
        let precision = weighted_mask * inverse_horizon;
        let precision_flat = flat(&precision);
        let mut terms = Vec::with_capacity(3 * last);
        let mut nll_terms = Vec::with_capacity(2 * last);
        let mut squares = Vec::with_capacity(last);
        for (channel, prediction) in predictions.into_iter().enumerate() {
            let scale = (&channels[last + channel] * log_scale_gain).tanh();
            let weight = (&scale * (-2.0 * cap)).exp() * &precision;
            let square = flat(&(targets.narrow(2, channel as i64, 1) - prediction).square());
            // The one line the whole change is. `detach` on the residual, not on the weight:
            // detaching the weight would leave the mean's gradient scaled by the predicted
            // precision's VALUE, which is the coupling, merely frozen for one step.
            let coupled = square.detach().dot(&flat(&weight)) * 0.5;
            let logarithmic = flat(&scale).dot(&weighted_flat) * cap;
            nll_terms.push(coupled.detach());
            nll_terms.push(logarithmic.detach());
            terms.push(square.dot(&precision_flat) * 0.5);
            terms.push(coupled);
            terms.push(logarithmic);
            squares.push(tch::no_grad(|| square.dot(&mask_flat)));
        }
        super::LossGeometry {
            close,
            terms: Tensor::stack(&terms, 0),
            squares: Tensor::stack(&squares, 0),
            nll_terms: Some(Tensor::stack(&nll_terms, 0)),
        }
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
        let defaulted = values
            .internal_fused_rms_norm([64], None::<&Tensor>, None)
            .0;
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
            _claim: CUDA_DEVICE
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
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
    /// ONE serialized test for all kernels and both loss modes: independent concurrent
    /// captures on one device would invalidate each other.
    #[test]
    fn every_kernel_captures_and_replays_inside_a_cuda_graph() {
        let Some(claim) = cuda() else { return };
        for decoupled in [false, true] {
            captured_kernel_case(*claim, decoupled);
        }
    }

    fn captured_kernel_case(device: Device, decoupled: bool) {
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
        // The loss chain allocates twelve reduction vectors and runs twelve `dot`s inside the
        // window, so it is the one op here whose capturability is not obvious.
        tch::manual_seed(37);
        let loss = loss_inputs(batch, 96, 16, device, 2.0);

        let step = || {
            let activation = relu_square(&hidden);
            let rotated = rope(&packed, &cosine, &sine, heads);
            let normed = qk_norm_rope(&raw_packed, &cosine, &sine, heads);
            let geometry = loss_geometry_impl(
                &loss.head,
                &loss.targets,
                &loss.weighted_mask,
                &loss.mask,
                &loss.sigma,
                &loss.range,
                &loss.horizon_scale,
                &loss.inverse_horizon,
                &loss.log_scale_gain,
                LOSS_CAP,
                LOSS_GEOMETRY_ROUNDING,
                decoupled,
            );
            let objective = activation.sum(Kind::Float)
                + rotated.sum(Kind::Float)
                + normed.sum(Kind::Float)
                + geometry.terms.sum(Kind::Float)
                + geometry.squares.sum(Kind::Float)
                + geometry.close.sum(Kind::Float);
            let gradients = Tensor::run_backward(
                &[&objective],
                &[&hidden, &packed, &raw_packed, &loss.head],
                false,
                false,
            );
            (
                gradients[0].shallow_clone(),
                gradients[1].shallow_clone(),
                gradients[2].shallow_clone(),
                gradients[3].shallow_clone(),
                objective,
                geometry.nll_terms,
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
        let fresh_head = bf16_randn(&loss.head.size(), device);
        tch::no_grad(|| {
            hidden.detach().copy_(&fresh_hidden);
            packed.detach().copy_(&fresh_packed);
            raw_packed.detach().copy_(&fresh_raw);
            loss.head.detach().copy_(&fresh_head);
            if decoupled {
                let _ = loss.log_scale_gain.shallow_clone().fill_(0.5);
            }
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
        let head_leaf = fresh_head.detach().copy().set_requires_grad(true);
        let eager = {
            let activation = reference::relu_square(&hidden_leaf);
            let rotated = reference::rope(&packed_leaf, &cosine, &sine, heads);
            let normed = reference::qk_norm_rope(&raw_leaf, &cosine, &sine, heads);
            let reference_loss = if decoupled {
                reference::decoupled_loss_geometry
            } else {
                reference::loss_geometry
            };
            let geometry = reference_loss(
                &head_leaf,
                &loss.targets,
                &loss.weighted_mask,
                &loss.mask,
                &loss.sigma,
                &loss.range,
                &loss.horizon_scale,
                &loss.inverse_horizon,
                &loss.log_scale_gain,
                LOSS_CAP,
            );
            let objective = activation.sum(Kind::Float)
                + rotated.sum(Kind::Float)
                + normed.sum(Kind::Float)
                + geometry.terms.sum(Kind::Float)
                + geometry.squares.sum(Kind::Float)
                + geometry.close.sum(Kind::Float);
            let gradients = Tensor::run_backward(
                &[&objective],
                &[&hidden_leaf, &packed_leaf, &raw_leaf, &head_leaf],
                false,
                false,
            );
            (
                gradients[0].shallow_clone(),
                gradients[1].shallow_clone(),
                gradients[2].shallow_clone(),
                gradients[3].shallow_clone(),
                objective,
                geometry.nll_terms,
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
        // The mean coordinate has an external consumer here, so this one channel is a
        // rounding rather than the bit - see
        // `fused_loss_geometry_carries_the_mean_coordinate_gradient`.
        for channel in 1..2 * LOSS_CHANNELS {
            assert!(
                identical(
                    &captured.3.narrow(2, channel, 1),
                    &eager.3.narrow(2, channel, 1)
                ),
                "the replayed loss gradient is not the composition's on channel {channel}"
            );
        }
        assert_eq!(
            captured.4.double_value(&[]),
            eager.4.double_value(&[]),
            "the replayed objective does not match an eager evaluation on the same bytes"
        );
        if decoupled {
            assert!(
                identical(captured.5.as_ref().unwrap(), eager.5.as_ref().unwrap()),
                "captured true NLL did not update with the head and scale-gain buffers"
            );
        }
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

    /// Everything `loss_geometry` takes, at one shape, with the ranges that actually reach
    /// every branch: the head is scaled so `softplus`'s `x > 20` cutover, the `tanh` cap's
    /// saturated ends and `log1p`'s near-`-1` region are all populated rather than hoped for.
    struct LossInputs {
        head: Tensor,
        targets: Tensor,
        weighted_mask: Tensor,
        mask: Tensor,
        sigma: Tensor,
        range: Tensor,
        horizon_scale: Tensor,
        inverse_horizon: Tensor,
        log_scale_gain: Tensor,
    }

    const LOSS_CAP: f64 = 4.0;

    fn loss_inputs(
        rows: i64,
        origins: i64,
        horizon: i64,
        device: Device,
        scale: f64,
    ) -> LossInputs {
        let float = (Kind::Float, device);
        let head = (Tensor::randn([rows, origins, 2 * LOSS_CHANNELS, horizon], float) * scale)
            .to_kind(Kind::BFloat16)
            .set_requires_grad(true);
        let mask = Tensor::rand([rows, origins, 1, horizon], float)
            .gt(0.25)
            .to_kind(Kind::Float);
        // Not the identity: a weight vector of ones would let a kernel that dropped the fold
        // entirely pass. Mean 1 over the axis is the model's own normalization.
        let weight = Tensor::rand([1, 1, 1, horizon], float) + 0.5;
        let index = Tensor::arange(horizon, float) + 1.0;
        LossInputs {
            head,
            targets: Tensor::randn([rows, origins, LOSS_CHANNELS, horizon], float),
            weighted_mask: &mask * &weight,
            mask,
            sigma: Tensor::rand([rows, origins, 1, 1], float) + 0.25,
            range: Tensor::rand([rows, origins, 1, 1], float) + 0.05,
            horizon_scale: index.sqrt().reshape([1, 1, 1, horizon]),
            inverse_horizon: index.reciprocal().reshape([1, 1, 1, horizon]),
            log_scale_gain: Tensor::full([1, 1, 1, 1], 1.0 / LOSS_CAP, float),
        }
    }

    /// The fused chain and the composition it replaces, driven by the SAME upstream gradient
    /// so the comparison is of the two kernels and not of two objectives. `close_upstream`
    /// present is the amplitude prior's path: a third addend on the mean coordinate.
    fn loss_pair(
        inputs: &LossInputs,
        upstream: &Tensor,
        close_upstream: Option<&Tensor>,
        rounding: i64,
    ) -> (LossGeometry, Tensor, LossGeometry, Tensor) {
        loss_pair_mode(inputs, upstream, close_upstream, rounding, LOSS_CAP, false)
    }

    fn loss_pair_mode(
        inputs: &LossInputs,
        upstream: &Tensor,
        close_upstream: Option<&Tensor>,
        rounding: i64,
        cap: f64,
        decoupled: bool,
    ) -> (LossGeometry, Tensor, LossGeometry, Tensor) {
        let run = |geometry: LossGeometry| {
            let mut objective = geometry.terms.dot(upstream);
            if let Some(weight) = close_upstream {
                objective = objective + (&geometry.close * weight).sum(Kind::Float);
            }
            let gradient =
                Tensor::run_backward(&[&objective], &[&inputs.head], false, false).remove(0);
            (geometry, gradient)
        };
        let reference = if decoupled {
            reference::decoupled_loss_geometry
        } else {
            reference::loss_geometry
        };
        let (composed, composed_gradient) = run(reference(
            &inputs.head,
            &inputs.targets,
            &inputs.weighted_mask,
            &inputs.mask,
            &inputs.sigma,
            &inputs.range,
            &inputs.horizon_scale,
            &inputs.inverse_horizon,
            &inputs.log_scale_gain,
            cap,
        ));
        let (fused, fused_gradient) = run(loss_geometry_impl(
            &inputs.head,
            &inputs.targets,
            &inputs.weighted_mask,
            &inputs.mask,
            &inputs.sigma,
            &inputs.range,
            &inputs.horizon_scale,
            &inputs.inverse_horizon,
            &inputs.log_scale_gain,
            cap,
            rounding,
            decoupled,
        ));
        (composed, composed_gradient, fused, fused_gradient)
    }

    /// Differing elements across the whole comparison: the mean coordinate, the eight
    /// objective terms, the four diagnostic squares and the dense head gradient. Every one
    /// is `differing`, which counts a sign flip on an exact zero as a difference - see
    /// `the_masked_signed_zero_is_the_compositions_and_reaches_a_parameter_gradient_intact`
    /// for why that bar is meetable here and what it took to meet it.
    fn loss_differences(
        composed: &LossGeometry,
        composed_gradient: &Tensor,
        fused: &LossGeometry,
        fused_gradient: &Tensor,
    ) -> [i64; 4] {
        [
            differing(&composed.close, &fused.close),
            differing(&composed.terms, &fused.terms),
            differing(&composed.squares, &fused.squares),
            differing(composed_gradient, fused_gradient),
        ]
    }

    /// WHICH fp32 forms ATen's own build emitted, measured over all thirty-two.
    ///
    /// Five bits, four questions, and not one of them answerable from the mathematics: a
    /// CUDA `div` by a host scalar becomes a multiply by an fp32 reciprocal, `nvcc` contracts
    /// `1 - y·y` into an `fma` unless stopped, and `sigmoid_backward` and
    /// `softplus_backward` were each written with one association out of two and three. Every
    /// one of them moves the last bit of a gradient over 147 M elements. This is the same
    /// discovery `qk_norm_rope_rounding_is_the_measured_pair` performs, for the same reason:
    /// the alternative is a kernel that is "close" and a training curve that quietly is not
    /// the one the checkpoint was selected on.
    ///
    /// It also answers the question that decides whether this fusion can exist at all. The
    /// chain runs `expf`, `tanhf`, `log1pf` and `softplus` inside the kernel, compiled by THIS
    /// box's nvcc, against a libtorch built with a different CUDA toolkit. If libdevice's fp32
    /// transcendentals disagreed across those two, no selector would rescue it and every one
    /// of the thirty-two would differ in the millions.
    #[test]
    fn loss_geometry_rounding_is_the_measured_form() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        tch::manual_seed(19);
        let inputs = loss_inputs(4, 1000, 64, device, 3.0);
        let upstream = Tensor::randn([2 * LOSS_CHANNELS], (Kind::Float, device));
        let mut matching = Vec::new();
        let mut best = (i64::MAX, -1i64);
        for rounding in 0..32 {
            let (composed, composed_gradient, fused, fused_gradient) =
                loss_pair(&inputs, &upstream, None, rounding);
            let counts = loss_differences(&composed, &composed_gradient, &fused, &fused_gradient);
            let total: i64 = counts.iter().sum();
            println!(
                "rounding {rounding:2}: close {}, terms {}, squares {}, gradient {}",
                counts[0], counts[1], counts[2], counts[3]
            );
            if total < best.0 {
                best = (total, rounding);
            }
            if total == 0 {
                matching.push(rounding);
            }
        }
        assert!(
            !matching.is_empty(),
            "no rounding form reproduces the composition; the closest is {} with {} \
             differing elements, which at zero is a selector question and in the millions is \
             a libdevice mismatch between this nvcc and libtorch's",
            best.1,
            best.0
        );
        assert!(
            matching.contains(&LOSS_GEOMETRY_ROUNDING),
            "LOSS_GEOMETRY_ROUNDING is {LOSS_GEOMETRY_ROUNDING} but the forms that reproduce \
             the composition are {matching:?}"
        );
    }

    /// Bit-for-bit against the composition at the REAL shape, forward and backward.
    ///
    /// `rows = 256`, `origins = 375`, `pred_len = 192`: 96 000 scored origins, a 147 M-element
    /// head and an 18.4 M-element channel slice. The shape is the point. A chain of `log1p`s
    /// and `exp`s that agrees on 128 elements and disagrees on one in 10^6 would pass every
    /// toy test and still move a training curve, and the grid-stride loop, the `index / horizon`
    /// addressing and the twelve-row workspace are all only exercised at scale.
    #[test]
    fn fused_loss_geometry_is_bit_identical_at_the_production_shape() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        tch::manual_seed(23);
        let inputs = loss_inputs(256, 375, 192, device, 2.0);
        let upstream = Tensor::randn([2 * LOSS_CHANNELS], (Kind::Float, device));
        let (composed, composed_gradient, fused, fused_gradient) =
            loss_pair(&inputs, &upstream, None, LOSS_GEOMETRY_ROUNDING);
        let counts = loss_differences(&composed, &composed_gradient, &fused, &fused_gradient);
        println!(
            "production shape: {} elements scored, close {}, terms {}, squares {}, \
             gradient {} differing",
            inputs.head.numel(),
            counts[0],
            counts[1],
            counts[2],
            counts[3]
        );
        assert_eq!(
            counts,
            [0, 0, 0, 0],
            "the fused loss is not bit-identical at the production shape; \
             max |delta| close {}, terms {}, squares {}, gradient {}",
            max_absolute(&composed.close, &fused.close),
            max_absolute(&composed.terms, &fused.terms),
            max_absolute(&composed.squares, &fused.squares),
            max_absolute(&composed_gradient, &fused_gradient)
        );
        // The objective is a real number, not an artefact of the comparison.
        let objective = composed.terms.sum(Kind::Float).double_value(&[]);
        assert!(
            objective.is_finite() && objective.abs() > 1.0,
            "{objective}"
        );
    }

    /// The targets as the TRAINING PATH builds them, strides and all.
    ///
    /// This is the defect job 5448 found, and it is the more important lesson of the two the
    /// fusion taught. `CausalPatchModel::targets` builds its future windows with `unfold` and
    /// then does arithmetic on that view; TensorIterator allocates the result in its inputs'
    /// own permuted layout, so what reaches the loss is channel-INNERMOST - sizes
    /// `[rows, origins, 4, horizon]`, strides `[origins·4·horizon, 4·horizon, 1, 4]` - and
    /// NOT dense. The ATen composition accepted it silently because every pointwise op it
    /// called is strided. A kernel that assumed `channel · horizon` addressing read the wrong
    /// elements, and the version before this one asserted contiguity and aborted the run.
    ///
    /// Every other test in this module builds its targets with `Tensor::randn`, which is
    /// contiguous by construction - shape-correct and stride-wrong, which is exactly the
    /// class of test that passes green while the real path crashes. So this one reproduces
    /// the construction rather than the dimensions, and asserts the layout it got is really
    /// the strided one before it compares anything.
    #[test]
    fn the_real_paths_strided_targets_are_read_as_the_composition_reads_them() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        tch::manual_seed(31);
        let (rows, origins, horizon, patch) = (32i64, 375i64, 192i64, 16i64);
        let context = patch * origins;
        let float = (Kind::Float, device);
        // `CausalPatchModel::future_windows` verbatim: narrow off the first patch, then
        // unfold `pred_len` windows at the patch stride, on a channel-major series.
        let series = Tensor::randn([rows, context + horizon, LOSS_CHANNELS], float);
        let future = series
            .narrow(1, patch, context + horizon - patch)
            .unfold(1, horizon, patch);
        let log_close = Tensor::randn([rows, origins, 1, 1], float);
        let sigma = Tensor::rand([rows, origins, 1, 1], float) + 0.25;
        let drift = Tensor::randn([rows, origins, 1, horizon], float) * 0.01;
        let targets = (future - &log_close) / &sigma - &drift;
        assert_eq!(targets.size(), [rows, origins, LOSS_CHANNELS, horizon]);
        assert_eq!(
            targets.stride(),
            [
                origins * LOSS_CHANNELS * horizon,
                LOSS_CHANNELS * horizon,
                1,
                LOSS_CHANNELS
            ],
            "the real path's targets are channel-innermost; if this layout changed, the \
             kernel's stride arguments are being tested against the wrong thing"
        );
        assert!(
            !targets.is_contiguous(),
            "this test exists for the NON-contiguous case and the tensor it built is dense"
        );
        let inputs = LossInputs {
            targets,
            ..loss_inputs(rows, origins, horizon, device, 2.0)
        };
        let upstream = Tensor::randn([2 * LOSS_CHANNELS], float);
        let (composed, composed_gradient, fused, fused_gradient) =
            loss_pair(&inputs, &upstream, None, LOSS_GEOMETRY_ROUNDING);
        let counts = loss_differences(&composed, &composed_gradient, &fused, &fused_gradient);
        assert_eq!(
            counts,
            [0, 0, 0, 0],
            "the fused loss does not read the real path's strided targets the way the \
             composition does; max |delta| close {}, terms {}, squares {}, gradient {}",
            max_absolute(&composed.close, &fused.close),
            max_absolute(&composed.terms, &fused.terms),
            max_absolute(&composed.squares, &fused.squares),
            max_absolute(&composed_gradient, &fused_gradient)
        );
        // Reading strided is INTERPRETING the strides, not tolerating them: the same values
        // in a dense copy, with every other operand shared, must give the same bits. Without
        // this, a kernel that ignored the strides and read the buffer densely could still
        // pass above whenever the composition happened to be compared against itself.
        let dense = targets_dense(&inputs);
        assert!(dense.targets.is_contiguous() && dense.targets.equal(&inputs.targets));
        let dense_geometry = loss_geometry_with_rounding(
            &dense.head,
            &dense.targets,
            &dense.weighted_mask,
            &dense.mask,
            &dense.sigma,
            &dense.range,
            &dense.horizon_scale,
            &dense.inverse_horizon,
            &dense.log_scale_gain,
            LOSS_CAP,
            LOSS_GEOMETRY_ROUNDING,
        );
        assert!(
            identical(&dense_geometry.terms, &fused.terms)
                && identical(&dense_geometry.squares, &fused.squares)
                && identical(&dense_geometry.close, &fused.close),
            "the fused loss read the strided targets differently from a dense copy of the \
             same values; max |delta| terms {}, squares {}, close {}",
            max_absolute(&dense_geometry.terms, &fused.terms),
            max_absolute(&dense_geometry.squares, &fused.squares),
            max_absolute(&dense_geometry.close, &fused.close)
        );
    }

    /// The same inputs with a DENSE copy of the targets and every other operand shared,
    /// including the head leaf, so a comparison between the two is a comparison of layouts.
    fn targets_dense(inputs: &LossInputs) -> LossInputs {
        LossInputs {
            head: inputs.head.shallow_clone(),
            targets: inputs.targets.contiguous(),
            weighted_mask: inputs.weighted_mask.shallow_clone(),
            mask: inputs.mask.shallow_clone(),
            sigma: inputs.sigma.shallow_clone(),
            range: inputs.range.shallow_clone(),
            horizon_scale: inputs.horizon_scale.shallow_clone(),
            inverse_horizon: inputs.inverse_horizon.shallow_clone(),
            log_scale_gain: inputs.log_scale_gain.shallow_clone(),
        }
    }

    /// NaN, ±inf and the mask, which is where a fused loss goes wrong silently.
    ///
    /// Two claims. First, the chain PROPAGATES nonfinite values exactly as the composition
    /// does - a NaN coordinate must poison the same terms and the same gradient elements, not
    /// a superset or a subset. Second, a masked element contributes EXACTLY zero: the mask
    /// enters through the precision weight and through the `dot`s' second operand, so
    /// `0 · anything` is the contribution, and the test proves it by zeroing the mask
    /// everywhere and requiring literal zeros out of the reductions rather than small numbers.
    #[test]
    fn fused_loss_geometry_matches_the_composition_at_nonfinite_and_masked_elements() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        tch::manual_seed(29);
        let inputs = loss_inputs(3, 257, 37, device, 6.0);
        // Forced, not hoped for: a NaN and both infinities in the coordinates, in the log
        // scales and in the targets, at elements that are valid and at elements that are not.
        let poison = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        tch::no_grad(|| {
            for (slot, value) in poison.iter().enumerate() {
                for channel in 0..2 * LOSS_CHANNELS {
                    let _ = inputs
                        .head
                        .select(0, 0)
                        .select(0, slot as i64)
                        .select(0, channel)
                        .narrow(0, slot as i64, 1)
                        .fill_(*value);
                }
                let _ = inputs
                    .targets
                    .select(0, 1)
                    .select(0, slot as i64)
                    .select(0, 0)
                    .narrow(0, slot as i64, 1)
                    .fill_(*value);
            }
            // One whole origin invalid, so a poisoned element sits behind a zero mask too.
            let _ = inputs.mask.select(0, 0).select(0, 1).fill_(0.0);
            let _ = inputs.weighted_mask.select(0, 0).select(0, 1).fill_(0.0);
        });
        let upstream = Tensor::randn([2 * LOSS_CHANNELS], (Kind::Float, device));
        let (composed, composed_gradient, fused, fused_gradient) =
            loss_pair(&inputs, &upstream, None, LOSS_GEOMETRY_ROUNDING);
        assert_eq!(
            loss_differences(&composed, &composed_gradient, &fused, &fused_gradient),
            [0, 0, 0, 0],
            "the fused loss disagrees with the composition on nonfinite or masked elements"
        );
        assert!(
            composed.terms.isnan().any().int64_value(&[]) != 0,
            "the poisoned elements did not reach the objective, so this test proved nothing"
        );

        // Now a CLEAN batch with the mask everywhere zero: every reduction must be a literal
        // zero from both forms. Clean, because a nonfinite element behind a zero mask is NOT
        // zero and must not be - `NaN · 0` is `NaN` in the composition too, and pretending
        // otherwise is exactly the kind of "helpful" divergence this bar exists to forbid.
        tch::manual_seed(41);
        let mut blind = loss_inputs(3, 257, 37, device, 6.0);
        tch::no_grad(|| {
            let _ = blind.mask.fill_(0.0);
            let _ = blind.weighted_mask.fill_(0.0);
        });
        let (composed, composed_gradient, fused, fused_gradient) =
            loss_pair(&blind, &upstream, None, LOSS_GEOMETRY_ROUNDING);
        assert_eq!(
            loss_differences(&composed, &composed_gradient, &fused, &fused_gradient),
            [0, 0, 0, 0],
            "the fused loss disagrees with the composition under a fully masked batch"
        );
        for (name, reduced) in [
            ("composed terms", &composed.terms),
            ("fused terms", &fused.terms),
            ("composed squares", &composed.squares),
            ("fused squares", &fused.squares),
        ] {
            let squared = reduced.square().sum(Kind::Float).double_value(&[]);
            assert_eq!(
                squared, 0.0,
                "{name} reduce to {squared} under a zero mask, not exactly zero"
            );
        }
        // The gradient of a fully masked batch is zero too, in both forms, because every
        // path to the head runs through the precision weight or the folded mask.
        for (name, gradient) in [("composed", &composed_gradient), ("fused", &fused_gradient)] {
            let magnitude = gradient.to_kind(Kind::Float).abs().max().double_value(&[]);
            assert_eq!(
                magnitude, 0.0,
                "{name} head gradient is {magnitude} under a zero mask, not exactly zero"
            );
        }
    }

    /// The amplitude prior's path: something outside the loss reduces the mean coordinate, so
    /// its gradient is a THIRD addend on channel 0.
    ///
    /// Verified NUMERICALLY and not bit-exactly, deliberately and only here. The kernel adds
    /// the external gradient after the two the loss contributes; the composition's autograd
    /// engine orders a buffer's addends by node creation, which for a consumer of the RETURNED
    /// coordinate puts it first. Three fp32 addends do not reassociate exactly, and there is
    /// no incumbent bit pattern to preserve for a term that did not exist before the prior.
    /// Everything else - all eight channels under `grad_close = None`, and the forward - stays
    /// bit-exact, which is what the production-shape test pins.
    #[test]
    fn fused_loss_geometry_carries_the_mean_coordinate_gradient() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        tch::manual_seed(31);
        let inputs = loss_inputs(4, 512, 48, device, 2.0);
        let upstream = Tensor::randn([2 * LOSS_CHANNELS], (Kind::Float, device));
        let close_upstream =
            Tensor::randn(inputs.mask.size(), (Kind::Float, device)) * &inputs.mask;
        let (composed, composed_gradient, fused, fused_gradient) = loss_pair(
            &inputs,
            &upstream,
            Some(&close_upstream),
            LOSS_GEOMETRY_ROUNDING,
        );
        // The forward is untouched by the extra consumer, so it stays exact.
        assert_eq!(
            differing(&composed.close, &fused.close),
            0,
            "the mean coordinate itself is not bit-identical"
        );
        let scale = composed_gradient
            .to_kind(Kind::Float)
            .abs()
            .max()
            .double_value(&[]);
        assert!(scale > 0.0, "the comparison ran on a zero gradient");
        let error = max_absolute(&composed_gradient, &fused_gradient) / scale;
        println!("mean-coordinate gradient path: {error:.3e} relative");
        assert!(
            error <= 1e-2,
            "the mean coordinate's gradient is not a rounding of the composition's: {error} \
             relative, which is a wrong derivative and not an accumulation order"
        );
        // And the eight channels that do NOT take the external addend are still exact: a
        // wrong chain would not confine its error to channel 0.
        for channel in 1..2 * LOSS_CHANNELS {
            assert_eq!(
                differing(
                    &composed_gradient.narrow(2, channel, 1),
                    &fused_gradient.narrow(2, channel, 1)
                ),
                0,
                "channel {channel} moved when only the mean coordinate gained a consumer"
            );
        }
    }

    /// TEMPORARY diagnosis: which channel diverges, under which rounding bits, and the exact
    /// fp32 inputs of the worst element so the CUDA form can be identified off-device.
    #[test]
    fn diagnose_loss_geometry_gradient() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        tch::manual_seed(23);
        let inputs = loss_inputs(64, 375, 192, device, 2.0);
        let upstream = Tensor::randn([2 * LOSS_CHANNELS], (Kind::Float, device));
        for rounding in 0..32 {
            let (composed, composed_gradient, fused, fused_gradient) =
                loss_pair(&inputs, &upstream, None, rounding);
            let per_channel: Vec<i64> = (0..2 * LOSS_CHANNELS)
                .map(|channel| {
                    differing(
                        &composed_gradient.narrow(2, channel, 1),
                        &fused_gradient.narrow(2, channel, 1),
                    )
                })
                .collect();
            println!(
                "rounding {rounding:2}: close {} terms {} squares {} channels {per_channel:?}",
                differing(&composed.close, &fused.close),
                differing(&composed.terms, &fused.terms),
                differing(&composed.squares, &fused.squares),
            );
        }

        let (_, composed_gradient, _, fused_gradient) =
            loss_pair(&inputs, &upstream, None, LOSS_GEOMETRY_ROUNDING);
        let horizon = inputs.head.size()[3];
        let flatten = |tensor: &Tensor| tensor.reshape([-1]).to_kind(Kind::Float);
        for channel in 0..2 * LOSS_CHANNELS {
            let left = flatten(&composed_gradient.narrow(2, channel, 1));
            let right = flatten(&fused_gradient.narrow(2, channel, 1));
            let mismatch = left
                .ne_tensor(&right)
                .logical_or(&left.signbit().logical_xor(&right.signbit()));
            let count = mismatch
                .to_kind(Kind::Int64)
                .sum(Kind::Int64)
                .int64_value(&[]);
            if count == 0 {
                continue;
            }
            let zero = (left.abs() + right.abs()).eq(0.0);
            let masked = flatten(&inputs.mask).eq(0.0);
            let tally = |predicate: &Tensor| {
                mismatch
                    .logical_and(predicate)
                    .to_kind(Kind::Int64)
                    .sum(Kind::Int64)
                    .int64_value(&[])
            };
            println!(
                "channel {channel}: {count} mismatched, {} both zero, {} behind a zero mask, \
                 {} composed-negative, {} fused-negative",
                tally(&zero),
                tally(&masked),
                tally(&left.signbit()),
                tally(&right.signbit()),
            );
            let flat = mismatch
                .to_kind(Kind::Float)
                .argmax(0, false)
                .int64_value(&[]);
            let (token, bar) = (flat / horizon, flat % horizon);
            let at = |tensor: &Tensor, index: i64| -> String {
                let value = tensor
                    .reshape([-1])
                    .narrow(0, index, 1)
                    .to_kind(Kind::Float)
                    .double_value(&[]) as f32;
                format!("{:#010x}", value.to_bits())
            };
            let head = inputs.head.reshape([-1, 2 * LOSS_CHANNELS, horizon]);
            let targets = inputs.targets.reshape([-1, LOSS_CHANNELS, horizon]);
            let heads: Vec<String> = (0..2 * LOSS_CHANNELS)
                .map(|source| at(&head.select(0, token).select(0, source), bar))
                .collect();
            let goals: Vec<String> = (0..LOSS_CHANNELS)
                .map(|source| at(&targets.select(0, token).select(0, source), bar))
                .collect();
            let terms: Vec<String> = (0..2 * LOSS_CHANNELS)
                .map(|index| at(&upstream, index))
                .collect();
            println!("  token {token} bar {bar} head {heads:?} target {goals:?}");
            println!(
                "  wmask {} mask {} sigma {} range {} hs {} ih {} gain {} upstream {terms:?}",
                at(&inputs.mask, flat),
                at(&inputs.weighted_mask, flat),
                at(&inputs.sigma, token),
                at(&inputs.range, token),
                at(&inputs.horizon_scale, bar),
                at(&inputs.inverse_horizon, bar),
                at(&inputs.log_scale_gain, 0),
            );
            println!("  composed {} fused {}", at(&left, flat), at(&right, flat));
        }

        // What the COMPOSITION hands to the mean coordinate itself, one level above the
        // head: if the sign of a zero is already decided here then the disagreement is in
        // the accumulation onto `close` and not in the cast or the `horizon_scale` product.
        let geometry = reference::loss_geometry(
            &inputs.head,
            &inputs.targets,
            &inputs.weighted_mask,
            &inputs.mask,
            &inputs.sigma,
            &inputs.range,
            &inputs.horizon_scale,
            &inputs.inverse_horizon,
            &inputs.log_scale_gain,
            LOSS_CAP,
        );
        let objective = geometry.terms.dot(&upstream);
        let grads = Tensor::run_backward(&[&objective], &[&geometry.close], false, false);
        let close_gradient = flatten(&grads[0]);
        let head_zero = flatten(&composed_gradient.narrow(2, 0, 1));
        let negative_close = close_gradient
            .signbit()
            .logical_and(&close_gradient.eq(0.0))
            .to_kind(Kind::Int64)
            .sum(Kind::Int64)
            .int64_value(&[]);
        let negative_head = head_zero
            .signbit()
            .logical_and(&head_zero.eq(0.0))
            .to_kind(Kind::Int64)
            .sum(Kind::Int64)
            .int64_value(&[]);
        let fused_zero = flatten(&fused_gradient.narrow(2, 0, 1));
        let negative_fused = fused_zero
            .signbit()
            .logical_and(&fused_zero.eq(0.0))
            .to_kind(Kind::Int64)
            .sum(Kind::Int64)
            .int64_value(&[]);
        println!(
            "negative zeros: composed close {negative_close}, composed head {negative_head}, \
             fused head {negative_fused}"
        );
    }

    /// The masked path's NEGATIVE ZERO is the composition's, and it reaches the optimizer
    /// unchanged. This is the test that found the one real defect in the fused backward.
    ///
    /// The mechanism, named rather than characterized. Where an origin is invalid the folded
    /// mask is `+0`, so `precision = mask·h⁻¹` is `+0` and `weight = exp(-2·cap·s)·precision`
    /// is `+0`. Every gradient below it is then a signed zero whose sign is a pure XOR of the
    /// signs of `grad_dot`, of the residual `target - prediction` and of the `neg` that the
    /// residual's backward applies - a product chain, so IEEE fixes it, and both forms agree
    /// on it. The mean coordinate is the ONE channel whose gradient is a SUM of two such
    /// zeros, the channel-3 residual plus the `low` path: `-0 + -0` is `-0` and every other
    /// pairing is `+0`, so it is the only channel where a spurious zero ADDEND is observable
    /// at all. Channels 1-3 re-derive the sign through a `sigmoid_backward` or
    /// `softplus_backward` product and channels 4-7 never touch `close`.
    ///
    /// The defect it exposed was a spurious addend, and it was in the BRIDGE. Autograd
    /// materializes an unused output's gradient as a freshly zeroed tensor, so the mean
    /// coordinate's `grad_close` arrived defined and full of `+0` even with nothing consuming
    /// it, and the kernel dutifully added it: `-0 + +0` is `+0`, against the composition's
    /// `-0`, on 3244 of 4,608,000 elements - every one of them an invalid origin.
    /// `ctx->set_materialize_grads(false)` is the fix, and the lesson generalizes past zeros:
    /// a materialized gradient is an EXTRA operand, and a fused op with optional outputs has
    /// to refuse it rather than trust that adding zero is free.
    ///
    /// The comparison is promoted one level above the head deliberately. What the bar exists
    /// to protect is the parameter gradient the optimizer consumes, so this test puts the
    /// real head-output GEMM under the loss and requires the weight AND activation gradients
    /// to be bit-identical, sign bit included. It also refuses to pass vacuously: the
    /// composition must actually produce negative zeros, or the sign convention was never
    /// exercised and the test proved nothing.
    #[test]
    fn the_masked_signed_zero_is_the_compositions_and_reaches_a_parameter_gradient_intact() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        tch::manual_seed(43);
        let (rows, origins, horizon, hidden) = (8i64, 375i64, 192i64, 64i64);
        let inputs = loss_inputs(rows, origins, horizon, device, 2.0);
        let upstream = Tensor::randn([2 * LOSS_CHANNELS], (Kind::Float, device));
        let activation = bf16_randn(&[rows * origins, hidden], device).set_requires_grad(true);
        let weight = (bf16_randn(&[hidden, 2 * LOSS_CHANNELS * horizon], device) * 0.25)
            .set_requires_grad(true);
        let shape = [rows, origins, 2 * LOSS_CHANNELS, horizon];

        let run = |fused: bool| {
            let head = activation.matmul(&weight).reshape(shape);
            let geometry = if fused {
                loss_geometry(
                    &head,
                    &inputs.targets,
                    &inputs.weighted_mask,
                    &inputs.mask,
                    &inputs.sigma,
                    &inputs.range,
                    &inputs.horizon_scale,
                    &inputs.inverse_horizon,
                    &inputs.log_scale_gain,
                    LOSS_CAP,
                )
            } else {
                reference::loss_geometry(
                    &head,
                    &inputs.targets,
                    &inputs.weighted_mask,
                    &inputs.mask,
                    &inputs.sigma,
                    &inputs.range,
                    &inputs.horizon_scale,
                    &inputs.inverse_horizon,
                    &inputs.log_scale_gain,
                    LOSS_CAP,
                )
            };
            let objective = geometry.terms.dot(&upstream);
            let gradients =
                Tensor::run_backward(&[&objective], &[&weight, &activation, &head], false, false);
            (
                gradients[0].shallow_clone(),
                gradients[1].shallow_clone(),
                gradients[2].shallow_clone(),
            )
        };
        let (composed_weight, composed_activation, composed_head) = run(false);
        let (fused_weight, fused_activation, fused_head) = run(true);

        // The masked path is LIVE - the batch really does contain invalid origins, so the
        // negative zeros this test exists for are present rather than hypothetical - and the
        // two forms agree on every one of them, sign bit included.
        let negatives = |tensor: &Tensor| {
            let flat = tensor.reshape([-1]).to_kind(Kind::Float);
            flat.signbit()
                .logical_and(&flat.eq(0.0))
                .to_kind(Kind::Int64)
                .sum(Kind::Int64)
                .int64_value(&[])
        };
        let seeds = negatives(&composed_head);
        assert!(
            seeds > 0,
            "the composition produced no negative zero at this shape, so the sign convention \
             this test exists for was never exercised"
        );
        assert!(
            identical(&composed_head, &fused_head),
            "the head gradient is not bit-identical: {} of {} elements differ, max |delta| {}, \
             composed negative zeros {seeds}, fused {}",
            differing(&composed_head, &fused_head),
            composed_head.numel(),
            max_absolute(&composed_head, &fused_head),
            negatives(&fused_head)
        );

        // And it washes out: what the optimizer receives is bit-identical, sign bit included.
        assert!(
            composed_weight
                .to_kind(Kind::Float)
                .abs()
                .max()
                .double_value(&[])
                > 0.0,
            "the weight gradient is all zero, so the comparison is vacuous"
        );
        assert!(
            identical(&composed_weight, &fused_weight),
            "the head-output weight gradient is not bit-identical: {} of {} elements differ, \
             max |delta| {}",
            differing(&composed_weight, &fused_weight),
            composed_weight.numel(),
            max_absolute(&composed_weight, &fused_weight)
        );
        assert!(
            identical(&composed_activation, &fused_activation),
            "the head activation gradient is not bit-identical: {} of {} elements differ, \
             max |delta| {}",
            differing(&composed_activation, &fused_activation),
            composed_activation.numel(),
            max_absolute(&composed_activation, &fused_activation)
        );
    }

    fn split_geometry(inputs: &LossInputs, cap: f64) -> LossGeometry {
        decoupled_loss_geometry(
            &inputs.head,
            &inputs.targets,
            &inputs.weighted_mask,
            &inputs.mask,
            &inputs.sigma,
            &inputs.range,
            &inputs.horizon_scale,
            &inputs.inverse_horizon,
            &inputs.log_scale_gain,
            cap,
        )
    }

    fn assert_split_pair(
        inputs: &LossInputs,
        upstream: &Tensor,
        cap: f64,
    ) -> (LossGeometry, Tensor) {
        let (composed, composed_gradient, fused, fused_gradient) =
            loss_pair_mode(inputs, upstream, None, LOSS_GEOMETRY_ROUNDING, cap, true);
        assert_eq!(
            loss_differences(&composed, &composed_gradient, &fused, &fused_gradient),
            [0, 0, 0, 0],
            "decoupled geometry/terms/squares/gradient differ from the composition"
        );
        assert!(identical(
            composed.nll_terms.as_ref().unwrap(),
            fused.nll_terms.as_ref().unwrap()
        ));
        assert!(!fused.nll_terms.as_ref().unwrap().requires_grad());
        assert!(!fused.squares.requires_grad());
        (fused, fused_gradient)
    }

    #[test]
    fn fused_decoupled_loss_geometry_matches_values_and_gradients() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        tch::manual_seed(59);
        // Real horizon with channel-innermost targets; odd sizes and saturated nonlinear
        // branches; then the h=1 / cap=0 boundary where the two precisions coincide.
        for (rows, origins, horizon, scale, cap) in [
            (8, 375, 192, 2.0, LOSS_CAP),
            (3, 257, 37, 24.0, 1.75),
            (2, 5, 1, 2.0, 0.0),
        ] {
            let mut inputs = loss_inputs(rows, origins, horizon, device, scale);
            inputs.targets = inputs.targets.transpose(2, 3).contiguous().transpose(2, 3);
            let upstream = Tensor::randn([3 * LOSS_CHANNELS], (Kind::Float, device));
            let (fused, _) = assert_split_pair(&inputs, &upstream, cap);
            let coupled = tch::no_grad(|| {
                reference::loss_geometry(
                    &inputs.head,
                    &inputs.targets,
                    &inputs.weighted_mask,
                    &inputs.mask,
                    &inputs.sigma,
                    &inputs.range,
                    &inputs.horizon_scale,
                    &inputs.inverse_horizon,
                    &inputs.log_scale_gain,
                    cap,
                )
            });
            assert!(
                identical(fused.nll_terms.as_ref().unwrap(), &coupled.terms),
                "decoupling changed the reported Gaussian NLL at cap={cap}"
            );
        }
    }

    #[test]
    fn fused_decoupled_loss_geometry_isolates_mean_and_scale_gradients() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        tch::manual_seed(61);
        let inputs = loss_inputs(3, 17, 37, device, 2.0);
        let ones = Tensor::ones([3 * LOSS_CHANNELS], (Kind::Float, device));
        let (geometry, gradient) = assert_split_pair(&inputs, &ones, LOSS_CAP);
        let mut changed = targets_dense(&inputs);
        changed.head = inputs.head.detach().copy().set_requires_grad(true);
        tch::no_grad(|| {
            changed
                .head
                .narrow(2, LOSS_CHANNELS, LOSS_CHANNELS)
                .copy_(&(inputs.head.narrow(2, LOSS_CHANNELS, LOSS_CHANNELS) + 3.5));
        });
        let (_, changed_gradient) = assert_split_pair(&changed, &ones, LOSS_CAP);
        assert!(
            identical(
                &gradient.narrow(2, 0, LOSS_CHANNELS),
                &changed_gradient.narrow(2, 0, LOSS_CHANNELS)
            ),
            "the mean gradient still depends on learned precision"
        );
        assert!(
            differing(
                &gradient.narrow(2, LOSS_CHANNELS, LOSS_CHANNELS),
                &changed_gradient.narrow(2, LOSS_CHANNELS, LOSS_CHANNELS)
            ) > 0,
            "the scale perturbation did not exercise its gradient"
        );
        let coupled_upstream = Tensor::ones([2 * LOSS_CHANNELS], (Kind::Float, device));
        let (_, _, _, coupled_gradient) =
            loss_pair(&inputs, &coupled_upstream, None, LOSS_GEOMETRY_ROUNDING);
        assert!(
            identical(
                &gradient.narrow(2, LOSS_CHANNELS, LOSS_CHANNELS),
                &coupled_gradient.narrow(2, LOSS_CHANNELS, LOSS_CHANNELS)
            ),
            "decoupling changed the original likelihood's scale gradient"
        );

        for (weights, disconnected) in [([1.0f32, 0.0, 0.0], LOSS_CHANNELS), ([0.0, 1.0, 1.0], 0)] {
            let upstream = Tensor::from_slice(&weights)
                .to_device(device)
                .repeat([LOSS_CHANNELS]);
            let (_, isolated) = assert_split_pair(&inputs, &upstream, LOSS_CAP);
            assert_eq!(
                isolated
                    .narrow(2, disconnected, LOSS_CHANNELS)
                    .to_kind(Kind::Float)
                    .abs()
                    .max()
                    .double_value(&[]),
                0.0,
                "the split quadratic leaked a gradient into its detached branch"
            );
        }

        // Exercise the consumer's actual contract, not just the three individual terms:
        // true NLL as the value, with the split objective's gradient.
        let fresh = split_geometry(&inputs, LOSS_CAP);
        let train = fresh.terms.sum(Kind::Float);
        let nll = fresh.nll_terms.as_ref().unwrap().sum(Kind::Float);
        let surrogate = &nll + (&train - train.detach());
        assert!(identical(
            &surrogate,
            &geometry.nll_terms.unwrap().sum(Kind::Float)
        ));
        let surrogate_gradient =
            Tensor::run_backward(&[&surrogate], &[&inputs.head], false, false).remove(0);
        assert!(identical(&surrogate_gradient, &gradient));
    }

    #[test]
    fn fused_decoupled_loss_geometry_preserves_invalid_mask_and_scale_semantics() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        tch::manual_seed(67);
        let inputs = loss_inputs(2, 17, 37, device, 6.0);
        let upstream = Tensor::randn([3 * LOSS_CHANNELS], (Kind::Float, device));
        tch::no_grad(|| {
            for (origin, value) in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY]
                .into_iter()
                .enumerate()
            {
                let _ = inputs
                    .head
                    .select(0, 0)
                    .select(0, origin as i64)
                    .narrow(0, LOSS_CHANNELS, LOSS_CHANNELS)
                    .fill_(value);
            }
            // A zero mask must not sanitize a NaN scale; a nonfinite mask must not be
            // silently clamped. Invalid normalization scales follow IEEE, as in ATen.
            let _ = inputs.mask.select(0, 0).select(0, 0).fill_(0.0);
            let _ = inputs.weighted_mask.select(0, 0).select(0, 0).fill_(0.0);
            let _ = inputs.mask.select(0, 1).select(0, 0).fill_(f64::NAN);
            let _ = inputs
                .weighted_mask
                .select(0, 1)
                .select(0, 0)
                .fill_(f64::NAN);
            let _ = inputs.sigma.select(0, 1).select(0, 1).fill_(0.0);
            let _ = inputs.range.select(0, 1).select(0, 2).fill_(-1.0);
        });
        let (geometry, gradient) = assert_split_pair(&inputs, &upstream, LOSS_CAP);
        assert!(geometry.nll_terms.unwrap().isnan().any().int64_value(&[]) != 0);
        assert!(
            gradient
                .select(0, 0)
                .narrow(1, 0, LOSS_CHANNELS)
                .isfinite()
                .all()
                .int64_value(&[])
                != 0,
            "a nonfinite learned scale poisoned the scale-independent mean gradient"
        );

        let mut blind = loss_inputs(2, 17, 37, device, 6.0);
        let _ = blind.mask.fill_(0.0);
        let _ = blind.weighted_mask.fill_(0.0);
        let (geometry, gradient) = assert_split_pair(&blind, &upstream, LOSS_CAP);
        for value in [
            &geometry.terms,
            &geometry.squares,
            geometry.nll_terms.as_ref().unwrap(),
            &gradient,
        ] {
            assert_eq!(
                value.to_kind(Kind::Float).abs().max().double_value(&[]),
                0.0
            );
        }
    }

    #[test]
    fn fused_decoupled_loss_geometry_carries_extra_and_close_only_gradients() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        tch::manual_seed(71);
        let mut inputs = loss_inputs(4, 129, 37, device, 2.0);
        let upstream = Tensor::randn([3 * LOSS_CHANNELS], (Kind::Float, device));
        let close_upstream = Tensor::randn(inputs.mask.size(), (Kind::Float, device));
        let (composed, composed_gradient, fused, fused_gradient) = loss_pair_mode(
            &inputs,
            &upstream,
            Some(&close_upstream),
            LOSS_GEOMETRY_ROUNDING,
            LOSS_CAP,
            true,
        );
        assert!(identical(&composed.close, &fused.close));
        assert!(identical(&composed.terms, &fused.terms));
        assert!(identical(
            &composed_gradient.narrow(2, 1, 2 * LOSS_CHANNELS - 1),
            &fused_gradient.narrow(2, 1, 2 * LOSS_CHANNELS - 1)
        ));
        // The extra consumer changes three-addend accumulation order only on coordinate 0.
        // Match the existing coupled-loss rounding bound; all other channels remain exact.
        let scale = composed_gradient
            .to_kind(Kind::Float)
            .abs()
            .max()
            .double_value(&[]);
        assert!(scale > 0.0);
        assert!(max_absolute(&composed_gradient, &fused_gradient) / scale <= 1e-2);

        tch::no_grad(|| {
            let _ = inputs.targets.fill_(f64::NAN);
            let _ = inputs
                .head
                .narrow(2, LOSS_CHANNELS, LOSS_CHANNELS)
                .fill_(f64::NAN);
        });
        let geometry = split_geometry(&inputs, LOSS_CAP);
        let objective = (&geometry.close * &close_upstream).sum(Kind::Float);
        let gradient = Tensor::run_backward(&[&objective], &[&inputs.head], false, false).remove(0);
        let reference_close =
            inputs.head.narrow(2, 0, 1).to_kind(Kind::Float) * &inputs.horizon_scale;
        let reference_objective = (reference_close * &close_upstream).sum(Kind::Float);
        let reference_gradient =
            Tensor::run_backward(&[&reference_objective], &[&inputs.head], false, false).remove(0);
        assert!(
            identical(&gradient, &reference_gradient),
            "unused likelihood outputs contaminated the close-only gradient"
        );
    }

    #[test]
    fn fused_decoupled_loss_geometry_rejects_malformed_mask_and_scale_buffers() {
        let Some(claim) = cuda() else { return };
        let device = *claim;
        for malformed in 0..3 {
            let mut inputs = loss_inputs(2, 17, 37, device, 2.0);
            match malformed {
                // Broadcastable is not sufficient: the kernel addresses one mask per bar.
                0 => inputs.weighted_mask = Tensor::ones([2, 17, 1, 1], (Kind::Float, device)),
                1 => inputs.inverse_horizon = Tensor::ones([36], (Kind::Float, device)),
                _ => inputs.log_scale_gain = Tensor::ones([2], (Kind::Float, device)),
            }
            let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                split_geometry(&inputs, LOSS_CAP)
            }));
            assert!(
                rejected.is_err(),
                "malformed buffer {malformed} reached a kernel launch"
            );
        }
    }
}
