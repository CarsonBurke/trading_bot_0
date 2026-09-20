// Pure-CUDA launcher declarations. No libtorch types cross this boundary: the .cu
// translation unit is compiled by nvcc and knows nothing but pointers, extents and a
// stream, so nvcc never has to parse a libtorch header and the ABI of this file is the
// only thing the C++ bridge and the kernels have to agree on.
//
// Every launcher is CUDA-graph-capturable: it allocates nothing, synchronizes nothing,
// reads no device memory from the host, and takes all of its extents as arguments so the
// grid is fixed at capture time.
#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// `y[i] = relu(x[i])^2` over `count` bf16 elements. Both buffers are dense.
// Returns a cudaError_t as int (0 == cudaSuccess) from the launch only.
int fk_relu_square_forward(const void *input, void *output, int64_t count,
                           void *stream);

// `dx[i] = x[i] > 0 ? 2 * grad[i] * x[i] : 0` over `count` bf16 elements.
int fk_relu_square_backward(const void *input, const void *grad, void *dx,
                            int64_t count, void *stream);

// Packed-`q‖k` rotary. `rows` is batch*length; each row holds `blocks = 2*heads`
// contiguous head blocks of `2*half` elements, the first `half` being the low half of
// the rotary pair and the second the high half. `cosine`/`sine` are `[length, half]`
// bf16 and are indexed by `row % length`.
//
//   out_low  = x_low * cos - x_high * sin
//   out_high = x_high * cos + x_low * sin
//
// with each product rounded to bf16 before the sum, which is what the ATen composition
// this replaces does.
int fk_rope_forward(const void *input, const void *cosine, const void *sine,
                    void *output, int64_t rows, int64_t length, int64_t blocks,
                    int64_t half, int64_t input_row_stride,
                    int64_t output_row_stride, void *stream);

// The transpose of the same rotation:
//   dx_low  = g_low * cos + g_high * sin
//   dx_high = g_high * cos - g_low * sin
int fk_rope_backward(const void *grad, const void *cosine, const void *sine,
                     void *dx, int64_t rows, int64_t length, int64_t blocks,
                     int64_t half, int64_t grad_row_stride,
                     int64_t dx_row_stride, void *stream);

// Fused per-head QK-normalization + the same packed rotary. `width` is `head_dim` as a
// float (ATen's `fH`), `eps` the RMSNorm epsilon, and `rounding` the fp32 contraction
// selector documented at the definition. Each head block of `2*half` elements is one
// gainless, biasless RMS normalization group AND one rotary block, which is why the two
// ops fuse without changing either one's addressing:
//
//   y        = bf16(rstd * x),  rstd = rsqrt(Σ x² / width + eps)
//   out_low  = y_low * cos - y_high * sin
//   out_high = y_high * cos + y_low * sin
//
// The normalized `y` is never written anywhere. `head_dim` must be a multiple of four and
// at most 128: outside that range ATen's own forward switches to a Welford kernel with a
// different summation order and bit-identity is no longer defined.
int fk_qk_norm_rope_forward(const void *input, const void *cosine, const void *sine,
                            void *output, int64_t rows, int64_t length, int64_t blocks,
                            int64_t half, int64_t input_row_stride,
                            int64_t output_row_stride, float width, float eps,
                            int rounding, void *stream);

// The composition's backward in one pass: the transposed rotation recovers the gradient of
// the normalized block, the per-head RMS is RECOMPUTED from `input` (a reduction over
// `head_dim` elements already in flight, so no `rstd` tensor is retained by the forward),
// and the RMSNorm gradient follows:
//
//   gy    = g_low * cos + g_high * sin  (low),  g_high * cos - g_low * sin  (high)
//   stats = Σ (gy * x) * rstd
//   dx    = (width * gy - x * rstd * stats) * (rstd / width)
int fk_qk_norm_rope_backward(const void *grad, const void *input, const void *cosine,
                             const void *sine, void *dx, int64_t rows, int64_t length,
                             int64_t blocks, int64_t half, int64_t grad_row_stride,
                             int64_t input_row_stride, int64_t dx_row_stride, float width,
                             float eps, int rounding, void *stream);

// A 128-bit vectorized bf16 copy: `out[i] = in[i]`. Not used by the model. It exists to
// measure this device's streaming roof with the SAME launch geometry and access width as
// the kernels above, so a percentage against it is an efficiency statement rather than a
// comparison with a differently-shaped kernel.
int fk_stream_copy(const void *input, void *output, int64_t count, void *stream);

// The candle-geometry + Gaussian-NLL element chain in one pass.
//
// `head` is `[tokens, 8, horizon]` bf16 (four candle coordinates then four log scales),
// `targets` `[tokens, 4, horizon]` fp32, `weighted_mask` `[tokens, horizon]` fp32,
// `sigma`/`range` `[tokens]` fp32 already clamped, and `horizon_scale`/`inverse_horizon`
// `[horizon]` fp32. `log_scale_gain` is a one-element fp32 buffer read on the device, never
// on the host, so the launch stays graph-capturable.
//
// The channel count is FOUR and is not a parameter: the geometry is a candle - one close,
// one range and two positions inside it - and there is no meaning to a fifth coordinate.
//
// Outputs, in the layout the reductions consume: `close` is `[tokens, horizon]` fp32 (the
// σ-scaled mean coordinate, which the amplitude prior reduces), and `workspace` is
// `[12, tokens*horizon]` fp32 - rows 0-3 the squared errors, 4-7 the precision weights,
// 8-11 the capped log scales. Each row is contiguous, so the twelve `dot`s that follow are
// ATen's own and their summation trees are untouched.
// Decoupled mode appends row 12, `weighted_mask * inverse_horizon`, shared by the four
// mean-only quadratic dots; rows 0-11 and all geometry arithmetic are unchanged.
//
// `rounding` selects the fp32 forms ATen's own build emitted; see the definition of
// `FK_LOSS_GEOMETRY_ROUNDING` for the fields and for why they are measured, not chosen.
// `targets` is `[tokens, 4, horizon]` fp32 addressed by the three strides below rather
// than assumed dense: the real path builds it with `unfold`, and TensorIterator allocates
// the arithmetic under it in the inputs' own permuted layout, which is channel-innermost.
int fk_loss_geometry_forward(const void *head, const void *targets,
                             const void *weighted_mask, const void *sigma,
                             const void *range, const void *horizon_scale,
                             const void *inverse_horizon, const void *log_scale_gain,
                             void *close, void *workspace, int64_t tokens, int64_t horizon,
                             int64_t target_token_stride, int64_t target_channel_stride,
                             int64_t target_bar_stride, float cap, float ln2,
                             float inverse_ln2, int rounding, int decoupled, void *stream);

// The same chain's transpose in one pass. `grad_terms` is `[8]` fp32 - per channel the
// gradient of `dot(square, weight)·½` then of `dot(log_scale, weighted_mask)·cap`.
// In decoupled mode `grad_terms` is `[12]`: per channel the fixed-precision mean
// quadratic, detached-residual scale quadratic, then the log-scale term. Mean gradients
// never use learned precision; scale gradients retain the original NLL derivative.
// `grad_close` is either `[tokens, horizon]` fp32 or null when nothing consumes the mean
// coordinate. Nothing the forward computed is retained: every intermediate is recomputed
// from `head` and the targets, which is one read of data already resident against twelve
// full-size fp32 tensors the composition kept alive.
//
// `grad_head` is the dense `[tokens, 8, horizon]` bf16 gradient, written once. The
// composition needed a `cat` over eight slices to produce it.
int fk_loss_geometry_backward(const void *head, const void *targets,
                              const void *weighted_mask, const void *sigma,
                              const void *range, const void *horizon_scale,
                              const void *inverse_horizon, const void *log_scale_gain,
                              const void *grad_terms, const void *grad_close,
                              void *grad_head, int64_t tokens, int64_t horizon,
                              int64_t target_token_stride, int64_t target_channel_stride,
                              int64_t target_bar_stride, float cap, float ln2,
                              float inverse_ln2, int rounding, int decoupled, void *stream);

#ifdef __cplusplus
}
#endif
