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

// A 128-bit vectorized bf16 copy: `out[i] = in[i]`. Not used by the model. It exists to
// measure this device's streaming roof with the SAME launch geometry and access width as
// the kernels above, so a percentage against it is an efficiency statement rather than a
// comparison with a differently-shaped kernel.
int fk_stream_copy(const void *input, void *output, int64_t count, void *stream);

#ifdef __cplusplus
}
#endif
