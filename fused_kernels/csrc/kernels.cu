// Fused bf16 kernels for the CausalPatch backbone.
//
// Both fusions exist because the composed-ATen forms are bandwidth-bound and re-read the
// same [tokens, width] activation once per elementary op. Every kernel here reads each
// input exactly once and writes each output exactly once.
//
// Rounding discipline: all arithmetic happens in fp32 and every value that the ATen
// composition would have MATERIALIZED as a bf16 tensor is rounded to bf16 at exactly that
// point. That is what makes these bit-identical to the composition rather than merely
// close to it; keeping the intermediates in fp32 would be strictly more accurate and
// would NOT reproduce the landed reference, and reproducing the reference is what lets
// this be swapped in without moving any training curve.
//
// Graph capture: no allocation, no synchronization, no device-to-host read, no dynamic
// shape. Grid extents are pure functions of the launch arguments, so a capture fixes them.

#include "kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;
// Grid cap. With a grid-stride loop the result is independent of the cap, so this only
// trades launch width against per-thread work; 32768 blocks is ~8.4 M threads, enough to
// saturate a GB202 while keeping several vectors of work per thread at the FFN shape.
constexpr int64_t kMaxBlocks = 32768;

// 16 bytes = one 128-bit access = eight bf16 lanes.
struct alignas(16) Vec8 {
    __nv_bfloat16 lane[8];
};

inline int64_t grid_for(int64_t items) {
    const int64_t blocks = (items + kThreads - 1) / kThreads;
    return blocks < 1 ? 1 : (blocks > kMaxBlocks ? kMaxBlocks : blocks);
}

__device__ __forceinline__ float relu_of(float value) {
    // NaN-propagating, like `clamp_min`, which is what `relu` lowers to. A plain
    // `value > 0 ? value : 0` would turn a NaN activation into a silent zero and the
    // forward would stop matching the reference exactly where it matters most.
    return isnan(value) ? value : fmaxf(value, 0.0f);
}

__global__ void relu_square_forward_vec(const Vec8 *__restrict__ input,
                                        Vec8 *__restrict__ output, int64_t vectors) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
         index < vectors; index += stride) {
        const Vec8 in = input[index];
        Vec8 out;
#pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            const float rectified = relu_of(__bfloat162float(in.lane[lane]));
            out.lane[lane] = __float2bfloat16(rectified * rectified);
        }
        output[index] = out;
    }
}

__global__ void relu_square_forward_scalar(const __nv_bfloat16 *__restrict__ input,
                                           __nv_bfloat16 *__restrict__ output,
                                           int64_t count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
         index < count; index += stride) {
        const float rectified = relu_of(__bfloat162float(input[index]));
        output[index] = __float2bfloat16(rectified * rectified);
    }
}

// `x <= 0 ? 0 : 2*g*x`, and the direction of that comparison is measured, not assumed.
// The composition ends in `threshold_backward(., relu(x), 0)`, whose kernel is
// `self <= threshold ? 0 : grad` - so the mask is the NEGATION of `x <= 0`, which is TRUE
// at NaN, and a NaN activation therefore carries a NaN gradient rather than a zero one.
// Writing the natural `x > 0 ? ... : 0` instead differs from the reference on exactly the
// inputs a numerical audit cares about; `composed_relu_square_zeroes_the_gradient_at_nan`
// is the test that caught it.
__device__ __forceinline__ __nv_bfloat16 relu_square_grad(float x, float grad) {
    // `2 * grad` only decrements a bf16 exponent, so it is exact, and a product of two
    // 8-bit mantissas is exact in fp32: this single rounding is the reference's single
    // rounding of `grad * 2 * self`.
    return x <= 0.0f ? __float2bfloat16(0.0f) : __float2bfloat16(2.0f * grad * x);
}

__global__ void relu_square_backward_vec(const Vec8 *__restrict__ input,
                                         const Vec8 *__restrict__ grad,
                                         Vec8 *__restrict__ dx, int64_t vectors) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
         index < vectors; index += stride) {
        const Vec8 in = input[index];
        const Vec8 upstream = grad[index];
        Vec8 out;
#pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            out.lane[lane] = relu_square_grad(__bfloat162float(in.lane[lane]),
                                              __bfloat162float(upstream.lane[lane]));
        }
        dx[index] = out;
    }
}

__global__ void relu_square_backward_scalar(const __nv_bfloat16 *__restrict__ input,
                                            const __nv_bfloat16 *__restrict__ grad,
                                            __nv_bfloat16 *__restrict__ dx,
                                            int64_t count) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
         index < count; index += stride) {
        dx[index] = relu_square_grad(__bfloat162float(input[index]),
                                     __bfloat162float(grad[index]));
    }
}

// One rounded bf16 product, which is what an ATen `mul` on two bf16 tensors produces.
__device__ __forceinline__ float rounded_product(float a, float b) {
    return __bfloat162float(__float2bfloat16(a * b));
}

// `Forward` selects the sign convention; the two directions differ only in which of the
// two sine products is negated, so they share the whole addressing body.
template <bool Forward>
__global__ void rope_vec(const Vec8 *__restrict__ input, const Vec8 *__restrict__ cosine,
                         const Vec8 *__restrict__ sine, Vec8 *__restrict__ output,
                         int64_t rows, int64_t length, int64_t blocks,
                         int64_t half_vectors, int64_t input_row_vectors,
                         int64_t output_row_vectors) {
    const int64_t per_row = blocks * half_vectors;
    const int64_t items = rows * per_row;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
         index < items; index += stride) {
        const int64_t row = index / per_row;
        const int64_t within = index - row * per_row;
        const int64_t block = within / half_vectors;
        const int64_t vector = within - block * half_vectors;
        const int64_t position = row % length;

        const int64_t in_low = row * input_row_vectors + block * 2 * half_vectors + vector;
        const int64_t out_low = row * output_row_vectors + block * 2 * half_vectors + vector;
        const int64_t rotation = position * half_vectors + vector;

        const Vec8 low = input[in_low];
        const Vec8 high = input[in_low + half_vectors];
        const Vec8 cos = cosine[rotation];
        const Vec8 sin = sine[rotation];
        Vec8 out_a;
        Vec8 out_b;
#pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            const float x_low = __bfloat162float(low.lane[lane]);
            const float x_high = __bfloat162float(high.lane[lane]);
            const float c = __bfloat162float(cos.lane[lane]);
            const float s = __bfloat162float(sin.lane[lane]);
            const float low_cos = rounded_product(x_low, c);
            const float high_cos = rounded_product(x_high, c);
            const float low_sin = rounded_product(x_low, s);
            const float high_sin = rounded_product(x_high, s);
            if (Forward) {
                out_a.lane[lane] = __float2bfloat16(low_cos - high_sin);
                out_b.lane[lane] = __float2bfloat16(high_cos + low_sin);
            } else {
                out_a.lane[lane] = __float2bfloat16(low_cos + high_sin);
                out_b.lane[lane] = __float2bfloat16(high_cos - low_sin);
            }
        }
        output[out_low] = out_a;
        output[out_low + half_vectors] = out_b;
    }
}

template <bool Forward>
__global__ void rope_scalar(const __nv_bfloat16 *__restrict__ input,
                            const __nv_bfloat16 *__restrict__ cosine,
                            const __nv_bfloat16 *__restrict__ sine,
                            __nv_bfloat16 *__restrict__ output, int64_t rows,
                            int64_t length, int64_t blocks, int64_t half,
                            int64_t input_row_stride, int64_t output_row_stride) {
    const int64_t per_row = blocks * half;
    const int64_t items = rows * per_row;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
         index < items; index += stride) {
        const int64_t row = index / per_row;
        const int64_t within = index - row * per_row;
        const int64_t block = within / half;
        const int64_t pair = within - block * half;
        const int64_t position = row % length;

        const int64_t in_low = row * input_row_stride + block * 2 * half + pair;
        const int64_t out_low = row * output_row_stride + block * 2 * half + pair;
        const float x_low = __bfloat162float(input[in_low]);
        const float x_high = __bfloat162float(input[in_low + half]);
        const float c = __bfloat162float(cosine[position * half + pair]);
        const float s = __bfloat162float(sine[position * half + pair]);
        const float low_cos = rounded_product(x_low, c);
        const float high_cos = rounded_product(x_high, c);
        const float low_sin = rounded_product(x_low, s);
        const float high_sin = rounded_product(x_high, s);
        if (Forward) {
            output[out_low] = __float2bfloat16(low_cos - high_sin);
            output[out_low + half] = __float2bfloat16(high_cos + low_sin);
        } else {
            output[out_low] = __float2bfloat16(low_cos + high_sin);
            output[out_low + half] = __float2bfloat16(high_cos - low_sin);
        }
    }
}

__global__ void stream_copy_vec(const Vec8 *__restrict__ input, Vec8 *__restrict__ output,
                                int64_t vectors) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
         index < vectors; index += stride) {
        output[index] = input[index];
    }
}

bool aligned16(const void *pointer) {
    return (reinterpret_cast<uintptr_t>(pointer) & 15u) == 0;
}

template <bool Forward>
int launch_rope(const void *input, const void *cosine, const void *sine, void *output,
                int64_t rows, int64_t length, int64_t blocks, int64_t half,
                int64_t input_row_stride, int64_t output_row_stride, void *stream) {
    cudaStream_t handle = static_cast<cudaStream_t>(stream);
    const bool vectorizable = half % 8 == 0 && input_row_stride % 8 == 0 &&
                              output_row_stride % 8 == 0 && aligned16(input) &&
                              aligned16(output) && aligned16(cosine) && aligned16(sine);
    if (vectorizable) {
        const int64_t half_vectors = half / 8;
        const int64_t items = rows * blocks * half_vectors;
        rope_vec<Forward><<<grid_for(items), kThreads, 0, handle>>>(
            static_cast<const Vec8 *>(input), static_cast<const Vec8 *>(cosine),
            static_cast<const Vec8 *>(sine), static_cast<Vec8 *>(output), rows, length,
            blocks, half_vectors, input_row_stride / 8, output_row_stride / 8);
    } else {
        const int64_t items = rows * blocks * half;
        rope_scalar<Forward><<<grid_for(items), kThreads, 0, handle>>>(
            static_cast<const __nv_bfloat16 *>(input),
            static_cast<const __nv_bfloat16 *>(cosine),
            static_cast<const __nv_bfloat16 *>(sine),
            static_cast<__nv_bfloat16 *>(output), rows, length, blocks, half,
            input_row_stride, output_row_stride);
    }
    return static_cast<int>(cudaGetLastError());
}

} // namespace

extern "C" int fk_relu_square_forward(const void *input, void *output, int64_t count,
                                      void *stream) {
    cudaStream_t handle = static_cast<cudaStream_t>(stream);
    if (count % 8 == 0 && aligned16(input) && aligned16(output)) {
        const int64_t vectors = count / 8;
        relu_square_forward_vec<<<grid_for(vectors), kThreads, 0, handle>>>(
            static_cast<const Vec8 *>(input), static_cast<Vec8 *>(output), vectors);
    } else {
        relu_square_forward_scalar<<<grid_for(count), kThreads, 0, handle>>>(
            static_cast<const __nv_bfloat16 *>(input),
            static_cast<__nv_bfloat16 *>(output), count);
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int fk_relu_square_backward(const void *input, const void *grad, void *dx,
                                       int64_t count, void *stream) {
    cudaStream_t handle = static_cast<cudaStream_t>(stream);
    if (count % 8 == 0 && aligned16(input) && aligned16(grad) && aligned16(dx)) {
        const int64_t vectors = count / 8;
        relu_square_backward_vec<<<grid_for(vectors), kThreads, 0, handle>>>(
            static_cast<const Vec8 *>(input), static_cast<const Vec8 *>(grad),
            static_cast<Vec8 *>(dx), vectors);
    } else {
        relu_square_backward_scalar<<<grid_for(count), kThreads, 0, handle>>>(
            static_cast<const __nv_bfloat16 *>(input),
            static_cast<const __nv_bfloat16 *>(grad), static_cast<__nv_bfloat16 *>(dx),
            count);
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int fk_rope_forward(const void *input, const void *cosine, const void *sine,
                               void *output, int64_t rows, int64_t length, int64_t blocks,
                               int64_t half, int64_t input_row_stride,
                               int64_t output_row_stride, void *stream) {
    return launch_rope<true>(input, cosine, sine, output, rows, length, blocks, half,
                             input_row_stride, output_row_stride, stream);
}

extern "C" int fk_rope_backward(const void *grad, const void *cosine, const void *sine,
                                void *dx, int64_t rows, int64_t length, int64_t blocks,
                                int64_t half, int64_t grad_row_stride,
                                int64_t dx_row_stride, void *stream) {
    return launch_rope<false>(grad, cosine, sine, dx, rows, length, blocks, half,
                              grad_row_stride, dx_row_stride, stream);
}

extern "C" int fk_stream_copy(const void *input, void *output, int64_t count, void *stream) {
    if (count % 8 != 0 || !aligned16(input) || !aligned16(output)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int64_t vectors = count / 8;
    stream_copy_vec<<<grid_for(vectors), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Vec8 *>(input), static_cast<Vec8 *>(output), vectors);
    return static_cast<int>(cudaGetLastError());
}
