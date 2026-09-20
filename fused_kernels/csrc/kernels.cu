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

// ---------------------------------------------------------------------------------------
// Fused per-head QK-normalization + packed rotary.
//
// The composition this replaces is `_fused_rms_norm(q‖k viewed as [.., 2*heads, head_dim])`
// followed by the packed rotation, so the normalized `q‖k` is a materialized bf16 tensor
// that only the rotation ever reads. Here it never exists: the kernel takes the RAW packed
// block, normalizes each head block of `head_dim` over itself, rotates, and writes only the
// rotated result. The backward recomputes the per-head RMS from the raw block it is already
// streaming for the `x·rstd·stats` term, so no `rstd` tensor is retained either.
//
// BIT-IDENTITY IS A REDUCTION-ORDER PROBLEM. `rstd` is fp32 and a bf16 output has an 8-bit
// mantissa, so a 1-ulp fp32 difference in `rstd` flips roughly one output element in 10^5 -
// about 1500 elements per layer at the training shape. Matching ATen therefore means
// reproducing its summation TREE, not merely summing the same numbers. ATen's forward for
// bf16 with `N % 4 == 0` is `vectorized_layer_norm_kernel<..., rms_norm=true>` launched with
// `dim3(32, num_threads()/32) = (32, 4)`; `compute_stats` gives thread `thrx` the vectors
// `i ≡ thrx (mod 128)` of four elements each, accumulates each vector's squares serially
// from `0.f`, reduces across the warp with `WARP_SHFL_DOWN` at offsets 16..1, then across
// the four warps, and finally divides by `N`. The backward is
// `layer_norm_grad_input_kernel_vectorized` with 128 threads, whose `BlockReduceSum` is the
// same 32-lane tree over the same four-element partials. So for `head_dim <= 128` both
// directions are: partials over four CONSECUTIVE elements, then a 32-slot binary tree.
//
// This kernel keeps the rotary kernel's thread mapping - one thread per `(row, head block,
// vector)` with `V = half/8` threads per head block, each holding one 128-bit vector of the
// low half and one of the high half - and reproduces that tree exactly on it. Thread `j`
// owns partials `2j, 2j+1` (low) and `2V+2j, 2V+2j+1` (high), so ATen's `offset == 2V` level
// is thread-local, every offset above it adds a structural zero, and the levels below it are
// `log2(V)` XOR butterflies over the head block's own lanes. Nothing is approximated and
// nothing is reassociated.

// `Fma` selects whether a squared/product accumulation contracts into one `fma`. It is a
// template parameter and not a taste question: which one ATen's own build emitted is a
// property of ITS compiler, it changes the last bit of the reduction, and the answer was
// measured (see `qk_norm_rope_rounding_is_the_measured_one`). `__fmul_rn`/`__fadd_rn` are
// used for the non-fused form because plain `*`/`+` would let nvcc contract them anyway.
template <bool Fma>
__device__ __forceinline__ float accumulate_square(float accumulator, float value) {
    return Fma ? __fmaf_rn(value, value, accumulator)
               : __fadd_rn(accumulator, __fmul_rn(value, value));
}

template <bool Fma>
__device__ __forceinline__ float accumulate_product(float accumulator, float left,
                                                    float right) {
    return Fma ? __fmaf_rn(left, right, accumulator)
               : __fadd_rn(accumulator, __fmul_rn(left, right));
}

// One of ATen's four-element partials over a strided run of a thread's own vector.
template <bool Fma>
__device__ __forceinline__ float square_partial(const float *values) {
    float accumulator = 0.0f;
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        accumulator = accumulate_square<Fma>(accumulator, values[lane]);
    }
    return accumulator;
}

// The `log2(V)` XOR butterflies plus the final local add: ATen's offsets `V, V/2, .., 1`
// over the `2V` values left after the thread-local `offset == 2V` level. Every lane of the
// head block ends with the identical total, which is what lets each lane normalize its own
// elements without a broadcast.
//
// The mask is the head block's own lanes, never `0xffffffff`: `items` is always a multiple
// of `V` and the grid stride is too, so a head block is never split across the loop's tail,
// but the WARP can be - and a full-mask shuffle against exited threads is undefined.
__device__ __forceinline__ float head_block_total(float first, float second,
                                                  int64_t half_vectors, unsigned lane) {
    const unsigned block_lanes = static_cast<unsigned>(half_vectors);
    const unsigned mask = ((1u << block_lanes) - 1u) << (lane & ~(block_lanes - 1u));
    for (int64_t offset = half_vectors / 2; offset >= 1; offset >>= 1) {
        first += __shfl_xor_sync(mask, first, static_cast<int>(offset));
        second += __shfl_xor_sync(mask, second, static_cast<int>(offset));
    }
    return first + second;
}

// `rsqrtf`, not `1.0f / sqrtf`: `c10::cuda::compat::rsqrt` is `rsqrtf`, which is a different
// instruction with a different result, and this is the one place where a two-ulp intrinsic
// has to be reproduced rather than improved on.
__device__ __forceinline__ float inverse_rms(float sum_of_squares, float width, float eps) {
    return rsqrtf(sum_of_squares / width + eps);
}

// The normalized value AS THE COMPOSITION MATERIALIZES IT: one fp32 multiply, rounded to
// bf16 exactly once, then widened again for the rotation. Keeping `rstd * x` in fp32 through
// the rotation would be strictly more accurate and would not be the reference.
__device__ __forceinline__ float normalized(float value, float rstd) {
    return __bfloat162float(__float2bfloat16(rstd * value));
}

template <bool Fma>
__global__ void qk_norm_rope_forward_vec(const Vec8 *__restrict__ input,
                                         const Vec8 *__restrict__ cosine,
                                         const Vec8 *__restrict__ sine,
                                         Vec8 *__restrict__ output, int64_t rows,
                                         int64_t length, int64_t blocks,
                                         int64_t half_vectors, int64_t input_row_vectors,
                                         int64_t output_row_vectors, float width,
                                         float eps) {
    const int64_t per_row = blocks * half_vectors;
    const int64_t items = rows * per_row;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const unsigned lane = threadIdx.x & 31u;
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
        float raw_low[8];
        float raw_high[8];
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            raw_low[slot] = __bfloat162float(low.lane[slot]);
            raw_high[slot] = __bfloat162float(high.lane[slot]);
        }
        // ATen's `offset == 2V` level, thread-local: partial `2j` pairs with `2j+2V`, which
        // is this thread's own high vector, and `2j+1` with `2j+1+2V`.
        const float first = square_partial<Fma>(raw_low) + square_partial<Fma>(raw_high);
        const float second =
            square_partial<Fma>(raw_low + 4) + square_partial<Fma>(raw_high + 4);
        const float rstd =
            inverse_rms(head_block_total(first, second, half_vectors, lane), width, eps);

        const Vec8 cos = cosine[rotation];
        const Vec8 sin = sine[rotation];
        Vec8 out_a;
        Vec8 out_b;
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const float y_low = normalized(raw_low[slot], rstd);
            const float y_high = normalized(raw_high[slot], rstd);
            const float c = __bfloat162float(cos.lane[slot]);
            const float s = __bfloat162float(sin.lane[slot]);
            out_a.lane[slot] =
                __float2bfloat16(rounded_product(y_low, c) - rounded_product(y_high, s));
            out_b.lane[slot] =
                __float2bfloat16(rounded_product(y_high, c) + rounded_product(y_low, s));
        }
        output[out_low] = out_a;
        output[out_low + half_vectors] = out_b;
    }
}

// `dx = ((width·gy) - (x·rstd)·stats) · ((1/width)·rstd)`, where `gy` is the gradient the
// composition would have MATERIALIZED between the two ops - the transposed rotation of
// `grad_output`, rounded to bf16 - and `stats = Σ (gy·x)·rstd` over the head block in ATen's
// tree. Written in ATen's operation order, term by term, because that order is the answer.
template <bool Fma, bool FmaStats, bool FmaGrad>
__global__ void qk_norm_rope_backward_vec(
    const Vec8 *__restrict__ grad, const Vec8 *__restrict__ input,
    const Vec8 *__restrict__ cosine, const Vec8 *__restrict__ sine, Vec8 *__restrict__ dx,
    int64_t rows, int64_t length, int64_t blocks, int64_t half_vectors,
    int64_t grad_row_vectors, int64_t input_row_vectors, int64_t dx_row_vectors, float width,
    float eps) {
    const int64_t per_row = blocks * half_vectors;
    const int64_t items = rows * per_row;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const unsigned lane = threadIdx.x & 31u;
    for (int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
         index < items; index += stride) {
        const int64_t row = index / per_row;
        const int64_t within = index - row * per_row;
        const int64_t block = within / half_vectors;
        const int64_t vector = within - block * half_vectors;
        const int64_t position = row % length;

        const int64_t offset = block * 2 * half_vectors + vector;
        const int64_t rotation = position * half_vectors + vector;
        const Vec8 grad_low = grad[row * grad_row_vectors + offset];
        const Vec8 grad_high = grad[row * grad_row_vectors + offset + half_vectors];
        const Vec8 low = input[row * input_row_vectors + offset];
        const Vec8 high = input[row * input_row_vectors + offset + half_vectors];
        const Vec8 cos = cosine[rotation];
        const Vec8 sin = sine[rotation];

        float raw_low[8];
        float raw_high[8];
        float gy_low[8];
        float gy_high[8];
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            raw_low[slot] = __bfloat162float(low.lane[slot]);
            raw_high[slot] = __bfloat162float(high.lane[slot]);
            const float g_low = __bfloat162float(grad_low.lane[slot]);
            const float g_high = __bfloat162float(grad_high.lane[slot]);
            const float c = __bfloat162float(cos.lane[slot]);
            const float s = __bfloat162float(sin.lane[slot]);
            // The transpose of the rotation, rounded to bf16 where the composition wrote a
            // bf16 tensor. This is the QK-norm backward's `dY`.
            gy_low[slot] = __bfloat162float(
                __float2bfloat16(rounded_product(g_low, c) + rounded_product(g_high, s)));
            gy_high[slot] = __bfloat162float(
                __float2bfloat16(rounded_product(g_high, c) - rounded_product(g_low, s)));
        }

        const float rstd = inverse_rms(
            head_block_total(square_partial<Fma>(raw_low) + square_partial<Fma>(raw_high),
                             square_partial<Fma>(raw_low + 4) + square_partial<Fma>(raw_high + 4),
                             half_vectors, lane),
            width, eps);

        float stats_first = 0.0f;
        float stats_second = 0.0f;
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            stats_first = accumulate_product<FmaStats>(
                stats_first, __fmul_rn(gy_low[slot], raw_low[slot]), rstd);
            stats_second = accumulate_product<FmaStats>(
                stats_second, __fmul_rn(gy_low[slot + 4], raw_low[slot + 4]), rstd);
        }
        float stats_high_first = 0.0f;
        float stats_high_second = 0.0f;
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            stats_high_first = accumulate_product<FmaStats>(
                stats_high_first, __fmul_rn(gy_high[slot], raw_high[slot]), rstd);
            stats_high_second = accumulate_product<FmaStats>(
                stats_high_second, __fmul_rn(gy_high[slot + 4], raw_high[slot + 4]), rstd);
        }
        const float stats = head_block_total(stats_first + stats_high_first,
                                             stats_second + stats_high_second, half_vectors,
                                             lane);

        const float term = __fmul_rn(1.0f / width, rstd);
        Vec8 out_a;
        Vec8 out_b;
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            float grad_input_low = __fmul_rn(width, gy_low[slot]);
            float grad_input_high = __fmul_rn(width, gy_high[slot]);
            const float scaled_low = __fmul_rn(raw_low[slot], rstd);
            const float scaled_high = __fmul_rn(raw_high[slot], rstd);
            if (FmaGrad) {
                grad_input_low = __fmaf_rn(-scaled_low, stats, grad_input_low);
                grad_input_high = __fmaf_rn(-scaled_high, stats, grad_input_high);
            } else {
                grad_input_low = __fsub_rn(grad_input_low, __fmul_rn(scaled_low, stats));
                grad_input_high = __fsub_rn(grad_input_high, __fmul_rn(scaled_high, stats));
            }
            out_a.lane[slot] = __float2bfloat16(__fmul_rn(grad_input_low, term));
            out_b.lane[slot] = __float2bfloat16(__fmul_rn(grad_input_high, term));
        }
        dx[row * dx_row_vectors + offset] = out_a;
        dx[row * dx_row_vectors + offset + half_vectors] = out_b;
    }
}

// The same reduction tree, emulated by ONE thread over a whole head block, for geometries
// the 128-bit path cannot take (`half` not a multiple of eight, or unaligned buffers). ATen
// still runs its vectorized kernel there - the choice is made on ITS contiguous copy, which
// is always aligned - so this path has to reproduce the same tree rather than a convenient
// one. `slot[i]` is ATen lane `i`'s partial; every partial lands in the first warp because
// `head_dim <= 128`, and the in-place ascending sweep is safe because level `offset` only
// reads slots above the one it writes.
struct AtenTree {
    float slot[32];

    __device__ __forceinline__ void clear() {
#pragma unroll
        for (int index = 0; index < 32; ++index) {
            slot[index] = 0.0f;
        }
    }

    __device__ __forceinline__ float total() {
        for (int offset = 16; offset > 0; offset >>= 1) {
            for (int index = 0; index + offset < 32; ++index) {
                slot[index] += slot[index + offset];
            }
        }
        return slot[0];
    }
};

template <bool Fma>
__global__ void qk_norm_rope_forward_scalar(const __nv_bfloat16 *__restrict__ input,
                                            const __nv_bfloat16 *__restrict__ cosine,
                                            const __nv_bfloat16 *__restrict__ sine,
                                            __nv_bfloat16 *__restrict__ output, int64_t rows,
                                            int64_t length, int64_t blocks, int64_t half,
                                            int64_t input_row_stride,
                                            int64_t output_row_stride, float width,
                                            float eps) {
    const int64_t items = rows * blocks;
    const int64_t count = 2 * half;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
         index < items; index += stride) {
        const int64_t row = index / blocks;
        const int64_t block = index - row * blocks;
        const int64_t position = row % length;
        const __nv_bfloat16 *source = input + row * input_row_stride + block * count;
        __nv_bfloat16 *target = output + row * output_row_stride + block * count;

        AtenTree tree;
        tree.clear();
        for (int64_t partial = 0; partial < count / 4; ++partial) {
            float accumulator = 0.0f;
            for (int64_t element = 0; element < 4; ++element) {
                accumulator = accumulate_square<Fma>(
                    accumulator, __bfloat162float(source[partial * 4 + element]));
            }
            tree.slot[partial] = accumulator;
        }
        const float rstd = inverse_rms(tree.total(), width, eps);

        for (int64_t pair = 0; pair < half; ++pair) {
            const float y_low = normalized(__bfloat162float(source[pair]), rstd);
            const float y_high = normalized(__bfloat162float(source[half + pair]), rstd);
            const float c = __bfloat162float(cosine[position * half + pair]);
            const float s = __bfloat162float(sine[position * half + pair]);
            target[pair] =
                __float2bfloat16(rounded_product(y_low, c) - rounded_product(y_high, s));
            target[half + pair] =
                __float2bfloat16(rounded_product(y_high, c) + rounded_product(y_low, s));
        }
    }
}

template <bool Fma, bool FmaStats, bool FmaGrad>
__global__ void qk_norm_rope_backward_scalar(
    const __nv_bfloat16 *__restrict__ grad, const __nv_bfloat16 *__restrict__ input,
    const __nv_bfloat16 *__restrict__ cosine, const __nv_bfloat16 *__restrict__ sine,
    __nv_bfloat16 *__restrict__ dx, int64_t rows, int64_t length, int64_t blocks,
    int64_t half, int64_t grad_row_stride, int64_t input_row_stride, int64_t dx_row_stride,
    float width, float eps) {
    const int64_t items = rows * blocks;
    const int64_t count = 2 * half;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
         index < items; index += stride) {
        const int64_t row = index / blocks;
        const int64_t block = index - row * blocks;
        const int64_t position = row % length;
        const __nv_bfloat16 *upstream = grad + row * grad_row_stride + block * count;
        const __nv_bfloat16 *source = input + row * input_row_stride + block * count;
        __nv_bfloat16 *target = dx + row * dx_row_stride + block * count;

        AtenTree squares;
        squares.clear();
        for (int64_t partial = 0; partial < count / 4; ++partial) {
            float accumulator = 0.0f;
            for (int64_t element = 0; element < 4; ++element) {
                accumulator = accumulate_square<Fma>(
                    accumulator, __bfloat162float(source[partial * 4 + element]));
            }
            squares.slot[partial] = accumulator;
        }
        const float rstd = inverse_rms(squares.total(), width, eps);

        // `gy` recovered per element, twice: once for the statistic and once for the
        // gradient. Two evaluations of four multiplies beat a `2*half` scratch array in a
        // path that exists for correctness rather than throughput.
        const auto grad_normalized = [&](int64_t element) {
            const bool is_low = element < half;
            const int64_t pair = is_low ? element : element - half;
            const float g_low = __bfloat162float(upstream[pair]);
            const float g_high = __bfloat162float(upstream[half + pair]);
            const float c = __bfloat162float(cosine[position * half + pair]);
            const float s = __bfloat162float(sine[position * half + pair]);
            return __bfloat162float(__float2bfloat16(
                is_low ? rounded_product(g_low, c) + rounded_product(g_high, s)
                       : rounded_product(g_high, c) - rounded_product(g_low, s)));
        };

        AtenTree stats;
        stats.clear();
        for (int64_t partial = 0; partial < count / 4; ++partial) {
            float accumulator = 0.0f;
            for (int64_t element = 0; element < 4; ++element) {
                const int64_t at = partial * 4 + element;
                accumulator = accumulate_product<FmaStats>(
                    accumulator,
                    __fmul_rn(grad_normalized(at), __bfloat162float(source[at])), rstd);
            }
            stats.slot[partial] = accumulator;
        }
        const float statistic = stats.total();

        const float term = __fmul_rn(1.0f / width, rstd);
        for (int64_t element = 0; element < count; ++element) {
            float grad_input = __fmul_rn(width, grad_normalized(element));
            const float scaled = __fmul_rn(__bfloat162float(source[element]), rstd);
            grad_input = FmaGrad ? __fmaf_rn(-scaled, statistic, grad_input)
                                 : __fsub_rn(grad_input, __fmul_rn(scaled, statistic));
            target[element] = __float2bfloat16(__fmul_rn(grad_input, term));
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

// The 128-bit path additionally needs `V = half/8` to be a POWER OF TWO. ATen's tree has
// offsets 16..1 only, so the level that pairs a thread's low vector with its own high
// vector exists exactly when `2V` is one of them; at `head_dim = 48` (`V = 3`) there is no
// such level and the butterfly would reassociate the sum. Those geometries take the
// emulated path, which reproduces the tree for any `head_dim`.
bool power_of_two(int64_t value) { return value > 0 && (value & (value - 1)) == 0; }

template <bool Fma>
int launch_qk_norm_rope_forward(const void *input, const void *cosine, const void *sine,
                                void *output, int64_t rows, int64_t length, int64_t blocks,
                                int64_t half, int64_t input_row_stride,
                                int64_t output_row_stride, float width, float eps,
                                void *stream) {
    cudaStream_t handle = static_cast<cudaStream_t>(stream);
    const bool vectorizable = half % 8 == 0 && power_of_two(half / 8) &&
                              input_row_stride % 8 == 0 && output_row_stride % 8 == 0 &&
                              aligned16(input) && aligned16(output) && aligned16(cosine) &&
                              aligned16(sine);
    if (vectorizable) {
        const int64_t half_vectors = half / 8;
        const int64_t items = rows * blocks * half_vectors;
        qk_norm_rope_forward_vec<Fma><<<grid_for(items), kThreads, 0, handle>>>(
            static_cast<const Vec8 *>(input), static_cast<const Vec8 *>(cosine),
            static_cast<const Vec8 *>(sine), static_cast<Vec8 *>(output), rows, length,
            blocks, half_vectors, input_row_stride / 8, output_row_stride / 8, width, eps);
    } else {
        const int64_t items = rows * blocks;
        qk_norm_rope_forward_scalar<Fma><<<grid_for(items), kThreads, 0, handle>>>(
            static_cast<const __nv_bfloat16 *>(input),
            static_cast<const __nv_bfloat16 *>(cosine),
            static_cast<const __nv_bfloat16 *>(sine),
            static_cast<__nv_bfloat16 *>(output), rows, length, blocks, half,
            input_row_stride, output_row_stride, width, eps);
    }
    return static_cast<int>(cudaGetLastError());
}

template <bool Fma, bool FmaStats, bool FmaGrad>
int launch_qk_norm_rope_backward(const void *grad, const void *input, const void *cosine,
                                 const void *sine, void *dx, int64_t rows, int64_t length,
                                 int64_t blocks, int64_t half, int64_t grad_row_stride,
                                 int64_t input_row_stride, int64_t dx_row_stride,
                                 float width, float eps, void *stream) {
    cudaStream_t handle = static_cast<cudaStream_t>(stream);
    const bool vectorizable = half % 8 == 0 && power_of_two(half / 8) &&
                              grad_row_stride % 8 == 0 && input_row_stride % 8 == 0 &&
                              dx_row_stride % 8 == 0 && aligned16(grad) &&
                              aligned16(input) && aligned16(dx) && aligned16(cosine) &&
                              aligned16(sine);
    if (vectorizable) {
        const int64_t half_vectors = half / 8;
        const int64_t items = rows * blocks * half_vectors;
        qk_norm_rope_backward_vec<Fma, FmaStats, FmaGrad>
            <<<grid_for(items), kThreads, 0, handle>>>(
                static_cast<const Vec8 *>(grad), static_cast<const Vec8 *>(input),
                static_cast<const Vec8 *>(cosine), static_cast<const Vec8 *>(sine),
                static_cast<Vec8 *>(dx), rows, length, blocks, half_vectors,
                grad_row_stride / 8, input_row_stride / 8, dx_row_stride / 8, width, eps);
    } else {
        const int64_t items = rows * blocks;
        qk_norm_rope_backward_scalar<Fma, FmaStats, FmaGrad>
            <<<grid_for(items), kThreads, 0, handle>>>(
                static_cast<const __nv_bfloat16 *>(grad),
                static_cast<const __nv_bfloat16 *>(input),
                static_cast<const __nv_bfloat16 *>(cosine),
                static_cast<const __nv_bfloat16 *>(sine),
                static_cast<__nv_bfloat16 *>(dx), rows, length, blocks, half,
                grad_row_stride, input_row_stride, dx_row_stride, width, eps);
    }
    return static_cast<int>(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------
// Fused candle geometry and Gaussian NLL.
//
// This one is not a two-op fusion: it is a forty-eight-op elementwise chain over the
// 147 M-element head space, and every op in it was its own full-size fp32 pass. What it
// does NOT fuse is the twelve `dot`s the chain feeds. A reduction's summation tree is
// cuBLAS's, not ours, so the kernel writes exactly the twelve fp32 vectors those `dot`s
// consume and ATen still performs them - the loss value and the MSE keep the bits they had.
// The saving is the forty-eight intermediates that never cross HBM, and a backward that
// recomputes the chain from `head` instead of reading twelve retained tensors.
// ---------------------------------------------------------------------------------------

// Fields of the `rounding` selector. Each one is a place where the fp32 form ATen's own
// build emitted is not deducible from the mathematics, only measured: which of two
// associations a backward kernel was written with, and whether `nvcc` contracted a
// multiply-add that changes the last bit. `FK_LOSS_GEOMETRY_ROUNDING` in `lib.rs` carries
// the measured value and `loss_geometry_rounding_is_the_measured_form` is the measurement.
constexpr int kLossDivTrue = 1;       // bit 0: `x / ln2` rather than `x * (1/ln2)`
constexpr int kLossTanhFma = 2;       // bit 1: tanh backward contracts `1 - y·y`
constexpr int kLossSigmoidSwap = 4;   // bit 2: sigmoid backward as `(g·y)·(1-y)`
constexpr int kLossSoftplusShift = 3; // bits 3-4: softplus backward's association
constexpr int kLossSoftplusMask = 3;

// Explicit round-to-nearest primitives, not `*`/`+`/`-`//`/`. Every one of these operations
// was a SEPARATE ATen kernel, so the composition had no opportunity to contract a product
// and a sum into one `fma` - and `nvcc` contracts by default, which would silently move the
// last bit of a fused chain. Writing the intrinsics is how the fusion stays bit-identical
// instead of merely more accurate.
__device__ __forceinline__ float fk_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float fk_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float fk_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float fk_div(float a, float b) { return __fdiv_rn(a, b); }

// `softplus` with ATen's defaults: `(x·β) > threshold ? x : log1p(exp(x·β))/β` at β = 1,
// threshold = 20. Both `·1` and `/1` are exact, so they are not written.
__device__ __forceinline__ float fk_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

// ATen's fp32 sigmoid is `1/(1+exp(-x))` in the opmath type, not a `tanh` identity.
__device__ __forceinline__ float fk_sigmoid(float x) {
    return fk_div(1.0f, fk_add(1.0f, expf(-x)));
}

// `div` by a HOST scalar on CUDA multiplies by the fp32 reciprocal - ATen computes
// `1/scalar` once in the opmath type and removes the operand - while `div` by a device
// tensor is a true division. `1/ln2` is not exactly representable, so the two disagree, and
// which one the composition used is a measured fact.
__device__ __forceinline__ float fk_scale_ln2(float value, float ln2, float inverse_ln2,
                                              int rounding) {
    return (rounding & kLossDivTrue) ? fk_div(value, ln2) : fk_mul(value, inverse_ln2);
}

__device__ __forceinline__ float fk_tanh_backward(float grad, float output, int rounding) {
    return (rounding & kLossTanhFma)
               ? fk_mul(grad, __fmaf_rn(-output, output, 1.0f))
               : fk_mul(grad, fk_sub(1.0f, fk_mul(output, output)));
}

__device__ __forceinline__ float fk_sigmoid_backward(float grad, float output,
                                                     int rounding) {
    return (rounding & kLossSigmoidSwap)
               ? fk_mul(fk_mul(grad, output), fk_sub(1.0f, output))
               : fk_mul(fk_mul(grad, fk_sub(1.0f, output)), output);
}

__device__ __forceinline__ float fk_softplus_backward(float grad, float input,
                                                      int rounding) {
    if (input > 20.0f) {
        return grad;
    }
    const float z = expf(input);
    switch ((rounding >> kLossSoftplusShift) & kLossSoftplusMask) {
    case 1:
        return fk_mul(grad, fk_div(z, fk_add(z, 1.0f)));
    case 2:
        return fk_mul(fk_div(grad, fk_add(z, 1.0f)), z);
    default:
        return fk_div(fk_mul(grad, z), fk_add(z, 1.0f));
    }
}

// The candle geometry for one (origin, bar): the σ-scaled close, the relative range, the
// two positions inside it and the three offsets that place low/high/open. `channel` points
// at this element's coordinate-0 lane; consecutive channels are `horizon` apart.
struct LossGeometry {
    float close, relative_range, position_low, position_open;
    float low, high, open, precision;
    float sigma, range, horizon_scale;
    float sigmoid_low, sigmoid_open;
};

__device__ __forceinline__ LossGeometry
loss_geometry_of(const __nv_bfloat16 *__restrict__ channel, int64_t horizon, float sigma,
                 float range, float horizon_scale, float weighted_mask,
                 float inverse_horizon, float gain, float ln2, float inverse_ln2,
                 int rounding) {
    LossGeometry geometry;
    geometry.sigma = sigma;
    geometry.range = range;
    geometry.horizon_scale = horizon_scale;
    geometry.close = fk_mul(__bfloat162float(channel[0]), horizon_scale);
    const float softplus = fk_softplus(__bfloat162float(channel[horizon]));
    geometry.relative_range =
        fk_scale_ln2(fk_mul(softplus, range), ln2, inverse_ln2, rounding);
    geometry.sigmoid_low = fk_sigmoid(__bfloat162float(channel[2 * horizon]));
    geometry.position_low = fk_mul(geometry.sigmoid_low, geometry.relative_range);
    geometry.low =
        fk_sub(geometry.close, fk_div(log1pf(geometry.position_low), sigma));
    geometry.high = fk_add(geometry.low, fk_div(log1pf(geometry.relative_range), sigma));
    geometry.sigmoid_open = fk_sigmoid(__bfloat162float(channel[3 * horizon]));
    geometry.position_open = fk_mul(geometry.sigmoid_open, geometry.relative_range);
    geometry.open = fk_add(geometry.low, fk_div(log1pf(geometry.position_open), sigma));
    geometry.precision = fk_mul(weighted_mask, inverse_horizon);
    (void)gain;
    return geometry;
}

// `targets` is addressed by STRIDE, not assumed dense. The real path builds it with
// `unfold` and TensorIterator allocates the arithmetic below it in the inputs' own
// permuted layout, so what arrives is channel-innermost - sizes
// `[rows, origins, 4, horizon]`, strides `[origins·4·horizon, 4·horizon, 1, 4]`. The
// composition tolerated that silently because every pointwise op it called accepts strided
// inputs; a fused kernel has to say so. Three extra int64 arguments and one multiply per
// channel read cost nothing, where a `.contiguous()` would copy 295 MB and read it back.
template <bool Decoupled>
__global__ void loss_geometry_forward_kernel(
    const __nv_bfloat16 *__restrict__ head, const float *__restrict__ targets,
    const float *__restrict__ weighted_mask, const float *__restrict__ sigma,
    const float *__restrict__ range, const float *__restrict__ horizon_scale,
    const float *__restrict__ inverse_horizon, const float *__restrict__ log_scale_gain,
    float *__restrict__ close_out, float *__restrict__ workspace, int64_t elements,
    int64_t horizon, int64_t target_token_stride, int64_t target_channel_stride,
    int64_t target_bar_stride, float cap, float ln2, float inverse_ln2, int rounding) {
    const float gain = log_scale_gain[0];
    const float slope = -2.0f * cap;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
         index < elements; index += stride) {
        const int64_t token = index / horizon;
        const int64_t bar = index - token * horizon;
        const __nv_bfloat16 *channel = head + token * 8 * horizon + bar;
        const LossGeometry geometry = loss_geometry_of(
            channel, horizon, sigma[token], range[token], horizon_scale[bar],
            weighted_mask[index], inverse_horizon[bar], gain, ln2, inverse_ln2, rounding);
        const float prediction[4] = {geometry.open, geometry.high, geometry.low,
                                     geometry.close};
        const float *target =
            targets + token * target_token_stride + bar * target_bar_stride;
        close_out[index] = geometry.close;
        if constexpr (Decoupled) {
            workspace[12 * elements + index] = geometry.precision;
        }
#pragma unroll
        for (int channel_index = 0; channel_index < 4; ++channel_index) {
            const float scale = tanhf(
                fk_mul(__bfloat162float(channel[(4 + channel_index) * horizon]), gain));
            const float weight =
                fk_mul(expf(fk_mul(scale, slope)), geometry.precision);
            const float difference =
                fk_sub(target[channel_index * target_channel_stride],
                       prediction[channel_index]);
            workspace[channel_index * elements + index] = fk_mul(difference, difference);
            workspace[(4 + channel_index) * elements + index] = weight;
            workspace[(8 + channel_index) * elements + index] = scale;
        }
    }
}

// The transpose of the whole chain. The accumulation ORDERS below are not free choices:
// `low` reaches three consumers and `relative_range` three, and the autograd engine sums a
// buffer's contributions in DESCENDING node-creation order (it pops the highest sequence
// number first). So `low` accumulates open, then the channel-2 residual, then high; and
// `relative_range` accumulates the open position, then the full-range `log1p`, then the low
// position. Any other grouping differs in the last bit, which
// `composed_loss_geometry_accumulates_in_descending_creation_order` pins.
template <bool Decoupled>
__global__ void loss_geometry_backward_kernel(
    const __nv_bfloat16 *__restrict__ head, const float *__restrict__ targets,
    const float *__restrict__ weighted_mask, const float *__restrict__ sigma,
    const float *__restrict__ range, const float *__restrict__ horizon_scale,
    const float *__restrict__ inverse_horizon, const float *__restrict__ log_scale_gain,
    const float *__restrict__ grad_terms, const float *__restrict__ grad_close,
    __nv_bfloat16 *__restrict__ grad_head, int64_t elements, int64_t horizon,
    int64_t target_token_stride, int64_t target_channel_stride, int64_t target_bar_stride,
    float cap, float ln2, float inverse_ln2, int rounding) {
    const float gain = log_scale_gain[0];
    const float slope = -2.0f * cap;
    constexpr int terms_per_channel = Decoupled ? 3 : 2;
    float term[4 * terms_per_channel];
#pragma unroll
    for (int i = 0; i < 4 * terms_per_channel; ++i) {
        term[i] = grad_terms == nullptr ? 0.0f : grad_terms[i];
    }
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
         index < elements; index += stride) {
        const int64_t token = index / horizon;
        const int64_t bar = index - token * horizon;
        if constexpr (Decoupled) {
            // With no consumer of `terms`, even a nonfinite residual or scale is outside
            // the gradient graph. Multiplying its derivative by zero would manufacture
            // NaNs instead of differentiating the returned close coordinate alone.
            if (grad_terms == nullptr) {
                __nv_bfloat16 *out = grad_head + token * 8 * horizon + bar;
                out[0] = __float2bfloat16(
                    grad_close == nullptr ? 0.0f
                                          : fk_mul(grad_close[index], horizon_scale[bar]));
#pragma unroll
                for (int channel_index = 1; channel_index < 8; ++channel_index) {
                    out[channel_index * horizon] = __float2bfloat16(0.0f);
                }
                continue;
            }
        }
        const __nv_bfloat16 *channel = head + token * 8 * horizon + bar;
        const float mask = weighted_mask[index];
        const LossGeometry geometry =
            loss_geometry_of(channel, horizon, sigma[token], range[token],
                             horizon_scale[bar], mask, inverse_horizon[bar], gain, ln2,
                             inverse_ln2, rounding);
        const float prediction[4] = {geometry.open, geometry.high, geometry.low,
                                     geometry.close};
        const float *target =
            targets + token * target_token_stride + bar * target_bar_stride;
        __nv_bfloat16 *out = grad_head + token * 8 * horizon + bar;
        float grad_prediction[4];
#pragma unroll
        for (int channel_index = 0; channel_index < 4; ++channel_index) {
            const float raw = __bfloat162float(channel[(4 + channel_index) * horizon]);
            const float scale = tanhf(fk_mul(raw, gain));
            const float exponential = expf(fk_mul(scale, slope));
            const float weight = fk_mul(exponential, geometry.precision);
            const float difference =
                fk_sub(target[channel_index * target_channel_stride],
                       prediction[channel_index]);
            const float square = fk_mul(difference, difference);
            // `dot`'s backward is `grad · other`, and the reference's `·½` and `·cap`
            // scalar multiplies ride on the term gradient first.
            const int base = terms_per_channel * channel_index;
            const float grad_dot = fk_mul(term[base], 0.5f);
            // Decoupling removes the learned precision from ONLY the residual branch.
            // The scale branch consumes the same residual VALUE with no derivative back
            // to the coordinates, exactly as `square.detach()` in the composition.
            const float grad_square =
                fk_mul(grad_dot, Decoupled ? geometry.precision : weight);
            const float grad_scale_dot =
                Decoupled ? fk_mul(term[base + 1], 0.5f) : grad_dot;
            const float grad_weight = fk_mul(grad_scale_dot, square);
            // `scale` has two consumers - the precision weight and its own `dot` against
            // the weighted mask - and two addends commute exactly, so this pair needs no
            // measured order.
            const float grad_scale = fk_add(
                fk_mul(fk_mul(term[base + terms_per_channel - 1], cap), mask),
                fk_mul(fk_mul(fk_mul(grad_weight, geometry.precision), exponential),
                       slope));
            out[(4 + channel_index) * horizon] = __float2bfloat16(
                fk_mul(fk_tanh_backward(grad_scale, scale, rounding), gain));
            // `square` is `difference · difference`, so the two identical products the
            // multiply's backward accumulates double exactly, and the `neg` the residual's
            // backward applies carries the sign of an exact zero: at an invalid origin
            // `weight` is `+0`, this sum is `+0` and the reference's gradient is `-0`.
            const float half = fk_mul(grad_square, difference);
            grad_prediction[channel_index] = -fk_add(half, half);
        }
        const float grad_low = fk_add(fk_add(grad_prediction[0], grad_prediction[2]),
                                      grad_prediction[1]);
        const float grad_position_open =
            fk_div(fk_div(grad_prediction[0], geometry.sigma),
                   fk_add(geometry.position_open, 1.0f));
        const float grad_position_low = fk_div(
            fk_div(-grad_low, geometry.sigma), fk_add(geometry.position_low, 1.0f));
        const float grad_relative_range = fk_add(
            fk_add(fk_mul(grad_position_open, geometry.sigmoid_open),
                   fk_div(fk_div(grad_prediction[1], geometry.sigma),
                          fk_add(geometry.relative_range, 1.0f))),
            fk_mul(grad_position_low, geometry.sigmoid_low));
        const float grad_softplus = fk_mul(
            fk_scale_ln2(grad_relative_range, ln2, inverse_ln2, rounding), geometry.range);
        out[horizon] = __float2bfloat16(fk_softplus_backward(
            grad_softplus, __bfloat162float(channel[horizon]), rounding));
        out[2 * horizon] = __float2bfloat16(fk_sigmoid_backward(
            fk_mul(grad_position_low, geometry.relative_range), geometry.sigmoid_low,
            rounding));
        out[3 * horizon] = __float2bfloat16(fk_sigmoid_backward(
            fk_mul(grad_position_open, geometry.relative_range), geometry.sigmoid_open,
            rounding));
        // `close` reaches `low` and the channel-3 residual, which commute; an external
        // consumer of the mean coordinate is a THIRD addend, and it is added last because
        // its node is created first and therefore runs last.
        float grad_close_total = fk_add(grad_prediction[3], grad_low);
        if (grad_close != nullptr) {
            grad_close_total = fk_add(grad_close_total, grad_close[index]);
        }
        out[0] = __float2bfloat16(fk_mul(grad_close_total, geometry.horizon_scale));
    }
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

// `rounding` selects the fp32 contraction forms, bit 0 for the squared accumulation, bit 1
// for the gradient statistic and bit 2 for the gradient's subtraction. It is a parameter
// only so that the discovery test can prove which one ATen's build emitted and that the
// other seven disagree; the model path passes `FK_QK_NORM_ROPE_ROUNDING`.
extern "C" int fk_qk_norm_rope_forward(const void *input, const void *cosine,
                                       const void *sine, void *output, int64_t rows,
                                       int64_t length, int64_t blocks, int64_t half,
                                       int64_t input_row_stride, int64_t output_row_stride,
                                       float width, float eps, int rounding, void *stream) {
    if ((rounding & 1) != 0) {
        return launch_qk_norm_rope_forward<true>(input, cosine, sine, output, rows, length,
                                                 blocks, half, input_row_stride,
                                                 output_row_stride, width, eps, stream);
    }
    return launch_qk_norm_rope_forward<false>(input, cosine, sine, output, rows, length,
                                              blocks, half, input_row_stride,
                                              output_row_stride, width, eps, stream);
}

extern "C" int fk_qk_norm_rope_backward(const void *grad, const void *input,
                                        const void *cosine, const void *sine, void *dx,
                                        int64_t rows, int64_t length, int64_t blocks,
                                        int64_t half, int64_t grad_row_stride,
                                        int64_t input_row_stride, int64_t dx_row_stride,
                                        float width, float eps, int rounding,
                                        void *stream) {
#define FK_LAUNCH(squares, stats, gradient)                                                \
    launch_qk_norm_rope_backward<squares, stats, gradient>(                                \
        grad, input, cosine, sine, dx, rows, length, blocks, half, grad_row_stride,        \
        input_row_stride, dx_row_stride, width, eps, stream)
    switch (rounding & 7) {
    case 0:
        return FK_LAUNCH(false, false, false);
    case 1:
        return FK_LAUNCH(true, false, false);
    case 2:
        return FK_LAUNCH(false, true, false);
    case 3:
        return FK_LAUNCH(true, true, false);
    case 4:
        return FK_LAUNCH(false, false, true);
    case 5:
        return FK_LAUNCH(true, false, true);
    case 6:
        return FK_LAUNCH(false, true, true);
    default:
        return FK_LAUNCH(true, true, true);
    }
#undef FK_LAUNCH
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

template <bool Decoupled>
static int launch_loss_geometry_forward(const void *head, const void *targets,
                         const void *weighted_mask, const void *sigma, const void *range,
                         const void *horizon_scale, const void *inverse_horizon,
                         const void *log_scale_gain, void *close, void *workspace,
                         int64_t tokens, int64_t horizon, int64_t target_token_stride,
                         int64_t target_channel_stride, int64_t target_bar_stride,
                         float cap, float ln2, float inverse_ln2, int rounding,
                         void *stream) {
    const int64_t elements = tokens * horizon;
    if (elements <= 0) {
        return static_cast<int>(cudaSuccess);
    }
    loss_geometry_forward_kernel<Decoupled><<<grid_for(elements), kThreads, 0,
                                              static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16 *>(head), static_cast<const float *>(targets),
        static_cast<const float *>(weighted_mask), static_cast<const float *>(sigma),
        static_cast<const float *>(range), static_cast<const float *>(horizon_scale),
        static_cast<const float *>(inverse_horizon),
        static_cast<const float *>(log_scale_gain), static_cast<float *>(close),
        static_cast<float *>(workspace), elements, horizon, target_token_stride,
        target_channel_stride, target_bar_stride, cap, ln2, inverse_ln2, rounding);
    return static_cast<int>(cudaGetLastError());
}

template <bool Decoupled>
static int launch_loss_geometry_backward(const void *head, const void *targets,
                          const void *weighted_mask, const void *sigma, const void *range,
                          const void *horizon_scale, const void *inverse_horizon,
                          const void *log_scale_gain, const void *grad_terms,
                          const void *grad_close, void *grad_head, int64_t tokens,
                          int64_t horizon, int64_t target_token_stride,
                          int64_t target_channel_stride, int64_t target_bar_stride,
                          float cap, float ln2, float inverse_ln2, int rounding,
                          void *stream) {
    const int64_t elements = tokens * horizon;
    if (elements <= 0) {
        return static_cast<int>(cudaSuccess);
    }
    loss_geometry_backward_kernel<Decoupled><<<grid_for(elements), kThreads, 0,
                                               static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16 *>(head), static_cast<const float *>(targets),
        static_cast<const float *>(weighted_mask), static_cast<const float *>(sigma),
        static_cast<const float *>(range), static_cast<const float *>(horizon_scale),
        static_cast<const float *>(inverse_horizon),
        static_cast<const float *>(log_scale_gain),
        static_cast<const float *>(grad_terms), static_cast<const float *>(grad_close),
        static_cast<__nv_bfloat16 *>(grad_head), elements, horizon, target_token_stride,
        target_channel_stride, target_bar_stride, cap, ln2, inverse_ln2, rounding);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int
fk_loss_geometry_forward(const void *head, const void *targets,
                         const void *weighted_mask, const void *sigma, const void *range,
                         const void *horizon_scale, const void *inverse_horizon,
                         const void *log_scale_gain, void *close, void *workspace,
                         int64_t tokens, int64_t horizon, int64_t target_token_stride,
                         int64_t target_channel_stride, int64_t target_bar_stride,
                         float cap, float ln2, float inverse_ln2, int rounding,
                         int decoupled, void *stream) {
    if (decoupled) {
        return launch_loss_geometry_forward<true>(
            head, targets, weighted_mask, sigma, range, horizon_scale, inverse_horizon,
            log_scale_gain, close, workspace, tokens, horizon, target_token_stride,
            target_channel_stride, target_bar_stride, cap, ln2, inverse_ln2, rounding,
            stream);
    }
    return launch_loss_geometry_forward<false>(
        head, targets, weighted_mask, sigma, range, horizon_scale, inverse_horizon,
        log_scale_gain, close, workspace, tokens, horizon, target_token_stride,
        target_channel_stride, target_bar_stride, cap, ln2, inverse_ln2, rounding, stream);
}

extern "C" int
fk_loss_geometry_backward(const void *head, const void *targets,
                          const void *weighted_mask, const void *sigma, const void *range,
                          const void *horizon_scale, const void *inverse_horizon,
                          const void *log_scale_gain, const void *grad_terms,
                          const void *grad_close, void *grad_head, int64_t tokens,
                          int64_t horizon, int64_t target_token_stride,
                          int64_t target_channel_stride, int64_t target_bar_stride,
                          float cap, float ln2, float inverse_ln2, int rounding,
                          int decoupled, void *stream) {
    if (decoupled) {
        return launch_loss_geometry_backward<true>(
            head, targets, weighted_mask, sigma, range, horizon_scale, inverse_horizon,
            log_scale_gain, grad_terms, grad_close, grad_head, tokens, horizon,
            target_token_stride, target_channel_stride, target_bar_stride, cap, ln2,
            inverse_ln2, rounding, stream);
    }
    return launch_loss_geometry_backward<false>(
        head, targets, weighted_mask, sigma, range, horizon_scale, inverse_horizon,
        log_scale_gain, grad_terms, grad_close, grad_head, tokens, horizon,
        target_token_stride, target_channel_stride, target_bar_stride, cap, ln2,
        inverse_ln2, rounding, stream);
}
