#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/convolution_backward.h>
#include <c10/util/Optional.h>
#include <c10/util/Type.h>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kMaxDState = 256;

__device__ inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ inline float softplus_f(float x) {
    if (x > 20.0f) {
        return x;
    }
    return log1pf(expf(x));
}

__device__ inline float silu_f(float x) {
    return x * sigmoid_f(x);
}

__device__ inline float silu_grad_f(float x) {
    float sig = sigmoid_f(x);
    return sig * (1.0f + x * (1.0f - sig));
}

template <typename scalar_t>
__device__ inline float to_float(scalar_t v) {
    return static_cast<float>(v);
}

template <typename scalar_t>
__device__ inline scalar_t from_float(float v) {
    return static_cast<scalar_t>(v);
}

template <typename scalar_t>
__global__ void check_finite_kernel(const scalar_t* data, int64_t total, int* flags) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    float v = to_float(data[idx]);
    if (isnan(v)) {
        atomicExch(&flags[0], 1);
    } else if (isinf(v)) {
        atomicExch(&flags[1], 1);
    }
}

template <typename scalar_t>
void check_tensor_finite(const torch::Tensor& t, const char* tag) {
    if (!t.defined() || t.numel() == 0) {
        return;
    }
    auto flags = torch::zeros({2}, t.options().dtype(torch::kInt32));
    const int threads = 256;
    int64_t total = t.numel();
    int blocks = (total + threads - 1) / threads;
    check_finite_kernel<scalar_t><<<blocks, threads>>>(
        t.data_ptr<scalar_t>(),
        total,
        flags.data_ptr<int>());
    auto cpu = flags.cpu();
    auto acc = cpu.accessor<int, 1>();
    if (acc[0] || acc[1]) {
        std::cerr << "mamba_fused_debug " << tag << " nan=" << (acc[0] != 0)
                  << " inf=" << (acc[1] != 0) << " shape=" << t.sizes() << "\n";
    }
}

torch::Tensor depthwise_conv1d_silu(
    const torch::Tensor& zxbcdt,
    const torch::Tensor& conv_w,
    const torch::Tensor& conv_b,
    int64_t d_mlp,
    int64_t d_ssm) {
    int64_t conv_dim = conv_w.size(0);
    int64_t conv_kernel = conv_w.size(1);
    int64_t offset_xbc = 2 * d_mlp + d_ssm;
    auto xbc = zxbcdt.narrow(2, offset_xbc, conv_dim).transpose(1, 2).contiguous();
    auto weight = conv_w.view({conv_dim, 1, conv_kernel});
    auto bias = conv_b.defined() && conv_b.numel() > 0 ? conv_b : torch::Tensor();
    c10::optional<at::Tensor> bias_opt = bias.defined() ? c10::optional<at::Tensor>(bias) : c10::nullopt;
    std::vector<int64_t> stride = {1};
    std::vector<int64_t> padding = {0};
    std::vector<int64_t> dilation = {1};
    auto xbc_pad = at::constant_pad_nd(xbc, {conv_kernel - 1, 0});
    auto conv = at::conv1d(xbc_pad, weight, bias_opt, stride, padding, dilation, conv_dim);
    auto conv_fp32 = conv.to(torch::kFloat);
    return at::silu(conv_fp32).transpose(1, 2).contiguous();
}

torch::Tensor depthwise_conv1d_pre(
    const torch::Tensor& zxbcdt,
    const torch::Tensor& conv_w,
    const torch::Tensor& conv_b,
    int64_t d_mlp,
    int64_t d_ssm) {
    int64_t conv_dim = conv_w.size(0);
    int64_t conv_kernel = conv_w.size(1);
    int64_t offset_xbc = 2 * d_mlp + d_ssm;
    auto xbc = zxbcdt.narrow(2, offset_xbc, conv_dim).transpose(1, 2).contiguous();
    auto weight = conv_w.view({conv_dim, 1, conv_kernel});
    auto bias = conv_b.defined() && conv_b.numel() > 0 ? conv_b : torch::Tensor();
    c10::optional<at::Tensor> bias_opt = bias.defined() ? c10::optional<at::Tensor>(bias) : c10::nullopt;
    std::vector<int64_t> stride = {1};
    std::vector<int64_t> padding = {0};
    std::vector<int64_t> dilation = {1};
    auto xbc_pad = at::constant_pad_nd(xbc, {conv_kernel - 1, 0});
    auto conv = at::conv1d(xbc_pad, weight, bias_opt, stride, padding, dilation, conv_dim);
    return conv.to(torch::kFloat).transpose(1, 2).contiguous();
}

__global__ void state_passing_fwd_kernel(
    const float* chunk_state,
    const float* exp_a_last,
    const float* initial_state,
    float* state_in,
    float* final_state,
    int64_t batch,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t num_chunks) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * nheads * headdim * d_state;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t n = tmp % d_state;
    tmp /= d_state;
    int64_t p = tmp % headdim;
    tmp /= headdim;
    int64_t h = tmp % nheads;
    int64_t b = tmp / nheads;

    int64_t base = ((b * nheads + h) * headdim + p) * d_state + n;
    float state = initial_state[base];
    for (int64_t c = 0; c < num_chunks; ++c) {
        int64_t cs_idx = ((((b * num_chunks + c) * nheads + h) * headdim + p) * d_state + n);
        state_in[cs_idx] = state;
        float decay = exp_a_last[(b * nheads + h) * num_chunks + c];
        state = decay * (state + chunk_state[cs_idx]);
    }
    final_state[base] = state;
}

__global__ void state_passing_bwd_kernel(
    const float* chunk_state,
    const float* state_in,
    const float* exp_a_last,
    const float* dstate_in,
    const float* grad_final_state,
    float* dchunk_state,
    float* ddA,
    float* dstate0,
    int64_t batch,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t seqlen,
    int64_t chunk_size,
    int64_t num_chunks) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * nheads * headdim * d_state;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t n = tmp % d_state;
    tmp /= d_state;
    int64_t p = tmp % headdim;
    tmp /= headdim;
    int64_t h = tmp % nheads;
    int64_t b = tmp / nheads;

    int64_t base = ((b * nheads + h) * headdim + p) * d_state + n;
    float gstate_out = grad_final_state[base];
    for (int64_t c = num_chunks - 1; c >= 0; --c) {
        int64_t last_t = (c + 1) * chunk_size;
        if (last_t > seqlen) {
            last_t = seqlen;
        }
        last_t -= 1;
        int64_t cs_idx = ((((b * num_chunks + c) * nheads + h) * headdim + p) * d_state + n);
        float exp_a = exp_a_last[(b * nheads + h) * num_chunks + c];
        float s_in = state_in[cs_idx];
        float s_chunk = chunk_state[cs_idx];
        float s_out = exp_a * (s_in + s_chunk);
        atomicAdd(&ddA[(b * nheads + h) * seqlen + last_t], gstate_out * s_out);
        dchunk_state[cs_idx] = gstate_out * exp_a;
        float gstate_in = gstate_out * exp_a + dstate_in[cs_idx];
        gstate_out = gstate_in;
    }
    dstate0[base] = gstate_out;
}

__global__ void batched_gemm_f32_kernel(
    const float* A,
    const float* B,
    float* C,
    int64_t batches,
    int64_t M,
    int64_t N,
    int64_t K) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;

    int64_t b = blockIdx.z;
    int64_t block_m = blockIdx.y * BM;
    int64_t block_n = blockIdx.x * BN;
    int64_t tid_m = threadIdx.y;
    int64_t tid_n = threadIdx.x;

    int64_t base_m = block_m + tid_m * TM;
    int64_t base_n = block_n + tid_n * TN;

    if (b >= batches || base_m >= M || base_n >= N) {
        return;
    }

    const float* A_ptr = A + b * M * K;
    const float* B_ptr = B + b * K * N;
    float* C_ptr = C + b * M * N;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int64_t k0 = 0; k0 < K; k0 += BK) {
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            int64_t row = base_m + i;
            int64_t col = k0 + tid_n;
            As[tid_m * TM + i][tid_n] = (row < M && col < K) ? A_ptr[row * K + col] : 0.0f;
        }
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int64_t row = k0 + tid_m;
            int64_t col = base_n + j;
            Bs[tid_m][tid_n * TN + j] = (row < K && col < N) ? B_ptr[row * N + col] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                float a = As[tid_m * TM + i][k];
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a * Bs[k][tid_n * TN + j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int64_t row = base_m + i;
        if (row >= M) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int64_t col = base_n + j;
            if (col < N) {
                C_ptr[row * N + col] = acc[i][j];
            }
        }
    }
}

torch::Tensor batched_gemm_f32(const torch::Tensor& A, const torch::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "batched_gemm_f32 expects CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "batched_gemm_f32 expects float A");
    TORCH_CHECK(B.scalar_type() == torch::kFloat, "batched_gemm_f32 expects float B");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "batched_gemm_f32 expects 3D tensors");
    TORCH_CHECK(A.size(0) == B.size(0), "batched_gemm_f32 batch mismatch");
    TORCH_CHECK(A.size(2) == B.size(1), "batched_gemm_f32 K mismatch");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "batched_gemm_f32 requires contiguous inputs");

    int64_t batches = A.size(0);
    int64_t M = A.size(1);
    int64_t K = A.size(2);
    int64_t N = B.size(2);

    auto C = torch::zeros({batches, M, N}, A.options());
    dim3 block(16, 16);
    dim3 grid((N + 63) / 64, (M + 63) / 64, batches);
    batched_gemm_f32_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batches,
        M,
        N,
        K);
    return C;
}

torch::Tensor batched_gemm_bmm(const torch::Tensor& A, const torch::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "batched_gemm_bmm expects CUDA tensors");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3, "batched_gemm_bmm expects 3D tensors");
    TORCH_CHECK(A.size(0) == B.size(0), "batched_gemm_bmm batch mismatch");
    TORCH_CHECK(A.size(2) == B.size(1), "batched_gemm_bmm K mismatch");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "batched_gemm_bmm requires contiguous inputs");
    return at::bmm(A, B).to(torch::kFloat);
}

__global__ void bmm_mk_kn_kernel(
    const float* A,
    const float* B,
    float* C,
    int64_t batches,
    int64_t M,
    int64_t K,
    int64_t N) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;

    int64_t b = blockIdx.z;
    int64_t tile_m = blockIdx.y;
    int64_t tile_n = blockIdx.x;
    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    int64_t tid_m = threadIdx.y;
    int64_t tid_n = threadIdx.x;
    int64_t m0 = row_base + tid_m * 4;
    int64_t n0 = col_base + tid_n * 4;

    if (b >= batches || m0 >= M || n0 >= N) {
        return;
    }

    const float* A_ptr = A + b * M * K;
    const float* B_ptr = B + b * K * N;
    float* C_ptr = C + b * M * N;

    __shared__ float A_s[BM][BK];
    __shared__ float B_s[BK][BN];

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int64_t k0 = 0; k0 < K; k0 += BK) {
        int load_idx = threadIdx.y * 16 + threadIdx.x;
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BK;
            int col = idx - row * BK;
            if (row < BM && col < BK) {
                int64_t m = row_base + row;
                int64_t k = k0 + col;
                float val = 0.0f;
                if (m < M && k < K) {
                    val = A_ptr[m * K + k];
                }
                A_s[row][col] = val;
            }
        }
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx - row * BN;
            if (row < BK && col < BN) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                float val = 0.0f;
                if (k < K && n < N) {
                    val = B_ptr[k * N + n];
                }
                B_s[row][col] = val;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = A_s[tid_m * 4 + i][k];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a * B_s[k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t m = m0 + i;
        if (m >= M) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t n = n0 + j;
            if (n < N) {
                C_ptr[m * N + n] = acc[i][j];
            }
        }
    }
}

__global__ void bmm_kt_kn_kernel(
    const float* A_t,
    const float* B,
    float* C,
    int64_t batches,
    int64_t M,
    int64_t K,
    int64_t N) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;

    int64_t b = blockIdx.z;
    int64_t tile_m = blockIdx.y;
    int64_t tile_n = blockIdx.x;
    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    int64_t tid_m = threadIdx.y;
    int64_t tid_n = threadIdx.x;
    int64_t m0 = row_base + tid_m * 4;
    int64_t n0 = col_base + tid_n * 4;

    if (b >= batches || m0 >= M || n0 >= N) {
        return;
    }

    const float* A_ptr = A_t + b * K * M;
    const float* B_ptr = B + b * K * N;
    float* C_ptr = C + b * M * N;

    __shared__ float A_s[BK][BM];
    __shared__ float B_s[BK][BN];

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int64_t k0 = 0; k0 < K; k0 += BK) {
        int load_idx = threadIdx.y * 16 + threadIdx.x;
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BM;
            int col = idx - row * BM;
            if (row < BK && col < BM) {
                int64_t k = k0 + row;
                int64_t m = row_base + col;
                float val = 0.0f;
                if (k < K && m < M) {
                    val = A_ptr[k * M + m];
                }
                A_s[row][col] = val;
            }
        }
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx - row * BN;
            if (row < BK && col < BN) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                float val = 0.0f;
                if (k < K && n < N) {
                    val = B_ptr[k * N + n];
                }
                B_s[row][col] = val;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = A_s[k][tid_m * 4 + i];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a * B_s[k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t m = m0 + i;
        if (m >= M) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t n = n0 + j;
            if (n < N) {
                C_ptr[m * N + n] = acc[i][j];
            }
        }
    }
}

__global__ void bmm_kt_kn_scale_x_kernel(
    const float* A_t,
    const float* X,
    const float* dt,
    const float* dA_cumsum,
    float* C,
    int64_t batches,
    int64_t M,
    int64_t K,
    int64_t N,
    int64_t num_chunks,
    int64_t ngroups,
    int64_t nheads,
    int64_t headdim,
    int64_t chunk_size) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;

    int64_t bg = blockIdx.z;
    int64_t tile_m = blockIdx.y;
    int64_t tile_n = blockIdx.x;
    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    int64_t tid_m = threadIdx.y;
    int64_t tid_n = threadIdx.x;
    int64_t m0 = row_base + tid_m * 4;
    int64_t n0 = col_base + tid_n * 4;

    if (bg >= batches || m0 >= M || n0 >= N) {
        return;
    }

    int64_t b = bg / (num_chunks * ngroups);
    int64_t rem = bg - b * num_chunks * ngroups;
    int64_t c = rem / ngroups;
    int64_t g = rem - c * ngroups;
    int64_t heads_per_group = nheads / ngroups;

    const float* A_ptr = A_t + bg * K * M;
    const float* X_ptr = X + bg * K * N;
    float* C_ptr = C + bg * M * N;

    __shared__ float A_s[BK][BM];
    __shared__ float X_s[BK][BN];

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int64_t k0 = 0; k0 < K; k0 += BK) {
        int load_idx = threadIdx.y * 16 + threadIdx.x;
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BM;
            int col = idx - row * BM;
            if (row < BK && col < BM) {
                int64_t k = k0 + row;
                int64_t m = row_base + col;
                float val = 0.0f;
                if (k < K && m < M) {
                    val = A_ptr[k * M + m];
                }
                A_s[row][col] = val;
            }
        }
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx - row * BN;
            if (row < BK && col < BN) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                float val = 0.0f;
                if (k < K && n < N) {
                    int64_t head_in_group = n / headdim;
                    int64_t head = g * heads_per_group + head_in_group;
                    int64_t dt_idx = ((b * nheads + head) * num_chunks + c) * chunk_size + k;
                    float scale = dt[dt_idx] * expf(fminf(-dA_cumsum[dt_idx], 0.0f));
                    val = X_ptr[k * N + n] * scale;
                }
                X_s[row][col] = val;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = A_s[k][tid_m * 4 + i];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a * X_s[k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t m = m0 + i;
        if (m >= M) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t n = n0 + j;
            if (n < N) {
                C_ptr[m * N + n] = acc[i][j];
            }
        }
    }
}

__global__ void bmm_mk_nk_kernel(
    const float* A,
    const float* B_t,
    float* C,
    int64_t batches,
    int64_t M,
    int64_t K,
    int64_t N) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;

    int64_t b = blockIdx.z;
    int64_t tile_m = blockIdx.y;
    int64_t tile_n = blockIdx.x;
    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    int64_t tid_m = threadIdx.y;
    int64_t tid_n = threadIdx.x;
    int64_t m0 = row_base + tid_m * 4;
    int64_t n0 = col_base + tid_n * 4;

    if (b >= batches || m0 >= M || n0 >= N) {
        return;
    }

    const float* A_ptr = A + b * M * K;
    const float* B_ptr = B_t + b * N * K;
    float* C_ptr = C + b * M * N;

    __shared__ float A_s[BM][BK];
    __shared__ float B_s[BN][BK];

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int64_t k0 = 0; k0 < K; k0 += BK) {
        int load_idx = threadIdx.y * 16 + threadIdx.x;
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BK;
            int col = idx - row * BK;
            if (row < BM && col < BK) {
                int64_t m = row_base + row;
                int64_t k = k0 + col;
                float val = 0.0f;
                if (m < M && k < K) {
                    val = A_ptr[m * K + k];
                }
                A_s[row][col] = val;
            }
        }
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BK;
            int col = idx - row * BK;
            if (row < BN && col < BK) {
                int64_t n = col_base + row;
                int64_t k = k0 + col;
                float val = 0.0f;
                if (n < N && k < K) {
                    val = B_ptr[n * K + k];
                }
                B_s[row][col] = val;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = A_s[tid_m * 4 + i][k];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a * B_s[tid_n * 4 + j][k];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t m = m0 + i;
        if (m >= M) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t n = n0 + j;
            if (n < N) {
                C_ptr[m * N + n] = acc[i][j];
            }
        }
    }
}

__global__ void chunk_scan_fwd_kernel(
    const float* cb,
    const float* x,
    const float* dt,
    const float* dA_cumsum,
    const float* C,
    const float* state_in,
    const float* D,
    int64_t d_has_hdim,
    float* y,
    int64_t batch,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;

    int64_t tile_n = blockIdx.x;
    int64_t tile_m = blockIdx.y;
    int64_t pid = blockIdx.z;
    int64_t b = pid / (num_chunks * nheads);
    int64_t rem = pid - b * num_chunks * nheads;
    int64_t c = rem / nheads;
    int64_t h = rem - c * nheads;
    if (b >= batch) {
        return;
    }

    int64_t heads_per_group = nheads / ngroups;
    int64_t g = h / heads_per_group;

    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    int64_t tid_m = threadIdx.y;
    int64_t tid_n = threadIdx.x;
    int64_t m0 = row_base + tid_m * 4;
    int64_t n0 = col_base + tid_n * 4;

    __shared__ float cb_s[BM][BK];
    __shared__ float x_s[BK][BN];
    __shared__ float c_s[BM][BK];
    __shared__ float st_s[BK][BN];
    __shared__ float dA_m[BM];
    __shared__ float dA_k[BK];
    __shared__ float dt_k[BK];
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    int64_t chunk_offset = (((b * num_chunks + c) * ngroups + g) * chunk_size) * chunk_size;
    int64_t x_offset = ((((b * num_chunks + c) * chunk_size) * nheads + h) * headdim);
    int64_t c_offset = (((b * num_chunks + c) * chunk_size) * ngroups + g) * d_state;
    int64_t state_offset = ((((b * num_chunks + c) * nheads + h) * headdim) * d_state);
    int64_t dt_offset = (((b * nheads + h) * num_chunks + c) * chunk_size);

    for (int64_t k0 = 0; k0 < chunk_size; k0 += BK) {
        int load_idx = threadIdx.y * 16 + threadIdx.x;
        if (load_idx < BM) {
            int64_t m = row_base + load_idx;
            float val = 0.0f;
            if (m < chunk_size) {
                val = dA_cumsum[dt_offset + m];
            }
            dA_m[load_idx] = val;
        }
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BK;
            int col = idx - row * BK;
            if (row < BM && col < BK) {
                int64_t m = row_base + row;
                int64_t k = k0 + col;
                float val = 0.0f;
                if (m < chunk_size && k < chunk_size) {
                    val = cb[chunk_offset + m * chunk_size + k];
                }
                cb_s[row][col] = val;
            }
        }
        for (int l = 0; l < 2; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx - row * BN;
            if (row < BK && col < BN) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                float val = 0.0f;
                if (k < chunk_size && n < headdim) {
                    val = x[x_offset + k * headdim + n];
                }
                x_s[row][col] = val;
            }
        }
        if (load_idx < BK) {
            int64_t k = k0 + load_idx;
            float val = 0.0f;
            float dtv = 0.0f;
            if (k < chunk_size) {
                val = dA_cumsum[dt_offset + k];
                dtv = dt[dt_offset + k];
            }
            dA_k[load_idx] = val;
            dt_k[load_idx] = dtv;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float dA_kv = dA_k[k];
            float dtv = dt_k[k];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int row = tid_m * 4 + i;
                float dA_m_v = dA_m[row];
                float scale = expf(fminf(dA_m_v - dA_kv, 0.0f)) * dtv;
                float cbv = cb_s[row][k] * scale;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int col = tid_n * 4 + j;
                    acc[i][j] += cbv * x_s[k][col];
                }
            }
        }
        __syncthreads();
    }

    for (int64_t k0 = 0; k0 < d_state; k0 += BK) {
        int load_idx = threadIdx.y * 16 + threadIdx.x;
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BK;
            int col = idx - row * BK;
            if (row < BM && col < BK) {
                int64_t m = row_base + row;
                int64_t k = k0 + col;
                float val = 0.0f;
                if (m < chunk_size && k < d_state) {
                    val = C[c_offset + m * d_state + k];
                }
                c_s[row][col] = val;
            }
        }
        for (int l = 0; l < 2; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx - row * BN;
            if (row < BK && col < BN) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                float val = 0.0f;
                if (k < d_state && n < headdim) {
                    val = state_in[state_offset + n * d_state + k];
                }
                st_s[row][col] = val;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int row = tid_m * 4 + i;
                float cv = c_s[row][k];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int col = tid_n * 4 + j;
                    acc[i][j] += cv * st_s[k][col];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t m = m0 + i;
        if (m >= chunk_size) {
            continue;
        }
        float exp_a = expf(fminf(dA_cumsum[dt_offset + m], 0.0f));
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t n = n0 + j;
            if (n < headdim) {
                int64_t out_idx = x_offset + m * headdim + n;
                float d_val = d_has_hdim ? D[h * headdim + n] : D[h];
                float x_val = x[out_idx];
                y[out_idx] = acc[i][j] * exp_a + x_val * d_val;
            }
        }
    }
}

__global__ void chunk_scan_bwd_dC_dstate_kernel(
    const float* dy,
    const float* dA_cumsum,
    const float* C,
    const float* state_in,
    const float* chunk_state,
    float* dC,
    float* dstate,
    int64_t batch_groups,
    int64_t chunk_size,
    int64_t hdim,
    int64_t d_state,
    int64_t num_chunks,
    int64_t ngroups,
    int64_t nheads,
    int64_t headdim) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;

    int64_t gid = blockIdx.z;
    int64_t tile_m = blockIdx.y;
    int64_t tile_n = blockIdx.x;
    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    int64_t tid_m = threadIdx.y;
    int64_t tid_n = threadIdx.x;
    int64_t m0 = row_base + tid_m * 4;
    int64_t n0 = col_base + tid_n * 4;
    if (gid >= batch_groups || m0 >= chunk_size || n0 >= d_state) {
        return;
    }

    const float* dy_ptr = dy + gid * chunk_size * hdim;
    const float* C_ptr = C + gid * chunk_size * d_state;
    const float* state_in_ptr = state_in + gid * d_state * hdim;
    const float* chunk_state_ptr = chunk_state + gid * d_state * hdim;
    float* dC_ptr = dC + gid * chunk_size * d_state;
    int64_t b = gid / (num_chunks * ngroups);
    int64_t rem = gid - b * num_chunks * ngroups;
    int64_t c = rem / ngroups;
    int64_t g = rem - c * ngroups;
    int64_t heads_per_group = nheads / ngroups;

    __shared__ float dy_s[BM][BK];
    __shared__ float st_s[BK][BN];
    float acc_dc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc_dc[i][j] = 0.0f;
        }
    }

    for (int64_t k0 = 0; k0 < hdim; k0 += BK) {
        int load_idx = threadIdx.y * 16 + threadIdx.x;
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BK;
            int col = idx - row * BK;
            if (row < BM && col < BK) {
                int64_t m = row_base + row;
                int64_t k = k0 + col;
                float val = 0.0f;
                if (m < chunk_size && k < hdim) {
                    int64_t head_in_group = k / headdim;
                    int64_t head = g * heads_per_group + head_in_group;
                    int64_t dt_idx = ((b * nheads + head) * num_chunks + c) * chunk_size + m;
                    float scale = expf(fminf(dA_cumsum[dt_idx], 0.0f));
                    val = dy_ptr[m * hdim + k] * scale;
                }
                dy_s[row][col] = val;
            }
        }
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx - row * BN;
            if (row < BK && col < BN) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                float val = 0.0f;
                if (k < hdim && n < d_state) {
                    val = state_in_ptr[n * hdim + k] + chunk_state_ptr[n * hdim + k];
                }
                st_s[row][col] = val;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = dy_s[tid_m * 4 + i][k];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc_dc[i][j] += a * st_s[k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t m = m0 + i;
        if (m >= chunk_size) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t n = n0 + j;
            if (n < d_state) {
                dC_ptr[m * d_state + n] = acc_dc[i][j];
            }
        }
    }

    int64_t tile_h = blockIdx.y;
    int64_t tile_p = blockIdx.x;
    int64_t row_base_h = tile_h * BM;
    int64_t col_base_p = tile_p * BN;
    int64_t m0_h = row_base_h + tid_m * 4;
    int64_t n0_p = col_base_p + tid_n * 4;
    if (m0_h >= d_state || n0_p >= hdim) {
        return;
    }
    float* dstate_ptr = dstate + gid * d_state * hdim;
    __shared__ float c_s[BM][BK];
    __shared__ float dy2_s[BK][BN];
    float acc_ds[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc_ds[i][j] = 0.0f;
        }
    }

    for (int64_t k0 = 0; k0 < chunk_size; k0 += BK) {
        int load_idx = threadIdx.y * 16 + threadIdx.x;
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BK;
            int col = idx - row * BK;
            if (row < BM && col < BK) {
                int64_t m = row_base_h + row;
                int64_t k = k0 + col;
                float val = 0.0f;
                if (m < d_state && k < chunk_size) {
                    val = C_ptr[k * d_state + m];
                }
                c_s[row][col] = val;
            }
        }
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx - row * BN;
            if (row < BK && col < BN) {
                int64_t k = k0 + row;
                int64_t n = col_base_p + col;
                float val = 0.0f;
                if (k < chunk_size && n < hdim) {
                    int64_t head_in_group = n / headdim;
                    int64_t head = g * heads_per_group + head_in_group;
                    int64_t dt_idx = ((b * nheads + head) * num_chunks + c) * chunk_size + k;
                    float scale = expf(fminf(dA_cumsum[dt_idx], 0.0f));
                    val = dy_ptr[k * hdim + n] * scale;
                }
                dy2_s[row][col] = val;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = c_s[tid_m * 4 + i][k];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc_ds[i][j] += a * dy2_s[k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t m = m0_h + i;
        if (m >= d_state) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t n = n0_p + j;
            if (n < hdim) {
                dstate_ptr[m * hdim + n] = acc_ds[i][j];
            }
        }
    }
}

template <typename scalar_t>
__global__ void conv1d_forward_kernel(
    const scalar_t* zxbcdt,
    const scalar_t* conv_w,
    const scalar_t* conv_b,
    float* conv_out,
    int64_t batch,
    int64_t seqlen,
    int64_t d_in_proj,
    int64_t conv_dim,
    int64_t conv_kernel,
    int64_t d_mlp,
    int64_t d_ssm,
    bool has_conv_bias) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * seqlen * conv_dim;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t c = tmp % conv_dim;
    tmp /= conv_dim;
    int64_t t = tmp % seqlen;
    int64_t b = tmp / seqlen;

    int64_t offset_xbc = 2 * d_mlp + d_ssm;
    float conv_pre = 0.0f;
    for (int64_t w = 0; w < conv_kernel; ++w) {
        int64_t t_in = t - w;
        if (t_in < 0) {
            continue;
        }
        int64_t xbc_idx = (b * seqlen + t_in) * d_in_proj + offset_xbc + c;
        float xbc_val = to_float(zxbcdt[xbc_idx]);
        float w_val = to_float(conv_w[c * conv_kernel + w]);
        conv_pre += w_val * xbc_val;
    }
    if (has_conv_bias) {
        conv_pre += to_float(conv_b[c]);
    }
    conv_out[idx] = silu_f(conv_pre);
}

template <typename scalar_t>
__global__ void conv1d_pack_kernel(
    const scalar_t* zxbcdt,
    const scalar_t* conv_w,
    const scalar_t* conv_b,
    float* x_buf,
    float* b_buf,
    float* c_buf,
    int64_t batch,
    int64_t seqlen,
    int64_t d_in_proj,
    int64_t conv_dim,
    int64_t conv_kernel,
    int64_t d_mlp,
    int64_t d_ssm,
    int64_t ngroups,
    int64_t d_state,
    int64_t chunk_size,
    int64_t num_chunks,
    int64_t headdim,
    bool has_conv_bias) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * seqlen * conv_dim;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t c = tmp % conv_dim;
    tmp /= conv_dim;
    int64_t t = tmp % seqlen;
    int64_t b = tmp / seqlen;

    int64_t offset_xbc = 2 * d_mlp + d_ssm;
    float conv_pre = 0.0f;
    for (int64_t w = 0; w < conv_kernel; ++w) {
        int64_t t_in = t - w;
        if (t_in < 0) {
            continue;
        }
        int64_t xbc_idx = (b * seqlen + t_in) * d_in_proj + offset_xbc + c;
        float xbc_val = to_float(zxbcdt[xbc_idx]);
        float w_val = to_float(conv_w[c * conv_kernel + w]);
        conv_pre += w_val * xbc_val;
    }
    if (has_conv_bias) {
        conv_pre += to_float(conv_b[c]);
    }
    float conv_val = silu_f(conv_pre);

    int64_t chunk = t / chunk_size;
    int64_t t_in_chunk = t - chunk * chunk_size;
    int64_t nheads = d_ssm / headdim;

    if (c < d_ssm) {
        int64_t h = c / headdim;
        int64_t p = c - h * headdim;
        int64_t out_idx = (((b * num_chunks + chunk) * chunk_size + t_in_chunk) * nheads + h) * headdim + p;
        x_buf[out_idx] = conv_val;
    } else if (c < d_ssm + ngroups * d_state) {
        int64_t c_off = c - d_ssm;
        int64_t g = c_off / d_state;
        int64_t n = c_off - g * d_state;
        int64_t out_idx = (((b * num_chunks + chunk) * chunk_size + t_in_chunk) * ngroups + g) * d_state + n;
        b_buf[out_idx] = conv_val;
    } else {
        int64_t c_off = c - d_ssm - ngroups * d_state;
        int64_t g = c_off / d_state;
        int64_t n = c_off - g * d_state;
        int64_t out_idx = (((b * num_chunks + chunk) * chunk_size + t_in_chunk) * ngroups + g) * d_state + n;
        c_buf[out_idx] = conv_val;
    }
}

template <typename scalar_t>
__global__ void pack_conv_out_kernel(
    const scalar_t* conv_out,
    float* x_buf,
    float* b_buf,
    float* c_buf,
    int64_t batch,
    int64_t seqlen,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t nheads,
    int64_t headdim,
    int64_t ngroups,
    int64_t d_state,
    int64_t conv_dim,
    int64_t d_ssm) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * num_chunks * chunk_size * conv_dim;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t c = tmp % conv_dim;
    tmp /= conv_dim;
    int64_t t = tmp % (num_chunks * chunk_size);
    int64_t b = tmp / (num_chunks * chunk_size);
    int64_t t_global = t;
    int64_t chunk = t / chunk_size;
    int64_t t_in = t - chunk * chunk_size;

    float val = 0.0f;
    if (t_global < seqlen) {
        val = to_float(conv_out[(b * seqlen + t_global) * conv_dim + c]);
    }
    if (c < d_ssm) {
        int64_t h = c / headdim;
        int64_t p = c - h * headdim;
        int64_t out_idx = ((((b * num_chunks + chunk) * chunk_size + t_in) * nheads + h) * headdim + p);
        x_buf[out_idx] = val;
        return;
    }
    int64_t b_start = d_ssm;
    int64_t c_start = d_ssm + ngroups * d_state;
    if (c >= b_start && c < c_start) {
        int64_t g = (c - b_start) / d_state;
        int64_t n = c - b_start - g * d_state;
        int64_t out_idx = ((((b * num_chunks + chunk) * chunk_size + t_in) * ngroups + g) * d_state + n);
        b_buf[out_idx] = val;
        return;
    }
    int64_t g = (c - c_start) / d_state;
    int64_t n = c - c_start - g * d_state;
    int64_t out_idx = ((((b * num_chunks + chunk) * chunk_size + t_in) * ngroups + g) * d_state + n);
    c_buf[out_idx] = val;
}

template <typename scalar_t>
__global__ void pack_gy_kernel(
    const scalar_t* gy,
    float* gy_buf,
    int64_t batch,
    int64_t seqlen,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t nheads,
    int64_t headdim) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * num_chunks * chunk_size * nheads * headdim;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t p = tmp % headdim;
    tmp /= headdim;
    int64_t h = tmp % nheads;
    tmp /= nheads;
    int64_t t = tmp % (num_chunks * chunk_size);
    int64_t b = tmp / (num_chunks * chunk_size);
    int64_t t_global = t;
    int64_t chunk = t / chunk_size;
    int64_t t_in = t - chunk * chunk_size;

    float val = 0.0f;
    if (t_global < seqlen) {
        val = to_float(gy[((b * seqlen + t_global) * nheads + h) * headdim + p]);
    }
    gy_buf[((((b * num_chunks + chunk) * chunk_size + t_in) * nheads + h) * headdim + p)] = val;
}

__global__ void x_scale_kernel(
    const float* x,
    const float* dt,
    const float* dA_cumsum,
    float* x_scaled,
    int64_t batch,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t nheads,
    int64_t headdim) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * num_chunks * chunk_size * nheads * headdim;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t p = tmp % headdim;
    tmp /= headdim;
    int64_t h = tmp % nheads;
    tmp /= nheads;
    int64_t t = tmp % chunk_size;
    int64_t c = (tmp / chunk_size) % num_chunks;
    int64_t b = tmp / (num_chunks * chunk_size);
    int64_t dt_offset = ((b * nheads + h) * num_chunks + c) * chunk_size + t;
    float dA = dA_cumsum[dt_offset];
    float scale = dt[dt_offset] * expf(fminf(-dA, 0.0f));
    x_scaled[idx] = x[idx] * scale;
}

__global__ void gy_scale_kernel(
    const float* gy,
    const float* dA_cumsum,
    float* gy_scaled,
    int64_t batch,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t nheads,
    int64_t headdim) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * num_chunks * chunk_size * nheads * headdim;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t p = tmp % headdim;
    tmp /= headdim;
    int64_t h = tmp % nheads;
    tmp /= nheads;
    int64_t t = tmp % chunk_size;
    int64_t c = (tmp / chunk_size) % num_chunks;
    int64_t b = tmp / (num_chunks * chunk_size);
    int64_t dt_offset = ((b * nheads + h) * num_chunks + c) * chunk_size + t;
    float dA = dA_cumsum[dt_offset];
    float scale = expf(fminf(dA, 0.0f));
    gy_scaled[idx] = gy[idx] * scale;
}

template <typename scalar_t>
__global__ void dt_cumsum_kernel(
    const scalar_t* zxbcdt,
    const scalar_t* dt_bias,
    const scalar_t* a_log,
    const scalar_t* dt_scale,
    float* dt_out,
    float* dA_cumsum,
    float* exp_a_last,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t d_in_proj,
    int64_t d_mlp,
    float dt_min,
    float dt_max,
    int64_t chunk_size,
    bool has_dt_scale) {
    int64_t b = blockIdx.x;
    int64_t h = blockIdx.y;
    int64_t c = blockIdx.z;
    int64_t t = threadIdx.x;

    int64_t start = c * chunk_size;
    if (start >= seqlen) {
        return;
    }
    int64_t chunk_len = seqlen - start;
    if (chunk_len > chunk_size) {
        chunk_len = chunk_size;
    }

    extern __shared__ float shmem[];
    float val = 0.0f;
    if (t < chunk_len) {
        int64_t offset_dt = d_in_proj - nheads;
        int64_t t_global = start + t;
        float dt_raw = to_float(zxbcdt[(b * seqlen + t_global) * d_in_proj + offset_dt + h]);
        float dt_pre = dt_raw + to_float(dt_bias[h]);
        float dt = softplus_f(dt_pre);
        if (has_dt_scale) {
            dt *= to_float(dt_scale[(b * seqlen + t_global) * nheads + h]);
        }
        if (dt < dt_min || dt > dt_max) {
            dt = fminf(fmaxf(dt, dt_min), dt_max);
        }
        int64_t dt_idx = ((b * nheads + h) * ((seqlen + chunk_size - 1) / chunk_size) + c) * chunk_size + t;
        dt_out[dt_idx] = dt;
        float a = -expf(to_float(a_log[h]));
        val = dt * a;
    }
    shmem[t] = val;
    __syncthreads();

    for (int64_t offset = 1; offset < chunk_size; offset <<= 1) {
        float add = 0.0f;
        if (t >= offset) {
            add = shmem[t - offset];
        }
        __syncthreads();
        shmem[t] += add;
        __syncthreads();
    }

    if (t < chunk_len) {
        int64_t da_idx = ((b * nheads + h) * ((seqlen + chunk_size - 1) / chunk_size) + c) * chunk_size + t;
        dA_cumsum[da_idx] = shmem[t];
    }
    if (t == 0 && chunk_len > 0) {
        float last = shmem[chunk_len - 1];
        exp_a_last[(b * nheads + h) * ((seqlen + chunk_size - 1) / chunk_size) + c] = expf(fminf(last, 0.0f));
    }
}

template <typename scalar_t>
__global__ void fused_scan_forward_kernel(
    const float* conv_out,
    const float* dt,
    const float* dA_cumsum,
    const scalar_t* initial_state,
    scalar_t* y,
    scalar_t* final_state,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t conv_dim,
    int64_t chunk_size) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * nheads * headdim;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t b = tmp / (nheads * headdim);
    tmp -= b * nheads * headdim;
    int64_t h = tmp / headdim;
    int64_t p = tmp - h * headdim;

    int64_t heads_per_group = nheads / ngroups;
    int64_t group = h / heads_per_group;
    int64_t d_ssm = nheads * headdim;

    float state[kMaxDState];
    for (int64_t n = 0; n < d_state; ++n) {
        int64_t s_idx = ((b * nheads + h) * headdim + p) * d_state + n;
        state[n] = to_float(initial_state[s_idx]);
    }

    int64_t num_chunks = (seqlen + chunk_size - 1) / chunk_size;
    for (int64_t c = 0; c < num_chunks; ++c) {
        int64_t start = c * chunk_size;
        int64_t end = start + chunk_size;
        if (end > seqlen) {
            end = seqlen;
        }
        float prefix[kMaxDState];
        for (int64_t n = 0; n < d_state; ++n) {
            prefix[n] = 0.0f;
        }

        for (int64_t t = start; t < end; ++t) {
            int64_t t_in_chunk = t - start;
            int64_t chunk_idx = ((b * nheads + h) * num_chunks + c) * chunk_size + t_in_chunk;
            float a_cumsum = dA_cumsum[chunk_idx];
            float exp_a = expf(fminf(a_cumsum, 0.0f));
            float exp_neg = expf(fminf(-a_cumsum, 0.0f));
            float dt_val = dt[chunk_idx];

            int64_t x_chan = h * headdim + p;
            float x_conv = conv_out[(b * seqlen + t) * conv_dim + x_chan];
            float x_dt = x_conv * dt_val;

            float y_diag = 0.0f;
            float y_off = 0.0f;
            for (int64_t n = 0; n < d_state; ++n) {
                int64_t b_chan = d_state * group + n;
                int64_t c_chan = d_state * ngroups + d_state * group + n;
                float b_conv = conv_out[(b * seqlen + t) * conv_dim + d_ssm + b_chan];
                float c_conv = conv_out[(b * seqlen + t) * conv_dim + d_ssm + c_chan];

                prefix[n] += exp_neg * b_conv * x_dt;
                y_diag += c_conv * prefix[n];
                y_off += c_conv * state[n];
            }

            int64_t y_idx = ((b * seqlen + t) * nheads + h) * headdim + p;
            y[y_idx] = from_float<scalar_t>(exp_a * (y_diag + y_off));
        }

        float exp_a_last = expf(fminf(dA_cumsum[((b * nheads + h) * num_chunks + c) * chunk_size + (end - start - 1)], 0.0f));
        for (int64_t n = 0; n < d_state; ++n) {
            state[n] = exp_a_last * state[n] + exp_a_last * prefix[n];
        }
    }

    for (int64_t n = 0; n < d_state; ++n) {
        int64_t s_idx = ((b * nheads + h) * headdim + p) * d_state + n;
        final_state[s_idx] = from_float<scalar_t>(state[n]);
    }
}

__global__ void cb_chunk_kernel(
    const float* conv_out,
    float* cb_chunk,
    int64_t batch,
    int64_t seqlen,
    int64_t ngroups,
    int64_t d_state,
    int64_t conv_dim,
    int64_t chunk_size,
    int64_t chunk_idx,
    int64_t chunk_len) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * ngroups * chunk_size * chunk_size;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t s = tmp % chunk_size;
    tmp /= chunk_size;
    int64_t t = tmp % chunk_size;
    tmp /= chunk_size;
    int64_t g = tmp % ngroups;
    int64_t b = tmp / ngroups;
    if (t >= chunk_len || s >= chunk_len) {
        cb_chunk[idx] = 0.0f;
        return;
    }

    int64_t t_global = chunk_idx * chunk_size + t;
    int64_t s_global = chunk_idx * chunk_size + s;

    float sum = 0.0f;
    int64_t c_base = conv_dim - 2 * ngroups * d_state;
    int64_t b_base = c_base + g * d_state;
    int64_t c_base_g = c_base + ngroups * d_state + g * d_state;
    for (int64_t n = 0; n < d_state; ++n) {
        float c_val = conv_out[(b * seqlen + t_global) * conv_dim + c_base_g + n];
        float b_val = conv_out[(b * seqlen + s_global) * conv_dim + b_base + n];
        sum += c_val * b_val;
    }
    cb_chunk[idx] = sum;
}

template <typename scalar_t>
__global__ void chunk_output_kernel(
    const float* conv_out,
    const float* dt,
    const float* dA_cumsum,
    const float* cb_chunk,
    const float* state,
    scalar_t* y,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t conv_dim,
    int64_t chunk_size,
    int64_t chunk_idx,
    int64_t chunk_len,
    int64_t num_chunks) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * nheads * headdim * chunk_size;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t t = tmp % chunk_size;
    tmp /= chunk_size;
    int64_t p = tmp % headdim;
    tmp /= headdim;
    int64_t h = tmp % nheads;
    int64_t b = tmp / nheads;
    if (t >= chunk_len) {
        return;
    }

    int64_t heads_per_group = nheads / ngroups;
    int64_t group = h / heads_per_group;
    int64_t t_global = chunk_idx * chunk_size + t;
    int64_t x_chan = h * headdim + p;
    int64_t d_ssm = nheads * headdim;

    int64_t chunk_offset = ((b * nheads + h) * num_chunks + chunk_idx) * chunk_size;
    float exp_a_t = expf(fminf(dA_cumsum[chunk_offset + t], 0.0f));

    float y_off = 0.0f;
    int64_t c_base = d_ssm + ngroups * d_state + group * d_state;
    int64_t state_base = ((b * nheads + h) * headdim + p) * d_state;
    for (int64_t n = 0; n < d_state; ++n) {
        float c_val = conv_out[(b * seqlen + t_global) * conv_dim + c_base + n];
        y_off += c_val * state[state_base + n];
    }

    float y_diag = 0.0f;
    int64_t cb_base = ((b * ngroups + group) * chunk_size + t) * chunk_size;
    for (int64_t s = 0; s <= t; ++s) {
        float cb_val = cb_chunk[cb_base + s];
        float x_val = conv_out[(b * seqlen + (chunk_idx * chunk_size + s)) * conv_dim + x_chan];
        float dt_val = dt[chunk_offset + s];
        float exp_neg_s = expf(fminf(-dA_cumsum[chunk_offset + s], 0.0f));
        y_diag += cb_val * x_val * dt_val * exp_neg_s;
    }

    int64_t y_idx = ((b * seqlen + t_global) * nheads + h) * headdim + p;
    y[y_idx] = from_float<scalar_t>(exp_a_t * (y_diag + y_off));
}

__global__ void chunk_state_update_kernel(
    const float* conv_out,
    const float* dt,
    const float* dA_cumsum,
    float* state,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t conv_dim,
    int64_t chunk_size,
    int64_t chunk_idx,
    int64_t chunk_len,
    int64_t num_chunks) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * nheads * headdim;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t p = tmp % headdim;
    tmp /= headdim;
    int64_t h = tmp % nheads;
    int64_t b = tmp / nheads;

    int64_t heads_per_group = nheads / ngroups;
    int64_t group = h / heads_per_group;
    int64_t d_ssm = nheads * headdim;
    int64_t x_chan = h * headdim + p;

    int64_t chunk_offset = ((b * nheads + h) * num_chunks + chunk_idx) * chunk_size;
    float exp_a_last = expf(fminf(dA_cumsum[chunk_offset + (chunk_len - 1)], 0.0f));

    int64_t state_base = ((b * nheads + h) * headdim + p) * d_state;
    for (int64_t n = 0; n < d_state; ++n) {
        float acc = 0.0f;
        int64_t b_base = d_ssm + group * d_state + n;
        for (int64_t s = 0; s < chunk_len; ++s) {
            int64_t s_global = chunk_idx * chunk_size + s;
            float x_val = conv_out[(b * seqlen + s_global) * conv_dim + x_chan];
            float b_val = conv_out[(b * seqlen + s_global) * conv_dim + b_base];
            float dt_val = dt[chunk_offset + s];
            float exp_neg_s = expf(fminf(-dA_cumsum[chunk_offset + s], 0.0f));
            acc += exp_neg_s * b_val * x_val * dt_val;
        }
        state[state_base + n] = exp_a_last * state[state_base + n] + exp_a_last * acc;
    }
}

__global__ void dcb_kernel(
    const float* conv_out,
    const float* dt,
    const float* dA_cumsum,
    const float* grad_y,
    float* dcb,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t conv_dim,
    int64_t chunk_size,
    int64_t chunk_idx,
    int64_t chunk_len,
    int64_t num_chunks) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * ngroups * chunk_size * chunk_size;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t s = tmp % chunk_size;
    tmp /= chunk_size;
    int64_t t = tmp % chunk_size;
    tmp /= chunk_size;
    int64_t g = tmp % ngroups;
    int64_t b = tmp / ngroups;
    if (t >= chunk_len || s >= chunk_len || s > t) {
        dcb[idx] = 0.0f;
        return;
    }

    int64_t t_global = chunk_idx * chunk_size + t;
    int64_t s_global = chunk_idx * chunk_size + s;
    int64_t heads_per_group = nheads / ngroups;
    float sum = 0.0f;
    for (int64_t h = g * heads_per_group; h < (g + 1) * heads_per_group; ++h) {
        int64_t chunk_offset = ((b * nheads + h) * num_chunks + chunk_idx) * chunk_size;
        float exp_a_t = expf(fminf(dA_cumsum[chunk_offset + t], 0.0f));
        float exp_neg_s = expf(fminf(-dA_cumsum[chunk_offset + s], 0.0f));
        float dt_s = dt[chunk_offset + s];
        for (int64_t p = 0; p < headdim; ++p) {
            float dy = grad_y[((b * seqlen + t_global) * nheads + h) * headdim + p];
            float x = conv_out[(b * seqlen + s_global) * conv_dim + h * headdim + p];
            sum += dy * exp_a_t * x * dt_s * exp_neg_s;
        }
    }
    dcb[idx] = sum;
}

__global__ void dstate_from_y_kernel(
    const float* conv_out,
    const float* dA_cumsum,
    const float* grad_y,
    const float* dstate_out,
    float* dstate_in,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t conv_dim,
    int64_t chunk_size,
    int64_t chunk_idx,
    int64_t chunk_len,
    int64_t num_chunks) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * nheads * headdim * d_state;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t n = tmp % d_state;
    tmp /= d_state;
    int64_t p = tmp % headdim;
    tmp /= headdim;
    int64_t h = tmp % nheads;
    int64_t b = tmp / nheads;

    int64_t heads_per_group = nheads / ngroups;
    int64_t group = h / heads_per_group;
    int64_t c_base = nheads * headdim + ngroups * d_state + group * d_state;

    float acc = 0.0f;
    int64_t chunk_offset = ((b * nheads + h) * num_chunks + chunk_idx) * chunk_size;
    for (int64_t t = 0; t < chunk_len; ++t) {
        int64_t t_global = chunk_idx * chunk_size + t;
        float exp_a_t = expf(fminf(dA_cumsum[chunk_offset + t], 0.0f));
        float dy = grad_y[((b * seqlen + t_global) * nheads + h) * headdim + p];
        float c_val = conv_out[(b * seqlen + t_global) * conv_dim + c_base + n];
        acc += dy * exp_a_t * c_val;
    }

    float exp_a_last = expf(fminf(dA_cumsum[chunk_offset + (chunk_len - 1)], 0.0f));
    int64_t state_idx = ((b * nheads + h) * headdim + p) * d_state + n;
    dstate_in[state_idx] = exp_a_last * dstate_out[state_idx] + acc;
}

__global__ void dC_kernel(
    const float* conv_out,
    const float* dcb,
    const float* dA_cumsum,
    const float* grad_y,
    const float* state_in,
    float* dC,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t conv_dim,
    int64_t chunk_size,
    int64_t chunk_idx,
    int64_t chunk_len,
    int64_t num_chunks) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * ngroups * chunk_size * d_state;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t n = tmp % d_state;
    tmp /= d_state;
    int64_t t = tmp % chunk_size;
    tmp /= chunk_size;
    int64_t g = tmp % ngroups;
    int64_t b = tmp / ngroups;
    if (t >= chunk_len) {
        return;
    }

    int64_t t_global = chunk_idx * chunk_size + t;
    int64_t c_base = nheads * headdim + ngroups * d_state + g * d_state;
    int64_t b_base = nheads * headdim + g * d_state;

    float acc = 0.0f;
    for (int64_t s = 0; s < chunk_len; ++s) {
        float dcb_val = dcb[((b * ngroups + g) * chunk_size + t) * chunk_size + s];
        float b_val = conv_out[(b * seqlen + (chunk_idx * chunk_size + s)) * conv_dim + b_base + n];
        acc += dcb_val * b_val;
    }

    int64_t heads_per_group = nheads / ngroups;
    float acc_state = 0.0f;
    for (int64_t h = g * heads_per_group; h < (g + 1) * heads_per_group; ++h) {
        int64_t chunk_offset = ((b * nheads + h) * num_chunks + chunk_idx) * chunk_size;
        float exp_a_t = expf(fminf(dA_cumsum[chunk_offset + t], 0.0f));
        for (int64_t p = 0; p < headdim; ++p) {
            float dy = grad_y[((b * seqlen + t_global) * nheads + h) * headdim + p];
            int64_t state_idx = ((b * nheads + h) * headdim + p) * d_state + n;
            acc_state += dy * exp_a_t * state_in[state_idx];
        }
    }
    int64_t out_idx = ((b * seqlen + t_global) * ngroups + g) * d_state + n;
    atomicAdd(&dC[out_idx], acc + acc_state);
}

__global__ void dB_kernel(
    const float* conv_out,
    const float* dt,
    const float* dA_cumsum,
    const float* dstate_out,
    const float* dcb,
    float* dB,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t headdim,
    int64_t ngroups,
    int64_t d_state,
    int64_t conv_dim,
    int64_t chunk_size,
    int64_t chunk_idx,
    int64_t chunk_len,
    int64_t num_chunks) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * ngroups * chunk_size * d_state;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t n = tmp % d_state;
    tmp /= d_state;
    int64_t s = tmp % chunk_size;
    tmp /= chunk_size;
    int64_t g = tmp % ngroups;
    int64_t b = tmp / ngroups;
    if (s >= chunk_len) {
        return;
    }

    int64_t s_global = chunk_idx * chunk_size + s;
    int64_t d_ssm = nheads * headdim;
    int64_t c_base = d_ssm + ngroups * d_state + g * d_state;
    float acc = 0.0f;
    for (int64_t t = s; t < chunk_len; ++t) {
        float dcb_val = dcb[((b * ngroups + g) * chunk_size + t) * chunk_size + s];
        float c_val = conv_out[(b * seqlen + (chunk_idx * chunk_size + t)) * conv_dim + c_base + n];
        acc += dcb_val * c_val;
    }

    int64_t heads_per_group = nheads / ngroups;
    float acc_state = 0.0f;
    for (int64_t h = g * heads_per_group; h < (g + 1) * heads_per_group; ++h) {
        int64_t chunk_offset = ((b * nheads + h) * num_chunks + chunk_idx) * chunk_size;
        float exp_a_last = expf(fminf(dA_cumsum[chunk_offset + (chunk_len - 1)], 0.0f));
        float exp_neg_s = expf(fminf(-dA_cumsum[chunk_offset + s], 0.0f));
        float dt_s = dt[chunk_offset + s];
        for (int64_t p = 0; p < headdim; ++p) {
            int64_t state_idx = ((b * nheads + h) * headdim + p) * d_state + n;
            float x_val = conv_out[(b * seqlen + s_global) * conv_dim + h * headdim + p];
            acc_state += dstate_out[state_idx] * exp_a_last * x_val * dt_s * exp_neg_s;
        }
    }
    int64_t out_idx = ((b * seqlen + s_global) * ngroups + g) * d_state + n;
    atomicAdd(&dB[out_idx], acc + acc_state);
}

__global__ void dx_ddt_ddA_kernel(
    const float* conv_out,
    const float* dt,
    const float* dA_cumsum,
    const float* grad_y,
    const float* cb,
    const float* dstate_out,
    float* dx,
    float* ddt,
    float* ddA,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t conv_dim,
    int64_t chunk_size,
    int64_t chunk_idx,
    int64_t chunk_len,
    int64_t num_chunks) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * nheads * headdim * chunk_size;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t s = tmp % chunk_size;
    tmp /= chunk_size;
    int64_t p = tmp % headdim;
    tmp /= headdim;
    int64_t h = tmp % nheads;
    int64_t b = tmp / nheads;
    if (s >= chunk_len) {
        return;
    }

    int64_t heads_per_group = nheads / ngroups;
    int64_t g = h / heads_per_group;
    int64_t s_global = chunk_idx * chunk_size + s;
    int64_t chunk_offset = ((b * nheads + h) * num_chunks + chunk_idx) * chunk_size;
    float exp_neg_s = expf(fminf(-dA_cumsum[chunk_offset + s], 0.0f));
    float dt_s = dt[chunk_offset + s];

    float sum = 0.0f;
    for (int64_t t = s; t < chunk_len; ++t) {
        int64_t t_global = chunk_idx * chunk_size + t;
        float exp_a_t = expf(fminf(dA_cumsum[chunk_offset + t], 0.0f));
        float dy = grad_y[((b * seqlen + t_global) * nheads + h) * headdim + p];
        float cb_val = cb[((b * ngroups + g) * chunk_size + t) * chunk_size + s];
        sum += dy * exp_a_t * cb_val;
    }
    float extra = 0.0f;
    int64_t b_base = nheads * headdim + g * d_state;
    float exp_a_last = expf(fminf(dA_cumsum[chunk_offset + (chunk_len - 1)], 0.0f));
    int64_t state_base = ((b * nheads + h) * headdim + p) * d_state;
    for (int64_t n = 0; n < d_state; ++n) {
        float b_val = conv_out[(b * seqlen + s_global) * conv_dim + b_base + n];
        extra += dstate_out[state_base + n] * exp_a_last * b_val;
    }
    float total_val = sum + extra;
    float x_val = conv_out[(b * seqlen + s_global) * conv_dim + h * headdim + p];
    float dx_val = total_val * dt_s * exp_neg_s;
    dx[((b * seqlen + s_global) * nheads + h) * headdim + p] = dx_val;

    float dtemp = total_val * x_val;
    float ddt_val = dtemp * exp_neg_s;
    atomicAdd(&ddt[(b * nheads + h) * seqlen + s_global], ddt_val);
    atomicAdd(&ddA[(b * nheads + h) * seqlen + s_global], -dtemp * dt_s * exp_neg_s);
}

__global__ void ddA_from_y_kernel(
    const float* y,
    const float* grad_y,
    float* ddA,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t headdim) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * nheads * seqlen;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t t = tmp % seqlen;
    tmp /= seqlen;
    int64_t h = tmp % nheads;
    int64_t b = tmp / nheads;
    float sum = 0.0f;
    for (int64_t p = 0; p < headdim; ++p) {
        float dy = grad_y[((b * seqlen + t) * nheads + h) * headdim + p];
        float y_val = y[((b * seqlen + t) * nheads + h) * headdim + p];
        sum += dy * y_val;
    }
    ddA[(b * nheads + h) * seqlen + t] += sum;
}

__global__ void ddA_from_state_kernel(
    const float* state_out,
    const float* dstate_out,
    float* ddA,
    int64_t batch,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t seqlen,
    int64_t chunk_size,
    int64_t chunk_idx,
    int64_t chunk_len) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * nheads;
    if (idx >= total) {
        return;
    }
    int64_t h = idx % nheads;
    int64_t b = idx / nheads;
    int64_t last_t = chunk_idx * chunk_size + (chunk_len - 1);
    float sum = 0.0f;
    for (int64_t p = 0; p < headdim; ++p) {
        int64_t base = ((b * nheads + h) * headdim + p) * d_state;
        for (int64_t n = 0; n < d_state; ++n) {
            sum += dstate_out[base + n] * state_out[base + n];
        }
    }
    atomicAdd(&ddA[(b * nheads + h) * seqlen + last_t], sum);
}
template <typename scalar_t>
__global__ void ddA_to_dtdA_kernel(
    const float* ddA,
    const float* dt,
    const scalar_t* a_log,
    float* ddt,
    float* dA_out,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t chunk_size) {
    int64_t b = blockIdx.x;
    int64_t h = blockIdx.y;
    int64_t c = blockIdx.z;
    int64_t t = threadIdx.x;
    int64_t start = c * chunk_size;
    if (start >= seqlen) {
        return;
    }
    int64_t chunk_len = seqlen - start;
    if (chunk_len > chunk_size) {
        chunk_len = chunk_size;
    }
    extern __shared__ float shmem[];
    float val = 0.0f;
    if (t < chunk_len) {
        val = ddA[(b * nheads + h) * seqlen + start + t];
    }
    shmem[t] = val;
    __syncthreads();
    for (int64_t offset = 1; offset < chunk_size; offset <<= 1) {
        float add = 0.0f;
        int64_t idx = chunk_len - 1 - t;
        if (idx >= offset) {
            add = shmem[chunk_len - 1 - (idx - offset)];
        }
        __syncthreads();
        shmem[chunk_len - 1 - t] += add;
        __syncthreads();
    }

    if (t < chunk_len) {
        float a = -expf(to_float(a_log[h]));
        float acc = shmem[t];
        int64_t idx_global = start + t;
        int64_t num_chunks = (seqlen + chunk_size - 1) / chunk_size;
        atomicAdd(&ddt[(b * nheads + h) * seqlen + idx_global], acc * a);
        atomicAdd(&dA_out[h], ddA[(b * nheads + h) * seqlen + idx_global] * dt[((b * nheads + h) * num_chunks + c) * chunk_size + t]);
    }
}

template <typename scalar_t>
__global__ void ddt_raw_kernel(
    const float* ddt,
    const scalar_t* zxbcdt,
    const scalar_t* dt_bias,
    const scalar_t* dt_scale,
    float* ddt_raw,
    float* ddt_bias,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t d_in_proj,
    float dt_min,
    float dt_max,
    bool has_dt_scale) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * seqlen * nheads;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t t = tmp % seqlen;
    tmp /= seqlen;
    int64_t h = tmp % nheads;
    int64_t b = tmp / nheads;

    int64_t offset_dt = d_in_proj - nheads;
    float dt_raw = to_float(zxbcdt[(b * seqlen + t) * d_in_proj + offset_dt + h]);
    float dt_pre = dt_raw + to_float(dt_bias[h]);
    float dt_val = softplus_f(dt_pre);
    float scale = 1.0f;
    if (has_dt_scale) {
        scale = to_float(dt_scale[(b * seqlen + t) * nheads + h]);
        dt_val *= scale;
    }
    bool clamp = dt_val < dt_min || dt_val > dt_max;
    if (clamp) {
        ddt_raw[idx] = 0.0f;
        return;
    }
    float grad = ddt[(b * nheads + h) * seqlen + t] * sigmoid_f(dt_pre);
    if (has_dt_scale) {
        grad *= scale;
    }
    ddt_raw[idx] = grad;
    atomicAdd(&ddt_bias[h], grad);
}

template <typename scalar_t>
__global__ void fused_conv_scan_backward_kernel(
    const scalar_t* zxbcdt,
    const scalar_t* conv_w,
    const scalar_t* conv_b,
    const scalar_t* dt_bias,
    const scalar_t* a_log,
    const scalar_t* dt_scale,
    const scalar_t* initial_state,
    const scalar_t* final_state,
    const scalar_t* grad_y,
    const scalar_t* grad_final_state,
    float* dx_conv,
    float* dB,
    float* dC,
    float* ddt_raw,
    float* dA,
    float* ddt_bias,
    float* dinitial_state,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t d_in_proj,
    int64_t d_ssm,
    int64_t d_mlp,
    int64_t conv_kernel,
    float dt_min,
    float dt_max,
    bool has_conv_bias,
    bool has_dt_scale) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * nheads * headdim;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t b = tmp / (nheads * headdim);
    tmp -= b * nheads * headdim;
    int64_t h = tmp / headdim;
    int64_t p = tmp - h * headdim;

    int64_t heads_per_group = nheads / ngroups;
    int64_t group = h / heads_per_group;

    float state[kMaxDState];
    float gstate[kMaxDState];

    for (int64_t n = 0; n < d_state; ++n) {
        int64_t s_idx = ((b * nheads + h) * headdim + p) * d_state + n;
        state[n] = to_float(final_state[s_idx]);
        gstate[n] = grad_final_state ? to_float(grad_final_state[s_idx]) : 0.0f;
    }

    float a = -expf(to_float(a_log[h]));

    int64_t offset_xbc = 2 * d_mlp + d_ssm;
    int64_t offset_dt = offset_xbc + d_ssm + 2 * ngroups * d_state;

    for (int64_t t = seqlen - 1; t >= 0; --t) {
        float dt_raw = to_float(zxbcdt[(b * seqlen + t) * d_in_proj + offset_dt + h]);
        float dt_pre = dt_raw + to_float(dt_bias[h]);
        float dt = softplus_f(dt_pre);
        float scale = 1.0f;
        if (has_dt_scale) {
            scale = to_float(dt_scale[(b * seqlen + t) * nheads + h]);
            dt *= scale;
        }
        bool clamp = false;
        if (dt < dt_min || dt > dt_max) {
            dt = fminf(fmaxf(dt, dt_min), dt_max);
            clamp = true;
        }
        float da = expf(dt * a);

        int64_t x_chan = h * headdim + p;
        float x_conv = 0.0f;
        for (int64_t w = 0; w < conv_kernel; ++w) {
            int64_t t_in = t - w;
            if (t_in < 0) {
                continue;
            }
            int64_t xbc_idx = (b * seqlen + t_in) * d_in_proj + offset_xbc + x_chan;
            float xbc_val = to_float(zxbcdt[xbc_idx]);
            float w_val = to_float(conv_w[x_chan * conv_kernel + w]);
            x_conv += w_val * xbc_val;
        }
        if (has_conv_bias) {
            x_conv += to_float(conv_b[x_chan]);
        }
        x_conv = silu_f(x_conv);

        float dy = to_float(grad_y[((b * seqlen + t) * nheads + h) * headdim + p]);
        float grad_da = 0.0f;
        float grad_dt = 0.0f;
        float grad_x = 0.0f;

        for (int64_t n = 0; n < d_state; ++n) {
            int64_t b_chan = d_ssm + group * d_state + n;
            int64_t c_chan = d_ssm + ngroups * d_state + group * d_state + n;

            float b_conv = 0.0f;
            float c_conv = 0.0f;
            for (int64_t w = 0; w < conv_kernel; ++w) {
                int64_t t_in = t - w;
                if (t_in < 0) {
                    continue;
                }
                int64_t b_idx = (b * seqlen + t_in) * d_in_proj + offset_xbc + b_chan;
                int64_t c_idx = (b * seqlen + t_in) * d_in_proj + offset_xbc + c_chan;
                float b_val = to_float(zxbcdt[b_idx]);
                float c_val = to_float(zxbcdt[c_idx]);
                float b_w = to_float(conv_w[b_chan * conv_kernel + w]);
                float c_w = to_float(conv_w[c_chan * conv_kernel + w]);
                b_conv += b_w * b_val;
                c_conv += c_w * c_val;
            }
            if (has_conv_bias) {
                b_conv += to_float(conv_b[b_chan]);
                c_conv += to_float(conv_b[c_chan]);
            }
            b_conv = silu_f(b_conv);
            c_conv = silu_f(c_conv);

            float gy = dy * c_conv;
            float grad_state = gstate[n] + gy;
            float s_prev = (state[n] - dt * b_conv * x_conv) / da;

            atomicAdd(&dC[((b * seqlen + t) * ngroups + group) * d_state + n], dy * state[n]);

            grad_da += grad_state * s_prev;

            float g_dbx = grad_state;
            grad_x += g_dbx * dt * b_conv;
            atomicAdd(&dB[((b * seqlen + t) * ngroups + group) * d_state + n], g_dbx * dt * x_conv);
            grad_dt += g_dbx * b_conv * x_conv;

            gstate[n] = grad_state * da;
            state[n] = s_prev;

        }

        dx_conv[((b * seqlen + t) * nheads + h) * headdim + p] = grad_x;

        float grad_dt_total = grad_dt + grad_da * a * da;
        if (clamp) {
            grad_dt_total = 0.0f;
        }
        float ddt_pre = grad_dt_total * sigmoid_f(dt_pre);
        if (has_dt_scale) {
            ddt_pre *= scale;
        }
        atomicAdd(&ddt_raw[(b * seqlen + t) * nheads + h], ddt_pre);
        atomicAdd(&ddt_bias[h], ddt_pre);
        atomicAdd(&dA[h], grad_da * dt * da);
    }

    for (int64_t n = 0; n < d_state; ++n) {
        int64_t s_idx = ((b * nheads + h) * headdim + p) * d_state + n;
        dinitial_state[s_idx] = gstate[n];
    }
}

template <typename scalar_t>
__global__ void scatter_xbc_grad_kernel(
    const float* dx_conv,
    const float* dB,
    const float* dC,
    float* d_xbc_conv,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t d_ssm) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * seqlen * (d_ssm + 2 * ngroups * d_state);
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t c = tmp % (d_ssm + 2 * ngroups * d_state);
    tmp /= (d_ssm + 2 * ngroups * d_state);
    int64_t t = tmp % seqlen;
    int64_t b = tmp / seqlen;

    if (c < d_ssm) {
        int64_t h = c / headdim;
        int64_t p = c - h * headdim;
        d_xbc_conv[idx] = dx_conv[((b * seqlen + t) * nheads + h) * headdim + p];
        return;
    }

    int64_t b_start = d_ssm;
    int64_t c_start = d_ssm + ngroups * d_state;
    if (c >= b_start && c < c_start) {
        int64_t g = (c - b_start) / d_state;
        int64_t n = c - b_start - g * d_state;
        d_xbc_conv[idx] = dB[((b * seqlen + t) * ngroups + g) * d_state + n];
        return;
    }
    int64_t g = (c - c_start) / d_state;
    int64_t n = c - c_start - g * d_state;
    d_xbc_conv[idx] = dC[((b * seqlen + t) * ngroups + g) * d_state + n];
}

template <typename scalar_t>
__global__ void conv1d_backward_kernel(
    const scalar_t* zxbcdt,
    const scalar_t* conv_w,
    const scalar_t* conv_b,
    const float* d_xbc_conv,
    float* d_xbc_in,
    float* d_conv_w,
    float* d_conv_b,
    int64_t batch,
    int64_t seqlen,
    int64_t d_in_proj,
    int64_t d_ssm,
    int64_t d_mlp,
    int64_t d_state,
    int64_t ngroups,
    int64_t conv_kernel,
    bool has_conv_bias) {
    int64_t conv_dim = d_ssm + 2 * ngroups * d_state;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * conv_dim;
    if (idx >= total) {
        return;
    }
    int64_t b = idx / conv_dim;
    int64_t c = idx - b * conv_dim;

    int64_t offset_xbc = 2 * d_mlp + d_ssm;

    for (int64_t t = 0; t < seqlen; ++t) {
        float conv_pre = 0.0f;
        for (int64_t w = 0; w < conv_kernel; ++w) {
            int64_t t_in = t - w;
            if (t_in < 0) {
                continue;
            }
            int64_t xbc_idx = (b * seqlen + t_in) * d_in_proj + offset_xbc + c;
            float xbc_val = to_float(zxbcdt[xbc_idx]);
            float w_val = to_float(conv_w[c * conv_kernel + w]);
            conv_pre += w_val * xbc_val;
        }
        if (has_conv_bias) {
            conv_pre += to_float(conv_b[c]);
        }
        float d_act = d_xbc_conv[(b * seqlen + t) * conv_dim + c] * silu_grad_f(conv_pre);
        if (has_conv_bias) {
            atomicAdd(&d_conv_b[c], d_act);
        }
        for (int64_t w = 0; w < conv_kernel; ++w) {
            int64_t t_in = t - w;
            if (t_in < 0) {
                continue;
            }
            int64_t xbc_idx = (b * seqlen + t_in) * d_in_proj + offset_xbc + c;
            float xbc_val = to_float(zxbcdt[xbc_idx]);
            atomicAdd(&d_conv_w[c * conv_kernel + w], d_act * xbc_val);
            atomicAdd(&d_xbc_in[(b * seqlen + t_in) * conv_dim + c], d_act * to_float(conv_w[c * conv_kernel + w]));
        }
    }
}

__global__ void d_skip_dx_kernel(
    const float* gy,
    const float* x,
    const float* d_param,
    float* dx,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t headdim,
    int64_t d_has_hdim) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * seqlen * nheads * headdim;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t p = tmp % headdim;
    tmp /= headdim;
    int64_t h = tmp % nheads;
    tmp /= nheads;
    int64_t t = tmp % seqlen;
    int64_t b = tmp / seqlen;

    int64_t off = ((b * seqlen + t) * nheads + h) * headdim + p;
    float d_val = d_has_hdim ? d_param[h * headdim + p] : d_param[h];
    float gy_val = gy[off];
    float x_val = x[off];
    dx[off] += gy_val * d_val;
}

__global__ void d_skip_dD_kernel(
    const float* gy,
    const float* x,
    float* dD,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t headdim,
    int64_t d_has_hdim) {
    int64_t idx = blockIdx.x;
    int64_t tid = threadIdx.x;
    int64_t total = batch * seqlen;
    int64_t stride = blockDim.x * gridDim.y;
    int64_t start = blockIdx.y * blockDim.x + tid;

    float acc = 0.0f;
    if (d_has_hdim) {
        int64_t h = idx / headdim;
        int64_t p = idx - h * headdim;
        for (int64_t i = start; i < total; i += stride) {
            int64_t b = i / seqlen;
            int64_t t = i - b * seqlen;
            int64_t off = ((b * seqlen + t) * nheads + h) * headdim + p;
            acc += gy[off] * x[off];
        }
    } else {
        int64_t h = idx;
        for (int64_t i = start; i < total; i += stride) {
            int64_t b = i / seqlen;
            int64_t t = i - b * seqlen;
            int64_t base = ((b * seqlen + t) * nheads + h) * headdim;
            for (int64_t p = 0; p < headdim; ++p) {
                acc += gy[base + p] * x[base + p];
            }
        }
    }
    __shared__ float sh[256];
    sh[tid] = acc;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sh[tid] += sh[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        if (d_has_hdim) {
            atomicAdd(&dD[idx], sh[0]);
        } else {
            atomicAdd(&dD[idx], sh[0]);
        }
    }
}

__global__ void ddA_y_bwd_kernel(
    const float* gy,
    const float* y_scan,
    float* ddA,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t headdim) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * seqlen * nheads;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t t = tmp % seqlen;
    tmp /= seqlen;
    int64_t h = tmp % nheads;
    int64_t b = tmp / nheads;
    int64_t base = ((b * seqlen + t) * nheads + h) * headdim;
    float acc = 0.0f;
    for (int64_t p = 0; p < headdim; ++p) {
        acc += gy[base + p] * y_scan[base + p];
    }
    ddA[(b * nheads + h) * seqlen + t] += acc;
}

__global__ void chunk_scan_bwd_dx_kernel(
    const float* b_mat,
    const float* dstate,
    const float* x_mat,
    const float* dt,
    const float* dA_cumsum,
    float* dx,
    float* dtemp,
    float* ddt,
    float* ddA_xscaled,
    int64_t batch_groups,
    int64_t chunk_size,
    int64_t d_state,
    int64_t hdim,
    int64_t headdim) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int kMaxHeadsTile = 8;

    int64_t gid = blockIdx.z;
    int64_t tile_m = blockIdx.y;
    int64_t tile_n = blockIdx.x;
    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    int64_t tid_m = threadIdx.y;
    int64_t tid_n = threadIdx.x;
    int64_t m0 = row_base + tid_m * 4;
    int64_t n0 = col_base + tid_n * 4;
    if (gid >= batch_groups || m0 >= chunk_size || n0 >= hdim) {
        return;
    }

    const float* b_ptr = b_mat + gid * chunk_size * d_state;
    const float* dstate_ptr = dstate + gid * d_state * hdim;
    const float* x_ptr = x_mat + gid * chunk_size * hdim;
    float* dx_ptr = dx + gid * chunk_size * hdim;
    float* dtemp_ptr = dtemp + gid * chunk_size * (hdim / headdim);
    float* ddt_ptr = ddt + gid * chunk_size * (hdim / headdim);
    float* ddA_ptr = ddA_xscaled + gid * chunk_size * (hdim / headdim);
    int64_t dt_offset = gid * chunk_size;

    __shared__ float sh_temp[BM][kMaxHeadsTile];
    __shared__ float sh_ddt[BM][kMaxHeadsTile];
    __shared__ float sh_ddA[BM][kMaxHeadsTile];
    int heads_in_tile = (BN + headdim - 1) / headdim;
    if (heads_in_tile > kMaxHeadsTile) {
        heads_in_tile = kMaxHeadsTile;
    }
    int head_tile_base = col_base / headdim;

    int load_idx = threadIdx.y * 16 + threadIdx.x;
    int total_head_slots = BM * heads_in_tile;
    for (int idx = load_idx; idx < total_head_slots; idx += 256) {
        int row = idx / heads_in_tile;
        int head_slot = idx - row * heads_in_tile;
        sh_temp[row][head_slot] = 0.0f;
        sh_ddt[row][head_slot] = 0.0f;
        sh_ddA[row][head_slot] = 0.0f;
    }
    __syncthreads();

    __shared__ float a_s[BM][BK];
    __shared__ float b_s[BK][BN];

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int64_t k0 = 0; k0 < d_state; k0 += BK) {
        int load_idx = threadIdx.y * 16 + threadIdx.x;
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BK;
            int col = idx - row * BK;
            if (row < BM && col < BK) {
                int64_t m = row_base + row;
                int64_t k = k0 + col;
                float val = 0.0f;
                if (m < chunk_size && k < d_state) {
                    val = b_ptr[m * d_state + k];
                }
                a_s[row][col] = val;
            }
        }
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx - row * BN;
            if (row < BK && col < BN) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                float val = 0.0f;
                if (k < d_state && n < hdim) {
                    val = dstate_ptr[k * hdim + n];
                }
                b_s[row][col] = val;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = a_s[tid_m * 4 + i][k];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a * b_s[k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t m = m0 + i;
        if (m >= chunk_size) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t n = n0 + j;
            if (n < hdim) {
                float val = acc[i][j];
                dx_ptr[m * hdim + n] = val;
                int64_t head = n / headdim;
                int head_slot = static_cast<int>(head) - head_tile_base;
                if (head_slot >= 0 && head_slot < heads_in_tile) {
                    float x_val = x_ptr[m * hdim + n];
                    float prod = val * x_val;
                    float dtv = dt[dt_offset + m];
                    float dA = dA_cumsum[dt_offset + m];
                    float exp_neg = expf(fminf(-dA, 0.0f));
                    atomicAdd(&sh_temp[m][head_slot], prod);
                    atomicAdd(&sh_ddt[m][head_slot], prod * exp_neg);
                    atomicAdd(&sh_ddA[m][head_slot], -prod * dtv * exp_neg);
                }
            }
        }
    }

    __syncthreads();
    for (int idx = load_idx; idx < total_head_slots; idx += 256) {
        int row = idx / heads_in_tile;
        int head_slot = idx - row * heads_in_tile;
        int64_t head = head_tile_base + head_slot;
        if (row < chunk_size && head < (hdim / headdim)) {
            int64_t out_idx = row * (hdim / headdim) + head;
            atomicAdd(&dtemp_ptr[out_idx], sh_temp[row][head_slot]);
            atomicAdd(&ddt_ptr[out_idx], sh_ddt[row][head_slot]);
            atomicAdd(&ddA_ptr[out_idx], sh_ddA[row][head_slot]);
        }
    }
}

__global__ void chunk_scan_bwd_dB_kernel(
    const float* x_mat,
    const float* dt,
    const float* dA_cumsum,
    const float* dstate,
    float* dB,
    int64_t batch_groups,
    int64_t chunk_size,
    int64_t hdim,
    int64_t d_state,
    int64_t num_chunks,
    int64_t ngroups,
    int64_t nheads,
    int64_t headdim) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;

    int64_t gid = blockIdx.z;
    int64_t tile_m = blockIdx.y;
    int64_t tile_n = blockIdx.x;
    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    int64_t tid_m = threadIdx.y;
    int64_t tid_n = threadIdx.x;
    int64_t m0 = row_base + tid_m * 4;
    int64_t n0 = col_base + tid_n * 4;
    if (gid >= batch_groups || m0 >= chunk_size || n0 >= d_state) {
        return;
    }

    const float* x_ptr = x_mat + gid * chunk_size * hdim;
    const float* dstate_ptr = dstate + gid * d_state * hdim;
    float* dB_ptr = dB + gid * chunk_size * d_state;
    int64_t b = gid / (num_chunks * ngroups);
    int64_t rem = gid - b * num_chunks * ngroups;
    int64_t c = rem / ngroups;
    int64_t g = rem - c * ngroups;
    int64_t heads_per_group = nheads / ngroups;

    __shared__ float a_s[BM][BK];
    __shared__ float b_s[BK][BN];

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int64_t k0 = 0; k0 < hdim; k0 += BK) {
        int load_idx = threadIdx.y * 16 + threadIdx.x;
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BK;
            int col = idx - row * BK;
            if (row < BM && col < BK) {
                int64_t m = row_base + row;
                int64_t k = k0 + col;
                float val = 0.0f;
                if (m < chunk_size && k < hdim) {
                    int64_t head_in_group = k / headdim;
                    int64_t head = g * heads_per_group + head_in_group;
                    int64_t dt_idx = ((b * nheads + head) * num_chunks + c) * chunk_size + m;
                    float scale = dt[dt_idx] * expf(fminf(-dA_cumsum[dt_idx], 0.0f));
                    val = x_ptr[m * hdim + k] * scale;
                }
                a_s[row][col] = val;
            }
        }
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx - row * BN;
            if (row < BK && col < BN) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                float val = 0.0f;
                if (k < hdim && n < d_state) {
                    val = dstate_ptr[n * hdim + k];
                }
                b_s[row][col] = val;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = a_s[tid_m * 4 + i][k];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a * b_s[k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t m = m0 + i;
        if (m >= chunk_size) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t n = n0 + j;
            if (n < d_state) {
                dB_ptr[m * d_state + n] = acc[i][j];
            }
        }
    }
}

} // namespace

std::vector<torch::Tensor> mamba_fused_forward_cuda(
    const torch::Tensor& zxbcdt,
    const torch::Tensor& conv_w,
    const torch::Tensor& conv_b,
    const torch::Tensor& dt_bias,
    const torch::Tensor& a_log,
    const torch::Tensor& d_param,
    const torch::Tensor& dt_scale,
    const torch::Tensor& initial_state,
    int64_t chunk_size,
    int64_t ngroups,
    int64_t headdim,
    double dt_min,
    double dt_max) {
    at::NoGradGuard no_grad;
    auto z = zxbcdt.contiguous();
    auto w = conv_w.contiguous();
    auto b = conv_b.defined() ? conv_b.contiguous() : torch::Tensor();
    bool has_conv_bias = b.defined() && b.numel() > 0;
    auto dtb = dt_bias.contiguous();
    auto alog = a_log.contiguous();
    auto d = d_param.to(torch::kFloat).contiguous();
    bool d_has_hdim = d.dim() == 2;
    auto state = initial_state.contiguous();
    bool has_dt_scale = dt_scale.defined() && dt_scale.numel() > 0;
    auto dt_s = has_dt_scale ? dt_scale.expand({z.size(0), z.size(1), a_log.size(0)}).contiguous() : torch::Tensor();

    int64_t batch = z.size(0);
    int64_t seqlen = z.size(1);
    int64_t d_in_proj = z.size(2);
    int64_t nheads = alog.size(0);
    int64_t d_ssm = nheads * headdim;
    int64_t d_state = (w.size(0) - d_ssm) / (2 * ngroups);
    int64_t d_inner = (d_in_proj - 2 * ngroups * d_state - nheads) / 2;
    int64_t d_mlp = d_inner - d_ssm;
    int64_t conv_kernel = w.size(1);
    int64_t conv_dim = d_ssm + 2 * ngroups * d_state;

    TORCH_CHECK(nheads % ngroups == 0, "nheads must be divisible by ngroups");
    TORCH_CHECK(d_state <= kMaxDState, "d_state exceeds max supported");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    TORCH_CHECK(chunk_size <= 256, "chunk_size exceeds max supported");
    TORCH_CHECK(d_has_hdim ? (d.size(0) == nheads && d.size(1) == headdim) : (d.numel() == nheads),
        "D shape must be [nheads] or [nheads, headdim]");

    auto y = torch::zeros({batch, seqlen, nheads, headdim}, z.options());
    auto final_state = torch::zeros_like(state);
    int64_t num_chunks = (seqlen + chunk_size - 1) / chunk_size;
    auto dt_buf = torch::zeros({batch, nheads, num_chunks, chunk_size}, z.options().dtype(torch::kFloat));
    auto dA_buf = torch::zeros({batch, nheads, num_chunks, chunk_size}, z.options().dtype(torch::kFloat));
    auto exp_a_last = torch::zeros({batch, nheads, num_chunks}, z.options().dtype(torch::kFloat));
    auto state_f = state.to(torch::kFloat);
    bool debug = false;
    if (const char* dbg = std::getenv("MAMBA_FUSED_DEBUG")) {
        debug = std::string(dbg) == "1";
    }
    if (debug) {
        switch (z.scalar_type()) {
            case at::kFloat:
                check_tensor_finite<float>(z, "zxbcdt");
                check_tensor_finite<float>(w, "conv_w");
                check_tensor_finite<float>(dtb, "dt_bias");
                check_tensor_finite<float>(alog, "a_log");
                if (has_conv_bias) {
                    check_tensor_finite<float>(b, "conv_b");
                }
                if (has_dt_scale) {
                    check_tensor_finite<float>(dt_s, "dt_scale");
                }
                break;
            case at::kHalf:
                check_tensor_finite<at::Half>(z, "zxbcdt");
                check_tensor_finite<at::Half>(w, "conv_w");
                check_tensor_finite<at::Half>(dtb, "dt_bias");
                check_tensor_finite<at::Half>(alog, "a_log");
                if (has_conv_bias) {
                    check_tensor_finite<at::Half>(b, "conv_b");
                }
                if (has_dt_scale) {
                    check_tensor_finite<at::Half>(dt_s, "dt_scale");
                }
                break;
            case at::kBFloat16:
                check_tensor_finite<at::BFloat16>(z, "zxbcdt");
                check_tensor_finite<at::BFloat16>(w, "conv_w");
                check_tensor_finite<at::BFloat16>(dtb, "dt_bias");
                check_tensor_finite<at::BFloat16>(alog, "a_log");
                if (has_conv_bias) {
                    check_tensor_finite<at::BFloat16>(b, "conv_b");
                }
                if (has_dt_scale) {
                    check_tensor_finite<at::BFloat16>(dt_s, "dt_scale");
                }
                break;
            default:
                break;
        }
    }
    const int threads = 256;

    dim3 dt_grid(batch, nheads, num_chunks);
    int dt_threads = 1;
    while (dt_threads < chunk_size) {
        dt_threads <<= 1;
    }
    size_t dt_shared = dt_threads * sizeof(float);
    switch (z.scalar_type()) {
        case at::kFloat:
            dt_cumsum_kernel<float><<<dt_grid, dt_threads, dt_shared>>>(
                z.data_ptr<float>(),
                dtb.data_ptr<float>(),
                alog.data_ptr<float>(),
                has_dt_scale ? dt_s.data_ptr<float>() : nullptr,
                dt_buf.data_ptr<float>(),
                dA_buf.data_ptr<float>(),
                exp_a_last.data_ptr<float>(),
                batch,
                seqlen,
                nheads,
                d_in_proj,
                d_mlp,
                static_cast<float>(dt_min),
                static_cast<float>(dt_max),
                chunk_size,
                has_dt_scale);
            break;
        case at::kHalf:
            dt_cumsum_kernel<at::Half><<<dt_grid, dt_threads, dt_shared>>>(
                z.data_ptr<at::Half>(),
                dtb.data_ptr<at::Half>(),
                alog.data_ptr<at::Half>(),
                has_dt_scale ? dt_s.data_ptr<at::Half>() : nullptr,
                dt_buf.data_ptr<float>(),
                dA_buf.data_ptr<float>(),
                exp_a_last.data_ptr<float>(),
                batch,
                seqlen,
                nheads,
                d_in_proj,
                d_mlp,
                static_cast<float>(dt_min),
                static_cast<float>(dt_max),
                chunk_size,
                has_dt_scale);
            break;
        case at::kBFloat16:
            dt_cumsum_kernel<at::BFloat16><<<dt_grid, dt_threads, dt_shared>>>(
                z.data_ptr<at::BFloat16>(),
                dtb.data_ptr<at::BFloat16>(),
                alog.data_ptr<at::BFloat16>(),
                has_dt_scale ? dt_s.data_ptr<at::BFloat16>() : nullptr,
                dt_buf.data_ptr<float>(),
                dA_buf.data_ptr<float>(),
                exp_a_last.data_ptr<float>(),
                batch,
                seqlen,
                nheads,
                d_in_proj,
                d_mlp,
                static_cast<float>(dt_min),
                static_cast<float>(dt_max),
                chunk_size,
                has_dt_scale);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype for dt_cumsum_kernel");
    }
    if (debug) {
        check_tensor_finite<float>(dt_buf, "dt");
        check_tensor_finite<float>(dA_buf, "dA_cumsum");
    }

    int64_t heads_per_group = nheads / ngroups;
    auto x_buf = torch::zeros({batch, num_chunks, chunk_size, nheads, headdim}, z.options().dtype(torch::kFloat));
    auto b_buf = torch::zeros({batch, num_chunks, chunk_size, ngroups, d_state}, z.options().dtype(torch::kFloat));
    auto c_buf = torch::zeros({batch, num_chunks, chunk_size, ngroups, d_state}, z.options().dtype(torch::kFloat));
    int64_t pack_total = batch * seqlen * conv_dim;
    int pack_blocks = (pack_total + threads - 1) / threads;
    switch (z.scalar_type()) {
        case at::kFloat:
            conv1d_pack_kernel<float><<<pack_blocks, threads>>>(
                z.data_ptr<float>(),
                w.data_ptr<float>(),
                has_conv_bias ? b.data_ptr<float>() : nullptr,
                x_buf.data_ptr<float>(),
                b_buf.data_ptr<float>(),
                c_buf.data_ptr<float>(),
                batch,
                seqlen,
                d_in_proj,
                conv_dim,
                conv_kernel,
                d_mlp,
                d_ssm,
                ngroups,
                d_state,
                chunk_size,
                num_chunks,
                headdim,
                has_conv_bias);
            break;
        case at::kHalf:
            conv1d_pack_kernel<at::Half><<<pack_blocks, threads>>>(
                z.data_ptr<at::Half>(),
                w.data_ptr<at::Half>(),
                has_conv_bias ? b.data_ptr<at::Half>() : nullptr,
                x_buf.data_ptr<float>(),
                b_buf.data_ptr<float>(),
                c_buf.data_ptr<float>(),
                batch,
                seqlen,
                d_in_proj,
                conv_dim,
                conv_kernel,
                d_mlp,
                d_ssm,
                ngroups,
                d_state,
                chunk_size,
                num_chunks,
                headdim,
                has_conv_bias);
            break;
        case at::kBFloat16:
            conv1d_pack_kernel<at::BFloat16><<<pack_blocks, threads>>>(
                z.data_ptr<at::BFloat16>(),
                w.data_ptr<at::BFloat16>(),
                has_conv_bias ? b.data_ptr<at::BFloat16>() : nullptr,
                x_buf.data_ptr<float>(),
                b_buf.data_ptr<float>(),
                c_buf.data_ptr<float>(),
                batch,
                seqlen,
                d_in_proj,
                conv_dim,
                conv_kernel,
                d_mlp,
                d_ssm,
                ngroups,
                d_state,
                chunk_size,
                num_chunks,
                headdim,
                has_conv_bias);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype for conv1d_pack_kernel");
    }
    auto x_all = x_buf;
    auto b_all = b_buf;
    auto c_all = c_buf;

    auto x_g_mat = x_all.view({batch, num_chunks, chunk_size, ngroups, heads_per_group, headdim})
                       .permute({0, 1, 3, 2, 4, 5})
                       .contiguous()
                       .view({batch * num_chunks * ngroups, chunk_size, heads_per_group * headdim});
    auto b_mat = b_all.permute({0, 1, 3, 2, 4})
                  .contiguous()
                  .view({batch * num_chunks * ngroups, chunk_size, d_state});
    auto c_mat = c_all.permute({0, 1, 3, 2, 4})
                  .contiguous()
                  .view({batch * num_chunks * ngroups, chunk_size, d_state});

    auto cb = torch::zeros(
        {batch * num_chunks * ngroups, chunk_size, chunk_size},
        x_all.options().dtype(torch::kFloat));
    auto chunk_state = torch::zeros(
        {batch * num_chunks * ngroups, d_state, heads_per_group * headdim},
        x_all.options().dtype(torch::kFloat));
    dim3 bmm_threads(16, 16);
    dim3 cb_grid((chunk_size + 63) / 64, (chunk_size + 63) / 64, batch * num_chunks * ngroups);
    bmm_mk_nk_kernel<<<cb_grid, bmm_threads>>>(
        c_mat.data_ptr<float>(),
        b_mat.data_ptr<float>(),
        cb.data_ptr<float>(),
        batch * num_chunks * ngroups,
        chunk_size,
        d_state,
        chunk_size);
    dim3 cs_grid((heads_per_group * headdim + 63) / 64, (d_state + 63) / 64, batch * num_chunks * ngroups);
    bmm_kt_kn_scale_x_kernel<<<cs_grid, bmm_threads>>>(
        b_mat.data_ptr<float>(),
        x_g_mat.data_ptr<float>(),
        dt_buf.data_ptr<float>(),
        dA_buf.data_ptr<float>(),
        chunk_state.data_ptr<float>(),
        batch * num_chunks * ngroups,
        d_state,
        chunk_size,
        heads_per_group * headdim,
        num_chunks,
        ngroups,
        nheads,
        headdim,
        chunk_size);

    auto chunk_state_view = chunk_state.transpose(1, 2)
                                .contiguous()
                                .view({batch, num_chunks, ngroups, heads_per_group, headdim, d_state})
                                .view({batch, num_chunks, nheads, headdim, d_state});
    auto state_in = torch::zeros_like(chunk_state_view);
    auto final_state_f = torch::zeros_like(state_f);
    int64_t state_total = batch * nheads * headdim * d_state;
    int state_blocks = (state_total + threads - 1) / threads;
    state_passing_fwd_kernel<<<state_blocks, threads>>>(
        chunk_state_view.data_ptr<float>(),
        exp_a_last.data_ptr<float>(),
        state_f.data_ptr<float>(),
        state_in.data_ptr<float>(),
        final_state_f.data_ptr<float>(),
        batch,
        nheads,
        headdim,
        d_state,
        num_chunks);

    int64_t padded_len = num_chunks * chunk_size;
    auto y_padded_f = torch::zeros({batch, padded_len, nheads, headdim}, x_all.options().dtype(torch::kFloat));
    dim3 block(16, 16);
    dim3 grid((headdim + 63) / 64, (chunk_size + 63) / 64, batch * num_chunks * nheads);
    chunk_scan_fwd_kernel<<<grid, block>>>(
        cb.contiguous().data_ptr<float>(),
        x_all.contiguous().data_ptr<float>(),
        dt_buf.contiguous().data_ptr<float>(),
        dA_buf.contiguous().data_ptr<float>(),
        c_all.contiguous().data_ptr<float>(),
        state_in.contiguous().data_ptr<float>(),
        d.data_ptr<float>(),
        d_has_hdim ? 1 : 0,
        y_padded_f.data_ptr<float>(),
        batch,
        num_chunks,
        chunk_size,
        nheads,
        headdim,
        d_state,
        ngroups);
    y.copy_(y_padded_f.slice(1, 0, seqlen).to(y.scalar_type()));
    state_f.copy_(final_state_f);

    if (debug) {
        switch (z.scalar_type()) {
            case at::kFloat:
                check_tensor_finite<float>(y, "y_chunk");
                break;
            case at::kHalf:
                check_tensor_finite<at::Half>(y, "y_chunk");
                break;
            case at::kBFloat16:
                check_tensor_finite<at::BFloat16>(y, "y_chunk");
                break;
            default:
                break;
        }
    }

    final_state.copy_(state_f.to(state.scalar_type()));
    if (debug) {
        switch (z.scalar_type()) {
            case at::kFloat:
                check_tensor_finite<float>(y, "y_final");
                break;
            case at::kHalf:
                check_tensor_finite<at::Half>(y, "y_final");
                break;
            case at::kBFloat16:
                check_tensor_finite<at::BFloat16>(y, "y_final");
                break;
            default:
                break;
        }
    }

    return {y, final_state};
}

std::vector<torch::Tensor> mamba_fused_backward_cuda(
    const torch::Tensor& zxbcdt,
    const torch::Tensor& conv_w,
    const torch::Tensor& conv_b,
    const torch::Tensor& dt_bias,
    const torch::Tensor& a_log,
    const torch::Tensor& d_param,
    const torch::Tensor& dt_scale,
    const torch::Tensor& initial_state,
    const torch::Tensor& final_state,
    const torch::Tensor& y,
    const torch::Tensor& grad_y,
    const torch::Tensor& grad_final_state,
    int64_t chunk_size,
    int64_t ngroups,
    int64_t headdim,
    double dt_min,
    double dt_max) {
    at::NoGradGuard no_grad;
    auto z = zxbcdt.contiguous();
    auto w = conv_w.contiguous();
    auto b = conv_b.defined() ? conv_b.contiguous() : torch::Tensor();
    bool has_conv_bias = b.defined() && b.numel() > 0;
    auto dtb = dt_bias.contiguous();
    auto alog = a_log.contiguous();
    auto d = d_param.to(torch::kFloat).contiguous();
    bool d_has_hdim = d.dim() == 2;
    auto state0 = initial_state.contiguous();
    auto stateN = final_state.contiguous();
    auto state0_f = state0.to(torch::kFloat);
    bool has_dt_scale = dt_scale.defined() && dt_scale.numel() > 0;
    auto dt_s = has_dt_scale ? dt_scale.expand({z.size(0), z.size(1), a_log.size(0)}).contiguous() : torch::Tensor();
    auto gy = grad_y.contiguous();
    auto gy_f = gy.to(torch::kFloat);
    auto y_f = y.to(torch::kFloat);
    auto gstate = grad_final_state.defined() ? grad_final_state.contiguous() : torch::zeros_like(stateN);

    int64_t batch = z.size(0);
    int64_t seqlen = z.size(1);
    int64_t d_in_proj = z.size(2);
    int64_t nheads = alog.size(0);
    int64_t d_ssm = nheads * headdim;
    int64_t d_state = (w.size(0) - d_ssm) / (2 * ngroups);
    int64_t d_inner = (d_in_proj - 2 * ngroups * d_state - nheads) / 2;
    int64_t d_mlp = d_inner - d_ssm;
    int64_t conv_kernel = w.size(1);
    int64_t conv_dim = d_ssm + 2 * ngroups * d_state;

    TORCH_CHECK(nheads % ngroups == 0, "nheads must be divisible by ngroups");
    TORCH_CHECK(d_state <= kMaxDState, "d_state exceeds max supported");
    TORCH_CHECK(d_has_hdim ? (d.size(0) == nheads && d.size(1) == headdim) : (d.numel() == nheads),
        "D shape must be [nheads] or [nheads, headdim]");

    auto conv_out = depthwise_conv1d_silu(z, w, b, d_mlp, d_ssm);
    int64_t num_chunks = (seqlen + chunk_size - 1) / chunk_size;
    auto dt_buf = torch::zeros({batch, nheads, num_chunks, chunk_size}, z.options().dtype(torch::kFloat));
    auto dA_buf = torch::zeros({batch, nheads, num_chunks, chunk_size}, z.options().dtype(torch::kFloat));
    auto exp_a_last = torch::zeros({batch, nheads, num_chunks}, z.options().dtype(torch::kFloat));
    const int threads = 256;

    dim3 dt_grid(batch, nheads, num_chunks);
    int dt_threads = 1;
    while (dt_threads < chunk_size) {
        dt_threads <<= 1;
    }
    size_t dt_shared = dt_threads * sizeof(float);

    switch (z.scalar_type()) {
        case at::kFloat:
            dt_cumsum_kernel<float><<<dt_grid, dt_threads, dt_shared>>>(
                z.data_ptr<float>(),
                dtb.data_ptr<float>(),
                alog.data_ptr<float>(),
                has_dt_scale ? dt_s.data_ptr<float>() : nullptr,
                dt_buf.data_ptr<float>(),
                dA_buf.data_ptr<float>(),
                exp_a_last.data_ptr<float>(),
                batch,
                seqlen,
                nheads,
                d_in_proj,
                d_mlp,
                static_cast<float>(dt_min),
                static_cast<float>(dt_max),
                chunk_size,
                has_dt_scale);
            break;
        case at::kHalf:
            dt_cumsum_kernel<at::Half><<<dt_grid, dt_threads, dt_shared>>>(
                z.data_ptr<at::Half>(),
                dtb.data_ptr<at::Half>(),
                alog.data_ptr<at::Half>(),
                has_dt_scale ? dt_s.data_ptr<at::Half>() : nullptr,
                dt_buf.data_ptr<float>(),
                dA_buf.data_ptr<float>(),
                exp_a_last.data_ptr<float>(),
                batch,
                seqlen,
                nheads,
                d_in_proj,
                d_mlp,
                static_cast<float>(dt_min),
                static_cast<float>(dt_max),
                chunk_size,
                has_dt_scale);
            break;
        case at::kBFloat16:
            dt_cumsum_kernel<at::BFloat16><<<dt_grid, dt_threads, dt_shared>>>(
                z.data_ptr<at::BFloat16>(),
                dtb.data_ptr<at::BFloat16>(),
                alog.data_ptr<at::BFloat16>(),
                has_dt_scale ? dt_s.data_ptr<at::BFloat16>() : nullptr,
                dt_buf.data_ptr<float>(),
                dA_buf.data_ptr<float>(),
                exp_a_last.data_ptr<float>(),
                batch,
                seqlen,
                nheads,
                d_in_proj,
                d_mlp,
                static_cast<float>(dt_min),
                static_cast<float>(dt_max),
                chunk_size,
                has_dt_scale);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype for dt_cumsum_kernel");
    }

    int64_t heads_per_group = nheads / ngroups;
    int64_t padded_len = num_chunks * chunk_size;
    auto x_buf = torch::zeros({batch, num_chunks, chunk_size, nheads, headdim}, conv_out.options().dtype(torch::kFloat));
    auto b_buf = torch::zeros({batch, num_chunks, chunk_size, ngroups, d_state}, conv_out.options().dtype(torch::kFloat));
    auto c_buf = torch::zeros({batch, num_chunks, chunk_size, ngroups, d_state}, conv_out.options().dtype(torch::kFloat));
    int64_t pack_total = batch * num_chunks * chunk_size * conv_dim;
    int pack_blocks = (pack_total + threads - 1) / threads;
    switch (conv_out.scalar_type()) {
        case at::kFloat:
            pack_conv_out_kernel<float><<<pack_blocks, threads>>>(
                conv_out.data_ptr<float>(),
                x_buf.data_ptr<float>(),
                b_buf.data_ptr<float>(),
                c_buf.data_ptr<float>(),
                batch,
                seqlen,
                num_chunks,
                chunk_size,
                nheads,
                headdim,
                ngroups,
                d_state,
                conv_dim,
                d_ssm);
            break;
        case at::kHalf:
            pack_conv_out_kernel<at::Half><<<pack_blocks, threads>>>(
                conv_out.data_ptr<at::Half>(),
                x_buf.data_ptr<float>(),
                b_buf.data_ptr<float>(),
                c_buf.data_ptr<float>(),
                batch,
                seqlen,
                num_chunks,
                chunk_size,
                nheads,
                headdim,
                ngroups,
                d_state,
                conv_dim,
                d_ssm);
            break;
        case at::kBFloat16:
            pack_conv_out_kernel<at::BFloat16><<<pack_blocks, threads>>>(
                conv_out.data_ptr<at::BFloat16>(),
                x_buf.data_ptr<float>(),
                b_buf.data_ptr<float>(),
                c_buf.data_ptr<float>(),
                batch,
                seqlen,
                num_chunks,
                chunk_size,
                nheads,
                headdim,
                ngroups,
                d_state,
                conv_dim,
                d_ssm);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype for pack_conv_out_kernel");
    }
    auto x_all = x_buf;
    auto b_all = b_buf;
    auto c_all = c_buf;

    int64_t x_total = batch * num_chunks * chunk_size * nheads * headdim;
    int x_blocks = (x_total + threads - 1) / threads;
    auto x_g_mat = x_all.view({batch, num_chunks, chunk_size, ngroups, heads_per_group, headdim})
                       .permute({0, 1, 3, 2, 4, 5})
                       .contiguous()
                       .view({batch * num_chunks * ngroups, chunk_size, heads_per_group * headdim});
    auto b_mat = b_all.permute({0, 1, 3, 2, 4})
                  .contiguous()
                  .view({batch * num_chunks * ngroups, chunk_size, d_state});
    auto c_mat = c_all.permute({0, 1, 3, 2, 4})
                  .contiguous()
                  .view({batch * num_chunks * ngroups, chunk_size, d_state});

    auto chunk_state_mat = torch::zeros(
        {batch * num_chunks * ngroups, d_state, heads_per_group * headdim},
        x_all.options().dtype(torch::kFloat));
    dim3 bmm_threads(16, 16);
    dim3 cs_grid((heads_per_group * headdim + 63) / 64, (d_state + 63) / 64, batch * num_chunks * ngroups);
    bmm_kt_kn_scale_x_kernel<<<cs_grid, bmm_threads>>>(
        b_mat.data_ptr<float>(),
        x_g_mat.data_ptr<float>(),
        dt_buf.data_ptr<float>(),
        dA_buf.data_ptr<float>(),
        chunk_state_mat.data_ptr<float>(),
        batch * num_chunks * ngroups,
        d_state,
        chunk_size,
        heads_per_group * headdim,
        num_chunks,
        ngroups,
        nheads,
        headdim,
        chunk_size);
    auto chunk_state = chunk_state_mat.transpose(1, 2)
                           .contiguous()
                           .view({batch, num_chunks, ngroups, heads_per_group, headdim, d_state})
                           .view({batch, num_chunks, nheads, headdim, d_state});

    auto state_in = torch::zeros_like(chunk_state);
    auto final_state_f = torch::zeros_like(state0, state0.options().dtype(torch::kFloat));
    int64_t state_total = batch * nheads * headdim * d_state;
    int state_blocks = (state_total + threads - 1) / threads;
    state_passing_fwd_kernel<<<state_blocks, threads>>>(
        chunk_state.data_ptr<float>(),
        exp_a_last.data_ptr<float>(),
        state0_f.data_ptr<float>(),
        state_in.data_ptr<float>(),
        final_state_f.data_ptr<float>(),
        batch,
        nheads,
        headdim,
        d_state,
        num_chunks);

    auto gy_chunk = torch::zeros({batch, num_chunks, chunk_size, nheads, headdim}, gy_f.options());
    int64_t gy_total = batch * num_chunks * chunk_size * nheads * headdim;
    int gy_blocks = (gy_total + threads - 1) / threads;
    switch (gy.scalar_type()) {
        case at::kFloat:
            pack_gy_kernel<float><<<gy_blocks, threads>>>(
                gy.data_ptr<float>(),
                gy_chunk.data_ptr<float>(),
                batch,
                seqlen,
                num_chunks,
                chunk_size,
                nheads,
                headdim);
            break;
        case at::kHalf:
            pack_gy_kernel<at::Half><<<gy_blocks, threads>>>(
                gy.data_ptr<at::Half>(),
                gy_chunk.data_ptr<float>(),
                batch,
                seqlen,
                num_chunks,
                chunk_size,
                nheads,
                headdim);
            break;
        case at::kBFloat16:
            pack_gy_kernel<at::BFloat16><<<gy_blocks, threads>>>(
                gy.data_ptr<at::BFloat16>(),
                gy_chunk.data_ptr<float>(),
                batch,
                seqlen,
                num_chunks,
                chunk_size,
                nheads,
                headdim);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype for pack_gy_kernel");
    }
    auto gy_g = gy_chunk.view({batch, num_chunks, chunk_size, ngroups, heads_per_group, headdim});
    auto gy_g_mat = gy_g.permute({0, 1, 3, 2, 4, 5})
                           .contiguous()
                           .view({batch * num_chunks * ngroups, chunk_size, heads_per_group * headdim});

    auto state_in_g = state_in.view({batch, num_chunks, ngroups, heads_per_group, headdim, d_state});
    auto state_in_g_mat = state_in_g.permute({0, 1, 2, 5, 3, 4})
                               .contiguous()
                               .view({batch * num_chunks * ngroups, d_state, heads_per_group * headdim});
    auto dchunk_state_y = torch::zeros(
        {batch * num_chunks * ngroups, d_state, heads_per_group * headdim},
        gy_f.options());
    auto dC_total = torch::zeros(
        {batch * num_chunks * ngroups, chunk_size, d_state},
        gy_f.options());
    dim3 bwd_c_threads(16, 16);
    dim3 bwd_c_grid(
        (d_state + 63) / 64,
        (chunk_size + 63) / 64,
        batch * num_chunks * ngroups);
    chunk_scan_bwd_dC_dstate_kernel<<<bwd_c_grid, bwd_c_threads>>>(
        gy_g_mat.data_ptr<float>(),
        dA_buf.data_ptr<float>(),
        c_mat.data_ptr<float>(),
        state_in_g_mat.data_ptr<float>(),
        chunk_state_mat.data_ptr<float>(),
        dC_total.data_ptr<float>(),
        dchunk_state_y.data_ptr<float>(),
        batch * num_chunks * ngroups,
        chunk_size,
        heads_per_group * headdim,
        d_state,
        num_chunks,
        ngroups,
        nheads,
        headdim);
    auto dchunk_state_y_t = dchunk_state_y.transpose(1, 2).contiguous();
    auto dstate_in = dchunk_state_y_t.view({batch, num_chunks, ngroups, heads_per_group, headdim, d_state})
                         .view({batch, num_chunks, nheads, headdim, d_state});
    auto dchunk_state = torch::zeros_like(chunk_state);
    auto ddA = torch::zeros({batch, nheads, seqlen}, z.options().dtype(torch::kFloat));
    auto dstate0 = torch::zeros_like(state0, state0.options().dtype(torch::kFloat));
    auto gstate_f = gstate.to(torch::kFloat);
    state_passing_bwd_kernel<<<state_blocks, threads>>>(
        chunk_state.data_ptr<float>(),
        state_in.data_ptr<float>(),
        exp_a_last.data_ptr<float>(),
        dstate_in.data_ptr<float>(),
        gstate_f.data_ptr<float>(),
        dchunk_state.data_ptr<float>(),
        ddA.data_ptr<float>(),
        dstate0.data_ptr<float>(),
        batch,
        nheads,
        headdim,
        d_state,
        seqlen,
        chunk_size,
        num_chunks);

    auto dchunk_state_total = (dchunk_state + dstate_in).contiguous();
    auto dchunk_state_mat = dchunk_state_total.view({batch, num_chunks, ngroups, heads_per_group, headdim, d_state})
                               .permute({0, 1, 2, 5, 3, 4})
                               .contiguous()
                               .view({batch * num_chunks * ngroups, d_state, heads_per_group * headdim});
    auto dx_scaled_state = torch::zeros(
        {batch * num_chunks * ngroups, chunk_size, heads_per_group * headdim},
        gy_f.options());
    auto dtemp_g = torch::zeros(
        {batch * num_chunks * ngroups, chunk_size, heads_per_group},
        gy_f.options());
    auto ddt_chunk_g = torch::zeros(
        {batch * num_chunks * ngroups, chunk_size, heads_per_group},
        gy_f.options());
    auto ddA_xscaled_g = torch::zeros(
        {batch * num_chunks * ngroups, chunk_size, heads_per_group},
        gy_f.options());
    dim3 bwd_dx_threads(16, 16);
    dim3 bwd_dx_grid(
        (heads_per_group * headdim + 63) / 64,
        (chunk_size + 63) / 64,
        batch * num_chunks * ngroups);
    chunk_scan_bwd_dx_kernel<<<bwd_dx_grid, bwd_dx_threads>>>(
        b_mat.data_ptr<float>(),
        dchunk_state_mat.data_ptr<float>(),
        x_g_mat.data_ptr<float>(),
        dt_buf.data_ptr<float>(),
        dA_buf.data_ptr<float>(),
        dx_scaled_state.data_ptr<float>(),
        dtemp_g.data_ptr<float>(),
        ddt_chunk_g.data_ptr<float>(),
        ddA_xscaled_g.data_ptr<float>(),
        batch * num_chunks * ngroups,
        chunk_size,
        d_state,
        heads_per_group * headdim,
        headdim);
    auto dB_state = torch::zeros(
        {batch * num_chunks * ngroups, chunk_size, d_state},
        gy_f.options());
    dim3 bwd_dB_threads(16, 16);
    dim3 bwd_dB_grid(
        (d_state + 63) / 64,
        (chunk_size + 63) / 64,
        batch * num_chunks * ngroups);
    chunk_scan_bwd_dB_kernel<<<bwd_dB_grid, bwd_dB_threads>>>(
        x_g_mat.data_ptr<float>(),
        dt_buf.data_ptr<float>(),
        dA_buf.data_ptr<float>(),
        dchunk_state_mat.data_ptr<float>(),
        dB_state.data_ptr<float>(),
        batch * num_chunks * ngroups,
        chunk_size,
        heads_per_group * headdim,
        d_state,
        num_chunks,
        ngroups,
        nheads,
        headdim);
    auto dx_scaled = dx_scaled_state.view({batch, num_chunks, ngroups, chunk_size, heads_per_group, headdim})
                         .permute({0, 1, 3, 2, 4, 5})
                         .contiguous()
                         .view({batch, num_chunks, chunk_size, nheads, headdim});
    auto dx = torch::zeros_like(dx_scaled);
    x_scale_kernel<<<x_blocks, threads>>>(
        dx_scaled.data_ptr<float>(),
        dt_buf.data_ptr<float>(),
        dA_buf.data_ptr<float>(),
        dx.data_ptr<float>(),
        batch,
        num_chunks,
        chunk_size,
        nheads,
        headdim);
    auto dx_conv = dx.view({batch, padded_len, nheads, headdim}).slice(1, 0, seqlen).contiguous();
    auto x_conv = x_all.view({batch, padded_len, nheads, headdim}).slice(1, 0, seqlen).contiguous();
    auto x_conv_f = x_conv.to(torch::kFloat);
    auto d_broadcast = d_has_hdim ? d.view({1, 1, nheads, headdim}) : d.view({1, 1, nheads, 1});

    auto dtemp = dtemp_g.view({batch, num_chunks, ngroups, chunk_size, heads_per_group})
                     .permute({0, 1, 3, 2, 4})
                     .contiguous()
                     .view({batch, num_chunks, chunk_size, nheads});
    auto ddt_chunk = ddt_chunk_g.view({batch, num_chunks, ngroups, chunk_size, heads_per_group})
                         .permute({0, 1, 3, 2, 4})
                         .contiguous()
                         .view({batch, num_chunks, chunk_size, nheads});
    auto ddA_xscaled = ddA_xscaled_g.view({batch, num_chunks, ngroups, chunk_size, heads_per_group})
                           .permute({0, 1, 3, 2, 4})
                           .contiguous()
                           .view({batch, num_chunks, chunk_size, nheads});
    auto ddt = ddt_chunk.permute({0, 3, 1, 2})
                   .contiguous()
                   .view({batch, nheads, padded_len})
                   .slice(2, 0, seqlen)
                   .contiguous();
    ddA += ddA_xscaled.permute({0, 3, 1, 2})
               .contiguous()
               .view({batch, nheads, padded_len})
               .slice(2, 0, seqlen)
               .contiguous();
    auto y_scan = y_f - x_conv_f * d_broadcast;
    int64_t ddA_total = batch * seqlen * nheads;
    int ddA_blocks = (ddA_total + threads - 1) / threads;
    ddA_y_bwd_kernel<<<ddA_blocks, threads>>>(
        gy_f.data_ptr<float>(),
        y_scan.data_ptr<float>(),
        ddA.data_ptr<float>(),
        batch,
        seqlen,
        nheads,
        headdim);

    auto dB = dB_state.view({batch, num_chunks, ngroups, chunk_size, d_state})
                  .view({batch, padded_len, ngroups, d_state})
                  .slice(1, 0, seqlen)
                  .contiguous();
    auto dC = dC_total.view({batch, num_chunks, ngroups, chunk_size, d_state})
                  .view({batch, padded_len, ngroups, d_state})
                  .slice(1, 0, seqlen)
                  .contiguous();
    auto dA = torch::zeros({nheads}, z.options().dtype(torch::kFloat));
    auto dD = d_has_hdim ? torch::zeros({nheads, headdim}, z.options().dtype(torch::kFloat))
                         : torch::zeros({nheads}, z.options().dtype(torch::kFloat));
    auto ddt_bias = torch::zeros({nheads}, z.options().dtype(torch::kFloat));

    dim3 ddA_grid(batch, nheads, num_chunks);
    size_t ddA_shared = dt_threads * sizeof(float);
    switch (z.scalar_type()) {
        case at::kFloat:
            ddA_to_dtdA_kernel<float><<<ddA_grid, dt_threads, ddA_shared>>>(
                ddA.data_ptr<float>(),
                dt_buf.data_ptr<float>(),
                alog.data_ptr<float>(),
                ddt.data_ptr<float>(),
                dA.data_ptr<float>(),
                batch,
                seqlen,
                nheads,
                chunk_size);
            break;
        case at::kHalf:
            ddA_to_dtdA_kernel<at::Half><<<ddA_grid, dt_threads, ddA_shared>>>(
                ddA.data_ptr<float>(),
                dt_buf.data_ptr<float>(),
                alog.data_ptr<at::Half>(),
                ddt.data_ptr<float>(),
                dA.data_ptr<float>(),
                batch,
                seqlen,
                nheads,
                chunk_size);
            break;
        case at::kBFloat16:
            ddA_to_dtdA_kernel<at::BFloat16><<<ddA_grid, dt_threads, ddA_shared>>>(
                ddA.data_ptr<float>(),
                dt_buf.data_ptr<float>(),
                alog.data_ptr<at::BFloat16>(),
                ddt.data_ptr<float>(),
                dA.data_ptr<float>(),
                batch,
                seqlen,
                nheads,
                chunk_size);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype for ddA_to_dtdA_kernel");
    }

    auto ddt_raw = torch::zeros({batch, seqlen, nheads}, z.options().dtype(torch::kFloat));
    int64_t ddt_total = batch * seqlen * nheads;
    int ddt_blocks = (ddt_total + threads - 1) / threads;
    switch (z.scalar_type()) {
        case at::kFloat:
            ddt_raw_kernel<float><<<ddt_blocks, threads>>>(
                ddt.data_ptr<float>(),
                z.data_ptr<float>(),
                dtb.data_ptr<float>(),
                has_dt_scale ? dt_s.data_ptr<float>() : nullptr,
                ddt_raw.data_ptr<float>(),
                ddt_bias.data_ptr<float>(),
                batch,
                seqlen,
                nheads,
                d_in_proj,
                static_cast<float>(dt_min),
                static_cast<float>(dt_max),
                has_dt_scale);
            break;
        case at::kHalf:
            ddt_raw_kernel<at::Half><<<ddt_blocks, threads>>>(
                ddt.data_ptr<float>(),
                z.data_ptr<at::Half>(),
                dtb.data_ptr<at::Half>(),
                has_dt_scale ? dt_s.data_ptr<at::Half>() : nullptr,
                ddt_raw.data_ptr<float>(),
                ddt_bias.data_ptr<float>(),
                batch,
                seqlen,
                nheads,
                d_in_proj,
                static_cast<float>(dt_min),
                static_cast<float>(dt_max),
                has_dt_scale);
            break;
        case at::kBFloat16:
            ddt_raw_kernel<at::BFloat16><<<ddt_blocks, threads>>>(
                ddt.data_ptr<float>(),
                z.data_ptr<at::BFloat16>(),
                dtb.data_ptr<at::BFloat16>(),
                has_dt_scale ? dt_s.data_ptr<at::BFloat16>() : nullptr,
                ddt_raw.data_ptr<float>(),
                ddt_bias.data_ptr<float>(),
                batch,
                seqlen,
                nheads,
                d_in_proj,
                static_cast<float>(dt_min),
                static_cast<float>(dt_max),
                has_dt_scale);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype for ddt_raw_kernel");
    }

    int64_t dy_total = batch * seqlen * nheads * headdim;
    int dy_blocks = (dy_total + threads - 1) / threads;
    d_skip_dx_kernel<<<dy_blocks, threads>>>(
        gy_f.data_ptr<float>(),
        x_conv_f.data_ptr<float>(),
        d.data_ptr<float>(),
        dx_conv.data_ptr<float>(),
        batch,
        seqlen,
        nheads,
        headdim,
        d_has_hdim ? 1 : 0);
    int64_t dD_dim = d_has_hdim ? (nheads * headdim) : nheads;
    dim3 dD_grid(dD_dim, 8);
    d_skip_dD_kernel<<<dD_grid, 256>>>(
        gy_f.data_ptr<float>(),
        x_conv_f.data_ptr<float>(),
        dD.data_ptr<float>(),
        batch,
        seqlen,
        nheads,
        headdim,
        d_has_hdim ? 1 : 0);

    auto d_xbc_conv = torch::zeros({batch, seqlen, conv_dim}, z.options().dtype(torch::kFloat));
    int64_t xbc_total = batch * seqlen * conv_dim;
    int xbc_blocks = (xbc_total + threads - 1) / threads;

    scatter_xbc_grad_kernel<float><<<xbc_blocks, threads>>>(
        dx_conv.data_ptr<float>(),
        dB.data_ptr<float>(),
        dC.data_ptr<float>(),
        d_xbc_conv.data_ptr<float>(),
        batch,
        seqlen,
        nheads,
        headdim,
        d_state,
        ngroups,
        d_ssm);

    auto conv_pre = depthwise_conv1d_pre(z, w, b, d_mlp, d_ssm);
    auto sig = torch::sigmoid(conv_pre);
    auto silu_grad = sig * (1.0f + conv_pre * (1.0f - sig));
    auto d_xbc_pre = d_xbc_conv * silu_grad;

    int64_t offset_xbc = 2 * d_mlp + d_ssm;
    auto xbc = z.narrow(2, offset_xbc, conv_dim).transpose(1, 2).contiguous();
    auto weight = w.view({conv_dim, 1, conv_kernel});
    auto xbc_pad = at::constant_pad_nd(xbc, {conv_kernel - 1, 0});
    auto grad_out = d_xbc_pre.permute({0, 2, 1}).contiguous().to(z.scalar_type());
    c10::optional<at::IntArrayRef> bias_sizes = has_conv_bias ? c10::optional<at::IntArrayRef>(b.sizes()) : c10::nullopt;
    std::vector<int64_t> stride = {1};
    std::vector<int64_t> padding = {0};
    std::vector<int64_t> dilation = {1};
    std::vector<int64_t> out_padding = {0};
    auto conv_grads = at::convolution_backward(
        grad_out,
        xbc_pad,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        false,
        out_padding,
        conv_dim,
        {true, true, has_conv_bias});

    auto d_xbc_in = std::get<0>(conv_grads).slice(2, conv_kernel - 1, conv_kernel - 1 + seqlen)
                        .transpose(1, 2)
                        .contiguous()
                        .to(torch::kFloat);
    auto d_conv_w = std::get<1>(conv_grads).view_as(w).to(torch::kFloat);
    auto d_conv_b = has_conv_bias ? std::get<2>(conv_grads).to(torch::kFloat)
                                  : torch::zeros({conv_dim}, z.options().dtype(torch::kFloat));

    auto dzxbcdt = torch::zeros_like(z, z.options().dtype(torch::kFloat));
    int64_t offset_dt = offset_xbc + d_ssm + 2 * ngroups * d_state;

    dzxbcdt.slice(2, offset_xbc, offset_xbc + conv_dim).copy_(d_xbc_in);
    dzxbcdt.slice(2, offset_dt, offset_dt + nheads).copy_(ddt_raw);

    auto dzxbcdt_out = dzxbcdt.to(z.scalar_type());
    auto d_conv_w_out = d_conv_w.to(w.scalar_type());
    auto d_conv_b_out = d_conv_b.to(w.scalar_type());
    auto ddt_bias_out = ddt_bias.to(dtb.scalar_type());
    auto dA_out = dA.to(alog.scalar_type());
    auto dstate0_out = dstate0.to(state0.scalar_type());
    auto dD_out = dD.to(d.scalar_type());

    return {dzxbcdt_out, d_conv_w_out, d_conv_b_out, ddt_bias_out, dA_out, dD_out, dstate0_out};
}
