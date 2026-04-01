#include "ssd_common.cuh"

// Apply causal mask (zero upper triangle) to CB matrix - simplified version without seq_idx
__global__ void apply_causal_mask_simple_kernel(
    float* CB,
    int64_t batches,
    int64_t chunk_size,
    int64_t seqlen,
    int64_t num_chunks) {
    int64_t bg = blockIdx.z;
    int64_t tile_m = blockIdx.y;
    int64_t tile_n = blockIdx.x;
    if (bg >= batches) return;

    int64_t c = bg % num_chunks;
    int64_t chunk_len = seqlen - c * chunk_size;
    if (chunk_len > chunk_size) chunk_len = chunk_size;

    constexpr int BM = 64;
    constexpr int BN = 64;
    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    int tid = threadIdx.y * 16 + threadIdx.x;

    float* CB_ptr = CB + bg * chunk_size * chunk_size;

    // Each thread handles multiple elements
    for (int l = 0; l < 16; ++l) {
        int idx = tid + l * 256;
        int m = row_base + idx / BN;
        int k = col_base + idx % BN;
        if (m < chunk_size && k < chunk_size) {
            float val = CB_ptr[m * chunk_size + k];
            // Apply causal mask: k > m means future position
            if (k > m) val = 0.0f;
            // Bounds check for partial chunks
            if (m >= chunk_len || k >= chunk_len) val = 0.0f;
            CB_ptr[m * chunk_size + k] = val;
        }
    }
}

__global__ void bmm_kt_kn_scale_x_kernel(
    const float* A_t,
    const float* X,
    const float* dt,
    const float* dA_cumsum,
    const int64_t* seq_idx,
    int64_t seq_stride,
    float* C,
    int64_t batches,
    int64_t M,
    int64_t K,
    int64_t N,
    int64_t seqlen,
    int64_t num_chunks,
    int64_t ngroups,
    int64_t nheads,
    int64_t headdim,
    int64_t chunk_size,
    int64_t has_seq_idx) {
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
    int64_t seq_base = b * seq_stride + c * chunk_size;
    int64_t chunk_len = seqlen - c * chunk_size;
    if (chunk_len > chunk_size) {
        chunk_len = chunk_size;
    }

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
                if (k < K && n < N && k < chunk_len) {
                    int64_t head_in_group = n / headdim;
                    int64_t head = g * heads_per_group + head_in_group;
                    int64_t chunk_base_idx = ((b * nheads + head) * num_chunks + c) * chunk_size;
                    int64_t dt_idx = chunk_base_idx + k;
                    float dA_L = dA_cumsum[chunk_base_idx + chunk_len - 1];
                    float scale = dt[dt_idx] * exp2f(fminf(dA_L - dA_cumsum[dt_idx], 0.0f) * kLog2e);
                    if (has_seq_idx) {
                        int64_t seq_k = seq_idx[seq_base + k];
                        int64_t seq_last = seq_idx[seq_base + (chunk_len - 1)];
                        if (seq_k != seq_last) {
                            scale = 0.0f;
                        }
                    }
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

// bf16-input version: loads bf16 A_t and X, accumulates fp32
__global__ void bmm_kt_kn_scale_x_kernel_bf16in(
    const at::BFloat16* A_t,
    const at::BFloat16* X,
    const float* dt,
    const float* dA_cumsum,
    const int64_t* seq_idx,
    int64_t seq_stride,
    float* C,
    int64_t batches,
    int64_t M,
    int64_t K,
    int64_t N,
    int64_t seqlen,
    int64_t num_chunks,
    int64_t ngroups,
    int64_t nheads,
    int64_t headdim,
    int64_t chunk_size,
    int64_t has_seq_idx) {
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
    int64_t seq_base = b * seq_stride + c * chunk_size;
    int64_t chunk_len = seqlen - c * chunk_size;
    if (chunk_len > chunk_size) {
        chunk_len = chunk_size;
    }

    const at::BFloat16* A_ptr = A_t + bg * K * M;
    const at::BFloat16* X_ptr = X + bg * K * N;
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
                    val = static_cast<float>(A_ptr[k * M + m]);
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
                if (k < K && n < N && k < chunk_len) {
                    int64_t head_in_group = n / headdim;
                    int64_t head = g * heads_per_group + head_in_group;
                    int64_t chunk_base_idx = ((b * nheads + head) * num_chunks + c) * chunk_size;
                    int64_t dt_idx = chunk_base_idx + k;
                    float dA_L = dA_cumsum[chunk_base_idx + chunk_len - 1];
                    float scale = dt[dt_idx] * exp2f(fminf(dA_L - dA_cumsum[dt_idx], 0.0f) * kLog2e);
                    if (has_seq_idx) {
                        int64_t seq_k = seq_idx[seq_base + k];
                        int64_t seq_last = seq_idx[seq_base + (chunk_len - 1)];
                        if (seq_k != seq_last) {
                            scale = 0.0f;
                        }
                    }
                    val = static_cast<float>(X_ptr[k * N + n]) * scale;
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

// bf16-input version: loads bf16 B,C, accumulates fp32, outputs fp32 CB
__global__ void bmm_cb_causal_kernel_bf16in(
    const at::BFloat16* C_mat,
    const at::BFloat16* B_mat,
    const float* dA_cumsum,
    const int64_t* seq_idx,
    int64_t seq_stride,
    float* CB,
    int64_t batches,
    int64_t chunk_size,
    int64_t d_state,
    int64_t seqlen,
    int64_t num_chunks,
    int64_t ngroups,
    int64_t nheads,
    int64_t has_seq_idx) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;

    int64_t bg = blockIdx.z;
    int64_t tile_m = blockIdx.y;
    int64_t tile_n = blockIdx.x;
    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    int64_t tid_m = threadIdx.y;
    int64_t tid_n = threadIdx.x;
    int64_t m0 = row_base + tid_m * 4;
    int64_t k0_out = col_base + tid_n * 4;

    if (bg >= batches) return;

    int64_t b = bg / (num_chunks * ngroups);
    int64_t rem = bg - b * num_chunks * ngroups;
    int64_t c = rem / ngroups;
    int64_t g = rem - c * ngroups;
    int64_t seq_base = b * seq_stride + c * chunk_size;
    int64_t chunk_len = seqlen - c * chunk_size;
    if (chunk_len > chunk_size) chunk_len = chunk_size;

    const at::BFloat16* C_ptr = C_mat + bg * chunk_size * d_state;
    const at::BFloat16* B_ptr = B_mat + bg * chunk_size * d_state;
    float* CB_ptr = CB + bg * chunk_size * chunk_size;

    __shared__ float C_s[BM][BK];
    __shared__ float B_s[BN][BK];

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int64_t kk = 0; kk < d_state; kk += BK) {
        int load_idx = threadIdx.y * 16 + threadIdx.x;
        // Load C[BM, BK] - bf16 -> fp32
        for (int l = 0; l < 8; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BK;
            int col = idx % BK;
            if (row < BM && col < BK) {
                int64_t m = row_base + row;
                int64_t k = kk + col;
                float val = (m < chunk_size && k < d_state)
                    ? static_cast<float>(C_ptr[m * d_state + k]) : 0.0f;
                C_s[row][col] = val;
            }
        }
        // Load B[BN, BK] - bf16 -> fp32
        for (int l = 0; l < 8; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BK;
            int col = idx % BK;
            if (row < BN && col < BK) {
                int64_t k_out = col_base + row;
                int64_t n = kk + col;
                float val = (k_out < chunk_size && n < d_state)
                    ? static_cast<float>(B_ptr[k_out * d_state + n]) : 0.0f;
                B_s[row][col] = val;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float c_val = C_s[tid_m * 4 + i][k];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += c_val * B_s[tid_n * 4 + j][k];
                }
            }
        }
        __syncthreads();
    }

    // Write with causal mask (k <= m) and seq_idx check
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t m = m0 + i;
        if (m >= chunk_size) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t k = k0_out + j;
            if (k >= chunk_size) continue;
            float val = acc[i][j];
            // Causal mask: k <= m
            if (k > m) val = 0.0f;
            // Bounds check
            if (m >= chunk_len || k >= chunk_len) val = 0.0f;
            // Seq idx check
            if (has_seq_idx && val != 0.0f) {
                if (seq_idx[seq_base + m] != seq_idx[seq_base + k]) {
                    val = 0.0f;
                }
            }
            CB_ptr[m * chunk_size + k] = val;
        }
    }
}
