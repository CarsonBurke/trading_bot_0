// V5 kernel with 8x8 register blocking (64 accumulators per thread)
// Trades occupancy for better ILP and register reuse
__global__ void chunk_scan_fwd_kernel_v5_8x8(
    const float* CB,
    const at::BFloat16* x,
    const float* dt,
    const float* dA_cumsum,
    const at::BFloat16* C,
    const float* state_in,
    const float* D,
    const int64_t* seq_idx,
    int64_t seq_stride,
    int64_t d_has_hdim,
    float* y,
    int64_t batch,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t seqlen,
    int64_t has_seq_idx) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int TM = 8;  // Elements per thread in M dimension
    constexpr int TN = 8;  // Elements per thread in N dimension

    int64_t tile_n = blockIdx.x;
    int64_t tile_m = blockIdx.y;
    int64_t pid = blockIdx.z;
    int64_t b = pid / (num_chunks * nheads);
    int64_t rem = pid - b * num_chunks * nheads;
    int64_t c = rem / nheads;
    int64_t h = rem - c * nheads;
    if (b >= batch) return;

    int64_t heads_per_group = nheads / ngroups;
    int64_t g = h / heads_per_group;

    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    // 8x8 thread block, each thread handles 8x8 output elements
    int64_t tid_m = threadIdx.y;  // 0-7
    int64_t tid_n = threadIdx.x;  // 0-7
    int64_t m0 = row_base + tid_m * TM;
    int64_t n0 = col_base + tid_n * TN;

    int64_t cb_offset = ((b * num_chunks + c) * ngroups + g) * chunk_size * chunk_size;
    int64_t c_offset = ((b * num_chunks + c) * ngroups + g) * chunk_size * d_state;
    int64_t x_offset = ((((b * num_chunks + c) * chunk_size) * nheads + h) * headdim);
    int64_t state_offset = ((((b * num_chunks + c) * nheads + h) * headdim) * d_state);
    int64_t dt_offset = (((b * nheads + h) * num_chunks + c) * chunk_size);
    int64_t seq_base = b * seq_stride + c * chunk_size;
    int64_t chunk_len = seqlen - c * chunk_size;
    if (chunk_len > chunk_size) chunk_len = chunk_size;
    int64_t seq_prev = -1;
    if (has_seq_idx && c > 0) seq_prev = seq_idx[seq_base - 1];

    __shared__ float cb_s[BM][BK];
    __shared__ float x_s[BK][BN];
    __shared__ float c_s[BM][BK];
    __shared__ float st_s[BK][BN];
    __shared__ float dA_m[BM];
    __shared__ float dA_k[BK];
    __shared__ float dt_k[BK];

    // 8x8 = 64 accumulators per thread
    float acc[TM][TN] = {{0}};
    int load_idx = threadIdx.y * 8 + threadIdx.x;  // 0-63

    // Load dA_m once
    if (load_idx < BM) {
        int64_t m = row_base + load_idx;
        dA_m[load_idx] = (m < chunk_size) ? dA_cumsum[dt_offset + m] : 0.0f;
    }
    __syncthreads();

    // Preload dA_m values for this thread's rows (8 values)
    float dA_m_vals[TM];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        dA_m_vals[i] = dA_m[tid_m * TM + i];
    }

    // Part 1: Intra-chunk scan
    for (int64_t k0 = 0; k0 < chunk_size; k0 += BK) {
        // Load CB tile [BM, BK] - 64 threads, 64*32/64 = 32 elements per thread
        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            int idx = load_idx + l * 64;
            int r = idx / BK;
            int k_in = idx % BK;
            if (r < BM) {
                int64_t m_global = row_base + r;
                int64_t k_global = k0 + k_in;
                cb_s[r][k_in] = (m_global < chunk_size && k_global <= m_global) ?
                    CB[cb_offset + m_global * chunk_size + k_global] : 0.0f;
            }
        }

        // Load X tile [BK, BN] - 32*64/64 = 32 elements per thread
        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            int idx = load_idx + l * 64;
            int row = idx / BN;
            int col = idx % BN;
            if (row < BK) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                x_s[row][col] = (k < chunk_size && n < headdim)
                    ? static_cast<float>(x[x_offset + k * headdim + n]) : 0.0f;
            }
        }

        // Load dA_k and dt_k
        if (load_idx < BK) {
            int64_t k = k0 + load_idx;
            dA_k[load_idx] = (k < chunk_size) ? dA_cumsum[dt_offset + k] : 0.0f;
            dt_k[load_idx] = (k < chunk_size) ? dt[dt_offset + k] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float dA_kv = dA_k[k];
            float dtv = dt_k[k];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                float scale = __expf(fminf(dA_m_vals[i] - dA_kv, 0.0f)) * dtv;
                float cbv = cb_s[tid_m * TM + i][k] * scale;
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += cbv * x_s[k][tid_n * TN + j];
                }
            }
        }
        __syncthreads();
    }

    // Precompute contrib_scales for inter-chunk state (8 values)
    float contrib_scales[TM];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = tid_m * TM + i;
        int64_t m = row_base + row;
        contrib_scales[i] = 0.0f;
        if (m < chunk_len && (!has_seq_idx || seq_idx[seq_base + m] == seq_prev)) {
            contrib_scales[i] = __expf(fminf(dA_m_vals[i], 0.0f));
        }
    }

    // Part 2: Inter-chunk state
    for (int64_t k0 = 0; k0 < d_state; k0 += BK) {
        // Load C tile
        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            int idx = load_idx + l * 64;
            int row = idx / BK;
            int col = idx % BK;
            if (row < BM) {
                int64_t m = row_base + row;
                int64_t k = k0 + col;
                c_s[row][col] = (m < chunk_size && k < d_state)
                    ? static_cast<float>(C[c_offset + m * d_state + k]) : 0.0f;
            }
        }

        // Load state_in tile
        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            int idx = load_idx + l * 64;
            int row = idx / BN;
            int col = idx % BN;
            if (row < BK) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                st_s[row][col] = (k < d_state && n < headdim)
                    ? state_in[state_offset + n * d_state + k] : 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                float cv = c_s[tid_m * TM + i][k] * contrib_scales[i];
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += cv * st_s[k][tid_n * TN + j];
                }
            }
        }
        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int64_t m = m0 + i;
        if (m >= chunk_size) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int64_t n = n0 + j;
            if (n < headdim) {
                int64_t out_idx = x_offset + m * headdim + n;
                float d_val = d_has_hdim ? D[h * headdim + n] : D[h];
                y[out_idx] = acc[i][j] + static_cast<float>(x[out_idx]) * d_val;
            }
        }
    }
}

// V4 kernel with software pipelining - only double buffer Part 1 (more iterations)
// Part 2 (state) has only 4 iterations for d_state=128 so doesn't benefit much
__global__ void chunk_scan_fwd_kernel_v4_async(
    const float* CB,
    const at::BFloat16* x,
    const float* dt,
    const float* dA_cumsum,
    const at::BFloat16* C,
    const float* state_in,
    const float* D,
    const int64_t* seq_idx,
    int64_t seq_stride,
    int64_t d_has_hdim,
    float* y,
    int64_t batch,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t seqlen,
    int64_t has_seq_idx) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int STAGES = 2;  // Double buffering for Part 1 only

    int64_t tile_n = blockIdx.x;
    int64_t tile_m = blockIdx.y;
    int64_t pid = blockIdx.z;
    int64_t b = pid / (num_chunks * nheads);
    int64_t rem = pid - b * num_chunks * nheads;
    int64_t c = rem / nheads;
    int64_t h = rem - c * nheads;
    if (b >= batch) return;

    int64_t heads_per_group = nheads / ngroups;
    int64_t g = h / heads_per_group;

    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    int64_t tid_m = threadIdx.y;
    int64_t tid_n = threadIdx.x;
    int64_t m0 = row_base + tid_m * 4;
    int64_t n0 = col_base + tid_n * 4;

    int64_t cb_offset = ((b * num_chunks + c) * ngroups + g) * chunk_size * chunk_size;
    int64_t c_offset = ((b * num_chunks + c) * ngroups + g) * chunk_size * d_state;
    int64_t x_offset = ((((b * num_chunks + c) * chunk_size) * nheads + h) * headdim);
    int64_t state_offset = ((((b * num_chunks + c) * nheads + h) * headdim) * d_state);
    int64_t dt_offset = (((b * nheads + h) * num_chunks + c) * chunk_size);
    int64_t seq_base = b * seq_stride + c * chunk_size;
    int64_t chunk_len = seqlen - c * chunk_size;
    if (chunk_len > chunk_size) chunk_len = chunk_size;
    int64_t seq_prev = -1;
    if (has_seq_idx && c > 0) seq_prev = seq_idx[seq_base - 1];

    // Double-buffered shared memory for Part 1 (8 iterations for chunk_size=256)
    // Single buffer for Part 2 (4 iterations for d_state=128) - reuses cb_s and x_s
    __shared__ float cb_s[STAGES][BM][BK];  // 16KB
    __shared__ float x_s[STAGES][BK][BN];   // 16KB
    __shared__ float dA_m[BM];              // 256B
    __shared__ float dA_k[STAGES][BK];      // 256B
    __shared__ float dt_k[STAGES][BK];      // 256B
    // Total Part 1: ~32.75KB

    float acc[4][4] = {{0}};
    int load_idx = threadIdx.y * 16 + threadIdx.x;

    // Load dA_m once (invariant across k-tiles)
    if (load_idx < BM) {
        int64_t m = row_base + load_idx;
        dA_m[load_idx] = (m < chunk_size) ? dA_cumsum[dt_offset + m] : 0.0f;
    }
    __syncthreads();

    // Preload dA_m values for this thread's rows
    float dA_m_vals[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        dA_m_vals[i] = dA_m[tid_m * 4 + i];
    }

    // Part 1: Intra-chunk scan with double buffering (8 iterations)
    int64_t num_k_tiles = (chunk_size + BK - 1) / BK;

    // Prologue: load first tile into stage 0
    #pragma unroll
    for (int l = 0; l < 8; ++l) {
        int idx = load_idx + l * 256;
        int r = idx / BK;
        int k_in = idx % BK;
        if (r < BM) {
            int64_t m_global = row_base + r;
            cb_s[0][r][k_in] = (m_global < chunk_size && k_in <= m_global) ?
                CB[cb_offset + m_global * chunk_size + k_in] : 0.0f;
        }
    }
    #pragma unroll
    for (int l = 0; l < 8; ++l) {
        int idx = load_idx + l * 256;
        int row = idx / BN;
        int col = idx % BN;
        if (row < BK) {
            int64_t n = col_base + col;
            x_s[0][row][col] = (row < chunk_size && n < headdim)
                ? static_cast<float>(x[x_offset + row * headdim + n]) : 0.0f;
        }
    }
    if (load_idx < BK) {
        dA_k[0][load_idx] = (load_idx < chunk_size) ? dA_cumsum[dt_offset + load_idx] : 0.0f;
        dt_k[0][load_idx] = (load_idx < chunk_size) ? dt[dt_offset + load_idx] : 0.0f;
    }
    __syncthreads();

    for (int64_t kt = 0; kt < num_k_tiles; ++kt) {
        int s_curr = kt & 1;
        int s_next = (kt + 1) & 1;
        int64_t k0_next = (kt + 1) * BK;

        // Start loading next tile while computing current
        if (kt + 1 < num_k_tiles) {
            #pragma unroll
            for (int l = 0; l < 8; ++l) {
                int idx = load_idx + l * 256;
                int r = idx / BK;
                int k_in = idx % BK;
                if (r < BM) {
                    int64_t m_global = row_base + r;
                    int64_t k_global = k0_next + k_in;
                    cb_s[s_next][r][k_in] = (m_global < chunk_size && k_global <= m_global) ?
                        CB[cb_offset + m_global * chunk_size + k_global] : 0.0f;
                }
            }
            #pragma unroll
            for (int l = 0; l < 8; ++l) {
                int idx = load_idx + l * 256;
                int row = idx / BN;
                int col = idx % BN;
                if (row < BK) {
                    int64_t k = k0_next + row;
                    int64_t n = col_base + col;
                    x_s[s_next][row][col] = (k < chunk_size && n < headdim)
                        ? static_cast<float>(x[x_offset + k * headdim + n]) : 0.0f;
                }
            }
            if (load_idx < BK) {
                int64_t k = k0_next + load_idx;
                dA_k[s_next][load_idx] = (k < chunk_size) ? dA_cumsum[dt_offset + k] : 0.0f;
                dt_k[s_next][load_idx] = (k < chunk_size) ? dt[dt_offset + k] : 0.0f;
            }
        }

        // Compute on current tile
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float dA_kv = dA_k[s_curr][k];
            float dtv = dt_k[s_curr][k];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float scale = __expf(fminf(dA_m_vals[i] - dA_kv, 0.0f)) * dtv;
                float cbv = cb_s[s_curr][tid_m * 4 + i][k] * scale;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += cbv * x_s[s_curr][k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    // Precompute contrib_scales for inter-chunk state
    float contrib_scales[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = tid_m * 4 + i;
        int64_t m = row_base + row;
        contrib_scales[i] = 0.0f;
        if (m < chunk_len && (!has_seq_idx || seq_idx[seq_base + m] == seq_prev)) {
            contrib_scales[i] = __expf(fminf(dA_m_vals[i], 0.0f));
        }
    }

    // Part 2: Inter-chunk state - reuse cb_s[0] for C, x_s[0] for state (single buffer)
    // Only 4 iterations for d_state=128, less benefit from double buffering
    for (int64_t k0 = 0; k0 < d_state; k0 += BK) {
        // Load C tile into cb_s[0]
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BK;
            int col = idx % BK;
            if (row < BM) {
                int64_t m = row_base + row;
                int64_t k = k0 + col;
                cb_s[0][row][col] = (m < chunk_size && k < d_state)
                    ? static_cast<float>(C[c_offset + m * d_state + k]) : 0.0f;
            }
        }
        // Load state_in tile into x_s[0]
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx % BN;
            if (row < BK) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                x_s[0][row][col] = (k < d_state && n < headdim)
                    ? state_in[state_offset + n * d_state + k] : 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float cv = cb_s[0][tid_m * 4 + i][k] * contrib_scales[i];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += cv * x_s[0][k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t m = m0 + i;
        if (m >= chunk_size) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t n = n0 + j;
            if (n < headdim) {
                int64_t out_idx = x_offset + m * headdim + n;
                float d_val = d_has_hdim ? D[h * headdim + n] : D[h];
                y[out_idx] = acc[i][j] + static_cast<float>(x[out_idx]) * d_val;
            }
        }
    }
}

// Optimized v3 kernel with larger BK and vectorized loads
__global__ void chunk_scan_fwd_kernel_v3(
    const float* CB,
    const float* x,
    const float* dt,
    const float* dA_cumsum,
    const float* C,
    const float* state_in,
    const float* D,
    const int64_t* seq_idx,
    int64_t seq_stride,
    int64_t d_has_hdim,
    float* y,
    int64_t batch,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t seqlen,
    int64_t has_seq_idx) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;  // Doubled for d_state=128

    int64_t tile_n = blockIdx.x;
    int64_t tile_m = blockIdx.y;
    int64_t pid = blockIdx.z;
    int64_t b = pid / (num_chunks * nheads);
    int64_t rem = pid - b * num_chunks * nheads;
    int64_t c = rem / nheads;
    int64_t h = rem - c * nheads;
    if (b >= batch) return;

    int64_t heads_per_group = nheads / ngroups;
    int64_t g = h / heads_per_group;

    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    int64_t tid_m = threadIdx.y;
    int64_t tid_n = threadIdx.x;
    int64_t m0 = row_base + tid_m * 4;
    int64_t n0 = col_base + tid_n * 4;

    int64_t cb_offset = ((b * num_chunks + c) * ngroups + g) * chunk_size * chunk_size;
    int64_t c_offset = ((b * num_chunks + c) * ngroups + g) * chunk_size * d_state;
    int64_t x_offset = ((((b * num_chunks + c) * chunk_size) * nheads + h) * headdim);
    int64_t state_offset = ((((b * num_chunks + c) * nheads + h) * headdim) * d_state);
    int64_t dt_offset = (((b * nheads + h) * num_chunks + c) * chunk_size);
    int64_t seq_base = b * seq_stride + c * chunk_size;
    int64_t chunk_len = seqlen - c * chunk_size;
    if (chunk_len > chunk_size) chunk_len = chunk_size;
    int64_t seq_prev = -1;
    if (has_seq_idx && c > 0) seq_prev = seq_idx[seq_base - 1];

    __shared__ float cb_s[BM][BK];
    __shared__ float x_s[BK][BN];
    __shared__ float c_s[BM][BK];
    __shared__ float st_s[BK][BN];
    __shared__ float dA_m[BM];
    __shared__ float dA_k[BK];
    __shared__ float dt_k[BK];

    float acc[4][4] = {{0}};

    // Load dA_m once
    int load_idx = threadIdx.y * 16 + threadIdx.x;
    if (load_idx < BM) {
        int64_t m = row_base + load_idx;
        dA_m[load_idx] = (m < chunk_size) ? dA_cumsum[dt_offset + m] : 0.0f;
    }
    if (load_idx + 64 < BM) {
        int64_t m = row_base + load_idx + 64;
        dA_m[load_idx + 64] = (m < chunk_size) ? dA_cumsum[dt_offset + m] : 0.0f;
    }
    __syncthreads();

    // Part 1: Intra-chunk scan
    for (int64_t k0 = 0; k0 < chunk_size; k0 += BK) {
        // Load CB tile [BM, BK] with 8 loads per thread (64*32 / 256 = 8)
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            int idx = load_idx + l * 256;
            int r = idx / BK;
            int k_in = idx % BK;
            if (r < BM) {
                int64_t m_global = row_base + r;
                int64_t k_global = k0 + k_in;
                cb_s[r][k_in] = (m_global < chunk_size && k_global <= m_global) ?
                    CB[cb_offset + m_global * chunk_size + k_global] : 0.0f;
            }
        }

        // Load X tile [BK, BN]
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx % BN;
            if (row < BK) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                x_s[row][col] = (k < chunk_size && n < headdim) ? x[x_offset + k * headdim + n] : 0.0f;
            }
        }

        // Load dA_k and dt_k
        if (load_idx < BK) {
            int64_t k = k0 + load_idx;
            dA_k[load_idx] = (k < chunk_size) ? dA_cumsum[dt_offset + k] : 0.0f;
            dt_k[load_idx] = (k < chunk_size) ? dt[dt_offset + k] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float dA_kv = dA_k[k];
            float dtv = dt_k[k];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int row = tid_m * 4 + i;
                float scale = expf(fminf(dA_m[row] - dA_kv, 0.0f)) * dtv;
                float cbv = cb_s[row][k] * scale;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += cbv * x_s[k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    // Part 2: Inter-chunk state (now only 4 iterations for d_state=128)
    for (int64_t k0 = 0; k0 < d_state; k0 += BK) {
        // Load C tile
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BK;
            int col = idx % BK;
            if (row < BM) {
                int64_t m = row_base + row;
                int64_t k = k0 + col;
                c_s[row][col] = (m < chunk_size && k < d_state) ? C[c_offset + m * d_state + k] : 0.0f;
            }
        }

        // Load state_in tile
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx % BN;
            if (row < BK) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                st_s[row][col] = (k < d_state && n < headdim) ? state_in[state_offset + n * d_state + k] : 0.0f;
            }
        }
        __syncthreads();

        float contrib_scales[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = tid_m * 4 + i;
            int64_t m = row_base + row;
            contrib_scales[i] = 0.0f;
            if (m < chunk_len && (!has_seq_idx || seq_idx[seq_base + m] == seq_prev)) {
                contrib_scales[i] = expf(fminf(dA_m[row], 0.0f));
            }
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float cv = c_s[tid_m * 4 + i][k] * contrib_scales[i];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += cv * st_s[k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t m = m0 + i;
        if (m >= chunk_size) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t n = n0 + j;
            if (n < headdim) {
                int64_t out_idx = x_offset + m * headdim + n;
                float d_val = d_has_hdim ? D[h * headdim + n] : D[h];
                y[out_idx] = acc[i][j] + x[out_idx] * d_val;
            }
        }
    }
}

// V3 kernel with BK=32 and bf16 inputs for x and C, with optional fused Z-gating
template <typename ZT, bool FUSE_GATE>
__global__ void chunk_scan_fwd_kernel_v3_bf16in_fused(
    const float* CB,
    const at::BFloat16* x,
    const float* dt,
    const float* dA_cumsum,
    const at::BFloat16* C,
    const float* state_in,
    const float* D,
    const ZT* z,                // New: z tensor for gating
    int64_t z_stride_b,         // New
    int64_t z_stride_l,         // New
    int64_t z_offset,           // New
    const int64_t* seq_idx,
    int64_t seq_stride,
    int64_t d_has_hdim,
    float* y,
    int64_t batch,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t seqlen,
    int64_t has_seq_idx) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;

    int64_t tile_n = blockIdx.x;
    int64_t tile_m = blockIdx.y;
    int64_t pid = blockIdx.z;
    int64_t b = pid / (num_chunks * nheads);
    int64_t rem = pid - b * num_chunks * nheads;
    int64_t c = rem / nheads;
    int64_t h = rem - c * nheads;
    if (b >= batch) return;

    int64_t heads_per_group = nheads / ngroups;
    int64_t g = h / heads_per_group;

    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    int64_t tid_m = threadIdx.y;
    int64_t tid_n = threadIdx.x;
    int64_t m0 = row_base + tid_m * 4;
    int64_t n0 = col_base + tid_n * 4;

    int64_t cb_offset = ((b * num_chunks + c) * ngroups + g) * chunk_size * chunk_size;
    int64_t c_offset = ((b * num_chunks + c) * ngroups + g) * chunk_size * d_state;
    int64_t x_offset = ((((b * num_chunks + c) * chunk_size) * nheads + h) * headdim);
    int64_t state_offset = ((((b * num_chunks + c) * nheads + h) * headdim) * d_state);
    int64_t dt_offset = (((b * nheads + h) * num_chunks + c) * chunk_size);
    int64_t seq_base = b * seq_stride + c * chunk_size;
    int64_t chunk_len = seqlen - c * chunk_size;
    if (chunk_len > chunk_size) chunk_len = chunk_size;
    int64_t seq_prev = -1;
    if (has_seq_idx && c > 0) seq_prev = seq_idx[seq_base - 1];

    __shared__ float cb_s[BM][BK];
    __shared__ float x_s[BK][BN];
    __shared__ float c_s[BM][BK];
    __shared__ float st_s[BK][BN];
    __shared__ float dA_m[BM];
    __shared__ float dA_k[BK];
    __shared__ float dt_k[BK];

    float acc[4][4] = {{0}};

    int load_idx = threadIdx.y * 16 + threadIdx.x;
    if (load_idx < BM) {
        int64_t m = row_base + load_idx;
        dA_m[load_idx] = (m < chunk_size) ? dA_cumsum[dt_offset + m] : 0.0f;
    }
    __syncthreads();

    // Preload dA_m values for this thread's rows
    float dA_m_vals[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        dA_m_vals[i] = dA_m[tid_m * 4 + i];
    }

    // Part 1: Intra-chunk scan with BK=32
    for (int64_t k0 = 0; k0 < chunk_size; k0 += BK) {
        // Load CB tile [BM, BK]
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            int idx = load_idx + l * 256;
            int r = idx / BK;
            int k_in = idx % BK;
            if (r < BM) {
                int64_t m_global = row_base + r;
                int64_t k_global = k0 + k_in;
                cb_s[r][k_in] = (m_global < chunk_size && k_global <= m_global) ?
                    CB[cb_offset + m_global * chunk_size + k_global] : 0.0f;
            }
        }

        // Load X tile [BK, BN] (bf16 -> fp32)
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx % BN;
            if (row < BK) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                x_s[row][col] = (k < chunk_size && n < headdim)
                    ? static_cast<float>(x[x_offset + k * headdim + n]) : 0.0f;
            }
        }

        // Load dA_k and dt_k
        if (load_idx < BK) {
            int64_t k = k0 + load_idx;
            dA_k[load_idx] = (k < chunk_size) ? dA_cumsum[dt_offset + k] : 0.0f;
            dt_k[load_idx] = (k < chunk_size) ? dt[dt_offset + k] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float dA_kv = dA_k[k];
            float dtv = dt_k[k];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float scale = __expf(fminf(dA_m_vals[i] - dA_kv, 0.0f)) * dtv;
                float cbv = cb_s[tid_m * 4 + i][k] * scale;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += cbv * x_s[k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    // Precompute contrib_scales for inter-chunk state
    float contrib_scales[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = tid_m * 4 + i;
        int64_t m = row_base + row;
        contrib_scales[i] = 0.0f;
        if (m < chunk_len && (!has_seq_idx || seq_idx[seq_base + m] == seq_prev)) {
            contrib_scales[i] = __expf(fminf(dA_m_vals[i], 0.0f));
        }
    }

    // Part 2: Inter-chunk state with BK=32
    for (int64_t k0 = 0; k0 < d_state; k0 += BK) {
        // Load C tile (bf16 -> fp32)
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BK;
            int col = idx % BK;
            if (row < BM) {
                int64_t m = row_base + row;
                int64_t k = k0 + col;
                c_s[row][col] = (m < chunk_size && k < d_state)
                    ? static_cast<float>(C[c_offset + m * d_state + k]) : 0.0f;
            }
        }

        // Load state_in tile
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx % BN;
            if (row < BK) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                st_s[row][col] = (k < d_state && n < headdim)
                    ? state_in[state_offset + n * d_state + k] : 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float cv = c_s[tid_m * 4 + i][k] * contrib_scales[i];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += cv * st_s[k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t m = m0 + i;
        if (m >= chunk_size) continue;
        
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t n = n0 + j;
            if (n < headdim) {
                int64_t out_idx = x_offset + m * headdim + n;
                float d_val = d_has_hdim ? D[h * headdim + n] : D[h];
                float val = acc[i][j] + static_cast<float>(x[out_idx]) * d_val;

                if (FUSE_GATE) {
                    int64_t t = c * chunk_size + m;
                    if (t < seqlen) {
                        int64_t z_idx = b * z_stride_b + t * z_stride_l + z_offset + h * headdim + n;
                        val = val * silu_f(to_float(z[z_idx]));
                    } else {
                        val = 0.0f;
                    }
                }

                y[out_idx] = val;
            }
        }
    }
}

// V2 kernel using precomputed CB matrix (BK=16)
__global__ void chunk_scan_fwd_kernel_v2(
    const float* CB,      // Precomputed [batch*chunks*groups, chunk_size, chunk_size]
    const float* x,
    const float* dt,
    const float* dA_cumsum,
    const float* C,
    const float* state_in,
    const float* D,
    const int64_t* seq_idx,
    int64_t seq_stride,
    int64_t d_has_hdim,
    float* y,
    int64_t batch,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t seqlen,
    int64_t has_seq_idx) {
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
    if (b >= batch) return;

    int64_t heads_per_group = nheads / ngroups;
    int64_t g = h / heads_per_group;

    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    int64_t tid_m = threadIdx.y;
    int64_t tid_n = threadIdx.x;
    int64_t m0 = row_base + tid_m * 4;
    int64_t n0 = col_base + tid_n * 4;

    // Offsets
    int64_t cb_offset = ((b * num_chunks + c) * ngroups + g) * chunk_size * chunk_size;
    int64_t c_offset = ((b * num_chunks + c) * ngroups + g) * chunk_size * d_state;
    int64_t x_offset = ((((b * num_chunks + c) * chunk_size) * nheads + h) * headdim);
    int64_t state_offset = ((((b * num_chunks + c) * nheads + h) * headdim) * d_state);
    int64_t dt_offset = (((b * nheads + h) * num_chunks + c) * chunk_size);
    int64_t seq_base = b * seq_stride + c * chunk_size;
    int64_t chunk_len = seqlen - c * chunk_size;
    if (chunk_len > chunk_size) chunk_len = chunk_size;
    int64_t seq_prev = -1;
    if (has_seq_idx && c > 0) seq_prev = seq_idx[seq_base - 1];

    __shared__ float cb_s[BM][BK];
    __shared__ float x_s[BK][BN];
    __shared__ float c_s[BM][BK];
    __shared__ float st_s[BK][BN];
    __shared__ float dA_m[BM];
    __shared__ float dA_k[BK];
    __shared__ float dt_k[BK];

    float acc[4][4] = {{0}};

    // Load dA_m once
    int load_idx_init = threadIdx.y * 16 + threadIdx.x;
    if (load_idx_init < BM) {
        int64_t m = row_base + load_idx_init;
        dA_m[load_idx_init] = (m < chunk_size) ? dA_cumsum[dt_offset + m] : 0.0f;
    }
    __syncthreads();

    // Part 1: Intra-chunk scan using precomputed CB
    for (int64_t k0 = 0; k0 < chunk_size; k0 += BK) {
        int load_idx = threadIdx.y * 16 + threadIdx.x;

        // Load CB tile [BM, BK]
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int r = idx / BK;
            int k_in = idx % BK;
            if (r < BM) {
                int64_t m_global = row_base + r;
                int64_t k_global = k0 + k_in;
                cb_s[r][k_in] = (m_global < chunk_size && k_global <= m_global) ?
                    CB[cb_offset + m_global * chunk_size + k_global] : 0.0f;
            }
        }

        // Load X tile
        for (int l = 0; l < 2; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx % BN;
            if (row < BK) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                x_s[row][col] = (k < chunk_size && n < headdim) ? x[x_offset + k * headdim + n] : 0.0f;
            }
        }

        // Load dA_k and dt_k
        if (load_idx < BK) {
            int64_t k = k0 + load_idx;
            dA_k[load_idx] = (k < chunk_size) ? dA_cumsum[dt_offset + k] : 0.0f;
            dt_k[load_idx] = (k < chunk_size) ? dt[dt_offset + k] : 0.0f;
        }
        __syncthreads();

        // Compute with predicated accumulation (no warp divergence)
        // Preload dA_m values for this thread's rows
        float dA_m_vals[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            dA_m_vals[i] = dA_m[tid_m * 4 + i];
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float dA_kv = dA_k[k];
            float dtv = dt_k[k];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                // Use __expf for faster single-precision exp
                float scale = __expf(fminf(dA_m_vals[i] - dA_kv, 0.0f)) * dtv;
                float cbv = cb_s[tid_m * 4 + i][k] * scale;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += cbv * x_s[k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    // Part 2: Inter-chunk state contribution
    for (int64_t k0 = 0; k0 < d_state; k0 += BK) {
        int load_idx = threadIdx.y * 16 + threadIdx.x;

        // Load C tile
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BK;
            int col = idx % BK;
            if (row < BM) {
                int64_t m = row_base + row;
                int64_t k = k0 + col;
                c_s[row][col] = (m < chunk_size && k < d_state) ? C[c_offset + m * d_state + k] : 0.0f;
            }
        }

        // Load state_in tile
        for (int l = 0; l < 2; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx % BN;
            if (row < BK) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                st_s[row][col] = (k < d_state && n < headdim) ? state_in[state_offset + n * d_state + k] : 0.0f;
            }
        }
        __syncthreads();

        // Precompute scales for state contribution (uses __expf for speed)
        float contrib_scales[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = tid_m * 4 + i;
            int64_t m = row_base + row;
            contrib_scales[i] = 0.0f;
            if (m < chunk_len && (!has_seq_idx || seq_idx[seq_base + m] == seq_prev)) {
                contrib_scales[i] = __expf(fminf(dA_m[row], 0.0f));
            }
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float cv = c_s[tid_m * 4 + i][k] * contrib_scales[i];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += cv * st_s[k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    // Write output with D skip connection
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t m = m0 + i;
        if (m >= chunk_size) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t n = n0 + j;
            if (n < headdim) {
                int64_t out_idx = x_offset + m * headdim + n;
                float d_val = d_has_hdim ? D[h * headdim + n] : D[h];
                y[out_idx] = acc[i][j] + x[out_idx] * d_val;
            }
        }
    }
}

// Fused kernel: computes CB on-the-fly, eliminating cb_buf
// This fuses bmm_cb_causal + chunk_scan into a single kernel
__global__ void chunk_scan_fwd_fused_kernel(
    const at::BFloat16* B,    // [batch*chunks*groups, chunk_size, d_state]
    const at::BFloat16* C,    // [batch*chunks*groups, chunk_size, d_state]
    const at::BFloat16* x,    // [batch, chunks, chunk_size, nheads, headdim] reshaped
    const float* dt,
    const float* dA_cumsum,
    const float* state_in,
    const float* D,
    const int64_t* seq_idx,
    int64_t seq_stride,
    int64_t d_has_hdim,
    float* y,
    int64_t batch,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t seqlen,
    int64_t has_seq_idx) {
    constexpr int BM = 64;    // Rows of output tile
    constexpr int BN = 64;    // Cols of output tile
    constexpr int BK = 16;    // Tile over chunk_size (k dimension of CB)
    constexpr int BS = 32;    // Tile over d_state for CB computation

    int64_t tile_n = blockIdx.x;
    int64_t tile_m = blockIdx.y;
    int64_t pid = blockIdx.z;
    int64_t b_idx = pid / (num_chunks * nheads);
    int64_t rem = pid - b_idx * num_chunks * nheads;
    int64_t c_idx = rem / nheads;
    int64_t h = rem - c_idx * nheads;
    if (b_idx >= batch) return;

    int64_t heads_per_group = nheads / ngroups;
    int64_t g = h / heads_per_group;

    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    int64_t tid_m = threadIdx.y;
    int64_t tid_n = threadIdx.x;
    int64_t m0 = row_base + tid_m * 4;
    int64_t n0 = col_base + tid_n * 4;

    int64_t bc_offset = ((b_idx * num_chunks + c_idx) * ngroups + g) * chunk_size * d_state;
    int64_t x_offset = ((((b_idx * num_chunks + c_idx) * chunk_size) * nheads + h) * headdim);
    int64_t state_offset = ((((b_idx * num_chunks + c_idx) * nheads + h) * headdim) * d_state);
    int64_t dt_offset = (((b_idx * nheads + h) * num_chunks + c_idx) * chunk_size);
    int64_t seq_base = b_idx * seq_stride + c_idx * chunk_size;
    int64_t chunk_len = seqlen - c_idx * chunk_size;
    if (chunk_len > chunk_size) chunk_len = chunk_size;
    int64_t seq_prev = -1;
    if (has_seq_idx && c_idx > 0) seq_prev = seq_idx[seq_base - 1];

    // Shared memory
    __shared__ float c_tile[BM][BS];       // C tile for CB computation
    __shared__ float b_tile[BK][BS];       // B tile for CB computation
    __shared__ float cb_tile[BM][BK];      // Computed CB tile
    __shared__ float x_s[BK][BN];          // X tile
    __shared__ float cs_tile[BM][BS];      // C tile for state contribution
    __shared__ float st_s[BS][BN];         // State tile
    __shared__ float dA_m[BM];
    __shared__ float dA_k[BK];
    __shared__ float dt_k[BK];

    float acc[4][4] = {{0}};
    int load_idx = threadIdx.y * 16 + threadIdx.x;

    // Load dA_m once
    if (load_idx < BM) {
        int64_t m = row_base + load_idx;
        dA_m[load_idx] = (m < chunk_size) ? dA_cumsum[dt_offset + m] : 0.0f;
    }
    __syncthreads();

    // Part 1: Intra-chunk scan with on-the-fly CB computation
    for (int64_t k0 = 0; k0 < chunk_size; k0 += BK) {
        // Zero CB tile
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int r = idx / BK;
            int c = idx % BK;
            if (r < BM) cb_tile[r][c] = 0.0f;
        }
        __syncthreads();

        // Compute CB tile: CB[m,k] = sum_n C[m,n] * B[k,n]
        for (int64_t s0 = 0; s0 < d_state; s0 += BS) {
            // Load C[BM, BS]
            for (int l = 0; l < 8; ++l) {
                int idx = load_idx + l * 256;
                int r = idx / BS;
                int c = idx % BS;
                if (r < BM && c < BS) {
                    int64_t m = row_base + r;
                    int64_t s = s0 + c;
                    c_tile[r][c] = (m < chunk_size && s < d_state)
                        ? static_cast<float>(C[bc_offset + m * d_state + s]) : 0.0f;
                }
            }
            // Load B[BK, BS]
            for (int l = 0; l < 2; ++l) {
                int idx = load_idx + l * 256;
                int r = idx / BS;
                int c = idx % BS;
                if (r < BK && c < BS) {
                    int64_t k = k0 + r;
                    int64_t s = s0 + c;
                    b_tile[r][c] = (k < chunk_size && s < d_state)
                        ? static_cast<float>(B[bc_offset + k * d_state + s]) : 0.0f;
                }
            }
            __syncthreads();

            // Accumulate CB = C @ B^T
            #pragma unroll
            for (int s = 0; s < BS; ++s) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float cv = c_tile[tid_m * 4 + i][s];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (tid_n * 4 + j < BK) {
                            cb_tile[tid_m * 4 + i][tid_n * 4 + j] += cv * b_tile[tid_n * 4 + j][s];
                        }
                    }
                }
            }
            __syncthreads();
        }

        // Apply causal mask to CB (k <= m) - done implicitly by not using values where k > m

        // Load X tile
        for (int l = 0; l < 2; ++l) {
            int idx = load_idx + l * 256;
            int r = idx / BN;
            int c = idx % BN;
            if (r < BK) {
                int64_t k = k0 + r;
                int64_t n = col_base + c;
                x_s[r][c] = (k < chunk_size && n < headdim)
                    ? static_cast<float>(x[x_offset + k * headdim + n]) : 0.0f;
            }
        }

        // Load dA_k and dt_k
        if (load_idx < BK) {
            int64_t k = k0 + load_idx;
            dA_k[load_idx] = (k < chunk_size) ? dA_cumsum[dt_offset + k] : 0.0f;
            dt_k[load_idx] = (k < chunk_size) ? dt[dt_offset + k] : 0.0f;
        }
        __syncthreads();

        // Compute scan with CB and scale
        float dA_m_vals[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            dA_m_vals[i] = dA_m[tid_m * 4 + i];
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            int64_t k_global = k0 + k;
            float dA_kv = dA_k[k];
            float dtv = dt_k[k];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int64_t m_global = row_base + tid_m * 4 + i;
                float scale = __expf(fminf(dA_m_vals[i] - dA_kv, 0.0f)) * dtv;
                // Apply causal mask: only use CB if k <= m
                float cbv = (k_global <= m_global && m_global < chunk_size) ? cb_tile[tid_m * 4 + i][k] * scale : 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += cbv * x_s[k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    // Part 2: Inter-chunk state contribution (same as before)
    for (int64_t s0 = 0; s0 < d_state; s0 += BS) {
        // Load C tile
        for (int l = 0; l < 8; ++l) {
            int idx = load_idx + l * 256;
            int r = idx / BS;
            int c = idx % BS;
            if (r < BM && c < BS) {
                int64_t m = row_base + r;
                int64_t s = s0 + c;
                cs_tile[r][c] = (m < chunk_size && s < d_state)
                    ? static_cast<float>(C[bc_offset + m * d_state + s]) : 0.0f;
            }
        }

        // Load state_in tile
        for (int l = 0; l < 8; ++l) {
            int idx = load_idx + l * 256;
            int r = idx / BN;
            int c = idx % BN;
            if (r < BS && c < BN) {
                int64_t s = s0 + r;
                int64_t n = col_base + c;
                st_s[r][c] = (s < d_state && n < headdim) ? state_in[state_offset + n * d_state + s] : 0.0f;
            }
        }
        __syncthreads();

        float contrib_scales[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = tid_m * 4 + i;
            int64_t m = row_base + row;
            contrib_scales[i] = 0.0f;
            if (m < chunk_len && (!has_seq_idx || seq_idx[seq_base + m] == seq_prev)) {
                contrib_scales[i] = __expf(fminf(dA_m[row], 0.0f));
            }
        }

        #pragma unroll
        for (int s = 0; s < BS; ++s) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float cv = cs_tile[tid_m * 4 + i][s] * contrib_scales[i];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += cv * st_s[s][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    // Write output with D skip connection
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t m = m0 + i;
        if (m >= chunk_size) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t n = n0 + j;
            if (n < headdim) {
                int64_t out_idx = x_offset + m * headdim + n;
                float d_val = d_has_hdim ? D[h * headdim + n] : D[h];
                y[out_idx] = acc[i][j] + static_cast<float>(x[out_idx]) * d_val;
            }
        }
    }
}

// bf16-input version: loads bf16 x and C, accumulates fp32
__global__ void chunk_scan_fwd_kernel_v2_bf16in(
    const float* CB,      // Precomputed [batch*chunks*groups, chunk_size, chunk_size] (fp32)
    const at::BFloat16* x,
    const float* dt,
    const float* dA_cumsum,
    const at::BFloat16* C,
    const float* state_in,
    const float* D,
    const int64_t* seq_idx,
    int64_t seq_stride,
    int64_t d_has_hdim,
    float* y,
    int64_t batch,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t seqlen,
    int64_t has_seq_idx) {
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
    if (b >= batch) return;

    int64_t heads_per_group = nheads / ngroups;
    int64_t g = h / heads_per_group;

    int64_t row_base = tile_m * BM;
    int64_t col_base = tile_n * BN;
    int64_t tid_m = threadIdx.y;
    int64_t tid_n = threadIdx.x;
    int64_t m0 = row_base + tid_m * 4;
    int64_t n0 = col_base + tid_n * 4;

    int64_t cb_offset = ((b * num_chunks + c) * ngroups + g) * chunk_size * chunk_size;
    int64_t c_offset = ((b * num_chunks + c) * ngroups + g) * chunk_size * d_state;
    int64_t x_offset = ((((b * num_chunks + c) * chunk_size) * nheads + h) * headdim);
    int64_t state_offset = ((((b * num_chunks + c) * nheads + h) * headdim) * d_state);
    int64_t dt_offset = (((b * nheads + h) * num_chunks + c) * chunk_size);
    int64_t seq_base = b * seq_stride + c * chunk_size;
    int64_t chunk_len = seqlen - c * chunk_size;
    if (chunk_len > chunk_size) chunk_len = chunk_size;
    int64_t seq_prev = -1;
    if (has_seq_idx && c > 0) seq_prev = seq_idx[seq_base - 1];

    __shared__ float cb_s[BM][BK];
    __shared__ float x_s[BK][BN];
    __shared__ float c_s[BM][BK];
    __shared__ float st_s[BK][BN];
    __shared__ float dA_m[BM];
    __shared__ float dA_k[BK];
    __shared__ float dt_k[BK];

    float acc[4][4] = {{0}};

    int load_idx_init = threadIdx.y * 16 + threadIdx.x;
    if (load_idx_init < BM) {
        int64_t m = row_base + load_idx_init;
        dA_m[load_idx_init] = (m < chunk_size) ? dA_cumsum[dt_offset + m] : 0.0f;
    }
    __syncthreads();

    // Part 1: Intra-chunk scan
    for (int64_t k0 = 0; k0 < chunk_size; k0 += BK) {
        int load_idx = threadIdx.y * 16 + threadIdx.x;

        // Load CB tile [BM, BK] (already fp32)
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int r = idx / BK;
            int k_in = idx % BK;
            if (r < BM) {
                int64_t m_global = row_base + r;
                int64_t k_global = k0 + k_in;
                cb_s[r][k_in] = (m_global < chunk_size && k_global <= m_global) ?
                    CB[cb_offset + m_global * chunk_size + k_global] : 0.0f;
            }
        }

        // Load X tile (bf16 -> fp32)
        for (int l = 0; l < 2; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx % BN;
            if (row < BK) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                x_s[row][col] = (k < chunk_size && n < headdim)
                    ? static_cast<float>(x[x_offset + k * headdim + n]) : 0.0f;
            }
        }

        // Load dA_k and dt_k
        if (load_idx < BK) {
            int64_t k = k0 + load_idx;
            dA_k[load_idx] = (k < chunk_size) ? dA_cumsum[dt_offset + k] : 0.0f;
            dt_k[load_idx] = (k < chunk_size) ? dt[dt_offset + k] : 0.0f;
        }
        __syncthreads();

        float dA_m_vals[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            dA_m_vals[i] = dA_m[tid_m * 4 + i];
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float dA_kv = dA_k[k];
            float dtv = dt_k[k];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float scale = __expf(fminf(dA_m_vals[i] - dA_kv, 0.0f)) * dtv;
                float cbv = cb_s[tid_m * 4 + i][k] * scale;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += cbv * x_s[k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    // Part 2: Inter-chunk state contribution
    for (int64_t k0 = 0; k0 < d_state; k0 += BK) {
        int load_idx = threadIdx.y * 16 + threadIdx.x;

        // Load C tile (bf16 -> fp32)
        for (int l = 0; l < 4; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BK;
            int col = idx % BK;
            if (row < BM) {
                int64_t m = row_base + row;
                int64_t k = k0 + col;
                c_s[row][col] = (m < chunk_size && k < d_state)
                    ? static_cast<float>(C[c_offset + m * d_state + k]) : 0.0f;
            }
        }

        // Load state_in tile (already fp32)
        for (int l = 0; l < 2; ++l) {
            int idx = load_idx + l * 256;
            int row = idx / BN;
            int col = idx % BN;
            if (row < BK) {
                int64_t k = k0 + row;
                int64_t n = col_base + col;
                st_s[row][col] = (k < d_state && n < headdim) ? state_in[state_offset + n * d_state + k] : 0.0f;
            }
        }
        __syncthreads();

        float contrib_scales[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = tid_m * 4 + i;
            int64_t m = row_base + row;
            contrib_scales[i] = 0.0f;
            if (m < chunk_len && (!has_seq_idx || seq_idx[seq_base + m] == seq_prev)) {
                contrib_scales[i] = __expf(fminf(dA_m[row], 0.0f));
            }
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float cv = c_s[tid_m * 4 + i][k] * contrib_scales[i];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += cv * st_s[k][tid_n * 4 + j];
                }
            }
        }
        __syncthreads();
    }

    // Write output with D skip connection (x is bf16)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t m = m0 + i;
        if (m >= chunk_size) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int64_t n = n0 + j;
            if (n < headdim) {
                int64_t out_idx = x_offset + m * headdim + n;
                float d_val = d_has_hdim ? D[h * headdim + n] : D[h];
                y[out_idx] = acc[i][j] + static_cast<float>(x[out_idx]) * d_val;
            }
        }
    }
}

// Legacy version kept for compatibility
__global__ void chunk_scan_fwd_kernel(
    const float* B,
    const float* x,
    const float* dt,
    const float* dA_cumsum,
    const float* C,
    const float* state_in,
    const float* D,
    const int64_t* seq_idx,
    int64_t seq_stride,
    int64_t d_has_hdim,
    float* y,
    int64_t batch,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t seqlen,
    int64_t has_seq_idx) {
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

        int64_t bc_offset = ((b * num_chunks + c) * ngroups + g) * chunk_size * d_state;
        int64_t x_offset = ((((b * num_chunks + c) * chunk_size) * nheads + h) * headdim);
        int64_t c_offset = bc_offset;
        int64_t b_offset = bc_offset;
        int64_t state_offset = ((((b * num_chunks + c) * nheads + h) * headdim) * d_state);
        int64_t dt_offset = (((b * nheads + h) * num_chunks + c) * chunk_size);
        int64_t seq_base = b * seq_stride + c * chunk_size;
        int64_t chunk_len = seqlen - c * chunk_size;
        if (chunk_len > chunk_size) {
            chunk_len = chunk_size;
        }
        int64_t seq_prev = -1;
        if (has_seq_idx && c > 0) {
            seq_prev = seq_idx[seq_base - 1];
        }

        // Load dA_m once for use in both loops and final scaling
        int load_idx_init = threadIdx.y * 16 + threadIdx.x;
        if (load_idx_init < BM) {
            int64_t m = row_base + load_idx_init;
            float val = 0.0f;
            if (m < chunk_size) {
                val = dA_cumsum[dt_offset + m];
            }
            dA_m[load_idx_init] = val;
        }
        __syncthreads();

        for (int64_t k0 = 0; k0 < chunk_size; k0 += BK) {
            int load_idx = threadIdx.y * 16 + threadIdx.x;

            // 1. Compute cb_s[BM][BK] on the fly from C and B
            for (int l = 0; l < 4; ++l) {
                int idx = load_idx + l * 256;
                int r = idx / BK;
                int k_in = idx % BK;
                if (r < BM && k_in < BK) {
                    float sum = 0.0f;
                    int64_t m_global = row_base + r;
                    int64_t k_global = k0 + k_in;
                    if (m_global < chunk_len && k_global < chunk_len &&
                        (!has_seq_idx || seq_idx[seq_base + m_global] == seq_idx[seq_base + k_global])) {
                        for (int64_t n = 0; n < d_state; ++n) {
                            sum += C[c_offset + m_global * d_state + n] *
                                   B[b_offset + k_global * d_state + n];
                        }
                    }
                    cb_s[r][k_in] = sum;
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
                    int64_t m = row_base + row;
                    if ((k0 + k) > m) continue;
                    float dA_m_v = dA_m[row];
                    float scale = expf(fminf(dA_m_v - dA_kv, 0.0f)) * dtv;
                    float cbv = cb_s[row][k] * scale;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[i][j] += cbv * x_s[k][tid_n * 4 + j];
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

                        float contrib_scales[4];
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            int row = tid_m * 4 + i;
                            int64_t m = row_base + row;
                            contrib_scales[i] = 0.0f;
                            if (row < BM) {
                                if (!(has_seq_idx && (m >= chunk_len || seq_idx[seq_base + m] != seq_prev))) {
                                    contrib_scales[i] = expf(fminf(dA_m[row], 0.0f));
                                }
                            }
                        }
                #pragma unroll
                for (int k = 0; k < BK; ++k) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float cv = c_s[tid_m * 4 + i][k] * contrib_scales[i];
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
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int64_t n = n0 + j;
                if (n < headdim) {
                    int64_t out_idx = x_offset + m * headdim + n;
                    float d_val = d_has_hdim ? D[h * headdim + n] : D[h];
                    float x_val = x[out_idx];
                    float acc_val = acc[i][j];
                    y[out_idx] = acc_val + x_val * d_val;
                }
            }
        }}

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
