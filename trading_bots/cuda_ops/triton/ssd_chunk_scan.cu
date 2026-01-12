__global__ void chunk_scan_fwd_kernel(
    const float* B, // Changed from cb
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
