#include "ssd_common.cuh"

// Version that outputs bf16 for x,b,c (matching Triton reference behavior)
template <typename scalar_t>
__global__ void conv1d_pack_dt_kernel_bf16out(
    const scalar_t* zxbcdt,
    const scalar_t* conv_w,
    const scalar_t* conv_b,
    const scalar_t* dt_bias,
    const scalar_t* dt_scale,
    const int64_t* seq_idx,
    int64_t seq_stride,
    at::BFloat16* x_buf,  // bf16 output
    at::BFloat16* b_buf,  // bf16 output
    at::BFloat16* c_buf,  // bf16 output
    float* dt_out,        // fp32 for cumsum precision
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
    int64_t nheads,
    bool has_conv_bias,
    int64_t has_seq_idx,
    float dt_min,
    float dt_max,
    int64_t dt_scale_stride_b,
    int64_t dt_scale_stride_t,
    int64_t dt_scale_batch,
    int64_t dt_scale_seqlen,
    bool has_dt_scale) {
    int64_t padded_len = num_chunks * chunk_size;
    int64_t total_channels = conv_dim + nheads;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * padded_len * total_channels;
    if (idx >= total) return;

    int64_t tmp = idx;
    int64_t c = tmp % total_channels;
    tmp /= total_channels;
    int64_t t = tmp % padded_len;
    int64_t b = tmp / padded_len;
    int64_t chunk = t / chunk_size;
    int64_t t_in_chunk = t - chunk * chunk_size;

    if (c >= conv_dim) {
        int64_t h = c - conv_dim;
        float dt_val = 0.0f;
        if (t < seqlen) {
            int64_t offset_dt = d_in_proj - nheads;
            float dt_raw = to_float(zxbcdt[(b * seqlen + t) * d_in_proj + offset_dt + h]);
            float dt_pre = dt_raw + to_float(dt_bias[h]);
            dt_val = softplus_f(dt_pre);
            if (has_dt_scale) {
                int64_t b_idx = dt_scale_batch == 1 ? 0 : b;
                int64_t t_idx = dt_scale_seqlen == 1 ? 0 : t;
                dt_val *= to_float(dt_scale[b_idx * dt_scale_stride_b + t_idx * dt_scale_stride_t]);
            }
            dt_val = fminf(fmaxf(dt_val, dt_min), dt_max);
        }
        dt_out[((b * nheads + h) * num_chunks + chunk) * chunk_size + t_in_chunk] = dt_val;
        return;
    }

    float conv_val = 0.0f;
    if (t < seqlen) {
        int64_t offset_xbc = 2 * d_mlp + d_ssm;
        int64_t seq_base = has_seq_idx ? (b * seq_stride) : 0;
        int64_t seq_cur = has_seq_idx ? seq_idx[seq_base + t] : 0;
        float conv_pre = 0.0f;
        for (int64_t w = 0; w < conv_kernel; ++w) {
            int64_t t_in = t - w;
            if (t_in < 0 || (has_seq_idx && seq_idx[seq_base + t_in] != seq_cur)) continue;
            conv_pre += to_float(conv_w[c * conv_kernel + w]) *
                        to_float(zxbcdt[(b * seqlen + t_in) * d_in_proj + offset_xbc + c]);
        }
        if (has_conv_bias) conv_pre += to_float(conv_b[c]);
        conv_val = silu_f(conv_pre);
    }

    // Output as bf16
    at::BFloat16 conv_bf16 = static_cast<at::BFloat16>(conv_val);
    if (c < d_ssm) {
        int64_t h = c / headdim;
        int64_t p = c - h * headdim;
        x_buf[(((b * num_chunks + chunk) * chunk_size + t_in_chunk) * nheads + h) * headdim + p] = conv_bf16;
    } else if (c < d_ssm + ngroups * d_state) {
        int64_t c_off = c - d_ssm;
        int64_t g = c_off / d_state;
        int64_t n = c_off - g * d_state;
        b_buf[(((b * num_chunks + chunk) * chunk_size + t_in_chunk) * ngroups + g) * d_state + n] = conv_bf16;
    } else {
        int64_t c_off = c - d_ssm - ngroups * d_state;
        int64_t g = c_off / d_state;
        int64_t n = c_off - g * d_state;
        c_buf[(((b * num_chunks + chunk) * chunk_size + t_in_chunk) * ngroups + g) * d_state + n] = conv_bf16;
    }
}

// Original version with fp32 output (kept for backward compatibility)
template <typename scalar_t>
__global__ void conv1d_pack_dt_kernel(
    const scalar_t* zxbcdt,
    const scalar_t* conv_w,
    const scalar_t* conv_b,
    const scalar_t* dt_bias,
    const scalar_t* dt_scale,
    const int64_t* seq_idx,
    int64_t seq_stride,
    float* x_buf,
    float* b_buf,
    float* c_buf,
    float* dt_out,
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
    int64_t nheads,
    bool has_conv_bias,
    int64_t has_seq_idx,
    float dt_min,
    float dt_max,
    int64_t dt_scale_stride_b,
    int64_t dt_scale_stride_t,
    int64_t dt_scale_batch,
    int64_t dt_scale_seqlen,
    bool has_dt_scale) {
    int64_t padded_len = num_chunks * chunk_size;
    int64_t total_channels = conv_dim + nheads;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * padded_len * total_channels;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t c = tmp % total_channels;
    tmp /= total_channels;
    int64_t t = tmp % padded_len;
    int64_t b = tmp / padded_len;

    int64_t chunk = t / chunk_size;
    int64_t t_in_chunk = t - chunk * chunk_size;

    if (c >= conv_dim) {
        int64_t h = c - conv_dim;
        float dt_val = 0.0f;
        if (t < seqlen) {
            int64_t offset_dt = d_in_proj - nheads;
            int64_t t_global = t;
            float dt_raw =
                to_float(zxbcdt[(b * seqlen + t_global) * d_in_proj + offset_dt + h]);
            float dt_pre = dt_raw + to_float(dt_bias[h]);
            dt_val = softplus_f(dt_pre);
            if (has_dt_scale) {
                int64_t b_idx = dt_scale_batch == 1 ? 0 : b;
                int64_t t_idx = dt_scale_seqlen == 1 ? 0 : t_global;
                dt_val *= to_float(dt_scale[b_idx * dt_scale_stride_b + t_idx * dt_scale_stride_t]);
            }
            if (dt_val < dt_min || dt_val > dt_max) {
                dt_val = fminf(fmaxf(dt_val, dt_min), dt_max);
            }
        }
        int64_t dt_idx = ((b * nheads + h) * num_chunks + chunk) * chunk_size + t_in_chunk;
        dt_out[dt_idx] = dt_val;
        return;
    }

    float conv_val = 0.0f;
    if (t < seqlen) {
        int64_t offset_xbc = 2 * d_mlp + d_ssm;
        int64_t seq_base = has_seq_idx ? (b * seq_stride) : 0;
        int64_t seq_cur = has_seq_idx ? seq_idx[seq_base + t] : 0;
        float conv_pre = 0.0f;
        for (int64_t w = 0; w < conv_kernel; ++w) {
            int64_t t_in = t - w;
            if (t_in < 0) {
                continue;
            }
            if (has_seq_idx && seq_idx[seq_base + t_in] != seq_cur) {
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
        conv_val = silu_f(conv_pre);
    }

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
__global__ void conv1d_pack_stateful_kernel(
    const scalar_t* zxbcdt,
    const scalar_t* conv_w,
    const scalar_t* conv_b,
    const scalar_t* conv_state,
    const int64_t* seq_idx,
    int64_t seq_stride,
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
    bool has_conv_bias,
    int64_t has_seq_idx,
    int64_t has_conv_state) {
    int64_t padded_len = num_chunks * chunk_size;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * padded_len * conv_dim;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t c = tmp % conv_dim;
    tmp /= conv_dim;
    int64_t t = tmp % padded_len;
    int64_t b = tmp / padded_len;

    int64_t chunk = t / chunk_size;
    int64_t t_in_chunk = t - chunk * chunk_size;

    float conv_val = 0.0f;
    if (t < seqlen) {
        int64_t offset_xbc = 2 * d_mlp + d_ssm;
        int64_t seq_base = has_seq_idx ? (b * seq_stride) : 0;
        int64_t seq_cur = has_seq_idx ? seq_idx[seq_base + t] : 0;
        float conv_pre = 0.0f;
        for (int64_t w = 0; w < conv_kernel; ++w) {
            int64_t t_in = t - w;
            if (t_in >= 0) {
                if (has_seq_idx && seq_idx[seq_base + t_in] != seq_cur) {
                    continue;
                }
                int64_t xbc_idx = (b * seqlen + t_in) * d_in_proj + offset_xbc + c;
                float xbc_val = to_float(zxbcdt[xbc_idx]);
                float w_val = to_float(conv_w[c * conv_kernel + w]);
                conv_pre += w_val * xbc_val;
            } else if (has_conv_state) {
                int64_t state_idx = conv_kernel - 1 + t_in;
                if (state_idx >= 0) {
                    int64_t state_offset = (b * conv_dim + c) * conv_kernel + state_idx;
                    float xbc_val = to_float(conv_state[state_offset]);
                    float w_val = to_float(conv_w[c * conv_kernel + w]);
                    conv_pre += w_val * xbc_val;
                }
            }
        }
        if (has_conv_bias) {
            conv_pre += to_float(conv_b[c]);
        }
        conv_val = silu_f(conv_pre);
    }

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

// BF16 output version for tensor core acceleration
template <typename scalar_t>
__global__ void conv1d_pack_stateful_kernel_bf16out(
    const scalar_t* zxbcdt,
    const scalar_t* conv_w,
    const scalar_t* conv_b,
    const scalar_t* conv_state,
    const int64_t* seq_idx,
    int64_t seq_stride,
    at::BFloat16* x_buf,
    at::BFloat16* b_buf,
    at::BFloat16* c_buf,
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
    bool has_conv_bias,
    int64_t has_seq_idx,
    int64_t has_conv_state) {
    int64_t padded_len = num_chunks * chunk_size;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * padded_len * conv_dim;
    if (idx >= total) return;

    int64_t tmp = idx;
    int64_t c = tmp % conv_dim;
    tmp /= conv_dim;
    int64_t t = tmp % padded_len;
    int64_t b = tmp / padded_len;
    int64_t chunk = t / chunk_size;
    int64_t t_in_chunk = t - chunk * chunk_size;

    float conv_val = 0.0f;
    if (t < seqlen) {
        int64_t offset_xbc = 2 * d_mlp + d_ssm;
        int64_t seq_base = has_seq_idx ? (b * seq_stride) : 0;
        int64_t seq_cur = has_seq_idx ? seq_idx[seq_base + t] : 0;
        float conv_pre = 0.0f;
        for (int64_t w = 0; w < conv_kernel; ++w) {
            int64_t t_in = t - w;
            if (t_in >= 0) {
                if (has_seq_idx && seq_idx[seq_base + t_in] != seq_cur) continue;
                int64_t xbc_idx = (b * seqlen + t_in) * d_in_proj + offset_xbc + c;
                conv_pre += to_float(conv_w[c * conv_kernel + w]) * to_float(zxbcdt[xbc_idx]);
            } else if (has_conv_state) {
                int64_t state_idx = conv_kernel - 1 + t_in;
                if (state_idx >= 0) {
                    int64_t state_offset = (b * conv_dim + c) * conv_kernel + state_idx;
                    conv_pre += to_float(conv_w[c * conv_kernel + w]) * to_float(conv_state[state_offset]);
                }
            }
        }
        if (has_conv_bias) conv_pre += to_float(conv_b[c]);
        conv_val = silu_f(conv_pre);
    }

    int64_t nheads = d_ssm / headdim;
    at::BFloat16 conv_bf16 = static_cast<at::BFloat16>(conv_val);

    if (c < d_ssm) {
        int64_t h = c / headdim;
        int64_t p = c - h * headdim;
        x_buf[(((b * num_chunks + chunk) * chunk_size + t_in_chunk) * nheads + h) * headdim + p] = conv_bf16;
    } else if (c < d_ssm + ngroups * d_state) {
        int64_t c_off = c - d_ssm;
        int64_t g = c_off / d_state;
        int64_t n = c_off - g * d_state;
        b_buf[(((b * num_chunks + chunk) * chunk_size + t_in_chunk) * ngroups + g) * d_state + n] = conv_bf16;
    } else {
        int64_t c_off = c - d_ssm - ngroups * d_state;
        int64_t g = c_off / d_state;
        int64_t n = c_off - g * d_state;
        c_buf[(((b * num_chunks + chunk) * chunk_size + t_in_chunk) * ngroups + g) * d_state + n] = conv_bf16;
    }
}

template <typename scalar_t>
__global__ void update_conv_state_kernel(
    const scalar_t* zxbcdt,
    const scalar_t* conv_state_in,
    scalar_t* conv_state_out,
    int64_t batch,
    int64_t seqlen,
    int64_t d_in_proj,
    int64_t conv_dim,
    int64_t conv_kernel,
    int64_t d_mlp,
    int64_t d_ssm) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * conv_dim * conv_kernel;
    if (idx >= total) {
        return;
    }
    int64_t k = idx % conv_kernel;
    int64_t tmp = idx / conv_kernel;
    int64_t c = tmp % conv_dim;
    int64_t b = tmp / conv_dim;
    int64_t offset_xbc = 2 * d_mlp + d_ssm;

    if (seqlen <= 0) {
        conv_state_out[idx] = conv_state_in[idx];
        return;
    }

    if (seqlen >= conv_kernel) {
        int64_t t_in = seqlen - conv_kernel + k;
        int64_t xbc_idx = (b * seqlen + t_in) * d_in_proj + offset_xbc + c;
        conv_state_out[idx] = from_float<scalar_t>(to_float(zxbcdt[xbc_idx]));
        return;
    }

    int64_t prev_count = conv_kernel - seqlen;
    if (k < prev_count) {
        int64_t prev_start = seqlen - 1;
        int64_t prev_idx = (b * conv_dim + c) * conv_kernel + (k + prev_start);
        conv_state_out[idx] = conv_state_in[prev_idx];
        return;
    }

    int64_t t_in = k - prev_count;
    int64_t xbc_idx = (b * seqlen + t_in) * d_in_proj + offset_xbc + c;
    conv_state_out[idx] = from_float<scalar_t>(to_float(zxbcdt[xbc_idx]));
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
    float scale = dt[dt_offset] * exp2f(fminf(-dA, 0.0f) * kLog2e);
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
    float scale = exp2f(fminf(dA, 0.0f) * kLog2e);
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
    int64_t dt_scale_stride_b,
    int64_t dt_scale_stride_t,
    int64_t dt_scale_batch,
    int64_t dt_scale_seqlen,
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
            int64_t b_idx = dt_scale_batch == 1 ? 0 : b;
            int64_t t_idx = dt_scale_seqlen == 1 ? 0 : t_global;
            dt *= to_float(dt_scale[b_idx * dt_scale_stride_b + t_idx * dt_scale_stride_t]);
        }
        if (dt < dt_min || dt > dt_max) {
            dt = fminf(fmaxf(dt, dt_min), dt_max);
        }
        int64_t dt_idx = ((b * nheads + h) * ((seqlen + chunk_size - 1) / chunk_size) + c) * chunk_size + t;
        dt_out[dt_idx] = dt;
        float a_log_val = to_float(a_log[h]);
        float a = -exp2f(a_log_val * kLog2e);
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
    } else if (t < chunk_size) {
        int64_t da_idx = ((b * nheads + h) * ((seqlen + chunk_size - 1) / chunk_size) + c) * chunk_size + t;
        dt_out[da_idx] = 0.0f;
        dA_cumsum[da_idx] = 0.0f;
    }
    if (t == 0 && chunk_len > 0) {
        float last = shmem[chunk_len - 1];
        exp_a_last[(b * nheads + h) * ((seqlen + chunk_size - 1) / chunk_size) + c] = exp2f(fminf(last, 0.0f) * kLog2e);
    }
}

__global__ void dt_cumsum_from_dt_kernel(
    const float* dt_in,
    const float* a_log,
    float* dA_cumsum,
    float* exp_a_last,
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

    int64_t num_chunks = (seqlen + chunk_size - 1) / chunk_size;
    int64_t dt_base = ((b * nheads + h) * num_chunks + c) * chunk_size;

    extern __shared__ float shmem[];
    float val = 0.0f;
    if (t < chunk_len) {
        float dt = dt_in[dt_base + t];
        float a_log_val = a_log[h];
        float a = -exp2f(a_log_val * kLog2e);
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
        dA_cumsum[dt_base + t] = shmem[t];
    } else if (t < chunk_size) {
        dA_cumsum[dt_base + t] = 0.0f;
    }
    if (t == 0 && chunk_len > 0) {
        float last = shmem[chunk_len - 1];
        exp_a_last[(b * nheads + h) * num_chunks + c] = exp2f(fminf(last, 0.0f) * kLog2e);
    }
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
        float a_log_val = to_float(a_log[h]);
        float a = -exp2f(a_log_val * kLog2e);
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
    int64_t dt_scale_stride_b,
    int64_t dt_scale_stride_t,
    int64_t dt_scale_batch,
    int64_t dt_scale_seqlen,
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
        int64_t b_idx = dt_scale_batch == 1 ? 0 : b;
        int64_t t_idx = dt_scale_seqlen == 1 ? 0 : t;
        scale = to_float(dt_scale[b_idx * dt_scale_stride_b + t_idx * dt_scale_stride_t]);
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
