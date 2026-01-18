#include "ssd_common.cuh"

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
    int64_t has_seq_idx) {
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
        // Clamp a_log to prevent exp2f(a_log) from being inf
        if (a_log_val > 10.0f) a_log_val = 10.0f;
        float a = -exp2f(a_log_val * kLog2e);
        // Clamp a to prevent 0 * inf = NaN or excessive decay
        if (a > -1e-6f) a = -1e-6f; 
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
    
    if (shmem[t] < -100.0f) shmem[t] = -100.0f;

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
        if (a_log_val > 10.0f) a_log_val = 10.0f;
        float a = -exp2f(a_log_val * kLog2e);
        if (a > -1e-6f) a = -1e-6f;
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

    if (shmem[t] < -100.0f) shmem[t] = -100.0f;

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
            float exp_a = exp2f(fminf(a_cumsum, 0.0f) * kLog2e);
            float exp_neg = exp2f(fminf(-a_cumsum, 0.0f) * kLog2e);
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

        float exp_a_last = exp2f(fminf(dA_cumsum[((b * nheads + h) * num_chunks + c) * chunk_size + (end - start - 1)], 0.0f) * kLog2e);
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
    float exp_a_t = exp2f(fminf(dA_cumsum[chunk_offset + t], 0.0f) * kLog2e);

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
        float exp_neg_s = exp2f(fminf(-dA_cumsum[chunk_offset + s], 0.0f) * kLog2e);
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
    float exp_a_last = exp2f(fminf(dA_cumsum[chunk_offset + (chunk_len - 1)], 0.0f) * kLog2e);

    int64_t state_base = ((b * nheads + h) * headdim + p) * d_state;
    for (int64_t n = 0; n < d_state; ++n) {
        float acc = 0.0f;
        int64_t b_base = d_ssm + group * d_state + n;
        for (int64_t s = 0; s < chunk_len; ++s) {
            int64_t s_global = chunk_idx * chunk_size + s;
            float x_val = conv_out[(b * seqlen + s_global) * conv_dim + x_chan];
            float b_val = conv_out[(b * seqlen + s_global) * conv_dim + b_base];
            float dt_val = dt[chunk_offset + s];
            float exp_neg_s = exp2f(fminf(-dA_cumsum[chunk_offset + s], 0.0f) * kLog2e);
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
        float exp_a_t = exp2f(fminf(dA_cumsum[chunk_offset + t], 0.0f) * kLog2e);
        float exp_neg_s = exp2f(fminf(-dA_cumsum[chunk_offset + s], 0.0f) * kLog2e);
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
        float exp_a_t = exp2f(fminf(dA_cumsum[chunk_offset + t], 0.0f) * kLog2e);
        float dy = grad_y[((b * seqlen + t_global) * nheads + h) * headdim + p];
        float c_val = conv_out[(b * seqlen + t_global) * conv_dim + c_base + n];
        acc += dy * exp_a_t * c_val;
    }

    float exp_a_last = exp2f(fminf(dA_cumsum[chunk_offset + (chunk_len - 1)], 0.0f) * kLog2e);
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
        float exp_a_t = exp2f(fminf(dA_cumsum[chunk_offset + t], 0.0f) * kLog2e);
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
        float exp_a_last = exp2f(fminf(dA_cumsum[chunk_offset + (chunk_len - 1)], 0.0f) * kLog2e);
        float exp_neg_s = exp2f(fminf(-dA_cumsum[chunk_offset + s], 0.0f) * kLog2e);
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
    float exp_neg_s = exp2f(fminf(-dA_cumsum[chunk_offset + s], 0.0f) * kLog2e);
    float dt_s = dt[chunk_offset + s];

    float sum = 0.0f;
    for (int64_t t = s; t < chunk_len; ++t) {
        int64_t t_global = chunk_idx * chunk_size + t;
        float exp_a_t = exp2f(fminf(dA_cumsum[chunk_offset + t], 0.0f) * kLog2e);
        float dy = grad_y[((b * seqlen + t_global) * nheads + h) * headdim + p];
        float cb_val = cb[((b * ngroups + g) * chunk_size + t) * chunk_size + s];
        sum += dy * exp_a_t * cb_val;
    }
    float extra = 0.0f;
    int64_t b_base = nheads * headdim + g * d_state;
    float exp_a_last = exp2f(fminf(dA_cumsum[chunk_offset + (chunk_len - 1)], 0.0f) * kLog2e);
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
        float a = -exp2f(to_float(a_log[h]) * kLog2e);
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
__global__ void fused_conv_scan_backward_kernel(
    const scalar_t* zxbcdt,
    const scalar_t* conv_w,
    const scalar_t* conv_b,
    const scalar_t* dt_bias,
    const scalar_t* a_log,
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
    bool has_conv_bias) {
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

    float a = -exp2f(to_float(a_log[h]) * kLog2e);

    int64_t offset_xbc = 2 * d_mlp + d_ssm;
    int64_t offset_dt = offset_xbc + d_ssm + 2 * ngroups * d_state;

    for (int64_t t = seqlen - 1; t >= 0; --t) {
        float dt_raw = to_float(zxbcdt[(b * seqlen + t) * d_in_proj + offset_dt + h]);
        float dt_pre = dt_raw + to_float(dt_bias[h]);
        float dt = softplus_f(dt_pre);
        bool clamp = false;
        if (dt < dt_min || dt > dt_max) {
            dt = fminf(fmaxf(dt, dt_min), dt_max);
            clamp = true;
        }
        float da = exp2f(dt * a * kLog2e);

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
