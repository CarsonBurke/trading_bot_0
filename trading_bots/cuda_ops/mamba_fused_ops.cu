#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Type.h>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
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
__global__ void fused_conv_scan_forward_kernel(
    const scalar_t* zxbcdt,
    const scalar_t* conv_w,
    const scalar_t* conv_b,
    const scalar_t* dt_bias,
    const scalar_t* a_log,
    const scalar_t* dt_scale,
    const scalar_t* initial_state,
    scalar_t* y,
    scalar_t* final_state,
    int64_t batch,
    int64_t seqlen,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t d_in_proj,
    int64_t d_ssm,
    int64_t d_mlp,
    int64_t conv_dim,
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
    for (int64_t n = 0; n < d_state; ++n) {
        int64_t s_idx = ((b * nheads + h) * headdim + p) * d_state + n;
        state[n] = to_float(initial_state[s_idx]);
    }

    float a = -expf(to_float(a_log[h]));

    int64_t offset_xbc = 2 * d_mlp + d_ssm;
    int64_t offset_dt = offset_xbc + d_ssm + 2 * ngroups * d_state;

    for (int64_t t = 0; t < seqlen; ++t) {
        float dt_raw = to_float(zxbcdt[(b * seqlen + t) * d_in_proj + offset_dt + h]);
        float dt = softplus_f(dt_raw + to_float(dt_bias[h]));
        if (has_dt_scale) {
            dt *= to_float(dt_scale[(b * seqlen + t) * nheads + h]);
        }
        if (dt < dt_min || dt > dt_max) {
            dt = fminf(fmaxf(dt, dt_min), dt_max);
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

        float y_val = 0.0f;
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

            float dbx = dt * b_conv * x_conv;
            state[n] = state[n] * da + dbx;
            y_val += state[n] * c_conv;
        }

        int64_t y_idx = ((b * seqlen + t) * nheads + h) * headdim + p;
        y[y_idx] = from_float<scalar_t>(y_val);
    }

    for (int64_t n = 0; n < d_state; ++n) {
        int64_t s_idx = ((b * nheads + h) * headdim + p) * d_state + n;
        final_state[s_idx] = from_float<scalar_t>(state[n]);
    }
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

} // namespace

std::vector<torch::Tensor> mamba_fused_forward_cuda(
    const torch::Tensor& zxbcdt,
    const torch::Tensor& conv_w,
    const torch::Tensor& conv_b,
    const torch::Tensor& dt_bias,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_scale,
    const torch::Tensor& initial_state,
    int64_t chunk_size,
    int64_t ngroups,
    int64_t headdim,
    double dt_min,
    double dt_max) {
    auto z = zxbcdt.contiguous();
    auto w = conv_w.contiguous();
    auto b = conv_b.defined() ? conv_b.contiguous() : torch::Tensor();
    bool has_conv_bias = b.defined() && b.numel() > 0;
    auto dtb = dt_bias.contiguous();
    auto alog = a_log.contiguous();
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

    TORCH_CHECK(nheads % ngroups == 0, "nheads must be divisible by ngroups");
    TORCH_CHECK(d_state <= kMaxDState, "d_state exceeds max supported");

    auto y = torch::zeros({batch, seqlen, nheads, headdim}, z.options());
    auto final_state = torch::zeros_like(state);

    const int threads = 256;
    int64_t total = batch * nheads * headdim;
    int blocks = (total + threads - 1) / threads;

    switch (z.scalar_type()) {
        case at::kFloat:
            fused_conv_scan_forward_kernel<float><<<blocks, threads>>>(
                z.data_ptr<float>(),
                w.data_ptr<float>(),
                has_conv_bias ? b.data_ptr<float>() : nullptr,
                dtb.data_ptr<float>(),
                alog.data_ptr<float>(),
                has_dt_scale ? dt_s.data_ptr<float>() : nullptr,
                state.data_ptr<float>(),
                y.data_ptr<float>(),
                final_state.data_ptr<float>(),
                batch,
                seqlen,
                nheads,
                headdim,
                d_state,
                ngroups,
                d_in_proj,
                d_ssm,
                d_mlp,
                d_ssm + 2 * ngroups * d_state,
                conv_kernel,
                static_cast<float>(dt_min),
                static_cast<float>(dt_max),
                has_conv_bias,
                has_dt_scale);
            break;
        case at::kHalf:
            fused_conv_scan_forward_kernel<at::Half><<<blocks, threads>>>(
                z.data_ptr<at::Half>(),
                w.data_ptr<at::Half>(),
                has_conv_bias ? b.data_ptr<at::Half>() : nullptr,
                dtb.data_ptr<at::Half>(),
                alog.data_ptr<at::Half>(),
                has_dt_scale ? dt_s.data_ptr<at::Half>() : nullptr,
                state.data_ptr<at::Half>(),
                y.data_ptr<at::Half>(),
                final_state.data_ptr<at::Half>(),
                batch,
                seqlen,
                nheads,
                headdim,
                d_state,
                ngroups,
                d_in_proj,
                d_ssm,
                d_mlp,
                d_ssm + 2 * ngroups * d_state,
                conv_kernel,
                static_cast<float>(dt_min),
                static_cast<float>(dt_max),
                has_conv_bias,
                has_dt_scale);
            break;
        case at::kBFloat16:
            fused_conv_scan_forward_kernel<at::BFloat16><<<blocks, threads>>>(
                z.data_ptr<at::BFloat16>(),
                w.data_ptr<at::BFloat16>(),
                has_conv_bias ? b.data_ptr<at::BFloat16>() : nullptr,
                dtb.data_ptr<at::BFloat16>(),
                alog.data_ptr<at::BFloat16>(),
                has_dt_scale ? dt_s.data_ptr<at::BFloat16>() : nullptr,
                state.data_ptr<at::BFloat16>(),
                y.data_ptr<at::BFloat16>(),
                final_state.data_ptr<at::BFloat16>(),
                batch,
                seqlen,
                nheads,
                headdim,
                d_state,
                ngroups,
                d_in_proj,
                d_ssm,
                d_mlp,
                d_ssm + 2 * ngroups * d_state,
                conv_kernel,
                static_cast<float>(dt_min),
                static_cast<float>(dt_max),
                has_conv_bias,
                has_dt_scale);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype for fused_conv_scan_forward");
    }

    return {y, final_state};
}

std::vector<torch::Tensor> mamba_fused_backward_cuda(
    const torch::Tensor& zxbcdt,
    const torch::Tensor& conv_w,
    const torch::Tensor& conv_b,
    const torch::Tensor& dt_bias,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_scale,
    const torch::Tensor& initial_state,
    const torch::Tensor& final_state,
    const torch::Tensor& grad_y,
    const torch::Tensor& grad_final_state,
    int64_t chunk_size,
    int64_t ngroups,
    int64_t headdim,
    double dt_min,
    double dt_max) {
    auto z = zxbcdt.contiguous();
    auto w = conv_w.contiguous();
    auto b = conv_b.defined() ? conv_b.contiguous() : torch::Tensor();
    bool has_conv_bias = b.defined() && b.numel() > 0;
    auto dtb = dt_bias.contiguous();
    auto alog = a_log.contiguous();
    auto state0 = initial_state.contiguous();
    auto stateN = final_state.contiguous();
    bool has_dt_scale = dt_scale.defined() && dt_scale.numel() > 0;
    auto dt_s = has_dt_scale ? dt_scale.expand({z.size(0), z.size(1), a_log.size(0)}).contiguous() : torch::Tensor();
    auto gy = grad_y.contiguous();
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

    auto dx_conv = torch::zeros({batch, seqlen, nheads, headdim}, z.options().dtype(torch::kFloat));
    auto dB = torch::zeros({batch, seqlen, ngroups, d_state}, z.options().dtype(torch::kFloat));
    auto dC = torch::zeros({batch, seqlen, ngroups, d_state}, z.options().dtype(torch::kFloat));
    auto ddt_raw = torch::zeros({batch, seqlen, nheads}, z.options().dtype(torch::kFloat));
    auto dA = torch::zeros({nheads}, z.options().dtype(torch::kFloat));
    auto ddt_bias = torch::zeros({nheads}, z.options().dtype(torch::kFloat));
    auto dstate0 = torch::zeros_like(state0, state0.options().dtype(torch::kFloat));

    const int threads = 256;
    int64_t total = batch * nheads * headdim;
    int blocks = (total + threads - 1) / threads;

    switch (z.scalar_type()) {
        case at::kFloat:
            fused_conv_scan_backward_kernel<float><<<blocks, threads>>>(
                z.data_ptr<float>(),
                w.data_ptr<float>(),
                has_conv_bias ? b.data_ptr<float>() : nullptr,
                dtb.data_ptr<float>(),
                alog.data_ptr<float>(),
                has_dt_scale ? dt_s.data_ptr<float>() : nullptr,
                state0.data_ptr<float>(),
                stateN.data_ptr<float>(),
                gy.data_ptr<float>(),
                gstate.defined() ? gstate.data_ptr<float>() : nullptr,
                dx_conv.data_ptr<float>(),
                dB.data_ptr<float>(),
                dC.data_ptr<float>(),
                ddt_raw.data_ptr<float>(),
                dA.data_ptr<float>(),
                ddt_bias.data_ptr<float>(),
                dstate0.data_ptr<float>(),
                batch,
                seqlen,
                nheads,
                headdim,
                d_state,
                ngroups,
                d_in_proj,
                d_ssm,
                d_mlp,
                conv_kernel,
                static_cast<float>(dt_min),
                static_cast<float>(dt_max),
                has_conv_bias,
                has_dt_scale);
            break;
        case at::kHalf:
            fused_conv_scan_backward_kernel<at::Half><<<blocks, threads>>>(
                z.data_ptr<at::Half>(),
                w.data_ptr<at::Half>(),
                has_conv_bias ? b.data_ptr<at::Half>() : nullptr,
                dtb.data_ptr<at::Half>(),
                alog.data_ptr<at::Half>(),
                has_dt_scale ? dt_s.data_ptr<at::Half>() : nullptr,
                state0.data_ptr<at::Half>(),
                stateN.data_ptr<at::Half>(),
                gy.data_ptr<at::Half>(),
                gstate.defined() ? gstate.data_ptr<at::Half>() : nullptr,
                dx_conv.data_ptr<float>(),
                dB.data_ptr<float>(),
                dC.data_ptr<float>(),
                ddt_raw.data_ptr<float>(),
                dA.data_ptr<float>(),
                ddt_bias.data_ptr<float>(),
                dstate0.data_ptr<float>(),
                batch,
                seqlen,
                nheads,
                headdim,
                d_state,
                ngroups,
                d_in_proj,
                d_ssm,
                d_mlp,
                conv_kernel,
                static_cast<float>(dt_min),
                static_cast<float>(dt_max),
                has_conv_bias,
                has_dt_scale);
            break;
        case at::kBFloat16:
            fused_conv_scan_backward_kernel<at::BFloat16><<<blocks, threads>>>(
                z.data_ptr<at::BFloat16>(),
                w.data_ptr<at::BFloat16>(),
                has_conv_bias ? b.data_ptr<at::BFloat16>() : nullptr,
                dtb.data_ptr<at::BFloat16>(),
                alog.data_ptr<at::BFloat16>(),
                has_dt_scale ? dt_s.data_ptr<at::BFloat16>() : nullptr,
                state0.data_ptr<at::BFloat16>(),
                stateN.data_ptr<at::BFloat16>(),
                gy.data_ptr<at::BFloat16>(),
                gstate.defined() ? gstate.data_ptr<at::BFloat16>() : nullptr,
                dx_conv.data_ptr<float>(),
                dB.data_ptr<float>(),
                dC.data_ptr<float>(),
                ddt_raw.data_ptr<float>(),
                dA.data_ptr<float>(),
                ddt_bias.data_ptr<float>(),
                dstate0.data_ptr<float>(),
                batch,
                seqlen,
                nheads,
                headdim,
                d_state,
                ngroups,
                d_in_proj,
                d_ssm,
                d_mlp,
                conv_kernel,
                static_cast<float>(dt_min),
                static_cast<float>(dt_max),
                has_conv_bias,
                has_dt_scale);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype for fused_conv_scan_backward");
    }

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

    auto d_xbc_in = torch::zeros({batch, seqlen, conv_dim}, z.options().dtype(torch::kFloat));
    auto d_conv_w = torch::zeros_like(w, w.options().dtype(torch::kFloat));
    auto d_conv_b = torch::zeros({conv_dim}, z.options().dtype(torch::kFloat));

    int64_t conv_total = batch * conv_dim;
    int conv_blocks = (conv_total + threads - 1) / threads;

    switch (z.scalar_type()) {
        case at::kFloat:
            conv1d_backward_kernel<float><<<conv_blocks, threads>>>(
                z.data_ptr<float>(),
                w.data_ptr<float>(),
                has_conv_bias ? b.data_ptr<float>() : nullptr,
                d_xbc_conv.data_ptr<float>(),
                d_xbc_in.data_ptr<float>(),
                d_conv_w.data_ptr<float>(),
                d_conv_b.data_ptr<float>(),
                batch,
                seqlen,
                d_in_proj,
                d_ssm,
                d_mlp,
                d_state,
                ngroups,
                conv_kernel,
                has_conv_bias);
            break;
        case at::kHalf:
            conv1d_backward_kernel<at::Half><<<conv_blocks, threads>>>(
                z.data_ptr<at::Half>(),
                w.data_ptr<at::Half>(),
                has_conv_bias ? b.data_ptr<at::Half>() : nullptr,
                d_xbc_conv.data_ptr<float>(),
                d_xbc_in.data_ptr<float>(),
                d_conv_w.data_ptr<float>(),
                d_conv_b.data_ptr<float>(),
                batch,
                seqlen,
                d_in_proj,
                d_ssm,
                d_mlp,
                d_state,
                ngroups,
                conv_kernel,
                has_conv_bias);
            break;
        case at::kBFloat16:
            conv1d_backward_kernel<at::BFloat16><<<conv_blocks, threads>>>(
                z.data_ptr<at::BFloat16>(),
                w.data_ptr<at::BFloat16>(),
                has_conv_bias ? b.data_ptr<at::BFloat16>() : nullptr,
                d_xbc_conv.data_ptr<float>(),
                d_xbc_in.data_ptr<float>(),
                d_conv_w.data_ptr<float>(),
                d_conv_b.data_ptr<float>(),
                batch,
                seqlen,
                d_in_proj,
                d_ssm,
                d_mlp,
                d_state,
                ngroups,
                conv_kernel,
                has_conv_bias);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype for conv1d_backward");
    }

    auto dzxbcdt = torch::zeros_like(z, z.options().dtype(torch::kFloat));
    int64_t offset_xbc = 2 * d_mlp + d_ssm;
    int64_t offset_dt = offset_xbc + d_ssm + 2 * ngroups * d_state;

    dzxbcdt.slice(2, offset_xbc, offset_xbc + conv_dim).copy_(d_xbc_in);
    dzxbcdt.slice(2, offset_dt, offset_dt + nheads).copy_(ddt_raw);

    auto dzxbcdt_out = dzxbcdt.to(z.scalar_type());
    auto d_conv_w_out = d_conv_w.to(w.scalar_type());
    auto d_conv_b_out = d_conv_b.to(w.scalar_type());
    auto ddt_bias_out = ddt_bias.to(dtb.scalar_type());
    auto dA_out = dA.to(alog.scalar_type());
    auto dstate0_out = dstate0.to(state0.scalar_type());

    return {dzxbcdt_out, d_conv_w_out, d_conv_b_out, ddt_bias_out, dA_out, dstate0_out};
}
