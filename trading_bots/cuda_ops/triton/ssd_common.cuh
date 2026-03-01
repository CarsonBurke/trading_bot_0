
#pragma once
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <iostream>

constexpr int kMaxDState = 256;
constexpr float kLog2e = 1.4426950408889634f;

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

