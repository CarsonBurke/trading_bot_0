#include "ssd_common.cuh"
#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace {
cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
    }
    return handle;
}
}

namespace {
struct MambaWorkspace {
  torch::Tensor dt_buf;
  torch::Tensor dA_buf;
  torch::Tensor exp_a_last;
  torch::Tensor x_buf;
  torch::Tensor b_buf;
  torch::Tensor c_buf;
  torch::Tensor cb_buf;  // Precomputed C @ B^T for optimized scan
  torch::Tensor y_padded_f;
  torch::Tensor chunk_state_mat;
  torch::Tensor gy_chunk;
  torch::Tensor dchunk_state_y;
  torch::Tensor dC_total;
  torch::Tensor dchunk_state;
  torch::Tensor ddA;
  torch::Tensor dstate0;
  torch::Tensor dx_scaled_state;
  torch::Tensor ddt_chunk_g;
  torch::Tensor ddA_xscaled_g;
  torch::Tensor dtemp_dummy;
  torch::Tensor dB_state;
  torch::Tensor dx_scan_total;
};

thread_local std::unordered_map<int, MambaWorkspace> WORKSPACE;

torch::Tensor workspace_tensor(torch::Tensor &slot,
                               at::IntArrayRef sizes,
                               const torch::TensorOptions &opts) {
  auto dtype = opts.dtype();
  auto device = opts.device();
  if (!slot.defined() || slot.dtype() != dtype || slot.device() != device) {
    slot = torch::empty(sizes, opts);
  } else {
    slot.resize_(sizes);
  }
  return slot;
}

MambaWorkspace &workspace_for_device(const torch::Device &device) {
  int index = device.index();
  return WORKSPACE[index];
}
} // namespace

// Kernels are defined in other files included via mamba_fused_ops.cu.
// We assume they are visible in this translation unit.

std::vector<torch::Tensor> mamba_fused_forward_cuda(
    const torch::Tensor &zxbcdt, const torch::Tensor &conv_w,
    const torch::Tensor &conv_b, const torch::Tensor &dt_bias,
    const torch::Tensor &a_log, const torch::Tensor &d_param,
    const torch::Tensor &dt_scale, const torch::Tensor &initial_state,
    const torch::Tensor &seq_idx, int64_t chunk_size, int64_t ngroups,
    int64_t headdim, double dt_min, double dt_max, bool fuse_gate = false) {
  at::NoGradGuard no_grad;
  auto z = zxbcdt.contiguous();
  auto w = conv_w.to(z.scalar_type()).contiguous();
  auto b = conv_b.defined() ? conv_b.to(z.scalar_type()).contiguous()
                            : torch::Tensor();
  bool has_conv_bias = b.defined() && b.numel() > 0;
  auto dtb = dt_bias.to(z.scalar_type()).contiguous();
  auto alog = a_log.to(torch::kFloat).contiguous();
  auto d = d_param.to(torch::kFloat).contiguous();
  bool d_has_hdim = d.dim() == 2;
  auto state = initial_state.to(z.scalar_type()).contiguous();
  bool has_seq_idx = seq_idx.defined() && seq_idx.numel() > 0;
  auto seq = has_seq_idx ? seq_idx.contiguous() : torch::Tensor();
  int64_t seq_stride = has_seq_idx ? seq.stride(0) : 0;
  bool has_dt_scale = dt_scale.defined() && dt_scale.numel() > 0;
  auto dt_scale_view = dt_scale;
  int64_t dt_scale_stride_b = 0, dt_scale_stride_t = 0, dt_scale_batch = 0,
          dt_scale_seqlen = 0;
  int64_t batch = z.size(0), seqlen = z.size(1), d_in_proj = z.size(2),
          nheads = a_log.size(0), d_ssm = nheads * headdim;
  int64_t d_state = (w.size(0) - d_ssm) / (2 * ngroups);
  int64_t d_inner = (d_in_proj - 2 * ngroups * d_state - nheads) / 2,
          d_mlp = d_inner - d_ssm, conv_dim = d_ssm + 2 * ngroups * d_state,
          conv_kernel = w.size(1);
  TORCH_CHECK(nheads % ngroups == 0, "nheads must be divisible by ngroups");
  TORCH_CHECK(w.dim() == 2, "conv_w must be 2D");
  TORCH_CHECK(d_state <= kMaxDState, "d_state exceeds max supported");
  if (has_dt_scale) {
    if (dt_scale_view.dim() == 3)
      dt_scale_view = dt_scale_view.squeeze(-1);
    dt_scale_stride_b = dt_scale_view.stride(0);
    dt_scale_stride_t = dt_scale_view.stride(1);
    dt_scale_batch = dt_scale_view.size(0);
    dt_scale_seqlen = dt_scale_view.size(1);
  }
  int64_t num_chunks = (seqlen + chunk_size - 1) / chunk_size;
  auto y = torch::empty({batch, seqlen, nheads, headdim}, z.options());
  auto float_opts = z.options().dtype(torch::kFloat);
  auto &ws = workspace_for_device(z.device());
  auto y_padded_f = workspace_tensor(
      ws.y_padded_f, {batch, num_chunks, chunk_size, nheads, headdim},
      float_opts);
  auto final_state = torch::empty_like(state);
  auto dt_buf = workspace_tensor(
      ws.dt_buf, {batch, nheads, num_chunks, chunk_size}, float_opts);
  auto dA_buf = workspace_tensor(
      ws.dA_buf, {batch, nheads, num_chunks, chunk_size}, float_opts);
  auto exp_a_last =
      workspace_tensor(ws.exp_a_last, {batch, nheads, num_chunks}, float_opts);
  auto state_f_local = initial_state.to(torch::kFloat);
  auto final_state_f_kernel =
      torch::empty_like(state, z.options().dtype(torch::kFloat));
  const int threads = 256;
  dim3 dt_grid(batch, nheads, num_chunks);
  int dt_threads = 1;
  while (dt_threads < chunk_size)
    dt_threads <<= 1;
  size_t dt_shared = dt_threads * sizeof(float);
  // Use bf16 for x,b,c buffers (matching Triton reference for bandwidth)
  auto bf16_opts = z.options().dtype(torch::kBFloat16);
  auto x_buf = workspace_tensor(
      ws.x_buf, {batch, num_chunks, chunk_size, nheads, headdim}, bf16_opts);
  auto b_buf = workspace_tensor(
      ws.b_buf, {batch, num_chunks, chunk_size, ngroups, d_state}, bf16_opts);
  auto c_buf = workspace_tensor(
      ws.c_buf, {batch, num_chunks, chunk_size, ngroups, d_state}, bf16_opts);
  int64_t pack_total = batch * num_chunks * chunk_size * (conv_dim + nheads);
  int pack_blocks = (pack_total + threads - 1) / threads;
  switch (z.scalar_type()) {
  case at::kFloat:
    conv1d_pack_dt_kernel_bf16out<float><<<pack_blocks, threads>>>(
        z.data_ptr<float>(), w.data_ptr<float>(),
        has_conv_bias ? b.data_ptr<float>() : nullptr, dtb.data_ptr<float>(),
        has_dt_scale ? dt_scale_view.data_ptr<float>() : nullptr,
        has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
        x_buf.data_ptr<at::BFloat16>(), b_buf.data_ptr<at::BFloat16>(),
        c_buf.data_ptr<at::BFloat16>(), dt_buf.data_ptr<float>(),
        batch, seqlen, d_in_proj, conv_dim, conv_kernel, d_mlp, d_ssm, ngroups,
        d_state, chunk_size, num_chunks, headdim, nheads, has_conv_bias,
        has_seq_idx ? 1 : 0, (float)dt_min, (float)dt_max, dt_scale_stride_b,
        dt_scale_stride_t, dt_scale_batch, dt_scale_seqlen, has_dt_scale);
    break;
  case at::kHalf:
    conv1d_pack_dt_kernel_bf16out<at::Half><<<pack_blocks, threads>>>(
        z.data_ptr<at::Half>(), w.data_ptr<at::Half>(),
        has_conv_bias ? b.data_ptr<at::Half>() : nullptr, dtb.data_ptr<at::Half>(),
        has_dt_scale ? dt_scale_view.data_ptr<at::Half>() : nullptr,
        has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
        x_buf.data_ptr<at::BFloat16>(), b_buf.data_ptr<at::BFloat16>(),
        c_buf.data_ptr<at::BFloat16>(), dt_buf.data_ptr<float>(),
        batch, seqlen, d_in_proj, conv_dim, conv_kernel, d_mlp, d_ssm, ngroups,
        d_state, chunk_size, num_chunks, headdim, nheads, has_conv_bias,
        has_seq_idx ? 1 : 0, (float)dt_min, (float)dt_max, dt_scale_stride_b,
        dt_scale_stride_t, dt_scale_batch, dt_scale_seqlen, has_dt_scale);
    break;
  case at::kBFloat16:
    conv1d_pack_dt_kernel_bf16out<at::BFloat16><<<pack_blocks, threads>>>(
        z.data_ptr<at::BFloat16>(), w.data_ptr<at::BFloat16>(),
        has_conv_bias ? b.data_ptr<at::BFloat16>() : nullptr,
        dtb.data_ptr<at::BFloat16>(),
        has_dt_scale ? dt_scale_view.data_ptr<at::BFloat16>() : nullptr,
        has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
        x_buf.data_ptr<at::BFloat16>(), b_buf.data_ptr<at::BFloat16>(),
        c_buf.data_ptr<at::BFloat16>(), dt_buf.data_ptr<float>(),
        batch, seqlen, d_in_proj, conv_dim, conv_kernel, d_mlp, d_ssm, ngroups,
        d_state, chunk_size, num_chunks, headdim, nheads, has_conv_bias,
        has_seq_idx ? 1 : 0, (float)dt_min, (float)dt_max, dt_scale_stride_b,
        dt_scale_stride_t, dt_scale_batch, dt_scale_seqlen, has_dt_scale);
    break;
  default:
    TORCH_CHECK(false, "unsupported dtype for conv1d_pack_dt_kernel");
  }
  dt_cumsum_from_dt_kernel<<<dt_grid, dt_threads, dt_shared>>>(
      dt_buf.data_ptr<float>(), alog.data_ptr<float>(), dA_buf.data_ptr<float>(),
      exp_a_last.data_ptr<float>(), batch, seqlen, nheads, chunk_size);
  int64_t heads_per_group = nheads / ngroups;
  auto x_g_mat = x_buf
                     .view({batch, num_chunks, chunk_size, ngroups,
                            heads_per_group, headdim})
                     .permute({0, 1, 3, 2, 4, 5})
                     .contiguous()
                     .view({batch * num_chunks * ngroups, chunk_size,
                            heads_per_group * headdim});
  auto b_mat = b_buf.permute({0, 1, 3, 2, 4})
                   .contiguous()
                   .view({batch * num_chunks * ngroups, chunk_size, d_state});
  auto c_mat = c_buf.permute({0, 1, 3, 2, 4})
                   .contiguous()
                   .view({batch * num_chunks * ngroups, chunk_size, d_state});
  auto chunk_state_mat = workspace_tensor(
      ws.chunk_state_mat,
      {batch * num_chunks * ngroups, d_state, heads_per_group * headdim},
      float_opts);
  chunk_state_mat.zero_();
  dim3 cs_grid((heads_per_group * headdim + 63) / 64, (d_state + 63) / 64,
               batch * num_chunks * ngroups);
  dim3 bmm_threads(16, 16);
  bmm_kt_kn_scale_x_kernel_bf16in<<<cs_grid, bmm_threads>>>(
      b_mat.data_ptr<at::BFloat16>(), x_g_mat.data_ptr<at::BFloat16>(),
      dt_buf.data_ptr<float>(), dA_buf.data_ptr<float>(),
      has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
      chunk_state_mat.data_ptr<float>(), batch * num_chunks * ngroups, d_state,
      chunk_size, heads_per_group * headdim, seqlen, num_chunks, ngroups,
      nheads, headdim, chunk_size, has_seq_idx ? 1 : 0);
  auto chunk_state = chunk_state_mat.transpose(1, 2).contiguous().view(
      {batch, num_chunks, nheads, headdim, d_state});
  auto state_in = torch::zeros_like(chunk_state);
  int64_t state_total = batch * nheads * headdim * d_state;
  int state_blocks = (state_total + threads - 1) / threads;
  state_passing_fwd_kernel<<<state_blocks, threads>>>(
      chunk_state.data_ptr<float>(), exp_a_last.data_ptr<float>(),
      state_f_local.data_ptr<float>(), state_in.data_ptr<float>(),
      final_state_f_kernel.data_ptr<float>(),
      has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride, batch,
      nheads, headdim, d_state, num_chunks, chunk_size, seqlen,
      has_seq_idx ? 1 : 0);
  final_state.copy_(final_state_f_kernel.to(final_state.scalar_type()));

  // Precompute CB = C @ B^T with causal masking
  auto cb_buf = workspace_tensor(
      ws.cb_buf, {batch * num_chunks * ngroups, chunk_size, chunk_size}, float_opts);

  if (!has_seq_idx) {
    // Fast path: use cuBLAS with bf16 inputs, fp32 compute, fp32 output (tensor cores)
    cublasHandle_t handle = get_cublas_handle();
    float alpha = 1.0f, beta = 0.0f;
    int64_t batches = batch * num_chunks * ngroups;

    // C @ B^T: C is [batches, M, K], B is [batches, N, K], output is [batches, M, N]
    // Use cublasGemmStridedBatchedEx for bf16 inputs with fp32 accumulation
    cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T,  // B transposed
        CUBLAS_OP_N,  // C not transposed
        chunk_size,   // N (cols of result)
        chunk_size,   // M (rows of result)
        d_state,      // K
        &alpha,
        b_mat.data_ptr<at::BFloat16>(), CUDA_R_16BF, d_state, chunk_size * d_state,
        c_mat.data_ptr<at::BFloat16>(), CUDA_R_16BF, d_state, chunk_size * d_state,
        &beta,
        cb_buf.data_ptr<float>(), CUDA_R_32F, chunk_size, chunk_size * chunk_size,
        batches,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Apply causal mask
    dim3 mask_grid((chunk_size + 63) / 64, (chunk_size + 63) / 64, batches);
    dim3 mask_threads(16, 16);
    apply_causal_mask_simple_kernel<<<mask_grid, mask_threads>>>(
        cb_buf.data_ptr<float>(), batches, chunk_size, seqlen, num_chunks);
  } else {
    // Slow path with seq_idx support
    dim3 cb_grid((chunk_size + 63) / 64, (chunk_size + 63) / 64,
                 batch * num_chunks * ngroups);
    dim3 cb_threads(16, 16);
    bmm_cb_causal_kernel_bf16in<<<cb_grid, cb_threads>>>(
        c_mat.data_ptr<at::BFloat16>(), b_mat.data_ptr<at::BFloat16>(),
        dA_buf.data_ptr<float>(),
        seq.data_ptr<int64_t>(), seq_stride,
        cb_buf.data_ptr<float>(), batch * num_chunks * ngroups, chunk_size,
        d_state, seqlen, num_chunks, ngroups, nheads, 1);
  }

  // Prefer WMMA kernel when aligned; fallback to v3 otherwise.
  const char* wmma_env = std::getenv("MAMBA_WMMA");
  bool wmma_enabled = !wmma_env || std::atoi(wmma_env) != 0;
  bool wmma_compatible = !has_seq_idx && (headdim % 16 == 0) && (chunk_size % 16 == 0);
  TORCH_CHECK(!wmma_enabled || wmma_compatible,
              "WMMA enabled but incompatible shape/seq_idx; disable with MAMBA_WMMA=0");
  bool use_wmma = wmma_enabled && wmma_compatible;
  dim3 block(16, 16);
  dim3 grid_v3((headdim + 63) / 64, (chunk_size + 63) / 64,
               batch * num_chunks * nheads);
  dim3 grid_wmma((headdim + 63) / 64, (chunk_size + 31) / 32,
                 batch * num_chunks * nheads);
  
  // Prepare gating params
  int64_t z_stride_b = d_in_proj * seqlen;
  int64_t z_stride_l = d_in_proj;
  int64_t z_offset = 2 * d_mlp;

  if (fuse_gate) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16, z.scalar_type(),
          "chunk_scan_fwd_fused", ([&] {
              if (use_wmma) {
                  chunk_scan_fwd_kernel_wmma_bf16in_fused<scalar_t, true><<<grid_wmma, block>>>(
                      cb_buf.data_ptr<float>(), x_buf.data_ptr<at::BFloat16>(),
                      dt_buf.data_ptr<float>(), dA_buf.data_ptr<float>(),
                      c_mat.data_ptr<at::BFloat16>(), state_in.data_ptr<float>(), d.data_ptr<float>(),
                      z.data_ptr<scalar_t>(), z_stride_b, z_stride_l, z_offset,
                      has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
                      d_has_hdim ? 1 : 0, y_padded_f.data_ptr<float>(), batch, num_chunks,
                      chunk_size, nheads, headdim, d_state, ngroups, seqlen,
                      has_seq_idx ? 1 : 0);
              } else {
                  chunk_scan_fwd_kernel_v3_bf16in_fused<scalar_t, true><<<grid_v3, block>>>(
                      cb_buf.data_ptr<float>(), x_buf.data_ptr<at::BFloat16>(),
                      dt_buf.data_ptr<float>(), dA_buf.data_ptr<float>(),
                      c_mat.data_ptr<at::BFloat16>(), state_in.data_ptr<float>(), d.data_ptr<float>(),
                      z.data_ptr<scalar_t>(), z_stride_b, z_stride_l, z_offset,
                      has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
                      d_has_hdim ? 1 : 0, y_padded_f.data_ptr<float>(), batch, num_chunks,
                      chunk_size, nheads, headdim, d_state, ngroups, seqlen,
                      has_seq_idx ? 1 : 0);
              }
          }));
  } else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16, z.scalar_type(),
          "chunk_scan_fwd_nofuse", ([&] {
              if (use_wmma) {
                  chunk_scan_fwd_kernel_wmma_bf16in_fused<scalar_t, false><<<grid_wmma, block>>>(
                      cb_buf.data_ptr<float>(), x_buf.data_ptr<at::BFloat16>(),
                      dt_buf.data_ptr<float>(), dA_buf.data_ptr<float>(),
                      c_mat.data_ptr<at::BFloat16>(), state_in.data_ptr<float>(), d.data_ptr<float>(),
                      z.data_ptr<scalar_t>(), z_stride_b, z_stride_l, z_offset,
                      has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
                      d_has_hdim ? 1 : 0, y_padded_f.data_ptr<float>(), batch, num_chunks,
                      chunk_size, nheads, headdim, d_state, ngroups, seqlen,
                      has_seq_idx ? 1 : 0);
              } else {
                  chunk_scan_fwd_kernel_v3_bf16in_fused<scalar_t, false><<<grid_v3, block>>>(
                      cb_buf.data_ptr<float>(), x_buf.data_ptr<at::BFloat16>(),
                      dt_buf.data_ptr<float>(), dA_buf.data_ptr<float>(),
                      c_mat.data_ptr<at::BFloat16>(), state_in.data_ptr<float>(), d.data_ptr<float>(),
                      z.data_ptr<scalar_t>(), z_stride_b, z_stride_l, z_offset,
                      has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
                      d_has_hdim ? 1 : 0, y_padded_f.data_ptr<float>(), batch, num_chunks,
                      chunk_size, nheads, headdim, d_state, ngroups, seqlen,
                      has_seq_idx ? 1 : 0);
              }
          }));
  }

  y.copy_(y_padded_f.view({batch, num_chunks * chunk_size, nheads, headdim})
              .slice(1, 0, seqlen)
              .to(y.scalar_type()));
  return {y, final_state};
}

std::vector<torch::Tensor> mamba_fused_forward_stateful_cuda(
    const torch::Tensor &zxbcdt, const torch::Tensor &conv_w,
    const torch::Tensor &conv_b, const torch::Tensor &dt_bias,
    const torch::Tensor &a_log, const torch::Tensor &d_param,
    const torch::Tensor &dt_scale, const torch::Tensor &initial_state,
    const torch::Tensor &conv_state, const torch::Tensor &seq_idx,
    int64_t chunk_size, int64_t ngroups, int64_t headdim, double dt_min,
    double dt_max, bool fuse_gate = false) {
  at::NoGradGuard no_grad;
  auto z = zxbcdt.contiguous();
  auto w = conv_w.to(z.scalar_type()).contiguous();
  auto b = conv_b.defined() ? conv_b.to(z.scalar_type()).contiguous()
                            : torch::Tensor();
  bool has_conv_bias = b.defined() && b.numel() > 0;
  auto dtb = dt_bias.to(z.scalar_type()).contiguous();
  auto alog = a_log.to(z.scalar_type()).contiguous();
  auto d = d_param.to(torch::kFloat).contiguous();
  bool d_has_hdim = d.dim() == 2;
  auto state = initial_state.to(z.scalar_type()).contiguous();
  auto conv_state_in = conv_state.to(z.scalar_type()).contiguous();
  bool has_conv_state = conv_state_in.defined() && conv_state_in.numel() > 0;
  bool has_seq_idx = seq_idx.defined() && seq_idx.numel() > 0;
  TORCH_CHECK(has_conv_state, "conv_state must be provided for stateful forward");
  TORCH_CHECK(!has_seq_idx, "stateful conv does not support seq_idx");
  auto seq = has_seq_idx ? seq_idx.contiguous() : torch::Tensor();
  int64_t seq_stride = has_seq_idx ? seq.stride(0) : 0;
  bool has_dt_scale = dt_scale.defined() && dt_scale.numel() > 0;
  auto dt_scale_view = dt_scale;
  int64_t dt_scale_stride_b = 0, dt_scale_stride_t = 0, dt_scale_batch = 0,
          dt_scale_seqlen = 0;
  int64_t batch = z.size(0), seqlen = z.size(1), d_in_proj = z.size(2),
          nheads = a_log.size(0), d_ssm = nheads * headdim;
  int64_t d_state = (w.size(0) - d_ssm) / (2 * ngroups);
  int64_t d_inner = (d_in_proj - 2 * ngroups * d_state - nheads) / 2,
          d_mlp = d_inner - d_ssm, conv_dim = d_ssm + 2 * ngroups * d_state,
          conv_kernel = w.size(1);
  TORCH_CHECK(nheads % ngroups == 0, "nheads must be divisible by ngroups");
  TORCH_CHECK(w.dim() == 2, "conv_w must be 2D");
  TORCH_CHECK(d_state <= kMaxDState, "d_state exceeds max supported");
  TORCH_CHECK(conv_state_in.dim() == 3, "conv_state must be [batch, conv_dim, d_conv]");
  TORCH_CHECK(conv_state_in.size(0) == batch, "conv_state batch mismatch");
  TORCH_CHECK(conv_state_in.size(1) == conv_dim, "conv_state dim mismatch");
  TORCH_CHECK(conv_state_in.size(2) == conv_kernel, "conv_state kernel mismatch");
  if (has_dt_scale) {
    if (dt_scale_view.dim() == 3)
      dt_scale_view = dt_scale_view.squeeze(-1);
    dt_scale_stride_b = dt_scale_view.stride(0);
    dt_scale_stride_t = dt_scale_view.stride(1);
    dt_scale_batch = dt_scale_view.size(0);
    dt_scale_seqlen = dt_scale_view.size(1);
  }
  int64_t num_chunks = (seqlen + chunk_size - 1) / chunk_size;
  auto y = torch::empty({batch, seqlen, nheads, headdim}, z.options());
  auto y_padded_f =
      torch::empty({batch, num_chunks, chunk_size, nheads, headdim},
                   z.options().dtype(torch::kFloat));
  auto final_state = torch::empty_like(state);
  auto float_opts = z.options().dtype(torch::kFloat);
  auto &ws = workspace_for_device(z.device());
  auto dt_buf = workspace_tensor(
      ws.dt_buf, {batch, nheads, num_chunks, chunk_size}, float_opts);
  auto dA_buf = workspace_tensor(
      ws.dA_buf, {batch, nheads, num_chunks, chunk_size}, float_opts);
  auto exp_a_last =
      workspace_tensor(ws.exp_a_last, {batch, nheads, num_chunks}, float_opts);
  auto state_f_local = initial_state.to(torch::kFloat);
  auto final_state_f_kernel =
      torch::empty_like(state, z.options().dtype(torch::kFloat));
  const int threads = 256;
  dim3 dt_grid(batch, nheads, num_chunks);
  int dt_threads = 1;
  while (dt_threads < chunk_size)
    dt_threads <<= 1;
  size_t dt_shared = dt_threads * sizeof(float);
  switch (z.scalar_type()) {
  case at::kFloat:
    dt_cumsum_kernel<float><<<dt_grid, dt_threads, dt_shared>>>(
        z.data_ptr<float>(), dtb.data_ptr<float>(), alog.data_ptr<float>(),
        has_dt_scale ? dt_scale_view.data_ptr<float>() : nullptr,
        dt_buf.data_ptr<float>(), dA_buf.data_ptr<float>(),
        exp_a_last.data_ptr<float>(), batch, seqlen, nheads, d_in_proj, d_mlp,
        (float)dt_min, (float)dt_max, chunk_size, dt_scale_stride_b,
        dt_scale_stride_t, dt_scale_batch, dt_scale_seqlen, has_dt_scale);
    break;
  case at::kHalf:
    dt_cumsum_kernel<at::Half><<<dt_grid, dt_threads, dt_shared>>>(
        z.data_ptr<at::Half>(), dtb.data_ptr<at::Half>(),
        alog.data_ptr<at::Half>(),
        has_dt_scale ? dt_scale_view.data_ptr<at::Half>() : nullptr,
        dt_buf.data_ptr<float>(), dA_buf.data_ptr<float>(),
        exp_a_last.data_ptr<float>(), batch, seqlen, nheads, d_in_proj, d_mlp,
        (float)dt_min, (float)dt_max, chunk_size, dt_scale_stride_b,
        dt_scale_stride_t, dt_scale_batch, dt_scale_seqlen, has_dt_scale);
    break;
  case at::kBFloat16:
    dt_cumsum_kernel<at::BFloat16><<<dt_grid, dt_threads, dt_shared>>>(
        z.data_ptr<at::BFloat16>(), dtb.data_ptr<at::BFloat16>(),
        alog.data_ptr<at::BFloat16>(),
        has_dt_scale ? dt_scale_view.data_ptr<at::BFloat16>() : nullptr,
        dt_buf.data_ptr<float>(), dA_buf.data_ptr<float>(),
        exp_a_last.data_ptr<float>(), batch, seqlen, nheads, d_in_proj, d_mlp,
        (float)dt_min, (float)dt_max, chunk_size, dt_scale_stride_b,
        dt_scale_stride_t, dt_scale_batch, dt_scale_seqlen, has_dt_scale);
    break;
  default:
    TORCH_CHECK(false, "unsupported dtype for forward");
  }
  // Use bf16 buffers for tensor core acceleration (matching training path)
  auto bf16_opts = z.options().dtype(torch::kBFloat16);
  auto x_buf = workspace_tensor(
      ws.x_buf, {batch, num_chunks, chunk_size, nheads, headdim}, bf16_opts);
  auto b_buf = workspace_tensor(
      ws.b_buf, {batch, num_chunks, chunk_size, ngroups, d_state}, bf16_opts);
  auto c_buf = workspace_tensor(
      ws.c_buf, {batch, num_chunks, chunk_size, ngroups, d_state}, bf16_opts);
  int64_t pack_total = batch * num_chunks * chunk_size * conv_dim;
  int pack_blocks = (pack_total + threads - 1) / threads;
  switch (z.scalar_type()) {
  case at::kFloat:
    conv1d_pack_stateful_kernel_bf16out<float><<<pack_blocks, threads>>>(
        z.data_ptr<float>(), w.data_ptr<float>(),
        has_conv_bias ? b.data_ptr<float>() : nullptr,
        conv_state_in.data_ptr<float>(),
        has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
        x_buf.data_ptr<at::BFloat16>(), b_buf.data_ptr<at::BFloat16>(), c_buf.data_ptr<at::BFloat16>(),
        batch, seqlen, d_in_proj, conv_dim, conv_kernel, d_mlp, d_ssm, ngroups,
        d_state, chunk_size, num_chunks, headdim, has_conv_bias,
        has_seq_idx ? 1 : 0, has_conv_state ? 1 : 0);
    break;
  case at::kHalf:
    conv1d_pack_stateful_kernel_bf16out<at::Half><<<pack_blocks, threads>>>(
        z.data_ptr<at::Half>(), w.data_ptr<at::Half>(),
        has_conv_bias ? b.data_ptr<at::Half>() : nullptr,
        conv_state_in.data_ptr<at::Half>(),
        has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
        x_buf.data_ptr<at::BFloat16>(), b_buf.data_ptr<at::BFloat16>(), c_buf.data_ptr<at::BFloat16>(),
        batch, seqlen, d_in_proj, conv_dim, conv_kernel, d_mlp, d_ssm, ngroups,
        d_state, chunk_size, num_chunks, headdim, has_conv_bias,
        has_seq_idx ? 1 : 0, has_conv_state ? 1 : 0);
    break;
  case at::kBFloat16:
    conv1d_pack_stateful_kernel_bf16out<at::BFloat16><<<pack_blocks, threads>>>(
        z.data_ptr<at::BFloat16>(), w.data_ptr<at::BFloat16>(),
        has_conv_bias ? b.data_ptr<at::BFloat16>() : nullptr,
        conv_state_in.data_ptr<at::BFloat16>(),
        has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
        x_buf.data_ptr<at::BFloat16>(), b_buf.data_ptr<at::BFloat16>(), c_buf.data_ptr<at::BFloat16>(),
        batch, seqlen, d_in_proj, conv_dim, conv_kernel, d_mlp, d_ssm, ngroups,
        d_state, chunk_size, num_chunks, headdim, has_conv_bias,
        has_seq_idx ? 1 : 0, has_conv_state ? 1 : 0);
    break;
  default:
    TORCH_CHECK(false, "unsupported dtype for conv1d_pack_stateful_kernel");
  }
  int64_t heads_per_group = nheads / ngroups;
  auto x_g_mat = x_buf
                     .view({batch, num_chunks, chunk_size, ngroups,
                            heads_per_group, headdim})
                     .permute({0, 1, 3, 2, 4, 5})
                     .contiguous()
                     .view({batch * num_chunks * ngroups, chunk_size,
                            heads_per_group * headdim});
  auto b_mat = b_buf.permute({0, 1, 3, 2, 4})
                   .contiguous()
                   .view({batch * num_chunks * ngroups, chunk_size, d_state});
  auto c_mat = c_buf.permute({0, 1, 3, 2, 4})
                   .contiguous()
                   .view({batch * num_chunks * ngroups, chunk_size, d_state});
  auto chunk_state_mat = workspace_tensor(
      ws.chunk_state_mat,
      {batch * num_chunks * ngroups, d_state, heads_per_group * headdim},
      float_opts);
  chunk_state_mat.zero_();
  // Use cuBLAS with tensor cores for chunk_state computation
  dim3 cs_grid((heads_per_group * headdim + 63) / 64, (d_state + 63) / 64,
               batch * num_chunks * ngroups);
  dim3 bmm_threads(16, 16);
  bmm_kt_kn_scale_x_kernel_bf16in<<<cs_grid, bmm_threads>>>(
      b_mat.data_ptr<at::BFloat16>(), x_g_mat.data_ptr<at::BFloat16>(),
      dt_buf.data_ptr<float>(), dA_buf.data_ptr<float>(),
      has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
      chunk_state_mat.data_ptr<float>(), batch * num_chunks * ngroups, d_state,
      chunk_size, heads_per_group * headdim, seqlen, num_chunks, ngroups,
      nheads, headdim, chunk_size, has_seq_idx ? 1 : 0);
  auto chunk_state = chunk_state_mat.transpose(1, 2).contiguous().view(
      {batch, num_chunks, nheads, headdim, d_state});
  auto state_in = torch::zeros_like(chunk_state);
  int64_t state_total = batch * nheads * headdim * d_state;
  int state_blocks = (state_total + threads - 1) / threads;
  state_passing_fwd_kernel<<<state_blocks, threads>>>(
      chunk_state.data_ptr<float>(), exp_a_last.data_ptr<float>(),
      state_f_local.data_ptr<float>(), state_in.data_ptr<float>(),
      final_state_f_kernel.data_ptr<float>(),
      has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride, batch,
      nheads, headdim, d_state, num_chunks, chunk_size, seqlen,
      has_seq_idx ? 1 : 0);
  final_state.copy_(final_state_f_kernel.to(final_state.scalar_type()));

  // Precompute CB = C @ B^T using cuBLAS with tensor cores
  auto cb_buf = workspace_tensor(
      ws.cb_buf, {batch * num_chunks * ngroups, chunk_size, chunk_size}, float_opts);
  {
    cublasHandle_t handle = get_cublas_handle();
    float alpha = 1.0f, beta = 0.0f;
    int64_t batches = batch * num_chunks * ngroups;
    cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        chunk_size, chunk_size, d_state,
        &alpha,
        b_mat.data_ptr<at::BFloat16>(), CUDA_R_16BF, d_state, chunk_size * d_state,
        c_mat.data_ptr<at::BFloat16>(), CUDA_R_16BF, d_state, chunk_size * d_state,
        &beta,
        cb_buf.data_ptr<float>(), CUDA_R_32F, chunk_size, chunk_size * chunk_size,
        batches,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    dim3 mask_grid((chunk_size + 63) / 64, (chunk_size + 63) / 64, batches);
    dim3 mask_threads(16, 16);
    apply_causal_mask_simple_kernel<<<mask_grid, mask_threads>>>(
        cb_buf.data_ptr<float>(), batches, chunk_size, seqlen, num_chunks);
  }

  // Prefer WMMA kernel when aligned; fallback to v3 otherwise.
  const char* wmma_env = std::getenv("MAMBA_WMMA");
  bool wmma_enabled = !wmma_env || std::atoi(wmma_env) != 0;
  bool wmma_compatible = !has_seq_idx && (headdim % 16 == 0) && (chunk_size % 16 == 0);
  TORCH_CHECK(!wmma_enabled || wmma_compatible,
              "WMMA enabled but incompatible shape/seq_idx; disable with MAMBA_WMMA=0");
  bool use_wmma = wmma_enabled && wmma_compatible;
  dim3 block(16, 16);
  dim3 grid_v3((headdim + 63) / 64, (chunk_size + 63) / 64,
               batch * num_chunks * nheads);
  dim3 grid_wmma((headdim + 63) / 64, (chunk_size + 31) / 32,
                 batch * num_chunks * nheads);
  
  // Prepare gating params
  int64_t z_stride_b = d_in_proj * seqlen;
  int64_t z_stride_l = d_in_proj;
  int64_t z_offset = 2 * d_mlp;

  if (fuse_gate) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16, z.scalar_type(),
          "chunk_scan_fwd_fused_stateful", ([&] {
              if (use_wmma) {
                  chunk_scan_fwd_kernel_wmma_bf16in_fused<scalar_t, true><<<grid_wmma, block>>>(
                      cb_buf.data_ptr<float>(), x_buf.data_ptr<at::BFloat16>(),
                      dt_buf.data_ptr<float>(), dA_buf.data_ptr<float>(),
                      c_mat.data_ptr<at::BFloat16>(), state_in.data_ptr<float>(), d.data_ptr<float>(),
                      z.data_ptr<scalar_t>(), z_stride_b, z_stride_l, z_offset,
                      has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
                      d_has_hdim ? 1 : 0, y_padded_f.data_ptr<float>(), batch, num_chunks,
                      chunk_size, nheads, headdim, d_state, ngroups, seqlen,
                      has_seq_idx ? 1 : 0);
              } else {
                  chunk_scan_fwd_kernel_v3_bf16in_fused<scalar_t, true><<<grid_v3, block>>>(
                      cb_buf.data_ptr<float>(), x_buf.data_ptr<at::BFloat16>(),
                      dt_buf.data_ptr<float>(), dA_buf.data_ptr<float>(),
                      c_mat.data_ptr<at::BFloat16>(), state_in.data_ptr<float>(), d.data_ptr<float>(),
                      z.data_ptr<scalar_t>(), z_stride_b, z_stride_l, z_offset,
                      has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
                      d_has_hdim ? 1 : 0, y_padded_f.data_ptr<float>(), batch, num_chunks,
                      chunk_size, nheads, headdim, d_state, ngroups, seqlen,
                      has_seq_idx ? 1 : 0);
              }
          }));
  } else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16, z.scalar_type(),
          "chunk_scan_fwd_nofuse_stateful", ([&] {
              if (use_wmma) {
                  chunk_scan_fwd_kernel_wmma_bf16in_fused<scalar_t, false><<<grid_wmma, block>>>(
                      cb_buf.data_ptr<float>(), x_buf.data_ptr<at::BFloat16>(),
                      dt_buf.data_ptr<float>(), dA_buf.data_ptr<float>(),
                      c_mat.data_ptr<at::BFloat16>(), state_in.data_ptr<float>(), d.data_ptr<float>(),
                      z.data_ptr<scalar_t>(), z_stride_b, z_stride_l, z_offset,
                      has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
                      d_has_hdim ? 1 : 0, y_padded_f.data_ptr<float>(), batch, num_chunks,
                      chunk_size, nheads, headdim, d_state, ngroups, seqlen,
                      has_seq_idx ? 1 : 0);
              } else {
                  chunk_scan_fwd_kernel_v3_bf16in_fused<scalar_t, false><<<grid_v3, block>>>(
                      cb_buf.data_ptr<float>(), x_buf.data_ptr<at::BFloat16>(),
                      dt_buf.data_ptr<float>(), dA_buf.data_ptr<float>(),
                      c_mat.data_ptr<at::BFloat16>(), state_in.data_ptr<float>(), d.data_ptr<float>(),
                      z.data_ptr<scalar_t>(), z_stride_b, z_stride_l, z_offset,
                      has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
                      d_has_hdim ? 1 : 0, y_padded_f.data_ptr<float>(), batch, num_chunks,
                      chunk_size, nheads, headdim, d_state, ngroups, seqlen,
                      has_seq_idx ? 1 : 0);
              }
          }));
  }

  y.copy_(y_padded_f.view({batch, num_chunks * chunk_size, nheads, headdim})
              .slice(1, 0, seqlen)
              .to(y.scalar_type()));

  auto conv_state_out = torch::empty_like(conv_state_in);
  int64_t conv_total = batch * conv_dim * conv_kernel;
  int conv_blocks = (conv_total + threads - 1) / threads;
  switch (z.scalar_type()) {
  case at::kFloat:
    update_conv_state_kernel<float><<<conv_blocks, threads>>>(
        z.data_ptr<float>(), conv_state_in.data_ptr<float>(),
        conv_state_out.data_ptr<float>(), batch, seqlen, d_in_proj, conv_dim,
        conv_kernel, d_mlp, d_ssm);
    break;
  case at::kHalf:
    update_conv_state_kernel<at::Half><<<conv_blocks, threads>>>(
        z.data_ptr<at::Half>(), conv_state_in.data_ptr<at::Half>(),
        conv_state_out.data_ptr<at::Half>(), batch, seqlen, d_in_proj, conv_dim,
        conv_kernel, d_mlp, d_ssm);
    break;
  case at::kBFloat16:
    update_conv_state_kernel<at::BFloat16><<<conv_blocks, threads>>>(
        z.data_ptr<at::BFloat16>(), conv_state_in.data_ptr<at::BFloat16>(),
        conv_state_out.data_ptr<at::BFloat16>(), batch, seqlen, d_in_proj,
        conv_dim, conv_kernel, d_mlp, d_ssm);
    break;
  default:
    TORCH_CHECK(false, "unsupported dtype for conv_state update");
  }

  return {y, final_state, conv_state_out};
}

namespace {
template <typename scalar_t, typename out_t>
__global__ void
fused_post_ssm_kernel(const float *y_ssm, const scalar_t *zxbcdt,
                      const float *rms_w, out_t *out, int64_t batch,
                      int64_t seqlen, int64_t d_in_proj, int64_t d_inner,
                      int64_t d_ssm, int64_t d_mlp, int64_t ngroups, float eps,
                      bool norm_before_gate, bool has_rmsnorm, bool is_gated) {
  int64_t row_idx = blockIdx.x;
  if (row_idx >= batch * seqlen)
    return;
  int64_t b = row_idx / seqlen;
  int64_t l = row_idx % seqlen;
  int64_t group_size = d_ssm / ngroups;
  int64_t tid = threadIdx.x;
  int64_t nt = blockDim.x;
  if (d_mlp > 0) {
    for (int64_t i = tid; i < d_mlp; i += nt) {
      float z0 = to_float(zxbcdt[(b * seqlen + l) * d_in_proj + i]);
      float x0 = to_float(zxbcdt[(b * seqlen + l) * d_in_proj + d_mlp + i]);
      out[(b * seqlen + l) * d_inner + i] = from_float<out_t>(silu_f(z0) * x0);
    }
  }
  int64_t z_start = 2 * d_mlp;
  extern __shared__ float shmem[];
  for (int64_t g = 0; g < ngroups; ++g) {
    float local_sum_sq = 0.0f;
    int64_t group_offset = g * group_size;
    for (int64_t i = tid; i < group_size; i += nt) {
      int64_t c = group_offset + i;
      float ys = y_ssm[(b * seqlen + l) * d_ssm + c];
      if (has_rmsnorm) {
        if (norm_before_gate) {
          local_sum_sq += ys * ys;
        } else {
          // If is_gated, ys is already ys * silu(z)
          if (is_gated) {
            local_sum_sq += ys * ys;
          } else {
            float z_val =
                to_float(zxbcdt[(b * seqlen + l) * d_in_proj + z_start + c]);
            float val = ys * silu_f(z_val);
            local_sum_sq += val * val;
          }
        }
      }
    }
    float group_rms = 1.0f;
    if (has_rmsnorm) {
      shmem[tid] = local_sum_sq;
      __syncthreads();
      for (int offset = nt / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
          shmem[tid] += shmem[tid + offset];
        __syncthreads();
      }
      group_rms = rsqrtf(shmem[0] / group_size + eps);
    }
    for (int64_t i = tid; i < group_size; i += nt) {
      int64_t c = group_offset + i;
      float ys = y_ssm[(b * seqlen + l) * d_ssm + c];
      float val;
      if (has_rmsnorm) {
        float weight = rms_w[c];
        if (norm_before_gate) {
          // If norm_before_gate, is_gated must be false (checked by caller or assumed)
          float z_val =
              to_float(zxbcdt[(b * seqlen + l) * d_in_proj + z_start + c]);
          val = (ys * group_rms) * weight * silu_f(z_val);
        } else {
          if (is_gated) {
            val = (ys * group_rms) * weight;
          } else {
            float z_val =
                to_float(zxbcdt[(b * seqlen + l) * d_in_proj + z_start + c]);
            val = (ys * silu_f(z_val) * group_rms) * weight;
          }
        }
      } else {
        if (is_gated) {
            val = ys;
        } else {
            float z_val =
                to_float(zxbcdt[(b * seqlen + l) * d_in_proj + z_start + c]);
            val = ys * silu_f(z_val);
        }
      }
      out[(b * seqlen + l) * d_inner + d_mlp + c] = from_float<out_t>(val);
    }
  }
}
} // namespace

torch::Tensor mamba_fused_post_ssm(const torch::Tensor &y_ssm,
                                   const torch::Tensor &zxbcdt,
                                   const torch::Tensor &rms_w, int64_t d_mlp,
                                   int64_t d_ssm, int64_t ngroups, double eps,
                                   bool norm_before_gate,
                                   torch::ScalarType out_type,
                                   bool is_gated = false) {
  auto y_f = y_ssm.to(torch::kFloat);
  auto rms_w_f =
      rms_w.defined() ? rms_w.to(torch::kFloat).contiguous() : torch::Tensor();
  int64_t batch = zxbcdt.size(0);
  int64_t seqlen = zxbcdt.size(1);
  int64_t d_in_proj = zxbcdt.size(2);
  int64_t d_inner = d_mlp + d_ssm;
  auto out =
      torch::empty({batch, seqlen, d_inner}, zxbcdt.options().dtype(out_type));
  int64_t total_rows = batch * seqlen;
  int threads = 256;
  if (d_ssm / ngroups > 256)
    threads = 512;
  if (d_ssm / ngroups > 512)
    threads = 1024;
  size_t shmem_size = threads * sizeof(float);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, zxbcdt.scalar_type(),
      "mamba_fused_post_ssm", ([&] {
        if (out_type == at::ScalarType::Half)
          fused_post_ssm_kernel<scalar_t, at::Half>
              <<<total_rows, threads, shmem_size>>>(
                  y_f.data_ptr<float>(), zxbcdt.data_ptr<scalar_t>(),
                  rms_w_f.defined() ? rms_w_f.data_ptr<float>() : nullptr,
                  reinterpret_cast<at::Half *>(out.data_ptr<at::Half>()), batch,
                  seqlen, d_in_proj, d_inner, d_ssm, d_mlp, ngroups, (float)eps,
                  norm_before_gate, rms_w_f.defined(), is_gated);
        else if (out_type == at::ScalarType::BFloat16)
          fused_post_ssm_kernel<scalar_t, at::BFloat16>
              <<<total_rows, threads, shmem_size>>>(
                  y_f.data_ptr<float>(), zxbcdt.data_ptr<scalar_t>(),
                  rms_w_f.defined() ? rms_w_f.data_ptr<float>() : nullptr,
                  reinterpret_cast<at::BFloat16 *>(
                      out.data_ptr<at::BFloat16>()),
                  batch, seqlen, d_in_proj, d_inner, d_ssm, d_mlp, ngroups,
                  (float)eps, norm_before_gate, rms_w_f.defined(), is_gated);
        else
          fused_post_ssm_kernel<scalar_t, float>
              <<<total_rows, threads, shmem_size>>>(
                  y_f.data_ptr<float>(), zxbcdt.data_ptr<scalar_t>(),
                  rms_w_f.defined() ? rms_w_f.data_ptr<float>() : nullptr,
                  out.data_ptr<float>(), batch, seqlen, d_in_proj, d_inner,
                  d_ssm, d_mlp, ngroups, (float)eps, norm_before_gate,
                  rms_w_f.defined(), is_gated);
      }));
  return out;
}

std::vector<torch::Tensor> mamba_fused_forward_infer_cuda(
    const torch::Tensor &zxbcdt, const torch::Tensor &conv_w,
    const torch::Tensor &conv_b, const torch::Tensor &dt_bias,
    const torch::Tensor &a_log, const torch::Tensor &d_param,
    const torch::Tensor &dt_scale, const torch::Tensor &initial_state,
    const torch::Tensor &seq_idx, int64_t chunk_size, int64_t ngroups,
    int64_t headdim, double dt_min, double dt_max,
    const torch::Tensor &rmsnorm_weight, double rmsnorm_eps,
    bool norm_before_gate, const torch::Tensor &outproj_w,
    const torch::Tensor &outproj_b) {
  
  // Fuse gate if NOT norm_before_gate
  bool fuse_gate = !norm_before_gate;

  auto outputs = mamba_fused_forward_cuda(
      zxbcdt, conv_w, conv_b, dt_bias, a_log, d_param, dt_scale, initial_state,
      seq_idx, chunk_size, ngroups, headdim, dt_min, dt_max, fuse_gate);
  auto y = outputs[0];
  auto final_state = outputs[1];
  int64_t batch = y.size(0);
  int64_t seqlen = y.size(1);
  int64_t nheads = y.size(2);
  int64_t d_ssm = nheads * headdim;
  int64_t d_in_proj = zxbcdt.size(2);
  int64_t d_state = (conv_w.size(0) - d_ssm) / (2 * ngroups);
  int64_t d_inner = (d_in_proj - 2 * ngroups * d_state - nheads) / 2;
  int64_t d_mlp = d_inner - d_ssm;
  auto y_out = mamba_fused_post_ssm(
      y.view({batch, seqlen, d_ssm}), zxbcdt, rmsnorm_weight, d_mlp, d_ssm,
      ngroups, rmsnorm_eps, norm_before_gate,
      outproj_w.defined() ? outproj_w.scalar_type() : zxbcdt.scalar_type(),
      fuse_gate);
  torch::Tensor out;
  if (outproj_w.defined() && outproj_w.numel() > 0) {
    auto out_2d = y_out.view({batch * seqlen, d_inner});
    c10::optional<torch::Tensor> bias =
        (outproj_b.defined() && outproj_b.numel() > 0)
            ? c10::optional<torch::Tensor>(outproj_b)
            : c10::nullopt;
    out = at::linear(out_2d, outproj_w, bias);
    out = out.view({batch, seqlen, outproj_w.size(0)});
  } else
    out = y_out;
  return {out, final_state};
}

std::vector<torch::Tensor> mamba_fused_backward_cuda(
    const torch::Tensor &zxbcdt, const torch::Tensor &conv_w,
    const torch::Tensor &conv_b, const torch::Tensor &dt_bias,
    const torch::Tensor &a_log, const torch::Tensor &d_param,
    const torch::Tensor &dt_scale, const torch::Tensor &initial_state,
    const torch::Tensor &final_state, const torch::Tensor &y,
    const torch::Tensor &grad_y, const torch::Tensor &grad_final_state,
    const torch::Tensor &seq_idx, int64_t chunk_size, int64_t ngroups,
    int64_t headdim, double dt_min, double dt_max, bool fuse_gate) {
  at::NoGradGuard no_grad;
  auto z = zxbcdt.contiguous();
  auto w = conv_w.to(z.scalar_type()).contiguous();
  auto b = conv_b.defined() ? conv_b.to(z.scalar_type()).contiguous()
                            : torch::Tensor();
  auto dtb = dt_bias.to(z.scalar_type()).contiguous();
  auto alog_f = a_log.to(torch::kFloat).contiguous();
  auto alog_s = a_log.to(z.scalar_type()).contiguous();
  auto d = d_param.to(torch::kFloat).contiguous();
  bool d_has_hdim = d.dim() == 2;
  auto state0 = initial_state.contiguous();
  auto stateN = final_state.contiguous();
  bool has_seq_idx = seq_idx.defined() && seq_idx.numel() > 0;
  auto seq = has_seq_idx ? seq_idx.contiguous() : torch::Tensor();
  int64_t seq_stride = has_seq_idx ? seq.stride(0) : 0;
  bool has_dt_scale = dt_scale.defined() && dt_scale.numel() > 0;
  auto dt_scale_view = dt_scale;
  int64_t dt_scale_stride_b = 0, dt_scale_stride_t = 0, dt_scale_batch = 0,
          dt_scale_seqlen = 0;
  auto gy = grad_y.contiguous();
  auto gy_f = gy.to(torch::kFloat);
  auto float_opts = gy_f.options();
  auto &ws = workspace_for_device(z.device());
  auto gy_input = gy.to(z.scalar_type());
  auto y_f = y.to(torch::kFloat);
  auto gstate = grad_final_state.defined() ? grad_final_state.contiguous()
                                           : torch::zeros_like(stateN);
  int64_t batch = z.size(0), seqlen = z.size(1), d_in_proj = z.size(2),
          nheads = a_log.size(0), d_ssm = nheads * headdim;
  int64_t d_state = (w.size(0) - d_ssm) / (2 * ngroups);
  int64_t d_inner = (d_in_proj - 2 * ngroups * d_state - nheads) / 2,
          d_mlp = d_inner - d_ssm, conv_dim = d_ssm + 2 * ngroups * d_state,
          conv_kernel = w.size(1);
  if (fuse_gate) {
    int64_t z_offset = 2 * d_mlp;
    auto z = zxbcdt.narrow(2, z_offset, d_ssm).to(torch::kFloat);
    auto z_silu = z * z.sigmoid();
    auto z_nonzero = z_silu.ne(0);
    auto denom = torch::where(z_nonzero, z_silu, torch::ones_like(z_silu));
    auto y_view = y_f.view({batch, seqlen, d_ssm});
    auto y_ungated = y_view / denom;
    y_ungated = torch::where(z_nonzero, y_ungated, torch::zeros_like(y_ungated));
    y_f = y_ungated.view({batch, seqlen, nheads, headdim});
    auto z_silu_view = z_silu.view({batch, seqlen, nheads, headdim});
    gy_f = gy_f * z_silu_view;
    gy_input = gy_input * z_silu_view.to(gy_input.scalar_type());
  }
  if (has_dt_scale) {
    dt_scale_view = dt_scale_view.to(z.scalar_type()).contiguous();
    if (dt_scale_view.dim() == 3)
      dt_scale_view = dt_scale_view.squeeze(-1);
    dt_scale_stride_b = dt_scale_view.stride(0);
    dt_scale_stride_t = dt_scale_view.stride(1);
    dt_scale_batch = dt_scale_view.size(0);
    dt_scale_seqlen = dt_scale_view.size(1);
  }
  int64_t num_chunks = (seqlen + chunk_size - 1) / chunk_size;
  int64_t padded_len = num_chunks * chunk_size;
  const int threads = 256;
  auto dt_buf = workspace_tensor(
      ws.dt_buf, {batch, nheads, num_chunks, chunk_size}, float_opts);
  auto dA_buf = workspace_tensor(
      ws.dA_buf, {batch, nheads, num_chunks, chunk_size}, float_opts);
  auto exp_a_last =
      workspace_tensor(ws.exp_a_last, {batch, nheads, num_chunks}, float_opts);
  dim3 dt_grid(batch, nheads, num_chunks);
  int dt_threads = 1;
  while (dt_threads < chunk_size)
    dt_threads <<= 1;
  size_t dt_shared = dt_threads * sizeof(float);
  auto x_buf = workspace_tensor(
      ws.x_buf, {batch, num_chunks, chunk_size, nheads, headdim}, float_opts);
  auto b_buf = workspace_tensor(
      ws.b_buf, {batch, num_chunks, chunk_size, ngroups, d_state}, float_opts);
  auto c_buf = workspace_tensor(
      ws.c_buf, {batch, num_chunks, chunk_size, ngroups, d_state}, float_opts);
  int64_t pack_total = batch * num_chunks * chunk_size * (conv_dim + nheads);
  int pack_blocks = (pack_total + threads - 1) / threads;
  switch (z.scalar_type()) {
  case at::kFloat:
    conv1d_pack_dt_kernel<float><<<pack_blocks, threads>>>(
        z.data_ptr<float>(), w.data_ptr<float>(),
        b.defined() ? b.data_ptr<float>() : nullptr, dtb.data_ptr<float>(),
        has_dt_scale ? dt_scale_view.data_ptr<float>() : nullptr,
        has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
        x_buf.data_ptr<float>(), b_buf.data_ptr<float>(), c_buf.data_ptr<float>(),
        dt_buf.data_ptr<float>(),
        batch, seqlen, d_in_proj, conv_dim, conv_kernel, d_mlp, d_ssm, ngroups,
        d_state, chunk_size, num_chunks, headdim, nheads, b.defined(),
        has_seq_idx ? 1 : 0, (float)dt_min, (float)dt_max, dt_scale_stride_b,
        dt_scale_stride_t, dt_scale_batch, dt_scale_seqlen, has_dt_scale);
    break;
  case at::kHalf:
    conv1d_pack_dt_kernel<at::Half><<<pack_blocks, threads>>>(
        z.data_ptr<at::Half>(), w.data_ptr<at::Half>(),
        b.defined() ? b.data_ptr<at::Half>() : nullptr, dtb.data_ptr<at::Half>(),
        has_dt_scale ? dt_scale_view.data_ptr<at::Half>() : nullptr,
        has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
        x_buf.data_ptr<float>(), b_buf.data_ptr<float>(), c_buf.data_ptr<float>(),
        dt_buf.data_ptr<float>(),
        batch, seqlen, d_in_proj, conv_dim, conv_kernel, d_mlp, d_ssm, ngroups,
        d_state, chunk_size, num_chunks, headdim, nheads, b.defined(),
        has_seq_idx ? 1 : 0, (float)dt_min, (float)dt_max, dt_scale_stride_b,
        dt_scale_stride_t, dt_scale_batch, dt_scale_seqlen, has_dt_scale);
    break;
  case at::kBFloat16:
    conv1d_pack_dt_kernel<at::BFloat16><<<pack_blocks, threads>>>(
        z.data_ptr<at::BFloat16>(), w.data_ptr<at::BFloat16>(),
        b.defined() ? b.data_ptr<at::BFloat16>() : nullptr,
        dtb.data_ptr<at::BFloat16>(),
        has_dt_scale ? dt_scale_view.data_ptr<at::BFloat16>() : nullptr,
        has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
        x_buf.data_ptr<float>(), b_buf.data_ptr<float>(), c_buf.data_ptr<float>(),
        dt_buf.data_ptr<float>(),
        batch, seqlen, d_in_proj, conv_dim, conv_kernel, d_mlp, d_ssm, ngroups,
        d_state, chunk_size, num_chunks, headdim, nheads, b.defined(),
        has_seq_idx ? 1 : 0, (float)dt_min, (float)dt_max, dt_scale_stride_b,
        dt_scale_stride_t, dt_scale_batch, dt_scale_seqlen, has_dt_scale);
    break;
  default:
    TORCH_CHECK(false, "unsupported dtype for backward pack");
  }
  dt_cumsum_from_dt_kernel<<<dt_grid, dt_threads, dt_shared>>>(
      dt_buf.data_ptr<float>(), alog_f.data_ptr<float>(), dA_buf.data_ptr<float>(),
      exp_a_last.data_ptr<float>(), batch, seqlen, nheads, chunk_size);
  int64_t heads_per_group = nheads / ngroups;
  auto x_g_mat = x_buf
                     .view({batch, num_chunks, chunk_size, ngroups,
                            heads_per_group, headdim})
                     .permute({0, 1, 3, 2, 4, 5})
                     .contiguous()
                     .view({batch * num_chunks * ngroups, chunk_size,
                            heads_per_group * headdim});
  auto b_mat = b_buf.permute({0, 1, 3, 2, 4})
                   .contiguous()
                   .view({batch * num_chunks * ngroups, chunk_size, d_state});
  auto c_mat = c_buf.permute({0, 1, 3, 2, 4})
                   .contiguous()
                   .view({batch * num_chunks * ngroups, chunk_size, d_state});
  auto chunk_state_mat = workspace_tensor(
      ws.chunk_state_mat,
      {batch * num_chunks * ngroups, d_state, heads_per_group * headdim},
      float_opts);
  chunk_state_mat.zero_();
  dim3 cs_grid((heads_per_group * headdim + 63) / 64, (d_state + 63) / 64,
               batch * num_chunks * ngroups);
  dim3 bmm_threads(16, 16);
  bmm_kt_kn_scale_x_kernel<<<cs_grid, bmm_threads>>>(
      b_mat.data_ptr<float>(), x_g_mat.data_ptr<float>(),
      dt_buf.data_ptr<float>(), dA_buf.data_ptr<float>(),
      has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride,
      chunk_state_mat.data_ptr<float>(), batch * num_chunks * ngroups, d_state,
      chunk_size, heads_per_group * headdim, seqlen, num_chunks, ngroups,
      nheads, headdim, chunk_size, has_seq_idx ? 1 : 0);
  auto chunk_state = chunk_state_mat.transpose(1, 2).contiguous().view(
      {batch, num_chunks, nheads, headdim, d_state});
  auto state_in = torch::zeros_like(chunk_state);
  auto state_f_local = initial_state.to(torch::kFloat);
  auto final_state_f_dummy = torch::empty_like(state_f_local);
  int64_t state_total = batch * nheads * headdim * d_state;
  int state_blocks = (state_total + threads - 1) / threads;
  state_passing_fwd_kernel<<<state_blocks, threads>>>(
      chunk_state.data_ptr<float>(), exp_a_last.data_ptr<float>(),
      state_f_local.data_ptr<float>(), state_in.data_ptr<float>(),
      final_state_f_dummy.data_ptr<float>(),
      has_seq_idx ? seq.data_ptr<int64_t>() : nullptr, seq_stride, batch,
      nheads, headdim, d_state, num_chunks, chunk_size, seqlen,
      has_seq_idx ? 1 : 0);
  auto gy_chunk = workspace_tensor(
      ws.gy_chunk, {batch, num_chunks, chunk_size, nheads, headdim}, float_opts);
  gy_chunk.zero_();
  int64_t gy_total = batch * num_chunks * chunk_size * nheads * headdim;
  int gy_blocks = (gy_total + threads - 1) / threads;
  switch (gy_input.scalar_type()) {
  case at::kFloat:
    pack_gy_kernel<float><<<gy_blocks, threads>>>(
        gy_input.data_ptr<float>(), gy_chunk.data_ptr<float>(), batch, seqlen,
        num_chunks, chunk_size, nheads, headdim);
    break;
  case at::kHalf:
    pack_gy_kernel<at::Half><<<gy_blocks, threads>>>(
        gy_input.data_ptr<at::Half>(), gy_chunk.data_ptr<float>(), batch,
        seqlen, num_chunks, chunk_size, nheads, headdim);
    break;
  case at::kBFloat16:
    pack_gy_kernel<at::BFloat16><<<gy_blocks, threads>>>(
        gy_input.data_ptr<at::BFloat16>(), gy_chunk.data_ptr<float>(), batch,
        seqlen, num_chunks, chunk_size, nheads, headdim);
    break;
  default:
    TORCH_CHECK(false, "unsupported grad_y_input dtype");
  }
  auto gy_g_mat = gy_chunk
                      .view({batch, num_chunks, chunk_size, ngroups,
                             heads_per_group, headdim})
                      .permute({0, 1, 3, 2, 4, 5})
                      .contiguous()
                      .view({batch * num_chunks * ngroups, chunk_size,
                             heads_per_group * headdim});
  auto state_in_g_mat =
      state_in
          .view({batch, num_chunks, ngroups, heads_per_group, headdim, d_state})
          .permute({0, 1, 2, 5, 3, 4})
          .contiguous()
          .view({batch * num_chunks * ngroups, d_state,
                 heads_per_group * headdim});
  auto dchunk_state_y = workspace_tensor(
      ws.dchunk_state_y,
      {batch * num_chunks * ngroups, d_state, heads_per_group * headdim},
      float_opts);
  auto dC_total = workspace_tensor(
      ws.dC_total, {batch * num_chunks * ngroups, chunk_size, d_state},
      float_opts);
  dim3 bwd_c_grid((d_state + 63) / 64, (chunk_size + 63) / 64,
                  batch * num_chunks * ngroups);
  chunk_scan_bwd_dC_dstate_kernel<<<bwd_c_grid, bmm_threads>>>(
      gy_g_mat.data_ptr<float>(), dA_buf.data_ptr<float>(),
      c_mat.data_ptr<float>(), state_in_g_mat.data_ptr<float>(),
      chunk_state_mat.data_ptr<float>(), dC_total.data_ptr<float>(),
      dchunk_state_y.data_ptr<float>(), batch * num_chunks * ngroups,
      chunk_size, heads_per_group * headdim, d_state, num_chunks, ngroups,
      nheads, headdim);
  auto dstate_in = dchunk_state_y.transpose(1, 2).contiguous().view(
      {batch, num_chunks, nheads, headdim, d_state});
  auto dchunk_state = workspace_tensor(
      ws.dchunk_state,
      {batch, num_chunks, nheads, headdim, d_state},
      float_opts);
  auto ddA = workspace_tensor(
      ws.ddA, {batch, nheads, seqlen}, float_opts);
  ddA.zero_();
  auto dstate0 = workspace_tensor(
      ws.dstate0, state0.sizes(), float_opts);
  auto gstate_f = gstate.to(torch::kFloat);
  state_passing_bwd_kernel<<<state_blocks, threads>>>(
      chunk_state.data_ptr<float>(), state_in.data_ptr<float>(),
      exp_a_last.data_ptr<float>(), dstate_in.data_ptr<float>(),
      gstate_f.data_ptr<float>(), dchunk_state.data_ptr<float>(),
      ddA.data_ptr<float>(), dstate0.data_ptr<float>(), batch, nheads, headdim,
      d_state, seqlen, chunk_size, num_chunks);
  auto dchunk_state_total = (dchunk_state + dstate_in).contiguous();
  auto dchunk_state_mat =
      dchunk_state_total
          .view({batch, num_chunks, ngroups, heads_per_group, headdim, d_state})
          .permute({0, 1, 2, 5, 3, 4})
          .contiguous()
          .view({batch * num_chunks * ngroups, d_state,
                 heads_per_group * headdim});
  auto dx_scaled_state = workspace_tensor(
      ws.dx_scaled_state,
      {batch * num_chunks * ngroups, chunk_size, heads_per_group * headdim},
      float_opts);
  auto ddt_chunk_g = workspace_tensor(
      ws.ddt_chunk_g,
      {batch * num_chunks * ngroups, chunk_size, heads_per_group},
      float_opts);
  ddt_chunk_g.zero_();
  auto ddA_xscaled_g = workspace_tensor(
      ws.ddA_xscaled_g,
      {batch * num_chunks * ngroups, chunk_size, heads_per_group},
      float_opts);
  ddA_xscaled_g.zero_();
  auto dtemp_dummy = workspace_tensor(
      ws.dtemp_dummy, ddt_chunk_g.sizes(), float_opts);
  dtemp_dummy.zero_();
  dim3 bwd_dx_grid((heads_per_group * headdim + 63) / 64,
                   (chunk_size + 63) / 64, batch * num_chunks * ngroups);
  chunk_scan_bwd_dx_kernel<<<bwd_dx_grid, bmm_threads>>>(
      b_mat.data_ptr<float>(), dchunk_state_mat.data_ptr<float>(),
      x_g_mat.data_ptr<float>(), dt_buf.data_ptr<float>(),
      dA_buf.data_ptr<float>(), dx_scaled_state.data_ptr<float>(),
      dtemp_dummy.data_ptr<float>(), ddt_chunk_g.data_ptr<float>(),
      ddA_xscaled_g.data_ptr<float>(), batch * num_chunks * ngroups, chunk_size,
      d_state, heads_per_group * headdim, headdim);
  auto dB_state = workspace_tensor(
      ws.dB_state, {batch * num_chunks * ngroups, chunk_size, d_state},
      float_opts);
  chunk_scan_bwd_dB_kernel<<<bwd_c_grid, bmm_threads>>>(
      x_g_mat.data_ptr<float>(), dt_buf.data_ptr<float>(),
      dA_buf.data_ptr<float>(), dchunk_state_mat.data_ptr<float>(),
      dB_state.data_ptr<float>(), batch * num_chunks * ngroups, chunk_size,
      heads_per_group * headdim, d_state, num_chunks, ngroups, nheads, headdim);
  auto dx_scaled = dx_scaled_state
                       .view({batch, num_chunks, ngroups, chunk_size,
                              heads_per_group, headdim})
                       .permute({0, 1, 3, 2, 4, 5})
                       .contiguous()
                       .view({batch, num_chunks, chunk_size, nheads, headdim});
  auto dx_scan_total = workspace_tensor(
      ws.dx_scan_total, dx_scaled.sizes(), float_opts);
  dx_scan_total.zero_();
  int x_blocks_n =
      (batch * num_chunks * chunk_size * nheads * headdim + threads - 1) /
      threads;
  x_scale_kernel<<<x_blocks_n, threads>>>(
      dx_scaled.data_ptr<float>(), dt_buf.data_ptr<float>(),
      dA_buf.data_ptr<float>(), dx_scan_total.data_ptr<float>(), batch,
      num_chunks, chunk_size, nheads, headdim);
  auto dx_conv_scan = dx_scan_total.view({batch, padded_len, nheads, headdim})
                          .slice(1, 0, seqlen)
                          .contiguous();
  auto x_conv_f = x_buf.view({batch, num_chunks * chunk_size, nheads, headdim})
                      .slice(1, 0, seqlen)
                      .contiguous();
  auto d_broadcast =
      d_has_hdim ? d.view({1, 1, nheads, headdim}) : d.view({1, 1, nheads, 1});
  auto ddt =
      ddt_chunk_g
          .view({batch, num_chunks, ngroups, chunk_size, heads_per_group})
          .permute({0, 1, 3, 2, 4})
          .contiguous()
          .view({batch, nheads, padded_len})
          .slice(2, 0, seqlen)
          .contiguous();
  auto ddA_from_scan =
      ddA_xscaled_g
          .view({batch, num_chunks, ngroups, chunk_size, heads_per_group})
          .permute({0, 1, 3, 2, 4})
          .contiguous()
          .view({batch, nheads, padded_len})
          .slice(2, 0, seqlen)
          .contiguous();
  ddA += ddA_from_scan;
  auto y_scan = y_f - x_conv_f * d_broadcast;
  int64_t ddA_total_n = batch * seqlen * nheads;
  int ddA_blocks_n = (ddA_total_n + threads - 1) / threads;
  ddA_y_bwd_kernel<<<ddA_blocks_n, threads>>>(
      gy_f.data_ptr<float>(), y_scan.data_ptr<float>(), ddA.data_ptr<float>(),
      batch, seqlen, nheads, headdim);
  auto dA_acc = torch::zeros({nheads}, z.options().dtype(torch::kFloat));
  auto dD =
      d_has_hdim
          ? torch::zeros({nheads, headdim}, z.options().dtype(torch::kFloat))
          : torch::zeros({nheads}, z.options().dtype(torch::kFloat));
  auto ddt_bias = torch::zeros({nheads}, z.options().dtype(torch::kFloat));
  dim3 ddA_grid_n(batch, nheads, num_chunks);
  switch (z.scalar_type()) {
  case at::kFloat:
    ddA_to_dtdA_kernel<float><<<ddA_grid_n, dt_threads, dt_shared>>>(
        ddA.data_ptr<float>(), dt_buf.data_ptr<float>(), alog_s.data_ptr<float>(),
        ddt.data_ptr<float>(), dA_acc.data_ptr<float>(), batch, seqlen, nheads,
        chunk_size);
    break;
  case at::kHalf:
    ddA_to_dtdA_kernel<at::Half><<<ddA_grid_n, dt_threads, dt_shared>>>(
        ddA.data_ptr<float>(), dt_buf.data_ptr<float>(),
        alog_s.data_ptr<at::Half>(), ddt.data_ptr<float>(),
        dA_acc.data_ptr<float>(), batch, seqlen, nheads, chunk_size);
    break;
  case at::kBFloat16:
    ddA_to_dtdA_kernel<at::BFloat16><<<ddA_grid_n, dt_threads, dt_shared>>>(
        ddA.data_ptr<float>(), dt_buf.data_ptr<float>(),
        alog_s.data_ptr<at::BFloat16>(), ddt.data_ptr<float>(),
        dA_acc.data_ptr<float>(), batch, seqlen, nheads, chunk_size);
    break;
  default:
    TORCH_CHECK(false, "unsupported dtype for backward dtda");
  }
  auto ddt_raw =
      torch::zeros({batch, seqlen, nheads}, z.options().dtype(torch::kFloat));
  switch (z.scalar_type()) {
  case at::kFloat:
    ddt_raw_kernel<float><<<ddA_blocks_n, threads>>>(
        ddt.data_ptr<float>(), z.data_ptr<float>(), dtb.data_ptr<float>(),
        has_dt_scale ? dt_scale_view.data_ptr<float>() : nullptr,
        ddt_raw.data_ptr<float>(), ddt_bias.data_ptr<float>(), batch, seqlen,
        nheads, d_in_proj, (float)dt_min, (float)dt_max, dt_scale_stride_b,
        dt_scale_stride_t, dt_scale_batch, dt_scale_seqlen, has_dt_scale);
    break;
  case at::kHalf:
    ddt_raw_kernel<at::Half><<<ddA_blocks_n, threads>>>(
        ddt.data_ptr<float>(), z.data_ptr<at::Half>(), dtb.data_ptr<at::Half>(),
        has_dt_scale ? dt_scale_view.data_ptr<at::Half>() : nullptr,
        ddt_raw.data_ptr<float>(), ddt_bias.data_ptr<float>(), batch, seqlen,
        nheads, d_in_proj, (float)dt_min, (float)dt_max, dt_scale_stride_b,
        dt_scale_stride_t, dt_scale_batch, dt_scale_seqlen, has_dt_scale);
    break;
  case at::kBFloat16:
    ddt_raw_kernel<at::BFloat16><<<ddA_blocks_n, threads>>>(
        ddt.data_ptr<float>(), z.data_ptr<at::BFloat16>(),
        dtb.data_ptr<at::BFloat16>(),
        has_dt_scale ? dt_scale_view.data_ptr<at::BFloat16>() : nullptr,
        ddt_raw.data_ptr<float>(), ddt_bias.data_ptr<float>(), batch, seqlen,
        nheads, d_in_proj, (float)dt_min, (float)dt_max, dt_scale_stride_b,
        dt_scale_stride_t, dt_scale_batch, dt_scale_seqlen, has_dt_scale);
    break;
  default:
    TORCH_CHECK(false, "unsupported dtype for backward dt_raw");
  }
  auto dx_skip = torch::zeros({batch, seqlen, nheads, headdim}, gy_f.options());
  d_skip_dx_kernel<<<gy_blocks, threads>>>(
      gy_f.data_ptr<float>(), x_conv_f.data_ptr<float>(), d.data_ptr<float>(),
      dx_skip.data_ptr<float>(), batch, seqlen, nheads, headdim,
      d_has_hdim ? 1 : 0);
  dim3 dD_grid(d_has_hdim ? (nheads * headdim) : nheads, 8);
  d_skip_dD_kernel<<<dD_grid, 256>>>(
      gy_f.data_ptr<float>(), x_conv_f.data_ptr<float>(), dD.data_ptr<float>(),
      batch, seqlen, nheads, headdim, d_has_hdim ? 1 : 0);
  dx_conv_scan.add_(dx_skip);
  auto d_xbc_conv =
      torch::zeros({batch, seqlen, conv_dim}, z.options().dtype(torch::kFloat));
  int64_t xbc_total = batch * seqlen * conv_dim;
  int xbc_blocks = (xbc_total + threads - 1) / threads;
  switch (z.scalar_type()) {
  case at::kFloat:
    scatter_xbc_grad_kernel<float><<<xbc_blocks, threads>>>(
        dx_conv_scan.data_ptr<float>(), b_buf.data_ptr<float>(),
        c_buf.data_ptr<float>(), d_xbc_conv.data_ptr<float>(), batch, seqlen,
        nheads, headdim, d_state, ngroups, d_ssm);
    break;
  default:
    scatter_xbc_grad_kernel<float><<<xbc_blocks, threads>>>(
        dx_conv_scan.data_ptr<float>(),
        b_buf.view({batch * num_chunks * ngroups, chunk_size, d_state})
            .data_ptr<float>(),
        c_buf.view({batch * num_chunks * ngroups, chunk_size, d_state})
            .data_ptr<float>(),
        d_xbc_conv.data_ptr<float>(), batch, seqlen, nheads, headdim, d_state,
        ngroups, d_ssm);
    break;
  }
  auto conv_pre = depthwise_conv1d_pre(z, w, b, d_mlp, d_ssm);
  auto sig = torch::sigmoid(conv_pre);
  auto silu_grad = sig * (1.0f + conv_pre * (1.0f - sig));
  auto d_xbc_pre = d_xbc_conv * silu_grad;
  int64_t offset_xbc = 2 * d_mlp + d_ssm;
  auto xbc = z.narrow(2, offset_xbc, conv_dim).transpose(1, 2).contiguous();
  auto weight = w.view({conv_dim, 1, conv_kernel});
  auto xbc_pad = at::constant_pad_nd(xbc, {conv_kernel - 1, 0});
  auto grad_out = d_xbc_pre.permute({0, 2, 1}).contiguous().to(z.scalar_type());
  auto conv_grads = at::convolution_backward(
      grad_out, xbc_pad, weight,
      b.defined() ? c10::optional<at::IntArrayRef>(b.sizes()) : c10::nullopt,
      {1}, {0}, {1}, false, {0}, conv_dim, {true, true, b.defined()});
  auto d_xbc_in = std::get<0>(conv_grads)
                      .slice(2, conv_kernel - 1, conv_kernel - 1 + seqlen)
                      .transpose(1, 2)
                      .contiguous()
                      .to(torch::kFloat);
  auto d_conv_w = std::get<1>(conv_grads).view_as(w).to(torch::kFloat);
  auto d_conv_b =
      b.defined() ? std::get<2>(conv_grads).to(torch::kFloat)
                  : torch::zeros({conv_dim}, z.options().dtype(torch::kFloat));
  auto dzxbcdt = torch::zeros_like(z, z.options().dtype(torch::kFloat));
  int64_t offset_dt = offset_xbc + d_ssm + 2 * ngroups * d_state;
  dzxbcdt.slice(2, offset_xbc, offset_xbc + conv_dim).copy_(d_xbc_in);
  dzxbcdt.slice(2, offset_dt, offset_dt + nheads).copy_(ddt_raw);
  return {dzxbcdt.to(z.scalar_type()),        d_conv_w.to(conv_w.scalar_type()),
          d_conv_b.to(conv_b.scalar_type()),  ddt_bias.to(dt_bias.scalar_type()),
          dA_acc.to(a_log.scalar_type()),     dD.to(d.scalar_type()),
          dstate0.to(state0.scalar_type())};
}
