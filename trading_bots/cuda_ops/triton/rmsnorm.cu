template <typename scalar_t, typename out_t>
__global__ void rmsnorm_kernel(
    const scalar_t* x,
    const float* weight,
    out_t* out,
    int64_t rows,
    int64_t cols,
    float eps) {
    int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int64_t tid = threadIdx.x;
    int64_t nt = blockDim.x;
    float sum_sq = 0.0f;
    const scalar_t* row_ptr = x + row * cols;
    for (int64_t i = tid; i < cols; i += nt) {
        float v = to_float(row_ptr[i]);
        sum_sq += v * v;
    }
    extern __shared__ float shmem[];
    shmem[tid] = sum_sq;
    __syncthreads();
    for (int offset = nt / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shmem[tid] += shmem[tid + offset];
        }
        __syncthreads();
    }
    float inv_rms = rsqrtf(shmem[0] / cols + eps);
    for (int64_t i = tid; i < cols; i += nt) {
        float v = to_float(row_ptr[i]);
        float w = weight[i];
        out[row * cols + i] = from_float<out_t>(v * inv_rms * w);
    }
}

torch::Tensor rmsnorm_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double eps) {
    auto x_c = x.contiguous();
    auto w_f = weight.to(torch::kFloat).contiguous();
    TORCH_CHECK(x_c.dim() >= 2, "rmsnorm expects at least 2D input");
    int64_t cols = x_c.size(-1);
    int64_t rows = x_c.numel() / cols;
    auto out = torch::empty_like(x_c);
    int threads = 256;
    if (cols > 256) threads = 512;
    if (cols > 512) threads = 1024;
    size_t shmem_size = threads * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x_c.scalar_type(),
        "rmsnorm_forward_cuda", ([&] {
            if (out.scalar_type() == at::ScalarType::Half) {
                rmsnorm_kernel<scalar_t, at::Half>
                    <<<rows, threads, shmem_size>>>(
                        x_c.data_ptr<scalar_t>(),
                        w_f.data_ptr<float>(),
                        reinterpret_cast<at::Half*>(out.data_ptr<at::Half>()),
                        rows,
                        cols,
                        static_cast<float>(eps));
            } else if (out.scalar_type() == at::ScalarType::BFloat16) {
                rmsnorm_kernel<scalar_t, at::BFloat16>
                    <<<rows, threads, shmem_size>>>(
                        x_c.data_ptr<scalar_t>(),
                        w_f.data_ptr<float>(),
                        reinterpret_cast<at::BFloat16*>(out.data_ptr<at::BFloat16>()),
                        rows,
                        cols,
                        static_cast<float>(eps));
            } else {
                rmsnorm_kernel<scalar_t, float>
                    <<<rows, threads, shmem_size>>>(
                        x_c.data_ptr<scalar_t>(),
                        w_f.data_ptr<float>(),
                        out.data_ptr<float>(),
                        rows,
                        cols,
                        static_cast<float>(eps));
            }
        }));
    return out;
}

template <typename scalar_t>
__global__ void rmsnorm_backward_kernel(
    const scalar_t* x,
    const scalar_t* grad_out,
    const float* weight,
    float* grad_w,
    scalar_t* grad_x,
    int64_t rows,
    int64_t cols,
    float eps) {
    int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int64_t tid = threadIdx.x;
    int64_t nt = blockDim.x;
    const scalar_t* x_row = x + row * cols;
    const scalar_t* go_row = grad_out + row * cols;

    float sum_x2 = 0.0f;
    float sum_dyx = 0.0f;
    for (int64_t i = tid; i < cols; i += nt) {
        float xv = to_float(x_row[i]);
        float wv = weight[i];
        float gov = to_float(go_row[i]);
        sum_x2 += xv * xv;
        sum_dyx += (gov * wv) * xv;
    }
    extern __shared__ float shmem[];
    float* sh_sum = shmem;
    float* sh_dyx = shmem + nt;
    sh_sum[tid] = sum_x2;
    sh_dyx[tid] = sum_dyx;
    __syncthreads();
    for (int offset = nt / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sh_sum[tid] += sh_sum[tid + offset];
            sh_dyx[tid] += sh_dyx[tid + offset];
        }
        __syncthreads();
    }
    float inv_rms = rsqrtf(sh_sum[0] / cols + eps);
    float mean_dyx = sh_dyx[0] / cols;
    float inv_rms3 = inv_rms * inv_rms * inv_rms;

    for (int64_t i = tid; i < cols; i += nt) {
        float xv = to_float(x_row[i]);
        float wv = weight[i];
        float gov = to_float(go_row[i]);
        float dxhat = gov * wv;
        float dx = dxhat * inv_rms - xv * inv_rms3 * mean_dyx;
        grad_x[row * cols + i] = from_float<scalar_t>(dx);
        atomicAdd(&grad_w[i], gov * xv * inv_rms);
    }
}

std::vector<torch::Tensor> rmsnorm_backward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& grad_out,
    double eps) {
    auto x_c = x.contiguous();
    auto w_f = weight.to(torch::kFloat).contiguous();
    auto go_c = grad_out.contiguous();
    TORCH_CHECK(x_c.dim() >= 2, "rmsnorm expects at least 2D input");
    int64_t cols = x_c.size(-1);
    int64_t rows = x_c.numel() / cols;
    auto grad_x = torch::empty_like(x_c);
    auto grad_w = torch::zeros({cols}, x_c.options().dtype(torch::kFloat));
    int threads = 256;
    if (cols > 256) threads = 512;
    if (cols > 512) threads = 1024;
    size_t shmem_size = 2 * threads * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x_c.scalar_type(),
        "rmsnorm_backward_cuda", ([&] {
            rmsnorm_backward_kernel<scalar_t>
                <<<rows, threads, shmem_size>>>(
                    x_c.data_ptr<scalar_t>(),
                    go_c.data_ptr<scalar_t>(),
                    w_f.data_ptr<float>(),
                    grad_w.data_ptr<float>(),
                    grad_x.data_ptr<scalar_t>(),
                    rows,
                    cols,
                    static_cast<float>(eps));
        }));
    auto grad_w_cast = grad_w.to(weight.scalar_type());
    return {grad_x, grad_w_cast};
}
