namespace {
__global__ void selective_state_update_kernel(
    const float* state_in,
    const float* x,
    const float* dt,
    const float* a_log,
    const float* b,
    const float* c,
    const float* d_param,
    const float* z,
    const float* dt_bias,
    float* y,
    float* state_out,
    int64_t batch,
    int64_t nheads,
    int64_t headdim,
    int64_t d_state,
    int64_t ngroups,
    int64_t d_has_hdim,
    int64_t has_z,
    int64_t dt_softplus,
    float dt_min,
    float dt_max,
    int64_t apply_dt_limit) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch * nheads * headdim;
    if (idx >= total) {
        return;
    }
    int64_t tmp = idx;
    int64_t p = tmp % headdim;
    tmp /= headdim;
    int64_t h = tmp % nheads;
    int64_t b_idx = tmp / nheads;
    int64_t heads_per_group = nheads / ngroups;
    int64_t group = h / heads_per_group;

    float dt_val = dt[b_idx * nheads + h] + dt_bias[h];
    if (dt_softplus) {
        dt_val = softplus_f(dt_val);
    }
    if (apply_dt_limit) {
        if (dt_val < dt_min || dt_val > dt_max) {
            dt_val = fminf(fmaxf(dt_val, dt_min), dt_max);
        }
    }
    float a = -expf(a_log[h]);
    float da = expf(fminf(dt_val * a, 0.0f));

    float x_val = x[(b_idx * nheads + h) * headdim + p];
    float y_val = 0.0f;
    int64_t state_base = ((b_idx * nheads + h) * headdim + p) * d_state;
    int64_t bc_base = (b_idx * ngroups + group) * d_state;

    if (d_state % 4 == 0) {
        const float4* s_in4 = reinterpret_cast<const float4*>(state_in + state_base);
        float4* s_out4 = reinterpret_cast<float4*>(state_out + state_base);
        const float4* b4 = reinterpret_cast<const float4*>(b + bc_base);
        const float4* c4 = reinterpret_cast<const float4*>(c + bc_base);

        for (int64_t n = 0; n < d_state / 4; ++n) {
            float4 bi = b4[n];
            float4 ci = c4[n];
            float4 si = s_in4[n];
            float4 so;
            so.x = si.x * da + dt_val * bi.x * x_val;
            so.y = si.y * da + dt_val * bi.y * x_val;
            so.z = si.z * da + dt_val * bi.z * x_val;
            so.w = si.w * da + dt_val * bi.w * x_val;
            s_out4[n] = so;
            y_val += so.x * ci.x + so.y * ci.y + so.z * ci.z + so.w * ci.w;
        }
    } else {
        for (int64_t n = 0; n < d_state; ++n) {
            float b_val = b[bc_base + n];
            float c_val = c[bc_base + n];
            float s_in = state_in[state_base + n];
            float s_out = s_in * da + dt_val * b_val * x_val;
            state_out[state_base + n] = s_out;
            y_val += s_out * c_val;
        }
    }

    float d_val = d_has_hdim ? d_param[h * headdim + p] : d_param[h];
    y_val += d_val * x_val;
    if (has_z) {
        float z_val = z[(b_idx * nheads + h) * headdim + p];
        y_val *= silu_f(z_val);
    }
    y[(b_idx * nheads + h) * headdim + p] = y_val;
}

} // namespace
std::tuple<torch::Tensor, torch::Tensor> selective_state_update_cuda(
    const torch::Tensor& state,
    const torch::Tensor& x,
    const torch::Tensor& dt,
    const torch::Tensor& a_log,
    const torch::Tensor& b,
    const torch::Tensor& c,
    const torch::Tensor& d_param,
    const torch::Tensor& z,
    const torch::Tensor& dt_bias,
    bool dt_softplus,
    double dt_min,
    double dt_max,
    int64_t ngroups,
    int64_t headdim,
    int64_t apply_dt_limit) {
    at::NoGradGuard no_grad;
    auto state_f = state.to(torch::kFloat).contiguous();
    auto x_f = x.to(torch::kFloat).contiguous();
    auto dt_f = dt.to(torch::kFloat).contiguous();
    auto a_log_f = a_log.to(torch::kFloat).contiguous();
    auto b_f = b.to(torch::kFloat).contiguous();
    auto c_f = c.to(torch::kFloat).contiguous();
    auto d_f = d_param.to(torch::kFloat).contiguous();
    auto z_f = z.defined() && z.numel() > 0 ? z.to(torch::kFloat).contiguous() : torch::Tensor();
    auto dtb_f = dt_bias.to(torch::kFloat).contiguous();

    int64_t batch = state_f.size(0);
    int64_t nheads = state_f.size(1);
    int64_t d_state = state_f.size(3);
    bool d_has_hdim = d_f.dim() == 2;
    bool has_z = z_f.defined() && z_f.numel() > 0;

    auto y = torch::empty({batch, nheads, headdim}, state_f.options());
    auto state_out = torch::empty_like(state_f);

    int64_t total = batch * nheads * headdim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    selective_state_update_kernel<<<blocks, threads>>>(
        state_f.data_ptr<float>(),
        x_f.data_ptr<float>(),
        dt_f.data_ptr<float>(),
        a_log_f.data_ptr<float>(),
        b_f.data_ptr<float>(),
        c_f.data_ptr<float>(),
        d_f.data_ptr<float>(),
        has_z ? z_f.data_ptr<float>() : nullptr,
        dtb_f.data_ptr<float>(),
        y.data_ptr<float>(),
        state_out.data_ptr<float>(),
        batch,
        nheads,
        headdim,
        d_state,
        ngroups,
        d_has_hdim ? 1 : 0,
        has_z ? 1 : 0,
        dt_softplus ? 1 : 0,
        static_cast<float>(dt_min),
        static_cast<float>(dt_max),
        apply_dt_limit);

    return {y, state_out.to(state.scalar_type())};
}
