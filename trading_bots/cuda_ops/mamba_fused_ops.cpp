#include <torch/library.h>
#include <torch/torch.h>
#include <vector>

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
    double dt_max);

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
    double dt_max);

struct MambaFusedFunction : public torch::autograd::Function<MambaFusedFunction> {
    static std::vector<torch::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
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
        auto outputs = mamba_fused_forward_cuda(
            zxbcdt, conv_w, conv_b, dt_bias, a_log, dt_scale, initial_state,
            chunk_size, ngroups, headdim, dt_min, dt_max);
        ctx->save_for_backward({zxbcdt, conv_w, conv_b, dt_bias, a_log, dt_scale, initial_state, outputs[1]});
        ctx->saved_data["chunk_size"] = chunk_size;
        ctx->saved_data["ngroups"] = ngroups;
        ctx->saved_data["headdim"] = headdim;
        ctx->saved_data["dt_min"] = dt_min;
        ctx->saved_data["dt_max"] = dt_max;
        return outputs;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {
        const auto saved = ctx->get_saved_variables();
        const auto& zxbcdt = saved[0];
        const auto& conv_w = saved[1];
        const auto& conv_b = saved[2];
        const auto& dt_bias = saved[3];
        const auto& a_log = saved[4];
        const auto& dt_scale = saved[5];
        const auto& initial_state = saved[6];
        const auto& final_state = saved[7];

        auto grad_y = grad_outputs.size() > 0 ? grad_outputs[0] : torch::Tensor();
        auto grad_final_state = grad_outputs.size() > 1 ? grad_outputs[1] : torch::Tensor();
        if (!grad_y.defined()) {
            auto batch = zxbcdt.size(0);
            auto seqlen = zxbcdt.size(1);
            auto nheads = a_log.size(0);
            auto headdim = static_cast<int64_t>(ctx->saved_data["headdim"].toInt());
            grad_y = torch::zeros({batch, seqlen, nheads, headdim}, zxbcdt.options());
        }
        if (!grad_final_state.defined()) {
            grad_final_state = torch::zeros_like(final_state);
        }

        auto grads = mamba_fused_backward_cuda(
            zxbcdt, conv_w, conv_b, dt_bias, a_log, dt_scale, initial_state, final_state,
            grad_y, grad_final_state,
            ctx->saved_data["chunk_size"].toInt(),
            ctx->saved_data["ngroups"].toInt(),
            ctx->saved_data["headdim"].toInt(),
            ctx->saved_data["dt_min"].toDouble(),
            ctx->saved_data["dt_max"].toDouble());

        return {
            grads[0],
            grads[1],
            grads[2],
            grads[3],
            grads[4],
            torch::Tensor(),
            grads[5],
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
        };
    }
};

std::tuple<torch::Tensor, torch::Tensor> mamba_fused_conv_scan(
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
    auto outputs = MambaFusedFunction::apply(
        zxbcdt, conv_w, conv_b, dt_bias, a_log, dt_scale, initial_state,
        chunk_size, ngroups, headdim, dt_min, dt_max);
    return {outputs[0], outputs[1]};
}

TORCH_LIBRARY(mamba_fused, m) {
    m.def("fused_conv_scan(Tensor zxbcdt, Tensor conv_w, Tensor conv_b, Tensor dt_bias, Tensor a_log, Tensor dt_scale, Tensor initial_state, int chunk_size, int ngroups, int headdim, float dt_min, float dt_max) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mamba_fused, CUDA, m) {
    m.impl("fused_conv_scan", &mamba_fused_conv_scan);
}
