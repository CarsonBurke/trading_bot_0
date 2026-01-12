#include <torch/library.h>
#include <torch/torch.h>
#include <sstream>
#include <vector>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Optional.h>
#include <ATen/cuda/CUDAGraph.h>
#include <mutex>
#include <unordered_map>

std::vector<torch::Tensor> mamba_fused_forward_cuda(
    const torch::Tensor& zxbcdt,
    const torch::Tensor& conv_w,
    const torch::Tensor& conv_b,
    const torch::Tensor& dt_bias,
    const torch::Tensor& a_log,
    const torch::Tensor& d_param,
    const torch::Tensor& dt_scale,
    const torch::Tensor& initial_state,
    const torch::Tensor& seq_idx,
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
    const torch::Tensor& d_param,
    const torch::Tensor& dt_scale,
    const torch::Tensor& initial_state,
    const torch::Tensor& final_state,
    const torch::Tensor& y,
    const torch::Tensor& grad_y,
    const torch::Tensor& grad_final_state,
    const torch::Tensor& seq_idx,
    int64_t chunk_size,
    int64_t ngroups,
    int64_t headdim,
    double dt_min,
    double dt_max);

std::vector<torch::Tensor> mamba_fused_forward_infer_cuda(
    const torch::Tensor& zxbcdt,
    const torch::Tensor& conv_w,
    const torch::Tensor& conv_b,
    const torch::Tensor& dt_bias,
    const torch::Tensor& a_log,
    const torch::Tensor& d_param,
    const torch::Tensor& dt_scale,
    const torch::Tensor& initial_state,
    const torch::Tensor& seq_idx,
    int64_t chunk_size,
    int64_t ngroups,
    int64_t headdim,
    double dt_min,
    double dt_max,
    const torch::Tensor& rmsnorm_weight,
    double rmsnorm_eps,
    bool norm_before_gate,
    const torch::Tensor& outproj_w,
    const torch::Tensor& outproj_b);

std::vector<torch::Tensor> mamba_fused_forward_stateful_cuda(
    const torch::Tensor& zxbcdt,
    const torch::Tensor& conv_w,
    const torch::Tensor& conv_b,
    const torch::Tensor& dt_bias,
    const torch::Tensor& a_log,
    const torch::Tensor& d_param,
    const torch::Tensor& dt_scale,
    const torch::Tensor& initial_state,
    const torch::Tensor& conv_state,
    const torch::Tensor& seq_idx,
    int64_t chunk_size,
    int64_t ngroups,
    int64_t headdim,
    double dt_min,
    double dt_max);

torch::Tensor mamba_fused_post_ssm(
    const torch::Tensor& y_ssm,
    const torch::Tensor& zxbcdt,
    const torch::Tensor& rms_w,
    int64_t d_mlp,
    int64_t d_ssm,
    int64_t ngroups,
    double eps,
    bool norm_before_gate,
    torch::ScalarType out_type);

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
    int64_t apply_dt_limit);

torch::Tensor rmsnorm_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double eps);
std::vector<torch::Tensor> rmsnorm_backward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& grad_out,
    double eps);

void cuda_empty_cache() {
    c10::cuda::CUDACachingAllocator::emptyCache();
}

std::vector<int64_t> cuda_memory_stats() {
    int device;
    cudaGetDevice(&device);
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    std::vector<int64_t> result;
    // StatType::AGGREGATE is 1 in CUDACachingAllocator? No, in c10::CachingAllocator it is 0?
    // Let's rely on integer indexing if StatType is hard to reach, but better to use correct namespace.
    // Based on Allocator.h, it is c10::CachingAllocator::StatType::AGGREGATE
    using StatType = c10::CachingAllocator::StatType;
    result.push_back(stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current);
    result.push_back(stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current);
    result.push_back(stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)].current);
    result.push_back(stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak);
    result.push_back(stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].peak);
    return result;
}

struct MambaFusedFunction : public torch::autograd::Function<MambaFusedFunction> {
    static std::vector<torch::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& zxbcdt,
        const torch::Tensor& conv_w,
        const torch::Tensor& conv_b,
        const torch::Tensor& dt_bias,
        const torch::Tensor& a_log,
        const torch::Tensor& d_param,
        const torch::Tensor& dt_scale,
        const torch::Tensor& initial_state,
        const torch::Tensor& seq_idx,
        int64_t chunk_size,
        int64_t ngroups,
        int64_t headdim,
        double dt_min,
        double dt_max) {
        auto outputs = mamba_fused_forward_cuda(
            zxbcdt, conv_w, conv_b, dt_bias, a_log, d_param, dt_scale, initial_state, seq_idx,
            chunk_size, ngroups, headdim, dt_min, dt_max);
        ctx->save_for_backward({zxbcdt, conv_w, conv_b, dt_bias, a_log, d_param, dt_scale, initial_state, seq_idx, outputs[1], outputs[0]});
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
        const auto& d_param = saved[5];
        const auto& dt_scale = saved[6];
        const auto& initial_state = saved[7];
        const auto& seq_idx = saved[8];
        const auto& final_state = saved[9];
        const auto& y = saved[10];

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
            zxbcdt, conv_w, conv_b, dt_bias, a_log, d_param, dt_scale, initial_state, final_state, y,
            grad_y, grad_final_state, seq_idx,
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
            grads[5],
            torch::Tensor(),
            grads[6],
            torch::Tensor(),
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
    const torch::Tensor& d_param,
    const torch::Tensor& dt_scale,
    const torch::Tensor& initial_state,
    const torch::Tensor& seq_idx,
    int64_t chunk_size,
    int64_t ngroups,
    int64_t headdim,
    double dt_min,
    double dt_max) {
    auto outputs = MambaFusedFunction::apply(
        zxbcdt, conv_w, conv_b, dt_bias, a_log, d_param, dt_scale, initial_state, seq_idx,
        chunk_size, ngroups, headdim, dt_min, dt_max);
    return {outputs[0], outputs[1]};
}

std::tuple<torch::Tensor, torch::Tensor> mamba_fused_conv_scan_infer(
    const torch::Tensor& zxbcdt,
    const torch::Tensor& conv_w,
    const torch::Tensor& conv_b,
    const torch::Tensor& dt_bias,
    const torch::Tensor& a_log,
    const torch::Tensor& d_param,
    const torch::Tensor& dt_scale,
    const torch::Tensor& initial_state,
    const torch::Tensor& seq_idx,
    int64_t chunk_size,
    int64_t ngroups,
    int64_t headdim,
    double dt_min,
    double dt_max,
    const torch::Tensor& rmsnorm_weight,
    double rmsnorm_eps,
    bool norm_before_gate,
    const torch::Tensor& outproj_w,
    const torch::Tensor& outproj_b) {
    auto outputs = mamba_fused_forward_infer_cuda(
        zxbcdt,
        conv_w,
        conv_b,
        dt_bias,
        a_log,
        d_param,
        dt_scale,
        initial_state,
        seq_idx,
        chunk_size,
        ngroups,
        headdim,
        dt_min,
        dt_max,
        rmsnorm_weight,
        rmsnorm_eps,
        norm_before_gate,
        outproj_w,
        outproj_b);
    return {outputs[0], outputs[1]};
}

std::tuple<torch::Tensor, torch::Tensor> mamba_fused_conv_scan_full(
    const torch::Tensor& zxbcdt,
    const torch::Tensor& conv_w,
    const torch::Tensor& conv_b,
    const torch::Tensor& dt_bias,
    const torch::Tensor& a_log,
    const torch::Tensor& d_param,
    const torch::Tensor& dt_scale,
    const torch::Tensor& initial_state,
    const torch::Tensor& seq_idx,
    int64_t chunk_size,
    int64_t ngroups,
    int64_t headdim,
    double dt_min,
    double dt_max,
    const torch::Tensor& rmsnorm_weight,
    double rmsnorm_eps,
    bool norm_before_gate,
    const torch::Tensor& outproj_w,
    const torch::Tensor& outproj_b) {
    auto outputs = mamba_fused_conv_scan(
        zxbcdt,
        conv_w,
        conv_b,
        dt_bias,
        a_log,
        d_param,
        dt_scale,
        initial_state,
        seq_idx,
        chunk_size,
        ngroups,
        headdim,
        dt_min,
        dt_max);
    auto y = std::get<0>(outputs);
    auto final_state = std::get<1>(outputs);

    int64_t nheads = y.size(2);
    int64_t d_ssm = nheads * headdim;
    int64_t d_in_proj = zxbcdt.size(2);
    int64_t d_state = (conv_w.size(0) - d_ssm) / (2 * ngroups);
    int64_t d_inner = (d_in_proj - 2 * ngroups * d_state - nheads) / 2;
    int64_t d_mlp = d_inner - d_ssm;

    auto y_out = mamba_fused_post_ssm(
        y.view({y.size(0), y.size(1), d_ssm}),
        zxbcdt,
        rmsnorm_weight,
        d_mlp,
        d_ssm,
        ngroups,
        rmsnorm_eps,
        norm_before_gate,
        outproj_w.defined() ? outproj_w.scalar_type() : zxbcdt.scalar_type());

    torch::Tensor out;
    if (outproj_w.defined() && outproj_w.numel() > 0) {
        auto out_in = y_out;
        auto out_2d = out_in.view({y.size(0) * y.size(1), d_inner});
        c10::optional<torch::Tensor> bias =
            (outproj_b.defined() && outproj_b.numel() > 0)
                ? c10::optional<torch::Tensor>(outproj_b)
                : c10::nullopt;
        out = at::linear(out_2d, outproj_w, bias);
        out = out.view({y.size(0), y.size(1), outproj_w.size(0)});
    } else {
        out = y_out;
    }

    return {out, final_state};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mamba_fused_conv_scan_stateful(
    const torch::Tensor& zxbcdt,
    const torch::Tensor& conv_w,
    const torch::Tensor& conv_b,
    const torch::Tensor& dt_bias,
    const torch::Tensor& a_log,
    const torch::Tensor& d_param,
    const torch::Tensor& dt_scale,
    const torch::Tensor& initial_state,
    const torch::Tensor& conv_state,
    const torch::Tensor& seq_idx,
    int64_t chunk_size,
    int64_t ngroups,
    int64_t headdim,
    double dt_min,
    double dt_max,
    const torch::Tensor& rmsnorm_weight,
    double rmsnorm_eps,
    bool norm_before_gate,
    const torch::Tensor& outproj_w,
    const torch::Tensor& outproj_b) {
    auto outputs = mamba_fused_forward_stateful_cuda(
        zxbcdt,
        conv_w,
        conv_b,
        dt_bias,
        a_log,
        d_param,
        dt_scale,
        initial_state,
        conv_state,
        seq_idx,
        chunk_size,
        ngroups,
        headdim,
        dt_min,
        dt_max);
    auto y = outputs[0];
    auto final_state = outputs[1];
    auto conv_state_out = outputs[2];

    int64_t nheads = y.size(2);
    int64_t d_ssm = nheads * headdim;
    int64_t d_in_proj = zxbcdt.size(2);
    int64_t d_state = (conv_w.size(0) - d_ssm) / (2 * ngroups);
    int64_t d_inner = (d_in_proj - 2 * ngroups * d_state - nheads) / 2;
    int64_t d_mlp = d_inner - d_ssm;

    auto y_out = mamba_fused_post_ssm(
        y.view({y.size(0), y.size(1), d_ssm}),
        zxbcdt,
        rmsnorm_weight,
        d_mlp,
        d_ssm,
        ngroups,
        rmsnorm_eps,
        norm_before_gate,
        outproj_w.defined() ? outproj_w.scalar_type() : zxbcdt.scalar_type());

    torch::Tensor out;
    if (outproj_w.defined() && outproj_w.numel() > 0) {
        auto out_in = y_out;
        auto out_2d = out_in.view({y.size(0) * y.size(1), d_inner});
        auto out_w_t = outproj_w.t().contiguous();
        if (outproj_b.defined() && outproj_b.numel() > 0) {
            out = at::addmm(outproj_b, out_2d, out_w_t);
        } else {
            out = out_2d.matmul(out_w_t);
        }
        out = out.view({y.size(0), y.size(1), outproj_w.size(0)});
    } else {
        out = y_out;
    }

    return {out, final_state, conv_state_out};
}

std::tuple<torch::Tensor, torch::Tensor> selective_state_update(
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
    return selective_state_update_cuda(
        state,
        x,
        dt,
        a_log,
        b,
        c,
        d_param,
        z,
        dt_bias,
        dt_softplus,
        dt_min,
        dt_max,
        ngroups,
        headdim,
        apply_dt_limit);
}

torch::Tensor rmsnorm_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double eps) {
    return rmsnorm_forward_cuda(x, weight, eps);
}

struct RMSNormFunction : public torch::autograd::Function<RMSNormFunction> {
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& x,
        const torch::Tensor& weight,
        double eps) {
        auto y = rmsnorm_forward_cuda(x, weight, eps);
        ctx->save_for_backward({x, weight});
        ctx->saved_data["eps"] = eps;
        return y;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs) {
        const auto saved = ctx->get_saved_variables();
        const auto& x = saved[0];
        const auto& weight = saved[1];
        auto grad_out = grad_outputs.size() > 0 ? grad_outputs[0] : torch::Tensor();
        if (!grad_out.defined()) {
            grad_out = torch::zeros_like(x);
        }
        auto grads = rmsnorm_backward_cuda(
            x,
            weight,
            grad_out,
            ctx->saved_data["eps"].toDouble());
        return {
            grads[0],
            grads[1],
            torch::Tensor(),
        };
    }
};

torch::Tensor rmsnorm_forward_autograd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double eps) {
    return RMSNormFunction::apply(x, weight, eps);
}

struct GraphCacheEntry {
    at::cuda::CUDAGraph graph;
    torch::Tensor zxbcdt;
    torch::Tensor conv_w;
    torch::Tensor conv_b;
    torch::Tensor dt_bias;
    torch::Tensor a_log;
    torch::Tensor d_param;
    torch::Tensor dt_scale;
    torch::Tensor initial_state;
    torch::Tensor seq_idx;
    torch::Tensor rmsnorm_weight;
    torch::Tensor outproj_w;
    torch::Tensor outproj_b;
    torch::Tensor out;
    torch::Tensor final_state;
    bool captured = false;
};

std::mutex GRAPH_MUTEX;
std::unordered_map<std::string, GraphCacheEntry> GRAPH_CACHE;

std::string graph_key(
    const torch::Tensor& zxbcdt,
    const torch::Tensor& conv_w,
    const torch::Tensor& dt_bias,
    const torch::Tensor& a_log,
    int64_t chunk_size,
    int64_t ngroups,
    int64_t headdim) {
    std::ostringstream oss;
    oss << zxbcdt.device().str() << ":";
    oss << zxbcdt.scalar_type() << ":";
    for (auto s : zxbcdt.sizes()) oss << s << ",";
    oss << "|w=";
    for (auto s : conv_w.sizes()) oss << s << ",";
    oss << "|dt=";
    for (auto s : dt_bias.sizes()) oss << s << ",";
    oss << "|a=";
    for (auto s : a_log.sizes()) oss << s << ",";
    oss << "|c=" << chunk_size << "|g=" << ngroups << "|h=" << headdim;
    return oss.str();
}

std::tuple<torch::Tensor, torch::Tensor> mamba_fused_conv_scan_full_graph(
    const torch::Tensor& zxbcdt,
    const torch::Tensor& conv_w,
    const torch::Tensor& conv_b,
    const torch::Tensor& dt_bias,
    const torch::Tensor& a_log,
    const torch::Tensor& d_param,
    const torch::Tensor& dt_scale,
    const torch::Tensor& initial_state,
    const torch::Tensor& seq_idx,
    int64_t chunk_size,
    int64_t ngroups,
    int64_t headdim,
    double dt_min,
    double dt_max,
    const torch::Tensor& rmsnorm_weight,
    double rmsnorm_eps,
    bool norm_before_gate,
    const torch::Tensor& outproj_w,
    const torch::Tensor& outproj_b) {
    auto key = graph_key(zxbcdt, conv_w, dt_bias, a_log, chunk_size, ngroups, headdim);
    std::lock_guard<std::mutex> lock(GRAPH_MUTEX);
    auto& entry = GRAPH_CACHE[key];
    if (!entry.captured) {
        entry.zxbcdt = torch::empty_like(zxbcdt);
        entry.conv_w = torch::empty_like(conv_w);
        entry.conv_b = conv_b.defined() ? torch::empty_like(conv_b) : torch::Tensor();
        entry.dt_bias = torch::empty_like(dt_bias);
        entry.a_log = torch::empty_like(a_log);
        entry.d_param = torch::empty_like(d_param);
        entry.dt_scale = dt_scale.defined() ? torch::empty_like(dt_scale) : torch::Tensor();
        entry.initial_state = torch::empty_like(initial_state);
        entry.seq_idx = seq_idx.defined() ? torch::empty_like(seq_idx) : torch::Tensor();
        entry.rmsnorm_weight = rmsnorm_weight.defined() ? torch::empty_like(rmsnorm_weight) : torch::Tensor();
        entry.outproj_w = outproj_w.defined() ? torch::empty_like(outproj_w) : torch::Tensor();
        entry.outproj_b = outproj_b.defined() ? torch::empty_like(outproj_b) : torch::Tensor();

        entry.zxbcdt.copy_(zxbcdt);
        entry.conv_w.copy_(conv_w);
        if (conv_b.defined()) entry.conv_b.copy_(conv_b);
        entry.dt_bias.copy_(dt_bias);
        entry.a_log.copy_(a_log);
        entry.d_param.copy_(d_param);
        if (dt_scale.defined()) entry.dt_scale.copy_(dt_scale);
        entry.initial_state.copy_(initial_state);
        if (seq_idx.defined()) entry.seq_idx.copy_(seq_idx);
        if (rmsnorm_weight.defined()) entry.rmsnorm_weight.copy_(rmsnorm_weight);
        if (outproj_w.defined()) entry.outproj_w.copy_(outproj_w);
        if (outproj_b.defined()) entry.outproj_b.copy_(outproj_b);

        entry.graph.capture_begin();
        auto outputs = mamba_fused_conv_scan_full(
            entry.zxbcdt,
            entry.conv_w,
            entry.conv_b,
            entry.dt_bias,
            entry.a_log,
            entry.d_param,
            entry.dt_scale,
            entry.initial_state,
            entry.seq_idx,
            chunk_size,
            ngroups,
            headdim,
            dt_min,
            dt_max,
            entry.rmsnorm_weight,
            rmsnorm_eps,
            norm_before_gate,
            entry.outproj_w,
            entry.outproj_b);
        entry.graph.capture_end();
        entry.out = std::get<0>(outputs);
        entry.final_state = std::get<1>(outputs);
        entry.captured = true;
    } else {
        entry.zxbcdt.copy_(zxbcdt);
        entry.conv_w.copy_(conv_w);
        if (conv_b.defined()) entry.conv_b.copy_(conv_b);
        entry.dt_bias.copy_(dt_bias);
        entry.a_log.copy_(a_log);
        entry.d_param.copy_(d_param);
        if (dt_scale.defined()) entry.dt_scale.copy_(dt_scale);
        entry.initial_state.copy_(initial_state);
        if (seq_idx.defined()) entry.seq_idx.copy_(seq_idx);
        if (rmsnorm_weight.defined()) entry.rmsnorm_weight.copy_(rmsnorm_weight);
        if (outproj_w.defined()) entry.outproj_w.copy_(outproj_w);
        if (outproj_b.defined()) entry.outproj_b.copy_(outproj_b);
        entry.graph.replay();
    }
    return {entry.out, entry.final_state};
}

TORCH_LIBRARY(mamba_fused, m) {
    m.def("fused_conv_scan(Tensor zxbcdt, Tensor conv_w, Tensor conv_b, Tensor dt_bias, Tensor a_log, Tensor d_param, Tensor dt_scale, Tensor initial_state, Tensor seq_idx, int chunk_size, int ngroups, int headdim, float dt_min, float dt_max) -> (Tensor, Tensor)");
    m.def("fused_conv_scan_infer(Tensor zxbcdt, Tensor conv_w, Tensor conv_b, Tensor dt_bias, Tensor a_log, Tensor d_param, Tensor dt_scale, Tensor initial_state, Tensor seq_idx, int chunk_size, int ngroups, int headdim, float dt_min, float dt_max, Tensor rmsnorm_weight, float rmsnorm_eps, bool norm_before_gate, Tensor outproj_w, Tensor outproj_b) -> (Tensor, Tensor)");
    m.def("fused_conv_scan_full(Tensor zxbcdt, Tensor conv_w, Tensor conv_b, Tensor dt_bias, Tensor a_log, Tensor d_param, Tensor dt_scale, Tensor initial_state, Tensor seq_idx, int chunk_size, int ngroups, int headdim, float dt_min, float dt_max, Tensor rmsnorm_weight, float rmsnorm_eps, bool norm_before_gate, Tensor outproj_w, Tensor outproj_b) -> (Tensor, Tensor)");
    m.def("fused_conv_scan_full_graph(Tensor zxbcdt, Tensor conv_w, Tensor conv_b, Tensor dt_bias, Tensor a_log, Tensor d_param, Tensor dt_scale, Tensor initial_state, Tensor seq_idx, int chunk_size, int ngroups, int headdim, float dt_min, float dt_max, Tensor rmsnorm_weight, float rmsnorm_eps, bool norm_before_gate, Tensor outproj_w, Tensor outproj_b) -> (Tensor, Tensor)");
    m.def("fused_conv_scan_stateful(Tensor zxbcdt, Tensor conv_w, Tensor conv_b, Tensor dt_bias, Tensor a_log, Tensor d_param, Tensor dt_scale, Tensor initial_state, Tensor conv_state, Tensor seq_idx, int chunk_size, int ngroups, int headdim, float dt_min, float dt_max, Tensor rmsnorm_weight, float rmsnorm_eps, bool norm_before_gate, Tensor outproj_w, Tensor outproj_b) -> (Tensor, Tensor, Tensor)");
    m.def("selective_state_update(Tensor state, Tensor x, Tensor dt, Tensor a_log, Tensor b, Tensor c, Tensor d_param, Tensor z, Tensor dt_bias, bool dt_softplus, float dt_min, float dt_max, int ngroups, int headdim, int apply_dt_limit) -> (Tensor, Tensor)");
    m.def("rmsnorm_forward(Tensor x, Tensor weight, float eps) -> Tensor");
    m.def("cuda_empty_cache", &cuda_empty_cache);
    m.def("cuda_memory_stats", &cuda_memory_stats);
}

TORCH_LIBRARY_IMPL(mamba_fused, CUDA, m) {
    m.impl("fused_conv_scan", &mamba_fused_conv_scan);
    m.impl("fused_conv_scan_infer", &mamba_fused_conv_scan_infer);
    m.impl("fused_conv_scan_stateful", &mamba_fused_conv_scan_stateful);
    m.impl("selective_state_update", &selective_state_update);
    m.impl("rmsnorm_forward", &rmsnorm_forward);
    m.impl("fused_conv_scan_full_graph", &mamba_fused_conv_scan_full_graph);
}

TORCH_LIBRARY_IMPL(mamba_fused, Autograd, m) {
    m.impl("fused_conv_scan", &mamba_fused_conv_scan);
    m.impl("rmsnorm_forward", &rmsnorm_forward_autograd);
}

TORCH_LIBRARY_IMPL(mamba_fused, CompositeImplicitAutograd, m) {
    m.impl("fused_conv_scan_full", &mamba_fused_conv_scan_full);
}
