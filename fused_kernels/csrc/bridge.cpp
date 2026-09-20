// libtorch side of the fused kernels: shape and dtype contract, output allocation,
// current-stream discovery, and the autograd registration.
//
// AUTOGRAD MECHANISM. Each fusion is a `torch::autograd::Function` subclass whose
// `apply` is what the C ABI calls. That is the mechanism libtorch itself uses for ops
// written outside the dispatcher, and it is the only one that works here: the alternative
// (a dispatcher op plus a `derivatives.yaml` entry) needs codegen we do not run, and the
// third option people reach for - composing the forward out of differentiable ATen calls -
// is precisely the cost being removed. `Function::apply` builds a real autograd node, so
// the tensor handed back to Rust carries a `grad_fn` and `Tensor::backward` /
// `Tensor::run_backward` from Rust drive these kernels with no further plumbing.
//
// Soundness: `apply` runs `forward` with grad mode off and records the node itself, so the
// output's version counter, `requires_grad` and graph edges are libtorch's, not ours. The
// backward is first-order only; it is written with raw kernels, so it creates no graph, and
// a second-order call would silently see a zero. That is checked for and rejected rather
// than tolerated - the backbone never takes a second derivative.

#include "kernels.h"

#include <torch/autograd.h>
#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <exception>
#include <string>
#include <vector>

namespace {

thread_local std::string last_error;

// The `[tokens, ffn]` and `[tokens, 2*d_model]` activations are far too big for anything
// but bf16, and a silent fp32 path would be a performance bug that still passes a
// numerical test, so the dtype is a contract rather than a dispatch.
void check_bf16_cuda(const torch::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.defined(), name, " is undefined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor, found ",
                tensor.device());
    TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name,
                " must be bfloat16, found ", tensor.scalar_type());
}

void check_launch(int status, const char *name) {
    TORCH_CHECK(status == 0, name, " launch failed: ",
                cudaGetErrorString(static_cast<cudaError_t>(status)));
}

void *current_stream(const torch::Tensor &tensor) {
    return static_cast<void *>(
        at::cuda::getCurrentCUDAStream(tensor.device().index()).stream());
}

void reject_double_backward(const torch::autograd::variable_list &grads,
                            const char *name) {
    for (const auto &grad : grads) {
        TORCH_CHECK(!grad.defined() || !grad.requires_grad(), name,
                    " has no second derivative: its backward is a raw kernel and would "
                    "report a zero double-gradient instead of failing");
    }
}

// The backward launch, shared by the autograd node and by the raw entry point the
// microbenchmark times. Timing the kernel through autograd would charge it the objective's
// own reduction and cast, which at the FFN shape is several milliseconds of traffic that
// has nothing to do with this kernel.
torch::Tensor relu_square_backward(const torch::Tensor &input, const torch::Tensor &grad_output) {
    const torch::Tensor grad = grad_output.contiguous();
    check_bf16_cuda(input, "relu_square backward input");
    check_bf16_cuda(grad, "relu_square gradient");
    TORCH_CHECK(input.is_contiguous(), "relu_square backward input must be contiguous");
    TORCH_CHECK(grad.numel() == input.numel(), "relu_square gradient has ", grad.numel(),
                " elements but the input had ", input.numel());
    torch::Tensor dx = at::empty_like(input, at::MemoryFormat::Contiguous);
    check_launch(fk_relu_square_backward(input.const_data_ptr(), grad.const_data_ptr(),
                                         dx.mutable_data_ptr(), input.numel(),
                                         current_stream(input)),
                 "relu_square backward");
    return dx;
}

struct ReluSquare : public torch::autograd::Function<ReluSquare> {
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor input) {
        check_bf16_cuda(input, "relu_square input");
        TORCH_CHECK(input.is_contiguous(), "relu_square input must be contiguous");
        torch::Tensor output = at::empty_like(input, at::MemoryFormat::Contiguous);
        // The INPUT is saved, not `relu(input)`. Both are one [tokens, ffn] bf16 tensor,
        // which is what the ATen composition retains too (`square` saves `relu`'s
        // output), so memory is unchanged - but saving `relu(input)` would force the
        // forward to write a second full-width tensor, +393 MiB of traffic per layer at
        // batch 256, for nothing: `threshold_backward`'s mask `NOT (relu(x) <= 0)` and the
        // factor `relu(x)` are both recoverable from `x` exactly, because that mask equals
        // `NOT (x <= 0)` and `relu(x) == x` wherever it holds.
        ctx->save_for_backward({input});
        check_launch(fk_relu_square_forward(input.const_data_ptr(),
                                           output.mutable_data_ptr(), input.numel(),
                                           current_stream(input)),
                     "relu_square forward");
        return output;
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_outputs) {
        reject_double_backward(grad_outputs, "relu_square");
        return {relu_square_backward(ctx->get_saved_variables()[0], grad_outputs[0])};
    }
};

// Row addressing shared by the rotary forward and backward: `[batch, length, columns]`
// with the last dimension contiguous and a uniform row stride, which is what a
// `split_with_sizes` view of the packed QKV projection is.
struct RowLayout {
    int64_t rows;
    int64_t length;
    int64_t stride;
};

RowLayout row_layout(const torch::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.dim() >= 3, name, " must be [batch, length, ...], found ",
                tensor.dim(), " dimensions");
    const int64_t batch = tensor.size(0);
    const int64_t length = tensor.size(1);
    // Trailing dimensions must be dense so a row is one contiguous run, and the batch
    // stride must be exactly `length` rows so `batch*length` is a single flat row index.
    int64_t dense = 1;
    for (int64_t dimension = tensor.dim() - 1; dimension >= 2; --dimension) {
        TORCH_CHECK(tensor.stride(dimension) == dense, name,
                    " needs dense trailing dimensions; dimension ", dimension,
                    " has stride ", tensor.stride(dimension), " not ", dense);
        dense *= tensor.size(dimension);
    }
    const int64_t stride = tensor.stride(1);
    TORCH_CHECK(stride >= dense, name, " row stride ", stride,
                " is smaller than its ", dense, " columns");
    TORCH_CHECK(batch == 1 || tensor.stride(0) == length * stride, name,
                " batch stride ", tensor.stride(0), " is not ", length, " rows of ",
                stride);
    return {batch * length, length, stride};
}

// The rotary backward launch, shared by the autograd node and by the raw entry point the
// microbenchmark times. `grad_output` is `[batch, length, 2, heads, head_dim]`; the result is
// the dense `[batch, length, 2*heads*head_dim]` gradient of the packed block.
torch::Tensor rope_backward(const torch::Tensor &grad_output, const torch::Tensor &cosine,
                            const torch::Tensor &sine, int64_t heads) {
    const torch::Tensor grad = grad_output.contiguous();
    check_bf16_cuda(grad, "rope gradient");
    check_bf16_cuda(cosine, "rope cosine");
    check_bf16_cuda(sine, "rope sine");
    TORCH_CHECK(heads > 0, "rope needs a positive head count, found ", heads);
    const RowLayout layout = row_layout(grad, "rope gradient");
    const int64_t blocks = 2 * heads;
    const int64_t half = cosine.size(1);
    const int64_t columns = blocks * 2 * half;
    TORCH_CHECK(grad.numel() == layout.rows * columns, "rope gradient has ", grad.numel(),
                " elements but ", heads, " heads of ", 2 * half, " imply ",
                layout.rows * columns);
    torch::Tensor dx = at::empty({grad.size(0), layout.length, columns}, grad.options());
    check_launch(fk_rope_backward(grad.const_data_ptr(), cosine.const_data_ptr(),
                                  sine.const_data_ptr(), dx.mutable_data_ptr(), layout.rows,
                                  layout.length, blocks, half, layout.stride, columns,
                                  current_stream(grad)),
                 "rope backward");
    return dx;
}

struct Rope : public torch::autograd::Function<Rope> {
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor input, torch::Tensor cosine,
                                 torch::Tensor sine, int64_t heads) {
        check_bf16_cuda(input, "rope input");
        check_bf16_cuda(cosine, "rope cosine");
        check_bf16_cuda(sine, "rope sine");
        TORCH_CHECK(input.dim() == 3, "rope input must be [batch, length, 2*d_model]");
        TORCH_CHECK(heads > 0, "rope needs a positive head count, found ", heads);
        const RowLayout layout = row_layout(input, "rope input");
        const int64_t columns = input.size(2);
        const int64_t blocks = 2 * heads;
        TORCH_CHECK(columns % (blocks * 2) == 0, "rope input has ", columns,
                    " columns, not a multiple of ", blocks * 2);
        const int64_t head_dim = columns / blocks;
        const int64_t half = head_dim / 2;
        TORCH_CHECK(head_dim % 2 == 0, "rope needs an even head dimension, found ",
                    head_dim);
        TORCH_CHECK(cosine.dim() == 2 && cosine.size(0) == layout.length &&
                        cosine.size(1) == half,
                    "rope cosine must be [", layout.length, ", ", half, "], found ",
                    cosine.sizes());
        TORCH_CHECK(sine.sizes() == cosine.sizes(),
                    "rope sine and cosine must have the same shape");
        TORCH_CHECK(cosine.is_contiguous() && sine.is_contiguous(),
                    "rope cosine and sine must be contiguous");
        TORCH_CHECK(!cosine.requires_grad() && !sine.requires_grad(),
                    "rope treats the rotation as a constant; cosine and sine must not "
                    "require gradients");

        torch::Tensor output =
            at::empty({input.size(0), layout.length, 2, heads, head_dim},
                      input.options());
        // Nothing about the INPUT is saved: the rotation is orthogonal, so its transpose
        // needs only the rotation itself. `cosine`/`sine` are the model's resident
        // [length, head_dim/2] constants - 24 KiB each at the real geometry - so the
        // backward of this op retains no activation at all.
        ctx->save_for_backward({cosine, sine});
        ctx->saved_data["heads"] = heads;
        ctx->saved_data["columns"] = columns;
        check_launch(fk_rope_forward(input.const_data_ptr(), cosine.const_data_ptr(),
                                     sine.const_data_ptr(), output.mutable_data_ptr(),
                                     layout.rows, layout.length, blocks, half,
                                     layout.stride, columns, current_stream(input)),
                     "rope forward");
        return output;
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_outputs) {
        reject_double_backward(grad_outputs, "rope");
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        return {rope_backward(grad_outputs[0], saved[0], saved[1],
                              ctx->saved_data["heads"].toInt()),
                torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

// The RMSNorm epsilon is BAKED IN rather than passed. `_fused_rms_norm` with `eps=None`
// resolves to `finfo(bf16).eps = 7.8e-3`, a 0.4% systematic shrink of every normalized
// activation, and the backbone deliberately passes 1e-6 (`world_model.rs`'s `BAR_NORM_EPS`,
// and the reference's own `train_gpt.py:1079`). A call site that could pass its own epsilon
// is a call site that can silently stop matching the composition it replaced, and this op
// exists for exactly one call site.
constexpr float kNormEps = 1e-6f;

// Geometry shared by the QK-norm rotary's forward and backward: one head block is
// simultaneously the normalization group and the rotary block.
struct HeadGeometry {
    int64_t blocks;
    int64_t head_dim;
    int64_t half;
    int64_t columns;
};

HeadGeometry head_geometry(int64_t columns, int64_t heads, int64_t rotary_half,
                           const char *name) {
    TORCH_CHECK(heads > 0, name, " needs a positive head count, found ", heads);
    const int64_t blocks = 2 * heads;
    TORCH_CHECK(columns % blocks == 0, name, " has ", columns,
                " columns, not a multiple of ", blocks);
    const int64_t head_dim = columns / blocks;
    TORCH_CHECK(head_dim == 2 * rotary_half, name, " has head dimension ", head_dim,
                " but the rotation carries ", rotary_half, " pairs");
    // ATen's own bf16 RMSNorm forward is `vectorized_layer_norm_kernel` only while
    // `head_dim % 4 == 0`; below that it switches to a Welford reduction with a different
    // summation order, and at more than 128 elements its partials stop fitting the first
    // warp of the tree this kernel reproduces. Both would break bit-identity silently, so
    // they are refused loudly.
    TORCH_CHECK(head_dim % 4 == 0 && head_dim <= 128, name, " needs a head dimension that "
                "is a multiple of four and at most 128 for the reduction order to match "
                "ATen's, found ", head_dim);
    return {blocks, head_dim, rotary_half, columns};
}

void check_rotation(const torch::Tensor &cosine, const torch::Tensor &sine, int64_t length,
                    const char *name) {
    check_bf16_cuda(cosine, "qk_norm_rope cosine");
    check_bf16_cuda(sine, "qk_norm_rope sine");
    TORCH_CHECK(cosine.dim() == 2 && cosine.size(0) == length, name,
                " cosine must be [", length, ", head_dim/2], found ", cosine.sizes());
    TORCH_CHECK(sine.sizes() == cosine.sizes(),
                "qk_norm_rope sine and cosine must have the same shape");
    TORCH_CHECK(cosine.is_contiguous() && sine.is_contiguous(),
                "qk_norm_rope cosine and sine must be contiguous");
    TORCH_CHECK(!cosine.requires_grad() && !sine.requires_grad(),
                "qk_norm_rope treats the rotation as a constant; cosine and sine must not "
                "require gradients");
}

// The fused backward launch, shared by the autograd node and by the raw entry point the
// microbenchmark times. `grad_output` is `[batch, length, 2, heads, head_dim]` and `input`
// is the RAW packed block the forward received, possibly a strided view; the result is the
// dense `[batch, length, 2*heads*head_dim]` gradient of that block.
torch::Tensor qk_norm_rope_backward(const torch::Tensor &grad_output,
                                    const torch::Tensor &input,
                                    const torch::Tensor &cosine, const torch::Tensor &sine,
                                    int64_t heads, int rounding) {
    const torch::Tensor grad = grad_output.contiguous();
    check_bf16_cuda(grad, "qk_norm_rope gradient");
    check_bf16_cuda(input, "qk_norm_rope backward input");
    const RowLayout layout = row_layout(input, "qk_norm_rope backward input");
    check_rotation(cosine, sine, layout.length, "qk_norm_rope");
    const HeadGeometry geometry =
        head_geometry(input.size(2), heads, cosine.size(1), "qk_norm_rope gradient");
    TORCH_CHECK(grad.numel() == layout.rows * geometry.columns, "qk_norm_rope gradient has ",
                grad.numel(), " elements but the input block had ",
                layout.rows * geometry.columns);
    torch::Tensor dx =
        at::empty({input.size(0), layout.length, geometry.columns}, input.options());
    check_launch(fk_qk_norm_rope_backward(
                     grad.const_data_ptr(), input.const_data_ptr(),
                     cosine.const_data_ptr(), sine.const_data_ptr(), dx.mutable_data_ptr(),
                     layout.rows, layout.length, geometry.blocks, geometry.half,
                     geometry.columns, layout.stride, geometry.columns,
                     static_cast<float>(geometry.head_dim), kNormEps, rounding,
                     current_stream(input)),
                 "qk_norm_rope backward");
    return dx;
}

struct QkNormRope : public torch::autograd::Function<QkNormRope> {
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor input, torch::Tensor cosine,
                                 torch::Tensor sine, int64_t heads, int64_t rounding) {
        check_bf16_cuda(input, "qk_norm_rope input");
        TORCH_CHECK(input.dim() == 3,
                    "qk_norm_rope input must be [batch, length, 2*d_model]");
        const RowLayout layout = row_layout(input, "qk_norm_rope input");
        check_rotation(cosine, sine, layout.length, "qk_norm_rope");
        const HeadGeometry geometry =
            head_geometry(input.size(2), heads, cosine.size(1), "qk_norm_rope input");

        torch::Tensor output =
            at::empty({input.size(0), layout.length, 2, heads, geometry.head_dim},
                      input.options());
        // The RAW block is saved and `rstd` is NOT. The composition retains an fp32
        // `[tokens, 2*heads]` statistic per layer and, in the forward, a contiguous copy of
        // this very block plus the normalized block itself; here the backward re-derives
        // the statistic from `input`, which it has to stream anyway for the `x*rstd*stats`
        // term, so the recompute is a reduction over `head_dim` values already in registers.
        ctx->save_for_backward({input, cosine, sine});
        ctx->saved_data["heads"] = heads;
        ctx->saved_data["rounding"] = rounding;
        check_launch(fk_qk_norm_rope_forward(
                         input.const_data_ptr(), cosine.const_data_ptr(),
                         sine.const_data_ptr(), output.mutable_data_ptr(), layout.rows,
                         layout.length, geometry.blocks, geometry.half, layout.stride,
                         geometry.columns, static_cast<float>(geometry.head_dim), kNormEps,
                         static_cast<int>(rounding), current_stream(input)),
                     "qk_norm_rope forward");
        return output;
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_outputs) {
        reject_double_backward(grad_outputs, "qk_norm_rope");
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        return {qk_norm_rope_backward(grad_outputs[0], saved[0], saved[1], saved[2],
                                      ctx->saved_data["heads"].toInt(),
                                      static_cast<int>(
                                          ctx->saved_data["rounding"].toInt())),
                torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

// The four candle channels the geometry is defined on. Not a parameter anywhere: a candle
// has one close, one range and two positions inside it.
constexpr int64_t kLossChannels = 4;

// Dtype and device only, for an operand whose layout the kernel reads from strides.
void check_f32_cuda_strided(const torch::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.defined(), name, " is undefined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor, found ", tensor.device());
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat, name, " must be float32, found ",
                tensor.scalar_type());
}

// Dtype, device AND density, for the operands the kernels index with plain arithmetic.
void check_f32_cuda(const torch::Tensor &tensor, const char *name) {
    check_f32_cuda_strided(tensor, name);
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Everything the fused loss addresses by hand, checked once so the kernels can index with
// plain arithmetic. `tokens` folds rows and origins: the chain is per (origin, bar) and
// never crosses either boundary.
struct LossLayout {
    int64_t tokens;
    int64_t horizon;
    int64_t elements;
    // `targets` is the one operand the real path does NOT hand over dense. It is built with
    // `unfold`, and TensorIterator allocates the arithmetic below it in its inputs' own
    // permuted layout, so what arrives is channel-innermost: sizes
    // `[rows, origins, 4, horizon]`, strides `[origins·4·horizon, 4·horizon, 1, 4]`. The
    // ATen composition accepted that silently because every pointwise op it called is
    // strided; the kernel is told instead of copied, because a `.contiguous()` here would
    // read and write 295 MB per step to gain nothing.
    int64_t target_token_stride;
    int64_t target_channel_stride;
    int64_t target_bar_stride;
};

LossLayout loss_layout(const torch::Tensor &head, const torch::Tensor &targets,
                       const torch::Tensor &weighted_mask, const torch::Tensor &mask,
                       const torch::Tensor &sigma, const torch::Tensor &range,
                       const torch::Tensor &horizon_scale,
                       const torch::Tensor &inverse_horizon,
                       const torch::Tensor &log_scale_gain) {
    check_bf16_cuda(head, "loss_geometry head");
    TORCH_CHECK(head.dim() == 4, "loss_geometry head must be [rows, origins, ",
                2 * kLossChannels, ", horizon], found ", head.sizes());
    TORCH_CHECK(head.size(2) == 2 * kLossChannels, "loss_geometry head has ", head.size(2),
                " channels, not ", 2 * kLossChannels);
    TORCH_CHECK(head.is_contiguous(), "loss_geometry head must be contiguous");
    const int64_t horizon = head.size(3);
    const int64_t tokens = head.size(0) * head.size(1);
    const auto slice = [&](int64_t channels) {
        return std::vector<int64_t>{head.size(0), head.size(1), channels, horizon};
    };
    check_f32_cuda_strided(targets, "loss_geometry targets");
    TORCH_CHECK(targets.sizes() == at::IntArrayRef(slice(kLossChannels)),
                "loss_geometry targets are ", targets.sizes(), ", not ",
                at::IntArrayRef(slice(kLossChannels)));
    // Rows and origins are folded into one token axis, so the kernel needs the row stride
    // to be exactly `origins` token strides. Every layout the model produces satisfies
    // this; a genuinely row-strided target tensor would be a different op.
    TORCH_CHECK(targets.stride(0) == targets.size(1) * targets.stride(1),
                "loss_geometry targets must fold rows and origins into one token axis, "
                "found sizes ",
                targets.sizes(), " with strides ", targets.strides());
    const auto check_mask = [&](const torch::Tensor &tensor, const char *name) {
        check_f32_cuda(tensor, name);
        TORCH_CHECK(tensor.sizes() == at::IntArrayRef(slice(1)), name, " is ",
                    tensor.sizes(), ", not ", at::IntArrayRef(slice(1)));
    };
    check_mask(weighted_mask, "loss_geometry weighted mask");
    check_mask(mask, "loss_geometry mask");
    check_f32_cuda(sigma, "loss_geometry sigma");
    check_f32_cuda(range, "loss_geometry range");
    TORCH_CHECK(sigma.numel() == tokens && range.numel() == tokens,
                "loss_geometry sigma and range must hold ", tokens,
                " per-origin values, found ", sigma.numel(), " and ", range.numel());
    check_f32_cuda(horizon_scale, "loss_geometry horizon scale");
    check_f32_cuda(inverse_horizon, "loss_geometry inverse horizon");
    TORCH_CHECK(horizon_scale.numel() == horizon && inverse_horizon.numel() == horizon,
                "loss_geometry horizon buffers must hold ", horizon, " values, found ",
                horizon_scale.numel(), " and ", inverse_horizon.numel());
    check_f32_cuda(log_scale_gain, "loss_geometry log scale gain");
    TORCH_CHECK(log_scale_gain.numel() == 1, "loss_geometry log scale gain must hold one "
                                             "value so the launch never reads the host");
    return {tokens,
            horizon,
            tokens * horizon,
            targets.stride(1),
            targets.stride(2),
            targets.stride(3)};
}

// The candle geometry, the per-channel precision weight and the squared residuals in ONE
// pass, with the twelve reductions they feed left to ATen.
//
// WHY THE REDUCTIONS STAY. `dot` is cuBLAS's `Sdot`, whose summation tree is neither
// documented nor reproducible; a hand-written reduction over 18.4 M elements would be a
// different tree and the loss value would move. So the fusion is drawn exactly at the
// elementwise/reduction boundary: forty-eight full-size fp32 passes collapse into one, the
// twelve fp32 vectors the `dot`s consume are materialized, and the reduction bits are
// untouched. The forward saves NOTHING for backward beyond its own inputs - the backward
// recomputes the chain - so the twelve vectors are freed the moment forward returns, where
// the composition kept them plus the geometry's own five alive across the whole step.
struct LossGeometryOp : public torch::autograd::Function<LossGeometryOp> {
    static torch::autograd::variable_list
    forward(torch::autograd::AutogradContext *ctx, torch::Tensor head,
            torch::Tensor targets, torch::Tensor weighted_mask, torch::Tensor mask,
            torch::Tensor sigma, torch::Tensor range, torch::Tensor horizon_scale,
            torch::Tensor inverse_horizon, torch::Tensor log_scale_gain, double cap,
            double ln2, int64_t rounding, bool decoupled) {
        const LossLayout layout =
            loss_layout(head, targets, weighted_mask, mask, sigma, range, horizon_scale,
                        inverse_horizon, log_scale_gain);
        torch::Tensor close = at::empty(
            {head.size(0), head.size(1), 1, layout.horizon}, weighted_mask.options());
        const int64_t term_count = (decoupled ? 3 : 2) * kLossChannels;
        torch::Tensor workspace = at::empty(
            {3 * kLossChannels + (decoupled ? 1 : 0), layout.elements},
            weighted_mask.options());
        const float narrow_ln2 = static_cast<float>(ln2);
        check_launch(fk_loss_geometry_forward(
                         head.const_data_ptr(), targets.const_data_ptr(),
                         weighted_mask.const_data_ptr(), sigma.const_data_ptr(),
                         range.const_data_ptr(), horizon_scale.const_data_ptr(),
                         inverse_horizon.const_data_ptr(),
                         log_scale_gain.const_data_ptr(), close.mutable_data_ptr(),
                         workspace.mutable_data_ptr(), layout.tokens, layout.horizon,
                         layout.target_token_stride, layout.target_channel_stride,
                         layout.target_bar_stride,
                         static_cast<float>(cap), narrow_ln2, 1.0f / narrow_ln2,
                         static_cast<int>(rounding), decoupled, current_stream(head)),
                     "loss_geometry forward");
        std::vector<torch::Tensor> terms;
        std::vector<torch::Tensor> squares;
        std::vector<torch::Tensor> nll_terms;
        terms.reserve(term_count);
        if (decoupled) {
            nll_terms.reserve(2 * kLossChannels);
        }
        squares.reserve(kLossChannels);
        const torch::Tensor weighted_flat = weighted_mask.reshape({-1});
        const torch::Tensor mask_flat = mask.reshape({-1});
        for (int64_t channel = 0; channel < kLossChannels; ++channel) {
            const torch::Tensor square = workspace.select(0, channel);
            const torch::Tensor quadratic =
                square.dot(workspace.select(0, kLossChannels + channel)) * 0.5;
            const torch::Tensor logarithmic =
                workspace.select(0, 2 * kLossChannels + channel).dot(weighted_flat) * cap;
            if (decoupled) {
                terms.push_back(square.dot(workspace.select(0, 3 * kLossChannels)) * 0.5);
                nll_terms.push_back(quadratic);
                nll_terms.push_back(logarithmic);
            }
            terms.push_back(quadratic);
            terms.push_back(logarithmic);
            squares.push_back(square.dot(mask_flat));
        }
        // `head` and the targets are already resident and the rest is per-origin or
        // per-horizon, so this retains no full-size fp32 tensor at all.
        ctx->save_for_backward({head, targets, weighted_mask, sigma, range, horizon_scale,
                                inverse_horizon, log_scale_gain});
        ctx->saved_data["cap"] = cap;
        ctx->saved_data["ln2"] = ln2;
        ctx->saved_data["rounding"] = rounding;
        ctx->saved_data["decoupled"] = decoupled;
        // NOT materialized. Autograd's default hands an unused output a freshly ZEROED
        // tensor, and for the mean coordinate that is not a no-op: the kernel would add
        // `+0` to a gradient that is `-0` wherever the origin is invalid, and `-0 + +0` is
        // `+0`. That is a REAL disagreement with the composition, which has no such addend
        // at all when nothing consumes `close`, and it cost this fusion its bit-exactness
        // on 3244 of 4,608,000 mean-coordinate elements until it was found. Undefined is
        // the honest encoding of "no external consumer", and both the check below and the
        // kernel's `nullptr` path already speak it.
        ctx->set_materialize_grads(false);
        torch::Tensor stacked_squares = at::stack(squares);
        // The diagnostic MSE carries no gradient in the composition either - it is inside
        // a `no_grad` there - and saying so here is what stops autograd from demanding a
        // gradient path for it.
        if (decoupled) {
            torch::Tensor stacked_nll = at::stack(nll_terms);
            ctx->mark_non_differentiable({stacked_squares, stacked_nll});
            return {close, at::stack(terms), stacked_squares, stacked_nll};
        }
        ctx->mark_non_differentiable({stacked_squares});
        return {close, at::stack(terms), stacked_squares};
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::variable_list grad_outputs) {
        reject_double_backward(grad_outputs, "loss_geometry");
        const torch::autograd::variable_list saved = ctx->get_saved_variables();
        const torch::Tensor &head = saved[0];
        const double cap = ctx->saved_data["cap"].toDouble();
        const double ln2 = ctx->saved_data["ln2"].toDouble();
        const int64_t rounding = ctx->saved_data["rounding"].toInt();
        const bool decoupled = ctx->saved_data["decoupled"].toBool();
        const int64_t term_count = (decoupled ? 3 : 2) * kLossChannels;
        const int64_t horizon = head.size(3);
        const int64_t tokens = head.size(0) * head.size(1);
        const torch::Tensor grad_close =
            grad_outputs[0].defined() ? grad_outputs[0].contiguous() : torch::Tensor();
        const torch::Tensor grad_terms =
            grad_outputs[1].defined() ? grad_outputs[1].contiguous() : torch::Tensor();
        if (grad_close.defined()) {
            check_f32_cuda(grad_close, "loss_geometry close gradient");
            TORCH_CHECK(grad_close.numel() == tokens * horizon,
                        "loss_geometry close gradient has ", grad_close.numel(),
                        " elements but the mean coordinate had ", tokens * horizon);
        }
        if (grad_terms.defined()) {
            check_f32_cuda(grad_terms, "loss_geometry term gradient");
            TORCH_CHECK(grad_terms.numel() == term_count,
                        "loss_geometry term gradient has ", grad_terms.numel(),
                        " elements, not ", term_count);
        }
        // The SAVED targets, strides and all: the forward retained the caller's tensor
        // rather than a dense copy of it, so the backward reads the same layout.
        const torch::Tensor &targets = saved[1];
        torch::Tensor grad_head = at::empty_like(head, at::MemoryFormat::Contiguous);
        const float narrow_ln2 = static_cast<float>(ln2);
        check_launch(
            fk_loss_geometry_backward(
                head.const_data_ptr(), saved[1].const_data_ptr(),
                saved[2].const_data_ptr(), saved[3].const_data_ptr(),
                saved[4].const_data_ptr(), saved[5].const_data_ptr(),
                saved[6].const_data_ptr(), saved[7].const_data_ptr(),
                grad_terms.defined() ? grad_terms.const_data_ptr() : nullptr,
                grad_close.defined() ? grad_close.const_data_ptr() : nullptr,
                grad_head.mutable_data_ptr(), tokens, horizon, targets.stride(1),
                targets.stride(2), targets.stride(3), static_cast<float>(cap),
                narrow_ln2, 1.0f / narrow_ln2, static_cast<int>(rounding),
                decoupled, current_stream(head)),
            "loss_geometry backward");
        return {grad_head,      torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor()};
    }
};

// A CUDA-event timer, exposed because the composed-versus-fused comparison has to be
// measured with events rather than a host clock. The repository's own profiler learned this
// the expensive way: a host-timed mean over ten rounds reported this card's copy peak at
// 630 GB/s while its own kernels measured 1569 GB/s, so every roofline fraction in that
// profile was halved by contention landing in the denominator. Events plus a best-of-N
// reduction fix it. `stop` synchronizes, which is exactly why these are probe-only entry
// points and are never reachable from the model path.
struct Timer {
    cudaEvent_t start;
    cudaEvent_t stop;
};

template <typename Body> void *protect(Body body) {
    last_error.clear();
    try {
        return static_cast<void *>(new torch::Tensor(body()));
    } catch (const std::exception &error) {
        last_error = error.what();
    } catch (...) {
        last_error = "unknown C++ exception";
    }
    return nullptr;
}

const torch::Tensor &borrow(const void *handle) {
    return *static_cast<const torch::Tensor *>(handle);
}

} // namespace

extern "C" {

// Non-null on failure, and cleared by the next call on this thread.
const char *fk_last_error() { return last_error.empty() ? nullptr : last_error.c_str(); }

void *fk_relu_square(const void *input) {
    return protect([&] { return ReluSquare::apply(borrow(input)); });
}

void *fk_rope(const void *input, const void *cosine, const void *sine, int64_t heads) {
    return protect([&] {
        return Rope::apply(borrow(input), borrow(cosine), borrow(sine), heads);
    });
}

void *fk_qk_norm_rope(const void *input, const void *cosine, const void *sine,
                      int64_t heads, int64_t rounding) {
    return protect([&] {
        return QkNormRope::apply(borrow(input), borrow(cosine), borrow(sine), heads,
                                 rounding);
    });
}

// Three coupled outputs, or four decoupled outputs (the fourth is detached true-NLL
// terms). Every output owns a freshly allocated `torch::Tensor` handle. Returns 0 on
// success and leaves `fk_last_error` set otherwise.
int fk_loss_geometry(const void *head, const void *targets, const void *weighted_mask,
                     const void *mask, const void *sigma, const void *range,
                     const void *horizon_scale, const void *inverse_horizon,
                     const void *log_scale_gain, double cap, double ln2, int64_t rounding,
                     int64_t decoupled, void **outputs) {
    last_error.clear();
    try {
        const torch::autograd::variable_list produced = LossGeometryOp::apply(
            borrow(head), borrow(targets), borrow(weighted_mask), borrow(mask),
            borrow(sigma), borrow(range), borrow(horizon_scale), borrow(inverse_horizon),
            borrow(log_scale_gain), cap, ln2, rounding, decoupled != 0);
        const size_t expected = decoupled ? 4 : 3;
        TORCH_CHECK(produced.size() == expected, "the fused loss returned ", produced.size(),
                    " tensors, not ", expected);
        for (size_t index = 0; index < produced.size(); ++index) {
            outputs[index] = static_cast<void *>(new torch::Tensor(produced[index]));
        }
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
    } catch (...) {
        last_error = "unknown C++ exception";
    }
    return -1;
}

// The backward kernels called directly, with no autograd node around them. NOT
// differentiable and not for the model path: they exist so the microbenchmark can time the
// kernel rather than the kernel plus whatever objective the harness needed to reach it.
void *fk_relu_square_backward_raw(const void *input, const void *grad) {
    return protect([&] { return relu_square_backward(borrow(input), borrow(grad)); });
}

void *fk_rope_backward_raw(const void *grad, const void *cosine, const void *sine,
                           int64_t heads) {
    return protect(
        [&] { return rope_backward(borrow(grad), borrow(cosine), borrow(sine), heads); });
}

void *fk_qk_norm_rope_backward_raw(const void *grad, const void *input, const void *cosine,
                                   const void *sine, int64_t heads, int64_t rounding) {
    return protect([&] {
        return qk_norm_rope_backward(borrow(grad), borrow(input), borrow(cosine),
                                     borrow(sine), heads, static_cast<int>(rounding));
    });
}

// A bf16 clone through the vectorized copy kernel: the streaming roof, measured the same
// way as the kernels measured against it.
void *fk_stream_copy_tensor(const void *input) {
    return protect([&] {
        const torch::Tensor &source = borrow(input);
        check_bf16_cuda(source, "stream_copy input");
        TORCH_CHECK(source.is_contiguous(), "stream_copy input must be contiguous");
        torch::Tensor output = at::empty_like(source, at::MemoryFormat::Contiguous);
        check_launch(fk_stream_copy(source.const_data_ptr(), output.mutable_data_ptr(),
                                      source.numel(), current_stream(source)),
                     "stream_copy");
        return output;
    });
}

void *fk_timer_new() {
    last_error.clear();
    auto *timer = new Timer{};
    if (cudaEventCreate(&timer->start) != cudaSuccess ||
        cudaEventCreate(&timer->stop) != cudaSuccess) {
        last_error = "could not create CUDA events";
        delete timer;
        return nullptr;
    }
    return static_cast<void *>(timer);
}

// Records on the stream the kernels launch on, so the interval brackets exactly their work.
int fk_timer_start(void *handle) {
    auto *timer = static_cast<Timer *>(handle);
    return static_cast<int>(
        cudaEventRecord(timer->start, at::cuda::getCurrentCUDAStream().stream()));
}

// Milliseconds since `fk_timer_start`, or a negative value on failure.
double fk_timer_stop(void *handle) {
    auto *timer = static_cast<Timer *>(handle);
    if (cudaEventRecord(timer->stop, at::cuda::getCurrentCUDAStream().stream()) !=
            cudaSuccess ||
        cudaEventSynchronize(timer->stop) != cudaSuccess) {
        return -1.0;
    }
    float milliseconds = 0.0f;
    if (cudaEventElapsedTime(&milliseconds, timer->start, timer->stop) != cudaSuccess) {
        return -1.0;
    }
    return static_cast<double>(milliseconds);
}

void fk_timer_free(void *handle) {
    auto *timer = static_cast<Timer *>(handle);
    if (timer == nullptr) {
        return;
    }
    cudaEventDestroy(timer->start);
    cudaEventDestroy(timer->stop);
    delete timer;
}
}
