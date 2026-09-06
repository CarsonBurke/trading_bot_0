use anyhow::{ensure, Result};
use fused_kernels::{qk_norm_rope, relu_square};
use serde::{Deserialize, Serialize};
use tch::{nn, Device, Kind, Tensor};

use super::{
    corpus::Batch,
    features::{Feature, FeatureSet},
};
use crate::torch::model::rope::RotaryEmbedding;

pub const CHANNELS: i64 = 4;
/// Per-bar log-return variance floor: one basis point of volatility per five-minute bar.
pub const RETURN_VARIANCE_FLOOR: f64 = 1e-8;
/// Ridge weight of the β=1 prior in bars: β_k is the OLS slope of the ticker's bar returns on
/// the market steps over `[0, t_k]`, shrunk toward 1 with the weight of `BETA_PRIOR_BARS` average
/// market-step squares, so origins with little history sit near 1 and β converges to the OLS
/// slope as history accumulates (half-way at 256 valid pairs).
pub const BETA_PRIOR_BARS: f64 = 256.0;
const COVARIATE_WIDTH: i64 = 256;
const HEAD_HIDDEN: i64 = 1024;
/// Four candle coordinates plus one log-scale per OHLC channel for every future bar.
const OUTPUTS_PER_BAR: i64 = 2 * CHANNELS;
/// μP-style output multiplier `1/√fan_in` on the zero-initialised head: Adam moves every weight
/// by ~lr per step, so an unscaled 1024-wide head would jump O(lr·fan_in) ≈ 8 per output.
const HEAD_OUTPUT_SCALE: f64 = 1.0 / 32.0;
/// tanh soft cap on the log predictive scale around the `½·ln h` prior: `s/σ√h ∈ [e⁻⁴, e⁴]`.
const LOG_SCALE_CAP: f64 = 4.0;
/// RMSNorm epsilon - see [`rms_norm`] for why it is explicit rather than the reference's
/// dtype-dependent default.
const NORM_EPS: f64 = 1e-6;
/// Residual-stream scale at init, per SUB-BLOCK. modded-nanogpt `train_gpt.py:1336-1338`:
/// "sqrt(1.1) per sublayer so cumulative per-layer scaling is 1.1", a raw (not sigmoid)
/// learnable scalar. The stack therefore multiplies the residual stream by 1.1^layers before
/// the final norm, which the final norm divides straight back out - see
/// `the_stack_is_the_identity_on_the_residual_stream_at_init`.
const RESID_LAMBDA_INIT: f64 = 1.048_808_848_170_151_6; // √1.1, `world_model.rs:112-113`
/// Sub-block output scale at init. modded-nanogpt `train_gpt.py:1334`:
/// `self.post_lambdas = nn.Parameter(torch.ones(num_layers, 2))`.
const POST_LAMBDA_INIT: f64 = 1.0;
/// Embedding re-injection scale at init. modded-nanogpt `train_gpt.py:1389`
/// (`bs_init[0, 10] = 0.0  # x0_lambda[10] (init 0)`) and `train_gpt_medium.py:1078`
/// (`self.x0_lambdas = nn.Parameter(torch.zeros(2*num_layers))`). Zero, so the injection
/// starts as an exact no-op and the stack starts as the identity map.
const X0_LAMBDA_INIT: f64 = 0.0;
/// Logit initialising every U-net skip gate, from modded-nanogpt: the skips entered the
/// reference at raw weight 1.0 (`records/track_1_short/2024-11-10_UNetDoubleLr/
/// c87bb826-797b-4f37-98c7-d3a5dad2de74.txt:244`) and were re-parameterised to
/// `σ(logit)` with this init in `records/track_1_short/2025-11-18_RefineSkip/
/// 00f4e1e6-0044-4a08-b88a-3b7ec0624081.txt:953-954`, which the current
/// `train_gpt.py:1346` still carries verbatim (`-1.5 * torch.ones(1),  # skip_lambda ->
/// σ(-1.5) ≈ 0.18`). σ(-1.5) = 0.18243: eight layers whose decoder half adds an encoder
/// output at weight 1 would double the early residual stream before a single gradient step,
/// and the sigmoid keeps the gate in (0, 1) however far the logit travels.
const SKIP_LOGIT_INIT: f64 = -1.5;
/// Value-residual mixing weight at init, from modded-nanogpt's learnable form
/// (`records/track_1_short/2024-11-06_ShortcutsTweaks/43f60c4f-0448-4de7-83d9-643ca26f61e7.txt`
/// `:168`, `self.lamb = nn.Parameter(torch.tensor(0.5))`): a RAW per-layer scalar - no sigmoid,
/// not shared - carrying the fixed 0.5 of the original value residual (arXiv:2410.17897) as its
/// starting point.
const VALUE_LAMBDA_INIT: f64 = 0.5;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, clap::Args)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    #[arg(long, default_value_t = 6000)]
    pub seq_len: i64,
    #[arg(long, default_value_t = 192)]
    pub pred_len: i64,
    #[arg(long, default_value_t = 16)]
    pub patch_len: i64,
    #[arg(long, default_value_t = 8)]
    pub layers: usize,
    #[arg(long, default_value_t = 512)]
    pub d_model: i64,
    #[arg(long, default_value_t = 8)]
    pub heads: i64,
    #[arg(long, default_value_t = 2048)]
    pub ffn: i64,
    #[arg(long, default_value_t = 0.0)]
    pub dropout: f64,
    /// Origins with fewer valid history bars are excluded from the loss.
    #[arg(long, default_value_t = 256)]
    pub min_history: i64,
    /// Comma-separated exogenous variates: time-of-day, day-of-week, session-gap, volume, market, spy; `all` or `none`.
    #[arg(long, default_value_t = FeatureSet::ALL)]
    pub features: FeatureSet,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            seq_len: 6000,
            pred_len: 192,
            patch_len: 16,
            layers: 8,
            d_model: 512,
            heads: 8,
            ffn: 2048,
            dropout: 0.0,
            min_history: 256,
            features: FeatureSet::ALL,
        }
    }
}

/// Arithmetic and traffic cost of one training step, forward plus backward.
///
/// `traffic_bytes` is a LOWER BOUND: it charges every materialized activation one read and one
/// write per pass, so the ratio of the measured step time to `traffic_bytes / peak bandwidth` is
/// the multiplier that elementwise re-reads and gradient plumbing add on top. That ratio, not
/// the bound itself, is what the fusion work moves.
#[derive(Debug, Clone, Copy)]
pub struct StepCost {
    pub matmul_flops: f64,
    pub traffic_bytes: f64,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.seq_len > 0 && self.pred_len > 0,
            "sequence lengths must be positive"
        );
        ensure!(
            self.patch_len > 0 && self.seq_len % self.patch_len == 0,
            "patches must cover the entire context exactly"
        );
        ensure!(
            self.heads > 0 && self.d_model > 0 && self.d_model % self.heads == 0,
            "model width must be divisible by attention heads"
        );
        ensure!(
            (self.d_model / self.heads) % 2 == 0,
            "rotary attention needs an even head dimension"
        );
        ensure!(
            self.layers > 0 && self.ffn > 0,
            "layer count and feedforward width must be positive"
        );
        ensure!(
            self.layers % 2 == 0,
            "the U-net skip stack pairs each encoder layer with one decoder layer, so the layer count must be even"
        );
        ensure!(
            self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout),
            "dropout must be in [0, 1)"
        );
        ensure!(
            (2..=self.seq_len).contains(&self.min_history),
            "min_history must lie in [2, seq_len]"
        );
        Ok(())
    }

    pub fn origins(&self) -> i64 {
        self.seq_len / self.patch_len
    }

    /// `(encoder source, decoder destination)` for every U-net skip, deepest encoder layer
    /// first: at eight layers this is `3->4, 2->5, 1->6, 0->7`. The order IS the stack
    /// discipline - encoder layers push in index order, decoder layers pop - and the `k`-th
    /// pair is gated by `skip_weights[k]`, matching the reference's `skip_weights[i - n]`
    /// indexing (`records/track_1_short/2024-11-10_UNetDoubleLr/
    /// c87bb826-797b-4f37-98c7-d3a5dad2de74.txt:266-270`).
    pub fn skip_pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let encoder = self.layers / 2;
        (0..self.layers - encoder).map(move |k| (encoder - 1 - k, encoder + k))
    }

    /// Arithmetic and traffic cost of one training step at `batch`, from shapes alone.
    pub fn step_cost(&self, batch: usize) -> StepCost {
        let (rows, origins, horizon) = (batch as f64, self.origins() as f64, self.pred_len as f64);
        let tokens = rows * origins;
        let (width, ffn) = (self.d_model as f64, self.ffn as f64);
        let aux = self.features.channels() as f64;
        let known = self
            .features
            .channel_mask(Feature::known_future)
            .iter()
            .filter(|known| **known)
            .count() as f64;
        let covariate_width = if known > 0. { COVARIATE_WIDTH as f64 } else { 0. };
        let head_input = width + covariate_width;
        let head_outputs = horizon * OUTPUTS_PER_BAR as f64;
        let gemm = |rows: f64, reduce: f64, out: f64| 2. * rows * reduce * out;
        let mut flops = gemm(
            tokens,
            self.patch_len as f64 * (CHANNELS as f64 + aux),
            width,
        );
        // Causal SDPA: `QKᵀ` and `AV` are each `2·rows·origins²·d_model` dense and half of that
        // under the mask, so the pair costs one dense GEMM's worth.
        flops += self.layers as f64
            * (gemm(tokens, width, 3. * width)
                + gemm(tokens, width, width)
                + 2. * gemm(tokens, width, ffn)
                + 2. * rows * origins * origins * width);
        flops += gemm(tokens, horizon * known, covariate_width)
            + gemm(tokens, head_input, HEAD_HIDDEN as f64)
            + gemm(tokens, HEAD_HIDDEN as f64, head_outputs);
        // Every activation the step materializes, written once and read once. Backward pays it
        // twice more (grad-in, grad-out), hence the factor three on both totals.
        let bf16 = |elements: f64| 2. * elements;
        let fp32 = |elements: f64| 4. * elements;
        let mut bytes = bf16(tokens * self.patch_len as f64 * (CHANNELS as f64 + aux));
        // Per layer: two pre-norms (2·width), the packed projection (3·width), the OUTPUT of the
        // fused QK-norm-plus-rotation (2·width - the only tensor `fused_kernels::qk_norm_rope`
        // materializes; the composed form charged 2·width for the normalized `q‖k` block it had
        // to hand on and 8·width for the rotation's two full-width products, its two
        // half-crossing sums and the buffer that interleaved them), the attention output, its
        // projection and the two residual `addcmul`s (6·width), the x0 injection's second
        // `addcmul` (1·width), and the feedforward PAIR (2·ffn - the up projection and
        // `fused_kernels::relu_square`'s single output; the `relu`-then-`square` composition
        // materialized three, GELU two). The rotation term was missing entirely before this
        // accounting was measured per kernel class, and at 8·width of 22 the bound it produced
        // was 40% low.
        //
        // Against the LayerNorm/GELU form (19·width + 2·ffn) the residual recipe now adds ONE
        // width-unit per layer, the x0 `addcmul`, and nothing else: the QK norm it introduced no
        // longer materializes anything, `square`'s extra `[tokens, ffn]` tensor is gone, and so
        // are six of the rotation's eight width-units. The post-lambdas cost nothing here
        // because they ride the projection weights ([`scaled_linear`]), and the residual scales
        // cost nothing because `addcmul` folds them into the add that was already there.
        //
        // The fp32 `rstd` a materializing RMSNorm writes is not charged here and never was;
        // the fused kernel writes none at all, recomputing the normalization in its backward.
        bytes += self.layers as f64
            * bf16(tokens * (2. * width + 3. * width + 2. * width + 6. * width + width + 2. * ffn));
        // The patch embedding's own norm: `x0 = rms_norm(patch(tokens))`, once per step.
        bytes += bf16(tokens * width);
        // The U-net skip: ONE fused `addcmul` per decoder layer (see `unet_stack`), so each
        // pair materializes a single `width` activation, not the two a separate `gate * skip`
        // product and add would. The encoder half's outputs are free - they are the very
        // tensors the next layer's norm already retains, so the stack itself adds no bytes.
        bytes += (self.layers / 2) as f64 * bf16(tokens * width);
        bytes += bf16(tokens * (width + horizon * known + covariate_width + head_input))
            + bf16(tokens * 2. * HEAD_HIDDEN as f64)
            + bf16(tokens * head_outputs);
        // The head geometry, the NLL and their backward are counted directly rather than
        // scaled: 275 reads-or-writes of the `[rows, origins', 1, pred_len]` fp32 channel
        // space, enumerated op by op from `CausalPatchModel::losses` (121 forward, 144
        // backward, 10 for building the targets and the mask, which carry no gradient).
        let head_loss = fp32(tokens * horizon) * 275.;
        // Value residual: `layers - 1` decoder layers each `lerp` their own value against layer
        // 0's. Ten passes over one `[tokens, d_model]` bf16 activation per such layer - three
        // forward (two reads and one write) and seven backward (`grad_self` and `grad_end` at a
        // read plus a write each, and the three-operand reduction that produces the lambda
        // gradient) - plus the in-place adds that accumulate those contributions into layer 0's
        // value gradient, three passes each. Counted here rather than folded into the per-layer
        // width units because the mix is not a plain write-once-read-once materialization.
        let decoders = (self.layers - 1).max(0) as f64;
        let value_residual =
            bf16(tokens * width) * (10. * decoders + 3. * (decoders - 1.).max(0.));
        StepCost {
            matmul_flops: 3. * flops,
            traffic_bytes: 3. * 2. * bytes + head_loss + value_residual,
        }
    }
}

fn projection(path: nn::Path, input: i64, output: i64, bias: bool) -> nn::Linear {
    // Match torch.nn.Linear, including its fan-in initialization.
    let bound = 1.0 / (input as f64).sqrt();
    let init = nn::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    nn::linear(
        path,
        input,
        output,
        nn::LinearConfig {
            ws_init: init,
            bs_init: Some(init),
            bias,
        },
    )
}

/// A projection whose weight AND bias start at exactly zero, so the branch it terminates
/// contributes nothing at step 0.
///
/// modded-nanogpt zero-initialises every residual-branch output projection: the MLP down
/// projection (`train_gpt_medium.py:1006` `self.c_proj.zero_()`, `train_gpt.py:1308`
/// `self.mlp_bank[:, 1, :, :].zero_()`, both "zero init suggested by @Grad62304977") and the
/// attention output projection, which lives in the same `vo_bank` and is zeroed for the padded
/// groups while the real ones are folded into `CastedLinearT`, whose `reset_parameters` is
/// `nn.init.zeros_(self.weight)` (`train_gpt.py:973-975`).
///
/// Bias-free, like every projection in the reference (`train_gpt.py:1103` and `:1161` call
/// `F.linear` with a weight only; `train_gpt_medium.py:995-996` stores bare weight
/// parameters). Dropping the block biases removes four AdamW parameters per layer from the
/// force-AdamW list and four fp32->bf16 parameter casts per layer, and it is what makes the
/// zero-init actually mean "this branch is off": a nonzero bias would leave a constant
/// per-channel offset on the residual stream at step 0.
fn zeroed_projection(path: nn::Path, input: i64, output: i64) -> nn::Linear {
    nn::linear(
        path,
        input,
        output,
        nn::LinearConfig {
            ws_init: nn::Init::Const(0.0),
            bs_init: None,
            bias: false,
        },
    )
}

/// A bias-free hidden projection at the reference's 2D init scale,
/// `uniform(±√3·0.5·fan_in^-½)` (std `0.5·fan_in^-½`, i.e. 0.866 of the `1/√fan_in` bound
/// `torch.nn.Linear` uses): `train_gpt.py:1288-1291` and `:1304-1307`, "improved init scale by
/// @YouJiacheng and @srashedll", `train_gpt_medium.py:1002-1005`, and in-repo
/// `world_model.rs:2903-2909` (`uniform_init`). Bias-free for the reasons in
/// [`zeroed_projection`].
fn hidden_projection(path: nn::Path, input: i64, output: i64) -> nn::Linear {
    let bound = 3f64.sqrt() * 0.5 / (input as f64).sqrt();
    nn::linear(
        path,
        input,
        output,
        nn::LinearConfig {
            ws_init: nn::Init::Uniform {
                lo: -bound,
                up: bound,
            },
            bs_init: None,
            bias: false,
        },
    )
}

fn linear(input: &Tensor, layer: &nn::Linear) -> Tensor {
    input.linear(
        &layer.ws.to_kind(input.kind()),
        layer.bs.as_ref().map(|bias| bias.to_kind(input.kind())),
    )
}

/// [`linear`] with a scalar folded onto the weight and bias copies the GEMM already needs.
///
/// modded-nanogpt folds its own sub-block scalars onto the projection this way rather than
/// scaling the activation: `train_gpt.py:1161`,
/// `y = F.linear(y, sa_lambdas[1] * qkvo_w[self.dim * 3:].type_as(y))` with the comment
/// "sa_lambdas[1] pre-multiplied to O @shenberg". The product and its gradient are identical
/// either way; a 512×512 weight is 1/96 000 of the `[rows, origins, d_model]` activation it
/// would otherwise multiply, so folding the post-lambda here costs no traffic at all.
fn scaled_linear(input: &Tensor, layer: &nn::Linear, scale: &Tensor) -> Tensor {
    input.linear(
        &(layer.ws.to_kind(input.kind()) * scale),
        layer
            .bs
            .as_ref()
            .map(|bias| bias.to_kind(input.kind()) * scale),
    )
}

/// `x * rsqrt(mean(x²) + eps)` over the last dimension, with NO learnable gain and no bias.
///
/// modded-nanogpt's only normalization, `train_gpt.py:952-953` and
/// `train_gpt_medium.py:839-840`:
///
/// ```python
/// def norm(x: Tensor):
///     return F.rms_norm(x, (x.size(-1),))
/// ```
///
/// `F.rms_norm` with `weight=None` is parameter-free, so this replaces LayerNorm's per-block
/// gain and bias with nothing at all: no 1024-element parameter cast per norm, no mean pass,
/// no gain/bias gradients, and - the reason it matters here - no scalar tensors that could be
/// misrouted into NorMuon's 2D-hidden group. The block's own scale freedom lives in the
/// residual lambdas instead (see [`BlockLambdas`]).
///
/// `eps` is passed explicitly, at the in-repo value: `world_model.rs:109-110`
/// (`BAR_NORM_EPS = 1e-6`, "The norm carries no learnable gain anywhere in this model"), which
/// is also what the reference passes where it passes one at all (`train_gpt.py:1079`).
///
/// What `eps=None` actually resolves to was MEASURED, not read off `finfo`: on a bf16 CUDA
/// input, `_fused_rms_norm(x, shape, None, None)` is bit-identical to `eps = 1.1920929e-7`
/// (max difference exactly 0.0) and nowhere near `finfo(bfloat16).eps = 7.8e-3`, which would
/// have shrunk the sample by 95%. The kernel resolves the default from the fp32 ACCUMULATE
/// type, not from the input dtype, so at unit RMS `None` and 1e-6 are indistinguishable.
/// Passing 1e-6 is still the right call and is not cosmetic: on a head block whose RMS has
/// collapsed - 4.2e-3 in the same probe, where eps is 5.6% of the mean square - the two
/// choices differ by 2.5%, and 1e-6 is the floor the rest of the repo normalizes against.
///
/// `_fused_rms_norm`, NOT `rms_norm`: `rms_norm`'s composite body dispatches to
/// `_fused_rms_norm`, but `rms_norm` itself registers as a math kernel on CUDA, so calling it
/// directly costs an extra dispatch layer and hides the real kernel from the profiler - see
/// `train/pretrain_profile.rs:95-99`, which names this exact trap, and `model/rmsnorm.rs:17`,
/// which is the in-repo form. The second return is the fp32 `rstd`, which the RMSNorm class
/// in [`CausalPatchModel::kernel_classes`] charges for.
fn rms_norm(input: &Tensor) -> Tensor {
    let width = *input
        .size()
        .last()
        .expect("rms_norm needs at least one dimension");
    input
        .internal_fused_rms_norm([width], None::<&Tensor>, Some(NORM_EPS))
        .0
}

/// The five residual-mixing scalars one [`Block`] consumes, as 0-dim views of the model-level
/// lambda vectors ([`CausalPatchModel::resid_lambdas`] and friends). Borrowed rather than
/// owned so the whole stack pays one cast and one `unbind` per step, exactly as the reference
/// does (`train_gpt.py:1509-1512`: `self.resid_lambdas[:, 0].bfloat16().unbind(0)`).
struct BlockLambdas<'a> {
    /// Residual-stream scale, `[attention, feedforward]`. `train_gpt.py:1338`.
    resid: [&'a Tensor; 2],
    /// Sub-block output scale, `[attention, feedforward]`. `train_gpt.py:1334`.
    post: [&'a Tensor; 2],
    /// Embedding re-injection scale, applied on the attention residual only.
    /// `train_gpt.py:1638` with the init-0 gate bias of `train_gpt.py:1389`.
    x0: &'a Tensor,
}

struct Block {
    qkv: nn::Linear,
    output: nn::Linear,
    first: nn::Linear,
    second: nn::Linear,
    /// Value-residual mixing weight, `[1]` fp32, `None` in the SOURCE layer.
    ///
    /// modded-nanogpt's value residual (`records/track_1_short/2024-11-06_ShortcutsTweaks/`
    /// `README.md:16-41`, code at `43f60c4f-0448-4de7-83d9-643ca26f61e7.txt:168,177`) gives every
    /// block a raw scalar `lamb`, initialized to 0.5, and mixes
    /// `v = (1 - lamb)·v + lamb·v_1` against layer 0's value. Layer 0 receives `v1=None` and
    /// sets `v1 = v`, so its own mix is the identity and its `lamb` gets an exactly zero
    /// gradient - a dead parameter. Here that dead scalar simply does not exist, which is also
    /// what makes layer 0 bit-identical to the model without this path.
    value_lambda: Option<Tensor>,
    heads: i64,
    width: i64,
    dropout: f64,
}

impl Block {
    /// `layer` selects the value-residual role: layer 0 publishes its value, every later layer
    /// mixes against it.
    fn new(path: nn::Path, config: &ModelConfig, layer: usize) -> Self {
        let width = config.d_model;
        Self {
            qkv: hidden_projection(&path / "qkv", width, 3 * width),
            output: zeroed_projection(&path / "output", width, width),
            first: hidden_projection(&path / "first", width, config.ffn),
            second: zeroed_projection(&path / "second", config.ffn, width),
            value_lambda: (layer > 0)
                .then(|| path.var("value_lambda", &[1], nn::Init::Const(VALUE_LAMBDA_INIT))),
            heads: config.heads,
            width,
            dropout: config.dropout,
        }
    }

    /// One pre-norm block: `x = λr·x + λp·O(attn(rms(x))) + λ0·x0`, then
    /// `x = λr·x + λp·W2(relu(W1(rms(x)))²)`.
    ///
    /// Pre-norm, matching the reference: `train_gpt_medium.py:1020-1023`
    /// (`x = x + self.attn(norm(x))`, `x = x + self.mlp(norm(x))`) and `train_gpt.py:1598`
    /// (`attn_in_normed = norm(cache.get(7, x))`) / `:1643` (`normed = norm(x)`). The residual
    /// lambda scheme only makes sense on a pre-norm stack - it rescales the raw residual
    /// stream, which a post-norm block would immediately normalize away - and this model was
    /// already pre-norm, so nothing had to be converted.
    ///
    /// The lambda lines are `train_gpt.py:1638` and `:1665`:
    ///
    /// ```python
    /// x = resid_lambdas_attn[i] * x + post_lambdas_attn[i] * attn_out + x0 * x0_gates[i]
    /// x = resid_lambdas_mlp[i] * x + post_lambdas_mlp[i] * ReLUSqrdMLP(normed, *mlp_args)
    /// ```
    ///
    /// with the post-lambdas folded onto the projection weights ([`scaled_linear`]) and the
    /// remaining two products issued as `addcmul`, so each residual line is ONE kernel over
    /// the residual stream instead of a multiply and an add. The attention line costs two
    /// (the x0 term); the feedforward line costs exactly what the old bare `state + ff` cost.
    ///
    /// `first_value` is the SOURCE layer's head-shaped value, `None` in the source layer itself.
    /// Returns the block output beside the value it published, which is `Some` exactly in the
    /// source layer.
    fn forward(
        &self,
        input: &Tensor,
        x0: &Tensor,
        first_value: Option<&Tensor>,
        lambdas: &BlockLambdas<'_>,
        rotation: (&Tensor, &Tensor),
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        assert_eq!(
            self.value_lambda.is_some(),
            first_value.is_some(),
            "every non-source layer must receive the source layer's value"
        );
        let (batch, length, _) = input.size3().unwrap();
        let head_dim = self.width / self.heads;
        // ONE `split_with_sizes`, not two `narrow`s. ATen's backward for `narrow` is
        // `zeros_like(input)` plus a copy into the slice, so splitting the packed projection
        // with narrows would zero-fill 295 MB twice per layer at batch 256 and then reduce
        // them; `split_with_sizes` records one node that scatters both gradients into a single
        // buffer. Forward is views either way, and `q‖k` stays one tensor so the rotation can
        // run as full-width products over it.
        let packed = linear(&rms_norm(input), &self.qkv)
            .split_with_sizes([2 * self.width, self.width], -1);
        // QK-norm BEFORE the rotation, which is the reference's order:
        // `train_gpt.py:1106` `q, k = norm(q), norm(k)  # QK norm @Grad62304977` and only then
        // `:1109` `q, k = yarn.rotary(q), yarn.rotary(k)`; identically
        // `train_gpt_medium.py:968` before `:970`, and in-repo `world_model.rs:1396-1397`
        // feeding `:1722-1724`. RoPE rotates each `(r, r+half)` pair, so it preserves the
        // head-dim norm exactly and the two orders agree in exact arithmetic - but not in
        // bf16, and not in what the rotation's inputs look like: normalizing first is what
        // keeps the rotary products O(1), which is the entire point of QK-norm. The V path is
        // deliberately NOT normalized (`packed[1]` goes straight to the mix and then to SDPA),
        // matching the reference, where `norm` touches only `q, k` - so both operands of the
        // value-residual mix below are unnormalized, as they are upstream.
        //
        // ONE kernel in each direction for the whole normalize-then-rotate pair:
        // `fused_kernels::qk_norm_rope` reads the RAW packed `q‖k` block, normalizes each
        // `head_dim` row of the free `[batch, length, 2·heads, head_dim]` view (column
        // `t·heads·head_dim + h·head_dim + d` already carries tensor `t`, head `h`, dim `d`),
        // rotates it against the untiled `[origins, head_dim/2]` rows and writes the
        // `[batch, length, 2, heads, head_dim]` buffer. The normalized block is never
        // materialized and no `rstd` is written or read back - the backward recomputes the
        // normalization from the raw block, which is the cheap half of a memory-bound kernel.
        // Bit-identical to `_fused_rms_norm`-then-`fused_kernels::rope` in both directions.
        let rotated =
            qk_norm_rope(&packed[0], rotation.0, rotation.1, self.heads).split(1, 2);
        // Value residual. `packed[1]` is the layer's own value, head-shaped by a VIEW (splitting
        // the last dimension is always expressible as a stride, so this costs nothing); the mix
        // is one `lerp` - a single read of each operand and one write - rather than the
        // reference's literal `(1-λ)·v + λ·v_1`, which is three kernels and twice the traffic
        // for the same value. `lerp` also makes the endpoints EXACT: ATen selects
        // `self + w·(end - self)` for `|w| < 0.5` and `end - (end - self)·(1 - w)` otherwise, so
        // λ = 0 returns this layer's value bit-for-bit and λ = 1 returns the source layer's.
        // The lambda is cast to the activation dtype first: an fp32 `[1]` operand would promote
        // the whole bf16 mix to fp32 and double every byte it moves.
        let value = packed[1].reshape([batch, length, self.heads, head_dim]);
        let value = match (&self.value_lambda, first_value) {
            (Some(lambda), Some(first)) => value.lerp_tensor(first, &lambda.to_kind(value.kind())),
            _ => value,
        };
        let query_key = |tensor: &Tensor| tensor.squeeze_dim(2).transpose(1, 2);
        // Every attention input is a strided VIEW of the projection or of the rotation buffer,
        // with `head_dim` contiguous: SDPA needs only unit stride on the last dimension, so
        // materializing contiguous q/k/v would be three 197 MB copies a layer for nothing.
        let attended = Tensor::scaled_dot_product_attention(
            &query_key(&rotated[0]),
            &query_key(&rotated[1]),
            &value.transpose(1, 2),
            None::<&Tensor>,
            if train { self.dropout } else { 0.0 },
            true,
            None,
            false,
        )
        // Structurally required, not ours: SDPA emits `[batch, heads, length, head_dim]` and
        // the output projection contracts over `heads·head_dim`, so the head and length axes
        // must be swapped before they can be merged, and no stride expresses that merge.
        .transpose(1, 2)
        .reshape([batch, length, self.width]);
        let state = scaled_linear(&attended, &self.output, lambdas.post[0])
            .dropout(self.dropout, train)
            .addcmul(input, lambdas.resid[0])
            .addcmul(x0, lambdas.x0);
        // ReLU² instead of GELU: `train_gpt_medium.py:1010`,
        // `x = F.relu(x).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than
        // GELU`, and the fused `relu(x @ W1.T)^2 @ W2.T` kernel of `train_gpt.py:46`. The
        // reference applies NO output-scale correction for the change of activation: the
        // down projection is zero-initialised and every consumer of the residual stream is
        // RMS-normalized, so the activation's second moment is absorbed rather than
        // compensated. Its hidden width is `4 * model_dim` (`train_gpt.py:1299`), which is
        // exactly our fixed 2048 at `d_model` 512, so there is no width mismatch to correct
        // either. ONE kernel in each direction: `fused_kernels::relu_square` is the same
        // fusion the reference has as a Triton kernel, bit-identical to `relu().square()`
        // down to the NaN conventions, and it removes the second full-width pass over the
        // `[tokens, ffn]` hidden activation - the largest single traffic cost of this recipe.
        let ff = scaled_linear(
            &relu_square(&linear(&rms_norm(&state), &self.first)).dropout(self.dropout, train),
            &self.second,
            lambdas.post[1],
        )
        .dropout(self.dropout, train);
        // Only the source layer publishes; a later layer's value is already the mix.
        (
            ff.addcmul(&state, lambdas.resid[1]),
            self.value_lambda.is_none().then(|| value.shallow_clone()),
        )
    }

    /// This block's learned mixing scalars, as `(kind, device tensor)`. The KIND only - the
    /// layer index is the caller's, so a block never has to know where in the stack it sits.
    /// Tensors, not `f64`: [`CausalPatchModel::recipe_scalars`] copies the whole stack in one
    /// host transfer, and the chart is written once per report interval, never per step.
    fn recipe_scalars(&self) -> Vec<(&'static str, &Tensor)> {
        self.value_lambda
            .iter()
            .map(|lambda| ("value lambda", lambda))
            .collect()
    }
}

/// The U-net encoder/decoder layer loop: the first half of the layers push their output onto a
/// stack, the second half pop one and fold it into the residual stream BEFORE their own compute,
/// which pairs the deepest unconsumed encoder layer with the shallowest decoder layer -
/// `3->4, 2->5, 1->6, 0->7` at eight layers.
///
/// Straight from modded-nanogpt, `records/track_1_short/2024-11-10_UNetDoubleLr/
/// c87bb826-797b-4f37-98c7-d3a5dad2de74.txt:257-270` (@brendanh0gan): encoder outputs are
/// appended, then `self.transformer.h[encoder_layers + i](x + skip_weights[i] * pop(), ...)`.
/// The current `train_gpt.py:1591-1595` keeps the same shape - `x = x + gate * cache[3]` ahead
/// of layer 6's compute - having pruned the twelve-layer U down to the single pair that paid
/// for itself there; we keep the full U because eight layers is not twelve and because the
/// pruning was a wall-clock decision on a 3-minute run, not a loss result.
///
/// `gates` are the POST-sigmoid gates, one per decoder layer, already in the residual stream's
/// dtype: the reference casts its residual scalars to bf16 for exactly this multiply
/// (`train_gpt.py:1509-1512`), and an fp32 gate would promote a 94 MB bf16 activation to fp32.
///
/// The fold is one `addcmul`, not a `gate * skip` product followed by an add: `addcmul`'s
/// backward saves only the two factors - both already retained - so the fused form materializes
/// one `[tokens, width]` tensor per pair instead of two and never keeps the product alive.
///
/// `layer` is the per-index compute so this loop can be exercised with distinguishable stub
/// layers; the model's forward passes the real blocks.
fn unet_stack<F>(mut state: Tensor, layers: usize, gates: &[Tensor], mut layer: F) -> Tensor
where
    F: FnMut(usize, &Tensor) -> Tensor,
{
    let encoder = layers / 2;
    assert_eq!(
        gates.len(),
        layers - encoder,
        "one skip gate per decoder layer"
    );
    let mut skips: Vec<Tensor> = Vec::with_capacity(encoder);
    for index in 0..layers {
        if index >= encoder {
            let skip = skips
                .pop()
                .expect("every decoder layer consumes one encoder output");
            state = state.addcmul(&skip, &gates[index - encoder]);
        }
        state = layer(index, &state);
        if index < encoder {
            // A reference, not a copy: this tensor is the next layer's input either way, so the
            // stack costs four pointers and no bytes.
            skips.push(state.shallow_clone());
        }
    }
    state
}

/// Causal per-origin statistics, `[batch, origins]` in fp32. Origin `k` is the last bar of
/// patch `k`; every value uses only bars at or before it.
pub struct Statistics {
    /// Expanding population std of valid close-to-close log returns, variance floored.
    pub sigma: Tensor,
    /// Expanding mean relative range `(high - low) / low`; falls back to σ when flat.
    pub range: Tensor,
    /// Origin close log-price relative to the row anchor.
    pub log_close: Tensor,
    /// Cumulative market log return at the origin, relative to the row's last context bar.
    pub market: Tensor,
    /// Causal ridge slope of the ticker's returns on the market steps, shrunk toward 1.
    pub beta: Tensor,
    /// 1 where at least `min_history` valid bars precede the origin inclusive.
    pub mask: Tensor,
}

impl Statistics {
    pub fn last(&self) -> Self {
        let last = self.sigma.size()[1] - 1;
        Self {
            sigma: self.sigma.narrow(1, last, 1),
            range: self.range.narrow(1, last, 1),
            log_close: self.log_close.narrow(1, last, 1),
            market: self.market.narrow(1, last, 1),
            beta: self.beta.narrow(1, last, 1),
            mask: self.mask.narrow(1, last, 1),
        }
    }
}

fn per_bar(statistic: &Tensor) -> Tensor {
    statistic.unsqueeze(-1).unsqueeze(-1)
}

/// Raw dense-head output, channel-major bf16 `[rows, origins', 2·CHANNELS, pred_len]`: the four
/// candle coordinates then the four log predictive scales, with the μP output multiplier already
/// folded into the head weight.
///
/// Channel-major is load-bearing, not cosmetic. Every consumer of this tensor slices ONE
/// channel; in the natural `[.., pred_len, 2·CHANNELS]` layout each such slice is a stride-8
/// gather that pulls a 32-byte sector per two useful bytes, and the dense head emits
/// `256 · 375 · 192 · 8` = 147 M elements per step. Here a channel slice is contiguous.
pub struct Head(Tensor);

impl Head {
    /// The `2·CHANNELS` channel slices, `[rows, origins', 1, pred_len]` each, produced by ONE
    /// `split` node: backward scatters the eight gradients into a single buffer instead of
    /// reducing eight zero-padded 295 MB ones.
    fn channels(&self) -> Vec<Tensor> {
        self.0.split(1, 2)
    }
}

/// Head outputs per (origin, future bar) in fp32, channel-major `[.., CHANNELS, pred_len]`:
/// candle coordinates and the log predictive scale per OHLC channel in σ units, the latter
/// already carrying the `½·ln h` random-walk prior. Materialized for the evaluation decoders
/// only; the training loss consumes [`Head`] directly.
pub struct Output {
    pub coordinates: Tensor,
    pub log_scale: Tensor,
}

/// One class of backbone kernel, as [`CausalPatchModel::kernel_classes`] hands it to
/// `benchmark --profile`: the ops themselves plus the bytes and arithmetic they cannot avoid.
pub struct KernelClass<'a> {
    pub name: &'static str,
    /// Shape and dtype of every tensor `run` consumes. The CALLER allocates them, so allocation
    /// and the first-touch page faults stay outside the timed region; entry 0 is the
    /// differentiable input backward is measured against, and an empty list means the class
    /// carries no gradient at all.
    pub inputs: Vec<(Vec<i64>, Kind)>,
    /// Bytes ONE forward pass must move: every input read once, every output written once. A
    /// floor, so `forward_bytes / measured seconds` is a floor on the class's bandwidth.
    pub forward_bytes: f64,
    /// Arithmetic ONE forward pass must do.
    pub forward_flops: f64,
    /// Leaf parameters whose gradients this class's backward must also produce. Without them
    /// `Tensor::run_backward` prunes the graph to the paths reaching the activation input and
    /// the weight-gradient GEMM - half of a projection's backward - never runs, so a GEMM class
    /// would measure as twice as efficient as it is.
    pub parameters: Vec<Tensor>,
    pub run: Box<dyn Fn(&[Tensor]) -> Tensor + 'a>,
}

/// Decoder-only causal transformer over non-overlapping OHLC+covariate patches with a dense
/// heteroscedastic multi-bar head at every origin.
pub struct CausalPatchModel {
    config: ModelConfig,
    patch: nn::Linear,
    /// Rotary `cos`/`sin` for the fixed position grid, `[origins, head_dim/2]` bf16 and
    /// UNTILED: row `t`, column `r` is the pair-`r` angle at origin `t`. 24 KiB apiece that the
    /// whole stack reads out of L2, because `fused_kernels::rope` indexes these rows directly -
    /// the composed rotation needed them broadcast to a `[1, origins, 2·d_model]` pair of
    /// 768 KiB tiles.
    rotation: (Tensor, Tensor),
    blocks: Vec<Block>,
    /// Per-sub-block residual-stream scale, `[2·layers]` fp32, laid out `[attn_0, ffn_0,
    /// attn_1, ...]`. The reference packs the same numbers as a `[layers, 2]` parameter
    /// (`train_gpt.py:1338`); ONE-dimensional here on purpose - a 2-D parameter is what
    /// NorMuon's router looks for (`optim/muon.rs:1316`), and a 2-D lambda bank would be one
    /// dropped name-allowlist entry away from being orthogonalized as if it were a weight
    /// matrix. 1-D cannot be misrouted by construction.
    resid_lambdas: Tensor,
    /// Per-sub-block branch-output scale, `[2·layers]` fp32, same layout. Folded onto the
    /// output projection weights rather than the activations - see [`scaled_linear`].
    post_lambdas: Tensor,
    /// Per-layer embedding re-injection scale, `[layers]` fp32.
    x0_lambdas: Tensor,
    /// LOGITS of the U-net skip gates, `[layers/2]` fp32, one per decoder layer in
    /// [`ModelConfig::skip_pairs`] order. The gate the residual stream sees is `σ(logit)`, so
    /// it can never leave `(0, 1)` and starts at σ([`SKIP_LOGIT_INIT`]) = 0.18243. Stored on the
    /// root path, 1-D and outside the `block_*` allowlist, which is what routes it to AdamW.
    skip_weights: Tensor,
    covariates: Option<nn::Linear>,
    head_hidden: nn::Linear,
    head_output: nn::Linear,
    known_index: Tensor,
    sigma_scale: Tensor,
    unit_scale: Tensor,
    horizon_scale: Tensor,
    half_log_horizon: Tensor,
    /// `1/h` per future bar, `[1, 1, 1, pred_len]`: `exp(-2·½·ln h)` pulled out of the
    /// per-element NLL so the `½·ln h` prior never enters a full-size kernel.
    inverse_horizon: Tensor,
    /// `1/LOG_SCALE_CAP`, shaped `[1, 1, 1, 1]` rather than 0-dim so that multiplying a bf16
    /// log-scale channel by it promotes the result to fp32 in a single kernel.
    log_scale_gain: Tensor,
}

impl CausalPatchModel {
    pub fn new(path: &nn::Path, config: &ModelConfig) -> Self {
        config
            .validate()
            .expect("invalid causal patch model configuration");
        let device = path.device();
        let head_dim = config.d_model / config.heads;
        let aux_channels = config.features.channels() as i64;
        let known: Vec<i64> = config
            .features
            .channel_mask(Feature::known_future)
            .iter()
            .enumerate()
            .filter_map(|(index, known)| known.then_some(index as i64))
            .collect();
        let sigma_scale: Vec<f32> = config
            .features
            .channel_mask(Feature::sigma_scaled)
            .iter()
            .map(|&scaled| f32::from(u8::from(scaled)))
            .collect();
        let covariates = (!known.is_empty()).then(|| {
            projection(
                path / "covariates",
                config.pred_len * known.len() as i64,
                COVARIATE_WIDTH,
                true,
            )
        });
        let head_input = config.d_model + covariates.as_ref().map_or(0, |_| COVARIATE_WIDTH);
        let horizon = (Tensor::arange(config.pred_len, (Kind::Float, device)) + 1.0)
            .reshape([1, 1, 1, config.pred_len]);
        let sigma_scale = Tensor::from_slice(&sigma_scale).to_device(device);
        // The dense causal-patch model attends over a FIXED position grid, so the rotation rows
        // are constant: building them once removes four small kernels per attention tensor per
        // layer (64 launches per forward), and `fused_kernels::rope` consumes them in exactly
        // this untiled form, so the broadcast tiles the composed rotation needed are never
        // built at all.
        let rope = RotaryEmbedding::new(config.origins(), head_dim, head_dim, device);
        let rotation = rope.cached_rotation(
            &Tensor::arange(config.origins(), (Kind::Int64, device)),
            Kind::BFloat16,
        );
        Self {
            patch: projection(
                path / "patch",
                config.patch_len * (CHANNELS + aux_channels),
                config.d_model,
                true,
            ),
            rotation,
            blocks: (0..config.layers)
                .map(|index| Block::new(path / format!("block_{index}"), config, index))
                .collect(),
            // `lambdas.*`, NOT `block_*`: the NorMuon allowlist is `["block_"]`
            // (`compute.rs:115`) and the routing assertion at `compute.rs:141-152` demands
            // that every NorMuon tensor be a 2-D `block_*.weight`. These are 1-D and outside
            // the allowlist, so they land in AdamW on both counts, and the `lambda` fragment
            // in `adamw_no_weight_decay_name_substrings` keeps weight decay off them - the
            // reference gives all three `wd_mul = 0` (`train_gpt.py:2035-2036`,
            // `train_gpt_medium.py:1081`).
            resid_lambdas: (path / "lambdas").var(
                "resid",
                &[2 * config.layers as i64],
                nn::Init::Const(RESID_LAMBDA_INIT),
            ),
            post_lambdas: (path / "lambdas").var(
                "post",
                &[2 * config.layers as i64],
                nn::Init::Const(POST_LAMBDA_INIT),
            ),
            x0_lambdas: (path / "lambdas").var(
                "x0",
                &[config.layers as i64],
                nn::Init::Const(X0_LAMBDA_INIT),
            ),
            skip_weights: path.var(
                "skip_weights",
                &[(config.layers / 2) as i64],
                nn::Init::Const(SKIP_LOGIT_INIT),
            ),
            covariates,
            head_hidden: projection(path / "head" / "hidden", head_input, HEAD_HIDDEN, true),
            head_output: nn::linear(
                path / "head" / "output",
                HEAD_HIDDEN,
                config.pred_len * OUTPUTS_PER_BAR,
                nn::LinearConfig {
                    ws_init: nn::Init::Const(0.0),
                    bs_init: Some(nn::Init::Const(0.0)),
                    bias: true,
                },
            ),
            known_index: Tensor::from_slice(&known).to_device(device),
            unit_scale: 1.0 - &sigma_scale,
            sigma_scale,
            horizon_scale: horizon.sqrt(),
            half_log_horizon: horizon.log() * 0.5,
            inverse_horizon: horizon.reciprocal(),
            log_scale_gain: Tensor::full(
                [1, 1, 1, 1],
                1.0 / LOG_SCALE_CAP,
                (Kind::Float, device),
            ),
            config: config.clone(),
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn horizon_scale(&self) -> &Tensor {
        &self.horizon_scale
    }

    pub fn half_log_horizon(&self) -> &Tensor {
        &self.half_log_horizon
    }

    pub fn statistics(&self, batch: &Batch) -> Statistics {
        let c = &self.config;
        let (context, patch, origins) = (c.seq_len, c.patch_len, c.origins());
        let log_prices = batch.log_prices.narrow(1, 0, context);
        let valid = batch.valid.narrow(1, 0, context);
        let rows = log_prices.size()[0];
        let close = log_prices.select(2, 3);
        let pair = valid.narrow(1, 1, context - 1) * valid.narrow(1, 0, context - 1);
        let returns = (close.narrow(1, 1, context - 1) - close.narrow(1, 0, context - 1)) * &pair;
        let lead = Tensor::zeros([rows, 1], (Kind::Float, log_prices.device()));
        let at_origin = |series: &Tensor| series.reshape([rows, origins, patch]).select(2, patch - 1);
        let cumulative = |series: &Tensor| {
            at_origin(&Tensor::cat(&[&lead, series], 1).cumsum(1, Kind::Float))
        };
        let pairs = cumulative(&pair).clamp_min(1.0);
        let mean = cumulative(&returns) / &pairs;
        let variance = cumulative(&returns.square()) / &pairs - mean.square();
        let sigma = (variance.clamp_min(0.0) + RETURN_VARIANCE_FLOOR).sqrt();
        let market = batch.market_cum.narrow(1, 0, context);
        let steps = (market.narrow(1, 1, context - 1) - market.narrow(1, 0, context - 1)) * &pair;
        let market_squares = cumulative(&steps.square());
        let ridge = (&market_squares / &pairs).clamp_min(RETURN_VARIANCE_FLOOR) * BETA_PRIOR_BARS;
        let beta = (cumulative(&(&returns * &steps)) + &ridge) / (market_squares + ridge);
        let bars = at_origin(&valid.cumsum(1, Kind::Float));
        let spread = ((log_prices.select(2, 1) - log_prices.select(2, 2)).exp() - 1.0) * &valid;
        let range = at_origin(&spread.cumsum(1, Kind::Float)) / bars.clamp_min(1.0);
        Statistics {
            range: range.where_self(&range.gt(0.0), &sigma),
            sigma,
            log_close: at_origin(&close),
            market: at_origin(&market),
            beta,
            mask: bars.ge(c.min_history as f64).to_kind(Kind::Float),
        }
    }

    /// Windows covering bars `t_k + 1 ..= t_k + pred_len` of a `[batch, seq_len + pred_len, ...]`
    /// series for every origin or the final origin only, channel-major:
    /// `[batch, origins', channels, pred_len]` for a 3-D series and `[batch, origins', pred_len]`
    /// for a 2-D one. Channel-major is what `unfold` produces natively and what every consumer
    /// slices along, so no transpose survives here.
    fn future_windows(&self, series: &Tensor, last_only: bool) -> Tensor {
        let c = &self.config;
        if last_only {
            let window = series.narrow(1, c.seq_len, c.pred_len).unsqueeze(1);
            return if series.dim() == 2 {
                window
            } else {
                window.transpose(2, 3)
            };
        }
        series
            .narrow(1, c.patch_len, c.seq_len + c.pred_len - c.patch_len)
            .unfold(1, c.pred_len, c.patch_len)
    }

    /// `stats` must be the full statistics; `last_only` restricts the head to the final origin.
    pub fn forward(&self, batch: &Batch, stats: &Statistics, train: bool, last_only: bool) -> Head {
        self.head(batch, &self.backbone(batch, stats, train, last_only), last_only)
    }

    /// The σ-normalised patch tokens the backbone embeds, bf16
    /// `[rows, origins, patch_len·(CHANNELS + aux)]`.
    ///
    /// The normalisation itself must be fp32: subtracting the origin close from a log price is a
    /// cancellation, and σ is a small fp32 statistic. But the CAST happens per part, BEFORE the
    /// concatenation, not after it. Casting after made the concatenation write and re-read a
    /// 98 MB fp32 tensor at batch 256 for a 49 MB bf16 result; casting first is bit-identical
    /// (an elementwise cast commutes with concatenation exactly) and moves 197 MB less.
    ///
    /// Nothing here carries a gradient - prices, auxiliaries and statistics are all data - so
    /// none of these fp32 intermediates is retained past the cast.
    pub fn tokens(&self, batch: &Batch, stats: &Statistics) -> Tensor {
        let c = &self.config;
        let (context, patch, horizon, origins) = (c.seq_len, c.patch_len, c.pred_len, c.origins());
        let aux_channels = c.features.channels() as i64;
        let rows = batch.log_prices.size()[0];
        assert_eq!(batch.log_prices.size(), [rows, context + horizon, CHANNELS]);
        assert_eq!(batch.aux.size(), [rows, context + horizon, aux_channels]);
        assert_eq!(stats.sigma.size(), [rows, origins]);
        let inv_sigma = per_bar(&stats.sigma.reciprocal());
        let prices = ((batch
            .log_prices
            .narrow(1, 0, context)
            .reshape([rows, origins, patch, CHANNELS])
            - per_bar(&stats.log_close))
            * &inv_sigma)
            .to_kind(Kind::BFloat16);
        let aux = (batch
            .aux
            .narrow(1, 0, context)
            .reshape([rows, origins, patch, aux_channels])
            * (&self.sigma_scale * inv_sigma + &self.unit_scale))
            .to_kind(Kind::BFloat16);
        Tensor::cat(&[prices, aux], 3).reshape([
            rows,
            origins,
            patch * (CHANNELS + aux_channels),
        ])
    }

    /// Patch embedding, the causal transformer stack and the final norm, narrowed to the scored
    /// origins. Separated from [`Self::head`] so the step timing can attribute the two phases
    /// without a synchronization inside the composed forward.
    ///
    /// `x0` is the normalized patch embedding, which is what the reference re-injects:
    /// `train_gpt.py:1549` `x = x0 = norm(x[None])` (and `train_gpt_medium.py:1158`), so the
    /// embedding is normalized ONCE here rather than only inside block 0's pre-norm. Every
    /// consumer of the residual stream is a gainless RMSNorm, so this changes nothing about
    /// block 0's input; it makes `x0` a unit-RMS tensor that the per-layer `x0_lambda` can
    /// mix in on a known scale.
    pub fn backbone(
        &self,
        batch: &Batch,
        stats: &Statistics,
        train: bool,
        last_only: bool,
    ) -> Tensor {
        let origins = self.config.origins();
        let x0 = rms_norm(
            &linear(&self.tokens(batch, stats), &self.patch).dropout(self.config.dropout, train),
        );
        // ONE cast and ONE `unbind` for the whole stack, as the reference does
        // (`train_gpt.py:1509-1512`, `self.resid_lambdas[:, 0].bfloat16().unbind(0)`). The cast
        // is mandatory, not cosmetic: this model runs without autocast, so a DIMENSIONED fp32
        // scalar against a bf16 activation would promote the whole result to fp32 and double
        // the residual stream's traffic (`docs/timexer_segment.md:16`). `unbind` yields 0-dim
        // views, which broadcast against anything.
        let kind = x0.kind();
        let resid = self.resid_lambdas.to_kind(kind).unbind(0);
        let post = self.post_lambdas.to_kind(kind).unbind(0);
        let x0_lambdas = self.x0_lambdas.to_kind(kind).unbind(0);
        // The U-net gates ride the same one-cast rule: POST-sigmoid, in the activation dtype.
        // One `unbind`, not `layers/2` `select`s - `select`'s backward is `zeros_like` plus a
        // copy, `unbind`'s is a single `stack`.
        let gates = self.skip_weights.sigmoid().to_kind(kind).unbind(0);
        // Layer 0's raw value, published once and mixed into every deeper layer's V. `None`
        // until layer 0 returns it, which is also why layer 0 owns no `value_lambda`.
        let mut first_value: Option<Tensor> = None;
        let state = unet_stack(
            x0.shallow_clone(),
            self.blocks.len(),
            &gates,
            |index, state| {
                let lambdas = BlockLambdas {
                    resid: [&resid[2 * index], &resid[2 * index + 1]],
                    post: [&post[2 * index], &post[2 * index + 1]],
                    x0: &x0_lambdas[index],
                };
                let (next, published) = self.blocks[index].forward(
                    state,
                    &x0,
                    first_value.as_ref(),
                    &lambdas,
                    (&self.rotation.0, &self.rotation.1),
                    train,
                );
                if let Some(value) = published {
                    first_value = Some(value);
                }
                next
            },
        );
        let state = rms_norm(&state);
        if last_only {
            state.narrow(1, origins - 1, 1)
        } else {
            state
        }
    }

    /// Every learned mixing scalar the backbone holds, as `(label, post-parameterization
    /// value)`, for the shared `timexer_segment_recipe_scalars` report base.
    ///
    /// Values are what the forward pass actually applies: the residual, post, x0 and value
    /// lambdas are stored raw and applied raw (`train_gpt.py:1334`, `:1338`,
    /// `train_gpt_medium.py:1078`, and the value residual's
    /// `records/track_1_short/2024-11-06_ShortcutsTweaks/…txt:168`), so they report raw; the
    /// U-net skip is stored as a LOGIT and applied as `σ(logit)`, so it reports the sigmoid -
    /// the question a reader asks of this panel is whether a gate moved off its 0.18 init, and
    /// a logit does not answer it.
    ///
    /// ONE host transfer for the whole set: every part is concatenated on device and copied
    /// once. Called per report interval, never per step - it synchronizes - and deliberately
    /// not `.item()` per scalar, which would be one synchronization each.
    pub fn recipe_scalars(&self) -> Vec<(String, f64)> {
        let (names, values) = self.recipe_scalar_parts();
        if names.is_empty() {
            return Vec::new();
        }
        let flat = Tensor::cat(&values, 0)
            .to_kind(Kind::Double)
            .to_device(Device::Cpu);
        assert_eq!(
            names.len() as i64,
            flat.numel() as i64,
            "every recipe scalar name must have exactly one value"
        );
        let values = Vec::<f64>::try_from(flat).expect("recipe scalars are a 1-D fp64 vector");
        names.into_iter().zip(values).collect()
    }

    /// The contributions to [`Self::recipe_scalars`], names and device-side values kept apart so
    /// that adding a family of scalars costs one `push` here and no host round-trip of its own.
    /// The names must be pushed in the same order as the tensors they label, which
    /// [`Self::recipe_scalars`]'s length assertion is there to catch.
    fn recipe_scalar_parts(&self) -> (Vec<String>, Vec<Tensor>) {
        let layers = self.config.layers;
        let mut names = Vec::new();
        let mut values = Vec::new();
        // Per sub-block, in the banks' own `[attn_0, ffn_0, attn_1, …]` layout.
        names.extend((0..layers).flat_map(|layer| {
            [
                format!("residual lambda L{layer} attn"),
                format!("residual lambda L{layer} ffn"),
            ]
        }));
        values.push(self.resid_lambdas.shallow_clone());
        names.extend((0..layers).flat_map(|layer| {
            [
                format!("post lambda L{layer} attn"),
                format!("post lambda L{layer} ffn"),
            ]
        }));
        values.push(self.post_lambdas.shallow_clone());
        names.extend((0..layers).map(|layer| format!("x0 lambda L{layer}")));
        values.push(self.x0_lambdas.shallow_clone());
        // POST-sigmoid, in `skip_pairs` order - the same order the gates are unbound in.
        names.extend(
            self.config
                .skip_pairs()
                .map(|(source, destination)| format!("skip weight {source}->{destination}")),
        );
        values.push(self.skip_weights.sigmoid());
        // The value residual, per layer that owns one: layer 0 is the source and has none.
        for (layer, block) in self.blocks.iter().enumerate() {
            for (kind, scalar) in block.recipe_scalars() {
                names.push(format!("{kind} L{layer}"));
                values.push(scalar.reshape([1]));
            }
        }
        (names, values)
    }

    /// The backbone's kernel classes at the real per-layer shapes: what `benchmark --profile`
    /// times with CUDA events, one entry per class of kernel the step issues.
    ///
    /// Every closure calls the SAME function the forward path calls - `Block::rotate`,
    /// `rms_norm`, `linear`, `Self::tokens` - so a class cannot drift from the model
    /// without the compiler noticing. The last entry is the composed layer, so the sum of the
    /// parts can be compared against the whole and the attribution error stated rather than
    /// assumed.
    pub fn kernel_classes<'a>(
        &'a self,
        batch: &'a Batch,
        stats: &'a Statistics,
        train: bool,
    ) -> Vec<KernelClass<'a>> {
        let c = &self.config;
        let (origins, width, ffn) = (c.origins(), c.d_model, c.ffn);
        let (heads, head_dim) = (c.heads, c.d_model / c.heads);
        let rows = batch.log_prices.size()[0];
        let block = &self.blocks[0];
        let rotation = (&self.rotation.0, &self.rotation.1);
        let tokens = (rows * origins) as f64;
        let (bf16, fp32) = (|n: f64| 2. * n, |n: f64| 4. * n);
        // One `[tokens, d_model]` bf16 activation, one `[tokens, ffn]` one, and the fp32
        // `rstd` a gainless RMSNorm writes - one reduction output per row, where the LayerNorm
        // this replaced wrote a `mean`/`rstd` PAIR.
        let state = bf16(tokens * width as f64);
        let hidden = bf16(tokens * ffn as f64);
        let rstd = |rows_of: f64| fp32(rows_of);
        let gemm = |reduce: i64, out: i64| 2. * tokens * reduce as f64 * out as f64;
        // fp32 master read plus bf16 copy written, per parameter the class casts.
        let cast = |elements: i64| fp32(elements as f64) + bf16(elements as f64);
        let activation = |last: i64| (vec![rows, origins, last], Kind::BFloat16);
        let projected = |layer: &nn::Linear| {
            let mut params = vec![layer.ws.shallow_clone()];
            params.extend(layer.bs.iter().map(Tensor::shallow_clone));
            params
        };
        let mut classes = vec![
            KernelClass {
                name: "RMSNorm",
                inputs: vec![activation(width)],
                // No parameter cast (gainless) and one reduction output instead of two: the
                // LayerNorm this replaced charged `2·state + fp32(2·tokens) + cast(2·width)`.
                forward_bytes: 2. * state + rstd(tokens),
                // `x²` summed, one `rsqrt`, one multiply: 3 per element against LayerNorm's 8
                // (mean, centred square, rsqrt, scale, gain, bias).
                forward_flops: 3. * tokens * width as f64,
                parameters: Vec::new(),
                run: Box::new(move |input| rms_norm(&input[0])),
            },
            KernelClass {
                name: "QK norm + rotary",
                // The RAW packed `q‖k` block: ONE kernel normalizes each of its `2·heads` rows
                // of `head_dim` per token AND rotates them, so this charges `4·state` - read the
                // block, write the rotated buffer. Nothing else crosses HBM: the normalized
                // block is never materialized, no `rstd` is written (the backward recomputes the
                // normalization from the raw block), and the untiled `[origins, head_dim/2]`
                // rotation rows are 24 KiB of L2. The composition charged `4·state +
                // fp32(2·heads·tokens)` for the norm and another `4·state` for the rotation, on
                // top of the `18·state` the pre-kernel composed rotation cost.
                inputs: vec![activation(2 * width)],
                forward_bytes: 4. * state,
                // The norm's three per element over `2·width` plus the rotation's six per
                // `width`: one product and one sum for each of the two half-crossing terms.
                forward_flops: 3. * tokens * 2. * width as f64 + 6. * tokens * width as f64,
                parameters: Vec::new(),
                run: Box::new(move |input| qk_norm_rope(&input[0], rotation.0, rotation.1, heads)),
            },
            KernelClass {
                name: "QKV projection",
                inputs: vec![activation(width)],
                forward_bytes: 4. * state + cast(3 * width * width),
                forward_flops: gemm(width, 3 * width),
                parameters: projected(&block.qkv),
                run: Box::new(move |input| linear(&input[0], &block.qkv)),
            },
            KernelClass {
                name: "causal SDPA",
                // The rotation buffer and the projection, exactly as `Block::forward` holds
                // them: q/k stride over 2·d_model, v over 3·d_model, both unit on head_dim.
                inputs: vec![activation(2 * width), activation(3 * width)],
                forward_bytes: 4. * state + fp32((rows * heads * origins) as f64),
                // `QKᵀ` and `AV` are each a dense `2·rows·origins²·d_model` and half of that
                // under the causal mask, so the pair costs one dense GEMM's worth.
                forward_flops: 2. * (rows * origins * origins * width) as f64,
                parameters: Vec::new(),
                run: Box::new(move |input| {
                    let rotated = input[0]
                        .reshape([rows, origins, 2, heads, head_dim])
                        .split(1, 2);
                    let query_key = |tensor: &Tensor| tensor.squeeze_dim(2).transpose(1, 2);
                    Tensor::scaled_dot_product_attention(
                        &query_key(&rotated[0]),
                        &query_key(&rotated[1]),
                        &input[1]
                            .narrow(-1, 2 * width, width)
                            .reshape([rows, origins, heads, head_dim])
                            .transpose(1, 2),
                        None::<&Tensor>,
                        if train { block.dropout } else { 0.0 },
                        true,
                        None,
                        false,
                    )
                }),
            },
            KernelClass {
                name: "attention output flatten",
                inputs: vec![(vec![rows, heads, origins, head_dim], Kind::BFloat16)],
                forward_bytes: 2. * state,
                forward_flops: 0.,
                parameters: Vec::new(),
                run: Box::new(move |input| {
                    input[0].transpose(1, 2).reshape([rows, origins, width])
                }),
            },
            KernelClass {
                name: "attention output projection",
                inputs: vec![activation(width)],
                // The post-lambda is folded onto the weight COPY the cast already makes, so
                // the extra traffic is one 512×512 bf16 read plus one write - 1/96 000 of the
                // activation it would otherwise scale (see [`scaled_linear`]).
                forward_bytes: 2. * state
                    + cast(width * width)
                    + bf16(2. * (width * width) as f64),
                forward_flops: gemm(width, width),
                parameters: projected(&block.output),
                run: Box::new(move |input| {
                    scaled_linear(
                        &input[0],
                        &block.output,
                        &self.post_lambdas.get(0).to_kind(input[0].kind()),
                    )
                }),
            },
            KernelClass {
                name: "residual addcmul",
                // `out + x·λ` in ONE kernel: two reads and a write, exactly what the bare
                // `out + x` it replaces cost. The residual scale is free; only the x0
                // injection is a second invocation.
                inputs: vec![activation(width), activation(width)],
                forward_bytes: 3. * state,
                forward_flops: 2. * tokens * width as f64,
                parameters: Vec::new(),
                run: Box::new(move |input| {
                    input[0].addcmul(&input[1], &self.resid_lambdas.get(0).to_kind(input[0].kind()))
                }),
            },
            KernelClass {
                name: "FFN up projection",
                inputs: vec![activation(width)],
                forward_bytes: state + hidden + cast(ffn * width),
                forward_flops: gemm(width, ffn),
                parameters: projected(&block.first),
                run: Box::new(move |input| linear(&input[0], &block.first)),
            },
            KernelClass {
                name: "ReLU^2",
                inputs: vec![activation(ffn)],
                // ONE kernel: `fused_kernels::relu_square` writes its result in a single pass,
                // so this charges `2·hidden` - what GELU charged - instead of the `4·hidden`
                // the `relu`-then-`square` composition charged, and the extra materialized
                // `[tokens, ffn]` tensor that was the single largest traffic cost of the
                // residual recipe does not exist. Same fusion the reference has as a Triton
                // kernel (`train_gpt.py:46-48`, `relu(x @ W1.T)^2 @ W2.T`), and ours is
                // bit-identical to the composition it replaces in both directions.
                forward_bytes: 2. * hidden,
                forward_flops: 2. * tokens * ffn as f64,
                parameters: Vec::new(),
                run: Box::new(move |input| relu_square(&input[0])),
            },
            KernelClass {
                name: "FFN down projection",
                inputs: vec![activation(ffn)],
                forward_bytes: hidden + state + cast(width * ffn) + bf16(2. * (width * ffn) as f64),
                forward_flops: gemm(ffn, width),
                parameters: projected(&block.second),
                run: Box::new(move |input| {
                    scaled_linear(
                        &input[0],
                        &block.second,
                        &self.post_lambdas.get(1).to_kind(input[0].kind()),
                    )
                }),
            },
        ];
        // The composed layer charges every class it invokes more than once again: the pre-norm
        // runs twice (before attention, before the feedforward) and the residual `addcmul`
        // three times (the residual scale on both sub-blocks, plus the x0 injection on the
        // attention line). The QK norm runs once.
        let charged = |name: &'static str| -> (f64, f64) {
            let class = classes
                .iter()
                .find(|class| class.name == name)
                .expect("kernel class present in the list above");
            (class.forward_bytes, class.forward_flops)
        };
        let (norm_bytes, norm_flops) = charged("RMSNorm");
        let (addcmul_bytes, addcmul_flops) = charged("residual addcmul");
        let layer_bytes: f64 = classes.iter().map(|class| class.forward_bytes).sum::<f64>()
            + norm_bytes
            + 2. * addcmul_bytes;
        let layer_flops: f64 = classes.iter().map(|class| class.forward_flops).sum::<f64>()
            + norm_flops
            + 2. * addcmul_flops;
        classes.push(KernelClass {
            name: "composed layer",
            // Entry 1 is `x0`, the normalized patch embedding the block re-injects.
            inputs: vec![activation(width), activation(width)],
            forward_bytes: layer_bytes,
            forward_flops: layer_flops,
            // No norm parameters any more: the RMSNorm is gainless and the projections are
            // bias-free, so a block's leaves are four matrices plus the five lambdas.
            parameters: projected(&block.qkv)
                .into_iter()
                .chain(projected(&block.output))
                .chain(projected(&block.first))
                .chain(projected(&block.second))
                .chain([
                    self.resid_lambdas.shallow_clone(),
                    self.post_lambdas.shallow_clone(),
                    self.x0_lambdas.shallow_clone(),
                ])
                .collect(),
            run: Box::new(move |input| {
                let kind = input[0].kind();
                let (resid, post, x0) = (
                    self.resid_lambdas.to_kind(kind),
                    self.post_lambdas.to_kind(kind),
                    self.x0_lambdas.to_kind(kind),
                );
                let (resid_attn, resid_ffn) = (resid.get(0), resid.get(1));
                let (post_attn, post_ffn) = (post.get(0), post.get(1));
                let x0_lambda = x0.get(0);
                let lambdas = BlockLambdas {
                    resid: [&resid_attn, &resid_ffn],
                    post: [&post_attn, &post_ffn],
                    x0: &x0_lambda,
                };
                block
                    .forward(&input[0], &input[1], None, &lambdas, rotation, train)
                    .0
            }),
        });
        // The value-residual mix, at the shape a decoder layer runs it: one `lerp` over two
        // head-shaped values. Its own class rather than part of the composed layer because the
        // source layer does not run it.
        if let Some(decoder) = self.blocks.get(1) {
            let lambda = decoder
                .value_lambda
                .as_ref()
                .expect("layer 1 mixes against layer 0's value");
            classes.push(KernelClass {
                name: "value residual mix",
                inputs: vec![activation(width), activation(width)],
                forward_bytes: 3. * state,
                forward_flops: 3. * tokens * width as f64,
                parameters: vec![lambda.shallow_clone()],
                run: Box::new(move |input| {
                    let head_shaped =
                        |value: &Tensor| value.reshape([rows, origins, heads, head_dim]);
                    head_shaped(&input[0]).lerp_tensor(
                        &head_shaped(&input[1]),
                        &lambda.to_kind(input[0].kind()),
                    )
                }),
            });
        }
        classes.push(KernelClass {
            name: "patch embedding tokens",
            // No gradient: prices, auxiliaries and statistics are all data, so nothing here is
            // retained past the cast and there is no backward pass to charge.
            inputs: Vec::new(),
            forward_bytes: {
                let patch = c.patch_len as f64;
                let aux_channels = c.features.channels() as f64;
                let prices = fp32(tokens * patch * CHANNELS as f64);
                let auxiliaries = fp32(tokens * patch * aux_channels);
                // Prices: subtract the origin close, scale by 1/σ (two fp32 passes, 2 each).
                // Auxiliaries: one fp32 pass. Per-origin auxiliary scale: two small fp32 ops.
                // Then one bf16 cast each (read fp32, write half) and the bf16 concatenation.
                4. * prices
                    + 2. * auxiliaries
                    + 4. * fp32(tokens * aux_channels)
                    + 1.5 * (prices + auxiliaries)
                    + (prices + auxiliaries)
            },
            parameters: Vec::new(),
            forward_flops: tokens * c.patch_len as f64 * (2. * CHANNELS as f64 + c.features.channels() as f64),
            run: Box::new(move |_| self.tokens(batch, stats)),
        });
        // The two forms this work replaced, measured in the SAME process against the same
        // device peaks so the before/after table is a comparison rather than two runs. They are
        // bit-identical to their replacements (`the_packed_rotation_and_split_match_the_per_
        // tensor_reference_bit_for_bit` pins that), so only their cost differs.
        let rope = RotaryEmbedding::new(origins, head_dim, head_dim, self.rotation.0.device());
        let (cosine, sine) = rope.cached_rotation(
            &Tensor::arange(origins, (Kind::Int64, self.rotation.0.device())),
            Kind::BFloat16,
        );
        classes.push(KernelClass {
            name: "reference rotary (per tensor)",
            inputs: vec![activation(2 * width)],
            forward_bytes: 18. * state,
            forward_flops: 6. * tokens * width as f64,
            parameters: Vec::new(),
            run: Box::new(move |input| {
                let parts = input[0].split(width, -1);
                let heads_of = |part: &Tensor| {
                    part.reshape([rows, origins, heads, head_dim]).transpose(1, 2)
                };
                let query = rope.apply_cached(&heads_of(&parts[0]), &cosine, &sine);
                let key = rope.apply_cached(&heads_of(&parts[1]), &cosine, &sine);
                Tensor::stack(&[query, key], 2)
            }),
        });
        classes.push(KernelClass {
            name: "reference embedding (cast after cat)",
            inputs: Vec::new(),
            forward_bytes: classes[classes.len() - 2].forward_bytes,
            forward_flops: classes[classes.len() - 2].forward_flops,
            parameters: Vec::new(),
            run: Box::new(move |_| {
                let (context, patch) = (c.seq_len, c.patch_len);
                let aux_channels = c.features.channels() as i64;
                let inv_sigma = per_bar(&stats.sigma.reciprocal());
                let prices = (batch
                    .log_prices
                    .narrow(1, 0, context)
                    .reshape([rows, origins, patch, CHANNELS])
                    - per_bar(&stats.log_close))
                    * &inv_sigma;
                let auxiliaries = batch
                    .aux
                    .narrow(1, 0, context)
                    .reshape([rows, origins, patch, aux_channels])
                    * (&self.sigma_scale * inv_sigma + &self.unit_scale);
                Tensor::cat(&[prices, auxiliaries], 3)
                    .reshape([rows, origins, patch * (CHANNELS + aux_channels)])
                    .to_kind(Kind::BFloat16)
            }),
        });
        classes
    }

    /// The dense per-origin head: known-future covariates, the two head GEMMs and the raw
    /// channel-major coordinate/log-scale space. `state` comes from [`Self::backbone`] under the
    /// same `last_only` scope.
    pub fn head(&self, batch: &Batch, state: &Tensor, last_only: bool) -> Head {
        let horizon = self.config.pred_len;
        let rows = state.size()[0];
        let head_input = match &self.covariates {
            Some(covariates) => {
                // bf16 BEFORE the gather: `flatten` over the unfolded window dimensions is a
                // copy either way, and casting first halves what it writes. The transpose keeps
                // the historical `(bar, channel)` feature order of the covariate projection.
                let known = self
                    .future_windows(
                        &batch
                            .aux
                            .index_select(2, &self.known_index)
                            .to_kind(Kind::BFloat16),
                        last_only,
                    )
                    .transpose(2, 3)
                    .flatten(2, 3);
                Tensor::cat(&[state, &linear(&known, covariates)], 2)
            }
            None => state.shallow_clone(),
        };
        let hidden = linear(&head_input, &self.head_hidden).gelu("none");
        // The μP output multiplier rides on the weight copy the GEMM already needs: scaling a
        // 1024×1536 weight instead of a 147 M-element output is the same product and the same
        // gradient (`HEAD_OUTPUT_SCALE` is a power of two, so the cast commutes exactly) for
        // 1/96 000 of the traffic.
        Head(
            hidden
                .linear(
                    &(self.head_output.ws.to_kind(hidden.kind()) * HEAD_OUTPUT_SCALE),
                    self.head_output
                        .bs
                        .as_ref()
                        .map(|bias| bias.to_kind(hidden.kind()) * HEAD_OUTPUT_SCALE),
                )
                .reshape([rows, -1, OUTPUTS_PER_BAR, horizon]),
        )
    }

    /// fp32 channel-major coordinates and log predictive scales for the evaluation decoders.
    /// The training loss never calls this: it would materialize the whole 590 MB fp32 space.
    pub fn output(&self, head: &Head) -> Output {
        Output {
            coordinates: head.0.narrow(2, 0, CHANNELS).to_kind(Kind::Float),
            log_scale: (head.0.narrow(2, CHANNELS, CHANNELS).to_kind(Kind::Float)
                / LOG_SCALE_CAP)
                .tanh()
                * LOG_SCALE_CAP
                + &self.half_log_horizon,
        }
    }

    /// Market log return from each origin to bars `t_k+1..=t_k+pred_len`, scaled by the origin's
    /// causal β and in units of its σ, `[batch, origins', 1, pred_len]`; `stats` narrowed like in
    /// [`Self::targets`]. Adding it back to the targets recovers the ticker's own σ-scaled return.
    pub fn market_drift(&self, batch: &Batch, stats: &Statistics, last_only: bool) -> Tensor {
        let future = self.future_windows(&batch.market_cum, last_only);
        assert_eq!(future.size()[1], stats.sigma.size()[1]);
        ((future - stats.market.unsqueeze(-1)) * (&stats.beta / &stats.sigma).unsqueeze(-1))
            .unsqueeze(2)
    }

    /// Market-neutral σ-scaled log-return targets relative to each origin close and their
    /// validity mask, `[batch, origins', 4, pred_len]` and `[batch, origins', 1, pred_len]`: the
    /// ticker's log return minus β_k times the cumulative market log return over the same bars,
    /// divided by the ticker's own causal σ (the ticker's exposure to the market drift over the
    /// horizon is removed from the mean; σ stays the causal ticker scale so the persistence
    /// prior and decoder are unchanged).
    /// `stats` must already be narrowed to the scored origins (`Statistics::last` when
    /// `last_only`).
    pub fn targets(&self, batch: &Batch, stats: &Statistics, last_only: bool) -> (Tensor, Tensor) {
        let future = self.future_windows(&batch.log_prices, last_only);
        assert_eq!(future.size()[1], stats.sigma.size()[1]);
        let targets = (future - per_bar(&stats.log_close)) / per_bar(&stats.sigma)
            - self.market_drift(batch, stats, last_only);
        let mask =
            (self.future_windows(&batch.valid, last_only) * stats.mask.unsqueeze(-1)).unsqueeze(2);
        (targets, mask)
    }

    /// Point forecast in σ units for `output` produced under the same `last_only` scope.
    pub fn decode(&self, output: &Output, stats: &Statistics) -> Tensor {
        decode_joint(
            &output.coordinates,
            &per_bar(&stats.sigma),
            &per_bar(&stats.range),
            &self.horizon_scale,
        )
    }

    /// Masked Gaussian NLL of the dense head, fused down to the ops the algebra needs.
    ///
    /// `targets` is `[rows, origins', CHANNELS, pred_len]` and `mask` `[rows, origins', 1,
    /// pred_len]`, both from [`Self::targets`]. Mathematically this is exactly
    /// `gaussian_nll(decode_joint(coordinates, ..), log_scale, targets, mask)`; what changes is
    /// how many times the 147 M-element head space crosses HBM:
    ///
    /// - the fp32 widening of the whole `[.., 2·CHANNELS, pred_len]` space is gone. Each channel
    ///   is promoted by the first arithmetic op that needs it, which reads bf16 and writes fp32
    ///   in one pass instead of a separate 885 MB cast whose backward is another one;
    /// - `1/σ` is applied to the three candle offsets rather than to their three differences, so
    ///   `low` is shared by `high` and `open` instead of recomputed;
    /// - `exp(-2·ls)` is factored as `exp(-2·CAP·tanh(u))·(1/h)`, which removes the `+ ½·ln h`
    ///   add over the full space, and the `Σ mask·½·ln h` it leaves behind is a gradient-free
    ///   constant reduced over `[pred_len]`;
    /// - the mask multiply is folded into the per-element precision weight that has to be
    ///   materialized anyway, and the two reductions are `dot`s, so nothing writes a full-size
    ///   masked copy of the NLL or of the squared error.
    ///
    /// fp32 is kept everywhere it matters: every geometry and NLL element is computed in fp32
    /// from a bf16 read, and both reductions accumulate in fp32 exactly as the `sum` they
    /// replace did. Against the reference chain
    /// (`gaussian_nll(decode_joint(..), ..)`, kept for the evaluation decoders and for
    /// `fused_loss_matches_the_reference_decode_and_nll_including_gradients`) the observed
    /// difference is 0.0 relative on the NLL, on the MSE and on all 34 parameter gradients:
    /// the reassociation is exact in fp32 for this algebra, not merely within tolerance.
    pub fn losses(
        &self,
        head: &Head,
        stats: &Statistics,
        targets: &Tensor,
        mask: &Tensor,
    ) -> Losses {
        let smallest = f64::from(f32::MIN_POSITIVE);
        let channels = head.channels();
        let sigma = per_bar(&stats.sigma).clamp_min(smallest);
        let range = per_bar(&stats.range).clamp_min(smallest);
        // `to_kind` where the channel enters a unary fp32 op, promotion where it enters a
        // binary one: `bf16 · [1, 1, 1, pred_len] fp32` reads bf16 and writes fp32 in one pass.
        // A 0-dim fp32 operand would NOT promote (ATen ranks dimensioned operands first), which
        // is why `log_scale_gain` is shaped.
        let coordinate = |index: usize| channels[index].to_kind(Kind::Float);
        let close = &channels[0] * &self.horizon_scale;
        let relative_range = coordinate(1).softplus() * range / std::f64::consts::LN_2;
        let low = &close - (coordinate(2).sigmoid() * &relative_range).log1p() / &sigma;
        let high = &low + relative_range.log1p() / &sigma;
        let open = &low + (coordinate(3).sigmoid() * &relative_range).log1p() / &sigma;
        let flat = |tensor: &Tensor| tensor.reshape([-1]);
        let mask_flat = flat(mask);
        let precision = mask * &self.inverse_horizon;
        let mut terms = Vec::with_capacity(2 * CHANNELS as usize);
        let mut squares = Vec::with_capacity(CHANNELS as usize);
        for (channel, prediction) in [open, high, low, close].into_iter().enumerate() {
            let tanh = (&channels[CHANNELS as usize + channel] * &self.log_scale_gain).tanh();
            let weight = (&tanh * (-2.0 * LOG_SCALE_CAP)).exp() * &precision;
            let square = (targets.narrow(2, channel as i64, 1) - prediction).square();
            terms.push(flat(&square).dot(&flat(&weight)) * 0.5);
            terms.push(flat(&tanh).dot(&mask_flat) * LOG_SCALE_CAP);
            squares.push(tch::no_grad(|| flat(&square).dot(&mask_flat)));
        }
        // `Σ mask·½·ln h` over channels: gradient-free, so it never belongs in a full-size
        // kernel. `mask` is `[rows, origins', 1, pred_len]`, so this reduces to `[pred_len]`.
        let prior = mask
            .sum_dim_intlist([0i64, 1, 2].as_slice(), false, Kind::Float)
            .dot(&flat(&self.half_log_horizon))
            * CHANNELS;
        let count = (mask.sum(Kind::Float) * CHANNELS).clamp_min(1.0);
        Losses {
            nll: (Tensor::stack(&terms, 0).sum(Kind::Float) + prior) / &count,
            mse: Tensor::stack(&squares, 0).sum(Kind::Float) / count,
        }
    }
}

/// Maps candle coordinates `[.., 4, pred_len]` to σ-scaled log returns relative to the origin
/// close; `sigma` and `range` broadcast as `[.., 1, 1]`. The close coordinate is in units of the
/// h-step persistence deviation σ√h, the range is a softplus multiple of the mean relative range,
/// and open/close positions are sigmoids inside it. Every channel is the close plus a monotone
/// offset, so `high >= max(open, close)` and `low <= min(open, close)` survive rounding.
pub fn decode_joint(
    coordinates: &Tensor,
    sigma: &Tensor,
    range: &Tensor,
    horizon_scale: &Tensor,
) -> Tensor {
    let smallest = f64::from(f32::MIN_POSITIVE);
    let sigma = sigma.clamp_min(smallest);
    let close = coordinates.narrow(-2, 0, 1) * horizon_scale;
    let relative_range =
        range.clamp_min(smallest) * coordinates.narrow(-2, 1, 1).softplus() / std::f64::consts::LN_2;
    let close_offset = (coordinates.narrow(-2, 2, 1).sigmoid() * &relative_range).log1p();
    let open_offset = (coordinates.narrow(-2, 3, 1).sigmoid() * &relative_range).log1p();
    let full = relative_range.log1p();
    let low = &close - &close_offset / &sigma;
    let high = &close + (full - &close_offset) / &sigma;
    let open = &close + (open_offset - close_offset) / sigma;
    Tensor::cat(&[open, high, low, close], -2)
}

/// Prices from σ-scaled log returns; fp64 intermediates clamped into the fp32 range.
pub fn decode_prices(scaled: &Tensor, anchor: &Tensor, sigma: &Tensor) -> Tensor {
    let smallest = f64::from(f32::MIN_POSITIVE);
    let largest = f64::from(f32::MAX);
    let log_prices = (scaled.to_kind(Kind::Double) * sigma.to_kind(Kind::Double))
        .clamp(smallest.ln(), largest.ln());
    (anchor.to_kind(Kind::Double).clamp_min(smallest) * log_prices.exp())
        .clamp(smallest, largest)
        .to_kind(Kind::Float)
}

/// Per-element Gaussian negative log-likelihood `½·((y - ŷ)/s)² + ln s` with `s = exp(log_scale)`.
pub fn nll_elements(prediction: &Tensor, log_scale: &Tensor, target: &Tensor) -> Tensor {
    ((target - prediction) * (-log_scale).exp()).square() * 0.5 + log_scale
}

pub struct Losses {
    pub nll: Tensor,
    pub mse: Tensor,
}

/// Masked means over valid (origin, channel, bar) triples; `mse` carries no gradient. The
/// reference form of [`CausalPatchModel::losses`], kept for the evaluation path and for the
/// equivalence test that pins the fused one.
pub fn gaussian_nll(prediction: &Tensor, log_scale: &Tensor, target: &Tensor, mask: &Tensor) -> Losses {
    let count = (mask.sum(Kind::Float) * CHANNELS).clamp_min(1.0);
    let nll = (nll_elements(prediction, log_scale, target) * mask).sum(Kind::Float) / &count;
    let mse = tch::no_grad(|| ((target - prediction).square() * mask).sum(Kind::Float) / count);
    Losses { nll, mse }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The composed reference the fused kernels are checked against: normalize the packed
    // block, then rotate it. `Block::forward` runs ONE kernel for the pair, and these tests
    // are what pins the two to the same bytes.
    use fused_kernels::rope as fused_rope;
    use tch::Device;

    fn small_config() -> ModelConfig {
        ModelConfig {
            seq_len: 64,
            pred_len: 8,
            patch_len: 16,
            layers: 2,
            d_model: 32,
            heads: 4,
            ffn: 64,
            dropout: 0.0,
            min_history: 16,
            features: FeatureSet::ALL,
        }
    }

    /// Random-walk rows in the corpus layout; `valid_future[i]` observed horizon bars per row.
    fn synthetic(config: &ModelConfig, valid_future: &[i64]) -> Batch {
        let rows = valid_future.len() as i64;
        let length = config.seq_len + config.pred_len;
        let aux_channels = config.features.channels() as i64;
        let close = Tensor::randn([rows, length, 1], (Kind::Float, Device::Cpu)).cumsum(1, Kind::Float) * 0.002;
        let close = &close - close.narrow(1, config.seq_len - 1, 1);
        let open = &close + Tensor::randn([rows, length, 1], (Kind::Float, Device::Cpu)) * 0.0005;
        let high = close.maximum(&open) + Tensor::rand([rows, length, 1], (Kind::Float, Device::Cpu)) * 0.001;
        let low = close.minimum(&open) - Tensor::rand([rows, length, 1], (Kind::Float, Device::Cpu)) * 0.001;
        let log_prices = Tensor::cat(&[open, high, low, close], 2);
        let valid = Tensor::ones([rows, length], (Kind::Float, Device::Cpu));
        for (row, &count) in valid_future.iter().enumerate() {
            let _ = valid
                .narrow(0, row as i64, 1)
                .narrow(1, config.seq_len + count, config.pred_len - count)
                .fill_(0.0);
        }
        let aux = Tensor::randn([rows, length, aux_channels], (Kind::Float, Device::Cpu));
        let market = Tensor::randn([rows, length], (Kind::Float, Device::Cpu)).cumsum(1, Kind::Float) * 0.001;
        let market = &market - market.narrow(1, config.seq_len - 1, 1);
        let anchor = Tensor::arange(rows, (Kind::Float, Device::Cpu)) + 100.0;
        let packed = Tensor::cat(
            &[
                log_prices.flatten(1, 2),
                valid,
                aux.flatten(1, 2),
                market,
                anchor.unsqueeze(1),
            ],
            1,
        );
        Batch::from_packed(
            packed,
            config.seq_len as usize,
            config.pred_len as usize,
            aux_channels as usize,
            valid_future.iter().sum::<i64>() as usize,
        )
    }

    #[test]
    fn requires_complete_uniform_patches_and_bounded_history() {
        let mut config = ModelConfig::default();
        config.validate().unwrap();
        config.seq_len = 97;
        assert!(config.validate().is_err());
        config.seq_len = 96;
        config.dropout = f64::NAN;
        assert!(config.validate().is_err());
        config.dropout = 0.0;
        config.min_history = 97;
        assert!(config.validate().is_err());
        config.min_history = 1;
        assert!(config.validate().is_err());
        config.min_history = 96;
        config.heads = 512 / 3;
        assert!(config.validate().is_err());
    }

    /// Channel-major `[.., 4, bars]`, matching what the model emits.
    fn assert_valid_candles(prices: &Tensor) {
        let open = prices.narrow(-2, 0, 1);
        let high = prices.narrow(-2, 1, 1);
        let low = prices.narrow(-2, 2, 1);
        let close = prices.narrow(-2, 3, 1);
        assert_eq!(prices.isfinite().all().int64_value(&[]), 1);
        assert_eq!(prices.gt(0.0).all().int64_value(&[]), 1);
        assert_eq!(
            high.ge_tensor(&open.maximum(&close)).all().int64_value(&[]),
            1
        );
        assert_eq!(
            low.le_tensor(&open.minimum(&close)).all().int64_value(&[]),
            1
        );
    }

    fn horizon(pred_len: i64) -> Tensor {
        (Tensor::arange(pred_len, (Kind::Float, Device::Cpu)) + 1.0)
            .sqrt()
            .reshape([1, 1, pred_len])
    }

    #[test]
    fn zero_coordinates_decode_to_persistence_and_keep_price_gradients() {
        let sigma = Tensor::from_slice(&[0.002f32, 0.05]).reshape([2, 1, 1]);
        let range = Tensor::from_slice(&[0.004f32, 0.3]).reshape([2, 1, 1]);
        let anchor = Tensor::from_slice(&[100.0f32, 0.000001]).reshape([2, 1, 1]);
        let coordinates =
            Tensor::zeros([2, 4, 3], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let scaled = decode_joint(&coordinates, &sigma, &range, &horizon(3));
        let prices = decode_prices(&scaled, &anchor, &sigma);
        assert_valid_candles(&prices);
        for step in 0..3 {
            assert_eq!(scaled.double_value(&[0, 3, step]), 0.0);
            assert_eq!(scaled.double_value(&[0, 0, step]), 0.0);
            assert_eq!(prices.double_value(&[0, 3, step]), 100.0);
            assert_eq!(prices.double_value(&[0, 0, step]), 100.0);
            assert_eq!(prices.double_value(&[1, 3, step]), f64::from(0.000001f32));
        }
        let low = prices.double_value(&[0, 2, 0]);
        let high = prices.double_value(&[0, 1, 0]);
        assert!(((high - low) / low - 0.004).abs() < 1e-6);
        let target = Tensor::from_slice(&[0.5f32, 1.0, -0.5, 0.25])
            .reshape([1, 4, 1])
            .expand_as(&scaled);
        (&scaled - target).square().mean(Kind::Float).backward();
        assert_eq!(coordinates.grad().isfinite().all().int64_value(&[]), 1);
        for coordinate in 0..4 {
            assert!(
                coordinates
                    .grad()
                    .narrow(1, coordinate, 1)
                    .abs()
                    .sum(Kind::Float)
                    .double_value(&[])
                    > 0.0
            );
        }
    }

    #[test]
    fn close_coordinate_is_measured_in_horizon_persistence_sigmas() {
        let sigma = Tensor::from_slice(&[0.01f32]).reshape([1, 1, 1]);
        let range = Tensor::from_slice(&[0.002f32]).reshape([1, 1, 1]);
        let coordinates = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0])
            .reshape([1, 4, 1])
            .expand([1, 4, 4], true);
        let scaled = decode_joint(&coordinates, &sigma, &range, &horizon(4));
        let prices = decode_prices(&scaled, &Tensor::from_slice(&[50.0f32]).reshape([1, 1, 1]), &sigma);
        for step in 0..4 {
            let steps = (step + 1) as f64;
            assert!((scaled.double_value(&[0, 3, step]) - steps.sqrt()).abs() < 1e-6);
            let expected = 50.0 * (0.01 * steps.sqrt()).exp();
            assert!((prices.double_value(&[0, 3, step]) - expected).abs() < 1e-4 * expected);
        }
    }

    #[test]
    fn joint_decoder_remains_valid_at_flat_and_saturated_position_limits() {
        let mut coordinates = Vec::new();
        for close in [-1e6f32, 0.0, 1e6] {
            for range in [-1000.0, 0.0, 1e6] {
                for close_position in [-1000.0, 0.0, 1000.0] {
                    for open_position in [-1000.0, 0.0, 1000.0] {
                        coordinates.extend([close, range, close_position, open_position]);
                    }
                }
            }
        }
        let count = coordinates.len() as i64 / 4;
        let coordinates = Tensor::from_slice(&coordinates)
            .reshape([1, count, 4])
            .transpose(1, 2)
            .contiguous()
            .set_requires_grad(true);
        let sigma = Tensor::from_slice(&[0.003f32]).reshape([1, 1, 1]);
        let range = Tensor::from_slice(&[0.001f32]).reshape([1, 1, 1]);
        let anchor = Tensor::from_slice(&[100.0f32]).reshape([1, 1, 1]);
        let scaled = decode_joint(&coordinates, &sigma, &range, &horizon(count));
        assert_valid_candles(&decode_prices(&scaled, &anchor, &sigma));
        scaled.mean(Kind::Float).backward();
        assert_eq!(coordinates.grad().isfinite().all().int64_value(&[]), 1);
        let flat = decode_prices(
            &decode_joint(
                &Tensor::from_slice(&[0.0f32, -1000.0, 0.0, 0.0]).reshape([1, 4, 1]),
                &sigma,
                &range,
                &horizon(1),
            ),
            &anchor,
            &sigma,
        );
        assert_eq!(flat.max().double_value(&[]), flat.min().double_value(&[]));
        assert_eq!(flat.max().double_value(&[]), 100.0);
        for extreme in [3e38f32, f32::MIN_POSITIVE] {
            let anchor = Tensor::from_slice(&[extreme]).reshape([1, 1, 1]);
            let coordinates = Tensor::from_slice(&[0.0f32, 100.0, -100.0, 0.0])
                .reshape([1, 4, 1])
                .set_requires_grad(true);
            let scaled = decode_joint(&coordinates, &sigma, &Tensor::from_slice(&[1.0f32]).reshape([1, 1, 1]), &horizon(1));
            let prices = decode_prices(&scaled, &anchor, &sigma);
            assert_valid_candles(&prices);
            assert_eq!(prices.double_value(&[0, 3, 0]), f64::from(extreme));
            scaled.mean(Kind::Float).backward();
            assert_eq!(coordinates.grad().isfinite().all().int64_value(&[]), 1);
        }
    }

    #[test]
    fn causal_sigma_matches_an_independent_expanding_std_over_masked_bars() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(7);
        let config = small_config();
        let batch = synthetic(&config, &[8, 3]);
        let _ = batch.valid.narrow(0, 1, 1).narrow(1, 2, 18).fill_(0.0);
        let _ = batch.valid.narrow(0, 1, 1).narrow(1, 40, 1).fill_(0.0);
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let stats = model.statistics(&batch);
        assert_eq!(stats.sigma.size(), [2, 4]);
        let closes: Vec<f32> = Vec::try_from(batch.log_prices.select(2, 3).flatten(0, 1)).unwrap();
        let valid: Vec<f32> = Vec::try_from(batch.valid.flatten(0, 1)).unwrap();
        let length = (config.seq_len + config.pred_len) as usize;
        for row in 0..2 {
            for origin in 0..4 {
                let end = 16 * (origin + 1);
                let base = row * length;
                let mut returns = Vec::new();
                let mut bars = 0;
                for t in 0..end {
                    if valid[base + t] > 0.0 {
                        bars += 1;
                    }
                    if t > 0 && valid[base + t] > 0.0 && valid[base + t - 1] > 0.0 {
                        returns.push(f64::from(closes[base + t]) - f64::from(closes[base + t - 1]));
                    }
                }
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance =
                    returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
                let expected = (variance + RETURN_VARIANCE_FLOOR).sqrt();
                let actual = stats.sigma.double_value(&[row as i64, origin as i64]);
                assert!((actual - expected).abs() <= 1e-5 * expected, "{actual} vs {expected}");
                assert_eq!(
                    stats.mask.double_value(&[row as i64, origin as i64]),
                    f64::from(u8::from(bars >= 16))
                );
                assert_eq!(
                    stats.log_close.double_value(&[row as i64, origin as i64]),
                    f64::from(closes[base + end - 1])
                );
            }
        }
        assert_eq!(stats.mask.double_value(&[1, 0]), 0.0);
        assert_eq!(stats.mask.double_value(&[1, 1]), 0.0);
        assert_eq!(stats.mask.double_value(&[1, 2]), 1.0);
        assert_eq!(stats.mask.double_value(&[0, 0]), 1.0);
    }

    #[test]
    fn targets_read_only_bars_after_each_origin() {
        let _rng = crate::torch::test_rng::shared();
        let config = small_config();
        let mut batch = synthetic(&config, &[8, 5]);
        let length = config.seq_len + config.pred_len;
        // Encode the bar index in every channel: y[k, h] must be (index - t_k) / σ_k with index = t_k + h.
        let indexed = (Tensor::arange(length, (Kind::Float, Device::Cpu)) * 1e-3)
            .reshape([1, length, 1])
            .expand([2, length, CHANNELS], true);
        batch.log_prices.copy_(&indexed);
        let _ = batch.market_cum.zero_();
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let stats = model.statistics(&batch);
        let (targets, mask) = model.targets(&batch, &stats, false);
        assert_eq!(targets.size(), [2, 4, 4, 8]);
        assert_eq!(mask.size(), [2, 4, 1, 8]);
        for origin in 0..4 {
            let sigma = stats.sigma.double_value(&[0, origin]);
            for step in 0..8 {
                for channel in 0..4 {
                    let expected = (step + 1) as f64 * 1e-3 / sigma;
                    let actual = targets.double_value(&[0, origin, channel, step]);
                    assert!((actual - expected).abs() <= 1e-3 * expected, "{actual} vs {expected}");
                }
                let bar = 16 * (origin + 1) + step;
                let expected_mask = if bar < config.seq_len + 5 { 1.0 } else { 0.0 };
                assert_eq!(mask.double_value(&[1, origin, 0, step]), expected_mask);
                assert_eq!(mask.double_value(&[0, origin, 0, step]), 1.0);
            }
        }
        let (last, last_mask) = model.targets(&batch, &stats.last(), true);
        assert!(last.equal(&targets.narrow(1, 3, 1)));
        assert!(last_mask.equal(&mask.narrow(1, 3, 1)));
        let windows = model.future_windows(&batch.log_prices, false).copy();
        let _ = batch.log_prices.narrow(1, 0, 32).fill_(5.0);
        let perturbed = model.future_windows(&batch.log_prices, false);
        assert!(
            perturbed.narrow(1, 1, 3).equal(&windows.narrow(1, 1, 3)),
            "windows of origins 1..3 must not read bars at or before their origin"
        );
        assert!(!perturbed.narrow(1, 0, 1).equal(&windows.narrow(1, 0, 1)));
    }

    #[test]
    fn a_ticker_tracking_the_market_has_zero_close_target() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(5);
        let config = small_config();
        let mut batch = synthetic(&config, &[8, 8]);
        let length = config.seq_len + config.pred_len;
        // Market path with drift and gaps; the ticker's close follows it bar for bar.
        let market = (Tensor::randn([2, length], (Kind::Float, Device::Cpu)) * 0.003 + 0.002)
            .cumsum(1, Kind::Float);
        let market = &market - market.narrow(1, config.seq_len - 1, 1);
        batch.market_cum.copy_(&market);
        batch
            .log_prices
            .copy_(&market.unsqueeze(-1).expand([2, length, CHANNELS], true));
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let stats = model.statistics(&batch);
        assert!(stats.market.narrow(1, 3, 1).abs().max().double_value(&[]) == 0.0);
        for origin in 0..4 {
            let expected = market.double_value(&[0, 16 * (origin + 1) - 1]);
            assert_eq!(stats.market.double_value(&[0, origin]), expected);
        }
        let (targets, _) = model.targets(&batch, &stats, false);
        assert!(targets.abs().max().double_value(&[]) < 1e-3, "{}", targets.abs().max());
        let drift = model.market_drift(&batch, &stats, false);
        assert_eq!(drift.size(), [2, 4, 1, 8]);
        assert!(drift.abs().max().double_value(&[]) > 0.0);
        let raw = (model.future_windows(&batch.log_prices, false) - per_bar(&stats.log_close))
            / per_bar(&stats.sigma);
        assert!((raw - &drift - &targets).abs().max().double_value(&[]) < 1e-5);
        let (last, _) = model.targets(&batch, &stats.last(), true);
        assert!(last.abs().max().double_value(&[]) < 1e-3);
    }

    #[test]
    fn a_half_beta_ticker_converges_to_a_zero_close_target_with_history() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(7);
        let config = ModelConfig {
            seq_len: 8192,
            ..small_config()
        };
        let origins = config.origins();
        let mut batch = synthetic(&config, &[8]);
        let length = config.seq_len + config.pred_len;
        let market = (Tensor::randn([1, length], (Kind::Float, Device::Cpu)) * 0.003 + 0.001)
            .cumsum(1, Kind::Float);
        let market = &market - market.narrow(1, config.seq_len - 1, 1);
        batch.market_cum.copy_(&market);
        batch
            .log_prices
            .copy_(&(&market * 0.5).unsqueeze(-1).expand([1, length, CHANNELS], true));
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let stats = model.statistics(&batch);
        // Exact proportionality makes the ridge slope closed-form: (0.5 n + 256) / (n + 256).
        for origin in [0, 15, origins - 1] {
            let pairs = (16 * (origin + 1) - 1) as f64;
            let expected = (0.5 * pairs + BETA_PRIOR_BARS) / (pairs + BETA_PRIOR_BARS);
            let beta = stats.beta.double_value(&[0, origin]);
            assert!((beta - expected).abs() < 1e-4, "origin {origin}: {beta} vs {expected}");
        }
        assert!(stats.beta.double_value(&[0, 0]) > 0.97);
        let raw = (model.future_windows(&batch.log_prices, false) - per_bar(&stats.log_close))
            / per_bar(&stats.sigma);
        let (targets, _) = model.targets(&batch, &stats, false);
        let residual = |origin: i64| {
            targets.select(1, origin).abs().max().double_value(&[])
                / raw.select(1, origin).abs().max().double_value(&[])
        };
        // Early origins still carry most of the market; the final origin keeps ~6% of it.
        assert!(residual(0) > 0.5, "{}", residual(0));
        assert!(residual(origins - 1) < 0.07, "{}", residual(origins - 1));
        let (last, _) = model.targets(&batch, &stats.last(), true);
        assert!(last.equal(&targets.narrow(1, origins - 1, 1)));
    }

    #[test]
    fn head_covariates_ignore_history_channels_beyond_the_context() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(11);
        let config = small_config();
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        tch::no_grad(|| {
            let _ = model.head_output.ws.shallow_clone().uniform_(-0.1, 0.1);
        });
        let batch = synthetic(&config, &[8, 8]);
        let stats = model.statistics(&batch);
        let reference = model.output(&model.forward(&batch, &stats, false, false));
        let history_channels: Vec<i64> = config
            .features
            .channel_mask(Feature::known_future)
            .iter()
            .enumerate()
            .filter_map(|(index, known)| (!known).then_some(index as i64))
            .collect();
        assert_eq!(history_channels, [6, 7, 8, 9, 10, 11]);
        let _ = batch
            .aux
            .narrow(1, config.seq_len, config.pred_len)
            .narrow(2, 6, 6)
            .fill_(7.0);
        let unchanged = model.output(&model.forward(&batch, &stats, false, false));
        assert!(unchanged.coordinates.equal(&reference.coordinates));
        assert!(unchanged.log_scale.equal(&reference.log_scale));
        let _ = batch
            .aux
            .narrow(1, config.seq_len, config.pred_len)
            .narrow(2, 0, 1)
            .fill_(7.0);
        let changed = model.output(&model.forward(&batch, &stats, false, false));
        assert!(!changed.coordinates.equal(&reference.coordinates));
    }

    #[test]
    fn zero_head_forecasts_persistence_with_root_horizon_scale_at_every_origin() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(3);
        let config = small_config();
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let batch = synthetic(&config, &[8, 5]);
        let stats = model.statistics(&batch);
        let head = model.forward(&batch, &stats, false, false);
        let output = model.output(&head);
        let scaled = model.decode(&output, &stats);
        assert_eq!(scaled.size(), [2, 4, 4, 8]);
        assert_eq!(scaled.narrow(2, 3, 1).abs().max().double_value(&[]), 0.0);
        assert_eq!(scaled.narrow(2, 0, 1).abs().max().double_value(&[]), 0.0);
        let expected_scale = model.half_log_horizon().expand_as(&output.log_scale);
        assert!(output.log_scale.equal(&expected_scale));
        let anchor = batch.anchor.reshape([2, 1, 1, 1]) * stats.log_close.unsqueeze(-1).unsqueeze(-1).exp();
        let prices = decode_prices(&scaled, &anchor, &per_bar(&stats.sigma));
        assert!((prices.narrow(2, 3, 1) - &anchor).abs().max().double_value(&[]) < 1e-3);
        let (targets, mask) = model.targets(&batch, &stats, false);
        let losses = gaussian_nll(&scaled, &output.log_scale, &targets, &mask);
        let nll = losses.nll.double_value(&[]);
        let mut by_hand = 0.0;
        let mut squared = 0.0;
        let mut count = 0.0;
        for row in 0..2 {
            for origin in 0..4 {
                let sigma = stats.sigma.double_value(&[row, origin]);
                let range = stats.range.double_value(&[row, origin]);
                let half = (0.5 * range).ln_1p();
                let candle = [0.0, (range.ln_1p() - half) / sigma, -half / sigma, 0.0];
                for step in 0..8 {
                    let weight = mask.double_value(&[row, origin, 0, step]);
                    for (channel, reference) in candle.iter().enumerate() {
                        let y = targets.double_value(&[row, origin, channel as i64, step]);
                        let h = (step + 1) as f64;
                        let residual = y - reference;
                        by_hand += weight * (0.5 * residual * residual / h + 0.5 * h.ln());
                        squared += weight * residual * residual;
                        count += weight;
                        assert!(
                            (scaled.double_value(&[row, origin, channel as i64, step]) - reference).abs()
                                < 1e-5,
                            "persistence candle mismatch at channel {channel}"
                        );
                    }
                }
            }
        }
        assert!((nll - by_hand / count).abs() < 1e-5, "{nll} vs {}", by_hand / count);
        assert!((losses.mse.double_value(&[]) - squared / count).abs() < 1e-5);
        assert!(mask.sum(Kind::Float).double_value(&[]) < 2.0 * 4.0 * 8.0);
        // The fused training loss must reproduce the reference chain it replaced.
        let fused = model.losses(&head, &stats, &targets, &mask);
        assert!(
            (fused.nll.double_value(&[]) - nll).abs() <= 1e-6 * nll.abs().max(1e-6),
            "fused {} vs reference {nll}",
            fused.nll.double_value(&[])
        );
        assert!(
            (fused.mse.double_value(&[]) - losses.mse.double_value(&[])).abs() < 1e-6
        );
    }

    /// The fused training loss against the decode+NLL chain it replaced, with a NONZERO head so
    /// every branch of the candle geometry and of the log-scale cap carries signal: the loss, the
    /// no-gradient MSE, and every parameter gradient.
    #[test]
    fn fused_loss_matches_the_reference_decode_and_nll_including_gradients() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(17);
        let config = small_config();
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        tch::no_grad(|| {
            let _ = model.head_output.ws.shallow_clone().uniform_(-0.5, 0.5);
            let _ = model
                .head_output
                .bs
                .as_ref()
                .unwrap()
                .shallow_clone()
                .uniform_(-0.5, 0.5);
            // The residual-branch output projections are zero at init (that is what makes the
            // stack the identity, see `the_stack_is_the_identity_on_the_residual_stream_at_init`),
            // so at init the QKV and FFN-up matrices and the post-lambdas legitimately receive
            // no gradient. This test is about the fused loss reaching every parameter, so it
            // needs a live network.
            for block in &model.blocks {
                let _ = block.output.ws.shallow_clone().uniform_(-0.1, 0.1);
                let _ = block.second.ws.shallow_clone().uniform_(-0.1, 0.1);
            }
        });
        let batch = synthetic(&config, &[8, 5]);
        let stats = model.statistics(&batch);
        let (targets, mask) = model.targets(&batch, &stats, false);
        let head = model.forward(&batch, &stats, false, false);
        let output = model.output(&head);
        let reference =
            gaussian_nll(&model.decode(&output, &stats), &output.log_scale, &targets, &mask);
        let fused = model.losses(&head, &stats, &targets, &mask);
        let expected = reference.nll.double_value(&[]);
        assert!(expected.is_finite() && expected.abs() > 1e-3, "{expected}");
        let relative = (fused.nll.double_value(&[]) - expected).abs() / expected.abs();
        assert!(relative <= 1e-4, "NLL relative error {relative}");
        let expected_mse = reference.mse.double_value(&[]);
        assert!(
            (fused.mse.double_value(&[]) - expected_mse).abs() <= 1e-4 * expected_mse.abs(),
            "MSE {} vs {expected_mse}",
            fused.mse.double_value(&[])
        );
        let parameters = store.trainable_variables();
        let reference_grads = Tensor::run_backward(&[&reference.nll], &parameters, true, false);
        let fused_grads = Tensor::run_backward(&[&fused.nll], &parameters, false, false);
        let mut touched = 0;
        let mut worst = 0.;
        for (index, (left, right)) in reference_grads.iter().zip(&fused_grads).enumerate() {
            let scale = left.abs().max().double_value(&[]);
            assert!(scale.is_finite(), "nonfinite reference gradient {index}");
            if scale > 0.0 {
                touched += 1;
            }
            let error = (left - right).abs().max().double_value(&[]) / scale.max(1e-8);
            worst = f64::max(worst, error);
            assert!(error <= 1e-4, "parameter {index} gradient relative error {error}");
        }
        assert_eq!(touched, parameters.len(), "a parameter received no gradient");
        println!(
            "fused vs reference: NLL {relative:.3e} relative, MSE {:.3e} relative, worst \
             parameter gradient {worst:.3e} relative over {touched} parameters",
            (fused.mse.double_value(&[]) - expected_mse).abs() / expected_mse.abs()
        );
    }

    #[test]
    fn cuda_fresh_model_trains_every_branch() {
        let _rng = crate::torch::test_rng::exclusive();
        if std::env::var("TIMEXER_SEGMENT_GPU_TEST").as_deref() != Ok("1") {
            return;
        }
        assert!(tch::Cuda::is_available());
        crate::torch::cuda::cfg::configure_cuda();
        crate::torch::cuda::cfg::enable_tf32_matmul().unwrap();
        let device = Device::Cuda(0);
        let _backward = crate::torch::cuda::cfg::disable_autograd_multithreading();
        tch::manual_seed(20260905);
        tch::Cuda::manual_seed_all(20260905);
        let config = ModelConfig::default();
        let store = nn::VarStore::new(device);
        let model = CausalPatchModel::new(&store.root(), &config);
        let batch = synthetic(&config, &[192, 100]).to_device(device);
        let stats = model.statistics(&batch);
        use tch::nn::OptimizerConfig;
        let mut optimizer = tch::nn::Sgd::default().build(&store, 0.1).unwrap();
        for _ in 0..2 {
            let head = model.forward(&batch, &stats, true, false);
            let (targets, mask) = model.targets(&batch, &stats, false);
            let losses = model.losses(&head, &stats, &targets, &mask);
            assert!(losses.nll.double_value(&[]).is_finite());
            optimizer.zero_grad();
            losses.nll.backward();
            optimizer.step();
        }
        for (name, parameter) in store.variables() {
            let gradient = parameter.grad();
            assert!(gradient.defined(), "missing gradient: {name}");
            assert_eq!(parameter.kind(), Kind::Float, "master dtype: {name}");
            assert_eq!(gradient.isfinite().all().int64_value(&[]), 1, "nonfinite gradient: {name}");
            assert!(
                gradient.abs().sum(Kind::Float).double_value(&[]) > 0.0,
                "a branch received no gradient: {name}"
            );
        }
    }

    /// A small config with the real head geometry: `head_dim` even, rotary over the whole head.
    fn narrow_config() -> ModelConfig {
        ModelConfig {
            seq_len: 96,
            pred_len: 8,
            patch_len: 8,
            layers: 2,
            d_model: 32,
            heads: 4,
            ffn: 64,
            min_history: 8,
            ..Default::default()
        }
    }

    /// The packed full-width rotation, the single `split_with_sizes` and the strided attention
    /// views must produce EXACTLY the tensors the per-tensor half-width form produced - not
    /// within a tolerance. The whole point of the rewrite is that it is a regrouping of
    /// kernels, so any nonzero difference means an operand moved, and the composed block
    /// output and every parameter gradient must be bit-identical too.
    #[test]
    fn the_packed_rotation_and_split_match_the_per_tensor_reference_bit_for_bit() {
        let _rng = crate::torch::test_rng::exclusive();
        let config = narrow_config();
        let store = nn::VarStore::new(Device::Cpu);
        let mut block = Block::new(store.root() / "block", &config, 0);
        let (width, heads) = (config.d_model, config.heads);
        let head_dim = width / heads;
        let (rows, length) = (2, config.origins());
        let rope = RotaryEmbedding::new(length, head_dim, head_dim, Device::Cpu);
        let (cosine, sine) = rope.cached_rotation(
            &Tensor::arange(length, (Kind::Int64, Device::Cpu)),
            Kind::BFloat16,
        );
        let input = (Tensor::randn([rows, length, width], (Kind::Float, Device::Cpu)) * 0.5)
            .to_kind(Kind::BFloat16)
            .set_requires_grad(true);
        let x0 = (Tensor::randn([rows, length, width], (Kind::Float, Device::Cpu)) * 0.5)
            .to_kind(Kind::BFloat16)
            .set_requires_grad(true);
        // Distinct, non-unit lambdas and non-zero residual-branch weights: at the real init the
        // output projections are zero and `x0_lambda` is zero, which would make the whole
        // feedforward half of this comparison `0 == 0` and leave four parameter gradients with
        // no signal at all.
        let lambdas = store.root() / "lambdas";
        let mut resid = lambdas.var("resid", &[2], nn::Init::Const(0.0));
        let mut post = lambdas.var("post", &[2], nn::Init::Const(0.0));
        let mut x0_lambda = lambdas.var("x0", &[1], nn::Init::Const(0.0));
        tch::no_grad(|| {
            resid.copy_(&Tensor::from_slice(&[1.05_f32, 0.9]));
            post.copy_(&Tensor::from_slice(&[0.8_f32, 1.3]));
            x0_lambda.copy_(&Tensor::from_slice(&[0.25_f32]));
            for weight in [&mut block.output.ws, &mut block.second.ws] {
                let noise =
                    Tensor::randn(weight.size(), (Kind::Float, Device::Cpu)) * 0.05;
                weight.copy_(&noise);
            }
        });
        let (resid_bf16, post_bf16, x0_bf16) = (
            resid.to_kind(Kind::BFloat16),
            post.to_kind(Kind::BFloat16),
            x0_lambda.to_kind(Kind::BFloat16),
        );
        let (resid_parts, post_parts) = (resid_bf16.unbind(0), post_bf16.unbind(0));
        let x0_scale = x0_bf16.get(0);
        let block_lambdas = BlockLambdas {
            resid: [&resid_parts[0], &resid_parts[1]],
            post: [&post_parts[0], &post_parts[1]],
            x0: &x0_scale,
        };
        let projection = linear(&rms_norm(&input), &block.qkv);
        let parts = projection.split(width, -1);
        let per_head = |part: &Tensor| {
            part.reshape([rows, length, heads, head_dim])
                .transpose(1, 2)
        };
        // The per-tensor reference for QK-norm: normalize q and k SEPARATELY over `head_dim`,
        // then rotate. Row-wise normalization is independent per row, so normalizing the packed
        // `q‖k` block in one kernel must reproduce this bit for bit.
        let normed_head = |part: &Tensor| {
            rms_norm(&part.reshape([rows, length, heads, head_dim])).transpose(1, 2)
        };
        let reference = [
            rope.apply_cached(&normed_head(&parts[0]), &cosine, &sine),
            rope.apply_cached(&normed_head(&parts[1]), &cosine, &sine),
            per_head(&parts[2]),
        ];
        let packed = projection.split_with_sizes([2 * width, width], -1);
        let normed_packed = rms_norm(&packed[0].reshape([rows, length, 2 * heads, head_dim]))
            .reshape([rows, length, 2 * width]);
        let rotated = fused_rope(&normed_packed, &cosine, &sine, heads).split(1, 2);
        let query_key = |tensor: &Tensor| tensor.squeeze_dim(2).transpose(1, 2);
        let fused = [
            query_key(&rotated[0]),
            query_key(&rotated[1]),
            per_head(&packed[1]),
        ];
        for (name, expected, actual) in [
            ("query", &reference[0], &fused[0]),
            ("key", &reference[1], &fused[1]),
            ("value", &reference[2], &fused[2]),
        ] {
            assert!(
                expected.abs().max().double_value(&[]) > 0.0,
                "{name} carries no signal, so equality proves nothing"
            );
            assert_eq!(
                (expected - actual).abs().max().double_value(&[]),
                0.0,
                "{name} is not bit-identical to the per-tensor rotation"
            );
        }
        // Rotation is not the identity: the kernel must actually rotate, or the test above
        // would pass on a pair of untouched projections.
        assert!(
            (&reference[0] - per_head(&parts[0])).abs().max().double_value(&[]) > 0.0,
            "the rotary rows left the query unchanged"
        );
        let attended = Tensor::scaled_dot_product_attention(
            &reference[0],
            &reference[1],
            &reference[2],
            None::<&Tensor>,
            0.0,
            true,
            None,
            false,
        )
        .transpose(1, 2)
        .reshape([rows, length, width]);
        let state = scaled_linear(&attended, &block.output, block_lambdas.post[0])
            .addcmul(&input, block_lambdas.resid[0])
            .addcmul(&x0, block_lambdas.x0);
        let expected = scaled_linear(
            &linear(&rms_norm(&state), &block.first).relu().square(),
            &block.second,
            block_lambdas.post[1],
        )
        .addcmul(&state, block_lambdas.resid[1]);
        let (actual, published) =
            block.forward(&input, &x0, None, &block_lambdas, (&cosine, &sine), false);
        // The SOURCE layer publishes exactly its own value, and mixes nothing into it.
        assert!(block.value_lambda.is_none(), "layer 0 owns no lambda");
        assert_eq!(
            (published.expect("layer 0 publishes its value").transpose(1, 2) - &fused[2])
                .abs()
                .max()
                .double_value(&[]),
            0.0,
            "the published value is not layer 0's own value"
        );
        assert_eq!(
            (&expected - &actual).abs().max().double_value(&[]),
            0.0,
            "the composed block output is not bit-identical"
        );
        let mut targets = vec![input.shallow_clone(), x0.shallow_clone()];
        targets.extend(store.trainable_variables());
        let expected_grads = Tensor::run_backward(&[&expected], &targets, true, false);
        let actual_grads = Tensor::run_backward(&[&actual], &targets, false, false);
        for (index, (left, right)) in expected_grads.iter().zip(&actual_grads).enumerate() {
            let scale = left.abs().max().double_value(&[]);
            assert!(scale > 0.0, "gradient {index} carries no signal");
            assert_eq!(
                (left - right).abs().max().double_value(&[]),
                0.0,
                "gradient {index} is not bit-identical"
            );
        }
        println!(
            "packed rotation vs per-tensor reference: 0.0e0 on q/k/v, on the composed block \
             output and on all {} gradients",
            targets.len()
        );
    }

    /// Casting each part to bf16 BEFORE the concatenation must give exactly the tensor that
    /// concatenating in fp32 and casting after gave: an elementwise cast commutes with
    /// concatenation, and that is the whole argument for moving 197 MB less per step.
    #[test]
    fn the_patch_tokens_cast_before_concatenation_are_bit_identical() {
        let _rng = crate::torch::test_rng::exclusive();
        let config = narrow_config();
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let batch = synthetic(&config, &[8, 5]);
        let stats = model.statistics(&batch);
        let (context, patch, origins) = (config.seq_len, config.patch_len, config.origins());
        let aux_channels = config.features.channels() as i64;
        let rows = batch.log_prices.size()[0];
        let inv_sigma = per_bar(&stats.sigma.reciprocal());
        let prices = (batch
            .log_prices
            .narrow(1, 0, context)
            .reshape([rows, origins, patch, CHANNELS])
            - per_bar(&stats.log_close))
            * &inv_sigma;
        let auxiliaries = batch
            .aux
            .narrow(1, 0, context)
            .reshape([rows, origins, patch, aux_channels])
            * (&model.sigma_scale * inv_sigma + &model.unit_scale);
        let expected = Tensor::cat(&[prices, auxiliaries], 3)
            .reshape([rows, origins, patch * (CHANNELS + aux_channels)])
            .to_kind(Kind::BFloat16);
        let actual = model.tokens(&batch, &stats);
        assert_eq!(actual.kind(), Kind::BFloat16);
        assert!(expected.abs().max().double_value(&[]) > 0.0);
        assert_eq!(
            (&expected - &actual).abs().max().double_value(&[]),
            0.0,
            "casting before the concatenation changed the tokens"
        );
    }

    /// RMSNorm must be `x·rsqrt(mean(x²)+eps)` over the last axis with NO gain and NO mean
    /// subtraction, computed against a scalar reference rather than against another ATen call.
    /// The mean subtraction is the part that is easy to leave in by reaching for LayerNorm with
    /// `elementwise_affine=False`: on a tensor with a nonzero row mean the two differ.
    #[test]
    fn rms_norm_matches_a_scalar_reference_and_keeps_the_row_mean() {
        let _rng = crate::torch::test_rng::exclusive();
        // A deliberate nonzero row mean, so subtracting it would change the answer.
        let rows: [[f64; 4]; 2] = [[1.0, 2.0, 3.0, 4.0], [-0.5, 0.25, 8.0, -2.0]];
        let flat: Vec<f32> = rows
            .iter()
            .flat_map(|row| row.iter().map(|&value| value as f32))
            .collect();
        let input = Tensor::from_slice(&flat).reshape([2, 4]);
        let actual = rms_norm(&input);
        for (row_index, row) in rows.iter().enumerate() {
            let mean_square = row.iter().map(|value| value * value).sum::<f64>() / 4.0;
            let scale = 1.0 / (mean_square + NORM_EPS).sqrt();
            for (column, value) in row.iter().enumerate() {
                let expected = value * scale;
                let got = actual.double_value(&[row_index as i64, column as i64]);
                assert!(
                    (got - expected).abs() < 1e-6,
                    "rms_norm[{row_index}][{column}] = {got}, expected {expected}"
                );
            }
            // A LayerNorm would have centred the row; RMSNorm must not.
            let row_mean: f64 = (0..4)
                .map(|column| actual.double_value(&[row_index as i64, column]))
                .sum::<f64>()
                / 4.0;
            let expected_mean = row.iter().sum::<f64>() / 4.0 * scale;
            assert!(
                (row_mean - expected_mean).abs() < 1e-6,
                "rms_norm centred row {row_index}: mean {row_mean}, expected {expected_mean}"
            );
        }
        // Gainless: the norm registers no variables, so there is nothing for an optimizer to
        // route. This is what makes `every_causal_patch_parameter_lands_in_its_intended_\
        // optimizer_group`'s NorMuon list exactly four matrices per block.
        let store = nn::VarStore::new(Device::Cpu);
        let before = store.len();
        let _ = rms_norm(&input.to_device(Device::Cpu));
        assert_eq!(store.len(), before, "rms_norm must register no parameters");
    }

    /// The FFN activation must be `relu(x)²`, not GELU and not `relu(x²)`. The two wrong forms
    /// differ exactly where it matters: GELU is smooth and nonzero for small negatives,
    /// `relu(x²)` is `x²` everywhere and never zero.
    #[test]
    fn relu_squared_matches_its_definition_and_is_not_gelu() {
        let points: [f64; 7] = [-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0];
        let flat: Vec<f32> = points.iter().map(|&value| value as f32).collect();
        let input = Tensor::from_slice(&flat);
        let actual = input.relu().square();
        for (index, &value) in points.iter().enumerate() {
            let expected = value.max(0.0).powi(2);
            let got = actual.double_value(&[index as i64]);
            assert!(
                (got - expected).abs() < 1e-6,
                "relu² at {value} = {got}, expected {expected}"
            );
        }
        // Nonzero separation from both plausible mistakes, on the negative half-line.
        let gelu = input.gelu("none");
        assert!(
            (&actual - &gelu).abs().max().double_value(&[]) > 1e-3,
            "relu² is indistinguishable from GELU on this range, so the test proves nothing"
        );
        assert!(
            (&actual - &input.square().relu())
                .abs()
                .max()
                .double_value(&[])
                > 1e-3,
            "relu² must not equal relu(x²)"
        );
    }

    /// QK-norm is applied per HEAD over `head_dim`, to q and k only, and BEFORE the rotation -
    /// modded-nanogpt `train_gpt.py:1106` then `:1109`. Checked against a hand-computed case
    /// small enough to write out: 1 token, 2 heads, `head_dim` 2, and a rotation angle of 0 at
    /// position 0 so the expected values are the normalized projections themselves.
    #[test]
    fn qk_norm_is_per_head_before_the_rotation_and_leaves_the_value_path_alone() {
        let _rng = crate::torch::test_rng::exclusive();
        let (heads, head_dim) = (2_i64, 2_i64);
        let width = heads * head_dim;
        // q‖k for one token: head 0 = (3, 4) with RMS √12.5, head 1 = (1, 0) with RMS √0.5,
        // then the same two heads again for k.
        let packed_rows: [f64; 8] = [3.0, 4.0, 1.0, 0.0, -2.0, 0.0, 0.5, 0.5];
        let flat: Vec<f32> = packed_rows.iter().map(|&value| value as f32).collect();
        let packed = Tensor::from_slice(&flat).reshape([1, 1, 2 * width]);
        let normed = rms_norm(&packed.reshape([1, 1, 2 * heads, head_dim]))
            .reshape([1, 1, 2 * width]);
        for head in 0..4_i64 {
            let pair = [
                packed_rows[(head * head_dim) as usize],
                packed_rows[(head * head_dim + 1) as usize],
            ];
            let scale = 1.0 / ((pair[0] * pair[0] + pair[1] * pair[1]) / 2.0 + NORM_EPS).sqrt();
            for lane in 0..head_dim {
                let expected = pair[lane as usize] * scale;
                let got = normed.double_value(&[0, 0, head * head_dim + lane]);
                assert!(
                    (got - expected).abs() < 1e-6,
                    "QK norm head {head} lane {lane} = {got}, expected {expected}"
                );
            }
        }
        // Normalizing over the whole packed axis instead of per head would give a DIFFERENT
        // answer, so the reshape is load-bearing and not decoration.
        assert!(
            (&normed - &rms_norm(&packed)).abs().max().double_value(&[]) > 1e-3,
            "per-head and whole-row normalization agree here, so this test proves nothing"
        );
        // Order: the block normalizes and THEN rotates. At position 0 the rotation is the
        // identity (cos 1, sin 0), so `rotate(norm(x))` must equal `norm(x)` exactly, while
        // `norm(rotate(x))` would too - the order is only observable off position 0. Use two
        // positions and compare the block's own path against the reference order.
        let config = ModelConfig {
            seq_len: 32,
            pred_len: 4,
            patch_len: 8,
            layers: 1,
            d_model: width,
            heads,
            ffn: 8,
            min_history: 4,
            ..Default::default()
        };
        let length = config.origins();
        let rope = RotaryEmbedding::new(length, head_dim, head_dim, Device::Cpu);
        let (cosine, sine) = rope.cached_rotation(
            &Tensor::arange(length, (Kind::Int64, Device::Cpu)),
            Kind::Float,
        );
        let projection = Tensor::randn([1, length, 2 * width], (Kind::Float, Device::Cpu));
        let norm_then_rotate = fused_rope(
            &rms_norm(&projection.reshape([1, length, 2 * heads, head_dim]))
                .reshape([1, length, 2 * width]),
            &cosine,
            &sine,
            heads,
        );
        let rotate_then_norm = rms_norm(
            &fused_rope(&projection, &cosine, &sine, heads)
                .reshape([1, length, 2 * heads, head_dim]),
        )
        .reshape([1, length, 2, heads, head_dim]);
        // RoPE preserves each head's norm, so in exact arithmetic the two orders agree; the
        // point of the assertion is that they agree to fp32 rounding and NOT further, which is
        // why the reference's order (and ours) is the one that keeps the rotation's inputs at
        // unit RMS.
        let gap = (&norm_then_rotate - &rotate_then_norm)
            .abs()
            .max()
            .double_value(&[]);
        assert!(
            gap < 1e-5,
            "rotation is not norm-preserving per head (gap {gap}), so QK-norm before RoPE is \
             not the reference's scheme"
        );
        let rotated_rms = norm_then_rotate
            .square()
            .mean_dim(-1, false, Kind::Float)
            .sqrt();
        assert!(
            (rotated_rms - 1.0).abs().max().double_value(&[]) < 1e-3,
            "the rotary inputs are not unit-RMS per head, which is the point of QK-norm"
        );
    }

    /// At initialization the whole stack must be the identity on the residual stream: every
    /// residual branch is zero (zero-init output projections, no biases) and `x0_lambda` is 0,
    /// so the only thing the eight layers do is multiply the stream by `√1.1` sixteen times -
    /// which the gainless final RMSNorm divides straight back out. `backbone` at init must
    /// therefore return the twice-normalized patch embedding and nothing else.
    ///
    /// The residual stream is bf16 (`docs/timexer_segment.md:16`), so sixteen multiplications
    /// by a bf16 `√1.1` are not exact: they round the stream by up to ~1 ULP per layer, which
    /// the final norm cannot undo because it only removes the SCALE. The test therefore pins
    /// two things - the direction to fp32 precision, and bit-exactness once the residual scale
    /// is set to 1, which isolates the rounding as the only deviation.
    ///
    /// modded-nanogpt: `train_gpt.py:1334` (post lambdas = 1), `:1338` (resid = √1.1),
    /// `:1389` (x0 lambda init 0), `:1308`/`:973-975` (zero-init branch outputs), `:1682`
    /// (final `norm`).
    #[test]
    fn the_stack_is_the_identity_on_the_residual_stream_at_init() {
        let _rng = crate::torch::test_rng::exclusive();
        let config = ModelConfig {
            layers: 8,
            ..narrow_config()
        };
        let store = nn::VarStore::new(Device::Cpu);
        let mut model = CausalPatchModel::new(&store.root(), &config);
        let batch = synthetic(&config, &[8, 5]);
        let stats = model.statistics(&batch);
        // The init values themselves, read back through the accessor the report base uses. The
        // accessor also carries the U-net gates and the value lambdas, which have their own
        // tests; this pins the three families the residual recipe owns.
        let scalars: Vec<(String, f64)> = model
            .recipe_scalars()
            .into_iter()
            .filter(|(label, _)| {
                label.starts_with("residual lambda")
                    || label.starts_with("post lambda")
                    || label.starts_with("x0 lambda")
            })
            .collect();
        assert_eq!(scalars.len(), 5 * config.layers);
        for (label, value) in &scalars {
            let reference = if label.starts_with("residual lambda") {
                RESID_LAMBDA_INIT
            } else if label.starts_with("post lambda") {
                POST_LAMBDA_INIT
            } else {
                X0_LAMBDA_INIT
            };
            // The parameters are stored fp32; the constants are f64 literals of the same value,
            // so the readback is exact only to single precision.
            assert!(
                (value - reference).abs() < 1e-7,
                "{label} = {value}, expected {reference}"
            );
        }
        assert!(scalars
            .iter()
            .any(|(label, _)| label == "residual lambda L7 ffn"));
        assert!(scalars.iter().any(|(label, _)| label == "post lambda L3 attn"));
        assert!(scalars.iter().any(|(label, _)| label == "x0 lambda L0"));
        // What the identity map returns: `x0 = rms_norm(embed)` through the final `rms_norm`.
        let x0 = rms_norm(&linear(&model.tokens(&batch, &stats), &model.patch));
        let expected = rms_norm(&x0);
        let actual = model.backbone(&batch, &stats, false, false);
        let scale = expected.abs().max().double_value(&[]);
        assert!(
            scale > 0.0,
            "the embedding carries no signal, so equality proves nothing"
        );
        // Direction: unchanged. The stack cannot have mixed anything into the stream. The bar
        // is bf16-limited, not arbitrary - the `1 - cos` a live branch produces is checked
        // against it below.
        let cosine = |left: &Tensor, right: &Tensor| {
            let (left, right) = (left.to_kind(Kind::Float), right.to_kind(Kind::Float));
            let norm = |tensor: &Tensor| tensor.square().sum(Kind::Float).double_value(&[]);
            (&left * &right).sum(Kind::Float).double_value(&[])
                / (norm(&left) * norm(&right)).sqrt()
        };
        let aligned = cosine(&expected, &actual);
        assert!(
            (1.0 - aligned).abs() < 1e-4,
            "the stack rotated the residual stream at init: cosine {aligned}"
        );
        // Magnitude: within bf16 rounding of sixteen `√1.1` multiplications (2^-8 per step).
        let gap = (&expected - &actual).abs().max().double_value(&[]) / scale;
        assert!(
            gap < 2. * (2. * config.layers as f64) * f64::powi(2., -9),
            "the stack is not the identity at init: max relative deviation {gap}"
        );
        // With the residual scale set to exactly 1 and the U-net gates closed there is nothing
        // left to round, and the identity must hold bit for bit. The gates have to be closed
        // for this line and only this line: `σ(-1.5) = 0.182` is a DELIBERATE nonzero addition
        // at init, and since every layer's output is collinear with `x0` here, a skip multiplies
        // the stream by `1.182` - a pure scale the final norm removes, but only to bf16
        // precision (measured 1.6e-2 relative, which is bf16's own ULP at this magnitude, not a
        // mixing error). The cosine and magnitude bounds above were taken with the gates OPEN,
        // so the U-net's contribution is still covered here; that it is a scale and nothing
        // more is what `closed_skip_gates_leave_the_backbone_bit_identical_to_a_plain_stack`
        // pins.
        tch::no_grad(|| {
            let _ = model.resid_lambdas.fill_(1.0);
            let _ = model.skip_weights.fill_(f64::NEG_INFINITY);
        });
        assert_eq!(
            (&expected - &model.backbone(&batch, &stats, false, false))
                .abs()
                .max()
                .double_value(&[]),
            0.0,
            "at unit residual scale the stack must be the identity exactly"
        );
        // The identity is not an accident of a dead network: the branches are live as soon as
        // one output projection is nonzero, and then the direction moves by orders of magnitude
        // more than the bf16 bar above.
        tch::no_grad(|| {
            let weight = &mut model.blocks[0].output.ws;
            let noise = Tensor::randn(weight.size(), (Kind::Float, Device::Cpu)) * 0.1;
            weight.copy_(&noise);
        });
        let moved = model.backbone(&batch, &stats, false, false);
        assert!(
            (&expected - &moved).abs().max().double_value(&[]) / scale > 1e-3,
            "a nonzero attention output projection did not change the backbone output, so the \
             identity above says nothing about the branches"
        );
        let disturbed = 1.0 - cosine(&expected, &moved);
        assert!(
            disturbed > 10. * (1.0 - aligned).abs(),
            "a live attention branch moved the direction by {disturbed}, no more than the \
             identity's own bf16 noise - the cosine bound above is vacuous"
        );
    }

    fn eight_layer_config() -> ModelConfig {
        ModelConfig {
            layers: 8,
            ..small_config()
        }
    }

    #[test]
    fn skip_gates_start_at_the_reference_sigmoid_of_minus_three_halves() {
        let config = eight_layer_config();
        let store = nn::VarStore::new(Device::Cpu);
        let model = CausalPatchModel::new(&store.root(), &config);
        let stored = store
            .variables()
            .remove("skip_weights")
            .expect("the skip gate logits must be a named varstore parameter");
        assert_eq!(stored.size(), [4]);
        assert!(stored.requires_grad());
        assert_eq!(
            (&stored - SKIP_LOGIT_INIT).abs().max().double_value(&[]),
            0.0,
            "the stored parameter is the LOGIT, so it must start at -1.5, not at 0.18"
        );
        // σ(-1.5), the reference's own annotation of this init.
        let expected = 1.0 / (1.0 + 1.5f64.exp());
        assert!((expected - 0.182_425_523_806_356_35).abs() < 1e-15);
        let reported: Vec<(String, f64)> = model
            .recipe_scalars()
            .into_iter()
            .filter(|(name, _)| name.starts_with("skip weight"))
            .collect();
        // The pairing order, which is also the order the gates are unbound in: the shared
        // accessor also carries the residual recipe's lambdas and the value residual's.
        assert_eq!(
            reported
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            [
                "skip weight 3->4",
                "skip weight 2->5",
                "skip weight 1->6",
                "skip weight 0->7"
            ]
        );
        for (name, value) in &reported {
            assert!(
                (value - expected).abs() < 1e-6,
                "{name} reported {value}, not the post-sigmoid init {expected}"
            );
        }
    }

    /// Closed gates must leave the backbone bit-identical to the same stack without the U-net,
    /// and open gates must not.
    ///
    /// The branch weights have to be randomized first, and that is not cosmetic: the residual
    /// recipe zero-initializes both output projections, so at init every layer's output is a
    /// scalar multiple of `x0`, a skip adds a multiple of `x0` to a multiple of `x0`, and the
    /// gainless final RMSNorm divides the scale straight back out - an OPEN gate would then be
    /// bit-identical too and this test would prove nothing about the skip. With live branches
    /// the encoder outputs are no longer collinear with the stream and the gate is observable.
    #[test]
    fn closed_skip_gates_leave_the_backbone_bit_identical_to_a_plain_stack() {
        let config = eight_layer_config();
        let store = nn::VarStore::new(Device::Cpu);
        let mut model = CausalPatchModel::new(&store.root(), &config);
        tch::no_grad(|| {
            for block in &mut model.blocks {
                for weight in [&mut block.output.ws, &mut block.second.ws] {
                    let noise =
                        Tensor::randn(weight.size(), (weight.kind(), weight.device())) * 0.1;
                    weight.copy_(&noise);
                }
            }
        });
        let batch = synthetic(&config, &[8, 8, 8]);
        let stats = model.statistics(&batch);
        let opened = model.backbone(&batch, &stats, false, false);
        // σ(-∞) = 0 exactly: the only way to close a sigmoid gate, and the only setting under
        // which the U-net stack is required to vanish.
        tch::no_grad(|| {
            let _ = model.skip_weights.fill_(f64::NEG_INFINITY);
        });
        let actual = model.backbone(&batch, &stats, false, false);
        // The plain stack, spelled out: the same per-layer compute `backbone` runs, with the
        // U-net fold and nothing else removed. Written as a loop over `Block::forward` rather
        // than re-deriving the block math, so this test stays about the SKIP - the residual
        // lambdas, the x0 injection and the value residual all still have to be threaded
        // exactly as the backbone threads them, or the comparison fails for the wrong reason.
        let x0 = rms_norm(&linear(&model.tokens(&batch, &stats), &model.patch));
        let kind = x0.kind();
        let resid = model.resid_lambdas.to_kind(kind).unbind(0);
        let post = model.post_lambdas.to_kind(kind).unbind(0);
        let x0_lambdas = model.x0_lambdas.to_kind(kind).unbind(0);
        let mut state = x0.shallow_clone();
        let mut first_value: Option<Tensor> = None;
        for (index, block) in model.blocks.iter().enumerate() {
            let lambdas = BlockLambdas {
                resid: [&resid[2 * index], &resid[2 * index + 1]],
                post: [&post[2 * index], &post[2 * index + 1]],
                x0: &x0_lambdas[index],
            };
            let (next, published) = block.forward(
                &state,
                &x0,
                first_value.as_ref(),
                &lambdas,
                (&model.rotation.0, &model.rotation.1),
                false,
            );
            state = next;
            if let Some(value) = published {
                first_value = Some(value);
            }
        }
        let expected = rms_norm(&state);
        assert_eq!(actual.size(), expected.size());
        assert!(
            expected.abs().max().double_value(&[]) > 0.0,
            "a zero backbone output would make this comparison vacuous"
        );
        assert!(
            actual.equal(&expected),
            "closed skip gates must not perturb a single bit of the backbone"
        );
        assert!(
            !opened.equal(&expected),
            "the skip gates at their σ(-1.5) init did not move the backbone, so closing them \
             proves nothing"
        );
    }

    #[test]
    fn the_skip_stack_pairs_the_deepest_encoder_layer_with_the_shallowest_decoder_layer() {
        let layers = 8usize;
        let options = (Kind::Double, Device::Cpu);
        let width = layers as i64;
        // Layer `i` adds a one-hot tag, so channel `i` of the result is exactly the coefficient
        // with which layer `i`'s output reached the output - which is what a pairing IS. The
        // gates are distinct primes so no two pairings can coincide by commutativity.
        let tag = |index: usize| {
            let value = Tensor::zeros([width], options);
            let _ = value.narrow(0, index as i64, 1).fill_(1.0);
            value
        };
        let gates: Vec<Tensor> = [2.0, 3.0, 5.0, 7.0]
            .into_iter()
            .map(|gate| Tensor::scalar_tensor(gate, options))
            .collect();
        let actual = unet_stack(
            Tensor::zeros([width], options),
            layers,
            &gates,
            |index, state| state + tag(index),
        );

        // The four encoder outputs, straight-line.
        let encoder_0 = Tensor::zeros([width], options) + tag(0);
        let encoder_1 = &encoder_0 + tag(1);
        let encoder_2 = &encoder_1 + tag(2);
        let encoder_3 = &encoder_2 + tag(3);
        // Straight-line, no loop: the pairing the reference specifies, one line per skip.
        let layer_4 = encoder_3.addcmul(&encoder_3, &gates[0]) + tag(4); // 3 -> 4
        let layer_5 = layer_4.addcmul(&encoder_2, &gates[1]) + tag(5); // 2 -> 5
        let layer_6 = layer_5.addcmul(&encoder_1, &gates[2]) + tag(6); // 1 -> 6
        let expected = layer_6.addcmul(&encoder_0, &gates[3]) + tag(7); // 0 -> 7
        assert!(
            actual.equal(&expected),
            "expected 3->4, 2->5, 1->6, 0->7; got {actual:?} against {expected:?}"
        );

        // Negative control: the queue order a `remove(0)` instead of a `pop` would produce. If
        // this matched, the assertion above would be proving nothing.
        let queue_4 = encoder_3.addcmul(&encoder_0, &gates[0]) + tag(4); // 0 -> 4
        let queue_5 = queue_4.addcmul(&encoder_1, &gates[1]) + tag(5); // 1 -> 5
        let queue_6 = queue_5.addcmul(&encoder_2, &gates[2]) + tag(6); // 2 -> 6
        let queue_7 = queue_6.addcmul(&encoder_3, &gates[3]) + tag(7); // 3 -> 7
        assert!(
            !actual.equal(&queue_7),
            "the tags and gates cannot distinguish stack order from queue order"
        );
    }

    #[test]
    fn an_odd_layer_count_cannot_split_into_encoder_and_decoder_halves() {
        let config = ModelConfig {
            layers: 7,
            ..small_config()
        };
        let error = config.validate().expect_err("7 layers must be rejected");
        assert!(error.to_string().contains("even"), "{error}");
    }

    /// The mixing weight is modded-nanogpt's raw per-layer scalar at 0.5, it exists on every
    /// layer EXCEPT the source, and the optimizer sees it as a 1-D non-hidden parameter - which
    /// is what keeps it on AdamW instead of NorMuon.
    #[test]
    fn the_value_lambda_is_a_raw_half_on_every_layer_but_the_source() {
        let store = nn::VarStore::new(Device::Cpu);
        let config = ModelConfig {
            layers: 4,
            ..small_config()
        };
        let model = CausalPatchModel::new(&store.root(), &config);
        assert!(
            model.blocks[0].value_lambda.is_none(),
            "the source layer must not carry a dead mixing weight"
        );
        let named: Vec<_> = store
            .variables()
            .into_iter()
            .filter(|(name, _)| name.contains("value_lambda"))
            .collect();
        let mut names: Vec<_> = named.iter().map(|(name, _)| name.clone()).collect();
        names.sort();
        assert_eq!(
            names,
            [
                "block_1.value_lambda",
                "block_2.value_lambda",
                "block_3.value_lambda"
            ]
        );
        for (name, lambda) in &named {
            assert_eq!(lambda.size(), [1], "{name} must stay 1-D");
            assert_eq!(lambda.kind(), Kind::Float, "{name} must be an fp32 master");
            assert!(
                lambda.requires_grad(),
                "{name} must be trainable, not a buffer"
            );
            // The optimizer routes `block_*` 2-D `.weight` tensors to NorMuon and everything
            // else to AdamW (`compute.rs::polar_express`), so a 1-D lambda not named `.weight`
            // is on AdamW by construction.
            assert!(!(name.ends_with(".weight") && lambda.dim() == 2));
            assert_eq!(lambda.double_value(&[0]), VALUE_LAMBDA_INIT);
        }
        // The shared accessor also carries the residual recipe's and the U-net's scalars; the
        // value residual's contribution is exactly one raw entry per non-source layer.
        assert_eq!(
            model
                .recipe_scalars()
                .into_iter()
                .filter(|(name, _)| name.starts_with("value lambda"))
                .collect::<Vec<_>>(),
            vec![
                ("value lambda L1".to_string(), 0.5),
                ("value lambda L2".to_string(), 0.5),
                ("value lambda L3".to_string(), 0.5),
            ]
        );
    }

    /// A decoder layer at λ = 0 must reproduce the model WITHOUT a value residual bit for bit -
    /// output and every gradient - and at λ = 1 must attend with the source layer's value
    /// instead of its own, also bit for bit. Both endpoints are exact because `lerp` selects
    /// `self + λ·(end - self)` below |λ| = 0.5 and `end - (end - self)·(1 - λ)` above it.
    #[test]
    fn the_value_mix_is_exactly_the_own_value_at_zero_and_the_source_value_at_one() {
        let _rng = crate::torch::test_rng::exclusive();
        let config = narrow_config();
        let store = nn::VarStore::new(Device::Cpu);
        let mut block = Block::new(store.root() / "block", &config, 1);
        // The recipe zero-initializes both branch output projections, so an untouched block
        // returns its input whatever V it attended with and BOTH endpoints would agree at 0.
        // Live branches are what makes the value the output depends on.
        tch::no_grad(|| {
            for weight in [&mut block.output.ws, &mut block.second.ws] {
                let noise = Tensor::randn(weight.size(), (Kind::Float, Device::Cpu)) * 0.05;
                weight.copy_(&noise);
            }
        });
        let mut lambda = block
            .value_lambda
            .as_ref()
            .expect("a decoder layer carries a lambda")
            .shallow_clone();
        let (width, heads) = (config.d_model, config.heads);
        let head_dim = width / heads;
        let (rows, length) = (2, config.origins());
        let rope = RotaryEmbedding::new(length, head_dim, head_dim, Device::Cpu);
        let (cosine, sine) = rope.cached_rotation(
            &Tensor::arange(length, (Kind::Int64, Device::Cpu)),
            Kind::BFloat16,
        );
        let input = (Tensor::randn([rows, length, width], (Kind::Float, Device::Cpu)) * 0.5)
            .to_kind(Kind::BFloat16)
            .set_requires_grad(true);
        let first = (Tensor::randn(
            [rows, length, heads, head_dim],
            (Kind::Float, Device::Cpu),
        ) * 0.5)
            .to_kind(Kind::BFloat16);
        let x0 = (Tensor::randn([rows, length, width], (Kind::Float, Device::Cpu)) * 0.5)
            .to_kind(Kind::BFloat16);
        let (resid_parts, post_parts) = (
            Tensor::from_slice(&[RESID_LAMBDA_INIT, 0.9])
                .to_kind(Kind::BFloat16)
                .unbind(0),
            Tensor::from_slice(&[POST_LAMBDA_INIT, 1.1])
                .to_kind(Kind::BFloat16)
                .unbind(0),
        );
        let x0_scale = Tensor::from_slice(&[0.25]).to_kind(Kind::BFloat16).get(0);
        let block_lambdas = BlockLambdas {
            resid: [&resid_parts[0], &resid_parts[1]],
            post: [&post_parts[0], &post_parts[1]],
            x0: &x0_scale,
        };
        // The model WITHOUT the residual: the same block math `Block::forward` runs - gainless
        // pre-norm, QK-norm before the rotation, ReLU² feedforward, folded post-lambdas and
        // `addcmul` residual lines - reading either its own value or a replacement standing in
        // for it. Only the V operand differs from the production path.
        let unmixed = |replacement: Option<&Tensor>| {
            let projection = linear(&rms_norm(&input), &block.qkv);
            let packed = projection.split_with_sizes([2 * width, width], -1);
            let normed = rms_norm(&packed[0].reshape([rows, length, 2 * heads, head_dim]))
                .reshape([rows, length, 2 * width]);
            let rotated = fused_rope(&normed, &cosine, &sine, heads).split(1, 2);
            let query_key = |tensor: &Tensor| tensor.squeeze_dim(2).transpose(1, 2);
            let value = replacement
                .map(Tensor::shallow_clone)
                .unwrap_or_else(|| packed[1].reshape([rows, length, heads, head_dim]));
            let attended = Tensor::scaled_dot_product_attention(
                &query_key(&rotated[0]),
                &query_key(&rotated[1]),
                &value.transpose(1, 2),
                None::<&Tensor>,
                0.0,
                true,
                None,
                false,
            )
            .transpose(1, 2)
            .reshape([rows, length, width]);
            let state = scaled_linear(&attended, &block.output, block_lambdas.post[0])
                .addcmul(&input, block_lambdas.resid[0])
                .addcmul(&x0, block_lambdas.x0);
            scaled_linear(
                &linear(&rms_norm(&state), &block.first).relu().square(),
                &block.second,
                block_lambdas.post[1],
            )
            .addcmul(&state, block_lambdas.resid[1])
        };
        let own_value = {
            let projection = linear(&rms_norm(&input), &block.qkv);
            projection
                .split_with_sizes([2 * width, width], -1)[1]
                .reshape([rows, length, heads, head_dim])
        };
        assert!(
            (&own_value - &first).abs().max().double_value(&[]) > 0.0,
            "the two values coincide, so neither endpoint proves anything"
        );
        assert!(
            (&unmixed(None) - &unmixed(Some(&first)))
                .abs()
                .max()
                .double_value(&[])
                > 0.0,
            "the block output does not depend on V at all, so the endpoints are vacuous"
        );
        for (name, weight, expected) in [
            ("zero", 0.0, unmixed(None)),
            ("one", 1.0, unmixed(Some(&first))),
        ] {
            tch::no_grad(|| {
                let _ = lambda.fill_(weight);
            });
            let (actual, published) = block.forward(
                &input,
                &x0,
                Some(&first),
                &block_lambdas,
                (&cosine, &sine),
                false,
            );
            assert!(
                published.is_none(),
                "only the source layer may publish a value"
            );
            assert!(expected.abs().max().double_value(&[]) > 0.0);
            assert_eq!(
                (&expected - &actual).abs().max().double_value(&[]),
                0.0,
                "lambda = {name} is not the exact endpoint"
            );
            let mut targets = vec![input.shallow_clone()];
            targets.extend(
                store
                    .trainable_variables()
                    .into_iter()
                    .filter(|tensor| tensor.dim() == 2),
            );
            let expected_grads = Tensor::run_backward(&[&expected], &targets, true, false);
            let actual_grads = Tensor::run_backward(&[&actual], &targets, true, false);
            for (index, (left, right)) in expected_grads.iter().zip(&actual_grads).enumerate() {
                assert!(
                    left.abs().max().double_value(&[]) > 0.0,
                    "gradient {index} carries no signal"
                );
                assert_eq!(
                    (left - right).abs().max().double_value(&[]),
                    0.0,
                    "gradient {index} moved at lambda = {name}"
                );
            }
        }
        // The mix must still TEACH the lambda at λ = 0, or the residual could never turn on:
        // ∂/∂λ = Σ g·(v_1 - v) is nonzero even where the forward is the identity.
        tch::no_grad(|| {
            let _ = lambda.fill_(0.0);
        });
        let (output, _) = block.forward(
            &input,
            &x0,
            Some(&first),
            &block_lambdas,
            (&cosine, &sine),
            false,
        );
        let grad = Tensor::run_backward(&[&output], &[lambda.shallow_clone()], false, false);
        assert!(
            grad[0].abs().double_value(&[0]) > 0.0,
            "lambda is unlearnable at its identity point"
        );
    }

    /// End to end: with λ = 1 every decoder layer must be attending with LAYER 0's value, so
    /// destroying a decoder layer's own value projection cannot move the output - while
    /// destroying layer 0's must. At λ = 0 the dependence is exactly the other way round.
    #[test]
    fn every_decoder_layer_attends_with_the_source_layers_value_at_lambda_one() {
        let _rng = crate::torch::test_rng::exclusive();
        // FOUR layers, not three: the U-net needs an even count, and four gives one encoder
        // pair plus two decoder layers that mix.
        let config = ModelConfig {
            layers: 4,
            ..small_config()
        };
        let store = nn::VarStore::new(Device::Cpu);
        let mut model = CausalPatchModel::new(&store.root(), &config);
        // Zero-init branch projections would make the backbone independent of V entirely, so
        // "breaking a value projection does not move the output" would hold for the wrong
        // reason. Live branches make V observable.
        tch::no_grad(|| {
            for block in &mut model.blocks {
                for weight in [&mut block.output.ws, &mut block.second.ws] {
                    let noise =
                        Tensor::randn(weight.size(), (weight.kind(), weight.device())) * 0.05;
                    weight.copy_(&noise);
                }
            }
        });
        let batch = synthetic(&config, &[8, 5]);
        let stats = model.statistics(&batch);
        let width = config.d_model;
        let set_lambdas = |weight: f64| {
            tch::no_grad(|| {
                for block in &model.blocks {
                    if let Some(lambda) = &block.value_lambda {
                        let _ = lambda.shallow_clone().fill_(weight);
                    }
                }
            })
        };
        // The value rows of a layer's packed `q‖k‖v` projection. The recipe leaves the four
        // block projections bias-free, so the weight rows are the whole of the V path.
        let break_value = |layer: usize| {
            tch::no_grad(|| {
                let qkv = &model.blocks[layer].qkv;
                assert!(qkv.bs.is_none(), "the block projections are bias-free");
                let _ = qkv.ws.narrow(0, 2 * width, width).fill_(0.0);
            })
        };
        let output = || {
            model
                .backbone(&batch, &stats, false, false)
                .to_kind(Kind::Float)
        };
        let moved = |before: &Tensor, after: &Tensor| {
            (before - after).abs().max().double_value(&[]) > 0.0
        };
        set_lambdas(1.0);
        let reference = output();
        assert!(reference.abs().max().double_value(&[]) > 0.0);
        break_value(1);
        assert!(
            !moved(&reference, &output()),
            "layer 1 still reads its own value at lambda = 1"
        );
        break_value(2);
        assert!(
            !moved(&reference, &output()),
            "layer 2 still reads its own value at lambda = 1"
        );
        // The same broken decoder projections must matter again at lambda = 0, which they only
        // can if the mix - not the plumbing - is what silenced them above. Checked BEFORE layer
        // 0 is broken: with every value projection dead the two settings agree trivially.
        set_lambdas(0.0);
        assert!(
            moved(&reference, &output()),
            "lambda = 0 did not restore each layer's own value"
        );
        set_lambdas(1.0);
        assert!(
            !moved(&reference, &output()),
            "the stack did not return to the source layer's value"
        );
        // Layer 0 is the source, so at lambda = 1 the whole stack depends on it alone.
        break_value(0);
        assert!(
            moved(&reference, &output()),
            "the stack ignored the source layer's value"
        );
    }
}
