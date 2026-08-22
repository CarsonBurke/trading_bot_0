//! Reference-aligned NorMuon optimizer: EMA momentum/Nesterov, Newton-Schulz 5
//! orthogonalization, per-row second-moment (NorMuon) rescaling with Frobenius-
//! norm preservation, and AdamW for non-matrix params.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{ensure, Context, Result};
use tch::{Device, Kind, Tensor};

const NS_A: f64 = 3.4445;
const NS_B: f64 = -4.7750;
const NS_C: f64 = 2.0315;
/// Canonical Newton-Schulz iteration count; the reference default and the only
/// value real training should ever use.
pub const DEFAULT_NS_STEPS: usize = 5;
/// Primary steps during which the row-wise learned learning-rate controller only
/// observes credit and leaves every multiplier at one.
pub const ROW_LR_WARMUP_STEPS: i64 = 100;
/// Adam learning rate for the row-wise controller.
pub const ROW_LR_CONTROLLER_LR: f64 = 1e-3;
pub const ROW_LR_CONTROLLER_BETAS: (f64, f64) = (0.9, 0.999);
pub const ROW_LR_CONTROLLER_EPS: f64 = 1e-8;
/// Full log-alpha span. `log(alpha) = span * (sigmoid(logit) - 1/2)`;
/// `span = 2` is the reference `c = 1`, bounding alpha in `[exp(-1), exp(1)]`.
pub const ROW_LR_LOG_SPAN: f64 = 2.0;

pub(crate) fn newton_schulz_polynomial_bits() -> [u64; 3] {
    [NS_A.to_bits(), NS_B.to_bits(), NS_C.to_bits()]
}

/// Coefficient triple `(a, b, c)` of one quintic orthogonalization step
/// `x <- a*x + (b*A + c*A*A)*x`, with `A = x*xᵀ` on the wide orientation.
type QuinticCoeffs = (f64, f64, f64);

/// Polar Express quintic schedule (`num_iters=5, safety_factor=2e-2, cushion=2`),
/// from <https://arxiv.org/pdf/2505.16932> as shipped in modded-nanogpt
/// `train_gpt.py:162-168`. Unlike Newton-Schulz this is deliberately *not* a
/// convergent fixed-point iteration: every step has its own triple and the
/// composition is tuned for exactly five steps. Never "correct" these
/// coefficients, never terminate early, and never iterate past the schedule.
const POLAR_EXPRESS_COEFFS: [QuinticCoeffs; 5] = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
];

/// Polar Express spectral-norm safety factor and floor (`train_gpt.py:198`):
/// `x <- x / (‖x‖_F * (1 + safety) + floor)`.
const POLAR_EXPRESS_SAFETY: f64 = 2e-2;
const POLAR_EXPRESS_FLOOR: f64 = 1e-6;

/// Which quintic iteration approximates the orthogonal polar factor.
///
/// Both cost the same five bf16 quintic steps; Polar Express converges markedly
/// closer to orthogonality on ill-conditioned gradients. Newton-Schulz remains
/// the default so existing PPO and planner runs stay bit-identical.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Orthogonalizer {
    #[default]
    NewtonSchulz5,
    PolarExpress5,
}

impl Orthogonalizer {
    /// Newton-Schulz repeats one tuned triple `ns_steps` times; Polar Express runs
    /// its own five-step schedule and ignores `ns_steps`.
    fn steps(self, ns_steps: usize) -> usize {
        match self {
            Self::NewtonSchulz5 => ns_steps,
            Self::PolarExpress5 => POLAR_EXPRESS_COEFFS.len(),
        }
    }

    fn coeffs(self, step: usize) -> QuinticCoeffs {
        match self {
            Self::NewtonSchulz5 => (NS_A, NS_B, NS_C),
            Self::PolarExpress5 => POLAR_EXPRESS_COEFFS[step],
        }
    }
}

pub struct MuonConfig {
    pub lr: f64,
    pub use_muon_for_2d: bool,
    pub momentum: f64,
    pub nesterov: bool,
    /// NorMuon second-moment EMA decay (beta2). Reference default 0.95.
    pub beta2: f64,
    pub weight_decay: f64,
    /// AdamW LR for scalar/1D params (biases, norms, embeddings).
    pub adamw_lr: f64,
    pub adamw_betas: (f64, f64),
    pub adamw_eps: f64,
    pub adamw_wd: f64,
    /// Parameter name fragments excluded from AdamW's decoupled weight decay.
    pub adamw_no_weight_decay_name_substrings: Vec<String>,
    /// Newton-Schulz iteration count for orthogonalization. Reference default 5.
    /// Exposed only so offline sweeps can map the NS-steps landscape; real
    /// training must leave this at `DEFAULT_NS_STEPS`.
    pub ns_steps: usize,
    /// Parameter name fragments that should use AdamW even if they are 2D.
    pub force_adamw_name_substrings: Vec<String>,
    /// Optional allowlist for Muon-routed 2D parameters. Empty permits every
    /// otherwise-eligible matrix; the AdamW blocklist always takes precedence.
    pub muon_name_allowlist: Vec<String>,
    /// Benchmark/experiment mode: split attention projection matrices into
    /// per-head 2D blocks before Newton-Schulz orthogonalization.
    pub per_attention_head_ortho: bool,
    /// Include attention output projections in `per_attention_head_ortho`.
    pub per_attention_output_head_ortho: bool,
    /// Self-attention head width used by `per_attention_head_ortho`.
    pub attention_head_dim: i64,
    /// Cross-attention head width used by `per_attention_head_ortho`.
    pub cross_attention_head_dim: i64,
    /// Suppress the one-line routing-split print at construction. Benchmarks
    /// that build many optimizers set this; real training leaves it false.
    pub quiet: bool,
    /// Quintic iteration used to orthogonalize the momentum buffer.
    pub orthogonalizer: Orthogonalizer,
    /// Apply the AdamW update once every N steps, accumulating the intervening gradients
    /// rather than discarding them. `1` updates every step and is the default, so every
    /// existing caller keeps its behaviour.
    ///
    /// modded-nanogpt sets the equivalent of `2` (`train_gpt.py:2104-2106`,
    /// `_is_adam_step(step) = step % 2 == 1`) and skips `param.grad = None` for Adam params
    /// on the intervening step (`:821-823`), so the update sees a summed, effective 2x
    /// batch. Adam is invariant to the gradient's overall scale, so the step size is
    /// unchanged and the gain is variance reduction plus half the optimizer work on the
    /// embedding tables and output heads. It is a property of THAT recipe, which is why it
    /// lives here rather than in `step`: the PPO and planner trainers share this optimizer
    /// and have no reason to inherit it.
    pub adamw_every: usize,
    /// Scale the decoupled weight decay by `lr` a second time, so the decay is
    /// quadratic in the learning rate. NorMuon then decays by
    /// `p * (wd*lr) * (lr_mul*per_matrix_lr_mul*lr)` and AdamW by `p * lr*lr*wd`
    /// (modded-nanogpt `train_gpt.py:845` and `:877-878` with `:928`). The NorMuon
    /// form picks the per-parameter multipliers up in only one of the two factors,
    /// so a matrix at 4x lr_mul decays 16x harder; that asymmetry is intentional.
    pub quadratic_lr_weight_decay: bool,
    /// Skip weight decay on coordinates where the step and the parameter disagree
    /// in sign. NorMuon masks non-strictly on `(update * p) >= 0`
    /// (`train_gpt.py:915-932`); AdamW masks strictly on `(update * p) > 0`
    /// (`train_gpt.py:856-863`). The strictness difference is reproduced verbatim.
    pub cautious_weight_decay: bool,
    /// Per-parameter AdamW beta overrides as `(name fragment, (beta1, beta2))`.
    /// First match wins; unmatched parameters use `adamw_betas`.
    pub adamw_beta_overrides: Vec<(String, (f64, f64))>,
    /// Per-parameter AdamW weight-decay multipliers as `(name fragment, wd_mul)`.
    /// First match wins; unmatched parameters use `1.0`. The reference gives the
    /// embedding tables and the output head `wd_mul = 150` (modded-nanogpt
    /// `train_gpt.py:2033`,`:2038`), which is what makes a quadratic-in-lr decay
    /// bite at all. `adamw_no_weight_decay_name_substrings` remains the `wd_mul = 0`
    /// case and takes precedence.
    pub adamw_weight_decay_multipliers: Vec<(String, f64)>,
    /// Enable the row-wise learned learning-rate controller on Muon-routed
    /// matrices. Disabled by default; the disabled branch allocates no controller
    /// tensors and leaves the existing optimizer arithmetic untouched.
    pub row_learned_lr: bool,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: 5e-3,
            use_muon_for_2d: true,
            momentum: 0.99,
            nesterov: true,
            beta2: 0.95,
            weight_decay: 0.0,
            adamw_lr: 3e-4,
            adamw_betas: (0.9, 0.95),
            adamw_eps: 1e-8,
            adamw_wd: 0.0,
            adamw_no_weight_decay_name_substrings: Vec::new(),
            ns_steps: DEFAULT_NS_STEPS,
            force_adamw_name_substrings: Vec::new(),
            muon_name_allowlist: Vec::new(),
            per_attention_head_ortho: false,
            per_attention_output_head_ortho: true,
            attention_head_dim: 0,
            cross_attention_head_dim: 0,
            quiet: false,
            orthogonalizer: Orthogonalizer::NewtonSchulz5,
            adamw_every: 1,
            quadratic_lr_weight_decay: false,
            cautious_weight_decay: false,
            adamw_beta_overrides: Vec::new(),
            adamw_weight_decay_multipliers: Vec::new(),
            row_learned_lr: false,
        }
    }
}

/// Which clock a [`Muon::step`] advances.
///
/// AdamW's [`MuonConfig::adamw_every`] cadence has to be a property of the PRIMARY
/// optimization sequence. An extra update from a side stream that advanced the same
/// counter would shift AdamW's phase and with it its effective learning rate, silently
/// and only when the side stream is enabled. The kind is a required argument so a new
/// extra-step call site must state which sequence it belongs to rather than inherit the
/// primary one by writing nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepKind {
    /// A step of the primary training sequence. Advances the step counter, and with it
    /// the AdamW cadence.
    Primary,
    /// An extra update outside the primary sequence, e.g. one auxiliary resolution's
    /// share of a pretraining step. NorMuon parameters take it immediately; the AdamW
    /// cadence never sees it and the AdamW gradients it leaves are held for the next
    /// primary tick, so enabling a side stream can change neither AdamW's update count
    /// per primary step nor the step ledger a checkpoint is validated against.
    Auxiliary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrthoLayout {
    Matrix,
    RowHeads { heads: i64, head_dim: i64 },
    ColHeads { heads: i64, head_dim: i64 },
}

/// Per-2D-param state. Each 2D param holds its own momentum.
/// We deliberately do *not* stack same-shape params into a batched state tensor:
/// the N× VRAM multiplier from batching NS5's intermediates dwarfs any step-time
/// savings on realistic models. Keeping state per-param caps NS5's peak working
/// set at a single [m, n] matrix's worth of transients.
struct Entry2D {
    idx: usize,
    layout: OrthoLayout,
    /// First-moment EMA buffer, shape [m, n].
    momentum: Tensor,
    /// NorMuon second-moment EMA buffer: mean of squares over the SHORTER of the last
    /// two axes, so it carries one moment per element of the LONGER axis. Its shape is
    /// exactly `second_momentum_shape` of the block shape the orthogonalizer produces.
    /// Kept in fp32 regardless of param dtype: an EMA at gain (1-beta2)=0.05 in
    /// bf16 silently stalls because small increments round to zero.
    second_momentum: Tensor,
    /// Allocated only when `row_learned_lr` is enabled.
    row_lr: Option<RowLrState>,
}

struct RowLrState {
    /// One logit and Adam pair per parameter row, always fp32 on the parameter device.
    logit: Tensor,
    adam_m: Tensor,
    adam_v: Tensor,
    /// Previous PRIMARY step's actual signed parameter delta, excluding decoupled decay.
    previous_delta: Tensor,
    adam_step: i64,
}

/// Number and order of controller diagnostics in
/// [`Muon::row_learned_lr_metrics_tensor`].
pub const ROW_LR_METRIC_COUNT: usize = 9;

/// Host representation of the latest primary-step diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowLearnedLrMetrics {
    pub alpha_mean: f64,
    pub alpha_std: f64,
    pub alpha_min: f64,
    pub alpha_max: f64,
    pub alpha_bound_fraction: f64,
    pub evidence_mean: f64,
    pub evidence_std: f64,
    pub objective: f64,
    pub update_magnitude: f64,
}

#[derive(Default)]
struct RowLrMetricAccumulator {
    observations: Vec<RowLrObservation>,
}

struct RowLrObservation {
    alpha: Tensor,
    evidence: Tensor,
    objective_per_row: Tensor,
    update_magnitude: Tensor,
}

struct AdamWParamState {
    m: Tensor,
    v: Tensor,
    step_count: i64,
}

pub struct Muon {
    cfg: MuonConfig,
    entries_2d: Vec<Entry2D>,
    adamw_indices: Vec<usize>,
    adamw_state: HashMap<usize, AdamWParamState>,
    /// Count of PRIMARY optimization steps. Auxiliary steps deliberately do not advance
    /// it: it is both the AdamW cadence clock and the step ledger a checkpoint's metadata
    /// is validated against, and both mean "training steps taken".
    step_count: i64,
    params: Vec<Tensor>,
    /// Variable name per param index; keys optimizer-state sidecar tensors so
    /// restoration is robust to param ordering. Empty for the unnamed `new` path.
    names: Vec<String>,
    /// Per-parameter multiplier over the routed optimizer's base learning rate.
    /// These are runtime schedule values, not optimizer moments; callers restore
    /// them from their training-controller state after loading a checkpoint.
    lr_scales: Vec<f64>,
    /// Runtime parameter-step mask. Disabled parameters retain both their value
    /// and optimizer moments, even when an earlier backward left a defined zero
    /// gradient in their slot.
    step_enabled: Vec<bool>,
    /// True when the last [`Self::step`] deliberately skipped the AdamW update, so those
    /// parameters still hold the gradient the next backward must accumulate into.
    /// [`Self::zero_grad`] reads it and leaves those slots alone.
    adamw_pending_grads: bool,
    /// Present only when the controller is enabled and the latest primary step
    /// observed at least one routed matrix gradient. Kept device-resident so the
    /// pretrainer can fold it into its existing single metrics transfer.
    row_lr_metrics: Option<Tensor>,
}

/// Single-matrix quintic orthogonalization.
/// Runs the orthogonalizer's schedule in bf16 for speed; returns in the input's kind.
///
/// Each iteration is:  x ← a·x + b·(A·x) + c·((A·A)·x), where A = x·xᵀ.
/// Factoring the two correction terms over the shared right-multiply by `x`,
///   b·(A·x) + c·((A·A)·x) = (b·A + c·(A·A))·x = B·x,
/// collapses the iteration to **three** matmuls (x·xᵀ, A·A, B·x) instead of the
/// four a naïve expansion needs, expressed as two `baddbmm` calls on [1,p,q]
/// views. tch's 2D `addmm` doesn't accept scalar beta/alpha, so we lift to 3D.
///
/// Peak live tensors during iter: `x` ([p,q]) + `A` ([p,p]) + `B` ([p,p]) =
/// ~[p,q] + 2·[p,p]. For p == q this is ~3·[p,q]. This is the inherent
/// working-set cost of a quintic iteration and cannot be eliminated without
/// changing the algorithm.
fn quintic_orthogonalize(g: &Tensor, orth: Orthogonalizer, ns_steps: usize) -> Tensor {
    let orig_kind = g.kind();
    let transposed = g.size()[0] > g.size()[1];
    let x2d = if orig_kind == Kind::BFloat16 {
        g.shallow_clone()
    } else {
        g.to_kind(Kind::BFloat16)
    };
    let nrm = prescale_divisor(&x2d.norm(), orth);
    let x2d = &x2d / &nrm;
    let x2d = if transposed { x2d.transpose(0, 1) } else { x2d };
    let mut x = x2d.unsqueeze(0); // [1, p, q] view — baddbmm needs 3D

    for step in 0..orth.steps(ns_steps) {
        let (ca, cb, cc) = orth.coeffs(step);
        let a = x.matmul(&x.transpose(-2, -1));
        // B = b·A + c·(A·A)  — fold both corrections into one matrix.
        let b = a.baddbmm(&a, &a, cb, cc);
        // x = a·x + B·x
        x = x.baddbmm(&b, &x, ca, 1.0);
    }

    let x = x.squeeze_dim(0);
    let x = if transposed {
        x.transpose(0, 1).contiguous()
    } else {
        x
    };
    if orig_kind == Kind::BFloat16 {
        x
    } else {
        x.to_kind(orig_kind)
    }
}

fn batched_quintic_orthogonalize(g: &Tensor, orth: Orthogonalizer, ns_steps: usize) -> Tensor {
    let orig_kind = g.kind();
    let transposed = g.size()[1] > g.size()[2];
    let x3d = if orig_kind == Kind::BFloat16 {
        g.shallow_clone()
    } else {
        g.to_kind(Kind::BFloat16)
    };
    let nrm = prescale_divisor(
        &x3d.square()
            .sum_dim_intlist([-2i64, -1].as_slice(), true, Kind::BFloat16)
            .sqrt(),
        orth,
    );
    let x3d = &x3d / &nrm;
    let mut x = if transposed {
        x3d.transpose(-2, -1).contiguous()
    } else {
        x3d
    };

    for step in 0..orth.steps(ns_steps) {
        let (ca, cb, cc) = orth.coeffs(step);
        let a = x.matmul(&x.transpose(-2, -1));
        let b = a.baddbmm(&a, &a, cb, cc);
        x = x.baddbmm(&b, &x, ca, 1.0);
    }

    let x = if transposed {
        x.transpose(-2, -1).contiguous()
    } else {
        x
    };
    if orig_kind == Kind::BFloat16 {
        x
    } else {
        x.to_kind(orig_kind)
    }
}

/// Divisor that brings the spectral norm below one before the iteration starts.
/// Newton-Schulz normalizes by the Frobenius norm with a hard floor; Polar Express
/// leaves a `1 + 2e-2` safety cushion instead, because its first step has a very
/// large leading coefficient and overshoots if any singular value exceeds one.
fn prescale_divisor(frobenius: &Tensor, orth: Orthogonalizer) -> Tensor {
    match orth {
        Orthogonalizer::NewtonSchulz5 => frobenius.clamp_min(1e-7),
        Orthogonalizer::PolarExpress5 => {
            frobenius * (1.0 + POLAR_EXPRESS_SAFETY) + POLAR_EXPRESS_FLOOR
        }
    }
}

#[cfg(test)]
fn newtonschulz5(g: &Tensor, ns_steps: usize) -> Tensor {
    quintic_orthogonalize(g, Orthogonalizer::NewtonSchulz5, ns_steps)
}

#[cfg(test)]
fn batched_newtonschulz5(g: &Tensor, ns_steps: usize) -> Tensor {
    batched_quintic_orthogonalize(g, Orthogonalizer::NewtonSchulz5, ns_steps)
}

fn attention_ortho_layout(name: &str, size: &[i64], cfg: &MuonConfig) -> OrthoLayout {
    if !cfg.per_attention_head_ortho || size.len() != 2 {
        return OrthoLayout::Matrix;
    }
    let Some(head_dim) = attention_head_dim_for_name(name, cfg) else {
        return OrthoLayout::Matrix;
    };

    let rows = size[0];
    let cols = size[1];
    let is_output = is_attention_output_projection_name(name);
    if is_output && !cfg.per_attention_output_head_ortho {
        OrthoLayout::Matrix
    } else if is_output && cols % head_dim == 0 {
        OrthoLayout::ColHeads {
            heads: cols / head_dim,
            head_dim,
        }
    } else if parameter_leaf_name(name) == "attn_qkv"
        && rows == 3 * cols
        && rows % (3 * head_dim) == 0
    {
        // Fused [3*dim, dim] QKV, split into per-head [3*head_dim, dim] blocks. Reached by
        // `optim_head_ortho`'s `FusedQkvGptModel` (benchmarks/src/optim_transformer.rs:131),
        // not by the bar model, whose fused weight is named `qkv_w`. The reference does not
        // fuse at all: it keeps a `qk_bank` and a `vo_bank`, orthogonalizes Q/K per
        // head-pair and V per layer (modded-nanogpt `train_gpt.py:1523-1525`, `:2027-2028`),
        // so a future per-head experiment should start from that split rather than from this
        // block form.
        OrthoLayout::RowHeads {
            heads: rows / (3 * head_dim),
            head_dim: 3 * head_dim,
        }
    } else if rows % head_dim == 0 {
        OrthoLayout::RowHeads {
            heads: rows / head_dim,
            head_dim,
        }
    } else {
        OrthoLayout::Matrix
    }
}

fn attention_head_dim_for_name(name: &str, cfg: &MuonConfig) -> Option<i64> {
    if is_cross_attention_projection_name(name) {
        (cfg.cross_attention_head_dim > 0).then_some(cfg.cross_attention_head_dim)
    } else if is_self_attention_projection_name(name) {
        (cfg.attention_head_dim > 0).then_some(cfg.attention_head_dim)
    } else {
        None
    }
}

/// A parameter's own name: the final segment of its VarStore path, which tch joins with `.`.
///
/// The attention matchers compare against this, EXACTLY. Substring-matching the whole path
/// made `bar_layer_3.attn_out_w` satisfy the `attn_o` matcher, so enabling
/// `per_attention_head_ortho` on the bar model would have silently split a [512, 512] output
/// projection into column head-blocks. One tensor's name being a prefix of another's is not
/// a hazard a routing table can be reviewed for; exact leaf comparison removes it.
fn parameter_leaf_name(name: &str) -> &str {
    match name.rfind('.') {
        Some(dot) => &name[dot + 1..],
        None => name,
    }
}

fn is_self_attention_projection_name(name: &str) -> bool {
    matches!(
        parameter_leaf_name(name),
        "attn_q" | "attn_k" | "attn_v" | "attn_qkv" | "attn_o"
    )
}

fn is_cross_attention_projection_name(name: &str) -> bool {
    matches!(
        parameter_leaf_name(name),
        "ca_q" | "ca_k" | "ca_v" | "ca_out"
    )
}

fn is_attention_output_projection_name(name: &str) -> bool {
    matches!(parameter_leaf_name(name), "attn_o" | "ca_out")
}

#[cfg(test)]
fn orthogonalize_update(
    update: &Tensor,
    layout: OrthoLayout,
    orth: Orthogonalizer,
    ns_steps: usize,
) -> Tensor {
    match layout {
        OrthoLayout::Matrix => quintic_orthogonalize(update, orth, ns_steps),
        OrthoLayout::RowHeads { heads, head_dim } => {
            let cols = update.size()[1];
            batched_quintic_orthogonalize(&update.reshape([heads, head_dim, cols]), orth, ns_steps)
                .reshape(update.size().as_slice())
        }
        OrthoLayout::ColHeads { heads, head_dim } => {
            let rows = update.size()[0];
            batched_quintic_orthogonalize(
                &update
                    .reshape([rows, heads, head_dim])
                    .permute([1, 0, 2])
                    .contiguous(),
                orth,
                ns_steps,
            )
            .permute([1, 0, 2])
            .contiguous()
            .reshape(update.size().as_slice())
        }
    }
}

/// Axis the NorMuon second moment reduces over, as a negative index into the last two
/// dims of the matrix (or stack of matrices) the orthogonalizer produced.
///
/// The reference keeps ONE moment per element of the LONGER axis and reduces over the
/// shorter one: `red_dim = -1 if chunk_shape[-2] >= chunk_shape[-1] else -2`
/// (modded-nanogpt `train_gpt.py:888`, with the matching buffer shape at `:566-569`).
/// Reducing over the fan-in unconditionally preconditions a wide matrix along its short
/// axis and estimates every moment from fewer samples than the reference does.
fn normuon_reduce_dim(size: &[i64]) -> i64 {
    let rank = size.len();
    if size[rank - 2] >= size[rank - 1] {
        -1
    } else {
        -2
    }
}

/// Shape the orthogonalizer and the rescale actually operate on for `layout`, and the
/// only shape the second-moment buffer may be derived from.
fn ortho_block_shape(size: &[i64], layout: OrthoLayout) -> Vec<i64> {
    match layout {
        OrthoLayout::Matrix => vec![size[0], size[1]],
        OrthoLayout::RowHeads { heads, head_dim } => vec![heads, head_dim, size[1]],
        OrthoLayout::ColHeads { heads, head_dim } => vec![heads, size[0], head_dim],
    }
}

/// Second-moment buffer shape: the block shape with the reduced axis collapsed to 1.
/// Derived from [`normuon_reduce_dim`] so buffer and reduction cannot disagree.
fn second_momentum_shape(size: &[i64], layout: OrthoLayout) -> Vec<i64> {
    let mut shape = ortho_block_shape(size, layout);
    let rank = shape.len();
    let axis = if normuon_reduce_dim(&shape) == -1 {
        rank - 1
    } else {
        rank - 2
    };
    shape[axis] = 1;
    shape
}

/// NorMuon second-moment rescale for a matrix or a stack of matrices, all math in fp32.
/// Scaling each slice along the reduced axis by `step_size * ratio` keeps the total
/// Frobenius norm of the update (approximately) equal to its pre-divide value, because
/// `ratio` is exactly the global correction `||U||_F / ||diag(step_size) U||_F`. The
/// `lerp_` writes the raw second-moment EMA in place. Reference: `train_gpt.py:936-947`.
fn normuon_rescale(update: &Tensor, second_momentum: &mut Tensor, beta2: f64) -> Tensor {
    let size = update.size();
    let rank = size.len();
    let red_dim = normuon_reduce_dim(&size);
    let red_len = size[if red_dim == -1 { rank - 1 } else { rank - 2 }] as f64;
    let uf = update.to_kind(Kind::Float);
    // Sum of squares over the shorter axis, keepdim: the buffer's shape.
    let sq_sum = uf
        .square()
        .sum_dim_intlist([red_dim].as_slice(), true, Kind::Float);
    // Per-slice MEAN of squares (note the /red_len).
    let v_mean = &sq_sum / red_len;
    // Frobenius^2 of the post-NS update, per matrix, BEFORE the divide.
    let vnorm_sq = sq_sum.sum_dim_intlist([-2i64, -1].as_slice(), true, Kind::Float);
    // A buffer allocated for the other axis would broadcast into a silently wrong
    // preconditioner instead of failing, so the agreement is checked, not assumed.
    debug_assert_eq!(
        second_momentum.size(),
        v_mean.size(),
        "second-moment buffer shape disagrees with the NorMuon reduction axis"
    );
    // Raw EMA from 0, no bias correction: v = v*beta2 + v_mean*(1-beta2).
    let _ = second_momentum.lerp_(&v_mean, 1.0 - beta2);
    // Per-slice step size = 1/(sqrt(v)+1e-10).
    let step_size = (second_momentum.sqrt() + 1e-10).reciprocal();
    // Analytic post-divide Frobenius^2 = sum_i step_size_i^2 * sq_sum_i.
    let vnorm_new_sq =
        (step_size.square() * &sq_sum).sum_dim_intlist([-2i64, -1].as_slice(), true, Kind::Float);
    // Frobenius-preservation ratio.
    let ratio = vnorm_sq.sqrt() / (vnorm_new_sq.sqrt() + 1e-10);
    // Fused per-slice scale, cast back to update kind.
    let scale = (&step_size * &ratio).to_kind(update.kind());
    update * &scale
}

fn normuon_transform(
    update: &Tensor,
    layout: OrthoLayout,
    second_momentum: &mut Tensor,
    beta2: f64,
    orth: Orthogonalizer,
    ns_steps: usize,
) -> (Tensor, f64) {
    match layout {
        OrthoLayout::Matrix => {
            let update = quintic_orthogonalize(update, orth, ns_steps);
            let update = normuon_rescale(&update, second_momentum, beta2);
            let size = update.size();
            let aspect_scale = (1.0_f64).max(size[0] as f64 / size[1] as f64).sqrt();
            (update, aspect_scale)
        }
        OrthoLayout::RowHeads { heads, head_dim } => {
            let cols = update.size()[1];
            let blocks = update.reshape([heads, head_dim, cols]);
            let blocks = batched_quintic_orthogonalize(&blocks, orth, ns_steps);
            let blocks = normuon_rescale(&blocks, second_momentum, beta2);
            let aspect_scale = (1.0_f64).max(head_dim as f64 / cols as f64).sqrt();
            (blocks.reshape(update.size().as_slice()), aspect_scale)
        }
        OrthoLayout::ColHeads { heads, head_dim } => {
            let rows = update.size()[0];
            let blocks = update
                .reshape([rows, heads, head_dim])
                .permute([1, 0, 2])
                .contiguous();
            let blocks = batched_quintic_orthogonalize(&blocks, orth, ns_steps);
            let blocks = normuon_rescale(&blocks, second_momentum, beta2);
            let aspect_scale = (1.0_f64).max(rows as f64 / head_dim as f64).sqrt();
            let update = blocks
                .permute([1, 0, 2])
                .contiguous()
                .reshape(update.size().as_slice());
            (update, aspect_scale)
        }
    }
}

impl RowLrState {
    fn new(rows: i64, cols: i64, device: Device) -> Self {
        Self {
            logit: Tensor::zeros([rows, 1], (Kind::Float, device)),
            adam_m: Tensor::zeros([rows, 1], (Kind::Float, device)),
            adam_v: Tensor::zeros([rows, 1], (Kind::Float, device)),
            previous_delta: Tensor::zeros([rows, cols], (Kind::Float, device)),
            adam_step: 0,
        }
    }

    /// Form this step's detached row multiplier from the OLD logits, then update
    /// those logits from current raw-gradient evidence for use on the NEXT primary
    /// step. Auxiliary calls pass `None` and use the frozen multiplier without
    /// touching any controller state.
    fn scale_gradient(
        &mut self,
        raw_gradient: &Tensor,
        primary_step: Option<i64>,
    ) -> (Tensor, Option<RowLrObservation>) {
        let sigmoid = self.logit.sigmoid();
        let log_alpha = &sigmoid * ROW_LR_LOG_SPAN - ROW_LR_LOG_SPAN / 2.0;
        let alpha = log_alpha.exp();
        let scaled = raw_gradient * alpha.to_kind(raw_gradient.kind());
        let Some(primary_step) = primary_step else {
            return (scaled, None);
        };

        let raw_f32 = raw_gradient
            .to_kind(Kind::Float)
            .nan_to_num(0.0, 0.0, 0.0);
        let evidence_raw = -(&raw_f32 * &self.previous_delta)
            .mean_dim([1i64].as_slice(), true, Kind::Float);
        let centered = &evidence_raw - evidence_raw.mean(Kind::Float);
        let std = centered.square().mean(Kind::Float).sqrt();
        let evidence = (&centered / std.clamp_min(1e-8))
            .clamp(-3.0, 3.0)
            .nan_to_num(0.0, 0.0, 0.0);
        let objective_per_row = -&evidence * &log_alpha;
        let mut update_magnitude = Tensor::zeros_like(&self.logit);

        if primary_step > ROW_LR_WARMUP_STEPS {
            let rows = self.logit.size()[0] as f64;
            let objective_gradient =
                -&evidence * ROW_LR_LOG_SPAN * &sigmoid * (1.0 - &sigmoid) / rows;
            let (beta1, beta2) = ROW_LR_CONTROLLER_BETAS;
            let _ = self.adam_m.lerp_(&objective_gradient, 1.0 - beta1);
            let _ = self
                .adam_v
                .lerp_(&objective_gradient.square(), 1.0 - beta2);
            self.adam_step += 1;
            let bc1 = 1.0 - beta1.powi(self.adam_step as i32);
            let bc2 = 1.0 - beta2.powi(self.adam_step as i32);
            let denom = self.adam_v.sqrt() / bc2.sqrt() + ROW_LR_CONTROLLER_EPS;
            let logit_delta =
                &self.adam_m / denom * (-ROW_LR_CONTROLLER_LR / bc1);
            update_magnitude = logit_delta.abs();
            let _ = self.logit.g_add_(&logit_delta);
        }

        (
            scaled,
            Some(RowLrObservation {
                alpha,
                evidence,
                objective_per_row,
                update_magnitude,
            }),
        )
    }
}

impl RowLrMetricAccumulator {
    fn push(&mut self, observation: RowLrObservation) {
        self.observations.push(observation);
    }

    fn finish(self) -> Option<Tensor> {
        if self.observations.is_empty() {
            return None;
        }
        let alpha = Tensor::cat(
            &self
                .observations
                .iter()
                .map(|item| item.alpha.reshape([-1]))
                .collect::<Vec<_>>(),
            0,
        );
        let evidence = Tensor::cat(
            &self
                .observations
                .iter()
                .map(|item| item.evidence.reshape([-1]))
                .collect::<Vec<_>>(),
            0,
        );
        let objective = Tensor::cat(
            &self
                .observations
                .iter()
                .map(|item| item.objective_per_row.reshape([-1]))
                .collect::<Vec<_>>(),
            0,
        );
        let update_magnitude = Tensor::cat(
            &self
                .observations
                .iter()
                .map(|item| item.update_magnitude.reshape([-1]))
                .collect::<Vec<_>>(),
            0,
        );
        let lower = (-ROW_LR_LOG_SPAN / 2.0).exp();
        let upper = (ROW_LR_LOG_SPAN / 2.0).exp();
        let at_bound = alpha
            .le(lower + 1e-6)
            .logical_or(&alpha.ge(upper - 1e-6))
            .to_kind(Kind::Float)
            .mean(Kind::Float);
        Some(Tensor::stack(
            &[
                alpha.mean(Kind::Float),
                alpha.std(false),
                alpha.min(),
                alpha.max(),
                at_bound,
                evidence.mean(Kind::Float),
                evidence.std(false),
                objective.mean(Kind::Float),
                update_magnitude.mean(Kind::Float),
            ],
            0,
        ))
    }
}

impl Muon {
    pub fn new(trainable_vars: &[Tensor], cfg: MuonConfig) -> Self {
        let named: Vec<(String, Tensor)> = trainable_vars
            .iter()
            .map(|t| (String::new(), t.shallow_clone()))
            .collect();
        Self::new_named(&named, cfg)
    }

    pub fn new_named(trainable_vars: &[(String, Tensor)], cfg: MuonConfig) -> Self {
        let params: Vec<Tensor> = trainable_vars
            .iter()
            .map(|(_, t)| t.shallow_clone())
            .collect();
        let names: Vec<String> = trainable_vars
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        let mut entries_2d = Vec::new();
        let mut adamw_indices = Vec::new();
        assert!(
            cfg.orthogonalizer != Orthogonalizer::PolarExpress5
                || cfg.ns_steps == POLAR_EXPRESS_COEFFS.len(),
            "Polar Express runs its own tuned {}-step schedule; ns_steps={} would be ignored",
            POLAR_EXPRESS_COEFFS.len(),
            cfg.ns_steps
        );

        for (i, (name, p)) in trainable_vars.iter().enumerate() {
            let force_adamw = cfg
                .force_adamw_name_substrings
                .iter()
                .any(|needle| name.contains(needle));
            let allowed = cfg.muon_name_allowlist.is_empty()
                || cfg
                    .muon_name_allowlist
                    .iter()
                    .any(|needle| name.contains(needle));
            if cfg.use_muon_for_2d && p.dim() == 2 && !force_adamw && allowed {
                let size = p.size();
                let (m, n) = (size[0], size[1]);
                let kind = p.kind();
                let device = p.device();
                let layout = attention_ortho_layout(name, &size, &cfg);
                entries_2d.push(Entry2D {
                    idx: i,
                    layout,
                    momentum: Tensor::zeros([m, n], (kind, device)),
                    second_momentum: Tensor::zeros(
                        second_momentum_shape(&size, layout).as_slice(),
                        (Kind::Float, device),
                    ),
                    row_lr: cfg
                        .row_learned_lr
                        .then(|| RowLrState::new(m, n, device)),
                });
            } else {
                adamw_indices.push(i);
            }
        }

        if !cfg.quiet {
            if cfg.use_muon_for_2d {
                println!(
                    "NorMuon optimizer: {} 2D params (NS5 + per-row second moment), {} other params (AdamW)",
                    entries_2d.len(),
                    adamw_indices.len()
                );
                let head_ortho = entries_2d
                    .iter()
                    .filter(|entry| entry.layout != OrthoLayout::Matrix)
                    .count();
                if head_ortho > 0 {
                    println!(
                        "  attention-head ortho: {} params split into batched NS blocks (self_head_dim={}, cross_head_dim={})",
                        head_ortho, cfg.attention_head_dim, cfg.cross_attention_head_dim
                    );
                }
            } else {
                println!(
                    "AdamW optimizer: {} params (Muon disabled for root-cause logging)",
                    adamw_indices.len()
                );
            }
        }

        Self {
            cfg,
            entries_2d,
            adamw_indices,
            adamw_state: HashMap::new(),
            step_count: 0,
            params,
            lr_scales: vec![1.0; names.len()],
            step_enabled: vec![true; names.len()],
            adamw_pending_grads: false,
            row_lr_metrics: None,
            names,
        }
    }

    /// One optimizer step of `kind`. NorMuon parameters update every step; AdamW
    /// parameters update once every [`MuonConfig::adamw_every`] PRIMARY steps, over the
    /// gradient ACCUMULATED across the intervening ones.
    ///
    /// At the default `1` this is an ordinary every-step update. At `2` it reproduces
    /// modded-nanogpt's `do_adam` cadence (`train_gpt.py:2104-2106`,
    /// `_is_adam_step(step) = step % 2 == 1`). That is NOT a halved update count with the
    /// intervening gradient discarded: the reference skips `param.grad = None` for Adam
    /// params on a non-Adam step (`train_gpt.py:821-823`), so the next backward sums into
    /// the retained gradient and the update sees an effective 2x batch. Adam is invariant
    /// to the gradient's overall scale, so summing rather than averaging leaves the step
    /// size alone and buys pure variance reduction, plus half the optimizer work on the
    /// embedding tables and output heads.
    ///
    /// The moments then advance once per N steps, so `beta1`/`beta2` span N times the
    /// wall-clock they otherwise would and decoupled weight decay applies N times less
    /// often. Both are properties of the recipe that chose N, not accidents of this port.
    ///
    /// The reference keys that cadence off the global TRAINING step rather than off a
    /// count of optimizer calls, and [`StepKind`] is what keeps the two the same thing
    /// here.
    pub fn step(&mut self, kind: StepKind) {
        tch::no_grad(|| {
            let primary = kind == StepKind::Primary;
            let do_adamw = if primary {
                self.step_count += 1;
                self.step_count % (self.cfg.adamw_every.max(1) as i64) == 0
            } else {
                false
            };
            self.step_all_normuon(primary);
            if do_adamw {
                self.step_all_adamw();
            }
            self.adamw_pending_grads = !do_adamw;
        });
    }

    fn step_all_normuon(&mut self, primary: bool) {
        let beta1 = self.cfg.momentum;
        let beta2 = self.cfg.beta2;
        let nesterov = self.cfg.nesterov;
        let wd = self.cfg.weight_decay;
        let base_lr = self.cfg.lr;
        let orth = self.cfg.orthogonalizer;
        let quadratic = self.cfg.quadratic_lr_weight_decay;
        let cautious = self.cfg.cautious_weight_decay;
        let mut row_lr_metrics = RowLrMetricAccumulator::default();

        for entry in &mut self.entries_2d {
            if !self.step_enabled[entry.idx] {
                continue;
            }
            let grad = self.params[entry.idx].grad();
            if !grad.defined() {
                continue;
            }
            let lr = base_lr * self.lr_scales[entry.idx];
            let (controlled_gradient, observation) = if primary {
                match entry.row_lr.as_mut() {
                    Some(controller) => {
                        let (gradient, observation) =
                            controller.scale_gradient(&grad, Some(self.step_count));
                        (Some(gradient), observation)
                    }
                    None => (None, None),
                }
            } else {
                // Auxiliary updates retain the exact static NorMuon path. In particular they
                // neither consume current logits nor alter the primary credit tensor.
                (None, None)
            };
            if let Some(observation) = observation {
                row_lr_metrics.push(observation);
            }
            let gradient = controlled_gradient.as_ref().unwrap_or(&grad);

            // The learned row scale acts on the raw gradient, before either optimizer
            // momentum or orthogonalization can mix its evidence across time or rows.
            let _ = entry.momentum.lerp_(gradient, 1.0 - beta1);

            let update = if nesterov {
                gradient.lerp(&entry.momentum, beta1)
            } else {
                entry.momentum.shallow_clone()
            };

            let (update, aspect_scale) = normuon_transform(
                &update,
                entry.layout,
                &mut entry.second_momentum,
                beta2,
                orth,
                self.cfg.ns_steps,
            );

            // Apply to param: decoupled weight decay, then the update. Both read
            // the pre-step parameter, so the composition is `p - decay*p - lr*u`.
            let mut p = self.params[entry.idx].shallow_clone();
            let update = update.to_kind(p.kind());
            let eff_lr = lr * aspect_scale;
            let decay = if quadratic {
                wd * base_lr * eff_lr
            } else {
                wd * lr
            };
            if decay > 0.0 {
                if cautious {
                    let keep = (&update * &p).ge(0).to_kind(p.kind()) * decay;
                    let _ = p.g_sub_(&(&p * keep));
                } else {
                    let _ = p.g_mul_scalar_(1.0 - decay);
                }
            }
            if primary {
                if let Some(controller) = entry.row_lr.as_mut() {
                    // This is theta_new - theta_old from the gradient update only. Decoupled
                    // decay above is intentionally absent from the next-step credit tensor.
                    let signed_delta = &update * (-eff_lr);
                    controller.previous_delta.copy_(
                        &signed_delta
                            .to_kind(Kind::Float)
                            .nan_to_num(0.0, 0.0, 0.0),
                    );
                    let _ = p.g_add_(&signed_delta);
                    continue;
                }
            }
            // Exact controller-off and auxiliary arithmetic.
            let _ = p.g_add_(&(update * (-eff_lr)));
        }
        if primary {
            self.row_lr_metrics = row_lr_metrics.finish();
        }
    }

    fn step_all_adamw(&mut self) {
        let eps = self.cfg.adamw_eps;
        let wd = self.cfg.adamw_wd;
        let quadratic = self.cfg.quadratic_lr_weight_decay;
        let cautious = self.cfg.cautious_weight_decay;

        for &idx in &self.adamw_indices {
            if !self.step_enabled[idx] {
                continue;
            }
            let mut p = self.params[idx].shallow_clone();
            let grad = p.grad();
            if !grad.defined() {
                continue;
            }
            let lr = self.cfg.adamw_lr * self.lr_scales[idx];
            let (beta1, beta2) = self.adamw_betas_for(idx);

            let state = self
                .adamw_state
                .entry(idx)
                .or_insert_with(|| AdamWParamState {
                    m: Tensor::zeros_like(&grad),
                    v: Tensor::zeros_like(&grad),
                    step_count: 0,
                });
            state.step_count += 1;
            let bc1 = 1.0 - beta1.powi(state.step_count as i32);
            let bc2 = 1.0 - beta2.powi(state.step_count as i32);
            let step_size = -lr / bc1;
            let inv_bc2_sqrt = 1.0 / bc2.sqrt();

            let apply_weight_decay = wd > 0.0
                && !self
                    .cfg
                    .adamw_no_weight_decay_name_substrings
                    .iter()
                    .any(|needle| self.names[idx].contains(needle));

            let _ = state.m.lerp_(&grad, 1.0 - beta1);
            let _ = state.v.lerp_(&grad.square(), 1.0 - beta2);

            let denom = state.v.sqrt() * inv_bc2_sqrt + eps;
            // `step` carries the negated descent direction, i.e. `p += step`.
            let step = &state.m / &denom * step_size;
            if apply_weight_decay {
                let wd_mul = self.adamw_wd_mul_for(idx);
                let decay = if quadratic {
                    lr * lr * wd * wd_mul
                } else {
                    lr * wd * wd_mul
                };
                if cautious {
                    // The reference masks strictly on `(descent_update * p) > 0`;
                    // `step` is the negation of that update, hence `< 0`.
                    let keep = (&step * &p).lt(0).to_kind(p.kind()) * decay;
                    let _ = p.g_sub_(&(&p * keep));
                } else {
                    let _ = p.g_mul_scalar_(1.0 - decay);
                }
            }
            let _ = p.g_add_(&step);
        }
    }

    /// AdamW weight-decay multiplier for one parameter: the first matching
    /// `adamw_weight_decay_multipliers` fragment wins, otherwise `1.0`.
    fn adamw_wd_mul_for(&self, idx: usize) -> f64 {
        let name = &self.names[idx];
        self.cfg
            .adamw_weight_decay_multipliers
            .iter()
            .find(|(needle, _)| name.contains(needle.as_str()))
            .map_or(1.0, |(_, mul)| *mul)
    }

    /// AdamW betas for one parameter: the first matching `adamw_beta_overrides`
    /// fragment wins, otherwise the shared `adamw_betas`.
    fn adamw_betas_for(&self, idx: usize) -> (f64, f64) {
        let name = &self.names[idx];
        self.cfg
            .adamw_beta_overrides
            .iter()
            .find(|(needle, _)| name.contains(needle.as_str()))
            .map_or(self.cfg.adamw_betas, |(_, betas)| *betas)
    }

    /// Zero the gradients the next backward accumulates into.
    ///
    /// NorMuon parameters are zeroed every step. AdamW parameters are zeroed ONLY when the
    /// previous [`Self::step`] actually applied their update; on a skipped step their
    /// gradient is left in place so the next backward sums into it. That retention is what
    /// makes the every-other-step cadence an effective 2x batch instead of throwing half
    /// the gradient away.
    pub fn zero_grad(&self) {
        let clear = |idx: usize| {
            let mut g = self.params[idx].grad();
            if g.defined() {
                let _ = g.zero_();
            }
        };
        // Every parameter is routed to exactly one of the two groups, so this covers all of
        // `self.params` — `assert_routing_partitions` in the pretrainer enforces that.
        for entry in &self.entries_2d {
            clear(entry.idx);
        }
        if !self.adamw_pending_grads {
            for &idx in &self.adamw_indices {
                clear(idx);
            }
        }
    }

    pub fn lr(&self) -> f64 {
        self.cfg.lr
    }
    /// Device-resident packed diagnostics from the latest primary step, in
    /// [`RowLearnedLrMetrics`] field order. `None` when disabled or when no routed
    /// matrix had a gradient.
    pub fn row_learned_lr_metrics_tensor(&self) -> Option<Tensor> {
        self.row_lr_metrics.as_ref().map(Tensor::shallow_clone)
    }

    /// Host view for tests and non-packed callers. Pretraining uses the tensor
    /// accessor above to preserve its one device-to-host metrics transfer.
    pub fn row_learned_lr_metrics(&self) -> Option<RowLearnedLrMetrics> {
        let packed = self.row_lr_metrics.as_ref()?;
        let values: [f64; ROW_LR_METRIC_COUNT] =
            Vec::<f64>::try_from(packed.to_kind(Kind::Double).reshape([-1]))
                .expect("controller metrics are convertible")
                .try_into()
                .expect("controller metric schema is fixed");
        Some(RowLearnedLrMetrics {
            alpha_mean: values[0],
            alpha_std: values[1],
            alpha_min: values[2],
            alpha_max: values[3],
            alpha_bound_fraction: values[4],
            evidence_mean: values[5],
            evidence_std: values[6],
            objective: values[7],
            update_magnitude: values[8],
        })
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.cfg.lr = lr;
    }

    pub fn set_momentum(&mut self, momentum: f64) {
        self.cfg.momentum = momentum;
    }

    pub fn set_adamw_lr(&mut self, lr: f64) {
        self.cfg.adamw_lr = lr;
    }

    /// Set a learning-rate multiplier for every named parameter matching at
    /// least one substring. Returns the number of matched parameters so callers
    /// can reject stale routing names instead of silently disabling a schedule.
    pub fn set_named_lr_scale(&mut self, name_substrings: &[&str], lr_scale: f64) -> usize {
        assert!(lr_scale.is_finite() && lr_scale > 0.0);
        let mut matched = 0;
        for (index, name) in self.names.iter().enumerate() {
            if name_substrings
                .iter()
                .any(|substring| name.contains(substring))
            {
                self.lr_scales[index] = lr_scale;
                matched += 1;
            }
        }
        matched
    }

    /// Names of every parameter routed to the NorMuon (2D) branch, in registration
    /// order. Lets callers assert their intended routing instead of trusting
    /// substring lists to have matched.
    pub fn muon_param_names(&self) -> Vec<String> {
        self.entries_2d
            .iter()
            .map(|entry| self.names[entry.idx].clone())
            .collect()
    }

    /// Names of every parameter routed to the AdamW branch, in registration order.
    pub fn adamw_param_names(&self) -> Vec<String> {
        self.adamw_indices
            .iter()
            .map(|&idx| self.names[idx].clone())
            .collect()
    }
    /// Resolved AdamW group settings for one exactly named parameter:
    /// `(learning-rate scale, betas, weight-decay multiplier)`.
    ///
    /// The multiplier is zero when the no-decay routing wins. This exposes the resolved
    /// first-match policy so a model-specific optimizer builder can assert named groups rather
    /// than merely asserting that a collection of substring overrides was populated.
    pub(crate) fn adamw_group_settings(
        &self,
        parameter_name: &str,
    ) -> Option<(f64, (f64, f64), f64)> {
        let idx = self.names.iter().position(|name| name == parameter_name)?;
        if !self.adamw_indices.contains(&idx) {
            return None;
        }
        let no_weight_decay = self
            .cfg
            .adamw_no_weight_decay_name_substrings
            .iter()
            .any(|needle| parameter_name.contains(needle));
        let wd_multiplier = if no_weight_decay {
            0.0
        } else {
            self.adamw_wd_mul_for(idx)
        };
        Some((
            self.lr_scales[idx],
            self.adamw_betas_for(idx),
            wd_multiplier,
        ))
    }

    /// Enable or disable optimizer steps for matching named parameters. This is
    /// stronger than zeroing gradients: disabled parameters also skip momentum,
    /// second-moment, and weight-decay updates.
    pub fn set_named_step_enabled(&mut self, name_substrings: &[&str], enabled: bool) -> usize {
        let mut matched = 0;
        for (index, name) in self.names.iter().enumerate() {
            if name_substrings
                .iter()
                .any(|substring| name.contains(substring))
            {
                self.step_enabled[index] = enabled;
                matched += 1;
            }
        }
        matched
    }

    /// Serialize all optimizer state (per-param momentum/second-momentum for the
    /// NorMuon 2D params, AdamW m/v for the rest, and the global step counter) to
    /// a named-tensor sidecar. Keying by variable name makes restoration robust to
    /// param ordering. Requires `new_named` (the unnamed path stores empty names
    /// and its keys would collide).
    pub fn save_state(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let mut named: Vec<(String, Tensor)> = Vec::new();
        // Suffix separator is `.` because tch's save/load round-trips `.`<->`|`
        // internally; any other separator would not survive the round trip.
        for entry in &self.entries_2d {
            let name = &self.names[entry.idx];
            named.push((
                format!("{name}.__momentum"),
                entry.momentum.to_device(Device::Cpu),
            ));
            named.push((
                format!("{name}.__second_momentum"),
                entry.second_momentum.to_device(Device::Cpu),
            ));
            if let Some(controller) = &entry.row_lr {
                named.push((
                    format!("{name}.__row_lr_logit"),
                    controller.logit.to_device(Device::Cpu),
                ));
                named.push((
                    format!("{name}.__row_lr_adam_m"),
                    controller.adam_m.to_device(Device::Cpu),
                ));
                named.push((
                    format!("{name}.__row_lr_adam_v"),
                    controller.adam_v.to_device(Device::Cpu),
                ));
                named.push((
                    format!("{name}.__row_lr_previous_delta"),
                    controller.previous_delta.to_device(Device::Cpu),
                ));
                named.push((
                    format!("{name}.__row_lr_adam_step"),
                    Tensor::from(controller.adam_step),
                ));
            }
        }
        for (&idx, state) in &self.adamw_state {
            let name = &self.names[idx];
            named.push((format!("{name}.__adamw_m"), state.m.to_device(Device::Cpu)));
            named.push((format!("{name}.__adamw_v"), state.v.to_device(Device::Cpu)));
            named.push((
                format!("{name}.__adamw_step_count"),
                Tensor::from(state.step_count),
            ));
        }
        named.push((
            "__muon_step_count__".to_owned(),
            Tensor::from(self.step_count),
        ));
        let refs: Vec<(&str, &Tensor)> = named.iter().map(|(n, t)| (n.as_str(), t)).collect();
        Tensor::save_multi(&refs, path)
            .with_context(|| format!("failed saving optimizer state {}", path.display()))
    }

    /// Restore optimizer state saved by [`Muon::save_state`], copying buffers in
    /// place so device/dtype match the live params. Absent 2D buffers are an
    /// error (the checkpoint is incomplete); absent AdamW buffers leave that param
    /// lazily re-initialized on its next step.
    pub fn load_state(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let device = self
            .params
            .first()
            .map(|p| p.device())
            .unwrap_or(Device::Cpu);
        let loaded: HashMap<String, Tensor> = Tensor::load_multi_with_device(path, device)
            .with_context(|| format!("failed loading optimizer state {}", path.display()))?
            .into_iter()
            .collect();
        let global_step_count = loaded
            .get("__muon_step_count__")
            .map(|t| t.int64_value(&[]))
            .unwrap_or(0);
        tch::no_grad(|| -> Result<()> {
            for entry in &mut self.entries_2d {
                let name = &self.names[entry.idx];
                let momentum = loaded
                    .get(&format!("{name}.__momentum"))
                    .with_context(|| format!("optimizer state missing momentum for {name}"))?;
                let second = loaded
                    .get(&format!("{name}.__second_momentum"))
                    .with_context(|| {
                        format!("optimizer state missing second_momentum for {name}")
                    })?;
                entry.momentum.copy_(momentum);
                entry.second_momentum.copy_(second);
                if let Some(controller) = entry.row_lr.as_mut() {
                    controller.logit.copy_(
                        loaded
                            .get(&format!("{name}.__row_lr_logit"))
                            .with_context(|| format!("optimizer state missing row LR logit for {name}"))?,
                    );
                    controller.adam_m.copy_(
                        loaded
                            .get(&format!("{name}.__row_lr_adam_m"))
                            .with_context(|| format!("optimizer state missing row LR m for {name}"))?,
                    );
                    controller.adam_v.copy_(
                        loaded
                            .get(&format!("{name}.__row_lr_adam_v"))
                            .with_context(|| format!("optimizer state missing row LR v for {name}"))?,
                    );
                    controller.previous_delta.copy_(
                        loaded
                            .get(&format!("{name}.__row_lr_previous_delta"))
                            .with_context(|| {
                                format!("optimizer state missing row LR previous delta for {name}")
                            })?,
                    );
                    controller.adam_step = loaded
                        .get(&format!("{name}.__row_lr_adam_step"))
                        .with_context(|| format!("optimizer state missing row LR step for {name}"))?
                        .int64_value(&[]);
                }
            }
            self.adamw_state.clear();
            for &idx in &self.adamw_indices {
                let name = &self.names[idx];
                if let (Some(m), Some(v)) = (
                    loaded.get(&format!("{name}.__adamw_m")),
                    loaded.get(&format!("{name}.__adamw_v")),
                ) {
                    self.adamw_state.insert(
                        idx,
                        AdamWParamState {
                            m: m.shallow_clone(),
                            v: v.shallow_clone(),
                            step_count: loaded
                                .get(&format!("{name}.__adamw_step_count"))
                                .map(|step| step.int64_value(&[]))
                                .unwrap_or(global_step_count),
                        },
                    );
                }
            }
            Ok(())
        })?;
        self.step_count = global_step_count;
        self.row_lr_metrics = None;
        Ok(())
    }

    /// Names of AdamW parameters whose lazy moments have been initialized.
    /// Persisting this set distinguishes a legitimate never-stepped parameter
    /// from a truncated resume sidecar.
    pub fn initialized_adamw_names(&self) -> Vec<String> {
        let mut names = self
            .adamw_state
            .keys()
            .map(|&idx| self.names[idx].clone())
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    /// Validate the complete optimizer tensor schema against a freshly
    /// constructed optimizer without mutating any live parameter or state.
    pub fn validate_state_strict(
        &self,
        path: impl AsRef<Path>,
        expected_initialized_adamw: &[String],
        expected_step: i64,
    ) -> Result<()> {
        let path = path.as_ref();
        let loaded: HashMap<String, Tensor> = Tensor::load_multi_with_device(path, Device::Cpu)
            .with_context(|| format!("failed loading optimizer state {}", path.display()))?
            .into_iter()
            .collect();
        let mut expected_keys = HashSet::new();
        expected_keys.insert("__muon_step_count__".to_owned());
        let global_step = loaded
            .get("__muon_step_count__")
            .context("optimizer state missing global step")?;
        ensure!(
            global_step.numel() == 1 && global_step.int64_value(&[]) == expected_step,
            "optimizer global step disagrees with checkpoint metadata"
        );

        for entry in &self.entries_2d {
            let name = &self.names[entry.idx];
            let momentum_name = format!("{name}.__momentum");
            let second_name = format!("{name}.__second_momentum");
            let momentum = loaded
                .get(&momentum_name)
                .with_context(|| format!("optimizer state missing momentum for {name}"))?;
            let second = loaded
                .get(&second_name)
                .with_context(|| format!("optimizer state missing second momentum for {name}"))?;
            ensure!(
                momentum.size() == entry.momentum.size()
                    && momentum.kind() == entry.momentum.kind(),
                "optimizer momentum schema mismatch for {name}"
            );
            ensure!(
                second.size() == entry.second_momentum.size()
                    && second.kind() == entry.second_momentum.kind(),
                "optimizer second-momentum schema mismatch for {name}"
            );
            expected_keys.insert(momentum_name);
            expected_keys.insert(second_name);
            if let Some(controller) = &entry.row_lr {
                for (suffix, expected) in [
                    ("__row_lr_logit", &controller.logit),
                    ("__row_lr_adam_m", &controller.adam_m),
                    ("__row_lr_adam_v", &controller.adam_v),
                    ("__row_lr_previous_delta", &controller.previous_delta),
                ] {
                    let key = format!("{name}.{suffix}");
                    let tensor = loaded
                        .get(&key)
                        .with_context(|| format!("optimizer state missing {suffix} for {name}"))?;
                    ensure!(
                        tensor.size() == expected.size() && tensor.kind() == expected.kind(),
                        "optimizer {suffix} schema mismatch for {name}"
                    );
                    expected_keys.insert(key);
                }
                let step_key = format!("{name}.__row_lr_adam_step");
                ensure!(
                    loaded
                        .get(&step_key)
                        .with_context(|| format!("optimizer state missing row LR step for {name}"))?
                        .numel()
                        == 1,
                    "optimizer row LR step is not scalar for {name}"
                );
                expected_keys.insert(step_key);
            }
        }

        let initialized = expected_initialized_adamw
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        ensure!(
            initialized.len() == expected_initialized_adamw.len(),
            "initialized AdamW names are not unique"
        );
        for name in &initialized {
            let idx = self
                .names
                .iter()
                .position(|candidate| candidate == name)
                .with_context(|| format!("optimizer checkpoint names unknown parameter {name}"))?;
            ensure!(
                self.adamw_indices.contains(&idx),
                "optimizer checkpoint routes non-AdamW parameter {name} through AdamW"
            );
            let m_name = format!("{name}.__adamw_m");
            let v_name = format!("{name}.__adamw_v");
            let step_name = format!("{name}.__adamw_step_count");
            let m = loaded
                .get(&m_name)
                .with_context(|| format!("optimizer state missing AdamW m for {name}"))?;
            let v = loaded
                .get(&v_name)
                .with_context(|| format!("optimizer state missing AdamW v for {name}"))?;
            let step = loaded
                .get(&step_name)
                .with_context(|| format!("optimizer state missing AdamW step for {name}"))?;
            ensure!(
                m.size() == self.params[idx].size() && v.size() == self.params[idx].size(),
                "optimizer AdamW moment shape mismatch for {name}"
            );
            ensure!(
                m.kind() == self.params[idx].kind() && v.kind() == self.params[idx].kind(),
                "optimizer AdamW moment dtype mismatch for {name}"
            );
            ensure!(
                step.numel() == 1,
                "optimizer AdamW step is not scalar for {name}"
            );
            expected_keys.extend([m_name, v_name, step_name]);
        }

        let actual_keys = loaded.keys().cloned().collect::<HashSet<_>>();
        ensure!(
            actual_keys == expected_keys,
            "optimizer tensor schema differs from the current model: missing={:?}, unexpected={:?}",
            expected_keys.difference(&actual_keys).collect::<Vec<_>>(),
            actual_keys.difference(&expected_keys).collect::<Vec<_>>()
        );
        ensure!(
            loaded.values().all(|tensor| {
                !tensor.is_floating_point() || tensor.isfinite().all().int64_value(&[]) != 0
            }),
            "optimizer state contains non-finite tensors"
        );
        Ok(())
    }

    pub fn load_state_strict(
        &mut self,
        path: impl AsRef<Path>,
        expected_initialized_adamw: &[String],
    ) -> Result<()> {
        self.load_state(path)?;
        let actual = self.initialized_adamw_names();
        anyhow::ensure!(
            actual == expected_initialized_adamw,
            "optimizer AdamW state is incomplete: expected {:?}, restored {:?}",
            expected_initialized_adamw,
            actual
        );
        Ok(())
    }

    /// Total bytes of optimizer state currently allocated.
    /// 2D params: NorMuon moments plus the optional row-controller tensors.
    /// 1D params: AdamW `m` + `v` per param (lazy — zero until first step).
    pub fn state_bytes(&self) -> usize {
        let tensor_bytes = |t: &Tensor| t.numel() * t.kind().elt_size_in_bytes();
        let muon: usize = self
            .entries_2d
            .iter()
            .map(|entry| {
                let base = tensor_bytes(&entry.momentum) + tensor_bytes(&entry.second_momentum);
                entry.row_lr.as_ref().map_or(base, |controller| {
                    base + tensor_bytes(&controller.logit)
                        + tensor_bytes(&controller.adam_m)
                        + tensor_bytes(&controller.adam_v)
                        + tensor_bytes(&controller.previous_delta)
                })
            })
            .sum();
        let adamw: usize = self
            .adamw_state
            .values()
            .map(|s| tensor_bytes(&s.m) + tensor_bytes(&s.v))
            .sum();
        muon + adamw
    }

    /// Bytes of optimizer state once EVERY branch has taken its first step.
    ///
    /// [`Self::state_bytes`] reports what is allocated now, and the AdamW moments are lazy,
    /// so before the first step it understates the steady state by two copies of every
    /// AdamW-routed parameter. Capacity planning has to reserve the steady state: a probe
    /// that measures a forward and a backward but never steps has not paid for these yet,
    /// and training will.
    pub fn steady_state_bytes(&self) -> usize {
        let tensor_bytes = |t: &Tensor| t.numel() * t.kind().elt_size_in_bytes();
        let muon: usize = self
            .entries_2d
            .iter()
            .map(|entry| {
                let base = tensor_bytes(&entry.momentum) + tensor_bytes(&entry.second_momentum);
                entry.row_lr.as_ref().map_or(base, |controller| {
                    base + tensor_bytes(&controller.logit)
                        + tensor_bytes(&controller.adam_m)
                        + tensor_bytes(&controller.adam_v)
                        + tensor_bytes(&controller.previous_delta)
                })
            })
            .sum();
        let adamw: usize = self
            .adamw_indices
            .iter()
            .map(|&idx| 2 * tensor_bytes(&self.params[idx]))
            .sum();
        muon + adamw
    }

    /// Test-only: shallow clone of the NorMuon second-moment buffer for the
    /// `n`-th 2D entry, so tests can assert it updates away from zero.
    #[cfg(test)]
    fn second_momentum_at(&self, n: usize) -> Tensor {
        self.entries_2d[n].second_momentum.shallow_clone()
    }
}

#[cfg(test)]
mod tests {
    use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

    use super::{
        attention_ortho_layout, batched_newtonschulz5, is_attention_output_projection_name,
        is_cross_attention_projection_name, is_self_attention_projection_name, newtonschulz5,
        normuon_reduce_dim, normuon_rescale, normuon_transform, orthogonalize_update,
        quintic_orthogonalize, second_momentum_shape, Muon, MuonConfig, OrthoLayout,
        Orthogonalizer, StepKind,
    };
    use crate::torch::test_rng;

    const HIDDEN: i64 = 128;
    const TRAIN_STEPS: usize = 500;
    const DATASET_SIZE: i64 = 2048;
    const BATCH_SIZE: i64 = 64;
    const INPUT_DIM: i64 = 16;

    fn build_mlp(vs: &nn::Path) -> impl Module {
        nn::seq()
            .add(nn::linear(
                vs / "fc1",
                INPUT_DIM,
                HIDDEN,
                Default::default(),
            ))
            .add_fn(|x| x.gelu("none"))
            .add(nn::linear(vs / "fc2", HIDDEN, HIDDEN, Default::default()))
            .add_fn(|x| x.gelu("none"))
            .add(nn::linear(vs / "fc3", HIDDEN, HIDDEN, Default::default()))
            .add_fn(|x| x.gelu("none"))
            .add(nn::linear(vs / "fc4", HIDDEN, 1, Default::default()))
    }

    /// Fixed dataset; training draws fresh minibatches via random indexing.
    fn make_dataset(device: Device) -> (Tensor, Tensor) {
        let _guard = tch::no_grad_guard();
        let w = Tensor::randn([INPUT_DIM, 4], (Kind::Float, device));
        let x = Tensor::randn([DATASET_SIZE, INPUT_DIM], (Kind::Float, device));
        let h = x.matmul(&w);
        let y = (h.slice(1, 0, 1, 1).sin() * h.slice(1, 1, 2, 1).cos()
            + 0.3 * h.slice(1, 2, 3, 1) * h.slice(1, 3, 4, 1).tanh())
            + 0.05 * Tensor::randn([DATASET_SIZE, 1], (Kind::Float, device));
        (x, y)
    }

    /// Eval loss over full dataset (no grad).
    fn eval_loss(net: &dyn Module, x: &Tensor, y: &Tensor) -> f64 {
        tch::no_grad(|| {
            let pred = net.forward(x);
            (&pred - y).square().mean(Kind::Float).double_value(&[])
        })
    }

    fn train_adamw(device: Device, seed: i64) -> Vec<f64> {
        tch::manual_seed(seed);
        let vs = nn::VarStore::new(device);
        let net = build_mlp(&vs.root());
        let mut opt = nn::AdamW::default().build(&vs, 1e-3).expect("adamw");

        tch::manual_seed(seed + 1000);
        let (x_all, y_all) = make_dataset(device);
        let mut losses = Vec::with_capacity(TRAIN_STEPS);

        for step in 0..TRAIN_STEPS {
            let idx = Tensor::randint(DATASET_SIZE, [BATCH_SIZE], (Kind::Int64, device));
            let xb = x_all.index_select(0, &idx);
            let yb = y_all.index_select(0, &idx);

            let pred = net.forward(&xb);
            let loss = (&pred - &yb).square().mean(Kind::Float);
            opt.backward_step(&loss);

            if step % 50 == 0 || step == TRAIN_STEPS - 1 {
                losses.push(eval_loss(&net, &x_all, &y_all));
            }
        }
        losses
    }

    fn train_muon(device: Device, seed: i64) -> Vec<f64> {
        tch::manual_seed(seed);
        let vs = nn::VarStore::new(device);
        let net = build_mlp(&vs.root());
        let trainable = vs.trainable_variables();
        let mut opt = Muon::new(
            &trainable,
            MuonConfig {
                lr: 5e-3,
                adamw_lr: 1e-3,
                ..MuonConfig::default()
            },
        );

        tch::manual_seed(seed + 1000);
        let (x_all, y_all) = make_dataset(device);
        let mut losses = Vec::with_capacity(TRAIN_STEPS);

        for step in 0..TRAIN_STEPS {
            let idx = Tensor::randint(DATASET_SIZE, [BATCH_SIZE], (Kind::Int64, device));
            let xb = x_all.index_select(0, &idx);
            let yb = y_all.index_select(0, &idx);

            let pred = net.forward(&xb);
            let loss = (&pred - &yb).square().mean(Kind::Float);
            loss.backward();
            opt.step(StepKind::Primary);
            opt.zero_grad();

            if step % 50 == 0 || step == TRAIN_STEPS - 1 {
                losses.push(eval_loss(&net, &x_all, &y_all));
            }
        }
        losses
    }

    /// Exercises the bf16 path of `batched_newtonschulz5`: the production
    /// trading bot runs after `vs.bfloat16()` so every ShapeGroup stores its
    /// momentum in bf16. This path has different code from fp32 (shallow vs
    /// kind-convert, in-place vs out-of-place ops) and can silently diverge
    /// if the NS5 implementation aliases caller storage.
    fn train_muon_bf16(device: Device, seed: i64) -> Vec<f64> {
        tch::manual_seed(seed);
        let mut vs = nn::VarStore::new(device);
        let net = build_mlp(&vs.root());
        vs.bfloat16();
        let trainable = vs.trainable_variables();
        let mut opt = Muon::new(
            &trainable,
            MuonConfig {
                lr: 5e-3,
                adamw_lr: 1e-3,
                ..MuonConfig::default()
            },
        );

        tch::manual_seed(seed + 1000);
        let (x_all_f32, y_all_f32) = make_dataset(device);
        let x_all = x_all_f32.to_kind(Kind::BFloat16);
        let y_all = y_all_f32.to_kind(Kind::BFloat16);
        let mut losses = Vec::with_capacity(TRAIN_STEPS);

        for step in 0..TRAIN_STEPS {
            let idx = Tensor::randint(DATASET_SIZE, [BATCH_SIZE], (Kind::Int64, device));
            let xb = x_all.index_select(0, &idx);
            let yb = y_all.index_select(0, &idx);

            let pred = net.forward(&xb);
            let loss = (&pred - &yb).square().mean(Kind::Float);
            loss.backward();
            opt.step(StepKind::Primary);
            opt.zero_grad();

            if step % 50 == 0 || step == TRAIN_STEPS - 1 {
                losses.push(eval_loss(&net, &x_all, &y_all));
            }
        }
        losses
    }

    #[test]
    fn muon_converges_bf16() {
        let _torch_rng_guard = test_rng::exclusive();
        let device = Device::Cpu;
        let losses = train_muon_bf16(device, 42);
        let first = losses[0];
        let last = *losses.last().unwrap();
        println!(
            "Muon bf16: loss {:.6} -> {:.6} ({:.1}x reduction)",
            first,
            last,
            first / last
        );
        assert!(
            last < first * 0.2,
            "Muon bf16 failed to converge: {:.6} -> {:.6}",
            first,
            last
        );
    }

    #[test]
    fn muon_converges_on_synthetic_regression() {
        let _torch_rng_guard = test_rng::exclusive();
        let device = Device::Cpu;
        let losses = train_muon(device, 42);
        let first = losses[0];
        let last = *losses.last().unwrap();
        println!(
            "Muon: loss {:.6} -> {:.6} ({:.1}x reduction)",
            first,
            last,
            first / last
        );
        assert!(
            last < first * 0.1,
            "Muon failed to converge: {:.6} -> {:.6}",
            first,
            last
        );
    }

    #[test]
    fn adamw_converges_on_synthetic_regression() {
        let _torch_rng_guard = test_rng::exclusive();
        let device = Device::Cpu;
        let losses = train_adamw(device, 42);
        let first = losses[0];
        let last = *losses.last().unwrap();
        println!(
            "AdamW: loss {:.6} -> {:.6} ({:.1}x reduction)",
            first,
            last,
            first / last
        );
        assert!(
            last < first * 0.1,
            "AdamW failed to converge: {:.6} -> {:.6}",
            first,
            last
        );
    }

    #[test]
    fn muon_vs_adamw_comparison() {
        let _torch_rng_guard = test_rng::exclusive();
        let device = Device::Cpu;
        let seed = 42;

        let adamw_losses = train_adamw(device, seed);
        let muon_losses = train_muon(device, seed);

        println!(
            "\n{:<8} {:>12} {:>12} {:>10}",
            "Step", "AdamW", "Muon", "Winner"
        );
        println!("{}", "-".repeat(46));
        let steps: Vec<usize> = (0..TRAIN_STEPS)
            .filter(|&s| s % 50 == 0 || s == TRAIN_STEPS - 1)
            .collect();
        for (i, &s) in steps.iter().enumerate() {
            let a = adamw_losses[i];
            let m = muon_losses[i];
            let winner = if m < a { "Muon" } else { "AdamW" };
            println!("{:<8} {:>12.6} {:>12.6} {:>10}", s + 1, a, m, winner);
        }

        let adamw_final = *adamw_losses.last().unwrap();
        let muon_final = *muon_losses.last().unwrap();
        println!(
            "\nFinal ratio (Muon/AdamW): {:.3}x  — {}",
            muon_final / adamw_final,
            if muon_final < adamw_final {
                "Muon wins"
            } else {
                "AdamW wins"
            }
        );

        // Both must converge
        assert!(
            adamw_final < 0.5,
            "AdamW did not converge: {:.6}",
            adamw_final
        );
        assert!(muon_final < 0.5, "Muon did not converge: {:.6}", muon_final);
    }

    /// Frobenius preservation has to hold in BOTH orientations, with the buffer allocated
    /// the way the optimizer allocates it. A wide matrix reduces over its ROWS, so a test
    /// that hard-codes `[rows, 1]` only ever exercises the tall case.
    #[test]
    fn normuon_rescale_preserves_frobenius_norm() {
        let _torch_rng_guard = test_rng::exclusive();
        let _g = tch::no_grad_guard();
        tch::manual_seed(7);
        let device = Device::Cpu;
        for size in [[37i64, 53], [53, 37]] {
            let update = Tensor::randn(size, (Kind::Float, device));
            let pre_norm = update.square().sum(Kind::Float).sqrt().double_value(&[]);

            let shape = second_momentum_shape(&size, OrthoLayout::Matrix);
            let mut second_momentum = Tensor::zeros(shape.as_slice(), (Kind::Float, device));
            let rescaled = normuon_rescale(&update, &mut second_momentum, 0.95);

            let post_norm = rescaled.square().sum(Kind::Float).sqrt().double_value(&[]);
            let rel = (post_norm - pre_norm).abs() / pre_norm;
            assert!(
                rel < 1e-4,
                "Frobenius norm not preserved for {size:?}: {pre_norm:.6} -> {post_norm:.6} (rel {rel:.2e})"
            );

            // Second moment must have moved off zero and stay finite.
            let v_min = second_momentum.min().double_value(&[]);
            let v_max = second_momentum.max().double_value(&[]);
            assert!(
                v_min > 0.0,
                "second_momentum stayed at zero for {size:?}: min={v_min}"
            );
            assert!(
                v_min.is_finite() && v_max.is_finite(),
                "second_momentum not finite for {size:?}: [{v_min}, {v_max}]"
            );
        }
    }

    /// The second moment lives on the LONGER axis and reduces over the shorter one, which
    /// is the reference's `red_dim = -1 if shape[-2] >= shape[-1] else -2`
    /// (modded-nanogpt `train_gpt.py:888`, buffer at `:566-569`). Reducing over the last
    /// axis unconditionally preconditions every wide matrix along its short axis; for the
    /// bar model that is `ff_out_w [512, 2048]` in all ten layers and `bar_dyn_fc3_w
    /// [512, 1664]`.
    #[test]
    fn normuon_second_moment_lives_on_the_longer_axis() {
        let _g = tch::no_grad_guard();
        let device = Device::Cpu;

        assert_eq!(normuon_reduce_dim(&[512, 2048]), -2);
        assert_eq!(normuon_reduce_dim(&[2048, 512]), -1);
        // Square is the reference's `>=` tie, which it breaks towards the last axis.
        assert_eq!(normuon_reduce_dim(&[512, 512]), -1);
        assert_eq!(
            second_momentum_shape(&[512, 2048], OrthoLayout::Matrix),
            vec![1, 2048]
        );
        assert_eq!(
            second_momentum_shape(&[2048, 512], OrthoLayout::Matrix),
            vec![2048, 1]
        );
        assert_eq!(
            second_momentum_shape(&[512, 512], OrthoLayout::Matrix),
            vec![512, 1]
        );

        // One EMA step from zero writes exactly `(1-beta2) * mean-of-squares` over the
        // SHORT axis, so a wide update yields one moment per column.
        let update = Tensor::arange(6, (Kind::Float, device)).reshape([2, 3]);
        let mut wide = Tensor::zeros([1, 3], (Kind::Float, device));
        let _ = normuon_rescale(&update, &mut wide, 0.9);
        let expected = update
            .square()
            .sum_dim_intlist([-2i64].as_slice(), true, Kind::Float)
            / 2.0
            * 0.1;
        assert!((&wide - &expected).abs().max().double_value(&[]) < 1e-9);

        // The statistic is a property of the matrix, not of how it is stored: the
        // transposed update must produce the transposed moments. This is what makes our
        // `ff_out_w [512, 2048]` preconditioner agree with the reference's `c_proj`, which
        // it stores transposed as `[mlp_hdim, dim]` (`train_gpt.py:1526`).
        let mut tall = Tensor::zeros([3, 1], (Kind::Float, device));
        let _ = normuon_rescale(&update.transpose(0, 1).contiguous(), &mut tall, 0.9);
        assert!(
            (&wide - &tall.transpose(0, 1))
                .abs()
                .max()
                .double_value(&[])
                < 1e-9,
            "wide and tall storage of the same matrix disagree on the second moment"
        );
    }

    /// Auxiliary steps must not be able to shift AdamW's cadence. When the parity keyed off
    /// a count of `step()` calls, one auxiliary-resolution update per primary step turned
    /// `adamw_every: 2` into an AdamW update on EVERY primary step and silently doubled the
    /// AdamW parameters' effective learning rate as soon as `--auxiliary-resolutions` was
    /// non-empty.
    #[test]
    fn auxiliary_steps_cannot_shift_the_adamw_cadence() {
        let device = Device::Cpu;
        let run = |aux_per_primary: usize| {
            let vs = nn::VarStore::new(device);
            let matrix = vs.root().ones("w", &[4, 4]);
            let bias = vs.root().ones("b", &[4]);
            let named = vec![
                ("w".to_owned(), matrix.shallow_clone()),
                ("b".to_owned(), bias.shallow_clone()),
            ];
            let mut opt = Muon::new_named(
                &named,
                MuonConfig {
                    adamw_every: 2,
                    quiet: true,
                    ..MuonConfig::default()
                },
            );
            let backward =
                || (matrix.square().sum(Kind::Float) + bias.square().sum(Kind::Float)).backward();
            let mut adamw_updates = Vec::new();
            for _ in 0..6 {
                backward();
                opt.step(StepKind::Primary);
                opt.zero_grad();
                for _ in 0..aux_per_primary {
                    backward();
                    opt.step(StepKind::Auxiliary);
                    opt.zero_grad();
                }
                adamw_updates.push(opt.adamw_state.get(&1).map_or(0, |state| state.step_count));
            }
            (adamw_updates, matrix.detach().copy())
        };

        let (no_aux, matrix_no_aux) = run(0);
        let (one_aux, matrix_one_aux) = run(1);
        let (three_aux, _) = run(3);

        assert_eq!(no_aux, vec![0, 1, 1, 2, 2, 3]);
        assert_eq!(
            one_aux, no_aux,
            "one auxiliary step per primary step shifted the AdamW cadence"
        );
        assert_eq!(
            three_aux, no_aux,
            "three auxiliary steps per primary step shifted the AdamW cadence"
        );
        // ...and the cadence is invariant because the auxiliary steps are excluded from the
        // clock, not because they were turned into no-ops: NorMuon still applies them.
        assert!(
            (matrix_one_aux - matrix_no_aux)
                .abs()
                .max()
                .double_value(&[])
                > 1e-6,
            "auxiliary steps did not reach the NorMuon parameters at all"
        );
    }

    #[test]
    fn batched_newtonschulz_matches_independent_single_matrix_path() {
        let _torch_rng_guard = test_rng::exclusive();
        let _g = tch::no_grad_guard();
        tch::manual_seed(17);
        let device = Device::Cpu;
        let heads = 4;
        let update = Tensor::randn([heads, 32, 96], (Kind::Float, device));
        let batched = batched_newtonschulz5(&update, 5);
        let independent: Vec<Tensor> = (0..heads)
            .map(|head| newtonschulz5(&update.get(head), 5))
            .collect();
        let independent = Tensor::stack(&independent, 0);
        let max_diff = (&batched - independent).abs().max().double_value(&[]);
        assert!(
            max_diff < 3e-2,
            "batched NS diverged from independent NS: max diff={max_diff:.3e}"
        );
    }

    #[test]
    fn attention_head_ortho_preserves_original_matrix_shape() {
        let _torch_rng_guard = test_rng::exclusive();
        let _g = tch::no_grad_guard();
        tch::manual_seed(23);
        let device = Device::Cpu;
        let row_split = Tensor::randn([256, 256], (Kind::Float, device));
        let col_split = Tensor::randn([256, 256], (Kind::Float, device));

        let row_out = orthogonalize_update(
            &row_split,
            OrthoLayout::RowHeads {
                heads: 4,
                head_dim: 64,
            },
            Orthogonalizer::NewtonSchulz5,
            5,
        );
        let col_out = orthogonalize_update(
            &col_split,
            OrthoLayout::ColHeads {
                heads: 4,
                head_dim: 64,
            },
            Orthogonalizer::NewtonSchulz5,
            5,
        );

        assert_eq!(row_out.size(), row_split.size());
        assert_eq!(col_out.size(), col_split.size());
        assert!(row_out.isfinite().all().int64_value(&[]) != 0);
        assert!(col_out.isfinite().all().int64_value(&[]) != 0);
    }

    #[test]
    fn cross_attention_requires_explicit_cross_head_dim() {
        let self_only = MuonConfig {
            per_attention_head_ortho: true,
            attention_head_dim: 64,
            ..MuonConfig::default()
        };
        assert_eq!(
            attention_ortho_layout("cross_attn.ca_q", &[256, 256], &self_only),
            OrthoLayout::Matrix
        );

        let with_cross = MuonConfig {
            cross_attention_head_dim: 128,
            ..self_only
        };
        assert_eq!(
            attention_ortho_layout("cross_attn.ca_q", &[256, 256], &with_cross),
            OrthoLayout::RowHeads {
                heads: 2,
                head_dim: 128
            }
        );
    }

    #[test]
    fn muon_name_allowlist_routes_only_matching_matrices() {
        let vars = vec![
            (
                "flow.fc1.weight".to_owned(),
                Tensor::zeros([8, 8], (Kind::Float, Device::Cpu)).set_requires_grad(true),
            ),
            (
                "flow.mod.weight".to_owned(),
                Tensor::zeros([8, 8], (Kind::Float, Device::Cpu)).set_requires_grad(true),
            ),
        ];
        let opt = Muon::new_named(
            &vars,
            MuonConfig {
                muon_name_allowlist: vec!["flow.fc".to_owned()],
                ..MuonConfig::default()
            },
        );
        assert_eq!(opt.entries_2d.len(), 1);
        assert_eq!(opt.adamw_indices.len(), 1);
        assert_eq!(opt.entries_2d[0].idx, 0);
        assert_eq!(opt.adamw_indices[0], 1);
    }

    #[test]
    fn named_lr_scale_changes_only_matching_parameter_updates() {
        let actor = Tensor::zeros([1], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let critic = Tensor::zeros([1], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let shared = Tensor::zeros([1], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let vars = vec![
            (
                "policy_concentration.bias".to_owned(),
                actor.shallow_clone(),
            ),
            ("value_projection.bias".to_owned(), critic.shallow_clone()),
            ("trunk_norm.weight".to_owned(), shared.shallow_clone()),
        ];
        let mut opt = Muon::new_named(
            &vars,
            MuonConfig {
                use_muon_for_2d: false,
                adamw_lr: 0.1,
                adamw_betas: (0.0, 0.0),
                adamw_eps: 0.0,
                quiet: true,
                ..MuonConfig::default()
            },
        );

        assert_eq!(opt.set_named_lr_scale(&["policy_concentration"], 0.25), 1);
        (&actor + &critic + &shared).sum(Kind::Float).backward();
        opt.step(StepKind::Primary);

        assert!((actor.double_value(&[]) + 0.025).abs() < 1e-7);
        assert!((critic.double_value(&[]) + 0.1).abs() < 1e-7);
        assert!((shared.double_value(&[]) + 0.1).abs() < 1e-7);
    }

    /// Capacity planning charges the optimizer's steady-state footprint against free VRAM
    /// before any step has run, so [`Muon::steady_state_bytes`] must EXCEED
    /// [`Muon::state_bytes`] by exactly the lazily-allocated AdamW moments and must equal it
    /// once every branch has stepped. The pretrainer charges only the difference: NorMuon's
    /// buffers are eager and already inside the device's baseline reading, so charging the
    /// whole steady state would count the 2D branch twice.
    #[test]
    fn the_steady_state_footprint_is_the_cold_one_plus_the_lazy_adamw_moments() {
        let matrix = Tensor::zeros([8, 4], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let bias = Tensor::zeros([8], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let vars = vec![
            ("layer.weight".to_owned(), matrix.shallow_clone()),
            ("layer.bias".to_owned(), bias.shallow_clone()),
        ];
        let mut opt = Muon::new_named(
            &vars,
            MuonConfig {
                quiet: true,
                ..MuonConfig::default()
            },
        );

        let element = std::mem::size_of::<f32>();
        // NorMuon: [8, 4] momentum plus an [8, 1] per-row second moment, both eager.
        let normuon = (32 + 8) * element;
        // AdamW: `m` and `v` over the [8] bias, both lazy.
        let adamw = 2 * 8 * element;
        assert_eq!(opt.state_bytes(), normuon);
        assert_eq!(opt.steady_state_bytes(), normuon + adamw);
        assert_eq!(
            opt.steady_state_bytes() - opt.state_bytes(),
            adamw,
            "the pending part is exactly what a forward-and-backward probe has not paid for"
        );

        (matrix.sum(Kind::Float) + bias.sum(Kind::Float)).backward();
        opt.step(StepKind::Primary);
        assert_eq!(
            opt.state_bytes(),
            opt.steady_state_bytes(),
            "after one step every branch is resident and the two figures must agree"
        );
    }

    #[test]
    fn disabled_named_parameters_skip_state_and_retain_their_adamw_clock() {
        let actor = Tensor::zeros([1], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let critic = Tensor::zeros([1], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let vars = vec![
            (
                "policy_concentration.bias".to_owned(),
                actor.shallow_clone(),
            ),
            ("value_projection.bias".to_owned(), critic.shallow_clone()),
        ];
        let mut opt = Muon::new_named(
            &vars,
            MuonConfig {
                use_muon_for_2d: false,
                adamw_lr: 0.1,
                quiet: true,
                ..MuonConfig::default()
            },
        );

        (&actor + &critic).sum(Kind::Float).backward();
        opt.step(StepKind::Primary);
        opt.zero_grad();
        let actor_before_critic_only_step = actor.double_value(&[]);

        assert_eq!(
            opt.set_named_step_enabled(&["policy_concentration"], false),
            1
        );
        critic.sum(Kind::Float).backward();
        opt.step(StepKind::Primary);

        assert_eq!(actor.double_value(&[]), actor_before_critic_only_step);
        assert!(critic.double_value(&[]) < actor_before_critic_only_step);
        assert_eq!(opt.adamw_state[&0].step_count, 1);
        assert_eq!(opt.adamw_state[&1].step_count, 2);

        let state_path = std::env::temp_dir().join(format!(
            "muon-per-param-clock-{}-{}.ot",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        opt.save_state(&state_path).unwrap();
        let restored_actor = Tensor::zeros([1], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let restored_critic =
            Tensor::zeros([1], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let mut restored = Muon::new_named(
            &[
                (
                    "policy_concentration.bias".to_owned(),
                    restored_actor.shallow_clone(),
                ),
                (
                    "value_projection.bias".to_owned(),
                    restored_critic.shallow_clone(),
                ),
            ],
            MuonConfig {
                use_muon_for_2d: false,
                quiet: true,
                ..MuonConfig::default()
            },
        );
        restored.load_state(&state_path).unwrap();
        assert_eq!(restored.adamw_state[&0].step_count, 1);
        assert_eq!(restored.adamw_state[&1].step_count, 2);
        std::fs::remove_file(state_path).unwrap();

        opt.zero_grad();
        opt.set_named_step_enabled(&["policy_concentration"], true);
        actor.sum(Kind::Float).backward();
        opt.step(StepKind::Primary);
        assert_eq!(opt.adamw_state[&0].step_count, 2);
        assert!((actor.double_value(&[]) + 0.2).abs() < 1e-6);
    }

    #[test]
    fn adamw_named_no_weight_decay_excludes_only_matching_parameters() {
        let phase = Tensor::ones([4], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let regular = Tensor::ones([4], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let vars = vec![
            ("layer.pope_theta_bias".to_owned(), phase.shallow_clone()),
            ("layer.bias".to_owned(), regular.shallow_clone()),
        ];
        let mut opt = Muon::new_named(
            &vars,
            MuonConfig {
                use_muon_for_2d: false,
                adamw_lr: 0.1,
                adamw_wd: 0.5,
                adamw_no_weight_decay_name_substrings: vec!["pope_theta_bias".to_owned()],
                ..MuonConfig::default()
            },
        );
        (&phase.sum(Kind::Float) * 0.0 + &regular.sum(Kind::Float) * 0.0).backward();
        opt.step(StepKind::Primary);
        assert_eq!(phase.min().double_value(&[]), 1.0);
        assert!((regular.max().double_value(&[]) - 0.95).abs() < 1e-6);
    }

    #[test]
    fn fused_qkv_groups_qkv_rows_by_attention_head() {
        let cfg = MuonConfig {
            per_attention_head_ortho: true,
            attention_head_dim: 64,
            ..MuonConfig::default()
        };
        assert_eq!(
            attention_ortho_layout("block0.attn_qkv", &[768, 256], &cfg),
            OrthoLayout::RowHeads {
                heads: 4,
                head_dim: 192
            }
        );
    }

    /// The head-ortho matchers must key off a tensor's OWN name, not off any substring of
    /// its path. `attn_out_w` contains `attn_o`, so under substring matching the bar model's
    /// [512, 512] output projection satisfied both the self-attention and the output-
    /// projection matcher, and enabling `per_attention_head_ortho` would have silently
    /// split it into column head-blocks. `benchmarks/src/optim_head_ortho.rs` already flips
    /// that flag, so this was a live trap and not a hypothetical one.
    #[test]
    fn head_ortho_matchers_key_off_exact_parameter_names() {
        let cfg = MuonConfig {
            per_attention_head_ortho: true,
            attention_head_dim: 64,
            cross_attention_head_dim: 64,
            ..MuonConfig::default()
        };

        // The bar model's names must not be matched by any of them.
        for name in [
            "bar_layer_3.attn_out_w",
            "bar_layer_3.qkv_w",
            "bar_layer_3.ff_out_w",
        ] {
            assert!(!is_self_attention_projection_name(name), "{name}");
            assert!(!is_cross_attention_projection_name(name), "{name}");
            assert!(!is_attention_output_projection_name(name), "{name}");
            assert_eq!(
                attention_ortho_layout(name, &[512, 512], &cfg),
                OrthoLayout::Matrix,
                "{name}"
            );
        }

        // The benchmark's names, which are exactly the matcher's own spellings, must be.
        assert!(is_self_attention_projection_name("block0.attn_qkv"));
        assert!(is_self_attention_projection_name("block0.attn_o"));
        assert!(is_attention_output_projection_name("block0.attn_o"));
        assert!(is_cross_attention_projection_name("cross_attn.ca_out"));
        assert!(is_attention_output_projection_name("cross_attn.ca_out"));
        assert_eq!(
            attention_ortho_layout("block0.attn_o", &[256, 256], &cfg),
            OrthoLayout::ColHeads {
                heads: 4,
                head_dim: 64
            }
        );
    }

    #[test]
    fn normuon_step_updates_second_moment_and_stays_finite() {
        let _torch_rng_guard = test_rng::exclusive();
        let device = Device::Cpu;
        tch::manual_seed(11);
        let vs = nn::VarStore::new(device);
        let net = build_mlp(&vs.root());
        let trainable = vs.trainable_variables();
        let mut opt = Muon::new(
            &trainable,
            MuonConfig {
                lr: 5e-3,
                adamw_lr: 1e-3,
                ..MuonConfig::default()
            },
        );

        tch::manual_seed(99);
        let (x_all, y_all) = make_dataset(device);

        // Before any step, every second-moment buffer is exactly zero.
        let before = opt.second_momentum_at(0);
        assert_eq!(before.max().double_value(&[]), 0.0);

        for _ in 0..5 {
            let idx = Tensor::randint(DATASET_SIZE, [BATCH_SIZE], (Kind::Int64, device));
            let xb = x_all.index_select(0, &idx);
            let yb = y_all.index_select(0, &idx);
            let pred = net.forward(&xb);
            let loss = (&pred - &yb).square().mean(Kind::Float);
            loss.backward();
            opt.step(StepKind::Primary);
            opt.zero_grad();
        }

        let after = opt.second_momentum_at(0);
        let v_min = after.min().double_value(&[]);
        let v_max = after.max().double_value(&[]);
        println!(
            "second_momentum[0] after 5 steps: [{:.3e}, {:.3e}]",
            v_min, v_max
        );
        assert!(v_min > 0.0, "second_momentum did not update: min={}", v_min);
        assert!(
            v_min.is_finite() && v_max.is_finite(),
            "second_momentum not finite: [{}, {}]",
            v_min,
            v_max
        );
        // second_momentum is per-row of the [out, in] weight => [out, 1].
        assert_eq!(after.size(), vec![HIDDEN, 1]);
        assert_eq!(after.kind(), Kind::Float);
    }

    /// Two parameters, deterministic grads, and a full read-back of the values the
    /// legacy (pre-Polar-Express, pre-cautious) code path produced. Any accidental
    /// flip of a new default, or any reordering of decay vs. update, breaks this.
    #[test]
    fn default_config_reproduces_legacy_decoupled_weight_decay() {
        let _torch_rng_guard = test_rng::exclusive();
        let device = Device::Cpu;
        tch::manual_seed(4242);
        let vs = nn::VarStore::new(device);
        let w = vs.root().randn("w", &[8, 4], 0.0, 1.0);
        let b = vs.root().randn("b", &[4], 0.0, 1.0);

        let probe = Tensor::randn([4, 3], (Kind::Float, device));
        let loss = w.matmul(&probe).square().sum(Kind::Float) + b.square().sum(Kind::Float) * 3.0;
        loss.backward();

        let w0 = w.detach().copy();
        let b0 = b.detach().copy();
        let gw = w.grad().detach().copy();
        let gb = b.grad().detach().copy();

        let (lr, wd, adamw_lr, adamw_wd) = (7e-3, 0.9, 4e-3, 0.05);
        let cfg = MuonConfig {
            lr,
            weight_decay: wd,
            adamw_lr,
            adamw_wd,
            momentum: 0.95,
            beta2: 0.9,
            adamw_eps: 1e-10,
            quiet: true,
            ..MuonConfig::default()
        };
        assert_eq!(cfg.orthogonalizer, Orthogonalizer::NewtonSchulz5);
        assert!(!cfg.cautious_weight_decay);
        assert!(!cfg.quadratic_lr_weight_decay);
        assert!(cfg.adamw_beta_overrides.is_empty());

        let named = vec![
            ("w".to_owned(), w.shallow_clone()),
            ("b".to_owned(), b.shallow_clone()),
        ];
        let mut opt = Muon::new_named(&named, cfg);
        opt.step(StepKind::Primary);

        // Legacy NorMuon branch: EMA from zero, Nesterov lerp, NS5 + per-row second
        // moment, then `p*(1 - lr*wd) - lr*aspect*update`.
        let expected_w = tch::no_grad(|| {
            let momentum = &gw * (1.0 - 0.95);
            let update = gw.lerp(&momentum, 0.95);
            let mut second = Tensor::zeros([8, 1], (Kind::Float, device));
            let (update, aspect) = normuon_transform(
                &update,
                OrthoLayout::Matrix,
                &mut second,
                0.9,
                Orthogonalizer::NewtonSchulz5,
                5,
            );
            assert!((aspect - 2.0_f64.sqrt()).abs() < 1e-12);
            &w0 * (1.0 - lr * wd) + update * (-lr * aspect)
        });

        // Legacy AdamW branch on the 1-D parameter.
        let expected_b = tch::no_grad(|| {
            let (ab1, ab2) = (0.9, 0.95);
            let m = &gb * (1.0 - ab1);
            let v = gb.square() * (1.0 - ab2);
            let bc1 = 1.0 - ab1;
            let bc2 = 1.0 - ab2;
            let denom = v.sqrt() * (1.0 / bc2.sqrt()) + 1e-10;
            &b0 * (1.0 - adamw_lr * adamw_wd) + m / denom * (-adamw_lr / bc1)
        });

        let dw = (w.detach() - expected_w).abs().max().double_value(&[]);
        let db = (b.detach() - expected_b).abs().max().double_value(&[]);
        assert!(
            dw < 1e-7,
            "NorMuon default path drifted from legacy: {dw:.3e}"
        );
        assert!(
            db < 1e-9,
            "AdamW default path drifted from legacy: {db:.3e}"
        );
    }

    /// The cautious mask must leave sign-agreeing coordinates untouched and skip
    /// decay entirely on the rest, on both branches, with the reference's strictness.
    #[test]
    fn cautious_weight_decay_skips_sign_disagreeing_coordinates() {
        let device = Device::Cpu;
        let build = || {
            let vs = nn::VarStore::new(device);
            let p = vs.root().var_copy(
                "p",
                &Tensor::from_slice(&[1.0f32, -1.0, 1.0, -1.0]).to_device(device),
            );
            let target = Tensor::from_slice(&[3.0f32, 3.0, -3.0, -3.0]).to_device(device);
            // dL/dp = target, so the descent update has the sign of `target`.
            let loss = (&p * &target).sum(Kind::Float);
            loss.backward();
            (vs, p)
        };

        let cfg = |cautious: bool| MuonConfig {
            adamw_lr: 0.1,
            adamw_wd: 0.5,
            adamw_eps: 1e-10,
            cautious_weight_decay: cautious,
            quiet: true,
            ..MuonConfig::default()
        };

        let (_vs_plain, p_plain) = build();
        let (_vs_caut, p_caut) = build();
        let p0 = p_plain.detach().copy();
        Muon::new_named(&[("p".to_owned(), p_plain.shallow_clone())], cfg(false))
            .step(StepKind::Primary);
        Muon::new_named(&[("p".to_owned(), p_caut.shallow_clone())], cfg(true))
            .step(StepKind::Primary);

        // Reference mask: (descent_update * p) > 0, i.e. sign(target) == sign(p).
        // Coordinates 0 and 3 agree; 1 and 2 disagree and must skip decay.
        let decay = 0.1 * 0.5;
        let diff = (p_caut.detach() - p_plain.detach()).to_kind(Kind::Double);
        let diff: Vec<f64> = Vec::<f64>::try_from(diff).expect("diff");
        let p0: Vec<f64> = Vec::<f64>::try_from(p0.to_kind(Kind::Double)).expect("p0");
        for i in [0usize, 3] {
            assert!(
                diff[i].abs() < 1e-7,
                "coordinate {i} agrees in sign and must still decay: diff={:.3e}",
                diff[i]
            );
        }
        for i in [1usize, 2] {
            let expected = p0[i] * decay;
            assert!(
                (diff[i] - expected).abs() < 1e-7,
                "coordinate {i} disagrees in sign and must skip decay: diff={:.3e}, expected={expected:.3e}",
                diff[i]
            );
        }
    }

    /// The NorMuon cautious mask is non-strict `(update * p) >= 0` and governs decay on
    /// every 2-D weight in a pretrain run, so it needs its own read-back: the 1-D test
    /// above only exercises AdamW's strict `.lt(0)` branch, and flipping the NorMuon
    /// comparison would otherwise leave this module green.
    #[test]
    fn cautious_weight_decay_masks_the_normuon_branch_too() {
        let _torch_rng_guard = test_rng::exclusive();
        let device = Device::Cpu;
        tch::manual_seed(5150);
        let w_init = Tensor::randn([8, 4], (Kind::Float, device));
        let probe = Tensor::randn([4, 3], (Kind::Float, device));
        let build = || {
            let vs = nn::VarStore::new(device);
            let w = vs.root().var_copy("w", &w_init);
            let loss = w.matmul(&probe).square().sum(Kind::Float);
            loss.backward();
            (vs, w)
        };
        let (lr, wd) = (1e-3f64, 0.9f64);
        let cfg = |cautious: bool| MuonConfig {
            lr,
            weight_decay: wd,
            momentum: 0.95,
            beta2: 0.9,
            cautious_weight_decay: cautious,
            quiet: true,
            ..MuonConfig::default()
        };

        let (_a, wa) = build();
        let (_b, wb) = build();
        Muon::new_named(&[("w".to_owned(), wa.shallow_clone())], cfg(false))
            .step(StepKind::Primary);
        Muon::new_named(&[("w".to_owned(), wb.shallow_clone())], cfg(true)).step(StepKind::Primary);

        // Reproduce the update the optimizer computed, to recover its sign per coordinate.
        let (update, _) = tch::no_grad(|| {
            let grad = wa.grad().detach().copy();
            let momentum = &grad * (1.0 - 0.95);
            let combined = grad.lerp(&momentum, 0.95);
            let mut second = Tensor::zeros([8, 1], (Kind::Float, device));
            normuon_transform(
                &combined,
                OrthoLayout::Matrix,
                &mut second,
                0.9,
                Orthogonalizer::NewtonSchulz5,
                5,
            )
        });
        // Coordinates where the update and the parameter AGREE in sign keep their decay;
        // the rest skip it, so the cautious run sits further from zero by exactly p*lr*wd.
        let keeps = (&update * &w_init).ge(0).to_kind(Kind::Float);
        let skipped = keeps.neg() + 1.0;
        let expected = &w_init * &skipped * (lr * wd);
        let diff = wb.detach() - wa.detach();
        let error = (&diff - &expected).abs().max().double_value(&[]);
        assert!(
            skipped.sum(Kind::Float).double_value(&[]) > 0.0
                && keeps.sum(Kind::Float).double_value(&[]) > 0.0,
            "the fixture must contain both agreeing and disagreeing coordinates"
        );
        assert!(
            error < 1e-6,
            "NorMuon cautious mask deviates from the non-strict (update*p) >= 0 rule: {error:.3e}"
        );
    }

    /// Quadratic weight decay multiplies the decay by one more factor of the
    /// learning rate — including the per-matrix multiplier on the NorMuon branch,
    /// which is why a wide matrix decays harder than a square one.
    #[test]
    fn quadratic_weight_decay_adds_one_factor_of_the_learning_rate() {
        let _torch_rng_guard = test_rng::exclusive();
        let device = Device::Cpu;
        let (lr, wd) = (0.05f64, 0.8f64);
        // Same hazard as the beta-override test: draw once, `var_copy` in.
        tch::manual_seed(7);
        let w_init = Tensor::randn([8, 4], (Kind::Float, device));
        let s_init = Tensor::randn([6], (Kind::Float, device));
        let build = || {
            let vs = nn::VarStore::new(device);
            let w = vs.root().var_copy("w", &w_init);
            let s = vs.root().var_copy("s", &s_init);
            let probe = Tensor::ones([4, 2], (Kind::Float, device));
            let loss = w.matmul(&probe).square().sum(Kind::Float) + s.square().sum(Kind::Float);
            loss.backward();
            (vs, w, s)
        };
        let cfg = |quadratic: bool| MuonConfig {
            lr,
            weight_decay: wd,
            adamw_lr: lr,
            adamw_wd: wd,
            adamw_eps: 1e-10,
            quadratic_lr_weight_decay: quadratic,
            quiet: true,
            ..MuonConfig::default()
        };

        let (_vs_a, wa, sa) = build();
        let (_vs_b, wb, sb) = build();
        let w0 = wa.detach().copy();
        let s0 = sa.detach().copy();
        let named = |w: &Tensor, s: &Tensor| {
            vec![
                ("w".to_owned(), w.shallow_clone()),
                ("s".to_owned(), s.shallow_clone()),
            ]
        };
        Muon::new_named(&named(&wa, &sa), cfg(false)).step(StepKind::Primary);
        Muon::new_named(&named(&wb, &sb), cfg(true)).step(StepKind::Primary);

        // The optimizer step itself is identical, so the whole difference is decay.
        let aspect = 2.0_f64.sqrt();
        let w_ratio = ((wb.detach() - wa.detach()) / (&w0 * (lr * wd)))
            .mean(Kind::Double)
            .double_value(&[]);
        assert!(
            (w_ratio - (1.0 - lr * aspect)).abs() < 1e-4,
            "NorMuon quadratic decay ratio {w_ratio:.6} != 1 - lr*aspect = {:.6}",
            1.0 - lr * aspect
        );
        let s_ratio = ((sb.detach() - sa.detach()) / (&s0 * (lr * wd)))
            .mean(Kind::Double)
            .double_value(&[]);
        assert!(
            (s_ratio - (1.0 - lr)).abs() < 1e-4,
            "AdamW quadratic decay ratio {s_ratio:.6} != 1 - lr = {:.6}",
            1.0 - lr
        );
    }

    #[test]
    fn adamw_beta_overrides_apply_only_to_matching_parameters() {
        let _torch_rng_guard = test_rng::exclusive();
        let device = Device::Cpu;
        // Drawn ONCE, outside the closure: `manual_seed` seeds the process-global ATen
        // generator, so two seeded draws inside a closure are not reproducible while
        // sibling tests in this module concurrently reseed and draw from it.
        tch::manual_seed(19);
        let embed_init = Tensor::randn([5], (Kind::Float, device));
        let gate_init = Tensor::randn([5], (Kind::Float, device));
        let build = || {
            let vs = nn::VarStore::new(device);
            let embed = vs.root().var_copy("bar_bin_embed", &embed_init);
            let gate = vs.root().var_copy("attn_resid_lambda", &gate_init);
            let loss = embed.square().sum(Kind::Float) + gate.square().sum(Kind::Float);
            loss.backward();
            (vs, embed, gate)
        };
        let cfg = |overrides: Vec<(String, (f64, f64))>| MuonConfig {
            adamw_lr: 0.01,
            adamw_betas: (0.9, 0.99),
            adamw_eps: 1e-10,
            adamw_beta_overrides: overrides,
            quiet: true,
            ..MuonConfig::default()
        };

        // Adam's bias correction cancels the betas exactly on the first step, so
        // the override only becomes observable once the EMAs have history. Two
        // steps with a persistent optimizer each.
        let mut runs = Vec::new();
        for overrides in [Vec::new(), vec![("bar_bin_embed".to_owned(), (0.5, 0.95))]] {
            let (vs, embed, gate) = build();
            let named = vec![
                ("bar_bin_embed".to_owned(), embed.shallow_clone()),
                ("attn_resid_lambda".to_owned(), gate.shallow_clone()),
            ];
            let mut opt = Muon::new_named(&named, cfg(overrides));
            opt.step(StepKind::Primary);
            opt.zero_grad();
            let loss = embed.square().sum(Kind::Float) * 3.0 + gate.square().sum(Kind::Float) * 3.0;
            loss.backward();
            opt.step(StepKind::Primary);
            runs.push((vs, embed.detach().copy(), gate.detach().copy()));
        }

        let embed_delta = (&runs[1].1 - &runs[0].1).abs().max().double_value(&[]);
        let gate_delta = (&runs[1].2 - &runs[0].2).abs().max().double_value(&[]);
        assert!(
            embed_delta > 1e-6,
            "beta override did not change the matching parameter: {embed_delta:.3e}"
        );
        assert!(
            gate_delta == 0.0,
            "beta override leaked into a non-matching parameter: {gate_delta:.3e}"
        );
    }

    /// Polar Express must land the singular values of an ill-conditioned gradient
    /// closer to one than Newton-Schulz at identical cost. Condition number 1e3 is
    /// representative of a real transformer weight gradient.
    #[test]
    fn polar_express_orthogonalizes_better_than_newton_schulz() {
        let _torch_rng_guard = test_rng::exclusive();
        let _g = tch::no_grad_guard();
        let device = Device::Cpu;
        tch::manual_seed(31337);
        let (u, _, v) = Tensor::randn([64, 64], (Kind::Float, device)).svd(true, true);
        let decades = Tensor::arange(64, (Kind::Float, device)) / 63.0 * -3.0;
        let spectrum = (decades * std::f64::consts::LN_10).exp();
        let g = (&u * spectrum.unsqueeze(0)).matmul(&v.transpose(0, 1));

        let deviation = |t: &Tensor| {
            let (_, s, _) = t.svd(true, false);
            let err = (s - 1.0).abs();
            (
                err.max().double_value(&[]),
                err.mean(Kind::Double).double_value(&[]),
            )
        };
        let (ns_max, ns_mean) = deviation(&newtonschulz5(&g, 5));
        let (pe_max, pe_mean) =
            deviation(&quintic_orthogonalize(&g, Orthogonalizer::PolarExpress5, 5));
        println!(
            "singular-value deviation from 1: NS5 max={ns_max:.4} mean={ns_mean:.4}, \
             PolarExpress5 max={pe_max:.4} mean={pe_mean:.4}"
        );
        assert!(
            pe_mean < ns_mean,
            "Polar Express mean deviation {pe_mean:.4} did not beat Newton-Schulz {ns_mean:.4}"
        );
        assert!(
            pe_max < ns_max,
            "Polar Express max deviation {pe_max:.4} did not beat Newton-Schulz {ns_max:.4}"
        );
        // Neither iteration may overshoot the unit ball by more than bf16 noise.
        assert!(pe_max < 1.0 && ns_max < 1.0);
    }

    #[test]
    #[should_panic(expected = "Polar Express runs its own tuned 5-step schedule")]
    fn polar_express_rejects_non_default_step_counts() {
        let _torch_rng_guard = test_rng::shared();
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let w = vs.root().randn("w", &[4, 4], 0.0, 1.0);
        Muon::new_named(
            &[("w".to_owned(), w)],
            MuonConfig {
                orthogonalizer: Orthogonalizer::PolarExpress5,
                ns_steps: 3,
                quiet: true,
                ..MuonConfig::default()
            },
        );
    }

    #[test]
    fn routing_names_report_the_realized_optimizer_split() {
        let _torch_rng_guard = test_rng::shared();
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let matrix = vs.root().randn("ff_out_w", &[8, 4], 0.0, 1.0);
        let gate = vs.root().randn("attn_resid_lambda", &[1], 0.0, 1.0);
        let opt = Muon::new_named(
            &[
                ("ff_out_w".to_owned(), matrix),
                ("attn_resid_lambda".to_owned(), gate),
            ],
            MuonConfig {
                quiet: true,
                ..MuonConfig::default()
            },
        );
        assert_eq!(opt.muon_param_names(), vec!["ff_out_w".to_owned()]);
        assert_eq!(
            opt.adamw_param_names(),
            vec!["attn_resid_lambda".to_owned()]
        );
    }
    fn controller_config() -> MuonConfig {
        MuonConfig {
            lr: 1e-2,
            momentum: 0.0,
            nesterov: false,
            beta2: 0.0,
            weight_decay: 0.0,
            row_learned_lr: true,
            quiet: true,
            ..MuonConfig::default()
        }
    }

    fn backward_rows(parameter: &Tensor, rows: &[f32]) {
        let gradient = Tensor::from_slice(rows).reshape(parameter.size().as_slice());
        (parameter * gradient).sum(Kind::Float).backward();
    }

    #[test]
    fn controller_off_and_enabled_warmup_are_exactly_equivalent() {
        let _torch_rng_guard = test_rng::shared();
        let initial = Tensor::from_slice(&[0.2f32, -0.4, 0.7, -0.1]).reshape([2, 2]);
        let off_param = initial.copy().set_requires_grad(true);
        let on_param = initial.copy().set_requires_grad(true);
        let mut off_cfg = controller_config();
        off_cfg.row_learned_lr = false;
        let mut off = Muon::new_named(&[("w".to_owned(), off_param.shallow_clone())], off_cfg);
        let mut on =
            Muon::new_named(&[("w".to_owned(), on_param.shallow_clone())], controller_config());

        backward_rows(&off_param, &[1.0, -2.0, 0.5, 3.0]);
        backward_rows(&on_param, &[1.0, -2.0, 0.5, 3.0]);
        off.step(StepKind::Primary);
        on.step(StepKind::Primary);

        assert!(off_param.equal(&on_param));
        assert!(off.entries_2d[0].row_lr.is_none());
        assert!(off.row_learned_lr_metrics().is_none());
        let state = on.entries_2d[0].row_lr.as_ref().unwrap();
        assert_eq!(state.logit.abs().max().double_value(&[]), 0.0);
        assert_eq!(state.logit.kind(), Kind::Float);
        assert_eq!(state.previous_delta.kind(), Kind::Float);
        assert_eq!(state.logit.device(), on_param.device());
        let metrics = on.row_learned_lr_metrics().unwrap();
        assert_eq!(metrics.alpha_mean, 1.0);
        assert_eq!(metrics.alpha_std, 0.0);
        assert_eq!(metrics.update_magnitude, 0.0);
    }

    #[test]
    fn productive_and_harmful_rows_move_alpha_in_opposite_directions_after_warmup() {
        let _torch_rng_guard = test_rng::shared();
        let parameter = Tensor::zeros([2, 2], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let mut optimizer = Muon::new_named(
            &[("w".to_owned(), parameter.shallow_clone())],
            controller_config(),
        );
        optimizer.step_count = super::ROW_LR_WARMUP_STEPS;
        optimizer.entries_2d[0]
            .row_lr
            .as_mut()
            .unwrap()
            .previous_delta
            .copy_(&Tensor::from_slice(&[-1.0f32, -1.0, 1.0, 1.0]).reshape([2, 2]));

        backward_rows(&parameter, &[1.0, 1.0, 1.0, 1.0]);
        optimizer.step(StepKind::Primary);
        let first = optimizer.row_learned_lr_metrics().unwrap();
        assert_eq!(first.alpha_mean, 1.0, "the logit update must take effect next step");
        assert!(first.evidence_mean.abs() < 1e-6);
        assert!((first.evidence_std - 1.0).abs() < 1e-6);
        let logit = &optimizer.entries_2d[0].row_lr.as_ref().unwrap().logit;
        assert!(logit.double_value(&[0, 0]) > 0.0, "productive row must speed up");
        assert!(logit.double_value(&[1, 0]) < 0.0, "harmful row must slow down");

        optimizer.zero_grad();
        backward_rows(&parameter, &[1.0, 1.0, 1.0, 1.0]);
        optimizer.step(StepKind::Primary);
        let second = optimizer.row_learned_lr_metrics().unwrap();
        assert!(second.alpha_max > 1.0);
        assert!(second.alpha_min < 1.0);
    }

    #[test]
    fn controller_alpha_is_finite_and_bounded_even_for_saturated_logits() {
        let _torch_rng_guard = test_rng::shared();
        let parameter = Tensor::zeros([2, 2], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let mut optimizer = Muon::new_named(
            &[("w".to_owned(), parameter.shallow_clone())],
            controller_config(),
        );
        optimizer.entries_2d[0]
            .row_lr
            .as_mut()
            .unwrap()
            .logit
            .copy_(&Tensor::from_slice(&[1000.0f32, -1000.0]).reshape([2, 1]));
        backward_rows(&parameter, &[0.0, 0.0, 0.0, 0.0]);
        optimizer.step(StepKind::Primary);
        let metrics = optimizer.row_learned_lr_metrics().unwrap();
        assert!(metrics.alpha_min.is_finite() && metrics.alpha_max.is_finite());
        assert!(metrics.alpha_min >= (-1.0f64).exp());
        assert!(metrics.alpha_max <= 1.0f64.exp());
        assert_eq!(metrics.alpha_bound_fraction, 1.0);
        assert_eq!(metrics.evidence_mean, 0.0);
        assert_eq!(metrics.evidence_std, 0.0);
    }

    #[test]
    fn controller_excludes_adamw_and_auxiliary_updates_from_its_state() {
        let _torch_rng_guard = test_rng::shared();
        let matrix = Tensor::zeros([2, 2], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let adamw = Tensor::zeros([2], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let mut optimizer = Muon::new_named(
            &[
                ("matrix".to_owned(), matrix.shallow_clone()),
                ("bias".to_owned(), adamw.shallow_clone()),
            ],
            controller_config(),
        );
        assert!(optimizer.entries_2d[0].row_lr.is_some());
        assert_eq!(optimizer.adamw_indices, vec![1]);

        backward_rows(&matrix, &[1.0, 2.0, 3.0, 4.0]);
        adamw.sum(Kind::Float).backward();
        optimizer.step(StepKind::Primary);
        let state = optimizer.entries_2d[0].row_lr.as_ref().unwrap();
        let before = (
            state.logit.copy(),
            state.adam_m.copy(),
            state.adam_v.copy(),
            state.previous_delta.copy(),
            state.adam_step,
        );

        optimizer.zero_grad();
        backward_rows(&matrix, &[-4.0, -3.0, -2.0, -1.0]);
        optimizer.step(StepKind::Auxiliary);
        let state = optimizer.entries_2d[0].row_lr.as_ref().unwrap();
        assert!(state.logit.equal(&before.0));
        assert!(state.adam_m.equal(&before.1));
        assert!(state.adam_v.equal(&before.2));
        assert!(state.previous_delta.equal(&before.3));
        assert_eq!(state.adam_step, before.4);
    }

    #[test]
    fn enabling_controller_does_not_change_adamw_routed_matrix_updates() {
        let _torch_rng_guard = test_rng::shared();
        let initial = Tensor::from_slice(&[0.2f32, -0.3, 0.5, 0.7]).reshape([2, 2]);
        let off_param = initial.copy().set_requires_grad(true);
        let on_param = initial.copy().set_requires_grad(true);
        let cfg = |enabled| MuonConfig {
            use_muon_for_2d: true,
            force_adamw_name_substrings: vec!["head".to_owned()],
            adamw_lr: 0.01,
            adamw_betas: (0.9, 0.999),
            row_learned_lr: enabled,
            quiet: true,
            ..MuonConfig::default()
        };
        let mut off =
            Muon::new_named(&[("head".to_owned(), off_param.shallow_clone())], cfg(false));
        let mut on =
            Muon::new_named(&[("head".to_owned(), on_param.shallow_clone())], cfg(true));
        backward_rows(&off_param, &[1.0, -2.0, 3.0, -4.0]);
        backward_rows(&on_param, &[1.0, -2.0, 3.0, -4.0]);
        off.step(StepKind::Primary);
        on.step(StepKind::Primary);
        assert!(off_param.equal(&on_param));
        assert!(on.entries_2d.is_empty());
        assert!(on.row_learned_lr_metrics().is_none());
    }

    #[test]
    fn previous_delta_is_the_actual_signed_muon_update_without_weight_decay() {
        let _torch_rng_guard = test_rng::shared();
        let parameter =
            Tensor::from_slice(&[0.5f32, -0.25, 0.75, -1.0]).reshape([2, 2]).set_requires_grad(true);
        let before = parameter.copy();
        let mut cfg = controller_config();
        cfg.weight_decay = 0.4;
        let decay = cfg.weight_decay * cfg.lr;
        let mut optimizer =
            Muon::new_named(&[("w".to_owned(), parameter.shallow_clone())], cfg);
        backward_rows(&parameter, &[1.0, -2.0, 3.0, -4.0]);
        optimizer.step(StepKind::Primary);

        let actual_without_decay = &parameter - &before * (1.0 - decay);
        let recorded = &optimizer.entries_2d[0]
            .row_lr
            .as_ref()
            .unwrap()
            .previous_delta;
        let error = (recorded - actual_without_decay.to_kind(Kind::Float))
            .abs()
            .max()
            .double_value(&[]);
        assert!(error < 1e-6, "credit delta included decay or missed the applied update: {error}");
    }
}
