//! Reference-aligned NorMuon optimizer: EMA momentum/Nesterov, Newton-Schulz 5
//! orthogonalization, per-row second-moment (NorMuon) rescaling with Frobenius-
//! norm preservation, and AdamW for non-matrix params.

use std::collections::{HashMap, HashSet};
use std::env;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use tch::{Device, Kind, Tensor};

use crate::torch::cuda::graph::{CudaGraph, CudaGraphPool};

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
    /// Whether this optimizer's primary step is ELIGIBLE for CUDA-graph capture. See
    /// [`Muon::try_graph_step`]; eligibility is necessary, not sufficient — the device,
    /// the routing and `PRETRAIN_CUDA_GRAPHS` all still get a veto.
    ///
    /// Default `false`, and the pretraining optimizer is the one caller that sets it.
    /// That step is what `PRETRAIN_CUDA_GRAPHS` is named for and the only one on which
    /// capture has been measured or validated, so eligibility is opt-in: a new caller
    /// that says nothing gets the eager step rather than silently inheriting an
    /// unvalidated fast path. The PPO update in particular could not take it — it runs
    /// its own capture (`PpoUpdateCudaGraph`) and its critic-only warmup leaves the actor
    /// without gradients for whole episodes.
    pub capture_step_graphs: bool,
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
            capture_step_graphs: false,
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

/// Per-2D-param state.
///
/// `momentum` and `second_momentum` are the tensors the rest of this file operates
/// on: the per-parameter step, the checkpoint sidecar, the footprint accounting and
/// the tests. When the parameter belongs to a [`Group2D`] they are `select(0, slot)`
/// VIEWS into that group's stacked buffer instead of standalone allocations, which is
/// what makes the batched step and the per-parameter step two ways of advancing the
/// same state rather than two states.
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
    /// `max(1, rows/cols).sqrt()` for the registered shape and layout. Precomputed so
    /// a step can resolve every effective learning rate on the host before it issues
    /// its first kernel.
    aspect_scale: f64,
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

/// Static per-parameter AdamW routing, resolved once from the name-fragment tables.
///
/// Every step used to rescan those tables for every AdamW parameter, which for the
/// pretrain split is over a thousand substring searches per step for an answer that
/// cannot change after construction.
struct AdamWSettings {
    betas: (f64, f64),
    /// `None` when the parameter is excluded from decoupled weight decay.
    weight_decay_mul: Option<f64>,
}

impl AdamWSettings {
    fn resolve(name: &str, cfg: &MuonConfig) -> Self {
        let betas = cfg
            .adamw_beta_overrides
            .iter()
            .find(|(needle, _)| name.contains(needle.as_str()))
            .map_or(cfg.adamw_betas, |(_, betas)| *betas);
        let excluded = cfg
            .adamw_no_weight_decay_name_substrings
            .iter()
            .any(|needle| name.contains(needle));
        let weight_decay_mul = (!excluded).then(|| {
            cfg.adamw_weight_decay_multipliers
                .iter()
                .find(|(needle, _)| name.contains(needle.as_str()))
                .map_or(1.0, |(_, mul)| *mul)
        });
        Self {
            betas,
            weight_decay_mul,
        }
    }
}

/// NorMuon parameters that share a shape, dtype, device and the plain `Matrix`
/// orthogonalizer layout, stepped as one batched kernel sequence.
///
/// This reverses an earlier decision not to stack same-shape parameters, on two
/// counts. The first is launch count: a 10-layer transformer holds ten identically
/// shaped copies of every projection bank, so the ~50-kernel chain per matrix —
/// momentum lerp, five quintic iterations of three matmuls each, the second-moment
/// rescale, the decay and the apply — ran ten times over. The second, and larger, is
/// that those matmuls are far too small to fill the device: one [512,512]x[512,2048]
/// bf16 `bmm` covers 64 128x128 output tiles on 170 SMs, so most of the machine idles
/// for the whole iteration, and the [512,512]x[512,512] one covers 16. Batching ten
/// of them fills 640 and 160 tiles instead.
///
/// The cost is the stacked gradient: parameter gradients are separate autograd
/// allocations, so each group gathers its members into one contiguous [G, m, n] tensor
/// per step. Across all 43 NorMuon matrices of the pretrain trunk that is ~126 MB
/// read plus ~126 MB written, roughly 180 us of bandwidth against several
/// milliseconds of matmul and dispatch time. Momentum and the second moment are NOT
/// gathered: the group owns them stacked and hands each member a view, so they cost
/// nothing per step.
///
/// Learning rate, weight decay and the aspect scale never enter the batched region.
/// They appear only in the per-parameter decay/apply tail, which reads `lr_scales` one
/// parameter at a time exactly as it always has. That is why this key is purely
/// structural: no later `set_named_lr_scale` can make a group wrong.
struct Group2D {
    /// Indices into `entries_2d`, in registration order.
    entries: Vec<usize>,
    /// `[G, m, n]`, owner of every member's momentum view.
    momentum: Tensor,
    /// `[G, ..]`, owner of every member's second-moment view.
    second_momentum: Tensor,
}

/// Cap on a group's stacked parameter bytes.
///
/// The batched region holds the bf16 iterate, two bf16 Gram matrices and the fp32
/// rescale transients at once, so its peak is a small multiple of the stacked bytes
/// rather than of one matrix's; this bounds that peak. It also makes the batching
/// self-selecting: matrices small enough to leave the device idle group deeply,
/// matrices already large enough to saturate it group shallowly or not at all, which
/// is exactly where each behaviour is wanted.
///
/// 64 MiB admits the whole of a 10-layer 512-wide transformer's largest family (ten
/// fp32 [512, 2048] copies is 40 MiB) with room to spare, at a transient peak in the
/// low hundreds of megabytes.
const GROUP_STACK_BYTES_LIMIT: usize = 64 << 20;

/// One host scalar the step's kernels consume, in whichever form the current path
/// needs.
///
/// `Host` passes an ATen `Scalar` argument, which is bit-for-bit what this optimizer
/// has always issued. `Device` passes a persistent 0-dim tensor, and that is the whole
/// reason a captured CUDA graph can follow the learning-rate and momentum schedules:
/// capture records a kernel's ARGUMENTS, so a scheduled `Scalar` would be frozen at
/// whatever the capture step happened to hold — a silent, invisible freeze of the
/// entire schedule. A tensor operand records an ADDRESS instead, and the host rewrites
/// its contents before every replay.
enum StepScalar {
    Host(f64),
    Device(Tensor),
}

impl StepScalar {
    fn mul(&self, tensor: &Tensor) -> Tensor {
        match self {
            Self::Host(value) => tensor.g_mul_scalar(*value),
            Self::Device(value) => tensor.g_mul(value),
        }
    }

    fn mul_(&self, tensor: &mut Tensor) {
        match self {
            Self::Host(value) => {
                let _ = tensor.g_mul_scalar_(*value);
            }
            Self::Device(value) => {
                let _ = tensor.g_mul_(value);
            }
        }
    }

    fn lerp_(&self, tensor: &mut Tensor, end: &Tensor) {
        match self {
            Self::Host(value) => {
                let _ = tensor.lerp_(end, *value);
            }
            Self::Device(value) => {
                let _ = tensor.lerp_tensor_(end, value);
            }
        }
    }

    fn lerp(&self, start: &Tensor, end: &Tensor) -> Tensor {
        match self {
            Self::Host(value) => start.lerp(end, *value),
            Self::Device(value) => start.lerp_tensor(end, value),
        }
    }
}

/// Every scheduled scalar one step needs, resolved once per step, for both paths, out
/// of the same host arithmetic.
struct StepScalars {
    /// NorMuon first-moment lerp weight, `1 - momentum`.
    normuon_lerp: StepScalar,
    /// Nesterov lerp weight, `momentum`.
    nesterov: StepScalar,
    /// Per `entries_2d` entry: `-(lr * lr_scale * aspect_scale)`.
    normuon_step: Vec<StepScalar>,
    /// Per `entries_2d` entry: the decay factor in the form the configured mask needs,
    /// `decay` when cautious and `1 - decay` otherwise. `None` omits the decay.
    normuon_decay: Vec<Option<StepScalar>>,
    /// Per `adamw_indices` position: `-lr * lr_scale / bias_correction1`.
    adamw_step: Vec<StepScalar>,
    /// Per `adamw_indices` position: `1 / sqrt(bias_correction2)`.
    adamw_inv_bc2_sqrt: Vec<StepScalar>,
    /// Per `adamw_indices` position, same convention as `normuon_decay`.
    adamw_decay: Vec<Option<StepScalar>>,
}

const SLOT_NORMUON_LERP: usize = 0;
const SLOT_NESTEROV: usize = 1;
const SHARED_SLOT_COUNT: usize = 2;
const NORMUON_SLOTS_PER_ENTRY: usize = 2;
const ADAMW_SLOTS_PER_PARAM: usize = 3;

/// Device-resident mirrors of every scheduled scalar, packed into one tensor.
///
/// One slot per (parameter, role), not one per distinct value: a refresh is a single
/// host-to-device copy of a couple of kilobytes, so the slot count is free, and a
/// fixed slot per role removes any possibility of two roles sharing a slot on one step
/// and not the next, which would silently corrupt a replay.
struct StepScalarPack {
    /// `[slots]`, in the parameter dtype, on the parameter device.
    values: Tensor,
    /// One 0-dim view per slot, materialized once. Capture records these addresses, so
    /// they have to outlive every replay.
    slots: Vec<Tensor>,
    /// Host staging buffer, rewritten in full every step. `f64` so the copy's
    /// narrowing to the parameter dtype is the same rounding ATen applies to a
    /// `Scalar` argument.
    host: Vec<f64>,
}

impl StepScalarPack {
    fn new(entries: usize, adamw: usize, kind: Kind, device: Device) -> Self {
        let count =
            SHARED_SLOT_COUNT + entries * NORMUON_SLOTS_PER_ENTRY + adamw * ADAMW_SLOTS_PER_PARAM;
        let values = Tensor::zeros([count as i64], (kind, device));
        let slots = (0..count as i64)
            .map(|slot| values.select(0, slot))
            .collect();
        Self {
            values,
            slots,
            host: vec![0.0; count],
        }
    }

    fn set(&mut self, slot: usize, value: f64) -> StepScalar {
        self.host[slot] = value;
        StepScalar::Device(self.slots[slot].shallow_clone())
    }

    /// Publish this step's schedule. Issued on the default stream, which the graph's
    /// stream scope then orders itself after, so the replay reads what was just written.
    ///
    /// ATen makes a copy out of pageable host memory stream-synchronizing, so this is
    /// also a host wait on the outstanding backward. That is why the pack exists only on
    /// the graph path: it buys the elimination of every launch in the step, and the
    /// fallback step keeps issuing `Scalar` arguments with no host-to-device traffic at
    /// all.
    fn upload(&self) {
        let mut values = self.values.shallow_clone();
        values.copy_(&Tensor::from_slice(&self.host));
    }
}

/// Warmup steps issued on the capture stream before capture. cuBLAS algorithm
/// selection and the caching allocator need a few iterations to reach steady state.
/// One optimizer step per training step is the only warmup shape available here:
/// `PpoUpdateCudaGraph` can run its body three times inside one call because that body
/// only computes gradients, whereas this one mutates parameters.
const GRAPH_WARMUP_STEPS: usize = 3;

/// How many times one run may capture its optimizer step before giving up on graphs.
///
/// Each arm mints a private mempool holding that capture's transient working set, and
/// those blocks are retained for the process's life, so the count is a VRAM multiplier.
/// Three admits the arming this design actually expects - the first capture, plus a
/// re-arm after a routing setter and after a checkpoint load - and refuses a run that has
/// started churning.
const MAX_STEP_GRAPH_ARMS: usize = 3;

#[derive(Clone, Copy)]
enum GraphSlotState {
    Warmup(usize),
    ReadyToCapture,
    Captured,
}

/// What a primary step's captured-graph attempt did to the update.
enum GraphStepOutcome {
    /// No graph was available. The caller runs the eager step.
    Fallback,
    /// The captured body ran and the update is applied.
    Applied,
    /// A capture or replay failed part-way through the update, so the step is neither
    /// applied nor safe to re-run. Anything it would have consumed stays where it is.
    Forfeited,
}

/// One captured optimizer body.
struct StepGraph {
    graph: CudaGraph,
    state: GraphSlotState,
    /// What this body was RECORDED reading, empty until it was. Checked before every
    /// replay. See [`Muon::step_participation`].
    participation: Vec<Option<usize>>,
}

impl StepGraph {
    fn new(graph: CudaGraph) -> Self {
        Self {
            graph,
            state: GraphSlotState::Warmup(GRAPH_WARMUP_STEPS),
            participation: Vec::new(),
        }
    }
}

/// `adamw_every` gives the step exactly two shapes — NorMuon alone, and NorMuon
/// followed by AdamW — so each gets its own graph and the host picks between them on
/// the step counter.
///
/// That cadence and the stepped-parameter set are the step's only data-dependent host
/// control flow. The cadence is a fixed function of the step counter, so each of its two
/// values gets a body; the parameter set is not, so it is recorded per body and checked
/// before every replay. Everything else that varies per step is a scalar, and every
/// scalar lives in `scalars`.
struct MuonStepGraphs {
    normuon_only: StepGraph,
    with_adamw: StepGraph,
    scalars: StepScalarPack,
}

/// Boxed because it starts dormant: the graphs arm lazily, on the first primary step,
/// once both the environment and the parameters' device are known.
enum StepGraphState {
    Unarmed,
    Disabled,
    Armed(Box<MuonStepGraphs>),
}

/// Why a step cannot be captured, and whether anything about the run could change it.
struct StepGraphBlocker {
    reason: String,
    /// Set when the obstacle is a property of the build, the machine, or the model's
    /// routing rather than a choice: with capture on by default, a run that lands on
    /// one of those has nothing to act on and no reason to be told.
    inherent: bool,
}

impl StepGraphBlocker {
    fn inherent(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            inherent: true,
        }
    }

    fn actionable(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            inherent: false,
        }
    }
}

pub struct Muon {
    cfg: MuonConfig,
    entries_2d: Vec<Entry2D>,
    /// `entries_2d` positions batched together, in group order. Built once, at
    /// construction, from registered shapes alone.
    groups_2d: Vec<Group2D>,
    adamw_indices: Vec<usize>,
    /// [`AdamWSettings`] per `adamw_indices` position.
    adamw_settings: Vec<AdamWSettings>,
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
    /// Captured optimizer bodies. Armed on the first CUDA primary step of an optimizer
    /// whose [`MuonConfig::capture_step_graphs`] is set, unless `PRETRAIN_CUDA_GRAPHS=0`.
    step_graphs: StepGraphState,
    /// Arms performed, against [`MAX_STEP_GRAPH_ARMS`]. Counts the retained mempools.
    step_graph_arms: usize,
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
    // `norm()`, which the per-parameter path uses, accumulates in fp32 and rounds once.
    // A `square().sum(.., BFloat16)` would round the sum of squares BEFORE the sqrt, and at
    // a sum of ~2.6e5 the bf16 spacing is 1024: 0.2% on the divisor the whole quintic is
    // normalized by. `norm.ScalarOpt_dim` is the same fp32-accumulated reduction, per slot.
    let nrm = prescale_divisor(
        &x3d.norm_scalaropt_dim(2, [-2i64, -1].as_slice(), true),
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

/// The per-matrix `max(1, rows/cols).sqrt()` learning-rate factor, derived from the
/// registered shape and layout alone.
fn ortho_aspect_scale(size: &[i64], layout: OrthoLayout) -> f64 {
    let (rows, cols) = match layout {
        OrthoLayout::Matrix => (size[0], size[1]),
        OrthoLayout::RowHeads { head_dim, .. } => (head_dim, size[1]),
        OrthoLayout::ColHeads { head_dim, .. } => (size[0], head_dim),
    };
    (1.0_f64).max(rows as f64 / cols as f64).sqrt()
}

fn normuon_transform(
    update: &Tensor,
    layout: OrthoLayout,
    second_momentum: &mut Tensor,
    beta2: f64,
    orth: Orthogonalizer,
    ns_steps: usize,
) -> Tensor {
    match layout {
        OrthoLayout::Matrix => {
            let update = quintic_orthogonalize(update, orth, ns_steps);
            normuon_rescale(&update, second_momentum, beta2)
        }
        OrthoLayout::RowHeads { heads, head_dim } => {
            let cols = update.size()[1];
            let blocks = update.reshape([heads, head_dim, cols]);
            let blocks = batched_quintic_orthogonalize(&blocks, orth, ns_steps);
            let blocks = normuon_rescale(&blocks, second_momentum, beta2);
            blocks.reshape(update.size().as_slice())
        }
        OrthoLayout::ColHeads { heads, head_dim } => {
            let rows = update.size()[0];
            let blocks = update
                .reshape([rows, heads, head_dim])
                .permute([1, 0, 2])
                .contiguous();
            let blocks = batched_quintic_orthogonalize(&blocks, orth, ns_steps);
            let blocks = normuon_rescale(&blocks, second_momentum, beta2);
            blocks
                .permute([1, 0, 2])
                .contiguous()
                .reshape(update.size().as_slice())
        }
    }
}

/// Decoupled weight decay and then the update itself, both reading the pre-step
/// parameter so the composition is `p - decay*p - lr*u`. Returns the signed
/// gradient-only delta, which is the row-learned-rate controller's credit tensor.
///
/// Shared verbatim by the batched and the per-parameter path. The only difference
/// between them is whether `update` is a slice of a group's stacked update.
fn apply_normuon_update(
    p: &mut Tensor,
    update: &Tensor,
    step: &StepScalar,
    decay: Option<&StepScalar>,
    cautious: bool,
) -> Tensor {
    if let Some(decay) = decay {
        if cautious {
            let keep = decay.mul(&(update * &*p).ge(0).to_kind(p.kind()));
            let decayed = &*p * keep;
            let _ = p.g_sub_(&decayed);
        } else {
            decay.mul_(p);
        }
    }
    let signed_delta = step.mul(update);
    let _ = p.g_add_(&signed_delta);
    signed_delta
}

/// Partition the `Matrix`-layout NorMuon entries into batched groups and rewire each
/// member's state onto a slice of its group's stacked buffers. Runs once, at
/// construction, off nothing but registered shapes.
///
/// Entries carrying a row-learned-rate controller are left ungrouped: that controller
/// normalizes its evidence over one parameter's rows and keeps a per-parameter credit
/// tensor, so batching it would have to reproduce both per slice before it could be
/// called equivalent.
fn group_entries_2d(entries: &mut [Entry2D], params: &[Tensor]) -> Vec<Group2D> {
    struct Candidate {
        size: Vec<i64>,
        kind: Kind,
        device: Device,
        members: Vec<usize>,
    }
    let mut candidates: Vec<Candidate> = Vec::new();
    for (position, entry) in entries.iter().enumerate() {
        if entry.layout != OrthoLayout::Matrix || entry.row_lr.is_some() {
            continue;
        }
        let param = &params[entry.idx];
        let (size, kind, device) = (param.size(), param.kind(), param.device());
        match candidates
            .iter_mut()
            .find(|c| c.size == size && c.kind == kind && c.device == device)
        {
            Some(candidate) => candidate.members.push(position),
            None => candidates.push(Candidate {
                size,
                kind,
                device,
                members: vec![position],
            }),
        }
    }

    let mut groups = Vec::new();
    for candidate in candidates {
        let elements: i64 = candidate.size.iter().product();
        let bytes = elements as usize * candidate.kind.elt_size_in_bytes();
        let cap = (GROUP_STACK_BYTES_LIMIT / bytes.max(1)).max(1);
        for chunk in candidate.members.chunks(cap) {
            if chunk.len() < 2 {
                continue;
            }
            let count = chunk.len() as i64;
            let stacked = |shape: &[i64]| {
                let mut batched = Vec::with_capacity(shape.len() + 1);
                batched.push(count);
                batched.extend_from_slice(shape);
                batched
            };
            let momentum = Tensor::zeros(
                stacked(&candidate.size).as_slice(),
                (candidate.kind, candidate.device),
            );
            let second_momentum = Tensor::zeros(
                stacked(&second_momentum_shape(&candidate.size, OrthoLayout::Matrix)).as_slice(),
                (Kind::Float, candidate.device),
            );
            for (slot, &position) in chunk.iter().enumerate() {
                entries[position].momentum = momentum.select(0, slot as i64);
                entries[position].second_momentum = second_momentum.select(0, slot as i64);
            }
            groups.push(Group2D {
                entries: chunk.to_vec(),
                momentum,
                second_momentum,
            });
        }
    }
    groups
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

        let raw_f32 = raw_gradient.to_kind(Kind::Float).nan_to_num(0.0, 0.0, 0.0);
        let evidence_raw =
            -(&raw_f32 * &self.previous_delta).mean_dim([1i64].as_slice(), true, Kind::Float);
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
            let _ = self.adam_v.lerp_(&objective_gradient.square(), 1.0 - beta2);
            self.adam_step += 1;
            let bc1 = 1.0 - beta1.powi(self.adam_step as i32);
            let bc2 = 1.0 - beta2.powi(self.adam_step as i32);
            let denom = self.adam_v.sqrt() / bc2.sqrt() + ROW_LR_CONTROLLER_EPS;
            let logit_delta = &self.adam_m / denom * (-ROW_LR_CONTROLLER_LR / bc1);
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
                    aspect_scale: ortho_aspect_scale(&size, layout),
                    row_lr: cfg.row_learned_lr.then(|| RowLrState::new(m, n, device)),
                });
            } else {
                adamw_indices.push(i);
            }
        }
        let groups_2d = group_entries_2d(&mut entries_2d, &params);
        let adamw_settings: Vec<AdamWSettings> = adamw_indices
            .iter()
            .map(|&idx| AdamWSettings::resolve(&names[idx], &cfg))
            .collect();

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
                let batched: usize = groups_2d.iter().map(|group| group.entries.len()).sum();
                if batched > 0 {
                    println!(
                        "  batched NorMuon groups: {} params in {} group(s) of identical shape",
                        batched,
                        groups_2d.len()
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
            groups_2d,
            adamw_indices,
            adamw_settings,
            adamw_state: HashMap::new(),
            step_count: 0,
            params,
            lr_scales: vec![1.0; names.len()],
            step_enabled: vec![true; names.len()],
            adamw_pending_grads: false,
            row_lr_metrics: None,
            step_graphs: StepGraphState::Unarmed,
            step_graph_arms: 0,
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
            let do_adamw = self.begin_step(primary);
            let applied = match if primary {
                self.try_graph_step(do_adamw)
            } else {
                GraphStepOutcome::Fallback
            } {
                GraphStepOutcome::Fallback => {
                    let scalars = self.resolve_step_scalars(None);
                    self.step_all_normuon(primary, &scalars);
                    if do_adamw {
                        self.step_all_adamw(&scalars);
                    }
                    true
                }
                GraphStepOutcome::Applied => true,
                GraphStepOutcome::Forfeited => false,
            };
            // AdamW's retained gradient is released only once an update has consumed it.
            // A forfeited step consumed nothing, so the accumulation carries into the next
            // one rather than being thrown away: the cadence's whole point is that the
            // gradient survives the steps that do not spend it.
            self.adamw_pending_grads = !(do_adamw && applied);
        });
    }

    /// Advance this step's host-side clocks and report whether AdamW runs.
    ///
    /// The AdamW moment clocks are advanced here rather than inside the update because
    /// the bias corrections they feed are host scalars: a captured graph cannot
    /// recompute them, and both paths must derive them from the same place.
    fn begin_step(&mut self, primary: bool) -> bool {
        let do_adamw = if primary {
            self.step_count += 1;
            self.step_count % (self.cfg.adamw_every.max(1) as i64) == 0
        } else {
            false
        };
        if do_adamw {
            self.advance_adamw_clocks();
        }
        do_adamw
    }

    /// Advance the moment clock of, and materialize the lazy moments for, exactly the
    /// AdamW parameters this step will update.
    fn advance_adamw_clocks(&mut self) {
        for position in 0..self.adamw_indices.len() {
            let idx = self.adamw_indices[position];
            if !self.step_enabled[idx] {
                continue;
            }
            let grad = self.params[idx].grad();
            if !grad.defined() {
                continue;
            }
            let state = self
                .adamw_state
                .entry(idx)
                .or_insert_with(|| AdamWParamState {
                    m: Tensor::zeros_like(&grad),
                    v: Tensor::zeros_like(&grad),
                    step_count: 0,
                });
            state.step_count += 1;
        }
    }

    /// Resolve every scheduled scalar this step needs, once.
    ///
    /// With `pack`, each one is written into a device-resident slot and handed back as
    /// a handle to that slot, which is what lets a captured graph follow the schedule.
    /// Without it they stay ATen `Scalar` arguments, bit-for-bit what this optimizer
    /// has always issued. Both forms come out of the same host arithmetic, so the
    /// graphed and the eager step cannot drift apart.
    fn resolve_step_scalars(&self, pack: Option<&mut StepScalarPack>) -> StepScalars {
        let mut pack = pack;
        let mut take = |slot: usize, value: f64| match pack.as_deref_mut() {
            Some(pack) => pack.set(slot, value),
            None => StepScalar::Host(value),
        };
        let base_lr = self.cfg.lr;
        let wd = self.cfg.weight_decay;
        let cautious = self.cfg.cautious_weight_decay;
        let quadratic = self.cfg.quadratic_lr_weight_decay;
        let normuon_lerp = take(SLOT_NORMUON_LERP, 1.0 - self.cfg.momentum);
        let nesterov = take(SLOT_NESTEROV, self.cfg.momentum);
        let mut normuon_step = Vec::with_capacity(self.entries_2d.len());
        let mut normuon_decay = Vec::with_capacity(self.entries_2d.len());
        for (position, entry) in self.entries_2d.iter().enumerate() {
            let slot = SHARED_SLOT_COUNT + position * NORMUON_SLOTS_PER_ENTRY;
            let lr = base_lr * self.lr_scales[entry.idx];
            let eff_lr = lr * entry.aspect_scale;
            let decay = if quadratic {
                wd * base_lr * eff_lr
            } else {
                wd * lr
            };
            normuon_step.push(take(slot, -eff_lr));
            // Whether the decay runs keys off the configured `weight_decay`, never off
            // this step's learning rate: a captured body cannot drop kernels when the
            // schedule anneals, and it does not need to, because a zero decay leaves
            // both `p * 1` and `p - p * 0` exact.
            normuon_decay.push(
                (wd > 0.0).then(|| take(slot + 1, if cautious { decay } else { 1.0 - decay })),
            );
        }
        let adamw_wd = self.cfg.adamw_wd;
        let adamw_base = self.adamw_slot_base();
        let mut adamw_step = Vec::with_capacity(self.adamw_indices.len());
        let mut adamw_inv_bc2_sqrt = Vec::with_capacity(self.adamw_indices.len());
        let mut adamw_decay = Vec::with_capacity(self.adamw_indices.len());
        for (position, &idx) in self.adamw_indices.iter().enumerate() {
            let slot = adamw_base + position * ADAMW_SLOTS_PER_PARAM;
            let lr = self.cfg.adamw_lr * self.lr_scales[idx];
            let settings = &self.adamw_settings[position];
            let (beta1, beta2) = settings.betas;
            // The clamp only covers parameters this step does not touch, whose slots
            // exist to keep the layout fixed and are never read.
            let count = self
                .adamw_state
                .get(&idx)
                .map_or(0, |state| state.step_count)
                .max(1) as i32;
            let bc1 = 1.0 - beta1.powi(count);
            let bc2 = 1.0 - beta2.powi(count);
            adamw_step.push(take(slot, -lr / bc1));
            adamw_inv_bc2_sqrt.push(take(slot + 1, 1.0 / bc2.sqrt()));
            let decay_mul = settings.weight_decay_mul.filter(|_| adamw_wd > 0.0);
            adamw_decay.push(decay_mul.map(|wd_mul| {
                let decay = if quadratic {
                    lr * lr * adamw_wd * wd_mul
                } else {
                    lr * adamw_wd * wd_mul
                };
                take(slot + 2, if cautious { decay } else { 1.0 - decay })
            }));
        }
        StepScalars {
            normuon_lerp,
            nesterov,
            normuon_step,
            normuon_decay,
            adamw_step,
            adamw_inv_bc2_sqrt,
            adamw_decay,
        }
    }

    /// Resolve this step's schedule into `pack`'s device slots and publish it, returning
    /// the handles the body must read.
    ///
    /// The two halves belong together and in this order: the slots have to hold this
    /// step's values before any kernel that reads them is issued, and a captured body
    /// reads them at replay rather than at capture. Kept in one place so that the CPU
    /// test of the device-scalar arithmetic exercises the production ordering instead of
    /// restating it.
    fn publish_step_scalars(&self, pack: &mut StepScalarPack) -> StepScalars {
        let scalars = self.resolve_step_scalars(Some(pack));
        pack.upload();
        scalars
    }

    /// First [`StepScalarPack`] slot belonging to the AdamW branch.
    fn adamw_slot_base(&self) -> usize {
        SHARED_SLOT_COUNT + self.entries_2d.len() * NORMUON_SLOTS_PER_ENTRY
    }

    /// Run one primary step out of a captured CUDA graph.
    ///
    /// On wherever capture is possible for an eligible optimizer, matching the
    /// `PPO_CUDA_GRAPHS` convention that the fast path is the default and `=0` opts out.
    /// Every case capture cannot serve is detected in [`Self::step_graph_blocker`] and
    /// degrades to the eager step. A capture or replay failure disables the graph
    /// permanently and forfeits that one step rather than re-run a body that may have
    /// already applied part of the update: one skipped optimizer step is invisible in a
    /// pretraining run, a doubled one is not.
    fn try_graph_step(&mut self, do_adamw: bool) -> GraphStepOutcome {
        let mut graphs = match std::mem::replace(&mut self.step_graphs, StepGraphState::Disabled) {
            StepGraphState::Unarmed => match self.arm_step_graphs() {
                Some(graphs) => graphs,
                None => return GraphStepOutcome::Fallback,
            },
            StepGraphState::Disabled => return GraphStepOutcome::Fallback,
            StepGraphState::Armed(graphs) => graphs,
        };
        let scalars = self.publish_step_scalars(&mut graphs.scalars);
        let (slot, label) = if do_adamw {
            (&mut graphs.with_adamw, "NorMuon+AdamW")
        } else {
            (&mut graphs.normuon_only, "NorMuon")
        };
        // What the step reads is baked into the capture: which parameters it touches is
        // host control flow, since a parameter with no gradient contributes no kernels at
        // all, and the gradient buffers themselves are addresses in the recorded kernels.
        // A body recorded against a different set would keep skipping — or keep stepping,
        // out of the wrong memory — silently, for the rest of the run, and nothing
        // downstream can detect it. So the record is checked before every replay and a
        // change forfeits both bodies. This is the one piece of the step that neither a
        // blocker nor a setter can rule out in advance. The print doubles as the churn
        // signal, since [`MAX_STEP_GRAPH_ARMS`] retained mempools is where it ends.
        if matches!(slot.state, GraphSlotState::Captured)
            && !self.participation_matches(&slot.participation)
        {
            println!(
                "pretrain CUDA graphs recapturing: what the {label} optimizer step reads \
                 changed, so the captured body is stale"
            );
            self.step_graphs = StepGraphState::Unarmed;
            return GraphStepOutcome::Fallback;
        }
        let state = slot.state;
        let outcome = match state {
            GraphSlotState::Warmup(remaining) => {
                slot.state = if remaining <= 1 {
                    GraphSlotState::ReadyToCapture
                } else {
                    GraphSlotState::Warmup(remaining - 1)
                };
                slot.graph
                    .with_stream_scope(|_| self.run_step_body(&scalars, do_adamw))
                    .map(|()| false)
            }
            GraphSlotState::ReadyToCapture => {
                slot.state = GraphSlotState::Captured;
                slot.participation = self.step_participation();
                // Capture RECORDS without executing, so the single replay inside the
                // same stream scope is this step's one and only update.
                slot.graph
                    .with_stream_scope(|graph| {
                        graph.capture(|| self.run_step_body(&scalars, do_adamw))?;
                        graph.replay()
                    })
                    .and_then(|inner| inner)
                    .map(|()| true)
            }
            GraphSlotState::Captured => slot
                .graph
                .with_stream_scope(CudaGraph::replay)
                .and_then(|inner| inner)
                .map(|()| false),
        };
        match outcome {
            Ok(captured) => {
                if captured {
                    println!("pretrain CUDA graph captured: {label} optimizer step");
                }
                self.step_graphs = StepGraphState::Armed(graphs);
                GraphStepOutcome::Applied
            }
            Err(err) => {
                println!(
                    "pretrain CUDA graphs disabled: {label} optimizer step failed ({err}); \
                     this step is forfeited and every later step runs eagerly"
                );
                GraphStepOutcome::Forfeited
            }
        }
    }

    /// Build both step graphs, or report why not. Runs on the first primary step, so
    /// construction stays independent of the environment and of the device the parameters
    /// ended up on, and again after anything that invalidates a capture.
    ///
    /// Bounded, because every arm mints a private mempool whose blocks are retained for
    /// the process's life: a run that keeps invalidating its captures would otherwise
    /// trade a few percent of step time for an unbounded reservation and eventually die
    /// of a device OOM with nothing pointing here. Past the bound it gives up for good
    /// and runs eagerly, which costs throughput and nothing else.
    fn arm_step_graphs(&mut self) -> Option<Box<MuonStepGraphs>> {
        if let Some(blocker) = self.step_graph_blocker() {
            // A run that asked for graphs and cannot have them must say so. A run that
            // took the default and merely landed somewhere capture is impossible has
            // nothing to act on, so it degrades quietly; anything the caller could
            // actually lift is reported either way.
            if !blocker.inherent || self.step_graphs_requested() {
                println!("pretrain CUDA graphs disabled: {}", blocker.reason);
            }
            return None;
        }
        if self.step_graph_arms >= MAX_STEP_GRAPH_ARMS {
            println!(
                "pretrain CUDA graphs disabled: {MAX_STEP_GRAPH_ARMS} captures already \
                 retained and the step keeps changing shape; running eagerly from here"
            );
            return None;
        }
        self.step_graph_arms += 1;
        let reference = &self.params[self.entries_2d[0].idx];
        let (kind, device) = (reference.kind(), reference.device());
        // The two bodies never overlap in time and never carry pool memory across a
        // step: exactly one of them is replayed per primary step, each inside a stream
        // scope that orders its work after the previous scope's, and every tensor a
        // capture allocates is dropped before that capture ends. Their transients can
        // therefore sit at the same addresses, so both capture into one private mempool
        // rather than each retaining a copy of the working set for the process's life.
        // `with_adamw` is a superset of `normuon_only`, so that is close to halving it.
        let pool = match CudaGraphPool::new() {
            Ok(pool) => pool,
            Err(err) => {
                println!("pretrain CUDA graphs disabled: init failed ({err})");
                return None;
            }
        };
        let graphs = match (
            CudaGraph::new_in_pool(device, &pool),
            CudaGraph::new_in_pool(device, &pool),
        ) {
            (Ok(Some(normuon_only)), Ok(Some(with_adamw))) => MuonStepGraphs {
                normuon_only: StepGraph::new(normuon_only),
                with_adamw: StepGraph::new(with_adamw),
                scalars: StepScalarPack::new(
                    self.entries_2d.len(),
                    self.adamw_indices.len(),
                    kind,
                    device,
                ),
            },
            (Err(err), _) | (_, Err(err)) => {
                println!("pretrain CUDA graphs disabled: init failed ({err})");
                return None;
            }
            _ => {
                println!("pretrain CUDA graphs disabled: CUDA Graph support is unavailable");
                return None;
            }
        };
        println!(
            "pretrain CUDA graphs armed: {} NorMuon params in {} batched group(s), {} AdamW \
             params, two captured bodies sharing one private mempool, \
             {GRAPH_WARMUP_STEPS} warmup steps each",
            self.entries_2d.len(),
            self.groups_2d.len(),
            self.adamw_indices.len()
        );
        Some(Box::new(graphs))
    }

    /// What the step's kernels will read, per parameter: `None` when the parameter is not
    /// stepped at all, otherwise the storage address of the gradient it steps out of.
    ///
    /// Both halves are capture inputs. Participation is host control flow —
    /// [`Self::step_all_normuon`], [`Self::step_all_adamw`] and
    /// [`Self::advance_adamw_clocks`] all skip anything outside the set, so a parameter
    /// outside it contributes no kernels at all. The address is a kernel ARGUMENT, baked
    /// in by capture. The parameters and the optimizer moments are owned here and mutated
    /// in place, so their addresses cannot move without going through a path that already
    /// re-arms; gradients are autograd's, not this optimizer's, which is the whole reason
    /// they are the half that has to be checked rather than assumed.
    fn step_participation(&self) -> Vec<Option<usize>> {
        self.params
            .iter()
            .zip(&self.step_enabled)
            .map(|(parameter, &enabled)| Self::stepped_gradient(parameter, enabled))
            .collect()
    }

    fn stepped_gradient(parameter: &Tensor, enabled: bool) -> Option<usize> {
        if !enabled {
            return None;
        }
        let gradient = parameter.grad();
        gradient.defined().then(|| gradient.data_ptr() as usize)
    }

    /// Whether the step still reads exactly what was `recorded` at capture.
    /// Allocation-free and short-circuiting, because this runs before every replay.
    fn participation_matches(&self, recorded: &[Option<usize>]) -> bool {
        recorded.len() == self.params.len()
            && self
                .params
                .iter()
                .zip(&self.step_enabled)
                .zip(recorded)
                .all(|((parameter, &enabled), &was)| {
                    Self::stepped_gradient(parameter, enabled) == was
                })
    }

    /// Whether the run explicitly turned graphs off. An eligible optimizer captures by
    /// default, so `PRETRAIN_CUDA_GRAPHS=0` is the opt-out and `=1` asks for nothing it
    /// would not already get - it only makes an unavoidable blocker speak up.
    fn step_graphs_opted_out(&self) -> bool {
        env::var("PRETRAIN_CUDA_GRAPHS").ok().as_deref() == Some("0")
    }

    /// Whether the run explicitly ASKED for graphs, as opposed to taking the default.
    fn step_graphs_requested(&self) -> bool {
        env::var("PRETRAIN_CUDA_GRAPHS").ok().as_deref() == Some("1")
    }

    /// Every reason this step cannot be captured, checked once.
    ///
    /// Ordered so that an optimizer capture was never offered to answers first and
    /// silently: PPO, the planner and every CPU unit test decline on eligibility before
    /// any check that reports, so they say nothing whatever the environment holds. The
    /// remaining order puts what nothing about this run could lift ahead of what the
    /// caller could act on, so a CPU-only process never reaches a reporting check either.
    fn step_graph_blocker(&self) -> Option<StepGraphBlocker> {
        if !self.cfg.capture_step_graphs {
            return Some(StepGraphBlocker::inherent(
                "capture is not enabled for this optimizer",
            ));
        }
        if self.step_graphs_opted_out() {
            return Some(StepGraphBlocker::actionable(
                "disabled by PRETRAIN_CUDA_GRAPHS=0",
            ));
        }
        if !CudaGraph::is_available() {
            return Some(StepGraphBlocker::inherent(
                "CUDA Graph support is unavailable",
            ));
        }
        let Some(first) = self.entries_2d.first() else {
            return Some(StepGraphBlocker::inherent("no NorMuon-routed parameters"));
        };
        let reference = &self.params[first.idx];
        let (kind, device) = (reference.kind(), reference.device());
        if !device.is_cuda() {
            return Some(StepGraphBlocker::inherent(format!(
                "parameters live on {device:?}"
            )));
        }
        // The schedule scalars are one packed tensor of one dtype, and every kernel that
        // reads a slot has to match its operand's dtype in place.
        if self
            .params
            .iter()
            .any(|p| p.kind() != kind || p.device() != device)
        {
            return Some(StepGraphBlocker::actionable(
                "parameters do not share one dtype and device",
            ));
        }
        if self.cfg.row_learned_lr {
            // The controller normalizes its evidence over one parameter's rows and
            // accumulates a diagnostics tensor per stepped matrix; both would have to be
            // reproduced under a fixed observation set before this body is capturable.
            return Some(StepGraphBlocker::actionable(
                "the row-learned learning-rate controller is enabled",
            ));
        }
        None
    }

    fn run_step_body(&mut self, scalars: &StepScalars, do_adamw: bool) {
        self.step_all_normuon(true, scalars);
        if do_adamw {
            self.step_all_adamw(scalars);
        }
    }

    fn step_all_normuon(&mut self, primary: bool, scalars: &StepScalars) {
        let beta2 = self.cfg.beta2;
        let nesterov = self.cfg.nesterov;
        let cautious = self.cfg.cautious_weight_decay;
        let orth = self.cfg.orthogonalizer;
        let ns_steps = self.cfg.ns_steps;
        let primary_step = self.step_count;
        let Self {
            entries_2d,
            groups_2d,
            params,
            step_enabled,
            row_lr_metrics,
            ..
        } = self;
        let mut accumulator = RowLrMetricAccumulator::default();
        let mut batched = vec![false; entries_2d.len()];

        // Batched groups first. Every member's momentum and second moment is a view into
        // the group's stacked buffer, so a group that is not wholly steppable this step
        // simply falls through to the per-parameter path below on exactly the same state.
        for group in groups_2d.iter_mut() {
            let steppable = group.entries.iter().all(|&position| {
                let idx = entries_2d[position].idx;
                step_enabled[idx] && params[idx].grad().defined()
            });
            if !steppable {
                continue;
            }
            let gradients: Vec<Tensor> = group
                .entries
                .iter()
                .map(|&position| params[entries_2d[position].idx].grad())
                .collect();
            let gradient = Tensor::stack(&gradients, 0);
            scalars.normuon_lerp.lerp_(&mut group.momentum, &gradient);
            let update = if nesterov {
                scalars.nesterov.lerp(&gradient, &group.momentum)
            } else {
                group.momentum.shallow_clone()
            };
            let update = batched_quintic_orthogonalize(&update, orth, ns_steps);
            // Same cast the per-parameter tail applies. The group's momentum carries the
            // members' shared dtype, and `to_kind` on a matching dtype is free.
            let update = normuon_rescale(&update, &mut group.second_momentum, beta2)
                .to_kind(group.momentum.kind());
            for (slot, &position) in group.entries.iter().enumerate() {
                let entry = &entries_2d[position];
                let mut p = params[entry.idx].shallow_clone();
                // The signed delta is the row-learned controller's credit tensor, and
                // grouped entries never carry that controller.
                let _ = apply_normuon_update(
                    &mut p,
                    &update.select(0, slot as i64),
                    &scalars.normuon_step[position],
                    scalars.normuon_decay[position].as_ref(),
                    cautious,
                );
                batched[position] = true;
            }
        }

        for (position, entry) in entries_2d.iter_mut().enumerate() {
            if batched[position] || !step_enabled[entry.idx] {
                continue;
            }
            let grad = params[entry.idx].grad();
            if !grad.defined() {
                continue;
            }
            let (controlled_gradient, observation) = if primary {
                match entry.row_lr.as_mut() {
                    Some(controller) => {
                        let (gradient, observation) =
                            controller.scale_gradient(&grad, Some(primary_step));
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
                accumulator.push(observation);
            }
            let gradient = controlled_gradient.as_ref().unwrap_or(&grad);

            // The learned row scale acts on the raw gradient, before either optimizer
            // momentum or orthogonalization can mix its evidence across time or rows.
            scalars.normuon_lerp.lerp_(&mut entry.momentum, gradient);

            let update = if nesterov {
                scalars.nesterov.lerp(gradient, &entry.momentum)
            } else {
                entry.momentum.shallow_clone()
            };

            let update = normuon_transform(
                &update,
                entry.layout,
                &mut entry.second_momentum,
                beta2,
                orth,
                ns_steps,
            );

            let mut p = params[entry.idx].shallow_clone();
            let update = update.to_kind(p.kind());
            let signed_delta = apply_normuon_update(
                &mut p,
                &update,
                &scalars.normuon_step[position],
                scalars.normuon_decay[position].as_ref(),
                cautious,
            );
            if primary {
                if let Some(controller) = entry.row_lr.as_mut() {
                    // This is theta_new - theta_old from the gradient update only. Decoupled
                    // decay above is intentionally absent from the next-step credit tensor.
                    controller
                        .previous_delta
                        .copy_(&signed_delta.to_kind(Kind::Float).nan_to_num(0.0, 0.0, 0.0));
                }
            }
        }
        if primary {
            *row_lr_metrics = accumulator.finish();
        }
    }

    fn step_all_adamw(&mut self, scalars: &StepScalars) {
        let eps = self.cfg.adamw_eps;
        let cautious = self.cfg.cautious_weight_decay;

        for position in 0..self.adamw_indices.len() {
            let idx = self.adamw_indices[position];
            if !self.step_enabled[idx] {
                continue;
            }
            let mut p = self.params[idx].shallow_clone();
            let grad = p.grad();
            if !grad.defined() {
                continue;
            }
            let (beta1, beta2) = self.adamw_settings[position].betas;
            let state = self
                .adamw_state
                .get_mut(&idx)
                .expect("advance_adamw_clocks allocates the moments this step will update");

            let _ = state.m.lerp_(&grad, 1.0 - beta1);
            let _ = state.v.lerp_(&grad.square(), 1.0 - beta2);

            let denom = scalars.adamw_inv_bc2_sqrt[position]
                .mul(&state.v.sqrt())
                .g_add_scalar(eps);
            // `step` carries the negated descent direction, i.e. `p += step`.
            let step = scalars.adamw_step[position].mul(&(&state.m / &denom));
            if let Some(decay) = scalars.adamw_decay[position].as_ref() {
                if cautious {
                    // The reference masks strictly on `(descent_update * p) > 0`;
                    // `step` is the negation of that update, hence `< 0`.
                    let keep = decay.mul(&(&step * &p).lt(0).to_kind(p.kind()));
                    let _ = p.g_sub_(&(&p * keep));
                } else {
                    decay.mul_(&mut p);
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

    /// Whether the primary optimizer body including AdamW has finished capture.
    pub(crate) fn with_adamw_step_graph_captured(&self) -> bool {
        matches!(&self.step_graphs, StepGraphState::Armed(graphs)
            if matches!(graphs.with_adamw.state, GraphSlotState::Captured))
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

    /// Schedule setters keep writing the host fields. Every device-resident mirror a
    /// captured graph reads is refreshed from those fields at the top of each step, in
    /// [`Self::resolve_step_scalars`], which is the only place that also knows the
    /// composite scalars no setter sees (the per-matrix effective rate, the decay
    /// factor, AdamW's bias corrections) — so one refresh covers all of them and none
    /// can be forgotten by a new setter.
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
                // Which parameters a step touches is host control flow a captured body
                // baked in, so the capture has to be redone.
                self.step_graphs = StepGraphState::Unarmed;
            }
        }
        matched
    }

    /// Serialize complete optimizer state, including AdamW gradients retained between cadence
    /// updates, to a named-tensor sidecar. Keying by variable name makes restoration robust to
    /// param ordering. Requires `new_named` (the unnamed path stores empty names and its keys
    /// would collide).
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
            "__adamw_pending_grads__".to_owned(),
            Tensor::from(if self.adamw_pending_grads { 1i64 } else { 0 }),
        ));
        if self.adamw_pending_grads {
            for &idx in &self.adamw_indices {
                let name = &self.names[idx];
                let grad = self.params[idx].grad();
                named.push((
                    format!("{name}.__adamw_pending_grad_defined"),
                    Tensor::from(if grad.defined() { 1i64 } else { 0 }),
                ));
                if grad.defined() {
                    named.push((
                        format!("{name}.__adamw_pending_grad"),
                        grad.to_device(Device::Cpu),
                    ));
                }
            }
        }
        named.push((
            "__muon_step_count__".to_owned(),
            Tensor::from(self.step_count),
        ));
        let refs: Vec<(&str, &Tensor)> = named.iter().map(|(n, t)| (n.as_str(), t)).collect();
        Tensor::save_multi(&refs, path)
            .with_context(|| format!("failed saving optimizer state {}", path.display()))
    }

    /// Restore optimizer state saved by [`Muon::save_state`], copying buffers and any retained
    /// AdamW gradients in place so device/dtype match the live params. Absent 2D buffers are an
    /// error (the checkpoint is incomplete); absent AdamW moment buffers leave that parameter
    /// lazily re-initialized on its next update.
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
            .context("optimizer state missing global step")?
            .int64_value(&[]);
        let adamw_pending_grads = loaded
            .get("__adamw_pending_grads__")
            .context("optimizer state missing AdamW accumulation state")?
            .int64_value(&[])
            != 0;
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
                            .with_context(|| {
                                format!("optimizer state missing row LR logit for {name}")
                            })?,
                    );
                    controller.adam_m.copy_(
                        loaded
                            .get(&format!("{name}.__row_lr_adam_m"))
                            .with_context(|| {
                                format!("optimizer state missing row LR m for {name}")
                            })?,
                    );
                    controller.adam_v.copy_(
                        loaded
                            .get(&format!("{name}.__row_lr_adam_v"))
                            .with_context(|| {
                                format!("optimizer state missing row LR v for {name}")
                            })?,
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
        // Restored AdamW moments are fresh allocations at fresh addresses, and a
        // captured graph reads addresses.
        self.step_graphs = StepGraphState::Unarmed;
        self.adamw_pending_grads = adamw_pending_grads;
        if adamw_pending_grads {
            for &idx in &self.adamw_indices {
                let name = &self.names[idx];
                let defined = loaded
                    .get(&format!("{name}.__adamw_pending_grad_defined"))
                    .with_context(|| {
                        format!("optimizer state missing pending-gradient marker for {name}")
                    })?
                    .int64_value(&[])
                    != 0;
                if !defined {
                    ensure!(
                        !self.params[idx].grad().defined(),
                        "cannot restore undefined pending AdamW gradient over an existing slot for {name}"
                    );
                    continue;
                }
                let saved = loaded
                    .get(&format!("{name}.__adamw_pending_grad"))
                    .with_context(|| {
                        format!("optimizer state missing pending AdamW gradient for {name}")
                    })?;
                let mut grad = self.params[idx].grad();
                if !grad.defined() {
                    // tch exposes no gradient-slot setter. Backpropagating from one scalar view
                    // allocates the ordinary dense leaf slot without reading the full parameter;
                    // the saved accumulation immediately replaces that synthetic unit gradient.
                    self.params[idx].flatten(0, -1).get(0).backward();
                    grad = self.params[idx].grad();
                }
                tch::no_grad(|| grad.copy_(saved));
            }
        } else {
            for &idx in &self.adamw_indices {
                let mut grad = self.params[idx].grad();
                if grad.defined() {
                    let _ = grad.zero_();
                }
            }
        }
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

    /// Whether the next backward must accumulate into AdamW gradients retained by the previous
    /// cadence-skipped primary step.
    pub fn adamw_accumulation_pending(&self) -> bool {
        self.adamw_pending_grads
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
        expected_keys.insert("__adamw_pending_grads__".to_owned());
        let global_step = loaded
            .get("__muon_step_count__")
            .context("optimizer state missing global step")?;
        ensure!(
            global_step.numel() == 1 && global_step.int64_value(&[]) == expected_step,
            "optimizer global step disagrees with checkpoint metadata"
        );
        let pending = loaded
            .get("__adamw_pending_grads__")
            .context("optimizer state missing AdamW accumulation state")?;
        ensure!(
            pending.numel() == 1,
            "optimizer AdamW accumulation state is not scalar"
        );
        let pending = pending.int64_value(&[]);
        ensure!(
            pending == 0 || pending == 1,
            "optimizer AdamW accumulation state must be 0 or 1"
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
        if pending != 0 {
            for &idx in &self.adamw_indices {
                let name = &self.names[idx];
                let marker_key = format!("{name}.__adamw_pending_grad_defined");
                let marker = loaded.get(&marker_key).with_context(|| {
                    format!("optimizer state missing pending-gradient marker for {name}")
                })?;
                ensure!(
                    marker.numel() == 1 && matches!(marker.int64_value(&[]), 0 | 1),
                    "optimizer pending-gradient marker must be 0 or 1 for {name}"
                );
                expected_keys.insert(marker_key);
                if marker.int64_value(&[]) != 0 {
                    let key = format!("{name}.__adamw_pending_grad");
                    let grad = loaded.get(&key).with_context(|| {
                        format!("optimizer state missing pending AdamW gradient for {name}")
                    })?;
                    ensure!(
                        grad.size() == self.params[idx].size()
                            && grad.kind() == self.params[idx].kind(),
                        "optimizer pending AdamW gradient schema mismatch for {name}"
                    );
                    expected_keys.insert(key);
                }
            }
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

    /// Test-only: drop the batched groups so every 2-D parameter takes the
    /// per-parameter path. The members' momentum and second-moment views keep the
    /// stacked storage alive, so this is the SAME state seen through the other path,
    /// which is what makes the two paths directly comparable.
    #[cfg(test)]
    fn ungroup(&mut self) {
        self.groups_2d.clear();
    }
}

#[cfg(test)]
mod tests {
    use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

    use super::{
        attention_ortho_layout, batched_newtonschulz5, is_attention_output_projection_name,
        is_cross_attention_projection_name, is_self_attention_projection_name, newtonschulz5,
        normuon_reduce_dim, normuon_rescale, normuon_transform, ortho_aspect_scale,
        orthogonalize_update, quintic_orthogonalize, second_momentum_shape, CudaGraph,
        GraphSlotState, Muon, MuonConfig, OrthoLayout, Orthogonalizer, StepGraphState, StepKind,
        StepScalar, StepScalarPack,
    };
    use crate::torch::test_rng;
    use crate::torch::train::smd_idbd::PretrainOptimizer;

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
    fn recovery_restores_the_pending_adamw_accumulation_before_the_next_update() {
        let state_path = std::env::temp_dir().join(format!(
            "muon-pending-adamw-{}-{}.ot",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let config = || MuonConfig {
            use_muon_for_2d: false,
            adamw_lr: 0.1,
            adamw_betas: (0.0, 0.0),
            adamw_eps: 0.0,
            adamw_wd: 0.0,
            adamw_every: 2,
            quiet: true,
            ..MuonConfig::default()
        };

        let uninterrupted = Tensor::ones([1], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let mut optimizer = Muon::new_named(
            &[("bias".to_owned(), uninterrupted.shallow_clone())],
            config(),
        );
        (&uninterrupted * 2.0).sum(Kind::Float).backward();
        optimizer.step(StepKind::Primary);
        assert!(optimizer.adamw_accumulation_pending());
        optimizer
            .save_state(&state_path)
            .expect("save pending state");

        let recovered = Tensor::ones([1], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let mut recovered_optimizer =
            Muon::new_named(&[("bias".to_owned(), recovered.shallow_clone())], config());
        recovered_optimizer
            .validate_state_strict(&state_path, &[], 1)
            .expect("pending state validates");
        recovered_optimizer
            .load_state_strict(&state_path, &[])
            .expect("pending state restores");
        assert!(recovered_optimizer.adamw_accumulation_pending());

        optimizer.zero_grad();
        (&uninterrupted * -1.0).sum(Kind::Float).backward();
        optimizer.step(StepKind::Primary);
        recovered_optimizer.zero_grad();
        (&recovered * -1.0).sum(Kind::Float).backward();
        recovered_optimizer.step(StepKind::Primary);

        assert_eq!(
            uninterrupted.double_value(&[]),
            recovered.double_value(&[]),
            "the recovered update must consume the same two accumulated gradients"
        );
        assert!(
            (recovered.double_value(&[]) - 0.9).abs() < 1e-7,
            "a next-batch-only update would move in the opposite direction"
        );
        std::fs::remove_file(state_path).expect("remove optimizer fixture");
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

    /// The production NorMuon recipe's decay semantics, so a grouping test exercises
    /// the whole per-parameter tail and not just the orthogonalizer.
    fn grouping_config() -> MuonConfig {
        MuonConfig {
            lr: 0.02,
            momentum: 0.9,
            beta2: 0.9,
            weight_decay: 1.2,
            quadratic_lr_weight_decay: true,
            cautious_weight_decay: true,
            orthogonalizer: Orthogonalizer::PolarExpress5,
            quiet: true,
            ..MuonConfig::default()
        }
    }

    /// Drive `steps` primary updates over freshly cloned parameters, with a fixed
    /// per-parameter gradient, and return the final parameters.
    fn run_grouped_or_not(
        initial: &[Tensor],
        gradients: &[Tensor],
        cfg: MuonConfig,
        lr_scale: Option<(&str, f64)>,
        grouped: bool,
        steps: usize,
    ) -> Vec<Tensor> {
        let named: Vec<(String, Tensor)> = initial
            .iter()
            .enumerate()
            .map(|(index, weight)| (format!("w{index}"), weight.copy().set_requires_grad(true)))
            .collect();
        let mut optimizer = Muon::new_named(&named, cfg);
        if let Some((name, scale)) = lr_scale {
            assert_eq!(optimizer.set_named_lr_scale(&[name], scale), 1);
        }
        if !grouped {
            optimizer.ungroup();
        }
        for _ in 0..steps {
            optimizer.zero_grad();
            let loss = named.iter().zip(gradients).fold(
                Tensor::zeros([], (Kind::Float, Device::Cpu)),
                |accumulated, ((_, weight), gradient)| {
                    accumulated + (weight * gradient).sum(Kind::Float)
                },
            );
            loss.backward();
            optimizer.step(StepKind::Primary);
        }
        named
            .iter()
            .map(|(_, weight)| weight.detach().copy())
            .collect()
    }

    /// The batched group path and the per-parameter path are two ways of advancing one
    /// state, so they have to agree. This is the whole safety claim of grouping: any
    /// member can fall out of its group on any step — a missing gradient, a disabled
    /// name — and continue on the per-parameter path.
    #[test]
    fn batched_groups_agree_with_the_per_parameter_path() {
        let _torch_rng_guard = test_rng::exclusive();
        tch::manual_seed(4242);
        let device = Device::Cpu;
        let (initial, gradients) = tch::no_grad(|| {
            let initial: Vec<Tensor> = (0..3)
                .map(|_| Tensor::randn([6, 4], (Kind::Float, device)))
                .collect();
            let gradients: Vec<Tensor> = (0..3)
                .map(|_| Tensor::randn([6, 4], (Kind::Float, device)))
                .collect();
            (initial, gradients)
        });

        let grouped = run_grouped_or_not(
            &initial,
            &gradients,
            grouping_config(),
            Some(("w1", 4.0)),
            true,
            3,
        );
        let separate = run_grouped_or_not(
            &initial,
            &gradients,
            grouping_config(),
            Some(("w1", 4.0)),
            false,
            3,
        );

        for (index, ((batched, single), start)) in
            grouped.iter().zip(&separate).zip(&initial).enumerate()
        {
            let moved = (batched - start).abs().max().double_value(&[]);
            let diff = (batched - single).abs().max().double_value(&[]);
            assert!(
                moved > 1e-4,
                "w{index} barely moved ({moved:.3e}); the comparison would be vacuous"
            );
            // The two paths issue different kernels over the same math: bmm vs mm, and a
            // per-slot vs whole-tensor Frobenius reduction. That is a reassociation of a
            // bf16 iteration, so a few times bf16 epsilon of the update. It is NOT room for
            // a different prescale divisor: a 0.2% divisor error lands at ~1e-3 here, which
            // is what the old 3e-2 bound was wide enough to admit.
            assert!(
                diff < 3e-4 * moved,
                "w{index} diverged between the batched and the per-parameter path: \
                 diff={diff:.3e}, update={moved:.3e}"
            );
        }
    }

    /// A group must apply each member ITS OWN learning-rate scale. Sharing one scalar
    /// across a group is the numerical bug batching invites, and it hides in every
    /// aggregate: two members with identical parameters and identical gradients differ
    /// only through the scale, so the ratio of their updates IS the scale ratio.
    ///
    /// The parameters start at zero and the decay is off, so `p + delta` is exact and
    /// `4` is a power of two: the 4x member's update is bit-for-bit four times the
    /// other's, and the assertion needs no tolerance at all.
    #[test]
    fn a_group_applies_each_member_its_own_learning_rate_scale() {
        let _torch_rng_guard = test_rng::exclusive();
        tch::manual_seed(99);
        let device = Device::Cpu;
        let (initial, gradients) = tch::no_grad(|| {
            let gradient = Tensor::randn([6, 4], (Kind::Float, device));
            (
                vec![
                    Tensor::zeros([6, 4], (Kind::Float, device)),
                    Tensor::zeros([6, 4], (Kind::Float, device)),
                ],
                vec![gradient.copy(), gradient.copy()],
            )
        });
        let cfg = MuonConfig {
            weight_decay: 0.0,
            ..grouping_config()
        };

        let final_weights =
            run_grouped_or_not(&initial, &gradients, cfg, Some(("w1", 4.0)), true, 1);
        let plain = &final_weights[0] - &initial[0];
        let scaled = &final_weights[1] - &initial[1];
        let moved = plain.abs().max().double_value(&[]);
        assert!(
            moved > 1e-4,
            "the shared update is degenerate ({moved:.3e})"
        );
        let error = (&scaled - plain * 4.0).abs().max().double_value(&[]);
        assert!(
            error < 1e-12,
            "the 4x member did not take 4x the step inside its group: error={error:.3e}"
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
            let update = normuon_transform(
                &update,
                OrthoLayout::Matrix,
                &mut second,
                0.9,
                Orthogonalizer::NewtonSchulz5,
                5,
            );
            let aspect = ortho_aspect_scale(&[8, 4], OrthoLayout::Matrix);
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
        let update = tch::no_grad(|| {
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
        let mut on = Muon::new_named(
            &[("w".to_owned(), on_param.shallow_clone())],
            controller_config(),
        );

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
        assert_eq!(
            first.alpha_mean, 1.0,
            "the logit update must take effect next step"
        );
        assert!(first.evidence_mean.abs() < 1e-6);
        assert!((first.evidence_std - 1.0).abs() < 1e-6);
        let logit = &optimizer.entries_2d[0].row_lr.as_ref().unwrap().logit;
        assert!(
            logit.double_value(&[0, 0]) > 0.0,
            "productive row must speed up"
        );
        assert!(
            logit.double_value(&[1, 0]) < 0.0,
            "harmful row must slow down"
        );

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
        let mut off = Muon::new_named(
            &[("head".to_owned(), off_param.shallow_clone())],
            cfg(false),
        );
        let mut on = Muon::new_named(&[("head".to_owned(), on_param.shallow_clone())], cfg(true));
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
        let parameter = Tensor::from_slice(&[0.5f32, -0.25, 0.75, -1.0])
            .reshape([2, 2])
            .set_requires_grad(true);
        let before = parameter.copy();
        let mut cfg = controller_config();
        cfg.weight_decay = 0.4;
        let decay = cfg.weight_decay * cfg.lr;
        let mut optimizer = Muon::new_named(&[("w".to_owned(), parameter.shallow_clone())], cfg);
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
        assert!(
            error < 1e-6,
            "credit delta included decay or missed the applied update: {error}"
        );
    }

    /// Every optional term one step can carry, so every slot the pack allocates is
    /// actually read: decoupled decay on both branches, the AdamW cadence that gives the
    /// step its two capturable shapes, and Nesterov.
    ///
    /// Capture-eligible, like the pretraining optimizer and unlike the default, so that
    /// the policy tests below reach the checks they are about instead of stopping at
    /// eligibility.
    fn schedule_config() -> MuonConfig {
        MuonConfig {
            lr: 0.02,
            momentum: 0.9,
            weight_decay: 0.1,
            adamw_lr: 0.005,
            adamw_wd: 0.2,
            adamw_every: 2,
            quiet: true,
            capture_step_graphs: true,
            ..MuonConfig::default()
        }
    }

    /// Two same-shape matrices, which batch into one group; one odd-shaped matrix, which
    /// takes the per-parameter tail; two vectors, which route to AdamW. Deterministic, so
    /// two arms built from separate calls start bit-identical without touching the RNG.
    fn schedule_params(device: Device) -> Vec<(String, Tensor)> {
        let make = |name: &str, dims: &[i64], phase: f64| {
            let count: i64 = dims.iter().product();
            let values = Tensor::arange(count, (Kind::Float, device)) * 0.017 + phase;
            (
                name.to_owned(),
                (values.sin() * 0.5).reshape(dims).set_requires_grad(true),
            )
        };
        vec![
            make("block.0.w", &[8, 8], 0.0),
            make("block.1.w", &[8, 8], 1.0),
            make("head.w", &[6, 4], 2.0),
            make("head.bias", &[6], 3.0),
            make("norm.gain", &[8], 4.0),
        ]
    }

    /// A device-resident pack laid out for this optimizer, exactly as arming builds one.
    fn schedule_pack(optimizer: &Muon) -> StepScalarPack {
        let reference = &optimizer.params[optimizer.entries_2d[0].idx];
        StepScalarPack::new(
            optimizer.entries_2d.len(),
            optimizer.adamw_indices.len(),
            reference.kind(),
            reference.device(),
        )
    }

    /// Read one packed slot. Panics on a host scalar: the pack path handing one back
    /// would be a silently frozen kernel argument, which is the failure being guarded.
    fn slot_value(scalar: &StepScalar) -> f64 {
        match scalar {
            StepScalar::Device(tensor) => tensor.double_value(&[]),
            StepScalar::Host(value) => panic!("expected a device-resident slot, got host {value}"),
        }
    }

    /// The pack stages in `f64` and narrows to the parameter dtype, so a slot read is
    /// fp32-exact at best.
    fn assert_slot(actual: f64, expected: f64, what: &str) {
        let tolerance = 1e-6 * expected.abs().max(1e-3);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{what}: {actual} != {expected}"
        );
    }

    /// Every capture-policy assertion below describes the default path.
    fn require_default_graph_policy(optimizer: &Muon) {
        assert!(
            !optimizer.step_graphs_opted_out(),
            "unset PRETRAIN_CUDA_GRAPHS to run the capture-policy tests"
        );
    }

    /// One primary step driven through the device-resident schedule mirrors instead of
    /// `Scalar` arguments: the exact sequence [`Muon::try_graph_step`] performs, minus
    /// the capture, which is what makes the `StepScalar::Device` arithmetic reachable on
    /// a CPU-only runner.
    fn packed_primary_step(optimizer: &mut Muon, pack: &mut StepScalarPack) {
        tch::no_grad(|| {
            let do_adamw = optimizer.begin_step(true);
            let scalars = optimizer.publish_step_scalars(pack);
            optimizer.run_step_body(&scalars, do_adamw);
            optimizer.adamw_pending_grads = !do_adamw;
        });
    }

    /// A schedule that moves every step, which is the point: a body that froze its
    /// scalars at capture would keep applying the capture step's values.
    fn apply_schedule(optimizer: &mut Muon, step: usize) {
        let t = step as f64;
        optimizer.set_lr(0.02 * 0.85f64.powf(t));
        optimizer.set_momentum(0.8 + 0.03 * (t % 4.0));
        optimizer.set_adamw_lr(0.005 * (1.0 + 0.2 * t));
    }

    /// Parameter-dependent gradients: identical across arms at equal parameters, and
    /// coupled to the parameter so a divergence compounds instead of cancelling.
    ///
    /// A parameter matching `excluded` is left out of the loss entirely, so autograd
    /// never defines its gradient and the step skips it. That is the only way to move the
    /// stepped-parameter set, because `zero_grad` zeroes in place and never undefines.
    fn backward_schedule_loss(named: &[(String, Tensor)], step: usize, excluded: Option<&str>) {
        let device = named[0].1.device();
        let mut loss = Tensor::zeros([], (Kind::Float, device));
        for (position, (name, parameter)) in named.iter().enumerate() {
            if excluded.is_some_and(|needle| name.contains(needle)) {
                continue;
            }
            let weight = 0.3 + 0.1 * position as f64 + 0.05 * step as f64;
            loss = loss + parameter.square().sum(Kind::Float) * weight;
        }
        loss.backward();
    }

    /// Largest relative parameter disagreement between two arms.
    fn arm_deviation(left: &[(String, Tensor)], right: &[(String, Tensor)]) -> f64 {
        left.iter()
            .zip(right)
            .map(|((_, a), (_, b))| {
                (a - b).abs().max().double_value(&[]) / a.abs().max().double_value(&[]).max(1e-6)
            })
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn a_pack_refresh_moves_the_value_the_update_reads() {
        let named = schedule_params(Device::Cpu);
        let mut optimizer = Muon::new_named(&named, schedule_config());
        let mut pack = schedule_pack(&optimizer);

        // The handles a capture would have baked into its kernel arguments.
        let captured = optimizer.publish_step_scalars(&mut pack);
        assert_slot(
            slot_value(&captured.normuon_lerp),
            1.0 - 0.9,
            "1 - momentum",
        );
        assert_slot(slot_value(&captured.nesterov), 0.9, "momentum");

        optimizer.set_lr(0.05);
        optimizer.set_momentum(0.5);
        optimizer.set_adamw_lr(0.001);
        assert_slot(
            slot_value(&captured.nesterov),
            0.9,
            "a host setter alone must not move a device slot",
        );
        let _ = optimizer.resolve_step_scalars(Some(&mut pack));
        assert_slot(
            slot_value(&captured.nesterov),
            0.9,
            "resolving only stages the host buffer; upload publishes it",
        );
        pack.upload();

        // The handles taken before the setters now read the new schedule, because they
        // are views into the slots the refresh rewrote. That is the whole mechanism.
        assert_slot(slot_value(&captured.nesterov), 0.5, "refreshed momentum");
        assert_slot(
            slot_value(&captured.normuon_lerp),
            0.5,
            "refreshed 1 - momentum",
        );
        for (position, entry) in optimizer.entries_2d.iter().enumerate() {
            assert_slot(
                slot_value(&captured.normuon_step[position]),
                -(0.05 * optimizer.lr_scales[entry.idx] * entry.aspect_scale),
                "refreshed NorMuon step",
            );
        }
        for position in 0..optimizer.adamw_indices.len() {
            // The moment clocks have not advanced, so the correction count clamps to one.
            let (beta1, _) = optimizer.adamw_settings[position].betas;
            assert_slot(
                slot_value(&captured.adamw_step[position]),
                -0.001 / (1.0 - beta1),
                "refreshed AdamW step",
            );
        }
    }

    #[test]
    fn the_pretrain_schedule_setters_all_reach_the_refreshed_slots() {
        let named = schedule_params(Device::Cpu);
        let optimizer = Muon::new_named(&named, schedule_config());
        let mut pack = schedule_pack(&optimizer);
        let captured = optimizer.publish_step_scalars(&mut pack);

        // The pretrainer drives this wrapper, never the Muon setters directly.
        let mut wrapped = PretrainOptimizer::production(optimizer);
        wrapped.set_learning_rates(0.11, 0.013, 0.0);
        wrapped.set_momentum(0.4);
        let optimizer = wrapped.production_ref().expect("the production arm");
        let _ = optimizer.publish_step_scalars(&mut pack);

        assert_slot(slot_value(&captured.nesterov), 0.4, "set_momentum");
        assert_slot(slot_value(&captured.normuon_lerp), 0.6, "set_momentum");
        let entry = &optimizer.entries_2d[0];
        assert_slot(
            slot_value(&captured.normuon_step[0]),
            -(0.11 * optimizer.lr_scales[entry.idx] * entry.aspect_scale),
            "set_learning_rates NorMuon rate",
        );
        let (beta1, beta2) = optimizer.adamw_settings[0].betas;
        assert_slot(
            slot_value(&captured.adamw_step[0]),
            -0.013 / (1.0 - beta1),
            "set_learning_rates AdamW rate",
        );
        assert_slot(
            slot_value(&captured.adamw_inv_bc2_sqrt[0]),
            1.0 / (1.0 - beta2).sqrt(),
            "AdamW second-moment bias correction",
        );
    }

    #[test]
    fn a_named_lr_scale_moves_only_its_own_parameters_slot() {
        let named = schedule_params(Device::Cpu);
        let mut optimizer = Muon::new_named(&named, schedule_config());
        let mut pack = schedule_pack(&optimizer);
        let captured = optimizer.publish_step_scalars(&mut pack);
        let normuon_before: Vec<f64> = captured.normuon_step.iter().map(slot_value).collect();
        let adamw_before: Vec<f64> = captured.adamw_step.iter().map(slot_value).collect();
        assert!(normuon_before.len() > 1 && adamw_before.len() > 1);

        assert_eq!(optimizer.set_named_lr_scale(&["block.1"], 0.25), 1);
        assert_eq!(optimizer.set_named_lr_scale(&["head.bias"], 4.0), 1);
        let _ = optimizer.publish_step_scalars(&mut pack);

        let scaled_matrix = optimizer
            .entries_2d
            .iter()
            .position(|entry| optimizer.names[entry.idx] == "block.1.w")
            .expect("block.1.w routes to NorMuon");
        for position in 0..normuon_before.len() {
            let factor = if position == scaled_matrix { 0.25 } else { 1.0 };
            assert_slot(
                slot_value(&captured.normuon_step[position]),
                normuon_before[position] * factor,
                "per-parameter NorMuon scale",
            );
        }
        let scaled_vector = optimizer
            .adamw_indices
            .iter()
            .position(|&idx| optimizer.names[idx] == "head.bias")
            .expect("head.bias routes to AdamW");
        for position in 0..adamw_before.len() {
            let factor = if position == scaled_vector { 4.0 } else { 1.0 };
            assert_slot(
                slot_value(&captured.adamw_step[position]),
                adamw_before[position] * factor,
                "per-parameter AdamW scale",
            );
        }
    }

    #[test]
    fn device_resident_scalars_reproduce_the_host_scalar_step_under_a_moving_schedule() {
        for cautious in [false, true] {
            let config = || MuonConfig {
                cautious_weight_decay: cautious,
                ..schedule_config()
            };
            let host_named = schedule_params(Device::Cpu);
            let packed_named = schedule_params(Device::Cpu);
            let mut host = Muon::new_named(&host_named, config());
            let mut packed = Muon::new_named(&packed_named, config());
            let mut pack = schedule_pack(&packed);
            for step in 0..12 {
                apply_schedule(&mut host, step);
                apply_schedule(&mut packed, step);
                backward_schedule_loss(&host_named, step, None);
                backward_schedule_loss(&packed_named, step, None);
                host.step(StepKind::Primary);
                packed_primary_step(&mut packed, &mut pack);
                host.zero_grad();
                packed.zero_grad();
            }
            // The default-on policy has to limit itself: CPU parameters cannot be
            // captured, so the first primary step must have given up permanently.
            assert!(
                matches!(host.step_graphs, StepGraphState::Disabled),
                "CPU parameters must disable capture rather than attempt it"
            );
            // Observed at exactly zero on CPU fp32 over both decay masks: the pack
            // stages in `f64` and narrows on copy, which is the same rounding ATen
            // applies to a `Scalar`. The tolerance is not tight against that
            // observation, only against a frozen or misrouted slot, which is O(1).
            let deviation = arm_deviation(&host_named, &packed_named);
            println!(
                "device-scalar vs host-scalar deviation (cautious {cautious}): {deviation:.3e}"
            );
            assert!(
                deviation < 1e-5,
                "the host-scalar and device-scalar steps diverged (cautious decay \
                 {cautious}): {deviation}"
            );
        }
    }

    #[test]
    fn capture_declines_every_case_it_cannot_serve() {
        let cpu = Muon::new_named(&schedule_params(Device::Cpu), schedule_config());
        require_default_graph_policy(&cpu);
        let blocker = cpu
            .step_graph_blocker()
            .expect("CPU parameters cannot be captured");
        assert!(
            blocker.inherent,
            "a CPU-only process has nothing to act on: {}",
            blocker.reason
        );
        if CudaGraph::is_available() {
            assert_eq!(blocker.reason, "parameters live on Cpu");
        }

        let vectors = [(
            "a.bias".to_owned(),
            Tensor::zeros([4], (Kind::Float, Device::Cpu)),
        )];
        let unrouted = Muon::new_named(&vectors, schedule_config());
        assert!(unrouted.entries_2d.is_empty());
        let blocker = unrouted
            .step_graph_blocker()
            .expect("a model with no matrices cannot be captured");
        assert!(blocker.inherent, "routing is not an env-var decision");
        if CudaGraph::is_available() {
            assert_eq!(blocker.reason, "no NorMuon-routed parameters");
        }

        // Ineligibility is the default and is checked before everything else, so PPO, the
        // planner and every caller that says nothing decline capture on any machine and
        // without consulting the environment.
        let opted_out = Muon::new_named(
            &schedule_params(Device::Cpu),
            MuonConfig {
                capture_step_graphs: false,
                ..schedule_config()
            },
        );
        let blocker = opted_out
            .step_graph_blocker()
            .expect("an optimizer that opted out cannot be captured");
        assert_eq!(blocker.reason, "capture is not enabled for this optimizer");
        assert!(blocker.inherent, "the caller already made this choice");

        let by_default = Muon::new_named(&schedule_params(Device::Cpu), MuonConfig::default());
        assert_eq!(
            by_default
                .step_graph_blocker()
                .expect("the default config is not capture-eligible")
                .reason,
            "capture is not enabled for this optimizer",
            "capture must be opt-in per optimizer so a new caller cannot inherit it"
        );
    }

    /// The one test that would catch a schedule frozen inside a replay.
    ///
    /// `#[ignore]`d because it needs a CUDA device and because capture tolerates no
    /// other CUDA work in the process while it records; [`test_rng::exclusive`] is the
    /// suite's only process-wide serialization point, so it doubles as that guard.
    /// Returns without asserting, rather than failing, when there is no device.
    #[test]
    #[ignore = "captures real CUDA graphs: needs a GPU and exclusive use of the process"]
    fn a_captured_step_tracks_a_changing_schedule_as_closely_as_the_eager_step() {
        let _torch_rng_guard = test_rng::exclusive();
        if !tch::Cuda::is_available() || !CudaGraph::is_available() {
            return;
        }
        // `adamw_every` is 2, so the NorMuon-only body runs on odd step counts and the
        // NorMuon+AdamW body on even ones. Each warms up for `GRAPH_WARMUP_STEPS` of its
        // own turns, so they capture on step counts 7 and 8 and replay after that.
        const STEPS: usize = 16;
        const CAPTURED_AT: usize = 6;

        let device = Device::Cuda(0);
        let graphed_named = schedule_params(device);
        let eager_named = schedule_params(device);
        let frozen_named = schedule_params(device);
        let mut graphed = Muon::new_named(&graphed_named, schedule_config());
        let mut eager = Muon::new_named(&eager_named, schedule_config());
        let mut frozen = Muon::new_named(&frozen_named, schedule_config());
        require_default_graph_policy(&graphed);
        // Both comparison arms take the fallback path outright, so no second capture and
        // no environment reading enters the measurement.
        eager.step_graphs = StepGraphState::Disabled;
        frozen.step_graphs = StepGraphState::Disabled;

        for step in 0..STEPS {
            apply_schedule(&mut graphed, step);
            apply_schedule(&mut eager, step);
            // What a frozen schedule would look like: the values held at capture, kept
            // for every later step. This arm is the test's own sensitivity yardstick.
            apply_schedule(&mut frozen, step.min(CAPTURED_AT));
            for named in [&graphed_named, &eager_named, &frozen_named] {
                backward_schedule_loss(named, step, None);
            }
            graphed.step(StepKind::Primary);
            eager.step(StepKind::Primary);
            frozen.step(StepKind::Primary);
            graphed.zero_grad();
            eager.zero_grad();
            frozen.zero_grad();
        }

        // A vacuous pass would be worse than a failure, so require that both bodies
        // really were recorded and replayed.
        match &graphed.step_graphs {
            StepGraphState::Armed(graphs) => {
                assert!(
                    matches!(graphs.normuon_only.state, GraphSlotState::Captured),
                    "the NorMuon-only body never captured"
                );
                assert!(
                    matches!(graphs.with_adamw.state, GraphSlotState::Captured),
                    "the NorMuon+AdamW body never captured"
                );
            }
            _ => panic!("the graph path never armed on a CUDA device"),
        }

        let against_eager = arm_deviation(&graphed_named, &eager_named);
        let against_frozen = arm_deviation(&graphed_named, &frozen_named);
        println!(
            "captured vs eager {against_eager:.3e}, captured vs frozen schedule \
             {against_frozen:.3e}"
        );
        // Observed on an RTX 5090: 0.0 against eager, i.e. bit-identical, against 6.6e-1
        // for a schedule frozen at capture. The tolerance stays loose because the graph's
        // warmup and capture run on a side stream, where cuBLAS is free to pick a
        // different algorithm for a large enough matrix; what the test pins is the
        // seven-orders-of-magnitude gap between tracking the schedule and freezing it.
        assert!(
            against_eager < 1e-4,
            "the captured step and the eager step disagree by {against_eager}"
        );
        assert!(
            against_frozen > 100.0 * against_eager.max(1e-9),
            "this test cannot distinguish a live schedule from a frozen one: live \
             {against_eager}, frozen {against_frozen}"
        );
    }

    /// The default-on path is self-limiting: a configuration capture cannot serve has to
    /// degrade to the eager step rather than break the run.
    #[test]
    #[ignore = "needs a CUDA device to reach the checks a CPU device short-circuits"]
    fn the_row_learned_lr_controller_blocks_capture_and_still_steps() {
        let _torch_rng_guard = test_rng::exclusive();
        if !tch::Cuda::is_available() || !CudaGraph::is_available() {
            return;
        }
        let named = schedule_params(Device::Cuda(0));
        let mut optimizer = Muon::new_named(
            &named,
            MuonConfig {
                row_learned_lr: true,
                ..schedule_config()
            },
        );
        require_default_graph_policy(&optimizer);
        assert!(!optimizer.entries_2d.is_empty());
        let blocker = optimizer
            .step_graph_blocker()
            .expect("the controller blocks capture");
        assert_eq!(
            blocker.reason,
            "the row-learned learning-rate controller is enabled"
        );
        assert!(
            !blocker.inherent,
            "the caller chose the controller and can unchoose it, so it gets told"
        );

        let before = named[0].1.copy();
        backward_schedule_loss(&named, 0, None);
        optimizer.step(StepKind::Primary);
        assert!(matches!(optimizer.step_graphs, StepGraphState::Disabled));
        assert!(
            (&named[0].1 - &before).abs().max().double_value(&[]) > 0.0,
            "the fallback step must still update the parameter"
        );
    }

    #[test]
    fn the_step_record_tracks_gradients_the_enable_mask_and_gradient_identity() {
        let named = schedule_params(Device::Cpu);
        let mut optimizer = Muon::new_named(&named, schedule_config());
        let idle = vec![None; named.len()];
        assert_eq!(optimizer.step_participation(), idle);
        assert!(optimizer.participation_matches(&idle));

        // One parameter in the loss: only that one has a gradient, so only it steps, and
        // what it records is the address of the gradient it will read.
        named[0].1.square().sum(Kind::Float).backward();
        let recorded = optimizer.step_participation();
        assert_eq!(
            recorded[0],
            Some(named[0].1.grad().data_ptr() as usize),
            "a stepped parameter records the gradient buffer its kernels will read"
        );
        assert!(recorded[1..].iter().all(Option::is_none));
        assert!(optimizer.participation_matches(&recorded));
        assert!(
            !optimizer.participation_matches(&idle),
            "a gradient appearing must invalidate a set recorded without it"
        );
        assert!(
            !optimizer.participation_matches(&recorded[..named.len() - 1]),
            "a set of the wrong length must never match on a prefix"
        );

        // Accumulation and `zero_grad` both write through the existing buffer, so a
        // capture stays valid across them. That is what makes the address worth recording
        // rather than fatal to record.
        named[0].1.square().sum(Kind::Float).backward();
        optimizer.zero_grad();
        assert!(
            optimizer.participation_matches(&recorded),
            "accumulating into and zeroing a gradient must not invalidate a capture"
        );
        assert!(
            !optimizer.participation_matches(&[Some(1usize); 0]),
            "an empty record must not match a live step"
        );

        // The enable mask removes a parameter that does have a gradient.
        backward_schedule_loss(&named, 0, None);
        assert!(optimizer.step_participation().iter().all(Option::is_some));
        assert_eq!(optimizer.set_named_step_enabled(&["block.0"], false), 1);
        let disabled = optimizer
            .names
            .iter()
            .position(|name| name == "block.0.w")
            .expect("block.0.w is registered");
        assert_eq!(optimizer.step_participation()[disabled], None);
    }

    /// A parameter that acquires a gradient AFTER capture must start moving.
    ///
    /// This is the hazard no blocker can rule out in advance: a parameter with no
    /// gradient contributes no kernels, so a body recorded while it was idle would keep
    /// skipping it for the rest of the run, silently. `#[ignore]`d and serialized for the
    /// same reasons as the agreement test above.
    #[test]
    #[ignore = "captures real CUDA graphs: needs a GPU and exclusive use of the process"]
    fn a_parameter_that_gains_a_gradient_after_capture_starts_stepping() {
        let _torch_rng_guard = test_rng::exclusive();
        if !tch::Cuda::is_available() || !CudaGraph::is_available() {
            return;
        }
        // Long enough for both bodies to capture and replay in each phase: three warmup
        // turns plus a capture plus replays, at one turn every other step.
        const PHASE: usize = 12;
        const IDLE: &str = "head.w";

        let graphed_named = schedule_params(Device::Cuda(0));
        let eager_named = schedule_params(Device::Cuda(0));
        let mut graphed = Muon::new_named(&graphed_named, schedule_config());
        let mut eager = Muon::new_named(&eager_named, schedule_config());
        require_default_graph_policy(&graphed);
        eager.step_graphs = StepGraphState::Disabled;
        let idle = graphed_named
            .iter()
            .position(|(name, _)| name == IDLE)
            .expect("head.w is registered");
        let initial = graphed_named[idle].1.copy();

        for step in 0..2 * PHASE {
            // The second phase brings the idle parameter into the loss for the first time.
            let excluded = (step < PHASE).then_some(IDLE);
            for (optimizer, named) in [(&mut graphed, &graphed_named), (&mut eager, &eager_named)] {
                apply_schedule(optimizer, step);
                backward_schedule_loss(named, step, excluded);
                optimizer.step(StepKind::Primary);
                optimizer.zero_grad();
            }
            if step + 1 == PHASE {
                assert!(
                    graphed_named[idle].1.equal(&initial),
                    "a parameter with no gradient must not be stepped at all"
                );
                assert!(matches!(
                    &graphed.step_graphs,
                    StepGraphState::Armed(graphs)
                        if matches!(graphs.normuon_only.state, GraphSlotState::Captured)
                ));
            }
        }

        // Without the participation check the replayed body would still be the one
        // recorded while this parameter was idle, and it would never have moved.
        let moved = (&graphed_named[idle].1 - &initial)
            .abs()
            .max()
            .double_value(&[]);
        let deviation = arm_deviation(&graphed_named, &eager_named);
        println!("recaptured parameter moved by {moved:.3e}, arms agree to {deviation:.3e}");
        // Observed on an RTX 5090: the parameter moved 6.0e-3 and the arms are again
        // bit-identical, against exactly 0.0 movement if the stale body had kept replaying.
        assert!(
            moved > 1e-6,
            "the captured body kept skipping a parameter that now has a gradient"
        );
        assert!(
            deviation < 1e-4,
            "the recaptured graph step and the eager step disagree by {deviation}"
        );
        match &graphed.step_graphs {
            StepGraphState::Armed(graphs) => {
                assert!(matches!(
                    graphs.normuon_only.state,
                    GraphSlotState::Captured
                ));
                assert!(matches!(graphs.with_adamw.state, GraphSlotState::Captured));
            }
            _ => panic!("the graph path did not recapture after the parameter set changed"),
        }
    }
}
