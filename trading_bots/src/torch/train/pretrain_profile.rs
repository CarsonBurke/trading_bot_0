//! Device-synchronized region profiler for the pretrain step.
//!
//! Enabled by `PRETRAIN_PROFILE=1`. Every region boundary synchronizes the training device,
//! so a region's share is a true attribution instead of an artifact of where the asynchronous
//! launch queue happened to back up. That makes a profiled run SLOWER than the run it
//! measures; the totals are for ranking work, never for reporting throughput.
//!
//! Disabled — the production default — a region costs one relaxed atomic load and nothing
//! else, so the instrumentation can stay on the hot path permanently.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use tch::{Cuda, Device};

static ENABLED: AtomicBool = AtomicBool::new(false);
static FINE: AtomicBool = AtomicBool::new(false);
static TRACE: AtomicBool = AtomicBool::new(false);
static DEVICE_INDEX: AtomicI64 = AtomicI64::new(-1);
static TOTALS: Mutex<Vec<(&'static str, f64, u64)>> = Mutex::new(Vec::new());

/// Arm the profiler from the environment and pin the device every boundary synchronizes on.
/// A no-op off CUDA: without a device to drain, a region boundary would measure launch queue
/// depth rather than work.
///
/// `PRETRAIN_PROFILE=1` opens the ~16 step-level regions, whose combined synchronization cost
/// is under a millisecond and does not visibly move the step. `PRETRAIN_PROFILE=2` also opens
/// the per-layer regions, which for a 10-layer trunk means roughly 860 extra device drains per
/// step and inflates `step.forward.trunk` by several milliseconds. Level 2 is for ranking the
/// trunk's internals against each other, never for reading the trunk's share of the step.
///
/// `PRETRAIN_PROFILE=3` additionally wraps one step in libtorch's own operator profiler. See
/// [`super::super::backward_probe`].
pub fn init(device: Device) {
    let level = std::env::var("PRETRAIN_PROFILE")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(0);
    let armed = matches!(device, Device::Cuda(_)) && level > 0;
    if let Device::Cuda(index) = device {
        DEVICE_INDEX.store(index as i64, Ordering::Relaxed);
    }
    TRACE.store(armed && level > 2, Ordering::Relaxed);
    ENABLED.store(armed, Ordering::Relaxed);
    FINE.store(armed && level > 1, Ordering::Relaxed);
    if armed {
        println!(
            "pretrain profile: level {level} region timing armed on {device:?}; every boundary \
             synchronizes, so step/s under PRETRAIN_PROFILE is not the deployed throughput{}",
            if level > 1 {
                ". Level 2 per-layer regions add ~860 drains/step and inflate the trunk"
            } else {
                ""
            }
        );
    }
}

#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

#[inline]
pub fn drain() {
    let index = DEVICE_INDEX.load(Ordering::Relaxed);
    if index >= 0 {
        Cuda::synchronize(index);
    }
}

/// Open a step-level region. `None` when profiling is off, which is what keeps the disabled
/// cost to the atomic load: no `Instant::now`, no synchronize, no lock.
#[inline]
pub fn region(name: &'static str) -> Option<Region> {
    open(name, ENABLED.load(Ordering::Relaxed))
}

/// Open a per-layer region. Silent below `PRETRAIN_PROFILE=2` because a region opened once per
/// layer per step costs more in synchronization than the work it measures.
///
/// READ THIS BEFORE QUOTING A LEVEL-2 NUMBER. At small batch, or while another tenant holds the
/// GPU, a boundary's device drain costs more than the region it brackets — a drain waits on the
/// other tenant's kernels too — and the totals then rank regions by BOUNDARY COUNT rather than
/// by work. The signature is a uniform cost per boundary: divide each region's total by its
/// call count before believing it, and if the quotients agree across regions you are reading
/// synchronization, not attribution. Level 2 ranks the trunk's internals only at production
/// batch on a quiet card. Per-kernel attribution that cannot be fooled this way needs
/// libtorch's own operator profiler; see [`super::super::backward_probe`].
///
/// AND READ THIS BEFORE PROFILING THE TRUNK ON THE CPU INSTEAD. CPU operator counts are not
/// CUDA kernel counts. A `CompositeImplicitAutograd` op decomposes on CPU, so autograd records
/// the decomposition and both the launch count and the saved-activation set look far worse than
/// they are; on CUDA the same call can collapse into one native fused node. `rms_norm` is
/// exactly this trap — its composite body calls `_fused_rms_norm`, which has a real CUDA kernel
/// and a real `AutogradCUDA` node, so on CPU it is 22 backward launches saving an fp32 upcast
/// and on CUDA it is one 27us kernel saving the bf16 input. Reading the dispatch table is not
/// enough either: `rms_norm` itself registers as a math kernel on CUDA. Only a CUDA measurement
/// settles it.
///
/// AND READ THIS BEFORE BENCHING A TRUNK INTERNAL IN ISOLATION. The trunk runs under
/// `autocast(bf16)`, and autocast rewrites op dtypes, so a bench built outside it measures a
/// different graph. Autocast's fp32 cast-policy ops — `softplus` and `exp` among them, while
/// `cos`, `mul` and `cat` are untouched — UPCAST their inputs and return fp32, which means
/// enabling autocast strictly ADDS traffic to any region those ops dominate. Outside autocast
/// the same call stays narrow and the bench reports a saving that production cannot deliver.
///
/// AND KNOW WHOSE TENSOR AUTOGRAD SAVES. It saves what the KERNEL received, not what the caller
/// passed. When an fp32-policy op widens its input, the fp32 copy is materialized inside the
/// call and retained for backward, so handing that op a narrow tensor to shrink the saved set
/// changes nothing: `pope_expand_qk` passing `kind` rather than `Float` into `softplus` measured
/// 0.0 MB saved and a bit-identical output. Confirm a saved-activation win by measuring
/// `torch.cuda.memory_allocated` across the forward, never by reading the source.
#[inline]
pub fn region_fine(name: &'static str) -> Option<Region> {
    open(name, FINE.load(Ordering::Relaxed))
}

#[inline]
fn open(name: &'static str, armed: bool) -> Option<Region> {
    if !armed {
        return None;
    }
    drain();
    Some(Region {
        name,
        started: Instant::now(),
    })
}

pub struct Region {
    name: &'static str,
    started: Instant,
}

impl Drop for Region {
    fn drop(&mut self) {
        drain();
        record(self.name, self.started.elapsed().as_secs_f64());
    }
}

/// Print every region's accumulated cost, widest first, and reset the accumulators.
///
/// `steps` is the number of optimizer steps the window covered, so the per-step column is
/// directly comparable against the wall clock of an unprofiled run. Dotted names nest: a
/// child's cost is also inside its parent's, so only siblings sum.
pub fn report(label: &str, steps: usize) {
    if !enabled() {
        return;
    }
    let mut totals = TOTALS.lock().expect("profile totals are not poisoned");
    if totals.is_empty() {
        return;
    }
    totals.sort_by(|left, right| right.1.total_cmp(&left.1));
    let root = totals
        .iter()
        .find(|(name, _, _)| *name == STEP)
        .map(|(_, secs, _)| *secs)
        .unwrap_or_else(|| totals.first().map(|(_, secs, _)| *secs).unwrap_or(0.0));
    println!("pretrain profile [{label}] over {steps} steps, share of `{STEP}`:");
    for (name, secs, count) in totals.iter() {
        println!(
            "  {name:<28} {:>9.2} ms/step  {:>6.1}%  ({count} calls)",
            1000.0 * secs / steps.max(1) as f64,
            if root > 0.0 {
                100.0 * secs / root
            } else {
                f64::NAN
            },
        );
    }
    totals.clear();
}

pub const STEP: &str = "step";
pub const DATA_DRAW: &str = "step.data.draw";
pub const DATA_BUILD: &str = "step.data.build";
pub const DATA_H2D: &str = "step.data.h2d";
pub const FORWARD: &str = "step.forward";
pub const FORWARD_BINS: &str = "step.forward.bins";
pub const FORWARD_TRUNK: &str = "step.forward.trunk";
pub const TRUNK_EMBED: &str = "step.forward.trunk.embed";
pub const TRUNK_POPE: &str = "step.forward.trunk.pope";
pub const TRUNK_ATTN: &str = "step.forward.trunk.attn";
pub const TRUNK_FFN: &str = "step.forward.trunk.ffn";
pub const TRUNK_NORM: &str = "step.forward.trunk.norm";
pub const FORWARD_HEAD: &str = "step.forward.head_nll";
pub const FORWARD_DIRECT: &str = "step.forward.direct";
pub const FORWARD_DYNAMICS: &str = "step.forward.dynamics";
pub const FORWARD_GROWTH: &str = "step.forward.growth";
pub const FORWARD_AUTOCORR: &str = "step.forward.autocorr";
pub const BACKWARD: &str = "step.backward";
pub const GRAD_NORM: &str = "step.grad_norm";
pub const METRIC_SYNC: &str = "step.metric_sync";
pub const OPTIMIZER: &str = "step.optimizer";
pub const ZERO_GRAD: &str = "step.zero_grad";

/// The evaluation path. `validate.*` are the passes a validation boundary runs, in the order
/// it runs them, and they sum to `validate`. `eval.*` are the KINDS of work inside one
/// `evaluate_impl` chunk loop; they are deliberately not nested under a `validate.*` parent
/// because the same chunk loop serves four call sites, and a per-kind total pooled across
/// them is what ranks the work. `eval.*` therefore sums to the sum of the `validate.*` passes
/// that run a chunk loop, not to any single one.
pub const VALIDATE: &str = "validate";
pub const VALIDATE_SHRINK: &str = "validate.shrink_fit";
pub const VALIDATE_DIAG: &str = "validate.diagnostic";
pub const VALIDATE_BENCH: &str = "validate.bench";
pub const VALIDATE_PROMOTION: &str = "validate.promotion_pass";
pub const VALIDATE_SCORES: &str = "validate.window_scores";
pub const VALIDATE_CONTEXT_BEST: &str = "validate.context_best";
pub const VALIDATE_CHECKPOINT: &str = "validate.checkpoint";
pub const VALIDATE_PROMOTE: &str = "validate.promote";
pub const VALIDATE_ROUNDTRIP: &str = "validate.promote.roundtrip";
pub const VALIDATE_SNAPSHOT: &str = "validate.snapshot";
pub const VALIDATE_REPORT: &str = "validate.reports";
pub const EVAL_DATA: &str = "eval.data";
pub const EVAL_BINS: &str = "eval.bins";
pub const EVAL_TRUNK: &str = "eval.trunk";
pub const EVAL_HEAD: &str = "eval.head";
pub const EVAL_TERMS: &str = "eval.terms";
pub const EVAL_CRPS: &str = "eval.crps";
pub const EVAL_PIT: &str = "eval.pit";
pub const EVAL_DIRECTION: &str = "eval.direction";
pub const EVAL_DECOMP: &str = "eval.decomp";
pub const EVAL_MARGINALS: &str = "eval.marginals";
pub const EVAL_TRADE: &str = "eval.trade_paths";
pub const EVAL_RANK: &str = "eval.rank";
pub const EVAL_HOST: &str = "eval.host_read";
pub const VALIDATE_ROLLOUT: &str = "validate.rollout";
pub const VALIDATE_DIRECT: &str = "validate.direct";
pub const DIRECT_DATA: &str = "direct.data";
pub const DIRECT_TRUNK: &str = "direct.trunk";
pub const DIRECT_SELECT: &str = "direct.select";
pub const DIRECT_SCORE: &str = "direct.score";

/// True when the per-layer regions are open, which is also the gate on the backward probe.
#[inline]
pub fn fine() -> bool {
    FINE.load(Ordering::Relaxed)
}

/// True at `PRETRAIN_PROFILE=3`, which additionally traces one step with libtorch's own
/// operator profiler.
#[inline]
pub fn trace() -> bool {
    TRACE.load(Ordering::Relaxed)
}

/// Charge `elapsed` seconds to `name`, for costs measured outside a [`Region`] guard.
///
/// The backward probe times intervals between autograd hooks rather than between the open and
/// close of a guard, and reports them through here so that one table ranks the whole step.
pub fn record(name: &'static str, elapsed: f64) {
    let mut totals = TOTALS.lock().expect("profile totals are not poisoned");
    match totals.iter_mut().find(|(name_of, _, _)| *name_of == name) {
        Some((_, secs, count)) => {
            *secs += elapsed;
            *count += 1;
        }
        None => totals.push((name, elapsed, 1)),
    }
}

/// Backward attribution. `step.backward.*` are intervals between autograd hooks on forward
/// boundary tensors, so consecutive names partition the backward wave the way consecutive
/// `step.forward.*` guards partition the forward. The four `step.backward.trunk.*` names are
/// consecutive and therefore sum to the trunk's whole backward; `step.backward` minus that sum
/// minus `step.backward.embed` is the heads.
pub const BACKWARD_ATTN: &str = "step.backward.trunk.attn";
pub const BACKWARD_FFN: &str = "step.backward.trunk.ffn";
pub const BACKWARD_POPE: &str = "step.backward.trunk.pope";
pub const BACKWARD_NORM: &str = "step.backward.trunk.norm";
pub const BACKWARD_EMBED: &str = "step.backward.embed";

/// FA4's Python re-entry, split by side. The forward names are host time inside
/// [`super::super::fa4`]'s call path; the backward names are host time inside the
/// `FlashAttnFunc` autograd node, charged from the node's own pre- and post-hooks. Neither
/// synchronizes: they are the cost of getting into and out of Python, not of the kernels.
pub const FA4_FORWARD_LOCK: &str = "step.forward.trunk.attn.fa4_lock";
pub const FA4_FORWARD_GIL: &str = "step.forward.trunk.attn.fa4_py.gil";
pub const FA4_FORWARD_PY: &str = "step.forward.trunk.attn.fa4_py";
pub const FA4_BACKWARD_LOCK: &str = "step.backward.trunk.attn.fa4_lock";
pub const FA4_BACKWARD_NODE: &str = "step.backward.trunk.attn.fa4_node";
