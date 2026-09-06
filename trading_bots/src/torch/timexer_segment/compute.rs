use crate::torch::cuda::{
    self,
    graph::{CudaGraph, CudaGraphPool},
};
use crate::torch::optim::muon::{Muon, MuonConfig, Orthogonalizer, StepKind};
use crate::torch::train::optimizer_glue::named_trainable_variables;
use anyhow::{ensure, Context, Result};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum OptimizerKind {
    #[default]
    PolarExpress,
    Adam,
}

/// modded-nanogpt `train_gpt.py` Adam / NorMuon bases (`:2065-2076`).
const NANOGPT_ADAMW_LR: f64 = 0.008;
const NANOGPT_MUON_LR: f64 = 0.023;
/// AdamW learning-rate multiplier for the learnable recipe scalars, from modded-nanogpt's
/// `"scalars"` group (`train_gpt.py:2030`, `lr_mul: 5.0`).
const NANOGPT_SCALAR_LR_MULTIPLIER: f64 = 5.0;
const NANOGPT_COOLDOWN_FRAC: f64 = 0.60;
const NANOGPT_COOLDOWN_FLOOR: f64 = 0.15;
const NANOGPT_MUON_WARMUP_STEPS: usize = 300;
const NANOGPT_MUON_COOLDOWN_STEPS: usize = 50;
const NANOGPT_MOMENTUM_MIN: f64 = 0.85;
const NANOGPT_MOMENTUM_MAX: f64 = 0.95;

impl OptimizerKind {
    pub fn default_learning_rate(self) -> f64 {
        match self {
            Self::PolarExpress => NANOGPT_ADAMW_LR,
            Self::Adam => 0.0001,
        }
    }

    pub fn muon_learning_rate(self, learning_rate: f64) -> Option<f64> {
        (self == Self::PolarExpress).then_some(learning_rate * (NANOGPT_MUON_LR / NANOGPT_ADAMW_LR))
    }

    pub fn recipe(self) -> &'static str {
        match self {
            Self::PolarExpress => {
                "normuon-pe5-block-hidden2D-only;patch-covariates-norm-bias-lambda-vlambda-skipw-head-AdamW;lambda-vlambda-skipw-noWD;resid-x0-lambda-skipw-lrmul=5;muonLR=.023;adamwLR=.008;muonWD=1.2;adamwWD=.005;head-embed-wd_mul=150;quadraticWD;cautiousWD;b2=.9;nesterov;momentum=.85-to-.95-over300-cd50;AdamWbetas=.9,.95;eps=1e-10;AdamWevery=1;stepgraphs;nanogpt-cooldown-frac=.60-floor=.15-v1"
            }
            Self::Adam => {
                "Adam-fp32-masters;betas=.9,.999;eps=1e-8;WD=0;nanogpt-cooldown-frac=.60-floor=.15-v1"
            }
        }
    }
}

/// Step multiplier from modded-nanogpt `TrainingSchedule.get_lr` without the
/// batch-size stage bumps (`train_gpt.py:1968-1976`, `cooldown_frac=0.60`).
pub fn nanogpt_lr_scale(step: usize, scheduled_steps: usize) -> f64 {
    if scheduled_steps == 0 {
        return 1.0;
    }
    let cooldown_start =
        ((scheduled_steps as f64) * (1.0 - NANOGPT_COOLDOWN_FRAC)).floor() as usize;
    if step < cooldown_start {
        return 1.0;
    }
    let span = (scheduled_steps - cooldown_start).max(1) as f64;
    let t = ((step - cooldown_start) as f64 / span).clamp(0.0, 1.0);
    (1.0 - t) + NANOGPT_COOLDOWN_FLOOR * t
}

/// NorMuon momentum from modded-nanogpt `get_muon_momentum` (`train_gpt.py:1995-2007`).
pub fn nanogpt_muon_momentum(step: usize, scheduled_steps: usize) -> f64 {
    if scheduled_steps == 0 {
        return NANOGPT_MOMENTUM_MAX;
    }
    let cooldown_start = scheduled_steps.saturating_sub(NANOGPT_MUON_COOLDOWN_STEPS);
    if step < NANOGPT_MUON_WARMUP_STEPS {
        let frac = step as f64 / NANOGPT_MUON_WARMUP_STEPS as f64;
        NANOGPT_MOMENTUM_MIN + frac * (NANOGPT_MOMENTUM_MAX - NANOGPT_MOMENTUM_MIN)
    } else if step > cooldown_start {
        let frac = ((step - cooldown_start) as f64 / NANOGPT_MUON_COOLDOWN_STEPS as f64).min(1.0);
        NANOGPT_MOMENTUM_MAX - frac * (NANOGPT_MOMENTUM_MAX - NANOGPT_MOMENTUM_MIN)
    } else {
        NANOGPT_MOMENTUM_MAX
    }
}

fn polar_express(named: &[(String, Tensor)], learning_rate: f64) -> Muon {
    let mut optimizer = Muon::new_named(
        named,
        MuonConfig {
            lr: OptimizerKind::PolarExpress
                .muon_learning_rate(learning_rate)
                .unwrap(),
            use_muon_for_2d: true,
            momentum: NANOGPT_MOMENTUM_MIN,
            nesterov: true,
            beta2: 0.9,
            weight_decay: 1.2,
            adamw_lr: learning_rate,
            adamw_betas: (0.9, 0.95),
            adamw_eps: 1e-10,
            adamw_wd: 0.005,
            // Every recipe scalar is `wd_mul = 0` in the reference: the residual/post/x0
            // lambdas at `train_gpt.py:2035-2036` and `train_gpt_medium.py:1081`, the value
            // residual's mixing lambda at the same lines, and the scalar group the U-net skip
            // logits live in at `train_gpt.py:2030`. With `quadratic_lr_weight_decay` a decay
            // on a scalar whose useful value is ~1.05 is a pure bias toward zero - toward
            // deleting the residual stream - and decaying a gate LOGIT toward 0 pulls the gate
            // to `σ(0) = 0.5`, which is not a neutral prior either. The `lambda` fragment
            // covers `lambdas.{resid,post,x0}` and every `block_*.value_lambda` in one entry.
            adamw_no_weight_decay_name_substrings: vec![
                "norm".into(),
                ".bias".into(),
                "lambda".into(),
                "skip_weights".into(),
            ],
            adamw_weight_decay_multipliers: vec![
                ("head.".into(), 150.),
                ("patch.".into(), 150.),
                ("covariates.".into(), 150.),
            ],
            muon_name_allowlist: vec!["block_".into()],
            force_adamw_name_substrings: vec![
                "patch.".into(),
                "covariates.".into(),
                "norm".into(),
                ".bias".into(),
                "head.".into(),
            ],
            orthogonalizer: Orthogonalizer::PolarExpress5,
            adamw_every: 1,
            quadratic_lr_weight_decay: true,
            cautious_weight_decay: true,
            row_learned_lr: false,
            capture_step_graphs: true,
            ..MuonConfig::default()
        },
    );
    // The reference runs the residual and x0 lambdas and its whole scalar group at 5x the Adam
    // base learning rate, and the post lambdas at 1x: `train_gpt.py:2035-2036`
    // (`"post_lambdas": {..., "lr_mul": 1.0, "wd_mul": 0.0}`,
    // `"resid_lambdas": {..., "lr_mul": 5.0, "wd_mul": 0.0}`), `train_gpt.py:2030`
    // (`"scalars": {..., "lr_mul": 5.0, "wd_mul": 0.0}`, which is where the U-net skip logits
    // live) and `train_gpt_medium.py:1078-1081` (`self.x0_lambdas.lr_mul = 5.0`). Five scalars
    // per layer move the whole residual stream, and a gate logit has to travel ~1.5 to close or
    // open, so neither can run at the rate a 512x512 matrix's per-coordinate step wants. The
    // value residual's `block_*.value_lambda` deliberately stays at 1x: the reference's 0.02 is
    // its scalar group's BASE learning rate, not a multiplier over an AdamW base.
    // The count is asserted here, not only in
    // `every_causal_patch_parameter_lands_in_its_intended_optimizer_group`: a silent zero-match
    // leaves the banks at 1x, and a rename should fail loudly at construction rather than train
    // a differently-tuned model. Same reason the disjointness below is an assert.
    assert_eq!(
        optimizer.set_named_lr_scale(
            &["lambdas.resid", "lambdas.x0", "skip_weights"],
            NANOGPT_SCALAR_LR_MULTIPLIER
        ),
        3,
        "the residual lambdas, the x0 lambdas and the U-net skip logits must each take the 5x \
         Adam learning rate exactly once"
    );
    let muon: HashSet<_> = optimizer.muon_param_names().into_iter().collect();
    let adamw: HashSet<_> = optimizer.adamw_param_names().into_iter().collect();
    assert!(muon.is_disjoint(&adamw));
    assert_eq!(
        muon.len() + adamw.len(),
        named.len(),
        "optimizer must cover every parameter exactly once"
    );
    for (name, tensor) in named {
        let hidden = name.starts_with("block_") && name.ends_with(".weight") && tensor.dim() == 2;
        assert_eq!(
            muon.contains(name),
            hidden,
            "unexpected Polar Express routing: {name}"
        );
        assert_eq!(
            adamw.contains(name),
            !hidden,
            "unexpected AdamW routing: {name}"
        );
    }
    optimizer
}

use super::{
    corpus::Batch,
    model::{CausalPatchModel, Losses, ModelConfig},
};

const MIB: f64 = 1048576.;

/// libtorch's CUDA caching allocator at one instant, in MiB.
///
/// `allocated` is what live tensors occupy, `reserved` is what the allocator holds from the
/// driver - and only the second is what the device has to fit, because a private graph
/// mempool's blocks stay reserved for the graph's whole life whether or not anything is
/// live in them.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemorySnapshot {
    pub allocated_mib: f64,
    pub reserved_mib: f64,
    pub peak_allocated_mib: f64,
}

fn memory_snapshot() -> Result<MemorySnapshot> {
    Python::attach(|py| -> Result<MemorySnapshot> {
        let cuda = py.import("torch")?.getattr("cuda")?;
        cuda.call_method0("init")?;
        let read = |name: &str| -> Result<f64> {
            Ok(cuda.call_method1(name, (0,))?.extract::<u64>()? as f64 / MIB)
        };
        Ok(MemorySnapshot {
            allocated_mib: read("memory_allocated")?,
            reserved_mib: read("memory_reserved")?,
            peak_allocated_mib: read("max_memory_allocated")?,
        })
    })
}

fn reset_memory_peaks() -> Result<()> {
    Python::attach(|py| -> Result<()> {
        let cuda = py.import("torch")?.getattr("cuda")?;
        cuda.call_method0("init")?;
        cuda.call_method1("reset_peak_memory_stats", (0,))?;
        Ok(())
    })
}

pub fn device_total_mib() -> Result<f64> {
    Python::attach(|py| -> Result<f64> {
        let cuda = py.import("torch")?.getattr("cuda")?;
        cuda.call_method0("init")?;
        Ok(cuda
            .call_method1("get_device_properties", (0,))?
            .getattr("total_memory")?
            .extract::<u64>()? as f64
            / MIB)
    })
}

/// What capturing the forward and backward actually cost in bytes, measured at each point
/// of the capture protocol rather than assumed.
///
/// The one number that decides whether capture is possible is `pool_reservation_mib`: a
/// private mempool the global allocator can never reuse, held for the process's life,
/// against `device_total_mib` minus everything the eager and evaluation paths still need.
#[derive(Clone, Copy, Debug, Default)]
pub struct CaptureBudget {
    pub before_warmup: MemorySnapshot,
    pub after_warmup: MemorySnapshot,
    pub after_empty_cache: MemorySnapshot,
    pub at_capture_start: MemorySnapshot,
    pub at_capture_end: MemorySnapshot,
    /// `at_capture_end.reserved - at_capture_start.reserved`: the private pool's own
    /// reservation, which is the capture's whole VRAM price.
    pub pool_reservation_mib: f64,
    pub device_total_mib: f64,
    pub warmup_steps: usize,
}

/// The forward and backward of one training step, on the stream that captures them.
///
/// This exists from the first step, not from the capture: before the capture it runs the body
/// EAGERLY ON THE CAPTURE STREAM, which is what makes those steps a warmup at all. cuBLAS
/// selects its algorithm and allocates its workspace per stream, so warming the default
/// stream and then capturing on another warms the wrong one - the capture would be the first
/// time those GEMMs ever ran on that stream.
///
/// The input is the engine's resident batch, refilled in place before every replay. The
/// outputs are the two loss scalars: they live in the capture's private mempool, which the
/// optimizer's own captured bodies share and replay into later in the same step, so a pool
/// address is only meaningful between one replay and the next and they are copied out the
/// instant the replay is issued.
struct StepGraph {
    graph: CudaGraph,
    /// The scalars the captured body writes, held for the graph's life. `None` until the
    /// capture; dropping them afterwards would return their blocks to the pool for a later
    /// capture to reuse while this graph still writes to them.
    outputs: Option<(Tensor, Tensor)>,
}

impl StepGraph {
    /// One forward and backward: the replay once captured, the same body eagerly on the
    /// capture stream until then.
    fn run(&self, model: &CausalPatchModel, batch: &Batch) -> Result<Losses> {
        match &self.outputs {
            Some((nll, mse)) => {
                self.graph
                    .with_stream_scope(CudaGraph::replay)
                    .and_then(|inner| inner)
                    .map_err(|err| {
                        anyhow::anyhow!("replaying the captured training step: {err}")
                    })?;
                Ok(Losses {
                    nll: nll.detach().copy(),
                    mse: mse.detach().copy(),
                })
            }
            None => self
                .graph
                .with_stream_scope(|_| {
                    let losses = Engine::forward_loss(model, batch, true);
                    losses.nll.backward();
                    Losses {
                        nll: losses.nll.detach(),
                        mse: losses.mse,
                    }
                })
                .map_err(|err| anyhow::anyhow!("warming the capture stream: {err}")),
        }
    }
}

/// Eager steps taken before the forward and backward are captured.
///
/// Four is the floor, not a preference: [`Muon::try_graph_step`] warms up three times and
/// captures on its fourth primary step, and the optimizer's captures must be placed in the
/// shared mempool BEFORE this capture fills it, so that the pool's block layout is decided
/// once. The fifth step is this capture's own warmup margin for cuBLAS algorithm selection
/// on the capture stream.
pub const CAPTURE_AFTER_STEPS: usize = 5;

enum Optimizer {
    PolarExpress(Box<Muon>),
    Native(nn::Optimizer),
    Fused(Py<PyAny>),
}

pub struct Engine {
    optimizer: Optimizer,
    completed_steps: usize,
    base_lr: f64,
    scheduled_steps: usize,
    device: Device,
    /// Every trainable parameter, for the gradient-address check the capture protocol
    /// depends on. Shallow clones: they share the store's storage.
    parameters: Vec<Tensor>,
    /// The device-resident batch every step reads, refilled in place. Allocated on the
    /// first step from the first batch's shape.
    resident: Option<Batch>,
    /// The forward and backward on their own stream. Present from construction wherever a
    /// capture is possible at all, because the steps before the capture have to warm the
    /// stream that will run it; `None` off CUDA or in a build without graph support.
    step_graph: Option<StepGraph>,
    capture_budget: Option<CaptureBudget>,
    /// Allocator state before any step ran, for the capture's memory budget.
    before_warmup: MemorySnapshot,
}

impl Engine {
    pub fn new(
        store: &nn::VarStore,
        learning_rate: f64,
        fused: bool,
        kind: OptimizerKind,
        scheduled_steps: usize,
    ) -> Result<Self> {
        ensure!(
            learning_rate.is_finite() && learning_rate > 0.,
            "invalid learning rate"
        );
        let device = store.device();
        // One pool for the whole process, minted before anything can capture into a pool
        // of its own. Every body that captures here - the optimizer's two, and the step
        // graph - allocates only tensors that are transient WITHIN its own replay, and the
        // step graph copies its two output scalars out before the optimizer replays, so
        // the bodies may safely hold the same addresses.
        let graph_pool = if device.is_cuda() && CudaGraph::is_available() {
            Some(Arc::new(CudaGraphPool::new().map_err(|err| {
                anyhow::anyhow!("minting the shared CUDA graph mempool: {err}")
            })?))
        } else {
            None
        };
        // Built now, captured later: every step from the first runs its forward and backward
        // on this graph's stream, so that the capture is not the first time those GEMMs meet
        // that stream.
        let step_graph = match &graph_pool {
            Some(pool) => CudaGraph::new_in_pool(device, pool)
                .map_err(|err| anyhow::anyhow!("building the step graph: {err}"))?
                .map(|graph| StepGraph {
                    graph,
                    outputs: None,
                }),
            None => None,
        };
        let optimizer = if kind == OptimizerKind::PolarExpress {
            let mut muon = polar_express(&named_trainable_variables(store), learning_rate);
            if let Some(pool) = &graph_pool {
                muon.install_graph_pool(Arc::clone(pool));
            }
            Optimizer::PolarExpress(Box::new(muon))
        } else if fused {
            let optimizer = Python::attach(|py| -> Result<Py<PyAny>> {
                let torch = py.import("torch")?;
                let parameters = PyList::empty(py);
                for tensor in store.trainable_variables() {
                    ensure!(
                        tensor.kind() == Kind::Float,
                        "Adam requires FP32 master parameters"
                    );
                    parameters.append(crate::torch::fa4::tensor_object(py, &tensor)?)?;
                }
                let options = PyDict::new(py);
                options.set_item("lr", learning_rate)?;
                options.set_item("fused", true)?;
                options.set_item("weight_decay", 0.)?;
                Ok(torch
                    .getattr("optim")?
                    .getattr("Adam")?
                    .call((parameters,), Some(&options))?
                    .unbind())
            })?;
            Optimizer::Fused(optimizer)
        } else {
            Optimizer::Native(nn::Adam::default().build(store, learning_rate)?)
        };
        let before_warmup = if device.is_cuda() {
            memory_snapshot()?
        } else {
            MemorySnapshot::default()
        };
        let mut engine = Self {
            optimizer,
            completed_steps: 0,
            base_lr: learning_rate,
            scheduled_steps,
            device,
            parameters: store.trainable_variables(),
            resident: None,
            step_graph,
            capture_budget: None,
            before_warmup,
        };
        engine.apply_schedule()?;
        Ok(engine)
    }

    pub fn learning_rate(&self) -> f64 {
        let step = self.completed_steps.saturating_sub(1);
        self.base_lr * nanogpt_lr_scale(step, self.scheduled_steps)
    }

    fn apply_schedule(&mut self) -> Result<()> {
        let scale = nanogpt_lr_scale(self.completed_steps, self.scheduled_steps);
        let adamw_lr = self.base_lr * scale;
        match &mut self.optimizer {
            Optimizer::PolarExpress(optimizer) => {
                optimizer.set_lr(
                    OptimizerKind::PolarExpress
                        .muon_learning_rate(adamw_lr)
                        .unwrap(),
                );
                optimizer.set_adamw_lr(adamw_lr);
                optimizer.set_momentum(nanogpt_muon_momentum(
                    self.completed_steps,
                    self.scheduled_steps,
                ));
            }
            Optimizer::Native(optimizer) => optimizer.set_lr(adamw_lr),
            Optimizer::Fused(optimizer) => Python::attach(|py| -> Result<()> {
                for group in optimizer.bind(py).getattr("param_groups")?.try_iter()? {
                    group?.set_item("lr", adamw_lr)?;
                }
                Ok(())
            })?,
        }
        Ok(())
    }

    pub fn set_lr(&mut self, learning_rate: f64) -> Result<()> {
        ensure!(
            learning_rate.is_finite() && learning_rate > 0.,
            "invalid learning rate"
        );
        self.base_lr = learning_rate;
        self.apply_schedule()
    }

    /// Zeroing, never freeing. `set_to_none=True` is PyTorch's default for the fused
    /// optimizer and releases every gradient buffer, so the next backward reallocates ~100 MB
    /// of them; the two other arms already zero in place, and matching them keeps the
    /// gradient addresses stable for the optimizer's own captured step.
    pub fn zero_grad(&mut self) -> Result<()> {
        match &mut self.optimizer {
            Optimizer::PolarExpress(optimizer) => optimizer.zero_grad(),
            Optimizer::Native(optimizer) => optimizer.zero_grad(),
            Optimizer::Fused(optimizer) => Python::attach(|py| -> PyResult<()> {
                let arguments = PyDict::new(py);
                arguments.set_item("set_to_none", false)?;
                optimizer
                    .bind(py)
                    .call_method("zero_grad", (), Some(&arguments))?;
                Ok(())
            })?,
        }
        Ok(())
    }

    pub fn optimizer_step(&mut self) -> Result<()> {
        self.apply_schedule()?;
        match &mut self.optimizer {
            Optimizer::PolarExpress(optimizer) => {
                optimizer.step(StepKind::Primary);
                assert!(
                    !optimizer.adamw_accumulation_pending(),
                    "every-step AdamW must consume its gradients"
                );
            }
            Optimizer::Native(optimizer) => optimizer.step(),
            Optimizer::Fused(optimizer) => Python::attach(|py| -> PyResult<()> {
                optimizer.bind(py).call_method0("step")?;
                Ok(())
            })?,
        }
        self.completed_steps += 1;
        self.ensure_optimizer_ready()
    }

    pub fn optimizer_graph_captured(&self) -> Option<bool> {
        match &self.optimizer {
            Optimizer::PolarExpress(optimizer) => Some(optimizer.with_adamw_step_graph_captured()),
            _ => None,
        }
    }

    pub fn ensure_optimizer_ready(&self) -> Result<()> {
        ensure!(
            self.completed_steps < 4 || self.optimizer_graph_captured() != Some(false),
            "Polar Express optimizer graph did not remain captured after warmup"
        );
        Ok(())
    }

    pub fn forward_loss(model: &CausalPatchModel, batch: &Batch, train: bool) -> Losses {
        let stats = model.statistics(batch);
        let head = model.forward(batch, &stats, train, false);
        let (targets, mask) = model.targets(batch, &stats, false);
        model.losses(&head, &stats, &targets, &mask)
    }

    /// Refill the engine's resident device batch from `source`, allocating it on the first
    /// call and never again.
    ///
    /// `source` may be a pinned host batch or a device one; either way the copy is
    /// asynchronous, so the host does not wait for the outstanding step to drain. The
    /// destination address is fixed for the run, which is what a captured graph needs and
    /// what keeps `to_device`'s 114 MB allocate-and-free off every step.
    fn upload(&mut self, source: &Batch) -> Result<()> {
        if self.resident.is_none() {
            self.resident = Some(source.resident(self.device));
        }
        source.upload(self.resident.as_mut().expect("just allocated"))
    }

    /// The captured replay once armed, the same body on the capture stream until then, and
    /// the default stream where no capture is possible at all.
    ///
    /// The replay and the warmup body are the same kernels reading the same addresses - the
    /// capture RECORDED this body - so this is a warmup path, not an alternative
    /// implementation. A capture cannot exist before the steps it records have run.
    fn forward_backward(&self, model: &CausalPatchModel) -> Result<Losses> {
        let batch = self
            .resident
            .as_ref()
            .context("no batch has been uploaded to the engine")?;
        match &self.step_graph {
            Some(graph) => graph.run(model, batch),
            None => {
                let losses = Self::forward_loss(model, batch, true);
                losses.nll.backward();
                Ok(Losses {
                    nll: losses.nll.detach(),
                    mse: losses.mse,
                })
            }
        }
    }

    /// One training step, from a host or device batch.
    ///
    /// Nothing here reads a device scalar. The objective's finiteness used to be asserted every
    /// step through `torch._assert_async`, which cost a GIL acquisition and a tensor bridge per
    /// step; the runner now accumulates the indicator on device and reads it once per report
    /// interval alongside the interval's mean loss.
    pub fn step(&mut self, model: &CausalPatchModel, batch: &Batch) -> Result<Losses> {
        self.upload(batch)?;
        self.zero_grad()?;
        let losses = self.forward_backward(model)?;
        self.optimizer_step()?;
        Ok(losses)
    }

    /// Storage address of every trainable parameter's gradient, or `None` where autograd has
    /// not produced one. A capture bakes these in as kernel arguments, and so does the
    /// optimizer's own captured body, so the two captures must agree on them.
    fn gradient_addresses(&self) -> Vec<Option<usize>> {
        self.parameters
            .iter()
            .map(|parameter| {
                let gradient = parameter.grad();
                gradient.defined().then(|| gradient.data_ptr() as usize)
            })
            .collect()
    }

    /// Capture the forward and backward as one graph, and take this step out of the capture.
    ///
    /// The protocol, in the order it has to happen:
    ///
    /// 1. [`CAPTURE_AFTER_STEPS`] warmup steps have already run, on the graph's own side
    ///    stream, so cuBLAS has picked its algorithms there and the optimizer's two bodies
    ///    are already captured into the pool this capture shares.
    /// 2. `empty_cache`, so warmup's cached blocks go back to the driver instead of being
    ///    held alongside the private pool. This is the step whose absence made the first
    ///    attempt at this OOM: two copies of a ~15 GB working set on a 32 GB device.
    /// 3. A static input - the resident batch, refilled by copy - and static outputs, copied
    ///    out of the pool after every replay.
    /// 4. No allocation the replay depends on: gradients exist and are zeroed in place
    ///    (`zero_grad(set_to_none=false)`), which is checked here by comparing every
    ///    gradient's address across the capture.
    ///
    /// The caller must have the host loader quiescent: `pin_memory` can call `cudaHostAlloc`,
    /// which synchronizes the device and would invalidate an in-flight capture.
    pub fn arm_step_graph(&mut self, model: &CausalPatchModel, batch: &Batch) -> Result<Losses> {
        ensure!(
            !self.step_graph_captured(),
            "the training step is already captured"
        );
        ensure!(
            self.completed_steps >= CAPTURE_AFTER_STEPS,
            "capturing the training step needs {CAPTURE_AFTER_STEPS} warmup steps, not {}",
            self.completed_steps
        );
        ensure!(
            self.optimizer_graph_captured() != Some(false),
            "an optimizer that captures its own bodies must have placed them in the shared \
             pool before this capture fills it"
        );
        ensure!(
            self.step_graph.is_some(),
            "capturing the training step needs a CUDA device with graph support"
        );
        self.upload(batch)?;
        self.zero_grad()?;
        tch::Cuda::synchronize(0);
        let after_warmup = memory_snapshot()?;
        cuda::empty_cache();
        let after_empty_cache = memory_snapshot()?;
        let gradients_before = self.gradient_addresses();
        let at_capture_start = memory_snapshot()?;
        let mut captured = None;
        {
            let batch = self.resident.as_ref().expect("uploaded above");
            let graph = &self.step_graph.as_ref().expect("checked above").graph;
            graph
                .with_stream_scope(|graph| {
                    graph.capture(|| {
                        let losses = Self::forward_loss(model, batch, true);
                        losses.nll.backward();
                        captured = Some(losses);
                    })?;
                    graph.replay()
                })
                .and_then(|inner| inner)
                .map_err(|err| anyhow::anyhow!("capturing the training step: {err}"))?;
        }
        let at_capture_end = memory_snapshot()?;
        let losses = captured.context("the capture body did not run")?;
        ensure!(
            self.gradient_addresses() == gradients_before,
            "the capture moved a gradient buffer into its private pool, where the optimizer's \
             already-captured body would not read it"
        );
        let budget = CaptureBudget {
            before_warmup: self.before_warmup,
            after_warmup,
            after_empty_cache,
            at_capture_start,
            at_capture_end,
            pool_reservation_mib: at_capture_end.reserved_mib - at_capture_start.reserved_mib,
            device_total_mib: device_total_mib()?,
            warmup_steps: self.completed_steps,
        };
        println!(
            "CausalPatch training step captured: forward and backward in one graph after {} \
             warmup steps; private mempool {:.0} MiB, reserved {:.0} of {:.0} MiB on device",
            budget.warmup_steps,
            budget.pool_reservation_mib,
            budget.at_capture_end.reserved_mib,
            budget.device_total_mib
        );
        let out = Losses {
            nll: losses.nll.detach().copy(),
            mse: losses.mse.detach().copy(),
        };
        self.step_graph
            .as_mut()
            .expect("checked above")
            .outputs = Some((losses.nll, losses.mse));
        self.capture_budget = Some(budget);
        self.optimizer_step()?;
        Ok(out)
    }

    /// Whether a capture of the step is possible now: the step's graph exists (a CUDA device
    /// and a build with graph support), it has not been captured yet, and the optimizer has
    /// already placed its own bodies in the shared pool. False leaves the run on the warmup
    /// body, exactly as the optimizer's own capture does when the device cannot serve one.
    pub fn capture_ready(&self) -> bool {
        self.step_graph.is_some()
            && !self.step_graph_captured()
            && self.optimizer_graph_captured() != Some(false)
    }

    pub fn capture_budget(&self) -> Option<CaptureBudget> {
        self.capture_budget
    }

    pub fn step_graph_captured(&self) -> bool {
        self.step_graph
            .as_ref()
            .is_some_and(|graph| graph.outputs.is_some())
    }

    /// The same step with a device synchronization between phases, for the report interval's
    /// timing sample and for `benchmark --profile`. One sampled step per interval costs one
    /// serialized step; synchronizing every step would drain the launch pipeline on all of them.
    ///
    /// Phases: batch upload, forward backbone, forward head and loss, backward, captured
    /// forward+backward replay, optimizer. A captured step cannot be split - one replay is one
    /// launch - so the three eager forward/backward phases and the replay phase are mutually
    /// exclusive, and the unavailable ones come back as `NaN` rather than as a zero a reader
    /// would mistake for "free".
    pub fn timed_step(
        &mut self,
        model: &CausalPatchModel,
        batch: &Batch,
    ) -> Result<(Losses, [f64; 6])> {
        let device = self.device;
        let phase = |started: Instant| {
            if device.is_cuda() {
                tch::Cuda::synchronize(0);
            }
            started.elapsed().as_secs_f64() * 1000.
        };
        let started = Instant::now();
        self.upload(batch)?;
        let upload_ms = phase(started);
        self.zero_grad()?;
        let (losses, phases) = if self.step_graph_captured() {
            let graph = self.step_graph.as_ref().expect("captured");
            let resident = self.resident.as_ref().expect("uploaded above");
            let started = Instant::now();
            let losses = graph.run(model, resident)?;
            let replay_ms = phase(started);
            (losses, [f64::NAN, f64::NAN, f64::NAN, replay_ms])
        } else {
            let resident = self
                .resident
                .as_ref()
                .context("no batch has been uploaded to the engine")?;
            // Split into phases, and on the capture stream wherever one exists, so that a
            // sampled step before the capture still warms the stream the capture will use.
            let body = || {
                let started = Instant::now();
                let stats = model.statistics(resident);
                let state = model.backbone(resident, &stats, true, false);
                let backbone_ms = phase(started);
                let started = Instant::now();
                let head = model.head(resident, &state, false);
                let (targets, mask) = model.targets(resident, &stats, false);
                let losses = model.losses(&head, &stats, &targets, &mask);
                let head_ms = phase(started);
                let started = Instant::now();
                losses.nll.backward();
                let backward_ms = phase(started);
                (
                    Losses {
                        nll: losses.nll.detach(),
                        mse: losses.mse,
                    },
                    [backbone_ms, head_ms, backward_ms, f64::NAN],
                )
            };
            match &self.step_graph {
                Some(graph) => graph
                    .graph
                    .with_stream_scope(|_| body())
                    .map_err(|err| anyhow::anyhow!("warming the capture stream: {err}"))?,
                None => body(),
            }
        };
        let started = Instant::now();
        self.optimizer_step()?;
        let optimizer_ms = phase(started);
        Ok((
            losses,
            [
                upload_ms,
                phases[0],
                phases[1],
                phases[2],
                phases[3],
                optimizer_ms,
            ],
        ))
    }
}

/// Fixed seed for both audit phases, so the two engines start from the same parameters.
const AUDIT_SEED: i64 = 20260906;

/// One phase of [`audit_step_capture`], reduced to host values so nothing device-resident
/// survives it: the private pool and the eager working set are each about half the device,
/// so the two phases cannot overlap.
struct AuditPhase {
    /// Objective value per compared step.
    losses: Vec<f64>,
    /// Final parameters by name, on the host.
    parameters: Vec<(String, Tensor)>,
    step_ms: f64,
    /// The same steps with one host read of the objective each - the `.item()` the training
    /// loop used to pay to assert finiteness. `NaN` in the capture phase, which does not run
    /// the serialization diagnostics.
    host_read_step_ms: f64,
    /// The same steps sourced from pinned host memory, as the training loader delivers.
    pinned_upload_step_ms: f64,
    /// The same steps sourced from pageable host memory: the copy stages through a driver
    /// buffer and blocks the host until the device drains.
    pageable_upload_step_ms: f64,
    peak_allocated_mib: f64,
    budget: Option<CaptureBudget>,
}

fn audit_phase(
    config: &ModelConfig,
    batch: &Batch,
    learning_rate: f64,
    fused: bool,
    kind: OptimizerKind,
    compared: usize,
    capture: bool,
) -> Result<AuditPhase> {
    let device = batch.log_prices.device();
    tch::manual_seed(AUDIT_SEED);
    tch::Cuda::manual_seed_all(AUDIT_SEED as u64);
    let store = nn::VarStore::new(device);
    let model = CausalPatchModel::new(&store.root(), config);
    let mut engine = Engine::new(&store, learning_rate, fused, kind, 0)?;
    let mut budget = None;
    for step in 0..=CAPTURE_AFTER_STEPS {
        if capture && step == CAPTURE_AFTER_STEPS {
            let _ = engine.arm_step_graph(&model, batch)?;
            budget = engine.capture_budget();
        } else {
            let _ = engine.step(&model, batch)?;
        }
    }
    ensure!(
        engine.step_graph_captured() == capture,
        "the audit phase did not reach the capture state it was asked for"
    );
    tch::Cuda::synchronize(0);
    reset_memory_peaks()?;
    let started = Instant::now();
    let mut objectives = Vec::with_capacity(compared);
    for _ in 0..compared {
        objectives.push(engine.step(&model, batch)?.nll);
    }
    tch::Cuda::synchronize(0);
    let step_ms = started.elapsed().as_secs_f64() * 1000. / compared as f64;
    let peak_allocated_mib = memory_snapshot()?.peak_allocated_mib;
    // One transfer for the whole window, after the timing closed.
    let losses = Vec::<f64>::try_from(
        Tensor::stack(&objectives, 0)
            .to_kind(Kind::Double)
            .to_device(Device::Cpu),
    )?;
    let mut parameters: Vec<(String, Tensor)> = store
        .variables()
        .into_iter()
        .map(|(name, tensor)| (name, tensor.detach().to_device(Device::Cpu)))
        .collect();
    parameters.sort_by(|left, right| left.0.cmp(&right.0));
    // The serialization diagnostics run only in the eager phase, and only after the compared
    // window has been read out: each of them takes more steps, which move the parameters.
    // The capture phase must not run them at all - its private pool plus a second host-side
    // 114 MB staging copy is not what this device has to spare.
    let mut host_read_step_ms = f64::NAN;
    let mut pinned_upload_step_ms = f64::NAN;
    let mut pageable_upload_step_ms = f64::NAN;
    if !capture {
        let timed = |engine: &mut Engine, source: &Batch, read: bool| -> Result<f64> {
            tch::Cuda::synchronize(0);
            let started = Instant::now();
            for _ in 0..compared {
                let losses = engine.step(&model, source)?;
                if read {
                    // What the removed per-step finiteness assertion cost: a device drain.
                    let _ = losses.nll.double_value(&[]);
                }
            }
            tch::Cuda::synchronize(0);
            Ok(started.elapsed().as_secs_f64() * 1000. / compared as f64)
        };
        host_read_step_ms = timed(&mut engine, batch, true)?;
        let pinned = batch.host_copy(Some(device));
        pinned_upload_step_ms = timed(&mut engine, &pinned, false)?;
        drop(pinned);
        let pageable = batch.host_copy(None);
        pageable_upload_step_ms = timed(&mut engine, &pageable, false)?;
    }
    drop(engine);
    drop(model);
    drop(store);
    cuda::empty_cache();
    Ok(AuditPhase {
        losses,
        parameters,
        step_ms,
        host_read_step_ms,
        pinned_upload_step_ms,
        pageable_upload_step_ms,
        peak_allocated_mib,
        budget,
    })
}

/// The capture verdict, measured in one process on one device.
///
/// Three phases run [`CAPTURE_AFTER_STEPS`] warmup steps and then `compared` more over the
/// SAME batch from the SAME seed: eager, eager again, and captured. Comparing the final
/// parameters is what makes this a gradient test - the optimizer integrates every step's
/// gradient into them, so a single differing element anywhere in `compared` consecutive
/// backward passes shows up here. The eager-against-eager pair is the null: this step is not
/// bit-reproducible run to run, so the capture's deviation is only meaningful beside it.
#[derive(Clone, Copy, Debug)]
pub struct CaptureAudit {
    pub budget: CaptureBudget,
    pub eager_step_ms: f64,
    /// The second eager phase, which is also this timing's own repeatability.
    pub eager_repeat_step_ms: f64,
    pub replay_step_ms: f64,
    /// Worst objective difference between the two EAGER phases, and worst parameter
    /// difference between them: the floor that the device's own nondeterminism sets.
    pub eager_loss_max_relative: f64,
    pub eager_parameter_max_relative: f64,
    /// The eager step with one host read of the objective per step: the serialization the
    /// training loop's per-step finiteness assertion used to pay.
    pub host_read_step_ms: f64,
    /// The eager step sourced from pinned host memory, as the training loader delivers it.
    pub pinned_upload_step_ms: f64,
    /// The eager step sourced from pageable host memory, which the driver has to stage.
    pub pageable_upload_step_ms: f64,
    pub eager_peak_allocated_mib: f64,
    pub compared_steps: usize,
    pub loss_max_relative: f64,
    pub parameter_max_relative: f64,
}

/// One pair of audit phases, compared. `worst_*` name the parameter tensor that diverged
/// most, with the absolute difference and the tensor's own scale, because a relative
/// difference on a tensor whose largest element is near zero says nothing on its own.
#[derive(Default)]
struct Comparison {
    loss_first_relative: f64,
    loss_max_relative: f64,
    parameter_max_relative: f64,
    worst_parameter: String,
    worst_absolute: f64,
    worst_scale: f64,
}

pub fn audit_step_capture(
    config: &ModelConfig,
    batch: &Batch,
    learning_rate: f64,
    fused: bool,
    kind: OptimizerKind,
    compared: usize,
) -> Result<CaptureAudit> {
    ensure!(compared >= 20, "the capture audit compares at least 20 steps");
    ensure!(
        batch.log_prices.device().is_cuda(),
        "the capture audit needs a CUDA device"
    );
    // Three phases: eager, eager again, captured. The second one is the NULL: whatever two
    // identical eager runs differ by is the floor this device and this optimizer impose, and
    // the capture can only be judged against that floor, never against zero.
    let eager = audit_phase(config, batch, learning_rate, fused, kind, compared, false)?;
    let repeated = audit_phase(config, batch, learning_rate, fused, kind, compared, false)?;
    let replayed = audit_phase(config, batch, learning_rate, fused, kind, compared, true)?;
    let compare = |left: &AuditPhase, right: &AuditPhase| -> Result<Comparison> {
        let relative = |a: f64, b: f64| {
            let scale = a.abs().max(b.abs());
            if scale == 0. {
                0.
            } else {
                (a - b).abs() / scale
            }
        };
        ensure!(
            left.parameters.len() == right.parameters.len(),
            "the audit phases registered different parameter sets"
        );
        let mut out = Comparison::default();
        // The FIRST compared step is the phases' own arithmetic: both reached it from the
        // same parameters, so anything nonzero there is a kernel that rounded differently.
        // Later steps add NorMuon's amplification of that, which is why they are apart.
        out.loss_first_relative = match (left.losses.first(), right.losses.first()) {
            (Some(&a), Some(&b)) => relative(a, b),
            _ => f64::NAN,
        };
        out.loss_max_relative = left
            .losses
            .iter()
            .zip(&right.losses)
            .map(|(&a, &b)| relative(a, b))
            .fold(0., f64::max);
        for ((name, a), (other, b)) in left.parameters.iter().zip(&right.parameters) {
            ensure!(name == other, "audit parameter order diverged at {name}");
            let difference = (a - b).abs().max().double_value(&[]);
            let scale = a.abs().max().double_value(&[]);
            let value = if scale == 0. {
                difference
            } else {
                difference / scale
            };
            if value > out.parameter_max_relative {
                out.parameter_max_relative = value;
                out.worst_parameter = name.clone();
                out.worst_absolute = difference;
                out.worst_scale = scale;
            }
        }
        Ok(out)
    };
    let null = compare(&eager, &repeated)?;
    let capture = compare(&eager, &replayed)?;
    let audit = CaptureAudit {
        budget: replayed
            .budget
            .context("the capture phase reported no memory budget")?,
        eager_step_ms: eager.step_ms,
        eager_repeat_step_ms: repeated.step_ms,
        replay_step_ms: replayed.step_ms,
        host_read_step_ms: eager.host_read_step_ms,
        pinned_upload_step_ms: eager.pinned_upload_step_ms,
        pageable_upload_step_ms: eager.pageable_upload_step_ms,
        eager_peak_allocated_mib: eager.peak_allocated_mib,
        compared_steps: compared,
        loss_max_relative: capture.loss_max_relative,
        parameter_max_relative: capture.parameter_max_relative,
        eager_loss_max_relative: null.loss_max_relative,
        eager_parameter_max_relative: null.parameter_max_relative,
    };
    for (label, phase, ms) in [
        ("eager vs eager", &null, repeated.step_ms),
        ("eager vs replay", &capture, replayed.step_ms),
    ] {
        println!(
            "CausalPatch capture audit, {label} over {compared} steps ({:.1} ms vs {ms:.1} ms \
             per step): objective relative difference {:.3e} on the first step, {:.3e} worst \
             over the window; worst parameter {} at {:.3e} relative ({:.3e} absolute on a \
             {:.3e} scale)",
            eager.step_ms,
            phase.loss_first_relative,
            phase.loss_max_relative,
            phase.worst_parameter,
            phase.parameter_max_relative,
            phase.worst_absolute,
            phase.worst_scale,
        );
    }
    println!(
        "CausalPatch step serialization: baseline {:.1} ms, one host objective read per step \
         {:.1} ms, pinned host source {:.1} ms, pageable host source {:.1} ms",
        audit.eager_step_ms,
        audit.host_read_step_ms,
        audit.pinned_upload_step_ms,
        audit.pageable_upload_step_ms,
    );
    Ok(audit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn polar_express_routes_only_hidden_matrices_and_keeps_readout_on_adamw() {
        let store = nn::VarStore::new(Device::Cpu);
        for (path, name, shape) in [
            ("block_0.qkv", "weight", vec![24, 8]),
            ("block_0.output", "weight", vec![8, 8]),
            ("block_0.first", "weight", vec![32, 8]),
            ("block_0.second", "weight", vec![8, 32]),
            // The recipe leaves the blocks with no bias and no norm gain, so the 1-D negative
            // control - "a non-matrix must not reach NorMuon" - is now the lambda banks, which
            // are also what `set_named_lr_scale` below must match.
            ("lambdas", "resid", vec![4]),
            ("lambdas", "post", vec![4]),
            ("lambdas", "x0", vec![2]),
            ("patch", "weight", vec![8, 4]),
            ("covariates", "weight", vec![8, 64]),
            ("head.hidden", "weight", vec![16, 16]),
            ("head.output", "weight", vec![4, 16]),
            ("head.output", "bias", vec![4]),
            ("block_1", "value_lambda", vec![1]),
        ] {
            let path = path.split('.').fold(store.root(), |path, part| path / part);
            let _ = path.var(name, &shape, nn::Init::Const(0.));
        }
        // Root-level, 1-D: the U-net skip gate logits, whose whole routing story is that they
        // are outside the `block_*` allowlist and so cannot reach NorMuon.
        let _ = store.root().var("skip_weights", &[4], nn::Init::Const(-1.5));
        let named = named_trainable_variables(&store);
        let optimizer = polar_express(&named, NANOGPT_ADAMW_LR);
        assert_eq!(optimizer.muon_param_names().len(), 4);
        assert_eq!(optimizer.adamw_param_names().len(), 10);
        let (lr_scale, betas, wd_multiplier) = optimizer
            .adamw_group_settings("skip_weights")
            .expect("the skip gate logits must be an AdamW parameter");
        assert!((lr_scale - NANOGPT_SCALAR_LR_MULTIPLIER).abs() < 1e-12);
        assert_eq!(betas, (0.9, 0.95));
        assert_eq!(wd_multiplier, 0.0);
        for name in [
            "head.hidden.weight",
            "head.output.weight",
            "covariates.weight",
            "patch.weight",
        ] {
            assert!(optimizer
                .adamw_param_names()
                .iter()
                .any(|candidate| candidate == name));
        }
        // The value-residual mixing weight is a `block_*` parameter, so it must be the NAME and
        // the rank - not the prefix - that keep it off NorMuon, and it must decay at zero.
        assert!(optimizer
            .adamw_param_names()
            .iter()
            .any(|candidate| candidate == "block_1.value_lambda"));
        assert_eq!(
            optimizer
                .adamw_group_settings("block_1.value_lambda")
                .expect("the mixing weight is an AdamW parameter")
                .2,
            0.0,
            "the mixing weight must not be decayed toward zero"
        );
        assert!((optimizer.lr() - NANOGPT_MUON_LR).abs() < 1e-12);
        assert!(!optimizer.with_adamw_step_graph_captured());
        let mut engine = Engine::new(
            &store,
            NANOGPT_ADAMW_LR,
            true,
            OptimizerKind::PolarExpress,
            0,
        )
        .unwrap();
        engine.set_lr(NANOGPT_ADAMW_LR / 2.0).unwrap();
        let Optimizer::PolarExpress(optimizer) = engine.optimizer else {
            unreachable!()
        };
        assert!((optimizer.lr() - NANOGPT_MUON_LR / 2.0).abs() < 1e-12);
        assert!(!optimizer.adamw_accumulation_pending());
    }

    /// Every parameter of a real model, enumerated with the optimizer group it lands in.
    ///
    /// Misrouting is silent: a recipe scalar in NorMuon's 2-D group would be orthogonalized as
    /// if it were a weight matrix, and a lambda left in AdamW's decaying group would be pulled
    /// toward zero - i.e. toward deleting the residual stream - by a decay that is quadratic in
    /// the learning rate. Neither shows up as an error, only as a worse run, so the whole
    /// parameter list is pinned here rather than spot-checked.
    #[test]
    fn every_causal_patch_parameter_lands_in_its_intended_optimizer_group() {
        let store = nn::VarStore::new(Device::Cpu);
        let config = ModelConfig {
            seq_len: 96,
            pred_len: 8,
            patch_len: 8,
            layers: 2,
            d_model: 32,
            heads: 4,
            ffn: 64,
            min_history: 8,
            ..Default::default()
        };
        let _model = CausalPatchModel::new(&store.root(), &config);
        let named = named_trainable_variables(&store);
        let optimizer = polar_express(&named, NANOGPT_ADAMW_LR);
        let muon = optimizer.muon_param_names();
        let mut expected_muon: Vec<String> = (0..config.layers)
            .flat_map(|layer| {
                ["qkv", "output", "first", "second"]
                    .into_iter()
                    .map(move |projection| format!("block_{layer}.{projection}.weight"))
            })
            .collect();
        expected_muon.sort();
        let mut sorted_muon = muon.clone();
        sorted_muon.sort();
        assert_eq!(
            sorted_muon, expected_muon,
            "NorMuon must hold exactly the four hidden matrices per block - no norm gains \
             (the RMSNorm is gainless), no biases (the block projections have none), and no \
             recipe scalars"
        );
        // Every remaining parameter, with the group settings the reference prescribes:
        // `(lr scale, betas, weight-decay multiplier)`.
        for (name, lr_scale, wd_mul) in [
            ("lambdas.resid", 5.0, 0.0),
            ("lambdas.post", 1.0, 0.0),
            ("lambdas.x0", 5.0, 0.0),
            ("patch.weight", 1.0, 150.0),
            ("patch.bias", 1.0, 0.0),
            ("head.hidden.weight", 1.0, 150.0),
            ("head.hidden.bias", 1.0, 0.0),
            ("head.output.weight", 1.0, 150.0),
            ("head.output.bias", 1.0, 0.0),
        ] {
            let (scale, betas, decay) = optimizer
                .adamw_group_settings(name)
                .unwrap_or_else(|| panic!("{name} is not an AdamW parameter"));
            assert!(
                (scale - lr_scale).abs() < 1e-12,
                "{name} learning-rate scale is {scale}, expected {lr_scale}"
            );
            assert!(
                (decay - wd_mul).abs() < 1e-12,
                "{name} weight-decay multiplier is {decay}, expected {wd_mul}"
            );
            assert_eq!(betas, (0.9, 0.95), "{name} betas");
        }
        // And nothing outside those two lists exists: the union is the whole store, which
        // `polar_express` already asserts, so it is enough to pin the AdamW count.
        let lambda_banks = 3;
        let per_block = 4;
        assert_eq!(
            muon.len(),
            per_block * config.layers,
            "one hidden matrix group per block projection"
        );
        assert_eq!(
            optimizer.adamw_param_names().len(),
            named.len() - per_block * config.layers,
            "every non-hidden parameter is AdamW"
        );
        assert!(
            optimizer
                .adamw_param_names()
                .iter()
                .filter(|name| name.starts_with("lambdas."))
                .count()
                == lambda_banks,
            "all three lambda banks are AdamW-routed"
        );
        for (name, tensor) in &named {
            if name.starts_with("lambdas.") {
                assert_eq!(
                    tensor.dim(),
                    1,
                    "{name} must stay 1-D: a 2-D lambda bank is what NorMuon's router looks \
                     for (optim/muon.rs:1316)"
                );
            }
        }
    }
    #[test]
    fn nanogpt_lr_scale_holds_then_cools_to_the_floor() {
        let scheduled = 1270;
        let start = ((scheduled as f64) * 0.4).floor() as usize;
        assert_eq!(start, 508);
        assert!((nanogpt_lr_scale(0, scheduled) - 1.0).abs() < 1e-12);
        assert!((nanogpt_lr_scale(start, scheduled) - 1.0).abs() < 1e-12);
        let mid = start + (scheduled - start) / 2;
        assert!((nanogpt_lr_scale(mid, scheduled) - 0.575).abs() < 1e-12);
        assert!((nanogpt_lr_scale(scheduled, scheduled) - 0.15).abs() < 1e-12);
        assert!((nanogpt_lr_scale(scheduled + 15, scheduled) - 0.15).abs() < 1e-12);
        assert!((nanogpt_lr_scale(0, 0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nanogpt_muon_momentum_warms_up_and_cools_down() {
        assert!((nanogpt_muon_momentum(0, 1270) - 0.85).abs() < 1e-12);
        assert!((nanogpt_muon_momentum(150, 1270) - 0.90).abs() < 1e-12);
        assert!((nanogpt_muon_momentum(300, 1270) - 0.95).abs() < 1e-12);
        assert!((nanogpt_muon_momentum(1000, 1270) - 0.95).abs() < 1e-12);
        assert!((nanogpt_muon_momentum(1245, 1270) - 0.90).abs() < 1e-12);
        assert!((nanogpt_muon_momentum(1270, 1270) - 0.85).abs() < 1e-12);
        assert!((nanogpt_muon_momentum(0, 0) - 0.95).abs() < 1e-12);
    }

}
