//! Attribution inside `loss.backward()`.
//!
//! The autograd engine runs the whole backward wave as one opaque call, so the region guards in
//! [`super::train::pretrain_profile`] can only bracket it end to end. This module opens it up
//! two ways, both armed by the same `PRETRAIN_PROFILE` gate as the per-layer forward regions
//! and both costing one relaxed atomic load when that gate is closed.
//!
//! * [`begin`] and [`mark`] register autograd hooks on the forward's boundary tensors. A hook
//!   fires when the wave has finished every consumer of its tensor and that tensor's gradient
//!   is complete, so a chain of marks partitions the backward exactly the way a chain of region
//!   guards partitions the forward: each mark closes the interval the mark below it opened.
//! * [`trace_step_boundary`] wraps one whole step in libtorch's own operator profiler, which
//!   attributes the engine's operators without any help from us.
//!
//! Neither goes through `tch`, which binds no hook and no profiler API, and neither goes
//! through the vendored `torch-sys`, which binds neither either. Both go through the in-process
//! Python interpreter that already serves FA4. That is sound rather than a hack: `torch-env.sh`
//! builds with `LIBTORCH_USE_PYTORCH=1`, so the wheel's libtorch is the one `tch` links, and
//! `THPVariable_Wrap` hands Python the same `Variable` with the same autograd metadata. A
//! Python-registered hook therefore fires on a Rust-built graph, and a Python-started profiler
//! sees Rust-issued operators.

use anyhow::{anyhow, Result};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyCFunction, PyDict, PyTuple};
use std::ffi::CString;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use tch::Tensor;

use super::fa4::tensor_object;
use super::train::pretrain_profile as profile;

/// The trunk pass the operator trace opens on, counted from the process's first pass. Far
/// enough in that the allocator, the autotuner and FA4's CuTe compilation have all settled, so
/// the trace describes the steady state rather than a warmup.
const TRACE_PASS: usize = 59;

/// When the backward wave last crossed a boundary. The engine runs one graph on one worker
/// thread, so this is uncontended; it needs a mutex only because an `Instant` will not fit in
/// an atomic.
static CURSOR: Mutex<Option<Instant>> = Mutex::new(None);
static TRACE_PASSES: AtomicUsize = AtomicUsize::new(0);
static TRACE_CONTROL: PyOnceLock<(Py<PyAny>, Py<PyAny>)> = PyOnceLock::new();

/// Open the backward wave's first interval at `tensor`, recording nothing.
///
/// Marks measure back from the previous boundary, so the topmost boundary has nothing to
/// measure back to and must only reset the cursor. Without this the first interval of a step
/// would be charged the gap since the last interval of the step before it.
pub fn begin(tensor: &Tensor) {
    if !profile::fine() {
        return;
    }
    install(tensor, None);
}

/// Charge the backward time between the previous boundary and `tensor`'s gradient to `closes`.
pub fn mark(tensor: &Tensor, closes: &'static str) {
    if !profile::fine() {
        return;
    }
    install(tensor, Some(closes));
}

fn install(tensor: &Tensor, closes: Option<&'static str>) {
    if !tensor.requires_grad() {
        return;
    }
    let label = closes.unwrap_or("backward entry");
    if let Err(error) = register(tensor, closes) {
        eprintln!("backward probe: no hook on `{label}`: {error:#}");
    }
}

fn register(tensor: &Tensor, closes: Option<&'static str>) -> Result<()> {
    Python::attach(|py| -> Result<()> {
        // `THPVariable_Wrap` dereferences the `torch._C._TensorBase` type object, which only
        // exists once Python has imported torch. Wrapping before that segfaults.
        py.import("torch")
            .map_err(|error| anyhow!("failed to initialize Python torch: {error:?}"))?;
        let object = tensor_object(py, tensor)?;
        // A tensor hook returns the gradient it wants substituted, or `None` to leave it
        // alone. Returning `()` from the closure would hand autograd an empty tuple.
        let hook = PyCFunction::new_closure(
            py,
            Some(c"backward_probe_boundary"),
            None,
            move |arguments: &Bound<'_, PyTuple>,
                  _keywords: Option<&Bound<'_, PyDict>>|
                  -> PyResult<Py<PyAny>> {
                boundary(closes);
                Ok(arguments.py().None())
            },
        )
        .map_err(|error| anyhow!("building the boundary hook failed: {error:?}"))?;
        object
            .call_method1("register_hook", (hook,))
            .map_err(|error| anyhow!("register_hook failed: {error:?}"))?;
        Ok(())
    })
}

/// Close the open interval and open the next one.
///
/// Draining the device first is what makes the interval an attribution rather than a reading of
/// launch queue depth, exactly as it is for a region guard. It also means the instrumented
/// backward is slower than the backward it measures, so the shares are the result and the
/// absolute total is not.
fn boundary(closes: Option<&'static str>) {
    profile::drain();
    let now = Instant::now();
    let previous = CURSOR
        .lock()
        .expect("the backward probe cursor is not poisoned")
        .replace(now);
    if let (Some(closes), Some(previous)) = (closes, previous) {
        profile::record(closes, (now - previous).as_secs_f64());
    }
}

/// Trace one whole step with libtorch's own operator profiler.
///
/// Called once per trunk forward. The trace opens on the [`TRACE_PASS`]-th pass and closes on
/// the next one, so it spans a whole step — forward, backward, optimizer — and every operator
/// is attributed by libtorch, including those only the autograd engine ever issues.
pub fn trace_step_boundary() {
    if !profile::trace() {
        return;
    }
    let pass = TRACE_PASSES.fetch_add(1, Ordering::Relaxed);
    let start = match pass {
        TRACE_PASS => true,
        _ if pass == TRACE_PASS + 1 => false,
        _ => return,
    };
    if let Err(error) = drive_trace(start) {
        eprintln!("backward probe: operator trace failed: {error:#}");
    }
}

fn drive_trace(start: bool) -> Result<()> {
    Python::attach(|py| -> Result<()> {
        let (open, close) = TRACE_CONTROL.get_or_try_init(py, || -> Result<(Py<PyAny>, Py<PyAny>)> {
            let source = CString::new(TRACE_HELPER)?;
            py.run(source.as_c_str(), None, None)
                .map_err(|error| anyhow!("installing the trace helper failed: {error:?}"))?;
            let main = py.import("__main__")?;
            Ok((
                main.getattr("_rust_probe_trace_start")?.unbind(),
                main.getattr("_rust_probe_trace_stop")?.unbind(),
            ))
        })?;
        let control = if start { open } else { close };
        control
            .bind(py)
            .call0()
            .map_err(|error| anyhow!("the trace helper raised: {error:?}"))?;
        Ok(())
    })
}

const TRACE_HELPER: &str = r#"
import torch as _rust_probe_torch

if "_RUST_PROBE_TRACE" not in globals():
    _RUST_PROBE_TRACE = None

if "_rust_probe_trace_start" not in globals():
    def _rust_probe_trace_start():
        global _RUST_PROBE_TRACE
        if _RUST_PROBE_TRACE is not None:
            return
        activity = _rust_probe_torch.profiler.ProfilerActivity
        _RUST_PROBE_TRACE = _rust_probe_torch.profiler.profile(
            activities=[activity.CPU, activity.CUDA],
        )
        _RUST_PROBE_TRACE.__enter__()

    def _rust_probe_trace_stop():
        global _RUST_PROBE_TRACE
        session = _RUST_PROBE_TRACE
        if session is None:
            return
        _RUST_PROBE_TRACE = None
        session.__exit__(None, None, None)
        averages = session.key_averages()
        for column in ("self_device_time_total", "self_cpu_time_total"):
            print(f"backward probe: operators by {column}", flush=True)
            print(averages.table(sort_by=column, row_limit=45), flush=True)
"#;
