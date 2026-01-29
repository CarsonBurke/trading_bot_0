//! Mamba 2 SSM reference implementation bridge
//!
//! Delegates to the Python mamba-ssm reference via PyO3 FFI.
//! Provides the same `StatefulMambaRef` interface as `StatefulMamba` in ssm.rs.
//!
//! Zero-copy tensor sharing: both tch-rs and Python torch link the same libtorch,
//! so `pyobject_wrap()`/`pyobject_unpack()` share the underlying `at::Tensor`.

use std::sync::LazyLock;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use tch::{nn, Kind, Tensor};

pub use super::ssm::{Mamba2Config, Mamba2State};

static BRIDGE_DIR: LazyLock<String> = LazyLock::new(|| {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_string());
    format!("{}/mamba2_ref", manifest)
});

static BRIDGE_INIT: LazyLock<()> = LazyLock::new(|| {
    Python::with_gil(|py| {
        let sys = py.import("sys").expect("failed to import sys");
        let path = sys.getattr("path").expect("no sys.path");
        let path = path.downcast::<pyo3::types::PyList>().expect("sys.path is not a list");
        path.insert(0, &*BRIDGE_DIR)
            .expect("failed to insert bridge dir into sys.path");
    });
});

fn ensure_bridge_init() {
    let _ = *BRIDGE_INIT;
}

fn device_str(device: tch::Device) -> &'static str {
    match device {
        tch::Device::Cpu => "cpu",
        tch::Device::Cuda(0) => "cuda:0",
        tch::Device::Cuda(1) => "cuda:1",
        tch::Device::Cuda(2) => "cuda:2",
        tch::Device::Cuda(3) => "cuda:3",
        tch::Device::Cuda(4) => "cuda:4",
        tch::Device::Cuda(5) => "cuda:5",
        tch::Device::Cuda(6) => "cuda:6",
        tch::Device::Cuda(7) => "cuda:7",
        _ => panic!("unsupported device {:?}", device),
    }
}

fn kind_str(kind: Kind) -> &'static str {
    match kind {
        Kind::Float => "float32",
        Kind::Half => "float16",
        Kind::BFloat16 => "bfloat16",
        Kind::Double => "float64",
        _ => "float32",
    }
}

// ---------------------------------------------------------------------------
// Zero-copy tensor ↔ PyObject conversion
// ---------------------------------------------------------------------------

/// Wrap a tch::Tensor as a Python torch.Tensor (zero-copy via shared storage).
fn tensor_to_py(py: Python<'_>, t: &Tensor) -> PyObject {
    let raw = t.pyobject_wrap().expect("pyobject_wrap failed");
    unsafe { Py::from_owned_ptr(py, raw as *mut pyo3::ffi::PyObject) }
}

/// Wrap an optional tensor; returns Python `None` if absent.
fn opt_tensor_to_py(py: Python<'_>, t: Option<&Tensor>) -> PyObject {
    match t {
        Some(tensor) => tensor_to_py(py, tensor),
        None => py.None(),
    }
}

/// Extract a tch::Tensor from a `PyObject` (zero-copy via shared storage).
fn py_to_tensor(obj: &PyObject) -> Tensor {
    unsafe { tensor_from_ptr(obj.as_ptr()) }
}

/// Extract a tch::Tensor from a raw borrowed `*mut ffi::PyObject`.
/// # Safety
/// `ptr` must be a valid Python torch.Tensor with active refcount.
unsafe fn tensor_from_ptr(ptr: *mut pyo3::ffi::PyObject) -> Tensor {
    Tensor::pyobject_unpack(ptr as *mut tch::python::CPyObject)
        .expect("pyobject_unpack failed")
        .expect("object is not a torch.Tensor")
}

/// Extract the `idx`-th element of a Python tuple as a tch::Tensor.
/// Uses `PyTuple_GetItem` (borrowed reference — no refcount overhead).
/// # Safety
/// `tup` must be a valid `PyTuple` and `idx` must be in bounds.
unsafe fn tuple_get_tensor(tup: *mut pyo3::ffi::PyObject, idx: isize) -> Tensor {
    let item = pyo3::ffi::PyTuple_GetItem(tup, idx as pyo3::ffi::Py_ssize_t);
    assert!(!item.is_null(), "PyTuple_GetItem null at index {idx}");
    tensor_from_ptr(item)
}

// ---------------------------------------------------------------------------
// Core bridge
// ---------------------------------------------------------------------------

pub struct Mamba2Ref {
    handle: i64,
    bridge: PyObject,
    /// (python_name, tensor) — tensor shares storage with Python's parameter
    params: Vec<(String, Tensor)>,
    pub config: Mamba2Config,
}

impl Mamba2Ref {
    pub fn new(p: &nn::Path, config: Mamba2Config) -> Self {
        ensure_bridge_init();

        Python::with_gil(|py| {
            let bridge: PyObject = PyModule::import(py, "bridge")
                .expect("failed to import mamba2_ref bridge")
                .into();

            let device = p.device();
            let kind = Kind::Float;

            macro_rules! py_val {
                ($v:expr) => { $v.into_pyobject(py).expect("into_pyobject").to_owned().into_any().unbind() }
            }
            let d_ssm_py: PyObject = match config.d_ssm {
                Some(v) => py_val!(v),
                None => py.None(),
            };
            let create_args = PyTuple::new(py, &[
                py_val!(config.d_model),
                py_val!(config.d_state),
                py_val!(config.d_conv),
                py_val!(config.expand),
                py_val!(config.headdim),
                d_ssm_py,
                py_val!(config.ngroups),
                py_val!(config.chunk_size),
                py_val!(config.dt_min),
                py_val!(config.dt_max),
                py_val!(config.norm_before_gate),
                py_val!(config.d_has_hdim),
                py_val!(device_str(device)),
                py_val!(kind_str(kind)),
            ]).expect("failed to create tuple");

            let handle: i64 = bridge
                .call_method1(py, "create_layer", create_args)
                .expect("create_layer failed")
                .extract(py)
                .expect("handle not i64");

            let py_params = bridge
                .call_method1(py, "get_named_parameters", (handle,))
                .expect("get_named_parameters failed");
            let py_params = py_params.bind(py);
            let n: usize = py_params.len().expect("len failed");

            let mut params = Vec::with_capacity(n);
            for i in 0..n {
                let item = py_params.get_item(i).expect("param item");
                let name: String = item.get_item(0).expect("name").extract().expect("name str");
                let py_tensor = item.get_item(1).expect("tensor").unbind();
                let tensor = py_to_tensor(&py_tensor);
                eprintln!("[ssm_ref] param {i}: {name} shape={:?}", tensor.size());
                let var_name = name.replace('.', "/");
                let mut var = p.var(&var_name, &tensor.size(), tch::nn::Init::Const(0.0));
                tch::no_grad(|| { let _ = var.copy_(&tensor); });
                params.push((name, var));
            }
            Self::replace_python_params(py, &bridge, handle, &params);

            Self { handle, bridge, params, config }
        })
    }

    fn replace_python_params(
        py: Python<'_>,
        bridge: &PyObject,
        handle: i64,
        params: &[(String, Tensor)],
    ) {
        for (name, rust_tensor) in params {
            let py_tensor = tensor_to_py(py, rust_tensor);
            bridge
                .call_method1(py, "set_param_tensor", (handle, name.as_str(), py_tensor))
                .expect("set_param_tensor failed");
        }
    }

    pub fn set_train(&self, mode: bool) {
        Python::with_gil(|py| {
            self.bridge.bind(py)
                .call_method1(intern!(py, "set_train"), (self.handle, mode))
                .expect("set_train failed");
        });
    }

    /// Training forward: full sequence, no state
    pub fn forward_with_pre_norm_seq_idx(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        norm_eps: f64,
        dt_scale: Option<&Tensor>,
        seq_idx: Option<&Tensor>,
    ) -> Tensor {
        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);
            let result = bridge.call_method1(
                intern!(py, "forward_with_pre_norm"),
                (
                    self.handle,
                    tensor_to_py(py, x),
                    tensor_to_py(py, norm_weight),
                    norm_eps,
                    opt_tensor_to_py(py, dt_scale),
                    opt_tensor_to_py(py, seq_idx),
                ),
            ).expect("forward_with_pre_norm failed");

            py_to_tensor(&result.unbind())
        })
    }

    /// Inference forward: full sequence, captures final state
    pub fn forward_with_state_pre_norm_dt_scale(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        norm_eps: f64,
        state: &mut Mamba2State,
        dt_scale: Option<&Tensor>,
    ) -> Tensor {
        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);
            let result = bridge.call_method1(
                intern!(py, "forward_with_pre_norm_stateful"),
                (
                    self.handle,
                    tensor_to_py(py, x),
                    tensor_to_py(py, norm_weight),
                    norm_eps,
                    tensor_to_py(py, &state.conv_state),
                    tensor_to_py(py, &state.ssm_state),
                    opt_tensor_to_py(py, dt_scale),
                ),
            ).expect("forward_with_pre_norm_stateful failed");

            let ptr = result.as_ptr();
            let y = unsafe { tuple_get_tensor(ptr, 0) };
            let new_conv = unsafe { tuple_get_tensor(ptr, 1) };
            let new_ssm = unsafe { tuple_get_tensor(ptr, 2) };

            tch::no_grad(|| {
                let _ = state.conv_state.copy_(&new_conv);
                let _ = state.ssm_state.copy_(&new_ssm);
            });
            state.has_conv_state = true;
            y
        })
    }

    /// Single-step inference with state
    pub fn step_with_pre_norm_dt_scale(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        norm_eps: f64,
        state: &mut Mamba2State,
        dt_scale: f64,
    ) -> Tensor {
        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);
            let result = bridge.call_method1(
                intern!(py, "step"),
                (
                    self.handle,
                    tensor_to_py(py, x),
                    tensor_to_py(py, norm_weight),
                    norm_eps,
                    tensor_to_py(py, &state.conv_state),
                    tensor_to_py(py, &state.ssm_state),
                    dt_scale,
                ),
            ).expect("step failed");

            let ptr = result.as_ptr();
            let y = unsafe { tuple_get_tensor(ptr, 0) };
            let new_conv = unsafe { tuple_get_tensor(ptr, 1) };
            let new_ssm = unsafe { tuple_get_tensor(ptr, 2) };

            tch::no_grad(|| {
                let _ = state.conv_state.copy_(&new_conv);
                let _ = state.ssm_state.copy_(&new_ssm);
            });
            state.has_conv_state = true;
            y
        })
    }

    /// Initialize inference state
    pub fn init_state(&self, batch_size: i64, device: tch::Device) -> Mamba2State {
        Python::with_gil(|py| {
            let bridge = self.bridge.bind(py);
            let result = bridge.call_method1(
                intern!(py, "init_state"),
                (self.handle, batch_size, device_str(device), "float32"),
            ).expect("init_state failed");

            let ptr = result.as_ptr();
            Mamba2State {
                conv_state: unsafe { tuple_get_tensor(ptr, 0) },
                ssm_state: unsafe { tuple_get_tensor(ptr, 1) },
                has_conv_state: false,
            }
        })
    }
}

impl Drop for Mamba2Ref {
    fn drop(&mut self) {
        Python::with_gil(|py| {
            let _ = self.bridge.bind(py).call_method1("destroy_layer", (self.handle,));
        });
    }
}

/// Stateful wrapper matching the `StatefulMamba` interface from ssm.rs
pub struct StatefulMambaRef {
    mamba: Mamba2Ref,
}

impl StatefulMambaRef {
    pub fn new(p: &nn::Path, config: Mamba2Config) -> Self {
        Self { mamba: Mamba2Ref::new(p, config) }
    }

    pub fn forward_with_pre_norm_seq_idx(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        norm_eps: f64,
        dt_scale: Option<&Tensor>,
        seq_idx: Option<&Tensor>,
    ) -> Tensor {
        self.mamba.forward_with_pre_norm_seq_idx(x, norm_weight, norm_eps, dt_scale, seq_idx)
    }

    pub fn forward_with_state_pre_norm_dt_scale(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        norm_eps: f64,
        state: &mut Mamba2State,
        dt_scale: Option<&Tensor>,
    ) -> Tensor {
        self.mamba.forward_with_state_pre_norm_dt_scale(x, norm_weight, norm_eps, state, dt_scale)
    }

    pub fn step_with_pre_norm_dt_scale(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        norm_eps: f64,
        state: &mut Mamba2State,
        dt_scale: f64,
    ) -> Tensor {
        self.mamba.step_with_pre_norm_dt_scale(x, norm_weight, norm_eps, state, dt_scale)
    }

    pub fn init_state(&self, batch_size: i64, device: tch::Device) -> Mamba2State {
        self.mamba.init_state(batch_size, device)
    }

    pub fn set_train(&self, mode: bool) {
        self.mamba.set_train(mode);
    }
}

/// Factory matching `stateful_mamba_block_cfg` from ssm.rs
pub fn stateful_mamba_block_cfg(p: &nn::Path, config: Mamba2Config) -> StatefulMambaRef {
    StatefulMambaRef::new(p, config)
}
