//! Mamba 2 SSM reference implementation bridge
//!
//! Delegates to the Python mamba-ssm reference via cpython FFI.
//! Provides the same `StatefulMambaRef` interface as `StatefulMamba` in ssm.rs.
//!
//! Zero-copy tensor sharing: both tch-rs and Python torch link the same libtorch,
//! so `pyobject_wrap()`/`pyobject_unpack()` share the underlying `at::Tensor`.

use std::sync::LazyLock;

use cpython::{ObjectProtocol, PyClone, PyModule, PyObject, PyTuple, Python, PythonObject, ToPyObject};
use tch::{nn, Kind, Tensor};

pub use super::ssm::{Mamba2Config, Mamba2State};

static BRIDGE_DIR: LazyLock<String> = LazyLock::new(|| {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_string());
    format!("{}/mamba2_ref", manifest)
});

static BRIDGE_INIT: LazyLock<()> = LazyLock::new(|| {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let sys = PyModule::import(py, "sys").expect("failed to import sys");
    let path: PyObject = sys.get(py, "path").expect("no sys.path");
    path.call_method(py, "insert", (0, &*BRIDGE_DIR), None)
        .expect("failed to insert bridge dir into sys.path");
});

fn ensure_bridge_init() {
    let _ = *BRIDGE_INIT;
}

fn device_str(device: tch::Device) -> &'static str {
    match device {
        tch::Device::Cpu => "cpu",
        tch::Device::Cuda(i) => Box::leak(format!("cuda:{}", i).into_boxed_str()),
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
// Zero-copy tensor conversion via pyobject_wrap / pyobject_unpack
// ---------------------------------------------------------------------------

/// Wrap a tch::Tensor as a Python torch.Tensor (zero-copy, same storage).
fn tensor_to_pyobj(py: Python, t: &Tensor) -> PyObject {
    let raw = t.pyobject_wrap().expect("pyobject_wrap failed");
    unsafe { PyObject::from_owned_ptr(py, raw as *mut _) }
}

/// Wrap an optional tensor; returns Python None if absent.
fn opt_tensor_to_pyobj(py: Python, t: Option<&Tensor>) -> PyObject {
    match t {
        Some(tensor) => tensor_to_pyobj(py, tensor),
        None => py.None(),
    }
}

/// Extract a tch::Tensor from a Python torch.Tensor (zero-copy, same storage).
fn pyobj_to_tensor(py: Python, obj: &PyObject) -> Tensor {
    let raw = obj.as_ptr() as *mut tch::python::CPyObject;
    unsafe { Tensor::pyobject_unpack(raw) }
        .expect("pyobject_unpack failed")
        .expect("object is not a torch.Tensor")
}

/// Extract a tuple element by index using __getitem__ (avoids PyTuple ABI issues).
fn tuple_get(py: Python, tup: &PyObject, idx: i32) -> PyObject {
    tup.call_method(py, "__getitem__", (idx,), None)
        .expect("tuple __getitem__ failed")
}

// ---------------------------------------------------------------------------
// Core bridge
// ---------------------------------------------------------------------------

pub struct Mamba2Ref {
    handle: i64,
    bridge: PyObject,
    /// (python_name, tensor) — tensor is the *same* storage as Python's parameter
    params: Vec<(String, Tensor)>,
    pub config: Mamba2Config,
}

impl Mamba2Ref {
    pub fn new(p: &nn::Path, config: Mamba2Config) -> Self {
        ensure_bridge_init();

        let gil = Python::acquire_gil();
        let py = gil.python();
        let bridge_mod =
            PyModule::import(py, "bridge").expect("failed to import mamba2_ref bridge");
        let bridge: PyObject = bridge_mod.as_object().clone_ref(py);

        let d_ssm_py: PyObject = match config.d_ssm {
            Some(v) => v.to_py_object(py).into_object(),
            None => py.None(),
        };

        let device = p.device();
        let kind = Kind::Float;

        let create_args = PyTuple::new(
            py,
            &[
                config.d_model.to_py_object(py).into_object(),
                config.d_state.to_py_object(py).into_object(),
                config.d_conv.to_py_object(py).into_object(),
                config.expand.to_py_object(py).into_object(),
                config.headdim.to_py_object(py).into_object(),
                d_ssm_py,
                config.ngroups.to_py_object(py).into_object(),
                config.chunk_size.to_py_object(py).into_object(),
                config.dt_min.to_py_object(py).into_object(),
                config.dt_max.to_py_object(py).into_object(),
                config.norm_before_gate.to_py_object(py).into_object(),
                config.d_has_hdim.to_py_object(py).into_object(),
                device_str(device).to_py_object(py).into_object(),
                kind_str(kind).to_py_object(py).into_object(),
            ],
        );
        let handle_obj = bridge
            .call_method(py, "create_layer", create_args, None)
            .expect("create_layer failed");
        let handle: i64 = handle_obj.extract(py).expect("handle not i64");

        // Extract Python parameters as shared tensors (zero-copy).
        // Register in VarStore so Rust optimizer can update them.
        let py_params = bridge
            .call_method(py, "get_named_parameters", (handle,), None)
            .expect("get_named_parameters failed");
        let n: usize = py_params
            .call_method(py, "__len__", cpython::NoArgs, None)
            .expect("len")
            .extract(py)
            .expect("len usize");

        let mut params = Vec::with_capacity(n);
        for i in 0..n {
            let item = tuple_get(py, &py_params, i as i32);
            let name: String = tuple_get(py, &item, 0).extract(py).expect("name");
            let py_tensor = tuple_get(py, &item, 1);
            // Zero-copy: unwrap the Python tensor into a tch::Tensor sharing the same storage
            let tensor = pyobj_to_tensor(py, &py_tensor);
            eprintln!("[ssm_ref] param {}: {} shape={:?}", i, name, tensor.size());
            // Register in VarStore with '/' separator
            let var_name = name.replace('.', "/");
            let mut var = p.var(&var_name, &tensor.size(), tch::nn::Init::Const(0.0));
            // Copy Python's init values into the VarStore tensor
            tch::no_grad(|| { let _ = var.copy_(&tensor); });
            params.push((name, var));
        }
        // Now point Python's parameters to the VarStore tensors
        // so optimizer updates are visible to Python.
        Self::replace_python_params(py, &bridge, handle, &params);

        Self { handle, bridge, params, config }
    }

    /// Replace Python layer's parameter data with the Rust VarStore tensors.
    /// After this, Python reads/writes the same storage the Rust optimizer updates.
    fn replace_python_params(py: Python, bridge: &PyObject, handle: i64, params: &[(String, Tensor)]) {
        for (name, rust_tensor) in params {
            let py_tensor = tensor_to_pyobj(py, rust_tensor);
            bridge
                .call_method(py, "set_param_tensor", (handle, name.as_str(), py_tensor), None)
                .expect("set_param_tensor failed");
        }
    }

    pub fn set_train(&self, mode: bool) {
        let gil = Python::acquire_gil();
        let py = gil.python();
        self.bridge
            .call_method(py, "set_train", (self.handle, mode), None)
            .expect("set_train failed");
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
        let gil = Python::acquire_gil();
        let py = gil.python();

        let result = self
            .bridge
            .call_method(
                py,
                "forward_with_pre_norm",
                (
                    self.handle,
                    tensor_to_pyobj(py, x),
                    tensor_to_pyobj(py, norm_weight),
                    norm_eps,
                    opt_tensor_to_pyobj(py, dt_scale),
                    opt_tensor_to_pyobj(py, seq_idx),
                ),
                None,
            )
            .expect("forward_with_pre_norm failed");

        pyobj_to_tensor(py, &result)
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
        let gil = Python::acquire_gil();
        let py = gil.python();

        let result = self
            .bridge
            .call_method(
                py,
                "forward_with_pre_norm_stateful",
                (
                    self.handle,
                    tensor_to_pyobj(py, x),
                    tensor_to_pyobj(py, norm_weight),
                    norm_eps,
                    tensor_to_pyobj(py, &state.conv_state),
                    tensor_to_pyobj(py, &state.ssm_state),
                    opt_tensor_to_pyobj(py, dt_scale),
                ),
                None,
            )
            .expect("forward_with_pre_norm_stateful failed");

        let y = pyobj_to_tensor(py, &tuple_get(py, &result, 0));
        let new_conv = pyobj_to_tensor(py, &tuple_get(py, &result, 1));
        let new_ssm = pyobj_to_tensor(py, &tuple_get(py, &result, 2));

        tch::no_grad(|| {
            let _ = state.conv_state.copy_(&new_conv);
            let _ = state.ssm_state.copy_(&new_ssm);
        });
        state.has_conv_state = true;
        y
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
        let gil = Python::acquire_gil();
        let py = gil.python();

        let result = self
            .bridge
            .call_method(
                py,
                "step",
                (
                    self.handle,
                    tensor_to_pyobj(py, x),
                    tensor_to_pyobj(py, norm_weight),
                    norm_eps,
                    tensor_to_pyobj(py, &state.conv_state),
                    tensor_to_pyobj(py, &state.ssm_state),
                    dt_scale,
                ),
                None,
            )
            .expect("step failed");

        let y = pyobj_to_tensor(py, &tuple_get(py, &result, 0));
        let new_conv = pyobj_to_tensor(py, &tuple_get(py, &result, 1));
        let new_ssm = pyobj_to_tensor(py, &tuple_get(py, &result, 2));

        tch::no_grad(|| {
            let _ = state.conv_state.copy_(&new_conv);
            let _ = state.ssm_state.copy_(&new_ssm);
        });
        state.has_conv_state = true;
        y
    }

    /// Initialize inference state
    pub fn init_state(&self, batch_size: i64, device: tch::Device) -> Mamba2State {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let result = self
            .bridge
            .call_method(
                py,
                "init_state",
                (self.handle, batch_size, device_str(device), "float32"),
                None,
            )
            .expect("init_state failed");

        let conv_state = pyobj_to_tensor(py, &tuple_get(py, &result, 0));
        let ssm_state = pyobj_to_tensor(py, &tuple_get(py, &result, 1));

        Mamba2State {
            conv_state,
            ssm_state,
            has_conv_state: false,
        }
    }
}

impl Drop for Mamba2Ref {
    fn drop(&mut self) {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let _ = self
            .bridge
            .call_method(py, "destroy_layer", (self.handle,), None);
    }
}

/// Stateful wrapper matching the `StatefulMamba` interface from ssm.rs
pub struct StatefulMambaRef {
    mamba: Mamba2Ref,
}

impl StatefulMambaRef {
    pub fn new(p: &nn::Path, config: Mamba2Config) -> Self {
        Self {
            mamba: Mamba2Ref::new(p, config),
        }
    }

    pub fn forward_with_pre_norm_seq_idx(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        norm_eps: f64,
        dt_scale: Option<&Tensor>,
        seq_idx: Option<&Tensor>,
    ) -> Tensor {
        self.mamba
            .forward_with_pre_norm_seq_idx(x, norm_weight, norm_eps, dt_scale, seq_idx)
    }

    pub fn forward_with_state_pre_norm_dt_scale(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        norm_eps: f64,
        state: &mut Mamba2State,
        dt_scale: Option<&Tensor>,
    ) -> Tensor {
        self.mamba
            .forward_with_state_pre_norm_dt_scale(x, norm_weight, norm_eps, state, dt_scale)
    }

    pub fn step_with_pre_norm_dt_scale(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        norm_eps: f64,
        state: &mut Mamba2State,
        dt_scale: f64,
    ) -> Tensor {
        self.mamba
            .step_with_pre_norm_dt_scale(x, norm_weight, norm_eps, state, dt_scale)
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
