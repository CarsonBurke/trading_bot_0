use anyhow::{anyhow, bail, Context, Result};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;
use std::ffi::CString;
use std::sync::Mutex;
use tch::{Kind, Tensor};

use super::pope::{PolarQk, POPE_ATTENTION_SCALE, POPE_DIM, POPE_QK_DIM};

static FA4_CALL_LOCK: Mutex<()> = Mutex::new(());
static FA4_SERIALIZED_CALL: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AttentionMode {
    CausalPrefill,
    DecodeQ1,
}

pub fn is_available() -> bool {
    let Ok(_call_guard) = FA4_CALL_LOCK.lock() else {
        return false;
    };
    Python::attach(|py| py.import("torch").is_ok() && py.import("flash_attn.cute").is_ok())
}

/// FA4 causal prefill over contiguous BSHD QK128/K128/V64 tensors.
pub fn pope_flash_attention_prefill(qk: &PolarQk, value_bshd: &Tensor) -> Result<Tensor> {
    call_fa4(&qk.query, &qk.key, value_bshd, AttentionMode::CausalPrefill)
}

/// FA4 cached decode with Q sequence length exactly one.
///
/// K/V must be contiguous and aligned. Their physical order is irrelevant for
/// non-causal Q=1 decode because PoPE has already embedded absolute key phases.
pub fn pope_flash_attention_decode_q1(
    query_b1hd: &Tensor,
    key_cache_bshd: &Tensor,
    value_cache_bshd: &Tensor,
) -> Result<Tensor> {
    call_fa4(
        query_b1hd,
        key_cache_bshd,
        value_cache_bshd,
        AttentionMode::DecodeQ1,
    )
}

fn call_fa4(query: &Tensor, key: &Tensor, value: &Tensor, mode: AttentionMode) -> Result<Tensor> {
    validate(query, key, value, mode)?;
    // CuTe DSL compilation and its AST-preprocessor session are process-global
    // and not thread-safe even across separately GIL-protected callers.
    let _call_guard = FA4_CALL_LOCK
        .lock()
        .map_err(|_| anyhow!("FA4 process-global call lock was poisoned"))?;
    let output = Python::attach(|py| -> Result<Tensor> {
        py.import("torch")
            .map_err(|error| anyhow!("failed to initialize Python torch: {error:?}"))?;
        let module = py
            .import("flash_attn.cute")
            .map_err(|error| anyhow!("failed to import flash_attn.cute: {error:?}"))?;
        let function = module
            .getattr("flash_attn_func")
            .map_err(|error| anyhow!("flash_attn_func is unavailable: {error:?}"))?;
        let serialized_function = serialized_python_call(py)?;

        let query_object = tensor_object(py, query).context("wrapping FA4 query")?;
        let key_object = tensor_object(py, key).context("wrapping FA4 key")?;
        let value_object = tensor_object(py, value).context("wrapping FA4 value")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("softmax_scale", POPE_ATTENTION_SCALE)?;
        kwargs.set_item("causal", mode == AttentionMode::CausalPrefill)?;
        kwargs.set_item("pack_gqa", true)?;

        let result = serialized_function
            .call(
                (function, query_object, key_object, value_object),
                Some(&kwargs),
            )
            .map_err(|error| anyhow!("FA4 execution failed: {error:?}"))?;
        let output_object = result
            .get_item(0)
            .map_err(|error| anyhow!("FA4 result has no tensor output: {error:?}"))?;
        unsafe { Tensor::pyobject_unpack(output_object.as_ptr().cast()) }
            .context("unwrapping FA4 output")?
            .ok_or_else(|| anyhow!("FA4 output is not a torch Tensor"))
    })?;
    let expected = [query.size()[0], query.size()[1], query.size()[2], POPE_DIM];
    if output.size() != expected {
        bail!(
            "FA4 output shape {:?}, expected {expected:?}",
            output.size()
        );
    }
    Ok(output)
}

fn serialized_python_call<'py>(py: Python<'py>) -> Result<Bound<'py, PyAny>> {
    // The CuTe AST-preprocessor session releases the GIL and is not thread-safe.
    // This Python lock spans each forward and, through node hooks, its eventual
    // FlashAttnFunc backward as well.
    let helper = FA4_SERIALIZED_CALL.get_or_try_init(py, || -> Result<Py<PyAny>> {
        let source = CString::new(
            r#"
import threading as _rust_fa4_threading
if "_RUST_FA4_LOCK" not in globals():
    _RUST_FA4_LOCK = _rust_fa4_threading.Lock()
if "_rust_fa4_serialized_call" not in globals():
    def _rust_fa4_serialized_call(fn, q, k, v, **kwargs):
        with _RUST_FA4_LOCK:
            result = fn(q, k, v, **kwargs)
        output = result[0]
        node = output.grad_fn
        if node is not None:
            state = [False]
            def acquire(grad_outputs):
                _RUST_FA4_LOCK.acquire()
                state[0] = True
                return grad_outputs
            def release(grad_inputs, grad_outputs):
                if state[0]:
                    state[0] = False
                    _RUST_FA4_LOCK.release()
                return grad_inputs
            node.register_prehook(acquire)
            node.register_hook(release)
        return result
"#,
        )?;
        py.run(source.as_c_str(), None, None)
            .map_err(|error| anyhow!("installing serialized FA4 helper failed: {error:?}"))?;
        Ok(py
            .import("__main__")?
            .getattr("_rust_fa4_serialized_call")?
            .unbind())
    })?;
    Ok(helper.bind(py).clone())
}

fn tensor_object<'py>(py: Python<'py>, tensor: &Tensor) -> Result<Bound<'py, PyAny>> {
    let pointer = tensor.pyobject_wrap()?;
    Ok(unsafe { Bound::from_owned_ptr(py, pointer.cast()) })
}

fn validate(query: &Tensor, key: &Tensor, value: &Tensor, mode: AttentionMode) -> Result<()> {
    for (name, tensor) in [("query", query), ("key", key), ("value", value)] {
        if tensor.dim() != 4 {
            bail!("FA4 {name} must be rank-4 BSHD, got {:?}", tensor.size());
        }
        if !tensor.device().is_cuda() {
            bail!("FA4 {name} must be CUDA, got {:?}", tensor.device());
        }
        if !matches!(tensor.kind(), Kind::Half | Kind::BFloat16) {
            bail!("FA4 {name} must be fp16 or bf16, got {:?}", tensor.kind());
        }
        if tensor.stride()[3] != 1 {
            bail!("FA4 {name} head dimension must be contiguous");
        }
    }
    if query.kind() != key.kind() || key.kind() != value.kind() {
        bail!("FA4 Q/K/V dtypes must match");
    }
    if query.device() != key.device() || key.device() != value.device() {
        bail!("FA4 Q/K/V devices must match");
    }
    if query.size()[0] != key.size()[0] || key.size()[0] != value.size()[0] {
        bail!("FA4 Q/K/V batch sizes must match");
    }
    if query.size()[3] != POPE_QK_DIM || key.size()[3] != POPE_QK_DIM {
        bail!("FA4 PoPE Q/K widths must both be 128");
    }
    if value.size()[3] != POPE_DIM {
        bail!("FA4 PoPE V width must be 64");
    }
    if key.size()[1] != value.size()[1] || key.size()[2] != value.size()[2] {
        bail!("FA4 K/V sequence and head dimensions must match");
    }
    if query.size()[2] % key.size()[2] != 0 {
        bail!("FA4 query heads must be divisible by KV heads");
    }
    match mode {
        AttentionMode::CausalPrefill if query.size()[1] != key.size()[1] => {
            bail!("FA4 causal prefill requires equal Q/K sequence lengths")
        }
        AttentionMode::DecodeQ1 if query.size()[1] != 1 => {
            bail!("FA4 decode requires Q sequence length exactly one")
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    use crate::torch::test_rng;

    fn ready() -> bool {
        tch::Cuda::is_available() && is_available()
    }

    fn inputs(query_len: i64, key_len: i64) -> (Tensor, Tensor, Tensor) {
        let device = Device::Cuda(0);
        let q =
            Tensor::randn([1, query_len, 8, 128], (Kind::BFloat16, device)).set_requires_grad(true);
        let k =
            Tensor::randn([1, key_len, 4, 128], (Kind::BFloat16, device)).set_requires_grad(true);
        let v =
            Tensor::randn([1, key_len, 4, 64], (Kind::BFloat16, device)).set_requires_grad(true);
        (q, k, v)
    }

    fn reference(query: &Tensor, key: &Tensor, value: &Tensor, causal: bool) -> Tensor {
        let padded_value = Tensor::cat(&[value, &Tensor::zeros_like(value)], -1);
        Tensor::scaled_dot_product_attention(
            &query.transpose(1, 2),
            &key.transpose(1, 2),
            &padded_value.transpose(1, 2),
            None::<&Tensor>,
            0.0,
            causal,
            Some(POPE_ATTENTION_SCALE),
            true,
        )
        .narrow(-1, 0, 64)
        .transpose(1, 2)
    }

    #[test]
    fn prefill_forward_and_backward_smoke() -> Result<()> {
        let _torch_rng_guard = test_rng::shared();
        if !ready() {
            return Ok(());
        }
        let (q, k, v) = inputs(128, 128);
        let actual = pope_flash_attention_prefill(
            &PolarQk {
                query: q.shallow_clone(),
                key: k.shallow_clone(),
            },
            &v,
        )?;
        let expected = reference(&q, &k, &v, true);
        let max_difference = (&actual - expected).abs().max().double_value(&[]);
        assert!(
            max_difference < 0.08,
            "FA4 prefill mismatch: {max_difference}"
        );
        actual
            .to_kind(Kind::Float)
            .square()
            .mean(Kind::Float)
            .backward();
        for (name, gradient) in [("q", q.grad()), ("k", k.grad()), ("v", v.grad())] {
            assert!(gradient.defined(), "missing FA4 {name} gradient");
            assert!(
                bool::try_from(gradient.isfinite().all())?,
                "non-finite FA4 {name} gradient"
            );
        }
        Ok(())
    }

    #[test]
    fn q1_decode_matches_reference() -> Result<()> {
        let _torch_rng_guard = test_rng::shared();
        if !ready() {
            return Ok(());
        }
        let (q, k, v) = inputs(1, 97);
        let actual = pope_flash_attention_decode_q1(&q, &k, &v)?;
        let expected = reference(&q, &k, &v, false);
        let max_difference = (&actual - expected).abs().max().double_value(&[]);
        assert!(
            max_difference < 0.05,
            "FA4 decode mismatch: {max_difference}"
        );
        Ok(())
    }

    #[test]
    fn q1_decode_accepts_strided_active_cache_prefix() -> Result<()> {
        let _torch_rng_guard = test_rng::shared();
        if !ready() {
            return Ok(());
        }
        let device = Device::Cuda(0);
        let query = Tensor::randn([2, 1, 8, 128], (Kind::BFloat16, device));
        let key_storage = Tensor::randn([2, 128, 4, 128], (Kind::BFloat16, device));
        let value_storage = Tensor::randn([2, 128, 4, 64], (Kind::BFloat16, device));
        let key = key_storage.narrow(1, 0, 97);
        let value = value_storage.narrow(1, 0, 97);
        assert!(!key.is_contiguous());
        assert!(!value.is_contiguous());
        let actual = pope_flash_attention_decode_q1(&query, &key, &value)?;
        let expected = reference(&query, &key, &value, false);
        let max_difference = (&actual - expected).abs().max().double_value(&[]);
        assert!(
            max_difference < 0.05,
            "strided FA4 decode mismatch: {max_difference}"
        );
        Ok(())
    }
}
