use std::ffi::CStr;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::NonNull;

use tch::Device;
use torch_sys::C_cuda_graph;

pub(crate) struct CudaGraph {
    raw: NonNull<C_cuda_graph>,
    device_index: i64,
}

impl CudaGraph {
    pub(crate) fn is_available() -> bool {
        unsafe { torch_sys::at_cuda_graph_is_available() }
    }

    pub(crate) fn new(device: Device) -> Result<Option<Self>, String> {
        let device_index = match device {
            Device::Cuda(index) => index as i64,
            _ => return Ok(None),
        };
        if !Self::is_available() {
            return Ok(None);
        }

        let raw = unsafe { torch_sys::at_cuda_graph_new() };
        read_torch_error()?;
        let raw =
            NonNull::new(raw).ok_or_else(|| "torch_sys returned a null CUDA graph".to_string())?;
        Ok(Some(Self { raw, device_index }))
    }

    pub(crate) fn capture<F>(&mut self, f: F) -> Result<(), String>
    where
        F: FnOnce(),
    {
        unsafe { torch_sys::at_cuda_graph_capture_begin(self.raw.as_ptr(), self.device_index) };
        if let Err(err) = read_torch_error() {
            self.abort_capture();
            return Err(err);
        }
        if let Err(payload) = catch_unwind(AssertUnwindSafe(f)) {
            self.abort_capture();
            return Err(format!(
                "capture body panicked: {}",
                panic_message(payload)
            ));
        }
        unsafe { torch_sys::at_cuda_graph_capture_end(self.raw.as_ptr()) };
        let end_result = read_torch_error();
        if let Err(err) = end_result {
            self.abort_capture();
            return Err(err);
        }
        Ok(())
    }

    pub(crate) fn replay(&self) -> Result<(), String> {
        unsafe { torch_sys::at_cuda_graph_replay(self.raw.as_ptr(), self.device_index) };
        read_torch_error()
    }

    fn abort_capture(&mut self) {
        unsafe { torch_sys::at_cuda_graph_capture_abort(self.raw.as_ptr()) };
        let _ = read_torch_error();
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe { torch_sys::at_cuda_graph_free(self.raw.as_ptr()) };
        let _ = read_torch_error();
    }
}

fn read_torch_error() -> Result<(), String> {
    let ptr = unsafe { torch_sys::get_and_reset_last_err() };
    if ptr.is_null() {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    unsafe { libc::free(ptr.cast()) };
    Err(message)
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}
