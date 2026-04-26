use std::ffi::CStr;
use std::ptr::NonNull;

use tch::Device;
use torch_sys::C_cuda_graph;

pub struct CudaGraph {
    ptr: NonNull<C_cuda_graph>,
    device_index: i64,
}

impl CudaGraph {
    pub fn is_available() -> bool {
        unsafe { torch_sys::at_cuda_graph_is_available() }
    }

    pub fn new(device: Device) -> Self {
        let device_index = cuda_device_index(device);
        let ptr = unsafe { torch_sys::at_cuda_graph_new() };
        check_torch_error();
        let ptr = NonNull::new(ptr).expect("failed to allocate CUDA graph");
        Self { ptr, device_index }
    }

    pub fn capture_begin(&mut self) {
        unsafe { torch_sys::at_cuda_graph_capture_begin(self.ptr.as_ptr(), self.device_index) };
        check_torch_error();
    }

    pub fn capture_end(&mut self) {
        unsafe { torch_sys::at_cuda_graph_capture_end(self.ptr.as_ptr()) };
        check_torch_error();
    }

    pub fn replay(&mut self) {
        unsafe { torch_sys::at_cuda_graph_replay(self.ptr.as_ptr(), self.device_index) };
        check_torch_error();
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe { torch_sys::at_cuda_graph_free(self.ptr.as_ptr()) };
        if let Some(msg) = take_torch_error() {
            eprintln!("CUDA graph free failed: {msg}");
        }
    }
}

pub fn synchronize_device(device: Device) {
    if let Device::Cuda(index) = device {
        unsafe { torch_sys::cuda::atc_synchronize(index as i64) };
        check_torch_error();
    }
}

pub fn empty_cuda_cache() {
    unsafe { torch_sys::at_cuda_empty_cache() };
    check_torch_error();
}

fn cuda_device_index(device: Device) -> i64 {
    match device {
        Device::Cuda(index) => index as i64,
        _ => panic!("CUDA graph requires a CUDA device"),
    }
}

fn check_torch_error() {
    if let Some(msg) = take_torch_error() {
        panic!("CUDA graph error: {msg}");
    }
}

fn take_torch_error() -> Option<String> {
    let ptr = unsafe { torch_sys::get_and_reset_last_err() };
    if ptr.is_null() {
        return None;
    }
    let msg = unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() };
    unsafe { libc::free(ptr as *mut libc::c_void) };
    Some(msg)
}
