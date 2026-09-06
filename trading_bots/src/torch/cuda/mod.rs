use std::ffi::CStr;

use tch::Tensor;

pub mod cfg;
pub(crate) mod graph;

/// Release every block the CUDA caching allocator holds but is not using, back to the driver.
///
/// Needed because every VRAM reading in this crate is DEVICE-WIDE, through NVML: the card is
/// shared, so what a training step has to fit into is what the other tenants leave, not what
/// this process believes it allocated. Torch's pool is invisible to that reading, so a
/// measurement taken while the pool still holds a previous shape's blocks attributes them to
/// the other tenants and under-reports free memory without bound. Calling this immediately
/// before a reading makes the reading mean what it says.
///
/// A no-op off CUDA, and a no-op in a torch-sys built without CUDA runtime headers.
pub fn empty_cache() {
    unsafe { torch_sys::at_cuda_empty_cache() }
}

/// Copy `src` into `dst`'s EXISTING storage, asynchronously.
///
/// `Tensor::copy_` passes ATen's `non_blocking=false`, which for a host-to-device copy
/// appends a `cudaStreamSynchronize`: the host waits for every kernel already queued,
/// which on a 300 ms step means the next step's several hundred launches are issued with
/// the device idle. `Tensor::to_device_` is asynchronous but ALLOCATES, so it cannot
/// write into the fixed address a captured graph recorded. This is the missing third
/// case: fixed destination, no drain.
///
/// The caller owes the non-blocking contract - the host side must be PINNED, so that
/// libtorch's caching host allocator holds the block until the copy retires - and both
/// tensors must already have the same shape.
pub(crate) fn copy_nonblocking(dst: &mut Tensor, src: &Tensor) -> Result<(), String> {
    debug_assert_eq!(dst.size(), src.size());
    unsafe {
        torch_sys::at_copy_nonblocking(dst.as_mut_ptr(), src.as_ptr().cast_mut());
    }
    read_torch_error()
}

/// Drain and clear libtorch's thread-local error slot left by a raw shim call.
pub(crate) fn read_torch_error() -> Result<(), String> {
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
