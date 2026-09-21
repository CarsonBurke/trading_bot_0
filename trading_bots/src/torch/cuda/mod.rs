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

/// A PINNED float32 host tensor, ALLOCATED rather than copied into place.
///
/// `Tensor::pin_memory` is a copy: it allocates the pinned block and memcpys a pageable
/// tensor into it. A producer that writes every byte of the block itself - the corpus
/// loader's packed row block, 114,131,968 bytes at batch 256 - pays a pageable allocation
/// of that size and 228 MB of single-threaded host read+write traffic for bytes it is about
/// to overwrite. This is the allocation without the copy.
///
/// The pinned allocator comes from torch's CUDA hooks, so this needs a CUDA-enabled torch;
/// callers that may run without a device must allocate pageable instead.
pub(crate) fn empty_pinned(size: &[i64]) -> Result<Tensor, String> {
    let ptr = unsafe { torch_sys::at_empty_pinned_float(size.as_ptr(), size.len() as libc::c_int) };
    read_torch_error()?;
    if ptr.is_null() {
        return Err("pinned host allocation returned no tensor".into());
    }
    Ok(unsafe { Tensor::from_ptr(ptr) })
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
