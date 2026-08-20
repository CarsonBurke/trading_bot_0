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
