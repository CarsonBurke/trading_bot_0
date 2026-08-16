use std::sync::Once;

use tch::Cuda;

static CONFIGURE: Once = Once::new();

/// One-time CUDA configuration: autocast dtype, SDP backends, cuDNN autotuner.
///
/// The autocast dtype matters more than it looks. `tch::autocast(true, ..)` toggles
/// only ATen's enable flag; the dtype stays at ATen's CUDA default, which this
/// toolchain reports as fp16. There is no gradient scaler anywhere in this crate, so
/// fp16 silently flushes small gradients to zero instead of failing, and `half +
/// bfloat16` promotes to `float32`, so fp16 linears meeting the explicitly-bf16
/// attention kernels were upcasting the residual stream on every layer — paying fp32
/// bandwidth while appearing to be in mixed precision. bf16 carries fp32's exponent
/// range, which is exactly why it needs no scaler.
pub fn configure_cuda() {
    CONFIGURE.call_once(|| {
        if !Cuda::is_available() {
            return;
        }

        unsafe {
            torch_sys::at_autocast_set_bfloat16();
            assert!(
                torch_sys::at_autocast_is_bfloat16() != 0,
                "failed to pin CUDA autocast to bf16"
            );

            // SDP backends
            torch_sys::at_sdp_set_use_flash(1);
            torch_sys::at_sdp_set_use_mem_efficient(0);
            torch_sys::at_sdp_set_use_math(0);
            torch_sys::at_sdp_set_use_cudnn(0);

            let flash = torch_sys::at_sdp_use_flash() != 0;
            let mem = torch_sys::at_sdp_use_mem_efficient() != 0;
            let math = torch_sys::at_sdp_use_math() != 0;
            let cudnn = torch_sys::at_sdp_use_cudnn() != 0;
            assert!(
                flash && !mem && !math && !cudnn,
                "failed to configure SDPA backends: flash={flash} mem={mem} math={math} cudnn={cudnn}"
            );
        }

        println!("CUDA configured: autocast bf16, SDPA flash only");
    });
}
