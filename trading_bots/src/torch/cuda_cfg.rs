use std::sync::Once;

use tch::Cuda;

static CONFIGURE: Once = Once::new();

/// One-time CUDA configuration: SDP backends, cuDNN autotuner, JIT CUDA fuser.
pub fn configure_cuda() {
    CONFIGURE.call_once(|| {
        if !Cuda::is_available() {
            return;
        }

        unsafe {
            // SDP backends
            torch_sys::at_sdp_set_use_flash(1);
            torch_sys::at_sdp_set_use_mem_efficient(1);
            torch_sys::at_sdp_set_use_math(1);
            torch_sys::at_sdp_set_use_cudnn(0);

            let flash = torch_sys::at_sdp_use_flash() != 0;
            let mem = torch_sys::at_sdp_use_mem_efficient() != 0;
            let math = torch_sys::at_sdp_use_math() != 0;
            let cudnn = torch_sys::at_sdp_use_cudnn() != 0;
            assert!(
                flash && mem && math && !cudnn,
                "failed to configure SDPA backends: flash={flash} mem={mem} math={math} cudnn={cudnn}"
            );
        }

        println!("CUDA configured: SDPA flash+mem+math");
    });
}
