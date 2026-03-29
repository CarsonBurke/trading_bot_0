use std::sync::Once;

use tch::Cuda;

static FORCE_FLASH_SDP: Once = Once::new();

pub fn force_flash_sdp() {
    FORCE_FLASH_SDP.call_once(|| {
        if !Cuda::is_available() {
            return;
        }

        unsafe {
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
                "failed to force flash-only SDPA backend: flash={flash} mem={mem} math={math} cudnn={cudnn}"
            );
        }

        println!("SDPA backend forced: flash=true mem=false math=false cudnn=false");
    });
}
