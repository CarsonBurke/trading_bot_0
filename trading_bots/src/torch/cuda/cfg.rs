use std::{marker::PhantomData, rc::Rc, sync::Once};

use tch::Cuda;

static CONFIGURE: Once = Once::new();

/// CUDA configuration: thread-local bf16 autocast plus process-wide SDP backends.
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
    pin_bfloat16_autocast();
    CONFIGURE.call_once(|| {
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
                "failed to configure SDPA backends: flash={flash} mem={mem} math={math} cudnn={cudnn}"
            );
        }

        println!("CUDA configured: autocast bf16, SDPA flash only");
    });
}

/// Pin ATen's thread-local CUDA autocast dtype before entering an autocast scope.
pub(crate) fn pin_bfloat16_autocast() {
    if !Cuda::is_available() {
        return;
    }
    unsafe {
        torch_sys::at_autocast_set_bfloat16();
        assert!(
            torch_sys::at_autocast_is_bfloat16() != 0,
            "failed to pin CUDA autocast to bf16"
        );
    }
}

/// Keep single-GPU pretraining backward scheduling on the caller thread.
///
/// PyTorch's indexed device-ready queues can reject valid CUDA device metadata after an
/// ordinary CUDA backward has initialized the engine. Fresh and resumed world-model graphs
/// have both reached that invalid ordinal path. Single-GPU pretraining does not need
/// cross-device autograd workers; the caller's ready queue preserves the CUDA kernels.
#[must_use]
pub(crate) struct AutogradMultithreadingGuard {
    was_enabled: bool,
    _thread_bound: PhantomData<Rc<()>>,
}

impl Drop for AutogradMultithreadingGuard {
    fn drop(&mut self) {
        let restored = unsafe {
            torch_sys::at_autograd_set_multithreading_enabled(i32::from(self.was_enabled))
        };
        assert!(
            restored >= 0 || std::thread::panicking(),
            "failed to restore PyTorch autograd multithreading"
        );
    }
}

pub(crate) fn disable_autograd_multithreading() -> AutogradMultithreadingGuard {
    let was_enabled = unsafe { torch_sys::at_autograd_set_multithreading_enabled(0) };
    assert!(
        was_enabled >= 0,
        "failed to disable PyTorch autograd multithreading"
    );
    println!("PyTorch autograd multithreading disabled for single-GPU pretraining");
    AutogradMultithreadingGuard {
        was_enabled: was_enabled != 0,
        _thread_bound: PhantomData,
    }
}
