import os
import torch

lib_path = os.environ["MAMBA_FUSED_LIB_PATH"]
out_path = os.environ["MAMBA_FUSED_WRAPPER_PATH"]

torch.ops.load_library(lib_path)

class MambaFusedWrapper(torch.nn.Module):
    def forward(
        self,
        zxbcdt: torch.Tensor,
        conv_w: torch.Tensor,
        conv_b: torch.Tensor,
        dt_bias: torch.Tensor,
        a_log: torch.Tensor,
        d_param: torch.Tensor,
        dt_scale: torch.Tensor,
        initial_state: torch.Tensor,
        chunk_size: int,
        ngroups: int,
        headdim: int,
        dt_min: float,
        dt_max: float,
    ):
        return torch.ops.mamba_fused.fused_conv_scan(
            zxbcdt,
            conv_w,
            conv_b,
            dt_bias,
            a_log,
            d_param,
            dt_scale,
            initial_state,
            chunk_size,
            ngroups,
            headdim,
            dt_min,
            dt_max,
        )

module = torch.jit.script(MambaFusedWrapper())
module.save(out_path)
