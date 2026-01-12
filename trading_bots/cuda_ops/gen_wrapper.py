import os
import torch
from torch.jit import export

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
        seq_idx: torch.Tensor,
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
            seq_idx,
            chunk_size,
            ngroups,
            headdim,
            dt_min,
            dt_max,
        )

    @export
    def forward_infer(
        self,
        zxbcdt: torch.Tensor,
        conv_w: torch.Tensor,
        conv_b: torch.Tensor,
        dt_bias: torch.Tensor,
        a_log: torch.Tensor,
        d_param: torch.Tensor,
        dt_scale: torch.Tensor,
        initial_state: torch.Tensor,
        seq_idx: torch.Tensor,
        chunk_size: int,
        ngroups: int,
        headdim: int,
        dt_min: float,
        dt_max: float,
        rmsnorm_weight: torch.Tensor,
        rmsnorm_eps: float,
        norm_before_gate: bool,
        outproj_w: torch.Tensor,
        outproj_b: torch.Tensor,
    ):
        return torch.ops.mamba_fused.fused_conv_scan_infer(
            zxbcdt,
            conv_w,
            conv_b,
            dt_bias,
            a_log,
            d_param,
            dt_scale,
            initial_state,
            seq_idx,
            chunk_size,
            ngroups,
            headdim,
            dt_min,
            dt_max,
            rmsnorm_weight,
            rmsnorm_eps,
            norm_before_gate,
            outproj_w,
            outproj_b,
        )

    @export
    def forward_full(
        self,
        zxbcdt: torch.Tensor,
        conv_w: torch.Tensor,
        conv_b: torch.Tensor,
        dt_bias: torch.Tensor,
        a_log: torch.Tensor,
        d_param: torch.Tensor,
        dt_scale: torch.Tensor,
        initial_state: torch.Tensor,
        seq_idx: torch.Tensor,
        chunk_size: int,
        ngroups: int,
        headdim: int,
        dt_min: float,
        dt_max: float,
        rmsnorm_weight: torch.Tensor,
        rmsnorm_eps: float,
        norm_before_gate: bool,
        outproj_w: torch.Tensor,
        outproj_b: torch.Tensor,
    ):
        return torch.ops.mamba_fused.fused_conv_scan_full(
            zxbcdt,
            conv_w,
            conv_b,
            dt_bias,
            a_log,
            d_param,
            dt_scale,
            initial_state,
            seq_idx,
            chunk_size,
            ngroups,
            headdim,
            dt_min,
            dt_max,
            rmsnorm_weight,
            rmsnorm_eps,
            norm_before_gate,
            outproj_w,
            outproj_b,
        )

    @export
    def forward_full_graph(
        self,
        zxbcdt: torch.Tensor,
        conv_w: torch.Tensor,
        conv_b: torch.Tensor,
        dt_bias: torch.Tensor,
        a_log: torch.Tensor,
        d_param: torch.Tensor,
        dt_scale: torch.Tensor,
        initial_state: torch.Tensor,
        seq_idx: torch.Tensor,
        chunk_size: int,
        ngroups: int,
        headdim: int,
        dt_min: float,
        dt_max: float,
        rmsnorm_weight: torch.Tensor,
        rmsnorm_eps: float,
        norm_before_gate: bool,
        outproj_w: torch.Tensor,
        outproj_b: torch.Tensor,
    ):
        return torch.ops.mamba_fused.fused_conv_scan_full_graph(
            zxbcdt,
            conv_w,
            conv_b,
            dt_bias,
            a_log,
            d_param,
            dt_scale,
            initial_state,
            seq_idx,
            chunk_size,
            ngroups,
            headdim,
            dt_min,
            dt_max,
            rmsnorm_weight,
            rmsnorm_eps,
            norm_before_gate,
            outproj_w,
            outproj_b,
        )

    @export
    def forward_stateful(
        self,
        zxbcdt: torch.Tensor,
        conv_w: torch.Tensor,
        conv_b: torch.Tensor,
        dt_bias: torch.Tensor,
        a_log: torch.Tensor,
        d_param: torch.Tensor,
        dt_scale: torch.Tensor,
        initial_state: torch.Tensor,
        conv_state: torch.Tensor,
        seq_idx: torch.Tensor,
        chunk_size: int,
        ngroups: int,
        headdim: int,
        dt_min: float,
        dt_max: float,
        rmsnorm_weight: torch.Tensor,
        rmsnorm_eps: float,
        norm_before_gate: bool,
        outproj_w: torch.Tensor,
        outproj_b: torch.Tensor,
    ):
        return torch.ops.mamba_fused.fused_conv_scan_stateful(
            zxbcdt,
            conv_w,
            conv_b,
            dt_bias,
            a_log,
            d_param,
            dt_scale,
            initial_state,
            conv_state,
            seq_idx,
            chunk_size,
            ngroups,
            headdim,
            dt_min,
            dt_max,
            rmsnorm_weight,
            rmsnorm_eps,
            norm_before_gate,
            outproj_w,
            outproj_b,
        )

    @export
    def selective_state_update(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        a_log: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d_param: torch.Tensor,
        z: torch.Tensor,
        dt_bias: torch.Tensor,
        dt_softplus: bool,
        dt_min: float,
        dt_max: float,
        ngroups: int,
        headdim: int,
        apply_dt_limit: int,
    ):
        return torch.ops.mamba_fused.selective_state_update(
            state,
            x,
            dt,
            a_log,
            b,
            c,
            d_param,
            z,
            dt_bias,
            dt_softplus,
            dt_min,
            dt_max,
            ngroups,
            headdim,
            apply_dt_limit,
        )

    @export
    def cuda_empty_cache(self):
        torch.ops.mamba_fused.cuda_empty_cache()

    @export
    def cuda_memory_stats(self) -> list[int]:
        return torch.ops.mamba_fused.cuda_memory_stats()

    @export
    def rmsnorm_forward(self, x: torch.Tensor, weight: torch.Tensor, eps: float):
        return torch.ops.mamba_fused.rmsnorm_forward(x, weight, eps)

module = torch.jit.script(MambaFusedWrapper())
module.save(out_path)
