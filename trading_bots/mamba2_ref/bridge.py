"""Bridge module for Rust cpython interop with mamba-ssm Mamba2.

Provides a handle-based registry for creating, running, and managing Mamba2 layers
from Rust via Python FFI. All tensor I/O uses standard PyTorch tensors.
"""
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

if causal_conv1d_fn is None or causal_conv1d_update is None:
    raise RuntimeError(
        "causal_conv1d is required for the Mamba2 reference bridge. "
        "Install causal_conv1d to avoid slow Python fallbacks."
    )

if selective_state_update is None:
    raise RuntimeError(
        "selective_state_update is required for the Mamba2 reference bridge. "
        "Install mamba-ssm with triton ops to avoid slow Python fallbacks."
    )

_registry = {}  # handle_id -> Mamba2 instance
_next_id = 0


def create_layer(d_model, d_state, d_conv, expand, headdim, d_ssm,
                 ngroups, chunk_size, dt_min, dt_max,
                 norm_before_gate, D_has_hdim, device_str, dtype_str):
    """Create a Mamba2 layer and return its handle ID."""
    global _next_id
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)

    d_ssm_val = None if d_ssm is None or (isinstance(d_ssm, int) and d_ssm < 0) else d_ssm

    layer = Mamba2(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=headdim,
        d_ssm=d_ssm_val,
        ngroups=ngroups,
        chunk_size=chunk_size,
        dt_min=dt_min,
        dt_max=dt_max,
        norm_before_gate=norm_before_gate,
        D_has_hdim=D_has_hdim,
        rmsnorm=True,
        use_mem_eff_path=False,
    ).to(device=device, dtype=dtype)

    handle = _next_id
    _next_id += 1
    _registry[handle] = layer
    return handle


def set_train(handle, mode):
    """Set train/eval mode."""
    _registry[handle].train(mode)


def _ensure_layout_cache(layer):
    """Cache layout constants and rearranged weights that are fixed for the lifetime of the layer."""
    if hasattr(layer, '_layout_cache'):
        return layer._layout_cache

    cache = {}
    cache['d_mlp'] = (layer.in_proj.weight.shape[0] - 2 * layer.d_ssm
                      - 2 * layer.ngroups * layer.d_state - layer.nheads) // 2

    # Pre-rearrange conv1d weight (view, not copy)
    cache['conv1d_weight_2d'] = rearrange(layer.conv1d.weight, "d 1 w -> d w")

    # D rearrangement for scan path (_forward_inner)
    headdim = layer.headdim
    if layer.D_has_hdim:
        cache['D_scan'] = rearrange(layer.D, "(h p) -> h p", p=headdim)
    else:
        cache['D_scan'] = layer.D

    # D for step fallback path (_step_inner)
    if layer.D_has_hdim:
        cache['D_step'] = layer.D.view(1, layer.nheads, headdim)
    else:
        cache['D_step'] = layer.D.view(1, layer.nheads, 1)

    # Avoid repeated tuple comparison each call
    cache['has_dt_limit'] = layer.dt_limit != (0.0, float("inf"))
    if cache['has_dt_limit']:
        cache['dt_limit'] = layer.dt_limit

    layer._layout_cache = cache
    return cache


def invalidate_layout_cache(handle):
    """Invalidate cached layout after parameter changes (e.g. loading new weights)."""
    layer = _registry.get(handle)
    if layer is not None and hasattr(layer, '_layout_cache'):
        del layer._layout_cache


def forward_with_pre_norm(handle, x, norm_weight, norm_eps, dt_scale, seq_idx):
    """Forward pass with pre-RMSNorm. seq_idx should already be int32 from Rust."""
    layer = _registry[handle]
    normed = _rmsnorm(x, norm_weight, norm_eps)
    return _forward_inner(layer, normed, dt_scale, seq_idx,
                          return_final_states=False)


def forward_with_pre_norm_stateful(handle, x, norm_weight, norm_eps,
                                   conv_state, ssm_state, dt_scale):
    """Forward pass that also returns final conv and ssm states."""
    layer = _registry[handle]
    normed = _rmsnorm(x, norm_weight, norm_eps)
    return _forward_inner(layer, normed, dt_scale, seq_idx=None,
                          return_final_states=True)


def step(handle, x, norm_weight, norm_eps, conv_state, ssm_state, dt_scale):
    """Single-step inference."""
    layer = _registry[handle]
    normed = _rmsnorm(x, norm_weight, norm_eps)
    return _step_inner(layer, normed, conv_state, ssm_state, dt_scale)


def get_named_parameters(handle):
    """Return list of (name, tensor) for all parameters."""
    layer = _registry[handle]
    return [(name, param) for name, param in layer.named_parameters()]


def get_param_info(handle):
    """Return list of (name, shape_list) for all parameters -- no tensor data."""
    layer = _registry[handle]
    return [(name, list(param.shape)) for name, param in layer.named_parameters()]


def get_param_flat_bytes(handle, name):
    """Return flat f32 bytes for a single named parameter."""
    import struct
    layer = _registry[handle]
    param = dict(layer.named_parameters())[name]
    flat = param.detach().float().cpu().contiguous().view(-1)
    return struct.pack(f'{flat.numel()}f', *flat.tolist())


def set_param_from_flat_bytes(handle, name, flat_bytes, shape):
    """Set a parameter from flat f32 bytes."""
    import struct
    layer = _registry[handle]
    params_dict = dict(layer.named_parameters())
    param = params_dict[name]
    numel = 1
    for s in shape:
        numel *= s
    floats = struct.unpack(f'{numel}f', flat_bytes)
    new_data = torch.tensor(floats, dtype=torch.float32).reshape(shape).to(
        dtype=param.dtype, device=param.device
    )
    with torch.no_grad():
        param.copy_(new_data)


def set_param_tensor(handle, name, new_tensor):
    """Replace a parameter's data with a new tensor (zero-copy shared storage)."""
    layer = _registry[handle]
    parts = name.split('.')
    obj = layer
    for part in parts[:-1]:
        obj = getattr(obj, part)
    param = getattr(obj, parts[-1])
    with torch.no_grad():
        param.data = new_tensor
    # New storage invalidates cached views
    if hasattr(layer, '_layout_cache'):
        del layer._layout_cache


def init_state(handle, batch_size, device_str, dtype_str):
    """Initialize conv and ssm states (zeros)."""
    layer = _registry[handle]
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)

    d_ssm = layer.d_ssm
    d_conv = layer.d_conv
    d_state = layer.d_state
    nheads = layer.nheads
    headdim = layer.headdim
    ngroups = layer.ngroups

    conv_dim = d_ssm + 2 * ngroups * d_state
    conv_state = torch.zeros(batch_size, conv_dim, d_conv,
                             device=device, dtype=dtype)
    ssm_state = torch.zeros(batch_size, nheads, headdim, d_state,
                            device=device, dtype=torch.float32)
    return conv_state, ssm_state


def destroy_layer(handle):
    """Remove layer from registry."""
    _registry.pop(handle, None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rmsnorm(x, weight, eps):
    """RMSNorm matching Rust implementation."""
    if x.dtype == torch.float32:
        rms = (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
        return x * rms * weight
    dtype = x.dtype
    x_f32 = x.float()
    rms = (x_f32.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    return (x_f32 * rms * weight.float()).to(dtype)


def _forward_inner(layer, x, dt_scale, seq_idx, return_final_states=False):
    """Non-mem-eff forward with dt_scale support (post-softplus scaling)."""
    layout = _ensure_layout_cache(layer)
    batch, seqlen, _ = x.shape
    d_ssm = layer.d_ssm
    nheads = layer.nheads
    headdim = layer.headdim
    d_state = layer.d_state
    ngroups = layer.ngroups
    d_mlp = layout['d_mlp']

    zxbcdt = layer.in_proj(x)
    z0, x0, z, xBC, dt = torch.split(
        zxbcdt,
        [d_mlp, d_mlp, d_ssm, d_ssm + 2 * ngroups * d_state, nheads],
        dim=-1
    )

    assert layer.activation in ("silu", "swish")
    if causal_conv1d_fn is None or layer.activation not in ("silu", "swish"):
        xBC = layer.act(
            layer.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(layer.d_conv - 1)]
        )
    else:
        xBC = causal_conv1d_fn(
            xBC.contiguous().transpose(1, 2),
            layout['conv1d_weight_2d'],
            bias=layer.conv1d.bias,
            activation=layer.activation,
            seq_idx=seq_idx,
        ).transpose(1, 2)

    x_ssm, B, C = torch.split(
        xBC, [d_ssm, ngroups * d_state, ngroups * d_state], dim=-1
    )

    A = -torch.exp(layer.A_log.float())

    dt_limit_kwargs = {} if not layout['has_dt_limit'] else dict(dt_limit=layout['dt_limit'])

    if dt_scale is not None:
        dt_processed = F.softplus(dt + layer.dt_bias) * dt_scale
        scan_dt_bias = None
        scan_dt_softplus = False
    else:
        dt_processed = dt
        scan_dt_bias = layer.dt_bias
        scan_dt_softplus = True

    scan_kwargs = dict(
        chunk_size=layer.chunk_size,
        D=layout['D_scan'],
        z=rearrange(z, "b l (h p) -> b l h p", p=headdim) if not layer.rmsnorm else None,
        dt_bias=scan_dt_bias,
        dt_softplus=scan_dt_softplus,
        seq_idx=seq_idx,
        return_final_states=return_final_states,
        **dt_limit_kwargs,
    )

    y = mamba_chunk_scan_combined(
        rearrange(x_ssm, "b l (h p) -> b l h p", p=headdim),
        dt_processed,
        A,
        rearrange(B, "b l (g n) -> b l g n", g=ngroups),
        rearrange(C, "b l (g n) -> b l g n", g=ngroups),
        **scan_kwargs,
    )

    if return_final_states:
        y, last_state = y
        _, _, _, xBC_raw, _ = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, d_ssm, d_ssm + 2 * ngroups * d_state, nheads],
            dim=-1
        )
        xBC_t = rearrange(xBC_raw, "b l d -> b d l")
        conv_state = F.pad(xBC_t, (layer.d_conv - xBC_t.shape[-1], 0))

    y = rearrange(y, "b l h p -> b l (h p)")

    if layer.rmsnorm:
        y = layer.norm(y, z)

    if d_mlp > 0:
        y = torch.cat([F.silu(z0) * x0, y], dim=-1)

    out = layer.out_proj(y)

    if return_final_states:
        return out, conv_state, last_state
    return out


def _step_inner(layer, x, conv_state, ssm_state, dt_scale):
    """Single-step recurrent inference through Mamba2 internals."""
    layout = _ensure_layout_cache(layer)
    dtype = x.dtype
    batch = x.shape[0]
    d_ssm = layer.d_ssm
    nheads = layer.nheads
    headdim = layer.headdim
    d_state = layer.d_state
    ngroups = layer.ngroups
    d_conv = layer.d_conv
    d_mlp = layout['d_mlp']

    zxbcdt = layer.in_proj(x)
    z0, x0, z, xBC, dt_raw = torch.split(
        zxbcdt,
        [d_mlp, d_mlp, d_ssm, d_ssm + 2 * ngroups * d_state, nheads],
        dim=-1
    )

    conv1d_weight_2d = layout['conv1d_weight_2d']
    if causal_conv1d_update is not None:
        xBC_conv = causal_conv1d_update(
            xBC,
            conv_state,
            conv1d_weight_2d,
            layer.conv1d.bias,
            layer.activation,
        )
    else:
        conv_state[:, :, :-1] = conv_state[:, :, 1:].clone()
        conv_state[:, :, -1] = xBC
        xBC_conv = (conv_state.float() * conv1d_weight_2d.float()).sum(-1)
        if layer.conv1d.bias is not None:
            xBC_conv = xBC_conv + layer.conv1d.bias.float()
        xBC_conv = F.silu(xBC_conv).to(dtype)

    x_ssm, B, C = torch.split(
        xBC_conv, [d_ssm, ngroups * d_state, ngroups * d_state], dim=-1
    )

    A = -torch.exp(layer.A_log.float())

    dt = F.softplus(dt_raw.float() + layer.dt_bias.float())
    if dt_scale is not None:
        dt = dt * dt_scale
    if layout['has_dt_limit']:
        dt = dt.clamp(layout['dt_limit'][0], layout['dt_limit'][1])

    if selective_state_update is not None and dt_scale is None:
        x_reshaped = rearrange(x_ssm, "b (h p) -> b h p", p=headdim)
        dt_for_kernel = rearrange(dt_raw, "b h -> b h") if dt_raw.dim() == 2 else dt_raw
        dt_bias_for_kernel = rearrange(layer.dt_bias, "h -> h p", p=headdim) if headdim > 1 else layer.dt_bias
        B_groups = rearrange(B, "b (g n) -> b g n", g=ngroups)
        C_groups = rearrange(C, "b (g n) -> b g n", g=ngroups)
        A_expanded = A.unsqueeze(-1).unsqueeze(-1).expand(-1, headdim, d_state).float()
        D_expanded = rearrange(layer.D, "h -> h p", p=headdim) if layer.D_has_hdim else layer.D.unsqueeze(-1).expand(-1, headdim)
        z_gate = rearrange(z, "b (h p) -> b h p", p=headdim) if not layer.rmsnorm else None

        y = selective_state_update(
            ssm_state, x_reshaped,
            rearrange(dt_raw, "b h -> b h 1").expand(-1, -1, headdim),
            A_expanded, B_groups, C_groups, D_expanded,
            z=z_gate,
            dt_bias=dt_bias_for_kernel,
            dt_softplus=True,
        )
        y_flat = rearrange(y, "b h p -> b (h p)")
    else:
        heads_per_group = nheads // ngroups
        dA = (dt.unsqueeze(-1) * A.view(1, nheads, 1)).exp()
        x_heads = x_ssm.float().view(batch, nheads, headdim)

        if ngroups == 1:
            B_heads = B.float().view(batch, 1, d_state).expand(batch, nheads, d_state)
            C_heads = C.float().view(batch, 1, d_state).expand(batch, nheads, d_state)
        else:
            B_heads = (B.float().view(batch, ngroups, 1, d_state)
                       .expand(batch, ngroups, heads_per_group, d_state)
                       .reshape(batch, nheads, d_state))
            C_heads = (C.float().view(batch, ngroups, 1, d_state)
                       .expand(batch, ngroups, heads_per_group, d_state)
                       .reshape(batch, nheads, d_state))

        dB = dt.unsqueeze(-1) * B_heads
        dBx = x_heads.unsqueeze(-1) * dB.unsqueeze(2)
        ssm_state.copy_(ssm_state * dA.unsqueeze(2) + dBx)
        y = (ssm_state.to(dtype) * C_heads.unsqueeze(2)).sum(-1)

        D_expanded = layout['D_step']
        y = y + x_heads.to(dtype) * D_expanded.to(dtype)
        y_flat = y.view(batch, d_ssm)

    if layer.rmsnorm:
        z_for_gate = z.view(batch, d_ssm)
        y_out = layer.norm(y_flat.unsqueeze(1), z_for_gate.unsqueeze(1)).squeeze(1)
    else:
        y_out = y_flat * F.silu(z)

    if d_mlp > 0:
        mlp = x0 * F.silu(z0)
        y_out = torch.cat([mlp, y_out], dim=-1)

    out = layer.out_proj(y_out)
    return out, conv_state, ssm_state
