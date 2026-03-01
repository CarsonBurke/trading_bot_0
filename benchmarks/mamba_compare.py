#!/usr/bin/env python3
"""Benchmark reference mamba-ssm SSD kernel for comparison with our implementation."""

import sys
import importlib.util
import types

mamba_ssm_path = '/home/marvin/Documents/GitHub/mamba-ssm'
sys.path.insert(0, mamba_ssm_path)

# Create stub packages to avoid triggering __init__.py files with heavy deps
def create_stub_package(name):
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod
    return mod

# Mock CUDA extensions
class MockModule:
    pass
sys.modules['selective_scan_cuda'] = MockModule()
sys.modules['causal_conv1d_cuda'] = MockModule()
sys.modules['causal_conv1d'] = types.ModuleType('causal_conv1d')
sys.modules['causal_conv1d'].causal_conv1d_fn = None
sys.modules['causal_conv1d'].causal_conv1d_fwd_function = None
sys.modules['causal_conv1d'].causal_conv1d_bwd_function = None
sys.modules['causal_conv1d'].causal_conv1d_update_function = None
sys.modules['causal_conv1d.cpp_functions'] = sys.modules['causal_conv1d']

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Create parent packages without importing their __init__
create_stub_package('mamba_ssm')
create_stub_package('mamba_ssm.ops')
create_stub_package('mamba_ssm.ops.triton')
create_stub_package('mamba_ssm.utils')

# Load utils.torch directly
spec = importlib.util.spec_from_file_location(
    "mamba_ssm.utils.torch",
    f"{mamba_ssm_path}/mamba_ssm/utils/torch.py"
)
utils_torch = importlib.util.module_from_spec(spec)
sys.modules['mamba_ssm.utils.torch'] = utils_torch
spec.loader.exec_module(utils_torch)

# Load triton submodules in dependency order
for mod_name in ['softplus', 'ssd_bmm', 'ssd_chunk_state', 'ssd_state_passing', 'ssd_chunk_scan', 'layernorm_gated', 'k_activations', 'ssd_combined']:
    spec = importlib.util.spec_from_file_location(
        f"mamba_ssm.ops.triton.{mod_name}",
        f"{mamba_ssm_path}/mamba_ssm/ops/triton/{mod_name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f'mamba_ssm.ops.triton.{mod_name}'] = mod
    spec.loader.exec_module(mod)

mamba_chunk_scan_combined = sys.modules['mamba_ssm.ops.triton.ssd_combined'].mamba_chunk_scan_combined

def sync():
    torch.cuda.synchronize()

def benchmark_ssd_kernel():
    """Benchmark the raw SSD Triton kernel with matching parameters."""
    device = "cuda"
    dtype = torch.bfloat16

    # Match our Rust benchmark config
    batch = 4
    seqlen = 4096
    d_model = 256
    headdim = 64
    d_state = 128
    chunk_size = 256
    expand = 2
    d_inner = d_model * expand  # 512
    nheads = d_inner // headdim  # 8
    ngroups = 1

    print("--- Reference Mamba-SSM SSD Kernel Benchmarks ---")
    print(f"  Batch: {batch}, SeqLen: {seqlen}, Dtype: BFloat16")
    print(f"  d_model: {d_model}, d_inner: {d_inner}, headdim: {headdim}")
    print(f"  d_state: {d_state}, nheads: {nheads}, chunk_size: {chunk_size}")
    print()

    # SSD kernel inputs (post-projection, post-conv)
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device, requires_grad=True)
    dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).to(dtype).requires_grad_()
    A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, d_state, dtype=dtype, device=device, requires_grad=True)
    C = torch.randn(batch, seqlen, ngroups, d_state, dtype=dtype, device=device, requires_grad=True)
    D = torch.randn(nheads, dtype=dtype, device=device, requires_grad=True)

    # Warmup
    for _ in range(20):
        y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D)
    sync()

    iters = 200

    # Forward only (SSD kernel)
    start = time.perf_counter()
    for _ in range(iters):
        y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D)
    sync()
    fwd_ms = (time.perf_counter() - start) * 1000 / iters
    print(f"  SSD Forward:             {fwd_ms:.3f} ms/iter")

    # Forward + Backward (SSD kernel)
    start = time.perf_counter()
    for _ in range(iters):
        x.grad = dt.grad = A.grad = B.grad = C.grad = D.grad = None
        y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D)
        y.sum().backward()
    sync()
    fwd_bwd_ms = (time.perf_counter() - start) * 1000 / iters
    print(f"  SSD Forward + Backward:  {fwd_bwd_ms:.3f} ms/iter")

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"  Peak Memory:             {peak_mem_mb:.0f} MB")
    print()

def benchmark_full_mamba_layer():
    """Benchmark a full Mamba2-like layer (in_proj + conv + SSD + norm + out_proj)."""
    device = "cuda"
    dtype = torch.bfloat16

    batch = 4
    seqlen = 4096
    d_model = 256
    headdim = 64
    d_state = 128
    chunk_size = 256
    expand = 2
    d_inner = d_model * expand
    nheads = d_inner // headdim
    ngroups = 1
    d_conv = 4

    print("--- Reference Full Mamba2 Layer Benchmarks ---")
    print(f"  Batch: {batch}, SeqLen: {seqlen}, Dtype: BFloat16")

    # Build equivalent layer components
    d_in_proj = 2 * d_inner + 2 * ngroups * d_state + nheads
    in_proj = nn.Linear(d_model, d_in_proj, bias=False).to(device).to(dtype)
    conv_dim = d_inner + 2 * ngroups * d_state
    conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, padding=d_conv-1, groups=conv_dim, bias=True).to(device).to(dtype)
    out_proj = nn.Linear(d_inner, d_model, bias=False).to(device).to(dtype)
    norm_weight = nn.Parameter(torch.ones(d_inner, dtype=dtype, device=device))

    dt_bias = nn.Parameter(torch.randn(nheads, dtype=dtype, device=device))
    A = nn.Parameter(-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device)))
    D_param = nn.Parameter(torch.ones(nheads, dtype=dtype, device=device))

    x = torch.randn(batch, seqlen, d_model, dtype=dtype, device=device, requires_grad=True)

    def forward(x):
        zxbcdt = in_proj(x)
        z = zxbcdt[..., :d_inner]
        xbc = zxbcdt[..., d_inner:d_inner + conv_dim]
        dt_raw = zxbcdt[..., -nheads:]

        # Conv1d
        xbc_t = xbc.transpose(1, 2)
        xbc_conv = F.silu(conv1d(xbc_t)[..., :seqlen].transpose(1, 2))

        x_ssm = xbc_conv[..., :d_inner].view(batch, seqlen, nheads, headdim)
        B = xbc_conv[..., d_inner:d_inner + ngroups * d_state].view(batch, seqlen, ngroups, d_state)
        C = xbc_conv[..., d_inner + ngroups * d_state:].view(batch, seqlen, ngroups, d_state)

        dt = F.softplus(dt_raw + dt_bias)

        y = mamba_chunk_scan_combined(x_ssm, dt, A, B, C, chunk_size, D=D_param)
        y = y.view(batch, seqlen, d_inner)

        # RMSNorm + gate
        rms = (y.float().pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt()
        y = ((y * rms * norm_weight) * F.silu(z)).to(dtype)

        return out_proj(y)

    # Warmup
    for _ in range(20):
        y = forward(x)
    sync()

    iters = 200

    # Forward
    start = time.perf_counter()
    for _ in range(iters):
        y = forward(x)
    sync()
    fwd_ms = (time.perf_counter() - start) * 1000 / iters
    print(f"  Forward (Training):      {fwd_ms:.3f} ms/iter")

    # Forward + Backward
    start = time.perf_counter()
    for _ in range(iters):
        x.grad = None
        y = forward(x)
        y.sum().backward()
    sync()
    fwd_bwd_ms = (time.perf_counter() - start) * 1000 / iters
    print(f"  Forward + Backward:      {fwd_bwd_ms:.3f} ms/iter")

    torch.cuda.reset_peak_memory_stats()
    _ = forward(x)
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"  Peak Memory:             {peak_mem_mb:.0f} MB")

    # Inference (Prefill)
    with torch.no_grad():
        for _ in range(10):
            y = forward(x)
        sync()

        start = time.perf_counter()
        for _ in range(iters):
            y = forward(x)
        sync()
        prefill_ms = (time.perf_counter() - start) * 1000 / iters
        print(f"  Inference (Prefill):     {prefill_ms:.3f} ms/iter")

    print()

if __name__ == "__main__":
    benchmark_ssd_kernel()
    benchmark_full_mamba_layer()
