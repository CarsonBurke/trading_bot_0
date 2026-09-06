# Wiring the fused kernels into the CausalPatch forecaster

Main tree: `6b28bad9` (off-CUDA dispatch), `2d99f2df` (ReLU² + packed rotary), `1a8f2a31`
(FusedNorm's fused QK-norm+RoPE, merged fast-forward), `7b0ec789` (that op wired, two classes
collapsed into one), plus one docs-and-report commit on top. Jobs: **5145**/**5185**/**5186**/**5189**
(`cargo test -p fused_kernels`), **5177** (pre-wiring A/B), **5147** + **5148** (first wiring,
plain and capture-audit), **5183** + **5184** (second wiring, plain and capture-audit), **5181**
(the `eps=None` resolution probe). All six benchmark probes ran on an idle device: their in-run
bf16 GEMM roofs agree to 0.13% (234.14-234.43 TFLOPS) and their copy roofs to 0.4% (1513-1519
GB/s), which is what makes them comparable to each other and what disqualifies job 5136 as a
baseline.

**Capture verdict: FITS at B=256, with 7,111 MiB to spare - and the premise that it could not was
wrong.** The private mempool does not sit on top of the eager working set. `GraphCapture` warms
up on a side stream, calls `empty_cache`, and only then captures, so the pool *replaces* the
eager reservation: reserved 18,912 MiB after warmup, 952 MiB after the release, 17,854 MiB of
private mempool, **18,806 MiB reserved at capture end** - 184 MiB *below* the same run's eager
peak reserved. What capture no longer buys is time: replay 167.38 ms against 166.96 ms eager,
**+0.25%**, where before this wiring it was -19.6%.

## 1. Files changed

| file | change |
| --- | --- |
| `trading_bots/src/torch/timexer_segment/model.rs` | both call sites, two `kernel_classes` entries merged into one and one re-pointed, `step_cost`, the `rotation` field, the constructor, the corrected `eps` doc, 4 test sites |
| `docs/timexer_segment.md` | new backbone bullet for the two kernels; the class list; the Throughput section's capture paragraph rewritten from measurement |
| `fused_kernels/src/lib.rs` | `relu_square`/`rope` dispatch to `reference::*` off CUDA (coordinated with FusedNorm, who added the same for `qk_norm_rope`) |
| merged from `fused-qknorm-20260906` | `fused_kernels` gains `qk_norm_rope`, its reference, tests, probe and a probe-only `pyo3` dependency |

Deleted, with nothing left in their place: `rotation_tiles`, `Block::rotate`, the
`rms_norm(&packed[0].reshape(..)).reshape(..)` binding, and `relu().square()`.
`CausalPatchModel::rotation` now holds the UNTILED `[origins, head_dim/2]` bf16 rows (24 KiB
each). No env flag, no composed branch on the model path, no dead code. The link args were
already landed by `dee5ebc5` (`fused_kernels/build.rs:190-193`, the same three as
`trading_bots/build.rs`), so `Cuda::is_available()` is true wherever the crate is linked.

The off-CUDA dispatch inside `fused_kernels` is the one edit that needs defending: ten of the
model's 50 tests call `Block::forward` on `Device::Cpu`, two in fp32, and `reference::*` is
bit-identical to each kernel by that crate's own tests - so it is the op's CPU implementation,
not a fallback on the training path, which asserts CUDA long before it reaches either op.

## 2. Probe table (all B=256, 40 steps, warmup 6, `--profile`)

| quantity | 5177 pre-wiring | 5147 ReLU²+rope | 5183 + QK-norm fused |
| --- | --- | --- | --- |
| end-to-end step ms | 221.16 | 184.97 | **168.59** |
| origins/s | 1,157.5 | 1,384.0 | **1,518.4** |
| achieved matmul TFLOPS | 76.80 | 91.82 | **100.74** |
| measured in-run GEMM roof | 234.29 | 234.43 | 234.14 |
| fraction of GEMM roof | 32.8% | 39.2% | **43.0%** |
| achieved GB/s (analytic) | 903.54 | 825.22 | 849.41 |
| measured in-run copy roof | 1,516.14 | 1,518.75 | 1,513.34 |
| fraction of copy roof | 59.6% | 54.3% | 56.1% |
| analytic step traffic | 199.827 GB | 152.642 GB | **143.204 GB** |
| peak allocator MiB | 18,590.02 | 18,620.60 | **18,573.73** |
| peak reserved MiB | 19,172 | 19,192 | 19,092 |

**-52.57 ms/step end to end (-23.8%)** against the pre-wiring commit `665396ac`, built and run
identically on the same idle device (job 5177, from a scratch worktree since the swap is
compile-time and an env toggle was not allowed). Increments: -36.19 ms (ReLU² + rope) and
**-16.38 ms** (QK-norm folded in, within 0.02 ms of FusedNorm's prediction). Peak allocator
**-46.87 MiB** on the second wiring (their -46.9 MiB: the eight fp32 `rstd` tensors), neutral
within run-to-run noise (±32 MiB) on the first.

Phases: forward backbone 49.9 -> **40.2 ms**, backward 118.5 -> **111.2 ms**, head+loss 10.67 ms
and optimizer 4.80 ms unchanged.

| class | 5177 fwd | 5177 bwd | 5177 step ms | 5183 fwd | 5183 bwd | 5183 step ms | 5183 % copy roof | saved MiB before | after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QK RMSNorm | 0.924 | 0.842 | 14.1 | - | - | - | - | 5.9 | - |
| rotary rotation (composed, tiled) | 1.150 | 1.664 | 22.5 | - | - | - | - | 0.0 | - |
| **QK norm + rotary (fused)** | - | - | - | **0.259** | **0.499** | **6.1** | **100.5** | - | **0.0** |
| **ReLU²** | 1.033 | 2.778 | 30.5 | **0.522** | **0.998** | **12.2** | **99.6** | 375.0 | **0.0** |

The two rows the wiring owns fall from 67.1 to 18.3 ms/step (-48.8 ms of the -52.57 ms
end-to-end). Both fused classes now run at 99.6% and 100.5% of the measured streaming roof - no
bandwidth headroom left in either. Composed layer 21.894 -> 15.406 ms; attribution error -5.1%
before, -4.6% after (the composed figure is recomputed from the class list, so it cannot go
stale). Everything else is unchanged: the three GEMMs at 89-96% of the GEMM roof, causal SDPA at
95.2% of the copy roof.

## 3. `step_cost`: 199.827 -> 152.641536 -> **143.204352 GB**

In-tree `ModelConfig::default().step_cost(256)`, confirmed by the benchmark's own
`step activation traffic GB` field in both runs. `matmul_flops` unchanged at 16.984572 TFLOP.

One width-unit = one materialized `[tokens, d_model]` bf16 activation, charged write-once
read-once (`×2`) and twice more for backward (`×3`): `3 × 2 × 2 B × 96,000 × 512 = 0.589824 GB`
per layer, **4.718592 GB** over 8 layers; one ffn-unit is **18.874368 GB** (`ffn = 4·width`).

| step | term | before | after | delta GB |
| --- | --- | --- | --- | --- |
| `2d99f2df` | rotation intermediates | 8·width | 2·width (fused output alone) | **-28.311552** |
| `2d99f2df` | feedforward | 3·ffn (up, `relu`, `square`) | 2·ffn (up, fused output) | **-18.874368** |
| `7b0ec789` | QK-norm output | 2·width | 0 (never materialized) | **-9.437184** |

`199.827 - 28.312 - 18.874 = 152.641` (exact 152.641536), `- 9.437 = 143.204` (exact
143.204352). Against Integrator's decomposition: baseline `155.20 - 28.312 = 126.89`; recipe
`33.62 - 18.874 - 9.437 = 5.31`; skips 2.359; value residual 8.651; total **143.21**. The
recipe's traffic surcharge is down from 33.62 to 5.31 GB/step - what is left of it is the x0
`addcmul`.

Three corrections to the briefs, all charging MORE than they assumed: the `aten::square` pass is
**18.874 GB/step**, not ~15.7; `"rotary rotation"`'s `forward_bytes` is **`4. * state`**, not
`2. * state` (the packed `q‖k` block is `2·state` by itself, and the measured 1,521 GB/s at
100.5% of roof is coherent only with `4·state`); and the fused QK-norm saves **46.9 MiB** of
retained activation, not ~1,500 MiB - ATen never retained the normalized block, only the fp32
`rstd`.

## 4. Memory at every capture stage

| stage | 5148 (ReLU²+rope) | 5184 (+QK-norm fused) |
| --- | --- | --- |
| reserved before warmup | 440 MiB | 440 MiB |
| reserved after warmup (5 steps, side stream) | 18,952 MiB | 18,912 MiB |
| reserved after `empty_cache` | 992 MiB | 952 MiB |
| allocated (live) at capture start | 599.15 MiB | 599.15 MiB |
| **private mempool the capture reserves** | **17,954 MiB** | **17,854 MiB** |
| **reserved at capture end** | **18,946 MiB** | **18,806 MiB** |
| device total | 32,116 MiB | 32,116 MiB |
| eager step peak allocator (same run) | 18,588.59 MiB | 18,541.72 MiB |
| eager peak reserved (same run) | 19,030 MiB | 18,990 MiB |
| eager / captured replay step | 183.36 / 183.64 ms | 166.96 / **167.38 ms** |

## 5. Capture verdict: **FITS**, by 7,111 MiB

Usable 32,116 - 6,199 (foreign tenants) = **25,917 MiB**; peak reserved with capture armed
**18,806 MiB**; headroom **7,111 MiB**. It fits at the first wiring too (18,946 MiB, 6,971 MiB
headroom), so this is not something the QK-norm fusion bought.

The `18.6 + 17.0 = 35.6 GiB` arithmetic double-counted the working set: the protocol releases the
eager reservation before capturing (952 MiB reserved afterward, 599 MiB live and pinned by the
optimizer's already-captured bodies), so the 17,854 MiB pool is where the transients move, not a
second copy. Job 5116 died on the older ordering that captured without releasing.

Replay correctness (5184): objective 2.753e-7 relative on the first compared step, 4.809e-4 worst
over 20, against an eager-vs-eager null of 9.176e-7 and 7.234e-4 on the identical window; worst
parameter 4.373e-1 vs a null of 4.053e-1. Inside the device's own nondeterminism floor.

Value of capture: **+0.25%**. Serialization probes agree the step is no longer latency-bound: one
host objective read per step 167.15 ms, pinned upload 174.86 ms, pageable 174.80 ms.

## 6. Verification

- `cargo check -p trading_bot_0 --tests`, `cargo check -p trading-bot-tui --tests`: zero errors
  after each wiring.
- `cargo test -p trading_bot_0 timexer_segment`: **50 passed** after both wirings. No tolerance
  added, no assertion weakened, no test deleted; the bit-identity of the kernels is what makes
  that possible. The composed `rms_norm`-then-`rope` pair now lives in `mod tests` as the
  independent reference (`use fused_kernels::rope as fused_rope` inside the test module), so no
  composed spelling remains reachable from the model.
- `cargo test -p fused_kernels`: 6/6 at `dee5ebc5` (job 5145). At the first merged tip it was
  **racy**: job 5185 got **8 passed, 6 failed** and job 5186 with `--test-threads=1` got 14/14.
  `every_kernel_captures_and_replays_inside_a_cuda_graph` captures while cargo's other test
  threads issue CUDA work, so siblings died with `cudaErrorStreamCaptureUnsupported` (from
  `memcpy_and_sync` / `CachingHostAllocator`) and the capture test then died with
  `cudaErrorStreamCaptureInvalidated`; FusedNorm's earlier 14/14 runs had used
  `--test-threads=1` and were not evidence. **Fixed by them in `a9d49064`, merged
  fast-forward: job 5189 runs `cargo test -p fused_kernels` at cargo's default thread count on
  this tip and passes 14/14.** The fix is a process-wide claim: `cuda()` hands out a
  `CudaClaim` (a static `MutexGuard` that `Deref`s to the `Device`), so a CUDA test in that
  module cannot name a device without holding the lock, and poison is drained so one failing
  test does not cascade. Worth keeping from the diagnosis: a private capture stream would NOT
  have helped - `CUDAGraph::capture_begin` uses `cudaStreamCaptureModeGlobal`, the only mode
  libtorch's C++ API exposes and the only one `torch-sys` binds, and it rejects unsafe CUDA
  actions process-wide for the duration of the capture. Nothing about the model path was ever
  affected: the ops are bit-identical and capturable, including inside the full
  forward+backward capture of jobs 5148/5184.
- Job 5181 settled `eps=None` by measurement: on a bf16 CUDA input,
  `_fused_rms_norm(x, shape, None, None)` is bit-identical to `eps = 1.1920929e-7` (max
  difference exactly 0.0) and differs from `finfo(bf16).eps = 7.8e-3` by 21× in RMS - the kernel
  resolves its default from the fp32 accumulate type. `model.rs`'s claim that `None` meant
  7.8e-3 and cost "a 0.4% systematic shrink" was false and is replaced by the measurement.
  Passing 1e-6 remains right and is not cosmetic: at a collapsed head-block RMS of 4.2e-3 the two
  choices differ by 2.5%.

## 7. Open items for Main

1. **Capture is now a decision, not a constraint.** It fits with 7.1 GiB of headroom and is
   provably correct against the eager null, but returns +0.25%. Arming it in training buys
   nothing measurable today; the audit's value is that it is the only instrument reporting the
   memory-stage table.
2. **`fused_kernels`' test race is fixed and verified in main** (`a9d49064`, merged
   fast-forward): nothing left to do, recorded here only because the diagnosis is reusable.
3. **`device_peaks` is still measured sequentially**, not interleaved with the classes it is a
   denominator for. Every probe here ran on an idle device and the roofs agree across all six, so
   no fraction in this report is contention-dependent - but the defect that produced job 5136's
   104.61 TFLOPS roof (against 234.3 on the same silicon) is unfixed, and it is a `benchmark.rs`
   change I did not make.
4. **The remaining ReLU² prize needs a GEMM epilogue**: at 99.6% of roof the kernel is done, and
   the only pass left to remove is `x` itself (cuBLASLt/CUTLASS up-GEMM with a `relu²` epilogue
   plus a backward that recovers `relu(x)` from `y`, inheriting the `sqrt` accuracy question).
   -18.87 GB/step, and a different piece of work.
