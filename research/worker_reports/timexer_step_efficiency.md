# CausalPatch training-step efficiency

Static accounting, the changes it justifies, and the one job that measures them.
Configuration throughout: `seq_len 6000`, `pred_len 192`, `patch_len 16`, `layers 8`,
`d_model 512`, `heads 8`, `ffn 2048`, `n_aux 12` (6 `known_future`), batch **256**,
375 causal-patch origins per row -> **96,000 tokens per step**. Device: RTX 5090
(1.79 TB/s spec HBM, ~209 TFLOPS dense bf16).

Everything below is derived from the source, not measured: no GPU work was run (mlq 5084
holds the device). `ModelConfig::step_cost` in `model.rs` is this accounting in code, and
`benchmark --profile` divides the measured step by it.

## 1. What one step materializes (before)

| tensor | shape | dtype | bytes |
| --- | --- | --- | --- |
| batch (resident packed row block) | `[256, 111457]` | fp32 | 114 MB |
| token input | `[256, 375, 256]` | bf16 | 49 MB |
| patch embedding | `[256, 375, 512]` | bf16 | 98 MB |
| per layer: 2 norms | `2x[256, 375, 512]` | bf16 | 197 MB |
| per layer: qkv | `[256, 375, 1536]` | bf16 | 295 MB |
| per layer: SDPA out, projection, residuals | `~6x[256, 375, 512]` | bf16 | 590 MB |
| per layer: FFN hidden + out | `2x[256, 375, 2048]` | bf16 | 786 MB |
| **per layer total** | | bf16 | **1,868 MB** |
| **8 layers** | | bf16 | **14.94 GB** |
| known-covariate window | `[256, 375, 1152]` | bf16 | 221 MB |
| covariate projection | `[256, 375, 256]` | bf16 | 49 MB |
| head input | `[256, 375, 768]` | bf16 | 147 MB |
| head hidden + GELU | `2x[256, 375, 1024]` | bf16 | 393 MB |
| **dense head output** | `[256, 375, 192, 8]` | bf16 | **295 MB** |
| head output widened to fp32 (**before**) | `[256, 375, 192, 8]` | fp32 | **590 MB** |
| ... times the `* HEAD_OUTPUT_SCALE` copy (**before**) | same | fp32 | **590 MB** |
| targets | `[256, 375, 4, 192]` | fp32 | 295 MB |
| mask | `[256, 375, 1, 192]` | fp32 | 74 MB |

The head output space is **147.456 M elements** per step (`96,000 x 192 x 8`): 295 MB in
bf16, 590 MB in fp32. One elementwise pass over a single channel of it
(`[256, 375, 1, 192]`, 18.432 M elements) moves 73.7 MB in fp32.

## 2. Kernel classes, FLOPs and traffic (before)

`matmul` (fwd + bwd = 3x forward):

| GEMM | fwd TFLOP | note |
| --- | --- | --- |
| patch embedding | 0.05 | `256x(4+12) -> 512` |
| per layer qkv / out / FFNx2 | 0.60 | dense |
| per layer causal SDPA (`QK^T` + `AV`) | 0.04 | 6% of a layer; 375 tokens is a short sequence |
| **backbone, 8 layers** | **5.13** | 90.6% of all matmul |
| covariate projection + head hidden + head out | 0.51 | 9.0% |
| **total forward** | **5.66** | |
| **total step (fwd+bwd)** | **16.98** | |

`traffic` (lower bound: every materialized activation written once and read once forward,
twice more in backward):

| class | GB per step | side |
| --- | --- | --- |
| backbone + head activations (`6 x 16.2 GB`) | 97.2 | traffic |
| head geometry + NLL, enumerated (below) | 34.7 | traffic |
| slice-backward zero padding (below) | ~7.2 | traffic, pure waste |
| strided channel reads, sector amplification (below) | ~4.1 | traffic, pure waste |
| targets + mask | 0.7 | traffic |
| **total** | **~144** | |

Roofline: matmul floor `16.98 TFLOP / 209 TFLOPS` = **81 ms** (113 ms at a realistic
150 TFLOPS for these shapes); traffic floor `144 GB / 1.79 TB/s` = **80 ms** (103 ms at a
realistic 1.4 TB/s). Measured **302 ms** is therefore **56 TFLOPS (27% of dense peak)** and
**~477 GB/s (27% of spec bandwidth)**.

`GPU busy 100%` with power at 66-74% and 53% memory-controller busy is consistent with that
number and *not* with launch starvation: the step issues roughly 800-900 kernels, so launch
latency is ~4-8 ms of 302 ms. The step is **traffic-bound with poor per-kernel bandwidth
efficiency**, and the three worst offenders are structural:

**(a) the head/loss chain, op by op.** Enumerated from the pre-change source:
forward = head GEMM write (295 MB bf16), one fp32 cast of the *whole* 8-channel space
(295 read + 590 write), the `HEAD_OUTPUT_SCALE` multiply on that space (590 + 590), 4 log-scale
ops on `[.., 4]` (295 + 295 each), 19 single-channel `decode_joint` ops, one 4-channel `cat`,
7 `nll_elements` ops on `[.., 4]`, the mask multiply, two reductions, and 3 more full-size ops
for the no-gradient MSE = **40 full-size forward kernels, 15.19 GB**; backward re-reads and
writes for the 35 differentiable ones = **19.54 GB**. Total **34.73 GB in 75 full-size
kernels**, all fp32, for an objective whose inputs are bf16.

**(b) slice backward materializes zero-padded full-size tensors.** `Output` narrowed the
8-channel space twice and `decode_joint` narrowed `coordinates` four more times. ATen's
backward for `narrow`/`slice` is `zeros_like(input)` plus a `copy_` into the slice, so each
of those six narrows costs a full-size zero-fill plus an accumulate: `4 x 295 MB` +
`2 x 590 MB` materialized, ~3 passes each -> **~7.2 GB of traffic that computes nothing**.

**(c) channel-last layout makes every channel slice a strided gather.** In
`[.., 192, 8]` a channel slice reads 4 useful bytes per 32-byte sector: 8x read
amplification on the ~8 strided read streams in the chain, **~4.1 GB effective for 0.6 GB
nominal**.

Phase verdicts: backbone = matmul-leaning but within 15% of its own traffic floor
(89.6 GB / 5.13 TFLOP fwd); head GEMMs = matmul (1.53 TFLOP fwd+bwd, 6.6 GB); head geometry +
NLL = **traffic, by a factor of infinity** (zero FLOPs of arithmetic value, 46 GB);
optimizer = already captured, not on the critical path; statistics/targets = ~40 kernels over
the 6,192-bar history, ~2.5 GB, negligible.

## 3. Changes, with expected saving

| # | change | saving |
| --- | --- | --- |
| 1 | **Head emits channel-major bf16** `[rows, origins, 8, 192]` and the training loss consumes it directly. The fp32 widening of the whole space and the `HEAD_OUTPUT_SCALE` copy are gone: the multiplier now scales the 1024x1536 *weight* the GEMM already copies (a power of two, so the cast commutes exactly). | -1.77 GB fwd, -3.5 GB bwd, -2 kernels; removes 2 x 590 MB fp32 materializations |
| 2 | **One `split` node** instead of six `narrow`s. Backward scatters 8 gradients into one buffer. | ~-7.2 GB, -18 backward kernels |
| 3 | **Channel-major slices are contiguous.** | ~-4.1 GB of sector amplification |
| 4 | **Hand-fused geometry + NLL** (`CausalPatchModel::losses`): promotion happens in the first arithmetic op that needs a channel (`bf16 x [1,1,1,pred_len] fp32` reads bf16, writes fp32, one pass) instead of a separate cast; `1/sigma` applied to the three offsets so `low` is shared by `high` and `open`; `exp(-2*ls)` factored as `exp(-2*CAP*tanh(u)) * (1/h)`, which lifts `+ 1/2*ln h` out of the full-size space into a gradient-free `[pred_len]` reduction; the mask folded into the precision weight that is materialized anyway; both reductions are `dot`s, so no masked full-size copy of the NLL or of the squared error is ever written. | 34.73 -> 19.54 GB (-44%); 75 full-size kernels -> 101 *single-channel* ones (fewer bytes, more launches - which is why change 6 belongs with it) |
| 5 | **fp32 kept exactly where it is required** - statistics, sigma, beta, every geometry and NLL element, both accumulations - and nowhere else. There is no autocast in this module to re-enter; activations are cast to bf16 once at the patch embedding and every fp32 master parameter is cast at its point of use, so LayerNorm and every GEMM see matching dtypes (the `Cannot dispatch to fused implementation` fallback cannot occur). The per-call weight casts cost ~0.3 GB/step total, 0.2% of traffic: measured negligible, left alone. | correctness of the dtype story; no regression |
| 6 | ~~Forward + backward captured as one CUDA graph~~ **implemented, then removed** - see section 3b. | none; withdrawn |
| 7 | **No host sync inside a step.** `check_objective`'s per-step `torch._assert_async` (a GIL acquisition plus a tensor bridge every step) is replaced by a device indicator accumulated into `total_loss[2]`, read once per report interval. The fused optimizer's `zero_grad` now passes `set_to_none=False`. | one GIL round-trip and one implicit sync per step |
| 8 | **Per-phase reporting**: six series on the existing `timexer_segment_timing` base (host batch, H2D, forward backbone, forward head and loss, backward, optimizer), sampled once per report interval so no ordinary step is serialized. `benchmark --profile` reports the same phases plus achieved TFLOPS / GB-s against *measured* device peaks (a large bf16 GEMM and a large device-to-device copy, not a spec sheet). | measurement, no runtime cost |

Net expected: traffic **~144 -> ~118 GB (-18%)**, kernels **~900 unchanged** (change 4 trades
bytes for launches), step **302 ms -> 260-275 ms** from the traffic reduction alone. Derivation:
the head/loss chain's own time at the 1.4 TB/s the step actually achieves falls 25 ms -> 14 ms,
and the removed padded-narrow backward plus the sector amplification are another ~11 GB, i.e.
~8 ms; the ~100 extra single-channel launches give ~1 ms back. Call it **-30 to -40 ms, 10-13%**.
The earlier 230-250 ms projection assumed the graph removed 4-8 ms of launch latency *and*
that the head's per-kernel bandwidth would rise toward the copy peak; neither is banked here.
The residual is not in the head.

## 3b. Why forward+backward is NOT captured (measured, not argued)

It was implemented (`StepGraph` in `compute.rs`, `Batch::resident`/`upload` in `corpus.rs`,
armed once per run in `runner.rs`) and it **does not fit in VRAM**. Benchmark job 5116 died in
the capture body: `CUDA out of memory. Tried to allocate 282.00 MiB`, at batch 256 on an
otherwise idle 32 GB device (attempt log `~/.local/state/mlqueue/attempts/3826`). The optimizer
capture succeeded; the forward+backward capture is what failed. Cause: a capture must allocate
the step's entire transient working set inside a *private* mempool that the global caching
allocator cannot reuse, while the warmup steps' blocks are still retained in the global pool -
~15 GB duplicated, plus fp32 master weights and optimizer state, against 32 GB.

The whole capture is therefore deleted - no smaller-batch capture, no env flag, no fallback
path. The accounting says the trade was bad even had it fit: the step issues ~800-900 kernels,
so host launch latency is **4-8 ms of a 302 ms step, at most 2.6%**, against 15 GB of
unreusable reservation and an OOM in a 12 h run. Everything the capture work *incidentally*
forced is kept, because each pays on its own: no per-step host bridge (item 7), `set_to_none=False`
so the backward does not reallocate ~100 MB of gradient buffers every step, and the phase
instrumentation. `Batch::resident`/`upload` existed only for the capture and are gone with it.

One graph over forward+backward+optimizer was never reachable in any case: the optimizer step
is already captured by `Muon::try_graph_step` (`trading_bots/src/torch/optim/muon.rs:1578`),
which installs its own side stream and calls `capture_begin` on it, and CUDA forbids nesting a
capture inside a capture. That capture stays exactly as it was - it is small (parameter-sized
tensors, no activations), it already works, and it is untouched by this change.

Phase attribution is therefore the plain eager one: the interval's last step pays four
synchronizations and fills the six `timexer_segment_timing` series; every other step keeps its
asynchronous launch pipeline.

## 4. Residual bottleneck

After these changes the step is **97.2 GB of backbone/head activation traffic and 17 TFLOP**,
i.e. a ~84 ms floor, against a projected ~265 ms. The head is no longer interesting; what
remains is per-kernel efficiency inside the backbone, and the accounting says where to look:
of the 1,868 MB a layer materializes, 786 MB is the FFN pair and 590 MB is the
SDPA-output/projection/residual chain. The next lever is fusing the residual-add and the norm
into their neighbours (or an epilogue-fused SDPA), not anything in the loss. Confirm with the
per-phase breakdown before spending on it: if `forward backbone` + `backward` is not ~80% of
the measured step, the model of this step is wrong.

## 5. Verification performed

- `fused_loss_matches_the_reference_decode_and_nll_including_gradients` (`model.rs`) pins the
  fused loss against the reference chain it replaced (`gaussian_nll(decode_joint(..), ..)`,
  kept for the evaluation decoders) with a nonzero head, so every branch of the candle
  geometry and of the log-scale cap carries signal. **Observed: 0.0e0 relative on the NLL,
  0.0e0 on the no-gradient MSE, 0.0e0 worst-case over all 34 parameter gradients** - the
  reassociation is exact in fp32 for this algebra, not merely inside the 1e-4 tolerance the
  test asserts.
- Two tests were flaky before this work and are now correct: they seed or draw from
  libtorch's process-global RNG without the `torch::test_rng` guard the crate's protocol
  requires (`tensor_scorer_matches_scalar_reference` failed 2 runs in 6 on
  `the reference must exercise a multi-element tail`, `zero_head_forecasts_...` once).
  Guards added to the nine RNG-touching `timexer_segment` tests.
- `cargo test -p trading_bot_0 timexer_segment`: **33 passed, 0 failed**, 15 runs total,
  including after the capture removal and the `device_peaks` ordering fix (the 33rd test is a
  sibling's, landed alongside).
- `cargo check -p trading_bot_0 --tests` and `-p trading-bot-tui --tests`: **zero errors**.
- Equivalence re-run after both removals: NLL 0.000e0, MSE 0.000e0, worst parameter gradient
  0.000e0 relative over 34 parameters - still exact.
- `OBJECTIVE` is unchanged at `causal_patch_market_neutral_nll_v2`, correctly: no change
  alters the objective beyond fp32 reassociation, and the equivalence test measures that
  reassociation at exactly zero.

## 5b. Benchmark harness abort (job 5117) and the ordering fix

With the capture gone, job 5117 no longer OOMs - it completed every timing window (stdout
reached `pretrain CUDA graph captured: NorMuon+AdamW optimizer step`) and then aborted in
`tch`: `Torch("cannot set number of interop threads after parallel work has started or
set_num_interop_threads called")`, surfacing at `main.rs:1877`
(attempt log `~/.local/state/mlqueue/attempts/3827`).

Cause, and it was mine: `device_peaks()` - added in this work to measure the roofline
denominators - resolved its device with
`single_ticker_timexer::runner::cuda_device()`, and that function is entry-point setup: it
calls `pretrain::configure_threads()` (`single_ticker_timexer/runner.rs:127`), whose
`tch::set_num_interop_threads` is a one-shot compare-exchange that raises on the *second* call
or after any parallel work (`pretrain.rs:4170`, and `lib.rs:60-63` states the rule explicitly).
`device_peaks` runs after the whole benchmark, so it was a second call after a great deal of
parallel work.

Fix: `device_peaks(device: Device)` takes the device the entry point already resolved
(`benchmark.rs:192`, call site `:390`). Thread configuration now happens exactly once, at
`benchmark.rs:288`, before any tensor or CUDA work. No `Once`, no swallowed error, no second
configuration path - the ordering itself is correct.

Audit of the rest of the benchmark path for entry-point-only setup done twice, all single-call:
`configure_threads` (only via `cuda_device` at `:288`), `disable_autograd_multithreading`
(`:289`, guard held for the whole run), `configure_cuda` + `enable_tf32_matmul` (inside
`cuda_device`, hence once), `HardwareSampler::start` (`:362`, one NVML handle, dropped on
`finish`). `cuda_memory(reset)` is allocator *instrumentation* (`reset_peak_memory_stats`,
`max_memory_allocated`), idempotent by construction, and is deliberately called three times.
The `.init_array` constructor that pins threads pre-`main` is `#[cfg(test)]` (`lib.rs:69`), so
it does not add a call in the production binary.


## 6. The job to queue

```bash
mlq submit --name benchmark-timexer-segment --max-parallel-runs 1 --time-limit 20m \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh benchmark-timexer-segment \
  --output training/runs/benchmark-timexer-segment/gens/0 \
  --batch-size 256 --steps 40 --warmup 6 --profile --verify-optimizer
```

Read, from `timexer_segment_benchmark`:

- `end-to-end step milliseconds` - compare against the 302 ms baseline; expect 260-275 ms;
- `achieved matmul TFLOPS` / `measured device bf16 GEMM TFLOPS` and
  `achieved HBM GB/s` / `measured device copy GB/s` - both roofs, measured on the same device
  in the same process;
- `peak allocator MiB` / `peak reserved MiB` - should now sit near the eager step's own
  working set, with no private capture pool on top;
- `timexer_segment_benchmark_phases`: the six per-phase milliseconds of the step. If
  `forward backbone` + `backward` is not ~80% of it, re-derive section 2 before optimizing
  further.
