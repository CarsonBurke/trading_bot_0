# CausalPatch forward+backward CUDA-graph capture, and the step's serialization points

Two questions, both answered by measurement on one otherwise-idle RTX 5090 (32 GB) at batch
256, `seq_len 6000`, `pred_len 192`, 8 layers, `d_model 512`, 375 causal-patch origins per row:

1. Can the forward and backward be captured as one CUDA graph, what does it cost in VRAM, and
   what is it worth? **Yes: 16,996 MiB of private mempool, and the step goes 402.1 ms ->
   323.3 ms, a 19.6% reduction.** That is not the ~2.6% the launch-overhead measurement
   predicted, and section 4 says what the extra came from.
2. Where does the loop still serialize the host against the device? Measured per item in
   section 6; every remaining item is at or below the step's own 7 ms run-to-run noise.

Everything numeric below is from **mlq job 5134** (`graphcapture-probe3`,
`benchmark-timexer-segment --batch-size 256 --steps 20 --warmup 6 --capture-audit`), whose
rows are in `training/runs/graphcapture-probe3/gens/0/timexer_segment_benchmark.report.bin`.

## 1. Why the first attempt OOMed, exactly

Job 5116 died inside the capture body with `Tried to allocate 282.00 MiB` at batch 256. The
cause was not that the working set is too large. It is that **two copies of it were resident
at once**, and that the allocator is forbidden from fixing that while a capture is underway.

libtorch's caching allocator routes every allocation made on the capturing stream into the
graph's *private mempool*, which the global pool can never reuse. The warmup steps that must
precede a capture allocate the same working set in the *global* pool, and when they finish
those blocks are cached, not returned. So at `capture_begin` the process holds the working set
twice. Worse, the allocator's usual escape - releasing cached blocks on an allocation failure
- is disabled during capture, because `cudaFree` synchronizes the device and an implicit
device synchronization invalidates an in-flight capture. It cannot reclaim the warmup blocks
it is sitting on; it can only raise.

The measured budget in section 3 turns that into arithmetic: 18,076 MiB held after warmup plus
16,996 MiB of pool is 35,072 MiB on a 32,116 MiB device, over by 2,956 MiB. One
`empty_cache()` between the last warmup step and `capture_begin` returns 17,120 MiB of that,
and the same capture then completes with 14,164 MiB still free. Live tensors are unaffected:
`emptyCache` returns only *unused* blocks.

Two of my own probe jobs (5131, and KernelTraffic's 5128) reproduced 5116's exact message -
`Tried to allocate 282.00 MiB` - for a completely different reason: foreign processes held
12.96 GiB and 21 GB respectively of the shared card, leaving ~11 GiB for a step that needs
~17.7 GiB. The message is identical, so **5116's log alone does not distinguish "the pool does
not fit" from "another tenant took the card"**. Job 5134 ran with the card idle and the
capture fitted.

## 2. Protocol checklist: what was missing

| protocol item | before | now |
| --- | --- | --- |
| warmup iterations on the stream that will capture | **MISSING** - warmup ran on the default stream and only the capture body entered the graph's side stream, so the capture was the first time those GEMMs ever ran there | `StepGraph` exists from `Engine::new`, and every step's forward+backward runs inside its stream scope from step 0 |
| `empty_cache()` between warmup and capture | **MISSING - this is the OOM** | `Engine::arm_step_graph`, after the warmup synchronize; measured at 18,076 -> 956 MiB reserved |
| ONE private mempool shared by the step graph and the optimizer's bodies | **MISSING** - `Muon::arm_step_graphs` minted its own pool and the step graph would have minted a second | `Engine::new` mints one `Arc<CudaGraphPool>` and installs it with `Muon::install_graph_pool` before the first step; both Muon bodies and the step graph capture into it |
| static input filled by copy | **MISSING** - the step read whatever `Batch::to_device` allocated that step | `Batch::resident` + `Batch::upload`: one fixed device block, refilled by one non-blocking copy |
| static outputs copied out of the pool | **MISSING** | `StepGraph.outputs` holds the two loss scalars for the graph's life and `run` copies them out before the optimizer's body replays into the same pool |
| no allocation inside the captured region | assumed | *checked*: `gradient_addresses()` is compared across the capture and a moved gradient is a hard error. `zero_grad(set_to_none=false)` is what makes that hold |
| capture ordered after the optimizer's own captures | accidental | explicit: `capture_ready()` requires the optimizer's bodies to be captured first, so the shared pool's block layout is decided once |
| host loader quiescent across the capture | **MISSING** | the loop's `loader.request` moved to after the step, so the prefetch thread is never inside `pin_memory` (a `cudaHostAlloc` synchronizes the device) during a capture |
| every step presents the shape the capture recorded | **MISSING** - the epoch's last batch was ragged | the epoch runs `origins.len() / batch_size` whole batches; the tail of at most `batch_size - 1` rows is a different tail every epoch because the order is reshuffled |

Two facts are why a *shared* pool is safe here. CUDA forbids nesting captures, so the
optimizer's step stays a separate graph - one graph over forward+backward+optimizer was never
reachable. Sharing one pool between graphs is sound only when no pool address is read across
another graph's replay: here every pool tensor is transient within its own replay, and the
step graph's two output scalars are copied into the global pool the instant the replay is
issued, before the optimizer replays. Gradients, which *are* read across that boundary, live
in the global pool and are asserted not to move.

## 3. Memory budget, batch 256 (job 5134)

| point | MiB | what it holds |
| --- | --- | --- |
| device total | 32,116 | |
| allocator reserved before warmup | 442 | parameters, AdamW/NorMuon moments |
| allocator reserved after 5 warmup steps | 18,076 | the step's working set, cached |
| allocator reserved after `empty_cache` | 956 | live blocks only |
| allocator live (allocated) at capture start | 601 | parameters, gradients, moments, resident batch |
| allocator reserved at capture end | 17,952 | of which the graph's private mempool is 16,996 |
| captured step private mempool | 16,996 | activations + backward temporaries, retained for the process |
| eager step peak allocated | 17,686 | the same working set, in the global pool |
| benchmark peak reserved (whole process) | 18,154 | |
| free at capture end | 14,164 | headroom for evaluation and for other tenants |

Held-out evaluation peak: **not measured here**. The benchmark has no evaluation path, and my
three probe submissions were spent on 5131 (killed by a foreign tenant), 5134 (this run), and
one build failure. The series (`held-out evaluation peak allocator`, MiB) is registered in the
`timexer_segment_timing` chart and any training run now emits it. The bound that matters is
above: evaluation runs under `no_grad` with a last-token-only head, so it retains no
activations, and it has 14,164 MiB to work in against a training forward+backward that peaks
at 17,686 MiB *with* its saved activations.

## 4. Capture verdict: land it

| quantity | ms |
| --- | --- |
| eager step | 402.1 |
| eager step, repeated (the same measurement's own repeatability) | 395.2 |
| captured replay step | 323.3 |
| replay as a fraction of eager | 0.804 |

**19.6% off the step against the first eager phase, 18.2% against the second.** The prediction
from the earlier launch-overhead measurement (4-8 ms of launches per step) was <=2.6%, so
72-79 ms is 10x that prediction and the prediction was wrong about what replay removes. What
it removes, beyond the launches themselves: per-step ATen dispatch, the construction and
teardown of ~800 autograd nodes, allocator bookkeeping for every transient (the pool's block
layout is decided once, at capture, instead of being re-served every step), and the host-side
gaps between kernels that a launch-rate measurement does not see because it measures launch
*cost*, not the idle device time between launches. I did not decompose the 72-79 ms further;
the per-kernel work belongs to `model.rs` and is KernelTraffic's measurement.

The step is armed once, on the step after the optimizer's own two bodies finish capturing
(`CAPTURE_AFTER_STEPS = 5`), and the training loop reports the pool through
`timexer_segment_capture`.

## 5. Numerics: not bit-identical, and neither is eager

The audit runs three phases from the same seed over the same batch - eager, eager again,
captured - and compares 20 consecutive steps' objectives and the final parameters. The second
eager phase is the null, and it is the whole story:

| pair | objective, step 1 | objective, worst of 20 | worst parameter |
| --- | --- | --- | --- |
| eager vs eager | **0.000e0** | 1.007e-3 | `block_2.first.weight`, 4.701e-1 relative (3.711e-2 absolute on a 7.894e-2 scale) |
| eager vs replay | 1.190e-6 | 8.074e-4 | `block_4.output.weight`, 5.088e-1 relative (3.904e-2 absolute on a 7.673e-2 scale) |

Read it as three statements. One: a single step is reproducible - two eager runs agree on step
1 to the bit. Two: twenty steps are not, in either arm; the same eager body run twice ends up
1.0e-3 apart in the objective and ~47% apart on the largest element of one weight tensor,
because NorMuon's Newton-Schulz orthogonalization amplifies whatever nondeterminism the
backward's reductions introduce. Three: the capture's own contribution is 1.190e-6 on step 1
(about ten fp32 ulps, from cuBLAS selecting a different algorithm while capturing) and its
20-step divergence, 8.074e-4, is *smaller* than the eager arm's own 1.007e-3.

So the acceptance criterion as written - bit-identical loss and gradients under replay over 20
steps - is unattainable by any change to this code, because the eager path does not meet it
against itself. The defensible criterion is the one measured: **the capture's deviation is
indistinguishable from, and on the objective smaller than, the run-to-run deviation the eager
path already has.** Both are reported as chart rows (`capture vs eager ...` and
`eager vs eager ...`) so this claim is re-checkable on every probe rather than trusted.

## 6. Host-device synchronization and serialization audit

Measured in job 5134 on the same engine and batch, 20 steps each, batch 256:

| item | measured | verdict |
| --- | --- | --- |
| baseline step, device-resident source, no host read | 402.1 ms | reference |
| step repeatability (two identical phases) | 402.1 vs 395.2 ms | **the noise floor is 7 ms (1.7%)**; nothing below it is measurable |
| one host read of the objective per step (`.item()` / `double_value`) | 403.6 ms, +1.5 ms | inside the noise floor. Already removed anyway: the runner accumulates `[nll, mse, nonfinite]` on device and reads once per report interval, which also removed a GIL acquisition per step that this measurement does not include |
| source is a pinned host batch (production) | 401.8 ms, -0.3 ms | free. `Corpus::host_batch` pins the packed block; `Batch::upload` copies it non-blocking into the resident buffer |
| source is a pageable host batch | 409.2 ms, **+7.4 ms vs pinned (1.8%)** | this is what `pin_memory` buys: a pageable source makes the copy blocking whatever `non_blocking` says |
| host loader wait per training step | 3.83 / 1.95 / 0.0012 ms at intervals 7000/8000/9000 of run `timexer-market-neutral-20260906` (217-346 ms steps) | **prefetch depth 1 is enough.** The request for batch N+1 is issued after batch N's launches, so the worker has the whole step (>200 ms) to build a 16-68 ms batch |
| evaluation host loader wait | 186-1181 ms per evaluation of 650-7722 ms, same run | the one place depth 1 costs real time: evaluation batches are pure forward, so the step is short and the host build is not hidden. Left as is - it is off the training critical path and it is a loader-depth change, not a sync |
| evaluation, two `Cuda::synchronize` per batch | now only batch 0 | the instrumentation drained the launch pipeline twice per batch, so no batch's forward could overlap the previous batch's metric reduction. One sampled batch, like the training loop's sampled step |
| evaluation, `to_device` per batch | now a resident buffer, reallocated only when the shape changes (twice per evaluation) | the same 114 MB allocate-and-free the training step no longer pays |
| H2D on its own stream, overlapping compute | no, by design | the upload writes the buffer the captured graph reads, so it must precede the replay (a WAR hazard), and the scope's exit event orders it after the optimizer. The pinned-vs-resident row bounds what overlapping could buy at under 7 ms; double buffering would cost a second 17 GB pool to collect it |
| `Tensor::from_slice` on the hot path | `StepScalarPack::upload`, ~10 f64 per step | required: the optimizer's captured kernels read the schedule as a device operand. Already pinned and asynchronous |
| `Vec<f32>` / `Vec<f64>` extraction | once per report interval and in candle windows | off the step path |

## 7. What changed, by file

- `compute.rs`: `StepGraph` (the forward+backward on the capture stream, warmup body before
  the capture, replay after); the shared `CudaGraphPool` minted in `Engine::new` and installed
  into `Muon`; `Engine::upload` and the resident batch; `arm_step_graph` with the protocol of
  section 2 and the memory snapshots; `CaptureBudget`; `audit_step_capture` with its three
  phases and the serialization measurements.
- `muon.rs`: `install_graph_pool`, so the optimizer captures into the caller's pool instead of
  minting a second one; `StepScalarPack::upload` publishes the schedule from pinned host
  memory asynchronously.
- `corpus.rs`: `Batch::resident`, `Batch::upload`, `Batch::rows`, `Batch::host_copy` (the
  audit's pinned/pageable sources), and a test that pins the invariant the resident buffer
  depends on - every `Batch` accessor is a view of one packed block, so one copy refreshes all
  of them, and a shape the buffer was not allocated for is refused rather than truncated.
- `runner.rs`: whole-batch epochs; the capture armed once at step 5; the loader request moved
  after the step; the evaluation loop's resident buffer and single-sampled-batch timing.
- `cuda/mod.rs`, `cuda/graph.rs`, `vendor/torch-sys-0.25.0`: `at_copy_nonblocking` and
  `at_cuda_empty_cache` shims, and `CudaGraph::new_in_pool`.
