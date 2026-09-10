# CpuAudit — read-only CPU/thread audit

## Executive conclusion

**DEMONSTRATED:** The production host path uses eight Rayon workers to rebuild roughly 1.585 million bar positions per batch, performs up to 9.413 million scalar `f64` logarithms, copies an uninitialized 114 MB allocation into pinned storage, then zeros that storage before largely overwriting it. That is substantial avoidable CPU work even if it is hidden behind GPU execution.

**DEMONSTRATED:** I found no perpetual application-level busy-wait in the training path. The loader uses blocking channels; its coordinating thread blocks while Rayon works; Rayon eventually parks; libtorch intra-op and inter-op parallelism explicitly default to one. The current executable dispatches training with `spawn_blocking`, not on an asynchronous Tokio scheduler worker.

**HYPOTHESIS:** These eight gather workers, memory traffic, and possible page faults are the strongest explanation of high CPU consumption. Static inspection cannot attribute a particular observed `%CPU`, and thread existence must not be confused with active CPU use.

**DEMONSTRATED:** This audit performed source/dependency inspection only: no files written, no benchmark/training execution, no queue submission, no GPU lease. References describe the baseline read before LoaderPerf's concurrent edits. Findings were sent to LoaderPerf; this Markdown is returned rather than written to disk, as explicitly requested.

## 1. Important timing correction

**DEMONSTRATED:** `runner.rs:1647-1650` measures elapsed time around `loader.receive()`. The sampled step waits for preceding GPU work only **after** receiving the batch (`runner.rs:1653-1654`). Consequently the approximately 55 ms loader wait can overlap GPU execution. The sampled `host batch` phase is also receive latency, not a timer inside `Corpus::host_batch`.

**DEMONSTRATED:** The supplied measurements establish approximately 177–184 ms synchronized interval-mean step time, approximately 55 ms host receive wait, and approximately 2 ms sampled H2D. They do **not** establish 55 ms of GPU idle, 31% starvation, or an attainable 1.44× speedup. Those interpretations are withdrawn.

**DEMONSTRATED — supplied observation:** Main additionally reports steady-state hardware-series GPU busy of 100%, memory-controller busy approximately 46–63%, and power-limit use approximately 75–90% on the same run. These observations supersede any inference that that run spends 31% of its time starved.

**HYPOTHESIS:** True exposed loader delay could be small or nearly zero. NVML utilization is sampled and cannot itself identify small inter-step gaps or attribute them to the loader.

### Is 168.59 ms benchmark versus 177–184 ms training a loader upper bound?

**DEMONSTRATED:** Not from the current implementation. The exact series containing 168.59 matters, but neither available benchmark definition establishes a strict loader upper bound:

| Property | Production training | Main synthetic benchmark | Separate captured audit |
|---|---|---|---|
| Evidence | `runner.rs:1643-1704` | `benchmark.rs:458-553` | `compute.rs:1213-1245` |
| Source batch | Newly assembled pinned host batch | Synthetic GPU batch rebuilt every iteration | One prematerialized GPU batch |
| Upload | H2D | D2D into resident buffer | D2D into resident buffer |
| Forward/backward capture | Explicitly armed at `runner.rs:1661-1662` | **Never armed by main benchmark loop** | Explicitly armed in captured arm |
| Optimizer capture | Engine optimizer capture | Engine optimizer capture | Engine optimizer capture |
| Timed gather | Host loader overlaps device | GPU `index_select`, centering, allocation and `cat` every iteration | None |
| Per-step bookkeeping | GPU metric accumulation, nonfinite indicator, CPU LR history | Retains objective tensors in a `Vec` | Retains objective tensors in a preallocated `Vec` |
| Interval sampling | One serialized phase sample per report interval | No per-step phase sample in principal interval | No per-step phase sample |
| Warmup | First interval includes startup/warmup/capture | Separate configurable warmup; default five | Explicit warmup through capture boundary |

**DEMONSTRATED:** `benchmark.rs:538-550` calls `engine.step(&model, &gather())` for warmup and timed iterations. `Engine::step` (`compute.rs:955-960`) does not arm the forward/backward graph. `--capture-audit` runs separate engines before the principal benchmark (`benchmark.rs:512-525`); it does not make the principal `end-to-end step milliseconds` series a captured measurement.

**DEMONSTRATED:** The audit's `captured replay step milliseconds` is closer, but uses a fixed **device** batch. Its pinned-host diagnostic runs only in the **uncaptured** arm (`compute.rs:1266-1286`). There is no existing matched captured-plus-pinned-host loader-free baseline in this path.

**DEMONSTRATED:** At identical `ModelConfig` and batch size, benchmark and production instantiate the same model and therefore the same dense-origin shape. However benchmark defaults batch size to 64 (`benchmark.rs:33-34`) versus training's 256 (`runner.rs:106-107`); identical shapes must be supplied, not assumed. The benchmark's displayed `origins per second` calculation (`benchmark.rs:575-578`) counts batch rows, not the 375 dense causal origins per row.

**DEMONSTRATED:** Evaluation, checkpointing, and report-writing do not contaminate production's principal step mean: its timer closes at `runner.rs:1704` before evaluation, and resets after reporting at `runner.rs:2023-2026`. They nevertheless consume elapsed run time and CPU outside that series. The main benchmark also writes reports and performs device-peak probes after its principal timer closes.

**HYPOTHESIS:** A numerical difference of 9–15 ms may be useful motivation for a matched experiment, but is neither a demonstrated loader penalty nor a rigorous upper bound. Capture, gather placement, D2D versus H2D, optimizer recipe settings, steady-state conditions, and contention can move the difference in either direction.

### Honest measurement of loader-induced idle

**HYPOTHESIS — proposed matched measurement:** Compare the *same steady-state captured production loop* in two conditions: real loader versus a controlled prematerialized pinned-host source. Keep batch/sequence/origin counts, model and optimizer configuration, H2D upload, metric kernels, hardware sampling, and interval boundary synchronization identical. Exclude warmup/capture and evaluation. The synchronized interval-mean difference measures the **net exposed cost of the loader and its resource contention**, not its total CPU work. A fixed batch needs explicit disclosure because it changes data locality and may change numerical behavior; representative prebuilt batches improve that control at an explicit host-memory cost.

**HYPOTHESIS — proposed timeline measurement:** Record device events around the previous full step's completion and the next H2D start on the correctly ordered streams; do not synchronize per step. Read accumulated event results only at report boundaries. The intervening device-timeline gap measures exposed pipeline starvation, but includes all host scheduling delays, so correlate with loader readiness or subtract a matched ready-batch baseline before calling it loader-induced. Recording only H2D duration does not measure the gap preceding it.

**HYPOTHESIS — proposal cost:** Both approaches add zero model parameters and zero model FLOPs. A two-batch pinned reference costs approximately 228.3 MB of payload, potentially more allocator-reserved memory. Event instrumentation adds O(steps per interval) event storage, several event operations per step, and O(interval length) report-boundary reduction; dispatch overhead is unmeasured and must be controlled. Keep any new timing series in `timexer_segment_timing`, with milliseconds and its reading rule in the y-label. There is no honest exact CLI for this matched measurement today because the existing harness does not implement it; I did not invent or run one.

## 2. Thread and pool inventory

**DEMONSTRATED:** Counts below are configured counts or source-reachable creation sites, not an observed live-thread census. CPU affinity, inherited environment, and external library implementation can alter runtime counts. `main.rs:1841` loads dotenv before creating threads. A targeted search found no thread/allocator overrides in the two repository dotenv files; `torch-env.sh:64` defaults only `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and preserves an inherited allocator setting.

| Pool/thread | Count on unrestricted 12-core/24-logical-CPU machine | Configuration and evidence | Activity / interpretation |
|---|---:|---|---|
| Main OS thread | 1 | `main.rs:1839-1847`, runtime `block_on(run())` | **DEMONSTRATED:** waits for async dispatch completion. |
| Tokio asynchronous scheduler | 24 by default | `main.rs:1842-1845` leaves worker count unset. Locked Tokio 1.48.0 `runtime/builder.rs:280,1728`; `loom/std/mod.rs:87-100` consults `TOKIO_WORKER_THREADS`/available parallelism | **DEMONSTRATED:** default is logical available parallelism, not physical 12. **HYPOTHESIS:** largely parked for this command; no evidence of 24 continuously busy scheduler workers. |
| Tokio blocking pool | One explicitly requested trainer worker; default pool cap 512, not 512 created threads | `main.rs:1879-1882`; Tokio 1.48.0 `runtime/builder.rs:282` | **DEMONSTRATED:** synchronous training runs on `spawn_blocking`. No additional blocking tasks are launched by this command's inspected training path. |
| Private Rayon gather pool | 8 | `corpus.rs:330-336`: `min(available_parallelism,8)`, names `timexer-data-{i}` | **DEMONSTRATED:** used for corpus initialization and every host batch. This is the principal explicit parallel CPU compute pool. |
| Rayon global pool | Not demonstrated to be created by this command; if independently activated, default available logical CPU count unless overridden | All inspected corpus/features parallel operations run inside the private pool's `install`: `corpus.rs:337,364-385,408-410,514-530` | **DEMONSTRATED:** importing Rayon does not establish an additional 24-thread global pool. Do not add it to the count merely because the dependency exists. |
| `timexer-prefetch` | 1 | `runner.rs:670-683` | **DEMONSTRATED:** receives requests, allocates/pins a batch, delegates row filling to Rayon, sends result. Blocks on channels and during external-pool `install`; not a ninth row-filling worker. |
| libtorch intra-op | Explicit default 1, overridden by `TORCH_NUM_THREADS` | `pretrain.rs:4170-4178`, called through `single_ticker_timexer/runner.rs:127` → segment `runner.rs:1520` | **DEMONSTRATED:** not defaulting to 12/24 in this production entry point. One is parallelism level, not proof of one additional OS worker. |
| libtorch inter-op | Explicit default 1, overridden by `TORCH_NUM_INTEROP_THREADS` | `pretrain.rs:4179` | **DEMONSTRATED:** configured capacity one; actual lazy worker creation depends on use. |
| Autograd device/reentrant workers | Not safely enumerable from repository creation sites | `runner.rs:1521`; `cuda/cfg.rs:110-118`; `vendor/torch-sys-0.25.0/libtch/torch_api.cpp:350-353` | **DEMONSTRATED:** thread-local multithreading flag disabled on trainer. **HYPOTHESIS:** native engine may still initialize internal worker infrastructure; disabling dispatch does not prove that no internal thread exists. Full captured replay does not execute Rust/Python backward scheduling each iteration. |
| Hardware sampler | 1 at a time, recreated each training interval | `benchmark.rs:92-109`; lifecycle `runner.rs:1628,1705,2021` | **DEMONSTRATED:** direct NVML queries, then blocking timed receive. |
| Pinned-allocator background event thread | Default disabled; 1 if enabled | Installed torch headers `c10/core/AllocatorConfig.h:219-224,356`; `ATen/core/CachingHostAllocator.h:313-325,735-736` | **DEMONSTRATED:** optional `pinned_use_background_threads` creates one thread that processes events then sleeps 100 **microseconds**. Not the hardware sampler's 100 **milliseconds**. |
| Pinned registration helpers | Registration default disabled; configured registration count default 1 if selected | Installed `c10/cuda/CUDAAllocatorConfig.h:78-83,210,219` | **DEMONSTRATED:** `pinned_use_cuda_host_register=false`, `pinned_num_register_threads=1` defaults. Repository does not request an eight-thread registration pool. Native implementation may create helpers when externally enabled. |
| CUDA driver/native library helpers | Unknown, implementation-dependent | CUDA initialization `single_ticker_timexer/runner.rs:127-130`, embedded torch import through CUDA configuration | **HYPOTHESIS:** driver/libtorch/loaded numerical dependencies can add lazily created threads. Repository source cannot give an honest exact count for a running process without a thread census. CUDA **streams are not CPU threads**. |

**DEMONSTRATED — arithmetic:** With defaults and unrestricted affinity, the explicitly identifiable Rust-owned steady-state threads total approximately **36**: main 1 + Tokio scheduler 24 + blocking trainer 1 + Rayon 8 + prefetch 1 + sampler 1. This excludes lazy/native workers and does not mean 36 runnable threads.

**HYPOTHESIS — CPU accounting illustration, not a measurement:** Eight workers busy for 50 ms in a 180 ms step correspond to 400 CPU-ms per step, or approximately 222% process CPU in the common `top` convention. They can show approximately 800% while gathering even when average CPU is lower. The actual 50 ms receive latency is **not** a measurement of eight-worker active time, so this arithmetic cannot be substituted for a CPU profile.

## 3. Busy-wait and spin audit

### Application-level waits

**DEMONSTRATED:** `Prefetcher` uses `incoming.recv()`, bounded-channel `send()`, and result `recv()` (`runner.rs:676-708`). It has no hot `try_recv` loop. Request/result channels each have capacity one.

**DEMONSTRATED:** Private Rayon `install` called from the non-Rayon prefetch thread follows Rayon 1.13.0 `registry.rs:517-536`: it injects the job and waits on a `LockLatch`. `latch.rs:228-230` implements that latch using a mutex and condition variable. The prefetch coordinator does not spin for the eight gather workers.

**DEMONSTRATED:** Rayon worker work-stealing does perform a bounded idle search/yield period: `rayon-core-1.13.0/src/sleep/mod.rs:56-57,94-106` allows 32 initial rounds and one sleepy round. At `:186-187` it waits on a condition variable. This is short wake/search overhead, not a demonstrated core permanently spinning at 100%.

**DEMONSTRATED:** Native graph stream dependencies use recorded CUDA events and device-side event waits (`torch_api.cpp:64-71`); the comment and implementation explicitly avoid host waiting. A mutex protects creation/reuse of shared capture streams (`:80-89`), not a per-bar spin lock.

### CUDA synchronization

**DEMONSTRATED:** `atc_synchronize` forwards to `torch::cuda::synchronize` (`torch_api.cpp:1220-1221`). Production explicitly synchronizes on initialization, sampled phase timing, and report boundaries (`runner.rs:1630,1654,1703,2023`; `compute.rs:1109-1112`). Ordinary packed upload uses `copy_(...,true)` (`torch_api.cpp:420`), not the synchronizing default copy.

**DEMONSTRATED:** Searches of the application/native bridge found no explicit `cudaSetDeviceFlags`/`cudaDeviceScheduleBlockingSync` policy selection. The native driver may spin/yield during CUDA waits under its effective scheduling policy; the repository does not establish that policy.

**HYPOTHESIS:** CUDA synchronization could burn one host core while waiting, especially during validation, capture, or sampled timing. There is no evidence here of a full-step host spin on **every captured training step**. At one serialized sample per 1000 steps, even 180 CPU-ms of spin for that sample amortizes to approximately 0.18 CPU-ms per step, before additional interval-boundary effects. This is not a demonstrated throughput loss of 0.18 ms.

**DEMONSTRATED:** `fused_kernels/csrc/bridge.cpp:450-472` creates default CUDA events and uses `cudaEventSynchronize` for probe timers. Its timer API is probe-only, not part of the production model path (`:361-365`). It cannot explain continuous training CPU burn by itself.

### What the autograd message actually means

**DEMONSTRATED:** The printed message is emitted by `disable_autograd_multithreading` (`cuda/cfg.rs:110-118`). Its bridge modifies `c10::AutogradState::get_tls_state().set_multithreading_enabled(0)` and returns the previous value (`torch_api.cpp:350-353`). A thread-bound guard restores that value on drop (`cuda/cfg.rs:94-107`).

**DEMONSTRATED:** This disables multithreaded autograd scheduling for that caller; it does not size Tokio, Rayon, OpenMP, or inter-op pools and does not globally destroy native threads. Separately, the entry point already sets intra/inter-op to one. Thus the message neither proves all CPU work is serial nor implies an uncontrolled 24-way libtorch pool remains active.

## 4. Ranked waste findings and cost estimates

**DEMONSTRATED:** All code mechanisms below were identified statically. **HYPOTHESIS:** Every millisecond saving below is an order-of-magnitude planning estimate, **not measured**. The honest measured production step-time saving remains unknown and could be zero when the work is completely hidden. Estimates overlap and must not be added.

### A. Work definitely required before each batch becomes ready — exposed step penalty unknown

| Rank | Finding and location | Work/traffic per full batch | Estimated removable batch service time; exposed step saving |
|---:|---|---|---|
| 1 | **DEMONSTRATED:** repeated raw OHLC and volume logarithms, `corpus.rs:650-657`, `features.rs:565-576` | 6,340,608 OHLC logarithms + up to 3,072,000 volume logarithms + 256 anchors; scalar double-precision transcendental work | **HYPOTHESIS:** 5–25 ms service time across eight workers; actual step saving 0 up to the unhidden portion, not a demonstrated 5–25 ms |
| 2 | **DEMONSTRATED:** `Tensor::empty` then `pin_memory` copies bytes that will be overwritten, `corpus.rs:509-515` in initial snapshot | One unnecessary 114.132 MB payload copy: approximately 228.264 MB read+write traffic; pageable allocation also exists temporarily | **HYPOTHESIS:** roughly 3–12 ms at effective 10–40 GB/s payload copy bandwidth, plus possible cold-page/allocation overhead; exposed step saving 0 to that unhidden amount |
| 3 | **DEMONSTRATED:** entire packed row zeroed before field assignment, `corpus.rs:639` | Approximately 114.132 MB stores, mostly overwritten immediately | **HYPOTHESIS:** approximately 1–6 ms aggregate service time depending on bandwidth/cache behavior; exposed step saving possibly zero |
| 4 | **DEMONSTRATED:** repeated calendar/DST lookup and formatting-free scalar feature branching, `features.rs:535-550`; `dataset.rs:1407-1410` | Up to 1,585,152 timestamp conversions and transition-table binary searches, plus integer quotient/remainder arithmetic | **HYPOTHESIS:** approximately 1–8 ms service time; exposed step saving possibly zero |
| 5 | **DEMONSTRATED:** `MADV_RANDOM` on whole mmap despite sequential reads within each row, `corpus.rs:379`, `shared/src/bars.rs:405-414` | Approximately 222,912 source bytes / 55 4-KiB pages per full 6192-bar row; approximately 57.065 MB source bytes per batch before overlap/cache reuse | **HYPOTHESIS:** near 0 ms when resident; potentially multiple to tens of ms when cold/pagecache-thrashing. No finite precise cold-page estimate from source alone |
| 6 | **DEMONSTRATED:** repeated gap calculation, market/SPY grid indexing, cumulative-market anchor subtraction, `features.rs:552-600`, `corpus.rs:645,660` | O(batch × bars) scalar indexing/branches and output writes; gap logarithms only across actual gaps | **HYPOTHESIS:** sub-ms to a few ms service time after larger items are removed; exposed step saving possibly zero |
| 7 | **DEMONSTRATED:** per-batch references copied and source descriptors allocated, `runner.rs:687-693`, `corpus.rs:494-508` | 256 references, a few KiB descriptor allocation; fixed-count tensor views | **HYPOTHESIS:** ordinarily much less than 0.1 ms; not a plausible explanation of immense CPU use |

**DEMONSTRATED — exact size arithmetic:** Raw context OHLC alone is `256 × 6000 × 4 × 4 = 24,576,000` bytes. Context OHLC plus 12 auxiliaries is `98,304,000` bytes. The actual packed payload additionally includes the 192-bar horizon, validity, cumulative market, and one anchor: `256 × ((6000+192) × (6+12)+1) × 4 = 114,131,968` bytes, approximately 108.845 MiB. H2D moves this larger block. These facts invalidate a model that attributes the whole host time only to 98 MB of mandatory copying.

### B. Background CPU or interval work that is not a demonstrated per-step bottleneck

| Rank | Item | Estimated cost and status |
|---:|---|---|
| 1 | **DEMONSTRATED:** eight gather workers doing the redundant work above while previous GPU work executes | **HYPOTHESIS:** potentially hundreds of CPU-ms per step even if net exposed step penalty is 0 ms; strongest known target for lowering process CPU and tenant contention |
| 2 | **DEMONSTRATED:** evaluation/report work outside principal step timer, `runner.rs:1717 onward`, reset `:2023-2026` | **HYPOTHESIS:** can cause conspicuous CPU bursts. Amortized cost is `(evaluation CPU + report CPU)/interval steps`, unmeasured; does not explain a change in the reported training-only interval mean directly |
| 3 | **DEMONSTRATED:** bounded Rayon work-search/yield on batch boundaries | **HYPOTHESIS:** expected well below 1 ms CPU-time overhead per batch in normal operation, but no measurement; not evidence for a continuously pinned idle core |
| 4 | **DEMONSTRATED:** one NVML sampler at approximately 10 Hz maximum | **HYPOTHESIS:** likely small; exact cost unknown. If one query cycle consumes q CPU-ms and wall duration q, a 180 ms step corresponds to roughly `180q/(100+q)` CPU-ms. Blocking driver time is not necessarily CPU time |
| 5 | **DEMONSTRATED:** optional pinned allocator event polling every 100 microseconds, **disabled by default** | **HYPOTHESIS:** can matter if an inherited allocator setting enables it; approximately up to 10,000 wakeups/s, not an unbounded spin. Default contribution 0 ms |
| 6 | **DEMONSTRATED:** default 24-thread Tokio scheduler, largely unnecessary capacity for this synchronous command | **HYPOTHESIS:** parked workers cost thread stacks and startup/teardown, not 24 cores of continuous burn; expected steady-state saving near 0 ms from merely lowering their count |

## 5. Reuse/precomputation opportunities and costs

### OHLC logarithms

**DEMONSTRATED:** Raw `ln(price)` is constant for a source bar, but centering against the row's final context anchor varies by row. Current code computes `f64 ln` and subtracts in `f64` before casting to `f32`. Adjacent training windows advance by horizon 192 while carrying 6000 context bars (`corpus.rs:423-428`), so much of the raw input is processed approximately 31 times across an epoch, apart from boundaries.

**HYPOTHESIS — proposal:** Cache exact `f64` raw logarithms, retaining per-row anchor subtraction, or pursue an equally accurate computation that avoids repeated logs. Cost: zero parameters, zero GPU-model FLOP change, removal of approximately 6.34 million host logarithms per full batch, and a cache of **32 bytes per valid source bar**. At even 470,946,393 bars this is approximately 15.07 GB; that target-bar count is not the full source count, so an entire-corpus cache is larger. A float cache would be cheaper but changes rounding and is not an automatic quality-preserving replacement. Estimated service-time saving is part of rank 1, not additional.

### Volume innovations

**DEMONSTRATED:** Each interior positive volume is logged as `current` and then logged again as `previous` on the next bar (`features.rs:565-576`). Innovations are constant per ticker and previous-valid/current-valid pair, but future rows must remain zero for this history-only feature.

**HYPOTHESIS — proposal:** Carry the previous computed `f64` log-volume in the cursor first. Cost: one scalar of cursor state per row/worker, zero parameters/model FLOPs, no meaningful added memory traffic, approximately **1.536 million fewer logarithms per full batch**; perhaps 1–6 ms service-time saving, potentially 0 ms exposed. Full per-ticker innovation caching additionally avoids recomputation across windows but needs at least value and validity storage and correct quarantine/previous-valid semantics.

### Calendar/session features

**DEMONSTRATED:** Clock and weekday sine/cosine are **already cached** in `LazyLock` tables (`features.rs:500-503`); no millions of trig calls occur per batch. DST transitions are also precomputed once, but `et_offset_secs` does a binary search per timestamp (`dataset.rs:1380-1410`). The same timestamp's calendar channels recur across many tickers.

**HYPOTHESIS — proposal:** Cache four calendar floats per shared timestamp-grid slot or use a monotone per-row transition cursor. Cost: zero parameters/model FLOPs; full grid cache adds `16 × grid_slots` bytes and loads up to 25.36 MB of calendar payload per full batch in place of repeated timestamp calculations. A transition cursor adds only small per-row state and occasional transition checks. Expected saving is rank 4's 1–8 ms service estimate, possibly no step-time improvement.

**DEMONSTRATED:** Session-gap channels are **not timestamp-global**: they depend on each ticker's previous valid observed bar (`features.rs:552-558`). They may be cached per ticker/valid-bar pair, never blindly shared across all tickers. A two-float cache adds eight bytes per valid source bar; expected marginal saving is smaller because continuous five-minute steps take the no-log branch.

### Market/SPY and cumulative market

**DEMONSTRATED:** Market means, SPY returns, and cumulative market paths are built **once at corpus load**, not refitted every step: `corpus.rs:408-410,566-620`; `features.rs:458-492`. The per-bar path does lookup/validity emission and origin-relative subtraction. No repeated whole-market cross-sectional computation was found in the host batch path.

**HYPOTHESIS — proposal:** Consolidate timestamp-slot indexing only if profiling shows it matters after removing logs/copies. Cost: zero parameters/FLOPs on GPU; extra cache storage depends on layout; preserving existing shared arrays avoids duplicating market channels per ticker. Expected saving sub-ms to a few ms of batch service, not a demonstrated step benefit.

### Per-origin sigma/range/beta

**DEMONSTRATED:** `model.rs:1472-1504` computes these with batched tensor cumulative sums on the resident device tensor. It computes prefix statistics once per row then samples patch endpoints; it does **not** rescan 6000 bars separately for 375 origins on CPU. Beta and sigma depend on the causal prefix of the current row, not one epoch-wide per-ticker constant.

**DEMONSTRATED:** Fixed horizon scaling, feature masks, and rotary tables are also initialized in the model constructor (`model.rs:1331-1417`). No CPU-side per-step beta fitting or missing constant-table cache was found.

**HYPOTHESIS — recommendation:** Do not move these statistics to a CPU epoch cache as a CPU optimization. Expected host saving approximately 0 ms in captured steady state; would add substantial per-row/origin cache traffic and introduce causal-window/rounding risk, without changing parameter count.

### Buffer initialization and mmap

**HYPOTHESIS — proposal:** Allocate writable pinned storage directly and initialize only genuinely unwritten tail slots; preserve all validity/future masking semantics. Cost: zero parameters/model FLOPs, unchanged H2D payload, removal of one approximately 228 MB CPU read/write pass plus most of 114 MB zero stores, and no need for a second pageable staging allocation. LoaderPerf owns implementation and verification.

**DEMONSTRATED:** `PackedBar` is mmap-backed, `#[repr(C, packed(4))]`, 36 bytes with alignment 4 (`shared/src/bars.rs:20-43`). Field offsets: timestamp 0, open 8, high 12, low 16, close 20, volume 24, VWAP 28, trade count 32. `BarFile::bars()` is a zero-copy cast (`:417-420`). A window is sequential in this 36-byte AoS layout, although ticker/window choices are shuffled.

**HYPOTHESIS — proposal:** Change mmap advice only after page-fault evidence demonstrates a problem. Re-enabling readahead can reduce cold sequential-window faults but can increase irrelevant I/O/cache eviction across shuffled windows. Cost: zero model parameters/FLOPs, potentially increased disk/memory traffic and pagecache residency; approximately 0 ms expected benefit when already resident. I sent the layout and advice details to LoaderPerf.

## 6. GPU-counter sampler cost and placement

**DEMONSTRATED:** `HardwareSampler::start` imports torch through PyO3 and resolves CUDA UUID on the **training thread**, initializes NVML, then spawns a sampling thread (`benchmark.rs:80-94`). That thread directly calls:

1. `device.utilization_rates()` (`:98`),
2. `device.memory_info()` (`:99`),
3. `device.power_usage()` (`:105`),
4. `recv_timeout(Duration::from_millis(100))` (`:107`).

**DEMONSTRATED:** No `nvidia-smi` subprocess or shell query is launched. The interval is **100 ms plus query duration**, not fixed 100 ms start-to-start. Samples append five `f64` values to a growable vector; around 180 seconds per 1000-step interval produces at most approximately 1800 samples, roughly 72 KB of values. Vector growth here is not a 114 MB-per-step allocation problem.

**DEMONSTRATED:** `finish()` sends the stop signal and joins (`benchmark.rs:119-125`). Timed receive is interruptible, so finishing does not necessarily wait 100 ms, though an in-progress NVML call can delay it. Production closes the step timer before finish (`runner.rs:1704-1705`); sampler startup for the next interval precedes the timer reset (`:2021-2024`). Those endpoint costs are outside the reported step mean, while sampling CPU and driver interaction overlap training.

**HYPOTHESIS:** NVML queries can block, but neither their duration nor their CPU consumption is measured by this code. It would be inaccurate to assign this sampler a demonstrated tens-of-ms cost based on `nvidia-smi` subprocess timings. Direct query latency should be timed separately if it remains suspect, with results written through the existing timing-report base. Proposal cost: zero parameters/model FLOPs/model traffic; a few `Instant` reads per 100 ms cycle and negligible sample metadata; expected steady-state step impact near zero unless NVML/driver contention is demonstrated.

## 7. Negative findings and priorities

**DEMONSTRATED:** No accidental quadratic host loop over dense origins, per-bar `Tensor` indexing, per-channel H2D upload, or `format!` inside the successful per-bar gather loop was found. `ValidBars` does initial invalid-index lookup, then advances monotonically (`corpus.rs:676-727`), rather than binary-searching invalid indices per bar. Batch views are a fixed number of `narrow`/`reshape` operations (`corpus.rs:171-192`), and upload is one packed transfer (`:262`).

**DEMONSTRATED:** `Vec` growth does occur in epoch/report histories and small per-batch descriptors, but the enormous per-bar output is written directly into a pre-sized tensor allocation. The main waste is scalar recomputation and redundant memory passes, not millions of tiny vectors or tensor dispatches.

**HYPOTHESIS — ranked action:** First remove unnecessary pinned-copy/zero passes and redundant scalar work under LoaderPerf's ownership, measuring CPU consumption separately from end-to-end throughput. Second establish matched captured/pinned-source measurement before promising any speedup. Third investigate cold-page and inherited native-thread settings if high CPU remains. Merely reducing Tokio's parked thread count or disabling a sampler without measured cost is unlikely to resolve immense active CPU use.

**DEMONSTRATED:** No claim in this report requires reducing batch size, changing bf16 compute, disabling graph capture, adding accumulation/chunking, or introducing a CPU/eager fallback. No report schema or registry was edited.
