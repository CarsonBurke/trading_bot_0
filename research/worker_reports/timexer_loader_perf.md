# TimeXer segment: host data path, harness truth, and where the captured replay's milliseconds go

`LoaderPerf`. Every number is tagged DEMONSTRATED (measured, reproducible from a command line
given here) or HYPOTHESIS (derived from shapes and measured peaks, not yet run).

The box is shared. During the second audit run three sibling agents were compiling and foreign
tenants held the GPU, so absolute host milliseconds inflate about 1.8x under contention while
ratios hold. Both runs are reported.

## 1. Attribution of the original ~50 ms of `phase: host batch`

DEMONSTRATED, CPU only, real 4,873-ticker corpus, batch 256, context 6000, horizon 192, 12
auxiliary channels. Reproduce:

```
./torch-env.sh cargo build --release -p trading_bot_0
./target/release/trading_bot_0 audit-timexer-segment-loader --batch-size 256 --rounds 20
```

Uncontended run (5 rounds, box otherwise idle), milliseconds per batch:

| component | ms | file:line |
|---|---|---|
| `pin_memory`'s staging copy of the whole packed block (108.8 MiB read + written) | 20.7 | `corpus.rs:509-515` |
| `row.fill(0.0)` over every row (108.8 MiB stored, then overwritten) | 4.0 | `corpus.rs:639` |
| window gather: valid-bar walk, raw OHLC store, validity flag | 1.5 | `corpus.rs:670-690` |
| f64 `ln` of every OHLC element minus the anchor's | 1.4 | `corpus.rs:650-657` |
| auxiliary variates, all six features, 12 channels | ~19 | `features.rs:525-603` |
| packed block allocation (`Tensor::empty`, 108.8 MiB) | 0.005 | `corpus.rs:509` |
| **production total, everything enabled** | **49.82** | |

Contended run (10 rounds, three concurrent cargo builds): production total 44.545 ms, the same
two removable rungs at 36.026 and 6.991 ms, before-total 87.562 ms.

Cross-check against the training run itself: `timexer_segment_timing`'s `phase: host batch` read
48.4, 58.2 and 49.8 ms at steps 1000/2000/3000 of `timexer-control-4k`. The audit's 49.82 ms lands
inside that, so the harness is measuring the same work the trainer does.

**Top two costs, named:** `corpus.rs:509-515` (`Tensor::empty` + `pin_memory`) at 20.7 ms and
`features.rs:525-603` (`AuxiliaryCursor::write`, scalar per-bar per-channel) at ~19 ms.

The per-feature marginal rungs are NOT additive: at batch 256 under contention they sum to 64.2 ms
against a 44.5 ms total, and the market-level rung goes to -0.812 ms. They measure nonadditive
store traffic and cache residency, not separable costs. No decision here rests on them, and the
20-round / feature-count / context sweeps were killed on that basis.

## 2. What changed, and why both changes are bit-identical by construction

Both are pure waste removal. Neither changes a single bit of the batch, so no bit-exactness table
is needed and none is claimed: the deviation is exactly zero by construction, not by measurement.

1. **The pinned block is allocated pinned, not copied pinned.** `Tensor::empty` followed by
   `pin_memory()` allocated a 108.8 MiB pageable block, allocated a *second* 108.8 MiB pinned
   block, and memcpy'd 108.8 MiB of *uninitialized* bytes from the first into the second — every
   one of which the row writers then overwrote. 217.6 MB of single-threaded traffic for nothing.
   ATen reaches the pinned allocator only through `TensorOptions`, so this needed a shim:
   `at_empty_pinned_float` in `vendor/torch-sys-0.25.0/libtch/torch_api.cpp:422`, declared in
   `torch_api.h:99` and `src/lib.rs:39`, wrapped by `torch::cuda::empty_pinned` in
   `trading_bots/src/torch/cuda/mod.rs`. Float-only deliberately: the generated bindings are
   dtype-generic and the scalar-type mapping is an ATen enum the shim would otherwise have to
   reproduce. **20.7 ms.**
2. **The row is no longer zeroed before being written.** `fill_row` zeroed all 108.8 MiB and then
   assigned essentially all of it. Now it zeroes only the tail the assignments do not cover, and
   nothing at all when that tail is empty. The audit's rung confirms the zero was 4.0 ms of
   entirely redundant stores. **4.0 ms.** The `fill_row` tests were pre-existing NaN-initialised
   ones, which is what makes them load-bearing here: a row element the writers do not assign is
   read as NaN, so every existing assertion is also a claim that the element was written. A
   pre-zeroed row would pass either way.

**Net: 49.82 -> 24.20 ms per batch, 2.06x** (uncontended). Contended: 87.56 -> 44.55, 1.97x.

### Refuted: precomputing the log-price array

HYPOTHESIS, refuted on measurement and on bit-identity. The proposal was to cache
`ln(price)` for 470,946,393 bars x 4 channels, on the argument that each distinct bar's logarithm
is recomputed ~32 times per epoch (15.2 billion bar-reads per epoch against 470.9 million distinct
bars). Three reasons it is wrong, and it is recorded here so nobody re-proposes it:

- **It is 2.8% of the problem.** The f64 `ln` rung measures 1.412 ms of a 49.8 ms batch. The work
  was never arithmetic-bound, so a 32x recompute-amplification argument bounds nothing.
- **It cannot be bit-identical.** The row stores `(f64::from(p).ln() - ln_anchor) as f32`. An f32
  log cache cannot reproduce that (the subtraction happens in f64 before the single rounding), and
  an f64 cache is 15.1 GB.
- **It makes the traffic worse.** The bar file must be read anyway for `ts_ms` and `volume`, so the
  cache adds ~198 KB/row of *additional* page traffic on top of the mmap the row already touches.

`StartupCost` has confirmed it is not building the artifact; recorded there as a joint decision.

Also rejected, same reasoning class: `corpus.rs:379` sets `MADV_RANDOM` on the whole mapping while
reads *within* a row are sequential over ~55 pages of a 6,192-bar window, which defeats readahead.
That is `StartupCost`'s region now.

## 3. The fourth instrument problem: the benchmark was not measuring the training path

DEMONSTRATED and fixed. `benchmark.rs`'s warmup and timed loops called `engine.step`, which routes
to `StepGraph::run` with `outputs: None` — the EAGER forward+backward branch — and never called
`arm_step_graph`. Training arms the full forward+backward capture at `runner.rs:1661-1662`. So the
168.59 ms/step this project had been quoting as production throughput, and the 16.98 TFLOP/step and
143.4 GB/step accounting derived from it, described a configuration we never train in.

Fixed: the benchmark now arms the same capture training arms, on the same step, with `warmup >
CAPTURE_AFTER_STEPS` enforced, and splits into three timed arms sharing ONE pre-materialized batch
so the synthetic `gather()`'s 114 MB of `index_select`+`cat` per timed step is out of the window:

- `captured step from a device-resident batch, milliseconds` — no H2D, no loader.
- `captured step from a pinned host batch, milliseconds` — the production configuration.
- `packed row block upload, milliseconds` — the difference between them.
- plus a real `Prefetcher`-fed arm over the actual corpus, so loader cost is measured, not assumed.

Two errors nearly cancelled, which is luck and not accuracy, and it should be stated that way:
`--capture-audit` had already measured replay 167.38 ms against eager 166.96 ms, both fed a
pre-materialized batch outside the timed window, so the eager-vs-captured defect was worth +0.25%
and 168.59 was an over-statement of only ~1.2 ms (0.7%). The downstream roofline reasoning
therefore survives — ~17 TFLOP in ~168 ms is ~101 TFLOPS against ~209 TFLOPS measured dense bf16
peak — but it survived by coincidence.

Docs corrected in the same pass: `docs/timexer_segment.md:71` asserted "Forward and backward are
not graph-captured in the training path" and the following sentence used that false premise to
justify keeping the capture. `runner.rs:1661` arms it. Both the claim and the argument built on it
are rewritten, and the benchmark-reports section now describes three arms.

## 4. The loader is not worth rewriting further, and this is the load-bearing conclusion

DEMONSTRATED. Loop period = `max(device work, host batch service time)`. No scalar crosses to the
host inside a step (the objective's finiteness is accumulated on device and read once per report
interval), so the host runs ahead and the two do not add.

- device: captured replay 160.6-168.8 ms + optimizer 4.85 ms
- host, after the two fixes: 24.2 ms

`runner.rs:1647-1650` measures the wall clock of a *blocking* `loader.receive`, and the sampled
`Cuda::synchronize` at `runner.rs:1653` drains previously queued work. A blocking receive that
returns while the GPU is still executing already-enqueued kernels costs ZERO step time. The 55 ms
`host loader wait` was slack, not stall — corroborated by `timexer_segment_hardware` on the same
run: steady-state `GPU busy` = 100, `memory controller busy` 46-63, `power limit used` 75-90%.
That is not a starved device. The "up to 1.44x" premise was unfounded and is retracted.

So the loader series were split into two distinct truths rather than one misleading one:

- `host batch assembly on the loader thread (interval mean; free below the step)` — from a new
  `loader_build_ms` timed *on the loader thread itself*, at `runner.rs:679-690`.
- `main thread blocked in the loader receive (interval mean; costs step time only above the step)`
  — the old series, with the reading rule now in its own label.

Both live in the existing `timexer_segment_timing` chart, no new base, same unit, same axis.
`phase: host batch` is retained: it is still the one-sampled-step device-synchronized attribution,
and it is what makes the "below the step" claim checkable.

The two fixes therefore buy **host cores, not step time** — real, because the box is shared with
foreign tenants and the user is watching cores saturate, and real again the moment device work gets
cheaper, because then 24 ms is the ceiling instead of 55 ms.

## 5. New instrument: the head, the loss and the SDPA backend

The per-class profile stopped at the last transformer block, so it could not account for its own
step. Added to `model.rs::kernel_classes` (they appear as extra rows in the existing
`timexer_segment_benchmark_kernels`, `_kernel_roofline` and `_kernel_activations` charts — no new
base, nothing to register):

- `head known-future covariate projection`
- `head hidden projection and GELU`
- `head output projection` (which now calls the extracted `head_output_weights`, so the class times
  the real weight/expansion fold rather than a paraphrase of it)
- `targets, market drift and validity mask` (no gradient: pure forward traffic)
- `fused loss geometry and NLL` (the whole `losses` call, driven from a synthetic `Head`)

`every_kernel_class_runs_at_the_shape_it_declares` (model.rs) pins each class to the shape the real
forward produces. A class that declared `d_model` instead of `d_model + COVARIATE_WIDTH` for the
hidden projection would still time a GEMM, just not the one the step runs, and nothing would raise.

**SDPA backend, identified by timing rather than by a precedence table.** `kernel_profile` now
appends four extra rows — `causal SDPA (flash only)`, `(mem-efficient only)`, `(cuDNN only)`,
`(math only)` — by toggling `torch.backends.cuda.enable_*_sdp` around the same class, then
restoring all-enabled. The unforced `causal SDPA` row must equal one of them, which says what
training gets; the others say what the alternatives would cost at 375 tokens, which is the
actionable part. A backend that cannot run the shape reads NaN, which is a measurement.

## 6. Device attribution of the captured replay, PRE-REGISTERED

HYPOTHESIS. Derived from exact shapes (rows 256, origins 375, tokens 96,000, d_model 512, heads 8,
head_dim 64, ffn 2048, layers 8, pred_len 192, HEAD_HIDDEN 1024, 1536 head outputs, 12 aux
channels) against the two measured device ceilings (209 TFLOPS dense bf16 GEMM, 1569 GB/s
device-to-device copy). One channel slice is `U` = 96,000 x 192 = 18.43 M elements = 73.7 MB fp32.

Total charged arithmetic: 5.126 TFLOP/layer-stack forward, 509.6 GFLOP head forward, x3 for
backward = **16.9 TFLOP/step**, which reproduces the project's ~17 TFLOP figure independently.

| class | fwd ms | bwd ms | instances/step | total ms |
|---|---|---|---|---|
| QKV projection | 1.00 | 2.10 | 8 | 24.8 |
| causal SDPA (expected: flash) | 0.50 | 1.20 | 8 | 13.6 |
| attention output flatten | 0.13 | 0.13 | 8 | 2.1 |
| attention output projection | 0.45 | 0.90 | 8 | 10.8 |
| FFN up projection | 1.20 | 2.50 | 8 | 29.6 |
| ReLU^2 | 0.55 | 0.80 | 8 | 10.8 |
| FFN down projection | 1.20 | 2.50 | 8 | 29.6 |
| RMSNorm | 0.15 | 0.28 | 16 | 6.9 |
| QK norm + rotary (fused) | 0.25 | 0.40 | 8 | 5.2 |
| residual/x0 addcmul | 0.19 | 0.30 | 24 | 11.8 |
| patch embedding tokens | 0.60 | 0.90 | 1 | 1.5 |
| head covariate projection | 0.45 | 0.90 | 1 | 1.35 |
| head hidden + GELU | 1.10 | 2.30 | 1 | 3.4 |
| head output projection | 1.90 | 3.90 | 1 | 5.8 |
| targets, drift and mask | 1.41 | 0.00 | 1 | 1.4 |
| fused loss geometry and NLL | 5.70 | 8.60 | 1 | 14.3 |
| **predicted total** | | | | **173** |

Against a measured replay of 160.6-168.8 ms, so the model is coherent to about 3-8%.

**Main's hypothesis is refuted in the form stated, and the refutation matters.** "FFN+QKV are 78%
of charged FLOPs, a nominal ~63 ms at peak, so ~100 ms of a 163 ms replay is elsewhere" charges the
GEMMs at peak. They do not run at peak: at K=512 and N=512/1536/2048 with 96,000 rows the four
projection classes plus the three head GEMMs predict **~105 ms**, not 63. The genuinely non-GEMM
remainder is **~66 ms (39%)**, distributed as:

| non-GEMM work | predicted ms |
|---|---|
| fused loss geometry and NLL (fwd+bwd) | 14.3 |
| causal SDPA | 13.6 |
| residual/x0 addcmul chains | 11.8 |
| ReLU^2 as a standalone kernel | 10.8 |
| RMSNorm x16 | 6.9 |
| QK norm + rotary | 5.2 |
| attention output flatten | 2.1 |
| targets/drift/mask | 1.4 |

The consequence for the fusion program: the ceiling is not 1.6x. Removing **every** elementwise
pass would leave ~105 ms of GEMM (174 TFLOPS, 83% of measured peak, i.e. unreachable). Realistic
bit-exact fusion of the four ranked candidates predicts **-18 to -21 ms of 163, i.e. 1.13-1.15x**.
Worth doing, and worth knowing it is not the 1.44x the loader was once thought to be.

### Fusion candidates, pre-registered predictions

| candidate | mechanism | predicted saving |
|---|---|---|
| fp32 head geometry + pointwise NLL chain | one pass over the head space instead of ~17 geometry passes and 4 x 7 per-channel fp32 passes: 8.98 GB fwd + ~13 GB bwd collapses to ~1.5 GB | **-6.5 ms** (range -4 to -9) |
| FFN GEMM epilogue with ReLU^2 | removes the 393 MB write + 393 MB read of the hidden activation per layer, forward only; backward epilogue fusion is a second step | **-4.0 ms** (range -3 to -5) |
| adjacent attention-residual / x0 elementwise passes | 3 addcmuls per layer over 294.9 MB each merged into 1-2 | **-6.0 ms** (range -4 to -7) |
| forward target temporaries | `(future - log_close)/sigma - drift` in one pass instead of three over 294.9 MB | **-0.9 ms** (range -0.7 to -1.2) |

**Blocker, stated plainly.** Every one of these is a new CUDA kernel in `fused_kernels/`, and the
acceptance bar is bit-exact forward AND backward against the composed ATen form with no tolerance.
Bit-exactness of a CUDA kernel can only be verified by executing it. I may not take a GPU lease or
submit to mlq, so I cannot verify one, and shipping an unverified kernel that alters the training
trajectory is worse than shipping nothing. What I can and did land is the instrument that ranks
them and the pre-registered predictions above; the implementations need one GPU session.

One thing was checked and found NOT available: there is no bit-exact pure-ATen restructuring left
in `losses`. The obvious candidates fail exactly — `(s*r)/LN_2` is not `s*(r/LN_2)` in fp32, and
the common subexpressions (`low` shared by `high` and `open`, `1/sigma` applied to offsets rather
than differences, `exp(-2*CAP*tanh)` factored out of `+ 1/2 ln h`, both reductions as `dot`s) have
already been hoisted. The only remaining exact rewrites are three `addcdiv` folds worth ~0.6 ms of
163, which is not worth touching a bit-exactness-verified loss for.

Rejected direction, restated: nothing here shortens the prediction horizon. `--pred-len 32` and
`cutoff:32` stay rejected; head work stays at 192 horizons.

## 7. GPU verification: exact command lines, predictions pre-registered

```
mlq submit --gpus 1 -- ./torch-env.sh cargo run --release -p trading_bot_0 -- \
  benchmark-timexer-segment --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 \
  --min-history 256 --batch-size 256 --profile \
  --output runs/bench/timexer-corrected-harness
```

Add `--corpus <bars dir> --common-context 6000` to enable the real-loader arm, and
`--capture-audit` to re-measure the eager-vs-captured delta under the corrected ordering.

Pre-registered predictions, before the run:

| quantity | predicted | range |
|---|---|---|
| captured step, device-resident batch | 167.5 ms | 160-176 |
| captured step, pinned host batch (production) | 168.0 ms | 160-177 |
| packed row block upload (pinned minus device-resident) | 0.4 ms | 0.0-1.5 |
| captured step, real corpus loader arm | 168.2 ms | 160-178 |
| origins per second | 1,524 | 1,455-1,600 |
| achieved TFLOPS (16.9 TFLOP / step) | 101 | 96-106 |
| selected SDPA backend | flash | |
| `causal SDPA (flash only)` forward | 0.50 ms | 0.35-0.75 |
| `causal SDPA (cuDNN only)` forward | 0.45 ms | 0.30-0.80 |
| `causal SDPA (mem-efficient only)` forward | 0.90 ms | 0.6-1.6 |
| `causal SDPA (math only)` forward | 6.0 ms | 4-10, eligible (1.15 GB fp32 score matrix fits) |
| `fused loss geometry and NLL` fwd+bwd | 14.3 ms | 9-20 |

And for the training run after the loader fixes (`timexer_segment_timing`):

| series | before | predicted after |
|---|---|---|
| training step (interval mean) | 177.2-183.7 ms | **unchanged**, 177-184 ms |
| host batch assembly on the loader thread | n/a (new series) | 24-26 ms |
| main thread blocked in the loader receive | 55.4-56.2 ms | 22-26 ms |
| phase: host batch | 48.4-58.2 ms | 24-26 ms |
| phase: H2D | 2.0-2.5 ms | unchanged |

The step time prediction is deliberately "unchanged". The loader fix removes 25 ms of host work
that sat inside a 165 ms device period, so predicting a throughput win from it would be predicting
the same error that produced "up to 1.44x". If step time *does* fall, the loop-period model is
wrong and that is the finding.

## 8. Files touched

- `trading_bots/src/torch/timexer_segment/corpus.rs` — pinned-direct allocation, tail-only zero,
  `LoaderPhase`/`audit_host_batch` ladder, `host_batch`, `fill_row` const-generic rungs.
- `trading_bots/src/torch/timexer_segment/features.rs` — audit hooks only, no arithmetic change.
- `trading_bots/src/torch/timexer_segment/benchmark.rs` — corrected three-arm harness, real-loader
  arm, `LoaderAuditArgs`, `kernel_profile` refactor, forced-SDPA-backend rows.
- `trading_bots/src/torch/timexer_segment/model.rs` — `head_output_weights` extraction, five new
  kernel classes, `every_kernel_class_runs_at_the_shape_it_declares`.
- `trading_bots/src/torch/timexer_segment/runner.rs` — `Prefetcher` loader-thread build timing and
  the two `loader.receive` call sites; the `loader_build_ms` accumulator.
- `trading_bots/src/torch/timexer_segment/reports.rs` — the `timexer_segment_timing` chart call and
  the two split loader series, inside the writer carve-out only.
- `trading_bots/src/main.rs` — `audit-timexer-segment-loader` subcommand.
- `trading_bots/src/torch/cuda/mod.rs` — `empty_pinned`.
- `vendor/torch-sys-0.25.0/libtch/torch_api.{cpp,h}`, `src/lib.rs` — `at_empty_pinned_float`.
- `docs/timexer_segment.md` — corrected the false capture claim and the argument built on it; three
  benchmark arms; host loader audit section.

## 9. Scoped verification, DEMONSTRATED

Re-run on the final tree, after `SignalHistory`, `StartupCost` and `AmplitudeCal` all landed, so
these are the state of the repository and not of my copy of it:

```
./torch-env.sh cargo check -p trading_bot_0 --tests          # 0 errors
./torch-env.sh cargo test -p trading_bot_0 timexer_segment   # 105 passed, 0 failed
./torch-env.sh cargo check -p trading-bot-tui --tests        # 0 errors
./torch-env.sh cargo test -p trading-bot-tui                 # 36 passed, 0 failed
```

`model::tests::every_kernel_class_runs_at_the_shape_it_declares` is the new one and it passes on
CPU, which is the whole point: the profile's rows are pinned to the shapes the real forward
produces without needing a device.

## 10. `timexer_segment_startup`, and why it is not a series on the timing chart

Arbitrated with `StartupCost` and landed. Their ten-plus corpus-load phases were proposed as
series on `timexer_segment_timing` with the axis changed to Symlog. Refused as owner of that
chart: it answers "where does one step's wall clock go" with every series a per-step millisecond
on one linear axis, and a 172,100 ms once-per-run constant is a different question in the same
unit. Symlog there would have traded a readable 2-180 ms axis - the axis on which the 55 ms slack
finding is legible at all - for a decade grid, and ten flat constants would have taken ten of that
chart's colours for numbers that never move.

Landed instead as its own base, `timexer_segment_startup`, `ScaleKind::Symlog`, y-label
"milliseconds, one-time", scope line carrying `audits_computed` of total audits and the rescanned
bar count so a rebuild is legible as a rebuild rather than as a regression. It consumes every
`LoadTiming::phases()` entry except the last, which is a bar count rather than a millisecond, so
`StartupCost` can add phases without touching the writer. Registered by `SignalHistory` in
`shared/src/report.rs:102`; `tui`'s `meta_chart_bases` extends from the registry, so both sides
agree and the bidirectional test stays green. A phase that did not run reads NaN, not zero, and
the chart helper passes NaN through - that is what makes a warm-cache run distinguishable from a
phase that ran and found nothing to do.

One correction to the brief's corpus arithmetic, from `StartupCost`'s measurement: startup
traverses 5,728 files / 783,074,510 records / 28.19 GB. The 470,946,393 figure is the
post-quarantine VALID training-target subset, not what the load walks.
