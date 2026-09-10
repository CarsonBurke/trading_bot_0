# Non-training pass throughput: what the diagnostics actually spend, and the one fix that mattered

Scope: `ceiling-timexer-segment`'s `ceiling_pass` / `ceiling_placement`, the `Prefetcher`/`Batch`
path as the non-training passes use it, `CeilingAccumulator`, and — granted mid-task — the
origin-timestamp gather in `corpus.rs`.

Reference checkpoint `training/runs/timexer-control-4k/weights/best`, step 2000, epoch 1,
`weights_sha256 = ccffa620654f1c95faa618501519e036249615e98d6034e425e3fed7c2309116`, verified
against the manifest **and** recomputed from `model.safetensors` (both equal) before the first
job.

Every claim is DEMONSTRATED with a command or `file:line`, HYPOTHESIS, or PRE-REGISTERED.

---

## 1. Headline: the pass body was never the problem; page-fault latency was

**DEMONSTRATED.** The scored loop of a ceiling pass is **forward-bound at 96.4%** and has no
structural fix in it. What a diagnostic lease was actually losing is host-side locality: the
origin-timestamp gather walked a **timestamp-ordered** reference list across a **ticker-major**
memory mapping, taking a cold page for essentially every element — ~3.6 M lookups at ~25 µs.

`Corpus::origin_timestamps` (`corpus.rs`) now sorts the indices by `(ticker, origin)`, gathers
in parallel — rayon splits an indexed parallel iterator into contiguous ranges, so each worker
walks one ticker's origins in ascending order and ~512 `i64` headers share a 4 KiB page — and
inverts the permutation on the way out. It is **order-preserving, not merely order-equivalent**:
the returned vector is element-for-element the one the serial `map` returned, so no statistic
downstream can distinguish them. Routed through `whole_timestamp_draw`, `ceiling_pass`, `score`,
both probe gathers and `distinct_timestamps`. `portfolio_data.rs` and `supervision.rs` were left
to their owners.

| quantity | before | after | pre-registered | verdict |
|---|---:|---:|---|---|
| per-pass `setup_ms`, 133,568 origins (job 5548 → 5554) | **6,600 ms** | **0 ms** (point 0), **500 ms** (point 1) | "under 1,000 ms" | **CONFIRMED** |

The prediction was written before job 5554 existed and is quoted verbatim in §5, PR-4.

---

## 2. Two retractions that preceded any measurement

**DEMONSTRATED — the 189 ms/batch premise divided the wrong numerator by the wrong denominator.**
`ceiling_placement` scores **three** populations (`runner.rs`: `training` whole-timestamp draw,
`calibration block`, `held-out full`), so the anchored pass is ~1.15 M origins / 256 =
**~4,483 batches**, not 1,693. And the `in {:.1} s` field it was divided from is
`started.elapsed()` from the top of `ceiling_placement`, covering three draws, three passes and
three `finish` calls — while the sentence it sits in names only `held-out full`. That labelling
defect is fixed: the line now reads `in 364.3 s of which 272.8 s scored three populations`.

Provenance of the quoted seconds: `319.2 s` was attempt 4182 (`/var/tmp/tb0_v19`,
`--placement both`), the anchored leg. Job 5529's own anchored figure is `337.5 s`.
Command: `grep -E "ceiling placement|origins in " ~/.local/state/mlqueue/attempts/{4123,4138,4182}/stdout.log`.

**DEMONSTRATED — the 225 W was the prologue, not this workload.** Sampled live during 5529's
scored phase: `451.64 W` of `575 W`, SM `2797 MHz`, `15,642 MiB` of `32,607 MiB` board
(`nvidia-smi --query-gpu=name,memory.used,memory.total,power.draw,power.limit,clocks.sm,utilization.gpu --format=csv`).
With the foreign tenant separately measured at `8,773 MiB / 71.47 W` on an otherwise idle card,
ours was ~6.9 GiB and ~380 W.

---

## 3. Per-phase attribution of the batch — DEMONSTRATED

Job 5544, batch 256, full three-population anchored ceiling. Binary `/var/tmp/tb0_evalthru`,
sha256 `2d492ec8da72bd982bfca263ca85b98bd3df4bf88f00b1224b9e02d6051e4b6e`, exit 0.

Method: `ceiling_pass` samples 64 batches per population; on a sampled batch the device is
drained *before* `receive` and each of the four spans is closed by `Cuda::synchronize`, so the
four partition that batch's wall clock exactly. 191 samples of 4,487 batches = 4.3%.

| phase | ms/batch | share of the 60.8 ms period | method |
|---|---:|---:|---|
| `final_origin` forward (statistics, 8-layer backbone over 375 tokens, last-origin head) | **58.61** | **96.4%** | DEMONSTRATED — synchronized span, mean of 191 samples |
| `CeilingAccumulator::accumulate`, nine fp64 per-timestamp `index_add_` | **0.86** | **1.4%** | DEMONSTRATED — synchronized span, same samples |
| exposed host loader wait | **0.56** | **0.9%** | DEMONSTRATED — host clock around `Prefetcher::receive` on **all** 4,487 batches |
| H2D upload of the packed rows, exposed | **≤ 0.77** | **≤ 1.3%** | DEMONSTRATED as a residual `60.8 − 58.61 − 0.86 − 0.56`; **NOT separated** from launch overhead in this job. Separated in job 5554: 4.23 ms at 256 rows and 8.52 at 512, exactly linear in bytes, fully hidden by run-ahead |
| host batch assembly, loader thread | 15.0, **overlapped** | 0.25× the period | DEMONSTRATED — timed on the worker thread in `Prefetcher::with_mode` |

### The `upload 183.60` that job 5544 printed is an instrumentation artefact, reported as one

A 183.6 ms span cannot live inside a 60.8 ms period. What it measured was the **device
backlog**: once the per-batch host sync was removed from `accumulate`, the host runs ahead of
the device, so the first synchronization inside a sampled batch drains everything queued and
charges it to whichever phase closed first. That is simultaneously the cleanest evidence that
**the host is not the bottleneck** and that **the removed sync was real**. Fixed in the binary:
a sampled batch now drains first, charges that drain to no phase, and reports run-ahead as its
own quantity — **164–329 ms, a steady 2.9 batches at both 256 and 512 rows** (job 5554).
`forward` and `accumulate` above were unaffected, because both were measured after the backlog
had already been drained by the upload sync preceding them.

### Every candidate in the brief, refuted

| candidate | verdict | evidence |
|---|---|---|
| (a) prefetch depth of one cannot hide host assembly | **REFUTED** | assembly 15.0 ms = 0.25× the period; exposed wait 0.56 ms; host measured 2.9 batches AHEAD |
| (b) `resident`/`upload` re-allocating or re-uploading per batch | **REFUTED** | upload 4.23 ms at 256 rows, linear in bytes, hidden by run-ahead; `resident` is reallocated only on a row-count change, twice per population |
| (c) the nine fp64 scatter-accumulates dominate | **REFUTED** | 0.86 ms/batch, 1.4%. fp64's 1/64 rate does not bite: `index_add_` is one add per element — 9 × 256 × 192 × 8 B = 3.5 MB of atomics, not arithmetic |
| (d) `final_origin` materializes all 192 horizons at every token position | **REFUTED** | `model.forward(batch, &stats, false, true)` already passes `last_only = true` (`runner.rs:996`). The genuinely unused tensors it still builds — `prices`, `rebased_prices`, `target_prices`, `mid` (`runner.rs:1012-1015`) — are 3 `exp()` and ~6 elementwise ops over `[256,192,4] = 196,608` elements, ~63 MB of traffic, **< 0.1 ms** at this card's measured copy peak: ≤ 0.16% of a batch. HYPOTHESIS by shape arithmetic, not measured separately |

**The pass body is at its hardware limit for this shape.** The residual inefficiency —
≤ 51% of the card's own measured bf16 GEMM peak — is inside the forward at fixed shape, the
same ~45–50% `timexer_flop_inventory.md` charges the training step, and it is a GEMM-efficiency
question at `d_model = 512`, explicitly out of this task's scope.

---

## 4. Where a diagnostic lease actually goes — DEMONSTRATED

Job 5544, wall 384.3 s for job 5529's equivalent (attempt 4195 `exec.json` 1788888658353 →
`result.json` 1788889042664):

| segment | seconds | GPU work |
|---|---:|---|
| corpus load total | 5.6 | none |
| `anchor_cross_section_draws` | 22.4 | none, CPU only |
| `ceiling_placement`, scored loop, three populations | 272.8 | yes |
| `ceiling_placement`, everything else — draw construction + three per-pass setups + three `finish` | **91.5** | almost none; `finish` is 0.19 s of it |
| CUDA context, model + checkpoint load, report writes, teardown | ~20 | none |

**~140 s of a ~420 s job — a third — had no GPU work, and the 91.5 s was invisible to every
number the run printed.** §1's locality fix targets exactly that 91.5 s. The corpus load and
the CPU-side anchoring belong to `corpus.rs`'s band owner and were left alone.

**Discrepancy recorded, not resolved:** `anchor_cross_section_draws` measured 20,524.9 ms
(5529), 15,654.9 ms (5542) and 22,448.3 ms (5544) on the identical command and corpus — 31%
spread. It is run-to-run variance, not a population difference, and it is not mine.

---

## 5. Pre-registrations and their outcomes

Each was written into this file before its measuring job was submitted; the git history is the
timestamp.

### PR-1 — instrumentation cost and phase split at 256 rows → **mostly CONFIRMED**

| prediction | outcome |
|---|---|
| forward 55–68 ms | **58.61** ✓ |
| accumulate 2–8 ms | **0.86** — below the band, i.e. the fp64 banks are even cheaper than predicted ✗(low) |
| H2D upload 2–5 ms | **4.23** ✓ (measured in 5554 after the artefact was fixed) |
| exposed loader wait 0–15 ms | **0.56** ✓ |
| loader assembly 40–70 ms/batch, below the period | **15.0**, 0.28× the period — far below the band, same direction ✗(low) |
| peak allocator 3,000–3,200 MiB | **3,071** ✓ |
| mean power 400–470 W | **399** ✓ (447–460 in the quieter 5554/5556 window) |
| ≤ 40% of measured GEMM peak | **≤ 50%** ✗ — the achieved fraction is higher than predicted |
| *direction of the bet: the body is near its limit and (a)–(d) are all small* | **CONFIRMED** |

### PR-2 — unpaired batch curve → **SUPERSEDED** by PR-3 before any job ran

Withdrawn on Main's ruling: with a 15.4% swing between identical runs (§7), four separately
leased points cannot distinguish a knee from a foreign tenant.

### PR-3 — paired A-B-A sweep → **REFUTED by its own guard**, and the guard was mine

Job 5548, exit 101. Predicted `max_statistic_difference == 0` at every point; measured
**4.879e-5** between 256 and 512 rows and the run refused itself. Predicted peak allocator
~1.1 GiB at 256; measured **2,690 MiB** — activations, not accumulator banks, dominate on this
population. The 4.879e-5 is eight orders above what fp64 regrouping of ~10⁵ terms can produce
(~10⁻¹³), so the pre-registration was too coarse: it lumped a claim that must hold with one
that cannot. See §6.

### PR-4 — locality fix and the split guard → **CONFIRMED**

Written before job 5554. "Per-pass `setup_ms` at 133,568 origins falls from the measured 6.6 s
to **under 1.0 s**": measured **0.0 s** and **0.5 s**. The companion prediction, that the
three-population draw construction falls from 91.5 s to under 8 s, is **UNMEASURED** — it needs
a full three-population run, which the queue did not permit; the sweep path builds one draw, not
three. A successor should measure it.

### PR-5 — final sweep with the two-claim bar → **CONFIRMED except the one clause that taught us something**

Written before job 5556 existed:

| prediction | measured (job 5556) | verdict |
|---|---|---|
| target-only ≤ 5e-6 at 512 and 1024, expect ~3e-8 and ~5e-8 | **2.730e-8 at both** | CONFIRMED |
| model difference ≤ 2e-4 | **4.879e-5** (512), **4.650e-5** (1024) | CONFIRMED |
| the two 256-row points bit-identical in the **model** series | **exactly 0.000e0** | CONFIRMED |
| the two 256-row points bit-identical in the **target-only** series | **4.441e-16** | **REFUTED** — see §7 |
| setup under 1.0 s at every point | 0.0 / 0.4 / 0.4 / 0.4 s | CONFIRMED |
| 1024 peak allocator ~10.5 GiB, no OOM | **10,334 MiB**, board 21.0 GiB with tenants | CONFIRMED |
| 1024 within 10% of 256 in origins/s | **−0.84%** | CONFIRMED |

---

## 6. The batch-size curve — a defended negative

Job 5556, `/var/tmp/tb0_evalsweep4`, **one process, one CUDA context, one corpus residency**,
133,568 held-out full origins of 408,239, ANCHORED, batch order `256, 512, 1024, 256`. A curve
assembled from separately leased runs is inadmissible on this machine (§8): the A-B-A form makes
the drift a first-class row instead of hiding it in the B.

**Read the artifact correctly: job 5556 exited 101, and the measurement is complete anyway.**
All four points ran, printed, and were compared; the process then failed on the *last* line of
its own acceptance guard, because that guard demanded exact equality of the target-only series
between the two 256-row points and got 4.441e-16 (§7). Nothing in the curve below is affected —
the guard runs after every point has been measured and reported. The guard has since been
split; the numbers were not re-measured, because a re-run would buy an exit status and nothing
else, on a card a training arm needs.

Achieved fraction of the card's **own measured peak** first, because that is the quantity that
decides the question; the wall clocks are downstream of it.

| rows | achieved fwd TFLOPS | measured bf16 GEMM peak | fraction | origins/s | vs. first A |
|---:|---:|---:|---:|---:|---:|
| 256 (A) | ≤ 105.7 | 202.6 | **≤ 52%** | 4,778 | — |
| 512 | ≤ 106.8 | 203.1 | **≤ 53%** | 4,830 | **+1.09%** |
| 1024 | ≤ 104.8 | 201.2 | **≤ 52%** | 4,738 | **−0.84%** |
| 256 (A′) | ≤ 106.4 | 200.7 | **≤ 53%** | 4,812 | **+0.71%** ← the error bar |

| rows | ms/batch | forward ms | upload ms | accumulate ms | exposed wait ms | host assembly ms | peak allocator MiB | board MiB | mean W | peak W |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 (A) | 53.6 | 53.43 | 4.09 | 0.59 | 0.73 | 14.0 | 2,690 | 13,527 | 458 | 477 |
| 512 | 106.0 | 97.14 | 8.10 | 0.56 | 1.05 | 27.2 | 5,238 | 15,540 | 475 | 489 |
| 1024 | 215.2 | 194.59 | 16.33 | 0.58 | 13.71 | 58.3 | 10,334 | 21,005 | 474 | 493 |
| 256 (A′) | 53.2 | 49.57 | 4.12 | 0.54 | 0.25 | 14.5 | 2,690 | 20,869 | 475 | 488 |

**The A-A′ drift is +0.71%. Every point lies within about one and a half drifts of every other
point, and every point sits at 52–53% of the card's own peak.** Batch size is a **dead lever**,
measured at three sizes across a 4× span rather than extrapolated from two.

Reading the rest of the table: upload is **exactly linear in bytes** (4.09 → 8.10 → 16.33 for
256 → 512 → 1024) and stays hidden; host assembly is a **constant 0.26–0.27× of the period** at
every size, so the loader scales with the work and never becomes the bottleneck; `accumulate`
is **flat at ~0.56 ms regardless of batch size**, which is the last nail in candidate (c) — the
nine fp64 banks do not grow with the batch because the row count per pass is fixed. The only
thing 4× more rows buys is 4× the allocator, and the exposed wait at 1024 (13.71 ms) is the
prefetch depth of one finally starting to show, at the size with the worst throughput.

**Default unchanged at 256, with a measured basis.** Raising it would trade 3.8 GiB of a
contended card for a difference smaller than the run-to-run drift, and would price the
diagnostics out of co-residency with a training arm — job 5556's own 1024 point took the board
to 21.0 GiB.

The residual is *inside the forward at a fixed shape*: the same ~45–52% of peak
`timexer_flop_inventory.md` charges the training step. That is a GEMM-efficiency question at
`d_model = 512` and is explicitly not this task.

---

## 7. The exactness bar, in three claims

Batch size regroups all nine fp64 `index_add_` reductions, so "the throughput knob does not move
the measurement" is a real claim that has to be checked on the real corpus. Three measurements
decided its shape.

**Across shapes, the model-dependent series** (`student_ic`, `gap`) moved by **4.879e-5**
between 256 and 512. They read the model's forecast, which arrives through bf16 GEMMs whose
cuBLAS kernel and split-K decomposition are chosen from the operand **shape**. Requiring
bit-equality here is requiring shape-invariance from cuBLAS, which it does not offer.

**Across shapes, the target-only series** (`ceiling`, `martingale_ratio`) moved by **2.730e-8**
— five orders above what fp64 regrouping can produce, so the nine banks are exonerated by
magnitude rather than by assertion. The targets are **fp32 by design** and reach the banks
through per-row reductions whose CUDA block decomposition is chosen from tensor size, exactly as
cuBLAS chooses a GEMM kernel. An fp32 quantity near 1 carries ~10⁻⁷ of its own representation
error, so 2.7e-8 is one ulp of arithmetic that was always present and that shape merely surfaces.

**At a REPEATED shape** — the comparison where nothing may select differently — the model series
came back **exactly 0.000e0** and the target-only series came back **4.441e-16**. That split is
the single most useful pair of numbers the sweep produced, and the first of them is a
project-level result rather than a throughput one:

- **The forward is bit-reproducible at a fixed shape.** Every paired comparison this project
  makes — `FuseLoss`'s A-B-A, the placement before/after, `LatentProbe`'s paired gap, every
  step-matched arm verdict — rests on that, and until job 5556 it was an assumption. It is now
  measured on the real corpus at the reference checkpoint: two passes over one population at
  one shape produced identical student IC and identical gap at every one of 192 horizons.
- The nine banks are **not** reproducible, and cannot be: `index_add_` scatters with **atomics**
  on CUDA, so the summation order of the ~10⁵ rows landing in one timestamp bank is whatever the
  hardware scheduled. 4.441e-16 is two ulps of an fp64 quantity near 1. Nothing selected
  differently; the *order* did.

That by-product is worth more than the guard it broke. **4.441e-16 is a direct run-time
measurement of the reordering error the fp64 accumulators exist to keep negligible** — ten
orders of magnitude inside the resolution anything is printed at. The same atomic scatter in
fp32 would sit near 10⁻⁷ per bank *before* the cancellation the doc comment on
`teacher::CeilingAccumulator` is worried about. That comment has been arguing the point from
first principles; it now has a number from the real corpus.

The bar the code enforces (`runner.rs` `throughput_sweep`, `PRINTED_STATISTIC_TOLERANCE`,
`FP64_SCATTER_TOLERANCE`, `teacher.rs` `CeilingCurve::max_statistic_difference`):

1. **Repeated shape, model series: exactly 0.** A determinism claim, exact bar, no band.
2. **Repeated shape, target-only series: ≤ 1e-12**, the fp64 scatter-reordering floor — four
   orders above the measured 4.441e-16, so a bank that silently dropped to fp32 (~10⁻⁷) or lost
   a term is caught while atomic scheduling is not.
3. **Across shapes, target-only series: ≤ 5e-6** — *half a unit in the last printed place*,
   since every ceiling console line and chart title carries five decimals. The bar's provenance
   is a **presentation fact**, not a tolerance: it is the resolution at which the measurement is
   read, and it cannot be widened without changing how the measurement is printed. The model
   difference is reported against the curve's own h=1 standard error (4.879e-5 = **0.31% of one
   SE**) and is never used to excuse a target-only failure.

Additionally, every sweep point prints `ceiling`, `variance ratio`, `student IC` and `gap` to
**nine decimals at each of `reports::DECISION_HORIZONS`**, so "unchanged to the printed
precision" is readable horizon by horizon rather than resting on one scalar. Job 5556's log
carries all four series at h = 1, 8, 16, 32, 64, 128, 192 for all four points; the target-only
columns are character-identical across every point.

---

## 8. Run-to-run variance and the card's own peak — DEMONSTRATED

| job | binary | anchored placement (ms) | `ceiling_placement` wall (s) | measured bf16 GEMM peak |
|---|---|---:|---:|---:|
| 5529 | `tb0_v20` | 20,524.9 | 337.5 | not recorded |
| 5542 | `tb0_v21` | 15,654.9 | 292.6 | not recorded |
| 5544 | `tb0_evalthru` | 22,448.3 | 364.3 (272.8 scored) | **187.4 TFLOPS** |
| 5548 | `tb0_evalsweep2` | 20,033.9 | sweep | 195.4 / 195.6 |
| 5554 | `tb0_evalsweep3` | — | sweep | 194.1 / 193.6 |
| 5556 | `tb0_evalsweep4` | — | sweep | 202.6 / 203.1 / 201.2 / 200.7 |

**15.4% between 5529 and 5542 on the identical quantity, 31% on the CPU-side anchoring.** Job
5544's card measured **187.4 TFLOPS against the 230.99 quiet-card reference**
(`benchmark.rs` `QUIET_CARD_GEMM_TFLOPS`, job 5399) — **18.9% down**, a contended card. No
cross-run wall-clock claim on this machine is admissible, which is why the batch curve is
measured inside one process with a repeated point as its error bar.

Job 5556's four points are the counter-example that makes the method work: measured **inside
one process**, the card's own peak varied by only **1.2%** (200.7–203.1) and the repeated point
drifted **0.71%**. The same quantity measured across separate leases swung 15.4–18.9%. The
error bar is not the hardware; it is the lease boundary.

---

## 9. Instrumentation landed

- `runner.rs` `PassTiming`, `ceiling_pass`, `report_pass_throughput`, `throughput_sweep`,
  `PRINTED_STATISTIC_TOLERANCE`, `FP64_SCATTER_TOLERANCE`; `CeilingArgs::throughput_sweep` /
  `throughput_origins`.
- `teacher.rs` `CeilingAccumulator::retained` moved onto the device (exact by construction —
  an `Int64` sum of a 0/1 mask — and it deletes the one per-batch device→host sync a
  non-training pass contained); `CeilingCurve::max_statistic_difference`.
- `teacher.rs` `CeilingAccumulator`'s fp64 rationale now quotes the measured 4.441e-16 atomic
  reordering floor next to the first-principles argument it confirms.
- `benchmark.rs` `write_eval_phases`; `device_peaks` raised to `pub(super)`.
- `corpus.rs` `Corpus::origin_timestamps`.
- `shared/src/report.rs`: **`timexer_segment_eval_phases`** appended at EOF of
  `TIMEXER_SEGMENT_REPORT_BASES`, with the reasoning for a new base rather than rows in
  `timexer_segment_timing` (that base is the TRAINING step's; its x axis is the optimizer step,
  this one's is the batch ordinal inside one diagnostic pass, and the two clocks cannot share an
  axis). `tui/src/main.rs` `meta_chart_bases` extends from the registry, so both directions of
  the bidirectional test follow automatically; the test was run and is green.
- Hardware is charted through the existing `timexer_segment_hardware` base rather than a
  duplicate, written into each pass's own output directory.

Presentation: `timexer_segment_eval_phases` carries one unit (milliseconds), one axis (batch
ordinal within the pass), one question (where does one batch of a diagnostic go). The four
series partition the sampled batch. Run-ahead is deliberately **not** a fifth series — it is a
different question in the same unit — and rides in the title instead.

### Verification

- `./torch-env.sh cargo check -p trading_bot_0 --tests` — zero errors (the one
  `unused_must_use` warning is pre-existing in `bar_dist.rs:7323`).
- `cargo test -p trading-bot-tui the_meta_chart_list_looks_for_every_registered_writer_base` —
  1 passed, 0 failed: the bidirectional registry test agrees in both directions after
  `timexer_segment_eval_phases` was appended.
- `./torch-env.sh cargo check -p trading-bot-tui --tests` — zero errors.
- `./torch-env.sh cargo test -p trading_bot_0 --lib timexer_segment::runner` — **16 passed, 0
  failed** (0.26 s); `::teacher` — **11 passed, 0 failed** (0.33 s); `::corpus` — **11 passed,
  0 failed, 2 ignored** (0.10 s). Those three modules are the entire blast radius of this task.

The unfiltered `cargo test -p trading_bot_0 timexer_segment` did **not** complete: 900 s at
`--test-threads 1` got as far as `model::tests::fused_loss_matches_the_reference_decode_...`,
i.e. it is still inside `model::tests`, which builds and trains CUDA models per test. That is
pre-existing and unrelated — every module this task touched runs in under a second, and the
individual tests the parallel harness reported as "running for over 60 seconds" complete in
**0.04 s** when run alone (`tensor_scorer_matches_scalar_reference`); they were blocked on the
serialized GPU lock, not slow. A successor should not read that as a regression.

Also cleared: `cargo test` binaries orphaned by earlier harness timeouts, one of them alive for
2 h 17 m, holding the serialized test lock. `pgrep -f out/trading_bot_0` before diagnosing a
slow suite here.

### The per-horizon reproducibility floor — DEMONSTRATED, job 5556 log

Asked for by `Main` on behalf of a pre-registered decision rule whose primary statistic is the
SHAPE difference Δ(h=1) − Δ(h=8). Read directly out of 5556's captured log; no reconstruction.

**At a repeated shape (A = point 0, A′ = point 3, both 256 rows, same population, same
checkpoint) every printed digit is identical and the aggregate difference over all 192 horizons
is exactly `0.000e0`.** The floor is not small, it is zero:

| h | close cross-sectional IC, A | A′ | A′ − A | market-neutral MSE ratio |
|---:|---:|---:|---:|---:|
| 1 | 0.165353799 | 0.165353799 | **0** | 0.894920 |
| 8 | 0.120567767 | 0.120567767 | **0** | NOT MEASURED |
| 16 | 0.099048402 | 0.099048402 | **0** | NOT MEASURED |
| 32 | 0.107512671 | 0.107512671 | **0** | NOT MEASURED |
| 64 | 0.109323363 | 0.109323363 | **0** | NOT MEASURED |
| 128 | 0.089309034 | 0.089309034 | **0** | NOT MEASURED |
| 192 | 0.079401548 | 0.079401548 | **0** | NOT MEASURED |

The MSE ratio column is `1 − ceiling²` and the ceiling is **NaN at every h ≥ 8 on this corpus**
— the ceiling instrument is null there, which is a separate project-level finding. It is
reported as NOT MEASURED rather than reconstructed from anything else.

**The +0.71% is a THROUGHPUT drift in origins/s. None of it is statistical.** Any reading that
treats 0.71% as an error bar on a scored quantity is a unit error.

Across a CHANGED shape the per-horizon differences do exist, and they **scatter in sign**:

| h | IC delta, 512 − 256 | IC delta, 1024 − 256 |
|---:|---:|---:|
| 1 | **+9.68e-6** | **+9.64e-6** |
| 8 | **−2.87e-6** | **−2.03e-6** |
| 16 | −2.67e-5 | −2.74e-5 |
| 32 | +2.08e-5 | +2.08e-5 |
| 64 | +2.60e-5 | +2.71e-5 |
| 128 | +4.00e-5 | +3.88e-5 |
| 192 | +2.33e-5 | +2.21e-5 |

So the premise "a common level drift cancels in Δ(h=1) − Δ(h=8)" is **not supported**: h=1 and
h=8 carry *opposite* signs, so the shape statistic ADDS them. Δ(1) − Δ(8) = **1.25e-5** at 512
and **1.17e-5** at 1024, larger than either single-horizon delta. Reassuringly the two
independent shape changes agree with each other to ~10%, so this is a reproducible property of
kernel selection and not noise.

**Consequence for any paired arm: score both sides at the SAME batch size and the floor is
exactly zero at every horizon, level and shape alike.** Score them at different batch sizes and
the shape statistic inherits ~1.2e-5, which is 0.15% of the h=1 standard error of 0.00839 —
still negligible, but no longer zero and no longer cancelling.

**The range of that 1.2e-5 is 256 → 512 and 256 → 1024. It is NOT measured downward, and in
particular 256 → 64 is UNMEASURED.** Both points above are *upward* power-of-two changes that
keep the GEMM in the same large-M regime; 64 rows at `d_model = 512` over 375 tokens may select
a small-M kernel family with a different split-K decomposition, where the delta could be larger
by an order of magnitude or more. Since `--eval-batch-size` now defaults to **64** while
`timexer-control-4k`'s step-2000 numbers were scored at **256**, the control comparison sits
exactly on the unmeasured edge. Quoting 1.2e-5 for it would be an extrapolation presented as a
measurement, which is the failure class this report exists to avoid.

Closing it costs one ~3-minute job and no code — `--throughput-sweep 256,64,256`, which
re-uses this harness and carries its own repeated-shape zero as a control. PRE-REGISTERED
before that job exists: target-only ≤ 5e-6 (expect ~3e-8); **model difference 2e-5 to 2e-4**,
deliberately wider than the upward 1.2e-5 because the prediction is that the small-M regime is
*worse*, not the same; repeated 256 exactly 0 in the model series; 64-row peak allocator
650–750 MiB; and origins/s **down at least 15%**, since 256 already runs at 52% of peak and 64
has four times less GEMM to hide the fixed costs behind.

**PR-6, extended on Main's request before job submission** — sweep `256, 64, 128, 256` so one
lease answers the default question as well as the floor question. Added predictions: the 128
model delta lands **between the 64 delta and the upward 1.2e-5**, closer to 1.2e-5, because 128
is nearer the measured regime; 128 peak allocator **1.30–1.40 GiB**; 128 origins/s **within 8%
of 256** while 64 is down **≥ 15%**. Main's own rule, pre-registered against these: move the
default to 128 iff the 128 delta is ≥ 3× smaller than 64's AND 64 costs > 15% of 128's
origins/s.

---

## 10. PR-6 outcome — job 5582, the downward shape change measured

`--throughput-sweep 256,64,128,256`, one process, 133,568 anchored `held-out full` origins,
binary `/var/tmp/tb0_evalsweep4`. **The card was contended** — a gated training arm was polling
and foreign tenants held ~17.9 GiB — so all four points are equally contended and mutually
comparable, and **none of them is comparable to §6's quiet-card numbers**. That is why the
repeated point exists.

| point | rows | ms/batch | origins/s | forward ms | upload ms | accumulate ms | peak allocator |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 256 | 110.9 | 2,306 | 111.90 | 4.09 | 3.71 | 2,690 MiB |
| 1 | 64 | 26.9 | **2,382** | 25.34 | 1.10 | 3.51 | **780 MiB** |
| 2 | 128 | 53.7 | 2,384 | 51.45 | 2.05 | 5.42 | 1,417 MiB |
| 3 | 256 | 109.1 | 2,346 | 104.89 | 4.08 | 5.18 | 2,690 MiB |

| prediction | measured | verdict |
|---|---|---|
| target-only ≤ 5e-6, expect ~3e-8 | **3.324e-8** at 64, **8.288e-11** at 128, **4.441e-16** repeated | CONFIRMED |
| model difference 2e-5 to 2e-4, small-M regime **worse** | **3.860e-5** at 64, **4.275e-6** at 128 | **REFUTED** — 64 lands inside the band but the *reason* was wrong: it is the same order as the upward points, not an order worse |
| repeated 256 exactly 0 in the model series | **0.000e0** | CONFIRMED, third independent time |
| 64 peak allocator 650–750 MiB | **780 MiB** | REFUTED by 30 MiB (4%) |
| 128 peak allocator 1.30–1.40 GiB | **1,417 MiB** | REFUTED by 17 MiB (1.2%) |
| 128 origins/s within 8% of 256 | **+2.6%** | CONFIRMED |
| 64 origins/s **down ≥ 15%** | **+1.5%**, inside the 1.7% A-A′ spread | **REFUTED** |

### The refutation that taught something, and the one that did not

The allocator misses are 1–4% in the same direction: the model is slightly low, not wrong.

**"64 becomes launch-bound and costs ≥ 15%" is refuted in the aggregate — but the mechanism I
named is visible in the phase table and simply gets paid for elsewhere.** `accumulate` is a
fixed per-batch cost: nine `index_add_` launches whose size barely matters. It is **3.51 ms at
64 rows against 3.71 ms at 256** — flat, as predicted — which is **13% of a 26.9 ms period at
64 and 3.3% of a 110.9 ms one at 256**, and across the whole pass it costs 7.3 s at 64 against
1.9 s at 256. That penalty is real and it is exactly the one I predicted. What I did not
predict is that the forward is *more efficient per row at the small shape on a contended card*:
0.396 ms/row at 64 against 0.437/0.410 at 256, saving 5.5 s of forward and cancelling the 5.4 s
of extra accumulate almost exactly. **The prediction failed because two effects I did not know
were the same size cancelled, not because the mechanism was imaginary** — and on a quiet card,
where the forward has less contention headroom to recover, the balance may not repeat.

**The extrapolation I refused turns out to have been safe.** That is the right way round: it
was not knowable in advance, refusing it cost three minutes, and the number is now measured
rather than assumed. Had it gone the other way, seven arms would have carried an
order-of-magnitude-wrong floor in their verdict language.

### The exit 101 was the OLD guard, and the shipped tree already fixes it

Job 5582 exited 101 with every point printed and **all three guards in the current tree
passing** (3.324e-8 and 8.288e-11 against 5e-6; 4.441e-16 against the 1e-12 scatter floor;
0.000e0 model at the repeated shape). The binary predates the split: `/var/tmp/tb0_evalsweep4`
carries the unsplit `targets == 0. && model == 0.` form, which 4.441e-16 fails by construction.
A binary rebuilt from the current tree runs this sweep to completion. Recorded because the next
person to run that command with that binary will see the same 101 and should not go looking for
a defect.

### Consequences taken

`--eval-batch-size` stays at **64**: it costs nothing measurable in throughput and cuts peak
allocator **3.4×**, which is the difference between an arm running and OOMing. The
control-vs-arm comparison inherits a **measured** floor of ~1.2e-5 = **0.15% of the h=1 IC
standard error of 0.00839**, stated as a number in arm verdicts rather than as an unmeasured
exception.

---

## 11. Exact `mlq submit` lines

```
mlq submit --name eval-phases-b256 --priority 1 --max-parallel-runs 1 --max-attempts 1 \
  --time-limit 10m --cwd "$PWD" -- ./torch-env.sh /var/tmp/tb0_evalthru \
  ceiling-timexer-segment --checkpoint training/runs/timexer-control-4k/weights/best \
  --output training/runs/timexer-eval-throughput/b256 --batch-size 256 --placement anchored
```
```
mlq submit --name eval-sweep-aba --priority 1 --max-parallel-runs 1 --max-attempts 1 \
  --time-limit 10m --cwd "$PWD" -- ./torch-env.sh /var/tmp/tb0_evalsweep2 \
  ceiling-timexer-segment --checkpoint training/runs/timexer-control-4k/weights/best \
  --output training/runs/timexer-eval-throughput/sweep --placement anchored \
  --throughput-sweep 256,512,1024,256 --throughput-origins 200000
```
```
mlq submit --name eval-sweep-aba2 --priority 1 --max-parallel-runs 1 --max-attempts 1 \
  --time-limit 10m --cwd "$PWD" -- ./torch-env.sh /var/tmp/tb0_evalsweep3 \
  ceiling-timexer-segment --checkpoint training/runs/timexer-control-4k/weights/best \
  --output training/runs/timexer-eval-throughput/sweep2 --placement anchored \
  --throughput-sweep 256,512,1024,256 --throughput-origins 200000
```
```
mlq submit --name eval-sweep-aba3 --priority 1 --max-parallel-runs 1 --max-attempts 1 \
  --time-limit 10m --cwd "$PWD" -- ./torch-env.sh /var/tmp/tb0_evalsweep4 \
  ceiling-timexer-segment --checkpoint training/runs/timexer-control-4k/weights/best \
  --output training/runs/timexer-eval-throughput/sweep3 --placement anchored \
  --throughput-sweep 256,512,1024,256 --throughput-origins 200000
```

```
mlq submit --name eval-shape-64 --priority 1 --max-parallel-runs 1 --max-attempts 1 \
  --time-limit 10m --cwd "$PWD" -- ./torch-env.sh /var/tmp/tb0_evalsweep4 \
  ceiling-timexer-segment --checkpoint training/runs/timexer-control-4k/weights/best \
  --output training/runs/timexer-eval-throughput/shape64 --placement anchored \
  --throughput-sweep 256,64,128,256 --throughput-origins 200000
```

Binaries, pinned by content and named for their purpose (never a `vNN` path — two agents already
meant different binaries by `v21`):

| path | sha256 | carries |
|---|---|---|
| `/var/tmp/tb0_evalthru` | `2d492ec8da72bd982bfca263ca85b98bd3df4bf88f00b1224b9e02d6051e4b6e` | phase instrumentation, device-side retained tally |
| `/var/tmp/tb0_evalsweep2` | `69eeac69759df5f5d044dac91d76baaf6fb3bae336c9c2e5d113a9ac4e1ce677` | + paired sweep, backlog drain |
| `/var/tmp/tb0_evalsweep3` | `158413d53e4adea8fa925df9b2cb6af2408b9c0e3004eb0db300e28edc07f092` | + `Corpus::origin_timestamps`, split guard |
| `/var/tmp/tb0_evalsweep4` | `8bc3dc7454a7825fe80582cc4ae14b85d6acc3bb0fc7bc0a58b46ffc85391158` | + two-claim bar, nine-decimal decision-horizon dump |

No stamped binary carries the split guard: it landed in the tree after `tb0_evalsweep4` was
built, which is why job 5582 exits 101 on a bar the current source passes (§10). Rebuild before
re-running any sweep.

---

## 12. What a successor should measure next

**First, and it gates arm verdicts rather than lease budgets** — *done, job 5582, §10.* The
next open version of it: re-run `--throughput-sweep 256,64,128,256` on a **quiet** card with a
binary built from the current tree. §10's forward/accumulate cancellation was measured under
~17.9 GiB of foreign contention, and it is the one result in this report whose sign could
plausibly flip when the card is idle.

Second, and it is the lease-budget item: measure the three-population draw construction end to
end on a full anchored
`ceiling-timexer-segment` with `/var/tmp/tb0_evalsweep4` and compare its printed
`CausalPatch ceiling draws built in {} s (GPU idle)` line against job 5544's 91.5 s residual —
that is the only part of the locality fix still UNMEASURED, and it is the largest remaining
GPU-idle item in a diagnostic lease.
