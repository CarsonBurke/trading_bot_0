# STATUS AT SESSION END - read this box before anything else

**LANDED AND GREEN.** The fusion ships: `fused_kernels`' `loss_geometry` (forward + backward
CUDA kernels, autograd node, strided-target support) and `CausalPatchModel::losses` calling
it on the real training path. `./torch-env.sh cargo check -p fused_kernels --tests` 0 errors,
`./torch-env.sh cargo check -p trading_bot_0 --tests` **0 errors**, `cargo check -p
trading-bot-tui --tests` 0 errors. Bit-exactness verified BY EXECUTION on the GPU: job
**5446** `cargo test -p fused_kernels` **20 passed / 0 failed** at the production shape
(147,456,000 elements, forward AND backward), job **5450** **21/0** after the strided-target
fix. Class win MEASURED, job **5452**: **15.192 -> 4.096 ms, delta -11.10 ms**, inside the
pre-registered band.

**LANDED AND UNRUN** - the paired harness (`benchmark.rs`, `--paired-loss`), which alternates
the fused and composed-ATen chains A-B-A-B inside ONE process so contention is common to both
arms and cancels in the difference. It exists because the step-level claim is the one thing
this session could not measure: job 5452's own card was 16.3% below job 5399's measured GEMM
peak, so a cross-run step-time difference is not repairable by any scalar. Exact line:

```
mlq submit --priority 0 --max-parallel-runs 1 --time-limit 10m --max-attempts 1 --cwd "$PWD" \
  --name fk-paired -- ./torch-env.sh cargo run --release -p trading_bot_0 -- \
  benchmark-timexer-segment --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 \
  --min-history 256 --batch-size 256 --steps 40 --warmup 20 --paired-loss --paired-repeats 5 \
  --output training/runs/bench-paired/gens/0
```

Read `paired loss chain delta as a multiple of the fused arm's own spread` BEFORE the delta.

**PRE-REGISTERED, UNMEASURED** (posted over hub before the harness existed):
- paired delta **+11 to +13 ms**, fused-arm spread **under 1 ms**;
- peak allocator **~2.3 GiB composed arm / under 1.0 GiB fused arm**, from the composition's
  liveness enumeration - if it returns 10 GiB+ I mis-modelled the liveness and the report is
  wrong on that point. Both now emitted as `paired loss chain peak allocator MiB` and
  `paired loss chain peak reserved MiB`.

**MY DEPENDENCY LIST ON OTHER AGENTS' NUMBERS: EMPTY.** Every figure here is a harness
artefact of a job I ran (5399, 5422, 5427, 5446, 5450, 5452) or my own op-by-op enumeration.
No IC, no NLL, no partition, no calibration number enters any claim, so none of the session's
three corrections and neither of the two IC rulings touch anything below. The one figure of
MINE that others may hold is the composed backward: **82 kernels / 14.82 GB / 1651 GB/s**,
which superseded my own earlier 70 / 13.20 GB / 1470 GB/s. The correction moved AGAINST the
fusion's case; anyone quoting the earlier one should replace it.

**THE TWO DEFECTS EXECUTION FOUND, both of which a green `cargo check` had passed** (§4-5):
autograd materializing an unused output's gradient as a freshly ZEROED tensor, which broke the
mean coordinate's signed zero on exactly the invalid origins, fixed with
`set_materialize_grads(false)`; and a fused kernel that had never run on the real path,
because `targets` arrives as a strided `unfold` view and the composition tolerated silently
what the kernel assumed dense. The second is the lesson worth carrying: **a fused kernel's
test must drive the op with a tensor produced the way the training path produces it**, not a
freshly built dense one. 19/20 passing told us nothing about it.

**TEST STATE, exactly.** `cargo test -p fused_kernels`: green by execution on the GPU, jobs
5446 (20/0) and 5450 (21/0). `cargo test -p trading_bot_0 timexer_segment::model`: **45 passed
/ 0 failed** on my last completed run - 44 pre-existing plus the new
`the_composed_benchmark_arm_is_the_same_objective_as_the_fused_one`, which compares NLL, MSE
and every parameter gradient bitwise between the two benchmark arms so a paired timing can
never compare two different objectives. Since that run I made three further edits - the
`empty_cache()` + peak-counter reset in `paired_loss_arm`, the two peak series, and
`test_rng::exclusive()` guards on the two seeded loss tests - and all three are covered by the
0-error `cargo check -p trading_bot_0 --tests` above, but the re-run of the model suite did not
finish inside the session: the lib-test link kept restarting behind siblings' red trees
(`portfolio_data.rs`, then `model.rs:2825`, then `runner.rs:2511`). **The successor should
re-run `cargo test -p trading_bot_0 timexer_segment::model` once before queueing the paired
job**; the expected result is 45/0.

The `exclusive()` guards close an RNG-isolation hole of exactly the class `OrthoTargets`
diagnosed elsewhere: `fused_matches_reference` and the new equivalence test both call
`tch::manual_seed`, and a concurrent seed in another test rewinds the global stream, so the two
arms could be handed different batches. Under `cargo test`'s default threading that is a
flaky-by-construction comparison, and the guard removes it.

# Fusing the candle geometry and the Gaussian NLL

Target: the `fused loss geometry and NLL` kernel class, the one class in the step that was
neither GEMM-bound at 90%+ of peak nor already at 95-108% of achievable bandwidth. Measured
at 15.192 ms/step (forward 6.215, backward 8.977) and reported at 783 GB/s against a
~1.79 TB/s roof while doing 0.2 TFLOPS.

Everything below is DEMONSTRATED unless tagged otherwise. Shapes throughout: batch 256,
L = 375 origins, `pred_len` 192, 8 layers, `d_model` 512, FFN 2048, context 6000, so
`tokens = 96,000` and one `[rows, origins', 1, pred_len]` channel slice is
`S = 18,432,000` elements. One fp32 slice pass is **73.728 MB**; that is the unit ("slice
unit") every count below is in. The head is `[rows, origins', 8, pred_len]` =
**147,456,000 elements = 294.9 MB at bf16, 589.8 MB at fp32**.

## 1. The headline: the 783 GB/s was an artefact, and it changes the target

`KernelClass::forward_bytes` is documented as a FLOOR ("every input read once, every output
written once ... so `forward_bytes / measured seconds` is a floor on the class's bandwidth").
For this class the floor was 16 fp32 slices plus the head and the targets = **4866 MB**.
Enumerating what the chain actually moves gives **10,027 MB**, which is **2.06x** the floor.
So the class was never running at 783 GB/s:

| | kernels | traffic | measured ms | achieved | % of 1.79 TB/s roof |
|---|---|---|---|---|---|
| forward, composed | 65 | 10,027 MB | 6.215 | 1613 GB/s | 90% |
| backward, composed | 82 | 14,819 MB | 8.977 | 1651 GB/s | 92% |
| **whole class, composed** | **147** | **24.85 GB** | **15.192** | **1635 GB/s** | **91%** |
| forward, fused | 17 | 3,760 MB | - | - | - |
| backward, fused | 1 | 958 MB | - | - | - |
| **whole class, fused** | **18** | **4.72 GB** | - | - | - |

Admissibility check on the enumeration: 15.192 ms at the roof could move at most 27.2 GB,
and 24.85 GB is 91% of that, so the count is physically possible and close to saturation.
An earlier, coarser pass of mine reported the backward as 70 kernels / 13.20 GB / 1470 GB/s;
**that was an undercount and is superseded by the row above** — the corrected number is worse
for the "it was slow" story and better for the real one.

**Interpretation.** Nothing in this chain was slow. It ran at 91% of the streaming roof. It
was simply **337 fp32 slice-passes** where the dependency structure only requires 64. The win
is deleted passes, not faster passes, exactly as the corrected harness has been saying about
the whole step.

## 2. Pass-by-pass attribution of the original 15.192 ms

Read/write counts are in slice units; every full-size operand is read once and every output
written once. `[1, 1, 1, pred_len]` and `[rows, origins', 1, 1]` operands (the horizon
weight, `horizon_scale`, `inverse_horizon`, σ, ρ, the gain) ride in cache and are not
charged. bf16 reads are half a unit.

### Forward, 65 kernels, 136 units, 10,027 MB

| stage | kernels | units |
|---|---|---|
| `mask · horizon_weight -> weighted_mask` | 1 | 2 |
| `close = ch0(bf16) · horizon_scale` | 1 | 1.5 |
| cast ch1 to fp32 | 1 | 1.5 |
| `softplus` | 1 | 2 |
| `· range` | 1 | 2 |
| `/ ln2 -> relative_range` | 1 | 2 |
| cast ch2, `sigmoid` | 2 | 3.5 |
| `· relative_range -> position_low` | 1 | 3 |
| `log1p`, `/ sigma` | 2 | 4 |
| `close - . -> low` | 1 | 3 |
| `relative_range.log1p`, `/ sigma` | 2 | 4 |
| `low + . -> high` | 1 | 3 |
| cast ch3, `sigmoid` | 2 | 3.5 |
| `· relative_range -> position_open` | 1 | 3 |
| `log1p`, `/ sigma` | 2 | 4 |
| `low + . -> open` | 1 | 3 |
| `precision = weighted_mask · inverse_horizon` | 1 | 2 |
| x4 channels: `ch(4+c)·gain`, `tanh`, `·(-2·cap)`, `exp` | 16 | 30 |
| x4: `weight = exp · precision` | 4 | 12 |
| x4: `target - prediction`, `square` | 8 | 20 |
| x4: three `dot`s (NLL term, cap term, diagnostic MSE) | 12 | 24 |
| `½·ln h` prior and the two denominators reduce the mask | 3 | 3 |

### Backward, 82 kernels, 201 units, 14,819 MB

Rules, stated because they are what makes the count reproducible: a pointwise op's backward
reads each full-size operand it needs once and writes once; `add`'s backward is the identity
and costs nothing; an autograd `InputBuffer` accumulation of k contributions costs k-1 adds;
`square`'s backward is `grad·2·self`; `log1p`'s is `grad/(self+1)`, two kernels; a dtype cast
is one kernel; `split`'s backward is one `cat` over the eight slices.

| stage | kernels | units |
|---|---|---|
| x4: `dot(scale, wmask)` backward | 4 | 8 |
| x4: `dot(square, weight)` backward, both operands | 8 | 16 |
| x4: `square` backward | 8 | 20 |
| x4: `(target - prediction)` backward, `neg` | 4 | 8 |
| x4: `weight = e·precision` backward | 4 | 12 |
| x4: `exp` backward | 4 | 12 |
| x4: `·(-2·cap)` backward | 4 | 8 |
| x4: accumulate the two `grad_scale` addends | 4 | 12 |
| x4: `tanh_backward` | 4 | 12 |
| x4: `·gain` backward and the fp32->bf16 cast | 8 | 14 |
| `open = low + d3` / `high = low + dF` backward | 0 | 0 |
| `d3 = l3/sigma`, `dF = full/sigma`, `d2 = l2/sigma` backward | 3 | 6 |
| three `log1p` backwards | 6 | 15 |
| `p3 = s3·rr` and `p2 = s2·rr` backward, both operands | 4 | 12 |
| `low = close - d2` backward, `neg` | 1 | 2 |
| `grad_low`: three consumers, two adds | 2 | 6 |
| `grad_relative_range`: three consumers, two adds | 2 | 6 |
| `rr = a/ln2`, `a = sp·range` backward | 2 | 4 |
| `softplus_backward` and its cast | 2 | 4.5 |
| two `sigmoid_backward`s and their casts | 4 | 9 |
| `grad_close`: two consumers, one add | 1 | 3 |
| `close = ch0·hs` backward and its cast | 2 | 3.5 |
| `split` backward: `cat` of eight bf16 slices | 1 | 8 |

**Where the 15.192 ms went, in one line:** 48 elementwise ops over the 147 M-element head,
each one a separate full-size fp32 kernel that materialized an intermediate nobody wanted,
plus their backward, plus twelve reductions that are only 24 of the 337 units.

## 3. What the fusion does, and what it deliberately does NOT

`fused_kernels::loss_geometry` is one forward kernel and one backward kernel.

- **Forward**, one pass: reads the whole bf16 head (4 units), the fp32 targets (4) and the
  folded mask (1); writes the mean coordinate and the twelve fp32 vectors the twelve `dot`s
  consume (13). 22 units, one kernel. Every one of the 48 intermediates stays in registers.
- **The twelve `dot`s are NOT fused.** A reduction's summation tree belongs to cuBLAS, not to
  us, and reassociating it is where bit-exactness dies. ATen still performs all twelve on the
  vectors the kernel writes, so the loss value, the MSE and the twelve gradients keep their
  bits exactly. 24 units, twelve kernels.
- **Backward**, one pass: recomputes the entire chain from `head` — 9 units read, 4 written
  as the dense bf16 head gradient — and retains no full-size fp32 tensor whatsoever. 13
  units, ONE kernel, a fifteenth of the composed backward's 201.
- **`close` is now a real differentiable output.** The amplitude prior reduces the mean
  coordinate, and emitting it from inside the kernel costs one of the 13 written units
  instead of a separate ~1.5 ms/step pass over the fp32 channel slice.

Cost accounting, all zero: no parameters, no FLOPs beyond the same 48 ops, and peak memory
falls — the forward retains eight tensors that were already resident plus the twelve
`[S]` fp32 vectors (884 MB) instead of ~50 full-size fp32 intermediates.

## 4. The stride defect: the fused op had never run on the real path

Job **5448** — the harness measurement, queued before verification precisely because a
perf number does not need a correct kernel — died on the capture-stream warmup with
`fused loss_geometry failed: loss_geometry targets must be contiguous`. That is the path
every training run takes, so it would have poisoned any arm built on that binary.

**Ground truth first**, obtained on CPU because tensor layout is device-independent, by
replicating `statistics()` and `targets()` op for op at the real geometry:

| operand | sizes | strides | dense |
|---|---|---|---|
| `targets` | `[3, 375, 4, 192]` | `(288000, 768, 1, 4)` | **NO** |
| `mask`, `weighted_mask` | `[3, 375, 1, 192]` | `(72000, 192, 192, 1)` | yes |
| `sigma`, `range`, `stats.mask` | `[3, 375]` | `(375, 1)` | yes |
| `per_bar(sigma)`, `per_bar(range)` | `[3, 375, 1, 1]` | `(375, 1, 1, 1)` | yes |

`targets` is **channel-innermost**: the four candle channels for one (origin, bar) sit in 16
consecutive bytes. Cause: `future_windows` builds its windows with `unfold`, and
TensorIterator allocates the arithmetic below a permuted input in the INPUT's layout rather
than in contiguous order. The ATen composition accepted this silently because every
pointwise op it called is strided. Note that `targets` is the ONLY non-dense operand — every
one the kernel touches was checked, not just the one that happened to abort first, so there
is no second crash queued behind the first.

**Fix: the kernel reads the strides.** Three int64 arguments
(`target_token_stride`, `target_channel_stride`, `target_bar_stride`), one integer multiply
per channel access, zero traffic and zero memory. The alternative, `.contiguous()` in the
bridge, would copy 4 fp32 slices — 295 MB read plus 295 MB written = **0.59 GB/step**, about
0.35 ms at the roof — plus 295 MB of extra peak memory and a second tensor retained across
the step, to gain nothing. The strided form is also the FASTER layout: with channel stride 1
a thread's four target reads are one 16-byte transaction instead of four reads 768 bytes
apart. The bridge still requires rows and origins to fold into one token axis
(`stride(0) == size(1)·stride(1)`) and still requires density of every operand the kernel
indexes with plain arithmetic; only `targets` is layout-general.

**The prediction is UNCHANGED by this fix and deliberately so.** The enumeration already
charged the targets exactly 4 units read forward and 4 backward, which is what a strided read
still costs; no bytes moved and no kernel was added. Widening the band after seeing a defect
would be bracket-fitting.

### The lesson, which is bigger than the missing stride

1. Nothing in `fused_kernels` is exercised by `cargo check -p trading_bot_0 --tests`.
2. The fused op had **never executed on the real training path** until job 5448 ran it. Every
   test that had passed — including bit-exactness at 147,456,000 elements — built its targets
   with `Tensor::randn`, which is contiguous by construction. **Shape-correct and
   stride-wrong.** 20/20 green told us nothing about the layout the model actually produces.
3. A perf job caught a correctness defect, and holding the binary back from training arms
   until verification passed is what prevented a contaminated trajectory.

So the new test reproduces the CONSTRUCTION, not the dimensions:
`the_real_paths_strided_targets_are_read_as_the_composition_reads_them` builds the targets by
`narrow` + `unfold` + arithmetic exactly as `future_windows`/`targets` do, at 32x375x192,
asserts the stride tuple and `!is_contiguous()` **before** comparing anything so it can never
silently decay into another dense test, requires zero differing elements against the
composition in forward and backward, and then requires the fused result on a dense COPY of the
same values with every other operand shared to be bit-identical — which is what separates
interpreting the strides from ignoring them.

## 5. Bit-exactness: four measured facts, one intrinsics rule, one bridge bug

The acceptance bar was bit-identity against the composed ATen form, forward AND backward, no
tolerance. Four things had to be MEASURED, because none is derivable from the mathematics and
each moves the last bit over 147 M elements. `loss_geometry_rounding_is_the_measured_form`
sweeps all 32 combinations against the composition on the device; the answer is carried in
`fused_kernels::LOSS_GEOMETRY_ROUNDING = 2`.

| bit | question | MEASURED answer |
|---|---|---|
| 0 | `x / ln2` with `ln2` a HOST scalar | **CLEAR: a multiply by the fp32 reciprocal** on CUDA — and a true DIVISION on CPU |
| 1 | `tanh_backward`'s `1 - y·y` | **SET: contracted into one `fma`** |
| 2 | `sigmoid_backward`'s association | **CLEAR: `(g·(1-y))·y`**, not `(g·y)·(1-y)` |
| 3-4 | `softplus_backward`'s association | **ZERO: `(g·z)/(z+1)`**, not `g·(z/(z+1))` nor `(g/(z+1))·z` |

The per-channel sweep is the evidence, at 64x375x192 (4,608,000 elements per channel):
forms 2 and 26 give `[3244, 0, 0, 0, 0, 0, 0, 0]` differing per head channel; all thirty
other forms are nonzero in at least one of channels 1-7.

**Bit 0 is the important one for anybody who writes the next kernel here.** `/ln2` is a
reciprocal multiply on CUDA and a true divide on CPU, so a CPU-side bit-exactness probe
agrees with itself and disagrees with the GPU. I ran exactly that probe (a 4.6 M-element CPU
reproduction of the whole backward, 0 differing elements) and it certified a kernel that was
wrong on the device. Off-device probes cannot settle this class of question.

**The intrinsics rule.** Every arithmetic step in the kernel uses `__fmul_rn`, `__fadd_rn`,
`__fsub_rn`, `__fdiv_rn` rather than `*`, `+`, `-`, `/`. nvcc contracts a product and a sum
into one `fma` by default; the composition it replaces had a separate kernel per operation
and therefore no opportunity to contract. Writing the intrinsics is how the fusion stays
bit-identical rather than merely more accurate.

**The bridge bug, which is the finding most likely to bite someone else.** Channel 0 — the
mean coordinate — disagreed with the composition on 3244 of 4,608,000 elements, every one of
them an invalid origin, every one of them `-0` in the composition against `+0` in the kernel,
and invariant under all 32 rounding forms. The mechanism:

1. Where an origin is invalid the folded mask is `+0`, so `precision = mask·h⁻¹` is `+0` and
   `weight = exp(-2·cap·s)·precision` is `+0`.
2. Every gradient below that is a signed zero whose sign is a pure XOR of the signs of
   `grad_dot`, of the residual `target - prediction`, and of the `neg` the residual's backward
   applies. That is a product chain, IEEE fixes it, and both forms agree on it.
3. The mean coordinate is the ONE channel whose gradient is a SUM of two such zeros — the
   channel-3 residual and the `low` path. `-0 + -0` is `-0`; every other pairing is `+0`. So
   it is the only channel where a spurious zero ADDEND is observable at all. Channels 1-3
   re-derive the sign through a `sigmoid_backward` or `softplus_backward` product; channels
   4-7 never touch `close`.
4. The spurious addend was autograd's: **it MATERIALIZES an unused output's gradient as a
   freshly zeroed tensor**, so `grad_close` arrived defined and full of `+0` with nothing
   consuming the mean coordinate, and the kernel added it. `-0 + +0` is `+0`.

`ctx->set_materialize_grads(false)` is the fix, and the lesson generalizes past zeros: a
materialized gradient is an EXTRA operand, and a fused op with optional outputs has to refuse
it rather than assume adding zero is free. With it, the strict bar stands with **no
exemption** — the comparison counts a sign flip on an exact zero as a difference and the
kernel meets it.

### Tests added

- `loss_geometry_rounding_is_the_measured_form` — the 32-form sweep at 4x1000x64, with
  per-form counts, asserting that the shipped constant is among the forms that reproduce the
  composition exactly.
- `fused_loss_geometry_is_bit_identical_at_the_production_shape` — 256x375x192,
  **147,456,000 elements**, forward and backward. The shape is the point: a chain of `log1p`s
  and `exp`s that agrees on 128 elements and disagrees on one in 10^6 passes every toy test
  and still moves a training curve.
- `fused_loss_geometry_matches_the_composition_at_nonfinite_and_masked_elements` — NaN, ±inf
  forced into the coordinates, the log scales and the targets, at valid AND invalid elements;
  then a clean batch with the mask zeroed everywhere, requiring literal zeros out of every
  reduction and out of the head gradient. A nonfinite element behind a zero mask is NOT zero
  and must not be: `NaN·0` is `NaN` in the composition too.
- `the_masked_signed_zero_is_the_compositions_and_reaches_a_parameter_gradient_intact` — puts
  the real head-output GEMM under the loss and requires the WEIGHT and ACTIVATION gradients,
  which is what the optimizer consumes, to be bit-identical sign bit included; refuses to
  pass vacuously by asserting the composition actually produces negative zeros.
- `fused_loss_geometry_carries_the_mean_coordinate_gradient` — the amplitude prior's path,
  where an external consumer is a third addend on channel 0; numeric, not bit-exact,
  deliberately and only here, because three fp32 addends do not reassociate and there is no
  incumbent bit pattern for a term that did not exist before the prior.
- `every_kernel_captures_and_replays_inside_a_cuda_graph` — extended to the loss chain, which
  allocates twelve reduction vectors and runs twelve `dot`s inside the capture window.


### Verification, by execution on the GPU

| job | target | result |
|---|---|---|
| **5446** | `cargo test -p fused_kernels` | **succeeded, 20 passed / 0 failed** — production-shape bit-exactness, the masked signed zero through a real GEMM's parameter gradient, the nonfinite/masked cases, the 32-form rounding sweep and the CUDA-graph capture |
| **5441** | same, before the bridge fix | failed 1/20 — `channel 0: 3244 mismatched, 3244 both zero, 3244 behind a zero mask, 3244 composed-negative, 0 fused-negative`, which is the materialized-gradient bug |
| **5450** | same, after the stride fix | **succeeded, 21 passed / 0 failed** — including `the_real_paths_strided_targets_are_read_as_the_composition_reads_them` |

All three ran under the batch's constraints: priority 0, one attempt, 10-minute limit, the
`fused_kernels` target only. 5450's suite finishes in 0.96 s.

On the optimizer path: once the parameter gradient is bit-identical nothing downstream can
differ. Checked anyway, because a signed zero survives `-0 + -0` and division by one: NorMuon
SQUARES the gradient before its `rsqrt`, so a signed zero enters as `+0`, and the polar
iteration is matmul-only. Nothing in the step takes a reciprocal or an `rsqrt` OF a gradient
element.

## 6. Accounting

- `kernel_classes`: the `fused loss geometry and NLL` entry's `forward_bytes` is now
  **ENUMERATED, not a floor** — 51 units — and the comment states each term, because after
  the fusion there is nothing left to guess.
- `step_cost`: `head_loss` goes from **278 to 74** slice units (51 forward + 13 backward + 10
  for building the targets and the mask, which carry no gradient). At 73.728 MB per unit that
  is **-15.04 GB/step**, and the reported analytic step traffic goes **143.204 -> 128.16 GB**.
- The old 278 units was 20.50 GB against the 24.85 GB actually moved, so the honest
  PRE-fusion step basis was **147.6 GB/step**, not 143.204. `step_cost`'s uniform
  "three passes both sides" rule undercounts chains of many small passes (autograd's
  accumulations and casts are not in the model) and overcounts where autograd elides work
  (`add` backward is free, a `reshape` backward is a view). Treat the step-wide figure as a
  shape-derived bound with roughly ±20% per-class error, which is what `FlopInventory` said.
- Which other class is understated: only this one is PROVEN. The GEMM classes' byte counts are
  exactly enumerable from shapes. The elementwise classes measure 1455-1787 GB/s against a
  1.79 TB/s roof, so their declared bytes cannot be far low or they would exceed the roof.
  What is systematic is the rule, not a per-class error I can name.
- `every_kernel_class_runs_at_the_shape_it_declares` stays green: the class still declares the
  `[rows, origins', 8, pred_len]` bf16 head as its input and still returns a scalar.

## 7. Prediction and measurement

**PRE-REGISTERED** (posted over `hub` before any measurement existed): class 15.192 ms ->
**2.9 ms**, band 2.6-4.5; class delta **-12.3 ms**, band -10.7 to -12.6; step 167.3 ->
**155.1 ms**, band 152.8-156.6, **1.079x**, band 1.068-1.095. Derivation: forward 3.760 GB at
the 1613 GB/s the composed forward achieved = 2.33 ms; backward 0.958 GB at 1651 GB/s =
0.58 ms; 18 launches at ~5 µs = 0.09 ms. The floor of the band assumes the fused kernel gets
only 1200 GB/s because it writes thirteen streams at once; the ceiling assumes the roof. The
backward is comfortably memory-bound: ~14 transcendentals per element over 18.4 M elements is
0.05 ms of SFU/ALU work against 0.58 ms of traffic.

This is far above the -6.5 ms (range -4 to -9) prior estimate, and the reason IS the
attribution finding: the prior estimate was computed against the declared floor, which was
2.06x low.

The measurement runs under the queue owner's lease (this worker may submit only
`cargo test -p fused_kernels` jobs). Command lines handed over:

```bash
./torch-env.sh cargo build --release -p trading_bot_0
cp "$(./torch-env.sh cargo metadata --format-version 1 --no-deps \
      | jq -r .target_directory)/release/trading_bot_0" /var/tmp/tb0_v13

mlq submit --priority 1 --max-parallel-runs 1 --time-limit 10m --max-attempts 1 --cwd "$PWD" \
  --name timexer-harness-postfusion -- \
  ./torch-env.sh /var/tmp/tb0_v13 benchmark-timexer-segment \
  --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 \
  --min-history 256 --batch-size 256 --profile --capture-audit \
  --output training/runs/bench-postfusion/gens/0
```

The BEFORE rows are job 5399's; the arguments above are byte-identical to it so the tables
are comparable, and every class except the loss one must be unchanged within noise in the
same table, which is a free internal control on the comparison.

### MEASURED, job 5452 on `/var/tmp/tb0_v14`

| quantity | before (5399) | after (5452) | predicted | verdict |
|---|---|---|---|---|
| class forward ms | 6.215 | **3.265** | 2.33 | inside band |
| class backward ms | 8.977 | **0.831** | 0.58 | inside band |
| class total ms | 15.192 | **4.096** | 2.9 (band 2.6-4.5) | **CONFIRMED, inside band** |
| class delta ms | n/a | **-11.10** | -12.3 (band -10.7 to -12.6) | **CONFIRMED, inside band** |
| step activation traffic GB | 143.426 | **128.385** | 128.16 | **CONFIRMED to 0.2%** |
| captured step ms | 168.52 | 199.98 | 155.1 (band 152.8-156.6) | **UNTESTED** — see below |
| measured device bf16 GEMM TFLOPS | 230.99 | **193.32** | unchanged | **INTERNAL CONTROL FAILED** |

**The step-level claim is UNPROVEN, not refuted.** The internal control I pre-registered —
every class except the loss one unchanged within noise — failed, and it failed against the
environment rather than the kernel: the harness's own measured bf16 GEMM peak fell 16.3%
during 5452 (foreign PPO tenants held the card), and every arithmetic-bound class fell with
it in proportion (QKV 208.3 -> 172.7 TFLOPS, SDPA 133.7 -> 112.5, FFN-down backward 1.919 ->
2.426 ms) while the two bandwidth-saturated classes did not move at all (RMSNorm ~1536 GB/s,
residual addcmul ~1589 GB/s). A run whose own measured peak differs from the baseline's by
16% cannot support a cross-run step-time comparison in either direction.

**What IS confirmed, and why the contention does not touch it.** Traffic is
contention-invariant: 128.385 GB measured against 128.16 predicted, 0.2% error, and it is
measured in the SAME run as the step time, so it needs no normalization at all. The class
time is confirmed too, and conservatively: both fused kernels ran at **1152 GB/s** forward
(3.760 GB / 3.265 ms) and **1153 GB/s** backward (0.958 GB / 0.831 ms), which is essentially
the 1200 GB/s FLOOR assumption my band was built on, on a card that was simultaneously 16%
down on its own peak. On a quiet card the class should land nearer the middle of the band, so
the measured -11.10 ms is a lower bound on the class win rather than an estimate of it.

**No normalization is defensible.** Scaling 199.98 ms by 230.99/193.32 gives 167.4 ms and
would imply the fusion bought nothing, but that scaling assumes contention taxes GEMM,
bandwidth and launch-bound work by the same factor, and the same table refutes that
assumption directly: the two bandwidth-saturated classes were untaxed. A single scalar cannot
normalize a step that is part arithmetic-bound, part bandwidth-bound and part launch-bound.
The honest answer is to re-measure on a quiet card.

**Two instrument changes this episode earns**, and they matter more than the number:

1. **Gate every cross-run step-time claim on the machine's own measured peak.** The harness
   already records `measured device bf16 GEMM TFLOPS`; nothing checked it. A benchmark that
   records the peak and does not compare it across runs cannot support a step-time claim.
   Require the re-run within a few percent of 5399's 230.99 TFLOPS before comparing anything.
2. **Prefer a PAIRED design to a gate.** The robust measurement is both binaries in ONE
   lease, interleaved A-B-A, differencing step times inside the job. Contention then appears
   in both arms and cancels in the difference, and the A-A gap measures the residual drift
   that bounds the comparison's own error. This costs one job instead of two and does not
   depend on the card being quiet — which, on a shared card, is not a condition anyone can
   schedule.

The prediction stated above is NOT moved or widened. It was posted before any measurement
existed; its class-level and traffic-level parts landed inside it; its step-level part is
untested and stays that way until a run passes the peak gate or a paired run replaces it.

## 8. Should this have been done at all

Yes, and the honest version of the recommendation is: **this is the last one.** The MEASURED
class win is 15.192 -> 4.096 ms, a bounded-below **-11.10 ms**, and it comes from deleting
319 of 337 slice-passes in the only chain that still had passes to delete. The measured step
traffic is 128.385 GB against a
roof that would move 27.2 GB in 15 ms — every remaining class is a GEMM at 90-96.7% of
arithmetic peak or an elementwise kernel at 92-100% of bandwidth peak. There is no third
fusion in this model worth the risk; the next speed win has to come from removing work, not
from removing passes.

## 9. The paired instrument, and how to run it

Built because the cross-run comparison is not repairable: 5452's own measured bf16 GEMM peak
was 16.3% below 5399's, and no scalar normalizes a step that is part arithmetic-bound, part
bandwidth-bound and part launch-bound. `benchmark-timexer-segment --paired-loss` runs the
fused chain and the composed-ATen chain **alternated inside one process**, A-B-A-B, and
differences them there, so whatever the machine is doing is common to both arms and cancels.

- **No production toggle.** `CausalPatchModel::losses` carries no flag, no branch on a kernel
  choice and no fallback. The composed arm is a separate `composed_losses`, reachable only
  from the benchmark and from tests. The two functions differ in exactly ONE line — which
  geometry provider they call — because the operand preparation (`loss_operands`) and all
  twelve reductions (`reduce_geometry`) are now shared code. That is what makes the paired
  difference a measurement of the fusion rather than of two separately-written objectives,
  and `the_composed_benchmark_arm_is_the_same_objective_as_the_fused_one` pins it: off CUDA
  both providers *are* the reference chain, so any inequality in the NLL, the MSE or any
  parameter gradient is prep-or-tail drift and fails the test.
- **Both arms EAGER, labelled as such.** The composed chain cannot be captured on the real
  path without a production switch, so pairing captured-against-eager would measure capture
  rather than fusion. Capture is worth ~0.4% on this step (audited: eager 167.3 vs replay
  168.0 ms), so eager-vs-eager is a faithful instrument for the DIFFERENCE — but the
  difference is what it measures, and the report says eager.
- **In the window:** trunk forward, head, loss, full backward. **Out:** the optimizer
  (identical in both arms and ~3x the difference being measured), the loader, and the
  harness's batch assembly — batch, statistics and targets are built once and shared.
- **Refuses to report a meaningless delta:** the arms' NLLs must be bit-identical
  (`to_bits()`), which they are by `fused_kernels`' suite. An inequality means the benchmark
  is comparing two objectives, and it errors instead of printing a number.
- **Series, not log lines**, on the existing `timexer_segment_benchmark` base (no new report
  base, so no registry contention): both arms' means, the delta, each arm's spread across
  repeats, **the delta as a multiple of the fused arm's own spread** — the honest
  signal-to-noise of the comparison — the alternation count, and every repeat individually so
  a reader can tell a trend from one disturbed alternation.
- **The gate is kept too**, as a validity check on any UNPAIRED run: a new row reports
  `measured bf16 GEMM TFLOPS as a fraction of the quiet-card reference (job 5399)`, against
  the measured 230.99 constant. A run far from 1.0 cannot support a cross-run step claim.

```bash
./torch-env.sh cargo build --release -p trading_bot_0
cp "$(./torch-env.sh cargo metadata --format-version 1 --no-deps \
      | jq -r .target_directory)/release/trading_bot_0" /var/tmp/tb0_v15

mlq submit --priority 1 --max-parallel-runs 1 --time-limit 10m --max-attempts 1 --cwd "$PWD" \
  --name timexer-paired-loss -- \
  ./torch-env.sh /var/tmp/tb0_v15 benchmark-timexer-segment \
  --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 \
  --min-history 256 --batch-size 256 --paired-loss --paired-repeats 5 \
  --output training/runs/bench-paired/gens/0
```

One job, ten alternations of 20 rounds each, no `--profile` so the per-class sweep does not
eat the lease. It does not need a quiet card — which was the flaw in gating on one.
