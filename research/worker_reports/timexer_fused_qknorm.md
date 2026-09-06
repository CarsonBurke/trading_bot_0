# Fused QK-norm + packed rotary

Worktree `worktrees/fused-qknorm`, branch `fused-qknorm-20260906`, rebased onto main
`2d99f2df` (the crate has to exist before a third op can join it, and main's `6b28bad9`
CPU-dispatch convention has to be matched). One new op in `fused_kernels/`; nothing under
`trading_bots/`, `shared/`, `tui/` or `docs/` is touched.

**Result.** Bit-identical to the composed-ATen form, forward AND backward, no tolerance
(job 5182, `cargo test -p fused_kernels` 14/14 including the 3-kernel graph capture/replay).
At B=256 it removes **16.4 ms/step** on top of the already-landed fused rotary (34.1 ms/step
against the pre-rotary composition) and **234.4 MiB** of peak allocator footprint over 8
layers, of which 46.9 MiB is retained-for-the-whole-step. Forward runs at 100.0% of the
device's measured 1517 GB/s streaming roof; the per-head reduction it added to the rotary
kernel costs **0.000 ms** in the forward and **0.117 ms/layer of pure DRAM traffic** in the
backward, which the composition pays anyway.

**Correction to the brief.** The QK-norm's new retained activation was inferred at ~1,500 MiB
across 8 layers. Measured, it is **46.9 MiB** - 32x smaller. `_fused_rms_norm` retains only
the fp32 `rstd`, `[96000, 16]` = 5.86 MiB per layer; the normalized `q||k` block is retained by
nothing, because RMSNorm's backward wants `X` and `rstd` (and `X` is the projection, already
retained) while the rotary's backward is linear and wants only `cos`/`sin`. **Do not budget
CUDA-graph capture on the 1,500 MiB figure**: this op does not buy capture room, it buys
16.4 ms/step and 234 MiB.

## 1. What the op is, and what it removes

`fused_kernels::qk_norm_rope(packed_qk, cosine, sine, heads)` takes the **RAW, un-normalized**
`q‖k` block, normalizes each head block of `head_dim` columns over itself (gainless, biasless,
eps `1e-6`), rotates the packed block, and writes **only** the rotated result.

The composition it replaces is `Block::forward`'s

```rust
let normed = rms_norm(&packed[0].reshape([batch, length, 2 * self.heads, head_dim]))
    .reshape([batch, length, 2 * self.width]);
let rotated = fused_rope(&normed, rotation.0, rotation.1, self.heads).split(1, 2);
```

Three things that composition materializes and this op does not:

| tensor | shape at B=256 | size | lifetime |
| --- | --- | --- | --- |
| ATen's contiguous copy of the STRIDED `q‖k` view | `[96000, 1024]` bf16 | 187.5 MiB | transient, forward AND again in backward |
| the normalized block `normed` | `[96000, 1024]` bf16 | 187.5 MiB | transient (the rotation is its only consumer) |
| `rstd` | `[96000, 16]` fp32 | 5.86 MiB | **retained per layer**, 46.9 MiB over 8 |
| the rotary's `grad_y` in backward | `[96000, 1024]` bf16 | 187.5 MiB | transient |

The copy is not obvious and is worth naming: `_fused_rms_norm` calls
`input.expect_contiguous()`, and the packed block is a `split_with_sizes` view with row stride
`3·d_model`, so ATen copies the whole block before it normalizes — once in the forward, and
once more in `_fused_rms_norm_backward`, which calls `expect_contiguous()` again.

The fused op reads the strided view directly. Forward is one read and one write; backward is
two reads (upstream gradient, raw block) and one write.

## 2. Why the packed layout makes this exact rather than approximate

The packed column index is `t·heads·head_dim + h·head_dim + d` (`Block::forward`'s own
contract, `model.rs:510-513`), so one head block of `head_dim` contiguous columns is **simultaneously**:

- one RMS normalization group — `rms_norm` over the last axis of
  `[batch, length, 2·heads, head_dim]` is exactly per-token, per-head; and
- one rotary block — the pair `(r, r+half)` that the packed two-full-width-product form
  rotates.

So the fusion needs no reassociation and no change of addressing: the kernel keeps the rotary
kernel's thread mapping (one thread per `(row, head block, vector)`, `V = half/8` threads per
head block, each holding one 128-bit vector of the low half and one of the high half) and adds
a reduction over the block's own lanes.

**`V` is not normalized and is not in this block.** The call site's
`split_with_sizes([2·width, width], -1)` returns `q‖k` first and `V` second; this op takes the
first only. The test asserts the value block's gradient columns come back as untouched zeros,
so a kernel that wrote past the `q‖k` half would fail rather than corrupt the value path.

## 3. Bit-identity is a reduction-ORDER problem

`rstd` is fp32 and the output is bf16 with an 8-bit mantissa, so a 1-ulp fp32 error in `rstd`
flips an output element whenever `rstd·x` lands within ~1.5e-5 of a rounding boundary — about
one element in 10^5, i.e. **~1500 elements per layer** at the training shape. Matching ATen
therefore means reproducing its summation **tree**, not merely summing the same 64 numbers.

ATen's forward for bf16 with `N % 4 == 0` and `N ≤ 2^24` is
`vectorized_layer_norm_kernel<..., rms_norm=true>` (`aten/src/ATen/native/cuda/layer_norm_kernel.cu`,
torch 2.12.1 = `7269437d`), launched `dim3(warp, num_threads()/warp) = (32, 4)`:

1. `compute_stats` gives thread `thrx = x + 32y` the four-element vectors `i ≡ thrx (mod 128)`
   and accumulates each vector's squares **serially from `0.f`** (`cuWelfordOnlineSum`'s
   rms branch is `curr_sum.sigma2 + val*val`);
2. an intra-warp `WARP_SHFL_DOWN` reduction at offsets 16, 8, 4, 2, 1;
3. an inter-warp reduction over the four warps through shared memory;
4. `sigma2 / float(N)`, then `c10::cuda::compat::rsqrt(sigma2 + eps)`.

The backward is `layer_norm_grad_input_kernel_vectorized` with 128 threads, whose
`cuda_utils::BlockReduceSum` is the **same 32-lane tree over the same four-element partials**.

So for `head_dim ≤ 128` both directions reduce as: partials over four CONSECUTIVE elements,
then a 32-slot binary tree with structural zeros above `N/4`. On the kernel's thread mapping
thread `j` owns partials `2j, 2j+1` (its low vector) and `2V+2j, 2V+2j+1` (its high vector),
which means

- every ATen offset **above** `2V` adds a structural zero — a no-op;
- the offset `2V` level is **thread-local** (low partial + this thread's own high partial);
- the offsets below it are `log2(V)` XOR butterflies over the head block's own lanes, after
  which every lane holds the identical total and can normalize its own elements with no
  broadcast.

`atens_rms_statistic_is_the_reduction_tree_the_kernel_reproduces` pins step 1-4 against ATen's
own `rstd` output using tensor ops only, so a failure says whether it was the tree or the
kernel that was wrong.

### The eight rounding forms, and why one of them is measured rather than chosen

`nvcc` contracts `a*b + c` into a single `fma` by default, and three of this op's fp32
operations are of that shape:

| bit | operation | ATen source |
| --- | --- | --- |
| 0 | `sigma2 + val*val` | `cuWelfordOnlineSum`, rms branch |
| 1 | `stats_x2 += c_loss * gamma * c_h * rstd` | `layer_norm_grad_input_kernel_vectorized` |
| 2 | `f_grad_input -= x * rstd * stats_x2` | same |

Whether each contracts is a property of the compiler that built **libtorch**, not a choice, and
each flips the last bit of a 64-element reduction. All eight forms are compiled and all eight
are run against ATen; `qk_norm_rope_rounding_is_the_measured_pair` asserts the matching SET.

The measured answer is `{2, 3}`: **bit 1 set, bit 2 clear, bit 0 irrelevant.** Bit 0 is
irrelevant provably rather than accidentally - the value it accumulates is `val*val` where
`val` is a widened bf16, so the product has 16 significant bits, is exact in fp32, and
`fmaf(val, val, acc)` and `acc + val*val` round identically. That is also why all eight forms
agree in the FORWARD, whose only contraction candidate is bit 0, and why the forward needed no
measurement at all. Bits 1 and 2 live only in the backward and only one of their four
combinations is ATen's. The crate ships `QK_NORM_ROUNDING = 2`.

### Where each rounding happens

One fp32 intermediate, one rounding, at exactly the points the composition materialized a bf16
tensor:

- **forward:** `y = bf16(rstd · x)` — one rounding, because that is what
  `vectorized_layer_norm_kernel` writes to memory. The rotation then consumes the ROUNDED `y`,
  and each of its four products is rounded before the half-crossing sum
  (`bf16(bf16(y_low·c) − bf16(y_high·s))`), which is what the tiled ATen rotation does.
  Keeping `rstd·x` in fp32 through the rotation would be strictly MORE accurate and would not
  be the reference.
- **backward:** `gy = bf16(bf16(g_low·c) + bf16(g_high·s))` — the rotary's transpose, rounded
  where the composition wrote its `grad_y` tensor; then the RMSNorm gradient in fp32 with
  `dx = bf16((width·gy − (x·rstd)·stats) · ((1/width)·rstd))`, a single bf16 rounding at the
  store. `width·gy` and `(1/width)·rstd`'s `1/width` are exact at `head_dim = 64` (powers of
  two); `rsqrtf` is reproduced as `rsqrtf`, not as `1/sqrtf`, because those are different
  instructions with different results.

### Edge conventions, established empirically

- an all-zero head block gives `rstd = rsqrt(0 + 1e-6) = 1000`, not a division by zero;
- a NaN anywhere in a head block makes `rstd` NaN and therefore the whole block's output NaN,
  in both forms;
- `eps = 1e-6` is not the ATen default, but **not for the reason the recipe report and
  `model.rs:376-377` give**. They say `_fused_rms_norm` resolves `eps = None` to
  `finfo(bf16).eps = 7.8e-3`, "a 0.4% systematic shrink of every normalized activation".
  `_fused_rms_norm_cuda` actually resolves it to the ACCUMULATE type's epsilon
  (`layer_norm_kernel.cu`: `eps.value_or(std::numeric_limits<float>::epsilon())` when
  `acc_type == Float`, which it is for a bf16 input), i.e. `FLT_EPSILON = 1.19e-7`. At unit
  scale that is bit-indistinguishable from 1e-6 - the first version of this test asserted the
  two differ on `randn` input and FAILED, which is how the stale claim surfaced. The epsilon is
  still load-bearing and still worth baking in: it is observable exactly where QK-norm has to
  survive, on a head block whose RMS is near the epsilon itself (a dead head, a collapsed
  token). `the_norm_epsilon_is_observable_where_it_is_load_bearing` pins it at that scale.
  Reported to `WireKernels`, whose file the stale comment is in.

## 4. Geometry limits, refused loudly

`head_dim % 4 == 0 && head_dim ≤ 128`, checked in `bridge.cpp`. Below four elements ATen's own
forward switches to `RowwiseMomentsCUDAKernel` (a Welford reduction, a different order); above
128 its partials stop fitting the first warp of the tree above. Both would break bit-identity
silently.

Inside that range there are two paths. The 128-bit path additionally needs `V = half/8` to be a
power of two, because the level that pairs a thread's low vector with its own high vector only
exists when `2V` is one of ATen's offsets (at `head_dim = 48`, `V = 3`, there is none).
Everything else takes a single-thread emulation of the same tree —
`fused_qk_norm_rope_is_bit_identical_on_the_emulated_reduction_path` runs it at `head_dim = 12`,
where ATen still takes its VECTORIZED kernel (it decides on its own contiguous copy, which is
always aligned), so the emulation has to reproduce that tree and not a convenient one.

## 5. Measurements

All from job 5182 (`fused_kernel_probe`, B=256 rows x 375 origins = 96,000 tokens,
`d_model` 512 over 2x for the packed block, 8 heads, `head_dim` 64, 8 layers). Every quantity
including the roof is measured **interleaved** in the same passes, best-of-8 batch means,
CUDA-event timed - the methodology `timexer_fused_kernels.md` had to adopt after a sequential
roof of 659 GB/s produced a 1,109 GB/s kernel at 168% of it. Numbers reproduced across jobs
5178/5180/5182 to within 0.003 ms.

Times are **one layer's** work; `step ms cut` is the 8-layer forward+backward saving.

| baseline | cmp fwd | fus fwd | cmp bwd | fus bwd | fus bwd kernel | fwd GB/s | bwd GB/s | % roof fwd | % roof bwd | step ms cut |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| composed ATen (`_fused_rms_norm` + tiled rotation) | 2.285 | 0.259 | 3.284 | 1.047 | 0.376 | 1516 | 1567 | 100.0 | 103.3 | **34.1** |
| `_fused_rms_norm` + fused rotary (today's call site) | 1.387 | 0.259 | 1.965 | 1.047 | 0.376 | 1516 | 1567 | 100.0 | 103.3 | **16.4** |

Roof: **1517 GB/s**, a 128-bit vectorized bf16 copy at the same launch geometry.

The second row is the honest one: the packed rotary already landed, so fusing the norm into it
is worth **16.4 ms/step**, not 34.1. Both rows are in the report; only the second is summed
into the crate's step total, which is why `Comparison::additive` exists.

Reading the table:

- **The normalization is free in the forward.** The fused QK-norm+rotary forward is 0.259 ms
  and the fused rotary-only forward is 0.261 ms - the same kernel time within noise, because
  both read `2*width` and write `2*width` and the reduction rides on bytes already in flight.
  The composition charged 1.126 ms/layer for `_fused_rms_norm`'s forward alone (2.285 - 1.159).
  That 1.126 ms/layer x 8 = 9.0 ms/step is removed outright.
- **>100% of the roof is real and is the roof's fault, not the kernel's.** The backward reads
  two operands and writes one; the roof is a 1-read-1-write copy. A 2:1 read:write mix beats a
  1:1 mix on this card by ~3%, so 103.3% means "at the roof, within the roof's own read/write
  mix". Both of the previously landed backward kernels show the same 102.9%. It is not a
  measurement artifact of the interleaving - the roof is measured in the same passes - it is
  the roof being the wrong shape by 3% for a 3-pass kernel.

### The recompute cost, measured rather than assumed

The backward recomputes `rstd` from the raw `q||k` block instead of reading a stored one. Cost:

| kernel | passes over the `[96000, 1024]` block | ms | GB/s |
| --- | --- | --- | --- |
| fused rotary backward (no recompute) | 2 (read `g`, write `dx`) | 0.259 | 1516 |
| fused QK-norm+rotary backward (recompute) | 3 (read `g`, read raw `x`, write `dx`) | 0.376 | 1567 |

The delta is **0.117 ms/layer = 0.94 ms/step**. One extra pass over 196.6 MB at the achieved
1567 GB/s is 0.125 ms, so the entire delta is the bytes and the 64-element reduction plus the
`rsqrtf` plus the renormalization is **below the noise floor of the bytes it rides on** - the
achieved bandwidth went UP 3% rather than down. And against the real baseline the recompute is
free twice over: `_fused_rms_norm_backward` reads `X` too (it saves the view, not the
normalized block), so the composition pays that same read and then also reads `rstd` and the
rotary's materialized `grad_y`. The 1.965 -> 1.047 backward saving is net OF the recompute.

### Activation saving, measured

`activation_saving` builds an 8-layer chain of the QK-norm-then-rotate step with each layer's
projection and rotated output held live exactly as the QKV backward and SDPA hold them, runs
both arms from an emptied cache with `reset_peak_memory_stats`, and reads
`torch.cuda.max_memory_allocated` / `memory_allocated`:

| | composed | fused | saved |
| --- | --- | --- | --- |
| peak allocator, 8 layers | 3984.4 MiB | 3750.0 MiB | **234.4 MiB** |
| live at the end (retained), 8 layers | 3796.9 MiB | 3750.0 MiB | **46.9 MiB** |

Both numbers agree with the byte-level model to 0.0 MiB, which is what makes them a
measurement of the op rather than of the harness:

- **retained** = `rstd` only, `[96000, 16]` fp32 = 5.859 MiB x 8 = 46.9 MiB. Nothing else. The
  normalized block is retained by neither consumer, and `_fused_rms_norm` saves the strided
  VIEW of the projection (already retained) rather than the contiguous copy it makes.
- **peak** = 46.9 MiB of that, plus 187.5 MiB of transient high-water at the one layer that is
  mid-step: the composition has ATen's contiguous copy of the strided view AND the normalized
  block live simultaneously (2 x 187.5 MiB), against the fusion's single 187.5 MiB output. The
  fused arm's peak EQUALS its retained set - the fusion adds no transient high-water at all,
  because the only tensor it allocates is the one that gets retained.
- The harness's first version drew the projection as fp32 and cast it, and an fp32
  `[96000, 1536]` scratch is 562.5 MiB - three times the transient being measured, so both
  peaks landed on the scratch and reported only the 41 MiB of retained difference the first 7
  layers had accumulated. It draws bf16 directly now. This is the same class of error as the
  sequential-roof one: the instrument was bigger than the signal.

### Report series

Written to `timexer_segment_fused_kernels.report.bin`, the base the crate already registers
with one line in `shared/src/report.rs`. **No new base, so no `shared/src/report.rs` edit and
no tui edit** - the new rows and the three memory series append to the existing base:
`QK norm + rotary` / `QK norm marginal` in the per-kernel series, plus
`QK-norm composed peak allocator MiB`, `QK-norm fused peak allocator MiB` and
`QK-norm peak MiB saved` / `QK-norm retained MiB saved`.

## 6. Wiring instructions

```rust
pub fn fused_kernels::qk_norm_rope(
    input: &Tensor, cosine: &Tensor, sine: &Tensor, heads: i64,
) -> Tensor
```

`input` is the RAW, un-normalized `packed[0]` straight out of `split_with_sizes`,
`[batch, length, 2*heads*head_dim]` bf16 on CUDA, strided view included - the equality test
feeds exactly that view, so the gradient landing in the right columns of the wider buffer is
proven, not hoped for. `cosine`/`sine` are the untiled `[origins, head_dim/2]` rows, the same
tensors `rope` already takes. The output is the contiguous
`[batch, length, 2, heads, head_dim]` buffer, so the call site keeps its `.split(1, 2)`.
`NORM_EPS = 1e-6` is baked in. Off CUDA it dispatches to `reference::qk_norm_rope`,
dtype-agnostic, matching main's `6b28bad9` convention for the other two ops - ten of the
model's CPU tests reach `Block::forward` off-device, two in fp32, and
`the_off_cuda_path_is_the_per_tensor_composition_in_fp32` covers that path forward and
backward against an independent per-tensor composition.

In `Block::forward` (`model.rs`), delete the `normed` binding and swap the rotation:

```rust
-        let normed = rms_norm(&packed[0].reshape([batch, length, 2 * self.heads, head_dim]))
-            .reshape([batch, length, 2 * self.width]);
-        let rotated = fused_rope(&normed, rotation.0, rotation.1, self.heads).split(1, 2);
+        let rotated = qk_norm_rope(&packed[0], rotation.0, rotation.1, self.heads).split(1, 2);
```

with `use fused_kernels::{qk_norm_rope, relu_square, rope as fused_rope};`. `fused_rope` stays:
`kernel_classes` and three model tests still name it. Two consequences on the model's side of
the ownership line, handed to `WireKernels`:

- `kernel_classes`: the RMSNorm class that charges the fp32 `rstd` and the rotary class
  collapse into one. The fusion reads `2*width` and writes `2*width` in the forward; in the
  backward it reads `2*width` upstream gradient plus `2*width` raw block and writes `2*width`,
  with no `rstd` written, read back, or retained.
- `model.rs:376-377` (and the `NORM_EPS` doc) state the `finfo(bf16).eps = 7.8e-3` default.
  That claim is not true of this torch; see the eps bullet in section 3.

## 7. The test suite was racy, and why a side stream is not the fix

`WireKernels` ran `cargo test -p fused_kernels` on the merged tip (job 5185) and got **8
passed, 6 FAILED**: five CUDA tests died in `CachingHostAllocator` / `memcpy_and_sync` with
`cudaErrorStreamCaptureUnsupported`, and `every_kernel_captures_and_replays_inside_a_cuda_graph`
then died with `cudaErrorStreamCaptureInvalidated`. My own 14/14 runs (jobs 5180, 5182) used
`--test-threads=1` and were therefore not evidence of anything. Recorded here because the
diagnosis is a property of libtorch that any future capture test in this repository will hit.

`CUDAGraph::capture_begin` captures in `cudaStreamCaptureModeGlobal` - the only mode libtorch's
C++ API exposes, and `torch-sys` binds no mode argument at all
(`at_cuda_graph_capture_begin(graph, device_index)`). Global mode rejects any "unsafe" CUDA
action **anywhere in the process** while a capture is open, and a host-to-device copy in a
sibling test thread is exactly such an action. **Capturing on a private stream does not fix
this**, which is the tempting wrong answer: the restriction is scoped to the process, not to the
stream, so the sibling still dies and still invalidates the capture.

The fix is therefore serialization, and the thing worth designing is making it unforgettable
rather than a `--test-threads=1` incantation. `cuda()` now returns a `CudaClaim`: a
`MutexGuard<'static, ()>` plus the `Device`, `Deref`ing to the `Device`. It is the only way for
a test in this module to name a CUDA device, so a new CUDA test cannot fail to serialize - the
type system hands out the device only under the lock. The mutex is drained of poison
(`unwrap_or_else(|e| e.into_inner())`) so a failing CUDA test reports its own assertion rather
than poisoning every test after it. The CPU tests are unaffected and still run concurrently:
they touch no CUDA.

Verified as the harness actually runs it - job 5187, `cargo test --release -p fused_kernels`
with cargo's default thread count (24 on this box), **five consecutive runs, 14/14 each**.
Total CUDA test time is 0.3 s, so the serialization costs nothing worth measuring.

## 8. What is NOT claimed

- No capture-budget relief. 234 MiB is 0.9% of a 25.9 GiB usable budget and 1.4% of the
  ~17.0 GiB a forward+backward private mempool needs. The op is a 16.4 ms/step win that also
  happens to lower the floor slightly; it does not make capture fit at B=256.
- No accuracy claim. Bit-identity means the training curve cannot move, which is the point:
  this is adoptable without a re-ablation.
- `head_dim` outside `4 <= head_dim <= 128` with `head_dim % 4 == 0` is REFUSED, not
  approximated, because ATen's own reduction order changes outside it (section 4).
