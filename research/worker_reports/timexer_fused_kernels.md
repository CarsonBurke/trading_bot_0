# Fused CausalPatch kernels: ReLU² and the packed rotary

Worktree `worktrees/fused-kernels`, branch `fused-kernels-20260906`, rebased onto main
`20dd3402`. One commit. New crate `fused_kernels/`; `model.rs` untouched.

**Result.** Both kernels are **bit-identical** to the composed-ATen forms they replace, in
forward AND backward, with no tolerance. At B=256 they remove **52.9 ms/step** (job 5140)
and both then run at **97-100% of the device's measured streaming roof**, i.e. there is no
bandwidth headroom left in either. Both are CUDA-graph-capturable, proven by capturing
forward+backward through both and replaying against a fresh eager evaluation.

## 1. Headline measurement (job 5140, B=256, 8 layers)

`cmp` = composed ATen, `fus` = fused. Forward is grad-off. Backward `cmp`/`fus` are through
the same autograd harness (identical on both sides, so the difference is exact); `fus bwd k`
is the backward kernel alone via `fused_kernels::raw`, and the backward GB/s comes from it.

| kernel | cmp fwd | fus fwd | cmp bwd | fus bwd | fus bwd k | fwd GB/s | bwd GB/s | % roof fwd | % roof bwd | step ms cut |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ReLU² | 1.444 | 0.698 | 4.382 | 1.774 | 1.059 | 1126 | 1114 | 98.5 | 97.4 | **26.8** |
| packed rotary | 1.680 | 0.347 | 3.278 | 1.351 | 0.344 | 1133 | 1143 | 99.1 | 100.0 | **26.1** |

Streaming roof **1143 GB/s**, measured in the same interleaved window (see §5). Total
**52.9 ms/step**. Speedups: ReLU² 2.07× forward, 2.47× backward; rotary **4.84×** forward,
2.43× backward.

Two earlier jobs measured the same code on a contended device and are kept for the caveat
they carry, not for their absolute values: **5139** roof 602 GB/s, ReLU² fwd 1.156 ms,
rotary fwd 0.571 ms, 96.5 ms/step cut; **5137** rotary forward at 1109 GB/s against a
sequentially-measured 659 GB/s roof - the defect §5 fixes. Absolute milliseconds inflate
under contention on BOTH sides, so the ms-saved figure inflates with them; the invariant
across all four runs is the ratio and the fraction-of-roof, both of which held.

Report base **`timexer_segment_fused_kernels`**, written to
`training/runs/fused-kernels/gens/0/` and read back through `report_cli`. Registered by
appending one line to `TIMEXER_SEGMENT_REPORT_BASES` in `shared/src/report.rs`; `tui`'s
`meta_chart_bases` pulls that slice with `extend_from_slice` (`tui/src/main.rs:426`), so a
new base is registered in both places by that single line and **no tui edit exists or is
needed**.

## 2. Autograd mechanism, and why it is sound

Each fusion is a **`torch::autograd::Function` subclass in C++** (`csrc/bridge.cpp`), and
the C ABI calls its `apply`.

- Not a dispatcher op plus `derivatives.yaml`: that needs codegen this repo does not run.
- Not a composition of differentiable ATen calls: that composition is the cost being
  removed.
- `Function::apply` runs `forward` with grad mode off and records the node itself, so the
  output's version counter, `requires_grad` and graph edges are libtorch's. The tensor
  handed back to Rust carries a real `grad_fn`, and `Tensor::backward` /
  `Tensor::run_backward` from Rust drive the fused backward kernels with no further
  plumbing. `aten::_addmm_activation` was not an option: KernelTraffic measured that it has
  no derivative.
- First order only. The backward launches raw kernels, so it creates no graph; a
  second-order call would silently see zero. `reject_double_backward` rejects it loudly
  instead. The backbone never takes a second derivative.

`fused_kernels::raw::{relu_square_backward, rope_backward}` expose the same launches with no
node around them - not differentiable, not for the model path, present so the microbenchmark
can time a kernel instead of a kernel plus a harness.

## 3. Numerics: bit-identical, and the two places that is not obvious

Rounding discipline: all arithmetic in fp32, and every value the composition would have
MATERIALIZED as a bf16 tensor is rounded to bf16 at exactly that point. Keeping intermediates
in fp32 would be strictly more accurate and would NOT reproduce the landed reference;
reproducing the reference is what lets these be swapped in without moving a training curve.

**ReLU².** Forward `r = relu(x); y = r·r`, one rounding - and it matches because `r·r` on an
8-bit mantissa is exact in fp32. Backward is `grad · 2 · self` in the composition, whose
first multiply is EXACT (multiplying a bf16 by 2 only decrements the exponent), so the
composition also carries exactly one inexact rounding and the single-fp32-rounding kernel is
simultaneously the most accurate form and the bit-identical one. Two conventions had to be
measured rather than assumed:

- `relu` lowers to `clamp_min`, which **propagates NaN**. `v > 0 ? v : 0` would silently turn
  a NaN activation into zero.
- The gradient mask is `threshold_backward`'s `self <= threshold ? 0 : grad`, i.e.
  `NOT (relu(x) <= 0)`, which is **TRUE at NaN** - so a NaN activation carries a NaN
  gradient, not a zero one. The obvious `x > 0 ? 2·g·x : 0` disagrees with the reference on
  exactly the inputs a numerical audit looks at. This was a real bug in the first version;
  `composed_relu_square_gradient_mask_is_not_x_greater_than_zero` is the test that caught it.

**Packed rotary.** The composition rounds each of the four products to bf16 before the
half-crossing sum, so the kernel does too: `bf16(bf16(x_low·c) - bf16(x_high·s))` and
`bf16(bf16(x_high·c) + bf16(x_low·s))`. Backward is the transpose, and it is exact for the
same two reasons: negation of a bf16 is exact, so
`round((-g_low)·s) = -round(g_low·s)`, and the accumulation of the cosine and sine
contributions is a single bf16 `add` either way round.

**Save x, not relu(x)** (ResidualRecipe independently reached the same ranking; two of their
figures needed correcting and they have recorded the corrections):

- Memory is **neutral**, not +375 MiB. The composition retains relu's OUTPUT (`square`'s
  saved input) and frees `x`, because `addmm`'s backward wants its input and weight and never
  its output. Today = one live `[96000, 2048]` bf16; save-x = one live. `y` is retained by
  the down-GEMM in both.
- Saving `relu(x)` instead would force the forward to write a second full-width tensor,
  **+393 MB per layer** (3·hidden forward instead of 2), to buy an arithmetic simplification
  on a kernel that is 100% memory-bound. Peak-neutral, purely wasted traffic.
- Not taken: `grad_x = 2·sqrt(y)·grad` recovers `relu(x)` from the output the down-GEMM
  already holds, so nothing is saved at all and peak drops 375.0 MiB. It costs bit-identity -
  `sqrt` of a bf16-rounded square recovers `r` with ~2⁻⁹ relative error, systematically
  biased in the rounding direction of `y`. Bit-identity is worth more than 375 MiB that may
  not bind; if it ever binds, this is a ~15-line kernel variant.
- Also not taken, and the bigger prize: **relu² as an epilogue on the up-GEMM**, so `x` is
  never materialized at all - 1·hidden forward, **-18.87 GB/step and 3.15 GB/step below the
  GELU baseline**. It needs a cuBLASLt/CUTLASS GEMM with a custom epilogue AND a backward
  that recovers `r` (so it inherits the `sqrt` accuracy question), which is a different piece
  of work from these two kernels.

## 4. Graph capture

`both_kernels_capture_and_replay_inside_a_cuda_graph` captures a forward AND a backward
through both kernels via `torch_sys::at_cuda_graph_*`, overwrites the input buffers in place,
replays, and requires the replayed gradients and objective to equal a fresh eager evaluation
of the **composition** on the new bytes. A host sync inside a kernel, an allocation outside
the capture's private pool, or a launch geometry depending on anything but the arguments
would fail the capture or leave a stale replay.

What makes them capturable: no host synchronization, no device-to-host read, no dynamic
shape, and grid extents that are pure functions of the launch arguments (a grid-stride loop
with a fixed 32768-block cap, so the result does not depend on the cap). The only allocation
is the output, taken from the caching allocator exactly as any ATen op's is - which is what a
capture's private mempool exists for. The event timer in `bridge.cpp` DOES synchronize and is
reachable only from the probe, never from the model path.

## 5. Measurement methodology, and a correction for the harness

Three choices, each forced by a measurement that came out wrong first.

1. **CUDA events, not a host clock.** `torch-sys` binds no `cudaEvent_t`, and the existing
   profiler works around that through pyo3. This crate compiles CUDA already, so
   `bridge.cpp` binds events directly against `at::cuda::getCurrentCUDAStream()` - no Python,
   and the interval brackets exactly the kernels' stream.
2. **Best of N, not the mean.** A peak is a maximum.
3. **Interleaved, not sequential** - this is the correction that matters for
   `benchmark.rs`'s own numbers. Best-of-N is not sufficient on its own: measuring the roof
   and then the kernels afterwards produced a 659 GB/s roof and an 1109 GB/s kernel **in one
   job** (5137), because a foreign tenant occupied the device long enough to swallow every
   batch of one measurement and none of another. `device_peaks`'s best-of-8 fix removed the
   mean-vs-max half of the defect but not this half. Every quantity in this probe is measured
   in every pass, so contention lands on all of them together; job 5140's fractions are all
   ≤100% for the first time.

The roof is a **128-bit vectorized copy kernel** (`fused_kernels::stream_copy`) over a buffer
the size of the FFN hidden - same launch geometry, same access width, same size as the
kernels measured against it - so a percentage of it is an efficiency statement rather than a
comparison with a differently-shaped kernel. Contention hazard confirmed once more: the three
CUDA tests fail together while another mlq job holds the device, and pass immediately after
it exits. Run them when `mlq status` shows no active lease.

## 6. Build glue

**A sibling crate, `fused_kernels/`, not `vendor/torch-sys-0.25.0`.** Three reasons, in order
of weight: (a) `torch-sys` is vendored third-party code carrying a 767 KB generated
translation unit, so a `.cu` there means every kernel edit recompiles it and collides with
anyone patching `torch_api.cpp` - which was happening in the main tree during this work;
(b) `torch-sys`'s build script has no CUDA-compilation path at all, it only *includes* CUDA
headers, so the nvcc invocation is new code either way; (c) this crate carries its own
`links = "fused_kernels"`, so Cargo enforces one copy of the kernels per binary. The only
thing given up is `torch_last_err`, which is why `bridge.cpp` carries its own thread-local
error channel behind `fk_last_error`.

Three translation units, deliberately split:

- `csrc/kernels.cu` - pure CUDA, no libtorch types cross into it, so nvcc never parses a
  libtorch header. `nvcc -std=c++17 -O3 -lineinfo --compiler-options -fPIC` for
  `sm_90/sm_100/sm_120` plus `compute_120` PTX, archived with `cc`'s archiver.
- `csrc/bridge.cpp` - shape/dtype/stride contract, output allocation, current-stream
  discovery, the autograd Functions, the event timer. Compiled by `cc` with torch's include
  dirs and reported `_GLIBCXX_USE_CXX11_ABI`.
- `src/lib.rs` - the Rust API, panicking like every other `tch` tensor op because every
  failure it can report is a call-site programming error.

One trap worth recording: naming `-ltorch` is not enough. `libtorch.so` is a stub whose only
job is to pull in `libtorch_cpu` and `libtorch_cuda`; nothing in Rust references a symbol
from it, so the default `--as-needed` drops it along with every CUDA dispatch key, and the
symptom is `Cuda::is_available()` returning **false** in a binary that linked cleanly.
`trading_bots/build.rs` already solves this; `build.rs` emits the same three link arguments
(`-Wl,-rpath=<torch lib>`, `-Wl,--no-as-needed`, `-ltorch`, `-lc10`), which apply to this
package's own bin and tests. The workspace binary is unaffected either way.

Diff to files anyone else owns, all append-only:

| file | change |
| --- | --- |
| `Cargo.toml` | `"fused_kernels"` in `[workspace].members` |
| `trading_bots/Cargo.toml` | `fused_kernels = { path = "../fused_kernels" }` |
| `shared/src/report.rs` | `"timexer_segment_fused_kernels"` appended to `TIMEXER_SEGMENT_REPORT_BASES` |

`benchmark.rs`, `main.rs`, `mod.rs` and `tui` are untouched: the probe is this crate's own
binary (`fused_kernels/run-probe.sh`, the `run-release-cuda.sh` pattern), because a probe
that needs no model, corpus or optimizer should not have to build one.

## 7. Tests

`cargo test -p fused_kernels`: 6 passed. `cargo check -p trading_bot_0 --tests`: zero errors.

CPU, no device needed - these pin the claims the CUDA tests would otherwise rely on silently:

- `composed_relu_square_gradient_is_one_rounded_product` - the composition's gradient IS the
  single-rounding form the kernel implements.
- `composed_relu_square_gradient_mask_is_not_x_greater_than_zero` - the mask is
  `NOT (x <= 0)`, NaN propagates through it, `x == 0` is masked off.
- `packed_rotation_equals_the_per_tensor_rotation` - the packed layout equals the per-tensor
  half-width rotation applied separately to q and k. Without this, an equality test against
  the packed reference would pass while both forms rotated the wrong pairs.

CUDA, gated on `tch::Cuda::is_available()` (skipped, not failed, off the training box):

- `fused_relu_square_is_bit_identical_including_gradients` - forward and backward, at
  `[4096, 2048]` and at `[1023]` so the vectorized body and the scalar path are both covered,
  with exact zeros, a NaN and negatives forced into the sample.
- `fused_rope_is_bit_identical_including_gradients` - forward and backward at the real
  geometry (375 origins, 8 heads, head_dim 64), on a `split_with_sizes` **strided view** of a
  `[.., 3·d_model]` projection, so the gradient has to land in the right columns of a wider
  buffer.
- `both_kernels_capture_and_replay_inside_a_cuda_graph` - §4.

Equality is bit-for-bit, not `allclose`: bf16 widens to f32 losslessly, so f32 equality is
bit equality apart from NaN (required to match NaN) and the sign of zero (pinned by a
`signbit` clause).

## 8. Wiring instructions - for whoever integrates, NOT done here

`model.rs` is untouched by this branch. Add `use fused_kernels::{relu_square, rope};`.

**ReLU², `Block::forward` (model.rs:610-618).** Replace

```rust
&linear(&rms_norm(&state), &self.first).relu().square().dropout(self.dropout, train),
```

with

```rust
&relu_square(&linear(&rms_norm(&state), &self.first)).dropout(self.dropout, train),
```

The input is contiguous bf16 `[batch, length, ffn]`, which is the contract. Bit-identical, so
nothing downstream moves.

**Packed rotary, `Block::rotate` (model.rs:475-494).** Delete the method and replace its one
call site (model.rs:563) with

```rust
let rotated = rope(&normed, rotation.0, rotation.1, self.heads).split(1, 2);
```

Then change what `rotation` carries. `CausalPatchModel::new` (model.rs:~858-860) already
builds the untiled `cosine`/`sine` as `[origins, head_dim/2]` bf16 and then discards them
into `rotation_tiles`; **store `(cosine, sine)` in the `rotation` field and delete
`rotation_tiles` entirely** (model.rs:411, plus its two test uses at 2390 and 2713). The
kernel indexes the untiled rows by `row % length` internally - 24 KiB each, L2-resident - so
the two `[1, origins, 2·d_model]` broadcast tiles are never built. `normed` is contiguous
(it is `rms_norm(...).reshape(...)`), and the kernel also accepts the strided pre-QK-norm
view if that order ever changes. The output shape `[batch, length, 2, heads, head_dim]` is
unchanged, so `.split(1, 2)` and everything after it is untouched.

**`kernel_classes` (model.rs:1256-1264 and 1345-1359).** Both classes must be re-pointed at
the kernels or the profile will keep charging the old traffic:

- `"rotary rotation"`: `run` becomes `rope(&input[0], cosine, sine, heads)`, and
  `forward_bytes` goes `18. * state` -> `2. * state` (read `q‖k`, write the rotated buffer;
  the cos/sin rows round to nothing). `forward_flops` stays `6. * tokens * width`.
- `"ReLU^2"`: `run` becomes `relu_square(&input[0])`, `forward_bytes` goes `4. * hidden` ->
  `2. * hidden`, and the comment ending "ATen has no fused ReLU², so until one exists this is
  the honest number" is now false and should say the kernel is `fused_kernels::relu_square`.

Re-pointing the two classes is not sufficient on its own. The composed-layer class charges
every class its own repeat count (`RMSNorm` twice, the residual `addcmul` three times, the
value-residual mix `layers-1` times, QK norm once), and `benchmark.rs` prints the attribution
error between that composed figure and the sum of the classes - so the composed layer has to
be recomputed from the SAME `forward_bytes` the two classes now carry, or the error line will
report the gap this change opens rather than a real one. Integrator flagged this; whoever
owns the class list should make both edits together. Separately, per §5, `device_peaks` should
be interleaved with the classes it is a denominator for, or its fractions stay
contention-dependent.

**Leave alone.** `relu_squared_matches_its_definition_and_is_not_gelu` (model.rs:2624) runs on
CPU fp32 and tests the definition, not the kernel; the kernels are CUDA-only. Model tests that
compare against `relu().square()` or the composed rotation keep passing unchanged, because
both kernels are bit-identical - **no test tolerance is needed anywhere**.

**Expected effect.** -52.9 ms/step at B=256 on the shapes measured here, of which -26.8 ms is
ReLU² (83% of ResidualRecipe's +18.87 GB/step traffic delta, the unfoldable `aten::square`
term) and -26.1 ms is the rotary. After wiring, the remaining rotary and ReLU² time is at
97-100% of the streaming roof: nothing further is available from either without removing a
pass, which for ReLU² means the GEMM epilogue in §3 and for the rotary means nothing at all.
