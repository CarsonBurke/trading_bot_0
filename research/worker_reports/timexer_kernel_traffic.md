# CausalPatch backbone: per-kernel-class traffic

What each class of backbone kernel actually costs, measured with CUDA events at the real
configuration; what autograd retains and in what dtype; which fusions landed and what they
bought; and which primitive the remaining traffic needs and ATen does not have.

Configuration throughout: `seq_len 6000`, `pred_len 192`, `patch_len 16`, `layers 8`,
`d_model 512`, `heads 8`, `ffn 2048`, `n_aux 12`, batch **256**, 375 causal-patch origins per
row -> **96,000 tokens per step**. One bf16 `[96000, 512]` activation ("one state unit") is
**98.3 MB**; one `[96000, 2048]` is 393.2 MB. Device: RTX 5090.

Measured by job **5133** (`kerneltraffic-probe4`, `benchmark-timexer-segment --batch-size 256
--steps 12 --warmup 6 --profile`). Every number below comes from that one process.

## 0. Read this first: the device was shared

Job 5133's own in-process peak probes measured **94 TFLOPS** and **630 GB/s** while its own
kernels measured **104 TFLOPS** and **1569 GB/s** - denominators smaller than their numerators.
`nvidia-smi` showed four non-mlq python processes holding ~21 GB at 100% utilisation for the
duration. So:

- **Absolute step time is not comparable to the 302 ms baseline.** 5133 measured **406.8 ms**.
  That is a contended-device number, not a regression: nothing in this work added a kernel.
- **Ratios inside the run are sound.** Every class was timed in the same process, minutes apart
  at most, so class-vs-class and before-vs-after comparisons hold.
- Denominators used below are therefore the **in-run empirical roofs**: the fastest bandwidth any
  kernel reached in this run (**1569 GB/s**, the residual add) and the fastest arithmetic
  (**104.4 TFLOPS**, the QKV projection forward). Both are lower bounds on the card's real roofs
  and both are honest for comparing classes against each other.
- The probe defect is fixed: `device_peaks` now takes the **best** of 8 CUDA-event-timed rounds
  instead of the mean of 10 host-timed ones. A peak is a maximum; one contended round must not
  halve every fraction in the profile. Locally the fixed probe measures **1524 GB/s** where the
  old form measured 630.

## 1. Per-kernel-class table (batch 256, job 5133)

`fwd`/`bwd` are milliseconds per invocation; `step ms` multiplies by invocations per step (16 for
the two norms and two residual adds, 8 for the rest, 1 for the embedding). `fwd MB` and the rates
are the **forward pass only**: its bytes and FLOPs are exactly enumerable from the shapes, so
`bytes/time` is a measurement rather than a convention. Backward bytes are deliberately not
charged per class - see section 2.

**This table is the pre-BarTrunk-recipe baseline**, measured on LayerNorm + GELU + plain
residual adds, which is what it is for: the recipe replacing them (RMSNorm, ReLU^2, QK-norm,
`x0` value residual, zero-init output projections) must be measured against these numbers rather
than against a fresh unknown. When it lands, three class names in the harness change
(`LayerNorm` -> `RMSNorm`, `GELU` -> `ReLU^2`, `residual add` -> `residual addcmul`, the last at
three invocations per layer) and the per-layer unit count goes `19*width + 2*ffn` ->
`22*width + 3*ffn`; `research/worker_reports/causalpatch_residual_recipe.md` carries that
accounting. The invocation counts quoted here are the pre-recipe ones.

| kernel class | fwd ms | bwd ms | step ms | % of step | fwd MB | GB/s | % roof | TFLOPS | % roof |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FFN up projection | 2.047 | 5.467 | **60.1** | 14.8% | 497.8 | 243 | - | 98.4 | 94% |
| FFN down projection | 1.973 | 4.871 | **54.8** | 13.5% | 497.8 | 252 | - | 102.0 | 98% |
| rotary rotation | 2.597 | 4.191 | **54.3** | 13.4% | 1769.5 | 681 | **43%** | - | - |
| QKV projection | 1.446 | 4.209 | **45.2** | 11.1% | 397.9 | 275 | - | 104.4 | 100% |
| causal SDPA | 0.635 | 3.760 | **35.2** | 8.7% | 396.3 | 624 | 40% | 58.1 | 56% |
| GELU | 0.868 | 2.528 | **27.2** | 6.7% | 786.4 | 906 | 58% | - | - |
| LayerNorm | 0.132 | 1.107 | **19.8** | 4.9% | 197.4 | 1498 | 95% | - | - |
| attention output projection | 0.620 | 1.323 | **15.5** | 3.8% | 198.2 | 320 | - | 81.2 | 78% |
| residual add | 0.188 | 0.062 | **4.0** | 1.0% | 294.9 | **1569** | 100% | - | - |
| attention output flatten | 0.135 | 0.060 | **1.6** | 0.4% | 196.6 | 1456 | 93% | - | - |
| patch embedding tokens | 0.687 | 0.000 | **0.7** | 0.2% | - | - | - | - | - |
| **sum of classes** | | | **318.4** | 78.3% | | | | | |
| composed layer x8 + embedding | 13.133 | 27.453 | **325.4** | 80.0% | | | | | |

**Attribution error -2.2%.** The classes account for 98% of the composed layer, so the 220 ms
residual the previous report attributed to "backbone" is now named. The step's remaining
81.4 ms (20%) is gather, H2D, the fused head/loss and the captured optimizer.

Backward rates, where the backward's arithmetic or traffic is exactly derivable:

| class | backward | rate | % of in-run roof |
| --- | --- | --- | --- |
| LayerNorm | 1.107 ms | 267 GB/s | **17%** (worst in the backbone) |
| causal SDPA | 3.760 ms | 24.5 TFLOPS | 23% |
| GELU | 2.528 ms | 467 GB/s | 30% |
| QKV projection | 4.209 ms | 71.8 TFLOPS | 69% |
| FFN up projection | 5.467 ms | 73.6 TFLOPS | 71% |
| attention output projection | 1.323 ms | 76.1 TFLOPS | 73% |
| FFN down projection | 4.871 ms | 82.7 TFLOPS | 79% |
| residual add / flatten | 0.062 / 0.060 ms | ~0 bytes | backward is views only |

### The two structural facts

- **The four GEMMs are 175.6 ms, 43.2% of the step, and they run at 82.5 TFLOPS = 79% of the
  in-run arithmetic roof.** Nearly half the step is already within a fifth of the fastest
  arithmetic this device delivered. There is no fusion to win there; only fewer FLOPs or a
  quieter device.
- **The non-GEMM backbone is 142.0 ms, 34.9% of the step, and that is where the roofline
  fractions are 17-58%.** The rotary rotation alone is 54.3 ms at 43% of a bandwidth roof that
  the residual add, a plain contiguous elementwise add, reaches in full in the same run.

## 2. Why backward bytes are not charged per class

`step_cost` charges the whole step "forward once, backward twice more". That is right in
aggregate and wrong per class, and the first version of this table proved it: it reported the
residual add at **3540 GB/s**, above this card's spec bandwidth, and the attention flatten at
3026 GB/s. Both are real measurements of a backward that moves nothing - `add`'s backward hands
the same gradient tensor to both inputs, and `transpose().reshape()`'s backward is a view chain,
so `run_backward` stores a strided view and copies nothing. A GEMM's backward, by contrast, is
two more GEMMs. The table therefore reports exact forward rates plus backward time, and derives
backward rates only for the classes whose backward formula is known (table above).

## 3. fp32-activation audit

Measured as the live-allocator delta across one forward with the output still held, so anything
autograd saved beyond the class's own output appears as a number. An fp32 `[96000, 512]`
activation would be **187.5 MiB** (196.6 MB); the largest thing measured is 2.93 MiB.

| class | retained beyond output | what it is | shape | dtype | accidental? |
| --- | --- | --- | --- | --- | --- |
| causal SDPA | 2.93 MiB | flash logsumexp | `[256, 8, 375]` | **fp32** | no - flash's own contract |
| FFN up projection | 2.00 MiB | bf16 weight cast | `[2048, 512]` | bf16 | no - autocast's cast cache |
| FFN down projection | 2.00 MiB | bf16 weight cast | `[512, 2048]` | bf16 | no |
| QKV projection | 1.50 MiB | bf16 weight cast | `[1536, 512]` | bf16 | no |
| LayerNorm (x2) | 0.73 MiB each | `mean`, `rstd` | `2 x [96000]` | **fp32** | no - `native_layer_norm`'s saved pair |
| attention output projection | 0.50 MiB | bf16 weight cast | `[512, 512]` | bf16 | no |
| rotary rotation | 0.00 MiB | - | - | - | - |
| GELU | 0.00 MiB | - | - | - | - |
| residual add | 0.00 MiB | - | - | - | - |
| attention output flatten | 0.00 MiB | - | - | - | - |

**Result: there is no accidental fp32 activation anywhere in the backbone.** Total non-activation
retention is 10.4 MiB per layer, of which **fp32 is 4.39 MiB** (the two LayerNorm moment pairs
plus the flash logsumexp) - 0.27% of the layer's 1604.15 MiB. Every one of the four candidate
mechanisms named in the assignment was checked and none fires:

- **`.to_kind` in a forward path**: the only casts on the training path are weight casts
  (fp32 master -> bf16, cached by autocast, the 0.50-2.00 MiB rows above) and the one deliberate
  activation cast in `tokens`.
- **fp32 master weights meeting bf16 activations**: they meet only inside the cast, never as an
  activation.
- **LayerNorm saving an fp32 input copy**: it does not. It saves the bf16 input it was given plus
  0.73 MiB of fp32 moments. Had it saved an fp32 copy the row would read 188.2 MiB.
- **fp32 activations surviving the embedding**: the fp32 price/auxiliary intermediates
  (`[256, 375, 64]` fp32 = 24.6 MB and `[256, 375, 192]` fp32 = 73.7 MB) carry no gradient, are
  freed inside the same call, and the `[256, 375, 256]` fp32 concatenation that used to sit
  between them is gone (section 4).

What the composed layer retains is **1593.8 MiB of bf16 activations per layer = 17.0 state
units**: norm0 out (1), packed qkv (3), rotation buffer (2), SDPA out (1), flattened out (1),
post-residual state (1), norm1 out (1), FFN hidden (4), GELU out (4). All bf16, all needed by a
backward. At 8 layers that is 12.4 GiB, consistent with the 17.7 GiB peak allocator.

## 4. Fusions landed, with measured savings

Both are pinned bit-identical by tests in `model.rs` that compare the new form against the old
one on the same inputs; the old forms are also compiled into the profile as `reference *` classes
so the before/after is one process, not two runs.

| fusion | before | after | saved | % of that term | % of step |
| --- | --- | --- | --- | --- | --- |
| rotary: two half-width products per tensor -> two full-width products over the packed `q\|\|k` block | 61.0 ms | 54.3 ms | **6.7 ms** | 10.9% | 1.64% |
| patch embedding: cast to bf16 **before** the concatenation instead of after | 0.869 ms | 0.687 ms | **0.18 ms** | 21% | 0.04% |
| **total** | | | **6.9 ms** | | **1.68%** |

- **Rotary.** Identical bytes (1769.5 MB forward either way): the win is kernel count and
  coalescing. Fourteen kernels per layer over strided half-views became five over contiguous
  full-width columns, and forward bandwidth rose **550 -> 681 GB/s (+24%)**, forward time
  3.219 -> 2.597 ms (-19.3%). Bit-identical because each output element is still one
  bf16-rounded product and one bf16-rounded sum of the same two operands.
- **Embedding.** Casting the two fp32 branches before `cat` replaces a 98.3 MB fp32 write and
  re-read with a 49.2 MB bf16 one. Bit-identical: the cast is elementwise and commutes with
  concatenation exactly.
- **Rotation tiles hoisted to construction** (nine small kernels per attention tensor per layer
  removed, ~576 launches per step). Not separately measured - the reference class is given
  pre-built tiles too, so the profile cannot isolate it. Claimed as a launch-count reduction
  only.

## 5. Fusions NOT landed, and the exact primitive that is missing

Each was checked against ATen/tch rather than assumed.

- **GELU into the FFN epilogue.** `aten::_addmm_activation` exists and tch binds it
  (`Tensor::internal_addmm_activation(mat1, mat2, use_gelu)`). It is **unusable in training**:
  `derivative for aten::_addmm_activation is not implemented`. Measured directly. It is also not
  bit-identical to either GELU form (2.78e-4 max relative error against both `gelu("none")` and
  `gelu("tanh")`), so even an inference path would need a tolerance decision. Its ReLU epilogue
  (`use_gelu=false`) **is** bit-identical to `relu`, which matters for the incoming ReLU^2
  recipe - but the missing derivative blocks it there too. **Missing primitive: a differentiable
  fused GEMM+activation epilogue.**
- **ReLU^2 as a single kernel** (the successor activation; `ResidualRecipe` measured the
  accounting). `relu().square()` is **two** ATen kernels and materializes an extra
  `[96000, 2048]`, so its forward traffic is `4 * hidden` = 1572.9 MB per layer against GELU's
  `2 * hidden` = 786.4 MB - the activation gets *cheaper* in FLOPs and **twice as expensive in
  traffic**, +18.87 GB/step. `dir(torch.ops.aten)` has `relu`, `relu6`, `relu_`, `glu` and
  **nothing squared or fused**, and the epilogue op above cannot carry it either (no
  derivative). **Missing primitive: fused ReLU^2** - upstream implementations are hand-written
  Triton. On the measured GELU term (27.2 ms at 58% of the bandwidth roof forward) a 2x traffic
  increase makes this the largest fusion opportunity in the new recipe, ahead of the rotary.
- **Residual add folded into the following normalization.** ATen has `native_layer_norm`,
  `native_layer_norm_backward`, `_fused_rms_norm`, `_fused_rms_norm_backward`, and no variant of
  any of them takes a residual input. **Missing primitive: `add`+norm fusion.** Measurement says
  do not care: the residual add is **4.0 ms of 406.8 (1.0%)**, it already runs at the in-run
  bandwidth roof (1569 GB/s, the fastest kernel in the backbone), and its backward moves nothing.
  Folding it saves the 2-of-3 units the fold removes, ~2 ms/step. This is the clearest case for
  measuring before fusing: the "obvious" fusion is worth 0.5% of the step.
- **Fused rotary.** `dir(torch.ops.aten)` contains **no** rotary or RoPE op at all. Upstream
  fused rotary lives in flash-attn and TransformerEngine, neither of which tch binds.
  **Missing primitive: fused rotary embedding.** This is the single largest identified
  opportunity - see section 6.
- **Elementwise chains collapsed into single passes.** Already done where tch allows it: the
  rotary's four half-width kernels became two full-width ones, the embedding's cast and scale
  ride the same pass, and the head/loss chain was collapsed in prior work. The remaining
  elementwise classes are each a *single* ATen kernel (`gelu`, `add`, `native_layer_norm`); there
  is no chain left to collapse without a custom kernel.

### Layout copies: which are ours, which are SDPA's

- **q, k, v are strided VIEWS, never copies.** q/k stride over `2*d_model` in the rotation
  buffer, v over `3*d_model` in the packed projection, all with `head_dim` contiguous. SDPA needs
  only unit stride on the last dimension. Confirmed by measurement (the SDPA class retains
  2.93 MiB - the logsumexp and nothing else; three materialized copies would be 281 MiB) and by
  configuration (the process runs **SDPA flash only**, so a layout flash could not accept would
  raise rather than silently fall back). Materializing them would cost three 98.3 MB copies a
  layer for nothing.
- **The attention output flatten is a real copy and is structurally required.** SDPA emits
  `[batch, heads, length, head_dim]`; the output projection consumes
  `[batch, length, heads*head_dim]`. The head and length axes must be swapped before they merge,
  and no stride expresses that merge. Cost: **1.6 ms of 406.8 (0.4%)** at 1456 GB/s = 93% of the
  in-run roof, with a free backward. Optimal already.
- **The `cat` in `tokens` is a real copy and is structurally required**: the patch-embedding GEMM
  needs one contiguous `[96000, 256]` operand. Its cost is now paid in bf16 rather than fp32.
- **`.contiguous()` in the training path: none.** The only occurrence left in `model.rs` is
  inside a test.
- The rotation-tile construction (`reshape -> expand -> reshape`) does materialize a copy, and it
  runs **once at construction**, not per step.

## 6. Residual bottleneck and the new binding constraint

**Binding constraint: the four projection GEMMs.** 175.6 ms, 43.2% of the step, at 79% of the
in-run arithmetic roof. Even if every elementwise term went to zero the step could not go below
175.6 + 81.4 = **257 ms in this device state**. Reducing it needs fewer FLOPs (narrower `d_model`
or `ffn`, or a smaller `n_aux`), not a fusion. On an uncontended device the same 14.5 TFLOP at
the previously measured 209 TFLOPS roof is ~70 ms, which is where the step-level roofline in
`timexer_step_efficiency.md` came from.

**Second-order constraint, and the largest remaining opportunity: the rotary rotation.**
54.3 ms, 13.4% of the step, at 43% of a bandwidth roof that a plain contiguous add reaches in
full in the same run. It moves **18 state units** per layer forward - two full-width products
(4 each), two half-crossing sums (3 each), and the interleaving stack (4) - where a fused kernel
reads 2 and writes 2. At 4 units and the in-run roof the forward would be **0.25 ms/layer**
against 2.60 measured; the whole term would fall to roughly 12-15 ms, worth **~40 ms, 10% of the
step**. It cannot be done with a tch primitive (section 5) and needs a hand-written kernel.

Then, in order of measured size:

- **LayerNorm backward: 17.7 of the 19.8 ms LN term, at 17% of the bandwidth roof** - the worst
  roofline fraction in the backbone. The forward is already at 95%; the cost is the `dgamma`/
  `dbeta` reductions over 96,000 rows. **`aten::_fused_rms_norm` is bound in tch
  (`internal_fused_rms_norm`), is differentiable, is bit-identical to `F.rms_norm`, and saves
  only an fp32 `[96000, 1]` rstd instead of LayerNorm's mean+rstd pair** - so the incoming
  RMSNorm swap has one fewer parameter reduction and one fewer saved tensor. This 19.8 ms is the
  baseline it should be measured against.
- **SDPA backward: 30 ms at 23% of the arithmetic roof.** flash-attn's own kernel at a 375-token
  sequence; not reachable from tch.
- **GELU: 27.2 ms**, forward at 58% and backward at 30% of the bandwidth roof. The ReLU^2
  replacement removes the erf entirely; the epilogue fusion that would remove the pass is blocked
  by the missing derivative.

## 7. What this work changed in the harness

- `benchmark.rs --profile` now measures the backbone at kernel-class granularity: per class, CUDA
  event forward and backward time, exact forward bytes and FLOPs, achieved GB/s and TFLOPS
  against both measured device roofs, and the live-allocator delta that exposes what autograd
  retained. The sum of the classes is printed against the composed layer so the attribution
  error is stated rather than assumed (-2.2%).
- Three new `.report.bin` bases, registered in `shared/src/report.rs` and reachable from
  `meta_chart_bases`, each answering one question in one unit:
  - `timexer_segment_benchmark_kernels` - **milliseconds**, per class, forward and backward
    separated;
  - `timexer_segment_benchmark_kernel_roofline` - **percent of the measured device peak**, per
    class, bandwidth and arithmetic separated;
  - `timexer_segment_benchmark_kernel_activations` - **mebibytes**, per class, saved-for-backward
    against class output.
- `device_peaks` fixed to best-of-8 CUDA-event rounds (section 0).
- `event_timed` releases the GIL around the timed operation: the autograd engine refuses to be
  entered by a thread holding it, so timing a backward inside `Python::attach` aborted the
  process. This cost two probe jobs (5128, 5132 - both died before producing a measurement).

Probe jobs submitted: 5124 (build failure, sibling's mid-flight tree), 5128 and 5132 (the GIL
abort above), **5133 - the measurement this report is built on**.

## 8. Verification

- **Numerics: bit-identical, observed relative error 0.0.**
  `the_packed_rotation_and_split_match_the_per_tensor_reference_bit_for_bit` builds the
  pre-change per-tensor rotation and the new packed one from the same weights and asserts
  `max|Δ| == 0.0` on q, k and v, on the composed block output, and on **all 13 gradients**
  (the block's every parameter plus the activation input). It prints
  `packed rotation vs per-tensor reference: 0.0e0 on q/k/v, on the composed block output and on
  all 13 gradients`. Each assertion is guarded by a non-degeneracy check (`> 0.0` signal) and by
  a check that the tiles actually rotate, so equality cannot pass on untouched tensors.
  `the_patch_tokens_cast_before_concatenation_are_bit_identical` does the same for the
  embedding, comparing against the cast-after-concatenation form. The objective
  `causal_patch_market_neutral_nll_v2` is untouched.
- `cargo test -p trading_bot_0 --lib timexer_segment`: **38 passed, 0 failed**.
- `cargo check -p trading_bot_0 --tests --bins`: **0 errors**.
  `cargo check -p trading-bot-tui --tests`: **0 errors** (the crate's package id is
  `trading-bot-tui`, not `tui`). `cargo test -p trading-bot-tui`: 36 passed, which includes both
  directions of the base-registry contract for the three new bases.
- Measured step time from the queue: **406.8 ms at batch 256, job 5133** - a contended-device
  number, see section 0.
