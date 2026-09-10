# CausalPatch device-work inventory

## Executive conclusion

**DEMONSTRATED:** The source does **not** support interpreting 143.4 GB as measured HBM traffic, nor interpreting its ratio to step time as a measured bandwidth utilization. `ModelConfig::step_cost` constructs an activation-materialization estimate using a blanket forward/backward multiplier; it omits some real work and charges some backward work that does not exist. The 16.98 TFLOP figure is also an estimate of matrix work, not a complete executed-instruction count.

**DEMONSTRATED:** Most matrix arithmetic is in the backbone, especially FFN up/down and QKV—not the head. The dense head is substantial, but its entire forward/backward accounting, including covariates and loss, is approximately **9% of charged matrix FLOPs and 19% of charged traffic**. It is not responsible for most of the 143.4 GB ledger.

**HYPOTHESIS:** The likely explanation is alternating compute-intensive projections and low-intensity normalization/activation/residual/gradient-plumbing work **throughout the backbone**, plus a bandwidth-intensive head/loss and short-sequence attention. The narrower explanation “compute-bound trunk, bandwidth-bound head” misses much of the trunk’s own memory work. Actual phase-time shares remain unmeasured by this investigation; source arithmetic cannot uniquely identify them.

**DEMONSTRATED — denominator warning from coordination:** LoaderPerf confirmed that the benchmark version inspected here never arms its main forward/backward graph and includes a fresh synthetic GPU gather in every timed iteration. Its separate capture audit uses separate engines. Consequently, old benchmark timing cannot establish the performance of the actual captured training path. This does not contradict the supplied observation that training captures at `runner.rs:1657`; it distinguishes the two execution paths. LoaderPerf owns the correction. All time predictions below are conditional estimates, not recovered measurements.

## 1. Shapes and conventions

**DEMONSTRATED:** These are the default dimensions in `model.rs:16-25,465-521` and `features.rs:26-54,134-142`:

| Quantity | Value |
|---|---:|
| Batch B | 256 |
| Context bars | 6,000 |
| Patch length | 16 |
| Attention sequence length L | **375**, not 6,000 |
| Tokens N = B×L | **96,000** |
| Model width D / heads / head dimension | 512 / 8 / 64 |
| Layers | 8 |
| FFN width F | 2,048 |
| Historical auxiliary channels | 12 |
| Known-future channels | 6 |
| Patch input width | 16×(4+12) = 256 |
| Known-future flattened width | 192×6 = 1,152 |
| Covariate projection width | 256 |
| Head input / hidden width | 768 / 1,024 |
| Forecast horizon H | 192 |
| Head outputs per horizon | **8 = 4 candle coordinates + 4 log scales** |

**DEMONSTRATED:** Eight head outputs are not eight price channels with another eight scales. The source has `CHANNELS = 4` and `OUTPUTS_PER_BAR = 2*CHANNELS`.

**DEMONSTRATED:** Convenient byte units, using decimal GB throughout:

- S = one bf16 `[N,512]` tensor = **98,304,000 bytes = 0.098304 GB**.
- One bf16 `[N,2048]` tensor = 4S = **0.393216 GB**.
- U = one fp32 `[B,L,1,H]` channel = **73,728,000 bytes = 0.073728 GB**.
- A matrix multiplication `[N,K]×[K,M]` costs `2NKM` mathematical FLOPs, counting FMA as two.
- RTX 5090 nominal machine balance under the supplied ceilings is `209/1.79 ≈ 116.76 FLOP/byte`.

**DEMONSTRATED:** The supplied 27,232,883-parameter count identifies the `basis:8:8` head, with the other dimensions above and x0 enabled. The free head has **27,954,483** parameters. This distinction matters for exact horizon-shortening parameter savings.

## 2. Exact reconciliation with the existing totals

### 2.1 What this table is—and is not

**DEMONSTRATED:** The following table exactly reconstructs `step_cost`, `model.rs:615-761`. Its FLOP columns include **only the matrix arithmetic that function charges**. Its byte columns implement its convention: a materialized activation of size A contributes `2A` forward and `4A` backward. They are **not** independently verified forward/backward memory transactions.

**DEMONSTRATED:** A zero in the matrix-FLOP column for an elementwise phase means **“not charged by step_cost,” not “no arithmetic runs.”** Actual elementwise work is detailed separately below. This distinction is necessary to sum to the claimed totals without manufacturing false precision.

All rows below are **DEMONSTRATED as accounting derivations**, not demonstrated hardware traffic. Layer rows include all eight layers. AI is charged forward matrix FLOPs / charged forward bytes; it must not be mistaken for the physical kernel AI.

| Phase and output shape | Forward TFLOP charged | Backward TFLOP charged | Forward GB charged | Backward GB charged | Ledger AI, FLOP/B |
|---|---:|---:|---:|---:|---:|
| Patch projection, `[N,256]→[N,512]`; byte charge is its input | 0.025165824 | 0.050331648 | 0.098304 | 0.196608 | 256 |
| QKV projections, 8×`[N,512]→[N,1536]` | 1.207959552 | 2.415919104 | 4.718592 | 9.437184 | 256 |
| Causal attention, 8×`[256,8,375,64]` | 0.294912000 | 0.589824000 | 1.572864 | 3.145728 | 187.5 |
| Attention output flatten/layout, 8×`[N,512]` | 0 | 0 | 1.572864 | 3.145728 | 0 |
| Attention output projections, 8×512→512 | 0.402653184 | 0.805306368 | 1.572864 | 3.145728 | 256 |
| FFN up, 8×512→2048 | 1.610612736 | 3.221225472 | 6.291456 | 12.582912 | 256 |
| FFN down, 8×2048→512 | 1.610612736 | 3.221225472 | 1.572864 | 3.145728 | 1024 |
| Fused ReLU², 8×`[N,2048]` | 0 | 0 | 6.291456 | 12.582912 | 0 charged |
| Fused QK-norm + RoPE, 8×`[N,1024]` | 0 | 0 | 3.145728 | 6.291456 | 0 charged |
| RMSNorm: 16 pre-norms + patch norm + final norm | 0 | 0 | 3.538944 | 7.077888 | 0 charged |
| Two residual addcmuls/layer, 16×`[N,512]` | 0 | 0 | 3.145728 | 6.291456 | 0 charged |
| Four U-net skip addcmuls | 0 | 0 | 0.786432 | 1.572864 | 0 charged |
| Eight x0 addcmuls | 0 | 0 | 1.572864 | 3.145728 | 0 charged |
| Known-future window + covariate projection, `[N,1152]→[N,256]` | 0.056623104 | 0.113246208 | 0.540672 | 1.081344 | 104.73 |
| Head input concatenation, `[N,768]` | 0 | 0 | 0.294912 | 0.589824 | 0 |
| Head hidden projection, 768→1024 | 0.150994944 | 0.301989888 | 0.393216 | 0.786432 | 384 |
| Head GELU, `[N,1024]` | 0 | 0 | 0.393216 | 0.786432 | 0 charged |
| Head output projection, 1024→1536 | 0.301989888 | 0.603979776 | 0.589824 | 1.179648 | 512 |
| Geometry + NLL + targets/masks: `(121+10+3)U` forward, `144U` backward | 0 | 0 | 9.879552 | 10.616832 | Not enumerated |
| Seven value-residual mixes + first-value gradient accumulation | 0 | 0 | 2.064384 | 6.586368 | 0 charged |
| **Free-head total** | **5.661523968** | **11.323047936** | **50.036736** | **93.388800** | **118.42 combined** |

**DEMONSTRATED:** Summing the table yields exactly:

- **16.984571904 TFLOP**.
- **143.425536 GB**.

**DEMONSTRATED:** For `basis:8:8`, output parameters have width `4×(192+8+8) = 832`, but emitted outputs remain 1,536. The weight-space expansion adds, as charged:

- Forward: `2×1536×832×1024 = 2,617,245,696 FLOPs`.
- Backward: the same, since the expansion matrix is constant.
- Traffic: **0.012582912 GB** combined under the existing four-materialization convention.
- Totals: **16.989806395392 TFLOP, 143.438118912 GB**.

**DEMONSTRATED:** Thus 16.98/143.4 are rounded ledger values; the parameter count supplied in the assignment belongs to the slightly larger-arithmetic basis case, not the free-head case.

### 2.2 Concrete accounting defects

**DEMONSTRATED — nonexistent input gradients:** Patch tokens and future covariates are data, not differentiable activations (`model.rs:1529-1576,2077-2095`; `compute.rs:903-908`). Their projections execute forward and weight-gradient GEMMs, **not** an additional input-gradient GEMM. The uniform multiplier overcharges:

- Patch: **0.025165824 TFLOP**.
- Covariates: **0.056623104 TFLOP**.
- Total: **0.081788928 TFLOP**, about 0.48% of the quoted total.

**DEMONSTRATED — fused-kernel backward is not twice forward bytes:** ReLU² explicitly reads saved input and upstream gradient and writes one gradient (`fused_kernels/csrc/kernels.cu:47-115`). Across eight layers this is **6.291456 GB forward + 9.437184 GB backward**, not the ledger’s 6.291456 + 12.582912. Likewise QK-norm/RoPE directly streams **3.145728 GB forward + 4.718592 GB backward**, excluding cached rotation tables and any bridge-side gradient copy, not the ledger’s 3.145728 + 6.291456 (`kernels.cu:306-468`). These two rows alone show that the blanket multiplier is not a strict per-phase lower bound.

**DEMONSTRATED — target work exceeds the ten-unit allowance:** `targets`, `model.rs:2165-2174`, now performs subtraction, division and market-drift subtraction for **four fp32 channels**, plus market-drift and mask construction. The first subtraction and division alone require **16U** of logical large-tensor reads/writes, already exceeding the entire `10U` allowance. With each full-sized operand access counted and small per-origin scales cached, the source expression is **34U**: `8+8+12` for targets, `4` for market drift and `2` for the mask. Some broadcast drift rereads can hit cache; that does not rescue the old op inventory.

**DEMONSTRATED — forward loss allowance also fails a direct logical-I/O enumeration:** Ignoring small cached per-origin/per-horizon buffers, the current loss body requires:

| Forward loss subphase | Logical channel-space passes |
|---|---:|
| Geometry, including three bf16→fp32 coordinate casts | 43U |
| Precision construction | 2U |
| Four scale/weight/error/square chains | 62U |
| Eight objective dots + four MSE dots | 24U |
| Weighted-mask construction + prior/objective/raw-mask reductions | 5U |
| **Loss subtotal** | **136U** |
| Targets, market drift and mask | **34U** |
| **Combined forward logical-I/O subtotal** | **170U = 12.533760 GB** |

**DEMONSTRATED:** This is not the ledger’s `134U = 9.879552 GB`. The difference is **36U = 2.654208 GB forward** before small tensors and reduction workspace. It is a logical-access inventory, **not a claim that all those reads miss L2**. Backward’s `144U` comment has no corresponding executable inventory and cannot be promoted to an exact measurement.

**DEMONSTRATED — other missing work:** `step_cost` does not enumerate statistics/cumsums, patch normalization arithmetic, casts of master weights and gradients, bias reductions, much autograd scatter/accumulation, RMSNorm statistics, optimizer work, gradient zeroing, resident-batch upload, or scalar-result copies (`model.rs:1440-1490,837-860`; `compute.rs:609-620,851-879,950-959`). It also omits the basis folded-bias matvec and expansion cast.

**HYPOTHESIS — attention execution count:** The ledger counts the causal triangle as exactly half a square and backward as twice forward. The exact useful triangle has `L(L+1)/2` entries. A recomputing flash backward normally adds QK recomputation to the four derivative matrix products, making its matrix work roughly **2.5× forward**, not 2×. Under that execution model, attention adds approximately **0.147456 TFLOP** above the ledger, before diagonal/tile effects. The selected backend/kernel must be observed before declaring the executed count.

**DEMONSTRATED:** These errors have opposite signs. It would be incorrect to “fix” 143.4 GB by subtracting one known overcount and call the result physical traffic.

## 3. Physical phase character: compute versus bandwidth

**DEMONSTRATED:** The following matrix AIs use the actual full activation inputs and outputs rather than assigning only output materializations. They neglect relatively small weight operands and cache effects. Both gradient GEMMs of a normal differentiable projection have similar large-activation AI because they exchange the same activation-sized operands; data-input projections have only the weight-gradient GEMM.

| Phase | Forward matrix FLOPs, all repeats | Mandatory large activation I/O forward | Approx. AI | Classification at 117 FLOP/B |
|---|---:|---:|---:|---|
| Patch projection | 25.165824 GF | 0.147456 GB | 170.67 | **DEMONSTRATED:** above balance; compute candidate |
| QKV | 1,207.959552 GF | 3.145728 GB | 384 | **DEMONSTRATED:** compute candidate |
| Attention output projection | 402.653184 GF | 1.572864 GB | 256 | **DEMONSTRATED:** compute candidate |
| FFN up | 1,610.612736 GF | 3.932160 GB | 409.6 | **DEMONSTRATED:** compute candidate |
| FFN down | 1,610.612736 GF | 3.932160 GB | 409.6 | **DEMONSTRATED:** compute candidate |
| Covariate projection, excluding window gather | 56.623104 GF | 0.270336 GB | 209.45 | **DEMONSTRATED:** compute candidate |
| Head hidden projection | 150.994944 GF | 0.344064 GB | 438.86 | **DEMONSTRATED:** compute candidate |
| Head output projection | 301.989888 GF | 0.491520 GB | 614.4 | **DEMONSTRATED:** strongly compute candidate |
| Causal attention, useful matrix work | 294.912 GF | ≥3.145728 GB plus logsumexp | ≤93.75 | **HYPOTHESIS:** short-sequence/memory or scheduling limited; not automatically compute-bound |

**DEMONSTRATED:** These classifications describe ideal roofline position, not attained utilization. A GEMM above balance can still be limited by tile utilization, occupancy or its selected algorithm.

**DEMONSTRATED:** The complete non-GEMM operation families are:

| Phase | Forward arithmetic and storage | Backward arithmetic and storage | Resource character |
|---|---|---|---|
| Fused ReLU² | Comparison + square over 1,572,864,000 elements; one bf16 input/output | Two multiplies + threshold selection per element; saved input + gradient → gradient | **DEMONSTRATED:** about 0.5 ordinary operations/byte forward; bandwidth candidate |
| QK RMSNorm + RoPE | About `12ND×8 = 4.718592` GF ordinary arithmetic, reductions/rsqrt plus explicit bf16 roundings; reads raw packed QK and writes rotated QK | Recomputes rstd from already-read raw QK, applies transposed rotation with bf16 rounding, reduces gradient statistic and computes dx | **DEMONSTRATED:** low AI; no normalized tensor or rstd stash; exact ordinary-op count depends on reduction/FMA convention |
| 18 RMSNorms | Square/reduce/rsqrt/scale; approximately 2.654208 GF under `kernel_classes` convention; writes fp32 rstd per row | Saved input/rstd plus upstream gradient, reduction and dx | **DEMONSTRATED:** low AI; 6.912 MB fp32 rstd outputs total; backward implementation is ATen-owned |
| Residual + U-net, 20 addcmuls | `out + state×lambda`; 40ND arithmetic operations, three bf16 streams per invocation | Direct gradient path, scaled gradient, lambda-product reduction, shared-input accumulation | **DEMONSTRATED:** bandwidth/reduction candidate; lambda-gradient work is not free |
| x0, eight addcmuls | Same operation over eight `[N,D]` tensors | Eight x0 contributions plus scalar-gradient reductions and accumulation into shared x0 | **DEMONSTRATED:** bandwidth/reduction candidate |
| Value residual, seven lerps | Subtract/multiply/add, bf16 inputs/output | Two scaled input gradients, scalar-gradient reduction, accumulation into first-layer value | **DEMONSTRATED:** bandwidth/reduction candidate |
| Post-lambdas/master casts | Small weight-space scaling/casts, not token-space scaling | Weight/scale gradients and fp32 gradient conversion | **DEMONSTRATED:** real but small compared with full activations |
| Head GELU | 98,304,000 bf16 elements, exact-erf GELU (`gelu("none")`) | Saved preactivation and upstream gradient; erf/exponential derivative | **DEMONSTRATED:** special-function/low-AI candidate, not counted in TFLOP ledger |
| Head geometry | Three casts, softplus, two sigmoids, three log1p chains, fp32 scale/divide/add/subtract | Corresponding fp32 elementwise derivatives and shared low/range/close accumulation | **DEMONSTRATED:** memory and special-function work |
| NLL/MSE/reductions | Four tanh/exp/precision/error/square chains; 12 dots; three full-mask reductions | NLL-only dot gradients, scale/square/geometry gradients; MSE explicitly no-grad | **DEMONSTRATED:** memory/reduction work; no MSE backward |
| Statistics and input preparation | fp32 differences/products/cumsums; fp32 price normalization, bf16 casts/cat; overlapping future-window copy | No gradient | **DEMONSTRATED:** real forward-only work missing from the global ledger |

**DEMONSTRATED:** There is no useful way to convert erf, exp, sigmoid, rsqrt, comparisons, reductions, explicit dtype roundings and all ATen backward implementations into a single exact “bf16 tensor-core FLOP” count from these Rust sources. Reporting them as zero executed arithmetic would be wrong; multiplying them by an arbitrary FLOP weight would be equally misleading.

### Attention dispatch and layout

**DEMONSTRATED:** The actual call is generic `Tensor::scaled_dot_product_attention`, `model.rs:1066-1083`, with bf16 Q/K/V, no explicit mask, dropout zero, `is_causal=true`, head dimension 64 and unit last-dimension stride. No backend-forcing call was found in the model or broader searched torch setup.

**HYPOTHESIS:** These are flash-compatible inputs and flash is a plausible selection. **Actual selection is not demonstrated by the call or the documentation’s “flash SDPA” wording.** The runtime dispatcher and installed PyTorch/CUDA kernel decide it.

**DEMONSTRATED:** Q/K/V are views; the source already avoids unconditional input `.contiguous()` copies. The output’s `.transpose(1,2).reshape(...)` can only be classified after inspecting the selected SDPA output strides. A logical `[B,heads,L,head_dim]` shape alone does not establish that its storage is contiguous in that order.

**DEMONSTRATED:** The standalone `attention output flatten` class creates a contiguous synthetic `[B,heads,L,head_dim]` input (`model.rs:1832-1840`), so it can charge/time a copy even if the real backend returns a token-major physical layout that permits a view. This is a live accounting risk, not permission to remove a required reshape blindly.

**HYPOTHESIS:** With a 128-token tile, 375 rounds to 384: only nine padded positions, 2.4% length overhead, or 4.86% square-area overhead before triangular scheduling. This is not intrinsically a disastrous sequence length. Other tile dimensions and diagonal tile handling can change the cost. Attention’s charged matrix work is only about **5.2%** of the total, so a small tail fix cannot plausibly explain a 2× whole-step gap.

## 4. Exact head size and loss reads

### Dense output tensor

**DEMONSTRATED:** The head emits `256×375×192×8 = 147,456,000` elements (`model.rs:1211-1235,2100-2137`).

| Representation | Exact bytes | Decimal GB | MiB |
|---|---:|---:|---:|
| bf16 raw head | **294,912,000** | **0.294912** | **281.25** |
| Hypothetical complete fp32 widening | **589,824,000** | **0.589824** | **562.5** |
| fp32 four-channel targets | **294,912,000** | **0.294912** | **281.25** |
| fp32 one-channel mask | **73,728,000** | **0.073728** | **70.3125** |

**DEMONSTRATED:** Training does **not** create the full fp32 output representation: `output()` is evaluation-only (`model.rs:2140-2150`); `losses()` promotes channels as needed. The raw tensor is already bf16. Channel-major storage and one `split` node also already remove the earlier stride-8 gather and multiple zero-padded backward buffers.

**DEMONSTRATED:** On the free-head ledger:

- Head output GEMM, forward + backward: **0.905969664 TFLOP**, **5.33%** of total.
- Raw head materialization charge: **1.769472 GB**, **1.234%** of total traffic.
- Geometry/targets/loss charge: **20.496384 GB**, **14.29%** of total traffic.
- Entire head, including covariate input/projection, head concat/hidden/GELU/output and geometry/loss, excluding final backbone norm: **1.528823808 TFLOP and 27.131904 GB**, approximately **9.00% and 18.92%**.
- The single bf16 raw output’s 0.294912 GB footprint is **0.206%** of the whole-step traffic ledger; footprint and repeated traffic must not be conflated.

### How many bytes do reductions read?

**DEMONSTRATED:** `losses`, `model.rs:2266-2302`, issues per channel:

1. `dot(square, weight)`—2U reads.
2. `dot(tanh, weighted_mask)`—2U reads.
3. No-grad `dot(square, raw_mask)`—2U reads.

For four channels that is **24U = 1,769,472,000 logical input bytes** read by the twelve large dots.

**DEMONSTRATED:** The prior’s horizon reduction, weighted-mask count and raw-mask count each read one U, adding **221,184,000 bytes**. Thus the large forward reductions read **1,990,656,000 logical input bytes**, plus tiny scalar/horizon buffers and implementation-specific reduction scratch. The NLL does not read a pre-materialized giant NLL tensor; it reduces its factors directly. These are logical reads: repeated mask reads can hit cache.

**DEMONSTRATED:** Price normalization, market-neutral target construction, geometry and loss currently use fp32 intentionally (`model.rs:1536-1544,2165-2174,2227-2256`). Prices/targets must remain fp32. No source evidence establishes that demoting geometry, predictive-scale transformations or loss accumulation to bf16 is numerically harmless, and doing so would not satisfy bit-exact acceptance anyway.

## 5. Where the time could go

**DEMONSTRATED:** The eight FFN up/down pairs contribute **9.663676416 TFLOP**, approximately **56.9%** of charged matrix work. QKV contributes **3.623878656 TFLOP**, another **21.3%**. Together they account for approximately **78.2%**. The eight layers’ materialization term alone is **103.809024 GB**, 72.4% of the traffic ledger, before value-residual mixing.

**DEMONSTRATED:** At supplied nominal peaks:

- `16.9846 TF / 209 TF/s ≈ 81.3 ms`.
- `143.4255 GB / 1.79 TB/s ≈ 80.1 ms`.
- Their near equality is simply the ledger’s aggregate AI, **118.42 FLOP/B**, being almost the nominal machine balance, **116.76 FLOP/B**.

**HYPOTHESIS:** Sequential phases limited by different resources can make both global fractions approximately one-half even when individual phases are efficient. This is unsurprising once the model’s aggregate AI happens to match machine balance. It is not evidence of a mysterious hardware mode.

**DEMONSTRATED:** Adding the two global lower-bound times is also not a valid prediction: GEMMs perform arithmetic and move data concurrently, and the byte ledger is not measured traffic. The correct model is a sum of **per-phase** roofline times, plus reductions/layout/optimizer effects, using actual kernel work and measured effective ceilings.

**DEMONSTRATED:** The full training step also includes the optimizer. Five-step Polar Express performs three matrix products per iteration (`optim/muon.rs:701-773`), on each block’s four matrices. With the default full-matrix layout, its matrix arithmetic is:

`8 layers × 5 iterations × Σq∈{1536,512,2048,2048} (4×512²×q + 2×512³)`

= **0.30064771072 TFLOP/step**, omitted from `step_cost`. The corresponding three-product logical bf16 operand/output traffic is approximately **1.76160768 GB**, before momentum, normalization, casts, state updates and packing. Nominal matrix-only compute time is 1.44 ms, but these smaller GEMMs need their own measured efficiency. Optimizer time is a separate phase, not unexplained head time.

**HYPOTHESIS:** The likely practical ranking is FFN/QKV GEMMs first, then the collection of backbone low-intensity forward/backward kernels, followed by head/loss and attention/optimizer. Exact milliseconds and ordering between the latter groups require corrected captured-step evidence. No source-only argument can establish that the head consumes the largest elapsed-time share.

## 6. Structural lever A: shorten the horizon, not just its weight

**DEMONSTRATED:** `cutoff:32` constructs a zero-weight band but still computes all 192 horizons. `head`, `targets`, geometry, scales, diagnostic MSE and backward still have the full shapes (`model.rs:699-710,2072-2137,2240-2302`). Zero loss weight is not an execution mask.

### Free 192-horizon head → free 32-horizon head

**DEMONSTRATED:** Exact reductions under current `step_cost`:

| Removed work | TFLOP saved | GB saved |
|---|---:|---:|
| Output GEMM’s removed 1,280 output columns | **0.754974720** | — |
| Covariate GEMM’s removed 960 input columns | **0.141557760** charged | — |
| Geometry/target/loss ledger, five-sixths | — | **17.080320** |
| Raw bf16 head materialization ledger, five-sixths | — | **1.474560** |
| Known-future window materialization ledger, five-sixths | — | **1.105920** |
| **Total ledger reduction** | **0.896532480** | **19.660800** |

**DEMONSTRATED:** This is **5.2785% of matrix FLOPs and 13.708% of traffic**, not merely the cited 4.45% and 11.91%. The cited 0.755 TFLOP/17.08 GB numbers describe only selected parts of the waste; shortening `pred_len` also shortens covariate inputs and raw output/window storage.

**DEMONSTRATED:** Correcting the nonexistent covariate input-gradient GEMM gives **0.849346560 TFLOP of removed actual projection GEMMs**: 0.754974720 output + 0.094371840 covariates. Byte savings remain accounting estimates, particularly given the stale loss-target inventory.

**DEMONSTRATED:** Parameters removed:

- Output rows: `1280×(1024+1) = 1,312,000`.
- Covariate weights: `960×256 = 245,760`.
- Total: **1,557,760 parameters**.
- Free model: **27,954,483 → 26,396,723** parameters.
- New raw output: **24,576,000 elements**, **49,152,000 bf16 bytes**, **98,304,000 fp32 bytes**.
- New ledger: **16.088039424 TFLOP**, **123.764736 GB**.

### Relative to the supplied basis:8:8 parameter count

**DEMONSTRATED:** There are two distinct changes, which must not be conflated:

1. **Keep `basis:8:8`, change pred_len 192→32:** output parameter width 832→192. Remove `640×1025 + 960×256 = 901,760` parameters: **27,232,883 → 26,331,123**. Including smaller expansion work, ledger savings are **0.9015656448 TFLOP and 19.67128576 GB**. Its basis timescales change too.
2. **Use free means with pred_len 32:** **27,232,883 → 26,396,723**, removing **836,160** parameters. Compared with the original basis ledger, save **0.901766971392 TFLOP and 19.673382912 GB**, including deletion of the expansion. This also changes the mean parameterization at horizons 9–32.

**HYPOTHESIS — time prediction:** **15–25 ms/step saved** is a defensible initial budget for 192→32 at batch 256, with unchanged backbone, once compared under the corrected captured path. The removed projection compute has a nominal floor around 4.1 ms; removed logical traffic has an 11 ms nominal-bandwidth scale, but these are overlapping components and the old loss ledger undercounts some work. Do not promise 25 ms or multiply the whole-step 178 ms by either percentage as if all phases had identical utilization.

**DEMONSTRATED:** This is **not bit-identical waste removal**. Loss normalization, diagnostic population, covariate input width, output GEMM shape, parameters, initialization and often sampling/eligibility change. Even retained predictions can follow a different trajectory. The supplied observation that skill is confined to h≤32 supports testing this structural change; it does not prove trajectory equivalence.

## 7. Structural lever B: dense origins

**DEMONSTRATED:** Every row computes all 375 backbone tokens. `last_only` narrows **after** the final backbone norm (`model.rs:1640-1648`), then restricts head/targets. Thus selecting fewer scored origins without changing the backbone saves head work, **not** the causal context computation needed to represent those origins.

**DEMONSTRATED:** For s scored origins per row, retaining the current full backbone, the free-head ledger is approximately:

- Matrix work: `15.455748096 + (s/375)×1.528823808 TFLOP`.
- Traffic: `116.293632 + (s/375)×27.131904 GB`.

Small head setup/basis work, statistics and uncounted traffic do not follow this formula exactly. Parameters remain unchanged.

**DEMONSTRATED:** Scoring only 32 origins/row would therefore remove about **1.398 TFLOP and 24.817 GB** from that ledger, but also remove **91.47% of supervised origin positions** at the same 256 rows. A one-origin head leaves approximately 91% of matrix work and 81% of ledger traffic, not 1/375 of a step.

**DEMONSTRATED:** Restoring the same number of scored origin targets by increasing rows multiplies full-backbone work by `375/s`. Head work per raw scored origin remains approximately constant; backbone work per scored origin becomes worse. At s=32 it requires **11.71875×** as many rows for the same raw origin count. This does not fit the existing memory budget by assumption, and chunking/accumulation are excluded.

**HYPOTHESIS:** Fewer, more independent rows/origins could still offer more useful gradient information per target because adjacent dense targets overlap and are correlated. Source cannot determine that effective sample-size benefit. It would need to outweigh the large loss of amortized backbone reuse. The source-derived cost argument favors **dense origins per unit of raw supervision**, not sparse origins with more rows.

**DEMONSTRATED:** Reducing attention sequence length itself is another change: increasing patch length changes the input representation; shortening context changes available information. Neither is equivalent to narrowing scored origins, and neither is a bit-identical optimization.

## 8. Ranked action list

Time ranges are **HYPOTHESIS**, conditional on the actual kernel accounting being repaired and the same captured batch-256 workload being compared. Traffic savings are logical stream counts or explicitly identified ledger deltas—not measured HBM savings. Candidate ranges overlap and must not be added blindly.

| Rank | Action and source evidence | Parameters / FLOPs / traffic cost | Predicted ms saved | Training bits | Cheapest confirming experiment |
|---|---|---|---:|---|---|
| 0 | **Repair attribution before choosing a kernel.** `step_cost:615-761`; `kernel_classes:1734-2067`; LoaderPerf’s owned benchmark correction | **DEMONSTRATED:** Δparameters=0, Δexecuted FLOPs=0, Δtraffic=0; fixes interpretation only | **0** | Unchanged | Corrected captured 192-horizon benchmark, actual backend/strides and complete gradient targets, existing report bases |
| 1 structural | **`pred_len 32`**, rather than `cutoff:32` | **DEMONSTRATED:** free-model −1,557,760 parameters; −0.84934656 TF actual projection work; −19.6608 GB current ledger; basis alternatives above | **15–25** | **Changes model/trajectory** | Captured 192 versus 32 with identical batch size/config and same measured-step count; retain held-out h≤32 skill criterion |
| 1 free candidate | **Fuse fp32 head geometry/pointwise NLL chains while preserving ATen rounding and existing reduction order.** `model.rs:2240-2302` | **HYPOTHESIS:** Δparameters=0; matrix FLOPs unchanged; pointwise arithmetic approximately unchanged; target **6–12 GB** fewer intermediate streams, possibly more with proven backward fusion | **4–12** | Bit-identical **only after** exact fwd+bwd acceptance; not yet demonstrated | First fuse one channel’s scale→tanh→exp→precision path, keep dot reductions unchanged, compare exact output and input/parameter-gradient bytes; then captured whole-step A/B |
| 2 free candidate | **FFN GEMM epilogue fusion with ReLU².** `model.rs:1110-1120`; `kernels.cu:47-115` | **HYPOTHESIS:** Δparameters=0; matrix FLOPs unchanged; remove the separate activation input read, **3.145728 GB/step** forward across eight layers while retaining preactivation for backward. Deleting both stash and read is not justified | **2–4** | Candidate must round GEMM result to bf16 before square, reproduce NaN behavior, and keep exact backward; arbitrary fused MMA epilogue can change bits | Exact forward/backward comparison of fused up-projection+activation against current linear+fused ReLU², followed by captured A/B |
| 3 free candidate | **Combine adjacent attention-residual/x0 elementwise passes.** `model.rs:1084-1090` | **HYPOTHESIS:** Δparameters=0; arithmetic unchanged; removing one intermediate write/read across eight layers saves **1.572864 GB** forward; backward may save more only if proven | **1–3** | Must preserve intermediate bf16 rounding and scalar-gradient reduction order | Exact composed two-addcmul versus candidate fwd/bwd at real dimensions; captured A/B |
| 4 free candidate | **Avoid large forward target temporaries without demoting fp32.** `model.rs:2155-2174` | **HYPOTHESIS:** Δparameters=0; arithmetic unchanged; fusing target subtraction/division removes two fp32 four-channel temporary write/read pairs, **1.179648 GB**, while preserving operation order | **0.7–1.5** | Candidate can be bit-identical; no target gradient exists | Exact target/mask bytes across real boundary/validity cases, then captured A/B; coordinate with LoaderPerf if moving work into its data path |
| 5 conditional | **Remove an attention flatten copy only if the actual selected backend really makes it.** `model.rs:1066-1083,1832-1840` | **HYPOTHESIS:** Δparameters=0; FLOPs unchanged; possible forward copy cost **1.572864 GB** across eight layers; zero saving if current reshape is already a view | **0–2** | Exact layout-only change may be free; backend replacement is not automatically exact | Record actual SDPA backend, output strides and copy kernels within the existing report system before changing code |
| 6 conditional | **Reduce duplicate loss-reduction passes**, retaining exact reduction trees | **HYPOTHESIS:** Δparameters=0; near-zero matrix Δ; sharing objective-count/prior mask input could save **0.073728 GB**; sharing two squared-error reduction reads could save up to **0.294912 GB**; reduction workspace/trees may offset it | **0–0.5** | High exactness risk: multi-output reductions commonly change summation order | Bitwise loss and all-gradient comparison with the current dot/sum implementation; reject on any discrepancy |
| 7 diagnostic, not first optimization | **Flash tiling/backend tuning at L=375** | **HYPOTHESIS:** Δparameters=0; useful FLOPs unchanged; executed padding/recompute/traffic change depends on selected kernel; simple 375→384 tail is small | **0–1** for tail-only improvement | Alternative attention kernels can change bits; exact comparison required | Identify selected kernel and actual attention device time before an exact candidate A/B |
| 8 structural | **Score fewer origins**, keeping context/backbone | **DEMONSTRATED:** Δparameters=0; at s=32 approximately −1.398 TFLOP and −24.817 GB ledger, but −91.47% raw supervised origins | **15–30** at fixed rows, **not** a prediction of cheaper learning | **Changes trajectory and information budget** | Fixed wall-clock held-out h≤32 comparison, report scored origins/rows and quality; reject “same gradient signal” assumption |

### Existing optimizations that must not be proposed again

**DEMONSTRATED:** The following waste is already removed:

- Separate ReLU then square: current forward/backward uses `fused_kernels::relu_square`.
- Materialized normalized QK and saved rstd: `qk_norm_rope` recomputes the cheap statistic from input already being streamed.
- Unconditional contiguous Q/K/V input copies: the actual model passes strided views.
- Full fp32 widening of the raw eight-channel head: training promotes channels at consumption.
- Multiple `narrow` gradient scatters for raw head/QKV splitting: actual paths use a single split node.
- Token-space output/post-lambda scaling: scale is folded into weights.
- Cast-after-cat patch construction and cast-after-window-gather future covariates: current code casts first.

**DEMONSTRATED:** No additional fp32→bf16 demotion can presently be labeled free. Master weights, optimizer state, prices, targets, causal statistics and fp32 loss arithmetic are not disposable dtype mistakes.

## 9. Confirming commands and evidence contract

**DEMONSTRATED:** The launcher’s command is `benchmark-timexer-segment`; `trading_bots/run-release-cuda.sh` delegates to `./torch-env.sh cargo run --release -p trading_bot_0`. Existing report bases are registered in `shared/src/report.rs:72-76`.

**HYPOTHESIS / proposed experiments:** After LoaderPerf’s benchmark correction is complete, the queue owner can run these commands under its own scheduling. They were **not executed** here; no mlq submission or GPU lease was taken.

```bash
# Captured production-shaped free-head baseline, after the harness fix.
./torch-env.sh cargo run --release -p trading_bot_0 -- \
  benchmark-timexer-segment \
  --output training/runs/device-inventory-h192/gens/0 \
  --batch-size 256 --seq-len 6000 --patch-len 16 \
  --pred-len 192 --layers 8 --d-model 512 --heads 8 --ffn 2048 \
  --features all --horizon-mean free --horizon-loss cutoff:32 \
  --optimizer polar-express --warmup 5 --steps 20 --profile

# Structural horizon comparison; same batch/context/backbone and capture.
./torch-env.sh cargo run --release -p trading_bot_0 -- \
  benchmark-timexer-segment \
  --output training/runs/device-inventory-h32/gens/0 \
  --batch-size 256 --seq-len 6000 --patch-len 16 \
  --pred-len 32 --layers 8 --d-model 512 --heads 8 --ffn 2048 \
  --features all --horizon-mean free --horizon-loss uniform \
  --optimizer polar-express --warmup 5 --steps 20 --profile
```

**DEMONSTRATED:** These are throughput experiments on synthetic resident data, not accuracy evidence. For the exact basis configuration replace `--horizon-mean free` with `--horizon-mean basis:8:8`; keep the interpretation of changed parameterization explicit.

**HYPOTHESIS / proposed acceptance commands:** For an implemented candidate, retain only the scoped permitted checks and have the queue owner run CUDA-dependent tests:

```bash
./torch-env.sh cargo check -p trading_bot_0 --tests
./torch-env.sh cargo test -p trading_bot_0 timexer_segment
./torch-env.sh cargo test -p fused_kernels
```

**DEMONSTRATED:** Current `--profile` is not sufficient on its own to prove a candidate free: its inspected implementation differentiates only the first activation input, its residual class omits lambda parameters, and its SDPA class has synthetic layout/slicing differences. These concerns were sent to LoaderPerf. Any replacement must measure the actual composed training operator and **every** activation/parameter gradient, not just a convenient standalone forward.

**HYPOTHESIS / proposed diagnostic scope:** Backend identity, actual strides, head/loss timing, graph-state validity, fwd/bwd kernel counts and corrected cost numerators should be additional series in the existing registered `.report.bin` bases—or a newly registered base with its bidirectional registry test—not ad hoc CSV/log-only metrics. Captured throughput must remain distinct from instrumented kernel attribution; the graph must not be disabled to manufacture the throughput comparison.

## 10. Verification and limits

**DEMONSTRATED:** This investigation was read-only: source/registry/launcher inspection, broad targeted code searches, exact shape-ledger reconciliation and peer coordination. No files were written; no builds, tests, GPU programs, queue submissions or leases were run.

**DEMONSTRATED:** Saved benchmark `.report.bin` files were located under `training/runs/benchmark-fused-qknorm-capture/gens/0`, but the available reader cannot decode their binary payload; no timing values were invented from them. More importantly, historical benchmark attribution has the concrete harness mismatches described above.

**DEMONSTRATED:** The remaining missing evidence is **a corrected captured-step timing/actual-kernel attribution and runtime SDPA backend/stride observation**, not further host-loader speculation. The source findings already invalidate treating the quoted aggregate fractions as two independent physical utilization measurements and identify the largest defensible waste-removal and structural experiments.