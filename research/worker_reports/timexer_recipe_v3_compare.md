# Recipe v3 vs baseline — step-matched comparison

**Verdict: CULL.** At matched step 4000 on the identical held-out sample (persistence NLL
2.3947663 in both runs), recipe held-out NLL is **2.0774** against baseline **1.9849**
(+0.0925 nats/bar), and the recipe has worsened at every eval since its best (2k 1.9999 → 3k
2.0425 → 4k 2.0774) while the baseline improved over the same steps (1.9894 → 1.9900 → 1.9849).
Even the recipe's best step (2k, 1.9999) is worse than the baseline at that same step (1.9894).

Runs: BASELINE `timexer-market-neutral-20260906` (final step 9000, early-stopped), RECIPE
`timexer-recipe-v3-20260906` (job 5188, running; evals at 1k, 2k, 3k, 4k exist at time of
reading). Every number below names its run and step. Baseline bases are the pre-clarity names
(`_validation`, `_absolute`, `_robust`); label mapping: baseline `preview` = recipe `held-out
sample`; baseline `preview relative MSE ratio` = recipe `market-neutral MSE ratio`; baseline
`preview absolute MSE ratio` = recipe `raw MSE ratio`. Both runs consume identical data per step
(`training target bars completed` = 0.104259826 at step 1000 in both).

## 1. Step-matched table (steps both runs have: 1k, 2k, 3k, 4k)

### Held-out NLL (nats/bar; persistence NLL = 2.3947663 at every step in both runs)

| step | baseline | recipe | Δ (recipe − baseline) |
| --- | --- | --- | --- |
| 1000 | 2.0276 | 2.0039 | −0.0237 |
| 2000 | 1.9894 | 1.9999 | +0.0105 |
| 3000 | 1.9900 | 2.0425 | +0.0525 |
| 4000 | 1.9849 | 2.0774 | +0.0925 |

### Training NLL and train − held-out gap

| step | baseline train | recipe train | Δ | baseline gap | recipe gap |
| --- | --- | --- | --- | --- | --- |
| 1000 | 2.2566 | 2.2534 | −0.0032 | 0.2290 | 0.2495 |
| 2000 | 2.2413 | 2.2174 | −0.0239 | 0.2519 | 0.2175 |
| 3000 | 2.2427 | 2.2035 | −0.0392 | 0.2527 | 0.1611 |
| 4000 | 2.2313 | 2.1535 | −0.0778 | 0.2464 | 0.0761 |

The recipe's training NLL falls 0.078 below the baseline's by 4k while its held-out NLL rises
0.093 above — and this is inside epoch 1 (41.7% of training targets consumed at 4k), so no
batch has been seen twice. The recipe is fitting structure that holds in the training period and
not in the held-out period.

### Aggregate MSE ratios (dimensionless; < 1 beats persistence)

| step | MN base | MN recipe | Δ | raw base | raw recipe | Δ | median-window base | median-window recipe | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.9841 | 0.9917 | +0.0076 | 0.9914 | 0.9977 | +0.0063 | 0.9975 | 0.9982 | +0.0007 |
| 2000 | 0.9762 | 1.0121 | +0.0359 | 0.9869 | 1.0180 | +0.0311 | 0.9949 | 1.0138 | +0.0189 |
| 3000 | 0.9884 | 1.0333 | +0.0449 | 0.9937 | 1.0399 | +0.0462 | 0.9935 | 1.0416 | +0.0481 |
| 4000 | 0.9748 | 1.0223 | +0.0475 | 0.9835 | 1.0246 | +0.0411 | 0.9973 | 1.0459 | +0.0486 |

Levels (σ-scaled MSE, held-out MN, persistence 45.389 in both): baseline 44.667 / 44.310 / 44.861 /
44.244; recipe 45.011 / 45.940 / 46.902 / 46.403 at 1k/2k/3k/4k. The recipe has lost to
persistence on aggregate at every eval from 2k on; the baseline never did before step 9000.

### Calibration (fraction within ±1σ, nominal 0.6827; within ±1.96σ, nominal 0.950)

| step | 1σ base | 1σ recipe | Δ | 1.96σ base | 1.96σ recipe | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| 1000 | 0.7359 | 0.7062 | −0.0297 | 0.9459 | 0.9385 | −0.0074 |
| 2000 | 0.7610 | 0.7187 | −0.0423 | 0.9600 | 0.9494 | −0.0106 |
| 3000 | 0.7218 | 0.6584 | −0.0634 | 0.9453 | 0.9186 | −0.0267 |
| 4000 | 0.7434 | 0.6299 | −0.1135 | 0.9560 | 0.8980 | −0.0580 |

Baseline stays over-covered (σ too wide, 0.71–0.76) throughout. The recipe crosses from
over-covered to under-covered between 2k and 3k and is at 0.630/0.898 by 4k — σ is now too
narrow, and shrinking every eval.

### Tail share of held-out squared error (`_progress`)

baseline 0.3046 / 0.3005 / 0.3047 / 0.3000; recipe 0.3097 / 0.2995 / 0.2871 / 0.2685 (1k–4k).
The recipe's tail share is falling while total error rises: bulk (non-tail) error rose from
45.01×0.690 = 31.1 at 1k to 46.40×0.732 = 34.0 at 4k (+9%), tail error fell 13.9 → 12.5.
The damage is in ordinary bars, which matches the median-window ratio being the worst of the
three aggregate ratios (1.0459 at 4k).

### Per-horizon ratio curve — NOT step-matched (report is a latest-snapshot overwrite)

`timexer_segment_horizon*` is rewritten each eval with the step in its title. Baseline's
snapshot title reads `TimeXer epoch 1 step 9000` (its final, worst eval; aggregate MN 1.0000);
recipe's reads `CausalPatch epoch 1 step 4000` (aggregate MN 1.0223). No matched-step
per-horizon data exists in the reports. The only step-matched baseline per-horizon points are
the context-provided step-5000 values (h=1 0.938, h=192 0.993), which were not re-read here.

| h | MN base@9k | MN recipe@4k | raw base@9k | raw recipe@4k | trimmed base@9k | trimmed recipe@4k | MAE base@9k | MAE recipe@4k | up-share base@9k | up-share recipe@4k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.9481 | 0.9529 | 0.9500 | 0.9545 | 0.8458 | 0.8452 | 0.9202 | 0.9206 | 0.585 | 0.551 |
| 2 | 0.9239 | 0.9287 | 0.9323 | 0.9350 | 0.8951 | 0.9020 | 0.9524 | 0.9547 | 0.627 | 0.691 |
| 4 | 0.9434 | 0.9340 | 0.9536 | 0.9447 | 0.9290 | 0.9330 | 0.9693 | 0.9716 | 0.592 | 0.584 |
| 8 | 0.9179 | 0.8940 | 0.9328 | 0.9119 | 0.9529 | 0.9553 | 0.9800 | 0.9839 | 0.534 | 0.637 |
| 16 | 0.9258 | 0.9125 | 0.9488 | 0.9409 | 0.9897 | 1.0041 | 0.9920 | 0.9976 | 0.423 | 0.611 |
| 32 | 0.9456 | 0.9462 | 0.9666 | 0.9694 | 0.9917 | 1.0001 | 0.9996 | 1.0021 | 0.445 | 0.565 |
| 64 | 0.9734 | 1.0146 | 0.9792 | 1.0110 | 1.0050 | 1.0720 | 1.0101 | 1.0357 | 0.487 | 0.528 |
| 128 | 1.0102 | 1.0340 | 1.0176 | 1.0353 | 1.0439 | 1.0940 | 1.0249 | 1.0393 | 0.498 | 0.526 |
| 192 | 1.0141 | 1.0329 | 1.0143 | 1.0281 | 1.0395 | 1.0958 | 1.0164 | 1.0315 | 0.504 | 0.536 |

First horizon with MN ratio > 1: baseline@9k h=114; recipe@4k h=58. Win rate vs persistence at
h=64/128/192: baseline@9k 0.463/0.466/0.472, recipe@4k 0.456/0.455/0.463.

## 2. Where the recipe is worse, and mean vs σ

DEMONSTRATED (recipe@4k vs baseline@9k, with the step caveat above):
- **Short horizons (h ≤ 32): not worse.** h=1 MN 0.9529 vs 0.9481, trimmed 0.8452 vs 0.8458,
  MAE 0.9206 vs 0.9202 — equal within noise; h=8 and h=16 are better (0.894 vs 0.918, 0.9125 vs
  0.926). Against the context's baseline@5k h=1 0.938 the recipe@4k h=1 is 0.015 worse.
- **Long horizons (h ≥ 64): clearly worse.** h=64 1.0146 vs 0.9734, h=192 1.0329 vs 1.0141 —
  worse even than the baseline's own worst snapshot, and +0.040 at h=192 against the context's
  baseline@5k 0.993. Because persistence MSE grows with h, h ≥ 64 dominates the aggregate; the
  aggregate ratio gap (+0.0475 at 4k) is a long-horizon phenomenon — the baseline's known
  failure mode (training-period drift), arriving 2–3× earlier and larger.
- **Mean or σ?** Both, in sequence. 1k→3k: NLL +0.0386 with MN ratio +0.0417 and 1σ coverage
  0.706→0.658 — mean and σ both degrade. 3k→4k: NLL +0.0349 while MN ratio *improved*
  1.0333→1.0223 and 1σ coverage fell 0.658→0.630 — a pure σ problem (overconfident spread).
- **What kind of mean error** (recipe@4k `_decomposition`, close channel, MN, share of
  persistence MSE): the unconditional-offset term is ≈0 at every horizon (−0.0003 at h=1,
  −0.0018 at h=64, −0.0039 at h=192) — it is *not* a constant tilt, even though the mean
  predicted close coordinate grows to +0.68σ at h=192 vs realized +0.10 (`_offset`). The
  demeaned forecast has real skill at all horizons (implied 1−ρ² gain 0.020 at h=1, 0.055 at
  h=8, 0.013–0.028 at h=64–192), but the **mis-scaling cross term** is −0.0127 at h=1, ≈−0.001
  at h=8/16, then −0.019 at h=32, −0.056 at h=64, −0.052 at h=128, −0.058 at h=192. The
  long-horizon loss is forecast *amplitude* — the conditional mean is 2–4× larger than its
  predictive power justifies (β̂ ≪ 1). Combined with σ too narrow, the picture is uniform
  overconfidence. For h=1 only, the eval-bisect report gives the baseline best checkpoint
  (step 6000) offset −1.3e-5 / demeaned 0.0209 / cross −0.0067: same demeaned skill as the
  recipe (0.0200), twice the mis-scaling loss in the recipe (−0.0127). Not step-matched.

## 3. Recipe scalars (`timexer_segment_recipe_scalars`, steps 1k / 2k / 3k / 4k)

Inits from the run context: residual λ 1.0488, x0 λ 0.0, skip logits −1.5 (sigmoid 0.182),
value λ 0.5. Post-λ init is not stated in the context or in any report.

| family | init | 1k | 2k | 3k | 4k | travel |
| --- | --- | --- | --- | --- | --- | --- |
| residual attn L0 | 1.0488 | 3.846 | 4.270 | 4.314 | 4.254 | +2.8 in 1k, then plateau (Δ/1k: +0.42, +0.04, −0.06) |
| residual attn L1–L7 | 1.0488 | 0.72–1.14 | 0.50–0.67 | 0.44–0.73 | 0.53–0.85 | fell below init by 1k, settled by 2k; L4/L5 drifting up (+0.09–0.11/1k) |
| residual ffn L0 | 1.0488 | 2.268 | 2.289 | 1.921 | 1.733 | rose then falling −0.19/1k |
| residual ffn L1–L6 | 1.0488 | 1.26–1.72 | 0.80–1.51 | 0.66–1.39 | 0.58–1.24 | still falling 0.03–0.20/1k |
| residual ffn L7 | 1.0488 | 1.294 | 1.877 | 2.158 | 2.213 | rising, decelerating |
| post attn L0 | ? | 0.312 | 0.187 | 0.146 | 0.167 | collapsed to ~0.15 |
| post attn L1–L7 | ? | 1.18–1.51 | 1.26–1.72 | 1.30–1.68 | 1.30–1.73 | slow rise, decelerating (+0.01–0.06/1k) |
| post ffn L0–L6 | ? | 0.38–0.81 | 0.29–0.54 | 0.27–0.40 | 0.30–0.46 | fell, now flat/slightly up |
| post ffn L7 | ? | 0.997 | 0.916 | 0.877 | 0.880 | settled |
| x0 L0 | 0.0 | 2.797 | 3.221 | 3.266 | 3.205 | plateau ~3.2 |
| x0 L1 | 0.0 | 1.562 | 2.064 | 2.369 | 2.000 | reversed at 4k (−0.37) |
| x0 L2 | 0.0 | 1.107 | 1.518 | 1.858 | 0.816 | reversed at 4k (−1.04) |
| x0 L3 | 0.0 | 0.845 | 1.403 | 1.718 | 0.382 | reversed at 4k (−1.34) |
| x0 L4 | 0.0 | 0.603 | 1.030 | 1.326 | 0.553 | reversed at 4k (−0.77) |
| x0 L5 | 0.0 | 0.537 | 0.935 | 1.067 | 0.408 | reversed at 4k (−0.66) |
| x0 L6 | 0.0 | 0.409 | 0.931 | 1.140 | 0.939 | reversed at 4k (−0.20) |
| x0 L7 | 0.0 | 0.313 | 0.915 | 1.372 | 2.396 | accelerating up (+1.02 in the last 1k) |
| skip 3→4 | 0.182 | 0.129 | 0.077 | 0.055 | 0.040 | monotone to zero (self-pruned) |
| skip 2→5 / 1→6 | 0.182 | 0.222 / 0.255 | 0.228 / 0.230 | 0.226 / 0.263 | 0.238 / 0.231 | flat |
| skip 0→7 | 0.182 | 0.222 | 0.226 | 0.319 | 0.297 | jumped at 3k, holding |
| value L1–L7 | 0.5 | 0.32–0.73 | 0.45–0.84 | 0.46–0.86 | 0.42–0.86 | monotone in depth (L1 0.42 → L7 0.86), settled (<0.06/1k after 2k) |

Flags:
- **x0 λ are not settled — they are oscillating.** Five of eight reversed direction between 3k
  and 4k with swings of 0.66–1.34 in 1000 steps, and L7 jumped +1.02 in the opposite
  direction. This is the signature of an over-large step size on these parameters, not of
  convergence.
- **residual attn L0 = 4.25 plus x0 L0 = 3.21** against post attn L0 = 0.167 and post ffn L0 =
  0.315. [INFERENCE, assuming the block form x ← λ_res·x + λ_x0·x0 + λ_post·f(norm(x)) and
  that L0's input is x0] the L0 stream is ~7.5× the embedding while the L0 branches add ~0.17×
  and ~0.32×, i.e. block 0 is ~95% bypass. It has plateaued (not accelerating).
- skip 3→4 has been driven to 0.04 (the network is switching it off); value λ and the other
  skips are in sane ranges and settled.

## 4. Throughput / evaluation (not a confound)

| step | baseline ms/step (`training step`) | recipe ms/step (`training step (interval mean)`) | baseline eval ms | recipe eval ms |
| --- | --- | --- | --- | --- |
| 1000 | 301.9 | 175.5 | 1567.9 | 1080.2 |
| 2000 | 461.5 | 175.3 | 1164.9 | 946.8 |
| 3000 | 458.8 | 175.3 | 1533.3 | 1613.0 |
| 4000 | 345.5 | 175.2 | 655.7 | 1102.3 |

Recipe step time is flat at 175 ms (captured forward+backward replay 162.2–163.0 ms, optimizer
4.8–4.9 ms, H2D 8.0 ms); baseline is 217–461 ms and noisy. Recipe eval time is dominated by
host batch wait (278–900 ms of it). Wall clock (`_hardware` index): recipe hit 4k at ≈701 s,
baseline hit 9k at ≈3006 s. GPU busy 100% in both; VRAM 90.3% (recipe) vs 85.8% (baseline);
power-limit use 88% vs 86%. Nothing in timing or hardware explains the learning difference; the
recipe simply reaches each (worse) eval 1.7–2.6× sooner.

## 5. Verdict and ranked causes

**CULL.** Demonstrated: worse at 2k, 3k, 4k on NLL, all three aggregate ratios and both
calibration bands; monotone worsening for 2000 steps; the recipe's best step is worse than the
baseline's same step; train/held-out gap collapsing (0.25 → 0.08) inside epoch 1; decomposition
shows structural over-amplitude at h ≥ 32 and σ under-coverage — overconfidence, not noise. The
run will also stop itself: best 2k, patience 3, so a 5k eval that does not beat 1.9999 ends it
(~3 min away at 175 ms/step). Letting it reach 5k costs nothing and banks one more point; do
not restart the recipe as configured.

Ranked candidate causes, each with the diagnostic that separates it:

1. **5.0 lr multiplier on residual/x0/skip scalars → scalar instability.** Evidence: x0 λ
   swings of 0.66–1.34 per 1k steps and direction reversals at 4k; residual attn L0 at 4.25.
   Diagnostic: identical rerun with multiplier 1.0; confirmed if `recipe_scalars` x0 λ move
   < 0.2/1k and 4k held-out NLL ≤ 1.99. Within this run: the 5k scalars — another |Δ| > 0.5
   reversal on x0 L2–L5 confirms oscillation rather than a one-off.
2. **x0 embedding shortcut lets the net fit training-period structure (drift) fast.**
   Evidence: training NLL 2.1535 vs 2.2313 at 4k while held-out is worse; cross term −0.056 at
   h=64 (over-amplitude); predicted-up share 0.53–0.69 at all horizons (recipe@4k) vs
   0.42–0.59 (baseline@9k, unmatched). Diagnostic: ablation with x0 λ fixed at 0; confirmed if
   the train−held-out gap stays ≈0.25 like the baseline and the h=64 cross term stays ≥ −0.02
   at 3k. Within this run: `_decomposition` at 5k — cross term at h ≥ 64 growing more negative
   while the demeaned gain is flat = amplitude blow-up, not lost signal.
3. **σ collapse: overfit residual vs activation-scale artifact (ReLU² / gainless norms with no
   output-scale correction).** Evidence: 1σ coverage 0.719 → 0.658 → 0.630 across 2k–4k while
   training MSE falls 67.4 → 65.5. Diagnostic: score the 4k checkpoint's calibration on a
   training-period window sample. ≈0.68 on train and 0.63 held-out → σ learned the training
   residual (same cause as 2); < 0.68 on both → head/activation scale problem, fix with an
   output-scale or log-σ bias correction.
4. **Global effective-LR mismatch after removing norm gains/biases and switching to ReLU².**
   Evidence: the recipe led at 1k (2.0039 vs 2.0276) then overshot; every scalar family
   travelled more in the first 1k than in the next 3k. Diagnostic: 2×2 to step 2000 — base LR
   {1×, 0.5×} × scalar multiplier {5, 1}; judged on 2k NLL vs the baseline's 1.9894 and 1σ
   coverage ≥ 0.68. Both LR halvings fixing it → global; only the multiplier fixing it → cause 1.
5. **Value residual.** Value λ are sane and settled (0.42–0.86, monotone in depth). Unlikely
   primary; ablate last.
6. **UNet skips.** 3→4 self-pruned to 0.04, others flat 0.23–0.32. Benign; ablate last.

HYPOTHESIS vs DEMONSTRATED: everything in sections 1–4 is read from the reports at the named
steps. The causal attributions in section 5 are hypotheses; the block-form reading of the L0
scalars is marked [INFERENCE]. Per-horizon comparisons are baseline@9k vs recipe@4k and are
not step-matched.
