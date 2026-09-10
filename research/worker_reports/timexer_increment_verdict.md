# `--horizon-mean increment:8` — REFUTED (job 5558, step-matched, 2026-09-08)

Written by Main. Pre-registration is at `timexer_ceiling_placement_20260908.md` §9.

## What ran

`train-timexer-segment --run timexer-increment8-2500 --horizon-mean increment:8 --max-steps 2500`,
binary `/var/tmp/tb0_inc8b`, single variable against `timexer-control-4k` (`--horizon-mean free`),
every other flag identical. -753,664 params (-2.70%), 16.53 TFLOP/step.

Reports for steps 1000 and 2000 landed; the run then **OOM'd inside the final
`held-out full` pass** (318 MiB refused with foreign tenants live), so this arm has no
full-split row. Every number below is `held-out sample` and `held-out cross-section`,
step-matched, which is what the pre-registration was written against.

## Pre-registered falsifier: technically passed, and it does not matter

> "h=192 market-neutral MSE ratio must fall below 1.00 (control 1.0245 — worse than
> forecasting zero). ≥ 1.01 refutes the arm."

Measured **0.99991** at step 2000. It clears 1.00 by 9e-5, i.e. the long end is now
*exactly persistence* rather than worse than persistence. The pathology is gone because the
long end stopped predicting, not because it started forecasting.

## What it cost, at the horizons that trade

`held-out sample` market-neutral four-channel MSE ratio, step 2000 (below 1.0 = better):

| h | control (`free`) | `increment:8` | skill margin |
|---|---|---|---|
| 1 | **0.95528** | 0.97117 | 4.47% -> **2.88%** |
| 8 | **0.95066** | 0.98731 | 4.93% -> **1.27%** |
| 64 | 0.98550 | 0.99833 | 1.45% -> 0.17% |
| 192 | 1.02446 | **0.99991** | -2.45% -> 0.01% |

`held-out cross-section` close cross-sectional IC, step 2000: h=1 **-0.0285**, h=8 **-0.0180**,
h=192 +0.0277. The reference arms on the same draw read h=1 +0.1429/+0.1575 at 1000/2000
(`timexer-invsqrt-ic-6k`) and +0.1621 on the anchored draw (job 5483). **The tradable ranking
signal is not merely weaker, it changes sign.**

## Verdict

REJECT. This is the fifth horizon-axis intervention with the same shape: `cutoff:32`,
`basis:8:8`, `inv`, `inv-sqrt` and now `increment:8` each bought long-end "stability" by
suppressing amplitude and each paid for it at h=1..8. It is the outcome `FutureTeacher`'s
flat ceiling predicted — under a ceiling that is constant in h at `c_1`, no reweighting or
reparametrization of the horizon axis can find horizon-specific information, because none
exists. The remaining lever is per-bar information, not horizon geometry.

## Byproduct, kept: `--eval-batch-size` (default 64)

Seven arms across two days died of CUDA OOM **inside an evaluation pass, never inside
training**. Cause: evaluation allocates from what is left of the card while the captured
training step still owns its 17.3 GiB private mempool, and a 256-row scoring pass peaks at
2,690 MiB measured (job 5556). Evaluation now has its own batch size, independent of
`--batch-size`, and every training-time pass (`score` at the sample, cross-section,
in-period and final-full sites) calls `empty_cache()` first so the allocator's training
history is not counted against the evaluation's requirement. The graph's private mempool is
owned by the capture and is not released. The training batch is untouched: the loss, the
optimizer step and the captured graph never see `eval_batch_size`.

Effect on this arm: both interval evaluations passed at eval batch 64 on a card where the
same arm had OOM'd at step ~1000 in nine consecutive attempts.

## Next arm, pre-registered before it exists: `--row-stride-multiple 30` (job 5573)

2,500 steps, single variable against the same control. `PeakCause` proved random thinning is
occupancy-invariant, so K=30 does NOT change how many times a distinct supervised outcome is
consumed by step 2,000 - it changes only whether those repeats arrive under 30 overlapping
contexts or under ~1. That makes it the last live mechanism for the step-2,000 peak:
memorization of a persistent factor realization seen from many windows.

PREDICTION (Main, before the run): the peak stays at ~2,000 optimizer steps and the
step-matched `held-out sample` objective NLL and per-horizon MSE ratios land within the
A-A' noise of the control, because exposure count is unchanged and only context diversity
moves. A peak that moves LATER than step 2,500, or a step-2,000 h=8 MSE ratio below 0.945,
refutes that and makes context diversity the lever. A peak that moves EARLIER refutes the
"diverse contexts are harmless" reading in the opposite direction.
