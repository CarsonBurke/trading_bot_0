# The ceiling under anchored placement — job 5516, and what it invalidates

Written by Main, morning of 2026-09-08, after threading `--placement` into `ceiling-timexer-segment`
(the handoff's item 1). One process, one checkpoint — `timexer-control-4k` step 2000, weights
`ccffa620654f1c95faa618501519e036249615e98d6034e425e3fed7c2309116` — scored twice, strided then
anchored, `--placement both`. Binary `/var/tmp/tb0_v19`. 326.7 s + 319.2 s + 16.8 s to re-place.

Everything below is DEMONSTRATED unless labelled otherwise.

## 1. The headline: the +0.1011 gap was a draw artifact, and its replacement is tautological

Paired within each placement, `held-out full`, h = 1:

| placement | ceiling `c₁` | student IC | gap | σ |
|---|---|---|---|---|
| strided | 0.16660 (assumed) | 0.06153 ± 0.00246 | **0.10507 ± 0.00246** | 42.7 |
| anchored | 0.16660 (assumed) | **0.16208 ± 0.00839** | **0.00452 ± 0.00839** | 0.54 |

The gap that has driven this project's direction for a session collapses from 42.7σ to 0.54σ under a
placement change at a fixed checkpoint. **But it must not be read as "the gap closed".** `c₁ = 0.1666`
was defined as *twice the measured h=1 IC*, and the measurement it doubled was the strided 0.0833. The
student moved 2.63×; the ceiling did not move because **it is not a measurement, it is an input that was
calibrated on the draw that just changed.**

Under the same convention applied to the anchored measurement, `c₁ = 2 × 0.16208 = 0.32416`, and the h=1
gap re-opens to ≈ 0.162. **Both readings are worthless**, because under "twice the measured" the h=1 gap
is `c₁ − student = student` by construction. That circularity was invisible while the two draws
disagreed by 2.63×; it is now written into `teacher::DEFAULT_ONE_BAR_CEILING_IC` so no successor can
quote the h=1 gap as evidence again. The h ≥ 8 gaps do not inherit the circularity — their ceilings
depend on `c₁` only through the measured variance ratio.

Landed with this result: `MEASURED_ONE_BAR_IC` 0.0833 → **0.16208** (SE 0.00839, 245 paired
cross-sections, provenance in the doc comment), `DEFAULT_ONE_BAR_CEILING_IC` 0.1666 → **0.32416**, and
the printed sensitivity sweep re-centred from `[0, .0833, .1666, .25]` onto the anchored bracket.
Job **5529** re-measures the anchored curve under the corrected `c₁` on `/var/tmp/tb0_v20`.

## 2. `TemporalSplit`'s precision claim is REFUTED in this direction — the trade is bias for variance

The anchored draw is **less** precise on the paired gap, not more:

| placement | paired cross-sections | h=1 student SE |
|---|---|---|
| strided | 9,506 | 0.00246 |
| anchored | **245** | **0.00839** |

3.4× **worse** standard error. The 4.7× precision gain was projected for the `held-out cross-section`
draw (width 40 → 256) and does not transfer to `held-out full`, where anchoring trades ~39× fewer
independent timestamps for ~157× wider cross-sections. That is still obviously the right trade — it
removes a 2.63× attenuation bias at h=1 — but the honest statement is **the strided draw was precise
about the wrong quantity**, and every threshold sized against 0.0164 or 0.0025 must be re-sized against
the anchored paired SE, which is 3-4× larger.

## 3. The MA(1) mechanism's four-decimal vindication was a coincidence of two distorted statistics

The strongest confirmation in yesterday's session was that two independent routes to `ρ₁` agreed:
plateau-implied `(Q−1)/2` and the directly measured lag-1 autocorrelation. On the training population:

| route | strided | anchored |
|---|---|---|
| plateau-implied `ρ₁` | −0.20694 | **−0.13976** |
| measured lag-1 at h=64 | −0.20669 | **−0.22698** |
| disagreement | **0.00025** | **0.08722** |

Agreement to four decimals under the defective draw; **62% disagreement under the correct one**. Both
statistics are within-timestamp cross-sectional second moments, so both were distorted by 10.6-name
cross-sections — in opposite directions, which is precisely how two wrong numbers can meet.

HYPOTHESIS (unresolved, and it is the next real question): the plateau fit is the fragile one. Its onset
moves h ≥ 5 → **h ≥ 55** and its level 0.58613 → 0.72048 on training, and the anchored training
`walk_ratio` at h=64 is **1.14463 > 1** — i.e. under proper placement the training targets are *more*
dispersed than √h at h=64, trending rather than reverting. The direct lag-1 measurement is comparatively
stable (−0.207 → −0.227). Falsifier: fit the plateau on the anchored draw with the onset FIXED at the
strided value and see whether the level or the onset carries the change.

## 4. The freezability verdict flips harder, in the direction already ruled

Three-population variance-ratio plateaus, pre-registered freezability bar ±0.05 spread:

| population | strided | anchored |
|---|---|---|
| training | 0.58613 (h≥5) | 0.72048 (h≥55) |
| calibration | 0.54751 (h≥9) | 0.56517 (h≥11) |
| held-out full | 0.52479 (h≥6) | 0.60187 (h≥4) |
| **spread** | **0.061** | **0.155** |

Strided already failed the bar marginally (0.061 vs 0.050). Anchored fails it by **3.1×**. The ruling
against a frozen aggregation constant, and for supervising `a(state)`, is unchanged in direction and
much stronger in evidence. Implied √h over-statement 1.306/1.351/1.380 → **1.178/1.330/1.289**.

## 5. The propagation arm's own gate is now straddled, so the arm stays blocked

`FutureTeacher`'s pre-registered gate: `ρ₁ ≤ −0.26` run; `−0.26 … −0.163` run with the ceiling
provisional; **`ρ₁ > −0.163` do not run**. Under anchored placement the two routes land on opposite
sides of that threshold — plateau-implied **−0.13976** (do not run) against measured lag-1 **−0.22698**
(run, provisional). The gate's input is no longer a single number, so **the arm must not run until §3 is
resolved.** This is the pre-registration working: the gate was written before the number existed, and it
is now doing its job by refusing an arm rather than by licensing one.

## 6. What did NOT change

- **Assumption (A1) is broken at every h ≥ 8 in BOTH placements.** The ceiling is unidentified past
  h=1 either way, and the sensitivity sweep still shows even a generous `c₁` implying 0.00000 at h=64
  and h=192. The martingale-residual bound is vacuous at the long end — this was never a placement
  artifact and the flat-ceiling reading `ceil_h = c₁` remains a plateau-model extrapolation, not a
  measurement.
- The horizon ORDER of student IC: .162 / .134 / .126 / .130 / .126 / .103 / .087 at h = 1/8/16/32/64/
  128/192 anchored, monotone-ish decay, same shape as strided at 2.1-2.7× the level.
- `best reachable market-neutral MSE ratio` at h=1 is **0.97224** under both placements at `c₁ = 0.1666`
  — invariant to the DRAW, because at h=1 the variance ratio is identically 1 and the quantity reduces
  to `1 − c₁²`. **RETRACTED in one direction** (job 5529, `EvalThroughput`): I wrote that it "cannot
  move", and it moves to **0.89492** the moment `c₁` becomes 0.32416, because it is a pure function of
  the assumption. It is draw-invariant and assumption-determined — which makes it a statement about
  `c₁`, not about the model, and it should never have been read as a reachable-accuracy target.
- **The ceiling instrument is NULL at every h ≥ 8 on this corpus, at every `c₁` up to 0.5.** Under
  `c₁ = 0.32416` every decision horizon prints "the gap is created by the assumed 0.3242" and FAILS the
  measured-`c₁` sensitivity, and the sweep implies 0.00000 at h=64 and h=192 even at `c₁ = 0.5`. So the
  only horizon where the construction yields a number is h=1, where the default convention makes it
  tautological. Its surviving deliverables are the measured variance-ratio curve and the reversal
  violation, not a ceiling.

## 7. Ranked consequences for the next session

1. **§3's falsifier**, because it decides whether the plateau or the direct lag-1 is the trustworthy
   route, and every frozen-scale and prewhitener tap depends on which.
2. **Re-size every IC threshold** against the anchored paired SE (3-4× larger). `LatentProbe`'s
   `+0.010` World A bar and the in-period arm's peak-location test both need this before they run.
3. **`OrthoTargets`' prewhitener taps** come from `min |ρ₁|` across disjoint spans. Anchored `min |ρ₁|`
   over the three populations is **0.13976** (plateau route) or 0.20061 (direct route) versus the
   0.207 assumed yesterday. The min-rule keeps this SAFE — a mis-fitted tap costs metric quality and
   cannot make the objective improper — but the taps change and job 5478's artifact should be re-read
   under the anchored draw before they are frozen.
4. **The trading family** was never scored on the anchored draw at all. Every published gross/breakeven
   number is a strided-draw quantity on cross-sections averaging 10.6 names.

## 8. The trading verdict under anchored placement — it inverts, and the mechanism is embarrassing

Job 5483 already wrote the whole `utility_*` family for both placements, so this cost no GPU. Same
checkpoint, `held-out full`, `mean decile long/short`, all DEMONSTRATED:

| h | gross payoff, bps per hold | | breakeven, bps per side | | net rate at 1 bps/side, bps per bar held | |
|---|---|---|---|---|---|---|
| | strided | **anchored** | strided | **anchored** | strided | **anchored** |
| 1 | 0.50 | 0.95 | 0.25 | 0.47 | −1.50 | −1.05 |
| 8 | 5.63 | **13.33** | 2.81 | **6.66** | 0.45 | **+1.42** |
| 16 | 10.54 | 22.39 | 5.27 | 11.20 | 0.53 | +1.27 |
| 32 | 17.22 | 38.16 | 8.61 | 19.09 | 0.48 | +1.13 |
| 64 | 37.06 | 63.28 | 18.52 | 31.65 | 0.55 | +0.96 |
| 128 | 46.22 | 84.20 | 23.07 | 42.08 | 0.35 | +0.64 |
| 192 | **−6.77** | **99.95** | −3.37 | 49.90 | −0.05 | +0.51 |

Turnover is **2.0** at every horizon (measured 1.9985-2.0030), active fraction **0.1981** — two deciles,
as designed. Cohort dispersion at 245 anchors: h=8 payoff 13.33 with cohort std 36.56 → **SE 2.34,
t ≈ 5.7**; h=64 63.28 ± 10.84 (t ≈ 5.8); h=1 0.95 ± 0.66 (**t ≈ 1.4, not significant**). Overlap
correction matters only at the long end — h=192 holds span ~2.5 days against ~1 anchor per day, so
effective n ≈ 98 and t falls to ≈ 4.7; at h=8 (40 minutes) overlap is negligible.

**The mechanism is that the strided draw's deciles contained ONE NAME.** Mean cross-section width
10.610549, so a decile is 1.06 names: the "decile long/short spread" that every trading verdict in this
project has quoted was a difference between two individual stocks picked by forecast rank. Under
anchored placement the width is 1,666.3 and a decile holds ~166 names. That is why the strided h=192
number is *negative* — a single-name spread at a 192-bar hold is a coin flip with a fat tail, and it
landed tails.

**Standing verdict, restated:** at h = 8 the model earns **+1.42 bps per bar held after a 1 bps/side
round trip** on a 1,666-name cross-section, with a **6.66 bps/side** cost allowance against a realistic
1-2 bps/side. The previous verdict — "only h ≥ 32 clears realistic costs, h = 1 unexecutable, breakeven
1.2089 bps/side" — was measured on 1.06-name deciles and is **retracted**; only its h=1 conclusion
survives, and for the right reason (t ≈ 1.4).

Not established, and none of it is optional before capital moves: no market-impact model, no borrow
cost or shortability constraint, no capacity estimate at 166 names per side, cohorts overlap and are not
simultaneously fundable, and the tail is real — worst observed cohort at h=64 is **−1785 bps**. The
external portfolio agent's synchronized evaluator measured 0 fills and 0 PnL on the STRIDED draw with
applied gains of 0; it has never been run on the anchored draw, and doing so is now the single
highest-value trading measurement available.

## 9. PRE-REGISTRATION — the increment arm, job 5538, stated before the run existed

Command (binary `/var/tmp/tb0_v21`, stamped from the green tree at 09:5x):

```
train-timexer-segment --run timexer-increment8-2500 --features all --layers 8 --d-model 512
  --heads 8 --ffn 2048 --min-history 256 --batch-size 256 --optimizer polar-express
  --preview-patience 3 --x0-lambdas disabled --horizon-mean increment:8 --max-steps 2500
```

Single variable against `timexer-control-4k`, whose argv is byte-identical minus `--horizon-mean`
(recovered from mlq attempt 3962). 2,500 steps because `PeakCause` measured 99.99% of distinct
supervised outcomes consumed by then and every step beyond is redundant by construction.

**Costs, from `FutureTeacher`'s ledger:** head output rows 1536 → 800, so **−753,664 parameters
(−2.70%)**, 16.98 → **16.53 TFLOP/step**, predicted step time **147-158 ms** against the control's 168.

**Predicted signs, in the order they falsify the arm:**

1. **h=192 market-neutral MSE ratio at step 2000 falls below 1.00** (control: 1.0245, i.e. *worse than
   forecasting zero*). This is the whole thesis — the h=192 cumulative target rests on ~504 independent
   192-bar windows while the h=192 *increment* target is one non-overlapping calendar bar backed by all
   2.46 M rows. **If it lands ≥ 1.01 the arm is refuted** and the long end is limited by representation
   or information, not by supervision, which would leave `LatentProbe`'s World A as the only live lever.
2. **h=1 and h=8 within ±0.01 of control** (0.9553 / 0.9507). Increments cannot add short-horizon
   information; a short-end *gain* would mean the change is acting as a regularizer and the attribution
   is wrong.
3. **Held-out IC at h=64 and h=192 improves.** Comparable across arms despite the strided draw, because
   the attenuation factor is a property of the draw and both sides share it — absolute levels are lower
   bounds, the ratio is not.
4. **The peak stays near step 2000.** Increments change what is supervised, not how many rows exist, so
   nothing about epoch exhaustion should move. A shifted peak would mean the arm changed the data
   geometry and not just the target.

**NLL is NOT comparable and must not be quoted against 2.3947663 or the control's curve** — the arm's
loss lives in increment space, which is a different metric with its own free lunch. That is
`OrthoTargets`' ruling from a target-basis boundary and it applies verbatim here. Selection inside the
arm is still valid; comparison across the boundary is not.
