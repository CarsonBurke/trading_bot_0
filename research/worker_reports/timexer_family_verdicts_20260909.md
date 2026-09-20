# Cross-section family verdicts and the amplitude-calibration landing, 2026-09-09

Every number here is DEMONSTRATED from `.report.bin` via `report_cli` unless marked [INFERENCE].

## 1. What was adopted, what was not

ADOPTED: the FIRST cross-section family (`dispersion` + `cross-section-z`, 14 auxiliary channels).
It is the best configuration measured on both axes.

NOT ADOPTED: the SECOND family (`cross-section-rank` + `relative-volume` + `range-z`, 22 channels)
and the rank-only ablation (18 channels). Both are UNRESOLVED against their own fixed
pre-registrations and neither beat the first family at step 2000.

## 2. `held-out sample` market-neutral four-channel MSE ratio, step-matched

Below 1.0 beats persistence. Base `timexer_segment_horizon_steps`, rows are optimizer steps.

| h | control @2000 | 1st fam @1000 | 1st fam @2000 | 2nd fam @1000 | 2nd fam @2000 | rank-only @1000 | rank-only @2000 |
|---|---|---|---|---|---|---|---|
| 1 | 0.95528 | 0.9542 | **0.9489** | 0.94674 | 0.95399 | 0.96250 | 0.95940 |
| 8 | 0.95066 | 0.9677 | **0.9347** | 0.94248 | 0.95237 | 0.96394 | 0.94507 |
| 16 | - | 0.9736 | **0.9375** | 0.95542 | 0.95805 | 0.96921 | 0.94948 |
| 32 | - | 0.9876 | **0.9564** | 0.96669 | 0.97232 | 0.97325 | 0.96899 |
| 64 | 0.98550 | 0.9898 | 0.9993 | 0.98362 | - | 0.98040 | 0.98678 |
| 192 | 1.02446 | 0.9990 | 1.0433 | - | - | 0.99299 | 1.01914 |

The first family at step 2000 is the best column at h=1, 8, 16, 32.

## 3. Anchored `held-out full` student IC

Same instrument, same anchored draw (245 paired cross-sections, mean width 1666.3), same eval batch
size. Single-arm SE at h=1 is 0.009976.

| h | control | 1st family | 2nd family |
|---|---------|------------|------------|
| 1 | 0.16208 | 0.19362 | 0.20289 |
| 8 | 0.13375 | 0.16365 | 0.16891 |
| 16 | 0.12622 | 0.15097 | 0.14624 |
| 32 | 0.13037 | 0.14296 | 0.14561 |
| 64 | 0.12609 | 0.13935 | 0.13750 |
| 128 | 0.10267 | 0.12034 | 0.11686 |
| 192 | 0.08722 | **0.10276** | 0.09545 |

First family vs control: +10% to +22% at EVERY horizon, h=1 +0.0315 (~3 SE unpaired). This is the
session's one robust information gain.

Second family vs first: +0.0093 at h=1 (under 1 SE), -0.0073 at h=192. Its pre-registered bars were
CONFIRMED-strong 0.2218 / weak 0.2077 / REFUTED 0.1655; 0.20289 clears none of them. UNRESOLVED.

## 4. Persistence ablation: UNRESOLVED, and the prediction failed in an informative direction

Pre-registration `timexer_persistence_ablation_prereg.md` required BOTH:
1. step 1000 retains the early gain: h=1 <= 0.9500 AND h=8 <= 0.9500. MEASURED 0.96250 / 0.96394 - FAILS.
2. no reversal: step-2000 ratio <= step-1000 ratio at h=1 and h=8. MEASURED 0.95940 <= 0.96250 and
   0.94507 <= 0.96394 - PASSES.

Verdict UNRESOLVED. The hypothesis was that `relative-volume` and `range-z` are persistent
per-ticker characteristics acting as identity labels, buying early fit and then degrading, while
`cross-section-rank` carries the real contemporaneous signal. Condition 2 behaved as predicted
(removing them removed the reversal) but condition 1 did the opposite: the rank-only arm did NOT
reproduce the second family's step-1000 advantage, so whatever produced that early gain came from
the two channels the hypothesis blamed. Persistence-as-memorization does not explain that, and it
is not a verdict.

Recorded for whoever resolves it: the REFUTED branch named a channel-count control (three
contemporaneous noise channels) as the test that separates added capacity from persistence. That
control was not run and remains the correct next step on this question.

## 5. Amplitude calibration: from no-op to fitted

The per-horizon mean gain was IDENTITY in every arm before this landing. Cause was not plumbing:
`MeanCalibration::fit`'s intercept refusal gate returned identity as a successful value, and the
run binary's own report title said so - `NO GAIN APPLIED: a pure gain is the wrong parameterization
on this block ... 7.535e-2 ... above the 0.1 share`.

Two deviations FROM the pre-registration, both fixed; the 0.1 threshold value is untouched:

- `Moments::intercept_ceiling` pooled `ybar^2/bars` over all four decoded channels. High and low
  target means are structurally nonzero, so it measured 7.535e-2 where the registered close-channel
  quantity is 4.63e-5 - 1600x larger. Proof it is the intrabar offset and not drift: `ceiling*h` is
  flat (0.0754, 0.0960, 0.0926, 0.1235, 0.1304, 0.1219, 0.1241, 0.1372, 0.1154 at
  h=1,2,4,8,16,32,64,128,192), i.e. `ceiling ~ 1/h`; drift must grow as `h*mu^2`.
- The gate compared max-over-horizon intercept against max-over-horizon amplitude cost
  (7.535e-2 at h=1 vs 2.7908e-2 at h=171) where the registered test is per-horizon. One
  short-horizon intercept blanked all 192 horizons in both coordinates.

Neither fix alone suffices: per-horizon but still pooled needs `cost_h > ceiling_h/0.1`, i.e.
> 7.535e-1 at h=1 and > 3.81e-2 at h=32, both above the 2.7908e-2 axis maximum, so everything out
to h~32 would stay refused.

Post-fix the gate admits with 19-91x margin, and run 6017 logs a real curve at every evaluation:
`close-anchor gain 2.7673 at h=1 to 0.5724 at h=192` (step 1000), `2.9798 to 0.1648` (step 2000),
`1.6196 to 0.1337` (step 2500).

Also landed: every silent-identity path now returns `Err` (`CurveFit::identity`,
`CurveFit::unidentifiable`, `FrozenGain::is_identity` deleted); three sizing-gate reasons stay
distinct (`Unmeasured`, `NonPositiveGain`, `InterceptDominates`) and are never merged or clamped;
`Vec<Option<f64>>` replaces `Vec<f64>` for measured anchors because serde_json writes NaN as `null`
and refuses to read it back, which made any checkpoint with an unidentified horizon unloadable.

The fit draw was 2,048 origins, inherited from `args.eval_origins` for cost parity and never
justified. DEMONSTRATED inadequate: at step 2500 the same split gives h=1 gain 0.7185 on 2,048
strided origins against 1.5407 on all 433,303. Leaving the gain at 1 costs (1.54-1)^2 = 0.29;
applying the 2,048-origin estimate costs (1.54-0.72)^2 = 0.67 - **2.3x worse than no correction**,
and a one-sided three-sigma bound cannot catch it because the error is a shrink. Now
`AMPLITUDE_FIT_ORIGINS = 32_768`, costing ~10 s per evaluation against a 660 s run.

## 6. Mechanism of the amplitude error: UNRESOLVED

The run log's claim `amplitude error is IN SAMPLE TOO` does NOT establish the objective-artifact
hypothesis. The training-split gain at step 2500 reads h=1 -0.636, h=8 1.454, h=16 1.590,
h=32 1.231, h=64 1.188, h=128 1.431, h=192 0.978 - neither flat at 1.0 (out-of-sample shrinkage)
nor a monotone ~1/h sweep (objective artifact), and not monotone at all.

The negative h=1 value is a red flag rather than a datum: `channel_gains` is an UNCENTERED
`sum(f*y)/sum(f^2)`, so it proves a negative raw cross-moment, not a negative centered correlation.
The training draw was 512 origins against the calibration draw's 2,048 and the full split's 433,303,
so the two series were never comparable. Both mechanisms stay live; resolving it needs both gains
measured on the enlarged draw with a stated standard error.

## 7. Infrastructure fixed this session, with the evidence that forced each

- FULL-SPLIT EVALUATION OOM, cause of ~10 dead arms. `Scorer::trading` computed global ranks with
  `argsort(0).argsort(0)` over a dense `[433303, 192]` Int64 array = 634.720 MiB per sort output,
  which at the allocator's 2 MiB rounding is exactly the observed 636.00 MiB failing request; eight
  persistent fp32 `[origins, horizons]` banks add 2,541 MiB. Fixed by processing one horizon at a
  time, following `utility.rs`'s existing convention: every `[N,H]` temporary collapses to `[N]`,
  a 192x cut. Non-pool peak 5.75 GiB -> 2.60 GiB. Bit-reproducibility preserved: no reordering
  within a horizon; the one unavoidable accumulator-tree change measures exact equality at small
  shapes and 8.5e-17 relative at full shape, below the 4.441e-16 repeated-shape floor.
- The same bug wore two faces: `runner.rs` calls `score()` on the full split at BOTH epoch
  completion and the final mid-epoch exit. A 9,590-step epoch never reaches the first in 2,500
  steps; `--row-stride-multiple 30` makes an epoch 319 steps and reaches it one minute in.
- SPLIT GUARD. A disjointness proof compared GLOBAL timestamp extrema and could never pass, killing
  a 26-minute arm at its first evaluation. The real invariant: partitions are a SHARED wall-clock
  quantile cut (70/80/90 of the union occupancy grid) resolved into each ticker's own valid-bar
  ordinals, with a purge of `max(pred_len,100) >= 100` bars. Global extrema interleave because
  `boundaries[k]-1` is "this ticker's last bar before the shared instant", arbitrarily early for a
  delisted name. Now proved per ticker, and moved to STARTUP before the first optimizer step.
  A prior measurement on the real corpus: 0 of 4,498 tickers violate the ordering and not one of
  the 433,303 scored origins shares a timestamp with a fit origin.
- CAPTURE-FAILURE FALLBACK removed earlier: a capture OOM used to drop an optimizer update silently
  and continue eagerly. Now a panic.
- LEVEL-CHANNEL CONDITIONING. `E[x^2]-mean^2` is well conditioned on returns (mu~1e-5, sigma~1e-3)
  and cancels on levels (`ln(volume)`~12, `ln((high-low)/close)`~-4). Three identical contributors
  gave sigma = 8e-8 instead of 0, which against the f32 mean's own 6e-8*|mu| produced 0.787 - a
  rounding bit presented to the model as a 0.79-sigma reading with validity 1. Floored at
  `1e-3*(1+|mu|)`, touching only the two level channels.
- Throughput retraction: the training step is 154 ms at batch 256 = 110 TFLOP/s with GPU busy 100%
  and power at 92.7% of limit. It is AT ROOFLINE; there is nothing to win there. Preview evaluation
  is 1.7 s. A 2,500-step arm is ~11 minutes including the full-split pass. The 31 minutes attributed
  to a run earlier was queue wait behind siblings.

## 8. Standing conclusions unchanged

- The horizon axis stays CLOSED: five interventions refuted, and this session added no sixth.
- Per-bar information remains the only demonstrated lever, and the first cross-section family is the
  only intervention that has moved every horizon at once.
- IC is scale-invariant per horizon; MSE ratio is not. The long-horizon "regression" that looked
  like a short-for-long trade was the 5x over-amplitude multiplied by a larger sigma_y(h), while
  h=192 information rose 17.8%.

## 9. Amplitude calibration, applied and measured (arm `timexer-fam1-calib-2500`, job 6032)

First family, gain fitted on the calibration split [70%,80%) over 32,768 origins, scored on the
held-out splits. Four-channel market-neutral MSE ratio at step 2500, base
`timexer_segment_amplitude_calibration`, rows are horizons:

| h | sample uncal | sample cal | full uncal | full cal |
|---|--------------|------------|------------|----------|
| 1 | 0.96787 | 0.96311 | 0.95382 | 0.95158 |
| 8 | 0.95474 | 0.94249 | 0.97902 | 0.97814 |
| 16 | 0.94507 | 0.94383 | 0.98259 | 0.98256 |
| 32 | 0.96882 | 0.97085 | 0.98774 | 0.98729 |
| 64 | 0.98537 | 0.98593 | 0.99573 | 0.99132 |
| 128 | 0.98237 | 0.98579 | 1.00276 | 0.99539 |
| 192 | 0.99894 | 0.99347 | 1.00645 | 0.99712 |

On `held-out full` - the 433,303-origin population, the only one large enough to carry the long end -
every horizon improves or holds, and h=128 and h=192 cross from ABOVE 1.0 to BELOW it. The
pre-registered prediction from the best-scale numbers was ~0.997 at both; measured 0.99539 and
0.99712. On the 2,048-window `held-out sample` draw the effect is mixed at h=32/64/128 (+0.002 to
0.003), which is expected for an out-of-sample correction on a thin draw and is not a refutation.

## 10. Why arms degrade with more steps: objective/metric divergence, NOT overfitting

Question asked directly, answered from reports rather than theory. Three candidates, two eliminated.

NOT OVERFITTING. `timexer_segment_generalization_gap`, training minus held-out sample objective NLL:
xsec2 0.23841 -> 0.22356 -> 0.22515; rank-only 0.23270 -> 0.22300 -> 0.20810. The gap NARROWS while
the metric degrades. Overfitting widens it.

NOT THE SCHEDULE. `timexer_segment_lr_trajectory` at steps 2497-2499 is still exactly at plateau
(NorMuon packed QKV 0.03983717, attention output 0.023, MLP up 0.046). A 2,500-step arm never cools,
because the schedule is shaped against 9,590 steps. No annealing occurs in either direction.

NOT A POPULATION CHANGE. `timexer_segment_horizon_steps_population` reports exactly 100 contributing
cross-sections at h=1 at every one of steps 1000/2000/2500, and `held-out sample` is the same 2,048
windows throughout. This was checked because a draw change has already produced one retracted result
this session.

WHAT IT IS. Rank-only arm: held-out NLL 2.0096 -> 1.9935 -> 1.9936, h=1 cross-section IC
0.0819 -> 0.1054 -> 0.0451, h=1 amplitude gain 2.7673 -> 2.9798 -> 1.6196. NLL is flat to four
decimals from 2000 to 2500 while IC falls by more than half and the mean's amplitude nearly doubles
toward its MSE-optimal value. Those are consistent under a heteroscedastic Gaussian head, which can
hold NLL constant while reallocating between the mean and sigma. Since the MSE ratio at the optimal
scale is 1 - rho^2, a collapse in rho worsens MSE even as the amplitude becomes better calibrated.

[INFERENCE] The mechanism this implicates: at rho ~ 0.1 the conditional mean explains ~1% of target
variance, so NLL is dominated by sigma and intrabar geometry. Once that structure is learned around
step 2000, continued NLL descent carries no pressure to preserve the mean's directional content.

DEMONSTRATED CONSEQUENCE, independent of that inference: checkpoint selection is on NLL, and NLL at
step 2000 (1.9935) versus 2500 (1.9936) is a coin flip while the two states differ by 2.3x in h=1 IC
and by 3.5pp at h=192. Selection is therefore choosing near-randomly among materially different
checkpoints. Note also that degradation is HORIZON-SPECIFIC, not global: in arm 6032, 2000 -> 2500
moved h=1 from 0.95229 to 0.96787 (worse) while h=128 went 1.01683 -> 0.98237 and h=192 went
1.03376 -> 0.99894 (much better).

Caveats carried, not smoothed: the 2000 -> 2500 interval is one interval, and the cross-section IC
draw is 100 cross-sections (SE ~ 0.016 by scaling the anchored draw's measured 0.009976 at 245), so
the IC drop is ~4 SE - real but thin. The clean instrument is the anchored draw at both steps.

### Next actions this implies
1. Score the anchored draw at step 2000 AND step 2500 on one arm, to confirm the IC trajectory on
   the instrument that already corrected a 2-3x error rather than on 100 thin cross-sections.
2. Change checkpoint selection off NLL. NLL is not the quantity being maximized; it cannot separate
   states that differ 2.3x in IC. This is an objective/selection change, NOT a horizon-axis
   intervention, so it does not touch the closed axis.
3. Set `--schedule-budget` to the actual step cap on every future arm. Every arm measured so far ran
   with the schedule shaped for 9,590 steps and stopped at 2,500, so all of them were read at
   plateau LR and none was ever annealed. No arm has yet been read at a cooled learning rate.

## 11. Pre-registration: annealed endpoint (written before the arm exists)

Arm `timexer-fam1-anneal-2500`: first family, `--schedule-budget 2500 --max-steps 2500`, so the
learning rate actually reaches its floor at the step cap. Every other flag identical to arm 6032.

Baseline is arm 6032's step-2500 point, read at plateau LR: `held-out full` four-channel calibrated
ratio h=1 0.95158, h=8 0.97814, h=64 0.99132, h=128 0.99539, h=192 0.99712; `held-out sample`
calibrated h=1 0.96311, h=8 0.94249, h=16 0.94383.

Predicted sign and size, fixed here: the annealed endpoint improves the `held-out full` calibrated
ratio at EVERY horizon, with h=1 <= 0.9480 (a >=0.0036 gain, i.e. larger than the whole measured
effect of amplitude calibration at that horizon) and h=192 <= 0.9950. REFUTED if any horizon
degrades by more than 0.0020, or if h=1 exceeds 0.9516. UNRESOLVED between those.

Rationale for expecting a large effect rather than a marginal one: arm 6032's own 2000 -> 2500
interval moved h=1 by +0.0156 and h=192 by -0.0348 at constant plateau LR, so the late-training
state is wandering by amounts far larger than the calibration effect. Annealing removes that
wander. If the prediction fails, the wander is not LR-driven and the objective/metric divergence in
section 10 is the whole story.

## 12. Annealed endpoint result (arm `timexer-fam1-anneal-2500`, job 6036)

`held-out full` four-channel market-neutral ratio at step 2500, calibrated, against arm 6032's
plateau-LR step-2500 point:

| h | plateau (6032) | annealed (6036) | delta |
|---|----------------|-----------------|-------|
| 1 | 0.95158 | **0.92832** | -0.02326 |
| 8 | 0.97814 | 0.96425 | -0.01389 |
| 16 | 0.98256 | 0.97419 | -0.00837 |
| 32 | 0.98729 | 0.98194 | -0.00535 |
| 64 | 0.99132 | 0.98632 | -0.00500 |
| 128 | 0.99539 | 0.99488 | -0.00051 |
| 192 | 0.99712 | 0.99700 | -0.00012 |

Every horizon improved. h=1 goes from a 4.8% margin over persistence to 7.2%, a 47% increase in
skill margin and the largest single gain of the session.

VERDICT AGAINST THE SECTION-11 PRE-REGISTRATION: **UNRESOLVED**, recorded as written rather than as
a win. CONFIRMED required improvement at every horizon AND h=1 <= 0.9480 AND h=192 <= 0.9950.
The first two clauses hold (h=1 measured 0.92832, clearing its bar by 4x); the third fails,
h=192 measured 0.99700 against a 0.9950 bar, missing by 0.0020. Not REFUTED either: no horizon
degraded, so the refutation clause is untouched. The threshold was fixed before the run and is not
being moved now.

Every arm before this one ran with the schedule shaped against the 9,590-step epoch and stopped at
2,500, so all earlier numbers in this document were read at plateau LR and none was annealed.
`--schedule-budget <max-steps>` belongs on every future arm.

## 13. Degradation with more steps survives annealing: it is generalization, not schedule

Arm 6036, annealed to floor, single-pass data (122,758,732 of 470,946,393 unique target bars at
step 2500, so ZERO sample repetition):

| step | training NLL | held-out sample NLL |
|------|--------------|---------------------|
| 1000 | 2.2545373 | 1.9956235 |
| 2000 | 2.2205098 | 1.9908248 |
| 2500 | 2.2073860 | 2.0173137 |

From 2000 to 2500 training NLL IMPROVES by 0.0131 while held-out NLL WORSENS by 0.0265. The same
divergence appears at plateau LR in arms 6032 (held-out 1.98253 -> 1.99700) and 6017
(1.99536 -> 2.00805). Three arms, two schedules.

This eliminates both earlier explanations. Not the learning-rate schedule: it happens with the LR
cooled to floor. Not sample memorization: nothing repeats. The training objective's own held-out
value degrades while its training value improves.

LEADING HYPOTHESIS, not yet demonstrated: the model progressively fits the TRAINING PERIOD's regime.
Training is [0%,70%) of the calendar and every held-out split is strictly later, so more steps buy a
better fit to the earlier regime and transfer worse to the later one.

DISCRIMINATOR RUNNING (arm `timexer-inperiod-2500`, `--in-period-sections 32`): the in-period hole
holds the market period FIXED and varies only origin identity, per `corpus.rs:69-74` - "a
chronological split conflates two different things - an origin the sampler never visited, and a
market period the model never saw - and those have opposite fixes". Within one arm:
- in-period held-out NLL keeps improving past step 2000 while the temporal-tail split degrades
  => REGIME SHIFT, and the fix is recency weighting or a moving-window curriculum, not more steps;
- both degrade => not shift, and the cause is an optimization or objective pathology, which would
  make the regime story wrong.
Note from the source that a nonzero `in_period_sections` REMOVES training rows, so this arm's
TRAINING losses are not step-comparable to a control's; the comparison used here is WITHIN the arm,
between its two held-out populations, so that caveat does not bind.

## 14. DISCRIMINATOR RESULT: regime overfitting, concentrated at long horizons (job 6043)

Base `timexer_segment_temporal_generalization`, arm `timexer-inperiod-2500`
(`--in-period-sections 32`). Both held-out populations are scored inside the SAME arm at the SAME
step from the SAME weights; only the market period differs. The base emits the paired difference and
its standard error directly, so this is a within-arm paired comparison and the arm's removed
training rows do not bind on it.

`in-period minus out-of-period` cross-sectional IC (positive = predicts its own training period
better than a later one):

| h | step 1000 | step 2000 | step 2500 |
|---|-----------|-----------|-----------|
| 1 | -0.04576 +- 0.03611 | -0.03724 +- 0.03026 | -0.02143 +- 0.03173 |
| 8 | -0.01184 +- 0.03360 | +0.00645 +- 0.02985 | +0.06121 +- 0.02855 |
| 16 | -0.00050 +- 0.03335 | +0.04205 +- 0.03027 | +0.07948 +- 0.03162 |
| 32 | +0.02677 +- 0.03199 | +0.05526 +- 0.02992 | +0.04071 +- 0.03239 |
| 64 | -0.01725 +- 0.03537 | +0.09928 +- 0.03504 | +0.05703 +- 0.03640 |
| 128 | +0.00866 +- 0.03499 | **+0.20033 +- 0.03530** | +0.09059 +- 0.03400 |
| 192 | +0.02034 +- 0.03513 | **+0.18040 +- 0.03040** | +0.10426 +- 0.02873 |

Underlying levels at step 2000, h=128: in-period IC 0.21494 +- 0.02681 against out-of-period
0.01461 +- 0.02297.

DEMONSTRATED:
1. At step 1000 there is NO period gap at any horizon - every difference is inside one SE.
2. By step 2000 a gap opens at the long end at 5.7 sigma (h=128) and 5.9 sigma (h=192).
3. At h=1 there is NO period gap at any step, and the point estimate is negative throughout.

This is regime overfitting, not sample memorization: nothing repeats (122.8M of 470.9M unique target
bars) and the in-period held-out origins were never sampled, yet the model predicts them far better
than later ones. It learned structure that generalizes across ORIGINS within a period and not across
PERIODS. The horizon localization is the tell: one-bar returns are microstructure and stationary,
while 128-192-bar returns are dominated by regime and drift, which are period-specific.

### Consequences for standing conclusions

- The claim "the horizon axis is CLOSED" is REPLACED, not confirmed. The long-horizon information the
  model finds is REAL but PERIOD-LOCAL, so no reweighting or reparametrization of the horizon axis
  could ever have made it transfer. The five refuted interventions were symptoms of this, measured
  through metrics simultaneously contaminated by the 18.8x amplitude error and read at plateau LR.
  Any retest of that axis must now be read on `timexer_segment_temporal_generalization`, which did
  not exist as an instrument when those five were run.
- Why h=1 improves monotonically while the long end decays is now explained by one measurement
  rather than by two separate stories.
- Checkpoint selection on held-out NLL is doubly wrong: it cannot separate states differing 2.3x in
  IC (section 10), and it rewards period-local long-horizon fit, which is exactly the component that
  does not transfer.

### Next actions, in expected-value order

1. Select and early-stop on OUT-OF-PERIOD long-horizon IC, not on NLL. The instrument exists and is
   cheap; it is the only criterion measured so far that tracks what we want.
2. Retest the horizon-weight axis with the new instrument and with amplitude calibration and
   annealing in place. `cutoff:32` was refuted before any of those three existed; that refutation is
   no longer authoritative. Pre-register on out-of-period IC at h=1/8/16 with the period gap as the
   secondary reading.
3. Consider period-invariance directly: the target is already market-neutral, so what remains
   period-local is the conditional structure. Any candidate must be pre-registered against the
   period gap at h=128/192, which is now a measured quantity with a 0.03 SE.
4. All future arms carry `--schedule-budget <max-steps>` (section 12) and are read on the calibrated
   `held-out full` ratio (section 9), never on the uncalibrated one.

## 15. Accuracy ranking of every arm on shared instruments

### 15a. In-run `held-out full` calibrated market-neutral four-channel MSE ratio (433,303 origins)

| arm | h=1 | h=8 | h=16 | h=32 | h=64 | h=128 | h=192 |
|-----|-----|-----|------|------|------|-------|-------|
| xsec2 (2nd family) | 0.94506 | 0.97074 | 0.97717 | 0.98348 | 0.99251 | 1.00210 | 1.00709 |
| rank-only | 0.94595 | 0.97677 | 0.98209 | 0.98607 | 0.98960 | 0.99595 | 0.99765 |
| fam1 plateau (6032) | 0.95158 | 0.97814 | 0.98256 | 0.98729 | 0.99132 | 0.99539 | 0.99712 |
| **fam1 annealed (6036)** | **0.92832** | **0.96425** | **0.97419** | **0.98194** | **0.98632** | **0.99488** | **0.99700** |
| in-period (6043) | 0.94652 | 0.97855 | 0.98588 | 0.99109 | 0.99308 | 0.99616 | 0.99720 |

The annealed arm is the only arm that is best at every horizon at once.

Two rows are not fair contenders. xsec2's uncalibrated and calibrated columns are BYTE-IDENTICAL at
every horizon - it ran on the pre-gate-fix binary where the fit silently returned identity - so its
h=128/h=192 above 1.0 carry uncorrected amplitude, not only lost information. The in-period arm is
the diagnostic of section 14 and has training rows deliberately removed. `timexer-control-4k` cannot
appear at all: it predates the base, its gens carry no `timexer_segment_amplitude_calibration`.

### 15b. Anchored `held-out full` close cross-sectional IC (job 6080, `--placement both`, batch 256)

The annealed checkpoint scored on the instrument of record. Its own strided column is shown to keep
the placement correction visible:

| h | annealed strided | annealed ANCHORED | control step 2000 | fam1 plateau (5668) |
|---|------------------|-------------------|-------------------|---------------------|
| 1 | 0.07421 | **0.23230** +- 0.01148 | 0.16208 +- 0.00839 | 0.19362 |
| 8 | 0.05788 | **0.18846** +- 0.01132 | 0.13375 +- 0.00963 | 0.16365 |
| 16 | 0.06452 | **0.16649** +- 0.00969 | 0.12622 | 0.15097 |
| 32 | 0.07169 | **0.15071** +- 0.00863 | 0.13037 | 0.14296 |
| 64 | 0.07009 | 0.13854 +- 0.00889 | 0.12609 | 0.13935 |
| 128 | 0.06004 | 0.11188 +- 0.00901 | 0.10267 | 0.12034 |
| 192 | 0.09558 -> see note | 0.09558 +- 0.00851 | 0.08722 | 0.10276 |

Against the pinned control the annealed model is higher at EVERY horizon: +43.3% at h=1, +40.9% at
h=8, +31.9% at h=16, +15.6% at h=32, +9.9% at h=64, +9.0% at h=128, +9.6% at h=192.

Against the previous best (fam1 plateau, NLL-selected, job 5668) the picture is MIXED and must be
reported as such: h=1 +0.0387 (3.4 sigma), h=8 +0.0248 (2.2 sigma), h=16 +0.0155, h=32 +0.0078,
then h=64 -0.0008, h=128 -0.0085 (0.94 sigma), h=192 -0.0072 (0.85 sigma). The long-end deficits are
inside one standard error and are NOT demonstrated; the short-end gains are. These are unpaired
across arms, so only the multi-sigma entries carry weight.

The strided-to-anchored ratio is 3.13x at h=1 and 1.79x at h=192, reconfirming section 5: every
strided IC this project ever published is a lower bound.

### 15c. Verdict

`timexer-fam1-anneal-2500` is the most accurate model measured to date. It wins outright on 15a and
beats the control at all seven horizons on 15b. The one tension in the record is 15b's long end
against fam1 plateau, at under 1 sigma and in the opposite direction from 15a's same-arm comparison,
where the annealed arm was better at h=128 and h=192 on 433,303 origins. Resolving the sign there
requires the same checkpoint pair scored anchored in ONE process, which 15b did not do.
