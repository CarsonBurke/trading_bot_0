# Per-horizon mean amplitude calibration: shipped mechanism, out-of-sample harness, and the GPU run that decides it

Owner: AmplitudeCal. Scope: post-hoc amplitude calibration of the conditional mean, fitted out
of sample, applied at scoring time, checkpoint bit-identical. All 192 horizons retained; no
horizon truncation, no trunk change, no new evaluation draw.

Tags: **DEMONSTRATED** = measured in this session or read from a report artifact.
**HYPOTHESIS** = derived but not yet measured on a real checkpoint.

---

## 1. Verdict up front

**DEMONSTRATED (host, synthetic ground truth):** the estimator and the whole application path
transfer out of sample on a two-block population with a known per-horizon over-amplitude and a
5% amplitude drift between blocks: 169 of 192 horizons improved on the untouched block, the
close-channel MSE ratio at h=192 went `1.093095 → 0.995233` against that block's own oracle
bound `0.993861`, and the worst within-timestamp `|ΔIC|` over all 192 horizons was `1.8e-8`.

**DEMONSTRATED (real curve, in population):** applied to `timexer-control-4k` step 3000's own
held-out-full `β̂` curve, the shipped estimator reproduces it to 0.2-2% and drives the amplitude
cross term to zero: `C = -0.025192 → -0.000000` at h=192, close ratio `1.021886 → 0.996694`,
and at h=64 `1.010849 → 0.989365`. This is the in-population oracle equivalent - it is the
bound, not the result.

**NOT YET DEMONSTRATED:** that the real forecaster's over-amplitude is stable across two
disjoint chronological blocks. That is one GPU run (§7) and it is the only thing that decides
whether step 4 is worth doing. Falsifier and pre-registered numbers in §8.

---

## 2. What shipped

| Where | What |
|---|---|
| `timexer_segment/calibration.rs` (new, 879 lines) | `Pairing`, `Blocks`, `Measured`, `MeanCalibration::{fit, write, read, gain_for}`, `chronological_blocks`, 8 tests |
| `timexer_segment/model.rs` | `CausalPatchModel::{fold_mean_gain, mean_gain}` + 1 test |
| `timexer_segment/runner.rs` | `CalibrateArgs`, `calibrate`, `load_checkpoint` (factored out of `evaluate`), `Manifest::pairing`, `EvaluateArgs::calibration`, the IC-invariance refusal, + 1 test |
| `timexer_segment/reports.rs` | `AppliedGain`, `write_calibration_gain`, `write_calibration_effect` + 1 test |
| `main.rs` | `calibrate-timexer-segment` subcommand |
| `docs/timexer_segment.md` | the two bases, the invariance rules, the command |

`cargo check -p trading_bot_0 --tests`: 0 errors. `cargo test -p trading_bot_0 timexer_segment`:
105 passed, 0 failed. `cargo check -p trading-bot-tui --tests`: 0 errors. **DEMONSTRATED.**

---

## 3. The four design decisions, and the measurement behind each

**Close anchor only, one coefficient per horizon - not four per horizon.** `decode_joint` emits
`low = close - o₁`, `high = close + o₂`, `open = close + o₃` with the offsets monotone
functions of the range and position coordinates; that construction is what makes
`high ≥ max(open, close) ≥ min(open, close) ≥ low` survive rounding. Four independent gains
rescale the offsets by four different numbers and can invert them - an invalid candle. Scaling
the anchor moves all four channels by the same amount and leaves the offsets exactly as the
head emitted them. The diagnosed error is also in the anchor: `O`, `D`, `C` are close-channel
quantities. Guarded by `a_folded_mean_gain_rescales_the_anchor_without_touching_geometry_ranks_or_weights`,
which asserts candle validity at every horizon after folding, on a live head, and that no
varstore tensor moved. Parameter count 192, not 768.

**Pure gain, no offset - and this is enforced, not assumed.** `offset_ceiling = ȳ²/E[y²] =
1 - D/ρ²` is the entire share of persistence MSE any constant forecast could ever earn.
`fit` refuses unless the worst horizon's ceiling is ≤ 0.1 × the worst horizon's amplitude cost
`-C`. **DEMONSTRATED on the real curve:** worst ceiling `4.634e-5` against worst amplitude cost
`2.995e-2` - a factor of 646, i.e. 65x inside the threshold. So the pure gain is confirmed from
the data rather than assumed, and if a future checkpoint's offset ever matters the fit stops
and says so instead of quietly leaving it on the table. Test:
`an_offset_worth_more_than_a_tenth_of_the_amplitude_error_refuses_a_pure_gain`.

**Smoothness: roughness-penalized weighted least squares on `ln β̂` against `ln h`, GCV-selected
- explicitly NOT monotone, and not a low-order parametric family.** Weights are
`n_h·ρ_h²/(1-ρ_h²)`, the inverse variance of `ln β̂`. Positivity is by construction (fit in
logs). Curvature is penalized, so the fit degenerates to a log-log straight line as the penalty
grows rather than being forced into a shape.
- Monotone is refuted by the data it would regularize: the measured curve rises `0.600` (h=1) →
  `0.989` (h=11) and then falls to `0.266` (h=192). **DEMONSTRATED.** A monotone prior would
  erase the short-horizon rise, which is the largest single feature of the curve.
- A global parametric shape is also refuted: weighted by their own measured precision, a
  log-log line leaves `χ²/dof = 23.6` and a log-quadratic `13.6`, while the curve's own second
  differences (`rms 0.0159` in log space) sit far below the noise scale `√6·SE ≈ 0.046`.
  **DEMONSTRATED.** The fine structure is real; a 2-parameter family cannot carry it.
- GCV over 41 geometric penalties picks `dof = 39.1` of 192 on the real curve - a 5x variance
  reduction against 192 independent fits while keeping the h=11 bump. **DEMONSTRATED.**
- Clamped to `(0, 1]`: shrinkage only. `β̂ > 1` means under-amplification, which no measured
  population here shows, and the MSE cost of amplifying on a noisy estimate is quadratic and
  unbounded. Test: `a_measured_gain_above_one_is_clamped_to_pure_shrinkage`.

**Horizons with no calibratable amplitude are carried by the penalty, not by a default.** Zero
weight for `β̂ ≤ 0`, `ρ² ∉ (0,1)` or no bars; those horizons are then interpolated by the
roughness term from their neighbours, which is the only defensible statement a smoothness prior
makes about them. Test: `horizons_with_no_calibratable_amplitude_are_carried_by_their_neighbours`.

---

## 4. Blocks, and the disjointness proof

**The population.** `Corpus::validation_refs` = the corpus's held-out full split: origins in
the chronological `[80%, 90%)` quantile band of the shared UTC timestamp grid
(`quantile_bounds` ranks at `7/10, 8/10, 9/10` of distinct timestamps; `target_count` admits an
origin only in `[boundaries[1]-1, retained_end(boundaries[2]) - pred_len)`), purged from train
by ≥ `pred_len` bars. **DEMONSTRATED from `corpus.rs:113-125, 1300-1324`.**

**The cut.** `chronological_blocks(&stamped, share)` sorts origins by their origin bar's UTC
millisecond, takes the earlier `share` (default 0.5) as the FIT block, and then **drops every
remaining origin whose origin timestamp is at or before the fit block's last TARGET
timestamp**. The surviving later origins are the untouched block. So:

1. No origin is in both blocks (the split is a partition of a sorted list, then a suffix
   filter).
2. **No bar the fit block's targets read is an origin of, or a target of, the untouched
   block** - that is the purge, and it is what makes the two blocks disjoint in DATA rather
   than merely in origin identity. An origin whose 192-bar cumulative target reach crosses the
   cut is scored by neither block.
3. A whole timestamp always lands on one side: the sort key is the timestamp, so a
   cross-section is never split across the two blocks.

All three are asserted in `the_two_blocks_are_disjoint_in_the_bars_their_targets_read` (12
timestamps × 3 tickers, targets reaching 2 timestamps ahead: 18 fit origins, 12 untouched, 6
purged, and the assertion that every untouched origin starts strictly after the fit block's
last target bar). A population whose targets span the whole block is refused rather than split:
`a_population_whose_targets_cover_the_whole_block_cannot_be_split`. **DEMONSTRATED.**

**The dating is emitted, not assumed.** `Blocks` carries
`calibration_{first,last}_origin_ms`, `calibration_last_target_ms`, `calibration_origins`,
`evaluation_{first,last}_origin_ms`, `evaluation_origins`, `purged_origins`. Those go into the
artifact, into both report titles, and into stdout. After the run in §7 the exact dates are
readable with `report_cli` without re-deriving anything.

**Caveat, stated because it bounds the claim.** Both halves are held out from TRAINING, but the
full validation split is also what checkpoint selection minimized, so both halves inherit the
same selection bias. The transfer comparison (fit on A, score B) is therefore a valid test of
whether the amplitude is stable across time; the absolute post-calibration ratios still carry
selection bias, exactly as every other held-out-full number in this project does. The corpus
reserves a `[70%, 80%)` partition that its schema string calls the calibration partition and
that currently produces no origins (`target_count` returns `None` there) - fitting on that
block instead would remove even the selection-bias caveat, at the cost of a new evaluation
draw, which is an explicit non-goal here. **HYPOTHESIS** that it would change the numbers
materially; **DEMONSTRATED** that the partition exists and is currently unused.

---

## 5. Stamping: how a calibration cannot be silently mispaired

`Pairing` carries eight fields, each of which independently changes what a stored gain MEANS:
checkpoint `format` string (architecture, x0 mode, mean parameterization), training objective,
`SHA-256(model.safetensors)`, `SHA-256(manifest)`, step, `pred_len`, corpus schema string,
corpus universe/boundary digest. `Manifest::pairing(&corpus.contract)` builds it from the
already-authenticated manifest digests, so the calibration inherits the checkpoint's own
integrity guard rather than inventing a second one. The artifact additionally stores
`CALIBRATION_FORMAT` (`...-close-anchor-shrinkage-log-roughness-penalized-wls`, because a curve
fitted on the four decoded channels or allowed above 1 is not interchangeable with this one
even at identical numbers) and `sha256` over its own serialization.

`gain_for(running)` refuses on any mismatch and NAMES the field. `read` verifies the
self-digest first, so a hand-edited curve or a swapped pairing block fails authentication
rather than being applied. Tests:
`a_calibration_is_refused_for_every_checkpoint_corpus_or_horizon_it_was_not_fitted_on`
(mutates each of the eight fields in turn and asserts eight distinct refusals) and
`a_calibration_whose_curve_or_pairing_was_edited_after_the_fit_fails_authentication`.
**DEMONSTRATED.**

The checkpoint is untouched: `fold_mean_gain` multiplies the derived `[1,1,1,pred_len]` `√h`
decode buffer, which is a function of `pred_len` and not a varstore variable, so no saved
tensor changes. Asserted in the model test by comparing every head weight before and after
folding.

---

## 6. Reports, and the invariance rules

Two bases, registered by `SignalHistory`:

- `timexer_segment_calibration_gain` - **written by every run, calibrated or not.** `train` and
  an uncalibrated `evaluate` write the applied gain as exactly 1.0 at all 192 horizons;
  `calibrate` writes its frozen curve plus the fit block's own `β̂` and the untouched block's
  own `β̂`, with both blocks' origin counts, boundary timestamps, purged origins and the fit's
  effective dof in the title. The identity is written as 1.0 and never NaN: 1.0 is a measured
  fact about what scoring multiplied by, while NaN would mean the different thing "no
  calibration was fitted". Asserted by
  `an_uncalibrated_run_writes_a_gain_of_exactly_one_at_every_horizon`, which also asserts the
  fitted case carries 4 series and that a horizon count disagreeing with the artifact is an
  error rather than a silently truncated axis. **DEMONSTRATED.**
- `timexer_segment_calibration_effect` - the untouched block's close and four-channel MSE
  ratios before and after, against that block's OWN best-scale ratio. The oracle series is an
  in-population bound this calibration is not allowed to reach; the gap is the price of having
  fitted out of sample.

**Invariant under a positive per-horizon gain, measured on the real reduction to 1e-4 or
better (DEMONSTRATED):** within-timestamp IC and its SE, pooled Pearson and Spearman, close and
top-decile hit rates, decile returns, conviction spread, and the demeaned gain
`D = Cov²/(Var·E[y²])`. `calibrate` asserts this on the reduction itself and REFUSES the run if
a within-timestamp statistic moves - if a positive scalar reorders one timestamp's tickers, the
implementation is broken and there is no result to report. Test:
`a_positive_per_horizon_gain_moves_mse_and_cannot_move_a_rank_statistic` (gain 0.6 at every
horizon on a synthetic 90×16 population; also asserts the close MSE ratio DOES move, or the
gain was never applied).

**NOT invariant, and named in `docs/timexer_segment.md` so a cross-run comparison cannot be
made blind:** the mid-anchored and first-bar-excluded series of `timexer_segment_tradable` and
`timexer_segment_tradable_rates` are affine in the forecast (`f - mid`, `f_h - f_1`) rather
than rank statistics of it, so a per-horizon gain moves them whenever a bar's forecast crosses
that nonzero reference. **DEMONSTRATED: mid-anchor hit rate at h=1 moves 0.53125 → 0.55208
under a gain of 0.6.** The close-anchored series in those same panels are invariant. Every
quadratic quantity moves by construction (NLL, MSE ratios, the cross term, coverage) - that is
the point of applying a gain.

---

## 7. Exact command lines

Blocks are cut inside one process from one authenticated checkpoint, so there is nothing to
sequence by hand and no way to feed the fit block's numbers into the scoring block.

**(1) Fit the curve on the earlier block, freeze it, and score the later untouched block before
and after - one run, three scoring passes:**

```bash
mlq submit --name timexer-calibrate-4k --max-parallel-runs 1 --time-limit 2h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh calibrate-timexer-segment \
  --checkpoint training/runs/timexer-control-4k/weights/best \
  --calibration training/runs/timexer-control-4k/mean-calibration-step3000.json \
  --output training/runs/timexer-control-4k/gens/2 \
  --batch-size 256
```

Stdout prints, per `h ∈ {1, 8, 16, 32, 64, 128, 192}`: the frozen gain, the fit-block `β̂`, the
untouched-block `β̂`, close ratio before → after, the oracle bound, the four-channel ratio
before → after, and IC before → after. The two report bases land in `gens/2`.

**(2) Read the curves back:**

```bash
./target/release/report_cli 2 timexer_segment_calibration_gain   --run timexer-control-4k
./target/release/report_cli 2 timexer_segment_calibration_effect --run timexer-control-4k
```

**(3) Apply the frozen curve in a normal evaluation (optional, and note this scores the WHOLE
validation split, which the fit block overlaps - the gain panel says so and claims nothing out
of sample):**

```bash
mlq submit --name timexer-eval-calibrated --max-parallel-runs 1 --time-limit 2h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh evaluate-timexer-segment \
  --checkpoint training/runs/timexer-control-4k/weights/best \
  --calibration training/runs/timexer-control-4k/mean-calibration-step3000.json \
  --output training/runs/timexer-control-4k/gens/3
```

Cost of (1): three scoring passes over half-populations = **1.5x one full `evaluate` forward
pass**, no backward, no optimizer, plus one 192×192 Cholesky per penalty grid point on the host
(41 × ~5.9 MFLOP ≈ 0.24 GFLOP, once, ~0.3 s). Zero added per-step device cost anywhere: the
fold is a host-side multiply into an existing `[1,1,1,192]` fp32 buffer (768 B) that every
decode already applies, so 0 extra FLOPs and 0 extra bytes against the 16.98 TFLOP / 143.4 GB
step. Artifact ≈ 20 KB of JSON.

---

## 8. Pre-registered predictions, before the run

Basis for the arithmetic: applying gain `g` to a forecast with true `β` leaves the amplitude
cross term `C(g) = -(β - g)²·Var(f)/P`, so the cost of a gain error is quadratic and locally
tiny. At h=192, `Var(f)/P = |C|/(1-β)² = 0.025192/0.53876 = 0.04676`, so a 10% relative gain
error costs `3.4e-5` of persistence MSE and even a 50% error costs `7.9e-4`. Estimation noise
on a half-sized fit block is therefore NOT the risk; genuine drift of `β` across time is.
Measured `SE(ln β̂) ≈ 0.019` at h=1 and `0.026` at h=192 in population, so ~0.027-0.037 on a
half block, further reduced by the GCV smoother's 39-of-192 dof.

**Fitted gain on the earlier block** (in-population full-split fit in parentheses):

| h | prediction | (full-population fitted) |
|---|---|---|
| 1 | 0.60 ± 0.06 | 0.60023 |
| 8 | 0.73 ± 0.08 | 0.73413 |
| 64 | 0.41 ± 0.06 | 0.41325 |
| 192 | 0.264 ± 0.045 | 0.26395 |

**Close-channel market-neutral MSE ratio on the untouched block, before → after.** The "before"
column carries ±0.010 because the untouched half is a different population than the full split
these came from:

| h | before (predicted) | after (predicted) | in-population bound |
|---|---|---|---|
| 1 | 0.9993 | 0.9987 ± 0.0004 | 0.998700 |
| 8 | 0.9919 | 0.9907 ± 0.0008 | 0.990671 |
| 16 | 0.9902 | 0.9883 ± 0.0010 | 0.988244 |
| 32 | 0.9990 | 0.9890 ± 0.0020 | 0.988492 |
| 64 | 1.0108 | 0.9900 ± 0.0030 | 0.989365 |
| 128 | 1.0170 | 0.9960 ± 0.0040 | 0.995099 |
| 192 | 1.0219 | 0.9970 ± 0.0040 | 0.996694 |

Central prediction, stated as recovery of the in-population improvement (`0.0252` of
persistence MSE at h=192, `0.0215` at h=64): **~99%**, because the arithmetic above says a
5-10% gain error costs `< 1e-4` of ratio, so if `β` is stationary the out-of-sample result is
within noise of the oracle. The single numbers I am committing to are **after ≤ 0.9980 at
h=192 and ≤ 0.9930 at h=64** on the untouched block.

Acceptance floor for calling the transfer successful: **≥ 50% of the in-population improvement
at h ≥ 64**, i.e. after `≤ 1.0093` at h=192 and `≤ 1.0001` at h=64. Below that floor the gain
curve is not carrying the amplitude across the cut even though the fit reproduced its own
block, which is the nonstationarity finding, not a tuning problem.

**Timestamp IC: UNCHANGED at every horizon**, `|ΔIC| < 1e-4`, enforced by refusal. If it moves,
the implementation is wrong and the run aborts rather than reporting.

**Falsifiers, pre-registered.** Any one of these means the over-amplitude is not a stable
per-horizon property and step 4 must NOT be built:
- the untouched block's own `β̂` at h ≥ 64 differs from the fit block's by more than ~3 half-block
  SEs in log space (e.g. h=192 `β̂` outside `[0.17, 0.40]`);
- `after > before` at h=64, 128 or 192;
- the fit refuses because the offset ceiling grew past 0.1 of the amplitude cost on a half block
  (then the parameterization, not the transfer, is what is wrong - report it as such);
- fewer than ~120 of 192 horizons improve.

**Host evidence that the mechanism itself is sound (DEMONSTRATED, synthetic):** a two-block
population built so that `f = a_h·ρ·s`, `y = ρ·s + ε`, with the real measured shape as `1/a_h`,
`ρ = 0.10`, 1,600 rows per block and a deliberate 5% amplitude drift between blocks, run
through the real `Scorer`, real `chronological_blocks`, real `MeanCalibration::fit` and the real
fold:

| h | truth | fit-block β̂ | frozen | close before → after | oracle |
|---|---|---|---|---|---|
| 1 | 0.5714 | 0.7290 | 0.7291 | 0.995612 → 0.990763 | 0.990005 |
| 8 | 0.9596 | 0.9868 | 1.0000 | 0.987338 → 0.987338 | 0.987155 |
| 64 | 0.5366 | 0.7803 | 0.6147 | 0.994828 → 0.988696 | 0.988642 |
| 128 | 0.3462 | 0.3979 | 0.3924 | 1.027343 → 0.990935 | 0.990665 |
| 192 | 0.2532 | 0.3010 | 0.2936 | 1.093095 → 0.995233 | 0.993861 |

169 of 192 horizons improved; worst `|ΔIC| = 1.8e-8`; the 23 that did not are mid-horizons where
the two blocks' realized `β̂` disagreed by more than the smoother could bridge at `ρ = 0.10` and
1,600 rows - one tenth the real population's size. This is evidence about the estimator, not
about the real forecaster's stationarity.

---

## 9. Step 4, pre-registered but NOT built (gated on §7)

Only if §7 transfers. Stated now so it cannot be reverse-fitted to the result.

**Form.** Penalize the predicted mean FUNCTION's cross-sectional energy per horizon, on the
σ-scaled close coordinate `m_{b,h}` the decoder already materializes (i.e. after the `√h`
factor), with the batch mean removed so the penalty attacks amplitude and not level:

```
R = (λ / (2·B·H)) · Σ_b Σ_h w_h · (m_{b,h} - m̄_{·,h})²,   m̄_{·,h} = (1/B) Σ_b m_{b,h}
∂R/∂m_{b,h} = (λ/(B·H)) · w_h · (m_{b,h} - m̄_{·,h})
```

`w_h` is the objective's existing horizon weight, so the prior cannot silently re-weight the
horizon axis. This is a penalty on OUTPUT VALUES, which is why the network cannot undo it the
way it undoes a fixed output multiplier: growing upstream weights increases `R` proportionally.
Removing the mean is what keeps it from fighting the (tiny, `4.6e-5`) offset opportunity.

**Cost.** Per step at `B=256`, `H=192`, one channel: 49,152 elements. One mean-reduce, one
subtract, one square-sum, and in backward one fused elementwise add into the already-existing
mean gradient: ≈ 0.25 MFLOP and ≈ 0.6 MB of traffic, against 16.98 TFLOP and 143.4 GB. That is
`1.5e-8` of the step's FLOPs and `4e-6` of its traffic; no new activation is retained (the mean
tensor is already live for the NLL), so peak memory is unchanged. Predicted step-time effect:
**below measurement noise** (< 0.05 ms on a 165-174 ms device step). If a measured step time
moves by more than 1 ms, the implementation allocated something it should not have.

**Pre-registered predicted effects at h=192, held-out full, one λ tuned on a single short run:**
- `C`: `-0.0252 → between -0.006 and -0.001` (the whole point).
- `D`: **unchanged within ±20%**, because `D = Cov²/(Var·E[y²])` is scale-free - shrinkage
  cannot move it except by changing the learned DIRECTION. If `D` collapses the way it did
  under `basis:8:8` (`0.00481 → 0.00116`), the prior is destroying signal and must be rejected.
- timestamp IC at h=64/128/192: **unchanged within 1 iid SE** (`±0.0028`).
- The post-hoc calibration fitted on the penalized model: **gain curve rises toward 0.8-1.0**.
  That is the cleanest falsifier available - if the training-time prior works, the post-hoc
  calibration it leaves behind is nearly the identity, and the two mechanisms measure each
  other.
- Close ratio at h=192: `1.0219 → 0.995-0.999` WITHOUT any post-hoc gain.
