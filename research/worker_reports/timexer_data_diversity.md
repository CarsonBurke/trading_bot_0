# Data-diversity knobs and the supervision-occupancy series

Scope: land the knobs that let occupancy saturation be TESTED rather than assumed, and make it a
per-step observation instead of arithmetic done afterwards. No architecture change, no objective
change, no new evaluation draw, no GPU job submitted from here.

Everything below tagged **DEMONSTRATED** was measured on the real 4,873-ticker corpus by this
build; **HYPOTHESIS** is pre-registered prediction and is labelled as such.

---

## 0. The correction that reshaped this ticket

The original brief's primary intervention - `--row-fraction 0.25`, expected to move saturation
from step 2000 to step 500 - **could not have worked**, and this is DEMONSTRATED twice over
(`PeakCause` algebraically, this build numerically).

`n` rows drawn from a uniformly random `F·R`-subset of `R` are still a uniformly random
`n`-subset of `R`. So for any fixed outcome with `M` supporting rows,

```
P[unseen after n rows] = C(R - M, n) / C(R, n)
```

is **independent of `F`** at a matched step: the `F`-fold cut in per-outcome multiplicity cancels
the `F`-fold cut in the pool exactly. Pinned as a test
(`uniform_thinning_leaves_matched_step_coverage_invariant`), which marginalizes over the real
`Binomial(30, 1/4)` retention law rather than over its mean - `E[q^M] ≠ q^{E[M]}`, and the mean
surrogate is off by 6e-3, thirty times the tolerance and large enough to have been misread as a
real occupancy effect.

`--row-fraction` therefore SHIPS, demoted to a **negative control**: it changes the pool without
changing matched-step occupancy, so any effect it has is not an occupancy effect. The primary
intervention is now `--row-stride-multiple`.

Two further wording corrections adopted throughout: the 99.91% figure is **interior** coverage,
not global (global at step 2000 is 98.4870%); and `--patch-phase random` **does** change origin
support - it adds origins, not shocks, so the support changes and the information ceiling does
not.

---

## 1. What the three knobs mean

All three are `TrainArgs` flags. All three default to today's behaviour. All three are stamped in
the run **manifest** (`Manifest::row_selection`, `skip_serializing_if = "Option::is_none"`) and
in **no** corpus artifact - see §5.

### `--row-stride-multiple K` (default 1) - the intervention

Keep every `K`-th training row of each ticker's chronological grid, with a **per-ticker residue**
in `0..K-1` drawn from `--seed`, so the retained lattice is not phase-aligned across the
universe.

This is the only one of the three that moves matched-step occupancy, because it thins the
**overlap** without thinning the **support**. Rows are enumerated at stride `pred_len = 192`
while each row supervises a dense origin lattice reaching `window = 5744` bars back, so an
interior outcome is supervised by `5744/192 + 1 = 30` distinct rows per epoch. `K` divides that.

DEMONSTRATED on the real corpus, `--seed 20260905`:

| arm | retained rows | steps/epoch @256 | distinct h=1 outcomes | mean exposures/sweep | multiplicity histogram |
|---|---|---|---|---|---|
| control (`K=1`) | 2,455,276 | 9,590 | 31,159,116 | 28.36728 | 27,783,348 at `M=30` (89.166%), edge ramps 19..29 at ~115k each |
| `K=4` | 613,830 | 2,397 | 30,981,408 | 7.13263 | 14,265,744 at `M=7`, 13,920,720 at `M=8`, ramps 1..6 at ~465k each |
| `K=30` | 81,893 | 319 | 29,481,480 | 1.00000 | 29,481,480 at `M=1`, exactly - nothing else |

Read the third column: `K=30` cuts the pool 30x and the outcome support by only 5.4%. That is the
whole point, and it is what `--row-fraction` cannot do.

`K=4`'s split between `M=7` and `M=8` is exactly what the two distinct origin phases predict
(`5744/768 = 7.479`), and it landed 50/50 as predicted.

### `--row-fraction F` (default 1) - the negative control

Keep a uniformly random subset of the whole pool, of **exact** size `round(F·R)` (a partial
Fisher-Yates, not a per-row coin flip, so the retained count is a function of `F` alone).

DEMONSTRATED, `F = 0.25`: 613,819 rows, 30,799,596 outcomes, mean exposures 7.17460. Note the
comparison with `K=4` at almost the same row count: the same pool size, the same mean
multiplicity, and a **1.15% smaller support** - because a uniform draw loses an outcome outright
whenever all of its supporting rows are dropped, while a stride never does.

### `--patch-phase <fixed|random>` (default `fixed`) - augmentation

`fixed` is today's corpus: training origins are `common_context - 1 + i·192`, and `5999 mod 16 =
15` with `192 = 12·16`, so **every** supervised absolute origin in the entire corpus is congruent
to 15 mod 16 and all 30 exposures of an outcome carry an IDENTICAL tokenization - same patch
boundaries, same RoPE positions, same token count before the origin.

`random` draws one offset in `0..patch_len` per row and shifts that row's final origin forward by
it.

DEMONSTRATED: 2,455,105 rows retained of 2,455,276 (171 dropped, 0.007%), support
31,159,116 -> **410,751,324** (13.2x), mean exposures 28.36728 -> **2.15176**, and the retained
timestamp decile drift is 0.00004.

It **adds origins, not shocks.** The phased lattice covers 13.2x as many absolute origins, but
the corpus's information ceiling is the number of distinct realized return shocks and no
re-tokenization raises it: for a phased origin `A` with `r = (A - 15) mod 16`,
`cum(A → A+h) = cum(A-r → A+h) - cum(A-r → A)`, and both right-hand terms are supervised from
the lattice origin `A-r`. So the support changes and the ceiling does not. That is precisely why
this is a regularizer against tokenization-specific memorization rather than a data fix.

---

## 2. The phase-within-row verdict: per-row, and it is not a choice

**Per-row is the only representable granularity, so "must the phase be constant within a row" is
not a design question - a per-origin phase does not exist as an object.** DEMONSTRATED from the
model's own geometry:

A row is patchified ONCE. `CausalPatchModel::tokens` reshapes the row's `seq_len = 6000` context
into `origins = seq_len / patch_len = 375` tokens of 16 bars, embeds them through one `patch`
projection, applies RoPE at the token index, and runs causal SDPA with `is_causal = true` over
that single sequence. The row's 375 causal origins **are the token positions of that one
sequence**; they are not 375 independent tokenizations that could each carry a phase. Giving
origin `j` a different phase from origin `j+1` would require the same row to hold two different
patch partitions of the same bars simultaneously, two different RoPE position assignments, and
therefore two different `is_causal` masks. There is nothing to implement.

Consequently the phase is realizable as a **pure shift of the row's final origin**: the row's
patch grid, its RoPE positions and its `is_causal` mask are all defined relative to its own
context start, so `O -> O + p` shifts every one of its 360 active absolute origins by `p` and
changes nothing else. `fill_row`, the aux cursor, the market lookup and every kernel are
untouched - the knob lives entirely in origin enumeration.

**Does the eligible-origin population change? YES**, and the direction matters:

* the supervised absolute-origin population changes (13.2x, measured above), and the residue
  class it occupies goes from one value to all 16 - pinned by test;
* the shift is **forward**, so target ownership can only be lost at a ticker's tail, never
  context at its head: 171 rows of 2.46M dropped;
* every retained phased origin is re-tested by `Corpus::training_target_count`, which spells the
  training band out itself rather than delegating to `CorpusTicker::target_count`. That
  distinction is load-bearing: `target_count` also admits calibration and validation origins, so
  a forward shift past `train_end` would have been silently accepted as a held-out origin and the
  arm would have trained on the reserved `[70%, 80%)` partition the amplitude fit is defined on.
  Pinned by test (`for reference in &corpus.train_refs { training_target_count(..).unwrap() }`);
* the same accessor applies `TemporalSplit`'s `supervision_clears_hole`, so `--patch-phase random`
  composes with `--in-period-sections` and a phased row whose dense supervision reaches the hole
  is DROPPED rather than leaking. Pinned by test, bar by bar, against the conservative supervised
  interval `[O - context + 2, O + pred_len]`.

**No corpus artifact invalidates**, and that is correct rather than convenient - see §5.

---

## 3. Loader cost, honestly

The loop period is `max(device, host)`, so host work is free only while it stays under the device
time. Basis: 24.197 ms host batch assembly budget, 215.7 ms/step observed under contention,
16.98 TFLOP and ~143 GB per step.

**Steady-state (per step): zero added cost, DEMONSTRATED.** Release build, 5 timed rounds after a
warmup, real corpus, batch 256, on the arm's own retained refs:

| arm | host batch assembly |
|---|---|
| control | 19.507 ms (second run 20.119 ms) |
| `--patch-phase random` | **18.350 ms** (second run 21.982 ms) |
| `--row-stride-multiple 30` | 19.133 ms (second run 23.062 ms) |
| `--row-stride-multiple 4` | 21.904 ms (second run 22.415 ms) |
| `--row-fraction 0.25` | 20.618 ms (second run 19.907 ms) |

Run-to-run spread is ±2.6 ms under contention from concurrent builds, and the phase arm lands
*below* the control in one run and above in the other, i.e. **within noise and with no systematic
component**. This is the expected result, not a lucky one: the phase is an integer added to an
origin at enumeration time, and `fill_row` walks the same number of bars whatever the origin is.

`K=4` and `F=0.25` sit ~1-2 ms high for a different and real reason: the first 256 refs of a
thinned pool span 4x more bars, so the mapped-file page locality of the gather is worse. It is
under 10% of a 24.197 ms budget and under 1% of a 215.7 ms step, so the loop period does not move.

Model FLOPs, bytes and parameters: **exactly zero change**. No tensor, no kernel, no shape, no
precision is touched by any of the three knobs. The `[3, 375, 4, 192]` target geometry, the
CUDA-graph capture shape and the batch size are all unchanged.

**One-time (startup): 1.9 s, charged to its own series.** `Corpus::load` -> `supervision::select`
costs 1.888 s for the control and 1.883 s for the phase arm on a quiet machine (3.27 s / 8.43 s
under heavy concurrent builds). It is two `par_iter` timestamp passes for the decile profile plus
an `O(rows log rows)` interval sweep for the histogram - explicitly NOT `O(origins)`, which is
what makes a 410-million-origin lattice affordable. Against a ~30 s warm startup that is ~6%, and
against a 2,500-step run's ~540 s it is 0.35%. It is billed as
`row selection and supervision census` on `timexer_segment_startup`, so a regression in it is a
line that moves.

---

## 4. The occupancy series

One new base, `timexer_segment_supervision_occupancy`, registered in
`shared/src/report.rs` (the TUI extends `meta_chart_bases` from that slice, so no TUI edit; the
bidirectional registry test covers it and is green).

* **Question:** how much of its own training signal has this arm consumed, per optimizer step?
* **Axis / reading rule:** `share of this arm's distinct supervised (origin, horizon) outcomes;
  1.0 is every outcome touched, and the step where the 1-pass curve flattens is the saturation
  step`. No `=` in the y-label or in any series label (pinned by test).
* **Series:** outcomes with at least 1, 2, 4, 8 and 16 gradient passes, plus
  `mean gradient passes per outcome, as a share of the arm's mean multiplicity`.
* **Why five curves instead of a mean:** they are all shares of one denominator, so one axis
  carries all of them, and they are strictly more informative than the mean - the 1-pass curve is
  saturation and the higher ones are how much of the budget after it went into re-presentation.
  The mean exposure COUNT is the last series times the mean multiplicity, which the title states,
  so a count never lands on an axis of fractions.
* **Everything is computed in closed form from the MEASURED histogram.** No constant from any
  brief is hardcoded. `P[X = 0] = Π_{j<M} (R-n-j)/(R-j)` is `M ≤ 360` terms with no factorial and
  no `lgamma`; the tail follows from the standard ratio recurrence. Multi-epoch is exact, not
  approximated: `e` complete shuffles give every outcome its full `M` and only the partial epoch
  is hypergeometric - which matters the moment a thinned pool sweeps more than once, i.e. in arm
  (b).
* Unmeasured reads **NaN, never 0**: `interior_share` and `interior_coverage` both return NaN when
  an arm has no outcome at the geometry's interior multiplicity, which is the case for every
  thinned arm.

**Cross-validation, DEMONSTRATED.** The implementation reproduces `PeakCause`'s independent
arithmetic to six significant figures on every quantity they published: support 31,159,116;
`M=30` count 27,783,348 (0.89166 share); mean multiplicity 28.36728; global 1-pass coverage at
step 2000 = 0.984870; interior coverage at step 2000 = 0.999103, i.e. unseen 8.97e-4. Two
independent derivations agreeing to six digits is the strongest available check on both.

**And it sharpens the headline.** The control's held-out IC peak at step 2000 coincides with
**interior** saturation at step 1972.4 - not with global saturation, which is step 2646 at the
0.99 threshold and step 7597 at 0.999. That is a materially different and much more falsifiable
statement than "the model runs out of data at 2000", and it is now readable off a chart.

---

## 5. Cache invalidation: nothing invalidates, and that is the correct answer

The rule is that a knob changing the eligible universe, the market grid or the origin population
must invalidate the affected artifact rather than silently reuse it. Worked through:

* **What is cached:** per-ticker bar audits, keyed on `BarIdentity{device, inode, size, mtime_ns,
  ctime_ns}` of the mapped descriptor; the shared partition boundaries; and the market grid, keyed
  on the ordered eligible-ticker fingerprints plus grid geometry plus `min_cross_section`.
* **What these knobs touch:** `Corpus::train_refs`, and nothing else. They run AFTER
  `Corpus::load` has built every contract, the market grid, the exogenous series and all four
  held-out populations. Ticker eligibility, bar bytes, partition boundaries and the market
  construction are all upstream of them and provably unreached.
* **Therefore no cached artifact is a function of them**, and stamping them in `CorpusContract`
  would have forced a cold 172 s rescan for a change no artifact depends on - and would have
  broken `corpus.rs`'s exact-18-key contract test, which exists to keep every authenticated
  checkpoint manifest on disk loadable.
* **They are stamped in the run manifest instead**, which is where a training configuration
  belongs, with the seed alongside so an arm reproduces and a pairing claiming the same knobs but
  a different pool is detectable. `Option` + `skip_serializing_if`, so a control manifest and its
  digest are byte-identical to one written before these knobs existed.
* **The existing rejection tests stay green:**
  `a_cache_built_against_one_corpus_contract_is_rejected_when_the_contract_changes` and the
  no-bar-audit assertions pass (`timexer_segment::corpus` 10 passed / 1 ignored inside the 166).

Held-out populations are untouched by construction and by test: the knobs mutate only
`train_refs`, and the `held-out sample` (persistence NLL 2.3947663), `held-out cross-section`,
`held-out full`, the `[70%, 80%)` calibration partition and `in_period_refs` are all built from
the other four reference lists.

---

## 6. Bit-exactness of the defaults, DEMONSTRATED

Test `the_default_selection_is_bit_exact_and_each_knob_moves_only_what_it_claims`, on a real
`Corpus::load` over a three-ticker scratch corpus (one ticker with quarantined OHLC so valid-bar
ordinals differ from raw indices):

* `--row-fraction 1 --patch-phase fixed --row-stride-multiple 1` leaves `train_refs` equal
  **element for element** to the enumerated pool. `select` takes no random draw at all under the
  identity, which is what makes this provable rather than probable.
* The serialized `CorpusContract` is **byte-identical** before and after.
* `Corpus::host_batch` over a fixed 37-row probe is bit-identical in **all five** tensors -
  `log_prices`, `valid`, `aux`, `market_cum`, `anchor` - plus `valid_target_bars`. The `valid`
  comparison is the one that matters: a changed mask is a changed dataset, not a rounding
  difference, and comparing the reference list alone would not have seen one.
* `phase_dropped_rows == 0` and `decile_drift() == 0.0` exactly.

The fixture geometry is `seq_len 64, patch_len 4, min_history 20, pred_len 8` giving `window 44`
and interior multiplicity 6, chosen so that `44 mod 8 == 8 - 4` and **every** interior origin
carries 6 - the same uniformity the production geometry has (`5744 mod 192 == 192 - 16`). That
makes the fixture structurally like production rather than merely smaller. The uniformity
condition is itself pinned (`has_uniform_interior`), including a counterexample.

## 7. Regime mix, DEMONSTRATED

The retained rows' timestamp decile profile against the full pool's, measured on the real corpus.
The full pool is NOT uniform in time - the corpus is denser in recent history - so the check is
tracking, not flatness:

```
full pool     0.05518 0.07945 0.08654 0.08935 0.09510 0.10786 0.11229 0.12407 0.12734 0.12281
K=30          0.05819 0.07958 0.08551 0.08913 0.09526 0.10790 0.11193 0.12200 0.12730 0.12321
K=4           0.05608 0.07978 0.08633 0.08834 0.09592 0.10773 0.11207 0.12362 0.12736 0.12276
phase random  0.05514 0.07946 0.08654 0.08936 0.09514 0.10782 0.11232 0.12405 0.12735 0.12282
F=0.25        0.05503 0.07949 0.08694 0.08878 0.09472 0.10801 0.11218 0.12410 0.12782 0.12294
```

Max absolute decile deviation: `K=30` 0.00300, `K=4` 0.00101, phase random 0.00004, `F=0.25`
0.00058. A chronological slice would drive this to ~0.1 per emptied decile, i.e. by a factor of
33 to 2,500. The statistic is exposed as `SupervisionCensus::decile_drift`, printed at startup and
carried in the chart title, so it is checked on every future arm rather than once here.

---

## 8. The three arms, with numbers pre-registered BEFORE any run exists

All three: `--time-limit 10m`, `--data-dir /var/tmp/tb0_v17`, and `--schedule-budget 9590` so
each is the first `N` steps of the SAME learning-rate trajectory job 5428 ran (realized LR flat at
0.039837 until step 3836, so no schedule event lands inside any of these budgets and a peak cannot
be a schedule artifact).

The corpus directory is `v17` and not the `v13` this ticket was written against: `v13` was BARRED
after its fused op crashed on the real path, `v14` never materialized because the release build
failed before the `cp`, and `v15`/`v16` both predate current work. `v17` or later is the standing
instruction. Nothing in the pre-registration below depends on which directory it is - the corpus
CONTENT is the same 4,873-ticker universe and every census number was measured against it - but a
stale directory would pay a 172 s cold load and blow the 10-minute lease, which is the whole
reason the bar exists.

### (a) Control, 2,500 steps

```
mlq submit --name timexer-occupancy-control-2500 --max-parallel-runs 1 --time-limit 10m \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment \
  --run timexer-occupancy-control-2500-20260907 \
  --data-dir /var/tmp/tb0_v17 \
  --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 --min-history 256 \
  --batch-size 256 --optimizer polar-express --preview-patience 3 --x0-lambdas disabled \
  --horizon-loss inv-sqrt \
  --eval-every 500 --schedule-budget 9590 --max-steps 2500
```

### (b) The intervention: `--row-stride-multiple 30`, 2,300 steps

2,300 and not 320: the pool is 81,893 rows, so an epoch is 319 steps and 2,300 steps is 7.2
sweeps. A 320-step stop would observe a threshold crossing, not a peak, and could not establish
one.

```
mlq submit --name timexer-occupancy-stride30-2300 --max-parallel-runs 1 --time-limit 10m \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment \
  --run timexer-occupancy-stride30-2300-20260907 \
  --data-dir /var/tmp/tb0_v17 \
  --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 --min-history 256 \
  --batch-size 256 --optimizer polar-express --preview-patience 3 --x0-lambdas disabled \
  --horizon-loss inv-sqrt \
  --row-stride-multiple 30 --epochs 10 \
  --eval-every 250 --schedule-budget 9590 --max-steps 2300
```

`--epochs 10` because the epoch is 319 steps; without it the run stops at one sweep.

### (c) Augmentation: `--patch-phase random`, 2,500 steps

```
mlq submit --name timexer-occupancy-phase-2500 --max-parallel-runs 1 --time-limit 10m \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment \
  --run timexer-occupancy-phase-2500-20260907 \
  --data-dir /var/tmp/tb0_v17 \
  --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 --min-history 256 \
  --batch-size 256 --optimizer polar-express --preview-patience 3 --x0-lambdas disabled \
  --horizon-loss inv-sqrt \
  --patch-phase random \
  --eval-every 500 --schedule-budget 9590 --max-steps 2500
```

### (d) Optional negative control: `--row-fraction 0.25`, 2,000 steps

Worth a slot precisely because it is invariant. Run it only if the queue allows.

```
mlq submit --name timexer-occupancy-fraction25-2000 --max-parallel-runs 1 --time-limit 10m \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment \
  --run timexer-occupancy-fraction25-2000-20260907 \
  --data-dir /var/tmp/tb0_v17 \
  --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 --min-history 256 \
  --batch-size 256 --optimizer polar-express --preview-patience 3 --x0-lambdas disabled \
  --horizon-loss inv-sqrt \
  --row-fraction 0.25 --epochs 4 \
  --eval-every 500 --schedule-budget 9590 --max-steps 2000
```

### Pre-registered numbers

Occupancy figures are **DEMONSTRATED** - computed by this build from the real corpus before any
arm runs. "Saturation step" is the step at which global 1-pass coverage first reaches **0.984870**,
the control's coverage at its own observed peak; this is the cross-arm comparable, because it asks
each arm to reach the same occupancy state rather than the same step. IC and step-ms rows are
**HYPOTHESIS**.

| | (a) control | (b) `K=30` | (c) phase random | (d) `F=0.25` |
|---|---|---|---|---|
| retained rows | 2,455,276 | 81,893 | 2,455,105 | 613,819 |
| steps/epoch | 9,590 | 319 | 9,590 | 2,397 |
| distinct outcomes | 31,159,116 | 29,481,480 | 410,751,324 | 30,799,596 |
| mean exposures/sweep | 28.36728 | 1.00000 | 2.15176 | 7.17460 |
| **saturation step** (cover1 0.98487) | **2,001** | **316** | **9,195, i.e. NOT reached** | **1,474** |
| cover1 at 0.99 / 0.999 | 2,646 / 7,597 | 317 / 320 | 9,326 / 9,564 | 1,638 / 2,265 |
| cover1 at final step | 0.98916 @2500 | 1.00000 @2300 | 0.42005 @2300, 0.44877 @2500 | 0.99638 @2000 |
| **IC peak step** | 2,000 | **~316** | none inside 2,500 | 2,000 |
| **peak IC h=16** | 0.114 | 0.114 if redundancy, ≤0.085 if not | ≥0.10 at 2,500 | 0.114 |
| **peak IC h=64** | 0.088 | 0.088 if redundancy, ≤0.060 if not | ≥0.060 at 2,500 | 0.088 |
| **step ms** | 216 ± 5 | 216 ± 5 | 216 ± 5 | 216 ± 5 |

Anchors the IC numbers are quoted against, from job 5428 / `timexer-invsqrt-ic-6k` on `held-out
cross-section`, close channel, at step 2000: h=16 0.1137, h=64 0.0879. Step-ms is 216 ± 5 for every
arm because no arm touches the model or the loop period; if the verified fused loss lands first,
all four become 155 ± 5.

**IC SE correction, and it retires one of my own thresholds.** I had quoted the draw's IC standard
error as ~0.011, from a ~40-names x ~100-timestamps reading. The measured number is
**~0.0164** - three independent arrivals at the same census now agree that of 40,837 distinct
`held-out cross-section` timestamps exactly **100** hold the 40 names `CROSS_SECTION_FLOOR`
requires, so the draw is 100 x 40 = 4,000 windows, not the wider block I assumed. Applying Main's
arithmetic to my own table:

| my criterion | band | in units of 0.0164 | verdict |
|---|---|---|---|
| peak LOCATION, ~316 vs ~2,000 | 6.3x in step index | no SE needed | **DECIDABLE** |
| occupancy saturation step | census arithmetic | no SE needed | **DECIDABLE** |
| (b) skill holds vs collapses, h=16 0.114 vs ≤0.085 | 0.029 | **1.8 SE unpaired** | **MARGINAL** |
| (b) skill holds vs collapses, h=64 0.088 vs ≤0.060 | 0.028 | **1.7 SE unpaired** | **MARGINAL** |
| (c) h=16 ≥ 0.10 at 2,500 | 0.014 below anchor | **0.85 SE** | **RETRACTED, not a test** |
| (c) h=64 ≥ 0.060 at 2,500 | 0.028 below anchor | 1.7 SE | MARGINAL |
| (d) peak at ~2,000 vs ~1,474 | 526 in step index | no SE needed | **DECIDABLE** |

So I retract `(c) h=16 ≥ 0.10` as a decision criterion outright - 0.85 SE is not a test and I will
not have it read as one - and I demote both of (b)'s magnitude thresholds to MARGINAL, decidable
only on the STEP-MATCHED PAIRED difference against the control, where the draw-level component is
common and cancels. Every arm here scores the same 4,000 windows at the same steps with the same
`--eval-every`, so the paired difference is available at no cost and is the form I will read.

**What survives unconditionally, and it is the whole point of the ticket:** this batch's primary
and secondary criteria are both STEP LOCATIONS - where the occupancy curve flattens and where the
IC peaks - and a peak location is read off a curve of ~5-10 evaluation points, not off a
magnitude difference. Confirming Main's ruling explicitly: **the occupancy result is unaffected by
the IC SE problem and by the `band` defect.** Every census number in the table above is exact
combinatorics over the retained row pool - row counts, outcome support, multiplicity histograms,
hypergeometric coverage - and touches neither the cross-section draw nor any IC estimate. Arm (b)
moving the saturation step from 2,001 to 316 is arithmetic that is already true.

**But one dependency IS live and I am naming it rather than leaving it implicit.** The peak
LOCATION is read from an IC curve, and while a location needs no SE, a location can still be
obscured if consecutive evaluation points are within noise of each other. At SE 0.0164 the
control's step-1000-to-2000 rise at h=16 (0.0738 -> 0.1137, i.e. 0.0399) is 2.4 unpaired SE and
survives; its h=64 rise (0.0862 -> 0.0879) is 0.1 SE and does not, which is why the h=64 peak
location was never my primary read. After the `band` fix takes the draw to 32,768 windows at
SE ~.0035, the same rises become 11.4 SE and 0.5 SE - so `band` does not rescue h=64's peak
location either, and the honest statement is that **h=16 carries the peak-location read and h=64
does not, before or after `band`**. Arm (b)'s 6.3x displacement is far larger than the
`--eval-every 250` grid it is measured on, so it is legible at either SE.

Timing margin, stated because it is tight: 2,500 steps at 215.7 ms is 539.3 s, plus ~30 s warm
startup, plus 1.9 s census, plus five held-out sample evaluations - which is 9.6-9.9 min against
the 10 min limit. **If the harness measurement reports above 216 ms/step, drop (a) and (c) to
2,300 steps rather than risk a kill.** At the fused loss's 155.1 ms/step, 2,500 steps is 388 s and
the margin is comfortable.

---

## 9. The reading rule, written down before any run exists

**On arm (b), `--row-stride-multiple 30`, which is the decisive one.** All three branches turn on
the PRIMARY criterion, peak location, which needs no standard error. The magnitude clause in
branches 1 and 3 is stated as a STEP-MATCHED PAIRED difference against the control at the same
optimizer step, because at the draw's measured SE of 0.0164 an unpaired 0.029 band is only
1.8 SE and would not decide anything on its own.

1. Peak at **~316 steps** while the paired h=16 difference against the control's peak is **within
   ±0.015** (and h=64 within ±0.015) → **occupancy saturation is DEMONSTRATED as the mechanism.**
   The intervention moved the saturation step by 6.3x and the peak followed it by the same factor,
   at unchanged peak skill. That is the coincidence broken.
2. Peak stays at **~2,000 steps** → **the hypothesis is REFUTED, and it is written down here in
   advance that it is refuted.** The saturation step moved 6.3x and the peak did not, so whatever
   sets the peak at 2000 is not the exhaustion of distinct supervised outcomes. Candidates then:
   the optimizer's own trajectory length at flat LR, or a curvature/interference effect indexed by
   step rather than by data. This branch needs no magnitude reading at all, which is why it is the
   one branch that is fully decidable today.
3. Peak at ~316 but the paired h=16 difference is **below -0.029 (1.8 SE unpaired, and the paired
   difference is what I will read)** → the binding constraint is **data volume rather than
   redundancy**, and stated with the caveat this deserves: a lower peak on a 30x-thinned arm does
   **not** uniquely prove a volume constraint, because block structure, regime composition,
   context diversity and boundary support all differ too. What arm (b) answers CLEANLY is
   narrower and worth having on its own: **do the 30 redundant exposures - which carry different
   causal σ, β and context - do real augmentation work, or do they only consume budget?**
   Outcome 1 says they only consume budget; outcome 3 says they do work.

If the paired difference lands between those two bands - a peak at ~316 with skill down by
0.015 to 0.029 - the correct report is **INCONCLUSIVE ON MAGNITUDE, MECHANISM CONFIRMED ON
LOCATION**, and the arm is re-read after `band` takes the draw to SE ~.0035, where the same 0.029
becomes 8 SE. I am writing that third outcome down now so nobody resolves an ambiguous magnitude
in the direction they were hoping for.

**On arm (c), `--patch-phase random`.** Its saturation step is 9,195 - outside the budget by 3.7x
- so under the occupancy hypothesis it should show **no peak at all inside 2,500 steps** and
improve monotonically to the cap. A peak at ~2,000 on this arm, where occupancy is at 0.375 and
nowhere near saturating, is direct evidence AGAINST occupancy and is the single cheapest
falsifier in the batch. Note also that (c) is the only arm that changes the occupancy DENOMINATOR
(13.2x more origins), so its saturation step is comparable across arms while its support is not.

**On arm (d), `--row-fraction 0.25`.** Occupancy over the FULL support is provably invariant, so
under the hypothesis it peaks at **~2,000**, exactly like the control, despite a 4x smaller pool
and 4x fewer exposures. Its own-support saturation is 1,474 rather than 2,001, and the whole of
that 527-step difference is the 1.15% of support a uniform draw destroys outright - a denominator
effect, not an occupancy effect. So: peak at ~2,000 confirms the control is behaving as the
algebra says; a peak at ~1,474 would say the lost 1.15% of support matters more than its size
suggests; and a peak near 2,398 would be an epoch-boundary effect, since that is where this arm
first wraps.

**A joint reading that costs nothing extra.** (b) and (d) sit at almost the same row count
(81,893 vs 613,819 differ, but (d) and `K=4` do not) and at very different multiplicities, while
(c) and (a) sit at the same row count and very different multiplicities. Between them the four
arms separate ROW COUNT from PER-OUTCOME MULTIPLICITY from TOKENIZATION DIVERSITY, which no single
arm can.

**What none of these arms establishes.** `K=30` at 2,300 steps against the control at 2,000 is an
APPROXIMATE matched contrast, not a clean one: first-exposure order, the occupancy trajectory, the
dispersion of per-outcome counts and the boundary support all differ, not only context diversity.
Nobody may write it up as isolating context diversity alone.

---

## 10. Files touched

| file | change |
|---|---|
| `trading_bots/src/torch/timexer_segment/supervision.rs` | NEW. `RowSelection`, `PatchPhase`, `SupervisionGeometry`, `SupervisionCensus`, `select`, the multiplicity-histogram sweep, the hypergeometric closed form, 10 tests |
| `trading_bots/src/torch/timexer_segment/mod.rs` | `pub mod supervision;` |
| `trading_bots/src/torch/timexer_segment/corpus.rs` | one additive `Corpus::training_target_count` |
| `trading_bots/src/torch/timexer_segment/cache.rs` | `LoadTiming::row_selection_ms` + its phase row |
| `trading_bots/src/torch/timexer_segment/reports.rs` | `write_supervision_occupancy` + its test, appended at EOF |
| `shared/src/report.rs` | one base, `timexer_segment_supervision_occupancy` |
| `trading_bots/src/torch/timexer_segment/runner.rs` | 3 flags + `row_selection()` + `validate` + `Default`; the `select` call and census print; `Manifest::row_selection`; one occupancy write per report interval |

No backward-compatibility shim, no alias, no dead flag, no fallback path.

## 11. Verification

* `./torch-env.sh cargo check -p trading_bot_0 --tests` - **0 errors**
* `./torch-env.sh cargo check -p trading-bot-tui --tests` - **0 errors**
* `./torch-env.sh cargo test -p trading_bot_0 timexer_segment` - **166 passed, 0 failed**
* `./torch-env.sh cargo test -p trading-bot-tui` - **36 passed** (bidirectional registry test)
* `./torch-env.sh cargo test -p shared` - **19 passed** (base registry)
* real-corpus census and loader timings measured in release through a throwaway harness, since
  removed; every number it produced is recorded above.
* re-verified after every sibling landing: `cargo test -p trading_bot_0 --lib
  timexer_segment::supervision` - **9 passed, 0 failed** (the 10th, `write_supervision_occupancy`'s
  panel test, lives in `reports.rs` and is inside the 166).

## 12. Status at session end

**LANDED AND GREEN. Nothing half-landed, nothing reverted, no stub, no uncompilable state left
behind.** All three flags, `select`, the census, the occupancy writer, the report base and 10
tests are complete and were verified after each sibling landing. The last scoped run - after
`FutureTeacher`'s `increment_pair` and `TemporalSplit`'s contract fields landed - is 9/9 on
`timexer_segment::supervision`.

**NOT RUN: all four arms.** No GPU job was submitted from here; subagents may not. The submit
lines in §8 are ready to paste and every number they will be checked against already exists.

**Prediction status after today's corrections:**

| prediction | status |
|---|---|
| `--row-fraction 0.25` moves the saturation step to ~500 | **RETRACTED before implementation.** Provably invariant; pinned as a test; knob shipped as a NEGATIVE CONTROL |
| `--patch-phase random` is augmentation that does not change support | **RETRACTED, wording wrong.** It adds ORIGINS (13.2x, measured) and no shocks; support changes, the information ceiling does not |
| interior coverage 99.91% at step 2000 | **STANDS**, 0.999103 - but it is INTERIOR, not global. Global at 2000 is 0.984870. My earlier global claim is retracted |
| draw IC SE ~0.011 | **RETRACTED, mine.** Measured ~0.0164 (100 usable timestamps x 40 names) |
| `(c) h=16 >= 0.10 at 2500` as a decision criterion | **RETRACTED.** 0.85 SE is not a test |
| `(b)` skill-holds thresholds as unpaired levels | **DEMOTED to paired step-matched differences**; unpaired form is 1.7-1.8 SE and would not decide |
| peak location moves 2,001 -> 316 under `K=30` | **OPEN, and it is the decisive one.** Needs no SE, unaffected by `band` |
| step ms unchanged by all three knobs | **OPEN**, predicted 216 ± 5; host-side cost already measured as zero |

**Dependency list on other agents' numbers.** Exactly one, and not load-bearing: the step-2000 IC
anchors from job 5428 (`held-out cross-section`, close, h=16 0.1137, h=64 0.0879) as the level the
paired differences are quoted against - if re-measured, the differences move with them and no
criterion changes shape. I use the 100-usable-timestamp census (three independent arrivals) for
the SE. I use NONE of: 1.70x/1.3805x amplitude share, the monotone calendar drift, the h=32 sign
inversion, `rho_1`, `V_h/V_1`, the plateau ceiling, or any basis/increment number. The occupancy
census depends on the corpus geometry alone.

**Consumed prerequisites:** `TemporalSplit`'s `supervision_clears_hole` (composition with
`--in-period-sections`, tested bar by bar) and `Corpus::training_target_count`, which I added
beside it. If `band` changes origin PLACEMENT for calibration/validation it does not touch
`train_refs` and therefore does not touch any census number here - but it WILL change the
`SupervisionGeometry` if it changes the training origin stride, and the census is derived from
that geometry rather than restating it (pinned by
`the_production_geometry_is_derived_from_the_model_config_and_not_restated`), so it follows
automatically.

## 13. Appendix: the raw census, so it is never re-derived

Measured in release on the real 4,873-ticker corpus, `--seed 20260905`, geometry
`seq_len 6000, patch_len 16, min_history 256, pred_len 192` → `origins_per_row 375`,
`active 360`, `row_stride 192`, `window 5744`, `interior_multiplicity 30`. `histogram[m]` = number
of distinct absolute origins supervised by exactly `m` retained rows.

```
control (identity)   rows 2,455,276  support 31,159,116  meanM 28.36728  interiorM 30 share 0.89166
  partial-tail rows 4,848   decile drift 0.00000
  hist  19:115,992  20:115,992  21:115,848  22:115,704  23:115,488  24:115,500
        25:115,128  26:115,128  27:115,032  28:115,020  29:114,912  30:27,783,348
  cover1 reaches 0.98487 @2001   0.99 @2646   0.999 @7597

K=30 (row-stride-multiple) rows 81,893  support 29,481,480  meanM 1.00000  interiorM share NaN
  partial-tail rows 154   decile drift 0.00300
  hist  1:29,481,480   (nothing else - a fully fresh sweep)
  cover1 reaches 0.98487 @316   0.99 @317   0.999 @320

K=4                  rows 613,830   support 30,981,408  meanM 7.13263  interiorM share NaN
  hist  1..6: ~465,000 each   7:14,265,744   8:13,920,720
  (the 7/8 split is 5744/768 = 7.479, two distinct origin phases, landing ~50/50 as predicted)

patch-phase random   rows 2,455,105 support 410,751,324 meanM 2.15176  interiorM share NaN
  phase-dropped rows 171 (0.007%)   decile drift 0.00004
  hist  1:145,087,800  2:132,668,988  3:80,584,092  4:35,808,360  5:12,306,264
        6:3,373,596  7:755,400  8:139,956  9:23,088  10:3,456  11:276  12:48
  cover1 reaches 0.98487 @9195   0.99 @9326   0.999 @9564; at 2500 it is only 0.44877

F=0.25 (row-fraction) rows 613,819  support 30,799,596  meanM 7.17460  interiorM share NaN
  partial-tail rows 1,237   decile drift 0.00058
  hist  10:2,568,444  11:1,545,252  12:814,056  13:372,456  14:150,336  15:52,668
        16:16,380  17:4,416  18:1,080  19:156  20:108  21:48   (plus the M<10 mass)
  cover1 reaches 0.98487 @1474   0.99 @1638   0.999 @2265
```

Two comparisons worth keeping, because they are the arithmetic the whole ticket turns on:

1. **`K=4` versus `F=0.25` at the same pool size** (613,830 vs 613,819 rows, meanM 7.13 vs 7.17):
   support 30,981,408 vs 30,799,596. The uniform draw destroys **1.15% more outcomes** than the
   stride, because it can drop every row supporting an outcome while a stride never can. That
   0.59% support difference is the ENTIRE reason `F=0.25`'s own-support saturation (1,474) differs
   from the control's (2,001) - a denominator effect, not an occupancy effect.
2. **`K=30` cuts the pool 30x and the support by 5.4%.** That ratio is what makes it the
   intervention: it thins the OVERLAP without thinning the SUPPORT, which is precisely what
   `--row-fraction` provably cannot do.

Startup cost of the census, for the successor's budget: 1.888 s control / 1.883 s phase on a quiet
machine (3.3-8.4 s under heavy concurrent builds), charged as `row selection and supervision
census` on `timexer_segment_startup`. Steady-state host batch assembly is unchanged within ±2.6 ms
run-to-run noise on every arm - see §3.
