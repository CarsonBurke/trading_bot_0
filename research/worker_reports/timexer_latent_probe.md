# The frozen-trunk latent probe: the falsifier for every latent-objective proposal

Worker: LatentProbe. Scope: read-only investigation plus one new authenticated subcommand,
`probe-timexer-segment`. No architecture change, no objective change, no corpus-contract change,
no split change, no NextLat implementation.

Everything below tagged **DEMONSTRATED** was measured in this repository. Everything tagged
**HYPOTHESIS** is a prediction. The probe's own numbers are produced by the job in §6 and are
not in this document, because they do not exist until it runs; §5 states the rule that converts
them into a verdict, and the rule is pre-registered so it cannot be chosen after the fact.

## 1. Why this measurement, and not another

**DEMONSTRATED (prior work, `timexer_amplitude_calibration.md`, `timexer_gain_two_sided.md`):**
oracle per-horizon rescaling moves the h = 192 close MSE ratio from `1.0212` to `0.99695` on
`timexer-control-4k` step 3000, and by construction moves the within-timestamp IC by nothing at
all — a positive per-horizon scalar cannot reorder one timestamp's tickers. So amplitude repair
is worth about 2.4% of MSE at the long end and exactly zero information, and it is essentially
exhausted. Every remaining proposal that only re-scales, re-weights or re-calibrates the
existing forecast is bounded by that number.

New long-horizon edge therefore has to be *information reaching the forecast that does not reach
it today*. There are exactly two places it can be missing from:

- **World A, extraction failure.** The frozen trunk latent linearly predicts the long-horizon
  target better than the trained head does. The trunk knows something the dense 192-row head and
  its count-normalized objective fail to route, and an objective that concentrates gradient on
  fewer, higher-SNR pathways has something to collect.
- **World B, information ceiling.** A linear probe merely matches the head. Then the head
  already extracts what the latent knows, no auxiliary loss defined on latents can manufacture
  edge, and the honest levers are context, features, capacity or the target definition.

A linear probe is the right instrument for a reason specific to this architecture and not a
generic one: **the head is itself a shallow decoder over that same latent** — one hidden GEMM to
1024 units, GELU, one output GEMM (`model.rs:2346-2352`). Anything a linear map extracts, the
head is expressively capable of extracting. A probe win is therefore unambiguously an
optimization/routing failure rather than a capacity one, which is exactly the claim a latent
auxiliary objective would have to rest on. The converse is weaker and is stated as such: a tie
bounds only the *linearly* decodable information, and it is the right verdict to act on only
because the head is shallow.

## 2. Which tensor, at which point, and why

`CausalPatchModel::backbone` (`model.rs:1705-1767`) ends with `rms_norm(&state)` and then, under
`last_only`, narrows to token position `origins - 1`. **The probe reads exactly that narrowed
post-norm tensor**, `[rows, 1, 512]`, and it reads it because it is literally the tensor
`CausalPatchModel::head` consumes as its `state` argument (`model.rs:2324-2345`). Three
consequences, all load-bearing:

- **Post-final-RMSNorm, not pre.** The head reads the normalized stream. RMSNorm's gain is a
  *per-token* positive scalar, not a global one, so pre-norm and post-norm are not related by any
  single linear map. A probe fitted before the norm would be fitted on a different object than
  the head sees, and a win could be attributed to the row scaling rather than to the
  representation. The comparison is only clean at the head's own input.
- **The origin token only.** Every scored quantity in this project is a final-origin quantity.
- **Latent only; the covariate block is withheld.** **DEMONSTRATED:** `head()` concatenates a
  256-wide projection of the known-future calendar/gap covariates
  (`COVARIATE_WIDTH = 256`, `model.rs:21, 1407-1415, 2327-2345`) onto the 512-wide latent before
  its hidden GEMM, so the head's input is 768-wide and the probe's is 512-wide. The probe sees a
  **strict subset** of the head's input. This makes the test conservative in the only direction
  that matters: a probe that beats the head does so while seeing strictly less, and that gap
  cannot be explained away by extra inputs.

Implementation: `runner::probe_origin` calls `backbone` and `head` separately rather than the
composed `forward`, precisely so the tensor between them can be captured. The head's own close
forecast comes out of the *same* forward over the *same* rows, which is what makes every reported
difference paired rather than a comparison across two draws.

## 3. The estimator: closed form, two classes, no SGD

All of it in `trading_bots/src/torch/timexer_segment/probe.rs`.

**Sufficient statistics.** One pass accumulates, per probed horizon, six fp64 device-resident
accumulators: `n`, `Σz`, `Σzzᵀ`, `Σzy`, `Σy`, `Σy²`, with the per-horizon validity mask applied.
Every downstream quantity — every ridge fit, every rank restriction, every penalty evaluation —
is a contraction of those six, which is why a 37-point penalty grid crossed with three classes
costs one pass rather than 111. fp64 and not fp32: the Gram is ~4·10⁵ rank-one updates of a
unit-RMS vector, so its diagonal reaches 10⁵ while the centred cross moment sits near 10⁻²; an
fp32 accumulator would lose the signal into the Gram's exponent.

**Class (a), ridge linear.** All 512 directions, `d + 1 = 513` fitted parameters.
`w = V(Λ + λI)⁻¹Vᵀc` from one symmetric eigendecomposition of the centred latent covariance.

**Class (b), rank-restricted.** `r ∈ {8, 32}`, so `r + 1 ∈ {9, 33}` fitted parameters. The
retained subspace is the top-`r` eigenvectors of the **latent's own second moment**, which sees
no target at all. A rank-8 probe therefore cannot select 8 directions *because* they correlate
with the future — it is handed the 8 directions the representation spends the most variance on.
That is strictly harder than reduced-rank regression and it is what removes "the probe just
memorized 512 free directions" from the list of available explanations for a win.

**Penalty selection: minimum held-out squared error, not maximum correlation.** This is not a
stylistic choice and getting it wrong was a real defect during development. Correlation is
scale-invariant, so it is indifferent between a probe and the same probe shrunk by a thousand;
the grid's argmax is then decided by rounding and the selected probe's amplitude — which the
reported MSE ratio is quadratic in — is arbitrary. Measured during development: under a
correlation criterion the estimator recovered a planted coefficient of `0.05` as `4.3e-5`, an
890-fold collapse with the correlation unmoved. Squared error is the criterion ridge is a
solution to, it is scale-sensitive, and it is the one criterion under which a shrunk probe is
correctly refused. The correlation is still computed and reported, as the diagnostic that says
whether transfer to the scored split degraded relative to the fit partition's own holdout.

**Conditioning is reported, not assumed away.** Each fit carries the condition number of the
system actually solved *and* of the same system with no penalty. A probe whose advantage only
appears when the unpenalized ratio is astronomical is inverting noise, and the panel is built so
that can be seen and the result refused.

## 4. Leakage: impossible by construction, with the proof

`probe::Partitions::split` is the **only** way to obtain origins in this subcommand, and it takes
both populations at once, so no caller can hand the fit and the score the same rows. It enforces,
on realized timestamps rather than on the boundary arithmetic that produced them:

1. the fit and scored origin sets are disjoint as `(ticker, origin)` pairs — a set claim gets a
   set proof, not a corollary of an inequality;
2. every bar any fit target reads completes strictly *before* the first scored origin, via
   `calibration::Blocks::spanning`. This is the check that matters: `pred_len` cumulative targets
   reach 192 valid bars past their own origin, so populations split on origin time alone would
   still share bars;
3. the ridge penalty is chosen on the chronologically **last** 20% of the fit partition, itself
   purged from the inner coefficient block by the same target-reach rule — so the penalty is
   never selected on the scored split either, and never on rows the coefficients saw.

The fit partition is the corpus's reserved `[70%, 80%)` band, which checkpoint selection never
read (selection minimizes held-out-full NLL over `validation_refs`), so the probe is clean of
selection bias on the fit side as well as of leakage.

**The test that proves it** — `probe::tests::no_origin_the_probe_fits_on_is_ever_an_origin_it_scores`
— builds three tickers × 4,000 dated origins with 192-bar target reach, runs the real `split`, and
asserts all three pairwise set disjointness facts, both purge gaps strictly positive, and the
exact size of the inner purge band (one target reach per ticker). Its sibling
`overlapping_populations_are_refused_or_priced` asserts each leakage mode is named by its own
error, and that the refusal is tight rather than a blanket.

### 4.1 What the guard caught on the real corpus, and what it cost to satisfy

**DEMONSTRATED.** Job 5458 was refused by this guard before it produced a number:

```
the evaluation block's first origin is at 1550178000000 but the calibration block's targets
reach 1723725600000
```

2019-02-14 against 2024-08-15 — the fit block ending five and a half years *after* the scored
block begins. Two hypotheses were on the table: genuinely reversed populations, or a guard
comparing per-ticker ordinals against wall clock. **Neither.** Both sides are wall clock from
`CorpusTicker::timestamp` over one universe, so the mechanism was right; what was wrong is
**granularity**. `boundaries[i]` is each ticker's OWN valid-bar ordinal, so `[70%, 80%)` is a
fraction of *that ticker's* history, and across a universe of unequal histories the global
extrema interleave even though every ticker is individually correct.

Measured by `probe::tests::the_real_reserved_partitions_are_dated_and_their_ordering_is_measured`
(ignored, real corpus, 85 s CPU):

| quantity | measured |
| --- | --- |
| tickers whose calibration target reaches at or after their own first validation origin | **0 of 4,498** |
| scored origins sharing a TIMESTAMP with a fit origin | **0 of 433,303** |
| fit target reach span | 2023-08-18 → 2024-08-15 |
| scored origin span | 2019-02-14 → 2025-08-15 |
| scored origins surviving a cut at the fit block's reach | **428,965 of 433,303, 99.0%** |

So no bar and no instant is common to the two populations, and the wall-clock violation is
carried entirely by a thin tail of very-long-history tickers whose 80% boundary falls early.

**The fix does not weaken the guard.** The scored block is cut to origins beginning strictly
after the fit block's last target bar, so the global inequality holds by construction and
`Blocks::spanning` remains as its postcondition on the populations actually used. The measured
price is **1.0%** of the scored population. Above `OUTER_PURGE_CEILING = 0.05` the pair is
refused outright — five times the measured price, so it is a live bar rather than a rubber
stamp, and the alternative of softening the check to the per-ticker claim (which passes
unaltered) was rejected precisely because it would have been free. `outer_purged` is a public
field, a console line and part of the chart title: the scored draw is a strict subset of
`held-out full` and no reader can miss by how much. Origin-set disjointness runs **before** the
cut, so a genuinely overlapping pair is still named as one rather than reported as an alignment
failure.

### 4.2 The same instrument on the split every other number rests on

**DEMONSTRATED**, `probe::tests::the_training_population_is_dated_against_every_held_out_draw`,
104 s CPU. If training and `held-out full` interleave the same way, the model trains on calendar
dates that are held-out dates for other tickers — and a market-neutral cross-sectional residual
is not independent across two tickers on one date.

Training target bars span 2016-10-07 → 2023-08-17 over 4,873 tickers.

| draw | origins | at/before last training target | exactly on one | same-ticker | cross-ticker |
| --- | --- | --- | --- | --- | --- |
| `held-out full` | 433,303 | 1 (0.0002%) | 1 | **0** | 1 |
| `held-out cross-section` | 4,000 | 0 | 0 | **0** | 0 |
| calibration | 433,721 | 4,307 (0.99%) | 4,246 | **0** | 4,246 |

Verdict **NO OVERLAP** for both scored draws. The distinction that makes it a "no": the span
overlap is 1,644 days but the population overlap is one origin in 433,303. The per-ticker
fractional boundary does interleave the date ranges, exactly as predicted, and the interleaved
tail is empirically almost empty. Same-ticker exposure is zero everywhere and is asserted, not
inferred, which is what licenses reading every hit as cross-ticker.

The one non-zero number, 0.98% on the calibration partition, is **fit-side**: it can make a
fitted gain or a fitted probe marginally less out-of-sample with respect to calendar, and it
cannot leak into the scored split, which the §4.1 cut places strictly behind every fit target.

## 5. What is reported, and the pre-registered verdict

Per horizon in `DECISION_HORIZONS = [1, 8, 16, 32, 64, 128, 192]`, all on `held-out full`, for
the trained head and each of the three probes, **all reduced from one forward pass over identical
origins**:

- within-timestamp IC and its standard error, on exactly the scorer's own eligibility rule
  (≥ `CROSS_SECTION_MIN` = 20 valid names in the timestamp, nonzero spread in both series), because
  a probe measured on a different rule would not be comparable to the head at all;
- market-neutral close MSE ratio against close-anchored persistence;
- the **oracle-rescaled** ratio for probe and head alike, so the two are compared with amplitude
  neutralized on both sides. Without it the panel re-measures the known amplitude defect instead
  of information content;
- `probe IC − head IC` with the SE of the **paired** difference, computed from the dispersion of
  per-timestamp differences over timestamps usable for *every* forecaster. Not the difference of
  two independent SEs: the two ICs share a draw, their sampling errors are strongly positively
  correlated, and treating them as independent would inflate the error enough to make World B
  unfalsifiable.

**Pre-registered decision rule** (`probe::WORLD_A_GAIN`, `WORLD_B_BAND`, `VERDICT_HORIZON_FLOOR`,
applied by `probe::verdict`, printed as a verdict line and stamped into the chart title):

> A paired IC gain of **at least +0.010** at any horizon `h ≥ 64` is **World A** and justifies
> building a latent objective. Every long horizon inside **±0.005** is **World B** and retires the
> entire NextLat line permanently. Anything between is reported as **indeterminate** and is not
> converted into a verdict by rounding.

`+0.010` was sized as ≈ 3.5 iid SE at the then-measured full-split IC precision (SE ≈ 0.0028) and
a 15% relative move against the then-measured h = 64 IC of 0.0655. The World B band is half the
World A bar so the two cannot both fire.

> **RETRACTION, job 5483.** Both numbers in that sentence are **strided-draw measurements pending
> anchored recomputation**. With `--placement anchored` the same weights on the same split give
> h = 64 IC **0.1261** rather than 0.0530, a 2.4x move in the LEVEL, because a 10.6-name
> cross-section cannot carry a within-timestamp statistic and demeaning at n ≈ 10 attenuates the
> quantity being measured. The `+0.010` bar is therefore no longer 15% of the h = 64 IC; against
> 0.1261 it is 7.9%. I am **not** moving the threshold in this report — a threshold re-derived
> after seeing which way the level went is not pre-registered — but the next session MUST re-size
> it against an anchored draw BEFORE the probe is run, and must do so from the anchored IC and
> the anchored PAIRED SE, not from either strided number.
>
> What does **not** move: the probe's gap is a **paired per-timestamp difference on identical
> origins**, computed by `PairedScorer` from the dispersion of per-timestamp differences over
> timestamps usable for every forecaster. The correlated component cancels, so the paired SE is
> the one the verdict reads and it was never the iid 0.0028. The batch-wide finding that unpaired
> IC thresholds were undecidable does not reach this gate; the re-sizing above is about the
> threshold's RELATIVE size, not its validity.

### 5.0 The gate is ASYMMETRIC: `+0.010 + discount`, never `+0.010`

**Ruled by Main and LANDED in `probe::verdict`, which now takes `Option<&Recency>`.** The probe's
fit targets span 2023-08-18 → 2024-08-15, its scored block 2024-08-19 → 2025-08-15, and the
head's training targets end 2023-08-17 — so the probe is fitted on a window strictly more recent
than anything the head trained on. Under the session's measured non-stationarity part of any
probe win is recency, not latent information the head failed to route.

`probe::Recency` prices it from data the probe already reads and at **zero extra forward pass**:
`RECENCY_TRANCHES = 3` equal-**population** chronological tranches of the fit partition (equal
population, not equal duration — the point is to vary fit-block AGE with sample size held fixed,
which is the control the extraction ladder does not provide), each fitted against the same fixed
penalty holdout, each scored on the same held-out block in the same pass, giving `dIC/dyear` by
least squares; multiplied by the head's own age deficit, measured as the realized last training
target bar rather than a nominal boundary. The World A bar becomes `WORLD_A_GAIN + discount(h)`
per horizon; the World B band is **not** widened. A discount that is missing, negative or
non-finite contributes zero, so an unmeasured discount can never loosen the gate.

Direction of the asymmetry, which is the whole reason it is not symmetric:

- a probe that **fails** despite a recency advantage is a **stronger** World B than the flat reading;
- a probe that **wins** must be discounted by that advantage first — a win driven by recency is an
  argument for retraining, not for a latent objective.

Reported on the fifth base `timexer_segment_latent_probe_recency`: tranche ICs as series, the
discount as its own series, the head's age deficit in the title.

**Verdict as of this report: NOT YET MEASURED.** The subcommand is built, tested and ready; no GPU
job has been run, because the queue is not mine. §6 is the exact line. §7 costs the proposal that
World A would justify, conditionally and with its criticisms, so that a World A result is
actionable the moment it lands and a World B result closes the line with nothing owed.

### Report bases

Four, not one, and the deviation from the brief is deliberate. The registry's own doc comment
says a dimensionless ratio near 1 beside a correlation near 0.05 renders all but one as a flat
line, and a condition number spans decades on top of that. One question, one unit, one axis:

| base | question | unit |
| --- | --- | --- |
| `timexer_segment_latent_probe` | does a linear read of the frozen latent beat the trained head? | correlation coefficient (ICs and paired differences), with both decision thresholds drawn as reference lines |
| `timexer_segment_latent_probe_ratio` | with amplitude neutralized on both sides, does it beat the head on MSE? | ratio vs close-anchored persistence |
| `timexer_segment_latent_probe_conditioning` | is the advantage an inverted near-singularity? | condition number, symlog; fitted parameter count in each series label |
| `timexer_segment_latent_probe_scaling` | is the probe limited by the latent's information or by its own fit sample? | correlation coefficient against **coefficient-fit origins**, not bars ahead — which is why it cannot share a panel with the first |

No label contains `=`. Unmeasured quantities render NaN, never 0 — asserted by
`a_draw_with_no_usable_cross_section_reports_gaps_not_zeros`.

### 5.1 The extraction ladder: separating an information ceiling from a sample-size one

A probe that merely matches the head is World B **only if the probe is not itself fit-limited**.
`--scaling` settles that. Seven nested fit sets spanning 64x in fit rows, drawn by whole
timestamp (`teacher::whole_timestamp_ladder`) so every cross-section survives entire, each
scored at h = 1 on the identical held-out block, each selecting its penalty on the **same fixed
held-back block** so `IC(N)` varies with coefficient rows and with nothing else. Then
`IC(N) = IC_inf − a·N^−b`, exponent by a 61-point geometric grid with `(IC_inf, a)` in closed
form at each candidate — deterministic, no SGD, a function of the data and the grid alone.

Two properties earn it its place rather than making it a nice-to-have:

- **Nesting is asserted, not trusted.** `scaling_shells` refuses a ladder whose rung *i* drops a
  row rung *i−1* kept, and refuses a finest rung that is not the whole population. Nested rungs
  share their common sampling mode, so the *differences* between rungs — which is all a power
  law is — are far better determined than the levels. With independent draws each rung would
  carry its own noise of order `1/√timestamps`, which at the coarse end is the same size as the
  curvature the exponent is read from: a non-nested ladder yields a curve that looks well behaved
  and means nothing.
- **It costs one pass, not seven.** Rung *i* minus rung *i−1* is a disjoint shell; each row is
  accumulated into exactly one shell by row-index selection, so total Gram work equals a single
  pass, and prefix-merging the shells' sufficient statistics reconstructs every nested fit set
  for free. No latent cache exists.

The x axis is **realized** fit origins, counted, not the nominal share: cross-sections are
unequal in width, so a 64x-coarser stamp set is not a 64x-smaller row count, and the power law
must be fitted against the number that existed.

### 5.2 The third explanation the verdict must be read against

**A probe that fails to beat the head has three explanations, not two:** an information ceiling
(World B), a fit-sample ceiling (§5.1 separates it), and a **regime change between the fit block
and the scored block**. The ladder is blind to the third, and the session's measured
non-stationarity — a full-span h=32 slope of +.0265 against −.1097 on a recent 5-session window,
and a recent `V_h/(h·V_1)` of .652 against .398 held-out — makes it a live possibility rather
than a hypothetical.

What partly protects the reading is an asymmetry that is measured rather than assumed. The
probe's fit targets span **2023-08-18 → 2024-08-15** and the scored block **2024-08-19 →
2025-08-15**, while the head's training targets end **2023-08-17**. The probe is therefore
fitted on a window strictly *more recent* than anything the head trained on. If the market moved,
that asymmetry **favours the probe**. So:

- a probe that still fails to beat the head is a **stronger** World B than the flat reading, because
  it failed while holding a recency advantage;
- a probe that wins must be **discounted by that advantage** before it counts as World A — a win
  driven by recency is an argument for retraining, not for a latent objective.

This does not change the pre-registered thresholds and is not licence to move them. It is the
interpretation clause, fixed in advance of the number.

## 6. The command line

**DEMONSTRATED budget arithmetic.** Measured held-out evaluation is 1,123 ms per pass over the
2,048-origin sample, i.e. ≈ 140 ms per 256-row batch. The probe makes three passes: coefficient
fit, penalty selection, and scoring. Scoring is 433,303 origins = 1,694 batches ≈ 237 s and is
**not** capped, because its origin count is exactly what sets the SE of the paired difference the
whole subcommand exists to measure. The fit side *is* capped at 200,000 origins (781 batches,
≈ 109 s) — subsampling there costs coefficient precision, which the penalty holdout then measures
and the ridge absorbs, and 160k inner rows against 512 dimensions is still 312 rows per parameter.
Total ≈ 346 s of forward plus ~30 s warm startup ≈ **376 s, 63% of the 600 s lease.** The added
fp64 Gram arithmetic is 0.94 GFLOP per batch, ≈ 0.5 s over the whole job.

```bash
mlq submit --name timexer-latent-probe --priority 1 --max-parallel-runs 1 \
  --time-limit 10m --max-attempts 1 --cwd "$PWD" -- \
  ./torch-env.sh /var/tmp/tb0_v17 probe-timexer-segment \
  --checkpoint training/runs/timexer-control-4k/weights/best \
  --output training/runs/timexer-control-4k/gens/latent-probe \
  --batch-size 256 --fit-origins 200000
```

The extraction-ladder arm, which is a **second job** rather than a flag on the first, because it
adds seven ridge solves and a second scorer to the same three passes and both must fit one lease:

```bash
mlq submit --name timexer-latent-probe-scaling --priority 1 --max-parallel-runs 1 \
  --time-limit 10m --max-attempts 1 --cwd "$PWD" -- \
  ./torch-env.sh /var/tmp/tb0_v17 probe-timexer-segment --scaling \
  --checkpoint training/runs/timexer-control-4k/weights/best \
  --output training/runs/timexer-control-4k/gens/latent-scaling \
  --batch-size 256 --fit-origins 200000
```

Notes on the binary: **v13 is barred, v14 incomplete, v15 and v16 predate this work** — the
binary must be v17 or later, built from a tree containing `probe.rs`'s outer wall-clock cut and
`teacher::whole_timestamp_ladder`. A v16 binary would run the probe with the pre-fix pairing and
abort on the `Blocks::spanning` refusal, which is what job 5458 did. The probe is inference-only —
it never calls `CausalPatchModel::losses` — so `FuseLoss`'s loss kernel is not on its path. It
loads through the same `load_checkpoint` as `evaluate`, so the `corpus.contract == manifest.data`
assertion and `check_head_layout` both still run; a mismatched head layout aborts rather than
being probed.

If the lease is ever tight, the only knob is `--fit-origins`. There is deliberately no scored-side
knob.

## 7. Conditional: the minimal latent objective, costed and criticized

**This section is contingent on World A and is written as a costing, not a recommendation.** If
the probe lands in World B it should be read only as the record of what was declined.

### 7.1 A correction to the design prior that changes the object

**DEMONSTRATED, and it invalidates the dyadic-grid-in-bars form as stated.** The trunk's latents
live on the *patch token* grid: `patch_len = 16`, `origins() = 375` token positions over a 6,000-bar
context. There is no latent at t + 1 bar. A dyadic grid in **bars** (`k = 1, 2, 4, 8, …`) has no
target to reach for below k = 16, so it cannot supervise any horizon shorter than one patch —
which is precisely the region where the measured defect is largest (fitted optimal gain 4.28 at
h = 1, i.e. the mean emitted at ~24% of true size, and above 1 through h = 19). A latent
propagator is structurally blind to the short end. That must be said out loud before anything is
built on it, and it is an argument for the short-end and long-end defects needing *different*
instruments rather than one.

The realizable grid is dyadic in **tokens**: `k ∈ {1, 2, 4, 8, 12}` tokens = `{16, 32, 64, 128, 192}`
bars. Five pathways, not eight; they land exactly on the decision horizons ≥ 16 and terminate on
192. That is the object costed below.

### 7.2 Form

`ẑ_{t+k} = z_t + MLP(LN(z_t) + e_k)`, one **shared** propagator with a learned per-k embedding
`e_k`, trained against `stopgrad(z_{t+k})`, train-time only, so inference stays a single direct
pass and the checkpoint's scoring path is untouched. Shared and not five separate MLPs: five
independent MLPs re-create the near-duplicate dilution at the parameter level, and the map from
`z_t` to `z_{t+k}` is a smooth function of k. This is NextLat's residual-MLP-with-LN update used
as a **forecasting propagator** rather than as a posterior update — which is the one change that
answers the standing objection to the paper, whose own update is conditioned on having already
seen `x_{t+1}`.

### 7.3 The four costs, at 96,000 tokens/step and `d = h = 512`

Valid token positions per pathway are `96,000·(1 − k/375)`, mean `94,618` over the five k.

| cost | value | against basis |
| --- | --- | --- |
| **parameters** | `2·512·512 + 5·512` = **526,848** (0.527 M) | +1.94% on 27.2 M. Five separate MLPs would be 2.62 M, +9.6% — the reason for sharing. |
| **FLOP/step** | forward `5 × 94,618 × 2 × (2·512·512)` = 0.496 TFLOP; with backward ≈ **1.49 TFLOP/step** | +8.8% on 16.98 → 18.47 TFLOP/step |
| **memory traffic/step** | LN shared once per token (~3 KB) + ~27 KB per token per pathway fwd+bwd ≈ **13.5 GB/step** | +9.4% on ~143 → ~156.5 GB/step |
| **ms/step** | compute side 1.49 TFLOP at the trunk's realized 101 TFLOP/s ≈ 14.7 ms; traffic side 13.5 GB at ~1.6 TB/s ≈ 8.4 ms; ~30 extra launches partially overlap → **+16 to +22 ms** | 168 → **184–190 ms/step, +10 to +13%** — **HYPOTHESIS**, a benchmark arm settles it |

Retained activation memory for backward ≈ `96,000 × 5 × 2 KB` (hidden + output) + 96 MB (shared
LN) ≈ **1.06 GB**, taking peak allocator from 18,573 MiB to ≈ 19.6 GiB of ~25.9 GiB usable. Fits.

**Lease consequence, stated because it changes the arm and not just the arithmetic:** at 190 ms
uncontended, 2,500 steps is 475 s and fits the 10-minute lease. At the *contended* 215.7 ms
baseline plus 13%, the step is 244 ms and 2,500 steps is 610 s, which does **not** fit. A latent
arm must be pre-registered at **2,300 steps**, not 2,500.

### 7.4 Representation collapse: what prevents it, and how it is detected

**The failure is real and the usual reassurance is not sufficient.** A stop-grad target the trunk
also produces is trivially satisfiable by making all latents equal.

What actually prevents it here: the probed latent is *also the head's input*, and the head's NLL
is the dominant term on the same tensor. A collapsed latent makes the forecast constant and the
NLL explodes, so the main loss is a genuine anti-collapse force rather than a hopeful one. The
post-RMSNorm constraint additionally makes the zero solution unreachable — collapse would have to
be "every latent equal to one unit vector". And the stop-grad is on the target only, giving the
standard BYOL-style asymmetry.

What that argument does **not** cover, and what will actually happen if it happens: *partial*
collapse — a rank reduction of the latent distribution that leaves the NLL almost unchanged while
quietly removing the directions the cross-sectional signal lives in. That is invisible to NLL and
would be attributed to the objective "not working".

So detection is first-class, not a footnote. Three series, all on one new base
`timexer_segment_latent_health`, on the same step axis as the decision panel:

1. **effective rank** of the latent covariance, `(Σλ)²/Σλ²` normalized by `d` — the single number
   that says whether the representation is using 512 directions or 12;
2. **mean coordinate dispersion**, the within-batch standard deviation per coordinate averaged
   over coordinates;
3. **mean |off-diagonal correlation|** of the 512 coordinates.

Cost: one 512×512 Gram over one batch per report interval, 50 GFLOP once per interval, ≈ 0.5 ms —
not per step. Pre-registered abort: effective rank falling below 0.25× the control arm's value at
matched step is a collapse and the arm is killed rather than interpreted.

### 7.5 Two criticisms of the prior that survive costing

- **"Latent targets are unit-scale after RMSNorm, so a latent loss is immune to the amplitude
  pathology."** True but the wrong word. The latent loss is not *immune* to the amplitude defect,
  it is *irrelevant* to it: the amplitude error lives in the head's output map, which a latent
  loss never touches. A latent objective is structurally incapable of fixing the measured MSE
  mis-scaling; its only route to MSE is a better representation. That is acceptable — the oracle
  bound says amplitude is worth ≤ 2.4% anyway — but it should not be sold as an advantage.
- **The target may be mostly unpredictable, and latent space offers no way to down-weight that.**
  `z_{t+k}` is the trunk's own summary of a context that includes the bars between t and t+k, most
  of whose variance is market noise the model cannot forecast. Predicting it spends gradient on
  the unpredictable part in proportion to that part's share of latent variance, and there is no
  latent analogue of the horizon reweighting that mattered in output space (`inv-sqrt` is the
  current best arm). The 5-pathway grid fixes the *count* problem — 5 pathways instead of 160
  near-duplicates — but not the *SNR-weighting* problem, and those are different problems.
  If World A holds, the first thing to measure on any latent arm is what fraction of `Var(z_{t+k})`
  is predictable from `z_t` at all.

**Relevance to the newly reported IC-rot trajectory (h = 16: .1137 → .0086, h = 64: .0879 → −.0125
between steps 2000 and 4000 on the `inv-sqrt` arm, `held-out cross-section` draw):** this probe is
measured on a *single frozen checkpoint* and answers a question about that checkpoint's
representation. It does not, and is not built to, explain a trajectory. If the trunk at step 3000
is already at its information ceiling, then the rot is happening in the representation itself and
a latent objective is a candidate regularizer rather than a source of edge — a different and
weaker claim than the one §7 is costing. The cheapest extension, if wanted, is to run the same
subcommand on the step-1000, step-2000 and step-4000 checkpoints and read the probe IC as a
trajectory beside the head's; that is four independent sub-10-minute jobs and needs no new code.

## 8. Status, verification, and handover

### 8.1 LANDED and green

| item | file | state |
| --- | --- | --- |
| `probe-timexer-segment` subcommand | `runner.rs` (EOF), `main.rs` | landed |
| estimator, partitions, paired scorer, verdict | `probe.rs` (new) | landed, 8 tests |
| outer wall-clock cut + `OUTER_PURGE_CEILING` | `probe.rs` | landed |
| extraction ladder (`--scaling`) | `probe.rs`, `runner.rs` | landed, unrun |
| recency discount + asymmetric gate | `probe.rs`, `runner.rs` | landed, unrun |
| five report bases and their writers | `reports.rs`, `shared/src/report.rs` | landed |

Nothing is half-landed. Nothing is stubbed. Every path above compiles and every unit test passes.

Last measured, on the shared tree with five peers landing concurrently:
`./torch-env.sh cargo check -p trading_bot_0 --tests` **0 errors**;
`cargo test -p trading_bot_0 timexer_segment::probe` **8 passed, 0 failed, 2 ignored** (both
ignored are real-corpus measurements, run explicitly and reported in §4.1 and §4.2);
`cargo check -p trading-bot-tui --tests` **0 errors**; the bidirectional registry test passes with
all five bases. The one failure seen in the wider `timexer_segment` suite at the close,
`model::tests::a_rotated_basis_refuses_a_horizon_weighting_and_a_rank_restriction`, is
`OrthoTargets`' file and not on any path this subcommand executes.

### 8.2 Predictions, retractions, current status

| pre-registered | status |
| --- | --- |
| World A at paired IC gain ≥ +0.010 for `h ≥ 64` | **STANDS as a rule, RE-SIZE before running.** The gate is now `+0.010 + recency discount(h)`, asymmetric by ruling. The `≈ 3.5 iid SE` and `15% of h=64 IC` justifications are RETRACTED as strided-draw quantities (job 5483: h=64 IC .0530 → .1261 anchored). Re-derive the bar from the anchored IC and the anchored PAIRED SE, before the run, not after. |
| World B inside ±0.005 at every long horizon | **STANDS**, band deliberately not widened by the discount. |
| Oracle rescaling is worth ≤ 2.4% MSE at h=192 | **STANDS**, unaffected by placement (an MSE ratio is not a within-timestamp statistic). |
| Dyadic propagator on a grid in BARS | **RETRACTED by me in §7.1** — latents live on the 16-bar token grid; the realizable grid is `k ∈ {1,2,4,8,12}` tokens and no propagator can supervise `h < 16`. |
| "Latent targets are unit-scale, so a latent loss is immune to the amplitude pathology" | **RETRACTED in §7.5** — not immune, irrelevant: the amplitude error lives in the head's output map a latent loss never touches. |

### 8.3 Dependencies on other agents' numbers

- **Cited, not load-bearing:** the 0.50/0.39/0.35 within-window variance ratios appear only as the
  MOTIVATION in a doc comment. The invariance they motivate is algebraic — the ridge grid is in
  units of the mean eigenvalue of the latent second moment, which no horizon touches — and holds
  if the ratios are flat, drifting or regime-dependent.
- **Load-bearing, and now retracted upstream:** the strided-draw `held-out full` ICs used to size
  the threshold. See §8.2.
- **Consumed:** `teacher::whole_timestamp_ladder` (FutureTeacher) for the ladder's draw.
  `teacher::whole_timestamp_keep`'s whole-cross-section property is what makes the rungs valid;
  the nesting is re-asserted independently in `probe::scaling_shells`.
- **Supplied:** the per-ticker-ordinal granularity defect (§4.1) and the calendar audit (§4.2),
  both of which fed `TemporalSplit`'s occurrence ledger.

### 8.4 Jobs still wanted, in priority order

**Neither has been run. Both need a v17-or-later binary.** Before running either, thread
`--placement anchored` reasoning through the threshold as §8.2 requires — the probe itself needs
no placement flag, because its verdict is a PAIRED difference on identical origins and the head
is re-scored in the same pass, but the SIZE of the pre-registered bar was set against a strided
level and must be re-derived.

```bash
mlq submit --name timexer-latent-probe --priority 1 --max-parallel-runs 1 \
  --time-limit 10m --max-attempts 1 --cwd "$PWD" -- \
  ./torch-env.sh /var/tmp/tb0_v17 probe-timexer-segment \
  --checkpoint training/runs/timexer-control-4k/weights/best \
  --output training/runs/timexer-control-4k/gens/latent-probe \
  --batch-size 256 --fit-origins 200000
```

```bash
mlq submit --name timexer-latent-probe-scaling --priority 1 --max-parallel-runs 1 \
  --time-limit 10m --max-attempts 1 --cwd "$PWD" -- \
  ./torch-env.sh /var/tmp/tb0_v17 probe-timexer-segment --scaling \
  --checkpoint training/runs/timexer-control-4k/weights/best \
  --output training/runs/timexer-control-4k/gens/latent-scaling \
  --batch-size 256 --fit-origins 200000
```

The recency discount rides the first job at no extra pass and needs no flag.

### 8.5 Two defects these tests caught, kept because they generalize

1. **Selecting the ridge on holdout CORRELATION** recovered a planted coefficient of 0.05 as
   4.3e-5: correlation is scale-invariant and cannot see an 890-fold collapse. The criterion is
   holdout squared error. Any penalty selected on a scale-invariant criterion has this bug.
2. **`tch::manual_seed` is process-wide** and the harness runs test threads in parallel, so a
   seeded fixture drew differently depending on which sibling ran beside it — observed as the same
   assertion producing 0.0326 and 0.0213 on identical code. The estimator fixture owns a
   self-contained xorshift stream. This is the same class `OrthoTargets` later diagnosed in the x0
   test.
