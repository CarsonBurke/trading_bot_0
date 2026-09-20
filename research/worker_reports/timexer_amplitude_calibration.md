# Amplitude calibration of the CausalPatch OHLC mean — landing report

Worker: `CalibFix`. Scope: finish the in-flight cutover left uncompilable (14 errors), verify the
seven pre-agreed design decisions against the landed code, and prove the CPU-testable half.
Line numbers refer to the tree as delivered.

**No GPU command, no training, no evaluation, no benchmark, no `mlq` command was run.** Every
number quoted from a real checkpoint in this document is a number handed to me, not a
measurement I made; every number I computed myself is derived arithmetic from those and is
labelled as such.

---

## 1. What the defect is

The forecaster's per-horizon information is fine and its per-horizon amplitude is not. On
`timexer_segment_horizon_steps_gain` at step 2000 the close channel's MSE-optimal single gain
sweeps `3.662` at h = 1 down to `0.195` at h = 192 — a factor of 18.8 across one axis — while the
anchored held-out IC is scale-invariant and improved at every horizon on that same checkpoint. At
h = 128 and h = 192 the amplitude error alone is the whole result: the market-neutral close MSE
ratio reads `1.039` and `1.0386`, WORSE than persistence, against `0.9968` and `0.9975` at those
horizons' own best scale.

A positive per-horizon gain is exactly the transform that repairs the first and provably cannot
touch the second. Within one timestamp and one horizon it multiplies every ticker's forecast by
the same positive number, so every rank, sign, decile membership and therefore every rank
statistic is invariant by construction. That invariance is the mechanism's strongest available
self-check and it is asserted on the real `Scorer` reduction, not on the algebra
(`runner.rs:4630`, tolerance `runner.rs:3019`).

---

## 2. The fit math

### 2.1 Two coordinates, not four channels

`decode_joint` does not emit four independent forecasts. It emits ONE close anchor and three
monotone offsets built from the range and position coordinates:

```
close = z·√h,   low = close − a/σ,   high = close + (r − a)/σ,   open = close + (o − a)/σ
```

with `0 ≤ a ≤ r` and `0 ≤ o ≤ r`. A decoded candle therefore has exactly TWO amplitude degrees of
freedom, and the estimator fits exactly two curves:

```
f_c(h) = g_anchor(h)·C(h) + g_offset(h)·O_c(h),      O_c = f_c − C,   O_close ≡ 0
```

Both gains are strictly positive, so every offset keeps its sign and its ordering relative to the
anchor and `high ≥ max(open, close) ≥ min(open, close) ≥ low` survives for exactly the reason it
survives uncalibrated. Four independently smoothed per-channel curves would not have this
property: at h = 192 the anchor is a ~14σ cumulative move while the intrabar offsets are ~1σ, so a
7 % disagreement between two channel gains already crosses the ordering.

### 2.2 The per-horizon solve

Per horizon, `(g_anchor, g_offset)` is the exact least-squares solution of the design `[C, O_c]`,
**pooled over the four channels** and every origin in the fit block, with no intercept
(`calibration.rs:863`). Pooling over channels is what makes the anchor gain minimize the
FOUR-channel squared error rather than the close channel's alone — the four rows of one bar share
one anchor and constrain both coordinates jointly.

```
G = [ ΣC²   ΣCO ]      b = [ ΣCy ]      ĝ = G⁻¹b
    [ ΣCO   ΣO² ]          [ ΣOy ]
SSE(ĝ) = Σy² − ĝ'b
```

**Rank degradation (fixed in this landing).** Either column can be empty. `O_close ≡ 0` by
construction, so a decode whose intrabar range has collapsed — a head early in training, or any
population scored on the close channel alone — leaves the offset column identically zero at every
channel and `det G = 0`. The previous code refused the whole horizon there. The anchor amplitude
is still *exactly* measurable in that state, and it is the state whose amplitude is most wrong, so
refusing it blanked the calibration precisely where it was needed. The solve now degrades to rank
1, estimating the coordinate that carries energy and reporting the other as unmeasured
(`calibration.rs:863-938`). This is exact, not approximate: `ΣO² = 0 ⇒ O ≡ 0 ⇒ ΣOy = 0`, so the
omitted term in `SSE` and in the amplitude cost is identically zero. The residual variance divides
by `elements − rank`, not a hard `− 2`.

### 2.3 Why no intercept

Refusing an intercept is a measured refusal, not an assumption. `ȳ²/E[y²]` is the ENTIRE share of
persistence MSE any constant forecast could earn at that horizon, and `MeanCalibration::fit`
declines to apply any gain unless that is below `OFFSET_CEILING_SHARE = 0.1` of the amplitude cost
it would compete with (`calibration.rs:132`, gate at `calibration.rs:635`). Measured margin on the
control checkpoint is 65× below the threshold. An intercept is also the least stationary thing in
the data — a bet on the next block's mean drift — and it is the one part of an affine map a
within-timestamp IC cannot see.

### 2.4 The weighting, and its justification

The fit weight on horizon `h` is the **inverse delta-method variance of `ln ĝ_h`**, taken from
that same solve's Gram matrix:

```
Var(ĝ) = k·s²·G⁻¹,     s² = SSE/(elements − rank),     k = channels
weight_h = ĝ² / Var(ĝ)                       (i.e. 1/Var(ln ĝ) to first order)
```

Three properties earn this choice over the obvious alternatives:

1. **It is on `ln g`, because the smoother is.** The curve is fitted on `ln g` against `ln h` so
   that the result is positive by construction. A weight on `g` would then be the wrong metric:
   the same absolute standard error means something completely different at `g = 3.662` and at
   `g = 0.195`, and the horizon axis sweeps exactly that range. `Var(ln ĝ) ≈ Var(ĝ)/ĝ²` is the
   right unit and the delta method is exact to the order the whole estimator claims.
2. **The channel multiplicity is charged as a design effect (`k = channels`).** The four candle
   channels of one bar share one anchor error, so one BAR is one independent residual, not four.
   Charging `k` doubles the standard error rather than pretending to four times the sample. A
   constant factor on every weight cannot move the smoother — the penalty grid is expressed in
   units of the mean weight — so this only widens the three-sigma amplification bound, which is
   the conservative direction. Not charging it would have understated `SE(ln ĝ)` by 2× and let the
   amplifying half of the estimator deploy gains the data does not prove.
3. **It is the block's own precision, not a prior.** Horizons with no measurable amplitude get
   weight exactly zero and are then determined by the roughness penalty alone — interpolated from
   their neighbours, which is the only defensible thing a smoothness prior can say about them —
   and their amplification ceiling is the identity, so a neighbour's evidence can shrink them but
   never amplify them.

### 2.5 The shape prior

Adjacent horizons' true gains cannot jump: `g` is a smooth functional of the joint law of
`(f_h, y_h)` and neighbouring cumulative returns overlap in 191 of 192 bars. Each curve is
therefore a **roughness-penalized weighted least squares on `ln g` against `ln h`**:

```
u = (W + λR)⁻¹ W y,      R = natural-cubic-spline roughness ∫(u″)²      (calibration.rs:1065)
λ selected by weighted GCV over 41 geometric strengths spanning 10⁻⁴…10⁶ × mean weight
effective dof = tr((W + λR)⁻¹W)
```

`R` uses divided second differences each weighted by the interval it integrates over, so a
non-uniform axis is penalized in its own units rather than per index. Deliberately **not** a
two-parameter power law: `192^-0.57` reproduces the 18.8× sweep to two digits, but weighted by
their own measured precision the control checkpoint's residuals leave `χ²/dof = 23.6` against a
log-log line and `13.6` against a log-quadratic, on a curve whose own second differences sit well
below the noise scale. The fine structure is real and a global shape would erase it.

### 2.6 The two-sided bound

Applying gain `g` to a forecast whose true amplitude is `β` leaves `−(β − g)²·Var(f)/P`, so MSE
supplies no ceiling anywhere and a measured gain above 1 is under-amplification that a
shrinkage-only clamp simply refuses to collect. What bounds the amplifying direction is estimation
error, because a gain multiplies deployed notional in every sizing rule affine in the forecast:

```
ceiling_h = max(1, exp(ln ĝ_h − 3·SE(ln ĝ_h))),    g_h = min(fitted_h, ceiling_h)
```

(`AMPLIFICATION_SIGMAS = 3.0`, `calibration.rs:136`.) A horizon whose gain is within three
standard errors of 1 is never amplified; a horizon with no measurable amplitude gets
`ceiling_h = 1`.

---

## 3. What the close-only fit was doing to the other channels

This is the part worth stating plainly, because the old mechanism was not merely narrower — it was
mis-aimed.

- **It moved all four channels but aimed at one.** Every decoded channel is built on the close
  anchor (`f_c = C + O_c`), so a gain on `C` moved open, high, low and close together. The old
  curve was fitted to minimize the **close channel's** squared error alone. The three offset
  channels were therefore rescaled by a number chosen without reference to their own residuals:
  correct for the close row, arbitrary for the other three.
- **It left the intrabar spread uncorrected.** `g_offset` was implicitly frozen at 1, whatever the
  data said. If the head's range/position coordinates were themselves over- or under-amplified —
  which nothing in the objective prevents, and which the `√h` prior on σ actively encourages to
  drift with the horizon — that error was carried straight through into every high/low forecast
  and into every intrabar-width consumer. The four-channel MSE ratio, the primary metric, was
  being scored against a candle whose *width* had never been calibrated.
- **The direction of the residual error is not knowable from the close fit.** Because
  `Var(O) ≪ Var(C)` at long horizons, the close-only fit is nearly the anchor-only fit there and
  the offset error is invisible in the close residual. At short horizons the two are comparable
  and the close-only fit's anchor gain is *biased by* the uncorrected offset: it absorbs part of
  the offset's amplitude error into `g_anchor`, which is precisely the regime (h = 1, gain 3.662)
  where the amplitude correction is largest.

The pooled 2×2 solve fixes all three at once: `g_anchor` now minimizes the four-channel error, and
`g_offset` is a measured quantity rather than an assumption. The reports make the difference
visible by plotting all four DECODED channels' own MSE-optimal single gains beside the two applied
curves (`reports.rs:2578-2586`) — the four channels are what the two degrees of freedom have to
reproduce, and a fit that reproduces close and misses high is legible on that panel.

---

## 4. Per-decision verification table

| # | Decision | Status | Citation |
|---|---|---|---|
| 1 | ONE mechanism: `MeanCalibration`, generalized from close-anchor-only to all channels; any second amplitude mechanism deleted, not guarded | **IMPLEMENTED** | Fit `calibration.rs:607`; pooled-over-channels solve `calibration.rs:863`; two coordinates `calibration.rs:518-527`; the only applied amplitude is `model.rs:1799`/`model.rs:1824`. The old `calibrate` subcommand and `evaluate --calibration` are gone (no `CalibrateArgs`, no `--calibration` flag anywhere in `main.rs`/`mod.rs`/`runner.rs`); `portfolio_data.rs`'s NNLS scalar is demoted to a diagnostic (row 6). I fixed one residual of the deletion: a dangling `[`CalibrateArgs`]` doc link, now `runner.rs:4728`. |
| 2 | Fit INSIDE the training run on the calibration split `[70 %, 80 %)`; evaluate on `[80 %, 90 %)`; fit and report never share origins | **IMPLEMENTED** | Fit draw from `corpus.calibration_refs` `runner.rs:2018`; scored draw from `corpus.validation_refs`; disjointness *proved* rather than assumed by `Blocks::spanning` `runner.rs:2041`, which compares the calibration block's **last target** timestamp (not its last origin) against the first evaluation origin — cumulative targets reach `pred_len` bars past their origin, so origin-time separation alone would share bars. Fit executed at every evaluation `runner.rs:2332`. Refusals covered by `calibration.rs` tests `the_two_partitions_are_dated_and_proven_disjoint_in_the_bars_their_targets_read` and `two_partitions_whose_targets_and_origins_overlap_are_refused_rather_than_scored`. |
| 3 | Application point is a single fold, so reports, `evaluate`, trading and deployment cannot disagree | **IMPLEMENTED**, with a correction to the *named* location | Single application point `model.rs:1824` (`gained`), reached only from `decode` (`model.rs:2747`, `model.rs:2772`); installed once at checkpoint load `runner.rs:2754`. **The decision named `model.fold_mean_gain`, a fold into the `√h` decode buffer. That is not where it can live and the landed code is right to differ:** `horizon_scale` multiplies the close coordinate only, so folding a gain there would rescale the anchor and leave the three intrabar offsets at their emitted size — it cannot express `g_offset` at all. The fold is applied to the DECODED candle instead, which is strictly downstream of `horizon_scale` and equally singular. The stale doc claiming `horizon_scale` was the calibration's target is corrected at `model.rs:1523-1528`. Training cannot see it: `losses` asserts `mean_gain.is_none()` (`model.rs:2901`). |
| 4 | Checkpoint-selection NLL stays computed from the UN-GAINED model; per-horizon ratios reported BOTH un-gained and gained as two labelled series | **IMPLEMENTED** | Selection scalar is the un-gained evaluation's `objective_nll` (`runner.rs:2338`, stored `runner.rs:2456`); the amplitude passes are explicitly scored un-gained and touch neither `best_*` nor either curve slot (`runner.rs:2320-2332`); the manifest field documents the reason it is not part of selection (`runner.rs:642-646`). σ is untouched by the gain — `gained` reads neither `half_log_horizon` nor the log-scale channels — so the NLL is arithmetically the un-gained model's. Both ratio series: `Emission` (`reports.rs:2443`) plus `AmplitudeSplit::ratios` (`reports.rs:2496`), which inverts the gain in closed form so one pass yields both readings, emitted as `"… ratio, uncalibrated"` and `"… ratio, calibrated"` (`reports.rs:2719/2723`). Pinned by `the_fitted_amplitude_panels_separate_training_from_held_out_gain_and_name_the_gate`, which checks both series against `Moments::pooled_ratio` and `Moments::pooled_gained_ratio` at every horizon. |
| 5 | The `calibrate` subcommand and the `evaluate --calibration` artifact path are DELETED; the checkpoint manifest carries the calibration | **IMPLEMENTED** | Manifest field `runner.rs:646`, mandatory and un-defaulted; authenticated by the manifest digest; structurally validated on read `runner.rs:730`. No `calibrate` subcommand and no `--calibration` argument remain. |
| 6 | NNLS `scalar_gain` becomes a residual DIAGNOSTIC (reported, not applied); NNLS's nonnegativity is preserved as an explicit GATE — a horizon whose fitted calibration-split gain is non-positive is sized to zero and NAMED with its fitted value; never silently clamped | **IMPLEMENTED** | Diagnostic: `measured_portfolio_gain` (`portfolio_data.rs:802`) is documented as measured-and-never-applied and lands in the summary as `measured_portfolio_contract_gain_ratio`; the applied amplitude is the checkpoint's (`portfolio_data.rs:1044-1048`). Gate: `sizing_refusal` (`calibration.rs:825`) → consumed at `portfolio_data.rs:1055`, and the zero lands on the **mean** at `portfolio_data.rs:1220` (`mean: if gated { 0. }`), which is the right place because sizing is affine in the mean, so a zero mean is a zero position at every name. Named, never clamped: the signed measurement rides in the checkpoint (`calibration.rs:752`), `validate` deliberately does **not** positivity-check it (`calibration.rs:779-789`), the gated horizons are listed in the panel title with their values (`reports.rs:2651`) and in the account assumptions (`portfolio_data.rs:1058`), and counted at `portfolio_data.rs:1088`. Pinned by `a_nonpositive_measured_gain_gates_its_horizon_to_zero_size_and_is_named_not_clamped`. |
| 7 | A diagnostic series accumulating the same MSE-optimal gain on a TRAINING draw, labelled with the `training` split word, on the same base as the calibration-split gain | **IMPLEMENTED** | Draw: `training_draw` from `corpus.train_refs`, 512 origins (`runner.rs:2023`, `runner.rs:51`), scored with the same `score` reduction on the same weights in the same evaluation (`runner.rs:2328`), carried as `AmplitudeFit::training` (`reports.rs:2471`) and emitted on the **same base and same axis** as the calibration-split gain at `reports.rs:2587-2596`, labelled `"training close MSE-optimal gain"` (`TRAINING`, `reports.rs:58`). Same unit (dimensionless gain), which is what makes the shared axis legitimate — the MSE ratios live on their own base for the opposite reason. The reading rule is in the chart title itself (`reports.rs:2682`): "is the forecast's over-amplitude an out-of-sample shrinkage problem or an in-sample objective one?" Pinned by the report test, which asserts the training series is present, flat at its injected value, and the **only** series carrying the `training` split word. |

**Nothing was found MISSING.** Decisions 4, 6 and 7 — the ones flagged as most likely dropped —
were all present in the production paths; what was missing was the compilable test layer around
them, plus the two real defects in §5.

---

## 5. Defects found and fixed beyond the compile errors

### 5.1 `measured_anchor: Vec<f64>` could not round-trip the JSON manifest

`serde_json` serializes a non-finite `f64` as `null` (`serde_json-1.0.145/src/ser.rs:169-180`,
verified by reading the source) and then refuses to deserialize `null` into an `f64`. The signed
measurement legitimately carries "no amplitude identified here" at any horizon, which the previous
code encoded as `NaN`. Consequence: **any checkpoint with one unidentified horizon wrote a
manifest that this build could not read back** — and an untrained head has *every* horizon
unidentified, so the very first preview checkpoint of a fresh run was unloadable. A run would have
trained for hours and then failed at `Manifest::read`.

Fixed by making the field `Vec<Option<f64>>` (`calibration.rs:752`), which is also the honest
type: `None` is "this horizon's amplitude was not identified" and `Some(g)` is a measurement,
including a negative one. The two compose correctly with the gate — both mean "size this horizon
to zero and name it", and they are reported as **distinct reasons**, never merged
(`calibration.rs:825-835`, `reports.rs:2657-2663`). The NaN/JSON reason is recorded in a comment
at the field itself.

Consumer audit, as requested:

- **Decode fold** — never reads `measured_anchor`. `set_mean_gain` consumes only the two
  strictly-positive curves (`model.rs:1799`), so no `None` can reach a forecast buffer. Nothing to
  do.
- **Portfolio tape entry/exit anchors** — read through `tradable`/`sizing_refusal`, which treat
  `None` and non-positive identically for the *decision* and differently for the *reason*
  (`portfolio_data.rs:1055`). The two summary scalars use `unwrap_or(f64::NAN)`
  (`portfolio_data.rs:1077-1084`); the summary is postcard-encoded, which is NaN-safe.
- **Reports path — the behaviour I chose, and why.** `None` renders as `f64::NAN` in the chart
  series (`reports.rs:2612`), i.e. it draws as a **gap**. It is explicitly *not* folded to 0.0.
  A 0.0 there would read as "this horizon's measured amplitude collapsed to zero", which is a
  measurement claim the block never made and the opposite of "unmeasured". (This is the same
  reason `Moments::channel_gain` returns `NaN` rather than 0 for a flat forecast.) The concern
  about a `None` folded as 0.0 into the decode buffer cannot arise here — see the first bullet —
  but the reports rule matters on its own terms: an amplitude of 0 combined with the MSE identity
  would make the ratio read exactly 1.0, i.e. indistinguishable from perfect persistence, which
  is the most misleading possible rendering of "we don't know".

### 5.2 A singular Gram matrix discarded a perfectly measurable anchor

Described in §2.2. `solve_horizon` returned `None` whenever `det G = 0`, which is the *normal*
state for any population with no intrabar range — including the close-anchor-only population the
existing closed-form test uses, which is why that test was failing on `expect("an identified
horizon")`. Now rank-aware (`calibration.rs:863-938`), with the degenerate cases pinned by
`the_exact_solve_recovers_an_injected_per_horizon_gain_in_both_coordinates` (rank-2, rank-1 with
the offset column emptied, and rank-0 refused).

### 5.3 A dead constant asserting an untrue claim

`IC_INVARIANCE_TOLERANCE` was declared in the library at `runner.rs:3016` and used only from the
test module, so the lib build warned it was never used while the module docs claimed it aborted
runs. Moved into the test module that owns it, with the doc corrected to what it actually does —
gate the real-`Scorer` invariance assertion (`runner.rs:3011-3019`).

---

## 6. The manifest FORMAT bump story

The stamp moved `v10 → v11`, adding `-mean-gain` to both x0 bases
(`runner.rs:451-454`):

```
causal-patch-ohlc-universe-v11-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres-mean-gain-x0-learned
causal-patch-ohlc-universe-v11-…-mean-gain-x0-none
```

with the mean spec appended by `format_stamp`, so a shipped stamp reads
`…-mean-gain-x0-learned-mean-basis:2:2`.

**`v11` is the most dangerous kind of bump: the parameter set did not change.** Same tensors, same
names, same shapes — the gain is applied to the decoded mean, not to any weight — so a `v10` state
dict loads into a `v11` build without a single shape error. What changed is what a checkpoint
*means*: a `v11` checkpoint emits a calibrated mean the moment it is loaded, so its MSE, its
trading diagnostics and its portfolio sizing are not the same numbers a `v10` checkpoint of the
same weights produced. Averaging the two together, or step-matching a calibrated arm against an
uncalibrated one, is exactly the confusion the mechanism exists to end. Both directions are
refused by name rather than tolerated:

**Old checkpoint into new build — two independent rejection paths.**

1. `runner.rs:674-677` — `Manifest::read` requires the stamp to start with one of the two `v11`
   bases before anything else is parsed. A `v10` manifest is named and refused: *"unsupported
   universe checkpoint format …; this build reads …-v11-…-mean-<spec>"*. Asserted for both `v10`
   stamps (and every stamp back to `v4`) at `runner.rs:3915-3936`.
2. `runner.rs:646` + `runner.rs:3937-3950` — `mean_gain` has **no serde default**, so a `v10`
   manifest hand-restamped to a `v11` string still fails to deserialize on the missing field.
   There is no such thing as a `v11` checkpoint whose amplitude is unstated. This is the path the
   stamp check cannot be edited around, and it is now asserted directly: the test strips
   `mean_gain` from the serialized manifest and requires the error to name the field.

There is a third, structural guard for a curve that is present but wrong:
`FrozenGain::validate` is called against the checkpoint's own `pred_len` on read
(`runner.rs:728-732`), so a wrong-length or nonpositive curve is named there rather than surfacing
as a shape error at the first decode (asserted at `runner.rs:3951-3962`).

**New checkpoint into old build.** A build that predates this field reads `v11` in `format` and
refuses it by its own unsupported-format check — the same `ensure!` shape that refused `v10` and
`v9`. That is what stops an old binary from loading `v11` weights, ignoring the gain, and
reporting an over-amplitudinal forecast under calibrated labels. This direction cannot be asserted
from inside this build (it is a property of the *old* binary's identical check); it is guaranteed
by the fact that the accepted-stamp list has always been an exact allowlist, never a prefix
wildcard on the family.

A checkpoint whose block identified no amplitude carries the identity **explicitly**
(`FrozenGain::is_identity`, `calibration.rs:839`) — a measurement, and not the same statement as a
missing field.

---

## 7. Predicted effect, per horizon

### 7.1 The identity everything below follows from

For a single gain `g` applied to a forecast `f` against target `y`, with uncentered second moments
`s_f² = mean(f²)`, `s_y² = mean(y²)` and uncentered correlation `ρ = mean(fy)/(s_f s_y)`:

```
MSE(g·f, y) / mean(y²) = 1 − 2·g·ρ·s_f/s_y + g²·s_f²/s_y²
```

Minimized at `g* = ρ·s_y/s_f`, where it takes the value `1 − ρ²`. Equivalently

```
ratio(g) = (1 − ρ²) + (g − g*)²·(s_f/s_y)²
```

This identity is asserted against hand arithmetic (not against the implementation's own algebra)
in `the_gained_ratio_matches_the_closed_form_and_a_gain_cannot_move_a_correlation`.

### 7.2 Recovering each horizon's geometry from the handed-in numbers

Two handed-in numbers per horizon — the MSE-optimal gain `g*` and the ratio at that gain — pin
`ρ` and `s_f/s_y` exactly: `ρ = √(1 − ratio(g*))` and `s_f/s_y = ρ/g*`. **Derived arithmetic,
not measured:**

| h | `g*` (given) | ratio at `g*` (given) | ratio at `g = 1` (given) | ⇒ `\|ρ\|` | ⇒ `s_f/s_y` | ratio at `g=1` recomputed |
|---|---|---|---|---|---|---|
| 1 | 3.662 | 0.97430 | 0.98825 | 0.1603 | 0.0438 | 0.98788 |
| 8 | 1.673 | 0.94995 | 0.95822 | 0.2237 | 0.1337 | 0.95805 |
| 16 | 1.593 | 0.93607 | 0.94493 | 0.2529 | 0.1587 | 0.94492 |
| 32 | 1.182 | — | — | — | — | — |
| 64 | 0.517 | 0.98735 | 0.99835 | 0.1125 | 0.2176 | 0.99839 |
| 128 | 0.215 | 0.9968 | 1.039 | 0.0566 | 0.2631 | 1.0395 |
| 192 | 0.195 | 0.9975 | 1.0386 | 0.0500 | 0.2564 | 1.0401 |

The last column is a consistency check, not new information: recomputing `ratio(1)` from the
recovered `(ρ, s_f/s_y)` reproduces the independently reported un-gained ratio to 4–5 decimal
places at every horizon. The identity in §7.1 is therefore the right model of this defect, and the
whole gap between the two given ratio columns is amplitude.

h = 32 has a handed-in gain (1.182) but no handed-in ratio pair, so its geometry is not
determined; I am not going to invent it. Its gain is close enough to 1 that the available headroom
`(1 − g*)²·(s_f/s_y)²` is small — interpolating `s_f/s_y ≈ 0.18` between h = 16 and h = 64 would
put it near `0.001`, i.e. a tenth of a point — but that is an interpolation, not a measurement.

### 7.3 What the fit is predicted to deliver

Best-case (fitted gain lands on each horizon's own optimum) improvement in the market-neutral
close MSE ratio:

| h | un-gained | at best scale | Δ | reading |
|---|---|---|---|---|
| 1 | 0.98825 | 0.97430 | **−0.01395** | already skilful, ~1.4 pt better |
| 8 | 0.95822 | 0.94995 | **−0.00827** | best absolute level either way |
| 16 | 0.94493 | 0.93607 | **−0.00886** | best absolute level either way |
| 32 | — | — | — | not determined by the handed-in numbers |
| 64 | 0.99835 | 0.98735 | **−0.01100** | marginal → clearly skilful |
| 128 | 1.039 | 0.9968 | **−0.0422** | **worse than persistence → skilful** |
| 192 | 1.0386 | 0.9975 | **−0.0411** | **worse than persistence → skilful** |

Every horizon improves, and the long end changes *sign of verdict*. The two long horizons are
where the mechanism is load-bearing: the amplitude error there is worth 4 points of ratio, an order
of magnitude more than at the short end, purely because `s_f/s_y` is ~6× larger while `ρ` is ~3×
smaller.

### 7.4 How much the fit is allowed to miss by

From §7.1, the fitted `ĝ` beats persistence iff `ratio(ĝ) < 1`, i.e.

```
(ĝ − g*)² < g*²    ⟺    0 < ĝ < 2·g*
```

**Any positive gain below twice the optimum beats persistence at that horizon.** At h = 192 that
is any `ĝ ∈ (0, 0.39)` against a current effective `ĝ = 1` — the un-gained forecast is 5× outside
the interval, which is why it loses. And the captured share of the available improvement is

```
captured = 1 − ((ĝ − g*) / (1 − g*))²
```

so at h = 192 a fitted gain anywhere in `[0.115, 0.275]` — ±40 % of `g* = 0.195` — still captures
90 % of the 4.1-point gain. This is the quantitative reason the roughness prior is safe here: the
long-horizon optimum sits far from 1 and the loss surface around it is flat in relative terms, so
smoothing across neighbouring horizons costs almost nothing while the variance reduction it buys
is large. At the short end the tolerance is tighter in relative terms (h = 1 needs
`ĝ ∈ (0, 7.3)` to beat persistence, trivially satisfied, but needs `ĝ ∈ [2.8, 4.5]` to capture
90 %), and correspondingly h = 1 has the smallest `(s_f/s_y)²` and so the least to lose.

Two effects work against these numbers and are not modelled above, deliberately:

1. **Out-of-sample shrinkage.** `g*` above is each horizon's optimum *on the population it was
   measured on*. The applied curve is fitted on `[70 %, 80 %)` and scored on `[80 %, 90 %)`, so
   the realized improvement is bounded above by these deltas and the gap is the genuine
   out-of-sample cost. This is exactly what the two ratio series on
   `timexer_segment_amplitude_calibration` measure and exactly why they are reported un-gained and
   gained on the same pass.
2. **The amplification bound.** h = 1, 8, 16 and 32 all want `g* > 1`. Whether any of that is
   deployed depends on `3·SE(ln ĝ)` at those horizons, which is a function of the fit block's
   size and residual and is not determinable without a run. If the short end's amplification is
   not proven to three sigma, those four horizons keep `g = 1` and the realized improvement is
   the long end's alone — which is still the whole of the sign change at h = 128/192.

### 7.5 Which horizons the gate would zero

**Not statically determinable, and not determinable at all without a GPU run.** The gate reads the
sign of the *fitted calibration-split* anchor gain, which is a property of the model weights and
the reserved partition, both of which require inference to evaluate.

What can be said from the handed-in numbers: on the step-2000 checkpoint every reported
per-horizon MSE-optimal gain is strictly positive (3.662 … 0.195), so **if the calibration split
reproduces the sign structure of the split those numbers came from, the gate would zero no
horizon** and `calibration_gated_horizons_count` would be 0. A horizon gets gated only where the
calibration-split solve returns a non-positive anchor (the forecast points the wrong way there) or
identifies no anchor amplitude at all (a flat forecast). The second case is expected to be common
at the *first* evaluation of a fresh run, where a zero-init head emits a constant — which is a
legible outcome, not a failure, and is why the fit returns the identity with a named reason rather
than an error that would abort a 4000-step run at step 0.

---

## 8. Every file:line changed in this landing

Line numbers are in the delivered tree.

**`trading_bots/src/torch/timexer_segment/calibration.rs`** (+280 / −38)
- `512-527` — `Solved`'s two coordinates become `Option<f64>`; doc states why `None` ≠ measured 0.
- `645-650` — refusal branch's `measured` closure adapted to optional solves.
- `678-692` — `fit`'s per-coordinate `read`/`identified` closures adapted; comment separates
  "measured negative" from "no energy to measure".
- `752` — `measured_anchor: Vec<Option<f64>>`, with the NaN/JSON reason at the field.
- `711-720` — `frozen()` maps a non-finite measurement to `None`.
- `779-789` — `validate` checks only the *present* measurements are finite; still deliberately not
  a positivity check.
- `800-806` — `tradable` reads through the option.
- `810-816` — `gated()` returns `Vec<(usize, Option<f64>)>`.
- `825-835` — **new** `sizing_refusal`, the deployed gate decision, returning prose so the two
  gating reasons stay distinct.
- `847-938` — `solve_horizon` rewritten rank-aware; residual variance divides by
  `elements − rank`.
- `1044` — the half-applied rename's `CurveFit::identity(coordinate, horizons, …)` →
  `measured_gain` (**one of the 14 original errors**).
- `1266-1305` — solve-recovery test extended with the rank-1 and rank-0 cases.
- `1308-1366` — **new** `the_fit_reproduces_a_noiseless_injected_gain_curve_to_a_tight_tolerance`
  (requirement 5a).
- `1368-1445` — **new** `a_nonpositive_measured_gain_gates_its_horizon_to_zero_size_and_is_named_not_clamped`
  (requirement 5c).
- `1742-1750` — closed-form test's solve assertion adapted to the rank-1 population it builds.

**`trading_bots/src/torch/timexer_segment/model.rs`** (+5 / −2)
- `1523-1528` — corrected the false claim that `horizon_scale` is where a mean calibration lands;
  names `gained` as the single application point and says why a `√h` fold cannot express
  `g_offset`.
- `3688-3699` — test fixture's `measured_anchor` (**one of the 14**), carrying one negative and one
  unmeasured horizon so a consumer that conflates them fails.

**`trading_bots/src/torch/timexer_segment/portfolio_data.rs`** (+4 / −12)
- `1055-1061` — the gate routed through `FrozenGain::sizing_refusal`; 14 lines of inline
  formatting collapsed into the call, so the deployed decision is the one the CPU test covers.
- `1077-1084` — the two measured-gain summary scalars read through the option.

**`trading_bots/src/torch/timexer_segment/reports.rs`** (+233 / −1)
- `2602-2616` — the checkpoint-carried measurement series maps `None → NaN` so an unmeasured
  horizon draws as a gap, never as 0.
- `2651-2670` — `gated_note` names an unmeasured horizon as `unmeasured` rather than at a value.
- `3904` — trimmed a now-unused import.
- `3998-4008` — **new** `mod amplitude_report_tests`, CPU-only, so the family runs under one narrow
  filter. The pre-existing identity-gain test moved out of `lr_trajectory_tests`, whose sibling
  test builds a compute `Engine` and cannot run under a CPU-only filter.
- `4077-4298` — **new** `the_fitted_amplitude_panels_separate_training_from_held_out_gain_and_name_the_gate`.
- (The stale `Measured` / `AppliedGain` / `write_calibration_gain` / three-arg
  `MeanCalibration::fit` / `calibration.gain` test — **8 of the 14 original errors** — was replaced
  wholesale against the landed `write_amplitude`/`AmplitudePanels` API. No compatibility alias was
  added for either side of the rename.)

**`trading_bots/src/torch/timexer_segment/runner.rs`** (+63 / −18)
- `3006-3019` — `IC_INVARIANCE_TOLERANCE` moved into the test module that uses it; doc corrected.
- `3804-3830` — test manifest's `mean_gain` fixture (**one of the 14**): a seven-horizon
  non-identity curve with one negative and one unmeasured horizon, so the digest covers a real
  curve and the option round-trip is exercised.
- `3909-3914`, `3924-3925` — both `v10` stamps added to the stale-format rejection list, with the
  reason they are the most dangerous entry on it.
- `3937-3975` — **new** assertions: the missing-`mean_gain` rejection path, the wrong-length-curve
  rejection path, and the `None`/`null` manifest round trip including `gated()` surviving it.
- `4626-4629`, `4728-4730` — two dangling doc links repaired (`fold_mean_gain`, `CalibrateArgs`).

No changes were needed in `shared/src/report.rs`, `tui/src/main.rs` or `trading_bots/src/main.rs`:
all three bases were already registered on both sides (§9) and the `calibrate` subcommand was
already deleted.

---

## 9. Report bases, both sides

All three amplitude bases are registered in `shared/src/report.rs` and scanned by the TUI. The TUI
does not restate the names — `meta_chart_bases` **extends** the writer's registry
(`tui/src/main.rs:428`), which is the stronger form of the contract, and the bidirectional test at
`tui/src/main.rs:1218-1228` asserts set equality in both directions.

| Base | Registry line | TUI line |
|---|---|---|
| `timexer_segment_calibration_gain` | `shared/src/report.rs:73`: `    "timexer_segment_calibration_gain",` | `tui/src/main.rs:428`: `    bases.extend_from_slice(shared::report::TIMEXER_SEGMENT_REPORT_BASES);` |
| `timexer_segment_amplitude_calibration` | `shared/src/report.rs:74`: `    "timexer_segment_amplitude_calibration",` | same, `tui/src/main.rs:428` |
| `timexer_segment_calibration_moments` | `shared/src/report.rs:80`: `    "timexer_segment_calibration_moments",` | same, `tui/src/main.rs:428` |

The forward direction (`registry ⊆ scanned`) is `tui/src/main.rs:1203-1213`; the reverse
(`scanned ∩ timexer_segment_* == registry`) is `tui/src/main.rs:1218-1228`, which is what catches a
base the TUI still scans after a writer retires it. My report test additionally asserts all three
names from the writer side (`reports.rs`, `the_fitted_amplitude_panels_…`), so the panel and the
registry cannot drift apart without a test failing on whichever side moved.

**Presentation rules honoured.** Gains and MSE ratios are different units and are on different
bases with different `y_label`s — `OPTIMAL_GAIN_UNIT` vs `BEST_SCALE_UNIT` — asserted unequal in
the test. The gain base carries only dimensionless gains (applied curves, four channel optima, the
training comparand, the three-sigma ceiling, the 1.0 reference); the ratio base carries only ratios
plus the `persistence 1.0` line; the moment base carries only shares of the same persistence MSE.
Split vocabulary is exactly `training` / `held-out sample` / `held-out cross-section` /
`held-out full` / `persistence`, plus `calibration partition`, which is named rather than folded
into a held-out word because it is out of sample for the WEIGHTS and in sample for the GAIN.
Spaces are `market-neutral` / `raw`. Every label reads `<split> <space> <quantity>`.

---

## 10. Verification actually performed

### 10.1 Scoped checks

`./torch-env.sh cargo check -p trading_bot_0 --tests` — **0 errors** (down from 14):

```
warning: `trading_bot_0` (lib test) generated 18 warnings (17 duplicates)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 12.19s
```

`cargo check -p trading-bot-tui --tests` — **0 errors**:

```
tui/src/main.rs:1100:8: warning: methods `next_log_line` and `previous_log_line` are never used
tui/src/chart_viewer.rs:23:9: warning: field `path` is never read
warning: `trading-bot-tui` (bin "trading-bot-tui" test) generated 2 warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.12s
```

All remaining warnings are pre-existing and outside this scope (`lr_disentangle.rs`,
`bar_dist.rs`, `basic_nn.rs`, the two TUI ones). The three warnings this cutover had introduced —
the unused `IC_INVARIANCE_TOLERANCE`, an unused `FrozenGain` import and an unused
`TIMEXER_SEGMENT_REPORT_BASES` import — are gone.

### 10.2 Narrow CPU-only filter

```
$ ./torch-env.sh cargo test -p trading_bot_0 --lib -- \
    torch::timexer_segment::calibration::tests \
    torch::timexer_segment::reports::amplitude_report_tests

running 15 tests
test torch::timexer_segment::calibration::tests::a_constant_forecast_yields_the_identity_with_its_reason_rather_than_an_error ... ok
test torch::timexer_segment::calibration::tests::the_exact_solve_recovers_an_injected_per_horizon_gain_in_both_coordinates ... ok
test torch::timexer_segment::calibration::tests::the_two_partitions_are_dated_and_proven_disjoint_in_the_bars_their_targets_read ... ok
test torch::timexer_segment::calibration::tests::the_gained_ratio_matches_the_closed_form_and_a_gain_cannot_move_a_correlation ... ok
test torch::timexer_segment::calibration::tests::two_partitions_whose_targets_and_origins_overlap_are_refused_rather_than_scored ... ok
test torch::timexer_segment::calibration::tests::a_frozen_gain_is_refused_for_the_wrong_horizon_count_or_a_nonpositive_curve ... ok
test torch::timexer_segment::calibration::tests::an_amplification_its_own_standard_error_cannot_prove_is_bounded_to_the_identity ... ok
test torch::timexer_segment::reports::amplitude_report_tests::an_uncalibrated_run_writes_identity_gain_and_split_moments ... ok
test torch::timexer_segment::calibration::tests::an_intercept_worth_more_than_a_tenth_of_the_amplitude_error_applies_no_gain ... ok
test torch::timexer_segment::reports::amplitude_report_tests::the_fitted_amplitude_panels_separate_training_from_held_out_gain_and_name_the_gate ... ok
test torch::timexer_segment::calibration::tests::an_amplification_the_data_proves_is_applied_rather_than_pinned_to_one ... ok
test torch::timexer_segment::calibration::tests::horizons_with_no_calibratable_amplitude_are_carried_by_their_neighbours ... ok
test torch::timexer_segment::calibration::tests::a_nonpositive_measured_gain_gates_its_horizon_to_zero_size_and_is_named_not_clamped ... ok
test torch::timexer_segment::calibration::tests::the_fit_reproduces_a_noiseless_injected_gain_curve_to_a_tight_tolerance ... ok
test torch::timexer_segment::calibration::tests::the_fitted_curve_recovers_a_smooth_amplitude_and_averages_out_per_horizon_noise ... ok

test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 1139 filtered out; finished in 3.42s
```

Two more narrow CPU-only filters, for the paths I touched outside those modules:

```
$ … -- torch::timexer_segment::runner::tests::universe_checkpoint_authenticates_objective_schema_and_weight_bytes
test … ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1153 filtered out; finished in 0.01s

$ … -- torch::timexer_segment::model::tests::an_applied_mean_gain_rescales_anchor_and_offsets_without_touching_geometry_or_weights \
       torch::timexer_segment::runner::tests::a_positive_per_horizon_gain_moves_mse_and_cannot_move_a_rank_statistic
test torch::timexer_segment::runner::tests::a_positive_per_horizon_gain_moves_mse_and_cannot_move_a_rank_statistic ... ok
test torch::timexer_segment::model::tests::an_applied_mean_gain_rescales_anchor_and_offsets_without_touching_geometry_or_weights ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 1152 filtered out; finished in 0.15s
```

Total wall time for all three filters: under 6 s of test execution. No CUDA model construction, no
corpus load, no global CUDA lock taken.

### 10.3 What the three required tests assert

**(a) Known injected gain recovered to tight tolerance** —
`the_fit_reproduces_a_noiseless_injected_gain_curve_to_a_tight_tolerance`. Injects the reciprocal
of a known curve of the fitted family (`0.85 × measured_shape`, log-quadratic in `ln h`, plus a
separate near-constant offset curve), runs the full `MeanCalibration::fit`, and asserts at **every
one of 192 horizons** that the raw solve reproduces the injection to `1e-9` relative and the
smoothed applied curve to `2e-3` relative, in **both** coordinates. Both injected curves sit
strictly below 1 on purpose: the amplifying direction is bounded by
`exp(ln ĝ − 3·SE)` by design, so a reproduction tolerance above 1 would be a tolerance on the
bound rather than on the fit — that half is covered by the two dedicated ceiling tests, which
assert the bound binds when the data cannot prove the amplification and does not when it can. The
recovered anchor curve is separately asserted to carry a >2× sweep across the axis, so the
tolerance is tight against something with real dynamic range.

**(b) IC unchanged, MSE ratio at the closed form** —
`the_gained_ratio_matches_the_closed_form_and_a_gain_cannot_move_a_correlation` builds an explicit
64-row `(f, y)` population, computes `1 − 2gρ·s_f/s_y + g²s_f²/s_y²` from hand-summed dot products,
and asserts `Moments::channel_gained_ratio` matches it to `1e-12`; then asserts the uncentered
correlation of `g·f` against `f` moves by less than `1e-15`. The same contract is asserted on the
**real reduction** rather than the algebra by
`a_positive_per_horizon_gain_moves_mse_and_cannot_move_a_rank_statistic`, which pushes a 12-horizon
gain curve (including one horizon at 0.01) through the actual `Scorer` and requires the
within-timestamp IC, pooled Pearson, pooled Spearman, close hit rate, top-decile hit rate,
top-decile return, conviction spread and demeaned gain to all hold to `1e-4` while the close MSE
ratio is required to move.

**(c) Non-positive gain gates to zero size, recorded as gated not clamped** —
`a_nonpositive_measured_gain_gates_its_horizon_to_zero_size_and_is_named_not_clamped` inverts the
anchor covariance at h = 41…44 and empties the design at h = 51, then asserts: the measurement
survives strictly negative (not clamped); the *applied* curve stays strictly positive there (a
negative applied gain would invert a position rather than shrink it); `validate` still accepts the
checkpoint; `tradable` is false at all five; every other horizon including the immediate
neighbours stays tradable; `gated()` returns exactly `[41, 42, 43, 44, 51]`; and the deployed
`sizing_refusal` names h = 42 at its measured value while naming h = 51 as `unmeasured` and
returning `None` for an ungated pair. The manifest test additionally proves the gate survives a
JSON round trip with both reasons intact.

---

## 11. What I did NOT verify

Everything in this list needs a GPU, a corpus, or a training/evaluation run, all of which were out
of scope by instruction (the coordinator owns the queue).

1. **Any realized number from §7.** The predicted ratio improvements are arithmetic on handed-in
   measurements, not measurements. Nothing here has been scored.
2. **Whether the out-of-sample fit reproduces the in-sample optimum.** The whole out-of-sample
   shrinkage question — the gap between `ratio(ĝ)` and `ratio(g*)` — is unmeasured. It is exactly
   what the two labelled ratio series exist to report.
3. **Whether the short-end amplification is deployed.** h = 1, 8, 16 and 32 want `g* > 1`; whether
   `3·SE(ln ĝ)` allows any of it depends on the fit block's realized residual and size. Untested.
4. **Which horizons the gate actually zeroes.** §7.5. Not determinable without a run.
5. **The training-vs-held-out gain contrast itself** — decision 7's entire finding. The code path
   is verified (a synthetic training draw flows through to a correctly labelled series on the
   right base with the right unit); which way the real contrast falls is unmeasured, and it is the
   difference between "this calibration is the whole fix" and "the amplitude is wrong in sample
   and the cause is upstream in the objective".
6. **End-to-end checkpoint save/load on a real run.** The manifest round trip is verified on a
   synthetic manifest with a fabricated weight payload. No real `model.safetensors` was written or
   read.
7. **The `evaluate` and `evaluate_portfolio` entry points end to end**, including the
   `measured_portfolio_gain` NNLS diagnostic's cache path and the `mean: if gated { 0. }` sizing
   consequence on a real tape. The gate *decision* is CPU-tested at `sizing_refusal`; the tape
   assembly that consumes it is not.
8. **Cost claims.** "Zero on the training path", "0.5 GFLOP per evaluation", "≈8 KB of JSON" are
   inherited assertions. I verified structurally that no loss chain reads the gain buffers
   (`losses` asserts `mean_gain.is_none()`, and `gained` is reachable only from `decode`), but I
   measured no timing.
9. **The new-checkpoint-into-old-build direction of the FORMAT guarantee.** It is a property of
   the *previous* binary and cannot be asserted from this tree; see §6.
10. **The full `trading_bot_0` test suite.** By instruction I ran only narrow CPU-only filters —
    the CUDA tests serialize on a global lock. Any interaction between this cutover and a
    GPU-resident test is unverified.
11. **Anything about h = 32 beyond its handed-in gain of 1.182.** No ratio pair was supplied, so
    its geometry is undetermined and I declined to interpolate it into the results table.

---

# 2026-09-09 — the calibration was a no-op in job 6004: true cause, and the four defects

Worker: `CalibNoop`. Scope: `calibration.rs`, `runner.rs`, `reports.rs`, `model.rs`. Every number
below marked DEMONSTRATED was read by me out of
`training/runs/timexer-xsec2-2500/gens/1/*.report.bin` with `report_cli`, or produced by a
CPU-only test I ran. **No GPU command, no `mlq` command, no training, no evaluation, no
benchmark, no broad `cargo test`.** Nothing here changes a pre-registered threshold.

## A. The defect, reproduced

DEMONSTRATED, `timexer_segment_amplitude_calibration`, gen 1, row 192:

```
held-out sample market-neutral close ratio, uncalibrated = 0.99558365
held-out sample market-neutral close ratio, calibrated   = 0.99558365
held-out full   market-neutral close ratio, uncalibrated = 1.0072936
held-out full   market-neutral close ratio, calibrated   = 1.0072936
```

and on `timexer_segment_calibration_gain`, `calibration partition close-anchor gain applied while
scoring` = `1` and `intrabar-offset gain applied while scoring` = `1` at **all 192 horizons**,
with `three-sigma close-anchor amplification ceiling` = `1` everywhere and
`0.0 anchor and 0.0 offset effective degrees of freedom` in the title.

## B. TRUE ROOT CAUSE

**`calibration.rs:760` (pre-fix): `MeanCalibration::fit`'s intercept refusal gate fired and
returned the identity as an `Ok` value**, through `CurveFit::identity` (`calibration.rs:779` and
`:784`, defined `:691`). The report title carries the refusal verbatim — DEMONSTRATED, extracted
from the `.report.bin`:

> `NO GAIN APPLIED: a pure gain is the wrong parameterization on this block: the best constant
> forecast could earn 7.535e-2 of the persistence MSE at its best horizon against the 2.791e-2
> the amplitude error costs at its worst, which is above the 0.1 share this estimator is allowed
> to leave on the table`

So of the three candidates in the assignment, it is **the fit itself returning identity**. The
other two are ruled out, not assumed:

- **Not "the fitted gain never reached the inversion."** The training path passes the fit's own
  frozen curves straight into the panel: `let frozen = calibration.frozen()` →
  `applied: &frozen` (`runner.rs:2454`/`:2736` pre-fix). The plumbing is correct.
- **Not the load-time install at `runner.rs:2754`, and not the training assert at
  `model.rs:2901`.** `set_mean_gain` is only reached from `load_checkpoint`, which `train` never
  calls; the assert never fires because a training run never installs a gain, by design. The
  `calibrated` series is produced arithmetically by `AmplitudeSplit::ratios`
  (`reports.rs:2496-2509`), which needs no installed gain at all.
- **The mechanism of the bit-identity is exact, not approximate.** At `anchor = offset = 1`,
  `ratios` evaluates `ratio(h, anchor, offset)` for the calibrated series and `ratio(h, 1., 1.)`
  for the uncalibrated one — literally the same call with the same arguments. Bit-identity is
  therefore the *signature* of a frozen unit gain, and it is what made the no-op invisible.

### B.1 Why the gate fired — two implementation deviations from the pre-registration

I am reporting these and **deliberately not fixing them**: both would change this arm's outcome
and I read the results first. Changing them is a scientific decision for the coordinator with the
pre-registration in hand, not a worker's post-hoc repair.

1. **The gate measures a four-channel `ȳ`, against a threshold pre-registered on the close
   channel.** `Moments::intercept_ceiling` sums `ȳ_c²/bars` over all four DECODED channels. The
   high and low target channels have *structurally* nonzero means — a bar's high is never below
   its close — so this quantity is nonzero on any candle data whatsoever. §2.3 above pins the
   threshold against a control-checkpoint margin of `4.63e-5`, a close-channel number; the landed
   code compares `7.535e-2` to it, a factor of 1600 larger.
   **DEMONSTRATED that this is the intrabar offset and not market drift**, from the shape of
   `calibration partition constant-forecast ceiling` on `timexer_segment_calibration_moments`:

   | h | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 192 |
   |---|---|---|---|---|---|----|----|----|-----|
   | ceiling | .07535 | .04801 | .02314 | .01544 | .00815 | .00381 | .00194 | .00107 | .00060 |
   | ceiling·h | .0754 | .0960 | .0926 | .1235 | .1304 | .1219 | .1241 | .1372 | .1154 |

   `ceiling·h` is flat to a factor of 1.8 across a 192-fold horizon sweep, i.e. the ceiling obeys
   `∝ 1/h`. A market-drift intercept must GROW: in these units `ȳ_h ∝ h·μ/σ` and `E[y_h²] ∝ h`,
   so its share `∝ h·μ²`. A NON-accumulating per-bar constant — exactly the intrabar spread —
   gives `ȳ_h ≈ const`, `E[y_h²] ∝ h`, share `∝ 1/h`. The measurement is the second law, not the
   first.
2. **The gate compares the worst horizon of one quantity against the worst horizon of another.**
   DEMONSTRATED: the intercept max `7.535e-2` is at **h = 1** and the amplitude-cost max
   `2.7908e-2` is at **h = 171**. The pre-registered statement in §2.3 is per-horizon ("could earn
   at that horizon"); a single short-horizon intercept therefore blanked the gain at all 192
   horizons and in both coordinates.

## C. Defect 2 — SILENT IDENTITY FALLBACK: every path found, and what each became

Four paths could yield the identity. All four now return `Err`; `CurveFit::identity` and
`CurveFit::unidentifiable` are **deleted**, as is `FrozenGain::is_identity` and the load-time
print that claimed a checkpoint "identified no amplitude" (which can no longer happen).

| Was | Now |
|---|---|
| intercept gate → identity + reason (`calibration.rs:760`) | `ensure!` naming both quantities **and their horizons**, plus "This aborts rather than applying the identity" |
| `usable < 3` → identity (`:1092`) | `ensure!(usable >= 3, ...)` naming the count and the coordinate |
| no solvable penalty in the grid → identity (`:1131`) | `bail!` naming the grid size and the identified-horizon count |
| fitted gain left the positive range → identity (`:1167`) | `ensure!` naming how many coefficients left it |

None of the four is category (a) "the data really says gain = 1": each is "the amplitude could not
be fitted", which under a gain-of-1 rendering is indistinguishable from a measured unit amplitude
on every chart and in the manifest. A measured gain of 1 remains a legitimate, reportable value —
it just can no longer be manufactured by a refusal. `fit_curve` returns `Result<CurveFit>` and
`MeanCalibration::fit` propagates; `runner.rs` wraps the call with the step and the fit-draw size
so the abort names where in the run it happened.

One defect found while writing the abort: with the identity gone, a block that solves NO horizon
left `worst_amplitude = -inf` and the intercept gate then refused with `h=0` and `-inf` in the
message — a parameterization verdict on a population that identified nothing to parameterize.
Guarded ahead of the gate with its own message; the CPU test below is what caught it.

## D. Defect 3 — the 2048-origin fit draw

**Where it came from.** `args.eval_origins`, `#[arg(long, default_value_t = 2048)]`
(`runner.rs:133`), reused verbatim at the fit draw with the stated reason "drawn to the same size
as the held-out sample, so fitting at every report interval costs one sample-sized pass rather
than a full-split one" (`runner.rs:2143`, pre-fix). It is the evaluation-preview knob. **Nothing
about the estimator's variance was ever charged against it.** Stated as asked: it was a default,
never justified.

**It is measurably inadequate.** DEMONSTRATED from `timexer_segment_calibration_gain` at step
2500, where the SAME held-out split is measured twice — once on the 2048-origin strided draw and
once on all 433,303 origins (that population size is itself DEMONSTRATED, from the
`timexer_segment_calibration` title):

| h | 2048-origin draw | full 433,303 | absolute error |
|---|---|---|---|
| 1 | 0.7185 | 1.5407 | 0.822 |
| 8 | 1.6263 | 1.4205 | 0.206 |
| 64 | 0.7553 | 0.6300 | 0.125 |
| 192 | 0.5822 | 0.3066 | 0.276 |

The decision-theoretic reading, since the applied gain's whole justification is the MSE cross term
`-(β-g)²·Var(f)/P`: at h = 1 the population's answer is `β = 1.54`, so leaving the gain at 1 costs
`(1.54-1)² = 0.29`, while applying the 2048-origin estimate costs `(1.54-0.72)² = 0.67`. **At the
short end the correction is 2.3x worse than no correction**, and the estimator's own guard cannot
catch it: the three-sigma bound is one-sided by design (`AMPLIFICATION_SIGMAS`, §2.6) and this
error is a SHRINK. At h = 192 the same arithmetic is favourable — `(0.31-1)² = 0.48` against
`(0.31-0.58)² = 0.076`, i.e. 84% of the benefit is still collected — so the inadequacy is
specifically at the short end, where `|β-1|` is small and the estimate's error is not.

**Replaced, not justified.** `AMPLITUDE_FIT_ORIGINS = 32_768` (`runner.rs`), its own constant so
the fit is never resized by an evaluation knob again. 16x the draw ⇒ SE/4 ⇒ the demonstrated h = 1
error of 0.82 falls to ≈0.21, inside the 0.54 the correction is worth. **Cost, in scoring
milliseconds**: job 6004 logged 833 ms for the step-1000 amplitude phase, which covers the 2048
calibration origins AND the 512 training origins, so ≤0.325 ms/origin (an upper bound — that
figure carries two passes' fixed overhead). +30,720 origins ⇒ **≤10.0 s per evaluation**, i.e.
833 ms becomes ≈10.8 s; ≈+30 s over that arm's three evaluations against a 660 s run, and 7.6% of
the full-split pass the same evaluation already pays for. Memory is unchanged: the pass batches at
`--eval-batch-size` (64) and the same function already runs 433,303 origins at epoch end.
Disjointness needs no new proof — `Blocks::per_ticker` is already run over the FULL calibration
and validation partitions at startup (`runner.rs:2066`), so any subset draw is covered.

## E. Defect 4 — placeholder zeros: the premise does not reproduce, and what I did find

**UNRESOLVED as stated, and I will not present it as a verdict.** On
`timexer_segment_horizon_steps_gain` the `held-out full close MSE-optimal forecast gain at horizon
h` series reads **exactly `NaN`**, not `0.000`, at steps 1000 and 2000 — DEMONSTRATED for all
seven decision horizons — and `1.8811907` at h = 1 / step 2500 (which is the `1.881` in the
assignment). The gap is already emitted: `scored_row` keys on `validation_is_full`
(`reports.rs:798`), a step with no full pass has no full row, and `per_horizon` maps that to
`f32::NAN` (`reports.rs:1070`). I audited all nine step-indexed per-horizon bases
(`horizon_steps`, `_signal`, `_signal_error`, `_pooled`, `_population`, `_decomposition`, `_gain`,
`_best_scale`, `_calibration`) and **every** `held-out full` series is `NaN` at step 1000. The TUI
renderer filters non-finite values into gaps (`tui/src/report_renderer.rs:226`, `:288`, `:1283`).
So a `0.000` reading did not come from this run's artifacts; it is either a different chart
surface or a stale reading, and the sibling's fix is already in effect here.

**Two real instances of the pattern, found on a per-horizon base and fixed** — both on
`timexer_segment_calibration_moments`:

1. `calibration partition amplitude cost of leaving the gain at 1` came from
   `solved.map_or(0., ...)` (`calibration.rs:747`, pre-fix): a horizon `solve_horizon` refused was
   charted as `0.0`, which reads as "fixing this horizon's amplitude is worth nothing" — a
   measurement the block never made. Now `f64::NAN`, i.e. a gap.
2. `calibration partition constant-forecast ceiling` returned `0.` when the horizon had no valid
   target bars or no persistence (`calibration.rs:621`, `:632`, pre-fix). This one had teeth: `0.`
   is the most PERMISSIVE value the quantity has, so an unmeasurable horizon passed the intercept
   gate on evidence that does not exist. Now `NaN` — and because `f64::max` silently skips `NaN`
   in the permissive direction, `fit` additionally `ensure!`s that every horizon's ceiling is
   finite and names the horizons that are not.

The `..._weight`, `..._standard_error` and `Moments::channel_gain` paths were already `NaN`-for-
unmeasured and needed nothing. The remaining `|_| 0.` occurrences in `reports.rs` are constant
REFERENCE lines ("zero information 0.0") and the `map_or(0, ...)` ones are `usize` counts behind
an `ensure!(horizon > 0)`; neither is a placeholder for an unmeasured value.

## F. `model.rs`: no change, and why

Audited and left alone. `set_mean_gain` validates the curve and installs it or fails
(`model.rs:1800-1815`); `gained` returns the decode untouched when there is no gain
(`model.rs:1825-1832`), which is the uncalibrated model rather than a fallback; the `losses`
assert (`model.rs:2901`) is correct and is not on the causal path of this defect. Ownership of the
file did not imply an edit to make.

## G. Verification

```
./torch-env.sh cargo check -p trading_bot_0 --tests   → 0 errors
cargo check -p trading-bot-tui --tests                → 0 errors (no base added or removed)

cargo test -p trading_bot_0 --lib -- amplitude_report_tests aborts_the_fit
running 5 tests ... test result: ok. 5 passed; 0 failed
cargo test -p trading_bot_0 --lib -- timexer_segment::calibration::tests
running 14 tests ... test result: ok. 14 passed; 0 failed
```

The two tests the fix is proved by:

- `reports::amplitude_report_tests::the_calibrated_ratio_differs_from_the_uncalibrated_one_by_the_applied_gain`
  — a hand-built non-unit curve (`anchor` 0.40→0.65, `offset` 1.30→1.05, never 1 at any horizon)
  over moments with the offset column empty on the close channel exactly as `decode_joint` leaves
  it. For every horizon it pins BOTH written series against the closed form expanded longhand
  from the moment fields (not against `pooled_gained_ratio`, which would check the reduction
  against itself), asserts the implied delta is `> 0.02` so the test cannot pass vacuously, and
  asserts `calibrated != uncalibrated` — the exact assertion job 6004 would have failed. It also
  covers the `Emission::Calibrated` direction, where the un-gained series is the ratio at `1/g`.
- `calibration::tests::a_constant_forecast_aborts_the_fit_instead_of_freezing_the_identity` and
  `..::an_intercept_worth_more_than_a_tenth_of_the_amplitude_error_aborts_the_fit` — both were
  tests that PINNED the banned behaviour (`unidentifiable.expect("a named refusal")`,
  `frozen().is_identity()`); rewritten to require `Err` with the naming message, and the second
  additionally requires the message to name the refusing horizons.
- `calibration::tests::horizons_with_no_calibratable_amplitude_are_carried_by_their_neighbours`
  gained the gap assertion for defect 4: the unsolved horizons' amplitude cost must be `NaN`, the
  solved ones positive, and every intercept ceiling finite.

## H. What a reader must NOT conclude

1. **Job 6004 is not evidence about the amplitude calibration.** Its gain was identically 1 at
   every horizon, so every `calibrated` series on that arm is the uncalibrated model relabelled.
   The arm's other findings stand; its calibration findings do not exist.
2. **This fix does not make the gain apply.** With the gate's math untouched, the same block now
   ABORTS the run at its first evaluation (~step 1000, ~4 minutes in) with the numbers and the
   horizons in the message, instead of shipping 2500 steps of checkpoints stamped `v11`
   "calibrated" that carry no calibration. Making the gain apply requires a decision on B.1 that
   is not a worker's to take after seeing the result.
3. **The 32,768-origin draw is sized from ONE observed sample-vs-full discrepancy**, treated as
   ≈1 SE. That is a single draw, so the SE it implies is itself noisy; the direction (2048 is too
   few at the short end) is DEMONSTRATED, the exact multiple is [INFERENCE].
4. **`AMPLIFICATION_SIGMAS` remains one-sided** and I did not change it, but D shows the
   unbounded-shrinkage side is where a noisy fit does its damage. That is a live estimator
   question, unaddressed here.

# 2026-09-09 — the intercept gate restored to its pre-registered form (per-horizon, close channel)

Follow-up to §B.1 above, which named two implementation deviations and deliberately left them.
This section lands them. **Nothing here moves a threshold.** `0.1` is unchanged and is now
`INTERCEPT_CEILING_SHARE` (renamed from `OFFSET_CEILING_SHARE`, which collided with the
`IntrabarOffset` coordinate). What changed is the QUANTITY the threshold is applied to and the
AXIS it is applied along — the two things §B.1 demonstrated were wrong.

## I.1 Deviation 1 — the quantity: close channel, not four pooled channels

`Moments::intercept_ceiling` now reads the close channel alone:
`bars·ȳ_close²/Σy_close²`, clamped to `[0, 1]`, `NaN` where the horizon has no bars or no
close-channel persistence.

Why the close channel is the right population, from the source rather than from the number it
produces. `decode_joint` emits ONE anchor and three monotone offsets, so `O_close ≡ 0` and the
close row carries the anchor alone. The anchor is the only coordinate a constant can compete
with: it is market-neutral within a timestamp, so no gain on it reaches any constant at all, and
a constant added to it is exactly the drift bet §2.3 refuses. The offset column's MEAN is the
structural intrabar spread — `0 ≤ a ≤ r` and `0 ≤ o ≤ r` fix the sign of every offset — so a
strictly positive `g_offset` already spans the constant it would be tested against. §B.1's
`ceiling ∝ 1/h` measurement is that spread, and charging it to the intercept refuses the
estimator on the very structure its second coordinate exists to model. **The offset coordinate
therefore carries no ceiling of its own**, and this is not the gate being half-deleted: a refused
horizon is refused in BOTH curves (one joint 2x2 solve produces both gains and the columns are
coupled through `ΣC·O`, so a misparameterized anchor contaminates the offset estimate beside it).
A separately-measured offset ceiling was considered and rejected: the only form the carried
moments can express, `Σ_{c≠close} bars·ȳ_c²/Σ_{c≠close} Σy_c²`, is ≈ the pooled `7.5e-2` at
`h = 1` on any candle data whatsoever, i.e. a gate that can never pass — the defect being fixed,
re-introduced under a new name.

## I.2 Deviation 2 — the axis: per horizon, refusing only that horizon

The `worst`-vs-`worst` fold is gone. `MeanCalibration::fit` now computes, per horizon,
`refused_h = cost_h.is_finite() && ceiling_h > 0 && ceiling_h > 0.1·cost_h`, and a refused
horizon:

- loses its WEIGHT, so both curves interpolate it from its neighbours under the roughness
  penalty and its amplification ceiling is the identity — the treatment an unidentified horizon
  already gets, so it can be shrunk on neighbours' evidence but never amplified;
- keeps its MEASUREMENT, signed and unmodified (the refusal is about the parameterization, not
  about the number);
- is gated out of SIZING under its own reason, and is still SCORED, because the gain really was
  applied to the emitted mean and the MSE ratio under it is a measurement either way.

`ceiling_h > 0` is not redundant beside the comparison: `cost_h = (uncalibrated − residual)/Σy²`
is non-negative in exact arithmetic (the unit gain is feasible for the same solve), so it can go
negative only by cancellation rounding, and a horizon whose constant forecast could earn NOTHING
must not be refused by the sign of a 1e-16 residue.

Two guards stay ahead of it as DISTINCT findings, never merged: every horizon's ceiling must be
finite (an unmeasured horizon folded to 0 would pass permissively), and at least one horizon must
have identified an amplitude (otherwise the verdict is about a population with nothing to
parameterize). A new third abort fires when fewer than three horizons survive the refusal, naming
the count and the refusing pairs — `fit_curve`'s `usable < 3` message now also carries the refused
count, so "unfittable" and "misparameterized" cannot be read as each other.

## I.3 Three sizing reasons, kept three

`FrozenGain` carries `intercept_refused: Vec<InterceptRefusal>` (horizon, intercept, amplitude
cost) inside the authenticated manifest — it must ride there, because the applied curve is
positive and smooth at a refused horizon and nothing downstream could reconstruct that its own
evidence was declined. `FrozenGain::gate` returns the new `SizingGate` enum and `gated()` returns
`Vec<(usize, SizingGate)>`:

| reason | cause | rendering |
|---|---|---|
| `InterceptDominates { intercept, amplitude_cost }` | pure gain refused AT that horizon | `intercept dominates, a constant forecast earning 5.000e-1 against the 1.100e-2 its amplitude error costs` |
| `NonPositiveGain(g)` | measured amplitude ≤ 0 | `at -0.0400` |
| `Unmeasured` | block identified no amplitude | `unmeasured` |

The refusal is reported ahead of the measurement when both apply, because it is the stronger
statement: the fit declined that horizon's own solve, so the measurement is not what determined
the applied gain. No clamping, no merging, one `Display` so the panel, the log line and an
account's assumption list cannot drift into three renderings.

## I.4 The gate now ADMITS the measured data — arithmetic, no run

All inputs from §B.1 and §2.3; nothing was executed to produce this.

The registered test at horizon `h` is `ceiling_h ≤ 0.1·cost_h`, i.e. it refuses iff
`cost_h < 10·ceiling_h`. With the close-channel ceiling at the control block's measured scale
`4.63e-5`, the refusal boundary is a horizon whose amplitude error costs **less than `4.63e-4`**
of its persistence MSE.

| horizon | amplitude cost | source | `cost/4.63e-4` | verdict |
|---|---|---|---|---|
| h = 171 | `2.7908e-2` | §B.1, axis max | 60.3× | ADMITS |
| h = 128 | `4.23e-2` | `1.0391 − 0.9968` close ratios, §1 | 91.4× | ADMITS |
| h = 192 | `4.11e-2` | `1.0386 − 0.9975` close ratios, §1 | 88.8× | ADMITS |
| h = 1 | `≈1.04e-2` | `ρ²(1/g*−1)²` at `g* = 3.662`, `ρ = 0.14` | 22.4× | ADMITS |

The h = 1 row is derived, not measured: `uncalibrated − calibrated = (σf/σy − ρ)²` and
`g* = ρσy/σf`, so `cost = ρ²(1/g* − 1)²`; over the observed IC band `ρ ∈ [0.13, 0.16]` it spans
`8.9e-3` to `1.35e-2`, a margin of 19.3× to 29.2×. Equivalently, at the axis's worst amplitude
cost the intercept share is `4.63e-5/2.7908e-2 = 1.659e-3` — **60× below the registered `0.1`**,
which reproduces §2.3's "65× below the threshold" claim to within the difference between the
`2.995e-2` cost quoted there and the `2.7908e-2` job 6004 measured.

**Both deviations had to be fixed; neither alone is sufficient, and neither alone is a
relaxation.**

- Landed form (pooled, max-vs-max): `7.535e-2 > 0.1·2.7908e-2 = 2.791e-3` — **27× over the bar**,
  comparing h = 1 against h = 171.
- Fixing only the AXIS (per-horizon, still pooled): admitting `h` would need
  `cost_h > ceiling_h/0.1`, i.e. `> 7.535e-1` at h = 1, `> 8.15e-2` at h = 16, `> 3.81e-2` at
  h = 32 — all above the axis's `2.7908e-2` maximum. Everything out to ≈ h = 32 stays refused, and
  only h ≳ 64 becomes admittable. Half the axis would still be blanked.
- Fixing only the QUANTITY (close channel, still max-vs-max): `4.63e-5 ≤ 2.791e-3` passes, so this
  is the deviation that produced the blanking. The axis fix is required for the registered
  SEMANTICS ("could earn at that horizon") and to stop one horizon speaking for 191, not to
  produce the admission.

**Where the gate would still legitimately refuse.** Only at a horizon whose amplitude cost is
below `4.63e-4` — a horizon where fixing the amplitude buys under 0.05 % of the persistence MSE,
which is the module's own "arithmetically real and economically empty" band. No horizon in job
6004's block is there: the smallest cost anywhere on the axis that any of the four sources above
bounds is `≈8.9e-3`, 19× above the boundary. A horizon whose close-channel `ȳ` genuinely GROWS
with `h` (share `∝ h·µ²`, real drift) would cross it at the long end, and that is exactly the
refusal the registration wants and the pooled quantity could never have detected, since its own
`1/h` shape ran the other way.

## I.5 Verification, and what was NOT verified

Scoped checks only. `./torch-env.sh cargo check -p trading_bot_0 --tests` → 0 errors;
`cargo check -p trading-bot-tui --tests` → 0 errors (no report base changed — one series label
did: `constant-forecast ceiling` → `close-channel constant-forecast ceiling`).

CPU-only, narrow filters, all passing:

- `calibration::tests::a_large_pooled_intercept_over_a_negligible_close_one_is_fitted_at_every_horizon`
  — job 6004's case as a fixture: close share stated at `4.63e-5`, intrabar share at `0.1` per
  channel (pooled `≈8.3e-2`, reproducing the measured `7.535e-2`). Asserts the fixture really
  carries the defect (`pooled > 0.1·cost` at every horizon, computed IN the test from the moment
  fields), that the registered close-channel form admits every horizon, and that the returned
  curve is the injected `0.6` rather than an identity.
- `..::a_dominant_intercept_refuses_its_own_horizon_and_leaves_the_rest_fitted` — one horizon at a
  `0.5` close share. Exactly that horizon is named with both quantities; its weight is zero in
  BOTH curves; the other 31 keep their weight and their `0.6`; the refused horizon is carried by
  its neighbours to `0.6` and is asserted NOT to be `1` (the bit-identity trap); its measurement
  survives as `Some(0.6)`; `tradable` is false there and true at its neighbours.
- `..::an_intercept_that_dominates_every_horizon_aborts_the_fit` — the whole-axis case still
  aborts, naming `32 of 32 horizons` and the refusing pairs.
- `..::the_three_sizing_refusals_remain_distinguishable_in_the_frozen_record` — one refused, one
  measured-negative, one unmeasured horizon, each built out of the MOMENTS rather than edited into
  the frozen gain, asserted distinct as enum variants and as text.
- `runner::tests::universe_checkpoint_authenticates_objective_schema_and_weight_bytes` — all three
  reasons round-trip through the manifest JSON and its digest, still distinguishable on read.
- `reports::amplitude_report_tests::` (3 tests) — including
  `the_calibrated_ratio_differs_from_the_uncalibrated_one_by_the_applied_gain`, kept intact.

**NOT verified, all of it GPU-bound and none of it run here:** no training step, no evaluation, no
preview, no benchmark, no probe, no mlq submission. Specifically unverified: that the re-run arm's
block actually admits at all 192 horizons (§I.4 is arithmetic on §B.1's numbers, not a
measurement); the fitted curve's shape on real moments; whether the calibrated MSE ratios move
below persistence at h = 128/192; and the interaction with the 32,768-origin fit draw. The
manifest format changed (`FrozenGain` gained a required field, no serde default), so **every
checkpoint written before this change fails to load** — deliberate, per the no-backward-
compatibility constraint, and the arm is being re-run anyway.

**One thing a reader must not conclude.** §H.2 above ("this fix does not make the gain apply") is
superseded for the intercept gate only: the gate no longer aborts on job 6004's block, on the
arithmetic in §I.4. Every other entry in §H stands, and the claim that the gain APPLIES and HELPS
remains unmeasured until the coordinator's re-run.
