# The predictable-information ceiling, and why the future-aware teacher was rejected

Status: **measured**. Job 5453 ran the ceiling on `held-out full` and it landed on the branch
I gave 15% probability: the martingale-residual assumption fails in the REVERSAL direction at
**191 of 192 horizons**. What that failure measured is worth more than the ceiling would have
been, and section 0 is now the headline. Estimator in
`trading_bots/src/torch/timexer_segment/teacher.rs`, driver at the end of `runner.rs`, three
bases: `timexer_segment_information_ceiling`, `timexer_segment_variance_ratio`,
`timexer_segment_return_autocorrelation`.

> **STRIDED-DRAW CAVEAT, job 5483, applies to EVERY number in this file measured on
> `held-out full`** - the `0.1666` one-bar ceiling input, the `+0.1011` gap at `h = 64`, every
> ceiling and student IC in §0-§11, the `V_h/D_h` curves and the plateau fits. All of them are
> **strided-draw measurements pending anchored recomputation**. They are the BEFORE column and
> are not deleted.
>
> `evaluate --placement both` on the same weights measured within-timestamp IC on `held-out
> full` at .0615 -> **.1621** (h=1), .0530 -> **.1261** (h=64), .0416 -> **.0872** (h=192),
> ratios 2.07-2.72x, with placement the only difference. The strided draw's mean cross-section
> width is **10.6 names**; demeaning at n≈10 attenuates the very quantity a within-timestamp
> statistic measures. The anchored draw has mean width 1,666.3 over 245 shared anchors.
>
> **The student moved 2.4x and the ceiling did not, because `ceiling-timexer-segment` has no
> `--placement` flag.** Naive arithmetic against the un-recomputed ceiling would read h=1 at
> .1621/.1666 = 97.3% of achievable and flip §11's decision rule to its opposite branch. **That
> flip is unearned** until both sides sit on the same draw, and taking it would be exactly the
> error this file spends §18 retracting. **First code task next session: thread `--placement`
> into the ceiling pass.** It is target-only, touches no model, and is cheap.

---

## 0. Result: the market-neutral targets mean-revert inside the window, hard

### 0.1 What job 5453 returned

433,303 of 433,303 origins, 106.5 s, peak allocator 3,111 MiB. `h = 1` printed
**0.16660** exactly - hard falsifier #1 passed on an arithmetic identity. Every other horizon
printed an unidentified ceiling, first failing at `h = 2`: the measured within-window cross-bar
covariance is more NEGATIVE than a one-bar ceiling of 0.1666 IC allows.

Measured variance ratio `V̄_h/D̄_h` on `held-out full`:

| h | 1 | 8 | 16 | 32 | 64 | 128 | 192 |
|---|---|---|---|---|---|---|---|
| `V̄_h/D̄_h` | 1.00000 | 0.50225 | 0.45587 | 0.39487 | 0.40010 | 0.32537 | 0.34706 |

A random walk reads 1.000 at every horizon. Our 192-bar cumulative market-neutral return
carries **35% of the variance that the same 192 bars would accumulate if they were serially
uncorrelated**.

### 0.2 Both pre-registrations were refuted, mine by more than the hypothesis it opposed

I pre-registered `VR'_192 ≥ 0.70` with a point at 0.89 and a plateau at `Q = 0.90`. R-ALL - the
hypothesis that the whole `h = 192` amplitude defect is reversal - needed `≤ 0.02`. Measured is
**0.347**: outside my stated interval by a factor of two, and 27x above R-ALL. **Neither
pre-registration survives.** A discriminator built with two orders of magnitude between its
branches returned an answer in the middle, and the honest reading is a third one, not a
confirmation of either party. I was wrong by more than the hypothesis I was arguing against.

What survives of my reasoning is the MECHANISM and not the MAGNITUDE, and the distinction is
load-bearing. The shape is an MA(1) shape - a steep fall that flattens - and my form
`VR_h = 1 + 2ρ₁(1 - 1/h)` fits it. But the plateau level implies `ρ₁ = (Q - 1)/2 ≈ -0.325`,
which is an order of magnitude larger than ordinary bid-ask bounce in 5-minute data. So
"small, transient microstructure" is refuted by the parameter of my own model. That is exactly
why `timexer_segment_return_autocorrelation` now MEASURES the lag-1 cross-sectional
autocorrelation directly and charts it beside the value the plateau implies: an MA(1) reading
that implies `ρ₁ = -0.325` must be checkable against the actual lag-1 number, not inferred
from the plateau it was fitted to.

### 0.3 One correction to the amplitude accounting, and it is not cosmetic

0.347 is `V̄_h/D̄_h`, which divides by the per-bar variances the window actually CONTAINS. The
decoder's `√h` shape gain assumes `V_h = h·V_1`, so the quantity that prices the gain is
`VR'_h = V̄_h/(h·V̄_1)`, and the two differ by `D̄_h/(h·V̄_1)` - the window-average per-bar
cross-sectional variance over the FIRST bar's. That factor is the intraday volatility profile,
not reversal, and attributing it to reversal is the same class of error as attributing rank to
information. Both curves are now emitted; neither is called "the" variance ratio.

[HYPOTHESIS] `VR'_192 ∈ [0.31, 0.40]`: origins are spread through the session, so the
first-bar variance averaged over origins should sit within ~15% of the window average.

On the measured `V̄/D̄`, the accounting is: `σ_y` grows like `√h·√VR`, so reversal contributes
`1/√0.347 = 1.698x` of the DEMONSTRATED 9.2x rise in `σ_f/σ_y`. On a log scale that is
`ln 1.698 / ln 9.2 = 23.9%` of the defect. **Reversal is real and large and still does not
explain the amplitude failure**; the residual 5.42x stays where `GainTwoSided` put it, in the
decoder's learned scale. The two fixes are not the same fix: a frozen data-derived `VR'` scale
removes 1.698x by construction and costs zero parameters, and the two-sided gain removes the
rest.

### 0.4 For the portfolio consumer

A 5-minute book that sizes risk on `√h` variance growth over-states the risk of a 64-bar hold
by `1/√0.40010 = 1.581x`, and of a 192-bar hold by `1.698x`. That is a position-sizing error in
the conservative direction, so it costs return rather than causing a blow-up, but it is a real
1.6x.

---

## 0.5 Item 2: is the ceiling re-identifiable once reversal is admitted?

**Assumption-light: no, and provably so.** The martingale construction rests on
`Var(S_e) = Σ_j Var(e_j)` for the unpredictable residual `e`. Reversal breaks it, and the only
bound left uses no serial structure at all. Because `S_m = E[y_h | F_t]` is a conditional mean,
`V_h = Var(S_m) + Var(S_e)` EXACTLY - no covariance term - so a ceiling needs a lower bound on
`Var(S_e)`, equivalently an upper bound on `Var(S_m)`. Cauchy-Schwarz gives one:
`Var(S_m) ≤ h·Σ_j Var(m_j) ≤ h·c₁²·D_h`, hence

```text
ceil_h ≤ c₁·√(h·g_h)  ,   g_h = D̄_h/V̄_h
```

This is exact at `h = 1` and vacuous - it reaches 1 and states nothing - at
`h ≥ VR_h/c₁² = 0.35/0.02776 = 12.6`. **So from `h = 13` onward the predictable-information
ceiling is UNIDENTIFIED from the targets' second moments, at every horizon this project
trades.** It is emitted as `accumulation-bound ceiling IC, no serial assumption, vacuous past
h 20` so the vacuity is visible on the panel rather than asserted here.

**With one further stated assumption: yes, and the plateau supplies it.** Let the unpredictable
residual's own long-run variance ratio be `Q_e`. Then `Var(S_e) = Q_e·Σ_j Var(e_j) ≥
Q_e·(1 - c₁²)·D_h` and

```text
ceil²_h = 1 - Q_e·(1 - c₁²)·g_h
```

**(A1'), the attribution assumption:** `Q_e` equals the fitted plateau `Q` - the whole long-run
variance-ratio deficit belongs to the unpredictable innovation and none of it to a reverting
predictable component. This is a genuine assumption and not a derivation: the only data-implied
bound is `Q_e ≥ VR_h - h·c₁²`, which goes negative at the same `h = 12.6`, so second moments of
the targets cannot separate "the predictable part reverts" from "the innovation reverts".

It is, however, indirectly falsifiable, which is why the autocorrelation base exists. If the
reversal is lag-1 - bid-ask bounce, liquidity provision - then attributing it to the
unpredictable innovation is physically correct, because those effects are not forecastable at
`t`. If the MEASURED `ρ₁` comes back far smaller than the implied `-0.325`, the reversal is
spread over many lags, slow predictable mean-reversion becomes the better explanation, and
**(A1') is wrong in the direction that INFLATES the ceiling** - the failure mode that matters.

**What (A1') yields, and it is a clean result.** Inside the plateau `VR_h = Q`, so `Q·g_h = 1`
and the bound collapses to

```text
ceil_h = c₁   exactly, for every h in the plateau
```

The long-horizon ceiling is neither larger nor smaller than the one-bar ceiling: the total and
the unpredictable accumulation take the same `Q` discount, so the predictable SHARE is
unchanged. At `c₁ = 0.1666` against the DEMONSTRATED student, the gap at `h ≥ 32` is
`0.1666 - 0.0655 = +0.1011` at `h = 64` and `0.1666 - 0.0509 = +0.1157` at `h = 192` - **the
`+.010` distillation gate clears by a factor of ten**.

And the (A2) sensitivity vanishes under this bound. The break-even one-bar ceiling reduces to
`c₁* = ρ̄_h` itself: the student is on the ceiling exactly when `c₁` equals its own IC. With
`ρ̄_64 = .0655` and `ρ̄_192 = .0509`, both BELOW the measured `h = 1` IC of `.0833`, the verdict
"there is IC left at `h ≥ 64`" holds for **every admissible `c₁`**. Unconditional in (A2),
conditional only on (A1'), and (A1') is checkable against the autocorrelation panel.

**The consequence of (A1') failing, stated as sharply as it deserves.** If the measured `ρ₁`
comes back far smaller in magnitude than the implied `-0.325`, the reversal is spread over
many lags, slow PREDICTABLE mean-reversion becomes the better explanation, and (A1') is wrong
in the direction that **INFLATES** the ceiling. That is the dangerous direction: it makes the
model look worse than it is, and it would have us spend compute chasing a gap that is not
there. This is the single most consequential falsifier in the batch and it costs one number
from a pass we are running anyway.

`tests::the_plateau_bound_is_exactly_the_one_bar_ceiling_inside_the_plateau` pins the collapse
to `c₁` on a reverting fixture and pins the accumulation bound reaching 1 at the long end.

### 0.6 What the ceiling licenses, and what it does not: read against the ESS geometry

The `+0.1011` gap at `h = 64` invites the conclusion "add long-horizon capacity". **The
ceiling does not license that conclusion. It licenses the opposite one**, and the reason is
the shape of the bound rather than its level.

Under (A1') the ceiling is **FLAT at `c₁` across the whole plateau**. It is not larger at
`h = 64` than at `h = 1`. So there is no additional long-horizon information: the long-horizon
predictable component is the accumulation of the per-bar predictable components and nothing
else. The gap widens with the horizon - `+0.083` at `h = 1`, `+0.1011` at 64, `+0.1157` at 192
- entirely because the STUDENT decays (`.0833 → .0655 → .0509`) against a constant bound, not
because the bound rises. **The gap at `h = 64` is the gap at `h = 1`, transported.**

Set that beside `TemporalSplit`'s demonstration that the long end is data-bound - about 504
independent non-overlapping 192-bar windows exist in the whole training span, essentially all
of them represented by step 2000. The two results are not in tension and they are not the same
claim:

- **The ceiling is about where the information IS.** It says the `h = 192` predictable share
  is the one-bar predictable share, so everything worth learning at the long end is already
  present in the one-bar relationship.
- **The ESS geometry is about where the EVIDENCE is.** Supervising a 192-bar cumulative target
  directly spends gradient on a statistic with ~504 independent draws, to learn a quantity
  that is a deterministic accumulation of a one-bar quantity backed by 2.46 M rows and 31 M
  distinct outcomes.

Read together they say the same thing from two directions: **supervise the short horizon well
and PROPAGATE, rather than buy long-horizon capacity.** The long-horizon scale should then come
from the measured `VR'` curve - a data-derived constant with the full sample behind it - and
not from a learned per-horizon scale with ~504 effective samples behind it. That is also why
the amplitude fix and the information fix are different fixes and only one of them is a
parameter.

A second consequence worth stating: under (A1') the ENTIRE 192-point ceiling curve is
determined by the single unknown `c₁`. The measurement problem collapses from "bound the
ceiling at 192 horizons" to "bound `c₁`", which is a far better-posed question on a population
with orders of magnitude more independent evidence. That is the natural follow-up and it is not
in this ticket.

**And this whole reading is conditional on (A1').** If the reversal lives in the PREDICTABLE
component rather than the innovation, then a reverting predictable component is a genuinely
multi-bar phenomenon that the one-bar relationship does not contain, there IS long-horizon
specific structure, and "supervise short and propagate" weakens. So the lag-1 measurement does
not merely validate a bound - **it decides which arm to build next**, and it costs one number.

---


## 1. The future-aware teacher is vacuous, and this is a proof

The assignment asked for a teacher that sees the realized window `t+1..t+H` through a latent
of the student's own width (512, and a narrower 32), measures its IC on the same targets, and
calls that the ceiling. It measures nothing, and the failure is not a matter of degree.

A latent that is an arbitrary function of the future window can carry `y_{t,h}` itself in one
coordinate. At width 7 it carries all seven `DECISION_HORIZONS` coordinates exactly. So the
teacher's within-timestamp IC is **1.000 at `d = 512`, at `d = 32`, and at `d = 7`**, with no
training, no optimization, and no data. Narrowing below 7 does not rescue it either: it only
forces the teacher to choose which horizons to reproduce, and the 192 targets are
near-duplicate cumulative sums, so a single coordinate already reproduces most of them.

**A width bottleneck is a RANK restriction. Rank is not information.** This is exactly the
error `--horizon-mean basis:8:8` was rejected for - restricting the predicted function to 8
basis columns was an amplitude prior in intent and a rank restriction in fact, and it
destroyed the correlation gain (D `.00481 -> .00116`) while barely moving the penalty. The
same confusion, applied to a teacher, produces a ceiling of 1.0 and a report nobody can use.

`tests::a_future_aware_bottleneck_teacher_reaches_unit_ic_at_any_useful_width` exhibits the
construction rather than asserting the claim: it builds the width-`d` selector, at
`d ∈ {512, 32, 7}`, and asserts within-timestamp correlation `> 0.999999` at every horizon.

Bounding the future channel in **bits** rather than in width would be a real construction - a
variational information bottleneck traces `IC(R)` against the rate `R` in nats. But the
quantity that answers this project's question is `IC(R -> 0)`, which is by definition the
causal ceiling and is identified by no point on that curve. It would cost a second encoder
pass, a rate estimator and an extrapolation, to arrive at an estimate of exactly the quantity
the estimator below computes in closed form from the targets alone, with zero parameters and
zero training. So the future-aware teacher is rejected, and Deliverable TWO (latent
distillation) is not gated on a number that does not exist - it is gated on the estimator
below.

## 2. The estimator

Fix a horizon `h` and an evaluation timestamp `τ` with a cross-section of `n_τ` tickers. The
scored target is the σ-normalized market-neutral cumulative log return
`y_{i,h} = Σ_{j≤h} Δ_{i,j}`. `CausalPatchModel::targets` emits the cumulative form directly
(`model.rs:2422-2430`: `(future - log_close)/σ - β·market_drift`, no `√h` division), so the
per-bar increments `Δ_{i,j}` are **one shifted difference along the horizon axis** and never a
re-derivation from prices, which would not agree bit for bit through the β and σ
normalizations.

Write `Δ_{i,j} = μ_{i,j} + e_{i,j}` with `μ_{i,j} = E[Δ_{i,j} | F_τ]` the ORACLE causal
conditional mean under the full filtration at `τ`. Every moment is cross-sectional at fixed
`τ` and then averaged over timestamps with equal weight, which is exactly the population the
reported IC lives on. Because `μ` is `F_τ`-measurable the law of total covariance splits the
observed cross-sectional covariance with no cross term:

```text
Γ(j, k) = Cov(Δ_j, Δ_k) = Cov(μ_j, μ_k) + F(j, k),   F(j, k) = E[ Cov(e_j, e_k | F_τ) ]
```

For any `F_τ`-measurable forecast `f`, `corr(f, y_h) ≤ corr(E[y_h|F_τ], y_h)`, so the ceiling
is `ceil_h² = Var(μ_{·,h}) / V_h` with `V_h = Var(y_h)`, and

```text
ceil_h² = 1 - ( Σ_{j≤h} F(j,j) + Σ_{j≠k≤h} F(j,k) ) / V_h        (identity, no assumption)
```

Two named inputs close it.

**(A1) The unpredictable component is a martingale difference sequence at the bar frequency:**
`F(j,k) ≥ 0` for `j ≠ k`. Violated exactly by IN-WINDOW FEEDBACK - a shock at `τ+3` that moves
the conditional mean at `τ+7`, which no forecaster standing at `τ` can know. Under
momentum-shaped feedback the omitted term is positive and the estimate is a **strict upper
bound**; under conditional reversal it is negative and the estimate is not a bound at all.
That failure is self-announcing: it drives `ceil²` below zero, and the module renders it as
NaN and names the horizons rather than clamping it into a number a reader would misread as a
small ceiling.

**(A2) A stated one-bar ceiling `c₁`:** `Var(μ_j) ≤ c₁²·Γ(j,j)` at every bar. The off-diagonal
structure cannot identify the diagonal - a signal predictable one bar ahead but carrying no
cross-bar structure is invisible to it - so `c₁` is a CLI input and its value is printed with
the result. It must be an UPPER bound on one-bar predictability, not the measured one-bar IC,
because `ceil²` is monotone increasing in `c₁`.

With `D_h = Σ_{j≤h} Γ(j,j)` and `g_h = D_h / V_h` (the inverse Lo-MacKinlay variance ratio):

```text
ceil_h² ≤ 1 - (1 - c₁²) · g_h
```

At `h = 1` this is an exact arithmetic identity: `D_1 = V_1` because the first target IS the
first increment, so `g_1 ≡ 1` and `ceil_1 ≡ c₁` on any data whatsoever. That is a hard
falsifier of the implementation and is pinned to `1e-9` by test.

### What it bounds

The within-timestamp IC of **any** forecast measurable with respect to information available
at `τ` - any architecture, any feature set, any amount of data, not merely the current trunk -
pooled over the held-out period, on the close channel. Because `min MSE / Var(y) = 1 - ceil²`,
it also bounds the market-neutral MSE ratio the same forecaster can reach, which says how much
of the MSE scoreboard is reachable at all.

### What it does NOT bound

- **A forecaster that trades a subset.** The ceiling is a pooled second moment. A policy
  active only in a predictable regime can exceed it on its own subsample.
- **Anything under conditional reversal.** If in-window feedback is negative, (A1) fails in
  the direction that breaks the bound. The estimator then reads NaN and states nothing.
- **The one-bar ceiling itself.** `c₁` is an input, and the headline swings by roughly 2x
  between `c₁ = .0833` and `c₁ = .1666`. The run prints the whole `c₁` sensitivity table
  because the curve is one monotone function of the measured `g_h`, so no extra pass is
  needed - and the `c₁ = 0` row is the model-free floor that needs no (A2) at all.
- **Non-close channels, or rank statistics.** It is a Pearson second-moment argument on the
  close coordinate.
- **Anything about a specific model.** It is a property of the targets.

### Leakage impossibility

There is no fit. The estimator has no parameters, reads only the target tensor of the scored
population, and performs no selection, so there is no fit block whose separation from the
scored block could be violated: it estimates the scored population's OWN ceiling rather than
predicting it. The single non-target input is the scalar `c₁` from the command line.

The test is `the_ceiling_is_a_function_of_the_targets_and_cannot_be_moved_by_any_forecast`: it
runs the accumulator twice on identical targets, once with an unrelated forecast and once with
the **target itself** as the forecast (the strongest possible leak), and asserts the ceiling
and variance-ratio vectors are bit-identical while the student IC moves from ~0.1 to >0.999. A
ceiling a model could raise would be a ceiling the model had leaked into. It also asserts the
oracle's gap is **negative** - an oracle is not a causal forecaster, so it is allowed to
exceed the bound, and the series must say so rather than clamp at zero.

The student IC it is paired against is scored on the identical retained origins in the
identical pass, so the paired difference is paired by construction rather than by alignment.
A row is retained only if **every** one of its 192 target bars is valid: the estimator
## 3. The base

`timexer_segment_information_ceiling`, horizon-indexed (1..192), **seven** series, all
correlation coefficients on one axis because they all answer one question:

1. `held-out full predictable-information ceiling IC` - the bound, on the paired population.
2. `held-out full ceiling IC over every cross-section of two or more` - the same estimate over
   the WHOLE held-out full split. The ceiling is a ratio of two **unbiased** cross-sectional
   variances, so a two-name timestamp contributes an unbiased summand; a within-timestamp
   *correlation* is neither unbiased nor meaningful at `n = 2`, so the student IC and the
   paired gap keep the project's `CROSS_SECTION_MIN = 20` floor. Two populations, two series,
   never averaged together.
3. `held-out full student IC`.
4. `held-out full ceiling minus student IC` - **the deliverable**.
5. `held-out full standard error of the paired ceiling minus student IC` - delta method in one
   linear combination per horizon, so the ceiling's sampling error and the student IC's are
   correlated through the timestamps they share rather than added as if independent.
6. `held-out full one-bar ceiling at which the student would already be optimal` - **the whole
   (A2) sensitivity, in one number per horizon, in the same unit.** See below.
7. `held-out full martingale-residual assumption violation IC` - **(A1)'s failure as a number
   rather than a silence.** See below.

`1 - ceil²` (the best reachable market-neutral MSE ratio) is a different unit and is printed
and put in the chart title, never drawn on the correlation axis.

### 3.1 The `c₁` sensitivity, as a reading rather than a second chart

`c₁` is a free parameter and the answer scales with it, so a single headline number would be
read without its condition. The whole curve is one *monotone* function of `c₁`
(`ceil² = 1 - (1 - c₁²)·g`), which means the entire sensitivity collapses to **one number per
horizon**: the `c₁` at which the student would already sit exactly ON the ceiling,

```text
c₁*_h = √( 1 - (1 - ρ̄_h²)/g_h )
```

in the same correlation unit as everything else on the axis. The reading rule is what makes it
worth more than a curve: **the MEASURED `h = 1` within-timestamp IC (0.0833) is a LOWER bound
on the true `c₁`** - a realized forecast cannot beat the bound it is measured against - so
wherever `c₁*_h` sits below 0.0833 the verdict "there is IC left at this horizon" holds for
**every admissible `c₁`** and stops being conditional on (A2) at all. `c₁* = 0` exactly is the
strongest such case: the student is below the ceiling even with zero one-bar predictability.

The run prints, per decision horizon, both the number and the verdict in words
(`HOLDS` / `FAILS` against the measured `h = 1` IC), plus the full curve at
`c₁ ∈ {0, .0833, .1666, .25}` computed from the measured `g_h` - so all four ceilings come out
of the one job and the assumption can be changed after the fact without a re-run.
`the_break_even_one_bar_ceiling_recovers_the_input_that_would_make_the_student_optimal` pins
it exactly: plant a persistent drift, hand the student that drift as its forecast so it sits
ON the ceiling by construction, and `c₁*` comes back as the planted one-bar IC at every
horizon (to 0.02) while the gap comes back as 0.

### 3.2 The two NaNs are two different facts, and the artifact says which

A NaN meaning "in-window reversal detected, the bound does not hold here" and a NaN meaning
"this horizon was never drawn" are different claims, and this project has already been burned
by a `0` that meant "never fired". Series 7 is
`√((1 - c₁²)·g_h - 1)`: **exactly 0 wherever (A1) holds**, positive - in correlation units,
so the magnitude of the breach is readable - exactly where in-window conditional reversal has
pushed the construction past the point where it bounds anything, and NaN where the horizon has
no population. So on the chart:

| ceiling | violation | meaning |
|---|---|---|
| finite | 0 | measured, assumption held |
| NaN | > 0 | assumption (A1) broken at this horizon, nothing is stated |
| NaN | NaN | never measured, no qualifying cross-section |

The run prints the same three-way census by horizon list, with the largest violation, and the
unit string carries the table's reading rule. Both branches are pinned:
`in_window_reversal_reads_as_unidentified_and_never_as_a_zero_ceiling` asserts a NaN ceiling
beside a positive violation and the "assumption (A1) broken" verdict;
`a_planted_predictable_component_is_recovered_at_its_closed_form` asserts the violation is
**exactly 0** at every horizon and the verdict is "measured";
`a_thin_cross_section_carries_the_ceiling_and_never_the_paired_gap` asserts the NaN/NaN
branch.

## 4. Costs

| quantity | value | against |
|---|---|---|
| teacher parameters | **0** | there is no teacher network |
| FLOP / training step | **+0** | 16.98 TFLOP unchanged |
| GB / training step | **+0** | ~143 GB unchanged |
| ms / training step | **+0** | 168 ms unchanged |
| peak allocator | **~2.5 GiB bound** | 18,573 MiB current, ~25,900 MiB usable |

Nothing touches the training step: this is an offline subcommand over a frozen checkpoint.
The persistent cost is seven fp64 banks of `[timestamps, 192]` = `7 x 38,190 x 192 x 8`
= **411 MiB**, plus roughly 500 MiB of `[timestamps, 192]` reduction temporaries in `finish`,
on top of the existing forward-only evaluation peak (no backward, and `final_origin` runs the
head `last_only`, so it is far below the training peak). The job resets the CUDA peak counter
before its loop and **prints the measured figure**, so this bound is checked rather than
asserted.

Per-batch traffic added by the accumulator: nine `index_add_` of `[256, 192]` fp64,
read-modify-write, = 5.5 MB per batch, 9.3 GB over the whole 1,694-batch pass. Against the
~238 s the forward pass itself takes, unmeasurable.

Wall clock: `held-out evaluation total` is 1,123 ms per 2,048-origin pass, so 433,721 origins
is ~238 s, plus ~30 s warm startup = **~270 s**, comfortably inside the 10-minute lease with
2x headroom. **One job.** The `d = 512` / `d = 32` pair Main asked to split does not exist,
because the teacher does not exist.

## 5. Pre-registration - written before any number exists

`c₁ = 0.1666` default, so `leak = 1 - c₁² = 0.972244`.

**Hard falsifier.** `h = 1` must print ceiling `0.16660` to five decimals, exactly, with zero
uncertainty. `g_1 ≡ 1` is an arithmetic identity. Anything else is an implementation fault,
not a result.

**Second hard falsifier.** At `h = 1`, `g_1 ≡ 1`, so the break-even one-bar ceiling reduces to
`c₁*_1 = √(1 - (1 - ρ̄_1²)) = ρ̄_1` - it must print EXACTLY the student's own `h = 1` IC. Two
independent series agreeing to five decimals at one horizon by arithmetic identity is the
cheapest available check that the whole reduction is wired correctly.

**Point predictions and 68% intervals** (the measurement's own SE, from a matched simulation
at `n = 24, T = 4000` and `n = 256, T = 128`, is `SE(g) ≈ 0.005` and hence
`SE(ceiling) ≈ 0.015` at `h = 192`):

| h | predicted `g_h` | predicted ceiling (c₁ = .1666) | predicted ceiling (c₁ = 0) | student (step 3000) | predicted gap |
|---|---|---|---|---|---|
| 1 | 1.000 exactly | **0.16660** exactly | 0.000 exactly | .0833 | +0.083 |
| 64 | 0.998 [.985, 1.010] | **0.182** [0.09, 0.235] | 0.045 | .0655 | **+0.116 ± 0.015** |
| 192 | 0.995 [.980, 1.015] | **0.198** [NaN, 0.257] | 0.071 | .0509 | **+0.147 ± 0.015** |

**Signs, pre-registered:**

- `ceiling - student > 0` at every `h ≥ 8`.
- The ceiling curve **rises** mildly with `h` (`g_h` monotone decreasing from 1.000 to ~0.995),
  because a persistent predictable component accumulates faster than its own noise. If instead
  `g_h ≡ 1` to within the SE at every horizon, the market-neutral returns are a clean
  cross-sectional martingale and the ceiling is FLAT at `c₁` - which is itself the answer, and
  is consistent with the measured `ρ_h` being roughly flat (.0833 / .0930 / .0784 / .0490).
- **The gate fires.** `P(gap at h = 64 exceeds +.010 paired IC) ≈ 0.75`.
  `P(the construction reads unidentified, i.e. g_64 > 1/leak = 1.0285) ≈ 0.15`. The remaining
  15% is the gate genuinely failing.
- **Model-free floor.** Even at `c₁ = 0`, which needs no (A2) at all, I predict
  `ceiling(192) ≈ 0.071` against a student `.0509`, so the gate fires on the assumption-free
  component alone.
- **The break-even one-bar ceiling, which is the (A2)-free reading.** Predicted
  `c₁*(64) ≈ 0.048` and `c₁*(192) ≈ 0.000` (clamped, i.e. the student is below the ceiling
  even at `c₁ = 0`), both **below the measured `h = 1` IC of 0.0833**. If that holds, the
  verdict "there is IC left at `h ≥ 64`" is unconditional: it survives every `c₁` consistent
  with what the model already demonstrates at one bar, and (A2) stops mattering. **The verdict
  flips only if `c₁*` comes back above 0.0833**, which needs `g_64 > (1 - ρ̄²)/(1 - 0.0833²)
  = 0.99571/0.99306 = 1.00267` - i.e. a variance ratio below 0.99734, a 0.27% within-window
  mean-reversion effect. That is inside the measurement's own ±0.005 SE on `g`, so this is
  the one number that could genuinely go either way, and it is exactly the number the series
  exists to show.
- **The martingale-residual violation is 0 at every horizon**, probability ≈ 0.85. A positive
  violation anywhere at `h ≥ 64` means in-window conditional reversal and the whole
  construction states nothing at those horizons - a real possibility for 5-minute equity bars,
  which is why it is a series and not a footnote.

**What each outcome means, decided in advance.** This is the measurement that separates Main's
two worlds - "the effective sample is far smaller than the bar count suggests" versus "there is
very little learnable long-horizon structure":

- **Ceiling at `h ≥ 64` well above the student (predicted).** The structure exists and we are
  not extracting it. The step-2000 IC peak and subsequent collapse
  (h=16 `.1137 -> .0086`, h=64 `.0879 -> -.0125`) is then an OPTIMIZATION / effective-sample
  failure, not a data ceiling, and `TemporalSplit`'s 98.5% distinct-outcome coverage by step
  2000 is the mechanism. Deliverable TWO is justified - though on the evidence above the
  higher-value intervention is the sample problem, not distillation.
- **Ceiling at `h ≥ 64` at or below the student.** There is nothing left at the long end, the
  edge is exhausted at `.05-.09` IC, and every architecture proposal aimed at long horizons
  should be cancelled. Note this outcome would also mean the tradable long-horizon edge is
  already fully harvested, which is a *result*, not a failure.
- **NaN at the long end.** In-window conditional reversal dominates; (A1) fails and this
  construction states nothing. The follow-up is then a lag-resolved decomposition of `Γ(j,k)`
  to separate feedback from `F_τ`-measurable signal, which is a second pass over the same
  targets and costs one 192x192 GEMM per batch.
- **`c₁*` at `h = 64` comes back above 0.0833**, i.e. `g_64 > 1.00267`, a 0.27% within-window
  mean-reversion effect. This is the one number that could genuinely go either way, and it is
  **a finding about market microstructure at our horizons, not a failed experiment**: it says
  the market-neutral residual mean-reverts inside the forecast window at a magnitude our
  measurement can resolve, which is itself a tradable structure and a reason the long-horizon
  cumulative targets are harder than their bar count suggests. It must be reported in those
  terms and never as a null.

## 6. Deliverable TWO, and its gate

Not implemented, by design: **the gate has not been evaluated**, because the gate reads the
ceiling and no run exists. If the measured gap at `h ≥ 64` exceeds `+.010` paired IC, the
design is specified here so it can be built without re-deriving it.

- **Target.** Not a future-aware teacher's latent - that object is vacuous and would distil
  the answer, not the predictable component. The only legitimate privileged target on this
  corpus is a *rate-limited* one, and the honest version is: distil toward the **oracle
  per-horizon conditional mean's realizable part**, which the ceiling now bounds. Concretely,
  the teacher becomes a second student trained on the SAME causal inputs with a longer context
  and a longer optimization budget, i.e. a self-distillation target, and the ceiling is what
  says whether that is worth its FLOPs.
- **Loss.** Whitened MSE, not Smooth-L1 and not cosine. Latents are unit-scale after RMSNorm,
  so raw MSE and Smooth-L1 coincide in the linear regime and Smooth-L1's kink only downweights
  the tail where the teacher disagrees most - exactly the informative part. Cosine discards the
  norm, which is where amplitude information lives, and amplitude is the project's measured
  defect. Whitening by the running per-coordinate teacher standard deviation removes the
  arbitrary basis of the teacher's latent, which is what makes the loss a statement about
  information rather than about the teacher's coordinate scaling.
- **Pathways.** Dyadic offsets `k ∈ {1, 2, 4, ..., 192}` - eight residual propagators of
  NextLat's form (`LN` then `MLP`, residual) rather than 192 near-duplicates. Eight is the
  same argument as `OrthoTargets`': 192 near-duplicate cumulative sums do not carry 192
  independent gradients.
- **Schedule.** Auxiliary weight linearly warmed over the first 200 steps then held; a
  cosine-decayed-to-zero auxiliary would make the last third of training a different objective
  from the first, and the whole trajectory is step-matched against control arms.
- **Teacher frozen, never co-trained.** A co-trained teacher is a moving target and reintroduces
  the self-referential collapse mode the whole design was chosen to avoid.
- **What stops the trunk degrading its forecast.** Nothing structural does, and pretending
  otherwise is how auxiliaries get shipped. What is required instead is measurement: the
  auxiliary must be evaluated against a step-matched control on `held-out full` NLL and on the
  per-horizon IC trajectory, and rejected if either degrades. Given the DEMONSTRATED step-2000
  IC peak, the honest test is not NLL at 2000 but whether the `h ≥ 16` IC peak MOVES LATER.

**Collapse detection, priced as first-class.** Two series would be added to the ceiling base's
family (own base, own axis - a variance and a correlation are different units): mean latent
coordinate variance and mean absolute off-diagonal correlation of the latent's coordinate
covariance, both over the origin-token positions of the training batch, written every report
interval so a collapsing run is visible in the TUI within one interval. **A healthy run,
numerically, before any run exists:** after RMSNorm the latent has unit RMS by construction, so
per-coordinate variance should sit at `1/512 = 0.00195` each and their MEAN must stay within
`[0.0015, 0.0025]`; mean absolute off-diagonal correlation at initialization is
`≈ √(2/(π·B))` for batch `B` - at 256 origins, `0.050` - and must stay below `0.15`. Collapse
reads as off-diagonal correlation climbing past 0.3 while the variance mean stays pinned at
`1/512` (RMSNorm hides a rank collapse from the variance but not from the correlation), so the
correlation series is the one that must be watched and the variance series is the control that
says the norm is not doing the work.

## 7. Command line

Registered as `ceiling-timexer-segment`. Run it against `timexer-control-4k` step 3000, the
checkpoint whose `held-out full` IC (`.0655 / .0610 / .0509` at h = 64 / 128 / 192, SE `.0028`)
the gap is measured against:

**As submitted (job 5453).** The launcher is the stamped binary `/var/tmp/tb0_v14`, not
`./trading_bots/run-release-cuda.sh`: that wrapper rebuilds from the live tree, five agents
are landing edits into it, and a measurement whose binary is whatever happened to compile at
launch time is not comparable to anything. v14 carries this code plus `FuseLoss`'s verified
fused loss kernel.

```
mlq submit --name timexer-ceiling-4k --max-parallel-runs 1 --time-limit 10m \
  --cwd "$PWD" -- /var/tmp/tb0_v14 ceiling-timexer-segment \
  --checkpoint training/runs/timexer-control-4k/weights/best \
  --output training/runs/timexer-control-4k/gens/3 \
  --batch-size 256 \
  --one-bar-ceiling-ic 0.1666
```

`c₁` is passed explicitly rather than defaulted so the artifact's own command line records the
assumption the number is conditional on.

The run additionally prints the curve at
`c₁ ∈ {0, .0833, .1666, .25}` from the measured `g_h`, so all four are available from this one
job and no re-run is needed to change the assumption.

### 7.1 The variance-ratio re-run, and why it is worth 350 s

Job 5453's log already carried the variance ratio at the seven decision horizons, so the
HEADLINE did not need a re-run. The shape did. Whether the ratio falls and FLATTENS, and at
what horizon, is the entire difference between "the ceiling is re-identified under (A1')" and
"no target-only bound exists" - and a plateau cannot be fitted to seven sampled points. The
re-run also produces the lag-1 autocorrelation that checks (A1'), the two ceiling series that
carry their assumptions in their labels, and the two populations that decide whether the
curve is freezable.

**Three populations, and the third is the point.** The variance-ratio curve exists to be
FROZEN into training as a data-derived replacement for `√h`. A constant fitted on held-out
data and baked into a training run leaks held-out information into the model, however mildly,
so the only clean source is `training` origins - I withdrew the `[70%,80%)` fit instruction on
that ground and it was accepted. The calibration block and `held-out full` are not the fit:
they are the out-of-sample evidence that licenses freezing it. **If `training` and
`held-out full` disagree materially at `h ≥ 32`, the constant is not freezable and the
disagreement is the deliverable.** All three are on one chart for exactly that reading.

**The training draw is subsampled by whole TIMESTAMP.** `teacher::whole_timestamp_keep` dedups
the origin timestamps, keeps every `k`-th DISTINCT one, and retains every origin sitting on a
kept timestamp, so a cross-section survives entire or not at all. A strided pick over the
ticker-major reference list keeps a scattered subset of each ticker's own timestamps, lands
the survivors on many timestamps with one name each, and makes every within-timestamp
statistic read as blank rather than as an error - the failure that has now cost this project a
trading verdict once. `tests::the_training_draw_keeps_whole_cross_sections_where_a_strided_pick_shreds_them`
pins it on a deliberately IRREGULAR fixture (each ticker over its own offset span, because a
regular one lets a stride pass by luck): every retained timestamp must carry the same name
count it carried in the full list, and the same fixture is then handed a strided pick and
asserted to VIOLATE that, so the test cannot pass by testing nothing.

Cost of the three passes: `3 x 106.5 s` plus ~30 s warm startup = **~350 s**, inside the
10-minute lease. The accumulator is now nine fp64 banks rather than seven; at 38,190
timestamps that is 528 MiB, and the training draw is subsampled to about the same origin count
(`k = 6` over ~267k distinct training timestamps, so ~44.5k timestamps and ~615 MiB). Peak
allocator bound **~3.6 GiB** against the measured 3,111 MiB for one pass and the 25,900 MiB
usable ceiling. Still `+0` FLOP, `+0` GB and `+0` ms on the 16.98 TFLOP / ~143 GB / 168 ms
training step: nothing here touches training.

```
mlq submit --name timexer-varratio-4k --max-parallel-runs 1 --time-limit 10m \
  --cwd "$PWD" -- /var/tmp/tb0_v17 ceiling-timexer-segment \
  --checkpoint training/runs/timexer-control-4k/weights/best \
  --output training/runs/timexer-control-4k/gens/4 \
  --batch-size 256 \
  --one-bar-ceiling-ic 0.1666
```

**Submitted as job 5471 on `/var/tmp/tb0_v17`.** My own line named `v15`, which was stamped
BEFORE this landing and therefore carries neither the plateau fit, `whole_timestamp_keep`, nor
the three-population pass - it would have re-run the same five-series measurement job 5453
already produced. The binary named must be stamped from the tree containing the code being
measured, and checking that is the submitter's job, not the queue owner's.

Pre-registered before the run, from the seven measured points: fitted plateau level
`Q ∈ [0.33, 0.39]` with onset `h₀ ∈ [16, 48]`; `VR'_192 ∈ [0.31, 0.40]`; measured lag-1
autocorrelation at `h = 192` within `±0.06` of the implied `-0.325` if the reversal is lag-1,
and materially smaller in magnitude if it is not - the latter voids (A1') and with it the
`c₁` ceiling at the long end; three-population agreement on `VR'_64` within `±0.05`.


## 8. Verification performed

- `./torch-env.sh cargo check -p trading_bot_0 --tests` - 0 errors.
- `./torch-env.sh cargo test -p trading_bot_0 timexer_segment::teacher` - **10 passed, 0
  failed**: the leakage-impossibility test, the martingale calibration (`h = 1` identity pinned
  to `1e-9`), the planted-component closed-form recovery, the break-even `c₁` closed-form
  recovery, the reversal-reads-NaN-with-a-positive-violation test, the thin-cross-section
  NaN/NaN population test, the teacher-vacuity demonstration, the whole-timestamp draw against
  a strided pick on an irregular fixture, the plateau fit being the tail mean and rejecting a
  `1/h` decay, and the plateau bound collapsing to exactly `c₁` inside the plateau beside the
  accumulation bound clamping to 1 past its crossing.
- `./torch-env.sh cargo check -p trading-bot-tui --tests` - 0 errors;
  `the_meta_chart_list_looks_for_every_registered_writer_base` green, so all three of
  `timexer_segment_information_ceiling`, `timexer_segment_variance_ratio` and
  `timexer_segment_return_autocorrelation` are registered and scanned.
- The estimator's three closed forms were independently reproduced in numpy before the Rust
  tests were written: the martingale case gives `g = 1` and `ceiling = c₁`; the planted
  persistent-drift case recovers `√(h·σ_s²/(h·σ_s² + σ_e²))` to a maximum absolute error of
  `0.0024` over `h = 1..8`; the reversal case gives `g = 180` and NaN.
- The wider `timexer_segment` family was last measured GREEN by siblings with this landing in
  it (`LatentProbe`, 164/0/2-ignored, before my final three tests). I measured my own module
  directly and did not re-run the family, which is the main agent's job once every sibling has
  landed.

## 9. Files

| file | change |
|---|---|
| `trading_bots/src/torch/timexer_segment/teacher.rs` | new: estimator, curve, plateau fit, whole-timestamp draw policy, three report writers, 10 tests |
| `trading_bots/src/torch/timexer_segment/mod.rs` | `pub mod teacher;` |
| `trading_bots/src/torch/timexer_segment/runner.rs` | `teacher` import; `CeilingArgs`, `pub fn ceiling`, `ceiling_pass`, `whole_timestamp_draw` appended at EOF |
| `shared/src/report.rs` | three bases: `timexer_segment_information_ceiling`, `timexer_segment_variance_ratio`, `timexer_segment_return_autocorrelation` |
| `trading_bots/src/main.rs` | `CeilingTimexerSegment` variant and dispatch arm |

---

## 10. The PROPAGATION arm - specification only, no code

Design request from Main, spec before implementation. The thesis: supervise the short horizon
densely, form the long end by accumulation with the measured `VR'` scale, and leave the emitted
192-horizon forecast unchanged in shape.

### 10.1 The trap this spec must not fall into

The obvious propagation arm - emit ONE mean direction and multiply it by a frozen horizon
shape - is `--horizon-mean basis:8:8` with the rank taken from 8 down to 1, and it would fail
harder for the same reason. The flat ceiling says the predictable SHARE is horizon-invariant.
**It does not say the predictable DIRECTION is.** Different names can carry different horizon
profiles, and restricting the emitted mean's rank is a model change, not an amplitude prior.
So: **the mean stays full rank at 192, and nothing about the emitted mean's freedom is
reduced.**

What the bound DOES license is a restriction on the SCALE's horizon profile, which is exactly
an amplitude prior and is exactly the object `basis:8:8` was confused about.

### 10.2 What the head emits and what the loss supervises

**Emit:** `4 × 192` per-bar increment means, plus `4 × 1` per-bar log-scales. Output width
`768 + 4 = 772`, down from `192 × 8 = 1536`.

**Form the deliverable, unchanged in shape:** all 192 cumulative forecasts are still emitted.

```text
cumulative mean_h   = Σ_{j≤h} increment mean_j          (full rank, nothing restricted)
cumulative logscale = per-bar logscale + ½·ln(V_h/V_1)  (frozen 192-vector from the artifact)
```

The trading family, the per-horizon IC series and the breakeven curve all read all 192 and
none of them changes.

**Supervise:** the heteroscedastic Gaussian NLL on the 192 per-bar INCREMENTS, not on the
cumulative sums. This is the whole arm, and the reason is effective sample, not elegance. The
`h = 192` CUMULATIVE column across origins is 2.46 M overlapping sums with ~504 independent
draws behind it. The `j = 192` INCREMENT column across origins is 2.46 M single calendar bars
which do not overlap at all. **Supervising increments moves the long-horizon evidence from ~504
independent draws to the full bar count**, and it does so without touching what the model can
represent. It is also the only form in which the measured `ρ₁ = -0.325` reversal is directly
visible to the loss: consecutive increment means can carry it, while the cumulative objective
sees it only through second differences.

Note the direction of the reweighting. Cumulative-target MSE implicitly weights increment `j`
by the `193 - j` sums it appears in, i.e. it over-weights the earliest bars. Increment
supervision weights them equally. That is a change of training objective and NOT of the scored
metric: everything is still scored on the cumulative targets, unchanged.

### 10.3 Where the scale enters

A frozen 192-vector `s_h = √(V_h/V_1) = √(h · VR'_h)`, read from the artifact the
variance-ratio job writes, applied at exactly the point the persistence-anchored `√h` shape
gain is applied today. `s_h` IS `√h` when `VR' ≡ 1`, so this is a strict generalization and
the current model is the special case.

**Below the plateau onset `h₀` and inside it, the same thing happens: the measured `VR'_h` is
used pointwise.** The plateau is a feature of the CEILING derivation, where a level `Q` has to
be attributed; the scale needs no plateau at all, only the measured curve. `h₀` therefore does
not appear in the arm.

Fitted on `training` origins, per the leakage argument, with the calibration block and
`held-out full` as the licence: **if `training` and `held-out full` disagree by more than
±0.05 on `VR'_64`, the constant is not freezable and this arm does not run.**

### 10.4 The four costs - this arm REMOVES parameters and FLOPs

Stated plainly because Main's prior is correct and it changes the class of the proposal.

| quantity | now | arm | delta |
|---|---|---|---|
| output projection | `1024 × 1536 + 1536` | `1024 × 772 + 772` | **−783,100 parameters** |
| total parameters | 27.2 M | 26.4 M | **−2.9%** |
| TFLOP / step | 16.98 | 16.53 | **−0.45 (−2.7%)** |
| GB / step | ~143 | ~142.1 | −0.88 (−0.6%) |
| ms / step | 168 | **150.3** [147, 158] | **−17.7 ms (−10.5%)** |
| peak allocator | 18,573 MiB | ~18,280 MiB | −293 MiB |

The frozen scale vector is 192 `f32` - 768 bytes, not a parameter, never a gradient.

Arithmetic: the output GEMM is `2 · 96,000 · 1024 · head_outputs` per forward, so 0.302 TFLOP
at 1536 and 0.152 at 772; `×3` for backward gives the 0.45 TFLOP/step. The time saving assumes
the profiler's 35.6 ms attribution to that GEMM and that its cost is linear in output width,
which it is. [INFERENCE] One inconsistency I am flagging rather than hiding: that GEMM is 5.8%
of forward FLOPs but 21% of step time, which is not compatible with it running at 94.3% of GEMM
peak. Either the attribution or the peak figure is off. If the FLOP share is the truer guide
the saving is only ~5.5 ms, which is why the band runs to 158.

### 10.5 Pre-registered predictions, against a step-matched control

- **Held-out unweighted NLL: −0.02 nats [−0.04, −0.01].** Deliberately small, and this
  corrects a claim I nearly made. The 192 log-scales are free and trained today, so they have
  ALREADY learned the correct cumulative `σ` including the reversal - the frozen curve removes
  no BIAS. What it removes is VARIANCE: the free per-horizon scale is estimated from the
  ~504-independent long-window evidence, the frozen one from the full sample. **Predict the
  train-minus-held-out NLL gap narrows by more than the level improves.** That, not the level,
  is the signature of the arm working.
- **Per-horizon market-neutral MSE ratio:** `h = 1` unchanged ±.002; `h = 8` unchanged ±.003;
  `h = 32` and `h = 64` improve .005-.015; **`h = 192`: 1.02125 → 0.997 ± 0.005.**
  **This is a PLUMBING CHECK, not evidence, and must never be quoted as a result.** It is
  nearly tautological: `s_h` IS the oracle per-horizon rescaling by construction and the
  DEMONSTRATED oracle is .99695, so hitting 0.997 confirms the frozen vector is wired to the
  right place and confirms nothing about information. If it MISSES, the plumbing is wrong.
  That is the whole content of the number.
- **Cross-sectional IC:** at step 2000, `h = 32` and `h = 64` at least the control's. The
  claim that matters is at 4000, where the control COLLAPSES (`h=32` .1137 → .0086,
  `h=64` .0879 → −.0125): **predict the arm holds `h=32` ≥ .07 and `h=64` ≥ .06 at step 4000**,
  because the collapse is an effective-sample failure and this arm multiplies the long-horizon
  effective sample.

**What REFUTES the propagation thesis.** The signature is specific and it is not "the arm is
worse": it is **`IC(64)` falling more than 1.5 iid SE (.0042) below the step-matched control
WHILE `IC(1)` rises.** Short improves, long degrades - that is the long end carrying multi-bar
structure which direct cumulative supervision was capturing and accumulation destroys. A
secondary form of the same refutation: `h = 192` MSE ratio hits its predicted 0.997 (the
amplitude fix lands) while `IC(64)` drops - amplitude gained, information lost.

### 10.6 The gate on `ρ₁`, before the number exists

MA(1) attributes the whole variance-ratio deficit to lag 1, so the share of the deficit that
lag 1 actually explains is `2·ρ₁ / (Q - 1)`, which at the implied `ρ₁ = -0.325` is 1.

| measured `ρ₁` | lag-1 share of the deficit | recommendation |
|---|---|---|
| `≤ -0.26` | ≥ 80% | run the arm as specified |
| `-0.26` to `-0.163` | 50-80% | run it, but the ceiling is provisional and `IC(64)` is the only evidence that counts |
| `> -0.163` | < 50% | **do not run it.** Reversal is multi-lag, a slow reverting PREDICTABLE component is the better explanation, that is genuine multi-bar structure, and the lag-resolved `Γ(j,k)` decomposition comes first |

`ρ₁ = -0.163` is the flip point and it is one number from job 5471.

---

## 11. `c₁` is NOT measurable, and here is the reachable-frontier estimate instead

The heading is the claim, and it is deliberately not "bounding `c₁`". Under (A1') the whole
192-point ceiling is the single scalar `c₁`, so it is tempting to go and measure it. **It
cannot be done, and the impossibility is structural rather than a gap in effort.**

**An upper bound on predictability is a statement about an INFORMATION SET, not about a
process.** A martingale and a perfectly predictable process with identical second moments are
indistinguishable from the targets, so no functional of the targets separates them. Every
lag-`k` cross-sectional autocovariance yields a LOWER bound - `Var(m) ≥ Γ_k²/Var(r)` by
Cauchy-Schwarz - and never an upper one. This is the same wall the ceiling hit, one level
down, and any construction that claims to clear it is smuggling in an assumption; the only
honest move is to name which. **`0.1666` was never a measurement of `c₁`. It is an input.**

That distinction has to survive being quoted. "The ceiling is `.1666`" and "the reachable
frontier of THIS architecture on THIS information set estimates to `IC_∞`" are materially
different claims, and only the second is a measurement.

**What IS estimable, cheaply, is the quantity that actually decides spending:** the asymptotic
`h = 1` IC of our own model class on our own information set. That is a learning-curve
extrapolation, and the machinery already exists.

- Take `LatentProbe`'s frozen-trunk closed-form ridge probe, through its corrected
  `Partitions::split` rather than a second draw. One forward pass over the training origins
  produces the trunk latent; the probe then solves `h = 1` in closed form.
- Solve it on NESTED training subsets - 1/64, 1/32, ..., 1 - drawn by whole timestamp, scoring
  each on `held-out full`. Nested and not independent draws, so the curve is monotone by
  construction and its noise is common-mode.
- Fit `IC(N) = IC_∞ − a·N^{−b}` and report `IC_∞` with its interval.

`IC_∞` is a LOWER bound on `c₁` - it is what this trunk and this feature set can reach with
unlimited data, not what is knowable - but it is the number that answers "is there anything
left to buy at `h = 1`", and under the flat ceiling that question sets the target at every
horizon at once. If `IC_∞` comes back near the measured `.0833`, the one-bar relationship is
exhausted for this architecture and the `+0.1011` gap at `h = 64` is unreachable without a new
information source; if it comes back near `.15`, the gap is real and reachable and the
propagation arm is the way to spend on it.

**Cost: one job, no training, ~200-300 s.** One frozen-trunk forward pass over a training draw
plus seven extra ridge solves on subsets of the SAME cached latents - the solves are
`d_model`-sized normal equations and cost seconds. Zero new parameters, zero training steps,
and it reuses a subcommand that already exists. It is the cheapest high-leverage measurement
left on the board.

---

# PART II - THE INCREMENT ARM (session 2)

## 12. What is LANDED and green

`./torch-env.sh cargo check -p trading_bot_0 --tests` = **0 errors** with all of the below in the
tree. Nothing here is half-landed; there are no stubs, no dead flags and no fallback paths.

| file | change | state |
|---|---|---|
| `teacher.rs` | `whole_timestamp_ladder(stamps, levels) -> Vec<Vec<bool>>`, strictly nested, coarsest first | landed, tested, **11/11** |
| `model.rs` | `HorizonMean::Increment { scales }` + parse/display/stamp/serde/validate | landed, compiles |
| `model.rs` | `HorizonMean::scale_basis` - `[scales, pred_len]` cosine family in `ln h` | landed |
| `model.rs` | `scale_expansion` and `increment_geometry` buffers | landed |
| `model.rs` | `head()` increment assembly - means passthrough + `Φ` expansion in activation space | landed |
| `model.rs` | `losses()` increment branch + `increment_pair` + `reduce_geometry(.., half_log)` | landed |
| `model.rs` | `output()` derived cumulative log-scale, `decode()` cumulative reassembly | landed |

**Nothing was reverted.** The one rule I broke twice is `implementation before call site`; both
occurrences were mine, both were caught by peers within minutes, and the second is why
`increment_pair` is an associated function landed in the same edit as its caller.

## 13. The arm, in one paragraph, and why its justification cannot be invalidated

`--horizon-mean increment:K` emits `CHANNELS·pred_len` per-BAR increment means plus `CHANNELS·K`
log-scale coefficients over a cosine basis in `ln h`, and computes the NLL on per-bar
increments. **It is not a rank restriction** - the difference operator is square and invertible,
every one of the 192 emitted horizons keeps every degree of freedom it has today, and this is
the precise respect in which it is not `basis:8:8`. What moves is the SPACE THE LOSS IS COMPUTED
IN, and the argument for moving it is **combinatorial, not statistical**: the `h = 192`
cumulative target is a 192-bar overlapping sum, so 2.46 M rows carry ~504 independent draws,
while the `j = 192` increment target is a single non-overlapping calendar bar and carries all of
them. That is a fact about sampling geometry. **It holds for any `ρ₁`, any `VR'`, any regime and
any IC**, so it survived every correction issued this session - the 1.70x amplitude retraction,
the withdrawal of the monotone calendar drift, and the 64% regime swing in `V_h/D_h`. It is the
only claim in this batch with no upstream dependency on a measured number.

`K = pred_len` recovers today's per-horizon dispersion freedom exactly (the family is a DCT-II
grid on `ln h`, hence full rank at 192), which is what makes the mean-side and scale-side
changes **separable knobs** rather than one bundled arm. Default **K = 8**.

## 14. Costs

`head_outputs = CHANNELS·(pred_len + K)`; output projection is `[1024] × [head_outputs]` over
96,000 tokens.

| K | head_outputs | output-proj params | TFLOP/step | out-proj ms | step ms |
|---|---|---|---|---|---|
| 192 (today) | 1536 | 1,574,400 | 16.98 | 35.6 | 168 |
| **8 (default)** | **800** | **820,000** | **16.55** | **18.5** | **~150.9** |
| 1 | 772 | 791,300 | 16.53 | 17.9 | ~150.3 |

GB/step ~143 -> ~142.1. Peak allocator 18,573 -> ~18,280 MiB of ~25,900 usable; the mode
materializes the same dense `[2·CHANNELS, pred_len]` block, so no new activation exists. The `Φ`
expansion is 1.2 GFLOP at K=8. The ms column is scaled linearly from the measured 35.6 ms and
carries the inconsistency already on record - that GEMM is 5.8% of forward FLOPs but 21% of step
time - so the step figures keep a **[147, 158]** band.

**K=1 versus K=8 is 28,700 parameters, 0.017 TFLOP/step and 0.6 ms.** K=1 was a default, not a
choice: it buys 0.36% of step time with the entire state-dependent horizon structure of the
uncertainty channel, on a channel with no observed defect (coverage nominal at 0.6827/0.9500).

## 15. Pre-registered predictions, with retractions and current status

### 15.1 RETRACTED

- **"`IC(64)` falling more than 1.5 iid SE (.0042) below the step-matched control".** The
  cross-section draw yields 4,000 windows over 100 timestamps at **IC SE ~.0164**, so .0042 is
  **0.26 SE**. Not a weak test - not a test. The .0028 iid figure assumed 433 k independent
  origins. Retracted in full.
- **The 5.68% aggregation-drift arithmetic**, which rested on the monotone `.586 / .548 / .525`
  ordering that has since been withdrawn. The CONCLUSION it supported does not move, for reasons
  in §16.

### 15.2 DEMOTED to secondary

- "The arm holds `h=32 ≥ .07` and `h=64 ≥ .06` at step 4000" against a control that collapses to
  .0086 / -.0125. Unpaired difference SE is `.0164·√2 = .0232`, so these are **2.6 and 3.1 SE**.
  Real, weak, and will not carry a verdict.

### 15.3 The replacement rule, decidable by construction

**Arm-versus-control IC is a PAIRED per-timestamp difference on identical origins**, never two
independently drawn levels: the cross-sectional noise producing the .0164 is overwhelmingly
common to two models sharing architecture, data and origins, so it cancels. The machinery
exists - `timexer_segment_information_ceiling` already reports `gap` beside `gap_error`, and
`CeilingAccumulator::standard_error` computes the paired SE.

**Criterion: `|paired ΔIC| > 3 × measured paired SE` at `h ∈ {32, 64}`, sign as pre-registered.**
The paired SE's magnitude is deliberately NOT pre-registered - it is a property of how correlated
the two arms turn out to be - and that is the point: the rule is decidable whatever it returns,
it assumes no independence, and the `band` fix buys it precision rather than validity.

### 15.4 Live predictions, by decidability

| criterion | through the cross-section draw? | status |
|---|---|---|
| coverage at h=1/64/192: **±.010 at 68.27%, ±.005 at 95%** on `held-out sample` | NO - 433 k origins, SE ~.0007 | **PRIMARY, decidable before `band`** |
| tercile spread in `VR'_192`, threshold **.05 absolute** | NO - 40,837 wide cross-sections | **decidable before `band`; already corroborated in one window** |
| paired `ΔIC` > 3 measured paired SE at h ∈ {32,64} | yes, but paired | decidable; precision improves with `band` |
| cumulative NLL **-0.02 nats [-0.04,-0.01]** vs **1.987704** | no draw; long-horizon part rests on ~504 draws | reported as a paired per-timestamp difference with its own SE |
| absolute IC levels at step 4000 | yes, unpaired | secondary, 2.6-3.1 SE |

The coverage tolerance is tighter than the ±0.02 originally proposed because the natural unit of
a tail band is the **exceedance rate**: 0.95 -> 0.93 is 5% -> 7%, a **40% relative error in the
tail rate**, which is what a position-sizing consumer reads. ±.005 is a 10% relative tail error
and ~7 SE.

**NLL comparability.** The reported scalar stays in CUMULATIVE space against **1.987704** and
**2.3947663**, so checkpoint selection keeps its meaning across the increment boundary - a strict
advantage over a basis arm, which gains ~2.1 nats/bar for free from the prior alone and whose NLL
must never be compared to either anchor. The increment-space training NLL is emitted too, on its
own axis with its own anchor, because within-arm optimization diagnosis must read the quantity
actually minimized. **If the increment NLL falls while the cumulative NLL does not, that is this
arm's most informative failure mode**, and it is visible only because both are charted.

## 16. The aggregation factor: why a frozen `A_h` is dead, and what replaces it

`σ_h² = (Σ_{j≤h} s_j²) · A_h`. The landed `output()` evaluates it at **`A_h ≡ 1`**, the SERIAL
INDEPENDENCE assumption, stated in the code rather than hidden. That assumption is MEASURED
FALSE - `V_h/D_h` ≈ .586 on training, ≈ .652 on a recent 5-session window, ≈ .398 at h=64 on
`held-out full` - so **this mode makes no cumulative calibration claim until an aggregation is
supplied**, and its coverage will read too wide by a known, signed factor.

**Option 1 (freeze `A_h`) is dead on ALGEBRA, not on any measured number.** The aggregation
factor multiplies from OUTSIDE the state-dependent per-bar coefficients, so **K absorbs exactly
zero of it at any K**. The withdrawn 5.68% only ever set the size of the miss; the 64% swing
makes the coverage table below a floor rather than an estimate.

A 5.68% unabsorbed scale error alone gives:

| band | nominal | at f = 1.0568 | miss | tolerance |
|---|---|---|---|---|
| 95% | .9500 | .9363 | **-.0137** | .005 |
| 68.27% | .6827 | .6559 | **-.0268** | .010 |

**`OrthoTargets`' determinant asymmetry is the decisive argument and depends on no measurement:**
`det M = 1` for a unit-triangular banded whitener, so a mis-fitted SECOND-moment metric costs at
most the nats it was meant to recover and can never make the objective improper, whereas a frozen
aggregation multiplies the predicted SCALE - first-moment-like in its failure mode - and biases
every interval by an amount nobody bounded.

**Option 2, approved and specified: `a(state)` with dyadic supervision.** Give `a` its own
coefficients over the same cosine-in-`ln h` basis and supervise them with a cumulative-space NLL
term restricted to the **CLOSE channel** and to a **DYADIC horizon set `k = 1,2,4,...,192`** -
8 pathways, one channel, `[rows, origins, 1, 8]` = 768 k elements against 147 M for the full
cumulative block, **~0.1% of the step**. Means DETACHED in that term, so the mean half stays
attributable and the increment objective is untouched. Eight pathways rather than 192
near-duplicates is the one idea worth keeping from the NextLat brief.

**Freezing direction, per the min-|ρ| rule.** Measure `A_h` on at least two DISJOINT spans and
freeze the value safe under either, never their mean. For a first-moment-like multiplier the safe
direction is the one that **UNDER-corrects** - leaves residual over-amplification - because an
over-correction INVERTS the sign of the amplitude error, which is strictly worse than leaving 5%
of it. So freeze the LARGEST admissible `A_h` (weakest reversal correction) and let `a(state)`
add strength. Conservative initialization costs almost nothing; aggressive costs calibration.

**Option 3 is the pre-declared retreat, not an improvisation:** if calibration still misses after
option 2, keep today's 192 free cumulative log-scales and take ONLY the increment means. Zero
uncertainty-channel risk, zero parameter saving.

## 17. The `ρ₁` gate, labelled

Job 5471 measured plateau .58613 on training, implying `ρ₁ = -.2069` against a DIRECTLY MEASURED
lag-1 of **-.20669** at h=64 - agreement to four decimals between a parameter inferred from the
plateau level and the same quantity measured from the autocovariance. Lag-1's share of the
deficit is **1.007**, above the 80% threshold, so the gate reads **RUN AS SPECIFIED** -
**conditional on the training regime**. If `V_h/D_h` swings 64% across regimes then the lag-1
behind it is not constant either. The gate never governed the mean half; it governed only whether
the frozen scale was worth building.

My earlier "small transient microstructure" magnitude claim was refuted against `ρ₁ = -0.325` and
is substantially rehabilitated at **-0.207**, which is a plausible bounce / liquidity-provision
coefficient in 5-minute market-neutral residuals in a way -0.325 was not.

## 18. Job 5453 is RETRACTED, and the cause is not the one first proposed

`held-out full` was scored WHOLE in both 5453 and 5471 - `whole_timestamp_draw` subsamples only
above target and 433,303 never exceeded it - so **the draw policy cannot explain the change and
`whole_timestamp_keep` gets no credit for the catch.**

The cause is two different fields on two different populations:
- `teacher.rs:357-360` `martingale_ratio = mean_diagonal / mean_variance` on the **PAIRED**
  population, `n ≥ CROSS_SECTION_MIN = 40`. 5453 printed `1. / martingale_ratio`.
- `teacher.rs:362-384` `variance_ratio = wide_variance / wide_diagonal` on the **WIDE**
  population, `n ≥ CEILING_MIN_WIDTH = 2`. 5471 printed this.

Occurrence 1 (`corpus.rs:1092-1099`) reduces the paired population to **100 of 40,837 timestamps
= 0.24%**, and not a random 0.24% but the `i=0` stride - one systematically selected set of
calendar moments. The tail non-monotonicity (.400 -> .325 -> .347) was the artifact announcing
itself: a real variance-ratio curve cannot rise at the long end. Corrected public numbers:
amplitude share **1.3805x**, log-scale share **14.5%**, residual **6.664x (85.5%)** in the
decoder's learned scale.

## 19. Non-stationarity in two moments

The second moment drifts and the first moment INVERTS. `V_h/D_h` reads .586 / .548 / .525 across
training / calibration / held-out full and then **.652 on the most recent post-training window**,
breaking the ordering outright - so it is regime dependence, not calendar drift. Meanwhile the
external sibling's `Cov/Var` at h=32 is **+.0265 ± .0106** on the full span and **-.1097** with a
t(4) interval of **[-.183, -.037]** on a recent 5-session window: opposite signs, both excluding
zero. (h=64 on that window is -.023 ± .041, uninformative; a 5-session window cannot carry a
horizon-selectivity claim.)

**A market whose reversal strength swings 64% between regimes while its cross-sectional
predictability changes sign is not a stationary process with a mis-specified scale.** That bears
on whether ANY frozen constant is the right object here - the aggregation curve included - and it
raises the bar on `a(state)`: if the first moment inverts across a year, a single strength scalar
fitted over the training span may be the wrong SHAPE of correction, not merely the wrong value.

`LatentProbe`'s third explanation is live for my frozen scale and is now the LEADING one: a
fit-then-score design confounds an information ceiling, a fit-sample ceiling and a regime change
between the blocks, and after a 64% swing the honest prior is the third.

## 20. Commands

Nothing of mine is queued. Both lines below train nothing and belong on the v17 snapshot binary,
**not** `run-release-cuda.sh` (which rebuilds from the live tree).

### 20.1 Tercile split of `VR'` - the decisive measurement, ~270 s

Requires the tercile split to be added to `ceiling_pass` first (~20 lines: split timestamps by
their own cross-sectional one-bar dispersion, accumulate three `CeilingAccumulator`s instead of
one). **Not yet implemented** - this is the one piece of specified-but-unbuilt work I am handing
over, and it is small.

```
mlq submit --name timexer-tercile-4k --max-parallel-runs 1 --time-limit 10m \
  --cwd "$PWD" -- /var/tmp/tb0_v17 ceiling-timexer-segment \
  --checkpoint training/runs/timexer-control-4k/weights/best \
  --output training/runs/timexer-control-4k/gens/5 \
  --batch-size 256 --one-bar-ceiling-ic 0.1666
```

**Pre-registered read: the high and low volatility terciles differ by MORE than 0.05 absolute in
`VR'_192`.** Already corroborated in one window - a specific 5-day regime measured .652 against
held-out .398 at h=64 - so state it as corroborated, not open. If it fires, `V_h/V_1` is
state-dependent in volatility AND in epoch and **is not a constant of this market**, which kills
the frozen aggregation outright and leaves the increment mean half completely untouched.

### 20.2 The increment arm itself - do NOT run before `band`

Its two decidable metrics (coverage, tercile spread) do not need `band`; its IC comparison does.
Add `--horizon-mean increment:8` to the standard 2,500-step training line.

## 21. Dependency list on other agents' numbers

| number | owner | how I depend on it |
|---|---|---|
| increment evidence: ~504 draws vs 2.46 M bars | none - combinatorial | **load-bearing, cannot be invalidated** |
| coverage falsifier ±.010/±.005 | mine, from exceedance-rate reasoning | load-bearing, no dependency |
| cost table | model's own `step_cost` + measured 35.6 ms | load-bearing, no dependency |
| IC draw SE ~.0164 | `TemporalSplit` occurrence-1 census, 3 independent arrivals | load-bearing for §15, and the reason those thresholds were retracted |
| `ρ₁ = -.207`, plateau .586 | Main / job 5471, TRAINING span | governs only the scale half, labelled conditional-on-regime |
| `V_h/D_h` = .652 recent vs .398 held-out | external sibling | kills the frozen `A_h`; does not touch the mean half |
| `det M = 1` asymmetry, min-|ρ| freezing rule | `OrthoTargets` | cited as decisive for option 2; depends on no measurement |
| 3-tap `M = L_Γ⁻¹A⁻¹`, `Γ(j,k)` from job 5478 | `OrthoTargets` | the prewhitener sits on the INCREMENT residual if this arm lands; taps come from 5478 |
| 1.3805x amplitude share | mine (§18), corrected | narrative only, in no threshold |

## 22. Handover - what must not be re-derived

1. **BEFORE ANYTHING ELSE: thread `--placement strided|anchored|both` into `ceiling_pass`.**
   Job 5483 moved the student's `held-out full` IC by 2.07-2.72x on placement alone while the
   ceiling stayed on the strided draw, so the ceiling, the gap and every threshold derived from
   them are currently measured against a DIFFERENT POPULATION than the student is. Target-only,
   no model, no lease worth naming. Until it lands **the ceiling is the blocker** and no gap
   arithmetic in this file may be quoted as a decision.
2. The increment argument is combinatorial and survived every correction today. Build on it.
3. Compare arm-versus-control IC as a **paired per-timestamp difference with its measured SE**.
   Never two independently drawn levels. `band` buys precision on that rule, not validity for it.
4. A frozen SECOND-moment metric with unit determinant is self-limiting; a frozen FIRST-moment
   multiplier is not. Freeze the **minimum |ρ| across disjoint spans, never the mean**.
5. `a(state)` cannot be supervised by an increment-space loss. The dyadic close-channel term in
   §16 is how it gets a gradient, and it is ~0.1% of the step.
6. The tercile split (§20.1) is ~20 lines and is the highest-value unbuilt measurement I leave -
   but it must be re-specified on the ANCHORED draw. A volatility-tercile statistic over
   10.6-name cross-sections would reproduce the same attenuation defect one level down, which
   would be the third independent instance of that class.

## 23. Verification actually performed, and what was NOT

- `./torch-env.sh cargo check -p trading_bot_0 --tests` - **0 errors** with everything in §12
  landed.
- `cargo test -p trading_bot_0 --lib timexer_segment::teacher` - **11 passed, 0 failed**,
  including the new `the_ladder_rungs_are_nested_coarsest_first_and_each_keeps_whole_cross_sections`.
- `cargo test -p trading_bot_0 --lib timexer_segment::model` - **ran to CUDA contention and did
  not finish**, twice, at 900 s and 1500 s; the card was shared with siblings and queued jobs.
  The CPU-side tests that DID complete are all `ok`, and they include the two the
  `reduce_geometry(.., half_log)` signature change could plausibly have broken:
  `cumulative_target_basis_is_a_bit_exact_identity_on_the_objective` and
  `free_horizon_mean_is_the_dense_head_and_the_fold_copies_one_hot_rows_exactly`. **The
  CUDA-resident model tests are UNVERIFIED against this landing** and are the first thing to run
  next session, before any new work.
- **No test exists yet for the increment path itself.** The three that must be written, and they
  are cheap and CPU-only:
  1. `increment_pair` inverts `decode`'s cumulative reassembly exactly - round-trip on a random
     `[rows, origins, 4, pred_len]` target, max abs error 0.
  2. `HorizonMean::Increment { scales: pred_len }` with an identity-equivalent basis reproduces
     the free head's emitted 192-horizon forecast, proving `K = pred_len` really does recover
     today's freedom rather than approximating it.
  3. The increment mask loses exactly one bar per gap and no more.
