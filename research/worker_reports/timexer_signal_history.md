# Per-horizon signal history: step-indexed IC, amplitude and coverage

Scope: `trading_bots/src/torch/timexer_segment/reports.rs`, `shared/src/report.rs`,
`tui/src/main.rs`, and the evaluation/report call sites plus metric reductions in
`runner.rs`. No model change, no objective change, no new evaluation draw, no change to the
`held-out sample` definition, no GPU work.

## The gap this closes

DEMONSTRATED (by `HorizonRot`, quoted in the assignment): control, held-out full, step 3000
carries cross-sectional IC .065486 / .061046 / .050928 at h = 64 / 128 / 192 against an iid
SE of ~.00279 - 18-23 sigma of real long-horizon information - while the per-horizon
market-neutral MSE ratio at the same horizons degrades past 1.0, because the MSE-optimal gain
on the model's own centered forecast at h = 192 is .26594, i.e. the conditional mean is
~3.8x over-amplified. MSE is quadratic in forecast amplitude; rank order, and therefore
cross-sectional tradability, is invariant to it.

DEMONSTRATED (this task, from the source): the IC was written only into
`timexer_segment_signal`, which is horizon-indexed and rewritten in place at every
evaluation. Nothing on disk retained a second value of it. `_decomposition`, `_offset`,
`_tradable`, `_tradable_rates` and the `_utility_*` family have the same shape.
`timexer_segment_calibration` is aggregate-by-step with no per-horizon row.
`timexer_segment_horizon_steps` retained history for exactly four horizons and only for the
four-channel MSE ratio and the cross term.

Worse, and not previously stated: the within-timestamp IC is undefined on the draw the
interval actually scores. `held-out sample` strides one origin per timestamp, so no timestamp
reaches `CROSS_SECTION_MIN` names; `held-out full` is scored once per epoch. The only
per-interval carrier of a measurable IC is the `held-out cross-section` pass, whose curves
never reached `Metrics` at all. So even a step-indexed transpose of the old point struct
could not have produced an IC trajectory. `Metrics` now carries that pass.

## Bases

### Added (8, all step-indexed, x = optimizer step)

| base | question | unit / reading rule | series |
| --- | --- | --- | --- |
| `timexer_segment_horizon_steps_signal` | does the per-horizon cross-sectional IC rise or fall over training? | correlation coefficient; 0 = no information | IC per horizon per split + `zero information 0.0` |
| `timexer_segment_horizon_steps_signal_error` | how precisely is each horizon's IC measured? | correlation coefficient, iid-timestamp SE, NOT a confidence interval; an IC beyond ±2 of these is not noise | SE per horizon per split |
| `timexer_segment_horizon_steps_pooled` | does the pooled correlation move, on the draws the IC is undefined on? | correlation coefficient over all (window, horizon) bars pooled, so NOT a cross-sectional statistic | pooled Pearson per horizon per split + zero |
| `timexer_segment_horizon_steps_population` | how much data is behind each horizon's statistics, per step? | count; each series names its own population; NaN = split not scored, 0 = scored and nothing qualified | contributing cross-sections and valid target bar-channels, per horizon per split (Symlog) |
| `timexer_segment_horizon_steps_decomposition` | is the close MSE gain a constant tilt, a conditional signal, or a mis-scaled amplitude? | share of the CLOSE-channel persistence MSE; offset + demeaned + cross term = the close total gain, NOT the four-channel headline | the three components per horizon per split + zero |
| `timexer_segment_horizon_steps_gain` | is the conditional mean's amplitude right, per horizon, over training? | gain on the demeaned forecast; 1 = perfect amplitude calibration, < 1 = over-amplified, > 1 = under-amplified | `β̂` per horizon per split + `perfect amplitude calibration 1.0` |
| `timexer_segment_horizon_steps_best_scale` | is a close MSE ratio above 1 absent signal, or signal at the wrong amplitude? | ratio vs close-anchored persistence; both near 1 = no signal, achieved above 1 with best scale below it = signal at the wrong amplitude | achieved close ratio and best-scale close ratio per horizon per split + `parity 1.0` |
| `timexer_segment_horizon_steps_calibration` | do the predicted σ bands cover the realized targets at each horizon, over training? | fraction of valid target bars | within 1σ and within 1.96σ per horizon per split + nominal .6827 / .9500 |

Splits are `held-out sample`, `held-out cross-section`, `held-out full`. A series with no
finite value at any step is dropped by the `chart` helper rather than drawn, so the
structurally undefined sample IC never appears as a flat line a reader could mistake for
"measured, and zero"; the population base is what explains the absence.

### Changed

- `timexer_segment_horizon_steps`: 4 -> 7 horizons; gains the `held-out cross-section` split;
  series relabelled `<split> <space> MSE ratio at h = N` -> `<split> <space> four-channel MSE
  ratio at horizon N`. "four-channel" because the decomposition family beside it is
  close-channel only and the two do not reconstruct each other. `at horizon N` because
  `report_cli --var` selects a series by splitting a rendered token on its first `=`, so a
  label carrying its own `=` is unselectable on the command line.
- `TradingCurve::cross_sectional_ic_se` now reads NaN below two contributing cross-sections
  (was 0). A zero-width band around an IC that never fired is the strongest possible claim on
  the weakest possible evidence. Pinned by `zero_forecast_scores_exactly_persistence_at_every_horizon`.

### Removed

- `timexer_segment_horizon_steps_scaling`. Its single series is now one of the three in
  `_horizon_steps_decomposition`, in the same unit, beside the two components the cross term
  only means anything against. `CROSS_UNIT` went with it. The TUI test asserts the retired
  name is no longer scanned, so it cannot come back as a blank panel.

### Registered for peers (writers owned by them, not by this task)

`timexer_segment_startup` (LoaderPerf), `timexer_segment_calibration_gain` and
`timexer_segment_calibration_effect` (AmplitudeCal). Registration is centralized under
Main's Amendment 2; a half-registered base fails the bidirectional test.

### Deliberately unchanged

The horizon-indexed latest-curve views - `timexer_segment_signal`, `_decomposition`,
`_offset`, `_tradable`, `_tradable_rates`, `_horizon*`, the `_utility_*` family and the
aggregate `_calibration` - keep their names, questions and meanings. The step-indexed family
is additive and distinctly named; nothing was silently reinterpreted.

## Horizon set: 1, 8, 16, 32, 64, 128, 192

Chosen as exactly `utility::HORIZONS`, pinned by
`the_decision_horizons_are_the_utility_holding_periods`. Justification:

1. The question that decides adoption is whether the IC at the horizon a policy would trade
   survives training. The endpoint-utility family already reports payoff, break-even,
   turnover and exposure on this grid; on any other grid the two families cannot be read
   against each other.
2. Four horizons cannot show a term structure. With 8 and 64 adjacent, a rotation between
   them was a jump between two points. DEMONSTRATED relevance: `AmplitudeCal` measured `β̂` on
   control-4k gen2 step 3000 held-out full as .600 at h = 1, peaking at .989 at h = 11, then
   falling monotonically to .266 at h = 192. The old set would have rendered .60 -> ~1 -> .27
   with no way to see that the peak is interior; h = 8 and 16 bracket it and 32/64/128 trace
   the decay.
3. Keeping all 192 horizons was rejected on both storage and readability: 192 x 11
   step-indexed quantities x 3 splits x 4 B = 25.3 KB per evaluation and 6,336 series across
   the family, against 924 B and 231 series at seven (analytic; the measured figure with the
   full split absent is 836 B, below). A 6,336-series panel is not a chart.

## Cost

All storage figures MEASURED on a 21-step synthetic history through the real
`.report.bin` writers (postcard), `held-out sample` + `held-out cross-section` finite and
`held-out full` all-NaN, which is the shape of a mid-epoch run:

| | 1 step | 21 steps | marginal per evaluation |
| --- | --- | --- | --- |
| 8 new bases | 15,827 B | 30,195 B | 718 B |
| `_horizon_steps` (widened) | 2,214 B | 4,580 B | 118 B |
| step-indexed family total | 18,041 B | 34,775 B | **836 B** |
| every base `write_metrics` writes | 22,144 B | 41,520 B | 969 B |

- Added bytes per evaluation: **836 B** for the step-indexed per-horizon family. One
  evaluation per report interval, so **836 B per 1000-step interval** at `--eval-every 1000`.
  Files are rewritten in place, not appended: a 20,000-step arm holds ~35 KB of step-indexed
  per-horizon history in total, and `write_metrics` as a whole writes 41.5 KB at 21 steps.
- Once an epoch end makes the `held-out full` series finite the marginal rises to ~1.25 KB per
  evaluation [INFERENCE: series count scales from two finite splits to three].
- Fixed cost, paid once per rewrite rather than per step: ~17.2 KB of series labels for the
  family. The per-base figures above were measured before the `at h = N` -> `at horizon N`
  rename, which added 4 B per series label (~0.8 KB of fixed cost across `write_metrics`,
  visible in the last row) and nothing per step.

Evaluation CPU, MEASURED (`--release`, 21-step history, 5 rounds, median of 5):
`write_metrics` 56.7 ms with the step family against 25.0 ms with the family suppressed, so
the family costs **31.7 ms** - 9 files at ~3.5 ms each, fsync-bound, not serialization-bound.
The pre-change family was 2 of those 9 files, so the ADDED host cost is **≈ +25 ms per
evaluation**, and it is spent outside the timed `score` path (it lands in the existing
`after held-out evaluation` accounting). `horizon_track` costs 3.3 µs per call, twice per
evaluation.

Evaluation GPU: **no second reduction pass over held-out forecasts.** `β̂ = Cov(f,y)/Var(f)`
and the best-scale ratio come from moments `Scorer::trading` already reduces (`col(f)`,
`col(y)`, `col(f²)`, `col(y²)`, `col(f·y)`); the IC's population count is `col(&usable)`,
also already computed. The only added device work is two f64 column reductions per batch over
the `within_1`/`within_2` masked tensors, which were already materialized for the aggregate
coverage, plus `horizon_sums` growing 11x192 -> 13x192 f64 (16.9 -> 20.0 KiB device).
[INFERENCE] ~13 µs per 2,048-origin evaluation (8 batches x 2 reductions x 786 KB read at
~1 TB/s); not measured, because this task holds no GPU lease.

## Reading the new series

```
./target/release/report_cli <gen> timexer_segment_horizon_steps_signal --run <RUN> \
  --var "held-out cross-section close cross-sectional IC at horizon 64"
```

Drop `--var` for the whole term structure per step; swap the base for
`timexer_segment_horizon_steps_gain` (`... close MSE-optimal forecast gain at horizon 192`),
`_best_scale`, `_calibration` or `_population`.

**Existing runs cannot show any of these series.** The data was never written: the per-horizon
IC, `β̂`, best-scale ratio and per-horizon coverage existed only in the horizon-indexed files
that each evaluation overwrote, and the `held-out cross-section` curves never entered the
report point at all. `timexer_segment_horizon_steps` on an existing run still reads, but only
as h = 1/8/64/192 four-channel ratios under the old labels. No backfill is possible and none
was fabricated; the command above returns a miss (listing the bases the generation directory
does hold) until a new arm writes them.

## Tests

- `the_signal_panel_retains_every_evaluation_rather_than_the_latest` - three evaluations with
  a rising IC; asserts all three values survive in `_signal`, `_population`, `_gain`,
  `_best_scale` and `_calibration`. This is the property the whole task exists for.
- `an_unmeasured_cross_section_reads_as_absent_and_never_as_zero` - the sample draw's IC
  series is absent rather than flat at 0; a step with no cross-section pass is a NaN gap; the
  pooled Pearson is finite and on its own base so it cannot be mistaken for a measured IC.
- `the_optimal_gain_reproduces_its_analytic_value_and_undefined_statistics_stay_nan` -
  synthetic fixture with `y_j = β_j·f + e`, `e` orthogonal to `f`, so `β̂_j = β_j` by
  construction; pins `β̂`, `ρ = β·σ_f/σ_y`, the achieved and best-scale close ratios, the cross
  term, the three-part identity, per-horizon coverage against 0.75 / 0.875, NaN IC and SE and
  a 0 population on an 8-name draw. It also reproduces the failure mode: at `β̂ = 0.25` the
  close ratio is 1.123 while the same forecast rescaled would score 0.985. And it pins WHY
  `β̂` is emitted rather than inverted out of the decomposition: the cross term fixes only
  `|β̂ - 1|`, so the inversion admits two positive branches (0.25 and 1.75 at the first
  horizon) and nothing in the decomposition chooses.
- `the_horizon_track_slices_one_based_horizons_and_skips_the_unevaluated_ones` - extended to
  every new field, each with a distinguishable per-index fixture ramp.
- `the_decision_horizons_are_the_utility_holding_periods` - the two families' grids cannot
  drift apart silently.
- `zero_forecast_scores_exactly_persistence_at_every_horizon` - extended: a flat forecast has
  NaN `β̂`, a best-scale ratio of exactly 1, NaN IC SE and a 0 population count.
- `tui`: the bidirectional registry test covers the 8 new names in both directions and
  asserts the retired `_horizon_steps_scaling` is no longer scanned.

## Verification

- `cargo check -p trading_bot_0 --tests`: zero errors.
- `cargo test -p trading_bot_0 timexer_segment`: 105 passed, 0 failed (two failures seen
  mid-run belonged to `calibration.rs`, a sibling's in-flight module; reported to its owner
  and fixed by them).
- `cargo check -p trading-bot-tui --tests` and `cargo test -p trading-bot-tui`: 36 passed.
- `report_cli` exercised against a synthetic run directory written by the real writers, which
  is what established the `--var` label-grammar constraint above.
