# Trading evaluation across a horizon set, with breakeven cost as the decision

## What changed

The trading evaluation used to answer "does the decile long/short pay at `h = 1`". It now
answers "at which holding period, and up to what per-side cost, is this model tradable".

- Holding-period set: `PORTFOLIO_HORIZONS` `[1, 4, 16, 64, 192]` → **`[1, 8, 16, 32, 64]`**, the
  decision set the horizon analyses use. `h = 8` is where the forecast carries the most skill
  (market-neutral MSE ratio 0.9256 against 0.9602 at `h = 1`) and was never in the old set.
- Cost sweep: `COSTS_BPS` `[0, 1, 2, 5, 10]` → **`[0, 0.5, 1, 1.5, 2, 3]`** bps per side, dense
  through the region where the sign flips (the historical `h = 1` net crossed zero between 1
  and 2 bps per side, which the old sweep straddled in one 1 bps step).
- `PortfolioCurve` grew from `{net_bps, net_sharpe}` to
  `{gross_bps, net_bps, gross_rate_bps, net_rate_bps, gross_sharpe, net_sharpe, breakeven_bps}`.
  Gross is a named quantity rather than "the sweep point that happens to charge nothing".

## New bases (registered in `shared/src/report.rs`)

`tui/src/main.rs` needed no edit: `meta_chart_bases()` extends from
`TIMEXER_SEGMENT_REPORT_BASES`, so both directions of the registry contract hold by
construction and the existing bidirectional test
(`the_meta_chart_list_looks_for_every_registered_writer_base` and the scanned-vs-registered
comparison beside it) covers the new names. Verified green: `cargo test -p trading-bot-tui`,
36 passed.

| base | x | y-label (verbatim) |
| --- | --- | --- |
| `timexer_segment_portfolio` (existing, re-unitted) | `bars held` | `basis points per non-overlapping round trip (> 0 = the labeled per-side cost is survived; not comparable across holding periods)` |
| `timexer_segment_portfolio_rate` (**new**) | `bars held` | `basis points per bar held (non-overlapping holds; > 0 = the labeled per-side cost is survived; comparable across holding periods)` |
| `timexer_segment_portfolio_sharpe` (existing) | `bars held` | `annualized Sharpe ratio (dimensionless; 19,656 five-minute bars per year, non-overlapping holds assumed)` |
| `timexer_segment_portfolio_breakeven` (**new**) | `bars held` | `basis points per side (the cost at which the net spread reaches zero; tradable only where this exceeds the cost actually paid, 0 = no gross edge to spend)` |
| `timexer_segment_horizon_loss_weight` (**new**, registered on `HorizonWeight`'s behalf) | `bars ahead` | `loss weight (dimensionless; mean 1 over all 192 horizons; 0 = horizon not trained)` |

The x label moved from `bars ahead` to `bars held` on the four portfolio bases, because a
holding period is not a forecast horizon and the two were sharing one axis label.

### Series

Per split (`held-out sample` / `held-out full`), grammar `<split> <space> <quantity>`:

- `<split> market-neutral decile spread, gross` and `..., net at {0.5,1.0,1.5,2.0,3.0} bps per side`
- `<split> market-neutral decile spread rate, gross` / `..., net at <c> bps per side`
- `<split> market-neutral decile spread Sharpe, gross` / `..., net at <c> bps per side`
- `<split> market-neutral decile spread breakeven per-side cost`
- reference lines: `zero` on the three sweep charts, `1.0 bps per side (a plausible cost floor)`
  on the breakeven chart

Cross-sectional IC with its ±1 standard-error band was already on `timexer_segment_signal` at
every horizon `1..pred_len`, so it covers every holding period reported here; duplicating it
into the portfolio family would have put the same number on two axes.

## Turnover assumption: NON-OVERLAPPING holds

Named in the labels, not just in the code. The book is **flat between round trips**: enter at an
evaluation timestamp, exit `h` bars later, pay the per-side cost four times (two legs × entry
and exit). A year holds `19,656 / h` such round trips, which is the annualization factor, so no
hold is credited with variance relief from overlapping with its neighbours.

Why this one: the alternative — `h` overlapping sleeves rebalanced every bar — reports the
**same rate and the same breakeven** (one bar turns over `1/h` of the book and earns `1/h` of a
round trip) and a **higher Sharpe** (averaging `h` staggered sleeves). Non-overlapping is
therefore the conservative reading of the identical edge, and the only quantity it penalizes is
the one most likely to be over-claimed.

The cost is charged **once per round trip, not per bar**. That is the entire reason a longer
hold can survive a cost a shorter one cannot, and the reason a verdict measured only at `h = 1`
cannot be extrapolated.

## Breakeven per horizon

**Not measurable from any checkpoint on disk.** Nothing was fabricated in its place. Precisely:

1. The four diagnostic runs (`timexer-scalarlr1-20260906`, `timexer-nox0-20260906`,
   `timexer-nox0-scalarlr1-20260906`, `timexer-lr50-scalar1-20260906`) have the trading family
   written at gen 1, but **every cross-sectional statistic in them is NaN and the gross spread
   is identically 0**. Read back:

   ```
   $ ./target/release/report_cli 1 timexer_segment_signal --run timexer-nox0-20260906
   1 ... held-out sample close cross-sectional IC (mean over timestamps)=NaN  ... +1 s.e.=NaN
   $ ./target/release/report_cli 1 timexer_segment_portfolio --run timexer-nox0-20260906
   1  held-out sample decile-spread return net of 0 bps per side=0
   ```

   Cause: only the **held-out sample** split was scored at that report interval, and that
   sample is a strided pick over the held-out origin list, so an evaluation timestamp almost
   never holds the 20 valid tickers `CROSS_SECTION_MIN` requires. The cross-sectional path
   never fired. This is pre-existing and unrelated to this change — it just means these runs
   carry no trading measurement to re-express. **Consequence for the coordinator: the new
   charts will be blank on a run whose only scored split is the strided preview sample. They
   populate on the full held-out split** (`evaluate`, or a report interval that scores
   `held-out full`), which is how the one historical trading measurement was produced.

2. `timexer-market-neutral-20260906/gens/trading` does carry a real full-split measurement, but
   it is a `v5` checkpoint with the horizon-major head permutation, and
   `research/worker_reports/timexer_eval_bisect.md` rules its `timexer_segment_portfolio`
   valid **at the `h = 1` column only**. At `h = 1` it reads, real:

   ```
   $ ./target/release/report_cli 1 timexer_segment_portfolio --run-root <that run>
   1  held-out full decile-spread return net of 0 bps per side=4.8356795
      ... net of 1 bps per side=0.8356796   ... net of 2 bps per side=-3.1643205
   ```

   `breakeven = gross / 4 = 4.8356795 / 4 = ` **1.2089 bps per side**, which reproduces the
   "dead at ~1.2 bps/side" that was previously derived by hand off the sweep. That is a real
   validation of the transform against a real checkpoint, at the one horizon the checkpoint can
   support. `h = 4, 16, 64, 192` in that file read permuted head rows and are not reportable.

3. A fresh GPU eval is not available to me: I am not permitted to run GPU jobs, and it would
   fail a legitimate incompatibility check anyway. In the same session `HorizonWeight` bumped
   `FORMAT` from `…-v8-…` to `…-v9-…` and `OBJECTIVE` from `causal_patch_market_neutral_nll_v4`
   to `_v5`, and renamed the manifest's `best_preview_nll` to `best_objective_nll`, while every
   checkpoint on disk carries the `v8` stamp, the `_v4` objective and the old field name.
   `Manifest::read` refuses those, correctly, and I did not fight it. **The per-horizon
   breakeven table needs one full-split evaluation of a run trained by this build.**

## Read-back verification

> **NOT A MEASUREMENT OF THIS MODEL.** Every number below comes from a SYNTHETIC fixture with
> a hand-set constant IC and no model in the loop. It proves the transform and the plumbing
> and nothing else. Do not quote the breakeven row as our forecaster's tradability; the only
> real trading number this repository holds is the historical full-split `h = 1` breakeven of
> 1.2089 bps per side.

Exercised: the real eval path from `Scorer::accumulate` → `Scorer::finish` → `Scorer::trading`
→ `Scorer::portfolio` → `reports::write_trading` → `.report.bin` → `report_cli`, over a
synthetic final-origin fixture (2,000 windows, `pred_len = 64`, 80 evaluation timestamps of 25
tickers, forecast = target + 19.97 × independent noise so the IC is ≈ 0.05, σ = 20 bps per bar).
Everything except the model forward pass is the production code path. The nine trading bases
were written and read back; the four portfolio ones:

```
$ ./target/release/report_cli 1 timexer_segment_portfolio_breakeven --run <scratch>
1   held-out sample market-neutral decile spread breakeven per-side cost=0.93660676   1.0 bps per side (a plausible cost floor)=1
8   held-out sample market-neutral decile spread breakeven per-side cost=4.977983     1.0 bps per side (a plausible cost floor)=1
16  held-out sample market-neutral decile spread breakeven per-side cost=1.5074383    1.0 bps per side (a plausible cost floor)=1
32  held-out sample market-neutral decile spread breakeven per-side cost=6.717173     1.0 bps per side (a plausible cost floor)=1
64  held-out sample market-neutral decile spread breakeven per-side cost=4.066153     1.0 bps per side (a plausible cost floor)=1

$ ./target/release/report_cli 1 timexer_segment_portfolio --run <scratch>
1   ... decile spread, gross=3.746427   ... net at 0.5 bps per side=1.746427   ... net at 1.0 bps per side=-0.25357288   ... net at 2.0 bps per side=-4.253573
8   ... decile spread, gross=19.911932  ... net at 0.5 bps per side=17.911932  ... net at 1.0 bps per side=15.911932     ... net at 3.0 bps per side=7.9119315

$ ./target/release/report_cli 1 timexer_segment_portfolio_rate --run <scratch>
1   ... decile spread rate, gross=3.746427     ... net at 1.0 bps per side=-0.25357288
8   ... decile spread rate, gross=2.4889915    ... net at 1.0 bps per side=1.9889915
64  ... decile spread rate, gross=0.25413457   ... net at 1.0 bps per side=0.19163457

$ ./target/release/report_cli 1 timexer_segment_portfolio_sharpe --run <scratch>
1   ... decile spread Sharpe, gross=28.554972  ... net at 1.0 bps per side=-1.9327126
64  ... decile spread Sharpe, gross=1.846753   ... net at 1.0 bps per side=1.392576
```

Titles and axis labels as written (from the same files):

```
CausalPatch epoch 0 step 2000 | up to what per-side cost is each holding period tradable at all? | held-out sample 2000 fixed windows, one scored origin each, 80 evaluation timestamps, 25 tickers
bars held
basis points per side (the cost at which the net spread reaches zero; tradable only where this exceeds the cost actually paid, 0 = no gross edge to spend)
```

These numbers are a **fixture**, not a measurement of the model. The fixture holds the IC
constant across horizons, so its rate necessarily decays like `1/√h`; the real question — does
`h = 8`'s extra skill beat `h = 1`'s cheaper-per-move cost — is exactly what the real
evaluation has to answer and cannot be simulated here. The scratch run directory and the
throwaway writer that produced it were deleted after the read-back.

## Tests that bind behaviour

In `runner.rs`, `mod tests`:

- `a_perfect_forecast_earns_a_positive_spread_and_buys_a_positive_breakeven_cost` — forecast =
  realized coordinate, 1,000 windows over 40 timestamps, `pred_len = 8`. Asserts strictly
  positive `gross_bps` and strictly positive `breakeven_bps` at both reported holds (so a
  swapped long/short leg fails), and pins as **bit equalities**
  `net = gross - 4·cost` at every swept level, `rate = round trip / h`, and
  `breakeven = gross / 4`. Also asserts the sweep and the breakeven are the same fact: the net
  spread is positive at exactly the swept levels below the breakeven.
- `a_pure_noise_forecast_has_no_information_and_no_breakeven_headroom` — a noise draw
  independent of the target, scored twice, as `+z` and as `−z`. The two are mirror images, so
  their decile spreads negate and **one arm necessarily loses**; that arm must report
  `breakeven_bps == 0.0` exactly (the floor: no gross edge means untradable at every cost, not
  tradable at a negative one), and the winning arm's headroom must sit below `COSTS_BPS[1]`, the
  cheapest level the sweep charges. Statistical half: at every one of the 8 horizons the
  cross-sectional IC sits inside its own ±2 standard-error band over 40 timestamps. Making the
  claim structural (mirror pair) rather than seed-lucky is the point — a single noise draw's
  apparent edge is a coin flip.
- `trading_statistics_match_scalar_reference` — extended. Its `pred_len` went 6 → 16 so the new
  horizon set actually reports `[1, 8, 16]` rather than collapsing to `[1]`, and the scalar
  reference now also pins `gross_bps`, `gross_rate_bps`, `gross_sharpe`, `breakeven_bps`,
  `net_rate_bps` and every `net_bps`/`net_sharpe` sweep point to `1e-9` relative.
- `zero_forecast_scores_exactly_persistence_at_every_horizon` — its finiteness sweep extended
  over the four new rows.

## Scoped checks

- `cargo check -p trading_bot_0 --tests` — 0 errors
- `cargo check -p trading-bot-tui --tests` — 0 errors
- `cargo test -p trading_bot_0 timexer_segment` — 61 passed, 0 failed
- `cargo test -p trading-bot-tui` — 36 passed, 0 failed

## Files

- `trading_bots/src/torch/timexer_segment/runner.rs` — `PORTFOLIO_HORIZONS`, `COSTS_BPS`,
  `Scorer::portfolio` host algebra, four tests.
- `trading_bots/src/torch/timexer_segment/reports.rs` — `PortfolioCurve` (+ the turnover
  contract as its doc), `RATE_UNIT_BPS`, `BREAKEVEN_UNIT`, `BPS_UNIT` re-worded, `write_trading`
  (per-base `x_label`, gross/net split, four portfolio bases).
- `shared/src/report.rs` — `timexer_segment_portfolio_rate`,
  `timexer_segment_portfolio_breakeven`, `timexer_segment_horizon_loss_weight`.
- `docs/timexer_segment.md` — the four portfolio bases, their units, the turnover assumption,
  and the NaN-on-a-strided-sample caveat.
- `tui/src/main.rs` — untouched, by construction.

---

# Addendum: the `held-out cross-section` draw

The finding above — that the trading family was structurally blank because the held-out sample
is one ticker per timestamp — is now fixed additively, per Main's decision.

## The draw

`runner::cross_section_origins(corpus, origins)` builds a THIRD held-out draw, used by the
trading family and nothing else. `runner::cross_section_blocks` is its pure half, over
`(timestamp, window)` pairs, so the block selection is testable without a corpus.

- Groups every held-out origin by its origin-bar UTC timestamp, keeps only timestamps that
  already hold `CROSS_SECTION_TICKERS = 256` windows, and drops thin ones WHOLE.
- Draws at most `CROSS_SECTION_TIMESTAMPS = 128` of them, and exactly 256 tickers from each,
  both by the same `fixed_origins` stride the held-out sample uses — first-to-last, so
  timestamps span the whole held-out period and tickers span the universe's own index order.
  Fixed across steps and across runs over the same universe: step-matchable like every other
  series.
- Refuses loudly (`anyhow`) when no timestamp clears 256, rather than scoring a family of NaNs.
- 256 tickers is a 12.8x margin over `CROSS_SECTION_MIN = 20` and a 25-name decile per side.

**The held-out sample is untouched, byte for byte.** No change to `fixed_origins`, to
`args.eval_origins`, to the preview draw, or to anything selection reads. `HorizonWeight`
confirmed over `hub` that selection reads `evaluation.objective_nll` from the FIRST pass only;
the new pass writes into its own local (`cross_evaluation` / `cross_result`) and never reaches
`best_*`, `preview_curve`, `full_curve` or `points.push(Metrics{..})`.

## Wired at EVERY report interval, not epoch end

Yes, and it is cheap. Measured facts from `timexer-nox0-20260906`'s own timing base: the
held-out sample pass is 2,048 origins = 8 batches of 256 and costs 943.1 ms at step 4,000 and
1,116.0 ms at step 5,000, i.e. **118-140 ms per batch of 256**. The production cross-section
draw is `min(available timestamps, 128) x 256`; the validation partition holds ~106 usable
timestamps (20,617 bars / `pred_len` 192), so ~27,136 origins = **106 batches**.

| quantity | value |
| --- | --- |
| cross-section pass, per report interval | **12.5 - 14.8 s** |
| report interval at `--eval-every 1000` | 168.59 ms/step x 1000 = **168.6 s** |
| overhead | **7.4 - 8.8 %** |
| extra device memory (`Scorer`'s 5 `[origins, pred_len]` fp32 arrays) | 27,136 x 192 x 4 B x 5 = **104 MiB** against an 18,574 / 25,917 MiB budget |

So breakeven is a training-time trajectory, which is what the h=8-improves / h=64-rots finding
needs. `train()` scores it inside the existing report block; `evaluate()` scores it too, so an
epoch-end verdict lands on the SAME draw as the trajectory it concludes (the full split's own
cross-sections are the whole universe and are not comparable with a fixed 256-ticker block).
The cost is charted, not hidden: `EvalTiming::cross_section_ms` -> the series
`held-out cross-section pass total` on `timexer_segment_timing`. If 8% is ever too much, halve
`CROSS_SECTION_TICKERS` to 128 for ~6-7 s, at the price of √2 more IC standard error.

## Unmeasured is no longer zero

The bug that hid all of this: an unpopulated horizon divided by `periods.clamp_min(1.)` and
returned a clean `0.0` bps, indistinguishable from a real flat spread. Now `cross_sections[h]
== 0` forces `gross_bps` and `deviation_bps` to NaN, which propagates to net, rate, Sharpe and
breakeven, and `trading_series` renders those as gaps.

New base **`timexer_segment_cross_section_census`** (registered in `shared/src/report.rs`;
`tui/src/main.rs` still needs no edit), x = `bars held`, y-label verbatim:

```
count (timestamps that cleared the threshold, and mean valid tickers per evaluation timestamp; a holding period whose tickers sit under 20 clears no timestamp, and its return series then read as gaps, never as 0)
```

Series per split: `<split> contributing evaluation timestamps`, `<split> mean valid tickers per
evaluation timestamp`, plus the reference line `20 tickers (the minimum a timestamp is scored
at)` driven by `runner::CROSS_SECTION_MIN` itself (now `pub(super)`) so the line cannot drift
from the threshold the scorer applies. `tickers_per_timestamp` averages over EVERY timestamp,
not only contributors, so it stays finite exactly where the returns go NaN and says WHY: a
census of 1 against a line at 20 is the whole explanation.

## Vocabulary

`docs/timexer_segment.md` "TUI reports" now lists four splits: `training` / `held-out sample` /
`held-out full` / `held-out cross-section` / `persistence`, with the reason the fourth exists
and the reason the sample was not fixed in place (bit-identical persistence NLL across runs;
selection defined on it). The census base is documented beside the four portfolio bases.

## Read-back verification

> **NOT A MEASUREMENT OF THIS MODEL.** Every number in this section, and in particular the
> per-horizon breakeven row `0.76 / 2.91 / 4.51 / 3.26 / 11.40` bps per side below, comes from
> a SYNTHETIC fixture with a hand-set constant IC and no model in the loop. It proves the
> transform, the draw and the plumbing — nothing about our forecaster's tradability. Do not
> quote it, plot it, or carry it into a decision. The only real trading number this repository
> holds remains the historical full-split `h = 1` gross spread of 4.8356795 bps, i.e. a
> breakeven of **1.2089 bps per side**, and it is `h = 1` only (v5 checkpoint, permuted head at
> every other horizon). The real per-horizon table does not exist until jobs 5206/5207/5208
> report.

Exercised the real path (`Scorer::accumulate` -> `finish` -> `trading` -> `portfolio` ->
`reports::write_trading` -> `.report.bin` -> `report_cli`) with two real draws side by side: a
`held-out sample`-shaped fixture (2,048 windows, one ticker per timestamp) and a
`held-out cross-section`-shaped one (40 timestamps x 256 tickers), `pred_len = 64`.

```
$ ./target/release/report_cli 1 timexer_segment_cross_section_census --run <scratch>
1   held-out sample contributing evaluation timestamps=0
    held-out sample mean valid tickers per evaluation timestamp=1
    held-out cross-section contributing evaluation timestamps=40
    held-out cross-section mean valid tickers per evaluation timestamp=256
    20 tickers (the minimum a timestamp is scored at)=20
... identical at h = 8, 16, 32, 64

$ ./target/release/report_cli 1 timexer_segment_portfolio_breakeven --run <scratch>
1   held-out sample ... breakeven per-side cost=NaN    held-out cross-section ... =0.76155937
8   held-out sample ... breakeven per-side cost=NaN    held-out cross-section ... =2.9106567
16  held-out sample ... breakeven per-side cost=NaN    held-out cross-section ... =4.509182
32  held-out sample ... breakeven per-side cost=NaN    held-out cross-section ... =3.2647722
64  held-out sample ... breakeven per-side cost=NaN    held-out cross-section ... =11.395747
    ^ SYNTHETIC FIXTURE. Not our model. See the warning above this block.

$ ./target/release/report_cli 1 timexer_segment_signal --run <scratch>       (h = 1 row)
    held-out sample close cross-sectional IC (mean over timestamps)=NaN   +1 s.e.=NaN
    held-out cross-section close cross-sectional IC (mean over timestamps)=0.03960312
    held-out cross-section close cross-sectional IC +1 s.e.=0.05061581  -1 s.e.=0.028590433
```

That is the requirement met end to end: the sample reads as gaps with a census that explains
them, the cross-section draw reads real numbers, and the cross-sectional IC finally has a
standard-error band. The measured band over 40 timestamps is ±0.0110, which scales to
**±0.0068 at the ~106 timestamps production will draw** — against a historically measured IC of
0.030, a ~4.4-sigma resolution, which is what the 256-ticker margin was chosen for. Scratch run
and throwaway writer deleted.

## Tests added

- `the_cross_section_draw_takes_whole_timestamps_and_drops_thin_ones` — blocks of 296, 255, 1
  and 256 windows. Asserts exactly two timestamps survive, each contributing exactly 256, that
  the 255- and 1-ticker blocks are dropped WHOLE, that two calls are identical (step-matchable),
  and that an all-thin corpus is refused with the explanatory error.
- `an_unpopulated_cross_section_reads_as_nan_and_never_as_zero_bps` — 10 tickers per timestamp,
  half of `CROSS_SECTION_MIN`. Asserts 0 contributing timestamps and NaN in gross bps, gross
  rate, gross Sharpe, breakeven and all three net rows at every horizon, that the census still
  reports the population that failed (10, under the threshold of 20), and that the same fixture
  at 50 tickers per timestamp reports finite numbers — so the NaN is the population, not a
  broken path.

## Scoped checks (re-run after this addendum)

- `cargo check -p trading_bot_0 --tests` — 0 errors
- `cargo check -p trading-bot-tui --tests` — 0 errors
- `cargo test -p trading_bot_0 timexer_segment` — 65 passed, 0 failed
- `cargo test -p trading-bot-tui` — 36 passed, 0 failed

## Correction to the table above

`HorizonWeight`'s final y-label for `timexer_segment_horizon_loss_weight` is
`loss weight (dimensionless; mean 1 over the full horizon; 0 = horizon not trained)` — the
literal "192" was dropped because `pred_len` is a CLI knob; the title carries the exact count.

---

# Addendum 2: the draw negotiates width, and what alignment actually is

## The stop-ship this caught

The first version of `cross_section_origins` REQUIRED 256 windows per evaluation timestamp and
`ensure!`d a hard error otherwise. It would have aborted three queued arms at startup. The real
full-split scope line says why:

```
CausalPatch epoch 1 step 6000 | how much information does the demeaned close forecast carry?
  | held-out full 433303 scored origins, 40837 evaluation timestamps, 4873 tickers
```

433,303 origins over **40,837 distinct timestamps is 10.6 origins per timestamp on average**, not
4,873. Validation origins are placed at each ticker's own valid-bar ordinal — `corpus.rs:431`,
`origin: start - 1 + i * pred_len`, with `start = boundaries[1].max(common_context)` a per-ticker
ordinal of the shared UTC boundary. Two tickers therefore share a timestamp only when they hold
the SAME NUMBER of valid bars between the split boundary and that origin. One missing bar shifts
every later origin of that ticker by one valid bar. What survives is a fully-aligned liquid core
(names with no gaps) plus a long tail of one- and two-ticker timestamps. That is the only way a
mean of 10.6 coexists with a populated cross-sectional IC.

## The draw now

`cross_section_blocks` asks for width instead of demanding it:

1. Group origins by timestamp; discard timestamps under `CROSS_SECTION_FLOOR = 2 ×
   CROSS_SECTION_MIN = 40`. Error only if NOTHING clears the floor.
2. Sort candidates by member count descending. The `k`-th entry then bounds the width of any
   `k`-timestamp UNIFORM draw, so one pass over `k ≤ 128` enumerates every uniform operating
   point the corpus offers.
3. Keep the `(k, width)` maximizing `k · (width − 1)`, with `width ≤ 256`.
4. Restore timestamp order; stride `fixed_origins` over each block's members.

Deterministic (stable sort, BTreeMap order, no RNG), so the series stays step-matchable and
run-matchable. Cost still capped at 128 × 256 = 32,768 origins.

`k · (width − 1)` is not a heuristic — it is the draw's Fisher information for the
cross-sectional IC, whose per-timestamp sampling variance is `1/(width − 1)`.

## Pooling: equal-per-timestamp, made correct by construction

Main asked whether the cross-sectional statistic weights timestamps equally or by width, and to
pick the one that does not let the thinnest dominate the variance.

The scorer averages **equally over contributing timestamps**. On a ragged draw that is the wrong
choice: a 45-name timestamp carries sampling variance `1/44 = 0.0227` against `1/399 = 0.0025`
for a 400-name one, so one thin timestamp would contribute **9×** the variance of a wide one and
a handful of them would set the standard error.

The fix is at the draw, not the estimator: **one common width**. Then equal weighting IS
inverse-variance weighting, identically, and there is no weighting machinery to get wrong. It
also keeps every timestamp's decile the same size, so the pooled spread is a spread of one thing
rather than an average over decile constructions of 4 names and of 25. Precision-weighting a
ragged draw would have been the alternative; it estimates a width-tilted IC under any real
heterogeneity and needs a second accumulator, and a uniform draw dominates it at equal cost.

## Census now carries the width distribution

`timexer_segment_cross_section_census` gained a third series per split,
`<split> narrowest evaluation timestamp`, beside `<split> mean valid tickers per evaluation
timestamp`. Reading rule, in the y-label: **`mean == narrowest` is a uniform draw; `mean >
narrowest` means the width beside it is an average of unequal cross-sections.** A mean of 200
built from a few 400-name timestamps and a tail of 45-name ones supports a completely different
decile than a flat 200, and only this pair distinguishes them. The draw is uniform by
construction, so the two coincide exactly while every drawn ticker is still valid at the
horizon — and separate as per-horizon validity masking bites at the longer holds, which is
precisely the leak worth seeing.

## The standing verdict was always a liquid-core measurement

Stated plainly, because it reframes rather than weakens the trading result: the historical
cross-sectional IC **0.029958 ± 0.00275** at h=1 was computed over whatever the aligned core
actually is — a few hundred names at best — and **never over a 4,873-name cross-section**. Same
for the decile spread of 4.8356795 bps and its 1.2089 bps/side breakeven: those deciles were
built inside the core. This is not bad news. The core is the set of names with no missing
five-minute bars across the validation period, which is very nearly the definition of the only
subset executable at 1-2 bps per side. Our one real measurement lives on the only tradable
population we have. But it must be said rather than implied, because "IC 0.030 across 4,873
tickers" is a materially different and unearned claim.

## Tickers versus timestamps, and is alignment an artifact?

**Precision model.** Per-timestamp IC has sampling variance `1/(N−1)`; genuine time-variation
adds `σ_t²`. Pooling `T` timestamps, `SE² = (1/(N−1) + σ_t²)/T`. Cost is `T·N` origins. Both
axes therefore buy precision through `T·(N−1)` — per second they are nearly interchangeable,
tickers marginally worse by `(N−1)/N`.

**Measured `σ_t`.** From the one real IC: `SE = 0.00275` over `T = 40,837` grouped timestamps is
not directly invertible, but the fixture case is. On the synthetic draw, 40 timestamps × 256
tickers measured `±0.0110` against a pure-sampling prediction of `1/√(40·255) = 0.0099` — 1.11×,
i.e. sampling-dominated, as a fixture with i.i.d. targets must be. On real data `σ_t` is
non-zero and unmeasured at fixed width; it is the term that caps what width can buy.

**Which lever.** At the achievable operating point (~90-300 aligned timestamps, ~300-name core)
the timestamp axis is **corpus-bound, not knob-bound**: the 128 cap is inert because fewer
aligned timestamps exist than the cap allows. Width is bounded by the core, ~300, and the draw
already takes 256 of it. So **neither knob has meaningful headroom** — the operating point is
essentially forced, and 256/128 is the right setting because it is the corner the corpus puts us
in. Retuning the sampler buys nothing.

**Is the alignment an artifact of `i·pred_len`? YES, entirely.** It is an artifact of indexing
origins in each ticker's own valid-bar ordinal space. Nothing about the market causes it. The
alternative is to place validation origins on the **shared UTC grid** — take every 192nd slot of
the 5-minute grid inside the validation span and, per ticker, the last valid bar at or before
that instant. Then every ticker with any valid bar near the instant joins the same
cross-section, and the aligned core becomes the whole universe.

**What it would cost.** The machinery exists: `corpus.rs` already builds a shared UTC slot grid
and an occupancy bitset to define market steps (`exogenous-variates-on-shared-utc-grid`,
`slots`/`occupied` around `corpus.rs:738`), and `CorpusTicker::timestamp` is the forward map, so
the inverse is a binary search per (ticker, instant) — ~520k searches once at load, sub-second.
No extra GPU compute: same origin count, same batch shape, same cost per origin. The real cost
is the **contract**: origin placement defines the validation partition, the purge and the
`next-valid-observed-bars` target semantics, so this is a corpus-schema and FORMAT bump that
invalidates every checkpoint's data contract and makes every historical held-out number
non-comparable.

**And the payoff is smaller than it looks.** At EQUAL COST (32,768 origins) an aligned-grid draw
would still be 128 timestamps × 256 tickers — identical `T·(N−1)`, identical standard error. It
buys **representativeness, not precision**: the 256 would be sampled from 4,873 names instead of
from a ~300-name core, and the width ceiling would rise from ~300 to 4,873 so precision could
then be *bought* by paying for it (a 487-name decile instead of 25 is the part that matters, the
decile being what breakeven is computed from). Against that: an at-or-before-instant origin
means each ticker's forecast is as stale as its own last bar, which is honest for a rebalance
but changes what the target means; and a universe-representative IC would be measured over names
we cannot execute at 1-2 bps per side. **Recommendation: it exists, it is affordable in compute
and expensive in contract, and it is NOT the next thing to do** — it improves the
generality of the measurement, not the measurement we need for the tradability decision.

## Read-back verification for addendum 2

> **STILL NOT A MEASUREMENT OF THIS MODEL.** Synthetic fixtures, no model in the loop. The
> breakeven values below are near zero because the fixture forecast is deliberately almost pure
> noise (`target + 19.97 · noise`); that is the expected reading, not a result.

Exercised the real path (`Scorer::new` -> `accumulate` -> `finish` -> `trading` -> `portfolio`
-> `reports::write_trading` -> `.report.bin` -> `report_cli`), `pred_len = 64`, with three draws
written into two scratch generations: a `held-out sample`-shaped one (512 timestamps x 1 ticker),
a UNIFORM cross-section (40 x 256), and a RAGGED one (one 400-name timestamp plus twenty 45-name
ones) built to make the new width series earn its place.

```
$ report_cli 1 timexer_segment_cross_section_census --run <scratch>      (uniform)
1   held-out sample contributing evaluation timestamps=0
    held-out sample mean valid tickers per evaluation timestamp=1
    held-out sample narrowest evaluation timestamp=1
    held-out cross-section contributing evaluation timestamps=40
    held-out cross-section mean valid tickers per evaluation timestamp=256
    held-out cross-section narrowest evaluation timestamp=256
    20 tickers (the minimum a timestamp is scored at)=20
... identical at h = 8, 16, 32, 64

$ report_cli 2 timexer_segment_cross_section_census --run <scratch>      (ragged)
1   held-out cross-section contributing evaluation timestamps=21
    held-out cross-section mean valid tickers per evaluation timestamp=61.904762
    held-out cross-section narrowest evaluation timestamp=45
    20 tickers (the minimum a timestamp is scored at)=20

$ report_cli 1 timexer_segment_portfolio_breakeven --run <scratch>
1   held-out sample ... breakeven per-side cost=NaN   held-out cross-section ... =0.05127943
8   held-out sample ... breakeven per-side cost=NaN   held-out cross-section ... =0.036644995
16  held-out sample ... breakeven per-side cost=NaN   held-out cross-section ... =0.060249172
32  held-out sample ... breakeven per-side cost=NaN   held-out cross-section ... =0.0421242
64  held-out sample ... breakeven per-side cost=NaN   held-out cross-section ... =0.04212308
    ^ a near-noise fixture buying ~0.04 bps of headroom, i.e. nothing. Expected.
```

The ragged generation is the point: `mean 61.90` beside `narrowest 45` is visibly an average of
unequal cross-sections, whereas `mean 256 == narrowest 256` proves uniformity. Before this series
both read as a single number and were indistinguishable. Census y-label as written:

```
count (timestamps that cleared the threshold, and the mean and narrowest valid-ticker width per
evaluation timestamp; mean == narrowest is a uniform draw, mean > narrowest means the width read
beside it is an average of unequal cross-sections; a holding period whose tickers sit under 20
clears no timestamp, and its return series then read as gaps, never as 0)
```

Scratch generations and the throwaway writer deleted.

## Tests (addendum 2)

- `the_cross_section_draw_takes_whole_timestamps_at_one_negotiated_width` — replaces the
  fixed-width test. Blocks of 296, 255, 1 and 256. Asserts the draw keeps THREE timestamps at a
  uniform 255 rather than two at 256, because `3·254 = 762 > 2·255 = 510`: it trades one ticker
  of width for a whole extra cross-section. Asserts the one-ticker timestamp is gone, the draw
  is byte-identical on a second call (step-matchable), and the all-below-floor corpus is refused
  with the explanatory error. The regression that matters: a fixture shaped like the REAL corpus
  — 90 aligned 300-name timestamps plus 20,000 one- and two-ticker timestamps — must YIELD a
  draw (90 x 256) rather than abort, and no member of a below-floor timestamp may enter it.
- `an_unpopulated_cross_section_reads_as_nan_and_never_as_zero_bps` — extended to pin
  `narrowest == mean` on a uniform draw at both 10 tickers (unscored, NaN returns, census still
  reporting the 10 that failed) and 50 tickers (finite).

## Scoped checks (after addendum 2)

- `cargo check -p trading_bot_0 --tests` — 0 errors
- `cargo check -p trading-bot-tui --tests` — 0 errors
- `cargo test -p trading_bot_0 timexer_segment` — 69 passed, 0 failed
- `cargo test -p trading-bot-tui` — 36 passed, 0 failed
