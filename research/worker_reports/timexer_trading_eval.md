# Is the CausalPatch forecaster tradable? Diagnostics, not MSE

Evaluation-only change. Objective, targets, architecture and the training loop are untouched.
Seven new `.report.bin` bases come out of the **existing single evaluation pass**, on device,
with one host transfer of 34 per-horizon sums plus three moments per reported holding period.
No CSV, no log scraping, no separate backtest harness, no per-element CPU loop.

## What each base answers

| Base | y unit | The question it settles |
| --- | --- | --- |
| `timexer_segment_decomposition` | share of the persistence MSE | Is the MSE gain a constant tilt or a conditional signal? |
| `timexer_segment_signal` | correlation coefficient | How much information does the demeaned close forecast carry, and is it significant? |
| `timexer_segment_offset` | σ-scaled mean log return | How big is the tilt, and what does conviction pay? |
| `timexer_segment_tradable` | ratio vs the anchor's own persistence | Does the edge survive a mid anchor and a one-bar execution delay? |
| `timexer_segment_tradable_rates` | fraction of scored bars | Is the *sign* right under those anchors and at high conviction? |
| `timexer_segment_portfolio` | basis points per holding period | What does a decile long/short earn after costs? |
| `timexer_segment_portfolio_sharpe` | annualized Sharpe | What Sharpe, after the same cost sweep? |

Seven bases and not four because the units are seven. A share of MSE near 0.06 beside a
correlation near 0.01 beside a σ-scaled mean near 0.001 beside a ratio near 1 beside a rate
near 0.5 beside basis points beside a Sharpe renders six of the seven as flat lines — the exact
failure `research/worker_reports/timexer_report_clarity.md` split the existing family to avoid.
Splits (`held-out sample` / `held-out full`), spaces (`market-neutral`), the
`<split> <space> <quantity>` label grammar, the `epoch/step | question | scope` title and the
unit-plus-reading-rule `y_label` all follow that vocabulary.

### 1. The decomposition — where the 0.938 comes from

Write the close forecast as `ŷ = μ + g`, `μ` the population mean forecast at that horizon,
`mean(g) = 0`, and let `β̂ = mean(y·g)/mean(g²)`. As shares of the persistence MSE `mean(y²)`:

- **offset gain** `= (2μȳ - μ²)/mean(y²)` — what a *constant* forecast equal to `μ` already
  earns. Not a prediction; not capturable.
- **demeaned gain** `= (mean(y·g)²/mean(g²))/mean(y²)` — the demeaned conditional forecast at
  its own best scale. This is *identically* `ρ²·var(y)/mean(y²)`, i.e. the `1 - ρ²` implied MSE
  ratio, in the same unit as the measured total. The task asked for `1 - ρ²` beside the measured
  ratio on one chart; that is these two lines, and no third redundant series is needed.
- **cross term** `= -(β̂-1)²·mean(g²)/mean(y²)` ≤ 0 — the cost of mis-scaling the demeaned
  forecast's amplitude.

The three sum to the total **exactly** (unit-tested against the closed form, and the smoke run
reproduced `0.014144 + 0.008462 - 0.517189 = -0.494583` to the last digit). Note that a
decomposition using a "demeaned gain" of `2·mean(y·g) - mean(g²)` has an identically-zero cross
term — algebraically forced, a useless chart line. The best-scale/mis-scaling split is the one
that carries information, and it is still literally `total - offset - demeaned`.

The all-channel total gain, the complement of the headline four-channel ratio, sits on the same
axis so the chart connects to the 0.938 the reader arrived with. The mean predicted close
coordinate lives on `_offset` because it is in σ units, not a share of MSE.

### 2. The signal family

Pooled Pearson and pooled Spearman (ordinal ranks via double `argsort` with invalid bars pushed
past every valid one by a sentinel; no tie averaging, immaterial for continuous coordinates),
plus the **cross-sectional IC**: the Pearson correlation computed *within* each evaluation
timestamp across that timestamp's tickers, averaged over timestamps, with its ±1 standard-error
band drawn as two series so significance is one glance. A timestamp needs ≥ 20 valid tickers to
contribute (`CROSS_SECTION_MIN`).

### 3. The microstructure test

Persistence anchors on the last *trade* close. Bid-ask bounce means the trade close is displaced
from the mid by roughly half a spread with alternating sign, so a forecast that leans toward the
mid beats close-anchored persistence on MSE while being uncapturable — your own fill pays the
spread. Two anchors expose that:

- **(i) Mid anchor.** Persistence anchored on the log midpoint of the origin bar's high/low,
  `(ln high + ln low)/2 / σ`. The model's error is unchanged; only the denominator moves. Hit
  rate is `sign(ŷ - mid)` against `sign(y - mid)`.
- **(ii) One-bar execution delay.** The bar `t+1` close to bar `t+h` close move, forecast by
  `ŷ_h - ŷ_1` (the origin's own two coordinates differenced — the β and market-drift terms
  telescope, so this needs no extra data), against that same delayed persistence, which is
  again zero. Undefined at `h = 1`, where the delayed move is identically zero; that column is
  reported as NaN and renders as a gap, which is the honest rendering.

Plus the hit rate and mean realized return on the side taken, conditioned on the **top and
bottom decile of `|g|`** — conviction measured on the *demeaned* signal, because a raw `|ŷ|`
decile with a large constant tilt selects on the tilt rather than on conviction.

### 4. The portfolio

At each evaluation timestamp, rank that timestamp's tickers by predicted `h`-bar market-neutral
close return (`ŷ · σ`, so the ranking is in return space, not σ space), go equal-weighted long
the top decile and short the bottom, hold `h` bars. `h = 1, 4, 16, 64, 192`. Mean per-period
return in bps and annualized Sharpe (252 × 78 = 19,656 five-minute bars per year), each with one
series per per-side cost in 0, 1, 2, 5, 10 bps. A per-side cost `c` is charged on entry and exit
of both legs, so the spread return loses `4c` bps.

**Stated plainly in `docs/timexer_segment.md`: this is an upper bound.** It ignores market impact
and borrow availability and cost; it treats consecutive holds as non-overlapping when for `h > 1`
they overlap, which inflates the Sharpe; and a timestamp's cross-section is the held-out origins
that happen to share that timestamp, not a tradable universe snapshot.

## How the cross-sections were solved (item 5)

The `Scorer` already held per-(window, bar) arrays. It now keeps the **signed close coordinate,
its mask, and per-window σ and mid anchor** — which is everything every new statistic needs. The
tail-trim threshold array (`|close target|` with `-1` on invalid bars) is *reconstructed* from
them in `finish` instead of stored, so the net memory change is one extra `[origins, pred_len]`
float array, not four.

Grouping by origin timestamp: `score` already has the `&[WindowRef]` origin list, and
`corpus.ticker(ref).timestamp(ref.origin)` resolves the origin bar's UTC millisecond from the
memory-mapped bar header — one header read per scored window, on the host, once, before the
batch loop. Dense-ranking the distinct timestamps turns every per-timestamp reduction into a
single `index_add` over dimension 0 into a `[timestamps, pred_len]` accumulator.

Per-timestamp *decile membership* without a loop over timestamps: rank the predicted returns
globally with a double `argsort`, then `argsort` the composite key `group·windows + global rank`,
which orders windows by timestamp and, within a timestamp, by predicted return. Subtracting the
group's cumulative start offset from the sorted position gives the within-timestamp rank.
Invalid windows carry a sentinel prediction, so they sort after every valid member of their own
timestamp and the `rank < valid count` test excludes them. Three `argsort`s of length `origins`
per reported horizon, five horizons.

## Files touched

- `trading_bots/src/torch/timexer_segment/runner.rs` — `FinalOrigin` gains `mid`/`sigma`;
  `Scorer` gains `bar_forecast`/`bar_target`/`bar_valid`/`window_mid`/`window_sigma`/`groups`
  and drops `bar_close`; new `Scorer::trading()` and `Scorer::portfolio()`; `Evaluation` gains
  `trading`/`portfolio`; `score` builds the timestamp group index; `train` retains both splits'
  curves in a new `Curves` struct; `train` and `evaluate` both call `reports::write_trading`.
  New constants `CROSS_SECTION_MIN = 20`, `PORTFOLIO_HORIZONS`, `COSTS_BPS`, `BARS_PER_YEAR`.
- `trading_bots/src/torch/timexer_segment/reports.rs` — `TradingCurve`, `PortfolioCurve`,
  `TradingSplit`, `write_trading`, seven unit constants.
- `shared/src/report.rs` — seven names appended to `TIMEXER_SEGMENT_REPORT_BASES` (21 bases).
- `tui/src/main.rs` — untouched: `meta_chart_bases` extends from the shared list, so both
  directions of the registry contract hold by construction and the existing bidirectional test
  proves it.
- `docs/timexer_segment.md` — the seven bases in the TUI report list, and a scoring-section
  paragraph on the single-pass mechanics and the timestamp grouping.

## Verification

- `./torch-env.sh cargo check -p trading_bot_0 --tests`: zero errors.
  `cargo check --manifest-path tui/Cargo.toml --tests`: zero errors. `report_cli` builds within
  the workspace check with zero errors.
- `./torch-env.sh cargo test -p trading_bot_0 timexer_segment`: **33 passed, 0 failed**.
- `cargo test --manifest-path tui/Cargo.toml the_meta_chart_list`: passed — the bidirectional
  registry assertion still agrees in both directions with 21 bases.
- `trading_statistics_match_scalar_reference` compares every new statistic against a plain
  scalar reference over 90 windows / 3 timestamps / 6 horizons accumulated in three uneven
  batches: total/offset/demeaned/cross gains (and the cross term against its closed form
  `-(β̂-1)²mean(g²)/mean(y²)` independently), all-channel gain, mean forecast and target,
  Pearson, Spearman, the mid-anchored ratio and hit rate, the close hit rate, the delayed ratio
  and hit rate (with `h = 1` asserted NaN), the decile thresholds, decile returns, conviction
  spread, decile hit rates, the cross-sectional IC and its standard error, and the decile-spread
  return and annualized Sharpe at every cost level. **Tolerances, set by where the arithmetic
  happens:** `1e-6` relative for the gain decomposition, correlations, anchored ratios and
  decile statistics (residuals and squares are formed elementwise in the resident fp32 arrays
  before an fp64 reduction, while the reference is fp64 throughout — the observed gap was
  1.1e-8 relative); `1e-4` for the cross-sectional IC, whose per-timestamp scatter also
  accumulates in fp32; `1e-9` for the decile-spread return and Sharpe, which the backtest
  computes in fp64 end to end.
- A throwaway writer smoke test (since removed) wrote all seven bases, read them back through
  `shared::report::read_report`, and confirmed each is an `IndexedLines` chart whose every
  series length equals its step count, with the intended labels, titles, units and x steps
  (`[1..6]` for the per-horizon bases, `[1, 4]` for the portfolio bases at `pred_len = 6`).

## The CLI to queue

Against run `timexer-market-neutral-20260906`, `weights/best`. Nothing was submitted from here:

```bash
mlq submit --name timexer-trading-eval --max-parallel-runs 1 --time-limit 3h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh evaluate-timexer-segment \
  --checkpoint training/runs/timexer-market-neutral-20260906/weights/best \
  --output training/runs/timexer-market-neutral-20260906/gens/trading \
  --batch-size 256
```

Then read the bases with, e.g.:

```bash
cargo run -p report_cli -- --dir training/runs/timexer-market-neutral-20260906/gens/trading \
  timexer_segment_decomposition timexer_segment_signal timexer_segment_tradable \
  timexer_segment_portfolio
```

`evaluate` scores the whole `held-out full` validation population, so every new base is written
under the `held-out full` split with `None` for the sample. The same series also appear at every
`--eval-every` interval of any future training run, for both splits, at no extra pass.

## Prediction, with reasoning

**The `h = 1` edge is almost entirely an unconditional offset plus microstructure, and the
tradable component is near zero.** Concretely, at `h = 1` I expect:

- **`_decomposition`:** close-channel total gain ≈ 0.05–0.08, of which the **offset component
  carries the large majority** — plausibly 0.04–0.07 — and the **demeaned component is ≤ 0.005**,
  i.e. `|ρ| ≤ 0.07`. The cross term will be small and negative. Reasoning: this is forced by
  arithmetic already in evidence. A genuine conditional mean with MSE ratio 0.938 requires
  `ρ ≈ 0.25`, which for a roughly symmetric residual return distribution implies a directional
  hit rate near `0.5 + arcsin(0.25)/π ≈ 0.58`. The measured close hit rate is 0.498. Those two
  facts cannot both describe a conditional signal; they *can* both describe a near-constant
  forecast, whose hit rate is the unconditional up-rate of the target and whose MSE gain is
  `(2μȳ - μ²)/mean(y²)` — nonzero whenever `μ` has the same sign as `ȳ` and `|μ| < 2|ȳ|`. The
  predicted-up share of 0.066 at `h = 1` says the forecast is essentially always negative:
  `μ < 0` with `|μ|` large relative to the forecast's own dispersion. So the offset is real and
  large, and it explains the gain without any signal.
- **`_offset`:** mean predicted close coordinate distinctly negative at `h = 1` and drifting
  toward 0 as `h` grows (0.066 → 0.161 → 0.428 predicted-up share is exactly the signature of a
  fixed negative tilt whose magnitude shrinks relative to the `√h`-growing forecast dispersion).
  Mean *realized* coordinate also slightly negative — the market-neutral residual of a
  `β`-shrunk-toward-1 demeaning over a universe of 4,873 mostly small-cap names carries a small
  negative mean, and it is that agreement of signs that makes the offset profitable in MSE.
- **`_tradable`:** the mid-anchored ratio at `h = 1` **well above the close-anchored ratio and
  likely at or above 1.0**, and the one-bar-delayed ratio at `h = 2` (its first defined point)
  **at or above 1.0**. The mid-anchored denominator removes the bounce variance that the
  close-anchored denominator contains, so if any part of the "gain" was leaning toward the mid,
  that part disappears; and the delay removes the current bar's bounce from the target
  altogether. Both hit rates at 0.50 ± 0.01. This is the decisive chart: **if the mid-anchored
  ratio stays near 0.94 while the close-anchored one does, the bounce story is wrong and the
  gain is genuinely an offset story alone**; if it jumps to ≥ 1.0, bounce is carrying part of it.
  I expect a split verdict — offset dominant, bounce a secondary contributor — because a
  constant tilt cannot exploit bounce (bounce is sign-alternating and zero-mean), so the two
  mechanisms are largely independent and both should show.
- **`_signal`:** cross-sectional IC at `h = 1` statistically indistinguishable from zero — mean
  within ±1 s.e. of 0. The cross-sectional IC is the one statistic in the family that is immune
  to *both* confounds: it demeans within each timestamp (killing any constant tilt, and any
  common market residual) and it is a rank/level correlation across tickers rather than a
  variance ratio against an anchor. This is therefore the number I would trust as "is there
  alpha", and I expect it to say no at `h = 1`.
- **`_signal` at longer horizons:** the most likely place for a small genuine IC is `h = 16–64`,
  where the hit rate already reads 0.507 and where a slow signal would not be swamped by
  microstructure. 0.507 corresponds to `ρ ≈ 0.011`, so expect a cross-sectional IC in the
  0.005–0.02 band — small, but with 4,873 tickers per timestamp and thousands of timestamps its
  standard error will be small enough that the band may clear zero. That is the only result here
  that would justify further work.
- **`_portfolio`:** gross decile-spread return at `h = 16` of order 0.5–3 bps per period, and
  **negative at every cost level ≥ 1 bps per side** (which deducts 4 bps). Annualized gross
  Sharpe may look impressive — the overlap and the impact/borrow omissions both flatter it — and
  that is exactly why the doc labels it an upper bound. At `h = 1` I expect the spread to be
  negative even gross, because the top-`|signal|` decile at `h = 1` is dominated by
  bounce-driven names where the predicted move is a mean-reversion of a spread crossing that the
  strategy itself cannot capture at the mid.

If those hold, the honest answer to "is this forecaster useful for trading?" is **not as it
stands**: the MSE improvement is a calibration of the unconditional mean plus an anchoring
artifact, and the conditional information content at `h = 1` is below the transaction-cost floor
by an order of magnitude. The place to look next is the `h = 16–64` cross-sectional IC.
