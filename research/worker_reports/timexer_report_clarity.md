# timexer_segment report clarity

## Vocabulary (used verbatim in every label, title, CLI string, doc)

Splits: `training`, `held-out sample`, `held-out full`, `persistence`.
Spaces: `market-neutral`, `raw`.
Label grammar: `<split> <space> <quantity>`, space omitted where a quantity has one space.
Titles: `CausalPatch epoch <e> step <s> | <question> | <scope facts>`.
Scope facts: origins scored per split (one scored origin = one forecast window, so the
held-out sample's count is its window count) and the ticker count; the error chart adds the
latest held-out price RMSE/MAE, timing adds the allocator peak, progress adds target-bar
coverage. Every `y_label` carries the unit and its reading rule.

Retired outright, no aliases and no back-compat readers: `timexer_segment_validation`,
`timexer_segment_absolute`, `timexer_segment_robust`. Old runs' report files are no longer
readable by name; that is expected. `report_cli` now lists the bases actually present in the
generation directory when a name misses, which is what a caller hits with an old base name.

## Base-by-base before / after

| Before | After | What changed |
| --- | --- | --- |
| `timexer_segment_validation` (NLL in nats ≈2 on the same axis as σ-scaled MSE ≈45 and dimensionless median-window ratios) | `timexer_segment_skill` + `timexer_segment_loss` + `timexer_segment_error` | Three units split into three bases. `_skill` is ratios only (`ratio vs persistence (dimensionless; < 1 = skill)`): aggregate + median-window MSE ratio, both splits, both spaces, parity 1.0. `_loss` is `nats per bar`. `_error` is `σ-scaled squared log-return` on a log (Symlog) axis. |
| `timexer_segment_absolute` (mixed ratios and σ-scaled levels on one linear axis) | folded into `timexer_segment_skill` (the ratios) and `timexer_segment_error` (the levels) | Base deleted. The raw-space MSE ratio now sits beside the market-neutral one on the dimensionless axis, and the raw levels beside the market-neutral levels on the log axis. |
| `timexer_segment_calibration` | `timexer_segment_calibration` | Content unchanged. Labels `held-out sample/full within 1σ / 1.96σ`; reference lines `nominal 1σ = 0.683`, `nominal 1.96σ = 0.950`; y `fraction of valid target bars`. |
| `timexer_segment_horizon` (forecast level + persistence level + ratio per split per space, Symlog, 12 series on one axis) | `timexer_segment_horizon` + `timexer_segment_horizon_error` | `_horizon` is ratio curves only + parity 1.0 (linear, dimensionless). `_horizon_error` carries the σ-scaled forecast and persistence levels on a log axis. |
| `timexer_segment_robust` (two ratios near 1 and three rates near 0.5, plus two parity lines, on one linear axis) | `timexer_segment_horizon_robust` + `timexer_segment_horizon_rates` | Base deleted. `_horizon_robust` = MAE ratio + trimmed MSE ratio + parity 1.0. `_horizon_rates` = close win rate, directional hit rate, predicted-up share + parity 0.5. |
| `timexer_segment_timing` | `timexer_segment_timing` | Contents untouched (StepEfficiency's). Labels only: `training step (interval mean)`, `training step host loader wait (interval mean; overlaps GPU work)`, `held-out evaluation total`, `held-out evaluation: host batch wait / H2D and forward / metric accumulation`, `after held-out evaluation: candle windows and checkpoints`. Their six `training step phase: ...` labels are verbatim. |
| `timexer_segment_progress` | `timexer_segment_progress` | Labels name their own denominator: `training target bars completed`, `held-out invalid forecast candles per forecast bar`, `held-out top-1% \|target\| share of squared error`. y is `fraction of the series' own population (dimensionless; 0 to 1)`, because this base legitimately mixes denominators. |
| `timexer_segment_candles` | `timexer_segment_candles` | Title is now `held-out full \| GOOGL \| 2026-02-25 11:05 EST \| 192 bars \| conditional-mean path re-based to origin close \| epoch 3 step 45000` — split, ticker, timestamp with timezone, horizon, the re-basing statement, in that order, 125 characters. y `price (USD)`. The old title advertised "market-neutral OHLC forecast re-based onto β times the realized market path", which is the construction, not what the reader sees drawn. |
| `timexer_segment_hardware`, `_benchmark`, `_benchmark_phases` | unchanged | benchmark.rs, no split/space vocabulary in them; StepEfficiency owns the TFLOPS/HBM series they are adding to `_benchmark`. |

## Taxonomy deviations from the brief, with reasons

- **`timexer_segment_error` was NOT split by space.** The brief allowed a second base if the
  raw/market-neutral level gap makes one axis unreadable. It does not: the two differ by a
  multiplicative factor (fixture 45.5 vs 62.5; per-horizon 38→152 vs 53.2→212.8), which a log
  axis renders as a constant vertical offset. Splitting them would destroy the side-by-side
  comparison the two spaces exist to support while fixing nothing.
- **`write_horizon` writes all four `timexer_segment_horizon*` bases** and `write_robust` is
  deleted. The four charts share one validation pass, one split iteration and one scope
  string; two entry points would duplicate all three and give runner.rs a second call site to
  keep in sync.
- **`timexer_segment_progress` y unit is `fraction of the series' own population`, not
  `fraction of valid target bars`.** Two of its three series are shares of the training
  corpus, so a single-denominator axis label would be a false claim.

## TUI ordering

`meta_chart_bases()` sorts alphabetically, which put `timexer_segment_skill` 13th of 14 in its
own family. `load_latest_meta_charts` now sorts by `(headline_rank(path), path)` with
`HEADLINE_CHART_BASES = ["timexer_segment_skill"]`, so the headline panel is panel one and
everything else keeps its stable alphabetical order.

## Registry, both directions

`TIMEXER_SEGMENT_REPORT_BASES` (shared/src/report.rs) is rewritten headline-first with the 14
live bases. The existing forward sweep in
`the_meta_chart_list_looks_for_every_registered_writer_base` catches a base the writer produces
that the TUI never scans; a new `assert_eq!` compares the sorted registry against every
`timexer_segment_*` name `meta_chart_bases()` scans, catching the reverse — a retired name the
TUI still scans, which renders as a permanently blank panel indistinguishable from a metric
that stopped moving. Two further assertions pin the headline rank.

## Internal identifiers deliberately kept

| Identifier | Why |
| --- | --- |
| `Metrics::validation_nll/_mse`, `persistence_nll/_mse` | Which split they hold is a runtime property of the point (`validation_is_full`), not of the field, so no split name belongs in the name. Renaming would churn runner.rs's `Evaluation` and the scorer while StepEfficiency was editing them. Documented on the fields. |
| `Metrics::absolute_mse`, `absolute_persistence_mse`, `HorizonCurve::absolute_*` | The internal spelling of `raw`. Same concurrent-edit reasoning; documented on both structs and in docs. |
| `--preview-patience` flag, `weights/preview-latest`, `weights/preview-best` | CLI and filesystem identifiers, not chart text. Renaming breaks queued job scripts and existing weight directories. Their `--help` text now uses the vocabulary; the docs state the retention explicitly. |
| `Manifest::best_preview_nll`, `CorpusContract::validation_target_bars/_remainder_bars` | Serialized manifest and contract fields; renaming is a format break for zero presentation gain. The corpus title they feed now reads `held-out targets` / `unused held-out remainder`. |
| `tui/src/chart_viewer.rs` `render_preview` | The chart preview pane. Unrelated to data splits. |

## Grep proof

`grep -rn <pat> trading_bots/src/torch/timexer_segment/reports.rs tui/src report_cli/src docs/timexer_segment.md`

| pattern | matches |
| --- | --- |
| `full-validation` | 0 |
| `relative MSE` | 0 |
| `absolute MSE` | 0 |
| `TimeXer epoch` | 0 |
| `preview` | 11, all accounted for: 2 in the reports.rs doc comment that names the retired spellings in order to explain the retirement; 3 in docs (`best_preview_nll`, `--preview-patience`, `weights/preview-*`, all listed above as retained identifiers); 6 in `tui/src/chart_viewer.rs` for the chart preview pane. |

A second sweep over string literals,
`grep -rnE '"[^"]*(validation\|preview\|relative\|absolute)[^"]*"'` across reports.rs,
tui/src/main.rs and report_cli/src/main.rs, leaves only: unrelated `ga_validation_*` /
`planner_validation_*` base names, report_cli's "safe relative components" path error, and the
reports.rs doc comment. No chart label, title or y_label contains a retired word.

## Render proof

The live run `timexer-market-neutral-20260906/gens/1` still holds the OLD bases
(`timexer_segment_validation`, `_absolute`, `_robust`), so it cannot demonstrate the new
format. Proof came from a throwaway `render_proof` module in reports.rs driving a synthetic
two-point `Metrics` fixture (one `held-out sample` point at step 44000, one `held-out full` at
45000), two `HorizonCurve`s and one `CandleWindow`, written to
`/tmp/report-clarity-proof/gens/1`, then read back with `report_cli --run-root`. The module and
the temp directory are deleted; reports.rs holds no test module.

Titles and y-axis units as written (character counts from the fixture):

```
timexer_segment_skill            [179] CausalPatch epoch 3 step 45000 | does the forecast beat persistence? | held-out sample 2048 fixed windows, one scored origin each, held-out full 24813 scored origins, 4187 tickers
                                       y: ratio vs persistence (dimensionless; < 1 = skill)
timexer_segment_loss             [209] ... | how good is the predictive density against the persistence prior? | <same scope>
                                       y: nats per bar
timexer_segment_error            [256] ... | how large is the squared error, forecast against persistence? | <scope>; latest held-out price RMSE 0.4123, MAE 0.2871 USD
                                       y: σ-scaled squared log-return
timexer_segment_calibration      [196] ... | do the predicted σ bands cover the realized targets? | <scope>
                                       y: fraction of valid target bars
timexer_segment_horizon          [195] ... | does the forecast beat persistence at each horizon? | <scope>
                                       y: ratio vs persistence (dimensionless; < 1 = skill)
timexer_segment_horizon_error    [221] ... | how large is the squared error at each horizon, forecast against persistence? | <scope>
                                       y: σ-scaled squared log-return
timexer_segment_horizon_robust   [224] ... | does the forecast still beat persistence once the |target| tail cannot dominate? | <scope>
                                       y: ratio vs persistence (dimensionless; < 1 = skill)
timexer_segment_horizon_rates    [194] ... | how often is the close forecast on the right side? | <scope>
                                       y: fraction of valid target bars (win and hit rates > 0.5 = skill; up share is bias)
timexer_segment_progress         [165] ... | how much of the corpus is consumed, and how clean is the held-out signal? | 90000 / 100000 unique training target bars, 4187 tickers
                                       y: fraction of the series' own population (dimensionless; 0 to 1)
timexer_segment_timing           [199] ... | where does the wall clock go? | <scope>; peak allocator 21000 MiB
                                       y: milliseconds
timexer_segment_candles          [125] held-out full | GOOGL | 2026-02-25 11:05 EST | 192 bars | conditional-mean path re-based to origin close | epoch 3 step 45000
                                       y: price (USD)
```

First two lines of every base through `report_cli 1 <base> --run-root /tmp/report-clarity-proof`:

```
timexer_segment_skill
44000  held-out sample market-neutral MSE ratio=0.98913044  held-out sample raw MSE ratio=0.99206346  held-out sample market-neutral median-window MSE ratio=0.985  held-out full market-neutral MSE ratio=NaN  held-out full raw MSE ratio=NaN  held-out full market-neutral median-window MSE ratio=NaN  parity 1.0=1
45000  held-out sample market-neutral MSE ratio=NaN  held-out sample raw MSE ratio=NaN  held-out sample market-neutral median-window MSE ratio=NaN  held-out full market-neutral MSE ratio=0.95652175  held-out full raw MSE ratio=0.96825397  held-out full market-neutral median-window MSE ratio=0.985  parity 1.0=1

timexer_segment_loss
44000  training NLL=1.94  held-out sample NLL=2.05  held-out sample persistence NLL=2.14  held-out full NLL=NaN  held-out full persistence NLL=NaN
45000  training NLL=1.94  held-out sample NLL=NaN  held-out sample persistence NLL=NaN  held-out full NLL=2.01  held-out full persistence NLL=2.14

timexer_segment_error
44000  training MSE=41.2  held-out sample market-neutral MSE=45.5  held-out sample market-neutral persistence MSE=46  held-out sample raw MSE=62.5  held-out sample raw persistence MSE=63  held-out full market-neutral MSE=NaN  held-out full market-neutral persistence MSE=NaN  held-out full raw MSE=NaN  held-out full raw persistence MSE=NaN
45000  training MSE=41.2  held-out sample market-neutral MSE=NaN  held-out sample market-neutral persistence MSE=NaN  held-out sample raw MSE=NaN  held-out sample raw persistence MSE=NaN  held-out full market-neutral MSE=44  held-out full market-neutral persistence MSE=46  held-out full raw MSE=61  held-out full raw persistence MSE=63

timexer_segment_calibration
44000  held-out sample within 1σ=0.66  held-out sample within 1.96σ=0.94  held-out full within 1σ=NaN  held-out full within 1.96σ=NaN  nominal 1σ = 0.683=0.6827  nominal 1.96σ = 0.950=0.95
45000  held-out sample within 1σ=NaN  held-out sample within 1.96σ=NaN  held-out full within 1σ=0.66  held-out full within 1.96σ=0.94  nominal 1σ = 0.683=0.6827  nominal 1.96σ = 0.950=0.95

timexer_segment_horizon
1  held-out sample market-neutral MSE ratio=0.96153843  held-out sample raw MSE ratio=0.9859155  held-out full market-neutral MSE ratio=0.96153843  held-out full raw MSE ratio=0.9859155  parity 1.0=1
2  held-out sample market-neutral MSE ratio=0.96153843  held-out sample raw MSE ratio=0.9859155  held-out full market-neutral MSE ratio=0.96153843  held-out full raw MSE ratio=0.9859155  parity 1.0=1

timexer_segment_horizon_error
1  held-out sample market-neutral MSE=38  held-out sample market-neutral persistence MSE=39.52  held-out sample raw MSE=53.2  held-out sample raw persistence MSE=53.96  held-out full market-neutral MSE=41  held-out full market-neutral persistence MSE=42.64  held-out full raw MSE=57.4  held-out full raw persistence MSE=58.22
2  held-out sample market-neutral MSE=76  held-out sample market-neutral persistence MSE=79.04  held-out sample raw MSE=106.4  held-out sample raw persistence MSE=107.92  held-out full market-neutral MSE=82  held-out full market-neutral persistence MSE=85.28  held-out full raw MSE=114.8  held-out full raw persistence MSE=116.44

timexer_segment_horizon_robust
1  held-out sample market-neutral MAE ratio=0.99  held-out sample market-neutral trimmed MSE ratio (top-1% |close| bars dropped)=0.97  held-out full market-neutral MAE ratio=0.99  held-out full market-neutral trimmed MSE ratio (top-1% |close| bars dropped)=0.97  parity 1.0=1
2  held-out sample market-neutral MAE ratio=0.99  held-out sample market-neutral trimmed MSE ratio (top-1% |close| bars dropped)=0.97  held-out full market-neutral MAE ratio=0.99  held-out full market-neutral trimmed MSE ratio (top-1% |close| bars dropped)=0.97  parity 1.0=1

timexer_segment_horizon_rates
1  held-out sample close win rate vs persistence=0.51  held-out sample close directional hit rate=0.52  held-out sample close predicted-up share=0.49  held-out full close win rate vs persistence=0.51  held-out full close directional hit rate=0.52  held-out full close predicted-up share=0.49  parity 0.5=0.5
2  held-out sample close win rate vs persistence=0.51  held-out sample close directional hit rate=0.52  held-out sample close predicted-up share=0.49  held-out full close win rate vs persistence=0.51  held-out full close directional hit rate=0.52  held-out full close predicted-up share=0.49  parity 0.5=0.5

timexer_segment_progress
44000  training target bars completed=0.9  held-out invalid forecast candles per forecast bar=0.001  held-out top-1% |target| share of squared error=0.31
45000  training target bars completed=0.9  held-out invalid forecast candles per forecast bar=0.001  held-out top-1% |target| share of squared error=0.31

timexer_segment_timing
44000  training step (interval mean)=310  training step host loader wait (interval mean; overlaps GPU work)=4  training step phase: host batch=3.5  training step phase: H2D=6.1  training step phase: forward backbone=120  training step phase: forward head and loss=74  training step phase: backward=96  training step phase: optimizer=11  held-out evaluation total=900  held-out evaluation: host batch wait=60  held-out evaluation: H2D and forward=700  held-out evaluation: metric accumulation=140  after held-out evaluation: candle windows and checkpoints=220
45000  (identical; the fixture repeats the timing point)

timexer_segment_candles
0  bar_from_origin=-3  actual=o:100.000000,h:101.000000,l:99.000000,c:100.500000
1  bar_from_origin=-2  actual=o:100.000000,h:101.000000,l:99.000000,c:100.500000
```

`report_cli` on a retired name now names the alternatives instead of only reporting ENOENT:

```
$ report_cli 1 timexer_segment_validation --run-root /tmp/report-clarity-proof
Error: failed to read report /tmp/report-clarity-proof/gens/1/timexer_segment_validation.report.bin;
bases in /tmp/report-clarity-proof/gens/1: timexer_segment_calibration, timexer_segment_candles,
timexer_segment_error, timexer_segment_horizon, timexer_segment_horizon_error,
timexer_segment_horizon_rates, timexer_segment_horizon_robust, timexer_segment_loss,
timexer_segment_progress, timexer_segment_skill, timexer_segment_timing
```

## Verification

- `./torch-env.sh cargo check -p trading_bot_0 --tests`: 0 errors.
- `cargo check -p trading-bot-tui --tests`, `-p report_cli --tests`: 0 errors each.
- `./torch-env.sh cargo test -p trading_bot_0 timexer_segment`: 32 passed, 0 failed.
- `cargo test -p trading-bot-tui`: 36 passed (includes
  `the_meta_chart_list_looks_for_every_registered_writer_base` with the new bidirectional
  assertion). `cargo test -p report_cli`: 3 passed.
- No new permanent tests beyond the registry assertion the brief required. The render proof was
  a throwaway and is deleted.
- A `CudaGraph` compile error appeared in `torch/optim/muon.rs` / `torch/cuda/graph.rs` after my
  clean readings, from StepEfficiency's concurrent `Engine::timed_step` work. Neither file is in
  my change set.

## Coordination with StepEfficiency

- Agreed single-writer split for reports.rs: I own the whole file, they own
  model.rs/compute.rs/runner.rs measurement. Their `StepPhases` struct went into reports.rs with
  their exact field names (`host_batch_ms`, `h2d_ms`, `forward_backbone_ms`, `forward_head_ms`,
  `backward_ms`, `optimizer_ms`), `#[derive(Debug, Clone, Copy, Default)]` as they asked, plus
  `Metrics::step_phases: Option<StepPhases>`.
- Their six `training step phase: ...` labels are written verbatim into the timing chart. They
  approved my renaming the pre-existing timing labels to the vocabulary and confirmed they would
  not re-touch them.
- They landed runner.rs first (final_origin channel-last adapter, phase-sampled step,
  non-finite counter, both `Metrics` literals carrying `step_phases`). I rebased onto that and
  preserved every line of it, including `Some(StepPhases { .. })` in the training loop — I did
  not reset it to `None`.
- My runner.rs footprint, disclosed to them before editing: `tickers` on both `Metrics`
  literals, both `write_horizon` calls (now `Option<HorizonSplit>` + ticker count), deletion of
  both `write_robust` calls, `HorizonSplit` added to the `use` list, three printlns and two
  clap doc comments moved to the vocabulary.
- They add TFLOPS/HBM series to `timexer_segment_benchmark` in benchmark.rs; I confirmed that is
  the right panel and did not touch benchmark.rs. They add no bases, so
  `TIMEXER_SEGMENT_REPORT_BASES` and `meta_chart_bases` were mine alone to rewrite.
