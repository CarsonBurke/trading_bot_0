# Step-indexed per-horizon and generalization-gap reports

Fixes the gap that cost `timexer_recipe_v3_compare.md` its central claim: the horizon-indexed
bases (`timexer_segment_horizon*`, `_decomposition`, `_offset`, `_signal`, `_tradable*`) are
rewritten in place at every evaluation, so the two runs' per-horizon curves existed only at
step 9000 (baseline) and step 4000 (recipe) and the WHERE-does-the-regression-live section had
to carry a "NOT step-matched" caveat. The horizon-indexed bases are unchanged — x = bars ahead
is the right view of one moment. Three step-indexed bases are added beside them.

## New bases

| base | x | y-label | series |
| --- | --- | --- | --- |
| `timexer_segment_generalization_gap` | optimizer step | `nats per bar (training NLL minus held-out NLL; \|gap\| shrinking while the held-out NLL rises = the training split's own structure is being fit)` | `training minus held-out sample NLL`, `training minus held-out full NLL`, `no gap 0.0` |
| `timexer_segment_horizon_steps` | optimizer step | `ratio vs persistence (dimensionless; < 1 = skill)` | `{held-out sample\|held-out full} {market-neutral\|raw} MSE ratio at h = {1,8,64,192}` (8 per split-space-horizon cross), `parity 1.0` |
| `timexer_segment_horizon_steps_scaling` | optimizer step | `share of the persistence MSE (dimensionless; 0 = the demeaned forecast is already at its best scale, < 0 = what mis-scaling its amplitude costs)` | `{held-out sample\|held-out full} market-neutral mis-scaling cross term at h = {1,8,64,192}`, `no mis-scaling 0.0` |

Vocabulary is the one fixed by the report-clarity pass: splits `training` / `held-out sample` /
`held-out full`, spaces `market-neutral` / `raw`, ratios dimensionless with `< 1 = skill`, NLL
and the gap in nats per bar. Decision horizons `{1, 8, 64, 192}` = `reports::DECISION_HORIZONS`
(h ≤ 8 is where the only real skill lives, h ≥ 64 is where the drift failure appears); a
horizon past the run's `pred_len` is omitted, never written as a gap. A split's series is NaN at
steps that scored the other split, exactly as in `timexer_segment_skill`. Plumbing:
`Metrics::horizons: Vec<HorizonPoint>`, filled by `reports::horizon_track(&horizon, &trading)`
at both `Metrics` construction sites in `runner.rs`; the step history already lived in `points`,
so nothing new is retained and no metric arrives by a side channel. All three bases are in
`shared::report::TIMEXER_SEGMENT_REPORT_BASES`; the TUI extends `meta_chart_bases` from that
slice and its bidirectional test now also pins the three names in both directions.

## Which analysis question each series answers

- `training minus held-out sample NLL` — "is this run fitting the training period?" The
  0.25 → 0.08 collapse inside epoch 1 was the clearest single signal in the recipe comparison
  and had to be subtracted by hand from two `timexer_segment_loss` curves at every step.
- `... MSE ratio at h = 1 / 8` — "did the short end regress?" (recipe: it did not, 0.953 vs
  0.948 and 0.894 vs 0.918) — now readable at any matched step instead of one snapshot.
- `... MSE ratio at h = 64 / 192` — "when does the long end cross 1.0, and how fast?" This is
  the claim that needed the step caveat.
- `... raw MSE ratio at h = ...` — "is a ratio move the forecast or the market drift added
  back?"
- `... mis-scaling cross term at h = ...` — "is a rising ratio an uninformative mean (cross
  term ≈ 0) or an over-amplitude one (cross term ≪ 0)?" The −0.052…−0.058 at h ≥ 64 identified
  over-amplitude conditional means from a single last-eval snapshot; it is now a trajectory.

## Verification

`cargo check -p trading_bot_0 --tests` and `cargo check -p trading-bot-tui --tests`: 0 errors.
`cargo test -p trading_bot_0 timexer_segment`: 55 passed (50 before, +3 mine in
`reports::tests`, +2 from the concurrent knobs work). `cargo test -p trading-bot-tui`: 36
passed, including the base-registry test with the new explicit both-direction assertions.

End to end, not just compiled. A throwaway `#[ignore]`d probe in `reports::tests` (since
removed) drove the real `reports::write_metrics` writer with the finished
`training/runs/timexer-recipe-v3-20260906/gens/1` evaluation's own numbers — per-step
`training NLL` / `held-out sample NLL` from its `timexer_segment_loss`, the per-horizon MSE and
persistence-MSE levels from its `timexer_segment_horizon_error`, and the cross term from its
`timexer_segment_decomposition` (its live snapshot is now step 5000) — plus that same run's
step-4000 snapshot transcribed from `timexer_recipe_v3_compare.md`, so the read-back shows two
real evaluations rather than a repeated constant. Steps 1k–3k carry no horizons: those
evaluations' curves were overwritten and are gone, which is what the new bases prevent going
forward. Written to `target/horizon-steps-probe/gens/1`, then read back with
`./target/release/report_cli 1 <base> --run-root target/horizon-steps-probe`:

```
== timexer_segment_generalization_gap
1000	training minus held-out sample NLL=0.24948788	no gap 0.0=0
2000	training minus held-out sample NLL=0.21750987	no gap 0.0=0
3000	training minus held-out sample NLL=0.16105533	no gap 0.0=0
4000	training minus held-out sample NLL=0.07608795	no gap 0.0=0
5000	training minus held-out sample NLL=0.0601995	no gap 0.0=0

== timexer_segment_horizon_steps        (h = 64 and h = 192 columns)
1000	... at h = 64=NaN                      ... at h = 192=NaN
4000	MN at h = 64=1.0146  raw=1.011          MN at h = 192=1.0329  raw=1.0281
5000	MN at h = 64=1.0443  raw=1.052021       MN at h = 192=1.0455  raw=1.0451653
   (h = 1 at 4k/5k: MN 0.9529 / 0.96013457; h = 8: MN 0.894 / 0.92244625; parity 1.0=1)

== timexer_segment_horizon_steps_scaling
4000	cross term h = 1=-0.0127  h = 8=-0.001         h = 64=-0.056       h = 192=-0.058
5000	cross term h = 1=-0.012722889  h = 8=-0.0014148682  h = 64=-0.055893444  h = 192=-0.05789496
```

The gap column is fully real per-step data and reproduces the reported 0.25 → 0.08 collapse
(0.2495 → 0.0761 at 4k, 0.0602 at 5k). What was NOT exercised: the CUDA evaluation that fills
`HorizonCurve`/`TradingCurve` (job 5188 held the queue lease), so the numbers came from that
evaluation's own persisted reports rather than from a fresh forward pass; the writer, the
`horizon_track` slice, the registry and the file format are exercised for real.
