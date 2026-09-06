# TimeXer evaluation speed and step-time audit

## 1. The "810 s" premise is a unit misread

`timexer_segment_timing` is in **milliseconds** (chart `y_label = "milliseconds"`, `Evaluation.elapsed_ms`). Report lines (report_cli):

| run | step | training step | evaluation | loader wait |
|---|---|---|---|---|
| timexer-market-neutral-20260905 | 1000 | 279.9 ms | **810.3 ms** | 0.070 ms |
| timexer-causal-20260905b | 1000 | 250.8 ms | 674.3 ms | 0.045 ms |
| timexer-causal-20260905b | 2000..9000 | 247–267 ms | 554–1225 ms | ≈0.001 ms |
| timexer-causal-20260905b | 9591 (full validation, 433,303 windows) | 234.9 ms (591-step partial interval) | 103,368 ms | |

A preview costs 0.81 s per 1000 steps × 280 ms = 280 s of training, i.e. **0.3 % of wall clock**. Hardware report means (whole run): GPU busy 99.8 % / 99.9 %, power 71.4 % / 77.0 % (market-neutral / causal-b). The GPU is not idling on evaluation; the lower power comes with a slower step (see §4), not with idle time. Both trainings ran alone on the GPU (mlq attempts 3783 19:39:07–19:47:23, 3766 18:22:28–19:05:49; the neighbouring hl_gauss benchmarks had finished before each start).

## 2. Before profile (what 810 ms was made of)

Code state at the market-neutral run: `score` already used tensor reductions on device, `no_grad`, the training batch size (256 → 8 batches), the one-ahead `Prefetcher`, bf16 tokens, and the `last_only` head (backbone over all 375 tokens, head on the final token). Two things were host-bound:

- RobustEval's `bar_squared/bar_persistence/bar_close` were **CPU** tensors written with `.copy_()` from device every batch: a device sync per batch, so the CPU could not enqueue batch i+1 (H2D 76 MB + forward) while the GPU ran batch i. The causal-b previews before those metrics were 554–674 ms; after them 810 ms (+140–250 ms ≈ 8 × (sync + H2D + launch gaps)).
- The first host batch is built with nothing to overlap.

CPU-measured components (release build, this workstation, real corpus of 4,873 tickers, 2,455,276 train / 433,303 validation refs; corpus load 71.8 s):

| component | measurement |
|---|---|
| `host_batch` 256 preview rows (2,048 fixed windows span ~2,048 tickers) | 68.3 / 38.7 / 45.6 ms (first call page-cache cold) |
| `host_batch` 256 consecutive train rows (one ticker) | 19.6 / 16.3 / 19.2 ms |
| all 8 preview batches sequential | 465 ms |
| metric accumulation, synthetic 2048×192×4 on CPU, 8 batches of 256 | accumulate 51–59 ms total (6.4–7.4 ms per batch), finish (window medians, per-horizon top-k thresholds, trimmed sums) 2.0–7.4 ms |
| forward, from the synthetic GPU benchmark (`timexer-hardware-20260905` gens 4–6, dense head + loss, train mode) | 50–66 ms per 256 rows [INFERENCE for the eval forward: last-only head is cheaper] |
| full validation 103.4 s / 1693 batches | 61 ms per batch → forward-bound (ticker-sorted refs build in ~20 ms) |

So per preview batch ≈ max(loader ≈ 45 ms, forward ≈ 55 ms) + sync/H2D gap ≈ 100 ms → 8 × 100 ms ≈ 810 ms. Metric accumulation is ≤ 7 ms/batch even on CPU and is launch-bound on the GPU; candle rendering and report writing were never inside the timed `score` (they run after it, untimed until now).

## 3. Changes

- `runner.rs`: `Scorer` (device-resident accumulators: 12 aggregate sums, 11 per-horizon rows, `[origins, pred_len]` per-bar close-channel sums for the trimmed ratio and window medians). `accumulate` is the former loop body; the per-bar tensors now live on `device`, the trimmed thresholds (`topk`/`gather` per horizon), `keep` masks, trimmed sums, window medians all finish on device, and `finish` does the single host transfer. `score` times three synchronized phases (host loader wait, H2D + forward, metric accumulation) plus total; `synchronized()` only syncs on CUDA so the scorer is unit-testable on CPU. `tail_count` is clamped to the batch's element count.
- `train()`: candle windows and checkpoint saves are timed as `reports_ms` (moved before the report writes so the value lands on the same point); `write_metrics` unchanged in ordering otherwise.
- `reports.rs`: `Metrics.eval: EvalTiming { total_ms, loader_ms, forward_ms, metrics_ms, reports_ms }` replaces `eval_ms`; `timexer_segment_timing` now carries: training step, host loader wait per training step (label states it is the mean per step and overlaps GPU work), evaluation, evaluation: host loader wait, evaluation: H2D + forward, evaluation: metric accumulation, after evaluation: candle windows + checkpoints. Same base name → no `meta_chart_bases` change.
- `docs/timexer_segment.md`: timing chart bullet.
- Not changed because already in place: eval batch size = training batch size, `Prefetcher` one batch ahead with the rayon gather pool, `no_grad`, bf16 backbone/heads, `forward(.., last_only = true)` (head on the final token only; `future_windows` narrows the known covariates likewise).

Test `tensor_scorer_matches_scalar_reference` (runner.rs): 13 windows × 7 bars × 4 channels over batches of 5/5/3, random mask (one all-invalid window, one flat close forecast, one zero close target); scalar f64 reference for MSE, persistence MSE, NLL, ±1σ coverage, per-horizon MSE/persistence, MAE ratio, win rate, hit rate (flat forecasts excluded), up fraction, trimmed MSE ratio (k_h = 1), median window ratio (lower median over scored windows), tail loss share (k = ⌈n·4/100⌉ ≥ 2 in the 5-row batches). All match to 1e-5 relative (fp32 elementwise, f64 sums).

## 4. Training step 235 → 280 ms

- The "235" is causal-b's last partial interval (591 steps); its first interval was 250.8 ms, so the like-for-like delta at step 1000 is +29 ms (12 %), within ~2× the interval-to-interval spread seen in causal-b (235–267 ms).
- Host loader: `loader_wait_ms` is the **mean per-step** wait (`Σ wait / interval_steps`), 0.07 ms → the loader is always ahead. Measured `host_batch` is 16–68 ms per 256 rows against a 250–280 ms step; the extra `market_cum` block is one O(1) `MarketPath::at` per bar and +6 % row width (114 MB pinned H2D per step, async from pinned memory). The host side cannot be the cause.
- Extra GPU arithmetic: `market_drift` at all 375 origins is an unfold view of `[256, 6192]`, two `[256, 375, 192]` elementwise ops and one broadcast subtract over the `[256, 375, 192, 4]` targets (~300 MB per pass): ≈ 1–3 ms per step, no backward through targets. Not 29 ms.
- What remains is only measurable on the GPU (no GPU runs permitted here): `benchmark-timexer-segment --profile --batch-size 256 --optimizer polar-express` on the current binary (its synthetic rows already carry the `market_cum` block) versus the 152–204 ms of `timexer-hardware-20260905` gens 5–7 (older binary) would separate kernel time from the real-data path; with `nvidia-smi -q -d CLOCK,PERFORMANCE` during the step to check for clock/thermal capping, which is the pattern that matches "slower and lower power at 99.8 % busy".

## 5. After timing of the metric path (CPU, release)

Synthetic 2048 × 192 × 4, 8 batches of 256, three repeats: total 67.7 / 58.3 / 53.3 ms (accumulate 59.0 / 55.7 / 51.0 ms, finish 7.4 / 2.1 / 2.0 ms). On the GPU the same ops are launch-bound (~120 small kernels per batch). The two syncs per batch added for attribution cost the launch/H2D gap they measure (≤ ~10 ms per batch, ≤ 0.03 % of the interval); they are what makes the per-phase series in `timexer_segment_timing` trustworthy.

## Verification

`./torch-env.sh cargo check -p trading_bot_0 --tests`: zero errors. `cargo test -p trading_bot_0 timexer_segment`: 30 passed, 0 failed (1 ignored: MarketSeriesFix's `diagnose_market_steps`).
