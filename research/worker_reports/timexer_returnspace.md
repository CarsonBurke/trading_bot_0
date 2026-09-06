# ReturnSpace: persistence-anchored σ-scaled log-return objective for timexer_segment

## Changed files (main tree)
- `trading_bots/src/torch/timexer_segment/corpus.rs` — `fill_row` computes in FP64 `c_last`, `σ = sqrt(var(log close returns) + 1e-8)` (`RETURN_VARIANCE_FLOOR`, one basis point per bar), mean relative range; inputs and targets are `ln(p / c_last) / σ`; packed row = `(context+pred_len)*4 + aux + pred_len mask + 3 geometry floats`; `Batch` loses `price_scaling`; `geometry_context` is `[batch, 3] = [c_last, σ, mean_rel_range]`; schema bumped to `timexer-pooled-mmap-v2`; `filtered_contract` called directly; tests rewritten for the new layout.
- `trading_bots/src/torch/timexer_segment/data.rs` — deleted the `Dataset` struct, per-ticker train-price scaler (`means`/`stds`/`scaler_fit_bars`), `denormalize`/`price_errors`, and their tests. Kept `DataContract` (now `train_end` instead of `scaler_fit_bars`), `filtered_contract`, `retained_partition_end`, `valid_ohlc`, fingerprinting. Schema `timexer-segment-ohlc-v2`.
- `trading_bots/src/torch/timexer_segment/model.rs` — removed `normalize` (RevIN), `DecoderKind`, legacy decoder, `geometry_mix: Option`, head-output dropout. `Forecast { scaled, prices }` with no branch. `decode_joint(coordinates, geometry_context, horizon_scale)`: `close = c_last·exp(σ√h·c0)`, softplus range on mean relative range, sigmoid positions; log-price clamped before `exp`. Tests rewritten (zero coordinates → persistence candle, √h units, CUDA fresh-model-trains-every-branch).
- `trading_bots/src/torch/timexer_segment/compute.rs` — `Engine::forward_loss`/`step` drop `price_scaling`; `masked_mse` unchanged (`/(4·valid_bars)`), applied to `forecast.scaled` vs σ-scaled targets.
- `trading_bots/src/torch/timexer_segment/runner.rs` — `FORMAT = timexer-ohlc-universe-v4`, `OBJECTIVE = logreturn_sigma_mse_v1`, `Manifest::read` rejects other objectives with a message naming both. `score` computes model MSE, flat-persistence MSE (`targets²`), USD RMSE/MAE from decoded prices vs `c_last·exp(σ·target)`, invalid-candle fraction, `tail_loss_share`, and full 192-step `HorizonCurve`. `--decoder`, `--project-candles`, `projected_*` removed from args/metrics/evaluate. Manifest test covers objective rejection.
- `trading_bots/src/torch/timexer_segment/reports.rs` — `Metrics` gains `tail_loss_share` (validated in `[0,1]`, plotted in `timexer_segment_progress`), loses `projected_*`; new `HorizonCurve` + `write_horizon` writing `timexer_segment_horizon.report.bin` (model/persistence/ratio for preview and full validation as `IndexedLines` over horizon 1..192).
- `trading_bots/src/torch/timexer_segment/benchmark.rs` — synthetic inputs built as σ-scaled log history with `[c_last, σ, range]` geometry; `verify_optimizer` call sites updated.
- `trading_bots/src/torch/timexer_segment/geometry.rs` — deleted (post-hoc weighted projection); `mod.rs` entry removed.
- `shared/src/report.rs` — `timexer_segment_horizon` added to `TIMEXER_SEGMENT_REPORT_BASES`.
- `tui/src/main.rs` — registry test asserts `timexer_segment_horizon` is scanned by `meta_chart_bases` (bases extend from `TIMEXER_SEGMENT_REPORT_BASES`).
- `docs/timexer_segment.md` — input representation, decoder, loss/objective, TUI report list, manifest paragraphs rewritten; legacy/projection paragraph deleted.

## Decisions the task left open
- **No extra per-window centering.** With `c_last` as the zero point, any additional centering would move the encoder's origin off the last close and reintroduce the level anchor the review identified as the main bias; the √h-normalized close coordinate already keeps outputs O(1).
- **√h horizon scaling kept.** Under a random-walk baseline the h-step log-return std is σ√h; dividing the close coordinate by that keeps the head's required output magnitude flat across horizons (coordinates O(1)), and zero still decodes to persistence. It is a parameterization choice only; the loss is still measured in one-bar σ units so the horizon report is directly comparable to persistence.
- **Persistence reference = flat candle** (`targets²`, all four channels at `c_last`), the classical baseline; a zero-coordinate joint decode is not identical to it (high/low sit half a mean range around `c_last`), which is intentional and is the same range anchoring as before.
- **USD RMSE/MAE kept** as the only price-level metric; computed from decoded prices, so the per-ticker scaler had no remaining consumer and was deleted entirely.
- **σ floor** expressed as a variance floor `1e-8` (σ ≥ 1e-4 per bar, ≈1 bp), mirroring the old `+1e-5` variance floor in return units.
- `tail_loss_share` is computed per evaluation batch via `topk` on `|target|·mask` (top 1% of valid target elements), accumulated as a sum ratio over the evaluation.

## Verification
- `./torch-env.sh cargo check -p trading_bot_0 --tests`: zero `^error` lines, zero warnings in `timexer_segment`. `cargo check --manifest-path tui/Cargo.toml --tests` and `-p shared --tests`: clean. (`LIBTORCH_BYPASS_VERSION_CHECK` is rejected by torch-sys build script; `torch-env.sh` is the supported entry.)
- `cargo test -p trading_bot_0 --lib timexer_segment`: 24 passed (includes the CUDA fresh-model test).
- Throwaway CPU check (deleted afterwards, CUDA assert temporarily relaxed via env var and restored): fresh model with `head*`/`geometry_mix*` zeroed, batch 3, `c_last = [100, 25, 3.5]`, `σ = [0.002, 0.01, 0.0005]`, range `[0.004, 0.02, 0.001]`, 8 horizons:
  - open/close vs `c_last`: max relative error `0e0` (exact) at every horizon.
  - high/low vs closed-form `c_last/(1+r/2)` and `low·(1+r)`: max relative error `7.6e-8`.
  - scaled OHLC at every horizon (batch 0): `[0, 0.99701, -0.99900, 0]` repeated ×8.
  - `masked_mse(forecast.scaled, target)` = `6.169862747`; independently computed MSE of the zero-coordinate candle = `6.169880904` (diff `1.8e-5`, fp32 accumulation vs fp64); flat-persistence MSE (`targets²`, the `score()` reference) = `5.445722558`.
- `grep timexer_segment_horizon`: `reports.rs:221`, `shared/src/report.rs:11`, `tui/src/main.rs:1193`.
