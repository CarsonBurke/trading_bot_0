# Market-neutral CausalPatch cutover

## Files changed
- `trading_bots/src/torch/timexer_segment/features.rs`: `MarketPath` + `market_path()` (cumulative equal-weighted market log return, gaps included, only adjacent-populated-slot pairs; unpopulated slots forward-filled; f64 storage); shared `accumulate()` helper reused by `market_series`; `Exogenous.market_cum` (non-optional). Test `market_path_includes_gaps_and_only_adjacent_populated_slots` (gap, ticker skipping a bar, invalid bar unpopulating a slot).
- `corpus.rs`: SCHEMA `timexer-pooled-mmap-v4;…;market-cumulative-log-return-per-bar-…`; `CorpusContract.market_fingerprint` (SHA-256 over ordered `ticker:fingerprint` + boundary timestamps); packed row `L*(6+n_aux)+1` with `market_cum [L]` after aux (value at bar ts minus value at origin ts); `Batch.market_cum [B,L]`; market path always built (independent of `--features`). Tests updated (one-ticker universe ⇒ row market_cum == own close path; masked horizon = 0; fingerprint length).
- `model.rs`: `Statistics.market` (market_cum at each origin), `market_drift()` = `(mc[t_k+h]-mc[t_k])/σ_k` `[B,origins',H,1]`, `targets()` subtracts it. σ stays the ticker's causal σ (documented). Test `a_ticker_tracking_the_market_has_zero_close_target`; synthetic batches carry a random market path.
- `benchmark.rs`: synthetic packed rows gain the zero `market_cum` block.
- `runner.rs`: OBJECTIVE `causal_patch_market_neutral_nll_v1`; `--preview-patience` (default 3; counting starts after the first two previews); `weights/best` → `preview-best` whenever preview NLL improves (epoch fallback only if no preview ever ran); manifest `best_step`/`best_preview_nll`; `--patience` now only stops on stale full-validation NLL. `FinalOrigin.drift`/`rebased_prices`; observed prices use absolute targets; `score` adds absolute-space MSE + persistence (aggregate + per horizon, sums[10..12], horizon rows 9..11); RobustEval's robust series untouched, relative space only. Stop log line prints triggering NLL, best NLL, best step.
- `reports.rs`: `Metrics`/`HorizonCurve` `absolute_mse`, `absolute_persistence_mse`; validation chart series renamed `… relative MSE`; new base `timexer_segment_absolute` (relative/absolute ratios + absolute MSE/persistence); horizon chart emits `{preview|full-validation} {relative|absolute} forecast/persistence/ratio`; candle title states re-basing onto the realized market path.
- `shared/src/report.rs` (`TIMEXER_SEGMENT_REPORT_BASES` += `timexer_segment_absolute`), `tui/src/main.rs` (registration assertion), `docs/timexer_segment.md`.

## Verification
- `cargo check -p trading_bot_0 --tests` and `cargo check --manifest-path tui/Cargo.toml --tests`: zero errors. `cargo test -p trading_bot_0 timexer_segment`: 28 passed. TUI `the_meta_chart_list_looks_for_every_registered_writer_base`: passed.
- Throwaway CPU check (deleted), B=2, seq 6000 / pred 192 / features all, fresh zero head: relative NLL 5.655063 vs baseline 5.655934 (close channel 5.652106 == 5.652106); relative MSE ratio close 1.000000 (all channels 0.999914); absolute MSE ratio close 1.000000 (all channels 0.999697); drift at h=192 row 0 = 7.670515 == (mc[t+192]-mc[t])/σ; rebased/absolute close price ratio 1.015686. The all-channel ratios are <1 because the zero head decodes the persistence *candle* (nonzero high/low offsets from ρ_k) while the persistence baseline is a zero forecast on every channel — pre-existing convention, unchanged.

## Train CLI
```
mlq submit --name timexer-market-neutral --max-parallel-runs 1 --time-limit 12h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment \
  --run timexer-market-neutral --features all --preview-patience 3
```
(`--eval-every 1000 --eval-origins 2048 --patience 3 --epochs 1` are the defaults.)

## Open decisions
- Old checkpoints (`causal_patch_nll_v1`, schema v3) are rejected; RobustEval evaluates the old run from a pre-edit snapshot at `/tmp/timexer-pre-market-neutral/` (byte-identical originals of the four files I changed first).
- Robust series (MAE ratio, win/hit rates, trimmed MSE, median window ratio) are relative-space only; absolute space carries MSE/persistence only.
- `--patience` (epoch-level, full-validation NLL) retained as a second stopping rule; it no longer selects `weights/best`.
