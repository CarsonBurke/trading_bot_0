# TimeXer runs — empirical state (ResultsLens, 2026-09-05, read-only)

All numbers below were read with `target/release/report_cli <gen> <report> --run <name>` (postcard reports under `training/runs/<run>/gens/<gen>/`), checkpoint `manifest.json`, mlq attempt logs (`~/.local/state/mlqueue/attempts/<id>/`), and `mlq status` / `mlq show`. MSE values are on train-standardized OHLC levels (per-ticker scaler for universe runs; global AAPL scaler for the single-ticker run), averaged uniformly over all 192 future bars x 4 fields (runner.rs:349-351).

## Summary table

| run | context | horizon | decoder | volume | epochs (planned/done) | train MSE | preview MSE | full-val MSE | persistence MSE | ratio model/persist | invalid-candle % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| timexer-universe-joint-20260905 (job 5031, RUNNING, step 8000/9591) | 6000 | 192 | joint | no | 1 / 0 (83.4% of epoch 1) | cum 1.395 (last-1000-step window ~0.022) | 0.03835 | n/a yet | preview 0.02738 | **1.40** (preview) | 0.0 % |
| timexer-universe-pe-20260905 (job 5008, SIGTERM-cancelled at step 9000/9591) | 6000 | 192 | default (per-field heads) | no | 1 / 0 (93.8%) | cum 0.0749 (window ~0.033) | 0.03357 | 0.03634 (separate evaluate job 5030 on preview-latest ckpt, `--project-candles`) | preview 0.02738; full-val 0.02963 | **1.23** (preview) / **1.23** (full-val) | 12.9 % forecast; 0 % after projection |
| timexer-aapl-segment-mse-20260905 (job 4977, exit 0, early-stopped) | 96 | 192 | default | no | 10 / 4 (patience-3 stop) | 0.000797 (epoch-4 cumulative) | 0.003422 | 0.003428 (epoch 4); best epoch 1 = 0.003424 | preview 0.003432; full-val 0.003415 | **1.004** (full-val) / 0.997 (preview) | 22.4 % at epoch end (11-28 % across epoch 4) |
| timexer-hardware-20260905 (benchmarks only, gens 1-7) | 6000 | 192 | gen7 = joint, gen1 = default | no | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| timexer-aapl-10m-20260904 (legacy probabilistic single-ticker `timexer`, not `timexer_segment`; job timed out at 10 min after epoch 9/20) | n/a (legacy format) | 6 horizons | n/a | n/a | 20 / 9 | objective 4.178 (gen 9) | n/a (different reports) | n/a | n/a | n/a | n/a |
| timexer-aapl-20260904-2210 | n/a | n/a | n/a | n/a | 20 / 0 | gens/ EMPTY — produced nothing | | | | | |

Notes on columns:
- `train MSE` in `timexer_segment_validation` is the **cumulative epoch mean** `total_loss / target_bars` (runner.rs:512-551), not a window mean. Per-1000-step window means were derived as `k*C_k - (k-1)*C_{k-1}` (see below). [DERIVED]
- `preview MSE` = fixed 2048 held-out validation origins (universe) / 256 origins (AAPL) evaluated every 1000 steps (`eval_origins`, manifest).
- `full-val` = complete validation partition, only produced at epoch end (AAPL: 44,265 origins; universe pe: from `evaluate-timexer-segment`).
- `persistence` = last-observed-bar naive baseline computed by the same `score()` (runner.rs:351).

## Per-run detail

### timexer-universe-joint-20260905 (the `--decoder joint` model under review)
- Command (mlq job 5031, attempt 3752, 3h limit, launched from a frozen binary `training/launchers/timexer-joint-20260905/trading_bot_0`): `train-timexer-segment --decoder joint --optimizer polar-express --seq-len 6000 --common-context 6000 --batch-size 256 --epochs 1 --eval-every 1000 --eval-origins 2048`.
- Corpus: 4873 tickers, 470,946,393 training target bars, 2,455,276 segments, 9591 steps/epoch (stdout.log). Manifest format `timexer-ohlc-universe-v3`; model d_model 512, 8 heads, 2 layers, d_ff 2048, dropout 0.1, patch 16; 48.45M params (hardware gen7). LR at step 7000 = 0.00426 with nanogpt cooldown (frac .60, floor .15); muonLR .023 / adamwLR .008, muonWD 1.2 (manifest `optimizer_recipe`).
- Validation trace (step: cum-train / preview; persistence fixed 0.027384):
  - 1000: 10.965 / 0.04493
  - 2000: 5.502 / 0.04983
  - 3000: 3.677 / 0.04653
  - 4000: 2.764 / 0.04481
  - 5000: 2.219 / 0.05873  (spike)
  - 6000: 1.853 / 0.03854
  - 7000: 1.592 / 0.03880
  - 8000: 1.395 / 0.03835
- Derived per-window train MSE: w1 = 10.96 (!), w2 = 0.040, w3 = 0.028, w4 = 0.025, w5 = 0.037, w6 = 0.024, w7 = 0.023, w8 = 0.022. => the first 1000 steps had an enormous loss transient (mean ~11 in standardized units, ~400x the steady state); after that the window loss is 0.022-0.04. The reported cumulative `training=` line is therefore dominated by that transient and is not comparable to the pe run at face value. [DERIVED; verify by adding a windowed train loss to reports.rs]
- Preview/persistence ratio 1.40 at step 8000: the joint model is **40 % worse than naive persistence** on the held-out preview, and preview MSE is noisy/non-monotonic (0.045 -> 0.050 -> 0.047 -> 0.045 -> 0.059 -> 0.039 -> 0.039 -> 0.038).
- Invalid forecast candles = 0.0 at every step (joint decoder enforces OHLC geometry by construction).
- Timing: 154-159 ms/step, eval 350-1006 ms per 1000 steps, loader wait ~0.2 ms (GPU-bound). Full epoch ~25 min + evals; hardware gen7 benchmark: 152.3 ms/step, 1681 origins/s, peak allocator 18.4 GiB, peak reserved 21.2 GiB, Polar-Express graph captured.
- Status: still running as of this read (step 8000 of 9591); full validation + `weights/epoch-0001` will land in `gens/1` when the epoch completes.

### timexer-universe-pe-20260905 (same model, default decoder, Polar-Express)
- First attempt (job 4998, attempt 3719, `--epochs 10 --patience 3`, 12h limit) **panicked at corpus load**: `TimeXer segment training failed: invalid OHLC at bar 85378` (main.rs:1882, exit 101). The corpus schema was then changed to `invalid-ohlc-rows-quarantined-without-repair` (contract schema string) and the run re-submitted as job 5008 (attempt 3730) with `--epochs 1`, 3h limit.
- Second attempt was **SIGTERM-cancelled at step 9000/9591 (93.8 %)** (result.json term_signal 15, not a time-out) — it never reached the epoch-end full validation. Full-val numbers come from job 5030 `evaluate-timexer-segment --checkpoint weights/preview-latest --project-candles` written to `gens/0`.
- Validation trace (step: cum-train / preview; persistence 0.027384): 1000: 0.4077/0.03474; 2000: 0.2216/0.04468; 3000: 0.1592/0.03757; 4000: 0.1277/0.04355; 5000: 0.1087/0.04715; 6000: 0.0960/0.04731; 7000: 0.0870/0.03354; 8000: 0.0802/0.03740; 9000: 0.0749/0.03357.
- Derived per-window train MSE: w1 0.408, then 0.0355, 0.0344, 0.0334, 0.0328, 0.0322, 0.0331, 0.0325, 0.0328 — **flat at ~0.033 from step 1000 onward**: essentially no further training progress during the epoch (plateau). Train window ~0.033 vs preview ~0.034-0.047: no train/val gap, i.e. the model is not overfitting, it is under-fitting relative to persistence. [DERIVED]
- Full validation (gens/0, evaluate job): forecast MSE 0.036337, persistence 0.029631, weighted projection 0.036335 -> ratio **1.226**. Invalid forecast candles 12.9 %; invalid projected candles 0 (projection repairs geometry but changes MSE by only 2e-6).
- LR 0.0003 (AdamW) with hidden Muon LR = 50/3 x that; `epoch-type1` schedule; all weight decay 0.
- Timing: 174-199 ms/step (slower than joint's 154 ms), eval 480-1012 ms.

### timexer-aapl-segment-mse-20260905 (single ticker AAPL, context 96)
- Job 4977 (`train-timexer-segment --ticker AAPL`, defaults: seq-len 96, batch 32, lr 2.5e-5, Adam, 10 epochs, patience 3), exit 0, finished ~2h before this read.
- Full-validation forecast MSE per epoch (44,265 origins): e1 0.0034237, e2 0.0034485, e3 0.0034251, e4 0.0034278; full-val persistence 0.0034146. Best = epoch 1; 3 non-improving epochs -> `TimeXer stopped after 3 complete epochs without improved full-validation MSE` (runner.rs:631-633). **Culled by patience.**
- Ratio model/persistence (full-val) = 1.0039 — the model never beat persistence; preview ratio ~0.997 within noise.
- Price-space error at epoch 4 full-val: RMSE 3.089 USD, MAE 1.920 USD over the 192-bar horizon.
- Cumulative train MSE at epoch 4 ≈ 0.00080 vs val 0.0034: a 4.3x gap, but persistence on validation is also 0.0034, so this reflects a lower-volatility training regime (2020-Sep 2023 vs Sep 2023-Sep 2024) under a global scaler, not memorisation. [INFERENCE; check by computing persistence MSE on the train partition]
- Invalid predicted OHLC: 11-28 % across epoch 4 (22.4 % at the end). Direct per-field heads routinely violate high>=max(o,c), low<=min(o,c).

### timexer-hardware-20260905
- Benchmark-only run dir (gens 1-7 from `benchmark-timexer-segment` jobs). gen1 (default decoder, batch 64): 39.8 ms/step, 1609 origins/s, 5.1 GiB. gen7 (job 5029 `--decoder joint --optimizer polar-express --seq-len 6000 --batch-size 256`): 152.3 ms/step, 1681 origins/s, 18.4 GiB alloc / 21.2 GiB reserved, 48,454,356 params.

### Legacy runs (older `timexer` probabilistic single-ticker model, different report set)
- timexer-aapl-10m-20260904 (job attempt 3682, 20 epochs planned, **timed out** at the 600 s mlq limit after epoch 9). Gen 9: direction Brier by horizon 0.252, 0.260, 0.296, 0.328, 0.362, 0.369 (coin-flip is 0.25 -> worse than chance beyond h0); return RMSE 0.00158 -> 0.0181; interval coverage collapses with horizon (90 % nominal: 0.887 at h0 -> 0.477 at h5; 50 % nominal: 0.503 -> 0.204) -> severely under-dispersed at longer horizons.
- timexer-aapl-20260904-2210: `gens/` empty; no reports produced.

## Campaign / queue status (mlq, read-only)
- `mlq status`: 1 active lease (limit 1). Running: **5031 timexer-joint-universe-one-epoch** (attempt 3752). Queued behind it: 5034/5035/5036 (cleanrl PPO HalfCheetah jobs, unrelated). No `timexer-universe-20260905-{c2048,c6000,c6000-volume,final}` jobs exist in status or in any attempt's command.json.
- `benchmarks/timexer_universe_campaign.py` defaults to reference job **4987** (`timexer-universe-context96`, run `timexer-universe-c96-20260905`): `mlq show 4987` -> **cancelled before start, 0/1 attempts**. Hence the campaign chain (`--after-success 4987`) was never submitted/run; `read_evidence()` would also fail today because no run has `weights/epoch-0001` with `validation_is_full: true` except the AAPL run (different contract).
- `benchmarks/timexer_campaign.py` is the older frozen three-seed single-ticker comparison (single-ticker-timexer, patch-tst, d-linear, n-linear, bar-trunk, raw-timexer) writing to `training/evaluations/<campaign>`; no timexer entries in `benchmark_results/` (grep empty; latest file there is 2026-06-05).
- `training/training.log` tail is a `train-planner` (world-model PPO) run, not TimeXer; TimeXer stdout lives in the mlq attempt logs.

## Divergence / plateau / cull summary
- **Diverged-then-recovered**: joint run first 1000 steps (window mean loss ~11 vs ~0.02 steady state). Not a crash, but a red flag for the joint decoder init / Muon LR .023 warm-up. Resolve: log windowed train loss; try lower peak LR or longer warm-up and compare preview at step 2000.
- **Plateau**: pe run — windowed train loss flat at 0.033 from step 1000 to 9000; preview never trends below ~0.0335 (persistence 0.0274). Joint run preview also flat ~0.038 from step 6000.
- **Culled**: AAPL context-96 run stopped by patience after 4 epochs, best full-val at epoch 1, ratio to persistence 1.004.
- **Killed**: pe run cancelled by request at 93.8 % (no epoch-end checkpoint/full-val); AAPL-10m legacy run timed out at 10 min limit.
- **Failed**: pe first attempt panicked on invalid OHLC bar; timexer-aapl-20260904-2210 produced nothing.

## Bottom line (demonstrated)
No TimeXer configuration tried so far beats the naive persistence baseline on held-out data at the 192-bar (16 trading-hour) horizon: AAPL c96 = 1.004x persistence (full validation); universe default decoder = 1.23x (full validation, 4873 tickers); universe joint decoder = 1.40x (2048-origin preview, epoch 83 % done). The joint decoder eliminates invalid candles (0 % vs 12.9 % / 22 %) but at the cost of higher MSE so far; whether it closes the gap by epoch end is pending (job 5031).

## Hypotheses and the experiment that resolves each
1. The 1.2-1.4x-persistence gap is an optimisation/objective problem (uniform level-MSE over 192 bars with a model that lacks a persistence skip) rather than data noise: **experiment** = add a residual-to-last-close head (predict deltas from last close, zero-init) and check preview ratio < 1.0 within the first 2000 steps.
2. Preview MSE noise (+-0.01 swings between 1000-step evals) hides real progress: **experiment** = raise `--eval-origins` to >=16k or report a moving average; compute the standard error of preview MSE across origins.
3. pe plateau at 0.033 is LR-limited (AdamW 3e-4 with no decay, hidden Muon LR 5e-3): **experiment** = LR sweep {1e-4, 3e-4, 1e-3} at 2000 steps on a fixed 10 % corpus shard.
4. Joint run's initial loss ~11 stems from the joint-decoder parameterisation at init: **experiment** = log loss at step 0/10/100 and per-field breakdown; compare with `--decoder` default under identical LR.
