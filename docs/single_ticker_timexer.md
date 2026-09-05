# SingleTickerTimeXer v1

This probabilistic forecaster remains available through `train-timexer`. The default
CLI and TUI training path is now [TimeXer OHLC Segment](timexer_segment.md). The
existing `pretrain`, `pretrain-mse-jepa`, PPO, and planner implementations also remain
explicitly available. Neither experiment has established an accuracy promotion.

The model reads one configured `<TICKER>.300.bars` file. It never opens a universe,
market-support artifact, proxy ticker, or cross-sectional dataset. Exactly 2,048
completed observed bars form each context. Targets are log(close[t+h]/close[t]) at
h = 1, 4, 16, 39, 78, 100, divided by sigma[t] times sqrt(h). Sigma is a causal
EWMA second moment with a 78-bar half-life and 256-bar initialization period;
sigma[i] excludes return[i]. No historical mean is added to these target returns.

The chronological partitions are 70% training, 10% calibration, 10% validation,
and 10% terminal test. The last 100 forecast origins before each boundary are
purged, so no target interval crosses the boundary. Historical contexts can reach
back into an earlier partition; labels cannot. The checkpoint pins the original
corpus, exact split instants, support fit, and complete preprocessing contract.

Future clock features follow the nominal US extended-session schedule, 04:00–20:00
New York time with the repository's full-day exchange holidays. They never read
future observed timestamps. Horizons count the next completed observed bars;
a halt or missing bar can make their realized clock differ from the nominal
clock. Origins are not filtered by future activity. Historical elapsed-time and
session features expose gaps to the model. Context is measured in observed bars,
so its calendar duration depends on that ticker's extended-session activity.

The endogenous patch regions, oldest first, are 1,408 bars in length-32 patches,
512 bars in length-8 patches, and the most recent 128 individual bars. Their 236
tokens receive elapsed-position embeddings and one learned global token. Four
pre-norm blocks use width 256, eight heads, and FFN width 1,024. Only the global
query cross-attends to the twelve identified, validity-aware variate tokens.
Six learned horizon queries combine future clocks and the global representation,
then attend to the final patches to emit six 128-bin categorical distributions.
CUDA uses BF16 activations, FP32 master parameters and AdamW states, and fused
attention. Every optimization step consumes one complete batch.

Supports are fitted separately per horizon from training labels only. Quantile
cuts define equal-mass bins as closely as tied returns allow. The two tails are
open exponential laws, and interior laws interpolate fitted conditional quantiles.
The manifest includes their fitted means, second moments, and distribution-law
parameters. Hard categorical NLL is the only training objective. Globally
normalized mean sample-uniqueness weights account for overlapping 100-bar labels.
CRPS and all calibration diagnostics use those same predicted probabilities and
frozen within-bin laws.

Reports distinguish standardized CRPS and standardized error from RMSE/MAE in
cumulative-log-return units. Direction Brier, randomized categorical PIT TV,
50/80/90% interval coverage, both 5% tails, batch-one p95 latency, and peak CUDA
allocator memory use `.report.bin`. Full-precision paired observations and
provenance live in the report's Evidence payload, alongside chartable series.
The TUI registers every new report base; `report_cli` reads the same files.

## Commands

Set the desired ticker explicitly. The following uses AAPL as an example, not an
implicit default:

```bash
export TIMEXER_TICKER=AAPL
mlq submit --name timexer-aapl --max-parallel-runs 1 --time-limit 48h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer \
  --ticker AAPL --run timexer-aapl
```

This command requires an explicit ticker or `TIMEXER_TICKER`. A missing ticker
fails before opening data; it never silently picks a symbol. Its defaults are 20 full epochs,
batch size 32, AdamW learning rate 0.0003, and initialization seed 20260904.

Training selects the best validation checkpoint. After the complete selection
protocol finishes, `weights/best` points to an immutable selected checkpoint.
Freezing creates a separate authenticated copy and is required for terminal-test
access and direct inference:

```bash
./trading_bots/run-release-cuda.sh freeze-timexer \
  --checkpoint training/runs/timexer-aapl/weights/best \
  --output training/runs/timexer-aapl/weights/frozen
mlq submit --name timexer-aapl-validation --max-parallel-runs 1 --time-limit 4h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh evaluate-timexer \
  --checkpoint training/runs/timexer-aapl/weights/frozen \
  --split validation --statistical-baselines \
  --output training/evaluations/timexer-aapl
mlq submit --name timexer-aapl-forecast --max-parallel-runs 1 --time-limit 30m \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh forecast-timexer \
  --checkpoint training/runs/timexer-aapl/weights/frozen \
  --output training/evaluations/timexer-aapl-latest
```

Evaluation requires the exact frozen corpus. Latest-origin forecasting permits
appended bars from the same ticker after verifying that the entire original
prefix is unchanged, retaining its split and supports. Forecasts are independent
marginals, not a joint price path or a portfolio/policy input.

## Comparisons and promotion

The campaign runs initialization seeds 20260904, 20260905, and 20260906 with
identical ticker, corpus, origins, partitions, supports, batch size, learning rate,
and epoch budget. It trains SingleTickerTimeXer, categorical DLinear and NLinear,
probabilistic PatchTST, the actual causal BarTrunk with a private same-ticker input
projection and cumulative-return heads, and raw TimeXer MSE. It additionally
scores the train-fitted unconditional marginal with Jeffreys count smoothing and
a causal HAR-volatility Student-t baseline. Raw TimeXer only contributes point
error diagnostics, never probabilistic promotion scores.

```bash
mlq submit --name timexer-comparisons --max-parallel-runs 1 --time-limit 336h \
  --cwd "$PWD" -- python3 benchmarks/timexer_campaign.py \
  --ticker AAPL --campaign timexer-aapl-v1
```

The long limit covers eighteen complete training runs plus separate validation
and calibration evaluations. The campaign fails on an unsuccessful command and
never opens the terminal test. It preserves every model's reports. No automatic
culling or reduced training budget is used as evidence for the three-seed gates.

`compare-timexer` requires three frozen candidate reports, three compliant
BarTrunk reports, and three matched PatchTST reports. It uses 4,000 paired
calendar-week bootstrap draws, shared across seeds, with at least eight weeks.
It enforces the proposed NLL, horizon regression, CRPS, raw-return RMSE, Brier,
PIT, interval coverage, tail, seed-regression, latency, and memory thresholds.
The bridge must also perform at least as well as PatchTST on mean NLL and CRPS.
Missing or mismatched exposure/provenance fails closed. Gate decisions are
reported, not silently used to rewrite the default or overwrite older models.

An explicit later `evaluate-timexer --split test` invocation is possible only
with a frozen checkpoint. Freeze the architecture, checkpoint selection policy,
and comparison protocol before using that terminal evidence.

## Performance verification

`benchmark-timexer` compares the original batch builder and synchronous objective
check against cached gathering and asynchronous finite checks, using the same
BF16 model and FP32 optimizer in both paths. It runs three paired trials with
alternating order, checks exact batch values and final parameter agreement, and
reports batch time, full training-step time, throughput, cache preparation, and
peak allocated CUDA memory in `timexer_performance.report.bin`. These are hardware
measurements, not reduced-training accuracy evidence; no checkpoint is saved.

```bash
mlq submit --name timexer-aapl-performance --max-parallel-runs 1 --time-limit 30m \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh benchmark-timexer \
  --ticker AAPL --output training/benchmarks/timexer-aapl-performance
```

Training and evaluation upload compact histories once and gather only the requested
2,048-bar windows. The cache costs about 268 bytes per source bar (roughly 114 MiB
for the current AAPL corpus), without materializing overlapping windows or caching
terminal-test labels. Benchmark peaks include this cache in both paths; its
resident cost is reported separately. Future clocks reuse a deterministic calendar
schedule, preserving gaps, holidays, and DST behavior. Training retains one
optimizer update per batch and checks objective finiteness asynchronously on CUDA;
only the epoch's reported objective transfers back to the host. Validation uses
support boundaries and fitted moments to avoid redundant interior-bin integration.

## Candle charts

Each probabilistic training epoch emits `timexer_candles.report.bin` and four
fixed validation windows under `candle_snapshots/`. Charts show actual OHLC bars,
a marked forecast origin, and independent median/90% close-price intervals at
horizons 1, 4, 16, 39, 78, and 100. They do not interpolate a predicted candle path.
The displayed recent history is 32 bars; model input remains the full 2,048 bars.

Existing checkpoints can produce the same charts without retraining or opening
the terminal test:

```bash
mlq submit --name timexer-candles --max-parallel-runs 1 --time-limit 10m \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh timexer-candles \
  --checkpoint training/runs/timexer-aapl-10m-20260904/weights/epoch-0002 \
  --output training/runs/timexer-aapl-10m-20260904/gens/2
```

Use the TUI's `render` command on the resulting report files to export PNGs.
