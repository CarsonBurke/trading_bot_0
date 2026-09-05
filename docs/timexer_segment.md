# TimeXer full-segment forecasting

This separate default model follows TimeXer’s deterministic, direct-segment MSE objective. Existing categorical, LeJEPA, and other models remain available.

Each row contains one ticker’s 6,000 completed five-minute OHLC bars and predicts all four OHLC values for each of the next 192 bars simultaneously. Training pools the entire eligible ticker universe, while every history, target, scaler, and attention operation stays within its row’s ticker. There is no cross-ticker attention or market-proxy feature. `--ticker` optionally selects an explicit comma-separated subset; the default requires no ticker environment variable.

The encoder uses uniform, non-overlapping 16-bar patches: 375 patches plus a learned global token per OHLC channel. Width is 512, with eight heads, two encoder layers, a 2,048-wide feedforward network, GELU, and 0.1 dropout. Global tokens read historical variate tokens, and a shared dense head emits the complete future segment. Train-only per-ticker standardization and per-window mean/variance normalization retain the reference level-forecasting behavior. Optional `--volume-features` adds the same ticker’s historical log-volume innovation and validity as auxiliary variates, without making volume a target.

The default `--decoder joint` maps four learned coordinates into one positive close, a nonnegative relative range, and bounded open/close positions. It uses the existing candle geometry with a small, randomly initialized joint projection. The close retains the historical-mean anchor; there is no persistence residual or zero-initialized prediction head. Raw historical close/range statistics are accumulated directly from valid source bars to avoid cancellation when reconstructing penny prices from globally standardized values. Geometry uses FP64 intermediates and returns authoritative FP32 prices; the encoder remains BF16. Training still minimizes the original standardized OHLC MSE after decoding.

`--decoder legacy` retains the unconstrained output for reproducible comparisons. Historical checkpoints without a decoder field load that path. `evaluate-timexer-segment --project-candles` compares the same checkpoint with its exact weighted valid-OHLC projection on identical validation origins. Weights match the training standard deviations. The comparison adds series to the existing validation/progress reports and displays projected candles; it does not add report bases or change training targets.

## Complete epochs and held-out data

Malformed source candles (nonpositive/nonfinite prices or impossible OHLC geometry) are omitted without changing valid prices. Sparse indices map the remaining observations to the original memory-mapped files. Contexts and horizons count valid observed bars, rather than fixed wall-clock intervals. The checkpoint authenticates the raw file fingerprint, omitted row indices, and valid observation count; the existing coverage report states the omission count.

One epoch covers every eligible training target bar once. Adjacent target segments are disjoint; their historical inputs can overlap. The final segment of each ticker uses a target mask so the remainder contributes without repeating labels. The loss sums squared OHLC errors over valid target bars and divides by four times the number of valid bars. The final optimizer batch is retained.

Shared UTC timestamp boundaries define chronological 70/10/10/10 train/calibration/validation/test partitions. Every boundary purges at least the forecast length, currently 192 target bars. `--common-context 6000` aligns eligible origins across shorter-history comparisons. Tickers without sufficient purged training history are excluded explicitly in the authenticated corpus contract. Full validation covers disjoint complete segments; unused validation remainders are recorded. The terminal test remains locked.

Every 1,000 optimizer steps, 2,048 fixed held-out segments provide a quick evaluation and four fixed windows provide candle previews. Epoch completion evaluates the full validation population. Preview scores do not select the best checkpoint or trigger stopping. A time limit can interrupt an epoch; its progress remains partial.

## Throughput

Defaults are batch 256, BF16 activations, FP32 normalization and master weights, flash SDPA, and the existing NorMuon optimizer with five-step Polar Express orthogonalization. Hidden encoder matrices use NorMuon at 0.023; patch/variate embeddings, the output head, normalization parameters, and biases use AdamW at 0.008. Those bases, the quadratic/cautious weight decay (NorMuon 1.2, AdamW 0.005, embed/head `wd_mul=150`), and the step schedule come from `../modded-nanogpt/train_gpt.py`: constant until 40% of planned optimizer steps, then a linear cooldown to 0.15 of peak over the remaining 60%. NorMuon momentum warms from 0.85 to 0.95 over 300 steps and cools over the last 50. `--learning-rate` sets the AdamW base and scales NorMuon by 0.023/0.008. `--optimizer adam` retains the TimeXer Adam recipe at 0.0001; `--fused false` selects its native implementation. Training defaults to one complete epoch; longer explicitly requested runs retain patience three completed epochs without improved full-validation MSE. The 1.52/1.73 stage multipliers in modded-nanogpt are omitted because they track a batch-size schedule this trainer does not have.

The corpus uses memory-mapped bar files, parallel batch construction, pinned host memory, and a bounded background prefetcher. It does not materialize every overlapping context or load the entire universe into VRAM. There is no gradient accumulation or model chunking.

Before the Polar Express change, a short synthetic Adam kernel benchmark measured 1,754 origins/second at batch 256, 100% GPU utilization, approximately 529 W steady power, and 18,226 MiB peak allocated memory on the workstation’s RTX 5090. Subsequent Polar Express benchmarks verified optimizer graph capture, 1,254–1,401 origins/second, 99–100% GPU activity, about 87% total device VRAM occupancy, and approximately 18 GiB peak allocated memory. These short measurements are hardware evidence, not forecast-accuracy evidence; production reports separately measure sustained data-loading and training performance.

## TUI reports

Five normal views use the existing `.report.bin` system and shared TUI registry:

- `timexer_segment_validation`: training, preview, and full-validation MSE with matching persistence baselines.
- `timexer_segment_progress`: unique target coverage and invalid forecast-candle fraction; corpus membership counts annotate the report.
- `timexer_segment_timing`: training-step, evaluation, and host-loader waiting time, with allocator peak noted.
- `timexer_segment_hardware`: GPU activity, memory-controller activity, VRAM occupancy, and power relative to device capacity.
- `timexer_segment_candles`: standard colored candles with transparent predicted candles; four windows are retained under `candle_snapshots`.

Preview and full-validation series remain distinct. Each candle chart identifies its actual ticker. Predictions are direct deterministic OHLC outputs, not probability intervals or sampled paths. The joint decoder enforces candle validity before scoring and rendering. Legacy unconstrained outputs remain unchanged and counted unless the explicit weighted-projection comparison is requested; the renderer itself does not repair forecasts.

## Running

All GPU work goes through the shared queue:

```bash
mlq submit --name timexer-universe --max-parallel-runs 1 --time-limit 12h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment \
  --run timexer-universe
```

The no-subcommand CLI and TUI select this model. `--seq-len`, `--pred-len`, and `--patch-len` remain configurable; patches must cover the history exactly. When increasing history, increase `--common-context` to at least that length.

Every evaluation saves the latest preview checkpoint. Completed epochs have separate checkpoints, and `weights/best` selects full-validation results only. Safetensors weights and the manifest authenticate model configuration, ticker sources, scalers, partitions, purge, feature choices, optimizer recipe, objective, and target coverage. These are inference checkpoints; optimizer resume is not implemented.

```bash
mlq submit --name timexer-validation --max-parallel-runs 1 --time-limit 3h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh evaluate-timexer-segment \
  --checkpoint training/runs/timexer-universe/weights/best \
  --output training/runs/timexer-universe/gens/0
```

## Matched context and volume campaign

`benchmarks/timexer_universe_campaign.py` queues three complete-corpus, one-epoch comparisons after an existing 96-bar reference job: 2,048-bar history, 6,000-bar history, and 6,000-bar history with volume. All use Polar Express, common context 6,000, horizon 192, batch 256, and seed 20260905. Jobs run serially with exclusive GPU admission and three-hour limits.

```bash
python benchmarks/timexer_universe_campaign.py --submit \
  --reference-job 4987 --reference-run timexer-universe-c96-20260905 \
  --campaign timexer-universe-20260905
```

The final queued job authenticates matching source fingerprints, ticker membership, chronological boundaries, scalers, target coverage, and comparison settings across all four completed runs. It reads full-validation MSE through `report_cli` from the actual reports. Volume is enabled only when its matched 6,000-bar experiment improves full-validation MSE; ties retain OHLC alone. The final context remains 6,000 regardless of shorter-history results. Final training runs for up to ten epochs with normal patience three and a twelve-hour queue limit. A failed or incomplete comparison blocks final selection instead of substituting partial evidence.
