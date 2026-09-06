# CausalPatch full-segment forecasting

`train-timexer-segment` / `evaluate-timexer-segment` train a decoder-only causal patch transformer with dense per-token multi-horizon heteroscedastic heads (TimesFM/Toto style). Existing categorical, LeJEPA, and other models remain available.

Each row is one ticker's 6,000 completed five-minute OHLC context bars plus the next 192 bars. Training pools the entire eligible ticker universe; every history, target, and attention operation stays within its row's ticker. The only cross-ticker information is the exogenous market/SPY variates and the cumulative market path the targets are demeaned by. `--ticker` optionally selects an explicit comma-separated subset.

## Row layout

The corpus stores per bar over the full `L = seq_len + pred_len` window: `log_prices [L,4] = ln(price) - ln(c_ctx)` with `c_ctx` the row's last context close (centered for fp32 precision), `valid [L]` (observed valid bar; targets past the ticker's owned segment are masked), `aux [L, n_aux]` with the fixed channel order time-of-day sin/cos, day-of-week sin/cos, session-gap flag + `ln(delta/5)`, volume innovation + validity, market return + validity, SPY return + validity, and `market_cum [L]`. Calendar and gap channels are `known_future`; volume/market/SPY are history-only and written as 0 with validity 0 beyond the context. The packed row is `L*4 + L + L*n_aux + L + 1` floats (`anchor = c_ctx`). All statistics move to the GPU; the CPU computes nothing else.

`market_cum` is the cumulative equal-weighted market log return on the shared UTC grid, gaps included. The grid carries every timestamp any universe ticker prints, including extended hours (04:00–20:00 ET), where only 150–500 mostly illiquid tickers trade (up to ~2,000 around 08:30 releases) against 2,400–4,300 during the regular session. A grid slot therefore *defines a market step* only when at least `--market-min-cross-section` tickers (default 2,000) hold a valid bar there; the step is the mean over tickers holding valid bars at that slot and at the previous defining slot of `ln(close / close at the previous defining slot)`, so a ticker that skipped a defining slot never injects a multi-step return, and extended-hours prints neither define steps nor enter one (the next defining step spans them, and bars at non-defining slots read the previous value). With the default the steps are the regular-session bars plus one overnight step 15:55 → 09:30 per session (contributors ≥ 1,500, median ≈ 3,400). Unlike the SPY channel, which excludes gap-crossing returns, this series and the aux market channel (the step ending at the bar's timestamp, invalid at non-defining slots) carry the overnight and weekend moves that make up most of the shared drift. Every value uses bars at or before its slot, so it is contemporaneous history, and each row stores it minus its value at the row's last context bar. The universe membership, partition boundaries, and threshold that define it are authenticated through the contract's `market_fingerprint`; the corpus report title records the step count, contributor minimum/median, step std, and largest step.

## Model

- Tokens: patch `k` covers context bars `[16k, 16k+16)`, one stream per row (no channel independence). Token input is the flatten over the 16 bars of the four log-price channels made origin-relative (`(ln p - ln c_k) / σ_k`) plus all `n_aux` covariates (the market and SPY return values divided by `σ_k`; their validity flags and every other channel unscaled), through `Linear(16*(4+n_aux) -> d_model)`. 375 tokens with rotary positions.
- Backbone: pre-norm transformer, defaults 8 layers, `d_model` 512, 8 heads, FFN 2048, dropout 0, flash SDPA with `is_causal=true`, final norm. Every norm is a gainless RMSNorm (`x·rsqrt(mean(x²)+1e-6)`, no gain, no bias, no mean subtraction), so the backbone carries no normalization parameters at all. No autocast: activations are cast to bf16 once, at the patch embedding, and every fp32 master parameter is cast to the activation dtype at its point of use, so every norm and GEMM sees matching dtypes and reaches its fused kernel instead of ATen's mixed-dtype fallback. σ, the causal statistics, the candle geometry, the NLL elements and both loss reductions are fp32; nothing else is.
- Block internals follow the modded-nanogpt residual recipe (`train_gpt.py`/`train_gpt_medium.py`, records 5 and 11): QK-norm — Q and K are RMS-normalized per head over `head_dim` and only THEN rotated, so the rotary products are unit-RMS (the V path is not normalized); the FFN activation is `relu(x)²` rather than GELU, at the reference's 4× width with no output-scale correction; the four block projections carry no bias; and the attention output and FFN down projections are zero-initialised while QKV and FFN-up use `uniform(±√3·0.5·fan_in^-½)`.
- Two custom CUDA kernels carry that recipe on the real path (`fused_kernels/`, a sibling crate compiled by nvcc for sm_90/100/120 with its autograd nodes written as `torch::autograd::Function` in C++): `qk_norm_rope` does the per-head QK-norm AND the packed rotation in one pass over the raw `q‖k` block, materializing neither the normalized block nor an `rstd` and recomputing the normalization in its backward, and `relu_square` does the FFN activation in one pass instead of `relu` writing a `[tokens, ffn]` tensor for `square` to read back. Both are bit-identical to the ATen compositions they replace in forward AND backward — no tolerance anywhere, which is why no test moved when they landed — and both are CUDA-graph-capturable. Off CUDA they dispatch to the same composition, which is what the model's CPU tests exercise. At B=256 the pair removes 52.6 ms/step (221.16 → 168.59, measured on one idle device) and cuts the analytic step traffic from 199.827 to 143.204 GB; both then run at 99.6-100.5% of the device's measured streaming roof, so nothing further is available from either without removing a pass.
- Residual mixing is learned per sub-block, `x ← λ_r·x + λ_p·branch(rmsnorm(x)) + λ_0·x0` on the attention line and `x ← λ_r·x + λ_p·branch(rmsnorm(x))` on the feedforward line, with `x0` the normalized patch embedding. All three are raw (not logit) scalars: `λ_r = √1.1` per sub-block so a layer scales the stream by 1.1, `λ_p = 1`, `λ_0 = 0`. At initialization every branch output is zero and `λ_0 = 0`, so the stack is the identity on the residual stream up to the `1.1^layers` scale the gainless final norm removes — the model starts as patch-embedding → head. The lambdas live in three 1-D banks (`lambdas.resid`, `lambdas.post`, `lambdas.x0`), are AdamW-routed with weight decay 0 and, for the residual and x0 banks, 5× the Adam base learning rate; `λ_p` is folded onto the output-projection weight copy the GEMM already needs rather than scaling the activation, and the residual and x0 terms are `addcmul`, so the whole scheme costs one extra `[tokens, d_model]` pass per layer.
- U-net skips over the backbone (modded-nanogpt record 11, `records/track_1_short/2024-11-10_UNetDoubleLr/c87bb826-797b-4f37-98c7-d3a5dad2de74.txt:239-244,257-270`): the encoder half of the layers pushes its output onto a stack, the decoder half pops one and folds it into the residual stream before its own compute, pairing `3->4, 2->5, 1->6, 0->7` at eight layers. The gate is `σ(logit)` with the logit initialised to `-1.5`, so each skip starts at `0.182` and is confined to `(0, 1)` however far the logit travels — record 11's raw init of `1.0` would nearly double the residual stream entering layer 4 before a single gradient step, and the current `train_gpt.py:1346` carries exactly this logit init (`-1.5 * torch.ones(1),  # skip_lambda -> σ(-1.5) ≈ 0.18`). The four logits are one root-level `skip_weights` parameter, AdamW at 5× the base rate and no weight decay, matching the reference's `"scalars"` group (`train_gpt.py:2030`); their post-sigmoid values are reported per interval as `skip weight <src>-><dst>`. Cost: the stack itself is four references to tensors the next layer's norm already retains, so it adds no bytes; the fold is one `addcmul` per pair, which materializes one `[rows·375, 512]` bf16 activation each — 375 MiB at batch 256 and +2.36 GB of the ~144 GB a step moves (+1.6%).
- Value residual (modded-nanogpt `records/track_1_short/2024-11-06_ShortcutsTweaks`, `README.md:10-11`, code `43f60c4f-…txt:168,177`): layer 0 publishes its attention value `V₁`, and every later layer attends with `(1 - λ_l)·V_l + λ_l·V₁` under a raw, unbounded, per-layer scalar `λ_l` initialised to 0.5 (`block_{l}.value_lambda`, `l ≥ 1`; layer 0 owns no lambda, since with `V₁ = V_0` its gradient is identically zero). The mix is one `lerp` — one read of each operand and one write, against three kernels for the literal `(1-λ)V + λV₁` — and `lerp`'s branch on `|λ| < 0.5` makes both endpoints exact, so `λ = 0` is bit-identically the model without the residual and `λ = 1` bit-identically layer 0's value. It stays on the flash path (flash needs only unit stride on `head_dim`), but it does materialise `V`: layers 1..7 write a real `[rows, origins, heads, head_dim]` bf16 tensor where V used to be a strided view of the packed projection, which is +656 MiB of retained activations and +8.65 GB (+5.6%) of the step's traffic at batch 256, charged in `ModelConfig::step_cost` and timed as the `value residual mix` kernel class. `λ_l` is trained by AdamW in the no-weight-decay group, mirroring the reference's `wd_mul = 0` for mixing lambdas, and reported per layer through `CausalPatchModel::recipe_scalars`.
- Causal statistics per origin `t_k = 16(k+1)-1` from `log_prices`/`valid`: `σ_k` = expanding std of valid close-to-close log returns over `[0, t_k]` (variance floor 1e-8), `ρ_k` = expanding mean relative range `exp(lh-ll)-1`, `c_k` = close at `t_k`, `β_k = (Σ r_i m_i + λ_k)/(Σ m_i² + λ_k)` over the valid consecutive-bar pairs in `[0, t_k]` with `r` the ticker's log close return and `m` the `market_cum` step over the same pair, `λ_k = 256 · max(mean(m²) so far, 1e-8)`: a ridge slope shrunk toward `β = 1` with the weight of 256 average market-step squares, so origins with little history sit near 1 (the old unit-β demeaning) and β converges to the OLS slope as history accumulates (half-way at 256 pairs, ~4% prior weight at 6,000). Equal-weight demeaning with β = 1 added variance for low-β tickers: over 61,901 validation windows the σ-scaled h=192 persistence MSE was 285 absolute vs 408 unit-β relative, and 953 vs 1,652 in the lowest-σ quintile (median β 0.30); the causal β gives 255 overall and 908 / 96 / 94 / 90 / 88 against 953 / 133 / 126 / 112 / 101 absolute per σ quintile (validation β quantiles p10 0.17, p50 0.66, p90 1.25). Origins with fewer than `--min-history` (256) valid bars are masked out of the loss.
- Head per token: future-known covariates for bars `t_k+1..t_k+192` (`known_future` channels only) -> `Linear(192*n_known -> 256)`, concatenated with the token state -> MLP (hidden 1024, GELU) -> `8×192` channel-major: four candle coordinates then four log-scales, each contiguous over the 192 future bars. The layout is load-bearing — every consumer slices one channel, and in the `192×8` layout each such slice is a stride-8 gather over a 147 M-element space, one 32-byte sector per two useful bytes. The output layer is zero-initialised and its `1/√1024` μP multiplier rides on the weight copy the GEMM already needs (scaling a 1024×1536 weight, not a 147 M-element output), so Adam's first steps move each output by O(lr) rather than O(lr·fan_in); the log-scale is soft-capped to `±4` around the `½·ln h` prior (`4·tanh(x/4)`), which bounds the NLL for any target.
- Decoder: the persistence-anchored joint OHLC decoder per (origin, bar): `close = c_k·exp(σ_k·√h·c0)`, range a softplus multiple of `ρ_k`, open/close positions sigmoids inside it. Predictions in loss units are `ŷ = ln(p̂ / c_k)/σ_k`; the scale is `s = √h·exp(logscale)`. Zero coordinates therefore give the persistence candle with the random-walk `σ√h` prior.
- Targets are market-neutral: `y[k,h,c] = (log_prices[t_k+h,c] - ln c_k - β_k·(market_cum[t_k+h] - market_cum[t_k]))/σ_k`, mask `valid[t_k+h]·origin_mask[k]`. The ticker's causal exposure to the cumulative market log return over the same bars is removed from the mean so the head cannot learn the training period's shared drift or regime, which does not transfer to the later validation period. `σ_k` stays the ticker's own causal return std: it is causal and simple, keeps the persistence prior and decoder unchanged, and the residual (market-neutral) volatility would be an alternative scale, not a requirement. The decoder is unchanged and emits ticker-relative candles anchored at `c_k`; the raw-space price forecast is the market-neutral forecast (market forecast zero), and persistence sits at zero coordinates in both spaces.
- Loss: masked Gaussian NLL `½((y-ŷ)/s)² + ln s` averaged over valid (origin, bar, channel) triples across all 375 origins per row. Dense MSE in σ units is reported without gradient.

Manifest `format: "causal-patch-ohlc-universe-v7-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres"`, `objective: "causal_patch_market_neutral_nll_v3"`; other formats/objectives are rejected on load, including `causal_patch_nll_v1` checkpoints trained on raw returns, `causal_patch_market_neutral_nll_v1` checkpoints trained with unit β, and the pre-recipe `causal-patch-ohlc-universe-v6-head-channel-major-folded-mup` state dicts, which carry 34 norm and 32 bias tensors this model does not have and none of the 51 recipe scalars it does. The manifest authenticates layers, `d_model`, heads, FFN, patch length, `min_history`, the feature set, and the corpus contract including `market_fingerprint`; it records `best_step` and `best_preview_nll` for the checkpoint `weights/best` points at.

## Validation and scoring

Scoring uses the row's final origin only (`k = 374`, `c = c_ctx`, σ over the whole context). Every series is computed in `market-neutral` space, the training objective: β-adjusted residual targets, σ-scaled MSE against flat persistence (`ŷ = 0`), NLL against the persistence-with-√h-prior baseline (`ŷ = 0`, `s = √h`), per-horizon curves, the robustness diagnostics, predictive-interval coverage, `tail_loss_share`, and the invalid-candle fraction. The MSE and its per-horizon curve are also computed in `raw` space so numbers stay comparable with runs trained on raw returns: `y_raw = y + β_k·(market_cum[t_k+h] - market_cum[t_k])/σ_k`, the model's raw forecast is the same `ŷ`, persistence is again zero. USD RMSE/MAE compare the raw forecast prices with observed prices. The code spells these two spaces `validation_*`/`absolute_*` internally; only the reports name them, as `market-neutral` and `raw`. Candle charts draw the market-neutral forecast re-based onto the realized market path (`ŷ + drift`), as their titles state, so the panel stays readable against observed prices.

The same single pass also answers whether the forecast is *tradable*, not merely more accurate
than persistence: the MSE-gain decomposition, the information-coefficient family, the
microstructure anchors and the cost-aware decile long/short backtest all come out of the
`Scorer` accumulators, on device, with one host transfer of 34 per-horizon sums plus three
moments per reported holding period. No separate harness and no per-element CPU loop. The
accumulators keep the signed close coordinate and its mask per (window, bar), which is
everything those statistics need — the tail-trim threshold array is reconstructed from them
rather than stored — and cross-sectional statistics group the windows by `groups[window]`, the
dense rank of the window's origin timestamp. That index is built once on the host from the
origin list `score` already holds (`corpus.ticker(ref).timestamp(ref.origin)`, one
memory-mapped bar header per scored window), so a per-timestamp reduction is a single
`index_add` over dimension 0. Per-timestamp decile membership needs no loop over timestamps
either: rank the predicted returns globally with a double `argsort`, then sort the composite
key `group·windows + global rank`, which orders windows by timestamp and, within a timestamp,
by predicted return; subtracting the group's start offset from the sorted position is the
within-timestamp rank, and invalid windows carry a sentinel prediction so they land after
every valid member of their own timestamp.

Every 1,000 optimizer steps (`--eval-every`), 2,048 fixed held-out segments (`--eval-origins`) form the `held-out sample`; each such evaluation is saved as `weights/preview-latest`, and whenever its NLL improves the weights are also saved as `weights/preview-best` and `weights/best` is pointed at them. `--preview-patience N` (default 3) stops the run once held-out sample NLL has gone N consecutive evaluations without improving, not counting the first two; the stop is logged with the triggering metrics and `weights/best` keeps the best of those weights. Epoch completion evaluates the `held-out full` population, saves `weights/epoch-NNNN`, and feeds `--patience` (complete epochs without improved held-out full NLL); only a run that never reaches a held-out sample evaluation points `weights/best` at its epoch checkpoint. The `--preview-patience` flag and the `preview-*` checkpoint directory names are retained CLI and filesystem identifiers, not report vocabulary.

## Complete epochs and held-out data

Malformed source candles are omitted without repair; sparse indices map the remaining observations to the memory-mapped files. One epoch covers every eligible training target bar once; adjacent target segments are disjoint and the final segment of each ticker is masked. Shared UTC timestamp boundaries define chronological 70/10/10/10 train/calibration/validation/test partitions, each purged by at least 192 bars. `--common-context 6000` aligns eligible origins across shorter-history comparisons. The terminal test remains locked.

## Throughput

Batch 256, bf16 activations, fp32 master weights, flash SDPA, NorMuon with five-step Polar Express orthogonalization. Transformer block matrices (`block_*`) use NorMuon at 0.023; the patch embedding, covariate projection, heads, norms, and biases use AdamW at 0.008. Weight decay, schedule (constant to 40% of planned steps, linear cooldown to 0.15), and momentum warmup/cooldown follow `../modded-nanogpt/train_gpt.py`. `--learning-rate` sets the AdamW base and scales NorMuon by 0.023/0.008. The optimizer step is CUDA-graphed. No gradient accumulation or model chunking. The corpus uses memory-mapped bar files, parallel batch construction, pinned host memory, and a bounded prefetcher.

Forward and backward are not graph-captured in the training path, and after the fused kernels the reason is that capture buys nothing - not that it cannot fit. Both halves of that were measured at batch 256 on an idle device with `benchmark-timexer-segment --capture-audit` (job 5184, `timexer_segment_benchmark`'s capture scalars): the audit warms up on a side stream, calls `empty_cache` (reserved 18,912 MiB after warmup, 952 MiB after the release, 599 MiB of it live and pinned by the optimizer's already-captured bodies) and only then captures, so the private mempool it reserves - 17,854 MiB - is the working set's new home rather than a second copy of it. Reserved at capture end is 18,806 MiB of 32,116 MiB, which is 184 MiB BELOW the same run's eager peak reserved. The earlier "~15 GB on top of the global pool" accounting, and job 5116's `CUDA out of memory. Tried to allocate 282.00 MiB`, were both from the older ordering that captured without releasing first. What capture is worth now: replay 167.38 ms against 166.96 ms eager, +0.25%, because the fusions removed six of eight rotation kernels, the QK-norm kernel and one of two activation kernels per layer, and the step is no longer launch-bound. Correctness is not the obstacle either - the replay's objective differs from eager by 4.8e-4 worst over 20 steps against an eager-vs-eager null of the same order. What the step does keep: no scalar crosses to the host inside a step (the objective's finiteness is accumulated on device as an indicator and read once per report interval with the interval's mean loss), and the optimizer step remains captured inside NorMuon.

## TUI reports

One vocabulary throughout, in series labels, chart titles and CLI output alike. **Splits**:
`training` (the running training-set estimate over the interval's batches), `held-out sample`
(the fixed fast window set scored every report interval, `--eval-origins`), `held-out full`
(the whole disjoint validation population, scored at epoch end), `persistence` (the
zero-forecast baseline with the √h scale prior). **Spaces**: `market-neutral` (the β-adjusted
residual return the model optimizes) and `raw` (market drift added back, so numbers stay
comparable across model generations). A series label reads `<split> <space> <quantity>`, with
the space omitted where the quantity exists in only one space — NLL, calibration and the
robustness diagnostics are market-neutral only. Every title reads
`CausalPatch epoch <e> step <s> | <what the chart answers> | <scope facts>`, and the scope
facts always state how many origins each split scored (one scored origin is one forecast
window) and over how many tickers. Every `y_label` carries the unit and its reading rule.

Each base answers one question in one unit. That is the reason for the split: a shared axis
carrying a dimensionless ratio near 1 beside a σ-scaled level beside a rate near 0.5 renders
all but one of them as a flat line.

- `timexer_segment_skill` — **the headline, checked first**, and first in the TUI panel order.
  Ratios only, `ratio vs persistence (dimensionless; < 1 = skill)`: aggregate forecast /
  persistence MSE for both splits in both spaces, the median over scored windows of each
  window's market-neutral ratio for both splits, and a parity line at 1.0.
- `timexer_segment_loss` — `nats per bar`: training NLL, and for each held-out split the
  forecast NLL beside its persistence NLL.
- `timexer_segment_error` — `σ-scaled squared log-return` on a log axis: training MSE, and for
  each held-out split both spaces plus their persistence counterparts. The two spaces share
  one axis deliberately: they differ by a multiplicative factor, which a log axis renders as
  an offset, and separating them would hide the comparison the two spaces exist to support.
  Latest held-out price RMSE/MAE in USD sit in the title.
- `timexer_segment_calibration` — `fraction of valid target bars`: share of valid targets
  inside the predicted ±1σ and ±1.96σ bands per held-out split, against the nominal 0.683 and
  0.950 reference lines.
- `timexer_segment_horizon` — x = bars ahead, ratios only: forecast / persistence MSE per
  split per space plus parity 1.0.
- `timexer_segment_horizon_error` — x = bars ahead, the σ-scaled levels those ratios divide,
  log axis: forecast and persistence MSE per split per space.
- `timexer_segment_horizon_robust` — x = bars ahead, ratios the top-1% |target| tail cannot
  dominate the way MSE is: MAE ratio (Σ|y-ŷ| / Σ|y| over valid (bar, channel), so persistence
  is the denominator) and the MSE ratio with each horizon's top-1% |close target| bars
  dropped, plus parity 1.0.
- `timexer_segment_horizon_rates` — x = bars ahead, `fraction of valid target bars`: close win
  rate against persistence, close directional hit rate over bars with a nonzero close
  forecast, close predicted-up share (a bias diagnostic), plus parity 0.5.
- `timexer_segment_decomposition` — x = bars ahead, `share of the persistence MSE`: where the
  MSE gain comes from. Writing the close forecast as `ŷ = μ + g` with `μ` the population mean
  forecast at that horizon and `mean(g) = 0`, and `β̂ = mean(y·g)/mean(g²)`, the gain splits
  exactly into the constant tilt's gain `(2μȳ - μ²)/mean(y²)`, the demeaned forecast's gain at
  its own best scale `(mean(y·g)²/mean(g²))/mean(y²)`, and the cross term
  `-(β̂-1)²·mean(g²)/mean(y²)` ≤ 0. The middle component is identically `ρ²·var(y)/mean(y²)`
  for the Pearson `ρ` on `_signal`, so it *is* the `1 - ρ²` MSE ratio a genuine conditional
  mean implies, plotted in the same unit as the measured total: the gap between the two lines
  is the offset/microstructure story. The all-channel total gain, the complement of the
  headline ratio, sits beside them. Close channel unless a series says otherwise.
- `timexer_segment_signal` — x = bars ahead, `correlation coefficient`: the
  information-coefficient family on the demeaned close forecast. Pooled Pearson, pooled
  Spearman rank correlation (ordinal ranks, no tie averaging), and the cross-sectional IC —
  the correlation computed within each evaluation timestamp across that timestamp's tickers,
  averaged over timestamps, with its ±1 standard-error band so significance is visible. A
  timestamp contributes only with at least 20 valid tickers.
- `timexer_segment_offset` — x = bars ahead, `σ-scaled mean log return`: the levels the ratios
  hide. The population mean predicted close coordinate (the size of the constant tilt, which
  is what a predicted-up share of 0.066 means), the mean realized coordinate, and the mean
  realized return on the side the forecast took within the top and bottom decile of `|g|`
  plus their spread — a strategy only acts on high-conviction predictions.
- `timexer_segment_tradable` / `timexer_segment_tradable_rates` — x = bars ahead, the
  microstructure test, in `ratio vs the anchor's own persistence` and `fraction of scored
  bars` respectively. Persistence anchors on the last *trade* close, so predicting something
  nearer the mid beats it on MSE while being uncapturable: your own execution pays the spread.
  Two additional anchors expose that. (i) Persistence anchored on the log midpoint of the
  origin bar's high/low: the forecast's error is unchanged, only the denominator moves.
  (ii) A one-bar execution delay: the bar `t+1` close to bar `t+h` close move, forecast by
  `ŷ_h - ŷ_1`, against that same delayed persistence — undefined at `h = 1`, where the move
  is identically zero. If the `h = 1` edge is bounce, both ratios sit at 1.0 and both hit
  rates at 0.5 while the close-anchored ratio does not. The rates chart also carries the
  top- and bottom-decile hit rates.
- `timexer_segment_portfolio` / `timexer_segment_portfolio_sharpe` — x = bars ahead over
  `h = 1, 4, 16, 64, 192`, in `basis points per holding period` and `annualized Sharpe`. At
  each evaluation timestamp, rank that timestamp's tickers by predicted `h`-bar
  market-neutral close return (the σ-scaled coordinate times the row's σ), go equal-weighted
  long the top decile and short the bottom decile, hold `h` bars; one series per per-side cost
  in 0, 1, 2, 5 and 10 bps, charged on entry and exit of both legs, so a level of `c` deducts
  `4c` bps from the spread return. Sharpe annualizes with 252 × 78 = 19,656 five-minute bars
  per year. **This is an upper bound.** It ignores market impact and borrow availability and
  cost, it treats consecutive holds as non-overlapping when for `h > 1` they overlap (which
  inflates the Sharpe), and a timestamp's cross-section is the held-out origins that share
  that timestamp, not a tradable universe snapshot.
- `timexer_segment_progress` — `fraction of valid target bars`: unique training target
  coverage, held-out invalid forecast-candle fraction, and the held-out top-1% |target| share
  of squared error. The corpus contract is appended to the title.
- `timexer_segment_timing` — `milliseconds`: the interval-mean training step and its host
  loader wait (the loader runs one batch ahead, so only the unoverlapped part shows); the six
  per-phase GPU timings of one sampled step (host batch, H2D, forward backbone, forward head
  and loss, backward, optimizer), which sum to that step's wall clock; the held-out evaluation
  total with its synchronized phases (host batch wait, H2D and forward, metric accumulation);
  and the post-evaluation candle windows plus checkpoint writes. Allocator peak in the title.
  Evaluation scores every batch on device (`Scorer`); the only host transfer is the summary.
- `timexer_segment_recipe_scalars` — `learned mixing coefficient (dimensionless)`: one series
  per scalar `CausalPatchModel::recipe_scalars()` returns, in its order — `residual lambda
  L<l> attn|ffn` and `post lambda L<l> attn|ffn` (raw, init `√1.1` and 1), `x0 lambda L<l>`
  (raw, init 0), `skip weight <src>-><dst>` (POST-sigmoid, init 0.182) and `value lambda L<l>`
  for every layer but the source (raw, init 0.5); 51 series at the 8-layer default. The panel
  answers one question — has a coefficient moved off its init — so it reports what the forward
  pass applies, never a stored logit. Written once per report interval, one host transfer.
- `timexer_segment_hardware` — GPU activity, memory-controller activity, VRAM occupancy, power.
- `timexer_segment_candles` — colored observed candles with transparent forecast candles.
  The title states, in order, the split, the ticker, the origin timestamp with timezone, the
  horizon, and that the drawn forecast is the conditional-mean path re-based to the origin
  close. Windows are drawn from the held-out validation population; four are retained under
  `candle_snapshots`.

### Benchmark reports (`benchmark-timexer-segment`)

`benchmark-timexer-segment --profile` writes these beside the training bases. It runs a
synthetic batch, so it measures kernels, never skill.

- `timexer_segment_benchmark` — `named units`: end-to-end step milliseconds, origins per
  second, allocator peaks, step matmul TFLOP with the achieved TFLOPS and the fraction of the
  measured bf16 GEMM peak, and the analytic activation-traffic bound with the achieved GB/s and
  the fraction of the measured copy peak. Both peaks are measured in the same process as the
  best of eight CUDA-event rounds, not looked up.
- `timexer_segment_benchmark_phases` — `milliseconds`: the six synchronized phases of one step
  (host batch, H2D, forward backbone, forward head and loss, backward, optimizer). Attribution,
  not throughput: the synchronization is the point.
- `timexer_segment_benchmark_kernels` — `milliseconds`: one backbone layer at the real shapes,
  per class of kernel (RMSNorm, QK norm + rotary, QKV projection, causal SDPA,
  attention output flatten and projection, residual addcmul, FFN up, ReLU², FFN down), forward
  and backward
  separated, plus the composed layer so the sum of the parts can be compared against the whole.
  Two `reference *` classes are the pre-fusion forms, timed in the same process, so a
  before/after is one run rather than two.
- `timexer_segment_benchmark_kernel_roofline` — `percent of the measured device peak`: per
  class, forward bandwidth and forward arithmetic separately. Forward only: a class's forward
  bytes and FLOPs are exactly enumerable from its shapes, whereas backward traffic is per-op
  ATen internals (`add`'s backward moves nothing, a GEMM's is two more GEMMs).
- `timexer_segment_benchmark_kernel_activations` — `mebibytes`: per class, what one forward left
  live beyond its own output. This is the dtype audit — a norm or a cast quietly retaining
  an fp32 `[tokens, d_model]` copy shows up here as 187 MiB the bf16 accounting cannot explain.

`research/worker_reports/timexer_kernel_traffic.md` is the measured reading of these.

Retired names, with no back-compatible readers: `timexer_segment_validation` (split into
`_skill`, `_loss`, `_error`), `timexer_segment_absolute` (folded into `_skill` and `_error`),
and `timexer_segment_robust` (split into `_horizon_robust` and `_horizon_rates`). Report files
from runs older than the rename are no longer readable by name. Every base is registered in
`TIMEXER_SEGMENT_REPORT_BASES` (`shared/src/report.rs`) and scanned by `meta_chart_bases`
(`tui/src/main.rs`); a test asserts the two agree in both directions.

## Running

```bash
mlq submit --name timexer-universe --max-parallel-runs 1 --time-limit 12h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment \
  --run timexer-universe --features all
```

Model flags: `--layers`, `--d-model`, `--heads`, `--ffn`, `--dropout`, `--min-history`, `--features`, `--seq-len`, `--pred-len`, `--patch-len` (`seq_len % patch_len == 0`), `--batch-size`, `--market-min-cross-section`, learning-rate and optimizer flags; run control: `--eval-every`, `--eval-origins`, `--preview-patience`, `--patience`, `--epochs`. When increasing history, increase `--common-context` to at least that length. Checkpoints are inference checkpoints; optimizer resume is not implemented.

```bash
mlq submit --name timexer-validation --max-parallel-runs 1 --time-limit 3h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh evaluate-timexer-segment \
  --checkpoint training/runs/timexer-universe/weights/best \
  --output training/runs/timexer-universe/gens/0
```

## Matched context and feature campaign

`benchmarks/timexer_universe_campaign.py` queues complete-corpus, one-epoch comparisons (2,048-bar history, 6,000-bar history, 6,000-bar history with all exogenous variates) after an existing 96-bar reference job and selects the final configuration from held-out full MSE read through `report_cli`. Ties retain OHLC alone; the final context is 6,000 regardless of shorter-history results.
