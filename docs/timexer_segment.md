# CausalPatch full-segment forecasting

Current leaders, comparison eligibility, and active research question: [short top-run ledger](top_runs.md).

Approved task-aligned temporal regularization design and experiment protocol: [conditional-moment SIGReg](temporal_sigreg.md).

`train-timexer-segment` / `evaluate-timexer-segment` train a decoder-only causal patch transformer with dense per-token multi-horizon heteroscedastic heads (TimesFM/Toto style). Existing categorical, LeJEPA, and other models remain available.

Each row is one ticker's 6,000 completed five-minute OHLC context bars plus the next 192 bars. Training pools the entire eligible ticker universe; every history, target, and attention operation stays within its row's ticker. The only cross-ticker information is the exogenous market/SPY variates and the cumulative market path the targets are demeaned by. `--ticker` optionally selects an explicit comma-separated subset.

## Row layout

The corpus stores per bar over the full `L = seq_len + pred_len` window: `log_prices [L,4] = ln(price) - ln(c_ctx)` with `c_ctx` the row's last context close (centered for fp32 precision), `valid [L]` (observed valid bar; targets past the ticker's owned segment are masked), `aux [L, n_aux]` with the fixed channel order time-of-day sin/cos, day-of-week sin/cos, session-gap flag + `ln(delta/5)`, volume innovation + validity, market return + validity, SPY return + validity, and `market_cum [L]`. Calendar and gap channels are `known_future`; volume/market/SPY are history-only and written as 0 with validity 0 beyond the context. The packed row is `L*4 + L + L*n_aux + L + 1` floats (`anchor = c_ctx`). All statistics move to the GPU; the CPU computes nothing else.

`market_cum` is the cumulative equal-weighted market log return on the shared UTC grid, gaps included. The grid carries every timestamp any universe ticker prints, including extended hours (04:00–20:00 ET), where only 150–500 mostly illiquid tickers trade (up to ~2,000 around 08:30 releases) against 2,400–4,300 during the regular session. A grid slot therefore *defines a market step* only when at least `--market-min-cross-section` tickers (default 2,000) hold a valid bar there; the step is the mean over tickers holding valid bars at that slot and at the previous defining slot of `ln(close / close at the previous defining slot)`, so a ticker that skipped a defining slot never injects a multi-step return, and extended-hours prints neither define steps nor enter one (the next defining step spans them, and bars at non-defining slots read the previous value). With the default the steps are the regular-session bars plus one overnight step 15:55 → 09:30 per session (contributors ≥ 1,500, median ≈ 3,400). Unlike the SPY channel, which excludes gap-crossing returns, this series and the aux market channel (the step ending at the bar's timestamp, invalid at non-defining slots) carry the overnight and weekend moves that make up most of the shared drift. Every value uses bars at or before its slot, so it is contemporaneous history, and each row stores it minus its value at the row's last context bar. The universe membership, partition boundaries, and threshold that define it are authenticated through the contract's `market_fingerprint`; the corpus report title records the step count, contributor minimum/median, step std, and largest step.

## Model

- Tokens: patch `k` covers context bars `[16k, 16k+16)`, one stream per row (no channel independence). Token input is the flatten over the 16 bars of the four log-price channels made origin-relative (`(ln p - ln c_k) / σ_k`) plus all `n_aux` covariates (the market and SPY return values divided by `σ_k`; their validity flags and every other channel unscaled), through `Linear(16*(4+n_aux) -> d_model)`. 375 tokens with rotary positions.
- Backbone: pre-norm transformer, defaults 8 layers, `d_model` 512, 8 heads, FFN 2048, dropout 0, flash SDPA with `is_causal=true`, final norm. Every norm is a gainless RMSNorm (`x·rsqrt(mean(x²)+1e-6)`, no gain, no bias, no mean subtraction), so the backbone carries no normalization parameters at all. No autocast: activations are cast to bf16 once, at the patch embedding, and every fp32 master parameter is cast to the activation dtype at its point of use, so every norm and GEMM sees matching dtypes and reaches its fused kernel instead of ATen's mixed-dtype fallback. σ, the causal statistics, the candle geometry, the NLL elements and both loss reductions are fp32; nothing else is.
- Block internals follow the modded-nanogpt residual recipe (`train_gpt.py`/`train_gpt_medium.py`, records 5 and 11): QK-norm — Q and K are RMS-normalized per head over `head_dim` and only THEN rotated, so the rotary products are unit-RMS (the V path is not normalized); the FFN activation is `relu(x)²` rather than GELU, at the reference's 4× width with no output-scale correction; the four block projections carry no bias; and the attention output and FFN down projections are zero-initialised while QKV and FFN-up use `uniform(±√3·0.5·fan_in^-½)`.
- Three custom CUDA kernels carry that recipe and the objective on the real path (`fused_kernels/`, a sibling crate compiled by nvcc for sm_90/100/120 with its autograd nodes written as `torch::autograd::Function` in C++): `qk_norm_rope` does the per-head QK-norm AND the packed rotation in one pass over the raw `q‖k` block, materializing neither the normalized block nor an `rstd` and recomputing the normalization in its backward; `relu_square` does the FFN activation in one pass instead of `relu` writing a `[tokens, ffn]` tensor for `square` to read back; and `loss_geometry` does the whole candle-geometry-plus-Gaussian-NLL chain — the σ scaling, the softplus range, the two sigmoid positions, the three `log1p` offsets, the `tanh` cap on every log scale, the `exp`, the residuals and their squares — in ONE pass over the `[rows, origins', 8, pred_len]` head, writing only the mean coordinate and the twelve fp32 vectors the twelve `dot`s consume, with a backward that recomputes the entire chain from `head` and retains no full-size fp32 tensor at all. All three are bit-identical to the ATen compositions they replace in forward AND backward — no tolerance anywhere, which is why no test moved when they landed — and all three are CUDA-graph-capturable. Off CUDA they dispatch to the same composition, which is what the model's CPU tests exercise. At B=256 the first pair removes 52.6 ms/step (221.16 → 168.59, measured on one idle device) and cuts the analytic step traffic from 199.827 to 143.204 GB; both then run at 99.6-100.5% of the device's measured streaming roof, so nothing further is available from either without removing a pass. `loss_geometry` removes 135 further fp32 slice-passes: 147 kernels and 24.85 GB/step become 18 kernels and 4.72 GB, and the analytic step traffic falls to 128.16 GB.
- Meeting that bar on a chain this long needed four facts about ATen's own CUDA build that are NOT derivable from the mathematics, are all measured by `loss_geometry_rounding_is_the_measured_form`'s 32-form sweep, and are carried in `fused_kernels::LOSS_GEOMETRY_ROUNDING`: `x / ln2` by a HOST scalar is a multiply by the fp32 reciprocal on CUDA and a true division on CPU; `tanh_backward`'s `1 - y·y` arrives contracted into one `fma`; `sigmoid_backward` is `(g·(1-y))·y`; and `softplus_backward` is `(g·z)/(z+1)`. The first of those is why an off-device probe cannot settle a bit-exactness question here — a CPU reference agrees with a CPU composition and both disagree with the GPU — and anyone writing the next fused kernel over this chain should read the four answers off that constant rather than re-derive them. Two further rules the same work established: every arithmetic step inside a fused kernel must use the explicit round-to-nearest intrinsics (`__fmul_rn`/`__fadd_rn`/`__fsub_rn`/`__fdiv_rn`), because nvcc contracts a product and a sum into one `fma` by default while the composition it replaces had a separate kernel per operation and no such opportunity; and a fused op with optional outputs must call `ctx->set_materialize_grads(false)`, because autograd otherwise hands an unused output a freshly ZEROED gradient and adding it is not free — `-0 + +0` is `+0`, which broke the mean coordinate's gradient on exactly the invalid origins until it was found.
- Residual mixing is learned per sub-block, `x ← λ_r·x + λ_p·branch(rmsnorm(x)) + λ_0·x0` on the attention line and `x ← λ_r·x + λ_p·branch(rmsnorm(x))` on the feedforward line, with `x0` the normalized patch embedding. All three are raw (not logit) scalars: `λ_r = √1.1` per sub-block so a layer scales the stream by 1.1, `λ_p = 1`, `λ_0 = 0`. At initialization every branch output is zero and `λ_0 = 0`, so the stack is the identity on the residual stream up to the `1.1^layers` scale the gainless final norm removes — the model starts as patch-embedding → head. The lambdas live in 1-D banks (`lambdas.resid`, `lambdas.post`, and `lambdas.x0` unless `--x0-lambdas disabled` removes the injection and its bank outright, which is a different parameter set and therefore a different FORMAT stamp), are AdamW-routed with weight decay 0 and, for the residual, x0 and skip banks, `--scalar-lr-mult` times the Adam base learning rate (default 5×, modded-nanogpt's, tuned on a run two orders of magnitude shorter than one epoch here); `λ_p` is folded onto the output-projection weight copy the GEMM already needs rather than scaling the activation, and the residual and x0 terms are `addcmul`, so the whole scheme costs one extra `[tokens, d_model]` pass per layer.
- U-net skips over the backbone (modded-nanogpt record 11, `records/track_1_short/2024-11-10_UNetDoubleLr/c87bb826-797b-4f37-98c7-d3a5dad2de74.txt:239-244,257-270`): the encoder half of the layers pushes its output onto a stack, the decoder half pops one and folds it into the residual stream before its own compute, pairing `3->4, 2->5, 1->6, 0->7` at eight layers. The gate is `σ(logit)` with the logit initialised to `-1.5`, so each skip starts at `0.182` and is confined to `(0, 1)` however far the logit travels — record 11's raw init of `1.0` would nearly double the residual stream entering layer 4 before a single gradient step, and the current `train_gpt.py:1346` carries exactly this logit init (`-1.5 * torch.ones(1),  # skip_lambda -> σ(-1.5) ≈ 0.18`). The four logits are one root-level `skip_weights` parameter, AdamW at 5× the base rate and no weight decay, matching the reference's `"scalars"` group (`train_gpt.py:2030`); their post-sigmoid values are reported per interval as `skip weight <src>-><dst>`. Cost: the stack itself is four references to tensors the next layer's norm already retains, so it adds no bytes; the fold is one `addcmul` per pair, which materializes one `[rows·375, 512]` bf16 activation each — 375 MiB at batch 256 and +2.36 GB of the ~144 GB a step moves (+1.6%).
- Value residual (modded-nanogpt `records/track_1_short/2024-11-06_ShortcutsTweaks`, `README.md:10-11`, code `43f60c4f-…txt:168,177`): layer 0 publishes its attention value `V₁`, and every later layer attends with `(1 - λ_l)·V_l + λ_l·V₁` under a raw, unbounded, per-layer scalar `λ_l` initialised to 0.5 (`block_{l}.value_lambda`, `l ≥ 1`; layer 0 owns no lambda, since with `V₁ = V_0` its gradient is identically zero). The mix is one `lerp` — one read of each operand and one write, against three kernels for the literal `(1-λ)V + λV₁` — and `lerp`'s branch on `|λ| < 0.5` makes both endpoints exact, so `λ = 0` is bit-identically the model without the residual and `λ = 1` bit-identically layer 0's value. It stays on the flash path (flash needs only unit stride on `head_dim`), but it does materialise `V`: layers 1..7 write a real `[rows, origins, heads, head_dim]` bf16 tensor where V used to be a strided view of the packed projection, which is +656 MiB of retained activations and +8.65 GB (+5.6%) of the step's traffic at batch 256, charged in `ModelConfig::step_cost` and timed as the `value residual mix` kernel class. `λ_l` is trained by AdamW in the no-weight-decay group, mirroring the reference's `wd_mul = 0` for mixing lambdas, and reported per layer through `CausalPatchModel::recipe_scalars`.
- Causal statistics per origin `t_k = 16(k+1)-1` from `log_prices`/`valid`: `σ_k` = expanding std of valid close-to-close log returns over `[0, t_k]` (variance floor 1e-8), `ρ_k` = expanding mean relative range `exp(lh-ll)-1`, `c_k` = close at `t_k`, `β_k = (Σ r_i m_i + λ_k)/(Σ m_i² + λ_k)` over the valid consecutive-bar pairs in `[0, t_k]` with `r` the ticker's log close return and `m` the `market_cum` step over the same pair, `λ_k = 256 · max(mean(m²) so far, 1e-8)`: a ridge slope shrunk toward `β = 1` with the weight of 256 average market-step squares, so origins with little history sit near 1 (the old unit-β demeaning) and β converges to the OLS slope as history accumulates (half-way at 256 pairs, ~4% prior weight at 6,000). Equal-weight demeaning with β = 1 added variance for low-β tickers: over 61,901 validation windows the σ-scaled h=192 persistence MSE was 285 absolute vs 408 unit-β relative, and 953 vs 1,652 in the lowest-σ quintile (median β 0.30); the causal β gives 255 overall and 908 / 96 / 94 / 90 / 88 against 953 / 133 / 126 / 112 / 101 absolute per σ quintile (validation β quantiles p10 0.17, p50 0.66, p90 1.25). Origins with fewer than `--min-history` (256) valid bars are masked out of the loss.
- Head per token: future-known covariates for bars `t_k+1..t_k+192` (`known_future` channels only) -> `Linear(192*n_known -> 256)`, concatenated with the token state -> MLP (hidden 1024, GELU) -> `8×192` channel-major: four candle coordinates then four log-scales, each contiguous over the 192 future bars. The layout is load-bearing — every consumer slices one channel, and in the `192×8` layout each such slice is a stride-8 gather over a 147 M-element space, one 32-byte sector per two useful bytes. The output layer is zero-initialised and its `1/√1024` μP multiplier rides on the weight copy the GEMM already needs (scaling a 1024×1536 weight, not a 147 M-element output), so Adam's first steps move each output by O(lr) rather than O(lr·fan_in); the log-scale is soft-capped to `±4` around the `½·ln h` prior (`4·tanh(x/4)`), which bounds the NLL for any target.
- Decoder: the persistence-anchored joint OHLC decoder per (origin, bar): `close = c_k·exp(σ_k·√h·c0)`, range a softplus multiple of `ρ_k`, open/close positions sigmoids inside it. Predictions in loss units are `ŷ = ln(p̂ / c_k)/σ_k`; the scale is `s = √h·exp(logscale)`. Zero coordinates therefore give the persistence candle with the random-walk `σ√h` prior.
- Targets are market-neutral: `y[k,h,c] = (log_prices[t_k+h,c] - ln c_k - β_k·(market_cum[t_k+h] - market_cum[t_k]))/σ_k`, mask `valid[t_k+h]·origin_mask[k]`. The ticker's causal exposure to the cumulative market log return over the same bars is removed from the mean so the head cannot learn the training period's shared drift or regime, which does not transfer to the later validation period. `σ_k` stays the ticker's own causal return std: it is causal and simple, keeps the persistence prior and decoder unchanged, and the residual (market-neutral) volatility would be an alternative scale, not a requirement. The decoder is unchanged and emits ticker-relative candles anchored at `c_k`; the raw-space price forecast is the market-neutral forecast (market forecast zero), and persistence sits at zero coordinates in both spaces.
- Loss: masked Gaussian NLL `½((y-ŷ)/s)² + ln s` over valid (origin, bar, channel) triples across all 375 origins per row, weighted along the horizon axis by `--horizon-loss` and normalized as a weighted MEAN — `Σ w·mask·nll / Σ w·mask·4`, so the number stays nats per bar and is invariant to the scale of `w`. Modes: `uniform` (`w ≡ 1`, bit-for-bit the pre-`v9` loss), `inv-sqrt` (`∝ 1/√h`), `inv` (`∝ 1/h`), and `cutoff:K` (horizons `1..=K` only, every horizon above `K` at weight exactly 0 and therefore zero gradient into the head rows exclusive to it). Every mode is normalized to mean 1 over all 192 horizons, and all 192 forecasts are still emitted in every mode — `cutoff:K` masks the loss, it does not shrink the head, so the per-horizon diagnostics keep reading the untrained end and the run pays the full head cost. The effective weight vector is charted once per report interval as `timexer_segment_horizon_loss_weight`. Checkpoint selection and both patience counters minimize the SAME weighted NLL on the held-out split (`Manifest::selection`, `best_objective_nll`), never the equal-weighted aggregate: with one trunk and 192 shared horizons the aggregate is dominated by horizons with no out-of-period predictability. Dense MSE in σ units keeps the unweighted definition and is reported without gradient.

Manifest `format: "causal-patch-ohlc-universe-v9-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres-x0-learned"` or `"…-x0-none"` under `--x0-lambdas disabled` — the stamp names the parameter set, and a manifest whose stamp and declared x0 mode disagree is refused — `objective: "causal_patch_market_neutral_nll_v5"`; the manifest also records `base_learning_rate`, `scalar_lr_mult`, `model.horizon_loss` and the `selection` criterion that names it, and the optimizer recipe string carries `x0-lambdas=<mode>` and the routed scalar banks with their multiplier, so a checkpoint is attributable to its configuration. Other formats/objectives are rejected on load, including `causal_patch_nll_v1` checkpoints trained on raw returns, `causal_patch_market_neutral_nll_v1` checkpoints trained with unit β, the pre-recipe `causal-patch-ohlc-universe-v6-head-channel-major-folded-mup` state dicts, which carry 34 norm and 32 bias tensors this model does not have and none of the 51 recipe scalars it does, every `v7` stamp, whose manifest records neither the scalar multiplier nor the x0 mode its weights were trained under, and every `v8` stamp, whose manifest records no horizon weighting at all even though its parameter set and its loss are identical to a `v9` `uniform` arm. The manifest authenticates layers, `d_model`, heads, FFN, patch length, `min_history`, the feature set, and the corpus contract including `market_fingerprint`; it records `best_step` and `best_objective_nll` — the objective-weighted held-out sample NLL, renamed from `best_preview_nll` with `_v5` because the quantity minimized changed — for the checkpoint `weights/best` points at.

## Validation and scoring

Forecast scoring uses the row's final origin only (`k = 374`, `c = c_ctx`, σ over the whole
context). Market-neutral forecasting metrics compare β-adjusted residual targets against
the predicted conditional mean, with zero-mean/√h-scale persistence. These include NLL,
MSE, per-horizon and robustness diagnostics, coverage and invalid-candle fraction.
Unweighted forecasting NLL remains separate from the horizon-weighted objective NLL, which is
reported and drives the train-versus-held-out objective gap but no longer selects anything.
Raw-space MSE adds realized market drift back to the target, keeps the same forecast and
zero persistence, and allows comparison with generations trained on raw returns. USD
RMSE/MAE compare raw forecast prices with observed prices. Candle charts re-base the
market-neutral forecast onto the realized market path for visualization; that is not the
raw entry-to-exit utility payoff defined below.

The same scoring pass measures prediction usefulness through the MSE-gain decomposition,
information coefficients, microstructure anchors, and fixed-policy **endpoint-cohort
diagnostics**. These are not deployable backtests. The device evaluator retains predicted
close means and log-scales, raw close targets, target validity, causal σ, the next valid
bar's observed open relative to the origin close, and the origin timestamp group. Only the
aggregated summaries move to the host; all inspectable metrics use `.report.bin`.

Positions depend only on origin-available forecasts, their conditional residual dispersion,
causal σ, and origin cohort membership. Outcomes, entry prices, and future validity never
select or resize positions. Cohorts need at least 20 original names. If any member lacks an
entry or terminal outcome, the entire cohort is excluded at that horizon for every policy.
This complete-outcome selection is a limitation, not an investable universe filter.

Every 1,000 optimizer steps (`--eval-every`), 2,048 fixed held-out segments (`--eval-origins`) form the `held-out sample`; each such evaluation is saved as `weights/preview-latest`, and whenever its SELECTION OBJECTIVE improves the weights are also saved as `weights/preview-best` and `weights/best` is pointed at them. `--preview-patience N` (default 3) stops the run once that scalar has gone N consecutive evaluations without improving, not counting the first two; the stop is logged with the triggering metrics and `weights/best` keeps the best of those weights. Epoch completion evaluates the `held-out full` population, saves `weights/epoch-NNNN`, and feeds `--patience` (complete epochs without an improved held-out full selection objective); only a run that never reaches a held-out sample evaluation points `weights/best` at its epoch checkpoint. The `--preview-patience` flag and the `preview-*` checkpoint directory names are retained CLI and filesystem identifiers, not report vocabulary.

Checkpoint and stopping selection minimize the SCALE-FREE objective: the training objective's
own per-horizon weighting (`model.horizon_weights()`, normalized) applied to the close
channel's MSE ratio at each horizon's own best scale, charted on `timexer_segment_skill` as
`<split> selection objective (horizon-weighted close best-scale MSE ratio)` and stamped in the
manifest as `best_scale_free_objective` under the `selection` string `min held-out sample
horizon-weighted close best-scale MSE ratio (horizon-loss=<spec>)`. It replaced the held-out
objective NLL because the NLL is a function of the learned input-dependent `log_scale` as much
as of the mean and the scale absorbs the mean's error: two states differing 2.3x in h = 1
information coefficient differed in NLL only in the fourth decimal. The scalar is free — it is
a host weighted mean over the per-horizon best-scale ratios the selection draw's existing
`TradingCurve` already carries — and the conditional half of it is exactly invariant to any
positive per-horizon amplitude gain. Every checkpoint written before this change carries the
old `selection` string and is refused by name on load. Utility never selects weights,
thresholds, costs, horizons, or policies on held-out data.
All seven policies and the cost sweep are frozen upfront. The terminal test remains locked.

## Synchronized strategy/account evaluation

`evaluate-timexer-portfolio` is the account-level counterpart to the endpoint probes.
It scans the full authenticated corpus using strictly preceding price, freshness and
regular-session dollar-volume history, fixes a liquid execution universe (default 256),
and constructs its own shared decision clock. It does **not** intersect the ticker-specific
disjoint `validation_refs`, select surviving stocks from future targets, or give overlapping
forecasts independent capital budgets. Defaults: $10,000, 60 consecutive observed sessions,
h=8, a rebalance decision every four completed five-minute bars, at most 30 names, 5% per
name, unit gross, 2% absolute net and 5% absolute beta-weighted exposure.

The allocator maximizes a deterministic cost/risk surrogate with integer-share coordinate
and paired moves, not a claimed global optimum. Sizing observes remaining concentration,
net, beta, funding and turnover headroom. New risk must pay entry and estimated exit costs;
existing positions carry unless an incremental move improves utility. Cash is a legitimate
answer, particularly when small-account minimum commissions exceed predicted edge.
Historical correlation groups and supplied sectors have separate gross concentration caps.
Predictive risk is the head's learned horizon residual dispersion, not epistemic confidence;
no independent random-walk variance scaling is imposed by the allocator.

Orders known at close fill at a later observed open, using preceding completed-bar volume
for participation limits. All intervening observed marks are replayed, including extended
hours. The ledger debits commissions, regulatory fees, spread/slippage and calendar-time
borrow exactly once, segregates short proceeds, checks margin, and retains stale/unfillable
inventory rather than inventing liquidation prices. Forecast expiry uses the deterministic
exchange calendar; nominal h×5 minutes is not substituted for overnight/weekend duration.
No historical borrow, spread, sector or corporate-action metadata is fabricated.

Absent `--calibration`, a frozen scalar gain is fitted on the **latest five eligible sessions
of the reserved calibration partition**, with complete labels and a strict pre-evaluation
target cutoff. Fit and evaluation both use causal projected future calendar/gap covariates,
not future ticker print availability. The scalar h gain applies to both terminal-close and
next-open close anchors; that shared-h1-amplitude approximation is recorded. The signal is
predicted terminal close minus predicted next open, never the realized entry price.
The gain solves nonnegative least squares: `max(0, Cov(f,y)/Var(f))`, with zero for an
unidentified constant forecast. Nonpositive calibration covariance is a valid zero-gain
boundary, not a reason to abort or invert the signal. The summary charts retain both the
unconstrained gain (when defined) and the applied gain/zero-boundary indicator.
Legacy full-curve artifacts require `--allow-observed-calendar-calibration` because their
observed-future-calendar fit contract differs, even when checkpoint authentication passes.

The account summaries also carry `recent_*` and `full_*` calibration diagnostics: raw
Cov/Var, nonnegative gain, covariance/variance moments, origin/timestamp/session counts,
cross-section widths and 40-name coverage, and delete-one-New-York-session jackknife SE.
The recent fit uses the same absolute session dates across names. Full diagnostics score
every `validation_refs` origin with projected calendar inputs, not a ticker-relative
"recent" fraction. These populations differ: the recent liquid cohort is synchronized;
the corpus-wide full reference can be very sparse. Their gain difference is not a controlled
temporal comparison. Five sessions and overlapping cross-day targets limit the recent SE;
normal-approximate bounds are not calibrated significance or regime-change evidence.
Undefined ratios/SE are omitted. Both shrinkage weights are explicitly zero. Full-window
gains are diagnostic only and never enter the account's calibrated means.

Variance diagnostics distinguish `Vh/Dh` (cumulative variance over the sum of individually
centered step variances), `Vh/(h*V1)` (the first-step random-walk normalization), and
`Dh/(h*V1)`. All use the same timestamp-centered target population. For raw gain `g`,
`a=sqrt(Vh/(h*V1))` and `g/a` separate a normalization factor from the remaining amplitude
algebraically; their product is still `g`. This does not change predictive risk or policy,
and does not identify the cause of miscalibration. Validation-estimated factors are never
fed back into the evaluator.

```bash
./torch-env.sh cargo build --release -p trading_bot_0 -p report_cli
mlq submit --name timexer-account-h64 --max-parallel-runs 1 --max-attempts 1 --time-limit 10m \
  --cwd "$PWD" -- ./torch-env.sh target/release/trading_bot_0 evaluate-timexer-portfolio \
  --checkpoint training/evaluations/timexer-portfolio-v5/checkpoint \
  --output training/evaluations/account-h64/gens/0 \
  --horizon 64 --sessions 60 --universe-size 256 --allow-assumed-short \
  --tape-cache training/evaluations/account-h64/tape.bin
target/release/report_cli 0 timexer_segment_account_summary_money \
  --run-root training/evaluations/account-h64
```

`--allow-assumed-short` makes this an explicitly hypothetical borrow-access scenario.
Without it, supply `--risk-metadata` containing
`{"as_of_ms": <timestamp before evaluation>, "assets": [{"symbol": "AAPL", "sector": "Technology", "shortable": true}]}`.
Missing symbols are not shortable. A frozen snapshot is not historical locate evidence.
No short permission plus the default near-neutral constraint may leave the account in cash;
a long-only experiment must explicitly choose its net/beta budgets rather than silently
changing mandates. The generic margin scenario does not validate broker jurisdiction/PDT
permissions for a $10,000 account.

The optional tape cache binds checkpoint/corpus identities, exact validation origin placement,
schedule, fitted gain, historical risk metadata and actual marks. It rejects mismatches rather
than silently regenerating a different experiment; use a fresh output/cache for another
horizon or window. Same-contract policy replay reuses predictions and full diagnostics but
still authenticates the corpus and rebuilds the causal plan. Reuse the authenticated scalar
artifact as well when changing only the output directory. This is not a CPU model fallback.
Strategy parameters and assumptions are recorded in
report titles; outputs use the registered `timexer_segment_account_*` binary chart family:
value, costs, risk, activity, daily/monthly P&L and returns, summary money/risk/census, timing.
All account charts state the checkpoint step. Separate output directories preserve comparisons.

**Interpretation:** validation influenced checkpoint selection and is not the terminal
holdout. A horizon sweep is strategy-development evidence, not an unbiased best-of-sweep
performance claim. Stored prices are adjusted: integer shares and per-share fees are
adjusted-unit approximations, not broker-exact historical execution. No extra dividends
are credited. Fee defaults are the explicit static
[IBKR fixed-price schedule](https://www.interactivebrokers.com/en/pricing/commissions-stocks.php)
scenario, not a date-matched historical tariff.

The pinned step2000 sweep uses weights SHA
`ccffa620654f1c95faa618501519e036249615e98d6034e425e3fed7c2309116`.
Its recent fit spans August 8–14, 2024 in absolute time. The independent full-corpus
training-date audit supplied for this comparison dates the global last training target
to August 17, 2023: this particular recent window is **certified post-training** across
the entire universe. That fact is distinct from validation-driven checkpoint selection.
The audit found one cross-ticker training-exposed origin among the 433,303 full validation
draws and zero same-ticker exposure; calendar-range overlap must not be confused with
actual sampled-population overlap.

Measured 60-session, liquid-256 sweep (same pinned step2000 weights above):

| h | Recent Cov/Var | Session-jackknife SE | Recent origins / timestamps / sessions | Full Cov/Var | Full SE | Applied gain | Process time |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | -0.109686 | 0.026347 | 12,424 / 81 / 5 | 0.026518 | 0.010551 | 0 | 209.8 s |
| 64 | -0.022959 | 0.041314 | 12,424 / 81 / 5 | 0.015875 | 0.008582 | 0 | 203.0 s |
| 128 | 0.018719 | 0.041446 | 12,413 / 80 / 4 | 0.013424 | 0.007593 | 0.018719 | 199.9 s |

Full diagnostic population: 433,303 origins at 40,837 timestamps over 252 sessions;
424,881 non-singleton origins at 32,415 timestamps over 251 sessions supply the estimator.
Only 100 timestamps have at least 40 names. This is not the account clock, whose decision
width is 252–256 names. Recent h128 has four usable origin sessions despite requesting five:
complete-label/purge rules remove the last session. Coefficients in the table are gains,
not ICs. h64 and h128 recent estimates are not distinguishable from zero by the reported
normal approximation; h32 excludes zero under that approximation, with the few-cluster
and population-mismatch qualifications above.

All three accounts retained $10,000 cash, with zero fills, fees and P&L. This is the result
under the declared small-account costs, neutral constraints and hypothetical short access,
not a proof that all policies lack edge. Runtime excludes queue waiting and includes the
entire full-window diagnostic pass. h64 authenticated replay took **35.1 s process /
31.4 s evaluator**, ran zero inference batches, and reproduced every equity, cost, risk,
activity, daily and monthly report point exactly (11,407 marks / 60 sessions / 4 months).

Recent normalization decomposition, diagnostic only:

| h | Vh/Dh | Vh/(h V1) | a = sqrt(Vh/(h V1)) | Remaining raw gain g/a |
|---|---:|---:|---:|---:|
| 32 | 0.517083 | 0.481112 | 0.693622 | -0.158134 |
| 64 | 0.594802 | 0.652357 | 0.807686 | -0.028426 |
| 128 | 0.640265 | 0.779482 | 0.882883 | 0.021202 |

Artifacts: `training/evaluations/timexer-portfolio-v7/h32/gens/0` (mlq 5472),
`training/evaluations/timexer-portfolio-v8/h64/gens/0` (5475),
`training/evaluations/timexer-portfolio-v8/h128/gens/0` (5476), and h64 `gens/1`
for replay (5477). v8 adds exact validation-origin binding to the cache without changing
the v7 inference, estimator or account policy; old cache formats are intentionally rejected.

## Complete epochs and held-out data

Malformed source candles are omitted without repair; sparse indices map the remaining observations to the memory-mapped files. One epoch covers every eligible training target bar once; adjacent target segments are disjoint and the final segment of each ticker is masked. Shared UTC timestamp boundaries define chronological 70/10/10/10 train/calibration/validation/test partitions, each purged by at least 192 bars. `--common-context 6000` aligns eligible origins across shorter-history comparisons. The terminal test remains locked.

The `[70%, 80%)` calibration partition reserved by those boundaries yielded ZERO origins until 2026-09-07: `Corpus::load` enumerated only the training and validation bands, and `CorpusTicker::target_count` admitted only those two, so an origin between `train_end` and `boundaries[1] - 1` was rejected by `Corpus::sources`. One placement rule now strides both held-out partitions (`CorpusTicker::owns_partition_targets`), and an empty calibration population is a hard load error. Measured on the real corpus at the production geometry: **433,721 origins**, 83,274,432 target bars, 38,190 distinct timestamps, 4,498 distinct tickers, mean cross-sectional width 11.36 (widest timestamp 1,548 tickers) — the same size to within 0.1% as the 433,303-origin validation partition. Calibration targets stop `purge` (192) bars short of `boundaries[1]` while the first validation origin is `boundaries[1] - 1`, so the two blocks are disjoint by corpus construction and no consumer does split arithmetic. Nothing in a training run reads this partition: gradients see only training targets, and checkpoint selection, early stopping, the held-out sample draw, the cross-section draw and the candle draw are all functions of `validation_refs` alone. That is what makes it the block a post-hoc calibration can be fitted on without inheriting the selection its scoring population performed. It is NOT in `CorpusContract`: the population is a pure function of the boundaries, purge, geometry and ticker set the contract already carries, so `Pairing::corpus_sha256` already pins it, and adding a derived field would make every authenticated checkpoint manifest on disk fail `corpus.contract == manifest.data`.

## Throughput

Batch 256, bf16 activations, fp32 master weights, flash SDPA, NorMuon with five-step Polar Express orthogonalization. Transformer block matrices (`block_*`) use NorMuon at 0.023; the patch embedding, covariate projection, heads, norms, and biases use AdamW at 0.008. Weight decay, schedule (constant to 40% of planned steps, linear cooldown to 0.15), and momentum warmup/cooldown follow `../modded-nanogpt/train_gpt.py`. `--learning-rate` sets the AdamW base and scales NorMuon by 0.023/0.008. The optimizer step is CUDA-graphed. No gradient accumulation or model chunking. The corpus uses memory-mapped bar files, parallel batch construction, pinned host memory, and a bounded prefetcher.

The schedule's endpoint is `schedule_budget(args, steps_per_epoch)`: `--schedule-budget N` where
stated, `--max-steps N` where a cap is set without an explicit budget, and `steps_per_epoch *
epochs` where neither is. A cap therefore ANNEALS by default. It did not before: the cooldown
occupies the last 60% of the budget, so at the production geometry (9,590 steps per epoch) it
starts at step 3,836, and six capped 2,500-step arms spent every step at rate multiplier 1.0
without ever measuring an annealed weight state. `--max-steps 4000 --schedule-budget 9590` still
runs the first 4,000 steps of a 9,590-step arm's own trajectory, which is the shape a
step-matched cross-arm comparison needs; it is now stated rather than the silent default. The
resolved budget, the cooldown start, both flags and the planned length are printed before the
first optimizer step of every run.

Forward and backward ARE graph-captured in the training path. `runner.rs` arms the capture on the step after `CAPTURE_AFTER_STEPS` (5) warmup steps and every later step is one replay; the sampled step's phase breakdown reports `captured forward+backward replay` and NaN for the three eager phases, which is how a report tells you which body ran. This paragraph previously opened by asserting the opposite and then used that assertion to argue that capture was declined on purpose. It was wrong about the binary, and any conclusion that leaned on it needs re-reading: what the audit actually establishes is that capture is nearly FREE, not that it is absent. Both halves of the budget were measured at batch 256 on an idle device with `benchmark-timexer-segment --capture-audit` (job 5184, `timexer_segment_benchmark`'s capture scalars): the audit warms up on a side stream, calls `empty_cache` (reserved 18,912 MiB after warmup, 952 MiB after the release, 599 MiB of it live and pinned by the optimizer's already-captured bodies) and only then captures, so the private mempool it reserves - 17,854 MiB - is the working set's new home rather than a second copy of it. Reserved at capture end is 18,806 MiB of 32,116 MiB, which is 184 MiB BELOW the same run's eager peak reserved. The earlier "~15 GB on top of the global pool" accounting, and job 5116's `CUDA out of memory. Tried to allocate 282.00 MiB`, were both from the older ordering that captured without releasing first. What capture is worth on time: replay 167.38 ms against 166.96 ms eager, +0.25%, because the fusions removed six of eight rotation kernels, the QK-norm kernel and one of two activation kernels per layer, and the step is no longer launch-bound. What it is worth structurally is the reason to keep it: one launch per step instead of hundreds, and a FIXED input address, which is what lets the loader upload into a resident batch asynchronously instead of allocating one per step. Correctness is not an obstacle - the replay's objective differs from eager by 4.8e-4 worst over 20 steps against an eager-vs-eager null of the same order. What the step also keeps: no scalar crosses to the host inside a step (the objective's finiteness is accumulated on device as an indicator and read once per report interval with the interval's mean loss), which is why the host runs ahead of the device and the loop period is `max(device work, host batch service time)` rather than their sum.

## TUI reports

One vocabulary throughout, in series labels, chart titles and CLI output alike. **Splits**:
`training` (the running training-set estimate over the interval's batches), `held-out sample`
(the fixed fast window set scored every report interval, `--eval-origins`), `held-out full`
(the whole disjoint validation population, scored at epoch end), `held-out cross-section` (a
third fixed draw, scored every report interval, used by the trading family and nothing else),
`persistence` (the zero-forecast baseline with the √h scale prior).

`held-out cross-section` exists because `held-out sample` is a strided pick over a
ticker-major origin list: its 2,048 origins land on 2,048 different timestamps with one
ticker each, so every within-timestamp quantity — the cross-sectional IC, the decile
long/short, the utility family — needs an origin cohort of at least 20 names. The utility
census makes missing measurements visible. The held-out sample and its checkpoint-selection
role are unchanged. The cross-section draw is additive: whole timestamp blocks of 256
tickers, at up to 128 timestamps spanning the held-out period, fixed across steps and runs.
It is not a subset of the sample; only the trading diagnostics read it. Utility and other
horizon-indexed report files are snapshots of the latest evaluation, not saved step
trajectories. Compare only explicitly matched evaluation steps; the dedicated
`_horizon_steps` and `_horizon_steps_scaling` bases preserve forecasting trajectories.
**Spaces**: `market-neutral` (the β-adjusted
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
  window's market-neutral ratio for both splits, the SELECTION OBJECTIVE for both splits
  (`<split> selection objective (horizon-weighted close best-scale MSE ratio)`, the scalar
  `weights/best` and both patience rules are actually decided on), and a parity line at 1.0.
- `timexer_segment_loss` — `nats per bar`: training objective NLL, held-out objective NLL
  with the same horizon weighting, and separate unweighted forecast/persistence NLL curves.
- `timexer_segment_generalization_gap` — training objective NLL minus held-out objective
  NLL, never weighted training minus unweighted scoring. Training uses dense origins and
  held-out scoring final origins, so matching weights does not make their populations equal.
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
  for the Pearson `ρ` on `_signal`. This is a **gain**, not the `1 - ρ²` MSE ratio:
  target-mean effects enter through `var(y)/mean(y²)`. The three components sum to the close
  total gain. The all-channel total gain, the complement of the headline ratio, sits beside
  them. Close channel unless a series says otherwise.
- `timexer_segment_signal` — x = bars ahead, `correlation coefficient`: the
  information-coefficient family on the demeaned close forecast. Pooled Pearson, pooled
  Spearman rank correlation (ordinal ranks, no tie averaging), and the cross-sectional IC —
  the correlation computed within each evaluation timestamp across that timestamp's tickers,
  averaged over timestamps, with a naive iid ±1 standard-error band. These bands ignore
  dependence between timestamps and are not significance tests or calibrated confidence
  intervals. A timestamp contributes only with at least 20 valid tickers.
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
  (ii) Excluding the first bar: the bar `t+1` close to bar `t+h` close move, forecast by
  `ŷ_h - ŷ_1`, against that same delayed persistence — undefined at `h = 1`, where the move
  is identically zero. This is a forecast-difference diagnostic, not a strategy that can
  execute at the predicted `t+1` close. The rates chart also carries the top- and
  bottom-decile hit rates. Actual next-open endpoint payoffs are reported separately below.
- `timexer_segment_utility_*` — x = **h next observed bars**, over
  `1, 8, 16, 32, 64, 128, 192` filtered to `pred_len`, plus `pred_len` if absent.
  Every origin timestamp is an independent endpoint cohort. The realized ticker payoff is
  `exp(σ * raw_close_target_h - entry_log) - 1`, where `entry_log` is the next valid bar's
  observed open relative to the origin close. It includes raw market moves, rather than
  pretending that the model's residual target is a realizable hedged return.

  Seven fixed policies share gross notional budget 1: cash, equal long,
  signed equal-notional (`sign(μ)/N`), mean-ranked decile long/short,
  mean/std-ranked decile long/short, sign gated at `|μ| >= std` with inactive allocations
  retained as cash, and a diagonal quadratic log-return proxy
  `μ/(std² + μ²)` projected to gross no greater than 1. Here `μ = σ * forecast` and
  `std = σ * exp(log_scale)` in raw log-return units. Deciles allocate +0.5/-0.5;
  tied forecasts do not manufacture trades. The proxy is neither a Kelly optimum nor a
  claim of portfolio-optimal allocation. Predictive std is **conditional residual outcome
  dispersion**, not epistemic uncertainty or total raw market risk; the policies test how
  useful that learned scale is, not whether it estimates all investment risk.

  The fixed cost sweep is `0, 0.5, 1, 2, 5, 10` bps per transacted notional.
  Entry plus exit turnover is `Σ |w| * (1 + exp(realized entry-to-exit log return))`,
  accounting for price drift of held shares; mean net bps is mean gross bps minus
  cost times mean turnover. It is not a fixed four-times-cost deduction.
  Cost levels have separate panels to keep policy comparisons readable:
  - `_utility_payoff`, plus `_utility_payoff_cost_0p5`, `_cost_1`, `_cost_2`,
    `_cost_5`, `_cost_10`: gross and five net payoff panels, bps of initial capital.
  - `_utility_rate`, with the same five `_cost_*` suffixes: payoff divided by `h`,
    bps per **observed bar**, not calendar return or an investable capital schedule.
  - `_utility_breakeven`: signed mean gross bps / mean turnover, bps per transacted
    notional. Negative means no gross edge; zero turnover means undefined. This is
    descriptive cost sensitivity, not a deployment verdict or a selected policy.
  - `_utility_gross_exposure`, `_utility_net_exposure`: initial gross and signed net
    notional / initial capital, separately charted.
  - `_utility_active_fraction`: fraction of origin names with nonzero positions.
  - `_utility_turnover`: entry plus drift-adjusted exit notional / initial capital.
  - `_utility_payoff_std`, `_utility_worst_payoff`: descriptive dispersion and worst
    observed gross cohort payoff in bps, not confidence intervals or future risk bounds.
  - `_utility_census`: complete eligible cohort count and mean/minimum valid
    entry-plus-endpoint width at each horizon over **every** origin group, including
    incomplete and below-20 groups, plus the 20-name threshold. Eligibility requires
    available count = original count and original count >= 20; positions use original
    membership only. No complete observations means complete-cohort count zero and payoff
    gaps (NaN), not earned zero; observed widths can remain positive (e.g. 10-name cohorts).
    With observations, cash/abstention means zero payoff and exposure, and undefined
    break-even. No per-target reweighting hides a missing outcome.

  Cohorts can overlap. Individual tickers exit at their own h-th next observed close,
  **not a common UTC exit**. There is no overlap/capital schedule, borrow availability or
  fee model, market impact, compounding, annualization, or Sharpe. These snapshots do not
  establish deployment readiness. The terminal test stays locked; policies, horizons and
  costs are frozen before evaluation, never tuned or selected using held-out utility.
  New utility bases deliberately do not redefine historical residual-spread portfolio
  reports; old `_portfolio*` and `_cross_section_census` bases are retired from current
  writers and the TUI registry.
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
- `timexer_segment_calibration_gain` — `gain on the demeaned forecast (dimensionless; 1 =
  perfect)`: **written by every run, calibrated or not.** `train` and an uncalibrated
  `evaluate-timexer-segment` write the applied gain as exactly 1.0 at every horizon; that is a
  measured fact about what scoring multiplied by, not an unmeasured zero, and NaN here would
  mean the different thing "no calibration was fitted". `calibrate-timexer-segment` writes its
  frozen curve plus the fit block's own optimal gain and the untouched block's own optimal
  gain, with the two blocks' origin counts, their boundary timestamps, the purged origins and
  the fit's effective degrees of freedom in the title. Reading rule: the fit-block `β̂` is what
  the curve was smoothed from, the untouched-block `β̂` is what it had to predict, and a frozen
  curve that tracks the first but not the second is an over-amplitude that was not a stable
  property of the forecaster — a different diagnosis, not a calibration.
- `timexer_segment_calibration_effect` — `ratio vs close-anchored persistence (dimensionless;
  < 1 = skill)`: the untouched block's per-horizon close and four-channel MSE ratios before and
  after the frozen gain, against that block's OWN best-scale ratio. That oracle series is an
  in-population bound this calibration is not allowed to reach; the gap between it and the
  calibrated series is the price of having fitted the amplitude out of sample.

#### What a per-horizon mean gain does and does not move

A positive per-horizon gain rescales the anchored close coordinate and shifts the other three
OHLC coordinates by the same amount, so candle geometry and every within-timestamp rank are
preserved. Measured on the reduction itself, to 1e-4 or better: the within-timestamp IC and its
standard error, pooled Pearson and Spearman, close and top-decile hit rates, decile returns,
conviction spread and the demeaned gain `D = Cov²/(Var·mean(y²))` are all **invariant**. That
invariance is what makes the IC-invariance assertion in `calibrate` a real self-check: if a
gain moves a within-timestamp statistic, the implementation is broken, and the run refuses
rather than reporting.

The exceptions are named here because a calibrated arm's charts must not be step-matched
against an uncalibrated arm's on these bases: **`timexer_segment_tradable` and
`timexer_segment_tradable_rates` are affine in the forecast (`f - mid`, `f_h - f_1`) rather
than rank statistics of it, so a per-horizon gain moves them** whenever a bar's forecast
crosses that nonzero reference. Measured: the mid-anchor hit rate at h=1 moves 0.53125 →
0.55208 under a gain of 0.6. The close-anchored series in those same two panels are invariant;
it is only the mid-anchored and first-bar-excluded ones that move. Every quadratic quantity
moves by construction — NLL, MSE ratios, the cross term of the offset/demeaned/cross
decomposition, coverage — which is the entire point of applying a gain, and those are
comparable across arms only when both arms' gain panels are read alongside them.

### Benchmark reports (`benchmark-timexer-segment`)

`benchmark-timexer-segment --profile` writes these beside the training bases. It runs a
synthetic batch, so it measures kernels, never skill. It arms the SAME forward+backward
capture the trainer arms, on the same step, and its batch is materialized once outside the
timed window; before that correction its timed loop ran an eager forward and backward with
only the optimizer captured, and rebuilt a 114 MB synthetic batch on device inside every
timed step, so the step time it reported described neither the trainer's configuration nor
the trainer's inputs.

- `timexer_segment_benchmark` — `named units`: three step times that bracket the host path -
  from a pinned host batch (the production configuration), from a device-resident batch (the
  same kernels without the upload), and from the real corpus loader with `--corpus` - plus the
  two differences between them, which ARE the packed-block upload and the loader's cost.
  Then origins per second, allocator peaks, step matmul TFLOP with the achieved TFLOPS and the
  fraction of the measured bf16 GEMM peak, and the analytic activation-traffic bound with the
  achieved GB/s and the fraction of the measured copy peak; the achieved figures divide by the
  pinned-host step, because that is what a production step costs. Both peaks are measured in
  the same process as the best of eight CUDA-event rounds, not looked up.
- `timexer_segment_benchmark_phases` — `milliseconds`: the six synchronized phases of one step
  fed from the pinned host batch (H2D, forward backbone, forward head and loss, backward,
  captured replay, optimizer). Attribution, not throughput: the synchronization is the point.
  Once the capture is armed the three eager forward/backward columns are NaN and the replay
  column carries the whole body, exactly as the trainer's sampled step reports it.
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

### Host loader audit (`audit-timexer-segment-loader`)

CPU only, no device, no report file: `audit-timexer-segment-loader` loads the real corpus and
charges host batch assembly component by component - block allocation, the two removed costs
(`pin_memory`'s staging copy and the whole-row zero fill), the window gather, the marginal cost
of the f64 price logarithms, of the market level lookup, and of each auxiliary variate - then
the production total, at each `--batch-size`. It exists because the training loop's period is
`max(device work, host batch service time)`, which hides host cost until it exceeds the step:
the number this prints is what the host spends, not what the step costs.

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

Model flags: `--layers`, `--d-model`, `--heads`, `--ffn`, `--dropout`, `--min-history`, `--features`, `--seq-len`, `--pred-len`, `--patch-len` (`seq_len % patch_len == 0`), `--x0-lambdas`, `--horizon-loss`, `--horizon-mean`, `--amplitude-prior`, `--scale-coupling`, `--batch-size`, `--market-min-cross-section`, learning-rate and optimizer flags; run control: `--eval-every`, `--eval-origins`, `--preview-patience`, `--patience`, `--epochs`. When increasing history, increase `--common-context` to at least that length. Checkpoints are inference checkpoints; optimizer resume is not implemented.

`--amplitude-prior <λ>` (default `0`, the control) adds `R = (λ/(2H))·Σ_h w_h·(1/N_h)·Σ_b mask·(m - m̄_h)²` to the objective, on the σ-scaled close mean coordinate `m` the decoder already materializes and with the objective's own `w_h`. It penalizes the predicted mean FUNCTION's cross-sectional energy, which is why it cannot be undone the way a fixed output multiplier is: scaling an upstream weight by `k` scales `R` by `k²`. Because the NLL's mean gradient carries `1/s²` with `s ≈ σ√h` while the penalty carries no `h`, one scalar λ produces the horizon-increasing shrinkage `1/(1 + 4λh)` — flat where the measured `β̂` is already at or above 1, strong at the long end where it is `0.26`. At `λ = 0` no kernel runs, `step_cost` charges nothing, no `kernel_classes` entry is registered and the manifest serializes byte-identically to one written before the knob existed; at `λ > 0` it costs 22 fp32 passes over the `[rows, origins, 1, pred_len]` close slice (1.62 GB, 1.1% of the step's 143.4 GB) and `selection_criterion` names the λ so a penalized arm cannot be read as an unpenalized one.

`--horizon-loss <spec>` selects the objective's horizon weighting: `uniform` (the control, `w ≡ 1`), `inv-sqrt`, `inv`, or `cutoff:K`. Invalid spellings and `cutoff:0` are refused by the parser; a `cutoff:K` past `--pred-len` is refused by `ModelConfig::validate`, the first statement of `train`, before the corpus loads. `cutoff:K` costs the same as `uniform`: the head still emits all `--pred-len` horizons and only the loss is masked, so at `cutoff:32` of 192 4.4% of the step's 16.98 TFLOP (the head-output GEMM) and 11.9% of its 143.43 GB (the head geometry and NLL traffic) are spent on horizons the objective weights at zero. That is deliberate — every per-horizon diagnostic keeps reading the untrained end. The weighting itself costs 3 of the 278 passes over the `[rows, origins, 1, pred_len]` fp32 channel space, 0.22 GB or +0.15% of step traffic, and zero matmul FLOPs, identically in every mode.

`--scale-coupling <full|decoupled>` (default `full`, the control) decides whether the squared-error term's gradient into the MEAN carries the head's own predicted precision. Under `full` — the textbook Gaussian NLL — one quadratic `½·r²·exp(-2·ls)` serves both parameter groups, so on an origin the trunk has memorized the residual shrinks, the head answers with a smaller scale, and the mean's gradient weight RISES: a super-linear reward for memorization, largest exactly where memorization is cheapest, which is the long end of the horizon axis where consecutive windows are almost the same window. `decoupled` splits the term into `½·dot(r², w·m·(1/h))` for the mean, `½·dot(detach(r²), w·m·(1/h)·exp(-2·CAP·s))` for the scale and the unchanged `CAP·dot(s, w·m)`. The stationary point in the scale is identical to the original NLL's — detaching `r²` removes no `s`-dependence — and the mean's `1/h` is precisely the horizon-fixed factor the objective already carried, so `--horizon-loss` remains the only knob on that axis. This is deliberately not Seitzer et al.'s multiplicative β-NLL (arXiv:2203.09168), whose `detach(σ^{2β})` factor would also rescale the log term by `exp(2·CAP·s)` — `e^-8` to `e^+8` element by element — or by `h`, up to 192×.

The training loss VALUE under `decoupled` is not a likelihood: its two quadratic rows count the same squared error under two weightings. `Losses::nll` therefore reports the true NLL to the bit — the full coupling's own twelve terms, reduced gradient-free — while handing back the decoupled gradient, so held-out NLL, the selection scalar and every chart built on them stay comparable to the control's. The cost is the fused loss kernel: the split quadratic is not what the CUDA forward computes, so the mode runs `fused_kernels::reference::decoupled_loss_geometry`. Measured at the production shape (`rows = 256`, `origins = 375`, `pred_len = 192`, forward plus backward, best of ten after two warmups on the 5090): fused `full` 2.88 ms, composed `full` 14.36 ms, `decoupled` 14.74 ms — `+11.9` ms, about `+7%` of the 168 ms step, of which the split itself is `+0.4` ms and the rest is the un-fused chain. At `full` nothing moves: the kernel, the term count and the manifest bytes are the pre-knob ones.

```bash
mlq submit --name timexer-validation --max-parallel-runs 1 --time-limit 3h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh evaluate-timexer-segment \
  --checkpoint training/runs/timexer-universe/weights/best \
  --output training/runs/timexer-universe/gens/0
```

Amplitude calibration is post-hoc and leaves the checkpoint bit-identical. `calibrate` fits the
per-horizon mean gain on the corpus's RESERVED `[70%, 80%)` calibration partition — which took
part in neither training nor checkpoint selection — freezes the curve to a stamped artifact,
and scores the held-out full `[80%, 90%)` split before and after applying it. There is no split
knob: cutting the held-out split in half instead would leave both halves carrying the same
selection bias, because `weights/best` was chosen by minimizing the selection objective over
that whole split.
`Blocks::spanning` measures the separation on the realized timestamps and refuses the run if
the first scored origin is not strictly after the last bar any fit target reads.

```bash
mlq submit --name timexer-calibrate --max-parallel-runs 1 --time-limit 3h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh calibrate-timexer-segment \
  --checkpoint training/runs/timexer-universe/weights/best \
  --calibration training/runs/timexer-universe/mean-calibration.json \
  --output training/runs/timexer-universe/gens/0
```

The curve is TWO-SIDED. Shrinkage is applied as fitted; amplification is applied only as far as
the horizon's own estimation error proves it, `g_h = min(fitted_h, max(1, exp(ln β̂_h - 3·SE(ln
β̂_h))))` with `SE = 1/√weight` and `weight = n·ρ²/(1-ρ²)`. There is no clamp at 1 and no MSE
justification for one: the expected-MSE-optimal gain is `E[β]` whatever the sampling
distribution, so the ceiling comes from deployed-notional risk instead, which is why it is
one-sided. A horizon whose `β̂` is within three standard errors of 1, or which carries no
measurable amplitude at all, is pinned to the identity. `timexer_segment_calibration_moments`
carries `Var(f)` and `Cov(f,y)` as shares of the same persistence MSE, because a `β̂` above 1
is a real under-amplitude only if the variance under it is large enough for correcting it to be
worth anything.

`evaluate-timexer-segment --calibration <artifact>` then applies a frozen curve while scoring.
The artifact is stamped with the checkpoint's weight and manifest digests, its step, the head
format string, the objective, `pred_len`, and the corpus schema and universe digest; any other
pairing is refused, so a calibration can never be silently applied to a different checkpoint,
architecture or corpus contract. `CALIBRATION_FORMAT` is `...-v2-close-anchor-two-sided-...`, so
a `v1` shrinkage-only artifact fails format authentication rather than being reinterpreted.

## Matched context and feature campaign

`benchmarks/timexer_universe_campaign.py` queues complete-corpus, one-epoch comparisons (2,048-bar history, 6,000-bar history, 6,000-bar history with all exogenous variates) after an existing 96-bar reference job and selects the final configuration from held-out full MSE read through `report_cli`. Ties retain OHLC alone; the final context is 6,000 regardless of shorter-history results.

## Fixed-step temporal LeJEPA research

`train-timexer-segment --research-panel` uses this same CausalPatch backbone, full eligible
training population, BF16/FA4 path, captured optimizer, and captured forward/backward.
It completes exactly `--max-steps`; `--schedule-budget` must be zero or that same budget.
There is no early stopping, best-checkpoint selection, calibration fit, or terminal-test use.
The separate research checkpoint is not accepted by the production portfolio loader.

All `--jepa-mode` arms retain the forecast head:

| Mode | Backbone objective |
| --- | --- |
| `off` | Fixed-observable forecasting only |
| `latent-one` | Attached future-observation prediction at one patch + population SIGReg |
| `latent-multi` | Attached future-observation prediction at 1, 2, 4, 8, 12 patches + population SIGReg |
| `anchored` | Multi-horizon latent objective + attached forecasting anchor |
| `anchored-no-sigreg` | Anchored objective without SIGReg |
| `anchored-reconstruct` | Anchored objective + current normalized OHLC-patch reconstruction |
| `anchored-projected` | Anchored latent objective on a separate D→D GELU→D target projector |
| `anchored-projected-no-sigreg` | Same projected targets without SIGReg |
| `anchored-projected-small` | Same projected SIGReg objective with 16-dimensional targets/predictions |
| `anchored-conditional` | Forecast anchor + fixed future-return conditional characteristic prediction; no SIGReg |

In latent-only arms, the forecast head trains on detached states: it measures an online
readout, not an implicit forecasting anchor. The raw observation is always the shared
full-width patch embedding entering the causal trunk. Original latent modes predict and
regularize it directly; projected modes use a disposable nonlinear target projection outside
the forecast path. Those learned future targets remain attached. Conditional mode instead
uses fixed data-derived future-return features, with no learned target branch. Where enabled,
SIGReg operates across batch rows at each sampled time view; it never pools correlated time
positions and does not Gaussianize causal states or predicted conditional means. Directions
and views refresh through an independent RNG into graph-resident buffers; conditional mode
does not allocate or refresh them. Gaussian geometry alone does not establish predictive sufficiency.

The matched campaign requires `--future-calendar false`: timestamps of future *observed*
bars can reveal gaps or halts and are not known at the decision. Historical calendar and
other requested exogenous features remain available. The shared baseline is context 6000,
patch 16, horizon 192, width 512, eight layers, all features, disabled x0 injection,
decoupled mean/scale gradients, and lattice supervision.

### Sampling, probes, and checkpoint authentication

`prepare-jepa` authenticates the corpus and writes the fixed research panel without loading
a model. Training still shuffles the entire eligible training pool. The bounded validation
and train-only probe-fit draws are deterministic, ticker-balanced and time-stratified,
without replacement. This is a ticker-balanced diagnostic estimand, not a full-population
row-weighted score; the manifest includes zero-coverage tickers, masks, timestamps and hashes.
The common probe source must be strictly later than the latest target used anywhere in
backbone training. Its lookback follows the actual patch boundary, including non-patch-aligned
forecast horizons. Invalid cached UTC split edges are rebuilt rather than trusted.

Frozen CUDA ridge probes compare direct patch input, observation embedding, a recomputed
four-patch recent state, full-history state, and each available matching-horizon predicted
latent. Learned representations retain their full model width; there is no random projection.
The direct-input control has its actual patch-input width and is labeled separately.
Future-close-return probes and same-time OHLC reconstruction answer different questions.
Ridge penalties use a target-purged chronological holdout inside the training panel, followed
by a training-only refit. Validation never chooses probe penalties. The realized purged fit
population is checked before training; insufficient rows are an error, not silent subsampling.
Explicit cache limits are 4096 fit origins and 2048 scored origins.

After endpoint probes succeed, training writes `weights/jepa.safetensors` and the authenticated
`weights/jepa-manifest.json`. `evaluate-jepa --run-root ... --output ...` verifies the weights,
manifest, regenerated sample plan and corpus, then reproduces forecasting and frozen probes.
Research metrics use `timexer_segment_jepa_*` `.report.bin` bases registered for the TUI and
readable with `report_cli`; existing horizon and trading reports are also emitted.

`benchmarks/lejepa_campaign.py plan` pins the executable, report CLI, driver and expected data
contract. It captures an explicit runtime environment allowlist or inherits the authenticated
reference campaign's environment; repeated `--env NAME=VALUE` supplies nonsecret overrides.
`submit --plan ...` queues **one exclusive normal-priority job per model**, followed by a
lightweight after-success collector. The default per-model watchdog is 420 seconds plus
30 seconds of termination grace; this is a failure bound, not a replacement for fixed updates.
`follow --plan ...` observes existing jobs; an observer timeout never resubmits them.
Submission recovery retains the original queue-client environment and idempotency key.
Completion requires verified exact-step endpoints, identities and binary forecast reports.

Named suites are `objective-comparison`, `forecasting-controls`, `matched-comparison`,
`sigreg-placement`, `sigreg-dimensionality`, and `temporal-conditional`.
The baseline is explicitly `decoupled-lattice-forecast`; `full-none-forecast` changes both
mean/scale coupling and supervision decimation, not the architecture or eligible population.
For example, `--suite forecasting-controls --select full-none-forecast --reference-plan OLD/plan.json`
trains only the missing control and reuses completed matched endpoints. Reference ancestry
is authenticated transitively so extending an extended comparison retains its original baseline.
Old one-job campaign snapshots remain immutable historical records; new plans use the
per-model protocol.

### Controlled-history evidence

`benchmark-jepa-memory` trains all six objectives on relevant-cue and irrelevant-cue tasks.
Paired examples have identical recent prices, covariates and normalization anchors but
opposite distant cues; the relevant cue changes a known future law. The price pulse returns
to zero at the context endpoint, preventing global centering from leaking the cue.
Reports include Bayes-reference error, regret, paired effect magnitude/sign, the irrelevant-cue
null, and exact input equality. This synthetic intervention supplies counterfactual ground
truth; arbitrary real-market history swaps do not.

Job 8414 (`benchmark_results/lejepa-controlled-history-20260919`) completed 1024 updates per
arm/task with a two-layer, width-128 model, batch 64 and 256 validation pairs. At horizon 64:

| Objective | MSE / persistence | Paired effect / known effect |
| --- | ---: | ---: |
| Forecast only | 0.18325 | 0.9988 |
| One-horizon latent | 0.33990 | 0.5620 |
| Multi-horizon latent | 0.20464 | 0.8837 |
| Anchored | 0.18386 | 0.9857 |
| Anchored without SIGReg | 0.18358 | 0.9870 |
| Anchored + reconstruction | 0.18492 | 0.9674 |
| Bayesian reference | 0.18209 | 1.0000 |

Recent-input differences were exactly zero at every evaluation. All arms had correct cue
effect signs; irrelevant-cue MSE ratios were 0.9997–0.9998. These results support the value of
multi-horizon prediction and a forecasting anchor on this task, not a market-performance claim.

The full-shape timing run 8437 (`benchmark_results/lejepa-throughput-lifetime-fix-20260919`)
measured 180.52 ms per captured update with the real corpus loader at batch 256. Five paired
forward/backward alternations measured 145.13 ms fused versus 157.50 ms composed, a 12.37 ms
saving against 1.35 ms fused-arm spread. Eager comparisons run before training-graph capture
so their activations do not compete with its 21.4 GiB private pool. These are throughput
measurements, not learning evidence.

### Matched market result: 1400 updates

Campaign job 8444 completed all six authenticated endpoints at batch 256, seed 20260919,
1400 updates and a matching 1400-update schedule. Each model process, including startup,
four evaluations, frozen probes and checkpointing, took 233–268 seconds. The queue displayed
one 25m33s job because it contained six sequential runs; this is not a per-model runtime.
The corpus contained 4873 eligible tickers and 2455276 training origins. Each arm used the
same 2048-origin ticker-balanced validation panel and 2048-origin train-only probe panel.
Artifacts: `benchmark_results/lejepa-campaigns/lejepa-fixed1400-20260919/plan.json` and
`training/runs/lejepa-fixed1400-20260919-{forecast,latent-one,latent-multi,anchored,no-sigreg,reconstruct}`.

| Arm | Neutral OHLC MSE / persistence | Fixed-horizon close MSE / persistence | Signed cross-sectional IC h64 | Wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Forecast only | 0.997576 | 0.978749 | 0.01723 | 232.60 |
| One-horizon latent | 0.998009 | 0.999352 | 0.03525 | 237.05 |
| Multi-horizon latent | 0.998833 | 0.999985 | 0.01824 | 266.92 |
| Anchored | 1.000274 | 1.000185 | -0.01461 | 260.15 |
| Anchored without SIGReg | 0.994165 | 0.987293 | 0.03318 | 268.13 |
| Anchored + reconstruction | 0.995588 | 0.982099 | 0.03309 | 268.12 |

The fixed-horizon score averages the predefined 1, 8, 16, 32, 64, 128 and 192-bar close ratios;
it does not choose a horizon after seeing results. Forecast-only also had the best Gaussian
NLL, 2.07101 versus persistence 2.36862. Without SIGReg was the strongest JEPA candidate on
aggregate OHLC MSE, while forecasting alone remained strongest on the fixed-horizon close
score and NLL. These one-seed, short-budget development-panel results do not establish
statistical significance, economic value, or terminal-test superiority.

Frozen full-state return probes stayed close to persistence: h64 ratios ranged from
0.999668 to 1.000019 and h192 from 0.998727 to 1.000242 across arms. They do not establish
strong linearly accessible long-memory alpha. Reconstruction improved the anchored full-state
mean coordinate error ratio from 0.3563 to 0.2015, but forecast-only was already 0.1962.
These reconstruction averages exclude the identically zero normalized endpoint-close
coordinate; its zero-baseline ratio is undefined. Better reconstruction is not proof of
better future prediction. Retain forecast-only as the control and anchored-without-SIGReg
as the leading research candidate; do not promote the default SIGReg objective on this evidence.

### Direct accuracy comparison against named baselines

`compare-timexer-accuracy` scores saved checkpoints without training, fitting probes, selecting
checkpoints, or refitting gains. An authenticated research reference supplies the exact
validation and synchronized cross-section panels; every candidate must match the corpus and
source/target geometry. Matched research models must additionally match seed, batch size,
completed updates, schedule and optimizer settings. Objective and coupling/decimation
treatments may differ. Legacy checkpoints retain their authenticated frozen gain and are
explicitly labeled **NONMATCHED historical; calibrated; future-calendar**. Both scored
panels must begin strictly after every included historical calibration's final target.

```bash
mlq submit --name paired-forecast-accuracy --max-parallel-runs 1 \
  --max-attempts 1 --time-limit 5m --cwd "$PWD" -- \
  ./torch-env.sh target/release/trading_bot_0 compare-timexer-accuracy \
  --reference-run training/runs/lejepa-fixed1400-20260919-forecast \
  --research-run decoupled-lattice-1400=training/runs/lejepa-fixed1400-20260919-forecast \
  --research-run jepa-no-sigreg-1400=training/runs/lejepa-fixed1400-20260919-no-sigreg \
  --research-run full-none-1400=training/runs/lejepa-full-none1400-20260920-full-none-forecast \
  --legacy-checkpoint historical-decoupled-lattice-2500=training/runs/timexer-decoupled-lattice-2500/weights/preview-latest \
  --output benchmark_results/paired-forecast-accuracy
```

Repeat `--research-run NAME=PATH` and `--legacy-checkpoint NAME=PATH` for additional models.
Naming the reference again does not rescore it. Output must be fresh. The 25
`timexer_accuracy_*` report bases contain actual close/delayed-return error, conditional
directional hit rates, signed pooled Pearson and cross-sectional IC, neutral/raw OHLC error,
Gaussian NLL, predictive coverage, baseline deltas and sample/IC populations. Latent JEPA
loss is deliberately absent. `accuracy-protocol.json` records identities, comparability and
panel hashes, not a second metric channel. Ratios below one beat persistence. Hit rates
exclude zero forecasts and zero targets; delayed metrics exclude the first future close
and are not execution P&L. Gaussian NLL omits the same model-independent constant as training.

Job 8480 compared nine checkpoints on 2048 identical validation origins plus 4000 identical
cross-sectional origins (100 eligible timestamps at h64), in 26.845 seconds.
Reports: `benchmark_results/accuracy-decoupled-comparison-20260920`.
The new full/none control completed 1400 updates in 237.02 seconds using the exact binary
of the previous six-run campaign. Job 8476's post-run verifier initially rejected omitted
Serde defaults for full coupling/no decimation after training had succeeded. The validator
was corrected and the checkpoint independently reverified without retraining; its original
failed lifecycle remains recorded. Reverification provenance is in
`benchmark_results/accuracy-decoupled-comparison-20260920-assets/control-endpoint-reverification.json`.

| Matched causal model, 1400 updates | Mean predefined close MSE ratio | Close hit h64 | Signed IC h64 |
| --- | ---: | ---: | ---: |
| Decoupled + lattice, forecasting only | 0.978749 | 51.90% | 0.01723 |
| Full coupling + no decimation, forecasting only | 0.975143 | 52.39% | 0.03563 |
| JEPA one-horizon | 0.999352 | 51.86% | 0.03525 |
| JEPA multi-horizon | 0.999985 | 50.00% | 0.01824 |
| JEPA anchored | 1.000185 | 50.93% | -0.01461 |
| JEPA anchored without SIGReg | 0.987293 | 52.20% | 0.03318 |
| JEPA anchored + reconstruction | 0.982099 | 52.05% | 0.03309 |

The full/none control is the strongest measured short-budget aggregate close-error baseline;
JEPA has not established a general accuracy improvement. Horizon dependence remains:
at h192, anchored without SIGReg scores 0.997561 versus full/none 1.000512 and decoupled/lattice
1.001884. Do not select the horizon after observing these results and call that a global win.
IC standard errors across timestamps at h64 are roughly 0.015–0.019 for these arms; this
single-seed development panel does not establish significant ranking improvements.

For historical reference on the **same scoring observations**, decoupled/lattice-2500 has
mean close ratio 0.978768, h64 close ratio 0.985270, h64 hit 51.76% and IC 0.04870.
Historical full/none-2500 has mean close ratio 0.995667, h64 close ratio 0.995200, h64 hit
51.03% and IC 0.05353. Their 2500-update schedule, different seed, calibrated means and
future-observed-calendar inputs prohibit attributing differences from the causal 1400-update
runs solely to architecture or objective. No historical manifest was rewritten to force
compatibility, and the terminal test remains locked.

### Temporal SIGReg diagnosis

`diagnose-temporal-sigreg --run-root RUN --output FRESH --batch-size 256` authenticates the
saved checkpoint/corpus/panel and performs no parameter update. Queue it exclusively through
`mlq`. It uses all 2048 validation rows for covariance, and the first predetermined 256 rows
for patch-weight/bias derivatives, at the common source whose decisions follow every training
target. Positions remain separate populations; time is never flattened into independent rows.
Metrics are registered `timexer_segment_sigreg_*` binary reports, with provenance beside them.

Source audit against `../lejepa/MINIMAL.md` and its Epps–Pulley library found the same
unit-sphere projections, doubled-half-interval quadrature and population-N multiplier.
The N multiplier is intentional. The finite IID Gaussian null has expected statistic
**1.052464**, not zero; correlated market rows need not follow that IID null.

The important differences are structural and statistical:

- Reference LeJEPA uses a separate nonlinear projector and same-image augmented views.
  The original temporal objective applies SIGReg directly to the affine patch embedding
  consumed by the forecast trunk. With all features, that map is 416→512; one input is the
  identically zero endpoint close, so full I512 covariance is impossible in exact arithmetic.
  Finite random projections can nevertheless approximate normality without full rank.
- Future temporal views introduce genuinely unpredictable information. Their price patch is
  centered on its own future endpoint, while calendar/other auxiliary coordinates and shared
  prefix-volatility information remain. This target is not the source-to-future return.
- Gainless RMSNorm approximately removes common input scale. Shrinking latent MSE without
  SIGReg does not itself show lost forecasting information; increasing marginal variance
  with SIGReg does not show more useful information.

Completed jobs **8489–8492** inspect the original forecast, anchored, no-SIGReg and
reconstruction endpoints. Reports are under
`benchmark_results/temporal-sigreg-diagnosis-20260920-{forecast,anchored,no-sigreg,reconstruct}`.
At the common source:

| Model | Observation participation rank | State participation rank |
| --- | ---: | ---: |
| Forecast-only | 3.061 | 14.022 |
| Anchored SIGReg | 23.486 | 5.915 |
| Anchored without SIGReg | 4.715 | 7.217 |
| Anchored + reconstruction | 24.510 | 8.780 |

On anchored, the configured-weight SIGReg patch derivative is **6.01×** the forecast-training
derivative, but its cosine with the predefined mean close-error-ratio derivative is **+0.03465**.
Its radial activation-gradient energy share is only **1.44%**. Thus strong, largely unrelated
geometry pressure is supported; a story of strongly opposite accuracy gradients or merely
radial scale chasing is not. These are endpoint derivatives on one held-out cohort, not a
reconstruction of historical Adam updates or proof of the entire training trajectory.

At h64, anchored latent target variance is **0.95579**, prediction variance **0.18853**,
their covariance **0.18532**, and latent MSE **0.80266** versus persistence **1.47549**.
The predictor captures real variation in its own target without delivering good price
accuracy. This does not prove which nuisance coordinate it uses. The same-mask identity
`MSE = Vtarget + Vpred - 2*covariance + squared_mean_bias` is reported directly; do not call
`Vtarget - Vpred` innovation variance unless conditional-mean orthogonality is established.

The `sigreg-placement` campaign tests `anchored-projected` against
`anchored-projected-no-sigreg`: a disposable D→D GELU→D target projector, with unchanged raw
forecast input, target dimension, attached targets and loss weights. A common improvement
belongs to the projection architecture; only the within-pair difference measures SIGReg.


The completed placement pair (jobs 8494/8495, comparison 8497) gives predefined close scores
**0.984327 with SIGReg** and **0.985620 without**, versus the original direct-input pair
**1.000185 with** and **0.987293 without**. The sign of the measured SIGReg effect reverses.
Runs took 276.06s and 280.62s; reports are in
`benchmark_results/accuracy-sigreg-projection-20260920`.
Neither beats the 0.978749 decoupled/lattice or 0.975143 full/none forecast-only controls.

Follow-up diagnostic jobs 8504/8505 show the unregularized projector and predictor have
zero measured population variance, with h64 latent MSE **3.03e-8**. The forecast state still
has participation rank **7.43**: auxiliary collapse is not whole-model collapse. Projected
SIGReg avoids that collapse (target rank **5.46**, h64 target variance **0.9980**, covariance
with prediction **0.6226**, latent MSE **0.3586**). Its patch gradient remains **13.19×**
the forecast-training gradient; the projector does not simply absorb all regularization
pressure. A separate `anchored-projected-small`/`sigreg-dimensionality` control reduces
the auxiliary target/prediction width to **16**, while retaining the D512 forecast backbone,
D512 projector hidden layer, weights, masks and horizons. This matches the output-dimensional
scale of the local LeJEPA example without bundling a BatchNorm or loss-weight change.

The subsequent `anchored-conditional` candidate predicts fixed characteristic features of
source-anchored future neutral returns, normalized by source sigma and square-root horizon.
Its target is interleaved cos/sin at frequencies 0.25, 0.5, 1, 2 and 4. Under squared feature
error, the population optimum is the conditional characteristic vector, not an isotropic
state or a unit-variance conditional mean. Fixed targets prevent encoder-driven target
collapse and reward persistent nuisance only insofar as it predicts the declared outcomes.
Five frequencies identify those moments, not the full distribution; volatility skill need
not improve point accuracy. Forecast accuracy remains the acceptance criterion.

### Completed temporal objective search

All five new runs use the same full-corpus 1400-update/B256/seed20260919 protocol, BF16
backbone and captured training. The terminal test remains untouched. Each ran as its own
exclusive normal-priority queue job and completed in under five minutes.

| Treatment | Mean predefined neutral close MSE ratio | h64 directional hit | h64 signed IC | Wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Existing full/none forecast-only | **0.975143** | **52.39%** | 0.03563 | 237.02 |
| Existing decoupled/lattice forecast-only | 0.978749 | 51.90% | 0.01723 | 232.60 |
| Projected D512 SIGReg | 0.984327 | 52.10% | 0.03967 | 276.06 |
| Projected D512 without SIGReg | 0.985620 | 49.51% | 0.03507 | 280.62 |
| Projected D16 SIGReg | 0.994632 | 50.49% | 0.01979 | 247.11 |
| Conditional return CF, decoupled/lattice | 0.980716 | 50.54% | 0.04137 | 243.97 |
| Conditional return CF, full/none | 0.980497 | 51.71% | 0.02468 | 242.36 |

The D16 target has participation rank **10.13/16**, h64 latent target variance **1.0437**,
prediction covariance **0.8358**, and latent MSE **0.2136**. Those diagnostics improve over
the D512 projected target while price accuracy deteriorates. Target dimensionality alone
is not the missing solution.

The fixed-return CF auxiliary has genuine held-out skill against its frozen training-mean
baseline. Error ratios at 16/32/64/128/192 bars are
**0.9248/0.9332/0.9511/0.9634/0.9672** with decoupled/lattice, and
**0.9238/0.9360/0.9521/0.9653/0.9670** with full/none. These are binary
`timexer_segment_jepa_conditional_cf_{error,ratio,count}` reports in the run directories.
The empirical mean is fitted only on the already-authenticated inner+holdout training
cache; validation never fits a baseline. No second model pass is required.
Nevertheless both actual forecast scores are worse than their corresponding forecast-only
control, and both h128/h192 close ratios exceed one. Conditional-distribution predictability
is not equivalent to better point forecasts; these aggregate feature scores do not isolate
whether the gain is in direction, volatility or another moment.

At the seven predefined horizons, full/none also retains lower mean neutral OHLC error,
raw OHLC error and NLL than these five new treatments. Conditional full/none nearly matches
its NLL (**1.450305 vs 1.450276**) but does not recover its close accuracy. Horizon-specific
IC differences remain descriptive, not significance or economic-value claims.

**Decision:** retain full/none forecasting-only as the measured aggregate leader and
decoupled/lattice as the mandatory baseline. Do not promote a temporal auxiliary based on
normality or prediction of its own targets. The evidence establishes a harmful transfer
design (direct affine-input regularization), and shows that repairing placement, reducing
target width and predicting fixed conditional outcome features are insufficient to beat
the controls in this representative budget. It does not prove that temporal SIGReg cannot
work, identify every nuisance component, or establish a global optimum.

Final shared-panel evidence: `benchmark_results/accuracy-temporal-sigreg-final-20260920`
(job **8523**, 14 checkpoints, **25.945s**). Immutable plans are
`benchmark_results/lejepa-campaigns/{sigreg-projection1400-20260920,temporal-conditional1400-20260920,sigreg-small1400-20260920}/plan.json`.
Frozen diagnostic jobs **8489–8492, 8504–8505, 8524** succeeded. Cargo checks/release builds,
19 focused CPU/CUDA regressions (job **8520**) and the TUI chart-discovery regression passed.
Review found a reconstruction-only assertion in the expanded capture test; it now checks
the appropriate objective. An earlier projector test fixture also incorrectly randomized
only one compared trunk before checking initialization; moving that intervention after
the initialization comparison fixed the test without changing any trained model.
