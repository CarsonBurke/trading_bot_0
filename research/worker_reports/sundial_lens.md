# Sundial lens — Liu et al. 2025, "Sundial: A Family of Highly Capable Time Series Foundation Models" (arXiv 2502.00816v4, ICML 2025)

Source: `research/papers/sundial_2025.pdf` (all 23 pages). Where the paper is silent I cite the released HF code (`thuml/sundial-base-128m`: `config.json`, `modeling_sundial.py`, `flow_loss.py`, `ts_generation_mixin.py`) and mark it **[code]**. My own inferences are marked **[inference]**.

## 1. Architecture

**Pre-training paradigm (§4, p.4).** Univariate ("S3 format" of Timer; multivariate data flattened to single series). Per-variable normalisation offline; varying-length training samples with **max context 2880**. Model must handle any input length at inference.

**Tokenisation (§4.1.1, p.4).**
- *Re-Normalization*: "stationarization (Liu et al. 2022), a non-parametric two-stage instance normalisation conducted within each sample" — i.e. subtract context mean, divide by context std, de-normalise outputs. Motivated by temporal distribution shift and outlier ranges for zero-shot. **[code]** `generate()` uses mean/std (+1e-5) of the *entire* input; training `forward(revin=True)` floors std at 1e-2 → 1.0. Note the stats are of the whole context, so during dense per-token training every token's target is scaled by full-window statistics (non-causal w.r.t. that token) **[inference from code]**.
- *Patch embedding*: patch length **P = 16**; left-pad to a multiple of P; a binary mask m_i ∈ R^P per patch marks padded positions; shared MLP R^{2P}→R^D on Concat(x_i, m_i) (Eq. 4). N = ⌈T/P⌉ tokens (≤180 at T=2880). Continuous values — no quantisation (contrast Chronos). **[code]** embed = Linear(32→D_ff) SiLU Linear(D_ff→D) + residual Linear(32→D), dropout 0.1.

**Backbone (§4.1.2, p.4).** Decoder-only, **Pre-LN** (for stability), causal self-attention with **RoPE** (Eq. 5), **FlashAttention** and **KV cache**. **[code]** LayerNorm (not RMSNorm), SwiGLU FFN with SiLU, q/k/v bias, no o-bias, RoPE θ=10000, final LayerNorm, dropout 0.1, init N(0, 0.02).

**Sizes (Table 5, p.14).** All P=16, context 2880, F ∈ {16, 720}:
| Model | Layers | D / D_ff | Heads | FM-Net (D_tf, L_tf) | Params |
|---|---|---|---|---|---|
| Small | 6 | 512 / 2048 | 8 | (512, 3) | 32M |
| Base | 12 | 768 / 3072 | 12 | (768, 3) | 128M |
| Large | 24 | 1024 / 4096 | 16 | (1024, 6) | 444M |

**TimeFlow loss (§4.1.3, p.5; §3.1 p.3).**
- Goal: at every position i, predict the next **F** values ŷ_i = x_{1+iP : F+iP} (**multi-patch prediction, F > P**, "reduces the steps of autoregressive inference"; motivated by TimesFM's observation that larger output patches help decoder-only models). F = 720 for TSLib/GIFT-Eval, F = 16 for FEV (App. B, p.13). Longer horizons → rolling forecast; shorter → truncate.
- Conditional flow-matching with the conditional OT (linear) path and Gaussian source: y_i^{(t)} = t·y_i + (1−t)·y_i^{(0)}, y_i^{(0)} ~ N(0, I_F), t ~ U[0,1] (Eq. 6). Velocity net u_θ(y^{(t)}, t | h_i) = FM-Net(y^{(t)}, t, h_i) (Eq. 7), h_i (last-layer token state) enters via **AdaLN** (DiT-style) as a time-invariant condition. Loss summed over all N tokens (Eq. 8) — dense next-patch supervision like an LM.
- FM-Net = "a small MLP" (D_tf, L_tf as above). **[code]** it is MAR's `SimpleMLPAdaLN` (Li et al. 2024): input_proj F→D_tf; timestep sinusoidal embed (t×1000) + cond_embed(h) summed → y; L_tf ResBlocks {LN → modulate(shift,scale) → MLP D_tf→D_tf→D_tf (SiLU) → gated residual}; FinalLayer LN+modulate+Linear→F. AdaLN modulation layers and output layer **zero-initialised**.
- **[code] The released loss is x1-prediction, not velocity prediction:** `loss = Σ_h w_h (net(y^{(t)}, t, h_i) − y_i)^2` with **w_h = 1/h** for h = 1..F (horizon-decaying weights), masked, and `diffusion_batch_mul = 4` — each (token, target) pair is duplicated 4× with independent (t, ε) draws per optimiser step (MAR trick). Sampling: `x ← x + (net(x, t, h) − ε_0)·dt` with the *initial* noise ε_0 — equivalent to velocity x1−x0 on the straight path.
- Inference (Alg. 1, p.5): Euler, **K uniform steps, K = 50** (App. B). Repeat with different initial noise; report **median and quantiles** of samples (Chronos convention). The lookback condition is computed once and shared across samples ("efficient repeated sampling"). **[code]** only the last token's h is used at inference; in rolling AR the **median across samples** is fed back as the next input (which is why App. E warns multi-step AR can over-smooth).

**Objective vs MSE / mode collapse (§2.1 p.2, §3.2 p.3, §5.3 p.7-8, App. C.1 p.13-14, Fig. 14-15).** Argument: MSE / quantile / parametric-density heads *specify* a prior (unimodal Gaussian for MSE); on heterogeneous corpora where "a similar lookback goes into divergent trending", the MSE optimum is the conditional mean → over-smooth predictions; they call this mode collapse "in representation learning". Generative modelling (flow) learns the conditional distribution without a prior and, they claim, "enhances representation learning of foundation models" (§2.3). Evidence: Table 3 / Table 7 (see §4 below) plus qualitative showcases. Diffusion (DDPM-style, Li et al. 2024) was tried and was worse than flow-matching.

## 2. Data — TimeBench (§4.2 p.5, App. A + Table 4 p.13)

1032B time points ("one trillion"). Composition (Table 4):
| Source | Points | % |
|---|---|---|
| Chronos datasets | 94B | 9.11 |
| ECG (PhysioNet) | 48B | 4.65 |
| **Finance (theirs)** | 10.5B | 1.02 |
| IoT (theirs) | 5.8B | 0.56 |
| LOTSA | 230B | 22.29 |
| Synthetic (KernelSynth) | 0.5B | 0.05 |
| ERA5 3h / 12h / daily / weekly / monthly / quarterly | 129 / 32 / 406 / 58 / 13.5 / 4.5B | 12.5 / 3.1 / 39.35 / 5.62 / 1.31 / 0.44 |

- ~62% is ERA5 reanalysis weather (chosen "because of the predictability of weather systems"). Finance is 1% and undescribed (frequency, instruments, fields not stated). No mention of OHLC or stock data beyond "finance".
- Curation: missing-value imputation, abnormality exclusion, normalisation; series characterised by non-stationarity, forecastability, seasonality "which affects the training stability of next-token prediction" (App. A). Non-stationarity is otherwise handled by the ReNorm above.
- Sampling: domain-weighted "predefined ratio", global shuffle via parquet, variable-length windows ≤2880 (§4, App. B). No augmentation beyond KernelSynth; no sequence packing mentioned.
- App. E: TimeBench "contains many middle- and low-frequency time series"; "performance on very high-frequency data is not guaranteed".
- Data-scaling (Table 8, p.15): same model on 94B → 230B → 1032B: ETTm1 MSE 0.367 → 0.352 → 0.336; ECL 0.172 → 0.171 → 0.169. Monotone but modest.

## 3. Training recipe (App. B p.13; §5.2; §5.6)

- PyTorch, **32× A100**, **AdamW**. **Learning rate, schedule, batch size, total steps, warmup: not stated anywhere.** Fig. 9(b) mentions 15k vs 30k iterations for the Pre-LN/Post-LN ablation, so runs are at least ~30k iterations. Model card: **FP32** precision. Dropout 0.1 **[code]**.
- Training curves (Fig. 6, §5.2): loss decreases with size; Large converges 15.38% lower than Small.
- Throughput tricks (§4.1.2, Fig. 9c-d): FlashAttention −14.8% memory; KV cache −43.6% inference time; no accuracy change. Multi-patch prediction (F=720 from one token) removes AR steps; shared condition across samples.
- Inference cost (§5.4, model card): 20 samples × 50 steps ≈ 1 s on CPU; M1 Pro: 2880→720, 1 sample 510 ms, 20 samples 949 ms; FEV: 35× faster than Chronos (Fig. 5), near N-BEATS.
- Fine-tuning (§5.5, Fig. 8): tuned once on all FEV datasets beats zero-shot, which beats training from scratch on FEV.

## 4. Evaluation

**Benchmarks (§5.1).** (1) TSLib long-term point forecasting, MSE/MAE at {96,192,336,720}, context 2880, outputs truncated from F=720 (Tables 1, 9). (2) GIFT-Eval (23 datasets, 97 configs), MASE + CRPS, 100 samples (Table 2). (3) FEV leaderboard (27 datasets), MASE + WQL, 20 samples (Fig. 4). All eval sets excluded from TimeBench.

**Headline numbers.** Table 1 avg MSE/MAE — Base: ETTm1 0.336/0.377, ETTm2 0.258/0.320, ETTh1 0.411/0.434, ETTh2 0.333/0.387, ECL 0.169/0.265, Weather 0.234/0.270; Large wins 16/16 first-place counts; family averages 7.57% MSE / 4.71% MAE below Time-MoE with fewer params. GIFT-Eval: MASE 0.673 (1st), CRPS 0.472 (2nd; best 0.465). FEV: 2nd zero-shot model after Chronos, 35× faster.

**Model-size scaling.** Table 1/9: Small→Base→Large monotone on most datasets (ETTh1 is non-monotone: Small 0.390, Base 0.411, Large 0.395).

**Objective ablation — the one that matters for us (Table 3 p.8, Table 7 p.14; same backbone, same TimeBench).**
| Objective | avg TSLib MSE | ETTm1 | ETTm2 | ETTh1 | ETTh2 | ECL | Weather |
|---|---|---|---|---|---|---|---|
| TimeFlow | **0.290** | 0.336 | 0.258 | 0.411 | 0.333 | 0.169 | 0.234 |
| Diffusion | 0.314 | 0.362 | 0.265 | 0.444 | 0.360 | 0.202 | 0.252 |
| MSE head | 0.296 | 0.360 | 0.264 | **0.404** | 0.341 | 0.175 | **0.231** |

CRPS (Table 7): TimeFlow 0.0059/0.0037/0.0057/0.0029/0.0082/0.0021 vs MSE-head 0.0063/0.0040/0.0058/0.0032/**0.0080**/0.0023 (ETTh1,ETTh2,ETTm1,ETTm2,ECL,Weather); GIFT-Eval CRPS 0.505 vs 0.642. So: **point MSE gain of the flow head over an MSE head is ~2% on average and the MSE head wins on 2/6 datasets**; the robust gain is distributional (CRPS), especially on the heterogeneous GIFT-Eval. (How CRPS was computed for a deterministic MSE head is not stated.) No quantile-loss or Gaussian-NLL head was compared.

**Test-time calibration (§5.4, Fig. 7, FEV).** More samples → better MASE *and* WQL ("conform to the central limit theorem"); more Euler steps → better both. Exact values are only in the plot; trend monotone with diminishing returns. Note MASE improves with samples only because the median-of-N estimator of the conditional median converges — not because the distribution is better.

**Context-length (App. C.3, Fig. 10).** Lookback 480→2880 at fixed 2880-trained model: effect is dataset-dependent, not monotone; authors: "should enhance fundamental long-context capabilities to handle high-frequency data".

**Architecture ablations (§5.6, Fig. 9, TSLib avg).** RoPE > no RoPE; Pre-LN improves with more iterations (15k→30k) while Post-LN degrades; FlashAttention/KV cache are pure efficiency.

**Absent:** patch-length ablation (P=16 fixed; cites TimesFM), F ablation, samples-vs-MSE curve, comparison to quantile/NLL heads, calibration-coverage numbers, any financial-only results.

## 5. Limitations stated by the authors (App. E p.16, §5.4, §2.2)

- May still hallucinate.
- High-frequency data not guaranteed (TimeBench is mostly mid/low frequency); future: multi-scale.
- Naive sampling (plain Gaussian start, uniform Euler); post-processing/frequency normalisation left open.
- Univariate pre-training: cannot use variate correlations or **covariates**.
- Multi-step autoregression can over-smooth and become unreliable (**[code]** median-of-samples feedback).
- Point-metric MSE benefit over an MSE head is small (their own Table 3); the CRPS claim on TSLib is 5/6 datasets.

## 6. Synthesis for the CausalPatch OHLC forecaster (`docs/timexer_segment.md`)

**Already aligned (nothing to change):** P=16 patches, MLP patch embedding on continuous values, decoder-only Pre-LN + RoPE + flash causal SDPA + final norm, dense per-token multi-patch heads (our 192 bars from every token ≈ their F=720 from every token), zero-init output layers, sizes (our 8L/512/2048 sits between Small and Base). Their evidence that RoPE + Pre-LN + dense next-patch supervision are the right skeleton is consistent with ours. Sundial itself is not a candidate for zero-shot use here: univariate, no covariates, 1% finance, authors disclaim high-frequency data.

**Normalisation — keep persistence/σ anchoring, do not adopt ReVIN.** Sundial's ReNorm removes the context mean and divides by context std of *levels*. For a near-random-walk price series the context mean is not a forecast anchor and the level-std is dominated by drift over 6000 bars, not by return volatility; the zero-vector prediction would be "revert to the window mean", a strong wrong prior. Our decoder makes the zero output equal to the persistence candle with a √h random-walk prior scaled by expanding return σ — this is the differenced, causal analogue of their stationarisation and is strictly better matched **[inference]**. Also their per-token training normalisation is non-causal (whole-window stats) whereas ours is expanding per origin. Two small borrowings: (a) concatenate the per-bar validity mask into the patch input as they do (they found masks necessary for arbitrary lengths; ours has session-gap flags but the raw `valid` bit is only in the loss mask) — trivial cost, expect ≈0 on metrics but cleaner handling of missing bars; (b) their std floor (1e-2 → 1.0) is a reminder to keep our σ floor (1e-8 variance) from producing extreme σ-scaled targets on illiquid tickers — check tail_loss_share.

**Patch length.** Paper gives no ablation; P=16 is asserted with a citation. Our 16×5 min = 80 min tokens, 375 tokens. Only worth a P∈{16,32} sweep if step time is the constraint (P=32 halves attention tokens); expected effect on per-horizon MSE ratio: unknown/small, and the paper offers no evidence either way.

**Replacing the Gaussian log-scale head with a TimeFlow head — assessment.**
- What the paper actually shows: same backbone, flow head vs MSE head → ~2% avg MSE, not uniform; large CRPS gains on heterogeneous data. It does *not* compare against a heteroscedastic Gaussian NLL head, which already captures the scale information that is the bulk of what is predictable in 5-min returns (vol clustering, intraday seasonality, range/ρ). Their "mode collapse" pathology (similar lookback, divergent trends → over-smooth mean) is exactly the conditional mean, which is the MSE-optimal point forecast; a flow head cannot beat it on MSE, only match it (via sample mean, at S×K FM-Net evaluations per token). **Expected change in per-horizon MSE/persistence ratio: 0 ± 1–2%.**
- Where a flow head can genuinely help: (i) fat tails / skew / session-boundary bimodality that a diagonal Gaussian mis-covers → shows up in **calibration coverage at 1.96σ and beyond (99%)** and in CRPS; (ii) **joint dependence across the 192 horizons and 4 OHLC coordinates** — our head is a diagonal Gaussian, so its sampled paths are white noise around the mean, not coherent trajectories; a flow over the 192×4 block yields coherent sample paths, which matters for path-dependent decisions (stops, drawdown) but is invisible to per-horizon marginal metrics; (iii) their representation-learning claim (the generative loss shapes the trunk) — testable.
- What it costs: NLL vs √h-prior is no longer available from the flow (no closed-form likelihood; an ODE log-det via Hutchinson trace is possible but expensive at 768 dims); validation would need CRPS/energy score from S samples × K steps at the final origin only (their K=50 Euler; our dbwm v3 uses Heun K=8, worth reusing). FM-Net of width 512 × 3 blocks ≈ 4M params; training rows = 256 × 375 × 4 (their `diffusion_batch_mul`) ≈ 384k per step — small relative to the backbone (**[inference]** +10–20% step time).
- Design if attempted: run the flow in the **pre-decoder candle-coordinate space** (c0, range pre-softplus, open/close pre-sigmoid) so every sample decodes to a valid candle and persistence remains the zero point; targets are the inverse-decoded ground truth (needs clipping where open==high etc.). Use x1-parameterisation with uniform CFM weight (our targets are already √h-scaled, so their 1/h horizon weighting is unnecessary), 4 (t, ε) draws per token, AdaLN conditioning on the same token state ⊕ known-future covariate projection, zero-init output.

**Do multi-sample generative outputs help point accuracy?** No, not beyond estimating the centre of the distribution: Fig. 7's MASE-vs-samples curve is the median-of-N converging to the conditional median (their own CLT remark); an analytic mean from a Gaussian head is that limit at zero sampling cost. Use sample **mean** (not their median) if you ever score a flow head on MSE. Multi-sample outputs only pay off on distributional/joint-path metrics.

**Ranked candidate changes.**
1. **Add a TimeFlow-style flow head beside (not instead of) the Gaussian head, shared trunk.** Benefit: coherent joint OHLC×horizon samples; better tail coverage; tests the representation claim. Cost: ~4M params, +10–20% step time, new CRPS/energy-score + coverage-from-samples evaluation. Metric: Gaussian head's per-horizon MSE ratio and NLL with vs without the auxiliary flow loss (representation effect, expect ≤1%); flow-sample CRPS vs closed-form Gaussian CRPS; 95%/99% coverage from samples vs Gaussian bands; joint-path energy score. Decision rule: keep only if CRPS or coverage improve materially, since MSE is expected flat.
2. **Cheaper tail fix first (no paper evidence, [inference]):** Student-t or diagonal+low-rank Gaussian head keeps exact NLL, addresses fat tails and cross-horizon correlation; if 1.96σ coverage is currently under nominal this closes most of the gap at near-zero cost. Metric: NLL vs √h-prior, coverage.
3. **Validity-mask channel in the patch input** (their m_i). Cost trivial; metric: tail_loss_share / invalid-candle fraction; expect ≈0 on MSE.
4. **SwiGLU FFN + dropout 0.1 ablation** [code, not paper]. Cheap; expect small; metric: full-validation NLL.
5. **P=32 sweep** only for throughput; no paper evidence.

**Do not adopt:** context ReVIN; univariate/no-covariate framing; rolling AR with median feedback (we predict the full 192 directly); FP32; their 1/h horizon weighting (already implied by our √h scaling); K=50 Euler (use Heun with fewer steps if sampling at all).

**Data-scale note.** Table 8 shows only ~8% MSE from 11× more data at 128M params; with ~470M target bars/epoch and per-token dense supervision on 375 origins we are not obviously data-starved for an 8L/512 model **[inference]**; multi-epoch training with the current corpus is the pragmatic path rather than corpus expansion.