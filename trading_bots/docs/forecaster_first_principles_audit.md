# Pretrained Bar Forecaster — First-Principles Audit

Reference design derived a priori, then measured against the implementation. Every
claim below carries `file:LINE` evidence from a read-only audit. No leakage was found;
the defects are stationarity, capacity-allocation, and decision-integrity defects.

---

## Part 1 — The reference optimum

### 1.1 The objective is a conditional path distribution, not a bar

Kelly maximizes `E[log(1 + f R)]`. For small returns `f* ~ mu / sigma^2`, so conditional
`sigma^2` enters with equal weight to `mu`, and the left tail enters through the `log`. A
point forecast of `E[R]` is not a partial solution — it is missing half the required
object and all of the risk control.

A trading policy is path-dependent (stops, targets, time exits). What Kelly actually
consumes is `E[log(1 + f * PnL(path))]`, a functional of the path, not of the close. The
theoretical maximum is therefore:

> The model emits a conditional generative simulator of the next-k-bar path; the trading
> layer samples paths, runs the actual policy (with costs, stops, slippage) on each, and
> maximizes expected log-wealth by Monte Carlo.

Every lossy summary — mean, sign, even quantiles of the close — is a projection that
destroys information the sizing layer provably needs.

### 1.2 Target parameterization

Naive `(O,H,L,C,V)` regression is structurally wrong: `O_{t+1} ~ C_t` is trivially
predictable and inflates every aggregate metric; `H` and `L` are order statistics of the
path, so independent heads violate `H >= max(O,C)` and `L <= min(O,C)` and burn capacity
learning an identity; levels are non-stationary.

Correct basis, all in causal-vol units (`sigma_hat_t` measurable on the information set):

- `r = log(C_{t+1}/C_t) / sigma_hat_t` — the tradeable quantity
- `g = log(O_{t+1}/C_t) / sigma_hat_t` — gap
- `u = log(H/max(O,C)) / sigma_hat_t >= 0`, `d = log(min(O,C)/L) / sigma_hat_t >= 0` — the
  excursions, i.e. exactly MFE/MAE, exactly what stops and targets need
- log dollar-volume innovation, spread, bar-internal RV (Garman-Klass, Rogers-Satchell)

Supports enforced by construction so ordering holds identically. Then add first-passage /
triple-barrier touch probabilities `P(hit +b before -a within h)`.

### 1.3 Loss: strictly proper, heteroscedastic, fat-tailed

MSE on returns estimates only the conditional mean, and its gradient is dominated by the
highest-vol bars where the target is pure noise. The dominant predictable structure in
financial data is volatility (`R^2 ~ 0.5`), not direction (`R^2 ~ 1e-3`).

- Loss = NLL of a flexible conditional density: Student-t with learned `nu`, a mixture, or
  a discretized bucket softmax.
- Known pathology: Gaussian/t NLL gives `d(loss)/d(mu) ~ 1/sigma^2`, so the model escapes
  hard examples by inflating `sigma` and the mean head stops learning. Must be countered
  (beta-NLL with detached `sigma^(2*beta)`, `beta ~ 0.5`, or detached-sigma staging).
- The `sigma` in the loss is also the correct per-sample normalizer.

### 1.4 Multi-horizon, jointly

Drift accumulates as `h`, noise as `sqrt(h)`, so SNR grows as `sqrt(h)` until alpha decay.
One-step targets are the least learnable point on the curve and the most
microstructure-polluted. Predict `h in {1,2,4,8,16,32,64}` jointly from a shared trunk.
Consequence: labels overlap, so sample-uniqueness weighting and purged/embargoed
validation become mandatory.

### 1.5 Auxiliary supervision is how a low-SNR task becomes learnable

With direction `R^2 ~ 1e-3` the trunk cannot be shaped by the direction gradient. It must
be shaped by high-SNR co-targets: realized vol at all horizons, excursions, volume,
spread, sign, quantile pinball, barrier touches, bar-internal RV, regime. Balance with
uncertainty-weighting or GradNorm so the easy vol task does not monopolize capacity;
better, give `sigma` a dedicated HAR-style head fed explicit RV features so the trunk is
freed for direction.

### 1.6 Representation

- Causality is absolute: every normalization statistic expanding/EWMA over the past only.
- RevIN (instance norm + de-norm at head) for non-stationary series.
- Multi-resolution; patch tokenization for cheap long context; channel-independence as a
  regularizer.
- Cross-sectional panel. A single-instrument model discards the factor structure and the
  only escape from tiny effective sample size. 10y of 1-min bars is ~1e6 bars — nothing.
  Times a universe, ~1e9 — a pretraining corpus. Cross-asset attention per patch step plus
  learned low-rank asset embeddings. This is the largest single lever available.
- Implied vol where options exist (a free market-consensus forecast of the modeled
  quantity), order-flow imbalance, depth, funding, OI, calendar as Fourier features,
  RoPE/ALiBi rather than learned absolute bar-index embeddings.

### 1.7 Uncertainty: aleatoric and epistemic

Plug-in Kelly over-bets because `f* = mu/sigma^2` is convex in an uncertain `mu`. Emit the
posterior predictive — deep ensemble, SWAG, or EMA snapshots — and integrate over
parameter uncertainty before sizing. In a low-SNR regime a single checkpoint is strictly
dominated.

### 1.8 Calibration is the bridge to Kelly

A well-ranked but over-confident distribution is more dangerous than no model:
over-confident `sigma` leads to over-leverage leads to ruin. Post-hoc distributional
recalibration / conformalized quantiles with finite-sample coverage, fit on a held-out
block, before any sizing.

### 1.9 Protocol

Purged walk-forward with embargo. Model selection on cost-net expected log-wealth, not
NLL. Explicit baselines: zero-drift, HAR-RV/EWMA vol, vol-targeted buy-and-hold. Multi-seed
with dispersion. Cost-aware objective with a no-trade band. Deflated Sharpe for the number
of configs tried.

### 1.10 Disqualifying-defect checklist

Non-causal normalization; level targets; MSE on raw returns; single horizon; no
vol/distributional head; Gaussian tails; NLL without beta-correction; independent OHLC
heads; shuffled or unpurged validation; no cost model; plug-in Kelly; no calibration step;
single asset; single seed; no HAR/zero baseline.

---

## Part 2 — Audit findings

**Verdict.** The target parameterization and scoring rule are near-optimal. The defects
are: the target is not volatility-standardized; the objective spends ~95% of its capacity
on untradeable factors; the model's uncertainty is discarded before it reaches the bet;
and the deployed path does not use the forecaster at all.

**No leakage.** Supports and the Kelly per-bin moment table are train-region only
(`support_moments.rs:616-621` -> `sample_train_dof` -> `dataset.rs:1810`), gated at 1e-12
per-bin mass identification and refused outright on absent provenance or moved split
bounds (`support_moments.rs:578-604`). Macro series key on release `available_at`. The
contemporaneous market channel is stripped from every target-bar conditioning path
(`world_model.rs:1774-1795`).

### Already at or near the optimum

| Reference item | Status |
|---|---|
| Distributional target, not point | 5-DOF joint autoregressive categorical `r -> s -> u -> v -> w`, 128 equal-mass bins/DOF (`bar_dist.rs:34,36,166`) |
| Excursions in the target | `s` log-range, `u`/`v` positions in range; `L <= O,C <= H` holds identically by construction (`bar_dist.rs:344-346`) |
| Fat tails without Gaussian assumption | Nonparametric quantile support with atoms; `BarScoring::Hard` indexed CE |
| beta-NLL / sigma-starvation pathology | Not applicable — no learned sigma in a gradient denominator. Structurally immune |
| Causal normalization | Volume EMA over bars `< t` (`bar_dist.rs:383-385`) |
| Path simulator exists | Ancestral rollout, 100 bars, 96 paths x 3 replicates (`horizon.rs:76-77,126,135`) |
| Calibration diagnostics | Per-DOF CRPS, randomized PIT at atoms, PIT-TV, coverage, tail exceedance |
| Economics-primary selection | Paired Kelly edge with NLL and `nll_dof[r]` vetoes at 2.0 / 1.0 paired SE (`pretrain.rs:5022-5061`) |
| Forecaster/policy isolation | `VarStore::freeze()` + `all_parameters_frozen()` + no_grad + detach; policy gradient cannot corrupt the trunk |

Positional encoding is PoPE (relative by construction), pre-RMSNorm, zero-init output
projections so the untrained model is the embedding identity.

### Tier 1 — changes what the model learns

#### T1.1 `r` and `s` are raw log quantities on one globally pooled grid

`encode_dof` applies no volatility divisor to `r` or `s` (`bar_dist.rs:312-331`).
`BarSupports::fit` fits 128 equal-mass bins over the pooled train draw across all symbols
and all time (`bar_dist.rs:883`). Four compounding costs:

1. A quiet name or regime concentrates in a handful of central bins, so discretization
   error is first-order exactly where per-bar Sharpe is best, while a volatile regime
   spreads across the grid. Effective resolution varies by an order of magnitude across
   the panel.
2. The head must re-derive the active bin window from context at every step, spending
   learned capacity on a quantity a causal EWMA/range estimator gives analytically at
   near-zero error. This is the wrong allocation: hand-code the cheap analytic part, spend
   the network on what is not analytic.
3. A 30 bp move on a quiet name and a 150 bp move on a volatile one are the same event in
   standardized units, but land in different bins and share zero parameters. This destroys
   most of the cross-sectional statistical strength, which is the main reason to have a
   panel.
4. Scale learning and shape learning are entangled in one softmax.

The tell that this is an oversight rather than a design choice: the other three DOF are
already handled correctly. `u`, `v` in `[0,1]` are scale-free; `w` is normalized by the
causal volume EMA. The defect is precisely 2 of 5 DOF, and the causal-EMA machinery is
already in the same file.

**Fix.** Bin `z_r = r / sigma_hat_t` and `z_s = ln(s) - ln(c * sigma_hat_t)`, with
`sigma_hat_t` a multi-scale causal estimator built on range volatility
(Parkinson/Garman-Klass, ~5x more efficient per observation than close-to-close, and `s`
is already in the data), ideally HAR-combined so the net predicts only the vol innovation.
Kelly stays exact: the fitted within-bin `z` law plus known `sigma_hat_t` gives
`E[expm1(sigma_hat_t z) | bin]` and its second moment by the same composite-midpoint
integration already used in `scoring_floor` (`SMOOTHING_FLOOR_NODES = 16`).

#### T1.2 The objective gives the only tradeable factor 1/20th of the gradient

Checkpoint metadata records per-factor NLL headroom as **r 0.110, s 0.192, u 1.092,
v 1.174, w 0.000 nats** (`world_model.rs:236-240`). `bar_nll_from_logits` returns a plain
unweighted `.sum()` over the five (`bar_dist.rs:3610-3616`), and
`SELECTION_WEIGHTS = [1.0; 5]` (`pretrain.rs:390`).

So ~91% of the available NLL improvement — hence ~91% of the shared-trunk gradient — sits
in `u`, `v`, `w`. `w` has literally zero headroom and still consumes 1/5 of the objective:
pure gradient noise into a shared trunk. Much of `u`/`v`'s large headroom is the model
learning an algebraic identity (`s == 0` forces `u == v == 0.5`, ~11-15% of bars), not
forecasting.

This is the textbook "easy high-SNR task monopolizes capacity" failure in categorical
form. The selection path already knows `r` is primary
(`SELECTION_GUARD_DOF = DOF_R`, `pretrain.rs:419`); training does not agree with it.

**Fix.** Explicit per-DOF loss weights with `r` up-weighted and `w` near zero, or
homoscedastic-uncertainty weighting over the five factors, recorded in checkpoint lineage.

#### T1.3 Multi-horizon is implemented and shipped off

`--direct-t2-weight` and `--direct-t3-weight` both `default_value_t = 0.0`
(`main.rs:288-294`, pinned by a test at `:2331-2332`), so the default trains a
single-horizon predictor and forfeits the `sqrt(h)` SNR growth. Worse, the zero-weight
branch still executes both direct heads and builds their graphs for matched compute
(`pretrain.rs:9598-9640`): the default run pays the multi-horizon compute and receives none
of the signal. Strictly dominated. The ladder is only `h in {1,2,3}`
(`DIRECT_MAX_HORIZON = 3`), not geometric.

#### T1.4 No cross-sectional structure whatsoever

No cross-asset attention and no symbol/asset embedding anywhere in `torch/`. The entire
panel is compressed into 3 quantized SPY-proxy buckets (`dataset.rs:175-232`). With ~255M
bars in the corpus this is the largest untapped lever, and it is the natural partner of
T1.1 — standardized targets are what make cross-asset parameter sharing meaningful.

Minor and free alongside: minute-of-day is a raw 1440-row embedding table with no cyclic
prior (~737k params to learn a smooth periodic function). Fourier features are strictly
better.

### Tier 2 — changes what reaches the bet

#### T2.1 The thin left tail is the diagnosed cause of cap saturation

`pretrain_aux.rs:10-16`, verbatim:

> That gap is not a theoretical worry, it is a MEASURED defect in the shipped predictor.
> The Kelly bench on the promoted checkpoint beats the unconditional-marginal null by
> +4.69 bps/bar on val and +4.28 on test, both intervals excluding zero — but 84.8% of its
> bars sit pinned at the 4x leverage cap, and one window was wiped out entirely. Kelly
> saturating at the cap means `E[log(1 + fR)]` is still increasing at `f = 4`, and that
> happens for exactly one reason: the predictive distribution's LEFT TAIL IS TOO THIN. The
> model does not believe in crashes, because it has never seen one.

Two mechanisms. `BAR_SUPPORT_CLIP_QUANTILE = 1e-4` collapses everything beyond the 1e-4
quantile into two catch-all bins whose decode is the *edge*, so a -20% bar's magnitude is
unrepresentable. And pooling across regimes dilutes crash mass. The leverage this has on
sizing is recorded in the tree itself (`bar_dist.rs:829-834`): those two bins hold 1.4474%
of mass but control **41% of the absolute first moment and 92.38% of the central second
moment**.

**Fix.** Splice a parametric (GPD) tail onto the outer bins so the catch-alls carry a
distribution, not a point.

#### T2.2 Production sizes with the quadratic surrogate; the exact solve exists and is never called

`kelly_fractions` uses `raw = mean / second` (`trade_bench.rs:464`).
`expected_log_growth` computes the true `E[ln(1+fR)]` over the 128-bin law
(`trade_bench.rs:370-383`) and is used only in tests.

The surrogate always over-levers under negative skew: `g'(f_quad) = f^2 E[R^3] - f^3 E[R^4]`,
negative for `E[R^3] < 0`, so the exact optimum lies strictly left of `f_quad`.

Quantified at the 4x cap on the tree's own worst decoded bin (-883.32 bps,
`trade_bench.rs:240`): `x = -0.338`, exact `ln(1+x) = -0.4129` versus surrogate
`x - x^2/2 = -0.3954` — the surrogate is too optimistic by `1.753e-2` nats on that outcome.
That equals the entire `0.5955` bps/bar selection edge once the outer bin carries 0.34%
mass, against 0.78% nominal for equal-mass bins. Same order as the whole edge at 4x.

At the 0.25x selection cap the gap is ~2.5e-4 bps/bar — negligible. **So the +4.69 bps/bar
headline is contaminated by the surrogate and the 0.25x selection number is not.** It also
explains cap saturation: at `f = 4` the surrogate genuinely does not see the loss it takes.

**Fix.** Newton on a strictly concave 1-D problem over 128 bins, 5-10 iterations, each two
elementwise passes plus a reduction over `[rows, 128]` — dwarfed by the transformer forward
that produced `probs`. Replace `trade_bench::kelly_fractions:405`; check against the
existing host-side `expected_log_growth`.

#### T2.3 Calibration is fitted correctly and then not used

`TradeSetup::new` sets `shrink: None` (`trade_bench.rs:716`), so every promotion-deciding
bench scores the **unrecalibrated** law. `MeanShrink` appears nowhere in `planner/`,
`infer/`, `model/`, or `world_model.rs`. The measured Mincer-Zarnowitz mean slope is
**0.6653 +/- 0.0286** fully annealed: predicted `mu` is ~50% too large, and Kelly is linear
in `mu`, so `f` is ~50% too large from this alone.

**Fix.** Persist the fitted `(alpha, beta)` as a checkpoint sidecar at promotion and apply
it where `f` is formed, so the number that is measured is the number that trades.

#### T2.4 The default LR schedule lands on the badly-calibrated operating point

`LR_PLATEAU_FRACTION = 0.40` (`pretrain.rs:200-206`) means a one-epoch run always finishes
fully annealed to the 0.15 floor and can never reach one full pass at peak rate — the
operating point where the MZ slope measured **1.0058 +/- 0.0355**, i.e. perfectly
calibrated, versus 0.6653 annealed. Calibration slope ~1 is exactly what Kelly needs. This
is a one-constant change with direct sizing impact.

#### T2.5 No epistemic uncertainty, no weight averaging, one seed

Repo-wide search for `ensemble|posterior|epistemic|SWAG|bayes` returns nothing; `dropout`
appears nowhere in the forecaster; `lr_disentangle.rs:196-206` states outright that no
retained best-so-far weight buffer exists anywhere in the promotion path. All production
runs use `train_seed 24301`.

Plug-in `f = mu / E[R^2]` is convex in uncertain `mu`, so this over-bets systematically —
the same symptom as T2.1 and T2.3. The categorical head makes the fix unusually clean: a
K-seed ensemble averages as a plain mean over the 5x128 simplex. EMA shadow weights inside
`Muon::step` are the cheapest first step and are free variance reduction.

#### T2.6 Barrier probabilities are nearly free and thrown away

The ancestral rollout samples the full intra-bar path, then the per-step reduction keeps
only the `r` bins and discards the sampled `s`, `u`, `v` (`horizon.rs:1109-1198`).
Retaining them makes first-passage / triple-barrier touch probabilities a byproduct — the
exact object a stop-based policy needs.

### Tier 3 — the measurement cannot currently support the conclusion

#### T3.1 The headline economic framing does not compose into an ownable book

The repo already indicts it (`portfolio.rs:11-15`):

> At step 20000 of `bardist_v2` that framing produced `+4.3202` bps/bar net which, at the
> bench's own hardcoded `93 * 252` bars per year, annualizes to `exp(10.125)` — about
> 24,900x per year. That number is not a profit estimate. It is a proof that the framing is
> broken.

The same applies to `+0.5955` bps/bar: times 93 times 252 gives ~+304%/yr at a 0.25x gross
cap. It is the mean of 256 independent per-window growth rates, each betting its own
wealth, with no shared capital, no cross-sectional allocation and no calendar.

Compounding on top:

- Selection charges 2.0 bps flat one-way (`pretrain.rs:7710`) against measured 10.620 bps
  impact-free and 26.351 bps all-in (`horizon.rs:180,184`), on a policy rotating the book
  **3.346x per bar** (`portfolio_cost.rs:1374-1378`). Reported break-even is 3.29 bps
  (`portfolio_cost.rs:6-9`) — below measured cost, so the verdict flips sign under the
  repo's own cost calibration.
- Sizing is cost-blind and covariance-blind: positions solved on raw moments with cost
  charged post hoc (`trade_bench.rs:122-125`); stacking per-name Kelly asserts a diagonal
  covariance (`portfolio_cost.rs:56-62`), and `leverage_error` exists precisely to measure
  the resulting over-leverage.
- The promoted checkpoint is the argmax over the run of this exact quantity on these exact
  windows (`pretrain.rs:6280-6283`), so the reported level is an order statistic that the
  paired block bootstrap does not price.

`pretrain.rs:332-336` already records that the receding single-book evaluation reversed the
independent-window ordering, and concludes that neither proves deployed profitability.

#### T3.2 No volatility scoring rule and no vol baseline

No QLIKE anywhere, no HAR-RV/EWMA vol baseline, no vol-targeted buy-and-hold (only `f = 1`
flat, `trade_bench.rs:60-75`). Volatility is the highest-SNR quantity in the data and `s`
is a direct vol observable, so we currently cannot say the model beats a 3-parameter HAR
regression at the thing it is best at.

Also no per-horizon rank-IC: `DecayPoint` carries a bare Pearson `f64` with no interval and
no rank version (`trade_bench.rs:312-322`). Rank-IC is the robust one under fat tails.

#### T3.3 No embargo at the seam; no label-uniqueness weighting

Purging-by-truncation is present — no label window crosses the boundary
(`dataset.rs:1952-1959`), which is the important half. But train and val abut exactly
(asserted at `dataset.rs:4443`) while `DOF_WARMUP_BARS = 256` plus the EMAs mean the first
~256 val bars carry features built from train bars. An embargo of
`max(context, warmup, horizon)` is cheap and strictly correct.

No uniqueness weighting for overlapping h2/h3 labels (~3x effective-N inflation on the h3
term).

#### T3.4 The forecaster is not connected to anything that trades

`infer/ibkr`, `infer/offline`, `env/`, and `planner/` run a legacy RL Beta policy emitting a
**long-only** weight in `[0,1]` and never touch the forecaster or any Kelly solve. Every
improvement above is currently unrealized in deployment, and long-only discards the short
half of a signed forecast.

### Flagged, not judged

- **No cost-aware or decision-focused term in the objective.** Growth is computed under
  `no_grad` with objective share hard-zeroed (`pretrain.rs:21-23,9662-9672`). Since costs
  flip the sign, this is the misaligned-proxy case; `growth.rs` already holds the
  differentiable pieces.
- **Gradient conflict is measured and discarded.** `shared_gradient_diagnostics`
  (`pretrain.rs:10335-10374`) computes cosine alignment and coordinate sign conflict between
  the h1 and h2+h3 gradients — exactly what a conflict-resolution scheme consumes — and only
  writes it to a chart.
- **`ADAMW_UPDATE_EVERY = 2`** is real gradient accumulation over the embedding tables and
  all five emission heads (`pretrain.rs:241-242`, `muon.rs:1125-1131`) — the construct repo
  policy names. Inherited from modded-nanogpt.

---

## Priority ordering

1. **T1.1** vol-standardize `r`, `s` with a range-based causal sigma — the one change that
   most alters what the trunk can learn, and the precondition that makes T1.4 worth
   building.
2. **T1.2** per-DOF loss weights — `r` up, `w` to ~0. Interacts with T1.1.
3. **T2.2 + T2.3 + T2.1** the sizing-integrity bundle. Cheap, pure correctness, and directly
   attacks the 84.8% cap saturation that currently makes the distribution's richness
   irrelevant at the decision boundary.
4. **T3.1 + T3.2** receding single-book metric with measured costs as primary, plus QLIKE and
   a HAR baseline. Without these we cannot tell whether Tier 1 work is helping.
5. **T1.3** rides along free: flip two defaults, extend the ladder.

---

# Part 3 — Cumulative ablation campaign

## 3.0 Operating facts the plan is built on

| Fact | Value | Evidence |
|---|---|---|
| Throughput, current default | ~4 step/s (range 3.0-5.5) | marginal rate 6.76 step/s at batch 16 from `training/bardist_v1.log:214-396`, scaled by the 16->24 batch change (`main.rs:225-226`) |
| Standard run | 19,635 steps, `stage_steps [6545,6545,6545]`, epochs 1 | live sidecar `directmtp_v3_privateheads_s24301_301491ee/weights/pretrain_step_19634.metadata.json` |
| Context ramp | 3 stages, `[896, 1472, 2048]` | `pretrain.rs:102,840-845` |
| Step-limit flag | `--steps` (NOT `--total-steps`/`--max-steps`) | `main.rs:196-199` |
| **`--steps S` overrides `total_steps` ONLY** | `steps_per_epoch` stays corpus-derived at 19,635, so **any S <= 6545 runs entirely in stage 0 at context 896 and can never promote** | `pretrain.rs:2179-2193, 1011-1022, 6105-6106` |
| What a stage-0 arm *does* write | `pretrain_best_diag896.{ot,windows.json}` at every improving validation, via `keep_context_best`, called unconditionally | `pretrain.rs:6200-6201, 7199-7253`; tested at `:11614-11620` |
| Eval windows are pinned independent of `--seed` | `EVAL_WINDOW_SEED = 0xE7A1_5E7D_0001`, enforced at every consumer | `pretrain.rs:312-324, 4267-4271` |
| Cross-run comparator exists | `pretrain-compare <baseline.windows.json> <candidate.windows.json>` | `main.rs:1177-1183, 2056-2069` |
| Paired SE, 0.25x-cap edge | residual sd 0.0200 bps on a 0.3796 bps base (5.3%); 2.0-SE band ~0.040 bps | `pretrain.rs:402-409` |
| Cross-run paired NLL MDE | 0.04-0.09 nats (vs ~0.41 unpaired) | `pretrain.rs:317-321` |
| Launcher | `./trading_bots/run-release-cuda.sh pretrain ...` — never bare cargo | `run-release-cuda.sh:5`, `torch-env.sh:52-61` |
| Artifacts | `training/runs/<name>/{meta.json,weights/,gens/,training.log}`; distinct `--run` required or `create_fresh` bails | `run_dir.rs:67-116, 88-98` |
| Disk | ~2.0 GB per completed run; ~1.5 GB per stage-0 arm | measured from run directories |

### Step budget

`10 min x 4 step/s = 2,400 steps`. **Use `--steps 3000`**: it lands on clean validation
boundaries (1000 / 2000 / final 2999 = 3 reads), and `floor(0.40 * 3000) = 1200` means the
arm fully anneals to the 0.15 LR floor — the same qualitative operating point as the
production 1-epoch run, so T2.4 stays a live variable rather than an artifact of the budget.

Screen tier also sets `--validation-windows 2048` (default 4096). A prefix of the pinned
draw is as pinned as the whole (`trade_bench.rs:190-193`), so pairing survives **provided
every arm including the control uses the identical count** — `paired_comparison` refuses
mismatched window lists (`pretrain_stats.rs:970-982`).

> **Unmeasured:** startup (451M-bar corpus load + 4M-row support fit) is outside the step
> budget and outside the printed step/s (`pretrain.rs` sets `started` immediately before the
> loop). Measure it on the first control run and add it to every wall-clock estimate.

### What a 3,000-step screen can and cannot see

It is a **sample-efficiency screen, not a final-performance screen.** It is biased *toward*
detecting exactly the mechanism T1.1 and T1.2 claim (better use of each gradient), and
biased *against* capacity arms (T1.4 cross-asset) that need data to pay off. Capacity arms
therefore skip screen and enter at confirm tier. It also measures at context 896, not the
deployed 2048; the rank correlation between the two is an assumption that must be validated
once (see V1).

---

## 3.1 Wave 0 — instrumentation. Blocking, not optional.

Two findings make every economic ablation unjudgeable until they are fixed.

**W0.1 — There is no per-window edge anywhere on disk.** `WindowScore` is exactly five
fields: `symbol`, `bar_index`, `ts_ms`, `nll_dof[5]`, `conditional_nll`
(`pretrain_stats.rs:396-408`). `WindowScores.trade` is the only economic field and it is
an already-bootstrapped aggregate whose SE is collapsed — it cannot be re-differenced. Worse,
`Trainer::window_scores` hardcodes `trade: None` (`pretrain.rs:8033-8037`) and the only
writer that attaches one is `write_epoch_checkpoint` (`:7157-7161`), which a `--steps 3000`
arm never reaches. The vector we want exists in memory — `selection_edge_windows`
(`pretrain.rs:7703-7729`) returns per-window `(model - marginal_null) * 1e4` bps/bar at
`SELECTION_CAP` — but it is never serialized *and* it is computed inside the `deployed_ready`
branch, so a stage-0 arm does not even reach the call.

Edits (four, no new statistics):
1. `WindowScores += selection_edge_bps: Option<Vec<f64>>` **and**
   `model_growth_bps: Option<Vec<f64>>`. Both are needed: the null leg is support-derived
   (`trade_bench::marginal_position(&supports_dev, ..)`, `pretrain.rs:7707-7708`), so the
   differenced edge is only valid within one geometry, while the raw model leg is pure
   realized-return arithmetic and is valid across all of them. Bump
   `WINDOW_SCORES_FORMAT_VERSION` 2 -> 3 (`pretrain_stats.rs:59`) so old vectors fail loudly.
2. Fill both in `Trainer::window_scores` (`pretrain.rs:8005-8044`), which already receives
   `stats: &EvalStats` carrying `trade_paths`. Move the computation **out of the
   `deployed_ready` branch** so it fires at every validation, stage 0 included, and lands in
   `pretrain_best_diag896.windows.json` automatically.
3. `PairedComparison += edge_difference / model_growth_difference: Option<Dispersion>`,
   block-bootstrapped with symbol-month blocks **truncated to the traded prefix**, matching
   `bootstrap_traded` (`pretrain.rs:7735-7745`) exactly so the interval is the same object
   selection uses.
4. Print both in the `Display` impl (`pretrain_stats.rs:920-965`).

**W0.2 — `paired_comparison` would silently lie across a support-geometry change.** It
compares `corpus_fingerprint`, `split_bounds`, `eval_window_seed`, `scoring`,
`realized_batch`, `realized_steps`, `split`, `context`, window count and per-index
`(symbol, bar_index)` — **none of which is a function of the bin geometry**. `scoring` is the
rule name only. The T1.1 arm passes every check and returns a meaningless nats difference
with a tight CI. `bar_dist.rs:501-505` states the failure exactly: *"two runs whose supports
were fitted on different data are not comparable no matter how carefully everything else is
pinned. A bin-count check cannot see that; this can."*

Fix: add `supports_sha256: Option<String>` to `WindowScores`, populate from the metadata the
writer already holds (`world_model.rs:213, 511-519`), and `ensure!` equality in
`paired_comparison`. Precedent to copy verbatim: `mem_probe::assert_one_geometry`
(`mem_probe.rs:507-525`), which already does this for checkpoints.

**W0.3 — Persist the Mincer-Zarnowitz slope at every validation.** Currently produced only
by the offline `pretrain-calibration` command. `beta` is dimensionless and invariant to
geometry, and `beta -> 1.0` is precisely what Kelly needs, so it is the campaign's
cross-geometry calibration ruler. Paired form already exists
(`trade_bench::mincer_zarnowitz_paired:2782-2783`).

**W0.4 — Missing scoring rules and baselines** (T3.2). Add QLIKE for the volatility forecast,
a HAR-RV/EWMA vol baseline, a vol-targeted buy-and-hold policy row alongside the existing
flat `f=1` `BUY_HOLD` (`trade_bench.rs:60-75`), and per-horizon **rank**-IC with intervals
(`DecayPoint.correlation` is today a bare Pearson `f64`, no interval, no rank version). Every
new series MUST be a `.report.bin` base registered in **both** `PRETRAIN_REPORT_BASES`
(`shared/src/report.rs`) and `meta_chart_bases` (`tui/src/main.rs`) — per repo policy, no
ad-hoc CSV or log scraping.

**W0.5 — Establish the seed noise floor. This is the single most important statistical step
in the campaign.** The block bootstrap prices *window* sampling noise; it does **not** see
*training* stochasticity, which is what actually separates two arms trained with different
seeds. Run the unmodified control at screen budget on **3 seeds** and record the between-seed
sd, `s_seed`, of every primary metric. Every cull threshold below is denominated in `s_seed`,
not in the bootstrap SE. Without this the campaign will confidently carry noise.

**W0.6 — Campaign driver.** No sweep/ablation driver exists (`training/campaign_binaries/`
and `campaign_contracts/` are hand-maintained operator conventions with no code behind them).
`recirculate.rs` supplies the right *staging pattern* — disjoint window ranges per tier, cheap
screen -> confirmation gate -> test, with `MIN_CONFIRMATION_GAIN_NATS = 0.05` sitting squarely
inside the 0.04-0.09 nat cross-run MDE — but it drives inference-time configs on one frozen
checkpoint and has no concept of launching a run. Write a thin driver that (a) launches
`pretrain --run <arm> --steps <S> --seed <s> <flags>`, (b) reads
`training/runs/<arm>/weights/pretrain_best_diag896.windows.json`, (c) shells to
`pretrain-compare` against the incumbent, (d) applies the cull rules in 3.3. **Reimplement no
statistics** — every primitive exists.

---

## 3.2 The cumulative ladder

Each wave is measured against the **carried stack**, not against the original control. A wave
enters with whatever survived confirm tier upstream of it.

### Wave A — decision-side. Frozen checkpoint, zero training runs.

T2.1/T2.2/T2.3 are evaluation-side: they change no weights. Run them as a
`recirculate`-style sweep over the existing promoted checkpoint — minutes each, not hours —
and lock them in before any training arm, so every downstream arm is measured through a
correct sizer.

| Arm | Change | Audit |
|---|---|---|
| A1 | Exact `E[ln(1+fR)]` Newton solve replacing `f = E[R]/E[R^2]` in `kelly_fractions:405`; check against existing `expected_log_growth:370-383` | T2.2 |
| A2 | Apply the fitted `MeanShrink` on the sizing path (`TradeSetup::new` currently `shrink: None`) | T2.3 |
| A3 | GPD tail spliced onto the two outer catch-all bins | T2.1 |

Cumulative: `A1 -> A1+A2 -> A1+A2+A3`. Judged at **both** the 0.25x and 4x caps, because the
quadratic error is negligible at 0.25x and first-order at 4x. Required direction:
`free_kelly_saturated` **down** (the 84.8% pin is the target), `max_drawdown` and `ruin_bars`
down, model-leg growth flat or up.

### Wave B — training flags. Zero code.

| Arm | Flags | Audit |
|---|---|---|
| B1 | `--direct-t2-weight 1.0 --direct-t3-weight 1.0` | T1.3 |
| B2 | `--lr-plateau-fraction 0.85` | T2.4 |

B1 is close to free: the direct heads already execute at weight 0 for matched compute
(`pretrain.rs:9598-9640`), so the control is paying for them and getting nothing. B2's
success criterion is the MZ slope moving from 0.6653 toward the 1.0058 measured at the
plateau operating point — a calibration arm, judged on `beta`, not on nats.

### Wave C — objective reweighting. Small code.

| Arm | Change | Audit |
|---|---|---|
| C1 | Per-DOF loss weight `w -> 0.0` (0.000 nats headroom; pure gradient noise into a shared trunk) | T1.2 |
| C2 | On top of C1, up-weight `r`: sweep `{2x, 4x, 8x}` | T1.2 |
| C3 | EMA shadow weights inside `Muon::step`, promoted alongside live weights | T2.5 |

C2 is the only intra-wave sweep in the campaign; take the argmax of the three and confirm
only that one, counting the selection against the multiple-testing budget in 3.4.

### Wave D — horizon structure. Medium code.

| Arm | Change | Audit |
|---|---|---|
| D1 | Geometric ladder `h in {1,2,4,8}` replacing `DIRECT_MAX_HORIZON = 3` | T1.3 |
| D2 | Label-uniqueness weighting for overlapping multi-horizon labels | T3.3 |
| D3 | Retain sampled `s,u,v` in the rollout reduction; emit barrier-touch probabilities | T2.6 |

D3 is a capability arm, not an accuracy arm: it cannot improve the primary metric and must be
judged on whether barrier probabilities are calibrated (PIT of realized first-passage against
predicted), then held for a future stop-based policy.

### Wave E — target reparameterization. Large code. NLL comparison INVALID.

| Arm | Change | Audit |
|---|---|---|
| E1 | Bin `z_r = r/sigma_hat_t` and `z_s = ln(s) - ln(c*sigma_hat_t)`; `sigma_hat_t` = multi-scale causal range vol (Parkinson/GK), HAR-combined | T1.1 |

**Gated on W0.2 landing first.** Judged *only* on invariant metrics — see 3.3. Kelly stays
exact by integrating `E[expm1(sigma_hat_t z)|bin]` and its second moment over the fitted
within-bin `z` law with the composite-midpoint machinery already in `scoring_floor`
(`SMOOTHING_FLOOR_NODES = 16`).

### Wave F — capacity. Enters at confirm tier, skipping screen.

| Arm | Change | Audit |
|---|---|---|
| F1 | Learned asset embeddings | T1.4 |
| F2 | Cross-asset attention per time step | T1.4 |
| F3 | Fourier time-of-day features replacing the 1440-row table | T1.4 |
| F4 | Embargo of `max(context, warmup, horizon)` at the train/val seam | T3.3 |

F4 is a *correctness* arm and is expected to make measured numbers slightly **worse**. Adopt
it regardless of sign; it is not subject to the cull rules.

### Wave G — epistemic. Reuses seeds already run.

| Arm | Change | Audit |
|---|---|---|
| G1 | K-seed ensemble averaged over the 5x128 simplex | T2.5 |
| G2 | Shrink `f` by ensemble dispersion in `mu` | T2.5, T1.7 |

---

## 3.3 Autocull rules

### Metric selection by arm class

**Geometry-preserving arms (A, B, C, D, F, G):**
- Primary: paired `edge_difference` (0.25x cap, bps/bar).
- Guard/veto: paired `dof_difference[DOF_R]`.
- **Never** the aggregate `difference` — it is dominated by `s`/`u`/`v`, each with over a nat
  of headroom that cannot affect P&L (`pretrain.rs:433-436`, and T1.2).

**Geometry-changing arms (E1, and A3/F1 if the tail becomes structural):**
- Primary: paired `model_growth_difference` — `window_growth_at(POLICY_MODEL, SELECTION_CAP,
  cost)`, pure realized-return arithmetic with no support in the metric.
- Guard: MZ slope `beta` toward 1.0; `dir_acc` (invariant — a fraction of bars, the decoded
  mean's *sign* is the treatment).
- **Forbidden:** every `nll_*`, `crps_dof` (its discretization floor moves with the geometry),
  and the differenced *edge* (its null leg is support-derived). PIT-TV is a within-arm check
  only, since standardization moves the atom set.

### Thresholds

`s_seed` = between-seed sd of the primary metric from W0.5. `se_win` = block-bootstrap paired SE.

| Tier | Config | Rule |
|---|---|---|
| **Screen** | `--steps 3000`, 1 seed, 2048 windows | **Cull** if `primary_delta < -1.0 * s_seed`. Lenient by design: screen only kills clear losers; nothing is carried here. |
| **Confirm** | `--steps 6500` (full stage 0), 3 seeds, 4096 windows | **Carry** if `mean(primary_delta) > +2.0 * sqrt(s_seed^2/3 + se_win^2)` **and** no guard veto. Otherwise discard and keep the incumbent. |
| **Deploy** | full 19,635 steps, ctx 2048, 3 seeds | Promotion-eligible. Scored on the **receding single-book** metric with **measured** costs (10.620 bps impact-free, `horizon.rs:180`), not the 2.0 bps flat that selection currently charges. |

### Hard tripwires — immediate cull at any tier, regardless of the primary metric

1. Non-finite loss, or `global_grad_norm` non-finite (already asserted, `pretrain.rs:10282-10286`).
2. `free_kelly_saturated` rises by more than 5 pp vs the incumbent — a thinner left tail, the
   T2.1 pathology, and a change that looks like edge while buying ruin risk.
3. `ruin_bars > 0` where the incumbent had 0.
4. Paired `correlation < 0.9` — the pairing has broken and the comparison is uninterpretable
   (`pretrain_stats.rs:888-909` warns at this level).
5. Resolved `dof_difference[r]` regression at 2.0 SE (geometry-preserving arms only).
6. `supports_sha256` mismatch on an arm that was not declared geometry-changing — a silent
   measure change, which means the arm is not what it claims to be.

---

## 3.4 Campaign hygiene

**Order dependence.** A cumulative ladder can carry a change that only helped in an earlier
context. Mitigation, mandatory: after the stack is final, run a **leave-one-out re-ablation at
deploy tier** — drop each carried change individually and confirm the stack degrades. Anything
whose removal does not hurt gets dropped. This is cheap relative to the campaign and is the
only defence against a stack of mutually-cancelling changes.

**Multiple testing.** Screen only culls, so its false-positive rate is irrelevant. Carries
happen at confirm, at 2.0 SE one-sided over ~10 arms: expected false carries `~10 * 0.023 =
0.23`, consistent with the 2.0-SE constant the repo already calibrated to bound noise
promotions at ~0.3 per run (`pretrain.rs:411-414`). The C2 intra-wave sweep of 3 adds one
argmax; charge it as ~1.5 effective tests.

**Wall clock, at 4 step/s, serial on the single 5090** (startup excluded — unmeasured):
screen arm ~17 min + startup; confirm arm ~45 min x 3 seeds = ~2.25 h; deploy arm ~1.8 h x 3
seeds = ~5.4 h. Wave A costs no training time at all.

**Concurrency.** One RTX 5090, 31.36 GiB; the recorded OOM was at stage 1 / ctx 1472
(`bardist_v1.log:397-400`). Screen tier is entirely stage 0 / ctx 896, so 2-way concurrency
may fit — test it once, do not assume it. `RunDir::activate()` atomically rewrites
`training/runs/latest`, so **concurrent arms must be addressed by name, never through
`latest`**.

**Disk.** ~1.5 GB per stage-0 arm. Retain, per screen arm, only `meta.json`,
`pretrain_best_diag896.{ot,windows.json,metadata.json}` and `gens/` (~155 MB); delete the
optimizer bundles. Without this an 18-arm screen costs ~27 GB.

**Naming.** `--run <wave><n>_<stack-hash>_s<seed>`, e.g. `C1_a1a2b1_s24301`, so the carried
stack is legible from the directory name and two arms can never collide in
`RunDir::create_fresh`.

**Never change `EVAL_WINDOW_SEED`.** *"Changing this value invalidates cross-run comparability
for the whole campaign"* (`pretrain.rs:323`). It is the invariant the entire paired design
rests on.

---

## 3.5 Validation checks the campaign owes itself

- **V1 — Does diag896 rank like deployed 2048?** Take 3 existing completed runs, compare their
  `pretrain_best_diag896` ordering to their `pretrain_best` (ctx 2048) ordering. If the
  ordering does not hold, the entire screen tier is measuring the wrong thing and confirm tier
  must move to a context-2048 budget.
- **V2 — Is `s_seed` smaller than the effects we intend to detect?** If the between-seed sd at
  screen budget exceeds the expected effect size of a Wave B/C arm, the screen budget is too
  short and must rise before any arm runs.
- **V3 — Does the model beat HAR-RV on volatility?** From W0.4. If it does not, the highest-SNR
  half of the forecast is worse than a 3-parameter regression and Wave E jumps the queue.


---

# Part 4 — Campaign log

Append-only. Every entry records the evidence a decision rested on.

## V1 — Does diag896 ranking predict deployed-2048 ranking? CONDITIONAL PASS.

Date 2026-08-27. Binary `training/campaign_binaries/wave0-v1-baseline-ea686530` (snapshot of
`target/release/trading_bot_0` @ `ea686530`, frozen before Wave 0 edits began so this result is
immune to them). Method: for every within-family run pair, `pretrain-compare` at
`pretrain_best_diag896` and again at `pretrain_best` (ctx 2048), then test sign agreement among
pairs resolved at 2 SE in **both** contexts. 21 pairs attempted, 12 comparable.

| Family | Corpus fingerprint | Runs | Effect range @896 | total NLL | `delta_r` |
|---|---|---|---|---|---|
| A — modern `directmtp` v1/v2/v3 | `334ee0f1a975` | 4 | 0.0004 - 0.0465 nats | 5/6 | **5/5** |
| B — older, incl. a warmstart run | `de43e745cdfa` | 6 | 0.0001 - 3.3012 nats | 5/5 | **3/6** |

**Verdict.** The guard metric `delta_r` — the factor closest to tradeable content — has its
ordering preserved perfectly (5/5, across |delta_r| 0.0019 to 0.0330) for *close variants of a
common recipe*, and fails badly (3/6) across *radically different training regimes*. A
cumulative ablation ladder produces close variants off a common incumbent by construction, so
the screen tier is valid **in the regime we will actually operate in**, and only there.

The two Family-B `delta_r` failures both involve `return_nll_only_warmstart_s24301` and are not
marginal: `bardist_v7 / return_nll_only` reads `+0.0991` at 896 and `-0.0163` at 2048. A large
896 effect can carry the wrong sign. Magnitude-gating on the 896 side alone does **not** protect
against this; family membership is what protects against it.

Guards adopted into the cull rules (section 3.3) as a result:

- **G1 — Close-variant guard.** Any arm whose total-NLL |delta| vs the incumbent at 896 exceeds
  **0.5 nats** is not a close variant. It is Family-B territory, where r-ordering broke; send it
  straight to confirm tier at ctx 2048 and do not cull it on screen evidence.
- **G2 — r-veto floor.** The screen-tier `dof_difference[r]` veto fires only when
  `|delta_r| >= 0.002`. Below the smallest observed agreeing magnitude (0.0019) the ordering is
  unvalidated, not merely noisy.
- **G3 — total-NLL floor.** Screen ordering on total NLL is unreliable below ~0.003 nats: the
  single Family-A disagreement was `+0.0004` @896 vs `-0.0027` @2048, resolved at 2 SE in both
  and still inverted.

**Open risk, explicitly not yet closed.** V1 tested NLL, because NLL is all that exists on disk
today. The campaign culls on the *economic* vectors, whose context-stability is **unknown**.
Once W0.1 lands, re-run this identical test on `selection_edge_bps` and `model_growth_bps`
(logged below as V1b). Until V1b passes, no economic cull decision at screen tier is defensible.

### Incidental findings

- **The paired SE is legitimately near-IID, and that is a property of the pinned draw.**
  Every comparison reports `4096 blocks / 4096 windows`. I first read this as the symbol-month
  blocking silently not being applied; that was **wrong**, and `EdgePersist` corrected it with
  proof. `paired_comparison:1153` does call `symbol_month_blocks()`, and `Dispersion.blocks` is
  not `blocks.len()` — `moving_block_bootstrap` groups the id slice into a `BTreeMap<u64,_>` and
  sets `blocks = totals.len()`, the number of DISTINCT ids
  (`pretrain_stats.rs:178-187,196,245`), pinned by the test
  `level_dispersion_blocks_by_calendar_month`, where 64 windows collapse to 8 blocks. So the real
  finding is about the corpus: on this split **almost every `(symbol, month)` holds exactly one
  pinned window**, which the module doc already states. The resampling units are genuinely
  distinct, so the ~0.0014 nat paired MDE is defensible rather than optimistic.
  Practical rule adopted: quote `dispersion` for a paired DIFFERENCE, and `level_dispersion`
  (calendar-month blocking, single-digit K) for a LEVEL.
- **Family B artifacts are pre-schema** and fail to load with `missing field conditional_nll`.
  The version gate behaves correctly — loudly. Consequence: old `.windows.json` cannot serve as
  the campaign incumbent; the incumbent baseline must be regenerated under the current schema
  (and again after W0.1 bumps the version).

### Canonical arm invocation (established while launching the W0.5 probe)

Three successive hard refusals had to be cleared, and each one was the corpus loader correctly
protecting comparability. The resolution is **not** to delete and refit anything — refitting
would silently re-mean the conditioning and change the output space, making every arm
incomparable to the incumbent.

1. `bar_market_supports.300.json` carries no provenance -> `--freeze-market-supports`.
2. `bar_supports.300.json` carries no provenance -> `--freeze-supports`.
3. That default file is **stale**: format_version 5, `provenance: null`, sha `aa5d24ad...`. The
   binary requires v6's directly fitted `E[R|bin]` / `E[R^2|bin]` for the growth diagnostic.

The incumbent `directmtp_v3_privateheads_s24301_301491ee` records
`supports_sha256 = 818e4448...`, and exactly one corpus file matches it byte for byte:
`long_data/bars/bar_supports.300.v6-moments.json` (v6, provenanced, fitted on corpus
`de43e745cdfa` with 4,000,000 samples, deliberately frozen across the current corpus
`334ee0f1a975` — the documented mid-campaign freeze). Pinning it with `--supports` reproduces the
incumbent's exact output space and touches no shared data.

```
./torch-env.sh <binary> pretrain \
  --run <arm> --steps 3000 --seed <seed> \
  --validate-every 1000 --validation-windows 2048 \
  --supports long_data/bars/bar_supports.300.v6-moments.json \
  --freeze-supports --freeze-market-supports \
  <arm-specific flags>
```

Every arm MUST carry the `--supports` / `--freeze-*` triple verbatim. Dropping it either fails
closed or silently changes the measure. Corpus as loaded: **5,479 symbols, 780,243,675 bars at
300s** (249 dropped), split `2025-10-07T12:10:00Z | 2026-03-13T18:45:00Z`. Startup to step 0 is
~12 s with `--supports` supplied, so the step budget dominates wall clock.

## W0.5 — Seed noise floor. **THE CAMPAIGN-DEFINING MEASUREMENT.**

Date 2026-08-27. Three control arms, identical in every respect except `--seed`, at the canonical
invocation above. All three completed 3000 steps and promoted. Seeds 24301 / 24302 / 24303.

| metric | mean | `s_seed` | `se_win` reported by `pretrain-compare` | ratio | MDE 1 seed | MDE 3 seeds |
|---|---|---|---|---|---|---|
| total NLL (nats/bar) | 16.30283 | **0.05754** | 0.0005 | **115x** | 0.1151 | 0.0664 |
| `r` NLL (nats/bar) | 3.75355 | **0.02465** | 0.0004 | **62x** | 0.0493 | 0.0285 |
| edge@4x (bps/bar) | 7.52611 | 0.49104 | 0.9853 (unpaired level) | — | 0.9821 | 0.5670 |
| `free_kelly_saturated` | 0.68880 | 0.00836 | — | — | 0.0167 | 0.0097 |

### Consequence 1 — the in-tree comparator's significance verdict is denominated in the wrong noise.

`pretrain-compare` prices only *sampling* noise over windows at a fixed seed. It does not price
*training* noise. On total NLL it therefore understates the real run-to-run dispersion by **two
orders of magnitude**. The V1 run printed `paired delta -0.0388 +/- 0.0005 nats, verdict:
SIGNIFICANT at 95%`. Against `s_seed` that same delta is **0.67 sigma — not resolvable from one
seed.** Every historical single-seed recipe comparison in this repo inherits the error:

| observed Family-A delta (single seed each) | nats | in `s_seed` units |
|---|---|---|
| v2routed -> v3, total | 0.0388 | 0.67x |
| v2routed -> v3, `r` | 0.0266 | 1.08x |
| contin -> v3, total | 0.0077 | 0.13x |
| contin -> v3, `r` | 0.0064 | 0.26x |

The production recipe lineage was selected on differences that sit inside the seed noise.

This does **not** retract V1. V1 asked whether a *given pair of runs* keeps its ordering across two
contexts; both measurements share the same two seeds, so the seed term is common and cancels from
the sign test. V1's conclusion (diag896 ranks like deployed-2048 for close variants) stands.

### Consequence 2 — the independent-arm campaign design is dead.

A 3-seed independent arm resolves 0.066 nats total / 0.029 nats on `r` at 2 SE. No recipe effect
we have ever measured in this tree is that large. An unpaired ladder would return "unresolved" for
every arm and cull nothing, at ~30 min per arm-seed.

### Consequence 3 — switch to a matched paired-by-seed design.

Run every arm at the **same seed set as the control** (24301/24302/24303), difference **within**
seed, then average the three differences. If the seed effect is common to arm and control it
cancels, and the residual is the arm effect plus whatever run-to-run noise survives at fixed seed.

This is only valid if `s_seed` is genuinely a *seed* effect rather than irreducible run noise
(CUDA nondeterminism, atomics ordering). If a rerun at a fixed seed reproduces to ~1e-6, pairing
recovers nearly all the power; if a fixed-seed rerun disperses as much as a seed change, pairing
buys nothing and the effect sizes we are chasing are simply unmeasurable at this budget.

**That single question decides whether the ablation campaign is feasible at all**, so it is being
measured before any further arm: `w05_det_s24301_rep` re-runs seed 24301 byte-identically against
`w05_probe_s24301`.

### Consequence 4 — thresholds and the tripwire denominator.

`s_seed` is now measured, so the campaign driver's refusal to cull without it is satisfiable. Cull
thresholds are denominated in the **paired-difference** SD once measured (not the table above,
which is the *unpaired* floor and is the wrong, pessimistic denominator for a matched design).

### Incidental — the screen tier does yield economics.

Contrary to the plan's assumption, a 3000-step stage-0 arm **does** write a `TradeSummary`: the
final step triggers `write_epoch_checkpoint`, so `pretrain_epoch_0_ctx896.windows.json` carries
`trade`. `pretrain_best*.windows.json` still carries `trade: null`. Screen-tier arms are therefore
judgeable on `free_kelly_saturated` and edge, not NLL alone.

**`free_kelly_saturated` = 0.689 +/- 0.008 at 3000 steps**, versus 0.848 fully trained. The T2.1
left-tail/over-betting pathology is present from the very beginning of training and is measurable
at screen budget for ~10 minutes of GPU. It is not an artifact of convergence.

### Operational note — GPU contention is a real hazard.

`w05_probe_s24303` was OOM-killed *after* promoting, during post-run reporting, when a subagent ran
`cargo test -p trading_bot_0 --lib` (12.03 GiB of CUDA) alongside two live runs. Its checkpoints and
window vectors survived and are valid; only the trailing report died. Rule for the campaign:
**`cargo check` is CPU and always safe; `cargo test` on this crate takes the GPU and must never run
while an arm is in flight.** Two concurrent 3000-step arms cost 10.2 + 6.0 GiB and run at 5.3 step/s
each (10.6 aggregate) versus 6.9 serial — a 1.53x throughput win, and the correct default.

## W0.5b — Is `s_seed` a SEED effect or irreducible run noise? **It is run noise. Pairing is dead.**

`w05_det_s24301_rep` re-ran seed 24301 at the identical invocation, identical binary, identical
frozen supports. If training were reproducible the two runs would agree bitwise.

| | nll_bar | nll_r |
|---|---|---|
| `w05_probe_s24301` | 16.286633 | 3.748105 |
| `w05_det_s24301_rep` | 16.339570 | 3.770910 |
| **fixed-seed rerun difference** | **0.052937** | **0.022805** |
| seed-change dispersion `s_seed` | 0.05754 | 0.02465 |
| **rerun / seed-change** | **0.92** | **0.93** |

**Re-running the same seed reproduces ~92% of the dispersion of changing the seed.** The seed
explains almost none of the variance. The dispersion is chaotic trajectory divergence from
nondeterministic CUDA reductions, amplified over 3000 steps.

Per-window structure of the rerun difference over the 2048 validation windows:

```
n = 2048   n_exactly_equal = 0
mean_diff = -0.052937   mean_abs = 0.053339   max_abs = 0.319773
```

Two things follow from `mean_abs` ≈ `|mean_diff|`. First, **zero windows agree exactly** — the
nondeterminism is total, not confined to a few kernels. Second, the difference is a near-uniform
shift: the rerun is worse on essentially every window by about the same amount. This is a global
model-quality difference, not evaluation scatter, so **it cannot be reduced by evaluating more
windows.** `se_win` was never the relevant denominator.

### What this means for the campaign

- **The matched paired-by-seed design of W0.5 Consequence 3 is withdrawn.** It cannot work: there
  is no common seed term to cancel. Two runs of the *same* config at the *same* seed already differ
  by a full noise unit.
- Determinism flags (disabling cuDNN autotune, `use_deterministic_algorithms`) would make two runs
  of an *identical* config agree, but would NOT help arm-vs-control: the intervention itself
  perturbs the trajectory, and in a chaotic optimizer any perturbation produces a difference of
  order the noise floor on top of its systematic effect. Determinism buys reproducibility, not
  statistical power. Not worth the throughput cost for this purpose.
- **The only lever left is n.** SE of an arm mean is `s_run / sqrt(n)`. To resolve an effect `d` on
  `r` at 2 SE: `n = (2 * 0.0246 / d)^2`. So d = 0.05 -> n = 1; d = 0.025 -> n = 4; d = 0.02 -> n = 6.
  At ~9.4 min per run and 2-way concurrency, n = 6 is ~28 min per arm. Affordable, and it is the
  price of an honest answer.
- **Corollary, and the reason the whole audit was worth doing: the production recipe lineage was
  selected on single-seed deltas of 0.13–1.08 `s_run`.** Those selections are not distinguishable
  from noise. The current "best" recipe is not established to be better than the recipes it beat.
- A cheaper route to the same end: anything that reduces trajectory noise also raises measurable
  power. Weight averaging (audit T2.5, EMA/SWA) is the obvious candidate and is independently
  motivated as a model improvement. It should be measured early for both reasons.

## B1 — Multi-horizon direct heads ON. **The implemented multi-horizon is the WRONG object.**

`--direct-t2-weight 1.0 --direct-t3-weight 1.0`. The fixed stage multipliers (0.5 for t+2, 0.25 for
t+3) already encode a geometric ladder, so effective weights are h1:1.0, h2:0.5, h3:0.25.

First, the control's heads are confirmed dead weight. Control logs `h2 23.8019 x0.0000 | h3 23.8645
x0.0000 | objective share 0.0%`: the heads run a full forward and build their graphs every step,
are multiplied by zero, and consequently sit at ~23.9 nats — between the marginal baseline 21.05 and
uniform 24.26. The shipped default pays the compute and gets nothing. With weights on they reach
`h2 16.5623 | h3 16.7012`, i.e. they do learn, and the marginal compute cost of enabling them is
approximately zero because it is already being spent.

Result at seed 24301, against the two control runs at the same seed (mean 16.3131 / 3.7595):

| | nll_bar | nll_r | edge@4x bps | `free_kelly_saturated` |
|---|---|---|---|---|
| control, mean of 2 | 16.3131 | 3.7595 | 7.30 | 0.6726 |
| B1 | 16.4602 | 3.8233 | 6.31 | 0.6614 |
| delta | **+0.1471** | **+0.0638** | −0.99 | −0.0112 |
| delta in `s_run` (t, unpaired) | **2.09** | **2.11** | — | — |

**Multi-horizon as implemented makes the tradeable factor worse**, by ~2.1 sigma at n=1 vs n=2.
Seeds 24302/24303 are running to settle it at n=3 vs n=3.

### Why — and this is the substantive finding

`world_model.rs:93-94` documents `v6 -> v7` as adding "direct-horizon **complete-bar**
supervision", and the head is `direct_logits(&h, &conditioning, &bins, 2)`: all five DOF of **the
individual bar at t+2**. That is not the multi-horizon that Part 1 §1.4 argues for.

- Predicting the *individual* bar at t+2 carries the **same one-bar SNR** as t+1 while conditioning
  on strictly less information. It is a harder task with **no** SNR benefit.
- The §1.4 benefit comes from the **cumulative** h-bar aggregate, where drift accumulates ∝ h and
  noise ∝ sqrt(h), so SNR grows ∝ sqrt(h). Nothing in the training objective predicts that.
- `shared-grad cosine 0.8991, conflict 15.2%` is the confirmation. Near-parallel gradients mean the
  t+2 task demands the *same* features as t+1 — it is **redundant**, not complementary. So handing
  it 43.7% of the objective buys dilution of the only tradeable factor and no new information. The
  conflict diagnostic was looking for the wrong failure: the problem is not that the auxiliary
  fights the primary task, it is that it duplicates it.

### The correct change

Supervise the **cumulative h-bar aggregate**: close at t+h relative to t, plus the running max and
min over those h bars. That object is (a) higher-SNR by sqrt(h), (b) exactly MFE/MAE, hence exactly
what stops, targets and triple-barrier touch probabilities need (audit T2.6), and (c) **already the
object the inference path consumes** — `horizon.rs` Monte-Carlos it with 96 paths x 3 replicates
over 100 bars via ancestral rollout, and carries a `plain_mu_se` diagnostic precisely because that
sampled estimator is noisy. A direct head would supply it with lower variance and lower cost.
Prediction, falsifiable: cumulative-h heads should show a **markedly lower** shared-grad cosine than
0.899, because they must demand genuinely longer-horizon features.

Note this also revises Part 2 T1.3, which recommended flipping the two default weights as a free
win. That recommendation is **wrong on the measurement** and is withdrawn. The defaults are already
correct; the heads should be replaced, not enabled. What remains true is that the control wastes the
forward pass on heads it zeroes.

### B1 settled at n=3 vs n=4 — **REFUTED, and the audit's T1.3 recommendation is withdrawn.**

All four control observations (three seeds plus the seed-24301 rerun, which is a legitimate
independent draw given W0.5b) against three B1 seeds:

| run | nll_bar | nll_r | edge@4x | satur |
|---|---|---|---|---|
| control s24301 | 16.28663 | 3.74811 | 7.54785 | 0.68160 |
| control s24302 | 16.25513 | 3.73208 | 8.00591 | 0.69798 |
| control s24303 | 16.36674 | 3.78047 | 7.02455 | 0.68683 |
| control s24301-rerun | 16.33957 | 3.77091 | 7.04975 | 0.66362 |
| B1 s24301 | 16.46022 | 3.82329 | 6.31175 | 0.66142 |
| B1 s24302 | 16.36419 | 3.78614 | 7.01682 | 0.68793 |
| B1 s24303 | 16.57815 | 3.86977 | 5.52455 | 0.67146 |

| metric | ctl mean (n=4) | B1 mean (n=3) | delta | SE | t | verdict |
|---|---|---|---|---|---|---|
| nll_bar | 16.31202 | 16.46752 | +0.15550 | 0.06682 | 2.33 | **B1 worse** |
| nll_r | 3.75789 | 3.82640 | +0.06851 | 0.02656 | 2.58 | **B1 worse** |
| edge@4x bps | 7.40702 | 6.28438 | −1.12264 | 0.49003 | −2.29 | **B1 worse** |
| `free_kelly_saturated` | 0.68251 | 0.67360 | −0.00890 | 0.01054 | −0.84 | unresolved |

Three independent metrics — a distributional one, its tradeable marginal, and a realized economic
one — all move against the arm, each beyond 2 SE, and all three B1 seeds are individually worse than
the control mean on `nll_r` (3/3 sign agreement). The conclusion is not marginal.

**Enabling the existing complete-bar multi-horizon heads costs ~0.069 nats on the tradeable factor
and ~1.1 bps/bar of edge.** The defaults were right. The heads should be replaced with cumulative
h-bar aggregate supervision, not switched on.

Methodological note, and the payoff from W0.5: this experiment was correctly sized in advance. The
effect on `nll_r` (0.0685) sits just above the n=3-vs-n=4 detection threshold (2 SE = 0.053). Under
the old in-tree practice — one seed per arm, judged against `se_win` ≈ 0.0004 — the same data would
have been reported at ~170 sigma, and a 0.0077-nat *non*-result elsewhere would have been reported
as significant too. Reusing the control pool across arms means each additional arm costs only its
own 3 runs, ~28 min at 2-way concurrency.

## C1 — The in-tree calibration panel. **Two corrections to Part 2, and independent corroboration of T1.1.**

Source: the diagnostic panel printed by any completed run. Numbers below are control seed 24301,
test split, 2048 windows / 256 blocks, at the 3000-step screen budget. None of this required new
code; it was already being measured and was not being read.

### C1.1 — T1.1 is corroborated by the repo's own diagnostic, which names the same mechanism.

```
fitted full law vol quartile 0: realized sd 15.67 bps/bar, predicted  22.39 (1.43x too wide), var slope +0.2322
fitted full law vol quartile 1: realized sd 30.12 bps/bar, predicted  42.21 (1.40x too wide), var slope +0.2965
fitted full law vol quartile 2: realized sd 45.18 bps/bar, predicted  62.80 (1.39x too wide), var slope +0.4005
fitted full law vol quartile 3: realized sd 85.83 bps/bar, predicted 107.63 (1.25x too wide), var slope +0.6980
slope gradient per decade of realized sd: var +0.5896 (se 0.0844), mean +0.1193 (se 0.0888)
  "the spread over-statement is WORST IN THE QUIETEST names, which is what an ABSOLUTE
   MISPLACEMENT OF MASS does and a bulk error does not"
```

And the interior-only decode inverts in the tails:

```
interior-only quartile 0: realized 15.69, predicted 15.08 (0.96x), var slope +1.3614
interior-only quartile 3: realized 87.33, predicted 48.12 (0.55x), var slope +6.1274
interior-only slope gradient per decade: var +6.5973 (se 0.8396)
```

On one global absolute grid, quiet names get mass placed too wide and volatile names too narrow, and
the error is **monotone in realized vol**. That is the signature of a single pooled absolute quantile
support, i.e. exactly T1.1. The diagnostic reaches the same conclusion unprompted and calls it an
absolute misplacement of mass. T1.1's priority is confirmed by measurement, not just by argument.

`slope gradient per decade of realized sd` is promoted to the **primary one-number test of T1.1**:
it is a scalar, it already carries a standard error, and it must collapse toward 0 if
standardization does what it claims. The four quartile ratios should also converge toward each other
(flattening matters more than the level).

Related, and traded-relevant: `mean slope across blocks: sd 0.4881 against a 0.1572 noise floor =
3.11x, excess sd 0.4621 — HETEROSKEDASTIC: the miscalibration is not common, so a scale-free ranking
of these forecasts sorts partly on WHICH BLOCK rather than on signal strength.` Cross-sectional
ranking is how the forecast gets traded, so a block-varying miscalibration is a direct alpha leak.

### C1.2 — CORRECTION to T2.3: the 0.6653 mean slope is budget-dependent, not a property of the model.

Part 2 T2.3 cited MZ mean slope 0.6653 +/- 0.0286 and concluded predicted `mu` is ~50% too large, so
Kelly `f` is ~50% too large. That figure comes from a full-length, fully-annealed one-epoch run
(`bardist_v3_rfirst_1ep`, step 10817). At the 3000-step screen budget the same quantity is:

```
moment-correct quadratic-Kelly recalibration: mu -> -3.05405e-5 + 0.9734 * mu
  (a bar at the median |mu| is repriced by -2.7%)
```

**0.9734, i.e. essentially calibrated.** So the mean miscalibration is a property of the annealed
end-state, not of the architecture. Consequences: A2 (`--mean-shrink`) is near-inert at screen
budget and MUST be validated at full length; and any cheap run that "disproves" A2 is measuring the
wrong regime. T2.3 stands for production, which is fully annealed, but its magnitude does not
transfer down to the screen tier.

### C1.3 — CORRECTION to T2.1/T2.2: the sizing error is NOT a simple over-bet.

```
predicted sd level: 66.73 bps/bar RMS over the traded bars
spread OVERSTATED: the predicted variance is 1.63x the realized one. Combined with the mean slope
  the implied Kelly scale is b_var/b_mean = 0.63x the growth optimum, so the two miscalibrations
  partly cancel in ABSOLUTE size and neither one alone describes the sizing error. What the mean
  recalibration corrects is the OVER-DISPERSION of the mean ACROSS bars (1.03x too variable),
  which is what decides the allocation once a cap binds and absolute scale stops mattering.
```

On the full law the model **over**-states variance by 1.63x, which alone makes Kelly size 0.61x too
small, and the net implied scale is **0.63x of the growth optimum — under-levered in absolute
terms.** Both facts hold simultaneously: the uncapped optimum exceeds the 4x cap on 68-85% of bars
while still sitting below what a correctly calibrated law would ask for.

So the Part 2 framing of `free_kelly_saturated` as evidence of over-betting is wrong as stated.
Saturation is evidence that the *ratio* `mu/sigma^2` is large and that the cap binds, not that the
model is over-levered relative to the growth optimum. A1 remains worth doing on its own merits —
the exact `E[ln(1+fR)]` versus the quadratic surrogate **on the same law** — but it should not be
sold as a cure for over-betting, and its sign should be reported rather than assumed.

What survives, and is arguably the sharper point: once a cap binds, absolute scale stops mattering
and **allocation is decided by the cross-bar dispersion of `f`**, not its level. That makes the
`1.03x too variable` mean over-dispersion the operative defect, and it makes cross-bar sd of `f` the
quantity to report for any sizing change.

### C1.4 — Where the tail risk actually lives.

The interior-only decode predicts 48.12 bps against 85.83 realized in the top vol quartile (0.55x,
var slope +6.13). The outer catch-all bins are therefore carrying the variance, and any exact
expected-log solve will be dominated by them. This makes A3 (GPD tail splice on the outer bins) and
A1 tightly coupled: an exact solve over a two-point catch-all is exact arithmetic over a wrong
object. A1 must consume the persisted fitted per-bin moments *including* the catch-alls.

## B2 — `--lr-plateau-fraction 0.9`. **REFUTED, decisively, and T2.4's premise is inverted.**

T2.4 argued that the default `F = 0.40` forces every one-epoch run to finish fully annealed at a
badly-calibrated operating point (MZ slope 0.6653), while one full pass at peak rate measured 1.0058,
and that raising `F` toward 1.0 would reach the calibrated point. Zero-code arm, so it was tested
first. It is wrong at this budget, and the failure is instructive.

Seed 24302 (completed) against the 4-run control pool:

| metric | control (n=4) | F=0.9 | delta | in `s_run` |
|---|---|---|---|---|
| nll_bar | 16.31202 | 16.91912 | **+0.60710** | ~10 sigma |
| nll_r | 3.75789 | 3.97378 | **+0.21589** | ~9 sigma |
| edge@4x bps | 7.40702 | 5.09859 | −2.30843 | — |
| `free_kelly_saturated` | 0.68251 | 0.66636 | −0.01615 | — |
| full-law var slope gradient / decade | +0.5896 (se 0.0844) | +0.8122 (se 0.2059) | worse | — |
| implied Kelly scale vs growth optimum | 0.63x | 0.67x | unchanged | — |

The effect is an order of magnitude above the noise floor, so n=2 is already conclusive and no
further seeds are needed for the verdict.

**Seed 24301 did not merely underperform, it diverged and was refused by a guard:**

```
pretraining failed: the promoted checkpoint's dynamics head is WORSE THAN DOING NOTHING:
dyn/identity is 185785783.145 on the test split at horizon 1, where 1.0 is the trivial
identity map.
```

That is 1.86e8 against an identity baseline of 1.0, versus 1.026 on the seed that completed. So
holding peak LR for 90% of the run is not just worse on average, it is **unstable**: one of two
seeds destroyed the dynamics head outright. The guard did its job and refused the artifact, which is
a point in the repo's favour — this failure mode is caught rather than shipped.

Calibration, the thing the arm was supposed to fix, got worse in every respect. On the failed seed
the panel reads `implied Kelly scale = 2.56x the growth optimum` and `OVER-DISPERSION of the mean
ACROSS bars 4.26x too variable` against control values of 0.63x and 1.03x, with the interior-only
var slope gradient at +17.86 against +6.60. **Annealing is what produces the good calibration at this
budget, not what destroys it.**

Reconciling with T2.4's cited evidence: the 1.0058 peak-rate slope came from a full-length run at
step 10364, where the model has seen a full pass. At 3000 steps, peak rate simply means
under-converged, and an under-converged model is both worse and worse-calibrated. T2.4's bracket is
a full-budget observation and cannot be reached from the screen tier. **T2.4 is withdrawn as
written**; the residual open question — whether a full-length run at high `F` reaches a better
economic operating point — costs ~20 GPU-hours to answer and is not worth it ahead of T1.1.

### B3 — the informative follow-up: anneal HARDER

The sign of the B2 result is the useful part. If holding peak rate for 90% of the run costs 0.61
nats, the derivative points the other way, and the default `F = 0.40` has never been checked against
a smaller value. `--lr-plateau-fraction 0.2` is running as B3. This arm was not in the original
ladder; it exists only because B2 failed in a direction that carried information, which is the
entire argument for running cheap zero-code arms before expensive ones.

### B3 result — `--lr-plateau-fraction 0.2`. **NULL. The zero-code knobs are exhausted.**

| metric | control (n=4) | F=0.2 (s24301) | delta | in `s_run` |
|---|---|---|---|---|
| nll_bar | 16.31202 | 16.31483 | +0.00281 | 0.05 |
| nll_r | 3.75789 | 3.76064 | +0.00275 | 0.11 |
| edge@4x bps | 7.40702 | 7.01956 | −0.38746 | 0.79 |
| `free_kelly_saturated` | 0.68251 | 0.67033 | −0.01218 | — |

Indistinguishable from the default. Twenty times below the detection floor on the primary metric, so
more seeds cannot rescue it — an effect this small is economically irrelevant even if real. Combined
with B2, the LR schedule is **flat between F = 0.2 and 0.4 and only breaks at the extreme**, so the
default 0.40 sits in a plateau, not on a slope. Arm closed at n=1; no further seeds spent.

### Interim conclusion of the cheap-arm phase

Every zero-code arm available on the shipped CLI has now been tested at the screen budget:

| arm | change | verdict | effect on `nll_r` |
|---|---|---|---|
| B1 | multi-horizon heads on | **refuted** (n=3) | +0.069, 2.6 sigma worse |
| B2 | LR plateau 0.9 | **refuted** (n=2, 1 seed diverged) | +0.216, ~9 sigma worse |
| B3 | LR plateau 0.2 | **null** (n=1) | +0.003, 0.11 sigma |

**The shipped recipe is at a local optimum with respect to its own exposed knobs.** Nothing is left
on the table that a flag can reach: two of three arms are actively harmful and the third is inert.
This is a genuinely useful negative result, because it settles that further gains must come from
*structural* change rather than tuning, and it cost ~2 GPU-hours to establish rather than being
assumed. It also retires two of the audit's own Tier-1/Tier-2 recommendations (T1.3, T2.4) on
measurement.

The campaign therefore moves to T1.1 (volatility-standardized targets), which is the item the
repo's own calibration panel independently corroborates in C1.1, and which requires a new bin
geometry rather than a flag.

## E1 — T1.1 first launch. **Crashed at step 6. The divisor has no floor.**

Geometry fit succeeded: `long_data/bars/bar_supports.volstd.300.json`, format_version 7,
`has_moments: true`, provenance present, fitted by a 1-step run under the flag. Both training seeds
then died within six steps:

```
pretraining failed: packed loss/gradient diagnostics are not finite at step 6:
[33.2297, 33.2100, 6.7057, ... , inf, 3.9508, 0.97957, -3.1559312158770804e20, ... NaN x20]
```

The shape is diagnostic: the first two entries are ~33.2 nats where the control runs ~16, then `inf`,
then `-3.16e20`, then everything downstream NaN. `standardize_dof` computes `z_r = r / sigma_t`, and
`har_variance` is a Garman-Klass-Yang-Zhang per-bar variance, so a near-zero-range bar (illiquid or
halted name) drives `sigma_t` toward zero and `r / sigma_t` overflows f32. `RangeVolHar::reference()`
correctly returns `None` before the first observation — the leak guard is right — but nothing floors
`sigma_t` from **below** once defined. On a 5479-symbol / 780M-bar corpus that is certain to be hit,
and the pre-crash NLL of 33.2 nats says extreme `z` was already landing in near-zero-mass catch-all
bins for several steps before the overflow.

The floor must be **scale-free** (a fraction of the name's own trailing scale), because an absolute
epsilon re-introduces precisely the cross-sectional scale dependence T1.1 exists to remove. Assigned
with a required clamp-rate counter — a silent clamp firing on a few percent of bars would invalidate
the arm invisibly.

## C2 — `pretrain_vol_vs_har` is **MEANINGLESS AS PRINTED.** The V3 verdict metric is broken.

A 1-step, untrained model — held-out NLL 24.3686 against uniform 24.2602, i.e. it has learned
nothing and is slightly *worse* than uniform — printed:

```
har-rv  4.83317 (2.69397..8.41065)   +0.00000   1.0291   -1.3866
the model BEATS the causal HAR-RV baseline by 3.13433 qlike (1.01858..6.70462)
```

What is **sound** and must not be touched: QLIKE is the Bregman form `z - ln z - 1`, `z =
realized/predicted` — non-negative, minimized at the proxy-noise floor, and invariant to a common
rescale of both sides. Causality is structural (forecasts emitted before bar `t` folds into state),
refitted per bar per window, never fitted on train, with a bit-identity perturbation test. The sign
is correct and the block bootstrap is genuinely paired (per-window differences, shared blocks and
seed). The metric's *shape* is right.

### The arithmetic that convicts it

`qlike = E[z] - E[ln z] - 1` and the printed `log bias` **is** `E[ln z]`. So HAR's 4.83317 with log
bias −1.3866 implies **`E[z] = 4.4466`**, while its printed level ratio `E[rv]/E[p] = 1.0291` says it
is mean-**unbiased**. Mean-unbiased with a mean per-bar ratio of 4.45 is the exact signature of
predictions that are far too small on a *subset* of bars: tiny predictions barely move `E[p]` but
explode `E[z]`, and QLIKE punishes that linearly. **That `E[z]` excess of ~3.45 is the entire printed
3.13433 gap.** Nothing about the model is in the number.

### Root cause: the baseline's magnitude is unfloored

`har.push(match fitted { Some(value) if value > 0.0 => value, _ => features[1] })` — the **sign**
failure is caught, the **magnitude** failure is not. And the regressors are degenerate: `component()`
clamps `from = bar.saturating_sub(lag)`, so with lags 93/465/1953 all three components are the same
expanding mean below bar 93 and two are identical below bar 465, while the ridge is 1e-8 relative.
Fitting starts at bar 186 and scoring at 372, so many scored bars are forecast by an OLS solved on
near-exactly-collinear rows of heavy-tailed RV. Occasional tiny positive fits pass the `> 0.0` gate.

### And the "model" row is not the model

The headline row is `VOL_MODEL_SCALED = scale * model` with a causal running
`scale = sum_{j<t} proxy / sum_{j<t} model`. For an untrained model whose `predicted_var` is
near-constant, that row collapses to the **expanding mean of the proxy** — a strong nonparametric
baseline. So the bench compared two baselines and printed the model's name on the winner.

### Confirmed unit mismatch under the T1.1 flag

`trade_bench.rs` never consults `DofScaling`. Under standardization `predicted_var` is in `z_r^2`
while the realized proxy is `GK_raw / (4 ln 2 * sigma_t^2)`, because `standardize_dof` divides `r` by
`sigma_t` but `s` by `BAR_RANGE_TO_SIGMA * sigma_t` (1.6651). The two sides differ by a constant
`4 ln 2 = 2.77259` absent on the raw path, and the docstring's `E[garman_klass] = sigma_t^2 = Var[r]`
is silently false under the flag. Falsifiable check: row 0's `E[rv]/E[p]` should read ~0.361.

### The structural point, which outlives the bug fix

`sigma_t` is itself a causal GKYZ HAR-EWMA. So under T1.1 the scored proxy is **the residual of a HAR
volatility forecast**: the signal HAR-RV exists to capture has already been divided out before either
side sees it. Even with a perfectly floored baseline, "does the model beat HAR-RV" is not the
question this bench answers under the flag. The fix is to de-standardize both sides
(`predicted_var * sigma_t^2` vs `proxy * (BAR_RANGE_TO_SIGMA * sigma_t)^2`) so the model must beat
HAR on **total** variance — which is the only version of the question worth asking, and is also the
honest test of whether T1.1's analytic divisor plus a learned innovation beats HAR outright.

### Why it was never caught

The guard exists and is correctly specified: `har.qlike.mean < 2.0 * scores[VOL_MODEL_SCALED].qlike.mean`,
commented "the har baseline is a straw man". It runs **only** on a synthetic 600-bar fixture
(`sigma2 * Exp(1)` on a smooth sine). The real run violates that exact assertion at 4.83317 vs
1.69884 = **2.85x**. A correct assertion that never sees real data is not a guard, and this is the
second time today the same pattern has appeared: a well-specified check exercised only where it
cannot fail. Promoting it to a runtime refusal is part of the assigned fix — a model may only be
credited with beating HAR when HAR is a functioning forecaster.

**Consequence for the campaign: V3 ("does the model beat HAR-RV on volatility") cannot be answered
until this lands, and the audit's T3.2 remains open rather than favourable.**

## A1 — Second-cumulant expected-log sizing. **CONFIRMED against the global quadratic surrogate.**

Measured by the recovered binary's then-named `pretrain-trade --exact-kelly` path against the same
command without it, on FROZEN control checkpoints. The implementation is not exact integration of
the empirical within-bin law: it evaluates log wealth at each fitted bin mean with a second-order
within-bin variance correction. The evidence below therefore confirms that specific cumulant
correction, not an exact expected-log optimum. The methodologically important part remains: a
decision-side change requires **no retraining**, so the comparison is exactly paired on the same
weights and bars and the 0.058-nat trajectory noise that dominates training arms is absent. Each
measurement took ~30 seconds and ~0 GPU-hours.

| checkpoint | paired growth Δ @ 4.0x (bps/bar) | 95% CI | se | sigma | control's ruin-domain bars |
|---|---|---|---|---|---|
| s24301 | **+0.2683** | +0.1915..+0.3542 | 0.0417 | 6.4 | 19.755% |
| s24302 | **+0.2384** | +0.1587..+0.3349 | 0.0442 | 5.4 | 22.903% |
| s24303 | **+0.2036** | +0.1396..+0.2732 | 0.0340 | 6.0 | 20.213% |

### The cap sweep shows where the correction matters

| cap | paired Δ (bps/bar) | verdict |
|---|---|---|
| 0.25x | −0.0001 | ns |
| 0.50x | −0.0002 | ns |
| 1.00x | −0.0030 | ns |
| 2.00x | +0.0543 | significant |
| **4.00x (deployed)** | **+0.2683** | **significant** |
| 8.00x | +0.2693 | significant |
| 16.00x / 32.00x | −0.1504 | ns |

The terms the global quadratic surrogate discards matter only once `|f|` is large, and the observed
shape is consistent with that mechanism: inert at small caps and beneficial in the mid band. It
also confirms the audit's prediction that the effect would be negligible at the 0.25x selection
cap — measured at −0.0001, ns. The audit's magnitude claim was too aggressive: it predicted the
neglected term was worth the entire ~0.6 bps/bar selection edge, whereas the measured gain at 4x is
~+0.24 bps/bar against a ~7.4 bps/bar edge, about 3%.

### The risk finding, which matters more than the growth number

**On ~1 bar in 5 (19.8-22.9%) the global quadratic surrogate selected an `f` outside the open
domain implied by live fitted bin means.** The cumulant-corrected rule refuses those positions.
This is strong evidence that the quadratic surrogate omits economically material tail information,
but it is not a proof of the ruin domain of the full within-bin empirical law: catch-all-bin
dispersion is still represented by fitted moments rather than direct quadrature. Uncapped, the
cumulant rule is also consistently more conservative (`|f*|` 8.21 vs 8.85, 9.10 vs 10.19, 8.28 vs
8.98) with slightly lower cross-bar dispersion (−1.7% to −2.1%).

**Promoted to default** on this paired evidence, with the global quadratic surrogate retained as an
explicit `--quadratic-kelly` opt-out. The cap sweep remains a standing regression test of the
sizing path; exact empirical-law Kelly remains a separate unimplemented method.

### Methodological conclusion worth generalizing

Decision-side changes must be measured on frozen checkpoints, not by retraining. Retraining injects
a 0.058-nat noise floor that has nothing to do with the intervention and forces n>=3 seeds per arm;
the frozen-checkpoint route is exactly paired, ~1000x cheaper, and yields 5-6 sigma on the first try.
A2, A3 and every future sizing or calibration change should be measured this way.

## V3 — Does the model beat a volatility baseline? **YES against EWMA-RV; HAR requires a repaired rerun.**

From the same frozen-checkpoint benches, on 256 windows / 113,841 scored bars / 256 blocks:

| forecast | qlike | vs har-rv | E[rv]/E[p] | log bias |
|---|---|---|---|---|
| model `Var[r|past]` | **1.129 – 1.152** | −1.408 to −1.430 | 0.459 – 0.472 | −1.45 to −1.52 |
| model, causal scale | 1.151 | −1.408 | 1.047 | −0.715 |
| **ewma rv** | **1.652** | −0.907 | 1.149 | −1.348 |
| har-rv | 2.559 | 0.000 | 1.016 | −1.255 |

**The model beats a causal EWMA realized-vol forecast by ~0.51 qlike on every checkpoint.** EWMA is
the right baseline to lean on here: it involves no fit, so it cannot be degenerate the way the HAR
OLS is, and it is a genuinely strong nonparametric vol forecaster. This closes T3.2 in the model's
favour and retires the audit's concern that no vol baseline existed.

The historical run simultaneously confirmed C2: **har-rv at 2.559 was far WORSE than plain EWMA at
1.652.** The recovered implementation now floors/reduces/refuses weak HAR fits and withholds every
versus-HAR dispersion when the remaining baseline is degenerate. That repairs the measurement
contract, not the historical number: `pretrain_vol_vs_har` is not quotable until a current-schema
run records a functioning baseline.

Note the tension with C1.1, which is real and not a contradiction: the model beats EWMA on the
*average* QLIKE while its variance forecast remains badly miscalibrated *across the cross-section*
(interior-only quartile ratios 0.97 / 0.90 / 0.79 / 0.60). Being right on average and wrong per-regime
is precisely the T1.1 defect, and `E[rv]/E[p] = 0.46` on the unscaled model row says the level is
over-predicted by ~2.2x even as the ranking is good. T1.1 remains the priority.

## Recovery — Wave 0 v6. **RAW CONTROL SELECTED; REPLICAS STOPPED AFTER PRIMARY EVIDENCE.**

Date 2026-08-28. The abandoned overlapping edits were recovered without reverting the wider
private-head/direct-objective work. `cargo check -p trading_bot_0 --tests` and the campaign
driver's offline decision suite pass. An independent red-team review rejected the first queue:
jobs 3964-3970 were cancelled or dependency-skipped before execution because their test gate read
the mutable checkout, their workers did not recheck hashes, and the raw and standardized support
files had different corpus fingerprints.

The replacement support pair was fitted back-to-back by the recovered immutable binary. Both
artifacts authenticate corpus
`334ee0f1a9751d6037991b2a9d8f54315bc4b2d95708c19b6ab39c58853489dc`, canonical split bounds,
4,000,000 training rows, seed 24301, and the same support-fitting semantics:

- raw v6: `long_data/bars/bar_supports.raw.300.seed24301.v6.json`, SHA-256
  `58c87dfc2f9dbba4e45981d8b07d4ec7d5befbe79ed898a63e1dda8f78344d45`
- standardized v8: `long_data/bars/bar_supports.volstd.300.seed24301.v8.json`, SHA-256
  `af3070feae1aa17ed1794d483521c35291f0878c1666bca8613a7230aa73ed2a`

The standardized contract includes the causal relative sigma floor and explicit numerical-fallback
and clamp diagnostics. A replay against the raw path with seed 24302 was refused because the
persisted fit seed is 24301; the fit command no longer accepts or reports an unauthenticated caller
seed.

The first full immutable harness exposed 12 failures before any training job ran. Those included
stale packed-schema/context fixtures, an unguarded Torch seed, numerically vacuous sizing/HAR
fixtures, a stale report label, obsolete uniform-head assumptions, and one real `+0.0.signum()`
bug that treated a zero signal as long instead of abstaining. Jobs 3991-3998 were dependency-skipped.
After repair, immutable mlq job 4003 completed the full library harness: 816 passed, 0 failed,
7 ignored.

The validated campaign executable is
`training/campaign_binaries/wave0-v5-recovered-9791829f`, SHA-256
`9791829f9c60632294df36fe3fb26e1071d177625b045f08ab424eddae72335b`. The matching frozen test
harness is `training/campaign_binaries/wave0-repaired-tests-b7da1fdc`, SHA-256
`b7da1fdcc2cb18f9568857d5b387766fe9f9088b676178fab647f89e85db6a9b`. Every worker
authenticates the launcher, executable or test harness, selected target supports, and frozen market
supports immediately before `exec`.

The earlier pre-execution review also caught that the promoted `pretrain_best.windows.json`
producer omitted the binding trade summary. Jobs 3981-3987 were cancelled or dependency-skipped.
The producer now attaches the TradeBench measured on the exact promoted panel, so saturation, ruin
and calibration evidence cannot be missing at confirmation.

Jobs 4005-4007 completed the current-schema raw controls at seeds 24301/24302/24303. Their
mean `selection_edge_bps` values were 0.741377, 0.737948 and 0.730441 bps/bar over 256 blocks,
giving the screen noise floor `s_seed = 0.00559334` bps/bar. E1 job 4008 then failed at step 1702:
the runtime's binding packed loss/gradient diagnostic check found non-finite values and exited 101.
This is hard tripwire 1, which culls at any tier without consulting a threshold. Jobs 4009-4012
were dependency-skipped. The last useful E1 diagnostic checkpoint from step 1000 is preserved;
neither it nor its incomplete treatment is eligible for model selection.

The full E1 dependency graph, resolved pins, fit provenance, failed harness evidence, repaired
harness result, control noise, hard-cull evidence hash, and all superseded batches are recorded in
`training/campaign_manifests/w0v6-e1-bootstrap.json`.

The raw-target control also remains ahead of B1. Corrected jobs 4023-4025 completed the matched
three-seed treatment with t+2/t+3 direct-head weights set to 1.0; comparator job 4026 observed
paired edge deltas +0.0187, -0.0139 and +0.0027 bps/bar, mean +0.0025. The apparent gain cannot
advance: seed 24302 regressed r NLL by +0.0073 +/- 0.0002 nats, a resolved hard-tripwire veto.
All other tripwires were clear, including 1.000 paired correlations, zero ruin bars and unchanged
support identity. Offline-replay recorder job 4033 persisted the `CULL` verdict. The failed
argument-only launch and two failed recorder attempts are preserved separately from the valid
training evidence in `training/campaign_manifests/w0v6-b1-screen.json`.

Jobs 4034, 4036 and 4038 completed B2's 0.85 learning-rate plateau screen. Exact comparator job
4039 measured edge deltas -0.0030, -0.0303 and -0.0055 bps/bar, mean -0.012933. Seeds 24301 and
24302 also regressed r NLL by +0.0075 and +0.0068 nats respectively, each with 0.0002-nat SE.
Recorder job 4050 therefore persisted another hard-tripwire `CULL`; all other tripwires were clear.
The evidence is in `training/campaign_manifests/w0v6-b2-screen.json`.

No treatment carried, so the raw control is the selected configuration. Predeclared primary job
4057 completed one exact 19,635-step pass and selected step 18,000 on validation economics at the
deployed 2,048-bar context. The immutable artifact is
`training/runs/w0v6_control_s24301_deploy/weights/pretrain_best.ot`, SHA-256
`5d29c9ab0879d136759365a0c4903af3259639970826fea0570a2ce0c7b80aa5`; its exact 4,096-window
sidecar has SHA-256 `48304471d16779feaef9b23f36805dbab9dfe6b15e1d11470d3b4f3f4303bc38`.
On terminal Test it scored 15.9365 nats/bar and beat the calibrated marginal by 5.1121 nats/bar.
At the frozen 2 bps research cost its 4x policy beat the marginal null by 5.8721 bps/bar
(95% CI 4.2849..7.6906), with zero ruined bars, mean MZ slope 0.9694, and break-even cost
3.71 bps. The economics-primary checkpoint also beat the run's NLL-only final-step comparator by
0.0050 bps/bar at 0.25x on the same Test panel.

This is evidence to keep the primary model, not evidence to run it live at any cost. The separately
measured production cost is 10.620 bps, above the 3.71 bps break-even, and the 4x policy's Test
maximum drawdown was 80.5%. Live-trading approval is therefore withheld even though the frozen
2 bps research benchmark is decisively positive. On user priority, jobs 4058 and 4059 were
cancelled rather than spend more GPU time on non-comparative same-configuration replicas; seed
24301 remains the predeclared selection, not a post-hoc winner. The completed decision and
immutable pins are in `training/campaign_manifests/w0v6-deploy.json`.
