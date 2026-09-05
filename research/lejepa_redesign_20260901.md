# Replacing the Deterministic LeJEPA Transition with a Distributional World Model

Date: 2026-09-01. Author: lejepa-design agent. Status: design proposal (no implementation).

## 0. Executive summary

The deterministic MSE latent transition + separately fitted emission probe fails for a
structural reason, not a tuning reason: MSE trains the predictor to output the conditional
mean of the next latent, and the post-hoc probe then conditions the entire bar distribution
on that single point summary. Both the transition and the emission discard exactly the
information a trading world model exists to carry — the conditional distribution.

Recommended design (**DBWM — Distributional Belief World Model**): keep the encoder,
SIGReg, and causal trunk; replace the deterministic predictor head + post-hoc probe with
two jointly trained heads on the trunk belief:

1. a **belief-conditioned 5-DOF autoregressive categorical emission head** (reusing the
   `BarEmissionHead` machinery from `bar_dist.rs`, 128 bins, smoothed CE) — the primary
   loss and the object that defines multi-horizon bar NLL; and
2. a **noise-conditioned stochastic latent predictor** trained with the energy score — a
   strictly proper scoring rule over samples — giving one-network-evaluation sampling of
   the next latent for cheap latent-only imagination rollouts.

Rollouts become ancestral: sample (latent or bar), feed the sample back, advance the
belief. The RL policy consumes fans of sampled trajectories, with per-horizon quantiles
derived from the fan. Fallback: swap the energy-score latent head for conditional flow
matching if calibration diagnostics show under-dispersion.

## 1. What le-wm actually is, and where we diverge

Read: `/home/marvin/Documents/repositories/le-wm/{jepa.py,module.py,train.py,eval.py,config/}`.

**Objective.** Exactly two losses (`train.py:39-41`): `pred_loss = (pred_emb - tgt_emb)^2`
(1-step MSE next-embedding prediction, `num_preds: 1`) and SIGReg (Epps–Pulley Gaussianity
statistic over 1024 random projections, weight 0.09). No EMA target, no stop-gradient, no
stochastic machinery of any kind.

**Architecture.** ViT-tiny encoder → CLS token → BatchNorm-MLP projector (192-d);
predictor is a 6-layer causal transformer with AdaLN-zero conditioning on an encoded
**action** sequence, over a context of only `history_size: 3` frames; a `pred_proj`
BatchNorm-MLP after the predictor.

**Rollout / end use** (`jepa.py:61-153`). The rollout appends **predicted embeddings**
back into the 3-frame context autoregressively — the same mean-feedback recursion we do.
But the rollout's only consumer is MPC planning (CEM/Adam solvers in `config/eval/solver/`):
cost = MSE between the final predicted embedding and an encoded **goal image**, per action
candidate. le-wm **never decodes observations**, never scores likelihoods, and never
needs the transition distribution.

**Why deterministic works there and not here.** In their environments (pusht, cube,
reacher, two-room) the transition is nearly deterministic *given the action*; the action
sequence carries most of the entropy and it is supplied exogenously by the planner. And
goal-distance planning is tolerant of conditional-mean predictions. Trading inverts both
premises: there is no exogenous action driving the market (our transition is a pure
forecast), the next bar is dominated by irreducible noise, and the end use — imagination
for an RL trading policy — needs calibrated conditional densities of future bars, not a
point in latent space.

**Conceptual diff vs. our `trading_bots/src/torch/lejepa/`:**

| aspect | le-wm | ours |
|---|---|---|
| encoder | ViT-tiny on pixels, CLS + BN-MLP projector | per-bar MLP (`bar_proj`+enrich) + RMS-norm MLP projector, 256-d; token has **no temporal context** |
| predictor | AdaLN transformer over 3-frame history, action-conditioned | 6-layer causal PoPE/FA4 trunk over 6000 bars → 2-block gated-MLP head |
| loss | MSE(1-step) + SIGReg(0.09) | same objective, faithfully mirrored |
| actions | yes (AdaLN conditioning) | none (correct — market is action-free) |
| emission | **none** | post-hoc fitted `TokenEmissionProbe` (our addition) |
| rollout use | MPC goal cost only | recursive generation + bar NLL scoring |

Conclusion of the diff: our implementation is a faithful port of le-wm's objective; the
failure is not an implementation divergence. le-wm simply contains nothing addressing
multimodality or stochasticity, because its use case never exercises them. There is
nothing to import from it that fixes this.

## 2. Failure-mode diagnosis

Observed: latent MSE 0.449 (h1) → 0.940 (h100); bar NLL 24.66 → 39.50. Emission/re-encode
feedback: latent MSE ~14% **worse**, NLL unchanged.

**Defect 1 — the transition is a conditional mean.** MSE trains
`z_hat = E[z_{t+1} | history]`. A bar is mostly irreducible noise, so the conditional mean
latent is heavily shrunk toward the regime mean and is not the embedding of any realizable
bar. Recursive feedback of means compounds the shrinkage: the context drifts into a region
no real sequence occupies, and both latent MSE and NLL degrade with horizon. This is
regression-to-the-mean rollout collapse, the standard failure of deterministic world
models under stochastic dynamics.

**Defect 2 — the emission conditions on the wrong variable.** In
`train/mse_jepa/probe.rs`, `TokenEmissionProbe::logits(tokens, ...)` conditions
`p(bar_{t+1})` on the **predicted point token** — a 256-d summary trained only to be a
conditional mean. Even though the trunk belief `h_t` demonstrably contains distributional
information (volatility regime, etc.), the MSE training signal never asks the predicted
token to carry second moments, and the probe never sees the belief. The full conditional
`p(bar_{t+1} | h_t)` is bottlenecked through `p(bar_{t+1} | E[z_{t+1}|h_t])`. The probe is
also fitted post-hoc against a frozen representation, so nothing in training ever
optimizes bar likelihood end-to-end.

**Reinterpreting the emission-feedback experiment.** The ~14% latent-MSE worsening under
sampled re-encoded feedback is **expected, not evidence of failure**: the conditional mean
minimizes L2 by construction, so any genuinely stochastic rollout scored by MSE against
the realized future must be worse by roughly the per-step conditional variance. The
informative result is that NLL was *unchanged*: injecting on-manifold samples into the
context cannot help when the emission's conditioning variable (the predicted point token)
still discards the belief. This localizes the binding constraint to Defects 1+2 jointly —
the conditional transition/emission distribution — exactly as suspected.

**The right frame.** Bars are fully observed. The true state is the history; the object
to model is `p(bar_{t+1} | bar_{<=t})`. A "belief state" here is just the trunk's causal
summary `h_t` — there is no hidden state requiring posterior inference, so POMDP machinery
(ELBO, KL balancing) is unnecessary. Under full observability, an RSSM reduces exactly to:
deterministic recurrent path (trunk + KV cache) + a stochastic variable injected per step
(the sampled next observation or next latent) + a distributional emission. Our winning
"LLM-like" `BarWorldModel` (`world_model.rs`) **is** this reduced form: trunk beliefs,
jointly trained 5-DOF AR categorical emission, ancestral sampled feedback with KV cache
(`imagine()`), plus a cheap drifting `BarDynamics` substitute. That is *why* it wins. The
redesign should give the JEPA pretrainer the same distributional skeleton while keeping
what JEPA adds (SIGReg-shaped continuous latent, cheap latent-only imagination).

## 3. Design-space analysis

Setting shared by all options: trunk belief `h_t` (256-d, 6000-bar causal context),
target latent `z_{t+1}` = encoder token of the next bar, and the trading requirement of
S-sample × 100-step imagination fans. "NFE" = predictor-network evaluations per imagined
step per sample; every option also pays one trunk KV-decode per fed-back token.

**(a) Conditional flow matching in latent space.** Learn velocity `v(z_tau, tau, h_t)`
(MLP; Fourier features of tau); CFM loss `||v - (z1 - z0)||^2` with `z0 ~ N(0, I)`;
sample by Heun integration. Fit to failure mode: excellent — models the full conditional,
handles multimodality; notable synergy with SIGReg, which already pushes token marginals
toward isotropic Gaussian, so the base distribution matches and transport paths are short.
Rollout: K = 8 Heun steps ≈ 16 NFE per step per sample; a 32-sample × 100-step fan costs
~51k small-MLP evals per window — feasible but 8–16× the one-shot options, and it recurs
inside RL imagination, our hottest loop. Training is simulation-free and cheap. tch-rs:
trivial (randn, lerp, MLP; hand-rolled Heun loop). Risks: no closed-form likelihood (need
an emission head anyway to score bar NLL); sample quality vs. K tradeoff; latent-space
calibration is unverifiable without decoding. Strong **fallback**, not the first choice.

**(b) Latent diffusion.** Same modeling power as (a) with more NFEs (or distillation
complexity), a noise schedule and SNR weighting to tune. Dominated by flow matching in
this regime (low-dim latents, MLP-scale networks). Rejected.

**(c) Discretized latents (FSQ/VQ) + AR cross-entropy transition.** Quantize the token
(FSQ, e.g. 8 dims × [8,8,8,6,5,5,4,4] levels) and train a categorical transition over
codes. CE over discrete supports demonstrably handles this data's multimodality (our LLM
baseline wins with it). But quantizing the latent of a *single bar* duplicates, one level
up, what the bar-level 5-DOF binning already does — with added quantization error and (for
VQ) codebook-collapse management. It becomes compelling only with temporal abstraction
(tokens summarizing multiple bars), which is out of scope here. Ranked below joint
emission; revisit for hierarchical horizons.

**(d) RSSM / Dreamer-style stochastic belief model.** Categorical or Gaussian stochastic
latent + deterministic recurrent state + KL-balanced ELBO. Designed for partial
observability of pixels; under fully observed bars the posterior collapses onto the
encoder and the KL machinery (free bits, balancing coefficients) adds instability knobs
without adding modeling power over the reduced form. Its essential insight — deterministic
path + per-step stochastic injection — is retained by the recommended design. Rejected in
full form.

**(e) Direct conditional density over bars, trained jointly.** Belief-conditioned 5-DOF
autoregressive categorical head (reuse `BarEmissionHead`: chain r→s→u→v→w, 128 bins,
prefix embeddings, smoothed CE with sigma-ratio 0.75), gradients flowing into the trunk.
Fit: attacks both defects head-on — joint training optimizes bar likelihood end-to-end,
and conditioning on `h_t` removes the point-token bottleneck. It is the exact mechanism
our winning baseline uses, so empirical risk is minimal. Rollout: ancestral — 5 sequential
categorical samples + tiny encoder re-encode + KV decode per step (≈ LLM-model cost).
Risks: convergence toward duplicating `BarWorldModel`; the JEPA-specific additions must
earn their keep (made an explicit evaluation gate below); 128-bin resolution limits
extreme-tail granularity (mitigated by existing smoothed/density scoring machinery).

**(f) Noise-conditioned one-shot stochastic predictor + proper scoring rule.**
`z_hat = g(h_t, eps)`, `eps ~ N(0, I_32)`; train with the **energy score** using m
samples per position: `ES = (1/m) sum_i ||z_hat_i - z|| - (1/(2 m (m-1))) sum_{i != j}
||z_hat_i - z_hat_j||`. Strictly proper for distributions with finite first moment, so the
minimizer is the true conditional distribution — this is the minimal genuinely
distributional upgrade of the current MSE head (MSE is the m=1, spread-free degenerate
case). Fit: gives calibrated *sampling* of the next latent at **1 NFE**, enabling cheap
latent-only imagination (no emission sampling + re-encode per step). tch-rs: trivial
(cdist or broadcasted norms). Risks: energy score is known to be statistically weak at
detecting dependence-structure errors in high dimensions and can under-disperse — must be
paired with explicit calibration diagnostics; m multiplies only the small predictor-head
activations (m=4 is cheap; no gradient accumulation needed or wanted).

**MC-dropout / ensembles** (rest of (f)): capture epistemic uncertainty only; the failure
mode here is aleatoric (irreducible bar noise). Rejected as the primary mechanism.

## 4. What the RL policy should consume

The policy learns from imagination rollouts; the uncertainty representation must be
(i) actionable per-trajectory (P&L is a path functional: stops, drawdown, compounding are
nonlinear in the path, so per-horizon marginal quantiles are insufficient), and
(ii) cheap. Therefore: **sampled trajectories as the primary interface** (fan of S
trajectories of beliefs + sampled bars, matching the existing `imagine()` contract), with
**derived per-horizon quantile summaries** (cumulative-return q05/q25/q50/q75/q95 across
the fan) available as compact risk features. Explicit belief distributions (parametric
heads over returns) are useful as auxiliary features but cannot replace path samples for
a policy that manages positions through time. This favors designs with cheap per-step
sampling — a further argument for (f) over (a)/(b) as the rollout workhorse.

## 5. Recommendation

### 5.1 Top design: DBWM — Distributional Belief World Model (= e + f)

Replace the deterministic transition + post-hoc probe entirely (no compatibility path;
delete the probe-fitting flow).

**Networks** (all in `lejepa/model.rs`):
- Encoder + projector + SIGReg: unchanged.
- Causal trunk: unchanged (beliefs `h_t`, 256-d).
- **EmissionHead** (new): belief-conditioned 5-DOF AR categorical over the *next* bar.
  Reuse `BarEmissionHead` from `bar_dist.rs` (BAR_CHAIN order, NUM_BAR_BINS=128,
  BAR_PREFIX_EMBED_DIM=32 prefix slots, smoothed CE at BAR_LABEL_SIGMA_RATIO=0.75),
  conditioned on `normalize_last_dim(h_t)`. Applied at all 6000 positions,
  teacher-forced.
- **StochasticPredictor** (new, replaces `PredictorHead`): same gated-MLP capacity
  (in_proj widened to LATENT_DIM + 32 for the noise input), `z_hat = g(h_t, eps)`,
  `eps ~ N(0, I_32)`.

**Losses** (in `train/mse_jepa/mod.rs::objective`):
```
L = L_CE_emission  +  alpha * L_ES_latent  +  lambda * L_SIGReg
```
- `L_CE_emission`: smoothed CE, mean over positions and DOF (identical scoring contract
  to the LLM baseline so NLLs are directly comparable).
- `L_ES_latent`: energy score with m = 4 samples per position against the same-encoder
  target token (both branches attached, as today; SIGReg plus the ES spread term jointly
  prevent collapse).
- Starting hyperparameters: alpha = 1.0, lambda = 0.09 (unchanged), noise dim 32, m = 4.
  Optimizer unchanged (extend the Muon allowlist to the new heads' 2-D weights; route
  `bar_dof_head`/`bar_prefix_embed`-style parameters to AdamW per
  `BAR_EMISSION_ADAMW_NAME_SUBSTRINGS`).
- Keep `prediction_mse` (computed from the mean of the m ES samples) and persistence
  skill as continuity diagnostics only — no longer losses... the ES already subsumes MSE.

**Training loop changes:** single teacher-forced pass as today; emission head adds one
gather + small GEMM chain per position; ES multiplies only predictor-head activations by
m. Batch 8 × 6000 remains comfortably in memory; no gradient accumulation.

**Rollout algorithm** (replaces `recursive_rollout` / `emission_feedback_rollout`):
- *Latent mode* (RL imagination workhorse): at step k, sample `z_hat = g(h, eps)` (1 NFE),
  append `z_hat` to the context via KV-cache decode, repeat. Decode bar distributions from
  the belief only where the policy needs bars/rewards.
- *Bar mode* (exact ancestral, for NLL scoring and candle fans): sample the 5 DOF
  sequentially from EmissionHead given `h`, decode to a transition-7 row
  (`emitted_dof_to_transition7` exists), re-encode through the frozen encoder, append.
- *Scoring*: predictive NLL at horizon k = `-log( (1/S) * sum_s p(bar_real | h_s,k) )`
  (log-mean-exp across an S-sample fan) — a proper predictive likelihood, unlike the
  current point-token probe NLL.
- *Policy interface*: S = 32 trajectory fan + per-horizon cumulative-return quantiles.

**Why this beats the alternatives for this failure mode:** it removes both defects
directly (joint training, belief conditioning), uses the mechanism already proven to win
on this data (CE over discrete bar supports), keeps rollouts at 1 NFE per step for the RL
hot loop, and retains JEPA's additions (SIGReg latent geometry, latent-only imagination)
as testable extras rather than load-bearing assumptions.

**Explicit evaluation gate (null hypothesis).** DBWM must beat the plain LLM-like
`BarWorldModel` on the per-horizon predictive-NLL profile at matched compute. If the
SIGReg latent + ES head contribute nothing over the baseline, the honest conclusion is to
retire the separate JEPA pretrainer, not to iterate on it.

### 5.2 Fallback: conditional flow matching transition (design a)

Swap StochasticPredictor's ES loss for CFM: velocity MLP `v(z_tau, tau, h)` (tau Fourier
features, 2× gated blocks), loss `||v(z_tau, tau, h) - (z1 - z0)||^2` with
`z_tau = (1 - tau) z0 + tau z1`, `z0 ~ N(0, I_256)`; sample with Heun, K = 8. Everything
else (emission head, rollout structure, scoring) is unchanged — the two latent heads are
drop-in alternatives behind the same sampling interface, which is the main reason to
build DBWM's rollout around "sample next latent" as an abstraction.

**Switch triggers** (all observable from the new reports):
- calibration ratio `E||z_hat - z_hat'|| / E||z_hat - z||` persistently below ~0.8 on
  validation (ES under-dispersion signature). Nominal is exactly 1.0: against a REALIZED
  draw `z` of the same conditional law, `E||z_hat - z_hat'|| = E||z_hat - z||` under
  perfect calibration. (An earlier revision divided by `sqrt(2) * E||z_hat - z||`; that
  factor is only correct against the conditional MEAN, and against a realized draw it
  pins the healthy value at `1/sqrt(2) ~ 0.707`, poisoning the trigger.)
- latent-mode rollout NLL exceeding bar-mode rollout NLL by > 2 nats at h >= 16
  (latent samples off-distribution for the trunk);
- STEP-LEVEL quantile coverage (50%/90%) under-covering by > 10 points at h >= 16 in
  either mode while bar-mode coverage is nominal, or bar-mode CUMULATIVE coverage
  under-covering likewise. Latent mode reports step-level coverage only: its per-horizon
  bar draws are marginals of independently advanced beliefs, so no cumulative-return law
  exists for it and none is reported.

### 5.3 Full ranking

1. **DBWM (e + f)** — direct fix, proven mechanism, cheapest rollouts.
2. **DBWM with flow-matching transition (a)** — strictly more expressive latent sampler
   at 16× rollout NFE; adopt on the triggers above.
3. **FSQ latent + AR CE (c)** — only worth it with multi-bar temporal abstraction;
   otherwise duplicates bar-level binning.
4. **Latent diffusion (b)** — dominated by (a).
5. **Full RSSM/ELBO (d)** — machinery for a partial-observability problem we do not have.

## 6. Metrics and reports (non-negotiable plumbing)

All emitted through `MseJepaReporter` (`train/mse_jepa/reports.rs`), with every new base
added to `MSE_JEPA_REPORT_BASES`, `shared::report::PRETRAIN_REPORT_BASES`, and
`meta_chart_bases` in `tui/src/main.rs` (the reporter asserts registry membership, and
the existing tests cross-check the registries).

New bases:
- `mse_jepa_emission` — train/val smoothed CE, per-DOF CE (r, s, u, v, w).
- `mse_jepa_energy_score` — ES, sample-spread `E||z_hat - z_hat'||`, calibration ratio.
- `mse_jepa_rollout_nll` — per-horizon (1, 4, 16, 39, 78, 100) predictive NLL, latent
  mode and bar mode side by side, plus the LLM-baseline NLL for the gate in 5.1.
- `mse_jepa_rollout_calibration` — per-horizon coverage (50%, 90%) of the return DOF
  from the sampled fans: step-level for both modes, cumulative for bar mode only (see
  5.2 — latent mode defines no cumulative law).
- Retained: `mse_jepa_loss`, `mse_jepa_objective`, `mse_jepa_representation`,
  `mse_jepa_optimization`; retire the four `mse_jepa_probe_*` post-hoc-probe bases with
  the probe-fitting flow (their evaluation roles move to the joint-model rollout bases).

## 7. Implementation scope (realistic, tch-rs)

- `lejepa/model.rs`: EmissionHead wiring + StochasticPredictor (~250 lines; all ops
  exist — embedding gathers, GEMMs, randn, multinomial sampling already used by
  `BarSupports::sample`).
- `train/mse_jepa/mod.rs`: objective + metrics packing (~150 lines).
- `train/mse_jepa/probe.rs`: delete post-hoc fitting; rewrite as joint-model rollout
  evaluation (~600 removed / ~300 added; `emitted_dof_to_transition7`,
  `sample_rollout_fans`, panel plumbing reusable).
- `reports.rs` + `shared/src/report.rs` + `tui/src/main.rs`: new bases (~150 lines).
- Checkpoint bundle (`lejepa/checkpoint.rs`): new tensors, version bump; no back-compat
  per project policy.
- No new dependencies; no gradient accumulation; fallback (5.2) is an additional ~150
  lines behind the same sampling interface.
