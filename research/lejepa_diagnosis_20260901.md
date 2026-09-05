# LeJEPA (MSE-JEPA) World-Model Failure: Root-Cause Diagnosis

Date: 2026-09-01. Author: lejepa-diagnosis agent.
Scope: why the MSE-JEPA latent world model produces poor bar predictions, especially under
recursive rollout, and why the non-JEPA "LLM-like" NTP pretraining wins.
Evidence sources: code in `trading_bots/src/torch/lejepa/` and
`trading_bots/src/torch/train/mse_jepa/`, plus existing `.report.bin` artifacts read via
`report_cli` from runs `mse_jepa_transition7_full_refit_20260901`,
`mse_jepa_transition7_token_rollout_probe_20260901`,
`mse_jepa_transition7_emission_feedback_probe_20260901`, and (for the baseline)
`bardist_v7_collapse_recovery`, `w0v6_control_s24301_deploy`, `mse_jepa_ntp_probe_best_20260831`.
No new GPU work was run.

## 1. System description (as-built)

**Encoder** (`lejepa/model.rs:256-285`): strictly **per-bar**. Each 300s bar is reduced to 7
"transition7" features (`lejepa/dataset.rs:365-427`: gap, body, upper wick, lower wick,
Garman–Klass volatility, log volume delta, volume-present flag), fixed-scaled, then mapped by a
per-row MLP (`bar_proj` + enrich residual + projector MLP) to a 256-dim token. The encoder sees
**zero temporal context**; the token is a deterministic nonlinear embedding of one bar's 7
features. The transition7 basis is (given the previous bar's volume-EMA state) essentially a
reparameterization of the 5 scoring DOF (r, s, u, v) plus the volume delta — so in principle the
token could determine the price DOF exactly; the fitted decoder recovers most but not all of
that (see 2.1).

**Targets** (`lejepa/model.rs:196-198`, `train/mse_jepa/mod.rs:444-453`): the target for position
t is the same encoder's token for bar t+1, taken from the same forward pass. Both branches are
**attached** — no stop-gradient, no EMA target network (faithful to the LeJEPA paper's design;
`checkpoint.rs` stores a single var store, and tests
`prediction_and_target_branches_remain_attached` pin the attachment).

**Predictor**: a 6-layer causal PoPE/FA4 transformer over up to 6000 source tokens
(`causal_beliefs`) followed by a 2-block gated-MLP head (`predict_next`). Trained with plain
latent MSE `mean((pred - target)^2)`. Conditioning is not a bottleneck: the predictor head sits
on beliefs that attend over the full 6000-bar context.

**Anti-collapse**: SIGReg (`lejepa/sigreg.rs`) — a sketched characteristic-function penalty
pushing 1024 random 1-d projections of the token cloud (128 independent windows x 16 temporal
views) toward N(0,1). λ = 0.09. This is the only force opposing the attached-MSE pressure to
make targets predictable.

**Probe / emission channel** (`train/mse_jepa/probe.rs`): a `BarEmissionHead` (5-factor
autoregressive categorical over 128 bins per DOF, chain r→s→u→v→w) is fit **same-time**
(token z[t] → bar[t]) on frozen real encoder tokens, with the head's forecast-conditioning input
zeroed. Rollout: iterate belief → `predict_next` → append predicted token to the window; at each
horizon, the predicted token is decoded by this same-time probe and scored with hard categorical
NLL against the true future bar. The emission-feedback variant instead re-encodes an ancestrally
sampled bar into a genuine token before appending (`emission_feedback_rollout`,
`emitted_dof_to_transition7`).

**"LLM-like" baseline** (`torch/world_model.rs`, `train/pretrain.rs`): a 512-dim, 10-layer causal
transformer over bar tokens (5 DOF bin embeddings + raw DOF projection + calendar/market clock),
trained GPT-style with teacher-forced next-bar hard categorical NLL through the same
`BarEmissionHead` family (plus direct h2/h3 heads and a NextLat dynamics distillation term). Its
readout is a full factorized conditional distribution p(r,s,u,v,w | context), trained end-to-end.

## 2. Evidence

All NLL numbers are hard categorical nats/bar (sum of 5 chain factors). Reference points:
uniform = 24.253; train-marginal ≈ 21.13 (this probe's registered reference; the pretrain
baselines report 21.04 train / 21.38 val on their supports).

### 2.1 The encoder token is highly informative about its own bar

`mse_jepa_probe_token_emission`, run `mse_jepa_transition7_token_rollout_probe_20260901`:

| panel | full-bar NLL | vs train marginal | r NLL | s | u | v | w | dir. hit |
|---|---|---|---|---|---|---|---|---|
| train | 11.10 | **+10.02** | 2.08 | 1.81 | 1.43 | 1.29 | 4.49 | 0.880 |
| validation | 11.89 | **+9.23** | 2.20 | 1.86 | 1.65 | 1.53 | 4.66 | 0.903 |

The token recovers 9.2 of the ~16.5 price-DOF marginal nats (r marginal ≈ 4.69, decoded to
2.20; residual 1.3–2.2 nats per price DOF is decoder imperfection, not missing information,
since the transition7→DOF algebra is bijective). w barely improves because the token carries
only the volume delta, not the EMA reference. **Conclusion: no encoder collapse — and equally,
no abstraction: the token retains the bulk of the per-bar noise.** (Contrast: the 2026-08-31 checkpoint probed in
`mse_jepa_ntp_probe_best_20260831` had *zero* same-time skill, val 21.127 = marginal exactly; the
transition7 refit fixed encoder informativeness. The failure analyzed here is not that one.)

### 2.2 The h1 latent prediction is an almost perfectly calibrated conditional mean

End of pretraining, `mse_jepa_representation` / `mse_jepa_objective`, run
`mse_jepa_transition7_full_refit_20260901` (final steps):

- target std 0.889 → per-element variance ≈ 0.79; representation std 0.866 (stable, no collapse)
- prediction std 0.534 → prediction variance ≈ 0.285 (**60% of target std — the textbook
  mean-shrinkage signature**)
- train prediction MSE ≈ 0.55, validation 0.50

Variance decomposition check: Var(target) ≈ Var(prediction) + MSE → 0.79 ≈ 0.285 + 0.51,
closing to within a few percent on noisy single-batch stats (the identity requires errors
orthogonal to predictions, which any converged scale-calibrated MSE predictor satisfies — so
this shows calibrated conditional-mean *behavior*, not optimality). The predictor explains ≈ 36–42% of token variance (val probe h1 MSE 0.449 against val token
variance ≈ 0.78, estimated from long-horizon persistence MSE ≈ 1.56 ≈ 2·Var) and shrinks away
the rest. Given 300s vol clustering, an R² of ~0.4 on tokens dominated by range/volatility
features is plausible **genuine** conditional structure — the transition model is not learning
"near-nothing"; it beats copy-last (0.983) and the unconditional-mean baseline (≈ 0.78) cleanly
at h1.

### 2.3 But the predicted token is worse than useless through the emission channel

`mse_jepa_probe_latent_rollout` / `mse_jepa_probe_token_rollout` (recursive, mean-token
feedback) and the emission-feedback run:

| h | latent MSE (recursive) | latent MSE (feedback) | persistence MSE | NLL (recursive) | NLL (feedback) | recursive NLL vs marginal | dir. hit |
|---|---|---|---|---|---|---|---|
| 1 | 0.449 | 0.449 | 0.983 | 24.66 | 24.66 | **−3.54** | 0.531 |
| 4 | 0.531 | 0.632 | 1.031 | 28.49 | 28.07 | −7.37 | 0.531 |
| 16 | 0.615 | 0.703 | 1.150 | 31.62 | 31.22 | −10.49 | 0.484 |
| 39 | 0.728 | 0.890 | 1.314 | 33.79 | 34.00 | −12.66 | — |
| 78 | 0.815 | 0.913 | 1.548 | 36.49 | 37.06 | −15.37 | — |
| 100 | 0.940 | 0.970 | 1.563 | 39.50 | 39.61 | −18.37 | — |

Three decisive facts:

1. **At h1 — before any recursion — the WM's predicted token scores 3.54 nats WORSE than the
   unconditional marginal.** A frozen histogram with no conditioning beats the world model at
   every horizon. Return directional hit rate is 0.53 (chance). The ~0.33 latent-variance R² the
   predictor genuinely has is not merely lost in the emission channel; conditioning on the
   off-manifold mean token actively corrupts the decoder's output. The probe was fit only on real
   tokens (embeddings of actual bars); E[f(bar')] is not f(any bar), so the head extrapolates and
   emits confidently wrong bin distributions (s-NLL blows from 1.86 same-time to 5.9 at h1 and
   10.5 by h16).
2. **By h100 the recursive latent MSE (0.94) exceeds the predict-the-dataset-mean baseline
   (≈ 0.78) by ~20%.** The reported "skill vs persistence ≈ 0.4–0.5 at all horizons" is an
   artifact of persistence being a weak baseline (2·Var·(1−ρ_h) ≥ Var for ρ ≤ 0.5; token
   autocorrelation ρ₁ ≈ 0.37, → 0 by h100). Against the correct trivial baseline, the recursion
   is net-harmful beyond ~h40.
3. **On-manifold feedback does not help** (the emission/re-encode experiment): latent MSE gets
   ~12–22% worse at h4–h78, NLL is unchanged (h1 conditioning is identical by construction;
   later horizons +0.01 to +0.6 nats). Sampling a valid-but-random branch from an
   already-worse-than-marginal conditional adds trajectory variance and removes nothing, because
   the deterministic WM cannot represent the branching distribution either way.

### 2.4 The LLM-like baseline on the same yardstick

From `pretrain_nll_bar` / rollout reports (extracted by a delegated code/report survey of the
`pretrain` path; MSE-JEPA numbers above were read first-hand via report_cli):

- Validation next-bar NLL at h1: **15.89–15.94** across `bardist_v7_collapse_recovery`,
  `w0v6_control_s24301_deploy`, `directmtp_v3_privateheads` — i.e. **≈ 5.15 nats better than the
  train marginal**, vs the JEPA WM's −3.54. Headline h1 gap: **≈ 8.7 nats/bar** of conditional
  structure delivered vs destroyed.
- Its teacher-forced (exact-belief) rollout stays ≈ 17–19 nats through h100. Notably its own
  *latent* dynamics-advance rollout also blows up (up to ~97 nats at h100 for v7) — independent
  confirmation that recursive deterministic latent iteration is fragile even there; the modes
  that work advance context with (observed or ancestrally sampled) *bars*, re-encoded through
  the trunk.

## 3. Ranked root causes

**R1 — Deterministic latent-MSE prediction of an aleatoric-noise-dominated target (mean
collapse), with no distributional readout. Confidence: very high. Primary.**
The target token is a near-invertible embedding of one bar's raw features (evidence 2.1), so the
JEPA objective is literally MSE regression of the next bar's noise in a warped 256-dim metric.
The optimum of that objective is the conditional mean, and the model found it almost exactly
(evidence 2.2: variance decomposition closes; prediction std = 0.60 × target std). A conditional
mean of a multimodal/heavy-tailed bar distribution is off the sample manifold, carries no
representation of spread or modes, and any decoder fit on samples scores it below the marginal
(evidence 2.3, −3.5 nats at h1). This is the same conclusion cross-entropy vs MSE theory gives:
a 5×128-bin factorized categorical (the LLM-like head) can represent the entire conditional law;
a point estimate in latent space cannot, and at 300s the law *is* the signal — the mean is
nearly worthless (hit rate 0.53). Explanation (e) of the candidate list is this same cause seen
from the baseline's side: the LLM-like pretraining wins because its objective (teacher-forced
factorized CE) both trains the trunk to shape features for the conditional law and gives it a
native distributional emission; the ≈ 5.15-nat h1 advantage over marginal is the measure of the
conditional structure available at this resolution.

**R2 — The JEPA target provides no abstraction for prediction to succeed on: per-bar,
noise-complete targets, with SIGReg pinning the noise in. Confidence: high. Structural
co-cause.**
JEPA's premise is that predicting *representations* skips unpredictable detail. Here the encoder
is per-bar and (on price DOF) invertible, so the target contains 100% of the innovation noise —
there is nothing abstract to predict. Worse, the equilibrium of attached-MSE + SIGReg is pinned:
the attached MSE branch pressures the encoder to shed unpredictable directions (that would be
benign, informative collapse toward slow features), and SIGReg's whole job is to forbid exactly
that shedding by enforcing unit-scale Gaussian marginals. Evidence: representation/target std
held at 0.87–0.89 throughout training while 9.2 nats of per-bar detail remain decodable. So the
representation converges to "full noisy bar, isotropically spread", the worst possible JEPA
target. This subsumes candidate (b) in its correct form — it is not that the encoder captures
only static/level info (refuted by 2.1 and the real 0.42 R² at h1), it is that it captures
*everything*, including what cannot be predicted. Candidate (d) (SIGReg fights the predictor) is
true only in this indirect sense; SIGReg is doing its assigned job, the job is what's wrong in
this configuration. Note the h1 latent skill it does achieve (R² ≈ 0.4) is exactly the
predictable slice (vol/volume level) — the representation *contains* forecastable state; the
objective just cannot cash it out.

**R3 — Compounding off-manifold context under recursion. Confidence: high, but secondary
(amplifier, not origin).**
Feeding mean tokens back degrades beliefs progressively (NLL −3.5 → −18.4 nats vs marginal from
h1 to h100; latent MSE crosses the unconditional-mean baseline near h40). The emission-feedback
experiment proves this is not primarily a manifold-validity problem: genuine re-encoded tokens
make latent MSE *worse* (branch variance) and NLL no better. The rollout degrades because each
step's conditional is already bad (R1) and because a deterministic recursion cannot marginalize
over branches — consistent with the LLM-like model's own latent dynamics-advance rollout also
failing while its bar-space ancestral rollout works.

**R4 — Probe/evaluation channel mismatch (same-time decoder applied to a conditional mean).
Confidence: high that it exaggerates the measured NLL; low that it changes the conclusion.**
The emission probe is fit as z[t]→bar[t] on real tokens and then evaluated on E[z']-type inputs
it never saw; some of the −3.5 nats at h1 is decoder extrapolation error, not missing
information. A fairer channel (a *forecast* probe fit on belief[t]→bar[t+1], which trains the
head on the actual conditioning distribution and lets it express uncertainty) would score at or
above marginal and quantify the usable nats in the belief (lower-bounded by the latent R² ≈ 0.4
on vol-heavy directions, so likely a positive but small fraction of the baseline's 5.15). This
is the one cheap decisive measurement still missing; it changes how bad the number looks, not
the ranking, because the production requirement is a rollout-capable emission and the current
system has none.

**Rejected: (c) predictor under-conditioning. Confidence in rejection: high.** The predictor
attends over 6000 bars (3× the baseline's deployed context) and its h1 output is a
near-perfectly calibrated conditional mean; no evidence points at capacity or context.

## 4. What any fix must address (requirements, not designs)

1. **A distributional emission trained on the model's own conditioning** — the model must output
   p(next bar | belief) (e.g. a factorized categorical head on beliefs, i.e. NTP-style CE),
   not a latent point that a same-time decoder is asked to interpret. This is the single change
   the h1 evidence demands: −3.5 nats → the baseline shows +5.15 is attainable.
2. **Rollout by ancestral sampling in bar space with re-encoding** (as the LLM-like production
   path and `horizon.rs` already do), never deterministic latent mean iteration — both models'
   evidence shows the latter compounds.
3. **If a JEPA-style latent objective is kept, its target must be able to drop innovation
   noise**: multi-bar/coarse targets, an EMA/stop-grad target that is not variance-pinned per
   bar, or an explicitly stochastic latent (distributional prediction in z). Attached per-bar
   MSE + SIGReg at λ=0.09 mathematically fixes the current bad equilibrium (R2).
4. **Fix the baseline in reported metrics**: skill-vs-persistence overstates skill; report skill
   vs the unconditional-mean-token MSE (≈ Var ≈ 0.78) alongside it, and NLL vs marginal is the
   headline number for any emission.
5. Optional decisive diagnostic before committing: fit a forecast probe (belief[t] → bar[t+1])
   on the frozen transition7 checkpoint to measure the usable conditional nats already in the
   belief; it separates "representation is salvageable with a CE head" from "retrain end-to-end".

## Appendix: key numbers ledger

- Train marginal NLL: ≈ 21.13 nats/bar (this probe's supports); uniform 24.25.
- Same-time real-token decode: 11.10 train / 11.89 val (+9.2 nats vs marginal), hit rate 0.90.
- WM predicted-token h1: 24.66 (−3.54), h100: 39.50 (−18.4). Feedback variant: 24.66 / 39.61.
- Latent: h1 MSE 0.449, persistence 0.983, unconditional-mean baseline ≈ 0.78 (Var from
  h100 persistence 1.563 ≈ 2·Var); h100 recursive MSE 0.940 (> mean baseline).
- Pretrain final: prediction std 0.534, target std 0.889, representation std 0.866;
  0.285 + 0.503 ≈ 0.788 = Var(target) (conditional-mean decomposition closes).
- LLM-like val h1 NLL: 15.89–15.94 (+5.15 vs train marginal); exact-belief rollout ≈ 17–19
  nats through h100; its latent dynamics-advance rollout degrades to 46–97 nats.
- 20260831 checkpoint (pre-transition7): same-time decode val = marginal exactly (dead tokens);
  superseded by the transition7 refit.
