# DBWM v3: conditional flow-matching latent transition

Date: 2026-09-01. Status: accepted design, implementation in progress. Supersedes the
energy-score `StochasticPredictor` of v2 (`research/lejepa_redesign_20260901.md` §5.1) with
the §5.2 fallback, built to current best practice.

## 0. Evidence that motivates v3

- v1 (256/6/1536/4 trunk + energy score, run `mse_jepa_dbwm_v1_rollout`): train calibration
  ratio 0.56 → 1.08 over training; latent-mode and bar-mode fan NLL are within ±0.1 nats of
  each other at every horizon (18.2–18.9), but both trail the LLM baseline by 0.55–0.96 nats
  at every horizon. Latent-mode 50% step coverage under-covers (0.36–0.47 vs 0.50 nominal)
  while bar-mode is near nominal.
- v2 (512/10/2048/8 trunk, ES with 4× predictor expansion) OOMed at step 1 on the 32 GB card.
- The energy score is only weakly sensitive to dependence structure in 512-d and its m=4
  estimate is high-variance; it also multiplies predictor activations by m. Flow matching
  gives a full conditional generative model at 1× head activations per position.

## 1. Mental model of the optimum (and where latent FM sits in it)

The encoder is per-bar and near-bijective on the 7 transition features, so
`p(z_{t+1} | h_t)` is the pushforward of `p(bar_{t+1} | h_t)` through the encoder: a ≤7-dim
manifold embedded in R^512. Consequences:

1. **Bar-mode ancestral rollout is the exact sampler** of the model's own likelihood and is
   cheaper than any iterative latent sampler (5 tiny AR heads + encoder per step). It stays
   the reference rollout and the NLL-scoring path.
2. **The latent transition's value is as a training signal on the trunk/encoder** (JEPA's
   predictive-representation claim) and as cheap latent-only imagination. Its objective
   must be a proper generative objective — the conditional mean (MSE) is off-manifold and
   the energy score is a weak proxy. Conditional flow matching is the strongest
   simulation-free choice: exact in the limit, no likelihood machinery needed, and SIGReg
   already makes the token marginal ≈ N(0, I), matching the flow's base distribution.
3. **The evaluation gate stays the LLM baseline's per-horizon NLL profile.** If neither
   rollout mode beats it, the JEPA latent path is not earning its keep. Note the v3 trunk
   (6 layers / FFN 1024) is smaller than the baseline (10 / 2048); the gate is therefore
   conservative against v3, not for it.

## 2. Architecture

Trunk: `LATENT_DIM=512, AR_LAYERS=6, AR_FF_DIM=1024, HEADS=8, HEAD_DIM=64`. Head dim stays
64 so the FA4/PoPE contract (`pope.rs`, `fa4.rs`) is untouched. Encoder, projector, SIGReg,
`BarEmissionHead` (belief-conditioned, in the loss), `token_probe` (diagnostic) unchanged.

**FlowPredictor** (replaces `StochasticPredictor`; velocity parameterization):

```
inputs: x_tau in R^512 (interpolant), tau in (0,1), h = normalize_last_dim(belief) in R^512
t_emb  = mlp(sinusoidal(tau; 256 freqs, DiT convention))      -> 512
c      = silu(h_proj(h) + t_emb)                               -> 512   (conditioning vector)
x      = in_proj(cat(x_tau, h))                                -> 512   (direct belief path)
for each of 3 blocks:
    (shift, scale, gate) = mod_k(c)                            -> 3 x 512, zero-init
    x = x + gate * mlp_k(norm(x) * (1 + scale) + shift)        (gated MLP, hidden 2048, as v2 PredictorBlock)
(shift_f, scale_f) = mod_final(c)                              zero-init
v = out_proj(norm(x) * (1 + scale_f) + shift_f)                out_proj zero-init
```

Zero-init of every modulation projection and `out_proj` makes the network the identity
map with zero velocity at init (adaLN-zero). Those tensors — `mod_*`, `out_proj`, the
time-embedding MLP, biases — are routed to AdamW; Muon's orthogonalised update destroys
zero-initialised identity paths (lesson from branch `4ae64d8d`). `in_proj` and block MLP
weights stay on Muon.

## 3. Objective

```
per position (batch x 6000):
  tau ~ LogitNormal(0, 1)        # sigmoid(randn); emphasises mid-path, SD3 default
  x0  ~ N(0, I_512)
  x_tau = (1 - tau) x0 + tau z1  # z1 = same-pass next-bar token, attached (SIGReg prevents collapse, as v2)
  u   = z1 - x0
  L_flow = mean_dims || v(x_tau, tau, h) - u ||^2

L = L_CE_emission + lambda_flow * L_flow + lambda_sigreg * L_SIGReg      (lambda_flow default 1.0)
```

One (tau, x0) draw per position; 48k positions per batch is ample. Velocity net runs under
the bf16 autocast region like the trunk; tau, x0, and the loss are fp32. No time-dependent
loss weighting: uniform CFM weight is the rectified-flow default and does not starve any
regime. No EMA, no stop-grad, no gradient accumulation.

## 4. Sampler

`sample_next_latents(beliefs, samples, steps)`: Heun on the uniform grid tau_k = k/K:

```
x <- randn(S*rows, 512)
for k in 0..K:
    v1 = v(x, tau_k);  xe = x + d v1;  v2 = v(xe, tau_{k+1});  x <- x + d/2 (v1 + v2)
```

Velocity parameterisation has no singularity at tau=1, so the last step is a full Heun
step. Default K=8 (16 NFE). All fan samples are batched through the network per ODE step.
Used by latent-mode rollout (`rollout.rs`) and by validation sample diagnostics.

## 5. Reports (all through `MseJepaReporter`, registered in `MSE_JEPA_REPORT_BASES`,
`shared::report::PRETRAIN_REPORT_BASES`, `tui::meta_chart_bases`; tests cross-check)

Retire `mse_jepa_energy_score`. Add:

- `mse_jepa_flow`: train/val flow loss; train flow loss by tau quartile (4 lines).
- `mse_jepa_flow_samples` (validation, fixed noise panel, K-step Heun samples):
  sample spread `E||z^ - z^'||`, sample distance `E||z^ - z||`, calibration ratio
  (spread/distance, 1.0 nominal), energy score of the samples (m=4, continuity with v1),
  token-probe entropy on sampled tokens vs on real tokens (on-manifold check: off-manifold
  samples make the probe diffuse).

Rollout bases (`mse_jepa_rollout_nll`, `mse_jepa_rollout_calibration`) unchanged; latent
mode now uses the flow sampler.

## 6. CLI, checkpoint, runs

- `pretrain-mse-jepa`: `--lambda-flow` (1.0), `--flow-steps` (8, validation sampling).
- `evaluate-mse-jepa-rollout`: `--flow-steps` (8).
- Checkpoint bundle: version bump, new architecture fingerprint, ES tensors removed. No
  back-compat.
- Runs (via mlq): `pretrain-mse-jepa --run mse_jepa_dbwm_v3 --epochs 2 --steps 19634`,
  then `evaluate-mse-jepa-rollout` against the LLM baseline, mirroring the v2 pair.
