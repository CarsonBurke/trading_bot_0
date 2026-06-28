# Generalized State-Dependent Noise Design (Lattice Successor)

Goals: maximize learnability and returns via emergent, state-dependent entropy; eliminate mean/noise interference; avoid collapse without hard clamps.

1) Actor architecture
- Shared trunk: keep the existing actor latent used for the mean.
- Noise adapter: small linear + GELU on the shared latent; gradients from the noise head do not flow into the mean head.
- Heads:
  - Mean head: unchanged action logits.
  - Noise head: outputs `c_raw ∈ R^k` (correlated coeffs) and `s_ind_raw ∈ R^{A}` (independent scales), where `k = min(32, A)`.

2) Noise basis and covariance
- Learn `W_noise ∈ R^{A×k}` with soft column orthogonality (‖WᵀW − I‖² penalty, small weight). No coupling to policy weights.
- Scales use tempered softplus: `softplus_τ(x) = τ·log1p(exp(x/τ))`, τ ≈ 0.7.
- Base scales to prevent collapse: `base_corr = base_ind = 0.1`.
- Gating (state dependent): compute `u = Huber(|GAE|)`; z-score with batch stats + EMA; stop-grad. `g = sigmoid(β·u)`, β ≈ 1.5.
- Apply gate to only half of the learned scale: `corr_scale = base_corr + softplus_τ(c_raw) * (0.5 + 0.5*g)`; same for `ind_scale`.
- Covariance: `Σ = W_noise · diag(corr_scale²) · W_noiseᵀ + diag(ind_scale² + ε)`, ε = 1e-6. No `-0.5 ln dim` shrink; no detach on the covariance path.

3) Stability (soft, not hard clamps)
- Orthogonality penalty on `W_noise`; tiny jitter ε in Σ.
- Optional soft spectral guard: light log-barrier on largest eigenvalue or Frobenius penalty on Σ if explosions appear.
- Gradient hygiene: smaller lr or gradient clip for noise-head params if early instability; keep entropy coef parameter available but set to 0 by default.

4) Defaults and toggles
- k = min(32, A). If ACTION_DIM grows or Σ underfits, bump k.
- Alignment regularizer to policy span: OFF by default; only enable (with detached policy SVD) if span drift correlates with return or conditioning regressions.
- Dimensional scaling (std · dim^{-0.5}): OFF by default; use only if early variance explosions persist.

5) Training/monitoring
- Track: logit entropy, Σ eigenvalue spread, principal angles between W_noise span and policy span, gate statistics, and action std mean/min/max.
- Fail-safe: if entropy collapses, raise `base_corr/base_ind` slightly; if Σ ill-conditioned, increase orthogonality penalty or turn on spectral guard.

Implementation recipe (any codebase)
- Add noise adapter + noise head MLP on the actor latent; keep mean head unchanged.
- Add learned W_noise with orthogonality regularization.
- Replace covariance construction with the gated softplus + base formula above; remove latent-dim shrink and detach.
- Add uncertainty gate (Huberized |GAE|, z-scored with EMA, stop-grad) and wire it into scales.
- Add monitoring hooks for Σ stats and gate stats; keep entropy coef exposed (default 0).
