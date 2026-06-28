# Meta-Architecture / RL-Math Review — Persistently LOW + increasingly SPIKY PPO KL

Symptom (established fact): PPO approx-KL is persistently LOW (policy barely moves) and grows
increasingly SPIKY over training. This document maps our RL-math pipeline end-to-end with the
SPACE of every quantity, then ranks mismatches vs the known-good references
(cleanrl `iterthink_v24` beta + dreamer4), separates confirmed from speculative, and gives the
minimal fixes in cheapest-first order.

References:
- cleanrl baseline: `cleanrl/cleanrl/ppo_continuous_action.py`
- iterthink_v24 beta (+tkl): `cleanrl/cleanrl/ppo_continuous_action_iterthink_v24_beta_nocriticbias_v1.py`
- iterthink_v24 beta + HL-Gauss: `cleanrl/cleanrl/ppo_continuous_action_iterthink_v24_beta_d4hlgauss_v1.py` (+ `..._symlog_v1.py`), `cleanrl/cleanrl/shared/hl_gauss.py`
- dreamer4: `dreamer4/dreamer4/dreamer4.py`

---

## 1. The RL-math pipeline, end-to-end, with SPACE annotations

The single load-bearing invariant shared by both references: **every scalar that touches
GAE / returns / advantages lives in RAW reward space; the symlog/symexp warp lives ONLY inside
the categorical critic's encode (target) and decode (scalar). The warp is never applied to a
scalar that subsequently enters GAE.** Our implementation obeys this invariant.

### Forward (rollout) path

| Step | Where (file:line) | SPACE | Notes |
|------|-------------------|-------|-------|
| Env reward (mean over tickers) | `rollout.rs:148-150` | RAW | no transform |
| Store reward | `rollout.rs:159-163` → `s_rewards` | RAW | no normalization, no symlog |
| Critic head → bin logits | `trading_model.rs` value head | LOGITS over 255 bins on a SYMLOG grid `[-3,3]` (`hl_gauss.rs:4,7-8,42-47`) | bins are `symexp(uniform symlog grid)` |
| Decode rollout value | `sample.rs:14` `hl_gauss.decode` | LOGITS → RAW | `symexp(E_symlog[centers])`, `hl_gauss.rs:111-125` |
| Store value | `rollout.rs:169-173` → `s_values` | RAW | decode-before-GAE ✓ |
| Sample Beta action (native z∈(0,1)) | `action_space.rs:9-13` | native (0,1) | `z = ga/(ga+gb)`, clamp `1e-6` |
| Store action + old log_prob | `rollout.rs:137-146`, `sample.rs:16-17` | native z / summed log-density | z-replay buffer ✓ |
| Store done of transition t | `rollout.rs:159-168` → `s_dones` | mask | same-step convention (done[t] = transition t→t+1 terminal) |

### Advantage path

| Step | Where (file:line) | SPACE | Notes |
|------|-------------------|-------|-------|
| Bootstrap value decode | `advantages.rs:34-39` `hl_gauss.decode` | LOGITS → RAW | decoded before GAE ✓ |
| GAE delta | `gae.rs:79` `delta = r + (1-done)·γ·next_v − cur_v` | RAW | γ=0.995, λ=0.95 (`advantages.rs:50-51`) |
| GAE carry | `gae.rs:80` `last_gae = delta + (1-done)·γλ·last_gae` | RAW | done gates BOTH bootstrap and carry ✓ |
| Returns | `gae.rs:85` `returns = last_gae + cur_v` | RAW | `returns = adv + V` ✓ |
| Adv stats (logged) | `advantages.rs:62-71` | RAW (pre-shaping) | **diagnostic blind spot** — logged BEFORE shaping |
| **Adv normalize #1 (whole batch)** | `advantages.rs:75` `rank_gaussian_normalize` | RAW → ~N(0,1), bounded ≈ ±2.33 | double-argsort → quantile → `√2·erfinv` |

### Update (per epoch / per minibatch) path

| Step | Where (file:line) | SPACE | Notes |
|------|-------------------|-------|-------|
| **Adv normalize #2 (per-minibatch)** | `update.rs:182-183` `(adv−mean)/(std+1e-8)` | re-standardized per non-iid minibatch | **stacked on #1 — references do this ONCE** |
| Replay forward → new α/β, new value logits, entropy | `update.rs:214-224` | grad-carrying | fresh `1+softplus` concentrations (`action_space.rs:5-7`) |
| New log_prob at stored z | `update.rs:222-223`, `action_space.rs:19-25` | summed over action dim | z-replay; clean importance ratio |
| log_ratio / ratio | `update.rs:238,245` | — | `exp(new_logp − old_logp)` |
| approx_kl (k3) | `update.rs:368-369` | scalar, `no_grad` | `(exp(logr)−1−logr).mean()` ✓ |
| Policy clip loss (DAPO asym) | `update.rs:25-32`, called `:291` | — | `max(−A·r, −A·clamp(r,0.80,1.28))`; sign + max correct ✓ |
| Value loss (CE in bin space) | `value_loss.rs:5-13`, called `:300` | RAW returns → `encode` (symlog inside) → bins vs `log_softmax(logits)` | no value clip; space-consistent ✓ |
| Optimizer step | `update.rs:420/440` | — | EVERY minibatch |
| KL early-stop leash | `update.rs:472-475` | — | break if `mean_epoch_kl > 0.045` **OR** `last_minibatch_kl > 0.045` |

### Critic encode/decode geometry (the only place the warp lives)

- Support: `support = linspace(-3, 3, 256)` edges (SYMLOG), `centers` = midpoints (SYMLOG),
  `bin_values = symexp(centers)` (RAW, used only in tests) — `hl_gauss.rs:42-54`.
- Encode (target): `t = clamp(symlog(raw_return), -3, 3)` then HL-Gauss erf-CDF smear — `hl_gauss.rs:90-107`.
- Decode (scalar): `symexp(Σ softmax(logits)·centers)` — `hl_gauss.rs:111-125`.
- Encode and decode are a matched symlog-space pair (identical convention to cleanrl
  `shared/hl_gauss.py:62-81`). Symexp/symlog each appear exactly once per direction.

**Verified clean (NOT contributors):** symlog/symexp space-consistency; critic representation;
GAE recursion / bootstrap / done-mask alignment / chunk-boundary bootstrap; value-target
construction; value-in-GAE space match; reward normalization (absent, matches refs); Beta
concentration / log_prob / entropy / z-replay / Jacobian; PPO clip-loss sign+max+asymmetry;
gradient isolation / detach. Each was checked against both references on the seven load-bearing
invariants and matches.

---

## 2. Ranked mismatches (most → least likely to cause LOW + SPIKY KL)

### #1 — Double advantage normalization: full-batch rank-Gaussian THEN per-minibatch z-score  [CONFIRMED — high]

- **Error (placement / space-consistency):** Advantages are normalized TWICE. First a whole-batch
  rank-Gaussian shaping (`advantages.rs:75`), then a SECOND per-minibatch z-score inside the
  epoch loop (`update.rs:182-183`). The code comment even states the intent: "per-minibatch
  standardization in update.rs refines it" (`advantages.rs:73-74`).
- **Ours vs reference:**
  - ours: `advantages.rs:75` (rank-Gaussian, whole batch) + `update.rs:182-183` (`(adv−mean)/(std+1e-8)` per minibatch)
  - cleanrl baseline: ONE per-minibatch z-score of RAW adv, `ppo_continuous_action.py:277`
  - iterthink beta d4hlgauss: rank-Gauss shaping XOR z-score, applied ONCE at BATCH scope and frozen across minibatches, `..._d4hlgauss_v1.py:771-773`
  - dreamer4: ONE masked batch z-score, `dreamer4.py:4556-4558`
- **Why THIS symptom:**
  - SPIKY: rank-Gauss already yields ~N(0,1) over the whole batch. The second per-minibatch
    z-score divides each minibatch's advantages by *that minibatch's own* sample std. Minibatches
    here are `index_select` of whole `ppo_chunk_len=60` contiguous single-env chunks
    (`update.rs:86-101`), i.e. small, temporally + per-env CORRELATED, non-iid populations whose
    sample std is high-variance. A low-spread minibatch → small std → advantages blown up →
    large step → KL spike; a high-spread minibatch → advantages shrink → tiny step. Composition
    reshuffles every epoch (`update.rs:74`), so the effective per-update gain jitters — exactly
    the SPIKY, growing-with-training pattern (variance grows as the policy specializes).
  - LOW: rank-Gauss hard-caps `|advantage| ≈ √2·erfinv(0.999) ≈ 2.33` regardless of true spread
    (`advantages.rs:18-21`). With clip 0.20/0.28 and only 3 epochs, this bounds per-update
    policy movement → contained, persistently small aggregate KL.
- **Verification:** Re-read `advantages.rs:62-80` and `update.rs:181-183` directly; both the
  rank-Gauss call and the stacked per-minibatch z-score are present as described. Confidence: **high.**

### #2 — KL early-stop OR-condition fires on a single high-variance last-minibatch estimate  [CONFIRMED placement — medium]

- **Error (logic / placement):** The epoch break triggers if EITHER the epoch-mean KL OR the
  single LAST-minibatch k3 KL exceeds `TARGET_KL·KL_STOP_MULTIPLIER = 0.03·1.5 = 0.045`
  (`update.rs:472-475`). The reference breaks on the last minibatch's k3 only — but as a
  deliberate single check, not OR'd with the mean — and (critically) the reference runs many
  more epochs/minibatches so the leash is a stable trust-region, not a per-update lottery.
- **Ours vs reference:**
  - ours: `update.rs:472-475` `if mean_epoch_kl > 0.045 || last_minibatch_approx_kl > 0.045 { break }`
  - reference: `..._iterthink_v24_beta_nocriticbias_v1.py:898-899` `if approx_kl > target_kl: break` (target_kl=0.03, epoch granularity)
- **Why THIS symptom:** The last minibatch is ONE shuffled ~1/16 chunk-correlated minibatch, so
  its k3 value is a high-variance estimate. As Beta concentrations sharpen over training, that
  per-minibatch KL variance grows, so the single last-minibatch estimate exceeds 0.045
  increasingly often and aborts the epoch after only the minibatches already stepped → (a) caps
  realized update (LOW aggregate KL: epochs end early), and (b) the abort point depends on which
  minibatch landed last under the shuffle (SPIKY, growing over training). Compounds with #1
  (which is the *source* of the per-minibatch KL variance this leash trips on).
- **Verification:** Re-read `update.rs:444,450-476`; the OR with `last_minibatch_approx_kl` is
  present. Confidence: **medium** (clearly a deviation; magnitude of its contribution vs #1 is
  uncertain because #1 supplies the variance it trips on).

### #3 — Diagnostic blind spot: advantage stats logged pre-shaping  [CONFIRMED — high; not a KL cause]

- **Error (logic / observability):** `adv_stats` (`advantages.rs:62-71`) are computed on RAW GAE
  BEFORE rank-Gauss shaping and before the per-minibatch z-score, while the policy trains on the
  doubly-normalized advantages. Logged mean/min/max cannot diagnose what the policy gradient sees.
- **Why it matters:** Not a KL cause, but it HIDES the per-minibatch scale instability of #1,
  making the spiky-KL symptom easy to misattribute to lr/clip. Confidence: **high.**

### Speculative / out-of-scope (NOT confirmed contributors)

- **Critic support window too narrow (calibration, not math):** symlog `±3` → raw `±(e³−1) ≈ ±19`,
  vs cleanrl's symlog `±10`. If GAE returns regularly exceed raw `±19` they clamp at encode
  (`hl_gauss.rs:96`) and saturate the critic, biasing V and hence advantages. This is a
  tuning/monitoring question (watch `range_stats` above/below fractions, `log.rs`), not a
  space/math bug. Confidence the window is the KL driver: **low.**
- **Decode is `symexp(E[center])` (Jensen-shifted) vs dreamer4's `E[symexp(center)]`:** a genuine
  convention difference, but it MATCHES cleanrl's HL-Gauss exactly and is self-consistent with
  our symlog-space encode. Small smooth stationary value bias; no credible path to LOW/SPIKY KL.
  Confidence it is the cause: **very low.**
- **Unbounded Beta concentration growth → sharpening → spiky KL:** real dynamic, but IDENTICAL to
  the references (neither caps concentration), so not a divergence; if sharpening is the true
  driver it is a learning-dynamics/entropy-coef/leash issue, not a parameterization bug.

---

## 3. Concrete fixes (cheapest-first ordering)

### FIX A (apply first — highest leverage, ~one-line) — collapse to a SINGLE advantage normalization

Restore the references' invariant: normalize advantages **exactly once, at whole-batch scope**.
Keep the whole-batch rank-Gaussian shaping (`advantages.rs:75`) as the sole transform and DELETE
the per-minibatch re-standardization. In `update.rs:181-183`, use the shaped advantages directly:

```rust
let adv_flat = adv_mb_by_chunk.reshape([-1]); // already rank-Gaussian ~N(0,1), batch-frozen
```

(Remove the `(adv_raw_flat - mean)/(std+1e-8)` recompute.) This eliminates the non-iid
per-minibatch gain that drives SPIKY KL, and the now-frozen-across-epoch advantage matches the
iterthink-beta/dreamer4 convention. Alternative if you prefer the d4/cleanrl exact recipe: drop
rank-Gauss and do a single `(adv−mean)/(std+1e-8)` over the FULL flattened batch once in
`compute_advantages`. Either way: ONE normalization, whole-batch scope, frozen across minibatches.

### FIX B (apply second — one-line) — make the KL leash a stable epoch-level check

Drop `last_minibatch_approx_kl` from the OR in `update.rs:472-475`, breaking only on the
epoch-mean KL:

```rust
if mean_epoch_kl > TARGET_KL * KL_STOP_MULTIPLIER {
    break 'epoch_loop;
}
```

This removes the high-variance single-minibatch trigger that aborts epochs erratically as
concentrations sharpen. (After Fix A reduces per-minibatch KL variance, this leash also stops
mis-firing.)

### FIX C (cheap, observability) — log advantage stats AFTER shaping

Move/duplicate the `adv_stats` computation (`advantages.rs:62-71`) to AFTER `rank_gaussian_normalize`
(or, with Fix A, ensure the logged stats reflect the single normalization the policy trains on),
so the metric reflects trained-on advantages and the spiky-scale hypothesis is observable.

### Then re-measure

Apply A → observe whether KL spikiness collapses. Then B → observe whether realized KL rises
toward the 0.045 leash without erratic early aborts. C in parallel for observability. Only if KL
is still LOW after A+B should you revisit the speculative support-window calibration (widen symlog
support toward `±10` and watch `range_stats`) or consider the gradient-carrying analytic-KL
trust region from the `jointkl` reference (`v25_jointkl.py:869-886`).

---

## Appendix — confirmed-clean invariants (do NOT regress these)

1. Decode-before-GAE: rollout (`sample.rs:14`) and bootstrap (`advantages.rs:38`) decoded to RAW.
2. Symlog/symexp confined to `HlGaussBins` encode (`hl_gauss.rs:96`) / decode (`hl_gauss.rs:124`); never on a GAE scalar.
3. Encode and decode share one support instance / convention.
4. `returns = adv + values` (`gae.rs:85`); done gates BOTH bootstrap term and carry (`gae.rs:79-80`).
5. k3 approx_kl `(exp(logr)−1−logr).mean()` under `no_grad` (`update.rs:368-369`).
6. Beta: `1+softplus` concentrations, summed log_prob/entropy, z-replay of native sample, affine Jacobian dropped on BOTH sides.
7. Policy loss: negative sign on both terms + `max` (pessimistic), asymmetric clip oriented correctly (`update.rs:25-32`).
8. Full gradient isolation of all "old"/target tensors; only new log_prob / new value logits / entropy carry grad.
