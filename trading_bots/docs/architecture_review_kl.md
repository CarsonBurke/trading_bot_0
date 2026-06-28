# Architecture Review: Persistently-Low + Increasingly-Spiky PPO KL

Synthesis of a full-stack architecture review investigating why the trading model's
per-update PPO KL divergence is **persistently low** (policy barely moves between
updates) yet becomes **increasingly spiky** (occasional KL bursts that grow in
magnitude and frequency as training proceeds).

Inputs: reference notes from two known-good baselines
(`orbit-wars`, an RL bot; `parameter-golf`, LLM pretraining), per-component findings,
and adversarial verification verdicts. Epistemic stance: no single confirmed root
cause; this is a multi-factor diagnosis with explicitly separated well-supported and
speculative items.

## Empirical grounding of the symptom

Two runs were inspected directly:

- Short run (`training/training.log`, ~40 episodes): KL pinned at 0.0000-0.0008, no
  spikes. This run is too short to exhibit the growing-spiky half.
- Long run (`training/runs/2026-06-05_18-22-18-868917453/training.log`, 3863 KL
  records): KL **mean grows monotonically** across quartiles
  (Q1 0.00718 -> Q2 0.01273 -> Q3 0.01623 -> Q4 0.01719) and the **max spike grows**
  to 0.1451 (vs TARGET_KL=0.03). Within a single late update, KL jumps ~11x from
  epoch 1 to epoch 2.

Conclusion: the dual symptom is **real in extended training**. Mean KL stays low
relative to the per-sample bursts, and the bursts intensify over training and across
PPO epochs within an update. This temporal structure (spikes grow with optimizer
steps, epoch 2-3 worse than epoch 1) is the single most discriminating piece of
evidence and rules out several otherwise-plausible mechanisms (notably static input
artifacts and weight-independent precision noise).

---

# 1. Architecture Map

## 1.1 Observation / Reward

**Price-delta token stream (dominant input).** Raw close prices -> per-step log
returns `ln(p_t/p_{t-1})` (`utils.rs:203`), first element 0. 5-minute bars; invalid
bars dropped (not interpolated, `historical.rs:203-212`). Per ticker
`PRICE_DELTAS_PER_TICKER=6000`, `TICKERS_COUNT=1` (`shared/src/lib.rs:13,17`). Scale
~1e-3 std, heavy-tailed (gaps reach +/-0.1..0.5), **never clamped/winsorized**. The
active variant is `UniformStream` (`trainer.rs:107`): deltas reshaped `[B,120,50]`,
NaN tail masked, `fill_fraction` appended, projected by a single Linear
`patch_stream_proj` (`trading_model.rs:531-573`). No per-patch standardization in the
active path (the unused Base variant does standardize, `trading_model.rs:494-504`).

**Static observation vector** (`STATIC_OBSERVATIONS=44/45`, `obs.rs:11-94`): 6
account + 14 macro globals + 25 per-ticker features. Almost everything is clamped to
[-1,1]/[0,1] (`macro_ind.rs:157-222`, `momentum.rs:69-128`, `earnings.rs:96-108`).
Outliers: `pnl` and `commissions/STARTING_CASH` are unbounded (`obs.rs:21,27`); `macd`
is clamped to +/-5 (5x its neighbors, `momentum.rs:128`).

**Reward** (`reward.rs:209-264`, live path `get_unrealized_pnl_reward_breakdown`,
wired at `step.rs:110,180`; ~10 other reward fns are dead code). Per-step reward =
`ln(total_assets_next / pre_total_assets) * REWARD_SCALE` with `REWARD_SCALE=20`
(`reward.rs:5,236`). With TICKERS_COUNT=1 the per-ticker decomposition is a no-op.
Typical per-step reward ~+/-0.1..1.0.

## 1.2 Tokenization / Input

**Patch construction** (`trading_model.rs:447-573`). Active path: 120 uniform
50-wide patches; each patch masks NaN, appends `fill_fraction`, projects
`51 -> model_dim`. **No per-patch mean/std normalization** in the active path.

**Input normalization (chokepoint #1).** Both paths feed `input_ln` —
an **affine-free RMSNorm** (eps 1e-6, no learnable gain, `rmsnorm.rs:6-19`) — applied
per-token immediately after projection, before the trunk (`trading_model.rs:469-470`).
This pins layer-0 token magnitude to unit RMS regardless of weight or input scale,
matching `parameter-golf` post-embed `rms_norm` and `orbit-wars` `embed_norm`. Init:
`patch_embed_weight`/`patch_stream_proj` Xavier/truncated-normal (non-orthogonal,
unlike references).

**Exogenous (portfolio) tokens** (`trading_model.rs:418-445`,
`blocks/exogenous.rs:120-138`). Each static feature becomes its own rank-1 token
`feat_i * exo_feat_w[i] + exo_feat_b[i]`, then `exo_embed_ln` (RMSNorm) + ExoMLP.
`NUM_EXO_TOKENS=45`. These are K/V for a cross-attention block injected after GQA
layer 0.

## 1.3 Backbone

**GQA self-attention trunk** (`blocks/gqa.rs:31-153`). 3 pre-norm layers, model_dim
256, ff_dim 512, 4 query heads / 1 KV head, head_dim 64. Per forward: resid-mix
`x*resid_mix[0] + x0*resid_mix[1]` (init (1,0) passthrough); attn_ln RMSNorm -> fused
QKV -> **per-head QK-RMSNorm on Q and K** -> RoPE on first 16 of 64 dims -> Q scaled
by learnable per-head `q_gain` (init **1.0**, lowered from references' 5.0 by commit
c6306d39); SDPA bidirectional, scale 1/sqrt(64); `attn_out` (zero-init) * `attn_scale`
(init 1.0) added to residual; then squared-ReLU FFN, `ffn_fc2` zero-init, * `mlp_scale`
(init 1.0). Zero-init residual projections make each sublayer identity at init.
Attention logits at init ~O(1), bounded.

**RoPE** (`rope.rs:1-69`). Standard partial RoPE, base 1e4, 16/64 dims rotated,
parameter-free, gradient-free, norm-preserving, applied after QK-RMSNorm. Active
UniformStream uses uniform spans so `arange` positions are semantically correct.

**Cross-attention** (`blocks/cross_attn.rs`, `blocks/exogenous.rs:63-117`). Portfolio
tokens injected at layer 0 via QK-RMSNorm (Q and K) + per-head `ca_q_gain` (init 1.0)
+ 1/sqrt(128) SDPA, zero-init out_proj, learnable `ca_attn_scale`/`mlp_scale`. Note:
this block does **not** use the resid_mix x0 anchor (`trading_model.rs:592`).

**Residual / depth scaling** (`init.rs:87-107`, `trading_model.rs:223-331`). Identity
at init via zero-init residual projections + ones-init scales + (1,0) resid_mix.
**`residual_init_scale = 1/sqrt(2N)` is computed but discarded** by every block — no
depth-aware downscale (`parameter-golf`'s `ln_scale_factor` analog is absent). All
scale params are 1D -> AdamW, `weight_decay=0.0` (`trainer.rs:196`). The learnable
gains `attn_scale`, `mlp_scale`, `q_gain`, `resid_mix` are **unbounded and undecayed**.

## 1.4 Readout / Heads

**PMA readout** (`blocks/pma.rs:13-64`). Set-Transformer pooling: 2 learned seeds
(actor=0, critic=1, Randn std 0.02) query the post-trunk patch embeddings via one MAB
(CrossAttnFfnBlock): QK-RMSNorm + `q_gain` (init 1.0) + 1/sqrt(128) attention. PMA
out_proj is **deliberately non-zero-init** (orthogonal gain 1.0) so seeds receive
patch context at step 0; FFN stays zero-init. Output -> `readout_ln` (affine-free
RMSNorm, **chokepoint #2 on input scale to heads**, `trading_model.rs:606`).

**Policy head** (`head.rs:15-21`, `action_space.rs:5-7`). Per-ticker Beta over (0,1).
`readout_ln` actor token -> plain Linear `policy_concentration` (model_dim->2,
orthogonal gain 0.01, no bias) -> `raw_alpha/raw_beta` ->
`beta_concentration = softplus(raw) + 1`. **Floor at 1 (prevents point mass on the
broad side); NO upper bound, NO tanh softcap, NO temperature.** At init concentrations
~1.69 (near-uniform Beta). log_prob/entropy are exact closed forms in fp32.

**Value head** (`head.rs:23-24`, `value/hl_gauss.rs`). Distributional HL-Gauss critic,
255 bins, symlog support [-3,3] => raw +/-19.1. `value_proj` orthogonal gain 0.1
(not zero-init). Targets: returns -> symlog -> **clamped to support** -> Gaussian-CDF
two-hot; SIGMA_RATIO=0.5 (sharp targets). Soft cross-entropy loss.

## 1.5 Learning / PPO

**Advantages / GAE** (`gae.rs:37-94`, `advantages.rs:12-111`). Standard GAE,
gamma=0.995, lambda=0.95 (hardcoded literals, `advantages.rs:50-51`). Bootstrap value
decoded from HL-Gauss. Then **two-stage normalization**: batch-level rank-Gaussian
transform over the full population (extremes +/-2.33, `advantages.rs:75`) then
per-minibatch mean/std standardize (`update.rs:182-183`). This decouples
policy-gradient scale from reward/value scale (matches `orbit-wars`).

**PPO loss** (`update.rs:25-32,222-308`). DAPO asymmetric hard clip,
eps_low 0.20 / eps_high 0.28. `ratio = exp(new_logprob - old_logprob)`.
`approx_kl = mean(exp(log_ratio) - 1 - log_ratio)` (Schulman k3, convex). **ENTROPY_COEF
= 0.0** (`config.rs:30`; entropy term is logging-only). VALUE_LOSS_COEF=1.0.
OPTIM_EPOCHS=3. KL early-stop at TARGET_KL*1.5 = 0.045, evaluated at **epoch end** on
epoch-mean AND last-minibatch KL — i.e. after all minibatch steps are already applied.

**Rollout / minibatching** (`update.rs:73-477`, `rollout.rs`). nprocs=16,
seq_len=2000, chunk_len resolves to 50, total ~32000 samples, 16 minibatches/epoch,
3 epochs, ~48 optimizer steps/episode. Old log_probs captured from the streaming
cached-patch forward (`cache.rs`); new log_probs recomputed via `windowed_replay_forward`
(`replay.rs`) — verified to be the **same op sequence** (patch_embed -> input_ln ->
backbone), differing only in batching layout.

## 1.6 Optimization

**Hybrid optimizer** (`muon.rs`, `optimizer_glue.rs`, `trainer.rs:187-205`). 2D matrices
-> NorMuon (EMA momentum 0.92->0.95 warmup over 50 steps, Newton-Schulz5
orthogonalization, per-row second-moment rescale with Frobenius-norm preservation).
1D params + force-routed names (`policy_concentration`, `value_proj`, `resid_mix`) ->
AdamW (betas 0.9/0.95, eps 1e-8, **wd=0**). **The policy and value heads are on AdamW,
not Muon.** LRs constant: MUON_LR=5e-3, AdamW 3e-4. **No LR warmup ramp, no warmdown.**
Grad clipping: separate actor/critic groups, each L2-clipped to MAX_GRAD_NORM=0.5;
note NS5 re-normalizes Muon updates by Frobenius norm **after** clipping, so the clip
is largely inert for the Muon trunk.

---

# 2. Root-Cause Analysis (ranked)

The dual symptom requires a quantity whose effect on per-update KL **grows over
training** (the spiky/growing half) layered on a baseline where the **mean update is
small** (the low half). The references defend against exactly this by bounding
magnitudes at three chokepoints — input, attention logits, output logits — plus
identity-at-init residual blocks and (in references) LR annealing / weight decay /
entropy. This model has the input chokepoint (input_ln) and attention chokepoint
(QK-norm) but is materially weaker at the **output/policy-distribution chokepoint** and
at **the regularizers that keep gains/concentrations from inflating over time**.

A critical empirical filter: the verified KL trajectory shows spikes **growing across
PPO epochs within a single late update** (epoch 1 lowest, epoch 2-3 spike). This is the
fingerprint of genuine on-policy divergence accumulating across gradient steps as the
distribution becomes more sensitive — it points at the **policy-distribution
sensitivity + weak/late trust region**, and it rules out static-data and
weight-independent-noise explanations.

## H1. Weak/mis-timed PPO trust region: hard clip dead-zone + epoch-end KL stop + zero entropy (WELL-SUPPORTED)

**Mechanism.** PPO's only trust-region enforcement here is (a) the hard asymmetric
clip and (b) an epoch-end KL early-stop. The hard clip gives **zero gradient pressure
inside the band** [0.80, 1.28]; when KL is low almost all ratios sit in-band, so the
full unclipped `-A*ratio` gradient applies and the rare in-band large-advantage sample
drives the update — a dead-zone-then-burst dynamic. The KL early-stop is checked only
at epoch end, **after** every minibatch optimizer step of that epoch is already applied
(`update.rs:420` step vs `:472-476` check), and triggers on the **mean** (persistently
low) — so it neither prevents nor reacts to intra-epoch bursts. ENTROPY_COEF=0.0
removes the one term that would resist distribution sharpening.

**Why THIS dual symptom.** Mean KL stays low because most ratios are in-band and the
mean estimator is dominated by the calm bulk. Spikes grow because (i) the clip cannot
cap per-update KL, (ii) the stop fires too late and too coarsely, and (iii) as the
distribution sharpens (H2) the same in-band ratio drift corresponds to larger
log-prob/KL. The verified epoch-2/3 > epoch-1 spike growth is exactly the trust region
failing to engage as on-policy divergence accumulates across the 3 epochs of reuse.

**Evidence.** `update.rs:25-32` (hard clip), `:420` vs `:472-476` (step-then-check at
epoch end on mean), `config.rs:25-26,30` (clip band, ENTROPY_COEF=0), long-run KL
growing across epochs (verified).

**Reference contrast.** `orbit-wars` uses a **smooth SPO quadratic** trust region
(graded pressure on every sample, eps 0.4/0.56, `ppo.py:593-601`) and treats KL as
diagnostic-only with no early-stop **because** the trust region is structural;
`parameter-golf` bounds outputs with a tanh softcap and a very tight grad clip.

**Verification verdict.** Not directly adjudicated as a standalone hypothesis, but
strongly corroborated by the verified epoch-ordering of spikes and consistent with the
reference contrast. The reward/advantage verdicts independently confirm that
advantage scaling is NOT the spike source, which throws weight onto the loss/trust-region
and distribution side.

**Confidence: medium-high** for being a primary contributor to the spiky-growing half.

## H2. Unbounded Beta concentration + zero entropy lets the policy distribution sharpen, raising per-update KL sensitivity over training (SUPPORTED as contributor; NOT the sole/early cause)

**Mechanism.** `beta_concentration = softplus(raw) + 1` is floored at 1 but
**unbounded above**, fed by a plain Linear with no softcap/temperature. Per-update KL
of a Beta scales with its Fisher information (trigamma terms), which **grows with
concentration**. So an identical-magnitude step in (alpha,beta) yields ever-larger KL
as the distribution sharpens; near-boundary sampled actions
(`(alpha-1)*log(x)`, `x` clamped to 1e-6) make log-prob hypersensitive precisely where
a committed policy increasingly samples.

**Why THIS dual symptom.** Early/calm: concentrations modest, Fisher small -> tiny
per-update KL (low). Late: concentrations inflate (nothing resists it — ENTROPY_COEF=0,
wd=0), so the same gradient step produces growing KL bursts on the sharpened tickers
(spiky-growing).

**Evidence.** `action_space.rs:5-7`, `head.rs:16-21`, `config.rs:30`,
`update.rs:222-245,369`. Long-run KL mean + max both grow (verified), consistent with
rising sensitivity.

**Reference contrast and the key caveat.** Adversarial verification **refuted the
strong framing** that the `1+softplus` formula is the differentiator: `orbit-wars` uses
the **identical** `1+softplus` Beta head with gain-0.01 init **and** fraction
entropy_coef=0.0, yet is healthy. The true divergence is the **absence of opposing
regularization**: `orbit-wars` runs weight_decay=1e-4 on its matrices and tunes control
LRs to damp actor drift; this repo has **wd=0 everywhere** (`trainer.rs:196-197`) and
ENTROPY_COEF=0. So the mechanism is real, but the correct statement is "unbounded
concentration with **no decay/entropy/softcap brake**," not "the formula lacks a cap."
Multiple verdicts also note the early low-KL exists **before** any sharpening can occur,
so this cannot be the source of the persistent-low baseline — only an amplifier of the
late spikes.

**Verification verdict.** Mixed: one verdict "plausible" (real flaw, plausible spike
contributor, unproven as primary, contradicted as the cause of early low-KL); several
"refuted" specifically against the strong "formula is root cause" framing. Net: a
genuine **contributing amplifier** of the spiky-growing half, gated on concentrations
actually inflating (currently un-instrumented).

**Confidence: medium** as a contributor to growing spikes; **low** as a standalone root
cause; the formula-level framing is refuted.

## H3. Unbounded/undecayed learnable gains (attn_scale, mlp_scale, q_gain, resid_mix) inflate network gain over training (SPECULATIVE; partly refuted in its naive form)

**Mechanism.** Per-channel residual gates and per-head attention temperatures are
learnable, unbounded, weight_decay=0, with no depth-scale factor
(`residual_init_scale` is discarded). As blocks ramp from identity, these gains can
drift upward over training, raising effective network gain and, for `q_gain`,
sharpening attention toward hard selection (routing flips).

**Why THIS dual symptom (claimed).** Low while gains ~1; growing spikes as inflated
gains make small upstream changes flip attention routing / sharpen activations feeding
the readout.

**Evidence.** `gqa.rs:47-49`, `init.rs:105-107` (scale discarded), `trainer.rs:196`
(wd=0). No per-update instrumentation of these norms exists.

**Reference contrast / refutation.** Verification **refuted** the specific causal
story for the residual gates: RMSNorm is scale-invariant, so uniform residual inflation
does **not** amplify a fixed perturbation (it shrinks the relative perturbation), and
`orbit-wars` uses the **identical** unbounded ones-init scales + final RMSNorm and is
healthy. The verdict notes the *anisotropic* `q_gain` growth (sharpening attention
softmax / routing flips) is the only sub-mechanism not foreclosed by RMSNorm
scale-invariance — but this is unproven and would be a smooth drift, not abrupt spikes.

**Verification verdict.** "Refuted" for the residual-gate framing; the `q_gain`-growth
sub-path survives as **speculative**.

**Confidence: low.** Worth instrumenting cheaply (it shares instrumentation with H2),
but the naive residual-scale story is mechanically wrong under RMSNorm.

## H4. Critic value-support saturation on trending episodes biases advantages over training (SPECULATIVE; scale-decoupled but rank-mediated)

**Mechanism.** HL-Gauss symlog support is only [-3,3] => raw +/-19.1, while
REWARD_SCALE=20 with gamma=0.995 lets discounted returns reach ~5..40 on trending
windows; `encode()` **clamps** out-of-support targets (`hl_gauss.rs:96`). Saturated
targets bias V on the most extreme (informative) trajectories; biased V -> noisier GAE
deltas -> advantage sign/rank flips concentrated in trending rollouts.

**Why THIS dual symptom.** rank-Gaussian crushes advantage **scale**, so this cannot
inflate KL via magnitude — but it can produce **rank reshuffles** concentrated in
trending regimes as the policy learns to ride trends, giving sporadic coordinated
policy shifts that grow as the policy exploits trends more.

**Evidence.** `hl_gauss.rs:7-8,96`, `reward.rs:5,236`, `advantages.rs:49`. Whether it
fires depends on the actual clamp rate, which is **not currently logged** (though
`range_stats()` exists, `hl_gauss.rs:60`).

**Reference contrast.** `orbit-wars` sizes symlog support at raw +/-100k (153 bins) so
returns essentially never clamp; ours is far tighter while applying a 20x multiplier —
the one place ours is plausibly worse than the RL reference on the value path.

**Verification verdict.** Not separately adjudicated; component findings rate it
"medium" with explicit dependence on an unmeasured clamp rate.

**Confidence: low-medium**, fully contingent on a non-trivial clamp rate. Cheap to
confirm/eliminate.

## H5. Constant LR with no warmdown removes late-training damping (SPECULATIVE)

**Mechanism.** Both LRs are constant; no warmup ramp, no warmdown. A fixed step size
late in training translates a widening advantage-error / sharper-policy regime into
progressively larger policy shifts.

**Why THIS dual symptom.** A warmdown would shrink the per-update KL ceiling over time;
its absence lets spikes grow as other factors (H2/H4) widen the gradient/sensitivity
distribution.

**Evidence.** `config.rs:10-16`; no `set_lr`/warmdown in the loop. Both references
anneal LR.

**Verification verdict.** Embedded in the optimizer hypothesis, which was **refuted**
for the NS5-fixed-step framing but the constant-LR observation stands as an
unaddressed divergence from both references.

**Confidence: low** as a primary driver; plausible as a multiplier on the growing-spike
trend and a cheap, low-risk mitigation.

## Explicitly EXONERATED (well-supported negatives)

- **Reward construction & scale.** rank-Gaussian + per-minibatch advantage
  normalization decouple reward scale from policy-gradient/KL magnitude entirely
  (`advantages.rs:75`, `update.rs:182-183`). REWARD_SCALE cannot directly cause low or
  spiky KL. **High confidence.**
- **Reward normalizer.** Does not exist (the `reward_norm.rs` in git status is
  stale/untracked and unwired). No running-var bug is possible. **High confidence.**
- **Advantage/GAE path.** Near-exact port of `orbit-wars`; actively **suppresses**
  scale-driven and outlier-driven KL. **High confidence** it is not the source.
- **Input price-delta stream (raw/unstandardized).** **Refuted** as the spiky-growing
  cause: `input_ln` (parameterless RMSNorm) pins the layer-0 token and is invariant to
  uniform weight scaling, breaking the "drifting input gain" loop; fat-tail direction
  flips are a **static** data property (same ratio at episode 1 and 1000), not a
  training-time divergence dynamic. May weakly contribute to low KL via a weak signal;
  worth a cheap winsorize for learnability, but not the symptom driver. **High confidence.**
- **RoPE.** Parameter-free, gradient-free, norm-preserving, applied after QK-norm;
  active UniformStream uses uniform spans so positions are correct. Cannot drift.
  **High confidence.**
- **PMA readout, exo cross-attention, GQA block structure, patch enrichment.** All
  match the references on the load-bearing chokepoints (QK-RMSNorm on Q **and** K,
  q_gain temperature, affine-free pre-head/entry RMSNorm, zero-init residual
  projections, Beta +1 floor). Not primary suspects. **Medium-high confidence.**
- **bf16 dual-path log_prob divergence.** **Refuted**: old and new log_probs use the
  **same op sequence** (differ only in batching), epoch-1 KL is routinely exactly
  0.0000 (proving sub-1e-4 path agreement), and the verified spikes concentrate in
  **epoch 2-3, not epoch 1** — the opposite of a weight-independent noise floor.
  **High confidence** it is not the cause.
- **Optimizer NS5-fixed-step at the readout.** **Refuted**: the policy/value heads are
  force-routed to **AdamW** (`trainer.rs:198-202`), which self-attenuates with shrinking
  gradients; `orbit-wars` uses the same Muon-trunk / AdamW-head split and is healthy.
  **High confidence** for the readout-specific claim.

---

# 3. Discriminating Experiments (cheap-first)

The cheapest, most decisive step is **instrumentation**, because the two surviving
amplifier hypotheses (H2, H4) and the speculative H3 are all currently un-measured, and
the trust-region hypothesis (H1) is confirmable from KL structure already in the logs.

**E1 (instrumentation, ~1 cheap change, decisive for H2/H3).** Log per PPO update:
(a) mean/p99.9 of `alpha+beta` (Beta concentration) and policy entropy; (b)
`||policy_concentration.weight||`, `q_gain` per block, `||attn_scale||`/`||mlp_scale||`;
(c) the **max and p99** per-minibatch `approx_kl` (not just mean/last-mb). Predictions:
H2 -> concentration rises monotonically and KL spikes coincide with high
concentration; H3 -> gains rise and spikes coincide; if both stay flat while KL still
grows, both are exonerated and attention turns fully to H1/H4.

**E2 (instrumentation, decisive for H4).** Log `hl_gauss.range_stats()` below/above
clamp fractions per rollout (function already exists, `hl_gauss.rs:60`). If clamp rate
is ~0, H4 is eliminated; if it grows with training and correlates with KL bursts, H4
is confirmed.

**E3 (frozen-weights control, decisive negative for residual/precision).** Run one PPO
update with LR=0 across all 3 epochs; log per-minibatch `approx_kl`. Expected ~0,
confirming the bf16/path and "fixed-step" stories are dead and that spikes require real
weight movement.

**E4 (trust-region A/B, tests H1).** Two short matched runs: baseline vs (i) check KL
**per-minibatch before** stepping (skip/scale steps over a cap) and gate early-stop on
mean only (drop the last-minibatch trigger). If intra-epoch spikes flatten, H1
confirmed as a primary contributor.

**E5 (concentration-cap / entropy A/B, isolates H2 from baseline).** Add a soft cap
(`alpha,beta = 1 + softcap*tanh(softplus(raw)/softcap)`, softcap ~30-50) OR
ENTROPY_COEF=1e-3..1e-2, run long enough to reach the spiky regime. If max/p99 KL
flattens while mean KL/learning is retained, H2 confirmed as the spike amplifier; if
spikes persist, H2 demoted.

**E6 (support-widen A/B, tests H4).** If E2 shows non-trivial clamp rate, widen symlog
support to +/-5..6 (or lower REWARD_SCALE) and recheck clamp rate + KL spikes.

---

# 4. Recommended Fixes (per top hypothesis)

**For H1 (trust region) — highest leverage, do first.**
1. Move the KL safety **per-minibatch and pre-step**: check `approx_kl` before
   `opt.step()` and skip/scale steps that exceed a cap, instead of an epoch-end mean
   check (`update.rs:420` vs `:472-476`).
2. Replace the hard asymmetric clip with a **smooth SPO-style quadratic penalty**
   (`|A|*(ratio-1)^2/(2*eps)`) so every sample receives graded pressure and the in-band
   dead-zone disappears (`orbit-wars` `ppo.py:593-601`). If keeping the hard clip,
   reduce `CLIP_EPS_HIGH` toward `CLIP_EPS_LOW` to cut the upward-sharpening asymmetry.
3. Log per-update **mean** KL and per-minibatch **max/p99**, not just the last
   minibatch (`log.rs:198`) — the current single-minibatch metric manufactures apparent
   spikiness and hides true drift.

**For H2 (concentration sharpening).**
1. Add a **soft upper bound** on concentration:
   `alpha,beta = 1 + softcap*tanh(softplus(raw)/softcap)` (softcap ~30-50), bounding
   the Beta's Fisher metric and hence achievable per-update KL while leaving the
   near-uniform init untouched.
2. Add **weight decay (~1e-4)** to the matrix params (and/or the
   `policy_concentration` head), matching `orbit-wars`; the current wd=0 everywhere
   removes the reference's primary brake on actor drift.
3. Consider a small **ENTROPY_COEF (1e-3..1e-2)** as a counter-pressure to sharpening,
   since the Beta is the sole policy here (unlike `orbit-wars`' categorical that carries
   the entropy floor).

**For H3 (gain drift) — only if E1 implicates it.**
1. Add weight decay or a soft cap specifically to `q_gain` (and optionally
   `attn_scale`/`mlp_scale`); `final_ln` already absorbs uniform residual inflation, so
   prioritize `q_gain` (the attention-sharpening / routing-flip path).
2. Either **use** `residual_init_scale` / a per-layer `1/sqrt(layer_idx+1)` depth
   factor, or delete the dead parameter to remove the illusion of depth scaling.

**For H4 (critic saturation) — only if E2 shows clamp.**
1. Widen HL-Gauss symlog support (e.g. +/-5..6) and/or lower REWARD_SCALE so discounted
   returns sit comfortably inside support; raise SIGMA_RATIO toward ~0.75-1.0 for
   proper soft-label smoothing.
2. Zero-init `value_proj` (matching `orbit-wars` and the codebase's own
   `linear_residual_out` idiom) for a uniform/V=0 cold start.

**For H5 (LR schedule) — cheap, low-risk.**
1. Add a linear **warmdown** of both LRs over the back portion of the run, as both
   references do; this directly shrinks the per-update KL ceiling as training proceeds.

**Hygiene (not KL-causal, but reduces mis-attribution).**
- Clamp the two unbounded global features (`pnl`, `commissions`) and rescale `macd` to
  match the rest of the static vector.
- Winsorize/soft-squash raw log returns (e.g. `clamp(+/-0.25)` or `asinh`) for
  learnability robustness.
- Delete dead reward variants; remove the stale `reward_norm.rs`.
- Promote `gamma`/`lambda` literals into `config.rs`.

---

## Summary judgment

No single confirmed root cause. The symptom is best explained as a **trust region that
is structurally too weak and temporally mis-timed (H1)** acting on a **policy
distribution whose per-update KL sensitivity is free to grow over training (H2),
unchecked by entropy or weight decay**, with **critic-support saturation (H4)** and
**constant LR (H5)** as plausible secondary amplifiers contingent on measurement. The
input, RoPE, advantage, optimizer-readout, and precision pathways are well-supported
negatives. The decisive next move is instrumentation (E1/E2) plus the frozen-weights
control (E3), which jointly partition the surviving hypotheses before any architectural
change.
