# Bidirectional Trunk + PMA Readout — Implementation Spec

Status: authoritative, accepted. SPEC ONLY (no code in this document).

This supersedes `causal_pooling_readout_spec.md`. Decision: the patch trunk becomes
**fully bidirectional** (no fork tokens, no causal mask), and the actor/critic readout
becomes a **PMA** (Set Transformer Pooling by Multihead Attention) module with two learned
seed queries. Portfolio/account state stays in the trunk. Head output contracts are
unchanged.

All citations are to the tree at the time of writing. Symbols: `S` = patch sequence length
(Base variant `S = 153`, from `BASE_PATCH_CONFIGS` in `config.rs:24-33` via
`compute_patch_totals` `config.rs:105`), `d` = `model_dim` (Base `256`,
`config.rs:5`), `B` = batch rows (`batch * TICKERS_COUNT`; `TICKERS_COUNT = 1`,
`constants.rs:3`), `H` = `GQA_NUM_Q_HEADS = 4`, `H_kv` = `GQA_NUM_KV_HEADS = 1`,
`head_dim = d / H = 64` (`gqa.rs:10-11,39`). Depth = `BASE_GQA_LAYERS = 3`
(`config.rs:7`).

---

## 1. Current state (cited)

### 1.1 Trunk: causal + fork (`trading_model.rs:608-657`)

`causal_patch_trunk` (`trading_model.rs:608`) runs over `S+1` positions:

- `e_last = patch_hidden[:, S-1]` (`:616`), `forks = e_last + role_embeds` (`:617`,
  `role_embeds` shape `[2, d]`, `trading_model.rs:276-283`).
- `past = patch_hidden[:, 0..S-1]` (`:618`); `x0 = cat([past, forks], dim=1)` →
  `[B, S+1, d]` (`:619`). Sequence length is `S+1` (`:620`).
- `role_bias = cat([zeros[1, S-1, d], role[1, 2, d]], dim=1)` → `[1, S+1, d]`
  (`:621-625`); re-injected at every block via `forward_with_rope_positions`'s
  `role_bias` arg (`gqa.rs:85,95`).
- `rope_positions = arange(S+1)` (`:627`).
- `attn_mask = fork_attention_mask(S+1)` (`:628`, def `:647-657`): lower-triangular
  causal mask PLUS a rule that fork query `S` may not attend to key `S-1` (the other
  fork). The two fork positions (`S-1` actor, `S` critic) are the readout slots.
- Loop over `gqa_layers` (`:629-643`) calls
  `layer.forward_with_rope_positions(x, x0, role_bias, rope, rope_positions, Some(mask), true)`
  (`:630-638`); after layer 0, exogenous (portfolio) cross-attention is applied
  (`exogenous_ticker_block.forward(x, exo_tokens)`, `:639-641`); then
  `maybe_apply_endogenous_ticker` (`:642`, def `:403`). Final `final_ln` (`:644`).

`backbone_with_actor_critic_cls` (`:659-670`): reads `actor_read = trunk[:, S+1-2]`,
`critic_read = trunk[:, S+1-1]` (`:667-668`) and calls `head_from_actor_critic_cls`
(`:669`).

### 1.2 GQA block (`gqa.rs`)

`forward_with_rope_positions` (`gqa.rs:81-113`): pre-residual mixes
`x*resid_mix[0] + x0*resid_mix[1] + role_bias` (`:92-95`), projects QKV with RoPE at
explicit `positions` (`:97`, `project_qkv_with_positions` `:147-177`), then
`scaled_dot_product_attention(q, k, v, attn_mask, 0.0, causal && attn_mask.is_none(), …)`
(`:99-108`). With a mask supplied, the SDPA `is_causal` flag is `false`
(`causal && attn_mask.is_none()` → `false`), so causality comes entirely from the mask
content. Output reshaped `[B,S+1,d]` (`:109`), `attn_out` proj + `attn_scale`
(`:110-111`), then FFN (`apply_ffn` `:179-187`: RMSNorm → squared-ReLU FF → `mlp_scale`
→ residual).

### 1.3 Heads (`head.rs`) — FIXED CONTRACTS

`head_from_actor_critic_cls` (`head.rs:9-31`):

- Actor: `actor.view([batch, TICKERS_COUNT, d])` (`:15`) → `policy_concentration`
  (`Linear d→2`, `trading_model.rs:356-357`) → split to `raw_alpha`/`raw_beta` →
  `beta_concentration` (`head.rs:18-21`). **2-output Beta. FIXED.**
- Critic: `critic.view([batch, TICKERS_COUNT*d])` (`:23`) → `value_proj`
  (`Linear (TICKERS_COUNT*d)→NUM_BINS`, `trading_model.rs:358-359`). **21-bin
  HL-Gaussian (`NUM_BINS`). FIXED.**

Both heads consume a single `[B, d]` vector each (`B = batch` since `TICKERS_COUNT = 1`).

### 1.4 Call sites of the readout (all must keep working)

`backbone_with_actor_critic_cls` is called from `forward.rs:21`, `forward.rs:70`,
`stream/cache.rs:64`, `stream/replay.rs:155`. All pass `(patch_hidden/x_stem, exo_tokens,
batch_size)` and receive `ModelOutput`. The PMA refactor is internal to the backbone;
these signatures stay identical.

---

## 2. Target architecture

### 2.1 Trunk — bidirectional, seq length `S` (no fork)

Input: `patch_hidden : [B, S, d]` (unchanged producer, `forward.rs:18` /
`patch_latent_stem_on_device`). The trunk now operates on exactly `S` positions.

Per layer `i` in `gqa_layers` (depth 3):

1. `x = layer.forward(... bidirectional ...)` over `[B, S, d]`.
   - **No causal mask.** SDPA is called with `attn_mask = None` and `causal = false`
     (i.e. `Tensor::scaled_dot_product_attention(q, k, v, None, 0.0, false, None, true)`),
     giving full all-to-all attention. (Equivalent alternative: pass an all-zeros
     `[S, S]` additive mask; prefer `None`/`causal=false` to avoid materializing an
     `S²` tensor — see §6.)
   - **`role_bias` is removed from the trunk path.** It only encoded fork identity at
     positions `S-1`/`S`, which no longer exist. The pre-residual mix becomes
     `x*resid_mix[0] + x0*resid_mix[1]` (drop the `+ role_bias` term, `gqa.rs:95`).
   - **RoPE positions = `arange(S)`** (`Int64`, on device).
   - `x0 = patch_hidden` (the post-input_ln stem; no fork concat). The x0 residual
     mixing (`gqa.rs:92-94`) is retained verbatim with this new `x0`.
2. After layer 0 only: `x = exogenous_ticker_block.forward(x, exo_tokens)` —
   **portfolio cross-attention, UNCHANGED** (`trading_model.rs:639-641`).
3. `x = maybe_apply_endogenous_ticker(x, i)` — UNCHANGED in mechanism, but its
   fork-aware `live_idx` branch (`trading_model.rs:418-422`) is now dead because
   `seq == S == self.seq_len` always; it falls through to `live_idx = seq - 1`
   (the last patch). For `TICKERS_COUNT = 1` it early-returns (`:408`) and is a no-op;
   keep it but remove the now-unreachable `seq > self.seq_len` fork branch (§3).

Output: `final_ln(x)` → `encoded : [B, S, d]` — the bidirectionally-encoded patch
embeddings, each already carrying portfolio context (via step 2) and full-sequence
context. This is the PMA key/value source.

### 2.2 PMA module — bidirectional Set-Transformer pooling

The PMA replaces the fork mechanism entirely. It has two learned seed vectors (actor,
critic) that attend over all `S` encoded patches.

**Parameters (learned):**

- `seeds : [2, d]` — `var("pma_seeds", [2, d], Init::Randn{mean:0, stdev:0.02})`
  (mirrors `role_embeds` init, `trading_model.rs:276-283`). Index 0 = actor, 1 = critic.
- A multihead cross-attention sublayer (MAB attention half) reusing the trunk's head
  convention `H = 4`, `H_kv = 1`, `head_dim = 64`. Reuse the existing
  `ExogenousTickerBlock` / `CrossAttnFfnBlock` machinery as the implementation vehicle
  where it fits, OR a dedicated GQA-style cross-attention block. The recommended,
  most-idiomatic choice: a **`CrossAttnFfnBlock`** (`blocks/cross_attn.rs:11-74`) —
  it already packages `project_source` (K/V from a source seq), multihead SDPA with
  q/k RMSNorm + `q_gain` + `attn_scale`, a residual output proj, and a squared-ReLU FFN
  with `mlp_scale` and residual. That FFN+residual IS the PMA `rFF` half (§2.3),
  matching codebase idioms exactly. Note: `CrossAttnFfnBlock` uses `CA_NUM_HEADS = 2`,
  `CA_HEAD_DIM = 128` (`cross_attn.rs:8-9`); either (a) accept those CA head dims for
  the pooling block, or (b) add a GQA-headed variant if matching `H=4, head_dim=64` is
  desired. **Decision: use `CrossAttnFfnBlock` as-is** (CA head config) — it is the
  established cross-attention block in this codebase and avoids new attention code;
  head-dim parity with the trunk is not load-bearing for pooling.

**Forward — exact shapes (per call, `B` rows):**

1. Expand seeds to batch: `q = seeds.unsqueeze(0).expand([B, 2, d])` → `[B, 2, d]`.
   (Seeds are shared across rows; no per-row content.)
2. K/V source = trunk `encoded : [B, S, d]`.
3. MAB attention: cross-attention with `q` as queries `[B, 2, d]`, `encoded` as
   keys/values `[B, S, d]`.
   - `project_source(encoded)` → `k, v : [B, CA_NUM_HEADS, S, CA_HEAD_DIM]`
     (`cross_attn.rs:54-56` → `exogenous.rs:63-75`).
   - `forward_with_projected_source(q, k, v)`:
     `q → [B, CA_NUM_HEADS, 2, CA_HEAD_DIM]` (`exogenous.rs:84-90`),
     `SDPA(q, k, v, None, 0.0, false, …)` → `[B, CA_NUM_HEADS, 2, CA_HEAD_DIM]`
     (`exogenous.rs:101-110`), reshape `[B, 2, CA_NUM_HEADS*CA_HEAD_DIM]`, out-proj
     `→ [B, 2, d]`, `attn_scale`, residual `q + out` (`exogenous.rs:111-116`).
   - **No causal mask** (`attn_mask = None`, `is_causal = false`) → bidirectional
     pooling over all `S` patches. This is the existing call form in
     `exogenous.rs:101-110`, so it is satisfied for free.
4. rFF half (already inside `CrossAttnFfnBlock::forward_with_projected_source`,
   `cross_attn.rs:64-72`): `x = x + mlp_scale ⊙ FF(RMSNorm(x))` (squared-ReLU FF).
   Output `pooled : [B, 2, d]`.

`pooled[:, 0]` is the actor readout `[B, d]`; `pooled[:, 1]` is the critic readout
`[B, d]`.

### 2.3 PMA convention compliance

Set-Transformer PMA = `rFF(MAB(seeds, X))`, where `MAB(Q,X) = LN(H + rFF(H))`,
`H = LN(Q + Multihead(Q, X, X))`. This codebase's residual/norm idiom is pre-norm
RMSNorm with learned residual-output scaling (`attn_scale`/`mlp_scale`) rather than
post-norm LN, and `CrossAttnFfnBlock` already implements exactly this two-sublayer
(attn-residual then FF-residual) structure (`cross_attn.rs:64-72`, `exogenous.rs:82-116`).
So one `CrossAttnFfnBlock.forward(seeds, encoded)` == one PMA block in this codebase's
idiom. **Use exactly one PMA block** (depth-1 pooling), consistent with the original
single-fork readout depth.

### 2.4 Positional encoding in pooling — design decision

**The PMA applies NO RoPE on seeds or keys.** Rationale: the patch embeddings already
carry absolute/relative position through the trunk's RoPE self-attention (§2.1), so the
pooled values are position-aware; adding RoPE to a permutation-pooling query would
impose a spurious ordering on the two seeds and is unnecessary. `CrossAttnFfnBlock`
already does no RoPE (`exogenous.rs` has no rope application), so this is automatic.

### 2.5 Heads — exact reshapes from `[B, 2, d]`

`B = batch` (since `TICKERS_COUNT = 1`). From `pooled : [batch, 2, d]`:

- `actor_read = pooled.select(1, 0)` → `[batch, d]` →
  `head_from_actor_critic_cls` reshapes to `[batch, TICKERS_COUNT, d]`
  (`head.rs:15`) → `policy_concentration` → Beta `(alpha, beta)`.
- `critic_read = pooled.select(1, 1)` → `[batch, d]` →
  reshape `[batch, TICKERS_COUNT * d]` (`head.rs:23`) → `value_proj` → 21-bin logits.

`head_from_actor_critic_cls` is **unchanged**; it already accepts two `[batch, d]`
tensors. `backbone_with_actor_critic_cls` now does: `encoded = trunk(...)`;
`pooled = pma(seeds, encoded)`; `head_from_actor_critic_cls(pooled.select(1,0),
pooled.select(1,1), batch)`.

---

## 3. Code to REMOVE

| Item | Location | Action |
|---|---|---|
| Fork construction `e_last`, `forks`, `past`, fork `x0` concat | `trading_model.rs:616-619` | Remove; trunk runs on `patch_hidden` directly, `x0 = patch_hidden`. |
| `seq_len = num_patches + 1` | `trading_model.rs:620` | Remove `+1`; seq length is `S`. |
| `role_bias` construction (`zeros_patches`, `cat`) | `trading_model.rs:621-625` | Remove. |
| `rope_positions = arange(S+1)` | `trading_model.rs:627` | Change to `arange(S)`. |
| `attn_mask = fork_attention_mask(...)` + the `Some(&attn_mask)` arg | `trading_model.rs:628,636` | Remove; pass `None`. |
| `fork_attention_mask` fn | `trading_model.rs:647-657` | Delete entirely. |
| `role_bias` term in pre-residual mix | `gqa.rs:95` | Remove `+ role_bias.to_kind(...)`. |
| `role_bias` parameter of `forward_with_rope_positions` | `gqa.rs:85` | Remove from signature; update the sole call site (`trading_model.rs:632`). |
| `causal` arg / its `true` at the trunk call | `trading_model.rs:637`, `gqa.rs:89,105` | Trunk passes `false`; with `attn_mask=None` SDPA is non-causal. (Keep `causal` param if other callers need it — `forward_prefix_and_cache` uses its own `true`, `gqa.rs:204`; the prefix/streaming cache path must also be debridged, see §3.1.) |
| `role_embeds` field + init | `trading_model.rs:98,276-283,368` | Remove (replaced by `pma_seeds`). |
| `rope` capacity `seq_len + 1` | `trading_model.rs:331` | Change to `seq_len` (no fork position). |
| `maybe_apply_endogenous_ticker` fork branch (`seq > self.seq_len → seq_len-2`) | `trading_model.rs:418-422` + the `live_idx + 1 == seq` fork-aware comment `:415-417,434` | Simplify: `live_idx = seq - 1` always; drop the fork comments. (No-op for `TICKERS_COUNT=1`.) |
| `actor_read/critic_read` index reads from trunk | `trading_model.rs:667-668` | Replace with PMA pooling + `select`. |
| Doc comment describing forks | `trading_model.rs:603-607,329-330` | Rewrite for bidirectional/PMA. |

### 3.1 Streaming/prefix-cache path

`forward_prefix_and_cache` (`gqa.rs:189-213`) and the streaming caches
(`stream/cache.rs`, `stream/replay.rs`, `StreamState` fields `trading_model.rs:72-87`)
assume a **causal** prefix so K/V can be cached and only the tail recomputed. **A
bidirectional trunk invalidates prefix caching** (every patch attends to every other,
including future ones, so no causal prefix is stable). Required handling:

- For the bidirectional trunk, the streamed step must recompute full self-attention over
  the current `S` window (no `forward_prefix_and_cache` reuse for the GQA layers).
  `stream/cache.rs:64` and `stream/replay.rs:155` call `backbone_with_actor_critic_cls`,
  which will route through the new bidirectional trunk + PMA automatically; the K/V
  prefix-cache fast path (`uniform_prefix_k/v`, `forward_prefix_and_cache`) becomes
  unused for the trunk and should be removed or bypassed to avoid stale/incorrect
  causal caches.
- This is a correctness requirement, not optional. Spec the implementer to verify no
  caller still feeds cached causal prefixes into the bidirectional layers.

---

## 4. Code to ADD (describe; do not implement)

### 4.1 `PmaReadout` struct (new, e.g. `model/blocks/pma.rs`)

Fields:
- `seeds: Tensor` — `[2, d]`, learned (`var("pma_seeds", [2,d], Init::Randn{0,0.02})`).
- `block: CrossAttnFfnBlock` — constructed via
  `CrossAttnFfnBlock::new(&(p/"pma"), model_dim, ff_dim, init_scale)`
  (`cross_attn.rs:20-28`). This supplies multihead cross-attention (q/k RMSNorm,
  `q_gain`, `attn_scale`, residual out-proj) plus the rFF (squared-ReLU FF, `mlp_scale`,
  residual).

Constructor `PmaReadout::new(p, model_dim, ff_dim, init_scale)`.

`forward(&self, encoded: &Tensor) -> Tensor` (returns `[B, 2, d]`):
1. `b = encoded.size()[0]`.
2. `q = self.seeds.unsqueeze(0).expand([b, 2, d]).to_kind(encoded.kind())` → `[B,2,d]`.
3. `self.block.forward(&q, encoded)` → `[B, 2, d]`. (Internally: `project_source`
   → K/V `[B, CA_NUM_HEADS, S, CA_HEAD_DIM]`; SDPA bidirectional; out `[B,2,d]`; FF
   residual.)

### 4.2 Wire into `TradingModel`

- Add field `pma: PmaReadout` (`trading_model.rs` struct, replacing `role_embeds`).
- Construct it in `TradingModel::new` (near `:356`), replacing the `role_embeds` var.
- Rename/rewrite `causal_patch_trunk` → `patch_trunk` (bidirectional) returning
  `[B, S, d]` per §2.1.
- `backbone_with_actor_critic_cls`:
  `let encoded = self.patch_trunk(patch_hidden, exo_tokens);`
  `let pooled = self.pma.forward(&encoded);`
  `self.head_from_actor_critic_cls(&pooled.select(1,0), &pooled.select(1,1), batch_size)`.

No change to `head.rs`, head linears, `forward.rs` signatures, or the four call sites.

---

## 5. Unchanged

- Portfolio/account injection: layer-0 `exogenous_ticker_block` cross-attention stays in
  the trunk, same position (`trading_model.rs:639-641`). No FiLM, no relocation.
- RoPE inside the trunk self-attention (now `arange(S)`); RoPE module itself unchanged
  except capacity `seq_len` instead of `seq_len+1`.
- GQA block internals: `project_qkv_with_positions`, q/k RMSNorm, `q_gain`,
  `attn_scale`, squared-ReLU FFN, `resid_mix` x0 mixing — all unchanged except the
  removed `role_bias` term.
- Head contracts: Beta 2-output actor, 21-bin HL-Gaussian critic, exact reshapes
  (`head.rs`), `policy_concentration`/`value_proj` linears.
- Patch embedding / stem / exo-token construction (`forward.rs`, `patch_embed_*`).
- `TICKERS_COUNT = 1`, `endogenous_ticker_block` (no-op for single ticker).

### 5.1 Future refinement (NOT in scope)

FiLM conditioning of the PMA seeds on portfolio state (per-seed scale/shift from
`exo_tokens`) is a possible future enhancement. The portfolio signal already reaches the
seeds through the pooled patch values, so FiLM is omitted now to keep the change minimal.

---

## 6. VRAM / compute note

Expected **~neutral**.

- Trunk self-attention cost is unchanged in order: `O(S² · H · head_dim)` either way
  (forks added only one position, `S+1 ≈ S`). The `S×S` causal+fork additive mask
  tensor (`fork_attention_mask`, `trading_model.rs:655`, `[S,S]` float ≈ `153²·4B ≈
  94 KB` per call) is **removed** by passing `attn_mask=None`/`causal=false`; flash/
  fused SDPA kernels run mask-free. Net: slightly less memory and a marginally faster
  attention kernel.
- PMA adds one cross-attention with only **2 queries** over `S` keys:
  `O(2 · S · CA_NUM_HEADS · CA_HEAD_DIM)` ≈ negligible vs the trunk, plus one small
  `[B,2,d]` FFN. Parameters added: `seeds [2,d]` + one `CrossAttnFfnBlock` (q/k/v/out
  proj `d↔ca_dim`, FF `d↔ff_dim`) — comparable to the removed `role_embeds [2,d]` plus
  the now-unused fork-residual work. Roughly parameter-neutral.
- Streaming: removing the causal prefix-cache fast path (§3.1) means each streamed step
  recomputes full trunk attention over `S`. With `S = 153` and depth 3 this is cheap;
  no gradient-accumulation/chunking (per project rule). Confirm step latency in the
  perf harness (`benchmarks`, `model_*_stream_step_b*`, `benchmarks/src/main.rs:156-179`).

---

## 7. Risks / ablation

- **Loss of frontier emphasis.** The fork seeded the readout directly from the last
  patch `e_{S-1}` (`trading_model.rs:616-617`), hard-wiring recency. Bidirectional PMA
  must *learn* to weight recent patches via the seed→patch attention. If trading
  decisions are dominated by the latest bar, PMA could under-weight it early in
  training. Mitigations to keep in mind (do not pre-implement): the trunk's RoPE
  preserves recency signal in keys; if needed later, add a learned recency prior or a
  small positional bias to PMA keys. **Measure:** actor/critic attention weight on the
  final patches (extend the existing temporal-attention debug, `DebugMetrics`
  `trading_model.rs:27-34`, e.g. `temporal_attn_last_weight`) — confirm PMA learns
  non-trivial weight on the frontier within early episodes.
- **Depth = 3 → modest expected gain.** Bidirectionality helps most with deeper context
  mixing; at 3 layers the upside is moderate. The main wins are (a) cleaner readout
  (no fork mask hack), (b) full-sequence context for both readouts, (c) decoupled
  actor/critic via independent seeds.
- **Possible regressions:** (1) streaming correctness if any causal prefix cache leaks
  into the bidirectional path (§3.1) — verify outputs match the full-forward path
  bit-for-bit on a fixed input; (2) value-head calibration if pooled critic
  representation differs in scale — watch HL-Gaussian value loss and explained variance;
  (3) entropy/exploration shifts from the new actor representation — watch policy
  entropy and Beta concentration ranges.
- **What to measure (learning quality):** episode reward/return curve, value loss,
  policy entropy, benchmark out/under-performance vs index (the headline metric the
  project tracks, README.md:56-65), via `training/training.log` and `report_cli`
  (`meta_chart_bases`).

---

## 8. A/B validation protocol (single A/B, NO grid)

Two arms, identical seed, identical config, identical data/episode schedule:

- **Arm A (baseline):** current causal trunk + fork readout (`master`/pre-change tree).
- **Arm B (new):** bidirectional trunk + PMA readout (this spec).

Setup:
- Run each arm in its own git worktree under `../trading_bot_0_worktrees/` (per
  CLAUDE.md) so binaries/checkpoints don't collide.
- Pin RNG seed and `TradingModelConfig::default()` (`ModelVariant::Base`) for both.
- Two distinct comparisons:
  1. **Perf/VRAM (mechanical, fast):** `cargo run -p benchmarks --release`
     (`benchmarks/src/main.rs:59`) — compare `model_base_forward_infer_b*`,
     `model_base_fwd_bwd_train_b*`, and `model_base_stream_step_b*` timings + reported
     memory between arms. Expectation per §6: ~neutral (B no worse than ~A on
     forward/train; stream step may be slightly slower due to §3.1). Results land in
     `../benchmark_results/latest.json` (`benchmarks/README.md`).
  2. **Learning quality (the deciding metric):** train both arms from scratch for a
     **matched episode budget** and compare. Recommended budget: **50 episodes**
     (report cadence is every 5 episodes per CLAUDE.md; 50 gives ≥10 reported points
     and reaches the regime shown in README results at ep ~51). Primary metric:
     **benchmark-relative return** (combined assets vs index, README.md:56-65) at the
     latest common reported episode, plus the trajectory of episode return. Secondary:
     value loss, policy entropy, explained variance — read from `training/training.log`
     (tail, don't ingest whole) and `report_cli` (`cargo run -p report_cli -- <episode>
     <report>`, rounding the episode down to the nearest 5).

Decision rule: adopt Arm B if benchmark-relative return at the matched episode is ≥ Arm A
within noise AND perf is ~neutral, with no value-loss/entropy pathology. Report the
delta (B − A) on the primary metric and the perf-harness deltas.

---

Written to: `trading_bots/docs/bidirectional_pma_readout_spec.md`.
