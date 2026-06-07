> **SUPERSEDED** by `bidirectional_pma_readout_spec.md` — the causal-reuse + portfolio-relocation variant was rejected; the model is going bidirectional trunk + bidirectional PMA readout with portfolio state retained in the trunk.

# Causal Attention-Pooling Readout Spec

Status: proposal (no implementation). Replaces the last-embedding "fork injection"
readout with causal multihead attention-pooling (PMA-style), moves portfolio/account
state out of the deep trunk into the readout query, and adds a dense readout mode that
emits actor/critic at every valid decision frontier from a single trunk encoding.

All citations are to the tree at the time of writing.

---

## 1. Current state

### 1.1 Readout / fork construction

`trading_bots/src/torch/model/trading_model.rs`

- `causal_patch_trunk` (lines 608-643): builds the decision tokens.
  - `e_last = patch_hidden.select(1, num_patches - 1)` (line 616): takes ONLY the last
    patch embedding.
  - `forks = &e_last + role.unsqueeze(0)` (line 617): two fork tokens = last embedding
    + the 2 learned role embeddings (`role_embeds`, shape `[2, model_dim]`, lines
    276-283).
  - `x0 = cat([past (num_patches-1), forks (2)], 1)` (lines 618-619): the live patch
    `S-1` is REPLACED by the two forks; sequence length becomes `S+1` where
    `S = num_patches` (line 620, `seq_len = num_patches + 1`).
  - `role_bias` (lines 621-625): zeros for the `num_patches-1` past patches, role embeds
    re-injected at the two fork positions; added inside every GQA block
    (`gqa.rs:94`, `role_bias.to_kind(x.kind())`).
  - One full masked-causal GQA pass over all `S+1` positions (lines 628-641), with
    decision-fork cross-attention blocked, layer-0 exogenous cross-attention (line 638),
    and `INTER_TICKER_AFTER` endogenous mixing (line 640).
- `backbone_with_actor_critic_cls` (lines 645-656): reads `actor_read = trunk[len-2]`,
  `critic_read = trunk[len-1]` (lines 653-654), i.e. positions `S-1` and `S`, then calls
  `head_from_actor_critic_cls`.
- RoPE table is sized `seq_len + 1` to hold the extra fork position
  (`trading_model.rs:331`, comment lines 329-330).
- `maybe_apply_endogenous_ticker` (lines 403-442) hardcodes fork-layout assumptions:
  `live_idx = self.seq_len - 2` when `seq > self.seq_len` (lines 418-422), explicitly to
  avoid mixing fork tokens across tickers. This logic is coupled to the fork readout and
  must change.

Consequence: each decision = one full `S+1`-length trunk pass. A rollout of `T`
decisions over an episode pays `T` independent trunk encodings even though the exogenous
price history they encode is almost entirely shared.

### 1.2 Portfolio / account state injection (CRITICAL)

Portfolio/account state currently enters the DEEP TRUNK, not the readout.

- Observation assembly: `trading_bots/src/torch/env/obs.rs`,
  `build_static_obs_array` (lines 11-94). The static vector interleaves:
  - Global account/portfolio (lines 19-28): `step_progress`, `cash_percent`
    (`account.cash/total_assets`), `pnl` (`total_assets/STARTING_CASH - 1`), `drawdown`
    (`total_assets/peak_assets - 1`), `commissions`, `last_fill_ratio`.
  - Global macro (exogenous) (lines 30-45): 14 macro indicators.
  - Per-ticker, first 6 of `PER_TICKER_STATIC_OBS` are portfolio/account (lines 53-66):
    `position_pct`, unrealized `appreciation`, `trade_activity_ema`, `steps_since_trade`,
    `position_age`, `realized_weights` (target weight). The remaining 19 per-ticker
    features (lines 67-90) are exogenous (momentum, oscillators, earnings).
- Constant layout: `shared/src/lib.rs` lines 19-34.
  `GLOBAL_STATIC_OBS = 6 + 14 = 20` (line 23), of which the first 6 are
  portfolio/account. `PER_TICKER_STATIC_OBS = 19 + 6 = 25` (line 32), of which the first
  6 are portfolio/account (comment lines 24-25). `STATIC_OBSERVATIONS = GLOBAL +
  TICKERS_COUNT * PER_TICKER` (lines 33-34). `TICKERS_COUNT = 1` (line 13).
- Injection into the model:
  - `forward.rs:16` `parse_static` splits into `global_static`,
    `per_ticker_static` (`trading_model.rs:387-401`).
  - `forward.rs:17` `build_exo_tokens` -> `build_exo_kv`
    (`trading_model.rs:446-473`) maps each of the `NUM_EXO_TOKENS = STATIC_OBSERVATIONS`
    scalars to one token via per-feature affine (`exo_feat_w/b`,
    `trading_model.rs:332-344`, einsum-free broadcast lines 458-461), then MLP-refines.
  - These tokens are consumed by the layer-0 exogenous cross-attention inside the trunk:
    `causal_patch_trunk` line 638, `self.exogenous_ticker_block.forward(&x, exo_tokens)`
    (`CrossAttnFfnBlock`, `blocks/cross_attn.rs:49-73`).

Therefore the deep trunk is currently NOT a pure function of exogenous price history: it
is conditioned on portfolio/account state via the 6 global + 6 per-ticker portfolio
features routed through the layer-0 cross-attention. This is exactly the coupling that
blocks trunk reuse across decision positions and must be relocated.

### 1.3 Multi-scale patch layout

`trading_bots/src/torch/model/config.rs`

- `BASE_PATCH_CONFIGS` (lines 24-33): coarse-to-fine `(days, patch_size)`:
  `(3072,128),(1536,64),(768,32),(384,16),(128,8),(64,4),(46,2),(2,1)`.
  Sum of `days` = `PRICE_DELTAS_PER_TICKER = 6000` (`shared/src/lib.rs:17`); enforced at
  `trading_model.rs:252-257`.
- Token count `S = sum(days/patch_size)` = `24+24+24+24+16+16+23+2 = 153`
  (`compute_patch_totals`, lines 105-117).
- Patch index != timestep. A patch's right edge (end day) is its causal frontier.
  `patch_ends_for_variant` (lines 123-136) returns cumulative end-day per patch token.
  Patches are laid out strictly left-to-right in increasing end-day, and `patch_config_ids`
  (`trading_model.rs:296-307`) records each token's scale. The finest scale `(2,1)`
  occupies the last 2 tokens (1-day patches), whose ends are the most recent two days.
- `patch_embed` (lines 503-557) enriches each patch with mean/std/slope over its own
  window only (`PATCH_SCALAR_FEATS = 3`), so a patch embedding is a function of the days
  within that patch — i.e. coarse patches summarize OLD spans, fine patches summarize
  recent spans. Coarse and fine patches overlap in absolute time only insofar as the
  layout partitions `[0, 6000)` contiguously (they do not overlap; configs are
  concatenated by `delta_offset`, lines 516-543).

Implication for dense readout: because all patches partition the same contiguous history
window and a coarse patch's window ends BEFORE the next finer block begins, the natural
causal order is patch-token order. The valid decision frontier is the finest-scale,
most-recent edge. See section 4.

---

## 2. Proposed architecture

### 2.1 Overview

1. Trunk becomes a pure encoder of exogenous price history + exogenous static features.
   It outputs per-patch contextual embeddings `h` of shape `[rows, S, model_dim]`
   (`rows = batch * TICKERS_COUNT`). No fork tokens, no `S+1`, no role embeds in the
   trunk.
2. A new `CausalPoolReadout` module holds two learned query seeds (actor, critic) and a
   GQA-conventioned cross-attention with a CAUSAL pool mask. For each decision position
   `t`, the actor/critic queries pool over `h_{<=t}` to produce `(actor_t, critic_t)`.
3. Portfolio/account state conditions the queries (FiLM), NOT the trunk.
4. Two modes: single-position (rollout) and dense (update).

### 2.2 Trunk changes

Modify `causal_patch_trunk` (rename to `encode_patch_trunk`) to drop fork construction:

- Remove `e_last`, `forks`, `role`, `role_bias`, the `+1` sequence extension
  (`trading_model.rs:616-625`).
- Input `x0 = patch_hidden` (`[rows, S, model_dim]`), `seq_len = S`.
- `rope_positions = arange(S)`; RoPE table can shrink to `seq_len` (drop the `+1` at
  `trading_model.rs:331`). `role_bias` passed to `forward_with_rope_positions` becomes a
  zero tensor (or the parameter is dropped from the trunk call; the GQA block keeps the
  arg for compatibility but the trunk passes zeros).
- Keep layer-0 exogenous cross-attention (line 638) but feed it ONLY exogenous static
  features (section 3); causal flag stays `true`.
- `maybe_apply_endogenous_ticker`: drop the fork-aware `live_idx` branch
  (`trading_model.rs:418-422`); mix all `S` patch positions across tickers, or restrict
  to a defined recent window. Recommended: mix every position (dense readout needs every
  position cross-ticker-aware). With `TICKERS_COUNT = 1` this path is a no-op
  (`trading_model.rs:408`), so it is non-blocking now but must be made layout-agnostic.
- Output: `h = final_ln(x)`, shape `[rows, S, model_dim]`.

`role_embeds` (`trading_model.rs:276-283`) is repurposed: instead of additive seeds on
`e_last`, the 2 rows become the actor/critic QUERY SEEDS in the readout (section 2.3). No
new param count if reused; rename to `readout_query_seeds`.

### 2.3 CausalPoolReadout module

New struct (suggested `model/blocks/pool_readout.rs`):

```
struct CausalPoolReadout {
    query_seeds: Tensor,      // [2, model_dim]  (actor, critic)  (reuse role_embeds)
    q_proj: Linear,           // model_dim -> q_dim   (GQA_NUM_Q_HEADS * head_dim)
    kv_proj: Linear,          // model_dim -> 2*kv_dim (GQA_NUM_KV_HEADS shared)
    q_norm: RMSNorm,          // head_dim
    k_norm: RMSNorm,          // head_dim
    q_gain: Tensor,           // [GQA_NUM_Q_HEADS]
    attn_out: Linear,         // model_dim -> model_dim
    out_ln: RMSNorm,
    // FiLM conditioning (section 3)
    film_actor: Linear,       // port_dim -> 2*model_dim (gamma,beta) for actor query
    film_critic: Linear,      // port_dim -> 2*model_dim
}
```

Conventions kept consistent with the trunk GQA (`gqa.rs`): `GQA_NUM_Q_HEADS = 4`,
`GQA_NUM_KV_HEADS = 1`, `head_dim = model_dim / 4`, q/k RMSNorm, `q_gain` per head, RoPE
with `ROPE_DIMS = 16`, scaling via `scaled_dot_product_attention`.

#### RoPE / role handling

- Keys are the trunk patch embeddings; key RoPE positions are the patch indices
  `arange(S)` (same indices used in the trunk), via `rope.apply_positions`
  (`rope.rs:50-66`). This keeps the readout's relative geometry identical to the trunk.
- Each actor/critic query at decision position `t` is RoPE-applied at position `t` (the
  query's own frontier index). This gives the query the same positional frame as key
  `t`, so "attend to <= t" is relative-position-consistent with the trunk.
- "Role" is now identity of the query seed (actor vs critic), not an additive bias inside
  blocks. The two seeds are distinct learned vectors; no per-block role re-injection.

#### Attention (per head, GQA broadcast of the single KV head)

- Q: `[rows, n_q (=2 per decision pos), n_heads, head_dim]`.
- K,V from `kv_proj(h)`: `[rows, S, 1, head_dim]` broadcast across q heads.
- Causal pool mask `M` of shape `[n_dec, S]` (decision positions x key patches):
  `M[t, j] = 0 if end_day(j) <= frontier_end(t) else -inf`. Because patch tokens are in
  increasing end-day order and `frontier_end(t)` equals the end-day of decision position
  `t` (section 4), this reduces to `M[t, j] = (j <= t_key_index)`. Passed as the
  `attn_mask` arg of `scaled_dot_product_attention` (currently `None` in the trunk,
  `gqa.rs:101-107`); the readout uses an explicit additive float mask, NOT the `causal`
  bool flag, because the query set is decision positions and the key set is patches.

### 2.4 Single-position mode (rollout)

- `n_dec = 1`, frontier = most-recent finest patch (`t = S-1`).
- Q = `[rows, 2, n_heads, head_dim]` from the two FiLM-conditioned seeds at RoPE position
  `S-1`. K,V = full `h` (`[rows, S, ...]`). Mask = all-visible (`<= S-1`), i.e. no mask
  needed.
- Output `pooled`: `[rows, 2, model_dim]` -> `attn_out` -> `out_ln`. Split into
  `actor_read [rows, model_dim]`, `critic_read [rows, model_dim]`.
- Feed to `head_from_actor_critic_cls(actor_read, critic_read, batch_size)` unchanged
  (`head.rs:9-31`).

### 2.5 Dense mode (update)

- Decision positions = all valid frontiers `D = {t_0, ..., t_{n_dec-1}}` (section 4),
  expressed as key indices into `h`.
- Q = `[rows, n_dec * 2, n_heads, head_dim]`: for each `t in D`, two FiLM-conditioned
  seeds RoPE-applied at position `t`. FiLM uses the portfolio state AT decision `t`
  (section 3), so per-position portfolio vectors are required as input.
- K,V = `h` `[rows, S, ...]`.
- Attention scores `[rows, n_heads, n_dec*2, S]` masked by `M` expanded to query rows:
  for the two queries of decision `t`, mask is `key_index <= t`. Build `M` as
  `[n_dec, S]` then `repeat_interleave(2, dim=0)` -> `[n_dec*2, S]` -> broadcast over
  `rows, n_heads`.
- Output `[rows, n_dec, 2, model_dim]`. Split last-but-one dim into actor/critic stacks:
  `actor_reads [rows, n_dec, model_dim]`, `critic_reads [rows, n_dec, model_dim]`.
- Reshape to `[rows * n_dec, model_dim]` and call `head_from_actor_critic_cls` with
  effective batch `batch_size * n_dec` (head is position-agnostic; it views
  `[batch, TICKERS_COUNT, model_dim]`). Output value logits `[batch*n_dec, NUM_BINS]`,
  alpha/beta `[batch*n_dec, TICKERS_COUNT]`. Caller reshapes back to `[batch, n_dec, ...]`.

Shapes summary (base variant, `S=153`, `model_dim=256`, `n_heads=4`, `head_dim=64`):

| tensor | single | dense |
|---|---|---|
| h (K,V source) | `[rows,153,256]` | `[rows,153,256]` |
| Q | `[rows,2,4,64]` | `[rows,2*n_dec,4,64]` |
| mask | none | `[n_dec,153]` -> `[2*n_dec,153]` |
| pooled | `[rows,2,256]` | `[rows,n_dec,2,256]` |
| actor/critic read | `[rows,256]` each | `[rows,n_dec,256]` each |

### 2.6 Trainer / rollout integration

- Rollout (`train/rollout.rs`) calls single-position readout per step. The trunk still
  runs per step in streaming today; the dense win is realized in the UPDATE path. If the
  trunk is cached across an episode (future work, enabled by section 3), rollout can also
  call dense over accumulated frontiers.
- Update: replace per-decision forward with one trunk encode per (episode-window,
  batch-row) + one dense readout producing all `n_dec` decisions. The optimizer then
  consumes `[batch, n_dec, ...]` logits against the matching stored actions/returns.

---

## 3. Portfolio conditioning design

Goal: deep trunk = pure function of exogenous inputs (price history + exogenous static
features), so it can be encoded once and reused across decision positions; portfolio/
account state enters ONLY at the readout query.

### 3.1 Split static features into exogenous vs portfolio

- Define `PORTFOLIO_GLOBAL = 6` (the first 6 globals, `obs.rs:19-28`) and
  `PORTFOLIO_PER_TICKER = 6` (the first 6 per-ticker, `obs.rs:53-66`). Total
  `port_dim = 6 + TICKERS_COUNT * 6 = 12` for `TICKERS_COUNT=1`.
- `parse_static` (`trading_model.rs:387-401`) gains a second split that separates
  portfolio scalars from exogenous scalars within both the global and per-ticker blocks.
  Cleanest: reorder the observation so portfolio features are contiguous at the front of
  each block (env change in `obs.rs` + constants in `shared/src/lib.rs`), OR keep the
  layout and slice the known portfolio indices. Reordering is preferred for clarity and
  to make `NUM_EXO_TOKENS` exclude portfolio cleanly.
- `build_exo_tokens` / `build_exo_kv` (`trading_model.rs:446-473`) now build tokens from
  EXOGENOUS features only. `NUM_EXO_TOKENS` drops by `port_dim`
  (`config.rs:21` derivation changes). The layer-0 cross-attention thus no longer sees
  portfolio state. THIS IS THE REMOVAL POINT for the old injection.

### 3.2 FiLM modulation of the query seeds

- Portfolio vector `p_t` of shape `[rows, port_dim]` for decision `t` (per-position in
  dense mode).
- `gamma_a, beta_a = chunk(film_actor(p_t), 2)`; `q_actor = query_seeds[0] * (1 +
  gamma_a) + beta_a`. Same with `film_critic` for the critic seed. (FiLM = feature-wise
  affine; `(1+gamma)` init so zero-init FiLM weights => identity, preserving the learned
  seed at start.)
- `film_actor/film_critic` initialized to zero weights, zero bias => readout starts as
  pure pooling with unconditioned seeds; portfolio influence is learned. This makes the
  whole change a clean ablation toggle (disable FiLM => exogenous-only readout).
- Alternative considered: concat-then-project (`q_seed_concat = proj([seed; p_t])`).
  FiLM is preferred: cheaper, identity-initializable, and keeps the seed semantics. Spec
  the concat variant as the secondary ablation.
- Per-position portfolio in dense mode: the env must expose `p_t` for every decision
  frontier (it already computes account state per step in `obs.rs`; the update path must
  store the per-step portfolio sub-vector alongside actions/returns).

### 3.3 What is removed

- `causal_patch_trunk` fork/role injection (`trading_model.rs:616-625, 653-654`).
- Portfolio scalars from `build_exo_kv` (`trading_model.rs:446-462`) and from
  `NUM_EXO_TOKENS`.
- Fork-aware `live_idx` logic in `maybe_apply_endogenous_ticker`
  (`trading_model.rs:418-422`).

---

## 4. Multi-scale dense-readout validity

### 4.1 Which positions are valid frontiers

A decision at frontier end-day `e` may see exactly the patches whose windows end at or
before `e`, and must not see any patch whose window extends past `e`.

- Patch tokens are ordered by increasing end-day (`patch_ends_for_variant`,
  `config.rs:123-136`), partitioning `[0, 6000)` contiguously. So "patches visible at
  end-day `e`" = a PREFIX of the patch-token sequence. The causal pool mask
  `key_index <= t` is therefore exact when `t` is chosen as a patch-token index whose
  end-day equals the decision's frontier end-day.
- The agent makes one decision per finest (1-day) step. The finest scale `(2,1)`
  contributes the last 2 tokens (1-day patches, `config.rs:32`), and these are the only
  tokens whose end-day advances by 1 day. Coarser tokens have end-days fixed at coarse
  boundaries and never coincide with arbitrary daily frontiers.
- Therefore, for a single fixed observation window, the only patch index whose end-day
  is the live decision day is the LAST finest token (`t = S-1`). Dense readout over many
  decisions within ONE static trunk encoding is only causally valid at the SET of
  finest-scale frontier indices that correspond to distinct decision days represented in
  that encoding.

### 4.2 The complication and the precise rule

The current observation is a SLIDING window ending at the live day; consecutive decisions
have DIFFERENT windows (the patch boundaries shift). A single trunk encode of one window
does NOT contain valid coarse summaries for a different day's window. Two valid regimes:

- Regime A (safe, recommended first): dense readout valid only at the finest-scale
  frontier indices present in the encoded window. For `BASE_PATCH_CONFIGS` that is just
  the 1-day tokens, i.e. up to 2 decision positions per encoded window
  (`days=2, patch_size=1`). This yields a small dense factor unless the finest block is
  widened.
- Regime B (the real compute win): redesign the layout so the finest scale spans the
  decision horizon you want to densify. To emit `K` decisions from one trunk encode, the
  finest `(days, patch_size=1)` block must have `days >= K`, and the readout treats each
  of the last `K` 1-day tokens as a decision frontier `t in {S-K, ..., S-1}`. The mask
  `key_index <= t` guarantees decision `t` sees only patches ending on/before day `t`.
  Coarse patches summarize the FIXED older span and are shared/valid across all `K`
  frontiers because their windows end before `S-K`'s frontier. This is leak-safe iff
  every coarse patch's end-day `<= end_day(S-K)`.

Position-selection rule (precise): `D = { j : patch_config_ids[j] == finest_config_id }`
intersected with the densify horizon `K`, i.e. the last `K` finest tokens. Frontier
end-day of token `j` = `patch_ends[j]`. Mask `M[t,j] = 0 if j <= t else -inf`. Because
ends are monotonic, `j <= t  <=>  patch_ends[j] <= patch_ends[t]`, so index masking is
end-day masking.

### 4.3 Leak-safety argument

- Keys are per-patch embeddings; each is a function only of days inside its own window
  (`patch_embed`, `trading_model.rs:516-543`, scalar feats over the patch only). No
  cross-patch leakage at embed time.
- Trunk self-attention is causal (`gqa.rs:98-107`, `causal=true`), so `h_j` depends only
  on patches `<= j` (end-day monotonic => only older/equal spans). Thus `h_j` carries no
  information past `end_day(j)`.
- Readout decision `t` attends only to `h_{<= t}` via mask. Hence decision `t` depends
  only on price information up to `end_day(t)`. No future leak, for any `t in D`.
- Exogenous static features in the trunk (macro/momentum/earnings) are read at
  `absolute_step` (`obs.rs:114, 134`); for dense readout they must be the values valid at
  each decision's day or be confined to features that are constant across the densified
  horizon. FLAG: if exogenous static features are time-varying within the densify horizon,
  feeding the SAME exo tokens to all `K` frontiers leaks/mismatches. Resolution: either
  (a) restrict dense readout to horizons over which exo static is constant, or (b) move
  time-varying exogenous static into the readout per-position too. Open question 8.3.

---

## 5. Preserved contracts

Head (`model/head.rs:9-31`) is UNCHANGED. The readout must deliver `actor_read` and
`critic_read` each shaped `[effective_batch, model_dim]` (effective_batch =
`batch_size` in single mode, `batch_size * n_dec` in dense mode):

- Actor: `policy_concentration` (`trading_model.rs:356-357`) -> 2 outputs per ticker ->
  `beta_concentration` -> `(alpha, beta)` each `[batch, TICKERS_COUNT]`. Unchanged.
- Critic: `value_proj` over `[batch, TICKERS_COUNT * model_dim]`
  (`trading_model.rs:358-359`, `head.rs:23-24`) -> `NUM_BINS = 21` HL-Gaussian logits.
  Unchanged. Note the critic read is flattened across tickers, so the critic query must
  produce a per-ticker `model_dim` vector exactly as today (rows include the ticker
  dim).
- `ModelOutput = (value_logits [.,21], alpha [.,T], beta [.,T])`
  (`trading_model.rs:25`). Unchanged; dense mode returns the same tuple with the leading
  dim multiplied by `n_dec`.

---

## 6. Compute analysis

Let `L = gqa_layers` (3 base), `S` = patch tokens (153 base), `T` = decisions per episode
(`EPISODE_TRANSITIONS = 2000`, `shared/src/lib.rs:38`), `d = model_dim`.

Current (per episode):
- Each decision runs a full trunk over `S+1` tokens: attention `O(L * (S+1)^2 * d)` plus
  FFN `O(L * (S+1) * d * ff)`. Over `T` decisions: `O(T * L * S^2 * d)`.
- This is the dominant cost and scales linearly in `T` with the full trunk constant.

Proposed:
- Trunk encoded ONCE per window: `O(L * S^2 * d)`. Across the episode, if the trunk is
  re-encoded per sliding window the trunk cost is still `O(T * L * S^2 * d)` UNLESS
  windows are batched / the finest block is widened so one encode serves `K` decisions,
  reducing trunk encodes to `O((T/K) * L * S^2 * d)`.
- Dense readout: cross-attention of `2*n_dec` queries over `S` keys =
  `O(n_dec * S * d)` per head set, negligible vs trunk. For one window densifying `K`
  decisions, readout is `O(K * S * d)`.
- Net: with the Regime-B layout (finest block width `K`), per-episode cost drops from
  `O(T * L * S^2 * d)` to `O((T/K) * L * S^2 * d) + O(T * S * d)`. The `K`x reduction in
  trunk encodes is the win; readout is cheap.

Assumptions: trunk reuse requires the trunk to be portfolio-independent (section 3),
which this spec enables. The full `K`x win additionally requires the layout/horizon
redesign (Regime B); Regime A gives a small constant (`K<=2` for base configs).

---

## 7. Risks / ablations

Risks (what could regress):
- Removing portfolio from the trunk may hurt if the trunk currently uses portfolio to
  shape price representations (it likely should not, but it is a behavioral change).
- Pooling readout has strictly less capacity than the current "one more full causal pass"
  fork readout (which runs FFNs over the decision tokens). May underfit; mitigate with a
  small post-pool FFN in the readout.
- FiLM on a single seed is low-capacity for portfolio conditioning; concat-project may be
  needed.
- Dense exo-static time-variance leak (section 4.3) if horizon misused.
- `maybe_apply_endogenous_ticker` change is inert at `TICKERS_COUNT=1` but must be
  correct before multi-ticker.

Ablations to run / measure:
- A0: current fork readout (baseline).
- A1: pooling readout, portfolio STILL in trunk (isolate readout change).
- A2: pooling readout + portfolio moved to FiLM query (the full proposal).
- A3: A2 with concat-project instead of FiLM.
- A4: A2 + post-pool FFN in readout.
- A5: dense vs single equivalence test (dense readout at `t=S-1` must match single mode
  bit-for-bit modulo numerics).
Metrics: episode return / Sharpe (benchmark_results), value loss & explained variance,
policy entropy, sample throughput (decisions/sec) to confirm the compute win, and a
leak-probe (shuffle future-only inputs and confirm no change in earlier decisions).

---

## 8. Open questions for the human

1. Layout: adopt Regime B (widen the finest 1-day block to a densify horizon `K`) now, or
   ship Regime A first (correctness, small `K`) and widen later? `K` choice trades trunk
   reuse against finest-scale token count / S growth.
2. Observation reorder: acceptable to reorder static features so portfolio scalars are
   contiguous (changes `shared/src/lib.rs` + `env/obs.rs` + any report tooling), or slice
   by fixed indices to avoid touching the env contract?
3. Time-varying exogenous static across the densify horizon (4.3): restrict horizon, or
   route per-position exo into the readout too?
4. FiLM vs concat-project as primary; and whether the critic and actor should share FiLM
   parameters or stay independent.
5. Should the readout add a post-pool FFN (capacity parity with the old fork pass), or
   stay a pure pooling layer for the compute win?
6. RoPE for queries: confirm query position = its own frontier index `t` (recommended) vs
   position 0 (PMA convention). Affects relative geometry vs the trunk.
7. Rollout: keep per-step trunk encode (no behavioral change) and realize the dense win
   only in the update, or also batch rollout windows for trunk reuse?
8. Do we keep `role_embeds` count at 2 (reuse as query seeds) or expand
   (e.g. multi-query pooling per head role)?

---

File written: `trading_bots/docs/causal_pooling_readout_spec.md`
