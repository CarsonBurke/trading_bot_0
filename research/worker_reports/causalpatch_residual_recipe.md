# CausalPatch residual recipe (Change A) — modded-nanogpt block internals

Worktree `worktrees/residual-recipe` (branch `worker/residual-recipe`). Reference checkout
`/home/marvin/Documents/repositories/modded-nanogpt` @ master; all `train_gpt.py` /
`train_gpt_medium.py` line numbers are from those files as they stand there now. In-repo
secondary evidence is `trading_bots/src/torch/world_model.rs` (BarTrunk already runs this recipe)
and `trading_bots/src/torch/model/rmsnorm.rs`.

## What I copied into the worktree

`git worktree add worktrees/residual-recipe -b worker/residual-recipe` off `85374302`, then
rsynced from the main tree every path `git status --short` listed as modified, plus the two
untracked directories and the one untracked file, and deleted the one path marked `D`:

- modified, copied verbatim: `benchmarks/timexer_universe_campaign.py`, `docs/timexer_segment.md`,
  `report_cli/src/main.rs`, `shared/src/report.rs`, `trading_bots/src/torch/cuda/graph.rs`,
  `trading_bots/src/torch/cuda/mod.rs`, `trading_bots/src/torch/dataset.rs`,
  `trading_bots/src/torch/model/mod.rs`, `trading_bots/src/torch/model/rope.rs`,
  `trading_bots/src/torch/optim/muon.rs`, and all of
  `trading_bots/src/torch/timexer_segment/{benchmark,compute,corpus,data,mod,model,reports,runner}.rs`,
  `tui/src/main.rs`, `tui/src/report_renderer.rs`, `vendor/torch-sys-0.25.0/{libtch/torch_api.cpp,libtch/torch_api.h,src/lib.rs}`
- untracked, copied: `trading_bots/src/torch/timexer_segment/features.rs`,
  `research/worker_reports/`, `research/papers/`
- deleted in the working tree, deleted here: `trading_bots/src/torch/timexer_segment/geometry.rs`

The snapshot therefore already contained `KernelTraffic`'s in-flight `Block::rotate`,
`rotation_tiles`, the `8·width` rotary term in `step_cost` and both of their new tests. I edited
nothing in the main tree. Builds/tests run as
`FA4_VENV=<main tree>/.venv-fa4 ./torch-env.sh cargo …` because the worktree has no `.venv-fa4`.

## How to integrate this (verified against the live main tree, not just my worktree)

My snapshot is now OLDER than the main tree in two files, so taking my worktree wholesale would
silently revert sibling work. I checked which files that applies to instead of guessing, and shipped
isolated patches for the ones where it does. All three were `git apply --check`ed against the live
main tree and apply cleanly:

| file | how to take it | why |
| --- | --- | --- |
| `trading_bots/src/torch/timexer_segment/model.rs` | **wholesale from my worktree** | I diffed every line my version removes from the live file: all 84 are pre-recipe code (`LayerNorm`, `GELU`, `residual add`, the old `Block::forward` signature) plus `KernelTraffic`'s `twice` closure, which they explicitly asked to be superseded by my `charged(name)` table. `KernelClass`'s field set is identical in both, and my `step_cost` keeps their `8·width` rotary term (22 = 2 norms + 3 packed + 8 rotary + 2 QK-norm + 6 attention/residual + 1 x0). Nothing newer is lost. |
| `trading_bots/src/torch/timexer_segment/compute.rs` | **`research/worker_reports/causalpatch_residual_recipe_compute.patch`** (7 hunks) | My snapshot predates `GraphCapture`'s rewrite of `audit_step_capture`/`CaptureAudit` (the three-phase eager-vs-eager null became two phases). Taking my file would revert it. The patch carries only my hunks: the recipe string, `mut optimizer`, the `"lambda"` no-decay entry, the asserting `set_named_lr_scale`, the fixture swap, the AdamW count 7→8, and the new routing test. |
| `trading_bots/src/torch/timexer_segment/benchmark.rs` | **`research/worker_reports/causalpatch_residual_recipe_benchmark.patch`** (3 hunks) | My snapshot predates `KernelTraffic`'s `KernelRow` forward/backward byte split, the forward-only rate columns and the GIL comment — 5 of the 9 diff hunks would revert those. The patch instead applies only the semantic renames on top of their current file: repeat table `"LayerNorm"→"RMSNorm"` (2·layers) and `"residual add"→"residual addcmul"` (**3**·layers), and the parts-sum special case becomes a `match` with the same 2/3 weights, keeping their `reference *` filter. |
| `docs/timexer_segment.md` | **`research/worker_reports/causalpatch_residual_recipe_docs.patch`** (3 hunks) | Same problem: my snapshot predates their whole `### Benchmark reports` section, which one of my two diff hunks would delete. The patch rewrites the backbone bullet, adds the two recipe bullets, and additionally renames the kernel classes inside **their** new section (`RMSNorm`, `QK RMSNorm`, `residual addcmul`, `ReLU²`) so the doc matches the code. |

The two `.rs` patches are apply-verified but not compile-verified against the live tree (the live
tree is still being edited); the compile and test proof for the equivalent edits is my worktree,
where `cargo check --tests` is clean and `cargo test … timexer_segment` is 43/43. Both patches are
mechanical — a match arm, a string, a closure→`match` on `row.name` — with no new types.

### Handoff to `FusedKernels` on the ReLU² term

Asked which tensor the fused backward should save at `[96000, 2048]` bf16, my answer with the
arithmetic (one such pass = 0.3932 GB; 8 layers, forward + 2×backward convention):

| form | fwd | bwd | GB/step | note |
| --- | --- | --- | --- | --- |
| GELU (pre-recipe baseline) | 2·hidden | 3·hidden | 15.73 | |
| `relu().square()` as shipped | 4·hidden | 6·hidden | 31.46 | the +18.87 GB I flagged |
| fused ReLU², save `x` | 2·hidden | 3·hidden | 15.73 | **exact, free, and peak-neutral** — `x` is the up-GEMM's output, so saving it costs zero extra write and zero extra allocation, only a delayed free; `grad_x = 2·max(x,0)·grad_y`. **Shipped.** |
| fused ReLU², save `r = relu(x)` | 3·hidden | 3·hidden | 18.87 | one extra `[tokens, ffn]` write for nothing: the only kernel output is `y`, so `r` has to be emitted, and it buys one `max`. Peak-neutral, purely wasted traffic |
| fused ReLU², recompute `sqrt(y)` | 2·hidden | 3·hidden | 15.73 | frees the last live copy: **−375.0 MiB peak**, since `y` is already retained by the down-GEMM's weight gradient. Costs ~2^-9 relative error on `grad_x`, a *systematic* bias (√ of a rounded square), and gives up bit-identity — `FusedKernels` correctly declined it and kept it as a documented variant |
| relu² as an up-GEMM epilogue | 1·hidden | 3·hidden | 12.58 | `x` never materializes: **−18.87 GB/step vs unfused and 3.15 GB below the GELU baseline**. The version worth building |

Never save `relu(x)`: unlike `x` it does not already exist once the kernel is fused (the only
output is `y = r²`), so emitting it costs an extra `[tokens, ffn]` write to save one `max` on a
kernel that is 100% bandwidth-bound. Sign is safe for the `√` variant — `r ≥ 0`, and `x < 0` gives
`y = 0 → grad_x` exactly 0, no select needed. `2·relu(x)` is unbounded in `x`, unlike GELU's
derivative — safe at this call site only because the FFN input is gainless-RMS-normalized, so
`x = O(1)` by construction, but the kernel must not assume boundedness if reused elsewhere.

**Two things I got wrong here, corrected by `FusedKernels` and worth keeping straight:**
- *Peak.* I implied save-`x` might cost peak and that save-`r` would cost +375 MiB live. Neither.
  `relu`'s backward reads its OUTPUT (`threshold_backward` on the result) and `addmm`'s backward
  never wants its output, so the shipped `relu().square()` already retains exactly one live
  `[96000, 2048]` bf16 — `r`, with `x` freed. Save-`x` is one live tensor too, so it is
  peak-NEUTRAL, and the −375 MiB of the `√y` variant is measured against today and against
  save-`x` equally. Save-`r`'s cost is write traffic only, not peak.
- *Rounding.* My "use one fp32 rounding, not two" caution was vacuous: `grad * 2` on a bf16 only
  decrements the exponent, so that multiply is EXACT, and `r * r` of two 8-bit mantissas is exact
  in fp32. The composition already has exactly one inexact rounding in each direction, so the
  fused kernel is **bit-identical** to `relu().square()` rather than merely more accurate — which
  is the stronger property, and the reason to prefer it over `√y` outright. No test tolerance is
  needed when their kernel lands at this call site.

Downstream note for whoever lands their RoPE kernel on top of this change: it consumes the
UNTILED `[origins, head_dim/2]` cos/sin (24 KiB, L2-resident) and indexes them internally, so
`rotation_tiles` and the `[1, origins, 2·d_model]` tiles disappear at the call site. That deletes
the `8·width` rotary term from my `step_cost` per-layer count (22·width → ~14·width + the kernel's
own 2 reads/2 writes) and replaces my `"rotary rotation"` kernel class. QK-norm is unaffected: the
kernel sees already-normalized `q`, `k`.

## Per item: reference, formula, init

**Norm placement — the reference is PRE-norm, and so were we already.** Task text said
post-norm; it is wrong. `train_gpt_medium.py:1020-1023` is
`x = x + self.attn(norm(x))` / `x = x + self.mlp(norm(x))`; `train_gpt.py:1598`
(`attn_in_normed = norm(cache.get(7, x))`), `:1643` (`normed = norm(x)`), `:1682` (final
`x = norm(x)`). Our `Block::forward` already normalized before QKV and before the FFN and added
onto the raw stream, and `docs/timexer_segment.md:16` says "pre-norm". Nothing was converted; the
lambda scheme needs pre-norm, since it rescales the raw residual stream.

1. **RMSNorm replaces LayerNorm.** `train_gpt.py:952-953` / `train_gpt_medium.py:839-840`:
   `def norm(x): return F.rms_norm(x, (x.size(-1),))` — `weight=None`, so it is **gainless and
   biasless**, i.e. the reference has *no* norm parameters anywhere. Implemented as a free
   function `rms_norm` (`model.rs`), `x.internal_fused_rms_norm([width], None, Some(1e-6)).0`.
   Two deliberate choices:
   - `_fused_rms_norm`, not `rms_norm`: `pretrain_profile.rs:95-99` documents that `rms_norm`
     registers as a *math composite* on CUDA and hides the real kernel; `model/rmsnorm.rs:17` is
     the in-repo form.
   - explicit `eps = 1e-6` (`world_model.rs:109-110`, `BAR_NORM_EPS`). `F.rms_norm(x, shape)`
     leaves `eps=None`, which ATen resolves to `finfo(bf16).eps = 7.8e-3` — a 0.4% systematic
     shrink of every normalized activation that would not reproduce in an fp32 CPU test. The
     reference itself passes `eps=1.0e-6` where it passes one (`train_gpt.py:1079`).
   Routing consequence: **zero new tensors**, and 34 tensors (17 norms × gain+bias) *removed*
   from AdamW. There is no norm gain to misroute.
2. **QK-norm, then RoPE.** `train_gpt.py:1103-1109`: `:1106` `q, k = norm(q), norm(k)  # QK norm
   @Grad62304977`, then `:1109` `q, k = yarn.rotary(q), yarn.rotary(k)`. Identically
   `train_gpt_medium.py:968` before `:970`, and in-repo `world_model.rs:1396-1397` →
   `:1722-1724`. Implemented as ONE kernel over the packed `q‖k` block:
   `rms_norm(packed[0].reshape([b, t, 2·heads, head_dim])).reshape([b, t, 2·width])`, which is
   exactly per-head, per-token normalization of both tensors because the packed column layout is
   already `tensor·heads·head_dim + head·head_dim + dim` (`Block::rotate`'s own doc). V is *not*
   normalized, matching the reference. `Block::rotate` itself is untouched.
3. **ReLU² replaces GELU** in the FFN. `train_gpt_medium.py:1010`
   (`x = F.relu(x).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU`) and
   the fused `relu(x @ W1.T)^2 @ W2.T` Triton kernel at `train_gpt.py:46-48`.
   **No output-scale correction in the reference**, and none here: the down projection is
   zero-initialised, the post-lambda can absorb any residual scale, and every consumer of the
   stream is a gainless RMSNorm. Width also needs no correction — the reference's MLP hidden is
   `4 * model_dim` (`train_gpt.py:1299`, `mlp_hdim = 4 * model_dim`), which is exactly our fixed
   2048 at `d_model` 512. The head MLP's GELU (`model.rs`, `head_hidden`) is **not** the FFN and
   is left alone.
4. **Zero-init branch outputs.** `train_gpt_medium.py:1006` (`self.c_proj.zero_()`),
   `train_gpt.py:1308` (`self.mlp_bank[:, 1, :, :].zero_()`), both "zero init suggested by
   @Grad62304977"; the attention output projection is `CastedLinearT`, whose
   `reset_parameters` is `nn.init.zeros_(self.weight)` (`train_gpt.py:973-975`). New
   `zeroed_projection` for `output` and `second`.
   Per Main's decision (a) the four block projections are now **bias-free**
   (`train_gpt.py:1103`, `:1161` call `F.linear` with a weight only; `train_gpt_medium.py:995-996`
   are bare weight parameters). QKV and FFN-up moved to the reference 2-D init
   `uniform(±√3·0.5·fan_in^-½)` (`train_gpt.py:1288-1291`, `:1304-1307`,
   `train_gpt_medium.py:1002-1005`, in-repo `world_model.rs:2903-2909`) — 0.866× the previous
   `1/√fan_in` bound.
5. **Residual / post / x0 lambdas, identity at init.** Parameterization is **raw scalars, not
   logits, one per sub-block** — `train_gpt.py:1334` `self.post_lambdas = nn.Parameter(torch.ones(num_layers, 2))`,
   `:1336-1338` `self.resid_lambdas = nn.Parameter(torch.full((num_layers, 2), 1.1**0.5))`
   ("sqrt(1.1) per sublayer so cumulative per-layer scaling is 1.1"), applied unmodified at
   `:1638` `x = resid_lambdas_attn[i] * x + post_lambdas_attn[i] * attn_out + x0 * x0_gates[i]`
   and `:1665` `x = resid_lambdas_mlp[i] * x + post_lambdas_mlp[i] * ReLUSqrdMLP(normed, ...)`.
   x0 lambda init 0: `train_gpt.py:1389` (`bs_init[0, 10] = 0.0  # x0_lambda[10] (init 0)`) and
   `train_gpt_medium.py:1078` (`self.x0_lambdas = nn.Parameter(torch.zeros(2*num_layers))`).
   `x0` is the *normalized* embedding: `train_gpt.py:1549` / `train_gpt_medium.py:1158`
   `x = x0 = norm(x[None])` — so `backbone` now does `x0 = rms_norm(patch(tokens))` and threads
   `&x0` into every block.

   Exact values used: `RESID_LAMBDA_INIT = 1.048_808_848_170_151_6` (= √1.1, matching in-repo
   `world_model.rs:112-113`), `POST_LAMBDA_INIT = 1.0`, `X0_LAMBDA_INIT = 0.0`.
   At init: branch outputs are 0 (zero weight, no bias), `λ_0 = 0`, so
   `x_L = 1.1^layers · x0` and the gainless final `rms_norm` removes the scale — the backbone
   returns `rms_norm(x0)`, i.e. the model *is* patch-embedding → head at step 0.

   `sa_lambdas` (`train_gpt.py:1344`, `:1103`, `:1161`) deliberately NOT ported (Main decision d).

## Optimizer routing (every new tensor)

Three 1-D banks under a non-`block_` path: `lambdas.resid` `[2·layers]`, `lambdas.post`
`[2·layers]`, `lambdas.x0` `[layers]`. They are AdamW-routed on two independent counts —
`optim/muon.rs:1316` requires `dim() == 2`, and `muon_name_allowlist = ["block_"]`
(`compute.rs:115`) excludes them anyway. 1-D is on purpose: the reference's packed `[layers, 2]`
form is 2-D and would be one dropped allowlist entry away from being orthogonalized as if it were
a weight matrix. The `compute.rs:141-152` assertion (NorMuon ⇔ 2-D `block_*.weight`) is unchanged
and still holds.

Two additive `compute.rs` edits (cleared with `GraphCapture`, who confirmed they do not touch
`polar_express`, the no-decay list, the routing branch, `MuonConfig`'s fields, or that test):
- `adamw_no_weight_decay_name_substrings` gains `"lambda"` — the reference gives resid/post/x0
  `wd_mul = 0` (`train_gpt.py:2035-2036`, `train_gpt_medium.py:1081`), and with
  `quadratic_lr_weight_decay` a decay on a scalar whose useful value is ~1.05 is a pure bias
  toward deleting the residual stream.
- `set_named_lr_scale(&["lambdas.resid", "lambdas.x0"], 5.0)` — `train_gpt.py:2036`
  (`"resid_lambdas": … "lr_mul": 5.0`), `train_gpt_medium.py:1080` (`x0_lambdas.lr_mul = 5.0`);
  `post_lambdas` stays 1× (`train_gpt.py:2035`).
- Optimizer recipe string extended: `…patch-covariates-norm-bias-lambda-head-AdamW;lambda-noWD;…`.

**Deviations to note:** betas for `lambdas.x0` are the default `(0.9, 0.95)`;
`train_gpt_medium.py:1077` puts x0 lambdas in a separate "no beta smoothing" group and BarTrunk
uses `(0.9, 0.99)`. `MuonConfig` has weight-decay multipliers but no per-name *beta* override for
this case beyond `adamw_beta_overrides`, which I left untouched to avoid colliding with
`GraphCapture`; flagging rather than guessing.

Net parameter change: **−54 272** (36 864 block biases + 17 408 norm gains/biases) and **+40**
(the three lambda banks).

## Traffic delta (batch 256, 96 000 tokens, `ModelConfig::step_cost`)

`step_cost` per-layer activation units go `19·width + 2·ffn` → `22·width + 3·ffn`, plus one
`width` unit for the embedding norm. Every number below is `3 × 2 × bf16(tokens · units)`, the
function's own convention (write+read, forward plus two backward passes):

| item | GB/step | % of the old 155.2 GB total |
|---|---|---|
| QK RMSNorm, `+2·width`/layer | +9.44 | +6.1% |
| x0 injection's second `addcmul`, `+1·width`/layer | +4.72 | +3.0% |
| unfused `square` after `relu`, `+1·ffn`/layer | +18.87 | +12.2% |
| embedding RMSNorm, once | +0.59 | +0.4% |
| **total** | **+33.62** | **+21.7%** (155.2 → 188.8 GB) |

**RMSNorm does reduce traffic, but the saving is small and I will not overclaim it.** It is
invisible to `step_cost`, which counts only activations; it shows up in the per-kernel-class
accounting: per norm invocation LayerNorm charged `2·state + fp32(2·tokens) + cast(2·width)` =
197.382 MB and gainless RMSNorm charges `2·state + fp32(tokens)` = 196.992 MB, i.e. **0.390 MB
per norm** (the dropped `mean` output and the dropped gain/bias parameter cast). Over the 17
shared norms and the ×3 forward/backward convention that is **19.9 MB/step = 0.013%**. Dropping
the four block biases removes another 0.66 MB of parameter casts. The real RMSNorm wins are
qualitative: one fewer reduction output, 34 fewer AdamW tensors, 32 fewer bias tensors → 66 fewer
tiny optimizer kernels per step, and no parameter that *can* be misrouted.

**Mitigations applied (Main decision c).** The naïve transcription of `train_gpt.py:1638`/`:1665`
is five elementwise passes per layer (`λ_r·x`, `λ_p·out`, `λ_0·x0`, two adds) where the old code
had two adds — about +30 GB on the lambda scheme alone. Instead:
- `λ_p` is folded onto the output-projection weight copy the cast already makes
  (`scaled_linear`), exactly as the reference does with its own sub-block scalar
  (`train_gpt.py:1161`, "sa_lambdas[1] pre-multiplied to O @shenberg") and as in-repo
  `world_model.rs:1414`. A 512×512 weight is 1/96 000 of the activation. **Saves ~9.4 GB.**
- `λ_r` rides `Tensor::addcmul` (`self + t1*t2`, one kernel, two reads and a write) — the
  feedforward residual line therefore costs *exactly* what the bare `state + ff` cost.
  **Saves ~9.4 GB.**
- QK-norm is one kernel over the packed `q‖k` block rather than two over `q` and `k`
  separately: same bytes, one fewer launch per layer.

**What could not be folded, named:** `aten::square` after `aten::relu` (+18.87 GB, 56% of the
whole delta). `relu(x)²` needs one materialized `[tokens, ffn]` tensor for the ReLU output and
one for the square; there is no ATen fused ReLU² and no exact single-kernel rewrite
(`x·relu(x)`, `clamp_min(0).pow(2)`, `x²·1[x>0]` are all two kernels; in-place `pow_` breaks
autograd because the power's backward needs its pre-op input). The reference pays this once, in a
fused Triton kernel (`train_gpt.py:46-48`). **This is the one candidate for the perf track: a
fused ReLU² would return 12.2% of the step.** Also irreducible: the QK-norm output (+9.44 GB, the
normalized `q‖k` must be materialized before the rotation reads it — folding `rstd` into the
rotary products would need a 3-operand fused multiply) and the x0 `addcmul` (+4.72 GB, one extra
pass; the medium-track alternative of scaling `x` once per layer before the block costs +5 units
instead of +3, i.e. more).

`matmul_flops` is unchanged. Kernel-class FLOPs: RMSNorm 3/element vs LayerNorm 8, ReLU²
2/element vs GELU 8.

## Files touched, and the shared surfaces

- `model.rs`: `NORM_EPS`/`RESID_LAMBDA_INIT`/`POST_LAMBDA_INIT`/`X0_LAMBDA_INIT`;
  `LayerNorm` struct → free `rms_norm`; new `BlockLambdas`, `zeroed_projection`,
  `hidden_projection`, `scaled_linear`; `Block` loses `norms`, `Block::new` init, `Block::forward`
  rewritten; `CausalPatchModel` loses `final_norm`, gains the three lambda banks; `new()`;
  `backbone()`; new `recipe_scalars()`; `kernel_classes` ("LayerNorm"→"RMSNorm", new "QK RMSNorm",
  "GELU"→"ReLU^2", "residual add"→"residual addcmul", folded-lambda projection classes, composed
  layer); `ModelConfig::step_cost` per-layer units.
- `compute.rs`: the two routing lines above + the recipe string + one new test.
- `benchmark.rs`: the per-class repeat table only — `"RMSNorm" => 2·layers`,
  `"residual addcmul" => 3·layers`.
- `docs/timexer_segment.md`: three backbone bullets rewritten.
- Untouched, as agreed: `reports.rs`, `shared/src/report.rs`, `runner.rs`, `tui/src/main.rs`,
  `optim/muon.rs`.

**`recipe_scalars()` (Main's shared-base contract).** `Vec<(String, f64)>`, `5·layers` entries,
ONE host transfer: the three banks are `Tensor::cat`-ed on device and copied once, no per-scalar
`.item()`, no per-step call site. All values are **raw parameters — no sigmoid to invert**, per
the reference's parameterization (item 5). Labels are per sub-block rather than per layer, because
the reference's lambdas are per sub-block: `residual lambda L<l> attn`, `residual lambda L<l> ffn`,
`post lambda L<l> attn`, `post lambda L<l> ffn`, `x0 lambda L<l>`. That is a superset of the
labels Main specified (same prefixes, one extra `attn`/`ffn` qualifier on the residual and post
entries); the integrator's base iterates whatever comes back, so no reports.rs change is implied.

## OBJECTIVE / FORMAT bump I need (do not apply — combined bump is the integrator's)

Every weight tensor and the forward numerics change, so old checkpoints must be rejected:
- `objective: "causal_patch_market_neutral_nll_v2"` → **`_v3`** (backbone numerics: gainless
  RMSNorm, QK-norm, ReLU², learned residual lambdas).
- `format`: the live stamp is `runner.rs:119`
  `"causal-patch-ohlc-universe-v6-head-channel-major-folded-mup"` (NOT the `v5` that
  `docs/timexer_segment.md:23` still claims — that doc line is pre-existing drift, unrelated to
  this change). It needs **`v7`** plus a recipe token, e.g.
  `"causal-patch-ohlc-universe-v7-head-channel-major-folded-mup-residual-recipe"`: the parameter
  set changed (34 norm tensors and 32 block biases gone, three lambda banks added), so old state
  dicts are not loadable. `runner.rs:2261`'s stale-stamp rejection list should gain the v6 stamp.
Both strings live in `runner.rs:119-120`; I changed neither, and `UNetSkips` needs a FORMAT bump
too (varstore gains `skip_weights`), so this is one combined bump for the integrator.

## Expected conflicts with the siblings

With **`UNetSkips`** (owns the `backbone` loop + a skip-weight vector):
- `backbone`'s loop body is the one real collision. I rewrote `backbone` wholesale: it now
  computes `x0 = rms_norm(patch(...))` before the loop, casts and `unbind`s the three lambda banks
  once, builds a `BlockLambdas` per index inside the loop, and calls
  `block.forward(&state, &x0, &lambdas, rotation, train)`. Their `x += w_i * pop()` inserts
  cleanly *before* the `block.forward` call in that same loop; the resolution is "keep my loop
  header and lambda setup, insert their push/pop lines". They confirmed by IRC that they touch
  only `CausalPatchModel`'s fields, `new()`, and the loop — no block internals.
- `CausalPatchModel`'s field list and `new()`'s struct literal: both of us add fields (mine three
  lambda banks, theirs a skip bank). Textually adjacent, semantically independent.
- `recipe_scalars`: they expose `skip_gates()`. The integrator's single accessor should call both.
- `step_cost` per-layer units: mine is now `22·width + 3·ffn` + one `width` for the embedding
  norm. Each of their skip `addcmul`s adds one more `width` unit for half the layers.
- `compute.rs`'s lr-scale call: I adopted their argument that the call site should assert, so mine
  is now `assert_eq!(set_named_lr_scale(&["lambdas.resid", "lambdas.x0"], 5.0), 2, …)` — a silent
  zero-match would train differently-tuned lambdas, and a rename should fail at construction.
  Merged it is one call over three names asserting **3**, with their
  `NANOGPT_SCALAR_LR_MULTIPLIER = 5.0` replacing my literal. Because that assert runs in
  production construction, the synthetic
  `polar_express_routes_only_hidden_matrices_and_keeps_readout_on_adamw` fixture must register
  every matched var: I replaced its now-nonexistent `block_0.first.bias` and `block_0.norm_0.weight`
  entries (the recipe leaves blocks bias-free and gainless) with the three lambda banks, which are
  the 1-D "must not reach NorMuon" negative control with a real producer, and moved its AdamW count
  7 → **8**; adding their `skip_weights` makes it **9**, NorMuon stays 4.
- Merged `compute.rs` form, agreed with `UNetSkips` and pinned identically in their report so the
  integrator has one authoritative copy: no-decay list = `["norm", ".bias", "skip_weights",
  "lambda"]` (`"norm"`/`".bias"` still earn their place — the head and the patch/covariate
  projections keep biases even though the blocks no longer do); one
  `assert_eq!(set_named_lr_scale(&["lambdas.resid", "lambdas.x0", "skip_weights"],
  NANOGPT_SCALAR_LR_MULTIPLIER), 3, …)`; synthetic fixture = 4 `block_0` matrices +
  `lambdas.resid[4]` + `lambdas.post[4]` + `lambdas.x0[2]` + `skip_weights[4]` + `patch.weight` +
  `covariates.weight` + `head.hidden.weight` + `head.output.weight` + `head.output.bias` →
  NorMuon 4 / AdamW 9. My real-model routing table also needs their
  `skip_weights → (5.0, (0.9, 0.95), 0.0)` row, and their synthetic test's
  `adamw_group_settings("skip_weights")` assertion carries over unchanged.

With **`ValueResidual`** (owns the attention V path):
- `Block::forward` is the collision. Their V operand is `packed[1]`; I did not move that line —
  it is still `packed[1].reshape([batch, length, heads, head_dim]).transpose(1, 2)` inside the
  SDPA call. But `Block::forward`'s **signature** changed (`input, x0, lambdas, rotation, train`)
  and they need to add `v1` plus a V return, so the two signature edits must be merged by hand
  into one: `fn forward(&self, input, x0, v1: Option<&Tensor>, lambdas, rotation, train) -> (Tensor, Tensor)`.
- My QK-norm inserts a statement between the `split_with_sizes` and `self.rotate`; their V change
  is after the split too. Same hunk, different lines.
- `kernel_classes`' "causal SDPA" class: they will want a value-residual class; mine renamed
  three neighbours in the same `vec![]`.
- Also worth flagging to them: with QK-norm, `q`/`k` are unit-RMS per head but `V` is not
  normalized (reference behaviour), so their `(1-λ)·V + λ·V₁` mixes two unnormalized tensors —
  unchanged semantics, just a note that QK-norm does not scale-normalize what they mix.

With **`KernelTraffic`** (agreed by IRC): they own `step_cost`'s byte accounting,
`rotation_tiles`, `Block::rotate`, `Block::forward`'s qkv/rotary/SDPA section, `tokens`,
`KernelClass`, `kernel_classes`, and their two bit-identity tests; I own the norm, the activation
and the residual/lambda lines. They asked for exactly three updates and I made all three, plus
the ones the no-bias decision forced:
`"LayerNorm" → "RMSNorm"` with `cast(2·width)` and the `mean` term dropped (`fp32(tokens)`, rstd
only); `"GELU" → "ReLU^2"` with `forward_flops` `8·T·ffn → 2·T·ffn` **and** `forward_bytes`
`2·hidden → 4·hidden` (two kernels, not one — this is the honest correction they should know
about); `step_cost`'s per-layer width units. Additionally: `cast(…)` terms lost their `+ width`
bias elements in the four projection classes, the two output-projection classes now call
`scaled_linear` and charge one bf16 weight read+write for the folded lambda, `"residual add"` is
now `"residual addcmul"` (charged 3× per layer), and their
`the_packed_rotation_and_split_match_the_per_tensor_reference_bit_for_bit` test was updated to
normalize q and k per head before rotating in its reference path, to randomize the two zero-init
output weights (otherwise the feedforward half of the comparison is `0 == 0`) and to use the new
`Block::forward` signature. `benchmark.rs:690-698`'s repeat table follows.

Their assignment closed after mine, so two provenance facts they confirmed by IRC are recorded
here for the integrator, since their report's table is explicitly the **pre-recipe baseline**:
- `benchmark.rs:767` (a second name-matched attribution parts sum) and `model.rs:840` (a `twice`
  closure) exist only in the **main tree** — they post-date my rsync snapshot and were their fix
  for a bug where the composed-layer byte total double-counted the norm but not the residual add.
  My `charged(name) -> (bytes, flops)` table supersedes both. **Take mine and delete theirs**,
  not a merge: one lookup instead of two name-matched sites plus a closure, with `norm × 2` and
  `addcmul × 3`. Drift is self-reporting — the printed attribution error moves off −2.2%.
- `internal_fused_rms_norm` verified differentiable and bit-identical to `F.rms_norm`; `.1` is the
  fp32 `[tokens, 1]` rstd, the norm's only retained fp32 tensor (LayerNorm retained mean + rstd).
- Their measured framing of the RMSNorm win, which is better than my byte-delta framing: the
  19.8 ms LayerNorm term is 17.7 ms of **backward at 17% of the bandwidth roof** — the worst
  roofline fraction in the backbone. What gainless RMSNorm removes is the `dgamma`/`dbeta`
  reductions and the mean-gradient path, so that term should fall visibly even though the byte
  delta is 0.013%. Do not restate the 0.39 MB/norm as a bandwidth win.
- Fused ReLU² is the top fusion target, with an independent evidence trail: `dir(torch.ops.aten)`
  holds only `relu`, `relu6`, `relu_`, `glu`, and `_addmm_activation` is bit-identical for ReLU
  but has **no derivative** (measured: `derivative for aten::_addmm_activation is not
  implemented`), so no exact single-kernel ReLU² exists today at any width. Against their measured
  GELU baseline (27.2 ms/step, 58% of the roof forward, 30% backward) the doubled traffic puts it
  ahead of the rotary's ~40 ms, and it is a two-kernel elementwise chain rather than a nine-kernel
  one, so one Triton kernel captures nearly all of it.

## Verification

`FA4_VENV=…/.venv-fa4 ./torch-env.sh cargo check -p trading_bot_0 --tests` → **zero errors**.
`… cargo test -p trading_bot_0 timexer_segment` → **43 passed, 0 failed**.

New CPU tests:
- `model::tests::the_stack_is_the_identity_on_the_residual_stream_at_init` — 8-layer model,
  asserts `backbone` at init equals `rms_norm(rms_norm(patch(tokens)))`: cosine within 1e-4 of 1
  and max relative deviation under `2·(2·layers)·2⁻⁹` (the residual stream is bf16, so sixteen
  multiplications by a bf16 √1.1 round it; measured `1 − cos = 1.05e-5`). Then sets
  `resid_lambdas = 1` and asserts **bit-exact** identity, which isolates that rounding as the only
  deviation. Then perturbs `block_0.output.ws` and asserts both that the output moves by >1e-3
  relative and that the direction moves >10× the identity's own noise — so neither bound is
  vacuous. Finally reads all 40 scalars back through `recipe_scalars()` and pins them to
  √1.1 / 1.0 / 0.0.
- `model::tests::rms_norm_matches_a_scalar_reference_and_keeps_the_row_mean` — scalar `f64`
  reference per element on rows with a deliberately nonzero mean, plus an assertion that the row
  mean is *not* centred (the LayerNorm-with-affine-off mistake) and that the norm registers no
  variables.
- `model::tests::relu_squared_matches_its_definition_and_is_not_gelu` — scalar reference over
  `[-4, 3]`, plus separation from both GELU and `relu(x²)`.
- `model::tests::qk_norm_is_per_head_before_the_rotation_and_leaves_the_value_path_alone` —
  hand-computed 2-head, `head_dim` 2 case: head 0 = (3,4) → RMS √12.5, head 1 = (1,0) → RMS √0.5,
  etc., each element checked against `value / √(mean(pair²)+eps)`; asserts that normalizing over
  the whole packed row instead of per head gives a *different* answer (so the reshape is
  load-bearing); asserts `rotate(norm(x)) == norm(rotate(x))` to 1e-5, which is the statement that
  the rotation is norm-preserving per head, and that the rotary inputs are unit-RMS — the actual
  reason the reference's order is norm→rotary.
- `compute::tests::every_causal_patch_parameter_lands_in_its_intended_optimizer_group` — builds a
  real 2-layer model, asserts the NorMuon list is *exactly* the four `block_i.<proj>.weight`
  matrices per block (no norm gains, no biases, no scalars), then asserts the resolved AdamW
  `(lr scale, betas, wd multiplier)` for all nine remaining parameters including
  `lambdas.resid → (5×, (0.9,0.95), 0)`, `lambdas.post → (1×, …, 0)`,
  `lambdas.x0 → (5×, …, 0)`, and that every `lambdas.*` tensor is 1-D.

Two pre-existing tests needed changes the new init forces, both in the same direction (the network
is the identity at step 0, so a liveness assertion needs a live network):
`fused_loss_matches_the_reference_decode_and_nll_including_gradients` asserts every parameter
receives a gradient — at init the QKV/FFN-up matrices and the post-lambdas legitimately receive
none, so its `no_grad` setup now also randomizes each block's `output.ws` and `second.ws`;
`the_packed_rotation_and_split_match_the_per_tensor_reference_bit_for_bit` as described above.

No GPU work and no mlq submission was performed.
