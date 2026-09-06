# CausalPatch U-net skips (change B)

Worktree `worktrees/unet-skips`, branch `unet-skips-20260906`, base `85374302`.

## What I copied into the worktree

Isolated diff for the integrator: `research/worker_reports/causalpatch_unet_skips.patch` — my hunks
only, against the main tree as it stood when I snapshotted it. The raw `diff` also picked up
`KernelTraffic`'s `kernel_classes` rewrite (`model.rs:833`) and `GraphCapture`'s `CaptureAudit`
rewrite (`compute.rs:900,930,956`) as reverse hunks, because both are still editing the main tree;
those three hunks are stripped, so the patch is mine and nothing else. Worktree commit `d7c2113a`
contains the snapshot and my change together, so use the patch (or the three named files) rather
than that commit's diff.

`git worktree add -b unet-skips-20260906 worktrees/unet-skips HEAD`, then rsync'd every modified and
untracked path from the main working tree (`git status --short --untracked-files=all`, 45 entries,
the one `D` entry excluded and then deleted in the worktree): all 24 modified tracked files
(`benchmarks/timexer_universe_campaign.py`, `docs/timexer_segment.md`, `report_cli/src/main.rs`,
`shared/src/report.rs`, `trading_bots/src/torch/cuda/{graph,mod}.rs`,
`trading_bots/src/torch/dataset.rs`, `trading_bots/src/torch/model/{mod,rope}.rs`,
`trading_bots/src/torch/optim/muon.rs`, all eight `trading_bots/src/torch/timexer_segment/*.rs`,
`tui/src/{main,report_renderer}.rs`, the three `vendor/torch-sys-0.25.0` files), the deletion of
`trading_bots/src/torch/timexer_segment/geometry.rs`, and the untracked `research/papers/`,
`research/sundial_lens_20260905.md`, `research/worker_reports/`,
`trading_bots/src/torch/timexer_segment/features.rs`. `git status --short` in the worktree is
byte-identical to the main tree's. Also symlinked `.venv-fa4 -> ../../.venv-fa4` so `torch-env.sh`
works without a second 3 GB PyTorch install (worktree-local, not committed).

## Reference grounding, and the divergence between record 11 and current

Two reference forms, and they disagree; I implemented the topology of the first with the
parameterization of the second, per the coordinator's decision.

**Record 11 — the record that introduced the U (topology, raw init 1.0, Adam routing).**
`records/track_1_short/2024-11-10_UNetDoubleLr/README.md:8` ("Added U-net-like skip connections
into the transformer"), code
`records/track_1_short/2024-11-10_UNetDoubleLr/c87bb826-797b-4f37-98c7-d3a5dad2de74.txt`:

- `:240-244` — `# U-net design by @brendanh0gan`; `encoder_layers = n_layer // 2`,
  `decoder_layers = n_layer - encoder_layers`, `skip_weights = nn.Parameter(torch.ones(decoder_layers))`.
  **Raw** weights, **init 1.0**, one per decoder layer.
- `:257-263` — `skip_connections = []`; encoder loop appends each block output.
- `:266-270` — decoder loop: `skip_connection = skip_connections.pop()`,
  `weighted_skip = self.skip_weights[i] * skip_connection`,
  `x, v1 = self.transformer.h[self.encoder_layers + i](x + weighted_skip, v1, x0)`. The add happens
  **before** the decoder block's own compute, and `pop()` is what makes the pairing LIFO:
  at eight layers, 3->4, 2->5, 1->6, 0->7. Gate `k` (0-based within the decoder) is
  `skip_weights[k]`, i.e. `skip_weights[i - encoder_layers]`.
- `:432-434` — `scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]`,
  `optimizer4 = torch.optim.Adam(scalar_params, lr=0.04, betas=(0.9, 0.95))`. Skip weights go to
  **Adam**, never to Muon.

Same form survived into `records/track_1_short/2024-12-08_UNetValueEmbedsTweaks/0069607b-…txt:257`
(`torch.ones(num_decoder_layers)`), `:311` (`x = x + self.skip_weights[i] * skip_connections.pop()`),
`:476` (Adam), and `records/track_1_short/2025-05-09_SkipMLPBlocks/07e7ae76-…txt:689,719,727`
(moved into the packed `scalars` vector, still `torch.ones`).

**The sigmoid re-parameterization with init -1.5.**
`records/track_1_short/2025-11-18_RefineSkip/00f4e1e6-0044-4a08-b88a-3b7ec0624081.txt:953-954` —
`-1.5 * torch.ones(num_layers),  # skip_weights -> σ(-1.5) ≈ 0.18`, sliced at `:991`
(`skip_weights = self.scalars[:(len(self.blocks) // 2)]`) and applied at `:1021-1022` —
`gate = torch.sigmoid(skip_weights[i - n])  # in (0, 1)` then `x = x + gate * skip_connections[0]`.
The stored parameter is a **logit**; the gate the residual stream sees is `σ(logit)`.
`self.scalars.lr_mul = 5.0` at `:973`.

**Current `train_gpt.py` (preferred where it differs).** `:1346` —
`-1.5 * torch.ones(1),  # skip_lambda -> σ(-1.5) ≈ 0.18`; `:1508` unbind;
`:1591-1595` — `# process attn. skip on layer 6 @YouJiacheng`, `skip_gate_out =
torch.sigmoid(skip_lambda) * post_skip_gate`, `x = x + skip_gate_out * cache[3]`, before layer 6's
compute; `cache_layers = [3, 7]` at `:1249`. Routing `:2030` — `"scalars": {"optim": "adam",
"adam_betas": [0.9, 0.99], "lr_mul": 5.0, "wd_mul": 0.0}`. Activation-dtype cast of residual
scalars `:1509-1512` (`.bfloat16()`).

**The divergence, stated plainly.** The current reference keeps **one** skip (3->6 in its
eleven-layer-with-a-null-layer indexing), not four. `2025-11-18_RefineSkip/README.md` is explicit
about why: "Replace the current skip U architecture of (1->10, 2->9, 3->8, 4->7, 5->6) with
(4->7). I picked 4 because it had a long attention window … I picked 7 because layer 7 does not
have an attention module, so its MLP may have free capacity … If I remove (4->7) performance drops
by roughly 0.0025 loss, which is substantially more costly than the compute time of one skip
connection." So the pruning from five/six skips to one was a **wall-clock** trade on a 135-second
run, on a twelve-layer model, and the single retained pair was chosen for properties our model does
not have (no per-layer attention windows, no attention-free layer). It is not evidence that four
skips hurt loss. `train_gpt_medium.py:1091,1122-1124,1174-1177` keeps **three**
(`skip_in=[2,4,6]`, `skip_out=[9,10,11]` over twelve layers, `σ(skip_lambdas[j]) * 2 *
σ(skip_gate(x0))`), which is the same U at a larger budget — so the number of skips tracks depth,
not a fixed answer. Also note medium multiplies by an extra `2 * σ(gate(x0[..., :16]))`, a
data-dependent gate initialised near 1.0; I did not port it (it needs a `Linear(16, 1)` on the
residual stream and belongs with change A's x0 threading if anyone wants it).

**Implemented:** record-11 four-skip topology (3->4, 2->5, 1->6, 0->7), current parameterization —
**logit, sigmoid-transformed, init -1.5 -> 0.18242552380635635**, one logit per decoder layer.
Record 11's `ones` init would put the residual stream entering layer 4 at `x + 1.0 * x_3` with
`x_3 ≈ x` at init, i.e. a factor-2 jump, before a single gradient step; `σ` also confines the gate
to `(0, 1)` for any logit, so no schedule can blow the stream up mid-run.

## Exact parameterization and init

| | |
|---|---|
| Stored parameter | `skip_weights`, root varstore path, shape `[layers/2]` = `[4]`, fp32, `requires_grad` |
| Stored value at init | `-1.5` (`SKIP_LOGIT_INIT`, `model.rs:37`) — a **logit**, not a weight |
| Gate used in the forward | `σ(logit)`, cast to bf16 before it touches an activation; `0.18242552380635635` at init |
| Fold | `state = state.addcmul(&skip, &gate)` = `state + gate * skip`, **before** the decoder block's compute |
| Pairing | `3->4, 2->5, 1->6, 0->7`; gate `k` (`skip_weights[k]`) drives pair `k`, matching the reference's `skip_weights[i - n]` |
| Optimizer | AdamW (1-D and outside the `block_*` NorMuon allowlist), lr ×5.0, betas (0.9, 0.95), **wd_mul 0** |
| Reported as | `skip weight 3->4`, `2->5`, `1->6`, `0->7`, post-sigmoid, via `CausalPatchModel::recipe_scalars()` |

`lr_mul 5.0` comes from `train_gpt.py:2030`; the reference's scalar betas there are (0.9, 0.99),
ours stay the shared (0.9, 0.95) because `MuonConfig::adamw_beta_overrides` is a per-name list I
chose not to populate for a single parameter — flagging it as a tuning knob, not a defect.
`wd_mul 0` is load-bearing rather than cosmetic: decaying a *logit* toward 0 pulls the gate toward
`σ(0) = 0.5`, which is not a neutral prior, so `skip_weights` joins `norm`/`.bias` in
`adamw_no_weight_decay_name_substrings`.

## Memory and traffic accounting (batch 256, 375 tokens/row, `d_model` 512)

`rows·origins = 256·375 = 96 000` tokens; one `[96 000, 512]` bf16 activation is
`96 000 · 512 · 2 = 98 304 000 B = 93.75 MiB`.

**The stack costs zero bytes.** The four tensors the encoder pushes are the *same* tensors that are
already retained for backward — each encoder layer's output is the next layer's input and is saved
by that layer's pre-norm. `skips.push(state.shallow_clone())` stores a refcount, not a copy, so the
"up to 4 tensors of `[rows·origins, 512]` bf16 alive for the whole forward" the brief asks me to
price is **already paid** in the baseline. Nominal size, for the record:
`4 · 98 304 000 = 393 216 000 B = 375.0 MiB = 0.393 GB`.

**What is genuinely new** is one extra activation per decoder layer: the fold's output, which the
decoder block's pre-norm then saves. I used `addcmul` rather than `gate * skip` followed by an add
precisely to keep this at one tensor per pair instead of two: `addcmul`'s backward saves only the
two factors (both already retained) and never the product, so no separate `gate * skip` buffer
exists at all.

- **Peak memory, upper bound: +4 × 93.75 MiB = +375 MiB (0.366 GiB).** That is `+1.14%` of the
  5090's 32 GiB, moving the previous run's 89% VRAM (28.48 GiB) to **90.1%**. Affordable.
- **Peak memory, actual expectation: +1 × 93.75 MiB (+0.29%, 89.3%).** For decoder layers 5, 6 and
  7 the *pre*-fold state is the previous block's output and, once the fold has consumed it,
  nothing saves it (`AddcmulBackward` saves the two factors, not `self`; the residual `add`
  backward saves nothing), so it is freed and the fold's output replaces it one-for-one. Only
  layer 4's input is genuinely double-retained, because it is also skip source 3. I am reporting
  the +375 MiB upper bound as the number to decide on, since it does not depend on this
  allocator/autograd reasoning.
- **Traffic: +2.359 GB/step** (`(layers/2) · 2 B · tokens · width · 3 · 2` — the model's own
  convention of write-once-read-once times three for forward plus the two backward passes;
  `model.rs:192-196`). Against the brief's `~144 GB/step` that is **+1.64%** (144 -> 146.4);
  against `ModelConfig::step_cost`'s own total at these shapes, 155.20 GB -> **157.56 GB, +1.52%**.
  At the run's measured effective bandwidth (144 GB / 302 ms ≈ 477 GB/s, the 27% roofline
  utilisation quoted) that is **≈ +4.9 ms/step, +1.6%**. Arithmetic added is elementwise and
  invisible against 16.98 TFLOP.

**Single-skip alternative, if the coordinator wants the cheaper version.** A lone 3->6-style pair
is exactly one quarter of the above: **+0.590 GB/step (+0.41% of 144 GB, +0.38% of 155.20 GB,
≈ +1.2 ms)** and **+93.75 MiB peak upper bound (+0.29%, 89.3% VRAM)**. My recommendation is to run
the four-skip version as specified: at +1.6% step time and +1.1% VRAM worst case, the four-skip U
is not where this model's cost is, and the four reported gates are a *measurement* the single skip
cannot give — if three of the four collapse toward 0 and one stays up, that identifies the pair
worth keeping and reduces the next iteration to exactly the RefineSkip pruning, with our own
evidence instead of a twelve-layer LM's. Dropping to one skip up front spends the same run and
learns a quarter as much. Reducing to one pair afterwards is a two-line change
(`ModelConfig::skip_pairs`).

## Changes

`trading_bots/src/torch/timexer_segment/model.rs`

- `:28-37` `SKIP_LOGIT_INIT = -1.5` with the reference citations.
- `:117-120` `validate`: layer count must be even (the U has no meaning otherwise).
- `:136-145` `ModelConfig::skip_pairs()` — the `(encoder source, decoder destination)` iterator,
  `3->4, 2->5, 1->6, 0->7`; single source of truth for the pairing and for the report names.
- `:192-196` `step_cost`: `+ (layers/2) · bf16(tokens · width)`, with the reasoning above.
- `:382-431` `unet_stack` — the loop, parameterized over the per-index compute so it can be driven
  by distinguishable stub layers in a test.
- `:529-533` `CausalPatchModel::skip_weights` field; `:605-609` registration
  (`path.var("skip_weights", &[layers/2], Init::Const(SKIP_LOGIT_INIT))`).
- `:650-677` `recipe_scalars()` / `recipe_scalar_parts()` — per the coordinator's decision, one
  method returning `Vec<(String, f64)>` of **post-sigmoid** values, names `skip weight <src>-><dst>`.
  Values are concatenated on device and copied to the host **once** (`Tensor::cat` then one
  `Vec<f64>::try_from`); no per-scalar `.item()`, and the doc comment says per report interval,
  never per step. `recipe_scalar_parts` is the seam A and C append to — one `push` each, still one
  transfer.
- `:761-774` `backbone`: `sigmoid().to_kind(BFloat16).unbind(0)` then `unet_stack`. The bf16 cast is
  mandatory here, not stylistic: per `causalpatch_recipe_spec.md` §0.2 a dimensioned fp32 tensor
  times a bf16 activation promotes the *result* to fp32. `unbind` rather than four `select`s
  because `select`'s backward is `zeros_like` plus a copy.
- `:1880` `narrow_config()` test helper: `layers: 1` -> `2` (an odd count no longer validates).

`trading_bots/src/torch/timexer_segment/compute.rs` (GraphCapture confirmed by IRC that it is not
touching `polar_express`, the no-weight-decay list, the AdamW branch, `MuonConfig`'s fields, or the
routing test, and that these additive edits are mine to make)

- `:29-31` `NANOGPT_SCALAR_LR_MULTIPLIER = 5.0` (`train_gpt.py:2030`).
- `:109-116` `skip_weights` added to `adamw_no_weight_decay_name_substrings`.
- `:139-146` `optimizer.set_named_lr_scale(&["skip_weights"], 5.0)` with `assert_eq!(…, 1)` so a
  rename cannot silently drop the multiplier. This uses the **existing** per-parameter `lr_scales`
  machinery (`muon.rs:2173-2186`, read at `:1533`); no new `MuonConfig` field, so GraphCapture's
  constraint that a multiplier stay a per-parameter kernel argument is met by construction.
- The NorMuon assert at `:147-166` is unchanged and still passes: `skip_weights` is 1-D, root-level
  and outside the `block_*` allowlist.

`docs/timexer_segment.md:17` — one added bullet in `## Model`.

**Nothing touched** in `reports.rs`, `shared/src/report.rs`, `tui/src/main.rs`, `runner.rs`,
`muon.rs`, `benchmark.rs`, or any `Block` internals.

## Tests

All in-worktree, CPU, `./torch-env.sh cargo test -p trading_bot_0 timexer_segment`: **42 passed,
0 failed**. `./torch-env.sh cargo check -p trading_bot_0 --tests`: **zero errors**.

- `model::tests::skip_gates_start_at_the_reference_sigmoid_of_minus_three_halves` — the *stored*
  varstore parameter is `[4]`, `requires_grad`, and exactly `-1.5` (asserted as a logit, with the
  failure message saying so, so a future "simplification" to storing 0.18 directly fails loudly);
  `recipe_scalars()` returns exactly the four expected names in pairing order and each value is
  `σ(-1.5)` to 1e-6.
- `model::tests::closed_skip_gates_leave_the_backbone_bit_identical_to_a_plain_stack` — logits
  forced to `-∞` (the only way to close a sigmoid gate exactly), then `backbone` is compared with
  `equal()` against the plain `patch -> blocks -> final_norm` stack written out by hand, at 8
  layers on synthetic rows, with a guard that the output is not all zero.
- `model::tests::the_skip_stack_pairs_the_deepest_encoder_layer_with_the_shallowest_decoder_layer` —
  drives the production `unet_stack` with stub layers that add a one-hot per-layer tag and gates
  `(2, 3, 5, 7)` in fp64, so each output channel is exactly the coefficient with which that layer's
  output reached the end. Compared with `equal()` against a straight-line, loop-free expectation
  written one line per skip (`3->4`, `2->5`, `1->6`, `0->7`), **plus a negative control**: the
  queue order a `remove(0)` would produce must *not* compare equal, which is what proves the
  positive assertion discriminates (distinct gates are required — with all gates 1 the two orders
  coincide by commutativity, 281 either way).
- `model::tests::an_odd_layer_count_cannot_split_into_encoder_and_decoder_halves`.
- `compute::tests::polar_express_routes_only_hidden_matrices_and_keeps_readout_on_adamw` — extended
  with a root-level `skip_weights` var; AdamW group count 7 -> 8, NorMuon count still 4, and
  `adamw_group_settings("skip_weights")` asserted to be `(lr_scale 5.0, betas (0.9, 0.95),
  wd_multiplier 0.0)`. `polar_express` now panics if `skip_weights` is absent, so this coverage
  cannot be dropped by accident.

No GPU work, no mlq submissions.

## OBJECTIVE / FORMAT bump I need (coordinator applies one combined bump)

The forward numerics change and the parameter set grows, so a checkpoint from
`timexer-market-neutral-20260906` cannot be loaded into this build and must be rejected rather than
silently mis-keyed.

- `runner.rs:119` `FORMAT`: needs a bump — the varstore gains `skip_weights`, which the manifest's
  architecture authentication should refuse to match against an older checkpoint. Suggested token to
  fold into the combined bump: `unet-skips` (e.g.
  `causal-patch-ohlc-universe-v7-head-channel-major-folded-mup-unet-skips`, with A's and C's tokens).
- `runner.rs:120` `OBJECTIVE` (`causal_patch_market_neutral_nll_v2`): **no bump needed from me.**
  The loss, the targets, the mask and the decoder are untouched; this is an architecture change, not
  an objective change. If A or C also needs none, leave `OBJECTIVE` alone.
- `compute.rs:48-52` `OptimizerKind::recipe()`: needs `;skip-weights-AdamW-lr_mul=5-wd=0` appended.
  I deliberately did not edit that string — all three of us would append to the same line. One
  combined edit, please.
- `docs/timexer_segment.md:23` still says `format: "causal-patch-ohlc-universe-v5"` while the code
  says `…-v6-head-channel-major-folded-mup`. Pre-existing staleness, worth fixing in the same pass.

## Expected conflicts with A (`ResidualRecipe`) and C (`ValueResidual`)

Coordination on record: agreed with `ValueResidual` and confirmed by `Main` that none of us touches
`reports.rs`, `shared/src/report.rs`, `tui/src/main.rs` or `runner.rs` for metrics; each exposes
`recipe_scalars()`-shaped accessors and the integrator writes the single
`timexer_segment_recipe_scalars` base. `GraphCapture` confirmed `compute.rs::polar_express` and the
routing test are mine.

1. **`model.rs` `backbone()` — the real conflict, guaranteed. Resolved with A by IRC.** A rewrote
   `backbone` wholesale (`worktrees/residual-recipe`): `x0 = rms_norm(dropout(linear(tokens,
   patch)))` before the loop, the three lambda banks cast to the activation kind once and unbound,
   `let mut state = x0.shallow_clone();`, then
   `for (index, block) in self.blocks.iter().enumerate() { let lambdas = BlockLambdas {…};
   state = block.forward(&state, &x0, &lambdas, (&self.rotation.0, &self.rotation.1), train); }`,
   then `rms_norm(&state)`. Agreed resolution: keep A's `x0`/lambda setup and its per-index
   `BlockLambdas` construction, and put that construction plus the `block.forward` call inside my
   `unet_stack` closure —
   `unet_stack(x0.shallow_clone(), self.blocks.len(), &gates, |index, state| { let lambdas = …;
   self.blocks[index].forward(state, &x0, &lambdas, rotation, train) })`. My fold then lands
   immediately before `block.forward` and my push immediately after, which is exactly where A
   describes them. `unet_stack` is agnostic to the block signature — that is why the compute is a
   closure parameter. Lines: my `:761-774` against A's rewrite of the same range.
   A's per-sub-block `√1.1` also confirms the init argument above: the stream entering layer `i`
   is `1.1^i · x0`, so a raw skip weight of 1.0 would add a same-magnitude copy of `x_3` to `x`.
   A's `step_cost` per-layer term becomes `22·width + 3·ffn` plus one width for the embedding
   norm; my `+ (layers/2) · bf16(tokens · width)` stacks on top unchanged, and A's independently
   derived "1 width unit = 0.59 GB/step at batch 256" matches my single-skip figure of 0.590 GB
   exactly.
2. **`model.rs` `recipe_scalars` / `recipe_scalar_parts` (`:650-677`) — additive conflict, all
   three.** Each change appends its names and pushes one device tensor in `recipe_scalar_parts`.
   Take the union of the `names.extend(...)` / `values.push(...)` pairs; `recipe_scalars` itself
   needs no edit and the single host transfer is preserved for any number of contributors.
3. **`model.rs` `step_cost` (`:190-196`) — adjacent, not overlapping.** My `bytes +=` line sits
   immediately after the per-layer term A will change (RMSNorm drops the mean, QK-norm adds a term,
   ReLU² changes the FFN count) and C will change (value-residual mix). Three separate `bytes +=`
   statements; keep all three.
4. **`model.rs` `Block` internals — no conflict.** I did not touch `Block`, its fields, its norms,
   its activation, `rotate`, or `kernel_classes`. If A drops `Block::forward`'s bias/norm fields or
   renames them, my diff is unaffected.
5. **`model.rs` test module — additive.** My four tests are appended at the end and use only
   `small_config`/`synthetic`. One shared edit: `narrow_config()`'s `layers: 1 -> 2` at `:1880`,
   needed because odd layer counts no longer validate; if A also edits that helper, keep the even
   count.
6. **`compute.rs` — settled with A, exact merged form below.** A and C also need their scalars in
   AdamW's no-decay group and at lr ×5. Union the `adamw_no_weight_decay_name_substrings` entries
   (`"norm"`, `".bias"`, `"skip_weights"`, `"lambda"`) and make it **one**
   `assert_eq!(optimizer.set_named_lr_scale(&["lambdas.resid", "lambdas.x0", "skip_weights"],
   NANOGPT_SCALAR_LR_MULTIPLIER), 3, …)` — A adopted the asserting form after finding their own
   call discarded its return value, which is the silent-drop-on-rename failure the assert exists
   for. Because that assert runs in **production** construction, the synthetic fixture
   `polar_express_routes_only_hidden_matrices_and_keeps_readout_on_adamw` must register every
   matched var or it fires there. A replaced its now-nonexistent `block_0.first.bias` and
   `block_0.norm_0.weight` entries (blocks are bias-free and gainless after the recipe) with
   `lambdas.resid [4]`, `lambdas.post [4]`, `lambdas.x0 [2]`, which keeps the 1-D
   "must not reach NorMuon" negative control with a real producer. **Merged fixture list:** the 4
   `block_0.*` matrices, the 3 lambda banks, `skip_weights`, `patch.weight`, `covariates.weight`,
   `head.hidden.weight`, `head.output.weight`, `head.output.bias` — **NorMuon 4, AdamW 9**
   (my extension alone asserted 8, A's alone 8). My
   `adamw_group_settings("skip_weights") == (5.0, (0.9, 0.95), 0.0)` assertion carries over
   unchanged.
7. **`docs/timexer_segment.md`** — I added one bullet at `:17`; A and C adding their own bullets in
   `## Model` will conflict only on line numbers.

## Post-yield merge notes (settled with A by IRC, integrator to apply)

1. **`recipe_scalars` name collision — resolved, no code change needed on my side.** A implements
   the same method with `5·layers` entries, and A/C settled that the model-level `recipe_scalars`
   cats A's three model-level banks, C's per-layer tensors reshaped to `[1]`, and my
   `skip_weights.sigmoid()`, labelling from offsets. My `recipe_scalar_parts` already returns
   `(Vec<String>, Vec<Tensor>)` and my `recipe_scalars` already does exactly **one** `Tensor::cat`
   and **one** `to_kind(Double)` host copy over that vector, so folding the other two families in
   is a `names.extend` + `values.push` each and the single-transfer guarantee survives. The merged
   version must keep one cat and one copy — not three.
2. **`compute.rs` combined assert — the one thing that WILL break if merged naively.** A also
   appends to `adamw_no_weight_decay_name_substrings` (`"lambda"`) and also calls
   `set_named_lr_scale(..., 5.0)`; fold A's literal into my `NANOGPT_SCALAR_LR_MULTIPLIER`. My call
   asserts the match count is 1; the merged call over
   `["lambdas.resid", "lambdas.x0", "skip_weights"]` must assert **3**, and
   `polar_express_routes_only_hidden_matrices_and_keeps_readout_on_adamw`'s synthetic store must
   register all three vars or that assert fires there. Keep the assert — a non-asserting
   `set_named_lr_scale` silently drops the multiplier on a rename. A confirmed this was live in
   their file (return value discarded) and adopted the asserting form. Exact merged fixture and
   counts are pinned in conflict item 6 above: NorMuon 4, AdamW 9.
3. **A's `every_causal_patch_parameter_lands_in_its_intended_optimizer_group`** (real 2-layer model)
   needs one row added: `skip_weights -> (lr_scale 5.0, betas (0.9, 0.95), wd_multiplier 0.0)`. Its
   `named.len() - 4·layers` NorMuon count is unaffected (`skip_weights` is 1-D). A's test and my
   extension of the synthetic one are complementary; keep both.
4. **Merged `Block::forward` signature.** A changes it to `(input, x0, lambdas, rotation, train)`
   and C adds `first_value` plus an `Option<Tensor>` return, so my `unet_stack` closure body
   becomes `let (next, published) = self.blocks[index].forward(state, &x0, first_value, &lambdas,
   rotation, train);`. `unet_stack` needs no change — the closure returns the new state and the
   caller keeps C's published value, since the closure captures mutably (`FnMut`). My fold stays
   immediately before that call, which is where record 11 puts it.
5. **FORMAT/OBJECTIVE, corrected.** The live stamp is
   `runner.rs:119 = "causal-patch-ohlc-universe-v6-head-channel-major-folded-mup"`, so it is
   **v6 -> v7** with tokens for all landed changes; `docs/timexer_segment.md:23`'s `v5` is
   pre-existing drift. A's `OBJECTIVE` bump `_v2 -> _v3` covers the backbone numerics and mine
   needs none, so exactly one combined bump of each string.
