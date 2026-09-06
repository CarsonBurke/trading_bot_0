# CausalPatch recipe integration (A + B + C into main)

Merged `worktrees/residual-recipe` (A), `worktrees/unet-skips` (B), `worktrees/value-residual` (C)
into the main tree in that order. Every worker's snapshot predates part of the live tree, so no
file was taken wholesale except A's `model.rs` (Main's verified verdict); everything else was a
3-way merge against a reconstructed pre-B base (`causalpatch_unet_skips.patch` reverse-applied to
`worktrees/unet-skips`) with each removed line audited against the live file. That audit caught one
real silent revert: A's and C's `benchmark.rs` snapshots would have restored the old
`Python::attach`-around-everything CUDA-event timer over KernelTraffic's GIL-narrow version (whose
comment says the autograd engine aborts if entered while holding the GIL) — `benchmark.rs` was
therefore built by applying `causalpatch_residual_recipe_benchmark.patch` to the live file instead.
`compute.rs` kept the live two-phase `CaptureAudit` (GraphCapture) and took only A/B/C's additive
lines; `docs` used A's three-hunk patch plus B's and C's single bullets, so KernelTraffic's new
"Benchmark reports" section survived.

## Resolved conflicts

| Site | Resolution |
| --- | --- |
| `Block::forward` | Union signature `(input, x0, first_value: Option<&Tensor>, lambdas, rotation, train) -> (Tensor, Option<Tensor>)`; C's pairing `assert!` kept, C's `Option` return kept. |
| `Block::forward` body | Order: `split_with_sizes` → A's packed QK-RMSNorm → `rotate` → C's `lerp` on the raw, unnormalized `packed[1]` → SDPA → A's `scaled_linear`/`addcmul` residual lines; return `(ff.addcmul(state, λr), value_lambda.is_none().then(value))`. Comment now states explicitly that V takes neither QK-norm nor RoPE, so both mix operands are unnormalized. |
| `Block::new` | C's `layer: usize` parameter with A's `hidden_projection`/`zeroed_projection` bodies (no `norms`, no biases) plus C's `value_lambda` init. |
| `backbone` loop | A's `x0 = rms_norm(patch(...))` + one cast/`unbind` of the three lambda banks; B's `sigmoid().to_kind(kind).unbind(0)` gates driving `unet_stack`, with A's per-index `BlockLambdas` construction and the `block.forward` call inside B's closure; C's `first_value: Option<Tensor>` threaded through the closure and published once by layer 0. Ends in A's `rms_norm(&state)`; `final_norm`/`LayerNorm` gone. |
| `CausalPatchModel` fields / `new()` | A's three lambda banks plus B's `skip_weights`, B's `final_norm` dropped. |
| `recipe_scalars` | ONE method: B's wrapper (single device `cat`, single host transfer, plus a new `names.len() == numel` assert) over one `recipe_scalar_parts()` seam carrying A's three banks, B's `skip_weights.sigmoid()` and C's per-block `Block::recipe_scalars()`. B's duplicate copy deleted. |
| `kernel_classes` | A's `charged(name)` table supersedes KernelTraffic's `twice` closure (same intent, renamed classes, `RMSNorm` ×2 and `residual addcmul` ×3); C's `"value residual mix"` stays a separate class; composed-layer `run` uses A's lambdas with `first_value = None` and `.0`. |
| `step_cost` | All three terms kept and verified additive (below). |
| `compute.rs` | No-decay list is `["norm", ".bias", "lambda", "skip_weights"]` — one `lambda` fragment covers `lambdas.{resid,post,x0}` and every `block_*.value_lambda`, so C's separate `"value_lambda"` entry is redundant and was dropped. One `set_named_lr_scale(&["lambdas.resid", "lambdas.x0", "skip_weights"], NANOGPT_SCALAR_LR_MULTIPLIER)` asserting 3 matches; `value_lambda` deliberately stays at 1×. The `compute.rs` NorMuon assertion (2-D `block_*.weight` only) is unchanged and passes: routing test now 4 NorMuon / 10 AdamW. |
| `benchmark.rs` | Repeat table `RMSNorm => 2·layers`, `residual addcmul => 3·layers`, `value residual mix => layers - 1`; the mix is excluded from the parts-vs-composed sum because the composed layer is layer 0, the mix's source. |

## Traffic and peak memory, reconciled

`ModelConfig::step_cost(256)` measured in-tree: **199.827 GB/step**, `matmul_flops` unchanged at
`1.6985e13`. Against the three reports: 155.20 (pre-recipe) + 33.62 (A: 33.03 per-layer
`22·width + 3·ffn` + 0.59 embedding norm) + 2.359 (B: `layers/2` folds) + 8.651 (C: 88 passes over
one 93.75 MiB V) = **199.83 GB**. No discrepancy to average away — each worker's term entered the
same expression unchanged, and each reproduces its own report's figure exactly (8.650752e9 for C,
2.359296e9 for B).

Peak memory: B **+375 MiB** upper bound (+93.75 MiB expected), C **+656.25 MiB** (7 materialized
mixed V tensors; layer-0 V costs nothing). A quantified traffic only and reported no peak figure;
[INFERENCE] its one new retained per-layer tensor is the normalized `q‖k` block, 2 × 93.75 MiB per
layer = +1500 MiB upper bound at 8 layers (the FFN is unchanged in retention: GELU kept
up-projection output + activation output, `relu`/`square` keeps `relu` output + `square` output).
Merged upper bound therefore ≈ **+2.47 GiB**, which is not measured and should be confirmed by the
first probe that runs `--profile` on device.

## Contract strings (one combined bump)

- `FORMAT` = `causal-patch-ohlc-universe-v7-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres`
  (was the `v6` EvalBisect landed). `runner.rs`'s stale-stamp test now also refuses
  `causal-patch-ohlc-universe-v6-head-channel-major-folded-mup`, and the pinned literal in
  `universe_checkpoint_authenticates_objective_schema_and_weight_bytes` was updated with it.
- `OBJECTIVE` = `causal_patch_market_neutral_nll_v3` (was `_v2`).
- `OptimizerKind::PolarExpress::recipe()` = one string covering all three:
  `…patch-covariates-norm-bias-lambda-vlambda-skipw-head-AdamW;lambda-vlambda-skipw-noWD;resid-x0-lambda-skipw-lrmul=5;…`.
- `docs/timexer_segment.md:27`'s stale `v5`/`_v2` manifest line fixed to the new strings.

## New report base

`timexer_segment_recipe_scalars`, y_label `learned mixing coefficient (dimensionless)`, registered
in `TIMEXER_SEGMENT_REPORT_BASES` (which `tui::meta_chart_bases()` extends from, so the TUI
bidirectional test covers it) and documented in `docs/timexer_segment.md`. Written by
`reports::write_recipe_scalars` from a `recipe_history` the runner pushes once per report interval,
driven by `CausalPatchModel::recipe_scalars()`, which iterates whatever `recipe_scalar_parts()
returns and does ONE device `cat` and ONE host transfer. Verified end to end on an 8-layer model
(throwaway probe, since removed): **51 entries** in this order — 16 `residual lambda L0..L7
attn|ffn` = 1.0488088, 16 `post lambda L0..L7 attn|ffn` = 1, 8 `x0 lambda L0..L7` = 0, 4
`skip weight 3->4|2->5|1->6|0->7` = 0.18242553 (post-sigmoid), 7 `value lambda L1..L7` = 0.5. The
written `.report.bin` round-trips with 51 series, `IndexedLines` over the pushed steps, and the
expected title and y_label.

## Tests

`cargo check -p trading_bot_0 --tests` and `-p trading-bot-tui --tests`: zero errors.
`cargo test -p trading_bot_0 timexer_segment`: **50 passed, 0 failed** (A 43 + B 42 + C 41 with the
shared pre-existing set counted once). `cargo test -p trading-bot-tui`: **36 passed, 0 failed**.
Every named test is present and passing: `the_stack_is_the_identity_on_the_residual_stream_at_init`,
`closed_skip_gates_leave_the_backbone_bit_identical_to_a_plain_stack`,
`the_value_mix_is_exactly_the_own_value_at_zero_and_the_source_value_at_one`,
`every_causal_patch_parameter_lands_in_its_intended_optimizer_group`,
`polar_express_routes_only_hidden_matrices_and_keeps_readout_on_adamw`, the TUI base-registry
bidirectional assertion, and `zero_forecast_scores_exactly_persistence_at_every_horizon`.

Three merged interactions made a sibling's assertion vacuous or wrong; each was fixed rather than
deleted, and here is which and why:

1. `closed_skip_gates_leave_the_backbone_bit_identical_to_a_plain_stack` — with A's zero-init
   output projections every layer's output is collinear with `x0`, so a skip adds a multiple of
   `x0` to a multiple of `x0` and the gainless final RMSNorm divides the difference out: an OPEN
   gate would have compared equal too. The test now randomizes both branch projections first,
   captures the open-gate output, and asserts BOTH that closed gates are bit-identical to the same
   loop without the fold and that the open gates were not. Its hand-written plain stack was also
   rewritten to thread A's lambdas/x0 and C's value, so it stays a test of the SKIP.
2. `the_stack_is_the_identity_on_the_residual_stream_at_init` — B's `σ(-1.5) = 0.182` is a
   deliberate nonzero addition at init, so "bit-exact identity at unit residual scale" now fails by
   1.6e-2 relative, which is bf16's own ULP for the `×1.182` the final norm removes. The cosine and
   magnitude bounds still run with the gates OPEN; only the bit-exact line closes them
   (`skip_weights = -∞`), which is exactly A's invariant with B's contribution isolated.
3. `the_value_mix_is_exactly_the_own_value_at_zero_and_the_source_value_at_one` and
   `every_decoder_layer_attends_with_the_source_layers_value_at_lambda_one` — C's inline reference
   was pre-recipe block math (`norms[i]`, GELU, bare adds) and its end-to-end model was 3 layers
   with a biased QKV. Reference rewritten to the merged math (`rms_norm`, packed QK-norm, `relu`²,
   `scaled_linear`, `addcmul`); layer count 3 → 4 (B's validate rejects odd counts); the bias zeroing
   replaced by an assert that the projections ARE bias-free; both tests now randomize the zero-init
   projections and assert up front that the block output depends on V at all, which the zero-init
   recipe would otherwise have made trivially true.
Scalar-count assertions in three tests were scoped by name prefix, since `recipe_scalars()` is now
one accessor for all three families.

## Commits

Committed with `git add -A` in four concern-separated commits (the module arrived as one
uncommitted snapshot, so the split is by subsystem; no intermediate commit is independently
buildable):

- `9a905eef` feat(torch): causal-patch forecaster with the modded-nanogpt residual recipe
- `91df17f5` perf(torch): CUDA graph capture, packed rotary and fused RMSNorm bindings
- `2123ccfe` feat(reports): causal-patch report bases including the recipe scalars
- docs(timexer-segment): causal-patch protocol, worker reports and campaign script
  (carries this report, so its own hash cannot be printed inside it; it is `main`'s tip)

The three merged worktrees (`worktrees/{residual-recipe,unet-skips,value-residual}`) and their
branches were removed after the merge landed.
