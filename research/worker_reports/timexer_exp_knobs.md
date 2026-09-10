# CausalPatch diagnostic knobs: `--scalar-lr-mult`, `--x0-lambdas`, and the lr axis

Three knobs were required by the four ranked diagnostics; **two were added and one already
existed**. Every knob is stamped into the checkpoint, so no run is anonymous.

## What changed

| file | change |
| --- | --- |
| `trading_bots/src/torch/timexer_segment/model.rs` | New `pub enum X0Lambdas { Enabled, Disabled }` (`ValueEnum` + serde, kebab-case) and `ModelConfig::x0_lambdas`, so the flag is `--x0-lambdas <enabled\|disabled>` through the existing `#[command(flatten)]`. `CausalPatchModel::x0_lambdas` is now `Option<Tensor>` — under `disabled` the `lambdas.x0` var is **never registered**. `BlockLambdas::x0` became `Option<(&Tensor, &Tensor)>` (the embedding and its scale as one thing) and `Block::forward` lost its separate `x0` parameter: the second `addcmul` is not issued, so there is no zero-multiplied tensor, no gradient and no traffic. `step_cost` drops the `1·width` per-layer term, `recipe_scalar_parts` drops the `layers` x0 series, and `kernel_classes`'s composed layer charges 2 rather than 3 residual `addcmul`s and takes one input instead of two. |
| `trading_bots/src/torch/timexer_segment/compute.rs` | New `pub struct RecipeKnobs { scalar_lr_mult, x0_lambdas }` (+ `RecipeKnobs::reference`) threaded into `Engine::new` and `polar_express`. `NANOGPT_SCALAR_LR_MULTIPLIER` is now `pub` and is only the CLI default. New `scalar_lr_banks(X0Lambdas) -> &'static [&str]` is the single source of truth: `polar_express` routes exactly that slice and asserts `matched == banks.len()` (3 with x0, **2** without), and `OptimizerKind::recipe(RecipeKnobs) -> String` stamps the same slice, so a bank cannot be routed without being recorded. |
| `trading_bots/src/torch/timexer_segment/runner.rs` | `TrainArgs::scalar_lr_mult` (`--scalar-lr-mult`, default 5.0, validated before the corpus load). `FORMAT` split into `FORMAT_X0_LEARNED`/`FORMAT_X0_NONE` + `format_stamp(X0Lambdas)`; `OBJECTIVE` bumped; `Manifest::read` accepts either stamp **and** requires the stamp to agree with the manifest's own declared x0 mode. Manifest gained `base_learning_rate` and `scalar_lr_mult` (and `learning_rate` is documented as the scheduled rate it always was). The stale-stamp list gained the whole `v7` string. |
| `trading_bots/src/torch/timexer_segment/benchmark.rs` | `Engine::new` call sites take `RecipeKnobs::reference(config.x0_lambdas)`; the kernel repeat table's `residual addcmul` count follows the mode (3 or 2) in both the per-class scaling and the parts-vs-composed sum. |
| `trading_bots/src/main.rs` | **No change needed** — `TrainArgs` is a flattened `clap::Args`, so both flags appear on `train-timexer-segment` automatically (verified in `--help`). |
| `docs/timexer_segment.md` | Backbone bullet and the manifest paragraph rewritten for the two stamps, the two modes and the multiplier. |

### The third knob already existed: use `--learning-rate`

`OptimizerKind::muon_learning_rate(lr) = lr · (0.023/0.008)` — the NorMuon rate is *exactly
proportional* to the AdamW rate, and `Engine::apply_schedule` re-derives it every step. So
`--learning-rate` already scales **both** optimizer groups by one number; `--lr-scale 0.5` would
be the same knob spelled relatively. It was therefore **not added** (a second knob for one degree
of freedom is the kind of thing that later trains a model nobody can attribute). The 2×2 uses
`--learning-rate 0.008` (= the default) against `--learning-rate 0.004`, and the manifest now
records `base_learning_rate` so a checkpoint names its arm.

## New stamps

- `FORMAT_X0_LEARNED` = `causal-patch-ohlc-universe-v8-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres-x0-learned`
- `FORMAT_X0_NONE` = `causal-patch-ohlc-universe-v8-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres-x0-none`
- `OBJECTIVE` = `causal_patch_market_neutral_nll_v4`
- Recipe fragment (was `;resid-x0-lambda-skipw-lrmul=5;`):
  `;x0-lambdas=enabled;lambdas.resid+lambdas.x0+skip_weights-lrmul=5;`
  → at `--scalar-lr-mult 1`: `…-lrmul=1`; at `--x0-lambdas disabled`:
  `;x0-lambdas=disabled;lambdas.resid+skip_weights-lrmul=1;`
- Consequence: **`v7` checkpoints (including job 5188's `timexer-recipe-v3-20260906`) no longer
  load.** Their reports and `.report.bin` metrics are unaffected; only weight loading is. That is
  the required cutover — a `v7` manifest cannot say which multiplier or x0 mode produced it.

## Parameter counts (production config: 8 layers, d_model 512, ffn 2048, features all)

| mode | varstore entries | trained parameters | `step_cost` traffic @ batch 256 |
| --- | --- | --- | --- |
| `--x0-lambdas enabled` | 51 | 27,954,483 | 143.20 GB/step |
| `--x0-lambdas disabled` | 50 | 27,954,475 | 138.49 GB/step (−3.3%) |

Exactly one entry and `layers = 8` scalars fewer; matmul FLOPs identical (16.98 TFLOP/step).

## The four diagnostic runs

Shared prefix (all four):

```
mlq submit --name NAME --max-parallel-runs 1 --time-limit 2h --max-attempts 1 --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment --run NAME --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 --min-history 256 --batch-size 256 --optimizer polar-express --preview-patience 3
```

with `NAME` and the differing flags:

1. **Diagnostic 1 — scalar multiplier 1.0** (also the 2×2's `1x lr / mult 1` cell)
   `NAME=timexer-scalarlr1-20260906`, add `--scalar-lr-mult 1`
2. **Diagnostic 2 — x0 shortcut removed**
   `NAME=timexer-nox0-20260906`, add `--x0-lambdas disabled`
3. **Diagnostic 4 — half base lr, reference multiplier**
   `NAME=timexer-lr50-scalar5-20260906`, add `--learning-rate 0.004`
4. **Diagnostic 4 — half base lr, flat multiplier**
   `NAME=timexer-lr50-scalar1-20260906`, add `--learning-rate 0.004 --scalar-lr-mult 1`

The 2×2's fourth cell (`1x lr / mult 5`) is the already-measured recipe-v3 arm (job 5188), whose
1k–4k held-out NLL trace is in `timexer_recipe_v3_compare.md`; re-running it is unnecessary. One
epoch is 9,590 steps ≈ 27 min at 168.59 ms/step, so each command above covers the 1k–4k
diagnostic window inside its first ~12 minutes with the standard 1k report cadence, and
`--preview-patience 3` still ends a run that stops improving.

Diagnostic 3 (sigma calibration on training windows) needs no knob: it is a scoring pass over the
4k checkpoint, not a training configuration.

## Verification

- `cargo check -p trading_bot_0 --tests`: 0 errors, 0 new warnings in `timexer_segment/`.
- `cargo test -p trading_bot_0 timexer_segment`: **55 passed** (50 pre-existing + `HorizonSteps`'
  new report test + two new here), 1 ignored (CUDA), 0 failed.
- New tests: `compute::the_optimizer_recipe_records_the_scalar_multiplier_and_the_x0_mode` (the
  string moves with the multiplier and the mode, and never mentions `lambdas.x0` when it is gone);
  `model::disabling_the_x0_injection_removes_its_parameters_and_its_kernel` (one fewer varstore
  entry, one fewer trainable, `layers` fewer chart series, lower traffic bound with identical
  GEMM count, bit-identical backbone to the learned mode at its `λ0 = 0` init, and provably
  different once `λ0 = 0.25`).
- `every_causal_patch_parameter_lands_in_its_intended_optimizer_group` now runs **both** modes at
  **different** multipliers, asserts NorMuon holds only 2-D `block_*.weight` tensors, that
  `lambdas.x0` has no AdamW group at all under `disabled`, that the residual/x0/skip banks carry
  the flag's value while `lambdas.post` stays at 1×, and that the disabled store has exactly one
  entry fewer. `polar_express`'s own `assert_eq!(set_named_lr_scale(banks, mult), banks.len())`
  therefore executes at 3 and at 2.
- `universe_checkpoint_authenticates_objective_schema_and_weight_bytes` covers the new
  objective, the four stale format stamps including `v7`, and a stamp/mode disagreement.
- CLI smoke: `train-timexer-segment --help` shows `--scalar-lr-mult` with `[default: 5]` and
  `--x0-lambdas` with possible values `enabled`/`disabled` and `[default: enabled]`;
  `--scalar-lr-mult 0` fails fast with
  "the scalar learning-rate multiplier must be positive"; `--x0-lambdas disabled
  --scalar-lr-mult 1 --learning-rate 0.004` parses and reaches `train`.
