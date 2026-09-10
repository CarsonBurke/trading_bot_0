# Horizon loss weighting and objective-aligned checkpoint selection

Worker: `HorizonWeight`. Scope: `trading_bots/src/torch/timexer_segment/{model.rs,runner.rs}`,
one appended writer in `reports.rs`, docs. No GPU runs submitted.

## Why

Jobs 5190-5193 refuted the scalar-lr and x0-shortcut hypotheses and localized the failure on the
HORIZON axis: between step 2k and 5k the held-out market-neutral MSE ratio kept improving at
h = 1 (0.9636 -> 0.9602) and h = 8 (0.9458 -> 0.9256) while h = 64 (0.9886 -> 1.0688) and
h = 192 (1.0257 -> 1.0854) rotted, with calibration nominal and the mis-scaling cross term
inverting along the axis. One trunk plus one equal-weighted loss over 192 horizons means the
long horizons' memorization of the training period owns both the gradient and the scalar that
selects checkpoints. Both halves are now configurable, and selection is aligned to the
objective an arm was actually trained on.

## Modes — `--horizon-loss <spec>`

`HorizonLoss` (`model.rs`), a `ModelConfig` field, so it flows through the existing
`#[command(flatten)]` in `TrainArgs` and is serialized into `manifest.model.horizon_loss`. No
`main.rs` edit was needed. Effective weights (mean 1 over all 192 horizons):

| mode | h=1 | h=8 | h=32 | h=64 | h=192 | mass on h<=8 | mass on h<=32 |
|---|---|---|---|---|---|---|---|
| `uniform` (control) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4.17% | 16.67% |
| `inv-sqrt` (`1/sqrt(h)`) | 7.3036 | 2.5822 | 1.2911 | 0.9129 | 0.5271 | 16.63% | 37.82% |
| `inv` (`1/h`) | 32.8918 | 4.1115 | 1.0279 | 0.5139 | 0.1713 | 46.56% | 69.53% |
| `cutoff:32` | 6.0000 | 6.0000 | 6.0000 | 0.0000 | 0.0000 | 25.00% | 100.00% |

The two decays genuinely bracket the rate rather than being two spellings of one: because both
carry mean 1 they cross exactly once (at h ~= 20), `inv` holding 46.6% of the objective on
h <= 8 against `inv-sqrt`'s 16.6% and 0.1713 vs 0.5271 at h = 192. A test pins the ratio
`w_inv / w_inv-sqrt` strictly decreasing with `>1` at h = 1 and `<1` at h = 192.

## Normalization, and why the number stays interpretable

Two independent choices, both stated because they do different work:

1. **The weight vector is normalized to mean 1 over ALL `pred_len` horizons** (`sum w = pred_len`).
   This makes `uniform` the vector `1` exactly — 192 ones sum exactly in fp64 and the division
   by 192 is exact, so `w = 1.0f32` and the fold `mask * w` is the identity on the bits. It also
   makes the chart readable: departure from the `uniform reference 1.0` line is the whole story.
2. **The loss divides by `Sum w·mask·CHANNELS`, not `Sum mask·CHANNELS`.** The reported NLL is
   therefore a weighted MEAN of the per-element NLL — still nats per bar, still directly
   comparable across modes, and mathematically INVARIANT to the overall scale of `w` (the
   expression is homogeneous of degree 1 in `w`, numerator and denominator alike). This is the
   reason the number remains interpretable: `cutoff:32` reports the mean nats per bar over the
   32 horizons it trains, `inv` reports the `1/h`-weighted mean over all 192, and neither is a
   sum whose magnitude moves with the mode. As a consequence no learning-rate retuning is
   implied by switching modes: the gradient scale is normalized by the same denominator.

The MSE keeps the UNWEIGHTED denominator and the raw mask. It is a diagnostic every report and
every arm has to compare; its definition does not move with the objective.

`cutoff:K` masks the loss, it does not shrink the head. All 192 forecasts are still emitted so
every per-horizon diagnostic keeps reading the untrained end — which is the entire point of the
arm. Because the head weight is channel-major (`row = channel·pred_len + h`) and the geometry is
per-horizon separable, weight 0 gives gradient exactly `0.0` on the rows exclusive to horizons
above `K`; a test asserts `max |grad| == 0.0` there and `min |grad| > 0.0` below, on both
`head.output.weight` and its bias, plus a live trunk gradient.

## Cost, against 16.98 TFLOP/step and 143.20 GB/step (batch 256)

The accounting below is `ModelConfig::step_cost`, which reproduces the stated basis exactly
(16.984571904 TFLOP, 143.204352 GB at 275 units).

- **Matmul FLOPs: unchanged, +0.000%.** The weighting is elementwise over
  `[rows, origins', 1, pred_len]`; `step_cost::matmul_flops` counts matmuls only. The raw
  multiply count is 18.4 M, 1.8e-5 TFLOP, 0.0001% of the step.
- **Traffic: +0.221184 GB/step, +0.154%**, in EVERY mode (the weight is a constant buffer, so
  `uniform` runs the identical kernels with `w = 1`). `head_loss` goes 275 -> 278 passes over the
  73.728 MB fp32 channel space: one read of the mask, one write of `mask·w`, and one extra
  reduction for the unweighted MSE denominator. Gradient-free, so zero backward cost. New basis
  **143.425536 GB/step**.
- **`cutoff:32` waste, paid deliberately:** the head-output GEMM is 0.905970 TFLOP of the 16.98
  (5.33%); 5/6 of it — **0.754975 TFLOP, 4.45% of the step** — is arithmetic on zero-weight
  horizons. The 278-unit head term is 20.496 GB of the 143.43 (14.29%); 5/6 — **17.080 GB,
  11.91% of the step** — is traffic on zero-weight horizons. Nothing else changes: same batch,
  same CUDA-graph capture, same peak allocator, no fallback path.

## Selection

`Evaluation::objective_nll` = `Sum w·mask·nll / Sum w·mask·CHANNELS` on the held-out split, from
two new `Scorer::sums` slots (12, 13) fed by the SAME vector the gradient reads
(`model.horizon_weights()`), accumulated on device with no extra host transfer. It now drives
`weights/preview-best`, `--preview-patience`, and `--patience`. `Evaluation::nll` keeps the
equal-weighted aggregate definition, so `timexer_segment_loss` and
`timexer_segment_generalization_gap` are unchanged.

The rename is explicit, not silent: `Manifest::best_preview_nll` -> `best_objective_nll`, plus a
new `Manifest::selection` string `min held-out sample objective-weighted NLL
(horizon-loss=<spec>)`. `Manifest::read` cross-checks `selection` against the manifest's own
`model.horizon_loss` and refuses a pair that disagrees, exactly as it already does for the x0
stamp. Console lines print both scalars (`objective-weighted NLL` and `aggregate NLL`). Under
`uniform` the selection scalar is bit-equal to the aggregate — a test asserts exact equality —
so a `uniform` arm is comparable to every curve recorded before the knob existed.

## Report

`timexer_segment_horizon_loss_weight`, registered in `shared/src/report.rs` by `TradeHorizon`
(coordinated over hub; `tui` extends from that slice, so the bidirectional test covers it).
Writer `reports::write_horizon_loss_weight`, appended at the end of `reports.rs`. x = `bars
ahead`, y = `loss weight (dimensionless; mean 1 over the full horizon; 0 = horizon not
trained)`, two series: `training horizon loss weight` and `uniform reference 1.0`. Title carries
the spec and the exact count (`N of M horizons trained`). The vector is read back from the
buffer the loss multiplies by, never recomputed, so the chart cannot disagree with the gradient;
the writer refuses a vector that is not mean 1. The literal `192` was dropped from the y-label
because `--pred-len` is a knob and the label would otherwise be wrong at any other value.

## Stamps

- `FORMAT` v8 -> **v9**: `causal-patch-ohlc-universe-v9-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres-x0-learned`
  and `…-x0-none`. The parameter set is identical to v8 and `uniform` is bit-for-bit the v8 loss,
  so a v8 state dict would load without a shape error — which is exactly why the stamp has to
  move: a v8 manifest carries no `horizon_loss` field, so a `cutoff:32` checkpoint and a
  `uniform` one would be indistinguishable and their curves silently averaged. Both v8 stamps are
  in the stale-rejection list and tested.
- `OBJECTIVE` -> **`causal_patch_market_neutral_nll_v5`**.
- `NUMERICS` unchanged.
- Updated: `docs/timexer_segment.md` (loss paragraph, manifest paragraph, `--horizon-loss` flag
  and its cost note), the stale-stamp list and every pinned literal in `runner.rs` tests.

## Verification

- `cargo check -p trading_bot_0 --tests`: 0 errors.
- `cargo test -p trading_bot_0 timexer_segment`: **63 passed, 0 failed** (was 55 before this
  task's two workers; my additions are 5 tests).
- New tests binding behaviour: `uniform_horizon_loss_reproduces_the_previous_unweighted_objective`
  (fold and denominator bit-exact identities via `Tensor::equal`, plus the value against the
  pre-knob expression written out verbatim within the fused kernel's own documented 1e-4
  reassociation tolerance — the weighting contributes exactly zero to that difference);
  `a_horizon_cutoff_leaves_exactly_zero_gradient_above_it_and_still_forecasts_every_horizon`;
  `every_horizon_loss_mode_normalizes_to_mean_one_over_the_whole_axis`;
  `horizon_loss_specs_round_trip_and_invalid_ones_are_refused_before_any_run`;
  `the_selection_scalar_is_the_training_weighted_held_out_nll` (scalar host reference per mode,
  exact equality with the aggregate under `uniform`);
  `fused_loss_matches_the_reference_decode_and_nll_including_gradients` now runs all five
  weightings; `the_horizon_weight_chart_states_the_weighting_and_refuses_an_unnormalized_one`.
- CLI smoke test on the built binary: `garbage` and `cutoff:0` rejected by clap before anything
  starts; `cutoff:193` rejected by `ModelConfig::validate` with no `loading eligible ticker
  universe` line emitted (i.e. before the corpus loads); `uniform`, `inv-sqrt`, `cutoff:32` reach
  corpus load.

## The four arms — queue these

Shared flags (identical across arms), 12h limit as the current basis:

```bash
mlq submit --name timexer-horizon-uniform --max-parallel-runs 1 --time-limit 12h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment \
  --run timexer-horizon-uniform-20260907 \
  --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 --min-history 256 \
  --batch-size 256 --optimizer polar-express --preview-patience 3 --x0-lambdas disabled \
  --horizon-loss uniform

mlq submit --name timexer-horizon-invsqrt --max-parallel-runs 1 --time-limit 12h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment \
  --run timexer-horizon-invsqrt-20260907 \
  --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 --min-history 256 \
  --batch-size 256 --optimizer polar-express --preview-patience 3 --x0-lambdas disabled \
  --horizon-loss inv-sqrt

mlq submit --name timexer-horizon-inv --max-parallel-runs 1 --time-limit 12h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment \
  --run timexer-horizon-inv-20260907 \
  --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 --min-history 256 \
  --batch-size 256 --optimizer polar-express --preview-patience 3 --x0-lambdas disabled \
  --horizon-loss inv

mlq submit --name timexer-horizon-cutoff32 --max-parallel-runs 1 --time-limit 12h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment \
  --run timexer-horizon-cutoff32-20260907 \
  --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 --min-history 256 \
  --batch-size 256 --optimizer polar-express --preview-patience 3 --x0-lambdas disabled \
  --horizon-loss cutoff:32
```

`--x0-lambdas disabled` means all four are stamped
`causal-patch-ohlc-universe-v9-…-x0-none`. Reading them back:

```bash
./target/release/report_cli <gen> timexer_segment_horizon_loss_weight --run timexer-horizon-inv-20260907
./target/release/report_cli <gen> timexer_segment_horizon_steps      --run timexer-horizon-inv-20260907
```

## What to look for

`uniform` is the control and must reproduce the 5190-5193 shape (held-out bottoming near step
2,000, then rising). The decays and the cutoff are informative in one specific way: if the
short-horizon skill that never stopped improving was being dragged by the long horizons, then
`inv` and `cutoff:32` should push the h = 1 and h = 8 MSE ratios below `uniform`'s 0.9602 /
0.9256 and move the selected step later. If they do not, the drag hypothesis is wrong and the
short-horizon ceiling is intrinsic. `cutoff:32` additionally reads the h > 32 diagnostics on a
head that received zero gradient there, which separates "the representation is being damaged" from
"those horizons are simply unpredictable".
