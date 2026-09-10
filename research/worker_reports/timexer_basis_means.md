# Structured-mean head (`--horizon-mean basis:S:B`): decision report

Evidence labels: **DEMONSTRATED** = observed in this repository's code, or produced by a command
run here and quoted verbatim. **HYPOTHESIS** = analytic projection, design judgment, or an
outcome no run has produced. Arithmetic identities are named as such. No GPU job was submitted
or run.

## 1. What was built

**DEMONSTRATED — Scope.** `--horizon-mean <free|basis:S:B>` on
`trading_bots/src/torch/timexer_segment/model.rs`, plus its `ModelConfig` field and the manifest
FORMAT stamp in `runner.rs`. `free` is today's dense head and the control arm. `basis:S:B` keeps
independent means for horizons `1..=S` and expresses the `pred_len - S` above them through `B`
fixed basis functions per channel. Log scales are untouched in both modes: one free parameter per
horizon per channel. The two stale comments the nanogpt ledger flagged were corrected in place;
neither the attention-O initialization nor the RMSNorm epsilon was changed.

## 2. The basis, fixed before any validation number was read

**DEMONSTRATED — Functional form** (`HorizonMean::basis`/`timescales`, `model.rs`). For a
restricted horizon `h > S`, with `u = h - S ∈ {1 … pred_len - S}`:

```
mean(channel, h) = Σ_{b=1..B} c(channel, b) · φ_b(u)
φ_b(u)           = exp(-u / τ_b) / rms_b
τ_b              = S · (pred_len / S)^((b-1)/(B-1))          (τ_1 = pred_len when B = 1)
rms_b            = sqrt( mean_u exp(-u/τ_b)² )
```

At `S = 8, B = 8, pred_len = 192` the timescales are exactly
`8, 12.60, 19.84, 31.23, 49.18, 77.44, 121.93, 192` bars (**DEMONSTRATED**, asserted in
`the_basis_represents_a_term_structure_and_refuses_per_horizon_idiosyncrasy`: fastest `= S`,
slowest `= pred_len`, strictly increasing).

**Why this family, and why it can represent a term structure but not per-horizon idiosyncrasy:**

1. **DEMONSTRATED — identity.** A nonzero real exponential sum `Σ_b c_b e^{-u/τ_b}` with distinct
   `τ_b` has at most `B - 1` zeros in `u` (Descartes' rule for exponential sums). Every mean the
   span can emit therefore changes sign at most `B - 1 = 7` times across all 184 restricted
   horizons. Per-horizon idiosyncrasy needs one sign change per horizon. That gap IS the
   restriction.
2. **HYPOTHESIS — economics.** The close coordinate is in units of the h-step persistence
   deviation `σ√h`, i.e. a per-horizon signal-to-noise ratio, not a price. A predictable
   component with half-life `T` contributes a `σ√h`-normalized profile decaying like `exp(-h/T)`,
   so a sum of `B` geometrically spaced exponentials is a discretized mixture over half-lives.
   By Bernstein's theorem every completely monotone decay (power laws included) is such a
   mixture, so the span covers the plausible term structures rather than one hand-picked curve.
3. **DEMONSTRATED — grid endpoints.** Fastest `τ = S`: anything decaying faster than the free
   band's own width is already representable inside the free band. Slowest `τ = pred_len`, which
   over the restricted band falls 0.995 → 0.383 — a drift, deliberately not a constant, because a
   flat level in `σ√h` units is a cumulative return growing like `√h` forever.
4. **DEMONSTRATED — zero is persistence, exactly.** The expansion is linear and homogeneous: no
   intercept function, no additive constant, and the head bias belongs to the coefficients
   themselves (zero-init). `zero_basis_coefficients_are_exactly_persistence_above_the_free_band`
   asserts `max |coordinate| == 0.0` and decoded open/close `== 0.0` above the band while the
   free band and the log scales carry nonzero values.
5. **DEMONSTRATED — no free lunch on conditioning.** Insisting on per-function monotonicity (an
   orthonormal span would break it) leaves the coefficient Gram matrix ill-conditioned: columns
   overlap near `h = S + 1`. Measured consequence: an fp64 least-squares fit reaches a hyperbolic
   term structure to `1.1e-4` relative L2 while the shipped fp32 head emits it at `7.5e-3`.

**DEMONSTRATED — The contrast is a test, not a claim.** Same test, both targets pushed through
the real head (fitted coefficients written into the head bias, expanded by the shipped `Ψ`):

| synthetic target over `h = 9..192` | best relative L2, span (fp64) | emitted by the head (fp32) |
|---|---:|---:|
| `0.8 / (1 + (h-8)/24)` hyperbolic term structure | 0.0001 | **0.0075** (tolerance: < 0.05) |
| `0.8 · (-1)^(h-8)` per-horizon alternating | 0.9993 | **1.9820** (required: > 0.95) |

The alternating case exceeding 1 (worse than emitting nothing) is fp32 rounding of large
cancelling coefficients, printed but not asserted on; the span's own 0.9993 is the measurement.

**DEMONSTRATED — Log scales are not restricted.** Coverage at 5k was 0.678/0.923 against nominal
0.683/0.950, so the distribution's shape is not the measured failure; restricting it would trade
a working part of the model for nothing.

## 3. Exact deltas at the shipped configuration

**DEMONSTRATED** — all numbers below printed by
`basis_means_delete_the_long_horizon_parameters_rather_than_masking_them` (batch 256,
`ModelConfig::default()`, `--features all`), and the parameter totals by a probe over the same
config. Baseline reproduces the cited basis: 16.984572 TFLOP, 143.425536 GB.

| quantity | `free` | `basis:8:8` | delta |
|---|---:|---:|---:|
| head output width | 1536 | 832 | −704 |
| `head.output.weight` | `[1536, 1024]` | `[832, 1024]` | −720,896 |
| `head.output.bias` | `[1536]` | `[832]` | −704 |
| model parameters | 27,954,483 | 27,232,883 | **−721,600 (−2.581%)** |
| varstore entries | 51 | 51 | **0** |
| step matmul | 16.984572 TFLOP | 16.989806 TFLOP | **+0.005234 TFLOP (+0.031%)** |
| step traffic (analytic bound) | 143.425536 GB | 143.438119 GB | **+0.012583 GB (+0.0088%)** |

**DEMONSTRATED — Where the arithmetic goes.** The expansion is folded onto the output WEIGHT, not
applied to activations: `Ψ · W` is `[1536, 832] × [832, 1024]`, charged as two passes (forward and
weight gradient; `Ψ` is constant and takes no gradient) = `2 · 2 · 1536 · 832 · 1024` = 5.234
GFLOP, and the folded weight plus its gradient are 3.15 MB each, charged at four materializations
= 12.58 MB. This is the same idiom `scaled_linear` already uses for the post-lambdas and the μP
multiplier.

**DEMONSTRATED — The dense-target traffic does NOT disappear.** The head still emits all
`2·CHANNELS·pred_len = 1536` outputs per token, the 278-unit fp32 loss/geometry term is unchanged,
and the likelihood still consumes all 192 means. `Head`, `output`, the fused `losses` and every
per-horizon diagnostic are shape-identical in both modes.

**DEMONSTRATED — The rejected alternative, costed.** Expanding in ACTIVATION space (emit 832,
matmul the tail, concatenate) would cut the token-space head GEMM by
`3 · 2 · tokens · 1024 · 704` = 0.415 TFLOP (−2.4%) and add 0.0034 TFLOP of tail expansion, but it
must materialize the restricted block and concatenate it with the free band and the log scales:
+0.92 GB/step (+0.64%) at best — and only after `Head` stops being one tensor and the fused loss
stops reading one `split` node (+2.69 GB if it stays one tensor and pays the second `cat`).
Rejected: this model sits within a factor of two of both roofs, and an ablation arm wants the
control's cost profile, not a cheaper one. **HYPOTHESIS:** at the measured 168.59 ms/step the
chosen fold should land inside noise (< 0.1 ms), and peak allocator should fall ≈5 MB (11.5 MB of
fp32 master + gradient + two AdamW moments for 721,600 deleted parameters, against 6.3 MB of
folded weight and gradient) out of 18,573.73 MiB.

## 4. Contract stamps

**DEMONSTRATED.**
- FORMAT `v9` → `v10`, and unlike the `v9` bump this one really changes the parameter set. Stamps
  are now a base plus a mean suffix:
  `causal-patch-ohlc-universe-v10-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres-x0-{learned|none}-mean-{free|basis-S-B}`.
  `(S, B)` is in the stamp because two `basis` arms with different pairs have different parameter
  sets.
- Both `v9` stamps were added to the stale-rejection list in
  `universe_checkpoint_authenticates_objective_schema_and_weight_bytes`; `Manifest::read` accepts
  only the two `v10` bases and cross-checks the exact `-mean-` suffix against the manifest's own
  `horizon_mean`.
- OBJECTIVE stays `causal_patch_market_neutral_nll_v5`. The likelihood, the targets, the mask, the
  per-element NLL, the horizon weighting and the selection scalar are all unchanged; what changed
  is what the model can express, which is the parameter set, not the objective.
- `--horizon-mean` is rejected before the corpus loads: garbage, `basis:0:8`, `basis:8:0`,
  `basis:8`, `basis::8`, `basis:8:8:8`, `basis:8.5:8` and negatives fail at clap parse;
  `basis:192:1` (no restricted horizon) and `basis:8:185` (more functions than horizons) fail in
  `ModelConfig::validate`, which runs as the first statement of `train`, 32 lines before
  `Corpus::load`. Verified on the built binary with `--data-dir /nonexistent-corpus-path`: the
  invalid arms never reach the corpus, `basis:8:8` does.

## 5. CUDA-graph capturability

**HYPOTHESIS (structural, not measured — no GPU run was made).** The fold adds exactly one
static-shape GEMM plus two casts of constant-size tensors per step. `Ψ` is allocated once in
`CausalPatchModel::new` at a fixed device address and is never written; the mode is an
`Option<Tensor>` resolved at construction, so there is no data-dependent dispatch inside the step;
there is no host `.item()`, no allocation whose size depends on data, and no shape below the head
changes. That is the same profile as the existing `horizon_weight` constant buffer, which is
already inside the captured region. The repository's own check for this is
`benchmark --profile`'s 20-step eager-vs-captured parameter audit (`CAPTURE_AUDIT_STEPS`), which
must be run on the arm before the numbers above are treated as measured.

## 6. The arm to queue

**DEMONSTRATED — exact submit line**, matched to the four horizon-loss arms' shared flags:

```bash
mlq submit --name timexer-basis88 --max-parallel-runs 1 --time-limit 12h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment \
  --run timexer-basis88-20260907 \
  --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 --min-history 256 \
  --batch-size 256 --optimizer polar-express --preview-patience 3 --x0-lambdas disabled \
  --horizon-mean basis:8:8
```

It differs from the `timexer-horizon-uniform` control line in exactly two places: `--run` and
`--horizon-mean basis:8:8` (the control's `--horizon-loss uniform` is the default and is omitted
here). Stamp: `…-v10-…-x0-none-mean-basis-8-8`.

**HYPOTHESIS — Falsifier, precommitted.** One matched ≤5k run against the `free` control at the
same seed and data order. Reject the structured mean if the 2k→5k h=64 market-neutral MSE ratio
degradation is not arrested and 5k h=64 does not come below the control's 1.0688, if h=1/h=8 lose
skill beyond day-block uncertainty, if `|coverage − 0.683|` worsens, or if the held-out sample
objective NLL minimum does not improve. A smaller |mis-scaling cross term| at h=64 with no h=64
ratio improvement is not a pass.

## 7. Verification run here

**DEMONSTRATED.**
- `cargo check -p trading_bot_0 --tests`: 0 errors.
- `cargo test -p trading_bot_0 timexer_segment`: 74 passed, 0 failed (65 before this batch; the
  five added here plus siblings' additions).
- New tests, each binding behaviour rather than plumbing:
  `free_horizon_mean_is_the_dense_head_and_the_fold_copies_one_hot_rows_exactly` (bit-equality
  against the transcribed pre-knob GEMM, and bit-equality of a one-hot fold against the unfolded
  head — the property that makes the free band and log scales under `basis` identical to the dense
  head); `basis_means_delete_the_long_horizon_parameters_rather_than_masking_them` (same varstore
  entries, exact `704 · (1024 + 1)` parameter drop, one restricted horizon's gradient reaches every
  coefficient row and no free-band or log-scale row, plus the exact FLOP/traffic deltas);
  `zero_basis_coefficients_are_exactly_persistence_above_the_free_band`;
  `the_basis_represents_a_term_structure_and_refuses_per_horizon_idiosyncrasy`;
  `horizon_mean_specs_round_trip_and_invalid_ones_are_refused_before_any_run`.
- CLI smoke test on the built debug binary, quoted in §4.
- No GPU job submitted or run; no batch size, capture or precision setting touched.

## 8. The two ledger comment fixes

**DEMONSTRATED — `zeroed_projection` (was: attention-O conflated with `CastedLinearT`).** The
comment claimed the reference's real attention-O weights are "folded into `CastedLinearT`, whose
`reset_parameters` is `nn.init.zeros_`". They are not: `train_gpt.py:1293` fills the whole real
`vo_bank` — every V and every O — from a uniform, only the `world_size` padding is zeroed at
`:1294`, and `:1161` uses O straight from that bank; `CastedLinearT` (`:973-975`) is the LM head
class (`:1221`). The comment now says what our code does — O is zero-init, a deliberate divergence
that delays the QKV projections' first gradient — and points at ledger entry N01. The
initialization is unchanged.

**DEMONSTRATED — `rms_norm` (was: a shrink figure derived from `finfo(bfloat16).eps`).**
`_fused_rms_norm` resolves `eps=None` from the fp32 accumulate type, `FLT_EPSILON = 1.1920929e-7`,
so `finfo(bfloat16).eps = 7.8125e-3` never reaches the kernel and no consequence of it is a
consequence of the default — not the 0.388% shrink it would give at unit RMS (the ledger's "0.4%"),
nor the 95.25% it would give at the collapsed RMS the old comment quoted. The comment now states
only checkable arithmetic for what passing `1e-6` actually changes, as `1 - √(ms/(ms + eps))`: at
unit RMS 5.0e-7 against 6.0e-8, a 4.4e-7 gap far below bf16's 2^-8 resolution; at a collapsed RMS
of 4.2e-3 (mean square 1.76e-5, where `1e-6` is 5.7% of it) 2.72% against 0.34%, a 2.4% gap. That
2.4% is the entire behavioural content of the explicit epsilon. The epsilon is unchanged.
