# Schedule budget, and the MLP-down effective-LR port

Two separable changes, one fidelity fix and one hypothesis-driven change. Both are LR-only:
no storage change, no forward-pass change, no gradient accumulation, no batch-size change, no
CUDA-graph change. Peak allocator, TFLOP/step and GB/step are untouched.

**Carry this warning first.** The ledger's own conclusion is that a faster-fitting fix may
IMPROVE training loss and WORSEN held-out NLL, because every arm already overfits inside epoch
1 (`timexer_nanogpt_ledger.md:174-177,199`). Change 2 is exactly such a fix. It is worth
knowing that 8,388,608 trunk weights were being under-trained by 4x, and it is not a candidate
cure for the long-horizon rot. Predicted signs are below, written before either arm exists.

| | Change 1 — schedule budget | Change 2 — MLP-down effective LR |
|---|---|---|
| Kind | **Hypothesis-driven** (shape/endpoint change) | **Fidelity fix** (mis-port correction) |
| Flag | `--schedule-budget <steps>` (0 = whole run) | `--mlp-down-lr <aspect-only\|upstream-4x>` |
| Predicted held-out NLL sign | **IMPROVES** (lower) at 5k | **WORSENS** (higher) at 5k |

## 1. Schedule budget — hypothesis-driven

DEMONSTRATED (code, before): the endpoint the warmdown was drawn against was
`steps_per_epoch * epochs`, i.e. 9,590 steps at batch 256, purely because that is how many
steps an epoch contains. `cooldown_frac = 0.60` then put the first sub-peak step at
`floor(9590 * 0.4) = 3836`, 1,836 steps after the measured held-out NLL optimum at 2,000.
The shape itself is a faithful port of `train_gpt.py:1968-1976`; only its scale was an
accident.

DEMONSTRATED (code, after): `LrSchedule::new(budget_steps, cooldown_frac, floor)`
(`compute.rs`) is the one place the shape lives, the budget is a stated CLI quantity
independent of the epoch length, and the fraction and floor are passed in explicitly at the
runner's call site rather than read from inside the function. `RecipeKnobs::schedule` carries
it, and the recipe string a checkpoint is stamped with is now
`…;mlp-down-lr=<mode>;schedule-budget=<N>-cooldown-frac=0.6-floor=0.15-cooldown-start=<S>-v2`
— the resolved cooldown start is in the stamp, so no reader has to recompute a floor of a
product. The superseded `…-v1` strings (`nanogpt-cooldown-frac=.60-floor=.15-v1`) named a
shape with no endpoint and are gone. `LrSchedule::flat()` replaces the old magic
`scheduled_steps = 0` for the benchmark/capture/equivalence harnesses.

Exact multiplier and per-family realized rates, base AdamW 0.008 / NorMuon 0.023
(DEMONSTRATED, computed from the shipped `LrSchedule::scale`, pinned in
`the_schedule_reproduces_the_ported_shape_at_the_budget_it_is_given` and
`a_shorter_budget_reaches_its_floor_at_the_stated_step`):

| step | mult @9,590 (before) | AdamW | NorMuon | mult @5,000 (after) | AdamW | NorMuon |
|---|---|---|---|---|---|---|
| 0 | 1.0000 | 0.008000 | 0.023000 | 1.0000 | 0.008000 | 0.023000 |
| 2,000 | 1.0000 | 0.008000 | 0.023000 | 1.0000 | 0.008000 | 0.023000 |
| 2,500 | 1.0000 | 0.008000 | 0.023000 | 0.8583 | 0.006867 | 0.019742 |
| 3,000 | 1.0000 | 0.008000 | 0.023000 | 0.7167 | 0.005733 | 0.016483 |
| 3,836 | 1.0000 | 0.008000 | 0.023000 | 0.4798 | 0.003838 | 0.011035 |
| 5,000 | 0.8281 | 0.006624 | 0.019045 | 0.1500 | 0.001200 | 0.003450 |
| 7,500 | 0.4587 | 0.003670 | 0.010551 | 0.1500 | 0.001200 | 0.003450 |
| 9,590 | 0.1500 | 0.001200 | 0.003450 | 0.1500 | 0.001200 | 0.003450 |

Cooldown start: 3,836 before, 2,000 after. Mean multiplier over steps 0–4,999: 0.9800 before,
0.7451 after — same peak, 24% less integrated step size over the interval where the rot
happens.

Coupled consequence, stated rather than hidden: the budget also shapes the NorMuon momentum
cooldown, exactly as upstream shapes both against one `num_steps`. A 5,000-step budget moves
the 0.95→0.85 rampdown from steps 9,540–9,590 to 4,950–5,000. That is 50 steps, 1% of the arm,
and it is pinned in `the_muon_momentum_cooldown_follows_the_same_budget`.

Deliberate non-change: `cooldown_frac`/`floor` are explicit constructor arguments and stamped,
but not CLI flags. Reason — `NUMERICS` in `runner.rs` still asserts
`nanogpt-lr-cooldown-frac=.60-floor=.15` for checkpoint loading, and a flag that can falsify a
validated stamp would either be a silent lie or would force a NUMERICS bump that rejects every
existing arm's `weights/best` for `evaluate`. The budget knob spans the question being asked;
if an arm ever needs a different fraction, that is a one-line call-site change plus a NUMERICS
bump with a reason. Say the word and I will make it a flag.

Predicted sign: **held-out NLL IMPROVES (lower at 5k)**, HYPOTHESIS. Reasoning: halving the
PEAK was measured and was strictly worse (2.2207 vs 2.0669 at 5k), which refutes "less LR is
better" as a level statement but says nothing about the endpoint. Every arm's minimum is at
2,000 and the mis-scaling cross term inverts along the horizon axis between 2k and 5k
(h=1 −0.070→−0.011, h=64 −0.007→−0.073) while the train-minus-held-out NLL gap collapses
0.1665→0.0393 — the signature of long-horizon conditional means fitting training-period
structure at full step size long after the useful fitting is done. Annealing into that regime
instead of holding the peak through it should reduce the amplitude drift without removing the
early fitting that produced the h=1/h=8 skill. Falsifier: if the 5,000-budget arm's held-out
objective NLL at 5k is not below 2.0669, or h=64's market-neutral MSE ratio is still above 1.0,
the schedule-shape hypothesis is dead and the mismatch was real but not causal.

## 2. Per-family effective LR — fidelity fix

DEMONSTRATED (ledger + code): upstream stores both MLP matrices tall `[4D, D]`, so
`max(1, rows/cols).sqrt()` hands both a shape multiplier of 2, and it gives the down matrix's
`c_proj` group a further `lr_mul = 2` (`train_gpt.py:509-523,1299-1301`) — effective 2x up, 4x
down. We store down as `[D, 4D]` (`model.rs:631`), whose aspect scale is 1
(`muon.rs:1000-1006`), with no override (`compute.rs`): effective 1x, a quarter of upstream, on
8,388,608 trunk weights.

Ported as LR-only: `MlpDownLr::Upstream4x` sets the `lr_scale` of every `block_*.second.weight`
to `UPSTREAM_MLP_DOWN_LR_MULTIPLIER = 4.0`, which multiplies its aspect scale of 1. The product
is what steps the weights, so the port pins the product and not upstream's decomposition. Under
`quadratic_lr_weight_decay` this carries the same weight-decay factor upstream's `lr_mul`
carries — both enter the decay through exactly one of its two LR factors
(`muon.rs:141-146`) — so the port is faithful in decay as well as in step size, without a
second knob. Realized rates at base 0.023, pinned in
`the_mlp_down_matrices_run_at_the_documented_upstream_multiplier`:

| NorMuon family | shape | aspect | lr_scale | before | after |
|---|---|---|---|---|---|
| packed QKV | `[3D, D]` | √3 | 1 | 0.039837 | 0.039837 (untouched) |
| attention output | `[D, D]` | 1 | 1 | 0.023000 | 0.023000 (untouched) |
| MLP up | `[4D, D]` | 2 | 1 | 0.046000 | 0.046000 (untouched) |
| **MLP down** | `[D, 4D]` | 1 | 1 → 4 | **0.023000** | **0.092000** |

Packed QKV deliberately untouched, with the reasoning recorded: the ledger's analysis is that
the √3 is a GEOMETRY mismatch against upstream's per-head-pair Q/K and square V banks, not an
excess update scale — the expected aggregate Frobenius update scale of our packed matrix is
already about 1 — so removing √3 alone would not be a faithful port either. A faithful port of
that row is N02 (separate Q/K head-pair and V/O matrices), which is an optimizer-geometry
change and out of this scope.

Predicted sign: **training NLL improves, held-out NLL WORSENS (higher at 5k)**, HYPOTHESIS.
Reasoning: the FFN-down matrix is the residual stream's write path and its rate is a pure
fitting accelerator with no horizon index and no SNR knowledge. Everything already overfits
inside epoch 1, the gap is already collapsing by 3k, and 4x on 8.4M weights of the write path
reaches the period-specific mean-fitting regime sooner. Secondary prediction: the best held-out
step moves EARLIER than 2,000. Falsifier: if held-out objective NLL at 5k improves, then our
trunk was genuinely under-trained rather than over-fit, and the whole "already overfitting"
frame needs revisiting rather than patching.

## Realized-LR reporting

New base `timexer_segment_lr_trajectory` (registered in `shared/src/report.rs`; the TUI's
`meta_chart_bases()` extends from that slice, so the bidirectional test covers it by
construction and no `tui/src/main.rs` edit was needed). Six series — `NorMuon packed QKV`,
`NorMuon attention output`, `NorMuon MLP up`, `NorMuon MLP down`, `AdamW dense`,
`AdamW recipe scalars` — x-axis `schedule step (0-based, the index the cooldown start names)`,
y-label `learning rate (absolute; higher = larger parameter step; 0 = family frozen)`. Title
states the budget and its resolved cooldown start.

Six series and not three because the aspect scale makes "the NorMuon rate" four different
numbers at one base rate. Values are read out of the optimizer's own state
(`Muon::applied_learning_rate` → `normuon_effective_lr`/`adamw_effective_lr`, the SAME
functions `resolve_step_scalars` publishes the device step scalars from), never recomputed from
`LrSchedule`, so the chart cannot describe a rate the step did not take; a disabled parameter
reports 0. Recorded every step, host-only: six `f64` into one flat buffer, ~460 KB for a
9,590-step run, no device read and no synchronization. `Engine::learning_rate()` likewise now
returns the value last handed to the optimizer instead of recomputing the schedule at
`completed_steps - 1`.

Read back with:

```bash
./target/release/report_cli 1 timexer_segment_lr_trajectory --run <run>
```

## The two arms — single-variable, against the shared flags

```bash
mlq submit --name timexer-budget5k --max-parallel-runs 1 --time-limit 12h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment \
  --run timexer-budget5k-20260907 \
  --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 --min-history 256 \
  --batch-size 256 --optimizer polar-express --preview-patience 3 --x0-lambdas disabled \
  --schedule-budget 5000

mlq submit --name timexer-mlpdown4x --max-parallel-runs 1 --time-limit 12h \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment \
  --run timexer-mlpdown4x-20260907 \
  --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 --min-history 256 \
  --batch-size 256 --optimizer polar-express --preview-patience 3 --x0-lambdas disabled \
  --mlp-down-lr upstream-4x
```

The differing flag is the only difference: `--schedule-budget 5000` versus
`--mlp-down-lr upstream-4x`. Everything else, including the peak LR, the scalar multiplier
default of 5x and the disabled x0 bank, matches the completed x0-disabled arm that reached
2.0669 at 5k, so each is a one-variable comparison against it. Not submitted: the mlq queue is
Main's and five foreign tenants hold it.

## Verification (DEMONSTRATED)

- `cargo check -p trading_bot_0 --tests`: zero errors.
- `cargo test -p trading_bot_0 timexer_segment`: 70 passed / 0 failed at the moment my last
  change landed. On the most recent run, with two sibling workstreams still editing, the count
  is 73 passed / 1 failed; the single failure is BasisMeans' in-flight
  `model::tests::the_basis_represents_a_term_structure_and_refuses_per_horizon_idiosyncrasy`
  (model.rs:3568, basis span fits to 1.1e-4 but 7.5e-3 through the head), reported to them and
  in no code path of mine. All six of my tests pass in that same run.
  CLI surface smoke-tested against the real binary: `train-timexer-segment --help` lists
  `--schedule-budget` (default 0) and `--mlp-down-lr` with `aspect-only`/`upstream-4x`.
- `cargo test -p trading_bot_0 muon`: 42 passed, 0 failed — the step-scalar publication tests
  assert `-(lr * lr_scale * aspect_scale)` slot-by-slot, which is what proves factoring the
  effective rate into one function left the published device scalars bit-identical.
- `cargo test -p trading-bot-tui`: 36 passed, 0 failed.

Tests that bind the behaviour, not the implementation:
`the_schedule_reproduces_the_ported_shape_at_the_budget_it_is_given` (budget 1,270 → the
upstream 508/0.575/0.15 curve; budget 9,590 → cooldown start 3,836, 0.828 at 5,000, i.e. today's
trajectory exactly), `a_shorter_budget_reaches_its_floor_at_the_stated_step` (budget 5,000 →
start 2,000, floor at 5,000, held past it, same peak),
`the_muon_momentum_cooldown_follows_the_same_budget`,
`the_mlp_down_matrices_run_at_the_documented_upstream_multiplier` (per-family realized rates in
both modes, all four NorMuon shapes, both AdamW families),
`the_realized_trajectory_records_every_family_at_its_own_rate`, and
`the_lr_trajectory_panel_reports_one_realized_rate_per_family_per_step` (end to end through the
`.report.bin` format). `every_causal_patch_parameter_lands_in_its_intended_optimizer_group` is
unchanged and still green: NorMuon receives exactly the four 2-D block matrices per layer and
nothing else.
