# Why job 5119 and the training run disagree about the same checkpoint

**Verdict: CAUSE 1 (silent checkpoint incompatibility), proven from the checkpoint itself.
CAUSE 2 (the new trading-metric path) is excluded by an exact invariant test.** The head weight
in `training/runs/timexer-market-neutral-20260906/weights/best` is **horizon-major**; the current
binary reshapes those same 1536 rows as **channel-major**. Every tensor name and every shape still
matches, the manifest still says `causal-patch-ohlc-universe-v5` / `causal_patch_market_neutral_nll_v2`,
so the load succeeded silently and job 5119 scored a permuted model.

## The experiment that was possible

The assigned bisect - build the commit immediately before the head-fusion rewrite - has no target.
`git log -- trading_bots/src/torch/timexer_segment/model.rs` returns exactly one commit
(`bfb0fe43`), and that commit's `model.rs` is a different architecture entirely (softplus close,
`price_scaling`, no `OUTPUTS_PER_BAR`, no `HEAD_OUTPUT_SCALE`). The whole causal-patch module -
including the fusion - lives in the uncommitted working tree, so there is no pre-fusion binary to
build. A worktree was created, source-snapshotted and reflink-seeded with `target/`, then removed
unused: the decisive evidence is in the artifacts, and it is stronger than a second GPU run because
it does not depend on reproducing anything.

### Evidence 1 - the checkpoint's own head weight is horizon-major

`head.output.weight` is `[1536, 1024]` = `[2·CHANNELS · pred_len, HEAD_HIDDEN]`. Rows normalized,
mean row-to-row cosine:

| statistic | value |
| --- | --- |
| mean cos(row r, row r+1) | **0.285** |
| mean cos(row r, row r+8) | **0.956** |

The head weight varies smoothly along the horizon and discontinuously across channels. Stride 8 -
not stride 1 - is what walks the horizon in this checkpoint, i.e. row `r = h·8 + c`. Two
independent confirmations of the same reading:

- **per-channel weight norms.** Under `[192, 8]`: 2.23, 4.15, 7.52, 3.05 for the four coordinates
  and ~2.61 for each of the four log scales - four distinct scales, as a trained candle head must
  have. Under `[8, 192]`: 3.39, 3.31, 3.43, 3.45, 3.43, 3.48, 3.48, 3.50 - eight indistinguishable
  values, the signature of interleaving unrelated channels.
- **the bias.** Under `[192, 8]` the log-scale channel 4 reads -9.22, -3.49, -1.54, -0.90, -0.52,
  +0.04 at h = 1, 2, 4, 16, 64, 192: the smooth correction to the `½·ln h` prior. Under `[8, 192]`
  the bias is white noise along "horizon" (roughness `mean|Δ|/std` ≈ 1.0 versus 0.04-0.23).

`model.rs:618` reshapes the head GEMM to `[rows, -1, OUTPUTS_PER_BAR, horizon]`, i.e. row
`r = c·192 + h`. So the current binary reads coordinate `c` at horizon `h` out of row `c·192+h`,
whose trained content is **channel `(c·192+h-1) mod 8` of horizon `⌊(c·192+h-1)/8⌋ + 1`**.

### Evidence 2 - job 5119's own per-horizon curve is periodic with period 8

`timexer_segment_horizon`, market-neutral 4-channel MSE ratio. gen 1 = the run's in-training
scoring under the code that trained it; gen `trading` = job 5119 under the current binary, same
`weights/best` (step 6000):

| reported h | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 16 | 17 | 32 | 33 | 64 | 65 | 185 | 192 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen 1 (in-training) | 0.948 | 0.924 | 0.921 | 0.943 | - | - | - | 0.918 | 0.916 | 0.926 | 0.929 | 0.946 | 0.949 | 0.973 | 0.971 | - | 1.014 |
| job 5119 | 0.946 | **1.476** | **1.070** | **1.009** | **2.863** | **2.466** | **2.601** | **2.177** | 0.977 | **2.506** | 0.984 | **2.706** | 0.988 | **2.835** | 0.991 | 0.997 | **2.417** |

gen 1 is a smooth monotone curve. Job 5119 is a **sawtooth of period exactly 8**: only
`h ≡ 1 (mod 8)` reads 0.95-0.99, and `h ≡ 5 (mod 8)` reads ~2.9 every single time. No forecaster's
MSE-versus-horizon curve is periodic in 8; a row index decoded with the wrong stride is.

### Evidence 3 - the `_offset` tilt is a log-scale coordinate times √h

`decode_joint` gives `close = coordinate(0) · √h`. Reported horizon `h` therefore reads row `h-1`:

| reported h | row | true (horizon, channel) | reported mean close coordinate | ÷ √h |
| --- | --- | --- | --- | --- |
| 1 | 0 | (1, 0 = close) | +0.004 | +0.004 |
| 16 | 15 | (2, 7 = close log-scale) | -2.753 | **-0.688** |
| 64 | 63 | (8, 7) | -6.020 | **-0.753** |
| 192 | 191 | (24, 7) | -10.231 | **-0.738** |

The "impossible -10 σ constant tilt" is one quantity - the close log-scale coordinate, ≈ -0.73 -
multiplied by `√h` three times over. Nothing but the row permutation predicts that agreement.

### Evidence 4 - calibration moved on an unchanged checkpoint

Within-σ coverage is a pure function of model and data, so it cannot move unless the loaded model
moved: in-training **0.6754 / 0.9246** (step 9000, `timexer_segment_calibration`, nominal
0.6827 / 0.9500) against job 5119's **0.5663 / 0.7512** on the step-6000 weights it selected. The
log-scale channels are permuted along with the coordinates.

### Evidence 5 - the trading-metric path is clean (cause 2 excluded)

New test `zero_forecast_scores_exactly_persistence_at_every_horizon`
(`runner.rs:2278`): an exact-zero forecast IS persistence, driven through the real
`Scorer::accumulate`/`finish` on CPU in three uneven batches, with nonzero per-window σ and mid and
a nonzero `½·ln h` prior so the σ and `√h` scalings are exercised, not bypassed. Asserted with
`assert_eq!`, not a tolerance, at every horizon: `close_mse_ratio == 1`, market-neutral
`mse == persistence_mse`, absolute-space (market-drift-rebased) `absolute_mse ==
absolute_persistence_mse`, `trimmed_mse_ratio == 1`, `mae_ratio == 1`, `delayed_mse_ratio == 1`
(NaN at h = 1), all four gain-decomposition terms `== 0`, `median_window_ratio == 1`, hit rates and
Pearson and cross-sectional IC at their nulls, decile returns `== 0`. **It passes.** The two
anchored ratios that must *not* be 1 are pinned against their closed form. One expectation of mine
was wrong and the code was right: `mid_anchor_hit_rate` is *not* undefined for a zero close
forecast - leaning from the trade close toward the mid is still a side - so it is checked against a
scalar reference instead.

The ratios in `trading()` are also scale-free by construction (`runner.rs:849`:
`error / persistence`, both accumulated from the same masked arrays), so no σ or `√h` factor can
enter them, and `absolute_squared` correctly scores the market-neutral forecast against
absolute-space persistence *without* adding the realized drift to the forecast - adding it would
be look-ahead. No bug found, and now none can appear silently.

## The fix (main tree)

1. **`runner.rs:119`** - `FORMAT` = `causal-patch-ohlc-universe-v6-head-channel-major-folded-mup`.
   The layout and the folded muP multiplier are named in the contract string, because the weight's
   name and shape are identical across the change and cannot signal it. No conversion shim, no
   permute-on-load: old checkpoints are worthless.
2. **`runner.rs:213-255`** - `check_head_layout`, called at the load site (`runner.rs:1505`) right
   after `store.load`. It compares the mean row-to-row cosine at stride 1 against stride
   `2·CHANNELS` and refuses anything whose horizon axis is the strided one, naming the mismatch.
   This is the gate a version string cannot be: it catches a hand-stamped manifest and a future
   layout change that forgets to bump `FORMAT`. On `weights/best` it fires (0.285 vs 0.956); it is
   deliberately silent when neither stride shows smoothness (`max < 0.5`), which is the
   zero-initialised head, because an untrained head carries no layout to check.
3. **Tests.** `horizon_major_head_weights_are_refused` (`runner.rs:2241`): a synthetic head with a
   per-channel signature plus a random walk along the horizon is accepted channel-major, refused
   after the permutation, and a zero head passes.
   `universe_checkpoint_authenticates_objective_schema_and_weight_bytes` now loops the stale stamps
   including `causal-patch-ohlc-universe-v5` specifically, asserting the error names both the stale
   string and `FORMAT`. `zero_forecast_scores_exactly_persistence_at_every_horizon` stays as the
   permanent regression test for the scoring path.

`model.rs` is untouched - the channel-major head is correct and stays. `KernelTraffic` owns that
file and has been told that any future change to the head reshape or to the muP folding must bump
`FORMAT` in the same commit, since `check_head_layout`'s stride assumption is tied to it.

## Validity of job 5119, per base

The close-channel forecast the trading bases score is `coordinate(0)[h-1] · √h`. At reported
`h = 8k+1` the content is the true close coordinate of horizon `k+1`, but the decode multiplies it
by `√h` rather than `√(k+1)`. **The two coincide only at h = 1.** Everything else - every other
horizon, every log-scale-dependent number, and the three non-close coordinates at *all* horizons
(they read rows 192/384/576 + h-1, i.e. close coordinates of horizons 25/49/73…) - is scoring a
model that was never trained.

| base | verdict | detail |
| --- | --- | --- |
| `timexer_segment_tradable`, `_tradable_rates`, `_signal`, `_offset`, `_decomposition` | **PARTIALLY VALID: the h = 1 column only** | these are functions of head row 0 alone at h = 1, which is the genuine trained close coordinate at its own horizon and scale. h ≥ 2: INVALID. |
| `timexer_segment_portfolio`, `_portfolio_sharpe` | **PARTIALLY VALID: the h = 1 column only** | h = 4, 16, 64, 192 read permuted rows. |
| one-bar-delay ratio and hit rate (in `_tradable`, `_tradable_rates`) | **INVALID at every horizon** | `ŷ_h - ŷ_1` differences a true-horizon-`(h-1)/8+1` coordinate against a horizon-`h` target; NaN at h = 1 by construction anyway. |
| `timexer_segment_horizon`, `_horizon_error`, `_horizon_rates`, `_horizon_robust`, `_error`, `_skill` | **INVALID at every horizon, h = 1 included** | 4-channel quantities; the range and position coordinates are permuted even at h = 1. That gen 1 reads 0.948 and 5119 reads 0.946 at h = 1 is a coincidence of contamination, not agreement. |
| `timexer_segment_calibration`, NLL, `_loss` | **INVALID** | log-scale channels permuted; 0.566/0.751 against the checkpoint's own 0.675/0.925. |
| `timexer_segment_candles` | **INVALID** | full candle geometry, all four coordinates. |

Concretely, on the quoted trading verdict: **cross-sectional IC 0.0300 ± 0.0027 at h = 1 is a valid
measurement**, and so are the h = 1 decile-spread Sharpes (9.05 gross, 1.56 at 1 bps, -5.92 at
2 bps), the h = 1 close-anchored close ratio 0.9858, the h = 1 mid-anchored ratio 0.9131 with hit
rate 0.6299, and the h = 1 decomposition (offset gain -1.3e-5, demeaned gain 0.0209, cross term
-0.0067). **The "decay to ~0.004" at h = 16-192 is INVALID**, as is every mid-anchor and delay
number beyond h = 1.

Two h = 1 conclusions therefore survive and are worth carrying forward: the mid anchor *improves*
the close-channel ratio (0.9131 < 0.9858), so the bid-ask-bounce story is refuted rather than
confirmed at h = 1; and the h = 1 gain is **not** an unconditional offset (offset gain ≈ -1.3e-5)
but a demeaned signal with `|ρ| ≈ 0.145` - the opposite of the prediction in
`timexer_trading_eval.md`. Both should be re-measured under current code before being relied on.

Real numbers require re-training under `causal-patch-ohlc-universe-v6-head-channel-major-folded-mup`;
`weights/best` cannot produce them and will now refuse to try, twice over.
`training/runs/timexer-market-neutral-20260906/gens/trading` is untouched; no new gen was written
because no evaluation was run (the GPU A/B was cancelled as unnecessary once the artifact evidence
settled the question).

## Verification

- `./torch-env.sh cargo check -p trading_bot_0 --tests`: zero errors.
  `cargo check --manifest-path tui/Cargo.toml --tests`: zero errors.
- `./torch-env.sh cargo test -p trading_bot_0 timexer_segment`: **35 passed, 0 failed** (33 before,
  plus the two added here).
- The layout probe's arithmetic was run against the real `weights/best` tensor before it was
  written into `runner.rs`: 0.2848 at stride 1, 0.9556 at stride 8, so `Manifest::read` rejects the
  v5 stamp and `check_head_layout` would reject the bytes even if the stamp were forged.
