# Orthonormal target basis: replacing the objective's horizon GEOMETRY without restricting its rank

`OrthoTargets`, 2026-09-07. Ships `--target-basis <cumulative|haar|dct>`, `--basis-weight
<uniform|snr>`, `--basis-stats <path>`, one new report base
`timexer_segment_target_basis`, and one new subcommand `basis-stats-timexer-segment`.

**Nothing was run on the GPU.** Every number below is either DEMONSTRATED on the host (a unit
test, an independent NumPy recomputation, or arithmetic over measured constants) or tagged
HYPOTHESIS and pre-registered.

## 0. Handoff state, for the successor

**LANDED AND GREEN, nothing half-landed.** `--target-basis <cumulative|haar|dct>`,
`--basis-weight <uniform|snr>`, `--basis-stats <path>`, `basis-stats-timexer-segment`, the
`timexer_segment_target_basis` base and its writer, and 14 tests. Last verification on a green
tree: `cargo check -p trading_bot_0 --tests` 0 errors, `cargo check -p trading-bot-tui --tests`
0 errors, `cargo test -p trading_bot_0 target_basis` 14/0, `cargo test -p trading-bot-tui` 36/0
with the bidirectional registry test, `cargo test -p trading_bot_0 timexer_segment` 166/0. The
default (`cumulative`, uniform) is a bit-exact identity with the pre-knob objective and
serializes byte-identically, so `timexer-control-4k` still loads and no existing arm moved.

**NOT LANDED, deliberately: `banded:b`.** §4.8 is a complete specification with its arithmetic
proven in fp64, and it is the operator I recommend over the dense rotation in both worlds. It
was not written because its tap count is the one number the design cannot derive from itself,
and job 5471 never ran. Writing a frozen tap vector first would be "a constant we are entitled
to write and not entitled to use", which is the failure mode this batch spent the session
learning.

**NO JOB NEEDS SUBMITTING, AND THE BLOCKER IS ALREADY GONE.** The measurement I was asking for
ran as **job 5478** (`--basis dct`, into `training/runs/timexer-control-4k/gens/1`), so `Γ` is
on disk: the taps are derivable from that artifact's measured per-coefficient second moment
WITHOUT 5471. **First task for the successor: read 5478, not re-queue it** — §8 lists what to
read and in what order, and it decides prediction 6 (this feature's kill criterion) as well as
unblocking `banded:b`. Note also that `/var/tmp/tb0_vNN` is a BINARY, not a data directory;
`--data-dir` should be omitted so it defaults to `shared::paths::DATA_PATH/bars`.

**MY DEPENDENCIES ON OTHER AGENTS' NUMBERS**, load-bearing first:

| number | source | used for | status |
|---|---|---|---|
| lag-1 −0.271 (h=8) … −0.218 (h=192), training span | `Main`/`FutureTeacher` | banded tap count and the 0.0242/0.0379 nats/bar recovery | LOAD-BEARING; exposure bounded by the min-\|ρ\| rule (§4.8) |
| `β = 0.26594` at h=192 | session brief | the 23%-of-log-amplitude reversal share (§4.5) | LOAD-BEARING; independently corroborated at 23.9% by `FutureTeacher` |
| 155.1 ms/step and ≈0.05 ms per fp32 slice-pass | `FuseLoss` | the entire cost model (§5, §4.8) | LOAD-BEARING |
| IC SE ≈.0164 on the `band` draw | `TemporalSplit` | retracting prediction 3, demoting prediction 2 | LOAD-BEARING on the retraction, which is the safe direction |
| `V_h/(h·V_1)` .398 held-out vs .652 recent | `Main` correction 2 | motivates the min-\|ρ\| freezing rule | cited, not load-bearing — the rule is safe under either |
| 9.2× rise in `σ_f/σ_y` | session brief | cross-check only | cited, not load-bearing |
| persistence anchor 2.3947663 | session brief | explicitly declared UNUSABLE across a basis boundary (§4.7) | not relied on |

**THE ONE RESULT THAT DEPENDS ON NO MEASURED NUMBER** and that `Main` asked be carried forward:
`det M = 1` for a unit-triangular banded whitener, so a mis-fitted SECOND-moment metric costs at
most the nats it was meant to recover and can never make the objective improper, while a frozen
FIRST-moment multiplier biases every forecast. Corollary: **freeze the MINIMUM |ρ| across
disjoint spans, never the mean** — the mean is the one choice that can be worse than doing
nothing.

---

## 1. What the change is, and why it is not `--horizon-mean basis:8:8`

`basis:8:8` restricted the emitted mean to eight exponential columns. That is a RANK
restriction on the predicted function, and it was refuted: the correlation gain collapsed
(`D` 0.00481 → 0.00116) while the mis-scaling penalty barely moved (`C` −0.0385 → −0.0373).

This restricts nothing. `W` is a full-rank orthonormal map on the horizon axis, hence a
bijection: every forecast function the head could emit before it can emit after, at the same
head shape, the same parameter count, and the same 192 emitted horizons. What changes is the
METRIC the loss measures error in. Concretely, with `r = y − ŷ` the horizon-space residual,
`ls_k` the head's k-th log-scale row and `w_k` the per-coefficient weight:

```
e   = W · (mask ⊙ r)                        one [tokens·4, 192] × [192, 192] fp32 GEMM
NLL = Σ_k w_k · rowmask · (½·(e_k·e^{-ls_k})² + ls_k) / Σ_k w_k · rowmask · 4
```

### 1.1 The scale had to move too, and that is forced

An orthonormal rotation of a diagonal covariance is not diagonal. Rotating the first moment
while keeping a per-horizon diagonal σ would leave the likelihood inconsistent with the
geometry of its own residual, and the restriction the rotation removed from the mean would
reappear in the second moment. So the head's four log-scale rows are REINTERPRETED as
per-COEFFICIENT log scales — same 192 rows, same head, no resize.

Two consequences, both load-bearing:

- **Joint horizon covariance, for free.** A diagonal scale in an orthonormal basis IS a
  structured FULL covariance over horizons, `Σ = Wᵀ diag(τ²) W`. Per-horizon σ is therefore
  DERIVED, `σ_h = √(Σ_k W_kh²·τ_k²)`, and it is derived at the ONE place that materializes the
  evaluation output (`CausalPatchModel::output`), so every existing per-horizon report —
  calibration coverage, the scale charts, the trading family — reads a per-horizon scale that
  still means a per-horizon scale, with no edit to any of them. DEMONSTRATED against an
  independent host computation of the covariance diagonal by
  `derived_horizon_variance_is_the_covariance_diagonal`.
- **The reported NLL stays in the same UNITS — and I initially over-claimed this, so read §4.7.**
  `|det W| = 1`, so the change of variables contributes NO Jacobian term: the coefficient-space
  NLL is *identically* the horizon-space negative log density of the induced full-covariance
  Gaussian, in the same nats per bar, of the same observed 192-vector. Whitening does not break
  this because it is applied as the log-scale PRIOR CENTRE (inside `τ_k`), not as a change of
  variable. **What that does NOT buy is comparability of the NUMBER against the 2.3947663
  anchor**: the joint density is a strictly better-specified density, worth ≈2.1 nats/bar at
  equal skill (§4.6), while the anchor is accumulated in horizon space at `runner.rs:1318`. Same
  units, different densities. §4.7 retracts the claim in full and states the consequence.

### 1.2 The mean is still decoded in horizon space, deliberately

The four mean coordinates go through `decode_joint` unchanged, so
`high ≥ max(open, close) ≥ min(open, close) ≥ low` still holds bar by bar and the emitted
forecast is still a valid candle. Reinterpreting the mean ROWS as coefficients directly would
have destroyed that invariant (a "range" in coefficient space is not a range), and by linearity
`W(y − ŷ) = Wy − Wŷ`, so rotating the decoded candle is the same bijection at one GEMM instead
of two.

---

## 2. The transform shipped, and its orthonormality evidence

Two non-identity maps ship; `cumulative` is the identity and the control.

| basis | fp32 `max\|WᵀW − I\|` | fp64 defect | rank | `tr(W·C·Wᵀ)` |
|---|---|---|---|---|
| `cumulative` | `0.000e0` | `0.000e0` | 192 | 18528 (= Σh, exact) |
| `haar` | `1.788e-7` | `3.33e-15` | 192 | 18528 |
| `dct` | `1.788e-7` | `1.29e-14` | 192 | 18528 |

`1.788e-7 = 2^-22.4`, three orders of magnitude under the shipped refusal threshold
`ORTHONORMALITY_TOLERANCE = 2.5e-5` (which is `192·2^-24`, the worst plain error bound for a
192-term fp32 dot product of unit-norm rows, with a factor of two of headroom). DEMONSTRATED
four independent ways:

1. `shipped_maps_are_orthonormal_in_fp32_at_the_production_shape` — the Gram matrix in fp32 at
   192, plus an fp64 check that it is orders tighter, i.e. that the fp32 number is arithmetic
   and not a construction error.
2. An independent NumPy recomputation of the same Gram defects and ranks (192, all three) and
   of `tr(W·C·Wᵀ) = 18528` exactly, from a separately written implementation.
3. `every_basis_preserves_the_residual_norm_at_the_production_horizon` — the operational form:
   `‖W·r‖ = ‖r‖` to `< 1e-5` relative on real fp32 residuals at `[16, 4, 192]`. This proves the
   GEMM that consumes the matrix, not just the matrix.
4. `BasisTransform::new` REFUSES a map whose fp32 defect exceeds the tolerance, and
   `construction_refuses_a_map_that_is_not_a_bijection` proves the refusal fires.

Full rank follows from orthonormality: `WᵀW = I` makes `W` invertible, so the map restricts
nothing. `a_rotated_objective_leaves_no_head_row_untrained` closes the loop empirically —
under both `haar` and `dct`, ZERO of the `2·4·pred_len` head-output rows receive exactly-zero
gradient. That is the direct refutation of the `basis:8:8` failure mode.

### 2.1 192 is not a power of two

`192 = 3·2^6`. **No padding and no projection.** The Haar cascade halves
`192 → 96 → 48 → 24 → 12 → 6 → 3` and meets an odd length exactly once, at 3, where it applies
an explicit orthonormal 3×3 completion:

```
[1, 1,  1]/√3      (the scaling function, which continues)
[1, 0, −1]/√2
[1,−2,  1]/√6
```

Those three are mutually orthonormal by inspection, so the composition of orthonormal maps is
orthonormal by construction rather than by tolerance — which is why the fp64 defect is `3e-15`.
The implementation generalizes rather than special-casing 192: an odd length `m` pairs the
first `m − 3` and applies the 3-block to the last three, and
`haar_handles_the_non_dyadic_length_without_padding_or_projection` proves orthonormality at
lengths 1, 2, 3, 5, 6, 7, 12, 96, 191 and 192. DCT-II has no length constraint at all.

Row order is fixed so "low index" means "low frequency" in all three bases: coefficient 0 is
the whole-window mean (every entry `1/√192`, asserted).

### 2.2 `cumulative` is a bit-exact identity — and what "bit-exact" precisely means

Two levels, and the distinction is not pedantry.

- **Structural (the one that matters for the control arm).** `self.basis` is `None` under
  `cumulative`, so `losses()` falls through to the identical fused call it made before the knob
  existed. No new buffer, no new branch in a hot path, no new tensor. The pre-existing
  `uniform_horizon_loss_reproduces_the_previous_unweighted_objective` still pins its value, and
  the new knobs are `skip_serializing_if` at their defaults, so a control manifest serializes —
  and therefore digests — byte-identically. A checkpoint written before today still loads.
- **Numerical, term for term.** `cumulative_target_basis_is_a_bit_exact_identity_on_the_objective`
  drives the coefficient path with the identity transform and asserts BIT equality of: the
  rotation against its own input, the coefficient prior `½·ln(diag(W·C·Wᵀ))` against
  `half_log_horizon`, the per-element NLL against `nll_elements`, the mask fold against
  `mask·w`, the weighted numerator, the denominator, and the MSE. The reduced NLL SCALAR is bit-
  equal once the reference's numerator is made contiguous, and differs by exactly ONE fp32 ulp
  otherwise (`1.7424697875976563` vs `1.7424696683883667`, 6.8e-8 relative). The cause was
  measured, not assumed: `gaussian_nll`'s numerator tensor is NON-contiguous (its log-scale term
  is a strided view of the head), so ATen's cascade sum visits it in a different order. The test
  asserts the contiguous equality AND bounds the strided difference at one ulp, so a regression
  that grows it fails.

---

## 3. Masking: the one place the rotation is genuinely hard

A coefficient is a weighted sum over the WHOLE horizon window, so an origin whose window is
partly unobserved has no defined coefficient vector. The mask stops being diagonal in the
rotated basis, and this is not a corner case that can be waved at.

**The mask structure was checked, not assumed.** `mask[t,h] = 1 ⟺ t + h < written`, a strict
PREFIX in `h`, so there are exactly 193 distinct patterns and per-pattern truncated maps
(DCT-II of size `K`, Haar of size `K`) would be exactly correct and precomputable in ~14 MB.

**That exactly-correct option is nonetheless unimplementable here**, and the reason is a hard
constraint rather than a preference: grouping rows by `K` is a DATA-DEPENDENT shape partition,
and the training step is captured as one CUDA graph — a capture records shapes. The same
constraint bars gradient accumulation and chunking. So the shipped rule is:

> **The rotated objective scores complete windows only.** `rowmask = min_h mask[·,h]`, and the
> residual is additionally multiplied by the per-bar mask BEFORE the rotation so an unobserved
> bar contributes exact zero rather than `0 · large` after a 192-term sum. On a complete window
> the mask is exactly 1, so both multiplies are the identity on the bits and §2.2 survives.

**The price is bounded analytically, not guessed.** An incomplete window requires
`written − 1 − t < 192`, i.e. only origins within 192 bars of the row's owned end — at most
`⌈192/16⌉ = 12` of the row's 375 origins, and ZERO for any row whose owned span reaches
`seq_len + pred_len`. Upper bound **3.2% of scored origins**, attained only if every row were
short; far below `TemporalSplit`'s 12.9%-of-rows purge price. Dropped elements leave BOTH the
numerator and the denominator, so the reported NLL stays a per-element weighted mean — one
objective, one scalar, nats per bar, still comparable to 2.3947663. The MEASURED share is
recorded in the statistics artifact (`whitening_complete_share`) and printed in the new chart's
title, so the 3.2% bound is replaced by a measurement the first time the fit runs.

`the_rotated_objective_ignores_every_incomplete_horizon_window` proves a TWO-SIDED invariance:
poisoning every unobserved entry with `1e6` (located from the mask itself, not assumed) moves
the objective by ZERO bits, and poisoning an OBSERVED entry MUST move it — so the test cannot
pass by ignoring everything. Rejected alternative, explicitly: zero-filling the unobserved tail
and rotating anyway injects a false zero return into all 192 coefficients.

**Independent corroboration that the redundancy is worth attacking**, from `TemporalSplit` by a
completely different route: all supervised origins are congruent to 15 mod 16, and for a
non-lattice origin `A` with `r = (A−15) mod 16`, `cum(A→A+h) = cum(A−r→A+h) − cum(A−r→A)` with
both right-hand terms supervised from the lattice origin `A−r`. The 192 cumulative targets from
one origin carry the information of 192 single-bar returns. The redundancy this reparametrizes
is ALGEBRAIC, not merely statistical.

---

## 4. Whitening and the weights

### 4.1 Where the statistics live and how they are authenticated

One JSON artifact, `BasisStatistics`, written by
`basis-stats-timexer-segment --artifact <path>` and named by `--basis-stats <path>` on the arm.
It carries a format stamp, the basis, `pred_len`, the fit partition of each vector, the
whitening scales, the complete-window share, `ρ̂`, `β̂`, the bar counts, and a `sha256` over its
own content. `BasisStatistics::authenticate` refuses on format, on basis, on `pred_len`, on
length and on digest, and NAMES which one failed;
`statistics_authenticate_on_format_basis_shape_and_digest` proves all four refusals. The model
loads and authenticates it in `CausalPatchModel::new`, beside the config validation it
completes — so a mispaired artifact dies before the corpus loads, not after the first optimizer
step. The vectors are constant buffers, never varstore variables, so they are safe inside the
captured CUDA graph and no saved tensor moves.

### 4.2 Fitted on training origins only, with the leakage test

The whitening scales are a second moment of the TARGETS: no head, no forward pass, nothing that
could depend on a checkpoint, so the same corpus produces the same scales for every arm and
every seed. `fit_statistics` tags the training walk `FitPartition::Training` and the calibration
walk `FitPartition::Calibration` beside the reference lists they read, and
`BasisStatistics::fit` REFUSES any other pairing.
`whitening_refuses_every_partition_but_training` proves the refusal for `Calibration`,
`Validation` and `Test` on the whitening side (the error contains the word "leakage" and names
the offending partition) and specifically refuses `Validation` — the population checkpoint
selection reads — on the SNR side. `target_moments_measure_the_rms_and_drop_incomplete_windows`
proves the accumulator reproduces a hand-computed fp64 rms and that an incomplete window
contributes nothing.

Whitening replaces the analytic persistence prior `diag(W·C·Wᵀ)` (with `C_hg = min(h,g)`) as the
centre of the tanh ±4 soft cap on the log scale. Under `cumulative` that analytic prior is
exactly `h` — `identity_prior_variance_is_the_horizon_itself` asserts equality as integers — so
`half_log_prior` is bit-for-bit `half_log_horizon`.

**Stated honestly:** with per-coefficient LEARNED scales, whitening is a reparametrization of
the log-scale offset and changes the geometry only through the cap window's centring. The thing
that actually changes the metric is the per-coefficient WEIGHT. Whitening earns its place
because the analytic random-walk prior is wrong for market-neutral returns at long horizons, so
a mis-centred window makes the model spend cap budget it should not have to; it is not the lever.

### 4.3 The weights

`uniform` (default) is equal weight per coefficient — equal weight per unit of target variance
once whitened, instead of per near-duplicate row. `snr` is `w_k ∝ max(ρ̂_k², 1/n_k)`, normalized
to mean 1. The floor is the estimator's own null level (`E[ρ̂²] = 1/n` under no signal), because
a zero weight is `--horizon-loss cutoff` on the coefficient axis and cutoff is a rejected arm.
`snr_weights_are_mean_one_and_never_delete_a_coefficient` pins mean 1 to `1e-12`, strict
positivity, sign-invariance in `ρ̂`, and refusal when no `ρ̂` was measured.

Composing a rotation with a per-HORIZON weighting or with a rank restriction is REFUSED at
config validation (`a_rotated_basis_refuses_a_horizon_weighting_and_a_rank_restriction`):
`w_h` would multiply coefficient `h`, which is not horizon `h`, and `--horizon-mean basis` is
the rank restriction whose composition would make neither effect attributable.

### 4.4 Why the rotation should help at all — the concentration measurement

DEMONSTRATED, on the persistence prior (independent NumPy recomputation), share of
`tr(W·C·Wᵀ) = 18528` carried by the leading coefficients:

| basis | coeff 0 | first 4 | first 8 | first 16 |
|---|---|---|---|---|
| `cumulative` | 0.0001 | 0.0005 | 0.0019 | 0.0073 |
| `haar` | 0.6684 | 0.9079 | 0.9540 | 0.9771 |
| `dct` | 0.6684 | 0.9428 | 0.9732 | 0.9871 |

This is the quantitative statement of the defect. Today's objective spends 192 equal gradient
shares on rows whose variance lives in **8 directions carrying 97.3%** of it. Under a rotation,
equal weight per coefficient is equal weight per direction.

### 4.5 `FutureTeacher`'s mean reversion separates the two candidate mechanisms cleanly

Input from job 5453: 191 of 192 horizons show within-window mean reversion strong enough that
the measured cross-bar covariance is more negative than a one-bar IC ceiling of 0.1666 permits.
`Main` asked which of two mechanisms then owns the long-horizon amplitude defect — this
rotation, or a measured per-horizon `√h` scale correction. The question is answerable on the
host, and the answer is clean. Modelling the one-bar series as AR(1) with `φ < 0` (autocovariance
`φ^|k|`) and forming `C = A Γ Aᵀ` with `A` the cumulative-sum matrix — DEMONSTRATED, independent
NumPy, fp64 orthonormality defect `1.3e-14`:

| `φ` | VR(64) | VR(192) | dct c0 | dct first 8 | haar first 8 |
|---|---|---|---|---|---|
| 0.00 (random walk) | 1.0000 | 1.0000 | 0.6684 | 0.9732 | 0.9540 |
| −0.05 | 0.9062 | 0.9052 | 0.6682 | 0.9727 | 0.9536 |
| −0.15 | 0.7427 | 0.7403 | 0.6679 | 0.9717 | 0.9526 |
| −0.30 | 0.5440 | 0.5403 | 0.6673 | 0.9701 | 0.9510 |

`VR(h) = Var(cum_h)/(h·Var(1 bar))`. Reversal moves the LEVEL of the target variance by up to
**1.85×** while moving the SHAPE of the coefficient spectrum by **0.3% of one share** — the
first-8 concentration goes 0.9732 → 0.9701 across the entire plausible range. So:

- **The two mechanisms are orthogonal, not competing, and I can say which dominates.** Reversal
  is a per-horizon SCALE (level) defect; near-duplicate concentration is a BASIS (shape) defect;
  the shape is nearly invariant to short-lag reversal. Neither can claim the other's win.
- **Reversal cannot be the main amplitude story.** DEMONSTRATED `β = 0.26594` at h = 192 is a
  3.76× over-amplification. A model that implicitly assumed random-walk `√h` scaling while the
  truth is `√(VR·h)` over-amplifies by exactly `1/√VR`, which at `φ = −0.30` is **1.36×** — i.e.
  `ln 1.36 / ln 3.76 = 23%` of the defect in log terms. Explaining the WHOLE 3.76× needs
  `VR = 1/3.76² = 0.0707`, hence `φ = −0.868`: near-perfect one-bar reversal, which is not a
  market and is not what 5453 measured. **HYPOTHESIS, with the arithmetic shown:** a measured
  per-horizon scale buys at most ≈1.36× of the 3.76×, and the residual 2.8× is not a scale
  defect at all.
- **This raises whitening's value above §4.2's estimate.** The analytic prior is wrong in level
  by `1/VR`, so the tanh cap's centre is off by `½·ln(1/0.54) = 0.31` nats at the long end. That
  is inside the ±4 cap and therefore learnable, but it is a real offset the model currently pays
  for out of its own capacity, and it is exactly what a measured whitening vector removes.
- **Pre-registered, and answerable by step 1 alone:** the measured per-coefficient variance
  spectrum will differ from the random-walk prior in LEVEL and not in SHAPE — `VR(192) < 1`,
  predicted 0.5-0.9, and the DCT first-8 share of measured target variance in [0.94, 0.99]. A
  measured first-8 share BELOW 0.90 would falsify the concentration argument this whole feature
  rests on, and I would rather find that out in 90 s than in a training arm.

### 4.6 What the rotation is worth in nats, and what `FutureTeacher`'s increments do to it

`Main` asked the right question and it deserves a number rather than an argument: if the head is
reparametrized to emit per-bar INCREMENTS at full rank, consecutive targets stop being
near-duplicates, and the near-duplicate argument is this feature's whole motivation. Does the
rotation still add anything?

The quantity that answers it exactly is the **diagonal penalty**: the log-likelihood a
diagonal-covariance Gaussian gives up against the joint, `½·(Σ_k ln diag_k − ln det Σ)/n`, in
nats per bar. It is the total correlation of the target vector, it is exactly what a
count-normalized diagonal NLL cannot represent, and it is exactly what an orthonormal rotation
recovers. DEMONSTRATED, fp64, independent NumPy:

| target space | one-bar model | diagonal penalty | left after dct | left after haar |
|---|---|---|---|---|
| cumulative sums | random walk | **2.1380** | 0.0109 | 0.2576 |
| cumulative sums | AR(1) `ρ₁ = −0.15` | **2.0027** | 0.0101 | 0.2061 |
| cumulative sums | AR(1) `ρ₁ = −0.325` | **1.8668** | 0.0094 | 0.1684 |
| increments | AR(1) `ρ₁ = −0.15` | **0.0113** | 0.0001 | 0.0050 |
| increments | AR(1) `ρ₁ = −0.325` | **0.0555** | 0.0005 | 0.0257 |

**Answer: `Main`'s option 1, with a revised operator, and I am rejecting my own dense GEMM in
the increment world.**

- **Increments consume ~97% of the motivation, not part of it.** 1.867 → 0.0555 nats/bar is a
  **34× reduction**. `FutureTeacher` argued for option 1 on the grounds that increments are not
  white and whitening a correlated increment vector is still motivated; that is true, and the
  arithmetic says it is true at 1/34 the size.
- **What survives is real.** 0.0555 nats/bar is 2.3% of the 2.3947663 anchor and LARGER than the
  0.01-0.06 I had pre-registered this arm would win. It is not zero.
- **But it is not a dense rotation's worth.** For a BANDED covariance the exact whitener is the
  Cholesky factor, and an AR(1) inverse is TRIDIAGONAL, so the exact operator is a **2-tap causal
  filter**: O(n), ≈2 slice-passes instead of 394, ≈0.1 ms instead of +19.7 ms, and it captures
  **100%** of the 0.0555 where DCT captures 99.1%. Job 5471's measured lag structure sets the tap
  count; a bandwidth `b` costs `b+1` taps and stays O(n·b).
- **Recommendation, stated as a decision and not a preference:** if increments land,
  `--target-basis dct|haar` is REJECTED and replaced by a measured banded prewhitener.
  `cumulative` stays the default identity at exactly zero cost, so the control arm is untouched
  in either world. If increments are rejected, the dense rotation is worth **1.87-2.14 nats/bar**
  of removable mis-specification, DCT captures **99.5%** of it against Haar's 88-91%, and the
  code is one `validate` refusal away from either outcome. The fork is cheap to hold open, which
  is the only reason I am not deleting it today.
- **Why DCT and not Haar, now with a number.** 0.0094 left against 0.1684 left: on the cumulative
  covariance the DCT is 18× closer to the KLT than the dyadic Haar cascade. This also retires the
  Haar/192 non-dyadic engineering as interesting-but-pointless — it is proven orthonormal (§2.1)
  and it is the worse basis.

### 4.7 RETRACTION: the reported NLL is not comparable across the basis boundary

This falls out of the same computation and it is the most important thing in this report, so it
is not buried in §7. The mean coefficient prior is `½·ln h` averaged = **2.138** nats/bar in
horizon space and **0.011** under DCT. At equal skill, a basis arm's NLL is therefore ≈2.1
nats/bar lower FOR FREE — purely from modelling the joint, with no forecast improved.

- **My pre-registered prediction 1 (§7) was wrong by two orders of magnitude and is RETRACTED.**
  I predicted −0.01 to −0.06 nats. The honest prediction is −1.8 to −2.1, and almost none of it
  is skill.
- A basis arm's NLL must **NEVER** be compared to 2.3947663 or to the control's 1.987704. The
  anchor has to be recomputed in the same basis or the number is meaningless. `losses()` rotates
  the model's residual, but the persistence anchor is accumulated in horizon space at
  `runner.rs:1318` (`persistence_nll: sums[7] / elements`), so the two are NOT in the same space
  today and the printed comparison at `runner.rs:2135` would be nonsense under a basis.
- **IC and the per-horizon MSE ratio are therefore the ONLY honest cross-basis decision
  metrics.** Checkpoint selection reads NLL, so it stays valid WITHIN an arm and is incomparable
  ACROSS the basis boundary.
- This is the strongest argument against the feature that exists, and it is mine: a change that
  improves the headline scalar by 2.1 nats without improving one forecast is a change that makes
  the scoreboard worse. It is why I am not asking for the training arm at the top of a starved
  queue.

### 4.8 The accepted operator: a 3-tap banded prewhitener that beats the dense rotation 65×

`Main` accepted the revised operator and CUT the dense rotation for the increment world. The
result is better than that, and it is DEMONSTRATED in fp64 rather than proposed: **the banded
prewhitener replaces the dense rotation in BOTH worlds, exactly, and it does not need the head
change at all.**

The construction. `C = A Γ Aᵀ` with `A` the unit lower triangular cumulative-sum matrix, so if
`Γ = L_Γ D L_Γᵀ` then `C = (A L_Γ) D (A L_Γ)ᵀ` and `A L_Γ` is unit lower triangular. Hence the
exact whitener of the CUMULATIVE residual is

```
M = L_C⁻¹ = L_Γ⁻¹ · A⁻¹        z = M·r,  prior_k = ½·ln d_k
```

`A⁻¹` is the 2-tap difference; `L_Γ⁻¹` for an AR(`b`) increment process is exactly `b`-banded
(the innovations algorithm terminates after `b` steps), so `M` is exactly **`b + 2` taps**.

MEASURED, fp64, `n = 192`, independent NumPy:

| target space | `ρ₁` | diagonal penalty | captured by LDLᵀ | leak beyond 2 taps | beyond 3 taps | `det M` |
|---|---|---|---|---|---|---|
| cumulative | −0.325 | 1.8668 | **1.8668** (residual 8.9e-16) | 3.25e-1 | **2.26e-14** | 1.000000000000 |
| cumulative | −0.15 | 2.0027 | **2.0027** (residual 4.0e-15) | 1.50e-1 | **5.98e-14** | 1.000000000000 |
| increments | −0.325 | 0.0555 | **0.0555** (residual 1.1e-16) | **1.11e-16** | 3.19e-17 | 1.000000000000 |
| increments | −0.15 | 0.0113 | **0.0113** (residual 6.6e-17) | **2.08e-17** | 2.69e-18 | 1.000000000000 |

Four properties, each load-bearing, none of them a hope:

1. **Exact, not approximate.** The LDLᵀ factorization captures the diagonal penalty to fp64
   machine zero — 100.0000% against the dense DCT's 99.5% on cumulative targets and 99.1% on
   increments.
2. **Three taps suffice on cumulative targets, two on increments**, and the table shows the
   cutoff as a 13-order-of-magnitude cliff (3.25e-1 → 2.26e-14), so the bandwidth is a
   structural fact to assert in a test, not a tolerance to tune.
3. **`det M = 1` exactly**, because `M` is unit triangular. This is the SAME property
   orthonormality bought me in §1.1 — no Jacobian term, so the reported NLL remains a proper
   negative log density in nats/bar. The scale lives entirely in `prior_k = ½·ln d_k`, which is
   the `half_log_prior` hook the dense fork already ships and tests.
4. **The cost collapses.** 3 taps: `96,000 × 4 × 192 × 3 × 2 = 0.442 GFLOP` (**0.0026%** of
   16.98 TFLOP), ≈3 slice-passes forward and ≈3 backward = 0.44 GB (**0.31%** of 143 GB), hence
   **+0.30 ms** against the dense rotation's +19.7 ms — a **65× cheaper** operator that captures
   strictly MORE. No GEMM, no 296 KB of buffers, three `[1,1,1,192]` tap vectors instead.

**Consequence for the dense fork: it is dominated, in both worlds, and I am saying so rather
than defending code I wrote today.** The dense rotation's remaining merits are that it is
landed, tested and green, and that it is basis-diagnostic — the per-coefficient `ρ̂`/`β̂` chart
(§6) is a measurement instrument, not an objective, and `basis-stats-timexer-segment` keeps
earning its place independent of which operator trains. My recommendation is therefore narrower
than "keep the fork": **keep `basis-stats-timexer-segment` and the `timexer_segment_target_basis`
chart; ship `banded:b` as the objective; leave `dct`/`haar` reachable only as the diagnostic's
basis argument.**

**What blocks shipping `banded:b` today, precisely.** The taps are `L_Γ⁻¹` from a MEASURED
increment autocovariance `γ_0..γ_b`, and `b` itself is set by where the measured partial
autocorrelation dies. That is job 5471's deliverable and 5471 has not started — the queue has
been starved by a foreign PPO tenant for over an hour. Fitting `Γ` needs one training-origin
walk, which is the walk `basis-stats-timexer-segment` already performs and authenticates; the
accumulator changes from per-coefficient second moments to `b + 1` lag products of the increment
series, and every leakage refusal, the sha256 authentication and the `pred_len` pinning carry
over unchanged. **I am not landing a tap vector before the lag structure that determines its
length exists**; a frozen constant we are entitled to write and not entitled to use is exactly
the failure mode `FutureTeacher` and I agreed to refuse for its own scale vector.

**Priced at the MEASURED lag-1, and the drift warning taken seriously.** `Main` supplied
training-population lag-1 of **−0.271 at h = 8 through −0.218 at h = 192**. Substituting those
for the illustrative AR(1) values, fp64:

| measured `ρ₁` | cumulative penalty | increment penalty | VR(192) | leak beyond 2 taps | beyond 3 taps |
|---|---|---|---|---|---|
| −0.271 | 1.9065 | 0.0379 | 0.5753 | 2.71e-1 | **4.59e-14** |
| −0.218 | 1.9474 | 0.0242 | 0.6436 | 2.18e-1 | **5.38e-14** |

So at the real lag-1 the numbers are: **3 taps on cumulative residuals, capturing 1.91-1.95
nats/bar exactly**; 2 taps on increments capturing 0.024-0.038; and VR(192) = 0.58-0.64, which
lands inside §4.5's pre-registered 0.5-0.9 band and is therefore the first independent
confirmation of that prediction.

`Main` also warns that the first moment INVERTS across epochs (a post-training window measures
Cov/Var −0.1097 ± .0263 against the full span's +0.0265 ± .0106) and asks whether the lag
structure drifts too. That is the right question and it changes the design, not just the
caveat: **`b` must be validated on two disjoint spans before it is frozen, and the artifact must
refuse a mismatch rather than average it.** Concretely, the fit walk measures `γ_0..γ_b` on the
earlier and later halves of TRAINING origins separately and refuses to freeze if the implied tap
vectors disagree by more than their own jackknife interval. The design is robust in one specific
way that is worth stating: a WRONG tap value costs only the unrecovered part of 0.024-0.038
nats/bar (it cannot inject bias, because `det M = 1` holds for ANY unit-triangular banded `M`,
so a mis-fitted whitener is a suboptimal metric and never an incorrect density). A drifting
SECOND moment is therefore survivable where a drifting first moment would not be.

**Regime sensitivity, and the design rule that makes it safe.** `Main`'s correction 2 refutes
the monotone-drift premise: a recent post-training window measures `V_h/(h·V_1) = .652` at
h = 64 against held-out full's `.398`, a 64% swing in the second moment we were about to freeze.
So the lag-1 behind it is not constant either, and a tap vector fitted on one span is exposed.
The exposure is bounded, and the bound is derivable rather than hoped for:

- **A wrong tap can never make the objective improper.** `det M = 1` for ANY unit-triangular
  banded `M`, so the reported number stays a proper negative log density in nats/bar whatever
  the taps are. A mis-fit costs metric quality, never correctness — unlike a frozen per-horizon
  aggregation constant, which biases the forecast itself.
- **Shrinking the tap toward zero is monotone-safe.** The whitener's recovered penalty is
  increasing in `|ρ̂₁|` for any `ρ̂₁` between 0 and the true `ρ₁` of the same sign, and `ρ̂₁ = 0`
  is exactly the identity. So the correct freezing rule is **the MINIMUM `|ρ̂₁|` across the
  disjoint fit spans, not their mean** — it recovers less on the strong-reversal regime and
  cannot over-whiten on the weak one. Fitting the mean is the one choice that can be worse than
  doing nothing.
- **Priced:** freezing at the conservative end of the measured `−0.271 … −0.218` range recovers
  0.0242 of the 0.0379 nats/bar, i.e. 64% of the available increment-space penalty, with zero
  downside exposure. On CUMULATIVE residuals the same 3-tap operator recovers 1.9474 of 1.9065
  — the cumulative penalty is dominated by the difference operator, which is `A⁻¹` and carries
  NO fitted parameter at all, so **the large win is parameter-free and only the last 2% of it is
  regime-exposed.** That is the strongest single argument for this operator over the dense
  rotation, which has 36,864 fitted-basis entries and recovers less.



---

## 5. Cost, exactly

Production shape: 96,000 scored origins/step, 4 channels, 192 horizons. One slice-unit is
`[tokens, 1, 192]` fp32 = 18.4 M elements = 73.7 MB; `FuseLoss` measured ≈0.05 ms per
slice-pass on this card (147-kernel composed chain, 337 slice-passes, 24.85 GB/step at
1635 GB/s = 91% of the streaming roof).

| quantity | control | `dct`/`haar` arm | delta |
|---|---|---|---|
| **parameters** | 27.2 M | 27.2 M | **0** (no head shape change) |
| constant buffers | — | `Wᵀ`, `W∘W`, prior, weight | +296 KB (L2-resident) |
| **FLOP/step** | 16.98 TFLOP | 17.04 TFLOP | **+56.6 GFLOP = +0.333%** |
| **GB/step** | ≈143 GB | ≈172 GB | **+29.0 GB = +20.3%** |
| **ms/step** | 155.1 (v14 fused) | ≈174.8 | **+19.7 ms = +12.7%** |

Rotation arithmetic: `96,000 × 4 × 192 × 192 × 2 = 28.31 GFLOP` forward; the backward is one
more GEMM against `W` (`W` is a constant buffer, so there is NO weight gradient), another
28.31 GFLOP. Traffic: the basis branch is a composed fp32 chain, ≈208 slice-passes forward and
≈250 backward against the fused path's 51 and 13, so ≈394 extra passes × 73.7 MB = 29.0 GB, and
394 × 0.05 ms = 19.7 ms.

**Is a fast transform worth it? No, decisively.** Haar is O(n) and DCT-II O(n log n), so both
delete arithmetic that already costs 0.33% of the step, and both replace ONE read-write pair
over the 295 MB coefficient space with `log2(192) ≈ 7.6` of them — ≈14 slice-passes against the
dense GEMM's 8. The step is bandwidth-bound at 90-96% of the roof, so the fast transform would
cost ≈1.75× the GEMM's traffic to save a rounding error's worth of FLOPs. The dense GEMM is
strictly cheaper in the currency that is scarce.

**What IS worth doing, if the arm survives:** the +19.7 ms is entirely the composed elementwise
chain around the GEMM, not the GEMM. `FuseLoss`'s kernel took the same chain from 15.192 ms to
2.9 ms; a coefficient-space variant of it would recover essentially all of the +19.7 ms. I am
NOT asking for that before the arm is judged — spending a kernel on an unmeasured objective is
the wrong order.

---

## 6. The new report base

`timexer_segment_target_basis`, registered at `shared/src/report.rs:167` (after
`FutureTeacher`'s `timexer_segment_information_ceiling`); the `tui` registry test is green,
36/36. Written by `basis-stats-timexer-segment`, which is the only place per-coefficient `ρ̂` is
measured, and skipped entirely when no `ρ̂` was measured — a panel of 192 NaNs is not a chart.

x = coefficient index (0 = lowest frequency). Four series on one axis, all dimensionless ratios
of one question: `ρ̂²` per coefficient, `β̂` per coefficient, the `1.0` correct-amplitude
reference, and the mean-1 objective weight actually applied. The title carries the basis, the
weighting, the measured fp32 orthonormality defect, the measured complete-window share and the
population. `NaN` where unmeasured, never 0.

This is the one panel where the amplitude defect is DIAGONAL. In horizon space it is a 192×192
object smeared across near-duplicate rows and the per-horizon `β̂` curve reads only its diagonal
in the wrong basis.

---

## 7. Pre-registered predictions

Arm: `--target-basis dct --basis-weight uniform`, analytic prior, against the `--horizon-loss
inv-sqrt` control at MATCHED steps. Signs and magnitudes fixed before the run exists. All
HYPOTHESIS.

**DECIDABILITY LEDGER, stated before the predictions rather than after them.** `Main` ruled
that every IC-based threshold in the batch is undecidable until the `band` fix lands, because
the step-indexed cross-sectional draw yields 100 × 40 = 4,000 windows at IC SE ≈.0164. Mine,
audited against that:

| prediction | channel | draw | SE | status |
|---|---|---|---|---|
| 1 NLL | likelihood | `held-out sample`, 2,048 origins | — | **incomparable across the basis boundary** (§4.7), not an SE problem |
| 2 IC at h=64/128/192 | cross-section | `band` draw | ≈.0164 | **confirmable, not refutable** |
| 3 IC peak location | cross-section | `band` draw | ≈.0164 | **RETRACTED as undecidable** |
| 4 per-horizon MSE ratio | second moment | `held-out full`, 433,721 origins | ≈1e-3 | **decidable now** |
| 5 step ms | timing | own series | ~1 ms | **decidable now** |
| 6 per-coefficient `β̂` | second moment | [70%,80%), 433,721 origins | ≈1e-3 | **decidable now, without a training arm** |
| 7 calibration coverage | uncertainty | `held-out full` | ≈7e-4 | **decidable now** |

So four of my seven survive, all in the second-moment and uncertainty channels, and the kill
criterion (6) is one of them — which is why the only line I am asking for is the 90 s
measurement and not the arm. The two IC criteria are conditional on the anchored placement, and
I would rather say that than have them read as tests.

**STRIDED-DRAW CAVEAT, mandatory on every IC number in this report.** Every `held-out full` IC
figure cited anywhere here — .0655/.0610/.0509 at h=64/128/192, the .1137/.0879/−.0125 step
trajectory, and the SE ≈.0164 the ledger above is built on — is a **strided-draw measurement
pending anchored recomputation**, and job 5483 has now measured how much that matters: with
placement as the only difference, within-timestamp IC on `held-out full` moves .0615 → .1621 at
h=1 and .0530 → .1261 at h=64, ratios of 2.07-2.72×, because a mean cross-section width of 10.6
names cannot carry a within-timestamp statistic (demeaning at n ≈ 10 attenuates the quantity
being measured). The anchored draw is 408,239 origins on 245 shared anchors at mean width
1,666.3. The numbers stay in this report as the BEFORE column.

Two consequences for this feature specifically, both good and neither claimed as a win:

- **My retraction of prediction 3 stands, but its reason is now the weaker of two.** I retracted
  it because 0.005 was 0.3 SE of the .0164 draw. The stronger reason is that the LEVEL of the
  quantity moves 2.4× with the draw, so the .0879 → −.0125 collapse it was pre-registered
  against is not yet established as a property of the model at all.
- **Both IC predictions become decidable next session, on the anchored draw**, and they should
  be re-derived against its measured paired SE rather than against .0164 — as a PAIRED
  per-timestamp arm-versus-control difference, per `Main`'s standing rule, never as two
  independently drawn levels.


1. **Held-out sample NLL: LOWER (better) by 1.8 to 2.1 nats — and that is a MEASUREMENT
   ARTEFACT, not skill.** This replaces my first pre-registration of −0.01 to −0.06, which was
   wrong by two orders of magnitude and is retracted in §4.7 with the arithmetic. The mean
   coefficient prior is 2.138 nats/bar in horizon space and 0.011 under DCT, so at equal skill
   the number falls by ≈2.1 for free. **Do not compare it to 2.3947663 or to the control's
   1.987704.** A move outside [−2.2, −1.6] is the informative event: less negative than −1.6
   means the model is failing to exploit the joint it was handed, more negative than −2.2 means
   the derived-σ construction is wrong and prediction 7 should also have fired.
2. **Per-horizon IC at h = 64 / 128 / 192, step 2000: UNCHANGED, `|ΔIC| ≤ 0.008`, with the
   direction POSITIVE and small (+0.002 to +0.008).** The rotation adds no information and IC is
   rank-based, so a large move in either direction would be surprising. Positive because
   equalizing gradient share across whitened directions should stop ~160 low-SNR rows from
   dominating the update direction.
   Same measurability caveat as prediction 3, in the *favourable* direction: 0.008 is under one
   SE of the current draw (≈.0164), so this prediction can be CONFIRMED by the existing draw —
   an observed `|ΔIC|` inside the band is consistent with it — but it cannot be REFUTED, since
   a true move of 0.02 would also be inside two SE. Refutation waits on the `band` fix too.
3. **The decisive prediction — RETRACTED AS UNDECIDABLE, and this is the second reason the arm
   is withdrawn.** I pre-registered: against job 5428's DEMONSTRATED collapse (h=16: .1137 at
   2000 → .0086 at 4000; h=64: .0879 → −.0125), `IC(h=64)` at step 2500 ≥ its step-2000 value
   minus 0.005, i.e. the post-2000 decay at least halved, with `IC(h=16)` at 2500 ≥ 0.06. The
   rationale stands and is still the most interesting question in the batch: if the collapse is
   the objective overfitting ~160 low-SNR near-duplicate directions, flattening weight per unit
   of variance should delay it; if it is sample exhaustion (`TemporalSplit`: 6.62 passes over
   98.5% of distinct outcomes by step 2000), the basis will not move it at all.
   **But the threshold is not measurable with the draw that produces the number.**
   `TemporalSplit`'s occurrence-1 defect — now ruled LIVE by `Main` and reproduced three
   independent ways — establishes that the cross-sectional draw yields 100 × 40 = 4,000 windows
   at IC SE **≈.0164**, against an anchored placement of the SAME population reaching
   128 × 256 = 32,768 windows at SE ≈.0035. My threshold of 0.005 is **0.3 SE**. So no basis arm
   can decide this until the `band` fix lands. Combined with prediction 1 being unusable across
   the basis boundary (§4.7), the dense arm has **no decidable primary metric today** — which is
   an independent reason for the withdrawal in §8, arrived at from the measurement side rather
   than the cost side.
4. **Per-horizon MSE ratio: WORSE at the short end, unchanged to slightly better at the long
   end.** h=1 ratio worse by +0.002 to +0.02 (the fitted optimal gain there is 4.28, i.e. the
   short end is UNDER-amplified, and the coefficient objective applies less direct per-horizon
   squared-error pressure); h=192 ratio 1.02125 → 1.00–1.02. I do NOT predict the long-horizon
   MSE ratio drops below 1 — oracle rescaling only buys 1.0212 → 0.99695 and that ceiling is
   unmoved by a change of coordinates.
5. **Step ms: HIGHER, +19.7 ms (+12.7%) against a v14 fused control at 155.1 ms/step**, 95%
   interval +12 to +26 ms. If the measured delta exceeds +30 ms the pass accounting in §5 is
   wrong and I want to know before reading anything else.
6. **The basis diagnostic (the assignment's own question): the over-amplitude CONCENTRATES.** I
   predict `β̂_k < 0.5` for `k ≤ 8` and `β̂_k → 1` for `k ≥ 64` on the DCT axis, because 97.3% of
   the persistence variance lives in the first 8 coefficients. **If instead `β̂ ≈ 0.27`
   uniformly across all 192 coefficients, the amplitude defect is ISOTROPIC, no reweighting of
   this axis can address it, and `--target-basis` should be rejected on that evidence alone.**
   That is a real, pre-registered kill criterion and it is answerable by
   `basis-stats-timexer-segment` alone, in ~90 s, WITHOUT a training arm.
7. **Calibration coverage stays nominal (0.6827 / 0.9500 ± 0.01).** This is the test that the
   derived-σ covariance construction is right. If coverage degrades, `Σ = Wᵀ diag(τ²) W` is
   wrong and that goes to `Main` over hub, not into a report.

**Joint reject rule, pre-registered:** if (1) is worse AND (2) is negative beyond 2 SE at
h = 64 at step 2000, the arm is rejected like `basis:8:8`, and the finding is that the
objective's degeneracy cannot be fixed by a change of coordinates alone.

---

## 8. Command lines

**NOTHING NEEDS TO BE SUBMITTED. The measurement I was asking for ALREADY RAN: job 5478**,
submitted by `Main` earlier in the session with `--basis dct --training-batches 16
--calibration-batches 8` into `training/runs/timexer-control-4k/gens/1`. **So `Γ` exists on
disk and the banded prewhitener's taps are already measurable — READ 5478's artifact, do not
re-queue this.** What to read out of it, in order:

1. `timexer_segment_target_basis`, per-coefficient `β̂` — prediction 6's kill criterion.
   Concentrated in the low-frequency coefficients confirms the argument; `β̂ ≈ 0.27` isotropic
   refutes it and `--target-basis` should be rejected on that alone.
2. The same panel's measured variance spectrum against §4.5 — LEVEL should move (VR(192) in
   0.5-0.9), SHAPE should not (DCT first-8 share in [0.94, 0.99]).
3. `whitening_complete_share` against §3's 3.2% analytic bound.
4. The whitening vector itself: it is the measured per-coefficient second moment, which is what
   `Γ` and hence `L_Γ⁻¹` are derived from. **This is the input that unblocks `banded:b`**, and
   it removes the "5471 never ran" blocker stated in §4.8 — the lag structure is recoverable
   from this artifact without a new job.

**Two corrections to the command line I originally handed over**, both from `Main`, both
recorded because they cost a lease elsewhere tonight:

- **`/var/tmp/tb0_vNN` is a BINARY, not a data directory.** My `--data-dir /var/tmp/tb0_v17`
  was wrong and was the fourth instance of that exact substitution in one session across three
  agents. `--data-dir` defaults to `crate::data::ingest::bars_dir()`
  (`shared::paths::DATA_PATH/bars`), which is what this pass wants, so the flag should simply be
  OMITTED rather than pointed anywhere.
- Job 5478 supersedes the submission, per the note above.

For the record, the corrected form of the line, should it ever need re-running (e.g. for
`--basis haar`, which §4.6 shows is the worse basis and is therefore only diagnostic interest):

```bash
mlq submit --name timexer-basis-stats-dct --max-parallel-runs 1 --time-limit 10m \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh basis-stats-timexer-segment \
  --checkpoint training/runs/timexer-control-4k/weights/best \
  --artifact training/runs/timexer-control-4k/timexer-segment-target-basis-dct.json \
  --output training/runs/timexer-control-4k/gens/1 \
  --basis dct --batch-size 256 --training-batches 16 --calibration-batches 8
```

**Step 2, the matched 2,500-step arm: WITHDRAWN by me, not by `Main`.** It is written out below
for the record because a pre-registration that is quietly deleted is worthless, and because it
becomes live again in exactly one world: increments rejected AND the measured spectrum
confirming §4.5. Three reasons it is withdrawn: (a) §4.7 — its headline NLL would improve ≈2.1
nats for free and its only honest metric is IC, which prediction 2 says moves `|ΔIC| ≤ 0.008`,
i.e. under one wide-draw SE of 0.011; (b) §4.8 — a 65× cheaper operator captures strictly more,
so a lease-slot spent on the dense arm measures a configuration we should not ship; (c) the
queue has one foreign PPO tenant at max-parallel-runs 6 and 5471, which sets `b`, has not
started. Timing, for the record: 2500 × 174.8 ms = 437 s + ~30 s startup + 3 evaluations at
1123 ms → **471 s**, and 601 s under 5428's measured 1.29× contention factor, i.e. at the edge
of a 10-minute lease.

```bash
# WITHDRAWN - do not submit; see §4.7 and §4.8.
mlq submit --name timexer-dct-2500 --max-parallel-runs 1 --time-limit 10m \
  --cwd "$PWD" -- ./trading_bots/run-release-cuda.sh train-timexer-segment \
  --run timexer-dct-2500-20260908 \
  --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 --min-history 256 \
  --batch-size 256 --optimizer polar-express --preview-patience 3 --x0-lambdas disabled \
  --target-basis dct \
  --eval-every 1000 --max-steps 2500
```

Differing flags versus the `inv-sqrt` control: `--target-basis dct` replaces
`--horizon-loss inv-sqrt` (they are refused together, §4.3), and `--max-steps 2500`.
`--basis-stats` is deliberately ABSENT: the analytic prior and uniform weights are the clean
first arm, and adding measured whitening plus SNR weights in the same arm would make neither
attributable.

Read step 1 on completion, in this order: (1) `timexer_segment_target_basis` `β̂` against
prediction 6 — isotropic `β̂` kills the concentration argument outright; (2) the same panel's
measured variance spectrum against §4.5's table — LEVEL should move, SHAPE should not; (3) the
measured complete-window share against §3's 3.2% analytic bound.

---

## 9. Verification performed

- `./torch-env.sh cargo check -p trading_bot_0 --tests` — **0 errors**.
- `./torch-env.sh cargo check -p trading-bot-tui --tests` — **0 errors**.
- `./torch-env.sh cargo test -p trading_bot_0 target_basis` — **14 passed, 0 failed** (9 in
  `target_basis`, 4 in `model`, 1 in `reports`).
- `./torch-env.sh cargo test -p trading-bot-tui --tests` — **36 passed, 0 failed**, including
  the bidirectional report-base registry test.
- `./torch-env.sh cargo test -p trading_bot_0 timexer_segment` — 165-166 passed, and the one
  intermittent failure is FIXED at its source. See §9.1.
- Independent NumPy recomputation, from a separately written implementation, of every
  orthonormality, rank, trace and concentration number in §2 and §4.4; the reversal spectrum in
  §4.5; the diagonal penalties in §4.6; and the banded exactness, bandwidth cliff and `det M`
  in §4.8.

**Not verified, and cannot be from here:** every number in §7, the +19.7 ms of §5, the +0.30 ms
of §4.8, and the measured complete-window share. Those need the GPU.

### 9.1 The x0 bit-exactness failure was an RNG-isolation race, and the assertion was not touched

`model::disabling_the_x0_injection_removes_its_parameters_and_its_kernel` failed with
`left 3.96875, right 0.0`, and `Main` read it as unobserved horizon entries entering through the
rotation — the same defect class as the earlier `30215596032 vs 0.7747`. It is not. Evidence,
in the order it decided the question:

1. Three consecutive runs of `cargo test -p trading_bot_0 --lib timexer_segment` on IDENTICAL
   code gave **FAILED / ok / ok**, and the test passes 1/1 in isolation every time. A
   deterministic contamination cannot do that.
2. `torch::test_rng`'s module doc states the mechanism exactly: a concurrent `manual_seed`
   REWINDS the one process-global generator a seeded test is reading, and a concurrent DRAW
   advances it. The x0 test seeds both of its two builds and compares them bit for bit, so it is
   the maximally sensitive victim.
3. `probe.rs:1250` called `tch::manual_seed(3)` with **no guard**. That is the rewind.

Fixed at the source in five tests, none by relaxing anything: `probe.rs`
`the_paired_gap_is_measured_on_one_common_set_of_timestamps` takes `test_rng::exclusive()`
(it seeds); `reports.rs the_lr_trajectory_panel_...` and `compute.rs`
`polar_express_routes_...`, `every_causal_patch_parameter_...`,
`the_mlp_down_matrices_...`, `the_realized_trajectory_...` take `test_rng::shared()` (they
construct models, which draws, and do not care what they get — `shared()` costs no
parallelism). The x0 assertion, its tolerance and its scope are byte-identical to before, and
`cumulative`'s bit-exactness test (§2.2) never failed at any point.

## 10. Files

- `trading_bots/src/torch/timexer_segment/target_basis.rs` — NEW. The maps, the orthonormality
  measurement, the coefficient prior, the moment accumulators, the authenticated artifact, the
  device-resident transform, the corpus fit driver, 9 tests.
- `trading_bots/src/torch/timexer_segment/model.rs` — 3 config knobs + validation, the
  `basis: Option<BasisTransform>` field and its construction, the 5-line dispatch at the top of
  `losses()`, `basis_losses`, the derived-σ branch in `output()`, 4 tests.
- `trading_bots/src/torch/timexer_segment/reports.rs` — `write_target_basis` + 1 test, appended
  at EOF.
- `trading_bots/src/torch/timexer_segment/runner.rs` — `BasisStatsArgs` + `basis_stats`,
  appended at EOF.
- `trading_bots/src/main.rs` — one `Commands` variant, one dispatch arm.
- `shared/src/report.rs` — one base, `timexer_segment_target_basis`.
- `trading_bots/src/torch/timexer_segment/mod.rs` — `pub mod target_basis;`.
- `trading_bots/src/torch/timexer_segment/probe.rs`, `compute.rs` — RNG guards only, §9.1. One
  `test_rng::exclusive()` and five `test_rng::shared()`; no assertion, tolerance or scope
  changed. `probe.rs` is `LatentProbe`'s file and the edit was announced to it over hub.
