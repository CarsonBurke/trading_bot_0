# Two-sided amplitude calibration and the training-time amplitude prior

Owner: GainTwoSided. Scope: (1) remove the `gain <= 1` clamp from the post-hoc mean calibration
and replace it with an estimation-error bound, (2) re-verify the invariants under amplification,
(3) implement and pre-register the training-time amplitude prior. All 192 horizons retained; no
trunk change; no horizon truncation.

Tags: **DEMONSTRATED** = measured here or read out of a stored artifact.
**[INFERENCE]** = derived from stored measurements but not itself measured.
**HYPOTHESIS** = a mechanism claim, not yet falsified on a real run.

Every number below with `Var(f)`, `Cov(f,y)`, `beta`, `SE` or `amplitude cost` in it is read from
`training/runs/timexer-control-4k/mean-calibration-step3000.json`, the artifact job 5400 wrote.
Note the filename lies about the step: the paired checkpoint is
`weights/best`, and its manifest reads `step 2000`, `best_step 2000`, selection
`min held-out sample objective-weighted NLL (horizon-loss=uniform)`. DEMONSTRATED. Every
`beta` in this report is therefore a step-2000 quantity fitted on the earlier chronological half
of the held-out full split, which is what that run had. The re-fit run below moves the fit onto
the reserved partition; the shape is not expected to move, the contamination caveat is.

---

## 1. Verdict up front

**The `beta > 1` short end is REAL, not a denominator artifact. DEMONSTRATED, three independent
grounds:**

1. *Estimation.* `SE(ln beta_hat) = 0.0257` at h=1 (`SE = 1/sqrt(weight)`,
   `weight = n*rho^2/(1-rho^2)`, the inverse delta-method variance of `ln beta_hat`).
   `ln 4.2823 = 1.4544` sits **56.6 standard errors** above 0. A 4.28 that were noise would need
   an SE two orders of magnitude larger than the one the data reports.
2. *The denominator is small but not degenerate.* `Var(f)/P = 3.788e-4` at h=1 - 1/85 of the
   h=64 value - but `weight = 1514` there, and `rho = 0.0833` is the LARGEST correlation in the
   short band, not a vanishing one. The ratio is small because `Var(f)` is small, not because
   `Cov(f, y)` is noise: `Cov(f,y)/P = 1.622e-3` at h=1 is the same order as at h=3..7.
3. *Economics.* The amplitude error at h=1 costs `4.081e-3` of the close-channel persistence MSE,
   which is **31% of what the h=192 error costs** (`1.300e-2`). Clamping the gain to 1 forfeits
   all of it. Summed over h=1..18 the shrinkage-only clamp forfeited `0.02115` of persistence MSE
   against the `2.05090` the shrinkage half already recovered - a **1.03% enlargement of the whole
   calibration's effect**, concentrated at the horizons whose breakeven cost is 0.1-1.2 bps/side.

**The bound that replaced the clamp**, shipped in `MeanCalibration::fit`:

```
g_h = min( fitted_h , max( 1 , exp( ln beta_hat_h - 3*SE(ln beta_hat_h) ) ) )
```

Shrinkage is applied exactly as the roughness-penalized WLS fits it, with no ceiling at all.
Amplification is applied only as far as the horizon's OWN 3-sigma lower confidence limit
proves it, and a horizon whose `beta_hat` is within three SEs of 1 - or which carries no
measurable amplitude (`weight = 0`) - is pinned to the identity. `CALIBRATION_FORMAT` is now
`timexer-segment-mean-gain-v2-close-anchor-two-sided-log-roughness-penalized-wls`, so a `v1`
shrinkage-only artifact fails format authentication instead of being silently reinterpreted as a
two-sided curve that happens to sit below 1.

**The prior and the post-hoc gain are complementary, not redundant.** [INFERENCE] The prior
cannot fix the short end - at `lambda = 0.0035` it moves `beta` at h=1 from `4.2823` to `4.3422`,
very slightly WORSE - because its equilibrium shrink `1/(1+4*lambda*h)` is `0.986` there. It is a
long-horizon instrument. The two-sided post-hoc gain is the only instrument that touches h < 20.
Shipping one is not a reason to skip the other.

---

## 2. The table the denominator question is answerable from

`Var(f)` and `Cov(f,y)` are reported as shares of the same close-channel persistence MSE `P`, so
they sit on one axis and are directly comparable with the amplitude cost. `rho` and
`sd(f)/sd(y)` are derived from them and from the stored offset ceiling (`E[y]^2/E[y^2] <= 1.0e-4`
at every horizon, so `Var(y)/P = 1` to four decimals everywhere). `v1` is what the
shrinkage-only clamp applied; `v2` is what the two-sided bound applies. DEMONSTRATED except the
`v2` column, which is [INFERENCE] - see §3.

| h | `Var(f)/P` | `Cov(f,y)/P` | `beta_hat` | `SE(ln b)` | `rho` | `sd(f)/sd(y)` | ceiling | `v1` | `v2` | amplitude cost / P |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `3.7879e-04` | `1.6221e-03` | `4.2823` | `0.0257` | `0.0833` | `0.0195` | `3.9647` | `1.0000` | `3.9647` | `4.081e-03` |
| 2 | `6.1210e-04` | `1.6716e-03` | `2.7309` | `0.0317` | `0.0676` | `0.0247` | `2.4830` | `1.0000` | `2.4830` | `1.834e-03` |
| 3 | `7.1652e-04` | `1.5853e-03` | `2.2125` | `0.0362` | `0.0592` | `0.0268` | `1.9847` | `1.0000` | `1.9847` | `1.053e-03` |
| 4 | `9.4191e-04` | `1.9454e-03` | `2.0654` | `0.0338` | `0.0634` | `0.0307` | `1.8661` | `1.0000` | `1.8661` | `1.069e-03` |
| 5 | `1.1855e-03` | `2.4669e-03` | `2.0809` | `0.0299` | `0.0716` | `0.0344` | `1.9024` | `1.0000` | `1.9024` | `1.385e-03` |
| 6 | `1.3875e-03` | `2.8529e-03` | `2.0562` | `0.0280` | `0.0766` | `0.0372` | `1.8907` | `1.0000` | `1.8907` | `1.548e-03` |
| 7 | `1.5833e-03` | `3.1785e-03` | `2.0076` | `0.0268` | `0.0799` | `0.0398` | `1.8524` | `1.0000` | `1.8524` | `1.607e-03` |
| 8 | `1.9064e-03` | `3.6480e-03` | `1.9136` | `0.0256` | `0.0836` | `0.0437` | `1.7720` | `1.0000` | `1.7720` | `1.591e-03` |
| 9 | `2.2642e-03` | `4.1757e-03` | `1.8443` | `0.0244` | `0.0878` | `0.0476` | `1.7142` | `1.0000` | `1.7142` | `1.614e-03` |
| 10 | `2.5505e-03` | `4.4287e-03` | `1.7364` | `0.0244` | `0.0877` | `0.0505` | `1.6138` | `1.0000` | `1.6138` | `1.383e-03` |
| 11 | `2.9446e-03` | `4.8872e-03` | `1.6597` | `0.0238` | `0.0901` | `0.0543` | `1.5455` | `1.0000` | `1.5455` | `1.282e-03` |
| 12 | `3.3848e-03` | `5.3064e-03` | `1.5677` | `0.0235` | `0.0912` | `0.0582` | `1.4612` | `1.0000` | `1.4612` | `1.091e-03` |
| 13 | `3.8679e-03` | `5.7418e-03` | `1.4845` | `0.0232` | `0.0923` | `0.0622` | `1.3848` | `1.0000` | `1.3848` | `9.079e-04` |
| 14 | `4.4596e-03` | `6.0889e-03` | `1.3654` | `0.0235` | `0.0912` | `0.0668` | `1.2726` | `1.0000` | `1.2726` | `5.953e-04` |
| 15 | `5.0297e-03` | `6.3438e-03` | `1.2613` | `0.0239` | `0.0894` | `0.0709` | `1.1739` | `1.0000` | `1.1739` | `3.433e-04` |
| 16 | `5.8169e-03` | `7.0649e-03` | `1.2145` | `0.0231` | `0.0926` | `0.0763` | `1.1332` | `1.0000` | `1.1332` | `2.677e-04` |
| 17 | `6.5589e-03` | `7.4769e-03` | `1.1400` | `0.0232` | `0.0923` | `0.0810` | `1.0634` | `1.0000` | `1.0634` | `1.285e-04` |
| 18 | `7.2731e-03` | `7.9225e-03` | `1.0893` | `0.0230` | `0.0929` | `0.0853` | `1.0166` | `1.0000` | `1.0166` | `5.798e-05` |
| 19 | `8.1154e-03` | `8.3451e-03` | `1.0283` | `0.0231` | `0.0926` | `0.0901` | `1.0000` | `1.0000` | `1.0000` | `6.501e-06` |
| 20 | `8.7415e-03` | `8.5272e-03` | `0.9755` | `0.0235` | `0.0912` | `0.0935` | `1.0000` | `0.9773` | `0.9773` | `5.254e-06` |
| 32 | `1.9155e-02` | `1.1962e-02` | `0.6245` | `0.0248` | `0.0864` | `0.1384` | `1.0000` | `0.6267` | `0.6267` | `2.701e-03` |
| 64 | `3.2176e-02` | `1.4021e-02` | `0.4357` | `0.0274` | `0.0782` | `0.1794` | `1.0000` | `0.4373` | `0.4373` | `1.024e-02` |
| 128 | `2.5988e-02` | `8.1533e-03` | `0.3137` | `0.0424` | `0.0506` | `0.1612` | `1.0000` | `0.3100` | `0.3100` | `1.224e-02` |
| 192 | `2.6709e-02` | `8.0736e-03` | `0.3023` | `0.0434` | `0.0494` | `0.1634` | `1.0000` | `0.3018` | `0.3018` | `1.300e-02` |

The two-column decomposition `beta_hat = rho / (sd(f)/sd(y))` is the whole finding:

- **`rho` is nearly flat**: `0.083` at h=1, rising to `0.093` at h=18, `0.086` at h=32, `0.078`
  at h=64, then halving to `0.049` at h=192. Factor 1.9 over 192 horizons.
- **`sd(f)/sd(y)` rises 9.2x monotonically** from `0.0195` at h=1 to `0.1794` at h=64, then
  flattens at `0.16`. Factor 9.2 over the first 64.

`beta_hat` crosses 1 at h=19/20 for one reason and it is not a short-horizon special case: it is
the horizon at which the head's relative forecast dispersion first catches up with its own
correlation. `sd(f)/sd(y) = rho` IS the definition of a calibrated amplitude, so the crossing is
mechanically where those two curves intersect. Short-end under-amplitude and long-end
over-amplitude are the SAME defect - a dispersion profile that grows steeply in `h` while skill
does not - read from opposite sides of the crossing. DEMONSTRATED.

### What it is NOT: the three structural suspects, ruled out

- **Not the `sqrt(h)` decode factor.** DEMONSTRATED by algebra plus the table.
  `f_h = c_h * sqrt(h)` and `sd(y_h) ~ sqrt(h)` in per-bar sigma units, so
  `beta_hat_h = rho_h * sd(y_h)/sd(f_h) = rho_h * (sd(y_h)/sqrt(h)) / sd(c_h)`: the factor
  cancels identically. Worse for the suspect: a raw coordinate with `h`-independent dispersion
  would give a FLAT `sd(f)/sd(y)`, and the measured one rises 9.2x. The head is not being pushed
  around by the `sqrt(h)` prior, it is overriding it - it learns a coordinate whose own
  dispersion grows with `h` on top of the growth the decode already supplies.
- **Not the persistence anchor / any level effect.** The stored offset ceiling
  `E[y]^2/E[y^2]` never exceeds `1.0007e-4` over all 192 horizons, and `beta` is fitted on the
  DEMEANED forecast. No constant, and no anchor mis-centering, can move a demeaned covariance
  ratio by 4x. DEMONSTRATED.
- **Not first-bar candle geometry.** The calibration acts on the close coordinate only; the
  geometry channels are reconstructed as monotone offsets from the calibrated close (see
  `decode_joint`). `beta` at h=1 is a close-channel statistic, computed before any geometry is
  formed. DEMONSTRATED.

### The mechanism I believe. HYPOTHESIS.

One head, two textbook failure modes, separated by SNR per effective observation.

- *Short end, `beta > 1` = under-amplitude = implicit SHRINKAGE.* For a shrunk predictor
  `f = c*mu` with `c < 1`, `beta = Cov(f,y)/Var(f) = c*Var(mu)/(c^2*Var(mu)) = 1/c` exactly. So
  `beta_hat = 4.2823` reads directly as: the h=1 conditional mean is emitted at **23.4% of its
  true size**. That is what weight decay, a finite step budget, and a uniform horizon loss that
  gives the h=1 row 1/192 of the objective do to the lowest-SNR row in the head. The short end
  is not mis-fit, it is *under-committed*.
- *Long end, `beta < 1` = over-amplitude = FITTED NOISE.* For `f = mu + eps` with `eps`
  uncorrelated with `y` out of sample, `beta = Var(mu)/(Var(mu)+Var(eps)) < 1`. Overlapping
  cumulative targets give horizon `h` roughly `n/h` independent observations, so with 192 free
  per-horizon output rows (`--horizon-mean free`) the long rows have the least effective data and
  the most freedom. `beta_hat = 0.3023` reads as `Var(eps)/Var(mu) = 2.31`: over two thirds of the
  h=192 forecast's variance is fitted noise.
- The crossing at h ~ 19 is where shrinkage stops dominating fitted noise. Nothing in the design
  pins it there; it is an empirical property of this checkpoint, which is exactly why the curve
  is fitted per horizon rather than parameterized.

Both readings are ordinary regression facts, both are consistent with the measured `rho`
flatness, and the second is independently corroborated: `--horizon-mean basis:8:8`, which cuts
the long end's freedom from 192 rows to 8 basis functions, collapsed `D` from `0.00481` to
`0.00116` - it destroyed the signal along with the noise, which is what a capacity restriction
does when the noise and the signal share a basis.

---

## 3. Why the bound is an estimation-error bound and not a clamp at 1

There is **no MSE justification for a ceiling at 1**. Applying gain `g` to a forecast whose true
amplitude is `beta` changes the cross term to `C(g) = -(beta-g)^2*Var(f)/P`, so the expected
improvement over the identity is `E[Delta(g)] = (2*E[beta]*(g-1) + 1 - g^2)*Var(f)/P`, maximized
at `g = E[beta]` for ANY sampling distribution of `beta_hat`. MSE therefore supplies no ceiling
at all - not at 1, not anywhere - and a `beta` above 1 is an under-amplification that a
shrinkage-only curve simply refuses to collect.

The reason to bound amplification at all is **deployed notional**, which is why the bound is
one-sided. The utility layer sizes positions off the forecast, so a gain of 4 quadruples the
notional the same signal commands. An amplification that is an estimation artifact converts
directly into 4x leverage on noise, while a shrinkage that is an estimation artifact only
under-trades. Asymmetric consequence, asymmetric bound: shrinkage as fitted, amplification only
to the horizon's own `3*SE` lower confidence limit.

**Three sigma, and what it costs.** `3*SE(ln beta_hat)` is `0.069` to `0.109` over h=1..18, so
the ceiling sits `6.7%`-`10.3%` below `beta_hat`. Cost of the bound at h=1: recovered amplitude
cost `4.043e-3` against the unbounded `4.081e-3`, i.e. **99.1% of the available recovery is kept**
and 0.9% is paid for the protection. Over h=1..18 the bound keeps `0.02115` of the `0.02132`
available. It is nearly free because `(beta-g)^2` is quadratic and the bound moves `g` by
<= 10%.

**The `v2` column is [INFERENCE], and here is why the inference is tight.** The applied gain is
`min(fitted, ceiling)`, and I have the `v1` artifact's fitted values only where the old clamp did
not bind. Over the 173 horizons where it did not, the smoothed fit tracks raw `beta_hat` to a
worst relative error of `1.85%` (h=129) and a median of `0.34%`. The `3*SE` ceiling at h <= 18 is
`6.9%`-`10.9%` below `beta_hat`, i.e. **4x to 30x the smoother's own tracking error**, so the
smoothed fit at every one of those horizons will land above the ceiling and the ceiling is what
gets applied. Hence `v2 = ceiling` for h <= 18, `v2 = 1` at h=19 (`beta_hat` within 3 SEs of 1),
and `v2 = v1` for h >= 20. The re-fit run measures it rather than inferring it.

---

## 4. Invariants, re-verified with the clamp gone

Amplification is the direction that could break things, so every invariant was re-exercised with
a curve whose largest entry is the measured `4.2823`, not a shrinkage curve.

- **`a_folded_mean_gain_rescales_the_anchor_without_touching_geometry_ranks_or_weights`**
  (`model.rs`) now folds `[4.2823, 2.7309, 2.2125, 1.0283, 0.85, 0.41, 0.32, 0.01]` - the measured
  short-end shape, a crossing, the middle band, and a near-zero - and asserts, at every horizon:
  candle validity `high >= max(open,close) >= min(open,close) >= low` on the decoded prices
  (`assert_valid_candles`); the close channel equals `gain * uncalibrated close` exactly; every
  channel moves by the SAME anchored shift; the within-timestamp row order is bit-identical; and
  no saved tensor moved (`horizon_scale` is a derived buffer, so the checkpoint stays
  bit-identical). PASSES. DEMONSTRATED.
- **Rank invariance is structural, not tolerated.** A positive scalar cannot reorder the rows of
  one timestamp at one horizon. `IC_INVARIANCE_TOLERANCE = 1e-4` - a thirtieth of the iid IC SE -
  and `calibrate` still **refuses the run** rather than reporting if any within-timestamp
  statistic moves further. Unchanged by this work: the refusal is the same code path, now
  exercised by gains above 1 as well as below.
- **The identity case still writes exactly `1.0`.** `an_uncalibrated_run_writes_a_gain_of_exactly_one_at_every_horizon`
  (`reports.rs`) is extended to also assert the applied gain never exceeds its own ceiling series,
  and that the panel carries all five series (fit-block optimum, untouched-block optimum, applied
  gain, ceiling, identity). PASSES.
- **A horizon with no measurable amplitude is pinned to the identity, not to its neighbours'
  evidence**: `horizons_with_no_calibratable_amplitude_are_carried_by_their_neighbours` asserts
  `amplification_ceiling == 1` and `gain <= 1` there;
  `an_amplification_its_own_standard_error_cannot_prove_is_bounded_to_the_identity` builds a
  population whose `SE(ln beta) = 0.8317` and asserts every ceiling is exactly 1;
  `an_amplification_the_data_proves_is_applied_rather_than_pinned` asserts `gain == ceiling` to
  `1e-9` where the data does prove it. PASSES.
- **`v1` artifacts are rejected, not reinterpreted**: format authentication on the new
  `CALIBRATION_FORMAT`. PASSES.

`cargo test -p trading_bot_0 timexer_segment`: **110 passed, 0 failed, 1 ignored**.
`cargo check -p trading_bot_0 --tests`: **0 errors**. `cargo check -p trading-bot-tui --tests`:
**0 errors**. `cargo test -p trading-bot-tui meta_chart`: the bidirectional report-base registry
test passes with the new `timexer_segment_calibration_moments` base registered. DEMONSTRATED.

### New reporting

`timexer_segment_calibration_moments` (new base, registered in `shared/src/report.rs`) writes
`Var(f)` and `Cov(f,y)` as shares of the same persistence MSE, on their own axis - deliberately
NOT on the gain chart, because a dimensionless gain and a variance share do not share a unit and
a `beta` above 1 is only worth correcting if the variance under it is large enough to matter.
`timexer_segment_calibration_gain` gains a `three-sigma amplification ceiling` series on the same
axis as the applied gain, so the reader can see where the fit wanted more amplification than the
horizon's own error supports. `calibrate` also now prints the fit/score separation as a multiple
of one origin's `pred_len`-bar target reach, so "shares no bar" is legible as a number.

---

## 5. The training-time amplitude prior

### What ships

`--amplitude-prior <lambda>`, default `0` (the control). `CausalPatchModel::amplitude_prior`,
called from `losses()` on the sigma-scaled close mean coordinate `m` the decoder already
materializes:

```
R      = (lambda/(2H)) * sum_h w_h * (1/N_h) * sum_b mask_bh * (m_bh - m_h)^2 ,   m_h = (1/N_h) sum_b mask_bh*m_bh
dR/dm  = (lambda/(H*N_h)) * w_h * mask_bh * (m_bh - m_h)
```

`w_h` is the objective's own horizon weight (exactly 1 under `--horizon-loss uniform`), `N_h` the
count of valid bars at that horizon, `H = pred_len`. `b` ranges over all `[rows, origins]` in the
step - 96,000 origin-columns at batch 256 - because every causal origin is a training example.
The demeaning is numerically almost inert here (the offset never exceeds `1.0e-4` of `E[y^2]`) and
is retained only so the penalty targets DISPERSION rather than level, keeping it orthogonal to
the offset the decomposition already reports.

`R` is added to the NORMALIZED objective, not folded into the NLL numerator, so the reported NLL
stays comparable to the control's only insofar as it genuinely includes the penalty - and
`selection_criterion` stamps `plus mean-amplitude prior (amplitude-prior lambda 0.0035)` into
every manifest, so a penalized checkpoint can never be compared against a control one as if they
minimized the same quantity.

### Why this is not a reparametrization

A fixed output multiplier `k` is undone by the upstream weights within a few hundred steps: the
loss is invariant under `(k, W) -> (k/a, aW)`, so nothing is constrained. That is exactly why
`--horizon-mean basis:8:8` failed as an amplitude intervention - it changed what the head
PARAMETERIZES, not what it is charged for. A penalty on output VALUES has no such invariance:
scaling an upstream weight by `a` scales `R` by `a^2`. The stationary point in the global-rescale
direction is a genuine equilibrium, and it is the one the falsifiers below are computed from.

### Cost. The pre-registered figure was wrong by three orders of magnitude; here is the real one.

The pre-registration costed `0.25 MFLOP` and `0.6 MB` per step, which is `[batch, pred_len]` =
`256*192` elements. But the penalty is charged at **every causal origin**, not once per row:
`256 rows * 375 origins * 192 horizons = 18.43e6` fp32 elements. Charged in `step_cost` as
**22 passes** over that slice - 8 forward (masked multiply's two reads and one write, its
reduction read, the square-multiply's two reads and one write, its reduction read) and 14 backward
(one gradient slice out of each of the two reductions at a read and a write each, the first
moment's broadcast, the two accumulations, and the mask multiply routing one back through the
other):

- **traffic: 1.62 GB/step** against the step's `143.4 GB`, i.e. **+1.13%**. At the measured
  `~1.5 TB/s` achieved bandwidth on the elementwise classes that is **~1.08 ms**, i.e.
  **+0.6% step time** on the 168.0 ms step. HYPOTHESIS (the arithmetic is DEMONSTRATED; the
  realized step time is not).
- **arithmetic: ~74 MFLOP/step** against `16.98 TFLOP` - `4.4e-6` of the step. Free.
- **parameters: zero.** No tensor, no buffer, no optimizer state. The knob is an `f64` in
  `ModelConfig`, skipped from the manifest at `0` so a control checkpoint's authenticating digest
  is byte-identical to what it was before the knob existed - pinned by
  `a_control_config_serializes_exactly_as_it_did_before_the_amplitude_prior_existed`, which
  round-trips the REAL `model` object out of
  `training/runs/timexer-control-4k/weights/best/manifest.json`. DEMONSTRATED.
- Registered in `kernel_classes` only when `lambda > 0`, so an unpenalized arm's class list and
  its `step_cost` describe the same kernels that actually run.

It is also the natural second fusion target after the loss geometry: both reductions read a
tensor the fused loss already holds in registers. Handed to `FuseLoss` as such.

### Verified

`the_amplitude_prior_is_the_pre_registered_mean_energy_and_its_gradient` checks the VALUE against
an f64 host recomputation of the formula above on a masked population, and the GRADIENT
elementwise against `(lambda/(H*N_h))*w_h*mask*(m - m_h)` via autograd, both to
`1e-6*(1+|expected|)` (fp32 against f64; the moment form's cancellation is <0.2% here so fp32
relative error stays at the 1e-7 level). Also asserts `lambda = 0` emits `None` - no penalty
tensor at all, so the control's loss is bit-identical to the pre-knob one rather than numerically
close. PASSES. DEMONSTRATED.

---

## 6. Pre-registered predictions for the prior, before the run

### Choosing lambda from the equilibrium

At the stationary point in the global-rescale direction,
`dNLL/dm + dR/dm = 0` with `dNLL/dm = w_h*mask*(m-y)/(s_h^2 * Q)` and
`Q = 4*sum_bh w_h*mask_bh`, which for `w = 1` and equal per-horizon counts gives
`Q/(H*N_h) = 4`. So the mean is shrunk by

```
shrink_h = 1/(1 + 4*lambda*s_h^2) ,   s_h^2 ~ h   (the close coordinate is in per-bar sigma units)
beta_post_h = beta_hat_h / shrink_h
```

[INFERENCE from the artifact + this algebra]:

| lambda | shrink h=64 | shrink h=192 | `beta` h=1 | h=8 | h=32 | h=64 | h=128 | h=192 |
|---|---|---|---|---|---|---|---|---|
| `0.002` | `0.661` | `0.394` | `4.317` | `2.036` | `0.784` | `0.659` | `0.635` | `0.767` |
| **`0.0035`** | `0.527` | `0.271` | `4.342` | `2.128` | `0.904` | `0.826` | `0.876` | `1.115` |
| `0.006` | `0.394` | `0.178` | `4.385` | `2.281` | `1.104` | `1.105` | `1.278` | `1.695` |

**`lambda = 0.0035` is the central arm**: it is the only value in the bracket that puts the whole
`h >= 32` band inside `[0.8, 1.15]`. `0.002` undershoots (`0.64`-`0.77`), `0.006` overshoots
(`1.10`-`1.70`). `0.002` and `0.006` are the bracket arms if the central one misses.

**Caveat on the direction of the miss.** `s_h` is a LEARNED scale. As the prior shrinks `m`,
residuals grow, the scale head grows `s`, and the effective shrink weakens. So the realized
`beta_post` will sit BETWEEN `beta_hat` and the table above - the predictions are optimistic, and
the bracket exists for that reason.

### Falsifiers, pre-registered

Measured on the same `weights/best`-style selection, same corpus, held-out full split. Baselines
are the step-2000 control numbers in §2, with the report's `-0.0252` gen-2 figure noted where the
brief used it.

1. **`C` at h=192 must collapse toward zero.** Baseline `-0.01300` (`-0.0252` in the gen-2
   full-draw reading). PASS if `|C| <= 6e-3`. Point prediction at `lambda = 0.0035`: `-3e-5`;
   at `0.002`: `-2.3e-4`. Note this is BELOW the pre-registered `-0.006..-0.001` floor - a
   well-tuned lambda can beat that band, and beating it is not a failure. `C` at h=64:
   `-0.01024 -> -2.7e-4` predicted, PASS if `|C| <= 5e-3`.
2. **`D` must be unchanged within +-20%.** `D = Cov^2/(Var*E[y^2])` is scale-free, so shrinkage
   CANNOT move it - only a change in the learned DIRECTION can. Baseline at h=192 from the
   `basis:8:8` comparison: `0.00481`. **REJECT the prior** if `D` collapses the way `basis:8:8`
   did (`0.00481 -> 0.00116`): that is signal destruction, not amplitude repair, and no amount of
   MSE improvement redeems it.
3. **Within-timestamp IC unchanged within 1 iid SE** (`+-0.0028`) at h=64/128/192. Baseline at
   h=192: `0.0509`. A prior that improves MSE by degrading IC is trading away the only
   tradable property.
4. **The mutual self-check, read PER BAND.** Re-fit the post-hoc calibration on the penalized
   checkpoint. PASS if the gain curve rises into `[0.8, 1.0]` for `h >= 32` - the two mechanisms
   then measure each other and agree. But it must STILL read above 1 below h=20 (predicted
   `~4.3` at h=1, essentially unmoved): the prior is a long-horizon instrument and cannot fix
   under-amplitude. A penalized model whose short end came back at `1.0` would mean the prior did
   something other than what its algebra says.
5. **Close MSE ratio at h=192: `1.0219 -> 0.995-0.999` WITHOUT any post-hoc gain.** This is the
   claim that the prior is a real repair rather than a re-parameterization.
6. **Step time within +1.0%** of the control's 168.0 ms (predicted +0.6%), and
   `step_cost.traffic_bytes` up by `1.62 GB` exactly. A larger move means the penalty is not the
   22-pass elementwise op it is costed as.
7. **The control arm must be bit-identical.** `--amplitude-prior 0` emits no penalty tensor and no
   kernel class, and its manifest bytes are unchanged. Already DEMONSTRATED by unit test; a
   re-run of the control that diverges from the recorded curve is a harness bug, not a finding.

**Disagreement protocol.** If the prior improves `C` at h=192 but the post-hoc re-fit on the
penalized model still reports `beta << 1` there, the prior moved the reported NLL without moving
the forecast's amplitude - suspect the scale head absorbing it, and check `s_h` directly. If the
post-hoc re-fit reports `beta ~ 1` but `C` did not improve, the amplitude moved and the MSE did
not, which means `Var(f)` fell faster than `Cov(f,y)` - the prior destroyed signal, and
falsifier 2 should already have caught it.

---

## 7. Command lines

Both are blocked on one prerequisite each and neither can be submitted by me.

**Prerequisite A (both runs): a release snapshot of the current tree.** The in-use `/var/tmp/tb0_v12`
predates the two-sided fit, the reserved-partition rewire, the prior, and the two new report
bases.

```bash
./torch-env.sh cargo build --release -p trading_bot_0 \
  && cp /var/tmp/cargo-build/a0/1fb30e0f0e2b3c/release/trading_bot_0 /var/tmp/tb0_v13
```

**Prerequisite B (calibrate only): `CalPartition`'s corpus change must land.** `calibrate` now
fits on `corpus.calibration_refs` - the reserved `[70%, 80%)` partition - and scores
`corpus.validation_refs`, the held-out full `[80%, 90%)` split. Until that partition yields
origins, `Blocks::spanning` refuses the run with "the calibration partition produced no origins;
there is nothing to fit a gain curve on" rather than silently falling back. That refusal is the
intended behaviour.

### (a) Re-fit the calibration, two-sided, on the reserved partition

```bash
mlq submit --name timexer-calibrate-twosided --priority 1 --max-parallel-runs 1 \
  --time-limit 45m --max-attempts 1 --cwd "$PWD" -- \
  ./torch-env.sh /var/tmp/tb0_v13 calibrate-timexer-segment \
  --checkpoint training/runs/timexer-control-4k/weights/best \
  --calibration training/runs/timexer-control-4k/mean-calibration-v2.json \
  --output training/runs/timexer-control-4k/gens/3 \
  --batch-size 256
```

Pre-registered, for this run specifically:

- Applied gain at h=1 `3.90-4.00` (predicted `3.9647`), h=8 `1.74-1.81` (`1.7720`), h=19 exactly
  `1.0000`, h=64 `0.42-0.45`, h=192 `0.29-0.32`. The curve must remain smooth in `ln h` -
  `effective_dof` in `35-45` over 192 horizons (`39.40` on the contaminated fit).
- Worst `|Delta IC|` over all 192 horizons `< 1e-4`, enforced by refusal.
- Close-only market-neutral ratio at h=64 `<= 0.9940` and at h=192 `<= 0.9980` on the untouched
  block, matching the contaminated run's `0.99357` / `0.99704` to within `5e-3`. A LARGER
  difference than that is the interesting outcome: it means the selection contamination was
  load-bearing, and the honest numbers are the new ones.
- h=1..18 must now IMPROVE rather than sit at the identity: total additional recovery `~0.021` of
  persistence MSE, close ratio at h=1 improving by `~0.004`.
- `timexer_segment_calibration_moments` must show `Var(f)/P` rising monotonically over h=1..64 and
  `Cov(f,y)/P` peaking near h=64. If `Var(f)/P` at h=1 comes back an order of magnitude larger
  than `3.8e-4`, the reserved partition is a different population and every `beta` in §2 needs
  re-reading before the gains are trusted.
- **Refusal is a valid outcome**, not a failure to route around: format authentication, pairing
  mismatch, empty partition, or a moved IC all abort the run by design.

### (b) The penalized training arm, matched to the control

The control is job 5277: `--run timexer-control-4k --features all --layers 8 --d-model 512
--heads 8 --ffn 2048 --min-history 256 --batch-size 256 --optimizer polar-express
--preview-patience 3 --x0-lambdas disabled --max-steps 4000`. Everything else was default,
including `--schedule-budget 0` (warmdown shaped against the whole epoch), `--seed 20260905`,
`--eval-every 1000` and `--eval-origins 2048`. The prior arm changes **exactly one flag**:

```bash
mlq submit --name timexer-amplitude-prior-4k --priority 1 --max-parallel-runs 1 \
  --time-limit 12h --max-attempts 1 --cwd "$PWD" -- \
  ./torch-env.sh /var/tmp/tb0_v13 train-timexer-segment \
  --run timexer-prior35-4k --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 \
  --min-history 256 --batch-size 256 --optimizer polar-express --preview-patience 3 \
  --x0-lambdas disabled --max-steps 4000 --amplitude-prior 0.0035
```

Bracket arms, only if the central one misses band 4: identical lines with
`--amplitude-prior 0.002 --run timexer-prior20-4k` and
`--amplitude-prior 0.006 --run timexer-prior60-4k`.

Then the mutual self-check, which is falsifier 4 and needs prerequisite B as well:

```bash
mlq submit --name timexer-calibrate-prior35 --priority 1 --max-parallel-runs 1 \
  --time-limit 45m --max-attempts 1 --cwd "$PWD" -- \
  ./torch-env.sh /var/tmp/tb0_v13 calibrate-timexer-segment \
  --checkpoint training/runs/timexer-prior35-4k/weights/best \
  --calibration training/runs/timexer-prior35-4k/mean-calibration-v2.json \
  --output training/runs/timexer-prior35-4k/gens/0 \
  --batch-size 256
```

---

## 8. What is left open

- The `v2` gain column is [INFERENCE] until run (a) lands. The inference is tight (§3) but it is
  not a measurement.
- `s_h` is learned, so the `lambda` -> `beta_post` map is an equilibrium argument, not a
  measurement. The bracket exists because of it.
- The prior's step-time cost is arithmetic plus a measured bandwidth, not a measured step. Its
  traffic charge in `step_cost` is exact by construction; the 168.0 -> ~169.1 ms prediction is not.
- The prior does nothing for h < 20 and is not meant to. If the two-sided post-hoc gain and the
  prior both land, the short end is carried entirely by the former, and its `3*SE` bound is then
  the only thing standing between a `4.28` estimate and 4x notional on an h=1 signal whose
  breakeven is 0.1-1.2 bps/side. That bound is the piece most worth attacking next: `3*SE`
  is a choice, and the risk-correct choice is a function of the utility layer's sizing rule,
  which nothing currently connects to the calibration.
