# Long-horizon failure: signal survives, but forecast amplitude overwhelms it

## Verdict and evidence boundaries

**DEMONSTRATED:** The measured long-horizon failure is predominantly **mis-scaled positively correlated close forecasts**, not a constant-offset failure and not universally zero long-horizon signal. At `timexer-control-4k`, step 3000, gen 2, held-out full, h=192, the close-MSE gain is `0.00002917 + 0.00330635 − 0.02519209 = −0.02185657`. The optimal positive multiplier on the centered forecast is approximately **0.26594**. Thus an oracle amplitude correction recovers positive but **small** skill: close-MSE ratio **0.996664**, not a dramatic improvement over persistence. On the same population, h=64 has a larger attainable gain: ratio **0.989442**.

**DEMONSTRATED:** There is important missing evidence. The requested historical three-part decomposition and long-horizon-specific calibration cannot be reconstructed completely from these reports. The former reports overwrite their horizon curves at each evaluation; the latter report only aggregate coverage indexed by training step. This is not a reason to guess either answer.

**HYPOTHESIS:** Training is fitting unstable long-horizon conditional means or shared representations whose amplitude generalizes poorly; overlapping cumulative targets and joint mean/scale optimization may contribute. The reports establish the resulting calibration-of-the-mean failure, but do not identify gradient interference, memorization, or mean/scale conflict as its unique training cause.

## Provenance and notation

**DEMONSTRATED — method:** Read-only executions of `./target/release/report_cli <gen> timexer_segment_<base> --run <run>`, with in-memory Python parsing and arithmetic, and source inspection. No files changed, training launched, GPU computation performed, or queue/lease acquired.

All numerical tables below are **DEMONSTRATED**, either directly printed report values or explicitly identified arithmetic from them. Values are rounded from the CLI's serialized float output. Table scope supplies the run, generation, step, draw, and report base for every cell.

Run aliases used in tables:

- `control`, `cutoff`, `basis`, `invsqrt`, `schedule`, `mlp` mean `timexer-control-4k`, `timexer-cutoff32-4k`, `timexer-basis88-4k`, `timexer-invsqrt-4k`, `timexer-sched5000-4k`, `timexer-mlpdown4x-4k`.
- `nox0`, `recipe`, `scalar`, `nox0scalar`, `lr50` mean `timexer-nox0-20260906`, `timexer-recipe-v3-20260906`, `timexer-scalarlr1-20260906`, `timexer-nox0-scalarlr1-20260906`, `timexer-lr50-scalar1-20260906`.
- `pred32` means `timexer-predlen32-3k`.
- `S` = held-out sample, `X` = held-out cross-section draw, `F` = held-out full. These are different populations, not interchangeable estimates. Gen-2 X duplicates the corresponding gen-1 X population/results, not an independent replication.

**DEMONSTRATED — storage limits:** In gen 1, the latest `_decomposition`, `_offset`, `_signal`, and `_horizon_error` curves are step 3000 for the six 4k arms, step 5000 for the five older arms, and step 2000 for pred32. `_horizon_steps` and `_horizon_steps_scaling` retain step histories. Gen 2 exists for control, cutoff, and invsqrt, at step 3000. No gen-2 directory was present for the other listed arms. Recipe lacks both horizon-step history bases. The directory census and `runner.rs:1963–2003` show latest-curve writing; the only historical snapshots found in the inspected generation layout are a few candle windows, not complete diagnostic populations.

## 1. What the decomposition actually measures

**DEMONSTRATED — source:** `trading_bots/src/torch/timexer_segment/reports.rs:265–305` and `runner.rs:1289–1336` define, for close forecast `f = μ + g`, `E[g]=0`, target `y`, and persistence error `P=E[y²]`:

\[
O=(2\mu E[y]-\mu^2)/P,\quad
D=\operatorname{Cov}(f,y)^2/(\operatorname{Var}(f)P),\quad
C=-(\beta-1)^2\operatorname{Var}(f)/P,
\]
\[
\beta=\operatorname{Cov}(f,y)/\operatorname{Var}(f),\qquad
R_{close}=1-O-D-C.
\]

`D` is the **best-scale demeaned gain**, not `1−ρ²`. Older saved series labels say `= implied 1 - ρ²`; that label is misleading. The implementation calculates `ρ² Var(y)/E[y²]`.

**DEMONSTRATED:** `_horizon_steps` reports **four-channel** MSE ratios. The exact three-part decomposition is **close-channel only**. Its `all-channel total gain` is provided separately but not decomposed. Consequently, adding the three close components must not be represented as exactly reconstructing the headline four-channel ratio. The close result establishes the same failure mechanically, while an exact decomposition of all four channels is structurally absent.

### Latest sample decompositions

**DEMONSTRATED; gen 1, S, base `_decomposition`; step 3000 for first six arms, step 5000 for older arms.** Each horizon cell is `Rclose ; O ; D ; C`.

| Run | h=64 | h=192 |
|---|---|---|
| control | 1.0104451 ; +0.000000018 ; 0.01294713 ; −0.02339220 | 1.0336208 ; +0.000074127 ; 0.004812212 ; −0.038507134 |
| cutoff | 1 ; 0 ; 0 ; 0 | 1 ; 0 ; 0 ; 0 |
| basis | 1.0124243 ; −0.000098598 ; 0.008633947 ; −0.020959698 | 1.0365156 ; −0.000356511 ; 0.001160052 ; −0.037319124 |
| invsqrt | 1.0109254 ; −0.000127239 ; 0.009256565 ; −0.02005476 | 1.0247997 ; −0.000040337 ; 0.004550662 ; −0.029310027 |
| schedule | 1.0133318 ; −0.000067820 ; 0.013477987 ; −0.026742015 | 1.0410708 ; +0.000073060 ; 0.008444913 ; −0.04958881 |
| mlp | 0.99994195 ; −0.000751822 ; 0.011851757 ; −0.011041888 | 1.0187769 ; −0.001600068 ; 0.006820106 ; −0.02399691 |
| nox0 | 1.0677632 ; −0.000983357 ; 0.006526193 ; −0.07330603 | 1.0792485 ; −0.001072381 ; 0.009734822 ; −0.087910905 |
| recipe | 1.0450359 ; −0.001811861 ; 0.012669443 ; −0.055893444 | 1.0414769 ; −0.003898256 ; 0.020316297 ; −0.05789496 |
| scalar | 1.0372458 ; −0.000456155 ; 0.017500622 ; −0.054290272 | 1.0815080 ; +0.000076875 ; 0.009444026 ; −0.091028914 |
| nox0scalar | 1.0379145 ; −0.001956846 ; 0.020498712 ; −0.056456394 | 1.0951426 ; −0.000559237 ; 0.008129772 ; −0.10271315 |
| lr50 | 1.1411125 ; −0.000176932 ; 0.018907314 ; −0.15984291 | 1.1856929 ; +0.000002135 ; 0.01612125 ; −0.20181625 |

**DEMONSTRATED:** In every above sample cell with close ratio above persistence, the negative scaling term dominates the negative offset term. Positive correlation below resolves its direction as over-amplitude, not an anticorrelated mean that merely needs sign reversal. MLP h=64 is a special case: its close ratio is slightly below persistence and the decomposition alone does not uniquely identify which side of the optimal amplitude it occupies.

### Latest cross-section and full decompositions

**DEMONSTRATED; all step 3000; X is gen 1 `_decomposition`, F is gen 2 `_decomposition`.** Same cell format.

| Run/draw | h=64 | h=192 |
|---|---|---|
| control X | 1.0166227 ; +0.008948800 ; 0.003952913 ; −0.029524397 | 1.0201344 ; +0.005876874 ; 0.004112317 ; −0.03012363 |
| cutoff X | 1 ; 0 ; 0 ; 0 | 1 ; 0 ; 0 ; 0 |
| basis X | 1.0278491 ; −0.007739703 ; 0.003365014 ; −0.023474462 | 1.0390205 ; −0.011569406 ; 0.002417179 ; −0.029868295 |
| invsqrt X | 1.0365162 ; −0.009765667 ; 0.002398375 ; −0.029148938 | 1.0224128 ; −0.006271055 ; 0.005754538 ; −0.021896258 |
| schedule X | 1.0119546 ; +0.006334215 ; 0.006106902 ; −0.024395674 | 1.0278915 ; +0.001765986 ; 0.008703180 ; −0.03836062 |
| mlp X | 1.0111570 ; +0.006334237 ; 0.002106633 ; −0.01959788 | 1.0310333 ; +0.006866166 ; 0.001076240 ; −0.038975716 |
| control F | 1.0109258 ; −0.000076937 ; 0.010635123 ; −0.021483976 | 1.0218566 ; +0.000029166 ; 0.003306347 ; −0.025192088 |
| cutoff F | 1 ; −0 ; 0 ; 0 | 1 ; −0 ; 0 ; 0 |
| invsqrt F | 1.0094376 ; −0.000287105 ; 0.009203260 ; −0.018353725 | 1.0186203 ; +0.000040456 ; 0.002935446 ; −0.02159621 |

**DEMONSTRATED:** Bias matters more on X than S/F, especially basis and invsqrt. For basis X h=192, fixing only centered amplitude while retaining its original offset still leaves ratio **1.0091522**. Removing or shrinking the offset is necessary there. Conversely control X has a *beneficial* constant tilt; counting that entire gain as conditional signal would overstate what the model knows about relative returns.

### Offset levels, not just normalized contributions

**DEMONSTRATED; `_offset`, same latest steps/generations as above.** Entries are `mean forecast / mean target` in origin-sigma return units, h64 then h192.

| Run/draw | h64 | h192 |
|---|---|---|
| control S | 0.000863684 / 0.000752844 | 0.0394222 / 0.0987011 |
| cutoff S | 0 / 0.000752844 | 0 / 0.0987011 |
| basis S | −0.0546464 / 0.000752844 | −0.100535 / 0.0987011 |
| invsqrt S | −0.0621790 / 0.000752844 | −0.0158889 / 0.0987011 |
| schedule S | 0.0467009 / 0.000752844 | 0.0386710 / 0.0987011 |
| mlp S | 0.153718 / 0.000752844 | 0.478405 / 0.0987011 |
| nox0 S | 0.175693 / 0.000752844 | 0.414676 / 0.0987011 |
| recipe S | 0.238215 / 0.000752844 | 0.679443 / 0.0987011 |
| scalar S | −0.118397 / 0.000752844 | 0.0414027 / 0.0987011 |
| nox0scalar S | 0.247533 / 0.000752844 | 0.336876 / 0.0987011 |
| lr50 S | −0.0734559 / 0.000752844 | 0.000912899 / 0.0987011 |
| control X | 0.220651 / 0.539919 | 0.319119 / 0.727880 |
| cutoff X | 0 / 0.539919 | 0 / 0.727880 |
| basis X | −0.134972 / 0.539919 | −0.387413 / 0.727880 |
| invsqrt X | −0.166054 / 0.539919 | −0.229649 / 0.727880 |
| schedule X | 0.143280 / 0.539919 | 0.0791799 / 0.727880 |
| mlp X | 0.143281 / 0.539919 | 0.402263 / 0.727880 |
| control F | −0.0539971 / −0.00396307 | −0.105191 / −0.0687591 |
| cutoff F | 0 / −0.00396307 | 0 / −0.0687591 |
| invsqrt F | −0.100395 / −0.00396307 | −0.0654501 / −0.0687591 |

### What can be attributed at earlier steps

**DEMONSTRATED; gen 1 S, `_horizon_steps` and `_horizon_steps_scaling`.** Each cell is `four-channel ratio / close scaling gain C`. These are deliberately **not** labeled a full historical decomposition: historical O and D are absent. No valid algebra recovers them by mixing the four-channel ratio with close-only C.

| Run | Step | h64 ratio / C | h192 ratio / C |
|---|---:|---|---|
| control | 1000 | .9906588 / −.000023842 | .9967067 / −.000570356 |
| control | 2000 | .9854981 / −.004986312 | 1.0244582 / −.02569155 |
| control | 3000 | 1.0128236 / −.02339220 | 1.0363796 / −.03850713 |
| cutoff | 1000, 2000, 3000 | .9980796 / **0** | .9992614 / **0** |
| basis | 1000 | .9829394 / −.002715898 | .9945533 / −.000320142 |
| basis | 2000 | .9827499 / −.003970773 | 1.0192720 / −.02152055 |
| basis | 3000 | 1.0117363 / −.02095970 | 1.0373456 / −.03731912 |
| invsqrt | 1000 | .9863583 / −.000845295 | .9932484 / −.000140146 |
| invsqrt | 2000 | .9882811 / −.004357409 | 1.0089053 / −.01440124 |
| invsqrt | 3000 | 1.0124418 / −.02005476 | 1.0273044 / −.02931003 |
| schedule | 1000 | .9796231 / −.001980574 | .9945441 / −.000059721 |
| schedule | 2000 | 1.0234768 / −.02753931 | 1.0524788 / −.05286467 |
| schedule | 3000 | 1.0226125 / −.02674202 | 1.0438504 / −.04958881 |
| mlp | 1000 | .9962003 / −.000842882 | .9987018 / −.000514105 |
| mlp | 2000 | .9800510 / −.004106179 | 1.0155239 / −.01984113 |
| mlp | 3000 | .9986621 / −.01104189 | 1.0208840 / −.02399691 |
| nox0 | 1000 | .9831483 / −.002589686 | .9949357 / −.000030675 |
| nox0 | 2000 | .9885970 / −.006954972 | 1.0256909 / −.02719145 |
| nox0 | 3000 | 1.0175143 / −.02827700 | 1.0224952 / −.03029346 |
| nox0 | 4000 | 1.0115403 / −.03422979 | 1.0393806 / −.05354049 |
| nox0 | 5000 | 1.0688473 / −.07330603 | 1.0853760 / −.08791091 |
| scalar | 1000 | .9846242 / −.000632695 | .9924023 / −.000060205 |
| scalar | 2000 | .9868574 / −.006208772 | 1.0230123 / −.02730701 |
| scalar | 3000 | 1.0122149 / −.02383976 | 1.0376681 / −.04088076 |
| scalar | 4000 | 1.0250827 / −.03603311 | 1.0714008 / −.07172599 |
| scalar | 5000 | 1.0334505 / −.05429027 | 1.0857874 / −.09102891 |
| nox0scalar | 1000 | .9830457 / −.000851583 | .9924284 / −.000001380 |
| nox0scalar | 2000 | 1.0037867 / −.01154044 | 1.0193120 / −.02250758 |
| nox0scalar | 3000 | 1.0173702 / −.02391448 | 1.0330327 / −.03737101 |
| nox0scalar | 4000 | .9950011 / −.02011502 | 1.0535929 / −.05757050 |
| nox0scalar | 5000 | 1.0388138 / −.05645639 | 1.0991403 / −.10271315 |
| lr50 | 1000 | .9780603 / −.000130547 | .9873288 / −.000598879 |
| lr50 | 2000 | 1.0098797 / −.02830796 | 1.0177616 / −.03320350 |
| lr50 | 3000 | .9834521 / −.02681251 | 1.0318027 / −.05379416 |
| lr50 | 4000 | 1.0144881 / −.05201236 | 1.1059809 / −.11320305 |
| lr50 | 5000 | 1.1389968 / −.15984291 | 1.1935241 / −.20181625 |

**DEMONSTRATED:** Recipe's historical ratios/C are structurally absent; its latest decomposition is above. Pred32 has only steps 1000/2000 in the read metrics, and no h64/h128/h192 outputs. The cutoff long-close forecasts are exactly persistence; its slightly sub-unity **all-channel** ratios are not long-close predictive skill.

## 2. Quantitative amplitude test

**DEMONSTRATED — algebra:** Set `q=sqrt(−C/D)`. If `D>0`, correlation is positive, and `q>1`, the only positive centered optimum is

\[
\beta=\frac{1}{1+q},\qquad R_{\mu+\beta g}=1-O-D.
\]

If `q<1`, there are two positive candidates, `1/(1+q)` and `1/(1−q)`. Correlation sign alone cannot distinguish them. Do not silently choose an over-amplitude explanation for that case. A forecast-variance or covariance report would resolve it.

**DEMONSTRATED — arithmetic from latest `_decomposition`; scope matches section 1.** Entries are `centered optimal gain / oracle ratio retaining original μ`.

| Run/draw | h64 | h128 | h192 |
|---|---|---|---|
| control S | .42659 / .9870529 | .42172 / .9864378 | .26118 / .9951137 |
| basis S | .39092 / .9914647 | .27601 / .9953114 | .14988 / .9991965 |
| invsqrt S | .40454 / .9908707 | .42468 / .9884085 | .28266 / .9954897 |
| schedule S | .41518 / .9865898 | .31285 / .9900605 | .29212 / .9914820 |
| mlp S | .50885 **or larger-than-one branch** / .9889001 | .46510 / .9880257 | .34773 / .9947800 |
| nox0 S | .22981 / .9944572 | .23668 / .9907233 | .24968 / .9913376 |
| recipe S | .32254 / .9891424 | .42293 / .9734269 | .37201 / .9835820 |
| scalar S | .36215 / .9829555 | .29515 / .9862194 | .24363 / .9904791 |
| nox0scalar S | .37600 / .9814581 | .32449 / .9826524 | .21956 / .9924295 |
| lr50 S | .25591 / .9812696 | .26106 / .9778832 | .22035 / .9838766 |
| control X | .26788 / .9870983 | .24088 / .9897837 | .26980 / .9900108 |
| basis X | .27463 / **1.0043747** | .26264 / **1.0100403** | .22147 / **1.0091522** |
| invsqrt X | .22291 / **1.0073673** | .32909 / **1.0014229** | .33891 / **1.0005165** |
| schedule X | .33348 / .9875589 | .27863 / .9912015 | .32264 / .9895308 |
| mlp X | .24691 / .9915591 | .08425 / .9919201 | .14249 / .9920576 |
| control F | .41300 / .9894418 | .32099 / .9951827 | .26594 / .9966645 |
| invsqrt F | .41456 / .9910839 | .31480 / .9957927 | .26937 / .9970241 |

**DEMONSTRATED:** All cutoff centered gains are undefined (`0/0`), not zero learned calibration coefficients. Its corrected close ratio remains exactly 1.

### Shrinking the actual forecast, not retaining its old offset

**DEMONSTRATED — algebra:** A directly deployable scalar shrinkage `a f` has optimum

\[
a=\frac{E[fy]}{E[f^2]},\quad R_{af}=1-\frac{E[fy]^2}{E[f^2]P}.
\]

For the unambiguous rows above, `_offset` plus the decomposition recover `P=(2μE[y]−μ²)/O`, `V=DP/β²`, `E[fy]=βV+μE[y]`, and `E[f²]=V+μ²`. This differs from retaining the measured population offset. The near-zero O rows make this recovery numerically delicate; the following rounded values are diagnostic, not deployment constants.

**DEMONSTRATED — latest `_decomposition` + `_offset`, same scope.** Entries are `raw forecast multiplier a / oracle close-MSE ratio`:

| Run/draw | h64 | h128 | h192 |
|---|---|---|---|
| control S | .42659 / .9870529 | .42152 / .9864199 | .26177 / .9951648 |
| basis S | .39023 / .9913817 | .27218 / .9948326 | .14725 / .9988777 |
| invsqrt S | .40363 / .9907649 | .42455 / .9883988 | .28231 / .9954601 |
| schedule S | .41482 / .9865331 | .31367 / .9901595 | .29253 / .9915300 |
| mlp S | **ambiguous** | .46113 / .9871328 | .34122 / .9931159 |
| nox0 S | .22801 / .9935238 | .23839 / .9907510 | .24953 / .9901495 |
| recipe S | .31783 / .9875138 | .41980 / .9716576 | .36383 / .9798401 |
| scalar S | .36091 / .9825603 | .29490 / .9861598 | .24390 / .9905335 |
| nox0scalar S | .37101 / .9797715 | .32433 / .9813097 | .22015 / .9917615 |
| lr50 S | .25575 / .9811049 | .26036 / .9774693 | .22036 / .9838783 |
| control X | .35516 / .9927621 | .30422 / .9944091 | .32686 / .9937877 |
| basis X | .19380 / .9982921 | .16619 / .9981286 | .12271 / .9992214 |
| invsqrt X | .13168 / .9991405 | .26596 / .9960646 | .28007 / .9960030 |
| schedule X | .39304 / .9913671 | .29495 / .9929886 | .33340 / .9906951 |
| mlp X | .34297 / .9958214 | .17669 / .9983710 | .22106 / .9972817 |
| control F | .41251 / .9893748 | .32072 / .9950902 | .26672 / .9966673 |
| invsqrt F | .41239 / .9908399 | .31496 / .9957837 | .27008 / .9970464 |

**DEMONSTRATED:** MLP S h64 admits raw-shrink solutions `.50062/.9883379` or `.53752/.9997765` under its two variance branches. It cannot support a unique oracle claim without the missing second moment.

**DEMONSTRATED:** The amplitude hypothesis passes as an *in-population algebraic diagnosis*, including raw forecast shrinkage. It does **not** demonstrate that coefficients fitted on these held-out targets would work on future data. These are oracle calibrations on the measured draw. Positive rescaling also leaves forecast ranks and sign-only decisions unchanged; fixing MSE does not automatically improve the signed equal-notional policy's economics.

**HYPOTHESIS:** A smooth, positive, chronologically cross-fitted amplitude calibration will transfer enough of this gain to repair the long-horizon MSE. The cheapest decisive test is to fit on an earlier calibration block and evaluate unchanged coefficients on a later untouched block, including net utility. Do not fit and score on the same held-out draw and call that generalization.

## 3. Is long-horizon sigma over- or under-dispersed?

**DEMONSTRATED:** This question is **not identified by the stored reports**. `_calibration` is step-indexed aggregate coverage; `_horizon_error` contains per-horizon forecast and persistence MSE, but no predicted variance, standardized residual moments, or horizon-specific coverage. Source: `reports.rs:735–744, 985–1060`; `runner.rs:915–918, 993–995, 1118–1120`. The evaluator sums coverage over horizons and channels before writing these metrics. No alternate long-horizon calibration series was found in the inspected report layouts/source.

**DEMONSTRATED — `_horizon_error`, S gen 1 latest step as defined above.** Entries are all-channel forecast MSE at h64/h128/h192. Persistence denominators for this S population are **31.050982 / 58.146870 / 84.196625**, identical across the listed 192-horizon arms.

| Run | Forecast MSE h64 / h128 / h192 |
|---|---|
| control | 31.449165 / 58.776720 / 87.259660 |
| cutoff | 30.991350 / 58.097233 / 84.134440 |
| basis | 31.415405 / 60.009100 / 87.341000 |
| invsqrt | 31.437311 / 58.714046 / 86.495560 |
| schedule | 31.753120 / 60.342293 / 87.888690 |
| mlp | 31.009438 / 58.292410 / 85.954994 |
| nox0 | 33.188755 / 63.203026 / 91.384995 |
| recipe | 32.427330 / 59.608803 / 88.031760 |
| scalar | 32.089650 / 62.057354 / 91.419630 |
| nox0scalar | 32.256187 / 61.832428 / 92.543900 |
| lr50 | 35.366970 / 67.735290 / 100.490710 |

**DEMONSTRATED — `_horizon_error`, F gen 2 step 3000:** persistence is **32.276173 / 85.677376 / 116.542500**. Control forecast MSE is **32.632347 / 87.096170 / 119.077900**; invsqrt is **32.580788 / 86.967630 / 118.692850**; cutoff is **32.218952 / 85.620130 / 116.483660**. X has no `_horizon_error` series; that is structural absence, not measured NaN MSE.

**DEMONSTRATED — `_calibration`, gen 1 S, latest steps:** aggregate `within 1σ / within 1.96σ` is control `.683378/.928312`, cutoff `.868383/.971856`, basis `.685317/.927362`, invsqrt `.666028/.920738`, schedule `.658059/.919443`, mlp `.670701/.923064` at step 3000; nox0 `.677570/.923437`, recipe `.677108/.926445`, scalar `.632298/.902590`, nox0scalar `.681218/.923960`, lr50 `.584096/.868891` at step 5000. Nominals in the same report are `.6827/.9500`. Gen-2 F step3000 is control `.680998/.927066`, invsqrt `.663349/.917810`, cutoff `.864712/.973029`.

**DEMONSTRATED:** Several aggregates have undercoverage, particularly in the outer band; control is not demonstrably globally over-dispersed merely because its central coverage is near nominal. Cutoff is overcovered in aggregate but does not train the requested long means. None of these aggregate results establishes the dispersion at h64 or h192.

**DEMONSTRATED — mathematical correction:** `sigma >> |mean|` and over-amplified conditional means are **not contradictory**. Predictive sigma measures residual uncertainty; mean calibration compares the forecast with the much smaller conditional expectation. If explainable variance is small, an excessively large mean can still be much smaller than residual sigma. A one-sigma gate almost never firing is compatible with a calibrated, low-SNR forecast and is not evidence of decoupled heads by itself.

**DEMONSTRATED — mechanism:** Gaussian NLL has gradients `∂L/∂m=(m−y)/s²` and `∂L/∂log s=1−(y−m)²/s²`. The mean and scale genuinely interact through the objective, in addition to sharing hidden activations. **HYPOTHESIS:** inflated conditional scales can attenuate mean learning, or shared updates can damage one head while improving the other. Demonstrating that requires per-horizon residual/scale statistics or gradient evidence, absent here.

## 4. Does the model extract long-horizon information?

**DEMONSTRATED — `_signal`, latest gen-1 S/X, steps 3000 for newer arms and 5000 for older arms.** Each cell is `pooled Pearson ; timestamp IC ± reported iid SE`. SE is recovered from the chart's `IC +1 s.e.` minus IC; its labels explicitly do not promise a confidence interval.

| Run/draw | h64 | h128 | h192 |
|---|---|---|---|
| control S | .113785 ; NaN ± NaN | .116605 ; NaN ± NaN | .069374 ; NaN ± NaN |
| control X | .063309 ; .033932 ± .020773 | .058720 ; .025203 ± .020863 | .064404 ; .052252 ± .018545 |
| cutoff S | NaN ; NaN ± NaN | NaN ; NaN ± NaN | NaN ; NaN ± NaN |
| cutoff X | NaN ; NaN ± NaN | NaN ; NaN ± NaN | NaN ; NaN ± NaN |
| basis S | .092919 ; NaN ± NaN | .072819 ; NaN ± NaN | .034062 ; NaN ± NaN |
| basis X | .058412 ; .018240 ± .021002 | .067190 ; .017579 ± .020396 | .049377 ; .027884 ± .019602 |
| invsqrt S | .096211 ; NaN ± NaN | .107757 ; NaN ± NaN | .067463 ; NaN ± NaN |
| invsqrt X | .049314 ; .021620 ± .022338 | .077388 ; .039904 ± .020953 | .076186 ; .059493 ± .016903 |
| schedule S | .116095 ; NaN ± NaN | .098939 ; NaN ± NaN | .091902 ; NaN ± NaN |
| schedule X | .078690 ; .029654 ± .020083 | .079452 ; .021385 ± .018869 | .093694 ; .047731 ± .017629 |
| mlp S | .108866 ; NaN ± NaN | .112306 ; NaN ± NaN | .082589 ; NaN ± NaN |
| mlp X | .046217 ; .032253 ± .021386 | .018917 ; **−.022135 ± .021794** | .032948 ; .007957 ± .021320 |
| nox0 S | .080785 ; NaN ± NaN | .095298 ; NaN ± NaN | .098671 ; NaN ± NaN |
| recipe S | .112559 ; NaN ± NaN | .167740 ; NaN ± NaN | .142544 ; NaN ± NaN |
| scalar S | .132290 ; NaN ± NaN | .117760 ; NaN ± NaN | .097186 ; NaN ± NaN |
| nox0scalar S | .143174 ; NaN ± NaN | .135694 ; NaN ± NaN | .090170 ; NaN ± NaN |
| lr50 S | .137504 ; NaN ± NaN | .150492 ; NaN ± NaN | .126977 ; NaN ± NaN |

**DEMONSTRATED — `_signal`, gen 2 F step 3000:**

| Run | h64 | h128 | h192 |
|---|---|---|---|
| control | .103127 ; .065486 ± .002787 | .070011 ; .061046 ± .002793 | .057502 ; .050928 ± .002760 |
| invsqrt | .095934 ; .059174 ± .002796 | .064870 ; .055582 ± .002838 | .054181 ; .043962 ± .002777 |
| cutoff | NaN ; NaN ± NaN | NaN ; NaN ± NaN | NaN ; NaN ± NaN |

**DEMONSTRATED:** Gen-2 X values repeat the corresponding gen-1 X rows to displayed precision. Older arms have no X series, and no F evaluation is present for them. Pred32 has no requested horizons. Historical per-horizon Pearson/IC/SE series are absent, so this is every available *stored latest curve*, not invented IC histories.

**DEMONSTRATED — verdict:** The claim that the current model extracts nothing at long horizons is contradicted by positive pooled correlations and positive full-draw timestamp IC at all three requested long horizons for control and invsqrt. The small X draw is inconclusive for several individual cells and is not uniformly positive: mlp h128 is negative. The strongest full evidence supports **small, positive long-horizon information plus substantial over-amplitude**, not an information-free trunk.

**HYPOTHESIS / statistical limit:** Positive population estimates here will persist out of period. Reported SEs use an iid timestamp approximation, while cumulative horizons, market regimes, and overlapping evaluation structure can induce dependence. They are not HAC/block-bootstrap significance estimates; repeated arms share data and are not independent replications. These data demonstrate extracted signal on the measured held-out populations, not a guaranteed causal/tradable future edge or an upper bound on all information in the dataset.

## 5. Why the basis attempt did not solve this

**DEMONSTRATED — source:** `model.rs:210–295, 1338–1346, 2078–2146, 2237–2256` shows:

1. The basis restricts the horizon shape to a linear span of exponential functions; coefficients are not bounded. Multiplying all coefficients by any scalar remains in that span. **Low rank is not an amplitude prior.** Smooth, low-rank over-amplitude is still permitted.
2. The close head coordinate is multiplied by `sqrt(h)` when decoded; targets are cumulative returns divided by origin sigma. The basis acts on the additional `sqrt(h)`-normalized mean coordinate. This distinction matters when choosing an appropriate shape prior.
3. The mean and scales continue to share the head hidden activation and the same NLL. The basis does not remove that interaction or the dense long-target objective.
4. The source itself notes substantial basis-column overlap and ill-conditioning. Adam's coordinatewise normalization does not mathematically make correlated-column conditioning disappear.

**DEMONSTRATED — matched results:** At step3000 S h192, basis has `D=.00116005, C=−.03731912, O=−.00035651`, versus control `D=.00481221, C=−.03850713, O=.00007413` (`_decomposition`, gen1). It retains nearly the same amplitude penalty while losing most attainable conditional gain. X additionally has adverse offset: `O=−.01156941`, versus control `+.00587687`. Its positive IC is weaker and individually inconclusive on the small draw. This is why the basis is not a successful calibration intervention.

**DEMONSTRATED — objective-specific qualification:** The supplied observation that basis was worse at every matched step is supported by aggregate S NLL: basis/control at steps1000/2000/3000 is `2.0056455/2.0014539`, `2.0011559/1.9985625`, `2.0202742/2.0135024` (`_loss`, gen1). It must not be generalized to every horizon MSE: basis h64 and h192 ratios are better than control at steps1000 and2000 in `_horizon_steps`, and basis h64 is slightly better on the four-channel metric at3000. It still fails to stop the long-horizon rot.

**HYPOTHESIS:** The chosen span may be poorly matched to cumulative-return dynamics. For an exponentially decaying expected *increment*, the cumulative expected return is proportional to `1−exp(−h/T)`; after `sqrt(h)` normalization it is proportional to `(1−exp(−h/T))/sqrt(h)`, not simply `exp(−h/T)`. Conditioning, shape mismatch, initialization/optimization changes, and shared-trunk interference are possible contributors. The reports cannot isolate which caused the lower D. Consequently this experiment argues against repeating the same exponential restriction, not against explicit amplitude calibration, all low-rank paths, or long horizons themselves.

## 6. Ranked fixes that keep the complete 192-bar forecast

Every proposal below is **HYPOTHESIS** until its stated falsification test succeeds. None removes long horizons, zeroes the long-horizon loss, or predicts fewer bars ahead. Architecture-cost estimates below are **HYPOTHESIS / analytical estimates** based on the measured configuration, not runtime benchmarks. Source constants establish head input width 768, hidden width1024, 1536 dense outputs, and 96,000 scored origins per training batch (`model.rs:18–29, 1338–1346, 1401–1405`; configuration supplied and saved manifest). FLOPs count a multiply-add as two operations.

### 1. Explicit mean-amplitude calibration, then a function-space amplitude prior

- **Mechanism repaired:** Directly addresses the demonstrated negative C while retaining the positive ranking signal. Fit a smooth positive per-horizon shrinkage curve on a chronological calibration block; distinguish raw-mean shrinkage from affine centering. Keep every horizon output. If the calibration transfers, regularize the *function's mean energy* during training rather than merely shrinking a head weight that upstream layers can compensate for.
- **Cost:** A per-horizon curve needs up to192 coefficients for close, or768 for independent channel calibration; negligible parameter/optimizer memory. One additional multiplication per mean output, ideally fused into decoding; no new dense activation. A training mean-energy term is O(origins × horizons × channels) elementwise/reduction work and should reuse existing tensors, not materialize another full output block.
- **Training bits:** Post-hoc calibration leaves checkpoint/training bits unchanged; training prior changes gradients and bits. Preserve candle geometry by changing its anchored coordinate consistently, not independently scaling OHLC levels into invalid candles.
- **Cheapest falsifier:** Fit coefficients on an earlier calibration segment, freeze them, and evaluate every horizon on the later untouched block. Reject if the shrinkage advantage vanishes, calibration drifts materially, or net utility worsens. Current on-draw oracle numbers justify this test but are not its result.
- **Important qualification:** A fixed train-time output multiplier alone is reparameterization; the network can increase its weights to undo it. The prior must penalize the predicted mean function or otherwise constrain effective amplitude, not just alter initialization scale.

### 2. Multi-resolution cumulative supervision with dense coverage retained

- **Mechanism repaired:** Reduces redundant contribution of highly correlated neighboring cumulative targets and gives long-range structure explicit representation, without eliminating h192 or any intermediate prediction. Use dyadic anchors plus192 as an auxiliary structural objective, or horizon-band-normalized sampling with **positive probability for every dense horizon**, and optionally multi-resolution return-increment consistency. Keep the dense term structure supervised; do not leave 183 free output rows untrained.
- **Cost:** A nine-anchor auxiliary cumulative term (`1,2,4,8,16,32,64,128,192`) adds O(9 × origins × channels) arithmetic with zero necessary learned parameters. With the current dense head, it does **not** remove the dense output GEMM or automatically save activation memory. Sampling/restructuring loss can reduce target-loss traffic only if implemented to avoid materialization, not merely multiplied by a mask afterward.
- **Training bits:** Changes objective/gradient bits; output shape and contract remain192.
- **Cheapest falsifier:** One matched full-horizon training comparison at the existing evaluation cadence, reporting D, C, timestamp IC and calibration by horizon, not only aggregate NLL. Reject if C decreases only by destroying D, or if long IC remains unchanged while short skill is merely traded away.
- **Caveat:** Coarsening does not automatically make cumulative targets independent or higher-SNR. Those claims need measured target covariance/effective rank and achievable skill; adjacent dyadic cumulative returns still overlap. This is a correlated-supervision hypothesis, not a proven remedy.

### 3. Predictive-SNR-informed weighting, not another variance prior

- **Mechanism repaired:** Allocate finite shared learning capacity using measured recoverable conditional signal rather than raw target variance. Estimate a regularized, smoothly varying horizon profile from cross-fitted achievable skill D or correlation, with uncertainty and economic relevance included. Maintain strictly positive support for all192 horizons.
- **Cost:**192 fixed weights; negligible storage, no new trainable parameters, and essentially the same cost as the existing horizon-weight multiplication. Cross-fitting and uncertainty estimation are analysis costs outside the training kernel.
- **Training bits:** Changes objective weights and all downstream gradient bits.
- **Cheapest falsifier:** Freeze the profile before the next comparison; compare against uniform and the already measured invsqrt arm on untouched time blocks. Reject if improved NLL merely shifts capacity toward already-easy horizons without improving long D or calibrated long utility.
- **Caveat:** There is no theorem that weights proportional to D are optimal; small long D could cause such a recipe to underemphasize economically valuable horizons. Evaluate both the rationale and the induced gradient allocation. Weighting using held-out results and scoring the same results is leakage. The existing NLL already includes `1/h` through its variance prior, and invsqrt adds an objective preference; neither measures predictive SNR.

### 4. Separate nonlinear mean and scale branches

- **Mechanism repaired:** Removes immediate competition in the single hidden head representation while retaining the trunk. Merely splitting the existing last linear matrix into two named matrices changes no representational capability; the current output rows already differ. A meaningful intervention supplies distinct hidden nonlinear projections, optionally testing which branch backpropagates into the shared trunk.
- **Cost:** Duplicating the 768→1024 hidden projection adds approximately **787,456 parameters**, **151 GFLOP forward per dense batch**, and one extra **196.6 MB bf16 hidden activation** before backward storage. Output projection parameter count can remain unchanged by partitioning its rows. Optimizer and gradient memory add several more bytes per added parameter; actual peak must be measured.
- **Training bits:** Yes. A stop-gradient control also changes the optimized objective/path, not merely implementation.
- **Cheapest falsifier:** Before adding capacity, test a carefully specified detached-scale/fixed-scale mean-gradient control over all horizons, or a width-budget-matched split-hidden branch. Reject the interference explanation if mean calibration and long D do not improve despite separating those gradients. Measure per-horizon sigma/coverage first; current aggregates cannot establish this mechanism.

### 5. Proper scoring rule less aggressively inverse-variance-weighted, e.g. Gaussian CRPS

- **Mechanism repaired:** Gaussian CRPS has bounded mean derivative `2Φ((m−y)/s)−1`, unlike Gaussian NLL's residual divided by variance, changing sensitivity to mean/scale interaction and outliers. It remains a proper probabilistic score and retains all horizons.
- **Cost:** No model parameters or additional output dimensions. O(origins × horizons × channels) arithmetic with Gaussian CDF/PDF or erf/exp evaluations; potentially material extra elementwise runtime and memory unless fused. No claim of a specific speedup is justified.
- **Training bits:** Yes, objective and gradients change.
- **Cheapest falsifier:** A matched complete-horizon objective comparison, normalized into comparable horizon units, checking long D/C, proper-score calibration and net utility. Reject if CRPS improves its own aggregate score but leaves long mean mis-scaling and out-of-period signal unchanged.
- **Caveat:** Proper NLL is not inherently inconsistent for the conditional mean. CRPS is still a joint location-scale score, not magical decoupling. Under symmetric irreducible noise both scores prefer the correct distribution; the motivation here concerns finite-capacity/optimization robustness.

### 6. Horizon-conditioned nonlinear pathway, preferably low-rank or grouped

- **Mechanism repaired:** Gives distant horizons a different feature transformation instead of forcing all means/scales through the same hidden activation. This targets insufficient horizon-specific conditional information, a weaker current diagnosis than over-amplitude but relevant if calibrated skill remains too small.
- **Cost:** A naive192-way 1024-hidden activation is prohibitive: approximately **37.75 GB bf16 forward activation** at96,000 origins, before gradients. Do not implement that broadcast. A horizon-embedding/rank-r adapter can add O(192r + input_width×r + r×output_width) parameters and O(origins×192×r) conditioning work with streaming/fusion, or a few grouped nonlinear branches can use multiples of the branch cost above. Actual FLOPs depend on the chosen factorization; an exact number before choosing it would be false precision.
- **Training bits:** Yes; checkpoint architecture and optimization change.
- **Cheapest falsifier:** A parameter-/FLOP-budget-matched grouped or low-rank nonlinear horizon adapter against an equal-capacity horizon-agnostic head. Reject horizon conditioning as the explanation if attainable long D/IC does not increase beyond amplitude-only calibration. Adding linear embeddings to an otherwise linear output is too weak a test if it just becomes a horizon bias.

**HYPOTHESIS — recommended decision order:** First establish transferable amplitude calibration and fill the missing per-horizon dispersion/second-moment evidence in a future permitted evaluation. Then test correlated-supervision/SNR allocation. Add separate nonlinear heads or horizon pathways only if those tests distinguish a representation problem from calibration. The evidence does not justify abandoning the complete192-bar target.

## 7. Missingness, NaNs, exact zeros, and nonclaims

**DEMONSTRATED:**

- All requested S timestamp IC and its plotted SE bounds are NaN: the sample's ticker-major strided origins do not form eligible within-timestamp cross-sections. Source `reports.rs:32–36`, `runner.rs:44–48`. These NaNs are not zero correlation or evidence against predictability.
- Cutoff h64/h128/h192 pooled Pearson and timestamp IC/bounds are NaN because close forecasts have zero variance. Its close `O`, `D`, `C`, total gain, mean forecast, top-/bottom-decile signed-return and conviction-spread diagnostics are **exactly zero**, including signed `−0` offset on F. These are degenerate persistence outputs, not successful learned long predictions. Rank diagnostics for tied constant forecasts must not be interpreted as information.
- Reference lines `zero gain`, `zero correlation`, `zero`, and `no mis-scaling 0.0` are exactly zero by construction, never empirical findings. Parity lines are reference values, not arm measurements.
- No numerical NaNs occurred in the retrieved decomposition/offset/MSE level curves for the requested trained horizons; the NaNs shown for X MSE in extraction correspond to **no X error series**, not a serialized NaN observation.
- Older X/F series, most gen-2 directories, recipe's horizon-step history, per-horizon calibration, all-channel component decompositions, and historical horizonwise O/D/offset/IC curves are structurally absent as described above. Pred32's requested long horizons are structurally absent and its objective scope differs; it is not a candidate solution or aggregate-NLL control for the192-horizon problem.
- Optimal gains are undefined for the cutoff zero-variance forecasts. The two-branch ambiguity for mlp S h64 is explicitly retained rather than fabricated away.
- No present experiment proves which shared gradients caused the observed rot, that iid SEs are valid confidence bounds, that cross-fitted shrinkage transfers, or that any ranked training fix succeeds. Those statements remain HYPOTHESIS.

**Bottom line — DEMONSTRATED:** The model already carries some long-horizon information. What training destroys is primarily the relationship between forecast magnitude and that small signal: the cost of excess forecast variance overtakes the conditional correlation gain. Mean calibration is the most directly evidenced repair, but the available full-draw oracle gain at the far end is modest, and the missing dispersion/history series prevent stronger claims about the optimizer's internal cause.