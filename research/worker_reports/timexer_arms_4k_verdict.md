# Matched-step arm investigation

## Provenance and scope

**DEMONSTRATED:** Run abbreviations below expand to `timexer-<abbreviation>-4k`; every report base abbreviation expands to `timexer_segment_<base>`, generation 1. All tabulated observed values are **DEMONSTRATED**, with run, step, and base supplied by their row/table heading. Derived differences and standard errors are explicitly identified as arithmetic from demonstrated values. No training, evaluation rerun, GPU use, or file modification was performed; the existing `report_cli` was executed read-only.

**DEMONSTRATED:** All six `_progress` reports contain exactly steps **1000, 2000, 3000**, not 4000. All six `weights/preview-latest/manifest.json` files record `step=3000`, `max_steps=4000`, `epoch_complete=false`, `completed_origins=768000`, and `completed_target_bars=147310502`. They do not record a termination-reason field. `meta.json` for control records commit `8a8baede24a848e24a747fd0d1b655e0ac4a3efd`.

The supervising agent supplied the authoritative termination diagnosis: all six failed with CUDA OOM during evaluation after the step-3000 report. That diagnosis was not independently read by this investigator and should be cited to the supervising agent's log evidence, not to the manifests. Per the revised assignment, the terminal comparison is **3000**, not 4000. No extrapolation or convergence claim is justified.

## Comparability

**DEMONSTRATED — all six latest manifests, step 3000:** OBJECTIVE stamp is `causal_patch_market_neutral_nll_v5`. FORMAT is `causal-patch-ohlc-universe-v10-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres-x0-none-mean-free`, except basis88's intended `...-mean-basis-8-8` suffix. Uniform loss applies to control/basis88/sched5000/mlpdown4x; cutoff32 records `cutoff:32`; invsqrt records `inv-sqrt`. Thus the stamp agrees, while the loss-weighted objectives deliberately differ.

**DEMONSTRATED — `_lr_trajectory`, recorded updates 0–2999:** Direct full-output string equality checks establish that control, cutoff32, basis88, and invsqrt have identical complete recorded learning-rate paths. Separate equality checks on all six named families establish that mlpdown4x matches control in packed QKV, attention output, MLP up, AdamW dense, and AdamW recipe scalars; **only MLP down differs**. This is the intended intervention, not a confound.

| Run(s) | `_lr_trajectory` at 0, 1000, 2000, 2999; family rates | Manifest schedule |
|---|---|---|
| control/cutoff32/basis88/invsqrt | QKV .03983717; attention output .023; MLP up .046; MLP down .023; dense .008; recipe scalars .04, unchanged | budget 9590, cooldown start 3836 |
| mlpdown4x | identical except MLP down .092, unchanged | budget 9590, cooldown start 3836 |
| sched5000 | matches control at 0/1000/2000; at 2999: QKV .028561259, attention/MLP down .01648985, MLP up .0329797, dense .0057356, scalars .028678 | budget 5000, cooldown start 2000 |

**DEMONSTRATED:** sched5000 first visibly decays at update 2001: dense .007997734 versus control .008 (`_lr_trajectory`). **HYPOTHESIS:** Differences between sched5000 and control already at step 1000 cannot be attributed to cooldown, because their measured rate paths still agree; stochastic/numerical run variation is present and small differences should not be treated as clean causal effect sizes.

**Exclusions:** Under the revised terminal-step and intended-family-LR criterion, **no arm is excluded from the 1000/2000/3000 descriptive comparison**. All six are excluded from any claimed completed-4000 verdict. Historical snapshot metrics unavailable at a requested checkpoint are separately excluded, never carried forward.

## Accuracy at every available interval

All values in this table: **DEMONSTRATED**, held-out sample; NLL from `_loss`, market-neutral MSE ratio versus persistence from `_skill`.

| Run | Step 1000 NLL / ratio | Step 2000 NLL / ratio | Step 3000 NLL / ratio | Observed unweighted NLL minimum step | Rise from minimum to 3000 (arithmetic) |
|---|---:|---:|---:|---:|---:|
| control | 2.0014539 / .99316704 | 1.9985625 / 1.009452 | 2.0135024 / 1.0140694 | 2000 | .0149399 |
| cutoff32 | 2.2986646 / .9964925 | 2.3024793 / .9962769 | 2.304595 / .99684435 | 1000 | .0059304 |
| basis88 | 2.0056455 / .9885171 | 2.0011559 / 1.0123658 | 2.0202742 / 1.0236186 | 2000 | .0191183 |
| sched5000 | 1.9940951 / .98550844 | 2.0012565 / 1.031716 | 2.034039 / 1.0273317 | 1000 | .0399439 |
| invsqrt | 2.0185077 / .9892862 | 1.987704 / 1.0003847 | 2.0386925 / 1.0128562 | 2000 | .0509885 |
| mlpdown4x | 2.0149841 / .9950437 | 1.9890882 / 1.0010644 | 2.0206172 / 1.0042043 | 2000 | .0315290 |

**DEMONSTRATED — all six `_loss`, each available step:** held-out sample persistence NLL is **2.3947663**. This is the common baseline, not a learned-arm measurement.

**DEMONSTRATED — latest manifests at step 3000:**

| Run | Selected best step | `best_objective_nll` | Selection objective |
|---|---:|---:|---|
| control | 2000 | 1.9985624809477918 | uniform |
| cutoff32 | 1000 | 1.049701191543634 | cutoff:32 |
| basis88 | 2000 | 2.001155815089945 | uniform |
| sched5000 | 1000 | 1.994095044058817 | uniform |
| invsqrt | 2000 | 1.6442281141699293 | inv-sqrt |
| mlpdown4x | 2000 | 1.9890881822721214 | uniform |

**DEMONSTRATED:** The stored `_loss` held-out sample NLL is unweighted, unlike weighted training NLL in cutoff32/invsqrt. For example cutoff32's selected objective is 1.049701191543634, not its step-1000 reported held-out sample NLL 2.2986646. The model loss weighting implementation (`model.rs:108-147`) normalizes horizon weights to mean one; uniform is exactly the unweighted objective.

**Ranking quantity:** use the **common per-horizon market-neutral MSE ratio versus persistence**, prioritizing preserving h=8 while keeping h≥64 below parity, with common unweighted held-out sample NLL as a secondary distributional check. These metrics do not inherit the objective weighting. Comparing manifest objective NLLs across uniform/cutoff/invsqrt is invalid. Uniform-arm objective NLLs are comparable; the common unweighted held-out sample NLL is also comparable across all arms, but cutoff32 deliberately leaves most horizons untrained and pays a substantial NLL penalty there.

## Core horizon comparison

Every number below is **DEMONSTRATED**, held-out sample market-neutral MSE ratio versus persistence. Source for h=1/8/64/192 at each step: `_horizon_steps`. Source for h=16/32 at 3000: `_horizon` latest snapshot, whose other four values exactly match the step-3000 history. `ABSENT` means not persisted historically, not zero. The available `_horizon_steps` does not contain h=16 or 32, and globbing the run directories found no historical horizon snapshots. Historical candle fans are not substitute aggregate measurements.

| Run | Step | h=1 | h=8 | h=16 | h=32 | h=64 | h=192 |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 1000 | .96065617 | .9713713 | ABSENT | ABSENT | .9906588 | .99670666 |
| control | 2000 | .95528346 | .9506608 | ABSENT | ABSENT | .9854981 | 1.0244582 |
| control | 3000 | .9605044 | .95291483 | .96888703 | .9798279 | 1.0128236 | 1.0363796 |
| cutoff32 | 1000 | .90009165 | .8893328 | ABSENT | ABSENT | .9980796 | .9992614 |
| cutoff32 | 2000 | .8680577 | .8683521 | ABSENT | ABSENT | .9980796 | .9992614 |
| cutoff32 | 3000 | .9293144 | .8732989 | .9316711 | .97970545 | .9980796 | .9992614 |
| basis88 | 1000 | .96105707 | .9666755 | ABSENT | ABSENT | .9829394 | .99455327 |
| basis88 | 2000 | .95488036 | .9522293 | ABSENT | ABSENT | .9827499 | 1.019272 |
| basis88 | 3000 | .9626927 | .9591973 | .9385883 | .97120655 | 1.0117363 | 1.0373456 |
| sched5000 | 1000 | .94656026 | .9507527 | ABSENT | ABSENT | .97962314 | .9945441 |
| sched5000 | 2000 | .9556089 | .9390215 | ABSENT | ABSENT | 1.0234768 | 1.0524788 |
| sched5000 | 3000 | .9581962 | .9048936 | .9341128 | .9818267 | 1.0226125 | 1.0438504 |
| invsqrt | 1000 | .94402516 | .95117337 | ABSENT | ABSENT | .98635834 | .9932484 |
| invsqrt | 2000 | .93821776 | .9354169 | ABSENT | ABSENT | .9882811 | 1.0089053 |
| invsqrt | 3000 | .9438679 | .9150961 | .93824595 | .97180945 | 1.0124418 | 1.0273044 |
| mlpdown4x | 1000 | .9640244 | .97749907 | ABSENT | ABSENT | .9962003 | .9987018 |
| mlpdown4x | 2000 | .9518082 | .9588806 | ABSENT | ABSENT | .98005104 | 1.0155239 |
| mlpdown4x | 3000 | .96024233 | .94750893 | .94931906 | .97825724 | .9986621 | 1.020884 |

**DEMONSTRATED:** cutoff32 uniquely keeps both requested long endpoints below parity while strongly improving h=8, but its long endpoints are **exactly unchanged across all three checkpoints**. The cutoff assigns exactly zero loss weight above its cutoff (`model.rs:118-123,136-138`); its latest `_tradable` close-only ratio versus close-anchored persistence is **exactly 1** at h=64 and 192 in both held-out sample and held-out cross-section. Its long-horizon Pearson/within-timestamp IC is **NaN**. **HYPOTHESIS:** this is successful prevention of harmful learned long-horizon movement by leaving the long end untrained/persistence-like, not learned long-horizon forecasting skill. It must not be sold as curing long-horizon learning.

### Joint-criterion ranking (descriptive, not significance-tested)

| Rank/tier | Arm | Evidence-backed assessment at terminal 3000 | Recommendation (HYPOTHESIS) |
|---|---|---|---|
| 1, qualified | cutoff32 | Best h=8 (.8732989), long endpoint ratios .9980796/.9992614; long close forecast is persistence, not learned signal | Only defensible candidate for explicitly short-horizon specialization; not a full-horizon solution |
| 2–3, tradeoff | invsqrt | h=8 .9150961, h=64 1.0124418, h=192 1.0273044; lowest observed common NLL at 2000, then largest rise | Best full-head short-horizon candidate, but reject as a rot cure |
| 2–3, tradeoff | mlpdown4x | h=8 .94750893, h=64 .9986621, h=192 1.020884 | Better terminal long endpoints than invsqrt, weaker h=8; still fails h=192 |
| 4–5, tradeoff | sched5000 | h=8 .9048936 but h=64 1.0226125 and h=192 1.0438504 | Reject for joint goal; short skill purchased alongside worse long forecasts |
| 4–5, tradeoff | control | h=8 .95291483, h=64 1.0128236, h=192 1.0363796 | Reference, not a solution |
| 6 | basis88 | h=8 .9591973 and h=192 1.0373456; common NLL worse than control at each matched step | Reject; no evidence for the intended improvement |

**HYPOTHESIS:** A total ordering among invsqrt/mlpdown4x or sched5000/control would require arbitrary weights on short skill versus long damage. The table intentionally preserves those tradeoffs. No trained-full-horizon arm satisfies both requested long endpoints at terminal 3000. The evidence does not justify saying all six are identical: cutoff materially changes short accuracy, but achieves long stability by not training those outputs.

## Calibration and generalization

All coverage pairs **DEMONSTRATED**, `_calibration`, held-out sample, reported as within 1σ / within 1.96σ. Reference series are **.6827 / .9500** at every available step.

| Run | 1000 | 2000 | 3000 |
|---|---:|---:|---:|
| control | .7053649 / .9383755 | .73257446 / .9531066 | .68337756 / .9283123 |
| cutoff32 | .88125926 / .9770101 | .8709132 / .9731083 | .8683834 / .9718564 |
| basis88 | .71321803 / .93993694 | .72401303 / .9495608 | .68531734 / .92736244 |
| sched5000 | .7100856 / .9404259 | .7137845 / .9454581 | .6580594 / .9194425 |
| invsqrt | .7145545 / .9386711 | .7393138 / .9552352 | .6660277 / .9207382 |
| mlpdown4x | .71748734 / .9421037 | .7347317 / .95556766 | .6707013 / .92306393 |

All gap values **DEMONSTRATED**, `_generalization_gap`, stored label `training minus held-out sample NLL`:

| Run | 1000 | 2000 | 3000 |
|---|---:|---:|---:|
| control | .2535381 | .22380069 | .18916039 |
| cutoff32 | -.9982784 | -1.031416 | -1.0390447 |
| basis88 | .24410634 | .21760307 | .18177056 |
| sched5000 | .2594383 | .21248879 | .15733539 |
| invsqrt | -.1270206 | -.12491332 | -.18234652 |
| mlpdown4x | .24247143 | .23743647 | .19475333 |

**DEMONSTRATED:** The existing cutoff32/invsqrt gaps mix **weighted training** NLL and **unweighted held-out sample** NLL. They are not same-objective generalization gaps and cannot be ranked as such. The evaluation refactor corrected this in the new code, but did not retroactively redefine stored values. For the uniform arms, training NLL continues falling while held-out sample NLL worsens after its observed minimum: control 2.2223632→2.2026627 versus 1.9985625→2.0135024, basis88 2.2187588→2.2020447 versus 2.0011559→2.0202742, mlpdown4x 2.2265246→2.2153707 versus 1.9890882→2.0206172 (`_loss`, 2000→3000); sched5000 training 2.2535334→2.1913745 while held-out sample 1.9940951→2.034039 (`_loss`,1000→3000).

**HYPOTHESIS:** These are overfitting signatures rather than broad underfitting. The positive uniform-arm gap is not proof against overfitting: training and held-out sample populations differ. Cutoff32's much broader aggregate coverage and poor all-horizon NLL are consistent with deliberately untrained long distributions, not improved full-horizon probabilistic generalization. Terminal calibration worsens in the trained-full-horizon arms, but the available coverage is not evidence of catastrophic sigma collapse.

## Trading snapshots: populated, thin, and not securely checkpoint-attributed

**DEMONSTRATED:** `_signal`, `_portfolio`, `_portfolio_rate`, `_portfolio_breakeven`, `_tradable`, `_tradable_rates`, and `_cross_section_census` are **horizon-indexed latest snapshots**, not step histories. `report_cli` does not print their title/checkpoint stamp. The latest `_horizon` matches step 3000 exactly; that does not itself establish the checkpoint of every subsequently written snapshot after an interrupted evaluation. Therefore every number in this section has training step **UNVERIFIED (latest stored snapshot)**. Do not relabel these as selected-best-checkpoint trading results or carry a prior snapshot forward to step 3000. The supervising agent may establish precise step attribution from additional metadata/log evidence.

**DEMONSTRATED — all six runs, latest `_cross_section_census`, h=1/8/16/32/64:** held-out sample contributing timestamps is **EXACTLY ZERO**, mean valid tickers **1.0661114**, narrowest timestamp **1**. Its cross-sectional IC and portfolio outputs are **NaN**, not measurements. Held-out cross-section is populated with **100** contributing timestamps, mean valid tickers **40**, narrowest **40**, versus scoring minimum **20**. This is a **thin forty-name draw**, not the full universe or the earlier few-hundred-name liquid core. It demonstrates the draw fired, not that estimates are robust or representative. The cutoff32 long-horizon zero-signal exception below invalidates alpha interpretation even with a populated census.

### Stored old-definition portfolio diagnostics

All rows **DEMONSTRATED**, training step **UNVERIFIED (latest snapshot)**. Columns: gross market-neutral decile spread in bps from `_portfolio`; gross bps/bar from `_portfolio_rate`; reported breakeven bps/side from `_portfolio_breakeven`. These are the old close-based, fixed-cost diagnostic, not the refactored executable raw next-open utility. Non-overlapping holds and four side-charges imply net = gross − 4×cost and rate = net/h. Positive-spread breakeven equals gross/4.

| Run | h | Gross bps | Gross bps/bar | Reported breakeven bps/side |
|---|---:|---:|---:|---:|
| control | 1 | 2.5751295 | 2.5751295 | .6437824 |
| control | 8 | 10.084158 | 1.2605197 | 2.5210395 |
| control | 16 | 11.644789 | .7277993 | 2.9111972 |
| control | 32 | 4.27431 | .13357219 | 1.0685775 |
| control | 64 | 33.455017 | .52273464 | 8.363754 |
| cutoff32 | 1 | 10.929033 | 10.929033 | 2.7322583 |
| cutoff32 | 8 | 25.45777 | 3.1822212 | 6.3644423 |
| cutoff32 | 16 | 25.405685 | 1.5878553 | 6.3514214 |
| cutoff32 | 32 | 28.939972 | .9043741 | 7.234993 |
| cutoff32 | 64 | **-14.817272, zero-signal tie artifact** | **-.23151988** | **EXACT ZERO** |
| basis88 | 1 | 6.0212493 | 6.0212493 | 1.5053123 |
| basis88 | 8 | 17.38969 | 2.1737113 | 4.3474226 |
| basis88 | 16 | 15.935482 | .9959676 | 3.9838705 |
| basis88 | 32 | 10.302008 | .32193774 | 2.575502 |
| basis88 | 64 | 22.359144 | .34936163 | 5.589786 |
| sched5000 | 1 | 8.016568 | 8.016568 | 2.004142 |
| sched5000 | 8 | 22.999302 | 2.8749127 | 5.7498255 |
| sched5000 | 16 | 19.046713 | 1.1904196 | 4.761678 |
| sched5000 | 32 | 18.514444 | .5785764 | 4.628611 |
| sched5000 | 64 | 29.71765 | .46433827 | 7.4294124 |
| invsqrt | 1 | 12.929879 | 12.929879 | 3.2324698 |
| invsqrt | 8 | 18.090397 | 2.2612996 | 4.522599 |
| invsqrt | 16 | 11.87521 | .7422006 | 2.9688025 |
| invsqrt | 32 | 3.6709278 | .11471649 | .91773194 |
| invsqrt | 64 | 53.427296 | .8348015 | 13.356824 |
| mlpdown4x | 1 | 6.126205 | 6.126205 | 1.5315512 |
| mlpdown4x | 8 | 21.787415 | 2.7234268 | 5.4468536 |
| mlpdown4x | 16 | 7.5498385 | .4718649 | 1.8874596 |
| mlpdown4x | 32 | 7.541202 | .23566256 | 1.8853005 |
| mlpdown4x | 64 | 28.184475 | .44038242 | 7.0461187 |

**DEMONSTRATED:** h=192 is structurally absent from all six portfolio/rate/breakeven/census reports (they have exactly five horizon rows). **HYPOTHESIS:** cutoff32 h=64 spread is arbitrary old-decile tie ordering of zero forecasts, not a forecaster trading measurement. Its arithmetic gross/4 is −3.704318 bps/side: no nonnegative cost breaks even. The reported zero is not evidence of costless viability.

### IC and standard error

All values **DEMONSTRATED**, latest `_signal`, held-out cross-section within-timestamp close IC averaged across timestamps; training step **UNVERIFIED**. Standard error is arithmetic `(reported IC +1 s.e.) − IC`, rounded below. This is not pooled Pearson, pooled Spearman, or the earlier standing estimate.

| Run | h=1 IC ± SE | h=8 | h=16 | h=32 | h=64 | h=192 |
|---|---:|---:|---:|---:|---:|---:|
| control | .055118 ± .021233 | .061668 ± .022543 | .058363 ± .019539 | .043896 ± .019272 | .033932 ± .020773 | .052252 ± .018545 |
| cutoff32 | .131398 ± .020131 | .079116 ± .022112 | .053980 ± .021249 | .051569 ± .018626 | **NaN ± NaN** | **NaN ± NaN** |
| basis88 | .077473 ± .018441 | .073131 ± .019441 | .066261 ± .018955 | .035541 ± .017430 | .018240 ± .021002 | .027884 ± .019602 |
| sched5000 | .088697 ± .020561 | .097065 ± .020728 | .074301 ± .019639 | .050244 ± .017247 | .029654 ± .020083 | .047731 ± .017629 |
| invsqrt | .134940 ± .019545 | .091019 ± .021454 | .070124 ± .018554 | .035201 ± .019200 | .021620 ± .022338 | .059493 ± .016903 |
| mlpdown4x | .078177 ± .020788 | .077651 ± .020464 | .069185 ± .020034 | .040447 ± .020861 | .032253 ± .021386 | .007957 ± .021320 |

### Close-only and delayed diagnostics at h=8

All **DEMONSTRATED**, latest `_tradable` and `_tradable_rates`, training step **UNVERIFIED**. Close-only market-neutral MSE ratios below use close-anchored persistence, not aggregate OHLC. Coverage here is held-out sample / held-out cross-section as labeled.

| Run | Held-out sample close ratio / delayed ratio | Held-out cross-section close ratio / delayed ratio | Held-out cross-section close hit / delayed hit |
|---|---:|---:|---:|
| control | .9843478 / .9984342 | .9826117 / .9890239 | .51825 / .51262814 |
| cutoff32 | .8981359 / 1.0196657 | .981291 / 1.0129468 | .53275 / .5233808 |
| basis88 | .99516 / 1.0069798 | .9878891 / .9936658 | .5165 / .51837957 |
| sched5000 | .9299351 / .9955738 | .9618375 / .97658587 | .54425 / .53213304 |
| invsqrt | .93482023 / 1.0132502 | .9597648 / .9806323 | .547 / .5261315 |
| mlpdown4x | .9826127 / .9922856 | .9866419 / .99286205 | .53675 / .53013253 |

**HYPOTHESIS:** cutoff's large aggregate h=8 advantage does not establish executable superiority: it becomes worse than persistence under the stored one-bar-delay MSE in both draws. These old delay diagnostics still must not be renamed as the refactored raw executable next-open payoff.

## Explicit NaN, zero, and absent-series register

All entries **DEMONSTRATED**, with latest-snapshot step attribution unverified where applicable.

- All six `_progress`, steps 1000/2000/3000: invalid forecast candles per forecast bar is **exactly 0**. This is the recorded geometry-invalid metric, not a trading return.
- All six `_generalization_gap`, steps 1000/2000/3000: `no gap 0.0` is **exactly 0**, an intentional reference line.
- All six latest `_signal`: held-out sample within-timestamp IC and both SE bands are **NaN at every horizon**; the `zero correlation` reference is **exactly 0**.
- All six latest `_portfolio` and `_portfolio_rate`: all held-out sample gross/net series are **NaN at all five reported horizons**; `zero` reference is **exactly 0**. All six latest `_portfolio_breakeven`: held-out sample series is **NaN at all five horizons**.
- All six latest `_cross_section_census`: held-out sample contributing timestamps is **exactly 0** at h=1/8/16/32/64. Held-out cross-section counts are populated, not zero.
- All six latest `_tradable` and `_tradable_rates`: h=1 one-bar-delay metrics are **NaN** in both draws; there is no remaining forecast hold after that delay.
- cutoff32 latest `_signal`: above h=32, pooled Pearson and within-timestamp IC/bands in both draws are **NaN** due to no forecast variation. Nonzero old pooled Spearman values at the long end are not proof of signal.
- cutoff32 latest `_tradable_rates`: above h=32, close-anchor hit rate and top/bottom signal-decile hit rates are **NaN** in both draws. The base has NaNs on **161 of its 192 horizon rows** (h=1 plus h=33–192); all other arms have NaNs only on h=1.
- cutoff32 latest `_portfolio_breakeven`, h=64: **exactly 0**, accompanying a negative gross spread and invalid zero-signal alpha interpretation, not a positive-viability measurement.
- All six: step 4000 is **structurally absent**. Historical h=16/32 at steps 1000/2000 is **not persisted** in the relevant histories. Historical trading snapshots at each best checkpoint are **not retained in these bases**. h=192 portfolio/rate/breakeven/census is **structurally absent**, not a failed-to-fire zero.

## Bottom line

**HYPOTHESIS — adoption decision:** Do **not adopt any knob as a demonstrated solution for the full 192-horizon accuracy-plus-executable-trading mandate**. The trained-full-horizon arms still show long-end deterioration; the one knob that cleanly prevents it, cutoff32, does so by removing long-end learning. Reject basis88 as unsupported, sched5000 as a long-rot cure, and mlpdown4x/invsqrt as full-horizon fixes. If the product explicitly narrows to short-horizon forecasting, cutoff32 is the clearest single-knob accuracy candidate, but not yet a demonstrated deployable trading improvement.

**DEMONSTRATED — best honest full-head checkpoint on common unweighted NLL:** invsqrt at step 2000, `_loss` **1.987704**, `_skill` **1.0003847**, `_horizon_steps` h=1 **.93821776**, h=8 **.9354169**, h=64 **.9882811**, h=192 **1.0089053**; `_calibration` **.7393138/.9552352**. This does not beat persistence over all horizons in aggregate and already loses at h=192. Its selected objective scalar is **1.6442281141699293**, a different quantity.

**DEMONSTRATED — best short-skill checkpoint:** cutoff32 step 2000, `_horizon_steps` h=1 **.8680577**, h=8 **.8683521**, long endpoints **.9980796/.9992614**; `_loss` unweighted NLL **2.3024793**, `_skill` aggregate ratio **.9962769**. The manifest instead selected step 1000 on weighted NLL **1.049701191543634**, whose h=8 ratio is **.8893328**. Do not call step 2000 the saved objective-selected checkpoint.

**Trading viability at those best checkpoints is unavailable, not proven.** Stored latest cross-section numbers demonstrate a small populated diagnostic draw, but are old-definition metrics with unverified checkpoint attribution, wide errors, incomplete horizon coverage, and zero-signal pathology in cutoff32's long end. They cannot certify selected-checkpoint trading profitability, full-universe performance, or the refactor's newly defined executable raw utility.