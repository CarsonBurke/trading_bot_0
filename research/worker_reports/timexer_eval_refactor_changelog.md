# Semantic changelog: evaluation refactor and the six arms’ metric vintage

**DEMONSTRATED — Scope distinction is essential.** The diff against `HEAD` includes several accumulated changes, not only the latest utility implementation: experimental model/optimizer knobs, weighted checkpoint selection, matched-step reports, a separate held-out cross-section draw, and the endpoint-utility replacement. More importantly, the six arms’ persisted report vocabulary is older than the current writer. I read the current implementation, the HEAD diffs, the relevant worker reports, and ran read-only `report_cli` commands for the cutoff arm’s loss and generalization gap. No files were changed, tests/builds run, or GPU work launched.

## 1. Changed meaning — trading metrics are the centrepiece

### 1.1 Historical spread/breakeven and current utility are different quantities

**DEMONSTRATED — Retire the historical figures as descriptions of the new utility, not as historical observations.** The old `4.8356795 bps` gross decile spread and derived `1.2089 bps/side` describe the historical market-neutral residual-log spread with a fixed `4 × cost` deduction. They are **not** measurements of current raw next-open utility and cannot be transformed into it from those two scalars. The historical report explicitly gives that formula and measurement (`research/worker_reports/timexer_trade_horizon.md:95-117`); the HEAD `runner.rs::Scorer::portfolio` diff confirms the old calculation used `bar_target × sigma`, full long-leg mean minus full short-leg mean, and `charge = 4 * cost`.

| Contract | Historical portfolio reports | Current utility | Evidence |
|---|---|---|---|
| Entry/exit | Origin-close-anchored residual coordinate through horizon h; no actual next-open execution | **Next observed bar’s open → the h-th next observed bar’s close**. At h=1 this is next-bar open-to-close, not origin-close-to-next-close and not a mid fill | `runner.rs:947-965`; `utility.rs:171-191` |
| Return convention | Market-neutral cumulative **log** return times 10,000 | **Raw simple** return `exp(log(P_exit/P_entry)) - 1`, multiplied by positions | `utility.rs:173-191,204-208` |
| Market drift | Payoff treated the market-neutral target as the spread return | Raw target restores realized market drift for outcomes. Forecast decisions remain based on market-neutral residual means and residual predictive dispersion; this is not a market hedge | `runner.rs:895-899,955-962`; `utility.rs:173-176` |
| Capital normalization | Long mean minus short mean: +1 long and −1 short, gross 2 under the spread convention | Initial gross **budget ≤1** for every policy and origin timestamp. Ordinary untied deciles allocate +0.5 and −0.5; cash/gates/ties can deploy less | `utility.rs:79-101` |
| Missing outcomes | Rebuilt ranks/decile sizes from horizon-valid names | Positions sized using original cohort membership and forecasts only; a missing entry or endpoint for **any** original member rejects that entire cohort, equally for every policy | `utility.rs:177-199`; `reports.rs:369-383` |
| Ties | Double argsort assigned ordinal membership, so equal forecasts could acquire arbitrary ticker-order trades | Fractional boundary ties; identical long/short selections cancel into cash | `utility.rs:47-81` |
| Transaction cost | Fixed `4c` per spread round trip | `c × Σ_i |w_i| (1 + P_exit,i/P_entry,i)`, per cohort. Entry is initial notional; exit is the changed notional of the same held shares | `utility.rs:214-221,263-274`; `docs/timexer_segment.md:185-188` |
| Breakeven | Historical `gross/4`; intermediate worker report also describes a zero floor for negative gross | **Signed** mean gross bps / mean turnover. Negative remains negative; zero turnover is NaN | `utility.rs:280-292`; `reports.rs:395-400` |
| Holding schedule / annualization | Described as non-overlapping holds and annualized with `19,656/h` | Independent endpoint cohorts; potentially overlapping, ticker-specific observed-bar exits, **no common UTC exit or capital schedule, no annualization, no Sharpe** | `reports.rs:88-101`; `docs/timexer_segment.md:211-219` |

**DEMONSTRATED — The 4× round-trip convention is no longer in force.** “Drifted turnover” means the exit fee is assessed on the exit value of the fixed shares, not on the entry notional again. For a fully deployed gross-1 book with unchanged prices it is 2 units of turnover, not 4. With changing prices it is not a fixed 2 either. The current breakeven is a cost allowance per transacted notional, not the old spread divided by four.

**DEMONSTRATED — The new evaluator is not a non-overlapping backtest.** `payoff/h` is bps of initial capital per **observed bar**; it is not calendar return, compounded portfolio growth, or evidence that overlapping cohorts could all be funded. Payoff standard deviation and worst observed payoff are descriptive endpoint-cohort summaries, not Sharpe, confidence bounds, or prospective risk limits.

**HYPOTHESIS — Operational implication:** keep `4.8356795` and `1.2089` only when explicitly identifying the historical h=1 market-neutral residual-log diagnostic. Stop calling `1.2089` the forecaster’s current executable fee ceiling. Current raw utility requires new actual-model measurements under the new policy/payoff contract.

### 1.2 Bases were retired, not silently reused for utility

**DEMONSTRATED — Against HEAD:** `timexer_segment_portfolio` and `timexer_segment_portfolio_sharpe` disappear from the registry/writer. They are replaced by distinct `timexer_segment_utility_*` names, not renamed files carrying equivalent quantities. The intermediate `_portfolio_rate`, `_portfolio_breakeven`, and `_cross_section_census` names present in the arms and worker report also have no current writer/registry entry (`shared/src/report.rs:10-78`; `reports.rs:1463-1577`; explicit retirement documented at `docs/timexer_segment.md:217-219`). They were not HEAD registry entries, so describing them as “removed versus HEAD” would be inaccurate; they are **retired versus the running-binary/intermediate-tree contract**.

**DEMONSTRATED — Existing files are not rewritten by changing the registry.** The current writer outputs different utility filenames; no conversion or old-name alias is present. Therefore old `_portfolio*` files continue to mean the old metric when read explicitly with `report_cli`. A new TUI using the current registry does not scan them. Comparing old `_portfolio` against new `_utility_payoff` as though only the name changed would be wrong, but the utility replacement itself does not silently overwrite the old name with a new definition.

### 1.3 Actual dangerous retained-name change: `_generalization_gap`

**DEMONSTRATED — The six-arm artifact vintage and current source disagree under the same base name.** Read-only command:

```text
./target/release/report_cli 1 timexer_segment_loss --run timexer-cutoff32-4k
1000 training NLL=1.3003862 held-out sample NLL=2.2986646 held-out sample persistence NLL=2.3947663
2000 training NLL=1.2710632 held-out sample NLL=2.3024793 held-out sample persistence NLL=2.3947663
3000 training NLL=1.2655503 held-out sample NLL=2.304595 held-out sample persistence NLL=2.3947663

./target/release/report_cli 1 timexer_segment_generalization_gap --run timexer-cutoff32-4k
1000 training minus held-out sample NLL=-0.9982784
2000 training minus held-out sample NLL=-1.031416
3000 training minus held-out sample NLL=-1.0390447
```

**DEMONSTRATED — Those saved gaps subtract objective-weighted training NLL from unweighted held-out sample NLL.** At 3k the saved gap is the displayed training-minus-held-out difference, subject to report f32 rounding. The current implementation instead computes `train_nll - validation_objective_nll` and labels both sides `objective NLL` (`reports.rs:731-747`). Its test explicitly distinguishes held-out objective 1.9 from unweighted 2.0 (`reports.rs:1870-1908`).

**DEMONSTRATED — This is kept-name/changed-meaning relative to the arms’ running binary, although `_generalization_gap` itself is NEW relative to HEAD.** The saved cutoff and inverse-weighting gaps are not apples-to-apples generalization gaps. Neither their levels nor comparisons with the corrected current writer may be used as evidence of period overfit without accounting for the different horizon weighting. Under `uniform`, the two definitions coincide.

### 1.4 `_loss`, selection, and manifest fields

**DEMONSTRATED — `_loss` keeps its base but now distinguishes objective and unweighted scoring.** Current labels are `training objective NLL`, `held-out sample objective NLL`, `held-out full objective NLL`, `held-out sample unweighted NLL`, `held-out full unweighted NLL`, and unchanged split-specific persistence NLL (`reports.rs:659-680`). The saved arms still use the older ambiguous `training NLL` / `held-out sample NLL` labels.

**DEMONSTRATED — Relative to HEAD, training NLL changes functional when `--horizon-loss` is nonuniform.** The objective is `Σ(w × mask × per-element NLL) / Σ(w × mask × CHANNELS)`. The weights have mean 1 across the full horizon, but normalization does **not** make averages over different horizon distributions the same benchmark (`model.rs:100-143,2221-2274`). The old held-out NLL functional survives as explicitly **unweighted NLL**, not as the objective scalar.

**DEMONSTRATED — Selection and early stopping minimize weighted NLL.** `weights/best`, `best_step`, and held-out sample patience use `evaluation.objective_nll`; epoch patience uses the corresponding held-out full objective (`runner.rs:1582-1593,1734-1757`). `best_preview_nll` is removed in favour of `best_objective_nll`; new `selection` authenticates `min held-out sample objective-weighted NLL (horizon-loss=<spec>)` (`runner.rs:427-440,517-522`). `Manifest.validation_nll` remains **unweighted** (`runner.rs:1836-1842`). `best_step` retains its field name but now identifies the minimizer of the weighted criterion, not necessarily the minimizer of the common aggregate.

**HYPOTHESIS — Ranking rule:** rank cross-arm held-out likelihood using the unchanged **unweighted** held-out sample NLL at matched steps and the same draw, together with per-horizon MSE ratios. Do not rank cutoff’s ~1.x selection objective against uniform’s ~2.x aggregate as an accuracy win. The source/report statements that weighted means are “directly comparable across modes” establish common units and scale invariance, not a common estimand (`model.rs:100-105`; `research/worker_reports/timexer_horizon_loss.md:35-60`).

### 1.5 Other retained names: presentation corrections versus arithmetic changes

**DEMONSTRATED — `_decomposition`: corrected interpretation, unchanged arithmetic.** The demeaned component is a **gain** `ρ² var(y)/mean(y²)`, not the MSE ratio `1−ρ²`; offset + best-scale demeaned gain + mis-scaling gain sum to close total. Old wording implied the wrong identity and counted the components incorrectly; current labels/doc fix that (`reports.rs:78-80,266-270,1195-1221`). Do not quote old prose as the statistic’s mathematical definition.

**DEMONSTRATED — `_tradable` and `_tradable_rates`: corrected execution claim, unchanged arithmetic.** The delayed series is the difference between h and first-bar forecast/target coordinates in market-neutral space. It is now labelled first-bar exclusion / not execution P&L, not “one-bar execution delay” (`runner.rs:1158-1174`; `reports.rs:1270-1307`). h=1 is structurally undefined.

**DEMONSTRATED — `_signal`: unchanged estimator, narrower claim about uncertainty.** The ±1 s.e. lines now say `iid approximation, not confidence`; calculations did not change (`reports.rs:1224-1245`; `runner.rs:1305-1323`). A new held-out cross-section series is an **additional draw**, not a redefinition of held-out sample or held-out full. The same new split is added to `_decomposition`, `_offset`, `_tradable`, and `_tradable_rates` (`reports.rs:1180-1194`). Do not interchange its values with historical held-out full values.

**DEMONSTRATED — `_timing`: added separate `held-out cross-section pass total`; existing primary evaluation timing is not silently widened to include it** (`reports.rs:815-873`; `runner.rs:1723-1733,1893-1899`). At a same-step held-out sample/full pair, split-agnostic timing chooses the first point; the terminal extra held-out full pass is not a second plotted total on that same step (`reports.rs:540-562`). This is accounting scope, not model accuracy.

**DEMONSTRATED — Same-step held-out full additions do not replace held-out sample measurements.** The writer now merges a sample/full pair into one step axis with separate split series (`reports.rs:480-562,617-625`), and mid-epoch normal termination requests an additional held-out full pass (`runner.rs:1766-1797,1910-1946`). The supplied OOM-interrupted arms did not complete that normal exit path; their 1k/2k/3k artifacts must not be described as completed 4k arms or as containing that final full pass.

## 2. New capability

### 2.1 What `utility.rs` computes, and whether it trains anything

**DEMONSTRATED — Key signatures:**

```rust
pub(super) fn evaluate(
    forecast: &Tensor,
    log_scale: &Tensor,
    target: &Tensor,
    valid: &Tensor,
    sigma: &Tensor,
    entry_log: &Tensor,
    groups: &Tensor,
    group_count: i64,
) -> Result<PortfolioCurve>

fn weights(&self, mean: &Tensor, std: &Tensor) -> [Tensor; 7]
fn deciles(&self, signal: &Tensor) -> Tensor
```

Evidence: `utility.rs:49,84,108-120`.

**DEMONSTRATED — Input contract:** forecast/log-scale/target/valid are `[origin windows, horizons]`; sigma/entry-log/groups are `[origin windows]`; groups are dense Int64 origin-timestamp ranks and every declared group must be populated. Forecast is the market-neutral cumulative close-log-return mean in causal origin-sigma units; log-scale is the logarithm of the close conditional residual standard deviation in those units. Target is **raw** cumulative close-log-return in origin-sigma units. Sigma converts coordinates to dimensionless log returns. Entry-log is `ln(next observed open / origin close)`. The generic utility comment says cumulative log-return coordinates; the caller establishes which inputs are market-neutral and which are raw (`utility.rs:108-165`; `runner.rs:947-965,1344-1357`).

**DEMONSTRATED — Seven frozen rules:** cash; equal long; `sign(mean)/N`; mean decile long/short; mean/std decile long/short; sign gated by `abs(mean) >= std` with inactive capital left in cash; `mean/(std²+mean²)` divided by `max(1, cohort sum absolute raw weights)`. This last rule is a diagonal quadratic log-return approximation, not Kelly and not a calibrated portfolio risk model (`utility.rs:12-20,84-101`).

**DEMONSTRATED — Output:** for h=`1,8,16,32,64,128,192` within available horizon, plus terminal horizon if absent, evaluate mean gross payoff, costs `0,0.5,1,2,5,10`, payoff/h, signed breakeven, gross/net exposure, active fraction, turnover, descriptive gross payoff std/worst, and the complete-cohort census (`utility.rs:10-20,163-309`; `reports.rs:369-406`). Means are equally weighted over complete eligible origin cohorts, not over names or elapsed calendar time.

**DEMONSTRATED — Evaluation-only, not a training objective or selection criterion.** The production call is `Scorer::portfolio → utility::evaluate`; scoring runs under `tch::no_grad_guard` (`runner.rs:1344-1357,1374-1389`). Training loss contains weighted Gaussian NLL only and unweighted diagnostic MSE (`model.rs:2221-2274`). Checkpoint selection reads objective NLL, never utility (`runner.rs:1734-1757`; `docs/timexer_segment.md:59-61`). The decision it informs is whether fixed forecast-to-position rules, including uncertainty-aware rules, retain descriptive endpoint payoff after stated costs; it does not optimize or select those rules on held-out data.

### 2.2 Additive held-out cross-section draw

**DEMONSTRATED —** `cross_section_origins(corpus, origins)` groups existing held-out origins by origin timestamp. `cross_section_blocks(stamped)` drops blocks narrower than 40, considers at most 128 widest blocks, caps width at 256, and chooses the uniform operating point maximizing `k × (width−1)`. It restores selected timestamps to chronological order and strides names inside each block (`runner.rs:599-660`). No qualifying timestamp is a hard error, not a blank successful evaluation.

**DEMONSTRATED —** This changes neither `fixed_origins` nor the held-out sample: `preview` and `cross_section` are separate locals (`runner.rs:586-598,1544-1551`). It supplies extra trading-diagnostic series at report intervals and in standalone evaluation (`runner.rs:1723-1733,1984-1989,2072-2080,2134-2141`). Selection and primary NLL/MSE do not consume the second draw.

**HYPOTHESIS —** The implementation’s `k(width−1)` is an information-motivated sampling heuristic, not a demonstrated universal optimality theorem for real dependent/heterogeneous ticker returns. Source/report assertions about exact inverse-variance optimality should not be treated as measured market evidence. Comments claiming exactly 256 names or uniform time coverage are stale relative to negotiated width/widest-first selection.

### 2.3 Complete registry delta

**DEMONSTRATED — NEW versus HEAD:**

- `timexer_segment_generalization_gap`
- `timexer_segment_horizon_steps`
- `timexer_segment_horizon_steps_scaling`
- `timexer_segment_horizon_loss_weight`
- `timexer_segment_lr_trajectory`
- `timexer_segment_utility_payoff` and suffixes `_cost_0p5`, `_cost_1`, `_cost_2`, `_cost_5`, `_cost_10`
- `timexer_segment_utility_rate` and the same five cost suffixes
- `timexer_segment_utility_breakeven`
- `timexer_segment_utility_gross_exposure`
- `timexer_segment_utility_net_exposure`
- `timexer_segment_utility_active_fraction`
- `timexer_segment_utility_turnover`
- `timexer_segment_utility_payoff_std`
- `timexer_segment_utility_worst_payoff`
- `timexer_segment_utility_census`

**DEMONSTRATED — REMOVED versus HEAD:** `_portfolio`, `_portfolio_sharpe`. **Additional intermediate/artifact names retired:** `_portfolio_rate`, `_portfolio_breakeven`, `_cross_section_census`. **Kept-name materially changed versus intermediate running code:** `_generalization_gap`; `_loss` also changes its series taxonomy to distinguish objective/unweighted. Other retained-name presentation/split additions are listed above rather than hidden under “unchanged.”

**DEMONSTRATED — Registry and TUI agree by construction.** `tui/src/main.rs:425-430` extends `meta_chart_bases` directly from `shared::report::TIMEXER_SEGMENT_REPORT_BASES`, sorts, and deduplicates. `tui/src/main.rs:1201-1227` contains a bidirectional registry test. Current utility writer names match the registry, including dynamically constructed cost suffixes (`reports.rs:1547-1577`; `shared/src/report.rs:41-61`). No separate stale portfolio list is retained in this TUI path. This is code inspection, not a fresh test-pass claim.

### 2.4 FORMAT, OBJECTIVE, manifest, compatibility

**DEMONSTRATED — Against HEAD, FORMAT moves from v7 to v10**, selecting x0-learned/x0-none and appending `-mean-<spec>`; OBJECTIVE moves from `causal_patch_market_neutral_nll_v3` to `_v5` (`runner.rs:240-289`). The intermediate history is v8 x0/config attribution, v9 weighted objective/selection, v10 structured means. This is broader than utility: utility itself does not introduce a separate new format/objective stamp.

**DEMONSTRATED — Manifest additions:** `model.x0_lambdas`, `model.horizon_loss`, `model.horizon_mean`; `base_learning_rate`; `scalar_lr_mult`; optional `max_steps` and `termination`; `selection`; renamed `best_objective_nll`. `optimizer_recipe` now records x0/scalar-bank choice, schedule budget/shape/start, MLP-down rate, and recipe v2 (`runner.rs:386-440`; `compute.rs:173-193,322-333`). `learning_rate` remains the scheduled/applied rate of the checkpoint step, not the configured base; current Engine reads applied state rather than recomputing it (`compute.rs:772-777`). `NUMERICS` is unchanged.

**DEMONSTRATED — Compatibility:** loader checks current format/objective, exact x0/mean agreement, selection/horizon-loss agreement, manifest digest and weight bytes (`runner.rs:451-515`). v7/v8/v9 manifests are rejected despite some modes retaining compatible tensor shapes. Basis head rows change from `2×CHANNELS×pred_len` to `CHANNELS×(free+functions+pred_len)`; disabled x0 removes its parameter bank (`runner.rs:247-280`; `model.rs:352-355`). Optional cap/termination fields omit absent values, avoiding an additional digest break solely for the cap. Historical report binaries remain readable independently of checkpoint loading.

## 3. Unchanged — the common accuracy yardsticks survive

**DEMONSTRATED — Held-out sample unweighted NLL, including the historical ~1.9965 quantity, keeps its definition.** The refactor still computes `nll_elements = 0.5*((target−prediction)/std)^2 + log(std)`, omitting the Gaussian additive constant as before; sums masked valid elements and divides by valid bars × four channels (`model.rs:2310-2313`; `runner.rs:899-906,978-993,1094-1110`). The fixed held-out sample is not replaced by the cross-section draw. Compare old `held-out sample NLL` with current `held-out sample unweighted NLL`, not a differently weighted objective.

**DEMONSTRATED — Persistence is unchanged.** Its NLL uses zero mean and `half_log_horizon`, i.e. std √h in origin-sigma units, the same targets/mask, and the same unweighted denominator (`runner.rs:901-906,979-982,1105-1108`). Neither utility nor horizon weights enter that baseline. My cutoff report read shows **exactly 2.3947663** at all three saved steps; the parent supplied that all six arms printed the same value. This corroborates the unchanged sampling/normalization path but is not a fresh recomputation of the corpus.

**DEMONSTRATED — Per-horizon market-neutral MSE ratios versus persistence are unchanged.** Numerators are masked four-channel squared prediction errors; denominators are masked target squares; horizon sums/counts and ratio construction do not use objective weights (`runner.rs:889-900,997-1020,1081-1123`). The raw counterparts still add realized market drift to the **target**, not forecast, for comparison with raw persistence. `_horizon_steps` merely preserves selected h=1/8/64/192 values along optimizer steps (`reports.rs:318-366`). A `cutoff:32` arm’s long horizons are still emitted/evaluated but not trained; an unchanged long-end ratio is not evidence that long-end signal improved (`model.rs:118-123`).

**DEMONSTRATED — Historical IC 0.029958 retains its estimator only on the same split/draw.** Within-timestamp Pearson over market-neutral close coordinates, ≥20 valid names with nonzero spread, equal timestamp average and existing iid standard-error arithmetic are unchanged (`runner.rs:1187-1210,1305-1323`). It is not a raw-return utility statistic, not pooled Pearson, and not an IC across all 4,873 names. A held-out cross-section IC uses a newly selected subset, so it is not directly interchangeable with historical held-out full IC. Historical report provenance also restricts the old permuted checkpoint’s valid close-only trading evidence to h=1 (`research/worker_reports/timexer_eval_bisect.md:130-161`).

**DEMONSTRATED — Calibration, robust errors/rates, offset/decomposition arithmetic, price errors and candles are not redefined by the utility path.** Existing masked forecast arrays remain separate from new unmasked policy means, close scales, raw outcomes, and next-open entry arrays (`runner.rs:929-975`). Their new extra split, where present, is explicitly labelled; no existing sample/full split is relabelled as cross-section.

## 4. Verification and zero/NaN hazards

**DEMONSTRATED — Tests exist; latest execution is NOT established by this audit.** Current tests cover:

- exact next-open simple payoff and drifted-exit cost (`utility.rs:350-373`);
- uncertainty-controlled decisions and cash-preserving abstention (`utility.rs:376-399`);
- distinct mean/std ranking under the same budget (`utility.rs:402-423`);
- tied forecasts not manufacturing trades (`utility.rs:426-436`);
- missing outcomes dropping whole cohorts and zero eligible cohorts producing NaN (`utility.rs:439-459`);
- terminal-horizon decisions, ≤1 gross budget, outcome-independent exposure (`utility.rs:462-484`, with function beginning earlier in the file);
- scorer integration for restored raw drift, next open, close uncertainty, uneven batches and missing endpoints (`runner.rs:3122-3189`);
- underpopulated cohort NaNs with positive/finite census (`runner.rs:3260-3303`);
- utility report serialization preserving costed payoff, cash zero and missing-evidence NaN (`reports.rs:1743-1786`);
- weighted selection scalar versus scalar reference and exact uniform equality (`runner.rs:2344-2430`);
- unchanged core scorer/trading scalar references and persistence identity (`runner.rs:2220-2340,2434-2684,2982-3077`).

**DEMONSTRATED — Historical worker reports claim scoped test passes and synthetic read-back probes, but those reports predate the latest utility replacement.** They do not prove current utility tests passed. The read-only constraints prohibit running tests/builds here. I found no worker report with a current utility execution log; the current docs describe the implementation, and test source shows intended coverage. No fixture number in this audit is a forecaster measurement.

**DEMONSTRATED — Important current zero/NaN semantics:**

1. **Zero complete eligible cohorts ⇒ NaN policy outcomes**, not zero payoff. `utility.rs:247-259` converts every policy column to NaN; census still reports zero contributors and observed widths. Cash with actual observations correctly reports zero payoff/exposure/turnover and **NaN breakeven**. Identical decile signals cancel to cash.
2. **Zero close signal/target variance ⇒ undefined IC**, while the internal s.e. can be **exactly zero** with no contributing timestamps because its denominator is clamped. The plotted bands remain NaN since they add to NaN mean IC (`runner.rs:1305-1323`; `reports.rs:1231-1245`). Do not interpret the internal zero s.e. as certainty.
3. **h=1 first-bar-excluded ratio/hit rate ⇒ NaN by construction**, not a failed model evaluation (`runner.rs:1158-1174,1324-1329`).
4. **Weighted selection with zero weighted observations is structurally capable of returning 0**: `objective_nll = sums[12] / max(sums[13], MIN_POSITIVE)` (`runner.rs:1106`); training also clamps objective count (`model.rs:2268-2273`). This audit does not demonstrate that the corpus can reach that state in an arm, but the zero-denominator behavior exists and lacks an explicit “no objective observations” error here.
5. **Old zero spread is unsafe.** The HEAD portfolio averaged over `periods.clamp_min(1)` and could return a clean zero when no timestamp qualified. The newer utility fixes this, but it does not retroactively repair old saved reports. Intermediate artifacts also floor negative breakeven to zero and retain arbitrary ordinal tied deciles, as documented in the older worker report; census and IC must accompany any verdict about those artifacts.
6. **Report completeness is not transactional.** Horizon and trading reports are sequential writes, not an atomically committed evaluation bundle (`runner.rs:1958-1992`; `reports.rs:1422-1577`). A horizon matching the latest step is not by itself proof that every later base finished at that step. Horizon-indexed trading files are overwritten snapshots, not 1k/2k/3k trajectories. Use their title metadata/provenance when aligning them; `report_cli`’s first column for these bases is horizon, not training step.

## 5. What the worker reports actually document

**DEMONSTRATED — `timexer_exp_knobs.md` is an earlier experimental-knob implementation report, not a report of the latest utility evaluator.** It documents scalar LR multiplier, true removal of x0 injection/parameters, existing global LR knob, optimizer recipe attribution, v8/v4 stamps, and historical scoped checks. Its stamp statements are superseded by current v10/v5 source.

**DEMONSTRATED — Evaluation-related reports form a chronology, not one current specification:**

- `timexer_trading_eval.md` built the older seven diagnostic bases and market-neutral residual-log portfolio/annualized Sharpe; its execution interpretation and some decomposition prose are corrected now.
- `timexer_trade_horizon.md` added horizon/cost sweeps and breakeven/rate reports, discovered empty held-out sample cross-sections, then added a separate draw and negotiated uniform width. Its portfolio/census names, fixed-4× cost, non-overlap language, and synthetic verification figures describe the **retired** evaluator, not utility.
- `timexer_horizon_loss.md` built horizon weighting and objective-aligned selection but explicitly left the old loss/gap presentation alone. Current source subsequently corrects that gap mismatch and adds objective/unweighted labels.
- `timexer_horizon_steps.md` built the matched-step transpose and gap; the horizon-indexed originals remain latest snapshots.
- `timexer_eval_speed.md` moved/scoped scoring reductions and timing; its performance fixtures are not accuracy evidence.
- `timexer_report_clarity.md` reorganized historical report units/names and vocabulary, not the latest payoff convention.
- `timexer_robust_eval.md` reports an earlier real checkpoint’s robust measurements, not measurements under current utility.
- `timexer_eval_bisect.md` establishes the historical head-layout incompatibility and why the old checkpoint’s h=1 close-only IC/spread can survive while other reported horizons do not.
- `timexer_nextlat_lens.md` discusses cost-aware utility **as a proposed training objective**. That proposal did not become the current training loss: `utility.rs` is evaluation-only.

**HYPOTHESIS — Bottom-line use:** the six arms remain rankable at their actually available matched 1k/2k/3k steps on common unweighted held-out sample NLL and unchanged per-horizon accuracy. Their old portfolio snapshots must retain their old residual-log contract; the new raw utility is neither a reinterpretation of those files nor a validated profitability verdict. The saved mixed-objective generalization gaps are the principal silently retained-name semantic trap.