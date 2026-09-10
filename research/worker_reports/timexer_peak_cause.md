# Step-2000 peak: repeated economic supervision is demonstrated; its causal role remains a hypothesis

## Executive verdict

**DEMONSTRATED:** The run does not repeat *row references*, but it does repeatedly supervise the same raw economic `(ticker, absolute origin, horizon)` outcomes. The default row stride is 192 bars, while each row supervises 360 history-valid origins spanning 5,760 bars on the same stride-16 lattice. Interior outcomes therefore occur in **exactly 30 rows**, with different preceding contexts and causal normalizations. The progress scalar counts only each row's terminal owned future bars; it is not the union of densely supervised outcomes.

**HYPOTHESIS:** Repeated weak-signal fitting, interacting with chronological shift and shared horizon learning, is now the leading explanation of the peak. Neither the overlap arithmetic nor its timing establishes causation. In particular, **multiplicity is horizon-flat in the interior and cannot explain the selective h=16–64 collapse by itself**.

**DEMONSTRATED, scope-qualified:** The inspected speedrun records show monotone improvement, late cooldown catch-up, isolated small validation upticks, and parameter instability at explicit schedule transitions—not an established counterpart of this sustained financial IC collapse. Their evidence supports cooldown as an achievable-loss-floor lever, not as the demonstrated cause or cure of our disease.

## Evidence conventions and scope

- **DEMONSTRATED** denotes supplied observations, inspected source/records, or the executed arithmetic described below. It does not promote a record author's causal interpretation into an independently controlled result.
- **HYPOTHESIS** denotes a mechanism, transfer prediction, cost estimate, or untested intervention.
- Internal aliases: `M` = `trading_bots/src/torch/timexer_segment/model.rs`; `C` = `.../compute.rs`; `P` = `.../corpus.rs`; `R` = `.../runner.rs`; `V` = `.../reports.rs`.
- External aliases: upstream root `/home/marvin/Documents/repositories/modded-nanogpt`; `S` = `records/track_1_short`; `D` = `records/track_2_medium`; `T` = `records/track_3_optimization`. **DEMONSTRATED:** this checkout names its second track `track_2_medium`, not `track_2_long`.
- **DEMONSTRATED:** I read `research/worker_reports/timexer_nanogpt_ledger.md` first. This extends it rather than repeating its 60 technique rows. Source line numbers below name the versions inspected during concurrent sibling edits; function names identify the stable anchors.
- **DEMONSTRATED:** No files were written, GPU jobs submitted, model experiments launched, or builds/tests run. Three read-only `python3 -B -c` arithmetic processes exited successfully. They checked the exact formula against explicit enumeration in 168 small geometry cases, calculated stride saturation curves, and read the original run manifest to aggregate the actual ticker-edge distribution. These are geometry/occupancy calculations, not model-performance measurements.

## 1. Exact repetition, distribution, and accounting reconciliation

### 1.1 Code path establishing the distinction

**DEMONSTRATED:**

1. `P:708–711` enumerates row references with `step_by(pred_len)`, hence stride192 at this run's settings.
2. `C:902–906`, `Engine::forward_loss`, calls both forward and targets with `last_only=false`.
3. `M:1631–1647`, `future_windows`, unfolds all patch origins with stride16; `M:1595–1626`, `statistics`, masks origins with fewer than256 history bars.
4. `M:2422–2429`, `targets`, multiplies future validity by the origin-history mask; it does **not** restrict supervision to the row's terminal192-bar ownership block.
5. `P:1170–1206`, `fill_row`, writes all6,000 context bars as valid and masks only the unwritten future tail.
6. The original manifest, `training/runs/timexer-invsqrt-ic-6k/weights/preview-best/manifest.json:3–23`, confirms context6000, patch16, horizon192, minimum history256, x0 disabled, inv-sqrt objective, free horizon means.

### 1.2 Exact formula

**DEMONSTRATED by code-derived arithmetic and executed enumeration:** Work in a ticker's **logical valid-bar ordinals**, not raw file offsets. Let:

- `T` be the exclusive purged training end;
- `N = ceil((T−6000)/192)` be its number of row references;
- row index `n` range from0 through `N−1`;
- row-final origin `r_n = 5999 + 192n`;
- patch number `k` range from1 through375.

The internal absolute origin is

`o(n,k) = 192n + 16k − 1`.

The history mask retains exactly `k = 16,…,375`: **360 active origins**. Their first and last values are `r_n−5744` and `r_n`. For a fixed target endpoint `b` and horizon `h`, put `o=b−h`. If `o` is not congruent to15 modulo16, multiplicity is zero. Otherwise define

`q = (o−255)/16`.

The exact multiplicity is

`M_h(b) = 1[b < T] · max(0, min(N−1, floor(q/12)) − max(0, ceil((q−359)/12)) + 1)`.

This formula includes the history mask, finite ticker endpoints, and the partially valid final future block. It is zero when the interval of allowed row indices is empty.

**DEMONSTRATED:** For a sufficiently long ticker with a complete last future block, the multiplicity histogram across eligible fixed-horizon lattice outcomes is:

- 24 outcomes at each multiplicity1,…,29:12 on the left ramp and12 on the right ramp;
- `12N−348` outcomes at multiplicity30.

For short tickers, the ramps overlap; the exact formula above handles them. A partial last block removes some of the rightmost multiplicity-one outcomes, increasingly with h. Thus **30 is an interior value, not the corpus-wide mean**.

### 1.3 Actual original-corpus distribution

**DEMONSTRATED arithmetic over the original run manifest:** There are4,873 tickers,2,455,276 row references,470,946,393 terminal-owned training target bars, and752,901,364 valid bars across the entire corpus. The470,946,393 number is not the entire corpus's valid-bar count.

| Horizon | Distinct supervised `(origin,h)` outcomes | Mean full-row-epoch multiplicity | Fraction at M30 | Expected unseen at step2000 |
|---|---:|---:|---:|---:|
|1|31,159,116|28.367280|89.1660%|1.51304%|
|8|31,158,956|28.367420|89.1665%|1.51264%|
|16|31,158,731|28.367618|89.1671%|1.51208%|
|32|31,157,919|28.368331|89.1695%|1.51006%|
|64|31,155,091|28.370815|89.1776%|1.50301%|
|128|31,144,636|28.380004|89.2075%|1.47695%|
|192|31,127,634|28.394959|89.2562%|1.43453%|

The common counts at multiplicities2 through30 are, respectively:

`119100, 117912, 117168, 116808, 117048, 116976, 117156, 117768, 117096, 116520, 117060, 116304, 116052, 116016, 116292, 116328, 115788, 115992, 115992, 115848, 115704, 115488, 115500, 115128, 115128, 115032, 115020, 114912, 27783348`.

Multiplicity-one counts for h1/8/16/32/64/128/192 are:

`118632 / 118472 / 118247 / 117435 / 114607 / 104152 / 87150`.

**DEMONSTRATED:** The very small horizon variation is a boundary effect. It is nowhere near enough to explain .1137→.0086 at h16 versus .1575→.1440 at h1. The interior multiplicity is exactly the same for both.

### 1.4 Occupancy is not an epoch definition or a causal result

**DEMONSTRATED conditional arithmetic:** Let `R_total=2,455,276` and draw `s=512,000` distinct rows, corresponding to2,000 steps at batch256. For an outcome represented in M rows:

`P(unseen) = C(R_total−M,s) / C(R_total,s)`.

For interior M30, this is `.0008974387`, with mean exposure count `30s/R_total = 6.255916`. But after weighting the actual edge histogram, **global expected coverage is about98.49%, not99.91%**. The latter is the interior result only. These are expectations under uniform shuffled row selection, not measurements of the realized seed's exposure histogram.

**HYPOTHESIS:** Coverage saturation and repeated fitting may help set the peak. Calling step2000 “the actual end of the epoch” is unjustified: saturation is gradual, boundary outcomes behave differently, and the threshold has not been causally tied to skill.

**DEMONSTRATED:** These repeats are raw economic outcomes, not identical tensors. Causal sigma, market beta, preceding context length and context content differ across row presentations (`M:1595–1626`, `statistics`). They can therefore act as useful augmentation as well as repeated supervision.

### 1.5 Why the log says98,205,945 unique target bars

**DEMONSTRATED:** `P:883–904`, `host_batch`, sets `Batch.valid_target_bars` to the sum of each row's terminal owned target count. `R:1829,1878` reads and adds that number; `R:2214` prints it as “unique target bars”; `V:1296–1301` divides it by the terminal-owned training total.

At step2000:

`512,000 × 192 = 98,304,000`.

The supplied count98,205,945 is98,055 lower, consistent with partial terminal blocks. This counter is legitimate **terminal block ownership progress**, but it is not dense-label-union progress, dense-origin occupancy, or an independent-sample estimate. Consequently .417 does not establish absence of repeated supervision.

## 2. Ranked mechanisms and resolving experiments

All metric arrows in the following table are **HYPOTHESIS predictions during post-peak continuation**, unless explicitly marked otherwise. “Beta” means the forecast calibration slope, not the causal market-exposure coefficient. “Gap” should compare the same loss definition and origin/history geometry on train and held-out; the current dense training objective and final-origin held-out scoring are not automatically a clean statistical gap (`V:171–172`; `C:902–906`).

|Rank / mechanism|Evidence and status|Training NLL|Held-out unweighted NLL|Per-horizon IC|Calibration beta / amplitude|Generalization gap|Explains observed h-pattern?|Resolving experiment and budget|
|---|---|---|---|---|---|---|---|---|
|**1. Repeated economic supervision enables weak-pattern fitting**|**DEMONSTRATED overlap; HYPOTHESIS cause.** Exact formula and original-manifest histogram above; `P:708–711`, `C:902–906`, `M:2422–2429`.|↓|↑ once repeated fitting hurts transfer|↓ in fitted nontransferable directions|Typically beta↓ and/or nontransferable amplitude↑; amplitude growth is not required|↑|**No by itself. M is flat in h.** Requires signal/noise, regime or objective selectivity.|**Primary intervention:** every30th row per ticker, random per-ticker residue, stride5760; fixed selected rows across epochs; repeat to1,920 steps, eval100, schedule budget9590. First pass~320steps; observe beyond it. **HYPOTHESIS cost:**414.7s stepping+~30s startup, leaving~155s for evaluation under600s. K4 interpolation arm below.|
|**2. Chronological shift: later optimization fits train-period-specific structure**|**DEMONSTRATED split/order; HYPOTHESIS cause.** `P:695–723` separates training and future validation; `R:1787` shuffles rows. Late optimizer steps do **not** mean later training dates.|↓|↑|Stable short-horizon structure may survive while medium-horizon regime relationships deteriorate|Beta↓ if forecasts become less aligned; amplitude need not change|↑|**Plausible, not sufficient:** explains selective transfer failure only if medium-horizon relationships shift more; does not derive the h16–64 trough or h192 recovery.|Train with a strictly excluded in-period time hole whose exclusion uses the full dense supervised support, then score that hole and future validation on matched draws. If in-period IC persists while future IC collapses, supports shift. A newly reserved hole requires a fresh training run. **HYPOTHESIS cost:**one≤2,500-step arm; use~2,300 if evaluation overhead needs margin.|
|**3. Overlapping returns and correlated market-time observations reduce effective information, especially at longer h**|**DEMONSTRATED overlap geometry; HYPOTHESIS learning effect.** `M:future_windows`, cumulative-return targets and shared market subtraction.|↓ through coherent repeated noise fitting|↑|Longer-h IC more fragile|Beta↓ / noise amplitude↑|↑|**Partial.** Simple return-overlap model gives no extra overlap penalty at h16 versus h1, and predicts increasing fragility through h192, unlike the exact observed shape.|Score uncertainty with date/ticker blocks; compare nonoverlapping-origin estimators and target-basis intervention separately from row stride. **HYPOTHESIS cost:** bounded checkpoint scoring or one≤2,500-step target intervention; no claim of exact runtime for pending evaluators.|
|**4. Shared representation receives harmful low-SNR horizon pressure**|**DEMONSTRATED shared backbone/head and dense Gaussian objective; HYPOTHESIS gradient conflict.** Ledger's verified shared-head analysis; `C:902–906`; `M:forward`/`losses`. Inv-sqrt arm still collapses; basis8:8 already rejected.|↓|↑|Selective medium/long-h IC↓ while h1 remains stable is possible|Beta↓ and/or amplitude↑ in harmful mean directions|↑|**Plausible, not established.** It can supply selectivity but must reproduce h16–64 rather than merely “longer is worse.”|Pending latent/head probes and orthogonal target intervention; distinguish deteriorating backbone information from readout fitting. Keep original-space NLL/IC evaluation. **HYPOTHESIS cost:** bounded existing-checkpoint probe or≤2,500-step arm; do not relaunch rejected basis8:8.|
|**5. Residual cross-sectional market exposure after approximate beta removal**|**DEMONSTRATED approximate correction; HYPOTHESIS failure.** `M:16–20` beta prior256, `statistics` ridge toward1, `targets` subtracts estimated market drift.|↓ if period-specific factor relationships fit|↑ under factor/regime mismatch|IC↓ only if heterogeneous residual exposures reorder names; a common additive market term alone does not change within-timestamp ranks|Calibration beta↓; residual exposure may grow; amplitude need not grow|↑|**Partial.** Can harm medium-horizon ranks but no derivation of the precise horizon trough.|At frozen early/late checkpoints, measure prediction/target exposure to market and sector components and residualized IC with fit/eval separation. Do not fit nuisance removal on the same draw and call it tradable skill. **HYPOTHESIS cost:**small bounded scoring jobs under10m; implementation-dependent, no invented runnable CLI.|
|**6. Unfinished cooldown leaves a better minimum unreached**|**DEMONSTRATED upstream late improvements and our flat LR; HYPOTHESIS remedy.** `C:275–306`; S/CautiousWD README46–59; numerical appendix below.|Usually↓ or plateaus|May↓ during cooldown|No fixed sign; may stabilize noisy readout, cannot guarantee rank recovery|May move beta toward1 if noise/amplitude is reduced|May narrow|**No intrinsic horizon selectivity.** Not the demonstrated collapse mechanism.|Hold peak LR unchanged, schedule-budget2500, cap2500, same data/order/objective, if retained as lower-priority arm. **HYPOTHESIS cost:**540s stepping+~30s startup leaves only30s evaluation margin;≤2300 steps with correspondingly matched budget is safer if overhead exceeds it.|
|**7. Generic divergence, absent WD, scalar/x0 instability, or scalar amplitude miscalibration**|**DEMONSTRATED counterevidence.** Stable supplied training decline; actual trunk WD; rejected scalar/global-LR/x0 arms; positive gains cannot change ranks.|Generic divergence would usually↑/spike, contrary to supplied trend|↑|Broad collapse expected, or no IC effect for positive scalar calibration|Explosive or globally mis-scaled outputs under divergence; calibration only rescales|Varies|**Does not fit as a standalone account.**|No repeat of refuted arms. Existing measurements are the exclusion evidence.|

**HYPOTHESIS ranking interpretation:** Rows1–4 can coexist. The arithmetic reopens ordinary repeated-outcome overfitting; it does not prove that two separate causal diseases are required. One coupled mechanism—repeated fitting of regime-dependent low-SNR targets—could yield both repetition and horizon selectivity, but neither component has been isolated yet.

## 3. Correct experimental specifications

### 3.1 Structured row stride: expected saturation triples

**DEMONSTRATED analytical predictions for interior outcomes, before ticker-edge and selected-residue corrections:** Let K select everyKth original row within each ticker. The selected row stride is192K. Preserve the selection across epochs; shuffle selected rows normally.

|K|Row stride|Interior multiplicity|Approximate steps per selected-row epoch at B256|90% interior coverage|99%|99.9%|Coverage matching control step2000|
|---:|---:|---|---:|---:|---:|---:|---:|
|1|192|30|9590.92|708.59|1364.83|1972.58|2000|
|4|768|7 or8, half of origin phases each|2397.73|636.64|1108.32|1456.51|1470.40|
|30|5760|1|319.70|287.73|316.50|319.38|319.41|

**DEMONSTRATED formula:** For K4, with `f=s/R_selected`, expected interior unseen fraction is `.5(1−f)^7 + .5(1−f)^8`. For K30 it is `1−f` during the first sweep. The precise finite-population calculation uses the hypergeometric formula and actual multiplicity histogram. A whole number of batches, random per-ticker residue, short tickers, excluded in-period holes and boundaries change exact epoch length and coverage. The nominal counts above are not substitutes for the actual selected-pool report.

**HYPOTHESIS preregistration:** If a near-saturation occupancy threshold controls the peak, K30 should peak much earlier than K4, which should peak earlier than control: roughly320 /1470 /2000 using the matched **interior** threshold. Declare the threshold before looking at curves. Global99.9% thresholds are different because ticker edges are substantial.

**Decision rules, corrected:**

- **HYPOTHESIS:** A K30 peak near320 followed by deterioration, together with an intermediate K4 peak near1470 and comparable attainable skill, would strongly support occupancy-linked fitting. It would not alone demonstrate that overlap is the sole cause.
- **DEMONSTRATED methodological point:** Stopping K30 at320 right-censors the curve. It cannot show a peak. Run at least~960 steps; the proposed1,920-step arm is better.
- **HYPOTHESIS:** Low skill at320 can reflect insufficient optimizer updates, removal of useful contextual augmentation, or altered boundary support. It does not uniquely identify any one of those.
- **HYPOTHESIS:** K30 repeated to1,920 versus control around2,000 is a useful near-matched test of diverse-context repeats versus fixed-context repeats. It is **not perfectly isolated**: first-exposure timing, per-outcome exposure-count dispersion, edge support and the80-step difference remain. A step2000 checkpoint for both would improve matching if available within the cap.
- **HYPOTHESIS:** If K30 is no worse after approximately six fixed-context sweeps, diverse-context repeats have not shown a necessary advantage in that comparison. If it is worse, augmentation is a candidate explanation, not the only one.
- **HYPOTHESIS / explicit non-prediction:** Occupancy alone gives **no numerical prediction** for h16 or h64 peak IC, nor a justified guarantee that either improves. It predicts a possible shift in timing under an additional causal assumption. Preserve the h1/8/16/32/64/128/192 profiles as separate acceptance evidence.

**DEMONSTRATED geometry:** Stride5760 makes the active absolute-origin sets disjoint:360 origins×16 bars. It does not make all underlying return shocks disjoint across row boundaries for h>16. Stride6000 also avoids repeated origins but leaves gaps in the active origin lattice and yields roughly307 steps per epoch, with exact counts requiring selected references. Stride5760 is the cleaner origin-coverage tiling.

**HYPOTHESIS cost:** Model shapes and per-step arithmetic are unchanged. K30 changes the amount of data in an epoch, not the FLOPs per step. At216ms/step,1,920 steps cost414.72s; startup~30s gives~445s before evaluation. K4/control/phase at2,500 steps cost540s+startup~30s, leaving only~30s for all evaluation and shutdown. A10-minute queue limit is a hard backstop, not evidence that these longer arms will finish. Use measured evaluator time or reduce the cap to leave real margin. No arm may use the barred v12 binary.

### 3.2 Why random row thinning is not the saturation falsifier

**DEMONSTRATED combinatorial invariance:** Choose a uniform subset of fR rows, then select s rows uniformly without replacement from that subset. Marginally, those s rows are a uniform s-subset of the original R. Therefore

`P(unseen at s) = C(R−M,s)/C(R,s)`

is exactly unchanged at every matched step before subset exhaustion. The reduced pool and reduced expected multiplicity cancel. A25% random-row subset **does not** move step2000 occupancy to step500. A peak staying at2000 would not refute occupancy. This invalidates the originally proposed quarter-row falsifier.

**HYPOTHESIS:** Restricting distinct support while retaining dense local multiplicity—whole tickers or sufficiently wide time blocks—can shift saturation earlier, but changes ticker/regime diversity and therefore peak-height interpretation. Those interventions are not required for the initial K30/K4 plan.

### 3.3 Patch-phase randomization: augmentation, not new independent information

**DEMONSTRATED geometry:** With the present lattice, all active absolute origins have the same phase modulo16. Shift the entire row by a single phase offset0,…,15, and the patch reshape/embedding retains its dimensions and causal relative-token geometry. `P:1170–1206`, `fill_row`, already gathers an arbitrary contiguous logical-valid-bar interval; `M:tokens` reshapes it into16-bar patches; the patch projection input width remains `16×(OHLC+auxiliary channels)`.

**HYPOTHESIS implementation cost:** This adds **zero model FLOPs** at unchanged shapes. Host cost is not necessarily zero: generating/storing phase, boundary admission, gathering different cache lines and adjusting ownership metadata cost work. No additional full-row copy is mathematically necessary. Measure loader build/wait time rather than assuming the previous24.2ms survives unchanged.

**DEMONSTRATED constraint:** The offset must be constant within a row for the existing contiguous patch/RoPE/statistics geometry. Independently jittering each token would require explicit different temporal geometry and causal statistics; it is not the proposed zero-shape-change intervention.

**DEMONSTRATED/HYPOTHESIS distinction:** Phase introduces additional absolute-origin phases, potentially accessing all16 lattices across many presentations. It does **not** create new market shocks, guarantee16× coverage in one run, or guarantee statistically independent examples. With30 randomly phased presentations, even the probability a specified phase appears is only `1−(15/16)^30≈85.6%` in a simplified interior model.

**HYPOTHESIS prediction:** If tokenization-specific fitting is important, phase should produce a later/flatter peak with similar or better peak IC. There is no horizon-specific numerical forecast. The step time should be approximately unchanged in model execution; total step time may change through the loader.

**DEMONSTRATED composition caveat:** Independent per-row phase can reintroduce small overlaps/gaps at the boundaries of a stride5760 M1 tiling. K30+random phase is not automatically exact M1. A constant per-ticker phase preserves the tiling but provides less per-row augmentation. Measure actual support and multiplicity before calling combined interventions orthogonal. Terminal192-bar ownership is also no longer disjoint after independent row jitter; its metadata cannot retain that claim.

## 4. Effective independent sample size: what is and is not implied

**DEMONSTRATED arithmetic:** The dense loss has360 active origins per row, not375 independent examples. At the236-row shape mentioned in the assignment that is84,960 active origins/step; the observed batch256 run has92,160. Do not silently combine the two batch sizes.

**HYPOTHESIS model, not measured ESS:** Under independent equal-variance one-bar return shocks, an h-bar cumulative return sampled every16 bars has lag-k correlation `rho_k=max(0,1−16k/h)`. The finite-window mean-estimator design effect is

`DE_h = 1 + 2 Σ[k=1..359] (1−k/360) rho_k`.

|h|Design effect|Illustrative ESS at236 rows|Illustrative ESS at256 rows|
|---:|---:|---:|---:|
|1|1|84,960|92,160|
|8|1|84,960|92,160|
|16|1|84,960|92,160|
|32|1.9972|42,539|46,144|
|64|3.9861|21,314|23,120|
|128|7.9417|10,698|11,605|
|192|11.8676|7,159|7,766|

**DEMONSTRATED limitation:** These values describe a simplified return-mean estimator, not the ESS of Gaussian-loss gradients, IC, or this market dataset. Cross-ticker factors, serial dependence, sigma/beta normalization and repeated rows can change them substantially. One cannot multiply a30× repetition factor into an arbitrary per-step ESS and call the result measured independence. The original manifest establishes about31.1M distinct fixed-horizon origin outcomes, not31.1M independent observations.

**HYPOTHESIS diagnostic consequence:** Simple overlap predicts increasing long-horizon fragility but no extra iid-shock overlap at h16 relative to h1. The h16 failure and relative h192 resilience remain unresolved by that model.

## 5. What the speedrun records actually establish

### 5.1 Disease classification of the requested mitigations

Each evidence entry below is **DEMONSTRATED as a source statement**. Its transfer to financial IC is **HYPOTHESIS**.

|Mitigation|Record evidence|Disease actually established or claimed|
|---|---|---|
|Attention-window ramp / long-short windows|`S/2025-01-16_Sub3Min/README.md:20–29` grows local/global context at different rates; `S/2026-02-10_ShortWindow/README.md:3–10,80–115` discusses sequence curriculum, late short-window changes and robustness cliffs.|Efficiency/convergence tuning. The latter explicitly describes avoiding late **stalling** via a higher LR floor and cross-node endpoint failures. Not a documented repair of sustained pre-cooldown validation regression.|
|Momentum warmup|`S/2024-11-06_ShortcutsTweaks/README.md:8–14,76–82` lists ablation logs and warms .85→.95 over500steps; `T/README.md:138–166` carries later warmup/cooldown and explicitly notes final momentum cooldown has not even started at the accepted current-record endpoint.|Optimization improvement; no explicit demonstration there of curing divergence or mid-run validation regression.|
|Weight-decay schedule / cautious WD|`S/2025-11-10_CautiousWD/README.md:46–59`: ordinary WD stabilizes but harms performance; CWD lags most of training and catches up in final cooldown; scheduling improves10–15steps.|Stability plus **late catch-up**, not a rising validation curve repaired by cooldown.|
|Logit soft-capping|`S/2024-11-06_ShortcutsTweaks/README.md:85–92`; `S/2025-01-04_SoftCap/README.md:3–10` reduces cap30→15 and cuts1490→1390steps, framed as helpful structure in the small-scale regime.|Faster endpoint convergence/regularization. Not explicitly a divergence or mid-run-regression cure. Positive forecast calibration is not its financial equivalent.|
|Value/embedding scaling|`S/2025-08-23_SparseAttnGate/README.md:6,40–43` reports head scaling caused very large weight norms and failed value-normalization alternatives; `S/2025-12-10_SALambdaOnWeights/README.md:49–62` discusses depth-dependent value scale and an unexplained loss penalty from alternative placement.|Conditioning, norms and endpoint quality. No sustained pre-cooldown validation-regression diagnosis.|
|Skip/value gates|`S/2025-12-29_VeSkipGates/README.md:3–12`: value gates supply most benefit, skip gate marginal, x0 gates unhelpful; scalar freezing worsens loss.|Endpoint efficiency; not demonstrated cure of our disease.|
|QK norm / attention scale|`S/2025-01-16_Sub3Min/README.md:94–109`: QK norm stabilizes coefficients but high entropy constrains late attention; scale tuning saves20steps.|Stability and a late representation/convergence constraint, **not** reported validation collapse. Already adopted in our model.|
|Scalar smoothing / transition handling|`S/2025-12-21_SmoothedScalars/README.md:27–44,76–95`: noisy smear/scalar behavior occurs at simultaneous batch/window/LR changes; clipping and initial LR changes fail to cleanly solve it.|**Parameter instability at a scheduled discontinuity**. Do not relabel it as demonstrated sustained validation regression. Our rise/collapse has no corresponding LR transition.|
|Warmdown / split cooldown|`D/2025-03-06_LongerCooldown/README.md:1–3` increases cooldown40→60%, reducing7050→6950steps. `T/README.md:80–83,135–138` records PowerCool/split schedules; current matrix/aux cooldowns differ.|Endpoint convergence improvement; not an established cure of pre-cooldown financial rank collapse.|

**DEMONSTRATED search result, limited inference:** Broad searches covered README claims in all three actual tracks and relevant nested optimizer-result READMEs, with direct inspection of the named original logs. The word “regression” in the window/QKV record describes a one-second runtime penalty, not validation worsening. Absence of a matching claim in these searches is not proof that every archived run is monotone.

### 5.2 Does fresh-token training held at high LR regress?

**DEMONSTRATED counterexample to the proposed inevitability:** `S/2024-11-04_50Bruns/530f3ee1-8862-4d21-be2b-da10eb05e6a9.txt` extends Muon to fresh50B tokens. Its embedded code sets95,367steps and27,247warmdown steps (`:343–347,408–419`), so LR is flat until68,120.

Validation loss at1,000 /10,000 /20,000 /30,000 /40,000 /50,000 /60,000 /68,000 is:

`3.6715 /3.2672 /3.2012 /3.1700 /3.1523 /3.1378 /3.1287 /3.1226`.

Evidence includes log lines1609,10681,20761,30841,40921,51001,61081,69145. These span far beyond the original4,578-step short budget while keeping the high-LR plateau. The final loss is3.0508 at95,367.

**DEMONSTRATED:** Small pre-cooldown non-monotonicity exists:7,625→7,750 rises3.2992→3.3020, then7,875 falls to3.2964 (`:8287,8413,8539`). This is an isolated .0028 bump, not persistent collapse. Similar small late plateau noise does not justify equating it with the supplied .1-scale IC deterioration.

**DEMONSTRATED:** The same record README explicitly compares50B fresh tokens with five passes over10B and reports broadly similar results (`:3–14`), while warning WSD/zero WD may be undertuned for long duration (`:23–25`). Thus upstream contains deliberate repetition experiments too. It does **not** show that repetition must hurt; its text-task result is a reason to test whether our repeated contexts help, not a license to assume they do.

**DEMONSTRATED:** Current upstream's loader advances through files rather than cycling (`train_gpt.py:1807–1849` explicitly suggests `itertools.cycle` only for multi-epoch use). Its basic next-token stream does not implement our30fold repeated dense-horizon window supervision. Hyperparameters transfer between different supervision regimes, not merely between different domain names.

### 5.3 How much improvement arrives during warmdown?

**DEMONSTRATED arithmetic from logs; not causal attribution:** Define the fraction as `(loss at cooldown start − final loss)/(initial loss − final loss)`. Cooldown starts between evaluations, so use neighboring logged values as brackets where trajectories observed on either side are decreasing. Improvement *during* cooldown includes continued training, and in some records changing evaluation attention windows; it is not the treatment effect of cooldown.

|Record/log|Cooldown boundary|Start-neighbor losses → final|Fraction of total initial-to-final improvement occurring after boundary, bracketed|Fraction of improvement after step1000, bracketed|
|---|---:|---|---:|---:|
|Short old record in `S/2024-11-06_ShortcutsTweaks/d6b50d71-f419-4d26-bb39-a60d55ae7a04.txt`|3270 of4578|step3250 3.4129;3375 3.4020;final3.2762|1.67–1.81%|32.9–35.8%|
|CWD log `S/2025-11-10_CautiousWD/a33fa276-234b-4c9c-9b78-43d85c411e8d.txt`|~1102.5 of2205 scheduled+40extension|step1000 3.5947;1250 3.5259;final3.2813|3.24–4.15%|78.0–100%|
|Medium `D/2025-03-06_LongerCooldown/779c041a-2a37-45d2-a18b-ec0f223c2bb7.txt`|2780 of6950|step2750 3.1805;2875 3.1710;final2.9184|3.19–3.31%|44.3–46.0%|
|Track3 baseline `T/results/7b8270c5-a9cd-4a73-b7d8-5d86a2d1e428.txt:289–295,359–388`|1080 of3600|step1000 3.61194;1125 3.58764;final3.27765|4.11–4.43%|92.7–100%|

**DEMONSTRATED conclusion:** A large share of the *late useful improvement* often arrives during cooldown, but a large share of *total loss reduction from random initialization* does not. The initial10.8258 loss dominates that denominator. Reporting “most improvement is warmdown” without specifying the baseline is misleading.

### 5.4 Explicit cooldown verdict and the earlier5000-budget arm

**DEMONSTRATED:** Our LR is flat through the rise and collapse; source schedule has a60%-duration linear decay to.15 (`C:275–291`; `R:127–135`). A5000-step budget starts cooldown at2000, whereas a2500-step budget starts at1000.

**DEMONSTRATED supplied observation:** The previous5000-budget arm had its best reported NLL1.9941 at1000, before that schedule's cooldown started. Therefore it is evidence against an uncomplicated “just finish the intended warmdown and the peak problem disappears” account. The budget change cannot explain a pre2000 trajectory difference by its future LR change alone; differing arm conditions or stochastic/numerical behavior must be retained as possible explanations rather than invented here.

**HYPOTHESIS retained:** Earlier cooling may improve the attainable floor or arrest harmful movement; it is cheap enough to test if prioritized. **Verdict:** upstream does not establish that our sustained collapse is the expected shape of an uncooled fresh-token run. Do not sell budget2500 as a demonstrated fix. The stride/occupancy interventions now provide the more direct diagnostic evidence.

## Delivery constraints for the parent

**DEMONSTRATED:** New K-stride and phase switches were still being implemented by a sibling during this read-only investigation. I have sent the exact geometry, random-thinning invariance and corrected saturation triples directly to that owner and Main. I do not fabricate a runnable command using unlanded flag names or an unannounced v13 binary path.

**HYPOTHESIS execution specification:** Parent-owned queue jobs should use the announced new binary, `--time-limit 10m`, fixed seed/objective/evaluation draws, explicit `--schedule-budget 9590` for control/K4/K30/phase, and sufficient epochs to reach the specified max-steps. K30 should be evaluated every100steps and continued beyond its first pass. All measurements belong in registered `.report.bin` bases, with separate labels for terminal-owned progress, actual dense-origin exposure, interior occupancy, corpus-wide occupancy and any new in-period population; labels contain no `=` and unmeasured values remainNaN.

**Final mechanical conclusion:** The progress counter never justified the fresh-supervision premise. Dense outcome repetition is proven and quantifiable; whether it sets the peak, whether diverse contexts help, and what supplies the h16–64 selectivity are distinct experimental questions—not conclusions to infer from one occupancy curve.