# Temporal SIGReg research

Status, 2026-09-21: clean unanchored temporal JEPA + SIGReg market and memory comparisons are complete. Local SIGReg improves frozen-reader old-cue recovery by **96.5%** in the controlled task, but that gain does not transfer to the market panel. State SIGReg has the best full-state market score, only marginally below persistence. Earlier supervised ablations remain ineligible as tests of the clean objective. Current standings: [top-run ledger](top_runs.md).

The conditional-moment sections below preserve the previous completed experiment. They are not the definition of the new [causal-reader SIGReg](#causal-reader-sigreg) treatment.

## Required unanchored evaluation contract

The user's requirement is no anchoring or interfering downstream gradients. Representation pretraining must therefore optimize only the temporal JEPA prediction objective and its SIGReg regularizer. No price-forecast, reconstruction, trading, or other downstream loss may update the encoder, temporal trunk, or learned representation targets. The shared forecasting loss in the completed four-arm comparison violated this requirement even though it was identical across arms.

Downstream readers are fitted **after pretraining with the complete representation frozen**, using train-only fitting/selection and held-out scoring. They cannot alter the representations being compared. Temporal JEPA and SIGReg gradients themselves are intended learning signals; do not add gradient surgery, or silently detach a JEPA branch to suppress their interaction. Any architectural or gradient-routing assumption must be declared as part of the method before a new comparison.

The earlier market and controlled delayed-cue runs under “Completed causal-reader comparison” used downstream supervision during representation training. Their later frozen probes and paired-input controls do not undo that anchoring. Preserve them as supervised-regularizer evidence, separate from the clean experiments below.

Use fresh representation initialization for the clean comparison; a supervised forecasting checkpoint is not an unanchored starting point. Previously trained supervised readers or representations cannot be reused as the clean pretraining endpoint.

## Clean unanchored temporal JEPA

`--jepa-mode unanchored --reader-norm none --sigreg-placement off|local|state|both --jepa-reconstruction-weight 0` selects the clean path. Fresh identical initialization; direct attached future-observation prediction at patch offsets 1/2/4/8/12, prediction weight 1; no learned target projector. Total SIGReg weight 0.09 on the actual local embedding or causal reader state, split 0.045 per site for both; off has no regularizer. The target and source branches both receive JEPA gradients. Forecast parameters remain allocated for initialization identity but are frozen before optimizer creation, never forwarded or optimized. No reconstruction, downstream supervision, target stop-gradient, or gradient surgery.

Representations use the existing causal prefix normalization and internal transformer normalization; all four remove the final reader RMSNorm. SIGReg populations are valid batch rows at each sampled causal source, never pooled time. Sites share positions, directions and masks. Fixed completed-budget checkpoints are authenticated; the optimizer/graph is destroyed and the complete store frozen before downstream fitting.

### Completed clean delayed-cue comparison

Job **8752 succeeded**: four placements × independently pretrained relevant/null tasks, each 1024 updates, B64, D128 × 2, context512/patch16/pred192, seed20260919, native BF16/CUDA graphs. After freezing, independent training-only samples supply 2048 inner-fit and 512 penalty-selection rows; refit uses all 2560. Three ridge readers (full state, recomputed recent state, local embedding) score the unchanged 256 held-out paired episodes. No validation pair participates in fitting.

| SIGReg placement | Mean paired-effect error ↓ | h64 paired-effect error ↓ | h64 close MSE/persistence ↓ | Mean null paired error ↓ |
| --- | ---: | ---: | ---: | ---: |
| Off: JEPA-only control | 0.033061 | 0.066808 | 0.202598 | 2.22e-9 |
| **Local** | **0.001168** | **0.001990** | **0.182696** | **2.24e-18** |
| State | 0.792244 | 1.638907 | 0.671672 | 3.22e-9 |
| Both | 0.041104 | 0.084085 | 0.218687 | 6.36e-16 |

Mean paired errors cover horizons16/32/64/128/192 and divide squared effect error by pulse amplitude². The h192 true pulse is zero; retain its error but not a direction claim. h64 paired-effect sign is 100% for off/local/both, 98.83% for state. Local's mean error is **96.5% lower** than off; its h64 price score approaches the analytic Bayesian reference **0.182086**. Mean Bayesian regret is off0.017946, local0.000521, state0.219144, both0.017549.

The recent/local readers have exactly zero paired response and mean paired error1.453590 in every relevant task; only full-history state recovers the cue. All paired recent input/statistic differences are exactly zero. All eight full stores retain **zero trainable tensors and bit-identical values across reader fitting/evaluation**, covering 23 tensors / 2,178,572 scalars each. This supports useful old-cue accessibility in this synthetic task, not market profitability or a universal SIGReg conclusion. Measured task runtime sums to19.272s, excluding queue wait.

[Binary reports and immutable memory protocol](../benchmark_results/unanchored-sigreg-memory-20260920/); [completed eight-task receipt](../benchmark_results/unanchored-sigreg-memory-20260920/memory-completion.json).

### Clean market protocol and operational evidence

Four fresh 1400-step arms, B256, D512 × 8, context6000/patch16/pred192, seed20260919, complete corpus and unchanged training schedule. Frozen CUDA ridge readers draw 4096 train-only origins: after chronological purging, 3260 fit rows and 820 penalty-selection rows are merged for refitting; 16 boundary rows are excluded. All arms score 2048 held-out origins at every horizon and recall/delayed-future interval. Primary score is equal-horizon full-state **raw close-return MSE / zero-return persistence MSE** at 16/32/64/128/192. This is not the supervised leaderboard's market-neutral seven-horizon score. Direction excludes zero targets (2032 scored at h64); zero prediction is a miss. Delayed recall is retrospective access, not unseen-future accuracy; ridge coefficients and penalties are independently fitted per output.

| SIGReg placement | Frozen full-state close score ↓ | h64 direction | h64 pooled correlation | Training seconds |
| --- | ---: | ---: | ---: | ---: |
| Off: JEPA-only control | 0.999643 | 49.36% | 0.03365 | 241.08 |
| Local | 0.999893 | 49.36% | -0.01264 | 242.09 |
| **State: full-state cohort leader** | **0.999213** | 49.36% | 0.01555 | 241.62 |
| Both | 1.001351 | 49.11% | 0.01975 | 242.77 |

State improves the aggregate over off by only **0.000430** (0.043 percentage points of persistence MSE), chiefly h32: 0.997488 versus 0.999540. This single-seed development panel does not establish a robust market edge. Direct-input readers score 0.999661 identically across arms; the JEPA-only recomputed-recent reader scores **0.999063**, better than every full-state reader. The synthetic local-SIGReg advantage therefore **does not transfer on this market protocol**. Both improves h16 to 0.996050 but worsens h128/h192 to 1.004812/1.005742. No production model or supervised forecasting replacement is promoted.

Full-state ratios in horizon16/32/64/128/192 order: off `[0.998769,0.999540,1.000283,0.999612,1.000010]`; local `[0.998769,0.999733,1.000283,0.999612,1.001069]`; state `[0.998769,0.997488,1.000119,0.999612,1.000075]`; both `[0.996050,0.999297,1.000853,1.004812,1.005742]`. Ridge selects its strongest penalty, 1e6, for 4/5 off, 3/5 local and 2/5 state horizons; both selects 10 throughout. These strongly regularized linear readers do not establish absence of nonlinear information.

Market recall ratios at lags64/256/1024: off `[1.000983,1.013524,0.999879]`, local `[1.012345,1.001226,1.002574]`, state `[0.998608,0.997089,1.000630]`, both `[1.004798,1.001226,1.000630]`. Delayed-future ratios at leads16/64/128: off `[1.000054,1.000229,0.998845]`, local `[1.000096,1.000098,0.998845]`, state `[0.999909,1.000098,1.002744]`, both `[1.003698,1.000098,0.998845]`. Different interval widths prevent interpreting these as a pure lead-decay curve.

Last 350-update mean local/state population standard deviations: off 0.0140/0.6462, local 1.0519/0.0277, state 0.2450/1.0365, both 1.0191/1.0224. The off predictor's last-source population std is zero; its tiny latent MSE, 0.000571, is not superior prediction. SIGReg changes the selected interface's spread without guaranteeing useful downstream information. Scale alone is not rank or accessible information. Similarly, memory null-task success measures reader rejection of an irrelevant cue, not proof that the representation forgot it.

Reports/checkpoints: [off](../training/runs/unanchored-sigreg1400-20260920-v3-unanchored-none/), [local](../training/runs/unanchored-sigreg1400-20260920-v3-unanchored-local/), [state](../training/runs/unanchored-sigreg1400-20260920-v3-unanchored-state/), [both](../training/runs/unanchored-sigreg1400-20260920-v3-unanchored-both/). [Completed authenticated collection](../benchmark_results/lejepa-campaigns/unanchored-sigreg1400-20260920-v3/complete.json).

GPU job **8743 passed 38 contracts**, including all-placement graph replay, unchanged forecast parameters, attached target/source gradients, and objective/gradient invariance to future forecast-only labels and poisoned forecast-head weights. Release build and cargo check passed. Two independent reviews found no remaining blockers. Initial test failures8711/8720 were test-only stream-owned snapshots and FP64-vs-FP32 decomposition assertions; fixed without changing the representation objective. Their dependent jobs8734–8739 were skipped, not trained. Both market-direction bases and all four memory-reader/freeze bases are registered through the shared TUI registry.

Market jobs8747–8749 failed during first warmup while another process held 26.04GiB; job8750 completed and is preserved outside the final cohort. After that process exited, the unchanged four-arm cohort **8757–8760 and collector8761 all succeeded**, one attempt each. No batch reduction, precision fallback, shorter budget or objective change. Every completed market arm reports captured CUDA graphs and zero routine downstream evaluation time; frozen fitting takes 2.69–4.06s after training. All original failures remain intact.

[Immutable market plan](../benchmark_results/lejepa-campaigns/unanchored-sigreg1400-20260920-v3/plan.json), SHA `444214e066e7e433403b07c5f4eb0f5a900be39ac26028144df73591bc8ef34e`; [execution, failed-run provenance and pinned binaries](../benchmark_results/unanchored-sigreg-20260920-assets/execution.json). Active unresolved question: which unanchored temporal representation objective preserves weak market-relevant information rather than merely improving Gaussian geometry or synthetic cue accessibility?

## What must change

SIGReg's unconditional Gaussian target does not identify forecasting-relevant information. Moving the penalty to another learned projector changes optimization, not this identification problem.

Let X be causal history and Y an actual future market-neutral close return. For any learned target g with E[g]=0 and Cov(g)=I_d, unrestricted squared prediction has optimum f(X)=E[g(Y)|X], with risk d-tr(Cov(E[g(Y)|X])). Thus a whitened target objective rewards predictable variance, including predictable nuisance variance. Two independent Gaussian target channels with history correlations 0.02 (return) and 0.9 (nuisance) already satisfy exact Gaussianity, yet the latent objective prefers the nuisance: predictable variance 0.0004 versus 0.81. Gaussianity cannot resolve that choice.

The law of total covariance is Cov(g(Y))=Cov(E[g(Y)|X])+E[Cov(g(Y)|X)]. Target variability and predictable variability are different objects. A Gaussian scalar target with history correlation rho has optimal prediction variance rho^2 and MSE 1-rho^2. Requiring unit prediction variance instead gives MSE 2-2rho for positive rho. At rho=0.02 these are 0.9996 and 1.96. This is a design counterexample, not a claim that the current code explicitly whitens predictions.

Residual Gaussianity alone also fails. If X and epsilon are independent standard normals and Y=rho X+sqrt(1-rho^2) epsilon, an ignorant forecast's residual Y is exactly standard normal. Its marginal SIGReg is perfect despite missed signal. However E[X exp(itY)]=i rho t exp(-t^2/2), whereas the correct standardized innovation epsilon has zero history witness. Deterministic 48-node Gaussian quadrature checked this identity at t=0.25,1,2 to 1e-12.

## Population objective

The primary decision score is mean squared close error, so its population optimum is mu*(X)=E[Y|X], equivalently E[Y-mu(X)|X]=0. This condition permits zero signal variance, heavy tails, heteroskedasticity and correlated innovations at overlapping horizons. It does not require Gaussian returns, white market paths, or an arbitrary latent rank.

Use bounded, fixed causal instruments phi(X) and residual e_h=(Y_h-mu_h(X))/sqrt(h), where Y and mu already use the model's source-causal sigma units. Define

R_h(mu)=||E[e_h phi(X)]||^2.

With a sufficiently rich instrument family, zero conditional moments identify the conditional mean. A finite family only detects its tested functions; failure to detect signal is not evidence that none exists. This replaces unconditional embedding Gaussianity with task-aligned conditional moment restrictions, using the characteristic-function/kernel toolbox. It is not the original LeJEPA theorem or a guarantee of better finite-sample learning.

At population level the Bayes conditional mean minimizes both squared loss and R. In a misspecified, finite-capacity model, adding R reweights approximation errors and may worsen the decision score. It introduces no new labels or information. Weight zero remains a legitimate optimum.

For a distributional objective the stronger analogue is history-conditioned calibration of the predicted conditional CDF: F_theta(Y|X) should be uniform conditional on X, or its Gaussianized transform standard normal conditional on X. Merely applying ordinary SIGReg to residuals is insufficient. That is not this experiment: current acceptance is close-mean accuracy, so we do not add a distribution head or assume Gaussian innovations.

## Concrete implementation contract

- Reuse the actual dense forecast's close coordinate and fixed observed future labels, never a learned target encoder or a second predictor. The close decoder is coordinate channel 0 times sqrt(h).
- Evaluate the predefined decision horizons 1,8,16,32,64,128,192. Normalize residuals by sqrt(h); do not estimate a denominator from validation or divide by learned predictive variance. This preserves each horizon's conditional-mean optimum but is not numerically identical to the final panel's empirical persistence weighting.
- Add `--temporal-moment-weight` and `--decision-mse-weight`, both default zero, omitted from default serialized model contracts. No parameter additions or changes to baseline initialization. Initially require JEPA off, cumulative target basis, non-increment means, causal future-calendar=false for enabled treatment; reject unsupported combinations explicitly.
- Causal instrument summaries at observed-bar lags 1,8,16,32,64,128,512,2048: tanh of source-beta-neutral return divided by source sigma sqrt(lag); tanh of market return in the same units; tanh of half log realized mean-square close-step return relative to source sigma squared; and a complete-lookback availability indicator. Each unavailable summary is zero, with its indicator zero. Every calculation stops at the source. Long lookbacks unavailable at early dense origins do not discard those origins.
- Thirty-two bounded summaries feed a fixed instrument vector: a constant (energy 1/3), normalized linear summaries (maximum energy 1/3), and 64 sine/cosine random Fourier pairs (energy 1/3). Local deterministic Gaussian frequency generation with seed 20260920, 16 projections at each bandwidth 0.5,1,2,4, scaled by 1/sqrt(32). Total width 161, norm squared at most one. No learned instrument, label dependence, global Torch RNG consumption, or validation fit.
- Allocate indices/frequencies once on the model's device. Use native BF16 model computation and FP32 objective reductions. No gradient accumulation, chunking, additional backbone pass or dense four-channel target copy.
- For each dense source index and horizon, reduce across batch rows only; never count flattened overlapping time points as independent populations. Let a_b=mask_b e_b phi_b, N=sum(mask). Compute U=(||sum_b a_b||^2-sum_b||a_b||^2)/(N(N-1)). Use batch matrix multiplication, not a B x origins x horizons x instruments product. Average eligible sources within each horizon, then give nonempty decision horizons equal weight. Fewer than two valid rows contribute no moment population; different eligible-source counts must not silently reweight horizons.
- Remove the diagonal term. A squared batch mean includes residual-noise energy divided by batch size and is not the desired independent-pair moment objective. U may be negative in a finite sample; do not clamp it to zero or multiply by a nominal IID batch size. Sampling without replacement and shared market outcomes still limit population inference; diagonal removal alone does not establish independent financial observations.
- The auxiliary uses undecimated fixed labels and original target validity. Match the existing observable close-score validity contract, not a secretly changed target definition. No future value enters instruments.
- Loss is existing forecasting objective + temporal_moment_weight * mean_h U_h + decision_mse_weight * mean_h masked_mean(e_h^2). The direct-MSE control uses the identical residual/mask geometry.
- Expose shared resident geometry, instrument construction and selected forecast/target extraction for frozen diagnostics. Training and evaluation must not implement competing definitions.

## Diagnostics and evidence

All inspected metrics must use `.report.bin`, registered through the shared writer-base registry consumed by the TUI. JSON contains configuration, identity and chronology only, never a parallel metric channel.

Training reports, by decision horizon: diagonal-free U, squared batch-mean V, removed diagonal contribution, normalized close MSE, mean residual, valid rows/pairs. Distinct chart bases separate moment energy, forecast MSE, signed bias, row counts, pair counts and applied weights. Report true forecast NLL separately from total objective, and CUDA graph capture/runtime through the existing research reports.

Frozen witness evaluation authenticates checkpoint, full corpus and fixed research panels. Fit a ridge correction to the existing model's close residual, with causal instruments and, separately, the model's frozen contextual state as feature families. Split the training-only fit origins chronologically with target purging; select ridge on the later training-only split; refit on eligible training fit origins; score once on the fixed validation panel. No validation fitting or calibrated-checkpoint mutation. Use actual final decision origins where chronology and complete target availability permit; record any explicitly excluded fit rows and exact source geometry.

Report per-horizon original and corrected close MSE/persistence, signed direction accuracy, residual predictable component, chosen ridge and counts; include chronological validation subpanels to distinguish stable improvement from one pooled regime. A correction measures decodable missed signal, not a deployable trading gain, not full information absence, and not a license to fit on validation. The separately fitted state witness can expose a limitation of the fixed instrument family.

## Predeclared matched experiment

Reuse authenticated full/none forecasting-only and decoupled/lattice endpoints. All new training uses full/none, the same complete corpus, 1400 updates and schedule, batch256, seed20260919, native BF16/CUDA graphs, D512 x8, seq6000/patch16/pred192, common validation2048 and synchronized cross-section4000. No terminal-test access, multiple seeds, reduced training or latent-loss selection.

First treatments:

1. Direct decision-MSE weight 0.125, moment weight 0.
2. Moment weight 0.125, direct decision-MSE weight 0.
3. Moment weight 0.5, direct decision-MSE weight 0.
4. Moment weight 0.125 plus direct decision-MSE weight 0.125.

These weights deliberately emphasize the seven decision horizons; they are not equal-weight NLL coefficients. The direct term averages seven close errors, whereas the forecast NLL averages four coordinates across 192 horizons. Near unit normalized precision, weight 0.125 therefore adds approximately 27.4 times the original quadratic weight at each selected close coordinate. Learned precision changes that ratio. The moment penalty has a different kernel-dependent gradient scale, so equal numeric weights do not imply equal gradient strength. The fourfold moment treatment tests strength without a broad post-hoc sweep. The direct-MSE and combined arms separate conditional structure from simply weighting the decision horizons more heavily.

Primary acceptance: lower equal-horizon average market-neutral close MSE/persistence on the unchanged matched panel. Also inspect every horizon, direction, cross-sectional IC, NLL, OHLC errors and runtime. A lower moment loss or better fitted witness alone does not qualify. One development seed provides no significance or global-optimality claim.

Frozen diagnostics run on the current leader before interpreting treatments. All compute-intensive work goes through mlq, normal priority, one concurrent managed job, maximum one attempt, immutable per-model execution receipts. Follow-up work must answer a concrete failure mode supported by these results, not expand a blind sweep. Update the short ledger after each completed model comparison; preserve failures and nonpromoted evidence.

## First frozen diagnostic

Job 8570 succeeded on the authenticated full/none leader. Train-only ridge selection and refitting followed by unchanged validation produced close scores **0.977356** (fixed instruments) and **0.977343** (frozen state), versus **0.975143** uncorrected. Later-training holdout correction gains are positive at horizons 8–192, but do not transfer to the validation panel. Neither correction is promoted. Uncentered second-moment participation ranks are **1.55 / 161** and **2.10 / 512**: concentrated kernel energy, not proof of centered representation collapse. These diagnostics do not establish that a stronger moment penalty will improve accuracy.

[Binary reports and immutable witness provenance](../benchmark_results/temporal-moment-witness-20260920-full-none/). All 46 emitted witness bases match the shared report registry. GPU job 8569 passed 20 contracts, including real CUDA graph replay, retained-report lifetime, causal instruments, masked row-pair moments and frozen ridge refitting.

## Completed matched experiment

All four full-data, fixed-1400 treatments completed. Final comparison job **8599** re-evaluated 18 authenticated checkpoints (16 matched and 2 explicitly historical) plus persistence on the unchanged panels. Scores below are the predeclared seven-horizon mean; lower is better.

| Full/none objective | Original close score | Fixed-witness correction | Frozen-state correction | h64 direction | h64 signed IC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forecasting only | **0.975143** | 0.977356 | 0.977343 | 52.39% | 0.03563 |
| Direct decision MSE 0.125 | 0.976482 | 0.978659 | 0.980447 | 52.83% | 0.03435 |
| Conditional moments 0.125 | 0.978702 | 0.981034 | 0.982452 | 52.29% | 0.03956 |
| Conditional moments 0.5 | 0.976320 | 0.978456 | 0.978477 | 52.44% | 0.03870 |
| Moments + decision MSE 0.125 each | 0.979404 | 0.981746 | 0.983311 | 50.88% | 0.05494 |

**Retain both auxiliary weights at zero.** Direct MSE improves h1 from 0.908646 to 0.894355 but worsens the aggregate. Moment weight 0.5 improves h8/h128/h192 to 0.967656/0.994510/0.998428, from 0.968513/0.996583/1.000512, while losing elsewhere. The combined objective raises h64 IC but worsens close error and direction. These are horizon/metric tradeoffs, not a better overall predictor.

Neither frozen correction transfers for any of the five models. This rejects promotion of these fitted corrections, not all residual predictability or every possible instrument family. At the final training report, moment-only weight 0.125 has horizon-mean U ≈3.34e-5, V ≈2.71e-3 and removed diagonal ≈2.69e-3. Most squared batch-mean energy is self-noise; removing it matters. Larger numeric weight does not by itself establish stronger useful conditioning, and reduced in-sample moment energy would not establish transfer.

Training jobs **8574–8577** succeeded in **263.60/252.23/252.04/247.32s**. Witness jobs **8570, 8584–8587** succeeded, each with all 46 literal chart bases registered. Collector **8578** failed only because its pinned copy resolved a certificate's repository-relative source plan against the campaign directory. Accuracy job **8583** consequently never ran. The corrected validator takes an explicit authenticated repository; `revalidate-collection` rechecks the immutable plan, validator hash, original failed receipt hash, source certificates and every endpoint before creating a separate recovery receipt and completion artifact. Relocated-driver execution and refusal of stale hashes/failed training receipts were exercised. Original failure receipts were preserved; no training was repeated.

Evidence: [paired price reports](../benchmark_results/accuracy-temporal-moments-20260920/), [completed collection](../benchmark_results/lejepa-campaigns/temporal-moments1400-20260920/complete.json), [revalidation receipt](../benchmark_results/lejepa-campaigns/temporal-moments1400-20260920/arms/collect-revalidated.json), [commands, pinned assets and queue provenance](../benchmark_results/temporal-moments-20260920-assets/execution.json). Frozen witness reports: [baseline](../benchmark_results/temporal-moment-witness-20260920-full-none/), [decision MSE](../benchmark_results/temporal-moment-witness-20260920-decision-mse/), [moments](../benchmark_results/temporal-moment-witness-20260920-moment/), [strong moments](../benchmark_results/temporal-moment-witness-20260920-moment-strong/), [combined](../benchmark_results/temporal-moment-witness-20260920-moment-plus-mse/).

## Causal-reader SIGReg

**Historical supervised ablation, ineligible for the required unanchored test.** This section preserves the actual implementation, objective, measurements and immutable receipts; it is not the corrected pretraining contract.

The new treatment retains ordinary SIGReg but regularizes the actual temporal representation consumed by the forecast head. `--reader-norm none` exposes the pre-final-RMS contextual state to both that head and the regularizer; internal transformer normalization is unchanged. A fixed-radius final RMS output cannot exactly follow a full-dimensional Gaussian. All four fresh arms therefore use the same unconstrained reader, without new parameters or an auxiliary latent predictor.

| Fresh arm | `--sigreg-placement` | Local weight | Reader-state weight |
| --- | --- | ---: | ---: |
| `reader-none` | `off` | 0 | 0 |
| `reader-local` | `local` | 0.09 | 0 |
| `reader-state` | `state` | 0 | 0.09 |
| `reader-both` | `both` | 0.045 | 0.045 |

The semantic objective is the unchanged forecasting NLL: JEPA off, full scale coupling, no horizon decimation, no conditional-moment or direct-MSE auxiliary, no future calendar. Local means the unconstrained patch observation, which already includes causal prefix normalization. State means the actual causal forecast-reader tensor, not a separate projector. Every arm has the same parameter initialization and training data stream. Both sites share the same sampled positions, projection directions and source-valid mask.

The regularizer reuses the existing N-scaled Epps–Pulley statistic: 256 independent-host-RNG projection directions, 17 quadrature knots, and eight sampled causal source views from minimum-history eligibility through the final context patch. Each view reduces across valid batch rows; time is never pooled as independent data. Complete source patches and actual valid-history counts determine eligibility; future labels do not enter the regularizer mask. Views with fewer than two rows contribute no statistic. Model computation remains BF16 with FP32 regularizer reductions and resident CUDA-graph operands; there is no accumulation or chunking.

The placement comparison includes SIGReg's mean, scale and distribution-shape effects; it does not isolate higher-order Gaussianity. Local pre-RMS radial rescaling is largely invisible to the normalized trunk, unlike rescaling the actual reader state. The original final-RMS full/none leader is additionally scored as a separate normalization-controlled reference. Restoring RMS only downstream of the state penalty would no longer regularize the actual reader interface. None of these losses require temporally white states, positive innovation variance, or a jointly Gaussian trajectory.

### Matched evaluation

Market protocol remains full corpus, 1400 updates and matching schedule, batch 256, seed 20260919, D512 × 8, context 6000/patch 16/prediction 192, validation 2048 and synchronized cross-section 4000. Primary score remains equal-horizon market-neutral close MSE/persistence at 1,8,16,32,64,128,192. All four reader arms are fresh; an inherited normalized-reader endpoint cannot substitute for `reader-none`.

Frozen CUDA ridge probes use the unchanged authenticated common source and train-only chronological penalty selection/refitting. Existing future-return probes remain. Ten additional registered binary-report bases cover error, persistence-relative error, correlation, availability/scored counts, and selected penalty for:

- Past recall: 16-bar close returns ending 64, 256 or 1024 observed bars before the source.
- Delayed future: interval returns at (lead, width) = (16,16), (64,32), (128,64), rather than cumulative source-to-end returns.

Direct input, local observation, genuinely recomputed recent state and full causal state are compared. Missing bounds/interval bars are masked; unavailable fits are NaN, not invented zero errors. Label reach governs purging. Better ridge performance measures accessible information, not proof that more information is stored. Varying delayed-future widths also mean that raw-error slope is not lead-only degradation.

The existing paired delayed-cue benchmark additionally supports `benchmark-jepa-memory --reader-sigreg`: four placements × relevant-cue/irrelevant-cue-null tasks, 1024 updates per task, batch 64, seed 20260919, 256 independent validation pairs, unchanged D128 × 2 controlled model and generator. Reader normalization and full-coupling forecasting are shared across those four arms. Recent observations, normalization inputs and future innovations are identical within intervention pairs; Bayesian means are known. This tests causal use of an old cue, not market performance. All outcomes use the existing registered memory `.report.bin` reports; `memory-protocol.json` contains only configuration and provenance.

### Completed causal-reader comparison

All four fresh arms completed the predeclared market protocol. Paired accuracy job **8646** scored six checkpoints (the four reader arms, normalized-reader full/none reference and decoupled/lattice reference) plus persistence, producing 25 registered binary-report bases.

| Reader arm | Close score ↓ | h64 direction | h64 signed IC | Training wall seconds |
| --- | ---: | ---: | ---: | ---: |
| None | **0.977267** | 52.15% | 0.04719 | 256.07 |
| Local SIGReg | 0.984570 | 52.00% | 0.02250 | 251.17 |
| State SIGReg | 0.991508 | 52.39% | 0.00690 | 259.49 |
| Both SIGReg | 0.990054 | 51.66% | 0.02231 | 259.14 |
| Normalized-reader full/none reference | **0.975143** | 52.39% | 0.03563 | previously completed |

The no-SIGReg reader wins this placement cohort; the original normalized-reader model remains better overall. There are isolated horizon gains: state beats reader-none at h64/h128, and both at h32/h64/h128, while both loses substantially at h1. These do not meet the aggregate criterion. The state-only raw SIGReg statistic falls from **9.2625** in the first 350-update interval to **1.6180** in the last, while held-out prediction worsens. Improved regularizer fit is not improved learning. Numeric weight alone also does not equalize optimization pressure: the first-interval weighted local contribution is **14.6971**, versus **0.8336** for state, despite both using 0.09; these are loss magnitudes, not gradient norms.

All five frozen evaluation panels (four reader arms and normalized reference) emitted all ten new delayed bases, with 2048 scored validation rows at every declared delay. Full-state delayed-future ratios stay at or above the zero-return baseline; no consistent gain appears. Past-recall ratio at lag64 is **0.879981** for reader-none versus **1.005053 / 1.000523 / 1.005806** for local/state/both. At lag1024 they are **1.000985 / 0.999935 / 1.005174 / 0.999935**. These describe linear accessibility, not proof of absent information.

The independent controlled-memory comparison completed all **eight 1024-update tasks** in **17.05s** summed task runtime, excluding queue delay. Every paired recent-input/statistic equality diagnostic is exactly zero at every evaluation. All four models learn the correct relevant-cue effect direction at h64, but SIGReg worsens effect precision:

| Reader arm | h64 paired-effect error ↓ | Mean paired-effect error over 16/32/64/128/192 ↓ | Mean irrelevant-cue false-effect error ↓ |
| --- | ---: | ---: | ---: |
| None | **0.004412** | **0.004825** | 5.23e-10 |
| Local SIGReg | 0.010042 | 0.009954 | 1.27e-9 |
| State SIGReg | 0.078722 | 0.087689 | 2.17e-6 |
| Both SIGReg | 0.060775 | 0.073240 | 3.04e-6 |

Errors are squared paired-effect errors divided by pulse-amplitude squared. The h192 true pulse is zero; its error remains in the displayed five-horizon mean, while effect-sign accuracy is undefined there. This is worse precision, not complete memory collapse, and the controlled task is not market evidence.

**Forecasting decision only: retain `reader_norm=rms`, `sigreg_placement=off` for the measured supervised forecaster.** The tested additions worsened its market score and supervised old-cue precision. This does not establish how unanchored temporal SIGReg learns representations: encoder/trunk updates included forecast NLL in every arm. The required next comparison separates representation-only JEPA + SIGReg pretraining from frozen downstream evaluation. Neither the standalone target-gradient diagnostic nor the post-training frozen probes can remove the anchoring already present in these runs.

### Verification and provenance

GPU contract job **8631 succeeded: 30 passed, zero failed or ignored**. Scoped formatting, release builds and `cargo check` for the trainer, shared reports, report CLI and TUI passed. Independent review found one diagnostic-provenance bug: `diagnose-temporal-sigreg` now records actual total/per-site reader SIGReg weights instead of incorrectly inferring zero from JEPA being off. Its standalone target-gradient diagnostic remains explicitly standalone.

[Immutable four-arm plan](../benchmark_results/lejepa-campaigns/reader-sigreg1400-20260920/plan.json), SHA `948821c4b5f65648dd0d56395d8a0b8aae5ea2b057f03ad16151aa0f62fe0fa1`. [Execution commands, pinned assets and completed queue receipts](../benchmark_results/reader-sigreg-20260920-assets/execution.json). Jobs **8631, 8640–8644, 8646–8649 all succeeded**, one attempt each. Collection authenticated 18 inherited/new endpoints; historical failures were not overwritten. The real reader-both diagnostic CLI verified total training weight **0.09**, split **0.045 / 0.045**, with no latent-target penalty. All jobs used exclusive, normal-priority admission; market watchdog 900s plus 30s queue grace, memory watchdog 15m.

Evidence: [paired market accuracy](../benchmark_results/accuracy-reader-sigreg-20260920/), [completed collection](../benchmark_results/lejepa-campaigns/reader-sigreg1400-20260920/complete.json), [normalized-reference frozen probes](../benchmark_results/reader-sigreg-probes-20260920-rms-reference/), [reader-both diagnostic](../benchmark_results/reader-sigreg-diagnostic-20260920-both/), [controlled delayed-cue reports and protocol](../benchmark_results/reader-sigreg-memory-20260920/). Reader runs and their frozen probes: [none](../training/runs/reader-sigreg1400-20260920-reader-none/), [local](../training/runs/reader-sigreg1400-20260920-reader-local/), [state](../training/runs/reader-sigreg1400-20260920-reader-state/), [both](../training/runs/reader-sigreg1400-20260920-reader-both/).
