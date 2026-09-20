# Task-aligned temporal SIGReg

Status: approved research, 2026-09-20. No new regularizer is promoted. Current measured leader and endpoint protocol: [top-run ledger](top_runs.md).

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
