,o# Top-run ledger

Updated: 2026-09-21. Replace current standings after every completed matched comparison; keep only decision-changing history. Accuracy means observable future-price/return accuracy, never latent JEPA loss. No production promotion or terminal-test claim from these development panels.

## Current standings

Protocol: `lejepa-fixed1400-20260919`; 1400 updates, batch 256, seed 20260919, full training pool, causal inputs, D512 × 8 layers. Identical 2048 validation origins and 4000 synchronized cross-sectional origins. Close score = equal average of **market-neutral close MSE/persistence** at predefined observed-bar horizons **1, 8, 16, 32, 64, 128, 192**; lower is better, 1 = persistence. Hit rates exclude zero forecasts/targets. One seed; no significance claim.

| Role / model | Close score ↓ | h64 direction ↑ | h64 signed IC ↑ | Run under `training/runs/` |
| --- | ---: | ---: | ---: | --- |
| **Matched aggregate leader: full coupling/no decimation** | **0.975143** | **52.39%** | **0.03563** | `lejepa-full-none1400-20260920-full-none-forecast` |
| Required baseline: decoupled+lattice | 0.978749 | 51.90% | 0.01723 | `lejepa-fixed1400-20260919-forecast` |
| Best learned-target JEPA: anchored+reconstruction | 0.982099 | 52.05% | 0.03309 | `lejepa-fixed1400-20260919-reconstruct` |
| Best JEPA h192 close score: anchored without SIGReg | 0.987293 | 52.20% | 0.03318 | `lejepa-fixed1400-20260919-no-sigreg` |
| Negative control: anchored with SIGReg | 1.000185 | 50.93% | -0.01461 | `lejepa-fixed1400-20260919-anchored` |
| Projected SIGReg placement control | 0.984327 | 52.10% | 0.03967 | `sigreg-projection1400-20260920-projected` |
| Same projector without SIGReg | 0.985620 | 49.51% | 0.03507 | `sigreg-projection1400-20260920-projected-no-sigreg` |
| Conditional return CF, decoupled/lattice | 0.980716 | 50.54% | 0.04137 | `temporal-conditional1400-20260920-conditional` |
| Conditional return CF, full/none | 0.980497 | 51.71% | 0.02468 | `temporal-conditional1400-20260920-conditional-full-none` |
| Reference-sized 16D projected SIGReg | 0.994632 | 50.49% | 0.01979 | `sigreg-small1400-20260920-projected-small` |
| Direct decision-MSE reweighting, 0.125 | 0.976482 | 52.83% | 0.03435 | `temporal-moments1400-20260920-decision-mse` |
| Conditional mean moments, 0.125 | 0.978702 | 52.29% | 0.03956 | `temporal-moments1400-20260920-moment` |
| Conditional mean moments, 0.5 | 0.976320 | 52.44% | 0.03870 | `temporal-moments1400-20260920-moment-strong` |
| Moments + decision MSE, 0.125 each | 0.979404 | 50.88% | 0.05494 | `temporal-moments1400-20260920-moment-plus-mse` |

At h192, no-SIGReg JEPA = **0.997561**, full/none = 1.000512, decoupled/lattice = 1.001884. A horizon-specific win is not a global win. Latent-only one/multi-horizon close scores ≈0.99935/0.99998: not competitive.

### Temporal projected-target matched comparison

Same fixed-1400 full-corpus protocol above; all arms used `future-calendar=false`,
full/none forecast geometry, cumulative targets and the anchored temporal projected target
\(q_{t+k}-q_t\). Lower close score remains primary.

| Temporal treatment | Close score ↓ | h64 direction ↑ | h64 signed IC ↑ | Run / reports |
| --- | ---: | ---: | ---: | --- |
| Temporal projected + per-offset SIGReg | 0.988730 | 50.68% | 0.05031 | [`temporal-projected`](../training/runs/temporal-projected1400-20260921-v2-temporal-projected/) |
| Temporal projected, no SIGReg | 0.986919 | 52.44% | 0.06731 | [`temporal-projected-no-sigreg`](../training/runs/temporal-projected1400-20260921-v2-temporal-projected-no-sigreg/) |
| Temporal projected + sign logistic, weight 0.05 | 0.998657 | 51.07% | 0.08190 | [`temporal-projected-sign`](../training/runs/temporal-projected-sign005-20260921-temporal-projected-sign/) |

The sign arm's h64 IC is highest in this small cohort, but its aggregate close score and
direction are worse than the no-SIGReg temporal control; this is not promotion evidence.
Temporal delta targets are not competitive with the 0.975143 matched leader; adding SIGReg worsens
this cohort's aggregate and h64 direction. The follow-up sign-aligned treatment is documented in
[temporal-projected-sign005](../benchmark_results/lejepa-campaigns/temporal-projected-sign005-20260921/plan.json).
The temporal and sign campaigns are authenticated in their respective
[complete.json](../benchmark_results/lejepa-campaigns/temporal-projected1400-20260921-v2/complete.json)
and [sign completion](../benchmark_results/lejepa-campaigns/temporal-projected-sign005-20260921/complete.json).

### Supervised reader-SIGReg ablation — not an unanchored representation test

Same 1400-step protocol and panels, but **all four fresh arms remove the final reader RMSNorm and receive forecasting gradients in their encoder/trunk**. Total SIGReg weight 0.09, split equally for both sites. The normalized-reader leader above is a separate normalization reference. These results evaluate adding SIGReg to supervised forecasting; they are ineligible as evidence about the user's required unanchored temporal representation learner.

| Reader placement | Close score ↓ | h64 direction | h64 signed IC | Run under `training/runs/` |
| --- | ---: | ---: | ---: | --- |
| **None: cohort leader** | **0.977267** | **52.15%** | **0.04719** | `reader-sigreg1400-20260920-reader-none` |
| Local | 0.984570 | 52.00% | 0.02250 | `reader-sigreg1400-20260920-reader-local` |
| State | 0.991508 | 52.39% | 0.00690 | `reader-sigreg1400-20260920-reader-state` |
| Both | 0.990054 | 51.66% | 0.02231 | `reader-sigreg1400-20260920-reader-both` |

No addition is promoted for this supervised forecaster. Supervised delayed-cue mean paired-effect error also worsens: none **0.004825**, local **0.009954**, state **0.087689**, both **0.073240**; all eight relevant/null tasks completed 1024 updates with exact paired input equality. Those encoders also received forecast gradients, so this is not a clean temporal SIGReg memory test.

### Clean unanchored memory comparison

Fresh attached temporal JEPA + SIGReg only; no forecasting/reconstruction gradients or gradient surgery. Eight relevant/null tasks each complete1024 updates, B64, D128×2, seed20260919, BF16/CUDA graphs. Freeze every parameter, then independently fit train-only ridge readers2048/512, refit2560, score256 held-out paired episodes.

Full-state mean paired-effect error at16/32/64/128/192: **local0.001168**, off0.033061, both0.041104, state0.792244. Local improves96.5% over the JEPA-only control; h64 close MSE/persistence **0.182696**, versus off0.202598 and analytic Bayes0.182086. Recent/local readers have zero paired response; every full store stays bit-identical throughout reader fitting/evaluation. This is controlled-memory evidence, not market accuracy. [Reports/protocol](../benchmark_results/unanchored-sigreg-memory-20260920/), job8752 succeeded; [interpretation](temporal_sigreg.md#completed-clean-delayed-cue-comparison).

### Clean unanchored market comparison — separate frozen-reader protocol

Four fresh 1400-step JEPA-only ±SIGReg arms, B256, D512×8, seed20260919; no downstream pretraining loss. Freeze, fit/select ridge on 4096 training origins (3260 inner + 820 selection after 16-row purge), refit, score 2048 validation origins. Score = mean **raw close-return MSE/persistence at 16/32/64/128/192**, not the supervised seven-horizon market-neutral score above.

| Placement | Frozen full-state score ↓ | h64 direction | h64 pooled correlation | Run under `training/runs/` |
| --- | ---: | ---: | ---: | --- |
| Off: JEPA-only | 0.999643 | 49.36% | 0.03365 | `unanchored-sigreg1400-20260920-v3-unanchored-none` |
| Local | 0.999893 | 49.36% | -0.01264 | `unanchored-sigreg1400-20260920-v3-unanchored-local` |
| **State: full-state cohort leader** | **0.999213** | 49.36% | 0.01555 | `unanchored-sigreg1400-20260920-v3-unanchored-state` |
| Both | 1.001351 | 49.11% | 0.01975 | `unanchored-sigreg1400-20260920-v3-unanchored-both` |

State's advantage over off is only 0.000430; all full-state results remain near persistence. The off arm's recomputed-recent reader scores 0.999063, better than every full-state reader. The large synthetic local-SIGReg gain does not transfer here; no production promotion. Direction has 2032 nonzero targets at h64; zero predictions are misses. [Completed collection](../benchmark_results/lejepa-campaigns/unanchored-sigreg1400-20260920-v3/complete.json), jobs8757–8761 succeeded; [per-horizon reports and interpretation](temporal_sigreg.md#clean-market-protocol-and-operational-evidence).

### Clean unanchored temporal-delta comparison

Fresh representation-only attached projected JEPA, no forecast/reconstruction gradients,
reader normalization `none`, 1400 updates, B256, D512×8, seed20260919. Frozen full-state
ridge readers use the same 4096 train-only origins and 2048 validation origins as the clean
unanchored protocol; score is raw close-return MSE/persistence at 16/32/64/128/192.

| Objective | Frozen full-state score ↓ | h64 direction | h64 pooled correlation | Run |
| --- | ---: | ---: | ---: | --- |
| **Temporal projected + per-offset target SIGReg** | **0.999253** | 49.36% | -0.00618 | [`unanchored-temporal`](../training/runs/unanchored-temporal1400-20260921-v2-unanchored-temporal/) |
| Temporal projected, no target SIGReg | 0.999701 | 49.36% | 0.01801 | [`unanchored-temporal-no-sigreg`](../training/runs/unanchored-temporal1400-20260921-v2-unanchored-temporal-no-sigreg/) |

Target SIGReg improves the temporal control by 0.000448, but remains worse than the
existing unanchored state-SIGReg cohort leader at 0.999213. No promotion. The first
campaign attempt failed during post-run validation because of a driver set-union bug and is
preserved at [`v1`](../benchmark_results/lejepa-campaigns/unanchored-temporal1400-20260921-v1/);
the authenticated successful comparison is [`v2`](../benchmark_results/lejepa-campaigns/unanchored-temporal1400-20260921-v2/).

### Clean unanchored temporal target + reader-state comparison

Fresh matched representation-only runs under the same 1400-update B256/D512×8
protocol, seed20260919, frozen readers, and raw close-return full-state score.
The combined arm splits the fixed aggregate SIGReg budget: target `.045` plus
reader-state `.045`.

| Objective | Frozen full-state score ↓ | h64 direction | h64 pooled correlation | Run |
| --- | ---: | ---: | ---: | --- |
| Temporal projected + target SIGReg `.09` | 0.999754 | 49.36% | 0.01481 | [`target`](../training/runs/unanchored-temporal-combined1400-20260921-v3-unanchored-temporal/) |
| Temporal projected, no SIGReg | **0.999703** | 48.97% | -0.00441 | [`control`](../training/runs/unanchored-temporal-combined1400-20260921-v3-unanchored-temporal-no-sigreg/) |
| Temporal projected + target `.045` + reader-state `.045` | 1.000006 | **49.61%** | 0.00722 | [`combined`](../training/runs/unanchored-temporal-combined1400-20260921-v3-unanchored-temporal-state/) |

The combined arm does not beat the existing clean state-SIGReg leader
`0.999213`, or even its fresh temporal no-SIGReg control. No promotion. The
first corrected campaign attempt preserved two post-training validation failures
caused by runner diagnostic routing; the authenticated v3 comparison is the
[completed collection](../benchmark_results/lejepa-campaigns/unanchored-temporal-combined1400-20260921-v3/complete.json).

## Historical references — not matched treatments

Same scoring observations, but **2500 updates, different seed, frozen calibrated means, future-calendar inputs**. Do not attribute differences solely to model design.
| Model | Close score ↓ | h64 close score ↓ | h64 direction ↑ | h64 signed IC ↑ |
| --- | ---: | ---: | ---: | ---: |
| `timexer-decoupled-lattice-2500` | 0.978768 | 0.985270 | 51.76% | 0.04870 |
| `timexer-full-none-control-2500` | 0.995667 | 0.995200 | 51.03% | 0.05353 |

## Evidence and operational state

- [Latest paired accuracy reports](../benchmark_results/accuracy-reader-sigreg-20260920/): **six checkpoints plus persistence**, 25 registered binary-report bases; job **8646 succeeded**. [Completed reader comparison and interpretation limits](temporal_sigreg.md#completed-causal-reader-comparison). [Earlier 18-checkpoint comparison](../benchmark_results/accuracy-temporal-moments-20260920/) retains the prior matched and historical evidence.
- Evaluation job **8480 succeeded**, nine models in **26.845s**. Original six models: job **8444 succeeded**, each 233–268s. Missing full/none trained in **237.02s**.
- Five new fixed-1400 runs completed: projected ±SIGReg **276.06/280.62s**, conditional CF decoupled/full **243.97/242.36s**, 16D projected **247.11s**. Training jobs **8494/8495/8512/8513/8521**; all old endpoints reused.
- Full/none job **8476** failed only in post-training validation of omitted Serde defaults. Validator corrected; saved endpoint independently authenticated and scored without retraining. Original failed receipt preserved; [reverification provenance](../benchmark_results/accuracy-decoupled-comparison-20260920-assets/control-endpoint-reverification.json).
- Campaigns use one exclusive normal-priority queue job per model and a fixed update budget; reader arms use a 900s watchdog (+30s queue grace). Extending campaigns reuses authenticated historical endpoints, never the differently normalized model as the reader-none control.
- [Frozen residual diagnosis](../benchmark_results/temporal-moment-witness-20260920-full-none/): train-only tuned/refitted fixed instruments and frozen-state corrections worsen the leader's close score to **0.977356 / 0.977343**, from 0.975143. Neither is promoted. Job **8570 succeeded**; all 46 emitted binary report bases are registered for the TUI.
- Four moment/direct-MSE arms completed in **263.60 / 252.23 / 252.04 / 247.32s**, jobs **8574–8577**. Collector **8578** then failed on relocated-validator certificate path resolution; dependent accuracy **8583** was skipped. [Explicit revalidation](../benchmark_results/lejepa-campaigns/temporal-moments1400-20260920/arms/collect-revalidated.json) authenticated all 14 collection endpoints and the corrected validator without retraining or overwriting the original failure; [execution provenance](../benchmark_results/temporal-moments-20260920-assets/execution.json).
- All four treatment witness jobs **8584–8587** succeeded. Both fixed and state corrections worsen every treatment's aggregate validation score; five witness runs each emit the same 46 registered report bases. Exact correction scores and report links are in the completed experiment.
- Reader jobs **8640–8643** completed in **256.07 / 251.17 / 259.49 / 259.14s**; collector **8644** authenticated 18 endpoints. Reference probes **8647**, corrected-provenance diagnostic **8648**, and controlled memory **8649** all succeeded. All five frozen panels emitted ten registered delayed bases, 2048 validation rows at every delay. [Execution provenance](../benchmark_results/reader-sigreg-20260920-assets/execution.json).
- Clean unanchored GPU contracts8743 passed 38/38; market8757–8760, collector8761 and memory8752 succeeded. Prior market8747–8749 warmup OOMs (external26.04GiB process), completed8750, test failures and skipped dependencies are preserved, not silently replaced. [Execution and exact plans](../benchmark_results/unanchored-sigreg-20260920-assets/execution.json).

## Active question

**Retain the measured supervised forecasting leader: 0.975143, `reader_norm=rms`, `sigreg_placement=off`.** The clean tests now establish that local SIGReg improves old-cue accessibility in the controlled task without downstream anchoring, but not market prediction under the matched frozen-reader protocol. State's small market advantage does not establish a robust edge, and splitting SIGReg between temporal targets and reader state worsened the fresh temporal control. The unresolved question is how an unanchored temporal objective preserves weak market-relevant information, rather than merely Gaussian geometry or synthetic memory. [Clean results and limitations](temporal_sigreg.md#clean-combined-targetstate-sigreg-follow-up).
