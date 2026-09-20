# Top-run ledger

Updated: 2026-09-20. Replace current standings after every completed matched comparison; keep only decision-changing history. Accuracy means observable future-price/return accuracy, never latent JEPA loss. No production promotion or terminal-test claim from these development panels.

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

At h192, no-SIGReg JEPA = **0.997561**, full/none = 1.000512, decoupled/lattice = 1.001884. A horizon-specific win is not a global win. Latent-only one/multi-horizon close scores ≈0.99935/0.99998: not competitive.

## Historical references — not matched treatments

Same scoring observations, but **2500 updates, different seed, frozen calibrated means, future-calendar inputs**. Do not attribute differences solely to model design.

| Model | Close score ↓ | h64 close score ↓ | h64 direction ↑ | h64 signed IC ↑ |
| --- | ---: | ---: | ---: | ---: |
| `timexer-decoupled-lattice-2500` | 0.978768 | 0.985270 | 51.76% | 0.04870 |
| `timexer-full-none-control-2500` | 0.995667 | 0.995200 | 51.03% | 0.05353 |

## Evidence and operational state

- [Latest paired accuracy reports](../benchmark_results/accuracy-temporal-sigreg-final-20260920/): **14 checkpoints**, 25 `timexer_accuracy_*` `.report.bin` bases and authenticated protocol; job **8523 succeeded**, **25.945s**. [Commands and interpretation](timexer_segment.md#temporal-sigreg-diagnosis).
- Evaluation job **8480 succeeded**, nine models in **26.845s**. Original six models: job **8444 succeeded**, each 233–268s. Missing full/none trained in **237.02s**.
- Five new fixed-1400 runs completed: projected ±SIGReg **276.06/280.62s**, conditional CF decoupled/full **243.97/242.36s**, 16D projected **247.11s**. Training jobs **8494/8495/8512/8513/8521**; all old endpoints reused.
- Full/none job **8476** failed only in post-training validation of omitted Serde defaults. Validator corrected; saved endpoint independently authenticated and scored without retraining. Original failed receipt preserved; [reverification provenance](../benchmark_results/accuracy-decoupled-comparison-20260920-assets/control-endpoint-reverification.json).
- New campaigns use one bounded queue job per model, fixed update budget, 420s watchdog; extending a matched campaign reuses authenticated endpoints.

## Active question

**Decision: retain full/none forecasting-only; no temporal auxiliary is promoted.** Moving SIGReg off the forecast input reverses its measured penalty, but neither 512D nor 16D projected SIGReg beats forecasting-only. Fixed-return CF learns held-out conditional features (3–8% error reduction versus frozen train-only means), yet worsens price accuracy. Better Gaussianity, latent prediction, or conditional-feature skill is insufficient. Which predictable components are task-irrelevant remains unisolated; no claim of a global optimum or absence of learnable price information.
