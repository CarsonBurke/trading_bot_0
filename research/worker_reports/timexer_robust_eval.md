# TimeXer causal patch — robust full-validation evaluation

Run `timexer-causal-20260905b`, checkpoint `weights/best` (epoch 1, step 9591), 433,303 validation origins, reports in `gens/2` (mlq job 5069, eval 107 s).
Aggregate (σ-units): MSE 75.80 vs persistence 70.82 (ratio 1.070); NLL 2.539 vs 2.497; `tail_loss_share` 0.368; **median-window MSE ratio 1.075**.

| h | MSE ratio | trimmed MSE ratio (top-1% \|close\| dropped) | MAE ratio | win rate (close) | directional hit rate | predicted-up share |
|---|---|---|---|---|---|---|
| 1 | 0.948 | 0.853 | 0.937 | 0.457 | 0.464 | 0.478 |
| 2 | 0.961 | 0.929 | 0.975 | 0.465 | 0.478 | 0.493 |
| 4 | 0.981 | 0.980 | 0.997 | 0.470 | 0.490 | 0.481 |
| 8 | 1.012 | 1.035 | 1.018 | 0.467 | 0.501 | 0.522 |
| 16 | 1.068 | 1.112 | 1.046 | 0.457 | 0.507 | 0.547 |
| 32 | 1.083 | 1.127 | 1.052 | 0.455 | 0.514 | 0.542 |
| 64 | 1.079 | 1.116 | 1.048 | 0.458 | 0.519 | 0.522 |
| 96 | 1.080 | 1.105 | 1.045 | 0.460 | 0.519 | 0.517 |
| 128 | 1.062 | 1.097 | 1.042 | 0.460 | 0.517 | 0.533 |
| 192 | 1.069 | 1.094 | 1.039 | 0.466 | 0.522 | 0.542 |

Interpretation:
1. The model is not better in typical windows: the median window loses to persistence (1.075), and dropping the top-1% |close| bars makes long-horizon ratios worse (1.09-1.13 vs 1.06-1.08), so at h≥8 the loss is in the body of the distribution, not the tail; the tail is where the model does relatively least badly.
2. Short-horizon skill (h≤4) is real and body-driven: trimmed ratio 0.85 at h=1 beats the untrimmed 0.95, and MAE ratio 0.94.
3. Win rate is below 0.5 at every horizon (0.455-0.47): the model's per-bar close error beats persistence on a minority of bars, so the h≤4 MSE gain comes from fewer, larger wins, not from being usually right.
4. There is an up-bias: predicted-up share rises from 0.48 at h≤4 to 0.52-0.55 at h≥8, coincident with the horizons where MSE/MAE turn negative-skill; the 0.51-0.52 directional hit rate at h≥16 is within what a level shift toward the period's drift would produce, so it is not evidence of conditional directional skill [INFERENCE].
5. Directional hit rate below 0.5 at h≤4 is deflated by exact-zero close targets (no price change over 5-20 min), which count as misses for any nonzero forecast; MAE/MSE ratios are the reliable short-horizon signal.
