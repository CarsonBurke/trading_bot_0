# First real executable-utility measurements on v10 checkpoints

Source: `evaluate-timexer-segment` on `weights/preview-latest` (step 3000) of
`timexer-{control,cutoff32,invsqrt}-4k`, mlq jobs 5349/5350/5351, reports in each run's
`gens/2`. New `utility.rs` definition: raw simple return, next observed open -> h-th
observed close, gross budget <= 1, cost `c * sum|w_i| (1 + P_exit/P_entry)`.
All numbers DEMONSTRATED unless tagged.

Census (identical for all three, held-out full): 9506 complete eligible origin cohorts,
mean observed valid width **10.610549**, minimum 1. Turnover is ~2.0 for every policy at
every horizon, so `net = gross - 2c` to first order, and breakeven = gross/turnover.

## Policy `mean decile long/short`, held-out full (N = 9506 cohorts)

| arm | h | gross bps | payoff std bps | t = mean/(std/sqrt N) | breakeven bps per unit turnover |
|---|---:|---:|---:|---:|---:|
| control | 1 | -0.053 | 25.9 | -0.20 | -0.026 |
| control | 8 | 5.081 | 108.2 | 4.58 | 2.540 |
| control | 16 | 13.577 | 140.7 | 9.41 | 6.787 |
| control | 32 | 21.922 | 183.0 | 11.68 | 10.955 |
| control | 64 | 40.830 | 215.4 | 18.48 | 20.398 |
| control | 192 | 3.436 | 4802.9 | 0.07 | 1.709 |
| cutoff32 | 1 | 0.401 | 25.9 | 1.51 | 0.200 |
| cutoff32 | 8 | **8.853** | 107.5 | **8.03** | 4.426 |
| cutoff32 | 16 | 14.608 | 142.4 | 10.00 | 7.302 |
| cutoff32 | 32 | 22.091 | 183.1 | 11.76 | 11.039 |
| cutoff32 | 64 | 0.0 exactly | 0.0 exactly | undefined | NaN |
| cutoff32 | 192 | 0.0 exactly | 0.0 exactly | undefined | NaN |
| invsqrt | 1 | 0.218 | 25.4 | 0.84 | 0.109 |
| invsqrt | 8 | 6.144 | 106.7 | 5.62 | 3.072 |
| invsqrt | 16 | 13.724 | 140.9 | 9.50 | 6.861 |
| invsqrt | 32 | 22.576 | 177.9 | 12.37 | 11.282 |
| invsqrt | 64 | 41.173 | 213.0 | 18.85 | 20.568 |
| invsqrt | 192 | 0.122 | 4797.3 | 0.00 | 0.061 |

The `t` column is arithmetic from the two demonstrated columns and the demonstrated cohort
count. HYPOTHESIS: it overstates significance, because cohorts overlap in calendar time and
share market factors; it is not an iid sample.

cutoff32's exact zeros above h=32 are the untrained head rows, not a measurement.

## Same policy, held-out cross-section draw (100 timestamps x 40 names)

| arm | h=1 | h=8 | h=16 | h=32 | h=64 | h=192 |
|---|---:|---:|---:|---:|---:|---:|
| control gross bps | -0.757 | 1.823 | -0.961 | -5.325 | 7.778 | 46.943 |
| cutoff32 gross bps | 0.131 | **9.329** | **11.079** | **11.450** | 0.0 exactly | 0.0 exactly |
| invsqrt gross bps | 0.787 | 5.843 | 1.578 | -8.403 | 16.193 | 35.119 |
| control breakeven | -0.379 | 0.912 | -0.480 | -2.655 | 3.878 | 23.443 |
| cutoff32 breakeven | 0.066 | **4.666** | **5.534** | **5.708** | NaN | NaN |
| invsqrt breakeven | 0.394 | 2.922 | 0.788 | -4.190 | 8.077 | 17.532 |

This is the decisive comparison. On the aligned cross-section draw the control's short-horizon
signal is not tradable at all - negative at h=1, 16 and 32 - while cutoff32 is consistently
positive across h = 8/16/32 with a 4.7-5.7 bps per unit turnover cost allowance. invsqrt sits
between them and also goes negative at h=32.

## Reference policies and what the long horizons actually are

`equal long` payoff, held-out full: 0.196 / 0.829 / 3.832 / 6.661 / 12.733 / 26.456 bps at
h = 1/8/32/64/128/192. That is market drift on a fully invested long book, not skill, and it
grows monotonically with h. HYPOTHESIS: the control's large h=64 long/short gross (40.8 bps,
t=18.5) is substantially drift/beta leakage through 10.6-name cohorts rather than
cross-sectional alpha, which is consistent with its h=64 held-out MSE ratio being ABOVE
persistence (1.0128) at the same checkpoint. A forecaster cannot simultaneously be worse than
persistence at h=64 and have genuine 40 bps of h=64 cross-sectional alpha.

`one-sigma gated sign` is near-inactive (active fraction 0.0001 at h=1, 0.0049 at h=8, 0.0109
at h=32): the head's sigma exceeds |mean| almost everywhere, so the uncertainty-gated policy
almost never fires. This is the same over-dispersion visible in cutoff32's calibration
(0.868 within 1 sigma vs nominal 0.683).

## Sizing the honest edge

At h=8, cutoff32, held-out full: gross 8.853 bps, turnover 2.0, so net at 1 bps/side cost is
8.853 - 2 = 6.85 bps per non-compounded 8-bar hold, and per-bar rate 1.107 bps. Cohort width
10.6 means the decile long/short is roughly one long and one short name, so this is a
concentrated, high-variance edge (std 107.5 bps), not a diversified book. The cross-section
draw's 40-name cohorts give 9.329 bps with std 38.2 - same edge, one third the dispersion,
which is the expected effect of diversifying the same signal.

## Open items

- Cohort width 10.6 is the binding measurement constraint and is an indexing artifact
  (`corpus.rs:431` places validation origins on each ticker's own valid-bar ordinal).
- No annualization, no Sharpe, no capital schedule: cohorts overlap and are not fundable
  simultaneously. `utility_rate` is bps of initial capital per observed bar, nothing more.
