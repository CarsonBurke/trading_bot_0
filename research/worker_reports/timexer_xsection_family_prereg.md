# Pre-registration: second cross-section feature family

Written before the arm exists. Thresholds fixed here are NOT adjusted after seeing results.

## What is already demonstrated

The first cross-section family (`Feature::Dispersion` = `ln(sigma_slot)`, `Feature::CrossSectionZ` =
`clamp((own 5-min log close return - market step)/sigma_slot, +-16)`) was CONFIRMED on the anchored
`held-out full` student IC at step 2000, against the pinned control checkpoint scored through the
same instrument, same anchored draw, same eval batch size:

| h | control | +cross-section | delta | rel |
|---|---------|----------------|-------|-----|
| 1 | 0.16208 | 0.19362 | +0.0315 | +19.5% |
| 8 | 0.13375 | 0.16365 | +0.0299 | +22.4% |
| 16 | 0.12622 | 0.15097 | +0.0248 | +19.6% |
| 32 | 0.13037 | 0.14296 | +0.0126 | +9.7% |
| 64 | 0.12609 | 0.13935 | +0.0133 | +10.5% |
| 128 | 0.10267 | 0.12034 | +0.0177 | +17.2% |
| 192 | 0.08722 | 0.10276 | +0.0155 | +17.8% |

Single-arm SE at h=1 is 0.009976 over 245 paired cross-sections. Every horizon improved; there is no
short-for-long trade on the information axis. The apparent long-horizon MSE-ratio regression on the
same checkpoint (h=192 1.0245 -> 1.0433) is amplitude, not information: IC is scale-invariant per
horizon and the forecast is 5x over-amplitudinal at h=192.

## Mechanism under test

The first family standardized ONE variable (the close return) against the contemporaneous
cross-section. The input side still describes no other own-bar attribute relative to the universe,
while the target is defined entirely inside that cross-section. Three additions, one family:

1. `CrossSectionRank` - normal score of the own close return's rank within the slot's contributing
   tickers, `Phi^-1((r - 0.5)/N)`. Robust where `CrossSectionZ`'s +-16 clamp is not; a linear head
   consumes a normal score better than a uniform rank. Validity identical to `CrossSectionZ`.
2. `RelativeVolume` - z-score of `ln(volume)` against the slot's cross-sectional `ln(volume)`
   distribution. Volume is already an own-feature but is never cross-sectionally standardized, so
   the model cannot currently tell a heavy tape from a heavy stock.
3. `RangeZ` - z-score of `ln((high - low)/close)` against the slot cross-section. Own intrabar
   volatility relative to the universe, distinct from `Dispersion` (which is the universe's own
   dispersion, carrying no own-bar information).

`Breadth` (signed fraction of the slot up) is deliberately EXCLUDED as near-collinear with the
existing `market` channel.

Both new z-scores need slot accumulators for `sum ln x` and `sum (ln x)^2` over contributing
tickers, alongside the existing `MarketSteps::squares`.

## Predicted cost, stated before the run

- Parameters: +16,384 per channel (measured on the first family: +32,768 for two channels), so
  +49,152, taking 27,987,243 -> 28,036,395 (+0.176%).
- Step FLOPs: +0.17% (first family measured +0.111% for two channels).
- Host batch: the first family cost +22.2% (114.13 -> 139.49 MB/batch). Three more channels take it
  to roughly 190 MB/batch. This is the one number that can bite: measured `host batch assembly on
  the loader thread` is 46-50 ms against a 154 ms step, so it is free only while it stays below the
  step. If assembly exceeds ~150 ms the loader stops being free and the arm's step time regresses.
  MUST read `timexer_segment_timing` on the arm and report assembly and the exposed loader wait.

## Thresholds

Primary metric: anchored `held-out full` student IC at h=1, step 2000, same instrument and same
eval batch size as the 0.19362 baseline. Baseline single-arm SE 0.009976; an unpaired difference of
two such arms has SE 0.0141.

- CONFIRMED-strong: h=1 IC >= 0.2218 (baseline + 2 unpaired SE).
- CONFIRMED-weak: h=1 IC >= 0.2077 (baseline + 1 unpaired SE). Labelled weak because the two arms
  share the anchored draw, so their errors are correlated and the unpaired SE overstates the true
  paired dispersion - but the paired statistic is not implemented, so the weak bar is reported as
  suggestive only and never as the verdict.
- REFUTED-harmful: h=1 IC <= 0.1655 (baseline - 2 unpaired SE).
- Otherwise UNRESOLVED, and must not be written up as a verdict.

Long-end guard, required in addition for any CONFIRMED reading: h=192 IC >= 0.0856 (baseline
0.10276 less 2 single-arm SE of 0.0086). A family that buys h=1 by giving up the long end fails
this pre-registration even if the primary bar is met - that is the standing instruction on this
model, not a metric preference.

Throughput guard: if `host batch assembly` exceeds the training step period, the arm is scored but
the family is re-run with the channels split before any of it is adopted, because a step-time
regression changes what a step-matched comparison means.

## What this pre-registration does not claim

- No prediction on the MSE ratio. Until the amplitude calibration lands, the per-horizon MSE ratio
  mixes information with a 18.8x amplitude sweep and cannot referee an information change.
- No prediction on trading metrics. The portfolio path is mid-cutover to the checkpoint-carried
  gain.
