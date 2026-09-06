# Market path fix: threshold-defined steps

Universe: 4,873 tickers, bounds [1692269700000, 1723816200000, 1755641700000]. Diagnosis by a throwaway `#[ignore]` test in corpus.rs (deleted) over the memory-mapped bars; "validation" = [bounds[1], bounds[2]); ticker own-return std = median over tickers of the std of consecutive 5-min returns / of overnight (ET-date-crossing) returns.

## Root cause (before, old construction == threshold 1)

The shared grid carries extended hours 04:00–20:00 ET. Population per slot in validation (p50 / max): 04h 236/1074, 05h 196/580, 06h 204/727, 07h 302/1394, 08h 513/1993, 09h 3515/4291 (09:00–09:25 sparse), 10h–15h 3700–3950/4180, 16h 424/3076, 17h 288/929, 18h 255/775, 19h 253/618. Regular-hours population by year (p1/p50): 2016 2392/2583, 2017 2526/2699, 2018 2716/2918, 2019 2894/3111, 2020 3147/3432, 2021 3489/3789, 2022 3829/4079, 2023 3750/3980, 2024 3603/3849, 2025 3556/3743, 2026 3433/3622.

Old construction (every populated slot is a step, contributors = tickers whose previous valid bar is the previous populated slot):
- all: 480,749 steps; contributors p0 1, p1 4, p5 11, p10 23, p25 79, p50 221, p75 3207, p90 3653, p99 4054; <50: 83,272, <200: 232,193, <1000: 284,304. Steps with <50 contributors by ET hour: 04h 12,342; 05h 14,595; 06h 14,108; 07h 8,316; 08h 682; 16h 3,512; 17h 9,436; 18h 10,127; 19h 10,062.
- The overnight move was a 20:00→04:00 step over 1–357 contributors plus ~110 sparse pre/after-market steps. Largest |steps| (all): +9.53% 2020-11-24 04:00 (1 contributor, pop 256), +8.75% 2018-09-24 04:40 (3), +8.34% 04:35 (4), −7.84% 2018-09-27 06:00 (7), −7.82% 2021-03-12 04:00 (1), −7.68% 2016-10-06 04:00 (5), −7.60% 2020-12-02 04:00 (1), +6.22% 04:35 (1), +5.69% 04:00 (8), −5.39% 05:05 (7), −5.20% 2025-04-07 04:00 (357), … 12 steps > 5%, all extended-hours, none market-wide. Validation top: −5.20% 2025-04-07 04:00 (357), +3.86% 2025-04-07 10:10 (4022, real), −3.49% 04:00 (195), −2.88% 04:00 (262), +2.72% 04:00 (282), −2.67% 10:20 (4034, real), … 04:00 steps with 1 contributor at +2.2% and +1.7%.
- Step std vs median ticker: all-period intraday 0.001123 vs 0.002814, overnight 0.007601 vs 0.014294; validation intraday 0.000900 vs 0.002488, overnight 0.007380 vs 0.013810 (aggregate std already below the ticker's — the damage is the ~114 noisy sparse steps summed into every overnight, not single-step std).
- Effect on targets (61,901 validation windows, close channel, σ = context 5-min return std as in the model): unscaled mean (y−m)²/mean y² = 0.969 (h=1), 0.931 (h=48), 0.983 (h=192); σ-scaled persistence MSE absolute vs relative: h=1 1.08 vs 1.79, h=48 29.61 vs 93.09, h=192 285.11 vs 619.40.

## Fix (features.rs `MarketSteps`, corpus.rs, runner.rs `--market-min-cross-section`)

A slot defines a step only if ≥ `market_min_cross_section` tickers hold a valid bar there; the step is the mean over tickers holding valid bars at the slot *and at the previous defining slot* of ln(close/close_prev_defining); non-defining slots add nothing and read the previous value; the next defining step spans them. Both `market_cum` and the aux market channel (`Feature::Market`, now the step ending at the bar's timestamp, invalid elsewhere) derive from the same `MarketSteps`; SPY channel unchanged. Mean kept: after the threshold every step has ≥1,503 contributors and the largest steps are real market-wide days (below), so a median is unnecessary. Manifest `market_fingerprint` now digests the threshold; contract gains `market_min_cross_section`; SCHEMA bumped to `timexer-pooled-mmap-v5`; `Corpus.market: MarketSummary` (steps, min/median contributors, step std, largest step + ts) is written into the `timexer_segment_progress` corpus title.

Threshold: **2000** (not 500). With 500, extended-hours bursts still define steps (08:xx max pop 1993, 07:00 500–900) with 200–400 contributors from a biased pre-market subset: all-period p1 contributors 313, p10 618; largest steps −9.0% 2020-03-16 07:00 (328 contributors), −6.0% 2020-03-09 07:00 (321), +5.5% 2020-03-10 08:00 (311); unscaled h=192 ratio 0.913 with 500 vs 0.889 with 2000. 2000 sits between the extended-hours maximum (1,993) and the regular-hours minimum (2016 p1 = 2,392), so exactly the regular session plus one 15:55→09:30 overnight step per session defines the path (197,362 steps ≈ regular-session slots).

## After (threshold 2000)

- all: 197,362 steps; contributors p0 1503, p1 2306, p5 2434, p10 2539, p25 2834, p50 3408, p75 3647, p90 3831, p99 4173, max 4498; <50: 0, <200: 0, <1000: 0. Validation: 19,797 steps; p0 1791, p1 2741, p10 3390, p50 3544, p99 4030.
- Step std vs median ticker: all-period intraday 0.000987 vs 0.002814, overnight 0.008760 vs 0.014294; validation intraday 0.001077 vs 0.002488, overnight 0.008682 vs 0.013810. Requirement met both ways.
- Steps > 5% (all period): 8, all 09:30 overnight steps on market-wide days: −10.03% 2020-03-16, −8.27% 2020-03-12, −7.96% 2020-03-09, +5.98% 2020-03-24, +5.43% 2020-03-13, −5.16% 2024-08-05 (yen-carry unwind), −5.15% 2020-03-18, +5.13% 2020-11-09 09:25 (Pfizer vaccine, 1843 contributors, pop 2250). Validation: 0 steps > 5%; largest −3.95% 2025-04-07 09:30 (2876), +3.86% 2025-04-07 10:10 (4022), +3.68% 2025-05-12 09:30 (2698) — tariff-pause/tariff days.
- Effect on targets: unscaled (y−m)²/y² = 0.970 (h=1), 0.872 (h=48), 0.889 (h=192); σ-scaled persistence MSE absolute vs relative: h=1 1.08 vs 1.67, h=48 29.61 vs 55.65, h=192 285.11 vs 407.67 (was 619.40).

## Remaining structural gap (not a construction issue)

h=192 σ-scaled by context-σ quintile (absolute / relative / pooled β of y on m): q0 σ<0.00205: 952.9 / 1652.2 / β 0.42; q1: 133.2 / 105.5 / 0.74; q2: 126.2 / 96.8 / 0.94; q3: 111.8 / 93.5 / 1.07; q4 σ>0.006: 101.5 / 90.3 / 1.44. Demeaning now helps in 80% of windows; the aggregate relative > absolute comes entirely from the lowest-σ quintile (β ≈ 0.4, and its absolute persistence MSE of 953 already shows those tickers are not random walks at their context σ). Subtracting the full equal-weight market from β≈0.4 tickers adds variance; a causal per-row β (cov/var over the context) scaling `market_drift` would close it — model.rs change, outside this task.

## Verification
- `./torch-env.sh cargo check -p trading_bot_0 --tests`: zero errors. `cargo test -p trading_bot_0 timexer_segment`: 30 passed.
- New unit test `sparse_slots_define_no_step_and_the_next_step_spans_them` (4-source synthetic universe: lone after-hours print, 2-ticker slot, a ticker skipping a defining slot; checks step definition, contributors, spanning path, NaN series at non-defining slots, summary, and that a lower threshold changes the path). Existing gap test renamed `market_steps_span_gaps_and_only_adjacent_defining_slots`; calendar test now asserts the market channel carries the weekend step while SPY stays invalid.
- Docs: `docs/timexer_segment.md` row-layout paragraph and flag list.

## Causal per-origin β (model.rs, follow-up requested by Main)

`Statistics.beta [B, origins]`: `β_k = (Σ r_i m_i + λ_k)/(Σ m_i² + λ_k)` over valid consecutive-bar pairs in `[0, t_k]` (r = ticker log close return, m = row `market_cum` step over the same pair; expanding cumsums like σ_k, fp32, from detached batch tensors), `λ_k = BETA_PRIOR_BARS(256) · max(mean m² so far, RETURN_VARIANCE_FLOOR)`. With no history β = λ/λ = 1; half-way to OLS at 256 pairs; ~4% prior weight at 6,000 bars. `market_drift()` = `β_k·Δmc/σ_k`, so `targets()` subtracts β·drift and the runner's absolute-space scoring (`targets + drift`), `rebased_prices` and `target_prices` add β·Δmc back unchanged. Persistence stays zero coordinates in both spaces. OBJECTIVE → `causal_patch_market_neutral_nll_v2`.

CPU diagnosis (same 61,901 validation windows, threshold 2000, σ-scaled persistence MSE absolute / unit-β relative / causal-β relative):
- h=1: 1.08 / 1.67 / 0.98; h=48: 29.61 / 55.65 / 22.73; h=192: 285.11 / 407.67 / 255.35. β quantiles p10 0.17, p50 0.66, p90 1.25.
- h=192 by context-σ quintile: q0 (σ<0.00205) 952.94 / 1652.15 / 908.13 (β p50 0.30); q1 133.18 / 105.52 / 96.49 (0.52); q2 126.15 / 96.77 / 94.34 (0.71); q3 111.78 / 93.54 / 89.83 (0.85); q4 101.45 / 90.29 / 87.92 (1.06). Relative ≤ absolute in every quintile.

Unit test `a_half_beta_ticker_converges_to_a_zero_close_target_with_history` (seq_len 8192, close = 0.5·market): β_k equals the closed form `(0.5 n_k + 256)/(n_k + 256)` at origins 0, 15, and last (1e-4), first origin > 0.97, the residual target at origin 0 keeps > 50% of the raw market term while the final origin keeps < 7% (exact zero is asymptotic under the ridge; at 8,191 pairs β = 0.515). Existing unit-β tracking test still passes (OLS β = 1 is a fixed point of the shrinkage). Docs updated (statistics, targets, objective, absolute scoring, candle title). cargo check zero errors; 31 timexer_segment tests pass.
