# The reserved calibration partition: root cause, population, and what it removes

Scope: `corpus.rs` origin placement and partition logic, the corpus contract and its schema
stamp, plus the `write_corpus` report title. No GPU work. The held-out sample draw, the
held-out cross-section draw, the held-out full population and the training origins are
untouched, and that is proven rather than asserted.

## 1. Root cause — DEMONSTRATED

Two lines, both in `corpus.rs`, and the partition was empty **by construction**, not by an
off-by-one, not by the `target_count` window arithmetic, and not by the quantile boundaries.

* **`corpus.rs:678-701` (pre-change), origin enumeration.** `Corpus::load` built exactly two
  populations. The validation loop read `start = c.boundaries[1].max(common_context)` and
  `available = retained_partition_end(c.boundaries[2], c.valid_bars, purge) - start`. There was
  no loop for `boundaries[0] -> boundaries[1]`. The band the schema reserves was enumerated by
  nothing.
* **`corpus.rs:113-125` (pre-change), `CorpusTicker::target_count`.** Its held-out arm was
  `start = boundaries[1] - 1`, `end = retained_partition_end(boundaries[2], ..) - pred_len`. An
  origin between `train_end - 1` and `boundaries[1] - 1` fell through both arms and returned
  `None`, so even a hand-constructed calibration origin was rejected at `Corpus::sources`
  (`corpus.rs:733-735`) with *"origin is outside unlocked purged targets"*.

Evidence that the other candidate mechanisms are innocent:

* `quantile_bounds` (`corpus.rs:1300-1324`) is correct and does produce three boundaries at the
  70/80/90 ranks; the real corpus reports `boundary_timestamps = [1692269700000, 1723816200000,
  1755641700000]`, three distinct instants roughly 12 months and 12 months apart. The band is
  not degenerate.
* `retained_partition_end` (`data.rs:155-161`) is correct: applied to `boundaries[1]` it yields
  `boundaries[1] - purge`, which is a *wide* admissible window, not an empty one. The measured
  population below is 433,721 origins, so the window was always there — nothing read it.
* The sibling bug Main named (per-ticker valid-bar ordinals, so one missing bar shifts later
  origins) is real but is a *misalignment* defect, not an *emptiness* defect. I mirrored the
  existing placement rule exactly rather than changing it, because the calibration block must
  be constructed the same way as the validation block it transfers to, and because changing
  validation placement would move the held-out sample draw — an explicit non-goal.

## 2. The fix

One rule, both held-out partitions.

* `CorpusTicker::owns_partition_targets(origin, first)` (`corpus.rs:127-136`): admissible iff
  `origin >= boundaries[first] - 1` and `origin < retained_partition_end(boundaries[first+1],
  valid_bars, purge) - pred_len`. `first = 0` is the calibration partition, `first = 1` the
  validation partition. `target_count` now returns `Some(pred_len)` for either.
* `Corpus::load` (`corpus.rs:685-719`): a `band(contract, ticker, first)` closure strides
  `start - 1 + i * pred_len` for `start = max(boundaries[first], common_context)`, and is called
  for `first = 0` into the new `Corpus::calibration_refs` and for `first = 1` into
  `validation_refs`. The `first = 1` call is character-for-character the arithmetic it replaced.
* `Corpus::load` now **hard-errors** on an empty calibration population, next to the existing
  train/validation check. A silently empty reserved partition cannot ship again.

Disjointness is a property of the corpus, not of a caller. Calibration targets live in
`[boundaries[0], boundaries[1] - purge)`; the first validation origin is `boundaries[1] - 1`;
`purge = max(pred_len, 100) = 192` at the production geometry. So the last bar any calibration
target reads precedes the first validation origin by at least 192 valid bars. The consumer needs
no split arithmetic and no purge-gap accounting of its own.

Cost: `calibration_refs` is 433,721 `WindowRef` (16 B each) = **6.9 MB** of host memory, built in
the existing origin-enumeration phase, measured at **30.9 ms** total for all three populations on
the real corpus (`row and origin enumeration`, warm caches). Zero FLOPs, zero device memory, zero
step time. Nothing is loaded until a caller asks for a batch.

## 3. Population statistics — DEMONSTRATED

Real corpus, production geometry (`context 6000, pred_len 192, common_context 6000,
market_min_cross_section 2000`, features as published), warm caches, zero bar records rescanned:

| population | origins | target bars | distinct timestamps | distinct tickers | mean cross-sectional width |
|---|---|---|---|---|---|
| calibration `[70%, 80%)` | **433,721** | 83,274,432 | 38,190 | 4,498 | **11.357** (widest timestamp 1,548) |
| validation `[80%, 90%)` | 433,303 | 83,194,176 | — | — | — |

The two held-out partitions are the same size to within 0.1% (433,721 vs 433,303), which is what
you expect from two adjacent deciles of the same union grid.

### SE(ln beta) comparison

The prior fit block is recorded in `training/runs/timexer-control-4k/mean-calibration-step3000.json`:
`calibration_origins = 216,664`, `evaluation_origins = 109,547`, `purged_origins = 107,092`
(sum 433,303 = the whole validation population). So the fit currently uses 216,664 origins — half
of validation — and the *scoring* block is only 109,547, because the chronological split throws
107,092 origins into the purge gap where neither block scores them.

For `beta_hat = Cov(f,y)/Var(f)`, `SE(ln beta_hat) ~= sqrt(1 - rho^2) / (rho * sqrt(N))`.

* Fit population: 216,664 -> **433,721**, a factor **2.0018**. At equal per-observation `rho`,
  `SE(ln beta)` scales by `1/sqrt(2.0018) = 0.7068`, so the measured 0.027-0.037 band becomes
  **0.019-0.026** — strictly tighter than the band that made the transfer work. DEMONSTRATED for
  the counts; HYPOTHESIS for the exact SEs, because `rho` in the `[70%, 80%)` decile has not been
  measured (that needs a scoring pass, i.e. the GPU I was told not to use).
* Scoring population: 109,547 -> **433,303**, a factor **3.956**. The purge gap disappears
  entirely, because the corpus's own purge band holds no origins at all, so no origin has to be
  discarded to separate the blocks.

Cross-sectional width 11.357 comes from the same per-ticker-ordinal placement validation uses, so
it is comparable to validation's. It is thin per timestamp but the widest calibration timestamp
holds 1,548 tickers, so a cross-sectional draw out of this partition is feasible if wanted. The
amplitude fit itself pools per horizon and does not need width.

## 4. Every existing split is byte-identical — DEMONSTRATED

Two independent proofs.

* **Real corpus, against a pre-change artifact.**
  `the_real_corpus_calibration_partition_is_measured_against_a_published_contract`
  (`#[ignore]`, `corpus.rs:2258-2311`) loads the real corpus at the geometry recorded in
  `training/runs/timexer-control-4k/timexer-segment-data-contract.json` and asserts
  `corpus.contract == published`. That artifact was written by a **pre-change binary** on
  2026-09-07 16:57, and no `*.300.bars` file has been written since (`find -newermt '2026-09-07
  16:00'` returns 0 of 10,476). The assertion passes, so all 4,873 per-ticker `DataContract`s,
  the three boundary timestamps, `train_target_bars = 470,946,393`,
  `validation_target_bars = 83,194,176`, `validation_remainder_bars = 408,842`, the market
  fingerprint and the 855 exclusion records are all bit-identical.
* **Fixture, against the pre-change rule restated.**
  `the_calibration_partition_has_a_population_and_perturbs_no_existing_split`
  (`corpus.rs:1947-2158`) builds a three-ticker scratch corpus, one ticker with quarantined OHLC
  rows so valid-bar ordinals really differ from raw indices and one ticker ending inside the
  calibration partition, then compares `corpus.train_refs` and `corpus.validation_refs` element
  by element against the pre-change placement expression written out in the test. It also pins
  the contract's exact 18-key serialized field set, proves every calibration origin owns a
  complete horizon, proves no target bar is claimed by two populations (a per-ticker ownership
  map over all three populations), and proves the last calibration target timestamp precedes the
  first validation origin timestamp.

### Held-out sample persistence NLL 2.3947663

The held-out sample is `fixed_origins(&corpus.validation_refs, args.eval_origins)`
(`runner.rs:1638`). `fixed_origins` is a deterministic stride over its input and I did not touch
it; `validation_refs` is proven identical above; the bars, the loss geometry and the persistence
baseline are untouched by this change. Therefore the persistence NLL is bit-identical at
2.3947663. I did not re-execute it — that is a GPU scoring pass and this batch's GPU policy
reserves submission to `FuseLoss` — so this is a derivation from proven-identical inputs plus
determinism, not a re-measurement. The command that re-measures it is in section 7.

## 5. Contract and schema stamp: deliberately NOT bumped — DEMONSTRATED

The corpus artifacts do **not** change shape, so there is nothing to bump, and bumping anyway
would be actively destructive:

* `load_checkpoint` asserts `corpus.contract == manifest.data` (`runner.rs:2177-2180`) against the
  contract serialized inside each checkpoint manifest, and `Manifest::read` first checks
  `manifest_sha256 == self.digest()` (`runner.rs:507-510`) where the digest covers `data`.
* `training/runs/timexer-control-4k/weights/best/manifest.json` carries `data` with exactly the
  18 keys the contract has today and today's `SCHEMA` verbatim, and its
  `weights_sha256 = ccffa620...09116` is the same forecaster the amplitude result was measured on.
* So an added contract field or a changed schema string makes that checkpoint fail to load with
  *"dataset differs from authenticated universe/splits"*, and re-stamping the manifest means
  recomputing the digest that authenticates it. There would be no calibration to fit.

It is also unnecessary. The calibration population is a pure function of `boundaries`,
`valid_bars`, `purge`, `pred_len`, `common_context` and the ordered ticker set — every one of
which is already inside `CorpusContract`, and therefore already inside
`Pairing::corpus_sha256`, a SHA-256 over the whole serialized contract (`runner.rs:537-541`). An
identical contract digest cannot yield a different calibration population, so adding a derived
scalar to the contract would authenticate nothing new. The population size is reported instead
through `reports::write_corpus`, which now takes `calibration_origins: usize` and states the
`[70%, 80%)` target-bar count beside the training and held-out counts in the
`timexer_segment_progress` title. No new report base, no registry change.

### The cache-invalidation guarantee, verified

`bumping_the_corpus_schema_invalidates_every_derived_cache_and_no_bar_audit`
(`corpus.rs:2160-2230`) drives the real `corpus::SCHEMA` constant through `BoundsCache` and
`MarketCache`, shows both hit under it and both **miss** under `SCHEMA` plus one appended clause,
and asserts `corpus::SCHEMA != data::SCHEMA` — which is why a partition-geometry bump would leave
the expensive per-ticker bar audits (keyed on `data::SCHEMA` and inode identity) alone. So the
stale pairing Main was worried about is unreachable *if* the stamp ever does move; today it does
not move, and it does not need to. The pre-existing
`a_cache_built_against_one_corpus_contract_is_rejected_when_the_contract_changes` still passes on
all three of its axes.

## 6. Does this remove the selection-contamination caveat? PARTIALLY — and here is which half

Confirmed from code, not assumed:

* The selection scalar is `selection_criterion(...) = "min held-out sample objective-weighted NLL
  (horizon-loss=...)"` (`runner.rs:552-554`), compared as `evaluation.objective_nll < best_objective`
  and `< best_full` (`runner.rs:1839, 1852`).
* `evaluation` is `score(..., selected, ...)` where `selected` is `&corpus.validation_refs` on a
  complete epoch and `&preview` otherwise, and `preview = fixed_origins(&corpus.validation_refs,
  args.eval_origins)` (`runner.rs:1638`). Early stopping (`stale_previews`, `stale_epochs`) reads
  the same two scalars. The candle draw (`runner.rs:1550`) and the cross-section draw
  (`runner.rs:1639`) are also functions of `validation_refs` alone.
* Training targets live in `[common_context, boundaries[0] - purge)` and every training origin's
  context is earlier still, so no gradient ever saw a `[70%, 80%)` bar as a target.
* Nothing anywhere reads a `[70%, 80%)` target. The only overlap is that a validation origin's
  6,000-bar *context* reaches back into the calibration partition — an input, carrying no label
  information about the calibration block's own targets.

Therefore:

* **Removed entirely, on the FIT side.** The frozen gain is no longer a function of any
  observation that selected the checkpoint. That was the load-bearing half: a gain fitted on the
  population that chose the weights is a curve-fit dressed as a calibration.
* **Not removed, on the SCORE side.** Selection minimized NLL over the validation split, and the
  validation split is where the calibrated-versus-uncalibrated ratios are reported. The absolute
  *levels* of those ratios remain measured on a population selection optimized — exactly like
  every other published number of this run. The *difference* between the two arms is not a
  selection artifact: both arms are the same weights on the same origins, differing only by a
  scalar fitted elsewhere.
* The only block untouched by both training and selection is the terminal test partition
  `[90%, 100%)`, which is locked (`terminal-test-locked` in the schema). Removing the score-side
  caveat means spending that partition, and that is a decision, not a fix.

Net: the previous caveat — *"both halves of the fit/score split are equally contaminated,
because selection minimized NLL over the whole validation split"* — is gone as stated. It is
replaced by a strictly weaker one: the scoring population is selection-optimized, the fit
population is not, and the two share no observation.

## 7. Commands

Fit the calibration on the new partition (requires `GainTwoSided`'s `calibrate` rewire, which
drops `--calibration-share` since the corpus now supplies both blocks):

```
OMP_NUM_THREADS=1 TORCH_NUM_THREADS=1 RAYON_NUM_THREADS=1 ./torch-env.sh \
    ./target/release/trading_bot_0 calibrate-timexer-segment \
    --checkpoint training/runs/timexer-control-4k/weights/best \
    --data-dir long_data/bars \
    --output training/runs/timexer-control-4k/gens/3 \
    --calibration training/runs/timexer-control-4k/mean-calibration-partition70.json \
    --batch-size 256
```

`weights/best` is step 2000 with `weights_sha256 = ccffa620...09116`, which is the exact
forecaster `mean-calibration-step3000.json` was fitted against, so the new curve is comparable to
the old one at fixed weights. `weights/preview-latest` is step 3000 if a later checkpoint is
wanted instead.

Re-measure the held-out sample persistence NLL (must print 2.3947663):

```
OMP_NUM_THREADS=1 TORCH_NUM_THREADS=1 RAYON_NUM_THREADS=1 ./torch-env.sh \
    ./target/release/trading_bot_0 evaluate-timexer-segment \
    --checkpoint training/runs/timexer-control-4k/weights/best \
    --output training/runs/timexer-control-4k/gens/3 \
    --batch-size 256
```

Re-measure the population statistics in section 3 (CPU only, ~16 s warm):

```
./torch-env.sh cargo test -p trading_bot_0 --lib \
    the_real_corpus_calibration_partition -- --ignored --nocapture
```

## 8. Verification run

* `cargo check -p trading_bot_0 --tests`: **0 errors** for everything I own. The shared tree was
  red for most of this batch from `GainTwoSided`'s in-flight `calibration::Measured` /
  `calibration::Blocks` API change (E0063/E0432/E0609 at `calibration.rs`, `runner.rs:2417`,
  `reports.rs:2172/3112/3114`, `model.rs:2668`); no error was ever attributed to `corpus.rs` or
  to my `write_corpus` change.
* To get a green target while the shared tree was red I built in a throwaway git worktree
  (`/var/tmp/tb0-calpartition`, separate `CARGO_TARGET_DIR`) holding the full working tree plus
  local compile stubs for the four sites `GainTwoSided` had not yet updated. Those stubs exist
  only there and touch nothing in the repository.
* `cargo test -p trading_bot_0 --lib timexer_segment::corpus`: **9 passed, 0 failed, 1 ignored**,
  including both new tests and the pre-existing cache-rejection test.
* `the_real_corpus_calibration_partition_is_measured_against_a_published_contract`
  (`--ignored`): **passed**, 14.0 s, warm caches, 0 bar records rescanned. This is where the
  section 3 numbers and the section 4 real-corpus contract equality come from.
