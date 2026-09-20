# The held-out split guard: what it proves now, and what the old one could never prove

Scope: `trading_bots/src/torch/timexer_segment/{calibration,runner,corpus,reports}.rs`.
No GPU, no mlq, no training, no benchmark. Static reading plus two CPU-only tests.

## 1. The real invariant, with citations

The reserved bands are **not** per-ticker fractions of each ticker's own history, and they are
**not** a shared chronological cut either. They are per-ticker *ordinals* of a shared
*timestamp*, and that difference is the whole defect.

- The three split boundaries are three SHARED wall clocks: quantiles (70/80/90) of the union
  grid of distinct occupied five-minute slots — `corpus.rs:1000` `let bounds =
  quantile_bounds(&occupancy, span.first)?;`, `corpus.rs:2289` `let ranks = [count * 7 / 10,
  count * 8 / 10, count * 9 / 10];`.
- Each ticker then converts those three instants into **its own valid-bar ordinals** —
  `corpus.rs:1007` `.map(|file| bounds.map(|bound| file.index_at_or_after(bound) as u64))`,
  with `shared/src/bars.rs:488` `index_at_or_after` = "Index of the first record with `ts_ms >=
  ts_ms`, or `len()` if none".
- The bands are cut on those ordinals — `corpus.rs:1298-1311`:
  `let start = c.boundaries[first].max(common_context);` and
  `available = retained_partition_end(c.boundaries[first + 1], c.valid_bars, purge) - start`,
  origins at `start - 1 + i * pred_len`.
- `data.rs:155-161`: `retained_partition_end(boundary, source_bars, purge)` is
  `boundary - purge`, except `boundary == source_bars` → `source_bars` (a ticker whose history
  stops before the next boundary keeps its band to its last bar, unpurged).

**Per ticker the ordering is exact and guaranteed by construction**, in two cases:

- `boundaries[1] != valid_bars`: the calibration band's last target ordinal is
  `retained_partition_end(boundaries[1]) - 1 = boundaries[1] - purge - 1`, while the validation
  band's first origin ordinal is `max(boundaries[1], common_context) - 1 >= boundaries[1] - 1`.
  Separation is at least `purge = max(pred_len, 100) >= 100` bars (`data.rs:135`).
- `boundaries[1] == valid_bars` (the ticker stops trading before the shared 80% instant): the
  calibration band runs to its last bar, and its validation band is empty
  (`available = retained_partition_end(boundaries[2]) - valid_bars = 0`), so there is nothing
  to leak into.

**Across tickers the realized wall clocks are incomparable.** `boundaries[k] - 1` is "this
ticker's last bar strictly before the shared instant", which for a ticker that goes quiet — or
delists — is arbitrarily earlier than the instant itself. So one name is already being *scored*
at a date at which another name is still being *fitted*. Prior measurement on the real
4,873-ticker corpus, recorded at `probe.rs:411-421`: fit targets reach `1723739700000`
(2024-08-15) while the earliest scored origin is `1550178000000` (2019-02-14), *"Per ticker the
ordering is exact - 0 of 4,498 violate it - and not one of the 433,303 scored origins even
shares a TIMESTAMP with a fit origin"*.

I reproduced the mechanism from scratch on the construction path itself, on a two-ticker
synthetic corpus (`corpus.rs` test `the_reserved_bands_are_disjoint_per_ticker_and_interleave_globally`):

```
measured on the fixture: boundaries [8400, 9600, 10800] for FULL and [8400, 9000, 9400] for
GAPPY; the pooled calibration reach is 1502849400000 and the pooled first validation origin is
1502699700000, 149700000 ms EARLIER, while the tightest per-ticker separation is 30300000 ms
```

`GAPPY` goes quiet from slot 9,000 to slot 10,400, so the shared 80% instant (slot 9,600) falls
inside its hole, its `boundaries[1]` resolves to the first bar *after* the hole, and its first
validation origin is the bar before it — slot 8,999, i.e. 499 slots *before* `FULL`'s last
calibration target. Per ticker the separation is exactly 101 slots (`purge` 100, plus one).

**Verdict on the brief's diagnosis:** the conclusion is right (per-ticker blocks, globally
interleaved extrema, guard unsatisfiable on this corpus), the stated cause is not: the split
is not a per-ticker *fraction* draw. It is a shared-timestamp cut resolved into each ticker's
own ordinals, and the interleave comes from tickers whose bars are absent at the shared
instant. Both stories point at the same fix, so nothing downstream changes.

## 2. What the old guard did wrong

`calibration.rs` `Blocks::spanning` folds each population into global extrema and compares
`max(calibration target)` with `min(evaluation origin)` — at HEAD, `runner.rs:2041`
`Blocks::spanning(&fit_dated, &dated(&corpus.validation_refs))?`. On per-ticker bands those two
numbers belong to *different tickers*, so the comparison is not a statement about any shared
observation. It cannot pass on this corpus at any draw, and it did not fail because of a
192-bar horizon overhang: the two dates were five and a half years apart.

`Blocks::spanning` is kept, unchanged, and is still correct where it is used: `probe.rs:484`
and `probe.rs:510` **cut** the scored block to origins beginning after every fit target
completes and pay a measured 1% of the scored population for that global claim
(`probe.rs:423-429`). Its doc now says so, so nobody points it at the raw corpus bands again.

**Correction to the brief's timeline.** The refusal was not at the first evaluation and no
optimizer step ran. At HEAD the only `Blocks::spanning` call in `runner.rs` is line 2041, which
is 63 lines after `Corpus::load` and 106 lines before the `"from process start to the first
optimizer step"` print (HEAD `runner.rs:2147`); the first `engine.step` is HEAD
`runner.rs:2242`. The 26 minutes was the corpus rebuild plus startup, which the guard cannot
precede — but the run *did* pay, needlessly, for the contract write, the corpus report, the
model and optimizer build and all four held-out draws before being told.

## 3. The new formulation

`calibration.rs` `Blocks::per_ticker(calibration, evaluation, horizons, name)` over
`DatedOrigin = (ticker index, origin ms, last target ms)`:

- Folds a `BTreeMap` of each ticker's maximum calibration target and each ticker's minimum
  evaluation origin — every origin of both populations, no sampling. `BTreeMap`, not
  `HashMap`, so the ticker a refusal names is deterministic.
- Refuses if any ticker's own calibration targets reach its own evaluation origins, reporting
  the count of violators and the worst one by name with both dates in ISO **and** epoch ms and
  the horizon count, verbatim: `"{violations} of {shared} tickers fit and score a gain on their
  own data: {name} is the worst - its calibration block's last {horizons}-step target completes
  at {iso} ({ms} ms epoch) but its own first evaluation origin is at {iso} ({ms} ms epoch),
  {delta} ms EARLIER, so every one of those {horizons} horizons is fitted on bars that ticker
  is then scored on"`.
- Refuses a pair sharing no instrument at all (that is not a scored fit either).
- `purge_gap_ms` now carries the **smallest per-ticker separation**, which is the only
  separation this split claims; the descriptive extrema fields keep their meaning. The struct's
  shape and serialization are unchanged, so no manifest on disk moves.
  `reports.rs:2691`'s title text was corrected to say "tightest per-ticker separation from that
  ticker's own first scored origin" instead of "before the first scored origin".

The fit/score design is untouched: the gain is still fitted on the `[70%, 80%)` calibration
band, scored on validation, and the selection NLL is still un-gained.

## 4. Where it runs now

`runner.rs:2066` — `Blocks::per_ticker(&calibration_dated, &validation_dated, …)`, immediately
after `Corpus::load` (`runner.rs:2043`) and `corpus.prepare` (`runner.rs:2053`), and **before**
the contract write (2100), the corpus report (2104), the model build, every held-out draw, the
startup-total print (`runner.rs:2268`) and the first optimizer step (`runner.rs:2363`, inside
the step loop). It proves the FULL `calibration_refs` × `validation_refs` populations, not a
draw.

**No per-evaluation re-check.** `calibration_draw` is built once at `runner.rs:2145` and only
read afterwards (`runner.rs:2447`), and `validation_refs` is never reassigned inside `train`;
both evaluation-time populations are subsets of the two the startup proof covers, and a subset
cannot introduce an overlap the superset excludes. The second `Blocks::per_ticker` call at
`runner.rs:2156` exists to record the *draw's* own dates in the checkpoint manifest — it runs
the same proof over a subset of an already-proven population and therefore cannot refuse
anything the startup proof admitted.

## 5. The residual assumption — stated, not fixed

With per-ticker bands the calibration and validation populations **overlap in absolute
wall-clock time across different tickers**. No ticker shares a bar, an origin or an instant
with itself across the split, and on the real corpus not one scored origin shares even a
timestamp with a fit origin (`probe.rs:420`, measured) — but the fit period and the scoring
period are the same calendar span, so the gain is fitted with market-wide contemporaneous
information present. For `pred_len` per-horizon amplitude scalars this is a weak leak. It is a
real one, and it is now written into the code at
`calibration.rs` `Blocks::per_ticker`'s doc and at `runner.rs:2072-2078`, not hidden.

Quantified: the startup print reports the share of calibration targets falling inside the
validation origins' calendar span, computed on the run's own corpus
(`runner.rs:2079-2098`). I could not compute the real-corpus number statically — it needs a
corpus load, which this ticket forbids. On the real corpus the prior probe measurement bounds
the sharper question (exact-instant coincidence) at **zero of 433,303 scored origins**, so the
residual is span overlap between different tickers, not instant overlap.

Closing it would mean a global wall-clock cut of the scored block — exactly what
`probe::Partitions::split` does, at a measured 1% of its scored population. That is a different
experiment and was not attempted here.

## 6. Evidence

- `./torch-env.sh cargo check -p trading_bot_0 --tests` → 0 errors
  (`Finished dev profile … in 10.83s`; only pre-existing warnings, none in the touched files).
- `cargo test -p trading_bot_0 --lib per_ticker_blocks_that_interleave_globally_are_proven_rather_than_refused`
  → 1 passed. Two synthetic tickers of different history length whose per-ticker blocks are
  individually disjoint while their global extrema interleave: asserts the pooled guard refuses
  it ("share bars"), the per-ticker guard admits it with `purge_gap_ms` equal to the tighter of
  the two names' own separations, and that a ticker whose targets reach its own scored block is
  refused **by name** with both dates and the horizon count. No CUDA, no corpus.
- `cargo test -p trading_bot_0 --lib the_reserved_bands_are_disjoint_per_ticker_and_interleave_globally`
  → 1 passed, output quoted in §1. Small synthetic *fixture* corpus (two files, 12,000 and
  10,600 bars, temp dir); it exercises the real `Corpus::load` band arithmetic. CPU only.

## 7. What I did not verify

- Nothing was run on the real 28 GB corpus. The real-corpus numbers in §1 are the prior
  measurement recorded at `probe.rs:411-421`, not a fresh run.
- No training, evaluation or benchmark was executed, so the startup print's exact text and its
  residual percentage have not been observed on a real run — only that they compile and that
  every value they read is produced by the proof.
- The cost of the proof on the real corpus (two dated passes over ~433k + ~434k refs, i.e.
  ~1.7M mapped bar-header reads) was not measured; the print reports it in ms so the first real
  run will state it. The previous code already dated the full validation population at the same
  place in startup, so the added cost is one pass over `calibration_refs`.
- No broad test suite was run: scoped `cargo check --tests` and the two narrow filters above
  only.
