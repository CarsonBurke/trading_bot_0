# The in-period held-out draw: separating overfitting from non-stationarity

Every claim tagged **DEMONSTRATED** (read in source at the cited `file:line`, or measured here) or
**HYPOTHESIS** (mechanism / forecast, never a measured result). No GPU job was submitted; the
queue is Main's.

Citation aliases: `C` = `trading_bots/src/torch/timexer_segment/corpus.rs`, `M` =
`.../model.rs`, `R` = `.../runner.rs`, `P` = `.../reports.rs`, `D` = `.../data.rs`,
`S` = `shared/src/report.rs`.

---

## 0. The one-paragraph result

A chronological split makes `held-out sample`, `held-out cross-section` and `held-out full`
simultaneously out-of-sample in **origin identity** and out-of-period in **market regime**. Those
two mechanisms have opposite fixes, so no existing panel can tell them apart. This report adds an
**in-period** draw: origins cut from a purged hole *inside* the training span, sharing no origin
and no target bar with any surviving training row, scored at every report interval beside the
out-of-period draw and emitted on one new base `timexer_segment_temporal_generalization`.

While building it, the effective-sample-size arithmetic turned out to overturn the framing of the
step-2000 peak, and that is the more important finding. **"0.417 of one epoch, no sample
repeated" is an artifact of counting ROWS.** Training supervises all 375 causal sub-origins of
every row (`M:1645-1648`, `last_only = false`), so one row supervises ~6,000 bars while
consecutive rows advance 192. Each distinct supervised outcome is therefore seen ~28.4 times per
epoch, and by step 2000 the model has taken **5.92 gradient passes over 98.49% of every distinct
supervised outcome in the entire training span**. Overfitting at step 2000 is not surprising; it
is the expected outcome.

A third finding arrived by way of a failed job and is the one with the widest blast radius:
**an ordinal offset from a shared boundary is not a shared moment.** The first version of this
hole was offset from `boundaries[0]` in ordinal space, which spread its first origin over
**1,914 days** across tickers and made the cross-sectional draw impossible (job 5460). The same
defect is live in the existing `band` closure that builds `validation_refs` and
`calibration_refs`: measured, **99.75% of the 433,303 validation origins form no usable
cross-section**, which is a 4.7× precision loss on every trading number this project has
published. §6.1 audits the whole class site by site.

---

## 1. What the training sampler actually supervises (DEMONSTRATED)

| Fact | Evidence |
|---|---|
| Training origins are a deterministic per-ticker stride `common_context - 1 + j·pred_len`, `j = 0..`, bounded by `train_end - 1`. The epoch shuffle reorders this list; it never changes the SET. | `C:829-847`, shuffle at `R:1309` |
| `train_end = retained_partition_end(boundaries[0], valid_bars, purge)`, i.e. the global 70% timestamp's own valid-bar ordinal minus `purge = max(pred_len, 100)`. | `D:135-136`, `D:155-161` |
| The head runs on EVERY causal sub-origin during training. `future_windows` with `last_only = false` is `narrow(1, patch_len, seq_len + pred_len - patch_len).unfold(1, pred_len, patch_len)`: 375 windows at stride `patch_len = 16`. | `M:1638-1648`, `M:1652` |
| Evaluation runs on the FINAL origin only (`final_origin(model, batch)`), so a scored origin's target bars are exactly `[origin+1, origin+pred_len]`. | `R:815` ("Final-origin held-out scores"), `R:1611` |
| Sub-origins below `min_history = 256` valid context bars are masked out. Sub-origin `o` holds `(o+1)·16` bars, so `o >= 15`; 360 of 375 sub-origins are live. | `M:1624-1626` (`bars.ge(min_history)`) |
| Therefore one training row with final origin `O` supervises absolute target bars `[O - 5743, O + min(192, train_end - O - 1)]`. Independently confirmed by `PeakCause` over 168 arithmetic cases. | derived from the four rows above |
| Every supervised absolute origin is `≡ 15 (mod 16)`: `common_context - 1 = 5999 ≡ 15`, the row stride `192 = 12·16`, the sub-origin stride is 16, and `5744 = 16·359`. | same |
| Interior multiplicity is exactly `floor(5744/192) + 1 = 30`. Corpus mean multiplicity is **28.3673** (edge ramps 1..29), measured by `PeakCause` against the real manifest. | measured (`PeakCause`) |

### 1.1 Why this kills origin-identity-only holdout (DEMONSTRATED, and it is algebra)

Because supervised origins are exactly the `≡ 15 (mod 16)` lattice, 15/16 of all ordinals are
origin-disjoint from training *for free*. That looks like a cheap in-period draw. It is worthless:

for a non-lattice origin `A` with `r = (A - 15) mod 16 ∈ 1..15`,

$$\mathrm{cum}(A \to A+h) \;=\; \mathrm{cum}(A-r \to A+h) \;-\; \mathrm{cum}(A-r \to A)$$

and **both** right-hand terms are supervised, from lattice origin `A - r` at horizons `h + r` and
`r`, each ~30 times per epoch. The held-out label is an exact linear combination of trained
labels. Contamination is not partial and not quantifiable-away; it is total and algebraic. A
chart labelled "contaminated" would be uninterpretable rather than merely caveated. This is why
the fallback of "keep origin-identity disjointness, measure target-bar overlap as a contamination
statistic" was proposed by Main, argued against here, and withdrawn by Main.

### 1.2 Note for `OrthoTargets` (HYPOTHESIS, from the same algebra)

The identity above is also why the 192-horizon objective is degenerate. From one lattice origin
the 192 cumulative labels are nested partial sums of exactly 192 single-bar increments, so they
carry 192 scalars of information in 192 head rows — but the head rows are *near-duplicates*, not
merely correlated: row `h+1` differs from row `h` by one increment. This is independent evidence
for the orthonormal-coefficient reparametrization, arrived at from a different direction (a
holdout-construction proof, not a search for a reason to reparametrize).

---

## 2. The disjointness construction

### 2.0 The construction this replaces, and what it cost (DEMONSTRATED, job 5460)

The first version of this experiment anchored the hole on an ordinal OFFSET from
`boundaries[0]`. It shipped, and job 5460 refused at 30 s with
`no held-out evaluation timestamp holds the 40 tickers a cross-sectional trading measurement
needs at all`. The defect, in one line: **`boundaries[0]` is a shared timestamp, but
`boundaries[0] - k` is not.** Two tickers with different bar densities over an 18,479-bar
backoff land at different moments.

MEASURED directly off the bar files at the published contract's geometry (`numpy` over the
64-byte header + 36-byte records of `long_data/bars`, valid ordinals reconstructed from each
ticker's `invalid_ohlc_indices`):

| Quantity | Ordinal-anchored (job 5460) | Wall-clock anchored (landed) |
|---|---|---|
| spread of the FIRST in-period origin across holed tickers | **1,914 days** | 0 (one timestamp by construction) |
| distinct in-period timestamps | 36,884 | **32** |
| widest timestamp block | **26 tickers** | **2,987 tickers** |
| timestamps clearing `CROSS_SECTION_FLOOR = 40` | **0** — hard refusal | **32 of 32** |

The same script measured `validation_refs` in the same pass: 433,114 origins, widest block
3,690, exactly 100 timestamps clearing 40 — reproducing the control's draw from the outside,
which is what proves the byte-identity claim of §2.4 rather than asserting it.

### 2.1 Geometry (`C:60-232`, `C:381-465`, `C:1095-1102`)

The hole is placed on **shared wall clocks**, one per section. Per ticker, `place`
(`C:427-465`) turns those anchors into an inclusive target-bar range and an origin list.

```
bracket = IN_PERIOD_BRACKET_ROWS · pred_len   = 64 · 192 = 12,288
above   = (context - 2) + bracket             = 5,998 + 12,288 = 18,286
span    = sections · pred_len                 = 32 · 192 = 6,144

admissible first-anchor ORDINALS, per ticker:
    first = common_context + bracket                                   = 18,288
    last  = boundaries[0] - (purge + 1 + above + span)
admissible first-anchor WALL CLOCKS: [ts(first), ts(last)]             (two probes per ticker)

H       = the LATEST candidate endpoint covered by >= IN_PERIOD_MIN_UNIVERSE tickers
ref     = the admissible ticker with the most valid bars, ties by symbol
anchors = [ ts_ref(ordinal_ref(H) + j·pred_len) : j = 0..sections )

per ticker: A_j = valid_ordinal_at_or_before(anchors[j])   (one binary search + a forward walk)
admitted iff  the A_j strictly increase
          and A_0 + 1 >= common_context + bracket
          and hi + purge + above < boundaries[0]
hole    = [A_0 + 1, A_{sections-1} + pred_len]
origins = { A_j : ts(A_j) == anchors[j] }      -- EXACT matches only
```

Six decisions, each load-bearing:

1. **Anchors are wall clocks, not ordinals.** This is §2.0. An ordinal offset from a shared
   boundary is safe iff nothing ever groups across tickers on the resulting position, and
   unsafe the moment two tickers' positions are pooled — which is precisely what a
   cross-sectional IC does. `boundaries[0] - k` is used here **only** as an admissibility
   bound (a per-ticker room test), never as a position (`C:118`, `C:464`).
2. **`H` is the LATEST wall clock covered by 4× the draw's per-timestamp cap** (1,024 tickers,
   `IN_PERIOD_MIN_UNIVERSE`). Coverage is flat over an 18-month plateau: strict argmax
   coverage is 3,449 tickers at 2020-10-15, the chosen rule takes 2,995 at 2022-06-09.
   MEASURED trade-off — 13% of the universe (still 11.7× the 256-name cap, so the draw is
   never width-limited) buys **20 months of recency**, which is what shrinks the regime gap to
   the out-of-period comparand. The floor is tied to a real constant, not a tolerance on a
   maximum.
3. **Spacing comes from the reference clock.** On this corpus "the admissible ticker with the
   most valid bars" selects **SPY** (468,054 valid bars), so consecutive anchors are `pred_len`
   SPY bars apart — the market's own clock. A calendar constant would have to guess how many
   bars a session holds (SPY trades 192 extended-hours bars/day, a regular-hours name 78) and
   would break on half days.
4. **Exact timestamp matches only.** A ticker that did not trade at an anchor contributes no
   origin there rather than a nearest-bar origin at a different moment, which would be the
   original bug in miniature.
5. **The exclusion bound is model-configuration-independent.** A row spans
   `[O - context + 1, O + pred_len]`, every sub-origin is inside the context, every target is
   ≥1 bar ahead of its own sub-origin, so no supervised bar can lie outside
   `[O - context + 2, O + pred_len]` at ANY `patch_len` / `min_history`. True reach
   `[O - 5743, O + 192]`; conservative bound `[O - 5998, O + 192]`, costing 255 extra bars
   ≈ 1.3 rows/ticker. That price buys a guarantee a change to the patch grid, the history
   floor, or a per-row origin phase shift cannot silently break.
6. **Bracketed by 64 trained rows on each side** = 12,288 bars ≈ 158 sessions of supervised
   history immediately before and after the hole. A hole with trained data on only one side is
   an extrapolation wearing an in-period label; the test asserts both brackets.

A load-time refusal replaces the draw-time one: if fewer than `sections/4` anchors are
saturated, `in_period_plan` retries up to `IN_PERIOD_ANCHOR_ATTEMPTS = 8` candidate wall clocks
and then fails with the per-anchor census in the message, naming the cause as a **time-of-day**
failure rather than a population one (`C:211-219`). On the real corpus it accepts at **attempt
1 of 8**.

### 2.2 The purge, exactly (`C:205-236`)

`CorpusTicker::supervision_clears_hole(origin)` is the single definition of the guarantee, called
by the enumeration (`C:838`) and by every consumer that re-selects rows:

```rust
origin + pred_len < lo  ||  (origin + 2).saturating_sub(context) > hi
```

so the excluded origin range is `[lo - pred_len, hi + context - 2]`, an ordinal span of
`sections·pred_len + pred_len + context - 3` = `sections·192 + 6,189`. `DataDiversity`'s
`--patch-phase random` calls the same predicate with the *phased* origin, so a phase shift that
pushes a row's reach into the hole drops that row instead of leaking.

### 2.3 What it cost in population (DEMONSTRATED — printed by the real load, not derived)

Every number below is read off the corpus load itself at the production geometry
(`the_real_corpus_in_period_draw_forms_usable_cross_sections`, release, warm cache):

```
CausalPatch in-period hole: 32 anchors on the SPY clock from 1654782000000 to 1658852400000,
  held by 2995 of 4873 tickers, 32 saturated cross-sections,
  per-anchor widths [2933, 2955, 2974, ... 2948, 2945], placed at attempt 1 of 8 in 8168.3 ms
CausalPatch in-period held-out population: 94369 origins over 32 sections per eligible ticker
  covering 18118848 target bars inside the TRAINING span, bracketed by 64 trained rows on each
  side; the purge ... removed 141762 training rows owning 27218304 target bars (5.78% ...)
CausalPatch held-out populations: 433721 calibration origins ... 433303 validation origins
  ... with 408842 unused remainder
```

| Quantity | Control | `--in-period-sections 32` | Δ |
|---|---|---|---|
| training rows | 2,455,276 | **2,313,514** | **−5.78%** (141,762 rows) |
| train-owned target bars | 470,946,393 | 443,728,089 | −5.78% (27,218,304 bars) |
| steps per epoch @ batch 256 | 9,590 | **9,037** | −5.78% |
| hole, in wall clock | — | **2022-06-09 → 2022-07-26** (32 SPY sessions) | — |
| tickers carrying a hole | — | **2,995 of 4,873** | — |
| in-period origins | 0 | **94,369** (31.5 of 32 anchors per eligible ticker) | — |
| in-period cross-sections | 0 | **32 timestamps, widths 2,869–2,987** | — |
| in-period draw the trading family selects | 0 | **32 × 256 = 8,192 windows**, IC SE ≈ **.0111** | — |
| out-of-period draw, same code | 100 × 40 = 4,000 windows, SE ≈ **.0164** | unchanged | 0 |
| `held-out sample` / `cross-section` / `full` draws | unchanged | **unchanged, byte-identical** | 0 |
| anchor placement, one-time startup | 12.7 ms (`per-ticker contracts`) | **8,168 ms** | +8.2 s once |

**Three corrections to earlier drafts of this report, all of which moved against the
convenience of its own conclusion and are recorded because the direction matters:**

1. The row price is **5.78%, not 10.66%**. A wall-clock hole is NARROWER in bars for a
   regular-hours ticker (median hole width 2,716 bars against the ordinal rule's 6,144) because
   32 SPY sessions hold 32·192 SPY bars but only ~32·78 bars of a regular-hours name, and fewer
   tickers qualify. The 12.9% total price quoted earlier is now ~8%.
2. The hole is **2022-06, 11 months more recent** than the ordinal rule implied, so the regime
   gap to the 2024-08 validation start is smaller than pre-registered. That strengthens §4's
   confound discussion rather than weakening it.
3. **The "5.3% buys SE parity" half of the earlier cost breakdown is RETIRED.** Parity is now
   bought by the anchoring, not by population: the in-period draw reaches 8,192 windows at
   SE ≈ .0111 against the existing draw's 4,000 at ≈ .0164, so the in-period draw is the *more*
   precise of the two and `SE_diff = sqrt(.0111² + .0164²) = .0198` is dominated by the
   EXISTING draw. That is the safe direction for the ratio reading of outcome 3: the comparison
   cannot be undersold by the new draw's own noise.

Of the 5.78%, the irreducible part is the **6,189-bar exclusion span** (`ROW REACH`, paid by a
hole of any size); the marginal cost of the 32 sections themselves is now small, because the
hole spans 32 sessions rather than 79.

The one alternative that is genuinely cheaper is a **ticker holdout** (hold `K` tickers out of
training entirely: cost `K/4873`, ~0.8% at `K = 40`). The model has no ticker-identity input at
all — inputs are σ-normalized log prices, market-relative covariates and calendar features — so
it cannot memorize a ticker, which makes this less biased than it looks. It is not implemented
because (a) it varies ticker identity as well as origin identity, so it answers a third
question, and (b) `market_cum` is built from all eligible tickers including the held-out ones
(`C:651`, `market_min_cross_section = 2000`), a real if small leak. Recorded as the cheap
follow-up if the 5.78% perturbation proves objectionable.

### 2.4 Proof, not assertion

`C:2626+` `in_period_origins_share_no_origin_and_no_target_bar_with_any_training_row`
(**DEMONSTRATED — passes**). It loads the same **four-ticker** fixture twice, at `sections = 0`
and `32`, and asserts six claims:

1. `validation_refs` and `calibration_refs` are **equal between the two loads**, plus
   `validation_target_bars` and `validation_remainder_bars`. Since `held-out sample` is
   `fixed_origins(&validation_refs, ..)` (`R:1735`) and `held-out cross-section` is
   `cross_section_origins(&corpus, &corpus.validation_refs)` (`R:1741`), an unchanged
   `validation_refs` list IS an unchanged draw — and therefore an unchanged persistence NLL.

   The anchor is confirmed **from the artifact, not by derivation** (DEMONSTRATED): the
   `held-out sample persistence NLL` series in
   `training/runs/timexer-invsqrt-ic-6k/gens/1/timexer_segment_loss.report.bin` reads
   `2.394766330718994` at all five of its report points, which is the stated 2.3947663 to every
   digit fp32 carries. The same title records the cross-section draw as `4000 fixed windows in
   whole timestamp` blocks, confirming the ~40 names × ~100 timestamps figure the SE of .011 was
   quoted against. The anchor is a function of that draw and the corpus bytes only; the hole
   touches training rows exclusively.
2. The census is exact: `kept + purged == control rows` and
   `train_target_bars + purged_target_bars == control train_target_bars`, so the reported cost is
   not an estimate. (With the hole off, the per-row `min(pred_len, train_end - origin - 1)` sum
   telescopes to `train_end - common_context` bar for bar, which is why `train_target_bars` is
   unchanged to the byte on control runs.)
3. **Disjointness, bar by bar.** It materializes the whole conservative supervised bar set
   `[O - context + 2, O + owned]` over every surviving training row, then asserts for every
   in-period origin that it is not any training row's origin, that all `pred_len` of its target
   bars are inside `[lo, hi]`, that none is in the supervised set, that it owns a COMPLETE
   horizon (`target_count == Some(pred_len)`), and that it lies inside the training span
   (`origin < train_end - 1`). Proving disjointness from the conservative superset implies
   disjointness from the true supervised set.
4. **The origins are SHARED WALL CLOCKS.** Every in-period origin's timestamp is one of the
   published `in_period_anchors`; the per-anchor width vector reconstructed from the refs
   equals the published `in_period_census`; every anchor is held by ≥2 tickers, so a
   cross-section exists there; and at least one ticker holds all 32.

   This claim has **teeth**, asserted rather than assumed: the fixture carries a fourth ticker
   `DDD` that skips every third slot over its first 3,000 bars, so from bar 3,000 onward its
   ordinal for any given moment runs ~1,000 behind the dense tickers' while its TIMESTAMPS
   still coincide exactly. The test then asserts that the tickers' ordinals for anchor 0 span
   more than `pred_len` — i.e. that the fixture really is misaligned — so a future edit cannot
   quietly make this claim vacuous, and the ordinal construction of §2.0 cannot pass it.
5. Both brackets are populated with ≥64 trained rows.
6. The predicate is correct for **every** phase shift `1..pred_len` of every surviving row,
   which is the contract `DataDiversity` composes against.

The real corpus is covered separately by `the_real_corpus_in_period_draw_forms_usable_cross_sections`
(`#[ignore]`, 19 s release), which histograms BOTH draws in one pass and asserts the in-period
one offers usable cross-sections. It is the fixture form of job 5460 at full scale: it fails on
the old construction by 0 timestamps clearing the floor, and passes on the new one with 32.

Checkpoint compatibility is proven by two **pre-existing** tests that needed no modification:
`C:2302` asserts the exact sorted serialized key set of `CorpusContract` (18 keys, none of mine)
with the message "adding a field to it would invalidate every authenticated checkpoint manifest
on disk", and `the_real_corpus_calibration_partition_is_measured_against_a_published_contract`
(`C:2491`) reloads the real corpus and asserts `corpus.contract == published` against
`training/runs/timexer-control-4k/timexer-segment-data-contract.json`. **Both pass unmodified,
re-run after this change** (`2 passed`). The seven new fields carry `#[serde(default,
skip_serializing_if = ...)]` — `is_zero` for the five counts, `Vec::is_empty` for
`in_period_anchors` and `in_period_census` — so with the feature off they are absent from the
JSON, an old manifest deserializes them as defaults, and `manifest_sha256` over `data` is
byte-identical. `SCHEMA` is deliberately NOT bumped: bar bytes, boundaries and the market grid
are untouched, so no byte cache is invalidated and no cold 172 s rescan is forced. Only row
enumeration changes, and nothing caches that.

---

## 3. The new base, and its cost

`timexer_segment_temporal_generalization` (`S:183-199`, written at `P:1177-1332`), unit
`IC_UNIT`, `ScaleKind::Linear`. Per horizon in `[1, 8, 16, 32, 64, 128, 192]`
(`DECISION_HORIZONS`, `P:446`), six series plus a zero line:

- `in-period held-out cross-sectional IC at horizon {h}`
- `in-period held-out cross-sectional IC standard error at horizon {h}`
- `out-of-period held-out cross-sectional IC at horizon {h}`
- `out-of-period held-out cross-sectional IC standard error at horizon {h}`
- `in-period minus out-of-period cross-sectional IC at horizon {h}`
- `in-period minus out-of-period cross-sectional IC standard error at horizon {h}`

No label contains `=`, because `report_cli --var` splits a rendered token on its first `=` and a
series carrying one cannot be selected on the command line. Unmeasured renders NaN and `chart`
drops an all-NaN series rather than drawing a zero line (`P:816-822`), so a control run emits no
file at all rather than a panel of zeros.

The difference's SE is `hypot(se_in, se_out)`, which **assumes the two ICs are independent**. The
two draws are disjoint bars in disjoint periods separated by the purge and the whole `[70%, 80%)`
calibration partition, so the assumption is that no common factor spans that gap. It is an
assumption, not a fact, and it is why both component SEs are charted beside the combined one: a
reader who rejects independence can still bound the difference by the larger component.

The out-of-period comparand is the **existing** `held-out cross-section` pass — already computed,
zero extra cost. The in-period pass is built by the same `cross_section_origins` rule off
`corpus.in_period_refs` (`R:1742-1755`) and scored by the same `score()` at every interval
(`R:1934-1945`), so the two are read at the same optimizer step. Reading them at different steps
would confound the trajectory the whole comparison is about.

### Added ms per interval

A single per-origin rate is WRONG here, and using one is the error a first pass makes. Two
measured points exist: the preview pass is 1,123 ms / 2,048 origins, and `R:2112` records the
final full pass as **103 s / 433,303 origins**. Those imply very different naive rates (0.548 vs
0.238 ms/origin), so the cost is affine and both parameters must be solved for:

```
b = (103,000 - 1,123) / (433,303 - 2,048) = 0.2362 ms per origin
a = 1,123 - 2,048 b                       = 639 ms fixed per pass
```

| | value | source |
|---|---|---|
| preview pass, 2,048 origins | 1,123 ms | supplied measurement |
| final full pass, 433,303 origins | 103,000 ms | `R:2112` (measured) |
| marginal / fixed cost | 0.2362 ms per origin / 639 ms per pass | derived from the two above |
| cross-section pass, 4,000 origins | ≈ 1,584 ms | derived |
| in-period pass, 8,192 origins | **≈ 2,574 ms** (HYPOTHESIS — forecast) | derived |
| **share of a 1,000-step interval** (216,000 ms) | **1.2%**, ceiling 15% | derived |
| **share of a 250-step interval** (53,900 ms) | **4.8%**, ceiling 15% | derived |

The earlier figure in this report was 4,491 ms, from applying the preview's naive 0.548 ms/origin
rate to 8,192 origins. That rate is 57% fixed overhead at 2,048 origins, so it over-charges any
larger pass; the affine model is the honest one and the in-period pass is **1.7x cheaper** than
first stated.

**The whole-arm budget, with the one-time cost now MEASURED.** The anchor placement is
8,168 ms of startup, paid once, and it is the only new fixed cost (the control path is
12.7 ms, so this is not a regression on unholed runs — it is a cost that exists only when the
feature is on). At 2,500 steps and `--eval-every 250`:

| Phase | ms | source |
|---|---|---|
| corpus load, warm cache, control path | 11,360 | measured |
| anchor placement | 8,168 | measured |
| CUDA context, model, captured step, draws | ~15,000 | prior arms |
| stepping, 2,500 × 216 ms | 540,000 | prior arms |
| 10 interval evaluations × (1,123 sample + 1,584 cross-section + 2,574 in-period) | 52,810 | derived |
| final full-split pass | 103,000 | `R:2112` measured |
| **total** | **≈ 730,000 = 12.2 min** | under the granted 15 min |

So the granted 15 m holds with ~2.8 min of slack and **no step reduction is needed**. The
in-period pass is 4.8% of a 250-step interval against the 15% ceiling. `EvalTiming::in_period_ms`
is wired onto `timexer_segment_timing`, so the 2,574 ms forecast is falsified or confirmed by
the arm itself rather than argued.

**The full pass is the real budget item, and it was missed.** A capped mid-epoch run pays the
103 s full-split pass unconditionally (`R:2120`, `ended.is_some() && !epoch_complete`; there is no
flag). Neither this report's first draft nor the ticket's "216 ms/step + 30 s startup ⇒ ~2,500
steps" arithmetic costed it. At `--eval-every 250` and 2,500 steps the total is
`539 s stepping + 30 s startup + 10 x ~6.3 s + 103 s` ≈ **735 s = 12.3 min**, over the
10-minute wall; solving `steps · (0.2157 + 6.3/250) ≤ 600 - 30 - 103` gives **steps ≤ ~1,938**.
So the arm needs either `--time-limit 15m` at 2,500 steps (recommended — the clock claim needs an
observation *after* the predicted peak) or `--max-steps 1900` inside 10 minutes, which can locate
the peak but not measure the depth of the decline the mechanism claim's ratio needs.

This is a forecast, not a measurement, so it is measured by the harness rather than trusted: a
new `EvalTiming::in_period_ms` field (`P:289-294`) carries the pass's own wall clock and appears
as `in-period held-out pass total` on `timexer_segment_timing` (`P:1214-1216`). The added ms per
interval is read off a chart on the first run.

---

## 4. Pre-registered reading rule

Written into `P:1177-1201` and `S:183-199` **before any run of this arm exists**, so it cannot be
chosen after the curves do.

| Observation at steps 1000 / 2000 / 2500 | Mechanism | Lever |
|---|---|---|
| in-period IC keeps improving past the out-of-period peak while out-of-period IC collapses | **NON-STATIONARITY** | data recency, target definition, online adaptation. **NOT** regularization — regularizing a model that fits its own period correctly would only remove skill. |
| BOTH collapse together | **genuine overfitting on a correlated-sample budget** | capacity, regularization, effective sample size |
| in-period ALSO peaks at ~2000 but less severely | undecided — report the **ratio** of the two declines from the difference series, pick no side | — |

Two disciplines on the reading:

- **The collapse is horizon-selective** (h=1 .1575→.1440 nearly flat; h=16 .1137→.0086; h=64
  .0879→−.0125). Multiplicity is flat in `h` — the lattice and the reach do not depend on the
  horizon — so repetition alone explains a collapse but not its *shape*. If both draws collapse
  together, the in-period/out-of-period split is confirmed as *not* the discriminator for the
  shape, and the horizon term structure is a separate question.
- **Draw precision, MEASURED on the real corpus.** The existing out-of-period cross-section
  draw is 100 timestamps × 40 names = 4,000 windows, IC SE ≈ **.0164**; the in-period draw is
  32 timestamps × 256 names = 8,192 windows, SE ≈ **.0111**. Difference SE = `hypot` =
  **.0198**, DOMINATED by the existing draw — so the new draw cannot undersell the comparison
  with its own noise, and the observed ~.10 collapse is a ~5σ effect. A single .01 move remains
  noise in either draw.
- **Do not put a holed arm's training NLL beside a control's.** It trains on **94.22%** of the
  rows. This is enforced rather than requested: `Metrics::in_period_purged_row_share` is folded
  into the scope line of EVERY chart `write_metrics` writes, `timexer_segment_loss` included, so
  each title carries the purged share and the sentence that training losses are NOT
  step-comparable to an unholed arm. A fact that has to be remembered is a fact that will be
  forgotten. The load-time census prints the same sentence, and `train_target_bars` in a holed
  manifest is already net of the purge.

### My pre-registered prediction

**Outcome 1: NON-STATIONARITY. In-period IC will still be rising at step 2500 at h ∈ {16, 32,
64} while out-of-period IC collapses; the difference series at h=64 will exceed +0.08 by step
2500 at >4σ.** (HYPOTHESIS.) Reasoning, stated so it can be wrong for a legible reason:

1. Long-horizon predictability is real and strongly measured at a *fixed* checkpoint — within-
   timestamp IC .0655/.0610/.0509 at h=64/128/192 on the step-3000 `timexer-control-4k`
   checkpoint, 18–23σ. A model that had destroyed its own long-horizon function by overfitting
   would not hold that.
2. h=64 goes *negative* (−.0125). Overfitting drives skill toward 0 (the model fits noise that
   does not recur out of sample); a systematically **inverted** rank ordering is the signature of
   a relationship the model learned correctly on one period and that reversed in another. Sign
   inversion is a non-stationarity fingerprint, not an overfitting one.
3. The horizon selectivity matches a drift/regime story: h=1 is dominated by
   microstructure-scale effects that are stable across periods, and the h≥16 band is where
   period-specific drift and factor rotation live. Multiplicity is horizon-flat and cannot
   produce that gradient.
4. Amplitude is already excluded as the mechanism: `β̂ = .26594` at h=192 and oracle per-horizon
   rescaling recovers only 1.0212 → .99695, and a positive per-horizon gain cannot change rank
   order, so the IC collapse is not calibration.

**Second, SEPARATE prediction from the same run (HYPOTHESIS).** This is a different claim from
the one above and must not be presented as one result with it: the in-period-versus-out-of-period
ratio tests the MECHANISM, this tests the CLOCK. Re-derived on the MEASURED 5.78% price (the
earlier 10.66% figure, and the ~1,787-step prediction that came from it, are retracted): the
holed arm's epoch is `2,313,514 / 256` = **9,037 steps**, so step 2000 sits at 0.2213 of its
epoch versus 0.2085 of the control's. If multiplicity drives the peak, the holed arm's
*out-of-period* peak should move earlier, to `2000 × 2,313,514/2,455,276` ≈ **1,885 steps**,
i.e. inside `[1870, 1900]` at `--eval-every 250` resolution — which is now only ONE report
interval earlier than the control's, so this handle is **weaker than first advertised** and
should be read as corroboration of `DataDiversity`'s two much larger handles rather than as an
independent test on its own. Stating that here rather than after the run, because a 6%
perturbation predicting a 5% shift is at the edge of what a 250-step grid can resolve.

What makes this handle worth more than its cost: the hole changes multiplicity **without changing
support**. It does not thin the origin lattice, alter the tokenization, or touch the patch grid —
it only runs out of pool sooner. Together with `DataDiversity`'s `K=4` (~1,457) and `K=30` (~319)
arms there are then three independent handles on the occupancy hypothesis, which predicts the
peak step tracks all three monotonically; two-of-three tracking is far more diagnostic than any
single arm.

Note the cost of being right about prediction 1: this arm would then both disambiguate the
mechanism AND confirm it, from one run. Keep the two readings in separate paragraphs of whatever
is written up.

### 4.1 Independent corroboration of the clock (DEMONSTRATED by `DataDiversity` / `PeakCause`)

`DataDiversity` has since MEASURED the occupancy census on the real corpus and it reproduces
`PeakCause`'s independent arithmetic to six digits: 31,159,116 distinct `h=1` outcomes, 27,783,348
of them at `M=30` (share 0.89166), mean `M` 28.36728, global one-pass coverage at step 2000
0.984870, interior 0.999103. Their sharpened reading is stronger than mine and I adopt it: the
control's IC peak at ~2000 coincides with **INTERIOR** saturation at step **1972.4**, not with
global saturation, which is step 2,646 for 0.99 and 7,597 for 0.999. That is a much tighter
coincidence than "0.21 epochs" suggested and it is the reason §5 is the primary deliverable.

It also sharpens my clock prediction two independent ways, which agree — re-derived on the
MEASURED 5.78% row price, superseding the 10.66% figure and the `[1,760, 1,790]` window an
earlier draft of this report published:

- from the row-count ratio: `2000 × 2,313,514/2,455,276` = **1,885**;
- from scaling their measured interior-saturation step by the same ratio:
  `1,972.4 × 2,313,514/2,455,276` = **1,859**.

So the holed arm's out-of-period peak is predicted in **[1,855, 1,890]**. At `--eval-every 250`
that is one report interval earlier than the control's, so the handle survives but is WEAK —
read it as corroboration of `DataDiversity`'s far larger `K=4` and `K=30` perturbations, not as
a test in its own right. A peak that stays at 2000 falsifies the clock claim without touching
the mechanism claim, which is exactly the separation Main required.

Falsifier for the whole report: if both ICs collapse together at h=64 with the difference series
inside ±2σ of zero at every interval, prediction 1 is wrong and the lever is capacity /
regularization / effective sample size, exactly as §5 would then imply.

---

## 5. Effective sample size (the primary deliverable)

Constants used, with provenance. **Corrected:** `470,946,393` is **train-owned target bars**, not
corpus size; actual valid bars are **752,901,364** (`PeakCause`). The reconciliation that pins it:
`2,455,276 rows × 192 = 471,412,992` and `4000 × 256 × 192 / 470,946,393 = 0.41751`, which is the
reported `training target bars completed = 0.417` exactly. `train_target_bars` was never a
coverage statistic and no chart may imply it is.

| Symbol | Value | Source |
|---|---|---|
| training rows / epoch | 2,455,276 (9,590 steps × 256) | supplied |
| train-owned target bars | 470,946,393 | supplied, reconciled above |
| valid bars | 752,901,364 | measured (`PeakCause`) |
| context / patch / horizon / min_history | 6,000 / 16 / 192 / 256 | `M:537`, `M:92-101` |
| live sub-origins per row | 360 of 375 | `M:1624-1626` |
| interior multiplicity `M` | 30 | derived, §1 |
| corpus mean multiplicity | 28.3673 | measured (`PeakCause`) |
| distinct `(origin, h=1)` pairs | 31,159,116, of which 89.166% at `M=30` | measured (`PeakCause`) |
| parameters | 27,232,883 | supplied |

### 5.1 Target-window overlap (assumptions: interior of one ticker, all context bars valid)

Supervised origins sit on a stride-16 lattice with 192-bar target windows, so:

- **every target bar lies inside exactly `192/16 = 12` supervised windows**;
- nearest-neighbour overlap is `176/192 = **91.67%**`;
- an origin overlaps exactly **22** others (`k = ±1..±11`), and its **mean overlap with those 22
  is `96/192 = 50.0%`** (`Σ 2(192−16k) = 2,112` bars over 22 neighbours);
- the mean over *all* pairs on a ticker is `2,112/(N−1) → 0` for `N ≈ 6,000` and is a useless
  statistic; the 12-windows-per-bar and 50%-of-22 figures are the meaningful ones.

### 5.2 The budget at step 2000

| Quantity | By step 2000 | Note |
|---|---|---|
| rows drawn | 512,000 = **0.2085 epoch** | the misleading headline |
| dense `(origin, horizon)` gradient terms | `2000 · 256 · 360 · 192` = **3.539 × 10¹⁰** | what the optimizer saw |
| distinct supervised `(origin, horizon)` outcomes in the whole training span | ≈ `192 · 31,159,116` = **5.98 × 10⁹** | measured `h=1` count × 192 |
| gradient terms per distinct outcome | **5.92** | `= 28.3673 × 0.2085`; identical two ways |
| coverage, GLOBAL | **98.4870%** | measured (`PeakCause`); h=64 98.4970%, h=192 98.5655% |
| coverage, INTERIOR only | 99.9103% = `1 − (1−0.2085)³⁰` | **label as interior**; the 99.91% figure is NOT global |
| independent non-overlapping 192-bar windows consumed | **512,000** of 2,452,846 = **20.87%** | one per row, by construction: row stride = `pred_len` |
| independent bar increments consumed | `512,000 · 192` = **9.83 × 10⁷** of 4.71 × 10⁸ | 20.87% |
| independent bar increments per parameter | **≤ 3.61** (an UPPER bound, see §5.4) | 27,232,883 params |
| gradient terms per parameter | **1,300** | |
| independent origin TIMESTAMPS in the whole training span | `96,740 / 192` = **≈ 504** | per-ticker lattice; the time-dimension `n` for a horizon-192 relationship |
| names sampled per such timestamp by step 2000 | `512,000 / 504` ≈ **1,016** of 4,873 | |

### 5.3 So is 2000 steps a lot or a little of independent information?

- **For the repetition question: a lot, and the framing was wrong.** 5.92 passes over 98.49% of
  every distinct outcome. There is no fresh-data regime at step 2000; there is a ~6× re-fit of
  essentially the entire training signal. Whatever memorization capacity 27.2M parameters have
  over 5.98 × 10⁹ outcomes has been exercised six times over.
- **For the time-dimension question: very little.** A horizon-192 relationship has an effective
  `n` of **~504 non-overlapping windows in time** for the entire corpus — ~5 years of one series
  — and by step 2000 essentially all 504 are already represented (each by ~1,016 of its ≤4,873
  names, since a uniform 20.85% row sample hits every timestamp). The cross-section does not
  rescue this: the targets are market-neutral, so the common factor is already removed, and the
  residual sector correlation `ρ` leaves an effective width of order `1/ρ` ≈ 5–50 names, not
  4,873. **Assumption, not measurement:** `ρ` is not measured here, so the effective width is a
  bracket. What is not a bracket is the 504.
- **Consequence (HYPOTHESIS):** 504 independent long-horizon time windows against 27.2M
  parameters is why long-horizon skill peaks and rots while h=1 — which has ~96,740 independent
  time steps per ticker — stays nearly flat. That is a capacity-versus-independent-information
  statement, and it is the reading the in-period draw is built to confirm or refute: if in-period
  IC *also* collapses, §5.3 is the whole story and the lever is effective sample size.

### 5.4 Does within-window mean reversion move these numbers? (robustness)

`FutureTeacher`'s job 5453 measured within-window mean reversion strong enough to break a
martingale-residual bound at 191 of 192 horizons. Neither number above assumed a martingale —
that assumption lived in the ceiling, not here — but reversion does touch them, and in both cases
in the direction that would weaken the conclusion, so the conclusion survives.

- **~504 is untouched and mildly conservative.** It is a GEOMETRIC count (96,740 training bars
  per ticker / 192), not a variance calculation, and reversion changes the joint distribution of
  those windows rather than how many disjoint ones exist. Second order, reversion makes adjacent
  disjoint windows *negatively* correlated, and negatively correlated samples carry more
  information per sample than independent ones (the antithetic-sampling effect): at a
  cross-window lag-1 autocorrelation of ~−0.05 the effective count rises ~10%, to ~555. So the
  data-bound conclusion survives a correction in the direction that would undo it.
- **≤ 3.61 increments per parameter moves DOWN.** That figure treats each bar increment as one
  independent shock. Under reversion part of each increment is a deterministic function of its
  predecessors, so the innovation variance per bar is strictly below the increment variance and
  the count of independent SHOCKS is strictly below 9.83 × 10⁷. **3.61 is an upper bound**, and
  the capacity conclusion rests on it being small, so `FutureTeacher`'s measurement only tightens
  it.
- **A consistency check, not a new claim.** Reversion means the true variance of a cumulative
  192-bar window is BELOW random-walk `sqrt(h)` scaling. A head extrapolating martingale-like
  scaling into a reverting process would over-amplify its long-horizon mean by exactly the ratio
  of the two scalings — which is what `β̂ = .26594` at h=192 (3.8x over-amplified) looks like.
  Three independent routes (the amplitude fit, the ceiling, this ESS geometry) now arrive at the
  same defect, which argues the over-amplification is a **mis-specified horizon scaling** — an
  `OrthoTargets` / `target_basis` question — and not a fitting failure to be regularized away.

### 5.5 What the split cannot separate (stated so it is not claimed)

Sample exhaustion is a property of the FITTED FUNCTION, not of the scoring draw: if the model has
run out of independent windows and begun fitting window-specific outcome noise, that damage is in
the weights and must appear in EVERY held-out draw at the same step and in similar proportion,
because exhaustion has no mechanism to be draw-selective. Non-stationarity is a property of the
MISMATCH between fitted and scored period, so it can only damage the draw whose period differs.
That asymmetry is the instrument, and it is why outcome 1 requires all three of: in-period h=64
retaining ≥ 70% of its own maximum at 2500; the difference exceeding +0.08 at > 4σ; and the two
peaks falling at DIFFERENT steps (exhaustion predicts one saturation step and therefore one
shared peak — which is why 250-step resolution is not a nicety).

What it does **not** separate: an in-period IC that holds up is consistent with either a
stationary relationship OR **memorization of a persistent factor realization spanning the hole**.
The hole is 6,144 bars (~79 sessions) with the nearest trained target bar ≥ 192 bars before `lo`
and ≥ 5,998 bars after `hi`, and the in-period origins' 6,000-bar contexts necessarily overlap
trained territory — that is what "in-period" means. A factor whose memory greatly exceeds ~79
sessions is therefore not excluded. This does not change the decision, because "a relationship
that holds locally in time and drifts" and "non-stationarity" are one phenomenon seen from two
sides and take the same lever; it does forbid the strong reading "stationary relationship plus
pure overfitting", which would need a hole wider than the factor's memory and which no
10-minute arm can afford.

---

## 6. The arm to run

Total **≈ 12.2 min** by the measured breakdown in §3, inside the 15 min Main granted, with no
step reduction. `--eval-every 250` rather than 1000, because the primary falsifier is the
LOCATION of two peaks (see §4) and a 1,000-step grid cannot locate a peak at 2000 at all.

**Corrected form (the earlier one in this report was wrong twice, both caught by Main):** it
passed `--data-dir /var/tmp/tb0_v17`, which is a stamped BINARY, not a data directory — that
substitution appeared twice in command lines of mine and is retracted here so the successor does
not copy it — and it launched `./trading_bots/run-release-cuda.sh`, which REBUILDS from the live
tree at launch, so an arm would run whatever the tree happened to be rather than a stamped
snapshot. The binary is the first argument; `--data-dir` is left at its default, which is the
real corpus.

```bash
mlq submit --name timexer-inperiod-2500 --max-parallel-runs 1 --time-limit 15m \
  --cwd "$PWD" -- ./torch-env.sh /var/tmp/tb0_v18 train-timexer-segment \
  --run timexer-inperiod-2500-20260908 \
  --features all --layers 8 --d-model 512 --heads 8 --ffn 2048 --min-history 256 \
  --batch-size 256 --optimizer polar-express --preview-patience 3 --x0-lambdas disabled \
  --horizon-loss inv-sqrt \
  --in-period-sections 32 \
  --eval-every 250 --max-steps 2500
```

Differing flags versus the `timexer-invsqrt-ic-6k` arm the observation came from:
`--in-period-sections 32` (the whole point), `--eval-every 250` instead of 1000,
`--max-steps 2500` instead of 6000, and `--time-limit 15m`. `--schedule-budget` is deliberately
left at its default so the warmdown is still shaped against the planned epoch and these are the
first 2,500 steps of the same trajectory — note the epoch is now 9,037 steps, so the schedule
differs from the control's by that ratio, which is a stated cost of the arm and not a bug.
**Not submitted: the queue is Main's.** `/var/tmp/tb0_v18` must be stamped from a GREEN tree
(v13 barred, v14 incomplete, v15/v16/v17 all predate the anchored placement).

**SEQUENCING, ruled by Main and it is binding: this arm must NOT run before the anchored
placement is measured.** Its out-of-period draw is the 4,000-window strided one at IC SE ≈ .0164
while its in-period draw is anchored at 8,192 windows and SE ≈ .0111, and the primary falsifier
compares the SHAPE of those two IC-versus-step curves. Running it on the defective
out-of-period draw spends a lease to compare a curve against a 4.7× noisier one on a 250-step
grid. Order: §7.1's before/after table first, then this arm with BOTH draws anchored (difference
SE ≈ .0117).

Read on completion, in this order: (1) the difference series at h=64 and h=16 on
`timexer_segment_temporal_generalization` against the ±2σ band from its own SE series — that
alone decides §4; (2) the STEP of each draw's peak, which is the criterion that needs no SE and
no threshold and is therefore the primary falsifier; (3) `in-period held-out pass total` on
`timexer_segment_timing`, to replace the 2,574 ms forecast with a measurement; (4) the
out-of-period peak against the predicted 1,885 (weak handle, §4); and (5) the `held-out sample
persistence NLL` series, which must still read 2.394766330718994 — if it does not, the hole
leaked into a held-out draw and everything above is void.

## 6.1 The ordinal-offset class: a full audit (assigned by Main after occurrence 4)

**Separating rule: an ordinal offset from a shared boundary is safe iff nothing ever GROUPS
ACROSS TICKERS on the resulting position, and unsafe the moment two tickers' positions are
compared, matched or pooled.**

| # | Site | Verdict | Evidence |
|---|---|---|---|
| 1 | `C:1095-1102` the `band` closure, `origin = start - 1 + i·pred_len`, `start = boundaries[first].max(common_context)`. Builds BOTH `calibration_refs` and `validation_refs`. | **LIVE DEFECT** | MEASURED: 433,303 validation origins over 40,837 distinct timestamps, exactly **100** hold the 40 names `CROSS_SECTION_FLOOR` needs — **99.75% of the population forms no usable cross-section**. Only `i = 0` is truly shared and it alone holds 3,692 tickers. Cost: 100×40 = 4,000 windows at SE ≈ .0164, against 128×256 = 32,768 at ≈ .0035 for an anchored placement of the SAME population — **4.7× precision on every published trading number**. `CROSS_SECTION_FLOOR` and the `k·(uniform−1)` search in `cross_section_blocks` are not a design; they are the workaround for this. `max(common_context)` is a second, smaller instance in the same line: a short-history ticker starts at an arbitrary moment, not at the boundary. |
| 2 | `D:136` `train_end = retained_partition_end(boundaries[0], …)` = `boundaries[0] − purge` | **LATENT, safe direction** | A sparser ticker's purge band is WIDER in wall clock, never narrower, so nothing leaks. What it costs is uniformity of interpretation: the training span does not end at one moment across the universe, so "the training cutoff", `train_target_bars` and `in_period_purged_rows` are per-ticker quantities reported as one. Even two tickers with identical start and end dates end at different wall clocks if their densities differ — a regular-hours name loses ~2.5 sessions to the purge where an extended-hours name loses ~1. |
| 3 | `C:388-389` `owns_partition_targets`: `boundaries[first] − 1`, `retained_partition_end(…) − pred_len` | **LATENT** | Per-ticker membership predicates that nothing pools. Live the instant a consumer groups on the first or last admissible origin. |
| 4 | `probe.rs` `Partitions::split` — the five-year gap | **FIXED CORRECTLY** (`LatentProbe`) | Cut on TIMESTAMPS rather than softening the guard; measured price 1.0% of the scored population (433,303 → ~428,900), refused above `OUTER_PURGE_CEILING = 5%`. Listed so the class ledger is complete. |
| 5 | `D:135` `boundaries = raw_boundaries.map(\|b\| b − invalid.partition_point(\|i\| i < b))` | **SAFE BY CONSTRUCTION — do not touch** | An exact, monotone raw-index→valid-ordinal relabelling of ONE shared timestamp. No offset. This is *why* `boundaries[i]` is a shared moment and the whole distinction is expressible. |
| 6 | `R` `fixed_origins(&validation_refs, count)` — the `held-out sample` draw | **SAFE — do not touch** | A strided pick over a list; it computes no cross-sectional statistic, so alignment is irrelevant. **The 2.3947663 persistence anchor is unaffected and stays comparable.** |
| 7 | `C:118`, `C:464` — this report's own new code | **SAFE BY CONSTRUCTION** | `boundaries[0] − (purge + 1 + above + span)` is used only as an ADMISSIBILITY bound (a per-ticker room test), never as a position. Positions come from `valid_ordinal_at_or_before(shared_timestamp)`. Documented at `C:70-79` with the 1,914-day measurement, so the next reader meets the counterexample before the temptation. |

### Independent corroboration that the corpus supports the anchored draw (DEMONSTRATED, external)

The portfolio sibling's account tape does **not** consume `validation_refs`; it builds its own
timestamp-keyed decisions, and measured on h=128: **1,180 decision timestamps, 301,513 origins,
width 252–256 with mean 255.519, over 60 sessions.** Read against row 1 of the table above: a
timestamp-keyed construction over the SAME corpus saturates the 256 cap on essentially every
timestamp, where the strided `band` placement yields 100 usable timestamps at width ~40. So the
**99.75%-unusable figure is a property of the placement rule and nothing else**, and the corpus
genuinely supports the 128×256-class draw the fix predicts. This is corroboration from a party
that was not trying to prove the point, which makes it the strongest kind available, and it
gives the fix a target to hit rather than a hope.

### The `band` fix, as LANDED and GREEN (DEMONSTRATED — compiles, tested; NOT yet run on GPU)

**What exists in the tree now:**

| Symbol | What it does |
|---|---|
| `C` `anchored_partition_refs(tickers, first, boundary_stamp, floor, pool)` | The anchored placement of one reserved partition. Picks a REFERENCE CLOCK — the admissible ticker with the most valid bars that owns complete partition targets at the partition's own first bar, `SPY` on the real corpus — walks it forward `pred_len` bars per anchor, and takes each ticker's bar at that EXACT timestamp or nothing. Per ticker: one binary search plus a forward walk over the partition's own bars (page-friendly; a search per anchor would be a random fault per anchor per ticker into 28 GB). Anchors thinner than `floor` are dropped. |
| `C` `Corpus::anchor_cross_section_draws(floor)` | Replaces `calibration_refs`/`validation_refs` in place after load, updates `validation_target_bars`, sets `validation_remainder_bars = 0`, publishes the anchors and the placement name, prints the before/after census AND the before/after digest, and **refuses** if the new digest equals the old one. |
| `C` `Corpus::origins_sha256(refs)` | SHA-256 over `(ticker SYMBOL, origin)` pairs. Hashed by symbol, not universe position, so a reordering cannot forge a match. This is what makes a placement change observable to the external tape's `validation_refs`-SHA cache binding. |
| `C` `Corpus::distinct_timestamps(refs)` | The quantity the strided rule destroys, so both placements can state it. |
| `C` `ANCHORED_PLACEMENT = "anchored-shared-wall-clocks"` | The value `cross_section_placement` carries. Empty = strided, which is what every manifest on disk says by omitting the field. |
| `C` 3 new `CorpusContract` fields | `cross_section_placement`, `calibration_anchors`, `validation_anchors`, all `skip_serializing_if` at their defaults, so **every existing manifest digests byte-identically** and `SCHEMA` is deliberately NOT bumped (no cache invalidation, no 172 s cold rescan imposed on anyone). |
| `R` `evaluate --placement strided\|anchored\|both` | `both` scores the SAME weights twice in ONE process — strided into `--output`, anchored into `--output/anchored-shared-wall-clocks`. `Arc::get_mut` replaces the placement in place rather than holding a second 28 GB corpus resident. Deliberately NOT a training flag: a placement change moves every published cross-sectional number, so it earns its way into training runs through the before/after table, not before it. |

**Requirement satisfied, and one carried requirement REVISED with the reason.** The earlier scope said `validation_refs` must stay byte-identical so the 2.3947663 anchor survives, with the anchored draw as an ADDITIONAL population. That is not what landed, and the change is deliberate: `held-out sample`, `held-out full` and the trading family all read `validation_refs`, so an "additional population" would have required a second parallel draw threaded through every scoring path and every report base — and it would have left the DEFECTIVE draw as the one every published number is measured on, which is the opposite of a correctness fix. Instead the placement is a per-process transformation, so the strided draw remains reachable from the same binary (`--placement strided`, the default), byte-identically, and nothing on disk changes meaning. What a reader must never do is compare an anchored run's persistence NLL to 2.3947663: it is a property of a DRAW, and under this placement it must be recomputed. The digest print exists to make the two populations impossible to confuse, and `--placement both` puts them in two directories rather than two rows of one chart.

**The test (`C`, `the_anchored_placement_puts_every_origin_on_a_published_wall_clock`, PASSES).** Four claims on the 4-ticker fixture whose `DDD` skips every third slot over its first 3,000 bars — so from bar 3,000 on, `DDD`'s ordinal for any moment runs ~1,000 behind the dense tickers' while its timestamps coincide exactly, the fixture form of the real failure. Claim 1 asserts the fixture's strided draw really is scattered (`distinct_timestamps > 1`), so the rest cannot pass vacuously on a corpus whose tickers share one ordinal line — which is exactly the configuration that hid the real defect. Claim 2: every anchored origin's timestamp is a PUBLISHED anchor, every origin still owns complete `[80%, 90%)` targets, `distinct_timestamps == anchors.len()` (i.e. no origin off the grid), every anchor clears the floor, and `DDD` is present — a placement that dropped the misaligned ticker would look aligned while having thrown the population away. Claim 3: the digest differs. Claim 4: the contract names the placement, `validation_target_bars` is exact, and **a control load still publishes none of the three fields**.

**Anchors are dropped below `floor` for a measured reason, not tidiness.** Eligible labels THIN near a purge boundary because a sparse ticker's `pred_len` forward bars run past `retained_partition_end`; the portfolio sibling measured exactly this on a post-training window (widths min 11, mean 153.4, max 256, against synchronized decisions at 252–256). A block the draw cannot consume is population that only makes the census harder to read.

**NOT DONE, and it is the deliverable:** the before/after table on `ccffa620…09116` has not been run. `--placement both` is the mechanism; §7.1 has the command line. The sibling's `portfolio_data.rs`, `portfolio.rs`, `portfolio_calibration.rs` were not touched.

---

## 7. Files changed

| File | Change |
|---|---|
| `C:31-57` | `IN_PERIOD_BRACKET_ROWS`, `IN_PERIOD_SATURATING_TICKERS`, `IN_PERIOD_MIN_UNIVERSE`, `IN_PERIOD_ANCHOR_ATTEMPTS`, `struct Placement` |
| `C:59-232` | `in_period_plan()` — the coverage sweep, the reference clock, the attempt loop, the census refusal and its `println!` |
| `C:234-240` | `saturated()` |
| `C:267-289` | 7 `CorpusContract` fields, all `skip_serializing_if` at their defaults (`is_zero`, `Vec::is_empty`) |
| `C:287-298` | `CorpusTicker.hole` and `CorpusTicker.in_period` |
| `C:376-465` | `in_period_origins()`, `valid_ordinal_at_or_before()`, `place()`, `supervision_clears_hole()` — the single definition of the guarantee |
| `C:955-974` | plan applied AFTER the eligibility retain, cost folded into `contract_ms` |
| `C:1080-1102` | per-row enumeration with the purge, its exact census, and the placed-origin extend |
| `C:1180-1190` | contract wiring incl. `in_period_anchors` / `in_period_census` |
| `C:2626+` | the disjointness test, the `DDD` sparse-clock fixture ticker, and the real-corpus draw test |
| `R:78-81` | `CROSS_SECTION_TICKERS` made `pub(super)` so the hole is placed to saturate the draw's own cap rather than repeating the constant |
| `R` | `TrainArgs.in_period_sections` + Default; `purged_row_share`; the in-period draw; the interval score; `in_period_horizons`; 3 `Metrics` literals |
| `P` | `IN_PERIOD`/`OUT_OF_PERIOD`; `Metrics.in_period_horizons` / `in_period_origins` / `in_period_purged_row_share`; `EvalTiming.in_period_ms`; 2 scope facts (one of them the holed-population warning on every title); 1 timing series; the new chart block |
| `S:183-199` | `timexer_segment_temporal_generalization` registered |
| `C` | `ANCHORED_PLACEMENT`; `anchored_partition_refs()`; `Corpus::anchor_cross_section_draws(floor)`; `Corpus::origins_sha256()`; `Corpus::distinct_timestamps()`; 3 more `CorpusContract` fields (`cross_section_placement`, `calibration_anchors`, `validation_anchors`), all `skip_serializing_if`; `the_anchored_placement_puts_every_origin_on_a_published_wall_clock` |
| `R` | `EvaluateArgs.placement` + `enum Placement {Strided, Anchored, Both}`; `evaluate()` split so `evaluate_placement()` runs one pass per placement in one process |

Verification run (all DEMONSTRATED): `./torch-env.sh cargo check -p trading_bot_0 --tests`
**0 errors**; `./torch-env.sh cargo check -p trading-bot-tui --tests` **0 errors**;
`cargo test -p trading_bot_0 timexer_segment::corpus` **10 passed, 0 failed, 2 ignored**,
including `in_period_origins_share_no_origin_and_no_target_bar_with_any_training_row`; and both
ignored real-corpus tests re-run explicitly after the change, **2 passed** — the checkpoint
compatibility test (`C:2491`, `corpus.contract == published` against `timexer-control-4k`) and
the draw test whose output is quoted verbatim in §2.3;
`cargo test -p trading_bot_0 timexer_segment::reports` **13/13**; `cargo test -p shared`
**19/19** (the base registry); `cargo test -p trading-bot-tui` **36/36**. (Plain
`cargo check` fails with a torch-sys `LIBTORCH_BYPASS_VERSION_CHECK` refusal; the wrapper is
mandatory and that failure is not a real error.)

Re-verified after the anchored placement landed: `cargo check -p trading_bot_0 --tests`
**0 errors**; `cargo check -p trading-bot-tui --tests` **0 errors**;
`cargo test -p trading_bot_0 timexer_segment::corpus` **11 passed, 0 failed, 2 ignored** (the
11th is the anchored-placement test). The two `--ignored` real-corpus tests were last run
BEFORE the anchored placement landed; they exercise `Corpus::load` only, which the placement
does not touch (it is a post-load transformation), so they are expected to stay green — but
that is `[INFERENCE]` and the successor should re-run them explicitly.

---

## 7.1 Jobs to run, in this order

**1. The before/after table on the pinned checkpoint. This is the licence for re-quoting every
cross-sectional number in the project, and it is CPU+1-GPU cheap: no training, two scoring
passes.**

```bash
mlq submit --name timexer-placement-ba --max-parallel-runs 1 --time-limit 10m \
  --cwd "$PWD" -- ./torch-env.sh /var/tmp/tb0_v18 evaluate-timexer-segment \
  --checkpoint training/runs/timexer-control-4k/weights/best \
  --output training/runs/timexer-placement-ba \
  --batch-size 256 \
  --placement both
```

(`--checkpoint` is a WEIGHTS directory, not a run output directory: an earlier draft of this
line said `gens/1`, which is where reports are written, and Main corrected it to
`weights/best` while queueing. Verified against that run's manifest rather than from memory:
step 2000, `weights_sha256 = ccffa620654f1c95faa618501519e036249615e98d6034e425e3fed7c2309116`.)
**SUBMITTED AND RUNNING as job 5483** on `/var/tmp/tb0_v18`, stamped from the green tree, 12m
limit, no training. Read from stdout: the placement census line
(`calibration:` / `validation:` origins, anchors, mean/min/max width) and the
`validation_refs sha256 <old> -> <new>` line, which is the artifact the external tape's cache
binding must also see change. Then read, from the two report directories, the table Main
specified: cross-section count, mean width, IC and SE at h=1/8/32/64/192, and the trading
family's gross/net/breakeven. **The strided set is the BEFORE and the anchored set is the AFTER
of the same weights in the same process**, so any difference is the placement.

**2. The in-period arm** (§6), only after 1, and with both draws anchored.

**3. Re-measure `martingale_ratio` on the anchored draw beside 5471's `variance_ratio`**
(assigned by Main, NOT built). The prediction is CONVERGENCE: `martingale_ratio` pools over the
PAIRED population at `n ≥ CROSS_SECTION_MIN = 40`, which under the strided placement is 100
timestamps = 0.24% of the population and not a random 0.24% but the `i = 0` stride — one
systematically selected set of calendar moments. Under the anchored placement the paired
population stops being a systematically-selected subset, so the two estimators should agree. If
they still disagree materially there is a second defect underneath this one. This needs a
`martingale_ratio` read on the anchored draw, which job 1 above does not emit; whoever builds it
should add it to the anchored pass rather than to a new binary.

---

## 8. Pre-registered predictions: status, with retractions

| # | Claim | Status |
|---|---|---|
| 1 | **Mechanism.** Outcome 1 (non-stationarity): in-period h=64 retains ≥ 70% of its own max at 2500 AND the difference exceeds +0.08 at > 4σ AND the two peaks fall at DIFFERENT steps. | **STANDS, unrun.** Main confirmed the sizing survives the batch-wide threshold audit: +0.08 against a difference SE ≈ .0156–.0164 is ~5σ, and the peak-LOCATION criterion needs no SE and no threshold at all, which is why it is primary. |
| 2 | **Clock.** The holed arm's out-of-period peak falls in [1,855, 1,890] rather than at 2000. | **STANDS, DOWNGRADED to corroboration.** Originally [1,760, 1,790] off a 10.66% row price; the measured price is 5.78%, so the window moved. A 6% perturbation predicting a ~5% shift sits at the edge of what a 250-step grid resolves, so this is read as support for `DataDiversity`'s far larger `K=4`/`K=30` perturbations, not as a test. |
| 3 | `martingale_ratio` and 5471's `variance_ratio` CONVERGE on the anchored draw. | **NEW, unrun** (job 3 above). |
| 4 | 4.7× IC precision from the anchored placement (SE ≈ .0164 → ≈ .0035). | **PARTIALLY CORROBORATED, not yet measured by me.** The external sibling's timestamp-keyed construction reaches 1,180 timestamps at mean width 255.5 on the same corpus, which is the target the fix must hit. Job 1 measures whether it does. |
| — | *Retracted:* the ~1,787-step clock prediction and the 10.66% row price. | Superseded by the measured 5.78%. |
| — | *Retracted:* `--data-dir /var/tmp/tb0_v17` and `run-release-cuda.sh` in the §6 command line. | A binary is not a data directory and a rebuild-at-launch script is not a stamped snapshot. Both were mine, twice. |
| — | *Withdrawn as a REQUIREMENT, not a prediction:* "`validation_refs` must stay byte-identical". | See §6.1's landed-fix note: keeping it would have left every published number measured on the defective draw. Reachability from one binary replaced byte-identity, and the persistence anchor 2.3947663 is now explicitly a property of the STRIDED draw. |

## 8.1 What I depend on from other agents (and how load-bearing it is)

| Number | Owner | Load-bearing? |
|---|---|---|
| Interior multiplicity `M = 30`, mean 28.3673, interior coverage 99.9103% at step 2000, interior saturation step 1972.4 | `PeakCause` / `DataDiversity` (independently reproduced to six digits) | **Yes**, for §5 and for prediction 2's clock. Rests on census arithmetic, not on IC magnitudes, so no IC correction touches it. |
| 1,180 timestamps at mean width 255.5 (h=128) | external portfolio sibling | Corroborative only. If it were wrong, prediction 4's TARGET would move; the defect measurement (100 usable timestamps of 40,837) is mine and independent. |
| Recent-window Cov/Var inversion −.1097 at h=32 | external sibling, via Main | **Cited, not load-bearing.** Per Main's Correction 1 this is ONE horizon; h=64 (−.0230 ± .0413) is uninformative at 5 sessions. So it is consistent with Outcome 1 and cannot be quoted as horizon-selective corroboration. My h=64 criterion remains untested by it. |
| The monotone VR drift .586 → .548 → .525 | `FutureTeacher`, via Main | **Not used.** Main withdrew the premise (the recent window reads .595/.652 at h=64, breaking the ordering). Nothing in this report rests on it. |
| `LatentProbe`'s dating audit (training targets end 2023-08-17; 1 overlapping origin in 433,303) | `LatentProbe` | **Yes**, for §6.1 row 2's "latent, safe" verdict on `train_end`. It is the measurement that separates a 1,644-day span overlap from a one-row population overlap. |
