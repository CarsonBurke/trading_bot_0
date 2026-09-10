# CausalPatch: two contemporaneous cross-section channels

Date: 2026-09-08. Worker: `CrossSectionFeatures`.
Scope: `trading_bots/src/torch/timexer_segment/{features,corpus,cache,reports,model,benchmark}.rs`.
Status at write time: code landed, checks and tests green, **no run exists**. The prediction in the
last section is therefore pre-registered in the strict sense.

## 1. Why the input side, and why the cross-section

Five horizon-axis interventions have been refuted (`cutoff:32`, `inv`, `inv-sqrt`, `basis:8:8`,
`increment:8`; see `timexer_increment_verdict.md`), and the measured predictable-information
ceiling is flat in the horizon with a constant `c_1`
(`timexer_ceiling_placement_20260908.md`). A flat ceiling cannot be reached by reweighting or
reparametrizing the horizon axis: there is no horizon-specific information to find. The only
remaining lever is per-bar information.

Every row the model sees is one ticker: its own OHLC in units of its own causal σ, calendar and
session channels, its own volume innovation, the equal-weighted corpus step, and the SPY step.
The training target is the beta-adjusted **cross-sectional residual** return. The row-local
numerator of that residual is already derivable inside a row — `market_cum` is a row channel and
the model fits its own β and σ from it. What is **not** derivable from any single row is the
**denominator**: how wide the contemporaneous cross-section is at this bar. Cross-sectional
dispersion is a property of the slot, not of the ticker, and no amount of own history yields it.

That is the one scalar per bar this change adds, plus the interaction it enables.

## 2. Exact channel definitions

Both channels are appended at the END of `Feature::ALL`, so every pre-existing channel index is
unchanged. `WIDTH` stays 2 (value, validity).

Let a *defining slot* be one where the market step is defined: at least
`--market-min-cross-section` sources hold a valid bar at the slot, and at least one source holds
valid bars at both this slot and the **previous defining slot** (`MarketSteps::step`). Let
`r_i` be the contributing log close returns at such a slot and `n` their count.

- `μ_slot = (1/n) Σ r_i` — this is exactly the existing `Feature::Market` value.
- `σ_slot = max( sqrt( max( (1/n) Σ r_i² − μ_slot² , 0 ) ), DISPERSION_FLOOR )`, population
  convention, `DISPERSION_FLOOR = 1e-6`.

**`Feature::Dispersion`** = `[ln σ_slot, 1]` at a defining slot, `[0, 0]` otherwise.
`ln` because dispersion is positive and approximately log-normal across slots, so the log is the
transform that makes the channel roughly symmetric and O(1)-varying; the constant offset
(`ln 2e-3 ≈ −6.2`) is absorbed by the patch embedding's bias, which is why no reference scale is
subtracted.

**`Feature::CrossSectionZ`** = `[clamp((own − μ_slot)/σ_slot, ±16), 1]` when the slot defines a
step **and** the bar's own five-minute return exists, `[0, 0]` otherwise. `own` follows
`single_series`'s rule exactly: `ln(close_t / close_prev)` only when the previous **valid** bar is
exactly one interval earlier, so no return is ever taken across a gap. This is the input-side
analogue of the market-neutral target.

Roles: `known_future = false` for both (they are history, and read `[0,0]` beyond the context).
`sigma_scaled = false` for both. `Market`/`Spy` values are log returns scaled into the origin's
causal σ because they are commensurate with prices; `ln σ_slot` is the log of a dimensionless
dispersion and the z is already standardized by that same dispersion, so dividing either by the
row's causal σ would compose two normalizations and make a corpus-wide fact depend on the origin.

### Causality

The dispersion and the z read the market step **ending at the bar's timestamp** — the same object
the `Market` channel reads, through the same `first_ts + slot * RESOLUTION_MS` indexing, with an
exact-timestamp match (`CrossSection::at` rejects off-grid offsets) and no forward fill. A slot's
step and dispersion are functions of closes at that slot and at the previous defining slot only,
both at or before the bar. `MarketPath`/`MarketSteps` were already built this way and the new
second moment rides the identical traversal, so no new information path exists.

Definedness is shared by construction: `MarketSteps::dispersion` returns `Some` on exactly the
slots `MarketSteps::step` returns `Some` on (it early-returns on `self.step(slot)?`). The
dispersion channel and the market channel therefore never disagree about what a bar is. When the
slot is undefined, both cross-section channels are exactly `[0, 0]` — zero information, not a
small number.

### Why the floor and the clip never fire in production

The control corpus reads `market: 197362 steps over >= 2000 tickers, contributors min 1503
median 3408, step std 0.00139`. Every defining slot's dispersion is therefore measured from at
least 1503 returns, so `DISPERSION_FLOOR` cannot bind (it binds only at n = 1 or at identical
closes) and the population/sample distinction is a factor `1 + 1/(2·1503) = 1.0003`. The `±16`
clip exists for the same reason a `GAP_LOG_CLIP` exists: at n ≥ 1503 a residual past sixteen slot
sigmas is a stale print, an unadjusted split or a halt reopen, not a move.

### Numerics

`E[r²] − μ²` is the streaming form, which is why the second moment needs no second pass. It is
safe here by a wide margin: μ is order 1e-5 and σ order 1e-3, so `μ²/E[r²] ≈ 2.5e-5` and the
subtraction cancels a fraction of one decimal digit of the sixteen f64 carries. The test asserts
the result against the mean-of-squared-deviations form, a different arithmetic path, at 1e-15.

## 3. Cost

Grid: `first_ts = 1471852800000` (2016-08-22T08:00Z) through 2026-08-20T00:00Z,
**1,051,104 slots** (read out of the live cache artifact
`long_data/bars/.timexer-cache/market-steps.bin`).

**Added memory**

| Item | Size | Lifetime |
|---|---|---|
| `MarketSteps::squares` (`Vec<f64>`) | `1,051,104 × 8 B = 8,408,832 B` (8.41 MB) | corpus load only; dropped with `MarketSteps` |
| `CrossSection` (3 × `Vec<f32>`: `market`, `log_sigma`, `inv_sigma`) | `1,051,104 × 12 B = 12,613,248 B` (12.61 MB) | whole run, inside `Exogenous` |
| market-grid cache artifact | +8.41 MB (postcard writes f64 fixed-width): 11.08 MB → ~19.5 MB | on disk |
| `accumulate` chunk-local accumulators | +8.41 MB per chunk × 16 chunks = +134.5 MB worst case (202 MB → 336 MB) | during the market-grid pass only |

No tickers × slots matrix is materialized anywhere; the second moment is a per-slot scalar.

**Added CPU**

- Market-grid pass: **no new traversal**. One multiply and one f64 accumulate-and-store per
  contributing return, inside the existing `valid_slots` walk over 752,901,372 source bars.
- `MarketSteps::cross_section()`: one pass over 1,051,104 slots, with one `ln` and one `recip`
  on the 197,362 defining slots. Sub-millisecond, once per load.
- Row assembly: 4 extra f32 stores and one grid lookup per bar. At batch 256 × 6192 bars that is
  6,340,608 extra stores = 25.36 MB per batch. Both channels are stored pre-transformed
  (`ln σ`, `1/σ`), so a bar pays two loads and a multiply, not a divide and a transcendental.
- Host row block: `length × (6 + aux) + 1` f32 per row, so 114,131,968 → **139,494,400 bytes per
  batch (+25,362,432, +22.2%)** of host assembly writes and H2D copy. This is the largest
  relative cost of the change; the prefetcher overlaps it with the step.

**Model width, parameters, arithmetic** — measured, not estimated: a CPU `VarStore` was built at
the control geometry (`seq_len 6000, pred_len 192, patch_len 16, layers 8, d_model 512, heads 8,
ffn 2048, x0 disabled, mean free`) for both feature sets and `ModelConfig::step_cost(256)` read
off. The measuring harness was a throwaway and has been deleted.

| | control (12 aux ch) | arm (16 aux ch) | Δ |
|---|---|---|---|
| aux channels | 12 | 16 | +4 |
| patch-embedding fan-in `patch_len × (CHANNELS + aux)` | `16 × 16 = 256` | `16 × 20 = 320` | +64 |
| trainable parameters | 27,954,475 | 27,987,243 | **+32,768 (+0.117%)** |
| TFLOP / step (`matmul_flops`) | 16.98457 | 17.00345 | **+0.01888 (+0.111%)** |
| traffic GB / step | 123.6664 | 123.7402 | +0.0738 (+0.060%) |

The arithmetic behind both deltas: the only weight whose shape moves is the patch embedding,
`Linear(patch_len·(4+aux) → d_model)`, so `Δparams = 512 × 16 × 4 = 32,768` (the bias is
unchanged), and `step_cost` charges that GEMM `2·tokens·fan_in·d_model` forward with a factor 3
for the step, so `Δflops = 3 × 2 × (256 × 375) × 64 × 512 = 1.888e10 = 0.0189 TFLOP`. Nothing
else in the model is a function of the aux width except the token cast/concat traffic, which is
the 0.0738 GB. No known-future channel is added, so the covariate branch, the head and every
attention shape are byte-identical.

## 4. Feature-set plumbing and checkpoint safety

`--features all` now means all eight. Every feature stays individually nameable:

- **OLD input, reproducing `timexer-control-4k` exactly:**
  `--features time-of-day,day-of-week,session-gap,volume,market,spy`
- **NEW arm:** `--features all`
  (identically `--features time-of-day,day-of-week,session-gap,volume,market,spy,dispersion,cross-section-z`)

The first string is pinned by two tests: it is asserted to yield exactly 12 channels and to be
`FeatureSet`'s own `Display` output, and the whole pre-existing channel-layout fixture
(`calendar_gap_and_exogenous_channels_follow_the_bar_clock`) is now driven by that string rather
than by `FeatureSet::ALL`, so a future append cannot silently move an index it asserts on.

### Compatibility proof, and why the asymmetry is deliberate

This project has already paid for one silent incompatibility: checkpoints written with a
horizon-major head were loadable by a channel-major build, which is why `check_head_layout` and
the `v10-head-channel-major` FORMAT bump exist at all, and why the failure surfaced as a bisect
rather than as an error. The asymmetry below is chosen so that a cross-section checkpoint cannot
repeat that: **loading old is safe and stays possible; loading new under a build that lacks the
channels is impossible.**

The feature set was already authenticated in two places: `ModelConfig` and `CorpusContract` both
carry `features`, `Manifest::read` requires them equal (`runner.rs`, the `model/data contract
mismatch` ensure), the contract also carries the spelled-out `auxiliary_schema`, and
`manifest_sha256` digests the whole manifest. The FORMAT constant is deliberately *not* touched —
it stamps the x0 mode and the mean parameterization, and bumping it would invalidate every
checkpoint on disk to restate something the manifest already carries.

The proof, both directions:

1. **New build, old manifest → loads, as the OLD feature set.** The two new `FeatureSet` fields
   are `#[serde(default, skip_serializing_if = "std::ops::Not::not")]`. `default` makes them
   `false`, which is the semantically correct reading of a manifest written before the channels
   existed; `skip_serializing_if` makes them absent on the way out, so the JSON
   `Manifest::read` re-serializes for its digest check is byte-identical to what was written and
   the SHA-256 still authenticates. Without the skip, every checkpoint on disk would fail
   authentication on this build.
2. **New build, new manifest → loads, as the NEW feature set.** Serialization always emits an
   enabled flag, and `model.features == data.features` is enforced, so the channel count the
   weights were fitted at is the channel count the corpus produces.
3. **Old build, new manifest → hard error, never a silent load.** `deny_unknown_fields` on
   `FeatureSet` rejects `dispersion`/`cross_section_z` outright. There is no defaulting path, no
   width coercion, and no way for the patch embedding's 320-wide fan-in to be read as 256.
4. **A new-channel checkpoint therefore cannot be loaded as an old one under any build.** The
   only build that can parse the manifest is one that also constructs the channels.

`a_checkpoint_written_before_the_cross_section_channels_still_authenticates` pins 1 and 2 against
the control's literal JSON, so a future append that forgets the attributes fails a test instead of
invalidating the checkpoint store.

### Cache invalidation is scoped to the market grid alone

`cache::MarketGrid` gains `squares`, and `MarketCache`'s key tag became
`market-steps-second-moment`, so a stale artifact misses cleanly instead of being probed for a
field it does not carry. **The recompute is scoped to one file.** The three cache artifacts have
independent keys:

| artifact | file | key | affected |
|---|---|---|---|
| bar audits | `bar-audits.bin` (`cache.rs:56`) | `format!("{CACHE_VERSION};{schema}")`, `cache.rs:120-122` — no `extra` | **no** |
| partition boundaries | `shared-bounds.bin` (`cache.rs:57`) | `universe_key(schema, universe, b"shared-bounds")`, `cache.rs:229` | **no** |
| market grid | `market-steps.bin` (`cache.rs:58`) | `universe_key(schema, universe, &extra)`, `cache.rs:299`, with `extra` built at `cache.rs:289-296` | **yes, only this one** |

My tag lives in the local `extra` buffer at `cache.rs:293`, which is passed to exactly one
`universe_key` call, `cache.rs:299`, and hashed into exactly one file's key. `CACHE_VERSION`
(`cache.rs`) and `corpus::SCHEMA` are untouched, so the bar-audit ledger's key and the bounds key
are bit-identical to before: **5,728 bar audits stay reused and no full rescan occurs.** The first
load prints `market grid reused false` once, pays the two existing market-grid passes, re-caches,
and every later load prints `true` again.

`CorpusContract::market_fingerprint` is a digest over ticker fingerprints, the three bounds and
`min_cross_section` only (`corpus.rs:1861-1880`) — it does not read the step vectors, so it is
unchanged, and an existing checkpoint's corpus contract still compares equal on every field except
`features`/`auxiliary_schema`, which differ only on the new arm by design.

## 5. Reports

The corpus report (`timexer_segment_progress`, already registered) gains a dispersion clause beside
the market clause in `reports::write_corpus`, because the two are defined on identical slots:

```
market: 197362 steps over >= 2000 tickers, contributors min 1503 median 3408, step std 0.00139,
largest step -0.10030 at 1584365400000; dispersion: defined at 18.78% of 1051104 grid slots,
sigma_slot median <m>, IQR <p25> to <p75>
```

`MarketSummary` gained `slots` and the three `σ_slot` quartiles (nearest-rank, the convention its
contributor median already used). **No new report base was needed**, so `shared/src/report.rs` and
`tui/src/main.rs`'s `meta_chart_bases` are untouched and the bidirectional registry test is
unaffected (re-run and green). This summary is written at corpus load, so a 2,500-step arm cannot
carry a blank dispersion reading.

**Pre-run gate.** Before reading any metric, the arm's corpus report must show the dispersion
share equal to the market clause's own share (`197362 / 1051104 = 18.78%` at the production
threshold) and a `σ_slot` median of order `1e-3` with a strictly positive IQR. If the share is
smaller than the market share, or the IQR is ~0, the channels were not populated and the arm
tested nothing.

## 6. Tests

In `features.rs`'s `mod tests`, on the existing fixture style. All nine pass
(`cargo test -p trading_bot_0 timexer_segment::features`).

- `a_slot_below_the_cross_section_floor_defines_neither_a_step_nor_a_dispersion` — slot 1 holds
  2 of 3 sources at `min_cross_section = 3`: `step(1) == None`, `dispersion(1) == None`,
  `cross_section().at(..) == None`, and the row reads `[0.0; 6]` exactly across
  `market,dispersion,cross-section-z`. The fixture is built so the ticker's **own** step at that
  slot IS defined, so the assertion isolates the slot rule from a missing own return.
- `the_slot_dispersion_is_the_cross_sectional_deviation_and_the_z_is_its_residual` — two-ticker
  case: the mean is the midpoint and σ the half-spread, so the z is exactly ±1 (checked on both
  tickers); three-ticker case: σ is asserted against the mean-of-squared-deviations form at 1e-15
  and the z against `(r_c − μ)/σ`, with `ln σ` on the value channel.
- `a_previous_valid_bar_more_than_one_interval_back_contributes_no_own_step` — D misses a slot, so
  its next return spans two intervals: the dispersion channel stays valid (the slot defines one)
  while the z reads `[0.0, 0.0]`. No return across a gap.
- `appending_the_cross_section_channels_leaves_every_earlier_channel_identical` — the same fixture
  written under the old explicit six-name list and under `ALL`, over two bars and both
  `future` values: the first 12 channels compare equal, the appended pair is zero beyond the
  context, and on a live bar the appended pair is populated (so the equality is not the trivial
  one). Also pins that both appended channels are false in the `known_future` and `sigma_scaled`
  masks.
- `a_checkpoint_written_before_the_cross_section_channels_still_authenticates` — §4.
- `sparse_slots_...` additionally pins `summary.slots` and the three dispersion quartiles against
  the defining slots' own dispersions, so the report clause cannot silently go stale.
- `corpus::tests::pooled_epoch_keeps_delisted_tickers_and_covers_each_training_target_once` now
  runs at 14 channels through the real corpus load, and asserts through the production path that
  the dispersion is live wherever the market channel is, that `ln σ < 0`, and that the bar after a
  four-interval gap carries a valid dispersion but **no** z — case (c) again, end to end.

No test was deleted: nothing in the touched area pinned wording or plumbing only. Two fixtures
were re-pointed from `FeatureSet::ALL` to the explicit control string, which is strictly stronger.

## 7. Pre-registered prediction

Stated before any run of this arm exists. Comparison is step-matched against `timexer-control-4k`
at step 2000 of the same 9,590-step epoch; the arm is `--max-steps 2500`.

Control step-2000 anchors: `held-out sample` market-neutral MSE ratio **h=1 0.95528**,
**h=8 0.95066**; `held-out cross-section` anchored-draw close cross-sectional **IC h=1 0.1621**.

**Mechanism I am betting on.** The row-local numerator of the target (`own − β·market`) is already
derivable from existing channels. The new information is the *denominator*: the contemporaneous
cross-sectional scale. Its value to a mean forecast is shrinkage — the optimal predictor's
shrinkage toward zero is `signal/(signal + noise)`, and the noise term here is a fast-moving,
observable, highly persistent quantity that the row's trailing causal σ cannot track (dispersion
jumps at the open, at macro prints, and in earnings clusters). Its value to the cross-sectional
readout is comparability: per-row forecasts live in per-ticker σ units, and IC ranks them against
each other, so a channel naming the slot's own scale should move IC more, in relative terms, than
it moves MSE.

**Correction, entered before any output of this arm exists.** This section originally quoted two
floors. One of them has been retired by measurement (`EvalThroughput`, out of job 5556's
A-B-A-A' log) and the correction is recorded here rather than silently applied, because the
prediction below is only worth anything if its revisions are dated relative to the run:

- **RETIRED: the 0.71% "reproducibility drift".** It is a throughput number in origins/s and
  carries no statistical content. At **fixed evaluation shape the forward is bit-reproducible, so
  the per-horizon floor is exactly `0.000e0`** — level and shape alike, at all 192 horizons. My
  ±0.0068 MSE-ratio band was built on it and is void.
- **WITHDRAWN: my shape statistic `Δ(h=1) − Δ(h=8)`.** I justified it by "a common level drift
  cancels in the difference". The same measurement refutes the premise: across a *changed* eval
  batch size the per-horizon deltas **scatter in sign** (h=1 `+9.7e-6` against h=8 `−2.9e-6`), so
  the difference ADDS them (`1.2e-5`) and is noisier than either endpoint. Two independent shape
  changes agree to ~10%, so this is reproducible kernel-selection behaviour, not sampling noise.
  The statistic I nominated as *primary* was the worst of the three available. It is withdrawn.
- **STANDING RULE I am scored under:** both sides of a comparison at the **same evaluation batch
  size**, under which the floor is exactly zero.
- **Known exception in this specific comparison, and it is being carried, not fixed:** the
  control's step-2000 numbers were scored at eval batch 256 and new arms are scored at 64
  (`--eval-batch-size`, default 64, landed because seven arms died of OOM inside an evaluation
  pass against the captured 17.3 GiB training mempool). So this comparison is cross-shape and
  inherits the `~1.2e-5` kernel-selection floor rather than exactly zero — three orders below my
  predicted MSE move, two and a half below the IC SE, immaterial but not zero. Job 5582 measures
  its size at 64 and the verdict states it.
- **RETRACTED: my "one eval pass" fix.** I proposed re-scoring the control's step-2000 checkpoint
  at eval batch 64. It does not work, and the reason is worth recording: `evaluate-timexer-segment`
  scores `corpus.validation_refs` (`held-out full`) plus the cross-section draw, while the
  2,048-origin `held-out sample` preview draw is built inside the training loop by
  `fixed_origins(&corpus.validation_refs, args.eval_origins)` and exists **only there**. A
  re-score therefore yields a matched-shape `held-out full` — which the arm will not have unless
  it survives its final full pass — and leaves the per-horizon `held-out sample` series my rule is
  actually written against still at 256 for the control and 64 for the arm. The exact fix runs the
  other way, scoring the arm at 256, and is correctly refused: that is the allocation that killed
  the seven arms, and with foreign tenants at 15.5 GiB an arm scored at 256 is an arm that
  produces nothing. **Exactness that costs the measurement is not exactness.**

**The resolution that remains.** Only one *published* floor survives, and it is sampling error,
not reproducibility: the control's anchored-draw h=1 IC is 0.1621 with **SE 0.00839 over 245
paired cross-sections**, so 2 SE = **±0.0168**.

For the held-out-sample MSE ratio there is no reproducibility floor at matched shape, and the
sampling SE I asked for **does not exist as I posed it.** I asked for the SE "over 83,194,176
target bars", which presumes those bars are independent. They are not, by three orders of
magnitude: each row unfolds 375 causal origins at stride 16 and each origin spans 192 horizons, so
the bars overlap many times over. `TemporalSplit`'s measured ceiling is about **504 independent
non-overlapping 192-bar windows in the entire training span**, and the same construction caps the
held-out draw. `1/√83.2M` would have understated the true floor by ~1000×, which is precisely the
class of number that has been retracted repeatedly in this session — mine included.

What *is* defensible is that the comparison is **paired**: identical draw, identical origins,
identical target bars, two checkpoints. Common variation cancels, which is the same reason
`LatentProbe` used a paired IC gap rather than two independent ones. So the floor for a paired
per-horizon MSE-ratio **difference** is the matched-shape reproducibility — exactly zero, or the
measured `~1.2e-5` for this cross-shape pair — plus a paired sampling term bounded by the
**independent-window count (≤504)**, never by the bar count. That term is not a constant anyone
can look up; it is the dispersion of the paired difference across those windows, and it is
measurable from the arm's own output at whatever window granularity the report already carries.

The axes have therefore swapped roles: **MSE is now the resolving axis and IC the under-powered
one**, the reverse of what this section said before the correction.

| metric | control | binding uncertainty | my predicted point move | resolvable? |
|---|---|---|---|---|
| `held-out sample` MSE ratio h=1 | 0.95528 | reproducibility 0 matched / `1.2e-5` cross-shape, plus the paired difference's dispersion over ≤504 independent windows | **−0.0010** (0.9543), range −0.0005 to −0.0030 | **yes, if the paired difference exceeds its own window dispersion** |
| `held-out sample` MSE ratio h=8 | 0.95066 | same | **−0.0007** (0.9500), range −0.0003 to −0.0020 | **yes, same condition** |
| `held-out cross-section` IC h=1 | 0.1621 | 2 SE = ±0.0168, sampling, 245 paired cross-sections | **+0.006** (0.168), range +0.002 to +0.012 | **no** |

**Mechanism prediction, restated without the withdrawn statistic.** h=1 should gain more than h=8
in absolute terms: the z channel is return-like and decays with horizon, while the dispersion
channel is volatility-like and persistent, so h=8 collects the shrinkage part and little of the z
part. I pre-register the two *levels* separately — `Δ(h=1) < 0` and `Δ(h=8) < 0` with
`|Δ(h=1)| > |Δ(h=8)|` — and explicitly NOT their difference, whose floor is now known to be worse
than either. If h=8 gains more than h=1, my mechanism is wrong even where the sign is right, and
the honest reading is "the channels supplied a conditional-variance state variable and nothing
about the residual".

**Pre-registered decision rule.** Unchanged from what I registered before the correction, and
deliberately so — it was accepted as falsifiable as written, and loosening or tightening
thresholds after new information but before results is exactly how a pre-registration stops being
one. It is now *conservative* rather than *marginal*, which is the right direction for a rule to
move:

1. **CONFIRMED** — h=1 MSE ratio ≤ 0.9485, **or** h=1 IC ≥ 0.1789.
2. **REFUTED, net harmful** — h=1 MSE ratio ≥ 0.9621, **or** h=1 IC ≤ 0.1453. Four extra channels
   against a fixed-width patch embedding can dilute; that outcome says they did.
3. **UNRESOLVED** — anything strictly inside both bands.

What the correction changes is not the rule but my expectation under it. Before: UNRESOLVED on
both axes was the modal outcome. Now, with reproducibility at zero, the MSE axis can resolve a
move far smaller than the 0.0068 band the rule inherited — so the *secondary* reading I register
alongside the rule is:

> At the arm's own evaluation shape, `Δ(h=1) < 0` **and** `Δ(h=8) < 0` **and**
> `|Δ(h=1)| > |Δ(h=8)|`, all three holding, with each `Δ` larger than the **dispersion of the
> paired difference across the ≤504 independent non-overlapping windows** the split admits, is
> positive evidence for the channels even where the move does not reach 0.9485.

The condition is stated on the paired difference's own window dispersion, not on a published SE.
There is no `1e-3` threshold here any more and there should not have been one: the quantity I was
conditioning on cannot be computed by anyone, whereas the paired difference's spread across
independent windows is measurable from the arm's own output at whatever window granularity the
report already carries. The comparison is paired — same draw, same origins, same target bars, two
checkpoints — so the shared component cancels and what is left is bounded by the window count,
never by the 83.2M bar count.

**How to make it resolvable, concretely.**

- IC: the SE is sampling error and scales `1/√draws`. Resolving a +0.006 move at 2 SE needs
  SE ≤ 0.003, i.e. `(0.00839/0.003)² ≈ 7.8×` the draws — about **1,900 paired cross-sections**
  instead of 245. Eval passes, not training steps. This is the only axis where more sampling buys
  anything.
- MSE ratio: nothing needs buying down. What the secondary reading needs is the paired
  difference's spread across independent windows, computed from the arm's own per-window output —
  not a new run, not a re-score, and explicitly **not** the two fixes I proposed earlier and have
  retracted above.

**Diagnostic split, if the arm does resolve.** MSE clears its floor while IC does not: the
dispersion channel helped the per-row conditional scale and left untouched the cross-sectional
ordering the target is defined on. The follow-up then is the z channel alone
(`--features time-of-day,day-of-week,session-gap,volume,market,spy,cross-section-z`), not more
channels. IC clears while MSE does not: the opposite reading — comparability improved without a
better conditional mean — and the follow-up is the dispersion channel alone.

**Pre-run gate, restated as a falsifiability condition.** If the corpus report's dispersion clause
does not show the market clause's own share (`197362 / 1051104 = 18.78%`) and a strictly positive
`σ_slot` IQR, the arm tested nothing and none of the branches above may be applied to it.
