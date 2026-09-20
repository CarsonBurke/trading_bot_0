# CausalPatch: the second cross-section family — rank, relative volume, range

Date: 2026-09-09. Worker: `XSecFamily`.
Scope: `trading_bots/src/torch/timexer_segment/{features,corpus,cache}.rs`, plus the
`--features` help text in `model.rs` and `benchmark.rs`. **No run exists.** Everything below is
static: read out of the code, out of the live cache artifacts, or out of the first family's
measured table (`research/worker_reports/timexer_cross_section_features.md`). Nothing here is a
result.

## 1. Why these three

The first family standardized exactly ONE variable — the own five-minute log close return —
against the contemporaneous cross-section, and moved anchored `held-out full` student IC by
10–22% at every horizon. It left the input side describing no OTHER own-bar attribute relative
to the universe, while the target is defined entirely inside that cross-section. Each channel
below is one more own-bar attribute expressed in universe units.

`Breadth` is deliberately EXCLUDED: the fraction of the slot's contributors printing a positive
return is a monotone function of the same slot mean the `market` channel already carries, and a
near-collinear covariate buys the head nothing while costing a full channel of width.

Channel order inside every auxiliary row is `Feature::ALL`'s order, which is the declaration
order of the enum (`features.rs:40-52`): the twelve pre-cross-section channels, then
`dispersion`, `cross-section-z`, `cross-section-rank`, `relative-volume`, `range-z`. Each
feature is a `[value, validity]` pair — `Feature::WIDTH = 2` — so `FeatureSet::ALL.channels()`
goes 16 → **22**.

## 2. The three definitions

All three read `SlotMoments` (`features.rs:328-343`), the per-slot record `CrossSection::at`
returns, and all three are `None`/`[0, 0]` at a slot that defines no market step — the same
definedness gate `Dispersion` and `CrossSectionZ` already use, so no cross-section channel can
disagree with another about what a slot is. All five are history-only: `known_future` is false
for every one of them, so a projected-future bar writes `[0, 0]`
(`features.rs:1226-1235`, asserted at `features.rs:2105-2106`).

### 2.1 `CrossSectionRank` — `cross-section-rank`

    value    = Φ⁻¹((r − 0.5) / N)
    validity = 1

where `r` is the **mid-rank** of the bar's own five-minute log close return
`ln(close_t / close_{t−5min})` inside the slot's contributing returns **with the own value
inserted**, and `N = n + 1` for `n` contributors.

Stored as `2·r` in a `u16` per valid bar (`cross_section_ranks`, `features.rs:1004-1105`).
Doubling keeps a tie's half-integer exact in an integer; a defined rank has `2·r ≥ 2`, so `0`
is an unreachable sentinel and validity needs no second array.

The write is `normal_quantile((2r − 1) · inv_ranked)` with `inv_ranked = 0.5/(n+1)` stored
per slot (`features.rs:1261-1275`, `features.rs:623-626`). `Φ⁻¹` is Acklam's rational
approximation (`features.rs:916-973`): relative error below 1.15e-9 — two orders inside f32's
own resolution — and a fixed arithmetic sequence, so it is bit-reproducible.

**Why insert own rather than locate it.** The own return follows `single_series`'s adjacency
rule (previous VALID bar exactly one interval earlier); a contribution follows the DEFINING-slot
rule. A bar can therefore carry a well-defined own return that is not one of the `n`
contributing values. Ranking it inside a set it need not belong to would reach `p = 1` exactly
and `Φ⁻¹(1) = +∞`. With own inserted, `p ∈ [1/N, 1 − 1/N]`, strictly interior on every slot
including a single-contributor one.

**Why a normal score and not the uniform rank.** A linear patch embedding consumes a normal
score directly; a uniform rank forces the head to learn `Φ⁻¹` through a piecewise-linear
projection. And it is robust exactly where `CrossSectionZ`'s ±16 clamp is not: a stale print or
an unadjusted split saturates the z at the clip and destroys the ordering information of every
bar past it, while the rank places it at the extreme plotting position and keeps the rest of the
distribution intact.

**Validity** is `CrossSectionZ`'s, exactly: the previous VALID bar must be one interval earlier,
both closes positive, and the slot must define a step (`features.rs:1085-1092`). Anything else
is the `0` sentinel and the channel emits `[0, 0]`.

### 2.2 `RelativeVolume` — `relative-volume`

    value    = clamp( (ln(volume) − μ_slot) / σ_slot , ±16 )
    validity = 1

with `μ_slot`, `σ_slot` the population moments of `ln(volume)` over the SAME contributing bars
that define the slot's market step (`features.rs:1276-1288`). Volume already exists as an own
feature (`Feature::Volume` is `ln(v_t) − ln(v_{t−1})`, an own-history innovation) but is never
cross-sectionally standardized, so the model cannot currently tell a heavy tape from a heavy
stock. The clip is `CROSS_SECTION_Z_CLIP`, shared with the z channel.

### 2.3 `RangeZ` — `range-z`

    value    = clamp( (ln((high − low)/close) − μ_slot) / σ_slot , ±16 )
    validity = 1

over the same contributing set (`features.rs:1289-1301`). This is own intrabar volatility in
universe units, and it is distinct from `Dispersion`: `Dispersion` is `ln σ` of the universe's
own return spread and carries NO own-bar information — it is the same number for every ticker
at that slot — while `RangeZ` is the bar's own range against that universe.

## 3. Degenerate-case rules, in full

| case | rule | where |
|---|---|---|
| slot defines no market step | all five cross-section channels `[0, 0]` | `features.rs:1226-1235` |
| projected-future bar | all five `[0, 0]`, unconditionally | `features.rs:1226` |
| previous valid bar not exactly one interval earlier | `cross-section-z` and `cross-section-rank` `[0, 0]`; `dispersion`, `relative-volume`, `range-z` stay live (they need no own return) | `features.rs:1241-1249`, `1085-1092` |
| single contributing ticker (`n = 1`) | rank: `N = 2`, `p = 1/2` exactly, value `0.0` with validity **1** — a defined middle, not a blank | `features.rs:623-626` |
| single contributing ticker | `relative-volume` / `range-z`: `counts < 2` → `[0, 0]` | `features.rs:459-460` |
| zero cross-sectional variance in `ln(volume)` or `ln(range)` | `[0, 0]` — a z against a degenerate cross-section is not a small number, it is no measurement | `features.rs:465` |
| zero cross-sectional variance in returns | rank: every contributor ties, `2r = n + 2`, `p = 1/2`, value `0.0`, validity **1**. Ranking is defined where the z is not; that is the point of ranking | `features.rs:1097` |
| `volume ≤ 0`, absent or non-finite | the bar enters NEITHER the slot's `ln(volume)` moments NOR its own channel: `[0, 0]`. A zero tape carries no volume information and `ln 0 = −∞` | `log_volume`, `features.rs:787-791` |
| `high == low` (single print, halt, limit lock) or `close ≤ 0` | same: excluded from the slot's `ln(range)` moments and its own channel is `[0, 0]` | `log_range`, `features.rs:793-804` |

"Excluded" means excluded, and the tests assert it as such: adding a silent-tape ticker to a
slot leaves every other ticker's `relative-volume` score bit-identical, and adding a locked bar
leaves every other `range-z` score bit-identical.

### 3.1 A conditioning bug the level channels forced out, and its fix

`E[x²] − mean²` is the streaming variance the first family uses, and it is well conditioned on
RETURNS: μ is order 1e-5, σ order 1e-3, so the subtraction cancels a fraction of one decimal
digit. It is NOT well conditioned on a LEVEL. `ln(volume)` sits near 12 and
`ln((high − low)/close)` near −4, and on a slot whose contributors all print the same tape the
difference cancels its leading digits and lands on a rounding residue instead of zero. Measured
in the corpus test: three contributors of `ln(2/100.5)` produced `σ = 8e-8` instead of `0`, and
the stored f32 mean carries its own `6e-8·|μ|`, so the quotient came out **0.787** — a pure
noise bit presented to the model as a four-fifths-of-a-sigma reading.

`LEVEL_SPREAD_FLOOR = 1e-3` (`features.rs:25-35`) is the fix: a level cross-section is a
measurement only when `σ > 1e-3·(1 + |μ|)`, applied in `SlotMoment::moments`
(`features.rs:454-466`). The cancellation floor is about `|μ|·√n·2⁻⁵²`, near `1e-6·|μ|` at
universe scale, and the f32 mean's rounding is `6e-8·|μ|`; the floor sits three orders above
both. It cannot reject a real slot: a universe of thousands whose log volumes agree to 0.1% or
whose log ranges agree to 0.5% does not exist, and their true sigmas are order one. The floor
touches only the two new level channels — `Dispersion` and `CrossSectionZ` keep
`DISPERSION_FLOOR` and are byte-unchanged.

## 4. Cost statement

Geometry, all read out of the code and the live artifacts, none of it re-derived:
`seq_len 6000, pred_len 192` (row length 6192), `patch_len 16, d_model 512, layers 8, heads 8,
ffn 2048`, batch 256, 375 origins per row, grid **1,051,104 slots**
(`long_data/bars/.timexer-cache/market-steps.bin`), **5,728** eligible tickers,
**752,901,364** valid bars in the corpus, 197,362 defining slots.

### 4.1 Parameters and arithmetic — DERIVED from the first family's measured table

The only weight whose shape is a function of the aux width is the patch embedding,
`Linear(patch_len·(CHANNELS + aux) → d_model)` (`model.rs:1651-1653`). The first family measured
`Δparams = 512 × 16 × 4 = 32,768` and `Δflops = 3 × 2 × (256 × 375) × 64 × 512 = 0.01888 TFLOP`
for its four channels; both are exactly linear in the channel count, so six channels give:

| | first family (16 aux ch) | this arm (22 aux ch) | Δ |
|---|---|---|---|
| aux channels | 16 | 22 | +6 |
| patch-embedding fan-in `patch_len × (4 + aux)` | `16 × 20 = 320` | `16 × 26 = 416` | +96 |
| trainable parameters | 27,987,243 | 28,036,395 | **+49,152 (+0.176%)** |
| TFLOP / step (`matmul_flops`) | 17.00345 | 17.03177 | **+0.02832 (+0.167%)** |
| traffic GB / step | 123.7402 | 123.8509 | +0.1107 (+0.089%) |

`Δparams = 512 × 16 × 6 = 49,152` (the bias is unchanged).
`Δflops = 3 × 2 × 96,000 × 96 × 512 = 2.8312e10 = 0.02832 TFLOP`.
No known-future channel is added, so the covariate branch, the head and every attention shape
are byte-identical. At roofline (110 TFLOP/s, 154 ms) +0.167% of FLOPs is **+0.26 ms** of step.

### 4.2 Host bytes per batch — EXACT, from `Batch::row_width`

`row_width = (context + pred_len)·(6 + aux) + 1` f32 (`corpus.rs:708-709`), 256 rows:

| aux channels | row width (f32) | bytes / batch |
|---|---|---|
| 12 (pre-cross-section control) | 111,457 | 114,131,968 |
| 16 (first family) | 136,225 | 139,494,400 |
| **22 (this arm)** | **173,377** | **177,538,048** |

**+38,043,648 bytes / batch, +27.27%** over the first family (+55.6% over the 12-channel
control). Equivalently: 6 extra f32 stores per written bar × 256 × 6192 bars = 9,510,912 stores.

**Predicted assembly cost — the load-bearing number.** Assembly is measured at 46–50 ms against
a 154 ms step and is free only while it stays below the step. Two terms:

- Byte-proportional: 46–50 ms × (177,538,048 / 139,494,400) = **58.5–63.6 ms**.
- New per-bar arithmetic: one `u16` load + one multiply + one `normal_quantile` (a fixed
  rational, ~10 fused multiply-adds and one divide; ~4.9% of bars take the tail branch with an
  extra `sqrt` and `ln`), one `ln` for volume, one divide and one `ln` for range. Call it
  ~15 ns per bar single-threaded: 1,585,152 bars × 15 ns = 23.8 ms, over the 8-thread gather
  pool (`corpus.rs:850-856`) ≈ **+3 ms**.

**Predicted total: 62–67 ms against a 154 ms step.** It does NOT approach 150 ms; the margin
after this change is still better than 2×. **Not flagged.** The next family of comparable size
would land near 80 ms and the one after that near 100 ms, so the headroom is finite but this arm
does not consume it.

### 4.3 Resident bytes for the new grid accumulators

| item | size | lifetime |
|---|---|---|
| `MarketSteps.log_volume` + `.log_range` (2 × `SlotMoment` = 2 × (f64 sums + f64 squares + u32 counts)) | `2 × 1,051,104 × 20 B = 42,044,160 B` (**42.04 MB**) | corpus load only; dropped with `MarketSteps` |
| `CrossSection`: 3 → 8 `Vec<f32>` (12 → 32 B/slot) | `1,051,104 × 32 B = 33,635,328 B`, **+21,022,080 B (+21.02 MB)** | whole run, inside `Exogenous` |
| `CorpusTicker::ranks`, one `u16` per valid bar | `752,901,364 × 2 B = 1,505,802,728 B` (**1.506 GB**) | whole run — **only when `cross-section-rank` is enabled** (`corpus.rs:1222`) |
| `cross_section_ranks` scatter buffer `values: Vec<f32>` | ≤ `4 × 752,901,364 = 3.01 GB` | during the rank build only |
| `cross_section_ranks` offsets + 16 chunk count vectors | `8.41 MB + 67.3 MB` | during the rank build only |
| `market-steps.bin` on disk | 19.49 MB → ~61.5 MB (postcard writes f64/u32 fixed-width) | on disk |
| `cross-section-ranks.bin` on disk | ~1.506 GB | on disk |

**The 1.506 GB rank array is the largest single cost of this change and the one to argue
about.** It is not an artifact of caching — the ranks must be resident at assembly time
whatever their provenance — and 2 B per valid bar is the minimum exact encoding of a mid-rank
over a 5,728-name universe (14 bits). The alternative, searching each slot's sorted segment
inside host assembly, needs the segments themselves resident (`4 × 752,901,364 = 3.01 GB`,
twice as much) plus about a dozen dependent cache misses on every one of the 1,585,152 bars in
every batch. On this host (60 GB, ~22 GB in use) it fits with room; on a smaller one it would
not, and the channel would have to go.

No tickers × slots matrix is materialized anywhere. Both new slot moments are per-slot scalars
accumulated inside the EXISTING `valid_slots` walk — the market-grid pass gains no traversal,
only two `ln`, two multiplies and six f64 accumulate-and-stores per contributing bar.

### 4.4 What the rank costs in the grid pass — stated honestly

It needs each slot's whole DISTRIBUTION, not its moments, and there is no existing ordering to
read it out of: the market grid stores sums, squares and counts, and nothing anywhere holds the
per-slot values. So a **full per-slot sort is required**, and `cross_section_ranks`
(`features.rs:1004-1105`) pays it as three extra passes over the corpus:

1. **Count** contributions per slot, 16 chunks in parallel (`features.rs:1025-1036`).
2. **Scatter** each contributing return into its slot's segment through an exclusive prefix sum
   of the per-chunk counts, so no two threads address the same element (`features.rs:1046-1063`).
3. **Sort** every segment of length > 1, in parallel across slots
   (`features.rs:1074-1076`), then one `partition_point` pair per valid bar to place the own
   return (`features.rs:1094-1097`).

Derived cost: ≤ 7.53e8 elements sorted across 197,362 segments averaging ~3,800 values,
`Σ n log₂ n ≈ 9.2e9` comparisons; plus 7.53e8 binary searches of ~12 steps each into a segment
that fits L2. Single-threaded that is tens of seconds; over the 8-thread gather pool it is a few
seconds, paid ONCE per (corpus, grid, threshold) and cached. Against a 172 s cold host load it
is small, but it is not free and it is not hidden.

## 5. Compatibility proof

Two paths, both named, both proved by
`a_checkpoint_written_before_the_cross_section_channels_still_authenticates`
(`features.rs:1364-1419`).

**Path A — an old manifest must re-serialize byte-identically.** `runner::Manifest::read`
re-serializes what it parsed and compares the SHA-256 the manifest carries, so a pinned control
checkpoint stays loadable only if absent channels stay absent on the way out. Every new
`FeatureSet` field carries `#[serde(default, skip_serializing_if = "std::ops::Not::not")]`,
exactly as the first family's two do. The test parses the literal twelve-channel control
manifest

    {"time_of_day":true,"day_of_week":true,"session_gap":true,"volume":true,"market":true,"spy":true}

asserts `channels() == 12` and `!cross_section()`, and asserts `to_string` returns that byte
string unchanged. It further asserts that a manifest enabling any ONE of the three new channels
round-trips to its own exact bytes, so a first-family checkpoint's SHA is equally safe.

**Path B — a build lacking a channel must REJECT a manifest enabling it.** `FeatureSet` carries
`#[serde(deny_unknown_fields)]`, so a binary predating a channel cannot read a manifest that
turns it on: it errors instead of silently defaulting the field off and authenticating against a
SHA that never covered it. This direction is normally undemonstrable from inside the new build,
so the test compiles the OLD shape — a local `FirstFamilyFeatureSet` with
`deny_unknown_fields` and the first family's eight fields — proves the control manifest parses
there, and proves that each of `cross_section_rank`, `relative_volume`, `range_z` makes it FAIL
while the current `FeatureSet` accepts the same bytes.

## 6. Cache scoping

`CACHE_VERSION` (`cache.rs:54`) is deliberately **NOT** bumped: it is part of the audit ledger's
version string (`cache.rs:124`), and bumping it would discard the 5,728 bar audits, which are
the corpus's dominant startup cost and have nothing to do with this change. Invalidation is done
per artifact through its own key tag instead.

| artifact | file const | key tag | line | status |
|---|---|---|---|---|
| bar audits | `AUDITS_FILE` `bar-audits.bin` | `CACHE_VERSION;data::SCHEMA` | `cache.rs:57`, `cache.rs:124` | **REUSED** — all 5,728 |
| partition boundaries | `BOUNDS_FILE` `shared-bounds.bin` | `b"shared-bounds"` | `cache.rs:58`, `cache.rs:231` | **REUSED** |
| market grid | `MARKET_FILE` `market-steps.bin` | `b"market-steps-return-volume-range-moments"` | `cache.rs:59`, `cache.rs:305-313` | **RECOMPUTED** |
| cross-section ranks | `RANKS_FILE` `cross-section-ranks.bin` | `b"cross-section-doubled-mid-ranks"` | `cache.rs:60`, `cache.rs:385-390` | **NEW** |

The market tag names the vector SET, not just the construction, so an artifact written before
the per-slot volume and range moments existed decodes to nothing here rather than being probed
for fields it does not carry. Both grid-derived keys go through the shared
`grid_key(tag, first_ts, slots, min_cross_section)` (`cache.rs:284-294`) on top of
`universe_key` (`cache.rs:187-201`), so the market grid and the ranks cannot drift apart on what
"the same grid" means, and a `--pred-len` or `--market-min-cross-section` change misses both
while still hitting every bar audit.

The ranks live in their OWN file with their OWN tag, so an arm that toggles only
`cross-section-rank` neither invalidates the market grid nor waits on it, and a control arm
never builds or reads the artifact at all (`corpus.rs:1222`). Rejection is also checked
structurally, not just by key: `MarketCache::get` refuses a stored grid unless all three
`SlotMoment` triples are exactly `slots` long (`cache.rs:323-336`), and `RankCache::get` refuses
unless every ticker's row is exactly twice that ticker's valid-bar count
(`cache.rs:399-423`) — an out-of-bounds read at every bar lookup is the wrong way to discover a
mismatch. `downstream_caches_are_rejected_when_the_corpus_contract_changes`
(`cache.rs:671-764`) pins both, including that the rank artifact and the market grid have
independent keys.

## 7. Tests

CPU-only, no CUDA, no corpus load. `torch::timexer_segment::features::tests::` — 12 tests, all
green; `::cache::tests::` — 3 green; the two touched `::corpus::tests::` — green.

- `the_rank_channel_is_the_normal_score_of_the_own_return_inside_its_slot` — three contributors
  with distinct returns: the doubled mid-ranks are pinned at `[5, 3, 7]` and the emitted values
  at `Φ⁻¹(1/4), Φ⁻¹(1/2) = 0, Φ⁻¹(3/4) = 0.6744897501960817`, computed from the plotting
  positions rather than transcribed. Then, each as its own hand-computed case: a SINGLE
  contributor (`2r = 3`, `p = 1/2`, `[0.0, 1.0]` — value zero, validity ONE); ZERO variance,
  three tied returns (`2r = 5` each, `[0.0, 1.0]`); a TIE of two inside a live slot
  (`2r = [4, 4, 7]`, `p = 3/8`, value `−0.318639363964375`); an INVALID previous bar two
  intervals back (`2r = 0` sentinel, `[0.0, 0.0]`, on a slot that DOES define a dispersion).
- `the_relative_volume_channel_is_the_slot_log_volume_z_and_a_silent_tape_is_invalid` —
  volumes 1, 4, 16 are equally spaced in `ln`, so the population z-scores are exactly
  `∓√(3/2)` and `0` whatever the spacing; asserted at 1e-5. Zero volume: `[0, 0]`, AND the other
  three read the same scores they did without that ticker, which is what "excluded" has to mean.
  Zero spread (all tapes 4, so `ln` is 1.386 and `E[x²] − mean²` cancels to a residue rather
  than a true zero — the `LEVEL_SPREAD_FLOOR` case): `[0, 0]`. Single contributing tape:
  `[0, 0]`.
- `the_range_channel_is_the_slot_log_range_z_and_a_locked_bar_is_invalid` — ranges 0.01, 0.04,
  0.16, same `∓√(3/2)`, `0`. Locked bar (`high == low`): `[0, 0]` and the others unmoved. Zero
  spread: `[0, 0]`. Single ranged bar: `[0, 0]`.
- `appending_the_cross_section_channels_leaves_every_earlier_channel_identical` — the twelve
  control channels are bit-identical between `control` and `ALL` on both a live and a projected
  bar, `all[12..] == [0.0; 10]` beyond the context, and on a live bar every one of the five
  appended validity flags is 1 with the rank pinned at `Φ⁻¹(3/4)` — so the equality is not the
  trivial one that would also hold if the channels never wrote anything.
- `feature_sets_round_trip_through_their_cli_spelling` — pins `channels() == 22`, the arm's
  exact `--features` string, and each new name's own two-channel parse.
- `pooled_epoch_keeps_delisted_tickers_and_covers_each_training_target_once` (corpus, CPU) — the
  full 20-channel row through `Corpus::load` on a scratch corpus; this is the test that caught
  the `E[x²] − mean²` conditioning bug.
- `a_cache_built_against_one_corpus_contract_is_rejected_when_the_contract_changes` (corpus,
  CPU) — its cold-load `bars_rescanned` expectation moved from `3·4000 + 2·3·4000` to
  `3·4000 + 2·3·4000 + 3·3·4000`: the rank build is three more full traversals of every source
  bar (count, scatter, place), charged to the market-grid phase because it has the same inputs
  and the same derivation (`corpus.rs:1244-1248`). A WARM load still rescans exactly zero
  records, which is what makes the new artifact worth its key.

## 8. The exact `--features` string

Arm (all five cross-section channels):

    --features time-of-day,day-of-week,session-gap,volume,market,spy,dispersion,cross-section-z,cross-section-rank,relative-volume,range-z

which is identical to `--features all` and is the CLI default, so an arm launched WITHOUT
`--features` now gets the three new channels and the 1.506 GB rank array. A control must be
explicit.

First-family control (the arm this one is measured against):

    --features time-of-day,day-of-week,session-gap,volume,market,spy,dispersion,cross-section-z

Pre-cross-section control:

    --features time-of-day,day-of-week,session-gap,volume,market,spy

Single channels are addressable by name for an ablation: `--features …,cross-section-rank`,
`…,relative-volume`, `…,range-z`.
