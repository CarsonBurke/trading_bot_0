# TimeXer segment startup cost

Owner: `StartupCost`. Scope: process start to optimizer step one — corpus load, eligible-universe
selection, chronological partition authentication, row/origin enumeration, market grid, and the
device-side run startup in `runner.rs`. Every number below is DEMONSTRATED on the real corpus
unless tagged HYPOTHESIS.

## Corpus size (DEMONSTRATED)

`long_data/bars`: **5,728 files, 783,074,510 records, 28.19 GB**. The 470,946,393 in the brief is
the post-quarantine VALID subset that training reads; startup traverses the raw 783M.

## Before: the breakdown, and the dominant cost

`Corpus::load` was **172.1 s** and it was **four full passes over all 783 M records**:

| # | pass | what it computed | file:line (pre-change) | order |
|---|------|------------------|------------------------|-------|
| P1 | `shared_bounds` | per-bar on-grid check, `valid_ohlc`, set a bit in the occupancy bitmap | `corpus.rs:918` | O(bars) |
| P2 | `audit_bars` per ticker | SHA-256 over the whole record region + the quarantine index list | `data.rs:50` | O(bars) |
| P3 | `market_steps` count | each slot's cross-section population | `features.rs:458` | O(bars) |
| P4 | `market_steps` returns | log return between consecutive DEFINING slots | `features.rs:458` | O(bars) |

Top two costs, measured: **`market_steps` (P3+P4) 35.9 s** and **`audit_bars` (P2) 37.3 s**, both
strictly O(bars). Everything O(tickers) or O(rows) was noise by comparison: per-ticker contracts
0.85 s, row and origin enumeration 15 ms, fingerprints 9 ms.

The brief's prior was **CONFIRMED in substance and wrong in one detail**: the ~470 M-bar
authentication was indeed hashing every bar on every run, but it was only 22% of the cost. The
market grid was the same size and nobody had named it.

Two further costs only became visible after the first fold, because they had been hidden inside
`directory scan` and inside an unattributed gap:

- **the 5,728 file opens**, serial: 18.8 s of a 29 s warm run. O(tickers), latency-bound.
- **the eligibility scan**, `corpus.rs` `audited.retain` — one `index_at_or_after` per ticker,
  a binary search over a mapped bar file, i.e. **~17 random page faults into 28 GB that no
  readahead predicts**. Serial, and it probed all three boundaries while reading only the first.
  **20.6 s of a 29 s warm run** — more than the entire cached corpus load it preceded. O(tickers)
  in count, O(log bars) in faults each.

## What changed

**1. Fold before caching.** P1 and P2 were two traversals of the same bytes in the same order for
no reason but the order they were written in: an audit depends on the bar bytes alone, and only
three `partition_point` probes need the boundaries the bitmap produces. They are now one streaming
pass (`corpus.rs` `scan`), which speeds up the *first* run too — that is why folding came first
and caching second, per the instruction. P3+P4 cannot join them: "defining slot" is a property of
the completed population vector, so the second walk needs the first walk's output. That layer is
cached rather than folded.

**2. Three authenticated artifacts** under `long_data/bars/.timexer-cache/`, 11.7 MB total:

| artifact | content | key |
|---|---|---|
| `bar-audits.bin` (575 KB) | per-ticker SHA-256 + quarantine list | `data::SCHEMA` + per-file `BarIdentity` |
| `shared-bounds.bin` (48 KB) | 3 partition timestamps **+ each ticker's 3 raw bar indices at them** | `corpus::SCHEMA` + every scanned ticker's `(symbol, fingerprint)` |
| `market-steps.bin` (11 MB) | population/sums/counts per grid slot | as above, over ELIGIBLE tickers, + grid origin, slot count, cross-section floor |

**Identity sufficient to make this safe.** Two layers, deliberately different:

- The audit ledger is keyed on **inode identity**: `BarIdentity { device, inode, size, mtime_ns,
  ctime_ns }`, captured by `fstat` on the very descriptor the mmap was made from
  (`shared/src/bars.rs`), so it names the bytes a `BarFile` exposes, not whatever the path
  resolves to later. `ctime_ns` is what makes it safe rather than merely likely: `append_bars`
  writes in place and could in principle preserve size and mtime, but not ctime. A stale pairing
  therefore cannot survive: any write to a file moves ctime, any replacement moves the inode, any
  truncation moves size.
- Everything downstream is keyed on **content** — the SHA-256 digests themselves, plus the
  derivation schema string and the geometric parameters that change the meaning of the stored
  numbers. Inode identity is deliberately *not* used below the ledger, so a corpus restored from
  backup rehashes once and then hits every layer beneath.

The digest itself is bandwidth-bound and unavoidable on a cold run: **7.3 s** for 783 M records
with the page cache hot, 37.3 s from disk. That is exactly the cost that should never be repeated,
which is the whole argument for persisting it.

**3. The per-ticker partition edges are cached with the bounds** (the change that mattered most on
a warm run). They are three binary searches per ticker and every consumer needs them: the
eligibility filter needs edge 0, `filtered_contract` needs all three. They are a pure function of
(that ticker's bytes, the boundaries), and both are already inside the bounds key, so they cannot
outlive what they describe. `filtered_contract` now takes them as an argument and touches no bar
but the last. Eligibility scan **5,858 ms → 4.8 ms**; per-ticker contracts **3,354 ms → 3.4 ms**.

**4. A separate IO pool for the opens.** Startup's opens and probes are latency-bound — a thread
parked on a page fault — so the useful width is device queue depth, not the core count that sizes
the gather pool. `io_pool` is 4x wider and dies with `load`; the gather pool `LoaderPerf` owns is
untouched. Directory scan **18.8 s serial → 2.1 s**.

## After (DEMONSTRATED)

Reproduce with:

```
./torch-env.sh cargo build --release -p trading_bot_0 --bin trading_bot_0
rm -rf long_data/bars/.timexer-cache                       # cold
./torch-env.sh bash -c 'exec ./target/release/trading_bot_0 audit-timexer-segment-loader --rounds 1 --batch-size 8'
./torch-env.sh bash -c 'exec ./target/release/trading_bot_0 audit-timexer-segment-loader --rounds 1 --batch-size 8'   # warm
```

| phase (ms) | before | cold, after | warm, after |
|---|---|---|---|
| directory scan and header open | (inside 172.1 total) | 2078.8 | **684.8** |
| bar ledger authentication (inode identity) | — | 31.6 | 2.5 |
| cached artifact read and decode | — | 3.9 | 9.6 |
| bar audit (grid checks and SHA-256) | 37271.9 | 7296.0 | — |
| shared partition boundaries | fused into the audit | 1255.3 | — |
| ticker eligibility scan | 20600 | 20.0 | 4.8 |
| per-ticker contracts | 848.5 | 80.6 | 3.4 |
| market grid | 35851.8 | 13152.4 | — |
| exogenous series and SPY | 36.9 | 21.0 | 13.6 |
| row and origin enumeration | 15.0 | 9.9 | 12.0 |
| corpus fingerprints | 9.3 | 9.3 | 8.3 |
| cache write | — | 20.7 | 0.0 |
| **corpus load total** | **172,100** | **23,980** | **739** |

**Warm corpus load: 172.1 s → 0.74 s, a 233x reduction.** Process start to the first work after
load is **~0.8 s**. The target was under 30 s; the corpus half of startup is now under one second
and the remaining pre-step budget is entirely device-side.

**First run after this change pays the build once: 24.0 s** with the page cache hot, **105.6 s**
from cold disk. It is dominated by two irreducible O(bars) passes — the SHA-256 (7.3 s hot) and
the market grid's two walks (13.2 s hot). An ingest that touches one ticker re-pays the market
grid and the boundaries in full, because a partition boundary is a quantile of the union and no
incremental update of a quantile exists; it re-pays the audit for the touched ticker only.

**Irreducible floor.** mmap setup and header probes are ~0.7 s for 5,728 files at queue depth 32
and will not go much lower on this device. CUDA context creation, allocator warmup and graph
capture are device-side, measured through the new `cuda_context_ms` / `model_build_ms` /
`held_out_draw_ms` phases, and are unchanged by this work — I hold no GPU lease, so those read
`-` in every measurement above. HYPOTHESIS: they are 3-8 s combined, so a full training run's
pre-first-step startup should now be under 10 s cold-artifact and under 5 s warm.

## Reporting

Every phase above is emitted as a series. `cache::LoadTiming::phases()` returns
`(label, milliseconds)` in load order and `LoaderPerf`'s chart writer takes all of them except the
trailing rescanned-bar counter, so a phase added here appears on the chart with no further edit.
Registered as **`timexer_segment_startup`** in `shared/src/report.rs:102` (inside
`TIMEXER_SEGMENT_REPORT_BASES`) and written at `reports.rs:1204`; `tui/src/main.rs:428`
`meta_chart_bases` extends from that same slice, so the TUI side is registered by construction and
`tui/src/main.rs:1218`'s bidirectional test covers it. Its own base rather than rows inside
`timexer_segment_timing`, because a 10^5 ms constant on a 2-180 ms per-step axis flattens every
series that panel exists to show.

Two design rules the labels enforce:

- **A phase that did not run reads NaN, never 0.** Zero would assert "this work took no time",
  which is the exact class of silent lie that already cost this project a trading verdict. NaN
  means "did not happen"; 0 means "happened instantly".
- **`bar ledger authentication` and `cached artifact read and decode` are always finite**, warm or
  cold. They are what a warm run pays *instead of* the rebuild, so leaving them NaN would hide the
  cost of the cache itself. The phases also sum to the total by construction — the fused pass
  charges the rebuild line when it rescanned and the cache line when it did not — so a gap in the
  chart is a real gap. That mattered: my first warm measurement had 18.3 s of a 29.1 s total
  sitting in no phase at all.

## Correctness: the rejection tests

Caching startup is only defensible if a cache built against one corpus contract is REFUSED, not
silently reused. Three tests, all green:

- `cache::tests::a_bar_audit_is_rejected_when_the_inode_state_moves` — a ctime move alone must
  miss; an unchanged inode must hit; a derivation-schema change discards the whole ledger.
- `cache::tests::downstream_caches_are_rejected_when_the_corpus_contract_changes` — schema change,
  dropped ticker, changed fingerprint, shifted grid origin, different slot count and a different
  cross-section floor each reject. Extended here: an edge list that does not cover the universe is
  refused rather than indexed into.
- `corpus::tests::a_cache_built_against_one_corpus_contract_is_rejected_when_the_contract_changes`
  — the integration test, on a real `Corpus::load` against a scratch corpus. A cold load audits
  every ticker and rescans; an untouched warm load rehashes nothing; rewriting one ticker's bytes
  rehashes exactly that ticker; a changed `pred_len` (different purge → different eligible
  universe) and a changed cross-section floor (different market grid) each force a rebuild while
  every byte-level audit still stands.

`cargo check -p trading_bot_0 --tests`: 0 errors. `cargo test -p trading_bot_0 timexer_segment`:
**105 passed, 0 failed.**

## Ownership split agreed with `LoaderPerf`

- **`corpus.rs`**: mine is the load and authentication path — the directory scan, the fused
  audit/occupancy pass, the caches, eligibility, contracts, the grid and origin construction, and
  everything up to the returned `Corpus`. `LoaderPerf` owns the per-batch host path below it —
  batch assembly, feature construction, prefetch — plus `benchmark.rs` and `compute.rs`.
- **`reports.rs`** is split by function, not by file: `LoaderPerf` owns the timing/hardware writers
  including the `timexer_segment_startup` chart; `SignalHistory` owns every other writer. I emit
  `LoadTiming` and hand the base name over rather than writing the chart myself.
- **Log prices: `LoaderPerf` builds them in process; I do NOT build a memory-mapped artifact.**
  Settled on their argument, which is correct and which I record here because it refutes the
  brief's prior: f32 storage would change the last bits of `ln p − ln anchor` relative to the f64
  computation the model is trained against, and f64 storage is 15.1 GB against ~20 GB available —
  it would also add mmap page traffic to a path whose whole problem is page traffic. In-process
  precomputation costs nothing at startup and is bit-identical. Built once, by them.
