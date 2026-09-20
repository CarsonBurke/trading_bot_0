# CausalPatch full-split evaluation memory

Why the `held-out full` pass OOMed outside the captured step's private pool, what the
finishing path now does instead, and exactly which claims are static accounting rather than
measurement. **No GPU work was run for this report**: no training, no evaluation, no
benchmark, no probe. Everything below is source accounting, CPU arithmetic, or a CPU-only
test. Configuration: `pred_len 192`, held-out split **433,303 origins**, `eval_batch_size`
per `TrainArgs`. All sizes are binary MiB.

Scope: `trading_bots/src/torch/timexer_segment/runner.rs`, `Scorer` and its finishing path
only. The training step, the captured graph, `teacher.rs` and every other file are untouched.

## 1. Diagnosis

`Scorer` accumulates eight persistent `[origins, pred_len]` fp32 banks during the pass and
consumes them at `finish`. The banks are affordable. The *finishing path* was not: both
`Scorer::trading` and the tail-trimmed block in `Scorer::finish` computed every diagnostic
densely over `[origins, pred_len]`, so each intermediate was another 317.361 MiB fp32 array
and each rank array was 634.721 MiB of Int64.

The failing request is exact, not approximate. Spearman needs global ranks:

```rust
(t + (1.0_f64 - m) * SENTINEL).argsort(0, false).argsort(0, false)
```

Each `argsort` output is `[433303, 192]` Int64 = 433,303 x 192 x 8 B = 665,553,408 B =
**634.721 MiB**, and the caching allocator rounds a large-block growth request up to its
2 MiB granularity: 634.721 -> **636.00 MiB**, which is the size the ten dead arms reported.
Two such arrays are live at once (the second `argsort` cannot free its input), on top of
`mid_target`, `mid_forecast`, `delay_mask`, `delayed_target`, `delayed_forecast` and `rf`.

## 2. Fix

One horizon at a time, which is already this module's convention - `utility.rs::evaluate`
loops over its horizon grid and reduces `[origins]` columns. The finishing path now follows
that structure:

- `Scorer::trading` is three lines: `moments()` -> `curve()` -> `portfolio()`.
- **`Scorer::moments`** (new) loops `j in 0..pred_len`, takes `select(1, j)` columns of the
  banks, and produces the same 34 column reductions as before. It stacks the 34 scalars into
  a `[34]` f64 row per horizon, then one `Tensor::stack(&rows, 1)` -> `[34, pred_len]` and
  **one** host transfer, so the host layout `flat[row * pred_len + j]` is unchanged.
  Per-horizon `topk` uses that horizon's own `k` instead of `widest` across all horizons, so
  the decile threshold is a `[k]` slice rather than a `[widest, pred_len]` block.
- **`Scorer::curve`** (new) is the pre-existing host algebra, moved verbatim, now reading the
  moment matrix it is handed. `TRADING_MOMENTS = 34` is asserted against the matrix length.
- The tail-trimmed reduction in `finish` loops the same way: `tail_counts` crosses to the
  host once as `Vec<i64>`, and each horizon's `|close|`, threshold and keep mask are
  `[origins]`.
- Nothing accumulates a per-horizon `[origins]` tensor. The only tensors that outlive an
  iteration are the `pred_len` scalar rows: 192 x 34 x 8 B = **52 KB** total.
- `score` now drops the resident upload buffer and the prefetcher before `finish`, after
  `total_ms` is taken. Those hold the 139.49 MB batch block, which the finishing path can
  otherwise only reuse by splitting it while it is still owned.
- Each iteration takes `.contiguous()` copies of that horizon's `bar_valid`, `bar_forecast`
  and `bar_target` columns (1.653 MiB each, plus three hoisted entry columns): ~15 reductions
  read each column, and a strided view walks the bank at a 768 B stride, one sector per
  element. The copy is bit-exact (section 4).

The eight persistent banks stay `[origins, pred_len]`, as specified: they are filled during
the pass and consumed at finish, and 2,538.88 MiB is an acceptable peak.

## 3. Every population-sized allocation, before and after

`O = 433,303`, `H = 192`. Unit sizes: `[O, H]` fp32 = 317.361 MiB, `[O, H]` Int64 =
634.721 MiB, `[O, H]` bool = 79.340 MiB; `[O]` fp32 = 1.6529 MiB, `[O]` f64/Int64 =
3.3058 MiB, `[O]` bool = 0.4132 MiB.

### 3.1 Persistent banks (unchanged)

| bank | shape | dtype | MiB |
| --- | --- | --- | --- |
| `bar_squared`, `bar_persistence`, `bar_forecast`, `bar_target`, `bar_valid`, `bar_policy_forecast`, `bar_log_scale`, `bar_raw_target` | 8x `[O, H]` | fp32 | 8 x 317.361 = **2,538.88** |
| `window_mid`, `window_sigma`, `window_entry_log` | 3x `[O]` | fp32 | 4.96 |
| `groups` | `[O]` | Int64 | 3.31 |

### 3.2 `Scorer::trading` / `Scorer::moments`

Per-tensor size, before (dense, once per pass) and after (per horizon, freed each iteration):

| tensor | before shape/dtype | before MiB | after shape/dtype | after MiB |
| --- | --- | --- | --- | --- |
| `sign`/`abs`/`bet`/`hit` per `rate` call (5 calls, ~6 temporaries each) | `[O, H]` fp32 | 317.361 each | `[O]` fp32 | 1.6529 each |
| `gt(0.)` predicate inside `rate` | `[O, H]` bool | 79.340 | `[O]` bool | 0.4132 |
| `mid_target`, `mid_forecast` (+ 2 subtraction temporaries) | `[O, H]` fp32 | 4 x 317.361 | `[O]` fp32 | 4 x 1.6529 |
| `delay_mask`, `delayed_target`, `delayed_forecast` (+ 2) | `[O, H]` fp32 | 5 x 317.361 | `[O]` fp32 | 5 x 1.6529 |
| sentinel-shifted rank input (x2, one per `ranks` call) | `[O, H]` fp32 | 2 x 317.361 | `[O]` fp32 | 2 x 1.6529 |
| **`argsort` output, first pass (x2)** | `[O, H]` **Int64** | **2 x 634.721** | `[O]` Int64 | 2 x 3.3058 |
| **`argsort` output, second pass (x2)** | `[O, H]` **Int64** | **2 x 634.721** | `[O]` Int64 | 2 x 3.3058 |
| `rf`, `ry` (fp32 cast x mask, both live) | `[O, H]` fp32 | 2 x 317.361 | `[O]` fp32 | 2 x 1.6529 |
| `g`, `magnitude`, `side` (+ 2) | `[O, H]` fp32 | 5 x 317.361 | `[O]` fp32 | 5 x 1.6529 |
| `high`, `low` (+ 4 temporaries) | `[O, H]` fp32 | 6 x 317.361 | `[O]` fp32 | 6 x 1.6529 |
| `topk` values, top and bottom | `[O/10, H]` fp32 | 2 x 31.736 | `[k]` fp32 | 2 x 0.1653 |
| `topk` indices, top and bottom (allocated, discarded) | `[O/10, H]` Int64 | 2 x 63.472 | `[k]` Int64 | 2 x 0.3306 |
| threshold predicates | `[O, H]` bool | 2 x 79.340 | `[O]` bool | 2 x 0.4132 |
| `top_mask`, `bottom_mask` | `[O, H]` fp32 | 2 x 317.361 | `[O]` fp32 | 2 x 1.6529 |
| squares/products fed to the 34 reductions (`f^2`, `y^2`, `f*y` twice, `(y-f)*m`, `mid_target^2`, delayed pair, `rf^2`, `ry^2`, `rf*ry`, `side*selected` x2) | ~15x `[O, H]` fp32 | ~15 x 317.361 | `[O]` fp32 | ~15 x 1.6529 |
| assembled moment matrix | `[34, H]` f64 | 0.05 | `[34, H]` f64 | 0.05 |
| cross-section scatters (`gn`, `gf`, `gy`, `gff`, `gyy`, `gfy`, `ic`, `usable`, ...) | `[G, H]` fp32/f64 | G-sized, not population-sized | `[G]` | same |

### 3.3 `Scorer::finish`, tail-trimmed sums

| tensor | before shape/dtype | before MiB | after shape/dtype | after MiB |
| --- | --- | --- | --- | --- |
| `bar_close` (`\|y\|*m + m - 1`, 3 temporaries) | `[O, H]` fp32 | 3 x 317.361 | `[O]` fp32 | 3 x 1.6529 |
| `topk` values / indices | `[O/100, H]` fp32 + Int64 | 3.17 + 6.35 | `[k]` | 0.017 + 0.033 |
| `keep` predicate + fp32 cast | `[O, H]` bool + fp32 | 79.340 + 317.361 | `[O]` | 0.4132 + 1.6529 |
| `bar_squared * keep`, `bar_persistence * keep` | 2x `[O, H]` fp32 | 2 x 317.361 | `[O]` fp32 | 2 x 1.6529 |
| `window_squared`, `window_persistence` (dim-1 reductions, unchanged) | 2x `[O]` f64 | 6.61 | same | 6.61 |

### 3.4 Predicted peak

Worst simultaneously-live set, dense: the eight banks (2,538.88) + `mid_target`,
`mid_forecast`, `delay_mask`, `delayed_target`, `delayed_forecast`, `rf` and the
sentinel-shifted input to `ranks(y)` (7 x 317.361 = 2,221.52) + the two Int64 `argsort`
arrays of the second `ranks` call (1,269.44) = **~5,890 MiB ~= 5.75 GiB** for the trading
reduction alone, before `utility::evaluate`, the trimmed block, the 139.49 MB resident batch
and the allocator fragmentation that turns a 636.00 MiB *growth* request into a failure. That
is consistent with the ~8.6 GiB of non-pool allocation measured on the dead arms.

Worst simultaneously-live set, per-horizon: the eight banks (2,538.88) + ~30 `[O]` fp32
temporaries (49.6) + 2 `[O]` Int64 rank arrays (6.6) + the `[O]` f64 window rows and
`utility`'s `[O]` f64 working set (~20 x 3.31 = 66) = **~2,660 MiB ~= 2.60 GiB**, i.e. within
the ~3 GiB target and dominated by the banks. `utility::evaluate` was already per-horizon and
is unchanged; the batch buffer is now released before the finishing path runs.

## 4. Bit-reproducibility

**Which reductions were reordered: none within a horizon.** Every reduction still runs over
the same origins, in ascending origin order, with the same f64 accumulator and the same
elementwise algebra feeding it. Three things changed and each is addressed:

1. **Column reduction shape.** A horizon's sum used to be one lane of
   `sum_dim_intlist([0], Kind::Double)` over `[O, H]`; it is now `sum(Kind::Double)` over
   `[O]`. The summand set and its order are identical, but the two kernels use different
   accumulator trees, so the *result* is only guaranteed identical when the tree happens to
   coincide. Measured on CPU with torch 2.14: at test scale (384 x 6, and 437 x 7) the dense
   dim-0 reduction and the per-column reduction are **exactly equal**, which is what the new
   test asserts; at production scale (433,303 x 192) they differ by a maximum **relative
   8.5e-17**, below the 4.441e-16 repeated-shape reordering floor already recorded for this
   pipeline. This is the one unavoidable perturbation, it is a last-bit effect on f64 sums,
   and no reported statistic carries more than ~4 significant digits.
2. **`.contiguous()` on a horizon column.** Measured on CPU at both 384 x 6 and
   433,303 x 192: a reduction over the strided column view and over its contiguous copy are
   **bit-identical**, so the coalescing copy costs nothing in reproducibility.
3. **`argsort`/`topk`/`index_add` launch shape.** Ranks are the count of strictly smaller
   entries, so valid entries hold exactly `0..n_h-1` in both forms (the sentinel keeps every
   invalid entry above every valid one). Tie-breaking among *exactly equal* valid entries is
   unspecified in both the dense and the per-horizon form - neither uses a stable sort - and
   was not made stable here, because that would change reported ranks. `topk`'s k-th largest
   *value* is uniquely determined by the multiset, so taking it from a `[k]` slice instead of
   gathering it out of a `[widest, H]` block is exact. The cross-section rows come from
   `index_add` scatters, whose CUDA accumulation order is set by atomic scheduling and is not
   fixed by shape in either form; this file already records a 1e-12 scatter reordering floor
   (`FP64_SCATTER_TOLERANCE`) for exactly that reason.

Nothing was moved between f32 and f64: the banks stay fp32, every reduction still accumulates
in `Kind::Double`, and the host algebra is untouched (`Scorer::curve` is the old code moved,
not rewritten).

**Determinism itself is unaffected.** At fixed shape the new path launches the same kernels
with the same shapes every pass, so "bit-reproducible at fixed shape" - the property
checkpoint selection and step-matched comparison depend on - survives by construction.

### Proof, CPU only

`torch::timexer_segment::runner::tests::per_horizon_finishing_path_is_bit_identical_to_the_dense_reduction`
builds a 384-origin, 6-horizon, 16-cross-section synthetic population on CPU tensors (no
model, no corpus, no CUDA), accumulates it in three uneven batches, and asserts:

- the 34 x 6 moment matrix is bit-identical (`f64::to_bits`) to `dense_moments`, the previous
  whole-population reduction kept verbatim in the test as the reference;
- all 28 `TradingCurve` vectors from the shipped `trading()` path are bit-identical to the
  curves the same host algebra produces from the dense moment matrix;
- `finish()`'s `trimmed_mse_ratio` is bit-identical to the dense tail-trimmed reduction.

```
running 1 test
test torch::timexer_segment::runner::tests::per_horizon_finishing_path_is_bit_identical_to_the_dense_reduction ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1154 filtered out; finished in 0.13s
```

The test has teeth: widening the conviction decile by one name (`k + 1`) fails it at
`moment row 26 at h=1: 32 != 31`. The seven pre-existing CPU scorer tests
(`tensor_scorer_matches_scalar_reference`, `trading_statistics_match_scalar_reference`,
`the_optimal_gain_reproduces_its_analytic_value_and_undefined_statistics_stay_nan`,
`zero_forecast_scores_exactly_persistence_at_every_horizon`,
`a_positive_per_horizon_gain_moves_mse_and_cannot_move_a_rank_statistic`,
`the_selection_scalar_is_the_training_weighted_held_out_nll`,
`scorer_utility_uses_raw_next_open_payoffs_and_close_uncertainty`) all pass unchanged; two of
them check the trading family against an independent host-side scalar reference.

## 5. `empty_cache` placement

Unchanged, and no new call was added:

- `runner.rs:2358`, before the interval evaluation `score(...)` whose `selected` is
  `corpus.validation_refs` at epoch completion: one `empty_cache()` then
  `cuda_memory(true)` to reset the peak.
- `runner.rs:2445`, before the final mid-epoch full pass: same pair.

**The per-horizon loop must not call it.** Every iteration allocates the same shapes as the
previous one, so the caching allocator hands back the same blocks with zero `cudaMalloc`;
`empty_cache()` inside the loop would return them to the driver and force 192 rounds of fresh
device allocation. It would also not move the reported number: `benchmark::cuda_memory` reads
`max_memory_allocated`, which `empty_cache` does not affect (it only lowers
`max_memory_reserved`). What the finishing path actually needed was for the batch buffers to
be *dropped* - now done explicitly in `score` before `finish` - not for the cache to be
flushed.

## 6. Both full-split branches still run a full pass

Both call sites are untouched and both still score all of `corpus.validation_refs`:

- `runner.rs:2345-2360`: `selected = &corpus.validation_refs` when `epoch_complete`, then
  `score(&corpus, &model, selected, args.eval_batch_size, device)?`.
- `runner.rs:2444-2454`: `if ended.is_some() && !epoch_complete`, then
  `score(&corpus, &model, &corpus.validation_refs, ...)`.

The bug was the allocation inside the pass, not the decision to run it.

## 7. Not verified

Everything that needs a GPU. Specifically:

- **The actual peak.** Section 3.4 is arithmetic over shapes and dtypes, not a
  `max_memory_allocated` reading. The ~2.60 GiB prediction and the ~5.75 GiB "before" figure
  are static accounting; the only measured memory number in this report is the 636.00 MiB
  request from the dead arms, which the 2 MiB rounding of 634.721 MiB reproduces exactly.
- **That the full split no longer OOMs.** Unproven until an arm runs it.
- **CUDA bit-identity at production shape.** The reordering bound in section 4 was measured
  on CPU. CUDA reduction trees differ from CPU's, so the perturbation there is stated as
  "same class, last-bit" and not as a measured figure; it is expected to sit at or below the
  4.441e-16 repeated-shape floor this pipeline already records.
- **Wall-clock cost of the loop.** Per-horizon launches replace 34 dense kernels with 192 x
  ~50 small ones, and the `.contiguous()` copies trade 5 MiB per iteration for coalesced
  reads. Estimated well under a second against a ~100 s full pass, but not timed.
- No formatter, linter, broad test filter or project-wide suite was run: scoped
  `cargo check -p trading_bot_0 --tests` and the eight narrow CPU test filters above only.
