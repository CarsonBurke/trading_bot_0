# Found Issues

_Generated on 2026-02-26_

## Summary

- **Total issues found:** 10
- **Critical:** 0 | **High:** 1 | **Medium:** 4 | **Low:** 5

---

## Issues

### Issue 1: Future earnings data leaks into observations (look-ahead bias)

- **Domain:** torch/env
- **Severity:** High
- **File(s):** `trading_bots/src/torch/env/earnings.rs:76-82`
- **Details:** When bar dates start before the first earnings report, `report_idx` starts at 0 and `reports[0]` is used immediately without checking whether its date precedes the current bar date. This means bars that predate all known reports use future earnings data (revenue_growth, eps, eps_surprise) the model shouldn't have access to yet. This is a look-ahead bias that corrupts the training signal.
- **Suggested fix:** Before using `reports[report_idx]` on line 82, check `reports[report_idx].date <= *bar_date`. If not, skip (leave zeros) for that bar's indicators.
- **Status:** Pending

---

### Issue 2: Cache validation only checks length, not data identity

- **Domain:** torch/env
- **Severity:** Medium
- **File(s):** `trading_bots/src/torch/env/earnings.rs:35`
- **Details:** The cache hit condition `cached.eps.len() == prices.len()` only checks that cached indicators have the same length as the current prices slice. If `get_or_compute` is called for the same ticker with different date ranges but equal length, stale cached results are silently returned.
- **Suggested fix:** Include first/last `bar_date` and length as part of the cache key or validation check.
- **Status:** Pending

---

### Issue 3: Non-standard Clone generates new UUID, causing elitism logic bug

- **Domain:** agent
- **Severity:** Medium
- **File(s):** `trading_bots/src/agent/mod.rs:32-40`
- **Details:** The custom `Clone` for `Agent` creates a new UUID on every clone. In `train.rs:81`, the best agent is cloned for elitism preservation, getting a new UUID. The original (with its old UUID) remains in the map and gets mutated at lines 101-103, then the unmutated clone is inserted under the new UUID at line 106. Result: both a mutated original and an unmutated copy coexist, growing the population by 1 each generation.
- **Suggested fix:** Remove the original from the map before re-inserting the clone (`agents.remove(&gen_best_agent_id)` before line 106), or use a dedicated `spawn_offspring()` method instead of overriding `Clone`.
- **Status:** Pending

---

### Issue 4: TICKERS global index coupling can panic out-of-bounds

- **Domain:** history
- **Severity:** Medium
- **File(s):** `trading_bots/src/history/episode_tickers_separate.rs:37`
- **Details:** `record()` indexes the global `TICKERS` constant with the loop variable `ticker_index`, but the struct is sized by an arbitrary `ticker_count` parameter and `mapped_historical` is passed from the caller. If `mapped_historical.len() > TICKERS.len()` (currently 7), this panics at runtime with index-out-of-bounds.
- **Suggested fix:** Pass ticker names as a parameter to `record()` instead of relying on the global `TICKERS` constant, matching `episode_tickers_combined`'s approach.
- **Status:** Pending

---

### Issue 5: Unwrap panics on empty data in ticker stats methods

- **Domain:** history
- **Severity:** Medium
- **File(s):** `trading_bots/src/history/episode_tickers_separate.rs:80-92`
- **Details:** `ticker_lowest_final_assets()` calls `.last().unwrap()` on inner `Vec`s that could be empty if no steps were recorded. `min_by(...).unwrap()` panics if ticker_count is 0. `partial_cmp(...).unwrap()` panics on NaN values. These methods are called from `meta_tickers_separate::MetaHistory::record` at episode end, so a panic here crashes training.
- **Suggested fix:** Return `Option` types, or use `f64::total_cmp` and guard against empty vectors.
- **Status:** Pending

---

### Issue 6: Observation type mismatch between reset() and step()

- **Domain:** torch
- **Severity:** Low
- **File(s):** `trading_bots/src/torch/vec_gym_env.rs:49-65`
- **Details:** `reset()` extracts observations as `Vec<f32>` (Python numeric conversion), while `step()` reads the raw buffer as `Vec<u8>` then casts to Float. For float32 environments, `step()` would read 4-byte float representations as 4 separate u8 values, producing 4x the elements and a shape mismatch or garbage data. File is currently dead code (not referenced by any module).
- **Suggested fix:** Use the same extraction method in both paths, or remove this dead file.
- **Status:** Pending

---

### Issue 7: Inaccurate date difference calculation

- **Domain:** torch/env
- **Severity:** Low
- **File(s):** `trading_bots/src/torch/env/earnings.rs:113-128`
- **Details:** `date_diff_days` uses `year * 365 + month * 30 + day`, where 12 months = 360 days instead of 365. A 1-day difference across a year boundary computes as ~5 days. The result is divided by 90 and clamped to [0,1], dampening the error, but the feature still encodes incorrect temporal distance to the next earnings report.
- **Suggested fix:** Use `chrono::NaiveDate` for proper date arithmetic, or at minimum use `(m - 1) * 30` to avoid double-counting.
- **Status:** Pending

---

### Issue 8: TOCTOU race in earnings cache lookup

- **Domain:** torch/env
- **Severity:** Low
- **File(s):** `trading_bots/src/torch/env/earnings.rs:32-49`
- **Details:** `get_or_compute` locks the mutex to check the cache, drops the lock, computes the result, then re-locks to insert. Two concurrent threads could both miss the cache and compute redundantly. Since `compute` is deterministic, this wastes CPU but doesn't corrupt data. Environment construction appears single-threaded in practice.
- **Suggested fix:** Hold the lock across the full check-compute-insert sequence, or use an entry API pattern.
- **Status:** Pending

---

### Issue 9: Uniform weight clamping allows semantically invalid negative values

- **Domain:** agent
- **Severity:** Low
- **File(s):** `trading_bots/src/agent/mod.rs:48-57`
- **Details:** `mutate()` clamps all weights to [-1.0, 1.0], but weights like EMA alphas (valid in (0,1)), `SellPercent`, and `BuyPercent` become nonsensical when negative. Negative EMA alpha causes divergent/oscillating values. Negative sell/buy percentages could cause incorrect portfolio state. The evolutionary algorithm will select against these, but it wastes compute.
- **Suggested fix:** Use per-weight clamping ranges, or clamp all to [0.0, 1.0] since no defined weights have meaningful negative interpretations.
- **Status:** Pending

---

### Issue 10: Chart generation errors silently discarded

- **Domain:** history
- **Severity:** Low
- **File(s):** `trading_bots/src/history/episode_tickers_separate.rs:43-61`
- **Details:** All three chart function results (`buy_sell_chart`, `reward_chart`, `assets_chart`) are silently discarded with `let _ = ...`. If the filesystem fills up or data is in an unexpected state, training continues with no indication that charts are missing or stale.
- **Suggested fix:** Log a warning on chart generation failure via `eprintln!` or the project's logging mechanism.
- **Status:** Pending

---
