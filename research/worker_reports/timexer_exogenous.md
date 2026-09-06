# TimeXer exogenous variates (worker Exogenous)

Branch `timexer-exog`, commit `b8ce968e` (base bfd3404c). Worktree `worktrees/timexer-exog`.
`git diff --stat bfd3404c..HEAD`: 10 files, +775/-239 — new `features.rs` (520), corpus.rs (198), data.rs (122, mostly deletions), model.rs (84), campaign script (41), docs (25), runner.rs (15), benchmark.rs (6), dataset.rs (2), mod.rs (1).

## What changed
- `--volume-features` is gone (no alias). `--features <list>` on `ModelConfig` takes a comma list of `time-of-day,day-of-week,session-gap,volume,market,spy`, or `all` (default) / `none` (no auxiliary tensor). `FeatureSet` (features.rs) is a struct of six explicit booleans; serde `deny_unknown_fields`, no `#[serde(default)]` anywhere → manifests/contracts without `features` are rejected. Removed the compat defaults on `DataContract.common_context` and the legacy-config test in model.rs that pinned missing-field fallbacks.
- Channel order (each feature = 2 channels, fixed `Feature::ALL` order):
  1. `time-of-day`: sin, cos of America/New_York minute-of-day (bars are UTC ms; ET wall clock via the existing `et_offset_secs` table in `torch/dataset.rs`, made `pub(crate)`; 288-entry sin/cos LUT). Exchange-local, not UTC, so DST does not shift the session.
  2. `day-of-week`: sin, cos of ET weekday (Mon=0).
  3. `session-gap`: flag (delta to previous valid bar > 5 min), `ln(delta_min/5)` clipped at 8.0 (~10 days). First bar of a ticker → (0,0).
  4. `volume`: log-volume innovation, validity (unchanged semantics).
  5. `market`: equal-weighted mean 5-min log close return over all retained corpus tickers at that timestamp, validity. Only pairs of valid bars exactly 5 min apart contribute (returns across gaps excluded so a slot never mixes overnight jumps with intraday moves). Precomputed once at `Corpus::load` on the shared UTC grid (dense `Vec<f32>`, NaN = missing, ~4 B/slot; 16 rayon chunks). Contemporaneous history: slot t uses only closes at t and t−5min.
  6. `spy`: **SPY exists** (`long_data/bars/SPY.300.bars`, 468k bars 2016-08-22…2026-08-19; QQQ exists too but only SPY is wired). Same 5-min log return of SPY, validity. `CorpusContract.spy_fingerprint` = SHA-256 of the SPY file. Missing file with `spy` enabled → load error.
- `auxiliary: Option<Tensor>` is `[batch, seq_len, n_aux]`, `n_aux = features.channels()`; aux row segment sits between targets and mask exactly as before (`aux_len = context * n_aux`). `Batch::from_packed` takes `aux_channels`. Per-bar values are written by `features::AuxiliaryCursor` inside `fill_row`'s existing loop; no per-row allocation.
- `normalize_auxiliary(aux, &FeatureSet)`: innovation groups (volume/market/spy) get the old window valid-only z-score (`normalize_innovation`), calendar/gap groups pass through; concatenated in order and fed as extra variate tokens through the existing `variate` projection.
- `CorpusContract`: `features`, `auxiliary_schema` (descriptive string from `FeatureSet::schema()`), `spy_fingerprint`; SCHEMA suffixed `;exogenous-variates-on-shared-utc-grid`. `DataContract` lost `volume_features`/`auxiliary_schema`. `Manifest::read` checks `model.features == data.features`.
- Deleted the dead volume path in `Dataset` (`auxiliary` Vec, `auxiliary_windows`, `auxiliary_batch`, `set_volume_features`, `auxiliary_schema()`, its test). `benchmarks/timexer_universe_campaign.py` now passes `--features all|none` and compares the `features` dict.
- Docs: new "Exogenous variates" section in docs/timexer_segment.md; campaign paragraph updated.

## Verification
- `./torch-env.sh cargo check -p trading_bot_0 --tests` in the worktree: 0 `^error` lines (the LIBTORCH/BYPASS env is rejected by the vendored torch-sys build script; the repo wrapper with `FA4_VENV=<main>/.venv-fa4` is the supported route).
- Scoped tests pass: features (2), corpus (5 incl. new market/gap/volume alignment assertions), data (11), model aux normalization, runner manifest.
- Throwaway real-data smoke (deleted): corpus of AAPL/MSFT/SPY, MSFT validation row origin 326515. First bar after a weekend gap (pos 206, 2025-01-06 04:00 ET, Monday): tod=(0.866025,0.500000) dow=(0.000000,1.000000) gap=(1, 6.511745 = ln(3365/5)) vol=(0.540656,1) market=(0,0) spy=(0,0). Last 3 context bars (2025-02-25 06:10/06:15/06:20 ET, Tuesday): tod=(0.999048,-0.043619)/(0.997859,-0.065403)/(0.996195,-0.087156), dow=(0.781832,0.623490), gap=(1,1.098612=ln 3 → 15-min pre-market gap)/(0,0)/(0,0), market=0.00035395/0.00028240/-0.00053999, spy=0.00030227/0.00006712/-0.00013425. Independent Python cross-section over the raw bar files (same strict 5-min rule) gives means 0.00035395 / 0.00028240 / -0.00053999 and SPY 0.000302270 / 0.0000671245 / -0.000134254 — exact match; post-weekend slot has no qualifying ticker → NA ⇒ validity 0 as designed.
- Throwaway clap test (deleted): `--features volume,market` parses, default renders `time-of-day,day-of-week,session-gap,volume,market,spy`, `--features bogus` and `--volume-features` are rejected.

## Design notes
- Market return uses strict 5-min pairs rather than "previous valid bar": mixing overnight (~1%) and intraday (~0.1%) returns in one cross-sectional mean would make the level depend on which tickers have pre-market bars. Gap structure is carried by `session-gap` and the ticker's own closes.
- The first RTH bar after a gap therefore has market validity 0 when no ticker has a bar 5 min earlier; liquid names with extended-hours bars make most slots valid.
- Gap log is left in natural units (0…8) rather than rescaled; it is a linear-projection input.

## Conflict-prone hunks outside features.rs (for merging onto main)
- corpus.rs: `use` block (imports `features::*`, `file_sha256`, `bar_file_path`); `RESOLUTION_MS` now `pub(super)`; `CorpusContract` fields; `Corpus` struct (`exogenous`); `Batch` (`aux_channels` replaces `volume_features`, `from_packed` signature); `Corpus::load` (signature `features: &FeatureSet`, `contract_with_bounds` call without bool, `exogenous_series` call, contract literal); `host_batch` (aux_len, `fill_row` call); new `exogenous_series` fn; `fill_row` signature + aux write inside the `position < context` block (removed `previous_volume` and the volume block); tests `pooled_rows_preserve_ticker_values_and_mask_partial_targets` and `pooled_epoch_keeps_delisted_tickers_...` (Corpus::load call + new aux assertions).
- data.rs: `DataContract` (removed 2 fields + defaults); `Dataset` struct/`PreparedHistory`/`prepare`/`from_bars_with_materialization`/`auxiliary_batch`/`set_volume_features` deletions (ReturnSpace deletes the whole `impl Dataset`; take deletion); `contract_with_bounds`/`load_with_bounds`/`filtered_contract` signatures (drop bool); `auxiliary_schema()` deleted; tests updated.
- model.rs: `use super::features::FeatureSet`; `ModelConfig.features` (+Default); `normalize_innovation`/`normalize_auxiliary`; `forward_with_aux` assert and `variate_history` block; deleted `historical_model_configuration_retains_...` test; replaced `volume_normalization_...` test.
- runner.rs: `Manifest::read` feature check; two `Corpus::load` call sites; manifest test contract literals + `FeatureSet` import.
- benchmark.rs: synthetic aux tensor (`aux_channels`).
- torch/dataset.rs: `et_offset_secs` → `pub(crate)`.
