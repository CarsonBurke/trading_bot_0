//! Planner policy view over the shared bar corpus.
//!
//! [`crate::torch::dataset::BarCorpus`] owns the loader, the mmap and the one global calendar
//! train/val/test split shared by every symbol. This file adds only the policy-side question:
//! which bars an episode may start at, and the DOF context an episode sees.

use std::path::Path;

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use rand::{rngs::StdRng, Rng};
use ring::digest::{Context as DigestContext, SHA256};
use tch::{Device, Tensor};

use crate::torch::dataset::{BarBatch, BarCorpus, Split};

pub use crate::torch::dataset::BarEndpoint as PlannerEndpoint;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum PlannerDataSplit {
    Train,
    Validation,
    Test,
}

impl PlannerDataSplit {
    pub fn split(self) -> Split {
        match self {
            PlannerDataSplit::Train => Split::Train,
            PlannerDataSplit::Validation => Split::Val,
            PlannerDataSplit::Test => Split::Test,
        }
    }

    fn tag(self) -> u8 {
        match self {
            PlannerDataSplit::Train => 0,
            PlannerDataSplit::Validation => 1,
            PlannerDataSplit::Test => 2,
        }
    }
}

/// Episode start points, addressed absolutely in the corpus.
#[derive(Clone, Debug)]
pub struct PlannerCorpus {
    corpus: BarCorpus,
    /// Series indices the planner may draw from, ascending. `None` is every series.
    allowed: Option<Vec<usize>>,
}

fn no_endpoints_err(
    split: PlannerDataSplit,
    context_bars: usize,
    actual_future_bars: usize,
) -> anyhow::Error {
    anyhow::anyhow!(
        "no {split:?} planner endpoints have context={context_bars} and future={actual_future_bars}"
    )
}

/// `index`-th bar of the concatenated eligible ranges. Endpoints are addressed positionally so
/// the selectors never materialize the (corpus-scale) set of eligible bars.
fn nth_eligible(eligible: &[(usize, usize, usize)], index: usize) -> PlannerEndpoint {
    let mut remaining = index;
    for &(series, start, end) in eligible {
        let width = end - start;
        if remaining < width {
            return PlannerEndpoint {
                series,
                bar: start + remaining,
            };
        }
        remaining -= width;
    }
    unreachable!("eligible index {index} is past the concatenated eligible width");
}

impl PlannerCorpus {
    pub fn load(dir: &Path, res_secs: u32, min_bars: usize) -> Result<Self> {
        Self::load_filtered(dir, res_secs, min_bars, None)
    }

    /// `load`, restricted to `tickers`. `None` keeps the whole corpus.
    pub fn load_filtered(
        dir: &Path,
        res_secs: u32,
        min_bars: usize,
        tickers: Option<&[String]>,
    ) -> Result<Self> {
        Self::restrict(BarCorpus::load(dir, res_secs, min_bars)?, tickers)
    }

    /// Resolve a ticker allow-list against `corpus`, matching case-insensitively. A ticker the
    /// corpus does not hold is fatal: silently planning over an empty or truncated universe
    /// wastes a whole training run before anyone notices.
    pub fn restrict(corpus: BarCorpus, tickers: Option<&[String]>) -> Result<Self> {
        let Some(tickers) = tickers else {
            return Ok(Self::from_corpus(corpus));
        };
        if tickers.is_empty() {
            bail!("planner ticker filter is empty; omit it to plan over the whole corpus");
        }
        let mut allowed = Vec::with_capacity(tickers.len());
        let mut missing = Vec::new();
        for ticker in tickers {
            match (0..corpus.series_count())
                .find(|&series| corpus.symbol(series).eq_ignore_ascii_case(ticker))
            {
                Some(series) => allowed.push(series),
                None => missing.push(ticker.as_str()),
            }
        }
        if !missing.is_empty() {
            bail!(
                "planner tickers absent from the {}-symbol corpus at {}s: {}",
                corpus.series_count(),
                corpus.res_secs(),
                missing.join(", ")
            );
        }
        allowed.sort_unstable();
        allowed.dedup();
        Ok(Self {
            corpus,
            allowed: Some(allowed),
        })
    }

    pub fn from_corpus(corpus: BarCorpus) -> Self {
        Self {
            corpus,
            allowed: None,
        }
    }

    pub fn corpus(&self) -> &BarCorpus {
        &self.corpus
    }

    pub fn series_count(&self) -> usize {
        self.corpus.series_count()
    }

    pub fn res_secs(&self) -> u32 {
        self.corpus.res_secs()
    }

    /// Whether the allow-list admits `series`.
    fn is_allowed(&self, series: usize) -> bool {
        self.allowed
            .as_deref()
            .is_none_or(|allowed| allowed.binary_search(&series).is_ok())
    }

    pub fn symbol(&self, series: usize) -> &str {
        self.corpus.symbol(series)
    }

    /// Mark-to-market price, widened from the corpus `f32`.
    pub fn close(&self, series: usize, bar: usize) -> f64 {
        self.corpus.close(series, bar) as f64
    }

    pub fn sample_endpoints(
        &self,
        split: PlannerDataSplit,
        count: usize,
        context_bars: usize,
        actual_future_bars: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<PlannerEndpoint>> {
        let eligible = self.eligible_ranges(split, context_bars, actual_future_bars);
        if eligible.is_empty() {
            return Err(no_endpoints_err(split, context_bars, actual_future_bars));
        }
        let total: usize = eligible.iter().map(|(_, start, end)| end - start).sum();
        Ok((0..count)
            .map(|_| nth_eligible(&eligible, rng.random_range(0..total)))
            .collect())
    }

    pub fn deterministic_endpoints(
        &self,
        split: PlannerDataSplit,
        count: usize,
        context_bars: usize,
        actual_future_bars: usize,
    ) -> Result<Vec<PlannerEndpoint>> {
        let eligible = self.eligible_ranges(split, context_bars, actual_future_bars);
        if eligible.is_empty() {
            return Err(no_endpoints_err(split, context_bars, actual_future_bars));
        }
        let total: usize = eligible.iter().map(|(_, start, end)| end - start).sum();
        let count = count.min(total);
        if count == 0 {
            bail!("planner deterministic endpoint count must be positive");
        }
        Ok((0..count)
            .map(|index| nth_eligible(&eligible, index * total / count))
            .collect())
    }

    pub fn deterministic_ticker_stratified_endpoints(
        &self,
        split: PlannerDataSplit,
        count: usize,
        context_bars: usize,
        actual_future_bars: usize,
    ) -> Result<Vec<PlannerEndpoint>> {
        if count == 0 {
            bail!("planner stratified endpoint count must be positive");
        }
        let eligible = self.eligible_ranges(split, context_bars, actual_future_bars);
        if eligible.is_empty() {
            return Err(no_endpoints_err(split, context_bars, actual_future_bars));
        }
        let count = count.min(eligible.len());
        Ok((0..count)
            .map(|index| {
                let (series, start, end) = eligible[index * eligible.len() / count];
                PlannerEndpoint {
                    series,
                    bar: start + (end - start) / 2,
                }
            })
            .collect())
    }

    pub fn deterministic_ticker_time_stratified_endpoints(
        &self,
        split: PlannerDataSplit,
        count: usize,
        context_bars: usize,
        actual_future_bars: usize,
    ) -> Result<Vec<PlannerEndpoint>> {
        if count == 0 {
            bail!("planner ticker-time stratified endpoint count must be positive");
        }
        let eligible = self.eligible_ranges(split, context_bars, actual_future_bars);
        if eligible.is_empty() {
            return Err(no_endpoints_err(split, context_bars, actual_future_bars));
        }

        const MIN_TIME_STRATA: usize = 4;
        let initial_series = eligible.len().min((count / MIN_TIME_STRATA).max(1));
        let mut selected_indices = (0..initial_series)
            .map(|index| index * eligible.len() / initial_series)
            .collect::<Vec<_>>();
        let mut selected = vec![false; eligible.len()];
        for &index in &selected_indices {
            selected[index] = true;
        }
        let mut capacity = selected_indices
            .iter()
            .map(|&index| eligible[index].2 - eligible[index].1)
            .sum::<usize>();
        if capacity < count {
            for (index, &(_, start, end)) in eligible.iter().enumerate() {
                if !selected[index] {
                    selected_indices.push(index);
                    capacity += end - start;
                    if capacity >= count {
                        break;
                    }
                }
            }
        }
        if capacity < count {
            bail!(
                "planner ticker-time stratification requires {count} distinct endpoints, but only {capacity} fit inside the requested split"
            );
        }

        let capacities = selected_indices
            .iter()
            .map(|&index| eligible[index].2 - eligible[index].1)
            .collect::<Vec<_>>();
        let mut allocations = vec![0usize; selected_indices.len()];
        let mut remaining = count;
        while remaining > 0 {
            for (allocation, &series_capacity) in allocations.iter_mut().zip(&capacities) {
                if *allocation < series_capacity && remaining > 0 {
                    *allocation += 1;
                    remaining -= 1;
                }
            }
        }
        let mut endpoints = Vec::with_capacity(count);
        for (&eligible_index, &slots) in selected_indices.iter().zip(&allocations) {
            let (series, start, end) = eligible[eligible_index];
            for slot in 0..slots {
                let offset = (2 * slot + 1) * (end - start) / (2 * slots);
                endpoints.push(PlannerEndpoint {
                    series,
                    bar: start + offset,
                });
            }
        }
        debug_assert_eq!(endpoints.len(), count);
        Ok(endpoints)
    }

    /// `[endpoints.len(), context_bars, BAR_DOF]` DOF and the matching
    /// `[endpoints.len(), context_bars, BAR_TIME_FEATURES]` calendar ids on `device`; row `i`
    /// ends at bar `endpoints[i].bar + advances[i]`.
    ///
    /// Every DOF carries a 256-bar causal volume-EMA warm-up, so one long window is far cheaper
    /// than many short ones: step a rollout by requesting the whole contiguous run at once.
    pub fn context_batch(
        &self,
        endpoints: &[PlannerEndpoint],
        advances: &[usize],
        context_bars: usize,
        device: Device,
    ) -> Result<BarBatch> {
        if endpoints.len() != advances.len() || endpoints.is_empty() {
            bail!("planner endpoints/advances must be non-empty and equally sized");
        }
        self.corpus
            .dof_window(endpoints, advances, context_bars as i64, device)
    }

    /// DOF alone, for consumers that already hold the matching calendar ids.
    pub fn dof_context(
        &self,
        endpoints: &[PlannerEndpoint],
        advances: &[usize],
        context_bars: usize,
        device: Device,
    ) -> Result<Tensor> {
        Ok(self
            .context_batch(endpoints, advances, context_bars, device)?
            .dof)
    }

    /// SHA-256 over the evaluation contract: split, lengths, corpus identity, and the exact
    /// endpoint list. Bar contents are deliberately absent — `identity_fingerprint` already
    /// pins every symbol's name, length and timestamp span, and the corpus is 24 GB and still
    /// growing, so re-hashing its bars would cost a full scan to learn nothing new.
    pub fn evaluation_fingerprint(
        &self,
        split: PlannerDataSplit,
        endpoints: &[PlannerEndpoint],
        horizon: usize,
        context_bars: usize,
        rollout_length: usize,
    ) -> Result<String> {
        if endpoints.is_empty() || context_bars == 0 || rollout_length == 0 {
            bail!("planner evaluation fingerprint requires non-empty endpoints and lengths");
        }
        let mut digest = DigestContext::new(&SHA256);
        digest.update(b"planner-evaluation-v2");
        digest.update(&[split.tag()]);
        for value in [horizon, context_bars, rollout_length, endpoints.len()] {
            digest.update(&(value as u64).to_le_bytes());
        }
        let identity = self.corpus.identity_fingerprint();
        digest.update(&(identity.len() as u64).to_le_bytes());
        digest.update(identity.as_bytes());
        // Two eval runs over different ticker subsets of one corpus are different evaluations.
        let allowed = (0..self.corpus.series_count()).filter(|&series| self.is_allowed(series));
        digest.update(&(allowed.clone().count() as u64).to_le_bytes());
        for series in allowed {
            let symbol = self.corpus.symbol(series);
            digest.update(&(symbol.len() as u64).to_le_bytes());
            digest.update(symbol.as_bytes());
        }

        for endpoint in endpoints {
            if endpoint.series >= self.corpus.series_count() {
                bail!(
                    "planner evaluation endpoint series {} is outside the {}-series corpus",
                    endpoint.series,
                    self.corpus.series_count()
                );
            }
            let series_len = self.corpus.series_len(endpoint.series);
            (endpoint.bar + 1)
                .checked_sub(context_bars)
                .filter(|&start| start >= 1)
                .context("planner evaluation context precedes available history")?;
            (endpoint.bar + rollout_length < series_len)
                .then_some(())
                .context("planner evaluation features exceed available history")?;
            let symbol = self.corpus.symbol(endpoint.series);
            digest.update(&(symbol.len() as u64).to_le_bytes());
            digest.update(symbol.as_bytes());
            digest.update(&(endpoint.bar as u64).to_le_bytes());
            digest.update(&self.corpus.ts_ms(endpoint.series, endpoint.bar).to_le_bytes());
        }
        Ok(digest
            .finish()
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect())
    }

    /// `(series, first, last_exclusive)` bar indices whose whole episode — context and future —
    /// stays inside `split`, for every series the ticker allow-list admits.
    fn eligible_ranges(
        &self,
        split: PlannerDataSplit,
        context_bars: usize,
        actual_future_bars: usize,
    ) -> Vec<(usize, usize, usize)> {
        (0..self.corpus.series_count())
            .filter(|&series| self.is_allowed(series))
            .filter_map(|series| {
                let (lo, hi) = self.corpus.split_range(series, split.split());
                // A DOF needs a predecessor close, so the first DOF bar index must be >= 1 and
                // the last must be < series_len, or `dof_window` errors.
                let start = lo.max(context_bars);
                let end = hi.min(
                    self.corpus
                        .series_len(series)
                        .saturating_sub(actual_future_bars),
                );
                (start < end).then_some((series, start, end))
            })
            .collect()
    }
}

pub fn planner_context_bars(
    metadata: &crate::torch::world_model::BarWorldModelMetadata,
    requested: Option<usize>,
) -> Result<usize> {
    let maximum = metadata.max_context_bars as usize;
    let context = requested.unwrap_or(maximum);
    if context == 0 || context > maximum {
        bail!("planner context bars must be in 1..={maximum}, got {context}");
    }
    Ok(context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::bar_dist::BAR_DOF;
    use crate::torch::dataset::{BAR_TIME_CARDINALITY, BAR_TIME_FEATURES};
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;
    use shared::bars::{write_bar_file, PackedBar, FILE_EXTENSION};
    use std::collections::HashSet;
    use std::path::PathBuf;

    const RES: u32 = 300;
    const RES_MS: i64 = RES as i64 * 1000;
    const MIN_BARS: usize = 100;
    const CONTEXT: usize = 64;
    const FUTURE: usize = 16;
    const SPLITS: [PlannerDataSplit; 3] = [
        PlannerDataSplit::Train,
        PlannerDataSplit::Validation,
        PlannerDataSplit::Test,
    ];

    /// `(symbol, seed, first slot, last slot)` on a shared 5-minute grid. Listings are
    /// staggered and `DDD` also stops early, so the series disagree on both their train width
    /// and their test width — a per-symbol index split would put them at different instants.
    const SERIES: [(&str, u64, i64, i64); 4] = [
        ("AAA", 1, 0, 2_999),
        ("BBB", 2, 300, 2_999),
        ("CCC", 3, 700, 2_999),
        ("DDD", 4, 1_200, 2_997),
    ];

    struct Fixture {
        dir: PathBuf,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn synth_bars(seed: u64, first_slot: i64, last_slot: i64) -> Vec<PackedBar> {
        let base = 1_600_000_000_000i64 / RES_MS * RES_MS;
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        let mut close = 100.0f32;
        (first_slot..=last_slot)
            .map(|slot| {
                let open = close;
                close = (close * (1.0 + rng.random_range(-0.01f32..0.01f32))).max(1.0);
                let spread = rng.random_range(0.0f32..0.02f32) * open;
                PackedBar {
                    ts_ms: base + slot * RES_MS,
                    open,
                    high: open.max(close) + spread,
                    low: (open.min(close) - spread).max(0.5),
                    close,
                    volume: rng.random_range(1_000.0f32..50_000.0f32),
                    vwap: 0.5 * (open + close),
                    trades: rng.random_range(1u32..500),
                }
            })
            .collect()
    }

    fn fixture(label: &str, extra: &[(&str, u64, i64, i64)]) -> (Fixture, PlannerCorpus) {
        let dir = std::env::temp_dir().join(format!(
            "trading_bot_0_planner_data_{label}_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for &(symbol, seed, first_slot, last_slot) in SERIES.iter().chain(extra) {
            let bars = synth_bars(seed, first_slot, last_slot);
            let path = dir.join(format!("{symbol}.{RES}.{FILE_EXTENSION}"));
            write_bar_file(&path, symbol, RES, &bars).unwrap();
        }
        let corpus = PlannerCorpus::load(&dir, RES, MIN_BARS).unwrap();
        (Fixture { dir }, corpus)
    }

    fn capacities(eligible: &[(usize, usize, usize)]) -> Vec<usize> {
        eligible.iter().map(|&(_, s, e)| e - s).collect()
    }

    /// A future length that leaves each series only a couple of eligible test endpoints, so the
    /// stratifier must redistribute capacity and can be pushed just past it.
    fn tight_test_future(corpus: &PlannerCorpus) -> usize {
        let (lo, _) = corpus.corpus().split_range(0, Split::Test);
        let width = corpus.corpus().series_len(0) - lo;
        assert!(lo >= CONTEXT, "test split must start past the context floor");
        assert!(width > 3, "test split of {width} bars is too narrow to trim");
        width - 3
    }

    #[test]
    fn ticker_time_stratification_covers_every_series_and_multiple_strata() {
        let (_fx, corpus) = fixture("strata", &[]);
        let eligible = corpus.eligible_ranges(PlannerDataSplit::Train, CONTEXT, FUTURE);
        assert_eq!(
            eligible.len(),
            corpus.series_count(),
            "every series must reach the train split"
        );

        let count = 4 * eligible.len();
        let endpoints = corpus
            .deterministic_ticker_time_stratified_endpoints(
                PlannerDataSplit::Train,
                count,
                CONTEXT,
                FUTURE,
            )
            .unwrap();
        assert_eq!(endpoints.len(), count);

        for &(series, start, end) in &eligible {
            let bars = endpoints
                .iter()
                .filter(|endpoint| endpoint.series == series)
                .map(|endpoint| endpoint.bar)
                .collect::<Vec<_>>();
            assert!(
                bars.len() >= 2,
                "series {series} must contribute multiple time strata, got {bars:?}"
            );
            assert!(
                bars.windows(2).all(|pair| pair[0] < pair[1]),
                "strata must be distinct and time ordered, got {bars:?}"
            );
            assert!(bars.iter().all(|bar| (start..end).contains(bar)));
        }
    }

    #[test]
    fn ticker_time_stratification_redistributes_capacity_then_fails_loudly() {
        let (_fx, corpus) = fixture("capacity", &[]);
        let future = tight_test_future(&corpus);
        let eligible = corpus.eligible_ranges(PlannerDataSplit::Test, CONTEXT, future);
        let caps = capacities(&eligible);
        let capacity: usize = caps.iter().sum();
        assert!(
            caps.len() >= 2 && caps.iter().any(|&c| c != caps[0]),
            "fixture must offer unequal per-series capacity, got {caps:?}"
        );
        assert!(
            capacity <= 4 * caps.len(),
            "capacity {capacity} must be tight enough to force series redistribution"
        );

        let endpoints = corpus
            .deterministic_ticker_time_stratified_endpoints(
                PlannerDataSplit::Test,
                capacity,
                CONTEXT,
                future,
            )
            .unwrap();
        assert_eq!(endpoints.len(), capacity);
        for (&(series, ..), &cap) in eligible.iter().zip(&caps) {
            assert_eq!(
                endpoints
                    .iter()
                    .filter(|endpoint| endpoint.series == series)
                    .count(),
                cap,
                "series {series} must receive exactly its capacity"
            );
        }
        assert_eq!(
            endpoints.iter().copied().collect::<HashSet<_>>().len(),
            capacity,
            "a saturated split must use each eligible bar once"
        );

        let error = corpus
            .deterministic_ticker_time_stratified_endpoints(
                PlannerDataSplit::Test,
                capacity + 1,
                CONTEXT,
                future,
            )
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("requires") && error.contains(&(capacity + 1).to_string()),
            "unexpected capacity error: {error}"
        );
    }

    #[test]
    fn every_selector_keeps_context_and_future_inside_its_split() {
        let (_fx, corpus) = fixture("bounds", &[]);
        let mut rng = StdRng::seed_from_u64(0xC0FF_EE00);
        for split in SPLITS {
            let selections = [
                corpus
                    .sample_endpoints(split, 64, CONTEXT, FUTURE, &mut rng)
                    .unwrap(),
                corpus
                    .deterministic_endpoints(split, 32, CONTEXT, FUTURE)
                    .unwrap(),
                corpus
                    .deterministic_ticker_stratified_endpoints(split, 32, CONTEXT, FUTURE)
                    .unwrap(),
                corpus
                    .deterministic_ticker_time_stratified_endpoints(split, 32, CONTEXT, FUTURE)
                    .unwrap(),
            ];
            for endpoints in selections {
                assert!(!endpoints.is_empty(), "{split:?} produced no endpoints");
                for endpoint in endpoints {
                    let series_len = corpus.corpus().series_len(endpoint.series);
                    let (lo, hi) = corpus.corpus().split_range(endpoint.series, split.split());
                    assert!(
                        endpoint.bar >= CONTEXT,
                        "{split:?} {endpoint:?} has no room for {CONTEXT} context bars"
                    );
                    assert!(
                        endpoint.bar + FUTURE < series_len,
                        "{split:?} {endpoint:?} runs past {series_len} bars"
                    );
                    assert!(
                        (lo..hi).contains(&endpoint.bar),
                        "{split:?} {endpoint:?} escapes its split range {lo}..{hi}"
                    );
                }
            }
        }
    }

    #[test]
    fn evaluation_fingerprint_pins_contract_endpoints_and_corpus_identity() {
        let (_fx, corpus) = fixture("fingerprint", &[]);
        let endpoints = corpus
            .deterministic_endpoints(PlannerDataSplit::Test, 4, CONTEXT, FUTURE)
            .unwrap();
        let fingerprint = |endpoints: &[PlannerEndpoint], horizon, rollout| {
            corpus
                .evaluation_fingerprint(PlannerDataSplit::Test, endpoints, horizon, CONTEXT, rollout)
                .unwrap()
        };

        let base = fingerprint(&endpoints, 4, 8);
        assert_eq!(base, fingerprint(&endpoints, 4, 8));
        assert_ne!(base, fingerprint(&endpoints, 5, 8));
        assert_ne!(base, fingerprint(&endpoints, 4, 9));
        assert_ne!(base, fingerprint(&endpoints[..3], 4, 8));
        assert_ne!(
            base,
            corpus
                .evaluation_fingerprint(PlannerDataSplit::Validation, &endpoints, 4, CONTEXT, 8)
                .unwrap()
        );

        let (_grown_fx, grown) = fixture("fingerprint_grown", &[("ZZZ", 9, 0, 2_999)]);
        assert!(grown.series_count() > corpus.series_count());
        assert_eq!(
            grown.symbol(endpoints[0].series),
            corpus.symbol(endpoints[0].series),
            "the extra symbol must sort last so the endpoint still names the same series"
        );
        assert_ne!(
            base,
            grown
                .evaluation_fingerprint(PlannerDataSplit::Test, &endpoints, 4, CONTEXT, 8)
                .unwrap(),
            "a grown corpus must not fingerprint as the same evaluation set"
        );
    }

    #[test]
    fn context_batch_is_shaped_by_endpoints_and_indexes_the_clock_tables() {
        let (_fx, corpus) = fixture("dof", &[]);
        let endpoints = corpus
            .deterministic_endpoints(PlannerDataSplit::Validation, 3, CONTEXT, FUTURE)
            .unwrap();
        let advances = vec![0usize; endpoints.len()];
        let batch = corpus
            .context_batch(&endpoints, &advances, CONTEXT, Device::Cpu)
            .unwrap();
        let rows = endpoints.len() as i64;
        assert_eq!(batch.dof.size(), vec![rows, CONTEXT as i64, BAR_DOF as i64]);
        assert_eq!(
            batch.time_ids.size(),
            vec![rows, CONTEXT as i64, BAR_TIME_FEATURES as i64]
        );
        assert!(bool::try_from(batch.dof.isfinite().all()).expect("finite check"));
        // Out-of-range ids surface as a bare CUDA assert inside the trunk's embedding lookup.
        for (channel, &cardinality) in BAR_TIME_CARDINALITY.iter().enumerate() {
            let ids = batch.time_ids.select(2, channel as i64);
            assert!(ids.min().int64_value(&[]) >= 0);
            assert!(
                ids.max().int64_value(&[]) < cardinality,
                "calendar channel {channel} must index a {cardinality}-row table"
            );
        }

        let dof = corpus
            .dof_context(&endpoints, &advances, CONTEXT, Device::Cpu)
            .unwrap();
        assert_eq!(dof.size(), batch.dof.size());
        let advanced = corpus
            .dof_context(
                &endpoints,
                &vec![FUTURE; endpoints.len()],
                CONTEXT,
                Device::Cpu,
            )
            .unwrap();
        assert!(
            f64::try_from((&advanced - &dof).abs().max()).expect("max") > 0.0,
            "advancing must slide the window forward"
        );

        assert!(corpus
            .dof_context(&endpoints, &advances[..2], CONTEXT, Device::Cpu)
            .is_err());
        assert!(corpus
            .context_batch(&[], &[], CONTEXT, Device::Cpu)
            .is_err());
    }

    #[test]
    fn ticker_filter_restricts_series_and_rejects_unknown_symbols() {
        let (fx, all) = fixture("filter", &[]);
        let series_count = all.series_count();
        assert!(series_count >= 3, "fixture must hold several symbols");
        let picked = [series_count - 3, series_count - 1];
        let wanted = picked
            .iter()
            .map(|&series| all.symbol(series).to_lowercase())
            .collect::<Vec<_>>();

        let filtered = PlannerCorpus::load_filtered(&fx.dir, RES, MIN_BARS, Some(&wanted)).unwrap();
        assert_eq!(filtered.res_secs(), RES);
        assert_eq!(
            filtered.series_count(),
            series_count,
            "a filter must not renumber series"
        );
        assert_eq!(
            filtered
                .eligible_ranges(PlannerDataSplit::Train, CONTEXT, FUTURE)
                .iter()
                .map(|&(series, ..)| series)
                .collect::<HashSet<_>>(),
            picked.iter().copied().collect::<HashSet<_>>()
        );
        let endpoints = filtered
            .deterministic_endpoints(PlannerDataSplit::Test, 8, CONTEXT, FUTURE)
            .unwrap();
        assert!(endpoints
            .iter()
            .all(|endpoint| picked.contains(&endpoint.series)));

        // The same endpoints over the same corpus but a different universe is a different
        // evaluation set.
        assert_ne!(
            filtered
                .evaluation_fingerprint(PlannerDataSplit::Test, &endpoints, 4, CONTEXT, 8)
                .unwrap(),
            all.evaluation_fingerprint(PlannerDataSplit::Test, &endpoints, 4, CONTEXT, 8)
                .unwrap()
        );

        let unknown = vec![all.symbol(0).to_owned(), "NoSuchTicker".to_owned()];
        let error = PlannerCorpus::load_filtered(&fx.dir, RES, MIN_BARS, Some(&unknown))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("NoSuchTicker") && !error.contains(all.symbol(0)),
            "the filter error must name only the missing tickers: {error}"
        );
        assert!(PlannerCorpus::load_filtered(&fx.dir, RES, MIN_BARS, Some(&[])).is_err());
    }
}
