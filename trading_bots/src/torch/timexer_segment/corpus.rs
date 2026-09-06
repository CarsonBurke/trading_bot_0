use std::{collections::BTreeSet, fs, path::Path};

use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use shared::{
    bars::{bar_file_path, parse_bar_file_name, BarFile, PackedBar},
    report::CandleBar,
};
use tch::{Device, Kind, Tensor};

use super::{
    data::{filtered_contract, retained_partition_end, valid_ohlc, DataContract},
    features::{
        market_steps, single_series, AuxiliaryCursor, Exogenous, FeatureSet, Grid, MarketSummary,
        SPY,
    },
};
use crate::torch::hashing::file_sha256;

pub(super) const RESOLUTION_MS: i64 = 300_000;
const SCHEMA: &str = "timexer-pooled-mmap-v5;independent-ticker-rows;all-valid-source-unique-utc-grid-quantiles70:10:10:10;purge-only-at-observed-partition-boundaries;centered-log-prices-with-bar-validity;covariates-over-context-and-horizon;next-valid-observed-bars;invalid-ohlc-rows-quarantined-without-repair;disjoint-target-epoch-with-masked-remainder;validation-disjoint-complete-targets;terminal-test-locked;exogenous-variates-on-shared-utc-grid;market-cumulative-log-return-over-steps-defined-by-min-cross-section-slots-spanning-sparse-slots-centered-at-origin";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowRef {
    pub ticker: usize,
    pub origin: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExcludedTicker {
    pub ticker: String,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CorpusContract {
    pub schema: String,
    pub tickers: Vec<DataContract>,
    pub boundary_timestamps: [i64; 3],
    pub context: usize,
    pub pred_len: usize,
    pub common_context: usize,
    pub purge: usize,
    pub features: FeatureSet,
    pub auxiliary_schema: String,
    pub spy_fingerprint: Option<String>,
    /// SHA-256 over the ordered universe fingerprints, partition boundaries, and the market step
    /// construction (threshold included) that define the cumulative market path every row's
    /// targets are demeaned by.
    pub market_fingerprint: String,
    /// Sources that must hold a valid bar at a grid slot for it to define a market step.
    pub market_min_cross_section: usize,
    pub minimum_source_bars: usize,
    pub minimum_training_bars: usize,
    pub train_target_bars: usize,
    pub validation_target_bars: usize,
    pub validation_remainder_bars: usize,
    pub excluded_tickers: Vec<ExcludedTicker>,
}

pub struct CorpusTicker {
    pub contract: DataContract,
    file: BarFile,
}

impl CorpusTicker {
    pub fn timestamp(&self, origin: usize) -> i64 {
        self.bar(origin).ts()
    }

    pub fn candle_window(
        &self,
        origin: usize,
        history: usize,
        future: usize,
    ) -> Result<Vec<CandleBar>> {
        let start = origin
            .checked_sub(history)
            .context("candle history precedes corpus")?;
        let end = origin
            .checked_add(future)
            .and_then(|i| i.checked_add(1))
            .context("candle window overflow")?;
        ensure!(
            end <= self.contract.valid_bars,
            "candle window exceeds valid observed corpus"
        );
        Ok(ValidBars::new(
            self.file.bars(),
            &self.contract.invalid_ohlc_indices,
            start,
            end - start,
        )
        .map(|bar| CandleBar {
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
        })
        .collect())
    }

    pub fn bar(&self, logical_index: usize) -> &PackedBar {
        assert!(
            logical_index < self.contract.valid_bars,
            "valid observed bar exceeds corpus"
        );
        &self.file.bars()[raw_index(logical_index, &self.contract.invalid_ohlc_indices)]
    }

    fn target_count(&self, origin: usize) -> Option<usize> {
        let c = &self.contract;
        if origin < c.common_context.max(c.context) - 1 {
            return None;
        }
        if origin < c.train_end - 1 {
            return Some(c.pred_len.min(c.train_end - origin - 1));
        }
        let start = c.boundaries[1].saturating_sub(1);
        let end = retained_partition_end(c.boundaries[2], c.valid_bars, c.purge)
            .checked_sub(c.pred_len)?;
        (origin >= start && origin < end).then_some(c.pred_len)
    }
}

pub struct Corpus {
    pub contract: CorpusContract,
    pub train_refs: Vec<WindowRef>,
    pub validation_refs: Vec<WindowRef>,
    pub excluded_tickers: Vec<ExcludedTicker>,
    pub market: MarketSummary,
    tickers: Vec<CorpusTicker>,
    exogenous: Exogenous,
    gather_pool: rayon::ThreadPool,
    device: Device,
}

/// One row is `seq_len + pred_len` bars: `log_prices[b, t, c] = ln(price) - ln(anchor)` where
/// `anchor` is the row's last context close, `valid[b, t]` marks observed bars (context bars are
/// always valid; horizon bars are valid up to the ticker's owned targets), `aux[b, t, :]`
/// carries the covariates for every bar with history-only channels zeroed beyond the context,
/// and `market_cum[b, t]` is the cumulative market log return at the bar's timestamp minus its
/// value at the row's last context bar.
pub struct Batch {
    pub log_prices: Tensor,
    pub valid: Tensor,
    pub aux: Tensor,
    pub market_cum: Tensor,
    pub anchor: Tensor,
    pub valid_target_bars: usize,
    packed: Tensor,
    context: usize,
    pred_len: usize,
    aux_channels: usize,
}

impl Batch {
    pub(super) fn row_width(context: usize, pred_len: usize, aux_channels: usize) -> usize {
        (context + pred_len) * (6 + aux_channels) + 1
    }

    pub(super) fn from_packed(
        packed: Tensor,
        context: usize,
        pred_len: usize,
        aux_channels: usize,
        valid_target_bars: usize,
    ) -> Self {
        let b = packed.size()[0];
        let length = (context + pred_len) as i64;
        assert_eq!(
            packed.size()[1] as usize,
            Self::row_width(context, pred_len, aux_channels)
        );
        let log_prices = packed.narrow(1, 0, length * 4).reshape([b, length, 4]);
        let valid = packed.narrow(1, length * 4, length);
        let aux = packed
            .narrow(1, length * 5, length * aux_channels as i64)
            .reshape([b, length, aux_channels as i64]);
        let market_cum = packed.narrow(1, length * (5 + aux_channels as i64), length);
        let anchor = packed
            .narrow(1, length * (6 + aux_channels as i64), 1)
            .reshape([b]);
        Self {
            log_prices,
            valid,
            aux,
            market_cum,
            anchor,
            valid_target_bars,
            packed,
            context,
            pred_len,
            aux_channels,
        }
    }

    /// Rows in the packed block: the batch dimension a captured graph or a resident buffer
    /// is fixed to.
    pub fn rows(&self) -> i64 {
        self.packed.size()[0]
    }

    pub fn to_device(self, device: Device) -> Self {
        Self::from_packed(
            self.packed.to_device_(device, Kind::Float, true, false),
            self.context,
            self.pred_len,
            self.aux_channels,
            self.valid_target_bars,
        )
    }

    /// A device-resident batch of this batch's exact shape whose packed storage NEVER
    /// moves, so [`Self::upload`] can refill it in place.
    ///
    /// Two things need that. A captured CUDA graph records addresses, so its input has to
    /// be one fixed buffer rather than whatever [`Self::to_device`] allocated this step.
    /// And even eagerly, `to_device` allocates and frees the whole packed row block every
    /// step - 114 MB at batch 256 - which the caching allocator has to keep re-serving.
    pub fn resident(&self, device: Device) -> Self {
        Self::from_packed(
            Tensor::zeros(self.packed.size(), (Kind::Float, device)),
            self.context,
            self.pred_len,
            self.aux_channels,
            self.valid_target_bars,
        )
    }

    /// A host copy of this batch, in pinned memory when a device is given.
    ///
    /// The capture audit builds both to measure what the loader's `pin_memory` buys: a
    /// pageable source makes the H2D copy blocking whatever `non_blocking` says, because the
    /// driver has to stage it, so the host waits for the outstanding step to drain.
    pub fn host_copy(&self, pinned: Option<Device>) -> Self {
        let packed = self.packed.to_device_(Device::Cpu, Kind::Float, true, false);
        Self::from_packed(
            match pinned {
                Some(device) => packed.pin_memory(device),
                None => packed,
            },
            self.context,
            self.pred_len,
            self.aux_channels,
            self.valid_target_bars,
        )
    }

    /// Refill `resident` from this host batch, asynchronously.
    ///
    /// `Corpus::host_batch` pins the packed block whenever the corpus is prepared for a
    /// CUDA device, which is what makes the non-blocking copy safe: libtorch's caching
    /// host allocator holds the block until the copy retires, so the host batch may drop
    /// the moment this returns.
    pub fn upload(&self, resident: &mut Self) -> Result<()> {
        ensure!(
            self.packed.size() == resident.packed.size(),
            "resident batch shape does not match the host batch"
        );
        crate::torch::cuda::copy_nonblocking(&mut resident.packed, &self.packed)
            .map_err(|err| anyhow::anyhow!("uploading the packed batch: {err}"))?;
        resident.valid_target_bars = self.valid_target_bars;
        Ok(())
    }
}

impl Corpus {
    pub fn load(
        directory: &Path,
        requested: &[String],
        context: usize,
        pred_len: usize,
        common_context: usize,
        features: &FeatureSet,
        market_min_cross_section: usize,
    ) -> Result<Self> {
        ensure!(
            context > 0 && pred_len > 0 && common_context >= context,
            "invalid context or forecast length"
        );
        ensure!(
            market_min_cross_section > 0,
            "market steps need a positive minimum cross-section"
        );
        let purge = pred_len.max(100);
        let minimum_source_bars = common_context + 1;
        let minimum_training_bars = common_context + 1;
        let requested_set: BTreeSet<_> = requested.iter().cloned().collect();
        ensure!(
            requested_set.len() == requested.len(),
            "duplicate requested ticker"
        );
        let mut files = Vec::new();
        let mut found = BTreeSet::new();
        let mut excluded_tickers = Vec::new();
        for entry in fs::read_dir(directory).context("reading five-minute corpus directory")? {
            let path = entry?.path();
            let Ok((ticker, resolution)) = parse_bar_file_name(&path) else {
                continue;
            };
            if resolution != 300 || (!requested_set.is_empty() && !requested_set.contains(&ticker))
            {
                continue;
            }
            let file = BarFile::open(&path)?;
            ensure!(
                file.symbol() == ticker && file.res_secs() == 300,
                "corpus filename/header mismatch"
            );
            found.insert(ticker.clone());
            if file.is_empty() {
                excluded_tickers.push(ExcludedTicker {
                    ticker,
                    reason: "empty source corpus".into(),
                });
            } else {
                files.push(file);
            }
        }
        for ticker in &requested_set {
            ensure!(
                found.contains(ticker),
                "requested ticker {ticker} has no five-minute corpus"
            );
        }
        files.sort_unstable_by(|a, b| a.symbol().cmp(b.symbol()));
        ensure!(!files.is_empty(), "no eligible five-minute ticker corpora");
        let workers = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(8);
        let gather_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|i| format!("timexer-data-{i}"))
            .build()?;
        let bounds = gather_pool.install(|| shared_bounds(&files))?;
        files.retain(|file| {
            let edges = bounds.map(|bound| file.index_at_or_after(bound));
            let eligible = edges[0] >= minimum_training_bars;
            if !eligible {
                excluded_tickers.push(ExcludedTicker {
                    ticker: file.symbol().to_owned(),
                    reason: "insufficient purged training history".into(),
                });
            }
            eligible
        });
        for ticker in &requested_set {
            ensure!(
                files.iter().any(|file| file.symbol() == ticker),
                "requested ticker {ticker} has insufficient purged training history"
            );
        }
        ensure!(
            !files.is_empty(),
            "no ticker has sufficient purged training history"
        );
        excluded_tickers.sort_unstable_by(|a, b| {
            a.ticker
                .cmp(&b.ticker)
                .then_with(|| a.reason.cmp(&b.reason))
        });
        let mut tickers: Vec<CorpusTicker> = gather_pool.install(|| {
            files
                .into_par_iter()
                .map(|file| {
                    let contract = filtered_contract(
                        file.symbol(),
                        file.bars(),
                        context,
                        pred_len,
                        common_context,
                        bounds,
                    )
                    .with_context(|| {
                        format!(
                            "authenticating five-minute ticker {} from {}",
                            file.symbol(),
                            file.path().display()
                        )
                    })?;
                    file.advise_random_access()?;
                    Ok(CorpusTicker { contract, file })
                })
                .collect::<Result<_>>()
        })?;
        tickers.retain(|ticker| {
            let eligible = ticker.contract.train_end >= minimum_training_bars;
            if !eligible {
                excluded_tickers.push(ExcludedTicker {
                    ticker: ticker.contract.ticker.clone(),
                    reason:
                        "insufficient valid purged training history after quarantining invalid OHLC"
                            .into(),
                });
            }
            eligible
        });
        for ticker in &requested_set {
            ensure!(tickers.iter().any(|data| &data.contract.ticker == ticker), "requested ticker {ticker} has insufficient valid purged training history after quarantining invalid OHLC");
        }
        excluded_tickers.sort_unstable_by(|a, b| {
            a.ticker
                .cmp(&b.ticker)
                .then_with(|| a.reason.cmp(&b.reason))
        });
        ensure!(
            !tickers.is_empty(),
            "no ticker has sufficient valid purged training history"
        );
        let (exogenous, market) = gather_pool.install(|| {
            exogenous_series(directory, features, &tickers, market_min_cross_section)
        })?;
        let mut train_refs = Vec::new();
        let mut validation_refs = Vec::new();
        let mut train_target_bars = 0;
        let mut validation_target_bars = 0;
        let mut validation_remainder_bars = 0;
        for (ticker, data) in tickers.iter().enumerate() {
            let c = &data.contract;
            train_refs.extend(
                (common_context - 1..c.train_end - 1)
                    .step_by(pred_len)
                    .map(|origin| WindowRef { ticker, origin }),
            );
            train_target_bars += c.train_end - common_context;
            let start = c.boundaries[1].max(common_context);
            let available =
                retained_partition_end(c.boundaries[2], c.valid_bars, purge).saturating_sub(start);
            validation_refs.extend((0..available / pred_len).map(|i| WindowRef {
                ticker,
                origin: start - 1 + i * pred_len,
            }));
            validation_target_bars += available / pred_len * pred_len;
            validation_remainder_bars += available % pred_len;
        }
        ensure!(
            !train_refs.is_empty() && !validation_refs.is_empty(),
            "empty unlocked corpus split"
        );
        Ok(Self {
            contract: CorpusContract {
                schema: SCHEMA.into(),
                tickers: tickers.iter().map(|t| t.contract.clone()).collect(),
                boundary_timestamps: bounds,
                context,
                pred_len,
                common_context,
                purge,
                features: *features,
                auxiliary_schema: features.schema(),
                market_fingerprint: market_fingerprint(&tickers, bounds, market_min_cross_section),
                market_min_cross_section,
                spy_fingerprint: features
                    .spy
                    .then(|| file_sha256(bar_file_path(directory, SPY, 300)))
                    .transpose()?,
                minimum_source_bars,
                minimum_training_bars,
                train_target_bars,
                validation_target_bars,
                validation_remainder_bars,
                excluded_tickers: excluded_tickers.clone(),
            },
            train_refs,
            validation_refs,
            excluded_tickers,
            market,
            tickers,
            exogenous,
            gather_pool,
            device: Device::Cpu,
        })
    }

    pub fn prepare(&mut self, device: Device) {
        self.device = device;
    }
    pub fn ticker(&self, reference: WindowRef) -> &CorpusTicker {
        &self.tickers[reference.ticker]
    }

    pub fn host_batch(&self, refs: &[WindowRef]) -> Result<Batch> {
        ensure!(!refs.is_empty(), "cannot construct an empty batch");
        let context = self.contract.context;
        let pred_len = self.contract.pred_len;
        let features = &self.contract.features;
        let width = Batch::row_width(context, pred_len, features.channels());
        let sources = refs
            .iter()
            .map(|reference| {
                let ticker = self
                    .tickers
                    .get(reference.ticker)
                    .context("unknown batch ticker")?;
                let targets = ticker
                    .target_count(reference.origin)
                    .context("origin is outside unlocked purged targets")?;
                Ok((ticker, reference.origin, targets))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut packed = Tensor::empty(
            [refs.len() as i64, width as i64],
            (Kind::Float, Device::Cpu),
        );
        if self.device.is_cuda() {
            packed = packed.pin_memory(self.device);
        }
        // The new tensor owns this exclusive contiguous CPU allocation; workers receive disjoint rows.
        let output = unsafe {
            std::slice::from_raw_parts_mut(packed.data_ptr().cast::<f32>(), refs.len() * width)
        };
        self.gather_pool.install(|| {
            output
                .par_chunks_mut(width)
                .zip(sources.par_iter())
                .for_each(|(row, &(ticker, origin, targets))| {
                    fill_row(
                        row,
                        ticker.file.bars(),
                        &ticker.contract,
                        features,
                        &self.exogenous,
                        origin,
                        targets,
                    );
                })
        });
        Ok(Batch::from_packed(
            packed,
            context,
            pred_len,
            features.channels(),
            sources.iter().map(|(_, _, targets)| *targets).sum(),
        ))
    }

    pub fn batch(&self, refs: &[WindowRef], device: Device) -> Result<Batch> {
        Ok(self.host_batch(refs)?.to_device(device))
    }
}

fn market_fingerprint(tickers: &[CorpusTicker], bounds: [i64; 3], min_cross_section: usize) -> String {
    let mut digest = ring::digest::Context::new(&ring::digest::SHA256);
    for ticker in tickers {
        digest.update(ticker.contract.ticker.as_bytes());
        digest.update(b":");
        digest.update(ticker.contract.fingerprint.as_bytes());
        digest.update(b";");
    }
    for bound in bounds {
        digest.update(&bound.to_le_bytes());
    }
    digest.update(b"market-steps-min-cross-section:");
    digest.update(&(min_cross_section as u64).to_le_bytes());
    digest
        .finish()
        .as_ref()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn exogenous_series(
    directory: &Path,
    features: &FeatureSet,
    tickers: &[CorpusTicker],
    market_min_cross_section: usize,
) -> Result<(Exogenous, MarketSummary)> {
    let first = tickers
        .iter()
        .filter_map(|ticker| ticker.file.first_ts_ms())
        .min()
        .context("empty exogenous timestamp universe")?;
    let last = tickers
        .iter()
        .filter_map(|ticker| ticker.file.last_ts_ms())
        .max()
        .context("empty exogenous timestamp universe")?;
    let grid = Grid::new(first, last)?;
    let sources: Vec<_> = tickers
        .iter()
        .map(|ticker| {
            (
                ticker.file.bars(),
                ticker.contract.invalid_ohlc_indices.as_slice(),
            )
        })
        .collect();
    let steps = market_steps(&sources, grid, market_min_cross_section);
    let mut exogenous = Exogenous {
        market: features.market.then(|| steps.series()),
        spy: None,
        market_cum: steps.path(),
    };
    if features.spy {
        let file = BarFile::open(&bar_file_path(directory, SPY, 300))
            .with_context(|| format!("loading the {SPY} exogenous variate"))?;
        ensure!(
            file.symbol() == SPY && file.res_secs() == 300 && !file.is_empty(),
            "{SPY} exogenous corpus header mismatch or empty"
        );
        let bars = file.bars();
        for (index, bar) in bars.iter().enumerate() {
            ensure!(
                (index == 0 || bars[index - 1].ts() < bar.ts())
                    && bar.ts().rem_euclid(RESOLUTION_MS) == 0,
                "{SPY}: timestamps must be strictly increasing on the five-minute grid at raw bar {index}"
            );
        }
        let invalid: Vec<usize> = bars
            .iter()
            .enumerate()
            .filter_map(|(index, bar)| (!valid_ohlc(bar)).then_some(index))
            .collect();
        exogenous.spy = Some(single_series(bars, &invalid, grid));
    }
    Ok((exogenous, steps.summary()))
}

fn fill_row(
    row: &mut [f32],
    bars: &[PackedBar],
    contract: &DataContract,
    features: &FeatureSet,
    exogenous: &Exogenous,
    origin: usize,
    targets: usize,
) {
    let context = contract.context;
    let length = context + contract.pred_len;
    let start = origin + 1 - context;
    let aux_channels = features.channels();
    let invalid = &contract.invalid_ohlc_indices;
    row.fill(0.0);
    let (prices, rest) = row.split_at_mut(length * 4);
    let (valid, rest) = rest.split_at_mut(length);
    let (aux, rest) = rest.split_at_mut(length * aux_channels);
    let (market_cum, anchor) = rest.split_at_mut(length);
    let market_anchor = exogenous.market_cum.at(bars[raw_index(origin, invalid)].ts());
    let c_last = f64::from(bars[raw_index(origin, invalid)].close);
    let ln_anchor = c_last.ln();
    let mut auxiliary = AuxiliaryCursor::new(
        features,
        exogenous,
        start
            .checked_sub(1)
            .map(|i| &bars[raw_index(i, invalid)]),
    );
    for (position, bar) in ValidBars::new(bars, invalid, start, context + targets).enumerate() {
        for (channel, value) in [bar.open, bar.high, bar.low, bar.close]
            .into_iter()
            .enumerate()
        {
            prices[position * 4 + channel] = (f64::from(value).ln() - ln_anchor) as f32;
        }
        valid[position] = 1.0;
        market_cum[position] = (exogenous.market_cum.at(bar.ts()) - market_anchor) as f32;
        if aux_channels > 0 {
            let offset = position * aux_channels;
            auxiliary.write(
                bar,
                &mut aux[offset..offset + aux_channels],
                position >= context,
            );
        }
    }
    anchor[0] = c_last as f32;
}

fn raw_index(logical_index: usize, invalid: &[usize]) -> usize {
    let (mut left, mut right) = (0, invalid.len());
    while left < right {
        let middle = left + (right - left) / 2;
        if invalid[middle] - middle <= logical_index {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    logical_index + left
}

struct ValidBars<'a> {
    bars: &'a [PackedBar],
    invalid: &'a [usize],
    invalid_cursor: usize,
    raw: usize,
    remaining: usize,
}

impl<'a> ValidBars<'a> {
    fn new(bars: &'a [PackedBar], invalid: &'a [usize], start: usize, length: usize) -> Self {
        let raw = raw_index(start, invalid);
        Self {
            bars,
            invalid,
            invalid_cursor: raw - start,
            raw,
            remaining: length,
        }
    }
}

impl<'a> Iterator for ValidBars<'a> {
    type Item = &'a PackedBar;
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        while self.invalid.get(self.invalid_cursor) == Some(&self.raw) {
            self.raw += 1;
            self.invalid_cursor += 1;
        }
        let bar = &self.bars[self.raw];
        self.raw += 1;
        self.remaining -= 1;
        Some(bar)
    }
}

fn shared_bounds(files: &[BarFile]) -> Result<[i64; 3]> {
    let first = files
        .iter()
        .filter_map(BarFile::first_ts_ms)
        .min()
        .context("empty timestamp universe")?;
    let last = files
        .iter()
        .filter_map(BarFile::last_ts_ms)
        .max()
        .context("empty timestamp universe")?;
    ensure!(
        first.rem_euclid(RESOLUTION_MS) == 0 && last.rem_euclid(RESOLUTION_MS) == 0,
        "off-grid universe timestamps"
    );
    let slots = usize::try_from((last - first) / RESOLUTION_MS + 1)?;
    ensure!(
        slots <= 20_000_000,
        "five-minute corpus spans more than 190 years"
    );
    let words = slots.div_ceil(64);
    let occupied = files
        .par_iter()
        .try_fold(
            || vec![0u64; words],
            |mut bits, file| -> Result<_> {
                for bar in file.bars() {
                    let offset = bar.ts() - first;
                    ensure!(
                        offset >= 0 && offset % RESOLUTION_MS == 0,
                        "off-grid timestamp in {}",
                        file.symbol()
                    );
                    if !valid_ohlc(bar) {
                        continue;
                    }
                    let index = usize::try_from(offset / RESOLUTION_MS)?;
                    ensure!(index < slots, "timestamp outside header span");
                    bits[index / 64] |= 1 << (index % 64);
                }
                Ok(bits)
            },
        )
        .try_reduce(
            || vec![0u64; words],
            |mut left, right| {
                for (a, b) in left.iter_mut().zip(right) {
                    *a |= b;
                }
                Ok(left)
            },
        )?;
    quantile_bounds(&occupied, first)
}

fn quantile_bounds(occupied: &[u64], first: i64) -> Result<[i64; 3]> {
    let count: usize = occupied.iter().map(|word| word.count_ones() as usize).sum();
    ensure!(
        count >= 10,
        "too few distinct timestamps for four chronological partitions"
    );
    let ranks = [count * 7 / 10, count * 8 / 10, count * 9 / 10];
    let mut result = [0; 3];
    let mut seen = 0;
    let mut boundary = 0;
    for (word_index, &word) in occupied.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            if boundary < 3 && seen == ranks[boundary] {
                result[boundary] = first + (word_index * 64 + bit) as i64 * RESOLUTION_MS;
                boundary += 1;
            }
            seen += 1;
            remaining &= remaining - 1;
        }
    }
    ensure!(boundary == 3, "incomplete global timestamp quantiles");
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_quantiles_count_calendar_positions_once() {
        let bits = [(1u64 << 20) - 1];
        assert_eq!(
            quantile_bounds(&bits, 300_000).unwrap(),
            [15 * 300_000, 17 * 300_000, 19 * 300_000]
        );
        let sparse = [0x55555];
        assert_eq!(
            quantile_bounds(&sparse, 0).unwrap(),
            [14 * 300_000, 16 * 300_000, 18 * 300_000]
        );
    }

    #[test]
    fn sparse_invalid_mapping_matches_valid_observed_order_for_every_small_pattern() {
        let bars = (0..8)
            .map(|i| PackedBar {
                ts_ms: i,
                close: i as f32,
                ..PackedBar::default()
            })
            .collect::<Vec<_>>();
        for pattern in 0usize..256 {
            let invalid = (0..8)
                .filter(|i| pattern & (1 << i) != 0)
                .collect::<Vec<_>>();
            let expected = (0..8).filter(|i| !invalid.contains(i)).collect::<Vec<_>>();
            for (logical, &raw) in expected.iter().enumerate() {
                assert_eq!(raw_index(logical, &invalid), raw);
            }
            for start in 0..=expected.len() {
                let actual = ValidBars::new(&bars, &invalid, start, expected.len() - start)
                    .map(|bar| bar.ts() as usize)
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected[start..]);
            }
        }
    }

    #[test]
    fn pooled_rows_preserve_ticker_values_and_mask_partial_targets() {
        let bars: Vec<_> = (0..20)
            .map(|i| PackedBar {
                ts_ms: i * RESOLUTION_MS,
                open: i as f32 + 1.0,
                high: i as f32 + 3.0,
                low: i as f32 + 0.5,
                close: i as f32 + 2.0,
                volume: 100.0 + i as f32,
                ..Default::default()
            })
            .collect();
        let contract = DataContract {
            schema: String::new(),
            ticker: "ONE".into(),
            fingerprint: String::new(),
            source_bars: bars.len(),
            valid_bars: bars.len(),
            invalid_ohlc_indices: Vec::new(),
            boundaries: [10, 13, 17],
            boundary_timestamps: [0; 3],
            context: 4,
            pred_len: 3,
            purge: 1,
            train_end: 9,
            common_context: 4,
        };
        let volume = FeatureSet {
            volume: true,
            ..FeatureSet::NONE
        };
        let exogenous = Exogenous {
            market: None,
            spy: None,
            market_cum: market_steps(&[(&bars, &[])], Grid::new(0, 19 * RESOLUTION_MS).unwrap(), 1)
                .path(),
        };
        let mut row = vec![f32::NAN; Batch::row_width(4, 3, 2)];
        assert_eq!(row.len(), 57);
        fill_row(&mut row, &bars, &contract, &volume, &exogenous, 6, 2);
        assert_eq!(row[3], (5.0f64 / 8.0).ln() as f32);
        assert_eq!(row[0], (4.0f64 / 8.0).ln() as f32);
        assert_eq!(row[15], 0.0);
        assert_eq!(row[19], (9.0f64 / 8.0).ln() as f32);
        assert_eq!(&row[24..28], &[0.0; 4]);
        assert_eq!(&row[28..35], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
        assert_eq!(row[35], (103.0f64.ln() - 102.0f64.ln()) as f32);
        assert_eq!(row[36], 1.0);
        assert_eq!(&row[41..43], &[(106.0f64.ln() - 105.0f64.ln()) as f32, 1.0]);
        assert_eq!(&row[43..49], &[0.0; 6], "history channels leak beyond the context");
        for position in 0..6 {
            assert!(
                (row[49 + position] - row[position * 4 + 3]).abs() <= 1e-6,
                "a one-ticker universe's market path is the ticker's own close path"
            );
        }
        assert_eq!(row[49 + 3], 0.0);
        assert_eq!(row[55], 0.0, "masked horizon bars carry no market path");
        assert_eq!(row[56], 8.0);
        let batch = Batch::from_packed(
            Tensor::from_slice(&row).reshape([1, row.len() as i64]),
            4,
            3,
            2,
            2,
        )
        .to_device(Device::Cpu);
        assert_eq!(batch.log_prices.size(), [1, 7, 4]);
        assert_eq!(batch.valid.size(), [1, 7]);
        assert_eq!(batch.aux.size(), [1, 7, 2]);
        assert_eq!(batch.anchor.size(), [1]);
        assert_eq!(batch.anchor.double_value(&[0]), 8.0);
        assert_eq!(
            Vec::<f32>::try_from(batch.valid.reshape([-1])).unwrap(),
            &row[28..35]
        );
        assert_eq!(batch.log_prices.double_value(&[0, 4, 3]), f64::from(row[19]));
        let mut doubled_bars = bars.clone();
        for bar in &mut doubled_bars {
            bar.open *= 2.0;
            bar.high *= 2.0;
            bar.low *= 2.0;
            bar.close *= 2.0;
        }
        let mut doubled = vec![0.0; row.len()];
        fill_row(&mut doubled, &doubled_bars, &contract, &volume, &exogenous, 6, 2);
        assert_eq!(&doubled[..56], &row[..56], "representation must be scale-free");
        assert_eq!(doubled[56], 16.0);
        let mut future_bars = bars.clone();
        for bar in &mut future_bars[7..] {
            bar.open *= 1000.0;
            bar.high *= 1000.0;
            bar.low *= 1000.0;
            bar.close *= 1000.0;
        }
        let mut changed = vec![0.0; row.len()];
        fill_row(&mut changed, &future_bars, &contract, &volume, &exogenous, 6, 2);
        assert_eq!(
            &changed[..16],
            &row[..16],
            "future targets changed historical inputs"
        );
        assert_eq!(changed[56], row[56], "future targets changed the anchor");
        let mut flat_bars = bars.clone();
        for bar in &mut flat_bars[..7] {
            bar.open = 8.0;
            bar.high = 8.0;
            bar.low = 8.0;
            bar.close = 8.0;
        }
        fill_row(&mut changed, &flat_bars, &contract, &volume, &exogenous, 6, 2);
        assert_eq!(&changed[..16], &[0.0; 16]);
        assert_eq!(changed[19], (9.0f64 / 8.0).ln() as f32);
    }

    #[test]
    fn pooled_epoch_keeps_delisted_tickers_and_covers_each_training_target_once() {
        use shared::bars::{bar_file_path, write_bar_file};
        struct Scratch(std::path::PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = Scratch(std::path::PathBuf::from(format!(
            "/var/tmp/timexer-corpus-test-{}-{nonce}",
            std::process::id()
        )));
        fs::create_dir(&directory.0).unwrap();
        let bars: Vec<_> = (0..10_000)
            .map(|i| {
                let price = 100.0 + i as f32 * 0.01;
                PackedBar {
                    ts_ms: 1_500_000_000_000 + i * RESOLUTION_MS,
                    open: price,
                    high: price + 1.0,
                    low: price - 1.0,
                    close: price + 0.5,
                    volume: 100.0,
                    vwap: price,
                    trades: 1,
                }
            })
            .collect();
        for (symbol, source) in [
            ("ALIVE", bars.as_slice()),
            ("DELISTED", &bars[..5_000]),
            ("MINIMAL", &bars[..33]),
            ("NEW", &bars[8_000..]),
        ] {
            write_bar_file(
                &bar_file_path(&directory.0, symbol, 300),
                symbol,
                300,
                source,
            )
            .unwrap();
        }
        let mut damaged = bars[..5_000].to_vec();
        for index in [0, 1, 47, 48, 49, 4000] {
            damaged[index].open = 0.0;
            damaged[index].high = 0.0;
            damaged[index].low = 0.0;
            damaged[index].close = 0.0;
        }
        write_bar_file(
            &bar_file_path(&directory.0, "DAMAGED", 300),
            "DAMAGED",
            300,
            &damaged,
        )
        .unwrap();
        let mut invalid = bars.clone();
        for bar in &mut invalid {
            bar.ts_ms -= 10_000 * RESOLUTION_MS;
            bar.close = f32::NAN;
        }
        write_bar_file(
            &bar_file_path(&directory.0, "INVALID", 300),
            "INVALID",
            300,
            &invalid,
        )
        .unwrap();
        let ancient = bars[..2]
            .iter()
            .enumerate()
            .map(|(index, bar)| PackedBar {
                ts_ms: bars[0].ts() - (2 - index) as i64 * RESOLUTION_MS,
                ..*bar
            })
            .collect::<Vec<_>>();
        write_bar_file(
            &bar_file_path(&directory.0, "ANCIENT", 300),
            "ANCIENT",
            300,
            &ancient,
        )
        .unwrap();
        let features = FeatureSet {
            spy: false,
            ..FeatureSet::ALL
        };
        let corpus = Corpus::load(&directory.0, &[], 16, 7, 32, &features, 1).unwrap();
        assert_eq!(corpus.contract.market_min_cross_section, 1);
        assert_eq!(corpus.contract.features, features);
        assert!(corpus.contract.spy_fingerprint.is_none());
        assert_eq!(
            corpus
                .contract
                .tickers
                .iter()
                .map(|t| t.ticker.as_str())
                .collect::<Vec<_>>(),
            ["ALIVE", "DAMAGED", "DELISTED", "MINIMAL"]
        );
        assert!(corpus.excluded_tickers.iter().any(|t| t.ticker == "NEW"));
        assert!(corpus.validation_refs.iter().all(|r| r.ticker == 0));
        assert_eq!(corpus.contract.boundary_timestamps[0], bars[6999].ts());
        assert_eq!(corpus.contract.tickers[0].train_end, 6999 - 100);
        assert_eq!(corpus.contract.tickers[1].train_end, 4994);
        assert_eq!(
            corpus.contract.tickers[1].invalid_ohlc_indices,
            [0, 1, 47, 48, 49, 4000]
        );
        assert_eq!(corpus.contract.tickers[2].train_end, 5000);
        assert_eq!(corpus.contract.tickers[3].train_end, 33);
        assert!(corpus
            .excluded_tickers
            .iter()
            .any(|ticker| ticker.ticker == "INVALID"));
        let reference = WindowRef {
            ticker: 1,
            origin: 46,
        };
        let source = corpus.ticker(reference);
        assert_eq!(source.timestamp(0), bars[2].ts());
        assert_eq!(source.timestamp(46), bars[51].ts());
        let actual = source.candle_window(46, 16, 7).unwrap();
        let valid_raw = (0..5000)
            .filter(|i| !source.contract.invalid_ohlc_indices.contains(i))
            .collect::<Vec<_>>();
        for (candle, &raw) in actual.iter().zip(&valid_raw[30..54]) {
            assert_eq!(candle.close, bars[raw].close);
        }
        let selected = corpus.host_batch(&[reference]).unwrap();
        let log_prices: Vec<f32> = Vec::try_from(selected.log_prices.reshape([-1])).unwrap();
        let valid: Vec<f32> = Vec::try_from(selected.valid.reshape([-1])).unwrap();
        assert_eq!(log_prices.len(), 23 * 4);
        assert_eq!(valid, vec![1.0; 23]);
        let closes: Vec<f64> = valid_raw[31..54]
            .iter()
            .map(|&raw| f64::from(bars[raw].close))
            .collect();
        let c_last = closes[15];
        assert_eq!(selected.anchor.double_value(&[0]), c_last);
        let market_cum: Vec<f32> = Vec::try_from(selected.market_cum.reshape([-1])).unwrap();
        assert_eq!(market_cum.len(), 23);
        assert_eq!(market_cum[15], 0.0);
        assert!(market_cum[..23].iter().all(|v| v.is_finite()));
        assert_eq!(corpus.contract.market_fingerprint.len(), 64);
        assert_eq!(log_prices[15 * 4 + 3], 0.0);
        for (position, close) in closes.iter().enumerate() {
            let expected = (close / c_last).ln();
            assert!((f64::from(log_prices[position * 4 + 3]) - expected).abs() <= 1e-7);
        }
        let auxiliary: Vec<f32> = Vec::try_from(selected.aux.reshape([-1])).unwrap();
        assert_eq!(auxiliary.len(), 23 * 10);
        for (position, &raw) in valid_raw[31..54].iter().enumerate() {
            let channels = &auxiliary[position * 10..position * 10 + 10];
            assert!(channels[..4].iter().all(|value| value.abs() <= 1.0));
            let gap = if raw == 50 { [1.0, 4.0f32.ln()] } else { [0.0, 0.0] };
            assert_eq!(&channels[4..6], &gap);
            if position < 16 {
                assert_eq!(&channels[6..8], &[0.0, 1.0]);
                let market =
                    (f64::from(bars[raw].close) / f64::from(bars[raw - 1].close)).ln() as f32;
                assert!((channels[8] - market).abs() < 1e-7 && channels[9] == 1.0);
            } else {
                assert_eq!(&channels[6..10], &[0.0; 4], "future history channels must be blank");
            }
        }
        let minimal = corpus
            .train_refs
            .iter()
            .filter(|r| r.ticker == 3)
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(
            minimal,
            [WindowRef {
                ticker: 3,
                origin: 31
            }]
        );
        assert_eq!(corpus.host_batch(&minimal).unwrap().valid_target_bars, 1);

        let mut total = 0;
        for (ticker, metadata) in corpus.contract.tickers.iter().enumerate() {
            let mut coverage = vec![0u8; metadata.valid_bars];
            for reference in corpus.train_refs.iter().filter(|r| r.ticker == ticker) {
                let count = corpus
                    .ticker(*reference)
                    .target_count(reference.origin)
                    .unwrap();
                for used in &mut coverage[reference.origin + 1..reference.origin + 1 + count] {
                    *used += 1;
                }
                total += count;
            }
            assert!(coverage[..32].iter().all(|&v| v == 0));
            assert!(coverage[32..metadata.train_end].iter().all(|&v| v == 1));
            assert!(coverage[metadata.train_end..].iter().all(|&v| v == 0));
        }
        assert_eq!(total, corpus.contract.train_target_bars);
        let refs = [corpus.train_refs[0], *corpus.train_refs.last().unwrap()];
        let batch = corpus.host_batch(&refs).unwrap();
        assert_eq!(batch.valid.size(), [2, 23]);
        assert_eq!(batch.anchor.size(), [2]);
        assert!(batch.anchor.gt(0.0).all().int64_value(&[]) == 1);
        assert_eq!(
            batch
                .valid
                .narrow(1, 16, 7)
                .sum(Kind::Float)
                .int64_value(&[]) as usize,
            batch.valid_target_bars
        );
        assert_eq!(
            batch.valid_target_bars,
            refs.iter()
                .map(|r| corpus.ticker(*r).target_count(r.origin).unwrap())
                .sum::<usize>()
        );
    }

    /// Every accessor on a `Batch` is a VIEW of one packed block. That is what lets a
    /// resident batch be refilled by a single copy into a fixed address - which is what a
    /// captured CUDA graph reads and what keeps a 114 MB allocate-and-free off every step.
    /// If `from_packed` ever returned copies instead, `upload` would refresh nothing and the
    /// model would train on whatever the resident batch held at construction.
    #[test]
    fn uploading_a_host_batch_refreshes_every_view_of_the_resident_batch() {
        let (context, pred_len, aux_channels) = (4, 3, 2);
        let width = Batch::row_width(context, pred_len, aux_channels);
        let batch_of = |rows: usize, offset: f32| {
            let values: Vec<f32> = (0..rows * width)
                .map(|index| offset + index as f32)
                .collect();
            Batch::from_packed(
                Tensor::from_slice(&values).reshape([rows as i64, width as i64]),
                context,
                pred_len,
                aux_channels,
                rows,
            )
        };
        let first = batch_of(2, 1.0);
        let second = batch_of(2, 1000.0);
        let mut resident = first.resident(Device::Cpu);
        first.upload(&mut resident).unwrap();
        for (mine, theirs) in [
            (&resident.log_prices, &first.log_prices),
            (&resident.valid, &first.valid),
            (&resident.aux, &first.aux),
            (&resident.market_cum, &first.market_cum),
            (&resident.anchor, &first.anchor),
        ] {
            assert!(mine.equal(theirs), "the first upload did not land in a view");
        }
        second.upload(&mut resident).unwrap();
        for (mine, theirs) in [
            (&resident.log_prices, &second.log_prices),
            (&resident.valid, &second.valid),
            (&resident.aux, &second.aux),
            (&resident.market_cum, &second.market_cum),
            (&resident.anchor, &second.anchor),
        ] {
            assert!(
                mine.equal(theirs),
                "a refill left a view reading the previous batch"
            );
        }
        assert_eq!(resident.valid_target_bars, second.valid_target_bars);
        // A shape the capture never recorded must be refused, not silently truncated.
        assert!(batch_of(1, 0.0).upload(&mut resident).is_err());
    }

    #[test]
    fn corpus_can_be_shared_with_a_prefetch_worker() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<Corpus>();
    }
}
