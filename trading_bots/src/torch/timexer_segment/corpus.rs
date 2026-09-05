use std::{collections::BTreeSet, fs, path::Path};

use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use shared::{
    bars::{parse_bar_file_name, BarFile, PackedBar},
    report::CandleBar,
};
use tch::{Device, Kind, Tensor};

use super::data::{retained_partition_end, valid_ohlc, DataContract, Dataset};

const RESOLUTION_MS: i64 = 300_000;
const SCHEMA: &str = "timexer-pooled-mmap-v1;independent-ticker-rows;all-valid-source-unique-utc-grid-quantiles70:10:10:10;purge-only-at-observed-partition-boundaries;train-only-per-ticker-scaler;next-valid-observed-bars;invalid-ohlc-rows-quarantined-without-repair;disjoint-target-epoch-with-masked-remainder;validation-disjoint-complete-targets;terminal-test-locked";

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
    pub volume_features: bool,
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
        if origin < c.scaler_fit_bars - 1 {
            return Some(c.pred_len.min(c.scaler_fit_bars - origin - 1));
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
    tickers: Vec<CorpusTicker>,
    gather_pool: rayon::ThreadPool,
    device: Device,
}

pub struct Batch {
    pub inputs: Tensor,
    pub targets: Tensor,
    pub target_mask: Tensor,
    pub price_scaling: Tensor,
    pub geometry_context: Tensor,
    pub auxiliary: Option<Tensor>,
    pub valid_target_bars: usize,
    packed: Tensor,
    context: usize,
    pred_len: usize,
    volume_features: bool,
}

impl Batch {
    fn from_packed(
        packed: Tensor,
        context: usize,
        pred_len: usize,
        volume_features: bool,
        valid_target_bars: usize,
    ) -> Self {
        let b = packed.size()[0];
        let input_len = context as i64 * 4;
        let target_len = pred_len as i64 * 4;
        let aux_len = if volume_features {
            context as i64 * 2
        } else {
            0
        };
        let inputs = packed
            .narrow(1, 0, input_len)
            .reshape([b, context as i64, 4]);
        let targets = packed
            .narrow(1, input_len, target_len)
            .reshape([b, pred_len as i64, 4]);
        let auxiliary = volume_features.then(|| {
            packed
                .narrow(1, input_len + target_len, aux_len)
                .reshape([b, context as i64, 2])
        });
        let target_mask = packed
            .narrow(1, input_len + target_len + aux_len, pred_len as i64)
            .reshape([b, pred_len as i64]);
        let price_scaling = packed
            .narrow(1, input_len + target_len + aux_len + pred_len as i64, 8)
            .reshape([b, 2, 4]);
        let geometry_context = packed
            .narrow(1, input_len + target_len + aux_len + pred_len as i64 + 8, 3)
            .reshape([b, 3]);
        Self {
            inputs,
            targets,
            target_mask,
            price_scaling,
            geometry_context,
            auxiliary,
            valid_target_bars,
            packed,
            context,
            pred_len,
            volume_features,
        }
    }

    pub fn to_device(self, device: Device) -> Self {
        Self::from_packed(
            self.packed.to_device_(device, Kind::Float, true, false),
            self.context,
            self.pred_len,
            self.volume_features,
            self.valid_target_bars,
        )
    }
}

impl Corpus {
    pub fn load(
        directory: &Path,
        requested: &[String],
        context: usize,
        pred_len: usize,
        common_context: usize,
        volume_features: bool,
    ) -> Result<Self> {
        ensure!(
            context > 0 && pred_len > 0 && common_context >= context,
            "invalid context or forecast length"
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
                    let contract = Dataset::contract_with_bounds(
                        &file,
                        context,
                        pred_len,
                        common_context,
                        volume_features,
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
            let eligible = ticker.contract.scaler_fit_bars >= minimum_training_bars;
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
        let mut train_refs = Vec::new();
        let mut validation_refs = Vec::new();
        let mut train_target_bars = 0;
        let mut validation_target_bars = 0;
        let mut validation_remainder_bars = 0;
        for (ticker, data) in tickers.iter().enumerate() {
            let c = &data.contract;
            train_refs.extend(
                (common_context - 1..c.scaler_fit_bars - 1)
                    .step_by(pred_len)
                    .map(|origin| WindowRef { ticker, origin }),
            );
            train_target_bars += c.scaler_fit_bars - common_context;
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
                volume_features,
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
            tickers,
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
        let auxiliary_len = if self.contract.volume_features {
            context * 2
        } else {
            0
        };
        let width = (context + pred_len) * 4 + auxiliary_len + pred_len + 11;
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
                    fill_row(row, ticker.file.bars(), &ticker.contract, origin, targets);
                })
        });
        Ok(Batch::from_packed(
            packed,
            context,
            pred_len,
            self.contract.volume_features,
            sources.iter().map(|(_, _, targets)| *targets).sum(),
        ))
    }

    pub fn batch(&self, refs: &[WindowRef], device: Device) -> Result<Batch> {
        Ok(self.host_batch(refs)?.to_device(device))
    }

    fn scales(&self, refs: &[WindowRef], device: Device) -> (Tensor, Tensor) {
        let means: Vec<f32> = refs
            .iter()
            .flat_map(|r| self.ticker(*r).contract.means.map(|v| v as f32))
            .collect();
        let stds: Vec<f32> = refs
            .iter()
            .flat_map(|r| self.ticker(*r).contract.stds.map(|v| v as f32))
            .collect();
        (
            Tensor::from_slice(&means)
                .reshape([refs.len() as i64, 1, 4])
                .to_device(device),
            Tensor::from_slice(&stds)
                .reshape([refs.len() as i64, 1, 4])
                .to_device(device),
        )
    }

    pub fn denormalize(&self, refs: &[WindowRef], prediction: &Tensor) -> Tensor {
        let (means, stds) = self.scales(refs, prediction.device());
        prediction.to_kind(Kind::Float) * stds + means
    }

    pub fn price_errors(&self, refs: &[WindowRef], errors: &Tensor) -> Tensor {
        let (_, stds) = self.scales(refs, errors.device());
        errors.to_kind(Kind::Float) * stds
    }
}

fn fill_row(
    row: &mut [f32],
    bars: &[PackedBar],
    contract: &DataContract,
    origin: usize,
    targets: usize,
) {
    let context = contract.context;
    let pred_len = contract.pred_len;
    let start = origin + 1 - context;
    let values_len = (context + pred_len) * 4;
    let aux_len = if contract.volume_features {
        context * 2
    } else {
        0
    };
    row.fill(0.0);
    let mut close_mean = 0.0f64;
    let mut close_m2 = 0.0f64;
    let mut relative_range_sum = 0.0f64;
    let mut previous_volume = start.checked_sub(1).map_or(0.0, |i| {
        bars[raw_index(i, &contract.invalid_ohlc_indices)].volume
    });
    for (position, bar) in ValidBars::new(
        bars,
        &contract.invalid_ohlc_indices,
        start,
        context + targets,
    )
    .enumerate()
    {
        for (channel, value) in [bar.open, bar.high, bar.low, bar.close]
            .into_iter()
            .enumerate()
        {
            row[position * 4 + channel] =
                ((f64::from(value) - contract.means[channel]) / contract.stds[channel]) as f32;
        }
        if position < context {
            let close = f64::from(bar.close);
            let delta = close - close_mean;
            close_mean += delta / (position + 1) as f64;
            close_m2 += delta * (close - close_mean);
            relative_range_sum += (f64::from(bar.high) - f64::from(bar.low)) / f64::from(bar.low);
        }
        if contract.volume_features && position < context {
            let current = bar.volume;
            let valid = current.is_finite()
                && current > 0.0
                && previous_volume.is_finite()
                && previous_volume > 0.0;
            if valid {
                row[values_len + position * 2] =
                    (f64::from(current).ln() - f64::from(previous_volume).ln()) as f32;
                row[values_len + position * 2 + 1] = 1.0;
            }
            previous_volume = current;
        }
    }
    row[values_len + aux_len..values_len + aux_len + targets].fill(1.0);
    let scaling_start = values_len + aux_len + pred_len;
    row[scaling_start..scaling_start + 4]
        .copy_from_slice(&contract.means.map(|value| value as f32));
    row[scaling_start + 4..scaling_start + 8]
        .copy_from_slice(&contract.stds.map(|value| value as f32));
    let close_scale = (close_m2 / context as f64 + 1e-5 * contract.stds[3].powi(2)).sqrt();
    let mean_range = relative_range_sum / context as f64;
    let relative_range = if mean_range > 0.0 {
        mean_range
    } else {
        let training_range = (contract.means[1] - contract.means[2]) / contract.means[2];
        if training_range > 0.0 {
            training_range
        } else {
            f64::from(f32::EPSILON)
        }
    };
    row[scaling_start + 8..scaling_start + 11].copy_from_slice(&[
        close_mean as f32,
        close_scale as f32,
        relative_range as f32,
    ]);
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
            scaler_fit_bars: 9,
            means: [2.0; 4],
            stds: [2.0; 4],
            common_context: 4,
            volume_features: true,
            auxiliary_schema: String::new(),
        };
        let mut row = vec![f32::NAN; 4 * 7 + 2 * 4 + 3 + 11];
        fill_row(&mut row, &bars, &contract, 6, 2);
        assert_eq!(&row[..4], &[1.0, 2.0, 0.75, 1.5]);
        assert_eq!(&row[24..28], &[0.0; 4]);
        assert_eq!(&row[36..39], &[1.0, 1.0, 0.0]);
        assert_eq!(&row[39..47], &[2.0; 8]);
        assert_eq!(row[47], 6.5);
        assert_eq!(row[48], (1.25f64 + 1e-5 * 4.0).sqrt() as f32);
        let batch = Batch::from_packed(
            Tensor::from_slice(&row).reshape([1, row.len() as i64]),
            4,
            3,
            true,
            2,
        )
        .to_device(Device::Cpu);
        assert_eq!(batch.inputs.size(), [1, 4, 4]);
        assert_eq!(batch.targets.size(), [1, 3, 4]);
        assert_eq!(batch.price_scaling.size(), [1, 2, 4]);
        assert_eq!(batch.geometry_context.size(), [1, 3]);
        assert_eq!(
            Vec::<f32>::try_from(batch.price_scaling.reshape([-1])).unwrap(),
            [2.0; 8]
        );
        assert_eq!(batch.auxiliary.unwrap().size(), [1, 4, 2]);
        assert_eq!(
            Vec::<f32>::try_from(batch.target_mask.reshape([-1])).unwrap(),
            [1.0, 1.0, 0.0]
        );
        let mut penny_contract = contract.clone();
        penny_contract.means = [1e8; 4];
        penny_contract.stds = [1e8; 4];
        let mut penny_bars = bars.clone();
        for (index, bar) in penny_bars.iter_mut().enumerate() {
            let close = 0.0001f32 + index as f32 * 0.000001;
            bar.open = close;
            bar.high = close * 1.1;
            bar.low = close * 0.9;
            bar.close = close;
        }
        let mut penny_row = vec![0.0; row.len()];
        fill_row(&mut penny_row, &penny_bars, &penny_contract, 6, 2);
        let historical = &penny_bars[3..7];
        let expected_mean = historical
            .iter()
            .map(|bar| f64::from(bar.close))
            .sum::<f64>()
            / 4.0;
        let expected_variance = historical
            .iter()
            .map(|bar| (f64::from(bar.close) - expected_mean).powi(2))
            .sum::<f64>()
            / 4.0;
        let expected_range = historical
            .iter()
            .map(|bar| (f64::from(bar.high) - f64::from(bar.low)) / f64::from(bar.low))
            .sum::<f64>()
            / 4.0;
        assert_eq!(penny_row[47], expected_mean as f32);
        assert_eq!(
            penny_row[48],
            (expected_variance + 1e-5 * 1e16).sqrt() as f32
        );
        assert_eq!(penny_row[49], expected_range as f32);
        assert!(penny_row[47] > 0.0);
        assert_eq!(
            penny_row[3] * 1e8f32 + 1e8f32,
            0.0,
            "fixture must expose normalized-price cancellation"
        );
        for bar in &mut penny_bars[7..] {
            bar.open *= 1000.0;
            bar.high *= 1000.0;
            bar.low *= 1000.0;
            bar.close *= 1000.0;
        }
        let mut changed = vec![0.0; row.len()];
        fill_row(&mut changed, &penny_bars, &penny_contract, 6, 2);
        assert_eq!(
            &penny_row[47..50],
            &changed[47..50],
            "future targets changed historical geometry context"
        );
        for bar in &mut penny_bars[..7] {
            bar.high = bar.close;
            bar.low = bar.close;
        }
        fill_row(&mut changed, &penny_bars, &penny_contract, 6, 2);
        assert_eq!(changed[49], f32::EPSILON);
        penny_contract.means[1] = 1.5e8;
        fill_row(&mut changed, &penny_bars, &penny_contract, 6, 2);
        assert_eq!(changed[49], 0.5);
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
        let corpus = Corpus::load(&directory.0, &[], 16, 7, 32, true).unwrap();
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
        assert_eq!(corpus.contract.tickers[0].scaler_fit_bars, 6999 - 100);
        assert_eq!(corpus.contract.tickers[1].scaler_fit_bars, 4994);
        assert_eq!(
            corpus.contract.tickers[1].invalid_ohlc_indices,
            [0, 1, 47, 48, 49, 4000]
        );
        assert_eq!(corpus.contract.tickers[2].scaler_fit_bars, 5000);
        assert_eq!(corpus.contract.tickers[3].scaler_fit_bars, 33);
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
        let normalized: Vec<f32> = Vec::try_from(selected.inputs.reshape([-1])).unwrap();
        for (position, &raw) in valid_raw[31..47].iter().enumerate() {
            let expected = ((f64::from(bars[raw].close) - source.contract.means[3])
                / source.contract.stds[3]) as f32;
            assert_eq!(normalized[position * 4 + 3], expected);
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
            assert!(coverage[32..metadata.scaler_fit_bars]
                .iter()
                .all(|&v| v == 1));
            assert!(coverage[metadata.scaler_fit_bars..].iter().all(|&v| v == 0));
        }
        assert_eq!(total, corpus.contract.train_target_bars);
        let refs = [corpus.train_refs[0], *corpus.train_refs.last().unwrap()];
        let batch = corpus.host_batch(&refs).unwrap();
        assert_eq!(batch.target_mask.size(), [2, 7]);
        assert_eq!(batch.price_scaling.size(), [2, 2, 4]);
        assert_eq!(batch.geometry_context.size(), [2, 3]);
        let expected_scaling = refs
            .iter()
            .flat_map(|reference| {
                let contract = &corpus.ticker(*reference).contract;
                contract
                    .means
                    .into_iter()
                    .chain(contract.stds)
                    .map(|value| value as f32)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            Vec::<f32>::try_from(batch.price_scaling.reshape([-1])).unwrap(),
            expected_scaling
        );
        assert_eq!(
            batch.valid_target_bars,
            refs.iter()
                .map(|r| corpus.ticker(*r).target_count(r.origin).unwrap())
                .sum::<usize>()
        );
    }

    #[test]
    fn corpus_can_be_shared_with_a_prefetch_worker() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<Corpus>();
    }
}
