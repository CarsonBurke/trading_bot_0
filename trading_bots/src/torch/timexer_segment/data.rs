use std::path::Path;

use anyhow::{ensure, Context, Result};
use ring::digest::{Context as Digest, SHA256};
use serde::{Deserialize, Serialize};
use shared::{
    bars::{bar_file_path, BarFile, PackedBar},
    report::CandleBar,
};
use tch::{Device, Kind, Tensor};

const RESOLUTION: u32 = 300;
const SCHEMA: &str = "timexer-segment-ohlc-v1;global-train-population-standard-scaler;next-observed-bars;split70:10:10:10;left-label-purge;completed-five-minute-bars";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DataContract {
    pub schema: String,
    pub ticker: String,
    pub fingerprint: String,
    pub source_bars: usize,
    pub valid_bars: usize,
    pub invalid_ohlc_indices: Vec<usize>,
    pub boundaries: [usize; 3],
    pub boundary_timestamps: [i64; 3],
    pub context: usize,
    pub pred_len: usize,
    pub purge: usize,
    pub scaler_fit_bars: usize,
    pub means: [f64; 4],
    pub stds: [f64; 4],
    #[serde(default)]
    pub common_context: usize,
    #[serde(default)]
    pub volume_features: bool,
    #[serde(default)]
    pub auxiliary_schema: String,
}

pub struct TrainPass {
    pub origins: Vec<usize>,
    pub target_bars: usize,
    pub remainder_bars: usize,
}

pub struct Dataset {
    pub contract: DataContract,
    pub train_origins: Vec<usize>,
    pub validation_origins: Vec<usize>,
    pub(super) timestamps: Vec<i64>,
    pub(super) candles: Vec<CandleBar>,
    pub(super) normalized: Vec<f32>,
    pub(super) auxiliary: Vec<f32>,
    prepared: Option<PreparedHistory>,
}

struct PreparedHistory {
    windows: Tensor,
    auxiliary_windows: Tensor,
    target_steps: Tensor,
    means: Tensor,
    stds: Tensor,
    device: Device,
}

impl Dataset {
    pub fn load(directory: &Path, ticker: &str, context: usize, pred_len: usize) -> Result<Self> {
        ensure!(
            !ticker.is_empty()
                && ticker
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_')),
            "ticker must be one explicit symbol without path separators"
        );
        let file = BarFile::open(&bar_file_path(directory, ticker, RESOLUTION))
            .with_context(|| format!("loading only ticker {ticker}"))?;
        ensure!(
            file.symbol() == ticker && file.res_secs() == RESOLUTION,
            "bar header does not match configured ticker and five-minute resolution"
        );
        Self::from_bars(ticker, file.bars(), context, pred_len)
    }

    pub fn load_with_bounds(
        directory: &Path,
        ticker: &str,
        context: usize,
        pred_len: usize,
        common_context: usize,
        volume_features: bool,
        bounds: [i64; 3],
    ) -> Result<Self> {
        ensure!(
            !ticker.is_empty()
                && ticker
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_')),
            "invalid ticker"
        );
        ensure!(
            bounds.windows(2).all(|pair| pair[0] < pair[1]),
            "split timestamps must be strictly increasing"
        );
        let file = BarFile::open(&bar_file_path(directory, ticker, RESOLUTION))
            .with_context(|| format!("loading only ticker {ticker}"))?;
        ensure!(
            file.symbol() == ticker && file.res_secs() == RESOLUTION,
            "bar header does not match requested ticker and resolution"
        );
        let boundaries =
            bounds.map(|timestamp| file.bars().partition_point(|bar| bar.ts() < timestamp));
        let mut dataset =
            Self::from_bars_at_boundaries(ticker, file.bars(), context, pred_len, boundaries)?;
        dataset.contract.boundary_timestamps = bounds;
        dataset.contract.schema =
            SCHEMA.replace("split70:10:10:10", "shared-universe-utc-split-boundaries");
        dataset.restrict_context_origins(common_context)?;
        dataset.set_volume_features(volume_features);
        Ok(dataset)
    }

    fn from_bars(
        ticker: &str,
        bars: &[PackedBar],
        context: usize,
        pred_len: usize,
    ) -> Result<Self> {
        let n = bars.len();
        Self::from_bars_at_boundaries(
            ticker,
            bars,
            context,
            pred_len,
            [n * 7 / 10, n * 8 / 10, n * 9 / 10],
        )
    }

    fn from_bars_at_boundaries(
        ticker: &str,
        bars: &[PackedBar],
        context: usize,
        pred_len: usize,
        boundaries: [usize; 3],
    ) -> Result<Self> {
        Self::from_bars_with_materialization(ticker, bars, context, pred_len, boundaries, true)
    }

    pub fn contract_with_bounds(
        file: &BarFile,
        context: usize,
        pred_len: usize,
        common_context: usize,
        volume_features: bool,
        bounds: [i64; 3],
    ) -> Result<DataContract> {
        ensure!(
            file.res_secs() == RESOLUTION,
            "bar resolution must be five minutes"
        );
        filtered_contract(
            file.symbol(),
            file.bars(),
            context,
            pred_len,
            common_context,
            volume_features,
            bounds,
        )
    }

    fn from_bars_with_materialization(
        ticker: &str,
        bars: &[PackedBar],
        context: usize,
        pred_len: usize,
        boundaries: [usize; 3],
        materialize: bool,
    ) -> Result<Self> {
        ensure!(
            context > 0 && pred_len > 0,
            "context and prediction length must be positive"
        );
        let n = bars.len();
        let purge = pred_len.max(100);
        ensure!(
            boundaries.windows(2).all(|pair| pair[0] <= pair[1]) && boundaries[2] <= n,
            "partition indices must be ordered within the corpus"
        );
        if materialize {
            ensure!(
                boundaries.windows(2).all(|pair| pair[0] < pair[1]) && boundaries[2] < n,
                "ticker must span every global partition"
            );
        }
        let edges = [0, boundaries[0], boundaries[1], boundaries[2], n];
        let fit_end = if materialize {
            boundaries[0].saturating_sub(purge)
        } else {
            retained_partition_end(boundaries[0], n, purge)
        };
        ensure!(
            context < fit_end,
            "context leaves no purged training targets"
        );
        for partition in 0..if materialize { 4 } else { 0 } {
            let end = edges[partition + 1].saturating_sub(if partition < 3 { purge } else { 0 });
            ensure!(
                !partition_origins(edges[partition], end, context, pred_len).is_empty(),
                "partition {partition} has no complete context and target after purging"
            );
        }
        let now = chrono::Utc::now().timestamp_millis();
        ensure!(
            bars.last()
                .unwrap()
                .ts()
                .checked_add(i64::from(RESOLUTION) * 1000)
                .is_some_and(|completion| completion <= now),
            "corpus contains an uncompleted bar"
        );

        let mut digest = Digest::new(&SHA256);
        digest.update(SCHEMA.as_bytes());
        digest.update(&(ticker.len() as u64).to_le_bytes());
        digest.update(ticker.as_bytes());
        digest.update(&(n as u64).to_le_bytes());
        let mut means = [0.0_f64; 4];
        let mut m2 = [0.0_f64; 4];
        let mut candles = Vec::with_capacity(if materialize { n } else { 0 });
        let mut timestamps = Vec::with_capacity(if materialize { n } else { 0 });
        for (i, bar) in bars.iter().enumerate() {
            ensure!(
                i == 0 || bars[i - 1].ts() < bar.ts(),
                "timestamps must be strictly increasing at bar {i}"
            );
            ensure!(
                bar.ts().rem_euclid(i64::from(RESOLUTION) * 1000) == 0,
                "five-minute timestamp is off-grid at bar {i}"
            );
            let values = [bar.open, bar.high, bar.low, bar.close];
            ensure!(
                values.iter().all(|v| v.is_finite() && *v > 0.0),
                "invalid OHLC at bar {i}"
            );
            ensure!(
                bar.high >= bar.open.max(bar.close) && bar.low <= bar.open.min(bar.close),
                "invalid OHLC geometry at bar {i}"
            );

            if i < fit_end {
                for channel in 0..4 {
                    let value = f64::from(values[channel]);
                    let delta = value - means[channel];
                    means[channel] += delta / (i + 1) as f64;
                    m2[channel] += delta * (value - means[channel]);
                }
            }
            if materialize {
                timestamps.push(bar.ts());
                candles.push(CandleBar {
                    open: bar.open,
                    high: bar.high,
                    low: bar.low,
                    close: bar.close,
                });
            }
        }
        hash_bars(&mut digest, bars);
        let stds = m2.map(|sum| {
            let std = (sum / fit_end as f64).sqrt();
            if std == 0.0 {
                1.0
            } else {
                std
            }
        });
        let mut auxiliary = Vec::with_capacity(if materialize { n * 2 } else { 0 });
        if materialize {
            for (index, bar) in bars.iter().enumerate() {
                let previous = index.checked_sub(1).map(|i| bars[i].volume);
                let valid = bar.volume.is_finite()
                    && bar.volume > 0.0
                    && previous.is_some_and(|volume| volume.is_finite() && volume > 0.0);
                let innovation = if valid {
                    (f64::from(bar.volume).ln() - f64::from(previous.unwrap()).ln()) as f32
                } else {
                    0.0
                };
                auxiliary.extend([innovation, f32::from(valid)]);
            }
        }
        let mut normalized = Vec::with_capacity(if materialize { n * 4 } else { 0 });
        if materialize {
            for bar in bars {
                for (channel, value) in [bar.open, bar.high, bar.low, bar.close]
                    .into_iter()
                    .enumerate()
                {
                    let standardized = ((f64::from(value) - means[channel]) / stds[channel]) as f32;
                    ensure!(standardized.is_finite(), "non-finite standardized OHLC");
                    normalized.push(standardized);
                }
            }
        }
        let fingerprint = digest
            .finish()
            .as_ref()
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect();
        Ok(Self {
            contract: DataContract {
                schema: SCHEMA.into(),
                ticker: ticker.into(),
                fingerprint,
                source_bars: n,
                valid_bars: n,
                invalid_ohlc_indices: Vec::new(),
                boundaries,
                boundary_timestamps: boundaries.map(|i| bars.get(i).map_or(0, PackedBar::ts)),
                context,
                pred_len,
                purge,
                scaler_fit_bars: fit_end,
                means,
                stds,
                common_context: context,
                volume_features: false,
                auxiliary_schema: String::new(),
            },
            train_origins: if materialize {
                partition_origins(0, fit_end, context, pred_len).collect()
            } else {
                Vec::new()
            },
            validation_origins: if materialize {
                partition_origins(boundaries[1], boundaries[2] - purge, context, pred_len).collect()
            } else {
                Vec::new()
            },
            timestamps,
            candles,
            normalized,
            auxiliary,
            prepared: None,
        })
    }

    pub fn set_volume_features(&mut self, enabled: bool) {
        self.contract.volume_features = enabled;
        self.contract.auxiliary_schema = auxiliary_schema(enabled);
    }

    pub fn restrict_context_origins(&mut self, common_context: usize) -> Result<()> {
        ensure!(
            common_context >= self.contract.context,
            "common context must cover model context"
        );
        ensure!(
            common_context < self.contract.scaler_fit_bars,
            "common context leaves no training targets"
        );
        self.contract.common_context = common_context;
        self.train_origins = partition_origins(
            0,
            self.contract.scaler_fit_bars,
            common_context,
            self.contract.pred_len,
        )
        .collect();
        self.validation_origins = partition_origins(
            self.contract.boundaries[1],
            self.contract.boundaries[2] - self.contract.purge,
            common_context,
            self.contract.pred_len,
        )
        .collect();
        ensure!(
            !self.train_origins.is_empty() && !self.validation_origins.is_empty(),
            "common context leaves no complete evaluation origins"
        );
        Ok(())
    }

    pub fn train_pass(&self, common_context: usize) -> Result<TrainPass> {
        ensure!(
            common_context >= self.contract.context
                && common_context >= self.contract.common_context,
            "training pass context is shorter than the authenticated context"
        );
        ensure!(
            common_context < self.contract.scaler_fit_bars,
            "context leaves no training targets"
        );
        let target_bars = self.contract.scaler_fit_bars - common_context;
        Ok(TrainPass {
            origins: (common_context - 1..self.contract.scaler_fit_bars - 1)
                .step_by(self.contract.pred_len)
                .collect(),
            target_bars,
            remainder_bars: target_bars % self.contract.pred_len,
        })
    }

    pub(super) fn unlocked_origin(&self, origin: usize, partial_train: bool) -> bool {
        let minimum = self.contract.common_context.max(self.contract.context) - 1;
        (origin >= minimum
            && origin < self.contract.scaler_fit_bars - 1
            && (partial_train || origin + self.contract.pred_len < self.contract.scaler_fit_bars))
            || self.validation_origins.binary_search(&origin).is_ok()
    }

    pub fn prepare(&mut self, device: Device) {
        if self
            .prepared
            .as_ref()
            .is_some_and(|cache| cache.device == device)
        {
            return;
        }
        let length = (self.contract.context + self.contract.pred_len) as i64;
        // Unfold is a view into the compact history; only selected batches allocate windows.
        let windows = Tensor::from_slice(&self.normalized)
            .reshape([self.contract.source_bars as i64, 4])
            .to_device(device)
            .unfold(0, length, 1)
            .permute([0, 2, 1]);
        let means =
            Tensor::from_slice(&self.contract.means.map(|value| value as f32)).to_device(device);
        let stds =
            Tensor::from_slice(&self.contract.stds.map(|value| value as f32)).to_device(device);
        let auxiliary_windows = Tensor::from_slice(&self.auxiliary)
            .reshape([self.contract.source_bars as i64, 2])
            .to_device(device)
            .unfold(0, self.contract.context as i64, 1)
            .permute([0, 2, 1]);
        self.prepared = Some(PreparedHistory {
            windows,
            auxiliary_windows,
            target_steps: Tensor::arange(self.contract.pred_len as i64, (Kind::Int64, device))
                .unsqueeze(0),
            means,
            stds,
            device,
        });
    }

    pub fn batch(&self, origins: &[usize], device: Device) -> Result<(Tensor, Tensor)> {
        let selected = self.select_windows(origins, device, false)?;
        Ok((
            selected.narrow(1, 0, self.contract.context as i64),
            selected.narrow(
                1,
                self.contract.context as i64,
                self.contract.pred_len as i64,
            ),
        ))
    }

    fn select_windows(
        &self,
        origins: &[usize],
        device: Device,
        partial_train: bool,
    ) -> Result<Tensor> {
        ensure!(!origins.is_empty(), "cannot construct an empty batch");
        let cache = self
            .prepared
            .as_ref()
            .filter(|cache| cache.device == device)
            .context("prepare the dataset on the requested device before batching")?;
        let mut starts = Vec::with_capacity(origins.len());
        for &origin in origins {
            ensure!(
                self.unlocked_origin(origin, partial_train),
                "origin {origin} is outside unlocked purged training/validation targets"
            );
            starts.push((origin + 1 - self.contract.context) as i64);
        }
        Ok(cache
            .windows
            .index_select(0, &Tensor::from_slice(&starts).to_device(device)))
    }

    pub fn batch_with_mask(
        &self,
        origins: &[usize],
        device: Device,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let selected = self.select_windows(origins, device, true)?;
        let cache = self.prepared.as_ref().unwrap();
        let lengths: Vec<i64> = origins
            .iter()
            .map(|&origin| {
                if origin < self.contract.scaler_fit_bars {
                    (self.contract.scaler_fit_bars - origin - 1).min(self.contract.pred_len) as i64
                } else {
                    self.contract.pred_len as i64
                }
            })
            .collect();
        let mask = cache
            .target_steps
            .lt_tensor(&Tensor::from_slice(&lengths).to_device(device).unsqueeze(1));
        let targets = selected
            .narrow(
                1,
                self.contract.context as i64,
                self.contract.pred_len as i64,
            )
            .masked_fill(&mask.logical_not().unsqueeze(2), 0.0);
        Ok((
            selected.narrow(1, 0, self.contract.context as i64),
            targets,
            mask.to_kind(Kind::Float),
        ))
    }

    pub fn auxiliary_batch(&self, origins: &[usize], device: Device) -> Result<Tensor> {
        ensure!(!origins.is_empty(), "cannot construct an empty batch");
        let cache = self
            .prepared
            .as_ref()
            .filter(|cache| cache.device == device)
            .context("prepare the dataset on the requested device before batching")?;
        let starts: Vec<i64> = origins
            .iter()
            .map(|&origin| {
                ensure!(
                    self.unlocked_origin(origin, true),
                    "auxiliary origin {origin} is outside unlocked targets"
                );
                Ok((origin + 1 - self.contract.context) as i64)
            })
            .collect::<Result<_>>()?;
        Ok(cache
            .auxiliary_windows
            .index_select(0, &Tensor::from_slice(&starts).to_device(device)))
    }

    pub fn candle_window(
        &self,
        origin: usize,
        history: usize,
        future: usize,
    ) -> Result<&[CandleBar]> {
        let start = origin
            .checked_sub(history)
            .context("candle history precedes corpus")?;
        let end = origin
            .checked_add(future)
            .and_then(|i| i.checked_add(1))
            .context("candle window overflow")?;
        self.candles
            .get(start..end)
            .context("candle window exceeds corpus")
    }

    pub fn timestamp(&self, origin: usize) -> i64 {
        self.timestamps[origin]
    }

    pub fn denormalize(&self, prediction: &Tensor) -> Tensor {
        let cache = self
            .prepared
            .as_ref()
            .filter(|cache| cache.device == prediction.device())
            .expect("prepare the dataset on the prediction device before denormalizing");
        prediction.to_kind(tch::Kind::Float) * &cache.stds + &cache.means
    }

    pub fn price_errors(&self, normalized_errors: &Tensor) -> Tensor {
        let cache = self
            .prepared
            .as_ref()
            .filter(|cache| cache.device == normalized_errors.device())
            .expect("prepare the dataset on the prediction device before scoring price errors");
        normalized_errors.to_kind(tch::Kind::Float) * &cache.stds
    }
}

fn auxiliary_schema(enabled: bool) -> String {
    if enabled {
        "same-ticker-log-volume-difference;valid-iff-current-and-previous-positive-finite;invalid-zero;validity-indicator;window-valid-only-population-normalization-v1".into()
    } else {
        String::new()
    }
}

pub(super) fn valid_ohlc(bar: &PackedBar) -> bool {
    [bar.open, bar.high, bar.low, bar.close]
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
        && bar.high >= bar.open.max(bar.close)
        && bar.low <= bar.open.min(bar.close)
}

fn filtered_contract(
    ticker: &str,
    bars: &[PackedBar],
    context: usize,
    pred_len: usize,
    common_context: usize,
    volume_features: bool,
    bounds: [i64; 3],
) -> Result<DataContract> {
    ensure!(
        context > 0 && pred_len > 0 && common_context >= context,
        "invalid history or prediction length"
    );
    ensure!(
        bounds.windows(2).all(|pair| pair[0] < pair[1]),
        "split timestamps must be strictly increasing"
    );
    let resolution_ms = i64::from(RESOLUTION) * 1000;
    let now = chrono::Utc::now().timestamp_millis();
    let mut invalid_ohlc_indices = Vec::new();
    for (index, bar) in bars.iter().enumerate() {
        ensure!(
            index == 0 || bars[index - 1].ts() < bar.ts(),
            "{ticker}: timestamps must be strictly increasing at raw bar {index}"
        );
        ensure!(
            bar.ts().rem_euclid(resolution_ms) == 0,
            "{ticker}: off-grid five-minute timestamp at raw bar {index}"
        );
        if !valid_ohlc(bar) {
            invalid_ohlc_indices.push(index);
        }
    }
    ensure!(
        bars.last().is_none_or(|bar| bar
            .ts()
            .checked_add(resolution_ms)
            .is_some_and(|completion| completion <= now)),
        "{ticker}: corpus contains an uncompleted bar"
    );
    let source_bars = bars.len();
    let valid_bars = source_bars - invalid_ohlc_indices.len();
    let raw_boundaries = bounds.map(|timestamp| bars.partition_point(|bar| bar.ts() < timestamp));
    let boundaries = raw_boundaries
        .map(|boundary| boundary - invalid_ohlc_indices.partition_point(|&index| index < boundary));
    let purge = pred_len.max(100);
    let scaler_fit_bars = retained_partition_end(boundaries[0], valid_bars, purge);
    let mut means = [0.0_f64; 4];
    let mut m2 = [0.0_f64; 4];
    let mut count = 0usize;
    let mut invalid = invalid_ohlc_indices.iter().copied().peekable();
    for (index, bar) in bars.iter().enumerate() {
        if count == scaler_fit_bars {
            break;
        }
        if invalid.peek() == Some(&index) {
            invalid.next();
            continue;
        }
        count += 1;
        for (channel, value) in [bar.open, bar.high, bar.low, bar.close]
            .into_iter()
            .enumerate()
        {
            let value = f64::from(value);
            let delta = value - means[channel];
            means[channel] += delta / count as f64;
            m2[channel] += delta * (value - means[channel]);
        }
    }
    let stds = m2.map(|sum| {
        if scaler_fit_bars == 0 || sum == 0.0 {
            1.0
        } else {
            (sum / scaler_fit_bars as f64).sqrt()
        }
    });
    let mut digest = Digest::new(&SHA256);
    digest.update(SCHEMA.as_bytes());
    digest.update(&(ticker.len() as u64).to_le_bytes());
    digest.update(ticker.as_bytes());
    digest.update(&(source_bars as u64).to_le_bytes());
    hash_bars(&mut digest, bars);
    let fingerprint = digest
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok(DataContract {
        schema: format!("{};purge-only-at-observed-partition-boundaries;strict-raw-timestamps;skip-only-nonfinite-nonpositive-or-inconsistent-source-OHLC-v1;logical-valid-bar-ordinals", SCHEMA.replace("split70:10:10:10", "shared-universe-utc-split-boundaries").replace("next-observed-bars", "next-valid-observed-bars")),
        ticker: ticker.into(), fingerprint, source_bars, valid_bars, invalid_ohlc_indices,
        boundaries, boundary_timestamps: bounds, context, pred_len, purge, scaler_fit_bars,
        means, stds, common_context, volume_features, auxiliary_schema: auxiliary_schema(volume_features),
    })
}

pub(super) fn retained_partition_end(boundary: usize, source_bars: usize, purge: usize) -> usize {
    if boundary == source_bars {
        source_bars
    } else {
        boundary.saturating_sub(purge)
    }
}

fn hash_bars(digest: &mut Digest, bars: &[PackedBar]) {
    #[cfg(target_endian = "little")]
    digest.update(bytemuck::cast_slice(bars));
    #[cfg(target_endian = "big")]
    for bar in bars {
        digest.update(&bar.ts().to_le_bytes());
        for value in [bar.open, bar.high, bar.low, bar.close, bar.volume, bar.vwap] {
            digest.update(&value.to_bits().to_le_bytes());
        }
        digest.update(&bar.trades.to_le_bytes());
    }
}

fn partition_origins(
    start: usize,
    end: usize,
    context: usize,
    pred_len: usize,
) -> std::ops::Range<usize> {
    // The first label must lie in this partition; the origin is its preceding observed bar.
    let first = start.saturating_sub(1).max(context - 1);
    let Some(exclusive_end) = end.checked_sub(pred_len) else {
        return first..first;
    };
    first..exclusive_end
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bars() -> Vec<PackedBar> {
        (0..5000)
            .map(|i| {
                let close = 100.0 + i as f32 * 0.01;
                PackedBar {
                    ts_ms: 1_500_000_000_000 + i * 300_000,
                    open: close - 0.25,
                    high: close + 0.5,
                    low: close - 0.5,
                    close,
                    volume: 100.0,
                    vwap: close,
                    trades: 1,
                }
            })
            .collect()
    }

    #[test]
    fn complete_origins_and_labels_obey_partition_and_purge() {
        let dataset = Dataset::from_bars("ONE", &bars(), 96, 96).unwrap();
        assert_eq!(dataset.train_origins, (95..3304).collect::<Vec<_>>());
        assert_eq!(dataset.validation_origins, (3999..4304).collect::<Vec<_>>());
        for origins in [&dataset.train_origins, &dataset.validation_origins] {
            assert!(origins.windows(2).all(|p| p[1] == p[0] + 1));
        }
        let long = Dataset::from_bars("ONE", &bars(), 96, 192).unwrap();
        assert_eq!(long.contract.purge, 192);
        assert_eq!(long.train_origins.last().unwrap() + 192, 3500 - 192 - 1);
        assert_eq!(long.validation_origins[0] + 1, 4000);
    }

    #[test]
    fn normalization_uses_only_retained_training_bars() {
        let original = bars();
        let dataset = Dataset::from_bars("ONE", &original, 96, 96).unwrap();
        let mut changed = original.clone();
        for bar in &mut changed[dataset.contract.scaler_fit_bars..] {
            bar.open *= 3.0;
            bar.high *= 3.0;
            bar.low *= 3.0;
            bar.close *= 3.0;
        }
        let changed = Dataset::from_bars("ONE", &changed, 96, 96).unwrap();
        assert_eq!(dataset.contract.means, changed.contract.means);
        assert_eq!(dataset.contract.stds, changed.contract.stds);
        assert_ne!(dataset.contract.fingerprint, changed.contract.fingerprint);
        let fit = dataset.contract.scaler_fit_bars;
        for channel in 0..4 {
            let mean = (0..fit)
                .map(|i| f64::from(dataset.normalized[i * 4 + channel]))
                .sum::<f64>()
                / fit as f64;
            let variance = (0..fit)
                .map(|i| f64::from(dataset.normalized[i * 4 + channel]).powi(2))
                .sum::<f64>()
                / fit as f64;
            assert!(mean.abs() < 1e-7);
            assert!((variance - 1.0).abs() < 1e-7);
        }
    }

    #[test]
    fn gathered_windows_match_exact_ohlc_and_keep_terminal_targets_locked() {
        let source = bars();
        let mut dataset = Dataset::from_bars("ONE", &source, 96, 96).unwrap();
        dataset.prepare(Device::Cpu);
        let origins = [dataset.validation_origins[7], dataset.train_origins[17]];
        let (inputs, targets) = dataset.batch(&origins, Device::Cpu).unwrap();
        assert_eq!(inputs.size(), [2, 96, 4]);
        assert_eq!(targets.size(), [2, 96, 4]);
        let inputs: Vec<f64> = Vec::try_from(dataset.denormalize(&inputs).reshape([-1])).unwrap();
        let targets: Vec<f64> = Vec::try_from(dataset.denormalize(&targets).reshape([-1])).unwrap();
        for (row, origin) in origins.into_iter().enumerate() {
            for step in 0..96 {
                assert!(
                    (inputs[row * 96 * 4 + step * 4 + 3]
                        - f64::from(source[origin + 1 - 96 + step].close))
                    .abs()
                        < 1e-5
                );
                assert!(
                    (targets[row * 96 * 4 + step * 4 + 3]
                        - f64::from(source[origin + 1 + step].close))
                    .abs()
                        < 1e-5
                );
            }
            let candles = dataset.candle_window(origin, 32, 96).unwrap();
            assert_eq!(candles.len(), 129);
            assert_eq!(candles[32].close, source[origin].close);
        }
        assert!(dataset.batch(&[3500], Device::Cpu).is_err());
        assert!(dataset.batch(&[4500], Device::Cpu).is_err());
        assert!(dataset.batch(&[3304], Device::Cpu).is_err());
    }

    #[test]
    fn tiled_pass_covers_each_retained_target_exactly_once_including_tail() {
        let mut dataset = Dataset::from_bars("ONE", &bars(), 96, 192).unwrap();
        dataset.restrict_context_origins(128).unwrap();
        let pass = dataset.train_pass(128).unwrap();
        let mut counts = vec![0u8; dataset.contract.scaler_fit_bars];
        for &origin in &pass.origins {
            for target in origin + 1..(origin + 1 + 192).min(counts.len()) {
                counts[target] += 1;
            }
        }
        assert!(counts[..128].iter().all(|&count| count == 0));
        assert!(counts[128..].iter().all(|&count| count == 1));
        assert_eq!(pass.target_bars, counts.len() - 128);
        assert_eq!(pass.remainder_bars, pass.target_bars % 192);
        assert_ne!(pass.remainder_bars, 0);
        dataset.prepare(Device::Cpu);
        let last = *pass.origins.last().unwrap();
        let (inputs, targets, mask) = dataset.batch_with_mask(&[last], Device::Cpu).unwrap();
        assert_eq!(inputs.size(), [1, 96, 4]);
        assert_eq!(
            mask.sum(Kind::Float).double_value(&[]),
            pass.remainder_bars as f64
        );
        assert_eq!(
            targets
                .narrow(
                    1,
                    pass.remainder_bars as i64,
                    (192 - pass.remainder_bars) as i64
                )
                .abs()
                .sum(Kind::Float)
                .double_value(&[]),
            0.0
        );
        assert!(dataset.batch(&[last], Device::Cpu).is_err());
        assert!(dataset
            .batch_with_mask(&[dataset.contract.scaler_fit_bars - 1], Device::Cpu)
            .is_err());
    }

    #[test]
    fn volume_innovation_is_causal_and_explicitly_marks_missing_values() {
        let mut source = bars();
        source[42].volume = 0.0;
        source[45].volume = 200.0;
        let mut dataset = Dataset::from_bars("ONE", &source, 96, 192).unwrap();
        assert_eq!(&dataset.auxiliary[..2], &[0.0, 0.0]);
        assert_eq!(&dataset.auxiliary[84..88], &[0.0, 0.0, 0.0, 0.0]);
        assert!((dataset.auxiliary[90] - 2.0f32.ln()).abs() < 1e-6);
        let origin = 199;
        for bar in &mut source[origin + 1..] {
            bar.volume *= 3.0;
        }
        let changed = Dataset::from_bars("ONE", &source, 96, 192).unwrap();
        assert_eq!(
            &dataset.auxiliary[..(origin + 1) * 2],
            &changed.auxiliary[..(origin + 1) * 2]
        );
        dataset.prepare(Device::Cpu);
        let batch: Vec<f32> = Vec::try_from(
            dataset
                .auxiliary_batch(&[origin], Device::Cpu)
                .unwrap()
                .reshape([-1]),
        )
        .unwrap();
        assert_eq!(
            batch,
            dataset.auxiliary[(origin + 1 - 96) * 2..(origin + 1) * 2]
        );
        dataset.set_volume_features(true);
        assert!(dataset.contract.volume_features);
        assert!(!dataset.contract.auxiliary_schema.is_empty());
    }

    #[test]
    fn global_timestamp_boundaries_preserve_purges_and_train_only_scaling() {
        let source = bars();
        let dataset =
            Dataset::from_bars_at_boundaries("ONE", &source, 96, 192, [3000, 3750, 4500]).unwrap();
        assert_eq!(dataset.contract.scaler_fit_bars, 2808);
        assert_eq!(dataset.validation_origins[0], 3749);
        assert_eq!(dataset.validation_origins.last().unwrap() + 192, 4307);
    }

    #[test]
    fn metadata_keeps_delisted_tickers_without_future_partitions() {
        let source = bars();
        let dataset =
            Dataset::from_bars_with_materialization("DELISTED", &source, 96, 192, [5000; 3], false)
                .unwrap();
        assert_eq!(dataset.contract.scaler_fit_bars, 5000);
        assert_eq!(dataset.contract.boundaries, [5000; 3]);
        assert!(dataset.normalized.is_empty() && dataset.auxiliary.is_empty());
        assert!(dataset.candles.is_empty() && dataset.timestamps.is_empty());
        assert!(dataset.train_origins.is_empty() && dataset.validation_origins.is_empty());
        assert_eq!(dataset.train_pass(96).unwrap().target_bars, 4904);
        assert!(Dataset::from_bars_at_boundaries("DELISTED", &source, 96, 192, [5000; 3]).is_err());
    }

    #[test]
    fn metadata_accepts_a_single_partial_training_segment() {
        let source = bars();
        let dataset = Dataset::from_bars_with_materialization(
            "SHORT",
            &source,
            96,
            192,
            [289, 289, 289],
            false,
        )
        .unwrap();
        assert_eq!(dataset.contract.scaler_fit_bars, 97);
        let pass = dataset.train_pass(96).unwrap();
        assert_eq!(pass.origins, [95]);
        assert_eq!(pass.target_bars, 1);
        assert_eq!(pass.remainder_bars, 1);
    }

    #[test]
    fn bulk_fingerprint_matches_canonical_field_encoding() {
        let source = bars();
        let mut expected = Digest::new(&SHA256);
        for bar in &source {
            expected.update(&bar.ts().to_le_bytes());
            for value in [bar.open, bar.high, bar.low, bar.close, bar.volume, bar.vwap] {
                expected.update(&value.to_bits().to_le_bytes());
            }
            expected.update(&bar.trades.to_le_bytes());
        }
        let mut actual = Digest::new(&SHA256);
        hash_bars(&mut actual, &source);
        assert_eq!(actual.finish().as_ref(), expected.finish().as_ref());
    }

    #[test]
    fn filtered_metadata_fits_only_valid_training_rows_and_preserves_raw_authentication() {
        let mut source = bars();
        source[42].high = source[42].low;
        source[3000].open = f32::NAN;
        let bounds = [source[3500].ts(), source[4000].ts(), source[4500].ts()];
        let contract = filtered_contract("ONE", &source, 96, 192, 128, true, bounds).unwrap();
        assert_eq!(contract.source_bars, 5000);
        assert_eq!(contract.valid_bars, 4998);
        assert_eq!(contract.invalid_ohlc_indices, [42, 3000]);
        assert_eq!(contract.boundaries, [3498, 3998, 4498]);
        assert_eq!(contract.boundary_timestamps, bounds);
        assert_eq!(contract.scaler_fit_bars, 3306);
        let valid: Vec<_> = source
            .iter()
            .filter(|bar| valid_ohlc(bar))
            .take(3306)
            .collect();
        let expected = valid.iter().map(|bar| f64::from(bar.close)).sum::<f64>() / 3306.;
        assert!((contract.means[3] - expected).abs() < 1e-10);
        source[42].volume *= 2.0;
        for bar in &mut source[3500..] {
            bar.open *= 2.0;
            bar.high *= 2.0;
            bar.low *= 2.0;
            bar.close *= 2.0;
        }
        let changed = filtered_contract("ONE", &source, 96, 192, 128, true, bounds).unwrap();
        assert_eq!(contract.means, changed.means);
        assert_eq!(contract.stds, changed.stds);
        assert_eq!(contract.invalid_ohlc_indices, changed.invalid_ohlc_indices);
        assert_ne!(contract.fingerprint, changed.fingerprint);
        source[42].ts_ms = source[41].ts();
        assert!(filtered_contract("ONE", &source, 96, 192, 128, true, bounds).is_err());
    }

    #[test]
    fn raw_source_fingerprint_is_identical_across_context_comparisons() {
        let source = bars();
        let bounds = [source[3500].ts(), source[4000].ts(), source[4500].ts()];
        let short = filtered_contract("ONE", &source, 96, 192, 2048, false, bounds).unwrap();
        let long = filtered_contract("ONE", &source, 2048, 192, 2048, true, bounds).unwrap();
        assert_eq!(short.fingerprint, long.fingerprint);
        assert_eq!(short.means, long.means);
        assert_eq!(short.stds, long.stds);
        assert_eq!(short.boundaries, long.boundaries);
        assert_eq!(
            short.fingerprint,
            Dataset::from_bars("ONE", &source, 96, 192)
                .unwrap()
                .contract
                .fingerprint
        );
    }

    #[test]
    fn all_invalid_training_metadata_is_finite_but_ineligible() {
        let mut source = bars();
        let bounds = [source[3500].ts(), source[4000].ts(), source[4500].ts()];
        for bar in &mut source[..3500] {
            bar.close = 0.0;
        }
        let contract = filtered_contract("ONE", &source, 96, 192, 128, false, bounds).unwrap();
        assert_eq!(contract.scaler_fit_bars, 0);
        assert_eq!(contract.means, [0.0; 4]);
        assert_eq!(contract.stds, [1.0; 4]);
        assert!(contract.scaler_fit_bars <= contract.common_context);
    }

    #[test]
    fn rejects_duplicate_timestamps_and_invalid_prices() {
        let mut source = bars();
        source[42].ts_ms = source[41].ts();
        assert!(Dataset::from_bars("ONE", &source, 96, 96).is_err());
        source = bars();
        source[42].close = f32::NAN;
        assert!(Dataset::from_bars("ONE", &source, 96, 96).is_err());
    }
}
