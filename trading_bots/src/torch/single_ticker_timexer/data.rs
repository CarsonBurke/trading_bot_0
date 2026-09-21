use std::{collections::VecDeque, path::Path};

use anyhow::{ensure, Context, Result};
use clap::ValueEnum;
use ring::digest::{Context as Digest, SHA256};
use serde::{Deserialize, Serialize};
use shared::bars::{bar_file_path, BarFile, PackedBar};
use shared::report::CandleBar;
use tch::{Device, Kind, Tensor};

use crate::torch::dataset::{bar_time_ids, forecast_schedule_after};

use super::support::Supports;

pub const CONTEXT: usize = 2048;
pub const HORIZONS: [usize; 6] = [1, 4, 16, 39, 78, 100];
pub const CLOCK_DIM: usize = 7;
pub const PURGE: usize = 100;
pub const FEATURES: [&str; 12] = [
    "standardized_log_range",
    "close_position",
    "open_position",
    "log_volume_innovation",
    "log_causal_volatility",
    "minute_sin",
    "minute_cos",
    "weekday_sin",
    "weekday_cos",
    "session_class",
    "elapsed_bars_bucket",
    "day_edge",
];
pub const DATA_CONTRACT: &str = "single-ticker-v1;300s-open-timestamps-completed;context2048;ewma-second-moment-prior-half-life78-warmup256;log-volume-prior-ewma-half-life78;12-features-validity;extended-US-equity-04:00-20:00-ET-fixed-grid-existing-exchange-holidays-v1;future-clock-nominal-schedule-targets-next-observed-bars;split-bars70:10:10:10;purge100-left-origin-bars;train-label-interval-uniqueness-global-mean1;train-only128-quantile-cuts-interpolated16-quantiles-exponential-tails";
const WARMUP: usize = 256;
const RESOLUTION: u32 = 300;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Calibration,
    Validation,
    Test,
}

impl Split {
    fn index(self) -> usize {
        match self {
            Self::Train => 0,
            Self::Calibration => 1,
            Self::Validation => 2,
            Self::Test => 3,
        }
    }
}

pub struct Batch {
    pub endogenous: Tensor,
    pub exogenous: Tensor,
    pub validity: Tensor,
    pub future_clock: Tensor,
    pub targets: Tensor,
    pub bins: Tensor,
    pub weights: Tensor,
}

pub struct Dataset {
    pub ticker: String,
    pub fingerprint: String,
    pub source_bars: usize,
    pub split_bounds: [i64; 3],
    pub supports: Supports,
    timestamps: Vec<i64>,
    candles: Vec<CandleBar>,
    closes: Vec<f64>,
    returns: Vec<f64>,
    sigmas: Vec<f64>,
    endogenous: Vec<f32>,
    exogenous: Vec<[f32; 12]>,
    validity: Vec<[f32; 12]>,
    split_origins: [Vec<usize>; 4],
    weights: Vec<f64>,
    valid_context: Vec<bool>,
    prepared: Option<PreparedHistory>,
}

struct PreparedHistory {
    windows: Tensor,
    future_clock: Tensor,
    device: Device,
}

impl Dataset {
    pub fn load(directory: &Path, ticker: &str) -> Result<Self> {
        ensure!(
            !ticker.is_empty()
                && ticker
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_')),
            "ticker must be one explicit symbol without path separators"
        );
        let path = bar_file_path(directory, ticker, RESOLUTION);
        let file = BarFile::open(&path).with_context(|| format!("loading only ticker {ticker}"))?;
        ensure!(
            file.symbol() == ticker && file.res_secs() == RESOLUTION,
            "bar header does not match configured ticker and five-minute resolution"
        );
        Self::from_bars(ticker, file.bars())
    }

    fn from_bars(ticker: &str, bars: &[PackedBar]) -> Result<Self> {
        Self::from_bars_with_frozen_partition(ticker, bars, bars.len())
    }

    pub fn load_for_inference(
        directory: &Path,
        ticker: &str,
        source_bars: usize,
        fingerprint: &str,
        supports: &Supports,
    ) -> Result<Self> {
        ensure!(
            !ticker.is_empty()
                && ticker
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_')),
            "invalid configured ticker"
        );
        let file = BarFile::open(&bar_file_path(directory, ticker, RESOLUTION))?;
        ensure!(
            file.symbol() == ticker && file.res_secs() == RESOLUTION,
            "inference ticker or resolution mismatch"
        );
        let dataset = Self::from_bars_with_frozen_partition(ticker, file.bars(), source_bars)?;
        ensure!(
            dataset.fingerprint == fingerprint && dataset.supports == *supports,
            "inference corpus changed the authenticated frozen prefix or its supports"
        );
        Ok(dataset)
    }

    fn from_bars_with_frozen_partition(
        ticker: &str,
        bars: &[PackedBar],
        source_bars: usize,
    ) -> Result<Self> {
        let n = bars.len();
        ensure!(
            source_bars <= n && source_bars >= 5000,
            "invalid frozen source bar count"
        );
        ensure!(n >= 5000, "single-ticker corpus needs at least 5000 bars for context, supports, and purged partitions");
        let now = chrono::Utc::now().timestamp_millis();
        ensure!(
            bars.last().unwrap().ts() + i64::from(RESOLUTION) * 1000 <= now,
            "corpus contains an uncompleted bar"
        );
        let cuts = [
            source_bars * 7 / 10,
            source_bars * 8 / 10,
            source_bars * 9 / 10,
        ];
        let mut digest = Digest::new(&SHA256);
        digest.update(DATA_CONTRACT.as_bytes());
        digest.update(ticker.as_bytes());
        let mut timestamps = Vec::with_capacity(n);
        let mut closes = Vec::with_capacity(n);
        let mut returns = Vec::with_capacity(n);
        let mut sigmas = Vec::with_capacity(n);
        let mut endogenous = Vec::with_capacity(n);
        let mut exogenous = Vec::with_capacity(n);
        let mut validity = Vec::with_capacity(n);
        let decay = (-std::f64::consts::LN_2 / 78.0).exp();
        let (mut variance, mut volume_mean, mut volume_seen) = (0.0_f64, 0.0_f64, false);
        let mut context_bad_prefix = vec![0usize];
        for (i, bar) in bars.iter().enumerate() {
            let close = f64::from(bar.close);
            ensure!(
                close.is_finite() && close > 0.0,
                "invalid close in configured ticker at bar {i}"
            );
            ensure!(
                i == 0 || bars[i - 1].ts() < bar.ts(),
                "timestamps must be strictly increasing"
            );
            ensure!(
                bar.ts().rem_euclid(300_000) == 0,
                "five-minute timestamp is off-grid at bar {i}"
            );
            if i < source_bars {
                digest.update(&bar.ts().to_le_bytes());
                for value in [bar.open, bar.high, bar.low, bar.close, bar.volume] {
                    digest.update(&value.to_bits().to_le_bytes());
                }
            }
            let previous = i.checked_sub(1).map(|j| bars[j].ts());
            let raw = if i == 0 {
                0.0
            } else {
                (close / f64::from(bars[i - 1].close)).ln()
            };
            let sigma = variance.sqrt();
            let ready = i >= WARMUP && sigma.is_finite() && sigma > 0.0;
            let e = if ready { (raw / sigma) as f32 } else { 0.0 };
            let ids = bar_time_ids(bar.ts(), previous, RESOLUTION, None);
            let clock = clock_features(ids);
            let (mut features, mut valid) = ([0.0_f32; 12], [1.0_f32; 12]);
            let (open, high, low) = (f64::from(bar.open), f64::from(bar.high), f64::from(bar.low));
            let geometry_valid = [open, high, low].iter().all(|v| v.is_finite() && *v > 0.0)
                && high >= low
                && high >= close
                && high >= open
                && low <= close
                && low <= open;
            if geometry_valid && ready {
                features[0] = ((high / low).ln() / sigma) as f32;
            } else {
                valid[0] = 0.0;
            }
            if geometry_valid && high > low {
                features[1] = ((close - low) / (high - low)) as f32;
                features[2] = ((open - low) / (high - low)) as f32;
            } else {
                valid[1] = 0.0;
                valid[2] = 0.0;
            }
            let volume = f64::from(bar.volume);
            if volume.is_finite() && volume >= 0.0 {
                let log_volume = volume.ln_1p();
                if volume_seen {
                    features[3] = (log_volume - volume_mean) as f32;
                } else {
                    valid[3] = 0.0;
                }
                volume_mean = if volume_seen {
                    decay * volume_mean + (1.0 - decay) * log_volume
                } else {
                    log_volume
                };
                volume_seen = true;
            } else {
                valid[3] = 0.0;
            }
            if ready {
                features[4] = sigma.ln() as f32;
            } else {
                valid[4] = 0.0;
            }
            features[5..].copy_from_slice(&clock);
            if previous.is_none() {
                valid[10] = 0.0;
                valid[11] = 0.0;
            }
            let finite = e.is_finite() && features.iter().all(|v| v.is_finite());
            ensure!(finite, "nonfinite standardized feature at bar {i}");
            timestamps.push(bar.ts());
            closes.push(close);
            returns.push(raw);
            sigmas.push(sigma);
            endogenous.push(e);
            exogenous.push(features);
            validity.push(valid);
            context_bad_prefix.push(context_bad_prefix[i] + usize::from(!ready));
            if i > 0 {
                variance = if i == 1 {
                    raw * raw
                } else {
                    decay * variance + (1.0 - decay) * raw * raw
                };
            }
        }
        let valid_context: Vec<_> = (0..n)
            .map(|i| {
                i + 1 >= CONTEXT && context_bad_prefix[i + 1] == context_bad_prefix[i + 1 - CONTEXT]
            })
            .collect();
        let mut split_origins: [Vec<usize>; 4] = std::array::from_fn(|_| Vec::new());
        let bounds = [0, cuts[0], cuts[1], cuts[2], source_bars];
        for split in 0..4 {
            for origin in bounds[split]..bounds[split + 1].saturating_sub(PURGE) {
                if valid_context[origin] {
                    split_origins[split].push(origin);
                }
            }
            ensure!(!split_origins[split].is_empty(), "partition {split} has no complete 2048-bar contexts with 100 observed targets after purge");
        }
        let training_targets: Vec<_> = split_origins[0]
            .iter()
            .map(|&origin| targets_at(&closes, &sigmas, origin))
            .collect();
        let supports = Supports::fit(&training_targets)?;
        let weights = uniqueness_weights(n, &split_origins[0]);
        let fingerprint = digest
            .finish()
            .as_ref()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        Ok(Self {
            ticker: ticker.to_owned(),
            fingerprint,
            source_bars,
            split_bounds: cuts.map(|i| timestamps[i]),
            supports,
            timestamps,
            candles: bars
                .iter()
                .map(|bar| CandleBar {
                    open: bar.open,
                    high: bar.high,
                    low: bar.low,
                    close: bar.close,
                })
                .collect(),
            closes,
            returns,
            sigmas,
            endogenous,
            exogenous,
            validity,
            split_origins,
            weights,
            valid_context,
            prepared: None,
        })
    }

    pub fn origins(&self, split: Split) -> &[usize] {
        &self.split_origins[split.index()]
    }

    pub(super) fn candle_window(
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
            .context("candle window index overflow")?;
        self.candles
            .get(start..end)
            .context("candle future exceeds corpus")
    }
    pub fn target(&self, origin: usize) -> [f64; 6] {
        targets_at(&self.closes, &self.sigmas, origin)
    }
    pub fn timestamp(&self, origin: usize) -> i64 {
        self.timestamps[origin]
    }
    pub fn sigma(&self, origin: usize) -> f64 {
        self.sigmas[origin]
    }
    pub fn history_returns(&self, origin: usize) -> &[f64] {
        &self.returns[origin + 1 - CONTEXT..=origin]
    }
    pub fn har_features(&self, origin: usize) -> [f64; 4] {
        let returns = self.history_returns(origin);
        let mut output = [1.0; 4];
        for (j, n) in [78, 390, 1560].into_iter().enumerate() {
            let rv = returns[CONTEXT - n..].iter().map(|r| r * r).sum::<f64>() / n as f64;
            // Positive origin sigma supplies the causal fallback for an entirely flat span.
            output[j + 1] = if rv > 0.0 {
                rv.ln()
            } else {
                (self.sigma(origin).powi(2)).ln()
            };
        }
        output
    }
    pub fn latest_origin(&self) -> usize {
        self.timestamps.len() - 1
    }

    /// Upload compact histories once; unfolding creates a view, not overlapping window copies.
    pub fn prepare(&mut self, device: Device) {
        if self
            .prepared
            .as_ref()
            .is_some_and(|cache| cache.device == device)
        {
            return;
        }
        let n = self.timestamps.len();
        let mut history = Vec::with_capacity(25 * n);
        history.extend_from_slice(&self.endogenous);
        for rows in [&self.exogenous, &self.validity] {
            for feature in 0..12 {
                history.extend(rows.iter().map(|row| row[feature]));
            }
        }
        let windows = Tensor::from_slice(&history)
            .reshape([25, n as i64])
            .to_device(device)
            .unfold(1, CONTEXT as i64, 1)
            .permute([1, 0, 2]);
        let clocks = scheduled_clocks(&self.timestamps[CONTEXT - 1..]);
        let future_clock = Tensor::from_slice(&clocks)
            .reshape([(n + 1 - CONTEXT) as i64, 6, CLOCK_DIM as i64])
            .to_device(device);
        self.prepared = Some(PreparedHistory {
            windows,
            future_clock,
            device,
        });
    }

    fn validate_origins(&self, origins: &[usize], labels: bool) -> Result<()> {
        ensure!(!origins.is_empty(), "cannot construct an empty batch");
        for &origin in origins {
            ensure!(
                origin < self.timestamps.len() && self.valid_context[origin],
                "origin {origin} lacks a valid completed causal context"
            );
            if labels {
                ensure!(
                    self.split_origins
                        .iter()
                        .any(|rows| rows.binary_search(&origin).is_ok()),
                    "origin {origin} is not an eligible purged forecast target"
                );
            }
        }
        Ok(())
    }

    pub fn batch(&self, origins: &[usize], device: Device) -> Result<Batch> {
        self.validate_origins(origins, true)?;
        self.prepared_batch(origins, device, true)
    }

    pub fn batch_reference(&self, origins: &[usize], device: Device) -> Result<Batch> {
        self.validate_origins(origins, true)?;
        self.build_batch(origins, device, true)
    }

    pub fn inference_batch(&self, origins: &[usize], device: Device) -> Result<Batch> {
        self.validate_origins(origins, false)?;
        self.prepared_batch(origins, device, false)
    }

    fn prepared_batch(&self, origins: &[usize], device: Device, labels: bool) -> Result<Batch> {
        let Some(cache) = self
            .prepared
            .as_ref()
            .filter(|cache| cache.device == device)
        else {
            return self.build_batch(origins, device, labels);
        };
        let starts: Vec<_> = origins
            .iter()
            .map(|&origin| (origin + 1 - CONTEXT) as i64)
            .collect();
        let mut labels_and_weights = Vec::with_capacity(origins.len() * 13);
        for &origin in origins {
            let target = if labels {
                self.target(origin)
            } else {
                [0.0; 6]
            };
            labels_and_weights.extend(target.map(|v| v as f32));
            // All 128 categorical indices are exactly representable in the shared float transfer.
            labels_and_weights.extend(
                target
                    .iter()
                    .zip(&self.supports.horizons)
                    .map(|(&v, support)| support.bin(v) as f32),
            );
            labels_and_weights.push(if labels {
                self.weights[origin] as f32
            } else {
                1.0
            });
        }
        let indices = Tensor::from_slice(&starts).to_device(device);
        let metadata = Tensor::from_slice(&labels_and_weights)
            .reshape([origins.len() as i64, 13])
            .to_device(device);
        let histories = cache.windows.index_select(0, &indices);
        Ok(Batch {
            endogenous: histories.select(1, 0),
            exogenous: histories.narrow(1, 1, 12),
            validity: histories.narrow(1, 13, 12),
            future_clock: cache.future_clock.index_select(0, &indices),
            targets: metadata.narrow(1, 0, 6),
            bins: metadata.narrow(1, 6, 6).to_kind(Kind::Int64),
            weights: metadata.select(1, 12),
        })
    }

    fn build_batch(&self, origins: &[usize], device: Device, labels: bool) -> Result<Batch> {
        ensure!(!origins.is_empty(), "cannot construct an empty batch");
        let b = origins.len();
        let mut endogenous = Vec::with_capacity(b * CONTEXT);
        let mut exogenous = Vec::with_capacity(b * CONTEXT * 12);
        let mut validity = Vec::with_capacity(b * CONTEXT * 12);
        let mut future_clock = Vec::with_capacity(b * 6 * CLOCK_DIM);
        let mut targets = Vec::with_capacity(b * 6);
        let mut bins = Vec::with_capacity(b * 6);
        let mut weights = Vec::with_capacity(b);
        for &origin in origins {
            ensure!(
                origin < self.timestamps.len() && self.valid_context[origin],
                "origin {origin} lacks a valid completed causal context"
            );
            let start = origin + 1 - CONTEXT;
            endogenous.extend_from_slice(&self.endogenous[start..=origin]);
            for feature in 0..12 {
                exogenous.extend(
                    self.exogenous[start..=origin]
                        .iter()
                        .map(|row| row[feature]),
                );
                validity.extend(self.validity[start..=origin].iter().map(|row| row[feature]));
            }
            let schedule = forecast_schedule_after(self.timestamps[origin], PURGE, RESOLUTION);
            for &h in &HORIZONS {
                let previous = if h == 1 {
                    self.timestamps[origin]
                } else {
                    schedule[h - 2]
                };
                future_clock.extend_from_slice(&clock_features(bar_time_ids(
                    schedule[h - 1],
                    Some(previous),
                    RESOLUTION,
                    None,
                )));
            }
            let target = if labels {
                self.target(origin)
            } else {
                [0.0; 6]
            };
            targets.extend(target.map(|v| v as f32));
            bins.extend(
                target
                    .iter()
                    .zip(&self.supports.horizons)
                    .map(|(&v, support)| support.bin(v) as i64),
            );
            weights.push(if labels {
                self.weights[origin] as f32
            } else {
                1.0
            });
        }
        let float = |values: &[f32], shape: &[i64]| {
            Tensor::from_slice(values).reshape(shape).to_device(device)
        };
        Ok(Batch {
            endogenous: float(&endogenous, &[b as i64, CONTEXT as i64]),
            exogenous: float(&exogenous, &[b as i64, 12, CONTEXT as i64]),
            validity: float(&validity, &[b as i64, 12, CONTEXT as i64]),
            future_clock: float(&future_clock, &[b as i64, 6, CLOCK_DIM as i64]),
            targets: float(&targets, &[b as i64, 6]),
            bins: Tensor::from_slice(&bins)
                .reshape([b as i64, 6])
                .to_device(device),
            weights: float(&weights, &[b as i64]),
        })
    }
}

fn scheduled_clocks(timestamps: &[i64]) -> Vec<f32> {
    let mut schedule: VecDeque<(i64, [f32; CLOCK_DIM])> = VecDeque::with_capacity(PURGE);
    let mut output = Vec::with_capacity(timestamps.len() * 6 * CLOCK_DIM);
    let mut predecessor = None;
    for &origin in timestamps {
        while schedule
            .front()
            .is_some_and(|&(timestamp, _)| timestamp <= origin)
        {
            predecessor = schedule.pop_front().map(|(timestamp, _)| timestamp);
        }
        let mut previous = schedule.back().map_or(origin, |&(timestamp, _)| timestamp);
        if schedule.len() < PURGE {
            for timestamp in forecast_schedule_after(previous, PURGE - schedule.len(), RESOLUTION) {
                let clock =
                    clock_features(bar_time_ids(timestamp, Some(previous), RESOLUTION, None));
                schedule.push_back((timestamp, clock));
                previous = timestamp;
            }
        }
        for &horizon in &HORIZONS {
            let (timestamp, mut clock) = schedule[horizon - 1];
            if horizon == 1 && predecessor != Some(origin) {
                clock = clock_features(bar_time_ids(timestamp, Some(origin), RESOLUTION, None));
            }
            output.extend_from_slice(&clock);
        }
    }
    output
}

fn targets_at(closes: &[f64], sigmas: &[f64], origin: usize) -> [f64; 6] {
    HORIZONS
        .map(|h| (closes[origin + h] / closes[origin]).ln() / (sigmas[origin] * (h as f64).sqrt()))
}

fn clock_features(ids: [i64; 9]) -> [f32; CLOCK_DIM] {
    let minute = std::f64::consts::TAU * (ids[0] - 240) as f64 / 960.0;
    let weekday = std::f64::consts::TAU * ids[1] as f64 / 7.0;
    [
        minute.sin() as f32,
        minute.cos() as f32,
        weekday.sin() as f32,
        weekday.cos() as f32,
        ids[2] as f32 / 3.0,
        ids[4] as f32 / 13.0,
        f32::from(ids[5] == 2),
    ]
}

fn uniqueness_weights(n: usize, origins: &[usize]) -> Vec<f64> {
    let mut changes = vec![0_i64; n + 1];
    for &origin in origins {
        changes[origin + 1] += 1;
        changes[origin + PURGE + 1] -= 1;
    }
    let mut inverse_prefix = vec![0.0; n + 1];
    let mut concurrent = 0_i64;
    for i in 0..n {
        concurrent += changes[i];
        inverse_prefix[i + 1] = inverse_prefix[i]
            + if concurrent > 0 {
                1.0 / concurrent as f64
            } else {
                0.0
            };
    }
    let mut weights = vec![1.0; n];
    let mut total = 0.0;
    for &origin in origins {
        weights[origin] =
            (inverse_prefix[origin + PURGE + 1] - inverse_prefix[origin + 1]) / PURGE as f64;
        total += weights[origin];
    }
    let mean = total / origins.len() as f64;
    for &origin in origins {
        weights[origin] /= mean;
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<PackedBar> {
        let start = chrono::DateTime::parse_from_rfc3339("2023-01-03T08:55:00-05:00")
            .unwrap()
            .timestamp_millis();
        let timestamps = forecast_schedule_after(start, 6000, RESOLUTION);
        let mut close = 100.0_f32;
        timestamps
            .into_iter()
            .enumerate()
            .map(|(i, ts_ms)| {
                let open = close;
                let r = 0.003 * ((i as f64 * 0.731).sin() + 0.4 * (i as f64 * 0.131).cos());
                close *= r.exp() as f32;
                PackedBar {
                    ts_ms,
                    open,
                    high: open.max(close) * 1.001,
                    low: open.min(close) * 0.999,
                    close,
                    volume: 1000.0 + (i % 73) as f32,
                    vwap: close,
                    trades: 10,
                }
            })
            .collect()
    }

    #[test]
    fn candle_windows_retain_exact_authenticated_ohlc_and_origin_alignment() {
        let bars = fixture();
        let dataset = Dataset::from_bars("ONLY", &bars).unwrap();
        let origin = dataset.origins(Split::Validation)[0];
        let window = dataset.candle_window(origin, 32, 100).unwrap();
        assert_eq!(window.len(), 133);
        for (candle, bar) in window.iter().zip(&bars[origin - 32..=origin + 100]) {
            assert_eq!(
                [candle.open, candle.high, candle.low, candle.close],
                [bar.open, bar.high, bar.low, bar.close]
            );
        }
        assert_eq!(window[32].close, bars[origin].close as f32);
        assert!(dataset.candle_window(0, 32, 100).is_err());
        assert!(dataset.candle_window(bars.len() - 1, 32, 100).is_err());
        assert!(dataset.candle_window(usize::MAX, 0, 1).is_err());
    }

    fn assert_batches_equal(left: &Batch, right: &Batch) {
        for (a, b) in [
            (&left.endogenous, &right.endogenous),
            (&left.exogenous, &right.exogenous),
            (&left.validity, &right.validity),
            (&left.future_clock, &right.future_clock),
            (&left.targets, &right.targets),
            (&left.bins, &right.bins),
            (&left.weights, &right.weights),
        ] {
            assert_eq!(a.size(), b.size());
            assert!(a.equal(b));
        }
    }

    #[test]
    fn prepared_windows_preserve_rows_masks_labels_and_latest_inference() {
        let mut bars = fixture();
        bars[3000].volume = f32::NAN;
        bars.remove(4100);
        let mut dataset = Dataset::from_bars("ONLY", &bars).unwrap();
        let first = dataset.origins(Split::Train)[0];
        let origins = [
            dataset.origins(Split::Validation)[0],
            first,
            first + 1,
            first,
        ];
        let reference = dataset.batch_reference(&origins, Device::Cpu).unwrap();
        let latest = dataset.latest_origin();
        let inference = dataset.inference_batch(&[latest], Device::Cpu).unwrap();
        dataset.prepare(Device::Cpu);
        assert_batches_equal(&reference, &dataset.batch(&origins, Device::Cpu).unwrap());
        assert_batches_equal(
            &inference,
            &dataset.inference_batch(&[latest], Device::Cpu).unwrap(),
        );
        assert!(dataset.batch(&[latest], Device::Cpu).is_err());
        assert!(dataset.batch(&[], Device::Cpu).is_err());
        assert!(dataset.inference_batch(&[0], Device::Cpu).is_err());
        let windows = &dataset.prepared.as_ref().unwrap().windows;
        assert_eq!(windows.stride(), [1, bars.len() as i64, 1]);
    }

    #[test]
    fn rolling_clock_cache_matches_per_origin_calendar_across_gaps_and_dst() {
        let timestamps: Vec<_> = [
            "2024-03-08T19:55:00-05:00",
            "2024-03-10T12:00:00-04:00",
            "2024-03-11T04:00:00-04:00",
            "2024-03-11T04:10:00-04:00",
            "2024-07-03T19:55:00-04:00",
            "2024-07-04T12:00:00-04:00",
            "2024-07-05T04:00:00-04:00",
            "2024-11-01T19:55:00-04:00",
            "2024-11-04T04:00:00-05:00",
        ]
        .map(|time| {
            chrono::DateTime::parse_from_rfc3339(time)
                .unwrap()
                .timestamp_millis()
        })
        .to_vec();
        let cached = scheduled_clocks(&timestamps);
        for (&origin, cached) in timestamps.iter().zip(cached.chunks_exact(6 * CLOCK_DIM)) {
            let schedule = forecast_schedule_after(origin, PURGE, RESOLUTION);
            for (&h, actual) in HORIZONS.iter().zip(cached.chunks_exact(CLOCK_DIM)) {
                let previous = if h == 1 { origin } else { schedule[h - 2] };
                assert_eq!(
                    actual,
                    clock_features(bar_time_ids(
                        schedule[h - 1],
                        Some(previous),
                        RESOLUTION,
                        None
                    ))
                );
            }
        }
    }

    #[test]
    fn future_perturbation_cannot_change_history_or_training_supports() {
        let bars = fixture();
        let original = Dataset::from_bars("ONLY", &bars).unwrap();
        let origin = original.origins(Split::Validation)[0];
        let mut changed = bars.clone();
        for row in &mut changed[origin + 1..] {
            row.close *= 1.001;
            row.open *= 1.001;
            row.high *= 1.001;
            row.low *= 1.001;
        }
        let altered = Dataset::from_bars("ONLY", &changed).unwrap();
        assert_eq!(original.supports, altered.supports);
        assert_eq!(
            &original.endogenous[..=origin],
            &altered.endogenous[..=origin]
        );
        assert_eq!(
            &original.exogenous[..=origin],
            &altered.exogenous[..=origin]
        );
        assert_eq!(original.sigmas[origin], altered.sigmas[origin]);
        assert_ne!(original.target(origin), altered.target(origin));
        assert_ne!(original.fingerprint, altered.fingerprint);
        for (split, boundary) in [Split::Train, Split::Calibration, Split::Validation]
            .into_iter()
            .zip(original.split_bounds)
        {
            let last = *original.origins(split).last().unwrap();
            assert!(original.timestamp(last + PURGE) < boundary);
        }
    }

    #[test]
    fn missing_future_bars_do_not_select_origins_or_shift_nominal_clocks() {
        let mut bars = fixture();
        let origin = 4599;
        let future = forecast_schedule_after(bars[origin].ts(), PURGE, RESOLUTION);
        bars.remove(origin + 1);
        let dataset = Dataset::from_bars("ONLY", &bars).unwrap();
        assert!(dataset.origins(Split::Calibration).contains(&origin));
        assert_eq!(
            future,
            forecast_schedule_after(dataset.timestamp(origin), PURGE, RESOLUTION)
        );
        assert_ne!(future[0], dataset.timestamp(origin + 1));
        assert_eq!(
            dataset.target(origin)[0],
            (dataset.closes[origin + 1] / dataset.closes[origin]).ln() / dataset.sigma(origin)
        );
    }

    #[test]
    fn appended_history_preserves_frozen_supports_splits_and_identity() {
        let bars = fixture();
        let original = Dataset::from_bars("ONLY", &bars[..5500]).unwrap();
        let appended = Dataset::from_bars_with_frozen_partition("ONLY", &bars, 5500).unwrap();
        assert_eq!(original.supports, appended.supports);
        assert_eq!(original.split_bounds, appended.split_bounds);
        assert_eq!(original.fingerprint, appended.fingerprint);
        assert_eq!(appended.latest_origin(), 5999);
        assert_eq!(&original.sigmas[..], &appended.sigmas[..5500]);
    }

    #[test]
    fn volatility_uses_only_prior_returns_and_targets_are_cumulative() {
        let bars = fixture();
        let original = Dataset::from_bars("ONLY", &bars).unwrap();
        let origin = original.origins(Split::Validation)[0];
        for (j, h) in HORIZONS.into_iter().enumerate() {
            let summed = original.returns[origin + 1..=origin + h]
                .iter()
                .sum::<f64>();
            assert!(
                (original.target(origin)[j]
                    - summed / (original.sigma(origin) * (h as f64).sqrt()))
                .abs()
                    < 1e-10
            );
        }
        let mut altered_bars = bars;
        altered_bars[origin].close *= 1.01;
        altered_bars[origin].high = altered_bars[origin].close.max(altered_bars[origin].high);
        let altered = Dataset::from_bars("ONLY", &altered_bars).unwrap();
        assert_eq!(original.sigma(origin), altered.sigma(origin));
        assert_ne!(original.endogenous[origin], altered.endogenous[origin]);
        assert_ne!(original.sigma(origin + 1), altered.sigma(origin + 1));
    }

    #[test]
    fn missing_geometry_and_volume_have_explicit_masks() {
        let mut bars = fixture();
        let origin = 5000;
        bars[origin].open = bars[origin].close;
        bars[origin].high = bars[origin].close;
        bars[origin].low = bars[origin].close;
        bars[origin].volume = f32::NAN;
        let dataset = Dataset::from_bars("ONLY", &bars).unwrap();
        assert_eq!(&dataset.validity[origin][..5], &[1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(&dataset.exogenous[origin][..4], &[0.0; 4]);
        assert!(dataset.exogenous[origin + 1][3].is_finite());
    }

    #[test]
    fn uniqueness_is_global_and_rewards_nonoverlapping_information() {
        let origins = [0, 1, 2, 200];
        let weights = uniqueness_weights(400, &origins);
        assert!((origins.iter().map(|&i| weights[i]).sum::<f64>() / 4.0 - 1.0).abs() < 1e-12);
        assert!(weights[200] > weights[0]);
        assert!(weights[0] > weights[1]);
    }

    #[test]
    fn calendar_skips_holiday_and_uses_future_predecessor_for_gap_ids() {
        let start = chrono::DateTime::parse_from_rfc3339("2024-07-03T19:55:00-04:00")
            .unwrap()
            .timestamp_millis();
        let next = forecast_schedule_after(start, 2, RESOLUTION);
        let expected = chrono::DateTime::parse_from_rfc3339("2024-07-05T04:00:00-04:00")
            .unwrap()
            .timestamp_millis();
        assert_eq!(next[0], expected);
        assert_eq!(
            clock_features(bar_time_ids(next[0], Some(start), RESOLUTION, None))[6],
            1.0
        );
        assert_eq!(
            clock_features(bar_time_ids(next[1], Some(next[0]), RESOLUTION, None))[6],
            0.0
        );
    }
}
