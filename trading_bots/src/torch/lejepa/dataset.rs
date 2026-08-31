use anyhow::{ensure, Result};
use rayon::prelude::*;
use shared::bars::{PackedBar, FILE_EXTENSION};
use tch::{Device, Tensor};

use crate::torch::dataset::{BarCorpus, BarSampler, Split, WindowRef};

use super::model::{BAR_FEATURES, MAX_CONTEXT_BARS};

pub const DEFAULT_MIN_BARS: usize = 10 * MAX_CONTEXT_BARS as usize;

pub struct MseJepaDataset {
    corpus: BarCorpus,
    train: BarSampler,
    validation: BarSampler,
}

impl MseJepaDataset {
    pub fn load(
        directory: &std::path::Path,
        resolution_secs: u32,
        min_bars: usize,
        split_bounds: Option<(i64, i64)>,
        seed: u64,
    ) -> Result<Self> {
        let corpus = match split_bounds {
            Some(bounds) => {
                BarCorpus::load_with_bounds(directory, resolution_secs, min_bars, bounds)?
            }
            None => BarCorpus::load(directory, resolution_secs, min_bars)?,
        };
        let train = BarSampler::new(&corpus, Split::Train, MAX_CONTEXT_BARS, seed);
        let validation = BarSampler::new(&corpus, Split::Val, MAX_CONTEXT_BARS, seed);
        ensure!(
            !train.is_empty(),
            "MSE-JEPA training split has no {MAX_CONTEXT_BARS}-bar windows in {} *.{resolution_secs}.{FILE_EXTENSION} corpus",
            directory.display()
        );
        ensure!(
            !validation.is_empty(),
            "MSE-JEPA validation split has no {MAX_CONTEXT_BARS}-bar windows in {} *.{resolution_secs}.{FILE_EXTENSION} corpus",
            directory.display()
        );
        Ok(Self {
            corpus,
            train,
            validation,
        })
    }

    pub fn batches_per_epoch(&self, batch_size: usize) -> usize {
        self.train.batches_per_epoch(batch_size)
    }

    pub fn train_batch(
        &self,
        epoch: usize,
        index: usize,
        batch_size: usize,
        device: Device,
    ) -> Tensor {
        let refs = self.train.batch_refs(epoch, index, batch_size);
        self.features(&refs, device)
    }

    pub fn validation_refs(&self, windows: usize) -> Vec<WindowRef> {
        self.validation.pinned_windows(windows)
    }

    pub fn features(&self, refs: &[WindowRef], device: Device) -> Tensor {
        assert!(!refs.is_empty(), "cannot build an empty MSE-JEPA batch");
        let length = (MAX_CONTEXT_BARS + 1) as usize;
        let window_features = length * BAR_FEATURES as usize;
        let mut features = vec![0.0; refs.len() * window_features];
        features
            .par_chunks_mut(window_features)
            .zip(refs.par_iter())
            .for_each(|(output, window)| {
                let bars = self.corpus.bars(window.symbol as usize);
                let start = window.bar_index as usize;
                assert!(start > 0, "MSE-JEPA feature windows need a predecessor bar");
                assert!(
                    start + length <= bars.len(),
                    "MSE-JEPA window escapes its mmap"
                );
                output
                    .chunks_exact_mut(BAR_FEATURES as usize)
                    .enumerate()
                    .for_each(|(offset, features)| {
                        let index = start + offset;
                        features.copy_from_slice(&ohlc_features(&bars[index - 1], &bars[index]));
                    });
            });
        Tensor::from_slice(&features)
            .view([refs.len() as i64, 1, length as i64, BAR_FEATURES])
            .to_device(device)
    }

    pub fn split_bounds(&self) -> (i64, i64) {
        self.corpus.split_bounds()
    }
}

pub fn ohlc_features(previous: &PackedBar, current: &PackedBar) -> [f32; BAR_FEATURES as usize] {
    let open = current.open as f64;
    let close = current.close as f64;
    let high = (current.high as f64).max(open).max(close);
    let low = (current.low as f64).min(open).min(close);
    let previous_open = previous.open as f64;
    let previous_close = previous.close as f64;
    let previous_high = (previous.high as f64)
        .max(previous_open)
        .max(previous_close);
    let previous_low = (previous.low as f64).min(previous_open).min(previous_close);
    [
        relative_delta(open, previous_open),
        relative_delta(high, previous_high),
        relative_delta(low, previous_low),
        relative_delta(close, previous_close),
        relative_delta(open, high),
        relative_delta(open, low),
        relative_delta(open, close),
        relative_delta(high, open),
        relative_delta(high, low),
        relative_delta(high, close),
        relative_delta(low, open),
        relative_delta(low, high),
        relative_delta(low, close),
        relative_delta(close, open),
        relative_delta(close, high),
        relative_delta(close, low),
    ]
}

fn relative_delta(value: f64, reference: f64) -> f32 {
    if value.is_finite() && reference.is_finite() && reference > 0.0 {
        (value / reference - 1.0) as f32
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::ohlc_features;
    use shared::bars::PackedBar;

    fn bar(open: f32, high: f32, low: f32, close: f32) -> PackedBar {
        PackedBar {
            open,
            high,
            low,
            close,
            ..PackedBar::default()
        }
    }

    #[test]
    fn mmap_adapter_reproduces_the_exact_sixteen_feature_layout() {
        let previous = bar(100.0, 105.0, 98.0, 102.0);
        let current = bar(102.0, 108.0, 101.0, 106.0);
        let actual = ohlc_features(&previous, &current);
        let delta = |value: f64, reference: f64| (value / reference - 1.0) as f32;
        let expected = [
            delta(102.0, 100.0),
            delta(108.0, 105.0),
            delta(101.0, 98.0),
            delta(106.0, 102.0),
            delta(102.0, 108.0),
            delta(102.0, 101.0),
            delta(102.0, 106.0),
            delta(108.0, 102.0),
            delta(108.0, 101.0),
            delta(108.0, 106.0),
            delta(101.0, 102.0),
            delta(101.0, 108.0),
            delta(101.0, 106.0),
            delta(106.0, 102.0),
            delta(106.0, 108.0),
            delta(106.0, 101.0),
        ];
        for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            assert!((actual - expected).abs() < 1e-7, "feature {index}");
        }
    }

    #[test]
    fn high_low_are_repaired_before_inter_and_intrabar_ratios() {
        let previous = bar(100.0, 99.0, 103.0, 102.0);
        let current = bar(104.0, 101.0, 108.0, 106.0);
        let actual = ohlc_features(&previous, &current);
        let delta = |value: f64, reference: f64| (value / reference - 1.0) as f32;
        assert!((actual[1] - delta(106.0, 102.0)).abs() < 1e-7);
        assert!((actual[2] - delta(104.0, 100.0)).abs() < 1e-7);
        assert!((actual[8] - delta(106.0, 104.0)).abs() < 1e-7);
    }

    #[test]
    fn invalid_prices_map_to_zero_like_the_historical_builder() {
        let previous = bar(0.0, f32::NAN, 0.0, 0.0);
        let current = bar(f32::INFINITY, f32::NAN, -1.0, f32::NAN);
        assert!(ohlc_features(&previous, &current)
            .into_iter()
            .all(|value| value == 0.0));
    }
}
