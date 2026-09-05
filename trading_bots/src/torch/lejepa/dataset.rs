use std::path::PathBuf;

use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
use shared::bars::{PackedBar, FILE_EXTENSION};
use tch::{Device, Tensor};

use crate::torch::bar_dist::BAR_DOF;
use crate::torch::dataset::{
    BarBatch, BarCorpus, BarSampler, BatchScratch, PassPlan, Split, WindowRef,
};

use super::model::{BAR_FEATURES, MAX_CONTEXT_BARS};

pub const DEFAULT_MIN_BARS: usize = 10 * MAX_CONTEXT_BARS as usize;
const SIGREG_SAMPLER_SEED_DOMAIN: u64 = 0x5349_4752_4547_3132;

/// A deterministic fixed panel for scoring ancestral world-model rollouts.
///
/// `current_dof[:, t]` is the raw bar encoded by `contexts[:, :, t]`;
/// `future_dof[:, h]` scores the held-out continuation at horizon `h + 1`.
pub struct MseJepaRolloutPanel {
    pub refs: Vec<WindowRef>,
    pub contexts: Tensor,
    pub current_dof: Tensor,
    pub future_dof: Tensor,
    /// `[windows, MAX_CONTEXT_BARS + future, BAR_TIME_FEATURES]` calendar ids over the whole
    /// window, aligned with `current_dof` then `future_dof`. Carried for scoring external
    /// calendar-conditioned baselines (the LLM `BarWorldModel`) on the identical panel.
    pub time_ids: Tensor,
}

pub struct MseJepaDataset {
    corpus: BarCorpus,
    train: BarSampler,
    train_pass: PassPlan,
    sigreg_train: BarSampler,
    validation: BarSampler,
    seed: u64,
}
pub struct MseJepaTrainHostBatch {
    bars: Tensor,
    next_dof: Tensor,
    sigreg_bars: Tensor,
    prediction_windows: usize,
}

/// Deterministic readout-only batch: aligned context features and next-bar targets, with no
/// SIGReg sampler, temporal-view generation, or auxiliary tensor allocation.
pub struct MseJepaReadoutHostBatch {
    bars: Tensor,
    next_dof: Tensor,
}

impl MseJepaReadoutHostBatch {
    pub fn to_device(self, device: Device) -> (Tensor, Tensor) {
        (self.bars.to_device(device), self.next_dof.to_device(device))
    }
}

impl MseJepaTrainHostBatch {
    pub fn prediction_windows(&self) -> usize {
        self.prediction_windows
    }

    pub fn to_device(self, device: Device) -> (Tensor, Tensor, Tensor) {
        (
            self.bars.to_device(device),
            self.next_dof.to_device(device),
            self.sigreg_bars.to_device(device),
        )
    }
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
        let train_pass =
            PassPlan::new_with_future(&corpus, Split::Train, &[MAX_CONTEXT_BARS], &[1.0], 1, seed)
                .context("failed partitioning the MSE-JEPA training split")?;
        let sigreg_train = BarSampler::new(
            &corpus,
            Split::Train,
            MAX_CONTEXT_BARS,
            seed ^ SIGREG_SAMPLER_SEED_DOMAIN,
        );
        let validation = BarSampler::new(&corpus, Split::Val, MAX_CONTEXT_BARS, seed);
        ensure!(
            train_pass.windows_per_stage()[0] > 0,
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
            train_pass,
            sigreg_train,
            validation,
            seed,
        })
    }

    pub fn prediction_windows_per_epoch(&self) -> usize {
        self.train_pass.windows_per_stage()[0]
    }

    pub fn batches_per_epoch(&self, batch_size: usize) -> usize {
        assert!(batch_size > 0, "batch size must be positive");
        self.train_pass.windows_per_stage()[0].div_ceil(batch_size)
    }

    pub fn sigreg_batches_per_pass(&self, batch_size: usize) -> usize {
        self.sigreg_train.batches_per_epoch(batch_size)
    }

    pub fn train_scratch(&self) -> BatchScratch {
        self.train.scratch()
    }

    pub fn train_host_batch(
        &self,
        epoch: usize,
        index: usize,
        prediction_batch_size: usize,
        step: usize,
        sigreg_batch_size: usize,
        temporal_offsets: &[i64],
        scratch: &mut BatchScratch,
    ) -> MseJepaTrainHostBatch {
        assert!(
            prediction_batch_size > 0,
            "prediction batch size must be positive"
        );
        let batches = self.batches_per_epoch(prediction_batch_size);
        assert!(
            index < batches,
            "prediction batch index {index} out of range for {batches} batches per epoch"
        );
        let cursor = index
            .checked_mul(prediction_batch_size)
            .expect("prediction batch cursor overflow");
        let layout = self.train_pass.layout(epoch);
        let prediction_refs = layout.draw(0, cursor, prediction_batch_size);
        assert!(
            !prediction_refs.is_empty(),
            "prediction pass returned an empty in-range batch"
        );
        let bars = self.feature_tensor(
            prediction_refs,
            (MAX_CONTEXT_BARS + 1) as usize,
            |position| position,
        );
        let sample = self
            .train
            .batch_of_into(prediction_refs, Device::Cpu, scratch);
        let next_dof = next_bar_dof_from_batch(sample);

        let sigreg_batches = self.sigreg_train.batches_per_epoch(sigreg_batch_size);
        assert!(
            sigreg_batches > 0,
            "SIGReg batch exceeds the training window count"
        );
        let sigreg_epoch = step / sigreg_batches;
        let sigreg_index = step % sigreg_batches;
        let sigreg_refs =
            self.sigreg_train
                .batch_refs(sigreg_epoch, sigreg_index, sigreg_batch_size);
        let sigreg_bars = self.features_at_offsets_host(&sigreg_refs, temporal_offsets);
        MseJepaTrainHostBatch {
            bars,
            next_dof,
            sigreg_bars,
            prediction_windows: prediction_refs.len(),
        }
    }

    /// One deterministic readout-only optimizer batch. `step` advances the existing
    /// pass-shuffled training plan and wraps by beginning the next deterministic pass.
    pub fn readout_train_host_batch(
        &self,
        step: usize,
        batch_size: usize,
        scratch: &mut BatchScratch,
    ) -> MseJepaReadoutHostBatch {
        assert!(batch_size > 0, "readout batch size must be positive");
        let batches = self.batches_per_epoch(batch_size);
        assert!(
            batches > 0,
            "readout batch exceeds the training window count"
        );
        let epoch = step / batches;
        let index = step % batches;
        let cursor = index
            .checked_mul(batch_size)
            .expect("readout batch cursor overflow");
        let layout = self.train_pass.layout(epoch);
        let refs = layout.draw(0, cursor, batch_size);
        assert!(!refs.is_empty(), "readout pass returned an empty batch");
        let bars = self.feature_tensor(refs, (MAX_CONTEXT_BARS + 1) as usize, |position| position);
        let next_dof =
            next_bar_dof_from_batch(self.train.batch_of_into(refs, Device::Cpu, scratch));
        MseJepaReadoutHostBatch { bars, next_dof }
    }

    /// Raw next-bar DOF targets `[windows, MAX_CONTEXT_BARS, BAR_DOF]` aligned with the
    /// validation feature windows: row `t` is the bar encoded at feature position `t + 1`.
    pub fn validation_next_dof(&self, refs: &[WindowRef], device: Device) -> Tensor {
        let mut scratch = self.validation.scratch();
        let sample = self
            .validation
            .batch_of_into(refs, Device::Cpu, &mut scratch);
        next_bar_dof_from_batch(sample).to_device(device)
    }

    pub fn validation_sigreg_batch(
        &self,
        batch_size: usize,
        temporal_offsets: &[i64],
        device: Device,
    ) -> Result<Tensor> {
        let refs = self.validation.pinned_windows(batch_size);
        ensure!(
            refs.len() == batch_size,
            "requested {batch_size} validation SIGReg samples, but only {} are available",
            refs.len()
        );
        Ok(self.features_at_offsets(&refs, temporal_offsets, device))
    }

    pub fn validation_refs(&self, windows: usize) -> Vec<WindowRef> {
        self.validation.pinned_windows(windows)
    }

    pub fn rollout_panel(
        &self,
        split: Split,
        windows: usize,
        future_bars: usize,
        device: Device,
    ) -> Result<MseJepaRolloutPanel> {
        ensure!(windows > 0, "rollout panel windows must be positive");
        ensure!(future_bars > 0, "rollout future horizon must be positive");
        let sampler = BarSampler::new_with_future(
            &self.corpus,
            split,
            MAX_CONTEXT_BARS,
            future_bars as i64,
            self.seed,
        );
        let refs = sampler.pinned_windows(windows);
        ensure!(
            refs.len() == windows,
            "requested {windows} {split:?} rollout windows, but only {} full-context windows \
             carry {future_bars} held-out future bars",
            refs.len()
        );
        let contexts = self.features_with_length(&refs, MAX_CONTEXT_BARS as usize, device);
        let mut scratch = sampler.scratch();
        let sample = sampler.batch_of_into(&refs, device, &mut scratch);
        let time_ids = sample.time_ids.shallow_clone();
        let (current_dof, future_dof) = split_raw_dof_from_batch(sample, future_bars);
        Ok(MseJepaRolloutPanel {
            refs,
            contexts,
            current_dof,
            future_dof,
            time_ids,
        })
    }
    pub fn advise_random_access(&self) -> Result<()> {
        self.corpus.advise_random_access()
    }

    pub fn corpus_fingerprint(&self) -> String {
        self.corpus.identity_fingerprint()
    }

    pub fn supports_path(&self) -> PathBuf {
        self.corpus.supports_path()
    }

    pub fn resolution_secs(&self) -> u32 {
        self.corpus.res_secs()
    }

    pub fn features(&self, refs: &[WindowRef], device: Device) -> Tensor {
        self.features_with_length(refs, (MAX_CONTEXT_BARS + 1) as usize, device)
    }

    fn features_with_length(&self, refs: &[WindowRef], length: usize, device: Device) -> Tensor {
        self.features_with_range(refs, 0, length, device)
    }

    fn features_with_range(
        &self,
        refs: &[WindowRef],
        offset: usize,
        length: usize,
        device: Device,
    ) -> Tensor {
        self.feature_tensor(refs, length, |position| offset + position)
            .to_device(device)
    }

    fn features_at_offsets(&self, refs: &[WindowRef], offsets: &[i64], device: Device) -> Tensor {
        self.features_at_offsets_host(refs, offsets)
            .to_device(device)
    }

    fn features_at_offsets_host(&self, refs: &[WindowRef], offsets: &[i64]) -> Tensor {
        assert!(
            offsets.iter().all(|&offset| offset >= 0),
            "MSE-JEPA feature offsets must be non-negative"
        );
        self.feature_tensor(refs, offsets.len(), |position| offsets[position] as usize)
    }

    fn feature_tensor(
        &self,
        refs: &[WindowRef],
        positions: usize,
        relative_index: impl Fn(usize) -> usize + Sync,
    ) -> Tensor {
        assert!(!refs.is_empty(), "cannot build an empty MSE-JEPA batch");
        assert!(
            positions > 0,
            "MSE-JEPA feature positions must be non-empty"
        );
        let window_features = positions * BAR_FEATURES as usize;
        let mut features = vec![0.0; refs.len() * window_features];
        features
            .par_chunks_mut(window_features)
            .zip(refs.par_iter())
            .for_each(|(output, window)| {
                let bars = self.corpus.bars(window.symbol as usize);
                let start = window.bar_index as usize;
                output
                    .chunks_exact_mut(BAR_FEATURES as usize)
                    .enumerate()
                    .for_each(|(position, features)| {
                        let index = start + relative_index(position);
                        assert!(
                            index < bars.len(),
                            "MSE-JEPA feature offset escapes its mmap"
                        );
                        assert!(
                            index > 0,
                            "MSE-JEPA transition feature requires a preceding bar"
                        );
                        features.copy_from_slice(&transition_bar_features(
                            &bars[index - 1],
                            &bars[index],
                        ));
                    });
            });
        Tensor::from_slice(&features).view([refs.len() as i64, 1, positions as i64, BAR_FEATURES])
    }

    pub fn split_bounds(&self) -> (i64, i64) {
        self.corpus.split_bounds()
    }
}

fn split_raw_dof_from_batch(sample: BarBatch, future_bars: usize) -> (Tensor, Tensor) {
    let raw = sample.raw_dof.unwrap_or(sample.dof);
    split_raw_dof(&raw, future_bars)
}

fn split_raw_dof(raw: &Tensor, future_bars: usize) -> (Tensor, Tensor) {
    let context = MAX_CONTEXT_BARS;
    assert_eq!(
        raw.size().as_slice(),
        [raw.size()[0], context + future_bars as i64, BAR_DOF as i64],
        "rollout sampler batch shape drifted"
    );
    let current = raw.narrow(1, 0, context).detach();
    let future = raw.narrow(1, context, future_bars as i64).detach();
    (current, future)
}

/// `[windows, MAX_CONTEXT_BARS, BAR_DOF]` raw DOF of the bar following each source
/// position: the emission target for teacher-forced training.
fn next_bar_dof_from_batch(sample: BarBatch) -> Tensor {
    let raw = sample.raw_dof.unwrap_or(sample.dof);
    assert_eq!(
        raw.size().as_slice(),
        [raw.size()[0], MAX_CONTEXT_BARS + 1, BAR_DOF as i64],
        "training sampler batch shape drifted"
    );
    raw.narrow(1, 1, MAX_CONTEXT_BARS).detach()
}

pub fn transition_bar_features(
    previous: &PackedBar,
    current: &PackedBar,
) -> [f32; BAR_FEATURES as usize] {
    // `PackedBar` is packed, so copy each field before doing arithmetic.
    let previous_close = previous.close as f64;
    let previous_volume = previous.volume as f64;
    let (open, high, low, close, volume) = (
        current.open as f64,
        current.high as f64,
        current.low as f64,
        current.close as f64,
        current.volume as f64,
    );

    let volume_transition_present =
        previous_volume.is_finite() && previous_volume > 0.0 && volume.is_finite() && volume > 0.0;
    let (volume_delta, volume_delta_present) = if volume_transition_present {
        ((volume / previous_volume).ln() as f32, 1.0)
    } else {
        (0.0, 0.0)
    };

    if ![open, close]
        .into_iter()
        .all(|price| price.is_finite() && price > 0.0)
    {
        return [0.0, 0.0, 0.0, 0.0, 0.0, volume_delta, volume_delta_present];
    }

    let high = if high.is_finite() && high > 0.0 {
        high.max(open).max(close)
    } else {
        open.max(close)
    };
    let low = if low.is_finite() && low > 0.0 {
        low.min(open).min(close)
    } else {
        open.min(close)
    };
    let gap = if previous_close.is_finite() && previous_close > 0.0 {
        (open / previous_close).ln()
    } else {
        0.0
    };
    let body = (close / open).ln();
    let upper_wick = (high / open.max(close)).ln();
    let lower_wick = (open.min(close) / low).ln();
    let log_range = (high / low).ln();
    let gk_variance =
        0.5 * log_range * log_range - (2.0 * std::f64::consts::LN_2 - 1.0) * body * body;
    let gk_volatility = gk_variance.max(0.0).sqrt();

    [
        gap as f32,
        body as f32,
        upper_wick as f32,
        lower_wick as f32,
        gk_volatility as f32,
        volume_delta,
        volume_delta_present,
    ]
}

#[cfg(test)]
mod tests {
    use super::{split_raw_dof, transition_bar_features};
    use crate::torch::lejepa::model::MAX_CONTEXT_BARS;
    use shared::bars::PackedBar;
    use tch::{Device, Kind, Tensor};

    fn bar(open: f32, high: f32, low: f32, close: f32, volume: f32) -> PackedBar {
        PackedBar {
            open,
            high,
            low,
            close,
            volume,
            ..PackedBar::default()
        }
    }

    fn assert_features(actual: [f32; 7], expected: [f32; 7]) {
        for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "feature {index}: actual={actual}, expected={expected}"
            );
        }
    }
    #[test]
    fn transition_bar_adapter_uses_the_exact_seven_feature_reconstruction_basis() {
        let previous = bar(98.0, 101.0, 97.0, 100.0, 10_000.0);
        let current = bar(102.0, 108.0, 101.0, 106.0, 12_345.0);
        let body = f64::ln(106.0 / 102.0);
        let log_range = f64::ln(108.0 / 101.0);
        let expected = [
            f64::ln(102.0 / 100.0) as f32,
            body as f32,
            f64::ln(108.0 / 106.0) as f32,
            f64::ln(102.0 / 101.0) as f32,
            (0.5 * log_range * log_range - (2.0 * std::f64::consts::LN_2 - 1.0) * body * body)
                .max(0.0)
                .sqrt() as f32,
            f64::ln(12_345.0 / 10_000.0) as f32,
            1.0,
        ];
        assert_features(transition_bar_features(&previous, &current), expected);
    }

    #[test]
    fn flat_bar_has_zero_price_features_and_present_zero_volume_delta() {
        let previous = bar(42.0, 42.0, 42.0, 42.0, 1.0);
        let current = bar(42.0, 42.0, 42.0, 42.0, 1.0);
        assert_features(
            transition_bar_features(&previous, &current),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn overnight_gap_is_separate_from_current_bar_shape() {
        let previous = bar(98.0, 101.0, 97.0, 100.0, 20.0);
        let current = bar(110.0, 110.0, 110.0, 110.0, 20.0);
        assert_features(
            transition_bar_features(&previous, &current),
            [f64::ln(1.1) as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn malformed_high_low_are_repaired_to_contain_open_and_close() {
        let previous = bar(100.0, 100.0, 100.0, 104.0, 5.0);
        let current = bar(104.0, 101.0, 108.0, 106.0, 10.0);
        let body = f64::ln(106.0 / 104.0);
        let gk = (0.5 * body * body - (2.0 * std::f64::consts::LN_2 - 1.0) * body * body)
            .max(0.0)
            .sqrt() as f32;
        assert_features(
            transition_bar_features(&previous, &current),
            [0.0, body as f32, 0.0, 0.0, gk, f64::ln(2.0) as f32, 1.0],
        );
    }

    #[test]
    fn invalid_previous_close_zeros_only_gap() {
        let current = bar(100.0, 103.0, 99.0, 102.0, 12.0);
        let valid = transition_bar_features(&bar(90.0, 100.0, 89.0, 98.0, 4.0), &current);
        for invalid_close in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let actual =
                transition_bar_features(&bar(90.0, 100.0, 89.0, invalid_close, 4.0), &current);
            assert_eq!(actual[0], 0.0);
            assert_eq!(&actual[1..], &valid[1..]);
        }
    }

    #[test]
    fn missing_current_or_previous_volume_zeros_delta_and_presence() {
        let previous = bar(98.0, 101.0, 97.0, 100.0, 4.0);
        let current = bar(100.0, 103.0, 99.0, 102.0, 12.0);
        let expected_prices = transition_bar_features(&previous, &current);
        for missing in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let missing_current =
                transition_bar_features(&previous, &bar(100.0, 103.0, 99.0, 102.0, missing));
            assert_eq!(&missing_current[..5], &expected_prices[..5]);
            assert_eq!(&missing_current[5..], &[0.0, 0.0]);

            let missing_previous =
                transition_bar_features(&bar(98.0, 101.0, 97.0, 100.0, missing), &current);
            assert_eq!(&missing_previous[..5], &expected_prices[..5]);
            assert_eq!(&missing_previous[5..], &[0.0, 0.0]);
        }
    }

    #[test]
    fn only_gap_and_volume_delta_depend_on_the_predecessor() {
        let current = bar(100.0, 105.0, 98.0, 102.0, 50_000.0);
        let first = transition_bar_features(&bar(90.0, 101.0, 89.0, 99.0, 25_000.0), &current);
        let second = transition_bar_features(&bar(900.0, 1_010.0, 890.0, 90.0, 10_000.0), &current);
        assert_ne!(first[0], second[0]);
        assert_eq!(&first[1..5], &second[1..5]);
        assert_ne!(first[5], second[5]);
        assert_eq!(first[6], second[6]);
    }

    #[test]
    fn sampler_positions_align_token_emission_and_future_h1() {
        let rows = MAX_CONTEXT_BARS + 2;
        let raw = Tensor::arange(rows * 5, (Kind::Float, Device::Cpu)).view([1, rows, 5]);
        let (current, future) = split_raw_dof(&raw, 2);
        assert_eq!(current.size(), [1, MAX_CONTEXT_BARS, 5]);
        assert_eq!(future.size(), [1, 2, 5]);
        assert_eq!(
            Vec::<f32>::try_from(current.narrow(1, 0, 1).view([-1])).unwrap(),
            Vec::<f32>::try_from(raw.narrow(1, 0, 1).view([-1])).unwrap()
        );
        assert_eq!(
            Vec::<f32>::try_from(future.narrow(1, 0, 1).view([-1])).unwrap(),
            Vec::<f32>::try_from(raw.narrow(1, MAX_CONTEXT_BARS, 1).view([-1])).unwrap()
        );
    }

    #[test]
    fn invalid_current_prices_zero_price_features_but_preserve_finite_volume_transition() {
        let previous = bar(1.0, 2.0, 0.5, 1.5, 3.0);
        let current = bar(f32::INFINITY, f32::NAN, -1.0, f32::NAN, 9.0);
        let actual = transition_bar_features(&previous, &current);
        assert_eq!(&actual[..5], &[0.0; 5]);
        assert_eq!(&actual[5..], &[f64::ln(3.0) as f32, 1.0]);
        assert!(actual.into_iter().all(f32::is_finite));
    }
}
