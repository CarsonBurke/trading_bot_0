//! Sparse chart series and decimated optimizer-step accumulation.
//!
//! This module owns record-axis padding and mean aggregation. Report assembly only consumes
//! named accumulated fields and never reimplements missing-value or decimation semantics.

use shared::report::ReportSeries;

use super::StepMetrics;
use crate::torch::bar_dist::{BAR_DOF, BAR_DOF_NAMES};
use crate::torch::dataset::DIRECT_RETURN_HORIZONS;

/// A sparse curve on the shared record-tick axis.
#[derive(Clone, Debug, Default)]
pub(super) struct Series(Vec<f32>);

impl Series {
    pub(super) fn set(&mut self, tick: usize, value: f64) {
        if !value.is_finite() {
            return;
        }
        if self.0.len() <= tick {
            self.0.resize(tick + 1, f32::NAN);
        }
        self.0[tick] = value as f32;
    }

    fn padded(&self, len: usize) -> Vec<f32> {
        let mut values = self.0.clone();
        values.resize(len, f32::NAN);
        values
    }

    pub(super) fn measured(&self) -> bool {
        self.0.iter().any(|value| value.is_finite())
    }

    /// Label the curve, explicitly marking a series that never received a finite value.
    pub(super) fn labeled(&self, label: &str, len: usize) -> ReportSeries {
        let measured = self.measured();
        ReportSeries {
            label: if measured {
                label.to_owned()
            } else {
                format!("{label} (NOT MEASURED)")
            },
            values: self.padded(len),
        }
    }
}

fn dof_label(name: &str, role: &str) -> String {
    if role.is_empty() {
        name.to_owned()
    } else {
        format!("{name} {role}")
    }
}

/// Assemble one consistently ordered family of per-DOF sparse curves.
pub(super) fn labeled_dof(series: &[Series; BAR_DOF], role: &str, len: usize) -> Vec<ReportSeries> {
    BAR_DOF_NAMES
        .iter()
        .zip(series)
        .map(|(name, values)| values.labeled(&dof_label(name, role), len))
        .collect()
}

/// Assemble two per-DOF families as adjacent pairs while preserving the serialized series order.
pub(super) fn labeled_dof_pairs(
    first: &[Series; BAR_DOF],
    first_role: &str,
    second: &[Series; BAR_DOF],
    second_role: &str,
    len: usize,
) -> Vec<ReportSeries> {
    let mut paired = Vec::with_capacity(2 * BAR_DOF);
    for ((name, first), second) in BAR_DOF_NAMES.iter().zip(first).zip(second) {
        paired.push(first.labeled(&dof_label(name, first_role), len));
        paired.push(second.labeled(&dof_label(name, second_role), len));
    }
    paired
}
/// Running mean that ignores non-finite contributions.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Mean {
    sum: f64,
    count: usize,
}

impl Mean {
    pub(super) fn push(&mut self, value: f64) {
        if value.is_finite() {
            self.sum += value;
            self.count += 1;
        }
    }

    pub(super) fn value(self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.sum / self.count as f64
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct StepAccumulator {
    pub(super) steps: usize,
    pub(super) nll_bar: Mean,
    pub(super) nll_dof: [Mean; BAR_DOF],
    pub(super) direct_return_nll_horizon: [Mean; DIRECT_RETURN_HORIZONS.len()],
    pub(super) direct_return_valid_horizon: [Mean; DIRECT_RETURN_HORIZONS.len()],
    pub(super) direct_return_nll_mean: Mean,
    pub(super) dyn_loss: Mean,
    pub(super) kl_loss: Mean,
    pub(super) total_loss: Mean,
    pub(super) nll_share: Mean,
    pub(super) dyn_share: Mean,
    pub(super) kl_share: Mean,
    pub(super) growth_loss: Mean,
    pub(super) growth_share: Mean,
    pub(super) growth_abs_f: Mean,
    pub(super) growth_clamp_bind: Mean,
    pub(super) belief_autocorr: Mean,
    pub(super) dyn_vs_identity: Mean,
    pub(super) lr_mult: Mean,
    pub(super) muon_momentum: Mean,
    pub(super) grad_norm: Mean,
    pub(super) sdlr_alpha_mean: Mean,
    pub(super) sdlr_alpha_std: Mean,
    pub(super) sdlr_alpha_min: Mean,
    pub(super) sdlr_alpha_max: Mean,
    pub(super) sdlr_alpha_bound_fraction: Mean,
    pub(super) sdlr_evidence_mean: Mean,
    pub(super) sdlr_evidence_std: Mean,
    pub(super) sdlr_objective: Mean,
    pub(super) sdlr_update_magnitude: Mean,
    pub(super) smd_gain_mean: Mean,
    pub(super) smd_gain_std: Mean,
    pub(super) smd_gain_min: Mean,
    pub(super) smd_gain_max: Mean,
    pub(super) smd_gain_bound_fraction: Mean,
    pub(super) smd_credit_mean: Mean,
    pub(super) smd_credit_std: Mean,
    pub(super) smd_beta_update_abs_mean: Mean,
    pub(super) smd_trace_rms: Mean,
    pub(super) smd_hv_gradient_rms_ratio: Mean,
    pub(super) context: Mean,
    pub(super) batch_size: Mean,
    pub(super) bars_seen: u64,
    pub(super) free_vram_gib: Mean,
    pub(super) bar_tokens: Mean,
    pub(super) projected_footprint_gib: Mean,
    pub(super) capacity_ceiling_gib: Mean,
    pub(super) market_missing_bars: u64,
    pub(super) market_total_bars: u64,
    pub(super) adjusted_daily_missing_bars: u64,
    pub(super) adjusted_daily_total_bars: u64,
}

impl StepAccumulator {
    pub(super) fn push(&mut self, step: &StepMetrics) {
        self.steps += 1;
        self.nll_bar.push(step.nll_bar);
        for (slot, &value) in self.nll_dof.iter_mut().zip(step.nll_dof.iter()) {
            slot.push(value);
        }
        for (slot, &value) in self
            .direct_return_nll_horizon
            .iter_mut()
            .zip(step.direct_return_nll_horizon.iter())
        {
            slot.push(value);
        }
        for (slot, &value) in self
            .direct_return_valid_horizon
            .iter_mut()
            .zip(step.direct_return_valid_horizon.iter())
        {
            slot.push(value);
        }
        self.direct_return_nll_mean
            .push(step.direct_return_nll_mean);
        self.dyn_loss.push(step.dyn_loss);
        self.kl_loss.push(step.kl_loss);
        self.total_loss.push(step.total_loss);
        self.nll_share.push(step.nll_share);
        self.dyn_share.push(step.dyn_share);
        self.kl_share.push(step.kl_share);
        self.growth_loss.push(step.growth_loss);
        self.growth_share.push(step.growth_share);
        self.growth_abs_f.push(step.growth_abs_f);
        self.growth_clamp_bind.push(step.growth_clamp_bind);
        self.belief_autocorr.push(step.belief_autocorr);
        self.dyn_vs_identity.push(step.dyn_vs_identity);
        self.lr_mult.push(step.lr_mult);
        self.muon_momentum.push(step.muon_momentum);
        self.grad_norm.push(step.grad_norm);
        self.sdlr_alpha_mean.push(step.sdlr_alpha_mean);
        self.sdlr_alpha_std.push(step.sdlr_alpha_std);
        self.sdlr_alpha_min.push(step.sdlr_alpha_min);
        self.sdlr_alpha_max.push(step.sdlr_alpha_max);
        self.sdlr_alpha_bound_fraction
            .push(step.sdlr_alpha_bound_fraction);
        self.sdlr_evidence_mean.push(step.sdlr_evidence_mean);
        self.sdlr_evidence_std.push(step.sdlr_evidence_std);
        self.sdlr_objective.push(step.sdlr_objective);
        self.sdlr_update_magnitude.push(step.sdlr_update_magnitude);
        self.smd_gain_mean.push(step.smd_gain_mean);
        self.smd_gain_std.push(step.smd_gain_std);
        self.smd_gain_min.push(step.smd_gain_min);
        self.smd_gain_max.push(step.smd_gain_max);
        self.smd_gain_bound_fraction
            .push(step.smd_gain_bound_fraction);
        self.smd_credit_mean.push(step.smd_credit_mean);
        self.smd_credit_std.push(step.smd_credit_std);
        self.smd_beta_update_abs_mean
            .push(step.smd_beta_update_abs_mean);
        self.smd_trace_rms.push(step.smd_trace_rms);
        self.smd_hv_gradient_rms_ratio
            .push(step.smd_hv_gradient_rms_ratio);
        self.context.push(step.context as f64);
        self.batch_size.push(step.batch_size as f64);
        self.bars_seen = self.bars_seen.max(step.bars_seen);
        self.free_vram_gib.push(step.free_vram_gib);
        self.bar_tokens.push(step.bar_tokens);
        self.projected_footprint_gib
            .push(step.projected_footprint_gib);
        self.capacity_ceiling_gib.push(step.capacity_ceiling_gib);
        self.market_missing_bars += step.market_missing_bars;
        self.market_total_bars += step.market_total_bars;
        self.adjusted_daily_missing_bars += step.adjusted_daily_missing_bars;
        self.adjusted_daily_total_bars += step.adjusted_daily_total_bars;
    }
}

#[cfg(test)]
mod tests {
    use super::{Mean, Series};

    #[test]
    fn series_nan_pads_skipped_ticks() {
        let mut series = Series::default();
        series.set(0, 1.0);
        series.set(3, 4.0);
        series.set(4, f64::NAN);
        let values = series.padded(6);
        assert_eq!(values[0], 1.0);
        assert!(values[1].is_nan() && values[2].is_nan());
        assert_eq!(values[3], 4.0);
        assert!(values[4].is_nan() && values[5].is_nan());
    }

    #[test]
    fn mean_ignores_non_finite_samples() {
        let mut mean = Mean::default();
        mean.push(f64::NAN);
        mean.push(2.0);
        mean.push(4.0);
        mean.push(f64::INFINITY);
        assert_eq!(mean.value(), 3.0);
        assert!(Mean::default().value().is_nan());
    }
}
