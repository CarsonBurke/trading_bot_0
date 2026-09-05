use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::torch::lejepa::checkpoint::{CheckpointProvenance, WeightReadout};
use anyhow::{ensure, Context, Result};
use shared::report::{read_report, write_report, Report, ReportKind, ReportSeries, ScaleKind};

pub const MSE_JEPA_REPORT_BASES: &[&str] = &[
    "mse_jepa_loss",
    "mse_jepa_objective",
    "mse_jepa_representation",
    "mse_jepa_optimization",
    "mse_jepa_tail_ema",
    "mse_jepa_emission",
    "mse_jepa_flow",
    "mse_jepa_flow_samples",
    "mse_jepa_posthoc_readout",
    "mse_jepa_rollout_nll",
    "mse_jepa_rollout_calibration",
];

const SOURCE_MSE_JEPA_REPORT_BASES: [&str; 8] = [
    "mse_jepa_loss",
    "mse_jepa_objective",
    "mse_jepa_representation",
    "mse_jepa_optimization",
    "mse_jepa_tail_ema",
    "mse_jepa_emission",
    "mse_jepa_flow",
    "mse_jepa_flow_samples",
];

const DOF_NAMES: [&str; 5] = ["r", "s", "u", "v", "w"];
const FLOW_TAU_QUARTILES: [&str; 4] = ["[0, 0.25)", "[0.25, 0.5)", "[0.5, 0.75)", "[0.75, 1)"];

pub(super) struct SourceMseJepaReports {
    reports: Vec<(&'static str, Vec<u8>)>,
}

#[derive(Clone, Copy, Debug)]
pub struct StepMetrics {
    pub total_loss: f64,
    pub emission_ce: f64,
    pub emission_hard_nll: f64,
    pub flow: f64,
    pub flow_quartiles: [f64; 4],
    pub persistence_mse: f64,
    pub sigreg: f64,
    pub representation_std: f64,
    pub target_std: f64,
    pub grad_norm: f64,
    pub emission_ce_dof: [f64; 5],
    pub training_pass: f64,
    pub learning_rate_multiplier: f64,
    pub muon_learning_rate: f64,
    pub adamw_learning_rate: f64,
    pub muon_momentum: f64,
    pub completed_step_seconds: f64,
    pub windows_per_second: f64,
    pub sigreg_samples_per_second: f64,
    pub predicted_latent_positions_per_second: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ValidationMetrics {
    pub total_loss: f64,
    pub emission_ce: f64,
    pub emission_hard_nll: f64,
    pub flow: f64,
    pub flow_quartiles: [f64; 4],
    pub persistence_mse: f64,
    pub sigreg: f64,
    pub emission_ce_dof: [f64; 5],
    pub sample_spread: f64,
    pub sample_distance: f64,
    pub calibration_ratio: f64,
    pub energy_score: f64,
    pub sample_std: f64,
    pub prediction_mse: f64,
    pub skill_vs_persistence: f64,
    pub skill_vs_mean: f64,
}

pub(super) struct PosthocReadoutReport<'a> {
    pub emission_train_smoothed: &'a [f64],
    pub emission_train_hard: &'a [f64],
    pub token_train_smoothed: &'a [f64],
    pub token_train_hard: &'a [f64],
    pub emission_validation_smoothed: f64,
    pub emission_validation_hard: f64,
    pub token_validation_smoothed: f64,
    pub token_validation_hard: f64,
    pub seed: u64,
    pub steps: usize,
    pub batch_size: usize,
    pub token_rows_per_step: usize,
    pub validation_windows: usize,
    pub source_checkpoint_sha256: &'a str,
    pub source_lineage_sha256: &'a str,
    pub source_kind: &'a str,
    pub emission_gradient_mode: &'a str,
    pub optimizer_recipe_id: &'a str,
}

#[derive(Clone, Debug)]
pub(super) struct RolloutCoverage {
    pub step_50: Vec<f64>,
    pub step_90: Vec<f64>,
    pub cumulative_50: Vec<f64>,
    pub cumulative_90: Vec<f64>,
}

/// Step-level-only coverage: the latent rollout mode reports a bar fan whose inter-step
/// dependence the model does not define (each step's bar is drawn from its own belief and
/// never fed back), so a cumulative-return band over it would measure a process nobody
/// samples. Only the genuine ancestral bar mode carries cumulative coverage.
#[derive(Clone, Debug)]
pub(super) struct RolloutStepCoverage {
    pub step_50: Vec<f64>,
    pub step_90: Vec<f64>,
}

pub(super) struct RolloutReport<'a> {
    pub bar_nll: &'a [f64],
    pub latent_nll: &'a [f64],
    /// Convention-matched LLM `BarWorldModel` fan NLL on the same windows and horizons.
    pub baseline_nll: Option<&'a [f64]>,
    pub marginal_nll: f64,
    pub bar_coverage: RolloutCoverage,
    pub latent_coverage: RolloutStepCoverage,
    pub selected_horizons: &'a [usize],
    pub samples: usize,
    pub validation_windows: usize,
    pub window_chunk: usize,
    pub flow_steps: usize,
    pub seed: u64,
    pub checkpoint_sha256: &'a str,
    pub checkpoint_lineage_sha256: &'a str,
    pub emission_gradient_mode: &'a str,
    pub source_kind: &'a str,
    pub corpus_fingerprint: &'a str,
    pub split_bounds: (i64, i64),
    pub split_bounds_pinned: bool,
    pub supports_scoring_sha256: &'a str,
    pub core_train_seed: u64,
    pub core_batch_size: u64,
}

pub struct MseJepaReporter {
    gens: PathBuf,
    marginal_hard_nll: f64,
    train: Vec<StepMetrics>,
    validation: Vec<Option<ValidationMetrics>>,
    tail_ema_validation: Vec<Option<ValidationMetrics>>,
}

impl MseJepaReporter {
    pub fn new(gens: impl AsRef<Path>, marginal_hard_nll: f64) -> Self {
        Self {
            gens: gens.as_ref().to_path_buf(),
            marginal_hard_nll,
            train: Vec::new(),
            validation: Vec::new(),
            tail_ema_validation: Vec::new(),
        }
    }

    pub fn record_step(&mut self, metrics: StepMetrics) {
        self.train.push(metrics);
        self.validation.push(None);
        self.tail_ema_validation.push(None);
    }

    pub fn record_validation(&mut self, metrics: ValidationMetrics) {
        if let Some(slot) = self.validation.last_mut() {
            *slot = Some(metrics);
        }
    }

    pub fn record_tail_ema_validation(&mut self, metrics: ValidationMetrics) {
        if let Some(slot) = self.tail_ema_validation.last_mut() {
            *slot = Some(metrics);
        }
    }

    pub fn write(&self, epoch: usize) -> Result<()> {
        let output = self.gens.join(epoch.to_string());
        std::fs::create_dir_all(&output)
            .with_context(|| format!("failed creating {}", output.display()))?;
        let train = |pick: fn(&StepMetrics) -> f64| {
            self.train
                .iter()
                .map(|metrics| pick(metrics) as f32)
                .collect::<Vec<_>>()
        };
        let validation = |pick: fn(&ValidationMetrics) -> f64| {
            self.validation
                .iter()
                .map(|metrics| metrics.map_or(f32::NAN, |metrics| pick(&metrics) as f32))
                .collect::<Vec<_>>()
        };
        let paired_validation = |pick: fn(&ValidationMetrics) -> f64| {
            let mut raw = Vec::with_capacity(self.validation.len());
            let mut tail_ema = Vec::with_capacity(self.validation.len());
            let mut delta = Vec::with_capacity(self.validation.len());
            for (raw_metrics, tail_ema_metrics) in
                self.validation.iter().zip(&self.tail_ema_validation)
            {
                if let (Some(raw_metrics), Some(tail_ema_metrics)) = (raw_metrics, tail_ema_metrics)
                {
                    let raw_value = pick(raw_metrics) as f32;
                    let tail_ema_value = pick(tail_ema_metrics) as f32;
                    raw.push(raw_value);
                    tail_ema.push(tail_ema_value);
                    delta.push(tail_ema_value - raw_value);
                } else {
                    raw.push(f32::NAN);
                    tail_ema.push(f32::NAN);
                    delta.push(f32::NAN);
                }
            }
            (raw, tail_ema, delta)
        };
        self.write_multiline(
            &output,
            "mse_jepa_loss",
            "MSE-JEPA DBWM Core Loss",
            vec![
                series("train core objective", &train(|m| m.total_loss)),
                series("validation core objective", &validation(|m| m.total_loss)),
            ],
        )?;
        self.write_multiline(
            &output,
            "mse_jepa_objective",
            "MSE-JEPA DBWM Objective Terms",
            vec![
                series("train emission smoothed CE", &train(|m| m.emission_ce)),
                series(
                    "validation emission smoothed CE",
                    &validation(|m| m.emission_ce),
                ),
                series("train latent flow-matching loss", &train(|m| m.flow)),
                series(
                    "validation latent flow-matching loss",
                    &validation(|m| m.flow),
                ),
                series("train SIGReg actual-N=128 statistic", &train(|m| m.sigreg)),
                series(
                    "validation SIGReg fixed-direction statistic",
                    &validation(|m| m.sigreg),
                ),
                series(
                    "train latent persistence MSE",
                    &train(|m| m.persistence_mse),
                ),
                series(
                    "validation latent persistence MSE",
                    &validation(|m| m.persistence_mse),
                ),
                series(
                    "validation Heun sample-mean next-latent MSE",
                    &validation(|m| m.prediction_mse),
                ),
                series(
                    "validation sample-mean skill vs latent persistence",
                    &validation(|m| m.skill_vs_persistence),
                ),
                series(
                    "validation sample-mean skill vs batch mean",
                    &validation(|m| m.skill_vs_mean),
                ),
            ],
        )?;
        let mut emission = vec![
            series("train smoothed CE, full bar", &train(|m| m.emission_ce)),
            series(
                "validation smoothed CE, full bar",
                &validation(|m| m.emission_ce),
            ),
            series(
                "train hard categorical NLL, full bar",
                &train(|m| m.emission_hard_nll),
            ),
            series(
                "validation hard categorical NLL, full bar",
                &validation(|m| m.emission_hard_nll),
            ),
            series(
                "train marginal hard NLL reference",
                &vec![self.marginal_hard_nll as f32; self.train.len()],
            ),
        ];
        for (dof, name) in DOF_NAMES.iter().enumerate() {
            emission.push(ReportSeries {
                label: format!("train {name} smoothed CE"),
                values: self
                    .train
                    .iter()
                    .map(|metrics| metrics.emission_ce_dof[dof] as f32)
                    .collect(),
            });
            emission.push(ReportSeries {
                label: format!("validation {name} smoothed CE"),
                values: self
                    .validation
                    .iter()
                    .map(|metrics| {
                        metrics.map_or(f32::NAN, |metrics| metrics.emission_ce_dof[dof] as f32)
                    })
                    .collect(),
            });
        }
        self.write_multiline(
            &output,
            "mse_jepa_emission",
            "MSE-JEPA DBWM Belief-Conditioned Bar Emission",
            emission,
        )?;
        let mut flow = vec![
            series("train flow-matching loss", &train(|m| m.flow)),
            series("validation flow-matching loss", &validation(|m| m.flow)),
        ];
        for (quartile, range) in FLOW_TAU_QUARTILES.iter().enumerate() {
            flow.push(ReportSeries {
                label: format!("train flow-matching loss, tau in {range}"),
                values: self
                    .train
                    .iter()
                    .map(|metrics| metrics.flow_quartiles[quartile] as f32)
                    .collect(),
            });
            flow.push(ReportSeries {
                label: format!("validation flow-matching loss, tau in {range}"),
                values: self
                    .validation
                    .iter()
                    .map(|metrics| {
                        metrics.map_or(f32::NAN, |metrics| metrics.flow_quartiles[quartile] as f32)
                    })
                    .collect(),
            });
        }
        self.write_multiline(
            &output,
            "mse_jepa_flow",
            "MSE-JEPA DBWM Conditional Flow Matching",
            flow,
        )?;
        self.write_multiline(
            &output,
            "mse_jepa_flow_samples",
            "MSE-JEPA DBWM Heun Sample Diagnostics",
            vec![
                series(
                    "validation sample spread E||z^ - z^'||",
                    &validation(|m| m.sample_spread),
                ),
                series(
                    "validation sample distance E||z^ - z||",
                    &validation(|m| m.sample_distance),
                ),
                series(
                    "validation calibration ratio (1.0 nominal)",
                    &validation(|m| m.calibration_ratio),
                ),
                series(
                    &format!(
                        "validation energy score (m={})",
                        super::VALIDATION_SAMPLE_DRAWS
                    ),
                    &validation(|m| m.energy_score),
                ),
                series("validation sample std", &validation(|m| m.sample_std)),
            ],
        )?;
        self.write_multiline(
            &output,
            "mse_jepa_representation",
            "MSE-JEPA DBWM Representation Scale",
            vec![
                series("representation std", &train(|m| m.representation_std)),
                series("target std", &train(|m| m.target_std)),
            ],
        )?;
        self.write_multiline(
            &output,
            "mse_jepa_optimization",
            "MSE-JEPA DBWM Optimization",
            vec![
                series("gradient norm", &train(|m| m.grad_norm)),
                series("training pass", &train(|m| m.training_pass)),
                series(
                    "learning-rate multiplier",
                    &train(|m| m.learning_rate_multiplier),
                ),
                series("NorMuon learning rate", &train(|m| m.muon_learning_rate)),
                series("AdamW learning rate", &train(|m| m.adamw_learning_rate)),
                series("NorMuon momentum", &train(|m| m.muon_momentum)),
                series(
                    "completed-step seconds",
                    &train(|m| m.completed_step_seconds),
                ),
                series(
                    "prediction windows/second",
                    &train(|m| m.windows_per_second),
                ),
                series(
                    "SIGReg independent samples/second",
                    &train(|m| m.sigreg_samples_per_second),
                ),
                series(
                    "predicted latent positions/second",
                    &train(|m| m.predicted_latent_positions_per_second),
                ),
            ],
        )?;
        let total_loss = paired_validation(|m| m.total_loss);
        let emission_ce = paired_validation(|m| m.emission_ce);
        let flow = paired_validation(|m| m.flow);
        let sigreg = paired_validation(|m| m.sigreg);
        self.write_multiline(
            &output,
            "mse_jepa_tail_ema",
            "MSE-JEPA DBWM Tail-EMA Validation",
            vec![
                series("raw validation total loss", &total_loss.0),
                series("tail-EMA validation total loss", &total_loss.1),
                series("tail-EMA minus raw validation total loss", &total_loss.2),
                series("raw validation emission smoothed CE", &emission_ce.0),
                series("tail-EMA validation emission smoothed CE", &emission_ce.1),
                series(
                    "tail-EMA minus raw validation emission smoothed CE",
                    &emission_ce.2,
                ),
                series("raw validation flow loss", &flow.0),
                series("tail-EMA validation flow loss", &flow.1),
                series("tail-EMA minus raw validation flow loss", &flow.2),
                series("raw validation SIGReg", &sigreg.0),
                series("tail-EMA validation SIGReg", &sigreg.1),
                series("tail-EMA minus raw validation SIGReg", &sigreg.2),
            ],
        )?;
        Ok(())
    }

    fn write_multiline(
        &self,
        output: &Path,
        base: &str,
        title: &str,
        series: Vec<ReportSeries>,
    ) -> Result<()> {
        write_registered(output, base, title, Some("optimizer step"), series)
    }
}

pub(super) fn preflight_source_reports(
    source_gens: &Path,
    completed_steps: usize,
) -> Result<SourceMseJepaReports> {
    ensure!(
        completed_steps > 0,
        "source MSE-JEPA checkpoint has no completed optimizer steps"
    );
    let source = fs::read_dir(source_gens)
        .with_context(|| format!("failed reading source reports {}", source_gens.display()))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let generation = entry.file_name().to_str()?.parse::<usize>().ok()?;
            entry
                .file_type()
                .ok()?
                .is_dir()
                .then_some((generation, entry.path()))
        })
        .max_by_key(|(generation, _)| *generation)
        .map(|(_, path)| path)
        .with_context(|| {
            format!(
                "source MSE-JEPA run has no numeric report generation under {}",
                source_gens.display()
            )
        })?;
    let mut reports = Vec::with_capacity(SOURCE_MSE_JEPA_REPORT_BASES.len());
    for base in SOURCE_MSE_JEPA_REPORT_BASES {
        let path = source.join(format!("{base}.report.bin"));
        let bytes = fs::read(&path)
            .with_context(|| format!("failed reading required source report {}", path.display()))?;
        let mut report = postcard::from_bytes::<Report>(&bytes).with_context(|| {
            format!("required source report {} failed to decode", path.display())
        })?;
        let ReportKind::MultiLine { series } = &mut report.kind else {
            anyhow::bail!(
                "required source report {} is not a step-aligned multiline report",
                path.display()
            );
        };

        ensure!(
            series
                .iter()
                .all(|line| line.values.len() >= completed_steps),
            "required source report {} does not reach checkpoint step {completed_steps}",
            path.display()
        );
        for line in series {
            line.values.truncate(completed_steps);
        }
        let bytes = postcard::to_stdvec(&report).with_context(|| {
            format!("failed encoding truncated source report {}", path.display())
        })?;
        reports.push((base, bytes));
    }
    Ok(SourceMseJepaReports { reports })
}
pub(super) fn preflight_posthoc_readout_report(
    source_gens: &Path,
    provenance: &CheckpointProvenance,
) -> Result<Vec<u8>> {
    let path = source_gens
        .join("0")
        .join("mse_jepa_posthoc_readout.report.bin");
    let bytes = fs::read(&path).with_context(|| {
        format!(
            "fitted endpoint is missing posthoc report {}",
            path.display()
        )
    })?;
    let report: Report = postcard::from_bytes(&bytes)
        .with_context(|| format!("posthoc readout report {} failed to decode", path.display()))?;
    ensure!(
        report.title == "MSE-JEPA Frozen-Backbone Posthoc Readouts"
            && report.x_label.as_deref() == Some("readout optimizer step"),
        "posthoc readout report title/axis contract mismatch"
    );
    let ReportKind::MultiLine { series } = report.kind else {
        anyhow::bail!("posthoc readout report is not multiline");
    };
    let fit = provenance
        .head_fit
        .as_ref()
        .context("fitted endpoint lacks posthoc fit provenance")?;
    let source_kind = match &provenance.weight_readout {
        WeightReadout::Raw => "raw",
        WeightReadout::TailEma { .. } => "tail_ema",
    };
    let expected_constants = posthoc_constants(
        fit.seed,
        fit.steps,
        fit.batch_size,
        fit.token_rows_per_step,
        fit.validation_windows,
        source_kind,
        &provenance.emission_gradient_mode.to_string(),
        &fit.optimizer_recipe_id,
        &fit.source_checkpoint_sha256,
        &fit.source_lineage_sha256,
    );
    let expected_label = format!("authenticated fit constants: {expected_constants}");
    ensure!(
        series.iter().any(|line| {
            line.label == "final pinned-validation belief-emission hard NLL"
                && line.values.len() == 1
                && line.values[0].is_finite()
        }),
        "posthoc readout report lacks a finite final held-out emission NLL"
    );
    ensure!(
        series.iter().any(|line| {
            line.label == expected_label
                && line.values.as_slice() == [fit.steps as f32]
        }),
        "posthoc readout report source hashes or canonical fit recipe disagree with checkpoint provenance"
    );
    Ok(bytes)
}

pub(super) fn inherit_posthoc_readout_report(gens: &Path, bytes: &[u8]) -> Result<()> {
    let output = gens.join("0");
    fs::create_dir_all(&output).with_context(|| format!("failed creating {}", output.display()))?;
    let destination = output.join("mse_jepa_posthoc_readout.report.bin");
    fs::write(&destination, bytes)
        .with_context(|| format!("failed inheriting {}", destination.display()))?;
    read_report(&destination).with_context(|| {
        format!(
            "inherited posthoc report {} failed to decode",
            destination.display()
        )
    })?;
    Ok(())
}

pub(super) fn write_posthoc_readout_report(
    gens: &Path,
    report: &PosthocReadoutReport<'_>,
) -> Result<()> {
    ensure!(
        report.emission_train_smoothed.len() == report.steps
            && report.emission_train_hard.len() == report.steps
            && report.token_train_smoothed.len() == report.steps
            && report.token_train_hard.len() == report.steps,
        "posthoc readout train curves must match the declared fit steps"
    );
    let output = gens.join("0");
    fs::create_dir_all(&output).with_context(|| format!("failed creating {}", output.display()))?;
    let constants = posthoc_constants(
        report.seed,
        report.steps as u64,
        report.batch_size as u64,
        report.token_rows_per_step as u64,
        report.validation_windows as u64,
        report.source_kind,
        report.emission_gradient_mode,
        report.optimizer_recipe_id,
        report.source_checkpoint_sha256,
        report.source_lineage_sha256,
    );
    let f32_curve = |values: &[f64]| values.iter().map(|value| *value as f32).collect::<Vec<_>>();
    write_registered(
        &output,
        "mse_jepa_posthoc_readout",
        "MSE-JEPA Frozen-Backbone Posthoc Readouts",
        Some("readout optimizer step"),
        vec![
            series(
                "train belief-emission smoothed NLL",
                &f32_curve(report.emission_train_smoothed),
            ),
            series(
                "train belief-emission hard NLL",
                &f32_curve(report.emission_train_hard),
            ),
            series(
                "train same-time token-probe smoothed NLL",
                &f32_curve(report.token_train_smoothed),
            ),
            series(
                "train same-time token-probe hard NLL",
                &f32_curve(report.token_train_hard),
            ),
            ReportSeries {
                label: "final pinned-validation belief-emission smoothed NLL".to_owned(),
                values: vec![report.emission_validation_smoothed as f32],
            },
            ReportSeries {
                label: "final pinned-validation belief-emission hard NLL".to_owned(),
                values: vec![report.emission_validation_hard as f32],
            },
            ReportSeries {
                label: "final pinned-validation same-time token-probe smoothed NLL".to_owned(),
                values: vec![report.token_validation_smoothed as f32],
            },
            ReportSeries {
                label: "final pinned-validation same-time token-probe hard NLL".to_owned(),
                values: vec![report.token_validation_hard as f32],
            },
            ReportSeries {
                label: format!("authenticated fit constants: {constants}"),
                values: vec![report.steps as f32],
            },
        ],
    )
}

fn posthoc_constants(
    seed: u64,
    steps: u64,
    batch_size: u64,
    token_rows_per_step: u64,
    validation_windows: u64,
    source_kind: &str,
    emission_gradient_mode: &str,
    optimizer_recipe_id: &str,
    source_checkpoint_sha256: &str,
    source_lineage_sha256: &str,
) -> String {
    format!(
        "seed={seed}; steps={steps}; batch_size={batch_size}; token_rows_per_step={token_rows_per_step}; validation_windows={validation_windows}; source_kind={source_kind}; emission_gradient={emission_gradient_mode}; optimizer={optimizer_recipe_id}; source_checkpoint_sha256={source_checkpoint_sha256}; source_lineage_sha256={source_lineage_sha256}"
    )
}

pub(super) fn inherit_source_reports(
    rollout_gens: &Path,
    source: &SourceMseJepaReports,
) -> Result<()> {
    ensure!(
        source.reports.len() == SOURCE_MSE_JEPA_REPORT_BASES.len(),
        "source MSE-JEPA report set is incomplete"
    );
    let output = rollout_gens.join("0");
    fs::create_dir_all(&output).with_context(|| format!("failed creating {}", output.display()))?;
    for (index, &(base, ref bytes)) in source.reports.iter().enumerate() {
        let destination = output.join(format!("{base}.report.bin"));
        let temporary = output.join(format!(
            ".{base}.report.bin.tmp-{}-{index}",
            std::process::id()
        ));
        let copied = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .with_context(|| format!("failed staging {}", destination.display()))?;
            file.write_all(bytes)
                .with_context(|| format!("failed staging {}", destination.display()))?;
            file.sync_all()
                .with_context(|| format!("failed syncing {}", destination.display()))?;
            drop(file);
            read_report(&temporary).with_context(|| {
                format!(
                    "staged source report for {} failed to decode",
                    destination.display()
                )
            })?;
            fs::rename(&temporary, &destination).with_context(|| {
                format!(
                    "failed atomically installing source report {}",
                    destination.display()
                )
            })
        })();
        if copied.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        copied?;
    }
    Ok(())
}

pub(super) fn write_rollout_reports(gens: &Path, report: &RolloutReport<'_>) -> Result<()> {
    validate_rollout_report(report)?;
    let output = gens.join("0");
    std::fs::create_dir_all(&output)
        .with_context(|| format!("failed creating {}", output.display()))?;
    let horizons = report.bar_nll.len();
    let all = |values: &[f64]| {
        std::iter::once(f32::NAN)
            .chain(values.iter().map(|value| *value as f32))
            .collect::<Vec<_>>()
    };
    let selected = |values: &[f64]| {
        std::iter::once(f32::NAN)
            .chain(values.iter().enumerate().map(|(index, value)| {
                if report.selected_horizons.binary_search(&(index + 1)).is_ok() {
                    *value as f32
                } else {
                    f32::NAN
                }
            }))
            .collect::<Vec<_>>()
    };
    let constant = |value: f64| vec![value as f32; horizons + 1];
    let identity_label = format!(
        "authenticated rollout constants: eval_seed={}; validation_windows={}; samples={}; flow_steps={}; window_chunk={}; horizons={:?}; source_kind={}; emission_gradient={}; core_train_seed={}; core_batch_size={}; checkpoint_sha256={}; checkpoint_lineage_sha256={}; split_start_ms={}; split_end_ms={}; split_bounds_pinned={}; corpus_fingerprint={}; supports_scoring_sha256={}",
        report.seed,
        report.validation_windows,
        report.samples,
        report.flow_steps,
        report.window_chunk,
        report.selected_horizons,
        report.source_kind,
        report.emission_gradient_mode,
        report.core_train_seed,
        report.core_batch_size,
        report.checkpoint_sha256,
        report.checkpoint_lineage_sha256,
        report.split_bounds.0,
        report.split_bounds.1,
        report.split_bounds_pinned,
        report.corpus_fingerprint,
        report.supports_scoring_sha256,
    );
    let identity_series = || ReportSeries {
        label: identity_label.clone(),
        values: constant(report.samples as f64),
    };
    let improvement = |values: &[f64]| {
        values
            .iter()
            .map(|value| report.marginal_nll - value)
            .collect::<Vec<_>>()
    };
    write_registered(
        &output,
        "mse_jepa_rollout_nll",
        "MSE-JEPA DBWM Ancestral Rollout Predictive NLL",
        Some("forecast horizon in bars"),
        vec![
            series(
                "bar-mode fan-marginalized NLL, every horizon",
                &all(report.bar_nll),
            ),
            series(
                "bar-mode fan-marginalized NLL, selected horizons",
                &selected(report.bar_nll),
            ),
            series(
                "latent-mode fan-marginalized NLL, every horizon",
                &all(report.latent_nll),
            ),
            series(
                "latent-mode fan-marginalized NLL, selected horizons",
                &selected(report.latent_nll),
            ),
            series(
                "train marginal NLL reference",
                &constant(report.marginal_nll),
            ),
            series(
                "bar-mode improvement vs train marginal, every horizon",
                &all(&improvement(report.bar_nll)),
            ),
            series(
                "latent-mode improvement vs train marginal, every horizon",
                &all(&improvement(report.latent_nll)),
            ),
            identity_series(),
        ]
        .into_iter()
        .chain(report.baseline_nll.iter().flat_map(|baseline| {
            let gap: Vec<f64> = report
                .bar_nll
                .iter()
                .zip(baseline.iter())
                .map(|(bar, llm)| bar - llm)
                .collect();
            [
                series(
                    "LLM baseline fan-marginalized NLL, every horizon",
                    &all(baseline),
                ),
                series(
                    "bar-mode minus LLM baseline NLL, every horizon (gate: below zero)",
                    &all(&gap),
                ),
            ]
        }))
        .collect(),
    )?;
    write_registered(
        &output,
        "mse_jepa_rollout_calibration",
        "MSE-JEPA DBWM Ancestral Rollout Return Coverage",
        Some("forecast horizon in bars"),
        vec![
            series(
                "bar-mode step-r 50% coverage",
                &all(&report.bar_coverage.step_50),
            ),
            series(
                "bar-mode step-r 90% coverage",
                &all(&report.bar_coverage.step_90),
            ),
            series(
                "bar-mode cumulative-r 50% coverage",
                &all(&report.bar_coverage.cumulative_50),
            ),
            series(
                "bar-mode cumulative-r 90% coverage",
                &all(&report.bar_coverage.cumulative_90),
            ),
            series(
                "latent-mode step-r 50% coverage",
                &all(&report.latent_coverage.step_50),
            ),
            series(
                "latent-mode step-r 90% coverage",
                &all(&report.latent_coverage.step_90),
            ),
            series("nominal 50% coverage", &constant(0.5)),
            series("nominal 90% coverage", &constant(0.9)),
            identity_series(),
        ],
    )
}

fn validate_rollout_report(report: &RolloutReport<'_>) -> Result<()> {
    let horizons = report.bar_nll.len();
    ensure!(horizons > 0, "rollout report needs at least one horizon");
    let bar_coverage = &report.bar_coverage;
    ensure!(
        report.latent_nll.len() == horizons
            && bar_coverage.step_50.len() == horizons
            && bar_coverage.step_90.len() == horizons
            && bar_coverage.cumulative_50.len() == horizons
            && bar_coverage.cumulative_90.len() == horizons
            && report.latent_coverage.step_50.len() == horizons
            && report.latent_coverage.step_90.len() == horizons,
        "rollout report bar/latent profiles are inconsistent"
    );
    ensure!(
        report
            .baseline_nll
            .is_none_or(|baseline| baseline.len() == horizons),
        "rollout LLM baseline profile must cover every horizon"
    );
    ensure!(
        !report.selected_horizons.is_empty()
            && report
                .selected_horizons
                .iter()
                .copied()
                .all(|horizon| (1..=horizons).contains(&horizon))
            && report
                .selected_horizons
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
        "rollout selected horizons must be strictly increasing and within the rollout"
    );
    ensure!(
        report.marginal_nll.is_finite(),
        "rollout marginal NLL reference must be finite"
    );
    ensure!(
        report.samples >= 2 && report.validation_windows > 0,
        "rollout report needs a real fan over real windows"
    );
    ensure!(
        report.core_batch_size > 0
            && report.window_chunk > 0
            && report.split_bounds.0 < report.split_bounds.1
            && !report.checkpoint_sha256.is_empty()
            && !report.checkpoint_lineage_sha256.is_empty()
            && !report.corpus_fingerprint.is_empty()
            && !report.supports_scoring_sha256.is_empty(),
        "rollout report authentication identity is incomplete"
    );
    Ok(())
}

fn write_registered(
    output: &Path,
    base: &str,
    title: &str,
    x_label: Option<&str>,
    series: Vec<ReportSeries>,
) -> Result<()> {
    ensure!(
        MSE_JEPA_REPORT_BASES.contains(&base),
        "unregistered MSE-JEPA report base {base}"
    );
    ensure!(
        shared::report::PRETRAIN_REPORT_BASES.contains(&base),
        "{base} is absent from the shared pretrain report registry"
    );
    write_report(
        output.join(format!("{base}.report.bin")),
        &Report {
            title: title.to_owned(),
            x_label: x_label.map(str::to_owned),
            y_label: None,
            scale: ScaleKind::Linear,
            kind: ReportKind::MultiLine { series },
        },
    )
    .with_context(|| format!("failed writing {base}.report.bin"))
}

fn series(label: &str, values: &[f32]) -> ReportSeries {
    ReportSeries {
        label: label.to_owned(),
        values: values.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        inherit_source_reports, preflight_posthoc_readout_report, preflight_source_reports,
        write_posthoc_readout_report, write_rollout_reports, MseJepaReporter, PosthocReadoutReport,
        RolloutCoverage, RolloutReport, RolloutStepCoverage, StepMetrics, ValidationMetrics,
        MSE_JEPA_REPORT_BASES, SOURCE_MSE_JEPA_REPORT_BASES,
    };
    use crate::torch::lejepa::checkpoint::{
        CheckpointProvenance, CoreInitialization, CoreTrainingOrigin, EmissionGradientMode,
        HeadFitProvenance, WeightReadout, CANONICAL_READOUT_FIT_BATCH_SIZE,
        CANONICAL_READOUT_FIT_SEED, CANONICAL_READOUT_FIT_STEPS,
        CANONICAL_READOUT_OPTIMIZER_RECIPE_ID, CANONICAL_READOUT_TOKEN_ROWS,
        CANONICAL_READOUT_VALIDATION_WINDOWS,
    };

    fn step_metrics() -> StepMetrics {
        StepMetrics {
            total_loss: 17.0,
            emission_ce: 14.5,
            emission_hard_nll: 16.0,
            flow: 1.4,
            flow_quartiles: [1.9, 1.5, 1.1, 0.8],
            persistence_mse: 1.2,
            sigreg: 10.0,
            representation_std: 1.1,
            target_std: 1.0,
            grad_norm: 0.4,
            emission_ce_dof: [3.2, 2.9, 2.8, 2.8, 2.8],
            training_pass: 1.5,
            learning_rate_multiplier: 0.75,
            muon_learning_rate: 5e-3,
            adamw_learning_rate: 3e-4,
            muon_momentum: 0.85,
            completed_step_seconds: 0.25,
            windows_per_second: 32.0,
            sigreg_samples_per_second: 512.0,
            predicted_latent_positions_per_second: 192_000.0,
        }
    }

    fn validation_metrics() -> ValidationMetrics {
        ValidationMetrics {
            total_loss: 16.5,
            emission_ce: 14.2,
            emission_hard_nll: 15.7,
            flow: 1.3,
            flow_quartiles: [1.7, 1.4, 1.0, 0.7],
            persistence_mse: 1.15,
            sigreg: 9.8,
            emission_ce_dof: [3.1, 2.8, 2.7, 2.7, 2.7],
            sample_spread: 1.0,
            sample_distance: 1.18,
            calibration_ratio: 0.85,
            energy_score: 0.68,
            sample_std: 0.95,
            prediction_mse: 0.75,
            skill_vs_persistence: 0.35,
            skill_vs_mean: 0.15,
        }
    }

    fn rollout_report<'a>(
        bar_nll: &'a [f64],
        latent_nll: &'a [f64],
        selected_horizons: &'a [usize],
    ) -> RolloutReport<'a> {
        RolloutReport {
            bar_nll,
            latent_nll,
            baseline_nll: None,
            marginal_nll: 21.13,
            bar_coverage: RolloutCoverage {
                step_50: bar_nll.iter().map(|_| 0.5).collect(),
                step_90: bar_nll.iter().map(|_| 0.9).collect(),
                cumulative_50: bar_nll.iter().map(|_| 0.45).collect(),
                cumulative_90: bar_nll.iter().map(|_| 0.88).collect(),
            },
            latent_coverage: RolloutStepCoverage {
                step_50: bar_nll.iter().map(|_| 0.52).collect(),
                step_90: bar_nll.iter().map(|_| 0.87).collect(),
            },
            selected_horizons,
            samples: 32,
            validation_windows: 64,
            window_chunk: 2,
            flow_steps: 8,
            seed: 0x5eed,
            checkpoint_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            checkpoint_lineage_sha256:
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            emission_gradient_mode: "attached",
            source_kind: "raw",
            corpus_fingerprint: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            split_bounds: (1_700_000_000_000, 1_710_000_000_000),
            split_bounds_pinned: true,
            supports_scoring_sha256:
                "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            core_train_seed: 0x5eed,
            core_batch_size: 8,
        }
    }

    #[test]
    fn writer_and_shared_registry_are_bidirectionally_complete() {
        let shared: Vec<&str> = shared::report::PRETRAIN_REPORT_BASES
            .iter()
            .copied()
            .filter(|base| base.starts_with("mse_jepa_"))
            .collect();
        assert_eq!(shared, MSE_JEPA_REPORT_BASES);

        let root = std::env::temp_dir().join(format!("mse-jepa-reports-{}", uuid::Uuid::new_v4()));
        let mut reporter = MseJepaReporter::new(&root, 21.13);
        reporter.record_step(step_metrics());
        reporter.record_validation(validation_metrics());
        reporter.record_step(step_metrics());
        reporter.record_validation(validation_metrics());
        let mut tail_ema = validation_metrics();
        tail_ema.total_loss = 15.5;
        tail_ema.emission_ce = 13.9;
        tail_ema.flow = 1.1;
        tail_ema.sigreg = 9.0;
        reporter.record_tail_ema_validation(tail_ema);
        reporter.record_step(step_metrics());
        reporter.write(0).unwrap();
        let mut with_baseline = rollout_report(&[16.2, 17.5, 18.9], &[16.4, 18.0, 19.6], &[1, 3]);
        with_baseline.baseline_nll = Some(&[15.9, 17.0, 18.2]);
        write_rollout_reports(&root, &with_baseline).unwrap();
        write_posthoc_readout_report(
            &root,
            &PosthocReadoutReport {
                emission_train_smoothed: &[14.0, 13.5],
                emission_train_hard: &[15.0, 14.5],
                token_train_smoothed: &[9.0, 8.5],
                token_train_hard: &[9.8, 9.2],
                emission_validation_smoothed: 13.8,
                emission_validation_hard: 15.0,
                token_validation_smoothed: 8.8,
                token_validation_hard: 9.4,
                seed: 17,
                steps: 2,
                batch_size: 8,
                token_rows_per_step: 4096,
                validation_windows: 8,
                source_checkpoint_sha256:
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                source_lineage_sha256:
                    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                source_kind: "raw",
                emission_gradient_mode: "attached",
                optimizer_recipe_id: "test-adamw",
            },
        )
        .unwrap();
        let output = root.join("0");
        for base in MSE_JEPA_REPORT_BASES {
            let report =
                shared::report::read_report(output.join(format!("{base}.report.bin"))).unwrap();
            let shared::report::ReportKind::MultiLine { series } = report.kind else {
                panic!("{base} is not multiline");
            };
            assert!(series
                .iter()
                .any(|line| line.values.iter().any(|value| value.is_finite())));
            if *base == "mse_jepa_optimization" {
                let labels = series
                    .iter()
                    .map(|line| line.label.as_str())
                    .collect::<Vec<_>>();
                assert!(labels.contains(&"completed-step seconds"));
                assert!(labels.contains(&"training pass"));
                assert!(labels.contains(&"learning-rate multiplier"));
                assert!(labels.contains(&"prediction windows/second"));
                assert!(labels.contains(&"SIGReg independent samples/second"));
                assert!(labels.contains(&"predicted latent positions/second"));
                for (label, expected) in
                    [("training pass", 1.5), ("learning-rate multiplier", 0.75)]
                {
                    let values = &series
                        .iter()
                        .find(|line| line.label == label)
                        .unwrap()
                        .values;
                    assert!(values.iter().all(|value| (*value - expected).abs() < 1e-6));
                }
            }
            if *base == "mse_jepa_flow" {
                let labels = series
                    .iter()
                    .map(|line| line.label.as_str())
                    .collect::<Vec<_>>();
                assert_eq!(labels.len(), 10);
                assert!(labels.contains(&"validation flow-matching loss"));
                for range in super::FLOW_TAU_QUARTILES {
                    for split in ["train", "validation"] {
                        assert!(labels.contains(
                            &format!("{split} flow-matching loss, tau in {range}").as_str()
                        ));
                    }
                }
            }
            if *base == "mse_jepa_flow_samples" {
                assert_eq!(series.len(), 5);
                for line in &series {
                    assert!(line.label.starts_with("validation"));
                    assert_eq!(line.values.len(), 3);
                    assert!(line.values[0].is_finite());
                    assert!(line.values[1].is_finite());
                    assert!(line.values[2].is_nan());
                }
            }
            if base.starts_with("mse_jepa_rollout") {
                for line in &series {
                    assert_eq!(line.values.len(), 4);
                    if !line.label.starts_with("nominal")
                        && !line.label.contains("reference")
                        && !line.label.starts_with("authenticated rollout constants")
                    {
                        assert!(line.values[0].is_nan());
                        assert!(line.values[1].is_finite());
                    }
                    assert!(
                        !line.label.contains("latent-mode cumulative"),
                        "latent-mode cumulative coverage is invalid by construction"
                    );
                }
            }
            if *base == "mse_jepa_rollout_nll" {
                let gate = series
                    .iter()
                    .find(|line| line.label.contains("minus LLM baseline"))
                    .expect("the design gate must be readable from the chart");
                assert!((gate.values[1] - (16.2 - 15.9) as f32).abs() < 1e-6);
            }
            if *base == "mse_jepa_tail_ema" {
                assert_eq!(series.len(), 12);
                for line in &series {
                    assert_eq!(line.values.len(), 3);
                    assert!(line.values[0].is_nan());
                    assert!(line.values[1].is_finite());
                    assert!(line.values[2].is_nan());
                }
                for (label, raw, tail_ema, delta) in [
                    ("total loss", 16.5, 15.5, -1.0),
                    ("emission smoothed CE", 14.2, 13.9, -0.3),
                    ("flow loss", 1.3, 1.1, -0.2),
                    ("SIGReg", 9.8, 9.0, -0.8),
                ] {
                    let value = |prefix: &str| {
                        series
                            .iter()
                            .find(|line| line.label == format!("{prefix} {label}"))
                            .unwrap()
                            .values[1]
                    };
                    assert!((value("raw validation") - raw).abs() < 1e-5);
                    assert!((value("tail-EMA validation") - tail_ema).abs() < 1e-5);
                    assert!((value("tail-EMA minus raw validation") - delta).abs() < 1e-5);
                }
            }
        }
        let emission =
            shared::report::read_report(output.join("mse_jepa_emission.report.bin")).unwrap();
        let shared::report::ReportKind::MultiLine { series } = emission.kind else {
            panic!("mse_jepa_emission is not multiline");
        };
        let labels = series
            .iter()
            .map(|line| line.label.as_str())
            .collect::<Vec<_>>();
        for name in ["r", "s", "u", "v", "w"] {
            assert!(labels.contains(&format!("train {name} smoothed CE").as_str()));
            assert!(labels.contains(&format!("validation {name} smoothed CE").as_str()));
        }
        let validation_ce = series
            .iter()
            .find(|line| line.label == "validation smoothed CE, full bar")
            .unwrap();
        assert!(validation_ce.values[0].is_finite());
        assert!(validation_ce.values[1].is_finite());
        assert!(validation_ce.values[2].is_nan());

        assert!(
            write_rollout_reports(&root, &rollout_report(&[16.2, 17.5], &[16.4], &[1]),).is_err()
        );
        assert!(write_rollout_reports(
            &root,
            &rollout_report(&[16.2, 17.5], &[16.4, 18.0], &[2, 1]),
        )
        .is_err());
        assert!(
            write_rollout_reports(&root, &rollout_report(&[16.2, 17.5], &[16.4, 18.0], &[3]),)
                .is_err()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn source_reports_are_required_decoded_and_inherited_byte_for_byte() {
        let expected_source = MSE_JEPA_REPORT_BASES
            .iter()
            .copied()
            .filter(|base| {
                !base.starts_with("mse_jepa_rollout_") && *base != "mse_jepa_posthoc_readout"
            })
            .collect::<Vec<_>>();
        assert_eq!(
            expected_source.as_slice(),
            SOURCE_MSE_JEPA_REPORT_BASES.as_slice()
        );
        let root = std::env::temp_dir().join(format!("mse-jepa-inherit-{}", uuid::Uuid::new_v4()));
        let source = root.join("source");
        let destination = root.join("destination");
        let mut reporter = MseJepaReporter::new(&source, 21.13);
        reporter.record_step(step_metrics());
        reporter.record_validation(validation_metrics());
        reporter.write(0).unwrap();
        let inherited = preflight_source_reports(&source, 1).unwrap();
        inherit_source_reports(&destination, &inherited).unwrap();
        for base in SOURCE_MSE_JEPA_REPORT_BASES {
            let name = format!("0/{base}.report.bin");
            assert_eq!(
                std::fs::read(source.join(&name)).unwrap(),
                std::fs::read(destination.join(&name)).unwrap()
            );
            shared::report::read_report(destination.join(name)).unwrap();
        }
        reporter.record_step(step_metrics());
        reporter.write(1).unwrap();
        let truncated = preflight_source_reports(&source, 1).unwrap();
        for (base, bytes) in &truncated.reports {
            assert_eq!(
                bytes,
                &std::fs::read(source.join(format!("0/{base}.report.bin"))).unwrap(),
                "latest generation must truncate exactly to the checkpoint step"
            );
        }
        let complete = preflight_source_reports(&source, 2).unwrap();
        for (base, bytes) in &complete.reports {
            let report: shared::report::Report = postcard::from_bytes(bytes).unwrap();
            let shared::report::ReportKind::MultiLine { series } = report.kind else {
                panic!("{base} is not multiline");
            };
            assert!(
                series.iter().all(|line| line.values.len() == 2),
                "{base} was not aligned to the checkpoint step"
            );
        }
        std::fs::remove_file(source.join("1/mse_jepa_tail_ema.report.bin")).unwrap();
        assert!(preflight_source_reports(&source, 2).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn posthoc_report_source_hashes_and_recipe_must_match_checkpoint() {
        let root =
            std::env::temp_dir().join(format!("mse-jepa-fit-report-{}", uuid::Uuid::new_v4()));
        let values = vec![1.0; CANONICAL_READOUT_FIT_STEPS as usize];
        write_posthoc_readout_report(
            &root,
            &PosthocReadoutReport {
                emission_train_smoothed: &values,
                emission_train_hard: &values,
                token_train_smoothed: &values,
                token_train_hard: &values,
                emission_validation_smoothed: 1.0,
                emission_validation_hard: 1.0,
                token_validation_smoothed: 1.0,
                token_validation_hard: 1.0,
                seed: CANONICAL_READOUT_FIT_SEED,
                steps: CANONICAL_READOUT_FIT_STEPS as usize,
                batch_size: CANONICAL_READOUT_FIT_BATCH_SIZE as usize,
                token_rows_per_step: CANONICAL_READOUT_TOKEN_ROWS as usize,
                validation_windows: CANONICAL_READOUT_VALIDATION_WINDOWS as usize,
                source_checkpoint_sha256: &"a".repeat(64),
                source_lineage_sha256: &"b".repeat(64),
                source_kind: "raw",
                emission_gradient_mode: "attached",
                optimizer_recipe_id: CANONICAL_READOUT_OPTIMIZER_RECIPE_ID,
            },
        )
        .unwrap();
        let origin = CoreTrainingOrigin {
            initialization: CoreInitialization::Fresh,
            train_seed: 0x5eed,
            batch_size: 8,
            resolution_secs: 300,
            min_bars: 60_002,
            split_bounds: (1_700_000_000_000, 1_710_000_000_000),
            split_bounds_pinned: true,
            corpus_fingerprint: "c".repeat(64),
            supports_scoring_sha256: "d".repeat(64),
        };
        let mut provenance = CheckpointProvenance {
            completed_steps: 10,
            planned_steps: 10,
            steps_per_pass: 10,
            optimizer_recipe_id: "core".to_owned(),
            emission_gradient_mode: EmissionGradientMode::Attached,
            weight_readout: WeightReadout::Raw,
            core_origin: origin.clone(),
            head_fit: Some(HeadFitProvenance {
                source_checkpoint_sha256: "a".repeat(64),
                source_lineage_sha256: "b".repeat(64),
                source_weight_readout: WeightReadout::Raw,
                source_core_origin: origin,
                seed: CANONICAL_READOUT_FIT_SEED,
                steps: CANONICAL_READOUT_FIT_STEPS,
                batch_size: CANONICAL_READOUT_FIT_BATCH_SIZE,
                token_rows_per_step: CANONICAL_READOUT_TOKEN_ROWS,
                validation_windows: CANONICAL_READOUT_VALIDATION_WINDOWS,
                optimizer_recipe_id: CANONICAL_READOUT_OPTIMIZER_RECIPE_ID.to_owned(),
            }),
        };
        assert!(preflight_posthoc_readout_report(&root, &provenance).is_ok());
        provenance
            .head_fit
            .as_mut()
            .unwrap()
            .source_checkpoint_sha256 = "e".repeat(64);
        assert!(preflight_posthoc_readout_report(&root, &provenance).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
