use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use shared::report::{write_report, Report, ReportKind, ReportSeries, ScaleKind};

pub const MSE_JEPA_REPORT_BASES: &[&str] = &[
    "mse_jepa_loss",
    "mse_jepa_objective",
    "mse_jepa_representation",
    "mse_jepa_optimization",
];

#[derive(Clone, Copy, Debug)]
pub struct StepMetrics {
    pub total_loss: f64,
    pub prediction_mse: f64,
    pub persistence_mse: f64,
    pub skill_vs_persistence: f64,
    pub sigreg: f64,
    pub representation_std: f64,
    pub prediction_std: f64,
    pub target_std: f64,
    pub grad_norm: f64,
    pub muon_learning_rate: f64,
    pub adamw_learning_rate: f64,
    pub muon_momentum: f64,
    pub completed_step_seconds: f64,
    pub windows_per_second: f64,
    pub predicted_latent_positions_per_second: f64,
}

pub struct MseJepaReporter {
    gens: PathBuf,
    train_total: Vec<f32>,
    validation_total: Vec<f32>,
    prediction_mse: Vec<f32>,
    validation_prediction_mse: Vec<f32>,
    persistence_mse: Vec<f32>,
    validation_persistence_mse: Vec<f32>,
    skill_vs_persistence: Vec<f32>,
    validation_skill_vs_persistence: Vec<f32>,
    sigreg: Vec<f32>,
    representation_std: Vec<f32>,
    prediction_std: Vec<f32>,
    target_std: Vec<f32>,
    grad_norm: Vec<f32>,
    muon_learning_rate: Vec<f32>,
    adamw_learning_rate: Vec<f32>,
    muon_momentum: Vec<f32>,
    completed_step_seconds: Vec<f32>,
    windows_per_second: Vec<f32>,
    predicted_latent_positions_per_second: Vec<f32>,
}

impl MseJepaReporter {
    pub fn new(gens: impl AsRef<Path>) -> Self {
        Self {
            gens: gens.as_ref().to_path_buf(),
            train_total: Vec::new(),
            validation_total: Vec::new(),
            prediction_mse: Vec::new(),
            validation_prediction_mse: Vec::new(),
            persistence_mse: Vec::new(),
            validation_persistence_mse: Vec::new(),
            skill_vs_persistence: Vec::new(),
            validation_skill_vs_persistence: Vec::new(),
            sigreg: Vec::new(),
            representation_std: Vec::new(),
            prediction_std: Vec::new(),
            target_std: Vec::new(),
            grad_norm: Vec::new(),
            muon_learning_rate: Vec::new(),
            adamw_learning_rate: Vec::new(),
            muon_momentum: Vec::new(),
            completed_step_seconds: Vec::new(),
            windows_per_second: Vec::new(),
            predicted_latent_positions_per_second: Vec::new(),
        }
    }

    pub fn record_step(&mut self, metrics: StepMetrics) {
        self.train_total.push(metrics.total_loss as f32);
        self.validation_total.push(f32::NAN);
        self.prediction_mse.push(metrics.prediction_mse as f32);
        self.validation_prediction_mse.push(f32::NAN);
        self.persistence_mse.push(metrics.persistence_mse as f32);
        self.validation_persistence_mse.push(f32::NAN);
        self.skill_vs_persistence
            .push(metrics.skill_vs_persistence as f32);
        self.validation_skill_vs_persistence.push(f32::NAN);
        self.sigreg.push(metrics.sigreg as f32);
        self.representation_std
            .push(metrics.representation_std as f32);
        self.prediction_std.push(metrics.prediction_std as f32);
        self.target_std.push(metrics.target_std as f32);
        self.grad_norm.push(metrics.grad_norm as f32);
        self.muon_learning_rate
            .push(metrics.muon_learning_rate as f32);
        self.adamw_learning_rate
            .push(metrics.adamw_learning_rate as f32);
        self.muon_momentum.push(metrics.muon_momentum as f32);
        self.completed_step_seconds
            .push(metrics.completed_step_seconds as f32);
        self.windows_per_second
            .push(metrics.windows_per_second as f32);
        self.predicted_latent_positions_per_second
            .push(metrics.predicted_latent_positions_per_second as f32);
    }

    pub fn record_validation(
        &mut self,
        total_loss: f64,
        prediction_mse: f64,
        persistence_mse: f64,
        skill_vs_persistence: f64,
    ) {
        if let Some(slot) = self.validation_total.last_mut() {
            *slot = total_loss as f32;
        }
        if let Some(slot) = self.validation_prediction_mse.last_mut() {
            *slot = prediction_mse as f32;
        }
        if let Some(slot) = self.validation_persistence_mse.last_mut() {
            *slot = persistence_mse as f32;
        }
        if let Some(slot) = self.validation_skill_vs_persistence.last_mut() {
            *slot = skill_vs_persistence as f32;
        }
    }

    pub fn write(&self, epoch: usize) -> Result<()> {
        let output = self.gens.join(epoch.to_string());
        std::fs::create_dir_all(&output)
            .with_context(|| format!("failed creating {}", output.display()))?;
        self.write_multiline(
            &output,
            "mse_jepa_loss",
            "MSE-JEPA Loss",
            vec![
                series("train total", &self.train_total),
                series("validation total", &self.validation_total),
            ],
        )?;
        self.write_multiline(
            &output,
            "mse_jepa_objective",
            "MSE-JEPA Objective Terms",
            vec![
                series("train attached next-latent MSE", &self.prediction_mse),
                series(
                    "validation attached next-latent MSE",
                    &self.validation_prediction_mse,
                ),
                series("train latent persistence MSE", &self.persistence_mse),
                series(
                    "validation latent persistence MSE",
                    &self.validation_persistence_mse,
                ),
                series(
                    "train skill vs latent persistence",
                    &self.skill_vs_persistence,
                ),
                series(
                    "validation skill vs latent persistence",
                    &self.validation_skill_vs_persistence,
                ),
                series("SIGReg actual-N statistic", &self.sigreg),
            ],
        )?;
        self.write_multiline(
            &output,
            "mse_jepa_representation",
            "MSE-JEPA Representation Scale",
            vec![
                series("representation std", &self.representation_std),
                series("prediction std", &self.prediction_std),
                series("target std", &self.target_std),
            ],
        )?;
        self.write_multiline(
            &output,
            "mse_jepa_optimization",
            "MSE-JEPA Optimization",
            vec![
                series("gradient norm", &self.grad_norm),
                series("NorMuon learning rate", &self.muon_learning_rate),
                series("AdamW learning rate", &self.adamw_learning_rate),
                series("NorMuon momentum", &self.muon_momentum),
                series("completed-step seconds", &self.completed_step_seconds),
                series("windows/second", &self.windows_per_second),
                series(
                    "predicted latent positions/second",
                    &self.predicted_latent_positions_per_second,
                ),
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
                x_label: Some("optimizer step".to_owned()),
                y_label: None,
                scale: ScaleKind::Linear,
                kind: ReportKind::MultiLine { series },
            },
        )
        .with_context(|| format!("failed writing {base}.report.bin"))
    }
}

fn series(label: &str, values: &[f32]) -> ReportSeries {
    ReportSeries {
        label: label.to_owned(),
        values: values.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::{MseJepaReporter, StepMetrics, MSE_JEPA_REPORT_BASES};

    #[test]
    fn writer_and_shared_registry_are_bidirectionally_complete() {
        let shared: Vec<&str> = shared::report::PRETRAIN_REPORT_BASES
            .iter()
            .copied()
            .filter(|base| base.starts_with("mse_jepa_"))
            .collect();
        assert_eq!(shared, MSE_JEPA_REPORT_BASES);

        let root = std::env::temp_dir().join(format!("mse-jepa-reports-{}", uuid::Uuid::new_v4()));
        let mut reporter = MseJepaReporter::new(&root);
        reporter.record_step(StepMetrics {
            total_loss: 1.0,
            prediction_mse: 0.8,
            persistence_mse: 1.2,
            skill_vs_persistence: 1.0 / 3.0,
            sigreg: 10.0,
            representation_std: 1.1,
            prediction_std: 0.9,
            target_std: 1.0,
            grad_norm: 0.4,
            muon_learning_rate: 5e-3,
            adamw_learning_rate: 3e-4,
            muon_momentum: 0.85,
            completed_step_seconds: 0.25,
            windows_per_second: 32.0,
            predicted_latent_positions_per_second: 192_000.0,
        });
        reporter.record_validation(0.9, 0.7, 1.1, 1.0 - 0.7 / 1.1);
        reporter.write(0).unwrap();
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
                assert!(labels.contains(&"windows/second"));
                assert!(labels.contains(&"predicted latent positions/second"));
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
