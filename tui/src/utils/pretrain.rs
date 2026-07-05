use shared::report::{Report, ReportKind, ReportSeries, ScaleKind};
use std::fs;
use std::path::Path;

const STEP_TOTAL_LOSS: usize = 2;
const STEP_VAL_TOTAL_LOSS: usize = 21;

const TEST_TOTAL_LOSS: usize = 2;
const TEST_PROBE_MSE: usize = 9;
const TEST_ROLLOUT_MEAN_MSE: usize = 20;

fn cell(row: &str, idx: usize) -> Option<&str> {
    row.split(',').nth(idx).map(str::trim)
}

fn parse_cell(row: &str, idx: usize) -> Option<f32> {
    match cell(row, idx) {
        Some(s) if !s.is_empty() => s.parse::<f32>().ok(),
        _ => None,
    }
}

// Reads the run's per-step training CSV and overlays train vs (sparse) validation
// total loss. Validation cells are empty on non-validation rows and are rendered as
// gaps (NaN) so the multi-line renderer skips them without misaligning the x-axis.
pub fn step_loss_report(run_root: &Path) -> Option<Report> {
    let content = fs::read_to_string(run_root.join("pretrain_train_steps.csv")).ok()?;
    let mut train = Vec::new();
    let mut val = Vec::new();
    for row in content.lines().skip(1) {
        let Some(train_v) = parse_cell(row, STEP_TOTAL_LOSS) else {
            continue;
        };
        train.push(train_v);
        val.push(parse_cell(row, STEP_VAL_TOTAL_LOSS).unwrap_or(f32::NAN));
    }
    if train.is_empty() {
        return None;
    }
    Some(Report {
        title: "Pretrain Step Loss (Train vs Val)".to_string(),
        x_label: Some("step".to_string()),
        y_label: Some("total loss".to_string()),
        scale: ScaleKind::Linear,
        kind: ReportKind::MultiLine {
            series: vec![
                ReportSeries {
                    label: "train".to_string(),
                    values: train,
                },
                ReportSeries {
                    label: "val".to_string(),
                    values: val,
                },
            ],
        },
    })
}

// Reads the run's test CSV (same schema as validation) and overlays the headline
// test metrics across epochs; the rightmost point is the latest test evaluation.
pub fn test_report(run_root: &Path) -> Option<Report> {
    let content = fs::read_to_string(run_root.join("pretrain_test.csv")).ok()?;
    let mut total_loss = Vec::new();
    let mut probe_mse = Vec::new();
    let mut rollout_mean_mse = Vec::new();
    for row in content.lines().skip(1) {
        let Some(total_v) = parse_cell(row, TEST_TOTAL_LOSS) else {
            continue;
        };
        total_loss.push(total_v);
        probe_mse.push(parse_cell(row, TEST_PROBE_MSE).unwrap_or(f32::NAN));
        rollout_mean_mse.push(parse_cell(row, TEST_ROLLOUT_MEAN_MSE).unwrap_or(f32::NAN));
    }
    if total_loss.is_empty() {
        return None;
    }
    Some(Report {
        title: "Pretrain Test".to_string(),
        x_label: Some("epoch".to_string()),
        y_label: Some("metric".to_string()),
        scale: ScaleKind::Linear,
        kind: ReportKind::MultiLine {
            series: vec![
                ReportSeries {
                    label: "total_loss".to_string(),
                    values: total_loss,
                },
                ReportSeries {
                    label: "probe_mse".to_string(),
                    values: probe_mse,
                },
                ReportSeries {
                    label: "rollout_mean_mse".to_string(),
                    values: rollout_mean_mse,
                },
            ],
        },
    })
}

// Both headline CSV-derived pretraining charts for the current run, ready to be
// injected as in-memory chart nodes alongside the on-disk meta reports.
pub fn run_reports(run_root: &Path) -> Vec<(String, Report)> {
    let mut reports = Vec::new();
    if let Some(report) = step_loss_report(run_root) {
        reports.push((report.title.clone(), report));
    }
    if let Some(report) = test_report(run_root) {
        reports.push((report.title.clone(), report));
    }
    reports
}
