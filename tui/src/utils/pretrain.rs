use shared::report::{Report, ReportKind, ReportSeries, ScaleKind};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const STEP_GLOBAL_STEP: usize = 0;
const STEP_TOTAL_LOSS: usize = 2;
const STEP_VAL_TOTAL_LOSS: usize = 21;

const VALIDATION_LABEL: usize = 0;
const VALIDATION_GLOBAL_STEP: usize = 1;
const VALIDATION_TOTAL_LOSS: usize = 2;
const MAX_STEP_REPORT_POINTS: usize = 2_000_001;

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

pub fn step_loss_report(run_root: &Path) -> Option<Report> {
    let train_content = fs::read_to_string(run_root.join("pretrain_train_steps.csv")).ok()?;
    let validation_content = fs::read_to_string(run_root.join("pretrain_validation.csv")).ok();
    step_loss_report_from_contents(&train_content, validation_content.as_deref())
}

fn step_loss_report_from_contents(
    train_content: &str,
    validation_content: Option<&str>,
) -> Option<Report> {
    let periodic_validation = validation_content
        .map(validation_loss_by_step)
        .unwrap_or_default();
    let mut rows = Vec::new();
    for row in train_content.lines().skip(1) {
        let Some(global_step) = cell(row, STEP_GLOBAL_STEP).and_then(|v| v.parse::<usize>().ok())
        else {
            continue;
        };
        let Some(train_v) = parse_cell(row, STEP_TOTAL_LOSS) else {
            continue;
        };
        let inline_val = parse_cell(row, STEP_VAL_TOTAL_LOSS).filter(|v| v.is_finite());
        rows.push((global_step, train_v, inline_val));
    }
    let max_step = rows.iter().map(|(step, _, _)| *step).max()?;
    let point_count = max_step.checked_add(1)?;
    if point_count > MAX_STEP_REPORT_POINTS {
        return None;
    }
    let mut train = vec![f32::NAN; point_count];
    let mut val = vec![f32::NAN; point_count];
    for (global_step, train_v, inline_val) in rows {
        train[global_step] = train_v;
        val[global_step] = periodic_validation
            .get(&global_step)
            .map(|entry| entry.loss)
            .or(inline_val)
            .unwrap_or(f32::NAN);
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

#[derive(Clone, Copy)]
struct ValidationLoss {
    loss: f32,
    priority: u8,
}

fn validation_loss_by_step(content: &str) -> HashMap<usize, ValidationLoss> {
    let mut values = HashMap::new();
    for row in content.lines().skip(1) {
        let Some(label) = cell(row, VALIDATION_LABEL) else {
            continue;
        };
        let Some(global_step) = cell(row, VALIDATION_GLOBAL_STEP).and_then(|v| v.parse().ok())
        else {
            continue;
        };
        let Some(loss) = parse_cell(row, VALIDATION_TOTAL_LOSS).filter(|v| v.is_finite()) else {
            continue;
        };
        let priority = validation_row_priority(label);
        let replace = values
            .get(&global_step)
            .is_none_or(|current: &ValidationLoss| priority >= current.priority);
        if replace {
            values.insert(global_step, ValidationLoss { loss, priority });
        }
    }
    values
}

fn validation_row_priority(label: &str) -> u8 {
    if label.starts_with("final") {
        3
    } else if label.parse::<usize>().is_ok() {
        2
    } else if label.starts_with("step:") {
        1
    } else {
        0
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn csv_row(len: usize, cells: &[(usize, &str)]) -> String {
        let mut row = vec![String::new(); len];
        for &(index, value) in cells {
            row[index] = value.to_string();
        }
        row.join(",")
    }

    fn report_values(report: Report) -> (Vec<f32>, Vec<f32>) {
        let ReportKind::MultiLine { series } = report.kind else {
            panic!("expected multi-line report");
        };
        (series[0].values.clone(), series[1].values.clone())
    }

    #[test]
    fn step_loss_merges_periodic_validation_by_global_step() {
        let train = [
            "header".to_string(),
            csv_row(26, &[(0, "10"), (2, "1.0")]),
            csv_row(26, &[(0, "20"), (2, "0.8")]),
            csv_row(26, &[(0, "30"), (2, "0.6")]),
        ]
        .join("\n");
        let validation = [
            "header".to_string(),
            csv_row(29, &[(0, "step:20"), (1, "20"), (2, "0.7")]),
        ]
        .join("\n");

        let (train_values, val_values) =
            report_values(step_loss_report_from_contents(&train, Some(&validation)).unwrap());

        assert_eq!(train_values.len(), 31);
        assert!(val_values[0].is_nan());
        assert_eq!(train_values[10], 1.0);
        assert_eq!(train_values[20], 0.8);
        assert_eq!(train_values[30], 0.6);
        assert_eq!(val_values[20], 0.7);
        assert!(val_values[19].is_nan());
    }

    #[test]
    fn step_loss_prefers_full_validation_and_prioritizes_final_rows() {
        let train = [
            "header".to_string(),
            csv_row(26, &[(0, "20"), (2, "0.8"), (21, "0.65")]),
            csv_row(26, &[(0, "30"), (2, "0.6")]),
        ]
        .join("\n");
        let validation = [
            "header".to_string(),
            csv_row(29, &[(0, "step:20"), (1, "20"), (2, "0.5")]),
            csv_row(29, &[(0, "final:1"), (1, "30"), (2, "0.4")]),
            csv_row(29, &[(0, "1"), (1, "30"), (2, "0.5")]),
            csv_row(29, &[(0, "step:30"), (1, "30"), (2, "0.7")]),
        ]
        .join("\n");

        let (_, val_values) =
            report_values(step_loss_report_from_contents(&train, Some(&validation)).unwrap());

        assert_eq!(val_values[20], 0.5);
        assert_eq!(val_values[30], 0.4);
    }

    #[test]
    fn step_loss_uses_global_step_as_the_series_index() {
        let train = [
            "header".to_string(),
            csv_row(26, &[(0, "2"), (2, "1.0"), (21, "0.9")]),
            csv_row(26, &[(0, "5"), (2, "0.7")]),
        ]
        .join("\n");

        let (train_values, val_values) =
            report_values(step_loss_report_from_contents(&train, None).unwrap());

        assert_eq!(train_values.len(), 6);
        assert!(train_values[0].is_nan());
        assert!(train_values[1].is_nan());
        assert_eq!(train_values[2], 1.0);
        assert!(train_values[3].is_nan());
        assert!(train_values[4].is_nan());
        assert_eq!(train_values[5], 0.7);
        assert_eq!(val_values[2], 0.9);
        assert!(val_values[5].is_nan());
    }

    #[test]
    fn step_loss_rejects_unreasonable_global_step_allocation() {
        let train = [
            "header".to_string(),
            csv_row(26, &[(0, &MAX_STEP_REPORT_POINTS.to_string()), (2, "1.0")]),
        ]
        .join("\n");

        assert!(step_loss_report_from_contents(&train, None).is_none());
    }
}
