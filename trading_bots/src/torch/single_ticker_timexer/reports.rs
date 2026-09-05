use std::{fs, path::Path};

use anyhow::{bail, ensure, Context, Result};
use shared::report::{read_report, write_report, Report, ReportKind, ReportSeries, ScaleKind};

use super::data::HORIZONS;
use super::evaluate::{Evaluation, EVIDENCE_SCHEMA};
use super::gates::GateDecision;
use super::support::Supports;

pub const REPORT_BASES: &[&str] = &[
    "timexer_training",
    "timexer_candles",
    "timexer_performance",
    "timexer_nll",
    "timexer_crps",
    "timexer_return_error",
    "timexer_standardized_error",
    "timexer_direction",
    "timexer_pit",
    "timexer_coverage",
    "timexer_tails",
    "timexer_resources",
    "timexer_evidence",
    "timexer_gates",
    "timexer_gate_statistics",
    "timexer_forecast",
    "timexer_forecast_probabilities",
];

pub fn write_benchmark(
    directory: &Path,
    ticker: &str,
    batch_size: usize,
    steps: usize,
    measurements: &[(String, f64)],
) -> Result<()> {
    let mut grouped = std::collections::BTreeMap::<&str, Vec<f64>>::new();
    for (label, value) in measurements {
        ensure!(value.is_finite(), "nonfinite performance measurement");
        grouped.entry(label).or_default().push(*value);
    }
    fs::create_dir_all(directory)?;
    chart(directory, "timexer_performance",
        &format!("{ticker} TimeXer CUDA performance; batch={batch_size}, steps={steps}"),
        "paired trial", "milliseconds / origins per second / parameter error",
        grouped.into_iter().map(|(label, values)| series(label, values)).collect())
}

pub fn write_training(directory: &Path, step: usize, loss: f64, learning_rate: f64) -> Result<()> {
    ensure!(
        loss.is_finite() && learning_rate.is_finite(),
        "nonfinite training report"
    );
    let directory = directory.join(step.to_string());
    fs::create_dir_all(&directory)?;
    chart(
        &directory,
        "timexer_training",
        "TimeXer one-stage training objective",
        "epoch",
        "loss / learning rate",
        vec![
            series("objective", [loss]),
            series("learning rate", [learning_rate]),
        ],
    )
}

pub fn write_evaluation(directory: &Path, step: usize, evaluation: &Evaluation) -> Result<()> {
    let directory = directory.join(step.to_string());
    fs::create_dir_all(&directory)?;
    let title = |name: &str| {
        format!(
            "{} {}: {name}; horizons 1,4,16,39,78,100",
            evaluation.model_kind, evaluation.split
        )
    };
    let summary = |label: &str, values: Vec<f64>| vec![series(label, values)];
    if evaluation.horizons.iter().all(|h| h.nll.is_some()) {
        chart(
            &directory,
            "timexer_nll",
            &title("hard categorical NLL"),
            "horizon index",
            "nat",
            summary(
                "NLL",
                evaluation.horizons.iter().map(|h| h.nll.unwrap()).collect(),
            ),
        )?;
        chart(
            &directory,
            "timexer_crps",
            &title("standardized CRPS"),
            "horizon index",
            "standardized return",
            summary(
                "CRPS",
                evaluation
                    .horizons
                    .iter()
                    .map(|h| h.crps.unwrap())
                    .collect(),
            ),
        )?;
        chart(
            &directory,
            "timexer_direction",
            &title("positive-return Brier score"),
            "horizon index",
            "Brier",
            summary(
                "Brier",
                evaluation
                    .horizons
                    .iter()
                    .map(|h| h.brier.unwrap())
                    .collect(),
            ),
        )?;
        let mut pit: Vec<_> = evaluation
            .horizons
            .iter()
            .map(|h| series(&format!("H{} PIT", h.horizon), h.pit_histogram.unwrap()))
            .collect();
        pit.push(series("uniform", [0.05; 20]));
        chart(
            &directory,
            "timexer_pit",
            &title("randomized categorical PIT"),
            "PIT bucket",
            "frequency",
            pit,
        )?;
        let mut coverage = Vec::new();
        for (i, nominal) in [0.5, 0.8, 0.9].into_iter().enumerate() {
            coverage.push(series(
                &format!("{:.0}% observed", nominal * 100.0),
                evaluation.horizons.iter().map(|h| h.coverage.unwrap()[i]),
            ));
            coverage.push(series(
                &format!("{:.0}% nominal", nominal * 100.0),
                [nominal; 6],
            ));
        }
        chart(
            &directory,
            "timexer_coverage",
            &title("central interval coverage"),
            "horizon index",
            "frequency",
            coverage,
        )?;
        chart(
            &directory,
            "timexer_tails",
            &title("nominal 5% tail exceedances"),
            "horizon index",
            "frequency",
            vec![
                series(
                    "lower tail",
                    evaluation.horizons.iter().map(|h| h.tails.unwrap()[0]),
                ),
                series(
                    "upper tail",
                    evaluation.horizons.iter().map(|h| h.tails.unwrap()[1]),
                ),
                series("nominal", [0.05; 6]),
                series(
                    "PIT total variation",
                    evaluation.horizons.iter().map(|h| h.pit_tv.unwrap()),
                ),
            ],
        )?;
    }
    chart(
        &directory,
        "timexer_return_error",
        &title("predictive mean errors"),
        "horizon index",
        "cumulative log return",
        vec![
            series("RMSE", evaluation.horizons.iter().map(|h| h.rmse)),
            series("MAE", evaluation.horizons.iter().map(|h| h.mae)),
        ],
    )?;
    chart(
        &directory,
        "timexer_standardized_error",
        &title("standardized predictive-mean error"),
        "horizon index",
        "standardized return",
        vec![series(
            "RMSE",
            evaluation.horizons.iter().map(|h| h.standardized_rmse),
        )],
    )?;
    let mut resources = Vec::new();
    if let Some(latency) = evaluation.latency_p95_ms {
        resources.push(series("batch-one forward p95 milliseconds", [latency]));
    }
    if let Some(bytes) = evaluation.peak_device_memory_bytes {
        resources.push(series(
            "peak CUDA allocated MiB",
            [bytes as f64 / 1_048_576.0],
        ));
    }
    if !resources.is_empty() {
        chart(
            &directory,
            "timexer_resources",
            &title("measured inference resources"),
            "evaluation",
            "milliseconds / MiB",
            resources,
        )?;
    }
    let paired = HORIZONS
        .iter()
        .enumerate()
        .map(|(h, horizon)| {
            series(
                &format!(
                    "H{horizon} {}",
                    if evaluation
                        .origins
                        .first()
                        .is_some_and(|row| row.nll.is_some())
                    {
                        "NLL"
                    } else {
                        "squared return error"
                    }
                ),
                evaluation
                    .origins
                    .iter()
                    .map(|row| row.nll.map(|nll| nll[h]).unwrap_or(row.squared_error[h])),
            )
        })
        .collect();
    write_report(
        directory.join("timexer_evidence.report.bin"),
        &Report {
            title: title("paired origin evidence"),
            x_label: Some("chronological origin".to_owned()),
            y_label: Some("NLL / squared return error".to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::Evidence {
                series: paired,
                metadata: postcard::to_stdvec(evaluation)?,
            },
        },
    )?;
    Ok(())
}

pub fn read_evaluation(path: &Path) -> Result<Evaluation> {
    let path = if path.is_dir() {
        path.join("timexer_evidence.report.bin")
    } else {
        path.to_path_buf()
    };
    let report = read_report(&path)
        .with_context(|| format!("read evaluation evidence {}", path.display()))?;
    let ReportKind::Evidence { metadata, .. } = report.kind else {
        bail!("report has no exact evaluation evidence");
    };
    let mut evaluation: Evaluation =
        postcard::from_bytes(&metadata).context("decode evaluation evidence")?;
    ensure!(
        evaluation.schema == EVIDENCE_SCHEMA,
        "unsupported evaluation evidence schema"
    );
    evaluation.summarize()?;
    Ok(evaluation)
}

pub fn write_gate_decision(directory: &Path, step: usize, decision: &GateDecision) -> Result<()> {
    let directory = directory.join(step.to_string());
    fs::create_dir_all(&directory)?;
    let statistics = decision
        .checks
        .iter()
        .map(|check| {
            let interval = check.confidence.unwrap_or([f64::NAN; 2]);
            series(
                &check.name,
                [check.value, check.threshold, interval[0], interval[1]],
            )
        })
        .collect();
    chart(
        &directory,
        "timexer_gate_statistics",
        "Replacement gate values, thresholds and paired calendar-block 95% intervals",
        "0=value, 1=threshold, 2=lower95, 3=upper95",
        "gate statistic",
        statistics,
    )?;
    let series = decision
        .checks
        .iter()
        .map(|check| series(&check.name, [f64::from(check.passed)]))
        .collect();
    write_report(
        directory.join("timexer_gates.report.bin"),
        &Report {
            title: format!(
                "TimeXer replacement gates: {} ({} calendar blocks)",
                if decision.promoted { "PASS" } else { "FAIL" },
                decision.calendar_blocks
            ),
            x_label: Some("assessment".to_owned()),
            y_label: Some("pass=1, fail=0".to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::Evidence {
                series,
                metadata: postcard::to_stdvec(decision)?,
            },
        },
    )?;
    Ok(())
}

pub fn write_forecast(
    directory: &Path,
    step: usize,
    supports: &Supports,
    probabilities: &[Vec<f64>],
    sigma: f64,
) -> Result<()> {
    ensure!(
        supports.horizons.len() == 6
            && probabilities.len() == 6
            && sigma.is_finite()
            && sigma > 0.0,
        "invalid direct forecast contract"
    );
    for probability in probabilities {
        ensure!(
            probability.len() == 128
                && probability.iter().all(|p| p.is_finite() && *p >= 0.0)
                && (probability.iter().sum::<f64>() - 1.0).abs() < 1e-5,
            "invalid forecast probabilities"
        );
    }
    let directory = directory.join(step.to_string());
    fs::create_dir_all(&directory)?;
    let mut forecasts = vec![series(
        "predictive mean",
        (0..6).map(|h| {
            supports.horizons[h].mean(&probabilities[h]) * sigma * (HORIZONS[h] as f64).sqrt()
        }),
    )];
    for quantile in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95] {
        forecasts.push(series(
            &format!("p{:.0}", quantile * 100.0),
            (0..6).map(|h| {
                supports.horizons[h].quantile(&probabilities[h], quantile)
                    * sigma
                    * (HORIZONS[h] as f64).sqrt()
            }),
        ));
    }
    chart(
        &directory,
        "timexer_forecast",
        "Direct cumulative-return marginals; horizons 1,4,16,39,78,100",
        "horizon index",
        "cumulative log return",
        forecasts,
    )?;
    chart(
        &directory,
        "timexer_forecast_probabilities",
        "Direct marginal probabilities on fitted horizon supports",
        "support bin",
        "probability",
        probabilities
            .iter()
            .zip(HORIZONS)
            .map(|(p, h)| series(&format!("H{h}"), p.iter().copied()))
            .collect(),
    )
}

fn series(label: &str, values: impl IntoIterator<Item = f64>) -> ReportSeries {
    ReportSeries {
        label: label.to_owned(),
        values: values.into_iter().map(|value| value as f32).collect(),
    }
}

fn chart(
    directory: &Path,
    stem: &str,
    title: &str,
    x: &str,
    y: &str,
    series: Vec<ReportSeries>,
) -> Result<()> {
    ensure!(
        REPORT_BASES.contains(&stem),
        "unregistered TimeXer report base {stem}"
    );
    write_report(
        directory.join(format!("{stem}.report.bin")),
        &Report {
            title: title.to_owned(),
            x_label: Some(x.to_owned()),
            y_label: Some(y.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::MultiLine { series },
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_timexer_reports_are_registered_in_tui() {
        let tui = include_str!("../../../../tui/src/main.rs");
        for base in REPORT_BASES {
            assert!(
                tui.contains(&format!("\"{base}\"")),
                "missing TUI base {base}"
            );
        }
    }

    #[test]
    fn evidence_preserves_exact_payload_and_remains_readable() {
        let report = Report {
            title: "evidence".into(),
            x_label: None,
            y_label: None,
            scale: ScaleKind::Linear,
            kind: ReportKind::Evidence {
                series: vec![series("NLL", [4.5])],
                metadata: vec![0, 255, 42],
            },
        };
        let bytes = postcard::to_stdvec(&report).unwrap();
        let decoded: Report = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.kind.to_lines(), vec!["0\tNLL=4.5"]);
        let ReportKind::Evidence { metadata, .. } = decoded.kind else {
            panic!("lost evidence");
        };
        assert_eq!(metadata, [0, 255, 42]);
    }
}
