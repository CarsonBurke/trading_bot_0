use std::{collections::BTreeMap, path::Path};

use super::corpus::CorpusContract;

use anyhow::{ensure, Context, Result};
use shared::report::{write_report, CandleBar, Report, ReportKind, ReportSeries, ScaleKind};

#[derive(Debug, Clone)]
pub struct Metrics {
    pub step: usize,
    pub epoch: usize,
    pub completed_origins: usize,
    pub total_origins: usize,
    pub completed_target_bars: usize,
    pub total_target_bars: usize,
    pub train_mse: Option<f64>,
    pub validation_mse: f64,
    pub projected_mse: Option<f64>,
    pub persistence_mse: f64,
    pub rmse_price: f64,
    pub mae_price: f64,
    pub invalid_ohlc_fraction: f64,
    pub projected_invalid_fraction: Option<f64>,
    pub eval_ms: f64,
    pub step_ms: Option<f64>,
    pub loader_wait_ms: Option<f64>,
    pub peak_allocator_mib: Option<f64>,
    pub validation_is_full: bool,
    pub validation_origins: usize,
}

pub struct CandleWindow {
    pub ticker: String,
    pub actual: Vec<CandleBar>,
    pub origin: usize,
    pub predicted: Vec<CandleBar>,
    /// Forecast-origin completion time, in Unix milliseconds.
    pub timestamp: i64,
}

pub fn write_metrics(output: &Path, points: &[Metrics]) -> Result<()> {
    let Some(last) = points.last() else {
        return Ok(());
    };
    ensure!(
        points.windows(2).all(|p| p[0].step < p[1].step),
        "report steps must increase"
    );
    ensure!(
        points.iter().all(|p| p.epoch == last.epoch
            && p.completed_origins <= p.total_origins
            && p.completed_target_bars <= p.total_target_bars
            && [
                p.validation_mse,
                p.persistence_mse,
                p.rmse_price,
                p.mae_price,
                p.invalid_ohlc_fraction,
                p.eval_ms
            ]
            .iter()
            .all(|v| v.is_finite())
            && p.train_mse.is_none_or(f64::is_finite)
            && p.projected_mse.is_none_or(|v| v.is_finite() && v >= 0.)
            && p.projected_invalid_fraction
                .is_none_or(|v| (0.0..=1.0).contains(&v))
            && p.step_ms.is_none_or(|v| v.is_finite() && v >= 0.)
            && p.loader_wait_ms.is_none_or(|v| v.is_finite() && v >= 0.)
            && p.peak_allocator_mib.is_none_or(|v| v.is_finite() && v > 0.)
            && (0.0..=1.0).contains(&p.invalid_ohlc_fraction)),
        "invalid segment report metrics"
    );
    let series = |label: &str, value: fn(&Metrics) -> f64| ReportSeries {
        label: label.to_owned(),
        values: points.iter().map(|p| value(p) as f32).collect(),
    };
    let chart = |base: &str, title: &str, unit: &str, series: Vec<ReportSeries>| -> Result<()> {
        let series: Vec<_> = series
            .into_iter()
            .filter(|s| s.values.iter().any(|v| v.is_finite()))
            .collect();
        if series.is_empty() {
            return Ok(());
        }
        write_report(
            output.join(format!("{base}.report.bin")),
            &Report {
                title: format!("TimeXer epoch {} | {title}", last.epoch),
                x_label: Some("optimizer step".to_owned()),
                y_label: Some(unit.to_owned()),
                scale: ScaleKind::Linear,
                kind: ReportKind::IndexedLines {
                    steps: points.iter().map(|p| p.step as u64).collect(),
                    series,
                },
            },
        )?;
        Ok(())
    };
    chart(
        "timexer_segment_validation",
        &format!(
            "forecast vs persistence | {} origins | price RMSE {:.4}, MAE {:.4}",
            last.validation_origins, last.rmse_price, last.mae_price
        ),
        "standardized OHLC MSE",
        vec![
            series("training", |p| p.train_mse.unwrap_or(f64::NAN)),
            series("held-out preview", |p| {
                if !p.validation_is_full {
                    p.validation_mse
                } else {
                    f64::NAN
                }
            }),
            series("preview persistence", |p| {
                if !p.validation_is_full {
                    p.persistence_mse
                } else {
                    f64::NAN
                }
            }),
            series("full validation", |p| {
                if p.validation_is_full {
                    p.validation_mse
                } else {
                    f64::NAN
                }
            }),
            series("full-validation persistence", |p| {
                if p.validation_is_full {
                    p.persistence_mse
                } else {
                    f64::NAN
                }
            }),
            series("weighted projection", |p| {
                p.projected_mse.unwrap_or(f64::NAN)
            }),
        ],
    )?;
    chart(
        "timexer_segment_timing",
        &last.peak_allocator_mib.map_or_else(
            || "training and evaluation".to_owned(),
            |peak| format!("training and evaluation | peak allocator {peak:.0} MiB"),
        ),
        "milliseconds",
        vec![
            series("training step", |p| p.step_ms.unwrap_or(f64::NAN)),
            series("evaluation", |p| p.eval_ms),
            series("host loader wait (can overlap GPU work)", |p| {
                p.loader_wait_ms.unwrap_or(f64::NAN)
            }),
        ],
    )?;
    chart(
        "timexer_segment_progress",
        &format!(
            "{} / {} unique training target bars",
            last.completed_target_bars, last.total_target_bars
        ),
        "fraction",
        vec![
            series("training targets completed", |p| {
                p.completed_target_bars as f64 / p.total_target_bars.max(1) as f64
            }),
            series("invalid forecast candles", |p| p.invalid_ohlc_fraction),
            series("invalid projected candles", |p| {
                p.projected_invalid_fraction.unwrap_or(f64::NAN)
            }),
        ],
    )
}

pub fn write_candles(
    output: &Path,
    projected: bool,
    epoch: usize,
    step: usize,
    windows: &[CandleWindow],
) -> Result<()> {
    for (index, window) in windows.iter().enumerate() {
        ensure!(
            window.origin < window.actual.len() && !window.predicted.is_empty(),
            "invalid candle window"
        );
        ensure!(
            window.actual.iter().all(CandleBar::is_valid_ohlc),
            "invalid observed OHLC in candle window"
        );
        let clock = chrono::DateTime::from_timestamp_millis(window.timestamp)
            .context("invalid candle origin timestamp")?
            .with_timezone(&chrono_tz::America::New_York);
        let ticker = &window.ticker;
        ensure!(!ticker.is_empty(), "candle window requires ticker identity");
        let forecast_label = if projected {
            "weighted projection of direct OHLC MSE forecast"
        } else {
            "direct OHLC MSE forecast"
        };
        let report = Report {
            title: format!("{ticker} epoch {epoch} step {step} | held-out window {} | {} | {}-bar {forecast_label}",
                index + 1, clock.format("%Y-%m-%d %H:%M %Z"), window.predicted.len()),
            x_label: Some("five-minute bars from forecast origin".to_owned()),
            y_label: Some("price".to_owned()), scale: ScaleKind::Linear,
            kind: ReportKind::CandleSegment {
                actual: window.actual.clone(), origin: window.origin, predicted: window.predicted.clone(),
            },
        };
        if index == 0 {
            write_report(output.join("timexer_segment_candles.report.bin"), &report)?;
        }
        write_report(
            output.join("candle_snapshots").join(format!(
                "step{step}_epoch{epoch:03}_window{:02}_fan.report.bin",
                index + 1
            )),
            &report,
        )?;
    }
    Ok(())
}

pub fn write_corpus(output: &Path, contract: &CorpusContract) -> Result<()> {
    let path = output.join("timexer_segment_progress.report.bin");
    let mut report = if path.exists() {
        shared::report::read_report(&path)?
    } else {
        Report {
            title: "TimeXer corpus ready".into(),
            x_label: Some("optimizer step".into()),
            y_label: Some("fraction".into()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: vec![0],
                series: vec![ReportSeries {
                    label: "training targets completed".into(),
                    values: vec![0.],
                }],
            },
        }
    };
    let mut reasons = BTreeMap::new();
    for ticker in &contract.excluded_tickers {
        *reasons.entry(ticker.reason.as_str()).or_insert(0usize) += 1;
    }
    let reasons = reasons
        .into_iter()
        .map(|(reason, count)| format!("{reason}: {count}"))
        .collect::<Vec<_>>()
        .join(", ");
    let source_bars: usize = contract
        .tickers
        .iter()
        .map(|ticker| ticker.source_bars)
        .sum();
    let invalid_bars: usize = contract
        .tickers
        .iter()
        .map(|ticker| ticker.invalid_ohlc_indices.len())
        .sum();
    report.title = format!("{} | corpus: {} tickers, {source_bars} source bars, {invalid_bars} malformed source bars omitted; {} training / {} validation targets, {} unused validation remainder; {} excluded ({reasons})",
        report.title.split(" | corpus:").next().unwrap_or(&report.title), contract.tickers.len(), contract.train_target_bars, contract.validation_target_bars,
        contract.validation_remainder_bars, contract.excluded_tickers.len());
    write_report(path, &report)?;
    Ok(())
}
