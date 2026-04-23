use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::Result;
use shared::report::{Report, ReportKind, ReportSeries, ScaleKind};

use crate::history::report::write_report;

use super::{
    backtest::{write_trace_reports, BacktestTrace},
    metrics::{BacktestMetrics, PopulationStats},
};

pub struct SessionLogger {
    log_file: PathBuf,
}

impl SessionLogger {
    pub fn new(_run_root: &Path, log_file: &Path) -> Result<Self> {
        Ok(Self {
            log_file: log_file.to_path_buf(),
        })
    }

    pub fn log_line(&mut self, line: &str) -> Result<()> {
        println!("{line}");
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_file)?;
        writeln!(file, "{line}")?;
        Ok(())
    }
}

#[derive(Default, Clone)]
pub struct GaHistory {
    pub train_fitness: Vec<f64>,
    pub validation_fitness: Vec<f64>,
    pub best_validation_fitness: Vec<f64>,
    pub train_return_pct: Vec<f64>,
    pub validation_return_pct: Vec<f64>,
    pub train_outperformance: Vec<f64>,
    pub validation_outperformance: Vec<f64>,
    pub train_max_drawdown: Vec<f64>,
    pub validation_max_drawdown: Vec<f64>,
    pub train_sharpe: Vec<f64>,
    pub validation_sharpe: Vec<f64>,
    pub train_turnover: Vec<f64>,
    pub validation_turnover: Vec<f64>,
    pub train_trade_count: Vec<f64>,
    pub validation_trade_count: Vec<f64>,
    pub validation_minus_train: Vec<f64>,
    pub train_p05: Vec<f64>,
    pub train_p50: Vec<f64>,
    pub train_p95: Vec<f64>,
    pub train_worst: Vec<f64>,
    pub train_best: Vec<f64>,
    pub mutation_entropy: Vec<f64>,
}

impl GaHistory {
    pub fn record_generation(
        &mut self,
        train_metrics: &BacktestMetrics,
        validation_metrics: &BacktestMetrics,
        best_validation_score: f64,
        train_population: PopulationStats,
        mutation_entropy: f64,
    ) {
        self.train_fitness.push(train_metrics.score);
        self.validation_fitness.push(validation_metrics.score);
        self.best_validation_fitness.push(best_validation_score);
        self.train_return_pct.push(train_metrics.return_pct);
        self.validation_return_pct
            .push(validation_metrics.return_pct);
        self.train_outperformance
            .push(train_metrics.outperformance_pct);
        self.validation_outperformance
            .push(validation_metrics.outperformance_pct);
        self.train_max_drawdown.push(train_metrics.max_drawdown_pct);
        self.validation_max_drawdown
            .push(validation_metrics.max_drawdown_pct);
        self.train_sharpe.push(train_metrics.sharpe);
        self.validation_sharpe.push(validation_metrics.sharpe);
        self.train_turnover.push(train_metrics.turnover);
        self.validation_turnover.push(validation_metrics.turnover);
        self.train_trade_count
            .push(train_metrics.trade_count as f64);
        self.validation_trade_count
            .push(validation_metrics.trade_count as f64);
        self.validation_minus_train
            .push(validation_metrics.score - train_metrics.score);
        self.train_p05.push(train_population.p05);
        self.train_p50.push(train_population.p50);
        self.train_p95.push(train_population.p95);
        self.train_worst.push(train_population.worst);
        self.train_best.push(train_population.best);
        self.mutation_entropy.push(mutation_entropy);
    }

    pub fn write_reports(&self, output_dir: &Path) -> Result<()> {
        fs::create_dir_all(output_dir)?;
        write_multi_line_report(
            output_dir,
            "ga_fitness",
            "Fitness",
            &[
                ("train", &self.train_fitness),
                ("validation", &self.validation_fitness),
                ("best_validation", &self.best_validation_fitness),
            ],
        )?;
        write_multi_line_report(
            output_dir,
            "ga_return_pct",
            "Return %",
            &[
                ("train", &self.train_return_pct),
                ("validation", &self.validation_return_pct),
            ],
        )?;
        write_multi_line_report(
            output_dir,
            "ga_outperformance",
            "Outperformance %",
            &[
                ("train", &self.train_outperformance),
                ("validation", &self.validation_outperformance),
            ],
        )?;
        write_multi_line_report(
            output_dir,
            "ga_max_drawdown",
            "Max Drawdown %",
            &[
                ("train", &self.train_max_drawdown),
                ("validation", &self.validation_max_drawdown),
            ],
        )?;
        write_multi_line_report(
            output_dir,
            "ga_sharpe",
            "Sharpe",
            &[
                ("train", &self.train_sharpe),
                ("validation", &self.validation_sharpe),
            ],
        )?;
        write_multi_line_report(
            output_dir,
            "ga_turnover",
            "Turnover",
            &[
                ("train", &self.train_turnover),
                ("validation", &self.validation_turnover),
            ],
        )?;
        write_multi_line_report(
            output_dir,
            "ga_trade_count",
            "Trade Count",
            &[
                ("train", &self.train_trade_count),
                ("validation", &self.validation_trade_count),
            ],
        )?;
        write_multi_line_report(
            output_dir,
            "ga_generalization_gap",
            "Validation - Train",
            &[("validation_minus_train", &self.validation_minus_train)],
        )?;
        write_multi_line_report(
            output_dir,
            "ga_distribution",
            "Train Population Score Distribution",
            &[
                ("p05", &self.train_p05),
                ("p50", &self.train_p50),
                ("p95", &self.train_p95),
                ("worst", &self.train_worst),
                ("best", &self.train_best),
            ],
        )?;
        write_multi_line_report(
            output_dir,
            "ga_mutation_entropy",
            "Mutation Entropy",
            &[("mutation_entropy", &self.mutation_entropy)],
        )?;
        Ok(())
    }
}

pub fn write_split_trace(
    output_dir: &Path,
    split_name: &str,
    tickers: &[String],
    trace: &BacktestTrace,
) -> Result<()> {
    write_trace_reports(output_dir, split_name, tickers, trace)
}

fn write_multi_line_report(
    output_dir: &Path,
    name: &str,
    title: &str,
    lines: &[(&str, &[f64])],
) -> Result<()> {
    let report = Report {
        title: title.to_string(),
        x_label: Some("Generation".to_string()),
        y_label: None,
        scale: ScaleKind::Linear,
        kind: ReportKind::MultiLine {
            series: lines
                .iter()
                .map(|(label, values)| ReportSeries {
                    label: (*label).to_string(),
                    values: values.iter().map(|value| *value as f32).collect(),
                })
                .collect(),
        },
    };
    write_report(
        output_dir
            .join(format!("{name}.report.bin"))
            .to_string_lossy()
            .as_ref(),
        &report,
    )?;
    Ok(())
}
