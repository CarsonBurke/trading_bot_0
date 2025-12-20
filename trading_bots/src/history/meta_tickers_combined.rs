use crate::constants::files::TRAINING_PATH;
use crate::history::episode_tickers_combined::EpisodeHistory;
use crate::history::report::{write_report, Report, ReportKind, ReportSeries, ScaleKind};
use crate::utils::create_folder_if_not_exists;

#[derive(Default, Debug)]
pub struct MetaHistory {
    pub final_assets: Vec<f64>,
    pub cumulative_reward: Vec<f64>,
    pub outperformance: Vec<f64>,
    pub loss: Vec<f64>,
    pub grad_norm: Vec<f64>,
    pub total_commissions: Vec<f64>,
    pub mean_std: Vec<f64>,
    pub min_std: Vec<f64>,
    pub max_std: Vec<f64>,
    pub rpo_alpha: Vec<f64>,
    pub mean_advantage: Vec<f64>,
    pub min_advantage: Vec<f64>,
    pub max_advantage: Vec<f64>,
    pub mean_divisor: Vec<f64>,
}

impl MetaHistory {
    pub fn record(&mut self, history: &EpisodeHistory, outperformance: f64) {
        self.final_assets.push(history.final_assets());
        self.cumulative_reward.push(history.rewards.iter().sum::<f64>());
        self.outperformance.push(outperformance);
        self.total_commissions.push(history.total_commissions);
    }

    pub fn record_loss(&mut self, loss: f64) {
        self.loss.push(loss);
    }

    pub fn record_grad_norm(&mut self, grad_norm: f64) {
        self.grad_norm.push(grad_norm);
    }

    pub fn record_std_stats(&mut self, mean_std: f64, min_std: f64, max_std: f64, rpo_alpha: f64) {
        self.mean_std.push(mean_std);
        self.min_std.push(min_std);
        self.max_std.push(max_std);
        self.rpo_alpha.push(rpo_alpha);
    }

    pub fn record_advantage_stats(&mut self, mean: f64, min: f64, max: f64) {
        self.mean_advantage.push(mean);
        self.min_advantage.push(min);
        self.max_advantage.push(max);
    }

    pub fn record_divisor(&mut self, mean_divisor: f64) {
        self.mean_divisor.push(mean_divisor);
    }

    pub fn write_reports(&self, episode: usize) {
        let base_dir = format!("{TRAINING_PATH}/gens/{}", episode);
        create_folder_if_not_exists(&base_dir);
        if !self.final_assets.is_empty() {
            let report = Report {
                title: "Final Assets".to_string(),
                x_label: Some("Episode".to_string()),
                y_label: Some("Assets".to_string()),
                scale: ScaleKind::Linear,
                kind: ReportKind::Simple {
                    values: self.final_assets.clone(),
                    ema_alpha: Some(0.05),
                },
            };
            let _ = write_report(&format!("{base_dir}/final_assets.report.bin"), &report);
        }
        if !self.cumulative_reward.is_empty() {
            let report = Report {
                title: "Cumulative Reward".to_string(),
                x_label: Some("Episode".to_string()),
                y_label: Some("Reward".to_string()),
                scale: ScaleKind::Linear,
                kind: ReportKind::Simple {
                    values: self.cumulative_reward.clone(),
                    ema_alpha: Some(0.05),
                },
            };
            let _ = write_report(&format!("{base_dir}/cumulative_reward.report.bin"), &report);
        }
        if !self.outperformance.is_empty() {
            let report = Report {
                title: "Outperformance".to_string(),
                x_label: Some("Episode".to_string()),
                y_label: Some("Outperformance".to_string()),
                scale: ScaleKind::Linear,
                kind: ReportKind::Simple {
                    values: self.outperformance.clone(),
                    ema_alpha: Some(0.05),
                },
            };
            let _ = write_report(&format!("{base_dir}/outperformance.report.bin"), &report);
        }
        if !self.loss.is_empty() {
            let report = Report {
                title: "Loss (Log)".to_string(),
                x_label: Some("Episode".to_string()),
                y_label: Some("Loss".to_string()),
                scale: ScaleKind::Symlog,
                kind: ReportKind::Simple {
                    values: self.loss.clone(),
                    ema_alpha: Some(0.05),
                },
            };
            let _ = write_report(&format!("{base_dir}/loss_log.report.bin"), &report);
        }
        if !self.grad_norm.is_empty() {
            let report = Report {
                title: "Grad Norm (Log)".to_string(),
                x_label: Some("Episode".to_string()),
                y_label: Some("Grad Norm".to_string()),
                scale: ScaleKind::Symlog,
                kind: ReportKind::Simple {
                    values: self.grad_norm.clone(),
                    ema_alpha: Some(0.05),
                },
            };
            let _ = write_report(&format!("{base_dir}/grad_norm_log.report.bin"), &report);
        }
        if !self.total_commissions.is_empty() {
            let report = Report {
                title: "Total Commissions".to_string(),
                x_label: Some("Episode".to_string()),
                y_label: Some("Commissions".to_string()),
                scale: ScaleKind::Linear,
                kind: ReportKind::Simple {
                    values: self.total_commissions.clone(),
                    ema_alpha: Some(0.05),
                },
            };
            let _ = write_report(&format!("{base_dir}/total_commissions.report.bin"), &report);
        }
        if !self.mean_std.is_empty() {
            let report = Report {
                title: "Std Stats".to_string(),
                x_label: Some("Episode".to_string()),
                y_label: None,
                scale: ScaleKind::Linear,
                kind: ReportKind::MultiLine {
                    series: vec![
                        ReportSeries {
                            label: "mean".to_string(),
                            values: self.mean_std.clone(),
                        },
                        ReportSeries {
                            label: "min".to_string(),
                            values: self.min_std.clone(),
                        },
                        ReportSeries {
                            label: "max".to_string(),
                            values: self.max_std.clone(),
                        },
                        ReportSeries {
                            label: "rpo_alpha".to_string(),
                            values: self.rpo_alpha.clone(),
                        },
                    ],
                },
            };
            let _ = write_report(&format!("{base_dir}/std_stats.report.bin"), &report);
        }
        if !self.mean_advantage.is_empty() {
            let report = Report {
                title: "Advantage Stats (Log)".to_string(),
                x_label: Some("Episode".to_string()),
                y_label: None,
                scale: ScaleKind::Symlog,
                kind: ReportKind::MultiLine {
                    series: vec![
                        ReportSeries {
                            label: "mean".to_string(),
                            values: self.mean_advantage.clone(),
                        },
                        ReportSeries {
                            label: "min".to_string(),
                            values: self.min_advantage.clone(),
                        },
                        ReportSeries {
                            label: "max".to_string(),
                            values: self.max_advantage.clone(),
                        },
                    ],
                },
            };
            let _ = write_report(&format!("{base_dir}/advantage_stats_log.report.bin"), &report);
        }
        if !self.mean_divisor.is_empty() {
            let report = Report {
                title: "Divisor".to_string(),
                x_label: Some("Episode".to_string()),
                y_label: None,
                scale: ScaleKind::Linear,
                kind: ReportKind::Simple {
                    values: self.mean_divisor.clone(),
                    ema_alpha: Some(0.05),
                },
            };
            let _ = write_report(&format!("{base_dir}/divisor.report.bin"), &report);
        }
    }
}
