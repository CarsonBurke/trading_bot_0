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
    pub policy_loss: Vec<f64>,
    pub value_loss: Vec<f64>,
    pub explained_var: Vec<f64>,
    pub grad_norm: Vec<f64>,
    pub total_commissions: Vec<f64>,
    pub logit_noise_mean: Vec<f64>,
    pub logit_noise_min: Vec<f64>,
    pub logit_noise_max: Vec<f64>,
    pub rpo_alpha: Vec<f64>,
    pub mean_advantage: Vec<f64>,
    pub min_advantage: Vec<f64>,
    pub max_advantage: Vec<f64>,
    pub logit_scale: Vec<f64>,
    pub clip_fraction: Vec<f64>,
    pub time_alpha_attn_mean: Vec<f64>,
    pub time_alpha_mlp_mean: Vec<f64>,
    pub cross_alpha_attn_mean: Vec<f64>,
    pub cross_alpha_mlp_mean: Vec<f64>,
    pub temporal_tau: Vec<f64>,
    pub temporal_attn_entropy: Vec<f64>,
    pub temporal_attn_max: Vec<f64>,
    pub temporal_attn_eff_len: Vec<f64>,
    pub temporal_attn_center: Vec<f64>,
    pub temporal_attn_last_weight: Vec<f64>,
    pub cross_ticker_embed_norm: Vec<f64>,
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

    pub fn record_policy_loss(&mut self, loss: f64) {
        self.policy_loss.push(loss);
    }

    pub fn record_value_loss(&mut self, loss: f64) {
        self.value_loss.push(loss);
    }

    pub fn record_explained_var(&mut self, ev: f64) {
        self.explained_var.push(ev);
    }

    pub fn record_grad_norm(&mut self, grad_norm: f64) {
        self.grad_norm.push(grad_norm);
    }

    pub fn record_logit_noise_stats(&mut self, mean: f64, min: f64, max: f64, rpo_alpha: f64) {
        self.logit_noise_mean.push(mean);
        self.logit_noise_min.push(min);
        self.logit_noise_max.push(max);
        self.rpo_alpha.push(rpo_alpha);
    }

    pub fn record_advantage_stats(&mut self, mean: f64, min: f64, max: f64) {
        self.mean_advantage.push(mean);
        self.min_advantage.push(min);
        self.max_advantage.push(max);
    }

    pub fn record_logit_scale(&mut self, logit_scale: f64) {
        self.logit_scale.push(logit_scale);
    }

    pub fn record_clip_fraction(&mut self, clip_fraction: f64) {
        self.clip_fraction.push(clip_fraction);
    }

    pub fn record_temporal_debug(
        &mut self,
        time_alpha_attn_mean: f64,
        time_alpha_mlp_mean: f64,
        cross_alpha_attn_mean: f64,
        cross_alpha_mlp_mean: f64,
        temporal_tau: f64,
        temporal_attn_entropy: f64,
        temporal_attn_max: f64,
        temporal_attn_eff_len: f64,
        temporal_attn_center: f64,
        temporal_attn_last_weight: f64,
        cross_ticker_embed_norm: f64,
    ) {
        self.time_alpha_attn_mean.push(time_alpha_attn_mean);
        self.time_alpha_mlp_mean.push(time_alpha_mlp_mean);
        self.cross_alpha_attn_mean.push(cross_alpha_attn_mean);
        self.cross_alpha_mlp_mean.push(cross_alpha_mlp_mean);
        self.temporal_tau.push(temporal_tau);
        self.temporal_attn_entropy.push(temporal_attn_entropy);
        self.temporal_attn_max.push(temporal_attn_max);
        self.temporal_attn_eff_len.push(temporal_attn_eff_len);
        self.temporal_attn_center.push(temporal_attn_center);
        self.temporal_attn_last_weight.push(temporal_attn_last_weight);
        self.cross_ticker_embed_norm.push(cross_ticker_embed_norm);
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
                    values: f64_to_f32(&self.final_assets),
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
                    values: f64_to_f32(&self.cumulative_reward),
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
                    values: f64_to_f32(&self.outperformance),
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
                kind: ReportKind::MultiLine {
                    series: vec![
                        ReportSeries {
                            label: "total".to_string(),
                            values: f64_to_f32(&self.loss),
                        },
                        ReportSeries {
                            label: "policy".to_string(),
                            values: f64_to_f32(&self.policy_loss),
                        },
                        ReportSeries {
                            label: "value".to_string(),
                            values: f64_to_f32(&self.value_loss),
                        },
                    ],
                },
            };
            let _ = write_report(&format!("{base_dir}/loss_log.report.bin"), &report);
        }
        if !self.explained_var.is_empty() {
            let report = Report {
                title: "Explained Variance".to_string(),
                x_label: Some("Episode".to_string()),
                y_label: Some("EV".to_string()),
                scale: ScaleKind::Linear,
                kind: ReportKind::Simple {
                    values: f64_to_f32(&self.explained_var),
                    ema_alpha: Some(0.05),
                },
            };
            let _ = write_report(&format!("{base_dir}/explained_var.report.bin"), &report);
        }
        if !self.grad_norm.is_empty() {
            let report = Report {
                title: "Grad Norm (Log)".to_string(),
                x_label: Some("Episode".to_string()),
                y_label: Some("Grad Norm".to_string()),
                scale: ScaleKind::Symlog,
                kind: ReportKind::Simple {
                    values: f64_to_f32(&self.grad_norm),
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
                    values: f64_to_f32(&self.total_commissions),
                    ema_alpha: Some(0.05),
                },
            };
            let _ = write_report(&format!("{base_dir}/total_commissions.report.bin"), &report);
        }
        if !self.logit_noise_mean.is_empty() {
            let report = Report {
                title: "Logit Noise".to_string(),
                x_label: Some("Episode".to_string()),
                y_label: None,
                scale: ScaleKind::Linear,
                kind: ReportKind::MultiLine {
                    series: vec![
                        ReportSeries {
                            label: "mean".to_string(),
                            values: f64_to_f32(&self.logit_noise_mean),
                        },
                        ReportSeries {
                            label: "min".to_string(),
                            values: f64_to_f32(&self.logit_noise_min),
                        },
                        ReportSeries {
                            label: "max".to_string(),
                            values: f64_to_f32(&self.logit_noise_max),
                        },
                        ReportSeries {
                            label: "rpo_alpha".to_string(),
                            values: f64_to_f32(&self.rpo_alpha),
                        },
                    ],
                },
            };
            let _ = write_report(&format!("{base_dir}/logit_noise.report.bin"), &report);
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
                            values: f64_to_f32(&self.mean_advantage),
                        },
                        ReportSeries {
                            label: "min".to_string(),
                            values: f64_to_f32(&self.min_advantage),
                        },
                        ReportSeries {
                            label: "max".to_string(),
                            values: f64_to_f32(&self.max_advantage),
                        },
                    ],
                },
            };
            let _ = write_report(&format!("{base_dir}/advantage_stats_log.report.bin"), &report);
        }
        if !self.logit_scale.is_empty() {
            let report = Report {
                title: "Logit Scale".to_string(),
                x_label: Some("Episode".to_string()),
                y_label: Some("Scale".to_string()),
                scale: ScaleKind::Linear,
                kind: ReportKind::Simple {
                    values: f64_to_f32(&self.logit_scale),
                    ema_alpha: Some(0.05),
                },
            };
            let _ = write_report(&format!("{base_dir}/logit_scale.report.bin"), &report);
        }
        if !self.clip_fraction.is_empty() {
            let report = Report {
                title: "Clip Fraction".to_string(),
                x_label: Some("Episode".to_string()),
                y_label: Some("Fraction".to_string()),
                scale: ScaleKind::Linear,
                kind: ReportKind::Simple {
                    values: f64_to_f32(&self.clip_fraction),
                    ema_alpha: Some(0.05),
                },
            };
            let _ = write_report(&format!("{base_dir}/clip_fraction.report.bin"), &report);
        }
        if !self.time_alpha_attn_mean.is_empty() {
            let report = Report {
                title: "Time/Cross Alpha Means".to_string(),
                x_label: Some("Episode".to_string()),
                y_label: Some("Alpha".to_string()),
                scale: ScaleKind::Linear,
                kind: ReportKind::MultiLine {
                    series: vec![
                        ReportSeries {
                            label: "time_attn".to_string(),
                            values: f64_to_f32(&self.time_alpha_attn_mean),
                        },
                        ReportSeries {
                            label: "time_mlp".to_string(),
                            values: f64_to_f32(&self.time_alpha_mlp_mean),
                        },
                        ReportSeries {
                            label: "cross_attn".to_string(),
                            values: f64_to_f32(&self.cross_alpha_attn_mean),
                        },
                        ReportSeries {
                            label: "cross_mlp".to_string(),
                            values: f64_to_f32(&self.cross_alpha_mlp_mean),
                        },
                    ],
                },
            };
            let _ = write_report(&format!("{base_dir}/time_cross_alpha_means.report.bin"), &report);
        }
        if !self.temporal_tau.is_empty() {
            let report = Report {
                title: "Temporal/Embed Debug".to_string(),
                x_label: Some("Episode".to_string()),
                y_label: None,
                scale: ScaleKind::Linear,
                kind: ReportKind::MultiLine {
                    series: vec![
                        ReportSeries {
                            label: "temporal_tau".to_string(),
                            values: f64_to_f32(&self.temporal_tau),
                        },
                        ReportSeries {
                            label: "temporal_entropy".to_string(),
                            values: f64_to_f32(&self.temporal_attn_entropy),
                        },
                        ReportSeries {
                            label: "temporal_attn_max".to_string(),
                            values: f64_to_f32(&self.temporal_attn_max),
                        },
                        ReportSeries {
                            label: "temporal_eff_len".to_string(),
                            values: f64_to_f32(&self.temporal_attn_eff_len),
                        },
                        ReportSeries {
                            label: "temporal_attn_center".to_string(),
                            values: f64_to_f32(&self.temporal_attn_center),
                        },
                        ReportSeries {
                            label: "temporal_attn_last".to_string(),
                            values: f64_to_f32(&self.temporal_attn_last_weight),
                        },
                        ReportSeries {
                            label: "cross_embed_norm".to_string(),
                            values: f64_to_f32(&self.cross_ticker_embed_norm),
                        },
                    ],
                },
            };
            let _ = write_report(&format!("{base_dir}/temporal_embed_debug.report.bin"), &report);
        }
    }
}

fn f64_to_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|v| *v as f32).collect()
}
