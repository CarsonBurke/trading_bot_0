use crate::constants::files::TRAINING_PATH;
use crate::history::episode_tickers_combined::EpisodeHistory;
use crate::history::report::{read_report, write_report, Report, ReportKind, ReportSeries, ScaleKind};
use crate::utils::create_folder_if_not_exists;

#[derive(Default, Debug)]
pub struct MetaHistory {
    pub final_assets: Vec<f64>,
    pub cumulative_reward: Vec<f64>,
    pub outperformance: Vec<f64>,
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

    /// Load meta history from existing reports at the given episode
    pub fn load_from_episode(&mut self, episode: usize) {
        let base_dir = format!("{TRAINING_PATH}/gens/{}", episode);
        let load_simple = |path: &str| -> Vec<f64> {
            read_report(path).ok().map(|r| match r.kind {
                ReportKind::Simple { values, .. } => values.into_iter().map(|v| v as f64).collect(),
                _ => vec![],
            }).unwrap_or_default()
        };
        let load_multiline = |path: &str, label: &str| -> Vec<f64> {
            read_report(path).ok().map(|r| match r.kind {
                ReportKind::MultiLine { series } => {
                    series.into_iter()
                        .find(|s| s.label == label)
                        .map(|s| s.values.into_iter().map(|v| v as f64).collect())
                        .unwrap_or_default()
                }
                _ => vec![],
            }).unwrap_or_default()
        };

        self.final_assets = load_simple(&format!("{base_dir}/final_assets.report.bin"));
        self.cumulative_reward = load_simple(&format!("{base_dir}/cumulative_reward.report.bin"));
        self.outperformance = load_simple(&format!("{base_dir}/outperformance.report.bin"));
        self.policy_loss = load_simple(&format!("{base_dir}/policy_loss.report.bin"));
        self.value_loss = load_simple(&format!("{base_dir}/value_loss.report.bin"));
        self.explained_var = load_simple(&format!("{base_dir}/explained_var.report.bin"));
        self.grad_norm = load_simple(&format!("{base_dir}/grad_norm_log.report.bin"));
        self.total_commissions = load_simple(&format!("{base_dir}/total_commissions.report.bin"));
        self.logit_scale = load_simple(&format!("{base_dir}/logit_scale.report.bin"));
        self.clip_fraction = load_simple(&format!("{base_dir}/clip_fraction.report.bin"));

        // MultiLine reports
        let logit_path = format!("{base_dir}/logit_noise.report.bin");
        self.logit_noise_mean = load_multiline(&logit_path, "mean");
        self.logit_noise_min = load_multiline(&logit_path, "min");
        self.logit_noise_max = load_multiline(&logit_path, "max");
        self.rpo_alpha = load_multiline(&logit_path, "rpo_alpha");

        let adv_path = format!("{base_dir}/advantage_stats_log.report.bin");
        self.mean_advantage = load_multiline(&adv_path, "mean");
        self.min_advantage = load_multiline(&adv_path, "min");
        self.max_advantage = load_multiline(&adv_path, "max");

        let alpha_path = format!("{base_dir}/time_cross_alpha_means.report.bin");
        self.time_alpha_attn_mean = load_multiline(&alpha_path, "time_attn");
        self.time_alpha_mlp_mean = load_multiline(&alpha_path, "time_mlp");
        self.cross_alpha_attn_mean = load_multiline(&alpha_path, "cross_attn");
        self.cross_alpha_mlp_mean = load_multiline(&alpha_path, "cross_mlp");

        let temporal_path = format!("{base_dir}/temporal_embed_debug.report.bin");
        self.temporal_tau = load_multiline(&temporal_path, "temporal_tau");
        self.temporal_attn_entropy = load_multiline(&temporal_path, "temporal_entropy");
        self.temporal_attn_max = load_multiline(&temporal_path, "temporal_attn_max");
        self.temporal_attn_eff_len = load_multiline(&temporal_path, "temporal_eff_len");
        self.temporal_attn_center = load_multiline(&temporal_path, "temporal_attn_center");
        self.temporal_attn_last_weight = load_multiline(&temporal_path, "temporal_attn_last");
        self.cross_ticker_embed_norm = load_multiline(&temporal_path, "cross_embed_norm");

        println!("Loaded meta history from episode {} ({} data points)", episode, self.final_assets.len());
    }

    fn report(title: &str, x_label: &str, y_label: Option<&str>, scale: ScaleKind, kind: ReportKind) -> Report {
        Report {
            title: title.to_string(),
            x_label: Some(x_label.to_string()),
            y_label: y_label.map(|s| s.to_string()),
            scale,
            kind,
        }
    }

    pub fn write_reports(&self, episode: usize) {
        let base_dir = format!("{TRAINING_PATH}/gens/{}", episode);
        create_folder_if_not_exists(&base_dir);
        let simple = |vals: &[f64]| ReportKind::Simple {
            values: f64_to_f32(vals),
            ema_alpha: Some(0.05),
        };
        if !self.final_assets.is_empty() {
            let r = Self::report("Final Assets", "Episode", Some("Assets"), ScaleKind::Linear, simple(&self.final_assets));
            let _ = write_report(&format!("{base_dir}/final_assets.report.bin"), &r);
        }
        if !self.cumulative_reward.is_empty() {
            let r = Self::report("Cumulative Reward", "Episode", Some("Reward"), ScaleKind::Linear, simple(&self.cumulative_reward));
            let _ = write_report(&format!("{base_dir}/cumulative_reward.report.bin"), &r);
        }
        if !self.outperformance.is_empty() {
            let r = Self::report("Outperformance", "Episode", Some("Outperformance"), ScaleKind::Linear, simple(&self.outperformance));
            let _ = write_report(&format!("{base_dir}/outperformance.report.bin"), &r);
        }
        if !self.policy_loss.is_empty() {
            let r = Self::report("Policy Loss", "Episode", Some("Loss"), ScaleKind::Linear, simple(&self.policy_loss));
            let _ = write_report(&format!("{base_dir}/policy_loss.report.bin"), &r);
        }
        if !self.value_loss.is_empty() {
            let r = Self::report("Value Loss", "Episode", Some("Loss"), ScaleKind::Linear, simple(&self.value_loss));
            let _ = write_report(&format!("{base_dir}/value_loss.report.bin"), &r);
        }
        if !self.explained_var.is_empty() {
            let r = Self::report("Explained Variance", "Episode", Some("EV"), ScaleKind::Linear, simple(&self.explained_var));
            let _ = write_report(&format!("{base_dir}/explained_var.report.bin"), &r);
        }
        if !self.grad_norm.is_empty() {
            let r = Self::report("Grad Norm (Log)", "Episode", Some("Grad Norm"), ScaleKind::Linear, simple(&self.grad_norm));
            let _ = write_report(&format!("{base_dir}/grad_norm_log.report.bin"), &r);
        }
        if !self.total_commissions.is_empty() {
            let r = Self::report("Total Commissions", "Episode", Some("Commissions"), ScaleKind::Linear, simple(&self.total_commissions));
            let _ = write_report(&format!("{base_dir}/total_commissions.report.bin"), &r);
        }
        if !self.logit_noise_mean.is_empty() {
            let r = Self::report("Logit Noise", "Episode", None, ScaleKind::Linear, ReportKind::MultiLine {
                series: vec![
                    ReportSeries { label: "mean".to_string(), values: f64_to_f32(&self.logit_noise_mean) },
                    ReportSeries { label: "min".to_string(), values: f64_to_f32(&self.logit_noise_min) },
                    ReportSeries { label: "max".to_string(), values: f64_to_f32(&self.logit_noise_max) },
                    ReportSeries { label: "rpo_alpha".to_string(), values: f64_to_f32(&self.rpo_alpha) },
                ],
            });
            let _ = write_report(&format!("{base_dir}/logit_noise.report.bin"), &r);
        }
        if !self.mean_advantage.is_empty() {
            let r = Self::report("Advantage Stats (Log)", "Episode", None, ScaleKind::Symlog, ReportKind::MultiLine {
                series: vec![
                    ReportSeries { label: "mean".to_string(), values: f64_to_f32(&self.mean_advantage) },
                    ReportSeries { label: "min".to_string(), values: f64_to_f32(&self.min_advantage) },
                    ReportSeries { label: "max".to_string(), values: f64_to_f32(&self.max_advantage) },
                ],
            });
            let _ = write_report(&format!("{base_dir}/advantage_stats_log.report.bin"), &r);
        }
        if !self.logit_scale.is_empty() {
            let r = Self::report("Logit Scale", "Episode", Some("Scale"), ScaleKind::Linear, simple(&self.logit_scale));
            let _ = write_report(&format!("{base_dir}/logit_scale.report.bin"), &r);
        }
        if !self.clip_fraction.is_empty() {
            let r = Self::report("Clip Fraction", "Episode", Some("Fraction"), ScaleKind::Linear, simple(&self.clip_fraction));
            let _ = write_report(&format!("{base_dir}/clip_fraction.report.bin"), &r);
        }
        if !self.time_alpha_attn_mean.is_empty() {
            let r = Self::report("Time/Cross Alpha Means", "Episode", Some("Alpha"), ScaleKind::Linear, ReportKind::MultiLine {
                series: vec![
                    ReportSeries { label: "time_attn".to_string(), values: f64_to_f32(&self.time_alpha_attn_mean) },
                    ReportSeries { label: "time_mlp".to_string(), values: f64_to_f32(&self.time_alpha_mlp_mean) },
                    ReportSeries { label: "cross_attn".to_string(), values: f64_to_f32(&self.cross_alpha_attn_mean) },
                    ReportSeries { label: "cross_mlp".to_string(), values: f64_to_f32(&self.cross_alpha_mlp_mean) },
                ],
            });
            let _ = write_report(&format!("{base_dir}/time_cross_alpha_means.report.bin"), &r);
        }
        if !self.temporal_tau.is_empty() {
            let r = Self::report("Temporal/Embed Debug", "Episode", None, ScaleKind::Linear, ReportKind::MultiLine {
                series: vec![
                    ReportSeries { label: "temporal_tau".to_string(), values: f64_to_f32(&self.temporal_tau) },
                    ReportSeries { label: "temporal_entropy".to_string(), values: f64_to_f32(&self.temporal_attn_entropy) },
                    ReportSeries { label: "temporal_attn_max".to_string(), values: f64_to_f32(&self.temporal_attn_max) },
                    ReportSeries { label: "temporal_eff_len".to_string(), values: f64_to_f32(&self.temporal_attn_eff_len) },
                    ReportSeries { label: "temporal_attn_center".to_string(), values: f64_to_f32(&self.temporal_attn_center) },
                    ReportSeries { label: "temporal_attn_last".to_string(), values: f64_to_f32(&self.temporal_attn_last_weight) },
                    ReportSeries { label: "cross_embed_norm".to_string(), values: f64_to_f32(&self.cross_ticker_embed_norm) },
                ],
            });
            let _ = write_report(&format!("{base_dir}/temporal_embed_debug.report.bin"), &r);
        }
    }
}

fn f64_to_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|v| *v as f32).collect()
}
