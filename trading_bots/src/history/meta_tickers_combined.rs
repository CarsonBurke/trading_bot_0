use crate::constants::files::TRAINING_PATH;
use crate::history::episode_tickers_combined::EpisodeHistory;
use crate::history::report::{
    read_report, write_report, Report, ReportKind, ReportSeries, ScaleKind,
};
use crate::utils::create_folder_if_not_exists;

#[derive(Default, Debug)]
pub struct MetaHistory {
    pub final_assets: Vec<f64>,
    pub cumulative_reward: Vec<f64>,
    pub outperformance: Vec<f64>,
    pub policy_loss: Vec<f64>,
    pub value_loss: Vec<f64>,
    pub explained_var: Vec<f64>,
    pub actor_grad_norm: Vec<f64>,
    pub critic_grad_norm: Vec<f64>,
    pub total_commissions: Vec<f64>,
    pub beta_alpha_mean: Vec<f64>,
    pub beta_action_mean: Vec<f64>,
    pub beta_beta_mean: Vec<f64>,
    pub beta_concentration_mean: Vec<f64>,
    pub mean_advantage: Vec<f64>,
    pub min_advantage: Vec<f64>,
    pub max_advantage: Vec<f64>,
    pub logit_scale: Vec<f64>,
    pub clip_fraction: Vec<f64>,
    pub clip_gap: Vec<f64>,
    pub temporal_tau: Vec<f64>,
    pub temporal_attn_entropy: Vec<f64>,
    pub temporal_attn_max: Vec<f64>,
    pub temporal_attn_eff_len: Vec<f64>,
    pub temporal_attn_center: Vec<f64>,
    pub temporal_attn_last_weight: Vec<f64>,
    pub policy_entropy_mean: Vec<f64>,
    pub policy_entropy_min: Vec<f64>,
    pub policy_entropy_max: Vec<f64>,
    pub approx_kl: Vec<f64>,
    pub kl_lr_scale: Vec<f64>,
    pub kl_lr_scale_next: Vec<f64>,
    pub kl_lr_ema: Vec<f64>,
    pub kl_lr_signal: Vec<f64>,
    pub gate_mean: Vec<f64>,
    pub gate_std: Vec<f64>,
    pub return_min: Vec<f64>,
    pub return_max: Vec<f64>,
    pub support_min: Vec<f64>,
    pub support_max: Vec<f64>,
    pub return_below_support_frac: Vec<f64>,
    pub return_above_support_frac: Vec<f64>,
}

impl MetaHistory {
    pub fn record(&mut self, history: &EpisodeHistory, outperformance: f64) {
        self.final_assets.push(history.final_assets());
        self.cumulative_reward
            .push(history.rewards.iter().sum::<f64>());
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

    pub fn record_grad_norm(&mut self, actor_grad_norm: f64, critic_grad_norm: f64) {
        self.actor_grad_norm.push(actor_grad_norm);
        self.critic_grad_norm.push(critic_grad_norm);
    }

    pub fn record_beta_policy_stats(
        &mut self,
        alpha_mean: f64,
        action_mean: f64,
        beta_mean: f64,
        concentration_mean: f64,
    ) {
        self.beta_alpha_mean.push(alpha_mean);
        self.beta_action_mean.push(action_mean);
        self.beta_beta_mean.push(beta_mean);
        self.beta_concentration_mean.push(concentration_mean);
    }

    pub fn record_advantage_stats(&mut self, mean: f64, min: f64, max: f64) {
        self.mean_advantage.push(mean);
        self.min_advantage.push(min);
        self.max_advantage.push(max);
    }

    pub fn record_clip_fraction(&mut self, clip_fraction: f64) {
        self.clip_fraction.push(clip_fraction);
    }

    pub fn record_clip_gap(&mut self, clip_gap: f64) {
        self.clip_gap.push(clip_gap);
    }

    pub fn record_policy_entropy(&mut self, mean: f64, min: f64, max: f64) {
        self.policy_entropy_mean.push(mean);
        self.policy_entropy_min.push(min);
        self.policy_entropy_max.push(max);
    }

    pub fn record_approx_kl(&mut self, kl: f64) {
        self.approx_kl.push(kl);
    }

    pub fn record_kl_lr(
        &mut self,
        lr_scale: f64,
        kl_lr_scale_next: f64,
        kl_lr_ema: f64,
        kl_lr_signal: f64,
    ) {
        self.kl_lr_scale.push(lr_scale);
        self.kl_lr_scale_next.push(kl_lr_scale_next);
        self.kl_lr_ema.push(kl_lr_ema);
        self.kl_lr_signal.push(kl_lr_signal);
    }

    pub fn record_gate_stats(&mut self, mean: f64, std: f64) {
        self.gate_mean.push(mean);
        self.gate_std.push(std);
    }

    pub fn record_hl_gauss_range_stats(
        &mut self,
        return_min: f64,
        return_max: f64,
        support_min: f64,
        support_max: f64,
        below_frac: f64,
        above_frac: f64,
    ) {
        self.return_min.push(return_min);
        self.return_max.push(return_max);
        self.support_min.push(support_min);
        self.support_max.push(support_max);
        self.return_below_support_frac.push(below_frac);
        self.return_above_support_frac.push(above_frac);
    }

    pub fn record_temporal_debug(
        &mut self,
        temporal_tau: f64,
        temporal_attn_entropy: f64,
        temporal_attn_max: f64,
        temporal_attn_eff_len: f64,
        temporal_attn_center: f64,
        temporal_attn_last_weight: f64,
    ) {
        self.temporal_tau.push(temporal_tau);
        self.temporal_attn_entropy.push(temporal_attn_entropy);
        self.temporal_attn_max.push(temporal_attn_max);
        self.temporal_attn_eff_len.push(temporal_attn_eff_len);
        self.temporal_attn_center.push(temporal_attn_center);
        self.temporal_attn_last_weight
            .push(temporal_attn_last_weight);
    }

    /// Load meta history from existing reports at the given episode
    pub fn load_from_episode(&mut self, episode: usize, gens_path: &str) {
        let base_dir = format!("{gens_path}/{}", episode);
        let load_simple = |path: &str| -> Vec<f64> {
            read_report(path)
                .ok()
                .map(|r| match r.kind {
                    ReportKind::Simple { values, .. } => {
                        values.into_iter().map(|v| v as f64).collect()
                    }
                    _ => vec![],
                })
                .unwrap_or_default()
        };
        let load_multiline = |path: &str, label: &str| -> Vec<f64> {
            read_report(path)
                .ok()
                .map(|r| match r.kind {
                    ReportKind::MultiLine { series } => series
                        .into_iter()
                        .find(|s| s.label == label)
                        .map(|s| s.values.into_iter().map(|v| v as f64).collect())
                        .unwrap_or_default(),
                    _ => vec![],
                })
                .unwrap_or_default()
        };

        self.final_assets = load_simple(&format!("{base_dir}/final_assets.report.bin"));
        self.cumulative_reward = load_simple(&format!("{base_dir}/cumulative_reward.report.bin"));
        self.outperformance = load_simple(&format!("{base_dir}/outperformance.report.bin"));
        self.policy_loss = load_simple(&format!("{base_dir}/policy_loss.report.bin"));
        self.value_loss = load_simple(&format!("{base_dir}/value_loss.report.bin"));
        self.explained_var = load_simple(&format!("{base_dir}/explained_var.report.bin"));
        self.actor_grad_norm = load_simple(&format!("{base_dir}/actor_grad_norm.report.bin"));
        self.critic_grad_norm = load_simple(&format!("{base_dir}/critic_grad_norm.report.bin"));
        self.total_commissions = load_simple(&format!("{base_dir}/total_commissions.report.bin"));
        self.logit_scale = load_simple(&format!("{base_dir}/logit_scale.report.bin"));
        self.clip_fraction = load_simple(&format!("{base_dir}/clip_fraction.report.bin"));
        if self.clip_fraction.is_empty() {
            self.clip_fraction = load_simple(&format!("{base_dir}/spo_bound_fraction.report.bin"));
        }
        self.clip_gap = load_simple(&format!("{base_dir}/clip_gap.report.bin"));

        // MultiLine reports
        let beta_policy_path = format!("{base_dir}/beta_policy.report.bin");
        self.beta_alpha_mean = load_multiline(&beta_policy_path, "alpha_mean");
        self.beta_action_mean = load_multiline(&beta_policy_path, "action_mean");
        self.beta_beta_mean = load_multiline(&beta_policy_path, "beta_mean");
        self.beta_concentration_mean = load_multiline(&beta_policy_path, "concentration");

        let adv_path = format!("{base_dir}/advantage_stats_log.report.bin");
        self.mean_advantage = load_multiline(&adv_path, "mean");
        self.min_advantage = load_multiline(&adv_path, "min");
        self.max_advantage = load_multiline(&adv_path, "max");

        let temporal_path = format!("{base_dir}/temporal_embed_debug.report.bin");
        self.temporal_tau = load_multiline(&temporal_path, "temporal_tau");
        self.temporal_attn_entropy = load_multiline(&temporal_path, "temporal_entropy");
        self.temporal_attn_max = load_multiline(&temporal_path, "temporal_attn_max");
        self.temporal_attn_eff_len = load_multiline(&temporal_path, "temporal_eff_len");
        self.temporal_attn_center = load_multiline(&temporal_path, "temporal_attn_center");
        self.temporal_attn_last_weight = load_multiline(&temporal_path, "temporal_attn_last");

        let entropy_path = format!("{base_dir}/policy_entropy.report.bin");
        self.policy_entropy_mean = load_multiline(&entropy_path, "mean");
        self.policy_entropy_min = load_multiline(&entropy_path, "min");
        self.policy_entropy_max = load_multiline(&entropy_path, "max");

        self.approx_kl = load_simple(&format!("{base_dir}/approx_kl.report.bin"));
        let kl_lr_path = format!("{base_dir}/kl_lr.report.bin");
        self.kl_lr_scale = load_multiline(&kl_lr_path, "lr_scale");
        self.kl_lr_scale_next = load_multiline(&kl_lr_path, "scale_next");
        self.kl_lr_ema = load_multiline(&kl_lr_path, "ema");
        self.kl_lr_signal = load_multiline(&kl_lr_path, "signal");
        let gate_path = format!("{base_dir}/gate_stats.report.bin");
        self.gate_mean = load_multiline(&gate_path, "mean");
        self.gate_std = load_multiline(&gate_path, "std");
        let hl_gauss_path = format!("{base_dir}/hl_gauss_return_range.report.bin");
        self.return_min = load_multiline(&hl_gauss_path, "return_min");
        self.return_max = load_multiline(&hl_gauss_path, "return_max");
        self.support_min = load_multiline(&hl_gauss_path, "support_min");
        self.support_max = load_multiline(&hl_gauss_path, "support_max");
        self.return_below_support_frac = load_multiline(&hl_gauss_path, "below_frac");
        self.return_above_support_frac = load_multiline(&hl_gauss_path, "above_frac");

        println!(
            "Loaded meta history from episode {} ({} data points)",
            episode,
            self.final_assets.len()
        );
    }

    fn report(
        title: &str,
        x_label: &str,
        y_label: Option<&str>,
        scale: ScaleKind,
        kind: ReportKind,
    ) -> Report {
        Report {
            title: title.to_string(),
            x_label: Some(x_label.to_string()),
            y_label: y_label.map(|s| s.to_string()),
            scale,
            kind,
        }
    }

    pub fn write_reports_default(&self, episode: usize) {
        self.write_reports(episode, &format!("{TRAINING_PATH}/gens"));
    }

    pub fn write_reports(&self, episode: usize, gens_path: &str) {
        let base_dir = format!("{gens_path}/{}", episode);
        create_folder_if_not_exists(&base_dir);
        let simple = |vals: &[f64]| ReportKind::Simple {
            values: f64_to_f32(vals),
            ema_alpha: Some(0.05),
        };
        if !self.final_assets.is_empty() {
            let r = Self::report(
                "Final Assets",
                "Episode",
                Some("Assets"),
                ScaleKind::Linear,
                simple(&self.final_assets),
            );
            let _ = write_report(&format!("{base_dir}/final_assets.report.bin"), &r);
        }
        if !self.cumulative_reward.is_empty() {
            let r = Self::report(
                "Cumulative Reward",
                "Episode",
                Some("Reward"),
                ScaleKind::Linear,
                simple(&self.cumulative_reward),
            );
            let _ = write_report(&format!("{base_dir}/cumulative_reward.report.bin"), &r);
        }
        if !self.outperformance.is_empty() {
            let r = Self::report(
                "Outperformance",
                "Episode",
                Some("Outperformance"),
                ScaleKind::Linear,
                simple(&self.outperformance),
            );
            let _ = write_report(&format!("{base_dir}/outperformance.report.bin"), &r);
        }
        if !self.policy_loss.is_empty() {
            let r = Self::report(
                "Policy Loss",
                "Episode",
                Some("Loss"),
                ScaleKind::Linear,
                simple(&self.policy_loss),
            );
            let _ = write_report(&format!("{base_dir}/policy_loss.report.bin"), &r);
        }
        if !self.value_loss.is_empty() {
            let r = Self::report(
                "Value Loss",
                "Episode",
                Some("Loss"),
                ScaleKind::Linear,
                simple(&self.value_loss),
            );
            let _ = write_report(&format!("{base_dir}/value_loss.report.bin"), &r);
        }
        if !self.explained_var.is_empty() {
            let r = Self::report(
                "Explained Variance",
                "Episode",
                Some("EV"),
                ScaleKind::Linear,
                simple(&self.explained_var),
            );
            let _ = write_report(&format!("{base_dir}/explained_var.report.bin"), &r);
        }
        if !self.actor_grad_norm.is_empty() {
            let r = Self::report(
                "Actor Grad Norm",
                "Episode",
                Some("Grad Norm"),
                ScaleKind::Linear,
                simple(&self.actor_grad_norm),
            );
            let _ = write_report(&format!("{base_dir}/actor_grad_norm.report.bin"), &r);
        }
        if !self.critic_grad_norm.is_empty() {
            let r = Self::report(
                "Critic Grad Norm",
                "Episode",
                Some("Grad Norm"),
                ScaleKind::Linear,
                simple(&self.critic_grad_norm),
            );
            let _ = write_report(&format!("{base_dir}/critic_grad_norm.report.bin"), &r);
        }
        if !self.total_commissions.is_empty() {
            let r = Self::report(
                "Total Commissions",
                "Episode",
                Some("Commissions"),
                ScaleKind::Linear,
                simple(&self.total_commissions),
            );
            let _ = write_report(&format!("{base_dir}/total_commissions.report.bin"), &r);
        }
        if !self.beta_alpha_mean.is_empty() {
            let r = Self::report(
                "Beta Policy",
                "Episode",
                None,
                ScaleKind::Linear,
                ReportKind::MultiLine {
                    series: vec![
                        ReportSeries {
                            label: "alpha_mean".to_string(),
                            values: f64_to_f32(&self.beta_alpha_mean),
                        },
                        ReportSeries {
                            label: "action_mean".to_string(),
                            values: f64_to_f32(&self.beta_action_mean),
                        },
                        ReportSeries {
                            label: "beta_mean".to_string(),
                            values: f64_to_f32(&self.beta_beta_mean),
                        },
                        ReportSeries {
                            label: "concentration".to_string(),
                            values: f64_to_f32(&self.beta_concentration_mean),
                        },
                    ],
                },
            );
            let _ = write_report(&format!("{base_dir}/beta_policy.report.bin"), &r);
        }
        if !self.mean_advantage.is_empty() {
            let r = Self::report(
                "Advantage Stats (Log)",
                "Episode",
                None,
                ScaleKind::Linear,
                ReportKind::MultiLine {
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
            );
            let _ = write_report(&format!("{base_dir}/advantage_stats_log.report.bin"), &r);
        }
        if !self.logit_scale.is_empty() {
            let r = Self::report(
                "Logit Scale",
                "Episode",
                Some("Scale"),
                ScaleKind::Linear,
                simple(&self.logit_scale),
            );
            let _ = write_report(&format!("{base_dir}/logit_scale.report.bin"), &r);
        }
        if !self.clip_fraction.is_empty() {
            let r = Self::report(
                "Clip Fraction",
                "Episode",
                Some("Fraction"),
                ScaleKind::Linear,
                simple(&self.clip_fraction),
            );
            let _ = write_report(&format!("{base_dir}/clip_fraction.report.bin"), &r);
        }
        if !self.clip_gap.is_empty() {
            let r = Self::report(
                "Clip Gap",
                "Episode",
                Some("Gap"),
                ScaleKind::Linear,
                simple(&self.clip_gap),
            );
            let _ = write_report(&format!("{base_dir}/clip_gap.report.bin"), &r);
        }
        if !self.approx_kl.is_empty() {
            let r = Self::report(
                "Policy KL",
                "Episode",
                Some("KL"),
                ScaleKind::Linear,
                simple(&self.approx_kl),
            );
            let _ = write_report(&format!("{base_dir}/approx_kl.report.bin"), &r);
        }
        if !self.kl_lr_scale.is_empty() {
            let r = Self::report(
                "KL-Adaptive LR",
                "Episode",
                None,
                ScaleKind::Linear,
                ReportKind::MultiLine {
                    series: vec![
                        ReportSeries {
                            label: "lr_scale".to_string(),
                            values: f64_to_f32(&self.kl_lr_scale),
                        },
                        ReportSeries {
                            label: "scale_next".to_string(),
                            values: f64_to_f32(&self.kl_lr_scale_next),
                        },
                        ReportSeries {
                            label: "ema".to_string(),
                            values: f64_to_f32(&self.kl_lr_ema),
                        },
                        ReportSeries {
                            label: "signal".to_string(),
                            values: f64_to_f32(&self.kl_lr_signal),
                        },
                    ],
                },
            );
            let _ = write_report(&format!("{base_dir}/kl_lr.report.bin"), &r);
        }
        if !self.policy_entropy_mean.is_empty() {
            let r = Self::report(
                "Policy Entropy",
                "Episode",
                Some("Entropy (nats)"),
                ScaleKind::Linear,
                ReportKind::MultiLine {
                    series: vec![
                        ReportSeries {
                            label: "mean".to_string(),
                            values: f64_to_f32(&self.policy_entropy_mean),
                        },
                        ReportSeries {
                            label: "min".to_string(),
                            values: f64_to_f32(&self.policy_entropy_min),
                        },
                        ReportSeries {
                            label: "max".to_string(),
                            values: f64_to_f32(&self.policy_entropy_max),
                        },
                    ],
                },
            );
            let _ = write_report(&format!("{base_dir}/policy_entropy.report.bin"), &r);
        }
        if !self.temporal_tau.is_empty() {
            let r = Self::report(
                "Temporal/Embed Debug",
                "Episode",
                None,
                ScaleKind::Linear,
                ReportKind::MultiLine {
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
                    ],
                },
            );
            let _ = write_report(&format!("{base_dir}/temporal_embed_debug.report.bin"), &r);
        }
        if !self.gate_mean.is_empty() {
            let r = Self::report(
                "Gate Stats",
                "Episode",
                None,
                ScaleKind::Linear,
                ReportKind::MultiLine {
                    series: vec![
                        ReportSeries {
                            label: "mean".to_string(),
                            values: f64_to_f32(&self.gate_mean),
                        },
                        ReportSeries {
                            label: "std".to_string(),
                            values: f64_to_f32(&self.gate_std),
                        },
                    ],
                },
            );
            let _ = write_report(&format!("{base_dir}/gate_stats.report.bin"), &r);
        }
        if !self.return_min.is_empty() {
            let r = Self::report(
                "HL-Gauss Return Range",
                "Episode",
                None,
                ScaleKind::Linear,
                ReportKind::MultiLine {
                    series: vec![
                        ReportSeries {
                            label: "return_min".to_string(),
                            values: f64_to_f32(&self.return_min),
                        },
                        ReportSeries {
                            label: "return_max".to_string(),
                            values: f64_to_f32(&self.return_max),
                        },
                        ReportSeries {
                            label: "support_min".to_string(),
                            values: f64_to_f32(&self.support_min),
                        },
                        ReportSeries {
                            label: "support_max".to_string(),
                            values: f64_to_f32(&self.support_max),
                        },
                        ReportSeries {
                            label: "below_frac".to_string(),
                            values: f64_to_f32(&self.return_below_support_frac),
                        },
                        ReportSeries {
                            label: "above_frac".to_string(),
                            values: f64_to_f32(&self.return_above_support_frac),
                        },
                    ],
                },
            );
            let _ = write_report(&format!("{base_dir}/hl_gauss_return_range.report.bin"), &r);
        }
    }
}

fn f64_to_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|v| *v as f32).collect()
}
