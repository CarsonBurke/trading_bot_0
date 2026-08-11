use crate::constants::files::TRAINING_PATH;
use crate::history::episode_tickers_combined::EpisodeHistory;
use crate::history::report::{
    read_report, write_report, Report, ReportKind, ReportSeries, ScaleKind,
};
use anyhow::{bail, Context, Result};
use std::io;
use std::path::Path;

macro_rules! assign_multiline {
    ($history:expr, $values:expr, $($field:ident),+ $(,)?) => {{
        let mut values = $values.into_iter();
        $(
            $history.$field = values
                .next()
                .expect("validated multiline report returned too few series");
        )+
        debug_assert!(values.next().is_none());
    }};
}

#[derive(Clone, Copy)]
enum MissingMetric {
    Required,
    AlignedNan,
    Inactive,
}

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

    /// Load a complete, schema-validated snapshot before mutating this history.
    pub fn load_from_episode(&mut self, episode: usize, gens_path: &str) -> Result<()> {
        let base_dir = Path::new(gens_path).join(episode.to_string());
        let expected = episode
            .checked_add(1)
            .context("meta history episode length overflowed")?;
        let mut loaded = Self::default();

        loaded.final_assets = load_simple(&base_dir, "final_assets", expected, true)?;
        loaded.cumulative_reward = load_simple(&base_dir, "cumulative_reward", expected, true)?;
        loaded.outperformance = load_simple(&base_dir, "outperformance", expected, true)?;
        loaded.total_commissions = load_simple(&base_dir, "total_commissions", expected, true)?;
        loaded.policy_loss = load_simple(&base_dir, "policy_loss", expected, true)?;
        loaded.value_loss = load_simple(&base_dir, "value_loss", expected, true)?;
        loaded.explained_var = load_simple(&base_dir, "explained_var", expected, true)?;
        loaded.actor_grad_norm = load_simple(&base_dir, "actor_grad_norm", expected, true)?;
        loaded.critic_grad_norm = load_simple(&base_dir, "critic_grad_norm", expected, true)?;
        // Validate historical inactive diagnostics if present, but do not carry them
        // forward: no current recorder appends to these fields, so retaining a
        // finite-length vector would create a stale, unresumable report later.
        let _ = load_simple_if_present(&base_dir, "logit_scale", expected)?;
        loaded.logit_scale = Vec::new();
        loaded.clip_gap = load_simple(&base_dir, "clip_gap", expected, false)?;
        loaded.approx_kl = load_simple(&base_dir, "approx_kl", expected, true)?;

        loaded.clip_fraction = match load_simple_if_present(&base_dir, "clip_fraction", expected)? {
            Some(values) => values,
            None => load_simple_if_present(&base_dir, "spo_bound_fraction", expected)?
                .unwrap_or_else(|| missing_history(expected)),
        };

        assign_multiline!(
            loaded,
            load_multiline(
                &base_dir,
                "beta_policy",
                &["alpha_mean", "action_mean", "beta_mean", "concentration"],
                expected,
                MissingMetric::Required,
            )?,
            beta_alpha_mean,
            beta_action_mean,
            beta_beta_mean,
            beta_concentration_mean
        );
        assign_multiline!(
            loaded,
            load_multiline(
                &base_dir,
                "advantage_stats_log",
                &["mean", "min", "max"],
                expected,
                MissingMetric::Required,
            )?,
            mean_advantage,
            min_advantage,
            max_advantage
        );
        assign_multiline!(
            loaded,
            load_multiline(
                &base_dir,
                "temporal_embed_debug",
                &[
                    "temporal_tau",
                    "temporal_entropy",
                    "temporal_attn_max",
                    "temporal_eff_len",
                    "temporal_attn_center",
                    "temporal_attn_last",
                ],
                expected,
                MissingMetric::Inactive,
            )?,
            temporal_tau,
            temporal_attn_entropy,
            temporal_attn_max,
            temporal_attn_eff_len,
            temporal_attn_center,
            temporal_attn_last_weight
        );
        assign_multiline!(
            loaded,
            load_multiline(
                &base_dir,
                "policy_entropy",
                &["mean", "min", "max"],
                expected,
                MissingMetric::Required,
            )?,
            policy_entropy_mean,
            policy_entropy_min,
            policy_entropy_max
        );
        assign_multiline!(
            loaded,
            load_multiline(
                &base_dir,
                "kl_lr",
                &["lr_scale", "scale_next", "ema", "signal"],
                expected,
                MissingMetric::AlignedNan,
            )?,
            kl_lr_scale,
            kl_lr_scale_next,
            kl_lr_ema,
            kl_lr_signal
        );
        assign_multiline!(
            loaded,
            load_multiline(
                &base_dir,
                "gate_stats",
                &["mean", "std"],
                expected,
                MissingMetric::Inactive,
            )?,
            gate_mean,
            gate_std
        );
        assign_multiline!(
            loaded,
            load_multiline(
                &base_dir,
                "hl_gauss_return_range",
                &[
                    "return_min",
                    "return_max",
                    "support_min",
                    "support_max",
                    "below_frac",
                    "above_frac",
                ],
                expected,
                MissingMetric::AlignedNan,
            )?,
            return_min,
            return_max,
            support_min,
            support_max,
            return_below_support_frac,
            return_above_support_frac
        );

        *self = loaded;
        println!("Loaded meta history from episode {episode} ({expected} data points)");
        Ok(())
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

    pub fn write_reports_default(&self, episode: usize) -> Result<()> {
        self.write_reports(episode, &format!("{TRAINING_PATH}/gens"))
    }

    pub fn write_reports(&self, episode: usize, gens_path: &str) -> Result<()> {
        let base_dir = format!("{gens_path}/{}", episode);
        std::fs::create_dir_all(&base_dir)
            .with_context(|| format!("failed creating meta report directory {base_dir}"))?;
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
            write_report(format!("{base_dir}/final_assets.report.bin"), &r)?;
        }
        if !self.cumulative_reward.is_empty() {
            let r = Self::report(
                "Cumulative Reward",
                "Episode",
                Some("Reward"),
                ScaleKind::Linear,
                simple(&self.cumulative_reward),
            );
            write_report(format!("{base_dir}/cumulative_reward.report.bin"), &r)?;
        }
        if !self.outperformance.is_empty() {
            let r = Self::report(
                "Outperformance",
                "Episode",
                Some("Outperformance"),
                ScaleKind::Linear,
                simple(&self.outperformance),
            );
            write_report(format!("{base_dir}/outperformance.report.bin"), &r)?;
        }
        if !self.policy_loss.is_empty() {
            let r = Self::report(
                "Policy Loss",
                "Episode",
                Some("Loss"),
                ScaleKind::Linear,
                simple(&self.policy_loss),
            );
            write_report(format!("{base_dir}/policy_loss.report.bin"), &r)?;
        }
        if !self.value_loss.is_empty() {
            let r = Self::report(
                "Value Loss",
                "Episode",
                Some("Loss"),
                ScaleKind::Linear,
                simple(&self.value_loss),
            );
            write_report(format!("{base_dir}/value_loss.report.bin"), &r)?;
        }
        if !self.explained_var.is_empty() {
            let r = Self::report(
                "Explained Variance",
                "Episode",
                Some("EV"),
                ScaleKind::Linear,
                simple(&self.explained_var),
            );
            write_report(format!("{base_dir}/explained_var.report.bin"), &r)?;
        }
        if !self.actor_grad_norm.is_empty() {
            let r = Self::report(
                "Actor Grad Norm",
                "Episode",
                Some("Grad Norm"),
                ScaleKind::Linear,
                simple(&self.actor_grad_norm),
            );
            write_report(format!("{base_dir}/actor_grad_norm.report.bin"), &r)?;
        }
        if !self.critic_grad_norm.is_empty() {
            let r = Self::report(
                "Critic Grad Norm",
                "Episode",
                Some("Grad Norm"),
                ScaleKind::Linear,
                simple(&self.critic_grad_norm),
            );
            write_report(format!("{base_dir}/critic_grad_norm.report.bin"), &r)?;
        }
        if !self.total_commissions.is_empty() {
            let r = Self::report(
                "Total Commissions",
                "Episode",
                Some("Commissions"),
                ScaleKind::Linear,
                simple(&self.total_commissions),
            );
            write_report(format!("{base_dir}/total_commissions.report.bin"), &r)?;
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
            write_report(format!("{base_dir}/beta_policy.report.bin"), &r)?;
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
            write_report(format!("{base_dir}/advantage_stats_log.report.bin"), &r)?;
        }
        if !self.logit_scale.is_empty() {
            let r = Self::report(
                "Logit Scale",
                "Episode",
                Some("Scale"),
                ScaleKind::Linear,
                simple(&self.logit_scale),
            );
            write_report(format!("{base_dir}/logit_scale.report.bin"), &r)?;
        }
        if !self.clip_fraction.is_empty() {
            let r = Self::report(
                "Clip Fraction",
                "Episode",
                Some("Fraction"),
                ScaleKind::Linear,
                simple(&self.clip_fraction),
            );
            write_report(format!("{base_dir}/clip_fraction.report.bin"), &r)?;
        }
        if !self.clip_gap.is_empty() {
            let r = Self::report(
                "Clip Gap",
                "Episode",
                Some("Gap"),
                ScaleKind::Linear,
                simple(&self.clip_gap),
            );
            write_report(format!("{base_dir}/clip_gap.report.bin"), &r)?;
        }
        if !self.approx_kl.is_empty() {
            let r = Self::report(
                "Policy KL",
                "Episode",
                Some("KL"),
                ScaleKind::Linear,
                simple(&self.approx_kl),
            );
            write_report(format!("{base_dir}/approx_kl.report.bin"), &r)?;
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
            write_report(format!("{base_dir}/kl_lr.report.bin"), &r)?;
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
            write_report(format!("{base_dir}/policy_entropy.report.bin"), &r)?;
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
            write_report(format!("{base_dir}/temporal_embed_debug.report.bin"), &r)?;
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
            write_report(format!("{base_dir}/gate_stats.report.bin"), &r)?;
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
            write_report(format!("{base_dir}/hl_gauss_return_range.report.bin"), &r)?;
        }
        Ok(())
    }
}

fn f64_to_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|v| *v as f32).collect()
}

fn load_simple(base_dir: &Path, base: &str, expected: usize, required: bool) -> Result<Vec<f64>> {
    match load_simple_if_present(base_dir, base, expected)? {
        Some(values) => Ok(values),
        None if required => bail!(
            "required meta history report is missing: {}",
            report_path(base_dir, base).display()
        ),
        None => Ok(missing_history(expected)),
    }
}

fn load_simple_if_present(
    base_dir: &Path,
    base: &str,
    expected: usize,
) -> Result<Option<Vec<f64>>> {
    let path = report_path(base_dir, base);
    let report = match read_report(&path) {
        Ok(report) => report,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed reading meta history {}", path.display()))
        }
    };
    let ReportKind::Simple { values, .. } = report.kind else {
        bail!("meta history {} is not a simple report", path.display());
    };
    validate_history_len(&path, expected, values.len())?;
    Ok(Some(values.into_iter().map(f64::from).collect()))
}

fn load_multiline(
    base_dir: &Path,
    base: &str,
    labels: &[&str],
    expected: usize,
    missing: MissingMetric,
) -> Result<Vec<Vec<f64>>> {
    let path = report_path(base_dir, base);
    let report = match read_report(&path) {
        Ok(report) => report,
        Err(error) if error.kind() == io::ErrorKind::NotFound => match missing {
            MissingMetric::Required => {
                bail!(
                    "required meta history report is missing: {}",
                    path.display()
                )
            }
            MissingMetric::AlignedNan => {
                return Ok(labels.iter().map(|_| missing_history(expected)).collect())
            }
            MissingMetric::Inactive => return Ok(labels.iter().map(|_| Vec::new()).collect()),
        },
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed reading meta history {}", path.display()))
        }
    };
    let ReportKind::MultiLine { series } = report.kind else {
        bail!("meta history {} is not a multiline report", path.display());
    };
    let mut seen = std::collections::HashSet::new();
    for item in &series {
        if !seen.insert(item.label.as_str()) {
            bail!(
                "meta history {} contains duplicate series {:?}",
                path.display(),
                item.label
            );
        }
    }
    let expected_labels = labels
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    if series.len() != labels.len()
        || series
            .iter()
            .any(|item| !expected_labels.contains(item.label.as_str()))
    {
        bail!(
            "meta history {} has an unexpected multiline schema",
            path.display()
        );
    }
    let values = labels
        .iter()
        .map(|label| {
            let item = series
                .iter()
                .find(|item| item.label == *label)
                .with_context(|| {
                    format!(
                        "meta history {} is missing series {label:?}",
                        path.display()
                    )
                })?;
            validate_history_len(&path, expected, item.values.len())?;
            Ok(item.values.iter().copied().map(f64::from).collect())
        })
        .collect::<Result<Vec<_>>>()?;
    if matches!(missing, MissingMetric::Inactive) {
        return Ok(labels.iter().map(|_| Vec::new()).collect());
    }
    Ok(values)
}

fn validate_history_len(path: &Path, expected: usize, actual: usize) -> Result<()> {
    if actual != expected {
        bail!(
            "meta history {} has {actual} values; expected {expected}",
            path.display()
        );
    }
    Ok(())
}

fn report_path(base_dir: &Path, base: &str) -> std::path::PathBuf {
    base_dir.join(format!("{base}.report.bin"))
}

fn missing_history(expected: usize) -> Vec<f64> {
    vec![f64::NAN; expected]
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::report::RL_META_REPORT_BASES;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "trading-bot-meta-{label}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn populated_history(len: usize) -> MetaHistory {
        let mut history = MetaHistory::default();
        macro_rules! fill {
            ($($field:ident),+ $(,)?) => {
                $(history.$field = vec![1.0; len];)+
            };
        }
        fill!(
            final_assets,
            cumulative_reward,
            outperformance,
            policy_loss,
            value_loss,
            explained_var,
            actor_grad_norm,
            critic_grad_norm,
            total_commissions,
            beta_alpha_mean,
            beta_action_mean,
            beta_beta_mean,
            beta_concentration_mean,
            mean_advantage,
            min_advantage,
            max_advantage,
            logit_scale,
            clip_fraction,
            clip_gap,
            temporal_tau,
            temporal_attn_entropy,
            temporal_attn_max,
            temporal_attn_eff_len,
            temporal_attn_center,
            temporal_attn_last_weight,
            policy_entropy_mean,
            policy_entropy_min,
            policy_entropy_max,
            approx_kl,
            kl_lr_scale,
            kl_lr_scale_next,
            kl_lr_ema,
            kl_lr_signal,
            gate_mean,
            gate_std,
            return_min,
            return_max,
            support_min,
            support_max,
            return_below_support_frac,
            return_above_support_frac,
        );
        history
    }

    #[test]
    fn produced_meta_reports_match_the_tui_registry() {
        let root = temp_dir("registry");
        populated_history(1)
            .write_reports(0, root.to_str().unwrap())
            .unwrap();

        let mut produced = fs::read_dir(root.join("0"))
            .unwrap()
            .map(|entry| {
                entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .trim_end_matches(".report.bin")
                    .to_owned()
            })
            .collect::<Vec<_>>();
        produced.sort();
        let mut registered = RL_META_REPORT_BASES
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        registered.sort();
        assert_eq!(produced, registered);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resume_rejects_corruption_without_partially_mutating_history() {
        let root = temp_dir("corrupt");
        populated_history(2)
            .write_reports(1, root.to_str().unwrap())
            .unwrap();
        let corrupt = Report {
            title: "Policy Loss".to_owned(),
            x_label: None,
            y_label: None,
            scale: ScaleKind::Linear,
            kind: ReportKind::Simple {
                values: vec![1.0],
                ema_alpha: None,
            },
        };
        write_report(root.join("1/policy_loss.report.bin"), &corrupt).unwrap();

        let mut destination = MetaHistory::default();
        destination.final_assets = vec![99.0];
        let error = destination
            .load_from_episode(1, root.to_str().unwrap())
            .unwrap_err();
        assert!(error.to_string().contains("expected 2"));
        assert_eq!(destination.final_assets, vec![99.0]);
        assert!(destination.policy_loss.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resume_aligns_missing_optional_metrics_with_nan_placeholders() {
        let root = temp_dir("optional");
        populated_history(2)
            .write_reports(1, root.to_str().unwrap())
            .unwrap();
        for optional in [
            "logit_scale",
            "clip_fraction",
            "clip_gap",
            "temporal_embed_debug",
            "kl_lr",
            "gate_stats",
            "hl_gauss_return_range",
        ] {
            fs::remove_file(root.join(format!("1/{optional}.report.bin"))).unwrap();
        }

        let mut loaded = MetaHistory::default();
        loaded.load_from_episode(1, root.to_str().unwrap()).unwrap();
        assert_eq!(loaded.final_assets, vec![1.0, 1.0]);
        assert_eq!(loaded.clip_gap.len(), 2);
        assert!(loaded.clip_gap.iter().all(|value| value.is_nan()));
        assert!(loaded.logit_scale.is_empty());
        assert!(loaded.temporal_tau.is_empty());
        assert!(loaded.gate_mean.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resume_rejects_a_missing_established_metric() {
        let root = temp_dir("missing-required");
        populated_history(2)
            .write_reports(1, root.to_str().unwrap())
            .unwrap();
        fs::remove_file(root.join("1/policy_loss.report.bin")).unwrap();

        let error = MetaHistory::default()
            .load_from_episode(1, root.to_str().unwrap())
            .unwrap_err();
        assert!(error.to_string().contains("policy_loss.report.bin"));
        fs::remove_dir_all(root).unwrap();
    }
}
