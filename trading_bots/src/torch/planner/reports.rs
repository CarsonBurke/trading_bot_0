use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use shared::report::{Report, ReportKind, ReportSeries, ScaleKind};

const PLANNER_GENERATION_MARKER: &str = ".planner-report-generation";
const PLANNER_INFERENCE_MANIFEST: &str = ".planner-inference.json";
const PLANNER_INFERENCE_REPORTS: [&str; 6] = [
    "planner_inference_wealth",
    "planner_inference_outperformance",
    "planner_inference_outperformance_fraction",
    "planner_inference_risk",
    "planner_inference_action",
    "planner_inference_commissions",
];

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct PlannerGenerationOwner {
    run_lineage_id: String,
    update: u64,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct PlannerInferenceManifest {
    version: u32,
    run_lineage_id: String,
    update: u64,
    split: String,
    evaluation_fingerprint: String,
    episodes: usize,
    rollout_length: usize,
}

pub fn cleanup_uncommitted_report_generations(
    gens: impl AsRef<Path>,
    committed_update: u64,
    run_lineage_id: &str,
) -> Result<()> {
    let gens = gens.as_ref();
    let entries = match fs::read_dir(gens) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    for entry in entries.filter_map(std::result::Result::ok) {
        if !entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let numeric_update = name.parse::<u64>().ok();
        let update = numeric_update.or_else(|| {
            name.strip_prefix('.')?
                .strip_suffix(".planner-reports.tmp")?
                .parse::<u64>()
                .ok()
        });
        let Some(update) = update.filter(|update| *update > committed_update) else {
            continue;
        };
        match read_owner(&entry.path())? {
            Some(owner) if owner.run_lineage_id == run_lineage_id && owner.update == update => {
                fs::remove_dir_all(entry.path()).with_context(|| {
                    format!("failed removing uncommitted planner report generation {name}")
                })?;
            }
            _ if numeric_update.is_some() => {
                bail!(
                    "uncommitted numeric generation {} is not owned by planner run {run_lineage_id}; preserving it and refusing resume",
                    entry.path().display()
                );
            }
            _ => {}
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlannerEpisodeTrace {
    pub ticker: String,
    pub cash: Vec<f64>,
    pub positioned: Vec<f64>,
    pub total: Vec<f64>,
    pub benchmark: Vec<f64>,
    pub rewards: Vec<f64>,
    pub commissions: Vec<f64>,
    pub turnover: Vec<f64>,
    pub requested_target_weight: Vec<f64>,
    pub executed_stock_weight: Vec<f64>,
}

impl PlannerEpisodeTrace {
    pub fn validate(&self) -> Result<()> {
        let asset_points = self.total.len();
        if asset_points == 0
            || self.cash.len() != asset_points
            || self.positioned.len() != asset_points
            || self.benchmark.len() != asset_points
        {
            bail!("planner trace asset curves must be equally sized and non-empty");
        }
        let action_points = self.requested_target_weight.len();
        if self.executed_stock_weight.len() != action_points
            || self.rewards.len() != action_points
            || self.commissions.len() != action_points
            || self.turnover.len() != action_points
            || asset_points != action_points + 1
        {
            bail!(
                "planner trace must contain one initial asset point and one action/reward per step"
            );
        }
        if self
            .cash
            .iter()
            .chain(&self.positioned)
            .chain(&self.total)
            .chain(&self.benchmark)
            .chain(&self.rewards)
            .chain(&self.commissions)
            .chain(&self.turnover)
            .chain(&self.requested_target_weight)
            .chain(&self.executed_stock_weight)
            .any(|value| !value.is_finite())
        {
            bail!("planner trace contains NaN or infinity");
        }
        for ((total, cash), positioned) in self.total.iter().zip(&self.cash).zip(&self.positioned) {
            let tolerance = 1e-8 * total.abs().max(1.0);
            if (total - (cash + positioned)).abs() > tolerance {
                bail!("planner trace total does not equal cash plus positioned value");
            }
        }
        if self
            .requested_target_weight
            .iter()
            .chain(&self.executed_stock_weight)
            .any(|weight| !(0.0..=1.0).contains(weight))
        {
            bail!("planner trace position weights must be in [0, 1]");
        }
        Ok(())
    }
}

pub struct PlannerStagedReports {
    staging: PathBuf,
    output: PathBuf,
    owner: PlannerGenerationOwner,
}

impl PlannerStagedReports {
    pub fn publish(self) -> Result<PathBuf> {
        if self.output.exists() {
            let existing = read_owner(&self.output)?;
            bail!(
                "refusing to replace existing planner generation {} (owner: {existing:?}); expected new owner {:?}",
                self.output.display(),
                self.owner,
            );
        }
        fs::rename(&self.staging, &self.output).with_context(|| {
            format!(
                "failed publishing planner reports {}",
                self.output.display()
            )
        })?;
        if let Some(parent) = self.output.parent() {
            fs::File::open(parent)?.sync_all()?;
        }
        Ok(self.output)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PlannerTrainingReportPoint {
    pub reward_mean: f64,
    pub wealth_ratio: f64,
    pub buy_and_hold_wealth_ratio: f64,
    pub mean_outperformance_ratio: f64,
    pub median_outperformance_ratio: f64,
    pub outperformance_fraction: f64,
    pub turnover_mean: f64,
    pub commissions: f64,
    pub requested_target_weight_mean: f64,
    pub executed_stock_weight_mean: f64,
    pub action_boundary_fraction: f64,
    pub deterministic_reward_mean: f64,
    pub deterministic_wealth_ratio: f64,
    pub deterministic_mean_outperformance_ratio: f64,
    pub deterministic_median_outperformance_ratio: f64,
    pub deterministic_outperformance_fraction: f64,
    pub deterministic_turnover_mean: f64,
    pub deterministic_commissions: f64,
    pub deterministic_requested_target_weight_mean: f64,
    pub deterministic_executed_stock_weight_mean: f64,
    pub deterministic_action_boundary_fraction: f64,
    pub beta_concentration: f64,
    pub critic_explained_variance: f64,
    pub actor_loss: f64,
    pub critic_loss: f64,
    pub reverse_kl: f64,
    pub max_reverse_kl: f64,
    pub kl_early_stopped: bool,
    pub entropy: f64,
    pub actor_grad_norm: f64,
    pub critic_grad_norm: f64,
}

impl PlannerTrainingReportPoint {
    fn validate(self) -> Result<()> {
        let values = [
            self.reward_mean,
            self.wealth_ratio,
            self.buy_and_hold_wealth_ratio,
            self.mean_outperformance_ratio,
            self.median_outperformance_ratio,
            self.outperformance_fraction,
            self.turnover_mean,
            self.commissions,
            self.requested_target_weight_mean,
            self.executed_stock_weight_mean,
            self.action_boundary_fraction,
            self.deterministic_reward_mean,
            self.deterministic_wealth_ratio,
            self.deterministic_mean_outperformance_ratio,
            self.deterministic_median_outperformance_ratio,
            self.deterministic_outperformance_fraction,
            self.deterministic_turnover_mean,
            self.deterministic_commissions,
            self.deterministic_requested_target_weight_mean,
            self.deterministic_executed_stock_weight_mean,
            self.deterministic_action_boundary_fraction,
            self.beta_concentration,
            self.actor_loss,
            self.critic_loss,
            self.reverse_kl,
            self.max_reverse_kl,
            self.entropy,
            self.actor_grad_norm,
            self.critic_grad_norm,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            bail!("planner training report point contains NaN or infinity");
        }
        if self.critic_explained_variance.is_infinite() {
            bail!("planner critic explained variance is infinite");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PlannerValidationReportPoint {
    pub median_wealth_ratio: f64,
    pub median_buy_and_hold_wealth_ratio: f64,
    pub mean_outperformance_ratio: f64,
    pub median_outperformance_ratio: f64,
    pub outperformance_fraction: f64,
    pub mean_max_drawdown: f64,
    pub mean_turnover: f64,
    pub eligible: bool,
    pub selected: bool,
}

#[derive(Debug, Default)]
pub struct PlannerReportHistory {
    gens: PathBuf,
    run_lineage_id: String,
    reward_mean: Vec<f32>,
    wealth_ratio: Vec<f32>,
    buy_and_hold_wealth_ratio: Vec<f32>,
    mean_outperformance_ratio: Vec<f32>,
    median_outperformance_ratio: Vec<f32>,
    outperformance_fraction: Vec<f32>,
    turnover_mean: Vec<f32>,
    commissions: Vec<f32>,
    requested_target_weight_mean: Vec<f32>,
    executed_stock_weight_mean: Vec<f32>,
    action_boundary_fraction: Vec<f32>,
    deterministic_reward_mean: Vec<f32>,
    deterministic_wealth_ratio: Vec<f32>,
    deterministic_mean_outperformance_ratio: Vec<f32>,
    deterministic_median_outperformance_ratio: Vec<f32>,
    deterministic_outperformance_fraction: Vec<f32>,
    deterministic_turnover_mean: Vec<f32>,
    deterministic_commissions: Vec<f32>,
    deterministic_requested_target_weight_mean: Vec<f32>,
    deterministic_executed_stock_weight_mean: Vec<f32>,
    deterministic_action_boundary_fraction: Vec<f32>,
    beta_concentration: Vec<f32>,
    critic_explained_variance: Vec<f32>,
    actor_loss: Vec<f32>,
    critic_loss: Vec<f32>,
    reverse_kl: Vec<f32>,
    max_reverse_kl: Vec<f32>,
    kl_early_stopped: Vec<f32>,
    entropy: Vec<f32>,
    actor_grad_norm: Vec<f32>,
    critic_grad_norm: Vec<f32>,
    validation_median_wealth_ratio: Vec<f32>,
    validation_median_buy_and_hold_wealth_ratio: Vec<f32>,
    validation_mean_outperformance_ratio: Vec<f32>,
    validation_median_outperformance_ratio: Vec<f32>,
    validation_outperformance_fraction: Vec<f32>,
    validation_mean_max_drawdown: Vec<f32>,
    validation_mean_turnover: Vec<f32>,
    validation_eligible: Vec<f32>,
    validation_selected: Vec<f32>,
}

impl PlannerReportHistory {
    pub fn load(
        gens: impl AsRef<Path>,
        completed_updates: u64,
        run_lineage_id: impl Into<String>,
    ) -> Result<Self> {
        let gens = gens.as_ref().to_path_buf();
        let mut history = Self {
            gens,
            run_lineage_id: run_lineage_id.into(),
            ..Self::default()
        };
        let completed = usize::try_from(completed_updates)
            .context("planner update does not fit report history index")?;
        if completed > 0 {
            history.load_generation(completed)?;
        }
        history.resize_training(completed);
        history.resize_validation(completed);
        Ok(history)
    }

    pub fn stage_training(
        &mut self,
        update: u64,
        point: PlannerTrainingReportPoint,
        sampled_trace: &PlannerEpisodeTrace,
        deterministic_trace: &PlannerEpisodeTrace,
    ) -> Result<PlannerStagedReports> {
        point.validate()?;
        sampled_trace.validate()?;
        deterministic_trace.validate()?;
        let index = update_index(update)?;
        self.resize_training(index + 1);
        self.resize_validation(index + 1);
        macro_rules! set {
            ($field:ident, $value:expr) => {
                self.$field[index] = $value as f32
            };
        }
        set!(reward_mean, point.reward_mean);
        set!(wealth_ratio, point.wealth_ratio);
        set!(buy_and_hold_wealth_ratio, point.buy_and_hold_wealth_ratio);
        set!(mean_outperformance_ratio, point.mean_outperformance_ratio);
        set!(
            median_outperformance_ratio,
            point.median_outperformance_ratio
        );
        set!(outperformance_fraction, point.outperformance_fraction);
        set!(turnover_mean, point.turnover_mean);
        set!(commissions, point.commissions);
        set!(
            requested_target_weight_mean,
            point.requested_target_weight_mean
        );
        set!(executed_stock_weight_mean, point.executed_stock_weight_mean);
        set!(action_boundary_fraction, point.action_boundary_fraction);
        set!(deterministic_reward_mean, point.deterministic_reward_mean);
        set!(deterministic_wealth_ratio, point.deterministic_wealth_ratio);
        set!(
            deterministic_mean_outperformance_ratio,
            point.deterministic_mean_outperformance_ratio
        );
        set!(
            deterministic_median_outperformance_ratio,
            point.deterministic_median_outperformance_ratio
        );
        set!(
            deterministic_outperformance_fraction,
            point.deterministic_outperformance_fraction
        );
        set!(
            deterministic_turnover_mean,
            point.deterministic_turnover_mean
        );
        set!(deterministic_commissions, point.deterministic_commissions);
        set!(
            deterministic_requested_target_weight_mean,
            point.deterministic_requested_target_weight_mean
        );
        set!(
            deterministic_executed_stock_weight_mean,
            point.deterministic_executed_stock_weight_mean
        );
        set!(
            deterministic_action_boundary_fraction,
            point.deterministic_action_boundary_fraction
        );
        set!(beta_concentration, point.beta_concentration);
        set!(critic_explained_variance, point.critic_explained_variance);
        set!(actor_loss, point.actor_loss);
        set!(critic_loss, point.critic_loss);
        set!(reverse_kl, point.reverse_kl);
        set!(max_reverse_kl, point.max_reverse_kl);
        set!(kl_early_stopped, f64::from(point.kl_early_stopped));
        set!(entropy, point.entropy);
        set!(actor_grad_norm, point.actor_grad_norm);
        set!(critic_grad_norm, point.critic_grad_norm);

        let output = self.gens.join(update.to_string());
        let staging = self.gens.join(format!(".{update}.planner-reports.tmp"));
        if staging.exists() {
            match read_owner(&staging)? {
                Some(owner)
                    if owner.run_lineage_id == self.run_lineage_id && owner.update == update =>
                {
                    fs::remove_dir_all(&staging)?;
                }
                owner => {
                    bail!(
                        "refusing to replace planner staging directory {} with owner {owner:?}",
                        staging.display()
                    );
                }
            }
        }
        fs::create_dir_all(&staging)?;
        let owner = PlannerGenerationOwner {
            run_lineage_id: self.run_lineage_id.clone(),
            update,
        };
        write_owner(&staging, &owner)?;
        self.write_training_reports(&staging)?;
        self.write_validation_reports(&staging)?;
        write_episode_trace(&staging, "Planner Sampled Training Episode", sampled_trace)?;
        let deterministic_output = staging.join("deterministic_mean");
        fs::create_dir_all(&deterministic_output)?;
        write_episode_trace(
            &deterministic_output,
            "Planner Deterministic Beta-Mean Training Episode",
            deterministic_trace,
        )?;
        fs::File::open(&staging)?.sync_all()?;
        Ok(PlannerStagedReports {
            staging,
            output,
            owner,
        })
    }

    pub fn record_validation(
        &mut self,
        update: u64,
        point: PlannerValidationReportPoint,
    ) -> Result<PathBuf> {
        let index = update_index(update)?;
        self.resize_validation(index + 1);
        macro_rules! set {
            ($field:ident, $value:expr) => {
                self.$field[index] = $value as f32
            };
        }
        set!(validation_median_wealth_ratio, point.median_wealth_ratio);
        set!(
            validation_median_buy_and_hold_wealth_ratio,
            point.median_buy_and_hold_wealth_ratio
        );
        set!(
            validation_mean_outperformance_ratio,
            point.mean_outperformance_ratio
        );
        set!(
            validation_median_outperformance_ratio,
            point.median_outperformance_ratio
        );
        set!(
            validation_outperformance_fraction,
            point.outperformance_fraction
        );
        set!(validation_mean_max_drawdown, point.mean_max_drawdown);
        set!(validation_mean_turnover, point.mean_turnover);
        set!(validation_eligible, f64::from(point.eligible));
        set!(validation_selected, f64::from(point.selected));
        let output = self.gens.join(update.to_string());
        require_owner(&output, &self.run_lineage_id, update)?;
        fs::create_dir_all(&output)?;
        self.write_validation_reports(&output)?;
        fs::File::open(&output)?.sync_all()?;
        Ok(output)
    }

    fn load_generation(&mut self, update: usize) -> Result<()> {
        let dir = self.gens.join(update.to_string());
        require_owner(&dir, &self.run_lineage_id, update as u64)?;
        macro_rules! simple {
            ($field:ident, $file:literal) => {
                self.$field = read_required_simple(&dir.join($file), update)?;
                validate_finite_history(&dir.join($file), &self.$field)?
            };
        }
        macro_rules! simple_allow_nan {
            ($field:ident, $file:literal) => {
                self.$field = read_required_simple(&dir.join($file), update)?;
                if self.$field.iter().any(|value| value.is_infinite()) {
                    bail!(
                        "planner report {} contains infinity",
                        dir.join($file).display()
                    );
                }
            };
        }
        macro_rules! line {
            ($field:ident, $file:literal, $label:literal) => {
                self.$field = read_required_line(&dir.join($file), $label, update)?;
                validate_finite_history(&dir.join($file), &self.$field)?
            };
        }
        macro_rules! optional_simple {
            ($field:ident, $file:literal) => {
                self.$field = read_optional_simple(&dir.join($file), update)?
            };
        }
        macro_rules! optional_line {
            ($field:ident, $file:literal, $label:literal) => {
                self.$field = read_optional_line(&dir.join($file), $label, update)?
            };
        }
        for file in [
            "assets.report.bin",
            "reward.report.bin",
            "planner_position.report.bin",
        ] {
            read_report(&dir.join(file)).with_context(|| {
                format!("committed planner generation {update} is missing or corrupt: {file}")
            })?;
        }
        simple!(reward_mean, "planner_reward.report.bin");
        line!(wealth_ratio, "planner_wealth.report.bin", "policy wealth");
        line!(
            buy_and_hold_wealth_ratio,
            "planner_wealth.report.bin",
            "buy-and-hold"
        );
        line!(
            mean_outperformance_ratio,
            "planner_outperformance.report.bin",
            "mean"
        );
        line!(
            median_outperformance_ratio,
            "planner_outperformance.report.bin",
            "median"
        );
        simple!(
            outperformance_fraction,
            "planner_outperformance_fraction.report.bin"
        );
        line!(turnover_mean, "planner_turnover.report.bin", "turnover");
        line!(
            action_boundary_fraction,
            "planner_turnover.report.bin",
            "action boundary fraction"
        );
        simple!(commissions, "planner_commissions.report.bin");
        line!(
            requested_target_weight_mean,
            "planner_position_mean.report.bin",
            "requested target weight"
        );
        line!(
            executed_stock_weight_mean,
            "planner_position_mean.report.bin",
            "executed stock exposure"
        );
        optional_simple!(
            deterministic_reward_mean,
            "planner_deterministic_reward.report.bin"
        );
        optional_line!(
            deterministic_wealth_ratio,
            "planner_deterministic_wealth.report.bin",
            "deterministic Beta-mean policy wealth"
        );
        optional_line!(
            deterministic_mean_outperformance_ratio,
            "planner_deterministic_outperformance.report.bin",
            "mean"
        );
        optional_line!(
            deterministic_median_outperformance_ratio,
            "planner_deterministic_outperformance.report.bin",
            "median"
        );
        optional_simple!(
            deterministic_outperformance_fraction,
            "planner_deterministic_outperformance_fraction.report.bin"
        );
        optional_line!(
            deterministic_turnover_mean,
            "planner_deterministic_turnover.report.bin",
            "turnover"
        );
        optional_line!(
            deterministic_action_boundary_fraction,
            "planner_deterministic_turnover.report.bin",
            "action boundary fraction"
        );
        optional_simple!(
            deterministic_commissions,
            "planner_deterministic_commissions.report.bin"
        );
        optional_line!(
            deterministic_requested_target_weight_mean,
            "planner_deterministic_position_mean.report.bin",
            "requested target weight"
        );
        optional_line!(
            deterministic_executed_stock_weight_mean,
            "planner_deterministic_position_mean.report.bin",
            "executed stock exposure"
        );
        simple!(beta_concentration, "beta_policy.report.bin");
        simple_allow_nan!(critic_explained_variance, "explained_var.report.bin");
        simple!(actor_loss, "policy_loss.report.bin");
        simple!(critic_loss, "value_loss.report.bin");
        line!(reverse_kl, "approx_kl.report.bin", "reverse KL");
        line!(max_reverse_kl, "approx_kl.report.bin", "max reverse KL");
        line!(kl_early_stopped, "approx_kl.report.bin", "early stopped");
        simple!(entropy, "policy_entropy.report.bin");
        simple!(actor_grad_norm, "actor_grad_norm.report.bin");
        simple!(critic_grad_norm, "critic_grad_norm.report.bin");
        optional_line!(
            validation_median_wealth_ratio,
            "planner_validation_wealth.report.bin",
            "policy median wealth"
        );
        optional_line!(
            validation_median_buy_and_hold_wealth_ratio,
            "planner_validation_wealth.report.bin",
            "buy-and-hold median wealth"
        );
        optional_line!(
            validation_mean_outperformance_ratio,
            "planner_validation_outperformance.report.bin",
            "mean"
        );
        optional_line!(
            validation_median_outperformance_ratio,
            "planner_validation_outperformance.report.bin",
            "median"
        );
        optional_simple!(
            validation_outperformance_fraction,
            "planner_validation_outperformance_fraction.report.bin"
        );
        optional_line!(
            validation_mean_max_drawdown,
            "planner_validation_risk.report.bin",
            "mean max drawdown"
        );
        optional_line!(
            validation_mean_turnover,
            "planner_validation_risk.report.bin",
            "mean turnover"
        );
        optional_line!(
            validation_eligible,
            "planner_validation_selection.report.bin",
            "eligible"
        );
        optional_line!(
            validation_selected,
            "planner_validation_selection.report.bin",
            "selected"
        );
        Ok(())
    }

    fn resize_training(&mut self, len: usize) {
        macro_rules! resize { ($($field:ident),+ $(,)?) => { $(self.$field.resize(len, f32::NAN);)+ }; }
        resize!(
            reward_mean,
            wealth_ratio,
            buy_and_hold_wealth_ratio,
            mean_outperformance_ratio,
            median_outperformance_ratio,
            outperformance_fraction,
            turnover_mean,
            commissions,
            requested_target_weight_mean,
            executed_stock_weight_mean,
            action_boundary_fraction,
            deterministic_reward_mean,
            deterministic_wealth_ratio,
            deterministic_mean_outperformance_ratio,
            deterministic_median_outperformance_ratio,
            deterministic_outperformance_fraction,
            deterministic_turnover_mean,
            deterministic_commissions,
            deterministic_requested_target_weight_mean,
            deterministic_executed_stock_weight_mean,
            deterministic_action_boundary_fraction,
            beta_concentration,
            critic_explained_variance,
            actor_loss,
            critic_loss,
            reverse_kl,
            max_reverse_kl,
            kl_early_stopped,
            entropy,
            actor_grad_norm,
            critic_grad_norm
        );
    }

    fn resize_validation(&mut self, len: usize) {
        macro_rules! resize { ($($field:ident),+ $(,)?) => { $(self.$field.resize(len, f32::NAN);)+ }; }
        resize!(
            validation_median_wealth_ratio,
            validation_median_buy_and_hold_wealth_ratio,
            validation_mean_outperformance_ratio,
            validation_median_outperformance_ratio,
            validation_outperformance_fraction,
            validation_mean_max_drawdown,
            validation_mean_turnover,
            validation_eligible,
            validation_selected
        );
    }

    fn write_training_reports(&self, output: &Path) -> Result<()> {
        write_multiline(
            output,
            "planner_wealth",
            "Planner Sampled On-Policy Training Wealth",
            "wealth / starting wealth",
            vec![
                series("policy wealth", &self.wealth_ratio),
                series("buy-and-hold", &self.buy_and_hold_wealth_ratio),
            ],
        )?;
        write_simple(
            output,
            "planner_reward",
            "Planner Sampled On-Policy Mean Reward",
            "mean scaled log return",
            &self.reward_mean,
            ScaleKind::Symlog,
        )?;
        write_multiline(
            output,
            "planner_outperformance",
            "Planner Sampled On-Policy Training Outperformance",
            "wealth ratio delta",
            vec![
                series("mean", &self.mean_outperformance_ratio),
                series("median", &self.median_outperformance_ratio),
            ],
        )?;
        write_simple(
            output,
            "planner_outperformance_fraction",
            "Planner Sampled On-Policy Training Outperformance Fraction",
            "fraction",
            &self.outperformance_fraction,
            ScaleKind::Linear,
        )?;
        write_multiline(
            output,
            "planner_position_mean",
            "Planner Sampled On-Policy Mean Position / Exposure",
            "portfolio fraction",
            vec![
                series(
                    "requested target weight",
                    &self.requested_target_weight_mean,
                ),
                series("executed stock exposure", &self.executed_stock_weight_mean),
            ],
        )?;
        write_multiline(
            output,
            "planner_turnover",
            "Planner Sampled On-Policy Turnover and Saturation",
            "fraction",
            vec![
                series("turnover", &self.turnover_mean),
                series("action boundary fraction", &self.action_boundary_fraction),
            ],
        )?;
        write_simple(
            output,
            "planner_commissions",
            "Planner Sampled On-Policy Commissions",
            "commission",
            &self.commissions,
            ScaleKind::Linear,
        )?;
        write_multiline(
            output,
            "planner_deterministic_wealth",
            "Planner Deterministic Beta-Mean Training Wealth",
            "wealth / starting wealth",
            vec![
                series(
                    "deterministic Beta-mean policy wealth",
                    &self.deterministic_wealth_ratio,
                ),
                series("buy-and-hold", &self.buy_and_hold_wealth_ratio),
            ],
        )?;
        write_simple(
            output,
            "planner_deterministic_reward",
            "Planner Deterministic Beta-Mean Step Reward",
            "mean scaled log return",
            &self.deterministic_reward_mean,
            ScaleKind::Symlog,
        )?;
        write_multiline(
            output,
            "planner_deterministic_outperformance",
            "Planner Deterministic Beta-Mean Training Outperformance",
            "wealth ratio delta",
            vec![
                series("mean", &self.deterministic_mean_outperformance_ratio),
                series("median", &self.deterministic_median_outperformance_ratio),
            ],
        )?;
        write_simple(
            output,
            "planner_deterministic_outperformance_fraction",
            "Planner Deterministic Beta-Mean Training Outperformance Fraction",
            "fraction",
            &self.deterministic_outperformance_fraction,
            ScaleKind::Linear,
        )?;
        write_multiline(
            output,
            "planner_deterministic_position_mean",
            "Planner Deterministic Beta-Mean Position / Exposure",
            "portfolio fraction",
            vec![
                series(
                    "requested target weight",
                    &self.deterministic_requested_target_weight_mean,
                ),
                series(
                    "executed stock exposure",
                    &self.deterministic_executed_stock_weight_mean,
                ),
            ],
        )?;
        write_multiline(
            output,
            "planner_deterministic_turnover",
            "Planner Deterministic Beta-Mean Turnover and Saturation",
            "fraction",
            vec![
                series("turnover", &self.deterministic_turnover_mean),
                series(
                    "action boundary fraction",
                    &self.deterministic_action_boundary_fraction,
                ),
            ],
        )?;
        write_simple(
            output,
            "planner_deterministic_commissions",
            "Planner Deterministic Beta-Mean Commissions",
            "commission",
            &self.deterministic_commissions,
            ScaleKind::Linear,
        )?;
        write_simple(
            output,
            "beta_policy",
            "Planner Beta Concentration",
            "concentration",
            &self.beta_concentration,
            ScaleKind::Linear,
        )?;
        write_simple_allow_nan(
            output,
            "explained_var",
            "Planner Critic Explained Variance",
            "explained variance",
            &self.critic_explained_variance,
            ScaleKind::Linear,
        )?;
        write_simple(
            output,
            "policy_loss",
            "Planner Policy Loss",
            "loss",
            &self.actor_loss,
            ScaleKind::Linear,
        )?;
        write_simple(
            output,
            "value_loss",
            "Planner Value Loss",
            "loss",
            &self.critic_loss,
            ScaleKind::Linear,
        )?;
        write_multiline(
            output,
            "approx_kl",
            "Planner Approx KL",
            "KL / indicator",
            vec![
                series("reverse KL", &self.reverse_kl),
                series("max reverse KL", &self.max_reverse_kl),
                series("early stopped", &self.kl_early_stopped),
            ],
        )?;
        write_simple(
            output,
            "policy_entropy",
            "Planner Policy Entropy",
            "entropy",
            &self.entropy,
            ScaleKind::Linear,
        )?;
        write_simple(
            output,
            "actor_grad_norm",
            "Planner Actor Gradient Norm",
            "gradient norm",
            &self.actor_grad_norm,
            ScaleKind::Linear,
        )?;
        write_simple(
            output,
            "critic_grad_norm",
            "Planner Critic Gradient Norm",
            "gradient norm",
            &self.critic_grad_norm,
            ScaleKind::Linear,
        )
    }

    fn write_validation_reports(&self, output: &Path) -> Result<()> {
        write_multiline(
            output,
            "planner_validation_wealth",
            "Planner Validation Wealth",
            "wealth / starting wealth",
            vec![
                series("policy median wealth", &self.validation_median_wealth_ratio),
                series(
                    "buy-and-hold median wealth",
                    &self.validation_median_buy_and_hold_wealth_ratio,
                ),
            ],
        )?;
        write_multiline(
            output,
            "planner_validation_outperformance",
            "Planner Validation Outperformance",
            "wealth ratio delta",
            vec![
                series("mean", &self.validation_mean_outperformance_ratio),
                series("median", &self.validation_median_outperformance_ratio),
            ],
        )?;
        write_simple(
            output,
            "planner_validation_outperformance_fraction",
            "Planner Validation Outperformance Fraction",
            "fraction",
            &self.validation_outperformance_fraction,
            ScaleKind::Linear,
        )?;
        write_multiline(
            output,
            "planner_validation_risk",
            "Planner Validation Risk",
            "fraction",
            vec![
                series("mean max drawdown", &self.validation_mean_max_drawdown),
                series("mean turnover", &self.validation_mean_turnover),
            ],
        )?;
        write_multiline(
            output,
            "planner_validation_selection",
            "Planner Validation Selection",
            "indicator",
            vec![
                series("eligible", &self.validation_eligible),
                series("selected", &self.validation_selected),
            ],
        )
    }
}

pub fn write_inference_reports(
    gens: impl AsRef<Path>,
    update: u64,
    run_lineage_id: &str,
    split: &str,
    traces: &[PlannerEpisodeTrace],
    evaluation_fingerprint: &str,
) -> Result<PathBuf> {
    if traces.is_empty() || evaluation_fingerprint.is_empty() {
        bail!("planner inference report requires episodes and an evaluation fingerprint");
    }
    for trace in traces {
        trace.validate()?;
    }
    let generation = gens.as_ref().join(update.to_string());
    require_owner(&generation, run_lineage_id, update)?;
    let display_split = display_split(split);
    let split_lower = sanitize_component(&split.to_ascii_lowercase());
    let rollout_length = traces[0].rewards.len();
    if traces
        .iter()
        .any(|trace| trace.rewards.len() != rollout_length)
    {
        bail!("planner inference traces must share one rollout length");
    }
    cleanup_inference_staging(&generation, &split_lower, run_lineage_id, update)?;
    let id = uuid::Uuid::new_v4();
    let output = generation.join(format!(".planner-inference-{split_lower}-{id}.tmp"));
    let published = generation.join(format!("planner_inference_{split_lower}_{id}"));
    fs::create_dir(&output)?;
    write_owner(
        &output,
        &PlannerGenerationOwner {
            run_lineage_id: run_lineage_id.to_owned(),
            update,
        },
    )?;
    let manifest = PlannerInferenceManifest {
        version: 1,
        run_lineage_id: run_lineage_id.to_owned(),
        update,
        split: split_lower.clone(),
        evaluation_fingerprint: evaluation_fingerprint.to_owned(),
        episodes: traces.len(),
        rollout_length,
    };
    let manifest_path = output.join(PLANNER_INFERENCE_MANIFEST);
    fs::write(&manifest_path, serde_json::to_vec(&manifest)?)?;
    fs::File::open(&manifest_path)?.sync_all()?;
    let wealth = traces
        .iter()
        .map(|trace| trace.total.last().copied().unwrap() / trace.total[0])
        .collect::<Vec<_>>();
    let benchmark = traces
        .iter()
        .map(|trace| trace.benchmark.last().copied().unwrap() / trace.benchmark[0])
        .collect::<Vec<_>>();
    let outperformance = wealth
        .iter()
        .zip(&benchmark)
        .map(|(policy, benchmark)| policy - benchmark)
        .collect::<Vec<_>>();
    let fraction = outperformance.iter().filter(|value| **value > 0.0).count() as f64
        / outperformance.len() as f64;
    let drawdown = traces
        .iter()
        .map(|trace| max_drawdown(&trace.total))
        .collect::<Vec<_>>();
    let turnover = traces
        .iter()
        .map(|trace| trace.turnover.iter().sum::<f64>() / trace.turnover.len() as f64)
        .collect::<Vec<_>>();
    let requested = traces
        .iter()
        .map(|trace| {
            trace.requested_target_weight.iter().sum::<f64>()
                / trace.requested_target_weight.len() as f64
        })
        .collect::<Vec<_>>();
    let executed = traces
        .iter()
        .map(|trace| {
            trace.executed_stock_weight.iter().sum::<f64>()
                / trace.executed_stock_weight.len() as f64
        })
        .collect::<Vec<_>>();
    let commissions = traces
        .iter()
        .map(|trace| trace.commissions.iter().sum::<f64>())
        .collect::<Vec<_>>();
    write_multiline_with_x(
        &output,
        "planner_inference_wealth",
        &format!("Planner {display_split} Wealth"),
        "wealth / starting wealth",
        "episode",
        vec![
            ReportSeries {
                label: "policy".to_owned(),
                values: f64_to_f32(&wealth),
            },
            ReportSeries {
                label: "buy-and-hold".to_owned(),
                values: f64_to_f32(&benchmark),
            },
        ],
    )?;
    write_simple_with_x(
        &output,
        "planner_inference_outperformance",
        &format!("Planner {display_split} Outperformance"),
        "wealth ratio delta",
        "episode",
        &f64_to_f32(&outperformance),
        ScaleKind::Linear,
    )?;
    write_multiline_with_x(
        &output,
        "planner_inference_risk",
        &format!("Planner {display_split} Risk"),
        "fraction",
        "episode",
        vec![
            ReportSeries {
                label: "max drawdown".to_owned(),
                values: f64_to_f32(&drawdown),
            },
            ReportSeries {
                label: "mean turnover".to_owned(),
                values: f64_to_f32(&turnover),
            },
        ],
    )?;
    write_multiline_with_x(
        &output,
        "planner_inference_action",
        &format!("Planner {display_split} Position / Exposure"),
        "portfolio fraction",
        "episode",
        vec![
            ReportSeries {
                label: "requested target weight".to_owned(),
                values: f64_to_f32(&requested),
            },
            ReportSeries {
                label: "executed stock exposure".to_owned(),
                values: f64_to_f32(&executed),
            },
        ],
    )?;
    write_simple_with_x(
        &output,
        "planner_inference_commissions",
        &format!("Planner {display_split} Commissions"),
        "commission",
        "episode",
        &f64_to_f32(&commissions),
        ScaleKind::Linear,
    )?;
    write_simple_with_x(
        &output,
        "planner_inference_outperformance_fraction",
        &format!("Planner {display_split} Outperformance Fraction"),
        "fraction",
        "evaluation",
        &[fraction as f32],
        ScaleKind::Linear,
    )?;
    for (index, trace) in traces.iter().enumerate() {
        let episode = output.join(format!(
            "planner_{split_lower}_{index:03}_{}",
            sanitize_component(&trace.ticker)
        ));
        fs::create_dir_all(&episode)?;
        write_episode_trace(
            &episode,
            &format!("Planner {display_split} Episode {index}"),
            trace,
        )?;
    }
    fs::File::open(&output)?.sync_all()?;
    fs::rename(&output, &published)?;
    fs::File::open(&generation)?.sync_all()?;
    prune_published_inference_sets(
        &generation,
        &split_lower,
        run_lineage_id,
        update,
        &published,
    )?;
    Ok(published)
}

pub fn has_complete_inference_reports(
    gens: impl AsRef<Path>,
    update: u64,
    run_lineage_id: &str,
    split: &str,
    evaluation_fingerprint: &str,
    expected_episodes: usize,
    expected_rollout_length: usize,
) -> Result<bool> {
    let generation = gens.as_ref().join(update.to_string());
    if !generation.is_dir() {
        return Ok(false);
    }
    require_owner(&generation, run_lineage_id, update)?;
    let split_lower = sanitize_component(&split.to_ascii_lowercase());
    let bundle_prefix = format!("planner_inference_{split_lower}_");
    let episode_prefix = format!("planner_{split_lower}_");

    for entry in fs::read_dir(&generation)?.filter_map(std::result::Result::ok) {
        if !entry.file_type().is_ok_and(|kind| kind.is_dir())
            || !entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(&bundle_prefix))
        {
            continue;
        }
        if read_owner(&entry.path())?.as_ref()
            != Some(&PlannerGenerationOwner {
                run_lineage_id: run_lineage_id.to_owned(),
                update,
            })
        {
            continue;
        }
        let manifest = serde_json::from_slice::<PlannerInferenceManifest>(
            &fs::read(entry.path().join(PLANNER_INFERENCE_MANIFEST)).unwrap_or_default(),
        )
        .ok();
        if manifest.as_ref()
            != Some(&PlannerInferenceManifest {
                version: 1,
                run_lineage_id: run_lineage_id.to_owned(),
                update,
                split: split_lower.clone(),
                evaluation_fingerprint: evaluation_fingerprint.to_owned(),
                episodes: expected_episodes,
                rollout_length: expected_rollout_length,
            })
        {
            continue;
        }
        if !PLANNER_INFERENCE_REPORTS
            .iter()
            .all(|stem| read_report(&entry.path().join(format!("{stem}.report.bin"))).is_some())
        {
            continue;
        }
        let episodes = fs::read_dir(entry.path())?
            .filter_map(std::result::Result::ok)
            .filter(|episode| {
                episode.file_type().is_ok_and(|kind| kind.is_dir())
                    && episode
                        .file_name()
                        .to_str()
                        .is_some_and(|name| name.starts_with(&episode_prefix))
            })
            .collect::<Vec<_>>();
        if episodes.len() == expected_episodes
            && episodes.iter().all(|episode| {
                read_report(&episode.path().join("assets.report.bin"))
                    .is_some_and(|report| report_has_points(&report, expected_rollout_length + 1))
                    && read_report(&episode.path().join("reward.report.bin"))
                        .is_some_and(|report| report_has_points(&report, expected_rollout_length))
                    && read_report(&episode.path().join("planner_position.report.bin"))
                        .is_some_and(|report| report_has_points(&report, expected_rollout_length))
            })
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn report_has_points(report: &Report, expected: usize) -> bool {
    match &report.kind {
        ReportKind::Simple { values, .. } => values.len() == expected,
        ReportKind::MultiLine { series } => {
            !series.is_empty() && series.iter().all(|series| series.values.len() == expected)
        }
        ReportKind::Assets {
            total,
            cash,
            positioned,
            benchmark,
        } => {
            total.len() == expected
                && cash.len() == expected
                && positioned
                    .as_ref()
                    .is_none_or(|values| values.len() == expected)
                && benchmark
                    .as_ref()
                    .is_none_or(|values| values.len() == expected)
        }
        _ => false,
    }
}

fn write_episode_trace(output: &Path, title: &str, trace: &PlannerEpisodeTrace) -> Result<()> {
    trace.validate()?;
    write_report(
        &output.join("assets.report.bin"),
        &Report {
            title: format!("{title} Assets"),
            x_label: Some("step".to_owned()),
            y_label: Some("assets".to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::Assets {
                total: f64_to_f32(&trace.total),
                cash: f64_to_f32(&trace.cash),
                positioned: Some(f64_to_f32(&trace.positioned)),
                benchmark: Some(f64_to_f32(&trace.benchmark)),
            },
        },
    )?;
    write_report(
        &output.join("reward.report.bin"),
        &Report {
            title: format!("{title} Rewards"),
            x_label: Some("step".to_owned()),
            y_label: Some("scaled log return".to_owned()),
            scale: ScaleKind::Symlog,
            kind: ReportKind::Simple {
                values: f64_to_f32(&trace.rewards),
                ema_alpha: None,
            },
        },
    )?;
    write_report(
        &output.join("planner_position.report.bin"),
        &Report {
            title: format!("{title} Position / Exposure"),
            x_label: Some("step".to_owned()),
            y_label: Some("portfolio fraction".to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::MultiLine {
                series: vec![
                    ReportSeries {
                        label: "requested target weight".to_owned(),
                        values: f64_to_f32(&trace.requested_target_weight),
                    },
                    ReportSeries {
                        label: "executed stock exposure".to_owned(),
                        values: f64_to_f32(&trace.executed_stock_weight),
                    },
                ],
            },
        },
    )?;
    fs::File::open(output)?.sync_all()?;
    Ok(())
}

fn write_simple(
    output: &Path,
    stem: &str,
    title: &str,
    y_label: &str,
    values: &[f32],
    scale: ScaleKind,
) -> Result<()> {
    write_simple_with_x(output, stem, title, y_label, "update", values, scale)
}

fn write_simple_allow_nan(
    output: &Path,
    stem: &str,
    title: &str,
    y_label: &str,
    values: &[f32],
    scale: ScaleKind,
) -> Result<()> {
    if values.is_empty() || values.iter().any(|value| value.is_infinite()) {
        bail!("planner report {stem} requires non-empty values without infinity");
    }
    write_report(
        &output.join(format!("{stem}.report.bin")),
        &Report {
            title: title.to_owned(),
            x_label: Some("update".to_owned()),
            y_label: Some(y_label.to_owned()),
            scale,
            kind: ReportKind::Simple {
                values: values.to_vec(),
                ema_alpha: None,
            },
        },
    )
}

fn write_simple_with_x(
    output: &Path,
    stem: &str,
    title: &str,
    y_label: &str,
    x_label: &str,
    values: &[f32],
    scale: ScaleKind,
) -> Result<()> {
    if !values.iter().any(|value| value.is_finite()) {
        return Ok(());
    }
    write_report(
        &output.join(format!("{stem}.report.bin")),
        &Report {
            title: title.to_owned(),
            x_label: Some(x_label.to_owned()),
            y_label: Some(y_label.to_owned()),
            scale,
            kind: ReportKind::Simple {
                values: values.to_vec(),
                ema_alpha: None,
            },
        },
    )
}

fn write_multiline(
    output: &Path,
    stem: &str,
    title: &str,
    y_label: &str,
    series: Vec<ReportSeries>,
) -> Result<()> {
    write_multiline_with_x(output, stem, title, y_label, "update", series)
}

fn write_multiline_with_x(
    output: &Path,
    stem: &str,
    title: &str,
    y_label: &str,
    x_label: &str,
    series: Vec<ReportSeries>,
) -> Result<()> {
    let series = series
        .into_iter()
        .filter(|series| series.values.iter().any(|value| value.is_finite()))
        .collect::<Vec<_>>();
    if series.is_empty() {
        return Ok(());
    }
    write_report(
        &output.join(format!("{stem}.report.bin")),
        &Report {
            title: title.to_owned(),
            x_label: Some(x_label.to_owned()),
            y_label: Some(y_label.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::MultiLine { series },
        },
    )
}

fn write_report(path: &Path, report: &Report) -> Result<()> {
    let bytes = postcard::to_stdvec(report).context("failed encoding planner report")?;
    let temporary = path.with_extension("report.bin.tmp");
    fs::write(&temporary, bytes)
        .with_context(|| format!("failed writing {}", temporary.display()))?;
    fs::File::open(&temporary)?.sync_all()?;
    fs::rename(&temporary, path).with_context(|| format!("failed committing {}", path.display()))
}

fn read_report(path: &Path) -> Option<Report> {
    postcard::from_bytes(&fs::read(path).ok()?).ok()
}

fn write_owner(directory: &Path, owner: &PlannerGenerationOwner) -> Result<()> {
    let marker = directory.join(PLANNER_GENERATION_MARKER);
    fs::write(&marker, serde_json::to_vec(owner)?)?;
    fs::File::open(&marker)?.sync_all()?;
    Ok(())
}

fn read_owner(directory: &Path) -> Result<Option<PlannerGenerationOwner>> {
    let marker = directory.join(PLANNER_GENERATION_MARKER);
    let bytes = match fs::read(&marker) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    Ok(Some(serde_json::from_slice(&bytes).with_context(|| {
        format!("invalid planner generation owner {}", marker.display())
    })?))
}

fn require_owner(directory: &Path, run_lineage_id: &str, update: u64) -> Result<()> {
    let owner = read_owner(directory)?.with_context(|| {
        format!(
            "planner generation {} has no ownership marker",
            directory.display()
        )
    })?;
    if owner.run_lineage_id != run_lineage_id || owner.update != update {
        bail!(
            "planner generation {} owner does not match run lineage/update",
            directory.display()
        );
    }
    Ok(())
}

fn read_simple(path: &Path) -> Option<Vec<f32>> {
    match read_report(path)?.kind {
        ReportKind::Simple { values, .. } => Some(values),
        _ => None,
    }
}

fn read_line(path: &Path, label: &str) -> Option<Vec<f32>> {
    match read_report(path)?.kind {
        ReportKind::MultiLine { series } => series
            .into_iter()
            .find(|series| series.label == label)
            .map(|series| series.values),
        _ => None,
    }
}

fn read_required_simple(path: &Path, expected_len: usize) -> Result<Vec<f32>> {
    let values = read_simple(path)
        .with_context(|| format!("missing or invalid planner report {}", path.display()))?;
    validate_history_len(path, &values, expected_len)?;
    Ok(values)
}

fn read_required_line(path: &Path, label: &str, expected_len: usize) -> Result<Vec<f32>> {
    let values = read_line(path, label).with_context(|| {
        format!(
            "missing planner report series {label:?} in {}",
            path.display()
        )
    })?;
    validate_history_len(path, &values, expected_len)?;
    Ok(values)
}

fn read_optional_simple(path: &Path, expected_len: usize) -> Result<Vec<f32>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    read_required_simple(path, expected_len)
}

fn read_optional_line(path: &Path, label: &str, expected_len: usize) -> Result<Vec<f32>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    read_required_line(path, label, expected_len)
}

fn validate_history_len(path: &Path, values: &[f32], expected_len: usize) -> Result<()> {
    if values.len() != expected_len {
        bail!(
            "planner report {} has {} points; committed update requires {expected_len}",
            path.display(),
            values.len()
        );
    }
    Ok(())
}

fn validate_finite_history(path: &Path, values: &[f32]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        bail!(
            "mandatory planner report {} contains NaN or infinity",
            path.display()
        );
    }
    Ok(())
}

fn update_index(update: u64) -> Result<usize> {
    if update == 0 {
        bail!("planner report updates are one-indexed");
    }
    usize::try_from(update - 1).context("planner update does not fit report index")
}

fn series(label: &str, values: &[f32]) -> ReportSeries {
    ReportSeries {
        label: label.to_owned(),
        values: values.to_vec(),
    }
}

fn f64_to_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|value| *value as f32).collect()
}

fn display_split(split: &str) -> String {
    let mut chars = split.chars();
    chars
        .next()
        .map(|first| first.to_uppercase().chain(chars).collect())
        .unwrap_or_else(|| "Held-Out".to_owned())
}

fn sanitize_component(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn cleanup_inference_staging(
    generation: &Path,
    split_lower: &str,
    run_lineage_id: &str,
    update: u64,
) -> Result<()> {
    let prefix = format!(".planner-inference-{split_lower}-");
    for entry in fs::read_dir(generation)?.filter_map(std::result::Result::ok) {
        if !entry.file_type().is_ok_and(|kind| kind.is_dir())
            || !entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(&prefix) && name.ends_with(".tmp"))
        {
            continue;
        }
        let Some(owner) = read_owner(&entry.path())? else {
            continue;
        };
        if owner.run_lineage_id == run_lineage_id && owner.update == update {
            fs::remove_dir_all(entry.path())?;
        }
    }
    Ok(())
}

fn prune_published_inference_sets(
    generation: &Path,
    split_lower: &str,
    run_lineage_id: &str,
    update: u64,
    keep: &Path,
) -> Result<()> {
    let prefix = format!("planner_inference_{split_lower}_");
    for entry in fs::read_dir(generation)?.filter_map(std::result::Result::ok) {
        if entry.path() == keep
            || !entry.file_type().is_ok_and(|kind| kind.is_dir())
            || !entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(&prefix))
        {
            continue;
        }
        let Some(owner) = read_owner(&entry.path())? else {
            continue;
        };
        if owner.run_lineage_id == run_lineage_id && owner.update == update {
            fs::remove_dir_all(entry.path())?;
        }
    }
    fs::File::open(generation)?.sync_all()?;
    Ok(())
}

fn max_drawdown(total: &[f64]) -> f64 {
    let mut peak = total[0];
    total.iter().fold(0.0_f64, |drawdown, &assets| {
        peak = peak.max(assets);
        drawdown.max(1.0 - assets / peak)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace() -> PlannerEpisodeTrace {
        PlannerEpisodeTrace {
            ticker: "TEST".to_owned(),
            cash: vec![100.0, 49.9, 49.9],
            positioned: vec![0.0, 55.0, 60.0],
            total: vec![100.0, 104.9, 109.9],
            benchmark: vec![100.0, 110.0, 120.0],
            rewards: vec![0.9, 0.8],
            commissions: vec![0.1, 0.0],
            turnover: vec![0.5, 0.1],
            requested_target_weight: vec![0.5, 0.6],
            executed_stock_weight: vec![0.5005, 0.55],
        }
    }

    fn deterministic_trace() -> PlannerEpisodeTrace {
        PlannerEpisodeTrace {
            ticker: "TEST".to_owned(),
            cash: vec![100.0, 74.95, 74.95],
            positioned: vec![0.0, 27.5, 30.0],
            total: vec![100.0, 102.45, 104.95],
            benchmark: vec![100.0, 110.0, 120.0],
            rewards: vec![0.7, 0.6],
            commissions: vec![0.05, 0.0],
            turnover: vec![0.25, 0.05],
            requested_target_weight: vec![0.25, 0.3],
            executed_stock_weight: vec![0.2501, 0.286],
        }
    }

    #[test]
    fn native_history_writes_exact_assets_and_requested_vs_executed_position() {
        let dir = std::env::temp_dir().join(format!(
            "planner-native-reports-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let mut history = PlannerReportHistory::load(&dir, 0, "run-a").unwrap();
        history
            .stage_training(
                1,
                PlannerTrainingReportPoint {
                    wealth_ratio: 1.099,
                    buy_and_hold_wealth_ratio: 1.2,
                    requested_target_weight_mean: 0.55,
                    executed_stock_weight_mean: 0.525,
                    deterministic_reward_mean: 0.65,
                    deterministic_wealth_ratio: 1.0495,
                    deterministic_mean_outperformance_ratio: -0.1505,
                    deterministic_median_outperformance_ratio: -0.15,
                    deterministic_outperformance_fraction: 0.25,
                    deterministic_turnover_mean: 0.15,
                    deterministic_commissions: 0.05,
                    deterministic_requested_target_weight_mean: 0.275,
                    deterministic_executed_stock_weight_mean: 0.268,
                    deterministic_action_boundary_fraction: 0.0,
                    ..PlannerTrainingReportPoint::default()
                },
                &trace(),
                &deterministic_trace(),
            )
            .unwrap()
            .publish()
            .unwrap();
        let assets = read_report(&dir.join("1/assets.report.bin")).unwrap();
        match assets.kind {
            ReportKind::Assets {
                cash,
                positioned,
                benchmark,
                ..
            } => {
                assert_eq!(cash, vec![100.0, 49.9, 49.9]);
                assert_eq!(positioned.unwrap(), vec![0.0, 55.0, 60.0]);
                assert_eq!(benchmark.unwrap(), vec![100.0, 110.0, 120.0]);
            }
            _ => panic!("expected assets report"),
        }
        let position = read_report(&dir.join("1/planner_position.report.bin")).unwrap();
        match position.kind {
            ReportKind::MultiLine { series } => {
                assert_eq!(series[0].label, "requested target weight");
                assert_eq!(series[1].label, "executed stock exposure");
                assert_ne!(series[0].values, series[1].values);
            }
            _ => panic!("expected position report"),
        }
        let sampled_wealth = read_report(&dir.join("1/planner_wealth.report.bin")).unwrap();
        assert!(sampled_wealth.title.contains("Sampled On-Policy"));
        let deterministic_wealth =
            read_report(&dir.join("1/planner_deterministic_wealth.report.bin")).unwrap();
        assert!(deterministic_wealth
            .title
            .contains("Deterministic Beta-Mean"));
        match deterministic_wealth.kind {
            ReportKind::MultiLine { series } => {
                assert_eq!(series[0].label, "deterministic Beta-mean policy wealth");
                assert_eq!(series[0].values, vec![1.0495]);
                assert_eq!(series[1].label, "buy-and-hold");
            }
            _ => panic!("expected deterministic wealth report"),
        }
        let deterministic_assets =
            read_report(&dir.join("1/deterministic_mean/assets.report.bin")).unwrap();
        match deterministic_assets.kind {
            ReportKind::Assets { total, .. } => {
                assert_eq!(total, vec![100.0, 102.45, 104.95]);
            }
            _ => panic!("expected deterministic assets report"),
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn resume_loads_native_history_and_keeps_update_alignment() {
        let dir = std::env::temp_dir().join(format!(
            "planner-native-resume-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let mut history = PlannerReportHistory::load(&dir, 0, "run-a").unwrap();
        history
            .stage_training(
                1,
                PlannerTrainingReportPoint {
                    reward_mean: 1.0,
                    ..PlannerTrainingReportPoint::default()
                },
                &trace(),
                &trace(),
            )
            .unwrap()
            .publish()
            .unwrap();
        let mut resumed = PlannerReportHistory::load(&dir, 1, "run-a").unwrap();
        resumed
            .stage_training(
                2,
                PlannerTrainingReportPoint {
                    reward_mean: 2.0,
                    ..PlannerTrainingReportPoint::default()
                },
                &trace(),
                &trace(),
            )
            .unwrap()
            .publish()
            .unwrap();
        assert_eq!(
            read_simple(&dir.join("2/planner_reward.report.bin")).unwrap(),
            vec![1.0, 2.0]
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn resume_from_generation_before_deterministic_reports_preserves_alignment() {
        let dir = std::env::temp_dir().join(format!(
            "planner-deterministic-resume-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let mut history = PlannerReportHistory::load(&dir, 0, "run-a").unwrap();
        history
            .stage_training(
                1,
                PlannerTrainingReportPoint::default(),
                &trace(),
                &deterministic_trace(),
            )
            .unwrap()
            .publish()
            .unwrap();
        for stem in [
            "planner_deterministic_wealth",
            "planner_deterministic_reward",
            "planner_deterministic_outperformance",
            "planner_deterministic_outperformance_fraction",
            "planner_deterministic_position_mean",
            "planner_deterministic_turnover",
            "planner_deterministic_commissions",
        ] {
            fs::remove_file(dir.join(format!("1/{stem}.report.bin"))).unwrap();
        }
        fs::remove_dir_all(dir.join("1/deterministic_mean")).unwrap();

        let mut resumed = PlannerReportHistory::load(&dir, 1, "run-a").unwrap();
        resumed
            .stage_training(
                2,
                PlannerTrainingReportPoint {
                    deterministic_reward_mean: 0.2,
                    ..PlannerTrainingReportPoint::default()
                },
                &trace(),
                &deterministic_trace(),
            )
            .unwrap()
            .publish()
            .unwrap();
        let values = read_simple(&dir.join("2/planner_deterministic_reward.report.bin")).unwrap();
        assert!(values[0].is_nan());
        assert_eq!(values[1], 0.2);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn staged_generation_is_invisible_until_publish_and_committed_corruption_fails_resume() {
        let dir = std::env::temp_dir().join(format!(
            "planner-native-atomic-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let mut history = PlannerReportHistory::load(&dir, 0, "run-a").unwrap();
        let staged = history
            .stage_training(1, PlannerTrainingReportPoint::default(), &trace(), &trace())
            .unwrap();
        assert!(!dir.join("1").exists());
        staged.publish().unwrap();
        assert!(PlannerReportHistory::load(&dir, 1, "run-a").is_ok());

        fs::remove_file(dir.join("1/planner_reward.report.bin")).unwrap();
        let error = PlannerReportHistory::load(&dir, 1, "run-a").unwrap_err();
        assert!(error.to_string().contains("planner_reward.report.bin"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn validation_history_survives_non_validation_updates_and_resume() {
        let dir = std::env::temp_dir().join(format!(
            "planner-native-validation-resume-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let mut history = PlannerReportHistory::load(&dir, 0, "run-a").unwrap();
        history
            .stage_training(1, PlannerTrainingReportPoint::default(), &trace(), &trace())
            .unwrap()
            .publish()
            .unwrap();
        history
            .record_validation(
                1,
                PlannerValidationReportPoint {
                    median_wealth_ratio: 1.1,
                    ..PlannerValidationReportPoint::default()
                },
            )
            .unwrap();
        history
            .stage_training(2, PlannerTrainingReportPoint::default(), &trace(), &trace())
            .unwrap()
            .publish()
            .unwrap();

        let mut resumed = PlannerReportHistory::load(&dir, 2, "run-a").unwrap();
        resumed
            .stage_training(3, PlannerTrainingReportPoint::default(), &trace(), &trace())
            .unwrap()
            .publish()
            .unwrap();
        resumed
            .record_validation(
                3,
                PlannerValidationReportPoint {
                    median_wealth_ratio: 1.3,
                    ..PlannerValidationReportPoint::default()
                },
            )
            .unwrap();

        let values = read_line(
            &dir.join("3/planner_validation_wealth.report.bin"),
            "policy median wealth",
        )
        .unwrap();
        assert_eq!(values[0], 1.1);
        assert!(values[1].is_nan());
        assert_eq!(values[2], 1.3);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn startup_cleanup_removes_only_uncommitted_planner_generations() {
        let dir = std::env::temp_dir().join(format!(
            "planner-native-generation-cleanup-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        for (generation, update) in [("1", 1), ("2", 2), (".3.planner-reports.tmp", 3)] {
            let path = dir.join(generation);
            fs::create_dir_all(&path).unwrap();
            write_owner(
                &path,
                &PlannerGenerationOwner {
                    run_lineage_id: "run-a".to_owned(),
                    update,
                },
            )
            .unwrap();
        }
        cleanup_uncommitted_report_generations(&dir, 1, "run-a").unwrap();

        assert!(dir.join("1").exists());
        assert!(!dir.join("2").exists());
        assert!(!dir.join(".3.planner-reports.tmp").exists());

        fs::create_dir_all(dir.join("4")).unwrap();
        fs::write(dir.join("4/unrelated.report.bin"), b"unrelated").unwrap();
        fs::create_dir_all(dir.join("5")).unwrap();
        write_owner(
            &dir.join("5"),
            &PlannerGenerationOwner {
                run_lineage_id: "run-b".to_owned(),
                update: 5,
            },
        )
        .unwrap();

        assert!(cleanup_uncommitted_report_generations(&dir, 1, "run-a").is_err());
        assert!(dir.join("4").exists());
        assert!(dir.join("5").exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn publish_refuses_and_preserves_an_existing_unowned_generation() {
        let dir = std::env::temp_dir().join(format!(
            "planner-native-unowned-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(dir.join("1")).unwrap();
        fs::write(dir.join("1/unrelated.report.bin"), b"keep me").unwrap();
        let mut history = PlannerReportHistory::load(&dir, 0, "run-a").unwrap();
        let staged = history
            .stage_training(1, PlannerTrainingReportPoint::default(), &trace(), &trace())
            .unwrap();

        assert!(staged.publish().is_err());
        assert_eq!(
            fs::read(dir.join("1/unrelated.report.bin")).unwrap(),
            b"keep me"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn nan_critic_ev_is_persisted_and_resumable() {
        let dir = std::env::temp_dir().join(format!(
            "planner-native-nan-ev-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let mut history = PlannerReportHistory::load(&dir, 0, "run-a").unwrap();
        history
            .stage_training(
                1,
                PlannerTrainingReportPoint {
                    critic_explained_variance: f64::NAN,
                    ..PlannerTrainingReportPoint::default()
                },
                &trace(),
                &trace(),
            )
            .unwrap()
            .publish()
            .unwrap();
        assert!(read_simple(&dir.join("1/explained_var.report.bin")).unwrap()[0].is_nan());
        PlannerReportHistory::load(&dir, 1, "run-a").unwrap();
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn interrupted_inference_keeps_previous_complete_set_visible() {
        let dir = std::env::temp_dir().join(format!(
            "planner-native-inference-atomic-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let mut history = PlannerReportHistory::load(&dir, 0, "run-a").unwrap();
        history
            .stage_training(1, PlannerTrainingReportPoint::default(), &trace(), &trace())
            .unwrap()
            .publish()
            .unwrap();
        let first =
            write_inference_reports(&dir, 1, "run-a", "Test", &[trace()], "contract-a").unwrap();
        let partial = dir.join("1/.planner-inference-test-interrupted.tmp");
        fs::create_dir(&partial).unwrap();
        write_owner(
            &partial,
            &PlannerGenerationOwner {
                run_lineage_id: "run-a".to_owned(),
                update: 1,
            },
        )
        .unwrap();
        fs::write(
            partial.join("planner_inference_wealth.report.bin"),
            b"partial",
        )
        .unwrap();

        assert!(first.join("planner_inference_wealth.report.bin").is_file());
        assert!(read_report(&first.join("planner_inference_wealth.report.bin")).is_some());
        let second =
            write_inference_reports(&dir, 1, "run-a", "Test", &[trace()], "contract-a").unwrap();
        assert!(!partial.exists());
        assert!(!first.exists());
        assert!(second.join("planner_inference_wealth.report.bin").is_file());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn complete_inference_detection_requires_owner_aggregates_and_all_episodes() {
        let dir = std::env::temp_dir().join(format!(
            "planner-native-inference-complete-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let mut history = PlannerReportHistory::load(&dir, 0, "run-a").unwrap();
        history
            .stage_training(1, PlannerTrainingReportPoint::default(), &trace(), &trace())
            .unwrap()
            .publish()
            .unwrap();
        let bundle =
            write_inference_reports(&dir, 1, "run-a", "Test", &[trace(), trace()], "contract-a")
                .unwrap();

        assert!(
            has_complete_inference_reports(&dir, 1, "run-a", "Test", "contract-a", 2, 2).unwrap()
        );
        assert!(
            !has_complete_inference_reports(&dir, 1, "run-a", "Test", "contract-b", 2, 2).unwrap()
        );
        assert!(
            !has_complete_inference_reports(&dir, 1, "run-a", "Test", "contract-a", 3, 2).unwrap()
        );
        let episode = fs::read_dir(&bundle)
            .unwrap()
            .filter_map(std::result::Result::ok)
            .find(|entry| {
                entry.file_type().is_ok_and(|kind| kind.is_dir())
                    && entry
                        .file_name()
                        .to_str()
                        .is_some_and(|name| name.starts_with("planner_test_"))
            })
            .unwrap();
        fs::remove_file(episode.path().join("assets.report.bin")).unwrap();
        assert!(
            !has_complete_inference_reports(&dir, 1, "run-a", "Test", "contract-a", 2, 2).unwrap()
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
