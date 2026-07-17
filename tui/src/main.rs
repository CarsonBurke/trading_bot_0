use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, MouseButton, MouseEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Frame, Terminal};
use serde::Deserialize;
use shared::paths::{RUNS_PATH, WEIGHTS_PATH};
use shared::run_dir::RunDir;
use std::{
    collections::hash_map::DefaultHasher,
    fs,
    hash::{Hash, Hasher},
    io,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

mod chart_viewer;
mod components;
mod pages;
mod report_renderer;
mod state;
mod theme;
mod utils;

use chart_viewer::ChartViewer;
use state::{GenerationBrowserState, InferenceBrowserState, LogsPageState, ProcessManagerState};
use state::{GeneticFamily as TuiGeneticFamily, TrainingKind};

const TRAINING_KINDS: [TrainingKind; 3] = [
    TrainingKind::Rl,
    TrainingKind::Genetic,
    TrainingKind::Pretrain,
];
const GENETIC_FAMILIES: [TuiGeneticFamily; 3] = [
    TuiGeneticFamily::TrendBreakout,
    TuiGeneticFamily::PriceRebound,
    TuiGeneticFamily::RsiRebound,
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AppMode {
    Main,
    GenerationBrowser,
    InferenceBrowser,
    ChartViewer,
    Logs,
    ModelObservations,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RunSelectorPurpose {
    View,
    Train,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunInfo {
    pub name: String,
    pub gen_count: usize,
    pub weights: Vec<String>, // .ot filenames sorted newest-first
    pub is_active: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DialogMode {
    None,
    InferenceInput {
        focused_field: InferenceField,
    },
    ConfirmQuit,
    ConfirmStopTraining,
    PageJump {
        selected: usize,
    },
    RunSelector {
        selected: usize,
        runs: Vec<RunInfo>,
        purpose: RunSelectorPurpose,
    },
    WeightsSelector {
        run_name: String,
        selected: usize,
        weights: Vec<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InferenceField {
    Weights,
    Ticker,
    Episodes,
}

pub struct App {
    pub mode: AppMode,
    pub previous_mode: AppMode,
    pub dialog_mode: DialogMode,
    pub chart_viewer: ChartViewer,
    pub input: String,
    pub ticker_input: String,
    pub episodes_input: String,
    pub weights_path: Option<String>,
    pub training_model_size: String,
    pub training_kind: TrainingKind,
    pub genetic_family: TuiGeneticFamily,
    pub latest_meta_charts: Vec<PathBuf>,
    meta_reports_revision: u64,
    last_refresh: Instant,
    pub generation_browser: GenerationBrowserState,
    pub inference_browser: InferenceBrowserState,
    pub logs_page: LogsPageState,
    pub process_manager: ProcessManagerState,
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip ESC [ ... (final byte is 0x40-0x7E)
            if chars.next() == Some('[') {
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() || c == '~' {
                        break;
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn newest_run_activity(path: &Path) -> Option<std::time::SystemTime> {
    let mut latest = fs::metadata(path).ok()?.modified().ok();

    for child in ["training.log", "gens", "weights"] {
        let modified = match fs::metadata(path.join(child))
            .ok()
            .and_then(|metadata| metadata.modified().ok())
        {
            Some(modified) => modified,
            None => continue,
        };
        latest = Some(latest.map_or(modified, |current| current.max(modified)));
    }

    latest
}

fn sort_run_dirs_newest_first(dirs: &mut [std::fs::DirEntry]) {
    dirs.sort_by(|a, b| {
        let key = |entry: &std::fs::DirEntry| {
            let name = entry.file_name().to_string_lossy().to_string();
            let activity = newest_run_activity(&entry.path());
            (activity, name)
        };

        key(b).cmp(&key(a))
    });
}

const PLANNER_INFERENCE_REPORTS: [&str; 6] = [
    "planner_inference_wealth",
    "planner_inference_outperformance",
    "planner_inference_outperformance_fraction",
    "planner_inference_risk",
    "planner_inference_action",
    "planner_inference_commissions",
];

fn latest_complete_planner_inference_bundle(generation: &Path) -> Option<PathBuf> {
    let generation_owner = serde_json::from_slice::<PlannerOwnerView>(
        &fs::read(generation.join(".planner-report-generation")).ok()?,
    )
    .ok()?;
    let mut bundles = fs::read_dir(generation)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| {
            let manifest = serde_json::from_slice::<PlannerInferenceManifestView>(
                &fs::read(entry.path().join(".planner-inference.json")).unwrap_or_default(),
            )
            .ok();
            entry.file_type().is_ok_and(|kind| kind.is_dir())
                && manifest.as_ref().is_some_and(|manifest| {
                    manifest.version == 1
                        && manifest.episodes > 0
                        && manifest.rollout_length > 0
                        && !manifest.evaluation_fingerprint.is_empty()
                        && manifest.run_lineage_id == generation_owner.run_lineage_id
                        && manifest.update == generation_owner.update
                        && entry.file_name().to_str().is_some_and(|name| {
                            name.starts_with(&format!("planner_inference_{}_", manifest.split))
                        })
                })
                && PLANNER_INFERENCE_REPORTS
                    .iter()
                    .all(|base| entry.path().join(format!("{base}.report.bin")).is_file())
                && serde_json::from_slice::<PlannerOwnerView>(
                    &fs::read(entry.path().join(".planner-report-generation")).unwrap_or_default(),
                )
                .ok()
                .as_ref()
                    == Some(&generation_owner)
        })
        .collect::<Vec<_>>();
    bundles.sort_by_key(|entry| {
        (
            fs::metadata(entry.path())
                .and_then(|metadata| metadata.modified())
                .ok(),
            entry.file_name(),
        )
    });
    bundles.pop().map(|entry| entry.path())
}

#[derive(Deserialize)]
struct PlannerManifestView {
    version: u32,
    run_lineage_id: String,
    update: u64,
    checkpoint_file: String,
}

#[derive(Deserialize, PartialEq, Eq)]
struct PlannerOwnerView {
    run_lineage_id: String,
    update: u64,
}

#[derive(Deserialize)]
struct PlannerInferenceManifestView {
    version: u32,
    run_lineage_id: String,
    update: u64,
    split: String,
    evaluation_fingerprint: String,
    episodes: usize,
    rollout_length: usize,
}

fn planner_committed_updates(gens: &Path) -> std::collections::HashMap<String, u64> {
    let Some(weights) = gens.parent().map(|root| root.join("weights")) else {
        return std::collections::HashMap::new();
    };
    fs::read_dir(&weights)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.ends_with(".resume.json"))
        })
        .filter_map(|entry| {
            let manifest =
                serde_json::from_slice::<PlannerManifestView>(&fs::read(entry.path()).ok()?)
                    .ok()?;
            let checkpoint = weights.join(&manifest.checkpoint_file);
            if manifest.version != 1
                || manifest.checkpoint_file.contains('/')
                || manifest.checkpoint_file.contains('\\')
                || !checkpoint.is_file()
                || !checkpoint.with_extension("metadata.json").is_file()
                || !checkpoint.with_extension("optimizer.ot").is_file()
            {
                return None;
            }
            Some((manifest.run_lineage_id, manifest.update))
        })
        .fold(
            std::collections::HashMap::new(),
            |mut updates, (lineage, update)| {
                updates
                    .entry(lineage)
                    .and_modify(|current| *current = (*current).max(update))
                    .or_insert(update);
                updates
            },
        )
}

fn planner_generation_visible(
    generation: &Path,
    generation_number: u64,
    committed_updates: &std::collections::HashMap<String, u64>,
) -> bool {
    let marker = generation.join(".planner-report-generation");
    if !marker.is_file() {
        return true;
    }
    serde_json::from_slice::<PlannerOwnerView>(&fs::read(marker).unwrap_or_default())
        .ok()
        .is_some_and(|owner| {
            owner.update == generation_number
                && committed_updates
                    .get(&owner.run_lineage_id)
                    .is_some_and(|committed| generation_number <= *committed)
        })
}

impl App {
    fn coerce_weights_filename(input: &str) -> String {
        let trimmed = input.trim();

        // If it's just a number, expand to ppo_ep{N}.ot
        if trimmed.parse::<u32>().is_ok() {
            return format!("ppo_ep{}.ot", trimmed);
        }

        // If it already has the pattern, use as-is
        trimmed.to_string()
    }

    fn new() -> Result<Self> {
        let mut inference_browser = InferenceBrowserState::new();
        inference_browser.load_inferences()?;

        let process_manager = ProcessManagerState::new();
        let mut generation_browser = GenerationBrowserState::new();
        if let Some(run) = &process_manager.active_run {
            generation_browser.gens_path = run.gens.clone();
        }
        generation_browser.load_generations()?;

        let mut app = App {
            mode: AppMode::Main,
            previous_mode: AppMode::Main,
            dialog_mode: DialogMode::None,
            chart_viewer: ChartViewer::new(),
            input: String::new(),
            ticker_input: String::new(),
            episodes_input: String::new(),
            weights_path: None,
            training_model_size: "uniform-stream".to_string(),
            training_kind: TrainingKind::Rl,
            genetic_family: TuiGeneticFamily::TrendBreakout,
            latest_meta_charts: Vec::new(),
            meta_reports_revision: 0,
            last_refresh: Instant::now(),
            generation_browser,
            inference_browser,
            logs_page: LogsPageState::new(),
            process_manager,
        };

        app.load_latest_meta_charts()?;
        app.meta_reports_revision = app.current_meta_reports_revision();
        Ok(app)
    }

    fn load_latest_meta_charts(&mut self) -> Result<()> {
        use std::collections::HashMap;
        use std::fs;
        use std::time::SystemTime;

        self.latest_meta_charts.clear();

        let gens_path = match &self.process_manager.active_run {
            Some(run) => run.gens.clone(),
            None => PathBuf::from("../training/runs/latest/gens"),
        };
        if !gens_path.exists() {
            return Ok(());
        }
        let planner_committed_updates = planner_committed_updates(&gens_path);

        // Meta chart base names (episode-level charts without ticker)
        let meta_chart_bases = vec![
            "assets",
            "reward",
            "normalized_reward",
            "final_assets",
            "cumulative_reward",
            "outperformance",
            "outperformance_fraction",
            "advantage_stats_log",
            "total_commissions",
            "beta_policy",
            "actor_grad_norm",
            "critic_grad_norm",
            "target_weights",
            "clip_fraction",
            "clip_gap",
            "explained_var",
            "value_loss",
            "policy_loss",
            "policy_entropy",
            "approx_kl",
            "kl_lr",
            "gate_stats",
            "ga_fitness",
            "ga_return_pct",
            "ga_outperformance",
            "ga_max_drawdown",
            "ga_sharpe",
            "ga_turnover",
            "ga_total_commissions",
            "ga_trade_count",
            "ga_generalization_gap",
            "ga_distribution",
            "ga_mutation_entropy",
            "ga_train_assets",
            "ga_validation_assets",
            "ga_test_assets",
            "pretrain_horizon_error",
            "pretrain_horizon_uncertainty",
            "pretrain_horizon_std_error",
            "pretrain_probe_mse",
            "pretrain_sigreg",
            "pretrain_jepa_mse",
            "pretrain_repr_std_mean",
            "pretrain_repr_std_min",
            "pretrain_pred_embed_std",
            "pretrain_target_embed_std",
            "pretrain_probe_mae",
            "pretrain_probe_explained_variance",
            "pretrain_pred_std",
            "pretrain_target_std",
            "pretrain_probe_terminal_mse",
            "planner_wealth",
            "planner_reward",
            "planner_position",
            "planner_position_mean",
            "planner_outperformance",
            "planner_turnover",
            "planner_commissions",
            "planner_deterministic_wealth",
            "planner_deterministic_reward",
            "planner_deterministic_position_mean",
            "planner_deterministic_outperformance",
            "planner_deterministic_outperformance_fraction",
            "planner_deterministic_turnover",
            "planner_deterministic_commissions",
            "planner_validation_wealth",
            "planner_validation_outperformance",
            "planner_validation_risk",
            "planner_validation_selection",
            "planner_validation_outperformance_fraction",
            "planner_inference_wealth",
            "planner_inference_outperformance",
            "planner_inference_outperformance_fraction",
            "planner_inference_risk",
            "planner_inference_action",
            "planner_inference_commissions",
        ];

        // Ticker-specific chart base names
        let ticker_chart_bases = vec![
            "assets",
            "buy_sell",
            "raw_action",
            "reward",
            "planner_position",
        ];

        // Track the latest file for each chart type: base_name -> (modified_time, path)
        let mut latest_per_type: HashMap<String, (SystemTime, PathBuf)> = HashMap::new();

        // Candle-snapshot window reports, kept only for the most recent global_step.
        let mut candle_snapshots: Vec<(usize, PathBuf)> = Vec::new();

        // Scan all generation directories
        if let Ok(entries) = fs::read_dir(&gens_path) {
            for entry in entries.filter_map(|e| e.ok()) {
                if !entry
                    .file_type()
                    .ok()
                    .map(|ft| ft.is_dir())
                    .unwrap_or(false)
                {
                    continue;
                }
                // Only process numeric directories (generation folders)
                let Some(generation_number) = entry
                    .file_name()
                    .to_str()
                    .and_then(|name| name.parse::<u64>().ok())
                else {
                    continue;
                };

                let gen_path = entry.path();
                if !planner_generation_visible(
                    &gen_path,
                    generation_number,
                    &planner_committed_updates,
                ) {
                    continue;
                }

                // Process episode-level meta charts
                for base in &meta_chart_bases {
                    let report_path = gen_path.join(format!("{base}.report.bin"));
                    if !report_path.exists() {
                        continue;
                    }
                    if let Ok(metadata) = fs::metadata(&report_path) {
                        if let Ok(modified) = metadata.modified() {
                            let key = format!("meta_{}", base);
                            if latest_per_type
                                .get(&key)
                                .map(|(t, _)| modified > *t)
                                .unwrap_or(true)
                            {
                                latest_per_type.insert(key, (modified, report_path));
                            }
                        }
                    }
                }

                // Process ticker-specific charts in subdirectories
                if let Ok(items) = fs::read_dir(&gen_path) {
                    for item in items.filter_map(|e| e.ok()) {
                        let item_path = item.path();
                        if !item.file_type().ok().map(|ft| ft.is_dir()).unwrap_or(false) {
                            continue;
                        }
                        let ticker_name = item.file_name();
                        let ticker_str = ticker_name.to_str().unwrap_or("");

                        for base in &ticker_chart_bases {
                            let report_path = item_path.join(format!("{base}.report.bin"));
                            if !report_path.exists() {
                                continue;
                            }
                            if let Ok(metadata) = fs::metadata(&report_path) {
                                if let Ok(modified) = metadata.modified() {
                                    let key = format!("{}_{}", ticker_str, base);
                                    if latest_per_type
                                        .get(&key)
                                        .map(|(t, _)| modified > *t)
                                        .unwrap_or(true)
                                    {
                                        latest_per_type.insert(key, (modified, report_path));
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some(bundle_path) = latest_complete_planner_inference_bundle(&gen_path) {
                    for base in PLANNER_INFERENCE_REPORTS {
                        let report_path = bundle_path.join(format!("{base}.report.bin"));
                        if let Some(modified) = fs::metadata(&report_path)
                            .ok()
                            .and_then(|metadata| metadata.modified().ok())
                        {
                            let key = format!("meta_{base}");
                            if latest_per_type
                                .get(&key)
                                .is_none_or(|(current, _)| modified > *current)
                            {
                                latest_per_type.insert(key, (modified, report_path));
                            }
                        }
                    }
                    if let Ok(episodes) = fs::read_dir(&bundle_path) {
                        for episode in episodes
                            .filter_map(Result::ok)
                            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
                        {
                            for base in &ticker_chart_bases {
                                let report_path = episode.path().join(format!("{base}.report.bin"));
                                if let Some(modified) = fs::metadata(&report_path)
                                    .ok()
                                    .and_then(|metadata| metadata.modified().ok())
                                {
                                    let key = format!(
                                        "{}_{}",
                                        episode.file_name().to_string_lossy(),
                                        base
                                    );
                                    if latest_per_type
                                        .get(&key)
                                        .is_none_or(|(current, _)| modified > *current)
                                    {
                                        latest_per_type.insert(key, (modified, report_path));
                                    }
                                }
                            }
                        }
                    }
                }

                let samples_path = gen_path.join("samples");
                if samples_path.is_dir() {
                    if let Ok(items) = fs::read_dir(&samples_path) {
                        for item in items.filter_map(|e| e.ok()) {
                            if !item
                                .file_type()
                                .ok()
                                .map(|ft| ft.is_file())
                                .unwrap_or(false)
                            {
                                continue;
                            }
                            let file_name = item.file_name();
                            let file_name = file_name.to_str().unwrap_or("");
                            if !file_name.ends_with(".report.bin") {
                                continue;
                            }
                            let report_path = item.path();
                            if let Ok(metadata) = fs::metadata(&report_path) {
                                if let Ok(modified) = metadata.modified() {
                                    let base = file_name.trim_end_matches(".report.bin");
                                    let key = format!("pretrain_samples_{base}");
                                    if latest_per_type
                                        .get(&key)
                                        .map(|(t, _)| modified > *t)
                                        .unwrap_or(true)
                                    {
                                        latest_per_type.insert(key, (modified, report_path));
                                    }
                                }
                            }
                        }
                    }
                }

                let snapshot_path = gen_path.join("candle_snapshots");
                if snapshot_path.is_dir() {
                    if let Ok(items) = fs::read_dir(&snapshot_path) {
                        for item in items.filter_map(|e| e.ok()) {
                            let file_name = item.file_name();
                            let file_name = file_name.to_str().unwrap_or("");
                            if !file_name.ends_with("_candles.report.bin") {
                                continue;
                            }
                            if let Some(step) = file_name
                                .strip_prefix("step")
                                .and_then(|s| s.split('_').next())
                                .and_then(|s| s.parse::<usize>().ok())
                            {
                                candle_snapshots.push((step, item.path()));
                            }
                        }
                    }
                }
            }
        }

        // Collect paths from all found chart types
        for (_, (_, path)) in latest_per_type {
            self.latest_meta_charts.push(path);
        }

        // Keep only the most recent global_step's candle-snapshot windows.
        if let Some(latest_step) = candle_snapshots.iter().map(|(s, _)| *s).max() {
            for (_, path) in candle_snapshots
                .into_iter()
                .filter(|(s, _)| *s == latest_step)
            {
                self.latest_meta_charts.push(path);
            }
        }

        // Sort by filename for consistent ordering
        self.latest_meta_charts.sort();

        Ok(())
    }

    fn pretrain_meta_reports(&self) -> Vec<(String, shared::report::Report)> {
        let run_root = match &self.process_manager.active_run {
            Some(run) => run.root.clone(),
            None => PathBuf::from("../training/runs/latest"),
        };
        utils::pretrain::run_reports(&run_root)
    }

    fn current_meta_reports_revision(&self) -> u64 {
        let run_root = self
            .process_manager
            .active_run
            .as_ref()
            .map(|run| run.root.clone())
            .unwrap_or_else(|| PathBuf::from("../training/runs/latest"));
        let mut hasher = DefaultHasher::new();
        run_root.hash(&mut hasher);
        let mut inputs = self.latest_meta_charts.clone();
        for directory in [&run_root, &run_root.join("weights")] {
            if let Ok(entries) = fs::read_dir(directory) {
                inputs.extend(
                    entries
                        .filter_map(Result::ok)
                        .map(|entry| entry.path())
                        .filter(|path| {
                            path.extension().is_some_and(|extension| extension == "csv")
                        }),
                );
            }
        }
        inputs.sort();
        inputs.dedup();
        for path in inputs {
            path.hash(&mut hasher);
            if let Ok(metadata) = fs::metadata(&path) {
                metadata.len().hash(&mut hasher);
                metadata
                    .modified()
                    .ok()
                    .and_then(|modified| modified.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|duration| duration.as_nanos())
                    .hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    pub fn is_training_running(&mut self) -> bool {
        self.process_manager.is_training_running()
    }

    fn is_anything_running(&mut self) -> bool {
        self.process_manager.is_anything_running()
    }

    pub fn get_current_episode(&self) -> Option<usize> {
        for line in self.logs_page.training_output.iter().rev() {
            // Look for actual episode completion logs: "Episode N - Total Assets..."
            // Skip RL progress logs: "[Ep N] Episodes: ..."
            if line.contains("Episode") && line.contains("Total Assets") && !line.starts_with("[Ep")
            {
                if let Some(ep_str) = line.split("Episode").nth(1) {
                    // Strip ANSI escape sequences first, then grab the number
                    let stripped: String = strip_ansi(ep_str);
                    if let Some(num_str) = stripped.trim().split_whitespace().next() {
                        if let Ok(ep) = num_str.parse::<usize>() {
                            return Some(ep);
                        }
                    }
                }
            }
        }
        None
    }

    pub fn has_training_progress(&self) -> bool {
        self.logs_page.training_output.iter().rev().any(|line| {
            line.contains("ppo update:")
                || line.contains("Epoch ")
                || line.contains("pretrain epoch ")
                || line.contains("pretrain step ")
                || line.contains("planner update=")
                || line.contains("planner validation update=")
                || line.contains("Policy:")
                || (line.contains("Episode") && line.contains("Total Assets"))
        })
    }

    fn maybe_refresh(&mut self) -> Result<()> {
        let now = Instant::now();
        if now.duration_since(self.last_refresh) >= Duration::from_secs(1) {
            self.process_manager.poll_training_process();
            self.sync_gens_path();
            self.generation_browser.load_generations()?;
            self.inference_browser.load_inferences()?;
            self.load_latest_meta_charts()?;
            let revision = self.current_meta_reports_revision();
            if revision != self.meta_reports_revision
                && self.mode == AppMode::ChartViewer
                && self.chart_viewer.is_viewing_meta_charts()
            {
                let extra = self.pretrain_meta_reports();
                self.chart_viewer
                    .load_charts(&self.latest_meta_charts, extra)?;
            }
            self.meta_reports_revision = revision;
            let log_path = self
                .process_manager
                .active_run
                .as_ref()
                .map(|r| r.log_file.to_string_lossy().to_string());
            self.logs_page.poll_training_output(log_path.as_deref());
            self.last_refresh = now;
        }
        Ok(())
    }

    fn start_training(&mut self, weights_path: Option<String>) -> Result<()> {
        let result = self.process_manager.start_training(
            self.training_kind,
            weights_path,
            self.genetic_family,
        );
        self.sync_gens_path();
        result
    }

    fn sync_gens_path(&mut self) {
        if let Some(run) = &self.process_manager.active_run {
            self.generation_browser.gens_path = run.gens.clone();
        }
    }

    fn open_run_selector(&mut self, purpose: RunSelectorPurpose) {
        let runs_dir = std::path::Path::new(RUNS_PATH);
        let mut dirs: Vec<_> = fs::read_dir(runs_dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir() && e.file_name() != "latest")
            .collect();
        sort_run_dirs_newest_first(&mut dirs);

        let viewed_name = self
            .process_manager
            .active_run
            .as_ref()
            .and_then(|r| r.root.file_name().map(|n| n.to_string_lossy().to_string()));
        let live_name = self
            .process_manager
            .live_run()
            .and_then(|run| run.root.file_name())
            .map(|name| name.to_string_lossy().to_string());

        let runs: Vec<RunInfo> = dirs
            .iter()
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().to_string();
                let gens = entry.path().join("gens");
                let gen_count = fs::read_dir(&gens)
                    .into_iter()
                    .flatten()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_type().map_or(false, |ft| ft.is_dir()))
                    .count();
                let weights_dir = entry.path().join("weights");
                let mut weights: Vec<String> = fs::read_dir(&weights_dir)
                    .into_iter()
                    .flatten()
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        e.path().extension().map_or(false, |ext| ext == "ot")
                            && !e
                                .file_name()
                                .to_string_lossy()
                                .starts_with("pretrain_heads")
                    })
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .collect();
                weights.sort_by(|a, b| {
                    let num = |s: &String| -> usize {
                        s.chars()
                            .filter(|c| c.is_ascii_digit())
                            .collect::<String>()
                            .parse()
                            .unwrap_or(0)
                    };
                    num(b).cmp(&num(a))
                });
                let is_active = live_name.as_deref() == Some(&name);
                let has_step_data = entry.path().join("pretrain_train_steps.csv").exists();
                if gen_count == 0 && weights.is_empty() && !has_step_data && !is_active {
                    return None;
                }
                Some(RunInfo {
                    name,
                    gen_count,
                    weights,
                    is_active,
                })
            })
            .collect();

        let selected = match purpose {
            RunSelectorPurpose::View => runs
                .iter()
                .position(|run| Some(run.name.as_str()) == viewed_name.as_deref())
                .unwrap_or(0),
            RunSelectorPurpose::Train => runs.iter().position(|r| r.is_active).unwrap_or(0),
        };
        self.dialog_mode = DialogMode::RunSelector {
            selected,
            runs,
            purpose,
        };
    }

    fn switch_to_run(&mut self, name: &str) -> Result<()> {
        let root = std::path::Path::new(RUNS_PATH).join(name);
        let gens = root.join("gens");
        let weights = root.join("weights");
        let log_file = root.join("training.log");
        let run = RunDir {
            root,
            gens,
            weights,
            log_file,
        };
        let is_live = self
            .process_manager
            .live_run()
            .is_some_and(|live| live.root == run.root);
        if is_live {
            self.process_manager.follow_live_run();
        } else {
            self.process_manager.pin_view_run(run);
        }
        self.sync_gens_path();
        self.generation_browser.load_generations()?;
        self.load_latest_meta_charts()?;
        Ok(())
    }

    fn toggle_training_kind(&mut self) {
        let next_idx = TRAINING_KINDS
            .iter()
            .position(|kind| *kind == self.training_kind)
            .map(|idx| (idx + 1) % TRAINING_KINDS.len())
            .unwrap_or(0);
        self.training_kind = TRAINING_KINDS[next_idx];
    }

    fn toggle_genetic_family(&mut self) {
        let next_idx = GENETIC_FAMILIES
            .iter()
            .position(|family| *family == self.genetic_family)
            .map(|idx| (idx + 1) % GENETIC_FAMILIES.len())
            .unwrap_or(0);
        self.genetic_family = GENETIC_FAMILIES[next_idx];
    }

    fn start_inference(
        &mut self,
        weights_file: Option<String>,
        ticker: Option<String>,
        episodes: Option<usize>,
    ) -> Result<()> {
        let weights = weights_file.unwrap_or_else(|| "infer.ot".to_string());
        let weights_path = if weights.starts_with('/') || weights.starts_with("..") {
            weights
        } else {
            format!("{}/{}", WEIGHTS_PATH, weights)
        };
        self.process_manager.start_inference(
            weights_path,
            ticker,
            episodes.unwrap_or(10),
            self.training_model_size.clone(),
        )
    }

    fn stop_training(&mut self) -> Result<()> {
        self.process_manager.stop_training()
    }

    fn select_generation(&mut self) -> Result<()> {
        if let Some(gen) = self.generation_browser.get_selected() {
            let path = gen.path.clone();
            self.generation_browser.selected_generation =
                self.generation_browser.list_state.selected();
            self.chart_viewer.load_generation(&path)?;
            self.previous_mode = self.mode;
            self.mode = AppMode::ChartViewer;
        }
        Ok(())
    }

    fn select_inference(&mut self) -> Result<()> {
        if let Some(inf) = self.inference_browser.get_selected() {
            let path = inf.path.clone();
            self.inference_browser.selected_inference =
                self.inference_browser.list_state.selected();
            self.chart_viewer.load_inference(&path)?;
            self.previous_mode = self.mode;
            self.mode = AppMode::ChartViewer;
        }
        Ok(())
    }

    fn view_meta_charts(&mut self) -> Result<()> {
        let extra = self.pretrain_meta_reports();
        if !self.latest_meta_charts.is_empty() || !extra.is_empty() {
            self.meta_reports_revision = self.current_meta_reports_revision();
            self.chart_viewer
                .load_charts(&self.latest_meta_charts, extra)?;
            self.previous_mode = self.mode;
            self.mode = AppMode::ChartViewer;
        }
        Ok(())
    }

    fn handle_generation_click(&mut self, row: u16) -> Result<()> {
        let adjusted_row = row.saturating_sub(2);
        let current_offset = self.generation_browser.list_state.offset();
        let actual_index = current_offset + adjusted_row as usize;

        if actual_index < self.generation_browser.filtered_generations.len() {
            self.generation_browser.center_list(actual_index);
        }
        Ok(())
    }

    fn next_log_line(&mut self) {
        self.logs_page.next();
    }

    fn previous_log_line(&mut self) {
        self.logs_page.previous();
    }
}

fn main() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new()?;
    let res = run_app(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{err:?}");
    }

    Ok(())
}

#[cfg(test)]
mod planner_inference_discovery_tests {
    use super::*;

    fn complete_bundle(generation: &Path, name: &str) -> PathBuf {
        let bundle = generation.join(name);
        fs::create_dir_all(&bundle).unwrap();
        fs::write(
            bundle.join(".planner-report-generation"),
            br#"{"run_lineage_id":"run-a","update":1}"#,
        )
        .unwrap();
        fs::write(
            bundle.join(".planner-inference.json"),
            br#"{"version":1,"run_lineage_id":"run-a","update":1,"split":"test","evaluation_fingerprint":"contract-a","episodes":1,"rollout_length":100}"#,
        )
        .unwrap();
        for base in PLANNER_INFERENCE_REPORTS {
            fs::write(bundle.join(format!("{base}.report.bin")), b"report").unwrap();
        }
        bundle
    }

    #[test]
    fn discovers_only_newest_complete_published_inference_bundle() {
        let root = std::env::temp_dir().join(format!(
            "tui-planner-inference-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join(".planner-report-generation"),
            br#"{"run_lineage_id":"run-a","update":1}"#,
        )
        .unwrap();
        let old = complete_bundle(&root, "planner_inference_test_a");
        let new = complete_bundle(&root, "planner_inference_test_b");
        complete_bundle(&root, ".planner-inference-test-temp.tmp");
        let incomplete = root.join("planner_inference_test_z");
        fs::create_dir(&incomplete).unwrap();
        fs::write(
            incomplete.join("planner_inference_wealth.report.bin"),
            b"partial",
        )
        .unwrap();

        assert_eq!(
            latest_complete_planner_inference_bundle(&root).unwrap(),
            new
        );
        assert_ne!(
            latest_complete_planner_inference_bundle(&root).unwrap(),
            old
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn hides_owned_generation_until_resume_manifest_commits_it() {
        let root = std::env::temp_dir().join(format!(
            "tui-planner-visibility-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let generation = root.join("2");
        fs::create_dir_all(&generation).unwrap();
        fs::write(
            generation.join(".planner-report-generation"),
            br#"{"run_lineage_id":"run-a","update":2}"#,
        )
        .unwrap();

        let committed_one = std::collections::HashMap::from([("run-a".to_owned(), 1)]);
        let committed_two = std::collections::HashMap::from([("run-a".to_owned(), 2)]);
        let unrelated = std::collections::HashMap::from([("run-b".to_owned(), 10)]);
        assert!(!planner_generation_visible(&generation, 2, &committed_one));
        assert!(planner_generation_visible(&generation, 2, &committed_two));
        assert!(!planner_generation_visible(&generation, 2, &unrelated));
        fs::remove_file(generation.join(".planner-report-generation")).unwrap();
        assert!(planner_generation_visible(&generation, 2, &committed_one));
        fs::remove_dir_all(root).unwrap();
    }
}

fn run_app<B: ratatui::backend::Backend>(terminal: &mut Terminal<B>, app: &mut App) -> Result<()> {
    loop {
        terminal.draw(|f| ui(f, app))?;

        app.maybe_refresh()?;
        let log_path = app
            .process_manager
            .active_run
            .as_ref()
            .map(|r| r.log_file.to_string_lossy().to_string());
        app.logs_page.poll_training_output(log_path.as_deref());

        // Wait for event, then drain all pending events before redrawing
        if event::poll(Duration::from_millis(16))? {
            // ~60fps
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    // Handle dialogs first (they take priority)
                    match app.dialog_mode.clone() {
                        DialogMode::WeightsSelector {
                            run_name,
                            selected,
                            weights,
                        } => {
                            let count = weights.len();
                            match key.code {
                                KeyCode::Esc => {
                                    app.dialog_mode = DialogMode::None;
                                }
                                KeyCode::Char('j') | KeyCode::Down => {
                                    if count > 0 {
                                        let (run_name, weights) =
                                            (run_name.clone(), weights.clone());
                                        app.dialog_mode = DialogMode::WeightsSelector {
                                            run_name,
                                            selected: (selected + 1) % count,
                                            weights,
                                        };
                                    }
                                }
                                KeyCode::Char('k') | KeyCode::Up => {
                                    if count > 0 {
                                        let (run_name, weights) =
                                            (run_name.clone(), weights.clone());
                                        app.dialog_mode = DialogMode::WeightsSelector {
                                            run_name,
                                            selected: if selected == 0 {
                                                count - 1
                                            } else {
                                                selected - 1
                                            },
                                            weights,
                                        };
                                    }
                                }
                                KeyCode::Enter => {
                                    if let Some(filename) = weights.get(selected) {
                                        let path = std::path::Path::new(RUNS_PATH)
                                            .join(&run_name)
                                            .join("weights")
                                            .join(filename);
                                        let weights_str = path.to_string_lossy().to_string();
                                        app.dialog_mode = DialogMode::None;
                                        app.start_training(Some(weights_str))?;
                                    }
                                }
                                _ => {}
                            }
                        }
                        DialogMode::InferenceInput { focused_field } => match key.code {
                            KeyCode::Esc => {
                                app.dialog_mode = DialogMode::None;
                                app.input.clear();
                                app.ticker_input.clear();
                                app.episodes_input.clear();
                            }
                            KeyCode::Tab => {
                                app.dialog_mode = DialogMode::InferenceInput {
                                    focused_field: match focused_field {
                                        InferenceField::Weights => InferenceField::Ticker,
                                        InferenceField::Ticker => InferenceField::Episodes,
                                        InferenceField::Episodes => InferenceField::Weights,
                                    },
                                };
                            }
                            KeyCode::BackTab => {
                                app.dialog_mode = DialogMode::InferenceInput {
                                    focused_field: match focused_field {
                                        InferenceField::Weights => InferenceField::Episodes,
                                        InferenceField::Ticker => InferenceField::Weights,
                                        InferenceField::Episodes => InferenceField::Ticker,
                                    },
                                };
                            }
                            KeyCode::Enter => {
                                let weights = if app.input.is_empty() {
                                    None
                                } else {
                                    Some(App::coerce_weights_filename(&app.input))
                                };
                                let ticker = if app.ticker_input.is_empty() {
                                    None
                                } else {
                                    Some(app.ticker_input.clone())
                                };
                                let episodes = if app.episodes_input.is_empty() {
                                    None
                                } else {
                                    app.episodes_input.parse::<usize>().ok()
                                };

                                app.input.clear();
                                app.ticker_input.clear();
                                app.episodes_input.clear();
                                app.dialog_mode = DialogMode::None;

                                app.start_inference(weights, ticker, episodes)?;
                            }
                            KeyCode::Char(c) => match focused_field {
                                InferenceField::Weights => app.input.push(c),
                                InferenceField::Ticker => app.ticker_input.push(c),
                                InferenceField::Episodes => {
                                    if c.is_numeric() {
                                        app.episodes_input.push(c);
                                    }
                                }
                            },
                            KeyCode::Backspace => match focused_field {
                                InferenceField::Weights => {
                                    app.input.pop();
                                }
                                InferenceField::Ticker => {
                                    app.ticker_input.pop();
                                }
                                InferenceField::Episodes => {
                                    app.episodes_input.pop();
                                }
                            },
                            _ => {}
                        },
                        DialogMode::ConfirmQuit => match key.code {
                            KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                                return Ok(());
                            }
                            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                                app.dialog_mode = DialogMode::None;
                            }
                            _ => {}
                        },
                        DialogMode::ConfirmStopTraining => match key.code {
                            KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                                app.stop_training()?;
                                app.dialog_mode = DialogMode::None;
                            }
                            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                                app.dialog_mode = DialogMode::None;
                            }
                            _ => {}
                        },
                        DialogMode::PageJump { selected } => {
                            const PAGE_COUNT: usize = 5;
                            match key.code {
                                KeyCode::Esc => {
                                    app.dialog_mode = DialogMode::None;
                                }
                                KeyCode::Char('j') | KeyCode::Down => {
                                    app.dialog_mode = DialogMode::PageJump {
                                        selected: (selected + 1) % PAGE_COUNT,
                                    };
                                }
                                KeyCode::Char('k') | KeyCode::Up => {
                                    app.dialog_mode = DialogMode::PageJump {
                                        selected: if selected == 0 {
                                            PAGE_COUNT - 1
                                        } else {
                                            selected - 1
                                        },
                                    };
                                }
                                KeyCode::Enter => {
                                    app.dialog_mode = DialogMode::None;
                                    match selected {
                                        0 => app.mode = AppMode::Main,
                                        1 => {
                                            app.generation_browser.load_generations()?;
                                            if !app
                                                .generation_browser
                                                .filtered_generations
                                                .is_empty()
                                                && app
                                                    .generation_browser
                                                    .list_state
                                                    .selected()
                                                    .is_none()
                                            {
                                                app.generation_browser.list_state.select(Some(0));
                                                app.generation_browser.center_list(0);
                                            }
                                            app.mode = AppMode::GenerationBrowser;
                                        }
                                        2 => {
                                            app.inference_browser.load_inferences()?;
                                            if !app.inference_browser.filtered_inferences.is_empty()
                                                && app
                                                    .inference_browser
                                                    .list_state
                                                    .selected()
                                                    .is_none()
                                            {
                                                app.inference_browser.list_state.select(Some(0));
                                                app.inference_browser.center_list(0);
                                            }
                                            app.mode = AppMode::InferenceBrowser;
                                        }
                                        3 => {
                                            app.view_meta_charts()?;
                                        }
                                        4 => {
                                            app.logs_page.enter();
                                            app.mode = AppMode::Logs;
                                        }
                                        _ => {}
                                    }
                                }
                                KeyCode::Char('1') => {
                                    app.dialog_mode = DialogMode::None;
                                    app.mode = AppMode::Main;
                                }
                                KeyCode::Char('2') => {
                                    app.dialog_mode = DialogMode::None;
                                    app.generation_browser.load_generations()?;
                                    if !app.generation_browser.filtered_generations.is_empty()
                                        && app.generation_browser.list_state.selected().is_none()
                                    {
                                        app.generation_browser.list_state.select(Some(0));
                                        app.generation_browser.center_list(0);
                                    }
                                    app.mode = AppMode::GenerationBrowser;
                                }
                                KeyCode::Char('3') => {
                                    app.dialog_mode = DialogMode::None;
                                    app.inference_browser.load_inferences()?;
                                    if !app.inference_browser.filtered_inferences.is_empty()
                                        && app.inference_browser.list_state.selected().is_none()
                                    {
                                        app.inference_browser.list_state.select(Some(0));
                                        app.inference_browser.center_list(0);
                                    }
                                    app.mode = AppMode::InferenceBrowser;
                                }
                                KeyCode::Char('4') => {
                                    app.dialog_mode = DialogMode::None;
                                    app.view_meta_charts()?;
                                }
                                KeyCode::Char('5') => {
                                    app.dialog_mode = DialogMode::None;
                                    app.logs_page.enter();
                                    app.mode = AppMode::Logs;
                                }
                                _ => {}
                            }
                        }
                        DialogMode::RunSelector {
                            selected,
                            runs,
                            purpose,
                        } => {
                            let count = runs.len();
                            match key.code {
                                KeyCode::Esc => {
                                    app.dialog_mode = DialogMode::None;
                                }
                                KeyCode::Char('j') | KeyCode::Down => {
                                    if count > 0 {
                                        let (runs, purpose) = (runs.clone(), purpose.clone());
                                        app.dialog_mode = DialogMode::RunSelector {
                                            selected: (selected + 1) % count,
                                            runs,
                                            purpose,
                                        };
                                    }
                                }
                                KeyCode::Char('k') | KeyCode::Up => {
                                    if count > 0 {
                                        let (runs, purpose) = (runs.clone(), purpose.clone());
                                        app.dialog_mode = DialogMode::RunSelector {
                                            selected: if selected == 0 {
                                                count - 1
                                            } else {
                                                selected - 1
                                            },
                                            runs,
                                            purpose,
                                        };
                                    }
                                }
                                KeyCode::Enter => {
                                    if let Some(run) = runs.get(selected) {
                                        let name = run.name.clone();
                                        match purpose {
                                            RunSelectorPurpose::View => {
                                                app.dialog_mode = DialogMode::None;
                                                app.switch_to_run(&name)?;
                                            }
                                            RunSelectorPurpose::Train => {
                                                if run.weights.is_empty() {
                                                    // No weights, start from scratch
                                                    app.dialog_mode = DialogMode::None;
                                                    app.start_training(None)?;
                                                } else {
                                                    let weights = run.weights.clone();
                                                    app.dialog_mode = DialogMode::WeightsSelector {
                                                        run_name: name,
                                                        selected: 0,
                                                        weights,
                                                    };
                                                }
                                            }
                                        }
                                    }
                                }
                                KeyCode::Char('n') => {
                                    if matches!(purpose, RunSelectorPurpose::Train) {
                                        app.dialog_mode = DialogMode::None;
                                        app.start_training(None)?;
                                    }
                                }
                                _ => {}
                            }
                        }
                        DialogMode::None => {
                            // Global keybindings (work in all modes when no dialog is open)
                            match key.code {
                                KeyCode::Char('c')
                                    if key
                                        .modifiers
                                        .contains(crossterm::event::KeyModifiers::CONTROL) =>
                                {
                                    if app.is_training_running() {
                                        app.dialog_mode = DialogMode::ConfirmQuit;
                                    } else {
                                        return Ok(());
                                    }
                                }
                                KeyCode::Char('d')
                                    if key
                                        .modifiers
                                        .contains(crossterm::event::KeyModifiers::CONTROL) =>
                                {
                                    if app.is_training_running() {
                                        app.dialog_mode = DialogMode::ConfirmQuit;
                                    } else {
                                        return Ok(());
                                    }
                                }
                                KeyCode::Char('o') => {
                                    // Don't open page jump when searching
                                    if !app.generation_browser.searching
                                        && !app.inference_browser.searching
                                    {
                                        app.dialog_mode = DialogMode::PageJump { selected: 0 };
                                    }
                                }
                                KeyCode::Char('R') => {
                                    if !app.generation_browser.searching
                                        && !app.inference_browser.searching
                                    {
                                        app.open_run_selector(RunSelectorPurpose::View);
                                    }
                                }
                                _ => {}
                            }

                            match app.mode {
                                AppMode::Main => match key.code {
                                    KeyCode::Char('q') => {
                                        if app.is_training_running() {
                                            app.dialog_mode = DialogMode::ConfirmQuit;
                                        } else {
                                            return Ok(());
                                        }
                                    }
                                    KeyCode::Char('s') => {
                                        if !app.is_training_running() {
                                            if matches!(
                                                app.training_kind,
                                                TrainingKind::Rl | TrainingKind::Pretrain
                                            ) {
                                                app.open_run_selector(RunSelectorPurpose::Train);
                                            } else {
                                                app.start_training(None)?;
                                            }
                                        }
                                    }
                                    KeyCode::Char('t') => {
                                        if !app.is_training_running() {
                                            app.toggle_training_kind();
                                        }
                                    }
                                    KeyCode::Char('g') => {
                                        if !app.is_training_running()
                                            && app.training_kind == TrainingKind::Genetic
                                        {
                                            app.toggle_genetic_family();
                                        }
                                    }
                                    KeyCode::Char('f') => {
                                        if !app.is_anything_running() {
                                            app.dialog_mode = DialogMode::InferenceInput {
                                                focused_field: InferenceField::Weights,
                                            };
                                        }
                                    }
                                    KeyCode::Char('x') => {
                                        if app.is_training_running() {
                                            app.dialog_mode = DialogMode::ConfirmStopTraining;
                                        }
                                    }
                                    KeyCode::Char('e') => {
                                        app.generation_browser.load_generations()?;
                                        // Auto-select latest generation (first in list) only if no selection exists
                                        if !app.generation_browser.filtered_generations.is_empty()
                                            && app
                                                .generation_browser
                                                .list_state
                                                .selected()
                                                .is_none()
                                        {
                                            app.generation_browser.list_state.select(Some(0));
                                            app.generation_browser.center_list(0);
                                        }
                                        app.mode = AppMode::GenerationBrowser;
                                    }
                                    KeyCode::Char('i') => {
                                        app.inference_browser.load_inferences()?;
                                        // Auto-select latest inference (first in list) only if no selection exists
                                        if !app.inference_browser.filtered_inferences.is_empty()
                                            && app.inference_browser.list_state.selected().is_none()
                                        {
                                            app.inference_browser.list_state.select(Some(0));
                                            app.inference_browser.center_list(0);
                                        }
                                        app.mode = AppMode::InferenceBrowser;
                                    }
                                    KeyCode::Char('m') => {
                                        app.view_meta_charts()?;
                                    }
                                    KeyCode::Char('l') => {
                                        app.logs_page.enter();
                                        app.mode = AppMode::Logs;
                                    }
                                    KeyCode::Char('v') => {
                                        app.mode = AppMode::ModelObservations;
                                    }
                                    _ => {}
                                },
                                AppMode::GenerationBrowser => {
                                    if app.generation_browser.searching {
                                        match key.code {
                                            KeyCode::Esc => {
                                                app.generation_browser.searching = false;
                                                app.generation_browser.search_input.clear();
                                                app.generation_browser.filter_generations();
                                            }
                                            KeyCode::Enter => {
                                                app.generation_browser.searching = false;
                                            }
                                            KeyCode::Char(c) => {
                                                app.generation_browser.search_input.push(c);
                                                app.generation_browser.filter_generations();
                                            }
                                            KeyCode::Backspace => {
                                                app.generation_browser.search_input.pop();
                                                app.generation_browser.filter_generations();
                                            }
                                            _ => {}
                                        }
                                    } else {
                                        match key.code {
                                            KeyCode::Esc | KeyCode::Char('q') => {
                                                app.mode = AppMode::Main;
                                            }
                                            KeyCode::Char('/') => {
                                                app.generation_browser.searching = true;
                                            }
                                            KeyCode::Down
                                                if key.modifiers.contains(
                                                    crossterm::event::KeyModifiers::CONTROL,
                                                ) =>
                                            {
                                                app.generation_browser.scroll_down(5);
                                            }
                                            KeyCode::Up
                                                if key.modifiers.contains(
                                                    crossterm::event::KeyModifiers::CONTROL,
                                                ) =>
                                            {
                                                app.generation_browser.scroll_up(5);
                                            }
                                            KeyCode::Down | KeyCode::Char('j') => {
                                                app.generation_browser.next();
                                            }
                                            KeyCode::Up | KeyCode::Char('k') => {
                                                app.generation_browser.previous();
                                            }
                                            KeyCode::Enter => {
                                                app.select_generation()?;
                                            }
                                            KeyCode::Char('r') => {
                                                app.generation_browser.load_generations()?;
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                AppMode::InferenceBrowser => {
                                    if app.inference_browser.searching {
                                        match key.code {
                                            KeyCode::Esc => {
                                                app.inference_browser.searching = false;
                                                app.inference_browser.search_input.clear();
                                                app.inference_browser.filter_inferences();
                                            }
                                            KeyCode::Enter => {
                                                app.inference_browser.searching = false;
                                            }
                                            KeyCode::Char(c) => {
                                                app.inference_browser.search_input.push(c);
                                                app.inference_browser.filter_inferences();
                                            }
                                            KeyCode::Backspace => {
                                                app.inference_browser.search_input.pop();
                                                app.inference_browser.filter_inferences();
                                            }
                                            _ => {}
                                        }
                                    } else {
                                        match key.code {
                                            KeyCode::Esc | KeyCode::Char('q') => {
                                                app.mode = AppMode::Main;
                                            }
                                            KeyCode::Enter => {
                                                app.select_inference()?;
                                            }
                                            KeyCode::Char('/') => {
                                                app.inference_browser.searching = true;
                                            }
                                            KeyCode::Down
                                                if key.modifiers.contains(
                                                    crossterm::event::KeyModifiers::CONTROL,
                                                ) =>
                                            {
                                                app.inference_browser.scroll_down(5);
                                            }
                                            KeyCode::Up
                                                if key.modifiers.contains(
                                                    crossterm::event::KeyModifiers::CONTROL,
                                                ) =>
                                            {
                                                app.inference_browser.scroll_up(5);
                                            }
                                            KeyCode::Down | KeyCode::Char('j') => {
                                                app.inference_browser.next();
                                            }
                                            KeyCode::Up | KeyCode::Char('k') => {
                                                app.inference_browser.previous();
                                            }
                                            KeyCode::Char('r') => {
                                                app.inference_browser.load_inferences()?;
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                AppMode::ChartViewer => {
                                    if app.chart_viewer.is_editing_row_skip() {
                                        match key.code {
                                            KeyCode::Esc => {
                                                app.chart_viewer.cancel_editing_row_skip();
                                            }
                                            KeyCode::Enter => {
                                                app.chart_viewer.stop_editing_row_skip();
                                            }
                                            KeyCode::Char(c) => {
                                                app.chart_viewer.row_skip_input_push(c);
                                            }
                                            KeyCode::Backspace => {
                                                app.chart_viewer.row_skip_input_pop();
                                            }
                                            _ => {}
                                        }
                                    } else {
                                        match key.code {
                                            KeyCode::Esc | KeyCode::Char('q') => {
                                                app.mode = app.previous_mode;
                                            }
                                            KeyCode::Char('/') => {
                                                if app.chart_viewer.is_viewing_meta_charts() {
                                                    app.chart_viewer.start_editing_row_skip();
                                                }
                                            }
                                            KeyCode::Down | KeyCode::Char('j') => {
                                                app.chart_viewer.next();
                                            }
                                            KeyCode::Up | KeyCode::Char('k') => {
                                                app.chart_viewer.previous();
                                            }
                                            KeyCode::Enter => {
                                                app.chart_viewer.toggle_expand();
                                            }
                                            KeyCode::Char('r') => {
                                                if app.chart_viewer.is_viewing_meta_charts() {
                                                    app.load_latest_meta_charts()?;
                                                    let extra = app.pretrain_meta_reports();
                                                    app.meta_reports_revision =
                                                        app.current_meta_reports_revision();
                                                    app.chart_viewer.load_charts(
                                                        &app.latest_meta_charts,
                                                        extra,
                                                    )?;
                                                }
                                            }
                                            KeyCode::Char('c') => {
                                                let _ = app.chart_viewer.copy_current_image();
                                            }
                                            KeyCode::Char('l') => {
                                                app.chart_viewer.toggle_legend();
                                            }
                                            KeyCode::Char(c @ '1'..='9') => {
                                                app.chart_viewer
                                                    .toggle_solo_series((c as u8 - b'1') as usize);
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                AppMode::Logs => match key.code {
                                    KeyCode::Esc | KeyCode::Char('q') => {
                                        app.mode = AppMode::Main;
                                    }
                                    KeyCode::Char('c') => {
                                        app.logs_page.clear_logs();
                                    }
                                    KeyCode::Down | KeyCode::Char('j') => {
                                        app.logs_page.next();
                                    }
                                    KeyCode::Up | KeyCode::Char('k') => {
                                        app.logs_page.previous();
                                    }
                                    KeyCode::PageDown => {
                                        app.logs_page.page_down();
                                    }
                                    KeyCode::PageUp => {
                                        app.logs_page.page_up();
                                    }
                                    KeyCode::Home => {
                                        app.logs_page.jump_to_top();
                                    }
                                    KeyCode::End => {
                                        app.logs_page.jump_to_bottom();
                                    }
                                    _ => {}
                                },
                                AppMode::ModelObservations => match key.code {
                                    KeyCode::Esc | KeyCode::Char('q') => {
                                        app.mode = AppMode::Main;
                                    }
                                    KeyCode::Char('r') => {}
                                    _ => {}
                                },
                            }
                        }
                    }
                }
                Event::Mouse(mouse) => match mouse.kind {
                    MouseEventKind::ScrollUp => match app.mode {
                        AppMode::GenerationBrowser => app.generation_browser.scroll_up(3),
                        AppMode::InferenceBrowser => app.inference_browser.scroll_up(3),
                        AppMode::ChartViewer => app.chart_viewer.scroll_up(3),
                        AppMode::Logs => {
                            for _ in 0..3 {
                                app.logs_page.previous();
                            }
                        }
                        _ => {}
                    },
                    MouseEventKind::ScrollDown => match app.mode {
                        AppMode::GenerationBrowser => app.generation_browser.scroll_down(3),
                        AppMode::InferenceBrowser => app.inference_browser.scroll_down(3),
                        AppMode::ChartViewer => app.chart_viewer.scroll_down(3),
                        AppMode::Logs => {
                            for _ in 0..3 {
                                app.logs_page.next();
                            }
                        }
                        _ => {}
                    },
                    MouseEventKind::Down(MouseButton::Left) => {
                        if app.mode == AppMode::GenerationBrowser {
                            let list_area = app.generation_browser.list_area;
                            if mouse.column >= list_area.x
                                && mouse.column < list_area.x + list_area.width
                                && mouse.row >= list_area.y
                                && mouse.row < list_area.y + list_area.height
                            {
                                let _ = app.handle_generation_click(mouse.row - list_area.y);
                            }
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }
}

fn ui(f: &mut Frame, app: &mut App) {
    match app.mode {
        AppMode::Main => pages::main_page::render(f, app),
        AppMode::GenerationBrowser => pages::generation_browser::render(f, app),
        AppMode::InferenceBrowser => pages::inference_browser::render(f, app),
        AppMode::ChartViewer => {
            let is_training = app.is_training_running();
            let current_episode = app.get_current_episode();
            let has_progress = app.has_training_progress();
            app.chart_viewer
                .render(f, is_training, current_episode, has_progress);
        }
        AppMode::Logs => pages::logs_page::render(f, app),
        AppMode::ModelObservations => pages::model_observations_page::render(f, app),
    }

    // Render dialog on top if active
    match &app.dialog_mode {
        DialogMode::InferenceInput { focused_field } => {
            components::dialogs::inference::render(f, app, *focused_field);
        }
        DialogMode::ConfirmQuit => {
            components::dialogs::confirm::render(
                f,
                "Quit?",
                "Training processes will continue running in background.",
            );
        }
        DialogMode::ConfirmStopTraining => {
            components::dialogs::confirm::render(
                f,
                "Stop Training?",
                "This will terminate the training process.",
            );
        }
        DialogMode::PageJump { selected } => {
            components::dialogs::page_jump::render(f, *selected, app.mode);
        }
        DialogMode::RunSelector {
            selected,
            runs,
            purpose,
        } => {
            components::dialogs::run_selector::render(f, *selected, runs, purpose);
        }
        DialogMode::WeightsSelector {
            run_name,
            selected,
            weights,
        } => {
            components::dialogs::run_selector::render_weights(f, run_name, *selected, weights);
        }
        DialogMode::None => {}
    }
}
