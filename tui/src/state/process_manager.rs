use anyhow::Result;
use shared::{paths::RUNS_PATH, run_dir::RunDir};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingKind {
    Rl,
    Genetic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneticFamily {
    PriceRebound,
    RsiRebound,
    TrendBreakout,
}

impl GeneticFamily {
    pub fn as_cli_str(self) -> &'static str {
        match self {
            Self::PriceRebound => "price-rebound",
            Self::RsiRebound => "rsi-rebound",
            Self::TrendBreakout => "trend-breakout",
        }
    }
}

pub struct ProcessManagerState {
    pub inference_process: Option<Child>,
    pub training_process: Option<Child>,
    pub active_run: Option<RunDir>,
    cached_training_running: bool,
    last_training_check: Instant,
}

impl ProcessManagerState {
    pub fn new() -> Self {
        let active_run = detect_active_training_run(None)
            .or_else(|| RunDir::latest_with_data(RUNS_PATH))
            .or_else(|| RunDir::latest(RUNS_PATH).ok());
        Self {
            inference_process: None,
            training_process: None,
            active_run,
            cached_training_running: false,
            last_training_check: Instant::now(),
        }
    }

    pub fn is_training_running(&mut self) -> bool {
        let now = Instant::now();
        if now.duration_since(self.last_training_check) < Duration::from_millis(500) {
            return self.cached_training_running;
        }

        self.last_training_check = now;

        if let Some(ref mut child) = self.training_process {
            match child.try_wait() {
                Ok(Some(_)) => {
                    self.training_process = None;
                }
                Ok(None) => {
                    self.cached_training_running = true;
                    self.active_run = detect_active_training_run(Some(child.id()))
                        .or_else(|| self.active_run.take())
                        .or_else(|| RunDir::latest(RUNS_PATH).ok());
                    return true;
                }
                Err(_) => {
                    self.training_process = None;
                }
            }
        }

        if !list_training_pids().is_empty() {
            self.cached_training_running = true;
            self.active_run =
                detect_active_training_run(self.training_process.as_ref().map(|c| c.id()))
                    .or_else(|| RunDir::latest_with_data(RUNS_PATH))
                    .or_else(|| RunDir::latest(RUNS_PATH).ok());
            return true;
        }

        self.cached_training_running = false;
        false
    }

    pub fn is_anything_running(&mut self) -> bool {
        self.is_training_running() || self.inference_process.is_some()
    }

    pub fn start_training(
        &mut self,
        kind: TrainingKind,
        weights: Option<String>,
        model_size: &str,
        genetic_family: GeneticFamily,
    ) -> Result<()> {
        if self.is_anything_running() {
            return Ok(());
        }

        let run_dir = match &weights {
            Some(w) => {
                let p = Path::new(w);
                // If weights are inside runs/*/weights/, reuse that run dir
                if p.parent()
                    .and_then(|d| d.file_name())
                    .map_or(false, |n| n == "weights")
                    && p.ancestors().any(|a| a.ends_with("runs"))
                {
                    RunDir::from_weights_path(p)?
                } else {
                    RunDir::create_fresh(RUNS_PATH, None)?
                }
            }
            None => RunDir::create_fresh(RUNS_PATH, None)?,
        };

        let log_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&run_dir.log_file)?;

        let mut cmd = Command::new("cargo");
        cmd.current_dir("../trading_bots")
            .arg("run")
            .arg("--release")
            .arg("--");

        match kind {
            TrainingKind::Rl => {
                cmd.arg("train").arg("--model-size").arg(model_size);

                if let Some(w) = weights {
                    cmd.arg("--weights").arg(w);
                }
            }
            TrainingKind::Genetic => {
                cmd.arg("genetic")
                    .arg("--family")
                    .arg(genetic_family.as_cli_str());
            }
        }

        if let Some(name) = run_dir.root.file_name().and_then(|n| n.to_str()) {
            cmd.arg("--run").arg(name);
        }

        cmd.env("CLICOLOR_FORCE", "1")
            .stdin(Stdio::null())
            .stdout(log_file.try_clone()?)
            .stderr(log_file);

        let child = cmd.spawn()?;
        self.training_process = Some(child);
        self.active_run = Some(run_dir);
        self.cached_training_running = true;

        Ok(())
    }

    pub fn start_inference(
        &mut self,
        weights: String,
        ticker: Option<String>,
        episodes: usize,
        model_size: String,
    ) -> Result<()> {
        if self.is_anything_running() {
            return Ok(());
        }

        let mut cmd = Command::new("cargo");
        cmd.current_dir("../trading_bots")
            .arg("run")
            .arg("--release")
            .arg("--")
            .arg("infer")
            .arg("--weights")
            .arg(weights)
            .arg("--model-size")
            .arg(model_size)
            .arg("--episodes")
            .arg(episodes.to_string());

        if let Some(t) = ticker {
            cmd.arg("--ticker").arg(t);
        }

        let child = cmd.spawn()?;
        self.inference_process = Some(child);

        Ok(())
    }

    pub fn stop_training(&mut self) -> Result<()> {
        if let Some(mut child) = self.training_process.take() {
            terminate_process_tree(child.id());
            let _ = child.try_wait();
        }

        for pid in list_training_pids() {
            terminate_process_tree(pid);
        }

        self.cached_training_running = false;
        Ok(())
    }

    pub fn check_inference_process(&mut self) {
        if let Some(ref mut child) = self.inference_process {
            if let Ok(Some(_)) = child.try_wait() {
                self.inference_process = None;
            }
        }
    }
}

fn list_training_pids() -> Vec<u32> {
    list_training_processes()
        .into_iter()
        .map(|(pid, _)| pid)
        .collect()
}

fn list_training_processes() -> Vec<(u32, String)> {
    let Ok(output) = Command::new("ps").args(["-eo", "pid=,args="]).output() else {
        return Vec::new();
    };

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            let split_idx = trimmed.find(char::is_whitespace)?;
            let (pid, cmdline) = trimmed.split_at(split_idx);
            let pid = pid.parse::<u32>().ok()?;
            let cmdline = cmdline.trim();
            is_training_cmdline(cmdline).then_some((pid, cmdline.to_string()))
        })
        .collect()
}

fn detect_active_training_run(preferred_pid: Option<u32>) -> Option<RunDir> {
    let mut processes = list_training_processes();
    processes.sort_by_key(|(pid, _)| *pid);

    if let Some(pid) = preferred_pid {
        if let Some((_, cmdline)) = processes.iter().find(|(candidate, _)| *candidate == pid) {
            if let Some(run_dir) = parse_run_dir_from_cmdline(cmdline) {
                return Some(run_dir);
            }
        }
    }

    for (_, cmdline) in processes.into_iter().rev() {
        if let Some(run_dir) = parse_run_dir_from_cmdline(&cmdline) {
            return Some(run_dir);
        }
    }

    None
}

fn parse_run_dir_from_cmdline(cmdline: &str) -> Option<RunDir> {
    let run_name = cmdline
        .split_whitespace()
        .collect::<Vec<_>>()
        .windows(2)
        .find_map(|window| (window[0] == "--run").then_some(window[1]))
        .or_else(|| {
            cmdline
                .split_whitespace()
                .find_map(|part| part.strip_prefix("--run="))
        })?;

    run_dir_from_name(run_name)
}

fn run_dir_from_name(name: &str) -> Option<RunDir> {
    let root = Path::new(RUNS_PATH).join(name);
    let gens = root.join("gens");
    let weights = root.join("weights");
    let log_file = root.join("training.log");

    (root.is_dir() && gens.is_dir() && weights.is_dir()).then_some(RunDir {
        root,
        gens,
        weights,
        log_file,
    })
}

fn terminate_process_tree(pid: u32) {
    let pid_str = pid.to_string();
    let _ = Command::new("pkill")
        .args(["-TERM", "-P", &pid_str])
        .output();
    let _ = Command::new("kill").args(["-TERM", &pid_str]).output();

    thread::sleep(Duration::from_millis(150));

    if process_exists(pid) {
        let _ = Command::new("pkill")
            .args(["-KILL", "-P", &pid_str])
            .output();
        let _ = Command::new("kill").args(["-KILL", &pid_str]).output();
    }
}

fn process_exists(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn is_training_cmdline(cmdline: &str) -> bool {
    !cmdline.contains("trading-bot-tui")
        && !cmdline.contains("ps -eo")
        && !cmdline.contains("pgrep -f")
        && (cmdline.contains(" train ")
            || cmdline.ends_with(" train")
            || cmdline.contains(" genetic ")
            || cmdline.ends_with(" genetic")
            || cmdline.contains("trading_bot_0 train")
            || cmdline.contains("trading_bot_0 genetic"))
}
