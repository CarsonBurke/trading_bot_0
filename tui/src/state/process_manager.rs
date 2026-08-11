use anyhow::Result;
use shared::{
    paths::{RUNS_PATH, WORKSPACE_ROOT},
    run_dir::RunDir,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingKind {
    Rl,
    Genetic,
    Pretrain,
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
    live_run: Option<RunDir>,
    view_pinned: bool,
    cached_training_running: bool,
    last_training_check: Instant,
}

impl ProcessManagerState {
    pub fn new() -> Self {
        let live_run = detect_active_training_run(None);
        let active_run = live_run
            .as_ref()
            .map(clone_run_dir)
            .or_else(latest_observable_run)
            .or_else(|| RunDir::latest_with_data(RUNS_PATH))
            .or_else(|| RunDir::latest(RUNS_PATH).ok());
        Self {
            inference_process: None,
            training_process: None,
            active_run,
            live_run,
            view_pinned: false,
            cached_training_running: false,
            last_training_check: Instant::now(),
        }
    }

    pub fn live_run(&self) -> Option<&RunDir> {
        self.live_run.as_ref()
    }

    pub fn pin_view_run(&mut self, run: RunDir) {
        self.active_run = Some(run);
        self.view_pinned = true;
    }

    pub fn follow_live_run(&mut self) {
        self.view_pinned = false;
        self.active_run = self
            .live_run
            .as_ref()
            .map(clone_run_dir)
            .or_else(latest_observable_run)
            .or_else(|| RunDir::latest_with_data(RUNS_PATH))
            .or_else(|| RunDir::latest(RUNS_PATH).ok());
    }

    fn update_live_run(&mut self, live_run: Option<RunDir>) {
        self.live_run = live_run;
        if !self.view_pinned {
            self.active_run = self.live_run.as_ref().map(clone_run_dir).or_else(|| {
                self.active_run
                    .as_ref()
                    .map(clone_run_dir)
                    .or_else(latest_observable_run)
            });
        }
    }

    pub fn is_training_running(&mut self) -> bool {
        let now = Instant::now();
        if now.duration_since(self.last_training_check) < Duration::from_millis(500) {
            return self.cached_training_running;
        }

        self.last_training_check = now;
        self.refresh_training_process()
    }

    pub fn poll_training_process(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_training_check) >= Duration::from_millis(500) {
            self.last_training_check = now;
            self.refresh_training_process();
        }
    }

    fn refresh_training_process(&mut self) -> bool {
        let mut exit_status = None;

        if let Some(ref mut child) = self.training_process {
            match child.try_wait() {
                Ok(Some(status)) => {
                    exit_status = Some((child.id(), status));
                }
                Ok(None) => {
                    self.cached_training_running = true;
                    let live_run = detect_active_training_run(Some(child.id()))
                        .or_else(|| self.live_run.as_ref().map(clone_run_dir));
                    self.update_live_run(live_run);
                    return true;
                }
                Err(_) => {
                    self.training_process = None;
                }
            }
        }

        if let Some((pid, status)) = exit_status {
            self.training_process = None;
            append_training_exit_status(
                self.live_run
                    .as_ref()
                    .or(self.active_run.as_ref())
                    .map(|run| run.log_file.as_path()),
                pid,
                status,
            );
        }

        if !list_training_pids().is_empty() {
            self.cached_training_running = true;
            self.update_live_run(detect_active_training_run(
                self.training_process.as_ref().map(|c| c.id()),
            ));
            return true;
        }

        self.update_live_run(None);
        self.cached_training_running = false;
        false
    }

    pub fn is_anything_running(&mut self) -> bool {
        self.check_inference_process();
        self.is_training_running() || self.inference_process.is_some()
    }

    pub fn start_training(
        &mut self,
        kind: TrainingKind,
        weights: Option<String>,
        genetic_family: GeneticFamily,
    ) -> Result<()> {
        if self.is_anything_running() {
            return Ok(());
        }

        let run_dir = match (kind, &weights) {
            (TrainingKind::Pretrain, _) => RunDir::create_fresh(RUNS_PATH, None)?,
            (_, Some(w)) => {
                let p = Path::new(w);
                if kind == TrainingKind::Rl && is_ppo_resume_path(p) {
                    let run = RunDir::from_weights_path_in(p, RUNS_PATH)?;
                    run.activate(RUNS_PATH)?;
                    run
                } else {
                    RunDir::create_fresh(RUNS_PATH, None)?
                }
            }
            (_, None) => RunDir::create_fresh(RUNS_PATH, None)?,
        };

        let log_file = open_training_log(&run_dir.log_file)?;

        let mut cmd = trading_bot_command();

        match kind {
            TrainingKind::Rl => {
                cmd.arg("train").arg("--model-size").arg("uniform-stream");

                if let Some(w) = weights {
                    cmd.arg("--weights").arg(w);
                }
            }
            TrainingKind::Genetic => {
                cmd.arg("genetic")
                    .arg("--family")
                    .arg(genetic_family.as_cli_str());
            }
            TrainingKind::Pretrain => {
                cmd.arg("pretrain")
                    .arg("--model-size")
                    .arg("uniform-stream")
                    .arg("--objective")
                    .arg("lejepa");

                if let Some(w) = weights {
                    cmd.arg("--weights").arg(w);
                }
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
        self.live_run = Some(clone_run_dir(&run_dir));
        self.active_run = Some(run_dir);
        self.view_pinned = false;
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

        let run_dir = RunDir::create_fresh(RUNS_PATH, None)?;
        let log_file = open_training_log(&run_dir.log_file)?;
        let run_name = run_dir
            .root
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("inference run has no UTF-8 name"))?;
        let mut cmd = trading_bot_command();
        cmd.arg("infer")
            .arg("--weights")
            .arg(weights)
            .arg("--model-size")
            .arg(model_size)
            .arg("--episodes")
            .arg(episodes.to_string())
            .arg("--run")
            .arg(run_name);

        if let Some(t) = ticker {
            cmd.arg("--tickers").arg(t);
        }

        cmd.stdin(Stdio::null())
            .stdout(log_file.try_clone()?)
            .stderr(log_file);
        let child = cmd.spawn()?;
        self.inference_process = Some(child);
        self.active_run = Some(run_dir);
        self.view_pinned = false;

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

fn clone_run_dir(run: &RunDir) -> RunDir {
    RunDir {
        root: run.root.clone(),
        gens: run.gens.clone(),
        weights: run.weights.clone(),
        log_file: run.log_file.clone(),
    }
}

fn open_training_log(path: &Path) -> Result<std::fs::File> {
    Ok(std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?)
}

fn is_ppo_resume_path(path: &Path) -> bool {
    path.with_extension("resume.json").is_file()
        || path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem| stem.strip_prefix("ppo_ep"))
            .is_some_and(|episode| episode.parse::<usize>().is_ok())
}

fn trading_bot_command() -> Command {
    let mut command = Command::new(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../trading_bots/run-release-cuda.sh"),
    );
    command.current_dir(Path::new(WORKSPACE_ROOT));
    command
}

fn list_training_processes() -> Vec<(u32, Vec<String>)> {
    let Ok(entries) = fs::read_dir("/proc") else {
        return Vec::new();
    };
    let workspace_root = canonical_or_original(Path::new(WORKSPACE_ROOT));
    let bots_root = canonical_or_original(&trading_bots_dir());

    entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let pid = entry.file_name().to_str()?.parse::<u32>().ok()?;
            let cwd = fs::read_link(entry.path().join("cwd")).ok()?;
            let executable_path = fs::read_link(entry.path().join("exe")).ok()?;
            let args = read_process_args(&entry.path().join("cmdline"))?;
            is_training_invocation(&cwd, &executable_path, &args, &workspace_root, &bots_root)
                .then_some((pid, args))
        })
        .collect()
}

fn detect_active_training_run(preferred_pid: Option<u32>) -> Option<RunDir> {
    let mut processes = list_training_processes();
    processes.sort_by_key(|(pid, _)| *pid);

    if let Some(pid) = preferred_pid {
        if let Some((_, args)) = processes.iter().find(|(candidate, _)| *candidate == pid) {
            if let Some(run_dir) = parse_run_dir_from_args(args) {
                return Some(run_dir);
            }
        }
    }

    for (_, args) in processes.into_iter().rev() {
        if let Some(run_dir) = parse_run_dir_from_args(&args) {
            return Some(run_dir);
        }
    }

    None
}

fn parse_run_dir_from_args(args: &[String]) -> Option<RunDir> {
    run_name_from_args(args).and_then(run_dir_from_name)
}

fn run_name_from_args(args: &[String]) -> Option<&str> {
    option_value(args, "--run").or_else(|| {
        let output = option_value(args, "--output")?;
        let output = Path::new(output);
        let weights = output.parent()?;
        if weights.file_name()?.to_str()? != "weights" {
            return None;
        }
        weights.parent()?.file_name()?.to_str()
    })
}

fn option_value<'a>(args: &'a [String], option: &str) -> Option<&'a str> {
    args.windows(2)
        .find_map(|window| (window[0] == option).then_some(window[1].as_str()))
        .or_else(|| {
            args.iter()
                .find_map(|part| part.strip_prefix(option)?.strip_prefix('='))
        })
}

fn read_process_args(path: &Path) -> Option<Vec<String>> {
    let bytes = fs::read(path).ok()?;
    let args = bytes
        .split(|byte| *byte == 0)
        .filter(|arg| !arg.is_empty())
        .map(|arg| String::from_utf8_lossy(arg).into_owned())
        .collect::<Vec<_>>();
    (!args.is_empty()).then_some(args)
}

fn canonical_or_original(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn is_training_invocation(
    cwd: &Path,
    executable_path: &Path,
    args: &[String],
    workspace_root: &Path,
    bots_root: &Path,
) -> bool {
    if cwd != workspace_root && cwd != bots_root {
        return false;
    }

    let Some(argv_executable) = args
        .first()
        .and_then(|arg| Path::new(arg).file_name())
        .and_then(|name| name.to_str())
    else {
        return false;
    };
    let Some(actual_executable) = executable_path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };

    match argv_executable {
        "trading_bot_0" => {
            actual_executable == "trading_bot_0"
                && args.get(1).is_some_and(|arg| is_training_subcommand(arg))
        }
        "cargo" => {
            if !matches!(actual_executable, "cargo" | "rustup") {
                return false;
            }
            if args.get(1).is_none_or(|arg| arg != "run") {
                return false;
            }
            if cwd == workspace_root && !cargo_selects_trading_bot(args) {
                return false;
            }
            args.iter()
                .position(|arg| arg == "--")
                .and_then(|separator| args.get(separator + 1))
                .is_some_and(|arg| is_training_subcommand(arg))
        }
        _ => false,
    }
}

fn cargo_selects_trading_bot(args: &[String]) -> bool {
    args.windows(2).any(|window| {
        matches!(window[0].as_str(), "-p" | "--package" | "--bin") && window[1] == "trading_bot_0"
    }) || args
        .iter()
        .any(|arg| ["--package=trading_bot_0", "--bin=trading_bot_0"].contains(&arg.as_str()))
}

fn is_training_subcommand(arg: &str) -> bool {
    matches!(arg, "train" | "train-planner" | "genetic" | "pretrain")
}

fn run_dir_from_name(name: &str) -> Option<RunDir> {
    RunDir::named(RUNS_PATH, name).ok()
}

fn latest_observable_run() -> Option<RunDir> {
    let mut runs = fs::read_dir(RUNS_PATH)
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter_map(|entry| {
            let root = entry.path();
            let weights = root.join("weights");
            if !weights.is_dir() || !has_observable_data(&root) {
                return None;
            }
            let activity = [
                root.clone(),
                root.join("training.log"),
                root.join("gens"),
                weights.clone(),
            ]
            .into_iter()
            .filter_map(|path| fs::metadata(path).ok()?.modified().ok())
            .max()?;
            Some((activity, entry.file_name(), root, weights))
        })
        .collect::<Vec<_>>();
    runs.sort_by(|a, b| (b.0, &b.1).cmp(&(a.0, &a.1)));
    let (_, _, root, weights) = runs.into_iter().next()?;
    Some(RunDir {
        gens: root.join("gens"),
        log_file: root.join("training.log"),
        root,
        weights,
    })
}

fn has_observable_data(root: &Path) -> bool {
    fs::read_dir(root.join("gens"))
        .ok()
        .is_some_and(|mut entries| entries.any(|entry| entry.is_ok()))
}

fn append_training_exit_status(log_file: Option<&Path>, pid: u32, status: ExitStatus) {
    let Some(log_file) = log_file else {
        return;
    };

    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_file)
    else {
        return;
    };

    use std::io::Write;
    let _ = writeln!(
        file,
        "\ntraining process exited: pid={} {}",
        pid,
        format_exit_status(status)
    );
}

fn format_exit_status(status: ExitStatus) -> String {
    if let Some(code) = status.code() {
        return format!("exit_code={} success={}", code, status.success());
    }

    #[cfg(unix)]
    if let Some(signal) = status.signal() {
        return format!("signal={} core_dumped={}", signal, status.core_dumped());
    }

    format!("status={status}")
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

fn trading_bots_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../trading_bots")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_run(name: &str) -> RunDir {
        let root = PathBuf::from("/tmp").join(name);
        RunDir {
            gens: root.join("gens"),
            weights: root.join("weights"),
            log_file: root.join("training.log"),
            root,
        }
    }

    #[test]
    fn pinned_historical_run_survives_live_detection_and_can_follow_live_again() {
        let live = test_run("live-run");
        let historical = test_run("historical-run");
        let mut state = ProcessManagerState {
            inference_process: None,
            training_process: None,
            active_run: Some(clone_run_dir(&live)),
            live_run: Some(clone_run_dir(&live)),
            view_pinned: false,
            cached_training_running: true,
            last_training_check: Instant::now(),
        };

        state.pin_view_run(historical);
        state.update_live_run(Some(clone_run_dir(&live)));
        assert_eq!(
            state.active_run.as_ref().map(|run| run.root.as_path()),
            Some(Path::new("/tmp/historical-run"))
        );

        state.follow_live_run();
        assert_eq!(
            state.active_run.as_ref().map(|run| run.root.as_path()),
            Some(Path::new("/tmp/live-run"))
        );
    }

    #[test]
    fn planner_output_identifies_its_run_without_run_flag() {
        let args = strings(&[
            "target/release/trading_bot_0",
            "train-planner",
            "--world-model-weights",
            "wm.ot",
            "--output",
            "training/runs/pope64_fa4_planner_rl_best_v1/weights/planner.ot",
            "--updates",
            "1000",
        ]);
        assert_eq!(
            run_name_from_args(&args),
            Some("pope64_fa4_planner_rl_best_v1")
        );
    }

    #[test]
    fn planner_output_equals_form_is_supported() {
        let args = strings(&[
            "trading_bot_0",
            "train-planner",
            "--output=/repo/training/runs/run-a/weights/custom.ot",
        ]);
        assert_eq!(run_name_from_args(&args), Some("run-a"));
    }

    #[test]
    fn output_outside_a_run_weights_directory_is_not_claimed() {
        let args = strings(&[
            "trading_bot_0",
            "train-planner",
            "--output",
            "weights/planner.ot",
        ]);
        assert_eq!(run_name_from_args(&args), None);
    }

    #[test]
    fn explicit_run_name_takes_precedence_over_output() {
        let args = strings(&[
            "trading_bot_0",
            "pretrain",
            "--run",
            "named",
            "--output",
            "training/runs/other/weights/model.ot",
        ]);
        assert_eq!(run_name_from_args(&args), Some("named"));
    }

    #[test]
    fn spawned_training_commands_use_the_hermetic_launcher() {
        let command = trading_bot_command();
        assert_eq!(
            Path::new(command.get_program()),
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../trading_bots/run-release-cuda.sh")
        );
        assert_eq!(command.get_current_dir(), Some(Path::new(WORKSPACE_ROOT)));
    }

    #[test]
    fn only_complete_ppo_checkpoint_names_resume_in_place() {
        assert!(is_ppo_resume_path(Path::new(
            "training/runs/source/weights/ppo_ep42.ot"
        )));
        assert!(!is_ppo_resume_path(Path::new(
            "training/runs/source/weights/pretrain_heads_best.ot"
        )));
        assert!(!is_ppo_resume_path(Path::new(
            "training/runs/source/weights/planner.ot"
        )));
    }

    #[test]
    fn only_repo_training_invocations_are_owned() {
        let workspace = Path::new("/repo");
        let bots = Path::new("/repo/trading_bots");

        assert!(is_training_invocation(
            bots,
            Path::new("/home/user/.cargo/bin/rustup"),
            &strings(&["cargo", "run", "--release", "--", "train", "--run", "run-a"]),
            workspace,
            bots,
        ));
        assert!(is_training_invocation(
            workspace,
            Path::new("/usr/bin/cargo"),
            &strings(&["cargo", "run", "-p", "trading_bot_0", "--", "train-planner",]),
            workspace,
            bots,
        ));
        assert!(is_training_invocation(
            bots,
            Path::new("/repo/target/release/trading_bot_0"),
            &strings(&["/repo/target/release/trading_bot_0", "pretrain"]),
            workspace,
            bots,
        ));

        assert!(!is_training_invocation(
            Path::new("/other"),
            Path::new("/usr/bin/cargo"),
            &strings(&["cargo", "run", "--", "train"]),
            workspace,
            bots,
        ));
        assert!(!is_training_invocation(
            workspace,
            Path::new("/usr/bin/cargo"),
            &strings(&["cargo", "run", "-p", "other", "--", "train"]),
            workspace,
            bots,
        ));
        assert!(!is_training_invocation(
            bots,
            Path::new("/usr/bin/python"),
            &strings(&["python", "tool.py", "train"]),
            workspace,
            bots,
        ));
        assert!(!is_training_invocation(
            bots,
            Path::new("/usr/bin/bash"),
            &strings(&["bash", "-lc", "trading_bot_0 train"]),
            workspace,
            bots,
        ));
        assert!(!is_training_invocation(
            bots,
            Path::new("/usr/bin/python"),
            &strings(&["trading_bot_0", "train"]),
            workspace,
            bots,
        ));
    }

    #[test]
    fn opening_training_log_preserves_existing_run_history() {
        use std::io::Write;

        let path = std::env::temp_dir().join(format!(
            "tui-training-log-{}-{}.log",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::write(&path, "previous session\n").unwrap();

        let mut log = open_training_log(&path).unwrap();
        writeln!(log, "new session").unwrap();
        drop(log);

        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "previous session\nnew session\n"
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn completed_inference_is_reaped_before_the_next_launch_check() {
        let child = Command::new("sh").args(["-c", "exit 0"]).spawn().unwrap();
        let mut state = ProcessManagerState {
            inference_process: Some(child),
            training_process: None,
            active_run: None,
            live_run: None,
            view_pinned: false,
            cached_training_running: false,
            last_training_check: Instant::now(),
        };
        for _ in 0..100 {
            if !state.is_anything_running() {
                assert!(state.inference_process.is_none());
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("inference child did not exit in time");
    }

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).to_owned()).collect()
    }
}
