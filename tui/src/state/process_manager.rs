use anyhow::Result;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub struct ProcessManagerState {
    pub inference_process: Option<Child>,
    pub training_process: Option<Child>,
    cached_training_running: bool,
    last_training_check: Instant,
}

impl ProcessManagerState {
    pub fn new() -> Self {
        Self {
            inference_process: None,
            training_process: None,
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

        if self.training_process.is_some() {
            self.cached_training_running = true;
            return true;
        }

        if let Ok(output) = Command::new("pgrep")
            .args(["-f", "trading.*train"])
            .output()
        {
            if !output.stdout.is_empty() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for pid in stdout.lines() {
                    if let Ok(pid_num) = pid.trim().parse::<u32>() {
                        if let Ok(cmdline_output) = Command::new("ps")
                            .args(["-p", &pid_num.to_string(), "-o", "args="])
                            .output()
                        {
                            let cmdline = String::from_utf8_lossy(&cmdline_output.stdout);
                            if cmdline.contains("train") && !cmdline.contains("tui") {
                                self.cached_training_running = true;
                                return true;
                            }
                        }
                    }
                }
            }
        }

        self.cached_training_running = false;
        false
    }

    pub fn is_anything_running(&mut self) -> bool {
        self.is_training_running() || self.inference_process.is_some()
    }

    pub fn start_training(&mut self, weights: Option<String>) -> Result<()> {
        if self.is_anything_running() {
            return Ok(());
        }

        let log_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("../training/training.log")?;

        let mut cmd = Command::new("cargo");
        cmd.current_dir("../trading_bots")
            .arg("run")
            .arg("--release")
            .arg("--")
            .arg("train");

        if let Some(w) = weights {
            cmd.arg("--weights").arg(w);
        }

        cmd.stdin(Stdio::null())
            .stdout(log_file.try_clone()?)
            .stderr(log_file);

        cmd.spawn()?;
        self.cached_training_running = true;

        Ok(())
    }

    pub fn start_inference(&mut self, weights: String, ticker: Option<String>, episodes: usize) -> Result<()> {
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
            let _ = child.kill();
        }

        Command::new("pkill")
            .args(["-f", "trading.*train"])
            .spawn()?;

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
