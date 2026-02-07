use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

const OUTPUT_DIR: &str = "benchmark_results";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRun {
    pub batch: i64,
    pub seq_len: i64,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub time_ms: f64,
    pub config: BenchmarkRun,
}

impl BenchmarkResult {
    pub fn new(name: &str, time_ms: f64, config: BenchmarkRun) -> Self {
        Self {
            name: name.to_string(),
            time_ms,
            config,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    pub timestamp: DateTime<Utc>,
    pub git_commit: Option<String>,
    pub cuda_version: Option<String>,
    pub results: Vec<BenchmarkResult>,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        let git_commit = std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string());

        let cuda_version = std::env::var("TORCH_CUDA_VERSION").ok();

        Self {
            timestamp: Utc::now(),
            git_commit,
            cuda_version,
            results: Vec::new(),
        }
    }

    pub fn add(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    pub fn save(&self) -> std::io::Result<PathBuf> {
        let output_dir = PathBuf::from(OUTPUT_DIR);
        fs::create_dir_all(&output_dir)?;

        let filename = format!("bench_{}.json", self.timestamp.format("%Y%m%d_%H%M%S"));
        let path = output_dir.join(&filename);

        let json = serde_json::to_string_pretty(self)?;
        fs::write(&path, json)?;

        // Also write to latest.json for easy access
        let latest_path = output_dir.join("latest.json");
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&latest_path, json)?;

        println!("Saved: {}", path.display());
        println!("Saved: {}", latest_path.display());

        Ok(path)
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}
