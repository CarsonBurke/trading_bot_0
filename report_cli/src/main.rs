use anyhow::{bail, Context, Result};
use shared::paths::TRAINING_PATH;
use shared::report::Report;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: report_cli <generation> <report_name> [ticker]");
    }

    let generation = args[1]
        .parse::<usize>()
        .context("generation must be an integer")?;
    let report_name = normalize_report_name(&args[2]);
    let ticker = args.get(3).map(|s| s.as_str());

    let base_path = find_training_path().unwrap_or_else(|| PathBuf::from(TRAINING_PATH));
    let report_path = build_report_path(&base_path, generation, &report_name, ticker);
    let bytes = fs::read(&report_path)
        .with_context(|| format!("failed to read report {}", report_path.display()))?;
    let report: Report = postcard::from_bytes(&bytes).context("failed to decode report")?;

    for line in report.kind.to_lines() {
        println!("{line}");
    }
    Ok(())
}

fn normalize_report_name(raw: &str) -> String {
    let mut name = raw.trim().to_string();
    if let Some(stripped) = name.strip_suffix(".report.bin") {
        name = stripped.to_string();
    } else if let Some(stripped) = name.strip_suffix(".report") {
        name = stripped.to_string();
    } else if let Some(stripped) = name.strip_suffix(".bin") {
        name = stripped.to_string();
    }
    name.to_ascii_lowercase().replace(' ', "_")
}

fn build_report_path(
    base_path: &PathBuf,
    generation: usize,
    report_name: &str,
    ticker: Option<&str>,
) -> PathBuf {
    let mut path = base_path.clone();
    path.push("gens");
    path.push(generation.to_string());
    if let Some(ticker) = ticker {
        path.push(ticker);
    }
    path.push(format!("{report_name}.report.bin"));
    path
}

fn find_training_path() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd);
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(parent) = manifest_dir.parent() {
        candidates.push(parent.to_path_buf());
    }

    for start in candidates {
        let mut dir = Some(start.as_path());
        while let Some(current) = dir {
            let training = current.join("training");
            if training.join("gens").exists() {
                return Some(training);
            }
            dir = current.parent();
        }
    }
    None
}
