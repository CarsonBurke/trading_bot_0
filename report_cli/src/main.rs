use anyhow::{bail, Context, Result};
use rand::seq::index;
use rand::thread_rng;
use shared::paths::TRAINING_PATH;
use shared::report::Report;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: report_cli <generation> <report_name> [ticker] [--sample N] [--min N] [--max N] [--var NAME]");
    }

    let generation = args[1]
        .parse::<usize>()
        .context("generation must be an integer")?;
    let report_name = normalize_report_name(&args[2]);
    let mut ticker: Option<&str> = None;
    let mut sample: Option<usize> = None;
    let mut min: Option<usize> = None;
    let mut max: Option<usize> = None;
    let mut var_filter: Option<String> = None;
    let mut i = 3;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--sample" || arg == "-s" {
            let next = args.get(i + 1).context("missing --sample value")?;
            let count = next.parse::<usize>().context("sample must be an integer")?;
            sample = Some(count);
            i += 2;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--sample=") {
            let count = value
                .parse::<usize>()
                .context("sample must be an integer")?;
            sample = Some(count);
            i += 1;
            continue;
        }
        if arg == "--min" {
            let next = args.get(i + 1).context("missing --min value")?;
            let count = next.parse::<usize>().context("min must be an integer")?;
            min = Some(count);
            i += 2;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--min=") {
            let count = value.parse::<usize>().context("min must be an integer")?;
            min = Some(count);
            i += 1;
            continue;
        }
        if arg == "--max" {
            let next = args.get(i + 1).context("missing --max value")?;
            let count = next.parse::<usize>().context("max must be an integer")?;
            max = Some(count);
            i += 2;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--max=") {
            let count = value.parse::<usize>().context("max must be an integer")?;
            max = Some(count);
            i += 1;
            continue;
        }
        if arg == "--var" || arg == "-v" {
            let next = args.get(i + 1).context("missing --var value")?;
            var_filter = Some(next.clone());
            i += 2;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--var=") {
            var_filter = Some(value.to_string());
            i += 1;
            continue;
        }
        if ticker.is_none() {
            ticker = Some(arg.as_str());
            i += 1;
            continue;
        }
        bail!("unexpected argument: {arg}");
    }
    if min.is_some() && max.is_some() {
        bail!("--min and --max are mutually exclusive");
    }

    let base_path = find_training_path().unwrap_or_else(|| PathBuf::from(TRAINING_PATH));
    let report_path = build_report_path(&base_path, generation, &report_name, ticker);
    let bytes = fs::read(&report_path)
        .with_context(|| format!("failed to read report {}", report_path.display()))?;
    let report: Report = postcard::from_bytes(&bytes).context("failed to decode report")?;

    let mut lines = report.kind.to_lines();
    if let Some(ref filter) = var_filter {
        lines.retain(|line| {
            line.split('\t')
                .any(|t| t.split_once('=').is_some_and(|(k, _)| k == filter))
        });
    }
    if let Some(count) = min {
        lines = select_by_value(lines, count, false, var_filter.as_deref());
    } else if let Some(count) = max {
        lines = select_by_value(lines, count, true, var_filter.as_deref());
    }
    if let Some(count) = sample {
        if count >= lines.len() {
            for line in lines {
                println!("{}", format_line(&line, var_filter.as_deref()));
            }
        } else {
            let mut rng = thread_rng();
            let indices = index::sample(&mut rng, lines.len(), count);
            for idx in indices.iter() {
                println!("{}", format_line(&lines[idx], var_filter.as_deref()));
            }
        }
    } else {
        for line in lines {
            println!("{}", format_line(&line, var_filter.as_deref()));
        }
    }
    Ok(())
}

fn select_by_value(
    lines: Vec<String>,
    count: usize,
    pick_max: bool,
    var_filter: Option<&str>,
) -> Vec<String> {
    let mut scored: Vec<(f32, String)> = lines
        .into_iter()
        .filter_map(|line| {
            let values = extract_values(&line, var_filter);
            if values.is_empty() {
                None
            } else if pick_max {
                values.iter().cloned().reduce(f32::max).map(|v| (v, line))
            } else {
                values.iter().cloned().reduce(f32::min).map(|v| (v, line))
            }
        })
        .collect();

    if pick_max {
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    let take_count = count.min(scored.len());
    scored
        .into_iter()
        .take(take_count)
        .map(|(_, line)| line)
        .collect()
}

fn extract_values(line: &str, var_filter: Option<&str>) -> Vec<f32> {
    let mut values = Vec::new();
    for token in line.split('\t') {
        if let Some((key, value)) = token.split_once('=') {
            if let Some(filter) = var_filter {
                if key == filter {
                    collect_values(value, &mut values);
                }
            } else {
                collect_values(value, &mut values);
            }
            continue;
        }
        if var_filter.is_some() {
            continue;
        }
        if token.parse::<usize>().is_ok() {
            continue;
        }
        collect_values(token, &mut values);
    }
    values
}

fn collect_values(token: &str, values: &mut Vec<f32>) {
    for part in token.split(',') {
        if let Ok(value) = part.parse::<f32>() {
            values.push(value);
        }
    }
}

fn format_line(line: &str, var_filter: Option<&str>) -> String {
    let Some(var) = var_filter else {
        return line.to_string();
    };
    let mut parts = Vec::new();
    for token in line.split('\t') {
        if let Some((key, _)) = token.split_once('=') {
            if key == var {
                parts.push(token);
            }
        } else {
            parts.push(token);
        }
    }
    parts.join("\t")
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
