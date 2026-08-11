use anyhow::{bail, Context, Result};
use rand::seq::index;
use shared::paths::RUNS_PATH;
use shared::report::{CandleBar, Report, ReportKind};
use shared::run_dir::RunDir;
use std::fs;
use std::path::{Component, Path, PathBuf};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: report_cli <generation> <report_name> [ticker|inference_set/episode] [--run NAME|--run-root PATH] [--runs-root PATH] [--sample N] [--min N] [--max N] [--var NAME]");
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
    let mut run_name: Option<String> = None;
    let mut run_root: Option<PathBuf> = None;
    let mut runs_root = PathBuf::from(RUNS_PATH);
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
        if arg == "--run" {
            i += 1;
            run_name = Some(args.get(i).context("missing --run value")?.clone());
            i += 1;
            continue;
        }
        if arg == "--run-root" {
            i += 1;
            run_root = Some(PathBuf::from(
                args.get(i).context("missing --run-root value")?,
            ));
            i += 1;
            continue;
        }
        if arg == "--runs-root" {
            i += 1;
            runs_root = PathBuf::from(args.get(i).context("missing --runs-root value")?);
            i += 1;
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
    if run_name.is_some() && run_root.is_some() {
        bail!("--run and --run-root are mutually exclusive");
    }

    let run = resolve_run(&runs_root, run_name.as_deref(), run_root.as_deref())?;
    let report_path = build_report_path(&run.gens, generation, &report_name, ticker)?;

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
        lines = select_by_report_value(&report.kind, lines, count, false, var_filter.as_deref());
    } else if let Some(count) = max {
        lines = select_by_report_value(&report.kind, lines, count, true, var_filter.as_deref());
    }
    if let Some(count) = sample {
        if count >= lines.len() {
            for line in lines {
                println!("{}", format_line(&line, var_filter.as_deref()));
            }
        } else {
            let mut rng = rand::rng();
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

fn resolve_run(
    runs_root: &Path,
    run_name: Option<&str>,
    run_root: Option<&Path>,
) -> Result<RunDir> {
    Ok(match (run_name, run_root) {
        (Some(name), None) => RunDir::select(runs_root, name)?,
        (None, Some(root)) => RunDir::open(root)?,
        (None, None) => RunDir::latest(runs_root.to_string_lossy().as_ref())?,
        (Some(_), Some(_)) => unreachable!(),
    })
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

fn select_by_report_value(
    kind: &ReportKind,
    lines: Vec<String>,
    count: usize,
    pick_max: bool,
    var_filter: Option<&str>,
) -> Vec<String> {
    let ReportKind::CandleCompare { actual, predicted } = kind else {
        return select_by_value(lines, count, pick_max, var_filter);
    };
    let mut scored = lines
        .into_iter()
        .enumerate()
        .filter_map(|(index, line)| {
            let mut values = Vec::with_capacity(8);
            if var_filter.is_none() || var_filter == Some("actual") {
                if let Some(candle) = actual.get(index) {
                    values.extend(candle_values(candle));
                }
            }
            if var_filter.is_none() || var_filter == Some("predicted") {
                if let Some(candle) = predicted.get(index) {
                    values.extend(candle_values(candle));
                }
            }
            values.retain(|value| value.is_finite());
            let score = if pick_max {
                values.into_iter().reduce(f32::max)
            } else {
                values.into_iter().reduce(f32::min)
            }?;
            Some((score, line))
        })
        .collect::<Vec<_>>();
    if pick_max {
        scored.sort_by(|left, right| right.0.total_cmp(&left.0));
    } else {
        scored.sort_by(|left, right| left.0.total_cmp(&right.0));
    }
    scored
        .into_iter()
        .take(count)
        .map(|(_, line)| line)
        .collect()
}

fn candle_values(candle: &CandleBar) -> [f32; 4] {
    [candle.open, candle.high, candle.low, candle.close]
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
    subpath: Option<&str>,
) -> Result<PathBuf> {
    let mut path = base_path.clone();
    path.push(generation.to_string());
    if let Some(subpath) = subpath {
        path.push(safe_report_subpath(subpath)?);
    }
    path.push(format!("{report_name}.report.bin"));
    Ok(path)
}

fn safe_report_subpath(raw: &str) -> Result<PathBuf> {
    let path = Path::new(raw);
    let components = path.components().collect::<Vec<_>>();
    if components.is_empty()
        || components.len() > 2
        || components
            .iter()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("report subpath must contain one or two safe relative components");
    }
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::report::{ReportKind, ScaleKind};

    #[test]
    fn nested_inference_episode_report_path_decodes() {
        let root = std::env::temp_dir().join(format!(
            "report-cli-nested-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let nested = root.join("gens/7/planner_inference_test_set/planner_test_000_TEST");
        fs::create_dir_all(&nested).unwrap();
        let report = Report {
            title: "Nested".to_owned(),
            x_label: Some("step".to_owned()),
            y_label: None,
            scale: ScaleKind::Linear,
            kind: ReportKind::Simple {
                values: vec![1.0],
                ema_alpha: None,
            },
        };
        let path = nested.join("planner_position.report.bin");
        fs::write(&path, postcard::to_stdvec(&report).unwrap()).unwrap();

        let resolved = build_report_path(
            &root.join("gens"),
            7,
            "planner_position",
            Some("planner_inference_test_set/planner_test_000_TEST"),
        )
        .unwrap();
        let decoded: Report = postcard::from_bytes(&fs::read(resolved).unwrap()).unwrap();
        assert_eq!(decoded.title, "Nested");
        assert!(build_report_path(&root, 7, "assets", Some("../escape")).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn candle_min_max_use_structured_ohlc_values() {
        let kind = ReportKind::CandleCompare {
            actual: vec![
                CandleBar {
                    open: 10.0,
                    high: 12.0,
                    low: 9.0,
                    close: 11.0,
                },
                CandleBar {
                    open: 3.0,
                    high: 5.0,
                    low: 2.0,
                    close: 4.0,
                },
                CandleBar {
                    open: 90.0,
                    high: 100.0,
                    low: 80.0,
                    close: 95.0,
                },
            ],
            predicted: vec![
                CandleBar {
                    open: 20.0,
                    high: 30.0,
                    low: 15.0,
                    close: 25.0,
                },
                CandleBar {
                    open: f32::NAN,
                    high: 8.0,
                    low: 7.0,
                    close: 7.5,
                },
            ],
        };
        let lines = kind.to_lines();
        assert!(select_by_report_value(&kind, lines.clone(), 1, true, None)[0].starts_with('2'));
        assert!(select_by_report_value(&kind, lines.clone(), 1, false, None)[0].starts_with('1'));
        assert!(
            select_by_report_value(&kind, lines.clone(), 1, true, Some("actual"))[0]
                .starts_with('2')
        );
        assert!(
            select_by_report_value(&kind, lines, 1, true, Some("predicted"))[0].starts_with('0')
        );
    }

    #[test]
    fn explicit_historical_run_selection_does_not_change_latest() {
        let runs = std::env::temp_dir().join(format!(
            "report-cli-runs-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let first = RunDir::create_fresh(runs.to_str().unwrap(), Some("first")).unwrap();
        let second = RunDir::create_fresh(runs.to_str().unwrap(), Some("second")).unwrap();
        assert_eq!(resolve_run(&runs, Some("first"), None).unwrap(), first);
        assert_eq!(RunDir::latest(runs.to_str().unwrap()).unwrap(), second);
        fs::remove_dir_all(runs).unwrap();
    }
}
