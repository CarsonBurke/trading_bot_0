//! Non-interactive entry point: turn `*.report.bin` files into PNG images.
//!
//! This shares the single rendering path with the interactive chart viewer
//! (`report_renderer::render_report_with_options`); nothing here re-implements
//! any drawing.

use anyhow::{anyhow, bail, Context, Result};
use shared::report::read_report;
use std::path::{Path, PathBuf};

use crate::report_renderer::render_report_with_options;

pub const REPORT_SUFFIX: &str = ".report.bin";

pub const USAGE: &str = "\
trading-bot-tui — training run browser

Usage:
  trading-bot-tui                                  launch the interactive TUI
  trading-bot-tui render <input>... --out <dir>    render reports to PNG files

render options:
  -o, --out <dir>    output directory (created if missing) [required]
      --skip <n>     drop the first n samples of every series (default 0)
      --no-legend    omit the series legend
  -h, --help         show this help

<input> is a `*.report.bin` file, or a directory that is searched recursively
for them. Directory inputs keep their relative layout under <dir>.
";

/// Command line as parsed from `argv[1..]`.
#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Help,
    Render(RenderArgs),
}

#[derive(Debug, PartialEq, Eq)]
pub struct RenderArgs {
    pub inputs: Vec<PathBuf>,
    pub out: PathBuf,
    pub skip: usize,
    pub show_legend: bool,
}

pub fn parse(args: &[String]) -> Result<Command> {
    let (head, rest) = args
        .split_first()
        .ok_or_else(|| anyhow!("no command given"))?;
    match head.as_str() {
        "-h" | "--help" | "help" => return Ok(Command::Help),
        "render" => {}
        other => bail!("unknown command `{other}`\n\n{USAGE}"),
    }

    let mut inputs = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut skip = 0usize;
    let mut show_legend = true;

    let mut iter = rest.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "-o" | "--out" => {
                let value = iter
                    .next()
                    .ok_or_else(|| anyhow!("`{arg}` needs a directory"))?;
                out = Some(PathBuf::from(value));
            }
            "--skip" => {
                let value = iter.next().ok_or_else(|| anyhow!("`--skip` needs a count"))?;
                skip = value
                    .parse()
                    .with_context(|| format!("`--skip` expects a count, got `{value}`"))?;
            }
            "--no-legend" => show_legend = false,
            other if other.starts_with('-') => bail!("unknown option `{other}`\n\n{USAGE}"),
            other => inputs.push(PathBuf::from(other)),
        }
    }

    if inputs.is_empty() {
        bail!("render needs at least one `{REPORT_SUFFIX}` file or directory\n\n{USAGE}");
    }
    let out = out.ok_or_else(|| anyhow!("render needs `--out <dir>`\n\n{USAGE}"))?;

    Ok(Command::Render(RenderArgs {
        inputs,
        out,
        skip,
        show_legend,
    }))
}

pub fn run(args: &[String]) -> Result<()> {
    match parse(args)? {
        Command::Help => {
            print!("{USAGE}");
            Ok(())
        }
        Command::Render(args) => {
            let rendered = render_all(&args)?;
            println!("rendered {} report(s) to {}", rendered, args.out.display());
            Ok(())
        }
    }
}

/// A report bin plus where its PNG belongs, relative to the output directory.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RenderTarget {
    pub source: PathBuf,
    pub relative_png: PathBuf,
}

fn render_all(args: &RenderArgs) -> Result<usize> {
    let targets = collect_targets(&args.inputs)?;
    if targets.is_empty() {
        bail!("no `{REPORT_SUFFIX}` files found in the given inputs");
    }
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("create output directory {}", args.out.display()))?;

    let mut failures = Vec::new();
    let mut rendered = 0usize;
    for target in &targets {
        let destination = args.out.join(&target.relative_png);
        match render_one(&target.source, &destination, args.skip, args.show_legend) {
            Ok(()) => {
                rendered += 1;
                println!("{} -> {}", target.source.display(), destination.display());
            }
            Err(error) => {
                eprintln!("{}: {error:#}", target.source.display());
                failures.push(target.source.clone());
            }
        }
    }

    if !failures.is_empty() {
        bail!(
            "{} of {} report(s) could not be rendered",
            failures.len(),
            targets.len()
        );
    }
    Ok(rendered)
}

pub fn render_one(source: &Path, destination: &Path, skip: usize, show_legend: bool) -> Result<()> {
    let report = read_report(source).with_context(|| format!("read {}", source.display()))?;
    let image = render_report_with_options(&report, skip, show_legend, None)
        .with_context(|| format!("render {}", source.display()))?;
    if let Some(parent) = destination.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create directory {}", parent.display()))?;
    }
    image
        .save(destination)
        .with_context(|| format!("write {}", destination.display()))?;
    Ok(())
}

/// Expand file and directory inputs into a sorted, de-duplicated target list.
pub fn collect_targets(inputs: &[PathBuf]) -> Result<Vec<RenderTarget>> {
    let mut targets = Vec::new();
    for input in inputs {
        let metadata = std::fs::metadata(input)
            .with_context(|| format!("stat input {}", input.display()))?;
        if metadata.is_dir() {
            for entry in walkdir::WalkDir::new(input).sort_by_file_name() {
                let entry = entry.with_context(|| format!("walk {}", input.display()))?;
                if !entry.file_type().is_file() || !is_report_bin(entry.path()) {
                    continue;
                }
                let relative = entry
                    .path()
                    .strip_prefix(input)
                    .unwrap_or_else(|_| Path::new(""))
                    .to_path_buf();
                targets.push(RenderTarget {
                    source: entry.path().to_path_buf(),
                    relative_png: png_name(&relative),
                });
            }
        } else {
            if !is_report_bin(input) {
                bail!("{} is not a `{REPORT_SUFFIX}` file", input.display());
            }
            targets.push(RenderTarget {
                source: input.clone(),
                relative_png: png_name(Path::new(
                    input
                        .file_name()
                        .ok_or_else(|| anyhow!("{} has no file name", input.display()))?,
                )),
            });
        }
    }
    targets.sort();
    targets.dedup();
    Ok(targets)
}

fn is_report_bin(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(REPORT_SUFFIX) && name.len() > REPORT_SUFFIX.len())
}

/// `a/b/foo.report.bin` -> `a/b/foo.png`.
fn png_name(relative: &Path) -> PathBuf {
    let name = relative
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let stem = name.strip_suffix(REPORT_SUFFIX).unwrap_or(name);
    match relative.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(format!("{stem}.png")),
        _ => PathBuf::from(format!("{stem}.png")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::report::{
        write_report, CandleBar, QuantileBand, Report, ReportKind, ReportSeries, ScaleKind,
    };
    use std::sync::atomic::{AtomicU32, Ordering};

    static SCRATCH_SEQUENCE: AtomicU32 = AtomicU32::new(0);

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(label: &str) -> Self {
            let unique = SCRATCH_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "tui-render-cli-{label}-{}-{unique}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    /// Deterministic synthetic snapshot: a realized walk, a quantile fan that
    /// widens with the horizon around it, and a few draws from the same fan.
    fn synthetic_fan() -> (Vec<CandleBar>, Vec<QuantileBand>, Vec<ReportSeries>) {
        let mut actual = Vec::new();
        let mut centre = Vec::new();
        let mut level = 100.0f32;
        for step in 0..24 {
            let drift = ((step as f32) * 0.7).sin() * 1.5;
            let open = level;
            let close = level + drift;
            actual.push(CandleBar {
                open,
                high: open.max(close) + 0.6,
                low: open.min(close) - 0.6,
                close,
            });
            centre.push(100.0 + drift * 0.25);
            level = close;
        }
        // Width grows as sqrt(horizon), which is what a chained sampler produces.
        let spread = |step: usize| (1.0 + step as f32).sqrt() * 0.9;
        let bands: Vec<QuantileBand> = [(0.10, -1.28f32), (0.50, 0.0), (0.90, 1.28)]
            .into_iter()
            .map(|(probability, z)| QuantileBand {
                probability,
                closes: centre
                    .iter()
                    .enumerate()
                    .map(|(step, mid)| mid + z * spread(step))
                    .collect(),
            })
            .collect();
        let samples: Vec<ReportSeries> = (0..3)
            .map(|draw| ReportSeries {
                label: format!("draw {}", draw + 1),
                values: centre
                    .iter()
                    .enumerate()
                    .map(|(step, mid)| {
                        mid + ((draw as f32 * 2.1 + step as f32 * 0.45).sin()) * spread(step)
                    })
                    .collect(),
            })
            .collect();
        (actual, bands, samples)
    }

    #[test]
    fn candle_fan_report_round_trips_from_bin_to_png() {
        let input = Scratch::new("candle-in");
        let output = Scratch::new("candle-out");
        let (actual, bands, samples) = synthetic_fan();
        let up_bars = actual.iter().filter(|c| c.close >= c.open).count();
        assert!(up_bars > 0 && up_bars < actual.len(), "need both directions");

        let report = Report {
            title: "pretrain candle rollout".to_owned(),
            x_label: Some("bar".to_owned()),
            y_label: Some("price".to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::CandleFan {
                actual,
                bands,
                samples,
            },
        };
        let source = input.path().join("pretrain_candle_rollout_0.report.bin");
        write_report(&source, &report).unwrap();

        run(&args(&[
            "render",
            source.to_str().unwrap(),
            "--out",
            output.path().to_str().unwrap(),
        ]))
        .unwrap();

        let png = output.path().join("pretrain_candle_rollout_0.png");
        let bytes = std::fs::metadata(&png).unwrap().len();
        assert!(bytes > 2_000, "png suspiciously small: {bytes} bytes");

        let image = image::open(&png).unwrap().to_rgb8();
        assert_eq!((image.width(), image.height()), (2560, 780));

        // Candle bodies are filled rectangles in the theme's green and red; a
        // line chart of the same data would produce neither in bulk.
        let green = image
            .pixels()
            .filter(|p| p[1] as i16 > p[0] as i16 + 30 && p[1] as i16 > p[2] as i16 + 30)
            .count();
        let red = image
            .pixels()
            .filter(|p| p[0] as i16 > p[1] as i16 + 30 && p[0] as i16 > p[2] as i16 + 30)
            .count();
        assert!(green > 2_000, "expected filled green bodies, got {green}");
        assert!(red > 2_000, "expected filled red bodies, got {red}");

        // The fan itself must be on the canvas, not just the realized bars: the
        // quantile loci are blue-dominant, which no candle body is.
        let fan = image
            .pixels()
            .filter(|p| p[2] as i16 > p[0] as i16 + 30 && p[2] as i16 > p[1] as i16 + 10)
            .count();
        assert!(fan > 500, "expected quantile loci to be drawn, got {fan}");
    }

    #[test]
    fn directory_input_renders_every_report_and_keeps_its_layout() {
        let input = Scratch::new("dir-in");
        let output = Scratch::new("dir-out");
        let simple = Report {
            title: "loss".to_owned(),
            x_label: Some("step".to_owned()),
            y_label: Some("nats".to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::Simple {
                values: vec![3.0, 2.5, 2.1, 1.9],
                ema_alpha: Some(0.5),
            },
        };
        write_report(input.path().join("top.report.bin"), &simple).unwrap();
        write_report(input.path().join("nested/inner.report.bin"), &simple).unwrap();
        std::fs::write(input.path().join("manifest.json"), b"{}").unwrap();

        run(&args(&[
            "render",
            input.path().to_str().unwrap(),
            "-o",
            output.path().to_str().unwrap(),
        ]))
        .unwrap();

        assert!(output.path().join("top.png").is_file());
        assert!(output.path().join("nested/inner.png").is_file());
        assert!(!output.path().join("manifest.png").exists());
    }

    #[test]
    fn argument_errors_are_explicit() {
        assert!(parse(&args(&["render", "a.report.bin"])).is_err());
        assert!(parse(&args(&["render", "--out", "x"])).is_err());
        assert!(parse(&args(&["render", "--skip", "many", "a", "-o", "x"])).is_err());
        assert!(parse(&args(&["renderr", "a", "-o", "x"])).is_err());
        assert_eq!(parse(&args(&["--help"])).unwrap(), Command::Help);
        assert_eq!(
            parse(&args(&["render", "a.report.bin", "-o", "out", "--no-legend"])).unwrap(),
            Command::Render(RenderArgs {
                inputs: vec![PathBuf::from("a.report.bin")],
                out: PathBuf::from("out"),
                skip: 0,
                show_legend: false,
            })
        );
    }

    #[test]
    fn only_report_bins_are_collected_and_names_lose_both_extensions() {
        assert_eq!(
            png_name(Path::new("gens/0/pretrain_nll_bar.report.bin")),
            PathBuf::from("gens/0/pretrain_nll_bar.png")
        );
        assert!(is_report_bin(Path::new("x/a.report.bin")));
        assert!(!is_report_bin(Path::new("x/.report.bin")));
        assert!(!is_report_bin(Path::new("x/a.bin")));
    }
}
