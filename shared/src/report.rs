use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub const RL_META_REPORT_BASES: &[&str] = &[
    "final_assets",
    "cumulative_reward",
    "outperformance",
    "policy_loss",
    "value_loss",
    "explained_var",
    "actor_grad_norm",
    "critic_grad_norm",
    "total_commissions",
    "beta_policy",
    "advantage_stats_log",
    "logit_scale",
    "clip_fraction",
    "clip_gap",
    "approx_kl",
    "kl_lr",
    "policy_entropy",
    "temporal_embed_debug",
    "gate_stats",
    "hl_gauss_return_range",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub title: String,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    pub scale: ScaleKind,
    pub kind: ReportKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSeries {
    pub label: String,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradePoint {
    pub index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleBar {
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportKind {
    Simple {
        values: Vec<f32>,
        ema_alpha: Option<f64>,
    },
    MultiLine {
        series: Vec<ReportSeries>,
    },
    Assets {
        total: Vec<f32>,
        cash: Vec<f32>,
        positioned: Option<Vec<f32>>,
        benchmark: Option<Vec<f32>>,
    },
    BuySell {
        prices: Vec<f32>,
        buys: Vec<TradePoint>,
        sells: Vec<TradePoint>,
    },
    CandleCompare {
        actual: Vec<CandleBar>,
        predicted: Vec<CandleBar>,
    },
    Observations {
        observation_tickers: Vec<String>,
        action_tickers: Vec<String>,
        static_observations: Vec<Vec<f32>>,
        attention_weights: Vec<Vec<f32>>,
        action_step0: Option<Vec<f32>>,
        action_final: Option<Vec<f32>>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ScaleKind {
    Linear,
    Symlog,
}

impl ReportKind {
    pub fn to_lines(&self) -> Vec<String> {
        match self {
            ReportKind::Simple { values, .. } => values
                .iter()
                .enumerate()
                .map(|(i, v)| format!("{i}\t{v}"))
                .collect(),
            ReportKind::MultiLine { series } => {
                let max_len = series.iter().map(|s| s.values.len()).max().unwrap_or(0);
                let mut lines = Vec::with_capacity(max_len);
                for i in 0..max_len {
                    let mut line = format!("{i}");
                    for s in series {
                        if let Some(v) = s.values.get(i) {
                            line.push('\t');
                            line.push_str(&s.label);
                            line.push('=');
                            line.push_str(&v.to_string());
                        }
                    }
                    lines.push(line);
                }
                lines
            }
            ReportKind::Assets {
                total,
                cash,
                positioned,
                benchmark,
            } => {
                let max_len = total.len().max(cash.len());
                let mut lines = Vec::with_capacity(max_len);
                for i in 0..max_len {
                    let mut line = format!("{i}");
                    if let Some(v) = total.get(i) {
                        line.push_str(&format!("\ttotal={v}"));
                    }
                    if let Some(v) = cash.get(i) {
                        line.push_str(&format!("\tcash={v}"));
                    }
                    if let Some(pos) = positioned.as_ref().and_then(|p| p.get(i)) {
                        line.push_str(&format!("\tpositioned={pos}"));
                    }
                    if let Some(bench) = benchmark.as_ref().and_then(|b| b.get(i)) {
                        line.push_str(&format!("\tbenchmark={bench}"));
                    }
                    lines.push(line);
                }
                lines
            }
            ReportKind::BuySell {
                prices,
                buys,
                sells,
            } => {
                let mut buy_map: std::collections::HashSet<usize> =
                    std::collections::HashSet::new();
                let mut sell_map: std::collections::HashSet<usize> =
                    std::collections::HashSet::new();
                for b in buys {
                    buy_map.insert(b.index as usize);
                }
                for s in sells {
                    sell_map.insert(s.index as usize);
                }
                let mut lines = Vec::with_capacity(prices.len());
                for (i, price) in prices.iter().enumerate() {
                    let mut line = format!("{i}\tprice={price}");
                    if buy_map.contains(&i) {
                        line.push_str("\tbuy=1");
                    }
                    if sell_map.contains(&i) {
                        line.push_str("\tsell=1");
                    }
                    lines.push(line);
                }
                lines
            }
            ReportKind::CandleCompare { actual, predicted } => {
                let max_len = actual.len().max(predicted.len());
                let mut lines = Vec::with_capacity(max_len);
                for i in 0..max_len {
                    let mut line = format!("{i}");
                    if let Some(c) = actual.get(i) {
                        line.push_str(&format!(
                            "\tactual=o:{:.6},h:{:.6},l:{:.6},c:{:.6}",
                            c.open, c.high, c.low, c.close
                        ));
                    }
                    if let Some(c) = predicted.get(i) {
                        line.push_str(&format!(
                            "\tpredicted=o:{:.6},h:{:.6},l:{:.6},c:{:.6}",
                            c.open, c.high, c.low, c.close
                        ));
                    }
                    lines.push(line);
                }
                lines
            }
            ReportKind::Observations {
                observation_tickers,
                action_tickers,
                static_observations,
                attention_weights,
                action_step0,
                action_final,
            } => {
                let mut lines = Vec::new();
                if !observation_tickers.is_empty() {
                    lines.push(format!(
                        "observation_tickers\t{}",
                        observation_tickers.join(",")
                    ));
                }
                if !action_tickers.is_empty() {
                    lines.push(format!("action_tickers\t{}", action_tickers.join(",")));
                }
                if let Some(action) = action_step0 {
                    lines.push(format!("action_step0\t{}", format_vec_f32(action)));
                }
                if let Some(action) = action_final {
                    lines.push(format!("action_final\t{}", format_vec_f32(action)));
                }
                for (i, obs) in static_observations.iter().enumerate() {
                    lines.push(format!("static\t{i}\t{}", format_vec_f32(obs)));
                }
                for (i, attn) in attention_weights.iter().enumerate() {
                    lines.push(format!("attn\t{i}\t{}", format_vec_f32(attn)));
                }
                lines
            }
        }
    }
}

fn format_vec_f32(values: &[f32]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

pub fn read_report(path: impl AsRef<Path>) -> io::Result<Report> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|error| report_io_error("read", path, error))?;
    postcard::from_bytes(&bytes).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("decode report {}: {error}", path.display()),
        )
    })
}

pub fn write_report(path: impl AsRef<Path>, report: &Report) -> io::Result<()> {
    let path = path.as_ref();
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        fs::create_dir_all(parent)
            .map_err(|error| report_io_error("create parent directory for", path, error))?;
    }

    let bytes = postcard::to_stdvec(report).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("encode report {}: {error}", path.display()),
        )
    })?;
    let (temporary, mut file) = create_temporary_sibling(path)?;
    let result = (|| {
        file.write_all(&bytes)
            .map_err(|error| report_io_error("write temporary report for", path, error))?;
        file.sync_all()
            .map_err(|error| report_io_error("sync temporary report for", path, error))?;
        drop(file);
        fs::rename(&temporary, path).map_err(|error| report_io_error("publish", path, error))?;
        if let Some(parent) = parent {
            File::open(parent)
                .and_then(|directory| directory.sync_all())
                .map_err(|error| report_io_error("sync parent directory for", path, error))?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn create_temporary_sibling(path: &Path) -> io::Result<(PathBuf, File)> {
    let file_name = path.file_name().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "report path has no file name")
    })?;
    for _ in 0..100 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let mut temporary_name = file_name.to_os_string();
        temporary_name.push(format!(".tmp-{}-{sequence}", std::process::id()));
        let temporary = path.with_file_name(temporary_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => return Ok((temporary, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(report_io_error("create temporary file for", path, error)),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique report temporary file",
    ))
}

fn report_io_error(operation: &str, path: &Path, error: io::Error) -> io::Error {
    io::Error::new(
        error.kind(),
        format!("{operation} report {}: {error}", path.display()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn test_report(value: f32) -> Report {
        Report {
            title: "atomic".to_owned(),
            x_label: None,
            y_label: None,
            scale: ScaleKind::Linear,
            kind: ReportKind::Simple {
                values: vec![value; 512],
                ema_alpha: None,
            },
        }
    }

    fn temp_path(test: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "shared-report-{test}-{}-{}.report.bin",
            std::process::id(),
            TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn atomic_writer_reports_directory_failures() {
        let parent = temp_path("not-a-directory");
        fs::write(&parent, b"file").unwrap();
        let error = write_report(parent.join("report.bin"), &test_report(1.0)).unwrap_err();
        assert!(matches!(
            error.kind(),
            io::ErrorKind::AlreadyExists | io::ErrorKind::NotADirectory
        ));
        fs::remove_file(parent).unwrap();
    }

    #[test]
    fn truncated_report_is_an_explicit_decode_error() {
        let path = temp_path("truncated");
        write_report(&path, &test_report(1.0)).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes.truncate(bytes.len() / 2);
        fs::write(&path, bytes).unwrap();
        assert_eq!(
            read_report(&path).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn concurrent_readers_never_observe_partial_reports() {
        let path = Arc::new(temp_path("concurrent"));
        write_report(path.as_ref(), &test_report(0.0)).unwrap();
        let reader_path = Arc::clone(&path);
        let reader = thread::spawn(move || {
            for _ in 0..2_000 {
                let report = read_report(reader_path.as_ref()).unwrap();
                let ReportKind::Simple { values, .. } = report.kind else {
                    panic!("unexpected report kind");
                };
                assert_eq!(values.len(), 512);
                assert!(values.iter().all(|value| *value == values[0]));
            }
        });
        for value in 1..=100 {
            write_report(path.as_ref(), &test_report(value as f32)).unwrap();
        }
        reader.join().unwrap();
        fs::remove_file(path.as_ref()).unwrap();
    }
}
