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

/// Registered base for the Corpus-owned causal-admission audit.
pub const PRETRAIN_UNIVERSE_INTEGRITY_REPORT_BASE: &str = "pretrain_universe_integrity";

/// Component responsible for writing a registered pretraining report.
///
/// Ownership is part of the registry contract rather than a parallel exemption list in a
/// writer test. A base can therefore be discovered by the TUI and attributed to exactly one
/// production writer from the same declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PretrainReportOwner {
    Run,
    Corpus,
    Auxiliary,
    Portfolio,
    CostCapacity,
    Calibration,
    SkillProfile,
    Horizon,
    SupportDecode,
    MemorizationProbe,
    BarFamily,
    FeatureScreen,
    SplitSeams,
    HeldoutPower,
    Recirculation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PretrainReportSpec {
    pub base: &'static str,
    pub owner: PretrainReportOwner,
}

macro_rules! pretrain_report_registry {
    ($( $base:literal => $owner:ident, )+) => {
        /// Every chart base in the pretraining reporting domain.
        ///
        /// The TUI consumes this exact slice. [`PRETRAIN_REPORT_SPECS`] is generated from the
        /// same declarations, so discovery names and writer ownership cannot drift.
        pub const PRETRAIN_REPORT_BASES: &[&str] = &[$($base,)+];

        pub const PRETRAIN_REPORT_SPECS: &[PretrainReportSpec] = &[
            $(PretrainReportSpec {
                base: $base,
                owner: PretrainReportOwner::$owner,
            },)+
        ];
    };
}

pretrain_report_registry! {
    "pretrain_nll_bar" => Run,
    "pretrain_nll_bar_diag896" => Run,
    "pretrain_independent_marginal_nll" => Run,
    "pretrain_nll_dof" => Run,
    "pretrain_nll_vs_baselines" => Run,
    "pretrain_crps_dof" => Run,
    "pretrain_pit_hist" => Run,
    "pretrain_dyn_loss" => Run,
    "pretrain_kl_loss" => Run,
    "pretrain_total_loss" => Run,
    "pretrain_loss_shares" => Run,
    "pretrain_growth_term" => Run,
    "pretrain_belief_autocorr" => Run,
    "pretrain_dyn_vs_identity" => Run,
    "pretrain_teacher_forced_rollout_score" => Run,
    "pretrain_ancestral_calibration" => Run,
    "pretrain_ancestral_tails" => Run,
    "pretrain_ancestral_bar_validity" => Run,
    "pretrain_ancestral_distribution_drift" => Run,
    "pretrain_dir_acc" => Run,
    "pretrain_lr" => Run,
    "pretrain_muon_momentum" => Run,
    "pretrain_sdlr_alpha" => Run,
    "pretrain_sdlr_evidence" => Run,
    "pretrain_smd_idbd_gain" => Run,
    "pretrain_smd_idbd_credit" => Run,
    "pretrain_direct_return_nll" => Run,
    "pretrain_direct_return_valid_coverage" => Run,
    "pretrain_direct_return_gain" => Run,
    "pretrain_direct_return_calibration" => Run,
    "pretrain_grad_norm" => Run,
    "pretrain_unique_bar_reuse" => Run,
    "pretrain_stage_coverage" => Run,
    "pretrain_pass_coverage" => Run,
    "pretrain_pass_multiplicity" => Run,
    "pretrain_pass_remainder" => Run,
    "pretrain_stage_conditioning" => Run,
    "pretrain_effective_rank" => Run,
    "pretrain_promotions" => Run,
    "pretrain_schedule" => Run,
    "pretrain_capacity" => Run,
    "pretrain_market_coverage" => Run,
    "pretrain_candle_rollout_pit" => Run,
    "pretrain_candle_rollout_dclose" => Run,
    "pretrain_candle_rollout_band" => Run,
    "pretrain_candle_rollout_coverage" => Run,
    "pretrain_trade_growth" => Run,
    "pretrain_trade_vs_baselines" => Run,
    "pretrain_trade_cost_curve" => Run,
    "pretrain_trade_sharpe" => Run,
    "pretrain_trade_exposure" => Run,
    "pretrain_trade_cap_curve" => Run,
    "pretrain_trade_free_kelly" => Run,
    "pretrain_trade_tail" => Run,
    "pretrain_epoch_trade_edge" => Run,
    "pretrain_epoch_trade" => Run,
    "pretrain_epoch_progress" => Run,
    "pretrain_corpus_anomalies" => Corpus,
    "pretrain_universe_integrity" => Corpus,
    "pretrain_feature_screen_ic" => FeatureScreen,
    "pretrain_feature_screen_spread" => FeatureScreen,
    "pretrain_feature_screen_controls" => FeatureScreen,
    "pretrain_auxiliary_nll" => Auxiliary,
    "pretrain_portfolio_equity" => Portfolio,
    "pretrain_portfolio_metrics" => Portfolio,
    "pretrain_portfolio_gross_curve" => Portfolio,
    "pretrain_portfolio_frontier" => Portfolio,
    "pretrain_portfolio_edge_vs_cost" => Portfolio,
    "pretrain_cost_deciles" => CostCapacity,
    "pretrain_capacity_curve" => CostCapacity,
    "pretrain_cross_correlation" => CostCapacity,
    "pretrain_mean_calibration" => Calibration,
    "pretrain_shrunk_policy" => Calibration,
    "pretrain_no_trade_band" => Calibration,
    "pretrain_edge_attribution" => Calibration,
    "pretrain_edge_panel" => Calibration,
    "pretrain_edge_confidence" => Calibration,
    "pretrain_edge_hysteresis" => Calibration,
    "pretrain_edge_composition" => Calibration,
    "pretrain_signal_decay" => Calibration,
    "pretrain_skill_profile" => SkillProfile,
    "pretrain_horizon_frontier" => Horizon,
    "pretrain_receding_kelly" => Horizon,
    "pretrain_receding_covariance" => Horizon,
    "pretrain_receding_attribution" => Horizon,
    "pretrain_receding_policy_frontier" => Horizon,
    "pretrain_receding_hysteresis" => Horizon,
    "support_decode_moments" => SupportDecode,
    "support_decode_bins" => SupportDecode,
    "cover_effective_epochs" => Run,
    "cover_run_bar_exposure" => Run,
    "memprobe_epoch_spine" => MemorizationProbe,
    "memprobe_one_repetition" => MemorizationProbe,
    "memprobe_recency" => MemorizationProbe,
    "memprobe_bootstrap_stability" => MemorizationProbe,
    "bar_family_density_r" => BarFamily,
    "bar_family_density_s" => BarFamily,
    "bar_family_density_u" => BarFamily,
    "bar_family_density_v" => BarFamily,
    "bar_family_density_w" => BarFamily,
    "bar_family_tail_r" => BarFamily,
    "bar_family_k_sweep" => BarFamily,
    "bar_family_nll" => BarFamily,
    "bar_family_atoms" => BarFamily,
    "bar_family_ruin_bound" => BarFamily,
    "bar_seam_census" => SplitSeams,
    "bar_seam_ratios" => SplitSeams,
    "bar_seam_context" => SplitSeams,
    "bar_seam_tail_r" => SplitSeams,
    "bar_seam_bin_mass" => SplitSeams,
    "bar_seam_ruin_licence" => SplitSeams,
    "pretrain_heldout_census" => HeldoutPower,
    "pretrain_heldout_power" => HeldoutPower,
    "pretrain_recirculation_sweep" => Recirculation,
}

pub fn pretrain_report_bases_owned_by(
    owner: PretrainReportOwner,
) -> impl Iterator<Item = &'static str> {
    PRETRAIN_REPORT_SPECS
        .iter()
        .filter(move |spec| spec.owner == owner)
        .map(|spec| spec.base)
}

pub fn pretrain_report_owner(base: &str) -> Option<PretrainReportOwner> {
    PRETRAIN_REPORT_SPECS
        .iter()
        .find(|spec| spec.base == base)
        .map(|spec| spec.owner)
}

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

/// One marginal-quantile locus of a sampled path, e.g. the p10 of the sampled
/// close at each horizon.
///
/// A quantile locus is NOT a path the process can take: `closes[t]` is a
/// property of the marginal distribution at horizon `t`, computed independently
/// per horizon, so consecutive entries need not belong to any single draw. That
/// is exactly why [`ReportKind::CandleFan`] carries genuine draws beside these.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantileBand {
    /// Probability in `(0, 1)`.
    pub probability: f64,
    pub closes: Vec<f32>,
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
    /// A realized path against the predictive law it was drawn from: the bars
    /// that happened, the quantile fan of the sampled continuations, and a few
    /// of the sampled continuations themselves.
    ///
    /// There is deliberately no "predicted" field. A single line cannot stand
    /// in for a distribution over paths, and a fan centre rendered against one
    /// realization invites the reader to score a pointwise error that no
    /// forecast ever claimed.
    CandleFan {
        /// Realized bars, the only thing here that actually happened.
        actual: Vec<CandleBar>,
        /// Quantile loci of the sampled close, ASCENDING in probability.
        bands: Vec<QuantileBand>,
        /// Genuine draws of the close path from the predictive law.
        samples: Vec<ReportSeries>,
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
            ReportKind::CandleFan {
                actual,
                bands,
                samples,
            } => {
                let max_len = actual
                    .len()
                    .max(bands.iter().map(|b| b.closes.len()).max().unwrap_or(0))
                    .max(samples.iter().map(|s| s.values.len()).max().unwrap_or(0));
                let mut lines = Vec::with_capacity(max_len);
                for i in 0..max_len {
                    let mut line = format!("{i}");
                    if let Some(c) = actual.get(i) {
                        line.push_str(&format!(
                            "\tactual=o:{:.6},h:{:.6},l:{:.6},c:{:.6}",
                            c.open, c.high, c.low, c.close
                        ));
                    }
                    for band in bands {
                        if let Some(close) = band.closes.get(i) {
                            line.push_str(&format!(
                                "\tp{:02}={close:.6}",
                                (band.probability * 100.0).round() as i64
                            ));
                        }
                    }
                    for series in samples {
                        if let Some(close) = series.values.get(i) {
                            line.push_str(&format!("\t{}={close:.6}", series.label));
                        }
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
    fn pretrain_registry_names_and_writer_ownership_share_one_complete_contract() {
        assert_eq!(PRETRAIN_REPORT_BASES.len(), PRETRAIN_REPORT_SPECS.len());
        let mut unique = std::collections::BTreeSet::new();
        for (base, spec) in PRETRAIN_REPORT_BASES.iter().zip(PRETRAIN_REPORT_SPECS) {
            assert_eq!(base, &spec.base);
            assert!(
                unique.insert(spec.base),
                "duplicate pretrain report base {}",
                spec.base
            );
        }
        let direct_bases = PRETRAIN_REPORT_BASES
            .iter()
            .copied()
            .filter(|base| base.starts_with("pretrain_direct_return_"))
            .collect::<Vec<_>>();
        assert_eq!(
            direct_bases,
            [
                "pretrain_direct_return_nll",
                "pretrain_direct_return_valid_coverage",
                "pretrain_direct_return_gain",
                "pretrain_direct_return_calibration",
            ],
            "every direct-return base is registry-owned by the run reporter"
        );
    }

    #[test]
    fn multiline_report_round_trips_series_order_labels_and_nonfinite_gaps() {
        let path = temp_path("multiline-roundtrip");
        let report = Report {
            title: "objective diagnostics".to_owned(),
            x_label: Some("record".to_owned()),
            y_label: Some("nats/bar".to_owned()),
            scale: ScaleKind::Symlog,
            kind: ReportKind::MultiLine {
                series: vec![
                    ReportSeries {
                        label: "production".to_owned(),
                        values: vec![1.0, f32::NAN, 3.0],
                    },
                    ReportSeries {
                        label: "ablation".to_owned(),
                        values: vec![2.0, 4.0, f32::INFINITY],
                    },
                ],
            },
        };
        write_report(&path, &report).unwrap();
        let decoded = read_report(&path).unwrap();
        assert_eq!(decoded.title, report.title);
        assert_eq!(decoded.x_label, report.x_label);
        assert_eq!(decoded.y_label, report.y_label);
        assert_eq!(decoded.scale, report.scale);
        let ReportKind::MultiLine { series } = decoded.kind else {
            panic!("multiline schema changed kind");
        };
        assert_eq!(
            series
                .iter()
                .map(|item| item.label.as_str())
                .collect::<Vec<_>>(),
            vec!["production", "ablation"]
        );
        assert_eq!(series[0].values[0], 1.0);
        assert!(series[0].values[1].is_nan());
        assert_eq!(series[0].values[2], 3.0);
        assert_eq!(series[1].values[..2], [2.0, 4.0]);
        assert!(series[1].values[2].is_infinite());
        fs::remove_file(path).unwrap();
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
