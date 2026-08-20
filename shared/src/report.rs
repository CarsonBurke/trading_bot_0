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

/// Every chart base the DISTRIBUTIONAL PRETRAINER writes, and the single source of truth
/// for it. The writer's own test asserts a full cycle produces each of these, and the TUI
/// builds its meta-chart list by extending from this slice.
///
/// It lives here rather than in either consumer because the two failure modes are silent
/// in opposite directions and both shipped: a base registered with no writer renders as a
/// permanently blank panel, and a base written but not registered is a chart nobody can
/// see. One list makes both unrepresentable.
pub const PRETRAIN_REPORT_BASES: &[&str] = &[
    "pretrain_nll_bar",
    "pretrain_nll_bar_diag896",
    "pretrain_forecast_nll",
    "pretrain_nll_dof",
    "pretrain_nll_vs_baselines",
    "pretrain_crps_dof",
    "pretrain_pit_hist",
    "pretrain_dyn_loss",
    "pretrain_kl_loss",
    "pretrain_total_loss",
    "pretrain_loss_shares",
    "pretrain_growth_term",
    "pretrain_belief_autocorr",
    "pretrain_dyn_vs_identity",
    "pretrain_rollout_nll",
    "pretrain_dir_acc",
    "pretrain_lr",
    "pretrain_muon_momentum",
    "pretrain_grad_norm",
    "pretrain_unique_bar_reuse",
    "pretrain_stage_coverage",
    "pretrain_pass_coverage",
    "pretrain_pass_multiplicity",
    "pretrain_pass_remainder",
    "pretrain_stage_conditioning",
    "pretrain_effective_rank",
    "pretrain_promotions",
    "pretrain_schedule",
    "pretrain_capacity",
    "pretrain_market_coverage",
    "pretrain_candle_rollout_pit",
    "pretrain_candle_rollout_dclose",
    "pretrain_candle_rollout_band",
    "pretrain_candle_rollout_coverage",
    "pretrain_trade_growth",
    "pretrain_trade_vs_baselines",
    "pretrain_trade_cost_curve",
    "pretrain_trade_sharpe",
    "pretrain_trade_exposure",
    "pretrain_trade_cap_curve",
    "pretrain_trade_free_kelly",
    "pretrain_trade_tail",
    // The EPOCH-INDEXED panel. Distinct bases from the trade series above on purpose:
    // those are the dense record-tick curves measured at every validation, these are one
    // point per pass over the corpus. Neither is derivable from the other and neither
    // overwrites the other.
    "pretrain_epoch_trade_edge",
    "pretrain_epoch_trade",
    "pretrain_epoch_progress",
    // Written once, at the end of a run, by `PretrainReporter::finish`.
    "pretrain_test",
    // Written by the corpus loader at startup rather than by the reporter, but it lands
    // in the same directory and is read the same way, so it is registered the same way.
    "pretrain_corpus_anomalies",
    // Written by the AUXILIARY-resolution stream at every epoch boundary, for the same
    // reason and by the same convention: the reporter's row schema is the deployment
    // resolution's, and an auxiliary resolution has its own supports, its own ramp and its
    // own held-out geometry, so its curve is a separate object rather than another column.
    // One point per pass per auxiliary resolution. Absent from a run that named none.
    "pretrain_auxiliary_nll",
    // Written by `trading_bots::torch::train::portfolio::write_portfolio_bench`, which runs
    // ONE book over a calendar-aligned panel of the held-out split rather than averaging
    // per-window bets. Distinct from the `pretrain_trade_*` bases above and not derivable
    // from them: those measure a single name's log-optimal bet, these measure a portfolio
    // under one shared capital constraint.
    "pretrain_portfolio_equity",
    "pretrain_portfolio_metrics",
    "pretrain_portfolio_gross_curve",
    "pretrain_portfolio_frontier",
    // The same writer's edge-versus-cost table: what one name-bar of forecast is worth in bps
    // beside what one one-way trade in it costs, per liquidity decile of the traded panel. The
    // two halves have to be measured on the same panel to be a comparison at all, which is why
    // this base is the portfolio writer's and not `pretrain_cost_deciles`'.
    "pretrain_portfolio_edge_vs_cost",
    // Written by `trading_bots::torch::train::portfolio_cost::write_cost_capacity_reports`.
    // Properties of the CORPUS rather than of a training step: a spread, an ADV and a
    // realized cross-sectional covariance are measured from stored bars and do not move
    // when a step does, so no in-run reporter cycle can produce them.
    "pretrain_cost_deciles",
    "pretrain_capacity_curve",
    "pretrain_cross_correlation",
    // Written by `trading_bots::torch::train::pretrain_reports::write_mean_calibration`, from
    // the multi-checkpoint mean-calibration experiment. One point per CHECKPOINT rather than
    // per step of one run: a Mincer-Zarnowitz slope needs a whole held-out pass, and the
    // recalibrated policy beside it needs a second pass on a block-disjoint fit slice, so
    // neither is producible from inside a training cycle.
    "pretrain_mean_calibration",
    "pretrain_shrunk_policy",
    // Written by the same writer, from the same two passes: the COST-AWARE sizing axis.
    // `trade_bench`'s Kelly solve maximizes `E[ln(1 + f R)]`, which carries no cost term, so
    // the position is chosen frictionlessly and the charge is levied afterwards on whatever
    // turnover that produced. Under proportional costs the optimal policy instead has a
    // no-trade region, so this base is the band swept as an axis, under both fill rules, with
    // the gain over the unbanded incumbent taken PAIRED window by window. Indexed by BAND
    // WIDTH rather than by step or by cap, and not derivable from `pretrain_shrunk_policy`:
    // that one varies the MEAN the solve is handed and this one varies how often the solve is
    // acted on, and whether the two overlap is the third panel of this base.
    "pretrain_no_trade_band",
    // Written by the same writer, from the same two passes: WHERE the measured edge lives.
    // A hit rate below a coin flip beside an edge whose interval excludes zero cannot be read
    // as directional skill, so the arm table re-scores the identical windows with the model's
    // MAGNITUDE destroyed at matched gross exposure, and again with its SIGN destroyed, each
    // paired against the null and against the actual policy over the same blocks. Indexed by
    // ARM, so it is not derivable from any base above: `pretrain_skill_profile` scores the
    // predictor with no policy at all and every `pretrain_trade_*` base conditions on the
    // undamaged Kelly policy.
    "pretrain_edge_attribution",
    // The panel underneath those arms, indexed by CHECKPOINT: `corr(f, R)`, `corr(|f|, |R|)`
    // and the mean size of a winning bar against a losing one, which is the arithmetic a
    // sub-coin-flip hit rate with positive growth has to satisfy.
    "pretrain_edge_panel",
    // The same panel cut by DECILE of the model's own uncapped `|f*|`. The confidence axis is
    // the discriminator the arm table cannot supply on its own: a hit rate flat across every
    // decile while growth concentrates in the top ones is a size-carried result, and a hit
    // rate that rises with `|f*|` is a direction predictor that knows where its sign is good.
    "pretrain_edge_confidence",
    // The sign-hysteresis frontier, indexed by FLIP MARGIN in bps of predicted mean. On a book
    // whose turnover is almost entirely sign flips, holding the sign longer is the only lever
    // left that can move the cost, and margin zero is the sign-only arm exactly - so this base
    // extends the arm table along an axis the arm table does not have. Not derivable from
    // `pretrain_no_trade_band`: that dead-zones the MAGNITUDE of the target, which on a
    // two-valued book suppresses re-sizings, while this one suppresses REVERSALS.
    "pretrain_edge_hysteresis",
    // The recalibration shrink crossed with sign hysteresis, indexed by CELL of a 2x2. Both
    // levers cut the cost of the same book by trading less, so neither their gains nor their
    // break-evens can be added, and the second difference that decides it is not recoverable
    // from `pretrain_shrunk_policy` and `pretrain_edge_hysteresis` side by side - those score
    // each lever against the incumbent, never against each other on the same windows. Distinct
    // from the band-versus-shrink overlap for the same reason the frontier is distinct from the
    // band: this crosses a REVERSAL rule with the shrink, not a magnitude dead-zone.
    "pretrain_edge_composition",
    // How fast the CURRENT one-bar signal's directional content decays with holding horizon,
    // indexed by HORIZON in bars, with no policy and no cost anywhere in the measurement. It
    // bounds what a one-bar signal HELD longer can be worth, and deliberately says nothing
    // about a model TRAINED on a k-bar target, whose predictable component and noise floor are
    // different quantities. Distinct from `pretrain_horizon_frontier`, which scores POLICIES
    // under two constructions rather than the bare signal.
    "pretrain_signal_decay",
    // Written by `trading_bots::torch::train::skill::write_skill_profile`: the DIRECTIONAL
    // skill of the predictor, scored with no trading policy anywhere in the measurement.
    // Decile-indexed rather than step-indexed - the x axis is the model's own confidence, not
    // training progress - so it is not producible from inside a training cycle and is not
    // derivable from any `pretrain_trade_*` base, which all condition on a Kelly policy.
    "pretrain_skill_profile",
    // Written by `trading_bots::torch::train::horizon::write_horizon_frontier`. Break-even
    // cost against the HOLDING HORIZON, for the model and its three baselines under both the
    // stale-one-bar and the k-bar-aggregate construction. Not derivable from
    // `pretrain_portfolio_frontier`: that curve varies a no-trade band on a one-bar forecast,
    // which freezes stale positions, while this one varies the horizon the forecast is OF.
    // A whole held-out panel and a sampled multi-bar rollout per point, so no in-run cycle
    // can produce it.
    "pretrain_horizon_frontier",
    // Written by `trading_bots::torch::train::support_moments::fit_support_moments` via
    // `pretrain_reports::write_support_decode`. Properties of the SUPPORT ARTIFACT alone: the
    // fitted per-bin conditional means measured against the persisted bin geometry, beside the
    // EDGE decode that every production first-moment consumer actually reads, beside the
    // hardcoded two-bin stand-in that preceded the measurement. No model, no checkpoint and no
    // step is involved, so an in-run reporter cycle cannot produce them and they do not move
    // when a step does. Registered here rather than only in the TUI because `meta_chart_bases`
    // extends from THIS slice, which is what makes a written-but-unregistered base
    // unrepresentable.
    "support_decode_moments",
    "support_decode_bins",
    // Written by `PretrainReporter::record_epoch`, and the ONLY run-scoped coverage bases in
    // this list. Every `pretrain_pass_*` and `pretrain_stage_*` base above is a PER-PASS census:
    // `CoverageAudit::require_full_pass` pins within-pass multiplicity to exactly one, so those
    // panels read "every bar once, twice: 0" on the third pass of a three-pass run exactly as on
    // the first. That is correct within a pass and it was read as a claim about the RUN for an
    // entire analysis session, in preference to `pretrain_unique_bar_reuse` showing 2.85 on the
    // same screen. These two carry the cross-pass fact — passes delivered, projected and asked
    // for, and bars by how many times the RUN has targeted them — and are the only bases that
    // can answer "how many times has the model seen this bar".
    "cover_effective_epochs",
    "cover_run_bar_exposure",
    // Written by `trading_bots::torch::train::mem_probe::mem_probe` via
    // `pretrain_reports::write_mem_probe`. The multi-epoch MEMORIZATION test: held-out NLL
    // against TRAIN-split NLL along the epoch spine, and the within-checkpoint contrast between
    // bars the run had trained on three times and bars it had trained on twice at the same
    // step. Neither is producible from inside a training cycle - the spine needs several
    // checkpoints and the contrast needs the training pass partition reconstructed at a
    // checkpoint's own step - and the two are deliberately separate bases because one is
    // contaminated by calendar and by learning rate while the other is randomized by
    // construction, and a reader must never mistake the first for a discriminator.
    "memprobe_epoch_spine",
    "memprobe_one_repetition",
    "memprobe_recency",
    "memprobe_bootstrap_stability",
    // Written by `trading_bots::torch::train::bar_family::fit_bar_families` via
    // `pretrain_reports::write_bar_family`. The offline GATE on replacing the 128-way discrete bar
    // support with a continuous per-DOF mixed likelihood: fitted density against the empirical
    // histogram per DOF, the `r` tail on log-log axes with the measured pairwise-slope band, the
    // component sweep, the marginal NLL against the discrete competitor on one stated footing, the
    // atom census with the u/v lattice probe, and the truncation bound a ruin licence implies.
    // Every panel is a property of a DRAW and a fitted family rather than of an optimizer step, so
    // no in-run reporter cycle can produce any of them and none moves when a step does. Registered
    // here rather than only in the TUI because `meta_chart_bases` extends from THIS slice, which
    // is what makes a written-but-unregistered base unrepresentable.
    "bar_family_density_r",
    "bar_family_density_s",
    "bar_family_density_u",
    "bar_family_density_v",
    "bar_family_density_w",
    "bar_family_tail_r",
    "bar_family_k_sweep",
    "bar_family_nll",
    "bar_family_atoms",
    "bar_family_ruin_bound",
    // Written by `trading_bots::torch::train::split_seams::audit_split_seams` via
    // `pretrain_reports::write_bar_seams`. The corporate-action SEAM audit: whether the extreme `r`
    // bars in the corpus are market moves or unadjusted split seams, and what the seams contaminate.
    // The exceedance census over all 451,507,140 bars, the nearest-simple-rational
    // cross-tabulation of `exp(r)` that decides the split hypothesis, the `s`/`w` comparison
    // against matched ordinary bars, the six pairwise tail slopes with and without the seams, the
    // catch-all bin contamination, and the ruin licence on both sides of the book. Every panel is a
    // property of the STORED BARS and of a support artifact read from disk, so no in-run reporter
    // cycle can produce any of them and none moves when a step does. Registered here rather than
    // only in the TUI because `meta_chart_bases` extends from THIS slice, which is what makes a
    // written-but-unregistered base unrepresentable.
    "bar_seam_census",
    "bar_seam_ratios",
    "bar_seam_context",
    "bar_seam_tail_r",
    "bar_seam_bin_mass",
    "bar_seam_ruin_licence",
    // Written by `trading_bots::torch::train::pretrain_reports::write_heldout_power`, from the
    // window draw that `pretrain-calibration` performs BEFORE it opens a checkpoint. The census
    // of every split at one context — bars, near-disjoint windows, symbols — and the interval a
    // traded prefix of the addressed split can support, as a function of the `(symbol, calendar
    // month)` BLOCK count counted over the real draw. Properties of the CORPUS and of a
    // seed-pinned draw, never of a model: no checkpoint is loaded and nothing is scored, which
    // is the point. `Split::Test` is scored once for the whole campaign, so whether it has the
    // power to resolve the effect being looked for has to be a chart that exists before the draw
    // is spent. Registered here rather than only in the TUI because `meta_chart_bases` extends
    // from THIS slice, which is what makes a written-but-unregistered base unrepresentable.
    "pretrain_heldout_census",
    "pretrain_heldout_power",
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
