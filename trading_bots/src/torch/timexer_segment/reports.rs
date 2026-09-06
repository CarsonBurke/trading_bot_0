use std::{collections::BTreeMap, path::Path};

use super::{compute::CaptureBudget, corpus::CorpusContract, features::MarketSummary};

use anyhow::{ensure, Context, Result};
use shared::report::{write_report, CandleBar, Report, ReportKind, ReportSeries, ScaleKind};

/// The report vocabulary. Every series label and chart title in this module is built from
/// these words and nothing else: the retired spellings ("preview", "full validation",
/// "relative", "absolute") left a reader guessing whether two panels named the same
/// population, which makes the charts unreadable regardless of how good the numbers are.
///
/// Splits: `training` is the running training-set estimate over the interval's batches,
/// [`SAMPLE`] the fixed fast window set scored every report interval, [`FULL`] the whole
/// disjoint held-out population scored at epoch end, and `persistence` the zero-forecast
/// baseline.
///
/// Spaces: [`NEUTRAL`] is the β-adjusted residual return the model actually optimizes;
/// [`RAW`] adds the realized market drift back so the number stays comparable with model
/// generations trained on raw returns.
///
/// A label reads `<split> <space> <quantity>`, with the space omitted wherever the quantity
/// exists in only one space: NLL, calibration and the robustness diagnostics are
/// market-neutral only.
const SAMPLE: &str = "held-out sample";
const FULL: &str = "held-out full";
const NEUTRAL: &str = "market-neutral";
const RAW: &str = "raw";

/// Axis units. Each carries the unit and, where one exists, the reading rule, so a panel is
/// interpretable without its documentation.
const RATIO_UNIT: &str = "ratio vs persistence (dimensionless; < 1 = skill)";
const LEVEL_UNIT: &str = "σ-scaled squared log-return";
const NATS_UNIT: &str = "nats per bar";
const FRACTION_UNIT: &str = "fraction of valid target bars";
const RATE_UNIT: &str =
    "fraction of valid target bars (win and hit rates > 0.5 = skill; up share is bias)";

/// Progress mixes a share of the training corpus with shares of held-out bars, so its axis
/// cannot claim one denominator; each series label names its own.
const SHARE_UNIT: &str = "fraction of the series' own population (dimensionless; 0 to 1)";

/// The recipe scalars are pure mixing weights - a fraction of a residual branch, of the
/// embedding, of an encoder layer's output or of the source layer's value - so they carry no
/// unit at all; only their distance from their init is readable.
const MIXING_UNIT: &str = "learned mixing coefficient (dimensionless)";

/// Axis units for the trading diagnostics. Each is a distinct unit, which is why the family
/// below is seven bases and not one: a share of MSE near 0.06, a correlation near 0.01, a
/// σ-scaled mean return near 0.001, a ratio near 1, a rate near 0.5, basis points near 1 and
/// an annualized Sharpe near 1 cannot share an axis without rendering six of the seven flat.
const GAIN_UNIT: &str = "share of the persistence MSE (dimensionless; > 0 = gain, and the four \
                         components sum to the total)";
const CORRELATION_UNIT: &str =
    "correlation coefficient (dimensionless; 0 = no information, ±1 = perfect)";
const COORDINATE_UNIT: &str = "σ-scaled mean log return (0 = persistence; the market-neutral \
                               close coordinate the model predicts)";
const ANCHOR_RATIO_UNIT: &str =
    "ratio vs the anchor's own persistence (dimensionless; < 1 = skill, 1 = no edge over the anchor)";
const ANCHOR_RATE_UNIT: &str =
    "fraction of scored bars where the sign is right (> 0.5 = skill; 0.5 = a coin flip)";
const BPS_UNIT: &str = "basis points per holding period (net of the labeled per-side cost, \
                        charged on entry and exit of both legs)";
const SHARPE_UNIT: &str = "annualized Sharpe ratio (dimensionless; 19,656 five-minute bars per \
                           year, non-overlapping holds assumed)";

/// The scope fact a split contributes to a title. Spelled out for the held-out sample because
/// its count is a fixed sample size, and a reader has to know it is a sample of the population
/// rather than the population.
fn split_scope(split: &str, origins: usize) -> String {
    if split == SAMPLE {
        format!("{split} {origins} fixed windows, one scored origin each")
    } else {
        format!("{split} {origins} scored origins")
    }
}

#[derive(Debug, Clone)]
pub struct Metrics {
    pub step: usize,
    pub epoch: usize,
    pub completed_origins: usize,
    pub total_origins: usize,
    pub completed_target_bars: usize,
    pub total_target_bars: usize,
    /// Dense training objective over every origin of the interval's batches.
    pub train_nll: Option<f64>,
    pub train_mse: Option<f64>,
    /// Held-out final-origin scores in σ units against the market-neutral (β-adjusted
    /// residual) targets the model optimizes; persistence is a zero forecast with √h scale.
    /// Which split these belong to is a runtime property of the point
    /// ([`Metrics::validation_is_full`]), not of the field, so the `validation_` prefix is
    /// retained as the internal spelling of "the held-out split scored here".
    pub validation_nll: f64,
    pub persistence_nll: f64,
    pub validation_mse: f64,
    pub persistence_mse: f64,
    /// The same forecast scored in raw space: the realized market drift is added back to the
    /// targets so the number stays comparable with generations trained on raw returns. The
    /// forecast is unchanged and persistence stays zero. `absolute_` is the retained internal
    /// spelling of `raw`.
    pub absolute_mse: f64,
    pub absolute_persistence_mse: f64,
    /// Fraction of valid targets inside the predicted ±1σ and ±1.96σ bands (nominal .683/.95).
    pub within_1_sigma: f64,
    pub within_2_sigma: f64,
    pub rmse_price: f64,
    pub mae_price: f64,
    pub invalid_ohlc_fraction: f64,
    /// Share of the held-out squared error contributed by the top 1% |target| elements of
    /// each batch.
    pub tail_loss_share: f64,
    /// Median over scored forecast windows of the window's forecast/persistence MSE ratio in
    /// market-neutral space (all horizons and channels); NaN when no window has a nonzero
    /// persistence error.
    pub median_window_ratio: f64,
    pub eval: EvalTiming,
    /// Mean optimizer-step wall clock over the interval.
    pub step_ms: Option<f64>,
    /// Mean per-step time the training loop waited for the host loader (the loader runs one
    /// batch ahead, so this is only the part that failed to overlap GPU work).
    pub loader_wait_ms: Option<f64>,
    pub peak_allocator_mib: Option<f64>,
    /// Peak allocator bytes during the held-out evaluation alone, in MiB. Reported because
    /// the evaluation allocates from the GLOBAL pool while the captured training step holds
    /// a private one, so the two have to fit on the device together.
    pub evaluation_peak_mib: Option<f64>,
    /// VRAM budget of the captured forward and backward, once it is armed.
    pub capture_budget: Option<CaptureBudget>,
    /// `true` when this point scored the whole held-out validation population at epoch end,
    /// `false` when it scored the fixed held-out sample window set.
    pub validation_is_full: bool,
    /// Origins this point scored, which is also its forecast-window count: scoring uses one
    /// final origin per row.
    pub validation_origins: usize,
    /// Tickers in the corpus, carried so every title can state the population averaged over.
    pub tickers: usize,
    /// Per-phase attribution of one sampled training step; `None` before the first sample.
    pub step_phases: Option<StepPhases>,
}

/// Wall-clock attribution of one evaluation in milliseconds. `total_ms` is the timed `score`
/// call and `loader_ms` the host batch wait summed over every batch. `forward_ms` and
/// `metrics_ms` are the phases of ONE sampled batch, not sums: synchronizing between the
/// phases of every batch drained the launch pipeline twice per batch, which made the
/// instrumentation a large part of what it measured. `reports_ms` is what follows outside
/// the timed path before the report files are written: candle windows and checkpoints.
#[derive(Debug, Clone, Copy, Default)]
pub struct EvalTiming {
    pub total_ms: f64,
    pub loader_ms: f64,
    pub forward_ms: f64,
    pub metrics_ms: f64,
    pub reports_ms: Option<f64>,
}

/// Per-phase attribution of ONE sampled training step in milliseconds. Only the sampled step
/// is synchronized between phases, so ordinary steps keep their asynchronous launch pipeline
/// and the measurement never serializes training. `host_batch_ms` is the loader wait, `h2d_ms`
/// the copy of the packed row block into the resident device batch, `forward_backbone_ms` the
/// patch embedding through the final norm, `forward_head_ms` the covariate projection, head
/// GEMMs, candle geometry and NLL, `backward_ms` the whole reverse pass,
/// `captured_replay_ms` the whole forward and backward as one CUDA-graph replay, and
/// `optimizer_ms` the gradient reset plus update.
///
/// A captured step cannot be split - one replay is one launch - so `captured_replay_ms` and
/// the three eager forward/backward phases are mutually exclusive: whichever does not apply
/// is `NaN`, never a zero a reader would take for "free".
#[derive(Debug, Clone, Copy, Default)]
pub struct StepPhases {
    pub host_batch_ms: f64,
    pub h2d_ms: f64,
    pub forward_backbone_ms: f64,
    pub forward_head_ms: f64,
    pub backward_ms: f64,
    pub captured_replay_ms: f64,
    pub optimizer_ms: f64,
}

/// Per-horizon held-out final-origin scores in σ-scaled log-return units, index 0 = one bar
/// ahead. `mse`/`persistence_mse` average over channels; the remaining series are
/// population-wide robustness diagnostics in market-neutral space only
/// (`hit_rate`/`up_fraction` use the close channel).
#[derive(Debug, Clone)]
pub struct HorizonCurve {
    pub mse: Vec<f64>,
    pub persistence_mse: Vec<f64>,
    /// The same forecast scored in raw space: the realized market drift added back to the
    /// targets. `absolute_` is the retained internal spelling of `raw`.
    pub absolute_mse: Vec<f64>,
    pub absolute_persistence_mse: Vec<f64>,
    /// Σ|y-ŷ| / Σ|y| over valid (bar, channel).
    pub mae_ratio: Vec<f64>,
    /// Share of valid bars where the close forecast error is strictly below persistence's.
    pub win_rate: Vec<f64>,
    /// Share of valid bars with a nonzero close forecast whose sign matches the target.
    pub hit_rate: Vec<f64>,
    /// Share of valid bars whose close forecast is positive.
    pub up_fraction: Vec<f64>,
    /// MSE ratio after dropping the bars whose |close target| is in the horizon's top 1%.
    pub trimmed_mse_ratio: Vec<f64>,
}

/// Per-horizon trading diagnostics on the market-neutral close coordinate, index 0 = one bar
/// ahead. These answer one question that MSE cannot: how much of the MSE gain is a tradable
/// conditional signal, as opposed to a constant tilt in the forecast or an artifact of
/// persistence anchoring on the last *trade* close.
///
/// The decomposition is exact. Write the forecast as `ŷ = μ + g` with `μ` the population mean
/// forecast at that horizon and `mean(g) = 0`, and let `β̂ = mean(y·g)/mean(g²)` be the
/// least-squares rescaling of the demeaned forecast. Then, as shares of the persistence MSE
/// `mean(y²)`:
///
/// - [`Self::offset_gain`] `= (2μȳ - μ²)/mean(y²)` — what a constant forecast equal to `μ`
///   already earns. Uncapturable: it is a tilt, not a conditional prediction.
/// - [`Self::demeaned_gain`] `= (mean(y·g)²/mean(g²))/mean(y²)` — the demeaned conditional
///   forecast's gain at its own best scale. This is identically `ρ²·var(y)/mean(y²)` for the
///   Pearson `ρ` in [`Self::pearson`], so it is the `1 - ρ²` implied MSE ratio the reader wants
///   to compare with the measured one, expressed in the same unit as the measured total.
/// - [`Self::scaling_gain`] `= -(β̂-1)²·mean(g²)/mean(y²)` — the residual cross term, always
///   ≤ 0: what mis-scaling the demeaned forecast's amplitude costs.
///
/// The three sum to [`Self::total_gain`] exactly.
#[derive(Debug, Clone)]
pub struct TradingCurve {
    /// `1 -` the close-channel MSE ratio.
    pub total_gain: Vec<f64>,
    /// `1 -` the four-channel MSE ratio, the quantity the headline ratio reports.
    pub all_channel_gain: Vec<f64>,
    pub offset_gain: Vec<f64>,
    pub demeaned_gain: Vec<f64>,
    pub scaling_gain: Vec<f64>,
    /// Population mean predicted close coordinate in σ units: the size of the constant tilt.
    pub mean_forecast: Vec<f64>,
    /// Population mean realized close coordinate in σ units.
    pub mean_target: Vec<f64>,
    /// Pearson correlation of the demeaned forecast with the realized coordinate, pooled.
    pub pearson: Vec<f64>,
    /// Spearman rank correlation, pooled, ordinal ranks without tie averaging.
    pub spearman: Vec<f64>,
    /// Mean over evaluation timestamps of the within-timestamp cross-ticker correlation.
    pub cross_sectional_ic: Vec<f64>,
    /// Standard error of that mean over the contributing timestamps.
    pub cross_sectional_ic_se: Vec<f64>,
    /// Close-channel MSE ratio against close-anchored persistence.
    pub close_mse_ratio: Vec<f64>,
    /// The same forecast error over persistence anchored on the mid of the origin bar's
    /// high/low instead of its close. Bid-ask bounce inflates the close-anchored denominator
    /// and nothing else, so this ratio rising toward 1 is the microstructure verdict.
    pub mid_anchor_mse_ratio: Vec<f64>,
    /// One-bar execution delay: the bar `t+1` close to bar `t+h` close move, forecast by
    /// `ŷ_h - ŷ_1`, against that same delayed persistence. Undefined (NaN) at `h = 1`.
    pub delayed_mse_ratio: Vec<f64>,
    pub close_hit_rate: Vec<f64>,
    pub mid_anchor_hit_rate: Vec<f64>,
    pub delayed_hit_rate: Vec<f64>,
    /// Hit rate over the top decile of `|g|`, the only bars a strategy would act on.
    pub top_decile_hit_rate: Vec<f64>,
    pub bottom_decile_hit_rate: Vec<f64>,
    /// Mean realized close coordinate on the side the forecast took, `mean(sign(g)·y)`, over
    /// the top decile of `|g|`. σ units.
    pub top_decile_return: Vec<f64>,
    pub bottom_decile_return: Vec<f64>,
    /// Top minus bottom: the conviction premium a decile-conditioned strategy harvests.
    pub conviction_spread_return: Vec<f64>,
    /// Distinct evaluation timestamps behind the cross-sectional statistics.
    pub cross_sections: usize,
}

/// Cost-aware long/short cross-sectional backtest driven by the same evaluation pass: at each
/// evaluation timestamp, rank tickers by predicted `h`-bar market-neutral close return, go
/// equal-weighted long the top decile and short the bottom, hold `h` bars.
///
/// This is an upper bound. It ignores market impact, borrow availability and cost, the
/// overlap between consecutive holds (which inflates the Sharpe), and the fact that a
/// timestamp's cross-section here is the held-out origins that happen to share that
/// timestamp, not a tradable universe snapshot.
#[derive(Debug, Clone)]
pub struct PortfolioCurve {
    /// Reported holding periods in bars.
    pub horizons: Vec<u64>,
    /// Per-side cost levels in basis points, charged on entry and exit of both legs.
    pub costs_bps: Vec<f64>,
    /// Outer index = cost level, inner = horizon. Mean per-period decile-spread return, bps.
    pub net_bps: Vec<Vec<f64>>,
    /// Same indexing, annualized Sharpe of that net per-period return.
    pub net_sharpe: Vec<Vec<f64>>,
    /// Contributing timestamps per horizon, the sample size behind the Sharpe.
    pub cross_sections: Vec<usize>,
}

pub struct CandleWindow {
    pub ticker: String,
    pub actual: Vec<CandleBar>,
    pub origin: usize,
    pub predicted: Vec<CandleBar>,
    /// Forecast-origin completion time, in Unix milliseconds.
    pub timestamp: i64,
}

/// The learned mixing coefficients over training, one series per scalar
/// `CausalPatchModel::recipe_scalars` returns, in the order it returns them.
///
/// The question this panel answers is whether a coefficient MOVED off its init: a U-net gate
/// that collapses toward 0 says the skip is unused, a residual lambda drifting off `√1.1` says
/// the stack is rescaling the stream, a value lambda leaving 0.5 says the layer prefers one V
/// over the other. It therefore reports POST-parameterization values - what the forward pass
/// applies - and never the stored logit.
///
/// The names are the model's, not this module's: adding a family of scalars needs no edit
/// here. `history` carries one entry per report interval, so the panel costs one host transfer
/// per interval and nothing per step.
pub fn write_recipe_scalars(
    output: &Path,
    epoch: usize,
    step: usize,
    history: &[(usize, Vec<(String, f64)>)],
) -> Result<()> {
    let Some((_, last)) = history.last() else {
        return Ok(());
    };
    ensure!(
        history.windows(2).all(|pair| pair[0].0 < pair[1].0),
        "report steps must increase"
    );
    ensure!(
        history.iter().all(|(_, scalars)| scalars.len() == last.len()
            && scalars
                .iter()
                .zip(last)
                .all(|((name, value), (expected, _))| name == expected && value.is_finite())),
        "every recipe-scalar point must carry the same finite, identically ordered scalars"
    );
    let steps: Vec<u64> = history.iter().map(|(step, _)| *step as u64).collect();
    let series = last
        .iter()
        .enumerate()
        .map(|(index, (name, _))| ReportSeries {
            label: name.clone(),
            values: history
                .iter()
                .map(|(_, scalars)| scalars[index].1 as f32)
                .collect(),
        })
        .collect();
    write_report(
        output.join("timexer_segment_recipe_scalars.report.bin"),
        &Report {
            title: format!(
                "CausalPatch epoch {epoch} step {step} | how far has each learned mixing coefficient moved from its init? | {} scalars, post-parameterization values (sigmoid applied to the U-net gate logits, the lambdas raw)",
                last.len()
            ),
            x_label: Some("optimizer step".into()),
            y_label: Some(MIXING_UNIT.into()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines { steps, series },
        },
    )?;
    Ok(())
}

pub fn write_metrics(output: &Path, points: &[Metrics]) -> Result<()> {
    let Some(last) = points.last() else {
        return Ok(());
    };
    ensure!(
        points.windows(2).all(|p| p[0].step < p[1].step),
        "report steps must increase"
    );
    ensure!(
        points.iter().all(|p| p.epoch == last.epoch
            && p.tickers > 0
            && p.validation_origins > 0
            && p.completed_origins <= p.total_origins
            && p.completed_target_bars <= p.total_target_bars
            && [
                p.validation_nll,
                p.persistence_nll,
                p.validation_mse,
                p.persistence_mse,
                p.absolute_mse,
                p.absolute_persistence_mse,
                p.rmse_price,
                p.mae_price,
                p.invalid_ohlc_fraction,
                p.eval.total_ms,
                p.eval.loader_ms,
                p.eval.forward_ms,
                p.eval.metrics_ms
            ]
            .iter()
            .all(|v| v.is_finite())
            && p.train_nll.is_none_or(f64::is_finite)
            && p.train_mse.is_none_or(f64::is_finite)
            && p.step_ms.is_none_or(|v| v.is_finite() && v >= 0.)
            && p.loader_wait_ms.is_none_or(|v| v.is_finite() && v >= 0.)
            && p.eval.reports_ms.is_none_or(|v| v.is_finite() && v >= 0.)
            && p.peak_allocator_mib.is_none_or(|v| v.is_finite() && v > 0.)
            && (0.0..=1.0).contains(&p.invalid_ohlc_fraction)
            && (0.0..=1.0).contains(&p.tail_loss_share)
            && (0.0..=1.0).contains(&p.within_1_sigma)
            && (0.0..=1.0).contains(&p.within_2_sigma)
            && (p.median_window_ratio.is_nan() || p.median_window_ratio >= 0.)),
        "invalid segment report metrics"
    );
    let series = |label: &str, value: fn(&Metrics) -> f64| ReportSeries {
        label: label.to_owned(),
        values: points.iter().map(|p| value(p) as f32).collect(),
    };
    // Scope facts every title carries, so a reader never has to guess how much data is
    // behind a curve. One scored origin is one forecast window, so the held-out sample's
    // origin count is also its window count.
    let origins = |full: bool| {
        points
            .iter()
            .rev()
            .find(|p| p.validation_is_full == full)
            .map(|p| p.validation_origins)
    };
    let mut facts = Vec::new();
    if let Some(count) = origins(false) {
        facts.push(split_scope(SAMPLE, count));
    }
    if let Some(count) = origins(true) {
        facts.push(split_scope(FULL, count));
    }
    facts.push(format!("{} tickers", last.tickers));
    let scope = facts.join(", ");
    let chart = |base: &str,
                 question: &str,
                 y_label: &str,
                 scale: ScaleKind,
                 scope: &str,
                 series: Vec<ReportSeries>|
     -> Result<()> {
        let series: Vec<_> = series
            .into_iter()
            .filter(|s| s.values.iter().any(|v| v.is_finite()))
            .collect();
        if series.is_empty() {
            return Ok(());
        }
        write_report(
            output.join(format!("{base}.report.bin")),
            &Report {
                title: format!(
                    "CausalPatch epoch {} step {} | {question} | {scope}",
                    last.epoch, last.step
                ),
                x_label: Some("optimizer step".to_owned()),
                y_label: Some(y_label.to_owned()),
                scale,
                kind: ReportKind::IndexedLines {
                    steps: points.iter().map(|p| p.step as u64).collect(),
                    series,
                },
            },
        )?;
        Ok(())
    };
    let split = |full: bool, value: fn(&Metrics) -> f64| {
        move |p: &Metrics| {
            if p.validation_is_full == full {
                value(p)
            } else {
                f64::NAN
            }
        }
    };
    let scored = |label: &str, full: bool, value: fn(&Metrics) -> f64| ReportSeries {
        label: label.to_owned(),
        values: points.iter().map(|p| split(full, value)(p) as f32).collect(),
    };
    // The headline. Ratios only, so every curve shares one dimensionless axis and the
    // parity line is the whole reading rule: below it the forecast beat persistence.
    chart(
        "timexer_segment_skill",
        "does the forecast beat persistence?",
        RATIO_UNIT,
        ScaleKind::Linear,
        &scope,
        vec![
            scored(&format!("{SAMPLE} {NEUTRAL} MSE ratio"), false, |p| {
                p.validation_mse / p.persistence_mse
            }),
            scored(&format!("{SAMPLE} {RAW} MSE ratio"), false, |p| {
                p.absolute_mse / p.absolute_persistence_mse
            }),
            scored(
                &format!("{SAMPLE} {NEUTRAL} median-window MSE ratio"),
                false,
                |p| p.median_window_ratio,
            ),
            scored(&format!("{FULL} {NEUTRAL} MSE ratio"), true, |p| {
                p.validation_mse / p.persistence_mse
            }),
            scored(&format!("{FULL} {RAW} MSE ratio"), true, |p| {
                p.absolute_mse / p.absolute_persistence_mse
            }),
            scored(
                &format!("{FULL} {NEUTRAL} median-window MSE ratio"),
                true,
                |p| p.median_window_ratio,
            ),
            series("parity 1.0", |_| 1.),
        ],
    )?;
    chart(
        "timexer_segment_loss",
        "how good is the predictive density against the persistence prior?",
        NATS_UNIT,
        ScaleKind::Linear,
        &scope,
        vec![
            series("training NLL", |p| p.train_nll.unwrap_or(f64::NAN)),
            scored(&format!("{SAMPLE} NLL"), false, |p| p.validation_nll),
            scored(&format!("{SAMPLE} persistence NLL"), false, |p| {
                p.persistence_nll
            }),
            scored(&format!("{FULL} NLL"), true, |p| p.validation_nll),
            scored(&format!("{FULL} persistence NLL"), true, |p| {
                p.persistence_nll
            }),
        ],
    )?;
    // Levels, not ratios: one unit, log scale. Market-neutral and raw sit on the same axis
    // because they differ by a multiplicative factor, which a log axis renders as an offset;
    // splitting them would hide exactly the comparison the two spaces exist to support.
    chart(
        "timexer_segment_error",
        "how large is the squared error, forecast against persistence?",
        LEVEL_UNIT,
        ScaleKind::Symlog,
        &format!(
            "{scope}; latest held-out price RMSE {:.4}, MAE {:.4} USD",
            last.rmse_price, last.mae_price
        ),
        vec![
            series("training MSE", |p| p.train_mse.unwrap_or(f64::NAN)),
            scored(&format!("{SAMPLE} {NEUTRAL} MSE"), false, |p| {
                p.validation_mse
            }),
            scored(&format!("{SAMPLE} {NEUTRAL} persistence MSE"), false, |p| {
                p.persistence_mse
            }),
            scored(&format!("{SAMPLE} {RAW} MSE"), false, |p| p.absolute_mse),
            scored(&format!("{SAMPLE} {RAW} persistence MSE"), false, |p| {
                p.absolute_persistence_mse
            }),
            scored(&format!("{FULL} {NEUTRAL} MSE"), true, |p| p.validation_mse),
            scored(&format!("{FULL} {NEUTRAL} persistence MSE"), true, |p| {
                p.persistence_mse
            }),
            scored(&format!("{FULL} {RAW} MSE"), true, |p| p.absolute_mse),
            scored(&format!("{FULL} {RAW} persistence MSE"), true, |p| {
                p.absolute_persistence_mse
            }),
        ],
    )?;
    chart(
        "timexer_segment_calibration",
        "do the predicted σ bands cover the realized targets?",
        FRACTION_UNIT,
        ScaleKind::Linear,
        &scope,
        vec![
            scored(&format!("{SAMPLE} within 1σ"), false, |p| p.within_1_sigma),
            scored(&format!("{SAMPLE} within 1.96σ"), false, |p| {
                p.within_2_sigma
            }),
            scored(&format!("{FULL} within 1σ"), true, |p| p.within_1_sigma),
            scored(&format!("{FULL} within 1.96σ"), true, |p| p.within_2_sigma),
            series("nominal 1σ = 0.683", |_| 0.6827),
            series("nominal 1.96σ = 0.950", |_| 0.95),
        ],
    )?;
    chart(
        "timexer_segment_timing",
        "where does the wall clock go?",
        "milliseconds",
        ScaleKind::Linear,
        &last.peak_allocator_mib.map_or_else(
            || scope.clone(),
            |peak| format!("{scope}; peak allocator {peak:.0} MiB"),
        ),
        vec![
            series("training step (interval mean)", |p| {
                p.step_ms.unwrap_or(f64::NAN)
            }),
            series(
                "training step host loader wait (interval mean; overlaps GPU work)",
                |p| p.loader_wait_ms.unwrap_or(f64::NAN),
            ),
            series("training step phase: host batch", |p| {
                p.step_phases.map_or(f64::NAN, |x| x.host_batch_ms)
            }),
            series("training step phase: H2D", |p| {
                p.step_phases.map_or(f64::NAN, |x| x.h2d_ms)
            }),
            series("training step phase: forward backbone", |p| {
                p.step_phases.map_or(f64::NAN, |x| x.forward_backbone_ms)
            }),
            series("training step phase: forward head and loss", |p| {
                p.step_phases.map_or(f64::NAN, |x| x.forward_head_ms)
            }),
            series("training step phase: backward", |p| {
                p.step_phases.map_or(f64::NAN, |x| x.backward_ms)
            }),
            series(
                "training step phase: captured forward+backward replay",
                |p| p.step_phases.map_or(f64::NAN, |x| x.captured_replay_ms),
            ),
            series("training step phase: optimizer", |p| {
                p.step_phases.map_or(f64::NAN, |x| x.optimizer_ms)
            }),
            series("held-out evaluation total", |p| p.eval.total_ms),
            series(
                "held-out evaluation: host batch wait (summed over batches)",
                |p| p.eval.loader_ms,
            ),
            series(
                "held-out evaluation phase: H2D and forward (one sampled batch)",
                |p| p.eval.forward_ms,
            ),
            series(
                "held-out evaluation phase: metric accumulation (one sampled batch)",
                |p| p.eval.metrics_ms,
            ),
            series(
                "after held-out evaluation: candle windows and checkpoints",
                |p| p.eval.reports_ms.unwrap_or(f64::NAN),
            ),
        ],
    )?;
    if let Some(budget) = last.capture_budget {
        chart(
            "timexer_segment_capture",
            "what does capturing the training step reserve, against what the device has?",
            "MiB",
            ScaleKind::Linear,
            &format!(
                "{scope}; captured after {} warmup steps",
                budget.warmup_steps
            ),
            vec![
                series("device total", |p| {
                    p.capture_budget.map_or(f64::NAN, |b| b.device_total_mib)
                }),
                series("captured step private mempool", |p| {
                    p.capture_budget
                        .map_or(f64::NAN, |b| b.pool_reservation_mib)
                }),
                series("allocator reserved before warmup", |p| {
                    p.capture_budget
                        .map_or(f64::NAN, |b| b.before_warmup.reserved_mib)
                }),
                series("allocator reserved after warmup", |p| {
                    p.capture_budget
                        .map_or(f64::NAN, |b| b.after_warmup.reserved_mib)
                }),
                series("allocator reserved after empty_cache", |p| {
                    p.capture_budget
                        .map_or(f64::NAN, |b| b.after_empty_cache.reserved_mib)
                }),
                series("allocator reserved at capture end", |p| {
                    p.capture_budget
                        .map_or(f64::NAN, |b| b.at_capture_end.reserved_mib)
                }),
                series("allocator live at capture start", |p| {
                    p.capture_budget
                        .map_or(f64::NAN, |b| b.at_capture_start.allocated_mib)
                }),
                series("eager step peak allocator", |p| {
                    p.capture_budget
                        .map_or(f64::NAN, |b| b.after_warmup.peak_allocated_mib)
                }),
                series("held-out evaluation peak allocator", |p| {
                    p.evaluation_peak_mib.unwrap_or(f64::NAN)
                }),
            ],
        )?;
    }
    chart(
        "timexer_segment_progress",
        "how much of the corpus is consumed, and how clean is the held-out signal?",
        SHARE_UNIT,
        ScaleKind::Linear,
        &format!(
            "{} / {} unique training target bars, {} tickers",
            last.completed_target_bars, last.total_target_bars, last.tickers
        ),
        vec![
            series("training target bars completed", |p| {
                p.completed_target_bars as f64 / p.total_target_bars.max(1) as f64
            }),
            series("held-out invalid forecast candles per forecast bar", |p| {
                p.invalid_ohlc_fraction
            }),
            series("held-out top-1% |target| share of squared error", |p| {
                p.tail_loss_share
            }),
        ],
    )
}

/// One evaluated split's per-horizon curves together with the scope fact its titles need.
#[derive(Debug, Clone, Copy)]
pub struct HorizonSplit<'a> {
    pub curve: &'a HorizonCurve,
    /// Origins scored, which is also the number of forecast windows behind every point:
    /// scoring uses one final origin per row.
    pub origins: usize,
}

/// The per-horizon family, x = bars ahead. Four bases rather than one because the curves
/// answer four questions in three incompatible units, and a single axis carrying a
/// dimensionless ratio near 1, a σ-scaled level spanning orders of magnitude and a rate near
/// 0.5 renders two of the three as flat lines:
///
/// - `timexer_segment_horizon`: forecast/persistence MSE per split per space, plus parity.
/// - `timexer_segment_horizon_error`: the σ-scaled levels those ratios divide, log scale.
/// - `timexer_segment_horizon_robust`: MAE ratio and tail-trimmed MSE ratio, plus parity.
/// - `timexer_segment_horizon_rates`: close win rate, directional hit rate, predicted-up
///   share, plus parity — rates the top-1% |target| tail cannot dominate the way MSE is.
pub fn write_horizon(
    output: &Path,
    epoch: usize,
    step: usize,
    sample: Option<HorizonSplit<'_>>,
    full: Option<HorizonSplit<'_>>,
    tickers: usize,
) -> Result<()> {
    let horizon = sample.or(full).map_or(0, |split| split.curve.mse.len());
    ensure!(
        horizon > 0 && tickers > 0,
        "per-horizon reports require one evaluated curve over a nonempty ticker universe"
    );
    let mut facts = Vec::new();
    let mut ratios = Vec::new();
    let mut levels = Vec::new();
    let mut robust = Vec::new();
    let mut rates = Vec::new();
    for (split, evaluated) in [(SAMPLE, sample), (FULL, full)] {
        let Some(HorizonSplit { curve, origins }) = evaluated else {
            continue;
        };
        ensure!(origins > 0, "an evaluated split must score an origin");
        ensure!(
            [
                &curve.mse,
                &curve.persistence_mse,
                &curve.absolute_mse,
                &curve.absolute_persistence_mse
            ]
            .iter()
            .all(|values| values.len() == horizon && values.iter().all(|v| v.is_finite() && *v >= 0.)),
            "invalid per-horizon metrics"
        );
        let rate_curves = [&curve.win_rate, &curve.hit_rate, &curve.up_fraction];
        let ratio_curves = [&curve.mae_ratio, &curve.trimmed_mse_ratio];
        ensure!(
            rate_curves
                .iter()
                .chain(&ratio_curves)
                .all(|values| values.len() == horizon)
                && rate_curves
                    .iter()
                    .flat_map(|values| values.iter())
                    .all(|v| v.is_nan() || (0.0..=1.0).contains(v))
                && ratio_curves
                    .iter()
                    .flat_map(|values| values.iter())
                    .all(|v| v.is_finite() && *v >= 0.),
            "invalid per-horizon robustness metrics"
        );
        facts.push(split_scope(split, origins));
        let curve_series = |label: String, values: &[f64]| ReportSeries {
            label,
            values: values.iter().map(|v| *v as f32).collect(),
        };
        for (space, mse, persistence) in [
            (NEUTRAL, &curve.mse, &curve.persistence_mse),
            (RAW, &curve.absolute_mse, &curve.absolute_persistence_mse),
        ] {
            ratios.push(ReportSeries {
                label: format!("{split} {space} MSE ratio"),
                values: mse
                    .iter()
                    .zip(persistence)
                    .map(|(mse, persistence)| (mse / persistence) as f32)
                    .collect(),
            });
            levels.push(curve_series(format!("{split} {space} MSE"), mse));
            levels.push(curve_series(
                format!("{split} {space} persistence MSE"),
                persistence,
            ));
        }
        robust.push(curve_series(
            format!("{split} {NEUTRAL} MAE ratio"),
            &curve.mae_ratio,
        ));
        robust.push(curve_series(
            format!("{split} {NEUTRAL} trimmed MSE ratio (top-1% |close| bars dropped)"),
            &curve.trimmed_mse_ratio,
        ));
        rates.push(curve_series(
            format!("{split} close win rate vs persistence"),
            &curve.win_rate,
        ));
        rates.push(curve_series(
            format!("{split} close directional hit rate"),
            &curve.hit_rate,
        ));
        rates.push(curve_series(
            format!("{split} close predicted-up share"),
            &curve.up_fraction,
        ));
    }
    facts.push(format!("{tickers} tickers"));
    let scope = facts.join(", ");
    let parity = |value: f32| ReportSeries {
        label: format!("parity {value:.1}"),
        values: vec![value; horizon],
    };
    ratios.push(parity(1.));
    robust.push(parity(1.));
    rates.push(parity(0.5));
    for (base, question, y_label, scale, series) in [
        (
            "timexer_segment_horizon",
            "does the forecast beat persistence at each horizon?",
            RATIO_UNIT,
            ScaleKind::Linear,
            ratios,
        ),
        (
            "timexer_segment_horizon_error",
            "how large is the squared error at each horizon, forecast against persistence?",
            LEVEL_UNIT,
            ScaleKind::Symlog,
            levels,
        ),
        (
            "timexer_segment_horizon_robust",
            "does the forecast still beat persistence once the |target| tail cannot dominate?",
            RATIO_UNIT,
            ScaleKind::Linear,
            robust,
        ),
        (
            "timexer_segment_horizon_rates",
            "how often is the close forecast on the right side?",
            RATE_UNIT,
            ScaleKind::Linear,
            rates,
        ),
    ] {
        write_report(
            output.join(format!("{base}.report.bin")),
            &Report {
                title: format!("CausalPatch epoch {epoch} step {step} | {question} | {scope}"),
                x_label: Some("bars ahead".to_owned()),
                y_label: Some(y_label.to_owned()),
                scale,
                kind: ReportKind::IndexedLines {
                    steps: (1..=horizon as u64).collect(),
                    series,
                },
            },
        )?;
    }
    Ok(())
}

/// One evaluated split's trading diagnostics with the scope facts its titles need.
#[derive(Debug, Clone, Copy)]
pub struct TradingSplit<'a> {
    pub curve: &'a TradingCurve,
    pub portfolio: &'a PortfolioCurve,
    /// Origins scored, which is also the number of forecast windows behind every point.
    pub origins: usize,
}

/// A trading series. Non-finite values become NaN rather than an error: the one-bar-delay
/// diagnostic is genuinely undefined at `h = 1` (the delayed move is identically zero), and a
/// gap in a line is the honest rendering of an undefined statistic.
fn trading_series(label: String, values: &[f64], horizon: usize) -> Result<ReportSeries> {
    ensure!(
        values.len() == horizon,
        "{label}: {} points for a {horizon}-bar horizon",
        values.len()
    );
    Ok(ReportSeries {
        label,
        values: values
            .iter()
            .map(|v| if v.is_finite() { *v as f32 } else { f32::NAN })
            .collect(),
    })
}

/// The trading family, x = bars ahead. Seven bases because the questions have seven units;
/// see the unit constants above.
///
/// - `timexer_segment_decomposition`: where the MSE gain comes from. The total sits beside the
///   demeaned-forecast component, which *is* the `1 - ρ²` a genuine conditional mean implies,
///   so the offset-versus-signal split is one glance.
/// - `timexer_segment_signal`: the information-coefficient family on the demeaned forecast,
///   with the cross-sectional IC's ±1 standard-error band so significance is visible.
/// - `timexer_segment_offset`: the σ-scaled levels — the constant tilt, the realized mean, and
///   the decile-conditioned realized returns a strategy would actually collect.
/// - `timexer_segment_tradable` / `_tradable_rates`: the microstructure test. If the h = 1 edge
///   is bid-ask bounce, the mid-anchored and one-bar-delayed ratios sit at 1.0 while the
///   close-anchored ratio does not.
/// - `timexer_segment_portfolio` / `_portfolio_sharpe`: the cost sweep.
pub fn write_trading(
    output: &Path,
    epoch: usize,
    step: usize,
    sample: Option<TradingSplit<'_>>,
    full: Option<TradingSplit<'_>>,
    tickers: usize,
) -> Result<()> {
    let horizon = sample.or(full).map_or(0, |split| split.curve.total_gain.len());
    ensure!(
        horizon > 0 && tickers > 0,
        "trading reports require one evaluated curve over a nonempty ticker universe"
    );
    let mut facts = Vec::new();
    let mut gains = Vec::new();
    let mut signal = Vec::new();
    let mut levels = Vec::new();
    let mut anchors = Vec::new();
    let mut anchor_rates = Vec::new();
    let mut bps = Vec::new();
    let mut sharpe = Vec::new();
    let mut portfolio_steps: Vec<u64> = Vec::new();
    for (split, evaluated) in [(SAMPLE, sample), (FULL, full)] {
        let Some(TradingSplit {
            curve,
            portfolio,
            origins,
        }) = evaluated
        else {
            continue;
        };
        ensure!(origins > 0, "an evaluated split must score an origin");
        facts.push(format!(
            "{}, {} evaluation timestamps",
            split_scope(split, origins),
            curve.cross_sections
        ));
        for (label, values) in [
            (format!("{split} {NEUTRAL} close total gain"), &curve.total_gain),
            (
                format!("{split} {NEUTRAL} all-channel total gain"),
                &curve.all_channel_gain,
            ),
            (
                format!("{split} {NEUTRAL} close gain from the unconditional offset"),
                &curve.offset_gain,
            ),
            (
                format!("{split} {NEUTRAL} close gain from the demeaned forecast (= implied 1 - ρ²)"),
                &curve.demeaned_gain,
            ),
            (
                format!("{split} {NEUTRAL} close gain lost to forecast mis-scaling (cross term)"),
                &curve.scaling_gain,
            ),
        ] {
            gains.push(trading_series(label, values, horizon)?);
        }
        for (label, values) in [
            (
                format!("{split} close Pearson IC (demeaned forecast, pooled)"),
                &curve.pearson,
            ),
            (
                format!("{split} close Spearman rank IC (pooled)"),
                &curve.spearman,
            ),
            (
                format!("{split} close cross-sectional IC (mean over timestamps)"),
                &curve.cross_sectional_ic,
            ),
        ] {
            signal.push(trading_series(label, values, horizon)?);
        }
        for (sign, edge) in [(1., "+1 s.e."), (-1., "-1 s.e.")] {
            let band: Vec<f64> = curve
                .cross_sectional_ic
                .iter()
                .zip(&curve.cross_sectional_ic_se)
                .map(|(ic, se)| ic + sign * se)
                .collect();
            signal.push(trading_series(
                format!("{split} close cross-sectional IC {edge}"),
                &band,
                horizon,
            )?);
        }
        for (label, values) in [
            (
                format!("{split} mean predicted close coordinate (the constant tilt)"),
                &curve.mean_forecast,
            ),
            (
                format!("{split} mean realized close coordinate"),
                &curve.mean_target,
            ),
            (
                format!("{split} top-decile |signal| realized return on the predicted side"),
                &curve.top_decile_return,
            ),
            (
                format!("{split} bottom-decile |signal| realized return on the predicted side"),
                &curve.bottom_decile_return,
            ),
            (
                format!("{split} conviction spread (top minus bottom decile)"),
                &curve.conviction_spread_return,
            ),
        ] {
            levels.push(trading_series(label, values, horizon)?);
        }
        for (label, values) in [
            (
                format!("{split} {NEUTRAL} close MSE ratio vs close-anchored persistence"),
                &curve.close_mse_ratio,
            ),
            (
                format!("{split} {NEUTRAL} close MSE ratio vs high/low-mid-anchored persistence"),
                &curve.mid_anchor_mse_ratio,
            ),
            (
                format!("{split} {NEUTRAL} close MSE ratio with a one-bar execution delay"),
                &curve.delayed_mse_ratio,
            ),
        ] {
            anchors.push(trading_series(label, values, horizon)?);
        }
        for (label, values) in [
            (
                format!("{split} close hit rate, close anchor"),
                &curve.close_hit_rate,
            ),
            (
                format!("{split} close hit rate, high/low-mid anchor"),
                &curve.mid_anchor_hit_rate,
            ),
            (
                format!("{split} close hit rate, one-bar execution delay"),
                &curve.delayed_hit_rate,
            ),
            (
                format!("{split} close hit rate, top decile of |signal|"),
                &curve.top_decile_hit_rate,
            ),
            (
                format!("{split} close hit rate, bottom decile of |signal|"),
                &curve.bottom_decile_hit_rate,
            ),
        ] {
            anchor_rates.push(trading_series(label, values, horizon)?);
        }
        let points = portfolio.horizons.len();
        ensure!(
            points > 0
                && portfolio.costs_bps.len() == portfolio.net_bps.len()
                && portfolio.costs_bps.len() == portfolio.net_sharpe.len()
                && portfolio.cross_sections.len() == points
                && portfolio
                    .net_bps
                    .iter()
                    .chain(&portfolio.net_sharpe)
                    .all(|row| row.len() == points),
            "portfolio curves must be rectangular over the reported horizons"
        );
        if portfolio_steps.is_empty() {
            portfolio_steps = portfolio.horizons.clone();
        }
        ensure!(
            portfolio_steps == portfolio.horizons,
            "both splits must report the same portfolio holding periods"
        );
        for (cost, (net, annualized)) in portfolio
            .costs_bps
            .iter()
            .zip(portfolio.net_bps.iter().zip(&portfolio.net_sharpe))
        {
            bps.push(trading_series(
                format!("{split} decile-spread return net of {cost:.0} bps per side"),
                net,
                points,
            )?);
            sharpe.push(trading_series(
                format!("{split} decile-spread Sharpe net of {cost:.0} bps per side"),
                annualized,
                points,
            )?);
        }
    }
    facts.push(format!("{tickers} tickers"));
    let scope = facts.join(", ");
    let line = |label: &str, value: f32, points: usize| ReportSeries {
        label: label.to_owned(),
        values: vec![value; points],
    };
    gains.push(line("zero gain", 0., horizon));
    signal.push(line("zero correlation", 0., horizon));
    levels.push(line("zero", 0., horizon));
    anchors.push(line("parity 1.0", 1., horizon));
    anchor_rates.push(line("parity 0.5", 0.5, horizon));
    let points = portfolio_steps.len();
    bps.push(line("zero", 0., points));
    sharpe.push(line("zero", 0., points));
    for (base, question, y_label, steps, series) in [
        (
            "timexer_segment_decomposition",
            "where does the MSE gain come from: a constant tilt or a conditional signal?",
            GAIN_UNIT,
            None,
            gains,
        ),
        (
            "timexer_segment_signal",
            "how much information does the demeaned close forecast carry?",
            CORRELATION_UNIT,
            None,
            signal,
        ),
        (
            "timexer_segment_offset",
            "how big is the constant tilt, and what does conviction actually pay?",
            COORDINATE_UNIT,
            None,
            levels,
        ),
        (
            "timexer_segment_tradable",
            "does the edge survive a mid anchor and a one-bar execution delay?",
            ANCHOR_RATIO_UNIT,
            None,
            anchors,
        ),
        (
            "timexer_segment_tradable_rates",
            "is the sign right once the anchor is the mid or execution is delayed?",
            ANCHOR_RATE_UNIT,
            None,
            anchor_rates,
        ),
        (
            "timexer_segment_portfolio",
            "what does a decile long/short earn per holding period after costs?",
            BPS_UNIT,
            Some(portfolio_steps.clone()),
            bps,
        ),
        (
            "timexer_segment_portfolio_sharpe",
            "what Sharpe does a decile long/short reach after costs?",
            SHARPE_UNIT,
            Some(portfolio_steps.clone()),
            sharpe,
        ),
    ] {
        write_report(
            output.join(format!("{base}.report.bin")),
            &Report {
                title: format!("CausalPatch epoch {epoch} step {step} | {question} | {scope}"),
                x_label: Some("bars ahead".to_owned()),
                y_label: Some(y_label.to_owned()),
                scale: ScaleKind::Linear,
                kind: ReportKind::IndexedLines {
                    steps: steps.unwrap_or_else(|| (1..=horizon as u64).collect()),
                    series,
                },
            },
        )?;
    }
    Ok(())
}

pub fn write_candles(
    output: &Path,
    epoch: usize,
    step: usize,
    windows: &[CandleWindow],
) -> Result<()> {
    for (index, window) in windows.iter().enumerate() {
        ensure!(
            window.origin < window.actual.len() && !window.predicted.is_empty(),
            "invalid candle window"
        );
        ensure!(
            window.actual.iter().all(CandleBar::is_valid_ohlc),
            "invalid observed OHLC in candle window"
        );
        let clock = chrono::DateTime::from_timestamp_millis(window.timestamp)
            .context("invalid candle origin timestamp")?
            .with_timezone(&chrono_tz::America::New_York);
        let ticker = &window.ticker;
        ensure!(!ticker.is_empty(), "candle window requires ticker identity");
        // The five facts a reader needs, in the order they are needed: which population the
        // window came from, which instrument, when the forecast was made, how far it reaches,
        // and what the drawn candles actually are. Candle windows are always drawn from the
        // held-out validation population, so the split is fixed rather than a parameter.
        let report = Report {
            title: format!(
                "{FULL} | {ticker} | {} | {} bars | conditional-mean path re-based to origin close | epoch {epoch} step {step}",
                clock.format("%Y-%m-%d %H:%M %Z"),
                window.predicted.len()
            ),
            x_label: Some("five-minute bars from forecast origin".to_owned()),
            y_label: Some("price (USD)".to_owned()), scale: ScaleKind::Linear,
            kind: ReportKind::CandleSegment {
                actual: window.actual.clone(), origin: window.origin, predicted: window.predicted.clone(),
            },
        };
        if index == 0 {
            write_report(output.join("timexer_segment_candles.report.bin"), &report)?;
        }
        write_report(
            output.join("candle_snapshots").join(format!(
                "step{step}_epoch{epoch:03}_window{:02}_fan.report.bin",
                index + 1
            )),
            &report,
        )?;
    }
    Ok(())
}

pub fn write_corpus(output: &Path, contract: &CorpusContract, market: &MarketSummary) -> Result<()> {
    let path = output.join("timexer_segment_progress.report.bin");
    let mut report = if path.exists() {
        shared::report::read_report(&path)?
    } else {
        Report {
            title: "CausalPatch corpus ready".into(),
            x_label: Some("optimizer step".into()),
            y_label: Some(SHARE_UNIT.into()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: vec![0],
                series: vec![ReportSeries {
                    label: "training target bars completed".into(),
                    values: vec![0.],
                }],
            },
        }
    };
    let mut reasons = BTreeMap::new();
    for ticker in &contract.excluded_tickers {
        *reasons.entry(ticker.reason.as_str()).or_insert(0usize) += 1;
    }
    let reasons = reasons
        .into_iter()
        .map(|(reason, count)| format!("{reason}: {count}"))
        .collect::<Vec<_>>()
        .join(", ");
    let source_bars: usize = contract
        .tickers
        .iter()
        .map(|ticker| ticker.source_bars)
        .sum();
    let invalid_bars: usize = contract
        .tickers
        .iter()
        .map(|ticker| ticker.invalid_ohlc_indices.len())
        .sum();
    report.title = format!("{} | corpus: {} tickers, {source_bars} source bars, {invalid_bars} malformed source bars omitted; {} training / {} held-out targets, {} unused held-out remainder; {} excluded ({reasons}); market: {} steps over >= {} tickers, contributors min {} median {}, step std {:.5}, largest step {:+.5} at {}",
        report.title.split(" | corpus:").next().unwrap_or(&report.title), contract.tickers.len(), contract.train_target_bars, contract.validation_target_bars,
        contract.validation_remainder_bars, contract.excluded_tickers.len(), market.steps, contract.market_min_cross_section,
        market.min_contributors, market.median_contributors, market.step_std, market.max_abs_step, market.max_abs_step_ts);
    write_report(path, &report)?;
    Ok(())
}
