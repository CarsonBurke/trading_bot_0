use std::{collections::BTreeMap, path::Path};

use super::{
    calibration::{FrozenGain, MeanCalibration, Moments, CLOSE_CHANNEL},
    compute::{CaptureBudget, LrSchedule, LrTrajectory},
    corpus::CorpusContract,
    features::MarketSummary,
    probe::{self, HorizonScore, ProbeReport},
};

use anyhow::{bail, ensure, Context, Result};
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
pub(super) const SAMPLE: &str = "held-out sample";
pub(super) const FULL: &str = "held-out full";
/// The third held-out draw, and the only one a cross-sectional statistic can be computed on.
/// [`SAMPLE`] is a strided pick over a ticker-major origin list, so each of its origins sits
/// on its own timestamp and every within-timestamp quantity is undefined; this draw is whole
/// timestamp blocks of 256 tickers, fixed across steps and runs. It is NOT a subset of the
/// sample and its NLL is not comparable to the sample's, which is why it carries its own
/// name in every label rather than being folded into either existing split.
const CROSS: &str = "held-out cross-section";
/// The IN-PERIOD draw's label: origins inside the TRAINING chronological span that no
/// surviving training row supervises. Named "in-period" and not "in-sample", because it is
/// emphatically NOT in-sample - it shares no origin and no target bar with any training row.
/// The only thing it shares with training is the market period, which is exactly the variable
/// under test.
const IN_PERIOD: &str = "in-period held-out";
/// The out-of-period comparand, which is the existing `held-out cross-section` draw renamed
/// for this one family so a reader is not asked to remember that "held-out cross-section" and
/// "in-period held-out" differ in period. Same draw, same rule, same width cap.
const OUT_OF_PERIOD: &str = "out-of-period held-out";
const NEUTRAL: &str = "market-neutral";
const RAW: &str = "raw";
/// The reserved `[70%, 80%)` partition the amplitude calibration is fitted on. A fourth
/// population, named rather than folded into a held-out word: it is out of sample for the
/// WEIGHTS and in sample for the GAIN, so a panel that called it held-out would claim an
/// out-of-sample amplitude result that only the [`SAMPLE`] and [`FULL`] draws can make.
const CALIBRATION: &str = "calibration partition";
/// The in-sample draw, spelled here because it is a split word and not a quantity.
const TRAINING: &str = "training";

/// Axis units. Each carries the unit and, where one exists, the reading rule, so a panel is
/// interpretable without its documentation.
const RATIO_UNIT: &str = "ratio vs persistence (dimensionless; < 1 = skill)";
const LEVEL_UNIT: &str = "σ-scaled squared log-return";
const NATS_UNIT: &str = "nats per bar";
const FRACTION_UNIT: &str = "fraction of valid target bars";
const RATE_UNIT: &str =
    "fraction of valid target bars (win and hit rates > 0.5 = skill; up share is bias)";

/// The train-versus-held-out NLL gap, step-indexed. It states its reading rule because the
/// informative event is a MOVEMENT, not a level: the gap shrinking toward 0 while the held-out
/// NLL rises is the signature of fitting the training period.
const GAP_UNIT: &str = "nats per bar (training objective NLL minus held-out objective NLL; \
                        same horizon weighting, different origin populations)";

/// The units of the step-indexed per-horizon family - the transpose that answers "when did
/// h = 64 turn", which no horizon-indexed base can answer because each evaluation overwrites
/// the previous curve.
///
/// [`IC_UNIT`] carries the decision metric. It is deliberately NOT on the same axis as the
/// pooled Pearson: the pooled statistic is finite on every draw while the within-timestamp IC
/// is structurally undefined on a draw whose origins each sit on their own timestamp, and a
/// reader who mistook one for the other would read "no cross-sectional signal" off a panel
/// that never measured any.
const IC_UNIT: &str = "correlation coefficient (dimensionless; mean over evaluation timestamps \
                       of the within-timestamp cross-ticker correlation; 0 = no information, \
                       ±1 = perfect)";
const IC_ERROR_UNIT: &str = "correlation coefficient (dimensionless; standard error of the \
                             timestamp mean under an iid-timestamp approximation, NOT a \
                             confidence interval; an IC beyond ±2 of these is not noise)";
const POOLED_UNIT: &str = "correlation coefficient (dimensionless; ONE correlation over all \
                           (window, horizon) bars pooled, so it is not a cross-sectional \
                           statistic; 0 = no information, ±1 = perfect)";
/// Counts, so a thin or empty draw is legible as thin instead of as a statistic that moved.
/// Symlog, because the two populations on it differ by three orders of magnitude.
const POPULATION_UNIT: &str = "count (each series names its own population; NaN = the split \
                               was not scored at that step, 0 = it was scored and nothing \
                               qualified)";
/// The three-part identity is exact on the CLOSE channel only. The headline
/// `timexer_segment_horizon_steps` ratio averages four channels, so these components do not
/// reconstruct it and the unit says so rather than leaving the arithmetic to a reader.
const DECOMPOSITION_UNIT: &str = "share of the close-channel persistence MSE (dimensionless; \
                                  offset + demeaned + cross term = the close total gain, NOT \
                                  the four-channel headline; > 0 = gain, cross term ≤ 0 = what \
                                  wrong amplitude costs)";
/// The direct amplitude diagnostic. Emitted from the moments themselves rather than inverted
/// out of the gain decomposition: `-(β̂-1)²·Var(g)/P` and `Cov²/(Var(g)·P)` admit TWO positive
/// β̂ branches whenever the cross term is smaller in magnitude than the demeaned gain, which is
/// the ordinary case, so a chart that inverted them would silently pick one.
const OPTIMAL_GAIN_UNIT: &str = "gain on the demeaned forecast (dimensionless; 1 = perfect \
                                 amplitude calibration, < 1 = forecast over-amplified, > 1 = \
                                 under-amplified; rank order and cross-sectional tradability \
                                 are invariant to it while MSE is quadratic in it)";
/// The moments `β̂` is a quotient of, normalized so they sit on one axis with what the
/// amplitude error costs. A gain far from 1 on a `Var(f)` share of `1e-5` is arithmetically
/// real and worth nothing, which is a different verdict from the same gain on a share of
/// `4.7e-2`, and only this panel distinguishes them.
const MOMENT_UNIT: &str = "share of the close-channel persistence MSE (dimensionless; Var(f) \
                           and Cov(f,y) divided by mean(y²); the amplitude cost is what \
                           leaving the gain at 1 forfeits, so a horizon whose cost is below \
                           the noise scale has nothing to calibrate)";
const BEST_SCALE_UNIT: &str = "ratio vs close-anchored persistence (dimensionless; < 1 = \
                               skill; the best-scale series is the SAME forecast with its \
                               amplitude corrected and nothing else changed, so both near 1 = \
                               no signal while achieved above 1 with best scale below it = \
                               signal at the wrong amplitude)";

/// Progress mixes a share of the training corpus with shares of held-out bars, so its axis
/// cannot claim one denominator; each series label names its own.
const SHARE_UNIT: &str = "fraction of the series' own population (dimensionless; 0 to 1)";

/// The recipe scalars are pure mixing weights - a fraction of a residual branch, of the
/// embedding, of an encoder layer's output or of the source layer's value - so they carry no
/// unit at all; only their distance from their init is readable.
const MIXING_UNIT: &str = "learned mixing coefficient (dimensionless)";

/// The realized learning rate is an absolute step size, not a multiplier on one: the whole
/// point of the panel is that six families under one global knob run at six different rates,
/// so a normalized axis would erase the quantity being reported. Zero is a real reading -
/// a parameter whose step is disabled moves at rate zero - and is distinguishable from a
/// floored schedule, which holds 0.15 of its peak and never reaches 0.
const LR_UNIT: &str =
    "learning rate (absolute; higher = larger parameter step; 0 = family frozen)";

/// Distinct units stay on separate axes; endpoint utility is not a backtest return stream.
const GAIN_UNIT: &str = "share of the persistence MSE (dimensionless; > 0 = gain, and the three \
                         components sum to the close total)";
const CORRELATION_UNIT: &str =
    "correlation coefficient (dimensionless; 0 = no information, ±1 = perfect)";
const COORDINATE_UNIT: &str = "σ-scaled mean log return (0 = persistence; the market-neutral \
                               close coordinate the model predicts)";
const ANCHOR_RATIO_UNIT: &str =
    "ratio vs the anchor's own persistence (dimensionless; < 1 = skill, 1 = no edge over the anchor)";
const ANCHOR_RATE_UNIT: &str =
    "fraction of scored bars where the sign is right (> 0.5 = skill; 0.5 = a coin flip)";
const BPS_UNIT: &str = "basis points of initial capital per endpoint cohort";
const RATE_UNIT_BPS: &str =
    "basis points of initial capital per observed bar (payoff / h; not calendar return)";
const BREAKEVEN_UNIT: &str =
    "signed bps per transacted notional (gross / turnover; negative = no edge; idle = undefined)";
const EXPOSURE_UNIT: &str = "notional / initial capital (gross budget <= 1)";
const ACTIVE_UNIT: &str = "fraction of origin-cohort names with nonzero positions";
const TURNOVER_UNIT: &str = "entry + exit transacted notional / initial capital";
const CENSUS_UNIT: &str =
    "count (complete eligible cohorts; mean/minimum observed valid width over all origin groups)";
const UTILITY_SCOPE: &str =
    "endpoint probes, not backtest; overlapping/unsynced exits; no borrow/impact; residual risk only";

/// The scope fact a split contributes to a title. Spelled out for the held-out sample because
/// its count is a fixed sample size, and a reader has to know it is a sample of the population
/// rather than the population.
fn split_scope(split: &str, origins: usize) -> String {
    if split == SAMPLE {
        format!("{split} {origins} fixed windows, one scored origin each")
    } else if split == CROSS {
        format!("{split} {origins} fixed windows in whole timestamp blocks")
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
    /// Held-out NLL with the training objective's horizon weighting.
    pub validation_objective_nll: f64,
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
    /// Mean per-step wall clock the training loop spent blocked in `Prefetcher::receive`.
    ///
    /// NOT a cost. Nothing in the training loop reads a device scalar per step, so the host
    /// runs ahead until the launch queue backs up and the loop period is `max(device work,
    /// host batch service time)`. This series is the part of the host's period that sat
    /// inside a blocking receive; it only becomes step time once it EXCEEDS `step_ms`.
    pub loader_wait_ms: Option<f64>,
    /// Mean wall clock the loader thread itself spent building one packed batch.
    ///
    /// The host-side service time, measured on the loader thread rather than inferred from
    /// the main thread's wait: this is the quantity that says how much host work a step
    /// costs, and the one that becomes the loop's ceiling if device work gets cheaper.
    pub loader_build_ms: Option<f64>,
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
    /// This evaluation's slice through the per-horizon curves at [`DECISION_HORIZONS`], the
    /// only carrier of per-horizon behaviour that survives across steps: the horizon-indexed
    /// bases are rewritten in place at every evaluation, so they answer "what does the curve
    /// look like NOW" and cannot answer "when did h = 64 turn". Built by [`horizon_track`];
    /// empty when the point has no scored curve.
    pub horizons: Vec<HorizonPoint>,
    /// The SAME slice through the `held-out cross-section` pass scored at this step, which is
    /// the only draw the within-timestamp IC is defined on at a report interval: the sample
    /// draw strides one origin per timestamp, so its cross-sections are structurally too thin,
    /// and the full draw is scored once per epoch. Without this the decision metric would have
    /// a trajectory only on epoch boundaries. Empty when the step scored no cross-section pass.
    pub cross_horizons: Vec<HorizonPoint>,
    /// Origins the cross-section pass scored; 0 when it was not run. Carried so the title can
    /// state that population instead of implying the sample's.
    pub cross_origins: usize,
    /// The SAME slice through the IN-PERIOD held-out pass: origins drawn from inside the
    /// TRAINING chronological span that no surviving training row supervises, scored at the
    /// same step and by the same draw rule as `held-out cross-section`.
    ///
    /// It exists to split what a chronological held-out split cannot: every `held-out *` draw
    /// is out-of-sample in origin identity AND out-of-period in market regime at once, and
    /// those two have opposite fixes. This one holds the period fixed. Empty on every run that
    /// did not reserve an in-period population, which is every run before the flag existed.
    pub in_period_horizons: Vec<HorizonPoint>,
    /// Origins the in-period pass scored; 0 when it was not run.
    pub in_period_origins: usize,
    /// Share of the unholed training rows this arm PURGED so that no surviving row could
    /// supervise an in-period target bar; 0.0 on an unholed arm.
    ///
    /// Carried on `Metrics` and folded into the scope line of EVERY chart `write_metrics`
    /// writes, `timexer_segment_loss` included, because the one way this experiment could
    /// produce a confident wrong conclusion is a reader step-matching a holed arm's TRAINING
    /// NLL against a control's without knowing it trained on less data. A fact that has to be
    /// remembered is a fact that will be forgotten; this one rides in the title.
    pub in_period_purged_row_share: f64,
    /// One-time cost of the corpus load, carried on every point because `write_metrics`'
    /// series accessors read a point and nothing else. Constant within a run, deliberately
    /// NOT validated as finite: [`super::cache::LoadTiming::default`] is all-NaN and a phase
    /// a warm cache never ran must stay NaN, which is exactly what makes a warm run legible
    /// beside a cold one.
    pub startup: super::cache::LoadTiming,
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
    /// Wall clock of the separate `held-out cross-section` pass the trading family is scored
    /// on. Charted so the price of a training-time breakeven trajectory is visible beside the
    /// interval it is paid out of, instead of being folded into the main evaluation's total.
    pub cross_section_ms: Option<f64>,
    /// Wall clock of the separate IN-PERIOD pass. Charted beside the cross-section pass for
    /// the same reason: the whole in-period experiment is only affordable if this number stays
    /// small against the interval it is paid out of, and a claim about that has to be a
    /// measured series rather than an estimate in a report.
    pub in_period_ms: Option<f64>,
}

/// Per-phase attribution of ONE sampled training step in milliseconds. Only the sampled step
/// is synchronized between phases, so ordinary steps keep their asynchronous launch pipeline
/// and the measurement never serializes training. `h2d_ms` is the copy of the packed row
/// block into the resident device batch, `forward_backbone_ms` the patch embedding through
/// the final norm, `forward_head_ms` the covariate projection, head GEMMs, candle geometry
/// and NLL, `backward_ms` the whole reverse pass, `captured_replay_ms` the whole forward and
/// backward as one CUDA-graph replay, and `optimizer_ms` the gradient reset plus update.
///
/// There is deliberately no host-batch phase here. The field that used to occupy that slot
/// was assigned the very same `loader.receive()` wall clock as `loader_wait_ms`, so the
/// timing chart carried one measurement twice under two names, one of which called it a
/// phase of the step it does not belong to.
///
/// A captured step cannot be split - one replay is one launch - so `captured_replay_ms` and
/// the three eager forward/backward phases are mutually exclusive: whichever does not apply
/// is `NaN`, never a zero a reader would take for "free".
#[derive(Debug, Clone, Copy, Default)]
pub struct StepPhases {
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
    /// Fraction of this horizon's valid targets inside the predicted ±1σ and ±1.96σ bands
    /// (nominal .6827/.9500). The aggregate [`Metrics::within_1_sigma`] cannot substitute:
    /// every horizon contributes the same bar count to it, so an over-dispersed scale at the
    /// long end is averaged against 191 easier horizons.
    pub within_1_sigma: Vec<f64>,
    pub within_2_sigma: Vec<f64>,
    /// Valid (bar, channel) elements behind this horizon's row - the population every ratio
    /// above divides by, carried so a thin horizon is legible as thin rather than as noisy.
    pub valid_elements: Vec<f64>,
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
///   Pearson `ρ` in [`Self::pearson`]. This is a gain, not the `1 - ρ²` MSE ratio;
///   target-mean effects enter through `var(y)/mean(y²)`.
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
    /// `β̂ = mean(y·g)/mean(g²) = ρ·σ_y/σ_f`, the gain that minimizes the MSE of the demeaned
    /// forecast. 1 is perfect amplitude calibration, below 1 an over-amplified forecast, above
    /// 1 an under-amplified one. NaN for a constant forecast, which has no amplitude to
    /// calibrate. This is the direct diagnostic for a rank-preserving MSE failure: rank order,
    /// and therefore cross-sectional tradability, is invariant to `β̂` while MSE is quadratic
    /// in it.
    pub optimal_gain: Vec<f64>,
    /// `Var(f)` and `Cov(f, y)` on the demeaned close coordinate, σ² units - the denominator
    /// and the numerator of [`Self::optimal_gain`], and [`Self::persistence`] `= mean(y²)` is
    /// what both are worth as a share of. Carried rather than left implicit because a `β̂`
    /// above 1 has two completely different readings - a real under-amplitude, or a forecast
    /// whose variance has collapsed toward zero - and only the ratio's two halves separate
    /// them.
    pub forecast_variance: Vec<f64>,
    pub covariance: Vec<f64>,
    pub persistence: Vec<f64>,
    /// The close-channel MSE ratio the SAME forecast would score with its amplitude rescaled
    /// to `β̂` and nothing else changed, `1 - offset_gain - demeaned_gain`. Beside
    /// [`Self::close_mse_ratio`] it separates "no signal" (both near 1) from "signal, wrong
    /// amplitude" (achieved above 1, this one below).
    pub best_scale_mse_ratio: Vec<f64>,
    /// Cross-sections that actually contributed to [`Self::cross_sectional_ic`] at this
    /// horizon: timestamps carrying at least `CROSS_SECTION_MIN` valid names
    /// and a nonzero spread in both the forecast and the target. Zero on a draw whose origins
    /// each sit on their own timestamp, which is why the IC there is NaN.
    pub cross_sectional_ic_moments: Vec<f64>,
    /// Distinct evaluation timestamps behind the cross-sectional statistics.
    pub cross_sections: usize,
}

/// The horizons whose behaviour over training decides a comparison, in bars ahead. Seven, and
/// these seven, because they are exactly the endpoint-utility grid: that
/// family already reports its payoffs, break-evens and turnover on this grid, and the question
/// that decides adoption - does the information coefficient at the horizon a policy trades
/// survive training - is only answerable if the signal family and the utility family are
/// indexed the same way. The regimes they cover: `h = 1` is the microstructure-adjacent bar
/// where the ratio is lowest, `h = 8` is where the only real skill has ever been measured,
/// `h = 16`/`32` fill in the term structure between the short end and the drift regime,
/// `h = 64` is where a drift failure first shows as `MSE ratio > 1`, and `h = 128`/`192` are
/// the long end that a run fitting the training period's drift damages first. Four horizons
/// could not show a term structure at all: with 8 and 64 adjacent, a rotation between them was
/// a jump between two points.
///
/// A horizon past the model's own `pred_len` is skipped rather than reported as a gap, so
/// shortening the forecast window silently drops its series instead of writing NaNs.
pub const DECISION_HORIZONS: &[usize] = &[1, 8, 16, 32, 64, 128, 192];

/// One decision horizon's step-trackable quantities, sliced out of the two per-horizon curves
/// of a single evaluation.
///
/// Every quantity here is one a run-versus-run comparison turns on, and none of them is
/// recoverable from the horizon-indexed bases: those are rewritten in place at every
/// evaluation, so they answer "what does the curve look like NOW" and cannot answer "when did
/// h = 64 turn".
///
/// The decisive one is [`Self::cross_sectional_ic`]. The MSE ratio is quadratic in forecast
/// amplitude and the IC is invariant to it, so the two disagree exactly when the mean is
/// mis-scaled - which is why [`Self::optimal_gain`] and [`Self::best_scale_mse_ratio`] sit
/// beside them: they say whether a ratio above 1 is absence of signal or presence of signal at
/// the wrong amplitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HorizonPoint {
    /// Bars ahead, one-based: 1 is the next bar.
    pub horizon: usize,
    /// Four-channel forecast/persistence MSE ratio in market-neutral space.
    pub neutral_ratio: f64,
    /// The same ratio with the realized market drift added back to the targets.
    pub raw_ratio: f64,
    /// [`TradingCurve::close_mse_ratio`]: the close-channel ratio the decomposition below
    /// divides, which is NOT [`Self::neutral_ratio`] - that one averages four channels.
    pub close_ratio: f64,
    /// [`TradingCurve::best_scale_mse_ratio`].
    pub best_scale_ratio: f64,
    /// [`TradingCurve::optimal_gain`]: `β̂`, 1 = perfect amplitude calibration.
    pub optimal_gain: f64,
    /// [`TradingCurve::offset_gain`]: what a constant forecast equal to the population mean
    /// already earns, as a share of the persistence MSE.
    pub offset_gain: f64,
    /// [`TradingCurve::demeaned_gain`]: the conditional forecast's gain at its own best scale.
    pub demeaned_gain: f64,
    /// [`TradingCurve::scaling_gain`]: the mis-scaling cross term on the market-neutral close
    /// coordinate, as a share of the persistence MSE.
    pub scaling_gain: f64,
    /// [`TradingCurve::cross_sectional_ic`]: the decision metric. NaN on a draw with no usable
    /// cross-section.
    pub cross_sectional_ic: f64,
    /// [`TradingCurve::cross_sectional_ic_se`]. NaN below two contributing cross-sections.
    pub cross_sectional_ic_se: f64,
    /// [`TradingCurve::pearson`]: the pooled correlation over all (window, bar) pairs at this
    /// horizon. Finite on every draw, including the ones the IC above is undefined on, and a
    /// DIFFERENT statistic - it mixes moments, so it can be carried by a slow common factor no
    /// cross-sectional strategy can trade. Charted on its own base for exactly that reason.
    pub pooled_pearson: f64,
    /// [`TradingCurve::cross_sectional_ic_moments`]: the IC's own population count.
    pub cross_sections: f64,
    /// [`HorizonCurve::within_1_sigma`] and [`HorizonCurve::within_2_sigma`].
    pub within_1_sigma: f64,
    pub within_2_sigma: f64,
    /// [`HorizonCurve::valid_elements`]: the coverage and ratio population count.
    pub valid_elements: f64,
}

/// One evaluation's slice through the per-horizon curves at every [`DECISION_HORIZONS`] entry
/// the evaluated window reaches. The ratios are formed here rather than in the caller so the
/// step-indexed series and the horizon-indexed ones divide the same two numbers.
pub fn horizon_track(horizon: &HorizonCurve, trading: &TradingCurve) -> Vec<HorizonPoint> {
    DECISION_HORIZONS
        .iter()
        .filter_map(|&h| {
            let i = h.checked_sub(1)?;
            // Every field is read through `get`, so a curve shorter than the horizon drops the
            // whole point instead of contributing a row of gaps.
            Some(HorizonPoint {
                horizon: h,
                neutral_ratio: horizon.mse.get(i)? / horizon.persistence_mse.get(i)?,
                raw_ratio: horizon.absolute_mse.get(i)? / horizon.absolute_persistence_mse.get(i)?,
                close_ratio: *trading.close_mse_ratio.get(i)?,
                best_scale_ratio: *trading.best_scale_mse_ratio.get(i)?,
                optimal_gain: *trading.optimal_gain.get(i)?,
                offset_gain: *trading.offset_gain.get(i)?,
                demeaned_gain: *trading.demeaned_gain.get(i)?,
                scaling_gain: *trading.scaling_gain.get(i)?,
                cross_sectional_ic: *trading.cross_sectional_ic.get(i)?,
                cross_sectional_ic_se: *trading.cross_sectional_ic_se.get(i)?,
                pooled_pearson: *trading.pearson.get(i)?,
                cross_sections: *trading.cross_sectional_ic_moments.get(i)?,
                within_1_sigma: *horizon.within_1_sigma.get(i)?,
                within_2_sigma: *horizon.within_2_sigma.get(i)?,
                valid_elements: *horizon.valid_elements.get(i)?,
            })
        })
        .collect()
}

/// Fixed-policy endpoint-cohort diagnostics, not a synchronized portfolio backtest.
/// Cohorts are formed at the origin (minimum 20 names); missing entry/outcome data
/// excludes the whole cohort at a horizon without changing any policy's positions.
#[derive(Debug, Clone)]
pub struct PortfolioCurve {
    pub horizons: Vec<u64>,
    /// Basis points per transacted notional, charged on both entry and exit.
    pub costs_bps: Vec<f64>,
    pub policies: Vec<PolicyCurve>,
    /// Mean valid entry+endpoint count over every origin group at each horizon.
    pub tickers_per_timestamp: Vec<f64>,
    /// Minimum valid entry+endpoint count, including incomplete and below-20 groups.
    pub narrowest_timestamp: Vec<f64>,
    /// Complete eligible cohorts per horizon; zero observations means NaN payoffs.
    pub cross_sections: Vec<usize>,
}

/// Each vector is per horizon; net matrices are indexed by cost, then horizon.
/// Cash with observations has zero payoff/exposure and undefined break-even.
#[derive(Debug, Clone)]
pub struct PolicyCurve {
    pub label: String,
    pub gross_bps: Vec<f64>,
    pub net_bps: Vec<Vec<f64>>,
    pub gross_rate_bps: Vec<f64>,
    pub net_rate_bps: Vec<Vec<f64>>,
    /// Signed mean gross bps / mean turnover; undefined when turnover is zero.
    pub breakeven_bps: Vec<f64>,
    pub gross_exposure: Vec<f64>,
    pub net_exposure: Vec<f64>,
    pub active_fraction: Vec<f64>,
    /// Sum |weight| * (1 + exp(realized entry-to-exit log return)).
    pub turnover: Vec<f64>,
    /// Descriptive cohort payoff dispersion, not a confidence interval.
    pub payoff_std_bps: Vec<f64>,
    pub worst_bps: Vec<f64>,
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

/// One step-indexed per-horizon track's admissibility. NaN is admitted for every quantity
/// because an undefined statistic has to reach the chart as a gap: the within-timestamp IC on
/// a draw with no usable cross-section, `β̂` on a constant forecast and the standard error of a
/// single cross-section are all genuinely unmeasured, and the one rendering that would lie
/// about them is 0.
fn valid_track(track: &[HorizonPoint]) -> bool {
    track.windows(2).all(|w| w[0].horizon < w[1].horizon)
        && track.iter().all(|h| {
            DECISION_HORIZONS.contains(&h.horizon)
                && [
                    h.neutral_ratio,
                    h.raw_ratio,
                    h.close_ratio,
                    h.best_scale_ratio,
                    h.within_1_sigma,
                    h.within_2_sigma,
                    h.cross_sections,
                    h.valid_elements,
                ]
                .iter()
                .all(|v| v.is_nan() || (v.is_finite() && *v >= 0.))
                && [
                    h.scaling_gain,
                    h.offset_gain,
                    h.demeaned_gain,
                    h.optimal_gain,
                    h.cross_sectional_ic,
                    h.cross_sectional_ic_se,
                    h.pooled_pearson,
                ]
                .iter()
                .all(|v| v.is_nan() || v.is_finite())
                && [h.cross_sectional_ic, h.pooled_pearson]
                    .iter()
                    .all(|v| v.is_nan() || (-1.0..=1.0).contains(v))
                && [h.within_1_sigma, h.within_2_sigma]
                    .iter()
                    .all(|v| v.is_nan() || (0.0..=1.0).contains(v))
        })
}

pub fn write_metrics(output: &Path, points: &[Metrics]) -> Result<()> {
    let Some(last) = points.last() else {
        return Ok(());
    };
    // The step axis is strictly increasing, with ONE stated exception: a step may carry its
    // `held-out sample` point followed by exactly one `held-out full` point, which is what
    // the terminal step of a `--max-steps` run writes (the interval's sample evaluation, then
    // the final full-split pass). The pair is written as two series on one x below, never as
    // two rows sharing an x. Any other collision - a repeated split at one step, a full point
    // a sample point follows, a step going backwards - is refused, because a duplicated step
    // is how a reader mistakes one split's number for another's.
    ensure!(
        points.windows(2).all(|p| p[0].step < p[1].step
            || (p[0].step == p[1].step
                && !p[0].validation_is_full
                && p[1].validation_is_full)),
        "report steps must increase, except for one held-out full point sharing the last \
         step's held-out sample point"
    );
    ensure!(
        points.iter().all(|p| p.epoch == last.epoch
            && p.tickers > 0
            && p.validation_origins > 0
            && p.completed_origins <= p.total_origins
            && p.completed_target_bars <= p.total_target_bars
            && [
                p.validation_nll,
                p.validation_objective_nll,
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
            && p.loader_build_ms.is_none_or(|v| v.is_finite() && v >= 0.)
            && p.eval.reports_ms.is_none_or(|v| v.is_finite() && v >= 0.)
            && p.peak_allocator_mib.is_none_or(|v| v.is_finite() && v > 0.)
            && (0.0..=1.0).contains(&p.invalid_ohlc_fraction)
            && (0.0..=1.0).contains(&p.tail_loss_share)
            && (0.0..=1.0).contains(&p.within_1_sigma)
            && (0.0..=1.0).contains(&p.within_2_sigma)
            && (p.median_window_ratio.is_nan() || p.median_window_ratio >= 0.)
            && valid_track(&p.horizons)
            && valid_track(&p.cross_horizons)
            && (p.cross_origins > 0) == !p.cross_horizons.is_empty()),
        "invalid segment report metrics"
    );
    // One row per STEP, not one per point, so the sample/full pair at a capped run's terminal
    // step lands as two series at one x.
    let mut axis: Vec<usize> = Vec::with_capacity(points.len());
    for point in points {
        if axis.last() != Some(&point.step) {
            axis.push(point.step);
        }
    }
    // Split-agnostic quantities - the training series, the reference lines - read the first
    // point at a step: every point at one step describes the same optimizer step and carries
    // the same training statistics.
    let row = |step: usize| points.iter().find(|p| p.step == step).unwrap_or(last);
    let scored_row = |step: usize, full: bool| {
        points
            .iter()
            .find(|p| p.step == step && p.validation_is_full == full)
    };
    let series = |label: &str, value: fn(&Metrics) -> f64| ReportSeries {
        label: label.to_owned(),
        values: axis.iter().map(|&step| value(row(step)) as f32).collect(),
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
    if let Some(count) = points
        .iter()
        .rev()
        .map(|p| p.cross_origins)
        .find(|count| *count > 0)
    {
        facts.push(split_scope(CROSS, count));
    }
    if let Some(count) = points
        .iter()
        .rev()
        .map(|p| p.in_period_origins)
        .find(|count| *count > 0)
    {
        facts.push(split_scope(IN_PERIOD, count));
        let purged = points
            .iter()
            .rev()
            .map(|p| p.in_period_purged_row_share)
            .find(|share| *share > 0.);
        if let Some(share) = purged {
            facts.push(format!(
                "TRAINED ON {:.1}% OF ROWS ({:.1}% purged for the in-period hole; training                  losses are NOT step-comparable to an unholed arm)",
                100. * (1. - share),
                100. * share
            ));
        }
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
                    steps: axis.iter().map(|&step| step as u64).collect(),
                    series,
                },
            },
        )?;
        Ok(())
    };
    let scored = |label: &str, full: bool, value: fn(&Metrics) -> f64| ReportSeries {
        label: label.to_owned(),
        values: axis
            .iter()
            .map(|&step| scored_row(step, full).map_or(f32::NAN, |p| value(p) as f32))
            .collect(),
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
            series("training objective NLL", |p| p.train_nll.unwrap_or(f64::NAN)),
            scored(&format!("{SAMPLE} objective NLL"), false, |p| {
                p.validation_objective_nll
            }),
            scored(&format!("{FULL} objective NLL"), true, |p| {
                p.validation_objective_nll
            }),
            scored(&format!("{SAMPLE} unweighted NLL"), false, |p| p.validation_nll),
            scored(&format!("{SAMPLE} persistence NLL"), false, |p| {
                p.persistence_nll
            }),
            scored(&format!("{FULL} unweighted NLL"), true, |p| p.validation_nll),
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
    // Compare the same weighted objective, not weighted training against unweighted scoring.
    let gap = |p: &Metrics| {
        p.train_nll
            .map_or(f64::NAN, |train| train - p.validation_objective_nll)
    };
    chart(
        "timexer_segment_generalization_gap",
        "is the training objective pulling away from the held-out one?",
        GAP_UNIT,
        ScaleKind::Linear,
        &scope,
        vec![
            scored(&format!("training minus {SAMPLE} objective NLL"), false, gap),
            scored(&format!("training minus {FULL} objective NLL"), true, gap),
            series("no gap 0.0", |_| 0.),
        ],
    )?;
    // x = optimizer step at a FIXED horizon: the transpose of the horizon-indexed family.
    // Those bases are rewritten in place at every evaluation, so two runs' per-horizon curves
    // exist only at whatever step each happened to write last - which made a step-matched
    // comparison of WHERE a regression lives impossible without rerunning both to the same
    // step, and made the question this family exists to answer - does the per-horizon
    // information coefficient RISE or FALL with training - unanswerable from disk at any cost
    // short of a rerun. These nine carry the decision horizons across steps instead.
    //
    // Nine bases and not one, because the quantities are in six incompatible units and the two
    // correlations are on different populations: a correlation near 0.06, a gain share near
    // 0.001, a ratio near 1, a coverage near 0.68 and a count near 10^5 on one axis renders
    // four of the five as flat lines.
    if points
        .iter()
        .any(|p| !p.horizons.is_empty() || !p.cross_horizons.is_empty())
    {
        // Which point's track a split reads at a step. `held-out cross-section` is NOT keyed by
        // `validation_is_full`: it is its own pass, scored beside whichever split the interval
        // evaluated, so it reads the first point at the step that carries one.
        let picked = |step: usize, split: &str| -> Option<&[HorizonPoint]> {
            if split == CROSS {
                points
                    .iter()
                    .find(|p| p.step == step && !p.cross_horizons.is_empty())
                    .map(|p| p.cross_horizons.as_slice())
            } else {
                scored_row(step, split == FULL).map(|p| p.horizons.as_slice())
            }
        };
        // One series per (horizon, split). A split that never measured the quantity is dropped
        // by `chart` rather than drawn as an empty line, which is what keeps the structurally
        // undefined `held-out sample` IC out of the signal panel entirely instead of leaving a
        // reader to decide whether a missing line means zero.
        //
        // `at horizon {h}` and not `at h = {h}`: `report_cli --var` selects a series by
        // splitting a rendered token on its first `=`, so a label carrying its own `=` cannot
        // be named on the command line - every horizon collapsed to the same unusable key.
        // The panel a decision is read off has to be greppable by series.
        let per_horizon = |quantity: &str, value: fn(&HorizonPoint) -> f64| -> Vec<ReportSeries> {
            let mut out = Vec::with_capacity(DECISION_HORIZONS.len() * 3);
            for &horizon in DECISION_HORIZONS {
                for split in [SAMPLE, CROSS, FULL] {
                    out.push(ReportSeries {
                        label: format!("{split} {quantity} at horizon {horizon}"),
                        values: axis
                            .iter()
                            .map(|&step| {
                                picked(step, split)
                                    .and_then(|track| {
                                        track.iter().find(|point| point.horizon == horizon)
                                    })
                                    .map_or(f32::NAN, |point| value(point) as f32)
                            })
                            .collect(),
                    });
                }
            }
            out
        };
        let mut ratios = per_horizon(&format!("{NEUTRAL} four-channel MSE ratio"), |p| {
            p.neutral_ratio
        });
        ratios.extend(per_horizon(&format!("{RAW} four-channel MSE ratio"), |p| {
            p.raw_ratio
        }));
        ratios.push(series("parity 1.0", |_| 1.));
        chart(
            "timexer_segment_horizon_steps",
            "does the forecast beat persistence at the decision horizons, over training?",
            RATIO_UNIT,
            ScaleKind::Linear,
            &scope,
            ratios,
        )?;
        // THE decision panel. A cross-sectional trading model is adopted or rejected on
        // whether the within-timestamp rank information at the horizon it would trade grows or
        // decays over training, and every other panel in this family exists to explain a move
        // in this one.
        let mut signal = per_horizon("close cross-sectional IC", |p| p.cross_sectional_ic);
        signal.push(series("zero information 0.0", |_| 0.));
        chart(
            "timexer_segment_horizon_steps_signal",
            "does the per-horizon cross-sectional information coefficient rise or fall over \
             training?",
            IC_UNIT,
            ScaleKind::Linear,
            &scope,
            signal,
        )?;
        chart(
            "timexer_segment_horizon_steps_signal_error",
            "how precisely is each horizon's cross-sectional IC measured?",
            IC_ERROR_UNIT,
            ScaleKind::Linear,
            &scope,
            per_horizon("close cross-sectional IC standard error", |p| {
                p.cross_sectional_ic_se
            }),
        )?;
        let mut pooled = per_horizon("close pooled Pearson IC", |p| p.pooled_pearson);
        pooled.push(series("zero information 0.0", |_| 0.));
        chart(
            "timexer_segment_horizon_steps_pooled",
            "does the pooled correlation rise or fall over training, on the draws the \
             cross-sectional IC is undefined on?",
            POOLED_UNIT,
            ScaleKind::Linear,
            &scope,
            pooled,
        )?;
        // The population behind both correlations and the coverages. A first-class series
        // rather than a title fact, because it moves: a draw whose timestamps thin out reads
        // as a noisier IC, and the two are only separable if the count has its own trajectory.
        let mut population = per_horizon("contributing cross-sections", |p| p.cross_sections);
        population.extend(per_horizon("valid target bar-channels", |p| p.valid_elements));
        chart(
            "timexer_segment_horizon_steps_population",
            "how much data is behind each horizon's statistics, per step?",
            POPULATION_UNIT,
            ScaleKind::Symlog,
            &scope,
            population,
        )?;
        let mut decomposition = per_horizon(
            &format!("{NEUTRAL} close gain from the unconditional offset"),
            |p| p.offset_gain,
        );
        decomposition.extend(per_horizon(
            &format!("{NEUTRAL} close gain from the demeaned forecast at its best scale"),
            |p| p.demeaned_gain,
        ));
        decomposition.extend(per_horizon(
            &format!("{NEUTRAL} close gain lost to forecast mis-scaling (cross term)"),
            |p| p.scaling_gain,
        ));
        decomposition.push(series("zero gain 0.0", |_| 0.));
        chart(
            "timexer_segment_horizon_steps_decomposition",
            "over training, is the close MSE gain a constant tilt, a conditional signal, or a \
             mis-scaled amplitude?",
            DECOMPOSITION_UNIT,
            ScaleKind::Linear,
            &scope,
            decomposition,
        )?;
        let mut gain = per_horizon("close MSE-optimal forecast gain", |p| p.optimal_gain);
        gain.push(series("perfect amplitude calibration 1.0", |_| 1.));
        chart(
            "timexer_segment_horizon_steps_gain",
            "is the conditional mean's amplitude right, per horizon, over training?",
            OPTIMAL_GAIN_UNIT,
            ScaleKind::Linear,
            &scope,
            gain,
        )?;
        let mut best_scale = per_horizon(&format!("{NEUTRAL} close MSE ratio"), |p| p.close_ratio);
        best_scale.extend(per_horizon(
            &format!("{NEUTRAL} close MSE ratio at the best scale"),
            |p| p.best_scale_ratio,
        ));
        best_scale.push(series("parity 1.0", |_| 1.));
        chart(
            "timexer_segment_horizon_steps_best_scale",
            "is a close MSE ratio above 1 absent signal, or signal at the wrong amplitude?",
            BEST_SCALE_UNIT,
            ScaleKind::Linear,
            &scope,
            best_scale,
        )?;
        // Per horizon, because the aggregate is dominated by the short end: every horizon
        // contributes the same bar count to it, so a scale that is over-dispersed only where
        // the mean is over-amplified is invisible there.
        let mut calibration = per_horizon("within 1σ", |p| p.within_1_sigma);
        calibration.extend(per_horizon("within 1.96σ", |p| p.within_2_sigma));
        calibration.push(series("nominal 1σ = 0.683", |_| 0.6827));
        calibration.push(series("nominal 1.96σ = 0.950", |_| 0.95));
        chart(
            "timexer_segment_horizon_steps_calibration",
            "do the predicted σ bands cover the realized targets at each horizon, over \
             training?",
            FRACTION_UNIT,
            ScaleKind::Linear,
            &scope,
            calibration,
        )?;
    }
    // THE fork. Every other `held-out *` draw is cut from the `[80%, 90%)` partition, so each
    // one is out-of-sample in ORIGIN IDENTITY and out-of-period in MARKET REGIME at the same
    // time, and a decay in any of them is consistent with two mechanisms whose fixes are
    // opposite: overfitting is answered with capacity, regularization or effective sample
    // size, and non-stationarity is answered with data recency, target definition or online
    // adaptation. This base holds the period fixed and varies only origin identity.
    //
    // The PRE-REGISTERED reading rule, stated here rather than in a report so it cannot be
    // chosen after the curves exist:
    //
    //   - in-period IC keeps improving past the out-of-period peak while out-of-period IC
    //     collapses  =>  NON-STATIONARITY. The lever is data/target/adaptation, not
    //     regularization.
    //   - BOTH collapse together  =>  genuine overfitting on a correlated-sample budget. The
    //     lever is capacity, regularization, or effective sample size.
    //   - in-period ALSO peaks but less severely  =>  report the RATIO of the two declines and
    //     pick no side. The difference series below is what that ratio is read off.
    //
    // The difference's standard error is `sqrt(se_in^2 + se_out^2)`, which assumes the two
    // draws' ICs are independent. They are drawn from disjoint bars in disjoint market periods
    // separated by the purge and the whole `[70%, 80%)` calibration partition, so the
    // assumption is that no common factor spans that gap. It is an assumption, not a fact, and
    // it is why the two component SEs are charted beside the difference's rather than only the
    // combined one: a reader who rejects the independence assumption can still bound the
    // difference by the larger component.
    if points.iter().any(|p| !p.in_period_horizons.is_empty()) {
        let track = |step: usize, in_period: bool| -> Option<&[HorizonPoint]> {
            points
                .iter()
                .find(|p| {
                    p.step == step
                        && if in_period {
                            !p.in_period_horizons.is_empty()
                        } else {
                            !p.cross_horizons.is_empty()
                        }
                })
                .map(|p| {
                    if in_period {
                        p.in_period_horizons.as_slice()
                    } else {
                        p.cross_horizons.as_slice()
                    }
                })
        };
        let at = |step: usize, in_period: bool, horizon: usize| -> Option<HorizonPoint> {
            track(step, in_period)?
                .iter()
                .find(|point| point.horizon == horizon)
                .copied()
        };
        // `at horizon {h}` and never `at h = {h}`: `report_cli --var` splits a rendered token
        // on its first `=`, so a label carrying one cannot be selected on the command line.
        let mut split = Vec::with_capacity(DECISION_HORIZONS.len() * 6);
        for &horizon in DECISION_HORIZONS {
            for (label, value) in [
                (
                    format!("{IN_PERIOD} cross-sectional IC at horizon {horizon}"),
                    (true, false) as (bool, bool),
                ),
                (
                    format!("{IN_PERIOD} cross-sectional IC standard error at horizon {horizon}"),
                    (true, true),
                ),
                (
                    format!("{OUT_OF_PERIOD} cross-sectional IC at horizon {horizon}"),
                    (false, false),
                ),
                (
                    format!(
                        "{OUT_OF_PERIOD} cross-sectional IC standard error at horizon {horizon}"
                    ),
                    (false, true),
                ),
            ] {
                let (in_period, error) = value;
                split.push(ReportSeries {
                    label,
                    values: axis
                        .iter()
                        .map(|&step| {
                            at(step, in_period, horizon).map_or(f32::NAN, |p| {
                                if error {
                                    p.cross_sectional_ic_se as f32
                                } else {
                                    p.cross_sectional_ic as f32
                                }
                            })
                        })
                        .collect(),
                });
            }
            split.push(ReportSeries {
                label: format!("in-period minus out-of-period cross-sectional IC at horizon {horizon}"),
                values: axis
                    .iter()
                    .map(|&step| {
                        match (at(step, true, horizon), at(step, false, horizon)) {
                            (Some(inside), Some(outside)) => {
                                (inside.cross_sectional_ic - outside.cross_sectional_ic) as f32
                            }
                            _ => f32::NAN,
                        }
                    })
                    .collect(),
            });
            split.push(ReportSeries {
                label: format!(
                    "in-period minus out-of-period cross-sectional IC standard error at horizon \
                     {horizon}"
                ),
                values: axis
                    .iter()
                    .map(|&step| {
                        match (at(step, true, horizon), at(step, false, horizon)) {
                            (Some(inside), Some(outside)) => (inside
                                .cross_sectional_ic_se
                                .hypot(outside.cross_sectional_ic_se))
                                as f32,
                            _ => f32::NAN,
                        }
                    })
                    .collect(),
            });
        }
        split.push(series("no period difference 0.0", |_| 0.));
        chart(
            "timexer_segment_temporal_generalization",
            "is the held-out decay loss of skill on unseen ORIGINS, or on an unseen market \
             PERIOD?",
            IC_UNIT,
            ScaleKind::Linear,
            &scope,
            split,
        )?;
    }
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
                "host batch assembly on the loader thread (interval mean; free below the step)",
                |p| p.loader_build_ms.unwrap_or(f64::NAN),
            ),
            series(
                "main thread blocked in the loader receive (interval mean; costs step time only above the step)",
                |p| p.loader_wait_ms.unwrap_or(f64::NAN),
            ),
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
            series("held-out cross-section pass total", |p| {
                p.eval.cross_section_ms.unwrap_or(f64::NAN)
            }),
            series("in-period held-out pass total", |p| {
                p.eval.in_period_ms.unwrap_or(f64::NAN)
            }),
        ],
    )?;
    // One-time cost, so it is NOT a series on the per-step timing chart above: 172,000 ms of a
    // cold corpus load on the same linear axis as a 2-180 ms step flattens every per-step series
    // to zero, and ten constants would occupy ten of that chart's colours for numbers that never
    // move. Its own base, on its own decade axis, answering its own question.
    //
    // Symlog rather than log because a warm cache legitimately reads 0.0 for a phase that ran and
    // found nothing to do, and NaN for one that did not run at all - the axis has to hold both
    // without dropping the row.
    let startup = last.startup;
    let phases = startup.phases();
    // The last entry is the rescan counter, which is a bar count and not a millisecond. It goes
    // in the scope line so the chart keeps one unit on one axis while still saying, in the place
    // a reader is already looking, whether they are seeing a rebuild or a cache hit.
    let (rescan, timed) = phases
        .split_last()
        .context("LoadTiming::phases must carry the rescan counter last")?;
    chart(
        "timexer_segment_startup",
        "what did this run pay before the first step?",
        "milliseconds, one-time",
        ScaleKind::Symlog,
        &format!(
            "{scope}; {} of {} bar audits recomputed, {:.0} M bar records rescanned",
            startup.audits_computed,
            startup.audits_computed + startup.audits_reused,
            rescan.1
        ),
        timed
            .iter()
            .map(|&(label, ms)| ReportSeries {
                label: label.to_owned(),
                // Constant over the run by construction: the load happened once, before the
                // first point. A flat line is the honest shape for it, and it lets the reader
                // put the startup cost beside the step cost on the same x.
                values: axis.iter().map(|_| ms as f32).collect(),
            })
            .collect(),
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

/// Forecast diagnostics plus fixed-policy raw endpoint utility.
/// Utility has one cost per payoff/rate panel, so policy comparisons do not drown in
/// policy × split × cost series. It is horizon-indexed, not a training-step trajectory.
pub fn write_trading(
    output: &Path,
    epoch: usize,
    step: usize,
    sample: Option<TradingSplit<'_>>,
    full: Option<TradingSplit<'_>>,
    cross_section: Option<TradingSplit<'_>>,
    tickers: usize,
) -> Result<()> {
    let horizon = sample
        .or(full)
        .or(cross_section)
        .map_or(0, |split| split.curve.total_gain.len());
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
    let costs = super::utility::COSTS_BPS;
    let mut bps: [Vec<ReportSeries>; 6] = std::array::from_fn(|_| Vec::new());
    let mut rate: [Vec<ReportSeries>; 6] = std::array::from_fn(|_| Vec::new());
    let mut breakeven = Vec::new();
    let mut gross_exposure = Vec::new();
    let mut net_exposure = Vec::new();
    let mut active_fraction = Vec::new();
    let mut turnover = Vec::new();
    let mut payoff_std = Vec::new();
    let mut worst_payoff = Vec::new();
    let mut census = Vec::new();
    let mut portfolio_steps: Vec<u64> = Vec::new();
    for (split, evaluated) in [(SAMPLE, sample), (FULL, full), (CROSS, cross_section)] {
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
                format!("{split} {NEUTRAL} close gain from the demeaned forecast (ρ² var(y) / mean(y²))"),
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
                format!("{split} close cross-sectional IC {edge} (iid approximation, not confidence)"),
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
                format!("{split} {NEUTRAL} close-difference MSE ratio excluding the first bar (not execution P&L)"),
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
                format!("{split} close-difference hit rate excluding the first bar"),
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
        let swept = |rows: &[Vec<f64>]| {
            rows.len() == portfolio.costs_bps.len() && rows.iter().all(|row| row.len() == points)
        };
        ensure!(
            points > 0
                && portfolio.costs_bps == costs
                && !portfolio.policies.is_empty()
                && [
                    portfolio.tickers_per_timestamp.len(),
                    portfolio.narrowest_timestamp.len(),
                    portfolio.cross_sections.len(),
                ]
                .iter()
                .all(|len| *len == points),
            "portfolio curves must be rectangular over the reported horizons"
        );
        if portfolio_steps.is_empty() {
            portfolio_steps = portfolio.horizons.clone();
        }
        ensure!(
            portfolio_steps == portfolio.horizons,
            "every split must report the same portfolio holding periods"
        );
        // Availability census never changes positions; only complete cohorts enter payoffs.
        let timestamps: Vec<f64> = portfolio
            .cross_sections
            .iter()
            .map(|count| *count as f64)
            .collect();
        for (label, values) in [
            (
                format!("{split} complete eligible origin cohorts"),
                &timestamps,
            ),
            (
                format!("{split} mean observed valid width (all origin groups)"),
                &portfolio.tickers_per_timestamp,
            ),
            (
                format!("{split} minimum observed valid width (all origin groups)"),
                &portfolio.narrowest_timestamp,
            ),
        ] {
            census.push(trading_series(label, values, points)?);
        }
        for policy in &portfolio.policies {
            ensure!(
                swept(&policy.net_bps)
                    && swept(&policy.net_rate_bps)
                    && [
                        policy.gross_bps.len(),
                        policy.gross_rate_bps.len(),
                        policy.breakeven_bps.len(),
                        policy.gross_exposure.len(),
                        policy.net_exposure.len(),
                        policy.active_fraction.len(),
                        policy.turnover.len(),
                        policy.payoff_std_bps.len(),
                        policy.worst_bps.len(),
                    ]
                    .iter()
                    .all(|len| *len == points),
                "utility policy curves must be rectangular over costs and horizons"
            );
            let label = format!("{split} {}", policy.label);
            for (values, into) in [
                (&policy.breakeven_bps, &mut breakeven),
                (&policy.gross_exposure, &mut gross_exposure),
                (&policy.net_exposure, &mut net_exposure),
                (&policy.active_fraction, &mut active_fraction),
                (&policy.turnover, &mut turnover),
                (&policy.payoff_std_bps, &mut payoff_std),
                (&policy.worst_bps, &mut worst_payoff),
            ] {
                into.push(trading_series(label.clone(), values, points)?);
            }
            for index in 0..costs.len() {
                let (payoff, per_bar) = if index == 0 {
                    (&policy.gross_bps, &policy.gross_rate_bps)
                } else {
                    (&policy.net_bps[index], &policy.net_rate_bps[index])
                };
                bps[index].push(trading_series(label.clone(), payoff, points)?);
                rate[index].push(trading_series(label.clone(), per_bar, points)?);
            }
        }
    }
    facts.push(format!("{tickers} tickers"));
    let scope = facts.join(", ");
    let utility_scope = format!(
        "origins sample/full/cross={}/{}/{}",
        sample.map_or(0, |split| split.origins),
        full.map_or(0, |split| split.origins),
        cross_section.map_or(0, |split| split.origins),
    );
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
    breakeven.push(line("zero gross edge", 0., points));
    census.push(line(
        "20 origin names (eligibility minimum before outcome filtering)",
        super::runner::CROSS_SECTION_MIN as f32,
        points,
    ));
    for (base, question, x_label, y_label, steps, series) in [
        (
            "timexer_segment_decomposition",
            "where does the MSE gain come from: a constant tilt or a conditional signal?",
            "bars ahead",
            GAIN_UNIT,
            None,
            gains,
        ),
        (
            "timexer_segment_signal",
            "how much information does the demeaned close forecast carry?",
            "bars ahead",
            CORRELATION_UNIT,
            None,
            signal,
        ),
        (
            "timexer_segment_offset",
            "how big is the constant tilt, and what does conviction actually pay?",
            "bars ahead",
            COORDINATE_UNIT,
            None,
            levels,
        ),
        (
            "timexer_segment_tradable",
            "how does forecast quality change under a mid anchor and a first-bar exclusion?",
            "bars ahead",
            ANCHOR_RATIO_UNIT,
            None,
            anchors,
        ),
        (
            "timexer_segment_tradable_rates",
            "is the forecast sign right under the mid anchor or first-bar exclusion?",
            "bars ahead",
            ANCHOR_RATE_UNIT,
            None,
            anchor_rates,
        ),
        (
            "timexer_segment_utility_breakeven",
            "what signed cost allowance does gross payoff / turnover imply?",
            "h: next observed bars (not common UTC exits)",
            BREAKEVEN_UNIT,
            Some(portfolio_steps.clone()),
            breakeven,
        ),
        (
            "timexer_segment_utility_gross_exposure",
            "how much initial gross notional does each fixed policy deploy?",
            "h: next observed bars (not common UTC exits)",
            EXPOSURE_UNIT,
            Some(portfolio_steps.clone()),
            gross_exposure,
        ),
        (
            "timexer_segment_utility_net_exposure",
            "what signed net notional does each fixed policy deploy?",
            "h: next observed bars (not common UTC exits)",
            EXPOSURE_UNIT,
            Some(portfolio_steps.clone()),
            net_exposure,
        ),
        (
            "timexer_segment_utility_active_fraction",
            "what fraction of origin names receive a nonzero position?",
            "h: next observed bars (not common UTC exits)",
            ACTIVE_UNIT,
            Some(portfolio_steps.clone()),
            active_fraction,
        ),
        (
            "timexer_segment_utility_turnover",
            "how much entry plus drift-adjusted exit notional is transacted?",
            "h: next observed bars (not common UTC exits)",
            TURNOVER_UNIT,
            Some(portfolio_steps.clone()),
            turnover,
        ),
        (
            "timexer_segment_utility_payoff_std",
            "what is descriptive gross cohort payoff dispersion (not confidence)?",
            "h: next observed bars (not common UTC exits)",
            BPS_UNIT,
            Some(portfolio_steps.clone()),
            payoff_std,
        ),
        (
            "timexer_segment_utility_worst_payoff",
            "what is the worst observed gross cohort payoff (not a risk bound)?",
            "h: next observed bars (not common UTC exits)",
            BPS_UNIT,
            Some(portfolio_steps.clone()),
            worst_payoff,
        ),
        (
            "timexer_segment_utility_census",
            "how many complete origin cohorts contribute without resizing positions?",
            "h: next observed bars (not common UTC exits)",
            CENSUS_UNIT,
            Some(portfolio_steps.clone()),
            census,
        ),
    ] {
        let title = if steps.is_some() {
            format!("CausalPatch e{epoch} s{step} | {question} | {utility_scope} | {UTILITY_SCOPE}")
        } else {
            format!("CausalPatch epoch {epoch} step {step} | {question} | {scope}")
        };
        write_report(
            output.join(format!("{base}.report.bin")),
            &Report {
                title,
                x_label: Some(x_label.to_owned()),
                y_label: Some(y_label.to_owned()),
                scale: ScaleKind::Linear,
                kind: ReportKind::IndexedLines {
                    steps: steps.unwrap_or_else(|| (1..=horizon as u64).collect()),
                    series,
                },
            },
        )?;
    }
    for (index, (suffix, payoff, per_bar)) in [
        "", "_cost_0p5", "_cost_1", "_cost_2", "_cost_5", "_cost_10",
    ]
    .into_iter()
    .zip(bps)
    .zip(rate)
    .map(|((suffix, payoff), per_bar)| (suffix, payoff, per_bar))
    .enumerate()
    {
        let cost = costs[index];
        for (quantity, unit, series) in [
            ("payoff", BPS_UNIT, payoff),
            ("rate", RATE_UNIT_BPS, per_bar),
        ] {
            write_report(
                output.join(format!("timexer_segment_utility_{quantity}{suffix}.report.bin")),
                &Report {
                    title: format!(
                        "CausalPatch e{epoch} s{step} | raw {quantity}, {cost} bps/side | \
                         {utility_scope} | {UTILITY_SCOPE}"
                    ),
                    x_label: Some("h: next observed bars (not common UTC exits)".to_owned()),
                    y_label: Some(unit.to_owned()),
                    scale: ScaleKind::Linear,
                    kind: ReportKind::IndexedLines {
                        steps: portfolio_steps.clone(),
                        series,
                    },
                },
            )?;
        }
    }
    Ok(())
}

pub(super) fn write_account(
    output: &Path,
    epoch: usize,
    step: usize,
    result: &super::portfolio::AccountEvaluation,
) -> Result<()> {
    use super::portfolio::AccountPoint;
    use std::collections::BTreeSet;

    fn unit(key: &str) -> &'static str {
        if key.ends_with("_usd") {
            "USD"
        } else if key.ends_with("_ms") {
            "milliseconds"
        } else if key.ends_with("_count") {
            "count"
        } else {
            "dimensionless"
        }
    }

    let summary = [AccountPoint {
        timestamp_ms: step as i64,
        values: result.summary.clone(),
    }];
    let scope = format!(
        "synchronized account | validation (checkpoint-selection exposed, not terminal test) | epoch {epoch} step {step}"
    );
    let panels: [(&str, &str, &[AccountPoint], &str); 12] = [
        ("timexer_segment_account_value", "account value and cash", &result.points, "USD"),
        ("timexer_segment_account_costs", "cumulative execution and borrow costs", &result.points, "USD"),
        ("timexer_segment_account_risk", "realized exposure and drawdown", &result.points, "dimensionless"),
        ("timexer_segment_account_activity", "holdings and execution activity", &result.points, "count"),
        ("timexer_segment_account_daily_pnl", "daily P&L and costs", &result.daily, "USD"),
        ("timexer_segment_account_daily_return", "daily account returns", &result.daily, "dimensionless"),
        ("timexer_segment_account_monthly_pnl", "monthly P&L and costs (partial months retained)", &result.monthly, "USD"),
        ("timexer_segment_account_monthly_return", "monthly account returns (partial months retained)", &result.monthly, "dimensionless"),
        ("timexer_segment_account_summary_money", "account outcome and execution costs", &summary, "USD"),
        ("timexer_segment_account_summary_risk", "account outcome and risk", &summary, "dimensionless"),
        ("timexer_segment_account_summary_census", "universe, signals and execution census", &summary, "count"),
        ("timexer_segment_account_timing", "evaluation wall-clock attribution", &summary, "milliseconds"),
    ];
    for (base, question, points, expected_unit) in panels {
        let keys: BTreeSet<&str> = points
            .iter()
            .flat_map(|point| point.values.keys().map(String::as_str))
            .filter(|key| unit(key) == expected_unit)
            .filter(|key| {
                let cost = matches!(
                    *key,
                    "commission_usd" | "regulatory_usd" | "spread_usd"
                        | "slippage_usd" | "borrow_usd" | "costs_usd"
                );
                match base {
                    "timexer_segment_account_value" => !cost,
                    "timexer_segment_account_costs" => cost,
                    _ => true,
                }
            })
            .collect();
        let series = keys
            .into_iter()
            .map(|key| {
                let values = points
                    .iter()
                    .map(|point| {
                        let value = point.values.get(key).copied().unwrap_or(f64::NAN);
                        ensure!(
                            value.is_nan() || (value.is_finite() && (value as f32).is_finite()),
                            "account report {key} contains an unrepresentable value"
                        );
                        Ok(value as f32)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(ReportSeries {
                    label: key.to_owned(),
                    values,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let is_summary = std::ptr::eq(points, summary.as_slice());
        let title = if is_summary {
            format!("{scope} | {question} | {}", result.assumptions.join("; "))
        } else {
            format!("{scope} | {question}")
        };
        write_report(
            output.join(format!("{base}.report.bin")),
            &Report {
                title,
                x_label: Some(if is_summary {
                    "checkpoint optimizer step".to_owned()
                } else {
                    "UTC timestamp (milliseconds; period end for daily/monthly)".to_owned()
                }),
                y_label: Some(expected_unit.to_owned()),
                scale: ScaleKind::Linear,
                kind: ReportKind::IndexedLines {
                    steps: points
                        .iter()
                        .map(|point| {
                            u64::try_from(point.timestamp_ms)
                                .context("account report timestamp must be nonnegative")
                        })
                        .collect::<Result<_>>()?,
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

/// `calibration_origins` is `Corpus::calibration_refs.len()`, and it is a PARAMETER rather
/// than a contract field on purpose: the `[70%, 80%)` population is a pure function of the
/// boundaries, purge, geometry and ticker set the contract already carries, and
/// `load_checkpoint` compares the whole contract against the one authenticated inside each
/// checkpoint manifest, so adding a derived scalar to it would invalidate every checkpoint on
/// disk to state something already implied.
pub fn write_corpus(
    output: &Path,
    contract: &CorpusContract,
    calibration_origins: usize,
    market: &MarketSummary,
) -> Result<()> {
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
    // The dispersion clause reads beside the market clause because the two are defined on
    // IDENTICAL slots: `steps / slots` is the share of the grid defining both, and the sigma
    // quartiles are what say the dispersion channel is populated and not degenerate.
    report.title = format!("{} | corpus: {} tickers, {source_bars} source bars, {invalid_bars} malformed source bars omitted; {} training / {} calibration / {} held-out targets, {} unused held-out remainder; {} excluded ({reasons}); market: {} steps over >= {} tickers, contributors min {} median {}, step std {:.5}, largest step {:+.5} at {}; dispersion: defined at {:.2}% of {} grid slots, sigma_slot median {:.5}, IQR {:.5} to {:.5}",
        report.title.split(" | corpus:").next().unwrap_or(&report.title), contract.tickers.len(), contract.train_target_bars,
        calibration_origins * contract.pred_len, contract.validation_target_bars,
        contract.validation_remainder_bars, contract.excluded_tickers.len(), market.steps, contract.market_min_cross_section,
        market.min_contributors, market.median_contributors, market.step_std, market.max_abs_step, market.max_abs_step_ts,
        100.0 * market.steps as f64 / market.slots.max(1) as f64, market.slots,
        market.median_dispersion, market.dispersion_p25, market.dispersion_p75);
    write_report(path, &report)?;
    Ok(())
}

/// Objective-weight unit. Declared here rather than in the unit block above because this is
/// the only panel that reads it, and the block above is a shared surface.
const WEIGHT_UNIT: &str = "loss weight (dimensionless; mean 1 over the full horizon; \
                           0 = horizon not trained)";

/// The objective's per-horizon weight, x = bars ahead. One question, one unit, one axis: what
/// does the loss weight at each horizon?
///
/// Read back from the buffer [`super::model::CausalPatchModel::losses`] actually multiplies by,
/// so the chart cannot disagree with the gradient. Constant for a whole run, hence written once
/// per report interval rather than carrying a step axis, with the `uniform` reference drawn
/// beside it so a mode's departure from equal weighting is one glance.
pub fn write_horizon_loss_weight(
    output: &Path,
    epoch: usize,
    step: usize,
    spec: &str,
    weights: &[f64],
) -> Result<()> {
    let horizon = weights.len();
    ensure!(
        horizon > 0 && weights.iter().all(|w| w.is_finite() && *w >= 0.),
        "the objective weight vector must be a nonempty vector of finite nonnegative weights"
    );
    let total: f64 = weights.iter().sum();
    ensure!(
        (total - horizon as f64).abs() <= 1e-6 * horizon as f64,
        "the objective weight vector sums to {total}, not the {horizon} its mean-1 \
         normalization requires"
    );
    let trained = weights.iter().filter(|w| **w > 0.).count();
    write_report(
        output.join("timexer_segment_horizon_loss_weight.report.bin"),
        &Report {
            title: format!(
                "CausalPatch epoch {epoch} step {step} | what does the objective weight at each horizon? | horizon loss {spec}, {trained} of {horizon} horizons trained, mean weight 1 by construction"
            ),
            x_label: Some("bars ahead".to_owned()),
            y_label: Some(WEIGHT_UNIT.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: (1..=horizon as u64).collect(),
                series: vec![
                    ReportSeries {
                        label: "training horizon loss weight".to_owned(),
                        values: weights.iter().map(|w| *w as f32).collect(),
                    },
                    ReportSeries {
                        label: "uniform reference 1.0".to_owned(),
                        values: vec![1.0; horizon],
                    },
                ],
            },
        },
    )?;
    Ok(())
}

/// Whether a scored pass's emitted mean already carried the applied gain.
///
/// Both states report the SAME two labelled series, because the transform is invertible: a
/// pass whose emission was uncalibrated gets the gained ratio by applying `g`, and a pass whose
/// emission was already calibrated gets the un-gained ratio by applying `1/g`. The training
/// loop scores un-gained (its NLL is the selection scalar and must stay the un-gained model's);
/// `evaluate` scores the checkpoint's calibrated output. Neither has to run a second pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Emission {
    Uncalibrated,
    Calibrated,
}

/// One scored split's amplitude evidence: its own per-(horizon, channel) moments and whether
/// the mean they were measured on carried the gain.
pub struct AmplitudeSplit<'a> {
    /// One of the split vocabulary words.
    pub split: &'static str,
    pub origins: usize,
    pub emission: Emission,
    pub moments: &'a Moments,
}

/// The in-run fit: the curves, the block they were fitted on, and the in-sample comparand that
/// says WHY the amplitude is wrong.
pub struct AmplitudeFit<'a> {
    pub calibration: &'a MeanCalibration,
    /// The reserved `[70%, 80%)` block's own moments - what the curves were fitted on.
    pub moments: &'a Moments,
    /// A small TRAINING-split draw, measured with the same reduction on the same weights. This
    /// is the one number that separates the two mechanisms behind an amplitude that sweeps with
    /// the horizon: if the training gain is flat near 1 while the held-out gain sweeps, the
    /// forecaster is over-fitting long horizons and this calibration is the whole fix; if the
    /// training gain sweeps too, the amplitude is wrong IN SAMPLE and the cause is upstream in
    /// the objective, where a σ carrying a `√h` prior decays the effective weight on the mean
    /// across the axis. A calibration would then be a patch over a live defect.
    pub training: Option<&'a Moments>,
    pub training_origins: usize,
}

/// Everything the three amplitude bases are written from, in one argument, so a caller cannot
/// write one of them against a different state than another.
pub struct AmplitudePanels<'a> {
    pub epoch: usize,
    pub step: usize,
    pub tickers: usize,
    /// The curves the model applies (or would apply): the checkpoint's own.
    pub applied: &'a FrozenGain,
    pub fit: Option<AmplitudeFit<'a>>,
    pub scored: Vec<AmplitudeSplit<'a>>,
}

impl AmplitudeSplit<'_> {
    /// The two labelled ratio series for one channel: as emitted, and with the gain state
    /// inverted. `(uncalibrated, calibrated)` in that order whichever way round the pass ran.
    fn ratios(&self, applied: &FrozenGain, channel: Option<usize>) -> (Vec<f64>, Vec<f64>) {
        let horizons = self.moments.pred_len;
        let ratio = |horizon: usize, anchor: f64, offset: f64| match channel {
            Some(channel) => self
                .moments
                .channel_gained_ratio(horizon, channel, anchor, offset),
            None => self.moments.pooled_gained_ratio(horizon, anchor, offset),
        };
        let (mut uncalibrated, mut calibrated) = (Vec::new(), Vec::new());
        for horizon in 0..horizons {
            let (anchor, offset) = (applied.anchor[horizon], applied.offset[horizon]);
            match self.emission {
                Emission::Uncalibrated => {
                    uncalibrated.push(ratio(horizon, 1., 1.));
                    calibrated.push(ratio(horizon, anchor, offset));
                }
                Emission::Calibrated => {
                    uncalibrated.push(ratio(horizon, anchor.recip(), offset.recip()));
                    calibrated.push(ratio(horizon, 1., 1.));
                }
            }
        }
        (uncalibrated, calibrated)
    }
}

/// The three amplitude panels, written together on every evaluation, calibrated or not.
///
/// The identity is written as 1.0 at every horizon rather than as NaN: that is a measured fact
/// about what the model multiplied by, and "the applied calibration was the identity" is a
/// different statement from "no calibration exists". Writing them always is what makes a
/// calibrated arm impossible to step-match against an uncalibrated one by accident.
pub fn write_amplitude(output: &Path, panels: &AmplitudePanels<'_>) -> Result<()> {
    let horizon = panels.applied.anchor.len();
    ensure!(horizon > 0, "a gain curve needs at least one horizon");
    ensure!(
        panels.applied.offset.len() == horizon,
        "the applied gain carries {} anchor and {} offset coefficients",
        horizon,
        panels.applied.offset.len()
    );
    for split in &panels.scored {
        ensure!(
            split.moments.pred_len == horizon,
            "the {} split carries {} horizons against the gain's {horizon}",
            split.split,
            split.moments.pred_len
        );
    }
    if let Some(fit) = &panels.fit {
        ensure!(
            fit.moments.pred_len == horizon,
            "the fit block carries {} horizons against the gain's {horizon}",
            fit.moments.pred_len
        );
    }
    write_gain_panel(output, panels, horizon)?;
    write_amplitude_ratio(output, panels, horizon)?;
    write_calibration_moments(output, panels, horizon)?;
    Ok(())
}

/// The dimensionless-gain panel: what was applied, what each population's own MSE-optimal
/// amplitude is, and - the reading rule this panel exists for - whether the amplitude error is
/// an out-of-sample shrinkage problem or an in-sample objective problem.
///
/// Every series here is a dimensionless gain on the same axis, so they share it legitimately.
/// The ratios live on their own base for the opposite reason.
fn write_gain_panel(output: &Path, panels: &AmplitudePanels<'_>, horizon: usize) -> Result<()> {
    let curve = |label: String, values: &[f64]| ReportSeries {
        label,
        values: values.iter().map(|value| *value as f32).collect(),
    };
    let mut series = vec![
        curve(
            format!("{CALIBRATION} close-anchor gain applied while scoring"),
            &panels.applied.anchor,
        ),
        curve(
            format!("{CALIBRATION} intrabar-offset gain applied while scoring"),
            &panels.applied.offset,
        ),
    ];
    if let Some(fit) = &panels.fit {
        // The four DECODED channels' own MSE-optimal single gains on the fit block: this is
        // what "generalize the close-only fit to all channels" is visible as. The two applied
        // curves above are the two degrees of freedom a decoded candle actually has, and these
        // four are what they have to reproduce.
        for (channel, gains) in fit.moments.channel_gains().into_iter().enumerate() {
            series.push(curve(
                format!(
                    "{CALIBRATION} {} MSE-optimal gain (fitted on)",
                    channel_name(channel)
                ),
                &gains,
            ));
        }
        if let Some(training) = fit.training {
            // THE mechanism series. Same reduction, same weights, in-sample origins.
            series.push(curve(
                format!(
                    "{TRAINING} {} MSE-optimal gain",
                    channel_name(CLOSE_CHANNEL)
                ),
                &training.channel_gains()[CLOSE_CHANNEL],
            ));
        }
        series.push(curve(
            "three-sigma close-anchor amplification ceiling".to_owned(),
            &fit.calibration.anchor.amplification_ceiling,
        ));
    }
    if panels.fit.is_none() {
        // A loaded checkpoint has no fit block resident, but it carries the measurement its
        // curve was smoothed from - signed and unmodified, which is what makes a gated horizon
        // legible on this panel rather than only in a log line.
        series.push(curve(
            format!("{CALIBRATION} close-anchor MSE-optimal gain carried by the checkpoint, signed"),
            &panels.applied.measured_anchor,
        ));
    }
    for split in &panels.scored {
        series.push(curve(
            format!(
                "{} {} MSE-optimal gain on the emitted mean",
                split.split,
                channel_name(CLOSE_CHANNEL)
            ),
            &split.moments.channel_gains()[CLOSE_CHANNEL],
        ));
    }
    series.push(ReportSeries {
        label: "perfect amplitude calibration 1.0".to_owned(),
        values: vec![1.0; horizon],
    });
    write_report(
        output.join("timexer_segment_calibration_gain.report.bin"),
        &Report {
            title: gain_title(panels, horizon),
            x_label: Some("bars ahead".to_owned()),
            y_label: Some(OPTIMAL_GAIN_UNIT.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: (1..=horizon as u64).collect(),
                series,
            },
        },
    )?;
    Ok(())
}

/// Named, never clipped: a horizon whose own measured amplitude is non-positive or absent is
/// gated out of position sizing by [`FrozenGain::tradable`], and this is where a reader finds
/// out which ones and at what value. A gate that does not appear in the panel is how an
/// all-zero trading result stays unexplained.
fn gated_note(applied: &FrozenGain) -> String {
    let gated = applied.gated();
    if gated.is_empty() {
        return String::new();
    }
    let listed = gated
        .iter()
        .take(8)
        .map(|(horizon, measured)| format!("h={horizon} at {measured:.4}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        " | {} of {} horizons GATED OUT OF SIZING on a non-positive or absent measured gain ({listed}{}), scored but never traded",
        gated.len(),
        applied.measured_anchor.len(),
        if gated.len() > 8 { ", …" } else { "" }
    )
}

fn gain_title(panels: &AmplitudePanels<'_>, horizon: usize) -> String {
    let (epoch, step, tickers) = (panels.epoch, panels.step, panels.tickers);
    let gated = gated_note(panels.applied);
    let Some(fit) = &panels.fit else {
        return format!(
            "CausalPatch epoch {epoch} step {step} | what per-horizon mean gain did scoring apply? | the checkpoint's own curves, fitted on {} reserved calibration-partition origins ending {}, over {horizon} horizons, {tickers} tickers{gated}",
            panels.applied.blocks.calibration_origins,
            panels.applied.blocks.calibration_last_origin_ms
        );
    };
    let blocks = &fit.calibration.blocks;
    let refusal = fit
        .calibration
        .anchor
        .unidentifiable
        .as_deref()
        .map_or_else(String::new, |reason| format!(" | NO GAIN APPLIED: {reason}"));
    format!(
        "CausalPatch epoch {epoch} step {step} | is the forecast's over-amplitude an out-of-sample shrinkage problem or an in-sample objective one? | fitted on {} reserved calibration-partition origins to {} ({} ms before the first scored origin, over a {horizon}-bar reach), {} in-sample training origins measured beside them, {tickers} tickers, {:.1} anchor and {:.1} offset effective degrees of freedom{refusal}{gated}",
        blocks.calibration_origins,
        blocks.calibration_last_origin_ms,
        blocks.purge_gap_ms,
        fit.training_origins,
        fit.calibration.anchor.effective_dof,
        fit.calibration.offset.effective_dof
    )
}

/// Does applying the gain move the MSE ratio below persistence, per horizon?
///
/// Its own base, and not a pair of series on the gain panel: a share of an MSE and a
/// dimensionless gain are different units and a shared axis would make one of them unreadable.
/// Both the close channel - the anchor the amplitude was diagnosed on - and the four-channel
/// ratio, which is the primary metric, because a gain fitted to the four-channel objective has
/// to be judged on it.
fn write_amplitude_ratio(
    output: &Path,
    panels: &AmplitudePanels<'_>,
    horizon: usize,
) -> Result<()> {
    let curve = |label: String, values: &[f64]| ReportSeries {
        label,
        values: values.iter().map(|value| *value as f32).collect(),
    };
    let mut series = Vec::new();
    for split in &panels.scored {
        for (channel, name) in [(Some(CLOSE_CHANNEL), "close"), (None, "four-channel")] {
            let (uncalibrated, calibrated) = split.ratios(panels.applied, channel);
            series.push(curve(
                format!("{} {NEUTRAL} {name} ratio, uncalibrated", split.split),
                &uncalibrated,
            ));
            series.push(curve(
                format!("{} {NEUTRAL} {name} ratio, calibrated", split.split),
                &calibrated,
            ));
        }
    }
    series.push(ReportSeries {
        label: "persistence 1.0".to_owned(),
        values: vec![1.0; horizon],
    });
    let scored = panels
        .scored
        .iter()
        .map(|split| format!("{} {} origins", split.origins, split.split))
        .collect::<Vec<_>>()
        .join(", ");
    write_report(
        output.join("timexer_segment_amplitude_calibration.report.bin"),
        &Report {
            title: format!(
                "CausalPatch epoch {epoch} step {step} | does the out-of-sample amplitude calibration move the MSE ratio below persistence? | {scored}, each scored once and reported twice: the same moments with the gain and without it, which is exact algebra on one pass and not two passes; the gain was fitted on {} reserved calibration-partition origins ending {}, {tickers} tickers",
                panels.applied.blocks.calibration_origins,
                panels.applied.blocks.calibration_last_origin_ms,
                epoch = panels.epoch,
                step = panels.step,
                tickers = panels.tickers
            ),
            x_label: Some("bars ahead".to_owned()),
            y_label: Some(BEST_SCALE_UNIT.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: (1..=horizon as u64).collect(),
                series,
            },
        },
    )?;
    Ok(())
}

/// The two halves of a measured gain, so a gain above 1 can be diagnosed instead of trusted.
///
/// A least-squares gain is a ratio, and a ratio above 1 has two readings that no function of
/// the ratio alone can separate: a forecast that is genuinely too small, or a forecast whose
/// amplitude has collapsed so far that the quotient means nothing. This panel puts the
/// numerator and the denominator on one axis, each as a share of the same persistence MSE the
/// whole decomposition is measured against, beside what the amplitude error at that horizon
/// actually costs. The reading rule: an amplitude cost below the noise scale means the
/// horizon's gain is arithmetically real and economically empty, however far from 1 it sits.
fn write_calibration_moments(
    output: &Path,
    panels: &AmplitudePanels<'_>,
    horizon: usize,
) -> Result<()> {
    let (source, label, origins) = match (&panels.fit, panels.scored.first()) {
        (Some(fit), _) => (
            fit.moments,
            CALIBRATION,
            fit.calibration.blocks.calibration_origins,
        ),
        (None, Some(split)) => (split.moments, split.split, split.origins),
        (None, None) => bail!("a moment panel needs a population to measure"),
    };
    let (mut energy, mut covariance) = (Vec::with_capacity(horizon), Vec::with_capacity(horizon));
    for h in 0..horizon {
        let (anchor_square, anchor_target, persistence) = source.gain_moments(h, CLOSE_CHANNEL);
        let share = |moment: f64| {
            if persistence > 0. {
                (moment / persistence) as f32
            } else {
                f32::NAN
            }
        };
        energy.push(share(anchor_square));
        covariance.push(share(anchor_target));
    }
    let mut series = vec![
        ReportSeries {
            label: format!("{label} close anchor energy mean(C²) over mean(y²)"),
            values: energy,
        },
        ReportSeries {
            label: format!("{label} close anchor covariance mean(C·y) over mean(y²)"),
            values: covariance,
        },
    ];
    if let Some(fit) = &panels.fit {
        series.push(ReportSeries {
            label: format!("{CALIBRATION} amplitude cost of leaving the gain at 1"),
            values: fit
                .calibration
                .amplitude_cost
                .iter()
                .map(|cost| *cost as f32)
                .collect(),
        });
        series.push(ReportSeries {
            label: format!("{CALIBRATION} constant-forecast ceiling, the whole intercept opportunity"),
            values: fit
                .calibration
                .intercept_ceiling
                .iter()
                .map(|ceiling| *ceiling as f32)
                .collect(),
        });
    }
    write_report(
        output.join("timexer_segment_calibration_moments.report.bin"),
        &Report {
            title: format!(
                "CausalPatch epoch {} step {} | is a measured gain above 1 a real under-amplitude or a vanishing denominator? | the two moments behind the least-squares gain on {origins} {label} origins, each as a share of that horizon's own persistence MSE",
                panels.epoch, panels.step
            ),
            x_label: Some("bars ahead".to_owned()),
            y_label: Some(MOMENT_UNIT.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: (1..=horizon as u64).collect(),
                series,
            },
        },
    )?;
    Ok(())
}

/// The decoded channel names, in `decode_joint`'s emission order.
fn channel_name(channel: usize) -> &'static str {
    ["open", "high", "low", "close"][channel]
}

/// Units of the frozen-trunk latent probe family.
///
/// The decision panel's axis carries BOTH an IC and a difference of ICs, which is one unit and
/// one question - "does a linear read of the frozen latent beat the trained head, and by how
/// much" - so they share an axis on purpose. The two pre-registered thresholds are drawn as
/// reference lines because a verdict a reader has to compute from the chart is a verdict they
/// will get wrong.
const LATENT_PROBE_IC_UNIT: &str = "correlation coefficient (dimensionless; mean over \
                                    evaluation timestamps of the within-timestamp cross-ticker \
                                    correlation, and paired differences of the same, measured \
                                    on identical origins; a paired gap above the World A line \
                                    is unrouted information in the trunk, inside the World B \
                                    band is an information ceiling)";
/// The probe/head MSE comparison. Both sides carry an achieved and an oracle-rescaled ratio:
/// the achieved pair re-measures the known amplitude defect, the oracle pair is the only
/// amplitude-neutral MSE comparison, and reading either alone is how the amplitude defect gets
/// mistaken for an information difference.
const LATENT_PROBE_RATIO_UNIT: &str = "ratio vs close-anchored persistence (dimensionless; < 1 \
                                       = skill; the oracle series is the SAME forecast with its \
                                       amplitude corrected and nothing else changed, so probe \
                                       and head are only comparable on the oracle pair)";
/// The discount panel. A probe that only beats the head through an ill-conditioned inverse is
/// not evidence, so the conditioning of the system actually solved is a first-class series
/// rather than a footnote, beside the same system with no penalty at all.
const LATENT_PROBE_CONDITION_UNIT: &str = "condition number of the normal equations \
                                           (dimensionless, ratio of largest to smallest \
                                           retained eigenvalue; the solved series includes the \
                                           selected ridge, the unpenalized series does not, and \
                                           a large gap between them means the win rests on the \
                                           penalty)";

/// Everything the latent-probe panels state about their own populations.
pub struct LatentProbeContext {
    pub fit_origins: usize,
    pub penalty_origins: usize,
    pub purged_origins: usize,
    pub scored_origins: usize,
    /// Scored origins dropped so the whole scored block sits behind the fit block's last
    /// target bar. In the title because a reader has to know the scored draw is a strict
    /// subset of `held-out full` and by how much, not discover it later.
    pub outer_purged: usize,
    pub purge_gap_ms: i64,
    pub first_scored_ms: i64,
    pub last_scored_ms: i64,
    pub width: i64,
    pub verdict: String,
}

/// One series per forecaster over the probed horizons, NaN wherever the quantity was not
/// measured. Never 0: an unmeasured IC and a measured zero IC are opposite claims.
fn probe_series(
    report: &ProbeReport,
    label: impl Fn(&str) -> String,
    value: impl Fn(&HorizonScore) -> f64,
    probes_only: bool,
) -> Vec<ReportSeries> {
    report
        .forecasters
        .iter()
        .filter(|forecaster| !probes_only || forecaster.parameters > 0)
        .map(|forecaster| ReportSeries {
            label: label(&forecaster.label),
            values: forecaster
                .per_horizon
                .iter()
                .map(|score| value(score) as f32)
                .collect(),
        })
        .collect()
}

fn probe_reference(label: &str, value: f64, horizons: usize) -> ReportSeries {
    ReportSeries {
        label: label.to_owned(),
        values: vec![value as f32; horizons],
    }
}

/// The verdict panel: does a linear read of the frozen trunk latent beat the trained head?
///
/// This is the falsifier for every latent-objective proposal. Oracle per-horizon rescaling is
/// worth about 2.4% of MSE at h = 192 and zero IC, so amplitude repair is exhausted and any new
/// long-horizon edge has to be information. If the probe beats the head materially, the trunk
/// holds information the dense 192-row head does not route and a latent objective has something
/// to collect; if it ties, no auxiliary loss on latents can manufacture edge and the levers are
/// context, features, capacity or the target.
///
/// The probe sees a STRICT SUBSET of the head's input - the latent without the known-future
/// covariate block - so a probe win cannot be explained by extra inputs, which is why the
/// comparison is drawn on one axis with the head as a first-class series rather than as a
/// remembered number from another panel.
pub fn write_latent_probe(
    output: &Path,
    epoch: usize,
    step: usize,
    report: &ProbeReport,
    context: &LatentProbeContext,
) -> Result<()> {
    let horizons = report.horizons.len();
    ensure!(horizons > 0, "a latent probe panel needs at least one horizon");
    let mut series = probe_series(
        report,
        |label| format!("{FULL} {NEUTRAL} close cross-sectional IC, {label}"),
        |score| score.ic,
        false,
    );
    series.extend(probe_series(
        report,
        |label| format!("{FULL} paired IC of {label} minus the trained head"),
        |score| score.gap,
        true,
    ));
    series.extend(probe_series(
        report,
        |label| format!("{FULL} standard error of that paired difference, {label}"),
        |score| score.gap_se,
        true,
    ));
    series.push(probe_reference(
        "World A threshold, paired gain of 0.010 justifies a latent objective",
        probe::WORLD_A_GAIN,
        horizons,
    ));
    series.push(probe_reference(
        "World B band upper edge 0.005, inside it the latent line is retired",
        probe::WORLD_B_BAND,
        horizons,
    ));
    series.push(probe_reference(
        "World B band lower edge minus 0.005",
        -probe::WORLD_B_BAND,
        horizons,
    ));
    series.push(probe_reference("zero information 0.0", 0., horizons));
    write_report(
        output.join("timexer_segment_latent_probe.report.bin"),
        &Report {
            title: format!(
                "CausalPatch epoch {epoch} step {step} | does the frozen trunk latent linearly predict the long horizon BETTER than the trained head does? | {} untouched {FULL} origins from {} to {} ({} dropped to seat the scored block behind every fit target), probes fitted in closed form on {} reserved calibration origins ({} more spent choosing the ridge, {} purged) whose targets all complete {} ms before the first scored origin | verdict {}",
                context.scored_origins,
                context.first_scored_ms,
                context.last_scored_ms,
                context.outer_purged,
                context.fit_origins,
                context.penalty_origins,
                context.purged_origins,
                context.purge_gap_ms,
                context.verdict
            ),
            x_label: Some("bars ahead".to_owned()),
            y_label: Some(LATENT_PROBE_IC_UNIT.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: report.horizons.iter().map(|h| *h as u64).collect(),
                series,
            },
        },
    )?;
    Ok(())
}

/// The same comparison in MSE, with amplitude neutralized on both sides.
pub fn write_latent_probe_ratio(
    output: &Path,
    epoch: usize,
    step: usize,
    report: &ProbeReport,
    context: &LatentProbeContext,
) -> Result<()> {
    let horizons = report.horizons.len();
    ensure!(horizons > 0, "a latent probe ratio panel needs at least one horizon");
    let mut series = probe_series(
        report,
        |label| format!("{FULL} {NEUTRAL} close ratio as emitted, {label}"),
        |score| score.mse_ratio,
        false,
    );
    series.extend(probe_series(
        report,
        |label| format!("{FULL} {NEUTRAL} close ratio oracle rescaled, {label}"),
        |score| score.oracle_mse_ratio,
        false,
    ));
    series.push(probe_reference("persistence 1.0", 1., horizons));
    write_report(
        output.join("timexer_segment_latent_probe_ratio.report.bin"),
        &Report {
            title: format!(
                "CausalPatch epoch {epoch} step {step} | with amplitude neutralized on both sides, does the latent probe beat the head on MSE? | the same {} {FULL} origins the IC panel scores, probe and head reduced from one forward pass each",
                context.scored_origins
            ),
            x_label: Some("bars ahead".to_owned()),
            y_label: Some(LATENT_PROBE_RATIO_UNIT.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: report.horizons.iter().map(|h| *h as u64).collect(),
                series,
            },
        },
    )?;
    Ok(())
}

/// The discount panel, so a win can be refused rather than believed.
pub fn write_latent_probe_conditioning(
    output: &Path,
    epoch: usize,
    step: usize,
    report: &ProbeReport,
    context: &LatentProbeContext,
) -> Result<()> {
    let horizons = report.horizons.len();
    ensure!(horizons > 0, "a conditioning panel needs at least one horizon");
    let mut series = probe_series(
        report,
        |label| format!("{label}, condition number of the solved system"),
        |score| score.condition,
        true,
    );
    series.extend(probe_series(
        report,
        |label| format!("{label}, condition number with no penalty"),
        |score| score.raw_condition,
        true,
    ));
    write_report(
        output.join("timexer_segment_latent_probe_conditioning.report.bin"),
        &Report {
            title: format!(
                "CausalPatch epoch {epoch} step {step} | is the probe's advantage an inverted near-singularity? | conditioning of the {}-dimensional centred latent second moment on {} reserved calibration origins, per probe class, fitted parameter counts in the series labels",
                context.width,
                context.fit_origins + context.penalty_origins
            ),
            x_label: Some("bars ahead".to_owned()),
            y_label: Some(LATENT_PROBE_CONDITION_UNIT.to_owned()),
            scale: ScaleKind::Symlog,
            kind: ReportKind::IndexedLines {
                steps: report.horizons.iter().map(|h| *h as u64).collect(),
                series,
            },
        },
    )?;
    Ok(())
}

/// The extraction scaling panel: is a probe that fails to beat the head at an INFORMATION
/// ceiling or at a SAMPLE-SIZE one?
///
/// The x axis is realized fit origins, not the nominal share, because cross-sections are
/// unequal in width and the power law must be fitted against the number that existed. Three
/// series on one axis, all IC: the measured rung, the fitted curve through the rungs, and the
/// extrapolated asymptote as a flat reference. A rung still climbing towards a distant
/// asymptote says the probe is fit-limited and a World B reading off the top rung alone would
/// be premature.
pub fn write_latent_probe_scaling(
    output: &Path,
    epoch: usize,
    step: usize,
    curve: &probe::ScalingCurve,
    context: &LatentProbeContext,
) -> Result<()> {
    ensure!(!curve.rungs.is_empty(), "a scaling panel needs at least one rung");
    let steps: Vec<u64> = curve.rungs.iter().map(|rung| rung.origins as u64).collect();
    let fitted: Vec<f32> = curve
        .rungs
        .iter()
        .map(|rung| {
            (curve.asymptote - curve.amplitude * (rung.origins as f64).powf(-curve.exponent)) as f32
        })
        .collect();
    let series = vec![
        ReportSeries {
            label: format!("{FULL} {NEUTRAL} close cross-sectional IC at h 1, ridge probe measured"),
            values: curve.rungs.iter().map(|rung| rung.ic as f32).collect(),
        },
        ReportSeries {
            label: format!(
                "fitted IC_inf minus a times N to the minus b, b {:.4}, residual {:.5}",
                curve.exponent, curve.residual
            ),
            values: fitted,
        },
        ReportSeries {
            label: format!("extrapolated asymptote at infinite fit sample {:+.5}", curve.asymptote),
            values: vec![curve.asymptote as f32; curve.rungs.len()],
        },
    ];
    write_report(
        output.join("timexer_segment_latent_probe_scaling.report.bin"),
        &Report {
            title: format!(
                "CausalPatch epoch {epoch} step {step} | is the probe limited by the latent's INFORMATION or by its own FIT SAMPLE? | h 1 ridge probes on {} nested whole-timestamp subsets of the reserved calibration partition, each scored on the same {} untouched {FULL} origins, penalty selected on the same held-back block at every rung | asymptote {:+.5} approached with exponent {:.4}",
                curve.rungs.len(),
                context.scored_origins,
                curve.asymptote,
                curve.exponent
            ),
            x_label: Some("coefficient-fit origins".to_owned()),
            y_label: Some(LATENT_PROBE_IC_UNIT.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines { steps, series },
        },
    )?;
    Ok(())
}

/// How much of the probe's edge is RECENCY rather than latent information.
///
/// The probe is fitted on a window strictly more recent than anything the trained head saw, so
/// under a non-stationary market part of any probe win is an advantage the head does not have.
/// This panel prices it: equal-population chronological tranches of the fit partition, each
/// scored on the same held-out block, so what varies across the series is fit-block AGE with fit
/// sample size held constant. The x axis is the probed horizon, because the discount enters the
/// verdict per horizon; the tranches are the series.
pub fn write_latent_probe_recency(
    output: &Path,
    epoch: usize,
    step: usize,
    recency: &probe::Recency,
    context: &LatentProbeContext,
) -> Result<()> {
    ensure!(!recency.horizons.is_empty(), "a recency panel needs at least one horizon");
    let mut series: Vec<ReportSeries> = recency
        .tranches
        .iter()
        .enumerate()
        .map(|(index, (origins, mid))| ReportSeries {
            label: format!(
                "{FULL} {NEUTRAL} close cross-sectional IC, probe fitted on {origins} origins centred at {mid} ms"
            ),
            values: recency.tranche_ic.iter().map(|row| row[index] as f32).collect(),
        })
        .collect();
    series.push(ReportSeries {
        label: format!(
            "recency discount added to the World A bar at {:.3} years of head age deficit",
            recency.age_gap_years
        ),
        values: recency.discount.iter().map(|value| *value as f32).collect(),
    });
    write_report(
        output.join("timexer_segment_latent_probe_recency.report.bin"),
        &Report {
            title: format!(
                "CausalPatch epoch {epoch} step {step} | how much of the probe's edge is RECENCY rather than latent information? | {} equal-population chronological tranches of the reserved calibration partition, fit sample size held constant, each scored on the same {} untouched {FULL} origins | the head's targets end {:.3} years before the fit block's centre, and the World A bar is raised by the discount series",
                recency.tranches.len(),
                context.scored_origins,
                recency.age_gap_years
            ),
            x_label: Some("bars ahead".to_owned()),
            y_label: Some(LATENT_PROBE_IC_UNIT.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: recency.horizons.iter().map(|h| *h as u64).collect(),
                series,
            },
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::report::read_report;
    use std::fs;

    #[test]
    fn utility_reports_preserve_costed_payoffs_cash_and_missing_evidence() {
        use tch::{Device, Kind, Tensor};
        let root = std::env::temp_dir().join(format!("timexer-utility-{}", uuid::Uuid::new_v4()));
        let cpu = (Kind::Double, Device::Cpu);
        let forecast = Tensor::full([20, 1], 0.01, cpu);
        let scale = Tensor::full([20, 1], 0.02_f64.ln(), cpu);
        let target = Tensor::full([20, 1], 1.02_f64.ln(), cpu);
        let entry = Tensor::full([20], 1.01_f64.ln(), cpu);
        let sigma = Tensor::ones([20], cpu);
        let groups = Tensor::zeros([20], (Kind::Int64, Device::Cpu));
        let evaluate = |valid: Tensor| super::super::utility::evaluate(
            &forecast, &scale, &target, &valid, &sigma, &entry, &groups, 1,
        ).unwrap();
        let complete = evaluate(Tensor::ones([20, 1], cpu));
        let missing = evaluate(Tensor::zeros([20, 1], cpu));
        let curve = trading_curve(1);
        write_trading(
            &root, 1, 1000,
            Some(TradingSplit { curve: &curve, portfolio: &missing, origins: 20 }),
            Some(TradingSplit { curve: &curve, portfolio: &complete, origins: 20 }),
            None, 20,
        ).unwrap();
        let value = |base: &str, label: &str| {
            let report = read_report(&root.join(format!("{base}.report.bin"))).unwrap();
            let ReportKind::IndexedLines { steps, series } = report.kind else {
                panic!("utility outcomes must retain their decision horizons");
            };
            assert_eq!(steps, vec![1]);
            series.iter().find(|series| series.label == label).unwrap().values[0] as f64
        };
        let gross = (1.02 / 1.01 - 1.) * 1e4;
        let net = gross - 10. * (1. + 1.02 / 1.01);
        assert!((value("timexer_segment_utility_payoff", "held-out full equal long") - gross).abs() < 1e-4);
        assert!((value("timexer_segment_utility_payoff_cost_10", "held-out full equal long") - net).abs() < 1e-4);
        assert_eq!(value("timexer_segment_utility_payoff_cost_10", "held-out full cash"), 0.);
        assert!(value("timexer_segment_utility_payoff", "held-out sample equal long").is_nan());
        assert!(value("timexer_segment_utility_breakeven", "held-out full cash").is_nan());
        fs::remove_dir_all(root).unwrap();
    }

    fn horizon_curve(width: usize) -> HorizonCurve {
        let ramp = |scale: f64| (1..=width).map(|h| scale * h as f64).collect::<Vec<_>>();
        HorizonCurve {
            mse: ramp(0.9),
            persistence_mse: ramp(1.0),
            absolute_mse: ramp(1.8),
            absolute_persistence_mse: ramp(2.0),
            mae_ratio: vec![0.97; width],
            win_rate: vec![0.51; width],
            hit_rate: vec![0.52; width],
            up_fraction: vec![0.5; width],
            trimmed_mse_ratio: vec![0.98; width],
            // Per-index so a mis-slice is visible: coverage falls with the horizon, which is
            // the direction an over-dispersed long end would show.
            within_1_sigma: (1..=width).map(|h| 0.7 - 0.0005 * h as f64).collect(),
            within_2_sigma: (1..=width).map(|h| 0.96 - 0.0002 * h as f64).collect(),
            valid_elements: (1..=width).map(|h| (4096 - 4 * h) as f64).collect(),
        }
    }

    fn trading_curve(width: usize) -> TradingCurve {
        let zeros = vec![0.; width];
        TradingCurve {
            total_gain: zeros.clone(),
            all_channel_gain: zeros.clone(),
            offset_gain: zeros.clone(),
            demeaned_gain: zeros.clone(),
            // The quantity under test: index 0 is h = 1, so a per-index value makes a
            // mis-slice visible instead of averaging away.
            scaling_gain: (1..=width).map(|h| -0.001 * h as f64).collect(),
            forecast_variance: (1..=width).map(|h| 0.01 * h as f64).collect(),
            covariance: (1..=width).map(|h| 0.004 * h as f64).collect(),
            persistence: (1..=width).map(|h| h as f64).collect(),
            mean_forecast: zeros.clone(),
            mean_target: zeros.clone(),
            // Finite pooled Pearson beside an UNDEFINED within-timestamp IC, which is what a
            // strided one-origin-per-timestamp draw really produces: the cross-section never
            // fired, so the IC and its standard error are NaN and the moment count is 0.
            pearson: (1..=width).map(|h| 0.02 - 0.00005 * h as f64).collect(),
            spearman: zeros.clone(),
            cross_sectional_ic: vec![f64::NAN; width],
            cross_sectional_ic_se: vec![f64::NAN; width],
            close_mse_ratio: (1..=width).map(|h| 0.995 + 0.0002 * h as f64).collect(),
            mid_anchor_mse_ratio: zeros.clone(),
            delayed_mse_ratio: zeros.clone(),
            close_hit_rate: zeros.clone(),
            mid_anchor_hit_rate: zeros.clone(),
            delayed_hit_rate: zeros.clone(),
            top_decile_hit_rate: zeros.clone(),
            bottom_decile_hit_rate: zeros.clone(),
            top_decile_return: zeros.clone(),
            bottom_decile_return: zeros.clone(),
            conviction_spread_return: zeros,
            // Per-index, and the three that a mis-slice would silently average away: the
            // optimal gain is the amplitude diagnostic, the best-scale ratio the bound it
            // implies, and the moment count the IC's own population.
            optimal_gain: (1..=width).map(|h| 1. - 0.004 * h as f64).collect(),
            best_scale_mse_ratio: (1..=width).map(|h| 0.999 - 0.0001 * h as f64).collect(),
            cross_sectional_ic_moments: vec![0.; width],
            cross_sections: 8,
        }
    }

    /// The `held-out cross-section` draw's curve: whole timestamp blocks, so the IC that the
    /// sample draw leaves undefined is measured here. Per-index values, so a series that reads
    /// the wrong horizon or the wrong split is a wrong number rather than a wrong shape.
    fn cross_curve(width: usize) -> TradingCurve {
        TradingCurve {
            cross_sectional_ic: (1..=width).map(|h| 0.08 - 0.0001 * h as f64).collect(),
            cross_sectional_ic_se: vec![0.0028; width],
            cross_sectional_ic_moments: vec![40.; width],
            ..trading_curve(width)
        }
    }

    fn point(step: usize, full: bool, train_nll: f64, width: usize) -> Metrics {
        Metrics {
            step,
            epoch: 1,
            completed_origins: 10,
            total_origins: 100,
            completed_target_bars: 20,
            total_target_bars: 200,
            train_nll: Some(train_nll),
            train_mse: Some(0.9),
            validation_nll: 2.0,
            validation_objective_nll: 1.9,
            persistence_nll: 2.4,
            validation_mse: 0.98,
            persistence_mse: 1.,
            absolute_mse: 1.96,
            absolute_persistence_mse: 2.,
            within_1_sigma: 0.68,
            within_2_sigma: 0.95,
            rmse_price: 1.5,
            mae_price: 1.,
            invalid_ohlc_fraction: 0.,
            tail_loss_share: 0.3,
            median_window_ratio: 0.99,
            eval: EvalTiming::default(),
            step_ms: Some(168.6),
            loader_wait_ms: Some(1.),
            loader_build_ms: Some(48.),
            peak_allocator_mib: Some(1024.),
            evaluation_peak_mib: Some(512.),
            capture_budget: None,
            validation_is_full: full,
            validation_origins: 256,
            tickers: 42,
            step_phases: None,
            horizons: horizon_track(&horizon_curve(width), &trading_curve(width)),
            cross_horizons: horizon_track(&horizon_curve(width), &cross_curve(width)),
            in_period_horizons: Vec::new(),
            in_period_origins: 0,
            in_period_purged_row_share: 0.,
            cross_origins: if width == 0 { 0 } else { 512 },
            startup: super::super::cache::LoadTiming::default(),
        }
    }

    /// A decision horizon past the evaluated window is absent, not a NaN row: the slice has
    /// to be readable as "this run never scored h = 192", which a written gap cannot say.
    #[test]
    fn the_horizon_track_slices_one_based_horizons_and_skips_the_unevaluated_ones() {
        let track = horizon_track(&horizon_curve(64), &trading_curve(64));
        assert_eq!(
            track.iter().map(|p| p.horizon).collect::<Vec<_>>(),
            vec![1, 8, 16, 32, 64]
        );
        let h64 = track.last().unwrap();
        assert!((h64.neutral_ratio - 0.9).abs() < 1e-12);
        assert!((h64.raw_ratio - 0.9).abs() < 1e-12);
        assert!(
            (h64.scaling_gain - -0.064).abs() < 1e-12,
            "h = 64 must read index 63, got {}",
            h64.scaling_gain
        );
        // Every quantity is sliced at the SAME index, so one mis-slice cannot hide behind a
        // neighbour: each fixture ramp is distinguishable at h = 64.
        assert!((h64.optimal_gain - (1. - 0.004 * 64.)).abs() < 1e-12);
        assert!((h64.best_scale_ratio - (0.999 - 0.0001 * 64.)).abs() < 1e-12);
        assert!((h64.close_ratio - (0.995 + 0.0002 * 64.)).abs() < 1e-12);
        assert!((h64.within_1_sigma - (0.7 - 0.0005 * 64.)).abs() < 1e-12);
        assert!((h64.within_2_sigma - (0.96 - 0.0002 * 64.)).abs() < 1e-12);
        assert_eq!(h64.valid_elements, (4096 - 4 * 64) as f64);
        assert!((h64.pooled_pearson - (0.02 - 0.00005 * 64.)).abs() < 1e-12);
        // The sample draw never fires a cross-section: NaN and a population count of 0, which
        // is a different statement from "the IC was measured at 0".
        assert!(h64.cross_sectional_ic.is_nan() && h64.cross_sectional_ic_se.is_nan());
        assert_eq!(h64.cross_sections, 0.);
        let measured = horizon_track(&horizon_curve(64), &cross_curve(64));
        let h64 = measured.last().unwrap();
        assert!((h64.cross_sectional_ic - (0.08 - 0.0001 * 64.)).abs() < 1e-12);
        assert_eq!(h64.cross_sections, 40.);
        assert_eq!(horizon_track(&horizon_curve(4), &trading_curve(4)).len(), 1);
    }

    /// The step family is indexed on the endpoint-utility grid, by construction of this
    /// assertion rather than by coincidence. The question that decides adoption - does the IC
    /// at the horizon a policy trades survive training - is only answerable if the signal
    /// family and the utility family name the same horizons.
    #[test]
    fn the_decision_horizons_are_the_utility_holding_periods() {
        assert_eq!(
            DECISION_HORIZONS
                .iter()
                .map(|&h| h as u64)
                .collect::<Vec<_>>(),
            super::super::utility::HORIZONS.to_vec()
        );
    }

    /// The step-indexed transposes and the gap, end to end through the file format: labels,
    /// step axis, split routing and values as a reader of the `.report.bin` sees them.
    #[test]
    fn the_step_indexed_horizon_and_gap_panels_carry_matched_step_values() {
        let root = std::env::temp_dir()
            .join(format!("timexer-horizon-steps-{}", uuid::Uuid::new_v4()));
        let points = vec![
            point(1000, false, 2.25, 192),
            point(2000, false, 2.08, 192),
            point(3000, true, 2.05, 192),
        ];
        write_metrics(&root, &points).unwrap();

        let gap = read_report(root.join("timexer_segment_generalization_gap.report.bin")).unwrap();
        let ReportKind::IndexedLines { steps, series } = &gap.kind else {
            panic!("the gap panel must be step-indexed");
        };
        assert_eq!(steps, &vec![1000, 2000, 3000]);
        let sample = series
            .iter()
            .find(|s| s.label == format!("training minus {SAMPLE} objective NLL"))
            .expect("the held-out sample gap is the in-epoch series");
        // The unweighted NLL is 2.0, but the matched weighted objective is 1.9.
        assert!((sample.values[0] - 0.35).abs() < 1e-6);
        assert!((sample.values[1] - 0.18).abs() < 1e-6);
        assert!(sample.values[2].is_nan());
        let full = series
            .iter()
            .find(|s| s.label == format!("training minus {FULL} objective NLL"))
            .expect("the epoch-end gap is its own series");
        assert!(full.values[0].is_nan() && full.values[1].is_nan());
        assert!((full.values[2] - 0.15).abs() < 1e-6);

        let ratios = read_report(root.join("timexer_segment_horizon_steps.report.bin")).unwrap();
        assert_eq!(ratios.y_label.as_deref(), Some(RATIO_UNIT));
        let ReportKind::IndexedLines { steps, series } = &ratios.kind else {
            panic!("the per-horizon step panel must be step-indexed");
        };
        assert_eq!(steps, &vec![1000, 2000, 3000]);
        for horizon in DECISION_HORIZONS {
            for space in [NEUTRAL, RAW] {
                let label = format!("{SAMPLE} {space} four-channel MSE ratio at horizon {horizon}");
                let found = series
                    .iter()
                    .find(|s| s.label == label)
                    .unwrap_or_else(|| panic!("{label} missing"));
                assert!((found.values[0] - 0.9).abs() < 1e-6);
                assert!(found.values[2].is_nan(), "{label} leaked across splits");
            }
        }
        assert!(series.iter().any(|s| s.label == "parity 1.0"));

        let decomposition =
            read_report(root.join("timexer_segment_horizon_steps_decomposition.report.bin"))
                .unwrap();
        assert_eq!(decomposition.y_label.as_deref(), Some(DECOMPOSITION_UNIT));
        let ReportKind::IndexedLines { series, .. } = &decomposition.kind else {
            panic!("the decomposition panel must be step-indexed");
        };
        let h64 = series
            .iter()
            .find(|s| {
                s.label
                    == format!(
                        "{SAMPLE} {NEUTRAL} close gain lost to forecast mis-scaling (cross term) \
                         at horizon 64"
                    )
            })
            .expect("h = 64 is the horizon that identified the over-amplitude mean");
        assert!((h64.values[1] - -0.064).abs() < 1e-6);

        // Every base this call produced is registered, so no panel is written and invisible.
        for entry in fs::read_dir(&root).unwrap() {
            let path = entry.unwrap().path();
            let base = path.file_stem().unwrap().to_str().unwrap();
            let base = base.strip_suffix(".report").unwrap_or(base);
            assert!(
                shared::report::TIMEXER_SEGMENT_REPORT_BASES.contains(&base),
                "{base} is written by write_metrics but unregistered"
            );
        }
        fs::remove_dir_all(&root).unwrap();
    }

    /// Without a scored curve the step-indexed horizon panels must not exist at all: a
    /// parity-only panel reads as "the ratio sits exactly at 1", which is a false statement.
    #[test]
    fn the_per_horizon_step_panels_are_absent_when_no_point_scored_a_curve() {
        let root =
            std::env::temp_dir().join(format!("timexer-horizon-empty-{}", uuid::Uuid::new_v4()));
        let bare = point(500, false, 2.2, 0);
        write_metrics(&root, &[bare]).unwrap();
        for base in [
            "timexer_segment_horizon_steps",
            "timexer_segment_horizon_steps_signal",
            "timexer_segment_horizon_steps_signal_error",
            "timexer_segment_horizon_steps_pooled",
            "timexer_segment_horizon_steps_population",
            "timexer_segment_horizon_steps_decomposition",
            "timexer_segment_horizon_steps_gain",
            "timexer_segment_horizon_steps_best_scale",
            "timexer_segment_horizon_steps_calibration",
        ] {
            assert!(
                !root.join(format!("{base}.report.bin")).exists(),
                "{base} must not exist as a reference-line-only panel"
            );
        }
        assert!(root
            .join("timexer_segment_generalization_gap.report.bin")
            .exists());
        fs::remove_dir_all(&root).unwrap();
    }

    /// A `--max-steps` run's terminal step carries two measurements: the interval's `held-out
    /// sample` evaluation and the final full-split pass. They must land as two SERIES on one
    /// x, never as two rows sharing an x, because a duplicated step is exactly how a reader
    /// mistakes one split's number for another's - and every other collision must stay
    /// refused, or this relaxation would have deleted the invariant instead of narrowing it.
    #[test]
    fn a_capped_terminal_step_carries_both_splits_as_series_on_one_step() {
        let root =
            std::env::temp_dir().join(format!("timexer-capped-step-{}", uuid::Uuid::new_v4()));
        let mut sample = point(4000, false, 2.08, 192);
        sample.validation_nll = 2.11;
        let mut full = point(4000, true, 2.08, 192);
        full.validation_nll = 2.02;
        full.validation_origins = 433303;
        let points = vec![point(1000, false, 2.25, 192), sample.clone(), full.clone()];
        write_metrics(&root, &points).unwrap();

        let loss = read_report(root.join("timexer_segment_loss.report.bin")).unwrap();
        let ReportKind::IndexedLines { steps, series } = &loss.kind else {
            panic!("the loss panel must be step-indexed");
        };
        assert_eq!(steps, &[1000, 4000], "the capped step must appear once");
        let value = |label: String, index: usize| {
            series
                .iter()
                .find(|s| s.label == label)
                .unwrap_or_else(|| panic!("{label} is missing"))
                .values[index]
        };
        assert_eq!(value(format!("{SAMPLE} unweighted NLL"), 1), 2.11);
        assert_eq!(value(format!("{FULL} unweighted NLL"), 1), 2.02);
        assert!(value(format!("{FULL} unweighted NLL"), 0).is_nan());
        assert_eq!(value(format!("{SAMPLE} unweighted NLL"), 0), 2.0);
        assert_eq!(value(format!("{SAMPLE} objective NLL"), 1), 1.9);
        assert_eq!(value(format!("{FULL} objective NLL"), 1), 1.9);
        // Split-agnostic series read the step's own point once, not twice.
        assert_eq!(value("training objective NLL".to_owned(), 1), 2.08);
        // Both populations are named in the scope, so the full-split point is legible as the
        // 433,303-origin pass it is rather than as another sample.
        assert!(loss.title.contains("433303"), "{}", loss.title);

        let ratios = read_report(root.join("timexer_segment_horizon_steps.report.bin")).unwrap();
        let ReportKind::IndexedLines { steps, series } = &ratios.kind else {
            panic!("the per-horizon step panel must be step-indexed");
        };
        assert_eq!(steps, &[1000, 4000]);
        for split in [SAMPLE, FULL] {
            let label = format!("{split} {NEUTRAL} four-channel MSE ratio at horizon 1");
            let track = series.iter().find(|s| s.label == label).unwrap();
            assert!(
                track.values[1].is_finite(),
                "{label} must be measured at the capped step"
            );
            assert_eq!(track.values[0].is_nan(), split == FULL);
        }

        // The narrowing, pinned: only a sample point followed by a full point may share a
        // step. A repeated split at one step, a sample point after a full one, and a step
        // going backwards are all still refused.
        for bad in [
            vec![sample.clone(), sample.clone()],
            vec![full.clone(), full.clone()],
            vec![full, sample],
            vec![point(2000, false, 2.2, 192), point(1000, false, 2.2, 192)],
        ] {
            let error = write_metrics(&root, &bad).unwrap_err().to_string();
            assert!(error.contains("steps must increase"), "{error}");
        }
        fs::remove_dir_all(&root).unwrap();
    }

    /// The property the whole family exists for: a base written at every report interval
    /// RETAINS every evaluation, so the per-horizon IC is a trajectory rather than the last
    /// evaluation's snapshot. This is the failure the horizon-indexed bases have - each
    /// evaluation overwrites the previous curve - and it is why "is the long-horizon IC rising
    /// or falling" was unanswerable from disk.
    #[test]
    fn the_signal_panel_retains_every_evaluation_rather_than_the_latest() {
        let root =
            std::env::temp_dir().join(format!("timexer-signal-steps-{}", uuid::Uuid::new_v4()));
        // Three intervals, each with its own cross-section pass, and a rising IC at h = 64:
        // the movement is the whole reading, so a panel that kept only the last evaluation
        // would show 0.0736 as a flat line and answer the question wrongly rather than not at
        // all.
        let points: Vec<Metrics> = [(1000usize, 1.0_f64), (2000, 1.5), (3000, 2.0)]
            .into_iter()
            .map(|(step, scale)| {
                let mut metrics = point(step, false, 2.2, 192);
                metrics.cross_horizons = horizon_track(&horizon_curve(192), &{
                    let mut curve = cross_curve(192);
                    curve.cross_sectional_ic =
                        curve.cross_sectional_ic.iter().map(|ic| ic * scale).collect();
                    curve
                });
                metrics
            })
            .collect();
        write_metrics(&root, &points).unwrap();

        let signal = read_report(root.join("timexer_segment_horizon_steps_signal.report.bin"))
            .unwrap();
        assert_eq!(signal.y_label.as_deref(), Some(IC_UNIT));
        let ReportKind::IndexedLines { steps, series } = &signal.kind else {
            panic!("the signal panel must be step-indexed");
        };
        assert_eq!(steps, &vec![1000, 2000, 3000]);
        let ic = series
            .iter()
            .find(|s| s.label == format!("{CROSS} close cross-sectional IC at horizon 64"))
            .expect("the cross-section pass is the draw the IC is measured on");
        let base = 0.08 - 0.0001 * 64.;
        for (index, scale) in [1.0_f32, 1.5, 2.0].into_iter().enumerate() {
            assert!(
                (ic.values[index] - base as f32 * scale).abs() < 1e-6,
                "evaluation {index} was overwritten: {:?}",
                ic.values
            );
        }
        // The population count is a series of its own, at every step, so a thinning draw is
        // separable from a decaying IC.
        let population =
            read_report(root.join("timexer_segment_horizon_steps_population.report.bin")).unwrap();
        let ReportKind::IndexedLines { series, .. } = &population.kind else {
            panic!("the population panel must be step-indexed");
        };
        let moments = series
            .iter()
            .find(|s| s.label == format!("{CROSS} contributing cross-sections at horizon 64"))
            .expect("the IC's own population is charted");
        assert_eq!(moments.values, vec![40.; 3]);
        // Every step-indexed quantity keeps its whole history, not just the decision metric.
        for (base, label, expected) in [
            (
                "timexer_segment_horizon_steps_gain",
                format!("{CROSS} close MSE-optimal forecast gain at horizon 192"),
                1. - 0.004 * 192.,
            ),
            (
                "timexer_segment_horizon_steps_best_scale",
                format!("{CROSS} {NEUTRAL} close MSE ratio at the best scale at horizon 192"),
                0.999 - 0.0001 * 192.,
            ),
            (
                "timexer_segment_horizon_steps_calibration",
                format!("{CROSS} within 1σ at horizon 192"),
                0.7 - 0.0005 * 192.,
            ),
        ] {
            let report = read_report(root.join(format!("{base}.report.bin"))).unwrap();
            let ReportKind::IndexedLines { series, .. } = &report.kind else {
                panic!("{base} must be step-indexed");
            };
            let found = series
                .iter()
                .find(|s| s.label == label)
                .unwrap_or_else(|| panic!("{label} missing from {base}"));
            assert_eq!(found.values.len(), 3, "{base} dropped an evaluation");
            assert!(found.values.iter().all(|v| (v - expected as f32).abs() < 1e-6));
        }
        fs::remove_dir_all(&root).unwrap();
    }

    /// An unmeasured cross-sectional statistic must reach a reader as a gap and never as 0.
    /// The `held-out sample` draw strides one origin per timestamp, so its within-timestamp IC
    /// never fires at any horizon; a 0 there would read as "measured, and there is no
    /// information", which is the opposite of what the draw supports. The series is dropped
    /// entirely rather than drawn flat at 0, and the population count of 0 is what explains
    /// the absence.
    #[test]
    fn an_unmeasured_cross_section_reads_as_absent_and_never_as_zero() {
        let root = std::env::temp_dir().join(format!("timexer-signal-nan-{}", uuid::Uuid::new_v4()));
        let mut points = vec![point(1000, false, 2.2, 192), point(2000, false, 2.1, 192)];
        // No cross-section pass at all at the second step: the split's own row is missing, not
        // zero.
        points[1].cross_horizons = Vec::new();
        points[1].cross_origins = 0;
        write_metrics(&root, &points).unwrap();

        let signal =
            read_report(root.join("timexer_segment_horizon_steps_signal.report.bin")).unwrap();
        let ReportKind::IndexedLines { series, .. } = &signal.kind else {
            panic!("the signal panel must be step-indexed");
        };
        assert!(
            !series
                .iter()
                .any(|s| s.label.starts_with(SAMPLE) && s.label.contains("cross-sectional IC")),
            "a structurally undefined IC must not be drawn at all"
        );
        let cross = series
            .iter()
            .find(|s| s.label == format!("{CROSS} close cross-sectional IC at horizon 64"))
            .expect("the cross-section pass measured the first step");
        assert!(cross.values[0].is_finite());
        assert!(
            cross.values[1].is_nan(),
            "a step with no cross-section pass must be a gap, got {}",
            cross.values[1]
        );
        // The pooled correlation IS finite on the sample draw, and lives on its own base so it
        // cannot be mistaken for a measured cross-sectional IC.
        let pooled =
            read_report(root.join("timexer_segment_horizon_steps_pooled.report.bin")).unwrap();
        assert_eq!(pooled.y_label.as_deref(), Some(POOLED_UNIT));
        let ReportKind::IndexedLines { series, .. } = &pooled.kind else {
            panic!("the pooled panel must be step-indexed");
        };
        let sample = series
            .iter()
            .find(|s| s.label == format!("{SAMPLE} close pooled Pearson IC at horizon 64"))
            .expect("the pooled statistic is defined on every draw");
        assert!((sample.values[0] - (0.02 - 0.00005 * 64.) as f32).abs() < 1e-6);
        // A horizon the run never reached is absent from the track, so its series is dropped
        // rather than written as a row of zeros.
        let short = point(3000, false, 2.0, 32);
        assert!(short.horizons.iter().all(|h| h.horizon <= 32));
        fs::remove_dir_all(&root).unwrap();
    }
}

#[cfg(test)]
mod horizon_loss_weight_tests {
    use super::*;
    use shared::report::read_report;
    use std::fs;

    /// The chart has to say what the objective weights and refuse to say anything else. The
    /// normalization guard is the load-bearing part: an unnormalized vector would still render,
    /// and a reader comparing a `mean 1` label against a curve averaging 6 has no way to tell
    /// which one is lying.
    #[test]
    fn the_horizon_weight_chart_states_the_weighting_and_refuses_an_unnormalized_one() {
        let root = std::env::temp_dir()
            .join(format!("timexer-horizon-weight-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("timexer_segment_horizon_loss_weight.report.bin");
        // `cutoff:2` over eight horizons: mean 1 means the two trained horizons carry 4.
        let weights = vec![4., 4., 0., 0., 0., 0., 0., 0.];
        write_horizon_loss_weight(&root, 1, 2000, "cutoff:2", &weights).unwrap();
        let report = read_report(&path).unwrap();
        assert!(report.title.contains("what does the objective weight at each horizon?"));
        assert!(
            report.title.contains("cutoff:2") && report.title.contains("2 of 8 horizons trained"),
            "{}",
            report.title
        );
        assert_eq!(report.x_label.as_deref(), Some("bars ahead"));
        let y_label = report.y_label.as_deref().unwrap();
        assert!(
            y_label.contains("dimensionless")
                && y_label.contains("mean 1")
                && y_label.contains("0 = horizon not trained"),
            "{y_label}"
        );
        let ReportKind::IndexedLines { steps, series } = report.kind else {
            panic!("the weight panel is indexed by horizon");
        };
        assert_eq!(steps, (1..=8).collect::<Vec<u64>>());
        assert_eq!(
            series.iter().map(|s| s.label.as_str()).collect::<Vec<_>>(),
            ["training horizon loss weight", "uniform reference 1.0"]
        );
        assert_eq!(series[0].values, [4., 4., 0., 0., 0., 0., 0., 0.]);
        assert_eq!(series[1].values, [1.0f32; 8]);
        // A vector that is not mean 1 is a mislabeled chart, so it is not written at all.
        let error = write_horizon_loss_weight(&root, 1, 2000, "broken", &[0.5, 0.5, 0.5])
            .unwrap_err()
            .to_string();
        assert!(error.contains("sums to"), "{error}");
        let error = write_horizon_loss_weight(&root, 1, 2000, "broken", &[])
            .unwrap_err()
            .to_string();
        assert!(error.contains("nonempty"), "{error}");
        fs::remove_dir_all(&root).unwrap();
    }
}

/// The rate each optimizer family actually stepped at, per step.
///
/// One question: what learning rate did the run apply, per family, at every step it took. The
/// values are read out of the optimizer's own state after each step ([`LrTrajectory`]), not
/// recomputed from the schedule, because the two disagree by construction - the NorMuon rate
/// is 0.023/0.008 of the AdamW rate, each 2-D matrix carries the shape multiplier its geometry
/// implies, and the recipe scalar banks carry their own multiplier. A chart of the intended
/// schedule would show one curve where the run had six.
///
/// The title states the budget the shape was drawn against and the step the cooldown starts,
/// because those two numbers are what one arm differs from another by, and a reader comparing
/// two runs cannot infer either from the curve alone until it has already bent.
pub fn write_lr_trajectory(
    output: &Path,
    epoch: usize,
    step: usize,
    schedule: LrSchedule,
    trajectory: &LrTrajectory,
) -> Result<()> {
    if trajectory.steps().is_empty() {
        return Ok(());
    }
    ensure!(
        trajectory
            .steps()
            .windows(2)
            .all(|pair| pair[0] < pair[1]),
        "report steps must increase"
    );
    let series: Vec<ReportSeries> = trajectory
        .labels()
        .iter()
        .enumerate()
        .map(|(family, label)| ReportSeries {
            label: (*label).to_owned(),
            values: trajectory.series(family).map(|rate| rate as f32).collect(),
        })
        .collect();
    ensure!(
        series
            .iter()
            .all(|family| family.values.len() == trajectory.steps().len()
                && family.values.iter().all(|rate| rate.is_finite() && *rate >= 0.)),
        "every family must carry one finite non-negative rate per recorded step"
    );
    let shape = match schedule.cooldown_start() {
        Some(start) => format!(
            "shaped against a {}-step budget, cooldown from step {start} to {} of the peak",
            schedule.budget_steps(),
            schedule.floor()
        ),
        None => "flat: no warmdown".to_owned(),
    };
    write_report(
        output.join("timexer_segment_lr_trajectory.report.bin"),
        &Report {
            title: format!(
                "CausalPatch epoch {epoch} step {step} | what learning rate did each optimizer family actually step at? | {} families, {shape}",
                series.len()
            ),
            x_label: Some("schedule step (0-based, the index the cooldown start names)".into()),
            y_label: Some(LR_UNIT.into()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: trajectory.steps().to_vec(),
                series,
            },
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod lr_trajectory_tests {
    use super::*;
    use crate::torch::timexer_segment::compute::{
        MlpDownLr, OptimizerKind, RecipeKnobs, NANOGPT_COOLDOWN_FLOOR, NANOGPT_COOLDOWN_FRAC,
    };
    use crate::torch::timexer_segment::model::{CausalPatchModel, ModelConfig};
    use shared::report::read_report;
    use std::fs;
    use tch::{nn, Device};

    /// The writer side of the registry contract, and the reading rule the panel promises: one
    /// series per family, the step axis the recorder produced, and a title that states the
    /// budget rather than leaving a reader to infer it from where the curve bends.
    #[test]
    fn the_lr_trajectory_panel_reports_one_realized_rate_per_family_per_step() {
        // Building a model is a DRAW from the process-global generator, so this has to stay out
        // of the seeded tests' sections even though it does not care what it draws.
        let _rng = crate::torch::test_rng::shared();
        let root = std::env::temp_dir().join(format!(
            "timexer-lr-trajectory-{}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let config = ModelConfig {
            seq_len: 96,
            pred_len: 8,
            patch_len: 8,
            layers: 2,
            d_model: 32,
            heads: 4,
            ffn: 128,
            min_history: 8,
            ..Default::default()
        };
        let store = nn::VarStore::new(Device::Cpu);
        let _model = CausalPatchModel::new(&store.root(), &config);
        let schedule = LrSchedule::new(5000, NANOGPT_COOLDOWN_FRAC, NANOGPT_COOLDOWN_FLOOR).unwrap();
        let mut engine = crate::torch::timexer_segment::compute::Engine::new(
            &store,
            0.008,
            RecipeKnobs {
                schedule,
                mlp_down_lr: MlpDownLr::Upstream4x,
                ..RecipeKnobs::reference(config.x0_lambdas)
            },
            false,
            OptimizerKind::PolarExpress,
        )
        .unwrap();
        let mut trajectory = LrTrajectory::new(&engine, 2);
        // Nothing recorded yet is not an empty chart: it is no chart.
        write_lr_trajectory(&root, 1, 0, schedule, &trajectory).unwrap();
        let path = root.join("timexer_segment_lr_trajectory.report.bin");
        assert!(!path.exists());
        trajectory.record(0, &engine).unwrap();
        engine.set_lr(0.004).unwrap();
        trajectory.record(1, &engine).unwrap();
        write_lr_trajectory(&root, 1, 1, schedule, &trajectory).unwrap();

        let report = read_report(&path).unwrap();
        assert!(
            shared::report::TIMEXER_SEGMENT_REPORT_BASES
                .contains(&"timexer_segment_lr_trajectory"),
            "the panel is written but unregistered, so the TUI never scans for it"
        );
        assert_eq!(report.y_label.as_deref(), Some(LR_UNIT));
        assert!(
            report.title.contains("5000-step budget") && report.title.contains("step 2000"),
            "the title must state the budget and its cooldown start: {}",
            report.title
        );
        let ReportKind::IndexedLines { steps, series } = report.kind else {
            panic!("the trajectory panel is step-indexed");
        };
        assert_eq!(steps, [0, 1]);
        assert_eq!(
            series.iter().map(|s| s.label.as_str()).collect::<Vec<_>>(),
            [
                "NorMuon packed QKV",
                "NorMuon attention output",
                "NorMuon MLP up",
                "NorMuon MLP down",
                "AdamW dense",
                "AdamW recipe scalars",
            ]
        );
        // The realized rates, not the schedule multiplier: NorMuon MLP-down at four times the
        // square attention-O rate, MLP-up at twice, the scalar banks at 5x the dense AdamW
        // rate, and every one of them halved by the second row's global change.
        let rate = |family: usize, row: usize| series[family].values[row] as f64;
        assert!((rate(1, 0) - 0.023).abs() < 1e-9);
        assert!((rate(2, 0) - 0.046).abs() < 1e-9);
        assert!((rate(3, 0) - 0.092).abs() < 1e-9);
        assert!((rate(4, 0) - 0.008).abs() < 1e-9);
        assert!((rate(5, 0) - 0.040).abs() < 1e-9);
        for family in 0..series.len() {
            assert!((rate(family, 1) - rate(family, 0) / 2.0).abs() < 1e-9);
        }
        fs::remove_dir_all(&root).unwrap();
    }

    /// An uncalibrated run's gain panel is the honest 1.0, never NaN and never absent.
    ///
    /// NaN would say "no calibration was fitted", which is a different statement from "the
    /// calibration applied was the identity", and the whole point of writing this panel on
    /// every run is that the second statement is what a reader needs before laying a
    /// calibrated arm over an uncalibrated one.
    #[test]
    fn an_uncalibrated_run_writes_a_gain_of_exactly_one_at_every_horizon() {
        use crate::torch::timexer_segment::calibration::{Blocks, Measured, Pairing};
        let root = std::env::temp_dir().join(format!(
            "timexer-identity-gain-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&root).unwrap();
        let horizon = 192;
        write_calibration_gain(&root, 2, 3000, horizon, AppliedGain::Identity, 64).unwrap();
        let report = read_report(root.join("timexer_segment_calibration_gain.report.bin")).unwrap();
        let ReportKind::IndexedLines { steps, series } = &report.kind else {
            panic!("the gain panel must be a horizon-indexed line chart");
        };
        assert_eq!(steps.len(), horizon);
        assert_eq!(series.len(), 1, "the identity has no fit block to compare to");
        assert_eq!(series[0].values, vec![1.0f32; horizon]);
        assert!(report.title.contains("none fitted"));
        // And the fitted case carries the frozen curve plus both populations' own optima, so
        // the two cases are readable side by side on one axis.
        let pairing = Pairing {
            checkpoint_format: "f".to_owned(),
            objective: "o".to_owned(),
            weights_sha256: "a".repeat(64),
            manifest_sha256: "b".repeat(64),
            step: 3000,
            pred_len: horizon,
            corpus_schema: "s".to_owned(),
            corpus_sha256: "c".repeat(64),
        };
        let blocks = Blocks {
            calibration_first_origin_ms: 1,
            calibration_last_origin_ms: 2,
            calibration_last_target_ms: 3,
            calibration_origins: 4,
            evaluation_first_origin_ms: 5,
            evaluation_last_origin_ms: 6,
            evaluation_origins: 7,
            purge_gap_ms: 2,
        };
        let optimal_gain: Vec<f64> = (1..=horizon).map(|h| 1. / (h as f64).sqrt()).collect();
        let measured = Measured {
            pearson: vec![0.1; horizon],
            demeaned_gain: vec![0.01; horizon],
            scaling_gain: vec![-0.02; horizon],
            forecast_variance: vec![0.05; horizon],
            covariance: optimal_gain.iter().map(|gain| gain * 0.05).collect(),
            persistence: vec![1.; horizon],
            bars: vec![4096.; horizon],
            optimal_gain,
        };
        let calibration = MeanCalibration::fit(pairing, blocks, &measured).unwrap();
        let evaluation_gain = vec![0.5; horizon];
        write_calibration_gain(
            &root,
            2,
            3000,
            horizon,
            AppliedGain::Fitted {
                calibration: &calibration,
                evaluation_gain: &evaluation_gain,
            },
            64,
        )
        .unwrap();
        let report = read_report(root.join("timexer_segment_calibration_gain.report.bin")).unwrap();
        let ReportKind::IndexedLines { series, .. } = &report.kind else {
            panic!("the gain panel must be a horizon-indexed line chart");
        };
        // Applied gain, the block it was fitted on, the block it had to predict, the bound the
        // amplifying direction is charged against, and the identity reference.
        assert_eq!(series.len(), 5);
        assert_eq!(
            series[0].values,
            calibration
                .gain
                .iter()
                .map(|gain| *gain as f32)
                .collect::<Vec<f32>>()
        );
        assert!(series[0].values.iter().all(|gain| gain.is_finite()));
        // The ceiling is on the same axis and is never below the identity, so the applied
        // curve can be read against it without converting units.
        let ceiling = series
            .iter()
            .find(|line| line.label.contains("amplification ceiling"))
            .expect("the gain panel must carry the bound the applied curve respects");
        assert!(ceiling.values.iter().all(|value| *value >= 1.));
        assert!(series[0]
            .values
            .iter()
            .zip(&ceiling.values)
            .all(|(gain, bound)| gain <= bound));
        // A horizon count that disagrees with the calibration is a pairing fault, not a chart
        // to draw with one axis silently truncated.
        assert!(write_calibration_gain(
            &root,
            2,
            3000,
            horizon - 1,
            AppliedGain::Frozen(&calibration),
            64
        )
        .is_err());
        fs::remove_dir_all(&root).unwrap();
    }
}

/// Per-coefficient unit. Both series are dimensionless ratios of one question, which is why
/// they share an axis: see the base's own comment in `shared::report`.
const BASIS_UNIT: &str = "dimensionless (rho-squared: share of a coefficient's variance the \
                          forecast explains, higher better; beta: fitted amplitude multiple, \
                          1.0 correct, below 1 over-amplified)";

/// Where along the orthonormal coefficient axis the forecast carries signal, and where its
/// amplitude is wrong. x = coefficient index, 0 the lowest-frequency coefficient.
///
/// This is the one panel where the measured amplitude defect is DIAGONAL. In horizon space the
/// defect is a 192x192 object smeared across near-duplicate cumulative rows, and the per-horizon
/// `β̂` curve reads it only along that object's diagonal in the WRONG basis; here the basis is
/// the one the objective is measured in, so a `β̂` that collapses at low coefficient index and
/// sits at 1 elsewhere localizes the whole over-amplitude to a subspace. If it does NOT - if
/// `β̂` is uniformly depressed across the coefficient axis - then the defect is isotropic and no
/// reweighting of this axis can address it, which is itself the finding.
///
/// Written on every run that uses a non-identity basis, and on none that does not: an identity
/// basis's coefficients ARE its horizons and the existing per-horizon family already carries
/// them, so writing this too would be the same numbers on a second axis label.
///
/// `correlation` and `amplitude_gain` are per-coefficient `ρ̂` and `β̂` on the scored
/// population, `NaN` where unmeasured; `weights` is the mean-1 objective weight actually
/// applied; `complete_share` is the share of scored rows whose horizon window was complete and
/// therefore rotatable, which is the price the rotation charges; `defect` is
/// `max |WᵀW - I|` in fp32 at the shipped shape.
#[allow(clippy::too_many_arguments)]
pub fn write_target_basis(
    output: &Path,
    epoch: usize,
    step: usize,
    basis: &str,
    weight_spec: &str,
    correlation: &[f64],
    amplitude_gain: &[f64],
    weights: &[f64],
    complete_share: f64,
    defect: f64,
    population: usize,
) -> Result<()> {
    let coefficients = correlation.len();
    ensure!(
        coefficients > 0
            && amplitude_gain.len() == coefficients
            && weights.len() == coefficients,
        "the coefficient panel needs one rho, one beta and one weight per coefficient, got \
         {coefficients}, {} and {}",
        amplitude_gain.len(),
        weights.len()
    );
    ensure!(
        (0.0..=1.0).contains(&complete_share),
        "the complete-window share is a share, got {complete_share}"
    );
    let series = vec![
        ReportSeries {
            label: format!("{FULL} rho-squared per coefficient"),
            values: correlation
                .iter()
                .map(|rho| (rho * rho) as f32)
                .collect(),
        },
        ReportSeries {
            label: format!("{FULL} beta fitted amplitude gain per coefficient"),
            values: amplitude_gain.iter().map(|beta| *beta as f32).collect(),
        },
        ReportSeries {
            label: "correct amplitude 1.0".to_owned(),
            values: vec![1.0; coefficients],
        },
        ReportSeries {
            label: "training objective weight per coefficient (mean 1)".to_owned(),
            values: weights.iter().map(|weight| *weight as f32).collect(),
        },
    ];
    write_report(
        output.join("timexer_segment_target_basis.report.bin"),
        &Report {
            title: format!(
                "CausalPatch epoch {epoch} step {step} | where along the {basis} coefficient axis is there signal, and where is the amplitude wrong? | {coefficients} coefficients, weight {weight_spec}, orthonormality defect {defect:.2e} in fp32, {:.4} of scored rows had a complete rotatable window, {population} origins",
                complete_share
            ),
            x_label: Some("coefficient index (0 = lowest frequency)".to_owned()),
            y_label: Some(BASIS_UNIT.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: (0..coefficients as u64).collect(),
                series,
            },
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod target_basis_report_tests {
    use super::*;
    use shared::report::read_report;
    use std::fs;

    /// The panel states its own basis, its own orthonormality and its own dropped share, and a
    /// mismatched vector length is a refusal rather than a chart with a silently short axis.
    #[test]
    fn the_coefficient_panel_states_the_basis_and_refuses_a_ragged_input() {
        let root = std::env::temp_dir().join(format!(
            "timexer_target_basis_report_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let path = root.join("timexer_segment_target_basis.report.bin");
        write_target_basis(
            &root,
            2,
            2000,
            "dct",
            "uniform",
            &[0.2, 0.1, f64::NAN, 0.0],
            &[0.27, 0.9, f64::NAN, 1.4],
            &[1.0, 1.0, 1.0, 1.0],
            0.9915,
            1.7e-6,
            433_721,
        )
        .unwrap();
        let report = read_report(&path).unwrap();
        assert!(report.title.contains("dct coefficient axis"), "{}", report.title);
        assert!(report.title.contains("orthonormality defect"), "{}", report.title);
        assert!(report.title.contains("complete rotatable window"), "{}", report.title);
        assert!(report.y_label.as_ref().unwrap().contains("over-amplified"));
        let ReportKind::IndexedLines { steps, series } = &report.kind else {
            panic!("the coefficient panel must be an index-indexed line chart");
        };
        assert_eq!(steps, &[0, 1, 2, 3]);
        assert_eq!(series.len(), 4);
        // ρ² is squared here, not in the caller, so a sign convention cannot leak into a
        // variance share; and an unmeasured coefficient renders NaN, never 0.
        assert!((series[0].values[0] - 0.04).abs() < 1e-6, "{:?}", series[0].values);
        assert!(series[0].values[2].is_nan(), "{:?}", series[0].values);
        assert!(series[1].values[2].is_nan(), "{:?}", series[1].values);
        assert_eq!(series[2].values, vec![1.0; 4]);
        let error = write_target_basis(
            &root, 2, 2000, "haar", "snr", &[0.2, 0.1], &[1.0], &[1.0, 1.0], 1.0, 0.0, 1,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("one weight per coefficient"), "{error}");
        let error = write_target_basis(
            &root, 2, 2000, "haar", "snr", &[0.2], &[1.0], &[1.0], 1.5, 0.0, 1,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("is a share"), "{error}");
        fs::remove_dir_all(&root).unwrap();
    }
}

/// Axis of the supervision-occupancy panel. Everything on it is a share of ONE denominator -
/// this arm's own distinct supervised outcomes - so the unit and the reading rule fit in one
/// line and no series needs a second axis.
const OCCUPANCY_UNIT: &str =
    "share of this arm's distinct supervised (origin, horizon) outcomes; 1.0 is every outcome touched, and the step where the 1-pass curve flattens is the saturation step";

/// Gradient-pass thresholds the panel draws. One curve per threshold turns the mean exposure
/// count into a distribution on the same axis: the 1-pass curve is saturation, and the higher
/// ones are how much of the budget after it went into re-presentation.
const OCCUPANCY_PASSES: [u32; 5] = [1, 2, 4, 8, 16];

/// How much of its own training signal this arm has consumed, per optimizer step.
///
/// Every value is computed in closed form from the census's MEASURED per-outcome multiplicity
/// histogram, so the chart is not a model of the run - it is the run's own row pool evaluated
/// at each step, and it is exact under the sampler the loop actually uses (one shuffle of the
/// retained pool per epoch, drawn without replacement). Two consequences worth stating: the
/// curve exists from step 0 with no measurement, so the saturation step is legible before the
/// held-out curve has bent; and its denominator is the ARM's own support, which a knob that
/// adds origins (a per-row patch phase) changes, so two arms' saturation STEPS are comparable
/// while their supports are not.
///
/// `axis` is the step grid, and it is drawn densely rather than at report intervals because
/// the interesting structure - a knee - can sit between two evaluations.
pub fn write_supervision_occupancy(
    output: &Path,
    epoch: usize,
    step: usize,
    batch_size: usize,
    census: &super::supervision::SupervisionCensus,
) -> Result<()> {
    ensure!(batch_size > 0, "occupancy needs a positive batch size");
    ensure!(
        census.rows > 0 && census.support() > 0,
        "an arm with no retained rows and no supervised outcome has no occupancy to chart"
    );
    // A dense axis reaching at least one full sweep past the current step, so the knee is on
    // the chart before the run gets there and stays on it afterwards; capped at 512 points so
    // the panel costs the same whatever the run length.
    let horizon = step
        .max(census.rows.div_ceil(batch_size))
        .saturating_add(1)
        .max(2);
    let stride = horizon.div_ceil(512).max(1);
    let axis: Vec<u64> = (0..=horizon)
        .step_by(stride)
        .map(|value| value as u64)
        .collect();
    let mut series: Vec<ReportSeries> = OCCUPANCY_PASSES
        .iter()
        .map(|passes| ReportSeries {
            label: match passes {
                1 => "outcomes with at least one gradient pass".to_owned(),
                _ => format!("outcomes with at least {passes} gradient passes"),
            },
            values: axis
                .iter()
                .map(|point| census.coverage(*point as usize, batch_size, *passes) as f32)
                .collect(),
        })
        .collect();
    // The epoch progress, which is exactly the mean gradient passes per outcome divided by the
    // mean multiplicity the title states. Dimensionless, so it belongs on this axis; it is the
    // only series allowed above 1, and it is what makes a multi-sweep arm's re-presentation
    // legible.
    series.push(ReportSeries {
        label: "mean gradient passes per outcome, as a share of the arm's mean multiplicity"
            .to_owned(),
        values: axis
            .iter()
            .map(|point| census.exposure_share(*point as usize, batch_size) as f32)
            .collect(),
    });
    ensure!(
        series
            .iter()
            .all(|line| line.values.len() == axis.len()
                && line.values.iter().all(|value| value.is_finite() && *value >= 0.)),
        "every occupancy series must carry one finite non-negative share per step"
    );
    let saturation = census.saturation_step(0.999, batch_size);
    write_report(
        output.join(super::supervision::OCCUPANCY_BASE.to_owned() + ".report.bin"),
        &Report {
            title: format!(
                "CausalPatch epoch {epoch} step {step} | how much of its own training signal has this arm consumed? | {}; {} rows, {} distinct supervised outcomes, {:.4} mean exposures per outcome per sweep, {:.4} of them interior at {} exposures; 0.999 of outcomes covered by step {saturation}; retained-versus-full timestamp decile drift {:.4}",
                census.selection.summary(),
                census.rows,
                census.support(),
                census.mean_multiplicity(),
                census.interior_share(),
                census.geometry.interior_multiplicity(),
                census.decile_drift(),
            ),
            x_label: Some("optimizer step".to_owned()),
            y_label: Some(OCCUPANCY_UNIT.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines { steps: axis, series },
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod supervision_occupancy_tests {
    use super::*;
    use crate::torch::timexer_segment::supervision::{
        RowSelection, SupervisionCensus, SupervisionGeometry, OCCUPANCY_BASE,
    };
    use shared::report::read_report;
    use std::fs;

    /// The panel is registered, states its own pool, and its 1-pass curve saturates where the
    /// census says it does - which is the whole reason the base exists.
    #[test]
    fn the_occupancy_panel_draws_the_saturation_step_the_census_predicts() {
        let root = std::env::temp_dir()
            .join(format!("timexer_occupancy_report_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        assert!(
            shared::report::TIMEXER_SEGMENT_REPORT_BASES.contains(&OCCUPANCY_BASE),
            "the occupancy panel is written but unregistered, so the TUI never scans for it"
        );
        let geometry = SupervisionGeometry::new(6000, 16, 256, 192).unwrap();
        let mut histogram = vec![0u64; geometry.origins_per_row + 1];
        histogram[30] = 27_783_348;
        histogram[15] = 3_375_768;
        let census = SupervisionCensus {
            selection: RowSelection::identity(20260905),
            geometry,
            rows: 2_455_276,
            histogram,
            partial_target_rows: 4_873,
            phase_dropped_rows: 0,
            retained_deciles: [0.1; 10],
            full_deciles: [0.1; 10],
        };
        write_supervision_occupancy(&root, 1, 2000, 256, &census).unwrap();
        let report = read_report(&root.join(format!("{OCCUPANCY_BASE}.report.bin"))).unwrap();
        assert!(
            report.y_label.as_ref().unwrap().contains("saturation step"),
            "{:?}",
            report.y_label
        );
        assert!(!report.y_label.as_ref().unwrap().contains('='));
        let ReportKind::IndexedLines { steps, series } = &report.kind else {
            panic!("occupancy is a step-indexed line chart");
        };
        assert_eq!(series.len(), OCCUPANCY_PASSES.len() + 1);
        assert!(series.iter().all(|line| !line.label.contains('=')));
        // Monotone in the step and ordered in the pass threshold: a coverage curve that fell,
        // or a 2-pass curve above the 1-pass curve, would be an arithmetic error.
        for line in &series[..OCCUPANCY_PASSES.len()] {
            assert!(
                line.values.windows(2).all(|pair| pair[1] >= pair[0] - 1e-6),
                "{} is not monotone",
                line.label
            );
        }
        for pair in series[..OCCUPANCY_PASSES.len()].windows(2) {
            assert!(
                pair[0]
                    .values
                    .iter()
                    .zip(pair[1].values.iter())
                    .all(|(low, high)| *low >= *high - 1e-6),
                "{} must dominate {}",
                pair[0].label,
                pair[1].label
            );
        }
        // The census's own saturation step, read off the drawn curve rather than from the
        // title: the first sampled step whose 1-pass coverage clears 0.999.
        let drawn = steps
            .iter()
            .zip(series[0].values.iter())
            .find(|(_, value)| **value >= 0.999)
            .map(|(step, _)| *step as usize)
            .expect("the axis must reach saturation");
        let predicted = census.saturation_step(0.999, 256);
        let stride = (steps[1] - steps[0]) as usize;
        assert!(
            drawn >= predicted && drawn < predicted + stride,
            "the chart saturates at {drawn}, the census predicts {predicted}"
        );
        assert!(report.title.contains(&format!("step {predicted}")), "{}", report.title);
        // An empty pool is a refusal, not a blank panel.
        let empty = SupervisionCensus { rows: 0, ..census.clone() };
        assert!(write_supervision_occupancy(&root, 1, 1, 256, &empty).is_err());
        assert!(write_supervision_occupancy(&root, 1, 1, 0, &census).is_err());
        fs::remove_dir_all(&root).unwrap();
    }
}
