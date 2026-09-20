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
    /// Held-out NLL with the training objective's horizon weighting. Reported, not selected on.
    pub validation_objective_nll: f64,
    /// THE selection scalar: the horizon-weighted close MSE ratio at each horizon's own best
    /// scale, which is what `weights/best` and both patience rules minimize. A ratio, so it
    /// charts on the headline `timexer_segment_skill` axis beside the ratios it is chosen
    /// against rather than on a base of its own.
    pub validation_scale_free_objective: f64,
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
    /// [`TradingCurve::close_mse_ratio`] recomputed at the per-horizon mean amplitude
    /// calibration the run fitted at this same evaluation, in closed form from the same
    /// moments. Carried BESIDE the achieved ratio and never instead of it: the delta between
    /// the two IS the amplitude diagnostic, and a long-horizon ratio that crosses 1 while its
    /// calibrated twin stays below is an over-amplified mean, not a mean that lost its signal.
    /// NaN on a pass scored with no calibration to apply.
    pub calibrated_close_ratio: f64,
    /// `β̂` of that calibrated emission, `optimal_gain / g`: 1 exactly where the applied gain
    /// is the amplitude this draw measures, so a fitted curve that does not move this toward 1
    /// is legible as a calibration that did not transfer. NaN with no calibration.
    pub calibrated_optimal_gain: f64,
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
///
/// `anchor_gain` is the per-horizon mean amplitude calibration to report this UN-GAINED
/// emission against - [`super::calibration::FrozenGain::anchor`], fitted on the reserved
/// partition at this same evaluation. `None` where there is nothing to add: a pass scored
/// through an already-calibrated model emits the gained mean, so its achieved ratio IS the
/// calibrated one and the un-gained comparand is the inverted panel
/// [`write_amplitude`] draws rather than a second series here.
pub fn horizon_track(
    horizon: &HorizonCurve,
    trading: &TradingCurve,
    anchor_gain: Option<&[f64]>,
) -> Vec<HorizonPoint> {
    DECISION_HORIZONS
        .iter()
        .filter_map(|&h| {
            let i = h.checked_sub(1)?;
            let (mean_forecast, mean_target) =
                (*trading.mean_forecast.get(i)?, *trading.mean_target.get(i)?);
            let persistence = *trading.persistence.get(i)?;
            // `E[f·y]` and `E[f²]` rebuilt from the curve's own centered moments, which is
            // every term `1 - (2g·E[fy] - g²·E[f²])/E[y²]` needs: a per-horizon gain is a
            // closed-form rescale of two numbers this pass already reduced, so reporting what
            // the calibration did to the ratio costs no second scoring pass and cannot
            // disagree with the achieved ratio beside it.
            let joint = *trading.covariance.get(i)? + mean_forecast * mean_target;
            let square = *trading.forecast_variance.get(i)? + mean_forecast * mean_forecast;
            let optimal_gain = *trading.optimal_gain.get(i)?;
            let (calibrated_close_ratio, calibrated_optimal_gain) =
                match anchor_gain.and_then(|curve| curve.get(i)) {
                    Some(&gain) => (
                        1. - (2. * gain * joint - gain * gain * square) / persistence,
                        optimal_gain / gain,
                    ),
                    None => (f64::NAN, f64::NAN),
                };
            // Every field is read through `get`, so a curve shorter than the horizon drops the
            // whole point instead of contributing a row of gaps.
            Some(HorizonPoint {
                horizon: h,
                neutral_ratio: horizon.mse.get(i)? / horizon.persistence_mse.get(i)?,
                raw_ratio: horizon.absolute_mse.get(i)? / horizon.absolute_persistence_mse.get(i)?,
                close_ratio: *trading.close_mse_ratio.get(i)?,
                best_scale_ratio: *trading.best_scale_mse_ratio.get(i)?,
                optimal_gain,
                calibrated_close_ratio,
                calibrated_optimal_gain,
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
            // The scalar `weights/best` and both patience rules are actually decided on, on
            // the headline base beside the ratios a reader would otherwise assume decided
            // them. It belongs here and not on a base of its own: it is a dimensionless ratio
            // against the same persistence prior and it reads against the same parity line.
            scored(
                &format!("{SAMPLE} selection objective (horizon-weighted close best-scale MSE ratio)"),
                false,
                |p| p.validation_scale_free_objective,
            ),
            scored(
                &format!("{FULL} selection objective (horizon-weighted close best-scale MSE ratio)"),
                true,
                |p| p.validation_scale_free_objective,
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
        // Both amplitudes, on one axis. The un-gained `β̂` is what the calibration is fitted
        // AGAINST and the gained one is what the run would actually deploy, and only the pair
        // separates "the fit transferred" from "the fit moved the number on the fit block
        // alone": a calibrated gain still far from 1 on a held-out draw is a curve fitted out
        // of period, which no single series can say.
        let mut gain = per_horizon("uncalibrated close MSE-optimal forecast gain", |p| {
            p.optimal_gain
        });
        gain.extend(per_horizon("calibrated close MSE-optimal forecast gain", |p| {
            p.calibrated_optimal_gain
        }));
        gain.push(series("perfect amplitude calibration 1.0", |_| 1.));
        chart(
            "timexer_segment_horizon_steps_gain",
            "is the conditional mean's amplitude right, per horizon, over training, before and \
             after the run's own calibration?",
            OPTIMAL_GAIN_UNIT,
            ScaleKind::Linear,
            &scope,
            gain,
        )?;
        // Three amplitudes of ONE forecast: as emitted, at the gain this run fitted out of
        // sample, and at the unattainable per-horizon optimum. The first two are the arm's
        // real choices and the third is the ceiling they are read against, so a long-horizon
        // ratio that crosses parity while the other two stay under it is an amplitude failure
        // stated in one panel instead of inferred across three.
        let mut best_scale = per_horizon(&format!("{NEUTRAL} uncalibrated close MSE ratio"), |p| {
            p.close_ratio
        });
        best_scale.extend(per_horizon(
            &format!("{NEUTRAL} calibrated close MSE ratio"),
            |p| p.calibrated_close_ratio,
        ));
        best_scale.extend(per_horizon(
            &format!("{NEUTRAL} close MSE ratio at the best scale"),
            |p| p.best_scale_ratio,
        ));
        best_scale.push(series("parity 1.0", |_| 1.));
        chart(
            "timexer_segment_horizon_steps_best_scale",
            "is a close MSE ratio above 1 absent signal, or signal at the wrong amplitude, and \
             does the run's own calibration recover it?",
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
                    "commission_usd" | "regulatory_usd" | "spread_usd" | "impact_usd"
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
        // legible on this panel rather than only in a log line. An unmeasured horizon draws as
        // a gap, not as a zero: a zero would read as a measured collapse.
        series.push(curve(
            format!("{CALIBRATION} close-anchor MSE-optimal gain carried by the checkpoint, signed"),
            &panels
                .applied
                .measured_anchor
                .iter()
                .map(|gain| gain.unwrap_or(f64::NAN))
                .collect::<Vec<f64>>(),
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

/// Named, never clipped: a horizon whose own measured amplitude is non-positive or absent, or
/// whose pure-gain parameterization the fit refused, is gated out of position sizing by
/// [`FrozenGain::tradable`], and this is where a reader finds out which ones and under which of
/// the three reasons. A gate that does not appear in the panel is how an all-zero trading
/// result stays unexplained, and three reasons collapsed into one is how it stays unexplained
/// after someone looks.
fn gated_note(applied: &FrozenGain) -> String {
    let gated = applied.gated();
    if gated.is_empty() {
        return String::new();
    }
    let listed = gated
        .iter()
        .take(8)
        .map(|(horizon, gate)| format!("h={horizon} {gate}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        " | {} of {} horizons GATED OUT OF SIZING on a non-positive or absent measured gain or a refused parameterization ({listed}{}), scored but never traded",
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
    format!(
        "CausalPatch epoch {epoch} step {step} | is the forecast's over-amplitude an out-of-sample shrinkage problem or an in-sample objective one? | fitted on {} reserved calibration-partition origins to {} ({} ms of tightest per-ticker separation from that ticker's own first scored origin, over a {horizon}-bar reach), {} in-sample training origins measured beside them, {tickers} tickers, {:.1} anchor and {:.1} offset effective degrees of freedom{gated}",
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
            label: format!(
                "{CALIBRATION} close-channel constant-forecast ceiling, the whole intercept opportunity"
            ),
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

    /// The per-horizon anchor gain the fixtures report against. Per-index and crossing 1, so a
    /// mis-sliced calibration is a wrong number rather than a wrong shape, and both the
    /// shrinking and the amplifying direction are exercised.
    fn anchor_gain(width: usize) -> Vec<f64> {
        (1..=width).map(|h| 0.5 + 0.01 * h as f64).collect()
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
            validation_scale_free_objective: 0.985,
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
            horizons: horizon_track(
                &horizon_curve(width),
                &trading_curve(width),
                Some(&anchor_gain(width)),
            ),
            cross_horizons: horizon_track(
                &horizon_curve(width),
                &cross_curve(width),
                Some(&anchor_gain(width)),
            ),
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
        let gain = anchor_gain(64);
        let track = horizon_track(&horizon_curve(64), &trading_curve(64), Some(&gain));
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
        // The calibrated pair, against the closed form the fixture makes exact: with a
        // mean-zero forecast the gained close ratio is `1 - 2g·Cov/P + g²·Var/P` and the gained
        // amplitude is `β̂/g`. At h = 64 the fixture's `Cov/P = 0.004`, `Var/P = 0.01` and
        // `g = 1.14`, so the gain AMPLIFIES an already over-amplified forecast and the ratio
        // moves the wrong way - which is the state a report has to be able to show.
        let g = 0.5 + 0.01 * 64.;
        assert!(
            (h64.calibrated_close_ratio - (1. - 2. * g * 0.004 + g * g * 0.01)).abs() < 1e-12,
            "h = 64 calibrated close ratio {}",
            h64.calibrated_close_ratio
        );
        assert!((h64.calibrated_optimal_gain - (1. - 0.004 * 64.) / g).abs() < 1e-12);
        // The sample draw never fires a cross-section: NaN and a population count of 0, which
        // is a different statement from "the IC was measured at 0".
        assert!(h64.cross_sectional_ic.is_nan() && h64.cross_sectional_ic_se.is_nan());
        assert_eq!(h64.cross_sections, 0.);
        let measured = horizon_track(&horizon_curve(64), &cross_curve(64), Some(&gain));
        let h64 = measured.last().unwrap();
        assert!((h64.cross_sectional_ic - (0.08 - 0.0001 * 64.)).abs() < 1e-12);
        assert_eq!(h64.cross_sections, 40.);
        // No calibration to report against - an already-gained emission - reads as ABSENT, not
        // as the identity gain: a 1.0 there would be bit-identical to a measured unit amplitude
        // and the chart would claim a calibration the pass never had.
        let ungained = horizon_track(&horizon_curve(64), &trading_curve(64), None);
        let h64 = ungained.last().unwrap();
        assert!(h64.calibrated_close_ratio.is_nan() && h64.calibrated_optimal_gain.is_nan());
        assert!((h64.close_ratio - (0.995 + 0.0002 * 64.)).abs() < 1e-12);
        // A gain curve shorter than the horizon leaves the calibrated pair absent at the
        // horizons it does not reach, and leaves the achieved ratio intact.
        let short = horizon_track(&horizon_curve(64), &trading_curve(64), Some(&gain[..32]));
        assert!(short.last().unwrap().calibrated_close_ratio.is_nan());
        assert!(short[3].calibrated_close_ratio.is_finite());
        assert_eq!(
            horizon_track(&horizon_curve(4), &trading_curve(4), Some(&anchor_gain(4))).len(),
            1
        );
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
        let mut points = vec![
            point(1000, false, 2.25, 192),
            point(2000, false, 2.08, 192),
            point(3000, true, 2.05, 192),
        ];
        // Distinct per point, so the selection scalar's split routing is proved rather than
        // satisfied by every row carrying the same number.
        for (index, value) in [0.991_f64, 0.984, 0.979].into_iter().enumerate() {
            points[index].validation_scale_free_objective = value;
        }
        write_metrics(&root, &points).unwrap();

        // The scalar `weights/best` is decided on has to be READABLE, on the headline base and
        // routed to its own split: a selection rule whose scalar is not charted is a rule no
        // run-versus-run comparison can check.
        let skill = read_report(root.join("timexer_segment_skill.report.bin")).unwrap();
        let ReportKind::IndexedLines { series, .. } = &skill.kind else {
            panic!("the headline panel must be step-indexed");
        };
        let selection = |split: &str| -> Vec<f32> {
            let label =
                format!("{split} selection objective (horizon-weighted close best-scale MSE ratio)");
            series
                .iter()
                .find(|s| s.label == label)
                .unwrap_or_else(|| panic!("{label} missing from the headline panel"))
                .values
                .clone()
        };
        let sample_selection = selection(SAMPLE);
        assert!((sample_selection[0] - 0.991).abs() < 1e-6);
        assert!((sample_selection[1] - 0.984).abs() < 1e-6);
        assert!(sample_selection[2].is_nan(), "the sample series leaked the epoch-end point");
        let full_selection = selection(FULL);
        assert!(full_selection[0].is_nan() && full_selection[1].is_nan());
        assert!((full_selection[2] - 0.979).abs() < 1e-6);
        // No `=` in any label: `report_cli --var` splits a rendered token on its first `=`, so
        // a series carrying one cannot be named on the command line.
        assert!(series.iter().all(|s| !s.label.contains('=')));

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
                metrics.cross_horizons = horizon_track(
                    &horizon_curve(192),
                    &{
                        let mut curve = cross_curve(192);
                        curve.cross_sectional_ic =
                            curve.cross_sectional_ic.iter().map(|ic| ic * scale).collect();
                        curve
                    },
                    Some(&anchor_gain(192)),
                );
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
                format!("{CROSS} uncalibrated close MSE-optimal forecast gain at horizon 192"),
                1. - 0.004 * 192.,
            ),
            (
                "timexer_segment_horizon_steps_gain",
                format!("{CROSS} calibrated close MSE-optimal forecast gain at horizon 192"),
                (1. - 0.004 * 192.) / (0.5 + 0.01 * 192.),
            ),
            (
                "timexer_segment_horizon_steps_best_scale",
                format!("{CROSS} {NEUTRAL} close MSE ratio at the best scale at horizon 192"),
                0.999 - 0.0001 * 192.,
            ),
            (
                "timexer_segment_horizon_steps_best_scale",
                format!("{CROSS} {NEUTRAL} calibrated close MSE ratio at horizon 192"),
                {
                    let g = 0.5 + 0.01 * 192.;
                    1. - 2. * g * 0.004 + g * g * 0.01
                },
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
}

/// The amplitude-calibration panels. A separate module because every test here is CPU-only -
/// no model, no device, no corpus - so the whole family runs under one narrow filter in
/// milliseconds, which is what makes it usable as the reduction's own self-check.
#[cfg(test)]
mod amplitude_report_tests {
    use super::*;
    use shared::report::{read_report, TIMEXER_SEGMENT_REPORT_BASES};
    use std::fs;

    /// A checkpoint whose MEASURED gain is 1 charts it as an explicit 1.0 at every horizon,
    /// never as NaN and never as an absent series: a unit gain is a measurement about what the
    /// model multiplied by, and the `evaluate` path has to render it as one. It is the only
    /// way a unit curve can reach a chart now - the estimator aborts rather than freezing one -
    /// so the series must not be mistaken for a missing calibration.
    #[test]
    fn a_measured_unit_gain_is_charted_as_1_and_not_as_a_missing_series() {
        use crate::torch::timexer_segment::calibration::{Blocks, FrozenGain, Moments};
        let root = std::env::temp_dir().join(format!(
            "timexer-identity-gain-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&root).unwrap();
        let horizon = 8;
        let cells = horizon * 4;
        let moments = Moments {
            pred_len: horizon,
            channels: 4,
            bars: vec![1.; horizon],
            target: vec![1.; cells],
            target_square: vec![1.; cells],
            anchor_square: vec![1.; cells],
            anchor_offset: vec![0.; cells],
            offset_square: vec![0.; cells],
            anchor_target: vec![1.; cells],
            offset_target: vec![0.; cells],
        };
        let applied = FrozenGain {
            estimator: "identity fixture".into(),
            blocks: Blocks {
                calibration_first_origin_ms: 1,
                calibration_last_origin_ms: 2,
                calibration_last_target_ms: 3,
                calibration_origins: 4,
                evaluation_first_origin_ms: 5,
                evaluation_last_origin_ms: 6,
                evaluation_origins: 7,
                purge_gap_ms: 8,
            },
            anchor: vec![1.; horizon],
            offset: vec![1.; horizon],
            intercept_refused: Vec::new(),
            measured_anchor: vec![Some(1.); horizon],
        };
        let panels = AmplitudePanels {
            epoch: 2,
            step: 3000,
            tickers: 4,
            applied: &applied,
            fit: None,
            scored: vec![AmplitudeSplit {
                split: FULL,
                origins: 7,
                emission: Emission::Uncalibrated,
                moments: &moments,
            }],
        };
        write_amplitude(&root, &panels).unwrap();
        let report =
            read_report(root.join("timexer_segment_calibration_gain.report.bin")).unwrap();
        let ReportKind::IndexedLines { steps, series } = &report.kind else {
            panic!("the gain panel must be a horizon-indexed line chart");
        };
        assert_eq!(steps.len(), horizon);
        assert_eq!(series[0].values, vec![1.0f32; horizon]);
        assert_eq!(series[1].values, vec![1.0f32; horizon]);
        assert!(report.title.contains("checkpoint's own curves"));
        assert!(TIMEXER_SEGMENT_REPORT_BASES.contains(&"timexer_segment_calibration_gain"));
        fs::remove_dir_all(&root).unwrap();
    }

    /// The fitted panels, on the three questions they exist to separate: what gain was applied,
    /// whether the amplitude error is out-of-sample shrinkage or an in-sample objective
    /// artifact, and whether applying the gain moves the MSE ratio below persistence.
    ///
    /// Also the presentation contract that makes them readable: the dimensionless gains and the
    /// MSE ratios are different units and live on different bases, so neither axis can be
    /// rescaled by the other; and a horizon the sizing gate would zero is NAMED in the title
    /// with its own measured value, because a gate that appears only in a log line is how an
    /// all-zero book stays unexplained.
    #[test]
    fn the_fitted_amplitude_panels_separate_training_from_held_out_gain_and_name_the_gate() {
        use crate::torch::timexer_segment::calibration::{Blocks, MeanCalibration, Moments};
        let root = std::env::temp_dir().join(format!("timexer-fitted-gain-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let horizon = 8;
        let cells = horizon * 4;
        // An orthogonal two-coordinate population whose exact solve is `anchor_gain(h)` on the
        // anchor and 1 on the offset, with the anchor's energy growing like a cumulative
        // return's, which is what makes the two coordinates' weights differ by orders of
        // magnitude the way the real block's do.
        let build = |anchor_gain: &dyn Fn(usize) -> f64| -> Moments {
            let mut moments = Moments {
                pred_len: horizon,
                channels: 4,
                bars: vec![4096.; horizon],
                target: vec![0.; cells],
                target_square: vec![0.; cells],
                anchor_square: vec![0.; cells],
                anchor_offset: vec![0.; cells],
                offset_square: vec![0.; cells],
                anchor_target: vec![0.; cells],
                offset_target: vec![0.; cells],
            };
            for bar in 0..horizon {
                for channel in 0..4 {
                    let cell = bar * 4 + channel;
                    let anchor_square = 4096. * (bar + 1) as f64;
                    let offset_square = if channel == CLOSE_CHANNEL { 0. } else { 1024. };
                    moments.anchor_square[cell] = anchor_square;
                    moments.offset_square[cell] = offset_square;
                    moments.anchor_target[cell] = anchor_square * anchor_gain(bar);
                    moments.offset_target[cell] = offset_square;
                    let forecast_square = anchor_square + offset_square;
                    let forecast_target =
                        moments.anchor_target[cell] + moments.offset_target[cell];
                    moments.target_square[cell] =
                        (forecast_target / 0.1).powi(2) / forecast_square;
                }
            }
            moments
        };
        // The measured signature: a held-out amplitude that sweeps across the horizon axis
        // against a training amplitude that sits flat near 1. That contrast IS the finding the
        // panel is built to make legible, so the fixture has to carry it.
        let fit_moments = build(&|bar| 3.0 * ((bar + 1) as f64).powf(-0.6));
        let training_moments = build(&|_| 1.02);
        let scored_moments = build(&|bar| 2.6 * ((bar + 1) as f64).powf(-0.55));
        let blocks = Blocks {
            calibration_first_origin_ms: 1,
            calibration_last_origin_ms: 2,
            calibration_last_target_ms: 3,
            calibration_origins: 4096,
            evaluation_first_origin_ms: 9,
            evaluation_last_origin_ms: 10,
            evaluation_origins: 2048,
            purge_gap_ms: 6,
        };
        let calibration = MeanCalibration::fit(blocks.clone(), &fit_moments).unwrap();
        let mut applied = calibration.frozen();
        // Two gated horizons, one of each kind: the panel has to distinguish a measured
        // sign inversion from an absent measurement.
        applied.measured_anchor[3] = Some(-0.07);
        applied.measured_anchor[7] = None;
        let panels = AmplitudePanels {
            epoch: 2,
            step: 2000,
            tickers: 256,
            applied: &applied,
            fit: Some(AmplitudeFit {
                calibration: &calibration,
                moments: &fit_moments,
                training: Some(&training_moments),
                training_origins: 512,
            }),
            scored: vec![AmplitudeSplit {
                split: SAMPLE,
                origins: 2048,
                emission: Emission::Uncalibrated,
                moments: &scored_moments,
            }],
        };
        write_amplitude(&root, &panels).unwrap();

        let gain = read_report(root.join("timexer_segment_calibration_gain.report.bin")).unwrap();
        assert_eq!(gain.y_label.as_deref(), Some(OPTIMAL_GAIN_UNIT));
        let ReportKind::IndexedLines { steps, series } = &gain.kind else {
            panic!("the gain panel is horizon-indexed");
        };
        assert_eq!(steps, &(1..=horizon as u64).collect::<Vec<u64>>());
        let label = |needle: &str| {
            series
                .iter()
                .find(|line| line.label.contains(needle))
                .unwrap_or_else(|| panic!("the gain panel is missing a {needle:?} series"))
        };
        assert_eq!(
            label("close-anchor gain applied while scoring").values,
            applied.anchor.iter().map(|g| *g as f32).collect::<Vec<f32>>()
        );
        assert_eq!(
            label("intrabar-offset gain applied while scoring").values,
            applied.offset.iter().map(|g| *g as f32).collect::<Vec<f32>>()
        );
        // All four DECODED channels' own optima, which is what generalizing the close-only fit
        // to every channel is visible as.
        for channel in ["open", "high", "low", "close"] {
            let series = label(&format!("{CALIBRATION} {channel} MSE-optimal gain (fitted on)"));
            assert!(series.values.iter().all(|value| value.is_finite()));
        }
        // THE mechanism series, and it must be the ONLY one carrying the training split word:
        // a second in-sample series on this axis would make the contrast unreadable.
        let training = label(&format!("{TRAINING} close MSE-optimal gain"));
        assert!(training
            .values
            .iter()
            .all(|value| (*value - 1.02).abs() < 1e-4));
        assert_eq!(
            series
                .iter()
                .filter(|line| line.label.starts_with(TRAINING))
                .count(),
            1
        );
        // The held-out sweep against that flat training curve, on the same base and the same
        // unit, which is the only reason they may share an axis.
        let held_out = label(&format!("{SAMPLE} close MSE-optimal gain on the emitted mean"));
        assert!(held_out.values[0] / held_out.values[horizon - 1] > 2.);
        let ceiling = label("amplification ceiling");
        assert!(ceiling.values.iter().all(|value| *value >= 1.));
        assert!(applied
            .anchor
            .iter()
            .zip(&ceiling.values)
            .all(|(applied, bound)| *applied as f32 <= *bound));
        assert_eq!(label("perfect amplitude calibration 1.0").values, vec![1.0f32; horizon]);
        // The gate, in the title, at its own value and with the unmeasured horizon named as
        // unmeasured rather than as a zero.
        assert!(
            gain.title.contains("2 of 8 horizons GATED OUT OF SIZING")
                && gain.title.contains("h=4 at -0.0700")
                && gain.title.contains("h=8 unmeasured"),
            "{}",
            gain.title
        );
        assert!(gain.title.contains("out-of-sample shrinkage problem or an in-sample objective one"));

        // The ratio panel: its own base, its own unit, and both readings of one pass.
        let ratio =
            read_report(root.join("timexer_segment_amplitude_calibration.report.bin")).unwrap();
        assert_eq!(ratio.y_label.as_deref(), Some(BEST_SCALE_UNIT));
        assert_ne!(gain.y_label, ratio.y_label, "a gain and an MSE ratio are different units");
        let ReportKind::IndexedLines { series, .. } = &ratio.kind else {
            panic!("the ratio panel is horizon-indexed");
        };
        let line = |needle: &str| {
            series
                .iter()
                .find(|line| line.label.contains(needle))
                .unwrap_or_else(|| panic!("the ratio panel is missing a {needle:?} series"))
        };
        // Exact algebra on the one pass, not a second scoring pass: the calibrated series is
        // the same moments evaluated at the applied gain.
        for bar in 0..horizon {
            let expected =
                scored_moments.pooled_gained_ratio(bar, applied.anchor[bar], applied.offset[bar]);
            assert!(
                (line(&format!("{SAMPLE} {NEUTRAL} four-channel ratio, calibrated")).values[bar]
                    - expected as f32)
                    .abs()
                    < 1e-6
            );
            assert!(
                (line(&format!("{SAMPLE} {NEUTRAL} four-channel ratio, uncalibrated")).values[bar]
                    - scored_moments.pooled_ratio(bar) as f32)
                    .abs()
                    < 1e-6
            );
        }
        assert_eq!(line("persistence 1.0").values, vec![1.0f32; horizon]);
        for base in [
            "timexer_segment_calibration_gain",
            "timexer_segment_amplitude_calibration",
            "timexer_segment_calibration_moments",
        ] {
            assert!(
                TIMEXER_SEGMENT_REPORT_BASES.contains(&base),
                "{base} is written but unregistered, so the TUI never scans for it"
            );
        }
        // A split whose horizon count disagrees with the applied curve is a pairing fault, not
        // a chart to draw with one axis silently truncated.
        let short = build(&|_| 1.);
        let mut short = short;
        short.pred_len = horizon - 1;
        short.bars.truncate(horizon - 1);
        assert!(write_amplitude(
            &root,
            &AmplitudePanels {
                scored: vec![AmplitudeSplit {
                    split: SAMPLE,
                    origins: 1,
                    emission: Emission::Uncalibrated,
                    moments: &short,
                }],
                fit: None,
                ..panels
            }
        )
        .is_err());
        fs::remove_dir_all(&root).unwrap();
    }

    /// The `calibrated` series must DIFFER from the `uncalibrated` one by exactly the amount
    /// the applied gain implies, and the difference is asserted against longhand arithmetic on
    /// the moment fields rather than against `pooled_gained_ratio`, which would be checking the
    /// reduction against itself.
    ///
    /// This is the test job 6004 needed and did not have. Every `..., calibrated` series in
    /// that arm was BIT-IDENTICAL to its `..., uncalibrated` twin at all 192 horizons - `held-out
    /// sample` market-neutral close ratio 0.99558365 in both at h=192 - because the fit had
    /// refused and frozen a gain of 1, and at `anchor = offset = 1` the two branches of
    /// [`AmplitudeSplit::ratios`] are the same call. Pinning the closed-form DELTA is what makes
    /// a unit gain unable to masquerade as a calibration.
    #[test]
    fn the_calibrated_ratio_differs_from_the_uncalibrated_one_by_the_applied_gain() {
        use crate::torch::timexer_segment::calibration::{Blocks, FrozenGain, Moments};
        let root = std::env::temp_dir().join(format!("timexer-gain-delta-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let horizon = 6;
        let cells = horizon * 4;
        let mut moments = Moments {
            pred_len: horizon,
            channels: 4,
            bars: vec![2048.; horizon],
            target: vec![0.; cells],
            target_square: vec![0.; cells],
            anchor_square: vec![0.; cells],
            anchor_offset: vec![0.; cells],
            offset_square: vec![0.; cells],
            anchor_target: vec![0.; cells],
            offset_target: vec![0.; cells],
        };
        // Every moment distinct per (horizon, channel), and the offset column empty on the
        // close channel exactly as `decode_joint` leaves it, so no term can cancel by accident.
        for bar in 0..horizon {
            for channel in 0..4 {
                let cell = bar * 4 + channel;
                let intrabar = channel != CLOSE_CHANNEL;
                moments.anchor_square[cell] = 100. * (bar + 1) as f64;
                moments.offset_square[cell] = if intrabar { 7. + channel as f64 } else { 0. };
                moments.anchor_offset[cell] = if intrabar { 3. + bar as f64 } else { 0. };
                moments.anchor_target[cell] = 60. * (bar + 1) as f64;
                moments.offset_target[cell] = if intrabar { 5. } else { 0. };
                moments.target_square[cell] = 500. * (bar + 1) as f64 + 20.;
            }
        }
        // A hand-built curve, never a fitted one: the point is the ARITHMETIC the report layer
        // applies, and both coordinates are away from 1 at every horizon in both directions.
        let applied = FrozenGain {
            estimator: "closed-form delta fixture".into(),
            blocks: Blocks {
                calibration_first_origin_ms: 1,
                calibration_last_origin_ms: 2,
                calibration_last_target_ms: 3,
                calibration_origins: 32_768,
                evaluation_first_origin_ms: 5,
                evaluation_last_origin_ms: 6,
                evaluation_origins: 7,
                purge_gap_ms: 8,
            },
            anchor: (0..horizon).map(|h| 0.4 + 0.05 * h as f64).collect(),
            offset: (0..horizon).map(|h| 1.3 - 0.05 * h as f64).collect(),
            intercept_refused: Vec::new(),
            measured_anchor: vec![Some(0.6); horizon],
        };
        let panels = AmplitudePanels {
            epoch: 1,
            step: 2500,
            tickers: 4873,
            applied: &applied,
            fit: None,
            scored: vec![AmplitudeSplit {
                split: SAMPLE,
                origins: 2048,
                emission: Emission::Uncalibrated,
                moments: &moments,
            }],
        };
        write_amplitude(&root, &panels).unwrap();
        let report =
            read_report(root.join("timexer_segment_amplitude_calibration.report.bin")).unwrap();
        let ReportKind::IndexedLines { series, .. } = &report.kind else {
            panic!("the ratio panel is horizon-indexed");
        };
        let line = |needle: String| {
            series
                .iter()
                .find(|line| line.label == needle)
                .unwrap_or_else(|| panic!("the ratio panel is missing {needle:?}"))
        };
        let calibrated = line(format!("{SAMPLE} {NEUTRAL} four-channel ratio, calibrated"));
        let uncalibrated = line(format!("{SAMPLE} {NEUTRAL} four-channel ratio, uncalibrated"));
        for bar in 0..horizon {
            let (mut anchor_square, mut offset_square, mut anchor_offset) = (0., 0., 0.);
            let (mut anchor_target, mut offset_target, mut persistence) = (0., 0., 0.);
            for channel in 0..4 {
                let cell = bar * 4 + channel;
                anchor_square += moments.anchor_square[cell];
                offset_square += moments.offset_square[cell];
                anchor_offset += moments.anchor_offset[cell];
                anchor_target += moments.anchor_target[cell];
                offset_target += moments.offset_target[cell];
                persistence += moments.target_square[cell];
            }
            let (gain_a, gain_o) = (applied.anchor[bar], applied.offset[bar]);
            let expected_uncalibrated = (anchor_square
                + 2. * anchor_offset
                + offset_square
                - 2. * (anchor_target + offset_target)
                + persistence)
                / persistence;
            let expected_calibrated = (gain_a * gain_a * anchor_square
                + gain_o * gain_o * offset_square
                + 2. * gain_a * gain_o * anchor_offset
                - 2. * gain_a * anchor_target
                - 2. * gain_o * offset_target
                + persistence)
                / persistence;
            let delta = expected_calibrated - expected_uncalibrated;
            assert!(
                delta.abs() > 0.02,
                "the fixture's own gain moves the ratio by only {delta} at h={}, so the \
                 assertions below would pass on an identity gain too",
                bar + 1
            );
            assert!(
                (uncalibrated.values[bar] - expected_uncalibrated as f32).abs() < 2e-6,
                "h={}: uncalibrated {} against {expected_uncalibrated}",
                bar + 1,
                uncalibrated.values[bar]
            );
            assert!(
                (calibrated.values[bar] - expected_calibrated as f32).abs() < 2e-6,
                "h={}: calibrated {} against {expected_calibrated}",
                bar + 1,
                calibrated.values[bar]
            );
            assert_ne!(
                calibrated.values[bar],
                uncalibrated.values[bar],
                "h={}: the calibrated series is bit-identical to the uncalibrated one, which is \
                 what a frozen gain of 1 produces - the calibration was a no-op",
                bar + 1
            );
        }
        // The other emission direction, on the same moments and the same curve: a pass whose
        // mean already carried the gain reports the un-gained ratio at `1/g`. Both series must
        // still move, or a calibrated pass would report its own emission twice.
        let inverted = AmplitudeSplit {
            split: SAMPLE,
            origins: 2048,
            emission: Emission::Calibrated,
            moments: &moments,
        };
        let (removed, as_emitted) = inverted.ratios(&applied, Some(CLOSE_CHANNEL));
        for bar in 0..horizon {
            let expected = moments.channel_gained_ratio(
                bar,
                CLOSE_CHANNEL,
                applied.anchor[bar].recip(),
                applied.offset[bar].recip(),
            );
            assert!((removed[bar] - expected).abs() < 1e-12, "h={}", bar + 1);
            assert_ne!(removed[bar], as_emitted[bar], "h={}", bar + 1);
        }
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

/// What the per-horizon sub-origin decimation actually did over one report interval.
///
/// Two bases, because the panel answers its question in two units five orders of magnitude
/// apart: the shares say whether the realized pattern is the intended one, and the counts say
/// whether what survived is still a large enough sample for the added gradient noise to be the
/// intended temperature. The load-bearing series is `compensated`, which MUST sit at 1.0: it
/// is the unbiasedness the whole knob rests on, measured from the masks the host actually
/// uploaded rather than assumed from the construction.
///
/// Horizon-indexed and rewritten in place at every interval, like the rest of the per-horizon
/// family: the profile is constant within a run and only the realized draw moves.
pub fn write_horizon_decimation(
    output: &Path,
    epoch: usize,
    step: usize,
    batch_size: usize,
    interval: &super::supervision::DecimationInterval,
) -> Result<()> {
    ensure!(
        interval.steps > 0 && batch_size > 0 && interval.active_origins > 0,
        "a decimation interval with no step, no row or no active sub-origin has nothing to chart"
    );
    let axis: Vec<u64> = (1..=interval.factors.len() as u64).collect();
    let (intended, realized, compensated) = (
        interval.intended_keep_fraction(),
        interval.keep_fraction(),
        interval.compensated_share(),
    );
    // Mask elements one step supervises at a horizon: rows times surviving sub-origins. The
    // channel axis is a constant factor 4 the reduction applies afterwards, so it is left out
    // rather than folded in and quietly multiplying every curve.
    let per_step = batch_size as f64 / interval.steps as f64;
    let elements = |share: &[f64]| -> Vec<f32> {
        share
            .iter()
            .map(|value| (value * interval.active_origins as f64 * batch_size as f64) as f32)
            .collect()
    };
    let worst = compensated
        .iter()
        .map(|share| (share - 1.).abs())
        .fold(0., f64::max);
    let title = format!(
        "CausalPatch epoch {epoch} step {step} | what did the horizon decimation keep? | origin \
         stride {} bars, {} active sub-origins per row, batch {batch_size}, {} steps in this \
         interval; factors /{} at h = 1 to /{} at h = {}; realized surviving elements {:.0} to \
         {:.0} per step, compensated to {:.0} and {:.0} against an undecimated {:.0}; worst \
         compensated deviation from 1.0 is {worst:.3e}",
        interval.stride,
        interval.active_origins,
        interval.steps,
        interval.factors[0],
        interval.factors[interval.factors.len() - 1],
        interval.factors.len(),
        interval.kept[0] * per_step,
        interval.kept[interval.kept.len() - 1] * per_step,
        interval.kept[0] * per_step * interval.factors[0] as f64,
        interval.kept[interval.kept.len() - 1] * per_step
            * interval.factors[interval.factors.len() - 1] as f64,
        interval.active_origins as f64 * batch_size as f64,
    );
    write_report(
        output.join(super::supervision::DECIMATION_BASE.to_owned() + ".report.bin"),
        &Report {
            title: title.clone(),
            x_label: Some("forecast horizon h, bars".to_owned()),
            y_label: Some(
                "share of the undecimated active sub-origin lattice at this horizon; the \
                 compensated series is the unbiasedness check and must read 1.0"
                    .to_owned(),
            ),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: axis.clone(),
                series: vec![
                    ReportSeries {
                        label: "intended keep fraction 1/decim".to_owned(),
                        values: intended.iter().map(|value| *value as f32).collect(),
                    },
                    ReportSeries {
                        label: "realized keep fraction".to_owned(),
                        values: realized.iter().map(|value| *value as f32).collect(),
                    },
                    ReportSeries {
                        label: "realized compensated share (1.0 is unbiased)".to_owned(),
                        values: compensated.iter().map(|value| *value as f32).collect(),
                    },
                ],
            },
        },
    )?;
    write_report(
        output.join(super::supervision::DECIMATION_COUNT_BASE.to_owned() + ".report.bin"),
        &Report {
            title,
            x_label: Some("forecast horizon h, bars".to_owned()),
            y_label: Some(
                "supervised mask elements per optimizer step (rows x sub-origins), before the \
                 constant 4-channel factor"
                    .to_owned(),
            ),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: axis,
                series: vec![
                    ReportSeries {
                        label: "undecimated lattice elements".to_owned(),
                        values: vec![
                            (interval.active_origins as f64 * batch_size as f64) as f32;
                            interval.factors.len()
                        ],
                    },
                    ReportSeries {
                        label: "surviving elements".to_owned(),
                        values: elements(&realized),
                    },
                    ReportSeries {
                        label: "compensated effective elements".to_owned(),
                        values: elements(&compensated),
                    },
                ],
            },
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

/// Fallback account clock, used ONLY when an evaluation carries no measurable window.
///
/// A regular US equity session is 78 five-minute bars and a year is 252 of them, so 19 656.
/// That is NOT the cadence this account simulates: its tape retains every observed mark
/// including extended hours, which is roughly 191 bars a session and 48 000 a year. Every
/// annualization below therefore measures its own interval count off the evaluation via
/// [`BookCadence`] and reaches these constants only for a degenerate window, where no
/// annualization is meaningful anyway.
const BOOK_BARS_PER_YEAR: f64 = 19_656.0;
const BOOK_SESSIONS_PER_YEAR: f64 = 252.0;
const BOOK_MONTHS_PER_YEAR: f64 = 12.0;

/// Book panel axes. Each states its denominator, because the book charts a return per
/// interval, a cost per traded dollar and a cost per dollar of gross P&L, and those three
/// share no axis: a 0.4 bps bar return beside a 600 bps month, or an 8 bps cost beside the
/// 0.9 share of P&L it consumed, renders one of each pair as a flat line.
const BOOK_EQUITY_UNIT: &str = "equity as a multiple of the run's own initial capital \
                                (1.0 = flat; normalized because a $25k curve and a $100M \
                                curve in USD cannot share an axis)";
/// The calendar-month series is the one interval whose trailing partial period is dropped
/// outright, so its mean times its count under-reports the window's total by that stub. Said
/// on the axis because a reader reconciling the panel against total P&L will otherwise read
/// the stub as a measurement error.
const BOOK_INTERVAL_UNIT: &str = "mean realized net return per interval (basis points of \
                                  equity; net of every modeled cost, NaN = fewer than two \
                                  complete intervals were observed; every series times its \
                                  own interval count reconciles with total net P&L except \
                                  the calendar month, whose trailing partial month is \
                                  dropped)";
const BOOK_SHARPE_UNIT: &str = "annualized Sharpe (dimensionless; interval mean / interval \
                                sample std × sqrt(intervals per year); NaN on a zero-variance \
                                or single-observation interval)";
const BOOK_RATIO_UNIT: &str = "annualized risk-adjusted ratio (dimensionless; NaN where the \
                               denominator is zero, which is not a good ratio)";
const BOOK_COST_UNIT: &str = "basis points of one-way traded notional (what each dollar \
                              transacted cost; the breakeven edge the signal must clear)";
const BOOK_COST_SHARE_UNIT: &str = "fraction of gross pre-cost P&L (1.0 = the component ate \
                                    the whole edge; negative = gross P&L was itself negative, \
                                    which no cost reduction rescues)";
const BOOK_SYMBOL_UNIT: &str = "symbols";
const BOOK_NAME_UNIT: &str = "names (mean over decision frames)";
const BOOK_EXPOSURE_UNIT: &str = "fraction (each series names its denominator: equity, the \
                                  tradable universe, or decision frames)";
const BOOK_TURNOVER_UNIT: &str = "annualized one-way turnover (multiples of equity per year)";
const BOOK_DRAWDOWN_UNIT: &str = "drawdown (fraction of peak equity; 0 = at the high-water \
                                  mark, negative = underwater)";
const BOOK_IC_UNIT: &str = "within-timestamp cross-ticker rank information coefficient \
                            (dimensionless; 0 = no information)";
const BOOK_HIT_UNIT: &str = "fraction of names whose realized sign matched the forecast's \
                             (> 0.5 = skill)";
const BOOK_COVERAGE_UNIT: &str = "fraction of realizations inside the predictive interval \
                                  (below nominal = the predictive σ is too tight, so the \
                                  ex-ante vol target undershoots realized vol)";

const BOOK_AUM_AXIS: &str = "account AUM (USD, ascending)";
const BOOK_BAR_AXIS: &str = "five-minute bar ordinal from the first scheduled frame";
const BOOK_HORIZON_AXIS: &str = "forecast horizon (five-minute bars)";

/// Every chart base [`write_book`] can produce, named here so the writer, the registry in
/// [`shared::report::TIMEXER_SEGMENT_REPORT_BASES`] and the registration test cannot disagree
/// about a string. Sixteen bases and not the nine questions they answer, for the reason the
/// registry's own header gives: the split is by UNIT, and an annualized return near 0.1 on
/// the same axis as a Sharpe near 1.5, or a cost in basis points beside the same cost as a
/// share of P&L, renders one of each pair unreadable.
pub(super) const BOOK_REPORT_BASES: &[&str] = &[
    "timexer_book_equity",
    "timexer_book_interval_return",
    "timexer_book_interval_sharpe",
    "timexer_book_annualized",
    "timexer_book_annualized_ratio",
    "timexer_book_cost_decomposition",
    "timexer_book_cost_share",
    "timexer_book_cost_coverage",
    "timexer_book_exposure",
    "timexer_book_turnover",
    "timexer_book_names",
    "timexer_book_drawdown",
    "timexer_book_signal_ic",
    "timexer_book_uncertainty_ic",
    "timexer_book_uncertainty_hit_rate",
    "timexer_book_calibration",
];

/// The bases that exist only when the diagnostics pass ran. `diagnostics: None` is a
/// legitimate configuration - the AUM sweep is the deliverable and the signal audit is a
/// separate optional pass - so these are ABSENT rather than written with reference lines and
/// no measurement, which is a panel a reader cannot distinguish from a measured zero.
pub(super) const BOOK_DIAGNOSTIC_BASES: &[&str] = &[
    "timexer_book_signal_ic",
    "timexer_book_uncertainty_ic",
    "timexer_book_uncertainty_hit_rate",
    "timexer_book_calibration",
];

/// The cost ledger, in the order a reader should read it: what the broker bills per share,
/// what the regulators take, what the market charges for immediacy, and what accrues on
/// calendar time. The measured impact term and the flat slippage proxy are SEPARATE rows and
/// are never summed: a run priced from the Roll/ADV calibration pays the former and zeroes
/// the latter, a flat-cost run does the reverse, and a panel that added them would double
/// count whichever one the run actually paid.
const BOOK_COST_COMPONENTS: [(&str, &str); 6] = [
    ("commission", "commission_usd"),
    ("regulatory (SEC + TAF + CAT)", "regulatory_usd"),
    ("spread", "spread_usd"),
    ("measured impact", "impact_usd"),
    ("flat slippage proxy", "slippage_usd"),
    ("borrow", "borrow_usd"),
];

/// The account's ACTUAL clock, measured off one evaluation instead of assumed.
///
/// Every annualization on these panels is a count of intervals per year, and the count is a
/// property of the tape rather than of the exchange calendar: the account marks and accrues on
/// every observed bar, extended hours included, so it simulates roughly 191 bars a session
/// where a 09:30-16:00 regular session holds 78. Annualizing that series with 19 656 bars a
/// year understates the geometric return in log space by 2.4x and every vol, Sharpe and
/// Sortino by sqrt(2.4), which is precisely how a 9.6% year came to be charted as 3.8%.
///
/// Derived from [`AccountEvaluation::span_years`] and the series' own lengths, so the count
/// and the window it is a count over cannot disagree. Calendar months are the one exception:
/// twelve a year is an identity, not an estimate.
#[derive(Clone, Copy)]
struct BookCadence {
    /// Wall-clock length of the window, the denominator of every count below.
    years: f64,
    bars_per_year: f64,
    sessions_per_year: f64,
}

impl BookCadence {
    fn measure(evaluation: &super::portfolio::AccountEvaluation) -> Self {
        let years = evaluation.span_years();
        let per_year = |count: usize, fallback: f64| {
            if years > 0.0 && count > 1 {
                count as f64 / years
            } else {
                fallback
            }
        };
        Self {
            years,
            bars_per_year: per_year(evaluation.points.len(), BOOK_BARS_PER_YEAR),
            sessions_per_year: per_year(evaluation.daily.len(), BOOK_SESSIONS_PER_YEAR),
        }
    }

    fn bars_per_session(&self) -> f64 {
        self.bars_per_year / self.sessions_per_year
    }

    /// The annualized panel's axis, stating the window and the bar count it annualized with
    /// rather than a constant a reader would have to trust.
    fn annual_unit(&self) -> String {
        format!(
            "annualized fraction of equity (geometric net return over the window's own \
             {:.3} years, return std and downside std x sqrt({:.0} simulated bars/yr measured \
             from the run itself rather than a 19 656-bar regular-session year), max drawdown \
             as a fraction of peak equity and therefore negative)",
            self.years, self.bars_per_year
        )
    }
}

/// How one charted interval's returns are assembled from the account's own series.
enum BookIntervalSource {
    /// Compound this many consecutive simulated bar returns.
    Bars(usize),
    /// Compound this many consecutive New-York-calendar session returns.
    Sessions(usize),
    /// The account's own calendar-month aggregation.
    Months,
}

struct BookInterval {
    /// Names the interval in every series label that reports it.
    name: &'static str,
    source: BookIntervalSource,
}

/// The five intervals the headline return panel reports, shortest first. Hours and weeks are
/// FIXED-LENGTH blocks of the interval below them rather than wall-clock calendar units: a
/// holiday-shortened calendar week and a full one are not the same bet, and a mean over both
/// denominators answers neither "what does a week earn" nor "what does a session earn". The
/// month is the account's own New-York calendar month, the one interval where the account
/// already did the grouping and where a fixed block would drift against the reporting period
/// a reader compares against.
const BOOK_INTERVALS: [BookInterval; 5] = [
    BookInterval {
        name: "five-minute bar",
        source: BookIntervalSource::Bars(1),
    },
    BookInterval {
        name: "12-bar hour",
        source: BookIntervalSource::Bars(12),
    },
    BookInterval {
        name: "session",
        source: BookIntervalSource::Sessions(1),
    },
    BookInterval {
        name: "5-session week",
        source: BookIntervalSource::Sessions(5),
    },
    BookInterval {
        name: "calendar month",
        source: BookIntervalSource::Months,
    },
];

impl BookInterval {
    /// One realized return per COMPLETE interval. A trailing partial block is dropped rather
    /// than annualized as if it were whole, which is the only treatment under which the mean
    /// of this series is a return per interval of the stated length.
    fn returns(&self, evaluation: &super::portfolio::AccountEvaluation) -> Vec<f64> {
        match self.source {
            BookIntervalSource::Bars(block) => {
                book_compound(&book_returns(&evaluation.points, "return"), block)
            }
            BookIntervalSource::Sessions(block) => {
                book_compound(&book_returns(&evaluation.daily, "return_fraction"), block)
            }
            BookIntervalSource::Months => book_returns(&evaluation.monthly, "return_fraction"),
        }
    }

    /// Intervals of this length per year, on the cadence the run actually simulated.
    fn per_year(&self, cadence: &BookCadence) -> f64 {
        match self.source {
            BookIntervalSource::Bars(block) => cadence.bars_per_year / block.max(1) as f64,
            BookIntervalSource::Sessions(block) => {
                cadence.sessions_per_year / block.max(1) as f64
            }
            BookIntervalSource::Months => BOOK_MONTHS_PER_YEAR,
        }
    }

    /// Interval length in simulated bars, for the labels that state it. Measured, so a session
    /// reads as the ~191 extended-hours bars the account really marked rather than the 78 a
    /// regular session would hold.
    fn bars(&self, cadence: &BookCadence) -> f64 {
        match self.source {
            BookIntervalSource::Bars(block) => block.max(1) as f64,
            BookIntervalSource::Sessions(block) => {
                block.max(1) as f64 * cadence.bars_per_session()
            }
            BookIntervalSource::Months => cadence.bars_per_year / BOOK_MONTHS_PER_YEAR,
        }
    }
}

/// Mean, sample dispersion and downside dispersion of one interval's realized returns, or an
/// all-NaN record when fewer than two complete intervals were observed. One observation is
/// reported as UNMEASURED on purpose: a single realization carries no dispersion and no
/// Sharpe, and a lone monthly return drawn as "the mean monthly return" beside a mean over
/// 20 000 bars is exactly the misreading these panels exist to prevent.
#[derive(Clone, Copy)]
struct BookMoments {
    mean: f64,
    std: f64,
    downside: f64,
}

impl BookMoments {
    /// Annualized Sharpe on the interval actually charted. NaN on a degenerate series: a
    /// constant return has no dispersion to divide by, and the infinity `mean / 0` produces
    /// would render as the best book in the sweep.
    fn sharpe(&self, per_year: f64) -> f64 {
        if self.std > 0.0 {
            self.mean / self.std * per_year.sqrt()
        } else {
            f64::NAN
        }
    }

    fn sortino(&self, per_year: f64) -> f64 {
        if self.downside > 0.0 {
            self.mean / self.downside * per_year.sqrt()
        } else {
            f64::NAN
        }
    }
}

fn book_moments(values: &[f64]) -> BookMoments {
    let usable: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if usable.len() < 2 {
        return BookMoments {
            mean: f64::NAN,
            std: f64::NAN,
            downside: f64::NAN,
        };
    }
    let count = usable.len() as f64;
    let mean = usable.iter().sum::<f64>() / count;
    let variance = usable.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1.0);
    // Downside deviation on the same denominator as the std beside it, so Sortino and Sharpe
    // differ only in which realizations they charge for.
    let downside = usable.iter().map(|v| v.min(0.0).powi(2)).sum::<f64>() / (count - 1.0);
    BookMoments {
        mean,
        std: variance.sqrt(),
        downside: downside.sqrt(),
    }
}

fn book_returns(points: &[super::portfolio::AccountPoint], key: &str) -> Vec<f64> {
    points
        .iter()
        .filter_map(|point| point.values.get(key).copied())
        .collect()
}

fn book_compound(returns: &[f64], block: usize) -> Vec<f64> {
    returns
        .chunks_exact(block.max(1))
        .map(|chunk| chunk.iter().map(|r| 1.0 + r).product::<f64>() - 1.0)
        .collect()
}

/// A summary scalar, or NaN when the run never measured it. Never 0: an account that
/// transacted nothing has no cost per traded basis point, and a 0 there would read as
/// execution that was free.
fn book_stat(summary: &BTreeMap<String, f64>, key: &str) -> f64 {
    summary
        .get(key)
        .copied()
        .filter(|value| value.is_finite())
        .unwrap_or(f64::NAN)
}

fn book_mean(
    points: &[super::portfolio::AccountPoint],
    key: &str,
    map: impl Fn(f64) -> f64,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for point in points {
        if let Some(value) = point.values.get(key).copied().filter(|v| v.is_finite()) {
            sum += map(value);
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn book_row(values: impl IntoIterator<Item = f64>) -> Vec<f32> {
    values.into_iter().map(|value| value as f32).collect()
}

/// `$25k`, `$1M`, `$100M`: the AUM as a reader says it. The numeric axis carries the exact
/// value, so this is legend shorthand only.
fn book_aum_label(aum: f64) -> String {
    let (value, suffix) = if aum >= 1e9 {
        (aum / 1e9, "B")
    } else if aum >= 1e6 {
        (aum / 1e6, "M")
    } else if aum >= 1e3 {
        (aum / 1e3, "k")
    } else {
        (aum, "")
    };
    let mut text = format!("{value:.3}");
    while text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    format!("${text}{suffix}")
}

/// Every scalar the variant × AUM panels draw, measured once per run. Assembled from the
/// account's own summary and points rather than recomputed from the tape: the simulator is
/// the only thing that knows what actually filled, and a second implementation of the same
/// arithmetic here would eventually disagree with it and there would be no way to tell which
/// number the charts were showing.
struct BookMetrics {
    cadence: BookCadence,
    annualized_return: f64,
    annualized_vol: f64,
    annualized_downside_vol: f64,
    max_drawdown: f64,
    annualized_turnover: f64,
    intervals: [BookMoments; BOOK_INTERVALS.len()],
    cost_usd: [f64; BOOK_COST_COMPONENTS.len()],
    total_cost_usd: f64,
    traded_notional_usd: f64,
    gross_pnl_usd: f64,
    spread_fallback_symbols: f64,
    spread_measured_symbols: f64,
    universe_symbols: f64,
    held_names: f64,
    active_names: f64,
    gross_exposure: f64,
    net_exposure: f64,
    absolute_net_exposure: f64,
    active_fraction: f64,
    turnover_capped_fraction: f64,
}

impl BookMetrics {
    fn measure(evaluation: &super::portfolio::AccountEvaluation) -> Self {
        let summary = &evaluation.summary;
        let cadence = BookCadence::measure(evaluation);
        let intervals: [BookMoments; BOOK_INTERVALS.len()] =
            std::array::from_fn(|index| book_moments(&BOOK_INTERVALS[index].returns(evaluation)));
        // Geometric, from the account's own terminal equity over the window's own wall-clock
        // length: the book compounds, and an arithmetic (bars per year) x of a mean bar return
        // overstates a volatile path, so it is not the number an investor receives. Using the
        // measured span rather than a bar count times an assumed bars-per-year is what makes
        // this figure identical to the headline table's `ann_ret`.
        let growth = 1.0 + book_stat(summary, "total_return_fraction");
        let annualized_return = if growth > 0.0 && cadence.years > 0.0 {
            growth.powf(1.0 / cadence.years) - 1.0
        } else {
            f64::NAN
        };
        let universe_symbols = book_stat(summary, "universe_count");
        let held_names = book_mean(&evaluation.points, "held_count", |value| value);
        let decisions = book_stat(summary, "decision_count");
        let bar_annualization = cadence.bars_per_year.sqrt();
        Self {
            cadence,
            annualized_return,
            annualized_vol: intervals[0].std * bar_annualization,
            annualized_downside_vol: intervals[0].downside * bar_annualization,
            max_drawdown: book_stat(summary, "max_drawdown_fraction"),
            annualized_turnover: book_stat(summary, "turnover_annualized"),
            intervals,
            cost_usd: std::array::from_fn(|index| {
                book_stat(summary, BOOK_COST_COMPONENTS[index].1)
            }),
            total_cost_usd: book_stat(summary, "costs_usd"),
            traded_notional_usd: book_stat(summary, "traded_notional_usd"),
            gross_pnl_usd: book_stat(summary, "gross_pnl_usd"),
            spread_fallback_symbols: book_stat(summary, "spread_fallback_count"),
            spread_measured_symbols: book_stat(summary, "spread_measured_count"),
            universe_symbols,
            held_names,
            active_names: book_stat(summary, "mean_active_names"),
            gross_exposure: book_stat(summary, "mean_gross_exposure"),
            net_exposure: book_stat(summary, "mean_net_exposure"),
            absolute_net_exposure: book_mean(&evaluation.points, "net_fraction", f64::abs),
            active_fraction: if universe_symbols > 0.0 {
                held_names / universe_symbols
            } else {
                f64::NAN
            },
            turnover_capped_fraction: if decisions > 0.0 {
                book_stat(summary, "turnover_capped_count") / decisions
            } else {
                f64::NAN
            },
        }
    }

    fn cost_bps(&self, index: usize) -> f64 {
        if self.traded_notional_usd > 0.0 {
            self.cost_usd[index] / self.traded_notional_usd * 1e4
        } else {
            f64::NAN
        }
    }

    fn total_cost_bps(&self) -> f64 {
        if self.traded_notional_usd > 0.0 {
            self.total_cost_usd / self.traded_notional_usd * 1e4
        } else {
            f64::NAN
        }
    }

    fn cost_share(&self, index: usize) -> f64 {
        if self.gross_pnl_usd != 0.0 && self.gross_pnl_usd.is_finite() {
            self.cost_usd[index] / self.gross_pnl_usd
        } else {
            f64::NAN
        }
    }

    fn total_cost_share(&self) -> f64 {
        if self.gross_pnl_usd != 0.0 && self.gross_pnl_usd.is_finite() {
            self.total_cost_usd / self.gross_pnl_usd
        } else {
            f64::NAN
        }
    }

    /// Return per unit of the deepest hole it was earned through. NaN at zero drawdown, which
    /// is either a book that never lost or a book that never traded, and neither has a Calmar.
    fn calmar(&self) -> f64 {
        if self.max_drawdown < 0.0 {
            self.annualized_return / self.max_drawdown.abs()
        } else {
            f64::NAN
        }
    }
}

/// The realized book, charted. Nine questions have to be answerable off these panels without
/// a reader recomputing anything: how did capital move, what does the book earn per bar, hour,
/// session, week and month, does that survive annualization, WHERE DOES IT STOP SCALING, what
/// did execution cost and against what, how much risk was on and was the book throttled by its
/// own turnover cap rather than by a weak signal, how deep was the worst hole, and - when the
/// diagnostics pass ran - whether the forecast's cross-sectional information, its predictive σ
/// and its calibration justify the selection the book made.
///
/// `runs` is the whole `(variant, AUM)` grid. The AUM axis is sorted ASCENDING here rather
/// than taken in the order the sweep was simulated, because the capacity decay is the SHAPE of
/// every AUM-indexed panel and a curve drawn on an unsorted axis is a zigzag that cannot be
/// told apart from an edge that is not monotone in size.
///
/// Every unmeasured quantity reaches the chart as NaN. A measured zero and an unmeasured
/// statistic are opposite claims: a book that transacted nothing has no cost per traded basis
/// point, and a 0 there would read as execution that was free.
pub(super) fn write_book(
    output: &Path,
    epoch: usize,
    step: usize,
    runs: &[(String, f64, super::portfolio::AccountEvaluation)],
    diagnostics: Option<&super::book_diagnostics::BookDiagnostics>,
) -> Result<()> {
    use std::collections::BTreeSet;

    ensure!(
        !runs.is_empty(),
        "a book report needs at least one simulated run"
    );
    let mut identities = BTreeSet::new();
    for (label, aum, evaluation) in runs {
        ensure!(!label.is_empty(), "every book run carries a variant label");
        ensure!(
            aum.is_finite() && *aum > 0.0,
            "book run {label} carries a nonpositive AUM"
        );
        ensure!(
            identities.insert((label.as_str(), aum.to_bits())),
            "book run {label} at AUM {aum} is duplicated; two identical legend entries cannot be told apart"
        );
        ensure!(
            !evaluation.points.is_empty(),
            "book run {label} at AUM {aum} produced no account points"
        );
        // The density of `points` IS the denominator of every annualization below, so a
        // series that contradicts the account's own bar census is refused rather than charted:
        // a second entry per fill, per AUM variant or per mark-and-decide pass would otherwise
        // arrive as a quietly diluted return per bar.
        let census = book_stat(&evaluation.summary, "bars_simulated");
        ensure!(
            !census.is_finite() || census == evaluation.points.len() as f64,
            "book run {label} at AUM {aum} simulated {census} bars but emitted {} account \
             points; the per-bar series would annualize on the wrong density",
            evaluation.points.len()
        );
    }
    let mut aums: Vec<f64> = runs.iter().map(|(_, aum, _)| *aum).collect();
    aums.sort_by(|left, right| left.partial_cmp(right).expect("validated finite AUM"));
    aums.dedup();
    let aum_steps: Vec<u64> = aums.iter().map(|aum| aum.round() as u64).collect();
    let mut variants: Vec<&str> = Vec::new();
    for (label, _, _) in runs {
        if !variants.contains(&label.as_str()) {
            variants.push(label.as_str());
        }
    }
    // The AUM the time-indexed panels are drawn at: the lower median of the sweep, not an end
    // of it. The smallest book in the sweep is dominated by the $0.35 per-order minimum and
    // the largest by impact, so either end would present an extreme as the representative
    // path. The value is in the title, so no reader has to guess which one it is.
    let reference_aum = aums[(aums.len() - 1) / 2];
    let metrics: Vec<BookMetrics> = runs
        .iter()
        .map(|(_, _, evaluation)| BookMetrics::measure(evaluation))
        .collect();
    // Label text quotes the FIRST run's measured cadence; every charted value annualizes on
    // its own run's. A sweep is one tape simulated at several sizes, so the two coincide.
    let cadence = metrics[0].cadence;
    let scope = format!(
        "timexer book backtest | validation partition (terminal test untouched) | {} variant(s) x {} AUM level(s), time-indexed panels at {} | epoch {epoch} step {step}",
        variants.len(),
        aums.len(),
        book_aum_label(reference_aum)
    );
    let chart = |base: &str,
                 question: &str,
                 unit: &str,
                 axis: &str,
                 scale: ScaleKind,
                 steps: &[u64],
                 series: Vec<ReportSeries>|
     -> Result<()> {
        ensure!(
            BOOK_REPORT_BASES.contains(&base),
            "book panel {base} is written but is not a registered book base"
        );
        ensure!(
            diagnostics.is_some() || !BOOK_DIAGNOSTIC_BASES.contains(&base),
            "book panel {base} was written without the diagnostics pass that measures it"
        );
        for line in &series {
            ensure!(
                line.values.len() == steps.len(),
                "book panel {base} series {} carries {} values for {} steps",
                line.label,
                line.values.len(),
                steps.len()
            );
            ensure!(
                line.values.iter().all(|v| v.is_nan() || v.is_finite()),
                "book panel {base} series {} carries an unrepresentable value",
                line.label
            );
        }
        write_report(
            output.join(format!("{base}.report.bin")),
            &Report {
                title: format!("{scope} | {question}"),
                x_label: Some(axis.to_owned()),
                y_label: Some(unit.to_owned()),
                scale,
                kind: ReportKind::IndexedLines {
                    steps: steps.to_vec(),
                    series,
                },
            },
        )?;
        Ok(())
    };
    // Every series on an AUM-indexed panel is one variant's quantity read across the sorted
    // axis, with NaN wherever that variant was not simulated at that AUM - a hole in the grid
    // is a run that did not happen, not a run that earned nothing.
    let aum_row = |variant: &str, pick: &dyn Fn(&BookMetrics) -> f64| -> Vec<f32> {
        book_row(aums.iter().map(|aum| {
            runs.iter()
                .zip(&metrics)
                .find(|(run, _)| run.0 == variant && run.1 == *aum)
                .map_or(f64::NAN, |(_, measured)| pick(measured))
        }))
    };
    let aum_series = |quantity: &str, pick: &dyn Fn(&BookMetrics) -> f64| -> Vec<ReportSeries> {
        variants
            .iter()
            .map(|variant| ReportSeries {
                label: format!("{variant} | {quantity}"),
                values: aum_row(variant, pick),
            })
            .collect()
    };
    let flat = |label: &str, value: f64, width: usize| ReportSeries {
        label: label.to_owned(),
        values: vec![value as f32; width],
    };

    // 1. The equity path, normalized. Every run on one axis in USD would draw the $100M book
    // and flatten the four below it; as a multiple of each run's own capital the five paths
    // are directly comparable and the capacity decay shows up as separation between them.
    let bars = runs
        .iter()
        .map(|(_, _, evaluation)| evaluation.points.len())
        .max()
        .expect("nonempty runs");
    let bar_steps: Vec<u64> = (0..bars as u64).collect();
    let equity = runs
        .iter()
        .map(|(label, aum, evaluation)| {
            let initial = book_stat(&evaluation.summary, "initial_cash_usd");
            let mut values = book_row(evaluation.points.iter().map(|point| {
                let equity = point.values.get("equity_usd").copied().unwrap_or(f64::NAN);
                if initial > 0.0 {
                    equity / initial
                } else {
                    f64::NAN
                }
            }));
            values.resize(bars, f32::NAN);
            ReportSeries {
                label: format!("{label} @ {}", book_aum_label(*aum)),
                values,
            }
        })
        .collect();
    chart(
        "timexer_book_equity",
        "how did the book's capital actually move?",
        BOOK_EQUITY_UNIT,
        BOOK_BAR_AXIS,
        ScaleKind::Symlog,
        &bar_steps,
        equity,
    )?;

    // 2 and 3. The headline: what one interval earns, and whether that survives risk. Both are
    // AUM-indexed with the interval in the SERIES label rather than on the axis, because the
    // question is how each interval's payoff decays with size and because a Sharpe whose
    // interval a reader has to infer from an axis position is a Sharpe that will be misread.
    let mut interval_return = Vec::new();
    let mut interval_sharpe = Vec::new();
    for (index, interval) in BOOK_INTERVALS.iter().enumerate() {
        interval_return.extend(aum_series(
            &format!("mean net return per {}", interval.name),
            &|measured| measured.intervals[index].mean * 1e4,
        ));
        interval_sharpe.extend(aum_series(
            &format!(
                "annualized Sharpe on {} returns ({:.0}-bar interval, x sqrt({:.1}/yr))",
                interval.name,
                interval.bars(&cadence),
                interval.per_year(&cadence)
            ),
            &|measured| measured.intervals[index].sharpe(interval.per_year(&measured.cadence)),
        ));
    }
    interval_return.push(flat("zero return 0.0", 0.0, aums.len()));
    interval_sharpe.push(flat("zero Sharpe 0.0", 0.0, aums.len()));
    interval_sharpe.push(flat("Sharpe 1.0", 1.0, aums.len()));
    chart(
        "timexer_book_interval_return",
        "how much does the book make per bar, hour, session, week and month?",
        BOOK_INTERVAL_UNIT,
        BOOK_AUM_AXIS,
        ScaleKind::Symlog,
        &aum_steps,
        interval_return,
    )?;
    chart(
        "timexer_book_interval_sharpe",
        "does the per-interval payoff survive its own dispersion, and at which interval?",
        BOOK_SHARPE_UNIT,
        BOOK_AUM_AXIS,
        ScaleKind::Linear,
        &aum_steps,
        interval_sharpe,
    )?;

    // 4 and 5. Capacity. The shape of these two panels IS the answer to "how large can this
    // run": the AUM where the return curve turns down and the ratio curve crosses its own
    // reference is where the strategy stops scaling.
    let mut annualized = aum_series("annualized net return", &|m| m.annualized_return);
    annualized.extend(aum_series("annualized vol", &|m| m.annualized_vol));
    annualized.extend(aum_series("annualized downside vol", &|m| {
        m.annualized_downside_vol
    }));
    annualized.extend(aum_series("max drawdown (negative)", &|m| m.max_drawdown));
    annualized.push(flat("flat 0.0", 0.0, aums.len()));
    chart(
        "timexer_book_annualized",
        "at which AUM does the book stop scaling?",
        &cadence.annual_unit(),
        BOOK_AUM_AXIS,
        ScaleKind::Linear,
        &aum_steps,
        annualized,
    )?;
    let mut ratios = aum_series(
        &format!(
            "annualized Sharpe (simulated bar returns, x sqrt({:.0}/yr))",
            cadence.bars_per_year
        ),
        &|m| m.intervals[0].sharpe(m.cadence.bars_per_year),
    );
    ratios.extend(aum_series(
        &format!(
            "annualized Sortino (simulated bar returns, x sqrt({:.0}/yr))",
            cadence.bars_per_year
        ),
        &|m| m.intervals[0].sortino(m.cadence.bars_per_year),
    ));
    ratios.extend(aum_series("Calmar (annualized return / |max drawdown|)", &|m| {
        m.calmar()
    }));
    ratios.push(flat("zero 0.0", 0.0, aums.len()));
    ratios.push(flat("unity 1.0", 1.0, aums.len()));
    chart(
        "timexer_book_annualized_ratio",
        "how much of the book's return is paid for in risk, and does that hold as it scales?",
        BOOK_RATIO_UNIT,
        BOOK_AUM_AXIS,
        ScaleKind::Linear,
        &aum_steps,
        ratios,
    )?;

    // 6, 7 and 8. What execution cost, in the two denominators that mean different things: per
    // traded dollar it is the edge the signal must clear, and per dollar of gross P&L it is how
    // much of the edge was left. Reading the first as the second is how a 9 bps cost on a 40
    // bps gross cohort payoff gets mistaken for a rounding error.
    let mut decomposition = Vec::new();
    let mut shares = Vec::new();
    for (index, &(name, _)) in BOOK_COST_COMPONENTS.iter().enumerate() {
        decomposition.extend(aum_series(name, &|m| m.cost_bps(index)));
        shares.extend(aum_series(name, &|m| m.cost_share(index)));
    }
    decomposition.extend(aum_series("all modeled costs", &|m| m.total_cost_bps()));
    shares.extend(aum_series("all modeled costs", &|m| m.total_cost_share()));
    shares.push(flat("the whole edge 1.0", 1.0, aums.len()));
    chart(
        "timexer_book_cost_decomposition",
        &format!(
            "what did one transacted dollar cost, and in which component? | {}",
            runs[0].2.assumptions.join("; ")
        ),
        BOOK_COST_UNIT,
        BOOK_AUM_AXIS,
        ScaleKind::Linear,
        &aum_steps,
        decomposition,
    )?;
    chart(
        "timexer_book_cost_share",
        "how much of the gross edge did each cost component consume?",
        BOOK_COST_SHARE_UNIT,
        BOOK_AUM_AXIS,
        ScaleKind::Linear,
        &aum_steps,
        shares,
    )?;
    // How much of the spread above was MEASURED. No bid/ask series exists anywhere in the
    // corpus, so every spread is either a Roll estimate from the symbol's own 5-minute
    // autocovariance or a flat assumption where Roll was unusable; a cost panel read without
    // this one cannot distinguish a measured 4 bps spread from an assumed one.
    let mut coverage = aum_series("spread assumed (Roll unusable)", &|m| {
        m.spread_fallback_symbols
    });
    coverage.extend(aum_series("spread Roll-measured", &|m| {
        m.spread_measured_symbols
    }));
    chart(
        "timexer_book_cost_coverage",
        "how much of the charged spread was measured rather than assumed?",
        BOOK_SYMBOL_UNIT,
        BOOK_AUM_AXIS,
        ScaleKind::Linear,
        &aum_steps,
        coverage,
    )?;

    // 9, plus the two census panels the exposure fractions cannot carry. The turnover-cap
    // binding fraction is the reason this family exists: a book throttled by its own cap and a
    // book with a weak signal both draw thin exposure and small returns, and only this series
    // separates them.
    let mut exposure = aum_series("mean gross exposure (x equity)", &|m| m.gross_exposure);
    exposure.extend(aum_series("mean net exposure (x equity)", &|m| m.net_exposure));
    exposure.extend(aum_series("mean |net| exposure (x equity)", &|m| {
        m.absolute_net_exposure
    }));
    exposure.extend(aum_series("mean held names / tradable universe", &|m| {
        m.active_fraction
    }));
    exposure.extend(aum_series(
        "decision frames where the turnover cap bound",
        &|m| m.turnover_capped_fraction,
    ));
    exposure.push(flat("market neutral 0.0", 0.0, aums.len()));
    chart(
        "timexer_book_exposure",
        "how much risk was on, and was the book throttled by its own turnover cap?",
        BOOK_EXPOSURE_UNIT,
        BOOK_AUM_AXIS,
        ScaleKind::Linear,
        &aum_steps,
        exposure,
    )?;
    chart(
        "timexer_book_turnover",
        "how many times over does the book trade its own equity in a year?",
        BOOK_TURNOVER_UNIT,
        BOOK_AUM_AXIS,
        ScaleKind::Linear,
        &aum_steps,
        aum_series("annualized one-way turnover", &|m| m.annualized_turnover),
    )?;
    let mut names = aum_series("mean held names", &|m| m.held_names);
    names.extend(aum_series("mean active (nonzero target) names", &|m| {
        m.active_names
    }));
    names.extend(aum_series("tradable universe", &|m| m.universe_symbols));
    chart(
        "timexer_book_names",
        "how many names does the book actually carry, and how many could it have?",
        BOOK_NAME_UNIT,
        BOOK_AUM_AXIS,
        ScaleKind::Linear,
        &aum_steps,
        names,
    )?;

    // 10. The underwater curve at the reference AUM. Its own panel and not a series on the
    // equity chart: a drawdown lives in [-1, 0] and an equity multiple near 1, so on one axis
    // the deepest hole in the run is a wobble at the bottom of the frame.
    let drawdown = runs
        .iter()
        .filter(|(_, aum, _)| *aum == reference_aum)
        .map(|(label, _, evaluation)| {
            let mut values = book_row(
                evaluation
                    .points
                    .iter()
                    .map(|point| point.values.get("drawdown_fraction").copied().unwrap_or(f64::NAN)),
            );
            values.resize(bars, f32::NAN);
            ReportSeries {
                label: label.clone(),
                values,
            }
        })
        .collect();
    chart(
        "timexer_book_drawdown",
        "how deep and how long was the worst hole the book had to sit in?",
        BOOK_DRAWDOWN_UNIT,
        BOOK_BAR_AXIS,
        ScaleKind::Linear,
        &bar_steps,
        drawdown,
    )?;

    let Some(measured) = diagnostics else {
        return Ok(());
    };
    ensure!(
        !measured.horizons.is_empty(),
        "the book diagnostics pass reported no horizons"
    );
    let horizon_steps: Vec<u64> = measured.horizons.iter().map(|h| u64::from(*h)).collect();
    let width = horizon_steps.len();
    ensure!(
        measured.ic_by_horizon.len() == width,
        "the book IC curve carries {} points for {width} horizons",
        measured.ic_by_horizon.len()
    );
    let ic_se = |index: usize| {
        measured
            .ic_se_by_horizon
            .get(index)
            .copied()
            .unwrap_or(f64::NAN)
    };
    // 11. The decision panel. The SE bands are there because the per-horizon ICs on this
    // corpus are single-digit thousandths against a standard error near 0.003, so a curve
    // read without them invites a horizon ranking that is entirely noise. The aggregated IC
    // is drawn FLAT across the same axis so "did cross-horizon precision weighting beat the
    // best single horizon" is answered by looking at whether the flat line clears the curve's
    // peak, rather than by a reader recomputing it.
    let signal = vec![
        ReportSeries {
            label: "cross-sectional IC".to_owned(),
            values: book_row(measured.ic_by_horizon.iter().copied()),
        },
        ReportSeries {
            label: "cross-sectional IC +1 SE".to_owned(),
            values: book_row(
                measured
                    .ic_by_horizon
                    .iter()
                    .enumerate()
                    .map(|(index, ic)| ic + ic_se(index)),
            ),
        },
        ReportSeries {
            label: "cross-sectional IC -1 SE".to_owned(),
            values: book_row(
                measured
                    .ic_by_horizon
                    .iter()
                    .enumerate()
                    .map(|(index, ic)| ic - ic_se(index)),
            ),
        },
        flat(
            &format!(
                "cross-horizon aggregated IC {:.4} (flat reference)",
                measured.aggregated_ic
            ),
            measured.aggregated_ic,
            width,
        ),
        flat(
            "aggregated IC +1 SE (flat reference)",
            measured.aggregated_ic + measured.aggregated_ic_se,
            width,
        ),
        flat(
            "aggregated IC -1 SE (flat reference)",
            measured.aggregated_ic - measured.aggregated_ic_se,
            width,
        ),
        flat("zero information 0.0", 0.0, width),
    ];
    chart(
        "timexer_book_signal_ic",
        &format!(
            "which horizons carry cross-sectional information, and did aggregating them beat the best single one? | {} cross-sections | trade horizon h={}",
            measured.cross_sections, measured.trade_horizon
        ),
        BOOK_IC_UNIT,
        BOOK_HORIZON_AXIS,
        ScaleKind::Linear,
        &horizon_steps,
        signal,
    )?;
    // 12 and 13. Whether the predictive σ carries SELECTION information. A flat family means
    // it does not - σ ranking is Sharpe-neutral at a fixed horizon, which is the measured
    // result this book is built around - and a fanned family means it does, in which case the
    // decile ordering says which end to trade. The reading is in the title because a reader
    // who inferred alpha from a fanned σ family would be re-deriving a conclusion the run
    // already rejected.
    let deciles = |rows: &[Vec<f64>], family: &str, into: &mut Vec<ReportSeries>| -> Result<()> {
        for (index, row) in rows.iter().enumerate() {
            ensure!(
                row.len() == width,
                "book diagnostics {family} decile {index} carries {} points for {width} horizons",
                row.len()
            );
            into.push(ReportSeries {
                label: format!("{family} decile {index}"),
                values: book_row(row.iter().copied()),
            });
        }
        Ok(())
    };
    let sigma_family = "ascending predicted-σ";
    let agreement_family = "ascending cross-horizon agreement";
    if !measured.ic_by_std_decile.is_empty() || !measured.ic_by_agreement_decile.is_empty() {
        let mut uncertainty = Vec::new();
        deciles(&measured.ic_by_std_decile, sigma_family, &mut uncertainty)?;
        deciles(
            &measured.ic_by_agreement_decile,
            agreement_family,
            &mut uncertainty,
        )?;
        uncertainty.push(flat("zero information 0.0", 0.0, width));
        chart(
            "timexer_book_uncertainty_ic",
            "does the predictive σ select? a FLAT family of deciles means the σ carries no selection information; a FANNED family means it does",
            BOOK_IC_UNIT,
            BOOK_HORIZON_AXIS,
            ScaleKind::Linear,
            &horizon_steps,
            uncertainty,
        )?;
    }
    if !measured.hit_rate_by_std_decile.is_empty()
        || !measured.hit_rate_by_agreement_decile.is_empty()
    {
        let mut hits = Vec::new();
        deciles(&measured.hit_rate_by_std_decile, sigma_family, &mut hits)?;
        deciles(
            &measured.hit_rate_by_agreement_decile,
            agreement_family,
            &mut hits,
        )?;
        hits.push(flat("coin flip 0.5", 0.5, width));
        hits.push(flat(
            &format!(
                "aggregated hit rate {:.4} (flat reference)",
                measured.aggregated_hit_rate
            ),
            measured.aggregated_hit_rate,
            width,
        ));
        chart(
            "timexer_book_uncertainty_hit_rate",
            "same question in sign space: does a σ or agreement decile get the direction right more often?",
            BOOK_HIT_UNIT,
            BOOK_HORIZON_AXIS,
            ScaleKind::Linear,
            &horizon_steps,
            hits,
        )?;
    }
    // 14. Whether the σ the vol target divides by is the σ the market delivered. Coverage
    // below nominal is over-dispersion in the realizations relative to the predictive law,
    // and it is the ONLY panel that explains a book whose realized vol overshot its ex-ante
    // target: the target was computed from a σ that was too small.
    if !measured.coverage_1_sigma.is_empty() || !measured.coverage_2_sigma.is_empty() {
        let mut calibration = Vec::new();
        for (label, coverage) in [
            ("within 1σ", &measured.coverage_1_sigma),
            ("within 2σ", &measured.coverage_2_sigma),
        ] {
            if coverage.is_empty() {
                continue;
            }
            ensure!(
                coverage.len() == width,
                "book diagnostics coverage {label} carries {} points for {width} horizons",
                coverage.len()
            );
            calibration.push(ReportSeries {
                label: format!("realized {label}"),
                values: book_row(coverage.iter().copied()),
            });
        }
        calibration.push(flat("nominal 1σ = 0.6827", 0.6827, width));
        calibration.push(flat("nominal 2σ = 0.9500", 0.95, width));
        chart(
            "timexer_book_calibration",
            "is the predictive σ the vol target divides by the σ the market delivered?",
            BOOK_COVERAGE_UNIT,
            BOOK_HORIZON_AXIS,
            ScaleKind::Linear,
            &horizon_steps,
            calibration,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod book_report_tests {
    use super::*;
    use crate::torch::timexer_segment::book_diagnostics::BookDiagnostics;
    use crate::torch::timexer_segment::portfolio::{AccountEvaluation, AccountPoint};
    use shared::report::{read_report, TIMEXER_SEGMENT_REPORT_BASES};
    use std::fs;

    fn temp(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("timexer_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn period(timestamp_ms: i64, ret: f64, equity: f64) -> AccountPoint {
        AccountPoint {
            timestamp_ms,
            values: BTreeMap::from([
                ("return_fraction".to_owned(), ret),
                ("equity_usd".to_owned(), equity),
            ]),
        }
    }

    /// A run whose per-bar returns are exactly `returns`, carrying the summary keys the panels
    /// read - and deliberately NOT carrying `slippage_usd`, because an absent key has to reach
    /// the chart as a gap and the tests below depend on that being the behavior rather than a
    /// zero default.
    fn run(
        label: &str,
        aum: f64,
        returns: &[f64],
        sessions: usize,
        months: usize,
    ) -> (String, f64, AccountEvaluation) {
        let mut equity = aum;
        let mut points = Vec::new();
        for (index, ret) in returns.iter().enumerate() {
            equity *= 1.0 + ret;
            points.push(AccountPoint {
                timestamp_ms: 1_700_000_000_000 + index as i64 * 300_000,
                values: BTreeMap::from([
                    ("return".to_owned(), *ret),
                    ("return_fraction".to_owned(), *ret),
                    ("equity_usd".to_owned(), equity),
                    ("held_count".to_owned(), 40.0),
                    ("gross_fraction".to_owned(), 1.8),
                    ("net_fraction".to_owned(), -0.02),
                    ("drawdown_fraction".to_owned(), -0.01),
                ]),
            });
        }
        let daily = (0..sessions)
            .map(|index| period(1_700_000_000_000 + index as i64 * 86_400_000, 0.001, aum))
            .collect();
        let monthly = (0..months)
            .map(|index| period(1_700_000_000_000 + index as i64 * 2_592_000_000, 0.02, aum))
            .collect();
        let summary = BTreeMap::from([
            ("initial_cash_usd".to_owned(), aum),
            ("equity_usd".to_owned(), equity),
            ("total_return_fraction".to_owned(), equity / aum - 1.0),
            ("max_drawdown_fraction".to_owned(), -0.05),
            ("bars_simulated".to_owned(), returns.len() as f64),
            ("universe_count".to_owned(), 500.0),
            ("decision_count".to_owned(), 100.0),
            ("turnover_capped_count".to_owned(), 25.0),
            ("turnover_annualized".to_owned(), 180.0),
            ("traded_notional_usd".to_owned(), aum * 12.0),
            ("gross_pnl_usd".to_owned(), aum * 0.02),
            ("pnl_usd".to_owned(), aum * 0.01),
            ("costs_usd".to_owned(), aum * 0.01),
            ("commission_usd".to_owned(), aum * 0.004),
            ("regulatory_usd".to_owned(), aum * 0.001),
            ("spread_usd".to_owned(), aum * 0.004),
            ("impact_usd".to_owned(), aum * 0.001),
            ("borrow_usd".to_owned(), 0.0),
            ("mean_gross_exposure".to_owned(), 1.8),
            ("mean_net_exposure".to_owned(), -0.02),
            ("mean_active_names".to_owned(), 40.0),
            ("spread_fallback_count".to_owned(), 12.0),
            ("spread_measured_count".to_owned(), 488.0),
        ]);
        (
            label.to_owned(),
            aum,
            AccountEvaluation {
                points,
                daily,
                monthly,
                summary,
                assumptions: vec!["synthetic scenario".to_owned()],
            },
        )
    }

    fn diagnostics() -> BookDiagnostics {
        BookDiagnostics {
            horizons: vec![1, 8, 16, 32, 64, 128],
            ic_by_horizon: vec![0.018, 0.0248, 0.037, 0.0473, 0.0547, 0.0561],
            ic_se_by_horizon: vec![0.0027; 6],
            ic_by_std_decile: (0..10).map(|_| vec![0.02; 6]).collect(),
            hit_rate_by_std_decile: (0..10).map(|_| vec![0.51; 6]).collect(),
            ic_by_agreement_decile: (0..10)
                .map(|decile| vec![0.01 * decile as f64; 6])
                .collect(),
            hit_rate_by_agreement_decile: (0..10).map(|_| vec![0.52; 6]).collect(),
            aggregated_ic: 0.0601,
            aggregated_ic_se: 0.0027,
            aggregated_hit_rate: 0.523,
            cross_sections: 4096,
            coverage_1_sigma: vec![0.66; 6],
            coverage_2_sigma: vec![0.93; 6],
            ..Default::default()
        }
    }

    fn lines(report: &Report) -> (&Vec<u64>, &Vec<ReportSeries>) {
        let ReportKind::IndexedLines { steps, series } = &report.kind else {
            panic!("every book panel is an indexed line chart");
        };
        (steps, series)
    }

    fn line<'a>(series: &'a [ReportSeries], needle: &str) -> &'a ReportSeries {
        series
            .iter()
            .find(|line| line.label.contains(needle))
            .unwrap_or_else(|| panic!("no book series mentions {needle:?}"))
    }

    /// The registration contract, the AUM ordering and the gap-versus-zero rule in one pass
    /// over a full sweep: two AUM levels of one variant, handed over in DESCENDING order,
    /// with one cost bucket left unmeasured.
    #[test]
    fn the_book_panels_are_registered_and_ordered_by_ascending_aum() {
        let root = temp("book_report_sweep");
        let returns: Vec<f64> = (0..200)
            .map(|index| if index % 2 == 0 { 0.0004 } else { -0.0002 })
            .collect();
        let runs = vec![
            run("precision decile", 1_000_000.0, &returns, 12, 3),
            run("precision decile", 25_000.0, &returns, 12, 3),
        ];
        write_book(&root, 3, 4000, &runs, Some(&diagnostics())).unwrap();
        for base in BOOK_REPORT_BASES {
            assert!(
                TIMEXER_SEGMENT_REPORT_BASES.contains(base),
                "{base} is written but unregistered, so the TUI never scans for it"
            );
            assert!(
                root.join(format!("{base}.report.bin")).exists(),
                "{base} is registered but was not written"
            );
        }
        let annualized = read_report(root.join("timexer_book_annualized.report.bin")).unwrap();
        let (steps, series) = lines(&annualized);
        assert_eq!(
            steps,
            &vec![25_000u64, 1_000_000],
            "the capacity axis must be ascending whatever order the sweep was simulated in"
        );
        // One variant seen twice is one legend entry, not two.
        assert_eq!(
            series
                .iter()
                .filter(|line| line.label.contains("annualized net return"))
                .count(),
            1
        );
        // An unmeasured cost bucket is a gap; the measured ones are exact basis points of the
        // traded notional the account reported.
        let costs =
            read_report(root.join("timexer_book_cost_decomposition.report.bin")).unwrap();
        let (_, costs) = lines(&costs);
        assert!(
            line(costs, "flat slippage proxy")
                .values
                .iter()
                .all(|value| value.is_nan()),
            "an unmeasured cost bucket must chart as a gap, never as free execution"
        );
        assert!(
            (line(costs, "commission").values[0] - 0.004 / 12.0 * 1e4).abs() < 1e-3,
            "{:?}",
            line(costs, "commission").values
        );
        // Equity is normalized, so the $25k and $1M paths are comparable rather than one
        // curve and four flat lines.
        let equity = read_report(root.join("timexer_book_equity.report.bin")).unwrap();
        let (bars, equity) = lines(&equity);
        assert_eq!(bars.len(), returns.len());
        assert!(equity
            .iter()
            .all(|line| (line.values[0] - 1.0).abs() < 0.01));
        // The time-indexed panels are drawn at the lower median of the sweep, so exactly one
        // of the two runs appears on the underwater curve.
        let drawdown = read_report(root.join("timexer_book_drawdown.report.bin")).unwrap();
        assert_eq!(lines(&drawdown).1.len(), 1);
        assert!(drawdown.title.contains("$25k"), "{}", drawdown.title);
        // The aggregation verdict is readable off the IC panel: a flat aggregated line above
        // the per-horizon curve's peak is what "aggregating beat the best single horizon"
        // looks like, and it is drawn rather than left to the reader.
        let signal = read_report(root.join("timexer_book_signal_ic.report.bin")).unwrap();
        let (horizons, signal) = lines(&signal);
        assert_eq!(horizons, &vec![1u64, 8, 16, 32, 64, 128]);
        let aggregated = line(signal, "aggregated IC 0.0601");
        let curve = line(signal, "cross-sectional IC");
        let peak = curve.values.iter().copied().fold(f32::MIN, f32::max);
        assert!(aggregated.values.iter().all(|value| *value > peak));
        assert!(line(signal, "zero information 0.0")
            .values
            .iter()
            .all(|value| *value == 0.0));
        // Coverage below nominal is the only available explanation for a vol target that
        // undershoots, so both nominal levels are on the calibration panel by name.
        let calibration = read_report(root.join("timexer_book_calibration.report.bin")).unwrap();
        let (_, calibration) = lines(&calibration);
        assert!((line(calibration, "nominal 1σ").values[0] - 0.6827).abs() < 1e-6);
        assert!((line(calibration, "nominal 2σ").values[0] - 0.95).abs() < 1e-6);
        fs::remove_dir_all(&root).unwrap();
    }

    /// The diagnostics pass is optional. Without it the account panels still land and the four
    /// signal panels are ABSENT rather than written as reference lines with no measurement.
    #[test]
    fn a_book_without_diagnostics_writes_the_account_panels_and_no_signal_panels() {
        let root = temp("book_report_no_diagnostics");
        let returns: Vec<f64> = (0..120)
            .map(|index| if index % 3 == 0 { -0.0003 } else { 0.0002 })
            .collect();
        let runs = vec![run("precision decile", 1_000_000.0, &returns, 12, 3)];
        write_book(&root, 1, 1, &runs, None).unwrap();
        for base in BOOK_REPORT_BASES {
            assert_eq!(
                root.join(format!("{base}.report.bin")).exists(),
                !BOOK_DIAGNOSTIC_BASES.contains(base),
                "{base} was written without the pass that measures it, or dropped with it"
            );
        }
        fs::remove_dir_all(&root).unwrap();
    }

    /// An interval with fewer than two complete observations is unmeasured, and unmeasured is
    /// NaN. A 20-bar run completes one 12-bar hour, one session and no week or month; charting
    /// any of those as 0.0 would claim the book earns nothing per month on evidence that never
    /// contained a month.
    #[test]
    fn an_interval_with_too_few_observations_is_a_gap_not_a_zero() {
        let root = temp("book_report_short");
        let returns: Vec<f64> = (0..20)
            .map(|index| if index % 2 == 0 { 0.0005 } else { -0.0001 })
            .collect();
        let runs = vec![run("precision decile", 1_000_000.0, &returns, 1, 0)];
        write_book(&root, 1, 1, &runs, None).unwrap();
        let report = read_report(root.join("timexer_book_interval_return.report.bin")).unwrap();
        let (_, series) = lines(&report);
        assert!(line(series, "per five-minute bar").values[0].is_finite());
        for unmeasured in ["per 12-bar hour", "per session", "per 5-session week", "per calendar month"] {
            let values = &line(series, unmeasured).values;
            assert!(
                values.iter().all(|value| value.is_nan()),
                "{unmeasured} charted {values:?} on fewer than two complete intervals"
            );
        }
        fs::remove_dir_all(&root).unwrap();
    }

    /// A zero-variance return series has no Sharpe. The failure this pins is an infinity: a
    /// constant book would otherwise chart as the best risk-adjusted run in the sweep, and the
    /// report writer would reject the value instead of drawing a gap.
    #[test]
    fn a_constant_return_series_has_no_sharpe_rather_than_an_infinite_one() {
        let root = temp("book_report_constant");
        let runs = vec![run("precision decile", 1_000_000.0, &[0.0003; 64], 12, 3)];
        write_book(&root, 1, 1, &runs, None).unwrap();
        let report = read_report(root.join("timexer_book_annualized_ratio.report.bin")).unwrap();
        let (_, series) = lines(&report);
        let sharpe = line(series, "annualized Sharpe");
        assert!(
            sharpe.values.iter().all(|value| value.is_nan()),
            "{:?}",
            sharpe.values
        );
        assert!(sharpe.values.iter().all(|value| !value.is_infinite()));
        // The vol itself IS measured, and measured zero: the book really did not move.
        let annualized = read_report(root.join("timexer_book_annualized.report.bin")).unwrap();
        let (_, annualized) = lines(&annualized);
        assert_eq!(line(annualized, "annualized vol").values[0], 0.0);
        assert!(line(annualized, "annualized net return").values[0] > 0.0);
        fs::remove_dir_all(&root).unwrap();
    }

    const DENSE_START_MS: i64 = 1_700_000_000_000;
    const DENSE_SESSION_MS: i64 = 86_400_000;

    /// A self-consistent synthetic account on the cadence the real one simulates: `sessions`
    /// sessions of `bars` marks each, `step_ms` apart, with the daily and monthly series
    /// compounded from those same bars. Extended hours included, so a session carries ~191
    /// marks rather than a regular session's 78 - which is the whole point: this is the
    /// density the account really emits, and the reports have to annualize on it.
    ///
    /// Complete 21-session months only; a trailing stub is dropped exactly as the account's
    /// own calendar aggregation drops it.
    fn dense(
        aum: f64,
        sessions: usize,
        bars: usize,
        step_ms: i64,
        ret: impl Fn(usize) -> f64,
    ) -> (String, f64, AccountEvaluation) {
        let mut points = Vec::with_capacity(sessions * bars);
        let mut session_returns = Vec::with_capacity(sessions);
        let mut equity = aum;
        for session in 0..sessions {
            let open = equity;
            for bar in 0..bars {
                let realized = ret(session * bars + bar);
                equity *= 1.0 + realized;
                points.push(AccountPoint {
                    timestamp_ms: DENSE_START_MS
                        + session as i64 * DENSE_SESSION_MS
                        + bar as i64 * step_ms,
                    values: BTreeMap::from([
                        ("return".to_owned(), realized),
                        ("return_fraction".to_owned(), realized),
                        ("equity_usd".to_owned(), equity),
                        ("held_count".to_owned(), 40.0),
                    ]),
                });
            }
            session_returns.push(equity / open - 1.0);
        }
        let daily = session_returns
            .iter()
            .enumerate()
            .map(|(index, realized)| {
                period(
                    DENSE_START_MS + index as i64 * DENSE_SESSION_MS,
                    *realized,
                    aum,
                )
            })
            .collect();
        let monthly = session_returns
            .chunks_exact(21)
            .enumerate()
            .map(|(index, month)| {
                period(
                    DENSE_START_MS + index as i64 * 21 * DENSE_SESSION_MS,
                    month.iter().map(|realized| 1.0 + realized).product::<f64>() - 1.0,
                    aum,
                )
            })
            .collect();
        let summary = BTreeMap::from([
            ("initial_cash_usd".to_owned(), aum),
            ("equity_usd".to_owned(), equity),
            ("total_return_fraction".to_owned(), equity / aum - 1.0),
            ("net_pnl_usd".to_owned(), equity - aum),
            ("max_drawdown_fraction".to_owned(), -0.05),
            ("bars_simulated".to_owned(), points.len() as f64),
            ("sessions_simulated".to_owned(), sessions as f64),
            ("universe_count".to_owned(), 256.0),
            ("decision_count".to_owned(), sessions as f64),
            ("turnover_capped_count".to_owned(), 0.0),
            ("turnover_annualized".to_owned(), 180.0),
            ("traded_notional_usd".to_owned(), aum * 12.0),
            ("gross_pnl_usd".to_owned(), aum * 0.08),
            ("costs_usd".to_owned(), aum * 0.03),
            ("mean_active_names".to_owned(), 40.0),
        ]);
        (
            "dense".to_owned(),
            aum,
            AccountEvaluation {
                points,
                daily,
                monthly,
                summary,
                assumptions: vec!["synthetic dense cadence".to_owned()],
            },
        )
    }

    /// Total net return the charted mean for one interval implies: the mean, in basis points,
    /// times the number of complete intervals the window held.
    fn implied(series: &[ReportSeries], interval: &str, count: usize) -> f64 {
        f64::from(line(series, interval).values[0]) / 1e4 * count as f64
    }

    /// THE invariant these panels exist for, and the defect it pins. Every interval series on
    /// the return panel, times the number of intervals the window held, must recover the run's
    /// own total net return, and the annualized figure must be that total annualized over the
    /// window's TRUE wall-clock length. The account marks every observed bar, extended hours
    /// included, so a year is roughly 48 000 of them; annualizing the same series with a
    /// 19 656-bar regular-session year understated this book's year by a factor of 3.6.
    #[test]
    fn every_book_interval_reconciles_with_the_total_and_the_true_window_length() {
        let root = temp("book_report_reconciliation");
        let (sessions, bars) = (255usize, 191usize);
        // A small positive drift under a much larger zero-mean wobble, so the dispersion the
        // Sharpe divides by is real and the reconciliation is not a test of a constant.
        let runs = vec![dense(1_000_000.0, sessions, bars, 300_000, |index| {
            9.7e-7 + if index % 2 == 0 { 8e-5 } else { -8e-5 }
        })];
        let total = runs[0].2.summary["total_return_fraction"];
        // The window's length, from the construction rather than from the code under test.
        let years = ((sessions - 1) as f64 * DENSE_SESSION_MS as f64
            + (bars - 1) as f64 * 300_000.0)
            / (365.25 * 86_400_000.0);
        write_book(&root, 1, 1, &runs, None).unwrap();
        let report = read_report(root.join("timexer_book_interval_return.report.bin")).unwrap();
        let (_, series) = lines(&report);
        let session_return = f64::from(line(series, "per session").values[0]) / 1e4;
        for (interval, count) in [
            ("per five-minute bar", sessions * bars),
            ("per 12-bar hour", sessions * bars / 12),
            ("per session", sessions),
            ("per 5-session week", sessions / 5),
        ] {
            let recovered = implied(series, interval, count);
            assert!(
                (recovered - total).abs() < 0.005,
                "{interval} implies {recovered:.6} over {count} intervals against a total of \
                 {total:.6}"
            );
        }
        // The calendar month is the documented exception: its trailing partial period is
        // dropped, so it recovers the total MINUS the dropped sessions - and the residual has
        // to stay well inside one month's return, which is what distinguishes "the stub was
        // dropped" from "a whole month went missing".
        let dropped = sessions % 21;
        let months = implied(series, "per calendar month", sessions / 21);
        let stub = total - dropped as f64 * session_return;
        assert!(
            months < total && (months - stub).abs() < 21.0 * session_return / 2.0,
            "the month series implies {months:.6}; dropping {dropped} trailing sessions off a \
             total of {total:.6} accounts for {stub:.6}"
        );
        // And the headline: the total annualized over the window's own length, which is also
        // the figure the stdout table prints.
        let annualized = read_report(root.join("timexer_book_annualized.report.bin")).unwrap();
        let (_, annualized) = lines(&annualized);
        let charted = f64::from(line(annualized, "annualized net return").values[0]);
        let expected = (1.0 + total).powf(1.0 / years) - 1.0;
        assert!(
            (charted - expected).abs() < 5e-4,
            "the panel annualizes {total:.6} over {years:.4} years as {charted:.6}, not \
             {expected:.6}"
        );
        fs::remove_dir_all(&root).unwrap();
    }

    /// The density of `points` is the denominator of every annualization on these panels, so
    /// the same year at twice the mark cadence must annualize to the SAME year rather than to
    /// half of it - and a bar census that contradicts the series it describes is refused rather
    /// than charted at a density nothing measured.
    #[test]
    fn a_denser_points_series_annualizes_to_the_same_year_and_a_false_census_is_refused() {
        let root = temp("book_report_density");
        let coarse = dense(1_000_000.0, 60, 78, 300_000, |_| 4e-6);
        // The identical path resampled: two half-size bars wherever the coarse run had one, on
        // the same wall clock and to the same terminal equity.
        let fine = dense(1_000_000.0, 60, 156, 150_000, |_| 2e-6 - 1e-12);
        let annualized_of = |run: &(String, f64, AccountEvaluation), name: &str| -> f64 {
            let root = temp(name);
            write_book(&root, 1, 1, std::slice::from_ref(run), None).unwrap();
            let report = read_report(root.join("timexer_book_annualized.report.bin")).unwrap();
            let value = f64::from(line(lines(&report).1, "annualized net return").values[0]);
            fs::remove_dir_all(&root).unwrap();
            value
        };
        let (slow, quick) = (
            annualized_of(&coarse, "book_report_density_coarse"),
            annualized_of(&fine, "book_report_density_fine"),
        );
        assert!(
            (slow - quick).abs() < 5e-4 && slow > 0.01,
            "the same year annualizes to {slow:.6} at 78 marks a session and {quick:.6} at 156"
        );
        // A census that disagrees with the series is a fault, not a chart.
        let mut lying = coarse;
        lying
            .2
            .summary
            .insert("bars_simulated".to_owned(), 19_500.0);
        assert!(write_book(&root, 1, 1, &[lying], None).is_err());
        fs::remove_dir_all(&root).unwrap();
    }

    /// A diagnostics record whose curves disagree with its own horizon axis is a pairing
    /// fault, not a chart to draw with one axis silently truncated.
    #[test]
    fn book_diagnostics_that_disagree_with_their_horizon_axis_are_refused() {
        let root = temp("book_report_mismatch");
        let runs = vec![run("precision decile", 1_000_000.0, &[0.0002, -0.0001, 0.0003], 2, 0)];
        let mut broken = diagnostics();
        broken.ic_by_horizon.pop();
        assert!(write_book(&root, 1, 1, &runs, Some(&broken)).is_err());
        let mut ragged = diagnostics();
        ragged.ic_by_std_decile[3].pop();
        assert!(write_book(&root, 1, 1, &runs, Some(&ragged)).is_err());
        assert!(write_book(&root, 1, 1, &[], None).is_err());
        fs::remove_dir_all(&root).unwrap();
    }
}
