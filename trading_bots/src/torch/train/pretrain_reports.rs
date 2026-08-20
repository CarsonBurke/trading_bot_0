//! Report emission for discrete distributional bar pretraining.
//!
//! Every metric the pretrainer produces leaves the process through this module
//! and through `shared::report::write_report` only; there is no second channel.
//! The base names written here are mirrored in `meta_chart_bases` in
//! `tui/src/main.rs`, which is what makes them visible.
//!
//! All scalar curves share one x-axis, the *record tick*. [`PretrainReporter::record_step`]
//! mean-aggregates [`STEP_DECIMATION`] optimizer steps into one tick, and
//! [`PretrainReporter::record_epoch`] commits a tick of its own carrying the
//! validation numbers. Series that only exist on one of the two paths are
//! NaN-padded on the other, which the report renderer already filters. The
//! practical effect is a dense training curve with validation markers
//! interleaved, instead of a chart with one point per epoch.
//!
//! Two validation contexts are reported separately and must not be conflated.
//! `pretrain_nll_bar` carries the *promotion* metric, measured at the full
//! context and therefore only defined once the context ramp has finished.
//! `pretrain_nll_bar_diag896` and every per-DOF, calibration and diagnostic
//! series carry the *fixed 896-context* evaluation, which is configured
//! identically in every run and is the only curve comparable across ablations.

use std::array;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use ring::digest::{Context as DigestContext, SHA256};
use shared::report::{
    write_report, CandleBar, QuantileBand, Report, ReportKind, ReportSeries, ScaleKind,
};
use tch::{Device, Kind, Tensor};

use super::pretrain::{
    HeldOutPower, SelectionLedger, SelectionOutcome, EVAL_WINDOW_SEED, SELECTION_CAP,
    SELECTION_CAP_SLOT,
};
use super::pretrain_stats::Dispersion;
use super::trade_bench::{
    BandShrinkOverlap, BandSweep, EdgeAttribution, HysteresisComposition, HysteresisSweep,
    MeanCalibration,
    HysteresisOos, OuterDecomposition, PolicyStats, ShrunkBench, SignalDecay, TradeBench, ATTRIBUTION_ARMS,
    ATTRIBUTION_DECILES, COMPOSITION_NAMES, DECAY_HORIZONS, HYSTERESIS_MARGINS,
    HYSTERESIS_NET_COSTS, HYSTERESIS_SELECTION_COST,
    ATTRIBUTION_NAMES, SIZING_KNOBS, SIZING_SHAPES, BARS_PER_YEAR, CAP_GRID, CELL_LABELS,
    COST_GRID_BPS, DEFAULT_COST_SLOT, FREE_KELLY_EDGES, LEVERAGE_CAP, MAX_BREAK_EVEN_BPS,
    MAX_LEVERAGE, PANEL_LABELS, POLICY_COUNT, POLICY_KELLY_MULTIPLE, POLICY_MARGINAL,
    POLICY_MODEL, POLICY_NAMES, POLICY_ORACLE, TAIL_LEVELS, TAIL_RATIO_WARN,
};
use crate::torch::bar_dist::{
    decode_dof, BarDof, BarScoring, BarSupports, BAR_CHAIN, BAR_DOF, BAR_DOF_NAMES, DOF_R, DOF_U,
    DOF_V, NUM_BAR_BINS,
};
use super::support_moments::SupportDecode;
use super::mem_probe::{Arm, GapPoint, PairedContrast, RecencyBucket, StabilityPoint};
use super::bar_family::{BarFamilyFit, DENSITY_BASES};
use super::split_seams::{
    deviation_edges, range_ratio_edges, ranked_ratios, volume_edges, SeamAudit, CENSUS_LOG_LEVELS,
    TIER_EXACT, TIER_NEAR,
};
use crate::torch::dataset::{mix64, Split, MULTIPLICITY_BUCKETS};

/// Resolution of the per-DOF PIT histogram.
pub const PIT_HIST_BINS: usize = 16;
/// Rollout horizons, in bars, reported by `pretrain_rollout_nll`.
///
/// The last entry is the depth of the realized continuation the snapshot windows
/// hold; `pretrain::SNAPSHOT_HORIZON` asserts the two agree, because a horizon
/// the continuation cannot reach is silently skipped rather than reported.
pub const ROLLOUT_HORIZONS: [usize; 5] = [1, 4, 16, 64, 100];
/// Context the comparable diagnostic evaluation is pinned to, for axis labels.
pub const DIAGNOSTIC_CONTEXT: i64 = 896;
/// Causes a training bar cannot be a prediction target in a pass, in the order
/// [`EpochMetrics::pass_remainder_bars`] carries them. Named rather than lumped into "not
/// covered": three of the four are structural properties of the corpus that no schedule can
/// remove, and the fourth is a schedule bug, so a reader has to be able to tell them apart.
pub const PASS_REMAINDER_CAUSES: [&str; 4] = [
    "head (no predecessor close / anchor is input-only)",
    "symbol shorter than the shortest ramp context",
    "sub-context hole (below one window, per symbol)",
    "assigned but never issued (schedule too short)",
];
/// Recommended ancestral sample count per snapshot window.
///
/// This count is what a snapshot's quantiles are ESTIMATED from, and the estimate
/// is not free: the standard error of a sample quantile at probability `q` is
/// `sqrt(q (1 - q) / n) / f(x_q)`, so the median of 256 draws from a law of scale
/// `sigma` carries `1.2533 * sigma / 16 = 0.078 * sigma` of noise. At the measured
/// per-step `sigma` of 2-4e-3 that is 2-3e-4 per window — two orders above the
/// `-sigma^2/2` median drift of a martingale — which is why every snapshot scalar
/// is charted beside its own standard error and never alone.
pub const SNAPSHOT_SAMPLES: usize = 256;
/// Quantile fan a snapshot window depicts, ASCENDING in probability.
pub const FAN_QUANTILES: [f64; 5] = [0.10, 0.25, 0.50, 0.75, 0.90];
/// Index of the nominal band's lower and upper edge within [`FAN_QUANTILES`].
const BAND_LOW_INDEX: usize = 0;
const BAND_HIGH_INDEX: usize = FAN_QUANTILES.len() - 1;
/// Index of the fan centre within [`FAN_QUANTILES`].
const FAN_CENTRE_INDEX: usize = 2;
const _: () = assert!(FAN_QUANTILES[FAN_CENTRE_INDEX] == 0.50);
/// [`CandleWindow::centre_log_se`] reads the median's standard error off the
/// INTERQUARTILE spacing, and the `1/f(m) ~ IQR/0.5` step it uses is only valid for
/// exactly p25 and p75. Relevelling the fan without moving that estimator would
/// silently redefine the error bar every snapshot number is checked against.
const _: () = assert!(FAN_QUANTILES[1] == 0.25 && FAN_QUANTILES[3] == 0.75);
/// The context this module labels its snapshot axes with is the context the pretrainer
/// actually evaluates at. Two independent 896s would let the pictures claim a geometry
/// the run does not have.
const _: () = assert!(DIAGNOSTIC_CONTEXT == super::pretrain::BAR_CONTEXT_RAMP_START);
/// Fewest ancestral draws a fan may be estimated from.
///
/// Below four, p25 and p75 are not distinct order statistics, the interquartile spacing
/// is exactly zero and every error bar the picture prints reads `0.0` — a claim of
/// infinite precision, which is strictly worse than the NaN a zero-sample fan reports.
/// `--samples` is documented as something to lower on a shared card, so this is
/// reachable from the command line and belongs at the one place both entry points pass
/// through.
pub const MIN_FAN_SAMPLES: usize = 4;
/// Genuine ancestral draws overlaid on each snapshot window.
///
/// Draws, not the fan centre: an individual path is a sample from the predictive
/// law and shows what the model believes CAN happen, while the fan centre is a
/// locus of per-horizon medians that no draw follows.
pub const SNAPSHOT_OVERLAY_PATHS: usize = 5;
/// Optimizer steps folded into one record tick.
const STEP_DECIMATION: usize = 20;
/// Record ticks between report flushes on the step path. Epoch and snapshot
/// records always flush.
const FLUSH_EVERY_TICKS: usize = 5;
const BAND_LOW: f64 = FAN_QUANTILES[BAND_LOW_INDEX];
const BAND_HIGH: f64 = FAN_QUANTILES[BAND_HIGH_INDEX];
/// Share of the objective's total magnitude at which an AUXILIARY term is considered to be
/// competing with the likelihood rather than shaping the latent.
///
/// Drawn on `pretrain_loss_shares` and enforced by the pretrainer's consecutive-step
/// warning. Not a hard cap: the right response to a term crossing it is a decision about
/// `--lambda-dyn`, not a silent clamp that would hide the miscalibration.
pub const AUX_SHARE_WARN: f64 = 0.25;
/// Consecutive steps an auxiliary term may sit above [`AUX_SHARE_WARN`] before the
/// pretrainer warns. One step is minibatch noise; a hundred is the objective.
pub const AUX_SHARE_WARN_STREAK: usize = 100;
/// Fraction of realized closes a calibrated 10/90 band should contain.
const NOMINAL_COVERAGE: f64 = BAND_HIGH - BAND_LOW;

/// Nats per bar of the uniform CATEGORICAL chain, `BAR_DOF * ln(NUM_BAR_BINS)`.
///
/// This is the discrete part only. Under `BarScoring::Density` a uniform head also pays the
/// measure term, so the mode-aware line a chart must use is
/// [`HeldOutBaselines::uniform_nll_bar`].
pub fn uniform_categorical_nll_bar() -> f64 {
    BarSupports::uniform_categorical_nll_bar()
}

// ---------------------------------------------------------------------------
// Caller-facing metric structs
// ---------------------------------------------------------------------------

/// Per-optimizer-step training metrics. Build with [`StepMetrics::nan`] and set
/// what is available; non-finite fields are skipped rather than plotted.
#[derive(Clone, Copy, Debug)]
pub struct StepMetrics {
    pub epoch: usize,
    pub step: usize,
    pub nll_bar: f64,
    pub nll_dof: [f64; BAR_DOF],
    pub dyn_loss: f64,
    pub kl_loss: f64,
    pub total_loss: f64,
    /// Mean `-log(1 + f_hat R)` in nats per bar under the deployed leverage cap, at the
    /// log-optimal fraction of `p(r|past)` with the same-bar `s` marginalized out. Recorded
    /// whatever `--lambda-growth` is, so a `lambda_growth = 0` control arm charts the same
    /// curve and the ablation is a comparison rather than one panel and one blank.
    pub growth_loss: f64,
    /// Share of the objective's total MAGNITUDE carried by each term, i.e. the weighted
    /// term over the sum of the four weighted magnitudes. They sum to one.
    ///
    /// Magnitudes and not the signed total: under `BarScoring::Density` the likelihood term
    /// is a log density and is routinely NEGATIVE, so a signed denominator would pass
    /// through zero and make every share meaningless exactly when the objective is most
    /// worth watching.
    pub nll_share: f64,
    pub dyn_share: f64,
    pub kl_share: f64,
    pub growth_share: f64,
    /// Mean `|f_hat|` the growth term sized at, under the deployed hard clamp. Comparable
    /// to the trade bench's `quarter-Kelly mean |f|` and `|f*| median` figures, which is
    /// what makes the training-time and evaluation-time views of the same decision one
    /// picture.
    pub growth_abs_f: f64,
    /// Fraction of bars where the LEVERAGE CAP chose the position size rather than the
    /// predictive law, i.e. `|mu_hat / var_hat| > cap`. 0.78-0.86 on the run that motivated
    /// the term, and the reason the growth term's backward map is a smooth saturation
    /// rather than the clamp itself.
    pub growth_clamp_bind: f64,
    /// Mean `cos(h_t, h_{t+1})` over the batch. A trunk that wins on the NextLat term by
    /// making beliefs SLOWLY VARYING — which the zero-init identity dynamics predicts
    /// perfectly — drives this to one while destroying the trajectory's temporal
    /// resolution. Diagnostic only; nothing optimizes it.
    pub belief_autocorr: f64,
    /// `dyn` divided by the TRIVIAL-IDENTITY baseline `smooth_l1(h_t, sg[h_{t+k}])`, i.e.
    /// what the term would score if the dynamics MLP returned its input unchanged. At one
    /// the MLP contributes nothing and `dyn` is measuring belief smoothness alone.
    pub dyn_vs_identity: f64,
    pub lr_mult: f64,
    pub muon_momentum: f64,
    /// Observed gradient norm. Nothing clips on it; Muon orthogonalization does
    /// that job, so this is a pure diagnostic.
    pub grad_norm: f64,
    pub context: i64,
    pub batch_size: usize,
    pub bars_seen: u64,
    /// Device-wide FREE VRAM at this step, in GiB. NaN off CUDA or without NVML.
    ///
    /// Device-wide and not process-local, deliberately: the card is shared, so what the next
    /// ramp step-up has to fit into is what the other tenants leave. A dip in this curve
    /// beside a flat `projected footprint` is a contention event, and is exactly what makes a
    /// runtime batch hold legible after the fact instead of arriving as an OOM.
    pub free_vram_gib: f64,
    /// Bar-tokens this step consumed, `batch_size * context`. The quantity the card actually
    /// caps: at a measured cost per bar-token, batch and context trade off directly.
    pub bar_tokens: f64,
    /// Device bytes this step was PROJECTED to cost by the startup capacity model, in GiB.
    /// NaN when capacity was never measured.
    pub projected_footprint_gib: f64,
    /// Free VRAM at the startup probe minus the shared-card reserve, in GiB: the ceiling the
    /// ramp was derived against, charted as a flat reference line so the distance between the
    /// plan and the wall is visible for the whole run.
    pub capacity_ceiling_gib: f64,
    /// Bars in this step's batch whose market-proxy channels are the reserved MISSING row, and
    /// the batch's total bar count.
    ///
    /// Charted because the market channel is an INPUT GROUP whose absence is invisible in every
    /// loss: a proxy with poor extended-hours coverage, a corpus directory with no proxy file at
    /// all, or a resolution whose instants never line up would each leave three conditioning
    /// channels pinned to one row while `nll_bar` looked entirely normal. The share is the only
    /// number that distinguishes "the market channel did not help" from "the market channel was
    /// never there".
    pub market_missing_bars: u64,
    pub market_total_bars: u64,
}

impl StepMetrics {
    pub fn nan() -> Self {
        Self {
            epoch: 0,
            step: 0,
            nll_bar: f64::NAN,
            nll_dof: [f64::NAN; BAR_DOF],
            dyn_loss: f64::NAN,
            kl_loss: f64::NAN,
            total_loss: f64::NAN,
            nll_share: f64::NAN,
            dyn_share: f64::NAN,
            kl_share: f64::NAN,
            growth_loss: f64::NAN,
            growth_share: f64::NAN,
            growth_abs_f: f64::NAN,
            growth_clamp_bind: f64::NAN,
            belief_autocorr: f64::NAN,
            dyn_vs_identity: f64::NAN,
            lr_mult: f64::NAN,
            muon_momentum: f64::NAN,
            grad_norm: f64::NAN,
            context: 0,
            batch_size: 0,
            bars_seen: 0,
            free_vram_gib: f64::NAN,
            bar_tokens: f64::NAN,
            projected_footprint_gib: f64::NAN,
            capacity_ceiling_gib: f64::NAN,
            market_missing_bars: 0,
            market_total_bars: 0,
        }
    }
}

/// Per-validation metrics. Build with [`EpochMetrics::nan`] and set what is
/// available.
///
/// Everything except `val_nll_bar` / `best_val_nll_bar` comes from the fixed
/// [`DIAGNOSTIC_CONTEXT`] evaluation, so it stays comparable across runs.
#[derive(Clone, Debug)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub global_step: usize,
    pub train_nll_bar: f64,
    /// Promotion metric at the full context; NaN before the ramp completes.
    pub val_nll_bar: f64,
    pub best_val_nll_bar: f64,
    /// Path the promoted checkpoint was written to, or `None` if this validation
    /// did not promote. The reporter fingerprints the artifact here so the
    /// end-of-run test battery can prove it scored that exact file.
    pub promoted_checkpoint: Option<PathBuf>,
    /// Across-run diagnostic at the fixed [`DIAGNOSTIC_CONTEXT`].
    pub val_nll_bar_diag: f64,
    pub train_nll_dof: [f64; BAR_DOF],
    pub val_nll_dof: [f64; BAR_DOF],
    pub val_crps_dof: [f64; BAR_DOF],
    pub val_pit: PitHistogram,
    /// Sign accuracy of `E[r]` against the realized return, validation only.
    pub val_dir_acc: f64,
    /// Teacher-forced rollout NLL at [`ROLLOUT_HORIZONS`] with beliefs advanced
    /// by the trunk. NaN for horizons the evaluation window cannot reach.
    pub rollout_nll_exact: [f64; ROLLOUT_HORIZONS.len()],
    /// The same horizons with beliefs advanced by `BarDynamics`. The gap against
    /// the exact series is what the dynamics KL term is there to close.
    pub rollout_nll_dynamics: [f64; ROLLOUT_HORIZONS.len()],
    /// Bars consumed divided by the corpus unique-bar count.
    pub unique_bar_reuse: f64,
    /// Participation ratio of the belief covariance; see [`belief_effective_rank`].
    pub effective_rank: f64,
    /// Block-bootstrap standard error of `val_nll_bar`, resampling by
    /// `(symbol, calendar month)`. NaN when the promotion set was not evaluated.
    pub val_nll_bar_se: f64,
    /// 95% block-bootstrap interval of `val_nll_bar`, charted as a band.
    pub val_nll_bar_ci: (f64, f64),
    /// The same standard error resampling CALENDAR MONTHS, i.e. treating the shared
    /// wall-clock slots as the resampling unit. This is the honest error bar on the
    /// absolute level and runs ~4x the `(symbol, month)` figure, because all 4096 windows
    /// sit in a handful of shared months and a regime shift moves them together.
    pub val_nll_bar_se_level: f64,
    /// `val_nll_bar_diag` with the encoding tautology excluded: `u` and `v` are scored only
    /// on bars with `s != 0`, where the encoding does not already determine them.
    ///
    /// From the DIAGNOSTIC pass, so it is defined at every ramp stage from step 0. The
    /// deployed-context twin is `val_nll_bar_conditional_deployed`, which is the number
    /// selection actually compares and is only defined once the ramp gets there.
    pub val_nll_bar_conditional: f64,
    pub val_nll_dof_conditional: [f64; BAR_DOF],
    /// The selection metric itself, at the deployed context. Unmeasured before the ramp
    /// reaches it.
    pub val_nll_bar_conditional_deployed: f64,
    /// Per-DOF MARGINALIZED forecast NLL: every factor conditioned on strictly PAST bars
    /// only, with the same-bar chain prefix integrated over the head's own predictive law.
    /// This is the honest forecasting number.
    pub val_forecast_nll_dof: [f64; BAR_DOF],
    /// Teacher-forced per-DOF NLL on EXACTLY the rows the forecast figure used, so the
    /// teacher-forcing inflation is a paired difference.
    pub val_forecast_teacher_nll_dof: [f64; BAR_DOF],
    /// Monte-Carlo standard error of the summed forecast figure.
    pub val_forecast_nll_se: f64,
    /// Context, in bars, the promotion decision was taken at this tick. Unmeasured when it
    /// was skipped.
    pub val_promotion_context: f64,
    /// Longest context the run has taken an optimizer step at.
    pub reached_context: f64,
    /// The promotion decision this tick took, in full: both criteria, both incumbents, the
    /// thresholds actually applied and the outcome.
    ///
    /// Promoted OR refused. A refusal is the interesting half — the rule exists to refuse —
    /// and a rule whose refusals leave no trace is one nobody can audit. Charted on
    /// `pretrain_promotions`, which is the promotion LEDGER and not a step count.
    pub selection: SelectionLedger,
    /// Metrics this tick did NOT measure, each with the reason.
    ///
    /// A NaN in a val column is indistinguishable from a measured catastrophe, which is how a
    /// run can be 62% complete with every held-out column empty and nothing saying so.
    /// Declaring the gap makes the reporter name the metric and the reason the FIRST time it
    /// is skipped, and the series omits the point rather than carrying a NaN.
    pub unmeasured: Vec<UnmeasuredMetric>,
    /// Fraction of each ramp stage's SHARE of the current pass actually issued so far. One
    /// entry per stage; empty when the caller does not track it. A completed pass reads 1.0 at
    /// every stage: the stages partition the corpus, so a stage falling short is a coverage
    /// hole rather than a property of the curriculum.
    pub stage_coverage: Vec<f64>,
    /// Training-split bars targeted EXACTLY ONCE this pass, as a fraction of the split.
    pub pass_coverage: f64,
    /// Bars by how many times a window targeted them THIS PASS: 0, 1, 2, 3-or-more. Charted
    /// because an aggregate coverage number cannot show unevenness — the pre-partition sampler
    /// read 28.7% / 45.9% / 22.3% / 3.2% here while reporting "one epoch".
    ///
    /// PER PASS, AND ONLY PER PASS. `CoverageAudit::require_full_pass` pins within-pass
    /// multiplicity to exactly one, so on a healthy multi-epoch run this reads "twice: 0, three
    /// or more: 0" on the third pass exactly as on the first. Read as a claim about the RUN it
    /// asserts that no bar was ever seen twice, which is how a three-pass run was believed
    /// single-pass for a whole analysis session. [`Self::run_exposure_bars`] is the run-scoped
    /// counterpart and the two are charted together for exactly that reason.
    pub pass_multiplicity_bars: [u64; MULTIPLICITY_BUCKETS],
    /// Bars by how many times the RUN has targeted them across EVERY pass so far: 0, 1, 2,
    /// 3-or-more. The cross-pass counterpart of [`Self::pass_multiplicity_bars`], on the same
    /// denominator and the same bar-token convention so the two are comparable on one panel.
    /// All zeros when the caller does not track it, which is why the reporter gates on the
    /// row's total rather than charting a zero share as a measured zero.
    pub run_exposure_bars: [u64; MULTIPLICITY_BUCKETS],
    /// Passes over the training split the run has DELIVERED, counting the partial one in
    /// progress. Above one means classical multi-epoch reuse is live.
    pub run_effective_epochs: f64,
    /// Passes the run will have delivered by its last step if the ramp it is executing holds.
    /// Known from step zero, so a multi-epoch recipe is visible at the FIRST validation tick
    /// rather than only once the realized curve has crossed one.
    pub projected_effective_epochs: f64,
    /// Passes the recipe ASKED for: `--epochs`. Charted beside the realized and projected
    /// curves because "intended three and got three" and "intended one and got three" call for
    /// opposite responses, and neither curve alone distinguishes them.
    pub planned_effective_epochs: f64,
    /// Bars a pass cannot reach, by named cause: head, symbols shorter than the shortest ramp
    /// context, sub-context hole, and windows the schedule never got to.
    pub pass_remainder_bars: [u64; 4],
    /// `(context + 1) / 2` per stage: mean bars of history a target bar of that stage is
    /// predicted from. Reported per stage because the stages own disjoint shares, so this is
    /// the depth a bar's partition decided for it.
    pub stage_conditioning_bars: Vec<f64>,
    /// Per-DOF split of the diagnostic NLL into the degeneracy class and the continuous
    /// shape. A head that only learned which bars are degenerate posts its whole gain in
    /// `class`, which the undivided number cannot distinguish from intra-bar skill.
    pub val_nll_dof_class: [f64; BAR_DOF],
    pub val_nll_dof_shape: [f64; BAR_DOF],
    /// Log-optimal (Kelly) trading bench on the pinned diagnostic windows: what the
    /// predictive distribution is worth in growth terms against the unconditional null.
    /// See [`super::trade_bench`]; `TradeBench::nan()` when it was not measured.
    pub trade: TradeBench,
}

impl EpochMetrics {
    pub fn nan() -> Self {
        Self {
            epoch: 0,
            global_step: 0,
            train_nll_bar: f64::NAN,
            val_nll_bar: f64::NAN,
            best_val_nll_bar: f64::NAN,
            promoted_checkpoint: None,
            val_nll_bar_diag: f64::NAN,
            train_nll_dof: [f64::NAN; BAR_DOF],
            val_nll_dof: [f64::NAN; BAR_DOF],
            val_crps_dof: [f64::NAN; BAR_DOF],
            val_pit: PitHistogram::default(),
            val_dir_acc: f64::NAN,
            rollout_nll_exact: [f64::NAN; ROLLOUT_HORIZONS.len()],
            rollout_nll_dynamics: [f64::NAN; ROLLOUT_HORIZONS.len()],
            unique_bar_reuse: f64::NAN,
            effective_rank: f64::NAN,
            val_nll_bar_se: f64::NAN,
            val_nll_bar_ci: (f64::NAN, f64::NAN),
            val_nll_bar_se_level: f64::NAN,
            val_nll_bar_conditional: f64::NAN,
            val_nll_bar_conditional_deployed: f64::NAN,
            val_forecast_nll_dof: [f64::NAN; BAR_DOF],
            val_forecast_teacher_nll_dof: [f64::NAN; BAR_DOF],
            val_forecast_nll_se: f64::NAN,
            val_promotion_context: f64::NAN,
            reached_context: f64::NAN,
            selection: SelectionLedger::unmeasured(),
            unmeasured: Vec::new(),
            val_nll_dof_class: [f64::NAN; BAR_DOF],
            val_nll_dof_shape: [f64::NAN; BAR_DOF],
            trade: TradeBench::nan(),
            val_nll_dof_conditional: [f64::NAN; BAR_DOF],
            stage_coverage: Vec::new(),
            pass_coverage: f64::NAN,
            pass_multiplicity_bars: [0; MULTIPLICITY_BUCKETS],
            run_exposure_bars: [0; MULTIPLICITY_BUCKETS],
            run_effective_epochs: f64::NAN,
            projected_effective_epochs: f64::NAN,
            planned_effective_epochs: f64::NAN,
            pass_remainder_bars: [0; 4],
            stage_conditioning_bars: Vec::new(),
        }
    }
}

/// One EPOCH BOUNDARY's progress record.
///
/// A second, much coarser axis than the record tick, and deliberately not folded into it.
/// [`EpochMetrics`] fires at every `--validate-every` interval, which on a production run
/// is hundreds of dense, noisy points; this fires once per pass over the corpus, which is
/// the granularity at which "is the predictor getting better" is a question with an
/// answer. The two sets of series live under different report bases and neither overwrites
/// the other.
///
/// Every bar-token count here is REALIZED — the sum of `batch * context` over the steps
/// that actually ran, at the batch ramp that actually executed. That is the whole point of
/// carrying them: a run whose ramp was held delivers a fraction of the tokens its step
/// count was sized for, and the fraction is what a reader must see rather than the number
/// of epochs that were requested.
#[derive(Clone, Debug)]
pub struct EpochBoundary {
    /// Epoch that just COMPLETED, matching the `pretrain_epoch_<n>_ctx<c>.ot` artifact
    /// written at the same boundary.
    pub epoch: usize,
    pub global_step: usize,
    /// Bar-tokens consumed inside this epoch alone.
    pub epoch_bar_tokens: u64,
    /// Bar-tokens one full pass over the unique training bars costs.
    pub full_pass_bar_tokens: u64,
    /// Bar-tokens consumed by the run so far.
    pub run_bar_tokens: u64,
    /// Bar-tokens `--epochs` asked for: `full_pass_bar_tokens * epochs`.
    pub run_target_bar_tokens: u64,
    /// Bar-tokens the run will have delivered at its last step IF the ramp it is executing
    /// right now holds. Compared against `run_target_bar_tokens` this turns an
    /// end-of-run surprise into a first-boundary one.
    pub projected_run_bar_tokens: u64,
    /// Wall clock of this epoch, boundary work included.
    pub epoch_secs: f64,
    /// Wall clock this boundary itself spent on the bench, the snapshots and the epoch
    /// artifact — the price of being able to watch progress instead of only its endpoint.
    pub boundary_secs: f64,
    pub bench_secs: f64,
    pub snapshot_secs: f64,
    /// Held-out NLL at the fixed diagnostic context.
    pub val_nll_bar: f64,
    /// The forecast-only figure on the same rows, and how much teacher-forcing flatters it.
    pub forecast_nll_bar: f64,
    pub teacher_forcing_inflation: f64,
    /// Mean `dyn / identity` over the epoch's optimizer steps.
    pub dyn_vs_identity: f64,
    /// The Kelly bench of this boundary's diagnostic pass.
    pub trade: TradeBench,
}

impl EpochBoundary {
    pub fn nan() -> Self {
        Self {
            epoch: 0,
            global_step: 0,
            epoch_bar_tokens: 0,
            full_pass_bar_tokens: 0,
            run_bar_tokens: 0,
            run_target_bar_tokens: 0,
            projected_run_bar_tokens: 0,
            epoch_secs: f64::NAN,
            boundary_secs: f64::NAN,
            bench_secs: f64::NAN,
            snapshot_secs: f64::NAN,
            val_nll_bar: f64::NAN,
            forecast_nll_bar: f64::NAN,
            teacher_forcing_inflation: f64::NAN,
            dyn_vs_identity: f64::NAN,
            trade: TradeBench::nan(),
        }
    }

    /// Bar-tokens this epoch delivered as a share of one full pass. Exactly `1.0` for an
    /// epoch that ran to a boundary; below it for the trailing partial epoch a run ends on.
    pub fn pass_fraction(&self) -> f64 {
        ratio(self.epoch_bar_tokens, self.full_pass_bar_tokens)
    }

    /// Bar-tokens delivered so far against everything `--epochs` asked for.
    pub fn delivered_fraction(&self) -> f64 {
        ratio(self.run_bar_tokens, self.run_target_bar_tokens)
    }

    /// The same at the last step, at the ramp currently executing. THE shortfall number:
    /// `--epochs 3` under a held ramp projects ~0.44 here from the very first boundary.
    pub fn projected_fraction(&self) -> f64 {
        ratio(self.projected_run_bar_tokens, self.run_target_bar_tokens)
    }

    /// Effective passes over the corpus the run will have made, at the current ramp.
    pub fn projected_epochs(&self) -> f64 {
        ratio(self.projected_run_bar_tokens, self.full_pass_bar_tokens)
    }

    /// Boundary cost as a share of the epoch it closed. This is the number that decides
    /// whether the bench budget has to come down.
    pub fn boundary_share(&self) -> f64 {
        if self.epoch_secs > 0.0 {
            self.boundary_secs / self.epoch_secs
        } else {
            f64::NAN
        }
    }

    /// The compact console line: everything a reader needs to judge one epoch, in order of
    /// what they will ask. Deliberately one line and deliberately not abbreviated — it is
    /// read out of a 40-hour log, not off a dashboard.
    pub fn console_line(&self) -> String {
        format!(
            "EPOCH {} at step {} | {:.1}M bar-tokens = {:.3} of a full pass ({:.1}M unique \
             bars) | run {:.1}M / {:.1}M requested = {:.3}, projecting {:.1}M = {:.2} \
             effective epochs | {:.1} min ({:.1} min boundary = {:.1}%, bench {:.1} min, \
             snapshots {:.1} min) | held-out nll {:.4}, forecast-only {:.4} \
             (teacher-forcing {:+.4} optimistic), dyn/identity {:.3} | trade edge {:+.4} \
             bps/bar (95% CI {:+.4}..{:+.4}), break-even {}, |f| {:.2} with {:.0}% of bars \
             at the {:.1}x cap, {} ruined bars",
            self.epoch,
            self.global_step,
            self.epoch_bar_tokens as f64 / 1e6,
            self.pass_fraction(),
            self.full_pass_bar_tokens as f64 / 1e6,
            self.run_bar_tokens as f64 / 1e6,
            self.run_target_bar_tokens as f64 / 1e6,
            self.delivered_fraction(),
            self.projected_run_bar_tokens as f64 / 1e6,
            self.projected_epochs(),
            self.epoch_secs / 60.0,
            self.boundary_secs / 60.0,
            100.0 * self.boundary_share(),
            self.bench_secs / 60.0,
            self.snapshot_secs / 60.0,
            self.val_nll_bar,
            self.forecast_nll_bar,
            self.teacher_forcing_inflation,
            self.dyn_vs_identity,
            self.trade.model_edge().mean * 1e4,
            self.trade.model_edge().ci_low * 1e4,
            self.trade.model_edge().ci_high * 1e4,
            break_even_label(&self.trade),
            self.trade.policies[POLICY_MODEL].mean_abs_position,
            100.0 * self.trade.policies[POLICY_MODEL].clamped_fraction,
            self.trade.leverage_cap,
            self.trade.policies[POLICY_MODEL].ruin_bars,
        )
    }
}

/// `numerator / denominator`, or NaN when the denominator says nothing was measured.
fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        f64::NAN
    } else {
        numerator as f64 / denominator as f64
    }
}

/// One metric a validation tick deliberately did not measure, and why.
#[derive(Clone, Debug)]
pub struct UnmeasuredMetric {
    /// Field name as it appears on [`EpochMetrics`], so the warning names the thing the
    /// reader will go looking for.
    pub metric: String,
    pub reason: String,
}

/// The [`EpochMetrics`] fields that exist only once the promotion pass has run at the
/// deployed context.
///
/// Named here rather than at the call site so the declaration and the reporter's own list of
/// charted val metrics cannot drift apart: `a_skipped_promotion_declares_every_gated_metric`
/// asserts every one of these is a metric the reporter actually checks.
pub const DEPLOYED_CONTEXT_METRICS: [&str; 6] = [
    "val_nll_bar",
    "val_nll_bar_se",
    "val_nll_bar_ci",
    "val_nll_bar_se_level",
    "val_nll_bar_conditional_deployed",
    "val_promotion_context",
];

/// End-of-run held-out battery, emitted exactly once as `pretrain_test`.
///
/// The validation split drives promotion, so across an ablation campaign it
/// stops being an unbiased estimate of generalization: we select against it
/// repeatedly and it drifts optimistic. This battery is the split that is
/// touched once, after the last promotion decision, and never feeds back into
/// any decision. [`PretrainReporter::finish`] enforces both properties; see its
/// documentation for exactly what is checked and what is not.
#[derive(Clone, Debug)]
pub struct TestBattery {
    /// The checkpoint passed to `BarWorldModel::load`. Must be the promoted one.
    pub checkpoint: PathBuf,
    /// `BarWorldModel::lineage_sha256()` of the reloaded model. Evidence that the
    /// numbers came from an artifact read back off disk rather than from the
    /// in-memory training model.
    pub model_lineage: String,
    pub nll_bar: f64,
    pub nll_dof: [f64; BAR_DOF],
    pub crps_dof: [f64; BAR_DOF],
    pub rollout_nll_exact: [f64; ROLLOUT_HORIZONS.len()],
    pub rollout_nll_dynamics: [f64; ROLLOUT_HORIZONS.len()],
    pub pit: PitHistogram,
    /// Sign accuracy of `E[r]` against the realized return.
    pub dir_acc: f64,
    /// `BarCorpus::identity_fingerprint()` of the corpus this battery scored. The corpus is
    /// live and the split instants are percentiles of it, so without this two batteries a
    /// week apart are not the same measurement and nothing says so.
    pub corpus_fingerprint: String,
    /// `(train|val, val|test)` instants, in epoch millis.
    pub split_bounds: (i64, i64),
    /// `nll_bar` with the `s == 0 => u = v = 0.5` encoding tautology excluded.
    pub nll_bar_conditional: f64,
    pub nll_dof_conditional: [f64; BAR_DOF],
    /// Block-bootstrap standard error and 95% interval of `nll_bar`.
    pub nll_bar_se: f64,
    pub nll_bar_ci: (f64, f64),
    /// Per-DOF MARGINALIZED forecast NLL, and the teacher-forced figure on identical rows.
    ///
    /// `nll_bar` above is the joint bar likelihood with every chain factor teacher-forced on
    /// the realized same-bar prefix, which makes its per-factor terms within-bar accounting
    /// rather than forecasts. These two are the honest forecasting number and its paired
    /// comparator, so the terminal report states the inflation instead of leaving it to be
    /// rediscovered by the next reviewer.
    pub forecast_nll_dof: [f64; BAR_DOF],
    pub forecast_teacher_nll_dof: [f64; BAR_DOF],
    /// Monte-Carlo standard error of `forecast_nll_dof.iter().sum()`.
    pub forecast_nll_se: f64,
    /// Context the scored checkpoint was SELECTED at, the context it is meant to be deployed
    /// at, and the longest context the run trained at. Equal on a full run; the first is
    /// shorter on a run that never reached the deployed context, and then every number here
    /// carries that caveat.
    pub selection_context: i64,
    pub deployed_context: i64,
    pub reached_context: i64,
    /// The run's `--lr-plateau-fraction`: the fraction of the run held at the flat
    /// learning-rate plateau before the linear decay to the floor.
    ///
    /// In the report because every figure below is a reading of ONE checkpoint at one point on
    /// that schedule, and past the plateau the passes and rate axes are the same axis. A
    /// one-epoch run at 0.40 ends fully annealed and at 0.90 ends at the peak rate; a reader
    /// comparing two reports has to be able to see which.
    pub lr_plateau_fraction: f64,
    /// The trading bench on the TEST split, with the identical policy set.
    pub trade: TradeBench,
    /// The artifact the NLL-PRIMARY rule would have shipped, scored on the same test set at
    /// the same context.
    ///
    /// Selection is now economic — the 0.25x-cap trade edge, guarded by paired density
    /// non-regression — because on the run that motivated the change the NLL-primary rule
    /// promoted the best conditional NLL of the run and one of its worst economic reads. That
    /// change is a claim, and a claim justified only by the run that produced it is an
    /// assertion. This field is the evidence: two artifacts, one held-out split each rule never
    /// saw, both currencies reported. `None` when both rules chose the same weights, which is
    /// itself a finding.
    pub nll_rule: Option<RivalSelection>,
}

/// The rival selection rule's artifact as the test split measures it.
///
/// Deliberately a small flat record rather than a second [`TestBattery`]: the comparison needs
/// the two currencies the rules disagree in and the step each one chose, and a full second
/// battery would invite the reader to treat the rival as a shipped artifact. It is not one —
/// the planner never loads it.
#[derive(Clone, Debug)]
pub struct RivalSelection {
    pub checkpoint: PathBuf,
    pub model_lineage: String,
    /// Global step the rival rule selected.
    pub step: usize,
    pub nll_bar_conditional: f64,
    pub nll_dof: [f64; BAR_DOF],
    /// Net Kelly edge over the unconditional-marginal null at the SELECTION cap, in bps/bar:
    /// the criterion the economic rule maximizes, measured on the rival's own weights.
    pub selection_edge_bps: f64,
    /// The same at the headline 4x cap, in bps/bar, where 85% of bars are at the cap.
    pub edge_at_default: f64,
    /// Quarter-Kelly annualized Sharpe, the fractional-Kelly row a deployable size would run.
    pub sharpe: f64,
}

impl TestBattery {
    pub fn nan(checkpoint: PathBuf, model_lineage: String) -> Self {
        Self {
            checkpoint,
            model_lineage,
            nll_bar: f64::NAN,
            nll_dof: [f64::NAN; BAR_DOF],
            crps_dof: [f64::NAN; BAR_DOF],
            rollout_nll_exact: [f64::NAN; ROLLOUT_HORIZONS.len()],
            rollout_nll_dynamics: [f64::NAN; ROLLOUT_HORIZONS.len()],
            pit: PitHistogram::default(),
            dir_acc: f64::NAN,
            corpus_fingerprint: String::new(),
            split_bounds: (0, 0),
            nll_bar_conditional: f64::NAN,
            nll_dof_conditional: [f64::NAN; BAR_DOF],
            nll_bar_se: f64::NAN,
            nll_bar_ci: (f64::NAN, f64::NAN),
            forecast_nll_dof: [f64::NAN; BAR_DOF],
            forecast_teacher_nll_dof: [f64::NAN; BAR_DOF],
            forecast_nll_se: f64::NAN,
            selection_context: 0,
            deployed_context: 0,
            reached_context: 0,
            lr_plateau_fraction: f64::NAN,
            trade: TradeBench::nan(),
            nll_rule: None,
        }
    }
}

/// Reference lines that need more than the fitted supports to compute, plus the scoring
/// rule they are all expressed in.
///
/// Two of them exist because the headline "X nats better than the calibrated marginal"
/// claim was, until now, comparing a held-out number against a TRAIN-fitted baseline that
/// also credits an arithmetic identity of the encoding as skill. Every line here is
/// recomputed per scoring mode, so a chart can never draw a `smoothed` yardstick under a
/// `density` curve.
#[derive(Clone, Copy, Debug)]
pub struct HeldOutBaselines {
    /// Scoring rule every figure in this struct, and every `nll` series it is drawn against,
    /// is measured under.
    pub scoring: BarScoring,
    /// Nats/bar a UNIFORM-over-bins head pays under `scoring`, which is where the
    /// zero-initialized emission head starts and the zero of the gain-vs-baselines chart.
    pub uniform_nll_bar: f64,
    /// Per-DOF marginal entropy with the `s == 0 => u = v = 0.5` identity removed: the `u`
    /// and `v` references are conditioned on a non-flat bar. This is the yardstick for
    /// `val_nll_dof_conditional`.
    pub marginal_nll_dof_conditional: [f64; BAR_DOF],
    /// The TRAIN-fitted `q*` scored as a fixed prediction against the pinned VALIDATION
    /// windows, per DOF. For `r` and `w` the equal-mass binning makes `q*` nearly uniform
    /// and this barely moves; for `s`, `u` and `v` the entire advantage over uniform is four
    /// measured point masses, and those are liquidity statistics that shift with the regime.
    /// The gap against the train figure IS the distribution shift.
    pub marginal_nll_dof_val: [f64; BAR_DOF],
    /// Nats/bar any model collects for free from `s == 0 => u = v = 0.5`. Roughly 0.690 on
    /// the live supports, i.e. ~19% of the gain a trained model currently reports over the
    /// calibrated marginal.
    pub encoding_identity_nats: f64,
    /// Nats/bar of the scoring rule's unreachable floor, so the reachable range is
    /// `marginal - floor`, not `marginal`. Exactly zero for `hard` and `density`, which
    /// score the bin the observation actually landed in; the label-smoothing entropy for
    /// `smoothed`.
    pub scoring_floor_bar: f64,
    /// Per-DOF split of the MARGINAL into the atom-vs-continuous indicator and the
    /// intra-continuous shape: the reference the measured `class` / `shape` curves are read
    /// against.
    pub marginal_class_dof: [f64; BAR_DOF],
    pub marginal_shape_dof: [f64; BAR_DOF],
}

impl HeldOutBaselines {
    pub fn nan() -> Self {
        Self {
            scoring: BarScoring::default(),
            uniform_nll_bar: f64::NAN,
            marginal_nll_dof_conditional: [f64::NAN; BAR_DOF],
            marginal_nll_dof_val: [f64::NAN; BAR_DOF],
            encoding_identity_nats: f64::NAN,
            scoring_floor_bar: f64::NAN,
            marginal_class_dof: [f64::NAN; BAR_DOF],
            marginal_shape_dof: [f64::NAN; BAR_DOF],
        }
    }

    fn sum(values: &[f64; BAR_DOF]) -> f64 {
        if values.iter().all(|v| v.is_finite()) {
            values.iter().sum()
        } else {
            f64::NAN
        }
    }

    /// Nats/bar of the conditional reference.
    pub fn marginal_nll_bar_conditional(&self) -> f64 {
        Self::sum(&self.marginal_nll_dof_conditional)
    }

    /// Nats/bar of the train-fitted marginal measured on the held-out windows.
    pub fn marginal_nll_bar_val(&self) -> f64 {
        Self::sum(&self.marginal_nll_dof_val)
    }
}

/// Ancestral samples for the candle snapshot pictures.
///
/// The reporter never invokes the model: the caller draws the samples with
/// `BarWorldModel::rollout` and hands the tensor over, which keeps report
/// emission independent of the model surface and directly testable.
pub struct SnapshotInput<'a> {
    /// `[W, samples, H, BAR_DOF]` ancestral rollout of the pinned windows.
    pub rollout: &'a Tensor,
    /// `[W, H, BAR_DOF]` realized continuation of the same windows.
    pub future_dof: &'a Tensor,
    pub epoch: usize,
    pub global_step: usize,
}

// ---------------------------------------------------------------------------
// PIT histogram
// ---------------------------------------------------------------------------

/// Per-DOF histogram of the probability integral transform. A calibrated
/// predictive gives a uniform PIT, so a flat histogram is the target shape.
#[derive(Clone, Debug, Default)]
pub struct PitHistogram {
    counts: [[u64; PIT_HIST_BINS]; BAR_DOF],
}

impl PitHistogram {
    /// Fold a `[..., BAR_DOF]` PIT tensor, as produced by
    /// `bar_dist::bar_pit_from_logits(logits, target_dof, supports, seed)`, into
    /// the running counts. Pass the run seed there so the atom-randomized PIT is
    /// reproducible and these histograms stay diffable across ablations.
    pub fn accumulate(&mut self, pit: &Tensor) {
        let flat = pit
            .detach()
            .to_device(Device::Cpu)
            .to_kind(Kind::Float)
            .reshape([-1, BAR_DOF as i64])
            .contiguous();
        let numel = flat.numel();
        let mut values = vec![0.0f32; numel];
        flat.copy_data(&mut values, numel);
        for row in values.chunks_exact(BAR_DOF) {
            for (dof, &p) in row.iter().enumerate() {
                if !p.is_finite() {
                    continue;
                }
                let bin = ((p as f64) * PIT_HIST_BINS as f64).floor() as isize;
                let bin = bin.clamp(0, PIT_HIST_BINS as isize - 1) as usize;
                self.counts[dof][bin] += 1;
            }
        }
    }

    pub fn reset(&mut self) {
        self.counts = [[0; PIT_HIST_BINS]; BAR_DOF];
    }

    pub fn is_empty(&self) -> bool {
        self.counts.iter().all(|dof| dof.iter().all(|&c| c == 0))
    }

    /// Bin densities scaled so a perfectly calibrated DOF sits flat at `1.0`.
    pub fn density(&self) -> [[f64; PIT_HIST_BINS]; BAR_DOF] {
        array::from_fn(|dof| {
            let total: u64 = self.counts[dof].iter().sum();
            array::from_fn(|bin| {
                if total == 0 {
                    f64::NAN
                } else {
                    self.counts[dof][bin] as f64 * PIT_HIST_BINS as f64 / total as f64
                }
            })
        })
    }

    /// Per-DOF total variation distance from the uniform PIT, in `[0, 1]`.
    /// Zero is perfect calibration; it is the scalar summary the end-of-run test
    /// battery reports instead of a whole histogram.
    pub fn total_variation(&self) -> [f64; BAR_DOF] {
        let density = self.density();
        array::from_fn(|dof| {
            if density[dof].iter().any(|v| !v.is_finite()) {
                return f64::NAN;
            }
            0.5 * density[dof]
                .iter()
                .map(|v| (v / PIT_HIST_BINS as f64 - 1.0 / PIT_HIST_BINS as f64).abs())
                .sum::<f64>()
        })
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// Participation ratio of the belief covariance, `(sum lambda)^2 / sum lambda^2`.
///
/// For a symmetric covariance that equals `trace(C)^2 / ||C||_F^2`, so no
/// eigendecomposition is needed. Ranges from `1` (rank-one collapse) to the
/// belief width (isotropic). Purely a diagnostic: nothing optimizes it.
pub fn belief_effective_rank(beliefs: &Tensor) -> f64 {
    tch::no_grad(|| {
        let dim = match beliefs.size().last() {
            Some(&d) if d > 0 => d,
            _ => return f64::NAN,
        };
        let flat = beliefs
            .detach()
            .to_kind(Kind::Float)
            .reshape([-1, dim])
            .contiguous();
        let rows = flat.size()[0];
        if rows < 2 {
            return f64::NAN;
        }
        let centered = &flat - flat.mean_dim([0i64].as_slice(), true, Kind::Float);
        let cov = centered.transpose(0, 1).matmul(&centered) / (rows - 1) as f64;
        let trace = cov.diagonal(0, 0, 1).sum(Kind::Float).double_value(&[]);
        let frobenius = cov.pow_tensor_scalar(2.0).sum(Kind::Float).double_value(&[]);
        if !trace.is_finite() || !frobenius.is_finite() || frobenius <= 0.0 {
            f64::NAN
        } else {
            trace * trace / frobenius
        }
    })
}

// ---------------------------------------------------------------------------
// Series plumbing
// ---------------------------------------------------------------------------

/// A sparse curve on the shared record-tick axis.
///
/// [`Self::set`] DROPS a non-finite value, so a metric that was not measured leaves a gap
/// instead of a NaN pretending to be a measurement, and a curve that was never measured at all
/// says so in its own label — see [`Self::labeled`].
#[derive(Clone, Debug, Default)]
struct Series(Vec<f32>);

impl Series {
    fn set(&mut self, tick: usize, value: f64) {
        if !value.is_finite() {
            return;
        }
        if self.0.len() <= tick {
            self.0.resize(tick + 1, f32::NAN);
        }
        self.0[tick] = value as f32;
    }

    fn padded(&self, len: usize) -> Vec<f32> {
        let mut values = self.0.clone();
        values.resize(len, f32::NAN);
        values
    }

    /// Whether the series ever received a finite value. Charts that only exist once a
    /// producer has run gate on it.
    fn measured(&self) -> bool {
        self.0.iter().any(|value| value.is_finite())
    }

    /// Label the curve, appending `(NOT MEASURED)` when it holds no finite value at all.
    ///
    /// A chart with an empty `val` line and a healthy `train` line is exactly what a run with
    /// a NaN held-out column looks like, and the two possible readings — "never evaluated" and
    /// "evaluated as a catastrophe" — call for opposite responses. The legend answers it.
    fn labeled(&self, label: &str, len: usize) -> ReportSeries {
        let measured = self.measured();
        ReportSeries {
            label: if measured {
                label.to_owned()
            } else {
                format!("{label} (NOT MEASURED)")
            },
            values: self.padded(len),
        }
    }
}

/// Running mean that ignores non-finite contributions.
#[derive(Clone, Copy, Debug, Default)]
struct Mean {
    sum: f64,
    count: usize,
}

impl Mean {
    fn push(&mut self, value: f64) {
        if value.is_finite() {
            self.sum += value;
            self.count += 1;
        }
    }

    fn value(self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.sum / self.count as f64
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct StepAccumulator {
    steps: usize,
    nll_bar: Mean,
    nll_dof: [Mean; BAR_DOF],
    dyn_loss: Mean,
    kl_loss: Mean,
    total_loss: Mean,
    nll_share: Mean,
    dyn_share: Mean,
    kl_share: Mean,
    growth_loss: Mean,
    growth_share: Mean,
    growth_abs_f: Mean,
    growth_clamp_bind: Mean,
    belief_autocorr: Mean,
    dyn_vs_identity: Mean,
    lr_mult: Mean,
    muon_momentum: Mean,
    grad_norm: Mean,
    context: Mean,
    batch_size: Mean,
    bars_seen: u64,
    free_vram_gib: Mean,
    bar_tokens: Mean,
    projected_footprint_gib: Mean,
    /// Constant over a run, so the mean IS the value; kept as a `Mean` only so a tick with no
    /// measured capacity leaves a gap like every other series here.
    capacity_ceiling_gib: Mean,
    /// Bars, not batches: a tick spans steps of different batch and context, so a mean of
    /// per-step SHARES would weight a 24x896 step like a 24x2048 one.
    market_missing_bars: u64,
    market_total_bars: u64,
}

impl StepAccumulator {
    fn push(&mut self, step: &StepMetrics) {
        self.steps += 1;
        self.nll_bar.push(step.nll_bar);
        for (slot, &value) in self.nll_dof.iter_mut().zip(step.nll_dof.iter()) {
            slot.push(value);
        }
        self.dyn_loss.push(step.dyn_loss);
        self.kl_loss.push(step.kl_loss);
        self.total_loss.push(step.total_loss);
        self.nll_share.push(step.nll_share);
        self.dyn_share.push(step.dyn_share);
        self.kl_share.push(step.kl_share);
        self.growth_loss.push(step.growth_loss);
        self.growth_share.push(step.growth_share);
        self.growth_abs_f.push(step.growth_abs_f);
        self.growth_clamp_bind.push(step.growth_clamp_bind);
        self.belief_autocorr.push(step.belief_autocorr);
        self.dyn_vs_identity.push(step.dyn_vs_identity);
        self.lr_mult.push(step.lr_mult);
        self.muon_momentum.push(step.muon_momentum);
        self.grad_norm.push(step.grad_norm);
        self.context.push(step.context as f64);
        self.batch_size.push(step.batch_size as f64);
        self.bars_seen = self.bars_seen.max(step.bars_seen);
        self.free_vram_gib.push(step.free_vram_gib);
        self.bar_tokens.push(step.bar_tokens);
        self.projected_footprint_gib.push(step.projected_footprint_gib);
        self.capacity_ceiling_gib.push(step.capacity_ceiling_gib);
        self.market_missing_bars += step.market_missing_bars;
        self.market_total_bars += step.market_total_bars;
    }
}

// ---------------------------------------------------------------------------
// Reporter
// ---------------------------------------------------------------------------

/// Accumulates every pretraining metric and writes the `.report.bin` set.
pub struct PretrainReporter {
    gens_dir: PathBuf,
    tick: usize,
    ticks_since_flush: usize,
    epoch: usize,
    global_step: usize,
    accumulator: StepAccumulator,
    promotions: usize,
    /// Canonical path and SHA-256 of the most recently promoted checkpoint; the
    /// end-of-run test battery is checked against this.
    promoted_checkpoint: Option<(PathBuf, [u8; 32])>,
    /// Per-DOF entropy of the fitted marginals, the honest yardstick. A head that
    /// only learned the unconditional marginals sits exactly here; beating it is
    /// the first evidence of anything conditional. NaN when the caller has not
    /// supplied it.
    marginal_nll_dof: [f64; BAR_DOF],
    marginal_nll_bar: f64,

    nll_bar_train: Series,
    nll_bar_val: Series,
    nll_bar_best: Series,
    nll_bar_diag: Series,
    nll_dof_train: [Series; BAR_DOF],
    nll_dof_val: [Series; BAR_DOF],
    vs_uniform_train: Series,
    vs_uniform_val: Series,
    vs_uniform_diag: Series,
    marginal_dof: [Series; BAR_DOF],
    crps_dof: [Series; BAR_DOF],
    dyn_loss: Series,
    kl_loss: Series,
    total_loss: Series,
    nll_share: Series,
    dyn_share: Series,
    kl_share: Series,
    growth_loss: Series,
    growth_share: Series,
    growth_abs_f: Series,
    growth_clamp_bind: Series,
    belief_autocorr: Series,
    dyn_vs_identity: Series,
    rollout_exact: [Series; ROLLOUT_HORIZONS.len()],
    rollout_dynamics: [Series; ROLLOUT_HORIZONS.len()],
    dir_acc: Series,
    lr: Series,
    muon_momentum: Series,
    grad_norm: Series,
    unique_bar_reuse: Series,
    effective_rank: Series,
    promotion_trace: Series,
    /// The promotion LEDGER, one point per eligible read. Both criteria, both incumbents, the
    /// thresholds applied, and a cumulative count per refusal reason, so `pretrain_promotions`
    /// answers "what did the rule decide and why" instead of only "how many times".
    selection_edge: Series,
    selection_edge_incumbent: Series,
    selection_edge_gain: Series,
    selection_edge_band: Series,
    selection_turnover: Series,
    selection_rotations: Series,
    selection_nll: Series,
    selection_nll_incumbent: Series,
    selection_nll_delta: Series,
    selection_nll_tolerance: Series,
    selection_dof_delta: Series,
    refused_noise_trace: Series,
    refused_nll_trace: Series,
    refused_dof_trace: Series,
    unmeasurable_trace: Series,
    /// Cumulative counters behind the four refusal traces.
    refused_noise: usize,
    refused_nll: usize,
    refused_dof: usize,
    unmeasurable: usize,
    context: Series,
    batch_size: Series,
    bars_seen: Series,
    /// The capacity panel: what the card had free, what the plan projected each step to cost,
    /// the ceiling that plan was derived against, and the realized shape it ran at.
    free_vram_gib: Series,
    bar_tokens: Series,
    projected_footprint_gib: Series,
    capacity_ceiling_gib: Series,
    /// Market-channel coverage: this tick's observed share, and the run's, both in percent.
    /// The run-scoped counters live here rather than in the trainer because the reporter is
    /// already the run-scoped object and a second pair of counters in the training loop would be
    /// a second thing to keep in step.
    market_observed_pct: Series,
    market_observed_run_pct: Series,
    market_missing_bars: u64,
    market_total_bars: u64,

    pit: Option<[[f64; PIT_HIST_BINS]; BAR_DOF]>,

    /// One entry per snapshot, in snapshot order. Every scalar here is paired with
    /// the standard error of its own estimator: a snapshot statistic taken from
    /// [`SNAPSHOT_SAMPLES`] draws over a handful of windows has a noise floor
    /// large enough to swallow the effects a reader will try to read off it.
    candle_dclose: Vec<f32>,
    candle_dclose_se: Vec<f32>,
    candle_dclose_mc_floor: Vec<f32>,
    candle_band: Vec<f32>,
    candle_coverage_first: Vec<f32>,
    candle_coverage_terminal: Vec<f32>,
    candle_coverage_se: Vec<f32>,
    candle_rank_first: Vec<f32>,
    candle_rank_terminal: Vec<f32>,
    candle_rank_se: Vec<f32>,

    /// Reference lines that need the corpus and the pinned val set, not just the supports.
    /// See [`HeldOutBaselines`]; all NaN until the caller supplies them.
    baselines: HeldOutBaselines,

    nll_bar_ci_low: Series,
    nll_bar_ci_high: Series,
    nll_bar_conditional: Series,
    nll_dof_conditional: [Series; BAR_DOF],
    nll_dof_class: [Series; BAR_DOF],
    nll_dof_shape: [Series; BAR_DOF],
    /// One coverage curve per ramp stage, grown on demand: the reporter does not know how
    /// many stages the schedule has.
    stage_coverage: Vec<Series>,
    /// Split bars targeted exactly once, as a fraction of the split.
    pass_coverage: Series,
    /// Bars at issue multiplicity 0, 1, 2 and 3+ WITHIN ONE PASS, as fractions of the split.
    /// The panel that makes uneven coverage visible directly instead of inferable — and the
    /// panel that, read as a run-level claim, asserts the negation of multi-epoch reuse. It is
    /// charted with `run_reused` beside it so that reading is not available.
    pass_multiplicity: [Series; MULTIPLICITY_BUCKETS],
    /// Bars the RUN has targeted 0, 1, 2 and 3+ times across every pass, as fractions of the
    /// split. Cross-pass, so on the third pass it reads ~0.85 at "3 or more" where
    /// `pass_multiplicity` reads 0 there.
    run_exposure: [Series; MULTIPLICITY_BUCKETS],
    /// Share of the split the RUN has targeted MORE THAN ONCE. Exactly zero on a single-pass
    /// run, so a non-zero value is unambiguous, and it is drawn ON the per-pass multiplicity
    /// panel as well as its own so the contradiction is adjacent rather than elsewhere.
    run_reused: Series,
    /// Passes over the corpus: delivered, projected at the final step, and the `--epochs` the
    /// recipe asked for. Three curves because "intended three" and "accidentally three" need
    /// opposite responses.
    run_effective_epochs: Series,
    projected_effective_epochs: Series,
    planned_effective_epochs: Series,
    /// Unreachable bars by named cause: head, short symbol, sub-context hole, never issued.
    pass_remainder: [Series; 4],
    /// Mean conditioning length per ramp stage, grown on demand like `stage_coverage`.
    stage_conditioning: Vec<Series>,
    nll_bar_conditional_deployed: Series,
    /// The marginalized forecast curves and their teacher-forced comparators, both from the
    /// diagnostic pass on identical rows.
    forecast_nll_bar: Series,
    vs_uniform_forecast: Series,
    forecast_nll_bar_se: Series,
    forecast_teacher_nll_bar: Series,
    forecast_inflation: Series,
    forecast_nll_dof: [Series; BAR_DOF],
    forecast_teacher_nll_dof: [Series; BAR_DOF],
    /// Context of the promotion decision and the longest context trained, charted so a run
    /// held below the deployed context is visible rather than inferred.
    promotion_context: Series,
    reached_context: Series,
    /// The trading bench. Growth series are in BASIS POINTS per bar, which is the unit a
    /// reader can judge: a 5-minute bar's log growth is a number like `4e-5` and charts
    /// of `4e-5` are unreadable.
    trade_growth: [Series; POLICY_COUNT],
    trade_gross: [Series; POLICY_COUNT],
    trade_sharpe: [Series; POLICY_COUNT],
    trade_edge: Series,
    trade_edge_low: Series,
    trade_edge_high: Series,
    trade_oracle_edge: Series,
    trade_break_even: Series,
    trade_capture: Series,
    trade_hit_rate: [Series; POLICY_COUNT],
    trade_turnover: [Series; POLICY_COUNT],
    trade_time_in_market: Series,
    trade_abs_position: Series,
    trade_drawdown_mean: Series,
    trade_drawdown_max: Series,
    /// Latest validation cost curve and, once the run ends, the test-split one. Both live
    /// on the COST axis rather than the record-tick axis, so they are held whole rather
    /// than appended per tick.
    trade_val: Option<TradeBench>,
    trade_test: Option<TradeBench>,
    /// One row per EPOCH BOUNDARY, on its own axis. Held whole rather than decimated into
    /// the record tick: a handful of rows per run, and every series drawn from them has to
    /// stay index-aligned with the others, which independent `Series` could not guarantee.
    epoch_rows: Vec<EpochBoundary>,
    /// Metrics already announced as unmeasured, so the warning fires once per metric per run
    /// instead of on every validation.
    warned_unmeasured: BTreeSet<String>,
}

impl PretrainReporter {
    /// `gens_dir` is the run's `gens` directory; reports land in `gens_dir/<epoch>`,
    /// which is where the TUI scans for them.
    ///
    /// `marginal_nll_dof` is the per-DOF entropy of the fitted supports, i.e. what
    /// a perfectly calibrated marginal head scores. Pass `[f64::NAN; BAR_DOF]` if
    /// it is not available and the marginal reference lines are simply omitted.
    pub fn new(gens_dir: &Path, marginal_nll_dof: [f64; BAR_DOF]) -> Self {
        let marginal_nll_bar = if marginal_nll_dof.iter().all(|v| v.is_finite()) {
            marginal_nll_dof.iter().sum()
        } else {
            f64::NAN
        };
        Self {
            gens_dir: gens_dir.to_path_buf(),
            tick: 0,
            ticks_since_flush: 0,
            epoch: 0,
            global_step: 0,
            accumulator: StepAccumulator::default(),
            promotions: 0,
            promoted_checkpoint: None,
            marginal_nll_dof,
            marginal_nll_bar,
            nll_bar_train: Series::default(),
            nll_bar_val: Series::default(),
            nll_bar_best: Series::default(),
            nll_bar_diag: Series::default(),
            nll_dof_train: array::from_fn(|_| Series::default()),
            nll_dof_val: array::from_fn(|_| Series::default()),
            vs_uniform_train: Series::default(),
            vs_uniform_val: Series::default(),
            vs_uniform_diag: Series::default(),
            marginal_dof: array::from_fn(|_| Series::default()),
            crps_dof: array::from_fn(|_| Series::default()),
            dyn_loss: Series::default(),
            kl_loss: Series::default(),
            total_loss: Series::default(),
            nll_share: Series::default(),
            dyn_share: Series::default(),
            kl_share: Series::default(),
            growth_loss: Series::default(),
            growth_share: Series::default(),
            growth_abs_f: Series::default(),
            growth_clamp_bind: Series::default(),
            belief_autocorr: Series::default(),
            dyn_vs_identity: Series::default(),
            rollout_exact: array::from_fn(|_| Series::default()),
            rollout_dynamics: array::from_fn(|_| Series::default()),
            dir_acc: Series::default(),
            lr: Series::default(),
            muon_momentum: Series::default(),
            grad_norm: Series::default(),
            unique_bar_reuse: Series::default(),
            effective_rank: Series::default(),
            promotion_trace: Series::default(),
            selection_edge: Series::default(),
            selection_edge_incumbent: Series::default(),
            selection_edge_gain: Series::default(),
            selection_edge_band: Series::default(),
            selection_turnover: Series::default(),
            selection_rotations: Series::default(),
            selection_nll: Series::default(),
            selection_nll_incumbent: Series::default(),
            selection_nll_delta: Series::default(),
            selection_nll_tolerance: Series::default(),
            selection_dof_delta: Series::default(),
            refused_noise_trace: Series::default(),
            refused_nll_trace: Series::default(),
            refused_dof_trace: Series::default(),
            unmeasurable_trace: Series::default(),
            refused_noise: 0,
            refused_nll: 0,
            refused_dof: 0,
            unmeasurable: 0,
            context: Series::default(),
            batch_size: Series::default(),
            bars_seen: Series::default(),
            free_vram_gib: Series::default(),
            bar_tokens: Series::default(),
            projected_footprint_gib: Series::default(),
            capacity_ceiling_gib: Series::default(),
            market_observed_pct: Series::default(),
            market_observed_run_pct: Series::default(),
            market_missing_bars: 0,
            market_total_bars: 0,
            pit: None,
            candle_dclose: Vec::new(),
            candle_dclose_se: Vec::new(),
            candle_dclose_mc_floor: Vec::new(),
            candle_band: Vec::new(),
            candle_coverage_first: Vec::new(),
            candle_coverage_terminal: Vec::new(),
            candle_coverage_se: Vec::new(),
            candle_rank_first: Vec::new(),
            candle_rank_terminal: Vec::new(),
            candle_rank_se: Vec::new(),
            baselines: HeldOutBaselines::nan(),
            nll_bar_ci_low: Series::default(),
            nll_bar_ci_high: Series::default(),
            nll_bar_conditional: Series::default(),
            nll_dof_conditional: array::from_fn(|_| Series::default()),
            nll_dof_class: array::from_fn(|_| Series::default()),
            nll_dof_shape: array::from_fn(|_| Series::default()),
            stage_coverage: Vec::new(),
            pass_coverage: Series::default(),
            pass_multiplicity: array::from_fn(|_| Series::default()),
            run_exposure: array::from_fn(|_| Series::default()),
            run_reused: Series::default(),
            run_effective_epochs: Series::default(),
            projected_effective_epochs: Series::default(),
            planned_effective_epochs: Series::default(),
            pass_remainder: array::from_fn(|_| Series::default()),
            stage_conditioning: Vec::new(),
            nll_bar_conditional_deployed: Series::default(),
            forecast_nll_bar: Series::default(),
            vs_uniform_forecast: Series::default(),
            forecast_nll_bar_se: Series::default(),
            forecast_teacher_nll_bar: Series::default(),
            forecast_inflation: Series::default(),
            forecast_nll_dof: array::from_fn(|_| Series::default()),
            forecast_teacher_nll_dof: array::from_fn(|_| Series::default()),
            promotion_context: Series::default(),
            reached_context: Series::default(),
            trade_growth: array::from_fn(|_| Series::default()),
            trade_gross: array::from_fn(|_| Series::default()),
            trade_sharpe: array::from_fn(|_| Series::default()),
            trade_edge: Series::default(),
            trade_edge_low: Series::default(),
            trade_edge_high: Series::default(),
            trade_oracle_edge: Series::default(),
            trade_break_even: Series::default(),
            trade_capture: Series::default(),
            trade_hit_rate: array::from_fn(|_| Series::default()),
            trade_turnover: array::from_fn(|_| Series::default()),
            trade_time_in_market: Series::default(),
            trade_abs_position: Series::default(),
            trade_drawdown_mean: Series::default(),
            trade_drawdown_max: Series::default(),
            trade_val: None,
            trade_test: None,
            epoch_rows: Vec::new(),
            warned_unmeasured: BTreeSet::new(),
        }
    }

    /// Supply the reference lines that only exist once the corpus and the pinned validation
    /// set are known. Call once, before the first [`Self::record_epoch`].
    pub fn set_held_out_baselines(&mut self, baselines: HeldOutBaselines) {
        self.baselines = baselines;
    }

    /// Fold one optimizer step in. Every [`STEP_DECIMATION`] steps this commits a
    /// record tick, and every [`FLUSH_EVERY_TICKS`] ticks it rewrites the charts.
    pub fn record_step(&mut self, step: &StepMetrics) -> Result<()> {
        self.epoch = step.epoch;
        self.global_step = step.step;
        self.accumulator.push(step);
        if self.accumulator.steps < STEP_DECIMATION {
            return Ok(());
        }
        self.commit_steps();
        if self.ticks_since_flush >= FLUSH_EVERY_TICKS {
            self.flush()?;
        }
        Ok(())
    }

    /// Commit a validation tick and rewrite every chart.
    pub fn record_epoch(&mut self, metrics: &EpochMetrics) -> Result<()> {
        self.announce_unmeasured(metrics);
        self.epoch = metrics.epoch;
        self.global_step = metrics.global_step;
        if self.accumulator.steps > 0 {
            self.commit_steps();
        }

        let tick = self.tick;
        let uniform = self.baselines.uniform_nll_bar;
        self.nll_bar_train.set(tick, metrics.train_nll_bar);
        self.nll_bar_val.set(tick, metrics.val_nll_bar);
        self.nll_bar_best.set(tick, metrics.best_val_nll_bar);
        self.nll_bar_diag.set(tick, metrics.val_nll_bar_diag);
        self.vs_uniform_train
            .set(tick, uniform - metrics.train_nll_bar);
        self.vs_uniform_val.set(tick, uniform - metrics.val_nll_bar);
        self.vs_uniform_diag
            .set(tick, uniform - metrics.val_nll_bar_diag);
        for dof in 0..BAR_DOF {
            self.nll_dof_train[dof].set(tick, metrics.train_nll_dof[dof]);
            self.nll_dof_val[dof].set(tick, metrics.val_nll_dof[dof]);
            self.crps_dof[dof].set(tick, metrics.val_crps_dof[dof]);
            self.marginal_dof[dof].set(tick, self.marginal_nll_dof[dof]);
        }
        self.nll_bar_conditional
            .set(tick, metrics.val_nll_bar_conditional);
        self.nll_bar_ci_low.set(tick, metrics.val_nll_bar_ci.0);
        self.nll_bar_ci_high.set(tick, metrics.val_nll_bar_ci.1);
        for dof in 0..BAR_DOF {
            self.nll_dof_conditional[dof].set(tick, metrics.val_nll_dof_conditional[dof]);
            self.nll_dof_class[dof].set(tick, metrics.val_nll_dof_class[dof]);
            self.nll_dof_shape[dof].set(tick, metrics.val_nll_dof_shape[dof]);
        }
        self.nll_bar_conditional_deployed
            .set(tick, metrics.val_nll_bar_conditional_deployed);
        // The forecast panel. `Series::set` drops a non-finite value, so a tick that did not
        // measure these leaves a GAP rather than a NaN that reads as a measured catastrophe.
        let forecast: f64 = metrics.val_forecast_nll_dof.iter().sum();
        let teacher: f64 = metrics.val_forecast_teacher_nll_dof.iter().sum();
        self.forecast_nll_bar.set(tick, forecast);
        self.forecast_nll_bar_se.set(tick, metrics.val_forecast_nll_se);
        self.forecast_teacher_nll_bar.set(tick, teacher);
        self.forecast_inflation.set(tick, forecast - teacher);
        self.vs_uniform_forecast.set(tick, uniform - forecast);
        for dof in 0..BAR_DOF {
            self.forecast_nll_dof[dof].set(tick, metrics.val_forecast_nll_dof[dof]);
            self.forecast_teacher_nll_dof[dof]
                .set(tick, metrics.val_forecast_teacher_nll_dof[dof]);
        }
        self.promotion_context.set(tick, metrics.val_promotion_context);
        self.reached_context.set(tick, metrics.reached_context);
        // Each ramp stage owns a DISJOINT share of the pass, so coverage is per stage and the
        // stage count is the schedule's business, not the reporter's.
        for (stage, fraction) in metrics.stage_coverage.iter().enumerate() {
            if self.stage_coverage.len() <= stage {
                self.stage_coverage.resize_with(stage + 1, Series::default);
            }
            self.stage_coverage[stage].set(tick, *fraction);
        }
        for (stage, bars) in metrics.stage_conditioning_bars.iter().enumerate() {
            if self.stage_conditioning.len() <= stage {
                self.stage_conditioning
                    .resize_with(stage + 1, Series::default);
            }
            self.stage_conditioning[stage].set(tick, *bars);
        }
        self.pass_coverage.set(tick, metrics.pass_coverage);
        // Fractions of the split, not raw bars: the split's bar count is a corpus fact that
        // moves with ingestion, and a reader comparing two runs needs shares.
        //
        // `.max(1)` USED TO BE THE DENOMINATOR HERE and it manufactured certainty from absence:
        // an all-zero row is the `EpochMetrics::nan()` default meaning "not tracked", and
        // 0 / max(1) = 0.0 is finite, so `Series::set` accepted it and the panel drew a measured
        // zero at every bucket. Gate on the row's TOTAL instead, so an untracked pass leaves a
        // gap and `labeled` says `(NOT MEASURED)`.
        let pass_bars = metrics.pass_multiplicity_bars.iter().sum::<u64>();
        if pass_bars > 0 {
            for bucket in 0..MULTIPLICITY_BUCKETS {
                self.pass_multiplicity[bucket]
                    .set(tick, metrics.pass_multiplicity_bars[bucket] as f64 / pass_bars as f64);
            }
        }
        // The cross-pass counterpart, on the SAME denominator so the two panels' shares are
        // literally comparable. This is the series whose absence let a three-pass run read as
        // single-pass: the per-pass histogram above is pinned to a spike at one by
        // `require_full_pass`, so nothing in it can ever report reuse ACROSS passes.
        let run_bars = metrics.run_exposure_bars.iter().sum::<u64>();
        if run_bars > 0 {
            for bucket in 0..MULTIPLICITY_BUCKETS {
                self.run_exposure[bucket]
                    .set(tick, metrics.run_exposure_bars[bucket] as f64 / run_bars as f64);
            }
            self.run_reused.set(
                tick,
                (metrics.run_exposure_bars[2] + metrics.run_exposure_bars[3]) as f64
                    / run_bars as f64,
            );
        }
        self.run_effective_epochs
            .set(tick, metrics.run_effective_epochs);
        self.projected_effective_epochs
            .set(tick, metrics.projected_effective_epochs);
        self.planned_effective_epochs
            .set(tick, metrics.planned_effective_epochs);
        for cause in 0..metrics.pass_remainder_bars.len() {
            self.pass_remainder[cause].set(tick, metrics.pass_remainder_bars[cause] as f64);
        }
        for horizon in 0..ROLLOUT_HORIZONS.len() {
            self.rollout_exact[horizon].set(tick, metrics.rollout_nll_exact[horizon]);
            self.rollout_dynamics[horizon].set(tick, metrics.rollout_nll_dynamics[horizon]);
        }
        self.dir_acc.set(tick, metrics.val_dir_acc);
        self.unique_bar_reuse.set(tick, metrics.unique_bar_reuse);
        self.effective_rank.set(tick, metrics.effective_rank);
        self.record_trade(tick, &metrics.trade);
        if let Some(checkpoint) = &metrics.promoted_checkpoint {
            // A promotion whose artifact is not on disk means the end-of-run test
            // battery would score the wrong file. Fail here, not in hour forty.
            let canonical = checkpoint.canonicalize().with_context(|| {
                format!(
                    "promoted checkpoint {} is not readable; save it before reporting the promotion",
                    checkpoint.display()
                )
            })?;
            let digest = file_digest(&canonical)?;
            self.promoted_checkpoint = Some((canonical, digest));
            self.promotions += 1;
        }
        self.promotion_trace.set(tick, self.promotions as f64);
        // The ledger, one point per read that actually took a decision. `NotEligible` reads are
        // left OUT of every series rather than written as zero: a read where the ramp had not
        // reached the deployed context did not refuse anything, and a zero here would read as
        // one.
        let selection = &metrics.selection;
        if selection.outcome != SelectionOutcome::NotEligible {
            self.selection_edge.set(tick, selection.edge_bps);
            self.selection_edge_incumbent
                .set(tick, selection.incumbent_edge_bps);
            self.selection_edge_gain.set(tick, selection.edge_gain_bps);
            self.selection_edge_band.set(tick, selection.edge_band_bps);
            self.selection_turnover.set(tick, selection.turnover);
            self.selection_rotations.set(tick, selection.rotations);
            self.selection_nll.set(tick, selection.nll_conditional);
            self.selection_nll_incumbent
                .set(tick, selection.incumbent_nll);
            self.selection_nll_delta.set(tick, selection.nll_delta);
            self.selection_nll_tolerance
                .set(tick, selection.nll_tolerance);
            self.selection_dof_delta.set(tick, selection.dof_delta);
            match selection.outcome {
                SelectionOutcome::RefusedInsideNoise => self.refused_noise += 1,
                SelectionOutcome::RefusedNllGuard => self.refused_nll += 1,
                SelectionOutcome::RefusedDofGuard => self.refused_dof += 1,
                SelectionOutcome::Unmeasurable => self.unmeasurable += 1,
                SelectionOutcome::Promoted | SelectionOutcome::NotEligible => {}
            }
            self.refused_noise_trace.set(tick, self.refused_noise as f64);
            self.refused_nll_trace.set(tick, self.refused_nll as f64);
            self.refused_dof_trace.set(tick, self.refused_dof as f64);
            self.unmeasurable_trace.set(tick, self.unmeasurable as f64);
        }
        if !metrics.val_pit.is_empty() {
            self.pit = Some(metrics.val_pit.density());
        }
        self.advance_tick();
        self.flush()
    }

    /// Commit one EPOCH BOUNDARY row and rewrite the epoch-indexed charts.
    ///
    /// Separate from [`Self::record_epoch`] and not a substitute for it. That one owns the
    /// record-tick axis, fires at every validation interval, and is dense and noisy by
    /// design; this one owns an axis whose index is the epoch, fires once per pass over the
    /// corpus, and writes bases no tick series touches. Calling both at the same boundary
    /// is the intended usage and neither clobbers the other's files.
    ///
    /// Ordering is the caller's contract: rows must arrive in boundary order, because the
    /// epoch axis is the row index. The reporter enforces it rather than trusting it,
    /// since an out-of-order row would silently plot one epoch's numbers under another's.
    pub fn record_epoch_boundary(&mut self, boundary: &EpochBoundary) -> Result<()> {
        if let Some(previous) = self.epoch_rows.last() {
            ensure!(
                boundary.global_step >= previous.global_step,
                "epoch boundary rows must arrive in order: epoch {} at step {} follows epoch \
                 {} at step {}",
                boundary.epoch,
                boundary.global_step,
                previous.epoch,
                previous.global_step,
            );
        }
        self.epoch = boundary.epoch;
        self.global_step = boundary.global_step;
        self.epoch_rows.push(boundary.clone());
        self.flush()
    }

    /// Fold one trading bench into the tick series.
    ///
    /// Growth is charted in basis points per bar and drawdown as a wealth fraction;
    /// everything else is already dimensionless. [`Series::set`] drops non-finite values,
    /// so an unmeasured bench leaves gaps rather than plotting a zero edge.
    fn record_trade(&mut self, tick: usize, trade: &TradeBench) {
        for policy in 0..POLICY_COUNT {
            let stats = &trade.policies[policy];
            self.trade_growth[policy].set(tick, stats.net_growth * 1e4);
            self.trade_gross[policy].set(tick, stats.gross_growth * 1e4);
            self.trade_sharpe[policy].set(tick, stats.sharpe);
            self.trade_hit_rate[policy].set(tick, stats.hit_rate);
            self.trade_turnover[policy].set(tick, stats.turnover);
        }
        let model = &trade.policies[POLICY_MODEL];
        self.trade_time_in_market.set(tick, model.time_in_market);
        self.trade_abs_position.set(tick, model.mean_abs_position);
        self.trade_drawdown_mean.set(tick, model.mean_drawdown);
        self.trade_drawdown_max.set(tick, model.max_drawdown);
        self.trade_edge.set(tick, trade.model_edge().mean * 1e4);
        self.trade_edge_low.set(tick, trade.model_edge().ci_low * 1e4);
        self.trade_edge_high.set(tick, trade.model_edge().ci_high * 1e4);
        self.trade_oracle_edge.set(
            tick,
            (trade.policies[POLICY_ORACLE].net_growth
                - trade.policies[POLICY_MARGINAL].net_growth)
                * 1e4,
        );
        // An infinite break-even is a real outcome (cost never removes the edge) but it is
        // not a chartable number, so the series omits it and the chart title states it.
        self.trade_break_even.set(tick, trade.model_break_even());
        self.trade_capture.set(tick, trade.model_capture());
        if trade.measured() {
            self.trade_val = Some(*trade);
        }
    }

    /// Name every val metric this tick did not measure, ONCE per metric per run.
    ///
    /// A chart gap and a NaN look identical to a reader, and "the model is catastrophically
    /// broken" and "nothing was measured here" are not the same finding. Every val metric the
    /// charts carry is checked: a non-finite one that the caller DECLARED unmeasured is
    /// announced with its reason, and one that was not declared is announced as an
    /// undeclared gap, which is a bug in the caller and says so. Either way the series omits
    /// the point — [`Series::set`] drops non-finite values — so no NaN is ever charted as if
    /// it were a measurement.
    ///
    /// Warn rather than fail: a 40-hour run must not die at hour 39 because one diagnostic
    /// went degenerate. The failure mode this closes is silence, not a missing error.
    fn announce_unmeasured(&mut self, metrics: &EpochMetrics) {
        let declared: BTreeMap<&str, &str> = metrics
            .unmeasured
            .iter()
            .map(|entry| (entry.metric.as_str(), entry.reason.as_str()))
            .collect();
        let step = metrics.global_step;
        let mut checked: Vec<(String, f64)> = vec![
            ("val_nll_bar".to_owned(), metrics.val_nll_bar),
            ("val_nll_bar_diag".to_owned(), metrics.val_nll_bar_diag),
            ("val_nll_bar_se".to_owned(), metrics.val_nll_bar_se),
            (
                "val_nll_bar_ci".to_owned(),
                metrics.val_nll_bar_ci.0 + metrics.val_nll_bar_ci.1,
            ),
            (
                "val_nll_bar_se_level".to_owned(),
                metrics.val_nll_bar_se_level,
            ),
            (
                "val_nll_bar_conditional".to_owned(),
                metrics.val_nll_bar_conditional,
            ),
            (
                "val_nll_bar_conditional_deployed".to_owned(),
                metrics.val_nll_bar_conditional_deployed,
            ),
            ("val_dir_acc".to_owned(), metrics.val_dir_acc),
            ("effective_rank".to_owned(), metrics.effective_rank),
            ("val_forecast_nll_se".to_owned(), metrics.val_forecast_nll_se),
            (
                "val_promotion_context".to_owned(),
                metrics.val_promotion_context,
            ),
            ("reached_context".to_owned(), metrics.reached_context),
            ("val_trade_edge".to_owned(), metrics.trade.model_edge().mean),
            (
                "val_trade_growth".to_owned(),
                metrics.trade.policies[POLICY_MODEL].net_growth,
            ),
        ];
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            for (label, values) in [
                ("val_nll_dof", &metrics.val_nll_dof),
                ("val_crps_dof", &metrics.val_crps_dof),
                ("val_nll_dof_conditional", &metrics.val_nll_dof_conditional),
                ("val_nll_dof_class", &metrics.val_nll_dof_class),
                ("val_nll_dof_shape", &metrics.val_nll_dof_shape),
                ("val_forecast_nll_dof", &metrics.val_forecast_nll_dof),
                (
                    "val_forecast_teacher_nll_dof",
                    &metrics.val_forecast_teacher_nll_dof,
                ),
            ] {
                checked.push((format!("{label}[{name}]"), values[dof]));
            }
        }
        for (index, horizon) in ROLLOUT_HORIZONS.iter().enumerate() {
            checked.push((
                format!("rollout_nll_exact[h{horizon}]"),
                metrics.rollout_nll_exact[index],
            ));
            checked.push((
                format!("rollout_nll_dynamics[h{horizon}]"),
                metrics.rollout_nll_dynamics[index],
            ));
        }

        for (metric, value) in checked {
            if value.is_finite() || self.warned_unmeasured.contains(&metric) {
                continue;
            }
            // The per-DOF and per-horizon entries are declared by their family name, because a
            // reason that applies to a pass applies to every column that pass feeds.
            let family = metric.split('[').next().unwrap_or(metric.as_str());
            let reason = declared
                .get(metric.as_str())
                .or_else(|| declared.get(family))
                .copied();
            match reason {
                Some(reason) => println!(
                    "WARNING step {step}: `{metric}` NOT MEASURED — {reason} It is OMITTED from \
                     its series, so the gap in the chart means \"not measured\", never \
                     \"measured and bad\". This is said once per metric per run."
                ),
                None => println!(
                    "WARNING step {step}: `{metric}` came back non-finite and was NOT declared \
                     unmeasured, so nothing can say whether it was skipped or measured as a \
                     catastrophe. It is OMITTED from its series and this is a defect in the \
                     caller: declare it through `EpochMetrics::unmeasured` with a reason. Said \
                     once per metric per run."
                ),
            }
            self.warned_unmeasured.insert(metric);
        }
    }

    /// Ancestral candle snapshots: one `CandleFan` per window — the realized bars
    /// against the ancestral quantile fan and a few genuine draws — plus the
    /// pooled drift, band, coverage and rank-PIT scalars, each with the standard
    /// error of its own estimator.
    pub fn record_snapshot(&mut self, input: &SnapshotInput<'_>) -> Result<()> {
        self.epoch = input.epoch;
        self.global_step = input.global_step;
        let dir = self
            .gens_dir
            .join(self.epoch.to_string())
            .join("candle_snapshots");
        let summary = tch::no_grad(|| {
            write_candle_windows(
                &dir,
                self.global_step,
                Some(self.epoch),
                &input.rollout.detach(),
                input.future_dof,
            )
        })
        .map(|windows| CandleSummary::from_windows(&windows))?;

        self.candle_dclose.push(summary.dclose as f32);
        self.candle_dclose_se.push(summary.dclose_se as f32);
        self.candle_dclose_mc_floor
            .push(summary.dclose_mc_floor as f32);
        self.candle_band.push(summary.band as f32);
        self.candle_coverage_first
            .push(summary.coverage_first as f32);
        self.candle_coverage_terminal
            .push(summary.coverage_terminal as f32);
        self.candle_coverage_se.push(summary.coverage_se as f32);
        self.candle_rank_first.push(summary.rank_first as f32);
        self.candle_rank_terminal.push(summary.rank_terminal as f32);
        self.candle_rank_se.push(summary.rank_se as f32);
        self.flush()
    }

    /// Emit the end-of-run held-out battery as `pretrain_test` and consume the
    /// reporter.
    ///
    /// Two properties are enforced here rather than left to convention, because
    /// a test number that leaks into model selection stops being a test number.
    ///
    /// 1. **It cannot precede the final promotion decision.** Taking `self` by
    ///    value means every promotion must already have been reported: once this
    ///    returns there is no reporter left to call [`Self::record_epoch`] on, so
    ///    a later promotion is a compile error, not a review comment. The old
    ///    loop ordered the battery before the last promotion only by luck.
    /// 2. **It must score the promoted artifact, read back off disk.** The
    ///    checkpoint named by the battery must be the last one reported promoted,
    ///    and its SHA-256 must still match the fingerprint taken at promotion
    ///    time, so scoring the in-memory model, a stale `*_best.ot`, or a file
    ///    rewritten since promotion all fail loudly. `model_lineage` must be
    ///    non-empty, which only a real `BarWorldModel::load` can supply.
    pub fn finish(mut self, battery: &TestBattery) -> Result<()> {
        let (promoted, promoted_digest) = self.promoted_checkpoint.clone().context(
            "no promotion was ever reported, so there is no checkpoint the held-out battery could \
             legitimately score; report the promotion through EpochMetrics::promoted_checkpoint first",
        )?;
        if battery.model_lineage.trim().is_empty() {
            anyhow::bail!(
                "held-out battery carries no model lineage; it must be scored on a model reloaded \
                 through BarWorldModel::load, not on the in-memory training model"
            );
        }
        let scored = battery.checkpoint.canonicalize().with_context(|| {
            format!(
                "held-out battery checkpoint {} is not readable",
                battery.checkpoint.display()
            )
        })?;
        if scored != promoted {
            anyhow::bail!(
                "held-out battery scored {} but the promoted checkpoint is {}; the test split must \
                 only ever measure the artifact that was actually selected",
                scored.display(),
                promoted.display()
            );
        }
        if file_digest(&scored)? != promoted_digest {
            anyhow::bail!(
                "{} changed on disk after it was promoted; the held-out battery would not be \
                 measuring the selected weights",
                scored.display()
            );
        }

        // Held before the flush, so the cost-curve chart carries the TEST curve beside the
        // validation one instead of two files disagreeing about which split they depict.
        self.trade_test = Some(battery.trade);
        self.flush()?;
        let dir = self.gens_dir.join(self.epoch.to_string());
        fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
        let uniform = self.baselines.uniform_nll_bar;
        let mut series = vec![
            point_series("nll_bar", battery.nll_bar),
            point_series("nll_bar vs uniform", uniform - battery.nll_bar),
            point_series(
                "nll_bar vs marginal",
                self.marginal_nll_bar - battery.nll_bar,
            ),
            point_series("uniform", uniform),
            point_series("marginal", self.marginal_nll_bar),
            point_series(
                "scoring mode (0 smoothed, 1 hard, 2 density)",
                match self.baselines.scoring {
                    BarScoring::Smoothed => 0.0,
                    BarScoring::Hard => 1.0,
                    BarScoring::Density => 2.0,
                },
            ),
        ];
        let pit_tv = battery.pit.total_variation();
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            series.push(point_series(&format!("nll {name}"), battery.nll_dof[dof]));
            series.push(point_series(&format!("crps {name}"), battery.crps_dof[dof]));
            series.push(point_series(&format!("pit tv {name}"), pit_tv[dof]));
        }
        for (i, horizon) in ROLLOUT_HORIZONS.iter().enumerate() {
            series.push(point_series(
                &format!("rollout h{horizon} exact"),
                battery.rollout_nll_exact[i],
            ));
            series.push(point_series(
                &format!("rollout h{horizon} dynamics"),
                battery.rollout_nll_dynamics[i],
            ));
        }
        series.push(point_series("dir acc", battery.dir_acc));
        series.push(point_series("nll_bar se", battery.nll_bar_se));
        series.push(point_series("nll_bar ci95 low", battery.nll_bar_ci.0));
        series.push(point_series("nll_bar ci95 high", battery.nll_bar_ci.1));
        series.push(point_series(
            "nll_bar conditional",
            battery.nll_bar_conditional,
        ));
        series.push(point_series(
            "marginal conditional",
            self.baselines.marginal_nll_bar_conditional(),
        ));
        series.push(point_series(
            "encoding identity nats",
            self.baselines.encoding_identity_nats,
        ));
        for dof in [DOF_U, DOF_V] {
            series.push(point_series(
                &format!("nll {} | s!=0", BAR_DOF_NAMES[dof]),
                battery.nll_dof_conditional[dof],
            ));
        }
        // Per-DOF deltas against the marginal, stated rather than left to be subtracted by
        // eye. The aggregate hides that `w` has exactly zero headroom below uniform while
        // `u` and `v` have over a nat each.
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            series.push(point_series(
                &format!("nll {name} vs marginal"),
                self.marginal_nll_dof[dof] - battery.nll_dof[dof],
            ));
        }
        // The honest forecasting number beside the teacher-forced one, on identical rows.
        // `nll_bar` above is a JOINT likelihood in which four of the five factors are handed
        // the realized values of the same bar's earlier factors; only `r` forecasts. These
        // lines say how much of the headline came from that.
        let forecast: f64 = battery.forecast_nll_dof.iter().sum();
        let teacher: f64 = battery.forecast_teacher_nll_dof.iter().sum();
        series.push(point_series("FORECAST nll_bar (marginalized)", forecast));
        series.push(point_series("forecast nll_bar se (MC)", battery.forecast_nll_se));
        series.push(point_series(
            "teacher-forced nll_bar (same rows)",
            teacher,
        ));
        series.push(point_series(
            "teacher-forcing inflation (forecast - teacher)",
            forecast - teacher,
        ));
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            let role = if dof == BAR_CHAIN[0] {
                "pure forecasting"
            } else {
                "marginalized over earlier same-bar factors"
            };
            series.push(point_series(
                &format!("forecast nll {name} ({role})"),
                battery.forecast_nll_dof[dof],
            ));
        }
        series.push(point_series(
            "selection context bars",
            battery.selection_context as f64,
        ));
        series.push(point_series(
            "deployed context bars",
            battery.deployed_context as f64,
        ));
        series.push(point_series(
            "reached context bars",
            battery.reached_context as f64,
        ));
        series.push(point_series(
            "lr plateau fraction",
            battery.lr_plateau_fraction,
        ));
        push_trade_series(&mut series, &battery.trade);
        // The RULE COMPARISON. Selection is economic now; the artifact the previous,
        // NLL-primary rule would have shipped is scored on this same split so the change is
        // evidence rather than an assertion. Both currencies for both artifacts, and the two
        // paired differences, because a promotion that bought edge at the cost of density has
        // to say so on the file itself.
        if let Some(rival) = &battery.nll_rule {
            let rival_lineage: String = rival.model_lineage.chars().take(12).collect();
            series.push(point_series(
                &format!(
                    "RIVAL nll-rule step (lineage {rival_lineage}, {})",
                    rival
                        .checkpoint
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default()
                ),
                rival.step as f64,
            ));
            series.push(point_series(
                &format!("rival edge @{SELECTION_CAP:.2}x cap bps/bar (the criterion)"),
                rival.selection_edge_bps,
            ));
            series.push(point_series(
                &format!("rival edge @{LEVERAGE_CAP:.2}x cap bps/bar (headline)"),
                rival.edge_at_default,
            ));
            series.push(point_series(
                "rival quarter-kelly sharpe (annualized)",
                rival.sharpe,
            ));
            series.push(point_series(
                "rival conditional nll_bar",
                rival.nll_bar_conditional,
            ));
            for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
                series.push(point_series(
                    &format!("rival nll {name}"),
                    rival.nll_dof[dof],
                ));
            }
            // `CapPoint::edge` is net log growth per bar; the criterion is quoted in bps, as
            // everywhere else the cap curve is printed.
            let promoted_edge = battery.trade.cap_curve[SELECTION_CAP_SLOT].edge * 1.0e4;
            series.push(point_series(
                &format!(
                    "RULE DELTA edge @{SELECTION_CAP:.2}x cap, economic - nll (bps/bar, + = the \
                     economic rule won on its own criterion out of sample)"
                ),
                promoted_edge - rival.selection_edge_bps,
            ));
            series.push(point_series(
                "RULE DELTA conditional nll, economic - nll (nats/bar, + = the edge was bought \
                 with density)",
                battery.nll_bar_conditional - rival.nll_bar_conditional,
            ));
        }

        let name = scored
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| scored.display().to_string());
        let lineage: String = battery.model_lineage.chars().take(12).collect();
        let corpus: String = battery.corpus_fingerprint.chars().take(12).collect();
        write_chart(
            &dir,
            "pretrain_test",
            format!(
                "Pretrain Held-out Test Battery - {name} - lineage {lineage} - corpus {corpus} \
                 - split {}|{} - step {}",
                battery.split_bounds.0, battery.split_bounds.1, self.global_step
            ),
            "single evaluation",
            "held-out test split, scored once, never used for selection",
            ScaleKind::Linear,
            series,
        )
    }

    fn commit_steps(&mut self) {
        let tick = self.tick;
        let acc = self.accumulator;
        let nll = acc.nll_bar.value();
        self.nll_bar_train.set(tick, nll);
        self.vs_uniform_train
            .set(tick, self.baselines.uniform_nll_bar - nll);
        for dof in 0..BAR_DOF {
            self.nll_dof_train[dof].set(tick, acc.nll_dof[dof].value());
        }
        self.dyn_loss.set(tick, acc.dyn_loss.value());
        self.kl_loss.set(tick, acc.kl_loss.value());
        self.total_loss.set(tick, acc.total_loss.value());
        self.nll_share.set(tick, acc.nll_share.value());
        self.dyn_share.set(tick, acc.dyn_share.value());
        self.kl_share.set(tick, acc.kl_share.value());
        self.growth_loss.set(tick, acc.growth_loss.value());
        self.growth_share.set(tick, acc.growth_share.value());
        self.growth_abs_f.set(tick, acc.growth_abs_f.value());
        self.growth_clamp_bind.set(tick, acc.growth_clamp_bind.value());
        self.belief_autocorr.set(tick, acc.belief_autocorr.value());
        self.dyn_vs_identity.set(tick, acc.dyn_vs_identity.value());
        self.lr.set(tick, acc.lr_mult.value());
        self.muon_momentum.set(tick, acc.muon_momentum.value());
        self.grad_norm.set(tick, acc.grad_norm.value());
        self.context.set(tick, acc.context.value());
        self.batch_size.set(tick, acc.batch_size.value());
        self.free_vram_gib.set(tick, acc.free_vram_gib.value());
        // In thousands, so the curve shares a symlog decade with the GiB figures beside it
        // instead of pushing them onto the axis floor.
        self.bar_tokens.set(tick, acc.bar_tokens.value() / 1.0e3);
        self.projected_footprint_gib
            .set(tick, acc.projected_footprint_gib.value());
        self.capacity_ceiling_gib
            .set(tick, acc.capacity_ceiling_gib.value());
        if acc.bars_seen > 0 {
            self.bars_seen.set(tick, acc.bars_seen as f64 / 1.0e6);
        }
        if acc.market_total_bars > 0 {
            self.market_missing_bars += acc.market_missing_bars;
            self.market_total_bars += acc.market_total_bars;
            let observed = |missing: u64, total: u64| {
                100.0 * (1.0 - missing as f64 / total as f64)
            };
            self.market_observed_pct.set(
                tick,
                observed(acc.market_missing_bars, acc.market_total_bars),
            );
            self.market_observed_run_pct.set(
                tick,
                observed(self.market_missing_bars, self.market_total_bars),
            );
        }
        self.accumulator = StepAccumulator::default();
        self.advance_tick();
    }

    fn advance_tick(&mut self) {
        self.tick += 1;
        self.ticks_since_flush += 1;
    }

    fn flush(&mut self) -> Result<()> {
        self.ticks_since_flush = 0;
        let dir = self.gens_dir.join(self.epoch.to_string());
        fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
        let len = self.tick;
        let epoch = self.epoch;
        let step = self.global_step;
        // Every nats axis below is in the units of the scoring rule in force, and the three
        // rules differ by tens of nats. The mode belongs in the title of every chart, not
        // only in the banner of a log nobody opens next to the picture.
        let suffix = format!(
            "epoch {epoch} step {step} - scoring {}",
            self.baselines.scoring
        );
        let diag = DIAGNOSTIC_CONTEXT;

        write_chart(
            &dir,
            "pretrain_nll_bar",
            format!("Pretrain Bar NLL - {suffix}"),
            "record",
            "nats/bar (val = promotion metric at the DEPLOYED context, absent until the ramp \
             reaches it; band = 95% block bootstrap)",
            ScaleKind::Linear,
            vec![
                self.nll_bar_train.labeled("train", len),
                self.nll_bar_val.labeled("val deployed", len),
                // The band is the whole point: a val curve without one invites reading a
                // 0.05-nat wiggle as an effect when the interval is four times that wide.
                self.nll_bar_ci_low.labeled("val ci95 low", len),
                self.nll_bar_ci_high.labeled("val ci95 high", len),
                self.nll_bar_best.labeled("best val", len),
                // Excludes the s == 0 => u = v = 0.5 identity, which is ~0.69 nats of the
                // reported gain and is arithmetic rather than prediction.
                self.nll_bar_conditional_deployed
                    .labeled("val deployed conditional", len),
                // A gap in the deployed curves means the ramp had not got there yet, and
                // these two lines are how a reader tells that from a broken model.
                self.promotion_context.labeled("deployed context bars", len),
                self.reached_context.labeled("reached context bars", len),
            ],
        )?;

        write_chart(
            &dir,
            "pretrain_nll_bar_diag896",
            format!("Pretrain Bar NLL (fixed {diag} context) - {suffix}"),
            "record",
            &format!(
                "nats/bar at a pinned {diag} context, measured at EVERY validation from step \
                 0 and comparable across runs"
            ),
            ScaleKind::Linear,
            vec![
                self.nll_bar_diag.labeled("val diag", len),
                self.nll_bar_conditional
                    .labeled("val diag conditional", len),
                self.forecast_nll_bar
                    .labeled("val diag FORECAST (marginalized)", len),
            ],
        )?;

        // The forecast panel. `nll_bar` teacher-forces every chain factor on the realized
        // values of the SAME bar's earlier factors, so four of its five terms are within-bar
        // accounting and only the first is prediction. This chart is the number a forecaster
        // may quote, its Monte-Carlo standard error, the teacher-forced figure on identical
        // rows, and the difference between them.
        let mut forecast = vec![
            self.forecast_nll_bar.labeled("forecast (marginalized)", len),
            self.forecast_teacher_nll_bar
                .labeled("teacher-forced (same rows)", len),
            self.forecast_inflation
                .labeled("teacher-forcing inflation", len),
            self.forecast_nll_bar_se.labeled("forecast MC se", len),
        ];
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            let role = if dof == BAR_CHAIN[0] {
                "forecast"
            } else {
                "forecast marginalized"
            };
            forecast.push(self.forecast_nll_dof[dof].labeled(&format!("{name} {role}"), len));
            forecast.push(
                self.forecast_teacher_nll_dof[dof].labeled(&format!("{name} teacher-forced"), len),
            );
        }
        write_chart(
            &dir,
            "pretrain_forecast_nll",
            format!("Pretrain FORECAST vs teacher-forced Bar NLL - {suffix}"),
            "record",
            &format!(
                "nats/bar at the fixed {diag} context. FORECAST conditions every factor on \
                 strictly PAST bars only, marginalizing the intra-bar chain over the head's own \
                 predictive law; TEACHER-FORCED hands each factor the realized value of the \
                 same bar's earlier factors, so only `{}` forecasts and the other four are \
                 within-bar accounting",
                BAR_DOF_NAMES[BAR_CHAIN[0]]
            ),
            ScaleKind::Linear,
            forecast,
        )?;

        let mut nll_dof = Vec::with_capacity(3 * BAR_DOF);
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            nll_dof.push(self.nll_dof_train[dof].labeled(&format!("{name} train"), len));
        }
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            nll_dof.push(self.nll_dof_val[dof].labeled(&format!("{name} val diag"), len));
        }
        // Per-DOF marginal floors. Without them a DOF like u, whose marginal
        // entropy is far below ln(128) because 42% of bars pin the close to a bar
        // extreme, looks like it is learning when it has only found its marginal.
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            nll_dof.push(self.marginal_dof[dof].labeled(&format!("{name} marginal"), len));
        }
        // The `u` and `v` curves conditioned on a non-flat bar, against the matching
        // conditional floors. Without this pair, a head that has learned only the flat-bar
        // identity shows a large gain on both DOF and nothing distinguishes it from one that
        // learned intra-bar shape.
        for dof in [DOF_U, DOF_V] {
            let name = BAR_DOF_NAMES[dof];
            nll_dof.push(
                self.nll_dof_conditional[dof].labeled(&format!("{name} val | s!=0"), len),
            );
            nll_dof.push(constant_series(
                &format!("{name} marginal | s!=0"),
                self.baselines.marginal_nll_dof_conditional[dof],
                len,
            ));
        }
        // Degeneracy class vs intra-continuous shape, against the marginal's own split. A
        // head that only learned which bars are degenerate posts its whole gain in `class`.
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            nll_dof.push(self.nll_dof_class[dof].labeled(&format!("{name} class"), len));
            nll_dof.push(self.nll_dof_shape[dof].labeled(&format!("{name} shape"), len));
            nll_dof.push(constant_series(
                &format!("{name} class marginal"),
                self.baselines.marginal_class_dof[dof],
                len,
            ));
            nll_dof.push(constant_series(
                &format!("{name} shape marginal"),
                self.baselines.marginal_shape_dof[dof],
                len,
            ));
        }
        write_chart(
            &dir,
            "pretrain_nll_dof",
            format!("Pretrain Bar NLL per DOF - {suffix}"),
            "record",
            &format!("nats (val series at the fixed {diag} context)"),
            ScaleKind::Linear,
            nll_dof,
        )?;

        // Gain over the uniform chain, with every yardstick drawn flat. Crossing `uniform`
        // only proves the head found the unconditional marginals. `marginal` is the
        // TRAIN-fitted reference; `marginal (val)` is that same fixed prediction scored on
        // the held-out windows, and the gap between the two IS the distribution shift.
        // `marginal + encoding identity` adds the ~0.69 nats that `s == 0 => u = v = 0.5`
        // hands out for free — that, not `marginal`, is the line a claim of conditional
        // structure has to clear.
        let uniform = self.baselines.uniform_nll_bar;
        let mut baseline_series = vec![
            self.vs_uniform_train.labeled("train", len),
            self.vs_uniform_val.labeled("val deployed", len),
            self.vs_uniform_diag.labeled("val diag", len),
            // The gain a forecaster actually has. The two curves above are joint likelihoods
            // in which four of five factors are conditioned on the same bar's realized values.
            self.vs_uniform_forecast
                .labeled("val diag FORECAST (marginalized)", len),
            constant_series("uniform", 0.0, len),
            constant_series("marginal", uniform - self.marginal_nll_bar, len),
            constant_series(
                "marginal (val)",
                uniform - self.baselines.marginal_nll_bar_val(),
                len,
            ),
            constant_series(
                "marginal + encoding identity",
                uniform - self.marginal_nll_bar + self.baselines.encoding_identity_nats,
                len,
            ),
        ];
        // Only the smoothed rule has a floor. Under `hard` and `density` the line would sit
        // exactly on `uniform` and assert something false — that an oracle cannot reach the
        // zero of this axis — so it is omitted rather than drawn at zero.
        if self.baselines.scoring_floor_bar > 0.0 {
            baseline_series.push(constant_series(
                "smoothing floor (unreachable)",
                uniform - self.baselines.scoring_floor_bar,
                len,
            ));
        }
        write_chart(
            &dir,
            "pretrain_nll_vs_baselines",
            format!("Pretrain NLL Gain vs Baselines - {suffix}"),
            "record",
            "nats/bar below the uniform chain",
            ScaleKind::Linear,
            baseline_series,
        )?;

        // Per-stage coverage of the pass. The stages own DISJOINT shares of the corpus, so a
        // completed pass reads 1.0 at every stage and anything below it is a coverage hole,
        // not a curriculum. Before the partition this chart topped out at 0.20 / 0.34 / 0.47
        // on the default invocation and it said so on every run, unread, for as long as it
        // existed — which is why the same fact is now also an error at every boundary.
        if !self.stage_coverage.is_empty() {
            write_chart(
                &dir,
                "pretrain_stage_coverage",
                format!("Pretrain Ramp Stage Coverage - {suffix}"),
                "record",
                "windows issued / windows the pass assigned, per ramp stage",
                ScaleKind::Linear,
                self.stage_coverage
                    .iter()
                    .enumerate()
                    .map(|(stage, series)| series.labeled(&format!("stage {stage}"), len))
                    .collect(),
            )?;
        }

        // The per-bar multiplicity distribution, beside the aggregate. An aggregate coverage
        // figure cannot distinguish "85% of bars once" from "70% once, 15% twice, 15% never",
        // and the pre-partition sampler was the second: 28.7% of training bars at zero, 45.9%
        // at one, 22.3% at two and 3.2% at three. A healthy pass is a single spike at one.
        if self.pass_multiplicity.iter().any(Series::measured) {
            // THE PANEL THAT BEAT SIX AGENTS TWICE. It is arithmetically correct and it is a
            // PER-PASS census, so `require_full_pass` guarantees it reads "1 time: ~99.4%,
            // 2 times: 0, 3+ times: 0" on the third pass of a three-pass run exactly as on the
            // first. Read at face value it asserts that no bar was ever seen twice. bardist_v2
            // emitted precisely that at every tick of its third pass while
            // `pretrain_unique_bar_reuse` beside it correctly showed 2.85, and the zeros won.
            //
            // The remedy is NOT stronger wording. The reader who was fooled already had the
            // correct number on screen, on a DIFFERENT panel. So the run-scoped share is drawn
            // ON THIS PANEL, and reaching the false conclusion now requires ignoring an adjacent
            // line that contradicts it rather than merely failing to go looking for a quieter
            // one. The scope lives in the SERIES LABELS because the TUI renders those verbatim
            // while it lowercases titles through `normalize_title`.
            let mut multiplicity: Vec<ReportSeries> = self
                .pass_multiplicity
                .iter()
                .enumerate()
                .map(|(times, series)| {
                    let label = if times + 1 == MULTIPLICITY_BUCKETS {
                        format!("targeted {times}+ times IN THIS PASS")
                    } else {
                        format!("targeted {times} times IN THIS PASS")
                    };
                    series.labeled(&label, len)
                })
                .collect();
            multiplicity.push(self.run_reused.labeled(
                "RUN TOTAL: targeted 2+ times across ALL passes so far (this is the cross-pass \
                 number; the lines above are ONE PASS and are pinned to zero there by the \
                 coverage invariant)",
                len,
            ));
            write_chart(
                &dir,
                "pretrain_pass_multiplicity",
                format!(
                    "Pretrain Bar Multiplicity WITHIN ONE PASS, with the cross-pass total \
                     beside it - {suffix}"
                ),
                "record",
                "share of training-split bars; per-pass curves say nothing about the run",
                ScaleKind::Linear,
                multiplicity,
            )?;
            write_chart(
                &dir,
                "pretrain_pass_coverage",
                format!("Pretrain Coverage of ONE Pass - {suffix}"),
                "record",
                "training-split bars targeted exactly once IN THIS PASS / split bars",
                ScaleKind::Linear,
                vec![self
                    .pass_coverage
                    .labeled("covered exactly once IN THIS PASS (per-pass census)", len)],
            )?;
        }
        // THE RUN-SCOPED PANELS. Everything above is a per-pass census by construction; these
        // two are the only places a reader can learn how many times the model has seen a bar.
        if self.run_effective_epochs.measured() || self.projected_effective_epochs.measured() {
            write_chart(
                &dir,
                "cover_effective_epochs",
                format!("Cover Passes Over The Training Split, Whole Run - {suffix}"),
                "record",
                "passes over the training split (1.0 = every bar seen once)",
                ScaleKind::Linear,
                vec![
                    self.run_effective_epochs.labeled(
                        "passes DELIVERED so far (cross-pass census; above 1.0 the model is \
                         re-reading bars it has already been trained on)",
                        len,
                    ),
                    self.projected_effective_epochs.labeled(
                        "passes PROJECTED by the final step (known from step zero, so a \
                         multi-epoch recipe is visible at the first validation tick)",
                        len,
                    ),
                    self.planned_effective_epochs.labeled(
                        "passes the recipe ASKED for (--epochs); equal to projected means \
                         deliberate, below it means accidental",
                        len,
                    ),
                    // The same quantity by an INDEPENDENT route: bar-tokens consumed over
                    // bar-tokens in one pass, accumulated by the trainer, versus the exposure
                    // census reconstructed from the partition. They must agree; drawing both
                    // makes a disagreement visible instead of leaving one number unaudited.
                    self.unique_bar_reuse.labeled(
                        "same quantity from bar-token throughput (independent path; a gap \
                         between this and DELIVERED is an accounting defect)",
                        len,
                    ),
                    constant_series("1.0 = a single pass over the corpus", 1.0, len),
                ],
            )?;
        }
        if self.run_exposure.iter().any(Series::measured) {
            write_chart(
                &dir,
                "cover_run_bar_exposure",
                format!("Cover Bar Exposure Across The Whole Run - {suffix}"),
                "record",
                "share of training-split bars, by times targeted SO FAR IN THE RUN",
                ScaleKind::Linear,
                self.run_exposure
                    .iter()
                    .enumerate()
                    .map(|(times, series)| {
                        let label = if times + 1 == MULTIPLICITY_BUCKETS {
                            format!(
                                "targeted {times}+ times SO FAR IN THIS RUN (cross-pass, NOT a \
                                 per-pass census)"
                            )
                        } else {
                            format!("targeted {times} times SO FAR IN THIS RUN (cross-pass)")
                        };
                        series.labeled(&label, len)
                    })
                    .collect(),
            )?;
        }
        // The named remainder, in bars. Every bar a pass cannot reach is here under a cause, so
        // "coverage is not 100%" always has an itemized answer instead of an inference.
        if self.pass_remainder.iter().any(Series::measured) {
            write_chart(
                &dir,
                "pretrain_pass_remainder",
                format!("Pretrain Pass Unreachable Bars - {suffix}"),
                "record",
                "training-split bars a pass cannot target, by cause",
                ScaleKind::Linear,
                PASS_REMAINDER_CAUSES
                    .iter()
                    .zip(self.pass_remainder.iter())
                    .map(|(cause, series)| series.labeled(cause, len))
                    .collect(),
            )?;
        }

        // Mean conditioning length per stage. With stride equal to context the `j`-th target of
        // a window is predicted from `j` bars, so the mean is `(context + 1) / 2` and it differs
        // by stage. Since the stages own disjoint shares, this IS the history depth a bar's
        // partition assignment decided for it, which is why the assignment must be independent
        // of symbol, calendar position and liquidity.
        if !self.stage_conditioning.is_empty() {
            write_chart(
                &dir,
                "pretrain_stage_conditioning",
                format!("Pretrain Stage Conditioning Depth - {suffix}"),
                "record",
                "mean bars of history a target bar is predicted from",
                ScaleKind::Linear,
                self.stage_conditioning
                    .iter()
                    .enumerate()
                    .map(|(stage, series)| series.labeled(&format!("stage {stage}"), len))
                    .collect(),
            )?;
        }

        write_chart(
            &dir,
            "pretrain_crps_dof",
            format!("Pretrain CRPS per DOF - {suffix}"),
            "record",
            &format!("CRPS at the fixed {diag} context"),
            ScaleKind::Linear,
            BAR_DOF_NAMES
                .iter()
                .enumerate()
                .map(|(dof, name)| self.crps_dof[dof].labeled(name, len))
                .collect(),
        )?;

        if let Some(density) = self.pit {
            let mut series: Vec<ReportSeries> = BAR_DOF_NAMES
                .iter()
                .enumerate()
                .map(|(dof, name)| ReportSeries {
                    label: (*name).to_owned(),
                    values: density[dof].iter().map(|&v| v as f32).collect(),
                })
                .collect();
            series.push(constant_series("uniform", 1.0, PIT_HIST_BINS));
            write_chart(
                &dir,
                "pretrain_pit_hist",
                format!("Pretrain PIT Histogram - {suffix}"),
                "PIT bin",
                &format!("density at the fixed {diag} context (1.0 == calibrated)"),
                ScaleKind::Linear,
                series,
            )?;
        }

        write_chart(
            &dir,
            "pretrain_dyn_loss",
            format!("Pretrain Dynamics Loss - {suffix}"),
            "record",
            "smooth L1 to the stop-grad belief",
            ScaleKind::Linear,
            vec![self.dyn_loss.labeled("train", len)],
        )?;

        write_chart(
            &dir,
            "pretrain_kl_loss",
            format!("Pretrain Dynamics KL - {suffix}"),
            "record",
            "nats/bar",
            ScaleKind::Linear,
            vec![self.kl_loss.labeled("train", len)],
        )?;

        write_chart(
            &dir,
            "pretrain_total_loss",
            format!("Pretrain Total Loss - {suffix}"),
            "record",
            "loss",
            ScaleKind::Linear,
            vec![self.total_loss.labeled("train", len)],
        )?;

        // What each term is actually WORTH in the objective. The absolute curves above
        // cannot show a term taking over: `dyn` rising 20x while `nll` drifts up looks like
        // two unrelated curves, and at a weight of 1.0 it was 62% of the loss.
        write_chart(
            &dir,
            "pretrain_loss_shares",
            format!("Pretrain Loss Term Shares - {suffix}"),
            "record",
            "weighted term / sum of weighted magnitudes (0.25 = the warning threshold)",
            ScaleKind::Linear,
            vec![
                self.nll_share.labeled("nll", len),
                self.dyn_share.labeled("dyn", len),
                self.kl_share.labeled("kl", len),
                self.growth_share.labeled("growth", len),
                constant_series("aux warning threshold", AUX_SHARE_WARN, len),
            ],
        )?;

        // The traded term, on its own panel because its scale is nothing like the others':
        // the whole tradeable content of the `r` prediction is 5.25e-4 nats/bar, so on the
        // shares chart above it is a flat line at zero and on any nats axis shared with
        // `nll` it is invisible. Four series, because four different things can go wrong.
        //
        // `growth` is the realized log growth of the DEPLOYED policy: negative is good, and
        // 0 is "took no position, or took one that exactly broke even". `share` is what it
        // is worth in the objective, which is tiny by construction — its WEIGHT was sized on
        // gradient norm, and the run prints that measurement separately. `mean |f_hat|` and
        // `cap binds` are the two that answer whether the term is doing anything: on the run
        // that motivated it, `|f*|` median rose 9.22 -> 10.69 and cap saturation 78% -> 86%
        // while the realized hit rate FELL, so a healthy run is one where those two stop
        // climbing.
        write_chart(
            &dir,
            "pretrain_growth_term",
            format!("Pretrain Expected-Log-Growth Term - {suffix}"),
            "record",
            "nats/bar (growth), fraction (share, cap binds), leverage (mean |f_hat|)",
            ScaleKind::Linear,
            vec![
                self.growth_loss.labeled("growth nats/bar", len),
                self.growth_share.labeled("objective share", len),
                self.growth_abs_f.labeled("mean |f_hat|", len),
                self.growth_clamp_bind.labeled("cap binds", len),
            ],
        )?;

        // The collapse direction the NextLat term has and `rms_norm` does not stop: nothing
        // prevents the trunk from making `h_{t+1} ~ h_t`, which the zero-init identity
        // dynamics predicts perfectly. That reduces `dyn` by destroying the belief
        // trajectory's temporal resolution rather than by learning any dynamics.
        write_chart(
            &dir,
            "pretrain_belief_autocorr",
            format!("Pretrain Lag-1 Belief Autocorrelation - {suffix}"),
            "record",
            "mean cos(h_t, h_t+1) over the batch (1.0 = the belief trajectory is frozen)",
            ScaleKind::Linear,
            vec![
                self.belief_autocorr.labeled("train", len),
                constant_series("frozen trajectory", 1.0, len),
            ],
        )?;

        // `dyn` against what it would be if the dynamics MLP returned its input unchanged.
        // At 1.0 the MLP contributes nothing and the term is a pure smoothness penalty on
        // the trunk; below 1.0 it is predicting something the identity cannot.
        write_chart(
            &dir,
            "pretrain_dyn_vs_identity",
            format!("Pretrain NextLat vs Identity Baseline - {suffix}"),
            "record",
            "dyn / smooth_l1(h_t, sg[h_t+k]) (1.0 = the dynamics MLP does nothing)",
            ScaleKind::Linear,
            vec![
                self.dyn_vs_identity.labeled("train", len),
                constant_series("MLP contributes nothing", 1.0, len),
            ],
        )?;

        let mut rollout = Vec::with_capacity(2 * ROLLOUT_HORIZONS.len());
        for (i, horizon) in ROLLOUT_HORIZONS.iter().enumerate() {
            rollout.push(self.rollout_exact[i].labeled(&format!("h{horizon} exact"), len));
        }
        for (i, horizon) in ROLLOUT_HORIZONS.iter().enumerate() {
            rollout.push(self.rollout_dynamics[i].labeled(&format!("h{horizon} dynamics"), len));
        }
        write_chart(
            &dir,
            "pretrain_rollout_nll",
            format!("Pretrain Rollout NLL by Horizon - {suffix}"),
            "record",
            "nats/bar (exact vs dynamics belief advance)",
            ScaleKind::Linear,
            rollout,
        )?;

        write_chart(
            &dir,
            "pretrain_dir_acc",
            format!("Pretrain Return Sign Accuracy - {suffix}"),
            "record",
            &format!("sign-agreement fraction at the fixed {diag} context"),
            ScaleKind::Linear,
            vec![
                self.dir_acc.labeled("val diag", len),
                constant_series("coin flip", 0.5, len),
            ],
        )?;

        write_chart(
            &dir,
            "pretrain_lr",
            format!("Pretrain LR Multiplier - {suffix}"),
            "record",
            "global lr multiplier",
            ScaleKind::Linear,
            vec![self.lr.labeled("lr", len)],
        )?;

        write_chart(
            &dir,
            "pretrain_muon_momentum",
            format!("Pretrain Muon Momentum - {suffix}"),
            "record",
            "momentum",
            ScaleKind::Linear,
            vec![self.muon_momentum.labeled("momentum", len)],
        )?;

        write_chart(
            &dir,
            "pretrain_grad_norm",
            format!("Pretrain Gradient Norm - {suffix}"),
            "record",
            "observed L2 norm (never clipped)",
            ScaleKind::Symlog,
            vec![self.grad_norm.labeled("grad norm", len)],
        )?;

        write_chart(
            &dir,
            "pretrain_unique_bar_reuse",
            format!("Pretrain Unique Bar Reuse - {suffix}"),
            "record",
            "bars seen / unique corpus bars",
            ScaleKind::Linear,
            vec![self.unique_bar_reuse.labeled("reuse", len)],
        )?;

        write_chart(
            &dir,
            "pretrain_effective_rank",
            format!("Pretrain Belief Effective Rank - {suffix}"),
            "record",
            "participation ratio (diagnostic only)",
            ScaleKind::Linear,
            vec![self.effective_rank.labeled("effective rank", len)],
        )?;

        // The promotion LEDGER, not a step count. Selection is on the 0.25x-cap trade edge and
        // the density is the guard, so a reader has to see both criteria, both incumbents, the
        // noise band the gain had to clear and the tolerance the guard allowed — on one panel,
        // in the units the decision was taken in. Cumulative refusal counts by REASON sit
        // beside the promotion count: a rule whose refusals are invisible cannot be audited,
        // and "refused inside the noise band" and "refused because the density regressed" are
        // different findings that a single "did not promote" would merge.
        write_chart(
            &dir,
            "pretrain_promotions",
            format!(
                "Pretrain Promotion Ledger (economic criterion at the {SELECTION_CAP:.2}x cap, \
                 density as the guard) - {suffix}"
            ),
            "record",
            "bps/bar, nats/bar, cumulative decisions",
            ScaleKind::Symlog,
            vec![
                self.promotion_trace.labeled("promotions", len),
                self.refused_noise_trace
                    .labeled("refused: inside the noise band", len),
                self.refused_nll_trace
                    .labeled("refused: conditional nll guard", len),
                self.refused_dof_trace.labeled("refused: r guard", len),
                self.unmeasurable_trace
                    .labeled("no comparable bench vector", len),
                self.selection_edge
                    .labeled("edge @0.25x cap, bps/bar", len),
                self.selection_edge_incumbent
                    .labeled("incumbent edge, bps/bar", len),
                self.selection_edge_gain
                    .labeled("paired edge gain, bps/bar", len),
                self.selection_edge_band
                    .labeled("noise band the gain must clear, bps/bar", len),
                self.selection_turnover.labeled(
                    &format!("turnover/bar @{SELECTION_CAP:.2}x cap (absolute weight units)"),
                    len,
                ),
                self.selection_rotations.labeled(
                    &format!(
                        "rotations/bar @{SELECTION_CAP:.2}x cap (1.0 = one full rotation, \
                         against MEASURED gross exposure)"
                    ),
                    len,
                ),
                self.selection_nll.labeled("conditional nll", len),
                self.selection_nll_incumbent
                    .labeled("incumbent conditional nll", len),
                self.selection_nll_delta
                    .labeled("paired nll delta (+ = worse)", len),
                self.selection_nll_tolerance
                    .labeled("nll tolerance the guard allows", len),
                self.selection_dof_delta
                    .labeled("paired r delta (+ = worse)", len),
            ],
        )?;

        write_chart(
            &dir,
            "pretrain_schedule",
            format!("Pretrain Ramp Schedule - {suffix}"),
            "record",
            "context / batch / bars seen (M)",
            ScaleKind::Symlog,
            vec![
                self.context.labeled("context", len),
                self.batch_size.labeled("batch", len),
                self.bars_seen.labeled("bars seen (M)", len),
            ],
        )?;

        // Market-channel coverage. Three conditioning channels carry the common factor, joined
        // to each bar on exact timestamp equality against the proxy's own bar, and a bar the
        // proxy never printed takes a reserved MISSING row. Nothing in any loss distinguishes a
        // channel that did not help from a channel that was never populated, so the share is
        // charted directly. The per-tick curve shows coverage moving with the ramp's mix of
        // extended-hours bars; the run curve is the number to quote when reading an ablation.
        write_chart(
            &dir,
            "pretrain_market_coverage",
            format!("Pretrain Market Channel Coverage - {suffix}"),
            "record",
            "% of bars with an observed market proxy bar",
            ScaleKind::Linear,
            vec![
                self.market_observed_pct.labeled("observed this tick (%)", len),
                self.market_observed_run_pct.labeled("observed, run to date (%)", len),
            ],
        )?;

        // The capacity panel. Five curves on one symlog axis because the question they answer
        // is a single one: how close did the plan run to the wall, and did the wall move?
        // `projected footprint` is what the startup capacity model said each step would cost,
        // `ceiling` is the free VRAM that model was derived against less the shared-card
        // reserve, and `free VRAM` is what the card actually had. Footprint crossing the
        // ceiling, or free VRAM sagging toward it, is a contention event; a flat realized
        // batch beside a rising context is the batch/context tradeoff being paid.
        write_chart(
            &dir,
            "pretrain_capacity",
            format!("Pretrain Device Capacity - {suffix}"),
            "record",
            "GiB / windows / bar-tokens (k)",
            ScaleKind::Symlog,
            vec![
                self.free_vram_gib.labeled("free VRAM (GiB)", len),
                self.projected_footprint_gib
                    .labeled("projected footprint (GiB)", len),
                self.capacity_ceiling_gib.labeled("ceiling (GiB)", len),
                self.batch_size.labeled("realized batch (windows)", len),
                self.bar_tokens.labeled("bar-tokens/step (k)", len),
            ],
        )?;

        // The snapshot scalars, each beside the noise of its own estimator.
        //
        // What used to sit here was a median-path-vs-realized MSE, which asserts that a
        // fan centre should track one realization; it cannot, by construction, and the
        // number it produced was read as the model being broken. It is gone. What
        // replaces it is the rank of the realized close inside the ancestral sample,
        // which is the calibration statement the snapshot should always have been making.
        write_chart(
            &dir,
            "pretrain_candle_rollout_dclose",
            format!("Pretrain Candle Rollout Fan-Centre Drift - {suffix}"),
            "snapshot",
            "mean per-bar log increment of the fan centre",
            ScaleKind::Linear,
            vec![
                ReportSeries {
                    label: "dclose".to_owned(),
                    values: self.candle_dclose.clone(),
                },
                ReportSeries {
                    label: "dclose +1 se (across windows)".to_owned(),
                    values: sum_series(&self.candle_dclose, &self.candle_dclose_se, 1.0),
                },
                ReportSeries {
                    label: "dclose -1 se (across windows)".to_owned(),
                    values: sum_series(&self.candle_dclose, &self.candle_dclose_se, -1.0),
                },
                // BOTH signs. The analytic median drift of a martingale is -sigma^2/2 < 0,
                // which is the sign this statistic is expected to take, so a one-sided
                // floor would sit on the far side of zero from every real measurement and
                // bound nothing.
                ReportSeries {
                    label: "median-estimator noise floor (+1 se)".to_owned(),
                    values: self.candle_dclose_mc_floor.clone(),
                },
                ReportSeries {
                    label: "median-estimator noise floor (-1 se)".to_owned(),
                    values: offset_series(&self.candle_dclose_mc_floor, 0.0, -1.0),
                },
                constant_series("zero", 0.0, self.candle_dclose.len()),
            ],
        )?;

        write_chart(
            &dir,
            "pretrain_candle_rollout_band",
            format!("Pretrain Candle Rollout Band Width - {suffix}"),
            "snapshot",
            "mean ln(p90 / p10)",
            ScaleKind::Linear,
            vec![ReportSeries {
                label: "band".to_owned(),
                values: self.candle_band.clone(),
            }],
        )?;

        // Measured at a FIXED horizon across windows, never pooled along the paths: the
        // closes of one chained path are nearly perfectly dependent, so pooling turns
        // `windows` draws into `windows * steps` apparent ones and shrinks the stated
        // error by an order of magnitude that does not exist.
        write_chart(
            &dir,
            "pretrain_candle_rollout_coverage",
            format!("Pretrain Candle Rollout Coverage - {suffix}"),
            "snapshot",
            "windows whose realized close fell inside the 10/90 band",
            ScaleKind::Linear,
            vec![
                ReportSeries {
                    label: "coverage h1".to_owned(),
                    values: self.candle_coverage_first.clone(),
                },
                ReportSeries {
                    label: "coverage terminal".to_owned(),
                    values: self.candle_coverage_terminal.clone(),
                },
                constant_series(
                    "nominal",
                    NOMINAL_COVERAGE,
                    self.candle_coverage_first.len(),
                ),
                // Clamped into [0, 1] and labelled with the window count: a rate cannot
                // exceed one, and at the default eight windows this Gaussian band is a
                // coarse read on Binomial(8, 0.8) — the -1 se line is crossed about 20% of
                // the time under perfect calibration rather than the 16% the label
                // suggests, so the count is the reader's warning that the band is
                // approximate.
                ReportSeries {
                    label: "nominal +1 se (iid across windows)".to_owned(),
                    values: clamped_unit(offset_series(
                        &self.candle_coverage_se,
                        NOMINAL_COVERAGE,
                        1.0,
                    )),
                },
                ReportSeries {
                    label: "nominal -1 se (iid across windows)".to_owned(),
                    values: clamped_unit(offset_series(
                        &self.candle_coverage_se,
                        NOMINAL_COVERAGE,
                        -1.0,
                    )),
                },
            ],
        )?;

        write_chart(
            &dir,
            "pretrain_candle_rollout_pit",
            format!("Pretrain Candle Rollout Rank PIT - {suffix}"),
            "snapshot",
            "rank of the realized close among the ancestral draws",
            ScaleKind::Linear,
            vec![
                ReportSeries {
                    label: "rank h1".to_owned(),
                    values: self.candle_rank_first.clone(),
                },
                ReportSeries {
                    label: "rank terminal".to_owned(),
                    values: self.candle_rank_terminal.clone(),
                },
                constant_series("uniform", 0.5, self.candle_rank_first.len()),
                ReportSeries {
                    label: "uniform +1 se".to_owned(),
                    values: offset_series(&self.candle_rank_se, 0.5, 1.0),
                },
                ReportSeries {
                    label: "uniform -1 se".to_owned(),
                    values: offset_series(&self.candle_rank_se, 0.5, -1.0),
                },
            ],
        )?;

        self.write_trade_charts(&dir, &suffix, len)?;
        self.write_epoch_charts(&dir, &suffix)?;

        Ok(())
    }

    /// The trading bench panel: eight charts, one unit each.
    ///
    /// Split by UNIT rather than crammed together: growth in basis points per bar, the
    /// paired edge with its bootstrap band in the same unit, the edge against cost on the
    /// COST axis, the same verdict against the LEVERAGE CAP, the distribution of the
    /// uncapped optimum, far-tail calibration of the traded law, annualized Sharpe, and the
    /// dimensionless exposure diagnostics. A reader asking "is this predictor making money
    /// against the unconditional null, and at what cost does that stop being true" can
    /// answer it from the second and third alone; the cap and tail charts are what say
    /// whether the answer is a property of the model or of the leverage it was handed.
    fn write_trade_charts(&self, dir: &Path, suffix: &str, len: usize) -> Result<()> {
        let windows = self
            .trade_val
            .map_or_else(|| "unmeasured".to_owned(), |t| format!("{} windows", t.windows));
        let cap = self.trade_val.map_or(f64::NAN, |t| t.leverage_cap);
        let cost = self.trade_val.map_or(f64::NAN, |t| t.cost_bps);

        let mut growth: Vec<ReportSeries> = POLICY_NAMES
            .iter()
            .enumerate()
            .map(|(policy, name)| self.trade_growth[policy].labeled(name, len))
            .collect();
        growth.push(self.trade_gross[POLICY_MODEL].labeled("model gross", len));
        growth.push(self.trade_gross[POLICY_MARGINAL].labeled("marginal gross", len));
        growth.push(constant_series("break even", 0.0, len));
        write_chart(
            dir,
            "pretrain_trade_growth",
            format!(
                "Pretrain Kelly Trade Growth ({windows}, cap {cap:.1}x, cost {cost:.2} bps) \
                 - {suffix}"
            ),
            "record",
            "realized log growth, bps/bar",
            ScaleKind::Symlog,
            growth,
        )?;

        // The only chart that answers the question. Everything above is a level; this is
        // the paired difference against the null, with the interval that says whether it is
        // resolvable at all.
        write_chart(
            dir,
            "pretrain_trade_vs_baselines",
            format!(
                "Pretrain Kelly Edge over the Unconditional Null (break-even {}) - {suffix}",
                self.trade_val
                    .map_or_else(|| "unmeasured".to_owned(), |t| break_even_label(&t)),
            ),
            "record",
            "net growth minus the marginal null, bps/bar",
            ScaleKind::Symlog,
            vec![
                self.trade_edge.labeled("edge vs marginal", len),
                self.trade_edge_low.labeled("edge ci95 low", len),
                self.trade_edge_high.labeled("edge ci95 high", len),
                self.trade_oracle_edge
                    .labeled("perfect-foresight ceiling", len),
                self.trade_capture.labeled("share of ceiling captured", len),
                constant_series("no edge", 0.0, len),
            ],
        )?;

        if let Some(val) = self.trade_val {
            // A different x axis from every other chart here: the index runs over
            // COST_GRID_BPS, and one curve per Kelly fraction is drawn on it. The TEST split
            // contributes the model's curve only: it is one number, and overlaying four more
            // one-off curves on top of the validation family would bury it.
            let mut series = cost_curve_series(&val, "val");
            if let Some(test) = self.trade_test {
                series.push(ReportSeries {
                    label: "TEST edge".to_owned(),
                    values: test
                        .model_cost_curve()
                        .iter()
                        .map(|edge| (edge * 1e4) as f32)
                        .collect(),
                });
            }
            write_chart(
                dir,
                "pretrain_trade_cost_curve",
                format!(
                    "Pretrain Kelly Edge vs Transaction Cost (val break-even {}{}) - {suffix}",
                    break_even_label(&val),
                    self.trade_test
                        .map_or_else(String::new, |t| format!(
                            ", TEST break-even {}",
                            break_even_label(&t)
                        )),
                ),
                "cost grid index (see the `cost (bps)` series)",
                "net growth minus the marginal null, bps/bar",
                ScaleKind::Symlog,
                series,
            )?;
        }

        write_chart(
            dir,
            "pretrain_trade_sharpe",
            format!("Pretrain Kelly Trade Sharpe ({windows}) - {suffix}"),
            "record",
            format!("annualized Sharpe at {BARS_PER_YEAR:.0} bars/year").as_str(),
            ScaleKind::Linear,
            POLICY_NAMES
                .iter()
                .enumerate()
                .map(|(policy, name)| self.trade_sharpe[policy].labeled(name, len))
                .collect(),
        )?;

        // Everything that says HOW the model traded, which is what decides whether the
        // break-even cost is 0.2 bps or 20: a policy that flips a full unit of notional
        // every bar cannot survive any realistic cost, however good its edge.
        write_chart(
            dir,
            "pretrain_trade_exposure",
            format!("Pretrain Kelly Trade Exposure (cap {cap:.1}x) - {suffix}"),
            "record",
            "fraction / notional per bar",
            ScaleKind::Linear,
            vec![
                self.trade_hit_rate[POLICY_MODEL].labeled("hit rate", len),
                self.trade_hit_rate[POLICY_MARGINAL].labeled("hit rate, marginal", len),
                self.trade_time_in_market.labeled("time in market", len),
                self.trade_abs_position.labeled("mean |f|", len),
                self.trade_turnover[POLICY_MODEL].labeled("turnover/bar", len),
                self.trade_turnover[POLICY_MARGINAL]
                    .labeled("turnover/bar, marginal", len),
                self.trade_drawdown_mean.labeled("mean drawdown", len),
                self.trade_drawdown_max.labeled("max drawdown", len),
                constant_series("coin flip", 0.5, len),
            ],
        )?;
        self.write_cap_and_tail_charts(dir, suffix)?;
        Ok(())
    }

    /// The three charts whose x-axis is neither the record tick nor cost: the cap grid and
    /// the two calibration panels. All three are whole-object charts of the LATEST measured
    /// bench rather than time series, for the same reason the cost curve is, and they are
    /// written by the same function the standalone command uses so the two paths cannot
    /// drift into different pictures of the same object.
    fn write_cap_and_tail_charts(&self, dir: &Path, suffix: &str) -> Result<()> {
        let Some(val) = self.trade_val else {
            return Ok(());
        };
        write_cap_and_tail_charts(dir, suffix, &val, self.trade_test.as_ref())
    }

    /// The EPOCH-INDEXED panel: three charts whose x-axis is the epoch, not the record
    /// tick.
    ///
    /// The tick charts already carry every one of these quantities, densely, and that is
    /// exactly why these exist. At a hundred-plus validations per run the tick trade curve
    /// is a noise band with a trend somewhere inside it; the question "did this pass over
    /// the corpus buy anything" has one point per pass and is answered by looking at four
    /// or five of them. Different bases, different files, and the tick series are
    /// untouched.
    fn write_epoch_charts(&self, dir: &Path, suffix: &str) -> Result<()> {
        let rows = &self.epoch_rows;
        if rows.is_empty() {
            return Ok(());
        }
        let len = rows.len();
        let series = |label: &str, project: &dyn Fn(&EpochBoundary) -> f64| ReportSeries {
            label: label.to_owned(),
            values: rows.iter().map(|row| project(row) as f32).collect(),
        };
        let last = rows.last().expect("a non-empty row set has a last row");

        // 1. THE HEADLINE. The paired edge over the unconditional null with the interval
        //    that says whether it is resolvable at all, and the cost at which it stops
        //    existing. Everything else in this panel is context for these two.
        write_chart(
            dir,
            "pretrain_epoch_trade_edge",
            format!(
                "Pretrain PER-EPOCH Kelly Edge over the Unconditional Null (epoch {}: {}, \
                 break-even {}, {:.0}% of bars at the {:.1}x cap) - {suffix}",
                last.epoch,
                if last.trade.model_edge().ci_low > 0.0 {
                    "resolvable"
                } else {
                    "NOT distinguishable from the null"
                },
                break_even_label(&last.trade),
                100.0 * last.trade.policies[POLICY_MODEL].clamped_fraction,
                last.trade.leverage_cap,
            ),
            "epoch boundary",
            "bps: edge in log growth per bar, break-even in round-trip cost",
            ScaleKind::Symlog,
            vec![
                series("EDGE vs marginal null (bps/bar)", &|row| {
                    row.trade.model_edge().mean * 1e4
                }),
                series("edge ci95 low", &|row| row.trade.model_edge().ci_low * 1e4),
                series("edge ci95 high", &|row| row.trade.model_edge().ci_high * 1e4),
                series("BREAK-EVEN cost (bps)", &|row| row.trade.model_break_even()),
                series("share of oracle ceiling captured", &|row| {
                    row.trade.model_capture()
                }),
                // Drawn on the headline too, not only on the growth panel: an edge that
                // grows while this line walks toward 1.0 is the cap deciding, not the
                // predictor improving, and the two must be visible in one glance.
                series("share of bars at the leverage cap", &|row| {
                    row.trade.policies[POLICY_MODEL].clamped_fraction
                }),
                constant_series("no edge", 0.0, len),
            ],
        )?;

        // 2. The levels the edge is a difference of, WITH the exposure that produced them.
        //
        //    Four policies, identical windows, identical solver, identical costs — the null
        //    and the ceiling are what make the model's own number mean anything. The
        //    exposure series are on the same picture rather than a panel away, because the
        //    measured run makes the failure mode concrete: the promoted checkpoint posts
        //    +4.69 bps/bar of resolvable edge while sitting at |f| 3.69 with 85% of bars
        //    PINNED AT THE 4x CAP, 3.5 of notional turned over per bar, and one window
        //    wiped out entirely. That growth is substantially a property of the cap, and a
        //    per-epoch curve that showed it rising on its own would read as the predictor
        //    improving when what improved was its willingness to be clamped. A reader must
        //    not be able to see one without the other.
        let model = |project: fn(&PolicyStats) -> f64| {
            move |row: &EpochBoundary| project(&row.trade.policies[POLICY_MODEL])
        };
        let mut growth: Vec<ReportSeries> = POLICY_NAMES
            .iter()
            .enumerate()
            .map(|(policy, name)| {
                series(name, &move |row: &EpochBoundary| {
                    row.trade.policies[policy].net_growth * 1e4
                })
            })
            .collect();
        growth.push(constant_series("break even", 0.0, len));
        growth.push(series(
            "SHARE OF BARS AT THE LEVERAGE CAP",
            &model(|stats| stats.clamped_fraction),
        ));
        growth.push(series("mean |f|", &model(|stats| stats.mean_abs_position)));
        growth.push(series(
            "turnover/bar",
            &model(|stats| stats.turnover),
        ));
        growth.push(series(
            "max drawdown (wealth fraction)",
            &model(|stats| stats.max_drawdown),
        ));
        // A count, not a rate, and never smoothed: one ruined bar is one window whose
        // wealth went to the floor, and it is the single most important thing a growth
        // curve can be hiding.
        growth.push(series("RUINED bars", &|row| {
            row.trade.policies[POLICY_MODEL].ruin_bars as f64
        }));
        write_chart(
            dir,
            "pretrain_epoch_trade",
            format!(
                "Pretrain PER-EPOCH Kelly Trade Growth AND EXPOSURE ({} windows, cap \
                 {:.1}x, cost {:.2} bps; epoch {}: |f| {:.2} with {:.0}% of bars AT THE \
                 CAP, {} ruined bars) - {suffix}",
                last.trade.windows,
                last.trade.leverage_cap,
                last.trade.cost_bps,
                last.epoch,
                last.trade.policies[POLICY_MODEL].mean_abs_position,
                100.0 * last.trade.policies[POLICY_MODEL].clamped_fraction,
                last.trade.policies[POLICY_MODEL].ruin_bars,
            ),
            "epoch boundary",
            "bps/bar of net log growth; exposure series are fractions, |f| is in units of \
             wealth, ruined bars is a count",
            ScaleKind::Symlog,
            growth,
        )?;

        // 3. The at-a-glance readout. Mixed units on purpose: this is the one picture that
        //    answers "what did this epoch cost and what did it buy", and splitting it by
        //    unit would defeat that. Every series names its own unit, and bar-token counts
        //    are in millions so a 368M-bar pass and a 5-nat likelihood share an axis a
        //    symlog scale can hold.
        write_chart(
            dir,
            "pretrain_epoch_progress",
            format!(
                "Pretrain PER-EPOCH Progress (epoch {}: {:.2} of a full pass, projecting \
                 {:.2} effective epochs against the {:.0} requested, boundary overhead \
                 {:.1}% of the epoch) - {suffix}",
                last.epoch,
                last.pass_fraction(),
                last.projected_epochs(),
                ratio(last.run_target_bar_tokens, last.full_pass_bar_tokens),
                100.0 * last.boundary_share(),
            ),
            "epoch boundary",
            "mixed; every series states its own unit",
            ScaleKind::Symlog,
            vec![
                // The budget. A run sized from a ramp it did not execute delivers a
                // fraction of the tokens its step count was priced for, and this is where
                // that becomes impossible to miss rather than an end-of-run footnote.
                series("epoch bar-tokens (M)", &|row| {
                    row.epoch_bar_tokens as f64 / 1e6
                }),
                series("full-pass target (M)", &|row| {
                    row.full_pass_bar_tokens as f64 / 1e6
                }),
                series("pass fraction of THIS epoch", &EpochBoundary::pass_fraction),
                series("cumulative bar-tokens (M)", &|row| {
                    row.run_bar_tokens as f64 / 1e6
                }),
                series("run bar-tokens requested (M)", &|row| {
                    row.run_target_bar_tokens as f64 / 1e6
                }),
                series("delivered / requested", &EpochBoundary::delivered_fraction),
                series(
                    "PROJECTED delivered / requested",
                    &EpochBoundary::projected_fraction,
                ),
                series("projected effective epochs", &EpochBoundary::projected_epochs),
                // The clock.
                series("epoch wall clock (min)", &|row| row.epoch_secs / 60.0),
                series("boundary overhead (min)", &|row| row.boundary_secs / 60.0),
                series("boundary overhead (fraction of epoch)", &EpochBoundary::boundary_share),
                // What the epoch bought.
                series("held-out nll (nats/bar)", &|row| row.val_nll_bar),
                series("forecast-only nll (nats/bar)", &|row| row.forecast_nll_bar),
                series("teacher-forcing inflation (nats/bar)", &|row| {
                    row.teacher_forcing_inflation
                }),
                series("dyn/identity", &|row| row.dyn_vs_identity),
                series("trade edge vs null (bps/bar)", &|row| {
                    row.trade.model_edge().mean * 1e4
                }),
            ],
        )?;
        Ok(())
    }
}

/// The cap curve and the two calibration panels of ONE measured bench.
///
/// Shared by the in-run reporter and the standalone command: unlike every other trade
/// chart these have no tick axis at all — their x-axes are the cap grid, the `|f*|` bucket
/// and the tail level — so the two callers would otherwise have written the same three
/// pictures twice and been free to disagree. `test` overlays the TEST split's cap curve
/// when one exists.
fn write_cap_and_tail_charts(
    dir: &Path,
    suffix: &str,
    val: &TradeBench,
    test: Option<&TradeBench>,
) -> Result<()> {
        // 1. The cap curve. The x axis is the cap grid index and the `cap (x)` series is
        //    the axis itself, exactly as the cost curve carries its own cost axis.
        let mut cap_series = vec![
            ReportSeries {
                label: "cap (x)".to_owned(),
                values: CAP_GRID.iter().map(|cap| *cap as f32).collect(),
            },
            ReportSeries {
                label: "val edge, bps/bar".to_owned(),
                values: val
                    .cap_curve
                    .iter()
                    .map(|point| (point.edge * 1e4) as f32)
                    .collect(),
            },
            ReportSeries {
                // Clipped, not dropped: `break_even_bps` is `inf` when cost never removes
                // the edge, the renderer filters non-finite points, and a missing point on a
                // curve reads as "never measured" rather than "never breaks even".
                //
                // Through [`charted_break_even`] rather than `min`, because `NaN.min(k) == k`:
                // a bare `min` maps an UNMEASURED break-even onto the ceiling, i.e. onto the
                // most profitable row on a chart that carries real cost reference lines. The
                // helper maps only `inf` to the ceiling and lets `NaN` stay non-finite so the
                // renderer drops it.
                label: format!("break-even cost, bps ({MAX_BREAK_EVEN_BPS:.0} = never)"),
                values: val
                    .cap_curve
                    .iter()
                    .map(|point| charted_break_even(point.break_even_bps) as f32)
                    .collect(),
            },
            ReportSeries {
                label: "share of bars at the cap".to_owned(),
                values: val
                    .cap_curve
                    .iter()
                    .map(|point| point.clamped_fraction as f32)
                    .collect(),
            },
            ReportSeries {
                label: "max drawdown".to_owned(),
                values: val
                    .cap_curve
                    .iter()
                    .map(|point| point.max_drawdown as f32)
                    .collect(),
            },
            constant_series("no edge", 0.0, CAP_GRID.len()),
        ];
        if let Some(test) = test {
            cap_series.push(ReportSeries {
                label: "TEST edge, bps/bar".to_owned(),
                values: test
                    .cap_curve
                    .iter()
                    .map(|point| (point.edge * 1e4) as f32)
                    .collect(),
            });
        }
        write_chart(
            dir,
            "pretrain_trade_cap_curve",
            format!(
                "Pretrain Kelly Edge vs the Leverage Cap (headline {:.1}x, {:.0}% of bars \
                 clipped there) - {suffix}",
                val.leverage_cap,
                100.0 * val.policies[POLICY_MODEL].clamped_fraction,
            ),
            "cap grid index (see the `cap (x)` series)",
            "bps/bar, bps, or fraction — see the series labels",
            ScaleKind::Symlog,
            cap_series,
        )?;

        // 2. The distribution of the uncapped optimum. Reading the mass at and beyond the
        //    cap is how one sees whether the reported policy is Kelly or is a constant.
        write_chart(
            dir,
            "pretrain_trade_free_kelly",
            format!(
                "Pretrain Uncapped Kelly |f*| Distribution (median {:.2}x, p95 {:.2}x, \
                 {:.0}% at the {:.1}x cap) - {suffix}",
                val.free_kelly.median,
                val.free_kelly.p95,
                100.0 * val.free_kelly.saturated,
                val.leverage_cap,
            ),
            "|f*| bucket index (see the `bucket floor (x)` series)",
            "share of traded bars",
            ScaleKind::Linear,
            vec![
                ReportSeries {
                    label: "bucket floor (x)".to_owned(),
                    values: FREE_KELLY_EDGES[..FREE_KELLY_EDGES.len() - 1]
                        .iter()
                        .map(|edge| *edge as f32)
                        .collect(),
                },
                ReportSeries {
                    label: "share of bars".to_owned(),
                    values: val
                        .free_kelly
                        .histogram
                        .iter()
                        .map(|share| *share as f32)
                        .collect(),
                },
            ],
        )?;

        // 3. Far-tail calibration, the one diagnostic the NLL provably cannot produce.
        //    Plotted as realized/promised so the honest line is a flat 1.0 and any bar above
        //    it is the traded law understating its own tail.
        let ratios = |pick: fn(&TradeBench, usize) -> f64| -> Vec<f32> {
            (0..TAIL_LEVELS.len())
                .map(|level| pick(&val, level) as f32)
                .collect()
        };
        write_chart(
            dir,
            "pretrain_trade_tail",
            format!(
                "Pretrain Traded-Law Far-Tail Calibration (worst {:.2}x promised on the {} \
                 side, {:.0}k bars) - {suffix}",
                val.tail.worst().0,
                if val.tail.worst().1 { "LOWER" } else { "upper" },
                val.tail.bars / 1000.0,
            ),
            "tail level index (see the `nominal (%)` series)",
            "realized exceedances / promised",
            ScaleKind::Symlog,
            vec![
                ReportSeries {
                    label: "nominal (%)".to_owned(),
                    values: TAIL_LEVELS.iter().map(|q| (100.0 * q) as f32).collect(),
                },
                ReportSeries {
                    label: "lower tail".to_owned(),
                    values: ratios(|t, level| t.tail.lower[level].ratio),
                },
                ReportSeries {
                    label: "lower ci95 low".to_owned(),
                    values: ratios(|t, level| {
                        t.tail.lower[level].blocked.0 / t.tail.lower[level].nominal
                    }),
                },
                ReportSeries {
                    label: "lower ci95 high".to_owned(),
                    values: ratios(|t, level| {
                        t.tail.lower[level].blocked.1 / t.tail.lower[level].nominal
                    }),
                },
                ReportSeries {
                    label: "upper tail".to_owned(),
                    values: ratios(|t, level| t.tail.upper[level].ratio),
                },
                ReportSeries {
                    label: "upper ci95 low".to_owned(),
                    values: ratios(|t, level| {
                        t.tail.upper[level].blocked.0 / t.tail.upper[level].nominal
                    }),
                },
                ReportSeries {
                    label: "upper ci95 high".to_owned(),
                    values: ratios(|t, level| {
                        t.tail.upper[level].blocked.1 / t.tail.upper[level].nominal
                    }),
                },
                constant_series("honest", 1.0, TAIL_LEVELS.len()),
                constant_series("warn", TAIL_RATIO_WARN, TAIL_LEVELS.len()),
            ],
        )?;
    Ok(())
}

/// Write the trading-bench panel for ONE measured bench, outside a training run.
///
/// The standalone bench command has no tick axis, no promotion and no checkpoint to
/// digest, so it cannot go through [`PretrainReporter`] — but it must still leave the
/// process through this module. It writes EVERY `pretrain_trade_*` base the TUI registers:
/// five as single-point charts, and the cost, cap and calibration curves through the same
/// [`write_cap_and_tail_charts`] the in-run reporter uses, since those three have their own
/// x-axes and read identically either way. Registering a base the standalone path never
/// writes is a blank panel, so the count is asserted in the tests below rather than left to
/// a comment.
pub fn write_trade_bench(dir: &Path, label: &str, trade: &TradeBench) -> Result<()> {
    ensure!(
        trade.measured(),
        "the bench traded no bars, so there is nothing to write"
    );
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let suffix = format!(
        "{label} - {} windows / {} bars / {} blocks, cap {:.1}x, {:.2} bps",
        trade.windows, trade.bars, trade.blocks, trade.leverage_cap, trade.cost_bps
    );

    let mut growth = Vec::with_capacity(2 * POLICY_COUNT);
    for (policy, name) in POLICY_NAMES.iter().enumerate() {
        growth.push(point_series(
            &format!("{name} net"),
            trade.policies[policy].net_growth * 1e4,
        ));
        growth.push(point_series(
            &format!("{name} gross"),
            trade.policies[policy].gross_growth * 1e4,
        ));
    }
    write_chart(
        dir,
        "pretrain_trade_growth",
        format!("Pretrain Kelly Trade Growth - {suffix}"),
        "single evaluation",
        "realized log growth, bps/bar",
        ScaleKind::Symlog,
        growth,
    )?;

    let mut battery = Vec::new();
    push_trade_series(&mut battery, trade);
    write_chart(
        dir,
        "pretrain_trade_vs_baselines",
        format!(
            "Pretrain Kelly Edge over the Unconditional Null (break-even {}) - {suffix}",
            break_even_label(trade)
        ),
        "single evaluation",
        "net growth minus the marginal null, bps/bar (plus the exposure scalars)",
        ScaleKind::Symlog,
        battery,
    )?;

    write_chart(
        dir,
        "pretrain_trade_cost_curve",
        format!(
            "Pretrain Kelly Edge vs Transaction Cost (break-even {}) - {suffix}",
            break_even_label(trade)
        ),
        "cost grid index (see the `cost (bps)` series)",
        "net growth minus the marginal null, bps/bar",
        ScaleKind::Symlog,
        cost_curve_series(trade, ""),
    )?;

    write_chart(
        dir,
        "pretrain_trade_sharpe",
        format!("Pretrain Kelly Trade Sharpe - {suffix}"),
        "single evaluation",
        format!("annualized Sharpe at {BARS_PER_YEAR:.0} bars/year").as_str(),
        ScaleKind::Linear,
        POLICY_NAMES
            .iter()
            .enumerate()
            .map(|(policy, name)| point_series(name, trade.policies[policy].sharpe))
            .collect(),
    )?;

    // Everything that says HOW the model traded, which is what decides whether the
    // break-even cost is 0.2 bps or 20: a policy that flips a full unit of notional every
    // bar cannot survive any realistic cost, however good its edge.
    write_chart(
        dir,
        "pretrain_trade_exposure",
        format!(
            "Pretrain Kelly Trade Exposure (cap {:.1}x) - {suffix}",
            trade.leverage_cap
        ),
        "single evaluation",
        "fraction / notional per bar",
        ScaleKind::Linear,
        vec![
            point_series("hit rate", trade.policies[POLICY_MODEL].hit_rate),
            point_series(
                "hit rate, marginal",
                trade.policies[POLICY_MARGINAL].hit_rate,
            ),
            point_series(
                "time in market",
                trade.policies[POLICY_MODEL].time_in_market,
            ),
            point_series("mean |f|", trade.policies[POLICY_MODEL].mean_abs_position),
            point_series(
                "fraction at the cap",
                trade.policies[POLICY_MODEL].clamped_fraction,
            ),
            point_series("turnover/bar", trade.policies[POLICY_MODEL].turnover),
            point_series(
                "turnover/bar, marginal",
                trade.policies[POLICY_MARGINAL].turnover,
            ),
            point_series("mean drawdown", trade.policies[POLICY_MODEL].mean_drawdown),
            point_series("max drawdown", trade.policies[POLICY_MODEL].max_drawdown),
            constant_series("coin flip", 0.5, 1),
        ],
    )?;

    // The cap curve and the two calibration panels, byte-for-byte the in-run pictures: one
    // measured bench is all they need, and there is no TEST split to overlay here.
    write_cap_and_tail_charts(dir, &suffix, trade, None)
}

// ---------------------------------------------------------------------------
// Mean calibration and the recalibrated policy
// ---------------------------------------------------------------------------

/// One checkpoint's calibration measurement, in the order the trend is charted.
#[derive(Clone, Debug)]
pub struct CalibrationPoint {
    /// Checkpoint file stem, so a reader can tell `pretrain_epoch_0_ctx2048` from
    /// `pretrain_best` without counting steps.
    pub label: String,
    /// Optimizer step the checkpoint was taken at. The x-axis of the trend, stated on the
    /// command line because the metadata sidecar does not record it.
    pub step: usize,
    pub nll_bar: f64,
    pub nll_bar_conditional: f64,
    /// Calibration measured on the TRADED windows: what the reported bench actually traded.
    pub eval: MeanCalibration,
    /// Calibration measured on the block-disjoint FIT slice: the slope the recalibrated
    /// policy was given. Charted beside the evaluation slice's so a reader can see whether the
    /// slope generalizes across blocks at all — if the two disagree wildly, the correction is
    /// not a stable property of the model and the recovered edge should not be believed.
    pub fit: MeanCalibration,
    pub trade: TradeBench,
    pub shrunk: ShrunkBench,
    /// The no-trade band swept on both fractions under both shape rules, at the headline cap.
    ///
    /// The Kelly solve is cost-BLIND, so the incumbent policy rebalances to the frictionless
    /// optimum every bar and pays for it afterwards. Under proportional costs that is not the
    /// optimal policy, and this is the axis that measures the size of the gap instead of
    /// asserting it is a conservative one. Empty when the pass did not run the sweep.
    pub bands: Vec<BandSweep>,
    /// Whether the recalibration and the band are substitutes, per band width.
    ///
    /// Both cut turnover, so their gains cannot be added; the interaction is the second
    /// difference, paired window by window. Empty when no recalibrated fraction was solved.
    pub band_overlap: Vec<BandShrinkOverlap>,
    /// Where the measured edge lives: what the model's SIGN is worth with the magnitude
    /// destroyed, what its SIZE is worth with the sign destroyed, and the traded panel's own
    /// correlations and win/loss asymmetry underneath both.
    ///
    /// A hit rate below a coin flip beside an edge whose interval excludes zero cannot be read
    /// as directional skill without this split, and every other economic number in the session
    /// is quoted against that edge.
    pub attribution: EdgeAttribution,
    /// The sign-hysteresis frontier: what suppressing reversals below a conviction margin costs
    /// in edge and buys in turnover. `None` when the windows carried no predicted mean, which
    /// is the only input the margin is compared against.
    pub hysteresis: Option<HysteresisSweep>,
    /// The same arm table measured on the FIT slice.
    ///
    /// Carried so a fit-slice arm has a slice-matched participation baseline. Scaling a fit arm
    /// against the TRADED book's participation would mix two populations - the two slices share
    /// NO name at all, intersection exactly zero - so without this there is no honest all-in
    /// comparison on the fit side.
    pub fit_attribution: Option<EdgeAttribution>,
    /// The flip margin fitted on the block-disjoint slice and scored on the traded one. One gate
    /// per conviction axis, each fitted and evaluated on its own axis's grid. Empty when either
    /// slice carried no predicted mean.
    pub gates: Vec<HysteresisOos>,
    /// The recalibration shrink crossed with sign hysteresis at the fitted margin, paired on
    /// identical blocks. `None` when no recalibrated fraction was solved, which is every pass
    /// whose shrink was not fitted on a disjoint slice first.
    ///
    /// Both levers cut the cost of the same book, so their gains cannot be added without the
    /// second difference this carries.
    pub composition: Option<HysteresisComposition>,
    pub decay: SignalDecay,
}

/// The mean-calibration trend and the recalibrated policy's cap curve.
///
/// Two bases from one measurement because they have different x-axes and neither is derivable
/// from the other: the trend is indexed by CHECKPOINT and the policy comparison by LEVERAGE
/// CAP. Both carry their own axis as an explicit series, the convention the cost curve already
/// uses, so an index maps onto a step or a cap without reading this source.
pub fn write_mean_calibration(
    dir: &Path,
    label: &str,
    points: &[CalibrationPoint],
) -> Result<()> {
    ensure!(
        !points.is_empty(),
        "the calibration experiment measured no checkpoint, so there is nothing to write"
    );
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let suffix = format!(
        "{label} - {} checkpoints, {} traded windows / {} blocks",
        points.len(),
        points[0].trade.windows,
        points[0].trade.blocks,
    );

    let series_of = |name: &str, values: Vec<f64>| ReportSeries {
        label: name.to_owned(),
        values: values.iter().map(|v| *v as f32).collect(),
    };
    let over = |pick: &dyn Fn(&CalibrationPoint) -> f64| -> Vec<f64> {
        points.iter().map(pick).collect()
    };
    // NaN when the pass did not form the decomposition, which is the absent state and not a
    // measured zero: a mass of zero would be a finding, and this is the lack of one.
    let arm = |point: &CalibrationPoint, pick: &dyn Fn(&OuterDecomposition) -> f64| -> f64 {
        point.eval.outer.as_ref().map_or(f64::NAN, pick)
    };

    // The trend. `beta` and its blocked interval are the headline; the fit slice's slope sits
    // beside them because a correction fitted on one set of blocks and applied to another is
    // only meaningful if the two agree. The economics ride along in the same picture: a slope
    // falling while the Sharpe falls is the entire claim, and splitting it across two charts
    // would leave the reader to align two x-axes by hand.
    let calibration = vec![
        series_of("step", over(&|p| p.step as f64)),
        series_of("perfect calibration", vec![1.0; points.len()]),
        series_of("beta, mean (traded)", over(&|p| p.eval.mean.beta)),
        series_of("beta ci low, mean (traded)", over(&|p| p.eval.mean.beta_ci.0)),
        series_of(
            "beta ci high, mean (traded)",
            over(&|p| p.eval.mean.beta_ci.1),
        ),
        series_of("beta, mean (fit slice)", over(&|p| p.fit.mean.beta)),
        series_of("beta, variance (traded)", over(&|p| p.eval.variance.beta)),
        series_of(
            "beta ci low, variance (traded)",
            over(&|p| p.eval.variance.beta_ci.0),
        ),
        series_of(
            "beta ci high, variance (traded)",
            over(&|p| p.eval.variance.beta_ci.1),
        ),
        series_of("alpha, mean (bps/bar)", over(&|p| p.eval.mean.alpha * 1e4)),
        series_of(
            "alpha se, mean (bps/bar)",
            over(&|p| p.eval.mean.alpha_se * 1e4),
        ),
        series_of("R^2 x 1e4, mean", over(&|p| p.eval.mean.r2 * 1e4)),
        series_of("R^2 x 1e4, variance", over(&|p| p.eval.variance.r2 * 1e4)),
        series_of(
            "sharpe, model",
            over(&|p| p.trade.policies[POLICY_MODEL].sharpe),
        ),
        series_of(
            "edge bps/bar, model",
            over(&|p| p.trade.model_edge().mean * 1e4),
        ),
        series_of("nll_bar conditional", over(&|p| p.nll_bar_conditional)),
        // The two decode arms beside the as-traded slope, which is the whole decomposition: the
        // as-traded slope is what the pipeline reads today, RE-DECODED is what it will read after
        // the fix, and ZEROED bounds the correction from above. A checkpoint whose pass did not
        // form the decomposition contributes a non-finite point, which the renderer drops, so an
        // unmeasured arm reads as absent rather than as zero.
        series_of(
            "beta, mean (re-decoded catch-alls)",
            over(&|p| arm(p, &|outer| outer.redecoded.mean.beta)),
        ),
        series_of(
            "beta, mean (zeroed catch-alls, upper bound)",
            over(&|p| arm(p, &|outer| outer.zeroed.mean.beta)),
        ),
        series_of(
            "beta, variance (re-decoded catch-alls)",
            over(&|p| arm(p, &|outer| outer.redecoded.variance.beta)),
        ),
        series_of(
            "beta, variance (zeroed catch-alls, upper bound)",
            over(&|p| arm(p, &|outer| outer.zeroed.variance.beta)),
        ),
        series_of(
            "catch-all mass, % of the law per bar",
            over(&|p| arm(p, &|outer| 100.0 * outer.mass)),
        ),
        series_of(
            "signed net catch-all mass, % per bar",
            over(&|p| arm(p, &|outer| 100.0 * outer.signed)),
        ),
    ];
    write_chart(
        dir,
        "pretrain_mean_calibration",
        format!(
            "Pretrain Mincer-Zarnowitz Calibration of the Traded Mean (perfect = 1.0) - {suffix}"
        ),
        "checkpoint index (see the `step` series)",
        "slope / intercept (bps/bar) / R^2 x 1e4 / sharpe",
        ScaleKind::Symlog,
        calibration,
    )?;

    // The recalibrated policy against the untouched one, at every cap. Everything the
    // acceptance of a sizing correction turns on: what it earns, how variable that is, what it
    // costs to trade, and how much leverage it asks for.
    let mut policy = vec![ReportSeries {
        label: "cap (x)".to_owned(),
        values: CAP_GRID.iter().map(|cap| *cap as f32).collect(),
    }];
    for point in points {
        let curve = &point.shrunk.curve;
        let tag = &point.label;
        let pick = |name: &str, extract: &dyn Fn(usize) -> f64| ReportSeries {
            label: format!("{tag} {name}"),
            values: (0..CAP_GRID.len()).map(|slot| extract(slot) as f32).collect(),
        };
        policy.push(pick("edge unshrunk (bps/bar)", &|slot| {
            curve[slot].unshrunk.edge * 1e4
        }));
        policy.push(pick("edge SHRUNK (bps/bar)", &|slot| {
            curve[slot].shrunk.edge * 1e4
        }));
        // The PAIRED gain is the only series on this chart that answers the question the
        // chart exists for. Both levels carry the market-common regime, so an eyeballed gap
        // between two edge curves is not evidence; the difference is taken window by window
        // and intervalled over the same blocks, and its band is what excludes zero or fails to.
        policy.push(pick("edge gain PAIRED (bps/bar)", &|slot| {
            curve[slot].paired.mean * 1e4
        }));
        policy.push(pick("edge gain CI low (bps/bar)", &|slot| {
            curve[slot].paired.ci_low * 1e4
        }));
        policy.push(pick("edge gain CI high (bps/bar)", &|slot| {
            curve[slot].paired.ci_high * 1e4
        }));
        policy.push(pick("sharpe unshrunk", &|slot| curve[slot].unshrunk.sharpe));
        policy.push(pick("sharpe SHRUNK", &|slot| curve[slot].shrunk.sharpe));
        policy.push(pick("break-even unshrunk (bps)", &|slot| {
            charted_break_even(curve[slot].unshrunk.break_even_bps)
        }));
        policy.push(pick("break-even SHRUNK (bps)", &|slot| {
            charted_break_even(curve[slot].shrunk.break_even_bps)
        }));
        policy.push(pick("mean |f| unshrunk", &|slot| {
            curve[slot].unshrunk.mean_abs_position
        }));
        policy.push(pick("mean |f| SHRUNK", &|slot| {
            curve[slot].shrunk.mean_abs_position
        }));
        policy.push(pick("capped share unshrunk", &|slot| {
            curve[slot].unshrunk.clamped_fraction
        }));
        policy.push(pick("capped share SHRUNK", &|slot| {
            curve[slot].shrunk.clamped_fraction
        }));
        policy.push(pick("turnover/bar unshrunk", &|slot| {
            curve[slot].unshrunk.turnover
        }));
        policy.push(pick("turnover/bar SHRUNK", &|slot| {
            curve[slot].shrunk.turnover
        }));
        policy.push(pick("max drawdown unshrunk", &|slot| {
            curve[slot].unshrunk.max_drawdown
        }));
        policy.push(pick("max drawdown SHRUNK", &|slot| {
            curve[slot].shrunk.max_drawdown
        }));
    }
    write_chart(
        dir,
        "pretrain_shrunk_policy",
        format!(
            "Pretrain Post-hoc Mean Shrinkage vs the Untouched Kelly Policy (slope fitted OUT \
             OF SAMPLE) - {suffix}"
        ),
        "leverage cap index (see the `cap (x)` series)",
        "bps/bar / sharpe / fraction / notional per bar",
        ScaleKind::Symlog,
        policy,
    )?;

    write_no_trade_band(dir, &suffix, points)?;
    write_edge_attribution(dir, &suffix, points)?;
    write_edge_panel(dir, &suffix, points)?;
    write_edge_confidence(dir, &suffix, points)?;
    write_edge_hysteresis(dir, &suffix, points)?;
    write_edge_composition(dir, &suffix, points)?;
    write_signal_decay(dir, &suffix, points)
}

/// The COST-AWARE sizing axis: every shape swept, with the gain over the incumbent paired.
///
/// Its own base rather than another series family on the shrinkage chart, because the x-axis
/// is KNOB INDEX and the shrinkage chart's is leverage cap: overlaying two different axes on
/// one index is exactly the confusion the explicit-axis-series convention exists to prevent.
///
/// The index is a slot rather than a value because the three shapes have different knob
/// GRIDS — band width for the two band shapes, `lambda` for partial adjustment — that agree
/// only at the two ends: slot 0 is the incumbent every-bar re-solve for all three and the
/// last slot is a frozen book for all three. Each shape's own knob values ride along as
/// their own series so no reader has to guess which grid a slot belongs to.
///
/// Three things a reader needs together, so all three are here. The break-even column rises
/// with the knob because break-even is gross edge over turnover and every shape removes
/// turnover. The PAIRED gain column is measured at ONE cost and falls again once the shape
/// starts suppressing the signal rather than the churn. And the interaction column says
/// whether the shape is buying anything the recalibration has not already bought — both
/// levers cut turnover, so a reader who adds their gains is double-counting, and only a
/// second difference can say by how much.
fn write_no_trade_band(dir: &Path, suffix: &str, points: &[CalibrationPoint]) -> Result<()> {
    if points.iter().all(|point| point.bands.is_empty()) {
        return Ok(());
    }
    let mut series = vec![ReportSeries {
        label: "knob slot".to_owned(),
        values: (0..SIZING_KNOBS).map(|slot| slot as f32).collect(),
    }];
    for shape in SIZING_SHAPES {
        series.push(ReportSeries {
            label: format!("{} knob ({})", shape.name(), shape.knob_name()),
            values: shape.knobs().iter().map(|knob| *knob as f32).collect(),
        });
    }
    for point in points {
        for sweep in &point.bands {
            let tag = format!(
                "{} {} {}",
                point.label,
                sweep.source.name(),
                sweep.shape.name()
            );
            let pick = |name: &str, extract: &dyn Fn(usize) -> f64| ReportSeries {
                label: format!("{tag} {name}"),
                values: (0..SIZING_KNOBS)
                    .map(|slot| extract(slot) as f32)
                    .collect(),
            };
            series.push(pick("edge (bps/bar)", &|slot| {
                sweep.points[slot].edge.mean * 1e4
            }));
            // The only column that answers the question. Both levels carry the same
            // market-common regime, so the difference is taken window by window and its
            // band is what excludes zero or fails to.
            series.push(pick("gain vs band 0 PAIRED (bps/bar)", &|slot| {
                sweep.points[slot].gain.mean * 1e4
            }));
            series.push(pick("gain CI low (bps/bar)", &|slot| {
                sweep.points[slot].gain.ci_low * 1e4
            }));
            series.push(pick("gain CI high (bps/bar)", &|slot| {
                sweep.points[slot].gain.ci_high * 1e4
            }));
            series.push(pick("break-even (bps)", &|slot| {
                charted_break_even(sweep.points[slot].break_even_bps)
            }));
            series.push(pick("gross growth (bps/bar)", &|slot| {
                sweep.points[slot].gross.net_growth * 1e4
            }));
            series.push(pick("sharpe net", &|slot| sweep.points[slot].policy.sharpe));
            series.push(pick("sharpe GROSS", &|slot| sweep.points[slot].gross.sharpe));
            series.push(pick("hit rate", &|slot| sweep.points[slot].policy.hit_rate));
            series.push(pick("turnover/bar", &|slot| {
                sweep.points[slot].policy.turnover
            }));
            series.push(pick("turnover share of unbanded", &|slot| {
                sweep.points[slot].turnover_share
            }));
            series.push(pick("mean |f|", &|slot| {
                sweep.points[slot].policy.mean_abs_position
            }));
            series.push(pick("cost share of gross", &|slot| {
                sweep.points[slot].cost_share_of_gross()
            }));
        }
        for shape in SIZING_SHAPES {
            let overlap: Vec<&BandShrinkOverlap> = point
                .band_overlap
                .iter()
                .filter(|row| row.shape == shape)
                .collect();
            if overlap.len() != SIZING_KNOBS {
                continue;
            }
            let tag = format!("{} overlap {}", point.label, shape.name());
            let pick = |name: &str, extract: &dyn Fn(&BandShrinkOverlap) -> f64| ReportSeries {
                label: format!("{tag} {name}"),
                values: overlap.iter().map(|row| extract(row) as f32).collect(),
            };
            series.push(pick("band gain, as-solved (bps/bar)", &|row| {
                row.gain_plain.mean * 1e4
            }));
            series.push(pick("band gain, recalibrated (bps/bar)", &|row| {
                row.gain_shrunk.mean * 1e4
            }));
            series.push(pick("INTERACTION (bps/bar)", &|row| {
                row.interaction.mean * 1e4
            }));
            series.push(pick("interaction CI low (bps/bar)", &|row| {
                row.interaction.ci_low * 1e4
            }));
            series.push(pick("interaction CI high (bps/bar)", &|row| {
                row.interaction.ci_high * 1e4
            }));
        }
    }
    write_chart(
        dir,
        "pretrain_no_trade_band",
        format!(
            "Pretrain Cost-Aware Sizing: the No-Trade Band the Cost-Blind Kelly Solve Does Not \
             Have (cap {LEVERAGE_CAP:.1}x) - {suffix}"
        ),
        "no-trade band index (see the `band` series)",
        "bps/bar / bps / sharpe / fraction / notional per bar",
        ScaleKind::Symlog,
        series,
    )
}

/// WHERE the edge lives: the five-way arm table, indexed by arm.
///
/// Its own base rather than another family on the shrinkage chart, because the x-axis is the
/// ARM — which half of the decision survived — and not a leverage cap. Every arm carries its
/// own paired interval against the null AND against the actual policy, because the levels
/// alone cannot say what destroying the magnitude cost: both sides trade the same bars of the
/// same months, so an eyeballed gap between two edge levels is the market-common regime and
/// not a finding.
fn write_edge_attribution(dir: &Path, suffix: &str, points: &[CalibrationPoint]) -> Result<()> {
    if points.iter().all(|point| !point.attribution.measured()) {
        return Ok(());
    }
    let mut series = vec![ReportSeries {
        label: "arm index".to_owned(),
        values: (0..ATTRIBUTION_ARMS).map(|arm| arm as f32).collect(),
    }];
    for point in points {
        let attribution = &point.attribution;
        let tag = &point.label;
        let pick = |name: &str, extract: &dyn Fn(usize) -> f64| ReportSeries {
            label: format!("{tag} {name}"),
            values: (0..ATTRIBUTION_ARMS)
                .map(|arm| extract(arm) as f32)
                .collect(),
        };
        series.push(pick("edge vs null (bps/bar)", &|arm| {
            attribution.arms[arm].edge.mean * 1e4
        }));
        series.push(pick("edge CI low (bps/bar)", &|arm| {
            attribution.arms[arm].edge.ci_low * 1e4
        }));
        series.push(pick("edge CI high (bps/bar)", &|arm| {
            attribution.arms[arm].edge.ci_high * 1e4
        }));
        series.push(pick("PAIRED vs actual (bps/bar)", &|arm| {
            attribution.arms[arm].paired_vs_actual.mean * 1e4
        }));
        series.push(pick("paired CI low (bps/bar)", &|arm| {
            attribution.arms[arm].paired_vs_actual.ci_low * 1e4
        }));
        series.push(pick("paired CI high (bps/bar)", &|arm| {
            attribution.arms[arm].paired_vs_actual.ci_high * 1e4
        }));
        series.push(pick("break-even (bps)", &|arm| {
            charted_break_even(attribution.arms[arm].break_even_bps)
        }));
        series.push(pick("sharpe", &|arm| attribution.arms[arm].policy.sharpe));
        series.push(pick("hit rate", &|arm| attribution.arms[arm].policy.hit_rate));
        series.push(pick("turnover/bar", &|arm| {
            attribution.arms[arm].policy.turnover
        }));
        series.push(pick("mean |f|", &|arm| {
            attribution.arms[arm].policy.mean_abs_position
        }));
        series.push(ReportSeries {
            label: format!("{tag} matched leverage"),
            values: vec![attribution.matched_leverage as f32; ATTRIBUTION_ARMS],
        });
    }
    write_chart(
        dir,
        "pretrain_edge_attribution",
        format!(
            "Pretrain Edge Attribution: What the SIGN Is Worth Without the Size and the SIZE \
             Without the Sign (arms in order: {}) - {suffix}",
            ATTRIBUTION_NAMES.join(" | "),
        ),
        "arm index (see the arm order in the title)",
        "bps/bar / bps / sharpe / fraction / notional per bar",
        ScaleKind::Symlog,
        series,
    )
}

/// The traded panel underneath the arms: the two correlations and the win/loss asymmetry, one
/// point per CHECKPOINT.
///
/// A different x-axis from the arm table and not derivable from it: `corr(f, R)` and the mean
/// size of a winning bar against a losing one are properties of the panel, measured with no
/// arm anywhere in them, and they are what makes a sub-coin-flip hit rate beside a positive
/// edge arithmetically possible rather than merely asserted.
fn write_edge_panel(dir: &Path, suffix: &str, points: &[CalibrationPoint]) -> Result<()> {
    if points.iter().all(|point| !point.attribution.panel.measured()) {
        return Ok(());
    }
    let mut series = vec![ReportSeries {
        label: "step".to_owned(),
        values: points.iter().map(|point| point.step as f32).collect(),
    }];
    let over = |pick: &dyn Fn(&CalibrationPoint) -> f64| -> Vec<f32> {
        points.iter().map(|point| pick(point) as f32).collect()
    };
    for (index, label) in PANEL_LABELS.iter().enumerate() {
        series.push(ReportSeries {
            label: (*label).to_owned(),
            values: over(&|point| point.attribution.panel.scalars()[index].point),
        });
        series.push(ReportSeries {
            label: format!("{label} CI low"),
            values: over(&|point| point.attribution.panel.scalars()[index].ci.0),
        });
        series.push(ReportSeries {
            label: format!("{label} CI high"),
            values: over(&|point| point.attribution.panel.scalars()[index].ci.1),
        });
    }
    series.push(ReportSeries {
        label: "coin flip".to_owned(),
        values: vec![0.5; points.len()],
    });
    series.push(ReportSeries {
        label: "hit-rate gradient, top |f*| decile minus bottom".to_owned(),
        values: over(&|point| point.attribution.panel.confidence_hit_gradient()),
    });
    // The 2x2's main effects and interaction, in bps/bar on the same axis as the edge they
    // decompose. Charted here rather than on the arm base because they are one number per
    // checkpoint, not one per arm, and because the sum of the four IS the actual edge.
    let effect = |label: &str, extract: &dyn Fn(&EdgeAttribution) -> Dispersion| {
        [
            ReportSeries {
                label: label.to_owned(),
                values: over(&|point| extract(&point.attribution).mean * 1e4),
            },
            ReportSeries {
                label: format!("{label} CI low"),
                values: over(&|point| extract(&point.attribution).ci_low * 1e4),
            },
            ReportSeries {
                label: format!("{label} CI high"),
                values: over(&|point| extract(&point.attribution).ci_high * 1e4),
            },
        ]
    };
    series.extend(effect("SIGN effect (bps/bar)", &|a: &EdgeAttribution| a.sign_effect));
    series.extend(effect("SIZE effect (bps/bar)", &|a: &EdgeAttribution| a.size_effect));
    series.extend(effect("INTERACTION (bps/bar)", &|a: &EdgeAttribution| a.interaction));
    series.extend(effect("DRIFT corner (bps/bar)", &|a: &EdgeAttribution| {
        a.drift_edge()
    }));
    series.extend(effect("actual edge (bps/bar)", &|a: &EdgeAttribution| {
        a.arms[super::trade_bench::ATTRIBUTION_ACTUAL].edge
    }));
    write_chart(
        dir,
        "pretrain_edge_panel",
        format!(
            "Pretrain Traded Panel: corr(f, R), corr(|f|, |R|) and the Win/Loss Size Asymmetry \
             a Sub-Coin-Flip Hit Rate Requires - {suffix}"
        ),
        "checkpoint index (see the `step` series)",
        "correlation / share / bps / ratio",
        ScaleKind::Symlog,
        series,
    )
}

/// The same panel cut by the model's own confidence: hit rate and realized growth per decile
/// of the UNCAPPED `|f*|`.
///
/// The decisive picture for the direction-versus-size question. A hit rate flat at its pooled
/// value across every decile while the growth concentrates in the top ones says the sign is
/// uninformative everywhere and the size is carrying the result; a hit rate that RISES with
/// `|f*|` says the model knows where its own sign is good, which is a direction predictor with
/// heterogeneous confidence and a different object entirely.
fn write_edge_confidence(dir: &Path, suffix: &str, points: &[CalibrationPoint]) -> Result<()> {
    if points.iter().all(|point| !point.attribution.panel.measured()) {
        return Ok(());
    }
    let mut series = vec![ReportSeries {
        label: "decile index".to_owned(),
        values: (0..ATTRIBUTION_DECILES)
            .map(|decile| decile as f32)
            .collect(),
    }];
    for point in points {
        let panel = &point.attribution.panel;
        let tag = &point.label;
        for (slot, label) in CELL_LABELS.iter().enumerate() {
            series.push(ReportSeries {
                label: format!("{tag} {label}"),
                values: (0..ATTRIBUTION_DECILES)
                    .map(|decile| panel.cells()[decile][slot].point as f32)
                    .collect(),
            });
        }
        // The hit rate's own interval, because the whole question is whether a per-decile hit
        // rate is distinguishable from the pooled one at all on this many bars.
        series.push(ReportSeries {
            label: format!("{tag} hit rate CI low"),
            values: (0..ATTRIBUTION_DECILES)
                .map(|decile| panel.hit(decile).ci.0 as f32)
                .collect(),
        });
        series.push(ReportSeries {
            label: format!("{tag} hit rate CI high"),
            values: (0..ATTRIBUTION_DECILES)
                .map(|decile| panel.hit(decile).ci.1 as f32)
                .collect(),
        });
        series.push(ReportSeries {
            label: format!("{tag} |f*| decile cut"),
            values: (0..ATTRIBUTION_DECILES)
                .map(|decile| {
                    // The last decile is open-ended above, so its upper cut is not a number the
                    // split produced and is reported absent rather than as the largest `|f*|`.
                    panel
                        .cuts()
                        .get(decile)
                        .copied()
                        .unwrap_or(f64::NAN) as f32
                })
                .collect(),
        });
    }
    series.push(ReportSeries {
        label: "coin flip".to_owned(),
        values: vec![0.5; ATTRIBUTION_DECILES],
    });
    write_chart(
        dir,
        "pretrain_edge_confidence",
        format!(
            "Pretrain Hit Rate and Realized Growth by Decile of the Model's Own UNCAPPED |f*| - \
             {suffix}"
        ),
        "confidence decile index of |f*| (see the `|f*| decile cut` series)",
        "hit rate / bps per bar / notional",
        ScaleKind::Symlog,
        series,
    )
}

/// The sign-hysteresis frontier, indexed by flip margin.
///
/// The x-axis is the MARGIN rather than the checkpoint or the cap, so this is its own base
/// rather than another series family on the arm chart. Break-even is drawn against the two
/// measured cost lines the frontier has to clear, because a frontier without its thresholds
/// invites reading a rising curve as a success when it is still far below the price.
fn write_edge_hysteresis(dir: &Path, suffix: &str, points: &[CalibrationPoint]) -> Result<()> {
    if points.iter().all(|point| point.hysteresis.is_none()) {
        return Ok(());
    }
    let knobs = HYSTERESIS_MARGINS.len();
    let mut series = vec![ReportSeries {
        label: "flip margin (bps of predicted mean)".to_owned(),
        values: HYSTERESIS_MARGINS.iter().map(|m| *m as f32).collect(),
    }];
    for point in points {
        let Some(sweep) = &point.hysteresis else {
            continue;
        };
        let tag = &point.label;
        let row = |extract: &dyn Fn(&super::trade_bench::HysteresisPoint) -> f64| -> Vec<f32> {
            sweep.points.iter().map(|p| extract(p) as f32).collect()
        };
        series.push(ReportSeries {
            label: format!("{tag} break-even (bps)"),
            values: row(&|p| charted_break_even(p.break_even_bps)),
        });
        series.push(ReportSeries {
            label: format!("{tag} edge (bps/bar)"),
            values: row(&|p| p.edge.mean * 1e4),
        });
        series.push(ReportSeries {
            label: format!("{tag} vs sign-only (bps/bar)"),
            values: row(&|p| p.vs_sign_only.mean * 1e4),
        });
        series.push(ReportSeries {
            label: format!("{tag} vs sign-only CI low"),
            values: row(&|p| p.vs_sign_only.ci_low * 1e4),
        });
        series.push(ReportSeries {
            label: format!("{tag} vs sign-only CI high"),
            values: row(&|p| p.vs_sign_only.ci_high * 1e4),
        });
        series.push(ReportSeries {
            label: format!("{tag} turnover/bar"),
            values: row(&|p| p.policy.turnover),
        });
        series.push(ReportSeries {
            label: format!("{tag} mean hold (bars)"),
            values: row(&|p| p.mean_hold_bars),
        });
        series.push(ReportSeries {
            label: format!("{tag} sharpe"),
            values: row(&|p| p.policy.sharpe),
        });
        // THE OBJECTIVE, one series per measured cost weighting. Break-even rises monotonically
        // across this whole grid, so a reader who ranks rows by it selects never-trade; these
        // are the curves that peak.
        for (slot, (name, bps)) in HYSTERESIS_NET_COSTS.iter().enumerate() {
            series.push(ReportSeries {
                label: format!("{tag} NET at {name} {bps:.3} (bps/bar)"),
                values: row(&|p| p.net_at_cost[slot].mean * 1e4),
            });
        }
        // CI bands only where the verdict is actually decided - the selection cost and the two
        // measured fitted-book anchors. A band on every one of six levels would bury the means.
        for slot in [
            HYSTERESIS_SELECTION_COST,
            HYSTERESIS_NET_COSTS.len() - 2,
            HYSTERESIS_NET_COSTS.len() - 1,
        ] {
            let (name, bps) = HYSTERESIS_NET_COSTS[slot];
            series.push(ReportSeries {
                label: format!("{tag} NET at {name} {bps:.3} CI low"),
                values: row(&|p| p.net_at_cost[slot].ci_low * 1e4),
            });
            series.push(ReportSeries {
                label: format!("{tag} NET at {name} {bps:.3} CI high"),
                values: row(&|p| p.net_at_cost[slot].ci_high * 1e4),
            });
        }
        series.push(ReportSeries {
            label: format!("{tag} NET at participation-scaled all-in (bps/bar) [INFERENCE]"),
            values: row(&|p| p.net_all_in_bps),
        });
        series.push(ReportSeries {
            label: format!("{tag} all-in cost (bps) [INFERENCE]"),
            values: row(&|p| p.all_in_cost_bps),
        });
        // Exact minus the linear reconstruction a reader would compute off the printed table.
        // Charted because it is the evidence that the two are not interchangeable.
        series.push(ReportSeries {
            label: format!("{tag} linear reconstruction gap (bps/bar)"),
            values: row(&|p| {
                p.net_at_cost_pooled[HYSTERESIS_SELECTION_COST] - p.net_reconstructed_bps
            }),
        });
    }
    // The two measured costs the frontier exists to clear, drawn flat so a reader sees the
    // gap rather than having to remember two constants.
    series.push(ReportSeries {
        label: "matched deepest-decile cost (bps)".to_owned(),
        values: vec![super::horizon::MATCHED_DEEPEST_DECILE_BPS as f32; knobs],
    });
    series.push(ReportSeries {
        label: "matched measured cost (bps)".to_owned(),
        values: vec![super::horizon::MATCHED_MEASURED_BPS as f32; knobs],
    });
    write_chart(
        dir,
        "pretrain_edge_hysteresis",
        format!(
            "Pretrain Sign Hysteresis: What Holding the Model's Sign Longer Costs and Buys - \
             {suffix}"
        ),
        "flip margin in bps of predicted mean (0 IS the sign-only arm, last knob never flips)",
        "bps / bps per bar / turnover / bars",
        ScaleKind::Symlog,
        series,
    )
}

/// The recalibration shrink crossed with sign hysteresis: four cells and their second
/// difference.
///
/// Its own base because the x-axis is the CELL of a 2x2, which is neither the flip margin the
/// frontier is indexed by nor the leverage cap the shrinkage chart uses. The interaction is
/// carried as its own series rather than left to be eyeballed off four bars, because "can these
/// two gains be added" is the whole question and the answer is a number with an interval.
fn write_edge_composition(dir: &Path, suffix: &str, points: &[CalibrationPoint]) -> Result<()> {
    if points.iter().all(|point| point.composition.is_none()) {
        return Ok(());
    }
    let cells = COMPOSITION_NAMES.len();
    let mut series = vec![ReportSeries {
        label: format!(
            "cell index: {}",
            COMPOSITION_NAMES
                .iter()
                .enumerate()
                .map(|(slot, name)| format!("{slot}={name}"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        values: (0..cells).map(|slot| slot as f32).collect(),
    }];
    for point in points {
        let Some(composition) = &point.composition else {
            continue;
        };
        let tag = &point.label;
        let row = |extract: &dyn Fn(&super::trade_bench::CompositionCell) -> f64| -> Vec<f32> {
            composition
                .cells
                .iter()
                .map(|cell| extract(cell) as f32)
                .collect()
        };
        for (label, values) in [
            ("NET (bps/bar)", row(&|cell| cell.net.mean * 1e4)),
            ("NET CI low", row(&|cell| cell.net.ci_low * 1e4)),
            ("NET CI high", row(&|cell| cell.net.ci_high * 1e4)),
            (
                "break-even (bps)",
                row(&|cell| charted_break_even(cell.break_even_bps)),
            ),
            ("turnover/bar", row(&|cell| cell.policy.turnover)),
            ("mean hold (bars)", row(&|cell| cell.mean_hold_bars)),
        ] {
            series.push(ReportSeries {
                label: format!("{tag} {label}"),
                values,
            });
        }
        // Flat across the cells: a 2x2's second difference is one number, and drawing it beside
        // the cells is what stops the two single-lever gains being read as addable.
        for (label, paired) in [
            ("hysteresis alone", &composition.hysteresis_effect),
            ("shrink alone", &composition.shrink_effect),
            ("INTERACTION", &composition.interaction),
            ("both vs hysteresis alone", &composition.both_vs_hysteresis),
        ] {
            series.push(ReportSeries {
                label: format!("{tag} {label} (bps/bar)"),
                values: vec![(paired.mean * 1e4) as f32; cells],
            });
            series.push(ReportSeries {
                label: format!("{tag} {label} CI low"),
                values: vec![(paired.ci_low * 1e4) as f32; cells],
            });
            series.push(ReportSeries {
                label: format!("{tag} {label} CI high"),
                values: vec![(paired.ci_high * 1e4) as f32; cells],
            });
        }
    }
    write_chart(
        dir,
        "pretrain_edge_composition",
        format!("Pretrain Shrink x Sign Hysteresis: Do the Two Cost Levers Add - {suffix}"),
        "2x2 cell index (see the cell index series)",
        "bps / bps per bar / turnover / bars",
        ScaleKind::Symlog,
        series,
    )
}

/// The one-bar signal's directional content against the horizon it is held over.
fn write_signal_decay(dir: &Path, suffix: &str, points: &[CalibrationPoint]) -> Result<()> {
    if points.iter().all(|point| !point.decay.measured()) {
        return Ok(());
    }
    let horizons = DECAY_HORIZONS.len();
    let mut series = vec![ReportSeries {
        label: "horizon (bars)".to_owned(),
        values: DECAY_HORIZONS.iter().map(|k| *k as f32).collect(),
    }];
    for point in points {
        if !point.decay.measured() {
            continue;
        }
        let tag = &point.label;
        let row = |extract: &dyn Fn(&super::trade_bench::DecayPoint) -> f64| -> Vec<f32> {
            point.decay.points.iter().map(|p| extract(p) as f32).collect()
        };
        series.push(ReportSeries {
            label: format!("{tag} hit rate"),
            values: row(&|p| p.hit_rate.mean),
        });
        series.push(ReportSeries {
            label: format!("{tag} hit rate CI low"),
            values: row(&|p| p.hit_rate.ci_low),
        });
        series.push(ReportSeries {
            label: format!("{tag} hit rate CI high"),
            values: row(&|p| p.hit_rate.ci_high),
        });
        series.push(ReportSeries {
            label: format!("{tag} edge (bps/bar)"),
            values: row(&|p| p.edge_per_bar.mean * 1e4),
        });
        series.push(ReportSeries {
            label: format!("{tag} edge CI low"),
            values: row(&|p| p.edge_per_bar.ci_low * 1e4),
        });
        series.push(ReportSeries {
            label: format!("{tag} edge CI high"),
            values: row(&|p| p.edge_per_bar.ci_high * 1e4),
        });
        series.push(ReportSeries {
            label: format!("{tag} corr(mu_hat, forward)"),
            values: row(&|p| p.correlation),
        });
    }
    series.push(ReportSeries {
        label: "coin flip".to_owned(),
        values: vec![0.5; horizons],
    });
    write_chart(
        dir,
        "pretrain_signal_decay",
        format!(
            "Pretrain Signal Decay: the ONE-BAR Signal Scored Against k-Bar-Ahead Returns, No \
             Policy and No Cost - {suffix}"
        ),
        "holding horizon in bars",
        "hit rate / bps per bar / correlation",
        ScaleKind::Symlog,
        series,
    )
}

/// A break-even cost as the chart carries it: `INFINITY` is a real finding — cost never
/// removes the edge — and the renderer drops non-finite points, which would make it look like
/// a metric that was never measured.
fn charted_break_even(bps: f64) -> f64 {
    if bps.is_infinite() {
        MAX_BREAK_EVEN_BPS
    } else {
        bps
    }
}

/// One line per checkpoint, in the shape the decisive comparison is read in: does correcting
/// the LATE checkpoint's mean recover the EARLY checkpoint's Sharpe?
pub fn calibration_verdict_lines(points: &[CalibrationPoint]) -> Vec<String> {
    if points.is_empty() {
        return vec!["mean calibration: no checkpoint measured".to_owned()];
    }
    let mut lines = vec![format!(
        "mean calibration over {} checkpoints (slope fitted on block-disjoint held-out windows)",
        points.len()
    )];
    for point in points {
        let model = &point.trade.policies[POLICY_MODEL];
        lines.push(format!(
            "  {:<26} step {:>6}  beta {:+.4} (se {:.4}) traded / {:+.4} fit, var beta {:+.4}, \
             sharpe {:+.2} -> {:+.2}, edge {:+.4} -> {:+.4} bps/bar, turnover {:.3} -> {:.3}",
            point.label,
            point.step,
            point.eval.mean.beta,
            point.eval.mean.beta_se,
            point.fit.mean.beta,
            point.eval.variance.beta,
            model.sharpe,
            point.shrunk.policy.sharpe,
            point.trade.model_edge().mean * 1e4,
            point.shrunk.edge.mean * 1e4,
            model.turnover,
            point.shrunk.policy.turnover,
        ));
    }
    // The recovery fraction, which is the number the whole experiment exists to produce.
    let early = &points[0];
    let late = points.last().expect("non-empty");
    let early_model = &early.trade.policies[POLICY_MODEL];
    let late_model = &late.trade.policies[POLICY_MODEL];
    // The paired gain at the HEADLINE cap, which is the only interval on the comparison that
    // is not dominated by the regime both sides share.
    let headline = late
        .shrunk
        .curve
        .iter()
        .find(|point| point.cap == late.shrunk.leverage_cap)
        .copied()
        .unwrap_or_else(super::trade_bench::ShrunkPoint::nan);
    let sharpe_lost = early_model.sharpe - late_model.sharpe;
    let sharpe_recovered = late.shrunk.policy.sharpe - late_model.sharpe;
    let edge_lost = early.trade.model_edge().mean - late.trade.model_edge().mean;
    lines.push(format!(
        "  PAIRED at the {:.1}x headline cap: recalibration is worth {:+.4} bps/bar (95% CI \
         {:+.4}..{:+.4}, se {:.4}) on the late checkpoint — {}",
        late.shrunk.leverage_cap,
        headline.paired.mean * 1e4,
        headline.paired.ci_low * 1e4,
        headline.paired.ci_high * 1e4,
        headline.paired.se * 1e4,
        if headline.resolvable() {
            "resolvably different from zero"
        } else {
            "NOT resolvable; the levels below are two noisy numbers and no verdict follows"
        },
    ));
    lines.push(format!(
        "  VERDICT: {} -> {} lost {:+.2} sharpe and {:+.4} bps/bar; recalibrating the late \
         mean recovers {:+.2} sharpe ({:.0}% of the sharpe loss) and {:+.4} bps/bar ({:.0}% of \
         the edge loss){}",
        early.label,
        late.label,
        -sharpe_lost,
        -edge_lost * 1e4,
        sharpe_recovered,
        if sharpe_lost.abs() > 0.0 {
            100.0 * sharpe_recovered / sharpe_lost
        } else {
            f64::NAN
        },
        headline.paired.mean * 1e4,
        if edge_lost.abs() > 0.0 {
            100.0 * headline.paired.mean / edge_lost
        } else {
            f64::NAN
        },
        // The gate is the PAIRED interval, not the ratio: a recovery fraction computed from
        // two unresolvable levels is a number with no content, and reporting a verdict off it
        // is the exact failure this experiment was built to avoid.
        if !headline.resolvable() {
            " — INCONCLUSIVE at this sample size"
        } else if headline.paired.mean <= 0.0 {
            " — recalibration does not help; the decay is NOT miscalibration"
        } else if sharpe_lost <= 0.0 {
            " — nothing was lost to recover, but the recalibration is worth having anyway"
        } else if sharpe_recovered >= 0.9 * sharpe_lost {
            " — the decay is MISCALIBRATION, fixable without retraining"
        } else if sharpe_recovered > 0.25 * sharpe_lost {
            " — the decay is PARTLY miscalibration; the remainder is lost information"
        } else {
            " — the decay is mostly NOT miscalibration; the model lost directional information"
        },
    ));
    lines
}

// ---------------------------------------------------------------------------
// Candle snapshot pictures
// ---------------------------------------------------------------------------

/// What one window's picture depicts: the realized close path, the ancestral
/// quantile fan at [`FAN_QUANTILES`], and where the realization ranked inside the
/// ancestral sample. Everything is chained from a previous close of `1.0`, so
/// windows at different price levels are comparable.
///
/// There is deliberately no `predicted` field and no median CANDLE. The
/// coordinate-wise median over sample paths is a locus of per-horizon medians: it
/// is not a draw from the predictive law, its `open` need not equal the previous
/// bar's median `close`, and a realized path wandering away from it is the
/// widening of the fan rather than an error.
#[derive(Clone, Debug)]
pub struct CandleWindow {
    pub actual_close: Vec<f32>,
    /// `quantiles[q][t]` is [`FAN_QUANTILES`]`[q]` of the sampled close at
    /// forecast bar `t`, estimated from [`Self::samples`] ancestral draws.
    pub quantiles: [Vec<f32>; FAN_QUANTILES.len()],
    /// Rank of the realized close among the ancestral sampled closes, in `[0, 1]`,
    /// with ties split. This is the PIT of the path forecast: uniform under a
    /// calibrated predictive law, and unlike a pointwise error it says WHICH way
    /// the law is wrong when it is wrong.
    pub rank: Vec<f32>,
    /// Ancestral draws the quantiles were estimated from. Carried because every
    /// quantile's standard error scales as `1 / sqrt(samples)` and a quantile
    /// without that scale attached invites a wiggle to be read as a signal.
    pub samples: usize,
}

impl CandleWindow {
    pub fn p10(&self) -> &[f32] {
        &self.quantiles[BAND_LOW_INDEX]
    }

    pub fn p90(&self) -> &[f32] {
        &self.quantiles[BAND_HIGH_INDEX]
    }

    /// The fan CENTRE. Named for what it is: the per-horizon median locus, not a
    /// path and not a forecast.
    pub fn fan_centre(&self) -> &[f32] {
        &self.quantiles[FAN_CENTRE_INDEX]
    }

    pub fn steps(&self) -> usize {
        self.actual_close.len()
    }

    /// Standard error of the fan centre at bar `t`, in LOG units.
    ///
    /// The sample median's asymptotic standard error is `1 / (2 f(m) sqrt(n))`.
    /// Estimating `1 / f(m)` by the interquartile spacing — `1/f ~ IQR / 0.5` —
    /// gives `se ~ IQR / sqrt(n)`, which needs no density estimate and no
    /// distributional assumption, and on a Gaussian recovers the exact
    /// `1.2533 sigma / sqrt(n)` to within 8%. The inputs are p25 and p75, so the
    /// fan carries its own error bar. Dividing by the centre converts the spacing
    /// from price units to the log units the drift is measured in, which is exact
    /// to first order because the fan's relative width is small.
    ///
    /// A non-finite or non-positive fan is reported as NaN, never clamped: the
    /// reporter's own NaN filter then drops the point, which is what a reader must
    /// see. Clamping the centre at `1e-12` would turn a poisoned sample into
    /// `ln(1e-12) / steps`, a large but entirely plausible-looking drift.
    pub fn centre_log_se(&self, t: usize) -> f64 {
        let iqr = (self.quantiles[3][t] - self.quantiles[1][t]) as f64;
        let centre = self.quantiles[FAN_CENTRE_INDEX][t] as f64;
        if !iqr.is_finite() || !(centre > 0.0) || self.samples == 0 {
            return f64::NAN;
        }
        iqr / (self.samples as f64).sqrt() / centre
    }

    /// Mean per-bar log increment of the fan centre.
    ///
    /// The increments telescope, so this is `ln(centre[last]) / steps` exactly and
    /// its standard error is [`Self::centre_log_se`] at the last bar divided by
    /// `steps` — the intermediate increments carry no independent information,
    /// which is why a monotone run of them is not evidence of drift.
    pub fn drift_per_bar(&self) -> f64 {
        let steps = self.steps();
        if steps == 0 {
            return f64::NAN;
        }
        let centre = self.fan_centre()[steps - 1] as f64;
        if !(centre > 0.0) {
            return f64::NAN;
        }
        centre.ln() / steps as f64
    }

    pub fn drift_per_bar_se(&self) -> f64 {
        let steps = self.steps();
        if steps == 0 {
            return f64::NAN;
        }
        self.centre_log_se(steps - 1) / steps as f64
    }

    pub fn in_band(&self, t: usize) -> bool {
        self.actual_close[t] >= self.p10()[t] && self.actual_close[t] <= self.p90()[t]
    }

    /// Bars of THIS window whose realized close fell inside the nominal band.
    ///
    /// A COUNT, because that is what the per-window chart states. It is deliberately not
    /// turned into a rate to be compared against the nominal: the closes of one chained
    /// path are almost perfectly dependent, so a path that leaves the band tends to stay
    /// out, and one window contributes closer to one draw than to `steps`. The rate that
    /// may be read against the nominal is measured across windows at a fixed horizon by
    /// [`CandleSummary`].
    pub fn in_band_count(&self) -> usize {
        (0..self.steps()).filter(|t| self.in_band(*t)).count()
    }

    /// The same as a fraction of this window's bars, for pooling diagnostics only.
    pub fn in_band_rate(&self) -> f64 {
        let steps = self.steps();
        if steps == 0 {
            return f64::NAN;
        }
        self.in_band_count() as f64 / steps as f64
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CandleSummary {
    dclose: f64,
    /// Standard error of `dclose` ACROSS windows: the spread of the per-window
    /// drifts, which carries both the median estimator's noise and the genuine
    /// difference between one window's conditional law and another's.
    dclose_se: f64,
    /// The median ESTIMATOR's noise alone, with every window's conditional law
    /// taken as given. A `dclose` inside this is not a measurement of anything.
    dclose_mc_floor: f64,
    band: f64,
    /// In-band rate at the FIRST forecast bar, across windows.
    coverage_first: f64,
    /// The same at the LAST forecast bar.
    coverage_terminal: f64,
    /// Binomial standard error of a per-horizon rate over `windows` windows, evaluated
    /// UNDER THE NULL that the forecast is calibrated. It is drawn as a band around the
    /// nominal line, not around the observed rate, because the question the chart answers
    /// is whether the observation is compatible with nominal — the Wald se of the
    /// observation itself would collapse to zero exactly when every window is covered,
    /// which is the case a reader most needs an error bar for.
    coverage_se: f64,
    rank_first: f64,
    rank_terminal: f64,
    /// Standard error of a mean of `windows` draws from the uniform law the rank
    /// follows when the forecast is calibrated: `sqrt(1 / (12 W))`.
    rank_se: f64,
}

impl CandleSummary {
    /// Pool the per-window fans into the scalars the `pretrain_candle_rollout_*`
    /// charts trace.
    ///
    /// Every pooled figure is accompanied by the standard error of its own
    /// estimator, and every rate is measured at a FIXED horizon across windows
    /// rather than pooled along the paths, because the bars of one chained path
    /// are not independent trials. The band width is the exception: it is a
    /// property of the forecast and not an estimate of a rate, so averaging it
    /// along the path is a description rather than an inference.
    fn from_windows(windows: &[CandleWindow]) -> Self {
        let count = windows.len();
        if count == 0 || windows[0].steps() == 0 {
            return Self {
                dclose: f64::NAN,
                dclose_se: f64::NAN,
                dclose_mc_floor: f64::NAN,
                band: f64::NAN,
                coverage_first: f64::NAN,
                coverage_terminal: f64::NAN,
                coverage_se: f64::NAN,
                rank_first: f64::NAN,
                rank_terminal: f64::NAN,
                rank_se: f64::NAN,
            };
        }
        let mut band = Mean::default();
        let mut drifts = Vec::with_capacity(count);
        let mut covered_first = 0usize;
        let mut covered_terminal = 0usize;
        let mut rank_first = Mean::default();
        let mut rank_terminal = Mean::default();
        for window in windows {
            let last = window.steps() - 1;
            for t in 0..window.steps() {
                let (p10, p90) = (window.p10()[t] as f64, window.p90()[t] as f64);
                // Guarded, not clamped: `f32::max` RETURNS the non-NaN operand, so
                // `NaN.max(1e-12)` is `1e-12` and a poisoned band would read as a
                // finite ln-width of about 28 nats instead of dropping out.
                band.push(if p10 > 0.0 && p90 > 0.0 {
                    (p90 / p10).ln()
                } else {
                    f64::NAN
                });
            }
            // Kept as a PAIR. A window can have a finite drift and a non-finite error
            // bar — the terminal p50 stays inside the finite draws while p75 interpolates
            // across the finite/non-finite boundary — and averaging the two through
            // separate NaN filters would scale the noise floor by sqrt(W / (W - k))
            // against the very drift it is supposed to bound.
            drifts.push((window.drift_per_bar(), window.drift_per_bar_se().powi(2)));
            covered_first += usize::from(window.in_band(0));
            covered_terminal += usize::from(window.in_band(last));
            rank_first.push(window.rank[0] as f64);
            rank_terminal.push(window.rank[last] as f64);
        }
        // ONE sample size for the mean, the variance and both standard errors. A window
        // whose fan is degenerate contributes a non-finite drift, and mixing the
        // NaN-skipping mean with a raw-count denominator would divide the spread of the
        // windows that DID measure something by a count that includes the ones that did
        // not, understating exactly the error bar a reader checks the drift against.
        let measured: Vec<(f64, f64)> = drifts
            .into_iter()
            .filter(|(drift, floor)| drift.is_finite() && floor.is_finite())
            .collect();
        let effective = measured.len();
        let mean_drift = if effective == 0 {
            f64::NAN
        } else {
            measured.iter().map(|(drift, _)| *drift).sum::<f64>() / effective as f64
        };
        let variance = if effective > 1 {
            measured
                .iter()
                .map(|(drift, _)| (drift - mean_drift).powi(2))
                .sum::<f64>()
                / (effective - 1) as f64
        } else {
            f64::NAN
        };
        let mean_floor_square = if effective == 0 {
            f64::NAN
        } else {
            measured.iter().map(|(_, floor)| *floor).sum::<f64>() / effective as f64
        };
        let rate_se = |rate: f64| (rate * (1.0 - rate) / count as f64).sqrt();
        Self {
            dclose: mean_drift,
            dclose_se: (variance / effective as f64).sqrt(),
            dclose_mc_floor: (mean_floor_square / effective as f64).sqrt(),
            band: band.value(),
            coverage_first: covered_first as f64 / count as f64,
            coverage_terminal: covered_terminal as f64 / count as f64,
            coverage_se: rate_se(NOMINAL_COVERAGE),
            rank_first: rank_first.value(),
            rank_terminal: rank_terminal.value(),
            rank_se: (1.0 / (12.0 * count as f64)).sqrt(),
        }
    }
}

/// Write one `CandleFan` per window into `dir` — the realized bars against the
/// ancestral quantile fan, with [`SNAPSHOT_OVERLAY_PATHS`] genuine draws overlaid —
/// and return the fans they depict.
///
/// `drawn` is the `[W, samples, H, BAR_DOF]` ancestral rollout and `future_dof` the
/// `[W, H, BAR_DOF]` realized continuation of the same windows. Both are DOF paths,
/// chained here onto a common relative scale so windows at different price levels
/// are comparable.
///
/// `global_step` and `epoch` only name the files. The step comes FIRST in the tag
/// because the TUI's snapshot discovery parses it off the front of the file name to
/// keep only the newest set; the epoch follows it because these pictures are now taken
/// at every epoch boundary on one pinned scene, and the whole point of that is being
/// able to lay epoch 0's fan beside epoch 3's. `None` is for the standalone entry
/// point, which pictures a checkpoint handed to it on the command line and has no
/// epoch to honestly claim.
///
/// The chart states, in its own subtitle, the in-band rate of the realized path and
/// the standard error of the fan centre at the first and last horizon. Those two
/// numbers are what separate miscalibration from ordinary dispersion, and from the
/// noise of estimating a quantile out of `samples` draws — the reading that a bare
/// median line invited and that this picture is built to refuse.
///
/// Deliberately independent of [`PretrainReporter`]: the same pictures are produced
/// in-run by [`PretrainReporter::record_snapshot`] and standalone by
/// `pretrain-candles`, and neither may drift from the other's chaining or quantile
/// convention.
pub fn write_candle_windows(
    dir: &Path,
    global_step: usize,
    epoch: Option<usize>,
    drawn: &Tensor,
    future_dof: &Tensor,
) -> Result<Vec<CandleWindow>> {
    let (windows, steps) = match future_dof.size().as_slice() {
        [w, horizon, dof] if *dof == BAR_DOF as i64 && *w > 0 && *horizon > 0 => {
            (*w as usize, *horizon as usize)
        }
        other => anyhow::bail!("snapshot future_dof must be [W, H, {BAR_DOF}], got {other:?}"),
    };
    let samples = match drawn.size().as_slice() {
        [w, s, t, d]
            if *w == windows as i64 && *t == steps as i64 && *d == BAR_DOF as i64 && *s > 0 =>
        {
            *s as usize
        }
        other => {
            anyhow::bail!("rollout must be [{windows}, samples, {steps}, {BAR_DOF}], got {other:?}")
        }
    };
    ensure!(
        samples >= MIN_FAN_SAMPLES,
        "a quantile fan needs at least {MIN_FAN_SAMPLES} ancestral draws for p25 and p75 to be \
         distinct order statistics, got {samples}"
    );
    // `decode_dof` is deliberately total: it maps a non-finite DOF onto a flat bar and
    // clamps every price into a legal range. Right for a live inference path, wrong here,
    // because a poisoned rollout would then chart as a perfectly calm market — zero band
    // width, full coverage, rank 0.5, no drift — the most reassuring picture this file can
    // draw. Catching it before the decode launders it is also what makes the NaN guards in
    // `CandleWindow` reachable from the production path at all.
    ensure!(
        bool::try_from(drawn.isfinite().all()).unwrap_or(false),
        "the ancestral rollout went non-finite, so no fan can be estimated from it"
    );
    ensure!(
        bool::try_from(future_dof.isfinite().all()).unwrap_or(false),
        "the realized continuation went non-finite, so nothing can be ranked against the fan"
    );

    let drawn_values = tensor_values(drawn);
    let future_values = tensor_values(future_dof);
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;

    let window_stride = samples * steps * BAR_DOF;
    let sample_stride = steps * BAR_DOF;
    let mut column = vec![0.0f32; samples];
    let mut out = Vec::with_capacity(windows);

    for window in 0..windows {
        let actual = chained_candles(
            &future_values[window * steps * BAR_DOF..(window + 1) * steps * BAR_DOF],
        );
        // Closes only: the fan is over the cumulative close path, and a
        // coordinate-wise quantile of the other three fields is the very object
        // this picture exists to stop presenting as a bar.
        let paths: Vec<Vec<f32>> = (0..samples)
            .map(|sample| {
                let start = window * window_stride + sample * sample_stride;
                chained_candles(&drawn_values[start..start + sample_stride])
                    .into_iter()
                    .map(|bar| bar.close)
                    .collect()
            })
            .collect();

        let mut quantiles: [Vec<f32>; FAN_QUANTILES.len()] =
            array::from_fn(|_| Vec::with_capacity(steps));
        let mut rank = Vec::with_capacity(steps);
        for t in 0..steps {
            for (slot, path) in column.iter_mut().zip(paths.iter()) {
                *slot = path[t];
            }
            column.sort_by(|a, b| a.total_cmp(b));
            for (slot, probability) in quantiles.iter_mut().zip(FAN_QUANTILES) {
                slot.push(quantile_sorted(&column, probability));
            }
            rank.push(sample_rank(&column, actual[t].close));
        }

        let fan = CandleWindow {
            actual_close: actual.iter().map(|bar| bar.close).collect(),
            quantiles,
            rank,
            samples,
        };
        let overlay = overlay_indices(window, samples);
        let tag = match epoch {
            Some(epoch) => format!("step{global_step}_epoch{epoch:03}_window{:02}", window + 1),
            None => format!("step{global_step}_window{:02}", window + 1),
        };
        write_report_at(
            &dir.join(format!("{tag}_fan.report.bin")),
            &Report {
                // The rate is along ONE dependent path, so it is stated as a count and
                // pointedly NOT against the nominal: a calibrated path that leaves the
                // band early tends to stay out, so 5/100 is an ordinary outcome that a
                // binomial read on 100 "trials" would score as an 18-sigma miss. The
                // nominal belongs to the across-window rate, which has its own chart.
                title: format!(
                    "Pretrain Rollout Fan - step {global_step} - window {:02} - realized CLOSE \
                     inside the {:.0}/{:.0} band on {}/{steps} bars (ONE dependent path; the \
                     nominal {:.0}% is a rate ACROSS windows - see \
                     pretrain_candle_rollout_coverage) - fan-centre se {:.1}e-4 at h1, \
                     {:.1}e-4 at h{steps} (log, from {samples} draws)",
                    window + 1,
                    BAND_LOW * 100.0,
                    BAND_HIGH * 100.0,
                    fan.in_band_count(),
                    NOMINAL_COVERAGE * 100.0,
                    fan.centre_log_se(0) * 1.0e4,
                    fan.centre_log_se(steps - 1) * 1.0e4,
                ),
                x_label: Some("forecast bar".to_owned()),
                y_label: Some("relative price".to_owned()),
                scale: ScaleKind::Linear,
                kind: ReportKind::CandleFan {
                    actual: actual.clone(),
                    bands: FAN_QUANTILES
                        .iter()
                        .zip(fan.quantiles.iter())
                        .map(|(probability, closes)| QuantileBand {
                            probability: *probability,
                            closes: closes.clone(),
                        })
                        .collect(),
                    samples: overlay
                        .iter()
                        .map(|index| ReportSeries {
                            label: format!("draw {index}"),
                            values: paths[*index].clone(),
                        })
                        .collect(),
                },
            },
        )?;
        out.push(fan);
    }

    Ok(out)
}

/// Which ancestral draws a window overlays, pinned to [`EVAL_WINDOW_SEED`].
///
/// The draws themselves come from the model's own sampler and move with the
/// training seed; WHICH of them is depicted must not, or two runs' pictures differ
/// for a reason that has nothing to do with either model. A partial Fisher-Yates
/// over the campaign seed mixed with the window index gives distinct indices in
/// bounded time and the same indices in every run and every replay.
fn overlay_indices(window: usize, samples: usize) -> Vec<usize> {
    let count = SNAPSHOT_OVERLAY_PATHS.min(samples);
    let mut pool: Vec<usize> = (0..samples).collect();
    let mut state = mix64(EVAL_WINDOW_SEED, window as u64);
    for slot in 0..count {
        state = mix64(state, slot as u64);
        let pick = slot + (state % (samples - slot) as u64) as usize;
        pool.swap(slot, pick);
    }
    let mut chosen = pool[..count].to_vec();
    chosen.sort_unstable();
    chosen
}

/// Rank of `value` within `sorted`, in `[0, 1]`, with ties split.
///
/// This is the randomization-free PIT of a path forecast: `P(sample < value)`
/// plus half the tie mass, which is uniform on `[0, 1]` when `value` is a draw
/// from the same law as the samples.
fn sample_rank(sorted: &[f32], value: f32) -> f32 {
    // `partition_point` with a comparison against NaN is always false, so a non-finite
    // realization would rank 0.0 — a confident "below every draw" — instead of dropping
    // out. The writer refuses non-finite input before this is reached; the guard is here
    // so the function is honest in isolation.
    if sorted.is_empty() || !value.is_finite() {
        return f32::NAN;
    }
    let below = sorted.partition_point(|sample| *sample < value);
    let at_or_below = sorted.partition_point(|sample| *sample <= value);
    (below + at_or_below) as f32 / (2 * sorted.len()) as f32
}

/// Clamp a rate series into `[0, 1]`, leaving NaN padding alone. A rate band drawn
/// outside the unit interval depicts an outcome that cannot occur.
fn clamped_unit(mut values: Vec<f32>) -> Vec<f32> {
    for value in values.iter_mut() {
        if value.is_finite() {
            *value = value.clamp(0.0, 1.0);
        }
    }
    values
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Decode a `[T, BAR_DOF]` DOF path into a chained candle path anchored at a
/// previous close of `1.0`, so every window is on a comparable relative scale.
/// Volume is not a candle field, so the EMA reference is irrelevant here.
fn chained_candles(dof_path: &[f32]) -> Vec<CandleBar> {
    let mut previous = 1.0f32;
    dof_path
        .chunks_exact(BAR_DOF)
        .map(|row| {
            let dof = BarDof::from_array([row[0], row[1], row[2], row[3], row[4]]);
            let bar = decode_dof(previous, &dof, 1.0);
            previous = bar.close;
            CandleBar {
                open: bar.open,
                high: bar.high,
                low: bar.low,
                close: bar.close,
            }
        })
        .collect()
}

fn quantile_sorted(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let position = q * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = (position - lower as f64) as f32;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

fn tensor_values(tensor: &Tensor) -> Vec<f32> {
    let flat = tensor
        .detach()
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .contiguous()
        .view([-1]);
    let numel = flat.numel();
    let mut values = vec![0.0f32; numel];
    flat.copy_data(&mut values, numel);
    values
}

/// The cost axis plus one edge curve per KELLY FRACTION, shared by the in-run reporter and
/// the standalone bench command so the two pictures of the same object cannot drift.
///
/// Every fraction is charted, not just full Kelly: the clamp's own opinion is the ceiling of
/// what the cap permits and is routinely not the fraction one would run, so the reader has to
/// be able to see where the fraction being quoted as the headline crosses zero. `tag` names
/// the pass, and is empty on the standalone path where there is only one.
fn cost_curve_series(trade: &TradeBench, tag: &str) -> Vec<ReportSeries> {
    let mut series = vec![ReportSeries {
        // The axis itself, so a reader maps an index onto a cost without reading the source.
        label: "cost (bps)".to_owned(),
        values: COST_GRID_BPS.iter().map(|bps| *bps as f32).collect(),
    }];
    for policy in 0..POLICY_COUNT {
        if !POLICY_KELLY_MULTIPLE[policy].is_finite() {
            continue;
        }
        series.push(ReportSeries {
            label: if tag.is_empty() {
                format!("edge, {}", POLICY_NAMES[policy])
            } else {
                format!("{tag} edge, {}", POLICY_NAMES[policy])
            },
            values: trade.cost_curve[policy]
                .iter()
                .map(|edge| (edge * 1e4) as f32)
                .collect(),
        });
    }
    series.push(constant_series("no edge", 0.0, COST_GRID_BPS.len()));
    series
}

/// Every trading-bench scalar as a one-point series on the terminal battery chart.
///
/// The charts carry the validation curve; this is the split that is touched once. Growth
/// is in basis points per bar so the two are read in the same unit.
fn push_trade_series(series: &mut Vec<ReportSeries>, trade: &TradeBench) {
    for (policy, name) in POLICY_NAMES.iter().enumerate() {
        let stats = &trade.policies[policy];
        series.push(point_series(
            &format!("trade {name} growth bps/bar"),
            stats.net_growth * 1e4,
        ));
        series.push(point_series(
            &format!("trade {name} gross bps/bar"),
            stats.gross_growth * 1e4,
        ));
        series.push(point_series(&format!("trade {name} sharpe"), stats.sharpe));
        series.push(point_series(&format!("trade {name} hit rate"), stats.hit_rate));
        series.push(point_series(
            &format!("trade {name} turnover/bar"),
            stats.turnover,
        ));
        series.push(point_series(
            &format!("trade {name} time in market"),
            stats.time_in_market,
        ));
        series.push(point_series(&format!("trade {name} mean |f|"), stats.mean_abs_position));
        series.push(point_series(
            &format!("trade {name} capped fraction"),
            stats.clamped_fraction,
        ));
        series.push(point_series(
            &format!("trade {name} mean drawdown"),
            stats.mean_drawdown,
        ));
        series.push(point_series(
            &format!("trade {name} max drawdown"),
            stats.max_drawdown,
        ));
        series.push(point_series(
            &format!("trade {name} ruined bars"),
            stats.ruin_bars as f64,
        ));
    }
    series.push(point_series(
        "trade EDGE vs marginal bps/bar",
        trade.model_edge().mean * 1e4,
    ));
    series.push(point_series("trade edge se bps/bar", trade.model_edge().se * 1e4));
    series.push(point_series("trade edge ci95 low bps/bar", trade.model_edge().ci_low * 1e4));
    series.push(point_series(
        "trade edge ci95 high bps/bar",
        trade.model_edge().ci_high * 1e4,
    ));
    series.push(point_series("trade edge blocks", trade.model_edge().blocks as f64));
    series.push(point_series("trade break-even cost bps", trade.model_break_even()));
    series.push(point_series(
        "trade share of the perfect-foresight ceiling",
        trade.model_capture(),
    ));
    series.push(point_series("trade cost charged bps", trade.cost_bps));
    series.push(point_series("trade leverage cap", trade.leverage_cap));
    series.push(point_series("trade windows", trade.windows as f64));
    series.push(point_series("trade bars", trade.bars as f64));
    for (slot, bps) in COST_GRID_BPS.iter().enumerate() {
        let marker = if slot == DEFAULT_COST_SLOT {
            " (charged)"
        } else {
            ""
        };
        series.push(point_series(
            &format!("trade edge at {bps} bps{marker}"),
            trade.model_cost_curve()[slot] * 1e4,
        ));
    }
}

/// The break-even cost as a chart title states it, including the two outcomes that are not
/// numbers: no gross edge to lose, and an edge no cost can reach.
fn break_even_label(trade: &TradeBench) -> String {
    if !trade.measured() {
        "unmeasured".to_owned()
    } else if trade.model_break_even().is_nan() {
        "n/a, no gross edge".to_owned()
    } else if trade.model_break_even().is_infinite() {
        "never".to_owned()
    } else {
        format!("{:.2} bps", trade.model_break_even())
    }
}

fn constant_series(label: &str, value: f64, len: usize) -> ReportSeries {
    ReportSeries {
        label: label.to_owned(),
        values: vec![value as f32; len],
    }
}

/// `values + multiple * errors`, element-wise: an error band drawn around a
/// series that moves. Shorter of the two lengths wins, so a half-written pair
/// truncates rather than inventing points.
fn sum_series(values: &[f32], errors: &[f32], multiple: f32) -> Vec<f32> {
    values
        .iter()
        .zip(errors.iter())
        .map(|(value, error)| value + multiple * error)
        .collect()
}

/// `reference + multiple * errors`, element-wise: an error band drawn around a
/// FIXED reference, e.g. the nominal coverage or the uniform PIT mean.
fn offset_series(errors: &[f32], reference: f64, multiple: f32) -> Vec<f32> {
    errors
        .iter()
        .map(|error| reference as f32 + multiple * error)
        .collect()
}

/// One-point series, the shape the end-of-run battery reports each scalar in.
pub(super) fn point_series(label: &str, value: f64) -> ReportSeries {
    ReportSeries {
        label: label.to_owned(),
        values: vec![value as f32],
    }
}

// ---------------------------------------------------------------------------
// Support decode comparison
// ---------------------------------------------------------------------------

/// The fitted-versus-edge decode comparison of a bar support, written by
/// `trading_bots::torch::train::support_moments::fit_support_moments`.
///
/// Two panels, both properties of the SUPPORT ARTIFACT and of nothing else: no model, no
/// checkpoint, no step. They cannot be produced from inside a training cycle and they do not
/// move when a step does, which is why they are their own bases rather than more columns on a
/// step-indexed chart.
///
/// EVERY CAVEAT LIVES IN A SERIES LABEL, NOT IN A TITLE, and that is deliberate: the TUI's
/// `normalize_title` lowercases everything after each word's first letter, so emphasis in a
/// title is destroyed before a reader sees it, while series legends render verbatim. The
/// qualification that matters here is that THE FITTED DECODE IS NOT THE PRODUCTION PATH — every
/// first-moment decode in the tree still reads `MeanDecode::Edge` — so a reader who sees a
/// fitted-decode line and concludes the pipeline computes it would be wrong, and the label is
/// the only place that correction survives rendering.
pub fn write_support_decode(dir: &Path, decode: &SupportDecode) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let series = |rows: Vec<(String, Vec<f64>)>| -> Vec<ReportSeries> {
        rows.into_iter()
            .map(|(label, values)| ReportSeries {
                label,
                values: values.iter().map(|v| *v as f32).collect(),
            })
            .collect()
    };

    write_chart(
        dir,
        "support_decode_moments",
        "Bar Support Catch-All Leverage, Fitted Conditional Means Against the Edge Decode In \
         Force - per DOF, model-free, measured on the support's own fit sample"
            .to_owned(),
        "dof index (see the `dof index` series; tensor order r, s, u, v, w)",
        "% share / bps",
        ScaleKind::Symlog,
        series(decode.summary_rows()),
    )?;

    write_chart(
        dir,
        "support_decode_bins",
        format!(
            "Bar Support Per-Bin Decode and Leverage for DOF {} - the two catch-alls against the \
             126 interior bins",
            BAR_DOF_NAMES[DOF_R]
        ),
        "bin index, 0 and 127 are the open-ended catch-alls",
        "bps / % share",
        ScaleKind::Symlog,
        series(decode.bin_rows(DOF_R)),
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Continuous bar family fit
// ---------------------------------------------------------------------------

/// The ten panels of `bar_family::fit_bar_families`: the per-DOF fitted density against the
/// empirical histogram, the `r` tail on log-log axes with the measured slope band drawn as a band,
/// the component sweep, the marginal NLL comparison, the atom census and the ruin table.
///
/// None of them is a function of an optimizer step. Every panel is a property of the DRAW and of
/// a fitted family, so no in-run reporter cycle can produce one, which is why all ten are exempt
/// from the cycle walk and named to `bar_family`'s own registry test.
///
/// EVERY CAVEAT LIVES IN A SERIES LABEL, NEVER IN A TITLE. The TUI's `normalize_title` lowercases
/// everything after each word's first letter, so emphasis in a title is destroyed before a reader
/// sees it, while series legends render verbatim. Four qualifications have to survive rendering:
/// the atom probabilities are EXACT BY CONSTRUCTION and not a fitted result; the 1.66-1.84 tail
/// figure is a SPREAD OF SIX PAIRWISE SLOPES with no point estimate and no standard error, so the
/// band is drawn as two reference power laws and never as a value; both NLL columns are already on
/// the SAME mixed-measure density footing because `scoring: density` adds `E[ln width]`; and the
/// ruin bound is set by the SHORT side of the book, not by the worst down bar.
///
/// All ten are [`ScaleKind::Symlog`]: each carries its own x-axis as an explicit series — a bin
/// edge, a threshold in bps, a component count, a leverage — sitting orders of magnitude away from
/// the quantities it indexes, and several of the quantities are signed log densities. `Linear`
/// would flatten every density onto zero beside its axis, and a pure log scale would drop exactly
/// the negative nats that carry the finding.
pub fn write_bar_family(dir: &Path, fit: &BarFamilyFit) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let of = |label: String, values: Vec<f64>| ReportSeries {
        label,
        values: values.iter().map(|v| *v as f32).collect(),
    };

    for dof in &fit.dofs {
        let midpoints: Vec<f64> = dof
            .grid_lo
            .iter()
            .zip(dof.grid_hi.iter())
            .map(|(lo, hi)| 0.5 * (lo + hi))
            .collect();
        let widths: Vec<f64> = dof
            .grid_lo
            .iter()
            .zip(dof.grid_hi.iter())
            .map(|(lo, hi)| hi - lo)
            .collect();
        let atom_note = if dof.atoms.is_empty() {
            "no atom promoted".to_owned()
        } else {
            dof.atoms
                .iter()
                .map(|a| format!("{:+.4} @ {:.3}%", a.value, 100.0 * a.drawn_share))
                .collect::<Vec<_>>()
                .join(", ")
        };
        write_chart(
            dir,
            DENSITY_BASES[dof.dof],
            format!(
                "Continuous Family Against The Empirical Histogram For DOF {} - {}, K = {}, on \
                 the discrete support's own continuous bins",
                BAR_DOF_NAMES[dof.dof],
                dof.kind.as_str(),
                dof.selected_components
            ),
            "continuous bin index, ascending in value; atom bins are dropped (see the `bin \
             midpoint` series)",
            "density per unit / bin edge",
            ScaleKind::Symlog,
            vec![
                of("bin midpoint (the x axis of this panel)".to_owned(), midpoints),
                of("bin width".to_owned(), widths),
                of(
                    format!(
                        "empirical density from THIS draw, mass/width, atom rows excluded; atoms: \
                         {atom_note}"
                    ),
                    dof.empirical_density.clone(),
                ),
                of(
                    format!(
                        "fitted bin-AVERAGE density, {} scaled by the {:.4}% continuous class; \
                         atom probabilities are the empirical shares BY CONSTRUCTION, not a fitted \
                         result",
                        dof.kind.as_str(),
                        100.0 * dof.continuous_share
                    ),
                    dof.fitted_density.clone(),
                ),
                of(
                    format!(
                        "resolution floor {:.4e} = the narrowest nonzero bin of the discrete \
                         competitor; without it a component collapses onto a tick-repeated value \
                         and buys unbounded nats",
                        dof.resolution_floor
                    ),
                    vec![dof.resolution_floor; dof.grid_lo.len()],
                ),
            ],
        )?;
    }

    let tail = &fit.tail;
    let thresholds: Vec<f64> = tail.grid.iter().map(|p| p.threshold * 10_000.0).collect();
    // The measured band drawn AS A BAND: two reference power laws through the first grid point,
    // one at each end of the six pairwise slopes. Never a value, because the band is a spread of
    // slopes and carries no point estimate.
    let band = |exponent: f64| -> Vec<f64> {
        let (anchor_x, anchor_p) = tail
            .grid
            .iter()
            .find(|p| p.empirical_exceedance > 0.0)
            .map(|p| (p.threshold, p.empirical_exceedance))
            .unwrap_or((1.0, 1.0));
        tail.grid
            .iter()
            .map(|p| anchor_p * (p.threshold / anchor_x).powf(-exponent))
            .collect()
    };
    write_chart(
        dir,
        "bar_family_tail_r",
        format!(
            "DOF r Tail On Log-Log Axes, Draw Against Fitted Family - {} rows, |r| reaching {:.2} \
             bps",
            tail.rows,
            tail.max_abs * 10_000.0
        ),
        "grid index, GEOMETRIC in threshold so a power law reads as a straight line (see the \
         `threshold` series)",
        "exceedance probability / threshold bps / rows",
        ScaleKind::Symlog,
        vec![
            of("threshold bps (the x axis of this panel)".to_owned(), thresholds),
            of(
                "empirical P(|r| > x) on THIS draw".to_owned(),
                tail.grid.iter().map(|p| p.empirical_exceedance).collect(),
            ),
            of(
                "empirical exceedance COUNT - read the right tail of this panel only where the \
                 count is large; the last points rest on a handful of bars"
                    .to_owned(),
                tail.grid.iter().map(|p| p.empirical_count as f64).collect(),
            ),
            of(
                format!(
                    "fitted P(|r| > x), truncated gaussian mixture K = {} times the {:.4}% \
                     continuous class",
                    fit.dofs[DOF_R].selected_components,
                    100.0 * fit.dofs[DOF_R].continuous_share
                ),
                tail.grid.iter().map(|p| p.fitted_exceedance).collect(),
            ),
            of(
                format!(
                    "BAND EDGE, slope {:.2}: the measured {:.2}-{:.2} figure is a SPREAD OF SIX \
                     PAIRWISE SLOPES, not a fitted index - it has no point estimate and no \
                     standard error, so it is drawn as a band and never as a value",
                    tail.measured_band.0, tail.measured_band.0, tail.measured_band.1
                ),
                band(tail.measured_band.0),
            ),
            of(
                format!("BAND EDGE, slope {:.2}: same band, other end", tail.measured_band.1),
                band(tail.measured_band.1),
            ),
        ],
    )?;

    let sweep_ks: Vec<f64> = fit
        .dofs
        .first()
        .map(|d| d.sweep.iter().map(|p| p.components as f64).collect())
        .unwrap_or_default();
    let mut sweep_series = vec![of(
        "component count K (the x axis of this panel)".to_owned(),
        sweep_ks,
    )];
    for dof in &fit.dofs {
        sweep_series.push(of(
            format!(
                "{} holdout nats/bar - THE SELECTION CRITERION, declared before the sweep ran; \
                 selected K = {}",
                BAR_DOF_NAMES[dof.dof], dof.selected_components
            ),
            dof.sweep.iter().map(|p| p.holdout_nll).collect(),
        ));
        sweep_series.push(of(
            format!("{} fit nats/bar (in sample, NOT the criterion)", BAR_DOF_NAMES[dof.dof]),
            dof.sweep.iter().map(|p| p.fit_nll).collect(),
        ));
        sweep_series.push(of(
            format!("{} bic per bar", BAR_DOF_NAMES[dof.dof]),
            dof.sweep.iter().map(|p| p.bic_per_bar).collect(),
        ));
        sweep_series.push(of(
            format!(
                "{} components NOT at the MLE (capped at the resolution floor or left at the \
                 moment start)",
                BAR_DOF_NAMES[dof.dof]
            ),
            dof.sweep
                .iter()
                .map(|p| p.unconverged_components as f64)
                .collect(),
        ));
    }
    write_chart(
        dir,
        "bar_family_k_sweep",
        "Continuous Family Component Sweep, Likelihood Against Complexity - per DOF, holdout is \
         one row in ten withheld from every fit"
            .to_owned(),
        "sweep index, ascending in component count K (see the `component count` series)",
        "nats/bar / components",
        ScaleKind::Symlog,
        sweep_series,
    )?;

    let dof_axis: Vec<f64> = (0..fit.dofs.len()).map(|d| d as f64).collect();
    write_chart(
        dir,
        "bar_family_nll",
        format!(
            "Continuous Family Against The 128-Way Discrete Marginal - one footing, {:+.4} \
             nats/bar total gain",
            fit.nats_gained()
        ),
        "dof index in tensor order r, s, u, v, w (see the `dof index` series)",
        "nats/bar / free parameters",
        ScaleKind::Symlog,
        vec![
            of("dof index (the x axis of this panel)".to_owned(), dof_axis.clone()),
            of(
                "continuous family nats/bar, mixed-measure density footing (counting on the \
                 atoms, Lebesgue elsewhere)"
                    .to_owned(),
                fit.dofs.iter().map(|d| d.family_nll).collect(),
            ),
            of(
                "discrete marginal nats/bar under scoring: density - ALREADY a log density on the \
                 SAME measure, because the density rule adds E[ln width]; this is NOT a bin \
                 probability and no offset is applied to either column"
                    .to_owned(),
                fit.dofs.iter().map(|d| d.discrete_nll).collect(),
            ),
            of(
                "nats gained by the continuous family, positive means it wins".to_owned(),
                fit.dofs.iter().map(|d| d.nats_gained()).collect(),
            ),
            of(
                "family holdout nats/bar (out of sample in the density AND in the class term)"
                    .to_owned(),
                fit.dofs.iter().map(|d| d.holdout_nll).collect(),
            ),
            of(
                "discrete histogram REFITTED on the same 90% and scored on the same withheld 10%: \
                 the symmetric counterpart of the family holdout, so the two holdout series are \
                 the fair comparison and the in-sample pair is not"
                    .to_owned(),
                fit.dofs.iter().map(|d| d.discrete_holdout_nll).collect(),
            ),
            of(
                "nats gained on the SYMMETRIC 90/10 footing, positive means the family wins where \
                 the histogram's 127 parameters are paid for out of sample too"
                    .to_owned(),
                fit.dofs.iter().map(|d| d.holdout_nats_gained()).collect(),
            ),
            of(
                "continuous family free parameters".to_owned(),
                fit.dofs.iter().map(|d| d.free_parameters() as f64).collect(),
            ),
            of(
                "discrete free parameters (127 masses per DOF)".to_owned(),
                fit.dofs
                    .iter()
                    .map(|d| d.discrete_free_parameters as f64)
                    .collect(),
            ),
            of(
                "density closure: atoms + fitted mass on and off the charted range, must be one"
                    .to_owned(),
                fit.dofs.iter().map(|d| d.integrated_mass).collect(),
            ),
        ],
    )?;

    // One row per measured atom, then one row per lattice probe, on a single index: an atom census
    // and the lattice diagnostic beside it, because the second is what tells a reader whether the
    // interior of `u` / `v` is a density at all.
    let mut atom_dof = Vec::new();
    let mut atom_value = Vec::new();
    let mut atom_drawn = Vec::new();
    let mut atom_artifact = Vec::new();
    let mut atom_deviation = Vec::new();
    let mut atom_is_family = Vec::new();
    for dof in &fit.dofs {
        for atom in &dof.atoms {
            atom_dof.push(dof.dof as f64);
            atom_value.push(atom.value);
            atom_drawn.push(100.0 * atom.drawn_share);
            atom_artifact.push(100.0 * atom.artifact_mass);
            atom_deviation.push(atom.deviation());
            atom_is_family.push(1.0);
        }
    }
    for probe in &fit.lattice {
        atom_dof.push(probe.dof as f64);
        atom_value.push(probe.value);
        atom_drawn.push(100.0 * probe.share);
        atom_artifact.push(f64::NAN);
        atom_deviation.push(f64::NAN);
        atom_is_family.push(if probe.is_artifact_atom { 1.0 } else { 0.0 });
    }
    write_chart(
        dir,
        "bar_family_atoms",
        format!(
            "Mixed Likelihood Atom Census And Lattice Probe - worst redraw deviation {:.3e}",
            fit.worst_atom_deviation
        ),
        "row index: every family atom first, then every u / v lattice probe (see the `dof index` \
         and `value` series)",
        "% of the draw / value / deviation",
        ScaleKind::Symlog,
        vec![
            of("dof index".to_owned(), atom_dof),
            of("value".to_owned(), atom_value),
            of(
                "share of THIS draw, %. For a family atom this IS the family's atom parameter, \
                 EXACT BY CONSTRUCTION and not a fitted result: the multinomial MLE of a class \
                 probability is the empirical share"
                    .to_owned(),
                atom_drawn,
            ),
            of(
                "share the persisted artifact recorded, % - NaN for a lattice probe, which the \
                 artifact never promoted to an atom"
                    .to_owned(),
                atom_artifact,
            ),
            of(
                "|redraw - artifact|: the only MEASURED quantity in this panel, and the check \
                 that the discrete geometry and this fit are looking at the same rows"
                    .to_owned(),
                atom_deviation,
            ),
            of(
                "1 = the family carries this value as an atom; 0 = mass at a lattice position the \
                 0.5% promotion threshold left inside the continuous class, which no smooth \
                 density on the interior can carry"
                    .to_owned(),
                atom_is_family,
            ),
        ],
    )?;

    write_chart(
        dir,
        "bar_family_ruin_bound",
        format!(
            "Truncation Bound R_max Implied By A Ruin Licence - the SUPPORT licenses {:.4}x, the \
             DRAW itself licenses {:.4}x",
            fit.ruin.support_max_leverage(),
            fit.ruin.draw_max_leverage()
        ),
        "row index, ascending in declared max leverage (see the `leverage` series)",
        "bps / leverage / simple return",
        ScaleKind::Symlog,
        vec![
            of(
                "declared max leverage F (the x axis of this panel)".to_owned(),
                fit.ruin.rows.iter().map(|r| r.leverage).collect(),
            ),
            of(
                "long-side bound -ln(1 - 1/F), bps: what a long at F survives".to_owned(),
                fit.ruin
                    .rows
                    .iter()
                    .map(|r| r.long_log_bound * 10_000.0)
                    .collect(),
            ),
            of(
                "short-side bound ln(1 + 1/F), bps - THIS SIDE ALWAYS BINDS, so a bound taken \
                 from the worst DOWN bar alone overstates licensed leverage"
                    .to_owned(),
                fit.ruin
                    .rows
                    .iter()
                    .map(|r| r.short_log_bound * 10_000.0)
                    .collect(),
            ),
            of(
                "R_max as a simple return at the binding bound, exp(r_max) - 1".to_owned(),
                fit.ruin
                    .rows
                    .iter()
                    .map(|r| r.binding_simple_return)
                    .collect(),
            ),
            of(
                format!(
                    "1 = the draw fits inside this leverage's licence; 0 = truncating r here \
                     would assign ZERO density to bars that happened. The draw reaches {:.2} bps",
                    fit.ruin.draw_min_r.abs().max(fit.ruin.draw_max_r) * 10_000.0
                ),
                fit.ruin
                    .rows
                    .iter()
                    .map(|r| if r.licensed_by_draw { 1.0 } else { 0.0 })
                    .collect(),
            ),
            of(
                format!(
                    "1 = the discrete SUPPORT's own r range fits inside this leverage's licence, \
                     which is the licence the live cap actually rests on; the support reaches \
                     {:.4} / {:+.4} bps",
                    fit.ruin.support_min_r * 10_000.0,
                    fit.ruin.support_max_r * 10_000.0
                ),
                fit.ruin
                    .rows
                    .iter()
                    .map(|r| if r.licensed_by_support { 1.0 } else { 0.0 })
                    .collect(),
            ),
            of(
                format!(
                    "the live LEVERAGE_CAP, {LEVERAGE_CAP}x, for reference against the F axis"
                ),
                vec![LEVERAGE_CAP; fit.ruin.rows.len()],
            ),
            of(
                format!(
                    "the declared MAX_LEVERAGE, {}x, for reference against the F axis",
                    MAX_LEVERAGE
                ),
                vec![MAX_LEVERAGE; fit.ruin.rows.len()],
            ),
        ],
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Corporate-action seam audit
// ---------------------------------------------------------------------------

/// The six panels of `split_seams::audit_split_seams`: the exceedance census over the WHOLE corpus,
/// the nearest-simple-rational cross-tabulation of `exp(r)`, the `s`/`w` comparison against matched
/// ordinary bars, the six pairwise tail slopes before and after the seams are removed, the
/// catch-all bin contamination, and the ruin licence on both sides of the book.
///
/// None of them is a function of an optimizer step. Every panel is a property of the STORED BARS
/// and of a support artifact read from disk, so no in-run reporter cycle can produce one, which is
/// why all six are exempt from the cycle walk and named to `split_seams`'s own registry test.
///
/// EVERY CAVEAT LIVES IN A SERIES LABEL, NEVER IN A TITLE. The TUI's `normalize_title` lowercases
/// everything after each word's first letter, so emphasis in a title is destroyed before a reader
/// sees it, while series legends render verbatim. Four qualifications have to survive rendering:
/// the four criteria are INDEPENDENT and the verdict rests on their CONJUNCTION, never on any one
/// of them; the deviation axis is RELATIVE, because a fixed absolute tolerance would call a 1%
/// error on a ratio of 100 a miss and the same error on 1/100 a bullseye; the cleaned tail and the
/// cleaned support edges are COUNTERFACTUALS and nothing on disk was changed to produce them; and
/// the ruin licence is set by the SHORT side of the book, which binds first.
///
/// All six are [`ScaleKind::Symlog`]: each carries its own x-axis as an explicit series — an
/// exceedance level, a ratio, a bucket edge, a threshold in bps — sitting orders of magnitude away
/// from the counts and shares it indexes, and several of the quantities are signed log ratios.
pub fn write_bar_seams(dir: &Path, audit: &SeamAudit) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let of = |label: String, values: Vec<f64>| ReportSeries {
        label,
        values: values.iter().map(|v| *v as f32).collect(),
    };
    let census = &audit.census;
    let bars = census.bars.max(1) as f64;
    let extremes = census.extremes.max(1) as f64;

    // Panel 1: the exceedance ladder over the whole corpus, and the criterion census beside it.
    // Both indexed on one row axis, because the question "how many bars are out there" and the
    // question "how many of them are splits" are only meaningful together.
    let mut level_axis: Vec<f64> = CENSUS_LOG_LEVELS.to_vec();
    let mut level_counts: Vec<f64> = census.level_counts.iter().map(|c| *c as f64).collect();
    let mut level_share: Vec<f64> = census.level_counts.iter().map(|c| *c as f64 / bars).collect();
    let mut level_kind: Vec<f64> = vec![0.0; CENSUS_LOG_LEVELS.len()];
    for count in census.criterion_counts() {
        level_axis.push(f64::NAN);
        level_counts.push(count as f64);
        level_share.push(count as f64 / extremes);
        level_kind.push(1.0);
    }
    write_chart(
        dir,
        "bar_seam_census",
        format!(
            "Extreme r Census Over The WHOLE Corpus - {} bars, {} above ln 1.5, {} to {} \
             classified corporate-action seams over {} to {} symbols",
            census.bars,
            census.extremes,
            census.seams[TIER_EXACT],
            census.seams[TIER_NEAR],
            census.seam_series[TIER_EXACT],
            census.seam_series[TIER_NEAR],
        ),
        "row index: the six exceedance levels 1.5x/2x/3x/4x/5x/10x first, then the seven criterion \
         counts extremes/on-rational/session-open/unremarkable/isolated/ALL FOUR exact/ALL FOUR \
         loose (see the `kind` series)",
        "bars / share",
        ScaleKind::Symlog,
        vec![
            of(
                "exceedance level as ln(ratio) - NaN on a criterion row, which is not a threshold"
                    .to_owned(),
                level_axis,
            ),
            of("bars".to_owned(), level_counts),
            of(
                "share: of ALL bars on an exceedance row, of the EXTREME bars on a criterion row"
                    .to_owned(),
                level_share,
            ),
            of(
                "kind: 0 = exceedance level, 1 = criterion count. The four criteria are \
                 INDEPENDENT and the verdict rests on their CONJUNCTION, which is the last two \
                 rows; no single criterion is evidence of a split on its own. The two conjunction \
                 rows BRACKET the seam population: the exact-ratio one is a LOWER bound, the loose \
                 one an UPPER bound"
                    .to_owned(),
                level_kind,
            ),
        ],
    )?;

    // Panel 2: where `exp(r)` actually lands. The whole split hypothesis is decided here.
    let ranked = ranked_ratios(census);
    let listed = ranked.len().min(SEAM_RATIOS_CHARTED);
    write_chart(
        dir,
        "bar_seam_ratios",
        format!(
            "Where exp(r) Lands For Extreme Bars - nearest simple rational, {} distinct ratios \
             carrying {} extreme bars",
            ranked.len(),
            census.extremes
        ),
        "row index, descending in extreme-bar count; the first rows are the populous ratios (see \
         the `ratio value` series)",
        "bars / ratio / relative deviation",
        ScaleKind::Symlog,
        vec![
            of(
                "ratio value num/den (the x axis of this panel)".to_owned(),
                ranked[..listed].iter().map(|(r, _, _)| r.value()).collect(),
            ),
            of(
                "extreme bars whose exp(r) is nearest this ratio".to_owned(),
                ranked[..listed].iter().map(|(_, total, _)| *total as f64).collect(),
            ),
            of(
                "of those, bars satisfying ALL FOUR criteria with exp(r) EXACTLY on the ratio - \
                 the lower bound on this ratio's seam population"
                    .to_owned(),
                ranked[..listed]
                    .iter()
                    .map(|(_, _, seams)| seams[TIER_EXACT] as f64)
                    .collect(),
            ),
            of(
                "of those, bars satisfying ALL FOUR with exp(r) within the LOOSE ratio tolerance, \
                 i.e. the ratio times one bar of market move - the upper bound"
                    .to_owned(),
                ranked[..listed]
                    .iter()
                    .map(|(_, _, seams)| seams[TIER_NEAR] as f64)
                    .collect(),
            ),
            of(
                "numerator of the ratio, so a reader can name it: 5 with denominator 1 is a 5:1 \
                 split, 1 with denominator 5 a 1:5 reverse split"
                    .to_owned(),
                ranked[..listed].iter().map(|(r, _, _)| f64::from(r.num)).collect(),
            ),
            of(
                "denominator of the ratio".to_owned(),
                ranked[..listed].iter().map(|(r, _, _)| f64::from(r.den)).collect(),
            ),
        ],
    )?;

    // Panel 3: is the bar otherwise unremarkable? Two histograms per population, on one index.
    let range_edges = range_ratio_edges();
    let volume_edges_lo = volume_edges();
    let mut context_axis: Vec<f64> = range_edges.clone();
    context_axis.extend(volume_edges_lo.iter().copied());
    let mut context_kind: Vec<f64> = vec![0.0; range_edges.len()];
    context_kind.extend(std::iter::repeat_n(1.0, volume_edges_lo.len()));
    let normalize = |counts: &[u64]| -> Vec<f64> {
        let total: u64 = counts.iter().sum();
        counts
            .iter()
            .map(|c| *c as f64 / total.max(1) as f64)
            .collect()
    };
    let mut extreme_share = normalize(&census.range_ratio_extreme);
    extreme_share.extend(normalize(&census.volume_extreme));
    let mut ordinary_share = normalize(&census.range_ratio_ordinary);
    ordinary_share.extend(normalize(&census.volume_ordinary));
    let mut deviation_axis: Vec<f64> = deviation_edges();
    let deviation_counts: Vec<f64> = census.deviation.iter().map(|c| *c as f64).collect();
    deviation_axis.resize(context_axis.len().max(deviation_axis.len()), f64::NAN);
    write_chart(
        dir,
        "bar_seam_context",
        format!(
            "Are Extreme r Bars Otherwise Unremarkable - {} extreme bars against {} ordinary \
             non-flat bars",
            census.extremes,
            census.bars - census.level_counts[0]
        ),
        "row index: the s/|r| histogram first, then the w histogram, then the rational-deviation \
         histogram (see the `kind` and `bucket edge` series)",
        "share of population / bucket edge",
        ScaleKind::Symlog,
        vec![
            of(
                "bucket lower edge (the x axis of this panel): s/|r| on kind 0, w in nats on kind 1"
                    .to_owned(),
                context_axis,
            ),
            of("kind: 0 = s/|r| bucket, 1 = w bucket".to_owned(), context_kind),
            of(
                "EXTREME bars, share of their own population. A genuine move of |r| has to trade \
                 through it, so its s/|r| is near or above 1 and its volume spikes; a level shift \
                 between two bars does neither"
                    .to_owned(),
                extreme_share,
            ),
            of(
                "MATCHED ORDINARY bars, share of their own population - every bar with r != 0, \
                 measured in the same pass with the same encoder"
                    .to_owned(),
                ordinary_share,
            ),
            of(
                "relative distance of exp(r) from the nearest simple rational, RELATIVE and never \
                 absolute: bucket edges are the `deviation edge` series"
                    .to_owned(),
                deviation_counts,
            ),
            of("deviation bucket lower edge, relative".to_owned(), deviation_axis),
        ],
    )?;

    // Panel 4: the tail estimator, control against both cleaned tiers, at the same four levels.
    let slope_axis: Vec<f64> = (0..audit.control.slopes.len()).map(|i| i as f64).collect();
    write_chart(
        dir,
        "bar_seam_tail_r",
        format!(
            "Six Pairwise Tail Slopes On |r|, Control Against Seam-Removed - {} to {} of {} draw \
             rows removed",
            audit.draw_rows_removed[TIER_EXACT], audit.draw_rows_removed[TIER_NEAR], audit.draw_rows
        ),
        "pair index over the four exceedance levels 1e-2/3e-3/1e-3/3e-4 (see the `p high` and `p \
         low` series)",
        "slope / exceedance level / threshold bps",
        ScaleKind::Symlog,
        vec![
            of("pair index (the x axis of this panel)".to_owned(), slope_axis),
            of(
                "p high, the nearer level of the pair".to_owned(),
                audit.control.slopes.iter().map(|s| s.p_high).collect(),
            ),
            of(
                "p low, the further-out level of the pair".to_owned(),
                audit.control.slopes.iter().map(|s| s.p_low).collect(),
            ),
            of(
                "CONTROL slope on the untouched draw - the same estimator at the same levels the \
                 live figure was measured with, reproduced here so the cleaned columns beside it \
                 are comparable to anything at all"
                    .to_owned(),
                audit.control.slopes.iter().map(|s| s.alpha).collect(),
            ),
            of(
                "CLEANED slope with every EXACT-ratio seam row removed. A COUNTERFACTUAL: no \
                 corpus file and no support artifact was changed to produce it"
                    .to_owned(),
                audit.cleaned[TIER_EXACT]
                    .slopes
                    .iter()
                    .map(|s| s.alpha)
                    .collect(),
            ),
            of(
                "CLEANED slope with every LOOSE-ratio seam row removed, i.e. the largest \
                 population the classification can defend. Also a COUNTERFACTUAL"
                    .to_owned(),
                audit.cleaned[TIER_NEAR]
                    .slopes
                    .iter()
                    .map(|s| s.alpha)
                    .collect(),
            ),
            of(
                "control threshold bps at p low".to_owned(),
                audit
                    .control
                    .slopes
                    .iter()
                    .map(|s| s.x_low * 10_000.0)
                    .collect(),
            ),
            of(
                "cleaned threshold bps at p low, loose tier".to_owned(),
                audit.cleaned[TIER_NEAR]
                    .slopes
                    .iter()
                    .map(|s| s.x_low * 10_000.0)
                    .collect(),
            ),
        ],
    )?;

    // Panel 5: what the seams contribute to the two catch-all bins.
    let side_axis = vec![0.0, f64::from(NUM_BAR_BINS as i32 - 1)];
    write_chart(
        dir,
        "bar_seam_bin_mass",
        format!(
            "Corporate-Action Seams Inside The r Catch-All Bins - {:.4}% to {:.4}% of bin 0 and \
             {:.4}% to {:.4}% of bin {}",
            100.0 * audit.catch_all_seam_share(TIER_EXACT)[0],
            100.0 * audit.catch_all_seam_share(TIER_NEAR)[0],
            100.0 * audit.catch_all_seam_share(TIER_EXACT)[1],
            100.0 * audit.catch_all_seam_share(TIER_NEAR)[1],
            NUM_BAR_BINS - 1
        ),
        "row index: bin 0 then bin 127, the two open-ended catch-alls (see the `bin index` series)",
        "bars / share",
        ScaleKind::Symlog,
        vec![
            of("bin index (the x axis of this panel)".to_owned(), side_axis),
            of(
                "mass the persisted artifact records for this bin, % of its 4M fitting draw"
                    .to_owned(),
                audit
                    .support_catch_all_mass
                    .iter()
                    .map(|m| 100.0 * m)
                    .collect(),
            ),
            of(
                "bars the WHOLE corpus puts in this bin".to_owned(),
                census.catch_all.iter().map(|c| *c as f64).collect(),
            ),
            of(
                "of those, seams on the EXACT ratio test - the LOWER bound on this bin's \
                 contamination"
                    .to_owned(),
                census.catch_all_seams[TIER_EXACT]
                    .iter()
                    .map(|c| *c as f64)
                    .collect(),
            ),
            of(
                "of those, seams on the LOOSE ratio test - the UPPER bound".to_owned(),
                census.catch_all_seams[TIER_NEAR]
                    .iter()
                    .map(|c| *c as f64)
                    .collect(),
            ),
            of(
                "exact-tier seam share of this bin's whole-corpus population, %".to_owned(),
                audit
                    .catch_all_seam_share(TIER_EXACT)
                    .iter()
                    .map(|s| 100.0 * s)
                    .collect(),
            ),
            of(
                "loose-tier seam share of this bin's whole-corpus population, %".to_owned(),
                audit
                    .catch_all_seam_share(TIER_NEAR)
                    .iter()
                    .map(|s| 100.0 * s)
                    .collect(),
            ),
            of(
                "bars the TRAIN REGION puts in this bin - the population the support was actually \
                 fitted from, and therefore the one its masses describe"
                    .to_owned(),
                census.catch_all_train.iter().map(|c| *c as f64).collect(),
            ),
            of(
                "of those, REVERTING bad prints - the competing non-market population, which is \
                 NOT a corporate action and is not removed by any cleaned reading in this audit"
                    .to_owned(),
                census.catch_all_reverts.iter().map(|c| *c as f64).collect(),
            ),
            of(
                "loose-tier seam share of this bin's train-region population, %".to_owned(),
                audit
                    .catch_all_seam_share_train(TIER_NEAR)
                    .iter()
                    .map(|s| 100.0 * s)
                    .collect(),
            ),
        ],
    )?;

    // Panel 6: the licence, on both sides, four ways.
    let licence_axis = vec![0.0, 1.0, 2.0, 3.0];
    let edges_lo = vec![
        audit.support_lo * 10_000.0,
        audit.control.clip_lo * 10_000.0,
        audit.cleaned[TIER_EXACT].clip_lo * 10_000.0,
        audit.cleaned[TIER_NEAR].clip_lo * 10_000.0,
    ];
    let edges_hi = vec![
        audit.support_hi * 10_000.0,
        audit.control.clip_hi * 10_000.0,
        audit.cleaned[TIER_EXACT].clip_hi * 10_000.0,
        audit.cleaned[TIER_NEAR].clip_hi * 10_000.0,
    ];
    let long_licence = vec![
        audit.support_long_max_leverage(),
        audit.control.long_max_leverage(),
        audit.cleaned[TIER_EXACT].long_max_leverage(),
        audit.cleaned[TIER_NEAR].long_max_leverage(),
    ];
    let short_licence = vec![
        audit.support_short_max_leverage(),
        audit.control.short_max_leverage(),
        audit.cleaned[TIER_EXACT].short_max_leverage(),
        audit.cleaned[TIER_NEAR].short_max_leverage(),
    ];
    let binding_licence = vec![
        audit.support_binding_max_leverage(),
        audit.control.binding_max_leverage(),
        audit.cleaned[TIER_EXACT].binding_max_leverage(),
        audit.cleaned[TIER_NEAR].binding_max_leverage(),
    ];
    write_chart(
        dir,
        "bar_seam_ruin_licence",
        format!(
            "Ruin Licence From The r Support Edges, Live Against Seam-Removed - the SHORT side \
             binds at {:.4}x live and {:.4}x with every loose-tier seam removed",
            audit.support_binding_max_leverage(),
            audit.cleaned[TIER_NEAR].binding_max_leverage()
        ),
        "row index: 0 = the LIVE artifact's own edges, 1 = the same clip quantiles recomputed on \
         the untouched draw, 2 = recomputed with the exact-ratio seams removed, 3 = with the \
         loose-ratio seams removed",
        "log bound / bps / leverage",
        ScaleKind::Symlog,
        vec![
            of("row index (the x axis of this panel)".to_owned(), licence_axis),
            of("lower edge lo[r][0] in bps".to_owned(), edges_lo),
            of("upper edge hi[r][127] in bps".to_owned(), edges_hi),
            of(
                "LONG licence 1/(1 - exp(r_min)) - the side that does NOT bind, and a bound \
                 derived from the worst down bar alone OVERSTATES the licensed leverage"
                    .to_owned(),
                long_licence,
            ),
            of(
                "SHORT licence 1/(exp(r_max) - 1) - THE BINDING SIDE, because ln(1 + y) < \
                 -ln(1 - y) for y in (0, 1)"
                    .to_owned(),
                short_licence,
            ),
            of(
                "binding licence, min of the two. Rows 1 to 3 are COUNTERFACTUALS: no support \
                 artifact was refitted and nothing on disk was changed to produce them"
                    .to_owned(),
                binding_licence,
            ),
        ],
    )?;
    Ok(())
}

/// Ratios charted in `bar_seam_ratios`, most populous first. Bounded so the panel stays legible
/// when a long tail of one-bar ratios exists.
const SEAM_RATIOS_CHARTED: usize = 48;

// ---------------------------------------------------------------------------
// Memorization probe
// ---------------------------------------------------------------------------

/// The four panels of `mem_probe::mem_probe`: the epoch spine, the one-repetition contrast, the
/// recency decomposition of the arm carrying the extra exposure, and the bootstrap's own
/// stability.
///
/// EVERY CAVEAT LIVES IN A SERIES LABEL, NEVER IN A TITLE. The TUI's `normalize_title`
/// lowercases everything after each word's first letter, so emphasis in a title is destroyed
/// before a reader sees it, while series legends render verbatim. Three qualifications have to
/// survive rendering: the spine's gap is CONTAMINATED by calendar in its level and by learning
/// rate in its trajectory and is never the discriminator; the symbol-paired gap is a strictly
/// LESS contaminated version of that same contaminated quantity and never a clean one; and the
/// recency profile lives inside ONE arm, so it is descriptive rather than a contrast. Only the
/// one-repetition panel carries a randomized comparison, and its label is where that is said.
///
/// All four panels are [`ScaleKind::Symlog`] for one reason: each carries its own x-axis as an
/// explicit series — a step, a bucket age, a draw count — which sits three to five orders of
/// magnitude above the estimates it indexes, while the estimates themselves are signed
/// differences. `Linear` would flatten every estimate onto zero beside its axis, and a log
/// scale would drop exactly the negative points that carry the finding.
pub fn write_mem_probe(
    dir: &Path,
    subtitle: &str,
    gap_points: &[GapPoint],
    seen_more: &Arm,
    seen_fewer: &Arm,
    contrast: &PairedContrast,
    recency: &[RecencyBucket],
    stability: &[StabilityPoint],
) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let series_of = |label: &str, values: Vec<f64>| ReportSeries {
        label: label.to_owned(),
        values: values.iter().map(|value| *value as f32).collect(),
    };

    let spine = |pick: &dyn Fn(&GapPoint) -> f64| -> Vec<f64> {
        gap_points.iter().map(pick).collect()
    };
    write_chart(
        dir,
        "memprobe_epoch_spine",
        format!(
            "Memorization Probe Epoch Spine, Held-Out Against Train-Split NLL At Each Pass \
             Count - {subtitle}"
        ),
        "spine index, ascending in optimizer step (see the `step` series)",
        "nats/bar / optimizer step / symbols",
        ScaleKind::Symlog,
        vec![
            series_of("step", spine(&|point| point.step as f64)),
            series_of(
                "train-split nll, nats/bar (bars the run TRAINED on)",
                spine(&|point| point.train_nll),
            ),
            series_of(
                "held-out nll, nats/bar (val split)",
                spine(&|point| point.heldout_nll),
            ),
            series_of(
                "train-split conditional nll, nats/bar",
                spine(&|point| point.train_nll_conditional),
            ),
            series_of(
                "held-out conditional nll, nats/bar",
                spine(&|point| point.heldout_nll_conditional),
            ),
            series_of(
                "gap = held-out minus train, CONTAMINATED and NOT a discriminator: the splits are \
                 calendar-disjoint so the LEVEL mixes regime, and lr_multiplier is affine in step \
                 past the plateau so the TRAJECTORY mixes passes with learning rate",
                spine(&|point| point.gap),
            ),
            series_of(
                "conditional gap, CONTAMINATED identically - same splits, same collinearity",
                spine(&|point| point.gap_conditional),
            ),
            series_of(
                "gap paired within symbol - STRICTLY LESS CONTAMINATED, NEVER CLEAN: pairing \
                 removes the cross-sectional component, the calendar component cannot be removed \
                 because the splits are calendar-disjoint by construction",
                spine(&|point| point.symbol_paired_gap.mean),
            ),
            series_of(
                "gap paired within symbol, ci low",
                spine(&|point| point.symbol_paired_gap.ci_low),
            ),
            series_of(
                "gap paired within symbol, ci high",
                spine(&|point| point.symbol_paired_gap.ci_high),
            ),
            series_of(
                "symbols paired - the resampling units that set the paired interval's width",
                spine(&|point| point.symbols_paired as f64),
            ),
            constant_series("zero gap", 0.0, gap_points.len()),
        ],
    )?;

    // One-point series, the shape this module already reports scalars in. Every number here is a
    // property of ONE checkpoint, so there is no axis to walk: the panel is a labelled battery.
    let mut repetition = vec![
        point_series(
            "paired nll delta, seen-more minus seen-fewer, nats/bar - RANDOMIZED BY \
             CONSTRUCTION and the only discriminator in this probe: same weights, same learning \
             rate, same ramp stage, same context, one variable",
            contrast.nll.mean,
        ),
        point_series("paired nll delta, ci low", contrast.nll.ci_low),
        point_series("paired nll delta, ci high", contrast.nll.ci_high),
        point_series(
            "paired conditional nll delta, nats/bar",
            contrast.nll_conditional.mean,
        ),
        point_series(
            "paired conditional nll delta, ci low",
            contrast.nll_conditional.ci_low,
        ),
        point_series(
            "paired conditional nll delta, ci high",
            contrast.nll_conditional.ci_high,
        ),
        point_series(
            "paired MZ mean-slope delta, (symbol,month) key - the campaign's key, which treats \
             two symbols in the same month as independent draws when same-instant cross-symbol \
             correlation is the dominant dependence in this corpus",
            contrast.slope.mean,
        ),
        point_series(
            "paired MZ mean-slope delta, (symbol,month) key, ci low",
            contrast.slope.ci_low,
        ),
        point_series(
            "paired MZ mean-slope delta, (symbol,month) key, ci high",
            contrast.slope.ci_high,
        ),
        point_series(
            "paired MZ mean-slope delta, MONTH-ALONE key - conservative against exactly that \
             cross-symbol term, so a delta resolved under the campaign key alone is NOT safe to \
             call resolved",
            contrast.slope_month.mean,
        ),
        point_series(
            "paired MZ mean-slope delta, MONTH-ALONE key, ci low",
            contrast.slope_month.ci_low,
        ),
        point_series(
            "paired MZ mean-slope delta, MONTH-ALONE key, ci high",
            contrast.slope_month.ci_high,
        ),
        point_series("zero, i.e. one extra exposure changed nothing", 0.0),
        point_series(
            "shared blocks carrying a paired nll observation",
            contrast.shared_blocks as f64,
        ),
        point_series(
            "blocks resolving a slope in BOTH arms",
            contrast.slope_blocks as f64,
        ),
        point_series(
            "months resolving a slope in BOTH arms",
            contrast.slope_month_blocks as f64,
        ),
    ];
    for arm in [seen_more, seen_fewer] {
        repetition.push(point_series(
            &format!("{} - nll LEVEL, nats/bar (a level, not a contrast)", arm.label),
            arm.nll.mean,
        ));
        repetition.push(point_series(
            &format!("{} - MZ mean slope, (symbol,month) key", arm.label),
            arm.mean_slope.beta,
        ));
        repetition.push(point_series(
            &format!("{} - MZ mean slope, MONTH-ALONE key", arm.label),
            arm.mean_slope_month.beta,
        ));
        repetition.push(point_series(
            &format!("{} - blocks (the interval's denominator, not windows)", arm.label),
            arm.blocks as f64,
        ));
        repetition.push(point_series(
            &format!(
                "{} - bars DROPPED as non-finite, which is a defect rather than a filter: \
                 mincer_zarnowitz skips them silently and would refit on a different population",
                arm.label
            ),
            arm.dropped_bars as f64,
        ));
    }
    write_chart(
        dir,
        "memprobe_one_repetition",
        format!(
            "Memorization Probe One-Repetition Contrast, Paired Within Shared Blocks At One \
             Checkpoint - {} exposures against {} - {subtitle}",
            seen_more.exposures, seen_fewer.exposures
        ),
        "one checkpoint, so a single index; every series is one labelled scalar",
        "nats/bar / slope / blocks / bars",
        ScaleKind::Symlog,
        repetition,
    )?;

    let bucket = |pick: &dyn Fn(&RecencyBucket) -> f64| -> Vec<f64> {
        recency.iter().map(pick).collect()
    };
    write_chart(
        dir,
        "memprobe_recency",
        format!(
            "Memorization Probe Recency Profile Within The {}-Exposure Arm, By How Long Ago The \
             Extra Exposure Happened - {subtitle}",
            seen_more.exposures
        ),
        "recency bucket, ascending in age (see the `steps since the extra exposure` series)",
        "nats/bar / slope / steps / windows",
        ScaleKind::Symlog,
        vec![
            series_of(
                "steps since the extra exposure, bucket mean",
                bucket(&|point| point.steps_ago),
            ),
            series_of(
                "nll, nats/bar - WITHIN one arm, so DESCRIPTIVE and NOT a contrast: flat across \
                 buckets is DURABLE memorization, concentrated at the newest bucket is TRANSIENT \
                 retention, and the two have different remedies",
                bucket(&|point| point.nll.mean),
            ),
            series_of("nll, ci low", bucket(&|point| point.nll.ci_low)),
            series_of("nll, ci high", bucket(&|point| point.nll.ci_high)),
            series_of(
                "MZ mean slope in bucket, (symbol,month) key - also within one arm, so it is a \
                 level and cannot on its own attribute over-dispersion to the extra exposure",
                bucket(&|point| point.slope.beta),
            ),
            series_of(
                "MZ mean slope, ci low",
                bucket(&|point| point.slope.beta_ci.0),
            ),
            series_of(
                "MZ mean slope, ci high",
                bucket(&|point| point.slope.beta_ci.1),
            ),
            series_of("windows in bucket", bucket(&|point| point.windows as f64)),
            series_of(
                "blocks in bucket - the resampling units, not the windows",
                bucket(&|point| point.blocks as f64),
            ),
        ],
    )?;

    let draw = |pick: &dyn Fn(&StabilityPoint) -> f64| -> Vec<f64> {
        stability.iter().map(pick).collect()
    };
    write_chart(
        dir,
        "memprobe_bootstrap_stability",
        format!(
            "Memorization Probe Bootstrap Stability, Slope Interval Width Against Draw Count - \
             {subtitle}"
        ),
        "draw-count index, ascending (see the `bootstrap draws` series)",
        "slope / interval width / draws",
        ScaleKind::Symlog,
        vec![
            series_of("bootstrap draws", draw(&|point| point.draws as f64)),
            series_of(
                "beta POINT ESTIMATE - not a bootstrap quantity, so it MUST NOT move across draw \
                 counts; movement here is a BUG and never a finding",
                draw(&|point| point.beta),
            ),
            series_of(
                "beta_se - a bootstrap quantity, and one of the two that MAY move",
                draw(&|point| point.beta_se),
            ),
            series_of(
                "percentile ci width - the other quantity that may move, and the one under \
                 suspicion: 92.384% of the decoded mean's per-bar sampling variance sits in the \
                 two catch-all bins, so this resamples a HEAVY-TAILED statistic whose percentiles \
                 may converge more slowly than BOOTSTRAP_DRAWS assumes",
                draw(&|point| point.ci_width),
            ),
            constant_series(
                "configured BOOTSTRAP_DRAWS, the draw count every other interval in this \
                 campaign was taken at",
                super::pretrain_stats::BOOTSTRAP_DRAWS as f64,
                stability.len(),
            ),
        ],
    )?;
    Ok(())
}

/// The POPULATION a held-out pass will measure on, and the interval that population can
/// support — charted before anything is scored.
///
/// Two panels because there are two questions and they have different x-axes. The census asks
/// what each split HOLDS at one context: bars, near-disjoint windows, and symbols carrying at
/// least one window. The ladder asks what a traded prefix of the addressed split can RESOLVE,
/// indexed on the prefix size, with the `(symbol, calendar month)` block count COUNTED over the
/// real draw at every rung rather than assumed equal to the window count.
///
/// Why this is a chart at all rather than a line of log output. `Split::Test` is scored ONCE for
/// the whole campaign; the question "does it have the power to resolve the effect we are looking
/// for" therefore has to be answerable, and answered, before the draw is spent. Both panels are
/// functions of the stored bars and of a draw pinned by [`EVAL_WINDOW_SEED`], so neither moves
/// when a step does and no model is involved in producing them.
///
/// Both are [`ScaleKind::Symlog`]: a bar count near 4e7, a window count near 4e4 and an interval
/// half-width near 1 bps sit six orders of magnitude apart on one index, and `Linear` would
/// flatten every count and every width onto the axis of the largest.
pub(super) fn write_heldout_power(dir: &Path, power: &HeldOutPower) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let of = |label: String, values: Vec<f64>| ReportSeries {
        label,
        values: values.iter().map(|v| *v as f32).collect(),
    };
    write_chart(
        dir,
        "pretrain_heldout_census",
        format!(
            "Held-Out Population At Context {} - the {} split holds {} bars, {} near-disjoint \
             windows and {} symbols with at least one window",
            power.context,
            power.split.as_str(),
            power
                .census
                .iter()
                .find(|row| row.split == power.split)
                .map_or(0, |row| row.bars),
            power
                .census
                .iter()
                .find(|row| row.split == power.split)
                .map_or(0, |row| row.anchors),
            power
                .census
                .iter()
                .find(|row| row.split == power.split)
                .map_or(0, |row| row.symbols),
        ),
        "row index: one per split, in calendar order train / val / test (see the `split` series)",
        "bars / windows / symbols",
        ScaleKind::Symlog,
        vec![
            of(
                "split: 0 = train, 1 = val, 2 = test (the x axis of this panel). The three are \
                 calendar-DISJOINT and half-open, cut at the two pinned instants"
                    .to_owned(),
                power
                    .census
                    .iter()
                    .map(|row| match row.split {
                        Split::Train => 0.0,
                        Split::Val => 1.0,
                        Split::Test => 2.0,
                    })
                    .collect(),
            ),
            of(
                "bars in the split - f32, so counts above 16.7M are rounded; the window manifest \
                 carries them exactly"
                    .to_owned(),
                power.census.iter().map(|row| row.bars as f64).collect(),
            ),
            of(
                format!(
                    "near-disjoint windows of {} bars the split can supply - the CEILING on \
                     --windows and therefore on any interval taken over it",
                    power.context
                ),
                power.census.iter().map(|row| row.anchors as f64).collect(),
            ),
            of(
                "symbols holding at least one such window - the pinned draw is quota-allocated \
                 per symbol, so this bounds how many DISTINCT symbols a draw can spread over, \
                 and a block is a (symbol, calendar month) pair"
                    .to_owned(),
                power.census.iter().map(|row| row.symbols as f64).collect(),
            ),
        ],
    )?;

    write_chart(
        dir,
        "pretrain_heldout_power",
        format!(
            "What The {} Split Can RESOLVE - {} drawn windows, traded prefix {} windows over {} \
             blocks, fit slice {} windows over {} blocks, NOTHING SCORED",
            power.split.as_str(),
            power.windows_drawn,
            power.traded_windows,
            power.traded_blocks,
            power.fit_windows,
            power.fit_blocks,
        ),
        "row index, ascending in traded-window count from the pinned prefix to the whole draw \
         (see the `traded windows` series)",
        "windows / blocks / bps per bar",
        ScaleKind::Symlog,
        vec![
            of(
                "traded windows (the x axis of this panel)".to_owned(),
                power.ladder.iter().map(|(n, _, _)| *n as f64).collect(),
            ),
            of(
                "distinct (symbol, calendar month) blocks in that prefix - COUNTED over the real \
                 draw, never assumed equal to the window count. Blocks are what the bootstrap \
                 resamples, so this and not the bar count is what sets the interval"
                    .to_owned(),
                power.ladder.iter().map(|(_, b, _)| *b as f64).collect(),
            ),
            of(
                "expected 95% half-width on net growth, bps/bar. EXTRAPOLATED, not measured: the \
                 campaign reference width scaled by sqrt(B_ref / B), which imports s_block - the \
                 cross-block dispersion of per-block net, a property of the REGIME rather than of \
                 the sample size. A split whose regime is more dispersed will be wider than this"
                    .to_owned(),
                power.ladder.iter().map(|(_, _, w)| *w).collect(),
            ),
        ],
    )?;
    Ok(())
}

/// SHA-256 of a checkpoint, streamed so a multi-hundred-megabyte artifact never
/// lands in memory. Fingerprinting the promotion is what lets the held-out
/// battery prove it scored the selected weights and not something else.
fn file_digest(path: &Path) -> Result<[u8; 32]> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to open {} for fingerprinting", path.display()))?;
    let mut context = DigestContext::new(&SHA256);
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if read == 0 {
            break;
        }
        context.update(&buffer[..read]);
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(context.finish().as_ref());
    Ok(digest)
}

/// Write a chart, skipping it while no series holds a finite value so the run
/// directory never fills with all-NaN placeholders.
pub(super) fn write_chart(
    dir: &Path,
    base: &str,
    title: String,
    x_label: &str,
    y_label: &str,
    scale: ScaleKind,
    series: Vec<ReportSeries>,
) -> Result<()> {
    if !series
        .iter()
        .any(|s| s.values.iter().any(|value| value.is_finite()))
    {
        return Ok(());
    }
    write_report_at(
        &dir.join(format!("{base}.report.bin")),
        &Report {
            title,
            x_label: Some(x_label.to_owned()),
            y_label: Some(y_label.to_owned()),
            scale,
            kind: ReportKind::MultiLine { series },
        },
    )
}

fn write_report_at(path: &Path, report: &Report) -> Result<()> {
    write_report(path, report).with_context(|| format!("failed to write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::trade_bench::{
        BenchConfig, TailCounts, WindowPaths, DEFAULT_COST_BPS, POLICY_HALF,
        POLICY_KELLY_MULTIPLE, POLICY_QUARTER,
    };
    use crate::torch::test_rng;
    use shared::report::read_report;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Every base this module can write. Aliased rather than restated: the list lives in
    /// `shared` so the TUI extends its `meta_chart_bases` from the SAME slice this test
    /// walks. A base registered with no writer and a base written with no registration
    /// are both unrepresentable rather than merely tested for.
    const EXPECTED_BASES: &[&str] = shared::report::PRETRAIN_REPORT_BASES;

    /// Registered bases a single in-run cycle cannot produce.
    ///
    /// **The rule, arrived at the hard way.** A base may be exempt from the cycle walk ONLY if
    /// some other test EXECUTES its writer, and that test is named in the entry. A stated reason
    /// is not coverage: `pretrain_corpus_anomalies` carried the reason "written by the corpus
    /// loader at startup, not by this module" while no writer existed at all, and the exemption
    /// is precisely what made that invisible to the bidirectional test built to find exactly
    /// that gap. The comment was true about the intent and false about the code, and only
    /// execution can tell those apart.
    ///
    /// Every name here is also asserted to BE in the registry, so a rename turns into a failure
    /// rather than into an exemption that silently covers for nothing.
    const CYCLE_EXEMPT: &[&str] = &[
        // `finish` writes it and consumes the reporter, so it belongs to the end of a run.
        // Executed by `the_held_out_battery_is_written_once_with_every_scalar`.
        "pretrain_test",
        // Written by the corpus loader — `dataset::CorpusAnomalies::write_report_of`, called
        // from `pretrain::build_trainer` once the generation directory exists — not by this
        // module. Executed by `dataset::tests::the_anomaly_report_carries_every_resolution_on_one_base`.
        "pretrain_corpus_anomalies",
        // Written by `pretrain_aux::AuxiliaryReport::write_report` at each pass boundary, and
        // only by a run that named `--auxiliary-resolutions`, which the default configuration
        // this fixture drives does not. Executed by
        // `pretrain_aux::tests::the_auxiliary_report_lands_with_one_distinguishable_series_pair_per_resolution`.
        "pretrain_auxiliary_nll",
        // Written by `portfolio::write_portfolio_bench`, which runs ONE book over a
        // calendar-aligned panel and is not part of a pretraining cycle at all: it needs a
        // whole held-out panel and a loaded checkpoint, neither of which this fixture has.
        // All five are executed by
        // `portfolio::tests::the_five_portfolio_bases_are_written_and_read_back`.
        "pretrain_portfolio_equity",
        "pretrain_portfolio_metrics",
        "pretrain_portfolio_gross_curve",
        "pretrain_portfolio_frontier",
        "pretrain_portfolio_edge_vs_cost",
        // Written by `portfolio_cost::write_cost_capacity_reports`. A measured spread, a
        // dollar ADV and a realized cross-sectional covariance are properties of the stored
        // bars, not of a training step, so no in-run cycle over synthetic step metrics can
        // produce any of them. All three are executed by
        // `portfolio_cost::tests::the_cost_capacity_battery_writes_all_three_registered_bases`.
        "pretrain_cost_deciles",
        "pretrain_capacity_curve",
        "pretrain_cross_correlation",
        // Written by `write_mean_calibration`, from the multi-checkpoint calibration
        // experiment: one point per CHECKPOINT, each needing its own held-out pass plus a
        // second pass on a block-disjoint fit slice, so an in-run cycle over one step's
        // metrics cannot produce either. All three are executed by
        // `the_calibration_experiment_writes_both_registered_bases`.
        "pretrain_mean_calibration",
        "pretrain_shrunk_policy",
        // The no-trade band needs the same two passes AND a re-scored ledger per band width
        // per shape rule, so it is exempt for the same reason and executed by the same test.
        "pretrain_no_trade_band",
        // The edge attribution re-scores the SAME two passes with the model's magnitude and
        // then its sign destroyed, so it is exempt for the same reason and executed by the
        // same test. Three bases because they have three different x-axes: the arm, the
        // checkpoint and the confidence decile.
        "pretrain_edge_attribution",
        "pretrain_edge_panel",
        "pretrain_edge_confidence",
        // The sign-hysteresis frontier and the signal-decay curve re-score the SAME held-out
        // pass along two axes the attribution does not have - the flip margin and the holding
        // horizon - so they are exempt for the same reason and executed by the same test.
        "pretrain_edge_hysteresis",
        "pretrain_signal_decay",
        // The shrink x hysteresis 2x2 needs the recalibrated fraction from a disjoint fit slice
        // AND the frontier's constant-stake reconstruction on the same windows, so it is exempt
        // for the same reason and executed by the same test.
        "pretrain_edge_composition",
        // Written by `horizon::write_horizon_frontier`. One point per holding horizon, each
        // needing a whole held-out panel, a loaded checkpoint and a sampled multi-bar rollout,
        // so an in-run cycle over one step's metrics cannot produce it. Executed by
        // `horizon::tests::the_horizon_frontier_base_is_written_and_read_back`.
        "pretrain_horizon_frontier",
        // Written by `skill::write_skill_profile`. Indexed by DECILE of the model's own
        // confidence rather than by step, and produced from a whole held-out panel scored with
        // no trading policy, so an in-run cycle over one step's metrics cannot produce it.
        // Executed by `skill::tests::the_skill_chart_round_trips_with_a_complete_finite_series`.
        "pretrain_skill_profile",
        // Written by `write_support_decode`, from the v4 -> v5 support upgrade in
        // `support_moments`. Indexed by DOF and by BIN rather than by step, and both need a
        // support carrying MEASURED per-bin moments, which the artifact a training run loads does
        // not have — that absence is the reason the module exists. Both are executed by
        // `support_moments::tests::the_support_decode_writes_both_registered_bases`.
        "support_decode_moments",
        "support_decode_bins",
        // Written by `write_mem_probe`, from the multi-epoch memorization probe in `mem_probe`.
        // The spine needs SEVERAL checkpoints, each scored on both splits; the contrast needs the
        // run's training pass partition rebuilt at one checkpoint's own step and split at the
        // issue cursor; the recency profile is indexed by how long ago a window was issued; and
        // the stability panel refits the same slope at six DIFFERENT bootstrap draw counts. None
        // of the four is a function of a step's metrics, so no in-run reporter cycle can produce
        // any of them. All four are executed by
        // `mem_probe::tests::the_mem_probe_writes_every_registered_base`.
        "memprobe_epoch_spine",
        "memprobe_one_repetition",
        "memprobe_recency",
        "memprobe_bootstrap_stability",
        // Written by `write_bar_family`, from the offline continuous-family gate in `bar_family`.
        // Each panel needs a whole drawn sample plus a fitted mixture battery — a component sweep
        // with a withheld holdout, upper order statistics of `|r|`, and a discrete support loaded
        // from disk to score against — none of which is a function of a step's metrics, so no
        // in-run reporter cycle can produce any of the ten. All ten are executed by
        // `bar_family::tests::the_bar_family_fit_writes_every_registered_base`.
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
        // Written by `write_bar_seams`, from the corporate-action seam audit in `split_seams`.
        // Every panel needs a STREAMING pass over all 451,507,140 stored bars plus a support
        // artifact loaded from disk, and the tail pair needs the 4M fitting draw read twice — once
        // untouched and once with the classified seam rows joined out by `(series, bar)`. None of
        // that is a function of a step's metrics, so no in-run reporter cycle can produce any of the
        // six. All six are executed by
        // `split_seams::tests::the_seam_audit_writes_every_registered_base`.
        "bar_seam_census",
        "bar_seam_ratios",
        "bar_seam_context",
        "bar_seam_tail_r",
        "bar_seam_bin_mass",
        "bar_seam_ruin_licence",
        // Written by `write_heldout_power`, from the window draw `pretrain-calibration` performs
        // BEFORE it opens a checkpoint. Both panels are functions of the stored bars and of a
        // seed-pinned draw and no model is involved, so no in-run reporter cycle can produce
        // either: the point of the pass is that nothing has been scored yet. Both are executed by
        // `the_heldout_power_census_writes_both_registered_bases`.
        "pretrain_heldout_census",
        "pretrain_heldout_power",
    ];

    static SCRATCH_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn scratch_dir(name: &str) -> PathBuf {
        let unique = SCRATCH_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "pretrain_reports_{name}_{}_{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// The two discrete rules both start a zero-init head at `BAR_DOF * ln(NUM_BAR_BINS)`.
    /// The density rule adds the measure term, which is why the mode-aware line lives on
    /// [`HeldOutBaselines`] and this one is explicitly the CATEGORICAL figure.
    #[test]
    fn uniform_baseline_is_five_log_one_twenty_eight() {
        // 5 * ln(128) = 24.2601513..., not the 24.2536 quoted in the brief.
        assert!((uniform_categorical_nll_bar() - 24.260_151_3).abs() < 1.0e-6);
        assert_eq!(uniform_categorical_nll_bar(), 5.0 * 128.0f64.ln());
    }

    #[test]
    fn series_nan_pads_skipped_ticks() {
        let mut series = Series::default();
        series.set(0, 1.0);
        series.set(3, 4.0);
        series.set(4, f64::NAN);
        let values = series.padded(6);
        assert_eq!(values[0], 1.0);
        assert!(values[1].is_nan() && values[2].is_nan());
        assert_eq!(values[3], 4.0);
        assert!(values[4].is_nan() && values[5].is_nan());
    }

    #[test]
    fn mean_ignores_non_finite_samples() {
        let mut mean = Mean::default();
        mean.push(f64::NAN);
        mean.push(2.0);
        mean.push(4.0);
        mean.push(f64::INFINITY);
        assert_eq!(mean.value(), 3.0);
        assert!(Mean::default().value().is_nan());
    }

    #[test]
    fn pit_density_is_flat_for_a_uniform_transform() {
        let rows = 4096i64;
        let uniform = Tensor::arange(rows, (Kind::Float, Device::Cpu)) / rows as f64;
        let pit = uniform.unsqueeze(-1).repeat([1, BAR_DOF as i64]);
        let mut histogram = PitHistogram::default();
        histogram.accumulate(&pit);
        assert!(!histogram.is_empty());
        for dof in histogram.density() {
            for bin in dof {
                assert!((bin - 1.0).abs() < 1.0e-6, "expected flat PIT, got {bin}");
            }
        }
        histogram.reset();
        assert!(histogram.is_empty());
    }

    #[test]
    fn pit_clamps_out_of_range_values_into_the_edge_bins() {
        let pit =
            Tensor::from_slice(&[-0.5f32, 1.5, 0.5, f32::NAN, 0.0]).reshape([1, BAR_DOF as i64]);
        let mut histogram = PitHistogram::default();
        histogram.accumulate(&pit);
        let density = histogram.density();
        assert_eq!(density[0][0], PIT_HIST_BINS as f64);
        assert_eq!(density[1][PIT_HIST_BINS - 1], PIT_HIST_BINS as f64);
        assert_eq!(density[2][PIT_HIST_BINS / 2], PIT_HIST_BINS as f64);
        assert!(density[3][0].is_nan());
        assert_eq!(density[4][0], PIT_HIST_BINS as f64);
    }

    #[test]
    fn effective_rank_spans_collapse_to_isotropy() {
        let _torch_rng_guard = test_rng::shared();
        let device = Device::Cpu;
        let rank_one = Tensor::arange(64, (Kind::Float, device))
            .unsqueeze(-1)
            .matmul(&Tensor::ones([1, 8], (Kind::Float, device)));
        let collapsed = belief_effective_rank(&rank_one);
        assert!(
            (collapsed - 1.0).abs() < 1.0e-3,
            "rank-one beliefs should score 1, got {collapsed}"
        );

        let isotropic = Tensor::randn([8192, 8], (Kind::Float, device));
        let spread = belief_effective_rank(&isotropic);
        assert!(
            spread > 7.0 && spread <= 8.0 + 1.0e-6,
            "isotropic beliefs should approach the width, got {spread}"
        );

        assert!(belief_effective_rank(&Tensor::zeros([1, 8], (Kind::Float, device))).is_nan());
    }

    #[test]
    fn chained_candles_stay_ordered_and_anchored() {
        let path = [
            0.01f32, 0.02, 0.25, 0.75, 0.0, //
            -0.03, 0.05, 0.9, 0.1, 0.5,
        ];
        let candles = chained_candles(&path);
        assert_eq!(candles.len(), 2);
        for candle in &candles {
            assert!(candle.low <= candle.open.min(candle.close));
            assert!(candle.high >= candle.open.max(candle.close));
        }
        assert!((candles[0].close as f64 - 0.01f64.exp()).abs() < 1.0e-5);
        assert!(
            (candles[1].close as f64 - (0.01f64 - 0.03).exp()).abs() < 1.0e-5,
            "the second bar must chain off the first close"
        );
    }

    #[test]
    fn quantiles_interpolate_between_order_statistics() {
        let sorted = [0.0f32, 1.0, 2.0, 3.0];
        assert_eq!(quantile_sorted(&sorted, 0.0), 0.0);
        assert_eq!(quantile_sorted(&sorted, 1.0), 3.0);
        assert_eq!(quantile_sorted(&sorted, 0.5), 1.5);
        assert!((quantile_sorted(&sorted, 0.1) - 0.3).abs() < 1.0e-6);
        assert!(quantile_sorted(&[], 0.5).is_nan());
    }

    /// A deterministic standard normal, so a fixture that has to resolve an effect of
    /// known size does not depend on the process-wide torch RNG that any other test in
    /// the binary may reseed.
    struct Gaussian(u64);

    impl Gaussian {
        fn unit(&mut self) -> f64 {
            self.0 = mix64(self.0, 0x9E37_79B9);
            // 53 bits, open at both ends: Box-Muller must never see an exact zero.
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let u = self.unit();
            let v = self.unit();
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }
    }

    /// `[windows, samples, steps, BAR_DOF]` of FLAT bars whose log returns are
    /// `N(-sigma^2/2, sigma^2)`: `s = 0` forces `u = v = 0.5`, so `decode_dof` maps the
    /// row to a zero-range bar at `prev * exp(r)` and the chained close path is exactly
    /// the exponential of a Gaussian random walk. `E[exp(r)] = 1`, so the process is a
    /// martingale and the analytic median of its close at horizon `t` is
    /// `exp(-(t + 1) sigma^2 / 2)` — nothing else.
    fn martingale_rollout(
        windows: i64,
        samples: i64,
        steps: i64,
        sigma: f64,
        seed: u64,
    ) -> Tensor {
        cap_torch_threads();
        let mut rng = Gaussian(seed);
        let mut values = Vec::with_capacity((windows * samples * steps) as usize * BAR_DOF);
        for _ in 0..windows * samples * steps {
            let r = -0.5 * sigma * sigma + sigma * rng.normal();
            values.extend([r as f32, 0.0, 0.5, 0.5, 0.0]);
        }
        Tensor::from_slice(&values).view([windows, samples, steps, BAR_DOF as i64])
    }

    /// Cap libtorch's INTRA-OP pool, once per test process.
    ///
    /// Every heavy fixture in this module goes through one of the two helpers that call
    /// this, so the cap cannot be forgotten by a test added later. It is needed because
    /// libtorch sizes its intra-op pool by PHYSICAL cores, measured at 12 on this box, and
    /// a SINGLE forward pass through the 10-layer, 512-wide trunk fans out across all of
    /// them; the libtest harness's `--test-threads` bounds none of it. Measured on an
    /// uncapped module: one heavy test costs 5.8x the CPU-seconds unpinned (9.1 cores
    /// against 1.1) to buy a 1.3x wall speedup, which is a bad trade alone on the box and
    /// a hostile one when sharing it. Intra-op is not the whole bill - interop and rayon
    /// add to it - so this caps the largest term, not every term.
    ///
    /// `TORCH_NUM_THREADS` is this repo's own convention, not a libtorch variable —
    /// nothing in tch or libtorch reads it — so it binds ONLY where repo code calls
    /// `tch::set_num_threads`: here, in `pretrain::configure_threads`, and in the
    /// `#[cfg(test)]` pre-main constructor in `lib.rs`. Measured in a binary without that
    /// constructor, `TORCH_NUM_THREADS=1` is a NO-OP and the pool stays at 12.
    /// `OMP_NUM_THREADS` binds the complementary case, any path that never calls
    /// `set_num_threads`. Neither subsumes the other, and an explicit call overrides OMP
    /// in BOTH directions, so the two must never be given conflicting values.
    ///
    /// Defaults to ONE, because an unset environment must not be the path that quietly
    /// costs cores on a shared workstation, and never RAISES the pool: the ceiling is
    /// taken against `get_num_threads()` so this is a monotone lowering even when
    /// something outside the environment already pinned the pool - a pre-main
    /// constructor, a nested harness, or a fixture that ran first and set 1 explicitly.
    /// Computing a small number and writing it unconditionally would raise those cases
    /// back up.
    fn cap_torch_threads() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let ceiling = std::env::var("TORCH_NUM_THREADS")
                .ok()
                .and_then(|value| value.parse::<i32>().ok())
                .unwrap_or(1)
                .clamp(1, 4);
            tch::set_num_threads(ceiling.min(tch::get_num_threads()).max(1));
        });
    }

    fn write_windows(name: &str, rollout: &Tensor, future: &Tensor) -> Vec<CandleWindow> {
        cap_torch_threads();
        let dir = scratch_dir(name);
        let out = write_candle_windows(&dir, 7, None, rollout, future).expect("snapshot writes");
        fs::remove_dir_all(&dir).ok();
        out
    }

    /// A poisoned fan must read as UNMEASURED, not as a large drift.
    ///
    /// `f32::max` returns the non-NaN operand, so the obvious `centre.max(1e-12)` floor
    /// turns a NaN fan centre into `ln(1e-12) / steps` — at 100 bars that is -0.276 per
    /// bar, four orders above anything real and entirely plausible-looking on a chart.
    /// The scalars must be NaN so the reporter's own filter drops the point.
    #[test]
    fn a_poisoned_fan_reports_nothing_rather_than_a_plausible_number() {
        let steps = 8usize;
        let clean = CandleWindow {
            actual_close: vec![1.0; steps],
            quantiles: array::from_fn(|q| vec![0.9 + 0.05 * q as f32; steps]),
            rank: vec![0.5; steps],
            samples: 256,
        };
        assert!(clean.drift_per_bar().is_finite());
        assert!(clean.centre_log_se(steps - 1) > 0.0);

        for poison in [f32::NAN, 0.0, -1.0] {
            let mut broken = clean.clone();
            for row in broken.quantiles.iter_mut() {
                row[steps - 1] = poison;
            }
            assert!(
                broken.drift_per_bar().is_nan(),
                "a fan centre of {poison} produced a drift of {}",
                broken.drift_per_bar()
            );
            assert!(broken.drift_per_bar_se().is_nan());
            let summary = CandleSummary::from_windows(&[broken, clean.clone()]);
            // The one measurable window still reports, and it reports ITS OWN spread:
            // a single finite draw has no across-window variance, so the se is NaN
            // rather than a zero that would claim certainty.
            assert!(summary.dclose.is_finite(), "{poison}: the clean window vanished");
            assert!(summary.dclose_se.is_nan(), "{poison}: one window claimed a spread");
            assert!(
                summary.band.is_finite(),
                "{poison}: the clean window's band width vanished"
            );
        }

        // A zero-sample fan cannot state an error bar at all.
        let empty = CandleWindow {
            samples: 0,
            ..clean.clone()
        };
        assert!(empty.centre_log_se(0).is_nan());
        assert!(empty.drift_per_bar_se().is_nan());
    }

    /// A quantile fan that crosses itself is not a fan, and a renderer that nests the
    /// bands would draw a lie. Monotonicity in probability has to hold at EVERY horizon,
    /// not on average over them.
    #[test]
    fn fan_quantiles_are_monotone_in_probability_at_every_horizon() {
        let (windows, samples, steps) = (3i64, 64i64, 40i64);
        let rollout = martingale_rollout(windows, samples, steps, 0.02, 0xFA_0001);
        let future = martingale_rollout(windows, 1, steps, 0.02, 0xFA_0002).squeeze_dim(1);
        let fans = write_windows("fan_monotone", &rollout, &future);
        assert_eq!(fans.len(), windows as usize);
        for (index, fan) in fans.iter().enumerate() {
            assert_eq!(fan.steps(), steps as usize);
            for t in 0..fan.steps() {
                for q in 1..FAN_QUANTILES.len() {
                    let (lower, upper) = (fan.quantiles[q - 1][t], fan.quantiles[q][t]);
                    assert!(
                        lower <= upper,
                        "window {index} bar {t}: p{:.0} = {lower} exceeds p{:.0} = {upper}",
                        FAN_QUANTILES[q - 1] * 100.0,
                        FAN_QUANTILES[q] * 100.0
                    );
                }
                assert!(fan.quantiles.iter().all(|row| row[t].is_finite()));
                assert!((0.0..=1.0).contains(&fan.rank[t]));
            }
            // The band has to actually widen with the horizon, or the fixture is
            // asserting monotonicity on a degenerate fan of coincident lines.
            assert!(
                fan.p90()[fan.steps() - 1] - fan.p10()[fan.steps() - 1]
                    > 2.0 * (fan.p90()[0] - fan.p10()[0]),
                "window {index}: the fan did not widen"
            );
        }
    }

    /// The draws come from the model's sampler and move with `--seed`. WHICH of them a
    /// window overlays must not, or two runs' pictures differ for a reason that has
    /// nothing to do with either model.
    #[test]
    fn overlaid_sample_paths_are_reproducible_under_the_pinned_seed() {
        let samples = 64usize;
        for window in 0..4usize {
            let chosen = overlay_indices(window, samples);
            assert_eq!(chosen.len(), SNAPSHOT_OVERLAY_PATHS);
            assert_eq!(
                chosen,
                overlay_indices(window, samples),
                "the overlay must be a pure function of the pinned seed and the window"
            );
            assert!(chosen.windows(2).all(|pair| pair[0] < pair[1]), "{chosen:?}");
            assert!(chosen.iter().all(|index| *index < samples));
        }
        // Distinct windows must not all show the same five draws, which a seed mixed on
        // the campaign constant alone would produce.
        assert_ne!(overlay_indices(0, samples), overlay_indices(1, samples));
        // Fewer draws than the overlay wants is a legitimate configuration, not a panic,
        // and it must still terminate.
        assert_eq!(overlay_indices(0, 3), vec![0, 1, 2]);

        // End to end, on a fan wide enough for the overlay to be a genuine subset: 64
        // draws, of which the pinned seed shows five.
        let steps = 40i64;
        let rollout = martingale_rollout(1, samples as i64, steps, 0.02, 0xFA_0011);
        let future = martingale_rollout(1, 1, steps, 0.02, 0xFA_0012).squeeze_dim(1);
        // End to end: the same rollout written twice puts identical draws on the chart.
        let read_back = |name: &str| -> Vec<Vec<f32>> {
            let dir = scratch_dir(name);
            write_candle_windows(&dir, 3, Some(2), &rollout, &future).expect("snapshot writes");
            let report =
                read_report(&dir.join("step3_epoch002_window01_fan.report.bin")).unwrap();
            fs::remove_dir_all(&dir).ok();
            let ReportKind::CandleFan { samples, .. } = report.kind else {
                panic!("expected CandleFan");
            };
            samples.into_iter().map(|series| series.values).collect()
        };
        let first = read_back("overlay_a");
        assert_eq!(first.len(), SNAPSHOT_OVERLAY_PATHS);
        assert_eq!(first, read_back("overlay_b"));

        // And they are genuine draws, not the fan centre: a real ancestral path is
        // rougher than a locus of per-horizon medians estimated from 32 of them.
        let fan = &write_windows("overlay_c", &rollout, &future)[0];
        let roughness = |path: &[f32]| -> f64 {
            path.windows(2)
                .map(|pair| ((pair[1] / pair[0]) as f64).abs().ln().abs())
                .sum()
        };
        assert!(
            roughness(&first[0]) > roughness(fan.fan_centre()),
            "the overlay is smoother than the fan centre, so it is not a draw"
        );
    }

    /// THE FIXTURE THAT WOULD HAVE CAUGHT THE MISREADING.
    ///
    /// Feed the writer an exact martingale of known volatility. The only median drift
    /// such a process has is the multiplicative `-sigma^2/2`, so anything the reported
    /// `dclose` shows beyond that came from the reporting path and not from the model.
    ///
    /// It also pins the scale of the estimator's own noise, which is the number whose
    /// absence made a fan-centre wiggle look like a bias: at `sigma = 0.1` over 4096
    /// draws the drift is `-5.0e-3` per bar and the standard error of measuring it is
    /// `1.2533 * sigma / sqrt(samples * steps) = 4.9e-4`, a ratio of ten. Shrink `sigma`
    /// to the 2.4e-3 the real snapshots run at and the ratio inverts: at
    /// [`SNAPSHOT_SAMPLES`] draws over 100 bars the drift is 2.9e-6 against a standard
    /// error of 1.9e-5, so the effect sits about 7x BELOW the noise of the statistic that
    /// was supposed to reveal it. Per BAR, which is how the chart reads it, the same
    /// numbers are 3.6e-6 against 2.1e-4: 58x below.
    #[test]
    fn a_synthetic_martingale_shows_only_its_analytic_median_drift() {
        let (samples, steps, sigma) = (4096i64, 16i64, 0.1f64);
        let rollout = martingale_rollout(1, samples, steps, sigma, 0xFA_0005);
        let future = martingale_rollout(1, 1, steps, sigma, 0xFA_0006).squeeze_dim(1);
        let fan = &write_windows("martingale", &rollout, &future)[0];

        let analytic = -0.5 * sigma * sigma;
        // Standard error of the sample median of a Gaussian, propagated through the
        // telescoping mean increment: `1.2533 * sigma * sqrt(steps) / sqrt(samples)`,
        // divided by `steps`.
        let se = 1.2533 * sigma / ((samples * steps) as f64).sqrt();
        let observed = fan.drift_per_bar();
        assert!(
            (observed - analytic).abs() < 4.0 * se,
            "per-bar median drift {observed:.3e} is not the analytic {analytic:.3e} \
             (se {se:.3e}); the reporting path is adding drift of its own"
        );
        assert!(
            analytic.abs() > 4.0 * se,
            "the fixture cannot resolve the effect it is asserting: |{analytic:.3e}| vs \
             se {se:.3e}"
        );
        // The fan's own error bar has to agree with the analytic one, because that is
        // the number the chart now prints and a reader will divide by.
        let reported = fan.drift_per_bar_se();
        assert!(
            reported > 0.4 * se && reported < 2.5 * se,
            "the fan reports a per-bar drift se of {reported:.3e} against the analytic \
             {se:.3e}"
        );

        // Every horizon, not just the endpoint: the median of the cumulative path at
        // `t` is `-(t + 1) sigma^2 / 2` and nothing accumulates on top of it.
        for t in 0..fan.steps() {
            let expected = analytic * (t + 1) as f64;
            let observed = (fan.fan_centre()[t] as f64).ln();
            let se_t = 1.2533 * sigma * ((t + 1) as f64).sqrt() / (samples as f64).sqrt();
            assert!(
                (observed - expected).abs() < 4.0 * se_t,
                "bar {t}: fan centre {observed:.3e} against analytic {expected:.3e} \
                 (se {se_t:.3e})"
            );
        }

        // A martingale is calibrated by construction, so the realized path's rank
        // inside the fan must be a draw from the uniform law rather than pinned to an
        // edge, and the terminal in-band rate must not be systematically short.
        let mean_rank = fan.rank.iter().map(|value| *value as f64).sum::<f64>() / steps as f64;
        assert!(
            (0.05..=0.95).contains(&mean_rank),
            "a calibrated martingale should not sit at the edge of its own fan: {mean_rank}"
        );
    }

    /// A genuine 100-step ancestral rollout, on CPU, through the real trunk, cache and
    /// emission head — the path the snapshot writer is fed in a run.
    ///
    /// Deepening the horizon from 64 to 100 puts 36 more sequential decode steps through
    /// the KV cache and 36 more chained `decode_dof` calls. Both are places where a
    /// silent non-finite or an inverted bar would first appear, and neither is exercised
    /// by any teacher-forced test: at zero init the head is uniform over all 128 bins per
    /// DOF, so the draws span the whole fitted support including its extreme `r` and `s`
    /// bins, which is the adversarial case for the chaining.
    #[test]
    fn a_hundred_bar_ancestral_rollout_stays_finite_and_decodes_to_valid_bars() {
        use crate::torch::dataset::{BAR_TIME_CARDINALITY, BAR_TIME_FEATURES};
        use crate::torch::world_model::{BarKvCache, BarModules, BarSupportSet, BAR_MAX_CONTEXT};
        use tch::nn::VarStore;
        let _torch_rng_guard = test_rng::shared();

        // The heaviest fixture in the module: the real trunk, cache and emission head.
        cap_torch_threads();

        let steps = *ROLLOUT_HORIZONS.last().expect("a deepest horizon") as i64;
        assert_eq!(steps, 100, "this fixture is the guard on the 100-bar depth");
        // The PRODUCTION geometry, kept: `pretrain::SNAPSHOT_HORIZON` holds the last
        // `steps` bars of the diagnostic context out as the realized continuation, so the
        // conditioning prefill is what is left. The KV-footprint assertion below is a
        // statement about the run only because this is the run's own width; shrinking it
        // to make the fixture cheaper would void exactly the invariant it carries. The
        // cost is bounded by the thread cap above, not by a smaller model.
        let history = DIAGNOSTIC_CONTEXT - steps;

        let mut rng = Gaussian(0xFA_0007);
        let fitted: Vec<BarDof> = (0..8192)
            .map(|index| {
                if index % 32 == 0 {
                    return BarDof::default();
                }
                BarDof {
                    r: (0.006 * rng.normal()) as f32,
                    s: (0.004 * rng.unit()) as f32,
                    u: rng.unit() as f32,
                    v: rng.unit() as f32,
                    w: (1.4 * rng.normal()) as f32,
                }
            })
            .collect();
        let supports = BarSupports::fit(&fitted);
        let set = BarSupportSet::new(vec![(300, supports)]).expect("support set");

        let history_dof = Tensor::from_slice(
            &fitted[..history as usize]
                .iter()
                .flat_map(|dof| dof.to_array())
                .collect::<Vec<f32>>(),
        )
        .view([1, history, BAR_DOF as i64]);
        // History bars carry a market row; the imagined future carries MARKET_MISSING, which is
        // exactly what `future_time_ids` hands a real rollout.
        let time_ids = |len: i64, start: i64, market: i64| -> Tensor {
            let values: Vec<i64> = (0..len)
                .flat_map(|t| {
                    let mut ids = [0i64; BAR_TIME_FEATURES];
                    ids[0] = (start + 5 * t) % BAR_TIME_CARDINALITY[0];
                    ids[1] = t % BAR_TIME_CARDINALITY[1];
                    ids[2] = 2;
                    ids[4] = 1;
                    ids[5] = 1;
                    for channel in 6..BAR_TIME_FEATURES {
                        ids[channel] = market;
                    }
                    ids
                })
                .collect();
            Tensor::from_slice(&values).view([1, len, BAR_TIME_FEATURES as i64])
        };
        let history_time_ids = time_ids(history, 570, 64);
        let future_time_ids = time_ids(steps, 600, 0);

        let vs = VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        let prefill_started = std::time::Instant::now();
        let mut cache = BarKvCache::new(BAR_MAX_CONTEXT);
        let prefill = modules.trunk.forward_cached(
            &history_dof,
            &set.bin_ids(&history_dof, &history_time_ids),
            &history_time_ids,
            &mut cache,
        );
        let prefill_elapsed = prefill_started.elapsed();
        // One prefill, then `MIN_FAN_SAMPLES` independent continuations off it, exactly as
        // `BarWorldModel::rollout` does for a snapshot window: the fan needs p25 and p75 to
        // be distinct order statistics, and a one-draw "fan" would print an error bar of
        // zero. The decode cost is per STEP, not per draw, at these widths.
        let paths = MIN_FAN_SAMPLES as i64;
        let mut cache = cache.repeat_batch(paths);
        let mut h = prefill.narrow(1, history - 1, 1).repeat([paths, 1, 1]);
        let future_ids = future_time_ids.repeat([paths, 1, 1]);
        let mut drawn = Vec::with_capacity(steps as usize);
        let decode_started = std::time::Instant::now();
        for step in 0..steps {
            let dof = modules.head.sample(&h, set.only(), 1.0);
            assert!(
                bool::try_from(dof.isfinite().all()).expect("finite"),
                "the ancestral draw at step {step} went non-finite"
            );
            let ids = future_ids.narrow(1, step, 1);
            h = modules
                .trunk
                .forward_cached(&dof, &set.bin_ids(&dof, &ids), &ids, &mut cache);
            assert!(
                bool::try_from(h.isfinite().all()).expect("finite"),
                "the belief at step {step} went non-finite, so the deeper rollout drifts"
            );
            drawn.push(dof);
        }
        let decode_elapsed = decode_started.elapsed();
        // THE MEMORY CLAIM, measured at the run's own width. Storage is allocated at
        // `next_power_of_two(prefill)` and doubles only when the length would exceed it
        // (`world_model.rs`, `BarTrunk::prefill` and `ensure_append_capacity`). Live half:
        // this rollout's ring holds 796 + 100 = 896 bars inside its 1024-token allocation
        // and never grew.
        assert_eq!(cache.cached_bars(), history + steps);
        assert!(
            cache.cached_bars() <= (history as u64).next_power_of_two() as i64,
            "the 100-bar rollout grew the KV ring past its prefill allocation: {} bars \
             against a {}-token allocation",
            cache.cached_bars(),
            (history as u64).next_power_of_two()
        );
        // Static half, so the comparison against the OLD horizon is pinned too: at 64 the
        // prefill was 832, in the same (512, 1024] bracket and therefore the same 1024-token
        // allocation, so deepening 64 -> 100 costs not one byte. The footprint is
        // `2 * BAR_LAYERS * samples * 1024 * BAR_MODEL_DIM` elements, 5.4 GB in bf16 at the
        // 256 draws a snapshot window takes.
        const DEEPEST: i64 = ROLLOUT_HORIZONS[ROLLOUT_HORIZONS.len() - 1] as i64;
        const PRODUCTION_PREFILL: i64 = DIAGNOSTIC_CONTEXT - DEEPEST;
        const _: () = assert!(PRODUCTION_PREFILL > 512 && PRODUCTION_PREFILL <= 1024);
        const _: () = assert!(PRODUCTION_PREFILL + DEEPEST <= 1024);
        const _: () = assert!(DIAGNOSTIC_CONTEXT - 64 > 512 && DIAGNOSTIC_CONTEXT - 64 + 64 <= 1024);

        // Through the writer, which is where the chaining and the fan live. `drawn[step]`
        // is `[paths, 1, BAR_DOF]`, so concatenating on the step axis gives
        // `[paths, steps, BAR_DOF]`; the fan is those paths and the realization is the
        // first of them, which is exactly the shape a snapshot window supplies.
        let stacked = Tensor::cat(&drawn, 1);
        let rollout = stacked.unsqueeze(0);
        let future = stacked.narrow(0, 0, 1);
        let fan = &write_windows("h100", &rollout, &future)[0];
        assert_eq!(fan.steps(), steps as usize);
        assert_eq!(fan.samples, paths as usize);
        for (index, horizon) in ROLLOUT_HORIZONS.iter().enumerate() {
            let t = horizon - 1;
            assert!(
                fan.quantiles.iter().all(|row| row[t].is_finite() && row[t] > 0.0),
                "horizon {horizon} (slot {index}) is not a finite positive price"
            );
            assert!(fan.rank[t].is_finite());
        }

        // Every emitted bar of every path, not just the ones a horizon lands on.
        let values = tensor_values(&stacked);
        assert_eq!(values.len(), (paths * steps) as usize * BAR_DOF);
        let bars = chained_candles(&values);
        assert_eq!(bars.len(), (paths * steps) as usize);
        for (t, bar) in bars.iter().enumerate() {
            assert!(
                bar.low <= bar.open.min(bar.close) && bar.high >= bar.open.max(bar.close),
                "bar {t} decoded to an invalid OHLC ordering: {bar:?}"
            );
            assert!(bar.low > 0.0 && bar.high.is_finite());
        }
        // Stated separately because only the decode leg scales with the horizon: the
        // prefill is paid once per window at either depth, so the 64 -> 100 change costs
        // 36 decode steps and nothing else.
        println!(
            "100-bar CPU ancestral rollout, {paths} draws: {:.0} ms prefill ({history} bars) + \
             {:.0} ms decode ({steps} steps, {:.2} ms/step); the 36 steps this deepening \
             added cost {:.0} ms",
            prefill_elapsed.as_secs_f64() * 1e3,
            decode_elapsed.as_secs_f64() * 1e3,
            decode_elapsed.as_secs_f64() * 1e3 / steps as f64,
            decode_elapsed.as_secs_f64() * 1e3 / steps as f64 * 36.0,
        );
    }

    const MARGINAL_DOF: [f64; BAR_DOF] = [4.760, 4.693, 3.920, 3.836, 4.852];

    fn checkpoint(root: &Path, name: &str, body: &[u8]) -> PathBuf {
        let path = root.join(name);
        fs::write(&path, body).expect("checkpoint fixture");
        path
    }

    fn populated_epoch(epoch: usize, step: usize, promoted: Option<PathBuf>) -> EpochMetrics {
        let mut metrics = EpochMetrics::nan();
        metrics.epoch = epoch;
        metrics.global_step = step;
        metrics.train_nll_bar = 20.0;
        metrics.val_nll_bar = 21.0;
        metrics.best_val_nll_bar = 21.0;
        metrics.val_nll_bar_diag = 21.5;
        metrics.promoted_checkpoint = promoted;
        metrics.train_nll_dof = [4.0; BAR_DOF];
        metrics.val_nll_dof = [4.2; BAR_DOF];
        metrics.val_crps_dof = [0.1; BAR_DOF];
        metrics.val_dir_acc = 0.52;
        metrics.rollout_nll_exact = [21.0, 22.0, 23.0, 24.0, 24.6];
        metrics.rollout_nll_dynamics = [21.1, 22.4, 23.9, 25.2, 26.0];
        metrics.unique_bar_reuse = 0.25;
        metrics.effective_rank = 42.0;
        metrics.val_nll_bar_conditional = 20.8;
        metrics.val_nll_bar_conditional_deployed = 20.4;
        metrics.val_nll_dof_conditional = [4.1; BAR_DOF];
        metrics.val_nll_dof_class = [1.0; BAR_DOF];
        metrics.val_nll_dof_shape = [3.2; BAR_DOF];
        metrics.val_nll_bar_se = 0.05;
        metrics.val_nll_bar_ci = (20.9, 21.1);
        metrics.val_nll_bar_se_level = 0.10;
        // Marginalizing the chain can only cost nats, so the forecast row sits above the
        // teacher-forced one on identical rows.
        metrics.val_forecast_nll_dof = [4.6; BAR_DOF];
        metrics.val_forecast_teacher_nll_dof = [4.2; BAR_DOF];
        // Three ramp stages, the last one still in progress: the shape a run reports at a
        // periodic validation mid-pass, and the shape the shortfall chart is for.
        metrics.stage_coverage = vec![1.0, 1.0, 0.31];
        metrics.stage_conditioning_bars = vec![448.5, 736.5, 1024.5];
        // A mid-pass reading: most bars still untargeted because stage 2 has not finished, a
        // handful unreachable. Deliberately NOT a completed pass, so the multiplicity chart is
        // exercised with mass in more than one bucket.
        metrics.pass_multiplicity_bars = [1_200_000, 2_800_000, 0, 0];
        metrics.pass_coverage = 2_800_000.0 / 4_000_000.0;
        metrics.pass_remainder_bars = [10_594, 0, 2_049_124, 1_140_282];
        // THE CROSS-PASS ROW, and the shape that makes the defect this panel pair exists for
        // reproducible in a test: the per-pass row above says "2 times: 0, 3+ times: 0" while
        // this one says most of the split has already been targeted THREE times. Both are
        // correct; only together are they unmisreadable. A fixture that left this at the
        // `nan()` default would make the registry walk pass on an empty panel.
        metrics.run_exposure_bars = [1_210_594, 40_000, 150_000, 2_599_406];
        metrics.run_effective_epochs = 2.8532;
        metrics.projected_effective_epochs = 3.0000625;
        metrics.planned_effective_epochs = 3.0;
        metrics.val_forecast_nll_se = 0.03;
        metrics.val_promotion_context = 2048.0;
        metrics.reached_context = 2048.0;
        metrics
            .val_pit
            .accumulate(&(Tensor::arange(512, (Kind::Float, Device::Cpu)) / 512.0)
                .unsqueeze(-1)
                .repeat([1, BAR_DOF as i64]));
        metrics.trade = populated_trade();
        metrics
    }

    /// A MEASURED bench, produced by running the real accounting over hand-made position
    /// paths. The fixture supplies positions, not distributions, because what these tests
    /// exercise is the report path; the solver has its own suite next door.
    fn populated_trade() -> TradeBench {
        let cap = 4.0;
        let windows: Vec<WindowPaths> = (0..6usize)
            .map(|window| {
                let bars = 48usize;
                let realized: Vec<f64> = (0..bars)
                    .map(|bar| if (bar + window) % 3 == 0 { 0.004 } else { -0.001 })
                    .collect();
                // The UNCAPPED log-optimal fraction is the primitive, exactly as
                // `WindowPaths::free` documents: the model leans the right way more often
                // than not, and on one bar in seven it asks for more than the cap allows, so
                // the `clamped_fraction` diagnostic has something to measure. Every capped
                // and fractional policy is then a clamp of that one vector - no second
                // solve - which is the identity the bench relies on.
                let free: Vec<f64> = realized
                    .iter()
                    .enumerate()
                    .map(|(bar, r)| {
                        let direction = if bar % 5 == 0 { -r.signum() } else { r.signum() };
                        let size = if bar % 7 == 0 { 1.5 * cap } else { 0.9 * cap };
                        direction * size
                    })
                    .collect();
                let clamped = |multiple: f64| -> Vec<f64> {
                    free.iter()
                        .map(|position| (multiple * position).clamp(-cap, cap))
                        .collect()
                };
                let model = clamped(1.0);
                let half = clamped(POLICY_KELLY_MULTIPLE[POLICY_HALF]);
                let quarter = clamped(POLICY_KELLY_MULTIPLE[POLICY_QUARTER]);
                let oracle: Vec<f64> = realized.iter().map(|r| cap * r.signum()).collect();
                WindowPaths::unmeasured(
                    realized,
                    free,
                    [model, half, quarter, vec![0.5; bars], vec![1.0; bars], oracle],
                )
            })
            .collect();
        // Two windows per block: the bootstrap needs more than one block to have an interval.
        let blocks: Vec<u64> = (0..windows.len() as u64).map(|window| window / 2).collect();
        // Hand-made tail exceedances, on the same windows. The loosest level fires a few
        // times and the tighter ones do not, which is what a well-calibrated 48-bar window
        // looks like and is enough for the calibration panel to hold real numbers instead of
        // NaN. The counts differ per window so the blocked interval has width.
        let mut tail = TailCounts::empty();
        tail.bars = vec![48.0; windows.len()];
        for level in 0..TAIL_LEVELS.len() {
            tail.lower[level] = vec![0.0; windows.len()];
            tail.upper[level] = vec![0.0; windows.len()];
        }
        let loosest = TAIL_LEVELS.len() - 1;
        tail.lower[loosest] = (0..windows.len()).map(|w| (w % 3) as f64).collect();
        tail.upper[loosest] = (0..windows.len()).map(|w| ((w + 1) % 2) as f64).collect();
        super::super::trade_bench::bench(
            &windows,
            &blocks,
            &tail,
            BenchConfig::new(DEFAULT_COST_BPS, cap, 0.5),
        )
    }

    /// Two checkpoints' worth of calibration measurement, built through the REAL bench so the
    /// writer is handed the same objects the experiment hands it.
    ///
    /// The recalibrated fraction is a genuine shrink of the free optimum, and the second point
    /// is built with a smaller `beta` so the charted trend actually has a direction.
    fn populated_calibration() -> Vec<CalibrationPoint> {
        let cap = 4.0;
        [(10_364usize, 0.85f64), (29_000usize, 0.62f64)]
            .into_iter()
            .map(|(step, beta)| {
                let windows: Vec<WindowPaths> = (0..6usize)
                    .map(|window| {
                        let bars = 48usize;
                        let realized: Vec<f64> = (0..bars)
                            .map(|bar| if (bar + window) % 3 == 0 { 0.004 } else { -0.001 })
                            .collect();
                        let free: Vec<f64> = realized
                            .iter()
                            .enumerate()
                            .map(|(bar, r)| {
                                let direction =
                                    if bar % 5 == 0 { -r.signum() } else { r.signum() };
                                direction * if bar % 7 == 0 { 1.5 * cap } else { 0.9 * cap }
                            })
                            .collect();
                        let clamped = |multiple: f64| -> Vec<f64> {
                            free.iter()
                                .map(|position| (multiple * position).clamp(-cap, cap))
                                .collect()
                        };
                        let oracle: Vec<f64> =
                            realized.iter().map(|r| cap * r.signum()).collect();
                        let mut paths = WindowPaths::unmeasured(
                            realized.clone(),
                            free.clone(),
                            [
                                clamped(1.0),
                                clamped(POLICY_KELLY_MULTIPLE[POLICY_HALF]),
                                clamped(POLICY_KELLY_MULTIPLE[POLICY_QUARTER]),
                                vec![0.5; bars],
                                vec![1.0; bars],
                                oracle,
                            ],
                        );
                        // A conditional mean that varies per bar, so the regression the writer
                        // charts is well posed rather than a division by zero.
                        paths.predicted_mean = (0..bars)
                            .map(|bar| 0.0006 * (((bar + window) as f64 * 0.11).sin() + 0.4))
                            .collect();
                        paths.predicted_var = vec![1.2e-5; bars];
                        paths.free_shrunk = Some(free.iter().map(|f| beta * f).collect());
                        paths
                    })
                    .collect();
                let blocks: Vec<u64> =
                    (0..windows.len() as u64).map(|window| window / 2).collect();
                let config = BenchConfig::new(DEFAULT_COST_BPS, cap, 0.5);
                let trade = super::super::trade_bench::bench(
                    &windows,
                    &blocks,
                    &TailCounts::empty(),
                    config,
                );
                let shrunk = super::super::trade_bench::shrunk_bench(
                    &windows,
                    &blocks,
                    config,
                    super::super::trade_bench::MeanShrink { alpha: 0.0, beta },
                )
                .expect("the fixture carries a recalibrated fraction");
                let mut bands = Vec::new();
                for source in [
                    super::super::trade_bench::BandSource::Frictionless,
                    super::super::trade_bench::BandSource::Recalibrated,
                ] {
                    for shape in super::super::trade_bench::SIZING_SHAPES {
                        if let Some(sweep) = super::super::trade_bench::band_sweep(
                            &windows,
                            &blocks,
                            config,
                            source,
                            shape,
                        ) {
                            bands.push(sweep);
                        }
                    }
                }
                let band_overlap = super::super::trade_bench::SIZING_SHAPES
                    .into_iter()
                    .filter_map(|shape| {
                        super::super::trade_bench::band_shrink_overlap(
                            &windows,
                            &blocks,
                            config,
                            shape,
                        )
                    })
                    .flatten()
                    .collect::<Vec<_>>();
                let attribution =
                    super::super::trade_bench::edge_attribution(&windows, &blocks, config);
                CalibrationPoint {
                    label: format!("checkpoint_{step}"),
                    step,
                    gates: Vec::new(),
                    nll_bar: -9.0 - beta,
                    nll_bar_conditional: -9.3 - beta,
                    eval: trade.calibration,
                    fit: trade.calibration,
                    trade,
                    shrunk,
                    hysteresis: super::super::trade_bench::hysteresis_sweep(
                        &windows,
                        &blocks,
                        config,
                        super::super::trade_bench::ConvictionAxis::Raw,
                    ),
                    composition: super::super::trade_bench::hysteresis_composition(
                        &windows,
                        &blocks,
                        config,
                        1.0,
                        super::super::trade_bench::ConvictionAxis::Raw,
                    ),
                    bands,
                    band_overlap,
                    attribution,
                    fit_attribution: None,
                    decay: super::super::trade_bench::signal_decay(&windows, &blocks),
                }
            })
            .collect()
    }

    /// All NINE calibration bases land on disk with finite values, which is the coverage their
    /// [`CYCLE_EXEMPT`] entries name. An exemption whose writer no test executes is how a
    /// permanently blank panel ships.
    #[test]
    fn the_calibration_experiment_writes_every_registered_base() {
        let root = scratch_dir("mean_calibration");
        let points = populated_calibration();
        assert!(
            points[0].eval.measured(),
            "the fixture must carry conditional moments, or the chart is all NaN"
        );
        write_mean_calibration(&root, "fixture", &points).expect("the calibration writes");

        assert!(
            !points[0].bands.is_empty() && !points[0].band_overlap.is_empty(),
            "the fixture must carry a band sweep, or the band base is never exercised"
        );
        assert!(
            points[0].attribution.measured() && points[0].attribution.panel.measured(),
            "the fixture must carry the edge attribution and its panel, or three bases are \
             never exercised"
        );
        for base in [
            "pretrain_mean_calibration",
            "pretrain_shrunk_policy",
            "pretrain_no_trade_band",
            "pretrain_edge_attribution",
            "pretrain_edge_panel",
            "pretrain_edge_confidence",
            "pretrain_edge_hysteresis",
            "pretrain_signal_decay",
            "pretrain_edge_composition",
        ] {
            assert!(
                EXPECTED_BASES.contains(&base),
                "{base} is written but not registered, so the TUI never scans for it"
            );
            assert!(
                CYCLE_EXEMPT.contains(&base),
                "{base} cannot be produced by an in-run cycle, so it must be exempt WITH this \
                 test named in its entry"
            );
            let path = root.join(format!("{base}.report.bin"));
            assert!(path.exists(), "{base} was never written");
            let report = read_report(&path).expect("report reads back");
            let ReportKind::MultiLine { series } = report.kind else {
                panic!("{base} must be a multi-line chart");
            };
            assert!(
                series.iter().any(|s| s.values.iter().any(|v| v.is_finite())),
                "{base} holds no finite value"
            );
            // Every series shares the base's x-axis, or the reader is aligning two different
            // grids on one picture.
            let expected = match base {
                "pretrain_mean_calibration" => points.len(),
                "pretrain_no_trade_band" => SIZING_KNOBS,
                "pretrain_edge_panel" => points.len(),
                "pretrain_edge_attribution" => ATTRIBUTION_ARMS,
                "pretrain_edge_confidence" => ATTRIBUTION_DECILES,
                "pretrain_edge_hysteresis" => HYSTERESIS_MARGINS.len(),
                "pretrain_signal_decay" => DECAY_HORIZONS.len(),
                "pretrain_edge_composition" => COMPOSITION_NAMES.len(),
                _ => CAP_GRID.len(),
            };
            for line in &series {
                assert_eq!(
                    line.values.len(),
                    expected,
                    "{base} series `{}` has {} points against an axis of {expected}",
                    line.label,
                    line.values.len()
                );
            }
        }

        // The trend chart has to carry the STEP axis and the perfect-calibration reference,
        // because a slope of 0.62 means nothing to a reader who cannot see where 1.0 is.
        let trend = read_report(&root.join("pretrain_mean_calibration.report.bin"))
            .expect("reads back");
        let ReportKind::MultiLine { series } = trend.kind else {
            panic!("multi-line");
        };
        let step = series
            .iter()
            .find(|s| s.label == "step")
            .expect("the step axis is charted");
        assert_eq!(step.values, vec![10_364.0, 29_000.0]);
        assert!(series.iter().any(|s| s.label == "perfect calibration"));
        let beta = series
            .iter()
            .find(|s| s.label == "beta, mean (traded)")
            .expect("the mean slope is charted");
        assert!(beta.values.iter().all(|v| v.is_finite()));

        // The policy chart must state both sides of the comparison at every cap: a chart with
        // only the shrunk column cannot answer whether shrinking helped.
        let policy =
            read_report(&root.join("pretrain_shrunk_policy.report.bin")).expect("reads back");
        let ReportKind::MultiLine { series } = policy.kind else {
            panic!("multi-line");
        };
        for point in &points {
            for suffix in [
                "edge unshrunk (bps/bar)",
                "edge SHRUNK (bps/bar)",
                "sharpe unshrunk",
                "sharpe SHRUNK",
                "break-even unshrunk (bps)",
                "break-even SHRUNK (bps)",
                "mean |f| unshrunk",
                "mean |f| SHRUNK",
                "capped share unshrunk",
                "capped share SHRUNK",
                "turnover/bar unshrunk",
                "turnover/bar SHRUNK",
                "max drawdown unshrunk",
                "max drawdown SHRUNK",
            ] {
                let label = format!("{} {suffix}", point.label);
                assert!(
                    series.iter().any(|s| s.label == label),
                    "`{label}` is not charted, so the comparison is incomplete"
                );
            }
        }
        // And the verdict line has to state the recovery fraction, which is the whole point.
        let lines = calibration_verdict_lines(&points);
        assert!(
            lines.iter().any(|line| line.contains("VERDICT")),
            "the console summary must state a verdict: {lines:?}"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// The cost panel is the chart the economic verdict is read off, and it has to carry the
    /// fraction being quoted rather than only the clamp's own opinion. The values must be the
    /// bench's own per-policy curves in basis points, not a re-derivation.
    #[test]
    fn the_cost_curve_panel_carries_every_kelly_fraction() {
        let trade = populated_trade();
        let series = cost_curve_series(&trade, "val");
        let fractions = (0..POLICY_COUNT)
            .filter(|policy| POLICY_KELLY_MULTIPLE[*policy].is_finite())
            .count();
        // The cost axis, one curve per fraction, and the zero line.
        assert_eq!(series.len(), fractions + 2);
        assert_eq!(series[0].label, "cost (bps)");
        assert_eq!(series[series.len() - 1].label, "no edge");
        for point in &series {
            assert_eq!(point.values.len(), COST_GRID_BPS.len());
        }
        for policy in [POLICY_MODEL, POLICY_HALF, POLICY_QUARTER] {
            let label = format!("val edge, {}", POLICY_NAMES[policy]);
            let curve = series
                .iter()
                .find(|point| point.label == label)
                .unwrap_or_else(|| panic!("{label} is not charted"));
            for (slot, value) in curve.values.iter().enumerate() {
                assert!(
                    (*value - (trade.cost_curve[policy][slot] * 1e4) as f32).abs() < 1e-9,
                    "{label} slot {slot} is not the bench's own number"
                );
            }
        }
        // The standalone path draws the same object with the pass tag dropped, and nothing
        // else: a second picture of one bench is how the two writers drifted before.
        let standalone = cost_curve_series(&trade, "");
        assert_eq!(standalone.len(), series.len());
        for (tagged, plain) in series.iter().zip(&standalone) {
            assert_eq!(tagged.values, plain.values);
        }
    }

    /// One epoch boundary's row, with the budget numbers of a run that is ON PLAN: a full
    /// pass delivered, and a projection that lands exactly on what was requested.
    ///
    /// `FULL_PASS` is small enough to hand-check and the arithmetic is exact in f64, so a
    /// test can assert the fractions rather than approximate them.
    const FULL_PASS: u64 = 1_000_000;
    const REQUESTED_EPOCHS: u64 = 3;

    fn populated_boundary(epoch: usize, step: usize) -> EpochBoundary {
        let delivered = FULL_PASS * (epoch as u64 + 1);
        EpochBoundary {
            epoch,
            global_step: step,
            epoch_bar_tokens: FULL_PASS,
            full_pass_bar_tokens: FULL_PASS,
            run_bar_tokens: delivered,
            run_target_bar_tokens: FULL_PASS * REQUESTED_EPOCHS,
            projected_run_bar_tokens: FULL_PASS * REQUESTED_EPOCHS,
            epoch_secs: 2280.0,
            boundary_secs: 21.0,
            bench_secs: 14.0,
            snapshot_secs: 6.0,
            val_nll_bar: 21.5 - epoch as f64 * 0.1,
            forecast_nll_bar: 22.4 - epoch as f64 * 0.1,
            teacher_forcing_inflation: 0.9,
            dyn_vs_identity: 0.87,
            trade: populated_trade(),
        }
    }

    #[test]
    fn a_full_cycle_writes_every_registered_base() {
        let _torch_rng_guard = test_rng::shared();
        let root = scratch_dir("full_cycle");
        let mut reporter = PretrainReporter::new(&root, MARGINAL_DOF);

        for step in 0..STEP_DECIMATION {
            let mut metrics = StepMetrics::nan();
            metrics.epoch = 0;
            metrics.step = step;
            metrics.nll_bar = 24.0 - step as f64 * 0.01;
            metrics.nll_dof = [4.8; BAR_DOF];
            metrics.dyn_loss = 0.5;
            metrics.kl_loss = 0.25;
            metrics.total_loss = 24.75;
            // Set because a real step always sets them: the growth term is computed and
            // charted on BOTH ablation arms, so a fixture that left them NaN would make the
            // registry walk pass only because `write_chart` skips an all-NaN panel.
            metrics.growth_loss = -5.2e-4;
            metrics.growth_share = 1.1e-4;
            metrics.growth_abs_f = 2.1;
            metrics.growth_clamp_bind = 0.82;
            metrics.lr_mult = 1.0;
            metrics.muon_momentum = 0.85;
            metrics.grad_norm = 3.5;
            metrics.context = 896;
            metrics.batch_size = 16;
            metrics.bars_seen = 1_000_000 * (step as u64 + 1);
            metrics.free_vram_gib = 11.1;
            metrics.bar_tokens = (metrics.batch_size * metrics.context as usize) as f64;
            metrics.projected_footprint_gib = 16.69;
            metrics.capacity_ceiling_gib = 26.9;
            // A real corpus never covers every bar, so the fixture does not either: an
            // all-observed fixture would pass the registry walk on a constant 100.
            metrics.market_total_bars = (metrics.batch_size * (metrics.context as usize + 1)) as u64;
            metrics.market_missing_bars = metrics.market_total_bars / 20;
            reporter.record_step(&metrics).unwrap();
        }

        let windows = 2i64;
        let samples = 8i64;
        let horizon = 4i64;
        let rollout = Tensor::rand(
            [windows, samples, horizon, BAR_DOF as i64],
            (Kind::Float, Device::Cpu),
        ) * 0.01;
        let future = Tensor::rand(
            [windows, horizon, BAR_DOF as i64],
            (Kind::Float, Device::Cpu),
        ) * 0.01;
        reporter
            .record_snapshot(&SnapshotInput {
                rollout: &rollout,
                future_dof: &future,
                epoch: 0,
                global_step: STEP_DECIMATION,
            })
            .unwrap();
        let weights = checkpoint(&root, "pretrain_best.ot", b"promoted-weights");
        reporter
            .record_epoch(&populated_epoch(0, STEP_DECIMATION, Some(weights.clone())))
            .unwrap();
        // The epoch-indexed panel is written from its own axis, so a full cycle has to
        // cross an epoch boundary or three registered bases have no writer.
        reporter
            .record_epoch_boundary(&populated_boundary(0, STEP_DECIMATION))
            .unwrap();

        let dir = root.join("0");
        for exempt in CYCLE_EXEMPT {
            assert!(
                EXPECTED_BASES.contains(exempt),
                "{exempt} is exempted from the cycle walk but is not a registered base at \
                 all; the exemption is now covering for nothing and hiding whatever \
                 replaced it"
            );
        }
        for base in EXPECTED_BASES.iter().filter(|b| !CYCLE_EXEMPT.contains(b)) {
            let path = dir.join(format!("{base}.report.bin"));
            assert!(path.exists(), "{base} was never written");
            let report = read_report(&path).expect("report reads back");
            match report.kind {
                ReportKind::MultiLine { series } => assert!(
                    series
                        .iter()
                        .any(|s| s.values.iter().any(|v| v.is_finite())),
                    "{base} holds no finite value"
                ),
                other => panic!("{base} has unexpected kind {other:?}"),
            }
        }

        // And the CONVERSE, which is the half that kept shipping: a chart this module
        // writes but nobody registered is invisible in the TUI and nothing else would
        // notice. Walking what actually landed on disk catches it at the moment the
        // writer is added, without scraping anyone's source.
        for entry in fs::read_dir(&dir).expect("the cycle wrote a generation directory") {
            let path = entry.expect("generation dir entry").path();
            let Some(base) = path
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| name.strip_suffix(".report.bin"))
            else {
                continue;
            };
            assert!(
                EXPECTED_BASES.contains(&base),
                "{base} was written but is not in shared::report::PRETRAIN_REPORT_BASES, so \
                 the TUI never scans for it and the chart is invisible; add it there"
            );
        }

        let snapshots = dir.join("candle_snapshots");
        for window in 1..=windows {
            let path = snapshots.join(format!(
                "step{}_epoch000_window{window:02}_fan.report.bin",
                STEP_DECIMATION
            ));
            assert!(path.exists(), "missing snapshot {}", path.display());
        }
        let compare = read_report(
            &snapshots.join(format!(
                "step{}_epoch000_window01_fan.report.bin",
                STEP_DECIMATION
            )),
        )
        .unwrap();
        // Nothing here may be called `predicted`, and the chart must carry the two
        // numbers that separate miscalibration from ordinary dispersion and from the
        // noise of estimating a quantile: the in-band rate and the centre's se.
        assert!(
            !compare.title.contains("predicted"),
            "the fan title reintroduced a predicted path: {}",
            compare.title
        );
        assert!(
            compare.title.contains("band on") && compare.title.contains("fan-centre se"),
            "the fan must state its in-band rate and its centre's standard error: {}",
            compare.title
        );
        match compare.kind {
            ReportKind::CandleFan {
                actual,
                bands,
                samples,
            } => {
                assert_eq!(actual.len(), horizon as usize);
                for bar in &actual {
                    assert!(bar.low <= bar.open.min(bar.close));
                    assert!(bar.high >= bar.open.max(bar.close));
                }
                assert_eq!(
                    bands.len(),
                    FAN_QUANTILES.len(),
                    "the fan must carry every configured quantile locus"
                );
                for (band, probability) in bands.iter().zip(FAN_QUANTILES) {
                    assert_eq!(band.probability, probability);
                    assert_eq!(band.closes.len(), horizon as usize);
                }
                // Ascending in probability, which is what lets the renderer nest the bands.
                for pair in bands.windows(2) {
                    assert!(
                        pair[0].probability < pair[1].probability,
                        "bands must ascend: {} then {}",
                        pair[0].probability,
                        pair[1].probability
                    );
                }
                assert_eq!(
                    samples.len(),
                    SNAPSHOT_OVERLAY_PATHS.min(rollout.size()[1] as usize),
                    "a fan without genuine draws cannot be told apart from a summary"
                );
                for draw in &samples {
                    assert_eq!(draw.values.len(), horizon as usize);
                }
            }
            other => panic!("expected CandleFan, got {other:?}"),
        }

        fs::remove_dir_all(&root).ok();
    }

    /// PER-EPOCH PROGRESSION, the whole point of the epoch-boundary path.
    ///
    /// Three passes must leave three trade points and three snapshot SETS, one per
    /// boundary — not one at the end, and not one per validation. The snapshots must
    /// depict the SAME scene every time, because a fan that tightens against a different
    /// realized path each epoch is not evidence of anything. And the epoch-indexed panel
    /// must be a different file from the tick panel carrying a different number of
    /// points, since the failure this replaces was a dense noisy curve being read as
    /// per-pass progress.
    #[test]
    fn every_epoch_boundary_leaves_one_trade_point_and_one_snapshot_set_on_one_scene() {
        let _torch_rng_guard = test_rng::shared();
        cap_torch_threads();
        let root = scratch_dir("per_epoch");
        let mut reporter = PretrainReporter::new(&root, MARGINAL_DOF);

        let epochs = 3usize;
        let windows = 2i64;
        let samples = 8i64;
        let horizon = 4i64;
        // ONE scene. Drawn once, outside the loop, and handed to every boundary: the
        // fixture makes a moving pinned set impossible rather than asserting it did not
        // move, and the trainer-side assertion that the real pinned set is equally fixed
        // lives next door in `pretrain.rs`.
        let future = Tensor::rand(
            [windows, horizon, BAR_DOF as i64],
            (Kind::Float, Device::Cpu),
        ) * 0.01;

        let ticks_per_epoch = 2usize;
        let mut boundary_steps = Vec::with_capacity(epochs);
        for epoch in 0..epochs {
            for tick in 0..ticks_per_epoch * STEP_DECIMATION {
                let mut metrics = StepMetrics::nan();
                metrics.epoch = epoch;
                metrics.step = epoch * ticks_per_epoch * STEP_DECIMATION + tick;
                metrics.nll_bar = 24.0 - metrics.step as f64 * 0.001;
                metrics.nll_dof = [4.8; BAR_DOF];
                metrics.total_loss = 24.0;
                metrics.context = 896;
                metrics.batch_size = 16;
                metrics.bar_tokens = (16 * 896) as f64;
                reporter.record_step(&metrics).unwrap();
            }
            let step = (epoch + 1) * ticks_per_epoch * STEP_DECIMATION;
            boundary_steps.push(step);
            // The model's draws move epoch over epoch; what they are drawn against does
            // not. That asymmetry is exactly what the picture is for.
            let rollout = Tensor::rand(
                [windows, samples, horizon, BAR_DOF as i64],
                (Kind::Float, Device::Cpu),
            ) * 0.01;
            reporter
                .record_snapshot(&SnapshotInput {
                    rollout: &rollout,
                    future_dof: &future,
                    epoch,
                    global_step: step,
                })
                .unwrap();
            reporter
                .record_epoch(&populated_epoch(epoch, step, None))
                .unwrap();
            reporter
                .record_epoch_boundary(&populated_boundary(epoch, step))
                .unwrap();
        }

        // 1. Exactly one snapshot set per boundary, tagged with both indices, and the
        //    realized path in it is byte-identical across every epoch.
        let mut scenes: Vec<Vec<f32>> = Vec::new();
        for (epoch, step) in boundary_steps.iter().copied().enumerate() {
            let dir = root.join(epoch.to_string()).join("candle_snapshots");
            let fans: Vec<PathBuf> = fs::read_dir(&dir)
                .expect("every boundary writes a snapshot directory")
                .map(|entry| entry.expect("snapshot dir entry").path())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.ends_with("_fan.report.bin"))
                })
                .collect();
            assert_eq!(
                fans.len(),
                windows as usize,
                "epoch {epoch} holds {} fans, not the {windows} pinned windows: {fans:?}",
                fans.len()
            );
            for window in 1..=windows {
                let path =
                    dir.join(format!("step{step}_epoch{epoch:03}_window{window:02}_fan.report.bin"));
                assert!(
                    path.exists(),
                    "a fan must name both its epoch and its step: {}",
                    path.display()
                );
            }
            let report = read_report(
                &dir.join(format!("step{step}_epoch{epoch:03}_window01_fan.report.bin")),
            )
            .expect("fan reads back");
            let ReportKind::CandleFan { actual, .. } = report.kind else {
                panic!("expected CandleFan at epoch {epoch}");
            };
            scenes.push(actual.iter().map(|bar| bar.close).collect::<Vec<f32>>());
        }
        for epoch in 1..epochs {
            assert_eq!(
                scenes[epoch], scenes[0],
                "the pinned scene moved between epoch 0 and epoch {epoch}; a fan drawn \
                 against a different realized path each pass cannot be compared to the \
                 previous one, which is the only thing these pictures are for"
            );
        }

        // 2. The epoch panel carries exactly one point per boundary, and it is a
        //    DIFFERENT file from the dense tick panel, which neither overwrote.
        let last = root.join((epochs - 1).to_string());
        let series_len = |base: &str| -> usize {
            let report =
                read_report(&last.join(format!("{base}.report.bin"))).expect("chart reads back");
            let ReportKind::MultiLine { series } = report.kind else {
                panic!("{base} is not a line chart");
            };
            let finite = series
                .iter()
                .map(|s| s.values.iter().filter(|v| v.is_finite()).count())
                .max()
                .expect("a chart has at least one series");
            finite
        };
        for base in ["pretrain_epoch_trade", "pretrain_epoch_trade_edge"] {
            assert_eq!(
                series_len(base),
                epochs,
                "{base} must hold exactly one point per pass over the corpus"
            );
        }
        // The tick panel measures the same bench at every validation, so it is denser by
        // construction. Equal lengths would mean one axis had quietly become the other.
        let tick_points = series_len("pretrain_trade_vs_baselines");
        assert!(
            tick_points > epochs,
            "the step-indexed trade curve holds {tick_points} points for {epochs} epochs; \
             the epoch series has replaced it rather than sitting beside it"
        );
        assert!(
            last.join("pretrain_trade_vs_baselines.report.bin").exists()
                && last.join("pretrain_epoch_trade_edge.report.bin").exists(),
            "the two axes must coexist on disk"
        );

        // 3. Earlier epoch directories are not rewritten by later boundaries: epoch 0's
        //    panel is a one-point chart forever, which is what makes an epoch artifact's
        //    neighbourhood a snapshot of the run at that moment.
        let first = root.join("0");
        let report = read_report(&first.join("pretrain_epoch_trade_edge.report.bin"))
            .expect("epoch 0 chart reads back");
        let ReportKind::MultiLine { series } = report.kind else {
            panic!("not a line chart");
        };
        assert_eq!(
            series
                .iter()
                .map(|s| s.values.len())
                .max()
                .expect("a chart has at least one series"),
            1,
            "epoch 0's directory must still depict one epoch"
        );

        fs::remove_dir_all(&root).ok();
    }

    /// The budget line is the one number that makes "3 epochs" falsifiable, so it is
    /// checked against job 2865's real shortfall rather than a round fixture: 13831 steps
    /// that declared 1,104.7M bar-tokens delivered 488.6M because the VRAM gate held the
    /// batch at 24. That run charted three epochs and ran 1.33 passes.
    #[test]
    fn the_epoch_row_states_a_held_ramps_shortfall_rather_than_the_epochs_that_were_asked_for() {
        let full_pass = 368_222_980u64;
        let requested = 3u64;
        let delivered = 13_831u64 * 24 * 1472;
        let mut boundary = populated_boundary(1, 11_382);
        boundary.full_pass_bar_tokens = full_pass;
        boundary.run_target_bar_tokens = full_pass * requested;
        boundary.run_bar_tokens = delivered / 2;
        boundary.projected_run_bar_tokens = delivered;
        boundary.epoch_bar_tokens = delivered / 2;

        // 488.6M of a 1,104.7M target: the run is on course for 1.33 passes.
        assert!(
            (boundary.projected_epochs() - 1.327).abs() < 5e-3,
            "projected {} passes",
            boundary.projected_epochs()
        );
        assert!(
            (boundary.projected_fraction() - 0.442).abs() < 5e-3,
            "projected fraction {}",
            boundary.projected_fraction()
        );
        assert!(
            boundary.projected_fraction() < super::super::pretrain::BAR_TOKEN_SHORTFALL_WARN,
            "a run delivering 44% of its declared budget must trip the shortfall warning"
        );

        // The console line has to SAY the shortfall. A reader who has to divide two
        // numbers off a chart to discover that a 40-hour run is a third of what it claims
        // is a reader who finds out in hour forty.
        let line = boundary.console_line();
        assert!(
            line.contains("1.33"),
            "the epoch line must state the passes actually projected: {line}"
        );

        // And the on-plan fixture must NOT trip it, or the warning is noise.
        let healthy = populated_boundary(0, 100);
        assert!((healthy.projected_epochs() - REQUESTED_EPOCHS as f64).abs() < 1e-9);
        assert!(
            healthy.projected_fraction() >= super::super::pretrain::BAR_TOKEN_SHORTFALL_WARN
        );
    }

    /// The STANDALONE bench path, which has no [`PretrainReporter`] and therefore no
    /// tick axis, must still leave every trade base the TUI registers on disk with a
    /// finite value. It shipped writing three of five, which renders as blank panels —
    /// indistinguishable, in the TUI, from a bench that measured nothing.
    ///
    /// The expectation is DERIVED from the registry rather than counted here, so adding a
    /// trade chart extends this test by itself. A chart the standalone path genuinely
    /// cannot produce belongs in [`CYCLE_EXEMPT`] with its reason, which is the same
    /// convention the in-run cycle walk uses — one exemption list, not two.
    #[test]
    fn the_standalone_bench_writes_every_registered_trade_base() {
        let root = scratch_dir("standalone_trade");
        let trade = populated_trade();
        assert!(trade.measured(), "the fixture must have traded bars");
        write_trade_bench(&root, "fixture", &trade).expect("the standalone bench writes");

        let bases: Vec<&str> = EXPECTED_BASES
            .iter()
            .copied()
            .filter(|base| base.starts_with("pretrain_trade_"))
            .filter(|base| !CYCLE_EXEMPT.contains(base))
            .collect();
        assert!(
            bases.len() >= 5,
            "the registry lists {} unexempted trade bases; the five this bench has always \
             emitted cannot have gone away, so the prefix or the registry moved and this \
             test is now covering nothing",
            bases.len()
        );
        for base in bases {
            let path = root.join(format!("{base}.report.bin"));
            assert!(path.exists(), "{base} was never written by the standalone bench");
            let report = read_report(&path).expect("report reads back");
            match report.kind {
                // A histogram lands as one of these two; anything else is not a chart of
                // per-bar trading quantities and the reader would be looking at the wrong
                // renderer.
                ReportKind::MultiLine { series } => {
                    assert!(!series.is_empty(), "{base} carries no series");
                    assert!(
                        series
                            .iter()
                            .any(|s| s.values.iter().any(|v| v.is_finite())),
                        "{base} holds no finite value"
                    );
                }
                ReportKind::Simple { values, .. } => {
                    assert!(
                        values.iter().any(|v| v.is_finite()),
                        "{base} holds no finite value"
                    );
                }
                other => panic!("{base} has unexpected kind {other:?}"),
            }
        }

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn train_is_dense_and_validation_is_sparse_on_the_shared_axis() {
        let root = scratch_dir("axis");
        let mut reporter = PretrainReporter::new(&root, MARGINAL_DOF);
        for step in 0..STEP_DECIMATION * 3 {
            let mut metrics = StepMetrics::nan();
            metrics.step = step;
            metrics.nll_bar = 24.0;
            reporter.record_step(&metrics).unwrap();
        }
        reporter
            .record_epoch(&populated_epoch(0, STEP_DECIMATION * 3, None))
            .unwrap();

        let report = read_report(&root.join("0").join("pretrain_nll_bar.report.bin")).unwrap();
        let ReportKind::MultiLine { series } = report.kind else {
            panic!("expected MultiLine");
        };
        let train = &series[0];
        let val = &series[1];
        assert_eq!(train.label, "train");
        assert_eq!(val.label, "val deployed");
        assert_eq!(train.values.len(), 4, "3 step ticks plus the epoch tick");
        assert_eq!(val.values.len(), 4);
        assert_eq!(train.values.iter().filter(|v| v.is_finite()).count(), 4);
        assert_eq!(
            val.values.iter().filter(|v| v.is_finite()).count(),
            1,
            "validation must be a single marker, not a dense curve"
        );
        assert_eq!(val.values[3], 21.0);

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn promotions_accumulate_and_absent_metrics_stay_unwritten() {
        let root = scratch_dir("promotions");
        let mut reporter = PretrainReporter::new(&root, MARGINAL_DOF);
        let weights = checkpoint(&root, "best.ot", b"weights-v1");
        for (epoch, promoted) in [
            (0usize, Some(weights.clone())),
            (1, None),
            (2, Some(weights.clone())),
        ] {
            let mut metrics = populated_epoch(epoch, epoch * 10, promoted);
            metrics.effective_rank = f64::NAN;
            reporter.record_epoch(&metrics).unwrap();
        }

        let report = read_report(&root.join("2").join("pretrain_promotions.report.bin")).unwrap();
        let ReportKind::MultiLine { series } = report.kind else {
            panic!("expected MultiLine");
        };
        assert_eq!(series[0].values, vec![1.0, 1.0, 2.0]);

        for epoch in 0..3 {
            assert!(
                !root
                    .join(epoch.to_string())
                    .join("pretrain_effective_rank.report.bin")
                    .exists(),
                "an all-NaN series must not produce a file"
            );
        }

        fs::remove_dir_all(&root).ok();
    }

    /// EVAL-GAP-001. An unmeasured metric leaves a GAP that names itself, never a NaN.
    ///
    /// A NaN in a val column and a genuine catastrophe are the same picture, and they call for
    /// opposite responses: one means "wait for the ramp", the other means "stop the run". So a
    /// metric the caller declares unmeasured must be omitted from its series, and the series
    /// must say in its own legend that it was never measured.
    #[test]
    fn an_unmeasured_metric_is_absent_from_the_series_and_says_so() {
        let root = scratch_dir("unmeasured");
        let mut reporter = PretrainReporter::new(&root, MARGINAL_DOF);
        let mut metrics = populated_epoch(0, 10, None);
        // Exactly the state of a validation before the ramp reaches the deployed context: the
        // fixed-context panel is measured, the deployed-context one is not.
        for metric in DEPLOYED_CONTEXT_METRICS {
            metrics.unmeasured.push(UnmeasuredMetric {
                metric: metric.to_owned(),
                reason: "the ramp has not reached the deployed context.".to_owned(),
            });
        }
        metrics.val_nll_bar = f64::NAN;
        metrics.val_nll_bar_se = f64::NAN;
        metrics.val_nll_bar_ci = (f64::NAN, f64::NAN);
        metrics.val_nll_bar_se_level = f64::NAN;
        metrics.val_nll_bar_conditional_deployed = f64::NAN;
        metrics.val_promotion_context = f64::NAN;
        reporter.record_epoch(&metrics).unwrap();

        let report = read_report(&root.join("0").join("pretrain_nll_bar.report.bin")).unwrap();
        let ReportKind::MultiLine { series } = report.kind else {
            panic!("expected MultiLine");
        };
        let deployed = series
            .iter()
            .find(|s| s.label.starts_with("val deployed") && !s.label.contains("conditional"))
            .expect("the deployed series must still be listed");
        assert!(
            deployed.values.iter().all(|v| !v.is_finite()),
            "an unmeasured metric must be absent from the series, not charted as NaN"
        );
        assert!(
            deployed.label.contains("NOT MEASURED"),
            "the legend must distinguish `not measured` from `measured and bad`, got {}",
            deployed.label
        );
        // The diagnostic panel, on the same tick, IS measured — that is the whole point of
        // decoupling it from the promotion gate.
        let diag = read_report(&root.join("0").join("pretrain_nll_bar_diag896.report.bin"))
            .unwrap();
        let ReportKind::MultiLine { series } = diag.kind else {
            panic!("expected MultiLine");
        };
        let val_diag = series
            .iter()
            .find(|s| s.label == "val diag")
            .expect("the fixed-context read must be present and measured");
        assert!(val_diag.values.iter().any(|v| v.is_finite()));

        fs::remove_dir_all(&root).ok();
    }

    fn populated_battery(checkpoint: PathBuf) -> TestBattery {
        let mut battery = TestBattery::nan(checkpoint, "0f1e2d3c4b5a".to_owned());
        battery.nll_bar = 21.4;
        battery.nll_dof = [4.2, 4.3, 4.2, 4.3, 4.4];
        battery.crps_dof = [0.003, 0.002, 0.19, 0.21, 0.44];
        battery.rollout_nll_exact = [21.4, 22.1, 22.9, 23.8, 24.4];
        battery.rollout_nll_dynamics = [21.5, 22.4, 23.6, 25.1, 26.2];
        battery.dir_acc = 0.514;
        battery
            .pit
            .accumulate(&(Tensor::arange(1024, (Kind::Float, Device::Cpu)) / 1024.0)
                .unsqueeze(-1)
                .repeat([1, BAR_DOF as i64]));
        battery.corpus_fingerprint = "c".repeat(64);
        battery.split_bounds = (1_600_000_000_000, 1_650_000_000_000);
        battery.nll_bar_conditional = 22.1;
        battery.nll_dof_conditional = [4.2, 4.3, 4.55, 4.65, 4.4];
        battery.nll_bar_se = 0.031;
        battery.nll_bar_ci = (21.34, 21.46);
        // Marginalizing the intra-bar chain can only cost nats, so the forecast row is above
        // the teacher-forced one on identical rows.
        battery.forecast_nll_dof = [4.35, 4.62, 4.71, 4.78, 4.58];
        battery.forecast_teacher_nll_dof = [4.2, 4.3, 4.2, 4.3, 4.4];
        battery.forecast_nll_se = 0.019;
        battery.selection_context = 896;
        battery.deployed_context = 2048;
        battery.reached_context = 1024;
        battery.lr_plateau_fraction = 0.40;
        battery.trade = populated_trade();
        battery
    }

    fn promoted_reporter(root: &Path, weights: &Path) -> PretrainReporter {
        let mut reporter = PretrainReporter::new(root, MARGINAL_DOF);
        reporter.set_held_out_baselines(smoothed_baselines());
        reporter
            .record_epoch(&populated_epoch(0, 10, Some(weights.to_path_buf())))
            .unwrap();
        reporter
    }

    /// The reference set a `smoothed` run reports, i.e. the one mode that has a floor.
    fn smoothed_baselines() -> HeldOutBaselines {
        HeldOutBaselines {
            scoring: BarScoring::Smoothed,
            uniform_nll_bar: uniform_categorical_nll_bar(),
            marginal_nll_dof_conditional: [4.74, 4.66, 3.83, 3.74, 4.85],
            marginal_nll_dof_val: [4.75, 4.67, 3.78, 3.70, 4.86],
            encoding_identity_nats: 0.690,
            scoring_floor_bar: 4.648,
            marginal_class_dof: [0.272, 0.345, 1.195, 1.217, 0.0],
            marginal_shape_dof: [4.469, 4.314, 2.554, 2.451, 4.851],
        }
    }

    #[test]
    fn the_held_out_battery_is_written_once_with_every_scalar() {
        let root = scratch_dir("battery");
        let weights = checkpoint(&root, "best.ot", b"promoted");
        let reporter = promoted_reporter(&root, &weights);
        reporter.finish(&populated_battery(weights)).unwrap();

        let report = read_report(&root.join("0").join("pretrain_test.report.bin")).unwrap();
        assert!(
            report.title.contains("best.ot") && report.title.contains("0f1e2d3c4b5a"),
            "the battery must name the artifact and lineage it scored: {}",
            report.title
        );
        let ReportKind::MultiLine { series } = report.kind else {
            panic!("expected MultiLine");
        };
        let labels: Vec<&str> = series.iter().map(|s| s.label.as_str()).collect();
        for expected in [
            "nll_bar",
            "nll_bar vs uniform",
            "nll_bar vs marginal",
            "uniform",
            "marginal",
            "nll r",
            "crps w",
            "pit tv u",
            "rollout h64 exact",
            "rollout h64 dynamics",
            "dir acc",
            "nll_bar se",
            "nll_bar ci95 low",
            "nll_bar conditional",
            "marginal conditional",
            "encoding identity nats",
            "nll u | s!=0",
            "nll r vs marginal",
        ] {
            assert!(labels.contains(&expected), "battery is missing {expected}");
        }
        for entry in &series {
            assert_eq!(entry.values.len(), 1, "{} is not a single point", entry.label);
            assert!(
                entry.values[0].is_finite(),
                "{} is not finite",
                entry.label
            );
        }
        let marginal_total: f64 = MARGINAL_DOF.iter().sum();
        let gain = series
            .iter()
            .find(|s| s.label == "nll_bar vs marginal")
            .unwrap()
            .values[0] as f64;
        assert!((gain - (marginal_total - 21.4)).abs() < 1.0e-4);

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn the_battery_refuses_a_checkpoint_that_was_never_promoted() {
        let root = scratch_dir("battery_wrong");
        let promoted = checkpoint(&root, "best.ot", b"promoted");
        let other = checkpoint(&root, "final.ot", b"not-promoted");
        let reporter = promoted_reporter(&root, &promoted);
        let error = reporter
            .finish(&populated_battery(other))
            .expect_err("scoring an unpromoted artifact must fail");
        assert!(
            error.to_string().contains("actually selected"),
            "unexpected error: {error}"
        );
        assert!(!root.join("0").join("pretrain_test.report.bin").exists());

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn the_battery_refuses_a_checkpoint_rewritten_after_promotion() {
        let root = scratch_dir("battery_mutated");
        let weights = checkpoint(&root, "best.ot", b"promoted");
        let reporter = promoted_reporter(&root, &weights);
        fs::write(&weights, b"rewritten-after-promotion").unwrap();
        let error = reporter
            .finish(&populated_battery(weights))
            .expect_err("a mutated artifact must fail");
        assert!(
            error.to_string().contains("changed on disk"),
            "unexpected error: {error}"
        );

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn the_battery_refuses_an_in_memory_model_and_an_unpromoted_run() {
        let root = scratch_dir("battery_guards");
        let weights = checkpoint(&root, "best.ot", b"promoted");

        let never_promoted = PretrainReporter::new(&root, MARGINAL_DOF);
        let error = never_promoted
            .finish(&populated_battery(weights.clone()))
            .expect_err("a run that never promoted has nothing to score");
        assert!(
            error.to_string().contains("no promotion was ever reported"),
            "unexpected error: {error}"
        );

        let reporter = promoted_reporter(&root, &weights);
        let mut battery = populated_battery(weights);
        battery.model_lineage = "   ".to_owned();
        let error = reporter
            .finish(&battery)
            .expect_err("an empty lineage means no reload happened");
        assert!(
            error.to_string().contains("reloaded through BarWorldModel::load"),
            "unexpected error: {error}"
        );

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_promotion_whose_artifact_is_missing_fails_immediately() {
        let root = scratch_dir("missing_artifact");
        let mut reporter = PretrainReporter::new(&root, MARGINAL_DOF);
        let error = reporter
            .record_epoch(&populated_epoch(0, 10, Some(root.join("absent.ot"))))
            .expect_err("promoting a file that is not on disk must fail at once");
        assert!(
            error.to_string().contains("save it before reporting the promotion"),
            "unexpected error: {error}"
        );

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn marginal_reference_lines_track_the_fitted_supports() {
        let root = scratch_dir("baselines");
        let mut reporter = PretrainReporter::new(&root, MARGINAL_DOF);
        let baselines = smoothed_baselines();
        reporter.set_held_out_baselines(baselines);
        reporter.record_epoch(&populated_epoch(0, 10, None)).unwrap();

        let report =
            read_report(&root.join("0").join("pretrain_nll_vs_baselines.report.bin")).unwrap();
        let ReportKind::MultiLine { series } = report.kind else {
            panic!("expected MultiLine");
        };
        let marginal_total: f64 = MARGINAL_DOF.iter().sum();
        let uniform = baselines.uniform_nll_bar;
        let line = series.iter().find(|s| s.label == "marginal").unwrap();
        assert!(
            (line.values[0] as f64 - (uniform - marginal_total)).abs() < 1.0e-4,
            "marginal reference should sit {} nats above uniform, got {}",
            uniform - marginal_total,
            line.values[0]
        );
        assert!(series.iter().any(|s| s.label == "uniform"));
        // The smoothed rule is the only one with a floor, so the unreachable line is drawn.
        let floor = series
            .iter()
            .find(|s| s.label == "smoothing floor (unreachable)")
            .expect("the smoothed rule must draw its floor");
        assert!(
            (floor.values[0] as f64 - (uniform - baselines.scoring_floor_bar)).abs() < 1.0e-4
        );

        let dof_report = read_report(&root.join("0").join("pretrain_nll_dof.report.bin")).unwrap();
        let ReportKind::MultiLine { series } = dof_report.kind else {
            panic!("expected MultiLine");
        };
        for (dof, name) in BAR_DOF_NAMES.iter().enumerate() {
            let label = format!("{name} marginal");
            let line = series.iter().find(|s| s.label == label).unwrap();
            assert!((line.values[0] as f64 - MARGINAL_DOF[dof]).abs() < 1.0e-4);
        }

        // Absent marginals must simply omit the reference, never invent one.
        let bare_root = scratch_dir("baselines_bare");
        let mut bare = PretrainReporter::new(&bare_root, [f64::NAN; BAR_DOF]);
        bare.record_epoch(&populated_epoch(0, 10, None)).unwrap();
        let report =
            read_report(&bare_root.join("0").join("pretrain_nll_vs_baselines.report.bin")).unwrap();
        let ReportKind::MultiLine { series } = report.kind else {
            panic!("expected MultiLine");
        };
        let line = series.iter().find(|s| s.label == "marginal").unwrap();
        assert!(line.values.iter().all(|v| !v.is_finite()));
        let dof_report =
            read_report(&bare_root.join("0").join("pretrain_nll_dof.report.bin")).unwrap();
        let ReportKind::MultiLine { series } = dof_report.kind else {
            panic!("expected MultiLine");
        };
        // The reference is absent AND says so in its own legend entry: an all-NaN curve is
        // indistinguishable from a measured catastrophe otherwise.
        for name in BAR_DOF_NAMES {
            let line = series
                .iter()
                .find(|s| s.label == format!("{name} marginal (NOT MEASURED)"))
                .unwrap_or_else(|| {
                    panic!(
                        "{name} marginal must be present and labelled unmeasured, got {:?}",
                        series.iter().map(|s| &s.label).collect::<Vec<_>>()
                    )
                });
            assert!(
                line.values.iter().all(|v| !v.is_finite()),
                "{name} marginal must stay absent, never be invented"
            );
        }

        fs::remove_dir_all(&root).ok();
        fs::remove_dir_all(&bare_root).ok();
    }

    /// The RUN-scoped coverage panels exist, carry the cross-pass fact, and contradict the
    /// per-pass panel ON THE PER-PASS PANEL ITSELF.
    ///
    /// This is a regression test for a READING failure, not a computation failure, so it asserts
    /// on what a reader is shown rather than on a number. `pretrain_pass_multiplicity` was
    /// always arithmetically correct: it is a per-pass census, `CoverageAudit::require_full_pass`
    /// pins within-pass multiplicity to exactly one, and it therefore reads "2 times: 0, 3+
    /// times: 0" on the third pass of a three-pass run exactly as on the first. bardist_v2
    /// emitted precisely that at every tick of its third pass. Six readers took it as a
    /// statement about the run, in preference to `pretrain_unique_bar_reuse` correctly showing
    /// 2.85 on the same screen, and a whole analysis session proceeded on a false premise.
    ///
    /// So the assertions below encode the three things that make that reading unavailable:
    /// the cross-pass share is drawn on the per-pass panel, every per-pass legend entry names
    /// its own scope, and the run-scoped bases exist and are non-empty. Legend labels and not
    /// titles, because the TUI renders `report.title` through `normalize_title`, which lowercases
    /// everything after each word's first letter, while it draws series labels VERBATIM.
    #[test]
    fn the_per_pass_multiplicity_panel_cannot_be_read_as_a_claim_about_the_run() {
        let _torch_rng_guard = test_rng::shared();
        let root = scratch_dir("cross_pass_legibility");
        let mut reporter = PretrainReporter::new(&root, MARGINAL_DOF);
        reporter
            .record_epoch(&populated_epoch(2, 30_000, None))
            .unwrap();
        // The reporter writes into `gens_dir/<epoch>`, and epoch 2 is the third pass — the exact
        // situation in which the per-pass panel's zeros are most misleading.
        let dir = root.join("2");

        let multi = read_report(&dir.join("pretrain_pass_multiplicity.report.bin"))
            .expect("the per-pass multiplicity panel is written");
        let ReportKind::MultiLine { series } = multi.kind else {
            panic!("expected MultiLine");
        };
        // Every per-pass curve says so where a reader looks. Without this the panel's zeros are
        // an unqualified claim, which is exactly how they were read.
        let per_pass: Vec<&ReportSeries> = series
            .iter()
            .filter(|s| s.label.contains("IN THIS PASS"))
            .collect();
        assert_eq!(
            per_pass.len(),
            MULTIPLICITY_BUCKETS,
            "every per-pass bucket must name its scope in its own legend entry, got {:?}",
            series.iter().map(|s| &s.label).collect::<Vec<_>>()
        );
        // The per-pass census reads exactly zero at two and three-or-more, as it must.
        for label in ["targeted 2 times IN THIS PASS", "targeted 3+ times IN THIS PASS"] {
            let line = series
                .iter()
                .find(|s| s.label == label)
                .unwrap_or_else(|| panic!("{label} is charted"));
            assert!(
                line.values.iter().filter(|v| v.is_finite()).all(|v| *v == 0.0),
                "{label} must be zero: a full pass targets each bar exactly once"
            );
        }
        // And the line that makes those zeros unmisreadable is on THIS panel, non-zero, and
        // names itself as the cross-pass number.
        let cross = series
            .iter()
            .find(|s| s.label.contains("across ALL passes"))
            .unwrap_or_else(|| {
                panic!(
                    "the cross-pass total must be drawn ON the per-pass panel; a correct number \
                     on a different panel is what already failed. got {:?}",
                    series.iter().map(|s| &s.label).collect::<Vec<_>>()
                )
            });
        assert!(
            cross.values.iter().any(|v| v.is_finite() && *v > 0.0),
            "the cross-pass share must be positive on a multi-pass fixture"
        );

        // The run-scoped panels: passes delivered, projected and asked for, plus the single-pass
        // reference the reader compares against.
        let epochs = read_report(&dir.join("cover_effective_epochs.report.bin"))
            .expect("cover_effective_epochs is written");
        let ReportKind::MultiLine { series } = epochs.kind else {
            panic!("expected MultiLine");
        };
        for needle in ["DELIVERED", "PROJECTED", "ASKED for", "1.0 = a single pass"] {
            assert!(
                series.iter().any(|s| s.label.contains(needle)),
                "cover_effective_epochs must carry a {needle} curve, got {:?}",
                series.iter().map(|s| &s.label).collect::<Vec<_>>()
            );
        }
        // A projection above one is the whole point: it is knowable at the first validation tick
        // and it is what nobody had.
        let projected = series
            .iter()
            .find(|s| s.label.contains("PROJECTED"))
            .expect("the projection is charted");
        assert!(
            projected
                .values
                .iter()
                .any(|v| v.is_finite() && *v > 1.0),
            "a three-epoch fixture must project above one pass"
        );

        let exposure = read_report(&dir.join("cover_run_bar_exposure.report.bin"))
            .expect("cover_run_bar_exposure is written");
        let ReportKind::MultiLine { series } = exposure.kind else {
            panic!("expected MultiLine");
        };
        assert_eq!(series.len(), MULTIPLICITY_BUCKETS);
        for label in [
            "targeted 2 times SO FAR IN THIS RUN (cross-pass)",
            "targeted 3+ times SO FAR IN THIS RUN (cross-pass, NOT a per-pass census)",
        ] {
            let line = series
                .iter()
                .find(|s| s.label == label)
                .unwrap_or_else(|| {
                    panic!(
                        "{label} is charted, got {:?}",
                        series.iter().map(|s| &s.label).collect::<Vec<_>>()
                    )
                });
            assert!(
                line.values.iter().any(|v| v.is_finite() && *v > 0.0),
                "{label} must be positive on a multi-pass fixture: this is the series whose \
                 absence let a three-pass run be believed single-pass"
            );
        }

        fs::remove_dir_all(&root).ok();
    }

    /// An untracked coverage row leaves a GAP rather than a measured zero.
    ///
    /// `EpochMetrics::nan()` cannot express "absent" for a `[u64; 4]`, so the all-zero default is
    /// the only signal available. The write site used to divide by `sum().max(1)`, and
    /// `0 / 1 = 0.0` is finite, so `Series::set` accepted it and the panel drew four measured
    /// zeros for a pass nobody measured — indistinguishable from a real all-zero census, and in
    /// the same family as `NaN.max(0.0)` returning 0.
    #[test]
    fn an_untracked_coverage_row_is_absent_rather_than_charted_as_zero() {
        let _torch_rng_guard = test_rng::shared();
        let root = scratch_dir("coverage_absent");
        let mut reporter = PretrainReporter::new(&root, MARGINAL_DOF);
        let mut metrics = populated_epoch(0, 10, None);
        metrics.pass_multiplicity_bars = [0; MULTIPLICITY_BUCKETS];
        metrics.run_exposure_bars = [0; MULTIPLICITY_BUCKETS];
        reporter.record_epoch(&metrics).unwrap();

        let dir = root.join("0");
        // Nothing measured at all, so the panel is not written rather than written full of
        // zeros: `write_chart` skips an all-absent chart and `Series::measured` is false.
        for base in ["pretrain_pass_multiplicity", "cover_run_bar_exposure"] {
            let path = dir.join(format!("{base}.report.bin"));
            if let Ok(report) = read_report(&path) {
                let ReportKind::MultiLine { series } = report.kind else {
                    panic!("expected MultiLine");
                };
                for line in &series {
                    assert!(
                        line.values.iter().all(|v| !v.is_finite()),
                        "{base} charted {} from an untracked row; an absent census must leave a \
                         gap, never a zero that reads as a measurement",
                        line.label
                    );
                }
            }
        }

        fs::remove_dir_all(&root).ok();
    }
}
