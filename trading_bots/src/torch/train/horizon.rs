//! Canonical production evaluation for the pretrained bar world model.
//!
//! The decision clock is always one calendar panel bar. At every row the evaluator conditions
//! only on available history, samples one common autoregressive continuation to 100 bars, reads
//! the 1/4/16/39/78/100 cumulative simple-return laws from prefixes of those same paths, solves
//! a multi-asset action from the actual drifted holdings, executes only that first action, and
//! repeats at the next row. Forecast horizon and rebalance frequency are therefore independent.
//! `--forecast-horizon` selects and highlights the production model run from that exact grid;
//! it does not discard the other prefixes, which remain the fixed comparison in both reports.
//!
//! The action objective is predicted simple-return growth minus
//! `0.5 w'(Cov(R) + E[R]E[R]')w` and [`super::portfolio::PanelCost`]. The covariance is a PSD
//! trailing/shrunk one-factor law; the mean outer product is exact. Spread, fees and
//! square-root impact are inside the solve, so retaining the current holding is endogenous.
//! The deterministic perfect-foresight oracle has zero covariance and its realized-H mean
//! outer product. Model and oracle pass through identical solver, cost, holding and constraint
//! contracts.
//!
//! Raw predictive diagnostics remain outside this module. Economic output is written only as
//! `.report.bin`, defaults to validation, and refuses the locked test split without an explicit
//! second opt-in.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use shared::report::{read_report, write_report, Report, ReportKind, ReportSeries, ScaleKind};
use tch::{Device, Kind, Tensor};

use crate::torch::bar_dist::{BarSupports, DOF_R, NUM_BAR_BINS};
use crate::torch::dataset::{
    bar_time_ids, forecast_schedule_after, forecast_schedule_ids_from, BarCorpus, BarEndpoint,
    Split, BAR_TIME_FEATURES,
};
use crate::torch::world_model::{world_model_metadata_path, BarWorldModel, BAR_MODEL_DIM};

use super::portfolio::{
    marginal_forecasts, solve_cost_aware_kelly, CostModel, CostParts, FactorCovariance, FlatCost,
    KellyConstraints, Panel, PanelConfig, PanelCost, PanelForecast, Policy,
    TrailingFactorCovariance, ADV_TRAILING_BARS, BELIEF_EMIT, BELIEF_PRE_CONTEXT, DEFAULT_COST_BPS,
    DEFAULT_GROSS_CAP, MAX_BREAK_EVEN_BPS, POLICIES,
};
use super::portfolio_cost::{BarCostModel, CostCalibration};
use super::pretrain_reports::write_chart;
use super::pretrain_stats::{block_bootstrap, Dispersion, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED};
use super::trade_bench::{forecast_r_probs, kelly_fractions, FREE_LEVERAGE, ROW_CHUNK};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Holding periods, in bars of the deployment resolution.
///
/// `39` is half a regular session, `78` is one, `195` is two and a half days and `390` is a
/// trading week at 5-minute RTH bars. The grid is geometric below a day because that is where
/// the turnover-versus-edge trade-off is decided, and it stops at a week because a five-month
/// held-out panel cannot carry enough non-overlapping weeks to say anything beyond it (see
/// [`MIN_CREDIBLE_PERIODS`]).
///
/// This is the axis of the experiment, not a tuned parameter: the deliverable is the SHAPE of
/// break-even against `k`, and a single favourable `k` would be a selected number.
pub const HOLD_HORIZONS: [usize; 9] = [1, 2, 4, 8, 16, 39, 78, 195, 390];
/// Production forecast horizons.  They are all evaluated on the same one-bar rebalance
/// clock and extracted from prefixes of one common 100-bar ancestral rollout.
pub const FORECAST_HORIZONS: [usize; 6] = [1, 4, 16, 39, 78, 100];
pub const DEFAULT_FORECAST_HORIZON: usize = 100;

/// Parse and validate the production horizon selected by the CLI.
///
/// Selection is deliberately restricted to the exact prefix laws in [`FORECAST_HORIZONS`]:
/// accepting an arbitrary value would either label a different run as selected or require a
/// second rollout that is no longer part of the fixed comparison grid.
pub fn parse_forecast_horizon(value: &str) -> std::result::Result<usize, String> {
    let horizon = value
        .parse::<usize>()
        .map_err(|_| format!("forecast horizon must be one of {FORECAST_HORIZONS:?}"))?;
    if FORECAST_HORIZONS.contains(&horizon) {
        Ok(horizon)
    } else {
        Err(format!(
            "forecast horizon {horizon} is not in the production grid {FORECAST_HORIZONS:?}"
        ))
    }
}

/// Deepest horizon `pretrain`'s rollout diagnostics reach, and therefore the deepest at which
/// `RolloutMode::Dynamics`'s drift against the exact trunk has ever been measured.
///
/// `pretrain_reports::ROLLOUT_HORIZONS` ends at 100 and `pretrain::SNAPSHOT_HORIZON` holds
/// exactly that many bars out, so a sampled row above this extrapolates the belief-advance
/// mechanism and says so.
pub const DYNAMICS_DIAGNOSED_HORIZON: usize = 100;

/// Fewest non-overlapping holding periods a row needs before its Sharpe and drawdown are
/// treated as measurements rather than as noise.
///
/// Twenty is where the standard error of an annualized Sharpe falls to roughly a quarter of
/// its own value; below it the ordering of two rows is not information.
pub const MIN_CREDIBLE_PERIODS: usize = 20;

/// Monte-Carlo paths drawn per (name, rebalance) at every sampled row.
///
/// Set from the variance-reduction arithmetic in the module docs, not by taste: the
/// Rao-Blackwellized drift has a per-path spread of the order of the conditional mean itself,
/// so `sqrt(N) / 2` is the signal-to-noise of one name's drift estimate and 96 puts that near
/// 5. The plain estimator would need ~500 at `k = 2` for the same, and both are reported so
/// the claim is checkable.
pub const DEFAULT_SAMPLES: usize = 96;

/// Independent replicate sample sets per sampled row.
///
/// Break-even cost is a bisection over a compounded book, so no closed form propagates path
/// noise into it. The only honest error bar is the spread of the whole pipeline across
/// independent sample sets, which costs a factor of this constant and is why it is 3 rather
/// than 10.
pub const DEFAULT_REPLICATES: usize = 3;

/// Sampling temperature of the rollout. Exactly `1.0`: any other value measures a different
/// predictive law than the one the checkpoint was selected on.
pub const ROLLOUT_TEMPERATURE: f64 = 1.0;

/// Rows (`pairs * samples`) held on the device at once inside the rollout.
///
/// `BarDynamics`'s hidden layer is 1664 wide, so a chunk holds `8192 * 1664` f32 of
/// activation — 54 MiB — and the whole pass stays inside the ~2 GiB inference budget a shared
/// GPU allows.
const SAMPLE_ROW_LIMIT: usize = 8192;

/// Floor on the aggregate log-return variance in the Kelly denominator, in nats squared.
///
/// A realistic one-bar value is ~1e-5, so this is seven orders below the quantity it guards
/// and exists only to keep a degenerate sample set from dividing by zero.
const VARIANCE_FLOOR: f64 = 1e-12;

/// Bisection steps of the break-even solve over `[0, MAX_BREAK_EVEN_BPS]`.
///
/// Fewer than [`super::portfolio`]'s 48 because each step re-runs the whole book rather than
/// replaying a stored path; 40 halvings still resolve the cost to `~1e-9` bps, eleven orders
/// below the distinction anyone acts on.
const BREAK_EVEN_ITERATIONS: usize = 40;

/// The MATCHED, MEASURED one-way cost the verdict is stated against, in bps: half-spread plus
/// commission plus regulatory fee, with NO impact model and no free parameter, equal-weighted
/// over exactly the 256 traded symbol-months a break-even was last measured on.
///
/// Three properties, each of which was got wrong at least once this session before it was got
/// right, and all three matter for reading a break-even against it.
///
/// * **Matched.** `super::portfolio_cost`'s restriction run priced the SAME 256 symbol-months
///   the policy traded, not a universe-wide decile. Decile occupancy came out
///   `[8, 24, 18, 18, 24, 27, 29, 38, 27, 43]` over thinnest-to-deepest, i.e. a draw that spans
///   the universe with a mild tilt to the deep end. The earlier practice of quoting the
///   deepest UNIVERSE decile's median — a different symbol set from any break-even — against a
///   break-even understated the cost by 2.6x, and that comparison is retired. Its same-symbols
///   replacement is [`MATCHED_DEEPEST_DECILE_BPS`].
/// * **Measured.** No impact term. The matched ALL-IN figure at 1% of ADV is
///   [`MATCHED_ALL_IN_BPS`], but most of the gap is square-root impact at the literature
///   default `IMPACT_K = 0.5` that nobody fitted to this corpus, so a conclusion drawn against
///   this column survives the impact coefficient being wrong by any factor. That is the
///   stronger claim, which is why it is the headline.
/// * **Equal-weighted, not median.** Each traded name contributes exactly one window, so the
///   book holds all 256 equally; per-name cost is heavily right-skewed, so the mean (10.620)
///   sits well above the median (7.230) and the mean is the dimensionally matched figure.
///
/// The one mismatch that remains, stated because it cannot be closed from here: a break-even
/// in this module is a TURNOVER-weighted flat-cost equivalent, while this constant is an
/// EQUAL-weighted mean over names. They coincide exactly for the equal-weight baseline and
/// only approximately for a book whose weights vary. Closing it needs per-symbol costs inside
/// the sweep, which is what [`CostModel`] is the seam for; [`UNIVERSE_MEASURED_BPS`] is
/// carried as the second comparator so the sensitivity to that choice is visible.
pub const MATCHED_MEASURED_BPS: f64 = 10.620;

/// Matched all-in one-way cost at 1% of ADV, impact included at `IMPACT_K = 0.5`, span-pooled.
/// Anchor-month pricing came out 1.19 bps CHEAPER (25.165), so no conclusion here turns on the
/// pooling. See [`MATCHED_MEASURED_BPS`] for why the verdict does not lead with this number.
pub const MATCHED_ALL_IN_BPS: f64 = 26.351;

/// Turnover-weighted matched measured one-way cost of the ACTUAL book, in bps, over the
/// INTERIOR turnover of the same 256 traded symbol-months.
///
/// This closes the weighting mismatch [`MATCHED_MEASURED_BPS`]'s docs name as open and cannot
/// close from there: a break-even is a TURNOVER-weighted flat-cost equivalent, so the
/// dimensionally matched comparator weights each name by what the book actually rotated in it
/// rather than equally. Interior turnover is the weight because a window's entry-from-flat and
/// terminal unwind are placed by the window SAMPLER and not by the model, and being near
/// uniform across names they dilute exactly the concentration a turnover weighting exists to
/// detect.
pub const MATCHED_ACTUAL_BOOK_BPS: f64 = 10.501;

/// The same, weighted by the SIGN-ONLY arm's interior turnover.
///
/// This is the anchor for any constant-`|f|` book, the whole sign-hysteresis frontier included:
/// such a book holds `|f|` fixed, so every unit of its turnover is a sign flip and its
/// composite is a REWEIGHTING of this arm's own flip cost over the same names. A wider flip
/// margin only ever REMOVES flips, so the frontier's composite is pinned between this weighting
/// and the equal-weighted [`MATCHED_MEASURED_BPS`] under any retention monotone in a name's
/// flip count - a 0.081 bps window. That bound is CONDITIONAL on the monotonicity;
/// unconditionally the arithmetic admits `[2.2, 27.4]`, because retention is a per-name
/// fraction and filling the budget from the cheapest or dearest names is not excluded by
/// anything except the assumption. Only the book's own measured weights close it.
pub const MATCHED_SIGN_ONLY_BPS: f64 = 10.539;

/// The same, weighted by the RECALIBRATED (shrunk-mean) book's interior turnover.
///
/// Dearer than every constant-`|f|` weighting, and that is the finding rather than an aside: the
/// shrink sizes smaller, which unbinds the leverage cap and switches magnitude modulation back
/// on as a turnover source, and magnitude-driven turnover concentrates in names that are
/// thinner, costlier and more volatile. Carried here because the shrink appears as a cell of the
/// shrink-by-hysteresis 2x2 and a cell has to be priced on its own weights.
pub const MATCHED_SHRUNK_BOOK_BPS: f64 = 12.379;

/// Measured impact-free one-way cost of the PRIMARY checkpoint's out-of-sample-fitted hysteresis
/// book: `epoch_0_ctx2048@10817` at flip margin 16 bps of raw `|mu_hat|`.
///
/// # Why this is a constant and not a bound
///
/// The conditional bound `[MATCHED_SIGN_ONLY_BPS, MATCHED_MEASURED_BPS]` documented above is
/// BROKEN for this book, by a factor of 1.9, and the reason is the mechanism rather than the
/// arithmetic. The bound assumed retention monotone in a name's flip count and uncorrelated with
/// the name's own cost. Retention is strongly correlated with it: a threshold on raw `|mu_hat|`
/// retains names whose predicted mean is large, which is partly a statement that the name is
/// VOLATILE, and volatile names are thin and dear. The book's turnover-weighted ADV percentile
/// falls from 0.5954 at margin zero to 0.4930 here, and twenty of 256 names stop trading. A
/// conviction filter is a covert liquidity filter, so a constant-`|f|` book's composite must be
/// MEASURED on its own turnover and can never be bounded from the sign-only arm's weights.
pub const MATCHED_HYSTERESIS_PRIMARY_BPS: f64 = 20.096;

/// The same for the SECONDARY checkpoint's fitted book: `pretrain_step_9728@9728` at margin 32.
///
/// Costs within 0.27 bps of the primary's despite 1.7x less turnover at twice the margin, which
/// is the only cross-checkpoint evidence available on the SHAPE of the cost-versus-margin curve
/// and points at steep-then-flat rather than continued climbing. Weak evidence, two points on
/// two different checkpoints; the per-margin grid settles it.
pub const MATCHED_HYSTERESIS_SECONDARY_BPS: f64 = 20.368;

/// Equal-weighted measured impact-free one-way cost over the WHOLE 5,297-symbol universe, as
/// the comparator for a book that is not restricted to the traded 256.
pub const UNIVERSE_MEASURED_BPS: f64 = 12.325;

/// Equal-weighted measured impact-free one-way cost of the 43 TRADED names that occupy the
/// deepest liquidity decile of the universe: the cheapest cell that exists on the same symbol
/// set any break-even was measured on, and therefore the FLOOR the verdict is stated against.
///
/// This replaces a retracted constant, and the reason is the whole lesson. The number carried
/// here before was 4.150, the MEDIAN of the deepest decile of the 5,297-symbol universe. It is
/// a real measurement and it was retracted as a comparator because it prices a DIFFERENT SYMBOL
/// SET from the one the break-even came from. `super::portfolio_cost` then ran the decile
/// restriction inside the traded 256, which is the matched question, and the answer is 19%
/// higher. The matched per-decile means, thinnest first, are
/// `[28.587, 22.019, 15.289, 10.004, 12.907, 9.098, 8.405, 8.099, 6.904, 4.955]` — NOT monotone
/// at deciles 3 and 4, so nothing here may assert that cost falls with liquidity.
///
/// The mismatch that remains, carried because it cannot be closed from this module: the
/// break-evens this floor is compared against are measured on ALL the traded names, so
/// restricting the book to these 43 would change the EDGE too and nobody has measured the edge
/// on that subset. A comparison against this floor is therefore cost-restricted against
/// edge-unrestricted: indicative, not matched. It is quoted as a floor precisely because a
/// break-even BELOW it fails even the most favourable matched cell, which is a conclusion that
/// survives the edge being remeasured; a break-even ABOVE it would prove nothing until the edge
/// on those 43 names is measured.
pub const MATCHED_DEEPEST_DECILE_BPS: f64 = 4.955;

/// Worst-case effect on [`MATCHED_DEEPEST_DECILE_BPS`] of the decile's membership moving by one
/// name, in bps: `max_i |x_i - mean| / n` over all 43 members, measured by
/// `super::portfolio_cost` on the real corpus.
///
/// It is carried because a reference line with a known sensitivity is harder to misread than a
/// bare constant, and because a sibling constant WAS retracted this session when a
/// floor-versus-round choice on a tail width moved a headline by 2x. That failure cannot reach
/// this number: `portfolio_cost::decile_of_symbol` cuts `lo = decile * count / DECILES` with
/// each decile's `hi` the next one's `lo` over the whole 5,297-symbol calibration universe, an
/// EXACT PARTITION with no name dropped, none double-counted and no rounding choice available.
/// One member is also only `1/43` of this statistic rather than all of it.
///
/// 5.21% of the mean is not negligible - a 43-name mean of a right-skewed quantity gives one
/// expensive name real weight - so the bound is stated rather than waved away. What it settles
/// is that it cannot cross the verdict: the shortfall of the best credible horizon break-even
/// against this floor is `4.955 / 2.337 = 2.120x`, and the perturbation in the MODEL'S FAVOUR
/// (floor down the full amount, `4.697`) still leaves `2.010x`. The sign of the risk is
/// favourable anyway, because the name at the boundary is by construction the LEAST liquid
/// member and therefore the most expensive, so losing it LOWERS the floor.
pub const MATCHED_DEEPEST_DECILE_BOUNDARY_BPS: f64 = 0.258;

/// Which side of a cost threshold a break-even falls on, with UNMEASURED as its own answer.
///
/// A bool over floats cannot carry three states, and this is the site where that bites: `bps`
/// is `NaN` for a policy whose net growth never crossed zero inside the bracket, and
/// `NaN > threshold` is `false`, so a plain comparison reports the ABSENCE of a measurement as
/// the POSITIVE finding "BELOW the cost" — the verdict this module exists to produce, asserted
/// from nothing. Four separate instances of this shape cost this session a campaign, two charts
/// and three retracted numbers, so it gets a third branch rather than a comment.
fn side(bps: f64, threshold: f64) -> &'static str {
    if !bps.is_finite() {
        "NOT MEASURED against"
    } else if bps > threshold {
        "ABOVE"
    } else {
        "BELOW"
    }
}

/// A break-even as a chart value and a table cell, clamped but never invented.
///
/// [`MAX_BREAK_EVEN_BPS`] exists because a book with no turnover has an unbounded break-even,
/// which is a real statement and clamps honestly. `f64::min` however IGNORES `NaN`, so
/// `f64::NAN.min(MAX_BREAK_EVEN_BPS)` is `1000.0` exactly: an UNMEASURED row would render as the
/// most profitable row on the chart, on a panel beside real cost reference lines at 4.955 and
/// 10.620. `NaN` is therefore passed through, which renders as a gap in the series and as `NaN`
/// in the table, and only a genuinely infinite break-even reaches the cap.
fn displayed_break_even(bps: f64) -> f64 {
    if bps.is_nan() {
        f64::NAN
    } else {
        bps.min(MAX_BREAK_EVEN_BPS)
    }
}

pub const HORIZON_FRONTIER_BASE: &str = "pretrain_horizon_frontier";

// ---------------------------------------------------------------------------
// Constructions
// ---------------------------------------------------------------------------

/// How a position's size is derived, which is the whole experiment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Construction {
    /// One-bar lognormal moment closure held `k` bars. An explicitly labeled diagnostic for
    /// the former production sizing path.
    Stale,
    /// One-bar fitted categorical `E[R]` and `E[R²]`, sized directly and held `k` bars.
    /// The direct H1 control for the sampled aggregate moment experiment.
    StaleCategoricalMoments,
    /// The sampled `k`-bar aggregate law, sized directly by its raw `E[R] / E[R²]`.
    Horizon,
}

/// Every construction, in report order.
pub const CONSTRUCTIONS: [Construction; 3] = [
    Construction::Stale,
    Construction::StaleCategoricalMoments,
    Construction::Horizon,
];

impl Construction {
    pub fn name(self) -> &'static str {
        match self {
            Construction::Stale => "stale-1bar-lognormal-closure",
            Construction::StaleCategoricalMoments => "stale-1bar-categorical-moments",
            Construction::Horizon => "horizon-k-direct-moments",
        }
    }

    /// Whether this construction reads sampled rollouts, and therefore carries Monte-Carlo
    /// error. The two stale constructions are deterministic functions of the checkpoint and
    /// panel; their reported standard errors are `0.0` because they ARE zero, not because
    /// they were not measured.
    pub fn is_sampled(self) -> bool {
        matches!(self, Construction::Horizon)
    }
}

// ---------------------------------------------------------------------------
// The one-bar scan
// ---------------------------------------------------------------------------

/// Every belief the panel has, plus the fitted one-bar categorical reduction at each.
///
/// The beliefs live on the HOST. At the default panel that is `instants * breadth * 512` f32,
/// about 0.7 GiB, affordable in RAM and not on a shared GPU; chunks go to the device as the
/// rollout needs them. The size is asserted against [`BELIEF_CACHE_LIMIT_BYTES`] before
/// anything is allocated, so an over-large panel fails with the knob to turn rather than with
/// an allocation error.
pub struct PanelBeliefs {
    /// Row-major `[rows, BAR_MODEL_DIM]`.
    beliefs: Vec<f32>,
    /// `row_of[t][slot]` indexes [`Self::beliefs`].
    row_of: Vec<Vec<u32>>,
    /// The one-bar law in [`super::portfolio`]'s own reduction: quadratic Kelly from
    /// [`kelly_fractions`], mean and variance of the SIMPLE return.
    pub one_bar: Vec<PanelForecast>,
    /// Exact fitted categorical first and raw-second moments of the SIMPLE return, per panel
    /// entry. [`PanelForecast`] stores mean/variance in f32 for the portfolio bench; keeping
    /// E[R] and E[R²] here in f64 avoids reconstructing the Kelly numerator and denominator
    /// through that lossy representation.
    pub mean_simple: Vec<Vec<f64>>,
    pub second_simple: Vec<Vec<f64>>,
    /// Mean of the same law's LOG return, per panel entry, in nats.
    pub mu_log: Vec<Vec<f64>>,
    /// Variance of the same law's LOG return, per panel entry, in nats squared.
    pub var_log: Vec<Vec<f64>>,
}

/// Largest belief cache this module will allocate, in bytes. Six gibibytes is generous for a
/// host allocation and small enough that the failure is a message rather than an OOM.
const BELIEF_CACHE_LIMIT_BYTES: usize = 6 << 30;

impl PanelBeliefs {
    pub fn bytes(&self) -> usize {
        self.beliefs.len() * std::mem::size_of::<f32>()
    }

    pub fn entries(&self) -> usize {
        self.beliefs.len() / BAR_MODEL_DIM as usize
    }

    fn belief_row(&self, row: u32) -> &[f32] {
        let dim = BAR_MODEL_DIM as usize;
        let at = row as usize * dim;
        &self.beliefs[at..at + dim]
    }
}

/// One block pass over every panel bar: the belief, the prefix-free one-bar law of `r`,
/// and both of that law's moment reductions.
///
/// Structurally [`super::portfolio::model_forecasts`], and deliberately so — the `k = 1` row
/// of this sweep has to be the portfolio bench's own number or the sweep is measuring a
/// different model. It differs only in keeping the belief (which the rollout needs) and the
/// LOG moments (which the aggregate closure needs) beside the simple-return reduction.
pub fn scan_panel(
    model: &BarWorldModel,
    corpus: &BarCorpus,
    panel: &Panel,
    res_secs: u32,
) -> Result<PanelBeliefs> {
    let supports = model
        .supports_for(res_secs)
        .with_context(|| format!("the checkpoint carries no supports at {res_secs}s"))?;
    let device = model.device();
    let (returns_host, second_host) = supports
        .simple_return_bin_moments()
        .context("supports lack fitted simple-return moments; refit the v6 artifact")?;
    let returns = Tensor::from_slice(returns_host)
        .view([1, NUM_BAR_BINS])
        .to_device(device);
    let second_returns = Tensor::from_slice(second_host)
        .view([NUM_BAR_BINS, 1])
        .to_device(device)
        .to_kind(Kind::Double);
    let centers_host: Vec<f64> = supports.centers(DOF_R).to_vec();
    let centers = Tensor::from_slice(&centers_host)
        .view([NUM_BAR_BINS, 1])
        .to_device(device)
        .to_kind(Kind::Double);
    let centers_sq = &centers * &centers;

    let entries: usize = panel.slices().iter().map(|s| s.symbols.len()).sum();
    let want = entries * BAR_MODEL_DIM as usize * std::mem::size_of::<f32>();
    ensure!(
        want <= BELIEF_CACHE_LIMIT_BYTES,
        "the belief cache for {entries} panel entries would need {:.1} GiB of host memory, \
         over the {:.1} GiB limit; lower --max-symbols or --max-instants",
        want as f64 / (1u64 << 30) as f64,
        BELIEF_CACHE_LIMIT_BYTES as f64 / (1u64 << 30) as f64
    );

    let mut out = PanelBeliefs {
        beliefs: vec![f32::NAN; entries * BAR_MODEL_DIM as usize],
        row_of: panel
            .slices()
            .iter()
            .map(|s| vec![u32::MAX; s.symbols.len()])
            .collect(),
        one_bar: panel
            .slices()
            .iter()
            .map(|s| PanelForecast {
                kelly_f: vec![f32::NAN; s.symbols.len()],
                mean_r: vec![f32::NAN; s.symbols.len()],
                var_r: vec![f32::NAN; s.symbols.len()],
            })
            .collect(),
        mean_simple: panel
            .slices()
            .iter()
            .map(|s| vec![f64::NAN; s.symbols.len()])
            .collect(),
        second_simple: panel
            .slices()
            .iter()
            .map(|s| vec![f64::NAN; s.symbols.len()])
            .collect(),
        mu_log: panel
            .slices()
            .iter()
            .map(|s| vec![f64::NAN; s.symbols.len()])
            .collect(),
        var_log: panel
            .slices()
            .iter()
            .map(|s| vec![f64::NAN; s.symbols.len()])
            .collect(),
    };

    // Where each (symbol, bar) lands, grouped by symbol and in bar order, and the belief-cache
    // row it owns. The cache is filled in this order, so it is contiguous per symbol and a
    // rollout batch spanning rebalances gathers scattered rows on the host once.
    let mut wanted: Vec<Vec<(u32, usize, usize)>> = vec![Vec::new(); panel.symbols().len()];
    for (t, slice) in panel.slices().iter().enumerate() {
        for (k, &id) in slice.symbols.iter().enumerate() {
            wanted[id as usize].push((panel.bar_index(t, k), t, k));
        }
    }
    let mut next_row = 0u32;
    for targets in &wanted {
        for &(_, t, k) in targets {
            out.row_of[t][k] = next_row;
            next_row += 1;
        }
    }
    debug_assert_eq!(next_row as usize, entries);

    for (id, targets) in wanted.iter().enumerate() {
        if targets.is_empty() {
            continue;
        }
        let series = panel.series_of(id as u32);
        let (first, last) = (targets[0].0 as usize, targets[targets.len() - 1].0 as usize);
        ensure!(
            first as i64 >= BELIEF_PRE_CONTEXT + 1,
            "{} is tradeable from bar {first}, which cannot carry a belief with \
             {BELIEF_PRE_CONTEXT} bars of causal history plus the predecessor close the \
             encoder needs; build the panel with a larger `min_history`",
            panel.symbols()[id]
        );
        let slot: BTreeMap<usize, usize> = targets
            .iter()
            .enumerate()
            .map(|(index, &(bar, _, _))| (bar as usize, index))
            .collect();

        let mut cursor = first;
        while cursor <= last {
            let emit = BELIEF_EMIT.min((last - cursor + 1) as i64);
            let end = cursor + emit as usize - 1;
            // One extra row supplies each predicted bar's exogenous target clock;
            // only the preceding `len` bars enter the causal trunk.
            let len = emit + BELIEF_PRE_CONTEXT;
            let batch = corpus
                .dof_window(&[BarEndpoint { series, bar: end }], &[0], len + 1, device)
                .with_context(|| {
                    format!(
                        "belief block of {} bars ending at {end} for {}",
                        len + 1,
                        panel.symbols()[id]
                    )
                })?;
            let input_dof = batch.dof.narrow(1, 0, len);
            let current_time = batch.time_ids.narrow(1, 0, len);
            let target_time = batch.time_ids.narrow(1, 1, len);
            let beliefs = model.beliefs(&input_dof, &current_time);
            let conditioning = model
                .trunk()
                .forecast_conditioning(&target_time, &current_time);
            let latent = *beliefs.size().last().expect("beliefs carry a feature dim");
            ensure!(
                latent == BAR_MODEL_DIM,
                "the checkpoint's belief width is {latent}, not {BAR_MODEL_DIM}"
            );
            let block = beliefs
                .narrow(1, len - emit, emit)
                .reshape([emit, latent])
                .contiguous();
            let conditioning_block = conditioning
                .narrow(1, len - emit, emit)
                .reshape([emit, latent])
                .contiguous();

            let mut start = 0i64;
            while start < emit {
                let rows = ROW_CHUNK.min(emit - start);
                let chunk = block.narrow(0, start, rows);
                let chunk_conditioning = conditioning_block.narrow(0, start, rows);
                // The decision law and its moments with autocast OFF: `mu = sum_i p_i c_i` is
                // a cancelling sum whose value is ~1e-4 against a term spread of ~1e-3, and
                // bf16's eight mantissa bits would destroy exactly the quantity this sweep is
                // about. The BELIEF above is computed under the ambient autocast on purpose —
                // that is the regime the checkpoint was trained and selected under.
                let (kelly, mean, second, var, mu_l, var_l) = tch::autocast(false, || {
                    let probs = forecast_r_probs(model.head(), &chunk, &chunk_conditioning);
                    let kelly = host_f64(&kelly_fractions(
                        &probs,
                        &returns,
                        &second_returns,
                        FREE_LEVERAGE,
                    ));
                    let probs = probs.to_kind(Kind::Double);
                    let mean = probs
                        .matmul(&returns.reshape([NUM_BAR_BINS, 1]).to_kind(Kind::Double))
                        .reshape([-1]);
                    let second = probs.matmul(&second_returns).reshape([-1]);
                    let var = (&second - &mean * &mean).clamp_min(0.0);
                    let mu_l = probs.matmul(&centers).reshape([-1]);
                    let second_l = probs.matmul(&centers_sq).reshape([-1]);
                    let var_l = (&second_l - &mu_l * &mu_l).clamp_min(0.0);
                    (
                        kelly,
                        host_f64(&mean),
                        host_f64(&second),
                        host_f64(&var),
                        host_f64(&mu_l),
                        host_f64(&var_l),
                    )
                });
                let flat = host_f32(&chunk);
                let dim = BAR_MODEL_DIM as usize;
                for row in 0..rows as usize {
                    let bar = cursor + (start as usize) + row;
                    let Some(&index) = slot.get(&bar) else {
                        continue;
                    };
                    let (_, t, k) = targets[index];
                    out.one_bar[t].kelly_f[k] = kelly[row] as f32;
                    out.one_bar[t].mean_r[k] = mean[row] as f32;
                    out.one_bar[t].var_r[k] = var[row] as f32;
                    out.mean_simple[t][k] = mean[row];
                    out.second_simple[t][k] = second[row];
                    out.mu_log[t][k] = mu_l[row];
                    out.var_log[t][k] = var_l[row];
                    let at = out.row_of[t][k] as usize * dim;
                    out.beliefs[at..at + dim].copy_from_slice(&flat[row * dim..(row + 1) * dim]);
                }
                start += rows;
            }
            cursor += emit as usize;
        }
    }

    for (t, forecast) in out.one_bar.iter().enumerate() {
        ensure!(
            forecast.kelly_f.iter().all(|f| f.is_finite())
                && out.mean_simple[t].iter().all(|m| m.is_finite())
                && out.second_simple[t].iter().all(|m| m.is_finite())
                && out.mu_log[t].iter().all(|m| m.is_finite())
                && out.var_log[t].iter().all(|v| v.is_finite()),
            "instant {t} has a symbol the belief pass never reached, so its position would be \
             sized from a NaN"
        );
    }
    Ok(out)
}

fn host_f32(tensor: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(tensor.to_kind(Kind::Float).reshape([-1]).to(Device::Cpu))
        .expect("a float tensor converts to a host vector")
}

fn host_f64(tensor: &Tensor) -> Vec<f64> {
    Vec::<f64>::try_from(tensor.to_kind(Kind::Double).reshape([-1]).to(Device::Cpu))
        .expect("a double tensor converts to a host vector")
}

// ---------------------------------------------------------------------------
// The rebalance schedule and the realized aggregate
// ---------------------------------------------------------------------------

/// One name's entry in one holding period.
#[derive(Clone, Copy, Debug)]
pub struct Leg {
    /// Panel symbol id.
    pub id: u32,
    /// Slot of the name in the rebalance instant's slice.
    pub slot: usize,
    /// Row of [`PanelBeliefs`] holding the decision-time belief.
    pub row: u32,
    /// Model steps in the holding window. Legacy non-overlapping periods may be shorter at a
    /// symbol boundary; receding periods always equal their declared forecast horizon exactly.
    pub steps: usize,
    /// Realized aggregate LOG return of a position held across the window:
    /// `ln(close(exit) / close(entry))`, where entry is the close at the panel instant BEFORE
    /// the rebalance and exit is the symbol's last close inside the window.
    pub realized_log: f64,
    /// Dollar ADV at the rebalance, trailing and strictly causal.
    pub adv_usd: f64,
}

/// One non-overlapping holding period.
#[derive(Clone, Debug)]
pub struct Period {
    /// Panel instant the rebalance happens at, and whose slice defines who is tradeable.
    pub instant: usize,
    pub ts_ms: i64,
    pub legs: Vec<Leg>,
}

/// Cut the panel into non-overlapping `k`-bar holding periods on the phase `t = 0, k, 2k...`
///
/// The exit mark is read off the corpus rather than accumulated from the panel's per-instant
/// returns, because a symbol that skips a panel instant has no panel return there while its
/// HELD position still earned the move: `ln(close(b + m - 1) / close(b - 1))` over the
/// symbol's own consecutive bars is the truth, and it needs no forward fill.
pub fn schedule(
    corpus: &BarCorpus,
    panel: &Panel,
    beliefs: &PanelBeliefs,
    k: usize,
) -> Result<Vec<Period>> {
    ensure!(k >= 1, "a holding period is at least one bar");
    let instants = panel.instants();
    let mut periods = Vec::with_capacity(instants.div_ceil(k));
    for start in (0..instants).step_by(k) {
        let slice = &panel.slices()[start];
        let end_instant = (start + k - 1).min(instants - 1);
        let window_end_ts = panel.slices()[end_instant].ts_ms;
        let mut legs = Vec::with_capacity(slice.symbols.len());
        for (slot, &id) in slice.symbols.iter().enumerate() {
            let series = panel.series_of(id);
            let bars = corpus.bars(series);
            let entry_bar = panel.bar_index(start, slot) as usize;
            ensure!(
                entry_bar >= 1,
                "{} is tradeable at its first bar, which has no entry close",
                panel.symbols()[id as usize]
            );
            let entry_close = f64::from(bars[entry_bar - 1].close);
            // Walk the symbol's OWN consecutive bars while they stay inside the window and
            // carry a usable close. The first is guaranteed by the panel.
            let mut steps = 0usize;
            let mut exit_close = entry_close;
            while steps < k && entry_bar + steps < bars.len() {
                let bar = bars[entry_bar + steps];
                if bar.ts() > window_end_ts {
                    break;
                }
                let close = f64::from(bar.close);
                if !(close > 0.0) {
                    break;
                }
                steps += 1;
                exit_close = close;
            }
            ensure!(
                steps >= 1 && entry_close > 0.0,
                "{} is in the panel at instant {start} but has no usable held bar there",
                panel.symbols()[id as usize]
            );
            let realized_log = (exit_close / entry_close).ln();
            ensure!(
                realized_log.is_finite(),
                "{} has a non-finite aggregate return over instant {start}",
                panel.symbols()[id as usize]
            );
            legs.push(Leg {
                id,
                slot,
                row: beliefs.row_of[start][slot],
                steps,
                realized_log,
                adv_usd: f64::from(panel.adv_usd(start, slot)),
            });
        }
        ensure!(
            !legs.is_empty(),
            "instant {start} is a rebalance with nothing tradeable"
        );
        periods.push(Period {
            instant: start,
            ts_ms: slice.ts_ms,
            legs,
        });
    }
    ensure!(
        periods.len() >= 2,
        "a {k}-bar holding period leaves {} periods in a {instants}-instant panel, which is \
         not a book",
        periods.len()
    );
    Ok(periods)
}

// ---------------------------------------------------------------------------
/// Build an overlapping forecast schedule on the fixed one-bar decision clock.
///
/// Each retained decision has a complete `horizon`-step deterministic exchange-calendar
/// clock and a realized mark through its final scheduled timestamp. Rows at the data boundary
/// without that evidence are explicitly excluded. A symbol halt does not shorten the horizon:
/// its mark is carried until its next print, and the full move is applied when that print falls
/// inside the window.
pub fn receding_schedule(
    corpus: &BarCorpus,
    panel: &Panel,
    beliefs: &PanelBeliefs,
    horizon: usize,
) -> Result<Vec<Period>> {
    ensure!(horizon >= 1, "a forecast horizon is at least one bar");
    let last_evidence_ts = panel
        .slices()
        .last()
        .expect("a panel has at least one instant")
        .ts_ms;
    let mut periods = Vec::with_capacity(panel.instants());
    for start in 0..panel.instants() {
        let slice = &panel.slices()[start];
        let window_end_ts = if horizon == 1 {
            slice.ts_ms
        } else {
            *forecast_schedule_after(slice.ts_ms, horizon - 1, corpus.res_secs())
                .last()
                .expect("a positive remaining horizon has a final timestamp")
        };
        if window_end_ts > last_evidence_ts {
            break;
        }
        let mut legs = Vec::with_capacity(slice.symbols.len());
        for (slot, &id) in slice.symbols.iter().enumerate() {
            let series = panel.series_of(id);
            let bars = corpus.bars(series);
            let entry_bar = panel.bar_index(start, slot) as usize;
            let entry_close = f64::from(bars[entry_bar - 1].close);
            let end = bars.partition_point(|bar| bar.ts() <= window_end_ts);
            let mut exit_close = entry_close;
            for bar in &bars[entry_bar..end] {
                if bar.close > 0.0 {
                    exit_close = f64::from(bar.close);
                }
            }
            let realized_log = (exit_close / entry_close).ln();
            ensure!(
                realized_log.is_finite(),
                "{} has no finite mark through the H={horizon} endpoint at instant {start}",
                panel.symbols()[id as usize]
            );
            legs.push(Leg {
                id,
                slot,
                row: beliefs.row_of[start][slot],
                steps: horizon,
                realized_log,
                adv_usd: f64::from(panel.adv_usd(start, slot)),
            });
        }
        periods.push(Period {
            instant: start,
            ts_ms: slice.ts_ms,
            legs,
        });
    }
    ensure!(
        periods.len() >= 2,
        "H={horizon} leaves only {} fully observed decisions out of {} panel rows",
        periods.len(),
        panel.instants()
    );
    debug_assert!(
        periods
            .iter()
            .all(|period| period.legs.iter().all(|leg| leg.steps == horizon)),
        "a declared receding horizon must never be shortened"
    );
    Ok(periods)
}

// The k-bar predictive law
// ---------------------------------------------------------------------------

/// The `k`-bar aggregate law of one leg, as the two moments the closure needs plus the
/// evidence that the estimator worked.
#[derive(Clone, Copy, Debug, Default)]
pub struct AggregateLaw {
    /// Rao-Blackwellized `E[sum_j r_j | past]`, in nats.
    pub mu_log: f64,
    /// Plain sampled-mean estimate of the same quantity, in nats. Reported beside `mu_log`
    /// because the ratio of their standard errors is the variance reduction the module docs
    /// claim, measured on the real panel.
    pub plain_mu_log: f64,
    /// Variance of the aggregate log return, in nats squared. Exact under the fitted
    /// categorical one-bar law at H1; sampled from ancestral paths at H>1.
    pub var_log: f64,
    /// Monte-Carlo standard error of [`Self::mu_log`], in nats. This is exactly zero at H1.
    pub mu_se: f64,
    /// Monte-Carlo standard error of [`Self::plain_mu_log`], in nats.
    pub plain_mu_se: f64,
    /// First moment of the cumulative SIMPLE return. This is reduced exactly from the fitted
    /// categorical law at H1. At H>1 it averages, over ancestral categorical paths, the
    /// conditional compounded gross first moment `product(1 + E[R | bin])`; the decoded
    /// within-bin draws advance the state but do not price the return law. It is the Kelly
    /// numerator and is not `expm1(mu_log)`.
    pub mean_simple: f64,
    /// Second raw moment of the cumulative SIMPLE return. This is reduced exactly from the
    /// fitted categorical law at H1. At H>1 it uses the same bin paths and compounds
    /// `product(1 + 2 E[R | bin] + E[R² | bin])`, preserving fitted within-bin variance,
    /// including the catch-all bins. It is the Kelly denominator and is never replaced by
    /// `mean_simple * mean_simple`.
    pub second_simple: f64,
    /// Prefix moments from the same conditionally compounded forecast law, indexed by
    /// [`FORECAST_HORIZONS`]. The H1 prefix is the exact fitted categorical reduction; later
    /// prefixes are Rao-Blackwellized over ancestral categorical paths.
    pub prefix_mean_simple: [f64; FORECAST_HORIZONS.len()],
    pub prefix_second_simple: [f64; FORECAST_HORIZONS.len()],
}

impl AggregateLaw {
    /// The analytic one-bar lognormal closure lifted into the aggregate shape.
    fn analytic_one_bar(mu_log: f64, var_log: f64) -> Self {
        let first = (mu_log + 0.5 * var_log).exp_m1();
        let second =
            (2.0 * mu_log + 2.0 * var_log).exp() - 2.0 * (mu_log + 0.5 * var_log).exp() + 1.0;
        Self {
            mu_log,
            plain_mu_log: mu_log,
            var_log,
            mu_se: 0.0,
            plain_mu_se: 0.0,
            mean_simple: first,
            second_simple: second.max(0.0),
            prefix_mean_simple: [first; FORECAST_HORIZONS.len()],
            prefix_second_simple: [second.max(0.0); FORECAST_HORIZONS.len()],
        }
    }
}

/// The second-order Kelly fraction of a buy-and-hold over an aggregate whose log return has
/// mean `mu_log` and variance `var_log`, under the lognormal closure derived in the module
/// docs. Clamped at [`FREE_LEVERAGE`], the same effectively-uncapped bound used by the
/// categorical quadratic reduction.
pub fn closure_kelly(mu_log: f64, var_log: f64) -> f64 {
    let var = var_log.max(VARIANCE_FLOOR);
    let m1 = (mu_log + 0.5 * var).exp();
    let m2 = (2.0 * mu_log + 2.0 * var).exp();
    let mean = m1 - 1.0;
    let second = m2 - 2.0 * m1 + 1.0;
    if !(second > 0.0) || !mean.is_finite() {
        return 0.0;
    }
    (mean / second).clamp(-FREE_LEVERAGE, FREE_LEVERAGE)
}

/// Variance of the SIMPLE aggregate return under the same lognormal closure, so the sizing
/// and the independence diagnostic describe one law.
fn closure_simple_var(mu_log: f64, var_log: f64) -> f64 {
    let var = var_log.max(0.0);
    ((2.0 * mu_log + 2.0 * var).exp() - (2.0 * mu_log + var).exp()).max(0.0)
}

/// Sample the `k`-bar aggregate law of every leg of every period.
///
/// One entry per period, aligned with [`Period::legs`]. The rollout is
/// `RolloutMode::Dynamics`'s mechanism run directly on cached beliefs; see the module docs for
/// why `imagine` is not called, and the tests for the proof that it is the same mechanism.
///
/// Batching is across (period, leg) pairs, which are independent, so the sequential depth is
/// the number of steps and not the number of rebalances.
pub fn horizon_laws(
    model: &BarWorldModel,
    corpus: &BarCorpus,
    panel: &Panel,
    beliefs: &PanelBeliefs,
    periods: &[Period],
    res_secs: u32,
    samples: usize,
) -> Result<Vec<Vec<AggregateLaw>>> {
    ensure!(samples >= 2, "a sampled law needs at least two paths");
    let supports = model
        .supports_for(res_secs)
        .with_context(|| format!("the checkpoint carries no supports at {res_secs}s"))?;
    let device = model.device();
    let head = model.head();
    let dynamics = model.dynamics();
    let trunk = model.trunk();
    let centers_host: Vec<f64> = supports.centers(DOF_R).to_vec();
    let centers = Tensor::from_slice(&centers_host)
        .view([NUM_BAR_BINS, 1])
        .to_device(device)
        .to_kind(Kind::Double);
    let (simple_first_host, simple_second_host) = supports
        .simple_return_bin_moments()
        .context("supports lack fitted simple-return moments; refit the v6 artifact")?;
    let simple_first = Tensor::from_slice(simple_first_host)
        .to_device(device)
        .to_kind(Kind::Double);
    let simple_second = Tensor::from_slice(simple_second_host)
        .to_device(device)
        .to_kind(Kind::Double);

    let mut out: Vec<Vec<AggregateLaw>> = periods
        .iter()
        .map(|p| vec![AggregateLaw::default(); p.legs.len()])
        .collect();

    // Flatten to (period, leg) pairs so one chunk can span rebalances.
    let pairs: Vec<(usize, usize)> = periods
        .iter()
        .enumerate()
        .flat_map(|(p, period)| (0..period.legs.len()).map(move |l| (p, l)))
        .collect();
    let chunk_pairs = (SAMPLE_ROW_LIMIT / samples).max(1);
    let samples_i = samples as i64;

    for chunk in pairs.chunks(chunk_pairs) {
        let count = chunk.len() as i64;
        let deepest = chunk
            .iter()
            .map(|&(p, l)| periods[p].legs[l].steps)
            .max()
            .expect("a chunk has at least one pair");
        if deepest == DEFAULT_FORECAST_HORIZON {
            ensure!(
                chunk
                    .iter()
                    .all(|&(p, l)| periods[p].legs[l].steps == deepest),
                "an H={DEFAULT_FORECAST_HORIZON} rollout chunk contains a shortened leg"
            );
        }
        // Every leg at a decision shares one deterministic exchange-calendar clock. It is
        // generated from the decision row and contains exactly `deepest` model steps; no
        // future symbol timestamp, print, halt or availability is consulted. Legacy holding
        // periods can still have shorter legs, which are masked from their aggregate law.
        let mut clock = vec![0i64; chunk.len() * deepest * BAR_TIME_FEATURES];
        let mut current_clock = vec![0i64; chunk.len() * BAR_TIME_FEATURES];
        let mut active = vec![0.0f64; chunk.len() * deepest];
        let mut flat = vec![0.0f32; chunk.len() * BAR_MODEL_DIM as usize];
        for (index, &(p, l)) in chunk.iter().enumerate() {
            let leg = periods[p].legs[l];
            let series = panel.series_of(leg.id);
            let bars = corpus.bars(series);
            let entry_bar = panel.bar_index(periods[p].instant, leg.slot) as usize;
            let current_bar = entry_bar
                .checked_sub(1)
                .expect("a tradeable entry has a predecessor");
            let current_ids = bar_time_ids(
                bars[current_bar].ts(),
                current_bar.checked_sub(1).map(|prev| bars[prev].ts()),
                res_secs,
                corpus.market_channel(),
            );
            let current_at = index * BAR_TIME_FEATURES;
            current_clock[current_at..current_at + BAR_TIME_FEATURES].copy_from_slice(&current_ids);
            let scheduled = forecast_schedule_ids_from(periods[p].ts_ms, deepest, res_secs);
            for (step, ids) in scheduled.into_iter().enumerate() {
                active[index * deepest + step] = f64::from(u8::from(step < leg.steps));
                let at = (index * deepest + step) * BAR_TIME_FEATURES;
                clock[at..at + BAR_TIME_FEATURES].copy_from_slice(&ids);
            }
            let dim = BAR_MODEL_DIM as usize;
            flat[index * dim..(index + 1) * dim].copy_from_slice(beliefs.belief_row(leg.row));
        }
        let clock = Tensor::from_slice(&clock)
            .view([count, deepest as i64, BAR_TIME_FEATURES as i64])
            .to_device(device);
        let current_clock = Tensor::from_slice(&current_clock)
            .view([count, BAR_TIME_FEATURES as i64])
            .to_device(device);
        let active = Tensor::from_slice(&active)
            .view([count, deepest as i64])
            .to_device(device);
        let seed = Tensor::from_slice(&flat)
            .view([count, BAR_MODEL_DIM])
            .to_device(device);

        let (rb, plain, gross_first, gross_second, prefix_gross) = tch::no_grad(|| {
            // One row per (pair, path), interleaved so row `pair * samples + n` is path `n`.
            let mut h = seed.repeat_interleave_self_int(samples_i, 0, None);
            let first_current = current_clock.repeat_interleave_self_int(samples_i, 0, None);
            let total = count * samples_i;
            let mut sum_m = Tensor::zeros([total], (Kind::Double, device));
            let mut sum_r = Tensor::zeros([total], (Kind::Double, device));
            let mut gross_first = Tensor::ones([total], (Kind::Double, device));
            let mut gross_second = Tensor::ones([total], (Kind::Double, device));
            let mut prefix_gross = Vec::with_capacity(FORECAST_HORIZONS.len());
            let mut next_prefix = 0usize;
            for step in 0..deepest as i64 {
                let live = active
                    .select(1, step)
                    .repeat_interleave_self_int(samples_i, 0, None);
                let ids = clock
                    .select(1, step)
                    .repeat_interleave_self_int(samples_i, 0, None);
                let current = if step == 0 {
                    first_current.shallow_clone()
                } else {
                    clock
                        .select(1, step - 1)
                        .repeat_interleave_self_int(samples_i, 0, None)
                };
                let conditioning = trunk.forecast_conditioning(&ids, &current);
                // The DECISION moment, BEFORE this bar is drawn: the conditional mean of its
                // log return under the head's prefix-free `r` row. Autocast off, for the
                // reason stated in `scan_panel`, and row-chunked so the peak stays bounded by
                // `ROW_CHUNK` rather than by the sample count.
                let m = tch::autocast(false, || {
                    let mut parts = Vec::with_capacity((total / ROW_CHUNK + 1) as usize);
                    let mut at = 0i64;
                    while at < total {
                        let rows = ROW_CHUNK.min(total - at);
                        let probs = forecast_r_probs(
                            head,
                            &h.narrow(0, at, rows),
                            &conditioning.narrow(0, at, rows),
                        )
                        .to_kind(Kind::Double);
                        parts.push(probs.matmul(&centers).reshape([-1]));
                        at += rows;
                    }
                    Tensor::cat(&parts, 0)
                });
                sum_m += &m * &live;

                // Keep the decoded draw for the autoregressive state and the plain log-return
                // diagnostic, but retain the exact categorical bins the chain sampled. The
                // economic law integrates out the within-bin draw using the v6 fitted moments:
                // E[1+R | b] = 1+m1_b and E[(1+R)^2 | b] = 1+2m1_b+m2_b.
                let (dof, bins) =
                    head.sample_binned(&h, &conditioning, supports, ROLLOUT_TEMPERATURE);
                let drawn = dof
                    .select(1, DOF_R as i64)
                    .to_kind(Kind::Double)
                    .reshape([-1]);
                sum_r += &drawn * &live;
                let r_bins = bins.select(1, DOF_R as i64).reshape([-1]);
                let m1 = simple_first.index_select(0, &r_bins);
                let m2 = simple_second.index_select(0, &r_bins);
                gross_first = &gross_first * (1.0 + &m1 * &live);
                gross_second = &gross_second * (1.0 + (2.0 * &m1 + &m2) * &live);
                while next_prefix < FORECAST_HORIZONS.len()
                    && FORECAST_HORIZONS[next_prefix] <= step as usize + 1
                {
                    prefix_gross.push((gross_first.shallow_clone(), gross_second.shallow_clone()));
                    next_prefix += 1;
                }
                let token = trunk
                    .token_embedding(&dof.unsqueeze(1), &bins.unsqueeze(1), &ids.unsqueeze(1))
                    .squeeze_dim(1);
                h = dynamics.step(&h, &token);
            }
            while prefix_gross.len() < FORECAST_HORIZONS.len() {
                prefix_gross.push((gross_first.shallow_clone(), gross_second.shallow_clone()));
            }
            (
                sum_m.view([count, samples_i]),
                sum_r.view([count, samples_i]),
                gross_first.view([count, samples_i]),
                gross_second.view([count, samples_i]),
                prefix_gross,
            )
        });

        let root = (samples as f64).sqrt();
        let rb_mu = host_f64(&rb.mean_dim([1i64].as_slice(), false, Kind::Double));
        let rb_sd = host_f64(&rb.std_dim([1i64].as_slice(), true, false));
        let plain_mu = host_f64(&plain.mean_dim([1i64].as_slice(), false, Kind::Double));
        let plain_sd = host_f64(&plain.std_dim([1i64].as_slice(), true, false));
        let simple = &gross_first - 1.0;
        let conditional_second: Tensor = (&gross_second - &gross_first * 2.0 + 1.0).clamp_min(0.0);
        let mean_simple = host_f64(&simple.mean_dim([1i64].as_slice(), false, Kind::Double));
        let second_simple =
            host_f64(&conditional_second.mean_dim([1i64].as_slice(), false, Kind::Double));
        let mut prefix_means = Vec::with_capacity(FORECAST_HORIZONS.len());
        let mut prefix_seconds = Vec::with_capacity(FORECAST_HORIZONS.len());
        for (prefix_first, prefix_second) in prefix_gross {
            let prefix_first = prefix_first.view([count, samples_i]);
            let prefix_second = prefix_second.view([count, samples_i]);
            let simple = &prefix_first - 1.0;
            let conditional_second: Tensor =
                (&prefix_second - &prefix_first * 2.0 + 1.0).clamp_min(0.0);
            prefix_means.push(host_f64(&simple.mean_dim(
                [1i64].as_slice(),
                false,
                Kind::Double,
            )));
            prefix_seconds.push(host_f64(&conditional_second.mean_dim(
                [1i64].as_slice(),
                false,
                Kind::Double,
            )));
        }
        for (index, &(p, l)) in chunk.iter().enumerate() {
            let period = &periods[p];
            let leg = period.legs[l];
            let exact_mean = beliefs.mean_simple[period.instant][leg.slot];
            let exact_second = beliefs.second_simple[period.instant][leg.slot];
            let one_bar = leg.steps == 1;
            let sampled_var = plain_sd[index] * plain_sd[index];
            let mut prefix_mean_simple = std::array::from_fn(|h| prefix_means[h][index]);
            let mut prefix_second_simple = std::array::from_fn(|h| prefix_seconds[h][index]);
            // H1 is a categorical law whose fitted within-bin simple-return moments are
            // available exactly. Sampling it would needlessly add both multinomial and
            // within-bin noise to the numerator and denominator that size the position.
            prefix_mean_simple[0] = exact_mean;
            prefix_second_simple[0] = exact_second;
            if one_bar {
                // A shortened legacy leg has no later aggregate prefix: every exposed prefix
                // is its same exact one-bar law.
                prefix_mean_simple.fill(exact_mean);
                prefix_second_simple.fill(exact_second);
            }
            let (mu_log, var_log, mu_se, aggregate_mean, aggregate_second) = if one_bar {
                (
                    beliefs.mu_log[period.instant][leg.slot],
                    beliefs.var_log[period.instant][leg.slot],
                    0.0,
                    exact_mean,
                    exact_second,
                )
            } else {
                (
                    rb_mu[index],
                    sampled_var,
                    rb_sd[index] / root,
                    mean_simple[index],
                    second_simple[index],
                )
            };
            ensure!(
                mu_log.is_finite()
                    && var_log.is_finite()
                    && aggregate_mean.is_finite()
                    && aggregate_second.is_finite(),
                "the {}-step rollout of {} at instant {} produced a non-finite law",
                leg.steps,
                panel.symbols()[leg.id as usize],
                period.instant
            );
            out[p][l] = AggregateLaw {
                mu_log,
                plain_mu_log: plain_mu[index],
                var_log,
                mu_se,
                plain_mu_se: plain_sd[index] / root,
                mean_simple: aggregate_mean,
                second_simple: aggregate_second,
                prefix_mean_simple,
                prefix_second_simple,
            };
        }
    }
    Ok(out)
}

/// The law every policy is sized from at one `k` under one construction, aligned with
/// [`Period::legs`], plus the null's own fraction.
#[derive(Clone, Debug)]
pub struct HorizonInputs {
    pub construction: Construction,
    /// `kelly[p][l]` is the uncapped preference of the model policies.
    pub kelly: Vec<Vec<f64>>,
    /// Predicted variance of the held aggregate's SIMPLE return, for the independence
    /// diagnostic behind [`HorizonMetrics::leverage_error`].
    pub pred_var: Vec<Vec<f64>>,
    /// The unconditional-marginal null's fraction. ONE number: every present name shares it,
    /// which is why its value cannot move the null's book at all — the gross projection is
    /// scale-free on a constant vector, and a test pins that.
    pub marginal_kelly: f64,
    /// Per-leg law summaries, for the mechanism columns.
    pub laws: Vec<Vec<AggregateLaw>>,
}

/// Assemble the sizing inputs for one construction.
pub fn build_inputs(
    construction: Construction,
    beliefs: &PanelBeliefs,
    periods: &[Period],
    sampled: Option<&[Vec<AggregateLaw>]>,
    marginal: &PanelForecast,
) -> Result<HorizonInputs> {
    let mut kelly = Vec::with_capacity(periods.len());
    let mut pred_var = Vec::with_capacity(periods.len());
    let mut laws = Vec::with_capacity(periods.len());
    for (p, period) in periods.iter().enumerate() {
        let mut k_row = Vec::with_capacity(period.legs.len());
        let mut v_row = Vec::with_capacity(period.legs.len());
        let mut l_row = Vec::with_capacity(period.legs.len());
        for (l, leg) in period.legs.iter().enumerate() {
            let law = match construction {
                Construction::Stale => AggregateLaw::analytic_one_bar(
                    beliefs.mu_log[period.instant][leg.slot],
                    beliefs.var_log[period.instant][leg.slot],
                ),
                Construction::StaleCategoricalMoments => {
                    let mut law = AggregateLaw::analytic_one_bar(
                        beliefs.mu_log[period.instant][leg.slot],
                        beliefs.var_log[period.instant][leg.slot],
                    );
                    law.mean_simple = beliefs.mean_simple[period.instant][leg.slot];
                    law.second_simple = beliefs.second_simple[period.instant][leg.slot];
                    law
                }
                Construction::Horizon => {
                    let sampled = sampled.context(
                        "the horizon construction needs sampled aggregate laws and got none",
                    )?;
                    ensure!(
                        sampled.len() == periods.len() && sampled[p].len() == period.legs.len(),
                        "the sampled laws do not align with the schedule"
                    );
                    sampled[p][l]
                }
            };
            let f = match construction {
                Construction::StaleCategoricalMoments => {
                    f64::from(beliefs.one_bar[period.instant].kelly_f[leg.slot])
                }
                Construction::Stale => closure_kelly(law.mu_log, law.var_log),
                Construction::Horizon if leg.steps == 1 => {
                    f64::from(beliefs.one_bar[period.instant].kelly_f[leg.slot])
                }
                Construction::Horizon if law.second_simple > 0.0 => {
                    (law.mean_simple / law.second_simple).clamp(-FREE_LEVERAGE, FREE_LEVERAGE)
                }
                Construction::Horizon => 0.0,
            };
            let variance = match construction {
                Construction::Horizon => {
                    (law.second_simple - law.mean_simple * law.mean_simple).max(0.0)
                }
                Construction::StaleCategoricalMoments => {
                    f64::from(beliefs.one_bar[period.instant].var_r[leg.slot]).max(0.0)
                }
                Construction::Stale => closure_simple_var(law.mu_log, law.var_log),
            };
            k_row.push(f);
            v_row.push(variance);
            l_row.push(law);
        }
        kelly.push(k_row);
        pred_var.push(v_row);
        laws.push(l_row);
    }
    let marginal_kelly = marginal.kelly_f.first().map_or(0.0, |f| f64::from(*f));
    Ok(HorizonInputs {
        construction,
        kelly,
        pred_var,
        marginal_kelly,
        laws,
    })
}

// ---------------------------------------------------------------------------
// Canonical every-bar receding-horizon evaluation
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ForecastMoment {
    pub mean_simple: f64,
    pub second_simple: f64,
    /// Uncapped quadratic Kelly `E[R] / E[R²]`, retained as a forecast diagnostic.
    pub frictionless_kelly: f64,
}

/// One common ancestral rollout, exposed at every production prefix.
#[derive(Clone, Debug)]
pub struct RecedingForecasts {
    /// `moments[h][p][k]` aligns with the kth symbol in `periods[p]`'s panel slice.
    pub moments: Vec<Vec<Vec<ForecastMoment>>>,
}

impl RecedingForecasts {
    pub fn from_common_rollout(
        panel: &Panel,
        beliefs: &PanelBeliefs,
        periods: &[Period],
        laws: &[Vec<AggregateLaw>],
    ) -> Result<Self> {
        ensure!(
            laws.len() == periods.len(),
            "the common rollout has {} rows for {} eligible decisions",
            laws.len(),
            periods.len()
        );
        let mut moments = vec![Vec::with_capacity(periods.len()); FORECAST_HORIZONS.len()];
        for (p, period) in periods.iter().enumerate() {
            let t = period.instant;
            let slice = &panel.slices()[t];
            ensure!(
                laws[p].len() == slice.symbols.len(),
                "rollout row {p} does not align with panel instant {t}"
            );
            ensure!(
                period
                    .legs
                    .iter()
                    .all(|leg| leg.steps == DEFAULT_FORECAST_HORIZON),
                "common rollout row {p} silently shortened H={DEFAULT_FORECAST_HORIZON}"
            );
            for (h, _) in FORECAST_HORIZONS.iter().enumerate() {
                let mut row = Vec::with_capacity(slice.symbols.len());
                for (k, law) in laws[p].iter().enumerate() {
                    let (mean, second, frictionless) = if h == 0 {
                        let mean = beliefs.mean_simple[t][k];
                        let second = beliefs.second_simple[t][k].max(0.0);
                        let fraction = if second > 0.0 { mean / second } else { 0.0 };
                        (mean, second, fraction)
                    } else {
                        let mean = law.prefix_mean_simple[h];
                        let second = law.prefix_second_simple[h].max(0.0);
                        let fraction = if second > 0.0 { mean / second } else { 0.0 };
                        (mean, second, fraction)
                    };
                    ensure!(
                        mean.is_finite() && second.is_finite() && frictionless.is_finite(),
                        "non-finite forecast moment at horizon {} row {p} (panel instant {t})",
                        FORECAST_HORIZONS[h]
                    );
                    row.push(ForecastMoment {
                        mean_simple: mean,
                        second_simple: second,
                        frictionless_kelly: frictionless,
                    });
                }
                moments[h].push(row);
            }
        }
        Ok(Self { moments })
    }

    pub fn horizon(&self, horizon: usize) -> Option<&[Vec<ForecastMoment>]> {
        FORECAST_HORIZONS
            .iter()
            .position(|h| *h == horizon)
            .map(|i| self.moments[i].as_slice())
    }
}

/// Train-fitted unconditional simple-return law compounded without peeking at evaluation.
pub fn marginal_horizon_moment(supports: &BarSupports, horizon: usize) -> Result<ForecastMoment> {
    ensure!(horizon >= 1, "a marginal horizon is at least one bar");
    let (first_by_bin, second_by_bin) = supports
        .simple_return_bin_moments()
        .context("supports lack fitted simple-return moments; refit the v6 support artifact")?;
    let masses = supports.bin_masses(DOF_R);
    ensure!(
        masses.len() == first_by_bin.len() && masses.len() == second_by_bin.len(),
        "support masses and simple-return moments do not align"
    );
    let first = masses
        .iter()
        .zip(first_by_bin)
        .map(|(p, m)| p * m)
        .sum::<f64>();
    let second = masses
        .iter()
        .zip(second_by_bin)
        .map(|(p, m)| p * m)
        .sum::<f64>();
    let gross_first = 1.0 + first;
    let gross_second = 1.0 + 2.0 * first + second;
    let compounded_first = gross_first.powi(horizon as i32);
    let mean = compounded_first - 1.0;
    let second = (gross_second.powi(horizon as i32) - 2.0 * compounded_first + 1.0).max(0.0);
    Ok(ForecastMoment {
        mean_simple: mean,
        second_simple: second,
        frictionless_kelly: if second > 0.0 { mean / second } else { 0.0 },
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecedingPolicy {
    Model,
    Marginal,
    EqualWeight,
    BuyHold,
    /// Deterministic realized-H moment benchmark. It is perfect foresight under the shared
    /// quadratic solver, not a claim of pathwise dominance under a different exact objective.
    Oracle,
}

pub const RECEDING_POLICIES: [RecedingPolicy; 5] = [
    RecedingPolicy::Model,
    RecedingPolicy::Marginal,
    RecedingPolicy::EqualWeight,
    RecedingPolicy::BuyHold,
    RecedingPolicy::Oracle,
];

impl RecedingPolicy {
    pub fn name(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Marginal => "marginal null",
            Self::EqualWeight => "equal weight",
            Self::BuyHold => "buy and hold",
            Self::Oracle => "deterministic perfect-foresight oracle",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RecedingRun {
    pub policy: RecedingPolicy,
    pub horizon: usize,
    pub reforecasts: usize,
    pub actions: usize,
    pub log_equity: Vec<f64>,
    /// Panel indices actually scored. Boundary rows without a full `horizon` are absent.
    pub decision_instants: Vec<usize>,
    /// Calendar years from the first scored decision to the last. Stored so stdout can use the
    /// same run-specific annualizer as reports after the panel has been released.
    pub decision_span_years: f64,
    pub turnover: f64,
    pub execution_cost: f64,
    pub max_gross: f64,
    pub max_abs_net: f64,
    pub max_name: f64,
    pub max_participation: f64,
    pub covariance_observations: usize,
    pub covariance_shrinkage: f64,
    pub cost_month_substitutions: usize,
    pub cost_cross_section_substitutions: usize,
    pub covariance_window: usize,
    pub mean_factor_variance: f64,
}

impl RecedingRun {
    pub fn annual_growth_dispersion(&self, panel: &Panel) -> Dispersion {
        let values: Vec<f64> = self
            .log_equity
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect();
        let blocks: Vec<u64> = self
            .decision_instants
            .iter()
            .map(|&instant| panel.slices()[instant].ts_ms.div_euclid(86_400_000) as u64)
            .collect();
        let mut result = block_bootstrap(&values, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
        let annualizer = self.instants_per_year(panel);
        result.mean *= annualizer;
        result.se *= annualizer;
        result.ci_low *= annualizer;
        result.ci_high *= annualizer;
        result
    }
    pub fn annual_log_growth(&self, panel: &Panel) -> f64 {
        let measured_span = self.span_years(panel);
        assert!(
            (self.decision_span_years.is_nan() && measured_span.is_nan())
                || self.decision_span_years == measured_span,
            "recorded receding annualizer must match the run's decision instants"
        );
        self.recorded_annual_log_growth()
    }
    fn recorded_annual_log_growth(&self) -> f64 {
        self.log_equity.last().copied().unwrap_or(0.0) / self.decision_span_years
    }

    pub fn annual_difference_dispersion(&self, baseline: &Self, panel: &Panel) -> Dispersion {
        let values: Vec<f64> = self
            .log_equity
            .windows(2)
            .zip(baseline.log_equity.windows(2))
            .map(|(model, null)| (model[1] - model[0]) - (null[1] - null[0]))
            .collect();
        assert_eq!(
            self.decision_instants, baseline.decision_instants,
            "paired runs must score identical eligible rows"
        );
        let blocks: Vec<u64> = self
            .decision_instants
            .iter()
            .map(|&instant| panel.slices()[instant].ts_ms.div_euclid(86_400_000) as u64)
            .collect();
        let mut result = block_bootstrap(&values, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED);
        let annualizer = self.instants_per_year(panel);
        result.mean *= annualizer;
        result.se *= annualizer;
        result.ci_low *= annualizer;
        result.ci_high *= annualizer;
        result
    }

    fn span_years(&self, panel: &Panel) -> f64 {
        self.decision_instants
            .first()
            .zip(self.decision_instants.last())
            .map(|(&first, &last)| {
                (panel.slices()[last].ts_ms - panel.slices()[first].ts_ms) as f64
                    / (365.25 * 86_400_000.0)
            })
            .unwrap_or(f64::NAN)
    }

    fn instants_per_year(&self, panel: &Panel) -> f64 {
        self.decision_instants.len() as f64 / self.span_years(panel)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RecedingConfig {
    pub capital_usd: f64,
    pub constraints: KellyConstraints,
    pub covariance_window: usize,
    pub covariance_shrinkage: f64,
}

impl Default for RecedingConfig {
    fn default() -> Self {
        Self {
            capital_usd: 1.0e7,
            constraints: KellyConstraints::default(),
            covariance_window: 20 * 93,
            covariance_shrinkage: 0.25,
        }
    }
}

/// Run one policy while recomputing its forecast and action on every boundary-eligible row.
///
/// Model risk keeps the strictly trailing covariance shape, scales it to `horizon`, and
/// reconciles its diagonal to `max(E[R_H²] - E[R_H]², 0)`. The shared solver then adds the
/// predicted-mean outer product. The oracle uses zero covariance plus its deterministic
/// realized-H mean outer product. Both use identical costs, drifted holdings and constraints.
#[allow(clippy::too_many_arguments)]
pub fn run_receding_book(
    panel: &Panel,
    moments: &[Vec<ForecastMoment>],
    oracle_periods: &[Period],
    marginal: ForecastMoment,
    policy: RecedingPolicy,
    horizon: usize,
    cost: &dyn CostModel,
    config: RecedingConfig,
) -> Result<RecedingRun> {
    ensure!(horizon >= 1, "a forecast horizon is at least one bar");
    ensure!(
        moments.len() == oracle_periods.len(),
        "model and oracle must score the same boundary-eligible rows"
    );
    let names = panel.symbols().len();
    let symbol_ids: Vec<u32> = (0..names as u32).collect();
    let mut trailing = TrailingFactorCovariance::new(
        names,
        config.covariance_window,
        config.covariance_shrinkage,
    )?;
    let mut held = vec![0.0; names];
    let mut adv = vec![0.0; names];
    let zero_second = vec![0.0; names];
    let mut one_bar_fallback = vec![0.0; names];
    let mut log_wealth: f64 = 0.0;
    let mut observed_until = 0usize;
    let mut run = RecedingRun {
        policy,
        horizon,
        reforecasts: 0,
        actions: 0,
        log_equity: vec![0.0],
        decision_instants: Vec::with_capacity(oracle_periods.len()),
        decision_span_years: f64::NAN,
        turnover: 0.0,
        execution_cost: 0.0,
        max_gross: 0.0,
        max_abs_net: 0.0,
        max_name: 0.0,
        max_participation: 0.0,
        covariance_observations: 0,
        covariance_shrinkage: config.covariance_shrinkage,
        cost_month_substitutions: 0,
        cost_cross_section_substitutions: 0,
        covariance_window: config.covariance_window,
        mean_factor_variance: 0.0,
    };
    for (p, period) in oracle_periods.iter().enumerate() {
        let t = period.instant;
        ensure!(
            t < panel.instants(),
            "oracle period {p} is outside the panel"
        );
        ensure!(
            p == 0 || oracle_periods[p - 1].instant < t,
            "oracle periods must be in ascending panel order"
        );
        // If an eligibility filter ever skips an interior row, it still belongs to the
        // strictly trailing covariance history even though it is not scored.
        for history in observed_until..t {
            let historical = &panel.slices()[history];
            let returns: Vec<f64> = historical
                .realized_r
                .iter()
                .map(|value| f64::from(*value).exp_m1())
                .collect();
            trailing.observe(
                &historical.symbols,
                &returns,
                panel.elapsed_steps_row(history),
            )?;
        }
        let slice = &panel.slices()[t];
        run.reforecasts += 1;
        run.decision_instants.push(t);
        ensure!(
            moments[p].len() == slice.symbols.len() && period.legs.len() == slice.symbols.len(),
            "eligible forecast row {p} does not align with panel row {t}"
        );
        ensure!(
            period
                .legs
                .iter()
                .zip(&slice.symbols)
                .all(|(leg, id)| leg.id == *id && leg.steps == horizon),
            "oracle row {p} is not a full aligned H={horizon} realization"
        );

        // Participation is a per-row availability mask, not a last-value cache. An absent
        // symbol and a symbol's first print after an absence both get a zero trade cap. The
        // latter print realizes the cumulative move from the symbol's prior close, so its
        // pre-existing holding must earn that move before it can trade again.
        adv.fill(0.0);
        for (k, &id) in slice.symbols.iter().enumerate() {
            if panel.elapsed_steps(t, k) > 1 {
                continue;
            }
            let measured = f64::from(panel.adv_usd(t, k));
            if measured.is_finite() && measured > 0.0 {
                adv[id as usize] = measured;
            }
        }
        let mut means = vec![0.0; names];
        let mut second = vec![0.0; names];
        match policy {
            RecedingPolicy::Model => {
                for (k, &id) in slice.symbols.iter().enumerate() {
                    means[id as usize] = moments[p][k].mean_simple;
                    second[id as usize] = moments[p][k].second_simple.max(0.0);
                }
            }
            RecedingPolicy::Marginal => {
                for &id in &slice.symbols {
                    means[id as usize] = marginal.mean_simple;
                    second[id as usize] = marginal.second_simple.max(0.0);
                }
            }
            RecedingPolicy::Oracle => {
                for leg in &period.legs {
                    let realized = leg.realized_log.exp_m1();
                    means[leg.id as usize] = realized;
                    second[leg.id as usize] = realized * realized;
                }
            }
            RecedingPolicy::EqualWeight | RecedingPolicy::BuyHold => {}
        }
        ensure!(
            means.iter().chain(&second).all(|value| value.is_finite()),
            "{} has a non-finite H={horizon} moment at row {t}",
            policy.name()
        );

        let risk = match policy {
            RecedingPolicy::Oracle => FactorCovariance {
                idiosyncratic: vec![0.0; names],
                loadings: vec![0.0; names],
                factor_variance: 0.0,
                observations: trailing.estimate(&zero_second)?.observations,
                shrinkage: 0.0,
                trailing_window: 0,
            },
            RecedingPolicy::Model | RecedingPolicy::Marginal => {
                // Estimate a one-bar covariance shape, scale BOTH its residual and factor
                // pieces to H, then reconcile its diagonal to the declared aggregate variance.
                for ((fallback, mean), raw_second) in
                    one_bar_fallback.iter_mut().zip(&means).zip(&second)
                {
                    *fallback = (raw_second - mean * mean).max(0.0) / horizon as f64;
                }
                let mut estimated = trailing.estimate(&one_bar_fallback)?;
                estimated.scale_horizon(horizon);
                estimated.match_forecast_moments(&means, &second)?;
                estimated
            }
            // These controls construct targets below and make no distributional risk claim.
            RecedingPolicy::EqualWeight | RecedingPolicy::BuyHold => {
                FactorCovariance::independent(vec![1.0; names])
            }
        };
        run.mean_factor_variance +=
            (risk.factor_variance - run.mean_factor_variance) / (p + 1) as f64;
        let equity = config.capital_usd * log_wealth.exp();
        let equal_target = || -> Result<Vec<f64>> {
            let mut desired = vec![0.0; names];
            let weight =
                1.0f64.min(config.constraints.gross_cap) / slice.symbols.len().max(1) as f64;
            for &id in &slice.symbols {
                desired[id as usize] = weight.min(config.constraints.per_name_cap);
            }
            Ok(solve_cost_aware_kelly(
                &symbol_ids,
                &desired,
                &FactorCovariance::independent(vec![1.0; names]),
                &held,
                &adv,
                slice.ts_ms,
                equity,
                cost,
                config.constraints,
            )?
            .target)
        };

        let target = match policy {
            RecedingPolicy::Model | RecedingPolicy::Marginal | RecedingPolicy::Oracle => {
                solve_cost_aware_kelly(
                    &symbol_ids,
                    &means,
                    &risk,
                    &held,
                    &adv,
                    slice.ts_ms,
                    equity,
                    cost,
                    config.constraints,
                )?
                .target
            }
            RecedingPolicy::EqualWeight => equal_target()?,
            RecedingPolicy::BuyHold if p == 0 => equal_target()?,
            RecedingPolicy::BuyHold => held.clone(),
        };
        let portfolio_violations = |weights: &[f64]| {
            let gross = weights.iter().map(|weight| weight.abs()).sum::<f64>();
            let net = weights.iter().sum::<f64>();
            [
                (gross - config.constraints.gross_cap).max(0.0),
                (config.constraints.net_min - net)
                    .max(0.0)
                    .max((net - config.constraints.net_max).max(0.0)),
                weights
                    .iter()
                    .map(|weight| (weight.abs() - config.constraints.per_name_cap).max(0.0))
                    .fold(0.0, f64::max),
            ]
        };
        let held_violations = portfolio_violations(&held);
        let target_violations = portfolio_violations(&target);
        ensure!(
            target.iter().all(|weight| weight.is_finite())
                && target_violations
                    .iter()
                    .zip(held_violations)
                    .all(|(target, held)| *target <= held + 1e-9)
                && target.iter().zip(&held).all(|(target, held)| {
                    (target.abs() - config.constraints.per_name_cap).max(0.0)
                        <= (held.abs() - config.constraints.per_name_cap).max(0.0) + 1e-9
                }),
            "{} worsened a hard portfolio-constraint violation at row {t}: \
             held={held_violations:?}, target={target_violations:?}",
            policy.name()
        );

        let mut cost_fraction = 0.0;
        let mut turnover = 0.0;
        for i in 0..names {
            let delta = (target[i] - held[i]).abs();
            let participation = if adv[i] > 0.0 {
                delta * equity / adv[i]
            } else if delta == 0.0 {
                0.0
            } else {
                f64::INFINITY
            };
            if adv[i] == 0.0 {
                ensure!(
                    delta <= 1e-12,
                    "{} traded absent symbol {} at row {t}",
                    policy.name(),
                    i
                );
            }
            if delta > 0.0 {
                run.actions += 1;
                let leg = cost.leg_cost(i as u32, slice.ts_ms, participation as f32);
                run.cost_month_substitutions += usize::from(leg.month_substituted);
                run.cost_cross_section_substitutions += usize::from(leg.substituted());
                cost_fraction += delta * f64::from(leg.bps) * 1.0e-4;
            }
            run.max_participation = run.max_participation.max(participation);
            turnover += delta;
        }
        ensure!(
            cost_fraction.is_finite()
                && run.max_participation <= config.constraints.max_adv_participation + 1e-9,
            "{} selected an unpriceable or over-participation action at row {t}",
            policy.name()
        );
        let realized: Vec<f64> = slice
            .realized_r
            .iter()
            .map(|value| f64::from(*value).exp_m1())
            .collect();
        let payoff = slice
            .symbols
            .iter()
            .zip(&realized)
            .map(|(&id, ret)| target[id as usize] * ret)
            .sum::<f64>();
        let multiplier = 1.0 + payoff - cost_fraction;
        ensure!(
            multiplier > 0.0 && multiplier.is_finite(),
            "{} H={horizon} ruined at panel row {t}",
            policy.name()
        );
        held.clone_from_slice(&target);
        for weight in &mut held {
            *weight /= multiplier;
        }
        for (&id, ret) in slice.symbols.iter().zip(&realized) {
            held[id as usize] = target[id as usize] * (1.0 + ret) / multiplier;
        }
        log_wealth += multiplier.ln();
        run.log_equity.push(log_wealth);
        run.turnover += turnover;
        run.execution_cost += cost_fraction;
        run.max_gross = run.max_gross.max(held.iter().map(|w| w.abs()).sum::<f64>());
        run.max_abs_net = run.max_abs_net.max(held.iter().sum::<f64>().abs());
        run.max_name = run
            .max_name
            .max(held.iter().map(|w| w.abs()).fold(0.0, f64::max));
        trailing.observe(&slice.symbols, &realized, panel.elapsed_steps_row(t))?;
        observed_until = t + 1;
        run.covariance_observations = trailing.estimate(&zero_second)?.observations;
    }
    run.decision_span_years = run.span_years(panel);
    ensure!(
        run.decision_span_years.is_finite() && run.decision_span_years > 0.0,
        "receding evaluation requires at least two decisions at distinct timestamps; got {}",
        run.decision_instants.len()
    );
    Ok(run)
}

pub const RECEDING_KELLY_BASE: &str = "pretrain_receding_kelly";
fn report_series(label: impl Into<String>, values: Vec<f64>) -> ReportSeries {
    ReportSeries {
        label: label.into(),
        values: values.into_iter().map(|value| value as f32).collect(),
    }
}

trait ReportSeriesNew {
    fn new(label: &str, values: Vec<f64>) -> Self;
}

impl ReportSeriesNew for ReportSeries {
    fn new(label: &str, values: Vec<f64>) -> Self {
        report_series(label, values)
    }
}

pub const RECEDING_COVARIANCE_BASE: &str = "pretrain_receding_covariance";

pub fn write_receding_reports(
    dir: &Path,
    label: &str,
    panel: &Panel,
    runs: &[RecedingRun],
    selected_horizon: usize,
) -> Result<()> {
    ensure!(
        FORECAST_HORIZONS.contains(&selected_horizon),
        "selected forecast horizon {selected_horizon} is not in the production grid {FORECAST_HORIZONS:?}"
    );
    let selected_model = runs
        .iter()
        .find(|run| run.policy == RecedingPolicy::Model && run.horizon == selected_horizon)
        .with_context(|| {
            format!("missing selected production model run at H={selected_horizon}")
        })?;
    let highlighted = |value: f64| {
        FORECAST_HORIZONS
            .iter()
            .map(|&horizon| {
                if horizon == selected_horizon {
                    value
                } else {
                    f64::NAN
                }
            })
            .collect()
    };
    let mut economic = Vec::new();
    for &policy in &RECEDING_POLICIES {
        let policy_runs: Vec<&RecedingRun> = FORECAST_HORIZONS
            .iter()
            .filter_map(|h| runs.iter().find(|r| r.policy == policy && r.horizon == *h))
            .collect();
        ensure!(
            policy_runs.len() == FORECAST_HORIZONS.len(),
            "{} report rows cover {} of {} production horizons",
            policy.name(),
            policy_runs.len(),
            FORECAST_HORIZONS.len()
        );
        economic.push(report_series(
            format!("{} net log growth/year", policy.name()),
            policy_runs
                .iter()
                .map(|r| r.annual_log_growth(panel))
                .collect(),
        ));
        economic.push(report_series(
            format!("{} turnover", policy.name()),
            policy_runs.iter().map(|r| r.turnover).collect(),
        ));
        economic.push(report_series(
            format!("{} execution cost", policy.name()),
            policy_runs.iter().map(|r| r.execution_cost).collect(),
        ));
        if policy == RecedingPolicy::Model {
            economic.push(report_series(
                "model net log growth/year, block-bootstrap ci low",
                policy_runs
                    .iter()
                    .map(|r| r.annual_growth_dispersion(panel).ci_low)
                    .collect(),
            ));
            economic.push(report_series(
                "model net log growth/year, block-bootstrap ci high",
                policy_runs
                    .iter()
                    .map(|r| r.annual_growth_dispersion(panel).ci_high)
                    .collect(),
            ));
        }
        economic.push(report_series(
            format!("{} reforecasts", policy.name()),
            policy_runs.iter().map(|r| r.reforecasts as f64).collect(),
        ));
    }
    economic.push(report_series(
        "forecast horizon",
        FORECAST_HORIZONS.iter().map(|h| *h as f64).collect(),
    ));
    let paired: Vec<Dispersion> = FORECAST_HORIZONS
        .iter()
        .map(|horizon| {
            let model = runs
                .iter()
                .find(|r| r.policy == RecedingPolicy::Model && r.horizon == *horizon)
                .expect("each horizon has a model run");
            let marginal = runs
                .iter()
                .find(|r| r.policy == RecedingPolicy::Marginal && r.horizon == *horizon)
                .expect("each horizon has a marginal run");
            model.annual_difference_dispersion(marginal, panel)
        })
        .collect();
    economic.push(report_series(
        "model minus marginal net log growth/year, paired block-bootstrap mean",
        paired.iter().map(|d| d.mean).collect(),
    ));
    economic.push(report_series(
        "model minus marginal net log growth/year, paired block-bootstrap ci low",
        paired.iter().map(|d| d.ci_low).collect(),
    ));
    economic.push(report_series(
        "model minus marginal net log growth/year, paired block-bootstrap ci high",
        paired.iter().map(|d| d.ci_high).collect(),
    ));
    economic.push(report_series(
        format!("SELECTED production H={selected_horizon} model net log growth/year"),
        highlighted(selected_model.annual_log_growth(panel)),
    ));
    let selected_index = FORECAST_HORIZONS
        .iter()
        .position(|&horizon| horizon == selected_horizon)
        .expect("selected horizon was validated against the grid");
    economic.push(report_series(
        format!(
            "SELECTED production H={selected_horizon} model minus marginal net log growth/year"
        ),
        highlighted(paired[selected_index].mean),
    ));
    write_chart(
        dir,
        RECEDING_KELLY_BASE,
        format!(
            "Every-Bar Receding-Horizon Kelly - {label} - SELECTED production H={selected_horizon}"
        ),
        "forecast horizon index (rebalance interval is always one bar)",
        "economic result",
        ScaleKind::Linear,
        economic,
    )?;

    let model: Vec<&RecedingRun> = FORECAST_HORIZONS
        .iter()
        .filter_map(|h| {
            runs.iter()
                .find(|r| r.policy == RecedingPolicy::Model && r.horizon == *h)
        })
        .collect();
    write_chart(
        dir,
        RECEDING_COVARIANCE_BASE,
        format!(
            "Causal H-Horizon One-Factor Model Risk - {label} - SELECTED production H={selected_horizon}"
        ),
        "forecast horizon index",
        "calibration / constraint audit",
        ScaleKind::Linear,
        vec![
            ReportSeries::new(
                "strictly trailing observations",
                model
                    .iter()
                    .map(|r| r.covariance_observations as f64)
                    .collect(),
            ),
            ReportSeries::new(
                "diagonal shrinkage",
                model.iter().map(|r| r.covariance_shrinkage).collect(),
            ),
            ReportSeries::new(
                "trailing window",
                model.iter().map(|r| r.covariance_window as f64).collect(),
            ),
            ReportSeries::new(
                "mean H-horizon factor variance",
                model.iter().map(|r| r.mean_factor_variance).collect(),
            ),
            ReportSeries::new(
                "PanelCost month-level substitutions",
                model
                    .iter()
                    .map(|r| r.cost_month_substitutions as f64)
                    .collect(),
            ),
            ReportSeries::new(
                "PanelCost cross-sectional substitutions",
                model
                    .iter()
                    .map(|r| r.cost_cross_section_substitutions as f64)
                    .collect(),
            ),
            ReportSeries::new("max gross", model.iter().map(|r| r.max_gross).collect()),
            ReportSeries::new("max abs net", model.iter().map(|r| r.max_abs_net).collect()),
            ReportSeries::new("max per-name", model.iter().map(|r| r.max_name).collect()),
            ReportSeries::new(
                "max ADV participation",
                model.iter().map(|r| r.max_participation).collect(),
            ),
            ReportSeries::new(
                &format!(
                    "SELECTED production H={selected_horizon} mean H-horizon factor variance"
                ),
                highlighted(selected_model.mean_factor_variance),
            ),
            ReportSeries::new(
                "forecast horizon",
                FORECAST_HORIZONS.iter().map(|h| *h as f64).collect(),
            ),
        ],
    )
}
#[derive(Clone, Debug)]
pub struct RecedingArgs {
    pub bars_dir: PathBuf,
    pub checkpoint: PathBuf,
    pub gens_dir: PathBuf,
    pub res_secs: u32,
    pub device: Device,
    pub split_bounds: (i64, i64),
    pub split: Split,
    /// Required in addition to `split == Test`; prevents an accidental locked-test read.
    pub allow_test: bool,
    pub max_symbols: usize,
    pub max_instants: usize,
    pub capital_usd: f64,
    pub forecast_horizon: usize,
    pub samples: usize,
    pub seed: i64,
    pub cost_threads: usize,
    pub config: RecedingConfig,
    pub label: String,
}

impl RecedingArgs {
    pub fn defaults(bars_dir: PathBuf, checkpoint: PathBuf, gens_dir: PathBuf) -> Self {
        let config = RecedingConfig::default();
        Self {
            bars_dir,
            checkpoint,
            gens_dir,
            res_secs: 300,
            device: Device::cuda_if_available(),
            split_bounds: crate::data::ingest::PINNED_SPLIT_BOUNDS,
            split: Split::Val,
            allow_test: false,
            max_symbols: 48,
            max_instants: 7_800,
            capital_usd: config.capital_usd,
            forecast_horizon: DEFAULT_FORECAST_HORIZON,
            samples: DEFAULT_SAMPLES,
            seed: 0x5EED,
            cost_threads: 4,
            config,
            label: "receding-kelly".to_owned(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RecedingBench {
    pub runs: Vec<RecedingRun>,
    pub split: Split,
    pub instants: usize,
    pub symbols: usize,
    pub samples: usize,
    pub selected_horizon: usize,
    pub checkpoint: String,
    pub lineage_sha256: String,
}

impl RecedingBench {
    pub fn table(&self) -> String {
        let mut out = format!(
            "every-bar receding Kelly: split={}, {} symbols, {} instants, {} paths, selected production H={} (marked *)\n\
             sel policy                    H  reforecasts  actions  log-growth/yr  turnover  cost\n",
            self.split.as_str(),
            self.symbols,
            self.instants,
            self.samples,
            self.selected_horizon,
        );
        for run in &self.runs {
            let growth = run.recorded_annual_log_growth();
            let marker =
                if run.policy == RecedingPolicy::Model && run.horizon == self.selected_horizon {
                    "*"
                } else {
                    ""
                };
            out.push_str(&format!(
                "{:<3} {:<24} {:>3} {:>12} {:>8} {:>14.6} {:>9.4} {:>8.5}\n",
                marker,
                run.policy.name(),
                run.horizon,
                run.reforecasts,
                run.actions,
                growth,
                run.turnover,
                run.execution_cost,
            ));
        }
        out
    }
}

pub fn validate_receding_split(split: Split, allow_test: bool) -> Result<()> {
    ensure!(
        matches!(split, Split::Val | Split::Test),
        "production evaluation is held-out only; choose validation or test"
    );
    ensure!(
        split != Split::Test || allow_test,
        "the test split is locked; pass --allow-test explicitly for the one final score"
    );
    Ok(())
}
/// Production evaluator: one causal panel scan, one common max-H ancestral rollout, then an
/// every-bar cost-aware solve for every horizon prefix and baseline.
pub fn run_receding_evaluation(args: &RecedingArgs) -> Result<RecedingBench> {
    crate::torch::cuda::cfg::configure_cuda();
    validate_receding_split(args.split, args.allow_test)?;

    ensure!(
        args.samples >= 2,
        "an ancestral forecast law needs at least two paths"
    );
    ensure!(
        args.cost_threads > 0,
        "production evaluation requires PanelCost calibration; --cost-threads must be positive"
    );
    ensure!(
        FORECAST_HORIZONS.contains(&args.forecast_horizon),
        "forecast horizon {} is not in the production grid {:?}",
        args.forecast_horizon,
        FORECAST_HORIZONS
    );
    let (b0, b1) = args.split_bounds;
    let span = match args.split {
        Split::Val => (b0, b1),
        Split::Test => (b1, i64::MAX),
        Split::Train => unreachable!("held-out split checked above"),
    };
    let panel_config = PanelConfig::new(span, args.max_symbols, args.max_instants);
    let corpus = BarCorpus::load_with_bounds(
        &args.bars_dir,
        args.res_secs,
        panel_config.min_history + ADV_TRAILING_BARS,
        args.split_bounds,
    )?;
    ensure!(
        corpus.split_bounds() == args.split_bounds,
        "the corpus did not take the pinned global split bounds"
    );
    let panel = Panel::build(&corpus, &panel_config)?;
    let metadata = world_model_metadata_path(&args.checkpoint);
    let model = BarWorldModel::load(&args.checkpoint, &metadata, args.device)
        .with_context(|| format!("loading {}", args.checkpoint.display()))?;
    ensure!(
        model.all_parameters_frozen(),
        "evaluation checkpoint is still trainable"
    );
    let supports = model
        .supports_for(args.res_secs)
        .with_context(|| format!("checkpoint has no {}s supports", args.res_secs))?;
    // Refuse legacy supports before doing the expensive rollout.
    supports
        .simple_return_bin_moments()
        .context("checkpoint supports lack fitted simple-return moments; refit v6 supports")?;
    let beliefs = scan_panel(&model, &corpus, &panel, args.res_secs)?;
    let max_periods = receding_schedule(
        &corpus,
        &panel,
        &beliefs,
        *FORECAST_HORIZONS.last().expect("horizon grid is non-empty"),
    )?;
    tch::manual_seed(args.seed);
    let laws = horizon_laws(
        &model,
        &corpus,
        &panel,
        &beliefs,
        &max_periods,
        args.res_secs,
        args.samples,
    )?;
    let forecasts = RecedingForecasts::from_common_rollout(&panel, &beliefs, &max_periods, &laws)?;

    // Validation is calibrated on training only; the locked test is calibrated on train+val.
    // No bar at or after this evaluated split's first instant may affect any fallback or month.
    let calibration = Arc::new(
        CostCalibration::from_corpus(&corpus, args.cost_threads, span.0)
            .context("measuring causal per-symbol PanelCost calibration")?,
    );
    let cost = PanelCost::new(&panel, BarCostModel::new(calibration), CostParts::All);
    let mut runs = Vec::with_capacity(FORECAST_HORIZONS.len() * RECEDING_POLICIES.len());
    for &horizon in &FORECAST_HORIZONS {
        let moments = forecasts
            .horizon(horizon)
            .expect("the production horizon is present");
        let marginal = marginal_horizon_moment(supports, horizon)?;
        let mut oracle = receding_schedule(&corpus, &panel, &beliefs, horizon)?;
        let common_last = max_periods
            .last()
            .expect("the common schedule has eligible decisions")
            .instant;
        oracle.retain(|period| period.instant <= common_last);
        ensure!(
            oracle.len() == max_periods.len(),
            "H={horizon} oracle has {} common decisions for {} model rows",
            oracle.len(),
            max_periods.len()
        );
        for &policy in &RECEDING_POLICIES {
            let mut config = args.config;
            config.capital_usd = args.capital_usd;
            runs.push(run_receding_book(
                &panel, moments, &oracle, marginal, policy, horizon, &cost, config,
            )?);
        }
    }
    write_receding_reports(
        &args.gens_dir,
        &args.label,
        &panel,
        &runs,
        args.forecast_horizon,
    )?;
    Ok(RecedingBench {
        runs,
        split: args.split,
        instants: panel.instants(),
        symbols: panel.symbols().len(),
        samples: args.samples,
        selected_horizon: args.forecast_horizon,
        checkpoint: args.checkpoint.display().to_string(),
        lineage_sha256: model.lineage_sha256().to_owned(),
    })
}

// ---------------------------------------------------------------------------
// The book
// ---------------------------------------------------------------------------

/// Scale `raw` onto the L1 ball of radius `budget`, in place. Returns whether it bound.
///
/// Proportional, never truncating, for the same reason [`super::portfolio`]'s own projection
/// is: a leverage limit scales a book rather than dropping its smallest names.
fn project_gross(raw: &mut [f64], budget: f64) -> bool {
    let gross: f64 = raw.iter().map(|w| w.abs()).sum();
    if !(gross > budget) || !gross.is_finite() {
        return false;
    }
    let scale = budget / gross;
    for w in raw.iter_mut() {
        *w *= scale;
    }
    true
}

/// The raw preference vector of one policy over one period's legs.
///
/// Mirrors [`super::portfolio::Policy`]'s private `raw_weights` with the payoff read at the
/// HOLDING horizon rather than at the next bar: the oracle's perfect foresight is of the
/// aggregate a held position actually earns, which is the only ceiling that means anything
/// once a position is held for more than one bar.
fn raw_weights(
    policy: Policy,
    period: &Period,
    inputs: &HorizonInputs,
    index: usize,
    budget: f64,
    out: &mut Vec<f64>,
) {
    out.clear();
    let n = period.legs.len();
    match policy {
        Policy::Model => out.extend(inputs.kelly[index].iter().copied()),
        Policy::MarketNeutral => {
            if n == 0 {
                return;
            }
            let mean = inputs.kelly[index].iter().sum::<f64>() / n as f64;
            out.extend(inputs.kelly[index].iter().map(|f| f - mean));
        }
        Policy::Marginal => out.extend(std::iter::repeat_n(inputs.marginal_kelly, n)),
        Policy::EqualWeight => out.extend(std::iter::repeat_n(1.0, n)),
        Policy::Oracle => {
            out.extend(std::iter::repeat_n(0.0, n));
            let best = period
                .legs
                .iter()
                .enumerate()
                .map(|(l, leg)| (l, leg.realized_log.exp_m1()))
                .filter(|(_, payoff)| payoff.is_finite() && *payoff != 0.0)
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()));
            if let Some((l, payoff)) = best {
                out[l] = budget * payoff.signum();
            }
        }
    }
}

/// One policy's realized path through the holding periods.
#[derive(Clone, Debug)]
pub struct HorizonBook {
    pub policy: Policy,
    pub construction: Construction,
    pub k: usize,
    pub gross_cap: f64,
    /// Natural log of wealth, one entry per period boundary, `[0] == 0.0`.
    pub log_equity: Vec<f64>,
    /// `sum_i w_i (exp(L_i) - 1)` over the period, BEFORE cost.
    pub payoff: Vec<f64>,
    /// Simple return of the period, net of cost.
    pub returns: Vec<f64>,
    pub gross: Vec<f64>,
    pub net: Vec<f64>,
    /// Gross exposure at the END of the period, after the hold drifted it.
    pub drifted_gross: Vec<f64>,
    pub turnover: Vec<f64>,
    pub cost: Vec<f64>,
    pub factor: Vec<f64>,
    pub pred_var: Vec<f64>,
    pub bound: Vec<bool>,
    pub ruined_at: Option<usize>,
}

/// Compound one policy through the holding periods at one gross cap.
///
/// Positions are established at the rebalance and then HELD: nothing trades until the next
/// rebalance, weights drift with prices, and the drifted vector is what the next target is
/// charged against. That is the whole difference from [`super::portfolio::backtest`], and at
/// `k = 1` it collapses onto it up to the single bar of drift inside each instant.
#[allow(clippy::too_many_arguments)]
pub fn run_book(
    panel: &Panel,
    periods: &[Period],
    inputs: &HorizonInputs,
    policy: Policy,
    k: usize,
    gross_cap: f64,
    cost: &dyn CostModel,
    capital_usd: f64,
) -> Result<HorizonBook> {
    ensure!(
        gross_cap > 0.0 && gross_cap.is_finite(),
        "the gross cap must be positive and finite, got {gross_cap}"
    );
    ensure!(
        inputs.kelly.len() == periods.len(),
        "the sizing inputs cover {} periods against {} rebalances",
        inputs.kelly.len(),
        periods.len()
    );
    let budget = policy.gross_budget(gross_cap);
    let names = panel.symbols().len();
    let loading = panel.first_factor();

    let mut log_wealth = 0.0f64;
    let mut held = vec![0.0f64; names];
    let mut target = vec![0.0f64; names];
    let mut realized = vec![0.0f64; names];
    let mut last_adv = vec![0.0f64; names];
    let mut raw: Vec<f64> = Vec::with_capacity(names);

    let mut book = HorizonBook {
        policy,
        construction: inputs.construction,
        k,
        gross_cap,
        log_equity: Vec::with_capacity(periods.len() + 1),
        payoff: Vec::with_capacity(periods.len()),
        returns: Vec::with_capacity(periods.len()),
        gross: Vec::with_capacity(periods.len()),
        net: Vec::with_capacity(periods.len()),
        drifted_gross: Vec::with_capacity(periods.len()),
        turnover: Vec::with_capacity(periods.len()),
        cost: Vec::with_capacity(periods.len()),
        factor: Vec::with_capacity(periods.len()),
        pred_var: Vec::with_capacity(periods.len()),
        bound: Vec::with_capacity(periods.len()),
        ruined_at: None,
    };
    book.log_equity.push(0.0);

    for (index, period) in periods.iter().enumerate() {
        for leg in &period.legs {
            if leg.adv_usd.is_finite() && leg.adv_usd > 0.0 {
                last_adv[leg.id as usize] = leg.adv_usd;
            }
        }
        if book.ruined_at.is_some() {
            // A dead book holds nothing, trades nothing and earns nothing, forever. It is
            // still recorded at every period so the curve keeps the rebalance clock.
            book.log_equity.push(f64::NEG_INFINITY);
            book.payoff.push(0.0);
            book.returns.push(0.0);
            book.gross.push(0.0);
            book.net.push(0.0);
            book.drifted_gross.push(0.0);
            book.turnover.push(0.0);
            book.cost.push(0.0);
            book.factor.push(0.0);
            book.pred_var.push(0.0);
            book.bound.push(false);
            continue;
        }

        raw_weights(policy, period, inputs, index, budget, &mut raw);
        ensure!(
            raw.len() == period.legs.len(),
            "policy {} produced {} weights for {} legs at period {index}",
            policy.name(),
            raw.len(),
            period.legs.len()
        );
        for w in raw.iter_mut() {
            if !w.is_finite() {
                *w = 0.0;
            }
        }
        let bound = project_gross(&mut raw, budget);

        // Absence is a zero target and an unwind, exactly as in `portfolio`: a name that is
        // not in this rebalance's slice cannot be held through the coming window, whatever it
        // was worth at the last one, and that unwind is charged like any other trade.
        target[..].fill(0.0);
        realized[..].fill(0.0);
        for (l, leg) in period.legs.iter().enumerate() {
            target[leg.id as usize] = raw[l];
            realized[leg.id as usize] = leg.realized_log.exp_m1();
        }

        let mut payoff = 0.0f64;
        let mut gross = 0.0f64;
        let mut net = 0.0f64;
        let mut factor = 0.0f64;
        let mut pred_var = 0.0f64;
        for (l, leg) in period.legs.iter().enumerate() {
            let w = target[leg.id as usize];
            gross += w.abs();
            net += w;
            factor += w * f64::from(loading[leg.id as usize]);
            let var = inputs.pred_var[index][l];
            if var.is_finite() && var > 0.0 {
                pred_var += w * w * var;
            }
            payoff += w * realized[leg.id as usize];
        }
        ensure!(
            gross <= budget * (1.0 + 1e-9) + 1e-12,
            "policy {} used gross {gross} against a budget of {budget} at period {index}",
            policy.name()
        );

        let mut turnover = 0.0f64;
        let mut cost_frac = 0.0f64;
        let wealth_usd = (log_wealth.exp() * capital_usd).min(f64::from(f32::MAX));
        for id in 0..names {
            let delta = (target[id] - held[id]).abs();
            if delta == 0.0 {
                continue;
            }
            turnover += delta;
            let adv = last_adv[id];
            // No observed liquidity means the size is unpriceable, not free: charge it at a
            // full-ADV clip, the worst bucket any sane cost curve carries.
            let frac = if adv > 0.0 {
                (((delta * wealth_usd) / adv) as f32).min(f32::MAX)
            } else {
                1.0
            };
            let bps = f64::from(cost.cost_bps(id as u32, period.ts_ms, frac));
            ensure!(
                bps.is_finite() && bps >= 0.0,
                "the cost model returned {bps} bps for symbol {id} at {}",
                period.ts_ms
            );
            cost_frac += delta * bps * 1e-4;
        }

        let multiplier = 1.0 + payoff - cost_frac;
        let (realized_return, drifted) = if multiplier > 0.0 {
            log_wealth += multiplier.ln();
            // Buy-and-hold drift: the position is worth `w (1 + R)` of the pre-cost wealth and
            // the book is worth `multiplier` of it, so this is the weight the next rebalance is
            // charged against. Exact, not an approximation.
            let mut drifted_gross = 0.0f64;
            for id in 0..names {
                held[id] = target[id] * (1.0 + realized[id]) / multiplier;
                drifted_gross += held[id].abs();
            }
            (multiplier - 1.0, drifted_gross)
        } else {
            log_wealth = f64::NEG_INFINITY;
            book.ruined_at = Some(index);
            held.fill(0.0);
            (-1.0, 0.0)
        };

        book.log_equity.push(log_wealth);
        book.payoff.push(payoff);
        book.returns.push(realized_return);
        book.gross.push(gross);
        book.net.push(net);
        book.drifted_gross.push(drifted);
        book.turnover.push(turnover);
        book.cost.push(cost_frac);
        book.factor.push(factor);
        book.pred_var.push(pred_var);
        book.bound.push(bound);
    }
    Ok(book)
}

/// What a trader would quote for one `(k, construction, policy)`, annualized from the panel's
/// own measured calendar.
#[derive(Clone, Copy, Debug)]
pub struct HorizonMetrics {
    pub periods: usize,
    pub span_years: f64,
    pub periods_per_year: f64,
    pub final_log_wealth: f64,
    /// Net log growth per year at the cost the sweep was run with.
    pub log_growth_per_year: f64,
    /// Log growth per year at EXACTLY zero cost: what the law was worth before paying to act
    /// on it. Measured by re-running the book at zero cost, not by subtraction.
    pub gross_log_growth_per_year: f64,
    /// Flat one-way cost, in bps, at which net log growth crosses zero.
    pub break_even_cost_bps: f64,
    pub cagr: f64,
    pub sharpe: f64,
    pub vol: f64,
    /// Measured on the PERIOD clock; intra-period drawdown is invisible to it.
    pub max_drawdown: f64,
    pub mean_gross: f64,
    /// Largest gross the book ever carried, INCLUDING the drift inside a holding period.
    pub max_gross: f64,
    pub mean_net: f64,
    /// Absolute weight traded per period, summed and divided by the panel's trading days.
    pub turnover_per_day: f64,
    /// Turnover per period as a multiple of the gross the book actually held.
    pub rotation_per_period: f64,
    pub bound_fraction: f64,
    /// Mean absolute projection of the book onto the panel's leading eigenvector, as a
    /// fraction of its gross. `0` is factor-neutral; `1` is a book that is one bet on the
    /// market wearing the costume of many.
    pub mean_first_factor_exposure: f64,
    pub first_factor_share: f64,
    /// Realized book volatility divided by the volatility the per-name laws imply under
    /// INDEPENDENCE. Per-name Kelly sizes as if this were `1.0`; whatever it is, is the factor
    /// by which the book is over-levered. This is the measured stand-in for a cross-sectional
    /// correlation that cannot be estimated past 12 bars on this panel.
    pub leverage_error: f64,
    pub ruined_at_period: f64,
}

impl HorizonMetrics {
    fn of(book: &HorizonBook, gross: &HorizonBook, panel: &Panel) -> Self {
        let n = book.returns.len();
        let years = panel.span_years();
        let per_year = if years > 0.0 {
            n as f64 / years
        } else {
            f64::NAN
        };
        let final_log_wealth = *book.log_equity.last().expect("the curve starts at 0.0");
        let log_growth_per_year = if years > 0.0 {
            final_log_wealth / years
        } else {
            f64::NAN
        };
        let mean = book.returns.iter().sum::<f64>() / n as f64;
        let variance = if n > 1 {
            book.returns
                .iter()
                .map(|r| (r - mean) * (r - mean))
                .sum::<f64>()
                / (n - 1) as f64
        } else {
            f64::NAN
        };
        let sd = variance.sqrt();
        let mut peak = f64::NEG_INFINITY;
        let mut max_drawdown = 0.0f64;
        for &log_w in &book.log_equity {
            peak = peak.max(log_w);
            if peak > f64::NEG_INFINITY {
                max_drawdown = max_drawdown.max(1.0 - (log_w - peak).exp());
            }
        }
        let mut exposure_sum = 0.0f64;
        let mut exposure_count = 0usize;
        for (f, g) in book.factor.iter().zip(&book.gross) {
            if *g > 0.0 {
                exposure_sum += (f / g).abs();
                exposure_count += 1;
            }
        }
        let payoff_mean = book.payoff.iter().sum::<f64>() / n as f64;
        let payoff_var = if n > 1 {
            book.payoff
                .iter()
                .map(|p| (p - payoff_mean) * (p - payoff_mean))
                .sum::<f64>()
                / (n - 1) as f64
        } else {
            f64::NAN
        };
        let independence_var = book.pred_var.iter().sum::<f64>() / n as f64;
        let days = panel.trading_days().max(1) as f64;
        let mean_of = |v: &[f64]| v.iter().sum::<f64>() / n as f64;
        let gross_final = *gross.log_equity.last().expect("the curve starts at 0.0");
        Self {
            periods: n,
            span_years: years,
            periods_per_year: per_year,
            final_log_wealth,
            log_growth_per_year,
            gross_log_growth_per_year: if years > 0.0 {
                gross_final / years
            } else {
                f64::NAN
            },
            break_even_cost_bps: f64::NAN,
            cagr: if final_log_wealth == f64::NEG_INFINITY {
                -1.0
            } else {
                log_growth_per_year.exp_m1()
            },
            sharpe: if sd > 0.0 {
                mean / sd * per_year.sqrt()
            } else {
                f64::NAN
            },
            vol: sd * per_year.sqrt(),
            max_drawdown,
            mean_gross: mean_of(&book.gross),
            max_gross: book
                .gross
                .iter()
                .chain(&book.drifted_gross)
                .copied()
                .fold(0.0, f64::max),
            mean_net: mean_of(&book.net),
            turnover_per_day: book.turnover.iter().sum::<f64>() / days,
            rotation_per_period: {
                let g = mean_of(&book.gross);
                if g > 0.0 {
                    mean_of(&book.turnover) / g
                } else {
                    f64::NAN
                }
            },
            bound_fraction: book.bound.iter().filter(|b| **b).count() as f64 / n as f64,
            mean_first_factor_exposure: if exposure_count > 0 {
                exposure_sum / exposure_count as f64
            } else {
                f64::NAN
            },
            first_factor_share: panel.first_factor_share(),
            leverage_error: if independence_var > 0.0 && payoff_var.is_finite() {
                (payoff_var / independence_var).sqrt()
            } else {
                f64::NAN
            },
            ruined_at_period: book.ruined_at.map_or(f64::NAN, |p| p as f64),
        }
    }
}

/// Run one policy at one `k` and measure it, including the zero-cost re-run and the break-even
/// solve.
///
/// The break-even bisection RE-RUNS the book at every trial cost. It has to: cost enters the
/// buy-and-hold drift through the period multiplier, so replaying a stored payoff and turnover
/// path at a different cost would be a linearization here. The book loop is `periods *
/// breadth` arithmetic, so 40 re-runs cost far less than the forecast that produced its
/// inputs.
#[allow(clippy::too_many_arguments)]
pub fn measure(
    panel: &Panel,
    periods: &[Period],
    inputs: &HorizonInputs,
    policy: Policy,
    k: usize,
    gross_cap: f64,
    cost: &dyn CostModel,
    capital_usd: f64,
) -> Result<HorizonMetrics> {
    let book = run_book(
        panel,
        periods,
        inputs,
        policy,
        k,
        gross_cap,
        cost,
        capital_usd,
    )?;
    let free = FlatCost::new(0.0);
    let gross = run_book(
        panel,
        periods,
        inputs,
        policy,
        k,
        gross_cap,
        &free,
        capital_usd,
    )?;
    let mut metrics = HorizonMetrics::of(&book, &gross, panel);

    let at = |bps: f64| -> Result<f64> {
        let flat = FlatCost::new(bps as f32);
        let run = run_book(
            panel,
            periods,
            inputs,
            policy,
            k,
            gross_cap,
            &flat,
            capital_usd,
        )?;
        Ok(*run.log_equity.last().expect("the curve starts at 0.0"))
    };
    metrics.break_even_cost_bps = if !(at(0.0)? > 0.0) {
        0.0
    } else if at(MAX_BREAK_EVEN_BPS)? > 0.0 {
        f64::INFINITY
    } else {
        let (mut lo, mut hi) = (0.0f64, MAX_BREAK_EVEN_BPS);
        for _ in 0..BREAK_EVEN_ITERATIONS {
            let mid = 0.5 * (lo + hi);
            if at(mid)? > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    Ok(metrics)
}

// ---------------------------------------------------------------------------
// The sweep
// ---------------------------------------------------------------------------

/// The mechanism behind one row: the numbers that say WHY break-even moved, not just that it
/// did. Everything in basis points of the held aggregate, averaged over every leg of every
/// period.
#[derive(Clone, Copy, Debug, Default)]
pub struct HorizonMechanism {
    /// Mean predicted conditional drift of the held aggregate, signed.
    pub rb_mu_bps: f64,
    /// The same drift from the plain sampled estimator, for the variance-reduction check.
    pub plain_mu_bps: f64,
    /// Mean Monte-Carlo standard error of the Rao-Blackwellized drift.
    pub rb_mu_se_bps: f64,
    /// Mean Monte-Carlo standard error of the plain drift. Its ratio to the line above is the
    /// measured variance reduction.
    pub plain_mu_se_bps: f64,
    /// Mean predicted volatility of the held aggregate.
    pub pred_sigma_bps: f64,
    /// Realized volatility of the held aggregate over the same legs.
    pub realized_sigma_bps: f64,
    /// Fraction of legs whose predicted drift has the sign of the realized aggregate. The only
    /// one of the three "win rates" that is a directional-skill statistic.
    pub sign_agreement: f64,
    /// Fraction of legs whose realized held aggregate was positive. A property of the market
    /// over the window, not of the model.
    pub realized_up_fraction: f64,
    pub legs: usize,
}

impl HorizonMechanism {
    fn of(periods: &[Period], inputs: &HorizonInputs) -> Self {
        let mut out = Self::default();
        let mut realized_sum = 0.0f64;
        let mut realized_sq = 0.0f64;
        for (p, period) in periods.iter().enumerate() {
            for (l, leg) in period.legs.iter().enumerate() {
                let law = inputs.laws[p][l];
                out.rb_mu_bps += law.mu_log;
                out.plain_mu_bps += law.plain_mu_log;
                out.rb_mu_se_bps += law.mu_se;
                out.plain_mu_se_bps += law.plain_mu_se;
                out.pred_sigma_bps += law.var_log.max(0.0).sqrt();
                realized_sum += leg.realized_log;
                realized_sq += leg.realized_log * leg.realized_log;
                if leg.realized_log != 0.0 && law.mu_log.signum() == leg.realized_log.signum() {
                    out.sign_agreement += 1.0;
                }
                if leg.realized_log > 0.0 {
                    out.realized_up_fraction += 1.0;
                }
                out.legs += 1;
            }
        }
        if out.legs == 0 {
            return out;
        }
        let n = out.legs as f64;
        let scale = 1.0e4 / n;
        out.rb_mu_bps *= scale;
        out.plain_mu_bps *= scale;
        out.rb_mu_se_bps *= scale;
        out.plain_mu_se_bps *= scale;
        out.pred_sigma_bps *= scale;
        let mean = realized_sum / n;
        out.realized_sigma_bps = (realized_sq / n - mean * mean).max(0.0).sqrt() * 1.0e4;
        out.sign_agreement /= n;
        out.realized_up_fraction /= n;
        out
    }
}

/// One `(k, construction, policy)` row of the frontier, with its replicate spread.
#[derive(Clone, Copy, Debug)]
pub struct HorizonRow {
    pub k: usize,
    pub construction: Construction,
    pub policy: Policy,
    pub periods: usize,
    /// Mean over replicates. Identical to the single value for the unsampled constructions.
    pub metrics: HorizonMetrics,
    /// Standard error ACROSS replicate sample sets of the four headline numbers. Exactly `0.0`
    /// where the construction is not sampled, because there the error is zero.
    pub break_even_se: f64,
    pub gross_growth_se: f64,
    pub net_growth_se: f64,
    pub sharpe_se: f64,
    pub replicates: usize,
    pub mechanism: HorizonMechanism,
}

impl HorizonRow {
    /// Whether the row has enough non-overlapping periods for its risk statistics to be
    /// information rather than noise.
    pub fn credible(&self) -> bool {
        self.periods >= MIN_CREDIBLE_PERIODS
    }

    /// Whether the row's `k` is past the horizon the belief-advance mechanism has ever been
    /// diagnosed at. Only the sampled construction advances a belief, so only it extrapolates.
    pub fn extrapolates_dynamics(&self) -> bool {
        self.construction.is_sampled() && self.k > DYNAMICS_DIAGNOSED_HORIZON
    }

    pub fn flags(&self) -> &'static str {
        match (self.credible(), self.extrapolates_dynamics()) {
            (true, false) => "",
            (true, true) => "DYN-EXTRAP",
            (false, false) => "FEW-PERIODS",
            (false, true) => "FEW-PERIODS DYN-EXTRAP",
        }
    }
}

/// Everything the sweep measured, plus the panel it measured on.
#[derive(Clone, Debug)]
pub struct HorizonFrontier {
    pub rows: Vec<HorizonRow>,
    pub gross_cap: f64,
    pub cost_bps: f64,
    pub cost_label: String,
    pub samples: usize,
    pub replicates: usize,
    pub instants: usize,
    pub symbols: usize,
    pub mean_breadth: f64,
    pub trading_days: usize,
    pub span_years: f64,
    pub first_ts_ms: i64,
    pub last_ts_ms: i64,
    pub checkpoint: String,
    pub lineage_sha256: String,
}

impl HorizonFrontier {
    pub fn row(&self, k: usize, construction: Construction, policy: Policy) -> Option<&HorizonRow> {
        self.rows
            .iter()
            .find(|r| r.k == k && r.construction == construction && r.policy == policy)
    }

    /// The largest break-even the MODEL achieves at any `k` under the HORIZON construction.
    /// The verdict, in one call.
    pub fn best_model_horizon(&self) -> Option<&HorizonRow> {
        self.rows
            .iter()
            .filter(|r| r.construction == Construction::Horizon && r.policy == Policy::Model)
            .max_by(|a, b| {
                a.metrics
                    .break_even_cost_bps
                    .total_cmp(&b.metrics.break_even_cost_bps)
            })
    }

    /// The best break-even any BASELINE achieves at the same `k` and construction as `row`. A
    /// model row that does not beat this is not a model result.
    pub fn best_baseline_at(&self, row: &HorizonRow) -> Option<&HorizonRow> {
        self.rows
            .iter()
            .filter(|r| {
                r.k == row.k
                    && r.construction == row.construction
                    && matches!(
                        r.policy,
                        Policy::EqualWeight | Policy::Marginal | Policy::Oracle
                    )
            })
            .max_by(|a, b| {
                a.metrics
                    .break_even_cost_bps
                    .total_cmp(&b.metrics.break_even_cost_bps)
            })
    }

    /// The one-line verdict this module exists to produce.
    ///
    /// Stated against [`MATCHED_MEASURED_BPS`], the matched impact-free cost over all the traded
    /// names, and against [`MATCHED_DEEPEST_DECILE_BPS`] as the floor: a break-even under the
    /// floor fails even the CHEAPEST matched cell that exists, under no impact model at all,
    /// which is a stronger statement than failing the headline figure and the one conclusion here
    /// that survives the edge being remeasured on a liquidity-restricted book.
    pub fn verdict(&self) -> String {
        let Some(best) = self.best_model_horizon() else {
            return "no horizon row was measured".to_owned();
        };
        let baseline = self.best_baseline_at(best);
        let beaten = baseline
            .is_some_and(|b| b.metrics.break_even_cost_bps >= best.metrics.break_even_cost_bps);
        let bps = best.metrics.break_even_cost_bps;
        format!(
            "model under the horizon construction: best break-even {bps:.4} +/- {:.4} bps at \
             k={} ({} periods{}). Against the MATCHED measured impact-free cost of \
             {MATCHED_MEASURED_BPS:.3} bps it is {}; against the \
             {MATCHED_DEEPEST_DECILE_BPS:.3} bps MATCHED deepest-decile FLOOR (equal-weighted \
             mean over 43 of the traded names, cost-restricted against edge-unrestricted) it is \
             {}; against the \
             {MATCHED_ALL_IN_BPS:.3} bps matched \
             all-in figure it is {}. Best baseline at that k is {} at {:.4} bps, which {} the \
             model.",
            best.break_even_se,
            best.k,
            best.periods,
            if best.flags().is_empty() {
                String::new()
            } else {
                format!(", {}", best.flags())
            },
            side(bps, MATCHED_MEASURED_BPS),
            side(bps, MATCHED_DEEPEST_DECILE_BPS),
            side(bps, MATCHED_ALL_IN_BPS),
            baseline.map_or("none", |b| b.policy.name()),
            baseline.map_or(f64::NAN, |b| b.metrics.break_even_cost_bps),
            if beaten { "BEATS" } else { "does not beat" },
        )
    }

    /// A printable table of the whole sweep.
    pub fn table(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "horizon sweep: {} symbols, {} instants, mean breadth {:.1}, {} trading days, \
             {:.3} years, gross {:.1}x, cost {}, {} samples x {} replicates\n\
             panel [{}, {}]  checkpoint {}  lineage {}\n",
            self.symbols,
            self.instants,
            self.mean_breadth,
            self.trading_days,
            self.span_years,
            self.gross_cap,
            self.cost_label,
            self.samples,
            self.replicates,
            self.first_ts_ms,
            self.last_ts_ms,
            self.checkpoint,
            &self.lineage_sha256[..self.lineage_sha256.len().min(12)],
        ));
        out.push_str(
            "construction        policy                  k  periods    break-even   +/-SE  \
             gross/yr    net/yr  Sharpe  turn/day  rot/per  bind   f1exp  lev-err   mu bps  \
             sig bps   sign  flags\n",
        );
        for row in &self.rows {
            out.push_str(&format!(
                "{:<18}  {:<20}  {:>4}  {:>7}  {:>11.4}  {:>6.4}  {:>8.3}  {:>8.3}  {:>6.2}  \
                 {:>8.2}  {:>7.3}  {:>4.2}  {:>6.3}  {:>7.3}  {:>7.3}  {:>7.2}  {:>5.3}  {}\n",
                row.construction.name(),
                row.policy.name(),
                row.k,
                row.periods,
                displayed_break_even(row.metrics.break_even_cost_bps),
                row.break_even_se,
                row.metrics.gross_log_growth_per_year,
                row.metrics.log_growth_per_year,
                row.metrics.sharpe,
                row.metrics.turnover_per_day,
                row.metrics.rotation_per_period,
                row.metrics.bound_fraction,
                row.metrics.mean_first_factor_exposure,
                row.metrics.leverage_error,
                row.mechanism.rb_mu_bps,
                row.mechanism.pred_sigma_bps,
                row.mechanism.sign_agreement,
                row.flags(),
            ));
        }
        out.push_str(&format!(
            "\nthe correlation term structure is measurable only to 12 bars on this panel, so \
             every k >= 39 row has NO rho input; `lev-err` is the realized substitute\n\
             VERDICT: {}\n",
            self.verdict()
        ));
        out
    }
}

/// Mean and standard error of a replicate set. A single value has zero error, which is the
/// truth for an unsampled construction rather than a missing measurement.
fn replicate_stats(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return (values[0], 0.0);
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    if finite.len() < 2 {
        return (mean, 0.0);
    }
    let var =
        finite.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (finite.len() - 1) as f64;
    (mean, (var / finite.len() as f64).sqrt())
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct HorizonArgs {
    /// Directory of `<SYM>.<res>.bars` files.
    pub bars_dir: PathBuf,
    /// The checkpoint to trade. Its metadata and supports sidecars resolve beside it.
    pub checkpoint: PathBuf,
    /// Generation directory the chart lands in.
    pub gens_dir: PathBuf,
    pub res_secs: u32,
    pub device: Device,
    /// The PINNED global split, so the panel is held out by construction.
    pub split_bounds: (i64, i64),
    pub max_symbols: usize,
    pub max_instants: usize,
    pub cost_bps: f32,
    pub capital_usd: f64,
    pub gross_cap: f64,
    pub samples: usize,
    pub replicates: usize,
    /// Seed of the first replicate. Replicate `r` uses `seed + r`.
    pub seed: i64,
    pub label: String,
}

impl HorizonArgs {
    /// The configuration the quoted numbers are measured at.
    pub fn defaults(bars_dir: PathBuf, checkpoint: PathBuf, gens_dir: PathBuf) -> Self {
        Self {
            bars_dir,
            checkpoint,
            gens_dir,
            res_secs: 300,
            device: Device::cuda_if_available(),
            split_bounds: crate::data::ingest::PINNED_SPLIT_BOUNDS,
            max_symbols: 48,
            max_instants: 7_800,
            cost_bps: DEFAULT_COST_BPS,
            capital_usd: 1.0e7,
            gross_cap: DEFAULT_GROSS_CAP,
            samples: DEFAULT_SAMPLES,
            replicates: DEFAULT_REPLICATES,
            seed: 0x5EED,
            label: "horizon".to_owned(),
        }
    }
}

/// Build the held-out panel, scan it once, then sweep every holding period under every
/// construction and write the chart.
pub fn run_horizon_sweep(args: &HorizonArgs) -> Result<HorizonFrontier> {
    // The world model asserts a bf16 autocast on CUDA and it is right to: a book measured
    // under a different numeric regime than the one the weights were selected under is
    // measuring a different model. The moment extractions opt out locally.
    crate::torch::cuda::cfg::configure_cuda();
    ensure!(
        args.samples >= 2,
        "a sampled aggregate law needs at least two paths, got {}",
        args.samples
    );
    ensure!(
        args.replicates >= 1,
        "the sweep needs at least one replicate"
    );
    let (val_start, val_end) = args.split_bounds;
    let config = PanelConfig::new((val_start, val_end), args.max_symbols, args.max_instants);
    let corpus = BarCorpus::load_with_bounds(
        &args.bars_dir,
        args.res_secs,
        config.min_history + ADV_TRAILING_BARS,
        (val_start, val_end),
    )?;
    ensure!(
        corpus.split_bounds() == (val_start, val_end),
        "the corpus did not take the pinned split bounds"
    );
    let panel = Panel::build(&corpus, &config)?;

    let metadata = world_model_metadata_path(&args.checkpoint);
    let model = BarWorldModel::load(&args.checkpoint, &metadata, args.device)
        .with_context(|| format!("loading {}", args.checkpoint.display()))?;
    ensure!(
        model.all_parameters_frozen(),
        "the checkpoint loaded for a horizon sweep is still trainable"
    );
    let supports = model
        .supports_for(args.res_secs)
        .with_context(|| format!("the checkpoint carries no supports at {}s", args.res_secs))?;

    let started = std::time::Instant::now();
    let beliefs = scan_panel(&model, &corpus, &panel, args.res_secs)?;
    println!(
        "[horizon] scanned {} panel entries in {:.1}s, belief cache {:.2} GiB",
        beliefs.entries(),
        started.elapsed().as_secs_f64(),
        beliefs.bytes() as f64 / (1u64 << 30) as f64
    );
    let marginal_panel = marginal_forecasts(&panel, supports);
    let marginal = marginal_panel
        .first()
        .cloned()
        .context("the panel has no instants")?;

    let cost = FlatCost::new(args.cost_bps);
    let mut rows = Vec::new();
    for &k in &HOLD_HORIZONS {
        let periods = schedule(&corpus, &panel, &beliefs, k)?;
        for &construction in &CONSTRUCTIONS {
            // At k = 1 the aggregate IS the fitted one-bar categorical law: both simple-return
            // moments and the log-return drift/variance are exact cached reductions. Ancestral
            // sampling remains active for the plain-estimator diagnostic, but cannot perturb
            // sizing. At k > 1 the aggregate first/raw-second moments come from the paths.
            let replicates = if construction.is_sampled() {
                args.replicates
            } else {
                1
            };
            let mut per_policy: BTreeMap<usize, Vec<HorizonMetrics>> = BTreeMap::new();
            let mut mechanism = HorizonMechanism::default();
            for replicate in 0..replicates {
                let sampled = if construction.is_sampled() {
                    tch::manual_seed(args.seed + replicate as i64);
                    Some(horizon_laws(
                        &model,
                        &corpus,
                        &panel,
                        &beliefs,
                        &periods,
                        args.res_secs,
                        args.samples,
                    )?)
                } else {
                    None
                };
                let inputs = build_inputs(
                    construction,
                    &beliefs,
                    &periods,
                    sampled.as_deref(),
                    &marginal,
                )?;
                if replicate == 0 {
                    mechanism = HorizonMechanism::of(&periods, &inputs);
                }
                for (p, &policy) in POLICIES.iter().enumerate() {
                    let metrics = measure(
                        &panel,
                        &periods,
                        &inputs,
                        policy,
                        k,
                        args.gross_cap,
                        &cost,
                        args.capital_usd,
                    )?;
                    per_policy.entry(p).or_default().push(metrics);
                }
            }
            for (p, &policy) in POLICIES.iter().enumerate() {
                let set = &per_policy[&p];
                let pick = |f: fn(&HorizonMetrics) -> f64| -> (f64, f64) {
                    replicate_stats(&set.iter().map(f).collect::<Vec<_>>())
                };
                let (break_even, break_even_se) =
                    pick(|m| displayed_break_even(m.break_even_cost_bps));
                let (gross_growth, gross_growth_se) = pick(|m| m.gross_log_growth_per_year);
                let (net_growth, net_growth_se) = pick(|m| m.log_growth_per_year);
                let (sharpe, sharpe_se) = pick(|m| m.sharpe);
                let mut metrics = set[0];
                metrics.break_even_cost_bps = break_even;
                metrics.gross_log_growth_per_year = gross_growth;
                metrics.log_growth_per_year = net_growth;
                metrics.sharpe = sharpe;
                rows.push(HorizonRow {
                    k,
                    construction,
                    policy,
                    periods: periods.len(),
                    metrics,
                    break_even_se,
                    gross_growth_se,
                    net_growth_se,
                    sharpe_se,
                    replicates,
                    mechanism,
                });
            }
        }
        println!(
            "[horizon] k={k} done at {:.1}s ({} periods)",
            started.elapsed().as_secs_f64(),
            periods.len()
        );
    }

    let breadth = panel.breadth();
    let frontier = HorizonFrontier {
        rows,
        gross_cap: args.gross_cap,
        cost_bps: f64::from(args.cost_bps),
        cost_label: format!("flat {:.2} bps one-way", args.cost_bps),
        samples: args.samples,
        replicates: args.replicates,
        instants: panel.instants(),
        symbols: panel.symbols().len(),
        mean_breadth: breadth.mean,
        trading_days: panel.trading_days(),
        span_years: panel.span_years(),
        first_ts_ms: panel.slices().first().map_or(0, |s| s.ts_ms),
        last_ts_ms: panel.slices().last().map_or(0, |s| s.ts_ms),
        checkpoint: args.checkpoint.display().to_string(),
        lineage_sha256: model.lineage_sha256().to_owned(),
    };
    write_horizon_frontier(&args.gens_dir, &args.label, &frontier)?;
    Ok(frontier)
}

// ---------------------------------------------------------------------------
// The report
// ---------------------------------------------------------------------------

/// Write [`HORIZON_FRONTIER_BASE`]: break-even, growth, turnover, Sharpe, factor exposure and
/// leverage error against the holding horizon, for every policy under every construction,
/// plus the replicate standard errors and the mechanism columns behind the model's curve.
///
/// The x-axis is the INDEX into [`HOLD_HORIZONS`], with `k` itself carried as a series, for
/// the same reason [`super::portfolio`]'s frontier carries its band: a `MultiLine` report has
/// no independent x values, and an axis running 1..390 on a linear scale hides everything
/// below a day.
pub fn write_horizon_frontier(dir: &Path, label: &str, frontier: &HorizonFrontier) -> Result<()> {
    ensure!(
        !frontier.rows.is_empty(),
        "the horizon sweep measured nothing, so there is nothing to write"
    );
    let ks: Vec<usize> = HOLD_HORIZONS.to_vec();
    let mut series = vec![
        ReportSeries {
            label: "k (bars held)".to_owned(),
            values: ks.iter().map(|k| *k as f32).collect(),
        },
        ReportSeries {
            label: "periods".to_owned(),
            values: ks
                .iter()
                .map(|k| {
                    frontier
                        .rows
                        .iter()
                        .find(|r| r.k == *k)
                        .map_or(f32::NAN, |r| r.periods as f32)
                })
                .collect(),
        },
        ReportSeries {
            label: format!(
                "matched measured cost {MATCHED_MEASURED_BPS:.3} bps (equal-weighted, 256 traded)"
            ),
            values: vec![MATCHED_MEASURED_BPS as f32; ks.len()],
        },
        ReportSeries {
            label: format!(
                "universe measured cost {UNIVERSE_MEASURED_BPS:.3} bps (equal-weighted, 5,297 \
                 symbols)"
            ),
            values: vec![UNIVERSE_MEASURED_BPS as f32; ks.len()],
        },
        ReportSeries {
            label: format!(
                "matched deepest-decile measured cost {MATCHED_DEEPEST_DECILE_BPS:.3} bps \
                 (equal-weighted mean, 43 of 256 traded names, \
                 +/-{MATCHED_DEEPEST_DECILE_BOUNDARY_BPS:.3} worst-case one-name boundary)"
            ),
            values: vec![MATCHED_DEEPEST_DECILE_BPS as f32; ks.len()],
        },
        ReportSeries {
            label: format!("matched all-in cost {MATCHED_ALL_IN_BPS:.3} bps"),
            values: vec![MATCHED_ALL_IN_BPS as f32; ks.len()],
        },
    ];

    let at = |k: usize, c: Construction, p: Policy, f: &dyn Fn(&HorizonRow) -> f64| -> f32 {
        frontier.row(k, c, p).map_or(f32::NAN, |r| f(r) as f32)
    };
    type Pick = &'static dyn Fn(&HorizonRow) -> f64;
    let per_policy: [(&str, Pick); 7] = [
        ("break-even bps", &|r: &HorizonRow| {
            displayed_break_even(r.metrics.break_even_cost_bps)
        }),
        ("gross log growth/yr", &|r: &HorizonRow| {
            r.metrics.gross_log_growth_per_year
        }),
        ("net log growth/yr", &|r: &HorizonRow| {
            r.metrics.log_growth_per_year
        }),
        ("turnover/day", &|r: &HorizonRow| r.metrics.turnover_per_day),
        ("Sharpe", &|r: &HorizonRow| r.metrics.sharpe),
        ("first-factor exposure", &|r: &HorizonRow| {
            r.metrics.mean_first_factor_exposure
        }),
        ("leverage error", &|r: &HorizonRow| r.metrics.leverage_error),
    ];
    for &construction in &CONSTRUCTIONS {
        for &policy in &POLICIES {
            for (name, pick) in &per_policy {
                series.push(ReportSeries {
                    label: format!("{} {} {name}", construction.name(), policy.name()),
                    values: ks
                        .iter()
                        .map(|k| at(*k, construction, policy, *pick))
                        .collect(),
                });
            }
        }
    }
    // The model's own error bars and the mechanism behind its curve. Only the model rows carry
    // these: a baseline has no forecast, so it has neither a drift nor a standard error of one.
    let model_only: [(&str, Pick); 11] = [
        ("break-even SE", &|r: &HorizonRow| r.break_even_se),
        ("gross growth SE", &|r: &HorizonRow| r.gross_growth_se),
        ("net growth SE", &|r: &HorizonRow| r.net_growth_se),
        ("Sharpe SE", &|r: &HorizonRow| r.sharpe_se),
        ("predicted drift bps", &|r: &HorizonRow| {
            r.mechanism.rb_mu_bps
        }),
        ("plain-estimator drift bps", &|r: &HorizonRow| {
            r.mechanism.plain_mu_bps
        }),
        ("drift MC SE bps", &|r: &HorizonRow| {
            r.mechanism.rb_mu_se_bps
        }),
        ("plain drift MC SE bps", &|r: &HorizonRow| {
            r.mechanism.plain_mu_se_bps
        }),
        ("predicted sigma bps", &|r: &HorizonRow| {
            r.mechanism.pred_sigma_bps
        }),
        ("realized sigma bps", &|r: &HorizonRow| {
            r.mechanism.realized_sigma_bps
        }),
        ("sign agreement", &|r: &HorizonRow| {
            r.mechanism.sign_agreement
        }),
    ];
    for (name, pick) in &model_only {
        for &construction in &CONSTRUCTIONS {
            series.push(ReportSeries {
                label: format!("{} model {name}", construction.name()),
                values: ks
                    .iter()
                    .map(|k| at(*k, construction, Policy::Model, *pick))
                    .collect(),
            });
        }
    }
    series.push(ReportSeries {
        label: "realized up fraction".to_owned(),
        values: ks
            .iter()
            .map(|k| {
                at(*k, Construction::Stale, Policy::Model, &|r: &HorizonRow| {
                    r.mechanism.realized_up_fraction
                })
            })
            .collect(),
    });

    let path = dir.join(format!("{HORIZON_FRONTIER_BASE}.report.bin"));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    write_report(
        &path,
        &Report {
            title: format!(
                "Break-even vs Holding Horizon at {:.1}x Gross - {label}",
                frontier.gross_cap
            ),
            x_label: Some("holding-horizon index (see the `k (bars held)` series)".to_owned()),
            y_label: Some("annualized; break-even and drift in bps".to_owned()),
            scale: ScaleKind::Symlog,
            kind: ReportKind::MultiLine { series },
        },
    )
    .with_context(|| format!("writing {}", path.display()))?;
    // Reading it back is what turns "the writer ran" into "the chart exists": a truncated or
    // all-NaN series renders as a blank panel and nothing else notices.
    let report = read_report(&path).with_context(|| format!("reading back {}", path.display()))?;
    match report.kind {
        ReportKind::MultiLine { series } => ensure!(
            series
                .iter()
                .any(|s| s.values.iter().any(|v| v.is_finite())),
            "{HORIZON_FRONTIER_BASE} holds no finite value"
        ),
        other => bail!("{HORIZON_FRONTIER_BASE} came back as {other:?}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::bar_dist::{BAR_CHAIN, BAR_DOF};
    use crate::torch::test_rng;
    use crate::torch::train::portfolio::{
        backtest, BacktestConfig, PanelSlice, PolicyInputs, GROSS_CAPS,
    };
    use crate::torch::world_model::{world_model_supports_path, BarModules, BarWorldModelMetadata};
    use shared::bars::{write_bar_file, PackedBar, FILE_EXTENSION};
    use std::sync::atomic::{AtomicU64, Ordering};
    use tch::nn;

    static SCRATCH: AtomicU64 = AtomicU64::new(0);

    fn scratch_dir(name: &str) -> PathBuf {
        let unique = SCRATCH.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("horizon_{name}_{}_{unique}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn mix(seed: u64, index: u64) -> f64 {
        crate::torch::dataset::mix64(seed, index) as f64 / u64::MAX as f64
    }

    const RES: u32 = 300;
    const STEP_MS: i64 = RES as i64 * 1_000;
    /// Enough history that the first tradeable bar clears [`BELIEF_PRE_CONTEXT`], and no more:
    /// every extra bar is a trunk token this test pays for on the CPU.
    const HISTORY_BARS: usize = BELIEF_PRE_CONTEXT as usize + 40;
    const VAL_BARS: usize = 48;
    const FIXTURE_SYMBOLS: usize = 3;

    /// A synthetic bar series with a stated random walk. Deterministic in `seed` alone, so two
    /// fixtures built from the same seed hold byte-identical files.
    fn fixture_bars(seed: u64) -> Vec<PackedBar> {
        let base = 1_600_000_000_000i64 / STEP_MS * STEP_MS;
        let mut close = 100.0f32;
        (0..(HISTORY_BARS + VAL_BARS) as u64)
            .map(|slot| {
                let open = close;
                close = (close * (1.0 + 0.01 * (2.0 * mix(seed, 4 * slot) - 1.0) as f32)).max(1.0);
                let spread = (0.004 * mix(seed, 4 * slot + 1)) as f32 * open;
                PackedBar {
                    ts_ms: base + slot as i64 * STEP_MS,
                    open,
                    high: open.max(close) + spread,
                    low: (open.min(close) - spread).max(0.5),
                    close,
                    volume: (1_000.0 + 49_000.0 * mix(seed, 4 * slot + 2)) as f32,
                    vwap: 0.5 * (open + close),
                    trades: 1 + (499.0 * mix(seed, 4 * slot + 3)) as u32,
                }
            })
            .collect()
    }

    /// Rotate the OHLCV payload of `bars[range]` by one slot while leaving every `ts_ms` where
    /// it was. Timestamps stay fixed so only realized observations move; the rollout clock
    /// itself comes from the deterministic exchange calendar and never reads these future rows.
    fn rotate_payload(bars: &mut [PackedBar], from: usize, len: usize) {
        assert!(
            len >= 2,
            "a rotation of fewer than two bars is the identity"
        );
        let stamps: Vec<i64> = bars[from..from + len].iter().map(|b| b.ts_ms).collect();
        bars[from..from + len].rotate_left(1);
        for (bar, ts) in bars[from..from + len].iter_mut().zip(&stamps) {
            bar.ts_ms = *ts;
        }
    }

    /// A corpus, a randomized frozen checkpoint and the panel over the held-out span.
    ///
    /// The checkpoint's weights are RANDOMIZED rather than left at `BarModules::new`'s
    /// initialization, and that is load-bearing rather than decorative: an untrained
    /// [`crate::torch::world_model::BarDynamics`] is documented to be the identity and an
    /// untrained head emits one law for every belief, so on a zero-init checkpoint every
    /// forecast in this module would be the same number and every equality assertion below
    /// would hold no matter how badly the code leaked. Each test that asserts an equality also
    /// asserts that the laws it compared are spread out, which is what turns that from an
    /// intention into a check.
    struct Fixture {
        dir: PathBuf,
        corpus: BarCorpus,
        panel: Panel,
        model: BarWorldModel,
        bounds: (i64, i64),
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    impl Fixture {
        /// `mutate` sees each symbol's bars before they are written, so a test can move the
        /// realized future out from under a forecast.
        ///
        /// # The seeding order is load-bearing, and it is a trap
        ///
        /// [`BarModules::new`] (`torch::world_model`) builds every projection through
        /// `uniform_init`, which draws from the GLOBAL torch generator. So the seed must be set
        /// BEFORE the constructor runs, not before the perturbation that follows it. Seeding
        /// after `new` makes only the perturbation reproducible and leaves the base weights
        /// dependent on whatever the harness happened to draw earlier in the process.
        ///
        /// The failure that produces is worth naming, because it cost a debug cycle here and it
        /// does not look like what it is: two fixtures built by identical code hold DIFFERENT
        /// checkpoints, so a differential test over them - "permute the realized future, the
        /// forecast must not move" - sees two completely unrelated belief vectors and reports a
        /// LOOKAHEAD LEAK. The signature is that the two beliefs are unrelated rather than
        /// slightly displaced: a real one-bar leak perturbs a belief, it does not replace it.
        ///
        /// Any test in this repository that builds two model instances and compares them is
        /// exposed to this, which is several. `RngIsolation` found the sibling defect in
        /// production code the same day (`build_trainer` reseeding the global generator), so
        /// this is the second instance of one class: a global generator consumed by a
        /// constructor that does not advertise it.
        fn new(label: &str, mutate: impl Fn(usize, &mut Vec<PackedBar>)) -> Self {
            let dir = scratch_dir(label);
            let mut first_val_ts = 0i64;
            for index in 0..FIXTURE_SYMBOLS {
                let mut bars = fixture_bars(11 + index as u64);
                first_val_ts = bars[HISTORY_BARS].ts_ms;
                mutate(index, &mut bars);
                let symbol = format!("S{index}");
                write_bar_file(
                    &dir.join(format!("{symbol}.{RES}.{FILE_EXTENSION}")),
                    &symbol,
                    RES,
                    &bars,
                )
                .expect("write fixture bars");
            }
            let last_ts = first_val_ts + (VAL_BARS as i64) * STEP_MS;
            let bounds = (first_val_ts, last_ts);
            let corpus = BarCorpus::load_with_bounds(&dir, RES, HISTORY_BARS, bounds)
                .expect("fixture corpus");
            assert_eq!(corpus.split_bounds(), bounds);

            // The supports are fitted on TRAIN bars only, which the mutations below never
            // touch, so two fixtures that differ by a val-span permutation share bin geometry
            // exactly and a law comparison between them is a comparison of the same quantity.
            let weights = dir.join("world_model.ot");
            corpus
                .fit_supports(4_096, 5)
                .save(&world_model_supports_path(&weights, RES))
                .expect("save supports");
            // The seed goes BEFORE `BarModules::new`, not after: `new` runs `uniform_init` on
            // every projection and therefore consumes the GLOBAL torch RNG. Seeding after it
            // makes only the perturbation reproducible, leaves the base weights dependent on
            // whatever the harness drew before this fixture, and produces two fixtures with
            // different checkpoints — which reads exactly like a lookahead failure and is not
            // one.
            let mut vs = nn::VarStore::new(Device::Cpu);
            tch::manual_seed(4_242);
            let _modules = BarModules::new(&vs.root());
            tch::no_grad(|| {
                // One draw per variable in NAME order, so the perturbation does not depend on
                // the var store's hash iteration order either.
                let mut named: Vec<(String, Tensor)> = vs.variables().into_iter().collect();
                named.sort_by(|a, b| a.0.cmp(&b.0));
                for (_, mut tensor) in named {
                    let perturbed = &tensor + Tensor::randn_like(&tensor) * 0.05;
                    tensor.copy_(&perturbed);
                }
            });
            vs.freeze();
            vs.save(&weights).expect("save weights");
            let metadata = BarWorldModelMetadata::save_for_checkpoint(&weights, &[RES], RES)
                .expect("save metadata");
            let model =
                BarWorldModel::load(&weights, &metadata, Device::Cpu).expect("load fixture model");

            let config = PanelConfig {
                start_ts_ms: bounds.0,
                end_ts_ms: bounds.1,
                max_symbols: FIXTURE_SYMBOLS,
                min_history: BELIEF_PRE_CONTEXT as usize + 1,
                max_instants: VAL_BARS,
            };
            let panel = Panel::build(&corpus, &config).expect("fixture panel");
            assert!(
                panel.instants() >= 16 && panel.symbols().len() == FIXTURE_SYMBOLS,
                "the fixture panel is {} instants over {} symbols, too small to schedule",
                panel.instants(),
                panel.symbols().len()
            );
            Self {
                dir,
                corpus,
                panel,
                model,
                bounds,
            }
        }

        fn beliefs(&self) -> PanelBeliefs {
            scan_panel(&self.model, &self.corpus, &self.panel, RES).expect("panel scan")
        }

        fn marginal(&self) -> PanelForecast {
            let supports = self.model.supports_for(RES).expect("fixture supports");
            marginal_forecasts(&self.panel, supports)
                .first()
                .cloned()
                .expect("a marginal forecast")
        }
    }

    #[test]
    fn receding_schedule_excludes_boundaries_and_never_shortens_horizon() {
        let _torch_rng_guard = test_rng::exclusive();
        const H: usize = 4;
        let fixture = Fixture::new("receding_boundaries", |_, _| {});
        let beliefs = fixture.beliefs();
        let periods =
            receding_schedule(&fixture.corpus, &fixture.panel, &beliefs, H).expect("schedule");
        assert_eq!(
            periods.len(),
            fixture.panel.instants() - (H - 1),
            "the final H-1 decisions lack a complete realized H-step endpoint"
        );
        assert_eq!(
            periods.last().expect("eligible decisions").instant,
            fixture.panel.instants() - H
        );
        assert!(periods.iter().all(|period| {
            period.legs.len() == fixture.panel.slices()[period.instant].symbols.len()
                && period.legs.iter().all(|leg| leg.steps == H)
        }));
    }

    /// Spread of a set of numbers, as the non-vacuity witness every equality test below carries.
    fn spread(values: &[f64]) -> f64 {
        let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.len() < 2 {
            return 0.0;
        }
        let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
        max - min
    }

    // -----------------------------------------------------------------------
    // Arithmetic, no model
    // -----------------------------------------------------------------------

    /// The lognormal closure has to share the categorical moment reduction's sign and zero,
    /// because those are the only two properties that survive the gross projection into the
    /// book: projection destroys the scale, so the book trades the sign and cross-sectional
    /// ordering of the Kelly vector and nothing else.
    ///
    /// Pinned against [`kelly_fractions`]'s fitted first/second-moment reduction on a
    /// discretized lognormal whose two moments are the closure's own inputs. Discretization
    /// need not reproduce the continuous magnitude exactly; the reductions must agree on sign
    /// and on where zero is.
    #[test]
    fn the_moment_closure_shares_the_categorical_reduction_sign_and_zero() {
        let var = 1.0e-5f64;
        let sd = var.sqrt();
        // A 128-bin discretization of `N(mu, var)` in LOG space, on the same +/- 6 sigma grid
        // for every `mu`, converted directly to simple returns with `expm1`.
        let categorical = |mu: f64| -> f64 {
            let bins = NUM_BAR_BINS as usize;
            let (lo, hi) = (-6.0 * sd, 6.0 * sd);
            let width = (hi - lo) / bins as f64;
            let mut probs = Vec::with_capacity(bins);
            let mut returns = Vec::with_capacity(bins);
            for bin in 0..bins {
                let center = lo + width * (bin as f64 + 0.5);
                let z = (center - mu) / sd;
                probs.push((-0.5 * z * z).exp());
                returns.push(center.exp_m1());
            }
            let mass: f64 = probs.iter().sum();
            for p in probs.iter_mut() {
                *p /= mass;
            }
            let return_seconds: Vec<f64> = returns.iter().map(|value| value * value).collect();
            let probs = Tensor::from_slice(&probs).view([1, NUM_BAR_BINS]);
            let returns = Tensor::from_slice(&returns).view([1, NUM_BAR_BINS]);
            let return_seconds = Tensor::from_slice(&return_seconds).view([1, NUM_BAR_BINS]);
            kelly_fractions(&probs, &returns, &return_seconds, FREE_LEVERAGE).double_value(&[0])
        };

        // At `mu_log = 0` the SIMPLE return still has a positive mean, `exp(var/2) - 1`, so the
        // quadratic Kelly fraction is positive rather than zero. Both must say so.
        assert!(
            closure_kelly(0.0, var) > 0.0 && categorical(0.0) > 0.0,
            "at mu_log = 0 both fractions must be positive: closure {} categorical {}",
            closure_kelly(0.0, var),
            categorical(0.0)
        );
        // The zero sits where the simple mean vanishes, at `mu_log = -var/2`, for both.
        let at_zero = closure_kelly(-0.5 * var, var);
        assert!(
            at_zero.abs() < 1.0e-4 * closure_kelly(1.0e-3, var).abs(),
            "the closure's zero should sit at mu_log = -var/2, got f = {at_zero}"
        );
        assert!(
            categorical(-0.5 * var).abs() < 0.05 * categorical(1.0e-3).abs(),
            "the categorical reduction's zero is not at mu_log = -var/2, so the grid is too \
             coarse to test anything: f = {}",
            categorical(-0.5 * var)
        );
        // Sign agreement across the range a 5-minute bar actually spans.
        let mut nonzero = 0usize;
        for step in 0..32 {
            let mu = -2.0e-3 + 4.0e-3 * f64::from(step) / 31.0;
            let (closed, solved) = (closure_kelly(mu, var), categorical(mu));
            assert_eq!(
                closed > 0.0,
                solved > 0.0,
                "the closure and categorical reduction disagree on the SIGN at mu_log = {mu}: \
                 {closed} vs {solved}"
            );
            if solved.abs() > 1.0e-6 {
                nonzero += 1;
            }
        }
        assert!(
            nonzero >= 24,
            "only {nonzero} of 32 probe drifts produced a nonzero categorical fraction, so the \
             sign agreement above is mostly the agreement of two zeros"
        );
        // Strictly increasing in the drift, which is what makes the cross-section an ordering.
        let mut previous = f64::NEG_INFINITY;
        for step in 0..64 {
            let mu = -2.0e-3 + 4.0e-3 * f64::from(step) / 63.0;
            let f = closure_kelly(mu, var);
            assert!(
                f > previous,
                "the closure is not monotone in the drift at mu_log = {mu}: {f} <= {previous}"
            );
            previous = f;
        }
        // A degenerate law cannot divide by zero, and a huge drift cannot escape the shared
        // leverage bound.
        assert!(closure_kelly(1.0e-4, 0.0).is_finite());
        assert!(closure_kelly(50.0, 1.0e-5).abs() <= FREE_LEVERAGE);
    }

    /// The report base has to exist on disk and hold finite values for every registered series,
    /// because a `MultiLine` report of all-NaN renders as a blank panel and nothing notices.
    ///
    /// This is the test named in `pretrain_reports::tests::CYCLE_EXEMPT` for
    /// `pretrain_horizon_frontier`.
    #[test]
    fn the_horizon_frontier_base_is_written_and_read_back() {
        let dir = scratch_dir("frontier");
        let mut rows = Vec::new();
        for (index, &k) in HOLD_HORIZONS.iter().enumerate() {
            for &construction in &CONSTRUCTIONS {
                for &policy in &POLICIES {
                    let base = 1.0 + index as f64;
                    rows.push(HorizonRow {
                        k,
                        construction,
                        policy,
                        periods: 400 / k.max(1),
                        metrics: HorizonMetrics {
                            periods: 400 / k.max(1),
                            span_years: 0.42,
                            periods_per_year: 1.0,
                            final_log_wealth: base,
                            log_growth_per_year: -base,
                            gross_log_growth_per_year: base,
                            break_even_cost_bps: base,
                            cagr: 0.1,
                            sharpe: base,
                            vol: 0.2,
                            max_drawdown: 0.3,
                            mean_gross: 2.0,
                            max_gross: 2.1,
                            mean_net: 0.0,
                            turnover_per_day: base,
                            rotation_per_period: 0.5,
                            bound_fraction: 1.0,
                            mean_first_factor_exposure: 0.27,
                            first_factor_share: 0.27,
                            leverage_error: 1.9,
                            ruined_at_period: f64::NAN,
                        },
                        break_even_se: 0.01,
                        gross_growth_se: 0.02,
                        net_growth_se: 0.03,
                        sharpe_se: 0.04,
                        replicates: 3,
                        mechanism: HorizonMechanism {
                            rb_mu_bps: 1.0,
                            plain_mu_bps: 1.1,
                            rb_mu_se_bps: 0.1,
                            plain_mu_se_bps: 3.0,
                            pred_sigma_bps: 31.0,
                            realized_sigma_bps: 30.0,
                            sign_agreement: 0.5,
                            realized_up_fraction: 0.49,
                            legs: 1_000,
                        },
                    });
                }
            }
        }
        let frontier = HorizonFrontier {
            rows,
            gross_cap: DEFAULT_GROSS_CAP,
            cost_bps: 2.0,
            cost_label: "flat 2.00 bps one-way".to_owned(),
            samples: DEFAULT_SAMPLES,
            replicates: DEFAULT_REPLICATES,
            instants: 400,
            symbols: 48,
            mean_breadth: 47.0,
            trading_days: 100,
            span_years: 0.42,
            first_ts_ms: 1,
            last_ts_ms: 2,
            checkpoint: "weights/pretrain_best.ot".to_owned(),
            lineage_sha256: "0".repeat(64),
        };
        write_horizon_frontier(&dir, "unit", &frontier).expect("write the frontier");

        let path = dir.join(format!("{HORIZON_FRONTIER_BASE}.report.bin"));
        let report = read_report(&path).expect("read the frontier back");
        let ReportKind::MultiLine { series } = report.kind else {
            panic!("the frontier must be a MultiLine report");
        };
        // Every construction x policy x column, plus the model-only columns, plus the axis and
        // the four cost reference lines. Each has one value per holding horizon.
        let expected =
            4 + 2 + CONSTRUCTIONS.len() * POLICIES.len() * 7 + CONSTRUCTIONS.len() * 11 + 1;
        assert_eq!(series.len(), expected, "the frontier's series count moved");
        for line in &series {
            assert_eq!(
                line.values.len(),
                HOLD_HORIZONS.len(),
                "series `{}` is not one value per holding horizon",
                line.label
            );
            assert!(
                line.values.iter().any(|v| v.is_finite()),
                "series `{}` came back with no finite value at all",
                line.label
            );
        }
        for construction in CONSTRUCTIONS {
            for policy in POLICIES {
                let label = format!("{} {} break-even bps", construction.name(), policy.name());
                assert!(
                    series.iter().any(|s| s.label == label),
                    "the frontier is missing the `{label}` series, so a policy is unreadable \
                     at some horizon"
                );
            }
        }
        assert!(
            series
                .iter()
                .any(|s| s.label.contains(&format!("{MATCHED_MEASURED_BPS:.3}"))),
            "the frontier must carry the matched measured cost as a reference line"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every field unmeasured. Only the field a test is about is then set, so a test cannot
    /// accidentally depend on a plausible-looking default.
    fn nan_metrics() -> HorizonMetrics {
        HorizonMetrics {
            periods: 0,
            span_years: f64::NAN,
            periods_per_year: f64::NAN,
            final_log_wealth: f64::NAN,
            log_growth_per_year: f64::NAN,
            gross_log_growth_per_year: f64::NAN,
            break_even_cost_bps: f64::NAN,
            cagr: f64::NAN,
            sharpe: f64::NAN,
            vol: f64::NAN,
            max_drawdown: f64::NAN,
            mean_gross: f64::NAN,
            max_gross: f64::NAN,
            mean_net: f64::NAN,
            turnover_per_day: f64::NAN,
            rotation_per_period: f64::NAN,
            bound_fraction: f64::NAN,
            mean_first_factor_exposure: f64::NAN,
            first_factor_share: f64::NAN,
            leverage_error: f64::NAN,
            ruined_at_period: f64::NAN,
        }
    }

    /// The verdict has to name the baseline when a baseline wins, because a corner where
    /// equal-weight beats the model is the exact failure the band frontier already walked into.
    #[test]
    fn the_verdict_says_so_when_a_baseline_beats_the_model() {
        let row = |policy: Policy, bps: f64| HorizonRow {
            k: 39,
            construction: Construction::Horizon,
            policy,
            periods: 200,
            metrics: HorizonMetrics {
                break_even_cost_bps: bps,
                ..nan_metrics()
            },
            break_even_se: 0.5,
            gross_growth_se: 0.0,
            net_growth_se: 0.0,
            sharpe_se: 0.0,
            replicates: 3,
            mechanism: HorizonMechanism::default(),
        };
        let mut frontier = HorizonFrontier {
            rows: vec![row(Policy::Model, 6.0), row(Policy::EqualWeight, 20.0)],
            gross_cap: 2.0,
            cost_bps: 2.0,
            cost_label: "flat".to_owned(),
            samples: 8,
            replicates: 3,
            instants: 200,
            symbols: 3,
            mean_breadth: 3.0,
            trading_days: 10,
            span_years: 0.1,
            first_ts_ms: 0,
            last_ts_ms: 1,
            checkpoint: String::new(),
            lineage_sha256: "0".repeat(64),
        };
        let verdict = frontier.verdict();
        assert!(
            verdict.contains("BEATS") && verdict.contains("equal weight"),
            "a losing model must be named as losing: {verdict}"
        );
        assert!(
            verdict.contains(&format!("{MATCHED_MEASURED_BPS:.3}")) && verdict.contains("BELOW"),
            "the verdict must state the matched threshold and which side it falls on: {verdict}"
        );
        // And the other way: a model above every baseline and above the matched cost.
        frontier.rows = vec![row(Policy::Model, 40.0), row(Policy::EqualWeight, 2.0)];
        let verdict = frontier.verdict();
        assert!(
            verdict.contains("does not beat") && verdict.contains("ABOVE"),
            "a winning model must be reported as winning: {verdict}"
        );
    }

    // -----------------------------------------------------------------------
    // No lookahead
    // -----------------------------------------------------------------------

    /// THE test this module exists to make trustworthy: a `k`-bar forecast must not see the `k`
    /// bars it forecasts.
    ///
    /// Two directions, both required, because either one alone is worthless.
    ///
    /// * **Invariance.** Rotating the PRICES and VOLUMES of the bars inside holding window 0,
    ///   on disk, leaves period 0's `k`-bar law BIT-IDENTICAL. Those bars are period 0's future,
    ///   and neither their payload nor their timestamps are inputs to the deterministic rollout
    ///   clock.
    /// * **Non-vacuity.** The SAME rotation is period 1's causal history, and period 1's law
    ///   must therefore MOVE. Without this half, a forecaster that returned a constant would
    ///   pass the invariance half perfectly.
    ///
    /// The spread assertions are the third guard: they prove the compared laws differ across
    /// legs at all, so the bit-identity is not the identity of one repeated number.
    #[test]
    fn the_k_bar_law_ignores_the_realized_bars_inside_its_own_window() {
        let _torch_rng_guard = test_rng::exclusive();
        const K: usize = 4;

        let base = Fixture::new("nolook_base", |_, _| {});
        let base_beliefs = base.beliefs();
        let base_periods =
            schedule(&base.corpus, &base.panel, &base_beliefs, K).expect("base schedule");
        assert!(
            base_periods.len() >= 3,
            "this test needs a period 0 and a period 1, got {}",
            base_periods.len()
        );
        let first_bar = base.panel.bar_index(base_periods[0].instant, 0) as usize;
        tch::manual_seed(7);
        let base_laws = horizon_laws(
            &base.model,
            &base.corpus,
            &base.panel,
            &base_beliefs,
            &base_periods,
            RES,
            8,
        )
        .expect("base laws");

        // The mutation: rotate holding window 0's payload for every symbol. Window 0 spans the
        // panel's first K instants, which for a dense fixture is bars `first_bar .. first_bar + K`.
        let moved = Fixture::new("nolook_moved", |_, bars| {
            rotate_payload(bars, first_bar, K);
        });
        assert_eq!(
            moved.bounds, base.bounds,
            "the two fixtures must be scored against the same split"
        );
        let moved_beliefs = moved.beliefs();
        let moved_periods =
            schedule(&moved.corpus, &moved.panel, &moved_beliefs, K).expect("moved schedule");
        assert_eq!(
            moved_periods.len(),
            base_periods.len(),
            "the rotation must not change the panel's clock"
        );
        tch::manual_seed(7);
        let moved_laws = horizon_laws(
            &moved.model,
            &moved.corpus,
            &moved.panel,
            &moved_beliefs,
            &moved_periods,
            RES,
            8,
        )
        .expect("moved laws");

        // Localize before comparing laws, so a failure names the layer that leaked rather than
        // the layer that reported. Each of these is a strictly stronger claim than the one
        // below it.
        let base_supports = base.model.supports_for(RES).expect("base supports");
        let moved_supports = moved.model.supports_for(RES).expect("moved supports");
        assert_eq!(
            base_supports.centers(DOF_R),
            moved_supports.centers(DOF_R),
            "the two fixtures fitted DIFFERENT bin geometry, so their laws are not comparable \
             at all; the permutation must be confined to the val span"
        );
        for (slot, (&a, &b)) in base_beliefs.row_of[0]
            .iter()
            .zip(&moved_beliefs.row_of[0])
            .enumerate()
        {
            assert_eq!(a, b, "slot {slot} of instant 0 moved in the belief cache");
            assert_eq!(
                base_beliefs.belief_row(a),
                moved_beliefs.belief_row(b),
                "the BELIEF at instant 0 slot {slot} changed when bars at and after instant 0 \
                 were permuted, so `scan_panel` is reading the bar it is about to predict"
            );
            assert_eq!(
                base_beliefs.mu_log[0][slot].to_bits(),
                moved_beliefs.mu_log[0][slot].to_bits(),
                "the one-bar drift at instant 0 slot {slot} changed under the same permutation"
            );
        }

        // Non-vacuity, first pass: the laws being compared are not all the same number.
        let mus: Vec<f64> = base_laws[0].iter().map(|l| l.mu_log).collect();
        assert!(
            spread(&mus) > 1.0e-9,
            "period 0's legs all carry the same drift ({mus:?}), so bit-identity below would \
             prove nothing; the fixture checkpoint is degenerate"
        );

        // INVARIANCE. Period 0's decision stands on bars strictly before `first_bar`, and its
        // rollout reads only its own samples plus the untouched calendar.
        for (leg, (want, got)) in base_laws[0].iter().zip(&moved_laws[0]).enumerate() {
            assert_eq!(
                want.mu_log.to_bits(),
                got.mu_log.to_bits(),
                "leg {leg} of period 0 changed its k-bar drift when the realized bars INSIDE \
                 its own holding window were permuted: {} -> {}. That is lookahead.",
                want.mu_log,
                got.mu_log
            );
            assert_eq!(
                want.var_log.to_bits(),
                got.var_log.to_bits(),
                "leg {leg} of period 0 changed its k-bar variance under the same permutation"
            );
            assert_eq!(
                want.plain_mu_log.to_bits(),
                got.plain_mu_log.to_bits(),
                "leg {leg} of period 0 changed its plain-estimator drift under the same \
                 permutation"
            );
        }

        // NON-VACUITY. The same bars are period 1's causal history, so period 1 must move. A
        // forecaster that ignored its inputs entirely would fail exactly here.
        let changed = base_laws[1]
            .iter()
            .zip(&moved_laws[1])
            .filter(|(want, got)| want.mu_log.to_bits() != got.mu_log.to_bits())
            .count();
        assert!(
            changed > 0,
            "period 1's laws are unchanged after its CAUSAL HISTORY was permuted, so the \
             invariance above is vacuous: the forecast is not reading its history at all"
        );
    }

    /// The one-bar law that sizes every stale row, and every step of every rollout, must be
    /// `p(r | strictly past bars)` — never a law that knows any part of the bar it predicts.
    ///
    /// `r`'s prefix set is DERIVED from [`BAR_CHAIN`] here rather than assumed, so this test
    /// stays correct under any factorization order. `r` currently HEADS the chain, so that set
    /// is empty, `p(r | past)` is the head's `r` row at ANY prefix, and the property asserted
    /// is exactly that: the row is bit-identical across a sweep of prefix assignments and the
    /// panel's drift is its mean. A reorder that hands `r` a prefix breaks the sweep and fails
    /// here rather than silently certifying a teacher-forced law.
    ///
    /// The reference is independent of the path under test: it calls
    /// [`BarEmissionHead::logits`] directly and reduces in `f64`, never touching
    /// [`forecast_r_probs`].
    #[test]
    fn the_one_bar_drift_uses_no_part_of_the_bar_it_predicts() {
        let _torch_rng_guard = test_rng::exclusive();
        let fixture = Fixture::new("marginal", |_, _| {});
        let beliefs = fixture.beliefs();
        let supports = fixture.model.supports_for(RES).expect("fixture supports");
        let centers = supports.centers(DOF_R).to_vec();
        let head = fixture.model.head();

        // `r`'s prefix set, derived rather than assumed: every factor ahead of it in the
        // chain. Empty while `r` heads the chain, which is what makes a direct read a
        // forecast; anything in it would have to be marginalized out instead.
        let r_position = BAR_CHAIN
            .iter()
            .position(|&dof| dof == DOF_R)
            .expect("r is a chain factor");
        assert!(
            BAR_CHAIN[..r_position].is_empty(),
            "r has chain prefix {:?}, so the panel's one-bar law is teacher-forced on it and \
             the direct reference below is the wrong law",
            &BAR_CHAIN[..r_position]
        );
        // The last chain factor carries every prefix slot, so it is the control that proves
        // the head's prefix pathway is live at all.
        let deepest = BAR_CHAIN[BAR_DOF - 1];

        // Prefix assignments the `r` row must be blind to. Bin 0 is the unvisited seed, so the
        // others are the ones that could leak.
        let prefixes: Vec<Tensor> = [0i64, 1, NUM_BAR_BINS / 2, NUM_BAR_BINS - 1]
            .iter()
            .map(|bin| Tensor::full([1, BAR_DOF as i64], *bin, (Kind::Int64, Device::Cpu)))
            .collect();

        let mut checked = 0usize;
        let mut worst = 0.0f64;
        let mut traded_response = 0.0f64;
        let mut deepest_response = 0.0f64;
        let mut seen = Vec::new();
        for (t, row) in beliefs.row_of.iter().enumerate().take(4) {
            for (slot, &belief_row) in row.iter().enumerate() {
                let h = Tensor::from_slice(beliefs.belief_row(belief_row)).view([1, BAR_MODEL_DIM]);
                let id = fixture.panel.slices()[t].symbols[slot];
                let series = fixture.panel.series_of(id);
                let bar = fixture.panel.bar_index(t, slot);
                let clocks = fixture
                    .corpus
                    .dof_window(
                        &[BarEndpoint {
                            series,
                            bar: bar as usize,
                        }],
                        &[0],
                        2,
                        Device::Cpu,
                    )
                    .expect("two-row clock window");
                let conditioning = fixture
                    .model
                    .trunk()
                    .forecast_conditioning(
                        &clocks.time_ids.narrow(1, 1, 1),
                        &clocks.time_ids.narrow(1, 0, 1),
                    )
                    .reshape([1, BAR_MODEL_DIM]);
                let (want, traded_drift, deepest_drift) = tch::no_grad(|| {
                    let row_at = |prefix: &Tensor, dof: usize| -> Vec<f64> {
                        Vec::<f64>::try_from(
                            head.logits(&h, &conditioning, prefix)
                                .select(1, dof as i64)
                                .softmax(-1, Kind::Double)
                                .reshape([-1]),
                        )
                        .expect("a probability row")
                    };
                    let spread_of = |a: &[f64], b: &[f64]| -> f64 {
                        a.iter()
                            .zip(b)
                            .map(|(x, y)| (x - y).abs())
                            .fold(0.0f64, f64::max)
                    };
                    let base = row_at(&prefixes[0], DOF_R);
                    let base_deepest = row_at(&prefixes[0], deepest);
                    let mut traded_drift = 0.0f64;
                    let mut deepest_drift = 0.0f64;
                    for prefix in &prefixes[1..] {
                        traded_drift = traded_drift.max(spread_of(&base, &row_at(prefix, DOF_R)));
                        deepest_drift =
                            deepest_drift.max(spread_of(&base_deepest, &row_at(prefix, deepest)));
                    }
                    let mass: f64 = base.iter().sum();
                    assert!(
                        (mass - 1.0).abs() < 1.0e-9,
                        "the reference row at ({t}, {slot}) has mass {mass}"
                    );
                    let mean: f64 = base.iter().zip(&centers).map(|(p, c)| p * c).sum();
                    (mean, traded_drift, deepest_drift)
                });
                let got = beliefs.mu_log[t][slot];
                worst = worst.max((got - want).abs());
                traded_response = traded_response.max(traded_drift);
                deepest_response = deepest_response.max(deepest_drift);
                seen.push(want);
                checked += 1;
            }
        }
        assert!(checked >= 8, "only {checked} beliefs were compared");
        // Both sides are `f32` head logits reduced in `f64`; the quantity is ~1e-4, so 1e-7
        // absolute is three to four significant figures of agreement while still being four
        // orders tighter than the ~1e-3 shift a teacher-forced law would introduce.
        assert!(
            worst < 1.0e-7,
            "the panel's one-bar drift disagrees with a direct read of the head's r row by \
             {worst:.3e} absolute nats, so it is not the law it claims to be"
        );
        // The property. Exactly zero, because a prefix-free row is the SAME arithmetic under
        // every prefix — a tolerance here would pass a law that leaked a little.
        assert_eq!(
            traded_response, 0.0,
            "the head's r row moved across a prefix sweep, so the one-bar law is conditioned \
             on the bar it predicts"
        );
        // Non-vacuity: the prefix pathway is live on this fixture, so the invariance above is
        // the chain position doing its job rather than a dead prefix embedding.
        assert!(
            deepest_response > 1.0e-6,
            "no row of this fixture's head responds to its prefix ({deepest_response:.3e}), so \
             the invariance above proves nothing"
        );
        assert!(
            spread(&seen) > 1.0e-8,
            "every reference drift came out the same number, so the agreement above is the \
             agreement of two constants"
        );
    }

    // -----------------------------------------------------------------------
    // The book
    // -----------------------------------------------------------------------

    /// At `k = 1` and zero cost this book IS [`super::portfolio::backtest`], and it has to be:
    /// a second accounting loop that quietly disagrees with the first would make every row of
    /// the sweep incomparable to the 5-minute number it exists to be compared against.
    ///
    /// Zero cost, because the two loops legitimately differ on what turnover is charged
    /// against — see the module docs — and `1e-12` relative rather than bit-identity because
    /// `portfolio` recomputes the realized simple return from the panel's `f32` log return
    /// while this book recomputes it in `f64` from the corpus closes.
    #[test]
    fn the_one_bar_book_reproduces_the_portfolio_backtest_at_zero_cost() {
        let _torch_rng_guard = test_rng::exclusive();
        let fixture = Fixture::new("onebar", |_, _| {});
        let beliefs = fixture.beliefs();
        let periods = schedule(&fixture.corpus, &fixture.panel, &beliefs, 1).expect("schedule");
        assert_eq!(
            periods.len(),
            fixture.panel.instants(),
            "a one-bar holding period is one period per instant"
        );
        let marginal = fixture.marginal();
        let inputs = build_inputs(
            Construction::StaleCategoricalMoments,
            &beliefs,
            &periods,
            None,
            &marginal,
        )
        .expect("stale-exact inputs");
        let marginal_panel =
            marginal_forecasts(&fixture.panel, fixture.model.supports_for(RES).unwrap());
        let free = FlatCost::new(0.0);

        for &cap in &GROSS_CAPS {
            for &policy in &POLICIES {
                let mine = run_book(
                    &fixture.panel,
                    &periods,
                    &inputs,
                    policy,
                    1,
                    cap,
                    &free,
                    1.0e7,
                )
                .expect("horizon book");
                let theirs = backtest(
                    &fixture.panel,
                    &PolicyInputs {
                        model: &beliefs.one_bar,
                        marginal: &marginal_panel,
                    },
                    policy,
                    cap,
                    &free,
                    &BacktestConfig {
                        capital_usd: 1.0e7,
                        band: 0.0,
                    },
                )
                .expect("portfolio book");
                assert_eq!(
                    mine.log_equity.len(),
                    theirs.log_equity.len(),
                    "{} at {cap}x: the two books have different clocks",
                    policy.name()
                );
                // Absolute, in nats of cumulative log wealth, not relative: log wealth passes
                // through zero, so a relative bound on it is a bound on nothing. `portfolio`
                // reads its realized return from the panel's `f32` log return while this book
                // recomputes it in `f64` from the corpus closes, which is a ~6e-8 relative
                // difference per bar and measured at 9e-11 nats per period here. 1e-6 nats over
                // the whole path is four orders of headroom above that and five orders below any
                // difference a reader would act on.
                for (index, (a, b)) in mine.log_equity.iter().zip(&theirs.log_equity).enumerate() {
                    assert!(
                        (a - b).abs() < 1.0e-6,
                        "{} at {cap}x diverges from the portfolio bench at period {index}: \
                         {a} vs {b}",
                        policy.name()
                    );
                }
                // TURNOVER is where the two books legitimately part, and the gap is bounded
                // rather than tolerated. `portfolio` charges the move from its previous TARGET;
                // this book charges the move from the weight the hold drifted to,
                // `held_i = w_i (1 + R_i) / M` with `M = 1 + sum_j w_j R_j` at zero cost. So
                // `|held_i - w_i| = |w_i| |R_i - (M - 1)| / M` and the two turnovers can differ
                // by at most `gross * (max_i |R_i| + |payoff|) / M`, evaluated on the PREVIOUS
                // period because that is the hold that drifted. Anything larger is not drift.
                assert_eq!(
                    mine.turnover[0],
                    theirs.turnover[0],
                    "{} at {cap}x: both books start flat, so the FIRST rebalance has no drift to \
                     disagree about",
                    policy.name()
                );
                for index in 1..mine.turnover.len() {
                    let previous = index - 1;
                    let max_abs_r = fixture.panel.slices()[previous]
                        .realized_r
                        .iter()
                        .map(|r| f64::from(*r).exp_m1().abs())
                        .fold(0.0, f64::max);
                    let payoff = mine.payoff[previous];
                    // The `1e-6` relative slack is not a fudge: the oracle holds ONE name, so
                    // for it the inequality is an EQUALITY (`max_i |R_i|` is that name's own
                    // return and `payoff` is `w R`), and it was measured to be attained to 2e-10
                    // absolute. The slack covers rounding of a bound assembled from the panel's
                    // `f32` returns while the book compounds in `f64`.
                    let bound = mine.gross[previous] * (max_abs_r + payoff.abs())
                        / (1.0 + payoff).max(1.0e-9)
                        * (1.0 + 1.0e-6)
                        + 1.0e-12;
                    let gap = (mine.turnover[index] - theirs.turnover[index]).abs();
                    assert!(
                        gap <= bound,
                        "{} at {cap}x differs from the portfolio bench by {gap} of turnover at \
                         period {index}, above the {bound} that one bar of buy-and-hold drift \
                         can explain",
                        policy.name()
                    );
                }
                // And the gap must actually BE the drift somewhere, or this book is silently
                // re-imposing its target weights and the k > 1 rows mean nothing.
                if policy == Policy::Model {
                    let moved = (1..mine.turnover.len())
                        .filter(|&i| (mine.turnover[i] - theirs.turnover[i]).abs() > 1.0e-9)
                        .count();
                    assert!(
                        moved > 0,
                        "at {cap}x this book's turnover is identical to the portfolio bench's at \
                         every period, so its positions are not drifting with prices at all"
                    );
                }
            }
        }
        // Non-vacuity: an equality between two flat lines proves nothing.
        let model = run_book(
            &fixture.panel,
            &periods,
            &inputs,
            Policy::Model,
            1,
            DEFAULT_GROSS_CAP,
            &free,
            1.0e7,
        )
        .expect("horizon book");
        assert!(
            spread(&model.log_equity) > 1.0e-6 && model.turnover.iter().sum::<f64>() > 0.0,
            "the compared book never moved, so the agreement above is trivial"
        );
    }

    /// The break-even solve has to be a real bracket of the zero of net growth, not a number
    /// the bisection happened to land on: growth must be positive just below it and negative
    /// just above.
    #[test]
    fn the_break_even_cost_brackets_the_zero_of_net_growth() {
        let _torch_rng_guard = test_rng::exclusive();
        let fixture = Fixture::new("breakeven", |_, _| {});
        let beliefs = fixture.beliefs();
        let periods = schedule(&fixture.corpus, &fixture.panel, &beliefs, 2).expect("schedule");
        let marginal = fixture.marginal();
        let inputs = build_inputs(Construction::Stale, &beliefs, &periods, None, &marginal)
            .expect("stale inputs");

        // The oracle is the one policy guaranteed to have a positive gross edge on any panel,
        // so it is the one whose break-even must be a finite interior crossing.
        let metrics = measure(
            &fixture.panel,
            &periods,
            &inputs,
            Policy::Oracle,
            2,
            DEFAULT_GROSS_CAP,
            &FlatCost::new(0.0),
            1.0e7,
        )
        .expect("oracle metrics");
        let bps = metrics.break_even_cost_bps;
        assert!(
            bps > 0.0 && bps < MAX_BREAK_EVEN_BPS,
            "the oracle's break-even should be a finite interior crossing, got {bps}"
        );
        let growth_at = |cost: f64| -> f64 {
            let book = run_book(
                &fixture.panel,
                &periods,
                &inputs,
                Policy::Oracle,
                2,
                DEFAULT_GROSS_CAP,
                &FlatCost::new(cost as f32),
                1.0e7,
            )
            .expect("book at a trial cost");
            *book.log_equity.last().expect("a curve")
        };
        assert!(
            growth_at(bps * 0.98) > 0.0,
            "net growth is not positive just BELOW the reported break-even of {bps} bps"
        );
        assert!(
            growth_at(bps * 1.02) < 0.0,
            "net growth is not negative just ABOVE the reported break-even of {bps} bps"
        );
        // And a policy with no edge at all must report exactly zero rather than a bisection
        // artifact.
        let flat = HorizonInputs {
            construction: Construction::Stale,
            kelly: periods.iter().map(|p| vec![0.0; p.legs.len()]).collect(),
            pred_var: periods.iter().map(|p| vec![1.0e-5; p.legs.len()]).collect(),
            marginal_kelly: 0.0,
            laws: periods
                .iter()
                .map(|p| vec![AggregateLaw::default(); p.legs.len()])
                .collect(),
        };
        let idle = measure(
            &fixture.panel,
            &periods,
            &flat,
            Policy::Model,
            2,
            DEFAULT_GROSS_CAP,
            &FlatCost::new(0.0),
            1.0e7,
        )
        .expect("idle metrics");
        assert_eq!(
            idle.break_even_cost_bps, 0.0,
            "a book that never trades and never earns must break even at exactly 0 bps"
        );
    }

    /// A held position earns the AGGREGATE move, so the schedule's realized log return has to
    /// be the sum of the per-bar log returns over the window, and the payoff has to be
    /// `exp` of it minus one rather than a sum of per-bar simple returns.
    ///
    /// This is the arithmetic that makes a `k > 1` row mean anything, and getting it wrong is
    /// invisible: a sum of simple returns is within `O(k * sigma^2)` of the truth and would
    /// simply bias every long-horizon row.
    #[test]
    fn the_schedule_aggregates_the_held_move_multiplicatively() {
        let _torch_rng_guard = test_rng::exclusive();
        let fixture = Fixture::new("aggregate", |_, _| {});
        let beliefs = fixture.beliefs();
        for k in [1usize, 3, 8] {
            let periods = schedule(&fixture.corpus, &fixture.panel, &beliefs, k).expect("sched");
            for period in &periods {
                for leg in &period.legs {
                    let bars = fixture.corpus.bars(fixture.panel.series_of(leg.id));
                    let entry = fixture.panel.bar_index(period.instant, leg.slot) as usize;
                    let mut want = 0.0f64;
                    for step in 0..leg.steps {
                        let now = f64::from(bars[entry + step].close);
                        let previous = f64::from(bars[entry + step - 1].close);
                        want += (now / previous).ln();
                    }
                    assert!(
                        (leg.realized_log - want).abs() <= 1.0e-12 * want.abs().max(1.0e-6),
                        "k={k}: the held aggregate of {} over instant {} is {} but the sum of \
                         its {} per-bar log returns is {want}",
                        fixture.panel.symbols()[leg.id as usize],
                        period.instant,
                        leg.realized_log,
                        leg.steps
                    );
                    assert!(
                        leg.steps >= 1 && leg.steps <= k,
                        "k={k}: a leg held {} bars",
                        leg.steps
                    );
                }
            }
            // Non-overlapping, and covering the panel.
            let instants: usize = periods.iter().map(|_| k).sum();
            assert!(
                instants >= fixture.panel.instants(),
                "k={k}: {} periods of {k} bars do not cover {} instants",
                periods.len(),
                fixture.panel.instants()
            );
        }
    }

    /// The Rao-Blackwellized drift estimator has to actually be the promised variance
    /// reduction, because the whole sample budget is set from that claim.
    ///
    /// Two checks, one of which cannot be argued with: at `k = 1` the estimator is EXACT (the
    /// only term is the real belief's own conditional mean, so its Monte-Carlo error is zero
    /// by construction and must be reported as zero), and at `k > 1` its standard error must
    /// be materially below the plain sampled-aggregate estimator's on the same draws.
    #[test]
    fn the_rao_blackwellized_drift_beats_the_plain_estimator() {
        let _torch_rng_guard = test_rng::exclusive();
        let fixture = Fixture::new("variance", |_, _| {});
        let beliefs = fixture.beliefs();

        let one = schedule(&fixture.corpus, &fixture.panel, &beliefs, 1).expect("k=1 schedule");
        tch::manual_seed(19);
        let laws = horizon_laws(
            &fixture.model,
            &fixture.corpus,
            &fixture.panel,
            &beliefs,
            &one,
            RES,
            8,
        )
        .expect("k=1 laws");
        for (p, row) in laws.iter().enumerate() {
            for (l, law) in row.iter().enumerate() {
                assert_eq!(
                    law.mu_se, 0.0,
                    "the k=1 Rao-Blackwellized drift at ({p}, {l}) reports a Monte-Carlo error \
                     of {}, but its only term is the real belief's exact conditional mean",
                    law.mu_se
                );
                // Every sizing moment at H1 comes from the same fitted categorical law as the
                // cached one-bar forecast, not from the ancestral draws used at H>1.
                let t = one[p].instant;
                let slot = one[p].legs[l].slot;
                let want_log = beliefs.mu_log[t][slot];
                let want_mean = beliefs.mean_simple[t][slot];
                let want_second = beliefs.second_simple[t][slot];
                assert_eq!(law.mu_log, want_log);
                assert_eq!(law.var_log, beliefs.var_log[t][slot]);
                assert_eq!(law.mean_simple, want_mean);
                assert_eq!(law.second_simple, want_second);
                assert_eq!(law.prefix_mean_simple, [want_mean; FORECAST_HORIZONS.len()]);
                assert_eq!(
                    law.prefix_second_simple,
                    [want_second; FORECAST_HORIZONS.len()]
                );
            }
        }

        let four = schedule(&fixture.corpus, &fixture.panel, &beliefs, 4).expect("k=4 schedule");
        tch::manual_seed(19);
        let laws = horizon_laws(
            &fixture.model,
            &fixture.corpus,
            &fixture.panel,
            &beliefs,
            &four,
            RES,
            32,
        )
        .expect("k=4 laws");
        let mut rb = 0.0f64;
        let mut plain = 0.0f64;
        let mut legs = 0usize;
        for (p, row) in laws.iter().enumerate() {
            for (l, law) in row.iter().enumerate() {
                if four[p].legs[l].steps <= 1 {
                    continue;
                }
                rb += law.mu_se;
                plain += law.plain_mu_se;
                legs += 1;
            }
        }
        assert!(legs > 0, "no H>1 legs were sampled");
        let (rb, plain) = (rb / legs as f64, plain / legs as f64);
        assert!(
            plain > 0.0 && rb > 0.0,
            "both estimators must carry a real error bar at k > 1, got rb {rb} plain {plain}"
        );
        assert!(
            rb < 0.5 * plain,
            "the Rao-Blackwellized drift's mean standard error is {rb:.3e} against the plain \
             estimator's {plain:.3e}. The sample budget is sized on this reduction being \
             large, so if it is not, DEFAULT_SAMPLES is wrong."
        );
    }

    /// The horizon construction has to be measuring the `k`-BAR aggregate, and has to be a
    /// different measurement from holding a one-bar forecast. Both halves matter: if it is not
    /// accumulating `k` bars it is not the experiment, and if it does not differ from the
    /// control the sweep's central comparison is comparing a thing with itself.
    ///
    /// The sharp check on accumulation is the VARIANCE. Under any chain, the variance of a sum
    /// of `k` roughly-uncorrelated per-bar returns grows about linearly in `k`, so the sampled
    /// aggregate variance at `k = 4` must be several times the one-bar variance and land near
    /// `4x` it. A rollout that silently dropped its steps, or double-counted them, moves that
    /// ratio out of any plausible band.
    ///
    /// At `k = 1` there is no sampling approximation: the horizon construction reduces the
    /// fitted categorical law with its fitted within-bin `E[R]` and `E[R²]`. The ancestral
    /// first and raw-second moments begin at `k > 1`, where the aggregate has no closed form.
    #[test]
    fn the_horizon_construction_accumulates_k_bars_and_differs_from_holding_one() {
        let _torch_rng_guard = test_rng::exclusive();
        let fixture = Fixture::new("constructions", |_, _| {});
        let beliefs = fixture.beliefs();
        let marginal = fixture.marginal();
        let free = FlatCost::new(0.0);

        let sized = |periods: &[Period], laws: Option<&[Vec<AggregateLaw>]>, c: Construction| {
            let inputs = build_inputs(c, &beliefs, periods, laws, &marginal).expect("inputs");
            inputs.kelly.clone()
        };
        let mean_var = |laws: &[Vec<AggregateLaw>]| -> f64 {
            let (sum, count) = laws
                .iter()
                .flatten()
                .fold((0.0, 0usize), |(s, n), law| (s + law.var_log, n + 1));
            sum / count.max(1) as f64
        };

        // H1 is the exact fitted categorical reduction. In particular, it must not inherit the
        // within-bin and multinomial noise of the draws retained for H>1 diagnostics.
        let one = schedule(&fixture.corpus, &fixture.panel, &beliefs, 1).expect("k=1");
        tch::manual_seed(3);
        let one_laws = horizon_laws(
            &fixture.model,
            &fixture.corpus,
            &fixture.panel,
            &beliefs,
            &one,
            RES,
            256,
        )
        .expect("k=1 laws");
        for (p, laws) in one_laws.iter().enumerate() {
            for (l, law) in laws.iter().enumerate() {
                let t = one[p].instant;
                let slot = one[p].legs[l].slot;
                let mean = beliefs.mean_simple[t][slot];
                let second = beliefs.second_simple[t][slot];
                assert_eq!(law.var_log, beliefs.var_log[t][slot]);
                assert_eq!(law.mean_simple, mean);
                assert_eq!(law.second_simple, second);
            }
        }
        let h1_var = mean_var(&one_laws);
        assert!(h1_var > 0.0, "the fitted H1 variance must be positive");

        // The exact categorical control and the H1 horizon construction are sized from the
        // same fitted E[R] and E[R²]. The former caches the fraction in f32; the latter retains
        // the f64 moment reduction, so compare that cache at its own precision while pinning
        // the horizon arithmetic exactly.
        let categorical_one = build_inputs(
            Construction::StaleCategoricalMoments,
            &beliefs,
            &one,
            None,
            &marginal,
        )
        .expect("categorical H1 inputs");
        let horizon_one = build_inputs(
            Construction::Horizon,
            &beliefs,
            &one,
            Some(&one_laws),
            &marginal,
        )
        .expect("horizon H1 inputs");
        for (p, laws) in one_laws.iter().enumerate() {
            for (l, law) in laws.iter().enumerate() {
                let expected_var = (law.second_simple - law.mean_simple * law.mean_simple).max(0.0);
                assert_eq!(horizon_one.pred_var[p][l], expected_var);
                assert_eq!(
                    horizon_one.kelly[p][l], categorical_one.kelly[p][l],
                    "the H1 horizon and categorical constructions must share one constrained fraction"
                );
                assert!(
                    (categorical_one.pred_var[p][l] - expected_var).abs()
                        <= 1.0e-6 * expected_var.max(1.0e-12),
                    "the categorical f32 cache and f64 H1 variance disagree at ({p}, {l})"
                );
            }
        }

        let four = schedule(&fixture.corpus, &fixture.panel, &beliefs, 4).expect("k=4");
        tch::manual_seed(3);
        let four_laws = horizon_laws(
            &fixture.model,
            &fixture.corpus,
            &fixture.panel,
            &beliefs,
            &four,
            RES,
            256,
        )
        .expect("k=4 laws");
        // ACCUMULATION. Four bars of a roughly-uncorrelated chain carry about four times the
        // variance of one, so the ratio has to land in a band around 4. A rollout that ran one
        // step, or ran four but summed only the last, or double-counted the padded steps, all
        // land outside [2, 8].
        let sampled_four = mean_var(&four_laws);
        let growth = sampled_four / h1_var;
        assert!(
            (2.0..8.0).contains(&growth),
            "the k=4 aggregate variance is {growth:.3}x the exact k=1 figure (k=4 \
             {sampled_four:.3e}, k=1 {h1_var:.3e}). A four-bar sum should be near 4x, so the \
             rollout is not accumulating four bars."
        );
        // Later steps must contribute to the drift. Their signed conditional means need not
        // reinforce the first step — cancellation is valid — so monotone absolute drift is not
        // an invariant. Compare the paired aggregate laws directly instead.
        let (drift_gap, drift_count) = four_laws
            .iter()
            .flatten()
            .zip(one_laws.iter().flatten())
            .fold((0.0, 0usize), |(sum, count), (four, one)| {
                (sum + (four.mu_log - one.mu_log).abs(), count + 1)
            });
        let drift_gap = drift_gap / drift_count.max(1) as f64;
        assert!(
            drift_gap > 1e-7,
            "the k=4 and k=1 aggregate drifts differ by only {drift_gap:.3e} on average, so \
             the rollout's later steps are contributing nothing"
        );
        let stale = sized(&four, None, Construction::Stale);
        let horizon_inputs = build_inputs(
            Construction::Horizon,
            &beliefs,
            &four,
            Some(&four_laws),
            &marginal,
        )
        .unwrap();
        let horizon = horizon_inputs.kelly.clone();
        for (p, laws) in four_laws.iter().enumerate() {
            for (l, law) in laws.iter().enumerate() {
                let expected_kelly = if law.second_simple > 0.0 {
                    (law.mean_simple / law.second_simple).clamp(-FREE_LEVERAGE, FREE_LEVERAGE)
                } else {
                    0.0
                };
                let expected_var = (law.second_simple - law.mean_simple * law.mean_simple).max(0.0);
                assert!((horizon_inputs.kelly[p][l] - expected_kelly).abs() < 1e-12);
                assert!((horizon_inputs.pred_var[p][l] - expected_var).abs() < 1e-12);
            }
        }
        let differing = stale
            .iter()
            .flatten()
            .zip(horizon.iter().flatten())
            .filter(|(a, b)| (*a - *b).abs() > 1.0e-9 * a.abs().max(1.0))
            .count();
        assert!(
            differing > 0,
            "at k=4 the one-bar-and-hold control and the k-bar-aggregate experiment size \
             identically, so the sweep's central comparison is vacuous"
        );

        // And the books they produce are distinguishable, which is the thing the report claims.
        let of = |inputs: &HorizonInputs| {
            measure(
                &fixture.panel,
                &four,
                inputs,
                Policy::Model,
                4,
                DEFAULT_GROSS_CAP,
                &free,
                1.0e7,
            )
            .expect("metrics")
            .gross_log_growth_per_year
        };
        let a = of(&build_inputs(Construction::Stale, &beliefs, &four, None, &marginal).unwrap());
        let b = of(&horizon_inputs);
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() > 0.0,
            "the two constructions produce the same gross growth at k=4 ({a} vs {b})"
        );
    }
    fn receding_fixture() -> (Panel, Vec<Vec<ForecastMoment>>, Vec<Period>, RecedingConfig) {
        let slices = vec![
            PanelSlice {
                ts_ms: 1,
                symbols: vec![0],
                realized_r: vec![0.001],
            },
            PanelSlice {
                ts_ms: 2,
                symbols: vec![0],
                realized_r: vec![-0.0005],
            },
            PanelSlice {
                ts_ms: 3,
                symbols: vec![0],
                realized_r: vec![0.0007],
            },
        ];
        let panel = Panel::from_parts(
            vec!["A".to_owned()],
            slices,
            vec![vec![1.0e9], vec![1.0e9], vec![1.0e9]],
        )
        .unwrap();
        let moments = vec![
            vec![ForecastMoment {
                mean_simple: 0.003,
                second_simple: 0.01,
                frictionless_kelly: 0.3,
            }];
            panel.instants()
        ];
        let oracle = panel
            .slices()
            .iter()
            .enumerate()
            .map(|(instant, slice)| Period {
                instant,
                ts_ms: slice.ts_ms,
                legs: vec![Leg {
                    id: 0,
                    slot: 0,
                    row: instant as u32,
                    steps: 1,
                    realized_log: f64::from(slice.realized_r[0]),
                    adv_usd: 1.0e9,
                }],
            })
            .collect();
        let config = RecedingConfig {
            capital_usd: 1.0e6,
            constraints: KellyConstraints {
                gross_cap: 1.0,
                net_min: -1.0,
                net_max: 1.0,
                per_name_cap: 1.0,
                max_adv_participation: 1.0,
            },
            covariance_window: 2,
            covariance_shrinkage: 0.25,
        };
        (panel, moments, oracle, config)
    }

    #[test]
    fn receding_book_rejects_a_zero_decision_span() {
        let (panel, mut moments, mut oracle, config) = receding_fixture();
        moments.truncate(1);
        oracle.truncate(1);
        let err = run_receding_book(
            &panel,
            &moments,
            &oracle,
            ForecastMoment::default(),
            RecedingPolicy::Model,
            1,
            &FlatCost::new(0.0),
            config,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("at least two decisions at distinct timestamps"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn every_horizon_reforecasts_and_rebalances_on_every_calendar_row() {
        let (panel, moments, oracle, config) = receding_fixture();
        let marginal = ForecastMoment {
            mean_simple: 0.001,
            second_simple: 0.01,
            frictionless_kelly: 0.1,
        };
        for horizon in [1, 100] {
            let mut horizon_oracle = oracle.clone();
            for period in &mut horizon_oracle {
                for leg in &mut period.legs {
                    leg.steps = horizon;
                }
            }
            let run = run_receding_book(
                &panel,
                &moments,
                &horizon_oracle,
                marginal,
                RecedingPolicy::Model,
                horizon,
                &FlatCost::new(0.0),
                config,
            )
            .unwrap();
            assert_eq!(run.reforecasts, panel.instants());
            assert_eq!(run.log_equity.len(), panel.instants() + 1);
        }
    }

    #[test]
    fn oracle_and_model_use_the_same_cost_and_constraint_contract() {
        let (panel, moments, oracle, config) = receding_fixture();
        let marginal = ForecastMoment::default();
        for policy in [RecedingPolicy::Model, RecedingPolicy::Oracle] {
            let run = run_receding_book(
                &panel,
                &moments,
                &oracle,
                marginal,
                policy,
                1,
                &FlatCost::new(3.0),
                config,
            )
            .unwrap();
            assert!(run.max_gross <= config.constraints.gross_cap + 1e-3);
            assert!(run.max_abs_net <= config.constraints.net_max.abs() + 1e-3);
            assert!(run.max_name <= config.constraints.per_name_cap + 1e-3);
            assert!(run.max_participation <= config.constraints.max_adv_participation + 1e-12);
        }
    }

    #[test]
    fn zero_cost_h1_uses_the_declared_first_and_second_moments() {
        let (panel, moments, oracle, config) = receding_fixture();
        let run = run_receding_book(
            &panel,
            &moments,
            &oracle,
            ForecastMoment::default(),
            RecedingPolicy::Model,
            1,
            &FlatCost::new(0.0),
            config,
        )
        .unwrap();
        let realized = f64::from(panel.slices()[0].realized_r[0]).exp_m1();
        let expected_weight = moments[0][0].mean_simple / moments[0][0].second_simple;
        let expected = (1.0 + expected_weight * realized).ln();
        assert!(
            (run.log_equity[1] - expected).abs() <= 1e-8,
            "H1 chose a weight inconsistent with E[R] / E[R²]"
        );
    }

    #[test]
    fn deterministic_oracle_matches_the_same_moments_under_the_shared_execution_contract() {
        let (panel, mut moments, oracle, config) = receding_fixture();
        for (row, period) in moments.iter_mut().zip(&oracle) {
            for (moment, leg) in row.iter_mut().zip(&period.legs) {
                let realized = leg.realized_log.exp_m1();
                moment.mean_simple = realized;
                moment.second_simple = realized * realized;
            }
        }
        let cost = FlatCost::new(3.0);
        let model = run_receding_book(
            &panel,
            &moments,
            &oracle,
            ForecastMoment::default(),
            RecedingPolicy::Model,
            1,
            &cost,
            config,
        )
        .unwrap();
        let oracle_run = run_receding_book(
            &panel,
            &moments,
            &oracle,
            ForecastMoment::default(),
            RecedingPolicy::Oracle,
            1,
            &cost,
            config,
        )
        .unwrap();
        assert!(model
            .log_equity
            .iter()
            .zip(&oracle_run.log_equity)
            .all(|(a, b)| (a - b).abs() < 1e-12));
        assert!((model.turnover - oracle_run.turnover).abs() < 1e-12);
        assert!((model.execution_cost - oracle_run.execution_cost).abs() < 1e-12);
        assert_eq!(model.actions, oracle_run.actions);
    }

    #[test]
    fn missing_print_carries_the_holding_without_trade_and_books_the_reappearance_move() {
        let gap_return = 1.2f64.ln() as f32;
        let panel = Panel::from_parts(
            vec!["A".to_owned(), "B".to_owned()],
            vec![
                PanelSlice {
                    ts_ms: 1,
                    symbols: vec![0, 1],
                    realized_r: vec![0.0, 0.0],
                },
                PanelSlice {
                    ts_ms: 2,
                    symbols: vec![0],
                    realized_r: vec![0.0],
                },
                PanelSlice {
                    ts_ms: 3,
                    symbols: vec![0, 1],
                    realized_r: vec![0.0, gap_return],
                },
            ],
            vec![vec![1.0e9, 1.0e9], vec![1.0e9], vec![1.0e9, 1.0e9]],
        )
        .unwrap();
        let moments: Vec<Vec<ForecastMoment>> = panel
            .slices()
            .iter()
            .map(|slice| vec![ForecastMoment::default(); slice.symbols.len()])
            .collect();
        let periods: Vec<Period> = panel
            .slices()
            .iter()
            .enumerate()
            .map(|(instant, slice)| Period {
                instant,
                ts_ms: slice.ts_ms,
                legs: slice
                    .symbols
                    .iter()
                    .enumerate()
                    .map(|(slot, &id)| Leg {
                        id,
                        slot,
                        row: instant as u32,
                        steps: 1,
                        realized_log: f64::from(slice.realized_r[slot]),
                        adv_usd: 1.0e9,
                    })
                    .collect(),
            })
            .collect();
        let config = RecedingConfig {
            capital_usd: 1.0e6,
            constraints: KellyConstraints {
                gross_cap: 1.0,
                net_min: -1.0,
                net_max: 1.0,
                per_name_cap: 1.0,
                max_adv_participation: 1.0,
            },
            covariance_window: 2,
            covariance_shrinkage: 0.25,
        };
        let run = run_receding_book(
            &panel,
            &moments,
            &periods,
            ForecastMoment::default(),
            RecedingPolicy::BuyHold,
            1,
            &FlatCost::new(10.0),
            config,
        )
        .unwrap();
        assert_eq!(run.actions, 2, "the absent row fabricated a trade leg");
        assert!(
            (run.log_equity[2] - run.log_equity[1]).abs() < 1e-14,
            "the absent row fabricated payoff or cost"
        );
        let held_gap_name = 0.5 * run.turnover / (1.0 - run.execution_cost);
        let expected_catch_up = (1.0 + held_gap_name * 0.2).ln();
        assert!(
            ((run.log_equity[3] - run.log_equity[2]) - expected_catch_up).abs() < 1e-7,
            "the reappearance payoff did not book the full cumulative move exactly once"
        );
        assert!(run.max_gross <= config.constraints.gross_cap + 1e-9);
        assert!(run.max_abs_net <= config.constraints.net_max + 1e-9);
        assert!(run.max_name <= config.constraints.per_name_cap + 1e-9);
        assert!(run.max_participation <= config.constraints.max_adv_participation + 1e-12);
    }

    #[test]
    fn model_cannot_trade_a_catch_up_print_before_booking_its_realized_move() {
        let gap_return = 1.2f64.ln() as f32;
        let panel = Panel::from_parts(
            vec!["CLOCK".to_owned(), "GAPPED".to_owned()],
            vec![
                PanelSlice {
                    ts_ms: 1,
                    symbols: vec![0, 1],
                    realized_r: vec![0.0, 0.0],
                },
                PanelSlice {
                    ts_ms: 2,
                    symbols: vec![0],
                    realized_r: vec![0.0],
                },
                PanelSlice {
                    ts_ms: 3,
                    symbols: vec![0, 1],
                    realized_r: vec![0.0, gap_return],
                },
                PanelSlice {
                    ts_ms: 4,
                    symbols: vec![0, 1],
                    realized_r: vec![0.0, 0.0],
                },
            ],
            vec![
                vec![1.0e9, 1.0e9],
                vec![1.0e9],
                vec![1.0e9, 1.0e9],
                vec![1.0e9, 1.0e9],
            ],
        )
        .unwrap();
        assert_eq!(panel.elapsed_steps(2, 1), 2);
        assert_eq!(panel.elapsed_steps(3, 1), 1);

        let forecast = |mean_simple: f64| ForecastMoment {
            mean_simple,
            second_simple: 1.0,
            frictionless_kelly: mean_simple,
        };
        let moments = vec![
            vec![forecast(0.0), forecast(0.2)],
            vec![forecast(0.0)],
            vec![forecast(0.0), forecast(-0.2)],
            vec![forecast(0.0), forecast(-0.2)],
        ];
        let periods: Vec<Period> = panel
            .slices()
            .iter()
            .enumerate()
            .map(|(instant, slice)| Period {
                instant,
                ts_ms: slice.ts_ms,
                legs: slice
                    .symbols
                    .iter()
                    .enumerate()
                    .map(|(slot, &id)| Leg {
                        id,
                        slot,
                        row: instant as u32,
                        steps: 1,
                        realized_log: f64::from(slice.realized_r[slot]),
                        adv_usd: 1.0e9,
                    })
                    .collect(),
            })
            .collect();
        let config = RecedingConfig {
            capital_usd: 1.0e6,
            constraints: KellyConstraints {
                gross_cap: 1.0,
                net_min: -1.0,
                net_max: 1.0,
                per_name_cap: 1.0,
                max_adv_participation: 1.0,
            },
            covariance_window: 2,
            covariance_shrinkage: 0.25,
        };
        let cost_bps = 10.0f32;
        let run = run_receding_book(
            &panel,
            &moments,
            &periods,
            ForecastMoment::default(),
            RecedingPolicy::Model,
            1,
            &FlatCost::new(cost_bps),
            config,
        )
        .unwrap();

        assert_eq!(
            run.actions, 2,
            "only the opening trade and the next-consecutive-print reversal may execute"
        );
        assert!(
            (run.log_equity[2] - run.log_equity[1]).abs() < 1e-14,
            "the missing row fabricated payoff or cost"
        );
        let opening_cost = 1.0 - run.log_equity[1].exp();
        assert!(opening_cost > 0.0);
        let opening_target = opening_cost / (f64::from(cost_bps) * 1.0e-4);
        let held_before_gap = opening_target / (1.0 - opening_cost);
        let realized_gap = f64::from(gap_return).exp_m1();
        let expected_catch_up = (1.0 + held_before_gap * realized_gap).ln();
        assert!(
            ((run.log_equity[3] - run.log_equity[2]) - expected_catch_up).abs() < 1e-9,
            "the changed forecast captured or dodged the already-realized catch-up move"
        );
        assert!(
            run.log_equity[4] < run.log_equity[3] && run.execution_cost > opening_cost + 1e-12,
            "the symbol did not resume trading on its next consecutive print"
        );
    }
    #[test]
    fn locked_test_split_is_opt_in() {
        assert!(validate_receding_split(Split::Val, false).is_ok());
        assert!(validate_receding_split(Split::Test, false).is_err());
        assert!(validate_receding_split(Split::Test, true).is_ok());
        assert!(validate_receding_split(Split::Train, true).is_err());
    }
    #[test]
    fn receding_reports_persist_the_selected_run_and_keep_the_full_grid() {
        let _torch_rng_guard = test_rng::exclusive();
        let (panel, _, _, config) = receding_fixture();
        let instants = panel.instants();
        let shrinkage = config.covariance_shrinkage;
        let window = config.covariance_window;
        // Deliberately score a strict interior decision span so panel-span annualization would
        // disagree with the report and stdout contract this fixture protects.
        let first_decision = 1usize;
        let decision_span_years = (panel.slices()[instants - 1].ts_ms
            - panel.slices()[first_decision].ts_ms) as f64
            / (365.25 * 86_400_000.0);
        let runs: Vec<RecedingRun> = FORECAST_HORIZONS
            .iter()
            .flat_map(|&horizon| {
                RECEDING_POLICIES
                    .into_iter()
                    .map(move |policy| RecedingRun {
                        policy,
                        horizon,
                        reforecasts: instants - first_decision,
                        actions: 1,
                        log_equity: vec![0.0, 0.001 * horizon as f64, 0.003 * horizon as f64],
                        decision_instants: (first_decision..instants).collect(),
                        decision_span_years,
                        turnover: 0.1,
                        execution_cost: 1e-5,
                        max_gross: 0.2,
                        max_abs_net: 0.2,
                        max_name: 0.2,
                        max_participation: 0.001,
                        covariance_observations: 2,
                        covariance_shrinkage: shrinkage,
                        cost_month_substitutions: 0,
                        cost_cross_section_substitutions: 0,
                        covariance_window: window,
                        mean_factor_variance: horizon as f64 * 1e-5,
                    })
            })
            .collect();
        let h1_dir = scratch_dir("receding_reports_h1");
        let h100_dir = scratch_dir("receding_reports_h100");
        write_receding_reports(&h1_dir, "fixture", &panel, &runs, 1).unwrap();
        write_receding_reports(&h100_dir, "fixture", &panel, &runs, 100).unwrap();
        assert!(
            write_receding_reports(&h1_dir, "fixture", &panel, &runs, 2).is_err(),
            "a horizon outside the exact evaluated grid must be rejected"
        );

        for dir in [&h1_dir, &h100_dir] {
            for base in [RECEDING_KELLY_BASE, RECEDING_COVARIANCE_BASE] {
                assert!(shared::report::PRETRAIN_REPORT_BASES.contains(&base));
                let report = read_report(&dir.join(format!("{base}.report.bin"))).unwrap();
                let ReportKind::MultiLine { series } = report.kind else {
                    panic!("{base} must be a multiline report")
                };
                assert!(series
                    .iter()
                    .any(|row| row.values.iter().any(|v| v.is_finite())));
            }
        }

        let h1 = read_report(&h1_dir.join(format!("{RECEDING_KELLY_BASE}.report.bin"))).unwrap();
        let h100 =
            read_report(&h100_dir.join(format!("{RECEDING_KELLY_BASE}.report.bin"))).unwrap();
        assert!(h1.title.contains("SELECTED production H=1"));
        assert!(h100.title.contains("SELECTED production H=100"));
        let ReportKind::MultiLine { series: h1_series } = h1.kind else {
            panic!("H1 Kelly report must be multiline")
        };
        let summary = RecedingBench {
            runs: runs.clone(),
            split: Split::Val,
            instants,
            symbols: panel.symbols().len(),
            samples: 1,
            selected_horizon: 1,
            checkpoint: "fixture.ot".to_owned(),
            lineage_sha256: "fixture".to_owned(),
        };
        let ReportKind::MultiLine {
            series: h100_series,
        } = h100.kind
        else {
            panic!("H100 Kelly report must be multiline")
        };
        let expected_grid: Vec<f32> = FORECAST_HORIZONS.iter().map(|&h| h as f32).collect();
        for series in [&h1_series, &h100_series] {
            let grid = series
                .iter()
                .find(|row| row.label == "forecast horizon")
                .expect("the fixed comparison grid remains in every selected report");
            assert_eq!(grid.values, expected_grid);
        }

        let report_model_h1 = h1_series
            .iter()
            .find(|row| row.label == "model net log growth/year")
            .expect("model annual-growth report row")
            .values[0] as f64;
        let selected_run = summary
            .runs
            .iter()
            .find(|run| run.policy == RecedingPolicy::Model && run.horizon == 1)
            .expect("selected H1 model run");
        let stdout_growth = selected_run.recorded_annual_log_growth();
        assert!(
            (stdout_growth - report_model_h1).abs()
                <= f64::from(f32::EPSILON) * stdout_growth.abs().max(1.0),
            "stdout and report must annualize over the same decision span: stdout={stdout_growth}, \
             report={report_model_h1}"
        );
        let table = summary.table();
        let selected_line = table
            .lines()
            .find(|line| line.starts_with('*'))
            .expect("stdout marks the selected production row");
        assert!(
            selected_line.contains(&format!("{stdout_growth:>14.6}")),
            "stdout selected row must print the decision-span report value: {selected_line}"
        );

        let selected_h1 = h1_series
            .iter()
            .find(|row| row.label == "SELECTED production H=1 model net log growth/year")
            .expect("H1 selected production metric");
        let selected_h100 = h100_series
            .iter()
            .find(|row| row.label == "SELECTED production H=100 model net log growth/year")
            .expect("H100 selected production metric");
        let h1_finite: Vec<(usize, f32)> = selected_h1
            .values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .collect();
        let h100_finite: Vec<(usize, f32)> = selected_h100
            .values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .collect();
        assert_eq!(h1_finite.len(), 1);
        assert_eq!(h1_finite[0].0, 0);
        assert_eq!(h100_finite.len(), 1);
        assert_eq!(h100_finite[0].0, FORECAST_HORIZONS.len() - 1);
        assert_ne!(
            h1_finite[0].1, h100_finite[0].1,
            "choosing H1 versus H100 must select different production metrics"
        );

        std::fs::remove_dir_all(h1_dir).unwrap();
        std::fs::remove_dir_all(h100_dir).unwrap();
    }
}
