//! Does the predictor have exploitable DIRECTIONAL skill, measured with no trading policy
//! anywhere in the measurement?
//!
//! # Why this module exists, and what it is NOT
//!
//! [`super::trade_bench`] answers "what is the predictive law WORTH under log-optimal
//! sizing". Every number it reports is therefore conditioned on a policy: a Kelly solve, a
//! leverage cap, a cost charge, a rebalance schedule. That is the right question for
//! deployment and the wrong question for diagnosis, because a policy can hide a signal and a
//! signal can be flattered by a policy. This module removes the policy entirely. It reads two
//! numbers per bar out of the SAME marginalized predictive law the bench trades —
//! `mu_hat = E[r | strictly past bars]` and `sigma_hat = sqrt(Var[r | strictly past bars])` —
//! and scores them against the realized `r` with statistics that have no free parameters. No
//! Kelly solve, no cap, no cost model, no position sizing, nothing fitted.
//!
//! # The scoring rule has to survive the class imbalance, and raw accuracy does not
//!
//! The bench's `hit` field is `P(f* r > 0 | f* != 0)` ([`super::trade_bench::PolicyStats`]),
//! and because `sign(f*) = sign(E[R | past])` it already IS a directional score over every
//! bar — the policy does not contaminate it, since the model's `time in market` is 1.000.
//! What it lacks is a baseline. Buy & hold stakes `f = +1` on every bar, so ITS `hit` is
//! exactly `P(r > 0)` over ALL bars, which measured 0.451 — and that reads as a 45/55 up/down
//! split only if no bar is FLAT. Bars are flat: `r` is exactly zero on 22,010 of 229,376 pinned
//! val bars, 9.6%, so the all-bar split is 45.1% up, 45.3% down, 9.6% unmoved, and the majority
//! class leads by two tenths of a point rather than by ten.
//!
//! TWO DENOMINATORS, never to be quoted against each other. Every 2x2 statistic below EXCLUDES
//! flats, because a bar that did not move has no direction to be right about, and on that
//! CLASSIFIED denominator the up rate is 0.504 and the best constant predictor scores 0.504
//! against the model's 0.537. On the ALL-BAR denominator the model's `hit`-style figure is 0.486
//! and always-down is 0.453. An earlier reading of this paragraph — "always down scores 0.549 and
//! beats the model's 0.486 outright" — compared an all-bar baseline against a classified score;
//! it is wrong, and the imbalance it invoked lives almost entirely in the excluded flats. Raw
//! accuracy is still not the headline, because whatever imbalance a draw does carry is a property
//! of the draw and not of the model, so [`ConfusionReport`] reports the full 2x2 with per-class
//! accuracy, BALANCED accuracy, both precisions, the flat count, and all three constant baselines
//! on the same axis, each labelled with the denominator it lives on, so the comparison cannot be
//! made against the wrong one.
//!
//! # The information coefficient is decomposed because the pooled one is not credible
//!
//! A pooled Mincer-Zarnowitz `R^2` of 0.062-0.069 on 5-minute equity returns
//! ([`super::trade_bench::MeanCalibration`]) implies `|corr(mu_hat, r)| ~ 0.26`, which is far
//! above anything this literature reports for a single-name return forecast. Before that is
//! believed it has to be split, because a POOLED correlation over thousands of names of
//! wildly different volatility is not a directional statistic:
//!
//! * `corr = E[mu r] / (sd(mu) sd(r))`, and writing `mu = sign_mu |mu|`, `r = sign_r |r|`
//!   gives `E[mu r] = E[sign_mu sign_r |mu| |r|]`. A model that gets the SIGN right only on
//!   the bars where `|mu| |r|` is large — which is exactly the model whose gross growth is
//!   positive at a sub-50% hit rate — posts a positive pooled `corr`. That is real, and it is
//!   a magnitude story, not a direction story.
//! * The same product is inflated by any co-movement of `|mu_hat|` with `|r|`, i.e. by
//!   volatility prediction: telling a volatile name from a quiet one raises `E[|mu||r|]`
//!   without contributing a single correct sign. [`MagnitudeIc`] measures that channel
//!   directly, `corr(|mu_hat|, |r|)` and `corr(sigma_hat, |r|)`, so it can be attributed
//!   rather than argued about.
//!
//! [`IcDecomposition`] therefore reports three numbers, and their DISAGREEMENT is the
//! finding:
//!
//! 1. POOLED, Pearson and Spearman, which is the number the `R^2` implies.
//! 2. WITHIN-NAME: one IC per symbol, then the distribution over symbols. This removes every
//!    cross-sectional scale effect by construction, because no term of any symbol's
//!    correlation involves another symbol.
//! 3. STANDARDIZED: `mu_hat` and `r` z-scored INSIDE each `(symbol, calendar month)` block
//!    and then pooled. This is algebraically the bar-count-weighted mean of the within-block
//!    correlations, so between-block heteroskedasticity cannot contribute to it at all, while
//!    it keeps the full sample instead of averaging thousands of noisy per-symbol estimates.
//!
//! If POOLED is much larger than WITHIN-NAME and STANDARDIZED, the apparent skill was
//! cross-sectional volatility prediction and not direction. [`SkillProfile::verdict`] says
//! which, in those words, without hedging.
//!
//! # The deliverable is the confidence curve
//!
//! [`ConfidenceCurve`] buckets bars by decile of `|mu_hat|`, and separately by decile of the
//! model's own predicted Sharpe `|mu_hat| / sigma_hat` — the better-motivated selector,
//! because it is scale-free and is the quantity a selective policy would actually rank on.
//! Per decile: directional accuracy, balanced accuracy, mean realized `sign(mu_hat) r`, IC,
//! and the bar count.
//!
//! Ten marginal intervals do not answer "is it monotone", so the decisive statistic is the
//! TOP-MINUS-BOTTOM difference, resampled PAIRED inside each block
//! ([`ConfidenceCurve::top_minus_bottom_accuracy`] and its siblings). A paired difference is
//! resolvable at a fraction of the width of two marginal intervals, and it is the only form
//! in which "skill concentrates in high-confidence bars" is a testable claim.
//!
//! # Every interval is blocked, because the binomial one is a fiction here
//!
//! 229,376 bars give a hit rate a binomial standard error of 0.001. Those bars sit inside 256
//! windows inside 256 `(symbol, calendar month)` blocks, and bars in one block share a
//! regime, a level of volatility and a market-common return. The interval is therefore a
//! nonparametric bootstrap over BLOCKS, using [`super::pretrain_stats::block_bootstrap`]'s
//! scheme deliberately down to the RNG — the same `ChaCha12Rng` seeded with the same
//! [`BOOTSTRAP_SEED`], the same [`BOOTSTRAP_DRAWS`], blocks visited in `BTreeMap` order — so
//! a draw index sequence here is literally the same sequence of blocks every other interval
//! this campaign reports was taken over.
//!
//! Every statistic in this module is a smooth function of BLOCK-ADDITIVE sufficient
//! statistics ([`Cell`], [`Moments`], [`Placements`], [`Selective`]), so a bootstrap draw is
//! one pass over ~256 accumulators rather than over 229,376 bars, and the refit is exact
//! rather than linearized.
//!
//! # No lookahead, structurally
//!
//! The traded factor HEADS the emission chain (`r -> s -> ...`), so the head's `r` row is
//! `p(r | strictly past bars)` outright: it conditions on no same-bar factor and there is
//! nothing to marginalize. This module consumes [`WindowPaths::predicted_mean`] and
//! [`WindowPaths::predicted_var`], which [`super::trade_bench::window_paths`] takes from
//! [`super::trade_bench::forecast_r_probs`] — that prefix-free row, the same law
//! [`super::growth::r_moments`] takes the objective's mean from. There is no parameter
//! anywhere in this module through which a realized same-bar `s` could arrive, and
//! `permuting_the_realized_same_bar_s_leaves_every_skill_statistic_bit_identical` asserts the
//! value: every reported scalar is BIT-identical under an arbitrary reassignment of the
//! realized `s`, compared on raw `f64` bit patterns rather than on formatted output, and the
//! same test proves non-vacuity by building a panel from a row that DOES carry the realized
//! `s` in its prefix and showing it moves the 2x2 and the headline IC.
//!
//! # What this module deliberately does not do
//!
//! It builds no policy and runs no backtest. [`SelectiveTable`] is arithmetic on the bars:
//! participation, mean edge per traded bar, turnover per traded bar, and their ratio as a
//! break-even cost. That is a SCREENING calculation, whose only purpose is to say whether a
//! selective policy could plausibly clear the measured cost of trading before anyone spends a
//! run finding out. It is not an estimate of what such a policy would earn: it charges no
//! cost, models no queue, ignores the borrow and the halt, and uses decile cutpoints that a
//! deployed rule would have to set out of sample.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{anyhow, bail, ensure, Context, Result};
use rand::seq::IndexedRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use tch::Device;

use shared::report::{read_report, write_report, Report, ReportKind, ReportSeries, ScaleKind};

use crate::torch::bar_dist::{BarScoring, BAR_CHAIN, DOF_R};
use crate::torch::cuda::cfg::configure_cuda;
use crate::torch::dataset::Split;
use crate::torch::world_model::{world_model_metadata_path, BarWorldModel};

use super::pretrain::{
    configure_threads, evaluate, load_corpus, pinned_blocks, CorpusFlags, PinnedSet,
    EVAL_WINDOW_SEED,
};
use super::pretrain_stats::{Dispersion, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, CI_MASS};
use super::trade_bench::{WindowPaths, MAX_BREAK_EVEN_BPS, TRADE_WINDOWS};

/// This module scores `p(r | past)`, which is the head's own `r` row only because `r` heads
/// the chain and therefore has no prefix. A reorder that puts any factor before `r` makes
/// [`WindowPaths::predicted_mean`] a teacher-forced object and every statement above about
/// lookahead false, so it is a compile error rather than a silently reinterpreted number.
const _: () = assert!(
    BAR_CHAIN[0] == DOF_R,
    "the skill audit scores p(r|past) read straight off the prefix-free head of the chain"
);

/// Chart base of the confidence curve and the selective screening table.
pub const SKILL_PROFILE_BASE: &str = "pretrain_skill_profile";

/// Confidence buckets. Ten because the question is "does skill concentrate", which needs
/// enough buckets to see a shape and enough bars per bucket that every block contributes to
/// every bucket: at 229k bars over 256 blocks a decile holds ~23k bars and ~90 per block, so
/// the top-minus-bottom difference is genuinely paired inside a block rather than a
/// comparison between blocks that happened to fall in different buckets.
pub const DECILES: usize = 10;

/// MATCHED, MEASURED one-way cost of the bars this module actually scores, in basis points.
///
/// PROVENANCE, and this is the reference line the verdict is stated against. `super::portfolio_cost`
/// priced the traded window list itself — the 256 windows over 256 distinct symbols at the fixed
/// 896-bar diagnostic context that the mean-calibration experiment and this audit both measure —
/// against the real 5,297-symbol cost calibration on the pinned validation span. All 256 matched
/// the corpus, 0 were impact-unpriceable, 6 fell back to a non-primary spread estimator. The
/// figure is half-spread (half the gated Roll estimate, i.e. crossing once) plus per-share
/// commission converted to bps plus regulatory fees: entirely measured, no free parameter, NO
/// impact model, so a conclusion resting on it survives the impact coefficient being wrong by any
/// factor.
///
/// EQUAL-WEIGHTED MEAN over the 256 names, `10.620` bps, not the `7.230` bps median. Each name
/// contributes exactly one 896-bar window, so a bar-pooled break-even is dimensionally a
/// bar-weighted average and every name enters it with identical weight; cost is heavily
/// right-skewed, so the median of an equal-weighted book understates what that book pays.
///
/// It supersedes two figures quoted earlier in this campaign, both of which are wrong here for
/// the same reason — they were universe statistics compared against a break-even measured on a
/// specific 256-symbol-month draw. `10.99` bps was retracted outright after two correctness
/// fixes in the cost path. `4.150` bps is a correct number about the wrong population: it is
/// the DEEPEST-decile median of the whole universe, while the traded draw occupies all ten
/// deciles — occupancy `[8, 24, 18, 18, 24, 27, 29, 38, 27, 43]` from thinnest to deepest — so
/// it understated the matched cost by 2.6x. Neither may be quoted against a break-even again.
pub const MEASURED_COST_BPS_MATCHED: f64 = 10.620;

/// The matched median rather than the mean, `7.230` bps. Carried so the right-skew that makes
/// [`MEASURED_COST_BPS_MATCHED`] the correct statistic is visible in the report instead of being
/// an assertion in a doc comment, and so a reader who wants the typical NAME rather than the
/// book's cost can see both.
pub const MEASURED_COST_BPS_MATCHED_MEDIAN: f64 = 7.230;

/// The same matched cost with the sized impact term added at 1% of ADV participation and
/// `k = 0.5`: `26.351` bps span-pooled, quoted here as the secondary line.
///
/// Never the reference. Roughly 16 of those bps is modelled impact at a literature default
/// nobody fitted to this corpus, and the whole point of judging against the impact-free figure
/// is that the verdict cannot be argued away by disputing `k`. Anchor-month pricing comes in
/// 1.19 bps CHEAPER at 25.165 bps, so the sign of the conclusion does not depend on the pooling
/// either.
pub const SIZED_COST_BPS_MATCHED_AT_1PCT_ADV: f64 = 26.351;

/// The universe-wide equal-weighted measured cost, `12.325` bps impact-free. Carried only as
/// the reference point for how ATYPICAL the traded draw is — it is 1.7 bps more expensive than
/// the matched draw, so the pinned windows are mildly liquidity-favoured but not a mega-cap
/// subset.
pub const MEASURED_COST_BPS_UNIVERSE: f64 = 12.325;

/// The same three costs are the reference line of [`super::horizon`]'s frontier, declared there
/// under its own names. This campaign has already retracted `10.99` and mis-applied `4.150`, so
/// these literals demonstrably MOVE, and two modules quoting the same measurement from two
/// places is how one of them goes stale while its provenance paragraph still reads as
/// authoritative. Tied at COMPILE TIME rather than by a runtime test, so the next revision of
/// the cost measurement cannot land in one module alone.
const _: () = assert!(
    MEASURED_COST_BPS_MATCHED == super::horizon::MATCHED_MEASURED_BPS,
    "the matched measured cost disagrees with the horizon sweep's reference line"
);
const _: () = assert!(
    SIZED_COST_BPS_MATCHED_AT_1PCT_ADV == super::horizon::MATCHED_ALL_IN_BPS,
    "the matched all-in cost disagrees with the horizon sweep's"
);
const _: () = assert!(
    MEASURED_COST_BPS_UNIVERSE == super::horizon::UNIVERSE_MEASURED_BPS,
    "the universe measured cost disagrees with the horizon sweep's"
);

// ---------------------------------------------------------------------------
// The panel: two predictions and one outcome per bar, plus its grouping
// ---------------------------------------------------------------------------

/// One bar's prediction and outcome, all in LOG-return space.
#[derive(Clone, Copy, Debug)]
pub struct SkillBar {
    /// `E[r | strictly past bars]`, from the head's prefix-free `r` row.
    pub mu: f64,
    /// `sqrt(Var[r | strictly past bars])`, from the same law.
    pub sigma: f64,
    /// The realized `r` the prediction is scored against.
    pub r: f64,
    /// The bench's UNCAPPED log-optimal fraction for this bar.
    ///
    /// Carried for exactly one purpose: reconciling this module's `sign(mu_hat)` accuracy
    /// with [`super::trade_bench::PolicyStats::hit_rate`], which scores `sign(f*)`. The two
    /// are not identical. `sign(f*) = sign(E[R])` while `sign(mu_hat) = sign(E[r])`, and
    /// `E[R] = E[e^r] - 1` carries a Jensen term of `Var[r]/2 ~ 4e-6` against a `|mu_hat|` of
    /// order `1e-4`, so a few percent of bars near zero disagree. The disagreement rate is
    /// reported rather than assumed away, because it is the whole difference between this
    /// module's headline accuracy and the bench's `hit`. Nothing else reads it.
    pub free: f64,
}

/// One window's bars, with the symbol it belongs to and its bootstrap block.
#[derive(Clone, Debug)]
pub struct SkillWindow {
    pub bars: Vec<SkillBar>,
    /// Corpus symbol index. The resampling unit of the WITHIN-NAME distribution.
    pub symbol: u32,
    /// `(symbol, calendar month)` id from [`pinned_blocks`]. The resampling unit of every
    /// other interval, and the standardization unit of the z-scored IC.
    pub block: u64,
}

/// The whole scored panel, retaining the window and block structure the intervals need.
#[derive(Clone, Debug)]
pub struct SkillPanel {
    /// One entry per scored window, in pinned order.
    pub windows: Vec<SkillWindow>,
}

impl SkillPanel {
    /// Build the panel from the bench's own per-window paths.
    ///
    /// Deliberately consumes [`WindowPaths`] rather than re-running the head: the moments it
    /// carries came out of [`super::trade_bench::forecast_r_probs`] on the same pass that
    /// produced the bench's positions, so this module and the bench are provably scoring one
    /// predictive law rather than two constructions of it.
    pub fn from_paths(windows: &[WindowPaths], symbols: &[u32], blocks: &[u64]) -> Result<Self> {
        ensure!(
            windows.len() == symbols.len() && windows.len() == blocks.len(),
            "every window needs a symbol and a block: {} windows, {} symbols, {} blocks",
            windows.len(),
            symbols.len(),
            blocks.len()
        );
        ensure!(!windows.is_empty(), "the skill audit was handed no windows");
        let built = windows
            .iter()
            .zip(symbols)
            .zip(blocks)
            .map(|((window, symbol), block)| {
                ensure!(
                    window.has_moments(),
                    "a scored window carries no conditional moments, so its bars cannot be \
                     scored for directional skill; the evaluation pass must run with the \
                     trading bench enabled"
                );
                let log_r = window.realized_log();
                let bars = (0..window.bars())
                    .map(|bar| SkillBar {
                        mu: window.predicted_mean[bar],
                        sigma: window.predicted_var[bar].max(0.0).sqrt(),
                        r: log_r[bar],
                        free: window.free[bar],
                    })
                    .collect();
                Ok(SkillWindow {
                    bars,
                    symbol: *symbol,
                    block: *block,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { windows: built })
    }

    pub fn bars(&self) -> usize {
        self.windows.iter().map(|window| window.bars.len()).sum()
    }

    fn symbol_count(&self) -> usize {
        self.windows
            .iter()
            .map(|window| window.symbol)
            .collect::<BTreeSet<_>>()
            .len()
    }

    fn block_count(&self) -> usize {
        self.windows
            .iter()
            .map(|window| window.block)
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// Every bar, flattened, paired with the window it came from.
    ///
    /// Deterministic in the panel's own order, which is what lets two passes over it be
    /// zipped by index (the pooled Spearman does exactly that).
    fn flat(&self) -> impl Iterator<Item = (&SkillWindow, &SkillBar)> {
        self.windows
            .iter()
            .flat_map(|window| window.bars.iter().map(move |bar| (window, bar)))
    }

    /// Per-block accumulators of `S`, in `BTreeMap` order.
    ///
    /// The order is the interval's identity: [`blocked`] resamples by index into the returned
    /// vector, so two statistics built through this function draw the same blocks in the same
    /// sequence and their intervals are comparable rather than merely similarly constructed.
    fn per_block<S: BlockSums>(&self, mut push: impl FnMut(&mut S, &SkillBar)) -> Vec<S> {
        let mut grouped: BTreeMap<u64, S> = BTreeMap::new();
        for (window, bar) in self.flat() {
            push(grouped.entry(window.block).or_default(), bar);
        }
        grouped.into_values().collect()
    }
}

// ---------------------------------------------------------------------------
// The block bootstrap, over sufficient statistics rather than over bars
// ---------------------------------------------------------------------------

/// A per-block accumulator a reported statistic is a function of.
///
/// Additivity is the whole requirement: a bootstrap draw is a sum of block accumulators, so
/// any statistic expressible as a function of one is refittable EXACTLY in one pass over the
/// blocks. Nothing here is linearized and no delta-method variance appears anywhere.
trait BlockSums: Copy + Default {
    fn absorb(&mut self, other: &Self);
    /// Observations behind the accumulator, reported as `Dispersion::samples`.
    fn count(&self) -> f64;
}

/// Resample BLOCKS with replacement and refit `stat` on each draw.
///
/// The scheme is [`super::pretrain_stats::block_bootstrap`]'s: the same `ChaCha12Rng` seeded
/// with the same [`BOOTSTRAP_SEED`], [`BOOTSTRAP_DRAWS`] draws of `blocks.len()` blocks,
/// percentiles of the draws at [`CI_MASS`], `se` their standard deviation. Because the seed
/// is fixed and the block order is fixed, every statistic in this module is intervalled over
/// the SAME sequence of resampled block sets. That is what makes two of them comparable, and
/// it is what makes a paired difference (see
/// [`ConfidenceCurve::top_minus_bottom_accuracy`]) an honest interval on the difference
/// rather than a combination of two marginal ones.
///
/// Draws whose statistic is not finite are dropped rather than counted as zero: a resample
/// that happens to contain no down-bar has no down-bar accuracy, and imputing one would
/// narrow the interval with fabricated data.
fn blocked<S: BlockSums>(blocks: &[S], stat: impl Fn(&S) -> f64) -> Dispersion {
    let mut pooled = S::default();
    for block in blocks {
        pooled.absorb(block);
    }
    let mut out = Dispersion {
        mean: stat(&pooled),
        se: f64::NAN,
        ci_low: f64::NAN,
        ci_high: f64::NAN,
        blocks: blocks.len(),
        samples: pooled.count() as usize,
    };
    if blocks.len() < 2 {
        // One block is one observation. A zero-width interval reported as precision is the
        // failure this refuses to commit.
        return out;
    }
    let mut rng = ChaCha12Rng::seed_from_u64(BOOTSTRAP_SEED);
    let mut draws: Vec<f64> = Vec::with_capacity(BOOTSTRAP_DRAWS);
    for _ in 0..BOOTSTRAP_DRAWS {
        let mut draw = S::default();
        for _ in 0..blocks.len() {
            draw.absorb(blocks.choose(&mut rng).expect("blocks is non-empty"));
        }
        let value = stat(&draw);
        if value.is_finite() {
            draws.push(value);
        }
    }
    if draws.len() < 2 {
        return out;
    }
    draws.sort_by(f64::total_cmp);
    out.se = standard_deviation(&draws);
    let tail = (1.0 - CI_MASS) / 2.0;
    out.ci_low = sorted_percentile(&draws, tail);
    out.ci_high = sorted_percentile(&draws, 1.0 - tail);
    out
}

fn standard_deviation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

/// Linear-interpolated percentile of an ascending slice, the convention every interval in
/// this repository is reported under.
fn sorted_percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let position = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    let weight = position - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

/// `numerator / denominator`, NaN on an empty denominator.
///
/// A rate over zero observations is UNMEASURED, and returning zero for it would let "we saw
/// no bars" render identically to "we saw bars and none of them hit".
fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        f64::NAN
    }
}

/// A break-even figure clipped for the chart's y-axis, with NON-FINITE PRESERVED.
///
/// `f64::min` IGNORES NaN — `f64::NAN.min(1000.0)` is `1000.0` — so a bare
/// `value.min(MAX_BREAK_EVEN_BPS)` renders an UNMEASURED break-even as the clip constant, here
/// 1000 bps beside a 10.620 bps reference line on the same panel. That is precisely the
/// confusion `MAX_BREAK_EVEN_BPS` was introduced to prevent, and it is reachable: a threshold
/// that trades no bars divides zero by zero, and a panel with fewer than two blocks leaves
/// both interval ends non-finite. The sibling series on the same row (participation, edge,
/// turnover, hit rate) are unclipped and therefore already render NaN, so preserving it here is
/// also what makes one row internally consistent.
/// Note the polarity, which is correct HERE and would be wrong elsewhere: this maps a
/// non-finite input - including `+INFINITY` - to NaN, i.e. to "not measured". That is right for
/// this module, where a non-finite break-even means the turnover denominator was zero and nothing
/// was measured. It is WRONG for [`super::trade_bench`]'s break-even solver, where `+INFINITY` is
/// produced deliberately to mean "the edge survives past `MAX_BREAK_EVEN_BPS`" and is therefore a
/// MEASURED lower bound. Converting a measured bound into an unmeasured value is the same rule
/// pointed the other way, so this helper must not be reused there.
/// A geometry threshold as text: the number, or the words saying it was never supplied.
fn show_option(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{value:.2}"),
        None => "NOT MEASURED (no support geometry)".to_owned(),
    }
}

/// A fraction as text, with NaN rendering as the third state rather than as a number.
fn show_fraction(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.4}")
    } else {
        "unmeasured".to_owned()
    }
}

fn clamp_break_even(value: f64) -> f64 {
    if value.is_finite() {
        value.min(MAX_BREAK_EVEN_BPS)
    } else {
        f64::NAN
    }
}

/// A [`Dispersion`] with its RESAMPLED-UNIT COUNT LABELLED.
///
/// `Dispersion`'s own `Display` hardcodes the word "windows" for `samples`, and in this module
/// `samples` is three different things: bars for every pooled statistic, CLASSIFIED bars for the
/// AUC, and SYMBOLS for the within-name distribution. Printing "229376 windows" beside a
/// 256-window panel is not a cosmetic slip — it misstates the sample size of every interval in
/// the report by three orders of magnitude — so the unit is passed in and stated.
fn show(value: &Dispersion, unit: &str) -> String {
    format!(
        "{:.4} +/- {:.4} (95% CI {:.4}..{:.4}, {} blocks / {} {unit})",
        value.mean, value.se, value.ci_low, value.ci_high, value.blocks, value.samples
    )
}

/// The direction a signed prediction or outcome points, `0` for neither.
///
/// Written once because `f64::signum` returns `+1.0` for `+0.0` and `-1.0` for `-0.0`, so
/// every "is this up or down" test in this module would otherwise have to remember to
/// exclude zero, and one that forgot would score a flat bar as a confident call.
fn direction(value: f64) -> i8 {
    if !value.is_finite() || value == 0.0 {
        0
    } else if value > 0.0 {
        1
    } else {
        -1
    }
}

// ---------------------------------------------------------------------------
// Sufficient statistics
// ---------------------------------------------------------------------------

/// Product moments of one `(x, y)` pair stream, enough for a Pearson correlation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Moments {
    n: f64,
    x: f64,
    y: f64,
    xx: f64,
    yy: f64,
    xy: f64,
}

impl Moments {
    fn push(&mut self, x: f64, y: f64) {
        if !x.is_finite() || !y.is_finite() {
            return;
        }
        self.n += 1.0;
        self.x += x;
        self.y += y;
        self.xx += x * x;
        self.yy += y * y;
        self.xy += x * y;
    }

    fn absorb(&mut self, other: &Self) {
        self.n += other.n;
        self.x += other.x;
        self.y += other.y;
        self.xx += other.xx;
        self.yy += other.yy;
        self.xy += other.xy;
    }

    /// Pearson `corr(x, y)`, NaN when either side does not vary.
    ///
    /// Written on the scaled cross-products `n Sxy - Sx Sy` rather than on centered sums so
    /// the whole statistic is a function of the six additive numbers above and a bootstrap
    /// Mean of the first argument, for the places a `Moments` is fed one quantity twice purely
    /// to accumulate its mean.
    fn mean_x(&self) -> f64 {
        ratio(self.x, self.n)
    }

    /// refit needs no second pass over the bars.
    fn corr(&self) -> f64 {
        if self.n < 2.0 {
            return f64::NAN;
        }
        let sxx = self.n * self.xx - self.x * self.x;
        let syy = self.n * self.yy - self.y * self.y;
        let sxy = self.n * self.xy - self.x * self.y;
        if !(sxx > 0.0) || !(syy > 0.0) {
            return f64::NAN;
        }
        sxy / (sxx * syy).sqrt()
    }
}

impl BlockSums for Moments {
    fn absorb(&mut self, other: &Self) {
        Moments::absorb(self, other);
    }
    fn count(&self) -> f64 {
        self.n
    }
}

const PRED_UP_REAL_UP: usize = 0;
const PRED_UP_REAL_DOWN: usize = 1;
const PRED_DOWN_REAL_UP: usize = 2;
const PRED_DOWN_REAL_DOWN: usize = 3;

/// The 2x2 counts and the sums every direction statistic of one bucket of bars is a function
/// of.
///
/// FLAT bars (`r == 0`, a bar whose close matched the previous one) are excluded from the 2x2
/// rather than scored as errors, because "wrong" and "there was nothing to be right about"
/// are different facts. They are counted, and [`Cell::up_rate_all_bars`] keeps them in the
/// denominator so the up-rate this reports is directly the quantity buy & hold's own `hit`
/// field measures.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Cell {
    /// `[pred up & up, pred up & down, pred down & up, pred down & down]`.
    conf: [f64; 4],
    /// Bars with `r == 0`, excluded from `conf`.
    flat: f64,
    /// Bars with `mu_hat == 0` or non-finite, excluded from `conf`: no direction expressed.
    undirected: f64,
    /// Every bar assigned to this cell.
    all: f64,
    /// Bars with `r > 0`, over the SAME `all` denominator.
    up_all: f64,
    /// Sum of `sign(mu_hat) r` over bars with a direction, in nats.
    edge: f64,
    /// Bars contributing to `edge`.
    edge_n: f64,
    /// `mu_hat` against `r`.
    m: Moments,
    /// Bars whose `|mu_hat|` exceeds the interior bound, i.e. whose MEAN is provably constituted by
    /// catch-all mass. NaN bound leaves these at zero and the reported fraction is NaN, never 0.0.
    over_bound_mu: f64,
    /// The same for `sigma_hat`.
    over_bound_sigma: f64,
    /// Bars above the marginal-interior RMS REFERENCE, which proves atypicality and not impossibility.
    over_reference_sigma: f64,
}

impl Cell {
    fn push_geometry(&mut self, bar: &SkillBar, bound_bps: f64, reference_bps: f64) {
        // A NaN threshold makes every comparison false, which would read as "no bar exceeded it".
        // The COUNTS therefore mean nothing on their own; `over_bound_mu_fraction` below returns NaN
        // unless the threshold was finite, so an absent support can only ever print as unmeasured.
        let mu_bps = 1.0e4 * bar.mu.abs();
        let sigma_bps = 1.0e4 * bar.sigma;
        if mu_bps > bound_bps {
            self.over_bound_mu += 1.0;
        }
        if sigma_bps > bound_bps {
            self.over_bound_sigma += 1.0;
        }
        if sigma_bps > reference_bps {
            self.over_reference_sigma += 1.0;
        }
    }

    fn push(&mut self, bar: &SkillBar) {
        self.all += 1.0;
        if bar.r > 0.0 {
            self.up_all += 1.0;
        }
        self.m.push(bar.mu, bar.r);
        let predicted = direction(bar.mu);
        if predicted == 0 {
            self.undirected += 1.0;
            return;
        }
        self.edge += f64::from(predicted) * bar.r;
        self.edge_n += 1.0;
        let realized = direction(bar.r);
        if realized == 0 {
            self.flat += 1.0;
            return;
        }
        let slot = match (predicted > 0, realized > 0) {
            (true, true) => PRED_UP_REAL_UP,
            (true, false) => PRED_UP_REAL_DOWN,
            (false, true) => PRED_DOWN_REAL_UP,
            (false, false) => PRED_DOWN_REAL_DOWN,
        };
        self.conf[slot] += 1.0;
    }

    /// Bars inside the 2x2: directed prediction, non-flat outcome.
    fn classified(&self) -> f64 {
        self.conf.iter().sum()
    }

    fn accuracy(&self) -> f64 {
        ratio(
            self.conf[PRED_UP_REAL_UP] + self.conf[PRED_DOWN_REAL_DOWN],
            self.classified(),
        )
    }

    /// `P(predict up | realized up)`: accuracy restricted to up-bars, which is also the
    /// recall of the up class.
    fn accuracy_on_up(&self) -> f64 {
        ratio(
            self.conf[PRED_UP_REAL_UP],
            self.conf[PRED_UP_REAL_UP] + self.conf[PRED_DOWN_REAL_UP],
        )
    }

    fn accuracy_on_down(&self) -> f64 {
        ratio(
            self.conf[PRED_DOWN_REAL_DOWN],
            self.conf[PRED_UP_REAL_DOWN] + self.conf[PRED_DOWN_REAL_DOWN],
        )
    }

    /// The imbalance-proof score: the unweighted mean of the two per-class accuracies, which
    /// every constant predictor scores exactly 0.5 on whatever the class balance is.
    fn balanced_accuracy(&self) -> f64 {
        0.5 * (self.accuracy_on_up() + self.accuracy_on_down())
    }

    fn precision_up(&self) -> f64 {
        ratio(
            self.conf[PRED_UP_REAL_UP],
            self.conf[PRED_UP_REAL_UP] + self.conf[PRED_UP_REAL_DOWN],
        )
    }

    fn precision_down(&self) -> f64 {
        ratio(
            self.conf[PRED_DOWN_REAL_DOWN],
            self.conf[PRED_DOWN_REAL_DOWN] + self.conf[PRED_DOWN_REAL_UP],
        )
    }

    /// Up-bar base rate over CLASSIFIED bars, the baseline the 2x2 must be read against.
    fn base_rate_up(&self) -> f64 {
        ratio(
            self.conf[PRED_UP_REAL_UP] + self.conf[PRED_DOWN_REAL_UP],
            self.classified(),
        )
    }

    /// Accuracy of the best CONSTANT predictor: always call the majority class.
    fn majority_accuracy(&self) -> f64 {
        let up = self.base_rate_up();
        up.max(1.0 - up)
    }

    /// `P(r > 0)` over EVERY bar, flats in the denominator. This is exactly what buy & hold's
    /// `hit` field reports, so it is the number that confirms or refutes a base rate derived
    /// from it.
    fn up_rate_all_bars(&self) -> f64 {
        ratio(self.up_all, self.all)
    }

    /// Mean realized `sign(mu_hat) r` in basis points: the edge a unit-sized, cost-free,
    /// leverage-free directional bet on the model's sign earned per bar.
    fn edge_bps(&self) -> f64 {
        1.0e4 * ratio(self.edge, self.edge_n)
    }

    /// The same edge over the bars that actually MOVED, i.e. `edge_n` less the flats.
    ///
    /// [`Self::edge_bps`] divides by every directed bar, flats included, and a flat bar
    /// contributes exactly zero to the numerator. So the reported per-decile edge is the
    /// per-moving-bar edge ATTENUATED by that decile's own `1 - flat_fraction`, and the deciles
    /// do not share a flat fraction: the bottom bucket of `|mu_hat|/sigma_hat` is by
    /// construction the quiet bars and holds more flats than the top. A top-minus-bottom
    /// difference and a top-over-all ratio taken from `edge_bps` are therefore differences and
    /// ratios of differently-attenuated quantities, and the ratio is biased UPWARD by
    /// `(1 - flat_top) / (1 - flat_all) > 1`.
    ///
    /// Both are reported. This one is the clean per-event edge; `edge_bps` is the one a book
    /// actually collects per bar it holds, and it is `edge_bps` that belongs in the break-even,
    /// because a flat bar still pays turnover.
    fn edge_bps_moving(&self) -> f64 {
        1.0e4 * ratio(self.edge, self.edge_n - self.flat)
    }

    fn ic(&self) -> f64 {
        self.m.corr()
    }

    /// Fraction of the bucket whose MEAN is provably catch-all-constituted, or NaN when no support
    /// geometry was supplied. The third state is representable and is what an absent support prints.
    fn over_bound_mu_fraction(&self, measured: bool) -> f64 {
        if measured {
            ratio(self.over_bound_mu, self.all)
        } else {
            f64::NAN
        }
    }

    fn over_bound_sigma_fraction(&self, measured: bool) -> f64 {
        if measured {
            ratio(self.over_bound_sigma, self.all)
        } else {
            f64::NAN
        }
    }

    fn over_reference_sigma_fraction(&self, measured: bool) -> f64 {
        if measured {
            ratio(self.over_reference_sigma, self.all)
        } else {
            f64::NAN
        }
    }
}

impl BlockSums for Cell {
    fn absorb(&mut self, other: &Self) {
        for (slot, value) in self.conf.iter_mut().zip(other.conf) {
            *slot += value;
        }
        self.flat += other.flat;
        self.undirected += other.undirected;
        self.all += other.all;
        self.up_all += other.up_all;
        self.edge += other.edge;
        self.edge_n += other.edge_n;
        self.over_bound_mu += other.over_bound_mu;
        self.over_bound_sigma += other.over_bound_sigma;
        self.over_reference_sigma += other.over_reference_sigma;
        self.m.absorb(&other.m);
    }
    fn count(&self) -> f64 {
        self.all
    }
}

/// Mean normalized rank of `mu_hat` inside each realized class, which is an EXACT
/// reparametrization of the AUC.
///
/// With `u_i = (R_i - 0.5) / N` the normalized pooled mid-rank of `mu_hat_i` over the
/// classified bars, the Mann-Whitney statistic satisfies exactly
///
/// ```text
/// AUC = 0.5 + mean(u | realized up) - mean(u | realized down)
/// ```
///
/// (substitute `sum(R | down) = N(N+1)/2 - sum(R | up)` into
/// `AUC = (sum(R|up) - n_up(n_up+1)/2) / (n_up n_down)`; the two expressions differ by
/// `(N - n_up) / (2 n_down) = 0.5`). That matters because the right-hand side is a difference
/// of two BLOCK-ADDITIVE means and therefore bootstraps exactly, while the rank-sum form does
/// not: it carries `n_up(n_up+1)/2`, a nonlinear function of a resampled count, against ranks
/// taken in the original sample.
///
/// The ranks are computed ONCE on the full pooled panel and held fixed across draws. That is
/// deliberate and is what the interval means: `mu_hat`'s ranking function is the estimand, and
/// re-ranking inside every resample would additionally bootstrap the transform.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Placements {
    up_n: f64,
    up_u: f64,
    down_n: f64,
    down_u: f64,
}

impl Placements {
    fn auc(&self) -> f64 {
        if !(self.up_n > 0.0) || !(self.down_n > 0.0) {
            return f64::NAN;
        }
        0.5 + self.up_u / self.up_n - self.down_u / self.down_n
    }
}

impl BlockSums for Placements {
    fn absorb(&mut self, other: &Self) {
        self.up_n += other.up_n;
        self.up_u += other.up_u;
        self.down_n += other.down_n;
        self.down_u += other.down_u;
    }
    fn count(&self) -> f64 {
        self.up_n + self.down_n
    }
}

/// Pooled mid-ranks of `values`, `1..=n`, with tied values sharing their mean rank.
///
/// Mid-ranks rather than ordinal ranks because a tie broken by array order is an arbitrary
/// preference between two identical predictions, and both the Spearman IC and the AUC identity
/// above are only exact under the mid-rank convention.
fn mid_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut ranks = vec![f64::NAN; values.len()];
    let mut start = 0usize;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        // Ranks are 1-based, so the tied group spans `start + 1 ..= end` and its mean rank is
        // the midpoint of that closed interval.
        let mean = 0.5 * ((start + 1) + end) as f64;
        for slot in &order[start..end] {
            ranks[*slot] = mean;
        }
        start = end;
    }
    ranks
}

// ---------------------------------------------------------------------------
// 1. The confusion matrix and the base rate
// ---------------------------------------------------------------------------

/// The full 2x2 of `sign(mu_hat)` against `sign(r)`, with the baselines it has to be read
/// against.
#[derive(Clone, Debug)]
pub struct ConfusionReport {
    /// `[pred up & up, pred up & down, pred down & up, pred down & down]` counts.
    pub counts: [f64; 4],
    /// Bars excluded from the 2x2: realized `r == 0`.
    pub flat_bars: f64,
    /// Bars excluded from the 2x2: no direction in `mu_hat`.
    pub undirected_bars: f64,
    /// Every bar in the panel.
    pub all_bars: f64,
    pub accuracy: Dispersion,
    pub accuracy_on_up: Dispersion,
    pub accuracy_on_down: Dispersion,
    pub balanced_accuracy: Dispersion,
    pub precision_up: Dispersion,
    pub precision_down: Dispersion,
    /// Up-bar base rate over CLASSIFIED bars.
    pub base_rate_up: Dispersion,
    /// `P(r > 0)` over ALL bars, the quantity buy & hold's `hit` field reports.
    pub up_rate_all_bars: Dispersion,
    /// Accuracy of the constant "always up" rule on the classified bars. Equal to the base
    /// rate by definition, and reported with its own interval so the comparison against the
    /// model is two intervals on one axis rather than a number against a recollection.
    pub always_up_accuracy: Dispersion,
    pub always_down_accuracy: Dispersion,
    /// Accuracy of "always call the majority class", the best constant rule.
    pub majority_accuracy: Dispersion,
    /// Model accuracy MINUS majority-class accuracy, resampled paired inside each block. The
    /// only form in which "beats the base rate" is a testable claim.
    pub accuracy_over_majority: Dispersion,
    /// Balanced accuracy minus 0.5, resampled paired. Every constant rule scores exactly 0.5
    /// on balanced accuracy, so this carries no baseline estimation error at all.
    pub balanced_over_chance: Dispersion,
    /// [`super::trade_bench::PolicyStats::hit_rate`] reproduced on this panel: the share of
    /// bars with `f* r > 0` among bars with `f* != 0`.
    pub kelly_sign_hit_rate: Dispersion,
    /// Share of bars where the Kelly solve's direction differs from `sign(mu_hat)`, which is
    /// the entire reason the two hit rates are not the same number.
    pub kelly_sign_disagreement: Dispersion,
}

/// The bench's own directional score, reproduced so this module's number can be reconciled
/// with it instead of merely compared to it.
#[derive(Clone, Copy, Debug, Default)]
struct KellySign {
    positioned: f64,
    hits: f64,
    all: f64,
    disagree: f64,
}

impl KellySign {
    fn push(&mut self, bar: &SkillBar) {
        self.all += 1.0;
        let kelly = direction(bar.free);
        if kelly != 0 {
            self.positioned += 1.0;
            if f64::from(kelly) * bar.r > 0.0 {
                self.hits += 1.0;
            }
        }
        if kelly != direction(bar.mu) {
            self.disagree += 1.0;
        }
    }

    fn hit_rate(&self) -> f64 {
        ratio(self.hits, self.positioned)
    }

    fn disagreement(&self) -> f64 {
        ratio(self.disagree, self.all)
    }
}

impl BlockSums for KellySign {
    fn absorb(&mut self, other: &Self) {
        self.positioned += other.positioned;
        self.hits += other.hits;
        self.all += other.all;
        self.disagree += other.disagree;
    }
    fn count(&self) -> f64 {
        self.all
    }
}

fn confusion_report(panel: &SkillPanel) -> ConfusionReport {
    let cells: Vec<Cell> = panel.per_block(Cell::push);
    let mut pooled = Cell::default();
    for cell in &cells {
        BlockSums::absorb(&mut pooled, cell);
    }
    let kelly: Vec<KellySign> = panel.per_block(KellySign::push);
    ConfusionReport {
        counts: pooled.conf,
        flat_bars: pooled.flat,
        undirected_bars: pooled.undirected,
        all_bars: pooled.all,
        accuracy: blocked(&cells, Cell::accuracy),
        accuracy_on_up: blocked(&cells, Cell::accuracy_on_up),
        accuracy_on_down: blocked(&cells, Cell::accuracy_on_down),
        balanced_accuracy: blocked(&cells, Cell::balanced_accuracy),
        precision_up: blocked(&cells, Cell::precision_up),
        precision_down: blocked(&cells, Cell::precision_down),
        base_rate_up: blocked(&cells, Cell::base_rate_up),
        up_rate_all_bars: blocked(&cells, Cell::up_rate_all_bars),
        always_up_accuracy: blocked(&cells, Cell::base_rate_up),
        always_down_accuracy: blocked(&cells, |cell| 1.0 - cell.base_rate_up()),
        majority_accuracy: blocked(&cells, Cell::majority_accuracy),
        accuracy_over_majority: blocked(&cells, |cell| {
            cell.accuracy() - cell.majority_accuracy()
        }),
        balanced_over_chance: blocked(&cells, |cell| cell.balanced_accuracy() - 0.5),
        kelly_sign_hit_rate: blocked(&kelly, KellySign::hit_rate),
        kelly_sign_disagreement: blocked(&kelly, KellySign::disagreement),
    }
}

// ---------------------------------------------------------------------------
// 2. The information coefficient, decomposed three ways
// ---------------------------------------------------------------------------

/// `corr(mu_hat, r)` measured three ways, because the pooled number alone cannot be
/// interpreted.
#[derive(Clone, Debug)]
pub struct IcDecomposition {
    /// Pooled Pearson over every bar of every name.
    pub pooled_pearson: Dispersion,
    /// Pooled Spearman: Pearson of the pooled mid-ranks of both series.
    pub pooled_spearman: Dispersion,
    /// The `R^2` a pooled Mincer-Zarnowitz regression reports, `pooled_pearson^2`, stated so
    /// the comparison against the 0.062-0.069 already measured is arithmetic rather than
    /// mental.
    pub pooled_r2: f64,
    /// One Pearson IC per symbol, ASCENDING. The unit of analysis of the within-name view.
    pub per_symbol: Vec<f64>,
    /// Bars behind each entry of `per_symbol`, in the same order.
    pub per_symbol_bars: Vec<usize>,
    pub within_median: Dispersion,
    pub within_mean: Dispersion,
    pub within_q1: f64,
    pub within_q3: f64,
    pub within_fraction_positive: Dispersion,
    /// Symbols carrying enough varying bars for an IC at all.
    pub symbols_measured: usize,
    /// Symbols dropped for a degenerate `mu_hat` or `r`.
    pub symbols_dropped: usize,
    /// Pooled Pearson after z-scoring both series inside each `(symbol, month)` block.
    pub standardized_pearson: Dispersion,
    /// The same, on within-block mid-ranks rather than within-block z-scores.
    pub standardized_spearman: Dispersion,
    /// Blocks too degenerate to standardize.
    pub blocks_dropped: usize,
}

/// The volatility-prediction channel, measured directly so the pooled IC can be attributed
/// instead of argued about.
///
/// If `corr(|mu_hat|, |r|)` and `corr(sigma_hat, |r|)` are large while the standardized
/// directional IC is near zero, the model predicts HOW MUCH a name moves and not WHICH WAY,
/// and the pooled directional IC was reading the magnitude channel through the product
/// `E[sign_mu sign_r |mu| |r|]`.
#[derive(Clone, Debug)]
pub struct MagnitudeIc {
    pub abs_mu_vs_abs_r: Dispersion,
    pub sigma_vs_abs_r: Dispersion,
    /// The same two, standardized inside each `(symbol, month)` block, so a magnitude IC
    /// cannot be credited to cross-sectional scale either.
    pub abs_mu_vs_abs_r_standardized: Dispersion,
    pub sigma_vs_abs_r_standardized: Dispersion,
}

/// Pairs of `(x, y)` grouped by block, non-finite pairs dropped.
fn block_pairs(
    panel: &SkillPanel,
    select: impl Fn(&SkillBar) -> (f64, f64),
) -> BTreeMap<u64, Vec<(f64, f64)>> {
    let mut grouped: BTreeMap<u64, Vec<(f64, f64)>> = BTreeMap::new();
    for (window, bar) in panel.flat() {
        let (x, y) = select(bar);
        if x.is_finite() && y.is_finite() {
            grouped.entry(window.block).or_default().push((x, y));
        }
    }
    grouped
}

/// Z-score `x` and `y` inside each block and return the per-block moments of the result.
///
/// The pooled correlation of the returned accumulators is algebraically
/// `sum_b n_b rho_b / sum_b n_b`, the bar-count-weighted mean of the within-block
/// correlations: inside a block both series have mean 0 and unit variance, so a block
/// contributes `(n_b, 0, 0, n_b, n_b, n_b rho_b)`. Between-block scale therefore cannot
/// enter, which is the entire point.
///
/// Blocks with fewer than two usable bars, or with a degenerate `x` or `y`, are dropped and
/// counted rather than contributing zeros.
fn standardized_blocks(
    panel: &SkillPanel,
    select: impl Fn(&SkillBar) -> (f64, f64),
) -> (Vec<Moments>, usize) {
    let grouped = block_pairs(panel, select);
    let mut dropped = 0usize;
    let mut out = Vec::with_capacity(grouped.len());
    for pairs in grouped.into_values() {
        if pairs.len() < 2 {
            dropped += 1;
            continue;
        }
        let n = pairs.len() as f64;
        let mean_x = pairs.iter().map(|(x, _)| *x).sum::<f64>() / n;
        let mean_y = pairs.iter().map(|(_, y)| *y).sum::<f64>() / n;
        // Population standard deviation. Any common divisor cancels out of the correlation as
        // long as the SAME one is used for both series inside a block.
        let sd_x = (pairs.iter().map(|(x, _)| (x - mean_x).powi(2)).sum::<f64>() / n).sqrt();
        let sd_y = (pairs.iter().map(|(_, y)| (y - mean_y).powi(2)).sum::<f64>() / n).sqrt();
        if !(sd_x > 0.0) || !(sd_y > 0.0) {
            dropped += 1;
            continue;
        }
        let mut moments = Moments::default();
        for (x, y) in &pairs {
            moments.push((x - mean_x) / sd_x, (y - mean_y) / sd_y);
        }
        out.push(moments);
    }
    (out, dropped)
}

/// The same, on within-block mid-ranks: a within-block Spearman, pooled the same way.
fn standardized_rank_blocks(
    panel: &SkillPanel,
    select: impl Fn(&SkillBar) -> (f64, f64),
) -> Vec<Moments> {
    let grouped = block_pairs(panel, select);
    let mut out = Vec::with_capacity(grouped.len());
    for pairs in grouped.into_values() {
        if pairs.len() < 2 {
            continue;
        }
        let xs: Vec<f64> = pairs.iter().map(|(x, _)| *x).collect();
        let ys: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();
        let rank_x = mid_ranks(&xs);
        let rank_y = mid_ranks(&ys);
        let n = pairs.len() as f64;
        // Mid-ranks of `1..=n` always have mean `(n + 1) / 2`, whatever the tie structure.
        let mean = 0.5 * (n + 1.0);
        let sd_x = (rank_x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
        let sd_y = (rank_y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
        if !(sd_x > 0.0) || !(sd_y > 0.0) {
            continue;
        }
        let mut moments = Moments::default();
        for (x, y) in rank_x.iter().zip(&rank_y) {
            moments.push((x - mean) / sd_x, (y - mean) / sd_y);
        }
        out.push(moments);
    }
    out
}

/// The within-name view: one IC per symbol, and the distribution over symbols.
struct WithinName {
    ics: Vec<f64>,
    bars: Vec<usize>,
    dropped: usize,
    median: Dispersion,
    mean: Dispersion,
    fraction_positive: Dispersion,
}

/// Per-symbol Pearson IC, with intervals taken by resampling SYMBOLS.
///
/// The resampling unit has to be the symbol here and nothing else: the estimand is a
/// distribution whose observations ARE the per-symbol ICs, so resampling `(symbol, month)`
/// blocks would break the grouping the statistic is defined by. The RNG, the draw count and
/// the percentile convention are every other interval's, so the construction is comparable
/// even though the unit deliberately is not — and `Dispersion::blocks` reports the symbol
/// count, so a reader can see which unit an interval was taken over.
fn within_name_ic(panel: &SkillPanel) -> WithinName {
    let mut grouped: BTreeMap<u32, Moments> = BTreeMap::new();
    for (window, bar) in panel.flat() {
        grouped
            .entry(window.symbol)
            .or_default()
            .push(bar.mu, bar.r);
    }
    let mut values: Vec<(f64, usize)> = Vec::with_capacity(grouped.len());
    let mut dropped = 0usize;
    for moments in grouped.into_values() {
        let ic = moments.corr();
        if ic.is_finite() {
            values.push((ic, moments.n as usize));
        } else {
            dropped += 1;
        }
    }
    values.sort_by(|a, b| a.0.total_cmp(&b.0));
    let ics: Vec<f64> = values.iter().map(|(ic, _)| *ic).collect();
    let bars: Vec<usize> = values.iter().map(|(_, n)| *n).collect();

    // `pick` receives an ASCENDING slice, so a percentile-based summary needs no second sort
    // and the median and the IQR are the same convention as every other interval's.
    let summarize = |pick: fn(&[f64]) -> f64| -> Dispersion {
        let mut out = Dispersion {
            mean: pick(&ics),
            se: f64::NAN,
            ci_low: f64::NAN,
            ci_high: f64::NAN,
            blocks: ics.len(),
            samples: ics.len(),
        };
        if ics.len() < 2 {
            return out;
        }
        let mut rng = ChaCha12Rng::seed_from_u64(BOOTSTRAP_SEED);
        let mut draws = Vec::with_capacity(BOOTSTRAP_DRAWS);
        let mut scratch = vec![0.0f64; ics.len()];
        for _ in 0..BOOTSTRAP_DRAWS {
            for slot in scratch.iter_mut() {
                *slot = *ics.choose(&mut rng).expect("ics is non-empty");
            }
            scratch.sort_by(f64::total_cmp);
            let value = pick(&scratch);
            if value.is_finite() {
                draws.push(value);
            }
        }
        if draws.len() < 2 {
            return out;
        }
        draws.sort_by(f64::total_cmp);
        out.se = standard_deviation(&draws);
        let tail = (1.0 - CI_MASS) / 2.0;
        out.ci_low = sorted_percentile(&draws, tail);
        out.ci_high = sorted_percentile(&draws, 1.0 - tail);
        out
    };
    WithinName {
        median: summarize(|sorted| sorted_percentile(sorted, 0.5)),
        mean: summarize(|values| {
            if values.is_empty() {
                f64::NAN
            } else {
                values.iter().sum::<f64>() / values.len() as f64
            }
        }),
        fraction_positive: summarize(|values| {
            if values.is_empty() {
                f64::NAN
            } else {
                values.iter().filter(|ic| **ic > 0.0).count() as f64 / values.len() as f64
            }
        }),
        ics,
        bars,
        dropped,
    }
}

fn ic_decomposition(panel: &SkillPanel) -> IcDecomposition {
    let pooled: Vec<Moments> = panel.per_block(|m: &mut Moments, bar| m.push(bar.mu, bar.r));
    let pooled_pearson = blocked(&pooled, Moments::corr);

    // Pooled Spearman: rank once, globally, then treat the rank pairs as the series. The ranks
    // are the estimand's transform and are held fixed across draws, for the same reason the
    // AUC's placements are. Bars whose `mu_hat` or `r` is not finite are dropped BEFORE
    // ranking, so a NaN cannot be handed a real rank at the end of the order.
    let usable: Vec<(u64, f64, f64)> = panel
        .flat()
        .filter(|(_, bar)| bar.mu.is_finite() && bar.r.is_finite())
        .map(|(window, bar)| (window.block, bar.mu, bar.r))
        .collect();
    let rank_mu = mid_ranks(&usable.iter().map(|(_, mu, _)| *mu).collect::<Vec<_>>());
    let rank_r = mid_ranks(&usable.iter().map(|(_, _, r)| *r).collect::<Vec<_>>());
    let mut rank_grouped: BTreeMap<u64, Moments> = BTreeMap::new();
    for (index, (block, _, _)) in usable.iter().enumerate() {
        rank_grouped
            .entry(*block)
            .or_default()
            .push(rank_mu[index], rank_r[index]);
    }
    let rank_blocks: Vec<Moments> = rank_grouped.into_values().collect();
    let pooled_spearman = blocked(&rank_blocks, Moments::corr);

    let within = within_name_ic(panel);
    let (standardized, blocks_dropped) = standardized_blocks(panel, |bar| (bar.mu, bar.r));
    let standardized_ranks = standardized_rank_blocks(panel, |bar| (bar.mu, bar.r));

    IcDecomposition {
        pooled_r2: pooled_pearson.mean * pooled_pearson.mean,
        pooled_pearson,
        pooled_spearman,
        within_q1: sorted_percentile(&within.ics, 0.25),
        within_q3: sorted_percentile(&within.ics, 0.75),
        symbols_measured: within.ics.len(),
        symbols_dropped: within.dropped,
        per_symbol: within.ics,
        per_symbol_bars: within.bars,
        within_median: within.median,
        within_mean: within.mean,
        within_fraction_positive: within.fraction_positive,
        standardized_pearson: blocked(&standardized, Moments::corr),
        standardized_spearman: blocked(&standardized_ranks, Moments::corr),
        blocks_dropped,
    }
}

fn magnitude_ic(panel: &SkillPanel) -> MagnitudeIc {
    let abs_mu: Vec<Moments> =
        panel.per_block(|m: &mut Moments, bar| m.push(bar.mu.abs(), bar.r.abs()));
    let sigma: Vec<Moments> =
        panel.per_block(|m: &mut Moments, bar| m.push(bar.sigma, bar.r.abs()));
    let (abs_mu_std, _) = standardized_blocks(panel, |bar| (bar.mu.abs(), bar.r.abs()));
    let (sigma_std, _) = standardized_blocks(panel, |bar| (bar.sigma, bar.r.abs()));
    MagnitudeIc {
        abs_mu_vs_abs_r: blocked(&abs_mu, Moments::corr),
        sigma_vs_abs_r: blocked(&sigma, Moments::corr),
        abs_mu_vs_abs_r_standardized: blocked(&abs_mu_std, Moments::corr),
        sigma_vs_abs_r_standardized: blocked(&sigma_std, Moments::corr),
    }
}

// ---------------------------------------------------------------------------
// 5. AUC
// ---------------------------------------------------------------------------

/// AUC of `mu_hat` as a ranker of the realized sign, with a blocked standard error.
///
/// Threshold-free, and invariant to any monotone rescaling of `mu_hat`, so it is the one
/// summary that cannot be moved by the over-dispersion the calibration fit measured: a
/// Mincer-Zarnowitz slope of 0.36 says `mu_hat` is 2.8x too large, and a 2.8x rescaling
/// changes no rank and therefore no digit of this number.
fn auc(panel: &SkillPanel) -> Dispersion {
    // The population is exactly the 2x2's: directed prediction, non-flat outcome. Ranking
    // bars with no realized direction would put mass in neither class and change the
    // normalization without changing the estimand.
    let classified: Vec<(u64, f64, bool)> = panel
        .flat()
        .filter(|(_, bar)| direction(bar.mu) != 0 && direction(bar.r) != 0)
        .map(|(window, bar)| (window.block, bar.mu, bar.r > 0.0))
        .collect();
    if classified.is_empty() {
        return Dispersion::nan();
    }
    let ranks = mid_ranks(&classified.iter().map(|(_, mu, _)| *mu).collect::<Vec<_>>());
    let n = classified.len() as f64;
    let mut grouped: BTreeMap<u64, Placements> = BTreeMap::new();
    for (index, (block, _, up)) in classified.iter().enumerate() {
        let placement = (ranks[index] - 0.5) / n;
        let entry = grouped.entry(*block).or_default();
        if *up {
            entry.up_n += 1.0;
            entry.up_u += placement;
        } else {
            entry.down_n += 1.0;
            entry.down_u += placement;
        }
    }
    let blocks: Vec<Placements> = grouped.into_values().collect();
    blocked(&blocks, Placements::auc)
}

// ---------------------------------------------------------------------------
// 3. The confidence curve
// ---------------------------------------------------------------------------

/// One confidence bucket's directional statistics.
#[derive(Clone, Debug)]
pub struct DecileRow {
    pub bars: usize,
    pub accuracy: Dispersion,
    pub balanced_accuracy: Dispersion,
    /// Mean realized `sign(mu_hat) r` in bps: the edge conditional on the position's sign.
    pub edge_bps: Dispersion,
    pub ic: Dispersion,
    /// Up-bar base rate INSIDE the bucket, because a bucket whose class balance differs from
    /// its neighbour's would make the two raw accuracies incomparable.
    pub base_rate_up: Dispersion,
    /// FLAT bars in the bucket, `r` exactly zero. Surfaced because it is the attenuation factor
    /// of [`Self::edge_bps`] and it differs across buckets; a reader comparing two `edge_bps`
    /// figures without it is comparing two different denominators.
    pub flat_bars: usize,
    /// [`Self::edge_bps`] with the flats removed from the denominator: the edge per bar that
    /// MOVED. See `Cell::edge_bps_moving`.
    pub edge_bps_moving: Dispersion,
    /// Fraction of the bucket whose MEAN is provably constituted by catch-all mass, i.e.
    /// `|mu_hat|` above the interior bound. NaN when no support geometry was supplied.
    pub over_bound_mu: Dispersion,
    /// The same for `sigma_hat` against the same hard bound.
    pub over_bound_sigma: Dispersion,
    /// `sigma_hat` above the marginal-interior RMS REFERENCE - atypical, not impossible.
    pub over_reference_sigma: Dispersion,
}

/// Per-block cells of every decile at once, so a difference between two deciles is resampled
/// PAIRED inside each block instead of as two independent marginals.
#[derive(Clone, Copy, Debug, Default)]
struct DecileCells {
    cells: [Cell; DECILES],
}

impl BlockSums for DecileCells {
    fn absorb(&mut self, other: &Self) {
        for (mine, theirs) in self.cells.iter_mut().zip(&other.cells) {
            BlockSums::absorb(mine, theirs);
        }
    }
    fn count(&self) -> f64 {
        self.cells.iter().map(|cell| cell.all).sum()
    }
}

/// Directional skill as a function of the model's own confidence: the deliverable.
#[derive(Clone, Debug)]
pub struct ConfidenceCurve {
    pub selector: &'static str,
    /// The nine interior decile cutpoints of the selector, held FIXED across bootstrap draws
    /// because the selection rule is the estimand rather than something being estimated.
    pub cutpoints: Vec<f64>,
    pub rows: Vec<DecileRow>,
    /// Bars the selector could not rank (non-finite `mu_hat`, or `sigma_hat <= 0` for the
    /// Sharpe selector), excluded and counted.
    pub excluded: usize,
    /// Top decile MINUS bottom decile, resampled paired. These, and not the ten marginal
    /// intervals, are what makes "skill concentrates in high-confidence bars" testable.
    pub top_minus_bottom_accuracy: Dispersion,
    pub top_minus_bottom_balanced: Dispersion,
    pub top_minus_bottom_edge_bps: Dispersion,
    pub top_minus_bottom_ic: Dispersion,
    /// Spearman correlation between the decile index and its accuracy, over the ten buckets.
    /// A SHAPE summary with no interval, because ten buckets sharing 256 blocks are not ten
    /// independent observations. Reported as a description and never as evidence.
    pub accuracy_rank_correlation: f64,
    pub edge_rank_correlation: f64,
    /// TOP-DECILE EDGE over ALL-BAR EDGE, as a ratio, resampled inside one draw so numerator and
    /// denominator move together and the interval is the ratio's own rather than a quotient of
    /// two marginal intervals.
    ///
    /// This is the multiplier the SELECTION buys on the quantity that is actually money.
    /// Accuracy is not money: a rule can gain fifteen points of directional accuracy on bars
    /// whose moves are too small to pay a spread and earn nothing, which is precisely why the
    /// concentration criterion is a direction statistic and the economic screen is a separate
    /// number. Stated as a RATIO because it is what composes multiplicatively with a cost
    /// reduction from an orthogonal axis — restricting to liquid names cuts the matched cost by
    /// about 2.1x, and a selection that raises edge by more than the residual shortfall is the
    /// only way the two axes together clear.
    pub top_edge_multiple: Dispersion,
    /// The same ratio on the per-MOVING-bar edge, which is the one that is not inflated by the
    /// bottom decile holding more flat bars than the top. This is the honest multiplier to
    /// compose against a cost reduction; [`Self::top_edge_multiple`] is reported beside it so the
    /// size of the attenuation bias is visible rather than asserted.
    pub top_edge_multiple_moving: Dispersion,
    /// Per-BLOCK bar counts of the bottom and top decile, as `[min, median, max]`.
    ///
    /// # Why the pairing claim needs this and cannot assume it
    ///
    /// A top-minus-bottom difference is only PAIRED WITHIN a block if both deciles are populated
    /// in essentially every block. For `|mu_hat|` that is plausible: the scale of `mu_hat` tracks
    /// the name's own volatility, so a volatile name's quiet hours still fall in its low deciles.
    /// For `|mu_hat|/sigma_hat` it is NOT plausible a priori, because the selector is scale-FREE:
    /// a structurally quiet instrument can sit above the pooled cutpoint at almost every bar and a
    /// volatile one below it at almost every bar, in which case decile 9 is a handful of blocks,
    /// decile 0 is a disjoint handful, and the "paired" difference is a BETWEEN-NAME comparison
    /// wearing a within-block interval.
    ///
    /// The same power family sizes the book — Kelly is `mu/sigma^2`, this selector is `mu/sigma` —
    /// so whichever way this lands is a statement about more than a diagnostic.
    ///
    /// Measured rather than argued, for both selectors, because the two answers can differ and the
    /// consequence is a retraction either way.
    pub occupancy_bottom: [f64; 3],
    pub occupancy_top: [f64; 3],
    /// Blocks in which the decile is EMPTY, the sharp form of the same question: a paired
    /// difference is undefined in a block that holds one side and not the other.
    pub blocks_missing_bottom: usize,
    pub blocks_missing_top: usize,
    pub blocks: usize,
    /// Symbol composition of the TOP decile: `(corpus symbol index, share of the decile's bars)`,
    /// descending, truncated to [`COMPOSITION_NAMES`], beside the count of distinct symbols and
    /// the Herfindahl index of the shares over EVERY name.
    ///
    /// This is what a cross against another selection axis must be checked against. If the top
    /// confidence decile is largely the same names as the deepest liquidity decile, the two axes
    /// are one axis and their benefits do not multiply.
    pub top_composition: Vec<(u32, f64)>,
    pub top_distinct_symbols: usize,
    pub top_herfindahl: f64,
    /// Mean `sigma_hat` in the bottom and top decile, in basis points. The direct read on whether
    /// the selector is sorting by forecast quality or by instrument volatility.
    pub sigma_bottom_bps: f64,
    pub sigma_top_bps: f64,
    /// The support geometry the catch-all indicators were computed against, carried so the report
    /// can say NOT MEASURED rather than print a fraction with no threshold beside it.
    pub interior_bound_bps: Option<f64>,
    pub interior_marginal_rms_bps: Option<f64>,
}

/// Names listed in a decile's composition summary.
pub const COMPOSITION_NAMES: usize = 12;

/// The nine interior decile cutpoints of both confidence selectors, over ONE panel.
///
/// # Why this is an explicit input and not derived per call
///
/// The decile boundaries are a property of the POPULATION, and the selection rule they define is
/// what a policy would deploy. Deriving them inside the measurement means restricting the panel -
/// to one liquidity decile, one sector, one month - silently re-derives them, so the population
/// and the rule change together and the two are no longer separable: the restricted run measures a
/// DIFFERENT strategy on a different subset rather than the same strategy on a subset. That is not
/// a subtle bias, it is a category error, and it is invisible at the call site if the parameter can
/// be omitted.
///
/// So it cannot be omitted. A cross of this module's confidence axis against any other axis must
/// build these on the FULL panel with [`Self::from_panel`] and carry them into the restricted
/// measurement unchanged.
#[derive(Clone, Debug)]
pub struct SkillCutpoints {
    /// Interior cutpoints of `|mu_hat|`, ascending, `DECILES - 1` of them.
    pub abs_mu: Vec<f64>,
    /// Interior cutpoints of `|mu_hat|/sigma_hat`, ascending.
    pub sharpe: Vec<f64>,
    /// HARD bound on `|mu_hat|` and on `sigma_hat` for any law supported on the INTERIOR bins of
    /// DOF `r`, in basis points, or `None` when no support was supplied.
    ///
    /// # What it proves and what it does not
    ///
    /// `|E[r]| <= max|r|` and `E[r^2] <= max(r^2)` over the support, so one constant - `max|center|`
    /// over the interior bins - bounds BOTH moments. A bar whose `|mu_hat|` or `sigma_hat` exceeds
    /// it CANNOT be produced by any distribution living on the interior, so that moment is
    /// necessarily constituted by mass in the two catch-all bins, which decode at the clipped
    /// support bound rather than at a fitted conditional mean. An exact indicator: one comparison
    /// per bar, nothing fitted, derived from the persisted support.
    ///
    /// This is the CONSERVATIVE test. The much lower "interior RMS" figure circulating beside it is
    /// the sd of the equal-mass marginal TRUNCATED to interior bins - a reference for a
    /// marginal-SHAPED law, NOT a bound on an arbitrary interior distribution - so a bar above that
    /// is atypical rather than provably catch-all-driven. Both are reported, under different names,
    /// because conflating them would let "atypical" print as "impossible".
    pub interior_bound_bps: Option<f64>,
    /// The equal-mass marginal's sd restricted to interior bins, in basis points. A REFERENCE, not
    /// a bound. `None` when no support was supplied.
    pub interior_marginal_rms_bps: Option<f64>,
}

impl SkillCutpoints {
    /// Derive both selectors' boundaries from the panel that will be measured. Correct only when
    /// the panel measured IS the population the rule is defined on.
    pub fn from_panel(panel: &SkillPanel) -> Self {
        Self {
            abs_mu: pooled_cutpoints(panel, selector_abs_mu),
            sharpe: pooled_cutpoints(panel, selector_sharpe),
            interior_bound_bps: None,
            interior_marginal_rms_bps: None,
        }
    }

    /// The same, plus the interior geometry of the DOF `r` support, so the catch-all indicators can
    /// be reported. `centers` is the decode vector the moments are actually taken off.
    pub fn with_support_geometry(mut self, centers: &[f64]) -> Self {
        if centers.len() < 3 {
            return self;
        }
        let interior = &centers[1..centers.len() - 1];
        let bound = interior.iter().fold(0.0f64, |worst, c| worst.max(c.abs()));
        // Equal-mass bins, so the marginal's interior second moment is the unweighted mean of the
        // squared interior centers.
        let rms = (interior.iter().map(|c| c * c).sum::<f64>() / interior.len() as f64).sqrt();
        self.interior_bound_bps = Some(1.0e4 * bound);
        self.interior_marginal_rms_bps = Some(1.0e4 * rms);
        self
    }

    fn of(&self, selector_name: &str) -> &[f64] {
        if selector_name == SELECTOR_SHARPE {
            &self.sharpe
        } else {
            &self.abs_mu
        }
    }
}

fn pooled_cutpoints(panel: &SkillPanel, selector: impl Fn(&SkillBar) -> f64) -> Vec<f64> {
    let mut values: Vec<f64> = panel
        .flat()
        .map(|(_, bar)| selector(bar))
        .filter(|value| value.is_finite())
        .collect();
    values.sort_by(f64::total_cmp);
    (1..DECILES)
        .map(|k| sorted_percentile(&values, k as f64 / DECILES as f64))
        .collect()
}

/// Per-block cells induced by GIVEN cutpoints, plus the count of unrankable bars.
fn decile_assignment(
    panel: &SkillPanel,
    selector: impl Fn(&SkillBar) -> f64,
    cutpoints: &[f64],
    geometry: (f64, f64),
) -> (Vec<DecileCells>, usize) {
    let excluded = panel
        .flat()
        .filter(|(_, bar)| !selector(bar).is_finite())
        .count();
    let mut grouped: BTreeMap<u64, DecileCells> = BTreeMap::new();
    for (window, bar) in panel.flat() {
        let value = selector(bar);
        if !value.is_finite() {
            continue;
        }
        let index = decile_of(&cutpoints, value);
        let cell = &mut grouped.entry(window.block).or_default().cells[index];
        cell.push(bar);
        cell.push_geometry(bar, geometry.0, geometry.1);
    }
    (grouped.into_values().collect(), excluded)
}

/// Bucket index of `value` against ascending interior `cutpoints`: the number of cutpoints
/// strictly below it, clamped into `0..DECILES`.
fn decile_of(cutpoints: &[f64], value: f64) -> usize {
    cutpoints
        .partition_point(|cut| *cut < value)
        .min(DECILES - 1)
}

/// Spearman correlation of a short series against its own index, for the shape summary.
fn rank_correlation_against_index(values: &[f64]) -> f64 {
    let usable: Vec<(f64, f64)> = values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(index, value)| (index as f64, *value))
        .collect();
    if usable.len() < 3 {
        return f64::NAN;
    }
    let rank_x = mid_ranks(&usable.iter().map(|(x, _)| *x).collect::<Vec<_>>());
    let rank_y = mid_ranks(&usable.iter().map(|(_, y)| *y).collect::<Vec<_>>());
    let mut moments = Moments::default();
    for (x, y) in rank_x.iter().zip(&rank_y) {
        moments.push(*x, *y);
    }
    moments.corr()
}

/// `[min, median, max]` of one decile's per-block bar count. Blocks are the resampling unit, so
/// this is the distribution over exactly the units every interval is built from.
fn occupancy(blocks: &[DecileCells], decile: usize) -> [f64; 3] {
    if blocks.is_empty() {
        return [f64::NAN; 3];
    }
    let mut counts: Vec<f64> = blocks.iter().map(|b| b.cells[decile].all).collect();
    counts.sort_by(f64::total_cmp);
    [
        counts[0],
        sorted_percentile(&counts, 0.5),
        counts[counts.len() - 1],
    ]
}

/// Which symbols populate the top decile, how concentrated that is, and the mean `sigma_hat` of
/// the bottom and top buckets.
fn top_decile_composition(
    panel: &SkillPanel,
    selector: &impl Fn(&SkillBar) -> f64,
    cutpoints: &[f64],
    top: usize,
) -> (Vec<(u32, f64)>, usize, f64, f64, f64) {
    let mut counts: BTreeMap<u32, f64> = BTreeMap::new();
    let mut total = 0.0f64;
    let mut sigma = [Moments::default(), Moments::default()];
    for (window, bar) in panel.flat() {
        let value = selector(bar);
        if !value.is_finite() {
            continue;
        }
        let decile = decile_of(cutpoints, value);
        if decile == 0 {
            sigma[0].push(bar.sigma, bar.sigma);
        }
        if decile != top {
            continue;
        }
        sigma[1].push(bar.sigma, bar.sigma);
        *counts.entry(window.symbol).or_default() += 1.0;
        total += 1.0;
    }
    let distinct = counts.len();
    let mut shares: Vec<(u32, f64)> = counts
        .into_iter()
        .map(|(symbol, count)| (symbol, ratio(count, total)))
        .collect();
    // Herfindahl over EVERY name, computed before truncation, so the concentration summary is not
    // a property of how many rows happen to be printed.
    let herfindahl = shares.iter().map(|(_, share)| share * share).sum();
    shares.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    shares.truncate(COMPOSITION_NAMES);
    (
        shares,
        distinct,
        herfindahl,
        1.0e4 * sigma[0].mean_x(),
        1.0e4 * sigma[1].mean_x(),
    )
}

fn confidence_curve(
    panel: &SkillPanel,
    selector_name: &'static str,
    selector: impl Fn(&SkillBar) -> f64,
    cuts: &SkillCutpoints,
) -> ConfidenceCurve {
    let cutpoints = cuts.of(selector_name).to_vec();
    // NaN when no support was supplied, which makes every comparison false and every reported
    // fraction NaN rather than zero: absent geometry prints as "not measured", never as "none found".
    let measured = cuts.interior_bound_bps.is_some() && cuts.interior_marginal_rms_bps.is_some();
    let geometry = (
        cuts.interior_bound_bps.unwrap_or(f64::NAN),
        cuts.interior_marginal_rms_bps.unwrap_or(f64::NAN),
    );
    let (blocks, excluded) = decile_assignment(panel, &selector, &cutpoints, geometry);
    let mut pooled = DecileCells::default();
    for block in &blocks {
        BlockSums::absorb(&mut pooled, block);
    }
    let rows: Vec<DecileRow> = (0..DECILES)
        .map(|decile| DecileRow {
            bars: pooled.cells[decile].all as usize,
            accuracy: blocked(&blocks, |sums| sums.cells[decile].accuracy()),
            balanced_accuracy: blocked(&blocks, |sums| {
                sums.cells[decile].balanced_accuracy()
            }),
            edge_bps: blocked(&blocks, |sums| sums.cells[decile].edge_bps()),
            edge_bps_moving: blocked(&blocks, |sums| sums.cells[decile].edge_bps_moving()),
            ic: blocked(&blocks, |sums| sums.cells[decile].ic()),
            base_rate_up: blocked(&blocks, |sums| sums.cells[decile].base_rate_up()),
            flat_bars: pooled.cells[decile].flat as usize,
            over_bound_mu: blocked(&blocks, |sums| {
                sums.cells[decile].over_bound_mu_fraction(measured)
            }),
            over_bound_sigma: blocked(&blocks, |sums| {
                sums.cells[decile].over_bound_sigma_fraction(measured)
            }),
            over_reference_sigma: blocked(&blocks, |sums| {
                sums.cells[decile].over_reference_sigma_fraction(measured)
            }),
        })
        .collect();
    let top = DECILES - 1;
    let composition = top_decile_composition(panel, &selector, &cutpoints, top);
    ConfidenceCurve {
        selector: selector_name,
        cutpoints,
        excluded,
        top_minus_bottom_accuracy: blocked(&blocks, |sums| {
            sums.cells[top].accuracy() - sums.cells[0].accuracy()
        }),
        top_minus_bottom_balanced: blocked(&blocks, |sums| {
            sums.cells[top].balanced_accuracy() - sums.cells[0].balanced_accuracy()
        }),
        top_minus_bottom_edge_bps: blocked(&blocks, |sums| {
            sums.cells[top].edge_bps() - sums.cells[0].edge_bps()
        }),
        top_minus_bottom_ic: blocked(&blocks, |sums| {
            sums.cells[top].ic() - sums.cells[0].ic()
        }),
        top_edge_multiple: blocked(&blocks, |sums| {
            // The all-bar denominator is rebuilt from the SAME draw's ten cells, so a draw that
            // happened to sample a high-edge month raises numerator and denominator together and
            // the interval reflects the ratio rather than the level.
            let mut all = Cell::default();
            for cell in &sums.cells {
                BlockSums::absorb(&mut all, cell);
            }
            ratio(sums.cells[top].edge_bps(), all.edge_bps())
        }),
        top_edge_multiple_moving: blocked(&blocks, |sums| {
            let mut all = Cell::default();
            for cell in &sums.cells {
                BlockSums::absorb(&mut all, cell);
            }
            ratio(sums.cells[top].edge_bps_moving(), all.edge_bps_moving())
        }),
        occupancy_bottom: occupancy(&blocks, 0),
        occupancy_top: occupancy(&blocks, top),
        blocks_missing_bottom: blocks.iter().filter(|b| b.cells[0].all == 0.0).count(),
        blocks_missing_top: blocks.iter().filter(|b| b.cells[top].all == 0.0).count(),
        blocks: blocks.len(),
        top_composition: composition.0,
        top_distinct_symbols: composition.1,
        top_herfindahl: composition.2,
        sigma_bottom_bps: composition.3,
        sigma_top_bps: composition.4,
        interior_bound_bps: cuts.interior_bound_bps,
        interior_marginal_rms_bps: cuts.interior_marginal_rms_bps,
        accuracy_rank_correlation: rank_correlation_against_index(
            &rows.iter().map(|row| row.accuracy.mean).collect::<Vec<_>>(),
        ),
        edge_rank_correlation: rank_correlation_against_index(
            &rows.iter().map(|row| row.edge_bps.mean).collect::<Vec<_>>(),
        ),
        rows,
    }
}

// ---------------------------------------------------------------------------
// 4. Selective-trading screening, arithmetic only
// ---------------------------------------------------------------------------

/// Sums a selective sign-following rule's break-even cost is a function of.
///
/// The position is `sign(mu_hat)` on a selected bar and `0` otherwise, held into the bar, so
/// it lives in `{-1, 0, +1}` and no leverage, Kelly solve or sizing rule appears anywhere.
/// Turnover is the one-way notional `|p_t - p_{t-1}|` accumulated over EVERY bar including
/// the unselected ones, plus the end-of-window unwind — the same convention
/// [`super::trade_bench::PolicyStats::turnover`] charges, so the two are comparable. Charging
/// turnover only on selected bars would silently make every exit free, which is exactly the
/// error a selective rule is most tempting to make.
#[derive(Clone, Copy, Debug, Default)]
struct Selective {
    /// Bars the rule trades.
    traded: f64,
    /// `sum sign(mu_hat) r` over traded bars, in nats.
    edge: f64,
    /// `sum |p_t - p_{t-1}|` over every bar, plus the final unwind.
    turnover: f64,
    /// Every bar, the participation denominator.
    all: f64,
    /// Traded bars with `sign(mu_hat) r > 0`.
    hits: f64,
    /// Traded bars with a realized direction to be right about.
    directed: f64,
    /// `sum sigma_hat` over TRADED bars.
    sigma_traded: f64,
    /// `sum sigma_hat` over EVERY bar.
    sigma_all: f64,
}

impl Selective {
    fn participation(&self) -> f64 {
        ratio(self.traded, self.all)
    }
    fn edge_bps(&self) -> f64 {
        1.0e4 * ratio(self.edge, self.traded)
    }
    fn turnover_per_traded_bar(&self) -> f64 {
        ratio(self.turnover, self.traded)
    }
    /// Cost per unit of one-way notional at which the gross edge is exactly consumed:
    /// `total edge / total turnover`, in bps. Identically
    /// `edge_bps / turnover_per_traded_bar`, so the table is internally consistent by
    /// construction rather than by two computations that happen to agree.
    fn break_even_bps(&self) -> f64 {
        1.0e4 * ratio(self.edge, self.turnover)
    }
    fn hit_rate(&self) -> f64 {
        ratio(self.hits, self.directed)
    }
    /// Mean predicted volatility of the bars the rule TRADES, over the mean predicted
    /// volatility of every bar.
    ///
    /// This is the number that decides whether the break-even comparison is legitimate, and it
    /// is the bar-level analogue of the symbol-level population caveat that governed the whole
    /// cost argument this session. [`MEASURED_COST_BPS_MATCHED`] is matched by SYMBOL-MONTH but
    /// it is UNCONDITIONAL IN THE BAR: it is what these names cost to trade on an average bar.
    /// A confidence selector does not pick average bars. `corr(sigma_hat, |r|)` is large on this
    /// panel, so the top decile of `|mu_hat|` is also the high-volatility tail, and effective
    /// spread widens with volatility. Comparing a volatility-selected edge against a
    /// volatility-unconditional cost is therefore the same category error one level down, and
    /// this ratio is what makes it visible instead of silent.
    ///
    /// It is a PROXY, not a measured cost: it assumes effective spread scales linearly in
    /// volatility, which is the textbook inventory-risk relation and is the reason the Roll
    /// estimator is a volatility estimator in disguise, but nobody fitted that exponent on this
    /// corpus. It is reported as a multiplier on the reference so a reader can apply their own
    /// exponent, and never folded silently into the break-even.
    fn sigma_ratio(&self) -> f64 {
        let traded_mean = ratio(self.sigma_traded, self.traded);
        let all_mean = ratio(self.sigma_all, self.all);
        ratio(traded_mean, all_mean)
    }
}

impl BlockSums for Selective {
    fn absorb(&mut self, other: &Self) {
        self.traded += other.traded;
        self.edge += other.edge;
        self.turnover += other.turnover;
        self.all += other.all;
        self.hits += other.hits;
        self.directed += other.directed;
        self.sigma_traded += other.sigma_traded;
        self.sigma_all += other.sigma_all;
    }
    fn count(&self) -> f64 {
        self.all
    }
}

/// One confidence threshold's screening arithmetic.
#[derive(Clone, Debug)]
pub struct SelectiveRow {
    /// Trade only bars in decile `>= threshold_decile` of the selector.
    pub threshold_decile: usize,
    /// The selector value the threshold sits at; `-inf` for "trade everything".
    pub threshold_value: f64,
    pub participation: Dispersion,
    pub edge_bps: Dispersion,
    pub turnover_per_traded_bar: Dispersion,
    pub break_even_bps: Dispersion,
    pub hit_rate: Dispersion,
    /// Mean `sigma_hat` on traded bars over mean `sigma_hat` on all bars. See
    /// [`Selective::sigma_ratio`]: the multiplier the unconditional reference cost has to be
    /// scaled by before the break-even comparison is between like and like.
    pub sigma_ratio: Dispersion,
}

/// Break-even cost as a function of how selective the rule is.
#[derive(Clone, Debug)]
pub struct SelectiveTable {
    pub selector: &'static str,
    pub rows: Vec<SelectiveRow>,
    /// The MEASURED, impact-free reference cost the verdict is stated against.
    pub reference_cost_bps: f64,
}

impl SelectiveTable {
    /// The best break-even any threshold achieves.
    pub fn best(&self) -> Option<&SelectiveRow> {
        self.rows
            .iter()
            .filter(|row| row.break_even_bps.mean.is_finite())
            .max_by(|a, b| a.break_even_bps.mean.total_cmp(&b.break_even_bps.mean))
    }

    /// SELECTION-INFLATED. This is the maximum over `2 * DECILES` correlated cells - two selectors
    /// times ten thresholds - so its point estimate is biased UPWARD by roughly +0.4 to +1.9 of its
    /// own standard errors and its marginal interval UNDER-covers the true best. It is the right
    /// screening number and the wrong viability number, and because a reader of the verdict string
    /// gets it whether or not they read this comment, [`Self::verdict`] says so in the sentence
    /// that quotes it. An IRC convention not to quote it does not survive this session; the string
    /// does.
    ///
    /// True only when the best break-even's POINT estimate clears the reference cost. The
    /// interval is reported beside it and is what a decision should use; this is the headline
    /// the screening question asks for.
    ///
    /// UNCONDITIONAL in the bar. See [`Self::clears_volatility_scaled_reference`] for the
    /// comparison that survives the selector picking volatile bars.
    pub fn clears_reference(&self) -> bool {
        self.best()
            .is_some_and(|row| row.break_even_bps.mean >= self.reference_cost_bps)
    }

    /// The reference cost scaled by the best row's measured volatility ratio: what the same
    /// names plausibly cost on the bars this rule actually selects, under a linear
    /// spread-in-volatility proxy.
    pub fn volatility_scaled_reference_bps(&self) -> f64 {
        self.best()
            .map_or(f64::NAN, |row| self.reference_cost_bps * row.sigma_ratio.mean)
    }

    /// The honest form of the screening question: does the best break-even clear the reference
    /// cost after that cost is put on the same kind of bar the rule trades?
    pub fn clears_volatility_scaled_reference(&self) -> bool {
        self.best().is_some_and(|row| {
            row.break_even_bps.mean >= self.reference_cost_bps * row.sigma_ratio.mean
        })
    }
}

fn selective_table(
    panel: &SkillPanel,
    selector_name: &'static str,
    selector: impl Fn(&SkillBar) -> f64,
    cuts: &SkillCutpoints,
) -> SelectiveTable {
    // The cutpoints come from the confidence curve's own function, so the two tables partition
    // the bars identically and a decile means one thing in both.
    let cutpoints = cuts.of(selector_name).to_vec();
    let rows = (0..DECILES)
        .map(|threshold| {
            let mut grouped: BTreeMap<u64, Selective> = BTreeMap::new();
            for window in &panel.windows {
                let entry = grouped.entry(window.block).or_default();
                let mut held = 0.0f64;
                for (index, bar) in window.bars.iter().enumerate() {
                    let value = selector(bar);
                    let predicted = direction(bar.mu);
                    let selected = value.is_finite()
                        && decile_of(&cutpoints, value) >= threshold
                        && predicted != 0;
                    let position = if selected { f64::from(predicted) } else { 0.0 };
                    entry.all += 1.0;
                    entry.sigma_all += bar.sigma;
                    entry.turnover += (position - held).abs();
                    if index + 1 == window.bars.len() {
                        entry.turnover += position.abs();
                    }
                    held = position;
                    if selected {
                        entry.traded += 1.0;
                        entry.sigma_traded += bar.sigma;
                        entry.edge += position * bar.r;
                        if direction(bar.r) != 0 {
                            entry.directed += 1.0;
                            if position * bar.r > 0.0 {
                                entry.hits += 1.0;
                            }
                        }
                    }
                }
            }
            let blocks: Vec<Selective> = grouped.into_values().collect();
            SelectiveRow {
                threshold_decile: threshold,
                threshold_value: if threshold == 0 {
                    f64::NEG_INFINITY
                } else {
                    cutpoints[threshold - 1]
                },
                participation: blocked(&blocks, Selective::participation),
                edge_bps: blocked(&blocks, Selective::edge_bps),
                turnover_per_traded_bar: blocked(&blocks, Selective::turnover_per_traded_bar),
                break_even_bps: blocked(&blocks, Selective::break_even_bps),
                hit_rate: blocked(&blocks, Selective::hit_rate),
                sigma_ratio: blocked(&blocks, Selective::sigma_ratio),
            }
        })
        .collect();
    SelectiveTable {
        selector: selector_name,
        rows,
        reference_cost_bps: MEASURED_COST_BPS_MATCHED,
    }
}

// ---------------------------------------------------------------------------
// The whole profile
// ---------------------------------------------------------------------------

pub const SELECTOR_ABS_MU: &str = "|mu_hat|";
pub const SELECTOR_SHARPE: &str = "|mu_hat|/sigma_hat";

/// Everything this module measures on one panel.
#[derive(Clone, Debug)]
pub struct SkillProfile {
    pub label: String,
    pub bars: usize,
    pub windows: usize,
    pub blocks: usize,
    pub symbols: usize,
    pub confusion: ConfusionReport,
    pub ic: IcDecomposition,
    pub magnitude: MagnitudeIc,
    pub auc: Dispersion,
    pub abs_mu_curve: ConfidenceCurve,
    pub sharpe_curve: ConfidenceCurve,
    pub abs_mu_selective: SelectiveTable,
    pub sharpe_selective: SelectiveTable,
}

/// `|mu_hat|`: the raw magnitude of the predicted conditional mean.
fn selector_abs_mu(bar: &SkillBar) -> f64 {
    bar.mu.abs()
}

/// `|mu_hat| / sigma_hat`, the model's own predicted per-bar Sharpe.
///
/// Scale-free, so it ranks a quiet name's confident call above a volatile name's ordinary one,
/// which is what a selective policy has to do and what `|mu_hat|` alone cannot. A non-positive
/// `sigma_hat` is a degenerate belief rather than infinite confidence, so it is excluded rather
/// than ranked at the top.
fn selector_sharpe(bar: &SkillBar) -> f64 {
    if bar.sigma > 0.0 {
        bar.mu.abs() / bar.sigma
    } else {
        f64::NAN
    }
}

impl SkillProfile {
    /// Measure every reported statistic on one panel.
    pub fn measure(panel: &SkillPanel, label: impl Into<String>, cuts: &SkillCutpoints) -> Self {
        Self {
            label: label.into(),
            bars: panel.bars(),
            windows: panel.windows.len(),
            blocks: panel.block_count(),
            symbols: panel.symbol_count(),
            confusion: confusion_report(panel),
            ic: ic_decomposition(panel),
            magnitude: magnitude_ic(panel),
            auc: auc(panel),
            abs_mu_curve: confidence_curve(panel, SELECTOR_ABS_MU, selector_abs_mu, cuts),
            sharpe_curve: confidence_curve(panel, SELECTOR_SHARPE, selector_sharpe, cuts),
            abs_mu_selective: selective_table(panel, SELECTOR_ABS_MU, selector_abs_mu, cuts),
            sharpe_selective: selective_table(panel, SELECTOR_SHARPE, selector_sharpe, cuts),
        }
    }

    /// Is the DIRECTIONAL signal concentrated in high-confidence bars?
    ///
    /// The criterion is a POSITIVE paired top-minus-bottom difference clearing two of its own
    /// blocked standard errors on the BALANCED-ACCURACY axis, for either selector. Two SE rather
    /// than "the interval excludes zero" because the two are nearly the same statement and the
    /// SE form states an effect size; the intervals are printed beside it either way.
    ///
    /// # Why balanced accuracy and NOT raw accuracy or edge
    ///
    /// Both of the obvious axes are confounded, each by a mechanism this module elsewhere
    /// insists on. Raw accuracy inherits the bucket's own class balance: a CONSTANT
    /// "always down" predictor scored on a bucket 0 that is 50/50 and a bucket 9 that is 20/80
    /// up/down reads `0.5000` against `0.8000`, a resolvable `+0.30` with literally zero
    /// directional skill, which is exactly the imbalance objection that motivates the whole
    /// module and which [`DecileRow::base_rate_up`] exists to expose. The edge axis is worse
    /// because it is confounded independently: `E[sign(mu_hat) r | decile]` is approximately
    /// `(2a - 1) E[|r| | decile]`, so it is the accuracy axis MULTIPLIED by the bucket's mean
    /// absolute return, and since the selector is `|mu_hat|` and `corr(|mu_hat|, |r|)` is large
    /// on this panel, the top bucket's edge is larger than the bottom's whether or not its
    /// SIGN is any better. A fixture with identical accuracy and identical up-rate in both
    /// buckets and only `|r|` tripled resolves on the edge axis at 13.8 sigma while balanced
    /// accuracy stays flat, so the edge axis would have answered the magnitude question while
    /// printing the answer to the direction one.
    ///
    /// Balanced accuracy is `(TPR + TNR)/2`, on which every constant rule scores exactly `0.5`
    /// at any class balance, and it is invariant to any monotone rescaling of `r`. It is
    /// therefore the only decile axis that answers the question actually asked. The accuracy,
    /// edge and IC axes are still measured and printed; they are description, not the criterion.
    ///
    /// The sign is checked. [`resolved`] tests a magnitude, so a resolvable NEGATIVE difference
    /// — high-confidence bars being reliably WORSE, a real and reportable finding — must not be
    /// allowed to return `true` here.
    pub fn concentrated(&self) -> bool {
        [&self.abs_mu_curve, &self.sharpe_curve].iter().any(|curve| {
            resolved(&curve.top_minus_bottom_balanced)
                && curve.top_minus_bottom_balanced.mean > 0.0
        })
    }

    /// The mirror of [`Self::concentrated`]: is the top decile resolvably WORSE than the bottom
    /// on the imbalance-proof axis? Reported so an inverted curve is stated as the finding it is
    /// rather than as an absence of power.
    pub fn anti_concentrated(&self) -> bool {
        [&self.abs_mu_curve, &self.sharpe_curve].iter().any(|curve| {
            resolved(&curve.top_minus_bottom_balanced)
                && curve.top_minus_bottom_balanced.mean < 0.0
        })
    }

    /// Does the paired top-minus-bottom IC resolve POSITIVE for either selector? The second
    /// scale-free axis, reported beside [`Self::concentrated`] rather than folded into it, so
    /// the binary is one STATISTIC rather than four - but it is still an `any` over the two
    /// SELECTORS, so the multiplicity is reduced from four to two and is not eliminated. A reader
    /// wanting a single pre-registered test should read the `|mu_hat|/sigma_hat` row alone, which
    /// is the better-motivated selector and was named as such before either was measured.
    pub fn concentrated_by_ic(&self) -> bool {
        [&self.abs_mu_curve, &self.sharpe_curve]
            .iter()
            .any(|curve| resolved(&curve.top_minus_bottom_ic) && curve.top_minus_bottom_ic.mean > 0.0)
    }

    /// Does the model beat the best CONSTANT predictor on the imbalance-proof score?
    pub fn beats_chance(&self) -> bool {
        resolved(&self.confusion.balanced_over_chance)
            && self.confusion.balanced_over_chance.mean > 0.0
    }

    /// The best break-even across both selectors, with the selector that achieved it.
    pub fn best_break_even(&self) -> Option<(&'static str, &SelectiveRow)> {
        let candidates = [
            (SELECTOR_ABS_MU, self.abs_mu_selective.best()),
            (SELECTOR_SHARPE, self.sharpe_selective.best()),
        ];
        candidates
            .into_iter()
            .filter_map(|(name, row)| row.map(|row| (name, row)))
            .max_by(|a, b| {
                a.1.break_even_bps
                    .mean
                    .total_cmp(&b.1.break_even_bps.mean)
            })
    }

    /// The plain-language verdict, so the numbers cannot be quoted without it.
    ///
    /// Three sentences answering exactly the three questions asked, then the screening
    /// comparison against [`MEASURED_COST_BPS_MATCHED`], stated with its matched provenance in
    /// the same sentence as the number rather than in a footnote a reader can skip.
    pub fn verdict(&self) -> String {
        let balanced = &self.confusion.balanced_accuracy;
        let pooled = self.ic.pooled_pearson.mean;
        let within = self.ic.within_mean.mean;
        let standardized = self.ic.standardized_pearson.mean;
        // "Much larger" is fixed at 2x rather than left to the reader, and it is a comparison
        // of MAGNITUDES so a sign flip between the two counts as maximal disagreement.
        //
        // Both operands are checked FINITE first. `f64::max` ignores NaN, so an unguarded
        // `standardized.abs().max(f64::MIN_POSITIVE)` turns a NaN standardized IC — which is
        // exactly what a head collapsed to a constant `mu_hat` inside every block produces, the
        // failure mode `blocks_dropped` exists to record — into `2.2e-308` and thereby into the
        // STRONGEST available claim, asserting that the pooled figure is cross-sectional while
        // printing NaN for the number it claims to have compared against.
        let comparable = pooled.is_finite() && standardized.is_finite();
        let cross_sectional =
            comparable && pooled.abs() > 2.0 * standardized.abs().max(f64::MIN_POSITIVE);
        let mut out = String::new();
        out.push_str(&format!(
            "DIRECTIONAL SKILL BEYOND THE BASE RATE: {}. Raw accuracy is {:.4} against an \
             up-bar base rate of {:.4} over classified bars ({:.4} over ALL bars, which is \
             exactly what buy & hold's hit field reports), so the best CONSTANT rule scores \
             {:.4} and raw accuracy is the wrong comparison; the model is {:+.4} +/- {:.4} \
             against it. On the imbalance-proof score the model is at {:.4} (blocked 95% CI \
             {:.4}..{:.4}) against 0.5 for every constant rule, a paired excess of {:+.4} +/- \
             {:.4}. ",
            if self.beats_chance() {
                if self.confusion.balanced_over_chance.mean >= 0.02 {
                    "YES"
                } else {
                    "YES, and it is small"
                }
            } else if resolved(&self.confusion.balanced_over_chance)
                && self.confusion.balanced_over_chance.mean < 0.0
            {
                "NO, and the model is resolvably WORSE than a coin flip on the imbalance-proof \
                 score - this is a measured negative result, not an absence of power"
            } else {
                "NOT RESOLVED at this sample size"
            },
            self.confusion.accuracy.mean,
            self.confusion.base_rate_up.mean,
            self.confusion.up_rate_all_bars.mean,
            self.confusion.majority_accuracy.mean,
            self.confusion.accuracy_over_majority.mean,
            self.confusion.accuracy_over_majority.se,
            balanced.mean,
            balanced.ci_low,
            balanced.ci_high,
            self.confusion.balanced_over_chance.mean,
            self.confusion.balanced_over_chance.se,
        ));
        out.push_str(&format!(
            "The POOLED IC is {pooled:+.4} (implying R^2 {:.4}) while the within-symbol-month \
             STANDARDIZED IC is {standardized:+.4} and the equal-weighted WITHIN-NAME IC is \
             {within:+.4}, so {}. The magnitude channel measures {:+.4} pooled and {:+.4} \
             standardized for corr(|mu_hat|, |r|). AUC is {:.4} +/- {:.4}. ",
            self.ic.pooled_r2,
            if !comparable {
                "one of the two is UNMEASURED (non-finite), so the cross-sectional question \
                 cannot be answered from this panel and no comparison is asserted"
            } else if cross_sectional {
                "the pooled number is MORE THAN TWICE the heteroskedasticity-free one and the \
                 apparent skill it reports is predominantly cross-sectional scale and \
                 volatility prediction, NOT direction"
            } else {
                "the three agree in order of magnitude and the pooled number is NOT an \
                 artifact of cross-sectional scale"
            },
            self.magnitude.abs_mu_vs_abs_r.mean,
            self.magnitude.abs_mu_vs_abs_r_standardized.mean,
            self.auc.mean,
            self.auc.se,
        ));
        out.push_str(&format!(
            "CONCENTRATION IN HIGH-CONFIDENCE BARS: {}. The criterion is the paired \
             top-minus-bottom BALANCED accuracy, which is the only decile axis that is both \
             imbalance-proof and invariant to the bucket's return scale: raw accuracy inherits \
             the bucket's class balance and the edge axis is that accuracy multiplied by the \
             bucket's mean |r|, so on a panel where |mu_hat| tracks volatility both would rise \
             with confidence whether or not the SIGN got better. Balanced, by \
             {SELECTOR_SHARPE}: {:+.4} +/- {:.4}; by {SELECTOR_ABS_MU}: {:+.4} +/- {:.4}. The \
             scale-free IC axis {} ({:+.4} +/- {:.4} and {:+.4} +/- {:.4}). Reported as \
             DESCRIPTION and not as the criterion, the confounded axes read: accuracy {:+.4} \
             +/- {:.4} and edge {:+.3} +/- {:.3} bps by {SELECTOR_SHARPE}, {:+.4} +/- {:.4} \
             and {:+.3} +/- {:.3} bps by {SELECTOR_ABS_MU}. ",
            if self.concentrated() {
                "YES - the paired top-minus-bottom BALANCED accuracy is resolvably POSITIVE, so \
                 a SELECTIVE rule has a genuinely more reliable SIGN to select on"
            } else if self.anti_concentrated() {
                "INVERTED - the top decile is resolvably WORSE than the bottom on the \
                 imbalance-proof axis, so confidence is anti-predictive of reliability"
            } else {
                "NO - the curve is FLAT within its blocked uncertainty on the imbalance-proof \
                 axis, so no subset of bars carries a materially more reliable sign and the \
                 magnitude story is all there is"
            },
            self.sharpe_curve.top_minus_bottom_balanced.mean,
            self.sharpe_curve.top_minus_bottom_balanced.se,
            self.abs_mu_curve.top_minus_bottom_balanced.mean,
            self.abs_mu_curve.top_minus_bottom_balanced.se,
            if self.concentrated_by_ic() {
                "AGREES"
            } else {
                "does NOT resolve positive"
            },
            self.sharpe_curve.top_minus_bottom_ic.mean,
            self.sharpe_curve.top_minus_bottom_ic.se,
            self.abs_mu_curve.top_minus_bottom_ic.mean,
            self.abs_mu_curve.top_minus_bottom_ic.se,
            self.sharpe_curve.top_minus_bottom_accuracy.mean,
            self.sharpe_curve.top_minus_bottom_accuracy.se,
            self.sharpe_curve.top_minus_bottom_edge_bps.mean,
            self.sharpe_curve.top_minus_bottom_edge_bps.se,
            self.abs_mu_curve.top_minus_bottom_accuracy.mean,
            self.abs_mu_curve.top_minus_bottom_accuracy.se,
            self.abs_mu_curve.top_minus_bottom_edge_bps.mean,
            self.abs_mu_curve.top_minus_bottom_edge_bps.se,
        ));
        out.push_str(&format!(
            "WHAT THE SELECTION IS WORTH, which is a different question from whether the sign \
             improves: the TOP-DECILE EDGE is {:.3}x the all-bar edge by {SELECTOR_SHARPE} \
             (blocked 95% CI {:.3}..{:.3}) and {:.3}x by {SELECTOR_ABS_MU} ({:.3}..{:.3}). \
             Accuracy is not money - a rule can gain fifteen points of directional accuracy on \
             bars whose moves cannot pay a spread and earn nothing - so it is this ratio, not the \
             accuracy gain, that composes with a cost reduction from an orthogonal axis. ",
            self.sharpe_curve.top_edge_multiple.mean,
            self.sharpe_curve.top_edge_multiple.ci_low,
            self.sharpe_curve.top_edge_multiple.ci_high,
            self.abs_mu_curve.top_edge_multiple.mean,
            self.abs_mu_curve.top_edge_multiple.ci_low,
            self.abs_mu_curve.top_edge_multiple.ci_high,
        ));
        out.push_str(&format!(
            "FLAT-CORRECTED, and this is the number to compose with anything: per MOVING bar the \
             top-decile edge is {:.3}x the all-bar edge by {SELECTOR_SHARPE} ({:.3}..{:.3}) and \
             {:.3}x by {SELECTOR_ABS_MU} ({:.3}..{:.3}). The uncorrected pair above divides two \
             quantities attenuated by DIFFERENT flat shares and is biased upward. ",
            self.sharpe_curve.top_edge_multiple_moving.mean,
            self.sharpe_curve.top_edge_multiple_moving.ci_low,
            self.sharpe_curve.top_edge_multiple_moving.ci_high,
            self.abs_mu_curve.top_edge_multiple_moving.mean,
            self.abs_mu_curve.top_edge_multiple_moving.ci_low,
            self.abs_mu_curve.top_edge_multiple_moving.ci_high,
        ));
        match self.best_break_even() {
            Some((selector, row)) => out.push_str(&format!(
                "SELECTIVE TRADING AGAINST {MEASURED_COST_BPS_MATCHED:.3} BPS: the best \
                 break-even any threshold reaches is {:.3} bps (blocked 95% CI {:.3}..{:.3}, \
                 which is that ROW's OWN marginal interval and is NOT corrected for this being \
                 the ARGMAX over {} in-sample thresholds across two selectors - the maximum of \
                 correlated rows sits roughly half a standard error above the truth, so the \
                 point estimate is biased UPWARD and the interval under-covers the true best) \
                 at decile {} of {selector}, participation {:.3}, mean edge {:.3} bps per \
                 traded bar on {:.3} units of one-way turnover, so it {} the MATCHED, MEASURED, \
                 impact-free one-way cost of these very symbol-months \
                 ({MEASURED_COST_BPS_MATCHED:.3} bps equal-weighted mean over the 256 names, \
                 {MEASURED_COST_BPS_MATCHED_MEDIAN:.3} bps median; the sized figure with an \
                 UNFITTED k = 0.5 impact term at 1% of ADV is \
                 {SIZED_COST_BPS_MATCHED_AT_1PCT_ADV:.3} bps, and the universe equal-weighted \
                 measured cost is {MEASURED_COST_BPS_UNIVERSE:.3} bps, so this draw is mildly \
                 liquidity-favoured rather than a mega-cap subset). The cost side is MATCHED BY \
                 SYMBOL-MONTH - the same 256 windows over the same 256 symbols at the same \
                 896-bar context, priced against the real 5,297-symbol calibration - but it is \
                 UNCONDITIONAL IN THE BAR, and that is the remaining category error: the \
                 selected bars carry {:.3}x the panel's mean predicted volatility, so under a \
                 LINEAR spread-in-volatility proxy the same names cost {:.3} bps on the bars \
                 this rule actually trades and the rule {} on that basis. The proxy exponent is \
                 assumed, not fitted on this corpus, so the volatility-conditioned cost is the \
                 first thing a full experiment must MEASURE rather than scale. This is screening \
                 arithmetic on a unit-sized sign-following rule with no cost charged and no \
                 policy built.",
                row.break_even_bps.mean,
                row.break_even_bps.ci_low,
                row.break_even_bps.ci_high,
                DECILES * 2,
                row.threshold_decile,
                row.participation.mean,
                row.edge_bps.mean,
                row.turnover_per_traded_bar.mean,
                if row.break_even_bps.mean >= MEASURED_COST_BPS_MATCHED {
                    "CLEARS"
                } else {
                    "does NOT clear"
                },
                row.sigma_ratio.mean,
                MEASURED_COST_BPS_MATCHED * row.sigma_ratio.mean,
                if row.break_even_bps.mean >= MEASURED_COST_BPS_MATCHED * row.sigma_ratio.mean {
                    "STILL CLEARS"
                } else {
                    "does NOT clear"
                },
            )),
            None => out.push_str(
                "SELECTIVE TRADING: no threshold produced a finite break-even, so the screening \
                 question cannot be answered from this panel.",
            ),
        }
        out
    }

    /// Every measured number, as printable lines.
    pub fn report_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(format!(
            "  skill audit of {}: {} bars over {} windows, {} (symbol, month) blocks, {} symbols",
            self.label, self.bars, self.windows, self.blocks, self.symbols
        ));

        let c = &self.confusion;
        lines.push(
            "  --- 1. confusion of sign(mu_hat) against sign(r), flat bars excluded from the \
             2x2 ---"
                .to_owned(),
        );
        lines.push(format!(
            "    pred up   : realized up {:>9.0}  realized down {:>9.0}   precision {:.4}",
            c.counts[PRED_UP_REAL_UP], c.counts[PRED_UP_REAL_DOWN], c.precision_up.mean
        ));
        lines.push(format!(
            "    pred down : realized up {:>9.0}  realized down {:>9.0}   precision {:.4}",
            c.counts[PRED_DOWN_REAL_UP], c.counts[PRED_DOWN_REAL_DOWN], c.precision_down.mean
        ));
        lines.push(format!(
            "    excluded  : {:.0} flat bars (r == 0), {:.0} undirected bars (mu_hat == 0), of \
             {:.0} total",
            c.flat_bars, c.undirected_bars, c.all_bars
        ));
        for (name, value) in [
            ("accuracy (raw)", &c.accuracy),
            ("accuracy on up-bars = recall up", &c.accuracy_on_up),
            ("accuracy on down-bars = recall down", &c.accuracy_on_down),
            ("BALANCED accuracy", &c.balanced_accuracy),
            ("up-bar base rate, classified bars", &c.base_rate_up),
            ("up-bar rate, ALL bars = buy&hold hit", &c.up_rate_all_bars),
            ("baseline: always up", &c.always_up_accuracy),
            ("baseline: always down", &c.always_down_accuracy),
            ("baseline: majority class", &c.majority_accuracy),
            ("model minus majority, PAIRED", &c.accuracy_over_majority),
            ("balanced minus 0.5, PAIRED", &c.balanced_over_chance),
            ("trade_bench hit rate on sign(f*)", &c.kelly_sign_hit_rate),
            ("sign(f*) != sign(mu_hat) share", &c.kelly_sign_disagreement),
        ] {
            lines.push(format!("    {name:<40} {}", show(value, "bars")));
        }

        lines.push("  --- 2. information coefficient, decomposed ---".to_owned());
        // Units differ INSIDE this block: the pooled and standardized ICs resample bars, the
        // within-name distribution resamples SYMBOLS. Stated per line rather than once in a
        // header, because the two sit adjacent and differ by three orders of magnitude.
        for (name, unit, value) in [
            (
                "POOLED Pearson corr(mu_hat, r)",
                "bars",
                &self.ic.pooled_pearson,
            ),
            ("POOLED Spearman", "bars", &self.ic.pooled_spearman),
            ("WITHIN-NAME median IC", "symbols", &self.ic.within_median),
            (
                "WITHIN-NAME equal-weighted mean IC",
                "symbols",
                &self.ic.within_mean,
            ),
            (
                "WITHIN-NAME fraction positive",
                "symbols",
                &self.ic.within_fraction_positive,
            ),
            (
                "STANDARDIZED Pearson, z within block",
                "bars",
                &self.ic.standardized_pearson,
            ),
            (
                "STANDARDIZED Spearman, rank within block",
                "bars",
                &self.ic.standardized_spearman,
            ),
        ] {
            lines.push(format!("    {name:<40} {}", show(value, unit)));
        }
        lines.push(format!(
            "    {:<40} {:.6}  (against the measured Mincer-Zarnowitz R^2)",
            "implied pooled R^2 = corr^2", self.ic.pooled_r2
        ));
        lines.push(format!(
            "    {:<40} IQR {:+.4}..{:+.4} over {} symbols, {} dropped as degenerate",
            "WITHIN-NAME spread",
            self.ic.within_q1,
            self.ic.within_q3,
            self.ic.symbols_measured,
            self.ic.symbols_dropped
        ));
        lines.push(format!(
            "    {:<40} {} blocks dropped as degenerate",
            "STANDARDIZED coverage", self.ic.blocks_dropped
        ));
        for (name, value) in [
            (
                "MAGNITUDE corr(|mu_hat|,|r|) pooled",
                &self.magnitude.abs_mu_vs_abs_r,
            ),
            (
                "MAGNITUDE corr(|mu_hat|,|r|) standardized",
                &self.magnitude.abs_mu_vs_abs_r_standardized,
            ),
            (
                "MAGNITUDE corr(sigma_hat,|r|) pooled",
                &self.magnitude.sigma_vs_abs_r,
            ),
            (
                "MAGNITUDE corr(sigma_hat,|r|) standardized",
                &self.magnitude.sigma_vs_abs_r_standardized,
            ),
        ] {
            lines.push(format!("    {name:<40} {}", show(value, "bars")));
        }

        lines.push("  --- 5. AUC of mu_hat as a ranker of sign(r) ---".to_owned());
        lines.push(format!(
            "    {:<40} {}",
            "AUC",
            show(&self.auc, "classified bars")
        ));

        for curve in [&self.abs_mu_curve, &self.sharpe_curve] {
            lines.push(format!(
                "    per-block occupancy  decile 0 [min {:.0} med {:.0} max {:.0}] empty in {} of \
                 {} blocks; decile 9 [min {:.0} med {:.0} max {:.0}] empty in {} blocks",
                curve.occupancy_bottom[0],
                curve.occupancy_bottom[1],
                curve.occupancy_bottom[2],
                curve.blocks_missing_bottom,
                curve.blocks,
                curve.occupancy_top[0],
                curve.occupancy_top[1],
                curve.occupancy_top[2],
                curve.blocks_missing_top,
            ));
            lines.push(format!(
                "    top decile holds {} distinct symbols, Herfindahl {:.4} (1/{:.1} effective \
                 names); mean sigma_hat {:.2} bps in decile 9 against {:.2} bps in decile 0",
                curve.top_distinct_symbols,
                curve.top_herfindahl,
                1.0 / curve.top_herfindahl,
                curve.sigma_top_bps,
                curve.sigma_bottom_bps,
            ));
            lines.push(format!(
                "    catch-all indicators: |mu_hat| over the {} bps per-bar interior BOUND, a \
                 LOWER BOUND on the contaminated share (large is conclusive, small is NOT \
                 exculpatory - catch-all mass can be substantial while cancelling below the bound) \
                 decile 0 {} decile 9 {}; sigma_hat over the same bound {} and {}; sigma_hat over \
                 the {} bps marginal-interior RMS, a TYPICAL VALUE and NOT a bound, withdrawn as \
                 one by its author and retained only as a reference {} and {}",
                show_option(curve.interior_bound_bps),
                show_fraction(curve.rows[0].over_bound_mu.mean),
                show_fraction(curve.rows[DECILES - 1].over_bound_mu.mean),
                show_fraction(curve.rows[0].over_bound_sigma.mean),
                show_fraction(curve.rows[DECILES - 1].over_bound_sigma.mean),
                show_option(curve.interior_marginal_rms_bps),
                show_fraction(curve.rows[0].over_reference_sigma.mean),
                show_fraction(curve.rows[DECILES - 1].over_reference_sigma.mean),
            ));
            lines.push(format!(
                "    top decile composition (symbol index: share) {}",
                curve
                    .top_composition
                    .iter()
                    .map(|(symbol, share)| format!("{symbol}:{:.3}", share))
                    .collect::<Vec<_>>()
                    .join(" ")
            ));
            lines.push(format!(
                "  --- 3. confidence curve by decile of {}, {} bars unrankable ---",
                curve.selector, curve.excluded
            ));
            lines.push(
                "    decile      bars   accuracy   balanced        IC   edge bps  edge/moving  flat%    up rate"
                    .to_owned(),
            );
            for (index, row) in curve.rows.iter().enumerate() {
                lines.push(format!(
                    "    {index:>6} {:>9} {:>10.4} {:>10.4} {:>9.4} {:>10.3} {:>12.3} {:>6.2} {:>10.4}",
                    row.bars,
                    row.accuracy.mean,
                    row.balanced_accuracy.mean,
                    row.ic.mean,
                    row.edge_bps.mean,
                    row.edge_bps_moving.mean,
                    // The attenuation factor of the column to its left: a flat bar contributes
                    // zero to `edge` and one to its denominator, and the share differs by decile.
                    100.0 * ratio(row.flat_bars as f64, row.bars as f64),
                    row.base_rate_up.mean,
                ));
            }
            for (name, value) in [
                // The CRITERION axis is marked, because the two above it are confounded by the
                // bucket's class balance and by its mean |r| respectively and are description.
                (
                    "top-bottom accuracy, PAIRED (confounded)",
                    &curve.top_minus_bottom_accuracy,
                ),
                (
                    "top-bottom edge bps, PAIRED (confounded)",
                    &curve.top_minus_bottom_edge_bps,
                ),
                (
                    "top-bottom BALANCED, PAIRED <- CRITERION",
                    &curve.top_minus_bottom_balanced,
                ),
                ("top-bottom IC, PAIRED", &curve.top_minus_bottom_ic),
                // ECONOMICS, not direction: what the selection multiplies the money by.
                ("top-decile EDGE / all-bar EDGE", &curve.top_edge_multiple),
                // FLAT-CORRECTED, and this is the honest one. The raw ratio above divides two
                // quantities attenuated by DIFFERENT flat shares - the bottom bucket of a
                // volatility-scaled selector holds more unmoved bars than the top - so it is biased
                // upward by (1 - flat_top) / (1 - flat_all). This version divides per MOVING bar
                // on both sides and is the multiplier that may be composed with anything.
                (
                    "top-decile EDGE / all-bar, per MOVING bar",
                    &curve.top_edge_multiple_moving,
                ),
            ] {
                lines.push(format!("    {name:<40} {}", show(value, "bars")));
            }
            lines.push(format!(
                "    {:<40} accuracy {:+.3}, edge {:+.3} (shape only, no interval)",
                "decile-index rank correlation",
                curve.accuracy_rank_correlation,
                curve.edge_rank_correlation
            ));
        }

        for table in [&self.abs_mu_selective, &self.sharpe_selective] {
            lines.push(format!(
                "  --- 4. selective screening by {} against {:.3} bps MEASURED one-way ---",
                table.selector, table.reference_cost_bps
            ));
            lines.push(
                "    from     partic.   edge bps   turnover  break-even          95% CI      hit  \
                 sigma x"
                    .to_owned(),
            );
            for row in &table.rows {
                lines.push(format!(
                    "    {:>4} {:>11.4} {:>10.3} {:>10.4} {:>11.3} {:>8.3}..{:<8.3} {:>7.4} \
                     {:>7.3}",
                    row.threshold_decile,
                    row.participation.mean,
                    row.edge_bps.mean,
                    row.turnover_per_traded_bar.mean,
                    row.break_even_bps.mean,
                    row.break_even_bps.ci_low,
                    row.break_even_bps.ci_high,
                    row.hit_rate.mean,
                    row.sigma_ratio.mean,
                ));
            }
            match table.best() {
                Some(row) => {
                    lines.push(format!(
                        "    best break-even {:.3} bps at decile {}: {} the UNCONDITIONAL \
                         measured {:.3} bps",
                        row.break_even_bps.mean,
                        row.threshold_decile,
                        if table.clears_reference() {
                            "CLEARS"
                        } else {
                            "below"
                        },
                        table.reference_cost_bps
                    ));
                    lines.push(format!(
                        "    those bars carry {:.3}x the panel's mean sigma_hat ({}), so under a \
                         LINEAR spread-in-volatility proxy the same names cost {:.3} bps there \
                         and the rule {} - this proxy is NOT a measured cost",
                        row.sigma_ratio.mean,
                        row.sigma_ratio,
                        table.volatility_scaled_reference_bps(),
                        if table.clears_volatility_scaled_reference() {
                            "STILL CLEARS"
                        } else {
                            "does NOT clear"
                        },
                    ));
                }
                None => lines.push("    no finite break-even at any threshold".to_owned()),
            }
        }
        lines.push(format!("  VERDICT: {}", self.verdict()));
        lines
    }
}

/// A blocked statistic whose point estimate clears two of its own standard errors.
fn resolved(value: &Dispersion) -> bool {
    value.mean.is_finite()
        && value.se.is_finite()
        && value.se > 0.0
        && value.mean.abs() > 2.0 * value.se
}

// ---------------------------------------------------------------------------
// The chart
// ---------------------------------------------------------------------------

/// Write the decile-indexed curves and the selective screening table as one chart.
///
/// Everything on it is indexed by decile `0..9`, which is what makes one base sufficient: the
/// confidence curve reads left to right as "least to most confident bucket", and the selective
/// table as "trade only deciles at or above this index". Both selectors sit on the same panel
/// because the question is which of the two ranks better, and two charts would put that
/// comparison in the reader's memory instead of in front of their eyes.
///
/// `Symlog`, because a bar count of ~23,000 shares the axis with an accuracy of ~0.5 and a
/// break-even of a few bps. Dropping the counts would leave a reader unable to see that every
/// bucket is equally populated, which is the fact that makes the participation column
/// mechanical rather than a finding.
pub fn write_skill_profile(dir: &Path, profile: &SkillProfile) -> Result<()> {
    type PickDecile = fn(&DecileRow) -> f64;
    type PickSelective = fn(&SelectiveRow) -> f64;
    let mut series = Vec::new();
    for curve in [&profile.abs_mu_curve, &profile.sharpe_curve] {
        let tag = curve.selector;
        for (label, pick) in [
            ("accuracy", (|row: &DecileRow| row.accuracy.mean) as PickDecile),
            ("accuracy CI low", |row: &DecileRow| row.accuracy.ci_low),
            ("accuracy CI high", |row: &DecileRow| row.accuracy.ci_high),
            ("balanced accuracy", |row: &DecileRow| {
                row.balanced_accuracy.mean
            }),
            ("IC", |row: &DecileRow| row.ic.mean),
            ("edge bps", |row: &DecileRow| row.edge_bps.mean),
            ("edge bps CI low", |row: &DecileRow| row.edge_bps.ci_low),
            ("edge bps CI high", |row: &DecileRow| row.edge_bps.ci_high),
            ("bars", |row: &DecileRow| row.bars as f64),
        ] {
            series.push(ReportSeries {
                label: format!("{tag} decile {label}"),
                values: curve.rows.iter().map(|row| pick(row) as f32).collect(),
            });
        }
    }
    for table in [&profile.abs_mu_selective, &profile.sharpe_selective] {
        let tag = table.selector;
        for (label, pick) in [
            (
                "participation",
                (|row: &SelectiveRow| row.participation.mean) as PickSelective,
            ),
            ("selective edge bps", |row: &SelectiveRow| row.edge_bps.mean),
            ("turnover per traded bar", |row: &SelectiveRow| {
                row.turnover_per_traded_bar.mean
            }),
            ("break-even bps", |row: &SelectiveRow| {
                clamp_break_even(row.break_even_bps.mean)
            }),
            ("break-even CI low", |row: &SelectiveRow| {
                clamp_break_even(row.break_even_bps.ci_low)
            }),
            ("break-even CI high", |row: &SelectiveRow| {
                clamp_break_even(row.break_even_bps.ci_high)
            }),
            ("selective hit rate", |row: &SelectiveRow| row.hit_rate.mean),
            ("traded sigma_hat multiple", |row: &SelectiveRow| {
                row.sigma_ratio.mean
            }),
            // The reference cost put on the KIND of bar the threshold selects. Charted beside
            // the break-even because the flat unconditional line invites exactly the comparison
            // that the volatility selection invalidates.
            ("vol-scaled cost bps (LINEAR proxy)", |row: &SelectiveRow| {
                clamp_break_even(MEASURED_COST_BPS_MATCHED * row.sigma_ratio.mean)
            }),
        ] {
            series.push(ReportSeries {
                label: format!("{tag} threshold {label}"),
                values: table.rows.iter().map(|row| pick(row) as f32).collect(),
            });
        }
    }
    // The four horizontal references a reader must have on the same axis to read anything: the
    // MATCHED measured cost the break-even is judged against, its median so the right-skew is
    // visible, the sized cost that inherits an unfitted impact model, and the coin-flip line
    // every constant predictor sits on.
    for (label, value) in [
        (
            format!("MATCHED measured one-way cost {MEASURED_COST_BPS_MATCHED:.3} bps (mean of 256 names)"),
            MEASURED_COST_BPS_MATCHED,
        ),
        (
            format!("MATCHED measured one-way cost {MEASURED_COST_BPS_MATCHED_MEDIAN:.3} bps (median)"),
            MEASURED_COST_BPS_MATCHED_MEDIAN,
        ),
        (
            format!(
                "sized @1% ADV, matched {SIZED_COST_BPS_MATCHED_AT_1PCT_ADV:.3} bps \
                 (UNFITTED k=0.5 impact)"
            ),
            SIZED_COST_BPS_MATCHED_AT_1PCT_ADV,
        ),
        ("balanced-accuracy chance 0.5".to_owned(), 0.5),
    ] {
        series.push(ReportSeries {
            label,
            values: vec![value as f32; DECILES],
        });
    }

    ensure!(
        !series.is_empty(),
        "{SKILL_PROFILE_BASE} would be an empty chart"
    );
    let path = dir.join(format!("{SKILL_PROFILE_BASE}.report.bin"));
    write_report(
        &path,
        &Report {
            title: format!("Directional Skill vs Confidence - {}", profile.label),
            x_label: Some(
                "decile of the selector (confidence curve), or lowest traded decile (selective \
                 table)"
                    .to_owned(),
            ),
            y_label: Some("accuracy / IC / bps / bar count".to_owned()),
            scale: ScaleKind::Symlog,
            kind: ReportKind::MultiLine { series },
        },
    )
    .with_context(|| format!("writing {}", path.display()))?;
    // Reading it back is what turns "the writer ran" into "the chart exists": a truncated or
    // all-non-finite series renders as a blank panel and nothing else would notice.
    let report = read_report(&path).with_context(|| format!("reading back {}", path.display()))?;
    match report.kind {
        ReportKind::MultiLine { series } => ensure!(
            series
                .iter()
                .any(|s| s.values.len() == DECILES && s.values.iter().all(|v| v.is_finite())),
            "{SKILL_PROFILE_BASE} holds no complete finite series"
        ),
        other => bail!("{SKILL_PROFILE_BASE} came back as {other:?}"),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The entry point
// ---------------------------------------------------------------------------

/// Arguments of the standalone directional-skill audit.
#[derive(Clone, Debug)]
pub struct SkillArgs {
    /// Checkpoint to audit. Its `.metadata.json` and `.supports.<res>.json` siblings are
    /// resolved from this path, so a copy must keep the same file stem.
    pub weights: String,
    /// Directory the `pretrain_skill_profile.report.bin` chart is written into.
    pub output: String,
    pub split: Split,
    /// Pinned windows to DRAW. The audit scores the first
    /// [`super::trade_bench::TRADE_WINDOWS`] of them, which is exactly the prefix the trading
    /// bench and the mean calibration measure, so every number here is on the same bars as
    /// theirs. The count is part of the pin, so it must equal the run's
    /// `--validation-windows`.
    pub windows: usize,
    /// Conditioning context. Must match the context the compared reads were taken at.
    pub context: i64,
    pub batch_size: usize,
    pub corpus: CorpusFlags,
}

/// Measure the predictor's directional skill on pinned held-out windows, with no policy.
///
/// Shares every piece that decides WHAT is measured with the trading bench and the mean
/// calibration: [`PinnedSet::pinned`] draws the windows under [`EVAL_WINDOW_SEED`],
/// [`evaluate`] produces the marginalized conditional moments, [`pinned_blocks`] blocks the
/// intervals. The panel is therefore the same panel, and a disagreement between this module's
/// numbers and the bench's is a real disagreement rather than two samples.
pub fn pretrain_skill(args: SkillArgs) -> Result<()> {
    ensure!(args.windows > 0, "--windows must be positive");
    ensure!(args.context > 0, "--context must be positive");
    ensure!(args.batch_size > 0, "--batch-size must be positive");
    configure_threads();
    configure_cuda();

    let device = Device::cuda_if_available();
    let weights = Path::new(&args.weights);
    let metadata = world_model_metadata_path(weights);
    ensure!(
        metadata.exists(),
        "no metadata sidecar beside {}; copy {} next to the weights",
        weights.display(),
        metadata.display()
    );
    let world = BarWorldModel::load(weights, &metadata, device)?;
    ensure!(
        world.metadata().res_secs == args.corpus.resolution_secs,
        "checkpoint was trained for {}s bars but --resolution-secs is {}",
        world.metadata().res_secs,
        args.corpus.resolution_secs
    );
    let corpus = load_corpus(&args.corpus)?;
    let fingerprint = corpus.identity_fingerprint();
    if let Some(trained) = world.metadata().training.as_ref() {
        if trained.corpus_fingerprint != fingerprint {
            println!(
                "WARNING corpus {} is not the {} the checkpoint was trained on; the pinned \
                 windows are drawn from a different symbol set and are NOT the run's own",
                &fingerprint[..12.min(fingerprint.len())],
                &trained.corpus_fingerprint[..12.min(trained.corpus_fingerprint.len())],
            );
        }
        ensure!(
            trained.eval_window_seed == EVAL_WINDOW_SEED,
            "checkpoint pinned its evaluation with eval_window_seed {:#x} but this build uses \
             {EVAL_WINDOW_SEED:#x}; the audit would score different data than the run's bench",
            trained.eval_window_seed
        );
    }
    // The scoring rule enters not one statistic below - the moments come from the head's
    // probabilities - but `evaluate` needs one to reduce its NLL, so it is read off the
    // artifact rather than re-declared.
    let scoring: BarScoring = world
        .metadata()
        .training
        .as_ref()
        .map(|trained| trained.scoring.parse())
        .transpose()
        .map_err(|reason| {
            anyhow!("the checkpoint records a scoring rule this build cannot parse: {reason}")
        })?
        .unwrap_or_default();

    let set = PinnedSet::pinned(&corpus, args.split, args.context, args.windows)?;
    let stats = evaluate(
        world.modules(),
        world.deployment_supports(),
        &set,
        args.batch_size,
        device,
        true,
        scoring,
        None,
        TRADE_WINDOWS,
    )?;
    let scored = stats.trade_paths.windows.len();
    ensure!(
        scored > 0,
        "the evaluation produced no scored windows, so there is nothing to audit"
    );
    let mut blocks = pinned_blocks(&set);
    blocks.truncate(scored);
    let symbols: Vec<u32> = set.windows[..scored]
        .iter()
        .map(|window| window.symbol)
        .collect();
    let panel = SkillPanel::from_paths(&stats.trade_paths.windows, &symbols, &blocks)?;
    // The rule is defined on the population being scored, so the cutpoints come from THIS panel.
    // A restricted cross must instead build them on the full panel and pass those in unchanged.
    //
    // The support geometry comes from the DEPLOYMENT supports - the same decode vector the Kelly
    // solve and every moment are taken off - so the catch-all indicators describe the artifact this
    // checkpoint actually carries rather than a nominal grid.
    let cuts = SkillCutpoints::from_panel(&panel)
        .with_support_geometry(world.deployment_supports().centers(DOF_R));
    let profile = SkillProfile::measure(
        &panel,
        format!("{:?} split, {}", args.split, weights.display()),
        &cuts,
    );

    println!(
        "directional skill audit of {} (lineage {}) on the pinned {:?} split at context {}: the \
         first {} of {} drawn windows (TRADE_WINDOWS = {}), nll {:.4} nats/bar",
        weights.display(),
        world.lineage_sha256(),
        args.split,
        set.context,
        scored,
        set.windows.len(),
        TRADE_WINDOWS,
        stats.nll_bar,
    );
    for line in profile.report_lines() {
        println!("{line}");
    }
    let output = Path::new(&args.output);
    write_skill_profile(output, &profile)?;
    println!("reports written to {}", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::bar_dist::{
        BarDof, BarEmissionHead, BarSupports, BAR_DOF, DOF_S, DOF_U, NUM_BAR_BINS,
    };
    use crate::torch::test_rng;
    use crate::torch::train::trade_bench::TradeSetup;
    use tch::{nn, Kind, Tensor};

    /// A panel built from explicit `(symbol, block, mu, sigma, r)` rows, one window per
    /// distinct `(symbol, block)`, so a statistic can be asserted against arithmetic rather
    /// than against a second implementation.
    fn panel_from(rows: &[(u32, u64, f64, f64, f64)]) -> SkillPanel {
        let mut grouped: BTreeMap<(u32, u64), Vec<SkillBar>> = BTreeMap::new();
        for (symbol, block, mu, sigma, r) in rows {
            grouped
                .entry((*symbol, *block))
                .or_default()
                .push(SkillBar {
                    mu: *mu,
                    sigma: *sigma,
                    r: *r,
                    free: *mu,
                });
        }
        SkillPanel {
            windows: grouped
                .into_iter()
                .map(|((symbol, block), bars)| SkillWindow {
                    bars,
                    symbol,
                    block,
                })
                .collect(),
        }
    }

    /// A deterministic xorshift, so a fixture with a planted signal is reproducible without a
    /// dependency on any RNG this repository might reseed.
    fn stream(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed | 1;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn synthetic_supports(count: usize, seed: u64) -> BarSupports {
        let mut next = stream(seed);
        let samples: Vec<BarDof> = (0..count)
            .map(|_| {
                let u = next().max(1e-9);
                let s = 0.004 * (-2.0 * u.ln()).sqrt();
                BarDof {
                    r: ((next() - 0.5) * 2.0 * s) as f32,
                    s: s as f32,
                    u: next() as f32,
                    v: next() as f32,
                    w: (next() - 0.5) as f32,
                }
            })
            .collect();
        BarSupports::fit(&samples)
    }

    /// A head whose weights and prefix table are non-trivial: a zero-init head has a uniform
    /// `r` law and no prefix response at all, which would make the lookahead test vacuous.
    fn seeded_perturbed_head(latent: i64, seed: i64) -> (nn::VarStore, BarEmissionHead) {
        let vs = nn::VarStore::new(Device::Cpu);
        let head = BarEmissionHead::new(&vs.root(), latent);
        tch::manual_seed(seed);
        tch::no_grad(|| {
            for variable in vs.trainable_variables() {
                let mut variable = variable;
                let _ = variable.normal_(0.0, 0.35);
            }
        });
        (vs, head)
    }

    /// Every reported scalar as a raw bit pattern.
    ///
    /// Bit patterns rather than values so `assert_eq!` is genuine BIT-identity and so a NaN
    /// compares equal to a NaN — a tolerance, or a `f64 == f64` over NaN, would let an
    /// implementation that mixed a little lookahead in pass the invariance test.
    fn fingerprint(profile: &SkillProfile) -> Vec<u64> {
        fn push_dispersion(out: &mut Vec<u64>, d: &Dispersion) {
            for value in [d.mean, d.se, d.ci_low, d.ci_high] {
                out.push(value.to_bits());
            }
            out.push(d.blocks as u64);
            out.push(d.samples as u64);
        }
        let mut out: Vec<u64> = Vec::new();
        let c = &profile.confusion;
        for value in c.counts {
            out.push(value.to_bits());
        }
        for value in [c.flat_bars, c.undirected_bars, c.all_bars] {
            out.push(value.to_bits());
        }
        for dispersion in [
            &c.accuracy,
            &c.accuracy_on_up,
            &c.accuracy_on_down,
            &c.balanced_accuracy,
            &c.precision_up,
            &c.precision_down,
            &c.base_rate_up,
            &c.up_rate_all_bars,
            &c.always_up_accuracy,
            &c.always_down_accuracy,
            &c.majority_accuracy,
            &c.accuracy_over_majority,
            &c.balanced_over_chance,
            &c.kelly_sign_hit_rate,
            &c.kelly_sign_disagreement,
        ] {
            push_dispersion(&mut out, dispersion);
        }
        let ic = &profile.ic;
        for dispersion in [
            &ic.pooled_pearson,
            &ic.pooled_spearman,
            &ic.within_median,
            &ic.within_mean,
            &ic.within_fraction_positive,
            &ic.standardized_pearson,
            &ic.standardized_spearman,
        ] {
            push_dispersion(&mut out, dispersion);
        }
        for value in [ic.pooled_r2, ic.within_q1, ic.within_q3] {
            out.push(value.to_bits());
        }
        for value in &ic.per_symbol {
            out.push(value.to_bits());
        }
        for value in &ic.per_symbol_bars {
            out.push(*value as u64);
        }
        for value in [ic.symbols_measured, ic.symbols_dropped, ic.blocks_dropped] {
            out.push(value as u64);
        }
        for dispersion in [
            &profile.magnitude.abs_mu_vs_abs_r,
            &profile.magnitude.sigma_vs_abs_r,
            &profile.magnitude.abs_mu_vs_abs_r_standardized,
            &profile.magnitude.sigma_vs_abs_r_standardized,
            &profile.auc,
        ] {
            push_dispersion(&mut out, dispersion);
        }
        for curve in [&profile.abs_mu_curve, &profile.sharpe_curve] {
            for value in &curve.cutpoints {
                out.push(value.to_bits());
            }
            out.push(curve.excluded as u64);
            for row in &curve.rows {
                out.push(row.bars as u64);
                for dispersion in [
                    &row.accuracy,
                    &row.balanced_accuracy,
                    &row.edge_bps,
                    &row.ic,
                    &row.base_rate_up,
                ] {
                    push_dispersion(&mut out, dispersion);
                }
            }
            for dispersion in [
                &curve.top_minus_bottom_accuracy,
                &curve.top_minus_bottom_balanced,
                &curve.top_minus_bottom_edge_bps,
                &curve.top_minus_bottom_ic,
            ] {
                push_dispersion(&mut out, dispersion);
            }
            push_dispersion(&mut out, &curve.top_edge_multiple);
            push_dispersion(&mut out, &curve.top_edge_multiple_moving);
            for row in &curve.rows {
                push_dispersion(&mut out, &row.edge_bps_moving);
                push_dispersion(&mut out, &row.over_bound_mu);
                push_dispersion(&mut out, &row.over_bound_sigma);
                push_dispersion(&mut out, &row.over_reference_sigma);
                out.push(row.flat_bars as u64);
            }
            for value in [
                curve.accuracy_rank_correlation,
                curve.edge_rank_correlation,
            ] {
                out.push(value.to_bits());
            }
        }
        for table in [&profile.abs_mu_selective, &profile.sharpe_selective] {
            for row in &table.rows {
                out.push(row.threshold_decile as u64);
                out.push(row.threshold_value.to_bits());
                for dispersion in [
                    &row.participation,
                    &row.edge_bps,
                    &row.turnover_per_traded_bar,
                    &row.break_even_bps,
                    &row.hit_rate,
                    &row.sigma_ratio,
                ] {
                    push_dispersion(&mut out, dispersion);
                }
            }
        }
        out
    }

    /// The 2x2 and every rate derived from it must be the arithmetic of the counts.
    ///
    /// Built with a deliberate 40/60 imbalance and a predictor that is right on every up-bar
    /// and wrong on most down-bars, which is the configuration where raw accuracy and balanced
    /// accuracy disagree maximally and where a reader could be fooled by the wrong baseline.
    #[test]
    fn the_confusion_matrix_is_the_arithmetic_of_its_own_counts() {
        // 40 up-bars, all predicted up. 60 down-bars, 20 predicted down.
        let mut rows = Vec::new();
        for index in 0..40u32 {
            rows.push((index % 4, u64::from(index % 8), 1e-4, 1e-3, 1e-3));
        }
        for index in 0..40u32 {
            rows.push((index % 4, u64::from(index % 8), 1e-4, 1e-3, -1e-3));
        }
        for index in 0..20u32 {
            rows.push((index % 4, u64::from(index % 8), -1e-4, 1e-3, -1e-3));
        }
        let panel = panel_from(&rows);
        let report = confusion_report(&panel);
        assert_eq!(report.counts, [40.0, 40.0, 0.0, 20.0]);
        assert_eq!(report.all_bars, 100.0);
        // accuracy = (40 + 20) / 100
        assert!((report.accuracy.mean - 0.60).abs() < 1e-12);
        // recall up = 40/40, recall down = 20/60
        assert!((report.accuracy_on_up.mean - 1.0).abs() < 1e-12);
        assert!((report.accuracy_on_down.mean - 20.0 / 60.0).abs() < 1e-12);
        assert!((report.balanced_accuracy.mean - 0.5 * (1.0 + 20.0 / 60.0)).abs() < 1e-12);
        // The base rate is 0.40 up, so the best constant rule is "always down" at 0.60 -
        // EXACTLY the model's raw accuracy. Raw accuracy therefore shows zero skill here while
        // balanced accuracy shows a lot, which is the whole reason both are reported.
        assert!((report.base_rate_up.mean - 0.40).abs() < 1e-12);
        assert!((report.majority_accuracy.mean - 0.60).abs() < 1e-12);
        assert!(report.accuracy_over_majority.mean.abs() < 1e-12);
        assert!(report.balanced_over_chance.mean > 0.16);
        assert!((report.up_rate_all_bars.mean - 0.40).abs() < 1e-12);
        // precision up = 40/80, precision down = 20/20
        assert!((report.precision_up.mean - 0.5).abs() < 1e-12);
        assert!((report.precision_down.mean - 1.0).abs() < 1e-12);
        // Eight blocks, so there is a real blocked interval rather than a NaN.
        assert_eq!(report.accuracy.blocks, 8);
        assert!(report.accuracy.se.is_finite() && report.accuracy.se > 0.0);
    }

    /// Flat bars leave the 2x2 and stay in the all-bar up rate, because that is the only
    /// convention under which the reported up rate equals buy & hold's own `hit`.
    #[test]
    fn flat_bars_leave_the_two_by_two_but_stay_in_the_all_bar_up_rate() {
        let rows = vec![
            (0u32, 0u64, 1e-4, 1e-3, 1e-3),
            (0, 0, 1e-4, 1e-3, -1e-3),
            (0, 1, 1e-4, 1e-3, 0.0),
            (0, 1, 1e-4, 1e-3, 0.0),
        ];
        let panel = panel_from(&rows);
        let report = confusion_report(&panel);
        assert_eq!(report.counts.iter().sum::<f64>(), 2.0);
        assert_eq!(report.flat_bars, 2.0);
        // Classified base rate: one up of two. All-bar rate: one up of four, which is what a
        // policy staking f = +1 on every bar would score.
        assert!((report.base_rate_up.mean - 0.5).abs() < 1e-12);
        assert!((report.up_rate_all_bars.mean - 0.25).abs() < 1e-12);
    }

    /// An UNMEASURED break-even must reach the chart as NaN, never as the clip constant.
    ///
    /// `f64::min` IGNORES NaN, so the natural `value.min(MAX_BREAK_EVEN_BPS)` renders "we could
    /// not measure this" as 1000 bps — ninety-four times the reference cost line drawn on the
    /// same panel, and the exact confusion `MAX_BREAK_EVEN_BPS` was introduced to prevent. The
    /// assertion is on the CLAMP rather than on a whole profile because the reachable route to a
    /// NaN break-even (a threshold that trades no bars, or a panel with fewer than two blocks)
    /// also makes every sibling series NaN and there would then be no finite series to compare
    /// against.
    #[test]
    fn an_unmeasured_break_even_is_charted_as_nan_and_not_as_the_clip() {
        assert!(
            clamp_break_even(f64::NAN).is_nan(),
            "NaN must survive the clamp; f64::min would have returned {}",
            f64::NAN.min(MAX_BREAK_EVEN_BPS)
        );
        // The bug this guards, stated as an executable fact rather than a comment.
        assert_eq!(
            f64::NAN.min(MAX_BREAK_EVEN_BPS),
            MAX_BREAK_EVEN_BPS,
            "if f64::min ever starts propagating NaN this guard is redundant"
        );
        assert!(clamp_break_even(f64::INFINITY).is_nan());
        assert!(clamp_break_even(f64::NEG_INFINITY).is_nan());
        // Finite values are clipped, not altered.
        assert!((clamp_break_even(7.5) - 7.5).abs() < 1e-12);
        assert!(
            (clamp_break_even(MAX_BREAK_EVEN_BPS * 10.0) - MAX_BREAK_EVEN_BPS).abs() < 1e-12
        );
        // A finite NEGATIVE break-even is a real measurement — the rule loses money gross — and
        // must pass through untouched rather than being floored.
        assert!((clamp_break_even(-3.25) + 3.25).abs() < 1e-12);
    }

    /// A selector determined by the SYMBOL rather than by the bar must show it in the occupancy.
    ///
    /// This is the between-name failure mode as a fixture. `sigma` is a per-symbol constant and
    /// `mu` is identical everywhere, so `|mu_hat|/sigma_hat` ranks purely by name: each block holds
    /// exactly one decile and the top-minus-bottom difference, whatever its interval says, compares
    /// two disjoint sets of blocks. The test asserts the diagnostic FIRES - top and bottom deciles
    /// empty in nine tenths of blocks - because a paired claim in that regime is a between-name
    /// claim, and nothing in the interval arithmetic reveals it.
    #[test]
    fn a_name_determined_selector_shows_up_as_empty_blocks_in_the_occupancy() {
        let mut rows = Vec::new();
        let mut next = stream(0x0CC0_1234);
        for block in 0..DECILES as u64 {
            // One symbol per block, its sigma fixed by the block: the selector cannot vary within.
            let sigma = 1.0e-4 * (block as f64 + 1.0);
            for _ in 0..64 {
                let r = if next() > 0.5 { sigma } else { -sigma };
                rows.push((block as u32, block, 1.0e-6, sigma, r));
            }
        }
        let panel = panel_from(&rows);
        let cuts = SkillCutpoints::from_panel(&panel);
        let curve = confidence_curve(&panel, SELECTOR_SHARPE, selector_sharpe, &cuts);
        assert_eq!(curve.blocks, DECILES, "one block per symbol");
        assert_eq!(
            curve.blocks_missing_top,
            DECILES - 1,
            "a name-determined selector must leave the top decile empty in every block but one,              got {} of {} with occupancy {:?}",
            curve.blocks_missing_top,
            curve.blocks,
            curve.occupancy_top
        );
        assert_eq!(curve.blocks_missing_bottom, DECILES - 1);
        // Non-vacuity: a BAR-determined selector on the same panel shape must NOT trip it, so the
        // diagnostic is reading the name-determination rather than the fixture's block structure.
        let mut bar_rows = Vec::new();
        let mut next = stream(0x0CC0_5678);
        for block in 0..DECILES as u64 {
            // 500 bars per block, not 64: ten deciles drawn from 64 bars leave some bucket empty in
            // some block by chance alone, which would make the non-vacuity check flaky rather than
            // wrong. The real panel carries ~896 bars per block.
            for _ in 0..500 {
                let sigma = 1.0e-4 * (1.0 + 9.0 * next());
                let r = if next() > 0.5 { sigma } else { -sigma };
                bar_rows.push((block as u32, block, 1.0e-6, sigma, r));
            }
        }
        let bar_panel = panel_from(&bar_rows);
        let bar_cuts = SkillCutpoints::from_panel(&bar_panel);
        let bar_curve = confidence_curve(&bar_panel, SELECTOR_SHARPE, selector_sharpe, &bar_cuts);
        assert_eq!(
            bar_curve.blocks_missing_top, 0,
            "a bar-determined selector must populate the top decile in every block, got {} empty              with occupancy {:?}",
            bar_curve.blocks_missing_top, bar_curve.occupancy_top
        );
        assert!(
            bar_curve.occupancy_top[0] > 0.0 && bar_curve.occupancy_bottom[0] > 0.0,
            "min occupancy must be positive when both deciles are populated everywhere"
        );
    }

    /// A CONSTANT predictor on buckets with different class balances must NOT be reported as
    /// concentrated, however large its raw-accuracy gradient.
    ///
    /// This is the base-rate confound, deterministic by construction. The fixture always
    /// predicts DOWN — literally zero directional skill, one fixed answer — but `|mu_hat|` rises
    /// across the deciles, and the top decile is 20/80 up/down while the rest are 50/50 or
    /// 25/75. Raw accuracy therefore reads about 0.5 at the bottom and exactly 0.8 at the top, a
    /// large resolvable gradient that is nothing but the imbalance the module was written to
    /// defend against. Balanced accuracy is exactly 0.5 in every bucket of every block, because
    /// an always-down rule has `TPR = 0` and `TNR = 1` at any class balance, so the criterion
    /// must refuse.
    #[test]
    fn a_constant_predictor_with_a_class_balance_gradient_is_not_concentrated() {
        let mut rows: Vec<(u32, u64, f64, f64, f64)> = Vec::new();
        for block in 0..40u64 {
            for slot in 0..200usize {
                let decile = slot / 20;
                let inner = slot % 20;
                // Up-rate: 0.2 in the top decile, and 0.5 or 0.25 elsewhere depending on the
                // block, so the bottom decile's accuracy VARIES across blocks and the paired
                // difference gets a non-degenerate blocked standard error. Balanced accuracy is
                // untouched by any of it, which is the entire point.
                let up = if decile == DECILES - 1 {
                    inner % 5 == 0
                } else if block % 2 == 0 {
                    inner % 2 == 0
                } else {
                    inner % 4 == 0
                };
                rows.push((
                    (block % 7) as u32,
                    block,
                    // Always DOWN, magnitude strictly increasing so the deciles are clean and
                    // no selector tie can starve a bucket.
                    -((slot as f64) + 1.0) * 1e-5,
                    1e-3,
                    if up { 1e-3 } else { -1e-3 },
                ));
            }
        }
        let panel = panel_from(&rows);
        let profile = SkillProfile::measure(&panel, "base-rate confound", &SkillCutpoints::from_panel(&panel));
        let curve = &profile.abs_mu_curve;
        // Balanced accuracy is EXACTLY 0.5 in every bucket, so the criterion axis is flat with a
        // degenerate interval and cannot resolve.
        for (index, row) in curve.rows.iter().enumerate() {
            assert!(
                (row.balanced_accuracy.mean - 0.5).abs() < 1e-12,
                "decile {index} balanced accuracy {} is not exactly 0.5 for a constant rule",
                row.balanced_accuracy.mean
            );
        }
        // The confounded axis DOES resolve, which is what makes this test non-vacuous: the old
        // criterion would have answered YES here. Measured on this fixture: 0.8000 at the top
        // against 0.6250 at the bottom (the bottom averages the 0.5 and 0.75 block regimes), a
        // paired +0.1750 +/- 0.0204, i.e. 8.6 of its own blocked standard errors.
        assert!(
            curve.rows[DECILES - 1].accuracy.mean > 0.75
                && curve.rows[0].accuracy.mean < 0.70
                && resolved(&curve.top_minus_bottom_accuracy)
                && curve.top_minus_bottom_accuracy.mean > 0.1
                && curve.top_minus_bottom_accuracy.mean
                    > 5.0 * curve.top_minus_bottom_accuracy.se,
            "the fixture must produce a large RESOLVED raw-accuracy gradient, got {} at the top, \
             {} at the bottom, paired {}",
            curve.rows[DECILES - 1].accuracy.mean,
            curve.rows[0].accuracy.mean,
            curve.top_minus_bottom_accuracy
        );
        assert!(
            !profile.concentrated(),
            "a constant predictor was reported as having concentrated DIRECTIONAL skill; \
             balanced paired {}, accuracy paired {}",
            curve.top_minus_bottom_balanced,
            curve.top_minus_bottom_accuracy
        );
        assert!(
            !profile.anti_concentrated(),
            "an exactly flat balanced axis must not report as inverted either"
        );
    }

    /// Bigger bars in the top decile must NOT be reported as concentrated DIRECTIONAL skill.
    ///
    /// This is the volatility confound, and it is independent of the base-rate one. Every decile
    /// here has the SAME accuracy (exactly 11 of 20) and the SAME up-rate (exactly 10 of 20); the
    /// only difference is that the top decile's realized moves are three times larger. Since
    /// `E[sign(mu_hat) r | decile]` is approximately `(2a - 1) E[|r| | decile]`, the edge axis
    /// rises 3x while the sign is not one bit more reliable. On the real panel this confound is
    /// CERTAIN to be present, because the selector is `|mu_hat|` and this module measures
    /// `corr(|mu_hat|, |r|)` at about +0.41, so the edge axis could not have been used as the
    /// criterion without conflating the magnitude story with the direction story.
    #[test]
    fn a_volatility_gradient_with_constant_accuracy_is_not_concentrated() {
        let mut rows: Vec<(u32, u64, f64, f64, f64)> = Vec::new();
        for block in 0..40u64 {
            // Per-block scale so the edge difference has a real blocked standard error while the
            // accuracy pattern stays bit-identical in every block.
            let scale = 1.0 + f64::from((block % 3) as u32) * 0.5;
            for slot in 0..200usize {
                let decile = slot / 20;
                let inner = slot % 20;
                // Exactly 10 up and 10 down per decile, and exactly 11 correct of 20 in EVERY
                // decile: TPR 0.6, TNR 0.5, balanced 0.55, identical everywhere.
                let up = inner % 2 == 0;
                let correct = inner < 11;
                let sign = if up { 1.0 } else { -1.0 };
                let magnitude = if decile == DECILES - 1 { 3e-3 } else { 1e-3 } * scale;
                rows.push((
                    (block % 7) as u32,
                    block,
                    ((slot as f64) + 1.0) * 1e-5 * if correct { sign } else { -sign },
                    1e-3,
                    sign * magnitude,
                ));
            }
        }
        let panel = panel_from(&rows);
        let profile = SkillProfile::measure(&panel, "volatility confound", &SkillCutpoints::from_panel(&panel));
        let curve = &profile.abs_mu_curve;
        for (index, row) in curve.rows.iter().enumerate() {
            assert!(
                (row.balanced_accuracy.mean - 0.55).abs() < 1e-12,
                "decile {index} balanced accuracy {} is not exactly 0.55",
                row.balanced_accuracy.mean
            );
        }
        // Non-vacuity: the edge axis resolves at a large multiple of its own error, so the old
        // criterion would have answered YES on a fixture with provably constant sign reliability.
        assert!(
            resolved(&curve.top_minus_bottom_edge_bps)
                && curve.top_minus_bottom_edge_bps.mean > 1.0,
            "the fixture must produce a RESOLVED edge gradient, got {}",
            curve.top_minus_bottom_edge_bps
        );
        assert!(
            !profile.concentrated(),
            "a pure volatility gradient was reported as concentrated DIRECTIONAL skill; \
             balanced paired {}, edge paired {}",
            curve.top_minus_bottom_balanced,
            curve.top_minus_bottom_edge_bps
        );
    }

    /// The placement identity must reproduce a brute-force Mann-Whitney AUC exactly, ties
    /// included, or the threshold-free summary is a different number from the one it claims.
    #[test]
    fn the_placement_form_of_the_auc_matches_a_brute_force_mann_whitney() {
        // Deliberate ties in `mu`, and a deliberately imperfect ranking.
        let mus = [3.0, 1.0, 2.0, 2.0, -1.0, 0.5, 2.0, -3.0, 4.0, 0.25];
        let ups = [
            true, false, true, false, false, true, true, false, true, false,
        ];
        let rows: Vec<(u32, u64, f64, f64, f64)> = mus
            .iter()
            .zip(ups)
            .enumerate()
            .map(|(index, (mu, up))| {
                (
                    0u32,
                    (index % 3) as u64,
                    *mu,
                    1.0,
                    if up { 1.0 } else { -1.0 },
                )
            })
            .collect();
        let panel = panel_from(&rows);
        let measured = auc(&panel).mean;

        let mut wins = 0.0f64;
        let mut pairs = 0.0f64;
        for (up_mu, up) in mus.iter().zip(ups) {
            if !up {
                continue;
            }
            for (down_mu, down) in mus.iter().zip(ups) {
                if down {
                    continue;
                }
                pairs += 1.0;
                wins += if up_mu > down_mu {
                    1.0
                } else if up_mu == down_mu {
                    0.5
                } else {
                    0.0
                };
            }
        }
        let brute = wins / pairs;
        assert!(
            (measured - brute).abs() < 1e-12,
            "placement AUC {measured} differs from the Mann-Whitney {brute}"
        );
    }

    /// The standardized IC must be the bar-count-weighted mean of the within-block
    /// correlations, which is the identity the whole heteroskedasticity argument rests on -
    /// and the un-standardized pooled IC on the same fixture must be swamped by scale, which
    /// is the contamination it removes.
    #[test]
    fn the_standardized_ic_is_the_count_weighted_mean_of_within_block_correlations() {
        // Two blocks of WILDLY different scale and OPPOSITE within-block sign structure.
        let mut rows = Vec::new();
        for index in 0..20 {
            let x = (index as f64) - 9.5;
            rows.push((0u32, 0u64, 1e-6 * x, 1e-3, 1e-6 * x));
        }
        for index in 0..40 {
            let x = (index as f64) - 19.5;
            rows.push((1u32, 1u64, x, 1e-3, -x));
        }
        let panel = panel_from(&rows);
        let (blocks, dropped) = standardized_blocks(&panel, |bar| (bar.mu, bar.r));
        assert_eq!(dropped, 0);
        assert_eq!(blocks.len(), 2);
        let mut pooled = Moments::default();
        for block in &blocks {
            Moments::absorb(&mut pooled, block);
        }
        // rho = +1 on 20 bars, rho = -1 on 40 bars.
        let want = (20.0 * 1.0 + 40.0 * -1.0) / 60.0;
        assert!(
            (pooled.corr() - want).abs() < 1e-9,
            "standardized IC {} is not the weighted mean {want}",
            pooled.corr()
        );
        let raw: Vec<Moments> = panel.per_block(|m: &mut Moments, bar| m.push(bar.mu, bar.r));
        let mut raw_pooled = Moments::default();
        for block in &raw {
            Moments::absorb(&mut raw_pooled, block);
        }
        assert!(
            raw_pooled.corr() < -0.99,
            "the fixture's pooled IC {} should be swamped by the large-scale block",
            raw_pooled.corr()
        );
    }

    /// The break-even column must equal the edge column over the turnover column, and the
    /// turnover must charge the exits a selective rule actually pays for.
    #[test]
    fn the_selective_break_even_is_the_edge_over_the_turnover_it_pays_for() {
        // One window per block, ten bars total, `|mu|` strictly increasing so every decile
        // holds exactly one bar. Signs alternate, so a position flips whenever two adjacent
        // bars are both traded.
        let rows: Vec<(u32, u64, f64, f64, f64)> = (0..10u32)
            .map(|index| {
                let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
                (
                    0u32,
                    u64::from(index % 2),
                    sign * (f64::from(index) + 1.0) * 1e-5,
                    1e-3,
                    sign * 1e-3,
                )
            })
            .collect();
        let panel = panel_from(&rows);
        let table = selective_table(&panel, SELECTOR_ABS_MU, selector_abs_mu, &SkillCutpoints::from_panel(&panel));
        assert_eq!(table.rows.len(), DECILES);
        for row in &table.rows {
            let implied = row.edge_bps.mean / row.turnover_per_traded_bar.mean;
            assert!(
                (row.break_even_bps.mean - implied).abs() < 1e-9,
                "decile {}: break-even {} is not edge {} over turnover {}",
                row.threshold_decile,
                row.break_even_bps.mean,
                row.edge_bps.mean,
                row.turnover_per_traded_bar.mean
            );
        }
        // Participation is MECHANICAL in equal-count deciles. Saying so is part of the finding
        // rather than a caveat, so it is asserted.
        for row in &table.rows {
            let want = (DECILES - row.threshold_decile) as f64 / DECILES as f64;
            assert!(
                (row.participation.mean - want).abs() < 1e-9,
                "decile {} participation {} != {want}",
                row.threshold_decile,
                row.participation.mean
            );
        }
        // The top decile trades ONE bar, at |mu| = 1e-4 with r of the same sign, so its edge
        // is 1e-3 nats = 10 bps; its turnover is one unit in and one unit out, i.e. 2.0 per
        // traded bar. Break-even is therefore exactly 5 bps.
        let top = &table.rows[DECILES - 1];
        assert!(
            (top.edge_bps.mean - 10.0).abs() < 1e-9,
            "{}",
            top.edge_bps.mean
        );
        assert!(
            (top.turnover_per_traded_bar.mean - 2.0).abs() < 1e-9,
            "{}",
            top.turnover_per_traded_bar.mean
        );
        assert!(
            (top.break_even_bps.mean - 5.0).abs() < 1e-9,
            "{}",
            top.break_even_bps.mean
        );
    }

    /// The traded-volatility multiple must be the ratio of two MEASURED means — mean `sigma_hat`
    /// on the bars the threshold trades over mean `sigma_hat` on every bar — and it must be
    /// exactly `1.0` when the threshold trades everything.
    ///
    /// This statistic is what keeps the break-even comparison from being a category error: the
    /// reference cost is unconditional in the bar, a confidence selector picks volatile bars, and
    /// this is the multiplier that says by how much. A bug that made it constant would silently
    /// restore the invalid comparison, so the fixture makes `sigma_hat` rise with `|mu_hat|` and
    /// pins all three of the full-participation, mid-threshold and top-decile values by hand.
    #[test]
    fn the_traded_volatility_multiple_is_the_ratio_of_two_measured_means() {
        // Ten bars, one decile each: `|mu|` and `sigma` both strictly increasing in the index,
        // so the selector and the volatility are perfectly rank-coupled, which is the real
        // panel's situation in exaggerated form.
        let rows: Vec<(u32, u64, f64, f64, f64)> = (0..10u32)
            .map(|index| {
                let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
                (
                    0u32,
                    u64::from(index % 2),
                    sign * (f64::from(index) + 1.0) * 1e-5,
                    (f64::from(index) + 1.0) * 1e-4,
                    sign * 1e-3,
                )
            })
            .collect();
        let panel = panel_from(&rows);
        let table = selective_table(&panel, SELECTOR_ABS_MU, selector_abs_mu, &SkillCutpoints::from_panel(&panel));
        // Mean sigma over all ten bars is (1 + .. + 10)/10 * 1e-4 = 5.5e-4.
        let all_mean = 5.5e-4;
        for (threshold, traded_mean) in [
            // Trades every bar, so the two means are the same sum over the same denominator.
            (0usize, 5.5e-4),
            // Trades indices 5..=9: (6 + 7 + 8 + 9 + 10)/5 * 1e-4.
            (5, 8.0e-4),
            // Trades index 9 alone.
            (9, 1.0e-3),
        ] {
            let want = traded_mean / all_mean;
            let got = table.rows[threshold].sigma_ratio.mean;
            assert!(
                (got - want).abs() < 1e-12,
                "threshold {threshold}: traded sigma multiple {got} is not {want}"
            );
        }
        assert!(
            (table.rows[0].sigma_ratio.mean - 1.0).abs() < 1e-12,
            "trading every bar must give a multiple of exactly 1, got {}",
            table.rows[0].sigma_ratio.mean
        );
        // The scaled reference must move with the multiple, or the verdict's conditioned
        // comparison is decoration. Asserted on the TOP decile explicitly rather than through
        // `best()`: in this fixture every threshold has an identical 5 bps break-even by
        // construction, so which row wins `best()` is decided by float noise and is not the
        // property under test.
        let top = &table.rows[DECILES - 1];
        assert!(
            top.sigma_ratio.mean > 1.5,
            "the top-decile fixture must select volatile bars, got {}",
            top.sigma_ratio.mean
        );
        assert!(
            table.reference_cost_bps * top.sigma_ratio.mean > table.reference_cost_bps,
            "a volatile selection must raise the reference, not lower it"
        );
        // And the reported scaled reference must be exactly that product for whichever row
        // `best()` selects, so the number in the verdict cannot drift from the table.
        let best = table.best().expect("a finite break-even");
        let scaled = table.volatility_scaled_reference_bps();
        assert!(
            (scaled - table.reference_cost_bps * best.sigma_ratio.mean).abs() < 1e-9,
            "scaled reference {scaled} is not the reference times the multiple"
        );
    }

    /// The paired top-minus-bottom difference must be the difference of the two marginal point
    /// estimates, and its interval must be NARROWER than combining the two marginals in
    /// quadrature — which is the entire reason it is the reported statistic.
    #[test]
    fn the_paired_decile_difference_is_tighter_than_two_marginal_intervals() {
        // A signal that strengthens with |mu|, on top of a per-BLOCK accuracy regime that shifts
        // every decile of that block in the SAME direction. The regime is the mechanism pairing
        // exists to defend against: on the real panel a month is either kind to the predictor or
        // not, in every confidence bucket at once, so each marginal decile accuracy inherits the
        // full regime variance while their difference does not. A fixture whose per-block
        // accuracies were independent across deciles would make the paired SE EQUAL the
        // quadrature and the assertion below vacuously near-true, which is why the common term
        // is keyed to the bootstrap block id rather than drawn per bar.
        let mut rows = Vec::new();
        let mut next = stream(0x51D5_0001);
        for block in 0..40u64 {
            let shock: f64 = if next() > 0.5 { 1.0 } else { -1.0 };
            let regime = if (block % 7) % 2 == 0 { 0.12 } else { -0.12 };
            for slot in 0..200 {
                let confidence = (f64::from(slot) + 1.0) / 200.0;
                let correct = next() < 0.45 + 0.4 * confidence + regime;
                let heads = next() > 0.5;
                let r: f64 = shock * 1e-4 + if heads { 1e-3 } else { -1e-3 };
                let mu = confidence * 1e-4 * if correct { r.signum() } else { -r.signum() };
                rows.push(((block % 7) as u32, block, mu, 1e-3, r));
            }
        }
        let panel = panel_from(&rows);
        let curve = confidence_curve(&panel, SELECTOR_ABS_MU, selector_abs_mu, &SkillCutpoints::from_panel(&panel));
        let top = &curve.rows[DECILES - 1].accuracy;
        let bottom = &curve.rows[0].accuracy;
        let paired = &curve.top_minus_bottom_accuracy;
        assert!(
            (paired.mean - (top.mean - bottom.mean)).abs() < 1e-12,
            "paired point {} is not {} - {}",
            paired.mean,
            top.mean,
            bottom.mean
        );
        let quadrature = (top.se * top.se + bottom.se * bottom.se).sqrt();
        assert!(
            paired.se < quadrature,
            "the paired SE {} is not tighter than the quadrature {quadrature}, so pairing \
             bought nothing",
            paired.se
        );
        // And the planted signal must actually be recovered, or this test checks arithmetic on
        // noise.
        assert!(
            paired.mean > 0.2 && resolved(paired),
            "the planted confidence signal was not recovered: {paired}"
        );
        assert!(curve.accuracy_rank_correlation > 0.9);
    }

    /// A flat curve must be reported as flat: `concentrated` has to be capable of returning
    /// false, or the deliverable is a rubber stamp.
    #[test]
    fn a_confidence_selector_with_no_signal_is_reported_as_flat() {
        let mut rows = Vec::new();
        let mut next = stream(0x51D5_0002);
        for block in 0..40u64 {
            for _ in 0..200 {
                // The magnitude of `mu` is independent of whether its sign is right, which is
                // exactly the null this deliverable has to be able to accept.
                let confidence = next();
                let heads = next() > 0.5;
                let correct = next() > 0.5;
                let r: f64 = if heads { 1e-3 } else { -1e-3 };
                let mu = confidence * 1e-4 * if correct { r.signum() } else { -r.signum() };
                rows.push(((block % 7) as u32, block, mu, 1e-3, r));
            }
        }
        let panel = panel_from(&rows);
        let profile = SkillProfile::measure(&panel, "null fixture", &SkillCutpoints::from_panel(&panel));
        assert!(
            !profile.concentrated(),
            "a selector independent of correctness was reported as concentrated: {} +/- {}",
            profile.abs_mu_curve.top_minus_bottom_accuracy.mean,
            profile.abs_mu_curve.top_minus_bottom_accuracy.se
        );
        assert!(profile.verdict().contains("NO - the curve is FLAT"));
    }

    /// Every reported scalar must be BIT-identical under an arbitrary reassignment of the
    /// realized same-bar `s`, and a panel built from a prefix-carrying row must move those
    /// same scalars a long way.
    ///
    /// This holds BY CONSTRUCTION now: `r` heads the chain, so the traded row conditions on
    /// no same-bar factor at all. The test is kept because that is exactly what could go
    /// wrong — a read that picked up a prefix-carrying row instead would type-check — and
    /// because a reorder that hands `r` a prefix has to fail here loudly. Two halves, the
    /// second of which is what keeps the first from being free:
    ///
    /// 1. INVARIANCE. The panel is built through [`TradeSetup::paths`], which takes the
    ///    realized continuation and selects `r` from it itself, and takes its moments through
    ///    [`super::super::trade_bench::forecast_r_probs`]. Setting the realized `s` column to
    ///    each of several probe bins across the alphabet must leave every scalar in
    ///    [`SkillProfile`] identical to the last bit — `assert_eq!` on `f64::to_bits`, never a
    ///    tolerance, because a tolerance would pass an implementation that mixed a little
    ///    lookahead in.
    /// 2. NON-VACUITY. The realized `s` DOES reach the rows that carry it in their prefix, and
    ///    a panel built from one of those — `u`, the factor directly behind `s` — produces a
    ///    different 2x2, a different headline IC, and moves when that `s` moves. If any of
    ///    that stops holding, the invariance above is guarding nothing and this test says so
    ///    instead of passing quietly.
    #[test]
    fn permuting_the_realized_same_bar_s_leaves_every_skill_statistic_bit_identical() {
        let _torch_rng_guard = test_rng::exclusive();
        let latent = 20i64;
        let supports = synthetic_supports(40_000, 0x5111_0001);
        let (_vs, head) = seeded_perturbed_head(latent, 0x5111_0002);
        let setup = TradeSetup::new(&supports, Device::Cpu, 4.0);
        let (windows, bars) = (6i64, 24i64);
        tch::manual_seed(0x5111_0003);
        let beliefs = Tensor::randn([windows, bars, latent], (Kind::Float, Device::Cpu));
        tch::manual_seed(0x5111_0004);
        let realized_r = Tensor::randn([windows, bars], (Kind::Float, Device::Cpu)) * 0.004;
        // `s` is a RANGE and is non-negative by construction; a signed fixture would be a
        // different and wrong one.
        let realized_s =
            Tensor::randn([windows, bars], (Kind::Float, Device::Cpu)).abs() * 0.004;
        let symbols: Vec<u32> = (0..windows as u32).map(|window| window % 3).collect();
        let blocks: Vec<u64> = (0..windows as u64).map(|window| window % 3).collect();
        let build = |s: &Tensor| -> SkillProfile {
            let dof = Tensor::zeros(
                [windows, bars, BAR_DOF as i64],
                (Kind::Float, Device::Cpu),
            );
            let mut r_column = dof.select(-1, DOF_R as i64);
            let _ = r_column.copy_(&realized_r);
            let mut s_column = dof.select(-1, DOF_S as i64);
            let _ = s_column.copy_(s);
            let paths = setup
                .paths(&head, &beliefs, &dof, windows as usize)
                .expect("paths");
            let panel =
                SkillPanel::from_paths(&paths.windows, &symbols, &blocks).expect("panel");
            SkillProfile::measure(&panel, "fixture", &SkillCutpoints::from_panel(&panel))
        };
        let baseline = build(&realized_s);
        let baseline_bits = fingerprint(&baseline);
        // A fixture whose 2x2 is degenerate could not detect anything, so it is rejected here
        // rather than silently passing.
        assert!(
            baseline.confusion.counts.iter().all(|count| *count > 0.0),
            "the fixture's 2x2 is degenerate: {:?}",
            baseline.confusion.counts
        );

        // Non-vacuity, part one: the realized same-bar `s` really does reach this head — on
        // the factors that carry it in their prefix. Without that, the bit-identity above
        // would be a property of a fixture nothing can move.
        let flat_beliefs = beliefs.reshape([-1, latent]);
        let rows = flat_beliefs.size()[0];
        let zero_prefix = Tensor::zeros([rows, BAR_DOF as i64], (Kind::Int64, Device::Cpu));
        let causal = head.logits(&flat_beliefs, &zero_prefix).select(1, DOF_U as i64);
        let mut prefix_response = 0.0f64;
        for bin in [0usize, 1, 37, 64, 91, NUM_BAR_BINS as usize - 1] {
            // Every bar told the same lie about its own range, once per probe bin: a sweep over
            // the prefix alphabet rather than one permutation of it.
            let lied = Tensor::full(
                [windows, bars],
                supports.centers(DOF_S)[bin],
                (Kind::Float, Device::Cpu),
            );
            assert_eq!(
                fingerprint(&build(&lied)),
                baseline_bits,
                "a reported skill statistic moved when the realized same-bar s was set to bin \
                 {bin}"
            );
            let prefix = Tensor::zeros([rows, BAR_DOF as i64], (Kind::Int64, Device::Cpu));
            let mut column = prefix.select(1, DOF_S as i64);
            let _ = column.fill_(bin as i64);
            // `u` sits behind `s` in the chain, so ITS row is conditioned on the realized
            // range. The traded `r` row is not, and that asymmetry is the whole property.
            let conditioned = head.logits(&flat_beliefs, &prefix).select(1, DOF_U as i64);
            prefix_response =
                prefix_response.max((&conditioned - &causal).abs().max().double_value(&[]));
        }
        assert!(
            prefix_response > 1e-2,
            "the realized same-bar s moves no row of this fixture's head at all \
             ({prefix_response:.3e} logits), so this test could not detect lookahead"
        );

        // NON-VACUITY on the STATISTICS and not merely on the logits: a panel built from a
        // row that DOES carry the realized `s` in its prefix must move the 2x2 and the
        // headline IC.
        let leaked = leaked_panel(
            &head,
            &supports,
            &beliefs,
            &realized_r,
            &realized_s,
            &symbols,
            &blocks,
        );
        let leaked_profile = SkillProfile::measure(&leaked, "leaked", &SkillCutpoints::from_panel(&leaked));
        assert_ne!(
            leaked_profile.confusion.counts, baseline.confusion.counts,
            "conditioning on the realized same-bar s did not change a single cell of the 2x2, \
             so this test cannot detect lookahead"
        );
        let ic_shift =
            (leaked_profile.ic.pooled_pearson.mean - baseline.ic.pooled_pearson.mean).abs();
        assert!(
            ic_shift > 1e-3,
            "the leaked panel's pooled IC differs by only {ic_shift:.3e}, so a leak \
             would be invisible in the headline statistic"
        );
        // And it is a leak OF THE SAME-BAR `s`: change that `s` and the leaked panel moves,
        // while every honest statistic above stayed bit-identical under the same change.
        let lied_s = Tensor::full(
            [windows, bars],
            supports.centers(DOF_S)[NUM_BAR_BINS as usize - 1],
            (Kind::Float, Device::Cpu),
        );
        let lied = leaked_panel(
            &head,
            &supports,
            &beliefs,
            &realized_r,
            &lied_s,
            &symbols,
            &blocks,
        );
        let lied_profile = SkillProfile::measure(&lied, "lied", &SkillCutpoints::from_panel(&lied));
        assert_ne!(
            fingerprint(&lied_profile),
            fingerprint(&leaked_profile),
            "the leaked panel did not move when the realized same-bar s did, so it is not a \
             same-bar-s leak and cannot certify this test's sensitivity"
        );
    }

    /// A panel whose `mu_hat` comes from a TEACHER-FORCED row that carries the realized
    /// same-bar `s` in its prefix — `u`, the factor directly behind `s` — i.e. the exact
    /// lookahead this module exists to avoid. Test-only, and the only place in this file
    /// where a realized same-bar value reaches a prediction. The traded `r` row cannot serve
    /// here: `r` heads the chain, so its teacher-forced row IS the forecast.
    fn leaked_panel(
        head: &BarEmissionHead,
        supports: &BarSupports,
        beliefs: &Tensor,
        realized_r: &Tensor,
        realized_s: &Tensor,
        symbols: &[u32],
        blocks: &[u64],
    ) -> SkillPanel {
        let shape = beliefs.size();
        let (windows, bars, latent) = (shape[0], shape[1], shape[2]);
        let flat = beliefs.reshape([-1, latent]);
        let dof = Tensor::zeros(
            [windows, bars, BAR_DOF as i64],
            (Kind::Float, Device::Cpu),
        );
        let mut r_column = dof.select(-1, DOF_R as i64);
        let _ = r_column.copy_(realized_r);
        let mut s_column = dof.select(-1, DOF_S as i64);
        let _ = s_column.copy_(realized_s);
        let prefix = supports.bin_ids(&dof).reshape([-1, BAR_DOF as i64]);
        let probs = head
            .logits(&flat, &prefix)
            .select(1, DOF_U as i64)
            .softmax(-1, Kind::Double);
        let centers = Tensor::from_slice(supports.centers(DOF_R))
            .to_kind(Kind::Double)
            .view([1, NUM_BAR_BINS as i64]);
        let mu = (&probs * &centers).sum_dim_intlist([-1i64].as_slice(), false, Kind::Double);
        let deviation = &centers - mu.unsqueeze(-1);
        let var = (&probs * &deviation * &deviation)
            .sum_dim_intlist([-1i64].as_slice(), false, Kind::Double);
        let mu = Vec::<f64>::try_from(mu).expect("mu");
        let var = Vec::<f64>::try_from(var).expect("var");
        let r = Vec::<f64>::try_from(realized_r.reshape([-1]).to_kind(Kind::Double)).expect("r");
        let per_window = bars as usize;
        SkillPanel {
            windows: (0..windows as usize)
                .map(|window| SkillWindow {
                    bars: (0..per_window)
                        .map(|bar| {
                            let index = window * per_window + bar;
                            SkillBar {
                                mu: mu[index],
                                sigma: var[index].max(0.0).sqrt(),
                                r: r[index],
                                free: mu[index],
                            }
                        })
                        .collect(),
                    symbol: symbols[window],
                    block: blocks[window],
                })
                .collect(),
        }
    }

    /// The chart must round-trip through the same reader the TUI and `report_cli` use, and its
    /// base must be registered, or it is a chart nobody can see.
    #[test]
    fn the_skill_chart_round_trips_with_a_complete_finite_series() {
        let mut rows = Vec::new();
        let mut next = stream(0x51D5_0007);
        for block in 0..12u64 {
            for slot in 0..300 {
                let confidence = (f64::from(slot) + 1.0) / 300.0;
                let heads = next() > 0.5;
                let correct = next() < 0.5 + 0.2 * confidence;
                let r: f64 = if heads { 1e-3 } else { -1e-3 };
                rows.push((
                    (block % 5) as u32,
                    block,
                    confidence * 1e-4 * if correct { r.signum() } else { -r.signum() },
                    1e-3,
                    r,
                ));
            }
        }
        let panel = panel_from(&rows);
        let profile = SkillProfile::measure(&panel, "round trip", &SkillCutpoints::from_panel(&panel));
        let dir = std::env::temp_dir().join(format!(
            "skill_profile_round_trip_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        write_skill_profile(&dir, &profile).expect("write");
        let path = dir.join(format!("{SKILL_PROFILE_BASE}.report.bin"));
        let report = read_report(&path).expect("read back");
        match report.kind {
            ReportKind::MultiLine { series } => {
                assert!(series.len() > 20, "only {} series", series.len());
                for entry in &series {
                    assert_eq!(
                        entry.values.len(),
                        DECILES,
                        "series {} has {} points",
                        entry.label,
                        entry.values.len()
                    );
                }
                assert!(series.iter().any(|s| s.label.contains("break-even bps")));
                assert!(series.iter().any(|s| s.label.contains("balanced accuracy")));
                assert!(series
                    .iter()
                    .any(|s| s.label.contains("MATCHED measured one-way cost")));
                assert!(series
                    .iter()
                    .any(|s| s.label.contains("vol-scaled cost bps")));
            }
            other => panic!("wrong kind: {other:?}"),
        }
        assert!(
            shared::report::PRETRAIN_REPORT_BASES.contains(&SKILL_PROFILE_BASE),
            "{SKILL_PROFILE_BASE} is not in PRETRAIN_REPORT_BASES"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
