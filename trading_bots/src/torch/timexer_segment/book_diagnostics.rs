//! The falsifier for the book's two open design bets, measured rather than argued.
//!
//! Bet one is that the predictive `std` head carries information a plain rank on `mu` does not.
//! Bet two is that combining horizons beats the single best horizon. Both are settled here, on
//! the validation partition, before any capital simulation reads them: if [`BookDiagnostics::
//! aggregated_ic`] does not clear the best entry of [`BookDiagnostics::ic_by_horizon`], the
//! `Precision` aggregation is decoration and the book should trade `Single` at the best horizon;
//! if IC is flat across predicted-sigma deciles, sigma is not a selection signal and its only
//! remaining jobs are vol targeting, inverse-variance weights and the no-trade band.
//!
//! Every correlation here is WITHIN-TIMESTAMP and cross-sectional, averaged over timestamps,
//! following [`super::runner`]'s scorer convention exactly: a timestamp contributes only when it
//! carries at least [`CROSS_SECTION_MIN`] usable names and both series have spread in it. A
//! timestamp that fails the gate is excluded and counted, never folded in as a zero - pooling
//! thin or degenerate cross-sections into the mean is the standard way a panel IC is silently
//! biased toward zero, and an unmeasured IC is reported as NaN rather than as a measured zero.
//!
//! Deciles are cut WITHIN each timestamp. A pooled decile on predicted `std` would mostly
//! re-rank tickers by their unconditional volatility, so it would answer a question about the
//! ticker universe rather than about the uncertainty head.

use std::collections::HashMap;

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

use super::book::{name_edge, BookConfig, HorizonPath};

/// Minimum names in a DECILE before a correlation is taken.
///
/// Deliberately below [`CROSS_SECTION_MIN`] of 20, which gates the WHOLE cross-section: a decile
/// of a 256-name draw is ~25 names, and the same floor applied to a decile would throw away every
/// decile of a draw narrower than 200 names. Eight is the point where a Pearson correlation still
/// has more signal than its own small-sample bias.
const MIN_NAMES: usize = 8;
/// Minimum names in a whole cross-section, identical to [`super::runner::CROSS_SECTION_MIN`], so
/// that a per-horizon IC measured here and the one the run reports gate the same timestamps.
const CROSS_SECTION_MIN: usize = 20;
const DECILES: usize = 10;
/// Below this centered sum of squares a series is treated as constant, matching the scorer's
/// `spread.gt(1e-12)` gate.
const SPREAD_FLOOR: f64 = 1e-12;
const TWO_SIGMA: f64 = 1.959_963_984_540_054;
const NOMINAL_1_SIGMA: f64 = 0.682_689_492_137_086;
const NOMINAL_2_SIGMA: f64 = 0.95;

/// The run's own reported validation cross-sectional IC, read from the `timexer_segment_signal`
/// report's `held-out full close cross-sectional IC (mean over timestamps)` series, as a
/// reproduction check. These numbers were produced by a different code path on the same
/// partition, so a large disagreement means this pass is misaligned (wrong horizon subset,
/// shifted realized window, gained forecasts), not that the edge moved. A disagreement is
/// reported in [`BookDiagnostics::warnings`].
///
/// Five anchors spanning the rise, the peak near h96 and the roll-off, not three: a
/// horizon-index permutation has to match the SHAPE of the curve, which a single midpoint
/// cannot pin down.
const REFERENCE_IC: [(u16, f64); 5] = [
    (1, 0.0180),
    (8, 0.0248),
    (32, 0.0473),
    (64, 0.0616),
    (192, 0.0553),
];
/// Roughly four times the reported IC standard error of `0.0027`, so ordinary draw-to-draw
/// variation between the two passes does not fire the alarm but a misalignment does.
const REFERENCE_TOLERANCE: f64 = 0.012;
/// Below this many contributing timestamps the panel is a thin draw and every verdict on it is
/// reported as provisional.
const THIN_PANEL: usize = 30;
/// Threshold for the RANGE across ten decile estimates, in units of one pairwise difference's
/// standard error. The range of ten iid standard normals averages 3.08 and has a 95th
/// percentile near 4.47, and a difference of two estimates carries `sqrt(2)` of one estimate's
/// standard error, so the familiar two-SE rule applied to a max-minus-min over ten cells fires
/// on roughly three pure-noise panels in five - measured, not asserted. This one fires on one
/// in twenty. The pre-registered DECISION stays a two-SE test, but on the single top-minus-
/// bottom decile contrast, which is one comparison rather than the largest of forty-five.
const RANGE_NULL_95: f64 = 3.2;

/// Populated by the diagnostics pass; charted by reports.rs.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BookDiagnostics {
    pub horizons: Vec<u16>,
    pub ic_by_horizon: Vec<f64>,
    pub ic_se_by_horizon: Vec<f64>,
    /// Rows = predicted-std decile 0..9 (ascending sigma), cols = horizons.
    pub ic_by_std_decile: Vec<Vec<f64>>,
    pub hit_rate_by_std_decile: Vec<Vec<f64>>,
    /// Rows = cross-horizon agreement decile 0..9, cols = horizons.
    pub ic_by_agreement_decile: Vec<Vec<f64>>,
    pub hit_rate_by_agreement_decile: Vec<Vec<f64>>,
    pub aggregated_ic: f64,
    pub aggregated_ic_se: f64,
    pub aggregated_hit_rate: f64,
    pub cross_sections: usize,
    pub coverage_1_sigma: Vec<f64>,
    pub coverage_2_sigma: Vec<f64>,
    /// The horizon the aggregate is scored at, so [`Self::verdict`] can name it without the
    /// config.
    #[serde(default)]
    pub trade_horizon: u16,
    /// (name, timestamp) pairs behind [`Self::aggregated_hit_rate`], which is what turns that
    /// rate into a testable distance from one half.
    #[serde(default)]
    pub aggregated_names: f64,
    #[serde(default)]
    pub ic_by_std_decile_se: Vec<Vec<f64>>,
    #[serde(default)]
    pub ic_by_agreement_decile_se: Vec<Vec<f64>>,
    /// Loud, not silent: misalignment against [`REFERENCE_IC`], dropped names, missing realized
    /// cross-sections and thin draws land here instead of quietly moving a reported number.
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy, Default)]
struct IcMoments {
    n: f64,
    sum: f64,
    sum_squares: f64,
}

impl IcMoments {
    fn push(&mut self, ic: f64) {
        if ic.is_finite() {
            self.n += 1.;
            self.sum += ic;
            self.sum_squares += ic * ic;
        }
    }

    fn mean(self) -> f64 {
        if self.n > 0. {
            self.sum / self.n
        } else {
            f64::NAN
        }
    }

    fn se(self) -> f64 {
        if self.n >= 2. {
            let mean = self.sum / self.n;
            ((self.sum_squares / self.n - mean * mean).max(0.) / self.n).sqrt()
        } else {
            f64::NAN
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Hits {
    hit: f64,
    n: f64,
}

impl Hits {
    fn push(&mut self, forecast: f64, realized: f64) {
        if forecast == 0. || realized == 0. {
            return;
        }
        self.n += 1.;
        if forecast.signum() == realized.signum() {
            self.hit += 1.;
        }
    }

    fn rate(self) -> f64 {
        if self.n > 0. {
            self.hit / self.n
        } else {
            f64::NAN
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Cell {
    ic: IcMoments,
    hits: Hits,
}

#[derive(Clone, Copy, Default)]
struct Coverage {
    one: f64,
    two: f64,
    n: f64,
}

impl Coverage {
    fn push(&mut self, realized: f64, deviation: f64) {
        self.n += 1.;
        let magnitude = realized.abs();
        if magnitude <= deviation {
            self.one += 1.;
        }
        if magnitude <= TWO_SIGMA * deviation {
            self.two += 1.;
        }
    }

    fn share(self, inside: f64) -> f64 {
        if self.n > 0. {
            inside / self.n
        } else {
            f64::NAN
        }
    }
}

fn correlation(pairs: &[(f64, f64)]) -> Option<f64> {
    if pairs.len() < MIN_NAMES {
        return None;
    }
    let n = pairs.len() as f64;
    let mean_x = pairs.iter().map(|pair| pair.0).sum::<f64>() / n;
    let mean_y = pairs.iter().map(|pair| pair.1).sum::<f64>() / n;
    let (mut xx, mut yy, mut xy) = (0., 0., 0.);
    for &(x, y) in pairs {
        let (dx, dy) = (x - mean_x, y - mean_y);
        xx += dx * dx;
        yy += dy * dy;
        xy += dx * dy;
    }
    if xx <= SPREAD_FLOOR || yy <= SPREAD_FLOOR {
        return None;
    }
    let ic = xy / (xx * yy).sqrt();
    ic.is_finite().then_some(ic)
}

/// Half-open index range of one decile of `n` ranked names. Contiguous by construction, so the
/// deciles partition the cross-section exactly however the ties fall.
fn bucket(n: usize, decile: usize) -> (usize, usize) {
    (n * decile / DECILES, n * (decile + 1) / DECILES)
}

/// Whether a ranking key has any spread in this cross-section. A tied key cuts ten arbitrary
/// buckets out of one value and would manufacture decile structure where the key carries no
/// information; [`super::book::NameEdge::agreement`] is exactly `0.5` for every name whenever a
/// single horizon is eligible, so this is an ordinary case rather than a corner.
fn spread(order: &[usize], key: impl Fn(usize) -> f64) -> bool {
    match (order.first(), order.last()) {
        (Some(&low), Some(&high)) => key(high) - key(low) > 0.,
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate(
    cells: &mut [Cell],
    decile: usize,
    horizon: usize,
    width: usize,
    members: &[usize],
    forecast: &[f64],
    outcome: &[f64],
    pairs: &mut Vec<(f64, f64)>,
) {
    pairs.clear();
    for &name in members {
        let (f, y) = (
            forecast[name * width + horizon],
            outcome[name * width + horizon],
        );
        if f.is_finite() && y.is_finite() {
            pairs.push((f, y));
        }
    }
    if pairs.len() < MIN_NAMES {
        return;
    }
    let cell = &mut cells[decile * width + horizon];
    if let Some(ic) = correlation(pairs) {
        cell.ic.push(ic);
    }
    for &(f, y) in pairs.iter() {
        cell.hits.push(f, y);
    }
}

/// The horizon subset the panel is scored on, validated to be identical in every path: two
/// different subsets in one panel would silently compare horizon `i` of one name against
/// horizon `j` of another.
fn panel_horizons(frames: &[(i64, Vec<(usize, f64, HorizonPath)>)]) -> Result<Vec<u16>> {
    let mut horizons: Option<&[u16]> = None;
    for (timestamp, quotes) in frames {
        for (asset, _, path) in quotes {
            ensure!(
                path.mean.len() == path.horizons.len() && path.std.len() == path.horizons.len(),
                "asset {asset} at {timestamp} carries {} means and {} stds against {} horizons",
                path.mean.len(),
                path.std.len(),
                path.horizons.len()
            );
            match horizons {
                None => horizons = Some(&path.horizons),
                Some(seen) => ensure!(
                    seen == path.horizons,
                    "asset {asset} at {timestamp} carries horizons {:?} against the panel's {:?}",
                    path.horizons,
                    seen
                ),
            }
        }
    }
    let horizons = horizons.unwrap_or_default().to_vec();
    ensure!(
        !horizons.is_empty(),
        "the forecast panel carries no horizons"
    );
    Ok(horizons)
}

/// `frames` is per-timestamp `(ts_ms, [(asset, sigma_k, un-gained path)])`; `realized` is
/// per-timestamp `(ts_ms, [(asset, cumulative sigma-unit residual return per horizon)])` on the
/// SAME horizon subset and in the same order. Names are matched by asset within a timestamp, so
/// the two sides need neither the same order nor the same membership; anything unmatched is
/// dropped and counted in [`BookDiagnostics::warnings`].
///
/// The aggregate is scored in SIGMA units, by dividing [`super::book::NameEdge::mu`] back by
/// `sigma_k`. Scoring the raw-unit `mu` against the raw-unit outcome would measure the
/// aggregation and the cross-sectional dispersion of `sigma_k` at once, and the whole point of
/// the number is to be comparable, like for like, against [`BookDiagnostics::ic_by_horizon`].
///
/// `reproduction`, when supplied, is the SAME per-horizon measurement taken on the in-run
/// scorer's own full held-out population. It is what separates the two things the
/// [`REFERENCE_IC`] check cannot tell apart on the book panel alone: a broken reconstruction,
/// which misses the reference on every population, and a traded universe whose edge genuinely
/// differs from the full cross-section's.
pub fn measure(
    frames: &[(i64, Vec<(usize, f64, HorizonPath)>)],
    realized: &[(i64, Vec<(usize, Vec<f32>)>)],
    cfg: &BookConfig,
    reproduction: Option<(&[u16], &CohortIc)>,
) -> Result<BookDiagnostics> {
    ensure!(
        !frames.is_empty(),
        "book diagnostics need at least one forecast cross-section"
    );
    let horizons = panel_horizons(frames)?;
    let width = horizons.len();
    let mut outcomes: HashMap<i64, &Vec<(usize, Vec<f32>)>> =
        HashMap::with_capacity(realized.len());
    for (timestamp, rows) in realized {
        ensure!(
            outcomes.insert(*timestamp, rows).is_none(),
            "duplicate realized cross-section at {timestamp}"
        );
    }
    let trade_index = horizons
        .iter()
        .position(|horizon| *horizon as usize == cfg.trade_horizon);

    let mut per_horizon = vec![IcMoments::default(); width];
    let mut coverage = vec![Coverage::default(); width];
    let mut std_cells = vec![Cell::default(); DECILES * width];
    let mut agreement_cells = vec![Cell::default(); DECILES * width];
    let mut aggregate = IcMoments::default();
    let mut aggregate_hits = Hits::default();

    let mut forecast: Vec<f64> = Vec::new();
    let mut deviation: Vec<f64> = Vec::new();
    let mut outcome: Vec<f64> = Vec::new();
    let mut agreement: Vec<f64> = Vec::new();
    let mut aggregated: Vec<f64> = Vec::new();
    let mut order: Vec<usize> = Vec::new();
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    let mut lookup: HashMap<usize, &[f32]> = HashMap::new();

    let (mut missing, mut unmatched, mut malformed, mut thin, mut edgeless) = (0usize, 0, 0, 0, 0);
    let (mut tied_agreement, mut tied_deviation) = (0usize, 0usize);
    let mut cross_sections = 0usize;
    for (timestamp, quotes) in frames {
        let Some(rows) = outcomes.get(timestamp) else {
            missing += 1;
            continue;
        };
        lookup.clear();
        for (asset, values) in rows.iter() {
            lookup.insert(*asset, values.as_slice());
        }
        forecast.clear();
        deviation.clear();
        outcome.clear();
        agreement.clear();
        aggregated.clear();
        for (asset, sigma, path) in quotes {
            if !sigma.is_finite() || *sigma <= 0. {
                malformed += 1;
                continue;
            }
            let Some(values) = lookup.get(asset) else {
                unmatched += 1;
                continue;
            };
            if values.len() != width {
                malformed += 1;
                continue;
            }
            for index in 0..width {
                forecast.push(path.mean[index] as f64);
                deviation.push(path.std[index] as f64);
                outcome.push(values[index] as f64);
            }
            match name_edge(path, *sigma, cfg) {
                Some(edge) => {
                    agreement.push(edge.agreement);
                    aggregated.push(edge.mu / sigma);
                }
                None => {
                    edgeless += 1;
                    agreement.push(f64::NAN);
                    aggregated.push(f64::NAN);
                }
            }
        }
        let names = agreement.len();
        if names < CROSS_SECTION_MIN {
            thin += 1;
            continue;
        }
        cross_sections += 1;

        for index in 0..width {
            pairs.clear();
            for name in 0..names {
                let (f, y) = (
                    forecast[name * width + index],
                    outcome[name * width + index],
                );
                if f.is_finite() && y.is_finite() {
                    pairs.push((f, y));
                }
            }
            if let Some(ic) = correlation(&pairs) {
                per_horizon[index].push(ic);
            }
            for name in 0..names {
                let (s, y) = (
                    deviation[name * width + index],
                    outcome[name * width + index],
                );
                if s.is_finite() && s > 0. && y.is_finite() {
                    coverage[index].push(y, s);
                }
            }
        }

        // Agreement is one number per name, so its ranking is cut once and every horizon reads
        // the same membership; predicted std is per horizon, so its ranking is cut per horizon.
        order.clear();
        order.extend((0..names).filter(|&name| agreement[name].is_finite()));
        order.sort_unstable_by(|&a, &b| agreement[a].total_cmp(&agreement[b]));
        if spread(&order, |name| agreement[name]) {
            for decile in 0..DECILES {
                let (lo, hi) = bucket(order.len(), decile);
                for index in 0..width {
                    accumulate(
                        &mut agreement_cells,
                        decile,
                        index,
                        width,
                        &order[lo..hi],
                        &forecast,
                        &outcome,
                        &mut pairs,
                    );
                }
            }
        } else {
            tied_agreement += 1;
        }
        for index in 0..width {
            order.clear();
            order.extend((0..names).filter(|&name| deviation[name * width + index].is_finite()));
            order.sort_unstable_by(|&a, &b| {
                deviation[a * width + index].total_cmp(&deviation[b * width + index])
            });
            if !spread(&order, |name| deviation[name * width + index]) {
                tied_deviation += 1;
                continue;
            }
            for decile in 0..DECILES {
                let (lo, hi) = bucket(order.len(), decile);
                accumulate(
                    &mut std_cells,
                    decile,
                    index,
                    width,
                    &order[lo..hi],
                    &forecast,
                    &outcome,
                    &mut pairs,
                );
            }
        }

        if let Some(index) = trade_index {
            pairs.clear();
            for name in 0..names {
                let (f, y) = (aggregated[name], outcome[name * width + index]);
                if f.is_finite() && y.is_finite() {
                    pairs.push((f, y));
                }
            }
            if let Some(ic) = correlation(&pairs) {
                aggregate.push(ic);
            }
            for &(f, y) in pairs.iter() {
                aggregate_hits.push(f, y);
            }
        }
    }

    let matrix = |cells: &[Cell], read: fn(&Cell) -> f64| -> Vec<Vec<f64>> {
        (0..DECILES)
            .map(|decile| {
                (0..width)
                    .map(|index| read(&cells[decile * width + index]))
                    .collect()
            })
            .collect()
    };
    let mut diagnostics = BookDiagnostics {
        ic_by_horizon: per_horizon.iter().map(|m| m.mean()).collect(),
        ic_se_by_horizon: per_horizon.iter().map(|m| m.se()).collect(),
        ic_by_std_decile: matrix(&std_cells, |cell| cell.ic.mean()),
        hit_rate_by_std_decile: matrix(&std_cells, |cell| cell.hits.rate()),
        ic_by_agreement_decile: matrix(&agreement_cells, |cell| cell.ic.mean()),
        hit_rate_by_agreement_decile: matrix(&agreement_cells, |cell| cell.hits.rate()),
        ic_by_std_decile_se: matrix(&std_cells, |cell| cell.ic.se()),
        ic_by_agreement_decile_se: matrix(&agreement_cells, |cell| cell.ic.se()),
        aggregated_ic: aggregate.mean(),
        aggregated_ic_se: aggregate.se(),
        aggregated_hit_rate: aggregate_hits.rate(),
        aggregated_names: aggregate_hits.n,
        cross_sections,
        coverage_1_sigma: coverage.iter().map(|c| c.share(c.one)).collect(),
        coverage_2_sigma: coverage.iter().map(|c| c.share(c.two)).collect(),
        trade_horizon: cfg.trade_horizon as u16,
        horizons,
        warnings: Vec::new(),
    };

    if trade_index.is_none() {
        diagnostics.warnings.push(format!(
            "trade horizon h{} is absent from the panel's horizons {:?}; the aggregate is \
             unmeasured and the Precision bet is untested",
            cfg.trade_horizon, diagnostics.horizons
        ));
    }
    if edgeless > 0 {
        diagnostics.warnings.push(format!(
            "book::name_edge returned nothing for {edgeless} names; they carry no aggregate and \
             no agreement decile"
        ));
    }
    for (horizon, reference) in REFERENCE_IC {
        let Some(index) = diagnostics.horizons.iter().position(|h| *h == horizon) else {
            continue;
        };
        let measured = diagnostics.ic_by_horizon[index];
        if !measured.is_finite() {
            diagnostics.warnings.push(format!(
                "h{horizon} IC is unmeasured while the run reports {reference:.4}; the panel is \
                 empty at that horizon"
            ));
            continue;
        }
        let se = diagnostics.ic_se_by_horizon[index];
        let allowed = REFERENCE_TOLERANCE.max(if se.is_finite() { 4. * se } else { 0. });
        let matched = reproduction.and_then(|(horizons, cohort)| {
            let position = horizons.iter().position(|h| *h == horizon)?;
            let ic = *cohort.ic.get(position)?;
            ic.is_finite().then(|| (ic, cohort.se[position]))
        });
        if let Some((full, full_se)) = matched {
            let full_allowed = REFERENCE_TOLERANCE.max(if full_se.is_finite() {
                4. * full_se
            } else {
                0.
            });
            if (full - reference).abs() > full_allowed {
                diagnostics.warnings.push(format!(
                    "h{horizon} IC {full:.4} on the scorer's OWN population disagrees with the \
                     run's reported {reference:.4} by {:.4} (allowed {full_allowed:.4}); this \
                     pass is misaligned - horizon subset, realized window or gain - and every \
                     number below it is suspect",
                    full - reference
                ));
            } else if (measured - full).abs() > allowed {
                diagnostics.warnings.push(format!(
                    "h{horizon} IC {measured:.4} on the traded universe against {full:.4} on the \
                     scorer's full cross-section, {:.4}; the same pass reproduces the run's \
                     reported {reference:.4} on that population, so this is the universe \
                     restriction and not a misalignment",
                    measured - full
                ));
            }
            continue;
        }
        if (measured - reference).abs() > allowed {
            diagnostics.warnings.push(format!(
                "h{horizon} IC {measured:.4} disagrees with the run's reported {reference:.4} by \
                 {:.4} (allowed {allowed:.4}); this pass is misaligned - horizon subset, realized \
                 window or gain - and every number below it is suspect",
                measured - reference
            ));
        }
    }
    if missing > 0 {
        diagnostics.warnings.push(format!(
            "{missing} forecast cross-sections had no realized cross-section at their timestamp"
        ));
    }
    if unmatched > 0 {
        diagnostics
            .warnings
            .push(format!("{unmatched} forecast names had no realized row"));
    }
    if malformed > 0 {
        diagnostics.warnings.push(format!(
            "{malformed} names were dropped for a nonpositive sigma or a realized row of the \
             wrong width"
        ));
    }
    if thin > 0 {
        diagnostics.warnings.push(format!(
            "{thin} cross-sections carried fewer than {CROSS_SECTION_MIN} usable names and were \
             excluded, not scored as zero"
        ));
    }
    if tied_agreement > 0 {
        diagnostics.warnings.push(format!(
            "{tied_agreement} cross-sections carried no spread in agreement and were not cut \
             into agreement deciles; a single eligible horizon pins every name at 0.5"
        ));
    }
    if tied_deviation > 0 {
        diagnostics.warnings.push(format!(
            "{tied_deviation} (cross-section, horizon) cells carried no spread in predicted std \
             and were not cut into std deciles"
        ));
    }
    if cross_sections < THIN_PANEL {
        diagnostics.warnings.push(format!(
            "{cross_sections} contributing cross-sections is a thin draw; every verdict here is \
             provisional"
        ));
    }
    Ok(diagnostics)
}

/// Per-horizon within-timestamp IC over an ARBITRARY cohort partition, gated exactly as
/// [`measure`] gates the book's own cross-sections.
///
/// This exists so the traded 256-name population and the in-run scorer's own population - every
/// `Corpus::validation_refs` origin, grouped into the coincidence cohorts their shared origin
/// timestamps form - can be measured by ONE piece of arithmetic and reported as two numbers. A
/// disagreement between the book's IC and the run's then has exactly one remaining explanation,
/// the universe, instead of two.
///
/// `bounds` are `(start, len)` spans into the row-major `forecast`/`realized` panels, both
/// `rows * width` long.
pub(super) struct CohortIc {
    pub ic: Vec<f64>,
    pub se: Vec<f64>,
    pub cross_sections: Vec<usize>,
}

pub(super) fn cohort_horizon_ic(
    width: usize,
    bounds: &[(usize, usize)],
    forecast: &[f32],
    realized: &[f32],
) -> Result<CohortIc> {
    ensure!(
        width > 0 && forecast.len() == realized.len() && forecast.len() % width == 0,
        "cohort panels must be rectangular and nonempty in the horizon axis"
    );
    let mut moments = vec![IcMoments::default(); width];
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    for &(start, len) in bounds {
        if len < CROSS_SECTION_MIN {
            continue;
        }
        for index in 0..width {
            pairs.clear();
            for row in start..start + len {
                let (f, y) = (
                    f64::from(forecast[row * width + index]),
                    f64::from(realized[row * width + index]),
                );
                if f.is_finite() && y.is_finite() {
                    pairs.push((f, y));
                }
            }
            if pairs.len() < CROSS_SECTION_MIN {
                continue;
            }
            if let Some(ic) = correlation(&pairs) {
                moments[index].push(ic);
            }
        }
    }
    Ok(CohortIc {
        ic: moments.iter().map(|m| m.mean()).collect(),
        se: moments.iter().map(|m| m.se()).collect(),
        cross_sections: moments.iter().map(|m| m.n as usize).collect(),
    })
}

fn best_horizon(diagnostics: &BookDiagnostics) -> Option<(usize, f64, f64)> {
    diagnostics
        .ic_by_horizon
        .iter()
        .enumerate()
        .filter(|(_, ic)| ic.is_finite())
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, ic)| (index, *ic, diagnostics.ic_se_by_horizon[index]))
}

/// Largest finite gap between two deciles of one horizon's column, with the standard error of
/// that difference treating the two deciles as independent draws.
fn decile_spread(
    values: &[Vec<f64>],
    errors: &[Vec<f64>],
    index: usize,
) -> Option<(usize, usize, f64, f64)> {
    let column: Vec<(usize, f64, f64)> = values
        .iter()
        .enumerate()
        .filter_map(|(decile, row)| {
            let ic = *row.get(index)?;
            let se = errors
                .get(decile)
                .and_then(|row| row.get(index))
                .copied()
                .unwrap_or(f64::NAN);
            ic.is_finite().then_some((decile, ic, se))
        })
        .collect();
    let low = column.iter().min_by(|a, b| a.1.total_cmp(&b.1))?;
    let high = column.iter().max_by(|a, b| a.1.total_cmp(&b.1))?;
    let se = (low.2 * low.2 + high.2 * high.2).sqrt();
    Some((low.0, high.0, high.1 - low.1, se))
}

/// The pre-registered contrast: top decile minus bottom decile of one horizon's column, and
/// the standard error of that single difference. One comparison, so a two-SE rule on it means
/// what a two-SE rule is supposed to mean.
fn decile_contrast(values: &[Vec<f64>], errors: &[Vec<f64>], index: usize) -> Option<(f64, f64)> {
    let read = |row: usize| -> Option<(f64, f64)> {
        let ic = *values.get(row)?.get(index)?;
        let se = *errors.get(row)?.get(index)?;
        (ic.is_finite() && se.is_finite()).then_some((ic, se))
    };
    let (low, low_se) = read(0)?;
    let (high, high_se) = read(DECILES - 1)?;
    Some((high - low, (low_se * low_se + high_se * high_se).sqrt()))
}

impl BookDiagnostics {
    /// Pre-registered conclusions, one per line, in the words the design was written in. Facts
    /// and distances in standard errors only; the caller decides nothing here.
    pub fn verdict(&self) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(format!(
            "panel: {} cross-sections over {} horizons, aggregate scored at h{}",
            self.cross_sections,
            self.horizons.len(),
            self.trade_horizon
        ));
        let index = self
            .horizons
            .iter()
            .position(|horizon| *horizon == self.trade_horizon);
        let best = best_horizon(self);
        match best {
            Some((position, ic, se)) => lines.push(format!(
                "best single horizon: h{} IC {ic:.4} (SE {se:.4})",
                self.horizons[position]
            )),
            None => lines.push("best single horizon: unmeasured, no horizon carried an IC".into()),
        }
        match (best, self.aggregated_ic.is_finite()) {
            (Some((position, ic, se)), true) => {
                let combined = (se * se + self.aggregated_ic_se * self.aggregated_ic_se).sqrt();
                let delta = self.aggregated_ic - ic;
                let sigmas = if combined > 0. {
                    delta / combined
                } else {
                    f64::NAN
                };
                let call = if sigmas >= 2. {
                    "cross-horizon aggregation BEATS the best single horizon"
                } else if sigmas <= -2. {
                    "cross-horizon aggregation LOSES to the best single horizon; trade Single"
                } else {
                    "cross-horizon aggregation is INDISTINGUISHABLE from the best single horizon; \
                     Precision is unproven"
                };
                lines.push(format!(
                    "{call}: aggregate IC {:.4} (SE {:.4}) vs h{} {ic:.4}, {delta:+.4} = \
                     {sigmas:+.2} SE (unpaired, so conservative)",
                    self.aggregated_ic, self.aggregated_ic_se, self.horizons[position]
                ));
            }
            _ => lines.push(
                "cross-horizon aggregation: unmeasured, the Precision bet is untested".into(),
            ),
        }
        if self.aggregated_names > 0. {
            let se = (0.25 / self.aggregated_names).sqrt();
            lines.push(format!(
                "aggregate directional hit rate {:.4} on {:.0} name-timestamps, {:+.2} SE from a \
                 coin flip",
                self.aggregated_hit_rate,
                self.aggregated_names,
                (self.aggregated_hit_rate - 0.5) / se
            ));
        }
        for (label, values, errors) in [
            (
                "predicted sigma",
                &self.ic_by_std_decile,
                &self.ic_by_std_decile_se,
            ),
            (
                "cross-horizon agreement",
                &self.ic_by_agreement_decile,
                &self.ic_by_agreement_decile_se,
            ),
        ] {
            match index.and_then(|index| decile_contrast(values, errors, index)) {
                Some((delta, se)) => {
                    let sigmas = if se > 0. { delta / se } else { f64::NAN };
                    let call = if sigmas.abs() >= 2. {
                        "SELECTS"
                    } else {
                        "does NOT select"
                    };
                    lines.push(format!(
                        "{label} {call} at h{}: top decile IC minus bottom decile IC is \
                         {delta:+.4} = {sigmas:+.2} SE (pre-registered at 2 SE)",
                        self.trade_horizon
                    ));
                }
                None => lines.push(format!(
                    "{label}: the top and bottom deciles carried no IC at h{}",
                    self.trade_horizon
                )),
            }
            if let Some((low, high, delta, se)) =
                index.and_then(|index| decile_spread(values, errors, index))
            {
                let sigmas = if se > 0. { delta / se } else { f64::NAN };
                let call = if sigmas >= RANGE_NULL_95 {
                    "carries non-monotone decile structure"
                } else {
                    "is flat across deciles"
                };
                lines.push(format!(
                    "{label} {call} at h{}: widest gap is decile {high} over decile {low}, \
                     {delta:.4} = {sigmas:.2} SE against a {RANGE_NULL_95:.1} SE null range",
                    self.trade_horizon
                ));
            }
        }
        match index.and_then(|index| {
            let one = *self.coverage_1_sigma.get(index)?;
            let two = *self.coverage_2_sigma.get(index)?;
            (one.is_finite() && two.is_finite()).then_some((one, two))
        }) {
            Some((one, two)) => {
                let call = if one > NOMINAL_1_SIGMA + 0.02 {
                    "OVER-dispersed (predicted std too wide); ex-ante vol targeting will \
                     systematically undershoot"
                } else if one < NOMINAL_1_SIGMA - 0.02 {
                    "UNDER-dispersed (predicted std too tight); ex-ante vol targeting will \
                     systematically overshoot"
                } else {
                    "CALIBRATED; predicted std is usable for vol targeting as is"
                };
                lines.push(format!(
                    "predictive distribution at h{} is {call}: 1-sigma coverage {one:.4} against \
                     {NOMINAL_1_SIGMA:.4}, 2-sigma {two:.4} against {NOMINAL_2_SIGMA:.4}",
                    self.trade_horizon
                ));
            }
            None => lines.push(format!(
                "predictive coverage at h{} is unmeasured",
                self.trade_horizon
            )),
        }
        for warning in &self.warnings {
            lines.push(format!("WARNING: {warning}"));
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::timexer_segment::book::{Aggregation, Selection, Weighting};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    const HORIZONS: [u16; 4] = [1, 8, 32, 64];
    const BAR_MS: i64 = 300_000;

    fn config() -> BookConfig {
        BookConfig {
            trade_horizon: 64,
            aggregation: Aggregation::Precision,
            agg_min_horizon: 8,
            agg_max_horizon: 128,
            agg_half_life: 32.,
            selection: Selection::Decile,
            selection_n: 25,
            weighting: Weighting::MeanOverVar,
            target_vol_annual: 0.15,
            gross_cap: 2.,
            per_name_cap: 0.05,
            no_trade_band_cost_multiple: 2.,
        }
    }

    fn normal(rng: &mut ChaCha8Rng) -> f64 {
        let u1: f64 = rng.random::<f64>().max(1e-12);
        let u2: f64 = rng.random::<f64>();
        (-2. * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn path(mean: &[f64], deviation: &[f64]) -> HorizonPath {
        HorizonPath {
            mean: mean.iter().map(|value| *value as f32).collect(),
            std: deviation.iter().map(|value| *value as f32).collect(),
            horizons: HORIZONS.to_vec(),
        }
    }

    fn flat(value: f64) -> Vec<f64> {
        vec![value; HORIZONS.len()]
    }

    /// `name(rng)` returns one name's `(mean, std, realized)` columns over [`HORIZONS`], all in
    /// sigma units, with `sigma_k` pinned at one so raw and sigma units coincide and the
    /// aggregate is scored on the same scale as the per-horizon columns.
    fn panel(
        timestamps: usize,
        names: usize,
        seed: u64,
        mut name: impl FnMut(&mut ChaCha8Rng) -> (Vec<f64>, Vec<f64>, Vec<f64>),
    ) -> (
        Vec<(i64, Vec<(usize, f64, HorizonPath)>)>,
        Vec<(i64, Vec<(usize, Vec<f32>)>)>,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut frames = Vec::with_capacity(timestamps);
        let mut realized = Vec::with_capacity(timestamps);
        for step in 0..timestamps {
            let timestamp = 1_735_833_600_000 + step as i64 * BAR_MS;
            let mut quotes = Vec::with_capacity(names);
            let mut rows = Vec::with_capacity(names);
            for index in 0..names {
                let (mean, deviation, outcome) = name(&mut rng);
                quotes.push((index, 1.0, path(&mean, &deviation)));
                rows.push((
                    index,
                    outcome
                        .iter()
                        .map(|value| *value as f32)
                        .collect::<Vec<f32>>(),
                ));
            }
            frames.push((timestamp, quotes));
            realized.push((timestamp, rows));
        }
        (frames, realized)
    }

    /// The forecast IS the outcome. The name's value is scaled up with the horizon so the path
    /// is not rate-constant, which leaves every horizon's cross-sectional RANKING identical and
    /// therefore every per-horizon IC exactly one.
    fn perfect(rng: &mut ChaCha8Rng) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let value = normal(rng);
        let mean = (0..HORIZONS.len())
            .map(|index| value * (1. + 0.1 * index as f64))
            .collect();
        (mean, flat(1.), flat(value))
    }

    #[test]
    fn a_perfect_forecast_scores_unit_ic_and_the_coverage_of_its_own_draw() {
        let (frames, realized) = panel(64, 40, 0x9E37, perfect);
        let diagnostics = measure(&frames, &realized, &config(), None).unwrap();
        assert_eq!(diagnostics.cross_sections, 64);
        for (index, horizon) in HORIZONS.iter().enumerate() {
            let ic = diagnostics.ic_by_horizon[index];
            assert!((ic - 1.).abs() < 1e-9, "h{horizon} scored {ic}");
        }
        // Realized is a standard normal draw against a predicted std of one, so coverage is the
        // draw's own tail mass rather than an artifact of the estimator.
        for index in 0..HORIZONS.len() {
            assert!(
                (diagnostics.coverage_1_sigma[index] - NOMINAL_1_SIGMA).abs() < 0.04,
                "1-sigma coverage {}",
                diagnostics.coverage_1_sigma[index]
            );
            assert!(
                (diagnostics.coverage_2_sigma[index] - NOMINAL_2_SIGMA).abs() < 0.04,
                "2-sigma coverage {}",
                diagnostics.coverage_2_sigma[index]
            );
        }
        // Every horizon of a name is a positive multiple of one value and every predicted std
        // is one, so any positively weighted aggregate ranks the cross-section perfectly.
        if diagnostics.aggregated_ic.is_finite() {
            assert!(
                diagnostics.aggregated_ic > 0.99,
                "aggregate scored {}",
                diagnostics.aggregated_ic
            );
            assert!(diagnostics.aggregated_hit_rate > 0.99);
        }
    }

    #[test]
    fn a_noise_panel_reports_no_edge_and_no_decile_structure() {
        // Independent means per horizon so the agreement cut has spread, a per-name predicted
        // std so the sigma cut has spread, and an outcome independent of both. 120 names is
        // twelve per decile, above the MIN_NAMES floor, so the decile cells are measured at all.
        let (frames, realized) = panel(128, 120, 0x5EED, |rng| {
            let deviation = (0.3 * normal(rng)).exp();
            let mean = (0..HORIZONS.len()).map(|_| normal(rng)).collect();
            let outcome = (0..HORIZONS.len()).map(|_| normal(rng)).collect();
            (mean, flat(deviation), outcome)
        });
        let diagnostics = measure(&frames, &realized, &config(), None).unwrap();
        // Four standard errors, not two: the draw is seeded, but a two-SE bound on four
        // horizons and eighty decile cells would fail one run in six against a CORRECT
        // estimator, and a broken one misses by ten SE or reports NaN.
        for (index, horizon) in HORIZONS.iter().enumerate() {
            let (ic, se) = (
                diagnostics.ic_by_horizon[index],
                diagnostics.ic_se_by_horizon[index],
            );
            assert!(ic.abs() < 4. * se, "h{horizon} scored {ic} against SE {se}");
        }
        if diagnostics.aggregated_ic.is_finite() {
            assert!(
                diagnostics.aggregated_ic.abs() < 4. * diagnostics.aggregated_ic_se,
                "aggregate scored {} against SE {}",
                diagnostics.aggregated_ic,
                diagnostics.aggregated_ic_se
            );
        }
        let mut measured = 0usize;
        for (label, values, errors) in [
            (
                "std",
                &diagnostics.ic_by_std_decile,
                &diagnostics.ic_by_std_decile_se,
            ),
            (
                "agreement",
                &diagnostics.ic_by_agreement_decile,
                &diagnostics.ic_by_agreement_decile_se,
            ),
        ] {
            for (decile, row) in values.iter().enumerate() {
                for (index, ic) in row.iter().enumerate() {
                    if !ic.is_finite() {
                        continue;
                    }
                    measured += 1;
                    let se = errors[decile][index];
                    assert!(
                        ic.abs() < 5. * se,
                        "{label} decile {decile} at h{} scored {ic} against SE {se}",
                        HORIZONS[index]
                    );
                }
            }
        }
        assert!(measured > 0, "no decile cell was measured at all");
        // The pre-registered contrast itself, at the trade horizon, on both cuts: a noise
        // panel must not put the top decile away from the bottom one.
        let horizon = HORIZONS.len() - 1;
        for (label, values, errors) in [
            (
                "std",
                &diagnostics.ic_by_std_decile,
                &diagnostics.ic_by_std_decile_se,
            ),
            (
                "agreement",
                &diagnostics.ic_by_agreement_decile,
                &diagnostics.ic_by_agreement_decile_se,
            ),
        ] {
            let (delta, se) = decile_contrast(values, errors, horizon).unwrap();
            assert!(
                delta.abs() < 4. * se,
                "{label} top-minus-bottom contrast {delta} against SE {se}"
            );
        }
        // The verdict reports the sigma cut at the trade horizon rather than going silent on
        // it; which way it calls a given seeded draw is the draw's business, not the test's.
        assert!(
            diagnostics
                .verdict()
                .iter()
                .any(|line| line.starts_with("predicted sigma")
                    && line.contains("top decile IC minus bottom decile IC")),
            "{:?}",
            diagnostics.verdict()
        );
    }

    #[test]
    fn thin_and_constant_cross_sections_are_excluded_rather_than_scored_as_zero() {
        let (mut frames, mut realized) = panel(20, 30, 0xC0FFEE, perfect);
        // Three names: below any correlation's floor, and a perfectly ANTI-correlated one, so
        // folding it in as a scored cross-section would be visible in the mean.
        let thin = 1_800_000_000_000i64;
        frames.push((
            thin,
            (0..3)
                .map(|name| (name, 1.0, path(&flat(name as f64), &flat(1.))))
                .collect(),
        ));
        realized.push((
            thin,
            (0..3)
                .map(|name| (name, vec![-(name as f32); HORIZONS.len()]))
                .collect(),
        ));
        // Wide cross-section, constant forecast: the correlation is undefined, not zero.
        let flat_stamp = thin + BAR_MS;
        frames.push((
            flat_stamp,
            (0..30)
                .map(|name| (name, 1.0, path(&flat(0.5), &flat(1.))))
                .collect(),
        ));
        realized.push((
            flat_stamp,
            (0..30)
                .map(|name| (name, vec![(name as f32) - 15.; HORIZONS.len()]))
                .collect(),
        ));
        let diagnostics = measure(&frames, &realized, &config(), None).unwrap();
        for (index, horizon) in HORIZONS.iter().enumerate() {
            let ic = diagnostics.ic_by_horizon[index];
            assert!(
                (ic - 1.).abs() < 1e-9,
                "h{horizon} was diluted to {ic} by an excluded cross-section"
            );
        }
        // The three-name stamp never enters; the constant stamp is a scored cross-section that
        // simply contributes no correlation at any horizon.
        assert_eq!(diagnostics.cross_sections, 21);
        assert!(
            diagnostics
                .warnings
                .iter()
                .any(|warning| warning.contains("fewer than")),
            "{:?}",
            diagnostics.warnings
        );
    }

    #[test]
    fn coverage_is_measured_on_the_sigma_unit_scale() {
        // Realized dispersion is exactly twice the predicted std, per name. A coverage on the
        // sigma-unit scale sees 2 * Phi(0.5) - 1 = 0.383 at one sigma and 2 * Phi(0.98) - 1 =
        // 0.673 at two; a coverage that rescaled by the realized dispersion would report the
        // nominal 0.683 and hide the over-confidence entirely.
        let (frames, realized) = panel(64, 40, 0xD15, |rng| {
            let deviation = (0.3 * normal(rng)).exp();
            let outcome = (0..HORIZONS.len())
                .map(|_| 2. * deviation * normal(rng))
                .collect();
            (flat(normal(rng)), flat(deviation), outcome)
        });
        let diagnostics = measure(&frames, &realized, &config(), None).unwrap();
        for index in 0..HORIZONS.len() {
            let one = diagnostics.coverage_1_sigma[index];
            let two = diagnostics.coverage_2_sigma[index];
            assert!((one - 0.3829).abs() < 0.04, "1-sigma coverage {one}");
            assert!((two - 0.6729).abs() < 0.04, "2-sigma coverage {two}");
        }
        assert!(
            diagnostics
                .verdict()
                .iter()
                .any(|line| line.contains("UNDER-dispersed")),
            "{:?}",
            diagnostics.verdict()
        );
    }

    #[test]
    fn a_realized_cross_section_that_is_missing_or_the_wrong_width_is_reported() {
        let (frames, mut realized) = panel(40, 24, 0xBEEF, perfect);
        realized.remove(0);
        realized[0].1[0].1.pop();
        let diagnostics = measure(&frames, &realized, &config(), None).unwrap();
        assert_eq!(diagnostics.cross_sections, 39);
        assert!(
            diagnostics
                .warnings
                .iter()
                .any(|warning| warning.contains("no realized cross-section")),
            "{:?}",
            diagnostics.warnings
        );
        assert!(
            diagnostics
                .warnings
                .iter()
                .any(|warning| warning.contains("wrong width")),
            "{:?}",
            diagnostics.warnings
        );
    }

    #[test]
    fn a_panel_mixing_horizon_subsets_is_refused() {
        let (mut frames, realized) = panel(12, 20, 0x1234, perfect);
        frames[3].1[2].2.horizons = vec![1, 8, 32, 128];
        let error = measure(&frames, &realized, &config(), None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("against the panel's"), "{error}");
    }

    /// Every horizon column must be read against ITS OWN horizon's outcome, and every name
    /// against ITS OWN label. The [`perfect`] panel cannot see either error: it plants one
    /// ranking and reuses it at every horizon, so a horizon shuffle leaves the IC at one.
    ///
    /// Here every (timestamp, horizon) cell carries an INDEPENDENT ranking, and the forecast is
    /// that cell's outcome exactly. Aligned, every per-horizon IC is one; read one horizon out
    /// of step, the two columns are independent draws and the IC collapses. That is the shape
    /// of a reconstruction defect - an index or a pairing, never a positive scalar.
    #[test]
    fn per_horizon_ic_reads_its_own_horizon_and_its_own_name() {
        let (names, steps) = (40usize, 40usize);
        let mut rng = ChaCha8Rng::seed_from_u64(0x5EED);
        let mut frames = Vec::with_capacity(steps);
        let mut realized = Vec::with_capacity(steps);
        for step in 0..steps {
            let timestamp = 1_735_833_600_000 + step as i64 * BAR_MS;
            let mut quotes = Vec::with_capacity(names);
            let mut rows = Vec::with_capacity(names);
            for index in 0..names {
                let outcome: Vec<f64> = (0..HORIZONS.len()).map(|_| normal(&mut rng)).collect();
                quotes.push((index, 1.0, path(&outcome, &flat(1.))));
                rows.push((
                    index,
                    outcome
                        .iter()
                        .map(|value| *value as f32)
                        .collect::<Vec<f32>>(),
                ));
            }
            frames.push((timestamp, quotes));
            realized.push((timestamp, rows));
        }
        let diagnostics = measure(&frames, &realized, &config(), None).unwrap();
        for (index, horizon) in HORIZONS.iter().enumerate() {
            let ic = diagnostics.ic_by_horizon[index];
            assert!((ic - 1.).abs() < 1e-6, "h{horizon} scored {ic}");
        }
        // The same panel read one horizon out of step scores nothing, which is what makes the
        // assertion above a measurement of the indexing rather than of the data.
        let shifted: Vec<(i64, Vec<(usize, Vec<f32>)>)> = realized
            .iter()
            .map(|(timestamp, rows)| {
                let rows = rows
                    .iter()
                    .map(|(asset, row)| {
                        let mut rotated = row.clone();
                        rotated.rotate_left(1);
                        (*asset, rotated)
                    })
                    .collect();
                (*timestamp, rows)
            })
            .collect();
        let confused = measure(&frames, &shifted, &config(), None).unwrap();
        assert!(
            confused.ic_by_horizon[0].abs() < 0.1,
            "a horizon-shifted label must not score like an aligned one: {}",
            confused.ic_by_horizon[0]
        );
        // And the same panel with the LABELS permuted across names, which is the other shape a
        // pairing defect takes.
        let permuted: Vec<(i64, Vec<(usize, Vec<f32>)>)> = realized
            .iter()
            .map(|(timestamp, rows)| {
                let rows = rows
                    .iter()
                    .enumerate()
                    .map(|(position, (_, row))| ((position + 1) % names, row.clone()))
                    .collect();
                (*timestamp, rows)
            })
            .collect();
        let scrambled = measure(&frames, &permuted, &config(), None).unwrap();
        assert!(
            scrambled.ic_by_horizon[0].abs() < 0.1,
            "a name-permuted label must not score like an aligned one: {}",
            scrambled.ic_by_horizon[0]
        );
    }

    /// [`cohort_horizon_ic`] measures the scorer's population with the book's arithmetic, so it
    /// has to recover a planted per-horizon correlation and has to apply the scorer's own
    /// twenty-name floor rather than the decile floor.
    #[test]
    fn cohort_horizon_ic_recovers_a_planted_correlation_and_refuses_thin_cohorts() {
        let width = 3usize;
        let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE);
        let (wide, thin) = (64usize, 19usize);
        let loading = [1.0, 0.5, 0.0];
        let mut forecast = Vec::new();
        let mut realized = Vec::new();
        let mut bounds = Vec::new();
        for cohort in 0..400 {
            let names = if cohort % 2 == 0 { wide } else { thin };
            bounds.push((forecast.len() / width, names));
            for _ in 0..names {
                let signal = normal(&mut rng);
                for index in 0..width {
                    let noise = normal(&mut rng);
                    forecast.push(signal as f32);
                    realized.push(
                        (loading[index] * signal
                            + (1. - loading[index] * loading[index]).sqrt() * noise)
                            as f32,
                    );
                }
            }
        }
        let measured = cohort_horizon_ic(width, &bounds, &forecast, &realized).unwrap();
        for index in 0..width {
            assert_eq!(
                measured.cross_sections[index], 200,
                "only the {wide}-name cohorts clear the {CROSS_SECTION_MIN}-name floor"
            );
            let error = (measured.ic[index] - loading[index]).abs();
            assert!(
                error < 6. * measured.se[index].max(1e-6) && error < 0.05,
                "horizon {index} planted {} measured {} (se {})",
                loading[index],
                measured.ic[index],
                measured.se[index]
            );
        }
    }

    /// The guard has to say which of two very different things happened. A pass that
    /// reconstructs the signal wrongly misses the run's reported IC on EVERY population; a
    /// tradable-universe restriction misses it on the book while the same pass reproduces it on
    /// the scorer's own cross-section. Before the reproduction pass existed the guard could
    /// only ever report the first, which is the right default and stays the default.
    #[test]
    fn the_reference_guard_separates_a_broken_pass_from_a_restricted_universe() {
        let (frames, realized) = panel(400, 40, 0x51DE, |rng| {
            (flat(normal(rng)), flat(1.), flat(normal(rng)))
        });
        let misaligned = |diagnostics: &BookDiagnostics| {
            diagnostics
                .warnings
                .iter()
                .filter(|warning| warning.contains("is misaligned"))
                .count()
        };
        let restricted = |diagnostics: &BookDiagnostics| {
            diagnostics
                .warnings
                .iter()
                .filter(|warning| warning.contains("universe restriction"))
                .count()
        };

        let alone = measure(&frames, &realized, &config(), None).unwrap();
        let anchors = misaligned(&alone);
        assert!(anchors > 0, "{:?}", alone.warnings);
        assert_eq!(restricted(&alone), 0);

        let cohort = |ic: Box<dyn Fn(u16) -> f64>| CohortIc {
            ic: HORIZONS.iter().map(|horizon| ic(*horizon)).collect(),
            se: vec![0.0025; HORIZONS.len()],
            cross_sections: vec![9_506; HORIZONS.len()],
        };
        let reference = |horizon: u16| {
            REFERENCE_IC
                .iter()
                .find(|(anchor, _)| *anchor == horizon)
                .map_or(0.05, |(_, ic)| *ic)
        };

        let reproduced = cohort(Box::new(reference));
        let restricted_pass = measure(
            &frames,
            &realized,
            &config(),
            Some((HORIZONS.as_slice(), &reproduced)),
        )
        .unwrap();
        assert_eq!(misaligned(&restricted_pass), 0);
        assert_eq!(restricted_pass.cross_sections, 400);
        assert_eq!(
            restricted(&restricted_pass),
            anchors,
            "{:?}",
            restricted_pass.warnings
        );

        // The same book panel, but the scorer's own population misses the reference too: that
        // is the reconstruction defect, and it must outrank the universe reading.
        let broken = cohort(Box::new(|_| 0.002));
        let broken_pass = measure(
            &frames,
            &realized,
            &config(),
            Some((HORIZONS.as_slice(), &broken)),
        )
        .unwrap();
        // Every anchor the panel carries fires, not just the ones the book's own wider standard
        // error could resolve: the reproduction is measured on the full cross-section, so the
        // misalignment test it licenses is the tighter one.
        let present = REFERENCE_IC
            .iter()
            .filter(|(horizon, _)| HORIZONS.contains(horizon))
            .count();
        assert!(present > anchors);
        assert_eq!(
            misaligned(&broken_pass),
            present,
            "{:?}",
            broken_pass.warnings
        );
        assert_eq!(restricted(&broken_pass), 0);
    }
}
