//! Forecast path to signed target weights. The model emits 192 independent head rows per
//! (origin, ticker) in sigma units; this module turns one such path into a per-name expected
//! residual return and then into a dollar-neutral, capped, vol-targeted cross-sectional book.
//! Nothing here touches cash, shares or fills - the account replay consumes these weights.
use anyhow::{ensure, Result};
use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};

/// Five-minute bars in a trading year: 252 sessions of 78 bars.
const BARS_PER_YEAR: f64 = 19_656.0;
/// The head emits `pred_len = 192` rows, so no path can contribute more eligible horizons.
const MAX_HORIZONS: usize = 192;
const VAR_FLOOR: f64 = 1e-12;
const CAP_EPS: f64 = 1e-12;
const CAP_PASSES: usize = 8;
/// Share of a name's own target weight the cost-aware no-trade band may occupy.
const BAND_TARGET_FRACTION: f64 = 0.5;

/// Raw, UN-GAINED sigma-unit forecast path for one (origin, ticker). Index i = horizon i+1.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HorizonPath {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub horizons: Vec<u16>,
}

#[derive(Clone, Debug)]
pub struct BookQuote {
    pub asset: usize,
    pub sigma: f64,
    pub price: f64,
    pub adv_usd: f64,
    pub path: Option<HorizonPath>,
}

#[derive(Clone, Debug)]
pub struct BookFrame {
    pub timestamp_ms: i64,
    pub quotes: Vec<BookQuote>,
}

/// How the per-bar drift estimates of the eligible horizons are combined into one number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Aggregation {
    /// The single horizon closest to the holding period: the control, and the only mode that
    /// ignores the rest of the path.
    Single,
    Uniform,
    /// Inverse-variance weights. At a FIXED horizon the predicted mean and std both carry the
    /// same `sqrt(h) * sigma_ticker` scaling, so any mean/std ranking is Sharpe-neutral and the
    /// measured Sharpe of `mean` and `mean/std` decile books is identical. ACROSS horizons the
    /// tanh-bounded state-dependent part of `log_scale` differs per row, so precision weighting
    /// is the one axis on which the predictive sigma carries information that survives ranking:
    /// it says WHERE on the forecast path this name is confidently predicted.
    Precision,
    /// Geometric decay in the horizon, half weight every `--agg-half-life` bars.
    Exponential,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Selection {
    Decile,
    Quintile,
    TopN,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Weighting {
    Equal,
    Mean,
    MeanOverVar,
    MeanOverVol,
}

#[derive(Args, Clone, Debug, Serialize, Deserialize)]
pub struct BookConfig {
    /// Holding period in five-minute bars. Measured cross-sectional IC still rises past 64, so
    /// this is the tradeoff knob between edge and turnover, not a model property.
    #[arg(long, default_value_t = 64)]
    pub trade_horizon: usize,
    #[arg(long, value_enum, default_value_t = Aggregation::Precision)]
    pub aggregation: Aggregation,
    /// Horizons below this are dropped: the one-bar IC is 0.018 against a 0.0027 standard error,
    /// so the short end is mostly noise on a per-bar axis where its variance is largest.
    #[arg(long, default_value_t = 8)]
    pub agg_min_horizon: usize,
    #[arg(long, default_value_t = 128)]
    pub agg_max_horizon: usize,
    #[arg(long, default_value_t = 32.0)]
    pub agg_half_life: f64,
    #[arg(long, value_enum, default_value_t = Selection::Decile)]
    pub selection: Selection,
    #[arg(long, default_value_t = 25)]
    pub selection_n: usize,
    #[arg(long, value_enum, default_value_t = Weighting::MeanOverVar)]
    pub weighting: Weighting,
    /// Ex-ante annualized vol the whole book is scaled to; `0` disables the scaling.
    #[arg(long, default_value_t = 0.15)]
    pub target_vol_annual: f64,
    #[arg(long, default_value_t = 2.0)]
    pub gross_cap: f64,
    #[arg(long, default_value_t = 0.05)]
    pub per_name_cap: f64,
    /// Multiple of the round-trip cost estimate a weight change must clear before it is traded.
    #[arg(long, default_value_t = 2.0)]
    pub no_trade_band_cost_multiple: f64,
}

impl Default for BookConfig {
    fn default() -> Self {
        Self {
            trade_horizon: 64,
            aggregation: Aggregation::Precision,
            agg_min_horizon: 8,
            agg_max_horizon: 128,
            agg_half_life: 32.0,
            selection: Selection::Decile,
            selection_n: 25,
            weighting: Weighting::MeanOverVar,
            target_vol_annual: 0.15,
            gross_cap: 2.0,
            per_name_cap: 0.05,
            no_trade_band_cost_multiple: 2.0,
        }
    }
}

impl BookConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.trade_horizon > 0, "trade horizon must be positive");
        ensure!(
            self.agg_min_horizon > 0,
            "aggregation minimum horizon must be positive"
        );
        ensure!(
            self.agg_min_horizon <= self.agg_max_horizon,
            "aggregation horizon window is empty"
        );
        ensure!(
            self.agg_half_life.is_finite() && self.agg_half_life > 0.0,
            "aggregation half life must be positive"
        );
        ensure!(
            self.gross_cap.is_finite() && self.gross_cap > 0.0,
            "gross cap must be positive"
        );
        ensure!(
            self.per_name_cap.is_finite() && self.per_name_cap > 0.0,
            "per-name cap must be positive"
        );
        ensure!(
            self.per_name_cap <= self.gross_cap,
            "per-name cap exceeds the gross cap"
        );
        ensure!(
            self.target_vol_annual.is_finite() && self.target_vol_annual >= 0.0,
            "target vol must be finite and nonnegative"
        );
        ensure!(
            self.no_trade_band_cost_multiple.is_finite() && self.no_trade_band_cost_multiple >= 0.0,
            "no-trade band multiple must be finite and nonnegative"
        );
        ensure!(
            self.selection != Selection::TopN || self.selection_n > 0,
            "top-n selection needs a positive name count"
        );
        Ok(())
    }
}

/// Per-name expected residual log return and variance at `trade_horizon`, in RAW (not sigma) units.
#[derive(Clone, Copy, Debug, Default)]
pub struct NameEdge {
    pub mu: f64,
    /// Variance of the AGGREGATED MEAN ESTIMATE, not of the return. It is the inverse-variance
    /// ranking statistic and it shrinks with the number of horizons combined, so it sits orders
    /// of magnitude below any return variance and must never be used to price risk.
    pub var: f64,
    /// Predicted RETURN variance over `trade_horizon` in raw units, read off the head nearest
    /// that horizon and scaled diffusively. This is the dispersion the calibration check
    /// measured, so it is the only variance the risk penalty, the no-trade band and the vol
    /// target may use.
    pub risk_var: f64,
    pub t_stat: f64,
    pub agreement: f64,
}

/// Signed target weights aligned index-for-index with `BookFrame::quotes`.
#[derive(Clone, Debug, Default)]
pub struct BookTarget {
    pub weights: Vec<f64>,
    pub gross: f64,
    pub net: f64,
    pub active: usize,
    pub ex_ante_vol_annual: f64,
}

/// Collapse a forecast path into one expected residual return over the holding period.
///
/// The path stores the CUMULATIVE sigma-unit residual log return at each horizon, so horizons are
/// not commensurable until they are divided by their own horizon: `rate_h = mean_h / h` is the
/// per-bar drift the model implies at that point on the path, and `(std_h / h)^2` is its variance.
/// Averaging the cumulative numbers instead is the standard way to get multi-horizon aggregation
/// wrong, because the long horizons then dominate purely through their `h` scaling.
///
/// The aggregated variance is `sum(w_h^2 * var_rate_h)`, which assumes the horizons are
/// independent. They are emphatically not - they are 192 heads reading one state and predicting
/// overlapping windows - so this UNDERSTATES the variance. It is used as a ranking statistic and
/// as the input to relative vol targeting, never as a confidence interval.
pub fn name_edge(path: &HorizonPath, sigma: f64, cfg: &BookConfig) -> Option<NameEdge> {
    if !(sigma.is_finite() && sigma > 0.0) {
        return None;
    }
    let mut rates = [0.0f64; MAX_HORIZONS];
    let mut var_rates = [0.0f64; MAX_HORIZONS];
    let mut weights = [0.0f64; MAX_HORIZONS];
    let mut bars = [0.0f64; MAX_HORIZONS];
    let mut count = 0usize;
    let rows = path.mean.len().min(path.std.len()).min(path.horizons.len());
    for row in 0..rows {
        let horizon = usize::from(path.horizons[row]);
        if horizon == 0 || horizon < cfg.agg_min_horizon || horizon > cfg.agg_max_horizon {
            continue;
        }
        let mean = f64::from(path.mean[row]);
        let std = f64::from(path.std[row]);
        if !mean.is_finite() || !std.is_finite() || std <= 0.0 {
            continue;
        }
        let horizon = horizon as f64;
        rates[count] = mean / horizon;
        var_rates[count] = (std / horizon) * (std / horizon);
        bars[count] = horizon;
        count += 1;
        if count == MAX_HORIZONS {
            break;
        }
    }
    if count == 0 {
        return None;
    }
    let horizon = cfg.trade_horizon as f64;
    let mut nearest = 0usize;
    for row in 1..count {
        if (bars[row] - horizon).abs() < (bars[nearest] - horizon).abs() {
            nearest = row;
        }
    }
    match cfg.aggregation {
        Aggregation::Single => weights[nearest] = 1.0,
        Aggregation::Uniform => weights[..count].fill(1.0),
        Aggregation::Precision => {
            for row in 0..count {
                weights[row] = 1.0 / var_rates[row].max(VAR_FLOOR);
            }
        }
        Aggregation::Exponential => {
            for row in 0..count {
                weights[row] = (-std::f64::consts::LN_2 * bars[row] / cfg.agg_half_life).exp();
            }
        }
    }
    let total: f64 = weights[..count].iter().sum();
    if total.is_finite() && total > 0.0 {
        for weight in &mut weights[..count] {
            *weight /= total;
        }
    } else {
        weights[..count].fill(1.0 / count as f64);
    }
    let rate: f64 = (0..count).map(|row| weights[row] * rates[row]).sum();
    let var_rate: f64 = (0..count)
        .map(|row| weights[row] * weights[row] * var_rates[row])
        .sum::<f64>()
        .max(VAR_FLOOR);
    let mu = rate * horizon * sigma;
    let var = (var_rate * horizon * horizon * sigma * sigma).max(VAR_FLOOR);
    // The return variance the risk terms need, which is NOT `var`: `var` is the dispersion of a
    // precision-weighted average of 121 overlapping heads and collapses with their count. The
    // head nearest the trade horizon carries the dispersion the calibration check measured, and
    // it is carried to the holding period diffusively rather than by rescaling `var_rate`, whose
    // per-bar form would scale variance with the square of the horizon.
    let risk_var = (var_rates[nearest] * bars[nearest] * horizon * sigma * sigma).max(VAR_FLOOR);
    if !mu.is_finite() || !var.is_finite() || !risk_var.is_finite() {
        return None;
    }
    let agreement = if count == 1 || rate == 0.0 {
        0.5
    } else {
        let positive = rate > 0.0;
        (0..count)
            .filter(|&row| match rates[row].partial_cmp(&0.0) {
                Some(std::cmp::Ordering::Greater) => positive,
                Some(std::cmp::Ordering::Less) => !positive,
                _ => false,
            })
            .map(|row| weights[row])
            .sum::<f64>()
            .clamp(0.0, 1.0)
    };
    Some(NameEdge {
        mu,
        var,
        risk_var,
        t_stat: rate / var_rate.sqrt(),
        agreement,
    })
}

struct Name {
    slot: usize,
    asset: usize,
    edge: NameEdge,
    stat: f64,
}

/// Build the cross-sectional book for one origin.
///
/// The training target is beta-hedged against the equal-weighted corpus mean but is NOT
/// cross-sectionally demeaned, so the raw signal still carries a common component that a
/// dollar-neutral book cannot express. The ranking statistic is therefore demeaned across the
/// cross-section before anything is selected or sized; without that step a cross-section with a
/// large common drift selects a "short" leg whose expected return is still positive.
pub fn target_weights(frame: &BookFrame, cfg: &BookConfig) -> BookTarget {
    let mut target = BookTarget {
        weights: vec![0.0; frame.quotes.len()],
        ..BookTarget::default()
    };
    let mut names: Vec<Name> = Vec::with_capacity(frame.quotes.len());
    for (slot, quote) in frame.quotes.iter().enumerate() {
        let Some(path) = quote.path.as_ref() else {
            continue;
        };
        let Some(edge) = name_edge(path, quote.sigma, cfg) else {
            continue;
        };
        let stat = match cfg.weighting {
            Weighting::Equal | Weighting::Mean => edge.mu,
            Weighting::MeanOverVar => edge.mu / edge.var,
            Weighting::MeanOverVol => edge.mu / edge.var.sqrt(),
        };
        if !stat.is_finite() {
            continue;
        }
        names.push(Name {
            slot,
            asset: quote.asset,
            edge,
            stat,
        });
    }
    if names.len() < 4 {
        return target;
    }
    let mean = names.iter().map(|name| name.stat).sum::<f64>() / names.len() as f64;
    for name in &mut names {
        name.stat -= mean;
    }
    names.sort_by(|a, b| b.stat.total_cmp(&a.stat).then(a.asset.cmp(&b.asset)));

    let count = names.len();
    let per_side = match cfg.selection {
        Selection::Decile => (count as f64 * 0.1).floor() as usize,
        Selection::Quintile => (count as f64 * 0.2).floor() as usize,
        Selection::TopN => cfg.selection_n.min(count / 2),
        Selection::All => count,
    };
    if cfg.selection == Selection::All {
        for name in &names {
            target.weights[name.slot] = raw_weight(name, cfg);
        }
    } else {
        if per_side < 2 || per_side * 2 > count {
            return target;
        }
        for name in names[..per_side]
            .iter()
            .chain(names[count - per_side..].iter())
        {
            target.weights[name.slot] = raw_weight(name, cfg);
        }
    }

    let mut longs = 0usize;
    let mut shorts = 0usize;
    for weight in &target.weights {
        if *weight > 0.0 {
            longs += 1;
        } else if *weight < 0.0 {
            shorts += 1;
        }
    }
    if longs < 2 || shorts < 2 {
        target.weights.fill(0.0);
        return target;
    }

    let side_target = cfg.gross_cap * 0.5;
    let long_gross = fill_side(&mut target.weights, true, side_target, cfg.per_name_cap);
    let short_gross = fill_side(&mut target.weights, false, side_target, cfg.per_name_cap);
    let balanced = long_gross.min(short_gross);
    if balanced <= 0.0 || !balanced.is_finite() {
        target.weights.fill(0.0);
        return target;
    }
    let long_scale = balanced / long_gross;
    let short_scale = balanced / short_gross;
    for weight in &mut target.weights {
        if *weight > 0.0 {
            *weight *= long_scale;
        } else if *weight < 0.0 {
            *weight *= short_scale;
        }
    }

    // Ex-ante vol under a DIAGONAL covariance of the PREDICTED RETURN variance, not of the
    // aggregation's estimator variance: the latter is a confidence statistic that collapses with
    // the number of heads combined and would read a 40% book as a 0.2% one. The residual targets
    // are already beta-hedged against the corpus mean, but sector and factor co-movement
    // survives in the residuals, so a diagonal book variance still UNDERSTATES realized vol. It
    // is the right scaling signal and the wrong risk number.
    let holding = cfg.trade_horizon as f64;
    let variance: f64 = names
        .iter()
        .map(|name| {
            let weight = target.weights[name.slot];
            weight * weight * name.edge.risk_var
        })
        .sum();
    target.ex_ante_vol_annual = (variance * BARS_PER_YEAR / holding).sqrt();
    if cfg.target_vol_annual > 0.0
        && target.ex_ante_vol_annual.is_finite()
        && target.ex_ante_vol_annual > 0.0
    {
        let mut gross = 0.0;
        let mut peak = 0.0f64;
        for weight in &target.weights {
            gross += weight.abs();
            peak = peak.max(weight.abs());
        }
        let mut scale = cfg.target_vol_annual / target.ex_ante_vol_annual;
        if gross > 0.0 {
            scale = scale.min(cfg.gross_cap / gross);
        }
        if peak > 0.0 {
            scale = scale.min(cfg.per_name_cap / peak);
        }
        if scale.is_finite() && scale > 0.0 {
            for weight in &mut target.weights {
                *weight *= scale;
            }
        }
    }

    for weight in &target.weights {
        target.gross += weight.abs();
        target.net += *weight;
        if *weight != 0.0 {
            target.active += 1;
        }
    }
    target
}

/// Weight the book asks for after the cost-aware no-trade band, given the weight it holds.
///
/// The band comes from the QUADRATIC risk term, not the linear alpha: a linear utility scales
/// gain and cost identically in the trade size, so it has no interior no-trade region at all.
/// Moving a name's weight by `d` recovers `(gamma/2)*var*d^2` of utility and pays `2*one_way*d`
/// for the round trip, so the smallest worthwhile move is `4*multiple*one_way/(gamma*var)`.
///
/// The second argument of the `min` is the whole defect this function exists to name. The band
/// was clamped to `per_name_cap`, and no weight may exceed `per_name_cap`, so `|d| > band` was
/// unsatisfiable and every name froze at the weight it held - which starts at zero, so the book
/// could never be built at all. A no-trade region must be a fraction of the position it governs,
/// never of the global cap. At `BAND_TARGET_FRACTION` of the name's own target, entry from flat
/// (`|d| = |target|`), a full exit (`target = 0`, so the band is zero) and any sign flip
/// (`|d| >= |target|`) clear the band by construction, while a rebalance smaller than half the
/// name's own size is refused. The cost-derived value binds whenever it is the smaller of the two
/// - which, with a risk aversion whose unconstrained optimum sits far above `per_name_cap`, is
/// the illiquid tail rather than the typical name.
pub fn banded_weight(
    target: f64,
    previous: f64,
    edge: &NameEdge,
    one_way_cost: f64,
    risk_aversion: f64,
    cfg: &BookConfig,
) -> f64 {
    if !(risk_aversion > 0.0 && edge.risk_var > 0.0) {
        return target;
    }
    let band = (4.0 * cfg.no_trade_band_cost_multiple * one_way_cost
        / (risk_aversion * edge.risk_var))
        .min(BAND_TARGET_FRACTION * target.abs());
    if (target - previous).abs() < band {
        previous
    } else {
        target
    }
}

fn raw_weight(name: &Name, cfg: &BookConfig) -> f64 {
    match cfg.weighting {
        Weighting::Equal => {
            if name.stat > 0.0 {
                1.0
            } else if name.stat < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        _ => name.stat,
    }
}

fn in_side(weight: f64, positive: bool) -> bool {
    if positive {
        weight > 0.0
    } else {
        weight < 0.0
    }
}

/// Water-fill one side of the book to `side_target` gross without breaching `cap` on any name.
/// Scaling the side and then clamping loses notional, so the shortfall is redistributed over the
/// names that are not yet at the cap and the pass repeats; a side whose capacity is genuinely
/// below the target is left short of it rather than silently violating the per-name limit.
fn fill_side(weights: &mut [f64], positive: bool, side_target: f64, cap: f64) -> f64 {
    let mut total = 0.0;
    for weight in weights.iter() {
        if in_side(*weight, positive) {
            total += weight.abs();
        }
    }
    if !(total.is_finite() && total > 0.0) {
        for weight in weights.iter_mut() {
            if in_side(*weight, positive) {
                *weight = 0.0;
            }
        }
        return 0.0;
    }
    let scale = side_target / total;
    for weight in weights.iter_mut() {
        if in_side(*weight, positive) {
            *weight *= scale;
        }
    }
    let clamped = if positive { cap } else { -cap };
    for _ in 0..CAP_PASSES {
        let mut free = 0.0;
        let mut held = 0.0;
        for weight in weights.iter_mut() {
            if !in_side(*weight, positive) {
                continue;
            }
            if weight.abs() > cap {
                *weight = clamped;
            }
            if weight.abs() + CAP_EPS < cap {
                free += weight.abs();
            } else {
                held += weight.abs();
            }
        }
        let deficit = side_target - held - free;
        if deficit <= CAP_EPS || free <= 0.0 {
            break;
        }
        let lift = (free + deficit) / free;
        for weight in weights.iter_mut() {
            if in_side(*weight, positive) && weight.abs() + CAP_EPS < cap {
                *weight *= lift;
            }
        }
    }
    let mut total = 0.0;
    for weight in weights.iter_mut() {
        if !in_side(*weight, positive) {
            continue;
        }
        if weight.abs() > cap {
            *weight = clamped;
        }
        total += weight.abs();
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(horizons: &[u16], mean: &[f32], std: &[f32]) -> HorizonPath {
        HorizonPath {
            mean: mean.to_vec(),
            std: std.to_vec(),
            horizons: horizons.to_vec(),
        }
    }

    fn quote(asset: usize, path: HorizonPath) -> BookQuote {
        BookQuote {
            asset,
            sigma: 0.01,
            price: 100.0,
            adv_usd: 10_000_000.0,
            path: Some(path),
        }
    }

    /// Implied weight of the first of two horizons, recovered from the aggregate per-bar rate.
    fn implied_first_weight(
        edge: &NameEdge,
        cfg: &BookConfig,
        sigma: f64,
        rates: (f64, f64),
    ) -> f64 {
        let rate = edge.mu / (cfg.trade_horizon as f64 * sigma);
        (rate - rates.1) / (rates.0 - rates.1)
    }

    #[test]
    fn precision_aggregation_favors_the_tighter_horizon_over_uniform() {
        // Binary-exact in f32: h=16 implies 0.125/bar at a 0.015625/bar rate std, h=64 implies
        // 0.25/bar at 0.125/bar, so the per-bar precision ratio is exactly 64.
        let p = path(&[16, 64], &[2.0, 16.0], &[0.25, 8.0]);
        let rates = (0.125, 0.25);
        let uniform = BookConfig {
            aggregation: Aggregation::Uniform,
            ..BookConfig::default()
        };
        let precision = BookConfig::default();
        let u = name_edge(&p, 0.01, &uniform).unwrap();
        let q = name_edge(&p, 0.01, &precision).unwrap();
        let uw = implied_first_weight(&u, &uniform, 0.01, rates);
        let pw = implied_first_weight(&q, &precision, 0.01, rates);
        assert!((uw - 0.5).abs() < 1e-9, "uniform weight {uw}");
        assert!(pw > uw, "precision {pw} did not exceed uniform {uw}");
        assert!((pw - 64.0 / 65.0).abs() < 1e-9, "precision weight {pw}");
        assert!(q.var < u.var, "precision variance {} vs {}", q.var, u.var);
    }

    #[test]
    fn agreement_separates_a_consistent_path_from_a_split_one() {
        let consistent = path(
            &[8, 16, 32, 64],
            &[0.8, 1.6, 3.2, 6.4],
            &[0.5, 0.7, 1.0, 1.4],
        );
        let edge = name_edge(&consistent, 0.01, &BookConfig::default()).unwrap();
        assert!((edge.agreement - 1.0).abs() < 1e-12);

        let split = path(&[16, 64], &[1.6, -3.2], &[1.0, 1.0]);
        let cfg = BookConfig {
            aggregation: Aggregation::Uniform,
            ..BookConfig::default()
        };
        let edge = name_edge(&split, 0.01, &cfg).unwrap();
        assert!((edge.agreement - 0.5).abs() < 1e-12, "{}", edge.agreement);

        let single = path(&[64], &[6.4], &[1.0]);
        let edge = name_edge(&single, 0.01, &BookConfig::default()).unwrap();
        assert!((edge.agreement - 0.5).abs() < 1e-12);
    }

    #[test]
    fn unusable_horizons_are_skipped_and_an_empty_window_has_no_edge() {
        let cfg = BookConfig::default();
        let broken = path(
            &[16, 32, 64],
            &[1.6, f32::NAN, 6.4],
            &[1.0, 1.0, f32::INFINITY],
        );
        let edge = name_edge(&broken, 0.01, &cfg).unwrap();
        let only = path(&[16], &[1.6], &[1.0]);
        let reference = name_edge(&only, 0.01, &cfg).unwrap();
        assert!((edge.mu - reference.mu).abs() < 1e-15);

        assert!(name_edge(&path(&[4, 192], &[1.0, 1.0], &[1.0, 1.0]), 0.01, &cfg).is_none());
        assert!(name_edge(&only, 0.0, &cfg).is_none());
        assert!(name_edge(&only, f64::NAN, &cfg).is_none());
        assert!(name_edge(&path(&[64], &[6.4], &[0.0]), 0.01, &cfg).is_none());
    }

    #[test]
    fn mu_scales_with_sigma_and_var_with_its_square() {
        let cfg = BookConfig::default();
        let p = path(&[16, 64], &[1.6, 6.4], &[0.8, 1.6]);
        let a = name_edge(&p, 0.01, &cfg).unwrap();
        let b = name_edge(&p, 0.02, &cfg).unwrap();
        assert!((b.mu - 2.0 * a.mu).abs() < 1e-12 * a.mu.abs().max(1.0));
        assert!((b.var - 4.0 * a.var).abs() < 1e-12 * a.var.abs().max(1.0));
        assert!((b.t_stat - a.t_stat).abs() < 1e-12);
    }

    /// A symmetric cross-section whose signal magnitude grows geometrically with rank, so the
    /// extreme names in the selected decile demand more than the per-name cap and the water-fill
    /// is actually exercised.
    fn spread_frame(count: usize, std: f32) -> BookFrame {
        let quotes = (0..count)
            .map(|i| {
                let centered = i as f64 - (count as f64 - 1.0) / 2.0;
                let mean = centered.signum() * 1.05_f64.powf(centered.abs());
                quote(i, path(&[64], &[mean as f32], &[std]))
            })
            .collect();
        BookFrame {
            timestamp_ms: 0,
            quotes,
        }
    }

    #[test]
    fn book_is_dollar_neutral_and_respects_both_caps() {
        let cfg = BookConfig {
            target_vol_annual: 0.0,
            ..BookConfig::default()
        };
        let target = target_weights(&spread_frame(400, 1.0), &cfg);
        assert_eq!(target.active, 80);
        let long: f64 = target.weights.iter().filter(|w| **w > 0.0).sum();
        let short: f64 = target
            .weights
            .iter()
            .filter(|w| **w < 0.0)
            .map(|w| -w)
            .sum();
        assert!((long - short).abs() < 1e-9, "long {long} short {short}");
        assert!(target.net.abs() < 1e-9, "net {}", target.net);
        assert!(
            (target.gross - cfg.gross_cap).abs() < 1e-9,
            "gross {}",
            target.gross
        );
        let peak = target
            .weights
            .iter()
            .fold(0.0f64, |acc, w| acc.max(w.abs()));
        assert!(peak <= cfg.per_name_cap + 1e-12, "peak {peak}");
        assert!(peak > cfg.per_name_cap - 1e-9, "cap was expected to bind");
    }

    #[test]
    fn a_cross_section_too_thin_for_two_names_per_side_trades_nothing() {
        let cfg = BookConfig {
            target_vol_annual: 0.0,
            ..BookConfig::default()
        };
        let target = target_weights(&spread_frame(10, 1.0), &cfg);
        assert_eq!(target.active, 0);
        assert_eq!(target.gross, 0.0);
        assert!(target.weights.iter().all(|w| *w == 0.0));
    }

    #[test]
    fn a_constant_cross_section_is_entirely_common_and_produces_no_book() {
        let quotes = (0..400)
            .map(|i| quote(i, path(&[64], &[6.4], &[1.0])))
            .collect();
        let frame = BookFrame {
            timestamp_ms: 0,
            quotes,
        };
        let target = target_weights(&frame, &BookConfig::default());
        assert_eq!(target.active, 0);
        assert_eq!(target.gross, 0.0);
    }

    #[test]
    fn vol_targeting_lands_the_book_on_the_requested_annual_vol() {
        let cfg = BookConfig::default();
        let frame = spread_frame(400, 10.0);
        let target = target_weights(&frame, &cfg);
        assert!(
            target.ex_ante_vol_annual > cfg.target_vol_annual,
            "pre-scaling vol {} must exceed the target for this to test scaling",
            target.ex_ante_vol_annual
        );
        let variance: f64 = frame
            .quotes
            .iter()
            .enumerate()
            .map(|(slot, q)| {
                let weight = target.weights[slot];
                let var = name_edge(q.path.as_ref().unwrap(), q.sigma, &cfg)
                    .unwrap()
                    .risk_var;
                weight * weight * var
            })
            .sum();
        let realized = (variance * BARS_PER_YEAR / cfg.trade_horizon as f64).sqrt();
        assert!(
            (realized - cfg.target_vol_annual).abs() < 1e-9,
            "scaled vol {realized}"
        );
        assert!(target.gross < cfg.gross_cap);
        assert!(target.net.abs() < 1e-9);
    }

    /// A cross-section whose monotone signal rotates by one rank per frame, so decile membership
    /// turns over gradually instead of being redrawn from scratch.
    fn rotating_frame(count: usize, shift: usize) -> BookFrame {
        let quotes = (0..count)
            .map(|i| {
                let rank = (i + shift) % count;
                let centered = rank as f64 - (count as f64 - 1.0) / 2.0;
                let mean = centered / (count as f64 * 0.5);
                quote(i, path(&[64], &[mean as f32], &[1.0]))
            })
            .collect();
        BookFrame {
            timestamp_ms: shift as i64,
            quotes,
        }
    }

    /// The defect the band was carrying: it was clamped to `per_name_cap`, and no weight may
    /// exceed `per_name_cap`, so `|w_new - w_prev| > band` was unsatisfiable and every name held
    /// the weight it started with - zero. Only the handful of names water-filled to exactly the
    /// cap could ever trade, which is a three-name book out of a 256-name decile. Checking the
    /// band arithmetic on its own would not have caught it: the failure is entirely in the
    /// interaction with the cap and with an all-zero starting book.
    #[test]
    fn the_band_lets_a_flat_book_enter_and_holds_breadth_through_a_rotating_signal() {
        let cfg = BookConfig::default();
        let (count, frames) = (256usize, 100usize);
        let expected = 2 * (count / 10);
        let mut previous = vec![0.0f64; count];
        let mut breadth = Vec::with_capacity(frames);
        for shift in 0..frames {
            let frame = rotating_frame(count, shift);
            let target = target_weights(&frame, &cfg);
            assert_eq!(target.active, expected, "selection at frame {shift}");
            let mut current = vec![0.0f64; count];
            for (slot, held) in current.iter_mut().enumerate() {
                let source = &frame.quotes[slot];
                let edge = name_edge(source.path.as_ref().unwrap(), source.sigma, &cfg).unwrap();
                *held = banded_weight(
                    target.weights[slot],
                    previous[slot],
                    &edge,
                    2.4e-4,
                    10.0,
                    &cfg,
                );
            }
            breadth.push(current.iter().filter(|weight| **weight != 0.0).count());
            previous = current;
        }
        assert_eq!(
            breadth[0], expected,
            "a flat book must be able to enter its whole decile"
        );
        let floor = breadth.iter().copied().min().expect("one frame per shift");
        assert!(
            floor >= expected,
            "breadth collapsed to {floor} names against {expected} over {frames} frames"
        );
    }

    #[test]
    fn the_band_refuses_a_small_rebalance_but_never_an_entry_exit_or_sign_flip() {
        let cfg = BookConfig::default();
        let edge = NameEdge {
            mu: 0.0,
            var: 1e-6,
            risk_var: 1e-4,
            t_stat: 0.0,
            agreement: 0.5,
        };
        let band = |target, previous| banded_weight(target, previous, &edge, 2.4e-4, 10.0, &cfg);
        assert_eq!(band(0.04, 0.0), 0.04, "entry from flat");
        assert_eq!(band(0.0, 0.04), 0.0, "full exit");
        assert_eq!(band(-0.04, 0.04), -0.04, "sign flip");
        assert_eq!(band(0.042, 0.04), 0.04, "rebalance inside the band");
    }

    #[test]
    fn validate_rejects_incoherent_configurations() {
        assert!(BookConfig::default().validate().is_ok());
        for cfg in [
            BookConfig {
                trade_horizon: 0,
                ..BookConfig::default()
            },
            BookConfig {
                agg_min_horizon: 0,
                ..BookConfig::default()
            },
            BookConfig {
                agg_min_horizon: 129,
                ..BookConfig::default()
            },
            BookConfig {
                gross_cap: 0.0,
                ..BookConfig::default()
            },
            BookConfig {
                per_name_cap: 0.0,
                ..BookConfig::default()
            },
            BookConfig {
                per_name_cap: 3.0,
                ..BookConfig::default()
            },
            BookConfig {
                target_vol_annual: -0.1,
                ..BookConfig::default()
            },
            BookConfig {
                agg_half_life: 0.0,
                ..BookConfig::default()
            },
            BookConfig {
                selection: Selection::TopN,
                selection_n: 0,
                ..BookConfig::default()
            },
        ] {
            assert!(cfg.validate().is_err(), "{cfg:?} should be rejected");
        }
    }
}
