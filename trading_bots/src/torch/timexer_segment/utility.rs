//! Fixed forecast-to-position probes on independent origin cohorts, not a portfolio simulator.
//! Decisions use only origin forecasts. Payoffs enter at the next observed bar's open and
//! exit at the forecast horizon's close. Missing outcomes reject a cohort, never resize it.

use anyhow::{ensure, Result};
use tch::{Device, Kind, Tensor};

use super::reports::{PolicyCurve, PortfolioCurve};

/// The endpoint-cohort holding periods, in next observed bars. Shared with
/// [`super::reports::DECISION_HORIZONS`] by test, not by construction: the two families are
/// only comparable - "does the IC at the horizon this policy trades survive training" - if
/// they are indexed on the same grid.
pub(super) const HORIZONS: &[u64] = &[1, 8, 16, 32, 64, 128, 192];
pub(super) const COSTS_BPS: &[f64] = &[0., 0.5, 1., 2., 5., 10.];
const LABELS: &[&str] = &[
    "cash",
    "equal long",
    "signed equal notional",
    "mean decile long/short",
    "mean/std decile long/short",
    "one-sigma gated sign",
    "diagonal quadratic log-return proxy",
];

struct Cohorts<'a> {
    groups: &'a Tensor,
    count: i64,
    sizes: Tensor,
    row_sizes: Tensor,
}

impl<'a> Cohorts<'a> {
    fn new(groups: &'a Tensor, count: i64) -> Self {
        let sizes = Tensor::zeros([count], (Kind::Double, groups.device())).index_add(
            0,
            groups,
            &Tensor::ones(groups.size(), (Kind::Double, groups.device())),
        );
        let row_sizes = sizes.index_select(0, groups);
        Self {
            groups,
            count,
            sizes,
            row_sizes,
        }
    }

    fn sum(&self, values: &Tensor) -> Tensor {
        Tensor::zeros([self.count], (Kind::Double, self.groups.device())).index_add(
            0,
            self.groups,
            values,
        )
    }

    /// Fractional membership at tied decile boundaries avoids ticker-order alpha. If every
    /// signal is equal, the identical long and short selections cancel into cash.
    fn deciles(&self, signal: &Tensor) -> Tensor {
        let windows = signal.size()[0];
        let rank = signal.argsort(0, false).argsort(0, false);
        let order = (self.groups * windows + rank).argsort(0, false);
        let sorted = signal.index_select(0, &order);
        let starts = (&self.sizes.cumsum(0, Kind::Double) - &self.sizes).to_kind(Kind::Int64);
        let side = (&self.sizes / 10.).floor().clamp_min(1.);
        let bottom_index = &starts + side.to_kind(Kind::Int64) - 1;
        let top_index = &starts + (&self.sizes - &side).to_kind(Kind::Int64);
        let bottom = sorted
            .index_select(0, &bottom_index)
            .index_select(0, self.groups);
        let top = sorted
            .index_select(0, &top_index)
            .index_select(0, self.groups);
        let membership = |boundary: &Tensor, upper: bool| {
            let beyond = if upper {
                signal.gt_tensor(boundary)
            } else {
                signal.lt_tensor(boundary)
            }
            .to_kind(Kind::Double);
            let tied = signal.eq_tensor(boundary).to_kind(Kind::Double);
            let remaining = &side - self.sum(&beyond);
            let share = remaining / self.sum(&tied).clamp_min(1.);
            beyond + tied * share.index_select(0, self.groups)
        };
        (membership(&top, true) - membership(&bottom, false)) * 0.5
            / side.index_select(0, self.groups)
    }

    fn weights(&self, mean: &Tensor, std: &Tensor) -> [Tensor; 7] {
        let signed = mean.sign() / &self.row_sizes;
        // This is an independent-risk log-return approximation, NOT lognormal Kelly or a
        // calibrated portfolio risk model. Projection imposes the same unit gross budget.
        let quadratic = mean / (std.square() + mean.square());
        let budget = self.sum(&quadratic.abs()).clamp_min(1.);
        [
            mean.zeros_like(),
            self.row_sizes.reciprocal(),
            signed.shallow_clone(),
            self.deciles(mean),
            self.deciles(&(mean / std)),
            signed * mean.abs().ge_tensor(std).to_kind(Kind::Double),
            quadratic / budget.index_select(0, self.groups),
        ]
    }
}

/// Forecast and outcome coordinates are cumulative log returns in origin-sigma units;
/// log_scale is the close predictive log standard deviation in those same units. The scale
/// describes residual outcomes, not epistemic confidence or unhedged market risk.
pub(super) fn evaluate(
    forecast: &Tensor,
    log_scale: &Tensor,
    target: &Tensor,
    valid: &Tensor,
    sigma: &Tensor,
    entry_log: &Tensor,
    groups: &Tensor,
    group_count: i64,
) -> Result<PortfolioCurve> {
    let shape = forecast.size();
    ensure!(
        shape.len() == 2 && shape[0] > 0 && shape[1] > 0,
        "utility needs nonempty forecast windows"
    );
    ensure!(
        log_scale.size() == shape
            && target.size() == shape
            && valid.size() == shape
            && sigma.size() == [shape[0]]
            && entry_log.size() == [shape[0]]
            && groups.size() == [shape[0]]
            && group_count > 0,
        "utility forecast, outcomes and origin groups must align"
    );
    ensure!(
        groups.kind() == Kind::Int64,
        "utility origin groups must be int64"
    );
    ensure!(
        groups
            .ge(0)
            .logical_and(&groups.lt(group_count))
            .all()
            .int64_value(&[])
            != 0,
        "utility group index outside cohort range"
    );
    let cohorts = Cohorts::new(groups, group_count);
    ensure!(
        cohorts.sizes.min().double_value(&[]) > 0.,
        "utility groups must be densely populated"
    );
    let finite = forecast
        .isfinite()
        .all()
        .logical_and(&log_scale.isfinite().all())
        .logical_and(&sigma.isfinite().logical_and(&sigma.gt(0.)).all());
    ensure!(
        finite.int64_value(&[]) != 0,
        "nonfinite utility forecast or invalid causal sigma"
    );
    let pred_len = shape[1] as u64;
    let mut horizons: Vec<u64> = HORIZONS
        .iter()
        .copied()
        .filter(|h| *h <= pred_len)
        .collect();
    if horizons.last() != Some(&pred_len) {
        horizons.push(pred_len);
    }
    let sigma = sigma.to_kind(Kind::Double);
    let entry_log = entry_log.to_kind(Kind::Double);
    let entry_valid = valid.select(1, 0).gt(0.).logical_and(&entry_log.isfinite());
    let mut moments = Vec::with_capacity(horizons.len());
    let mut census = Vec::with_capacity(horizons.len());
    let mut validity_checks = Vec::with_capacity(horizons.len());
    for &horizon in &horizons {
        let j = horizon as i64 - 1;
        let mean = forecast.select(1, j).to_kind(Kind::Double) * &sigma;
        let std = log_scale.select(1, j).to_kind(Kind::Double).exp() * &sigma;
        let valid_scale = std.isfinite().logical_and(&std.gt(0.)).all();
        let observed = target.select(1, j).to_kind(Kind::Double) * &sigma - &entry_log;
        let valid_row = valid
            .select(1, j)
            .gt(0.)
            .logical_and(&entry_valid)
            .logical_and(&observed.isfinite());
        // Replacing missing observations here protects reductions from 0 * NaN. It does
        // not admit them: a cohort must have an outcome for EVERY originally sized name.
        let observed = observed.where_self(&valid_row, &observed.zeros_like());
        let gross_price = observed.exp();
        validity_checks.push(valid_scale.logical_and(&gross_price.isfinite().all()));
        let realized = observed.expm1();
        let available = cohorts.sum(&valid_row.to_kind(Kind::Double));
        let usable_bool = available
            .eq_tensor(&cohorts.sizes)
            .logical_and(&cohorts.sizes.ge(super::runner::CROSS_SECTION_MIN));
        let usable = usable_bool.to_kind(Kind::Double);
        let periods = usable.sum(Kind::Double);
        let denominator = periods.clamp_min(1.);
        let average = |values: &Tensor| (values * &usable).sum(Kind::Double) / &denominator;
        let mut policy_moments = Vec::with_capacity(LABELS.len());
        for weights in cohorts.weights(&mean, &std) {
            let pnl = cohorts.sum(&(&weights * &realized));
            let gross = average(&pnl);
            let variance = average(&(&pnl - &gross).square());
            let worst = pnl
                .where_self(&usable_bool, &pnl.full_like(f64::INFINITY))
                .min();
            let exposure = cohorts.sum(&weights.abs());
            let net = cohorts.sum(&weights);
            let active = cohorts.sum(&weights.ne(0.).to_kind(Kind::Double)) / &cohorts.sizes;
            // A fixed-share position exits at changed notional. Fees on both sides therefore
            // depend on 1 + exit_price/entry_price, not a constant 2 or 4 per cohort.
            let turnover = cohorts.sum(&(weights.abs() * (&gross_price + 1.)));
            policy_moments.push(Tensor::stack(
                &[
                    gross,
                    variance.sqrt(),
                    worst,
                    average(&exposure),
                    average(&net),
                    average(&active),
                    average(&turnover),
                ],
                0,
            ));
        }
        moments.push(Tensor::stack(&policy_moments, 0));
        census.push(Tensor::stack(
            &[periods, available.mean(Kind::Double), available.min()],
            0,
        ));
    }
    ensure!(
        Tensor::stack(&validity_checks, 0).all().int64_value(&[]) != 0,
        "invalid predictive dispersion or overflowing utility payoff"
    );
    let values = Vec::<f64>::try_from(
        Tensor::stack(&moments, 0)
            .to_device(Device::Cpu)
            .flatten(0, -1),
    )?;
    let counts = Vec::<f64>::try_from(
        Tensor::stack(&census, 0)
            .to_device(Device::Cpu)
            .flatten(0, -1),
    )?;
    let cross_sections: Vec<usize> = (0..horizons.len())
        .map(|h| counts[h * 3] as usize)
        .collect();
    let mut policies = Vec::with_capacity(LABELS.len());
    for (policy, label) in LABELS.iter().enumerate() {
        let column = |field: usize, scale: f64| -> Vec<f64> {
            (0..horizons.len())
                .map(|h| {
                    if cross_sections[h] == 0 {
                        f64::NAN
                    } else {
                        values[(h * LABELS.len() + policy) * 7 + field] * scale
                    }
                })
                .collect()
        };
        let gross_bps = column(0, 1e4);
        let turnover = column(6, 1.);
        let rate = |values: &[f64]| {
            values
                .iter()
                .zip(&horizons)
                .map(|(v, h)| v / *h as f64)
                .collect()
        };
        let net_bps: Vec<Vec<f64>> = COSTS_BPS
            .iter()
            .map(|cost| {
                gross_bps
                    .iter()
                    .zip(&turnover)
                    .map(|(gross, traded)| gross - cost * traded)
                    .collect()
            })
            .collect();
        policies.push(PolicyCurve {
            label: (*label).to_owned(),
            gross_rate_bps: rate(&gross_bps),
            net_rate_bps: net_bps.iter().map(|row| rate(row)).collect(),
            breakeven_bps: gross_bps
                .iter()
                .zip(&turnover)
                .map(|(gross, traded)| {
                    if *traded > 0. {
                        gross / traded
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
            gross_bps,
            net_bps,
            gross_exposure: column(3, 1.),
            net_exposure: column(4, 1.),
            active_fraction: column(5, 1.),
            turnover,
            payoff_std_bps: column(1, 1e4),
            worst_bps: column(2, 1e4),
        });
    }
    Ok(PortfolioCurve {
        tickers_per_timestamp: (0..horizons.len()).map(|h| counts[h * 3 + 1]).collect(),
        narrowest_timestamp: (0..horizons.len()).map(|h| counts[h * 3 + 2]).collect(),
        horizons,
        costs_bps: COSTS_BPS.to_vec(),
        policies,
        cross_sections,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9 * (1. + expected.abs()),
            "{actual} != {expected}"
        );
    }

    fn fixture(
        mean: &[f64],
        std: &[f64],
        target: &[f64],
        entry: &[f64],
        valid: &[f64],
        h: i64,
    ) -> PortfolioCurve {
        let n = mean.len() as i64 / h;
        evaluate(
            &Tensor::from_slice(mean).reshape([n, h]),
            &Tensor::from_slice(std).log().reshape([n, h]),
            &Tensor::from_slice(target).reshape([n, h]),
            &Tensor::from_slice(valid).reshape([n, h]),
            &Tensor::ones([n], (Kind::Double, Device::Cpu)),
            &Tensor::from_slice(entry),
            &Tensor::from_slice(&(0..n).map(|i| i / 20).collect::<Vec<_>>()),
            n / 20,
        )
        .unwrap()
    }

    #[test]
    fn next_open_simple_payoff_and_drifted_exit_cost_are_exact() {
        // A 100 -> 120 gap is untradeable before the next open. Only 120 -> 150 is earned.
        let curve = fixture(
            &[0.1; 20],
            &[0.2; 20],
            &[1.5_f64.ln(); 20],
            &[1.2_f64.ln(); 20],
            &[1.; 20],
            1,
        );
        let long = &curve.policies[1];
        near(long.gross_bps[0], 2500.);
        near(long.turnover[0], 2.25);
        for (i, cost) in curve.costs_bps.iter().enumerate() {
            near(long.net_bps[i][0], 2500. - 2.25 * cost);
        }
        near(long.breakeven_bps[0], 2500. / 2.25);
        near(curve.policies[0].net_bps[5][0], 0.);
        assert!(curve.policies[0].breakeven_bps[0].is_nan());
    }

    #[test]
    fn uncertainty_changes_decisions_and_abstention_keeps_cash() {
        let mean: Vec<f64> = (0..20).map(|i| if i < 10 { -0.02 } else { 0.02 }).collect();
        let mut std = vec![0.01; 20];
        let target = mean.clone();
        let confident = fixture(&mean, &std, &target, &[0.; 20], &[1.; 20], 1);
        std[..10].fill(0.1);
        let uncertain = fixture(&mean, &std, &target, &[0.; 20], &[1.; 20], 1);
        near(confident.policies[5].gross_exposure[0], 1.);
        near(uncertain.policies[5].gross_exposure[0], 0.5);
        near(uncertain.policies[5].net_exposure[0], 0.5);
        near(uncertain.policies[5].active_fraction[0], 0.5);
        near(
            confident.policies[2].gross_bps[0],
            uncertain.policies[2].gross_bps[0],
        );
        assert!(uncertain.policies[6].net_exposure[0] > confident.policies[6].net_exposure[0]);
    }

    #[test]
    fn uncertainty_ranking_is_a_distinct_policy_with_the_same_budget() {
        let mean: Vec<f64> = (1..=20).map(|i| i as f64 * 0.001).collect();
        let mut std = vec![1.; 20];
        let mut target = vec![-0.01; 20];
        target[8..10].fill(0.01);
        let uniform = fixture(&mean, &std, &target, &[0.; 20], &[1.; 20], 1);
        near(
            uniform.policies[3].gross_bps[0],
            uniform.policies[4].gross_bps[0],
        );
        std[..10].fill(0.001);
        let uncertain = fixture(&mean, &std, &target, &[0.; 20], &[1.; 20], 1);
        assert!(uncertain.policies[4].gross_bps[0] > uncertain.policies[3].gross_bps[0]);
        for policy in &uncertain.policies[3..5] {
            near(policy.gross_exposure[0], 1.);
            near(policy.net_exposure[0], 0.);
        }
    }

    #[test]
    fn tied_predictions_never_manufacture_decile_trades() {
        let target: Vec<f64> = (0..20).map(|i| (i as f64 - 10.) * 0.01).collect();
        let curve = fixture(&[0.; 20], &[1.; 20], &target, &[0.; 20], &[1.; 20], 1);
        for policy in &curve.policies[2..] {
            near(policy.gross_exposure[0], 0.);
            near(policy.turnover[0], 0.);
            near(policy.net_bps[5][0], 0.);
        }
    }

    #[test]
    fn missing_outcome_rejects_cohort_instead_of_redistributing_capital() {
        let mut target = vec![0.01; 40];
        target[20..].fill(-0.02);
        let mut valid = vec![1.; 40];
        valid[0] = 0.;
        target[0] = f64::NAN;
        let curve = fixture(&[0.1; 40], &[0.1; 40], &target, &[0.; 40], &valid, 1);
        assert_eq!(curve.cross_sections, vec![1]);
        near(curve.policies[1].gross_bps[0], (-0.02_f64).exp_m1() * 1e4);
        near(curve.narrowest_timestamp[0], 19.);
        valid[20] = 0.;
        let empty = fixture(&[0.1; 40], &[0.1; 40], &target, &[0.; 40], &valid, 1);
        assert_eq!(empty.cross_sections, vec![0]);
        assert!(empty
            .policies
            .iter()
            .all(|policy| policy.gross_bps[0].is_nan()));
    }

    #[test]
    fn terminal_forecast_not_near_term_direction_controls_long_hold() {
        let h = 192;
        let mut mean = vec![0.02; 20 * h];
        let mut target = vec![0.01; 20 * h];
        for row in 0..20 {
            mean[row * h + h - 1] = -0.02;
            target[row * h + h - 1] = -0.03;
        }
        let curve = fixture(
            &mean,
            &vec![0.01; 20 * h],
            &target,
            &[0.; 20],
            &vec![1.; 20 * h],
            h as i64,
        );
        let last = curve.horizons.len() - 1;
        assert_eq!(curve.horizons[last], 192);
        near(curve.policies[2].net_exposure[0], 1.);
        near(curve.policies[2].net_exposure[last], -1.);
        near(
            curve.policies[2].gross_bps[last],
            -(-0.03_f64).exp_m1() * 1e4,
        );
        for policy in &curve.policies {
            assert!(policy.gross_exposure.iter().all(|g| *g <= 1. + 1e-12));
        }
        // Realized returns cannot change the origin decision, even when they reverse sign.
        target.iter_mut().for_each(|value| *value = -*value);
        let reversed = fixture(
            &mean,
            &vec![0.01; 20 * h],
            &target,
            &[0.; 20],
            &vec![1.; 20 * h],
            h as i64,
        );
        for (original, changed) in curve.policies.iter().zip(&reversed.policies) {
            assert_eq!(original.gross_exposure, changed.gross_exposure);
            assert_eq!(original.net_exposure, changed.net_exposure);
        }
    }
}
