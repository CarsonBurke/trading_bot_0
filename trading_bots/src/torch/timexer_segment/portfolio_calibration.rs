//! Diagnostics for the exact scalar portfolio estimator, not a second calibration policy.
//!
//! Moments are pooled after centering each origin timestamp's cross-section. Every applicable
//! row has equal weight: moments divide by N, not by each cohort's N-1. Singleton cohorts supply
//! no cross-sectional information and are excluded from ALL variance denominators. Counts expose
//! this population and its N-G residual degrees of freedom; neither is an independent sample size.
//! Uncertainty deletes whole New York calendar sessions, retaining within-session dependence.
//! Adjacent sessions may still be dependent (in particular with overlapping forward windows), so
//! the jackknife does not establish independent clusters. Its normal interval is approximate,
//! especially with few sessions; fewer than 30 informative sessions is explicitly flagged.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, ensure};
use chrono::{NaiveDate, TimeZone};
use chrono_tz::America::New_York;

#[derive(Clone, Copy, Default)]
struct Moments {
    xx: f64,
    xy: f64,
    yy: f64,
    one_bar: f64,
    diagonal: f64,
}

impl Moments {
    fn plus(self, other: Self) -> Self {
        Self {
            xx: self.xx + other.xx,
            xy: self.xy + other.xy,
            yy: self.yy + other.yy,
            one_bar: self.one_bar + other.one_bar,
            diagonal: self.diagonal + other.diagonal,
        }
    }

    fn finite(self) -> bool {
        [self.xx, self.xy, self.yy, self.one_bar, self.diagonal]
            .into_iter()
            .all(f64::is_finite)
    }

    fn gain(self) -> Option<f64> {
        (self.xx > 0.)
            .then(|| self.xy / self.xx)
            .filter(|gain| gain.is_finite())
    }
}

#[derive(Default)]
struct Cohort {
    n: usize,
    means: [f64; 3],
    moments: Moments,
}

impl Cohort {
    fn push(&mut self, row: &[f64; 4]) {
        self.n += 1;
        let delta = [
            row[0] - self.means[0],
            row[1] - self.means[1],
            row[2] - self.means[2],
        ];
        for (mean, delta) in self.means.iter_mut().zip(delta) {
            *mean += delta / self.n as f64;
        }
        // Online centered co-moments avoid subtracting two large uncentered sums of squares.
        self.moments.xx += delta[0] * (row[0] - self.means[0]);
        self.moments.xy += delta[0] * (row[1] - self.means[1]);
        self.moments.yy += delta[1] * (row[1] - self.means[1]);
        self.moments.one_bar += delta[2] * (row[2] - self.means[2]);
        self.moments.diagonal += row[3];
    }
}

fn metric(summary: &mut BTreeMap<String, f64>, prefix: &str, name: &str, value: f64) {
    if value.is_finite() {
        summary.insert(format!("{prefix}_{name}"), value);
    }
}

/// `rows[i]` is `[raw_forecast_h, residual_h, residual_1, centered_increment_squares]`,
/// all in the same origin's sigma units (the fourth coordinate is squared sigma units).
/// Its fourth coordinate MUST be `sum_{s=1..h}(increment[i,s] - mean_timestamp[s])^2`,
/// with increments including step 1 from the zero residual origin. Center on precisely the
/// complete timestamp cohort supplied here, including across inference batches; merely summing
/// uncentered squared increments does NOT identify the diagonal variance denominator Dh.
///
/// Timestamps are origin bar starts in Unix milliseconds. Input order is immaterial; nonfinite
/// data are rejected rather than silently changing the population used to construct coordinate 4.
/// Undefined derived metrics are omitted. Zero forecast variance leaves raw gain undefined,
/// but the nonempty flat NNLS objective retains its established minimum-norm minimizer zero.
/// A defined negative raw gain also produces the unchanged exact NNLS boundary zero.
///
/// `_gain_ratio` is Cov(f,y)/Var(f); `_nnls_gain_ratio` is its nonnegative projection when defined.
/// `_target_to_diagonal_variance_ratio` is Vh/Dh; `_target_to_walk_variance_ratio` is Vh/(h V1).
/// These have distinct measured denominators. `_variance_scale_ratio = sqrt(Vh/(h V1))`
/// and `_normalized_forecast_gain_ratio = gain / variance_scale` separate variance normalization
/// from signal amplitude algebraically: multiplying the forecast by both leaves its calibrated
/// mean unchanged. None of these diagnostics supplies a policy beyond the reported scalar gain.
/// Jackknife SE and normal-approximate bounds refer to the RAW, not boundary-constrained, gain.
pub(super) fn diagnostics(
    rows: &[[f64; 4]],
    timestamps: &[i64],
    horizon: usize,
    prefix: &str,
) -> Result<BTreeMap<String, f64>> {
    ensure!(
        horizon > 0,
        "calibration diagnostic horizon must be positive"
    );
    ensure!(!prefix.is_empty(), "calibration diagnostic prefix is empty");
    ensure!(
        rows.len() == timestamps.len(),
        "calibration rows/timestamps differ in length"
    );
    let mut cohorts = BTreeMap::<i64, Cohort>::new();
    for (index, (row, &timestamp)) in rows.iter().zip(timestamps).enumerate() {
        ensure!(
            row.iter().all(|value| value.is_finite()),
            "nonfinite calibration row {index}"
        );
        ensure!(
            row[3] >= 0.,
            "negative centered increment squares at row {index}"
        );
        cohorts.entry(timestamp).or_default().push(row);
    }

    let mut sessions = BTreeSet::<NaiveDate>::new();
    let mut blocks = BTreeMap::<NaiveDate, Moments>::new();
    let mut applicable_origins = 0usize;
    let mut applicable_timestamps = 0usize;
    let mut informative_origins = 0usize;
    let mut informative_timestamps = 0usize;
    for (&timestamp, cohort) in &cohorts {
        let session = New_York
            .timestamp_millis_opt(timestamp)
            .single()
            .with_context(|| format!("invalid calibration origin timestamp {timestamp}"))?
            .date_naive();
        sessions.insert(session);
        ensure!(
            cohort.moments.finite(),
            "overflow in calibration timestamp {timestamp}"
        );
        if cohort.n < 2 {
            continue;
        }
        applicable_origins += cohort.n;
        applicable_timestamps += 1;
        if cohort.moments.xx > 0. {
            informative_origins += cohort.n;
            informative_timestamps += 1;
        }
        let block = blocks.entry(session).or_default();
        *block = block.plus(cohort.moments);
        ensure!(block.finite(), "overflow in calibration session {session}");
    }
    let informative_sessions = blocks.values().filter(|block| block.xx > 0.).count();
    let mut summary = BTreeMap::new();
    for (name, value) in [
        ("origins_count", rows.len()),
        ("timestamps_count", cohorts.len()),
        ("sessions_count", sessions.len()),
        ("applicable_origins_count", applicable_origins),
        ("applicable_timestamps_count", applicable_timestamps),
        ("applicable_sessions_count", blocks.len()),
        ("singleton_origins_count", rows.len() - applicable_origins),
        (
            "within_timestamp_degrees_of_freedom_count",
            applicable_origins - applicable_timestamps,
        ),
        ("gain_informative_origins_count", informative_origins),
        ("gain_informative_timestamps_count", informative_timestamps),
        ("gain_informative_sessions_count", informative_sessions),
        (
            "gain_uncertainty_low_precision_count",
            usize::from(informative_sessions < 30),
        ),
        ("horizon_count", horizon),
    ] {
        metric(&mut summary, prefix, name, value as f64);
    }
    let mut width_bins = [0usize; 5];
    let mut min_width = usize::MAX;
    let mut max_width = 0usize;
    for cohort in cohorts.values() {
        min_width = min_width.min(cohort.n);
        max_width = max_width.max(cohort.n);
        let bin = match cohort.n {
            1 => 0,
            2..=9 => 1,
            10..=39 => 2,
            40..=99 => 3,
            _ => 4,
        };
        width_bins[bin] += 1;
    }
    for (name, count) in [
        "timestamp_width_1_count",
        "timestamp_width_2_to_9_count",
        "timestamp_width_10_to_39_count",
        "timestamp_width_40_to_99_count",
        "timestamp_width_100_plus_count",
    ]
    .into_iter()
    .zip(width_bins)
    {
        metric(&mut summary, prefix, name, count as f64);
    }
    if !cohorts.is_empty() {
        metric(
            &mut summary,
            prefix,
            "cohort_width_min_count",
            min_width as f64,
        );
        metric(
            &mut summary,
            prefix,
            "cohort_width_max_count",
            max_width as f64,
        );
        metric(
            &mut summary,
            prefix,
            "cohort_width_mean_count",
            rows.len() as f64 / cohorts.len() as f64,
        );
    }

    metric(&mut summary, prefix, "shrinkage_weight_fraction", 0.);

    // Prefix/suffix sums avoid cancellation from total-minus-dominant-session in leave-one-out
    // variances, and make all delete-one computations O(number of sessions), not O(N * sessions).
    let blocks: Vec<_> = blocks.into_values().collect();
    let mut suffix = vec![Moments::default(); blocks.len() + 1];
    for index in (0..blocks.len()).rev() {
        suffix[index] = blocks[index].plus(suffix[index + 1]);
        ensure!(
            suffix[index].finite(),
            "overflow in pooled calibration moments"
        );
    }
    let total = suffix[0];
    if applicable_origins > 0 {
        let n = applicable_origins as f64;
        for (name, value) in [
            ("forecast_variance_ratio", total.xx / n),
            ("forecast_target_covariance_ratio", total.xy / n),
            ("target_variance_ratio", total.yy / n),
            ("target_one_bar_variance_ratio", total.one_bar / n),
            ("target_diagonal_variance_ratio", total.diagonal / n),
        ] {
            metric(&mut summary, prefix, name, value);
        }
    }
    if total.diagonal > 0. {
        metric(
            &mut summary,
            prefix,
            "target_to_diagonal_variance_ratio",
            total.yy / total.diagonal,
        );
    }
    let walk_ratio = if total.one_bar > 0. {
        let ratio = (total.yy / horizon as f64) / total.one_bar;
        metric(&mut summary, prefix, "target_to_walk_variance_ratio", ratio);
        metric(
            &mut summary,
            prefix,
            "target_diagonal_to_walk_variance_ratio",
            (total.diagonal / horizon as f64) / total.one_bar,
        );
        ratio.is_finite().then_some(ratio)
    } else {
        None
    };
    if let Some(ratio) = walk_ratio {
        metric(&mut summary, prefix, "variance_scale_ratio", ratio.sqrt());
    }
    if let Some(gain) = total.gain() {
        metric(&mut summary, prefix, "gain_ratio", gain);
        metric(&mut summary, prefix, "nnls_gain_ratio", gain.max(0.));
        if let Some(scale) = walk_ratio.map(f64::sqrt).filter(|scale| *scale > 0.) {
            metric(
                &mut summary,
                prefix,
                "normalized_forecast_gain_ratio",
                gain / scale,
            );
            metric(
                &mut summary,
                prefix,
                "normalized_forecast_nnls_gain_ratio",
                gain.max(0.) / scale,
            );
        }
    }
    if applicable_origins > 0 && total.xx == 0. {
        // Every nonnegative gain minimizes a flat objective; preserve the minimum-norm policy.
        // This is not a numerical substitute for the undefined raw covariance/variance ratio.
        metric(&mut summary, prefix, "nnls_gain_ratio", 0.);
    }

    let mut prefix_sum = Moments::default();
    let mut valid_replicates = 0usize;
    let mut replicate_mean = 0.;
    let mut replicate_m2 = 0.;
    for (index, block) in blocks.iter().enumerate() {
        if let Some(gain) = prefix_sum.plus(suffix[index + 1]).gain() {
            valid_replicates += 1;
            let delta = gain - replicate_mean;
            replicate_mean += delta / valid_replicates as f64;
            replicate_m2 += delta * (gain - replicate_mean);
        }
        prefix_sum = prefix_sum.plus(*block);
    }
    metric(
        &mut summary,
        prefix,
        "gain_jackknife_replicates_count",
        valid_replicates as f64,
    );
    // Dropping undefined replicates changes the estimand: require every deletion to be valid.
    if blocks.len() >= 2 && valid_replicates == blocks.len() {
        let m = blocks.len() as f64;
        let standard_error = ((m - 1.) / m * replicate_m2).sqrt();
        metric(
            &mut summary,
            prefix,
            "gain_jackknife_standard_error_ratio",
            standard_error,
        );
        if let Some(gain) = total.gain() {
            metric(
                &mut summary,
                prefix,
                "gain_normal_approx_95_lower_ratio",
                gain - 1.959963984540054 * standard_error,
            );
            metric(
                &mut summary,
                prefix,
                "gain_normal_approx_95_upper_ratio",
                gain + 1.959963984540054 * standard_error,
            );
        }
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timestamp(value: &str) -> i64 {
        chrono::DateTime::parse_from_rfc3339(value)
            .unwrap()
            .timestamp_millis()
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs().max(1.),
            "{actual} != {expected}"
        );
    }

    #[test]
    fn session_jackknife_does_not_invent_precision_from_duplicate_names() {
        let times = [
            timestamp("2025-01-06T20:00:00Z"),
            timestamp("2025-01-07T20:00:00Z"),
            timestamp("2025-01-08T20:00:00Z"),
        ];
        let mut rows = Vec::new();
        let mut origins = Vec::new();
        for (&time, gain) in times.iter().zip([1., 2., 4.]) {
            rows.extend([[-1., -gain, -1., 1.], [1., gain, 1., 1.]]);
            origins.extend([time; 2]);
        }
        let result = diagnostics(&rows, &origins, 1, "recent").unwrap();
        close(result["recent_gain_ratio"], 7. / 3.);
        close(
            result["recent_gain_jackknife_standard_error_ratio"],
            7f64.sqrt() / 3.,
        );
        close(result["recent_gain_jackknife_replicates_count"], 3.);
        close(result["recent_gain_uncertainty_low_precision_count"], 1.);
        rows.extend_from_within(..);
        origins.extend_from_within(..);
        let repeated = diagnostics(&rows, &origins, 1, "recent").unwrap();
        close(
            repeated["recent_gain_jackknife_standard_error_ratio"],
            result["recent_gain_jackknife_standard_error_ratio"],
        );
        close(repeated["recent_applicable_origins_count"], 12.);
        close(repeated["recent_applicable_sessions_count"], 3.);
    }

    #[test]
    fn unequal_cohort_widths_pool_centered_information_not_cohort_gains() {
        let a = timestamp("2025-01-06T20:00:00Z");
        let b = timestamp("2025-01-07T20:00:00Z");
        // Interleaved timestamps: Sxx=(2,8), Sxy=(2,32). Averaging cohort gains gives
        // 2.5, not the pooled estimator 3.4. The two delete-day gains are 4 and 1.
        let rows = [
            [9., 19., -1., 1.],
            [98., 192., -2., 4.],
            [11., 21., 1., 1.],
            [100., 200., 0., 0.],
            [102., 208., 2., 4.],
        ];
        let result = diagnostics(&rows, &[a, b, a, b, b], 1, "full").unwrap();
        close(result["full_gain_ratio"], 3.4);
        close(result["full_forecast_variance_ratio"], 2.);
        close(result["full_forecast_target_covariance_ratio"], 6.8);
        close(result["full_within_timestamp_degrees_of_freedom_count"], 3.);
        close(result["full_gain_jackknife_standard_error_ratio"], 1.5);
    }

    #[test]
    fn unequal_step_variances_are_not_the_random_walk_denominator() {
        // Centered increments are +/-1 then +/-2: V2=9, D2=1+4=5, 2V1=2.
        // Large common offsets must not become target or forecast cross-sectional variance.
        let rows = [
            [9., 17., 29., 5.],
            [11., 23., 31., 5.],
            [100., -100., 7., 0.],
        ];
        let origin = timestamp("2025-01-06T20:00:00Z");
        let result = diagnostics(&rows, &[origin, origin, origin + 300_000], 2, "full").unwrap();
        close(result["full_target_variance_ratio"], 9.);
        close(result["full_target_to_diagonal_variance_ratio"], 9. / 5.);
        close(result["full_target_to_walk_variance_ratio"], 9. / 2.);
        close(
            result["full_target_diagonal_to_walk_variance_ratio"],
            5. / 2.,
        );
        close(
            result["full_target_to_diagonal_variance_ratio"]
                * result["full_target_diagonal_to_walk_variance_ratio"],
            result["full_target_to_walk_variance_ratio"],
        );
        close(result["full_gain_ratio"], 3.);
        close(result["full_applicable_origins_count"], 2.);
        close(result["full_timestamp_width_1_count"], 1.);
        close(result["full_timestamp_width_2_to_9_count"], 1.);
        close(result["full_cohort_width_mean_count"], 1.5);
        close(result["full_singleton_origins_count"], 1.);
        close(result["full_within_timestamp_degrees_of_freedom_count"], 1.);
        for forecast in [-2., 0., 3.] {
            close(
                result["full_variance_scale_ratio"]
                    * result["full_normalized_forecast_gain_ratio"]
                    * forecast,
                result["full_gain_ratio"] * forecast,
            );
        }
        assert!(!result.contains_key("full_gain_jackknife_standard_error_ratio"));
    }

    #[test]
    fn new_york_days_not_utc_days_define_clusters() {
        let time_a = timestamp("2025-01-06T20:00:00Z");
        let time_b = timestamp("2025-01-07T01:00:00Z");
        let rows = [
            [-1., -1., -1., 1.],
            [1., 1., 1., 1.],
            [-1., -2., -1., 1.],
            [1., 2., 1., 1.],
        ];
        let result = diagnostics(&rows, &[time_a, time_a, time_b, time_b], 1, "recent").unwrap();
        close(result["recent_applicable_sessions_count"], 1.);
        close(result["recent_applicable_timestamps_count"], 2.);
        assert!(!result.contains_key("recent_gain_jackknife_standard_error_ratio"));
    }

    #[test]
    fn negative_gain_keeps_nnls_boundary_and_undefined_deletions_have_no_se() {
        let a = timestamp("2025-01-06T20:00:00Z");
        let b = timestamp("2025-01-07T20:00:00Z");
        let rows = [
            [-1., 2., -1., 1.],
            [1., -2., 1., 1.],
            [0., -2., -1., 1.],
            [0., 2., 1., 1.],
        ];
        let result = diagnostics(&rows, &[a, a, b, b], 1, "recent").unwrap();
        close(result["recent_gain_ratio"], -2.);
        close(result["recent_nnls_gain_ratio"], 0.);
        close(result["recent_gain_jackknife_replicates_count"], 1.);
        assert!(!result.contains_key("recent_gain_jackknife_standard_error_ratio"));
        assert!(!result.contains_key("recent_gain_normal_approx_95_lower_ratio"));
    }

    #[test]
    fn undefined_denominators_are_absent_and_invalid_data_are_errors() {
        let origin = timestamp("2025-01-06T20:00:00Z");
        let rows = [[0., -1., 0., 0.], [0., 1., 0., 0.]];
        let result = diagnostics(&rows, &[origin; 2], 2, "full").unwrap();
        assert!(!result.contains_key("full_gain_ratio"));
        close(result["full_nnls_gain_ratio"], 0.);
        assert!(!result.contains_key("full_target_to_walk_variance_ratio"));
        assert!(!result.contains_key("full_target_to_diagonal_variance_ratio"));
        let empty = diagnostics(&[], &[], 2, "full").unwrap();
        assert!(!empty.contains_key("full_target_variance_ratio"));
        assert!(!empty.contains_key("full_nnls_gain_ratio"));
        close(empty["full_applicable_origins_count"], 0.);
        assert!(diagnostics(&[[f64::NAN, 0., 0., 0.]], &[origin], 2, "full").is_err());
        assert!(diagnostics(&[[0., 0., 0., -1.]], &[origin], 2, "full").is_err());
        assert!(diagnostics(&rows, &[origin], 2, "full").is_err());
        assert!(diagnostics(&rows, &[origin; 2], 0, "full").is_err());
        assert!(diagnostics(&rows, &[i64::MAX; 2], 2, "full").is_err());
    }
}
