//! Measure fitted conditional moments against an EXISTING support and persist them without
//! changing its geometry.
//!
//! The current upgrade adds, for DOF `r`, direct train-sample estimates of
//! `E[expm1(r) | bin]` and `E[expm1(r)^2 | bin]` alongside the existing log-space moments.
//! The bin edges, atom table, histogram, smoothed marginal, and provenance are copied
//! value-for-value. This is a clean artifact migration: consumers requiring economic moments
//! reject older files rather than reconstructing them from centers or log-space moments.
//!
//! The pass is one exact scan over the fixed training draw with fixed-size accumulators.
//! Observations are routed through the support's existing atom-aware binner. The two open tail
//! bins therefore retain raw, unclamped tail observations, atom bins retain exact point-mass
//! provenance, and neither case is laundered into a geometric representative.

use std::path::Path;

use anyhow::{ensure, Context, Result};

use crate::torch::bar_dist::{
    BarDof, BarSupports, MeanDecode, BAR_DOF, BAR_DOF_NAMES, DOF_R, NUM_BAR_BINS,
};
use crate::torch::train::pretrain::{load_corpus, CorpusFlags};

use super::pretrain_reports::write_support_decode;

/// Everything the moments pass needs. Deliberately separate from `PretrainArgs`: this touches
/// no model, no device and no schedule, and coupling it to the trainer's argument surface would
/// make an artifact upgrade require a training configuration it has no use for.
#[derive(Clone, Debug)]
pub struct SupportMomentsArgs {
    pub corpus: CorpusFlags,
    /// Support to measure moments FOR. Read, never written.
    pub supports: String,
    /// Where the upgraded current-schema artifact lands. A DIFFERENT path from `supports` by
    /// intent: every checkpoint's `.supports.<res>.json` sidecar is covered by its own
    /// `supports_sha256` and by `lineage_sha256`, so rewriting a support in place would make an
    /// existing checkpoint unloadable against its own training geometry.
    pub output_supports: String,
    /// Reports directory, i.e. a run's `gens/<n>`.
    pub output: String,
    /// Rows to draw. MUST match the `sample_count` the support's provenance records, or the
    /// moments describe a different population than the masses beside them.
    pub samples: usize,
    /// Draw seed. MUST be the `train_seed` of the run that fitted the support.
    pub seed: u64,
    /// Largest per-bin absolute mass deviation accepted when identifying the redrawn sample
    /// against the persisted histogram.
    pub mass_tolerance: f64,
}

/// Default agreement slack for the histogram identification.
///
/// The persisted masses are `count / total` in `f64` and the redraw recomputes them the same
/// way, so an IDENTICAL sample under an IDENTICAL binning rule agrees EXACTLY and any tolerance
/// at all is generous. It is not zero only because the persisted values have been through
/// decimal JSON: `serde_json` round-trips `f64` faithfully, but a support hand-edited or written
/// by another serializer need not, and a rejection on the seventeenth digit would be a false
/// alarm. `1e-12` is five orders below the smallest per-bin mass a 128-way equal-mass support
/// carries, so it cannot hide a real population difference: ONE misbinned row in four million
/// moves a mass by 2.5e-7, which is 250,000 times this tolerance.
pub const DEFAULT_MASS_TOLERANCE: f64 = 1e-12;

/// One open-tail bin under geometric and fitted-moment conventions.
#[derive(Clone, Copy, Debug)]
struct CatchAll {
    /// Bin index, i.e. `0` or `NUM_BAR_BINS - 1`.
    bin: usize,
    /// Marginal probability of the bin.
    mass: f64,
    /// Geometric support representative used by sampling and the CRPS grid.
    edge: f64,
    /// Persisted train-fitted conditional mean used by moment consumers.
    fitted: f64,
}

/// The census of one DOF's decode, under one convention.
///
/// Every share here is a share of a TOTAL over absolute or squared quantities, never of the
/// signed sum. The signed marginal mean of `r` is -0.0949 bps — near-total cancellation between
/// the two catch-alls — so a "share" of it is a ratio of two nearly-zero numbers and means
/// nothing. The NET is what moves the mean; the TOTAL is what drives the estimation variance,
/// and the shares below are about the latter.
#[derive(Clone, Copy, Debug)]
struct DecodeCensus {
    /// `sum_b p_b d_b`, the marginal mean this decode implies.
    mean: f64,
    /// Outer share of `sum_b p_b |d_b|`.
    first_share: f64,
    /// Outer share of `sum_b p_b (d_b - mean)^2`.
    second_share: f64,
    /// `max_b d_b - min_b d_b`, the span of means this DOF can represent.
    span: f64,
    /// Outer share of that span, i.e. `1 - interior_span / span`.
    span_share: f64,
    /// `max_b |d_b|`.
    ceiling: f64,
    /// `max_b |d_b|` over the interior bins only.
    interior_ceiling: f64,
}

impl DecodeCensus {
    /// The census of `decode` weighted by `mass`.
    ///
    /// `sum_j p_j (d_j - mu)^2` is not merely a leverage heuristic. For the plug-in decoded mean
    /// `mu_hat = sum_j (n_j / N) d_j` over `N` iid rows, the counts are multinomial and
    /// `Var(mu_hat) = (1/N) sum_j p_j (d_j - mu)^2` EXACTLY. So `second_share` IS the share of
    /// the decoded mean's estimation variance that sits in the two catch-alls. Second-moment
    /// leverage and first-moment controllability are one arithmetic object, not two.
    fn measure(decode: &[f64], mass: &[f64]) -> Self {
        let last = decode.len() - 1;
        let outer = |row: &[f64]| row[0] + row[last];
        let mean: f64 = decode.iter().zip(mass).map(|(d, p)| p * d).sum();

        let first: Vec<f64> = decode.iter().zip(mass).map(|(d, p)| p * d.abs()).collect();
        let second: Vec<f64> = decode
            .iter()
            .zip(mass)
            .map(|(d, p)| p * (d - mean) * (d - mean))
            .collect();
        let (first_total, second_total) = (first.iter().sum::<f64>(), second.iter().sum::<f64>());

        let extent = |row: &[f64]| -> f64 {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for x in row {
                lo = lo.min(*x);
                hi = hi.max(*x);
            }
            hi - lo
        };
        let span = extent(decode);
        let interior_span = extent(&decode[1..last]);
        let peak = |row: &[f64]| row.iter().fold(0.0f64, |worst, x| worst.max(x.abs()));

        Self {
            mean,
            // A share of a zero total is UNDEFINED, not zero: it would be the ratio of two
            // absent quantities. Only reachable on a degenerate DOF whose whole law sits at 0.
            first_share: ratio(outer(&first), first_total),
            second_share: ratio(outer(&second), second_total),
            span,
            span_share: if span > 0.0 {
                1.0 - interior_span / span
            } else {
                f64::NAN
            },
            ceiling: peak(decode),
            interior_ceiling: peak(&decode[1..last]),
        }
    }
}

/// `numerator / denominator`, or `NaN` when the denominator carries nothing to take a share of.
fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        f64::NAN
    }
}

/// Everything this pass measured, per DOF.
pub struct SupportDecode {
    /// `[BAR_DOF]` bin masses, straight off the artifact.
    mass: [Vec<f64>; BAR_DOF],
    /// `[BAR_DOF]` edge decode, i.e. `centers` with the catch-alls at the bounds.
    edge: [Vec<f64>; BAR_DOF],
    /// `[BAR_DOF]` measured conditional means.
    fitted: [Vec<f64>; BAR_DOF],
    /// `[BAR_DOF]` measured conditional second moments.
    second: [Vec<f64>; BAR_DOF],
    /// Directly measured `E[R | bin]` for `R = expm1(r)`.
    simple_return_mean: Vec<f64>,
    /// Directly measured `E[R^2 | bin]`, including within-bin variance.
    simple_return_second: Vec<f64>,
    /// Numeric provenance for the r-bin diagnostic: -1 lower open tail, 0 continuous,
    /// 1 upper open tail, 2 explicit atom.
    r_bin_provenance: Vec<f64>,
    /// Per-DOF census under the geometric decode.
    edge_census: [DecodeCensus; BAR_DOF],
    /// Per-DOF census under the persisted fitted-moment decode.
    fitted_census: [DecodeCensus; BAR_DOF],
    /// The lower and upper open-tail bins of every DOF.
    catch_alls: [[CatchAll; 2]; BAR_DOF],
    /// Worst per-bin mass deviation the histogram identification tolerated, per DOF.
    mass_agreement: [f64; BAR_DOF],
    /// Rows the moments were measured on.
    rows: usize,
}

impl SupportDecode {
    /// Census a support that already carries measured moments.
    pub fn of(supports: &BarSupports, mass_agreement: [f64; BAR_DOF], rows: usize) -> Result<Self> {
        let last = NUM_BAR_BINS as usize - 1;
        let mut mass: [Vec<f64>; BAR_DOF] = std::array::from_fn(|_| Vec::new());
        let mut edge: [Vec<f64>; BAR_DOF] = std::array::from_fn(|_| Vec::new());
        let mut fitted: [Vec<f64>; BAR_DOF] = std::array::from_fn(|_| Vec::new());
        let mut second: [Vec<f64>; BAR_DOF] = std::array::from_fn(|_| Vec::new());
        for dof in 0..BAR_DOF {
            mass[dof] = supports.bin_masses(dof).to_vec();
            edge[dof] = supports.mean_decode(dof, MeanDecode::Edge)?.to_vec();
            fitted[dof] = supports.mean_decode(dof, MeanDecode::Fitted)?.to_vec();
            second[dof] = supports
                .bin_second_moments(dof)
                .context("a support carrying fitted means must carry second moments too")?
                .to_vec();
        }
        let (simple_return_mean, simple_return_second) =
            supports.simple_return_bin_moments().context(
                "the support decode diagnostic requires directly fitted r simple-return first \
                 and second moments",
            )?;
        let atom_bins: std::collections::HashSet<usize> =
            supports.atoms(DOF_R).iter().map(|atom| atom.bin).collect();
        let r_bin_provenance = (0..=last)
            .map(|bin| {
                if atom_bins.contains(&bin) {
                    2.0
                } else if bin == 0 {
                    -1.0
                } else if bin == last {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        let edge_census = std::array::from_fn(|dof| DecodeCensus::measure(&edge[dof], &mass[dof]));
        let fitted_census =
            std::array::from_fn(|dof| DecodeCensus::measure(&fitted[dof], &mass[dof]));
        let catch_alls = std::array::from_fn(|dof| {
            [0usize, last].map(|bin| CatchAll {
                bin,
                mass: mass[dof][bin],
                edge: edge[dof][bin],
                fitted: fitted[dof][bin],
            })
        });
        Ok(Self {
            mass,
            edge,
            fitted,
            second,
            simple_return_mean: simple_return_mean.to_vec(),
            simple_return_second: simple_return_second.to_vec(),
            r_bin_provenance,
            edge_census,
            fitted_census,
            catch_alls,
            mass_agreement,
            rows,
        })
    }

    /// Per-DOF summary series, in the order the report writer consumes them.
    ///
    /// Labels distinguish geometric representatives from fitted moments at every use. Geometry
    /// remains exact for sampling/CRPS; predictive first and second moments use the persisted
    /// fitted rows.
    pub fn summary_rows(&self) -> Vec<(String, Vec<f64>)> {
        let per_dof =
            |pick: &dyn Fn(usize) -> f64| -> Vec<f64> { (0..BAR_DOF).map(pick).collect() };
        let bps = 1e4;
        let mut rows: Vec<(String, Vec<f64>)> = vec![
            ("dof index".to_owned(), per_dof(&|dof| dof as f64)),
            (
                "catch-all mass, % of the law".to_owned(),
                per_dof(&|dof| {
                    100.0 * (self.catch_alls[dof][0].mass + self.catch_alls[dof][1].mass)
                }),
            ),
        ];
        for (tag, census) in [
            (
                "geometric support decode [sampling/CRPS]",
                &self.edge_census,
            ),
            (
                "persisted fitted conditional means [moments]",
                &self.fitted_census,
            ),
        ] {
            rows.extend([
                (
                    format!("marginal mean, bps [{tag}]"),
                    per_dof(&|dof| census[dof].mean * bps),
                ),
                (
                    format!("catch-all share of |first moment|, % [{tag}]"),
                    per_dof(&|dof| 100.0 * census[dof].first_share),
                ),
                (
                    format!("catch-all share of central 2nd moment, % [{tag}]"),
                    per_dof(&|dof| 100.0 * census[dof].second_share),
                ),
                (
                    format!("reachable mean span, bps [{tag}]"),
                    per_dof(&|dof| census[dof].span * bps),
                ),
                (
                    format!("catch-all share of that span, % [{tag}]"),
                    per_dof(&|dof| 100.0 * census[dof].span_share),
                ),
                (
                    format!("representable mean ceiling, bps [{tag}]"),
                    per_dof(&|dof| census[dof].ceiling * bps),
                ),
                (
                    format!("interior mean ceiling, bps [{tag}]"),
                    per_dof(&|dof| census[dof].interior_ceiling * bps),
                ),
            ]);
        }
        // Every fitted value below comes directly from the persisted all-bin moment rows.
        for (slot, side) in [(0usize, "lower"), (1usize, "upper")] {
            rows.extend([
                (
                    format!("{side} open-tail geometric representative, bps [sampling/CRPS]"),
                    per_dof(&|dof| self.catch_alls[dof][slot].edge * bps),
                ),
                (
                    format!("{side} open-tail E[x|bin], bps [PERSISTED fitted moment]"),
                    per_dof(&|dof| self.catch_alls[dof][slot].fitted * bps),
                ),
                (
                    format!("{side} open-tail: geometry minus fitted E[x|bin], bps"),
                    per_dof(&|dof| {
                        (self.catch_alls[dof][slot].edge - self.catch_alls[dof][slot].fitted) * bps
                    }),
                ),
            ]);
        }
        let r_only = |value: f64| per_dof(&|dof| if dof == DOF_R { value } else { f64::NAN });
        let simple_mean: f64 = self.mass[DOF_R]
            .iter()
            .zip(&self.simple_return_mean)
            .map(|(p, mean)| p * mean)
            .sum();
        let simple_second: f64 = self.mass[DOF_R]
            .iter()
            .zip(&self.simple_return_second)
            .map(|(p, second)| p * second)
            .sum();
        let squared_first: f64 = self.mass[DOF_R]
            .iter()
            .zip(&self.simple_return_mean)
            .map(|(p, mean)| p * mean * mean)
            .sum();
        rows.extend([
            (
                "r marginal E[R], bps [DIRECT train-fitted simple return]".to_owned(),
                r_only(simple_mean * bps),
            ),
            (
                "r marginal sqrt(E[R^2]), bps [DIRECT train-fitted simple return]".to_owned(),
                r_only(simple_second.sqrt() * bps),
            ),
            (
                "r marginal sd, bps [E[R^2]-E[R]^2, DIRECT]".to_owned(),
                r_only((simple_second - simple_mean * simple_mean).max(0.0).sqrt() * bps),
            ),
            (
                "r within-bin contribution to E[R^2], % [lost by squaring E[R|bin]]".to_owned(),
                r_only(100.0 * ratio(simple_second - squared_first, simple_second)),
            ),
        ]);
        rows.push((
            "histogram identification slack used, per-bin mass".to_owned(),
            per_dof(&|dof| self.mass_agreement[dof]),
        ));
        rows
    }

    /// Per-bin series for one DOF, in the order the report writer consumes them.
    pub fn bin_rows(&self, dof: usize) -> Vec<(String, Vec<f64>)> {
        assert_eq!(
            dof, DOF_R,
            "the per-bin support report includes r-only simple-return moments"
        );
        let bins = NUM_BAR_BINS as usize;
        let bps = 1e4;
        let (edge, fitted, mass) = (&self.edge[dof], &self.fitted[dof], &self.mass[dof]);
        let mean_edge = self.edge_census[dof].mean;
        let mean_fitted = self.fitted_census[dof].mean;
        let share = |decode: &[f64], mean: f64, square: bool| -> Vec<f64> {
            let weighted: Vec<f64> = decode
                .iter()
                .zip(mass)
                .map(|(d, p)| {
                    if square {
                        p * (d - mean) * (d - mean)
                    } else {
                        p * d.abs()
                    }
                })
                .collect();
            let total: f64 = weighted.iter().sum();
            weighted.iter().map(|w| 100.0 * ratio(*w, total)).collect()
        };
        vec![
            (
                "bin index".to_owned(),
                (0..bins).map(|bin| bin as f64).collect(),
            ),
            (
                "bin provenance [-1 lower open tail, 0 continuous, 1 upper open tail, 2 atom]"
                    .to_owned(),
                self.r_bin_provenance.clone(),
            ),
            (
                "E[R|bin], bps [DIRECT train-fitted simple return]".to_owned(),
                self.simple_return_mean.iter().map(|m| m * bps).collect(),
            ),
            (
                "sqrt(E[R^2|bin]), bps [DIRECT, includes within-bin dispersion]".to_owned(),
                self.simple_return_second
                    .iter()
                    .map(|second| second.sqrt() * bps)
                    .collect(),
            ),
            (
                "simple-return within-bin sd, bps [DIRECT E[R^2]-E[R]^2]".to_owned(),
                self.simple_return_second
                    .iter()
                    .zip(&self.simple_return_mean)
                    .map(|(second, mean)| (second - mean * mean).max(0.0).sqrt() * bps)
                    .collect(),
            ),
            (
                "E[R|bin] minus expm1(E[r|bin]), bps [nonlinear approximation error]".to_owned(),
                self.simple_return_mean
                    .iter()
                    .zip(fitted)
                    .map(|(simple, log_mean)| (simple - log_mean.exp_m1()) * bps)
                    .collect(),
            ),
            (
                "geometric representative, bps [sampling/CRPS]".to_owned(),
                edge.iter().map(|d| d * bps).collect(),
            ),
            (
                "E[x|bin], bps [PERSISTED train-fitted moment]".to_owned(),
                fitted.iter().map(|d| d * bps).collect(),
            ),
            (
                "fitted minus edge, bps".to_owned(),
                fitted
                    .iter()
                    .zip(edge)
                    .map(|(f, e)| (f - e) * bps)
                    .collect(),
            ),
            (
                "within-bin sd, bps [MEASURED, r's tail exponent ~1.8 so the outer entries are \
                 sample statistics, not population ones]"
                    .to_owned(),
                self.second[dof]
                    .iter()
                    .zip(fitted)
                    .map(|(s, m)| (s - m * m).max(0.0).sqrt() * bps)
                    .collect(),
            ),
            (
                "bin mass, % of the law".to_owned(),
                mass.iter().map(|p| 100.0 * p).collect(),
            ),
            (
                "share of |first moment|, % [geometric representative]".to_owned(),
                share(edge, mean_edge, false),
            ),
            (
                "share of |first moment|, % [persisted fitted E[x|bin]]".to_owned(),
                share(fitted, mean_fitted, false),
            ),
            (
                "share of decoded-mean estimation variance, % [geometric representative]"
                    .to_owned(),
                share(edge, mean_edge, true),
            ),
            (
                "share of decoded-mean estimation variance, % [persisted fitted E[x|bin]]"
                    .to_owned(),
                share(fitted, mean_fitted, true),
            ),
        ]
    }

    /// The lines the operator reads: persisted fitted moments against geometric support
    /// representatives, plus the shares that tell them whether the moments pass is sane.
    pub fn report_lines(&self) -> Vec<String> {
        let bps = 1e4;
        let mut lines = vec![format!(
            "measured on {} rows; histogram identification used at most {:.3e} of per-bin mass \
             slack across the five DOF",
            self.rows,
            self.mass_agreement
                .iter()
                .fold(0.0f64, |worst, x| worst.max(*x))
        )];
        for dof in 0..BAR_DOF {
            let (edge, fitted) = (&self.edge_census[dof], &self.fitted_census[dof]);
            lines.push(format!(
                "DOF {}: open tails hold {:.4}% of the mass and control, under GEOMETRIC \
                 representatives, {:.4}% of |first moment| / {:.4}% of the decoded mean's \
                 estimation variance / {:.4}% of a {:.4} bps reachable span; under PERSISTED \
                 fitted E[x|bin], {:.4}% / {:.4}% / {:.4}% of {:.4} bps. All-bin ceiling \
                 {:.4} -> {:.4} bps, interior ceiling {:.4} -> {:.4} bps.",
                BAR_DOF_NAMES[dof],
                100.0 * (self.catch_alls[dof][0].mass + self.catch_alls[dof][1].mass),
                100.0 * edge.first_share,
                100.0 * edge.second_share,
                100.0 * edge.span_share,
                edge.span * bps,
                100.0 * fitted.first_share,
                100.0 * fitted.second_share,
                100.0 * fitted.span_share,
                fitted.span * bps,
                edge.ceiling * bps,
                fitted.ceiling * bps,
                edge.interior_ceiling * bps,
                fitted.interior_ceiling * bps,
            ));
            for side in &self.catch_alls[dof] {
                let name = if side.bin == 0 { "lower" } else { "upper" };
                lines.push(format!(
                    "DOF {} {name} open-tail bin {} (mass {:.6}%): geometric representative \
                     {:.4} bps, persisted fitted E[x|bin] {:.4} bps; geometry differs by \
                     {:+.4} bps and is {:.4}x as far from zero.",
                    BAR_DOF_NAMES[dof],
                    side.bin,
                    100.0 * side.mass,
                    side.edge * bps,
                    side.fitted * bps,
                    (side.edge - side.fitted) * bps,
                    ratio(side.edge.abs(), side.fitted.abs()),
                ));
            }
        }
        lines
    }
}

/// Measure per-bin conditional moments against an EXISTING support's geometry, persist the
/// upgraded artifact, and emit the fitted-versus-edge decode comparison.
pub fn fit_support_moments(args: SupportMomentsArgs) -> Result<()> {
    ensure!(args.samples > 0, "--samples must be positive");
    ensure!(
        args.mass_tolerance >= 0.0 && args.mass_tolerance.is_finite(),
        "--mass-tolerance must be a finite non-negative probability"
    );
    let source = Path::new(&args.supports);
    let destination = Path::new(&args.output_supports);
    ensure!(
        source != destination,
        "refusing to rewrite {} in place: every checkpoint's supports sidecar is covered by its \
         own supports_sha256 and by lineage_sha256, so an in-place upgrade makes existing \
         checkpoints unloadable against their own training geometry. Name a different \
         --output-supports.",
        source.display()
    );

    let supports = BarSupports::load(source).with_context(|| {
        format!(
            "reading the support to measure moments for, {}",
            source.display()
        )
    })?;
    ensure!(
        supports.num_bins() == NUM_BAR_BINS,
        "{} has {} bins, this build uses {NUM_BAR_BINS}",
        source.display(),
        supports.num_bins()
    );

    let corpus = load_corpus(&args.corpus)?;
    // The provenance is the ONLY record of which corpus and which split the geometry was fitted
    // against, so an absent one is refused rather than worked around: moments measured on a
    // train region that is not the region the bins came from are a measurement of another
    // population, and nothing downstream could tell.
    let provenance = supports.provenance().with_context(|| {
        format!(
            "{} carries no provenance, so the corpus and split its bins were fitted against \
             cannot be verified and the sample this pass draws cannot be identified as the same \
             one. Refit the support with provenance before upgrading it.",
            source.display()
        )
    })?;
    ensure!(
        provenance.split_bounds == corpus.split_bounds(),
        "{} was fitted against split bounds {:?} but this corpus resolves {:?}; the train region \
         the moments would be measured on is not the one the bins came from",
        source.display(),
        provenance.split_bounds,
        corpus.split_bounds()
    );
    ensure!(
        provenance.sample_count == args.samples,
        "{} records a fit sample of {} rows but --samples is {}; the moments must be measured on \
         the SAME draw the masses were, or `bin_means` and `masses` describe different \
         populations",
        source.display(),
        provenance.sample_count,
        args.samples
    );
    println!(
        "measuring per-bin moments for {} against its own geometry: {} rows from the train region \
         of {:?}, seed {} (0x{:X}), corpus fingerprint {}",
        source.display(),
        args.samples,
        provenance.split_bounds,
        args.seed,
        args.seed,
        provenance.corpus_fingerprint,
    );

    // The SAME draw `fit_supports` made, by construction: same accessor, same budget, same seed.
    let samples: Vec<BarDof> = corpus
        .sample_train_dof(args.samples, args.seed)
        .into_iter()
        .map(|(_, dof)| dof)
        .collect();
    ensure!(
        !samples.is_empty(),
        "the train region yielded no DOF rows, so there is nothing to measure"
    );
    let rows = samples.len();

    let (upgraded, mass_agreement) = supports
        .with_verified_bin_moments(&samples, args.mass_tolerance)
        .with_context(|| {
            format!(
                "the redrawn sample does not reproduce {}'s own histogram, so per-bin moments \
                 measured on it would not describe the population its masses do",
                source.display()
            )
        })?;
    drop(samples);
    println!(
        "histogram identified: the redraw of {rows} rows reproduces the persisted masses to \
         {:.3e} per bin, worst over all {BAR_DOF} DOF",
        mass_agreement.iter().fold(0.0f64, |w, x| w.max(*x))
    );

    upgraded
        .save(destination)
        .with_context(|| format!("writing the upgraded support to {}", destination.display()))?;
    let differing = changed_members(source, destination)?;
    ensure!(
        differing.is_empty(),
        "the upgraded support at {} differs from {} in {:?}, which it must not: only the \
         schema version and fitted log/simple-return moment members may change, because the \
         bin edges define the `nll_bar` scale and every persisted report uses that geometry",
        destination.display(),
        source.display(),
        differing
    );
    println!(
        "wrote {} — every other JSON member is value-identical to the source, so the geometry, \
         the histogram and the smoothed marginal are unchanged and `nll_bar` stays on its scale",
        destination.display()
    );

    let decode = SupportDecode::of(&upgraded, mass_agreement, rows)?;
    for line in decode.report_lines() {
        println!("{line}");
    }
    write_support_decode(Path::new(&args.output), &decode)?;
    println!("support decode comparison written to {}", args.output);
    Ok(())
}

/// Names of JSON members that differ between two support files, excluding the schema version
/// and four fitted-moment members this upgrade is allowed to touch.
///
/// Compared as parsed JSON rather than as text so key order and whitespace cannot register as a
/// geometry change, and member-wise over the UNION of both key sets so a member this build does
/// not know about is still checked and a DISAPPEARED member is caught too. This is the
/// mechanical form of "preserving the bin edges exactly is REQUIRED": the requirement is not
/// that the reader agrees, it is that the FILE agrees.
fn changed_members(source: &Path, destination: &Path) -> Result<Vec<String>> {
    let read = |path: &Path| -> Result<serde_json::Map<String, serde_json::Value>> {
        let body = std::fs::read(path).with_context(|| format!("re-reading {}", path.display()))?;
        let value: serde_json::Value = serde_json::from_slice(&body)
            .with_context(|| format!("re-parsing {}", path.display()))?;
        value
            .as_object()
            .cloned()
            .with_context(|| format!("{} is not a JSON object", path.display()))
    };
    Ok(compare_members(&read(source)?, &read(destination)?))
}

/// [`changed_members`] on two already-parsed objects.
fn compare_members(
    before: &serde_json::Map<String, serde_json::Value>,
    after: &serde_json::Map<String, serde_json::Value>,
) -> Vec<String> {
    const MAY_CHANGE: [&str; 5] = [
        "format_version",
        "bin_means",
        "bin_second_moments",
        "bin_simple_return_means",
        "bin_simple_return_second_moments",
    ];
    let mut differing: Vec<String> = Vec::new();
    for key in before.keys().chain(after.keys()) {
        if MAY_CHANGE.contains(&key.as_str()) || differing.iter().any(|seen| seen == key) {
            continue;
        }
        if before.get(key) != after.get(key) {
            differing.push(key.clone());
        }
    }
    differing
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared::report::{read_report, ReportKind};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::{fs, path::PathBuf};

    static SCRATCH: AtomicU64 = AtomicU64::new(0);

    fn scratch_dir(name: &str) -> PathBuf {
        let unique = SCRATCH.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "support_moments_{name}_{}_{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn object(body: &str) -> serde_json::Map<String, serde_json::Value> {
        serde_json::from_str(body).expect("fixture parses")
    }

    /// The upgrade's whole licence is that it changes NOTHING but the moment members and version.
    #[test]
    fn only_the_version_and_fitted_moment_rows_may_change() {
        let before = object(
            r#"{"format_version":4,"lo":[[1.0,2.0]],"hi":[[2.0,3.0]],"masses":[[0.5,0.5]],
                "smoothed_marginal":[[0.5,0.5]],"provenance":{"sample_count":7},
                "a_member_this_build_does_not_know":[1,2,3]}"#,
        );

        let upgraded = object(
            r#"{"format_version":6,"lo":[[1.0,2.0]],"hi":[[2.0,3.0]],"masses":[[0.5,0.5]],
                "smoothed_marginal":[[0.5,0.5]],"provenance":{"sample_count":7},
                "a_member_this_build_does_not_know":[1,2,3],
                "bin_means":[[1.5,2.5]],"bin_second_moments":[[2.3,6.3]],
                "bin_simple_return_means":[1.6,2.7],
                "bin_simple_return_second_moments":[2.7,7.5]}"#,
        );
        assert!(
            compare_members(&before, &upgraded).is_empty(),
            "a legitimate geometry-preserving upgrade must register no change"
        );

        // The failure this exists to catch: a refit that moved one edge by one ulp. Nothing
        // else in the artifact would look different, and every persisted `nll_bar` would be on
        // a different scale.
        for (what, body) in [
            (
                "lo",
                r#"{"format_version":5,"lo":[[1.0,2.0000000000000004]],"hi":[[2.0,3.0]],
                    "masses":[[0.5,0.5]],"smoothed_marginal":[[0.5,0.5]],
                    "provenance":{"sample_count":7},
                    "a_member_this_build_does_not_know":[1,2,3],
                    "bin_means":[[1.5,2.5]],"bin_second_moments":[[2.3,6.3]]}"#,
            ),
            (
                "masses",
                r#"{"format_version":5,"lo":[[1.0,2.0]],"hi":[[2.0,3.0]],
                    "masses":[[0.4,0.6]],"smoothed_marginal":[[0.5,0.5]],
                    "provenance":{"sample_count":7},
                    "a_member_this_build_does_not_know":[1,2,3],
                    "bin_means":[[1.5,2.5]],"bin_second_moments":[[2.3,6.3]]}"#,
            ),
            (
                "provenance",
                r#"{"format_version":5,"lo":[[1.0,2.0]],"hi":[[2.0,3.0]],
                    "masses":[[0.5,0.5]],"smoothed_marginal":[[0.5,0.5]],
                    "provenance":{"sample_count":8},
                    "a_member_this_build_does_not_know":[1,2,3],
                    "bin_means":[[1.5,2.5]],"bin_second_moments":[[2.3,6.3]]}"#,
            ),
            (
                "a_member_this_build_does_not_know",
                r#"{"format_version":5,"lo":[[1.0,2.0]],"hi":[[2.0,3.0]],
                    "masses":[[0.5,0.5]],"smoothed_marginal":[[0.5,0.5]],
                    "provenance":{"sample_count":7},
                    "a_member_this_build_does_not_know":[1,2,4],
                    "bin_means":[[1.5,2.5]],"bin_second_moments":[[2.3,6.3]]}"#,
            ),
        ] {
            assert_eq!(
                compare_members(&before, &object(body)),
                vec![what.to_owned()],
                "a changed `{what}` must be reported"
            );
        }

        // And a member that VANISHES, which a key-by-key walk of only the source would catch
        // but a walk of only the destination would not.
        let stripped = object(
            r#"{"format_version":5,"lo":[[1.0,2.0]],"hi":[[2.0,3.0]],"masses":[[0.5,0.5]],
                "smoothed_marginal":[[0.5,0.5]],"provenance":{"sample_count":7},
                "bin_means":[[1.5,2.5]],"bin_second_moments":[[2.3,6.3]]}"#,
        );
        assert_eq!(
            compare_members(&before, &stripped),
            vec!["a_member_this_build_does_not_know".to_owned()]
        );
    }

    /// The census must reproduce the arithmetic it claims, on numbers small enough to check by
    /// hand, and must report an undefined share as ABSENT rather than as zero.
    #[test]
    fn the_decode_census_is_the_arithmetic_it_claims() {
        // Four bins: two catch-alls at +/-10 holding 1% each, two interior at +/-1 holding 49%.
        let decode = [-10.0, -1.0, 1.0, 10.0];
        let mass = [0.01, 0.49, 0.49, 0.01];
        let census = DecodeCensus::measure(&decode, &mass);
        assert!(census.mean.abs() < 1e-15, "a symmetric law has zero mean");

        // sum p|d| = 0.1 + 0.49 + 0.49 + 0.1 = 1.18, outer = 0.2.
        assert!((census.first_share - 0.2 / 1.18).abs() < 1e-12);
        // sum p d^2 = 1.0 + 0.49 + 0.49 + 1.0 = 2.98, outer = 2.0.
        assert!((census.second_share - 2.0 / 2.98).abs() < 1e-12);
        // Span 20 all-bin against 2 interior.
        assert!((census.span - 20.0).abs() < 1e-12);
        assert!((census.span_share - 0.9).abs() < 1e-12);
        assert!((census.ceiling - 10.0).abs() < 1e-12);
        assert!((census.interior_ceiling - 1.0).abs() < 1e-12);

        // THE IDENTITY THE SECOND-MOMENT SHARE RESTS ON, checked rather than asserted in a
        // doc comment: for the plug-in decoded mean over N iid rows the multinomial counts give
        // Var(mu_hat) = (1/N) sum_j p_j (d_j - mu)^2, so `second_share` is literally the share
        // of that variance sitting in the two catch-alls. Recomputed here through the
        // multinomial covariance form, which is a different expression of the same quantity.
        let mean = census.mean;
        let direct: f64 = (0..4)
            .map(|j| mass[j] * (decode[j] - mean) * (decode[j] - mean))
            .sum();
        let via_covariance: f64 = {
            let second: f64 = (0..4).map(|j| mass[j] * decode[j] * decode[j]).sum();
            second - mean * mean
        };
        assert!(
            (direct - via_covariance).abs() < 1e-12,
            "the centred sum and E[d^2] - mu^2 must agree; they are the same variance"
        );

        // A law with no first moment at all has an UNDEFINED outer share, not a zero one.
        let degenerate = DecodeCensus::measure(&[0.0, 0.0, 0.0, 0.0], &mass);
        assert!(degenerate.first_share.is_nan());
        assert!(degenerate.second_share.is_nan());
        assert!(degenerate.span_share.is_nan());
    }

    /// A support fitted in memory, carrying the same full moment set the upgrade writes.
    fn synthetic_supports(count: usize, seed: u64) -> BarSupports {
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
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

    /// Executes the writer for both bases owned by `PretrainReportOwner::SupportDecode`.
    /// They need measured support moments plus an upgraded artifact on disk, neither of which
    /// exists in an optimizer-step report cycle.
    #[test]
    fn the_support_decode_writes_both_registered_bases() {
        let rows = 40_000usize;
        let supports = synthetic_supports(rows, 0x5D0F_0001);
        assert!(
            supports.bin_means_measured(),
            "a freshly fitted support must carry the moments the census reads"
        );
        let decode = SupportDecode::of(&supports, [1e-9; BAR_DOF], rows).expect("the census runs");

        let dir = scratch_dir("decode");
        write_support_decode(&dir, &decode).expect("both charts write");
        for base in ["support_decode_moments", "support_decode_bins"] {
            assert_eq!(
                shared::report::pretrain_report_owner(base),
                Some(shared::report::PretrainReportOwner::SupportDecode),
                "{base} must remain owned by the support-decode writer"
            );
            let path = dir.join(format!("{base}.report.bin"));
            assert!(path.exists(), "{base} was not written");
            let read = read_report(&path).expect("the report reads back");
            let ReportKind::MultiLine { series } = &read.kind else {
                panic!("{base} must be a MultiLine chart");
            };
            assert!(
                series
                    .iter()
                    .any(|s| s.values.iter().any(|v| v.is_finite())),
                "{base} carries no finite value, so it is a blank panel"
            );
            if base == "support_decode_bins" {
                for required in [
                    "E[R|bin], bps [DIRECT train-fitted simple return]",
                    "sqrt(E[R^2|bin]), bps [DIRECT, includes within-bin dispersion]",
                    "simple-return within-bin sd, bps [DIRECT E[R^2]-E[R]^2]",
                    "bin provenance [-1 lower open tail, 0 continuous, 1 upper open tail, 2 atom]",
                ] {
                    assert!(
                        series.iter().any(|row| row.label == required),
                        "{base} is missing `{required}`"
                    );
                }
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
