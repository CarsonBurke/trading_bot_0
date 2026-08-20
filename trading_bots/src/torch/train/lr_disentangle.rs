//! Why this recipe cannot separate "another pass over the corpus" from "a lower learning rate",
//! and exactly how much identifying variation it does contain.
//!
//! # The two axes are one axis
//!
//! Between two checkpoints of one pretraining run at the same ramp stage, context, batch and
//! conditioning depth, only two things move: the number of PASSES the optimizer has taken over
//! the corpus, and the LEARNING-RATE multiplier. Both are monotone in the step index, so pooled
//! they are confounded — that much is obvious and is why this module exists. What is not obvious,
//! and is the whole result, is that past the learning-rate plateau they are not merely correlated
//! but EXACTLY AFFINELY DEPENDENT:
//!
//! ```text
//! passes(step)  = step / steps_per_epoch
//! lr_mult(step) = P - (P - LR_FLOOR_MULTIPLIER) * (step/total_steps - F) / (1 - F)
//! ```
//!
//! with `F = LR_PLATEAU_FRACTION` and `P` the stage's batch bump. Both are affine in `step`, so
//! their ratio is a CONSTANT, and because `total_steps = epochs * steps_per_epoch` the corpus
//! cancels out of it entirely:
//!
//! ```text
//! d(passes)/d(lr_mult) = -epochs * (1 - F) / (P - LR_FLOOR_MULTIPLIER)
//! ```
//!
//! A pure function of the recipe — independent of corpus size, of `steps_per_epoch`, and of the
//! batch. For the run this was written for that is `-3 * 0.6 / 0.85 = -36/17 = -2.1176...`.
//!
//! The consequence is a design fact, not a precision problem. EVERY pair of checkpoints past the
//! plateau moves along ONE direction in `(passes, lr_mult)` space. The design is rank 1 there. A
//! tightly-spaced triple of step-cadence checkpoints looks like a local learning-rate slice "at
//! essentially fixed passes", and it is not: its passes move too, in exactly that fixed ratio. It
//! measures a single scalar `d(beta)/d(step)`. Reading that as `d(beta)/d(lr_mult)` requires
//! assuming the passes coefficient is zero, which is the very hypothesis such a measurement is
//! convened to test. A measurement cannot supply its own identifying assumption.
//!
//! # Where the identification actually lives
//!
//! In exactly one place: the PLATEAU CLIP. For `step/total_steps <= F` the multiplier is pinned
//! flat at `P`, so across that stretch passes accumulate at ZERO learning-rate contrast. It spans
//! `epochs * F` passes — 1.2 passes for this recipe. Any two checkpoints inside it would identify
//! the passes coefficient outright. A run that retains none, or one, has zero such pairs, and the
//! decomposition then rests entirely on how far the single plateau checkpoint sits BELOW the
//! plateau's end: that overhang is the only stretch of the whole run over which the two axes move
//! at different relative rates.
//!
//! [`Attribution`] measures that overhang and reports it as `identified_passes`. The arithmetic
//! below is exact, and it is what turns "we lacked precision" into "the design has rank 1 and the
//! identified window is 0.2 of a pass against a 1.9-pass contrast".
//!
//! # What this module does NOT do
//!
//! It does not fit anything. Every slope it consumes is produced by
//! [`super::trade_bench::mincer_zarnowitz`] through [`super::trade_bench::mean_calibration`], on a
//! pinned population, with that estimator's own `(symbol, calendar month)` block bootstrap. This
//! module is arithmetic on those slopes and on the schedule, and its own uncertainty input is a
//! DETERMINISTIC one — see [`JitterFloor`].

use anyhow::{Result, ensure};

use super::pretrain::Schedule;

/// Checkpoints one analysis will accept. A bound rather than a `Vec` because every quantity here
/// is `O(n^3)` in distinct steps and because an unbounded caller is how an analysis pass turns
/// into a resource incident.
pub const MAX_CHECKPOINTS: usize = 64;

/// Points on the assumed-local-movement sweep of [`Disentangle::grid`].
///
/// Odd, so the sweep passes exactly through zero: "the local movement was indistinguishable from
/// nothing" is the case the whole chart exists to make visible, and it must be a drawn point
/// rather than an interpolation between two others.
pub const GRID_POINTS: usize = 41;

/// Half-width of the sweep, in units of the larger of the observed local movement and the jitter
/// floor. Three is enough to show the amplified band crossing the entire spine movement without
/// compressing the interesting middle into one pixel.
const GRID_SPAN_MULTIPLE: f64 = 3.0;

/// One checkpoint placed on both axes, beside the slopes measured at it.
///
/// `beta_*_blocks` and `beta_*_samples` are carried deliberately and are not decoration: they are
/// the observable precondition for treating two checkpoints' bootstraps as PAIRED.
/// `mincer_zarnowitz` drops rows whose `x` is non-finite, and `x` is the predicted mean, which
/// differs per checkpoint — so a single non-finite forecast silently changes the surviving block
/// set and re-keys every bootstrap draw for that checkpoint alone, with no error raised. Equal
/// `blocks` AND equal `samples` across checkpoints is what rules that out.
#[derive(Clone, Debug)]
pub struct CheckpointAxes {
    pub label: String,
    pub step: usize,
    /// `step / total_steps`, the argument the learning-rate schedule is written in.
    pub progress: f64,
    pub lr_mult: f64,
    /// Passes over the training corpus, fractional mid-pass.
    pub passes: f64,
    pub in_lr_plateau: bool,
    /// Mincer-Zarnowitz mean slope on the TRADED prefix.
    pub beta_traded: f64,
    /// Mincer-Zarnowitz mean slope on the block-disjoint FIT SLICE. Named separately because a
    /// slope without its population is not a slope, and these two are different numbers.
    pub beta_fit: f64,
    pub beta_fit_ci: (f64, f64),
    pub beta_fit_blocks: usize,
    pub beta_fit_samples: usize,
}

/// Which population a decomposition was computed on. Both are measured; neither is a default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlopePopulation {
    /// The first [`super::trade_bench::TRADE_WINDOWS`] pinned windows — the ones the bench trades.
    TradedPrefix,
    /// Windows whose `(symbol, calendar month)` blocks are absent from the traded prefix.
    FitSlice,
}

impl SlopePopulation {
    pub fn label(self) -> &'static str {
        match self {
            Self::TradedPrefix => "traded prefix",
            Self::FitSlice => "fit slice",
        }
    }

    fn beta(self, axes: &CheckpointAxes) -> f64 {
        match self {
            Self::TradedPrefix => axes.beta_traded,
            Self::FitSlice => axes.beta_fit,
        }
    }
}

/// A movement between two checkpoints, on both axes at once.
#[derive(Clone, Copy, Debug)]
pub struct Contrast {
    pub lo_step: usize,
    pub hi_step: usize,
    pub d_passes: f64,
    pub d_lr: f64,
    pub d_beta: f64,
}

impl Contrast {
    fn between(lo: &CheckpointAxes, hi: &CheckpointAxes, population: SlopePopulation) -> Self {
        Self {
            lo_step: lo.step,
            hi_step: hi.step,
            d_passes: hi.passes - lo.passes,
            d_lr: hi.lr_mult - lo.lr_mult,
            d_beta: population.beta(hi) - population.beta(lo),
        }
    }

    /// Passes per unit of learning-rate multiplier along this movement. Equal to
    /// [`Schedule::passes_per_lr_unit`] for any contrast wholly past the plateau, which is the
    /// rank-1 statement in observable form.
    pub fn passes_per_lr(&self) -> f64 {
        self.d_passes / self.d_lr
    }
}

/// How much slope movement is MEANINGLESS.
///
/// The primary source is `disagreement`. The step-cadence artifacts are equally spaced by
/// construction — `write_step_artifacts` fires on `step % checkpoint_every == 0` and formats the
/// filename from the loop's own step — so consecutive differences along that slice are repeated
/// estimates of the SAME local derivative, and their disagreement is what "meaningless" means at
/// that spacing.
///
/// This is NOT a sampling interval and does not need one. The pinned evaluation population is
/// FIXED, so a slope is a deterministic functional of `(weights, population)`; the spread among
/// these numbers is a descriptive fact about that functional. It requires no independence
/// assumption, no block key, and is unmoved by re-clustering the bars — which is exactly why it
/// is reported separately from anything that does need an interval.
#[derive(Clone, Debug)]
pub struct JitterFloor {
    /// Steps of the equally-spaced cadence slice, ascending.
    pub cadence_steps: Vec<usize>,
    /// Step spacing of that slice.
    pub cadence_spacing: usize,
    /// Consecutive slope differences along it.
    pub consecutive: Vec<f64>,
    /// Largest gap between any two of `consecutive`. Zero when fewer than two exist.
    pub disagreement: f64,
    /// Largest minus smallest slope over the cadence slice.
    pub span: f64,
    /// Slope difference between two artifacts recording the SAME step, if any.
    ///
    /// A NULL CONTROL, not a jitter datum, and it must read EXACTLY `0.0`.
    ///
    /// It is tempting as "two promotion races at zero passes and zero learning-rate contrast", and
    /// that reading is wrong twice over. `Trainer::write_checkpoint` serializes the LIVE
    /// `VarStore`; there is no retained best-so-far weight buffer anywhere in the promotion path,
    /// so a promoted artifact holds the weights of the step its promotion fired at. Two criteria
    /// that promote at one step therefore write THE SAME WEIGHTS TWICE. Their differing
    /// `checkpoint_sha256` is libtorch writing the FILE STEM as the internal zip archive name and
    /// says nothing about the parameters — confirmed by comparing the multiset of zip-record
    /// CRC-32s, which is identical across `pretrain_best`, `pretrain_best_diag896` and
    /// `pretrain_last` and differs from `pretrain_step_30720`.
    ///
    /// So this difference is a tautology about the weights and a real test of everything else: a
    /// nonzero value means the evaluation path is NONDETERMINISTIC — same weights, same pinned
    /// population, same block partition, different slope — which is a defect to report, not a
    /// floor to widen a band with. Nothing else in this analysis exercises that end to end.
    ///
    /// `NaN` when the run retained no such pair: a zero would read as a passed control, and not
    /// running a control is not passing it.
    pub null_control: f64,
    pub null_control_labels: Option<(String, String)>,
}

impl JitterFloor {
    /// The floor a movement has to clear to mean anything. Cadence-derived only.
    ///
    /// `null_control` is excluded because it is identically zero when the harness is correct, so
    /// including it could only ever widen the band by importing a bug.
    pub fn worst(&self) -> f64 {
        self.disagreement.max(self.span)
    }

    /// True when the null control ran AND the evaluation path proved deterministic on it.
    /// `false` when it ran and did not; `None` when the run retained no same-step pair.
    pub fn determinism_verified(&self) -> Option<bool> {
        self.null_control_labels
            .as_ref()
            .map(|_| self.null_control == 0.0)
    }
}

/// The attribution of a spine movement into a learning-rate part and a passes part, and the exact
/// reason the split is or is not supportable.
///
/// The procedure is the obvious one: measure a LOCAL learning-rate sensitivity on a tight slice,
/// extrapolate it across the spine's much larger learning-rate change, and call the residual the
/// passes effect. Every constant below quantifies what that procedure actually returns.
///
/// Writing the true movement as `d_beta = a * d_passes + b * d_lr`, the residual
/// `d_beta_spine - extrapolation_factor * d_beta_local` equals `a * identified_passes` EXACTLY —
/// the `b` terms cancel algebraically, and so does most of `a`. So the residual is the passes
/// effect over `identified_passes` of a pass, not over the spine's `d_passes`. Recovering the
/// spine's passes effect multiplies it by `attenuation`, and multiplies the local movement's error
/// by `amplification = attenuation * extrapolation_factor`.
#[derive(Clone, Debug)]
pub struct Attribution {
    pub population: SlopePopulation,
    pub spine: Contrast,
    pub local: Contrast,
    /// `d_lr_spine / d_lr_local`. How far the local sensitivity is carried beyond where it was
    /// measured.
    pub extrapolation_factor: f64,
    /// `d_passes_spine - d_passes_local * extrapolation_factor`, in passes. The ONLY stretch of
    /// the run over which the two axes move at different relative rates, and therefore the only
    /// passes variation this design identifies. Equals the plateau overhang of the spine's lower
    /// checkpoint whenever that checkpoint is the run's only plateau artifact.
    pub identified_passes: f64,
    /// `d_passes_spine / identified_passes`.
    pub attenuation: f64,
    /// `attenuation * extrapolation_factor`. The factor by which an error in the local movement
    /// is multiplied on its way into the passes component.
    pub amplification: f64,
    /// `d_beta_local / d_lr_local`, the local sensitivity, valid only under the assumption the
    /// design cannot test.
    pub local_lr_sensitivity: f64,
    /// `attenuation * (d_beta_spine - extrapolation_factor * d_beta_local)`.
    pub passes_component: f64,
    /// `d_beta_spine - passes_component`. The two sum to the spine movement by construction.
    pub lr_component: f64,
    /// Half-width the jitter floor puts on `passes_component`, and on `lr_component` equally and
    /// oppositely: `amplification * jitter`.
    pub half_width: f64,
    /// Jitter floor the half-width was taken from.
    pub jitter: f64,
    /// True only when the amplified band is narrower than the movement being decomposed. False
    /// means the split is unresolvable and the components must not be quoted as a decomposition.
    pub resolved: bool,
}

impl Attribution {
    /// The caveat this attribution must be drawn with.
    ///
    /// Returned as text destined for SERIES LABELS rather than a chart title: the renderer's
    /// `normalize_title` lowercases everything after each word's first character, so emphasis in a
    /// title is destroyed before a reader sees it, while legend labels are drawn verbatim. A
    /// qualification that does not survive rendering is not a qualification.
    pub fn caveat(&self) -> &'static str {
        if self.resolved {
            "rank-1 design, split rests on the plateau overhang only"
        } else {
            "UNRESOLVED: rank-1 design, amplified jitter exceeds the movement"
        }
    }
}

/// One point of the assumed-local-movement sweep.
///
/// The sweep exists because the amplification is the finding. Quoting a single decomposition hides
/// it; drawing the components against the assumed local movement makes it a visible slope, and a
/// reader can see for themselves that stepping one jitter-width along the axis swings the passes
/// component across the whole spine movement.
#[derive(Clone, Copy, Debug)]
pub struct AttributionAtAssumed {
    pub assumed_d_beta_local: f64,
    pub passes_component: f64,
    pub lr_component: f64,
}

/// The one thing this recipe CAN identify that the linear decomposition cannot: the KINK in the
/// slope trend at a pass boundary.
///
/// The rank-1 result is about two LINEAR coefficients, one on the step index and one on the
/// learning-rate multiplier, and past the plateau those are inseparable. It says nothing about a
/// NONLINEARITY, and repetition is one: no bar is seen twice before the first pass boundary at
/// `steps_per_epoch`, and after it some are. Passes and steps are the same variable — `passes =
/// step / steps_per_epoch` — so there is no passes-versus-duration contrast to identify. What
/// there is, is a switch that flips at one known step.
///
/// Inside the plateau the multiplier is clipped exactly flat, `Schedule::stage_at` is
/// `step % steps_per_epoch` so the ramp restarts identically, the realized batch is flat, and
/// `PassPlan::counts` is computed once in `PassPlan::new` — which takes no epoch — so stage
/// composition is byte-identical across the boundary. NOTHING ELSE IN THE RECIPE CHANGES THERE.
/// So a kink measured from anchors that all sit inside the plateau is repetition and nothing else:
/// a regression discontinuity at a pre-registered step, needing no extrapolation, where the linear
/// decomposition needs a 17x one.
///
/// This is why anchors at the plateau's two ENDS are the wrong placement. The plateau spans
/// `epochs * LR_PLATEAU_FRACTION` passes — 1.2 here — of which only 0.2 carries any repetition, so
/// an end-to-end pair measures a training-duration coefficient that mixes "still learning" with
/// "started seeing bars twice", and its low anchor is near a random init.
#[derive(Clone, Debug)]
pub struct RepetitionKink {
    /// Step the first repetition begins at: one `steps_per_epoch`.
    pub boundary_step: usize,
    /// `d(beta)/d(step)` from plateau anchors strictly BELOW the boundary — the all-fresh trend.
    pub before: f64,
    /// `d(beta)/d(step)` from the plateau pair STRADDLING the boundary.
    pub after: f64,
    /// `after - before`. Repetition, at zero learning-rate contrast, with no extrapolation.
    pub kink: f64,
    /// Anchor steps the measurement used, ascending. Empty when unavailable.
    pub anchors: Vec<usize>,
    /// Anchors a future run must RETAIN to make this measurable, snapped to the checkpoint
    /// cadence and all inside the plateau. Emitted whether or not the kink is available, so the
    /// next run's retention policy is read off an artifact rather than off a writeup.
    pub required_anchors: [usize; 3],
}

impl RepetitionKink {
    /// `NaN` throughout rather than zero, because "no anchors straddle the boundary" is an absence
    /// of measurement and a zero kink would be a finding.
    fn unavailable(boundary_step: usize, required_anchors: [usize; 3]) -> Self {
        Self {
            boundary_step,
            before: f64::NAN,
            after: f64::NAN,
            kink: f64::NAN,
            anchors: Vec::new(),
            required_anchors,
        }
    }

    pub fn available(&self) -> bool {
        self.kink.is_finite()
    }
}

/// Anchor steps that would identify the repetition discontinuity, snapped to `cadence` so they
/// land on steps the checkpointer already writes.
///
/// `A2` is the last cadence step at or below the boundary and `A3` the first above it, so the
/// straddling pair is as tight as the cadence permits — the discontinuity matters more than the
/// lever arm, because a kink at a pre-registered step needs no extrapolation. `A1` sits half way
/// back from `A2` toward zero: far enough from a random init to be a trend, still all-fresh.
fn required_plateau_anchors(
    steps_per_epoch: usize,
    plateau_last_step: usize,
    cadence: usize,
) -> [usize; 3] {
    let cadence = cadence.max(1);
    let snap_down = |step: usize| step / cadence * cadence;
    let a2 = snap_down(steps_per_epoch.min(plateau_last_step));
    let a3 = (a2 + cadence).min(snap_down(plateau_last_step));
    let a1 = snap_down(a2 / 2);
    [a1, a2, a3]
}

/// Everything the disentanglement produces, for one population.
#[derive(Clone, Debug)]
pub struct Disentangle {
    pub axes: Vec<CheckpointAxes>,
    pub population: SlopePopulation,
    /// [`Schedule::passes_per_lr_unit`]: the exact rank-1 constant of the recipe.
    pub passes_per_lr_unit: f64,
    /// Last step still on the flat learning-rate plateau.
    pub plateau_last_step: usize,
    /// That step in passes — the length of the only zero-learning-rate-contrast stretch.
    pub plateau_passes: f64,
    /// Checkpoints retained inside the plateau. Two or more would identify the passes coefficient
    /// outright; this is the number that decides whether the question is answerable at all.
    pub plateau_checkpoints: usize,
    pub spine: Contrast,
    pub local: Contrast,
    pub jitter: JitterFloor,
    pub attribution: Attribution,
    pub grid: Vec<AttributionAtAssumed>,
    /// The repetition discontinuity at the first pass boundary — identifiable in principle even
    /// where the linear split is not, and unavailable on a run that retained too few plateau
    /// artifacts to see it.
    pub kink: RepetitionKink,
    /// Whether every checkpoint's fit-slice bootstrap ran over the same block set and row count,
    /// which is the precondition for treating their intervals as paired.
    pub pairing_precondition_holds: bool,
}

/// Place a run's checkpoints on both axes and decompose the epoch spine.
///
/// `total_steps` and `steps_per_epoch` are STATED rather than read: neither is recorded in a
/// checkpoint's metadata sidecar, the same reason the trend's x-axis is stated as `path@step`. The
/// per-stage batch MULTIPLIERS are derived from what IS recorded — `training.batch_ramp` holds the
/// realized batch per stage, and stage 0's declared multiplier is `1` and a memory hold can only
/// lower a stage in place, so `multiplier[s] = realized[s] / realized[0]` exactly.
pub(super) fn disentangle(
    axes: Vec<CheckpointAxes>,
    schedule: &Schedule,
    population: SlopePopulation,
) -> Result<Disentangle> {
    ensure!(
        axes.len() >= 4 && axes.len() <= MAX_CHECKPOINTS,
        "the disentanglement needs at least four checkpoints — one on the learning-rate plateau \
         and an equally-spaced cadence slice of at least three past it — and refuses more than \
         {MAX_CHECKPOINTS}; got {}",
        axes.len()
    );
    ensure!(
        schedule.lr_affine_in_step(),
        "the ramp stages carry different batch bumps, so the learning-rate multiplier is only \
         piecewise-affine in the step index and the passes-per-learning-rate constant is not one \
         number; this analysis does not apply to that schedule"
    );

    let cadence = equally_spaced_cadence(&axes, schedule)?;
    let local_lo = &axes[*cadence.first().expect("cadence is non-empty")];
    let local_hi = &axes[*cadence.last().expect("cadence is non-empty")];
    let local = Contrast::between(local_lo, local_hi, population);

    // The spine's low end is the plateau checkpoint: the only one whose learning rate is CLIPPED,
    // and therefore the only one that gives a contrast pointing anywhere other than along the
    // rank-1 direction. Its high end is the widest checkpoint that is NOT part of the cadence
    // slice, so the spine and the local slice do not share an endpoint and their errors do not
    // share a term.
    let in_cadence = |index: usize| cadence.contains(&index);
    let spine_lo = axes
        .iter()
        .enumerate()
        .filter(|(_, axis)| axis.in_lr_plateau)
        .max_by_key(|(_, axis)| axis.step)
        .map(|(index, _)| index);
    let spine_hi = axes
        .iter()
        .enumerate()
        .filter(|(index, _)| !in_cadence(*index))
        .max_by_key(|(_, axis)| axis.step)
        .map(|(index, _)| index);
    let (spine_lo, spine_hi) = match (spine_lo, spine_hi) {
        (Some(lo), Some(hi)) if axes[lo].step < axes[hi].step => (lo, hi),
        _ => anyhow::bail!(
            "no checkpoint sits on the learning-rate plateau (steps 0..={}), so the run contains \
             NO contrast off the rank-1 direction and the two axes are not separable at any \
             precision; retain checkpoints inside the plateau to make this measurable",
            schedule.plateau_last_step()
        ),
    };
    let spine = Contrast::between(&axes[spine_lo], &axes[spine_hi], population);

    let jitter = jitter_floor(&axes, &cadence, population);
    let attribution = attribute(population, spine, local, jitter.worst());
    let grid = sweep(&attribution);

    let plateau_checkpoints = axes.iter().filter(|axis| axis.in_lr_plateau).count();
    let pairing_precondition_holds = axes
        .windows(2)
        .all(|pair| {
            pair[0].beta_fit_blocks == pair[1].beta_fit_blocks
                && pair[0].beta_fit_samples == pair[1].beta_fit_samples
        });
    let kink = repetition_kink(&axes, schedule, population, jitter.cadence_spacing);

    Ok(Disentangle {
        axes,
        population,
        passes_per_lr_unit: schedule.passes_per_lr_unit(),
        plateau_last_step: schedule.plateau_last_step(),
        plateau_passes: schedule.passes_at(schedule.plateau_last_step()),
        plateau_checkpoints,
        spine,
        local,
        jitter,
        attribution,
        grid,
        kink,
        pairing_precondition_holds,
    })
}

/// The longest run of EQUALLY SPACED distinct steps past the plateau, as indices into `axes`.
///
/// Equal spacing is what identifies the step-cadence artifacts — `write_step_artifacts` fires on
/// `step % checkpoint_every == 0`, so those and only those are evenly spaced — without matching on
/// a filename. A run that retains more of them lengthens this slice and tightens the jitter floor
/// for free.
fn equally_spaced_cadence(axes: &[CheckpointAxes], schedule: &Schedule) -> Result<Vec<usize>> {
    let mut candidates: Vec<usize> = (0..axes.len())
        .filter(|index| !axes[*index].in_lr_plateau)
        .collect();
    candidates.sort_by_key(|index| axes[*index].step);
    candidates.dedup_by_key(|index| axes[*index].step);
    ensure!(
        candidates.len() >= 3,
        "only {} checkpoints with distinct steps sit past the learning-rate plateau; a jitter \
         floor needs three so that two consecutive differences can disagree",
        candidates.len()
    );

    let mut best: Vec<usize> = Vec::new();
    for start in 0..candidates.len() {
        for next in (start + 1)..candidates.len() {
            let spacing = axes[candidates[next]].step - axes[candidates[start]].step;
            let mut run = vec![candidates[start], candidates[next]];
            let mut cursor = next;
            while let Some(found) = candidates[(cursor + 1)..].iter().find(|index| {
                axes[**index].step == axes[candidates[cursor]].step + spacing
            }) {
                cursor = candidates.iter().position(|index| index == found).expect("member");
                run.push(candidates[cursor]);
            }
            if run.len() > best.len() {
                best = run;
            }
        }
    }
    ensure!(
        best.len() >= 3,
        "no three checkpoints past the plateau are equally spaced, so there is no step-cadence \
         slice to read a jitter floor off; retained steps are {:?}",
        candidates
            .iter()
            .map(|index| axes[*index].step)
            .collect::<Vec<_>>()
    );
    let _ = schedule;
    Ok(best)
}

/// The slope's kink at the first pass boundary, measured from plateau anchors only.
///
/// Every anchor must be INSIDE the plateau, because that is what makes the two derivatives
/// comparable: outside it the learning rate moves and `after - before` picks up an LR term that no
/// amount of arithmetic separates. `before` is the trend over anchors strictly below the boundary,
/// `after` the straddling pair. `cadence` is the checkpoint spacing, used only to state the anchors
/// a future run should retain.
fn repetition_kink(
    axes: &[CheckpointAxes],
    schedule: &Schedule,
    population: SlopePopulation,
    cadence: usize,
) -> RepetitionKink {
    let boundary = schedule.steps_per_epoch();
    let required = required_plateau_anchors(boundary, schedule.plateau_last_step(), cadence);

    let mut plateau: Vec<&CheckpointAxes> = axes.iter().filter(|axis| axis.in_lr_plateau).collect();
    plateau.sort_by_key(|axis| axis.step);
    plateau.dedup_by_key(|axis| axis.step);

    let trend = |lo: &CheckpointAxes, hi: &CheckpointAxes| -> f64 {
        (population.beta(hi) - population.beta(lo)) / (hi.step - lo.step) as f64
    };
    // The straddling pair: the last anchor below the boundary and the first at or above it.
    let split = plateau.partition_point(|axis| axis.step < boundary);
    if split == 0 || split == plateau.len() || split < 2 {
        return RepetitionKink::unavailable(boundary, required);
    }
    let before = trend(plateau[split - 2], plateau[split - 1]);
    let after = trend(plateau[split - 1], plateau[split]);
    RepetitionKink {
        boundary_step: boundary,
        before,
        after,
        kink: after - before,
        anchors: plateau[(split - 2)..=split]
            .iter()
            .map(|axis| axis.step)
            .collect(),
        required_anchors: required,
    }
}

fn jitter_floor(
    axes: &[CheckpointAxes],
    cadence: &[usize],
    population: SlopePopulation,
) -> JitterFloor {
    let betas: Vec<f64> = cadence
        .iter()
        .map(|index| population.beta(&axes[*index]))
        .collect();
    let consecutive: Vec<f64> = betas.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let disagreement = if consecutive.len() >= 2 {
        let lo = consecutive.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = consecutive.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    } else {
        0.0
    };
    let span = {
        let lo = betas.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = betas.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    };

    // Two artifacts recording one step hold the SAME WEIGHTS — `write_checkpoint` serializes the
    // live VarStore and no retained-best buffer exists — so this is a DETERMINISM null control
    // that must read exactly 0.0, not a jitter datum. Excluded from `worst()`. NaN rather than
    // zero when no such pair exists: not running a control is not passing it.
    let mut null_control = f64::NAN;
    let mut null_control_labels = None;
    'outer: for (i, left) in axes.iter().enumerate() {
        for right in axes.iter().skip(i + 1) {
            if left.step == right.step {
                null_control = population.beta(right) - population.beta(left);
                null_control_labels = Some((left.label.clone(), right.label.clone()));
                break 'outer;
            }
        }
    }

    JitterFloor {
        cadence_steps: cadence.iter().map(|index| axes[*index].step).collect(),
        cadence_spacing: if cadence.len() >= 2 {
            axes[cadence[1]].step - axes[cadence[0]].step
        } else {
            0
        },
        consecutive,
        disagreement,
        span,
        null_control,
        null_control_labels,
    }
}

fn attribute(
    population: SlopePopulation,
    spine: Contrast,
    local: Contrast,
    jitter: f64,
) -> Attribution {
    let extrapolation_factor = spine.d_lr / local.d_lr;
    let identified_passes = spine.d_passes - local.d_passes * extrapolation_factor;
    let attenuation = spine.d_passes / identified_passes;
    let amplification = attenuation * extrapolation_factor;
    let local_lr_sensitivity = local.d_beta / local.d_lr;
    let residual = spine.d_beta - extrapolation_factor * local.d_beta;
    let passes_component = attenuation * residual;
    let lr_component = spine.d_beta - passes_component;
    let half_width = amplification.abs() * jitter;
    Attribution {
        population,
        spine,
        local,
        extrapolation_factor,
        identified_passes,
        attenuation,
        amplification,
        local_lr_sensitivity,
        passes_component,
        lr_component,
        half_width,
        jitter,
        resolved: half_width.is_finite() && half_width < spine.d_beta.abs(),
    }
}

fn sweep(attribution: &Attribution) -> Vec<AttributionAtAssumed> {
    let observed = attribution.local.d_beta.abs();
    let reach = GRID_SPAN_MULTIPLE * observed.max(attribution.jitter).max(f64::MIN_POSITIVE);
    (0..GRID_POINTS)
        .map(|slot| {
            let unit = 2.0 * slot as f64 / (GRID_POINTS - 1) as f64 - 1.0;
            let assumed = unit * reach;
            let residual = attribution.spine.d_beta - attribution.extrapolation_factor * assumed;
            let passes_component = attribution.attenuation * residual;
            AttributionAtAssumed {
                assumed_d_beta_local: assumed,
                passes_component,
                lr_component: attribution.spine.d_beta - passes_component,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::train::pretrain::LR_PLATEAU_FRACTION;

    /// The run this was written for. `stage_steps` sum to `steps_per_epoch`; the split among them
    /// is irrelevant to every quantity here BECAUSE `lr_affine_in_step` holds, which
    /// `disentangle` asserts rather than assumes.
    fn bardist_v2() -> Schedule {
        Schedule::new(
            [3455, 3455, 3455],
            31095,
            24,
            [1, 1, 1],
            LR_PLATEAU_FRACTION,
        )
    }

    fn axis(label: &str, step: usize, schedule: &Schedule, beta: f64) -> CheckpointAxes {
        CheckpointAxes {
            label: label.to_owned(),
            step,
            progress: step as f64 / schedule.total_steps() as f64,
            lr_mult: schedule.lr_multiplier(step),
            passes: schedule.passes_at(step),
            in_lr_plateau: schedule.in_lr_plateau(step),
            beta_traded: beta,
            beta_fit: beta,
            beta_fit_ci: (f64::NAN, f64::NAN),
            beta_fit_blocks: 40,
            beta_fit_samples: 100_000,
        }
    }

    /// The rank-1 constant is `-epochs * (1 - F) / (P - floor)` and NOTHING ELSE. Asserted
    /// against a hand-evaluated `-36/17`, and asserted to be invariant to the corpus: a schedule
    /// with a different `steps_per_epoch` and hence a different `total_steps` at the same epoch
    /// count must return the identical number, because that is the claim.
    #[test]
    fn the_passes_per_learning_rate_constant_is_pure_recipe() {
        let schedule = bardist_v2();
        assert!((schedule.passes_per_lr_unit() - (-36.0 / 17.0)).abs() < 1e-12);

        let smaller = Schedule::new([100, 100, 100], 900, 24, [1, 1, 1], LR_PLATEAU_FRACTION);
        assert_eq!(smaller.total_steps() / smaller.steps_per_epoch(), 3);
        assert!(
            (smaller.passes_per_lr_unit() - schedule.passes_per_lr_unit()).abs() < 1e-12,
            "a 300-step epoch and a 10365-step epoch at three epochs must give the same constant"
        );
    }

    /// The constant is not an approximation of the schedule: it is the schedule. Any two steps
    /// past the plateau must realize it exactly.
    #[test]
    fn every_post_plateau_contrast_points_the_same_way() {
        let schedule = bardist_v2();
        let exact = schedule.passes_per_lr_unit();
        for (lo, hi) in [(12600usize, 31000usize), (29696, 30720), (20729, 30000)] {
            let ratio = (schedule.passes_at(hi) - schedule.passes_at(lo))
                / (schedule.lr_multiplier(hi) - schedule.lr_multiplier(lo));
            assert!(
                (ratio - exact).abs() < 1e-9,
                "contrast {lo}->{hi} gave {ratio}, not {exact}: the design would not be rank 1"
            );
        }
    }

    /// The plateau is `epochs * LR_PLATEAU_FRACTION` passes long and its multiplier does not move
    /// inside it. Both halves matter: the length is the identification budget a future run has,
    /// and the flatness is what makes a pair inside it a PURE passes contrast.
    #[test]
    fn the_plateau_is_the_only_zero_learning_rate_contrast_stretch() {
        let schedule = bardist_v2();
        assert_eq!(schedule.plateau_last_step(), 12438);
        assert!((schedule.passes_at(12438) - 1.2).abs() < 1e-12);
        assert!(schedule.in_lr_plateau(10364) && !schedule.in_lr_plateau(12439));
        let flat = schedule.lr_multiplier(0) - schedule.lr_multiplier(12438);
        assert_eq!(flat, 0.0, "the plateau must be exactly flat, not nearly flat");
    }

    /// The identified window is the plateau OVERHANG of the spine's lower checkpoint, and the
    /// amplification is the product of the two factors. These are the numbers a verdict is quoted
    /// against, so they are pinned here rather than recomputed by a reader.
    #[test]
    fn the_identified_window_is_the_plateau_overhang() {
        let schedule = bardist_v2();
        let axes = vec![
            axis("epoch_0", 10364, &schedule, 1.0058),
            axis("epoch_1", 20729, &schedule, 0.95),
            axis("best", 30000, &schedule, 0.8777),
            axis("s29696", 29696, &schedule, 0.88),
            axis("s30208", 30208, &schedule, 0.877),
            axis("s30720", 30720, &schedule, 0.876),
        ];
        let out = disentangle(axes, &schedule, SlopePopulation::FitSlice).expect("decomposes");

        assert_eq!(out.jitter.cadence_steps, vec![29696, 30208, 30720]);
        assert_eq!(out.jitter.cadence_spacing, 512);
        assert_eq!((out.spine.lo_step, out.spine.hi_step), (10364, 30000));
        assert_eq!(out.plateau_checkpoints, 1);

        let attribution = &out.attribution;
        assert!((attribution.extrapolation_factor - 17562.0 / 1024.0).abs() < 1e-9);
        // 12438 - 10364 = 2074 steps of plateau overhang, in passes.
        assert!((attribution.identified_passes - 2074.0 / 10365.0).abs() < 1e-9);
        assert!((attribution.attenuation - 19636.0 / 2074.0).abs() < 1e-6);
        assert!(
            (attribution.amplification - (19636.0 / 2074.0) * (17562.0 / 1024.0)).abs() < 1e-5
        );
        // The components are a decomposition: they sum to the movement exactly.
        assert!(
            (attribution.passes_component + attribution.lr_component - out.spine.d_beta).abs()
                < 1e-12
        );
    }

    /// The residual recovers the passes coefficient over `identified_passes` and nothing more.
    /// Constructed forward from a known `(a, b)` so the recovery is checked against truth rather
    /// than against itself: with jitter zero the decomposition must be EXACT.
    #[test]
    fn the_decomposition_is_exact_when_the_local_movement_is_exact() {
        let schedule = bardist_v2();
        let (a, b) = (-0.0400, 0.0900);
        let beta = |step: usize| {
            1.0 + a * schedule.passes_at(step) + b * (schedule.lr_multiplier(step) - 1.0)
        };
        let axes = vec![
            axis("epoch_0", 10364, &schedule, beta(10364)),
            axis("epoch_1", 20729, &schedule, beta(20729)),
            axis("best", 30000, &schedule, beta(30000)),
            axis("s29696", 29696, &schedule, beta(29696)),
            axis("s30208", 30208, &schedule, beta(30208)),
            axis("s30720", 30720, &schedule, beta(30720)),
        ];
        let out = disentangle(axes, &schedule, SlopePopulation::FitSlice).expect("decomposes");
        let attribution = &out.attribution;
        assert!(
            (attribution.passes_component - a * out.spine.d_passes).abs() < 1e-9,
            "passes component {} should be a * d_passes = {}",
            attribution.passes_component,
            a * out.spine.d_passes
        );
        assert!(
            (attribution.lr_component - b * out.spine.d_lr).abs() < 1e-9,
            "lr component {} should be b * d_lr = {}",
            attribution.lr_component,
            b * out.spine.d_lr
        );
        assert!(
            (attribution.local_lr_sensitivity - (b + a * schedule.passes_per_lr_unit())).abs()
                < 1e-9,
            "the local sensitivity is biased by exactly a * passes_per_lr_unit, by construction"
        );
    }

    /// A jitter floor larger than the movement divided by the amplification must refuse to call
    /// the split resolved, and must say so in a string that survives the renderer.
    #[test]
    fn an_amplified_jitter_floor_refuses_to_resolve_the_split() {
        let schedule = bardist_v2();
        // Cadence slopes that disagree: the two 512-step differences are +0.004 and -0.004.
        let axes = vec![
            axis("epoch_0", 10364, &schedule, 1.0058),
            axis("epoch_1", 20729, &schedule, 0.95),
            axis("best", 30000, &schedule, 0.8777),
            axis("s29696", 29696, &schedule, 0.8800),
            axis("s30208", 30208, &schedule, 0.8840),
            axis("s30720", 30720, &schedule, 0.8800),
        ];
        let out = disentangle(axes, &schedule, SlopePopulation::FitSlice).expect("decomposes");
        assert!((out.jitter.disagreement - 0.008).abs() < 1e-9);
        assert!(
            out.attribution.half_width > out.spine.d_beta.abs(),
            "0.008 amplified 162x is 1.3, far wider than a 0.128 movement"
        );
        assert!(!out.attribution.resolved);
        assert!(out.attribution.caveat().contains("UNRESOLVED"));
    }

    /// A pair of artifacts recording ONE step is carried but must NOT enter the floor, and its
    /// absence must read as absent rather than as zero jitter.
    ///
    /// It looks like the cleanest possible jitter datum — zero passes contrast, zero learning-rate
    /// contrast — and it is not a jitter datum at all. Two criteria promoting at one step write THE
    /// SAME WEIGHTS TWICE, because `write_checkpoint` serializes the live `VarStore` and no
    /// retained best-so-far buffer exists in the promotion path. Confirmed on the artifacts: the
    /// multiset of zip-record CRC-32s is identical across `pretrain_best`, `pretrain_best_diag896`
    /// and `pretrain_last`, and differs from `pretrain_step_30720`, so the differing
    /// `checkpoint_sha256` is libtorch recording the file stem as the archive name.
    ///
    /// So the real difference is identically zero, and this field is a NULL CONTROL on evaluation
    /// determinism rather than a measurement of anything. `worst()` must ignore it: a correct
    /// harness contributes zero, so including it could only ever widen the band by importing a bug.
    /// The fixture below uses a deliberately nonzero synthetic value to prove the exclusion holds
    /// even then.
    #[test]
    fn a_null_control_is_carried_but_never_enters_the_floor() {
        let schedule = bardist_v2();
        let mut axes = vec![
            axis("epoch_0", 10364, &schedule, 1.0058),
            axis("best", 30000, &schedule, 0.8777),
            axis("s29696", 29696, &schedule, 0.8800),
            axis("s30208", 30208, &schedule, 0.8790),
            axis("s30720", 30720, &schedule, 0.8780),
        ];
        let bare = disentangle(axes.clone(), &schedule, SlopePopulation::FitSlice)
            .expect("decomposes");
        assert!(
            bare.jitter.null_control.is_nan()
                && bare.jitter.null_control_labels.is_none(),
            "no such pair must read as NaN, never as a measured zero"
        );
        let floor_without = bare.jitter.worst();

        axes.push(axis("best_diag896", 30000, &schedule, 0.8600));
        let with = disentangle(axes, &schedule, SlopePopulation::FitSlice).expect("decomposes");
        assert!((with.jitter.null_control.abs() - 0.0177).abs() < 1e-9);
        assert_eq!(
            with.jitter.null_control_labels.as_ref().map(|pair| pair.1.as_str()),
            Some("best_diag896")
        );
        assert_eq!(
            with.jitter.worst(),
            floor_without,
            "a 0.0177 step-ambiguous contrast must not move a 0.001 cadence floor"
        );
    }

    /// With no plateau checkpoint the run contains no contrast off the rank-1 direction, and the
    /// analysis must REFUSE rather than return a decomposition its inputs cannot support.
    #[test]
    fn a_run_with_no_plateau_checkpoint_is_refused() {
        let schedule = bardist_v2();
        let axes = vec![
            axis("epoch_1", 20729, &schedule, 0.95),
            axis("best", 30000, &schedule, 0.8777),
            axis("s29696", 29696, &schedule, 0.880),
            axis("s30208", 30208, &schedule, 0.879),
            axis("s30720", 30720, &schedule, 0.878),
        ];
        let error = disentangle(axes, &schedule, SlopePopulation::FitSlice)
            .expect_err("must refuse")
            .to_string();
        assert!(error.contains("rank-1"), "{error}");
    }

    /// The sweep must straddle zero exactly and must make the amplification visible as a slope.
    #[test]
    fn the_sweep_straddles_zero_and_shows_the_amplification() {
        let schedule = bardist_v2();
        let axes = vec![
            axis("epoch_0", 10364, &schedule, 1.0058),
            axis("best", 30000, &schedule, 0.8777),
            axis("s29696", 29696, &schedule, 0.8800),
            axis("s30208", 30208, &schedule, 0.8790),
            axis("s30720", 30720, &schedule, 0.8780),
        ];
        let out = disentangle(axes, &schedule, SlopePopulation::FitSlice).expect("decomposes");
        assert_eq!(out.grid.len(), GRID_POINTS);
        let middle = out.grid[GRID_POINTS / 2];
        assert_eq!(middle.assumed_d_beta_local, 0.0);
        assert!(
            (middle.passes_component - out.spine.d_beta * out.attribution.attenuation).abs()
                < 1e-9,
            "at zero assumed local movement the whole spine movement is attributed to passes, \
             scaled by the attenuation"
        );
        for point in &out.grid {
            assert!(
                (point.passes_component + point.lr_component - out.spine.d_beta).abs() < 1e-9
            );
        }
        let rise = out.grid[GRID_POINTS - 1].passes_component - out.grid[0].passes_component;
        let run = out.grid[GRID_POINTS - 1].assumed_d_beta_local - out.grid[0].assumed_d_beta_local;
        assert!(
            ((rise / run) + out.attribution.amplification).abs() < 1e-6,
            "the sweep's slope IS the amplification, negated"
        );
    }

    /// Non-affine schedules are refused: with different batch bumps per stage the multiplier
    /// jumps at each stage boundary and there is no single passes-per-learning-rate constant.
    #[test]
    fn a_piecewise_schedule_is_refused_rather_than_averaged() {
        let ramped = Schedule::new(
            [3455, 3455, 3455],
            31095,
            24,
            [1, 2, 3],
            LR_PLATEAU_FRACTION,
        );
        assert!(!ramped.lr_affine_in_step());
        assert!(ramped.passes_per_lr_unit().is_nan());
        let axes = vec![
            axis("epoch_0", 10364, &ramped, 1.0),
            axis("best", 30000, &ramped, 0.9),
            axis("s29696", 29696, &ramped, 0.9),
            axis("s30208", 30208, &ramped, 0.9),
            axis("s30720", 30720, &ramped, 0.9),
        ];
        let error = disentangle(axes, &ramped, SlopePopulation::FitSlice)
            .expect_err("must refuse")
            .to_string();
        assert!(error.contains("piecewise-affine"), "{error}");
    }
}
