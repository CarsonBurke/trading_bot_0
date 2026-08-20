//! Does the run's THIRD pass over the corpus memorize, and does that memorization move the
//! held-out mean slope?
//!
//! # Why this module exists rather than another flag on [`super::pretrain::pretrain_calibration`]
//!
//! The calibration experiment measures a pinned VAL population. The question here is about
//! bars the model TRAINED on, partitioned by HOW MANY TIMES it trained on them, so it needs a
//! different population and cannot be a flag on that tool. Nothing here changes what the
//! calibration trend reports.
//!
//! # The two measurements, and which one carries a verdict
//!
//! **The epoch spine (supporting only).** Train-split against val-split held-out NLL at
//! `pretrain_epoch_0_ctx2048` (step 10364, 1.000 passes), `pretrain_epoch_1_ctx2048` (20729,
//! 2.000) and `pretrain_best` (30000, 2.894). Under a genuinely single-epoch run the gap would
//! be flat; under 2.89 passes it should open.
//!
//! This spine is CONTAMINATED and no estimator repairs it. Train and val are disjoint in
//! CALENDAR — one global `split_bounds` pair, identical for all 5,297 symbols — so the LEVEL of
//! the gap mixes memorization with regime. Worse, the TRAJECTORY is rank-deficient:
//! `Schedule::lr_multiplier` is exactly affine in step past the plateau, so across the decay
//! region passes and learning rate are the SAME VARIABLE, and a lower learning rate reduces
//! gradient noise and therefore weakens implicit regularization, which opens a train/val gap on
//! its own. Only the plateau clip at step 12438 breaks the collinearity, and it contributes
//! 0.2001 passes of identifying variation against a 1.8945-pass contrast. So the spine is
//! reported WITH its contamination attached and is never the discriminator.
//!
//! **The one-repetition contrast (the discriminator).** At step 30000 the run is 9270 steps
//! into epoch 2. That falls in ramp stage 2, 2360 steps into the stage, so stage 2 of epoch 2
//! has issued `2360 * 24 = 56,640` of its ~82,919 windows.
//!
//! Bars in an ISSUED epoch-2 stage-2 window have been trained on THREE times. Bars in a
//! not-yet-issued one, twice. Everything else is physically identical: the same weights, hence
//! the same learning rate and the same momentum; the same ramp stage, hence the same context
//! 2048, the same batch 24 and the same mean conditioning depth 1024.5; and the same
//! `PassPlan::counts`, which `PassPlan::new` computes without an epoch argument at all.
//!
//! And the split is RANDOMIZED, not merely matched. `PassPlan::build_layout` ends with a GLOBAL
//! `windows.shuffle(&mut rng)` per stage, so the issued prefix of a stage is a uniformly random
//! subset of that stage's windows. One variable, randomized, everything else held fixed by
//! physics rather than by adjustment.
//!
//! Exposure counts are LITERAL rather than approximate because a pass PARTITIONS bars: stride
//! equals context, so a symbol's windows tile its axis exactly and a covered bar is targeted
//! exactly once per pass. Which stage owns a bar is redrawn every epoch, so the two arms are
//! exchangeable in their prior exposure CONTEXTS as well as in composition.
//!
//! # What a result means, and what it does not
//!
//! A better NLL on the seen-three-times arm establishes that memorization OCCURS. It does not
//! by itself establish that memorization explains the VAL slope decay: "fits training rows it
//! has seen more often" and "is over-dispersed out of sample" are different propositions and
//! the second does not follow from the first. The MZ MEAN SLOPE across the same boundary is
//! what connects them. If the slope is materially lower on the seen-three-times arm, one extra
//! repetition demonstrably produces over-dispersion and the val decay is that mechanism
//! extended. If NLL improves while the slope does not move, the model memorizes WITHOUT
//! over-dispersing and memorization is the wrong explanation for the slope — a clean
//! refutation, and the more valuable outcome.
//!
//! # Estimators
//!
//! No new estimator exists here. Levels come from [`trade_bench::mincer_zarnowitz`] and
//! [`super::pretrain_stats::block_bootstrap`] at the campaign's `BOOTSTRAP_DRAWS` and
//! `BOOTSTRAP_SEED`. Contrasts are PAIRED on `(symbol, calendar month)` blocks present in both
//! arms: two independently-blocked intervals differenced would keep the block's own regime in
//! both and produce a band several times too wide. A per-block slope comes from calling
//! `mincer_zarnowitz` on that block's rows with a single block id, which returns the pooled OLS
//! slope under the module's own non-finite filter rather than a second implementation of it.
//!
//! Two BLOCK KEYS are reported for every slope, on identical rows. `(symbol, month)` is the
//! campaign's key; it treats two different symbols in the same month as independent draws,
//! which they are not — same-instant cross-symbol correlation is the dominant dependence in
//! this corpus. MONTH ALONE pools every symbol in a month and is therefore conservative against
//! exactly that term. `blocks: &[u64]` is an opaque key, so this is a key change and never a
//! second estimator.
//!
//! # Memory
//!
//! Every allocation is bounded and asserted. The partition is a CURSOR COMPARISON on window
//! indices — no per-bar occupancy structure is ever built — and the layout itself is three
//! vectors of 8-byte `WindowRef`, about 2 MB. Arm bars are capped by [`MAX_ARM_BARS`]. The row
//! dump streams to disk through a fixed buffer and never accumulates rows.

use anyhow::{anyhow, ensure, Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use tch::Device;

use crate::torch::bar_dist::BarScoring;
use crate::torch::cuda::cfg::configure_cuda;
use crate::torch::dataset::{PassPlan, Split, WindowRef};
use crate::torch::world_model::{world_model_metadata_path, BarWorldModel};

use super::pretrain::{
    evaluate, load_corpus, parse_checkpoint_at_step, pinned_blocks, ramp_token_weights,
    stage_contexts, CorpusFlags, PinnedSet, EVAL_WINDOW_SEED, RAMP_STAGES,
};
use super::pretrain_stats::{
    block_bootstrap, calendar_month, Dispersion, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED,
};
use super::trade_bench::{self, MzFit, WindowPaths, TRADE_WINDOWS};

/// Windows on EACH side of the issue cursor that neither arm may claim.
///
/// The cursor at a validation step is `within_stage_step * batch`, but whether the step's own
/// draw had already been taken when the checkpoint was written is a one-batch ambiguity. Ten
/// steps of slack is an order of magnitude more than that ambiguity and costs 0.9% of the
/// smaller arm, so the partition is unambiguous at no measurable price.
const CURSOR_GUARD_WINDOWS: usize = 240;

/// Ceiling on the bars ONE arm may pool.
///
/// [`trade_bench::mean_calibration`] builds six per-bar vectors, five `f64` plus a four-`f64`
/// `OuterBar`, so a bar costs 72 bytes. Three million bars is about 216 MB per arm, which is
/// the largest footprint this measurement is allowed to have. Exceeding it is REFUSED rather
/// than silently swapped: this box has already been OOM-killed once by an unbounded census.
const MAX_ARM_BARS: usize = 3_000_000;

/// Draw counts the bootstrap's stability in `draws` is characterized over.
///
/// 92.384% of a decoded mean's per-bar sampling variance sits in the two catch-all bins, so the
/// bootstrap resamples a heavy-tailed statistic and its percentile interval may converge more
/// slowly than the module's doc assumes. Whether `BOOTSTRAP_DRAWS` suffices is a property of
/// the DATA and has never been measured on this one.
const STABILITY_DRAWS: [usize; 6] = [125, 250, 500, 1000, 2000, 4000];

/// Recency buckets the extra-exposure arm is split into.
///
/// The third exposure of an issued window happened somewhere in the 2360 steps of stage 2 that
/// preceded the checkpoint, and the exact step is recoverable because the stage cursor is
/// sequential and never wraps. Bucketing by it separates DURABLE memorization, flat across the
/// epoch, from TRANSIENT retention of recently-visited rows, concentrated at the newest bucket.
/// Those have different remedies — fewer passes against a different schedule — so collapsing
/// them would hide the actionable half.
const RECENCY_BUCKETS: usize = 5;

/// Row-dump layout version. A reader that misinterprets a binary layout produces plausible
/// numbers rather than an error, so the layout is versioned and self-describing.
const ROW_DUMP_FORMAT_VERSION: u32 = 1;

/// Bytes of one dumped row: the window key, two flag bytes, then seven `f64`.
const ROW_BYTES: usize = 4 + 4 + 4 + 8 + 1 + 1 + 7 * 8;

/// Arguments of the memorization probe.
#[derive(Clone, Debug)]
pub struct MemProbeArgs {
    /// Epoch-spine checkpoints as `path@step`, for the train-versus-held-out gap. The step is
    /// stated because the metadata sidecar does not record it: `.training` carries
    /// `reached_context` and the seeds but there is no `global_step` anywhere in the file.
    pub checkpoints: Vec<String>,
    /// Checkpoint whose weights carry the one-repetition contrast, as `path@step`. Its step
    /// decides the partition, so it is parsed rather than assumed.
    pub partition_checkpoint: String,
    /// Directory the `memprobe_*` charts are written into.
    pub output: String,
    /// Windows drawn per split for the gap. Both splits get the same count through the same
    /// constructor at the same seed, so the two draws differ only in split range.
    pub gap_windows: usize,
    /// Windows sampled from EACH arm of the one-repetition contrast.
    pub arm_windows: usize,
    pub context: i64,
    pub batch_size: usize,
    /// The run's REALIZED per-stage batch. `PassPlan` normalizes its token weights, so the
    /// realized batches `[24,24,24]` and the multipliers `[1,1,1]` give a byte-identical
    /// partition; the realized figure is taken because it is what the metadata records.
    pub batch_ramp: [usize; RAMP_STAGES],
    /// The run's `train_seed`. NOT `EVAL_WINDOW_SEED`: the partition being reconstructed is the
    /// TRAINING sampler's, keyed by `(train_seed, epoch)` through `PASS_STREAM`.
    pub train_seed: u64,
    pub corpus: CorpusFlags,
}

/// `--batch-ramp` as exactly one entry per ramp stage.
///
/// A parser rather than a `Vec<usize>` widened at the call site: an arity mistake has to be a
/// command-line error, not a panic after a multi-gigabyte corpus has already been loaded.
pub fn parse_batch_ramp(raw: &str) -> Result<[usize; RAMP_STAGES], String> {
    let stages = raw
        .split(',')
        .map(|entry| {
            entry
                .trim()
                .parse::<usize>()
                .map_err(|err| format!("`{entry}` is not a batch size: {err}"))
        })
        .collect::<Result<Vec<usize>, String>>()?;
    stages.try_into().map_err(|stages: Vec<usize>| {
        format!(
            "--batch-ramp takes exactly {RAMP_STAGES} comma-separated entries, one per ramp \
             stage; got {}",
            stages.len()
        )
    })
}

/// One arm of a contrast, measured.
#[derive(Clone, Debug)]
pub struct Arm {
    pub(super) label: String,
    /// Times the model trained on this arm's bars before the checkpoint was written.
    pub(super) exposures: usize,
    pub(super) windows: usize,
    /// `(symbol, calendar month)` resampling units. The interval's power is set by THIS, not by
    /// the window count.
    pub(super) blocks: usize,
    /// Bars whose predicted mean and realized return were both finite, and bars that were not.
    /// `mincer_zarnowitz` silently skips non-finite rows, so a non-finite predicted mean would
    /// quietly change the population a slope is fitted on; counting the drops makes it visible.
    pub(super) finite_bars: usize,
    pub(super) dropped_bars: usize,
    pub(super) nll: Dispersion,
    pub(super) nll_conditional: Dispersion,
    /// Slope under the campaign's `(symbol, month)` key.
    pub(super) mean_slope: MzFit,
    /// The SAME slope on the SAME rows under a MONTH-ALONE key, which pools every symbol in a
    /// month and is therefore conservative against same-instant cross-symbol dependence.
    pub(super) mean_slope_month: MzFit,
    rows: Vec<ArmRow>,
}

/// One window of one arm.
#[derive(Clone, Debug)]
struct ArmRow {
    block: u64,
    month: u64,
    nll: f64,
    nll_conditional: f64,
    /// Optimizer step at which this window was issued in the partition's epoch, when it was.
    visit_step: Option<usize>,
    paths: WindowPaths,
}

/// A paired contrast between two arms over the blocks they share.
#[derive(Clone, Debug)]
pub struct PairedContrast {
    /// Blocks present in BOTH arms. Only these can carry a paired observation.
    pub(super) shared_blocks: usize,
    /// Blocks that additionally supported an OLS slope in both arms.
    pub(super) slope_blocks: usize,
    pub(super) nll: Dispersion,
    pub(super) nll_conditional: Dispersion,
    pub(super) slope: Dispersion,
    /// The slope contrast re-keyed by month alone, same rows, same estimator.
    pub(super) slope_month: Dispersion,
    pub(super) slope_month_blocks: usize,
}

/// One point of the epoch spine.
#[derive(Clone, Debug)]
pub struct GapPoint {
    pub(super) label: String,
    pub(super) step: usize,
    pub(super) train_nll: f64,
    pub(super) heldout_nll: f64,
    pub(super) train_nll_conditional: f64,
    pub(super) heldout_nll_conditional: f64,
    /// Difference of the two pooled means. The headline, and contaminated.
    pub(super) gap: f64,
    pub(super) gap_conditional: f64,
    /// Difference paired WITHIN SYMBOL: for each symbol drawn in both splits, its mean train
    /// window NLL minus its mean val window NLL, intervalled over symbols. This removes the
    /// cross-sectional component — per-name liquidity and volatility level — and leaves the
    /// calendar component, which no pairing can remove because the splits are calendar-disjoint
    /// by construction. Strictly less contaminated, never clean.
    pub(super) symbol_paired_gap: Dispersion,
    pub(super) symbols_paired: usize,
}

/// One recency bucket of the arm that carries the extra exposure.
#[derive(Clone, Debug)]
pub struct RecencyBucket {
    /// Mean steps between the window's issue and the checkpoint.
    pub(super) steps_ago: f64,
    pub(super) windows: usize,
    pub(super) blocks: usize,
    pub(super) nll: Dispersion,
    pub(super) slope: MzFit,
}

/// `beta_se` and the interval's width as a function of bootstrap `draws`.
#[derive(Clone, Copy, Debug)]
pub struct StabilityPoint {
    pub(super) draws: usize,
    pub(super) beta: f64,
    pub(super) beta_se: f64,
    pub(super) ci_width: f64,
}

/// Run the probe.
pub fn mem_probe(args: MemProbeArgs) -> Result<()> {
    ensure!(
        !args.checkpoints.is_empty(),
        "--checkpoint must be given at least once, as path@step"
    );
    ensure!(args.gap_windows > 0, "--gap-windows must be positive");
    ensure!(args.arm_windows > 0, "--arm-windows must be positive");
    ensure!(args.context > 0, "--context must be positive");
    ensure!(args.batch_size > 0, "--batch-size must be positive");
    let arm_bars = args.arm_windows.saturating_mul(args.context as usize);
    ensure!(
        arm_bars <= MAX_ARM_BARS,
        "--arm-windows {} at context {} pools {arm_bars} bars per arm, past the {MAX_ARM_BARS} \
         cap; the calibration builds six per-bar vectors at 72 bytes a bar, so this would ask \
         for {} MiB per arm. Lower --arm-windows.",
        args.arm_windows,
        args.context,
        arm_bars * 72 / (1 << 20),
    );

    configure_cuda();
    let device = Device::cuda_if_available();

    let spine = args
        .checkpoints
        .iter()
        .map(|entry| parse_checkpoint_at_step(entry))
        .collect::<Result<Vec<_>>>()?;
    let (partition_path, partition_step) = parse_checkpoint_at_step(&args.partition_checkpoint)?;

    // ONE geometry across every checkpoint this pass touches. Two checkpoints resolving
    // different supports would decode their means on different bins and produce individually
    // correct, mutually incomparable slopes, with nothing anywhere reporting a problem.
    let geometry = assert_one_geometry(
        spine
            .iter()
            .map(|(path, _)| path.as_str())
            .chain(std::iter::once(partition_path.as_str())),
    )?;
    println!("one bin geometry across all checkpoints: supports_sha256 {geometry}");

    let corpus = load_corpus(&args.corpus)?;

    // ---------------------------------------------------------------------
    // The epoch spine: train against held-out, at each pass count.
    // ---------------------------------------------------------------------
    let heldout = PinnedSet::pinned(&corpus, Split::Val, args.context, args.gap_windows)?;
    let train = PinnedSet::pinned(&corpus, Split::Train, args.context, args.gap_windows)?;
    println!(
        "epoch spine: {} pinned val windows against {} pinned train windows, both drawn by \
         PinnedSet::pinned at eval_window_seed {EVAL_WINDOW_SEED:#x} and context {}. The two \
         draws differ ONLY in split range, and the split ranges are calendar-DISJOINT, which is \
         why the LEVEL of this gap is contaminated by regime and its TRAJECTORY by the learning \
         rate.",
        heldout.windows.len(),
        train.windows.len(),
        args.context,
    );
    let heldout_blocks = pinned_blocks(&heldout);
    let train_blocks = pinned_blocks(&train);
    println!(
        "  resampling units: {} val blocks, {} train blocks",
        heldout_blocks.iter().collect::<BTreeSet<_>>().len(),
        train_blocks.iter().collect::<BTreeSet<_>>().len(),
    );

    let mut gap_points = Vec::with_capacity(spine.len());
    for (path, step) in &spine {
        let (world, scoring) = load_checkpoint(path, &args, device)?;
        let held = evaluate(
            world.modules(),
            world.deployment_supports(),
            &heldout,
            args.batch_size,
            device,
            false,
            scoring,
            None,
            // `full: false`, so no path is retained at all and the budget is inert. Stated as
            // the campaign default rather than 0 so this call reads identically to every other.
            TRADE_WINDOWS,
        )?;
        let seen = evaluate(
            world.modules(),
            world.deployment_supports(),
            &train,
            args.batch_size,
            device,
            false,
            scoring,
            None,
            TRADE_WINDOWS,
        )?;
        let (symbol_paired_gap, symbols_paired) =
            symbol_paired_difference(&train, &seen.window_nll, &heldout, &held.window_nll);
        let point = GapPoint {
            label: stem_of(path),
            step: *step,
            train_nll: seen.nll_bar,
            heldout_nll: held.nll_bar,
            train_nll_conditional: seen.nll_bar_conditional,
            heldout_nll_conditional: held.nll_bar_conditional,
            gap: held.nll_bar - seen.nll_bar,
            gap_conditional: held.nll_bar_conditional - seen.nll_bar_conditional,
            symbol_paired_gap,
            symbols_paired,
        };
        println!(
            "  {} @{step}: train {:.4} nats/bar, held-out {:.4}, gap {:+.4}; conditional gap \
             {:+.4}; symbol-paired gap {:+.4} [{:+.4}, {:+.4}] over {} symbols",
            point.label,
            point.train_nll,
            point.heldout_nll,
            point.gap,
            point.gap_conditional,
            point.symbol_paired_gap.mean,
            point.symbol_paired_gap.ci_low,
            point.symbol_paired_gap.ci_high,
            point.symbols_paired,
        );
        gap_points.push(point);
    }

    // ---------------------------------------------------------------------
    // The one-repetition contrast.
    // ---------------------------------------------------------------------
    let plan = PassPlan::new(
        &corpus,
        Split::Train,
        &stage_contexts(),
        &ramp_token_weights(&args.batch_ramp),
        args.train_seed,
    )
    .context("failed rebuilding the run's training partition")?;
    let partition = Partition::of(&plan, &args, partition_step)?;
    println!("{}", partition.describe());

    let (world, scoring) = load_checkpoint(&partition_path, &args, device)?;
    let mut template = PinnedSet::pinned(&corpus, Split::Train, args.context, 1)?;

    let seen_more = measure_arm(
        &world,
        &mut template,
        &partition.seen_more,
        partition.exposures_seen_more,
        &format!(
            "seen {}x - issued in epoch {}",
            partition.exposures_seen_more, partition.epoch
        ),
        &args,
        device,
        scoring,
        Some(&partition),
    )?;
    let seen_fewer = measure_arm(
        &world,
        &mut template,
        &partition.seen_fewer,
        partition.exposures_seen_fewer,
        &format!(
            "seen {}x - not yet issued in epoch {}",
            partition.exposures_seen_fewer, partition.epoch
        ),
        &args,
        device,
        scoring,
        None,
    )?;
    for arm in [&seen_more, &seen_fewer] {
        println!("{}", arm.describe());
    }
    let contrast = paired_contrast(&seen_more, &seen_fewer);
    println!("{}", contrast.describe(&seen_more, &seen_fewer));

    let recency = recency_profile(&seen_more, &partition);
    for line in recency_lines(&recency) {
        println!("{line}");
    }

    let stability = bootstrap_stability(&seen_more);
    for line in stability_lines(&stability) {
        println!("{line}");
    }

    let output = Path::new(&args.output);
    std::fs::create_dir_all(output)
        .with_context(|| format!("failed to create {}", output.display()))?;
    super::pretrain_reports::write_mem_probe(
        output,
        &format!(
            "{} pinned windows per split, {} windows per arm, context {}",
            args.gap_windows, args.arm_windows, args.context
        ),
        &gap_points,
        &seen_more,
        &seen_fewer,
        &contrast,
        &recency,
        &stability,
    )?;
    println!("reports written to {}", output.display());
    for line in verdict_lines(&gap_points, &seen_more, &seen_fewer, &contrast) {
        println!("{line}");
    }
    Ok(())
}

fn stem_of(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_owned())
}

/// `supports_sha256` of every checkpoint, refusing to proceed unless they agree.
fn assert_one_geometry<'a>(paths: impl Iterator<Item = &'a str>) -> Result<String> {
    let mut geometries: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in paths {
        let metadata = world_model_metadata_path(Path::new(path));
        ensure!(
            metadata.exists(),
            "no metadata sidecar beside {path}; copy {} next to the weights",
            metadata.display()
        );
        let text = std::fs::read_to_string(&metadata)
            .with_context(|| format!("failed to read {}", metadata.display()))?;
        let parsed: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("{} is not JSON", metadata.display()))?;
        let sha = parsed
            .get("supports_sha256")
            .map(|value| value.to_string())
            .unwrap_or_else(|| "ABSENT".to_owned());
        geometries.entry(sha).or_default().push(path.to_owned());
    }
    ensure!(
        geometries.len() == 1,
        "the checkpoints resolve {} DIFFERENT bin geometries, so no slope among them is \
         comparable to any other even though each is individually correct: {:?}",
        geometries.len(),
        geometries
    );
    Ok(geometries
        .into_keys()
        .next()
        .expect("exactly one geometry survived the check"))
}

fn load_checkpoint(
    path: &str,
    args: &MemProbeArgs,
    device: Device,
) -> Result<(BarWorldModel, BarScoring)> {
    let weights = Path::new(path);
    let metadata = world_model_metadata_path(weights);
    let world = BarWorldModel::load(weights, &metadata, device)?;
    ensure!(
        world.metadata().res_secs == args.corpus.resolution_secs,
        "{path} was trained for {}s bars but --resolution-secs is {}",
        world.metadata().res_secs,
        args.corpus.resolution_secs
    );
    if let Some(trained) = world.metadata().training.as_ref() {
        ensure!(
            trained.eval_window_seed == EVAL_WINDOW_SEED,
            "{path} pinned its evaluation with eval_window_seed {:#x} but this build uses \
             {EVAL_WINDOW_SEED:#x}; the windows would not be the run's own",
            trained.eval_window_seed
        );
        ensure!(
            trained.train_seed == args.train_seed,
            "{path} was trained at train_seed {} but --train-seed is {}; the reconstructed pass \
             partition would be a DIFFERENT partition from the one this checkpoint saw, and \
             every exposure count derived from it would be wrong while looking right",
            trained.train_seed,
            args.train_seed
        );
    }
    let scoring: BarScoring = world
        .metadata()
        .training
        .as_ref()
        .map(|trained| trained.scoring.parse())
        .transpose()
        .map_err(|reason| {
            anyhow!("{path} records a scoring rule this build cannot parse: {reason}")
        })?
        .unwrap_or_default();
    Ok((world, scoring))
}

/// The seen-more / seen-fewer split of one ramp stage of one epoch.
struct Partition {
    epoch: usize,
    stage: usize,
    within_stage_step: usize,
    steps_per_epoch: usize,
    stage_steps: [usize; RAMP_STAGES],
    batch: usize,
    cursor: usize,
    stage_windows: usize,
    exposures_seen_more: usize,
    exposures_seen_fewer: usize,
    seen_more: Vec<WindowRef>,
    seen_fewer: Vec<WindowRef>,
    /// Issue index within the stage of each sampled `seen_more` window, parallel to it, so the
    /// visit step is recoverable.
    seen_more_issue_index: Vec<usize>,
}

impl Partition {
    fn of(plan: &PassPlan, args: &MemProbeArgs, step: usize) -> Result<Self> {
        // `stage_steps[s] = ceil(windows[s] / batch[s])`, straight off the partition, exactly as
        // `Schedule` derives it. Recomputed rather than read so the reconstruction can be
        // checked against the run's own recorded totals.
        let per_stage = plan.windows_per_stage();
        ensure!(
            per_stage.len() == RAMP_STAGES,
            "the rebuilt partition has {} stages against the recipe's {RAMP_STAGES}",
            per_stage.len()
        );
        ensure!(
            args.batch_ramp.iter().all(|batch| *batch > 0),
            "--batch-ramp {:?} has a zero stage; the run cannot have taken zero-window steps",
            args.batch_ramp
        );
        let stage_steps: [usize; RAMP_STAGES] =
            std::array::from_fn(|stage| per_stage[stage].div_ceil(args.batch_ramp[stage]));
        let steps_per_epoch = stage_steps.iter().sum::<usize>().max(1);
        let epoch = step / steps_per_epoch;
        let mut within = step % steps_per_epoch;
        let mut stage = RAMP_STAGES - 1;
        for candidate in 0..RAMP_STAGES {
            if within < stage_steps[candidate] {
                stage = candidate;
                break;
            }
            within -= stage_steps[candidate];
        }
        ensure!(
            stage == RAMP_STAGES - 1,
            "step {step} falls in ramp stage {stage} at context {}, not the deployed stage. Only \
             the deployed stage carries a within-epoch exposure boundary at the deployed \
             context: earlier stages are either fully issued or not yet started at a validation \
             step, and comparing across stages would vary context and conditioning depth as \
             well as exposure count.",
            plan.contexts()[stage],
        );
        ensure!(
            plan.contexts()[stage] == args.context,
            "the deployed ramp stage tiles at context {} but --context is {}; the arms would be \
             scored at a context the run never trained them at",
            plan.contexts()[stage],
            args.context
        );
        let batch = args.batch_ramp[stage];

        let layout = plan.layout(epoch);
        let windows = layout.windows(stage);
        let cursor = within * batch;
        ensure!(
            cursor < windows.len(),
            "step {step} puts the stage-{stage} cursor at {cursor} of {} assigned windows, so \
             the stage is fully issued and there is no seen-fewer arm to contrast against",
            windows.len()
        );
        let issued_end = cursor.saturating_sub(CURSOR_GUARD_WINDOWS);
        let unissued_start = (cursor + CURSOR_GUARD_WINDOWS).min(windows.len());
        ensure!(
            issued_end >= 2 && unissued_start + 2 <= windows.len(),
            "the {CURSOR_GUARD_WINDOWS}-window guard band leaves {issued_end} issued and {} \
             unissued windows, which cannot support two arms",
            windows.len().saturating_sub(unissued_start)
        );

        // Systematic subsampling of an already uniformly-shuffled list. `build_layout` ends with
        // a global per-stage shuffle, so an evenly spaced sample of it is a uniformly random
        // subset AND it spreads the issued arm evenly over the epoch's visit steps, which is
        // what the recency decomposition needs. Deterministic, and identical logic on both arms
        // so neither is drawn differently from the other.
        let (seen_more, seen_more_issue_index) =
            systematic(windows, 0, issued_end, args.arm_windows);
        let (seen_fewer, _) = systematic(windows, unissued_start, windows.len(), args.arm_windows);

        Ok(Self {
            epoch,
            stage,
            within_stage_step: within,
            steps_per_epoch,
            stage_steps,
            batch,
            cursor,
            stage_windows: windows.len(),
            exposures_seen_more: epoch + 1,
            exposures_seen_fewer: epoch,
            seen_more,
            seen_fewer,
            seen_more_issue_index,
        })
    }

    /// The step at which this checkpoint was written, in the run's own step numbering.
    fn checkpoint_step(&self) -> usize {
        self.epoch * self.steps_per_epoch
            + self.stage_steps[..self.stage].iter().sum::<usize>()
            + self.within_stage_step
    }

    /// Optimizer step at which the window sampled at `slot` of the seen-more arm was issued.
    fn visit_step(&self, slot: usize) -> Option<usize> {
        let issue = *self.seen_more_issue_index.get(slot)?;
        let stage_start: usize = self.stage_steps[..self.stage].iter().sum();
        Some(self.epoch * self.steps_per_epoch + stage_start + issue / self.batch)
    }

    fn describe(&self) -> String {
        format!(
            "one-repetition partition: epoch {}, ramp stage {}, within-stage step {} of {}, \
             batch {}. Stage cursor {} of {} assigned windows, so {:.2}% of the stage was \
             issued. Arms: {} windows seen {}x against {} windows seen {}x, with a \
             {CURSOR_GUARD_WINDOWS}-window guard band on each side of the cursor. \
             steps_per_epoch {} (stage_steps {:?}), checkpoint at step {}.",
            self.epoch,
            self.stage,
            self.within_stage_step,
            self.stage_steps[self.stage],
            self.batch,
            self.cursor,
            self.stage_windows,
            100.0 * self.cursor as f64 / self.stage_windows as f64,
            self.seen_more.len(),
            self.exposures_seen_more,
            self.seen_fewer.len(),
            self.exposures_seen_fewer,
            self.steps_per_epoch,
            self.stage_steps,
            self.checkpoint_step(),
        )
    }
}

/// `count` evenly spaced entries of `windows[lo..hi]`, with their offsets from `lo`.
fn systematic(
    windows: &[WindowRef],
    lo: usize,
    hi: usize,
    count: usize,
) -> (Vec<WindowRef>, Vec<usize>) {
    let span = hi.saturating_sub(lo);
    let take = count.min(span);
    let mut picked = Vec::with_capacity(take);
    let mut indices = Vec::with_capacity(take);
    for slot in 0..take {
        // Midpoint of the slot's stratum, so the sample is balanced rather than front-loaded.
        let index = (lo + (2 * slot + 1) * span / (2 * take)).min(hi - 1);
        picked.push(windows[index]);
        indices.push(index - lo);
    }
    (picked, indices)
}

/// Score one arm, chunked to the [`TRADE_WINDOWS`] cap on conditional-moment retention.
#[allow(clippy::too_many_arguments)]
fn measure_arm(
    world: &BarWorldModel,
    template: &mut PinnedSet,
    windows: &[WindowRef],
    exposures: usize,
    label: &str,
    args: &MemProbeArgs,
    device: Device,
    scoring: BarScoring,
    partition: Option<&Partition>,
) -> Result<Arm> {
    ensure!(!windows.is_empty(), "arm `{label}` has no window");
    let mut rows: Vec<ArmRow> = Vec::with_capacity(windows.len());
    let mut slot = 0usize;
    // `evaluate` retains conditional moments for only the first TRADE_WINDOWS windows of a set,
    // so a longer arm must be walked in slices of exactly that size. Nothing about the
    // measurement changes with the slice boundary: every window is scored independently of
    // every other, and the slices are consecutive so the arm is covered exactly once.
    for chunk in windows.chunks(TRADE_WINDOWS) {
        template.windows = chunk.to_vec();
        let blocks = pinned_blocks(template);
        let stats = evaluate(
            world.modules(),
            world.deployment_supports(),
            template,
            args.batch_size,
            device,
            true,
            scoring,
            None,
            // The campaign default, deliberately, NOT `chunk.len()`. This module's chunking
            // exists because the retention cap was 256; keeping the cap at 256 and the chunk at
            // `TRADE_WINDOWS` leaves the memorization probe measuring exactly what it measured
            // before the budget became an argument. The `ensure!` below still proves the arm is
            // covered, since a chunk never exceeds the cap.
            TRADE_WINDOWS,
        )?;
        ensure!(
            stats.trade_paths.windows.len() == chunk.len(),
            "arm `{label}` asked for conditional moments on {} windows and got {}; the \
             TRADE_WINDOWS slicing is wrong and the arm would not cover what it claims",
            chunk.len(),
            stats.trade_paths.windows.len()
        );
        for (index, paths) in stats.trade_paths.windows.iter().enumerate() {
            rows.push(ArmRow {
                block: blocks[index],
                month: calendar_month(template.sampler.anchor_ts_ms(&chunk[index])) as i64 as u64,
                nll: stats.window_nll[index],
                nll_conditional: stats.window_nll_conditional[index],
                visit_step: partition.and_then(|p| p.visit_step(slot + index)),
                paths: paths.clone(),
            });
        }
        slot += chunk.len();
        print!("\r  arm `{label}`: {slot}/{} windows scored", windows.len());
        std::io::stdout().flush().ok();
    }
    println!();

    let block_ids: Vec<u64> = rows.iter().map(|row| row.block).collect();
    let month_ids: Vec<u64> = rows.iter().map(|row| row.month).collect();
    let nll: Vec<f64> = rows.iter().map(|row| row.nll).collect();
    let nll_conditional: Vec<f64> = rows.iter().map(|row| row.nll_conditional).collect();
    let paths: Vec<WindowPaths> = rows.iter().map(|row| row.paths.clone()).collect();
    let (finite_bars, dropped_bars) = finite_bar_counts(&paths);

    Ok(Arm {
        label: label.to_owned(),
        exposures,
        windows: rows.len(),
        blocks: block_ids.iter().collect::<BTreeSet<_>>().len(),
        finite_bars,
        dropped_bars,
        nll: block_bootstrap(&nll, &block_ids, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
        nll_conditional: block_bootstrap(
            &nll_conditional,
            &block_ids,
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED,
        ),
        mean_slope: trade_bench::mean_calibration(&paths, &block_ids).mean,
        mean_slope_month: trade_bench::mean_calibration(&paths, &month_ids).mean,
        rows,
    })
}

/// Bars whose `(predicted_mean, realized_log)` pair is usable, and bars that are not.
fn finite_bar_counts(windows: &[WindowPaths]) -> (usize, usize) {
    let mut finite = 0usize;
    let mut dropped = 0usize;
    for window in windows {
        if !window.has_moments() {
            dropped += window.bars();
            continue;
        }
        for (r, m) in window.realized_log().iter().zip(&window.predicted_mean) {
            if r.is_finite() && m.is_finite() {
                finite += 1;
            } else {
                dropped += 1;
            }
        }
    }
    (finite, dropped)
}

impl Arm {
    fn describe(&self) -> String {
        format!(
            "  arm `{}`: {} windows over {} BLOCKS ({} finite bars, {} dropped).\n    \
             nll {:.4} [{:.4}, {:.4}] nats/bar; conditional {:.4} [{:.4}, {:.4}]\n    \
             MZ mean slope, (symbol,month) key: {:.4} [{:.4}, {:.4}] se {:.4} over {} blocks / \
             {} samples; block sd {:.4} against noise sd {:.4} ({} resolved)\n    \
             MZ mean slope, MONTH-ALONE key:    {:.4} [{:.4}, {:.4}] se {:.4} over {} blocks / \
             {} samples",
            self.label,
            self.windows,
            self.blocks,
            self.finite_bars,
            self.dropped_bars,
            self.nll.mean,
            self.nll.ci_low,
            self.nll.ci_high,
            self.nll_conditional.mean,
            self.nll_conditional.ci_low,
            self.nll_conditional.ci_high,
            self.mean_slope.beta,
            self.mean_slope.beta_ci.0,
            self.mean_slope.beta_ci.1,
            self.mean_slope.beta_se,
            self.mean_slope.blocks,
            self.mean_slope.samples,
            self.mean_slope.beta_block_sd,
            self.mean_slope.beta_block_noise_sd,
            self.mean_slope.beta_blocks_resolved,
            self.mean_slope_month.beta,
            self.mean_slope_month.beta_ci.0,
            self.mean_slope_month.beta_ci.1,
            self.mean_slope_month.beta_se,
            self.mean_slope_month.blocks,
            self.mean_slope_month.samples,
        )
    }

    /// Per-key OLS slope of realized on predicted.
    ///
    /// Obtained by calling [`trade_bench::mincer_zarnowitz`] on one key's rows with a single
    /// block id, which returns that key's pooled OLS slope under the module's own non-finite
    /// filter. `draws = 0` skips the bootstrap, which has nothing to resample at one block.
    fn key_slopes(&self, key: &dyn Fn(&ArmRow) -> u64) -> BTreeMap<u64, f64> {
        let mut grouped: BTreeMap<u64, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
        for row in &self.rows {
            if !row.paths.has_moments() {
                continue;
            }
            let cell = grouped.entry(key(row)).or_default();
            for (r, m) in row
                .paths
                .realized_log()
                .iter()
                .zip(&row.paths.predicted_mean)
            {
                cell.0.push(*m);
                cell.1.push(*r);
            }
        }
        grouped
            .into_iter()
            .filter_map(|(id, (mu, realized))| {
                let ids = vec![id; mu.len()];
                let fit = trade_bench::mincer_zarnowitz(&mu, &realized, &ids, 0, BOOTSTRAP_SEED);
                fit.beta.is_finite().then_some((id, fit.beta))
            })
            .collect()
    }

    /// Mean window NLL per block, and the conditional variant.
    fn block_nll(&self) -> BTreeMap<u64, (f64, f64)> {
        let mut sums: BTreeMap<u64, (f64, f64, f64)> = BTreeMap::new();
        for row in &self.rows {
            if !row.nll.is_finite() || !row.nll_conditional.is_finite() {
                continue;
            }
            let cell = sums.entry(row.block).or_default();
            cell.0 += row.nll;
            cell.1 += row.nll_conditional;
            cell.2 += 1.0;
        }
        sums.into_iter()
            .map(|(block, (nll, conditional, count))| (block, (nll / count, conditional / count)))
            .collect()
    }
}

/// The paired contrast: `more - fewer`, differenced WITHIN each shared block.
///
/// Pairing matters by a large factor here. Both arms are drawn from the same stage of the same
/// epoch, so a `(symbol, month)` block usually appears in both, and the block's own regime — its
/// volatility level, its news, its liquidity — is common to the two arms and cancels in the
/// difference. Differencing two independently-blocked intervals would keep that common variance
/// in both and produce a band several times too wide.
fn paired_contrast(more: &Arm, fewer: &Arm) -> PairedContrast {
    let (more_nll, fewer_nll) = (more.block_nll(), fewer.block_nll());
    let mut blocks: Vec<u64> = Vec::new();
    let mut nll_delta: Vec<f64> = Vec::new();
    let mut conditional_delta: Vec<f64> = Vec::new();
    for (block, (a, a_cond)) in &more_nll {
        if let Some((b, b_cond)) = fewer_nll.get(block) {
            blocks.push(*block);
            nll_delta.push(a - b);
            conditional_delta.push(a_cond - b_cond);
        }
    }

    let paired_slope = |key: &dyn Fn(&ArmRow) -> u64| -> (Dispersion, usize) {
        let (a, b) = (more.key_slopes(key), fewer.key_slopes(key));
        let mut ids: Vec<u64> = Vec::new();
        let mut deltas: Vec<f64> = Vec::new();
        for (id, slope) in &a {
            if let Some(other) = b.get(id) {
                ids.push(*id);
                deltas.push(slope - other);
            }
        }
        let count = ids.len();
        (
            block_bootstrap(&deltas, &ids, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
            count,
        )
    };
    let (slope, slope_blocks) = paired_slope(&|row: &ArmRow| row.block);
    let (slope_month, slope_month_blocks) = paired_slope(&|row: &ArmRow| row.month);

    PairedContrast {
        shared_blocks: blocks.len(),
        slope_blocks,
        nll: block_bootstrap(&nll_delta, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
        nll_conditional: block_bootstrap(
            &conditional_delta,
            &blocks,
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED,
        ),
        slope,
        slope_month,
        slope_month_blocks,
    }
}

impl PairedContrast {
    fn describe(&self, more: &Arm, fewer: &Arm) -> String {
        format!(
            "PAIRED one-repetition contrast ({}x minus {}x), differenced within each shared \
             block:\n  \
             nll                     {:+.5} [{:+.5}, {:+.5}] nats/bar over {} shared blocks\n  \
             nll conditional         {:+.5} [{:+.5}, {:+.5}] nats/bar\n  \
             MZ slope, (sym,month)   {:+.5} [{:+.5}, {:+.5}] over {} blocks resolved in BOTH \
             arms\n  \
             MZ slope, MONTH-ALONE   {:+.5} [{:+.5}, {:+.5}] over {} months resolved in BOTH \
             arms\n  \
             A negative nll means the extra repetition FIT those bars better. A negative slope \
             means the extra repetition made the mean MORE over-dispersed, which is the only \
             thing that connects this contrast to the held-out slope decay.",
            more.exposures,
            fewer.exposures,
            self.nll.mean,
            self.nll.ci_low,
            self.nll.ci_high,
            self.shared_blocks,
            self.nll_conditional.mean,
            self.nll_conditional.ci_low,
            self.nll_conditional.ci_high,
            self.slope.mean,
            self.slope.ci_low,
            self.slope.ci_high,
            self.slope_blocks,
            self.slope_month.mean,
            self.slope_month.ci_low,
            self.slope_month.ci_high,
            self.slope_month_blocks,
        )
    }
}

/// Split the extra-exposure arm by how long ago its extra exposure happened.
fn recency_profile(more: &Arm, partition: &Partition) -> Vec<RecencyBucket> {
    let checkpoint_step = partition.checkpoint_step();
    let mut ages: Vec<(usize, usize)> = more
        .rows
        .iter()
        .enumerate()
        .filter_map(|(index, row)| {
            row.visit_step
                .map(|step| (checkpoint_step.saturating_sub(step), index))
        })
        .collect();
    if ages.len() < RECENCY_BUCKETS * 2 {
        return Vec::new();
    }
    ages.sort_unstable();
    let per_bucket = ages.len() / RECENCY_BUCKETS;
    (0..RECENCY_BUCKETS)
        .map(|bucket| {
            let lo = bucket * per_bucket;
            let hi = if bucket + 1 == RECENCY_BUCKETS {
                ages.len()
            } else {
                (bucket + 1) * per_bucket
            };
            let slice = &ages[lo..hi];
            let steps_ago =
                slice.iter().map(|(age, _)| *age as f64).sum::<f64>() / slice.len() as f64;
            let block_ids: Vec<u64> = slice.iter().map(|(_, i)| more.rows[*i].block).collect();
            let nll: Vec<f64> = slice.iter().map(|(_, i)| more.rows[*i].nll).collect();
            let paths: Vec<WindowPaths> = slice
                .iter()
                .map(|(_, i)| more.rows[*i].paths.clone())
                .collect();
            RecencyBucket {
                steps_ago,
                windows: slice.len(),
                blocks: block_ids.iter().collect::<BTreeSet<_>>().len(),
                nll: block_bootstrap(&nll, &block_ids, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
                slope: trade_bench::mean_calibration(&paths, &block_ids).mean,
            }
        })
        .collect()
}

fn recency_lines(buckets: &[RecencyBucket]) -> Vec<String> {
    if buckets.is_empty() {
        return vec!["recency profile: NOT MEASURED - too few issued windows carried a \
                     recoverable visit step to bucket."
            .to_owned()];
    }
    let mut lines = vec!["recency profile within the extra-exposure arm. A flat nll across \
                         buckets is DURABLE memorization; a nll concentrated at the newest \
                         bucket is TRANSIENT retention of recently-visited rows, and the two \
                         have different remedies."
        .to_owned()];
    for bucket in buckets {
        lines.push(format!(
            "  {:>7.0} steps ago: {} windows / {} blocks, nll {:.4} [{:.4}, {:.4}], \
             MZ slope {:.4} [{:.4}, {:.4}]",
            bucket.steps_ago,
            bucket.windows,
            bucket.blocks,
            bucket.nll.mean,
            bucket.nll.ci_low,
            bucket.nll.ci_high,
            bucket.slope.beta,
            bucket.slope.beta_ci.0,
            bucket.slope.beta_ci.1,
        ));
    }
    lines
}

/// Is [`BOOTSTRAP_DRAWS`] enough for the slope's interval on THIS data?
///
/// The point estimate does not move with `draws` — it is not a bootstrap quantity — so any
/// movement in `beta` would be a bug and is charted as a control. What can move is `beta_se` and
/// the percentile width, and the reason to doubt them here is specific: the outer two bins hold
/// 92.384% of the decoded mean's per-bar sampling variance, so the resampled statistic is
/// heavy-tailed and its percentiles converge more slowly than a light-tailed one's.
fn bootstrap_stability(arm: &Arm) -> Vec<StabilityPoint> {
    let mut mu = Vec::new();
    let mut realized = Vec::new();
    let mut blocks = Vec::new();
    for row in &arm.rows {
        if !row.paths.has_moments() {
            continue;
        }
        for (r, m) in row
            .paths
            .realized_log()
            .iter()
            .zip(&row.paths.predicted_mean)
        {
            mu.push(*m);
            realized.push(*r);
            blocks.push(row.block);
        }
    }
    STABILITY_DRAWS
        .iter()
        .map(|draws| {
            let fit =
                trade_bench::mincer_zarnowitz(&mu, &realized, &blocks, *draws, BOOTSTRAP_SEED);
            StabilityPoint {
                draws: *draws,
                beta: fit.beta,
                beta_se: fit.beta_se,
                ci_width: fit.beta_ci.1 - fit.beta_ci.0,
            }
        })
        .collect()
}

fn stability_lines(points: &[StabilityPoint]) -> Vec<String> {
    let mut lines = vec![format!(
        "bootstrap stability of the slope interval in `draws`, configured at BOOTSTRAP_DRAWS = \
         {BOOTSTRAP_DRAWS}. The point estimate must NOT move; only the interval may."
    )];
    let reference = points.last().copied();
    for point in points {
        let drift = reference.map_or(f64::NAN, |r| {
            100.0 * (point.beta_se - r.beta_se) / r.beta_se
        });
        lines.push(format!(
            "  draws {:>5}: beta {:.6}, beta_se {:.6} ({:+.2}% against the {}-draw reference), \
             ci width {:.6}",
            point.draws,
            point.beta,
            point.beta_se,
            drift,
            reference.map_or(0, |r| r.draws),
            point.ci_width,
        ));
    }
    if let (Some(configured), Some(high)) = (
        points.iter().find(|p| p.draws == BOOTSTRAP_DRAWS).copied(),
        reference,
    ) {
        let drift = 100.0 * (configured.beta_se - high.beta_se).abs() / high.beta_se;
        let moved = points
            .iter()
            .map(|p| (p.beta - configured.beta).abs())
            .fold(0.0f64, f64::max);
        lines.push(format!(
            "  VERDICT: beta_se at the configured {BOOTSTRAP_DRAWS} draws differs from the \
             {}-draw reference by {drift:.2}%, and the point estimate moved by {moved:.2e} \
             across every draw count (must be 0). The module's doc claims percentiles are \
             stable to about a percent of the interval width; that claim {} on this data.",
            high.draws,
            if drift <= 5.0 {
                "SURVIVES"
            } else {
                "FAILS, and every interval taken at the configured draw count is optimistic \
                 about its own reproducibility"
            }
        ));
    }
    lines
}

/// The train/val gap, paired within symbol.
///
/// Both draws are made by the same constructor at the same seed, so the two window sets carry
/// overlapping symbol sets. Differencing within symbol removes the cross-sectional component —
/// a name's own liquidity and volatility level — while leaving the calendar component, which
/// cannot be removed because the splits are calendar-disjoint by construction. So this is a
/// STRICTLY less contaminated version of the same contaminated quantity, never a clean one.
fn symbol_paired_difference(
    train: &PinnedSet,
    train_nll: &[f64],
    heldout: &PinnedSet,
    heldout_nll: &[f64],
) -> (Dispersion, usize) {
    let mean_by_symbol = |set: &PinnedSet, values: &[f64]| -> BTreeMap<u32, (f64, f64)> {
        let mut sums: BTreeMap<u32, (f64, f64)> = BTreeMap::new();
        for (window, value) in set.windows.iter().zip(values) {
            if !value.is_finite() {
                continue;
            }
            let cell = sums.entry(window.symbol).or_default();
            cell.0 += *value;
            cell.1 += 1.0;
        }
        sums
    };
    let seen = mean_by_symbol(train, train_nll);
    let held = mean_by_symbol(heldout, heldout_nll);
    let mut deltas = Vec::new();
    let mut symbols = Vec::new();
    for (symbol, (train_sum, train_count)) in &seen {
        if let Some((held_sum, held_count)) = held.get(symbol) {
            deltas.push(held_sum / held_count - train_sum / train_count);
            symbols.push(*symbol as u64);
        }
    }
    let count = deltas.len();
    (
        block_bootstrap(&deltas, &symbols, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED),
        count,
    )
}

/// The labelled verdict, printed and identical to what the report carries.
pub(super) fn verdict_lines(
    spine: &[GapPoint],
    more: &Arm,
    fewer: &Arm,
    contrast: &PairedContrast,
) -> Vec<String> {
    let mut lines = vec![
        String::new(),
        "=== MEMORIZATION VERDICT ===".to_owned(),
        format!(
            "EPOCH SPINE (CONTAMINATED, supporting only). Held-out minus train NLL at each pass \
             count: {}",
            spine
                .iter()
                .map(|p| format!("{}@{} {:+.4}", p.label, p.step, p.gap))
                .collect::<Vec<_>>()
                .join("  ")
        ),
        "  This trajectory cannot discriminate. Train and val are calendar-disjoint so the LEVEL \
         mixes regime, and lr_mult is affine in step past the plateau so the TRAJECTORY mixes \
         passes with learning-rate-driven loss of implicit regularization. Rank 1."
            .to_owned(),
    ];

    let excludes_zero = |d: &Dispersion| {
        d.ci_low.is_finite() && d.ci_high.is_finite() && (d.ci_high < 0.0 || d.ci_low > 0.0)
    };
    let nll_resolved = excludes_zero(&contrast.nll);
    let slope_resolved = excludes_zero(&contrast.slope);
    let slope_month_resolved = excludes_zero(&contrast.slope_month);

    lines.push(format!(
        "ONE-REPETITION CONTRAST (MODEL-FREE, randomized by construction). {}x against {}x, same \
         weights, same lr, same ramp stage, same context, same conditioning depth; the split is \
         a uniformly random subset because build_layout shuffles each stage globally.",
        more.exposures, fewer.exposures
    ));
    lines.push(format!(
        "  nll delta   {:+.5} [{:+.5}, {:+.5}] over {} shared blocks -> {}",
        contrast.nll.mean,
        contrast.nll.ci_low,
        contrast.nll.ci_high,
        contrast.shared_blocks,
        if !nll_resolved {
            "UNRESOLVED: the interval spans zero, so one extra exposure has no detectable effect \
             on fit at this power"
        } else if contrast.nll.mean < 0.0 {
            "RESOLVED: the extra exposure FITS those bars better. Memorization OCCURS."
        } else {
            "RESOLVED WITH THE WRONG SIGN: the extra exposure fits those bars WORSE, which no \
             memorization account predicts"
        }
    ));
    lines.push(format!(
        "  slope delta {:+.5} [{:+.5}, {:+.5}] over {} blocks; month-alone key {:+.5} [{:+.5}, \
         {:+.5}] over {} months -> {}",
        contrast.slope.mean,
        contrast.slope.ci_low,
        contrast.slope.ci_high,
        contrast.slope_blocks,
        contrast.slope_month.mean,
        contrast.slope_month.ci_low,
        contrast.slope_month.ci_high,
        contrast.slope_month_blocks,
        match (slope_resolved, slope_month_resolved) {
            (true, true) if contrast.slope.mean < 0.0 =>
                "RESOLVED NEGATIVE under BOTH keys: one repetition demonstrably over-disperses \
                 the mean",
            (true, true) => "RESOLVED POSITIVE under both keys",
            (true, false) =>
                "RESOLVED under the (symbol,month) key ONLY. The month-alone key spans zero, and \
                 month-alone is the conservative key against same-instant cross-symbol \
                 dependence, so this is NOT safe to call resolved",
            (false, _) => "UNRESOLVED",
        }
    ));
    lines.push(
        match (
            nll_resolved && contrast.nll.mean < 0.0,
            slope_resolved && slope_month_resolved,
        ) {
            (true, true) if contrast.slope.mean < 0.0 => "CONCLUSION: memorization OCCURS AND one \
                 repetition moves the mean slope in the SAME DIRECTION as the held-out decay, \
                 under both block keys. Memorization EXPLAINS the slope movement, and the \
                 recipe's three passes are the mechanism."
                .to_owned(),
            (true, true) => "CONCLUSION: memorization occurs but the slope moves the OTHER way, \
                 so memorization does NOT explain the held-out decay."
                .to_owned(),
            (true, false) => format!(
                "CONCLUSION: memorization OCCURS - one extra exposure improves fit by {:+.5} \
                 nats/bar with the interval excluding zero - but its effect on the mean SLOPE is \
                 NOT resolved at {} blocks under both keys. Memorization is ESTABLISHED and its \
                 link to the held-out slope decay is NOT. Do not report the decay as explained.",
                contrast.nll.mean, contrast.slope_blocks
            ),
            (false, _) => "CONCLUSION: UNRESOLVED. One additional exposure produced no detectable \
                 change in fit, which is evidence AGAINST classical multi-epoch memorization \
                 being the live mechanism at this point in the run - but only at this power, and \
                 the number of BLOCKS, not windows, is the denominator that set it."
                .to_owned(),
        },
    );
    lines.push(format!(
        "  Power: the smaller arm carried {} blocks over {} windows. Levels: slope {:.4} at {}x \
         against {:.4} at {}x. Finite bars {} against {}, dropped {} against {} - a mismatch \
         here would mean a non-finite predicted mean, which is a defect and not a filter.",
        more.blocks.min(fewer.blocks),
        more.windows.min(fewer.windows),
        more.mean_slope.beta,
        more.exposures,
        fewer.mean_slope.beta,
        fewer.exposures,
        more.finite_bars,
        fewer.finite_bars,
        more.dropped_bars,
        fewer.dropped_bars,
    ));
    lines
}

// ---------------------------------------------------------------------------
// Row dump
// ---------------------------------------------------------------------------

/// Field order of one binary record, written into the header so a reader never guesses.
const ROW_FIELDS: [&str; 13] = [
    "u32 window_index",
    "u32 symbol_index",
    "u32 bar_index",
    "u64 block",
    "u8 population_flags (bit0 traded_prefix, bit1 disjoint_fit_slice)",
    "u8 all_finite",
    "f64 predicted_mean",
    "f64 predicted_var",
    "f64 outer_mass",
    "f64 outer_signed",
    "f64 trimmed_mean",
    "f64 trimmed_var",
    "f64 realized_log",
];

/// Which slice of a drawn set a dump covers, and which population flags its rows carry.
#[derive(Clone, Copy, Debug)]
pub enum RowPopulation {
    TradedPrefix,
    DisjointFitSlice,
}

impl RowPopulation {
    fn tag(self) -> &'static str {
        match self {
            Self::TradedPrefix => "traded_prefix",
            Self::DisjointFitSlice => "disjoint_fit_slice",
        }
    }

    fn flags(self) -> u8 {
        match self {
            Self::TradedPrefix => 0b01,
            Self::DisjointFitSlice => 0b10,
        }
    }
}

/// Stream one population's per-bar rows to disk.
///
/// Written as a fixed-layout little-endian body plus a JSON header, and STREAMED: rows are never
/// accumulated, in any language. `trimmed_mean` / `trimmed_var` are emitted rather than any
/// single decode's mean, because `OuterBar::redecoded` is a closed form in
/// `(mass, signed, interior_mean, interior_var)`, so those four reconstruct EVERY decode arm at
/// ANY decode pair, offline, forever. Emitting one arm's `mu` would lock the artifact to one
/// convention; emitting the sufficient statistics locks it to none. `predicted_mean` and
/// `predicted_var` ride along because they are what the pipeline reads TODAY, and the per-row
/// `all_finite` flag exists so the INTERSECTION of usable rows across checkpoints is
/// reconstructible without a rerun — `mincer_zarnowitz` filters on `x`, so two checkpoints can
/// silently be fitted on different row sets and no cross-checkpoint pairing would be valid.
///
/// The population is written INTO the artifact. Every measurement failure this campaign has
/// suffered was a statistic correct over a population someone assumed.
#[allow(clippy::too_many_arguments)]
pub fn write_row_dump(
    dir: &Path,
    stem: &str,
    population: RowPopulation,
    world: &BarWorldModel,
    step: usize,
    split: Split,
    context: i64,
    windows_drawn: usize,
    window_refs: &[WindowRef],
    blocks: &[u64],
    paths: &[WindowPaths],
) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let base = dir.join(format!("{stem}.{}.memprobe_rows", population.tag()));
    let body_path = base.with_extension("bin");
    let mut body = BufWriter::with_capacity(
        1 << 16,
        File::create(&body_path)
            .with_context(|| format!("failed to create {}", body_path.display()))?,
    );

    let mut bars = 0usize;
    let mut skipped = 0usize;
    let mut record = [0u8; ROW_BYTES];
    for (index, window) in paths.iter().enumerate() {
        if !window.has_decomposition() || !window.has_moments() {
            skipped += 1;
            continue;
        }
        let reference = window_refs.get(index).copied().unwrap_or(WindowRef {
            symbol: u32::MAX,
            bar_index: u32::MAX,
        });
        let block = blocks.get(index).copied().unwrap_or(u64::MAX);
        let realized = window.realized_log();
        for bar in 0..window.bars() {
            let values = [
                window.predicted_mean[bar],
                window.predicted_var[bar],
                window.outer_mass[bar],
                window.outer_signed[bar],
                window.trimmed_mean[bar],
                window.trimmed_var[bar],
                realized[bar],
            ];
            record[0..4].copy_from_slice(&(index as u32).to_le_bytes());
            record[4..8].copy_from_slice(&reference.symbol.to_le_bytes());
            // The TARGET bar, not the anchor: a window anchored at `a` predicts `a+1 ..= a+C`.
            record[8..12]
                .copy_from_slice(&reference.bar_index.saturating_add(1 + bar as u32).to_le_bytes());
            record[12..20].copy_from_slice(&block.to_le_bytes());
            record[20] = population.flags();
            record[21] = u8::from(values.iter().all(|value| value.is_finite()));
            for (slot, value) in values.iter().enumerate() {
                let at = 22 + slot * 8;
                record[at..at + 8].copy_from_slice(&value.to_le_bytes());
            }
            body.write_all(&record)
                .with_context(|| format!("failed writing {}", body_path.display()))?;
            bars += 1;
        }
    }
    body.into_inner()
        .with_context(|| format!("failed closing {}", body_path.display()))?
        .sync_all()
        .with_context(|| format!("failed syncing {}", body_path.display()))?;

    let header = serde_json::json!({
        "format_version": ROW_DUMP_FORMAT_VERSION,
        "checkpoint": stem,
        "checkpoint_step": step,
        "lineage_sha256": world.lineage_sha256(),
        "split": format!("{split:?}"),
        "eval_window_seed": EVAL_WINDOW_SEED,
        "context": context,
        "windows_drawn": windows_drawn,
        "population": population.tag(),
        "windows": paths.len(),
        "windows_without_moments": skipped,
        "bars": bars,
        "record_fields": ROW_FIELDS,
        "record_bytes": ROW_BYTES,
        "byte_order": "little-endian",
    });
    let header_path = base.with_extension("json");
    std::fs::write(&header_path, serde_json::to_vec_pretty(&header)?)
        .with_context(|| format!("failed to write {}", header_path.display()))?;
    println!(
        "  rows: {bars} bars over {} windows ({skipped} without moments) -> {} ({} MiB)",
        paths.len(),
        body_path.display(),
        bars * ROW_BYTES / (1 << 20),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::pretrain_reports::write_mem_probe;
    use super::*;
    use shared::report::{read_report, ReportKind};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::fs;
    use std::path::PathBuf;

    static SCRATCH: AtomicU64 = AtomicU64::new(0);

    fn scratch_dir(name: &str) -> PathBuf {
        let unique = SCRATCH.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "mem_probe_{name}_{}_{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn dispersion(mean: f64) -> Dispersion {
        Dispersion {
            mean,
            se: 0.25 * mean.abs().max(1.0e-4),
            ci_low: mean - 0.5 * mean.abs().max(1.0e-4),
            ci_high: mean + 0.5 * mean.abs().max(1.0e-4),
            blocks: 96,
            samples: 1024,
        }
    }

    /// Only the members the charts read are given values; everything else stays NaN, so a
    /// future series that starts reading an unset member shows up as a blank line rather than
    /// as a plausible number.
    fn fit(beta: f64) -> MzFit {
        MzFit {
            beta,
            beta_se: 0.02,
            beta_ci: (beta - 0.04, beta + 0.04),
            blocks: 96,
            samples: 2048,
            ..MzFit::nan()
        }
    }

    fn arm(label: &str, exposures: usize, nll: f64, slope: f64) -> Arm {
        Arm {
            label: label.to_owned(),
            exposures,
            windows: 1024,
            blocks: 96,
            finite_bars: 1024 * 2048,
            dropped_bars: 0,
            nll: dispersion(nll),
            nll_conditional: dispersion(nll - 0.010),
            mean_slope: fit(slope),
            mean_slope_month: fit(slope + 0.012),
            // The writer reads no per-window row, and building one would need a `WindowPaths`
            // out of a model pass. Nothing this test asserts is a function of the rows.
            rows: Vec::new(),
        }
    }

    /// The writer named in `pretrain_reports::tests::CYCLE_EXEMPT` for all four `memprobe_*`
    /// bases.
    ///
    /// The exemption is honest only if something EXECUTES the writer: this module shipped with
    /// four registered bases, a stated reason, and no writer at all, in a file that was not in
    /// the module tree. CPU-only and corpus-free by construction — the writer consumes measured
    /// summaries, never a model — so it seeds no RNG, torch or otherwise.
    #[test]
    fn the_mem_probe_writes_every_registered_base() {
        let gap_points: Vec<GapPoint> = [
            ("pretrain_epoch_0_ctx2048", 10_364usize, 3.480, 3.512),
            ("pretrain_epoch_1_ctx2048", 20_729, 3.401, 3.498),
            ("pretrain_best", 30_000, 3.352, 3.494),
        ]
        .into_iter()
        .map(|(label, step, train_nll, heldout_nll)| GapPoint {
            label: label.to_owned(),
            step,
            train_nll,
            heldout_nll,
            train_nll_conditional: train_nll - 0.020,
            heldout_nll_conditional: heldout_nll - 0.018,
            gap: heldout_nll - train_nll,
            gap_conditional: (heldout_nll - 0.018) - (train_nll - 0.020),
            symbol_paired_gap: dispersion(0.6 * (heldout_nll - train_nll)),
            symbols_paired: 1_742,
        })
        .collect();

        let seen_more = arm("seen 3x - issued in epoch 2", 3, 3.344, 0.681);
        let seen_fewer = arm("seen 2x - not yet issued in epoch 2", 2, 3.351, 0.724);
        let contrast = PairedContrast {
            shared_blocks: 88,
            slope_blocks: 71,
            nll: dispersion(-0.007),
            nll_conditional: dispersion(-0.006),
            slope: dispersion(-0.043),
            slope_month: dispersion(-0.038),
            slope_month_blocks: 11,
        };
        let recency: Vec<RecencyBucket> = (0..RECENCY_BUCKETS)
            .map(|bucket| RecencyBucket {
                steps_ago: 236.0 * (bucket as f64 + 0.5),
                windows: 204,
                blocks: 61,
                nll: dispersion(3.340 + 0.004 * bucket as f64),
                slope: fit(0.670 + 0.006 * bucket as f64),
            })
            .collect();
        // `beta` is IDENTICAL at every draw count, which is the invariant the panel exists to
        // display: only `beta_se` and the percentile width may move with `draws`.
        let stability: Vec<StabilityPoint> = STABILITY_DRAWS
            .iter()
            .enumerate()
            .map(|(index, draws)| StabilityPoint {
                draws: *draws,
                beta: 0.681,
                beta_se: 0.030 - 0.002 * index as f64,
                ci_width: 0.118 - 0.008 * index as f64,
            })
            .collect();

        let dir = scratch_dir("bases");
        write_mem_probe(
            &dir,
            "fixture - 4096 pinned windows per split, 1024 windows per arm, context 2048",
            &gap_points,
            &seen_more,
            &seen_fewer,
            &contrast,
            &recency,
            &stability,
        )
        .expect("all four panels write");

        for base in [
            "memprobe_epoch_spine",
            "memprobe_one_repetition",
            "memprobe_recency",
            "memprobe_bootstrap_stability",
        ] {
            assert!(
                shared::report::PRETRAIN_REPORT_BASES.contains(&base),
                "{base} must be registered in shared::report::PRETRAIN_REPORT_BASES or the TUI \
                 never scans for it"
            );
            let path = dir.join(format!("{base}.report.bin"));
            assert!(path.exists(), "{base} was never written");
            let report = read_report(&path).expect("the report reads back");
            let ReportKind::MultiLine { series } = &report.kind else {
                panic!("{base} must be a MultiLine chart");
            };
            assert!(
                series
                    .iter()
                    .any(|line| line.values.iter().any(|value| value.is_finite())),
                "{base} carries no finite value, so it is a blank panel"
            );
        }

        // The contamination has to be legible from the CHART, and a title cannot carry it: the
        // TUI's `normalize_title` lowercases everything after each word's first letter, so
        // emphasis survives only in a series legend. This is the assertion that keeps the
        // qualification from migrating back into a title.
        let spine = read_report(&dir.join("memprobe_epoch_spine.report.bin")).unwrap();
        let ReportKind::MultiLine { series } = &spine.kind else {
            panic!("the spine must be a MultiLine chart");
        };
        assert!(
            series
                .iter()
                .any(|line| line.label.contains("CONTAMINATED")
                    && line.label.contains("NOT a discriminator")),
            "the pooled gap's own label must say it is contaminated and not a discriminator: {:?}",
            series.iter().map(|line| &line.label).collect::<Vec<_>>()
        );
        assert!(
            series
                .iter()
                .any(|line| line.label.contains("STRICTLY LESS CONTAMINATED, NEVER CLEAN")),
            "the symbol-paired gap must say it is less contaminated and never clean"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
