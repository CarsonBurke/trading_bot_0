//! Discrete distributional next-bar pretraining.
//!
//! The world model factorizes `p(bar_{t+1} | bar_{<=t})` into five categorical
//! factors over equal-mass bins (see [`crate::torch::bar_dist`]) and is trained by
//! maximum likelihood. Two auxiliary terms make the *latent* dynamics usable for
//! planning without ever displacing the likelihood as the learning signal:
//!
//! ```text
//! L = nll_bar                                            (primary, attached)
//!   + lambda_dyn * smooth_l1(z_hat_{t+k}, sg[h_{t+k}])   (NextLat, stop-grad target)
//!   + lambda_kl  * KL(sg[p(.|h_{t+k})] || p(.|z_hat_{t+k}))
//! ```
//!
//! `z_hat` is a recursive `--dyn-horizon`-step rollout of
//! [`crate::torch::world_model::BarDynamics`] over
//! teacher-forced bars. The transformer runs exactly once per optimizer step: every
//! horizon reuses the same belief sequence, shifted. Emission-head parameters are
//! detached in both dynamics branches, so the KL shapes the latent and never the
//! decoder.
//!
//! There is no SIGReg term and no latent-target JEPA term. Isotropy and effective
//! rank are diagnostics only.
//!
//! Optimization follows the modded-nanogpt speedrun record: NorMuon with Polar
//! Express orthogonalization on every 2-D weight, AdamW on embeddings, the five
//! emission heads and every scalar gate; cautious weight decay that is quadratic in
//! the learning rate; **no** learning-rate warmup and **no** gradient clipping.
//! Momentum warmup replaces LR warmup, and orthogonalization replaces clipping.

use anyhow::{anyhow, ensure, Context, Result};
use nvml_wrapper::Nvml;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Instant;
use tch::{autocast, nn, Device, Kind, Reduction, Tensor};

use crate::torch::bar_dist::{
    bar_categorical_kl, bar_crps_from_logits, bar_nll_decomposition, bar_nll_from_logits,
    bar_nll_terms, bar_pit_from_logits, BarScoring, BarSupports, BarSupportsProvenance, BAR_DOF,
    BAR_DOF_NAMES, BAR_EMISSION_ADAMW_NAME_SUBSTRINGS, BAR_LABEL_SIGMA_RATIO, DOF_R, DOF_S,
    DOF_U, DOF_V, DOF_W, NUM_BAR_BINS,
};
use crate::torch::cuda::cfg::configure_cuda;
use crate::torch::dataset::{
    iso_ms, mix64, BarBatch, BarCorpus, BarSampler, Split, WindowRef, BAR_TIME_CARDINALITY,
    BAR_TIME_CONDITIONING,
};
use crate::torch::load::load_var_store_partial;
use crate::torch::optim::muon::{Muon, MuonConfig, Orthogonalizer, StepKind, DEFAULT_NS_STEPS};
use crate::torch::world_model::{
    bar_adamw_embedding_substrings, bar_adamw_scalar_substrings,
    bar_muon_down_projection_substrings, bar_muon_name_substrings, world_model_metadata_path,
    world_model_supports_path, BarModules, BarSupportSet, BarTrainingProvenance, BarWorldModel,
    BarWorldModelMetadata, RolloutMode, BAR_ARCHITECTURE, BAR_LAYERS, BAR_MAX_CONTEXT,
    BAR_MODEL_DIM,
};
use shared::{paths::RUNS_PATH, run_dir::RunDir};

use super::optimizer_glue::named_trainable_variables;
use super::pretrain_reports::{
    belief_effective_rank, EpochMetrics, HeldOutBaselines, PitHistogram, PretrainReporter,
    SnapshotInput, StepMetrics, TestBattery, AUX_SHARE_WARN, AUX_SHARE_WARN_STREAK,
    ROLLOUT_HORIZONS, SNAPSHOT_SAMPLES,
};
use super::pretrain_stats::{
    block_bootstrap, calendar_month, window_scores_path, Dispersion, WindowScore, WindowScores,
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, WINDOW_SCORES_FORMAT_VERSION,
};

/// Context length at the start of the ramp. Also the fixed context of the
/// across-run diagnostic evaluation, which must never vary between runs.
pub const BAR_CONTEXT_RAMP_START: i64 = 896;

/// Number of ramp stages. Batch size and context both step at each stage boundary,
/// which sits at an equal fraction of total steps.
const RAMP_STAGES: usize = 3;
/// Batch-size multipliers per stage.
const BATCH_RAMP: [usize; RAMP_STAGES] = [1, 2, 3];
/// Fraction of the projected activation INCREMENT that must be free on top of the increment
/// itself before the ramp is allowed to step up.
///
/// The caching allocator cannot reuse a block of the wrong shape, so growing the context
/// fragments the pool before it settles: the transient peak is materially above the steady
/// state. Half again is measured to cover it on this model.
const RAMP_MEMORY_MARGIN: f64 = 0.5;
/// VRAM that must still be free after the projected increment.
///
/// The card is shared with the user's own jobs. Growing until the device is exactly full
/// hands the next OOM to whichever process allocates next, which is not a scheduling policy.
const RAMP_MEMORY_RESERVE_BYTES: u64 = 1 << 29;
/// Optimizer steps into a ramp stage before its activation footprint is measured. The
/// allocator's pool is warm by then, so the reading reflects the steady state rather than
/// the first step's transient.
const RAMP_PROBE_AFTER_STEPS: usize = 4;
/// Fraction of training spent at the flat learning-rate plateau.
const LR_PLATEAU_FRACTION: f64 = 0.40;
/// Terminal learning-rate multiplier reached by the linear decay.
const LR_FLOOR_MULTIPLIER: f64 = 0.15;
/// Momentum warmup stands in for learning-rate warmup: there is no LR warmup.
const MOMENTUM_START: f64 = 0.85;
const MOMENTUM_PEAK: f64 = 0.95;
const MOMENTUM_WARMUP_STEPS: usize = 300;
const MOMENTUM_COOLDOWN_STEPS: usize = 50;

const NORMUON_LR: f64 = 0.023;
const NORMUON_BETA2: f64 = 0.9;
const NORMUON_WEIGHT_DECAY: f64 = 1.2;
/// Extra learning-rate multiplier on MLP down-projections.
const NORMUON_DOWN_PROJECTION_LR_MULT: f64 = 2.0;

const ADAMW_LR: f64 = 0.008;
const ADAMW_EPS: f64 = 1e-10;
const ADAMW_WEIGHT_DECAY: f64 = 0.005;
/// Betas for scalars and gates.
const ADAMW_SCALAR_BETAS: (f64, f64) = (0.9, 0.99);
/// Betas for embedding tables and the emission heads.
const ADAMW_TABLE_BETAS: (f64, f64) = (0.5, 0.95);
/// Extra learning-rate multiplier on the learned residual lambdas. The post lambdas
/// stay at 1.0x, matching the reference's separate `resid_lambdas` / `post_lambdas`
/// groups (modded-nanogpt `train_gpt.py:2035-2036`).
const ADAMW_RESID_LAMBDA_LR_MULT: f64 = 5.0;
/// Weight-decay multiplier on the embedding tables and the emission heads. Without it
/// a decay that is quadratic in the learning rate is inert: `lr*lr*wd` is 3.2e-7 per
/// step, which moves a weight by 0.3% over ten thousand steps
/// (`train_gpt.py:2033`, `:2038`).
const ADAMW_TABLE_WEIGHT_DECAY_MULT: f64 = 150.0;

/// Ancestral samples backing the directional-accuracy diagnostic.
const DIRECTION_SAMPLES: i64 = 8;
/// Realized continuation length handed to the rollout diagnostics and the candle
/// snapshot writer. Must cover the longest reported rollout horizon.
const SNAPSHOT_HORIZON: i64 = 64;
/// Maximum belief rows fed to the effective-rank diagnostic, which is `O(D^2 * N)`.
const EFFECTIVE_RANK_ROWS: i64 = 8192;
/// Tolerance, in nats per bar, for the reloaded-checkpoint verification.
const PROMOTION_ROUNDTRIP_TOLERANCE: f64 = 1e-4;

/// Seed of every pinned evaluation set and of the randomized PIT. A CAMPAIGN CONSTANT,
/// deliberately not `--seed`.
///
/// The pinned promotion, diagnostic and test windows used to be drawn with the training
/// seed, which made the two inseparable: changing `--seed` to obtain a training replicate
/// also resampled all 4096 promotion windows, so the run-to-run noise floor could never be
/// measured and every ablation delta was read against zero. Splitting them means seed
/// replicates measure exactly the thing they are supposed to — training stochasticity — on
/// a fixed bench, and every run in the campaign is paired on identical windows, which is
/// what takes the minimum detectable effect from ~0.41 nats down to ~0.04-0.09.
///
/// Changing this value invalidates cross-run comparability for the whole campaign.
const EVAL_WINDOW_SEED: u64 = 0xE7A1_5E7D_0001;

/// What promotion compares, recorded into every checkpoint's metadata and folded into its
/// lineage hash.
///
/// `nll_bar_conditional`, not the raw `nll_bar`: 0.690 nats of the raw sum's gain over the
/// calibrated marginal is an arithmetic identity of the encoder — `s == 0` forces
/// `u = v = 0.5`, and the free gain is exactly the binary entropy of the flat-bar rate,
/// `2 * H_b(0.109327) = 0.690`. A metric where a fifth of the apparent progress is a
/// tautology is not a selection metric. The conditional form scores `u` and `v` only on
/// bars with `s != 0` and leaves every other factor alone, so it removes the identity
/// without discarding signal and keeps the five-factor variance advantage that gives the
/// paired bench its 0.04-0.09 nat resolution.
///
/// The weights stay all-ones on purpose. The asymmetry the campaign needs is not a weight
/// vector — a free parameter nobody can defend — but [`SELECTION_GUARD_DOF`]: a model must
/// not buy an aggregate win by regressing on `r`, the only DOF that determines P&L and the
/// one with an order of magnitude less headroom than the shape factors it would be traded
/// against.
const SELECTION_METRIC: &str =
    "nll_bar_conditional on the pinned val set at the deployed context (the five per-DOF \
     soft-CE terms, with u and v scored only on bars where s != 0 so the s=0 => u=v=0.5 \
     encoding identity is never counted as skill), gated by a non-regression guard on \
     nll_dof[r] measured as a PAIRED difference against the incumbent on identical windows";
const SELECTION_WEIGHTS: [f64; BAR_DOF] = [1.0; BAR_DOF];
/// The factor the guard protects.
const SELECTION_GUARD_DOF: usize = DOF_R;
/// Standard errors of the PAIRED `r` difference a candidate may drift before promotion is
/// refused. At 1.0 any regression the bench can actually resolve blocks the promotion, and
/// one it cannot resolve is by definition not evidence.
const SELECTION_GUARD_SE_MULTIPLE: f64 = 1.0;

#[derive(Clone, Debug)]
pub struct PretrainArgs {
    /// Optional checkpoint to initialize from. Weights only; training restarts at
    /// step zero with a fresh optimizer and a freshly derived schedule.
    pub weights: Option<String>,
    pub run: Option<String>,
    /// One epoch is one pass worth of BAR-TOKENS over the training split, not a guaranteed
    /// pass over every unique bar: the context ramp gives each stage its own anchor list and
    /// splits the token budget unevenly across them. `pretrain_stage_coverage` charts what
    /// each stage actually visited.
    pub epochs: usize,
    /// Override the derived total step count. Diagnostic use only: it decouples the
    /// schedule from the corpus.
    pub steps: Option<usize>,
    /// Batch size at ramp stage 0. Stages 1 and 2 use 2x and 3x this.
    pub batch_size: usize,
    /// Seeds the TRAINING sampler, support fitting, and the torch and CUDA RNGs — and
    /// nothing else. The pinned evaluation windows and the PIT draws are pinned by
    /// [`EVAL_WINDOW_SEED`] instead, so a seed replicate measures training stochasticity on
    /// an unchanged bench rather than confounding it with evaluation-set noise.
    pub seed: u64,
    pub data_dir: String,
    pub resolution_secs: u32,
    /// Symbols with fewer bars than this are dropped from the corpus.
    pub min_bars: usize,
    /// Bars drawn from the training split to fit the bin supports.
    pub support_samples: usize,
    /// Scoring rule the objective AND every reported baseline are expressed in.
    ///
    /// One knob, threaded everywhere: the loss, the uniform / marginal /
    /// conditional-marginal / encoding-identity / floor reference lines, the banner, the
    /// charts, the per-window vectors and the checkpoint lineage all read this. The three
    /// modes are not comparable in absolute nats, so `pretrain-compare` refuses to pair two
    /// runs that disagree.
    pub scoring: BarScoring,
    /// Recursive dynamics rollout depth.
    pub dyn_horizon: usize,
    /// Weight on the NextLat term. `dyn` is summed over the feature axis, so this is
    /// commensurate with `nll`; see the CLI's help for why the default is `1e-2`.
    pub lambda_dyn: f64,
    pub lambda_kl: f64,
    /// Held-out windows in each pinned evaluation set. Pinned by [`EVAL_WINDOW_SEED`], so
    /// they are identical across runs, seeds and ablations.
    pub validation_windows: usize,
    /// Fixed context of the across-run diagnostic evaluation.
    pub diagnostic_context: i64,
    /// Pinned windows carried into the candle-rollout snapshot reports.
    pub snapshot_windows: usize,
    /// Validate every N optimizer steps. Validation also always runs at every epoch
    /// boundary and at the end of the run.
    pub validate_every: usize,
    /// Write a step-tagged checkpoint every N optimizer steps (0 disables).
    pub checkpoint_every: usize,
    /// Print a training line every N optimizer steps.
    pub log_every: usize,
    /// Pin the two split instants as `<b0>,<b1>` epoch millis.
    ///
    /// `None` means the campaign pin, [`crate::data::ingest::PINNED_SPLIT_BOUNDS`], unless
    /// `derive_split_bounds` asks for the live percentiles instead. The corpus is live and
    /// the bounds are percentiles of its trading-time axis, so a derived boundary moves
    /// with the data: after the survivorship expansion it lands 26 days EARLIER, which
    /// drops universe-ranking sessions into validation and reopens the selection leak.
    /// Pinning is the default for exactly that reason.
    pub split_bounds: Option<(i64, i64)>,
    /// Re-derive the split instants from the current corpus instead of using the campaign
    /// pin. Diagnostic use only: two runs that each derive their own boundary are not
    /// comparable, and `pretrain-compare` refuses to pair them.
    pub derive_split_bounds: bool,
    /// Explicit path to the bin supports, instead of the corpus default. Use it to point a
    /// whole campaign at one frozen artifact.
    pub supports: Option<String>,
    /// Accept cached supports whose provenance does not match this corpus.
    ///
    /// Freezing the supports across a campaign is the RIGHT call — they define the output
    /// space and therefore the `nll_bar` scale, so refitting mid-campaign makes every prior
    /// number incomparable. Without this flag a mismatch is a hard error, so the freeze is a
    /// recorded decision rather than a filesystem accident; with it, the fact is printed and
    /// written into the checkpoint metadata.
    pub freeze_supports: bool,
    /// Keep only symbols whose cached median dollar volume clears this floor. `0.0` uses
    /// every file on disk.
    ///
    /// The restriction is applied AFTER the split instants are derived from the full symbol
    /// set, so both arms of a "does the thin tail help or hurt?" ablation are scored over
    /// the same wall-clock held-out window and their `nll_bar` are commensurable. It does
    /// change the corpus fingerprint, because the symbol set is part of what decides which
    /// bars a split contains — so a restricted run will need `--freeze-supports` to reuse
    /// the unrestricted fit, which is the correct choice for comparability and is recorded.
    pub min_dollar_volume: f64,
}

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

/// Context length at a ramp stage, 64-aligned for the attention kernels.
fn stage_context(stage: usize) -> i64 {
    debug_assert!(stage < RAMP_STAGES);
    let span = BAR_MAX_CONTEXT - BAR_CONTEXT_RAMP_START;
    let raw = BAR_CONTEXT_RAMP_START + span * stage as i64 / (RAMP_STAGES as i64 - 1);
    raw - raw % 64
}

/// Mean bar-tokens per step per unit of base batch size, averaged over the ramp.
fn ramp_bars_per_batch_unit() -> f64 {
    let total: i64 = (0..RAMP_STAGES)
        .map(|stage| BATCH_RAMP[stage] as i64 * stage_context(stage))
        .sum();
    total as f64 / RAMP_STAGES as f64
}

/// Resolved training schedule. Every per-step quantity is a pure function of the
/// step index, so a resumed or replayed run cannot drift.
///
/// `batch_ramp` starts at [`BATCH_RAMP`] and is the ONE place a memory hold is applied:
/// [`Trainer::hold_batch_if_short_of_vram`] lowers a stage's multiplier in place, and
/// [`Self::batch`], [`Self::bars_per_step`] and [`Self::lr_multiplier`] all read it, so the
/// learning-rate plateau bump can never describe a batch the run is not using.
#[derive(Clone, Copy, Debug)]
struct Schedule {
    total_steps: usize,
    base_batch: usize,
    /// Batch-size multiplier ACTUALLY used at each ramp stage.
    batch_ramp: [usize; RAMP_STAGES],
    momentum_warmup: usize,
    momentum_cooldown: usize,
}

impl Schedule {
    fn new(total_steps: usize, base_batch: usize) -> Self {
        let total_steps = total_steps.max(1);
        let momentum_warmup = MOMENTUM_WARMUP_STEPS.min(total_steps / 2);
        let momentum_cooldown =
            MOMENTUM_COOLDOWN_STEPS.min(total_steps.saturating_sub(momentum_warmup));
        Self {
            total_steps,
            base_batch,
            batch_ramp: BATCH_RAMP,
            momentum_warmup,
            momentum_cooldown,
        }
    }

    /// Steps required to consume `target_bars` bar-tokens under the full ramp.
    fn steps_for_bars(target_bars: u64, base_batch: usize) -> usize {
        let per_step = ramp_bars_per_batch_unit() * base_batch as f64;
        ((target_bars as f64 / per_step).ceil() as usize).max(RAMP_STAGES)
    }

    fn stage(&self, step: usize) -> usize {
        ((step * RAMP_STAGES) / self.total_steps).min(RAMP_STAGES - 1)
    }

    fn batch(&self, step: usize) -> usize {
        self.base_batch * self.batch_ramp[self.stage(step)]
    }

    fn context(&self, step: usize) -> i64 {
        stage_context(self.stage(step))
    }

    fn bars_per_step(&self, step: usize) -> u64 {
        self.batch(step) as u64 * self.context(step) as u64
    }

    /// One global multiplier for every parameter group. Flat at `sqrt(batch_ratio)`
    /// across the plateau, then linear to an ABSOLUTE `LR_FLOOR_MULTIPLIER`.
    ///
    /// The floor is absolute, not `0.15 * sqrt(batch_ratio)`: the reference
    /// interpolates the stage multiplier toward 0.15 (`lr*(1-t) + 0.15*t`,
    /// modded-nanogpt `train_gpt.py:1975`), so the batch-size bump is annealed away
    /// over the decay rather than preserved into the final step. Keeping the bump
    /// would end training at `sqrt(3) = 1.73x` the intended terminal rate.
    fn lr_multiplier(&self, step: usize) -> f64 {
        let plateau = (self.batch_ramp[self.stage(step)] as f64).sqrt();
        let progress = step as f64 / self.total_steps as f64;
        if progress <= LR_PLATEAU_FRACTION {
            return plateau;
        }
        let decayed =
            ((progress - LR_PLATEAU_FRACTION) / (1.0 - LR_PLATEAU_FRACTION)).min(1.0);
        plateau + (LR_FLOOR_MULTIPLIER - plateau) * decayed
    }

    /// `MOMENTUM_START -> MOMENTUM_PEAK` over the warmup, hold, then back down over
    /// the cooldown. This is the only warmup in the recipe.
    fn momentum(&self, step: usize) -> f64 {
        let cooldown_start = self.total_steps - self.momentum_cooldown;
        if self.momentum_warmup > 0 && step < self.momentum_warmup {
            let frac = step as f64 / self.momentum_warmup as f64;
            MOMENTUM_START + (MOMENTUM_PEAK - MOMENTUM_START) * frac
        } else if self.momentum_cooldown > 0 && step >= cooldown_start {
            let frac = (step - cooldown_start) as f64 / self.momentum_cooldown as f64;
            MOMENTUM_PEAK + (MOMENTUM_START - MOMENTUM_PEAK) * frac.min(1.0)
        } else {
            MOMENTUM_PEAK
        }
    }

    /// Promotion is only meaningful once the model has trained at the deployed
    /// context; before the final stage, evaluating there is positional extrapolation.
    fn in_final_stage(&self, step: usize) -> bool {
        self.stage(step) == RAMP_STAGES - 1
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn pretrain(args: PretrainArgs) -> Result<()> {
    validate_args(&args)?;
    configure_threads();
    configure_cuda();

    let device = Device::cuda_if_available();
    tch::manual_seed(args.seed as i64);
    if device.is_cuda() {
        tch::Cuda::manual_seed_all(args.seed);
    }

    let run = RunDir::create_fresh(RUNS_PATH, args.run.as_deref())
        .context("failed to create pretrain run dir")?;

    let corpus = load_corpus(&args)?;
    // Taken AFTER any symbol restriction, because the symbol set decides which bars a split
    // contains. The corpus also grows under running jobs and the split instants are
    // percentiles of it, so the identity of the data is a first-class output of the run.
    let corpus_fingerprint = corpus.identity_fingerprint();

    let train_bars = corpus.split_bars(Split::Train) as u64;
    ensure!(
        train_bars > 0,
        "training split is empty; check --data-dir, --resolution-secs and --min-bars"
    );

    let (supports, supports_frozen) = fit_supports(&corpus, &args, &corpus_fingerprint)?;
    let supports_dev = supports.to_device(device);
    let support_set_dev = BarSupportSet::new(vec![(args.resolution_secs, supports.to_device(device))])
        .context("failed building the resolution-keyed support set")?;

    let total_steps = match args.steps {
        Some(steps) => {
            ensure!(steps > 0, "--steps must be positive");
            steps
        }
        None => Schedule::steps_for_bars(train_bars * args.epochs as u64, args.batch_size),
    };
    let schedule = Schedule::new(total_steps, args.batch_size);

    let mut vs = nn::VarStore::new(device);
    let modules = BarModules::new(&vs.root());
    if let Some(path) = args.weights.as_deref() {
        let summary = load_var_store_partial(&mut vs, path)
            .map_err(|err| anyhow!("failed loading {path}: {err}"))?;
        summary
            .require_complete()
            .map_err(|err| anyhow!("incomplete initialization from {path}: {err}"))?;
        println!("initialized {} tensors from {path}", summary.loaded);
    }

    let named = named_trainable_variables(&vs);
    let optimizer = build_optimizer(&named)?;

    let (train_samplers, eval) = build_samplers(&corpus, &args)?;

    // ONE scoring rule for the whole run. Every reference below is recomputed in it, so a
    // banner line, a chart baseline and the gradient can never disagree about which
    // objective is in force.
    let scoring = args.scoring;
    let marginal_nll_dof = supports.marginal_nll_dof(scoring);
    let marginal_nll_bar = supports.marginal_nll_bar(scoring);
    // Score the TRAIN-fitted q* as a fixed prediction against the pinned val windows. This
    // is model-free, costs one encode pass over the pinned set, and turns every "X nats
    // better than the calibrated marginal" claim from a train-vs-val comparison into an
    // honest one. The gap between the two figures is the distribution shift.
    let marginal_nll_dof_val =
        marginal_nll_dof_on(&supports, &eval.promotion, args.batch_size, device, scoring)
            .context("failed scoring the train-fitted marginal on the pinned val windows")?;
    let parts = supports.marginal_nll_parts(scoring);
    let baselines = HeldOutBaselines {
        scoring,
        uniform_nll_bar: supports.uniform_nll_bar(scoring),
        marginal_nll_dof_conditional: supports.marginal_nll_dof_conditional(scoring),
        marginal_nll_dof_val,
        encoding_identity_nats: supports.encoding_identity_nats(scoring),
        scoring_floor_bar: supports.scoring_floor_bar(scoring),
        marginal_class_dof: parts.class,
        marginal_shape_dof: parts.shape,
    };
    print_banner(
        &args,
        &corpus,
        &corpus_fingerprint,
        &schedule,
        &named,
        train_bars,
        marginal_nll_bar,
        &supports,
        &baselines,
        &support_set_dev,
    );

    let mut reporter = PretrainReporter::new(&run.gens, marginal_nll_dof);
    reporter.set_held_out_baselines(baselines);
    Trainer {
        args,
        device,
        schedule,
        run,
        supports,
        supports_dev,
        support_set_dev,
        vs,
        modules,
        optimizer,
        train_samplers,
        eval,
        reporter,
        train_bars,
        marginal_nll_bar,
        marginal_nll_dof,
        marginal_nll_dof_val,
        baselines,
        corpus_fingerprint,
        supports_frozen,
        symbol_count: corpus.symbols().len(),
        stage_coverage: vec![HashSet::new(); RAMP_STAGES],
        bars_seen: 0,
        epoch: 0,
        best_val_nll_bar: f64::INFINITY,
        best_val_nll_bar_conditional: f64::INFINITY,
        best_scores: None,
        promotions: 0,
        train_nll_sum: 0.0,
        train_nll_dof_sum: [0.0; BAR_DOF],
        train_steps: 0,
        aux_share_streak: 0,
        vram_baseline_bytes: None,
        activation_bytes_per_token: None,
        stage_step: 0,
    }
    .run_training()
}

fn validate_args(args: &PretrainArgs) -> Result<()> {
    ensure!(args.epochs > 0, "--epochs must be at least 1");
    if args.epochs > 4 {
        println!(
            "warning: --epochs {} exceeds the useful range; a ~350M bar corpus saturates near 4",
            args.epochs
        );
    }
    ensure!(args.batch_size > 0, "--batch-size must be positive");
    ensure!(
        args.dyn_horizon > 0,
        "--dyn-horizon must be at least 1; the dynamics model needs one step to train"
    );
    ensure!(
        (args.dyn_horizon as i64) < BAR_CONTEXT_RAMP_START,
        "--dyn-horizon {} does not fit in the shortest ramp context {}",
        args.dyn_horizon,
        BAR_CONTEXT_RAMP_START
    );
    ensure!(
        args.lambda_dyn >= 0.0 && args.lambda_kl >= 0.0,
        "dynamics loss weights must be non-negative"
    );
    ensure!(
        args.validation_windows > 0,
        "--validation-windows must be positive; promotion needs a held-out set"
    );
    ensure!(
        args.diagnostic_context > 0 && args.diagnostic_context <= BAR_MAX_CONTEXT,
        "--diagnostic-context must lie in 1..={BAR_MAX_CONTEXT}"
    );
    ensure!(
        args.snapshot_windows > 0,
        "--snapshot-windows must be positive"
    );
    ensure!(args.support_samples > 0, "--support-samples must be positive");
    ensure!(
        args.min_dollar_volume >= 0.0,
        "--min-dollar-volume must be non-negative"
    );
    if let Some((b0, b1)) = args.split_bounds {
        ensure!(
            b0 < b1,
            "--split-bounds must be ascending: got {b0} | {b1} (epoch millis)"
        );
    }
    Ok(())
}

fn configure_threads() {
    let read = |key: &str| {
        std::env::var(key)
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
    };
    tch::set_num_threads(read("TORCH_NUM_THREADS").unwrap_or(1));
    tch::set_num_interop_threads(read("TORCH_NUM_INTEROP_THREADS").unwrap_or(1));
}

/// Open the corpus, pinning the split instants and applying the liquidity gate if asked.
///
/// Order matters and is the reason this is not inline. The split instants are percentiles of
/// the trading-time axis, so they move when symbols leave; deriving them from the FULL set
/// and only then dropping symbols is what keeps both arms of a universe ablation scored over
/// the same wall-clock held-out window.
fn load_corpus(args: &PretrainArgs) -> Result<BarCorpus> {
    let dir = Path::new(&args.data_dir);
    let bounds = effective_split_bounds(args)?;
    let corpus = if args.min_dollar_volume > 0.0 {
        let entries = crate::data::ingest::universe_entries(args.min_dollar_volume)
            .context("failed reading the cached liquidity ranking")?
            .with_context(|| {
                format!(
                    "--min-dollar-volume {} was given but liquidity has never been measured; \
                     run the universe rebuild first, or drop the flag to train on every file",
                    args.min_dollar_volume
                )
            })?;
        let keep: HashSet<String> = entries.into_iter().map(|entry| entry.symbol).collect();
        ensure!(
            !keep.is_empty(),
            "no symbol in the cached ranking clears --min-dollar-volume {}",
            args.min_dollar_volume
        );
        println!(
            "[pretrain] liquidity gate: {} symbols clear ${:.0}/day in the cached ranking",
            keep.len(),
            args.min_dollar_volume
        );
        BarCorpus::load_restricted(
            dir,
            args.resolution_secs,
            args.min_bars,
            bounds,
            &keep,
        )
    } else {
        match bounds {
            Some(bounds) => {
                BarCorpus::load_with_bounds(dir, args.resolution_secs, args.min_bars, bounds)
            }
            None => BarCorpus::load(dir, args.resolution_secs, args.min_bars),
        }
    };
    corpus.with_context(|| format!("failed to load bar corpus from {}", args.data_dir))
}

/// The split instants this run will use: `--split-bounds` if given, otherwise the campaign
/// pin, unless `--derive-split-bounds` asks for the live percentiles.
///
/// Deriving is NOT the safe default. The boundary is the `TRAIN_FRACTION` percentile of
/// pooled bar timestamps, so it moves whenever the corpus does — and after the survivorship
/// expansion it moves 26 days EARLIER, not later, because the newly admitted files are
/// dominated by recent listings and thin names whose bar density rises across the window.
/// A boundary that early drops universe-ranking sessions into validation and reopens the
/// selection leak the pin exists to close, so the pin is the default and derivation is the
/// flag.
///
/// When the bounds are pinned, they are checked against the instant the symbol universe was
/// ranked as of. A corpus SELECTED under one notion of "train" and SCORED under another is
/// precisely the leak this record exists to surface, so the disagreement is fatal rather
/// than logged.
fn effective_split_bounds(args: &PretrainArgs) -> Result<Option<(i64, i64)>> {
    if args.derive_split_bounds {
        ensure!(
            args.split_bounds.is_none(),
            "--derive-split-bounds and --split-bounds contradict each other"
        );
        println!(
            "[pretrain] WARNING deriving split instants from the live corpus. They move with \
             every ingestion, so this run is comparable to nothing; pretrain-compare will \
             refuse to pair it."
        );
        return Ok(None);
    }
    let bounds = args
        .split_bounds
        .unwrap_or(crate::data::ingest::PINNED_SPLIT_BOUNDS);
    if let Some(train_end) = crate::data::ingest::universe_train_end()
        .context("failed reading the universe ranking's train_end")?
    {
        let ranked_at = train_end.timestamp_millis();
        ensure!(
            ranked_at == bounds.0,
            "the symbol universe was ranked as of {} ({ranked_at} ms) but this run splits \
             train|val at {} ({} ms). The corpus would be SELECTED under a different notion \
             of `train` than it is SCORED under, which is the universe leak in a different \
             disguise. Re-rank the universe against this boundary, or pass the boundary the \
             ranking used.",
            iso_ms(ranked_at),
            iso_ms(bounds.0),
            bounds.0
        );
    }
    Ok(Some(bounds))
}

/// Fit the bin supports on the training region only, or reuse a cached fit whose provenance
/// proves it belongs to this corpus.
///
/// The supports define the model's output space and therefore the `nll_bar` scale, so a
/// stale or foreign file makes the metric silently incomparable. A bin-count check cannot
/// see that; the recorded [`BarSupportsProvenance`] can. Returns the supports and whether
/// they were accepted under `--freeze-supports` despite a mismatch, which is a fact the
/// checkpoint records.
fn fit_supports(
    corpus: &BarCorpus,
    args: &PretrainArgs,
    corpus_fingerprint: &str,
) -> Result<(BarSupports, bool)> {
    let path = args
        .supports
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| corpus.supports_path());
    if path.exists() {
        let supports = BarSupports::load(&path)
            .with_context(|| format!("cached supports {} are unreadable", path.display()))?;
        ensure!(
            supports.num_bins() == NUM_BAR_BINS,
            "cached supports {} have {} bins, this build uses {NUM_BAR_BINS}",
            path.display(),
            supports.num_bins()
        );
        let frozen = require_supports_provenance(
            supports.provenance(),
            &path,
            corpus_fingerprint,
            corpus.split_bounds(),
            args.freeze_supports,
        )?;
        return Ok((supports, frozen));
    }
    ensure!(
        !args.freeze_supports,
        "--freeze-supports was given but {} does not exist; point --supports at the frozen \
         artifact, or drop the flag to fit a new one",
        path.display()
    );
    println!(
        "fitting bin supports from {} training bars (seed 0x{:X})",
        args.support_samples, args.seed
    );
    let supports = corpus
        .fit_supports(args.support_samples, args.seed)
        .with_provenance(BarSupportsProvenance {
            corpus_fingerprint: corpus_fingerprint.to_owned(),
            split_bounds: corpus.split_bounds(),
            sample_count: args.support_samples,
            fitted_utc: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        });
    // `BarCorpus::fit_supports` already persisted the provenance-free object, so rewrite it
    // with the stamp attached rather than leaving an unverifiable artifact on disk.
    supports
        .save(&path)
        .with_context(|| format!("failed writing {}", path.display()))?;
    Ok((supports, false))
}

/// Decide whether cached supports may be used against this corpus.
///
/// Three cases, and only one of them is silent:
/// * provenance matches -> reuse, log it;
/// * provenance mismatches or is absent, `--freeze-supports` given -> reuse under a loud
///   warning and return `true` so the fact reaches the checkpoint metadata;
/// * otherwise -> hard error.
///
/// Freezing is the right call for an ablation campaign: refitting mid-campaign moves the
/// `nll_bar` scale and makes every prior run incomparable. The point of this function is
/// that the freeze is a decision somebody took, not a file that happened to be there.
fn require_supports_provenance(
    provenance: Option<&BarSupportsProvenance>,
    path: &Path,
    corpus_fingerprint: &str,
    split_bounds: (i64, i64),
    freeze: bool,
) -> Result<bool> {
    let complaint = match provenance {
        Some(recorded) if recorded.corpus_fingerprint == corpus_fingerprint => {
            println!(
                "reusing bin supports {} — fitted {} from {} train DOF on this exact corpus",
                path.display(),
                recorded.fitted_utc,
                recorded.sample_count
            );
            if recorded.split_bounds != split_bounds {
                // Same corpus content, different instants: possible only when one side
                // pinned --split-bounds. The fit region differs, so say so.
                println!(
                    "  note: supports were fitted against split {} | {}, this run uses {} | {}",
                    iso_ms(recorded.split_bounds.0),
                    iso_ms(recorded.split_bounds.1),
                    iso_ms(split_bounds.0),
                    iso_ms(split_bounds.1),
                );
            }
            return Ok(false);
        }
        Some(recorded) => format!(
            "were fitted on corpus {} (split {} | {}), but this run loaded corpus {}",
            &recorded.corpus_fingerprint[..12.min(recorded.corpus_fingerprint.len())],
            iso_ms(recorded.split_bounds.0),
            iso_ms(recorded.split_bounds.1),
            &corpus_fingerprint[..12.min(corpus_fingerprint.len())],
        ),
        None => "carry no provenance at all, so nothing can confirm which corpus, split or \
                 fit produced them"
            .to_owned(),
    };
    ensure!(
        freeze,
        "cached supports {} {complaint}. They define the output space and therefore the \
         nll_bar scale, so reusing them silently would make this run's number incomparable \
         to the fit it claims. Pass --freeze-supports to reuse them deliberately (the right \
         call mid-campaign, and it is recorded in the checkpoint), or delete the file to \
         refit.",
        path.display()
    );
    println!(
        "WARNING: reusing FROZEN bin supports {} — they {complaint}. nll_bar stays comparable \
         to other runs on these supports and to nothing else. Recorded in the checkpoint.",
        path.display()
    );
    Ok(true)
}

/// One training sampler per ramp stage plus the pinned evaluation sets.
fn build_samplers(
    corpus: &BarCorpus,
    args: &PretrainArgs,
) -> Result<(Vec<BarSampler>, EvaluationSets)> {
    let train = (0..RAMP_STAGES)
        .map(|stage| BarSampler::new(corpus, Split::Train, stage_context(stage), args.seed))
        .collect::<Vec<_>>();
    let eval = EvaluationSets::new(corpus, args)?;
    Ok((train, eval))
}

// ---------------------------------------------------------------------------
// Pinned evaluation sets
// ---------------------------------------------------------------------------

/// Every held-out set, all pinned by [`EVAL_WINDOW_SEED`] so they are byte-identical across
/// runs, across seeds and across ablations.
///
/// * `diagnostic` runs at a fixed context for every run. It carries the calibration
///   metrics and is the curve to compare between experiments.
/// * `promotion` runs at the deployed context, and is the only input to checkpoint
///   selection.
/// * `snapshot` supplies the candle pictures and the rollout diagnostics.
/// * `test` and `test_snapshot` are touched exactly once, by the terminal battery,
///   and never inform any decision during the run.
struct EvaluationSets {
    diagnostic: PinnedSet,
    promotion: PinnedSet,
    snapshot: PinnedSet,
    test: PinnedSet,
    test_snapshot: PinnedSet,
}

struct PinnedSet {
    sampler: BarSampler,
    windows: Vec<WindowRef>,
    context: i64,
}

impl EvaluationSets {
    fn new(corpus: &BarCorpus, args: &PretrainArgs) -> Result<Self> {
        let build = |split: Split, context: i64, count: usize| -> Result<PinnedSet> {
            // EVAL_WINDOW_SEED, never args.seed: the bench must not move when the training
            // seed does, or a seed replicate measures two things at once and neither.
            let sampler = BarSampler::new(corpus, split, context, EVAL_WINDOW_SEED);
            let windows = sampler.pinned_windows(count);
            ensure!(
                !windows.is_empty(),
                "the {} split has no window of {context} bars; the corpus is too small",
                split.as_str()
            );
            Ok(PinnedSet {
                sampler,
                windows,
                context,
            })
        };
        let deployed = stage_context(RAMP_STAGES - 1);
        ensure!(
            args.diagnostic_context > SNAPSHOT_HORIZON,
            "--diagnostic-context must exceed the {SNAPSHOT_HORIZON}-bar snapshot horizon"
        );
        Ok(Self {
            diagnostic: build(Split::Val, args.diagnostic_context, args.validation_windows)?,
            promotion: build(Split::Val, deployed, args.validation_windows)?,
            snapshot: build(Split::Val, args.diagnostic_context, args.snapshot_windows)?,
            test: build(Split::Test, deployed, args.validation_windows)?,
            test_snapshot: build(Split::Test, args.diagnostic_context, args.snapshot_windows)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------

/// NorMuon on every 2-D weight, AdamW on the embedding tables, the five emission
/// heads and every scalar gate. The two routings must exactly partition the
/// VarStore: a parameter that matches neither list would be silently frozen, and a
/// parameter that matches both would be routed by precedence rather than by intent.
fn build_optimizer(named: &[(String, Tensor)]) -> Result<Muon> {
    let muon: Vec<String> = bar_muon_name_substrings()
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    let adamw_tables: Vec<String> = bar_adamw_embedding_substrings()
        .iter()
        .chain(BAR_EMISSION_ADAMW_NAME_SUBSTRINGS.iter())
        .map(|s| (*s).to_owned())
        .collect();
    let adamw_scalars: Vec<String> = bar_adamw_scalar_substrings()
        .iter()
        .map(|s| (*s).to_owned())
        .collect();

    let force_adamw: Vec<String> = adamw_tables
        .iter()
        .chain(adamw_scalars.iter())
        .cloned()
        .collect();
    let beta_overrides: Vec<(String, (f64, f64))> = adamw_tables
        .iter()
        .map(|needle| (needle.clone(), ADAMW_TABLE_BETAS))
        .collect();
    let wd_multipliers: Vec<(String, f64)> = adamw_tables
        .iter()
        .map(|needle| (needle.clone(), ADAMW_TABLE_WEIGHT_DECAY_MULT))
        .collect();

    let cfg = MuonConfig {
        lr: NORMUON_LR,
        use_muon_for_2d: true,
        momentum: MOMENTUM_START,
        nesterov: true,
        beta2: NORMUON_BETA2,
        weight_decay: NORMUON_WEIGHT_DECAY,
        adamw_lr: ADAMW_LR,
        adamw_betas: ADAMW_SCALAR_BETAS,
        adamw_eps: ADAMW_EPS,
        adamw_wd: ADAMW_WEIGHT_DECAY,
        // `wd_mul = 0` on every scalar and gate.
        adamw_no_weight_decay_name_substrings: adamw_scalars.clone(),
        ns_steps: DEFAULT_NS_STEPS,
        force_adamw_name_substrings: force_adamw,
        muon_name_allowlist: muon.clone(),
        orthogonalizer: Orthogonalizer::PolarExpress5,
        quadratic_lr_weight_decay: true,
        cautious_weight_decay: true,
        adamw_beta_overrides: beta_overrides,
        adamw_weight_decay_multipliers: wd_multipliers,
        ..MuonConfig::default()
    };

    let mut optimizer = Muon::new_named(named, cfg);
    assert_routing_partitions(named, &optimizer, &muon, &adamw_tables, &adamw_scalars)?;

    // The per-matrix `max(1, rows/cols).sqrt()` multiplier is applied natively by
    // the NorMuon step; only the extra bumps are configured here.
    let down = bar_muon_down_projection_substrings();
    let matched = optimizer.set_named_lr_scale(down, NORMUON_DOWN_PROJECTION_LR_MULT);
    ensure!(
        matched > 0,
        "no MLP down-projection matched {down:?}; the 2x learning-rate bump would be a no-op"
    );
    // Only the residual lambdas take the 5x bump; post lambdas and the PoPE phase
    // bias stay at 1.0x.
    let matched = optimizer.set_named_lr_scale(&["resid_lambda"], ADAMW_RESID_LAMBDA_LR_MULT);
    ensure!(
        matched > 0,
        "no parameter matched `resid_lambda`; the 5x learning-rate bump would be a no-op"
    );
    Ok(optimizer)
}

fn assert_routing_partitions(
    named: &[(String, Tensor)],
    optimizer: &Muon,
    muon: &[String],
    adamw_tables: &[String],
    adamw_scalars: &[String],
) -> Result<()> {
    let matches = |name: &str, needles: &[String]| needles.iter().any(|n| name.contains(n.as_str()));
    let mut unclaimed = Vec::new();
    let mut ambiguous = Vec::new();
    for (name, _) in named {
        let claims = [muon, adamw_tables, adamw_scalars]
            .iter()
            .filter(|needles| matches(name, needles))
            .count();
        match claims {
            0 => unclaimed.push(name.clone()),
            1 => {}
            _ => ambiguous.push(name.clone()),
        }
    }
    ensure!(
        unclaimed.is_empty(),
        "these trainable parameters match no optimizer routing list and would never be \
         updated: {unclaimed:?}"
    );
    ensure!(
        ambiguous.is_empty(),
        "these trainable parameters match more than one optimizer routing list: {ambiguous:?}"
    );

    let routed = optimizer.muon_param_names().len() + optimizer.adamw_param_names().len();
    ensure!(
        routed == named.len(),
        "optimizer routed {routed} of {} trainable parameters",
        named.len()
    );
    // Bidirectional: a NorMuon-listed weight that is not 2-D would silently fall
    // through to AdamW, and an AdamW-listed weight must never reach NorMuon.
    let muon_routed = optimizer.muon_param_names();
    for name in &muon_routed {
        ensure!(
            matches(name, muon),
            "{name} was routed to NorMuon but is not on the NorMuon list"
        );
    }
    let missing: Vec<&String> = named
        .iter()
        .filter(|(name, _)| matches(name, muon) && !muon_routed.contains(name))
        .map(|(name, _)| name)
        .collect();
    ensure!(
        missing.is_empty(),
        "these NorMuon-listed parameters were routed to AdamW instead, most likely because \
         they are not 2-D: {missing:?}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

struct Trainer {
    args: PretrainArgs,
    device: Device,
    schedule: Schedule,
    run: RunDir,
    supports: BarSupports,
    supports_dev: BarSupports,
    /// Device-resident support set. One entry today; the row-routing set is what
    /// `rollout_beliefs` and a future merged-resolution corpus need.
    support_set_dev: BarSupportSet,
    vs: nn::VarStore,
    modules: BarModules,
    optimizer: Muon,
    train_samplers: Vec<BarSampler>,
    eval: EvaluationSets,
    reporter: PretrainReporter,
    train_bars: u64,
    /// NLL a perfectly calibrated *marginal* head would achieve on this corpus. The
    /// uniform baseline is trivially beatable; this is the first number whose
    /// improvement is evidence of conditional structure, so it is what the promotion
    /// log leads with.
    marginal_nll_bar: f64,
    /// The same reference per DOF, in the scoring rule in force. Cached because the
    /// per-DOF promotion line recomputes it on every validation otherwise.
    marginal_nll_dof: [f64; BAR_DOF],
    /// The same train-fitted `q*`, scored as a FIXED prediction against the pinned val
    /// windows. `marginal_nll_bar` is a train quantity; comparing a held-out number to it
    /// silently attributes the distribution shift to the model.
    marginal_nll_dof_val: [f64; BAR_DOF],
    baselines: HeldOutBaselines,
    /// Identity of the corpus every number in this run was measured on.
    corpus_fingerprint: String,
    /// Supports were reused under `--freeze-supports` despite mismatched provenance.
    supports_frozen: bool,
    /// Symbols the corpus held after the liquidity gate, recorded in the checkpoint.
    symbol_count: usize,
    /// Distinct anchors issued per ramp stage. Each stage owns its own stride-C anchor list
    /// and restarts at index 0, and the token budget splits unevenly across the ramp, so
    /// coverage is per stage and nowhere near one pass for the early ones.
    stage_coverage: Vec<HashSet<WindowRef>>,
    bars_seen: u64,
    epoch: usize,
    best_val_nll_bar: f64,
    /// The value promotion actually compares. `best_val_nll_bar` is still tracked and
    /// charted so the campaign stays comparable to runs scored before the objective was
    /// corrected, but it no longer decides anything.
    best_val_nll_bar_conditional: f64,
    /// Per-window vector of the currently promoted checkpoint, which is what the returns
    /// guard pairs a candidate against.
    best_scores: Option<WindowScores>,
    promotions: usize,
    /// Training NLL accumulated since the last epoch report, so the reported train
    /// curve is an average over the interval rather than one noisy minibatch.
    train_nll_sum: f64,
    train_nll_dof_sum: [f64; BAR_DOF],
    train_steps: usize,
    /// Consecutive steps an AUXILIARY term has held more than [`AUX_SHARE_WARN`] of the
    /// objective's magnitude. One step is minibatch noise; [`AUX_SHARE_WARN_STREAK`] of
    /// them is the objective, and the run says so.
    aux_share_streak: usize,
    /// Device memory in use before the first optimizer step: the weights, the CUDA context
    /// and whatever the card's other tenants already held. Subtracted from a later reading
    /// to attribute the remainder to activations.
    ///
    /// The optimizer's momentum buffers are allocated lazily on the first step and are
    /// therefore counted as "activations", which OVERSTATES the per-token footprint. That
    /// biases the ramp toward holding, which is the only safe direction on a shared card.
    vram_baseline_bytes: Option<u64>,
    /// Measured activation bytes per bar-token, from a probe taken
    /// [`RAMP_PROBE_AFTER_STEPS`] into the current stage once the allocator pool is warm.
    /// `None` before the first probe, which is why stage 0 never holds.
    activation_bytes_per_token: Option<f64>,
    /// Optimizer steps taken inside the current ramp stage.
    stage_step: usize,
}

/// One optimizer step's losses, already reduced to host scalars.
struct StepLoss {
    nll_bar: f64,
    nll_dof: [f64; BAR_DOF],
    dyn_loss: f64,
    kl_loss: f64,
    total: f64,
    /// Share of the objective's total MAGNITUDE carried by each weighted term, in
    /// `(nll, dyn, kl)` order. They sum to one.
    shares: (f64, f64, f64),
    /// Mean `cos(h_t, h_{t+1})` over the batch.
    belief_autocorr: f64,
    /// `dyn` over the trivial-identity baseline `smooth_l1(h_t, sg[h_{t+k}])`.
    dyn_vs_identity: f64,
    grad_norm: f64,
}

/// Which held-out set a promotion decision was taken on. `Deployed` is the only
/// value a full run ever uses; `Diagnostic` exists so a run too short to reach the
/// final ramp stage still produces something loadable, under a logged warning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PromotionTarget {
    Deployed,
    Diagnostic,
}

impl Trainer {
    /// Consumes the trainer: `PretrainReporter::finish` takes the reporter by value,
    /// which makes reporting a promotion after the terminal battery a compile error.
    fn run_training(mut self) -> Result<()> {
        let started = Instant::now();
        let mut window_index = 0usize;
        let mut last_stage = usize::MAX;
        // Everything the card already holds before a single activation is allocated: the
        // weights, the CUDA context, and whatever the other tenants of a shared GPU are
        // using. The ramp's headroom test is measured against this.
        self.vram_baseline_bytes = device_used_bytes(self.device);

        for step in 0..self.schedule.total_steps {
            let stage = self.schedule.stage(step);
            if stage != last_stage {
                // A stage boundary is where a shared card kills a run: two of them died
                // with CUDA OOM entering stage 1. Check the headroom BEFORE the first step
                // at the new shape and hold the batch if the projected increment misses.
                if last_stage != usize::MAX {
                    self.hold_batch_if_short_of_vram(step, last_stage, stage);
                }
                // Anchors are strided by the context length, so each stage walks its
                // own window list from the start.
                window_index = 0;
                last_stage = stage;
                self.stage_step = 0;
                println!(
                    "step {step}: ramp stage {stage} — batch {} (x{} of the base {}), context \
                     {}, lr plateau x{:.3}",
                    self.schedule.batch(step),
                    self.schedule.batch_ramp[stage],
                    self.schedule.base_batch,
                    self.schedule.context(step),
                    (self.schedule.batch_ramp[stage] as f64).sqrt(),
                );
            }

            let batch = self.schedule.batch(step);
            let (refs, sample) = {
                let sampler = &self.train_samplers[stage];
                let batches = sampler.batches_per_epoch(batch).max(1);
                let refs = sampler.batch_refs(self.epoch, window_index % batches, batch);
                let sample = sampler.batch_of(&refs, self.device);
                (refs, sample)
            };
            // Coverage, not throughput: `unique_bar_reuse` counts bar-tokens and cannot see
            // that a stage which wraps its window list re-issues anchors it has already
            // trained on instead of advancing.
            self.stage_coverage[stage].extend(refs);
            window_index += 1;

            let lr_mult = self.schedule.lr_multiplier(step);
            self.optimizer.set_lr(NORMUON_LR * lr_mult);
            self.optimizer.set_adamw_lr(ADAMW_LR * lr_mult);
            let momentum = self.schedule.momentum(step);
            self.optimizer.set_momentum(momentum);

            let loss = self.optimizer_step(&sample, step)?;
            self.bars_seen += self.schedule.bars_per_step(step);

            self.train_nll_sum += loss.nll_bar;
            for (acc, value) in self.train_nll_dof_sum.iter_mut().zip(loss.nll_dof) {
                *acc += value;
            }
            self.train_steps += 1;
            self.stage_step += 1;
            if self.stage_step == RAMP_PROBE_AFTER_STEPS {
                self.probe_activation_footprint(step);
            }

            let (nll_share, dyn_share, kl_share) = loss.shares;
            let mut metrics = StepMetrics::nan();
            metrics.epoch = self.epoch;
            metrics.step = step;
            metrics.nll_bar = loss.nll_bar;
            metrics.nll_dof = loss.nll_dof;
            metrics.dyn_loss = loss.dyn_loss;
            metrics.kl_loss = loss.kl_loss;
            metrics.total_loss = loss.total;
            metrics.nll_share = nll_share;
            metrics.dyn_share = dyn_share;
            metrics.kl_share = kl_share;
            metrics.belief_autocorr = loss.belief_autocorr;
            metrics.dyn_vs_identity = loss.dyn_vs_identity;
            metrics.lr_mult = lr_mult;
            metrics.muon_momentum = momentum;
            metrics.grad_norm = loss.grad_norm;
            metrics.context = self.schedule.context(step);
            metrics.batch_size = batch;
            metrics.bars_seen = self.bars_seen;
            self.reporter.record_step(&metrics)?;
            self.warn_on_auxiliary_domination(step, dyn_share, kl_share);

            let log_now = self.args.log_every > 0
                && (step % self.args.log_every == 0 || step + 1 == self.schedule.total_steps);
            if log_now {
                let elapsed = started.elapsed().as_secs_f64();
                // Absolute AND share. At `lambda_dyn = 1.0` the dynamics term measured 28
                // against `nll` 17 — 62% of the objective — and `nll` ROSE for 4000 steps
                // while this line showed two numbers going up and said nothing about which
                // one the optimizer was actually serving.
                println!(
                    "step {step}/{} | nll {:.4} nats/bar ({:.0}%) | dyn {:.4} x{:e} ({:.0}%) \
                     | kl {:.4} x{:e} ({:.0}%) | total {:.4} | autocorr {:.3} | dyn/identity \
                     {:.3} | lr x{lr_mult:.3} | mom {momentum:.3} | grad {:.3} | {:.2} step/s",
                    self.schedule.total_steps,
                    loss.nll_bar,
                    100.0 * nll_share,
                    loss.dyn_loss,
                    self.args.lambda_dyn,
                    100.0 * dyn_share,
                    loss.kl_loss,
                    self.args.lambda_kl,
                    100.0 * kl_share,
                    loss.total,
                    loss.belief_autocorr,
                    loss.dyn_vs_identity,
                    loss.grad_norm,
                    (step + 1) as f64 / elapsed.max(1e-9)
                );
            }

            let completed_epochs = (self.bars_seen / self.train_bars) as usize;
            let epoch_boundary = completed_epochs > self.epoch;
            let periodic = self.args.validate_every > 0
                && step > 0
                && step % self.args.validate_every == 0;
            let final_step = step + 1 == self.schedule.total_steps;

            if self.args.checkpoint_every > 0
                && step > 0
                && step % self.args.checkpoint_every == 0
            {
                let path = self.run.weights.join(format!("pretrain_step_{step}.ot"));
                self.write_checkpoint(&path)?;
            }

            if epoch_boundary || periodic || final_step {
                self.validate(step, epoch_boundary, final_step)?;
                if epoch_boundary {
                    self.epoch = completed_epochs;
                    // A new epoch's windows are a fresh deterministic permutation.
                    window_index = 0;
                }
            }
        }

        let elapsed = started.elapsed().as_secs_f64();
        println!(
            "pretrain finished: {} steps in {elapsed:.1}s ({:.2} step/s), {} promotions, best \
             held-out nll {:.4} nats/bar under {} scoring ({:+.4} vs the calibrated marginal \
             {:.4}, {:+.4} vs uniform {:.4})",
            self.schedule.total_steps,
            self.schedule.total_steps as f64 / elapsed.max(1e-9),
            self.promotions,
            self.best_val_nll_bar,
            self.args.scoring,
            self.marginal_nll_bar - self.best_val_nll_bar,
            self.marginal_nll_bar,
            self.baselines.uniform_nll_bar - self.best_val_nll_bar,
            self.baselines.uniform_nll_bar,
        );
        ensure!(
            self.promotions > 0,
            "no checkpoint was ever promoted; there is nothing for the planner to load"
        );

        let battery = self.test_battery()?;
        self.reporter.finish(&battery)
    }

    /// Score the promoted checkpoint on the TEST split, exactly once, at the very end.
    /// The model is reloaded from disk rather than read out of memory so the reported
    /// numbers provably belong to the artifact the planner will load.
    fn test_battery(&self) -> Result<TestBattery> {
        let checkpoint = self.run.weights.join("pretrain_best.ot");
        let metadata = world_model_metadata_path(&checkpoint);
        let world = BarWorldModel::load(&checkpoint, &metadata, self.device).with_context(|| {
            format!(
                "the promoted checkpoint {} could not be reloaded for the test battery",
                checkpoint.display()
            )
        })?;
        let lineage = world.lineage_sha256().to_owned();
        ensure!(
            !lineage.is_empty(),
            "the promoted checkpoint carries no lineage hash"
        );

        let stats = evaluate(
            world.modules(),
            world.deployment_supports(),
            &self.eval.test,
            self.args.batch_size,
            self.device,
            true,
            self.args.scoring,
        )?;
        let dispersion = self.dispersion(&self.eval.test, &stats);
        let window = pinned_snapshot_window(&self.eval.test_snapshot, self.device);
        let exact = rollout_nll(
            world.modules(),
            world.supports(),
            &window,
            RolloutMode::Exact,
            self.args.scoring,
        );
        let dynamics = rollout_nll(
            world.modules(),
            world.supports(),
            &window,
            RolloutMode::Dynamics,
            self.args.scoring,
        );

        println!(
            "test split ({} windows at context {}, {} scoring): nll {} nats/bar, {:+.4} vs the \
             calibrated marginal {:.4}, {:+.4} vs uniform {:.4}; rollout h1 {:.4} exact / \
             {:.4} dynamics",
            self.eval.test.windows.len(),
            self.eval.test.context,
            self.args.scoring,
            dispersion,
            self.marginal_nll_bar - stats.nll_bar,
            self.marginal_nll_bar,
            self.baselines.uniform_nll_bar - stats.nll_bar,
            self.baselines.uniform_nll_bar,
            exact[0],
            dynamics[0],
        );
        println!("test split {}", self.per_dof_line(&stats));
        println!(
            "test split conditional nll {:.4} nats/bar ({:+.4} vs the conditional marginal \
             {:.4}); the {:.4}-nat s=0 => u=v=0.5 identity is excluded",
            stats.nll_bar_conditional,
            self.baselines.marginal_nll_bar_conditional() - stats.nll_bar_conditional,
            self.baselines.marginal_nll_bar_conditional(),
            self.baselines.encoding_identity_nats,
        );

        let mut battery = TestBattery::nan(checkpoint, lineage);
        battery.nll_bar = stats.nll_bar;
        battery.nll_dof = stats.nll_dof;
        battery.crps_dof = stats.crps_dof;
        battery.rollout_nll_exact = exact;
        battery.rollout_nll_dynamics = dynamics;
        battery.pit = stats.pit;
        battery.dir_acc = stats.dir_acc;
        battery.corpus_fingerprint = self.corpus_fingerprint.clone();
        battery.split_bounds = self.split_bounds();
        battery.nll_bar_conditional = stats.nll_bar_conditional;
        battery.nll_dof_conditional = stats.nll_dof_conditional;
        battery.nll_bar_se = dispersion.se;
        battery.nll_bar_ci = (dispersion.ci_low, dispersion.ci_high);
        Ok(battery)
    }

    /// Forward, backward and update for one batch of `[B, T+1, 5]` DOF plus its
    /// `[B, T+1, 4]` calendar ids.
    fn optimizer_step(&mut self, sample: &BarBatch, step: usize) -> Result<StepLoss> {
        let dof = &sample.dof;
        let time_ids = &sample.time_ids;
        let context = dof.size()[1] - 1;
        let horizon = self.args.dyn_horizon as i64;
        ensure!(
            horizon < context,
            "--dyn-horizon {horizon} does not fit in a {context}-bar context"
        );

        self.optimizer.zero_grad();
        // bf16 autocast, pinned by `configure_cuda`. Without that pin this would
        // silently be fp16, which needs a gradient scaler this repo does not have; with
        // it, the linears match the bf16 attention kernels instead of promoting the
        // residual stream to fp32 on every layer.
        let (loss, nll, nll_dof, dyn_loss, kl_loss, identity, autocorr) =
            autocast(self.device.is_cuda(), || {
                let input = dof.narrow(1, 0, context);
                let target = dof.narrow(1, 1, context);
                // `prepare`/`locate` are elementwise, so binning commutes with narrowing:
                // one pass over `[B, T + 1, BAR_DOF]` serves the trunk's input, the head's
                // teacher-forced target and every dynamics horizon. Each pass materializes
                // an `[N, BAR_DOF, NUM_BAR_BINS]` comparison tensor, so this is worth
                // hoisting even though it is small beside the transformer.
                let bins = self.supports_dev.bin_ids(dof);
                // One transformer pass. Every dynamics horizon reuses this belief
                // sequence, shifted, so recursion costs only MLP evaluations.
                let beliefs = self.modules.trunk.forward(
                    &input,
                    &bins.narrow(1, 0, context),
                    &time_ids.narrow(1, 0, context),
                    0,
                    true,
                );

                let logits = self
                    .modules
                    .head
                    .logits(&beliefs, &bins.narrow(1, 1, context));
                // The objective and every reported baseline read the same `--scoring`.
                let (nll, nll_dof) = bar_nll_from_logits(
                    &logits,
                    &self.supports_dev.targets(&target, self.args.scoring),
                );

                let (dyn_loss, kl_loss, identity) = dynamics_losses(
                    &self.modules,
                    dof,
                    &bins,
                    time_ids,
                    &beliefs,
                    context,
                    horizon,
                    self.device,
                );
                let autocorr = belief_autocorrelation(&beliefs);
                let loss =
                    &nll + self.args.lambda_dyn * &dyn_loss + self.args.lambda_kl * &kl_loss;
                (loss, nll, nll_dof, dyn_loss, kl_loss, identity, autocorr)
            });

        let total = loss.double_value(&[]);
        ensure!(
            total.is_finite(),
            "loss is not finite at step {step}: {total}"
        );
        loss.backward();

        // Reported, never applied: orthogonalization replaces gradient clipping.
        let grad_norm = global_grad_norm(&self.vs, self.device);
        ensure!(
            grad_norm.is_finite(),
            "gradient norm is not finite at step {step}: {grad_norm}"
        );
        self.optimizer.step(StepKind::Primary);

        let dyn_value = dyn_loss.double_value(&[]);
        let kl_value = kl_loss.double_value(&[]);
        let identity = identity.double_value(&[]);
        Ok(StepLoss {
            nll_bar: nll.double_value(&[]),
            nll_dof: dof_array(&nll_dof),
            dyn_loss: dyn_value,
            kl_loss: kl_value,
            total,
            shares: loss_shares(
                nll.double_value(&[]),
                self.args.lambda_dyn * dyn_value,
                self.args.lambda_kl * kl_value,
            ),
            belief_autocorr: autocorr.double_value(&[]),
            // A zero-init dynamics MLP is exactly the identity, so the ratio starts at 1.0
            // by construction and any departure is the MLP doing something. A degenerate
            // baseline — beliefs already frozen — would divide by zero, so it reports NaN
            // and the chart skips the point rather than plotting an artefact.
            dyn_vs_identity: if identity > 0.0 {
                dyn_value / identity
            } else {
                f64::NAN
            },
            grad_norm,
        })
    }


    /// Diagnostics on the fixed-context set, promotion on the deployed-context set,
    /// then reports and checkpoints.
    fn validate(&mut self, step: usize, epoch_boundary: bool, final_step: bool) -> Result<()> {
        let eval_batch = self.args.batch_size;
        let diagnostic = evaluate(
            &self.modules,
            &self.supports_dev,
            &self.eval.diagnostic,
            eval_batch,
            self.device,
            true,
            self.args.scoring,
        )?;

        // Promotion evaluates at the deployed context, and only once the model has
        // actually trained there. A checkpoint selected earlier would be selected on
        // positional extrapolation, and would never be the winner anyway.
        let promotion = if self.schedule.in_final_stage(step) {
            let stats = evaluate(
                &self.modules,
                &self.supports_dev,
                &self.eval.promotion,
                eval_batch,
                self.device,
                false,
                self.args.scoring,
            )?;
            Some((PromotionTarget::Deployed, stats))
        } else if final_step {
            // A run too short to reach the final stage would otherwise promote
            // nothing at all. Fall back to the diagnostic context, loudly, so a
            // smoke run can never be mistaken for a real selection.
            println!(
                "WARNING step {step}: the run ended at ramp stage {} without reaching the \
                 deployed {}-bar context. Promoting on the {}-bar diagnostic set instead — this \
                 checkpoint was selected under different rules than a full run.",
                self.schedule.stage(step),
                self.eval.promotion.context,
                self.eval.diagnostic.context
            );
            Some((PromotionTarget::Diagnostic, diagnostic.clone()))
        } else {
            println!(
                "step {step}: skipping promotion — ramp stage {} has not reached the deployed \
                 {}-bar context",
                self.schedule.stage(step),
                self.eval.promotion.context
            );
            None
        };

        self.write_checkpoint(&self.run.weights.join("pretrain_last.ot"))?;

        let mut promoted_checkpoint = None;
        let mut promotion_nll = f64::NAN;
        let mut promotion_stats: Option<EvalStats> = None;
        let mut dispersion = Dispersion::nan();
        let mut level = Dispersion::nan();
        if let Some((target, stats)) = promotion {
            let nll = stats.nll_bar;
            promotion_nll = nll;
            let set = self.promotion_set(target);
            dispersion = self.dispersion(set, &stats);
            level = self.level_dispersion(set, &stats);
            let margin = self.marginal_nll_bar - nll;
            println!(
                "step {step}: held-out nll {dispersion} nats/bar at context {}, {margin:+.4} vs \
                 the calibrated marginal {:.4}{} (diagnostic {:.4} at context {})",
                set.context,
                self.marginal_nll_bar,
                if margin > 0.0 {
                    ""
                } else {
                    " — STILL NO CONDITIONAL STRUCTURE"
                },
                diagnostic.nll_bar,
                self.eval.diagnostic.context
            );
            // The aggregate is an unweighted sum over factors with 10x different headroom,
            // and ~0.69 nats of the gain is the s=0 => u=v=0.5 encoding identity. Both
            // facts are invisible in one number, so every promotion line states the five
            // deltas and the conditional figure alongside it.
            println!("step {step}: {}", self.per_dof_line(&stats));
            println!(
                "step {step}: conditional nll {:.4} ({:+.4} vs the conditional marginal {:.4}); \
                 vs the train-fitted marginal scored on THESE windows {:.4} the gain is {:+.4}; \
                 level SE {:.4} over {} calendar blocks (paired MDE {:.4} nats)",
                stats.nll_bar_conditional,
                self.baselines.marginal_nll_bar_conditional() - stats.nll_bar_conditional,
                self.baselines.marginal_nll_bar_conditional(),
                self.baselines.marginal_nll_bar_val(),
                self.baselines.marginal_nll_bar_val() - nll,
                level.se,
                level.blocks,
                dispersion.minimum_detectable_effect(),
            );
            // Selection is on the CONDITIONAL aggregate, and a candidate that regresses on
            // `r` is refused however good the aggregate looks. The guard is a PAIRED
            // difference against the incumbent on identical windows, which is the only
            // comparison this bench can resolve at the scale of the effects involved.
            let selection = stats.nll_bar_conditional;
            let scores = self.window_scores(set, &stats, step);
            let guard = self.returns_regression(set, &scores);
            let regressed = guard.is_some_and(|delta| {
                delta.mean > SELECTION_GUARD_SE_MULTIPLE * delta.se.max(0.0)
            });
            if selection < self.best_val_nll_bar_conditional && regressed {
                let delta = guard.expect("a regression implies a measured delta");
                println!(
                    "step {step}: REFUSING promotion — conditional nll improved to \
                     {selection:.4} but {} regressed by {:+.4} nats against the incumbent, \
                     which is more than the {:.1} paired SE ({:.4}) the guard allows. {} is \
                     the only DOF that determines P&L and has ~10x less headroom than the \
                     intra-bar shape factors this trade would have bought.",
                    BAR_DOF_NAMES[SELECTION_GUARD_DOF],
                    delta.mean,
                    SELECTION_GUARD_SE_MULTIPLE,
                    delta.se,
                    BAR_DOF_NAMES[SELECTION_GUARD_DOF],
                );
            } else if selection < self.best_val_nll_bar_conditional {
                if let Some(delta) = guard {
                    println!(
                        "step {step}: guard clear — paired {} delta {:+.4} +/- {:.4} nats",
                        BAR_DOF_NAMES[SELECTION_GUARD_DOF],
                        delta.mean,
                        delta.se
                    );
                }
                promoted_checkpoint = Some(self.promote(nll, target, eval_batch, &scores)?);
                self.best_val_nll_bar_conditional = selection;
                // Raw `nll_bar` is still reported and charted, so the campaign stays
                // comparable to every run scored before the objective was corrected. It
                // just no longer decides anything.
                self.best_val_nll_bar = nll;
                self.best_scores = Some(scores);
                self.promotions += 1;
            }
            promotion_stats = Some(stats);
        }

        let (exact, dynamics) = self.rollout_diagnostics();
        if epoch_boundary || final_step {
            self.write_snapshot(step)?;
        }

        let train_scale = 1.0 / self.train_steps.max(1) as f64;
        let mut metrics = EpochMetrics::nan();
        metrics.epoch = self.epoch;
        metrics.global_step = step;
        metrics.train_nll_bar = self.train_nll_sum * train_scale;
        metrics.train_nll_dof = self.train_nll_dof_sum.map(|v| v * train_scale);
        metrics.val_nll_bar = promotion_nll;
        metrics.val_nll_bar_diag = diagnostic.nll_bar;
        metrics.val_nll_dof = diagnostic.nll_dof;
        metrics.val_crps_dof = diagnostic.crps_dof;
        metrics.val_pit = diagnostic.pit;
        metrics.val_dir_acc = diagnostic.dir_acc;
        metrics.effective_rank = diagnostic.effective_rank;
        metrics.rollout_nll_exact = exact;
        metrics.rollout_nll_dynamics = dynamics;
        metrics.best_val_nll_bar = self.best_val_nll_bar;
        metrics.val_nll_bar_se = dispersion.se;
        metrics.val_nll_bar_ci = (dispersion.ci_low, dispersion.ci_high);
        metrics.val_nll_bar_se_level = level.se;
        metrics.val_nll_bar_conditional = promotion_stats
            .as_ref()
            .map_or(f64::NAN, |stats| stats.nll_bar_conditional);
        metrics.val_nll_dof_conditional = promotion_stats
            .as_ref()
            .map_or([f64::NAN; BAR_DOF], |stats| stats.nll_dof_conditional);
        metrics.val_nll_dof_class = diagnostic.nll_dof_class;
        metrics.val_nll_dof_shape = diagnostic.nll_dof_shape;
        metrics.stage_coverage = self.stage_coverage_fractions();
        metrics.promoted_checkpoint = promoted_checkpoint;
        // Bar-tokens consumed per unique training bar. It is a throughput ratio, not
        // a coverage measure: each symbol's last `context` bars cannot anchor a
        // window and the trailing partial batch of each stage is dropped, so at
        // reuse 1.000 a fraction of a percent of the corpus is unvisited and an equal
        // fraction has been visited twice. It exists so redundancy can never silently
        // grow without showing up on a chart.
        metrics.unique_bar_reuse = self.bars_seen as f64 / self.train_bars as f64;
        self.reporter.record_epoch(&metrics)?;

        self.train_nll_sum = 0.0;
        self.train_nll_dof_sum = [0.0; BAR_DOF];
        self.train_steps = 0;

        println!(
            "step {step}: unique_bar_reuse {:.4} ({} bar-tokens consumed / {} unique training \
             bars), rollout nll h1 {:.4} exact / {:.4} dynamics",
            metrics.unique_bar_reuse, self.bars_seen, self.train_bars, exact[0], dynamics[0]
        );
        Ok(())
    }

    /// Rollout NLL at [`ROLLOUT_HORIZONS`] under both belief-advance mechanisms, on
    /// the pinned snapshot windows. The gap between the two is what the KL term is
    /// there to close.
    fn rollout_diagnostics(
        &self,
    ) -> ([f64; ROLLOUT_HORIZONS.len()], [f64; ROLLOUT_HORIZONS.len()]) {
        let window = self.snapshot_windows();
        let exact = rollout_nll(
            &self.modules,
            &self.support_set_dev,
            &window,
            RolloutMode::Exact,
            self.args.scoring,
        );
        let dynamics = rollout_nll(
            &self.modules,
            &self.support_set_dev,
            &window,
            RolloutMode::Dynamics,
            self.args.scoring,
        );
        (exact, dynamics)
    }

    /// The pinned validation snapshot windows.
    fn snapshot_windows(&self) -> SnapshotWindow {
        pinned_snapshot_window(&self.eval.snapshot, self.device)
    }

    /// Write weights, supports and metadata together, in the order the metadata hashes
    /// require. The metadata carries the corpus fingerprint, the split instants and the
    /// selection rule, all folded into the lineage hash.
    fn write_checkpoint(&self, weights: &Path) -> Result<PathBuf> {
        let res = self.args.resolution_secs;
        let supports_path = world_model_supports_path(weights, res);
        self.supports
            .save(&supports_path)
            .with_context(|| format!("failed writing {}", supports_path.display()))?;
        self.vs
            .save(weights)
            .with_context(|| format!("failed writing {}", weights.display()))?;
        // Every fitted resolution is folded into the lineage; the last argument is the
        // deployment/selection resolution, which Main fixed at 300s.
        BarWorldModelMetadata::save_for_checkpoint_with(
            weights,
            &[res],
            res,
            Some(self.training_provenance()),
        )
        .with_context(|| format!("failed writing metadata for {}", weights.display()))
    }

    /// What this run was trained and selected on, for the checkpoint sidecar.
    fn training_provenance(&self) -> BarTrainingProvenance {
        BarTrainingProvenance {
            corpus_fingerprint: self.corpus_fingerprint.clone(),
            split_bounds: self.split_bounds(),
            split_bounds_pinned: !self.args.derive_split_bounds,
            eval_window_seed: EVAL_WINDOW_SEED,
            train_seed: self.args.seed,
            selection_metric: SELECTION_METRIC.to_owned(),
            selection_weights: SELECTION_WEIGHTS,
            selection_guard_dof: BAR_DOF_NAMES[SELECTION_GUARD_DOF].to_owned(),
            selection_guard_se_multiple: SELECTION_GUARD_SE_MULTIPLE,
            universe_fingerprint: crate::data::ingest::universe_fingerprint()
                .ok()
                .flatten(),
            universe_train_end_ms: crate::data::ingest::universe_train_end()
                .ok()
                .flatten()
                .map(|instant| instant.timestamp_millis()),
            min_dollar_volume: self.args.min_dollar_volume,
            symbols: self.symbol_count,
            supports_frozen: self.supports_frozen,
            supports_corpus_fingerprint: self
                .supports
                .provenance()
                .map(|p| p.corpus_fingerprint.clone()),
            scoring: self.args.scoring.to_string(),
        }
    }

    /// Which pinned set a promotion decision was taken on.
    fn promotion_set(&self, target: PromotionTarget) -> &PinnedSet {
        match target {
            PromotionTarget::Deployed => &self.eval.promotion,
            PromotionTarget::Diagnostic => &self.eval.diagnostic,
        }
    }

    /// Paired difference of the guarded factor against the currently promoted checkpoint,
    /// window by window, or `None` when there is nothing comparable to pair against.
    ///
    /// Paired and not two levels: the unpaired SE of a level is ~0.10 nats and its minimum
    /// detectable difference ~0.41, which would let any realistic regression through. On
    /// identical windows the per-window correlation between two checkpoints of the same run
    /// is very high, so the difference is resolvable at a few hundredths of a nat.
    ///
    /// `None` on the first promotion (nothing to regress against) and whenever the
    /// incumbent was scored on a different set, which only happens on a run too short to
    /// reach the deployed context.
    fn returns_regression(&self, set: &PinnedSet, candidate: &WindowScores) -> Option<Dispersion> {
        let incumbent = self.best_scores.as_ref()?;
        if incumbent.context != candidate.context
            || incumbent.split != candidate.split
            || incumbent.windows.len() != candidate.windows.len()
        {
            return None;
        }
        let deltas: Vec<f64> = candidate
            .windows
            .iter()
            .zip(incumbent.windows.iter())
            .map(|(new, old)| new.nll_dof[SELECTION_GUARD_DOF] - old.nll_dof[SELECTION_GUARD_DOF])
            .collect();
        Some(block_bootstrap(
            &deltas,
            &self.blocks(set),
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED,
        ))
    }

    /// The split instants every number in this run was measured against.
    fn split_bounds(&self) -> (i64, i64) {
        self.eval.promotion.sampler.split_bounds()
    }

    /// `(symbol, calendar month)` block id of every window in a pinned set, so windows of
    /// one ticker inside one month count as a single draw.
    fn blocks(&self, set: &PinnedSet) -> Vec<u64> {
        let mut ids: BTreeMap<(u32, i32), u64> = BTreeMap::new();
        let mut next = 0u64;
        set.windows
            .iter()
            .map(|window| {
                let key = (window.symbol, calendar_month(set.sampler.anchor_ts_ms(window)));
                *ids.entry(key).or_insert_with(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect()
    }

    /// Held-out mean with a `(symbol, month)` block-bootstrap 95% interval.
    fn dispersion(&self, set: &PinnedSet, stats: &EvalStats) -> Dispersion {
        block_bootstrap(
            &stats.window_nll,
            &self.blocks(set),
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED,
        )
    }

    /// The same mean with CALENDAR-MONTH blocks, which is the honest error bar on the
    /// LEVEL: every symbol shares the same handful of wall-clock months, so the common
    /// regime term never averages down and the finer blocking understates it ~4x.
    fn level_dispersion(&self, set: &PinnedSet, stats: &EvalStats) -> Dispersion {
        let mut ids: BTreeMap<i32, u64> = BTreeMap::new();
        let mut next = 0u64;
        let blocks: Vec<u64> = set
            .windows
            .iter()
            .map(|window| {
                *ids.entry(calendar_month(set.sampler.anchor_ts_ms(window)))
                    .or_insert_with(|| {
                        next += 1;
                        next - 1
                    })
            })
            .collect();
        block_bootstrap(&stats.window_nll, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    }

    /// The five per-DOF deltas against the marginal, as one line.
    ///
    /// The aggregate cannot show that `w` sits exactly at uniform with no headroom at all
    /// while `u` and `v` have over a nat each, so a model that regresses on `r` — the only
    /// DOF that determines P&L — can still be promoted on an intra-bar gain. Printed on
    /// every promotion and on the terminal battery.
    fn per_dof_line(&self, stats: &EvalStats) -> String {
        let parts: Vec<String> = BAR_DOF_NAMES
            .iter()
            .enumerate()
            .map(|(dof, name)| {
                format!(
                    "{name} {:.4} ({:+.4} vs marginal {:.4}, {:+.4} vs marginal-on-val {:.4})",
                    stats.nll_dof[dof],
                    self.marginal_nll_dof[dof] - stats.nll_dof[dof],
                    self.marginal_nll_dof[dof],
                    self.marginal_nll_dof_val[dof] - stats.nll_dof[dof],
                    self.marginal_nll_dof_val[dof],
                )
            })
            .collect();
        format!("per-DOF | {}", parts.join(" | "))
    }

    /// Fraction of each ramp stage's anchor list that has actually been issued.
    fn stage_coverage_fractions(&self) -> Vec<f64> {
        self.stage_coverage
            .iter()
            .zip(self.train_samplers.iter())
            .map(|(seen, sampler)| seen.len() as f64 / sampler.windows().max(1) as f64)
            .collect()
    }

    /// The per-window vector of one evaluation, in a form another run can be paired
    /// against.
    fn window_scores(&self, set: &PinnedSet, stats: &EvalStats, step: usize) -> WindowScores {
        let windows = set
            .windows
            .iter()
            .enumerate()
            .map(|(index, window)| WindowScore {
                symbol: set.sampler.symbol(window.symbol).to_owned(),
                bar_index: window.bar_index,
                ts_ms: set.sampler.anchor_ts_ms(window),
                nll_dof: stats.window_nll_dof[index],
                nll_bar_conditional: stats.window_nll_conditional[index],
            })
            .collect();
        WindowScores {
            format_version: WINDOW_SCORES_FORMAT_VERSION,
            run: self
                .run
                .root
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| self.run.root.display().to_string()),
            global_step: step,
            split: set.sampler.split().as_str().to_owned(),
            context: set.context,
            eval_window_seed: EVAL_WINDOW_SEED,
            corpus_fingerprint: self.corpus_fingerprint.clone(),
            split_bounds: self.split_bounds(),
            marginal_nll_bar: self.marginal_nll_bar,
            scoring: Some(self.args.scoring.to_string()),
            windows,
        }
    }

    /// Save a candidate, load it back through the real world-model loader, confirm
    /// the reloaded model reproduces the held-out NLL, and only then swap it in.
    fn promote(
        &self,
        expected_nll: f64,
        target: PromotionTarget,
        eval_batch: usize,
        scores: &WindowScores,
    ) -> Result<PathBuf> {
        let candidate = self.run.weights.join("pretrain_promotion_candidate.ot");
        let metadata = self.write_checkpoint(&candidate)?;
        let world = BarWorldModel::load(&candidate, &metadata, self.device).with_context(|| {
            format!(
                "promotion candidate {} failed to load back",
                candidate.display()
            )
        })?;
        let reloaded = evaluate(
            world.modules(),
            world.deployment_supports(),
            self.promotion_set(target),
            eval_batch,
            self.device,
            false,
            self.args.scoring,
        )?;
        let drift = (reloaded.nll_bar - expected_nll).abs();
        ensure!(
            drift < PROMOTION_ROUNDTRIP_TOLERANCE,
            "reloaded checkpoint disagrees with the live model: {:.6} vs {expected_nll:.6} \
             nats/bar (drift {drift:.2e})",
            reloaded.nll_bar
        );

        let best = self.run.weights.join("pretrain_best.ot");
        for (from, to) in [
            (candidate.clone(), best.clone()),
            (
                world_model_supports_path(&candidate, self.args.resolution_secs),
                world_model_supports_path(&best, self.args.resolution_secs),
            ),
            (metadata, world_model_metadata_path(&best)),
        ] {
            std::fs::rename(&from, &to).with_context(|| {
                format!("failed promoting {} to {}", from.display(), to.display())
            })?;
        }
        // The per-window vector is what makes the next run's comparison PAIRED, so it is an
        // artifact of the promotion, written under the same name as the weights. Without it
        // an ablation can only compare two levels, at four times the standard error.
        let scores_path = window_scores_path(&best);
        scores.save(&scores_path).with_context(|| {
            format!("failed writing per-window scores {}", scores_path.display())
        })?;
        println!(
            "promoted {} — held-out {:.4} nats/bar under {} scoring, {:+.4} vs the marginal \
             baseline {:.4} and {:+.4} vs uniform {:.4} (lineage {})",
            best.display(),
            expected_nll,
            self.args.scoring,
            self.marginal_nll_bar - expected_nll,
            self.marginal_nll_bar,
            self.baselines.uniform_nll_bar - expected_nll,
            self.baselines.uniform_nll_bar,
            world.lineage_sha256()
        );
        Ok(best)
    }

    /// Ancestral candle-rollout pictures on the pinned snapshot windows, taken from
    /// the currently promoted checkpoint so the report always depicts a real
    /// deployable model. Skipped before the first promotion.
    fn write_snapshot(&mut self, step: usize) -> Result<()> {
        let best = self.run.weights.join("pretrain_best.ot");
        let metadata = world_model_metadata_path(&best);
        if !best.exists() || !metadata.exists() {
            return Ok(());
        }
        let world = match BarWorldModel::load(&best, &metadata, self.device) {
            Ok(world) => world,
            Err(err) => {
                println!("skipping candle snapshot: {err}");
                return Ok(());
            }
        };
        let window = self.snapshot_windows();
        // The rollout KV cache is `windows * SNAPSHOT_SAMPLES` sequences deep, so a
        // batched call over every window at once would need tens of gigabytes. One
        // window at a time keeps the peak at a few, and the result is identical
        // because each window's ancestral samples are independent.
        let parts: Vec<Tensor> = (0..window.history_dof.size()[0])
            .map(|index| {
                world.rollout(
                    &window.history_dof.narrow(0, index, 1),
                    &window.history_time_ids.narrow(0, index, 1),
                    &window.future_time_ids.narrow(0, index, 1),
                    SNAPSHOT_SAMPLES,
                    1.0,
                )
            })
            .collect();
        let rollout = Tensor::cat(&parts, 0);
        self.reporter.record_snapshot(&SnapshotInput {
            rollout: &rollout,
            future_dof: &window.future_dof,
            epoch: self.epoch,
            global_step: step,
        })
    }

    /// Measure the activation footprint of the current ramp stage, once the allocator's
    /// pool has settled.
    ///
    /// The reading is device-wide `used` minus the pre-training baseline, divided by the
    /// stage's bar-tokens. On a SHARED card another tenant's growth inflates it, which
    /// biases the ramp toward holding — the safe direction, and the only one that does not
    /// hand the next OOM to whichever process allocates after us.
    fn probe_activation_footprint(&mut self, step: usize) {
        let (Some(baseline), Some(used)) = (self.vram_baseline_bytes, device_used_bytes(self.device))
        else {
            return;
        };
        let tokens = self.schedule.bars_per_step(step) as f64;
        if tokens <= 0.0 {
            return;
        }
        let activations = used.saturating_sub(baseline) as f64;
        if activations <= 0.0 {
            return;
        }
        let per_token = activations / tokens;
        self.activation_bytes_per_token = Some(per_token);
        println!(
            "step {step}: ramp stage {} activation footprint {:.2} GiB over {tokens:.0} \
             bar-tokens ({:.0} B/token); {:.2} GiB free on the device",
            self.schedule.stage(step),
            activations / (1u64 << 30) as f64,
            per_token,
            device_free_bytes(self.device).unwrap_or(0) as f64 / (1u64 << 30) as f64,
        );
    }

    /// Hold the batch at the previous stage's multiplier when the next stage's projected
    /// activation increment does not fit in free VRAM with [`RAMP_MEMORY_MARGIN`] to spare
    /// and [`RAMP_MEMORY_RESERVE_BYTES`] left over.
    ///
    /// The CONTEXT ramp is never held: the deployed model is selected and promoted at the
    /// full context, so a run that never trains there is not the run we asked for. The
    /// batch is the part that only buys gradient-noise reduction, so it is the part that
    /// yields. Holding it also moves the `sqrt(bs_ratio)` learning-rate plateau bump, which
    /// [`Schedule::lr_multiplier`] reads from the same array — the schedule cannot describe
    /// a batch the run is not using — and the downgrade line states the new bump.
    fn hold_batch_if_short_of_vram(&mut self, step: usize, previous: usize, stage: usize) {
        let held = self.schedule.batch_ramp[previous];
        let planned = self.schedule.batch_ramp[stage];
        if planned <= held {
            return;
        }
        let Some(per_token) = self.activation_bytes_per_token else {
            return;
        };
        let Some(free) = device_free_bytes(self.device) else {
            return;
        };

        let base = self.schedule.base_batch as f64;
        let current_tokens = base * held as f64 * stage_context(previous) as f64;
        let tokens_of = |multiplier: usize| base * multiplier as f64 * stage_context(stage) as f64;
        let increment = |multiplier: usize| {
            per_token * (tokens_of(multiplier) - current_tokens).max(0.0)
        };
        let required = |multiplier: usize| {
            increment(multiplier) * (1.0 + RAMP_MEMORY_MARGIN) + RAMP_MEMORY_RESERVE_BYTES as f64
        };
        let gib = |bytes: f64| bytes / (1u64 << 30) as f64;

        let planned_required = required(planned);
        if planned_required <= free as f64 {
            return;
        }

        self.schedule.batch_ramp[stage] = held;
        println!(
            "WARNING step {step}: HOLDING THE BATCH at x{held} ({} windows) entering ramp \
             stage {stage} instead of the planned x{planned} ({} windows). The context ramp \
             still steps to {}. Projected activation increment {:.2} GiB needs {:.2} GiB \
             with the {:.0}% transient margin and the {:.2} GiB shared-card reserve, but \
             only {:.2} GiB is free — this card is shared and two earlier runs died of CUDA \
             OOM at exactly this transition.",
            self.schedule.base_batch * held,
            self.schedule.base_batch * planned,
            stage_context(stage),
            gib(increment(planned)),
            gib(planned_required),
            RAMP_MEMORY_MARGIN * 100.0,
            gib(RAMP_MEMORY_RESERVE_BYTES as f64),
            gib(free as f64),
        );
        println!(
            "WARNING step {step}: the learning-rate plateau bump follows the batch actually \
             used, so it is now sqrt({held}) = {:.3}x rather than the planned sqrt({planned}) \
             = {:.3}x, a {:+.1}% change to every parameter group's rate for the rest of the \
             plateau.",
            (held as f64).sqrt(),
            (planned as f64).sqrt(),
            100.0 * ((held as f64).sqrt() / (planned as f64).sqrt() - 1.0),
        );
        // Holding the batch removes the batch half of the increment. If the context half
        // alone still does not fit there is nothing further to give up without abandoning
        // the deployed context, so say so rather than pretending the hold was sufficient.
        let held_required = required(held);
        if held_required > free as f64 {
            println!(
                "WARNING step {step}: even the held batch needs {:.2} GiB for the context \
                 step to {} against {:.2} GiB free. The context ramp is NOT held — promotion \
                 happens at the deployed context — so this stage may still OOM.",
                gib(held_required),
                stage_context(stage),
                gib(free as f64),
            );
        }
    }

    /// Warn when an AUXILIARY term has held more than [`AUX_SHARE_WARN`] of the objective's
    /// magnitude for [`AUX_SHARE_WARN_STREAK`] consecutive steps.
    ///
    /// Not a clamp. The right response to `dyn` taking over is a decision about
    /// `--lambda-dyn`, and silently rescaling it would hide exactly the miscalibration that
    /// let a 512x reduction fix turn a `1.0` default into 62% of the loss.
    fn warn_on_auxiliary_domination(&mut self, step: usize, dyn_share: f64, kl_share: f64) {
        let worst = dyn_share.max(kl_share);
        if !worst.is_finite() || worst <= AUX_SHARE_WARN {
            self.aux_share_streak = 0;
            return;
        }
        self.aux_share_streak += 1;
        if self.aux_share_streak % AUX_SHARE_WARN_STREAK != 0 {
            return;
        }
        let (name, share, lambda) = if dyn_share >= kl_share {
            ("dyn", dyn_share, self.args.lambda_dyn)
        } else {
            ("kl", kl_share, self.args.lambda_kl)
        };
        println!(
            "WARNING step {step}: the {name} term has held {:.0}% of the objective's \
             magnitude — above the {:.0}% threshold — for {} consecutive steps. It is an \
             AUXILIARY term shaping the latent, not the learning signal; at lambda {:e} it \
             is competing with the likelihood. Lower it or accept that this run is not a \
             maximum-likelihood run.",
            100.0 * share,
            100.0 * AUX_SHARE_WARN,
            self.aux_share_streak,
            lambda,
        );
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Held-out statistics over one pinned window set.
#[derive(Clone, Debug)]
struct EvalStats {
    nll_bar: f64,
    nll_dof: [f64; BAR_DOF],
    /// `nll_bar` with the encoding tautology excluded. `encode_dof` sets `u = v = 0.5`
    /// whenever `s == 0`, and the chain predicts `s` first, so those two factors are free
    /// on a flat bar — worth ~0.69 nats/bar, roughly a fifth of the reported gain over the
    /// calibrated marginal. Here `u` and `v` are averaged only over bars with `s != 0`.
    nll_bar_conditional: f64,
    nll_dof_conditional: [f64; BAR_DOF],
    /// Per-DOF split of the NLL into the degeneracy class and the continuous shape.
    nll_dof_class: [f64; BAR_DOF],
    nll_dof_shape: [f64; BAR_DOF],
    /// One entry per window of the set, in `set.windows` order. The whole point: a mean
    /// with no dispersion is not a measurement, and pairing two runs window by window is
    /// what makes an ablation detectable at 0.04-0.09 nats instead of 0.41.
    window_nll: Vec<f64>,
    window_nll_conditional: Vec<f64>,
    window_nll_dof: Vec<[f64; BAR_DOF]>,
    crps_dof: [f64; BAR_DOF],
    pit: PitHistogram,
    dir_acc: f64,
    effective_rank: f64,
}

/// One pinned snapshot window: conditioning history plus the realized continuation,
/// each with its calendar. The continuation calendar is exogenous — weekends,
/// holidays and the 20:00->04:00 gap make it unextrapolable — so it is supplied to
/// every rollout rather than derived.
struct SnapshotWindow {
    history_dof: Tensor,
    history_time_ids: Tensor,
    future_dof: Tensor,
    future_time_ids: Tensor,
}

/// Split a pinned window set into conditioning history and realized continuation.
/// The continuation calendar is exogenous and known, so it is supplied to every
/// rollout rather than extrapolated from the last history timestamp — weekends,
/// holidays and the 20:00->04:00 gap make such extrapolation wrong.
fn pinned_snapshot_window(set: &PinnedSet, device: Device) -> SnapshotWindow {
    let batch = set.sampler.batch_of(&set.windows, device);
    let history_len = set.context - SNAPSHOT_HORIZON;
    SnapshotWindow {
        history_dof: batch.dof.narrow(1, 0, history_len),
        history_time_ids: batch.time_ids.narrow(1, 0, history_len),
        future_dof: batch.dof.narrow(1, history_len, SNAPSHOT_HORIZON),
        future_time_ids: batch.time_ids.narrow(1, history_len, SNAPSHOT_HORIZON),
    }
}

/// Teacher-forced evaluation over a pinned window set, in full precision so the
/// number is reproducible independently of the training autocast policy. `full` adds
/// the calibration diagnostics; promotion only needs the NLL, and the diagnostics
/// it does not compute are returned as NaN rather than zero.
///
/// The per-window vector is always retained, for both paths: it costs one `[B]` host
/// transfer per chunk and it is the only thing that makes the held-out mean a measurement
/// rather than a number.
fn evaluate(
    modules: &BarModules,
    supports: &BarSupports,
    set: &PinnedSet,
    batch: usize,
    device: Device,
    full: bool,
    scoring: BarScoring,
) -> Result<EvalStats> {
    let mut nll_dof_sum = [0.0f64; BAR_DOF];
    let mut crps_dof_sum = [0.0f64; BAR_DOF];
    let mut class_dof_sum = [0.0f64; BAR_DOF];
    let mut shape_dof_sum = [0.0f64; BAR_DOF];
    // Live-bar sums for the conditional metric, pooled over the WHOLE set: a per-window
    // ratio averaged over windows would weight a mostly-flat window's few live bars as
    // heavily as a fully live one's.
    let mut live_dof_sum = [0.0f64; BAR_DOF];
    let mut live_bars = 0.0f64;
    let mut rows_total = 0.0f64;
    let mut direction_correct = 0.0f64;
    let mut direction_total = 0.0f64;
    let mut pit = PitHistogram::default();
    let mut effective_rank = f64::NAN;
    let mut window_nll: Vec<f64> = Vec::with_capacity(set.windows.len());
    let mut window_nll_conditional: Vec<f64> = Vec::with_capacity(set.windows.len());
    let mut window_nll_dof: Vec<[f64; BAR_DOF]> = Vec::with_capacity(set.windows.len());

    for (chunk_index, chunk) in set.windows.chunks(batch.max(1)).enumerate() {
        let sample = set.sampler.batch_of(chunk, device);
        let context = sample.dof.size()[1] - 1;
        let rows = chunk.len() as f64;

        let (per_window, live, extras) = tch::no_grad(|| {
            let input = sample.dof.narrow(1, 0, context);
            let target = sample.dof.narrow(1, 1, context);
            let bin_ids = supports.bin_ids(&input);
            let beliefs = modules.trunk.forward(
                &input,
                &bin_ids,
                &sample.time_ids.narrow(1, 0, context),
                0,
                false,
            );
            let target_bins = supports.bin_ids(&target);
            let logits = modules.head.logits(&beliefs, &target_bins);
            let soft_targets = supports.targets(&target, scoring);
            // `[B, T, BAR_DOF]`, unreduced: everything below is a reduction of this.
            let terms = bar_nll_terms(&logits, &soft_targets);
            // A bar is LIVE when `s != 0`. On a flat bar the encoding fixes `u = v = 0.5`,
            // so those two factors carry no information and must not be counted as skill.
            let live_mask = target
                .select(-1, DOF_S as i64)
                .not_equal(0.0)
                .to_kind(Kind::Float);
            let per_window_dof = terms.mean_dim([1i64].as_slice(), false, Kind::Float);
            let live_dof = (&terms * live_mask.unsqueeze(-1)).sum_dim_intlist(
                [1i64].as_slice(),
                false,
                Kind::Float,
            );
            let live_count = live_mask.sum_dim_intlist([1i64].as_slice(), false, Kind::Float);

            let extras = full.then(|| {
                let crps = dof_array(&bar_crps_from_logits(&logits, &target, supports));
                // A per-chunk key, not one stream reused 171 times: `counter_uniforms` is
                // keyed by (seed, flat element index), so a constant seed makes element j of
                // every chunk draw the identical uniform and the atom half of the PIT
                // histogram — half the u/v mass — has a far smaller effective sample size
                // than its counts suggest.
                let pit_values = bar_pit_from_logits(
                    &logits,
                    &target,
                    supports,
                    mix64(EVAL_WINDOW_SEED, chunk_index as u64),
                );
                let direction = direction_hits(modules, supports, &beliefs, &target, context);
                let rank = (chunk_index == 0)
                    .then(|| belief_effective_rank(&flatten_beliefs(&beliefs)));
                let parts = bar_nll_decomposition(&logits, &soft_targets, supports);
                (
                    crps,
                    pit_values,
                    direction,
                    rank,
                    dof_array(&parts.class),
                    dof_array(&parts.shape),
                )
            });
            (
                host_rows(&per_window_dof, chunk.len()),
                (
                    host_rows(&live_dof, chunk.len()),
                    Vec::<f64>::try_from(live_count.to_kind(Kind::Double).reshape([-1]))
                        .expect("live-bar counts are convertible"),
                ),
                extras,
            )
        });

        for row in &per_window {
            let total: f64 = row.iter().sum();
            ensure!(
                total.is_finite(),
                "held-out nll is not finite on window chunk {chunk_index} of the {} split: \
                 {total}",
                set.sampler.split().as_str()
            );
            window_nll.push(total);
            window_nll_dof.push(*row);
            for (acc, value) in nll_dof_sum.iter_mut().zip(row) {
                *acc += value;
            }
        }
        let (live_dof, live_counts) = live;
        for (index, row) in live_dof.iter().enumerate() {
            window_nll_conditional.push(conditional_window_nll(
                &per_window[index],
                row,
                live_counts[index],
            ));
            for (acc, value) in live_dof_sum.iter_mut().zip(row) {
                *acc += value;
            }
        }
        live_bars += live_counts.iter().sum::<f64>();

        if let Some((crps, pit_values, (hits, count), rank, class, shape)) = extras {
            for (acc, value) in crps_dof_sum.iter_mut().zip(crps) {
                *acc += value * rows;
            }
            for (acc, value) in class_dof_sum.iter_mut().zip(class) {
                *acc += value * rows;
            }
            for (acc, value) in shape_dof_sum.iter_mut().zip(shape) {
                *acc += value * rows;
            }
            pit.accumulate(&pit_values);
            direction_correct += hits;
            direction_total += count;
            if let Some(rank) = rank {
                effective_rank = rank;
            }
        }
        rows_total += rows;
    }

    ensure!(rows_total > 0.0, "evaluation set produced no windows");
    let scale = 1.0 / rows_total;
    let nll_dof = nll_dof_sum.map(|v| v * scale);
    let nll_dof_conditional = conditional_nll_dof(&nll_dof, &live_dof_sum, live_bars);
    Ok(EvalStats {
        nll_bar: nll_dof.iter().sum(),
        nll_dof,
        nll_bar_conditional: nll_dof_conditional.iter().sum(),
        nll_dof_conditional,
        nll_dof_class: if full {
            class_dof_sum.map(|v| v * scale)
        } else {
            [f64::NAN; BAR_DOF]
        },
        nll_dof_shape: if full {
            shape_dof_sum.map(|v| v * scale)
        } else {
            [f64::NAN; BAR_DOF]
        },
        window_nll,
        window_nll_conditional,
        window_nll_dof,
        crps_dof: if full {
            crps_dof_sum.map(|v| v * scale)
        } else {
            [f64::NAN; BAR_DOF]
        },
        pit,
        dir_acc: if direction_total > 0.0 {
            direction_correct / direction_total
        } else {
            f64::NAN
        },
        effective_rank,
    })
}

/// `[rows, BAR_DOF]` to host, one fixed-size array per row.
fn host_rows(t: &Tensor, rows: usize) -> Vec<[f64; BAR_DOF]> {
    let flat = Vec::<f64>::try_from(t.to_kind(Kind::Double).reshape([-1]))
        .expect("per-window tensor is convertible");
    debug_assert_eq!(flat.len(), rows * BAR_DOF);
    flat.chunks_exact(BAR_DOF)
        .map(|row| std::array::from_fn(|dof| row[dof]))
        .collect()
}

/// Per-DOF NLL with the ENCODING TAUTOLOGY excluded.
///
/// `encode_dof` sets `u = v = 0.5` on every flat bar, and the chain predicts `s` first, so
/// on a flat bar those two factors are determined and cost a well-fitted head nothing. Left
/// in, they are ~0.69 nats/bar of "gain over the calibrated marginal" that is arithmetic
/// rather than prediction. Here `u` and `v` are the mean over LIVE bars only —
/// `live_dof_sum` is the summed per-bar NLL over bars with `s != 0` and `live_bars` their
/// count, pooled over the whole set so a mostly-flat window cannot outweigh a live one.
/// `r`, `s` and `w` are untouched: nothing in the encoding determines them.
fn conditional_nll_dof(
    nll_dof: &[f64; BAR_DOF],
    live_dof_sum: &[f64; BAR_DOF],
    live_bars: f64,
) -> [f64; BAR_DOF] {
    let scale = if live_bars > 0.0 { 1.0 / live_bars } else { 0.0 };
    std::array::from_fn(|dof| match dof {
        DOF_U | DOF_V => live_dof_sum[dof] * scale,
        _ => nll_dof[dof],
    })
}

/// The same exclusion for a single window, summed over the five factors.
///
/// A window with no live bar at all contributes zero for `u` and `v`, which is the only
/// honest value: it carries no evidence about intra-bar shape, and charging it the flat-bar
/// factors would put the tautology straight back in.
fn conditional_window_nll(
    mean_row: &[f64; BAR_DOF],
    live_row: &[f64; BAR_DOF],
    live_bars: f64,
) -> f64 {
    let scale = if live_bars > 0.0 { 1.0 / live_bars } else { 0.0 };
    mean_row[DOF_R]
        + mean_row[DOF_S]
        + mean_row[DOF_W]
        + live_row[DOF_U] * scale
        + live_row[DOF_V] * scale
}

/// Score the TRAIN-fitted reference row as a FIXED prediction against a pinned held-out
/// set, under `scoring`.
///
/// `BarSupports::marginal_nll_dof` is measured on the 4M-row TRAIN fit, and every "X nats
/// better than the calibrated marginal" claim compares a held-out number to it. For `r` and
/// `w` the equal-mass binning makes the row nearly uniform and the comparison is
/// shift-robust; for `s`, `u` and `v` it is not, because 100% of the marginal's advantage
/// over uniform is four measured point masses and those are liquidity statistics that move
/// with the volume and volatility regime.
///
/// This is the same quantity measured on the held-out windows: `CE(q_val, q*_train) =
/// -sum_b q_val(b) ln q*_train(b)`, plus, under [`BarScoring::Density`], the mean log bin
/// width of THESE observations — the measure term belongs to the data, not to the
/// prediction, so the held-out figure has to carry the held-out one. It needs no model, so
/// it runs once at startup. `CE - H(q*) = KL(q_val || q*_train) >= 0` is exactly the
/// distribution shift, and it belongs in the log.
fn marginal_nll_dof_on(
    supports: &BarSupports,
    set: &PinnedSet,
    batch: usize,
    device: Device,
    scoring: BarScoring,
) -> Result<[f64; BAR_DOF]> {
    let bins = NUM_BAR_BINS;
    let mut totals = Tensor::zeros([BAR_DOF as i64, bins], (Kind::Double, Device::Cpu));
    let mut rows_total = 0.0f64;
    for chunk in set.windows.chunks(batch.max(1)) {
        let sample = set.sampler.batch_of(chunk, device);
        let context = sample.dof.size()[1] - 1;
        let chunk_total = tch::no_grad(|| {
            let target = sample.dof.narrow(1, 1, context);
            supports
                .targets(&target, scoring)
                .targets()
                .reshape([-1, BAR_DOF as i64, bins])
                .sum_dim_intlist([0i64].as_slice(), false, Kind::Double)
                .to_device(Device::Cpu)
        });
        totals += chunk_total;
        rows_total += (chunk.len() as i64 * context) as f64;
    }
    ensure!(
        rows_total > 0.0,
        "the pinned {} set produced no bars to measure the held-out marginal on",
        set.sampler.split().as_str()
    );

    let q_val = Vec::<f64>::try_from(totals.reshape([-1]))
        .expect("held-out target histogram is convertible");
    let density = scoring.is_density();
    Ok(std::array::from_fn(|dof| {
        let row = &q_val[dof * bins as usize..(dof + 1) * bins as usize];
        let train = supports.reference_row(dof, scoring);
        let widths = supports.widths(dof);
        row.iter()
            .enumerate()
            .filter(|(_, observed)| **observed > 0.0)
            .map(|(bin, observed)| {
                // The reference row is normalized, so a zero entry means the training fit
                // assigned a held-out outcome literally zero mass. Charge it the
                // uniform-floor surprise rather than an infinity that would hide every
                // other number on the chart.
                let floor = train[bin].max(f64::MIN_POSITIVE);
                let share = observed / rows_total;
                // An atom bin has zero width and carries a MASS, so it takes no correction.
                let measure = if density && widths[bin] > 0.0 {
                    widths[bin].ln()
                } else {
                    0.0
                };
                share * (measure - floor.ln())
            })
            .sum()
    }))
}

/// Directional accuracy of the model's *marginal* return sign at the final position
/// of each window — the one position that is a genuine next-bar forecast rather than
/// a mid-sequence conditional. The head factorizes `r` after `s`, so a teacher-forced
/// expectation would be conditioned on the realized move size; averaging the sign of
/// ancestral samples marginalizes that away.
fn direction_hits(
    modules: &BarModules,
    supports: &BarSupports,
    beliefs: &Tensor,
    target: &Tensor,
    context: i64,
) -> (f64, f64) {
    let last = beliefs.narrow(1, context - 1, 1);
    let repeated = last.repeat([1, DIRECTION_SAMPLES, 1]);
    let samples = modules.head.sample(&repeated, supports, 1.0);
    let predicted = samples
        .select(-1, DOF_R as i64)
        .sign()
        .mean_dim([1i64].as_slice(), false, Kind::Float)
        .sign();
    let realized = target
        .narrow(1, context - 1, 1)
        .squeeze_dim(1)
        .select(-1, DOF_R as i64)
        .sign();
    // Flat bars carry no direction to predict, and an exact tie among the samples is
    // an abstention rather than a wrong call. With eight samples a coin-flip model
    // ties 70/256 of the time, so counting ties as misses would cap `dir_acc` near
    // 0.36 and make an uninformative model read as anti-predictive.
    let scored = realized.ne(0.0).logical_and(&predicted.ne(0.0));
    let hits = (&predicted * &realized).gt(0.0).logical_and(&scored);
    (
        hits.sum(Kind::Float).double_value(&[]),
        scored.sum(Kind::Float).double_value(&[]),
    )
}

/// `[B, T, D] -> [min(B*T, EFFECTIVE_RANK_ROWS), D]`, strided.
///
/// The stride matters: the flattened order is window-major, so a prefix would draw
/// every row from the first few windows and the participation ratio would measure
/// drift along one trajectory instead of the spread of the belief distribution.
fn flatten_beliefs(beliefs: &Tensor) -> Tensor {
    let dim = *beliefs.size().last().expect("belief dim");
    let flat = beliefs.reshape([-1, dim]);
    let total = flat.size()[0];
    let rows = total.min(EFFECTIVE_RANK_ROWS).max(1);
    let stride = (total / rows).max(1);
    flat.slice(0, 0, rows * stride, stride)
}

/// NextLat residual between a predicted belief and its stop-gradient target:
/// `smooth_l1` SUMMED over the feature axis, averaged over tokens.
///
/// The reduction is the whole point. `Reduction::Mean` divides by `B * T *
/// BAR_MODEL_DIM` while `bar_nll_from_logits` divides by `B * T`, so a
/// mean-reduced term makes `lambda_dyn = 1.0` mean `lambda ~ 1/512` of a per-token
/// loss: the observed `dyn = 0.0017` inverted to a per-component belief residual of
/// 0.058 and a belief gradient well under 1% of the NLL's, i.e. a knob that reads
/// as O(1) and is inert. Summing the features puts `dyn` on the same per-token
/// footing as `nll`, so `lambda_dyn` means what it looks like.
fn next_lat_loss(predicted: &Tensor, target: &Tensor) -> Tensor {
    predicted
        .smooth_l1_loss(target, Reduction::None, 1.0)
        .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
        .mean(Kind::Float)
}

/// Recursive `horizon`-step latent rollout. `beliefs[:, t]` is the belief after
/// bar `t`; `z` after `k` steps predicts `beliefs[:, t+k]`.
///
/// Both targets are stop-gradient and the emission head is detached in both
/// branches, so these terms train the trunk and the dynamics MLP only. The
/// calendar of the advancing bar is fed alongside its DOF: `h_{t+k}` is a
/// function of `time_ids_{t+k}`, so withholding it would leave the dynamics model
/// predicting a target it has no information about.
///
/// `bins` is `bin_ids(dof)` over the same `[B, T + 1, BAR_DOF]` window, passed in
/// so the whole step bins once.
///
/// Returns `(dyn, kl, identity)`. The third is the TRIVIAL-IDENTITY baseline: the same
/// NextLat residual with the dynamics MLP replaced by the identity map, i.e. `z_k` left at
/// `h_t`. It is detached and never enters the objective. `dyn / identity` is the only thing
/// that separates "the MLP learned the dynamics" from "the trunk stopped moving the
/// belief", and `rms_norm` cannot tell them apart: it stops beliefs SHRINKING, not from
/// being slowly varying, and a zero-init identity MLP predicts a slowly varying trajectory
/// perfectly.
#[allow(clippy::too_many_arguments)]
fn dynamics_losses(
    modules: &BarModules,
    dof: &Tensor,
    bins: &Tensor,
    time_ids: &Tensor,
    beliefs: &Tensor,
    context: i64,
    horizon: i64,
    device: Device,
) -> (Tensor, Tensor, Tensor) {
    let anchors = context - horizon;
    let anchor_beliefs = beliefs.narrow(1, 0, anchors);
    let mut z = anchor_beliefs.shallow_clone();
    let mut dyn_total: Option<Tensor> = None;
    let mut kl_total: Option<Tensor> = None;
    let mut identity_total: Option<Tensor> = None;

    for k in 1..=horizon {
        // Teacher-forced bar t+k, with its calendar, advances the latent one step.
        let advance = dof.narrow(1, k, anchors);
        let advance_time = time_ids.narrow(1, k, anchors);
        z = modules.dynamics.step(&z, &advance, &advance_time);

        let target = beliefs.narrow(1, k, anchors).detach();
        let dyn_term = next_lat_loss(&z, &target);
        dyn_total = Some(match dyn_total {
            Some(acc) => acc + dyn_term,
            None => dyn_term,
        });

        let identity_term = tch::no_grad(|| next_lat_loss(&anchor_beliefs.detach(), &target));
        identity_total = Some(match identity_total {
            Some(acc) => acc + identity_term,
            None => identity_term,
        });

        // Both categoricals predict bar t+k+1 from their respective latents.
        let emitted = bins.narrow(1, k + 1, anchors);
        let target_logits = modules.head.logits_frozen(&target, &emitted).detach();
        let predicted_logits = modules.head.logits_frozen(&z, &emitted);
        let (kl, _) = bar_categorical_kl(&target_logits, &predicted_logits);
        kl_total = Some(match kl_total {
            Some(acc) => acc + kl,
            None => kl,
        });
    }

    let scale = horizon as f64;
    let zero = || Tensor::zeros([], (Kind::Float, device));
    (
        dyn_total.map_or_else(zero, |t| t / scale),
        kl_total.map_or_else(zero, |t| t / scale),
        identity_total.map_or_else(zero, |t| t / scale),
    )
}

/// Mean lag-1 cosine similarity of a `[B, T, D]` belief sequence.
///
/// Diagnostic only, and detached: nothing optimizes it. It is the axis `rms_norm` does not
/// cover. Normalizing the belief stops it SHRINKING, which is the collapse the isotropy
/// and effective-rank diagnostics were built for; it does nothing about the trunk making
/// `h_{t+1} ~ h_t`, which a zero-init identity dynamics MLP predicts perfectly and which
/// lowers the NextLat term by destroying the trajectory's temporal resolution.
fn belief_autocorrelation(beliefs: &Tensor) -> Tensor {
    let steps = beliefs.size()[1];
    if steps < 2 {
        return Tensor::zeros([], (Kind::Float, beliefs.device()));
    }
    tch::no_grad(|| {
        let current = beliefs.narrow(1, 0, steps - 1).detach();
        let next = beliefs.narrow(1, 1, steps - 1).detach();
        Tensor::cosine_similarity(&current, &next, -1, 1e-8)
            .to_kind(Kind::Float)
            .mean(Kind::Float)
    })
}

/// Share of the objective's total MAGNITUDE carried by each already-weighted term.
///
/// Magnitudes, not the signed total: under [`BarScoring::Density`] the likelihood term is a
/// log density and is routinely NEGATIVE, so a signed denominator would pass through zero
/// and make every share meaningless exactly when the objective is most worth watching.
fn loss_shares(nll: f64, weighted_dyn: f64, weighted_kl: f64) -> (f64, f64, f64) {
    let total = nll.abs() + weighted_dyn.abs() + weighted_kl.abs();
    if !(total > 0.0) || !total.is_finite() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    (
        nll.abs() / total,
        weighted_dyn.abs() / total,
        weighted_kl.abs() / total,
    )
}

/// NVML handle, opened once. `None` when the library or the driver is unavailable, in
/// which case the ramp simply never holds — the same behaviour as before the check existed.
static NVML: LazyLock<Option<Nvml>> = LazyLock::new(|| Nvml::init().ok());

/// `(free, used)` device bytes from NVML. `None` off CUDA or without NVML.
///
/// Device-wide, deliberately: the card is shared, so what a ramp step has to fit into is
/// what the OTHER tenants have left, not what this process believes it allocated.
fn device_memory(device: Device) -> Option<(u64, u64)> {
    let Device::Cuda(index) = device else {
        return None;
    };
    let info = NVML
        .as_ref()?
        .device_by_index(index as u32)
        .ok()?
        .memory_info()
        .ok()?;
    Some((info.free, info.used))
}

fn device_free_bytes(device: Device) -> Option<u64> {
    device_memory(device).map(|(free, _)| free)
}

fn device_used_bytes(device: Device) -> Option<u64> {
    device_memory(device).map(|(_, used)| used)
}

/// Rollout NLL in nats per bar at [`ROLLOUT_HORIZONS`], under one belief-advance
/// mechanism and the run's scoring rule. `beliefs[:, j]` is the belief that predicts
/// `future_dof[:, j]`, so horizon `h` scores index `h - 1`.
fn rollout_nll(
    modules: &BarModules,
    supports: &BarSupportSet,
    window: &SnapshotWindow,
    mode: RolloutMode,
    scoring: BarScoring,
) -> [f64; ROLLOUT_HORIZONS.len()] {
    // Selection and every reported horizon are on the deployment resolution.
    let deployment = supports.only();
    let mut out = [f64::NAN; ROLLOUT_HORIZONS.len()];
    let steps = window.future_dof.size()[1];
    tch::no_grad(|| {
        let beliefs = modules.rollout_beliefs(
            supports,
            &window.history_dof,
            &window.history_time_ids,
            &window.future_dof,
            &window.future_time_ids,
            mode,
        );
        for (slot, horizon) in out.iter_mut().zip(ROLLOUT_HORIZONS) {
            let index = horizon as i64 - 1;
            if index >= steps {
                continue;
            }
            let belief = beliefs.narrow(1, index, 1);
            let target = window.future_dof.narrow(1, index, 1);
            let logits = modules.head.logits(&belief, &deployment.bin_ids(&target));
            let (nll, _) =
                bar_nll_from_logits(&logits, &deployment.targets(&target, scoring));
            *slot = nll.double_value(&[]);
        }
    });
    out
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dof_array(per_dof: &Tensor) -> [f64; BAR_DOF] {
    let values = Vec::<f64>::try_from(per_dof.to_kind(Kind::Double).reshape([-1]))
        .expect("per-DOF tensor is convertible");
    debug_assert_eq!(values.len(), BAR_DOF);
    let mut out = [f64::NAN; BAR_DOF];
    for (slot, value) in out.iter_mut().zip(values) {
        *slot = value;
    }
    out
}

/// Global L2 gradient norm, observed only. The recipe deliberately does not clip:
/// Newton-Schulz/Polar Express orthogonalization already bounds the update.
///
/// Reduces in sorted parameter-name order. `vs.variables()` hands back a `HashMap`
/// whose iteration order is seeded per process, and an fp32 sum is order-dependent,
/// so an unsorted reduction would make this number differ in its last digits between
/// two otherwise bit-identical replays of the same seed.
fn global_grad_norm(vs: &nn::VarStore, device: Device) -> f64 {
    tch::no_grad(|| {
        let squares: Vec<Tensor> = named_trainable_variables(vs)
            .into_iter()
            .filter_map(|(_, tensor)| {
                let grad = tensor.grad();
                grad.defined()
                    .then(|| grad.to_kind(Kind::Float).square().sum(Kind::Float))
            })
            .collect();
        if squares.is_empty() {
            return 0.0;
        }
        Tensor::stack(&squares, 0)
            .to_device(device)
            .sum(Kind::Float)
            .sqrt()
            .double_value(&[])
    })
}

fn print_banner(
    args: &PretrainArgs,
    corpus: &BarCorpus,
    corpus_fingerprint: &str,
    schedule: &Schedule,
    named: &[(String, Tensor)],
    train_bars: u64,
    marginal_nll_bar: f64,
    deployment_supports: &BarSupports,
    baselines: &HeldOutBaselines,
    supports: &BarSupportSet,
) {
    let parameters: i64 = named.iter().map(|(_, t)| t.numel() as i64).sum();
    let mut by_group: BTreeMap<&str, i64> = BTreeMap::new();
    for (_, tensor) in named {
        let group = if tensor.dim() == 2 { "2d" } else { "scalar" };
        *by_group.entry(group).or_default() += tensor.numel() as i64;
    }
    let planned_bars: u64 = (0..schedule.total_steps)
        .map(|step| schedule.bars_per_step(step))
        .sum();

    println!("architecture   {BAR_ARCHITECTURE}");
    println!(
        "model          dim {BAR_MODEL_DIM}, {BAR_LAYERS} layers, {} bins/DOF, {parameters} \
         parameters ({:?})",
        NUM_BAR_BINS, by_group
    );
    println!(
        "corpus         {} symbols, {} unique bars at {}s ({} train / {} val / {} test)",
        corpus.symbols().len(),
        corpus.unique_bars(),
        args.resolution_secs,
        corpus.split_bars(Split::Train),
        corpus.split_bars(Split::Val),
        corpus.split_bars(Split::Test),
    );
    let (train_val, val_test) = corpus.split_bounds();
    // The corpus is live and these instants are percentiles of it, so two runs a week apart
    // score different windows unless the bounds are pinned. Print the identity of the data
    // next to the boundary it produced, and say which of the two it is.
    println!(
        "split          global calendar boundaries {train_val} | {val_test} (ms) = {} | {} {}",
        iso_ms(train_val),
        iso_ms(val_test),
        if args.derive_split_bounds {
            "[DERIVED from the live corpus — comparable to nothing]"
        } else if args.split_bounds.is_some() {
            "[PINNED by --split-bounds]"
        } else {
            "[PINNED to the campaign default ingest::PINNED_SPLIT_BOUNDS]"
        }
    );
    println!("corpus id      {corpus_fingerprint}");
    println!(
        "schedule       {} steps, batch {}->{}, context {}->{}, lr flat {:.0}% then linear to \
         {:.2}x, momentum {MOMENTUM_START}->{MOMENTUM_PEAK} over {} steps and back over {}. \
         Each batch step-up is gated on free VRAM and is HELD (context ramp kept, lr plateau \
         bump lowered to match) when the projected activation increment does not fit with a \
         {:.0}% transient margin and a {:.2} GiB reserve for the card's other tenants.",
        schedule.total_steps,
        schedule.base_batch * BATCH_RAMP[0],
        schedule.base_batch * BATCH_RAMP[RAMP_STAGES - 1],
        stage_context(0),
        stage_context(RAMP_STAGES - 1),
        LR_PLATEAU_FRACTION * 100.0,
        LR_FLOOR_MULTIPLIER,
        schedule.momentum_warmup,
        schedule.momentum_cooldown,
        RAMP_MEMORY_MARGIN * 100.0,
        RAMP_MEMORY_RESERVE_BYTES as f64 / (1u64 << 30) as f64,
    );
    println!(
        "budget         {} epochs = {planned_bars} bar-tokens over {train_bars} unique training \
         bars (target reuse {:.3})",
        args.epochs,
        planned_bars as f64 / train_bars as f64
    );
    // `lambda_dyn` is swept over orders of magnitude, so a fixed three-decimal format would
    // print the sweep's whole lower half as `0.000` — i.e. as if the NextLat term were
    // switched off — in the one artifact that records which objective a run trained under.
    println!(
        "objective      nll + {:e}*dyn + {:e}*kl, dynamics horizon {}, scored under {}. \
         `dyn` is smooth_l1 SUMMED over the {BAR_MODEL_DIM}-wide feature axis and averaged \
         over tokens, so it is commensurate with nll and this weight means what it looks \
         like. Every step prints each term's share of the objective's magnitude and the run \
         warns when an auxiliary term holds more than {:.0}% of it for {} consecutive steps.",
        args.lambda_dyn,
        args.lambda_kl,
        args.dyn_horizon,
        args.scoring,
        AUX_SHARE_WARN * 100.0,
        AUX_SHARE_WARN_STREAK,
    );
    // The one line that says which units every nats figure below is in. The three modes
    // differ by additive constants that depend on the binning, so a `density` figure sits
    // tens of nats below a `smoothed` one on the identical model.
    println!(
        "scoring        {} — {}. Recorded in the checkpoint metadata, folded into the \
         lineage hash, and checked by pretrain-compare, which REFUSES to pair two runs that \
         disagree. The three modes are NOT comparable in absolute nats.",
        args.scoring,
        match args.scoring {
            BarScoring::Smoothed =>
                "Gaussian label smoothing at 0.75x the local bin width; proper for the \
                 SMOOTHED law, not for the one we observe, and it pays an unreachable floor",
            BarScoring::Hard =>
                "one-hot cross entropy on the containing bin; proper for the discretized \
                 law, no floor, but its scale moves with the bin count",
            BarScoring::Density =>
                "the mixed-measure log-likelihood: log P(atom) on an atom and log P_b - log \
                 width_b inside a continuous bin; no floor and, up to discretization error, \
                 no dependence on the bin count",
        },
    );
    println!(
        "evaluation     promotion on {} windows at context {}, diagnostic on {} windows at \
         context {}, windows pinned by EVAL_WINDOW_SEED 0x{EVAL_WINDOW_SEED:X} (train seed \
         0x{:X} moves the sampler and the init, never the bench)",
        args.validation_windows,
        stage_context(RAMP_STAGES - 1),
        args.validation_windows,
        args.diagnostic_context,
        args.seed,
    );
    println!("selection      {SELECTION_METRIC}, weights {SELECTION_WEIGHTS:?}");
    println!(
        "conditioning   {BAR_TIME_CONDITIONING}, calendar cardinality {BAR_TIME_CARDINALITY:?}"
    );
    // A merged-resolution run that silently promoted on the wrong timeframe would be
    // very hard to spot afterwards, so the mix and the selection resolution are stated
    // explicitly. Daily bars are auxiliary training data bought for regime diversity,
    // never a deployment target: selection is on the deployment resolution alone and is
    // never blended, or a model could win by improving on a timeframe we never trade.
    println!(
        "resolutions    fitted {:?}s, SELECTION AND PROMOTION ON {}s ONLY (per-resolution \
         marginal: {})",
        supports.resolutions(),
        args.resolution_secs,
        supports
            .resolutions()
            .iter()
            .map(|res| format!(
                "{res}s -> {:.4}",
                supports
                    .get(*res)
                    .expect("listed resolution is present")
                    .marginal_nll_bar(args.scoring)
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "baseline       dof order {BAR_DOF_NAMES:?}; uniform {:.4} nats/bar (where zero-init \
         heads start), calibrated marginal {marginal_nll_bar:.4}, both under {}. Only \
         progress past the MARGINAL is evidence of conditional structure: beating uniform \
         only proves the unconditional bin masses were learned.",
        baselines.uniform_nll_bar,
        args.scoring,
    );
    // Three corrections to the headline comparison, all of which move it the same way: the
    // reported gain over the calibrated marginal is smaller than it looks.
    println!(
        "baseline (val) the SAME train-fitted q*, scored as a fixed prediction on the pinned \
         val windows: {:.4} nats/bar, i.e. {:+.4} of distribution shift the train figure \
         attributes to the model. Per DOF {}.",
        baselines.marginal_nll_bar_val(),
        baselines.marginal_nll_bar_val() - marginal_nll_bar,
        baselines
            .marginal_nll_dof_val
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
            .join("/"),
    );
    println!(
        "identity       encode_dof forces u = v = 0.5 whenever s = 0, and the chain predicts \
         s first, so {:.4} nats/bar of any gain over the marginal is an ARITHMETIC IDENTITY \
         of the encoding. A head that learned only that bit scores {:.4}; that, not \
         {marginal_nll_bar:.4}, is the line conditional structure has to clear. The reported \
         nll_bar_conditional excludes it by scoring u and v only where s != 0 (conditional \
         marginal {:.4}).",
        deployment_supports.encoding_identity_nats(args.scoring),
        deployment_supports.marginal_plus_identity_nll_bar(args.scoring),
        baselines.marginal_nll_bar_conditional(),
    );

    // The deployment resolution owns every number below: these are properties of one
    // fitted support, and blending them across timeframes would report a floor no
    // single corpus pays.
    let deployment = supports
        .get(args.resolution_secs)
        .expect("the deployment resolution is always fitted");
    let floor_dof = deployment.scoring_floor(args.scoring);
    let floor_bar = deployment.scoring_floor_bar(args.scoring);
    if floor_bar > 0.0 {
        println!(
            "floor          label smoothing at sigma {BAR_LABEL_SIGMA_RATIO:.2}x bin width \
             makes nll_bar a proper rule for the SMOOTHED law, so {floor_bar:.4} nats/bar is \
             UNREACHABLE even by an oracle (per DOF {}). The marginal reference pays none of \
             it, so the reachable range is {:.4}, not {marginal_nll_bar:.4}. This floor is \
             exactly why --scoring density is the default.",
            floor_dof
                .iter()
                .map(|f| format!("{f:.3}"))
                .collect::<Vec<_>>()
                .join("/"),
            marginal_nll_bar - floor_bar,
        );
    } else {
        println!(
            "floor          {} scores the bin the observation actually landed in, so the \
             floor is exactly zero: an oracle pays nothing and the whole {marginal_nll_bar:.4} \
             nats of the marginal reference is reachable. Under smoothed it would be \
             {:.4} nats of it that no model can ever recover.",
            args.scoring,
            deployment.scoring_floor_bar(BarScoring::Smoothed),
        );
    }
    let split = deployment.marginal_nll_parts(args.scoring);
    println!(
        "degeneracy     atom mass per DOF {}; of the {marginal_nll_bar:.4} marginal, \
         {:.4} is the atom-vs-continuous INDICATOR and {:.4} is intra-continuous shape. \
         A head that only learned which bars are degenerate would post the former as gain.",
        (0..BAR_DOF)
            .map(|dof| format!("{:.3}", deployment.atom_mass(dof)))
            .collect::<Vec<_>>()
            .join("/"),
        split.class_bar(),
        split.shape_bar(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::dataset::BAR_TIME_FEATURES;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha12Rng;
    use shared::bars::{write_bar_file, PackedBar, FILE_EXTENSION};

    const TEST_RES: u32 = 300;

    struct Fixture {
        dir: PathBuf,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// A corpus just large enough that the 10% validation and test regions each hold a
    /// full deployed-context window, which is what `EvaluationSets::new` requires.
    fn corpus_fixture(label: &str) -> (Fixture, BarCorpus) {
        let dir = std::env::temp_dir().join(format!(
            "trading_bot_0_pretrain_{label}_{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let step_ms = TEST_RES as i64 * 1000;
        let base = 1_600_000_000_000i64 / step_ms * step_ms;
        for (symbol, seed, offset) in [("AAA", 1u64, 0i64), ("BBB", 2, 700), ("CCC", 3, 300)] {
            let mut rng = ChaCha12Rng::seed_from_u64(seed);
            let mut close = 100.0f32;
            let bars: Vec<PackedBar> = (0..26_000)
                .map(|i| {
                    let open = close;
                    close = (close * (1.0 + rng.random_range(-0.01f32..0.01))).max(1.0);
                    let spread = rng.random_range(0.0f32..0.02) * open;
                    PackedBar {
                        ts_ms: base + (offset + i) * step_ms,
                        open,
                        high: open.max(close) + spread,
                        low: (open.min(close) - spread).max(0.5),
                        close,
                        volume: rng.random_range(1_000.0f32..50_000.0),
                        vwap: 0.25 * (open + close + open + close),
                        trades: rng.random_range(1u32..500),
                    }
                })
                .collect();
            write_bar_file(
                &dir.join(format!("{symbol}.{TEST_RES}.{FILE_EXTENSION}")),
                symbol,
                TEST_RES,
                &bars,
            )
            .expect("write bars");
        }
        let corpus = BarCorpus::load(&dir, TEST_RES, 100).expect("load corpus");
        (Fixture { dir }, corpus)
    }

    /// SETUP-SEL-005. The conditional metric must drop exactly the deterministic mass: the
    /// `u`/`v` factors on flat bars, and nothing else.
    #[test]
    fn conditional_nll_excludes_exactly_the_flat_bar_contribution() {
        // 100 bars, 30 of them flat. On a flat bar the head pays 0 for u and v because the
        // encoding already fixed them; on a live bar it pays 5.0 each.
        let bars = 100.0;
        let live_bars = 70.0;
        let per_live = 5.0;
        let unconditional = per_live * live_bars / bars; // 3.5, diluted by the free bars
        let nll_dof = [4.0, 4.5, unconditional, unconditional, 4.8];
        let live_dof_sum = [
            4.0 * bars,
            4.5 * bars,
            per_live * live_bars,
            per_live * live_bars,
            4.8 * bars,
        ];

        let conditional = conditional_nll_dof(&nll_dof, &live_dof_sum, live_bars);
        // r, s and w are untouched, bit for bit.
        for dof in [DOF_R, DOF_S, DOF_W] {
            assert_eq!(conditional[dof], nll_dof[dof], "DOF {}", BAR_DOF_NAMES[dof]);
        }
        // u and v now read the price of a LIVE bar, not the flat-diluted average.
        for dof in [DOF_U, DOF_V] {
            assert!(
                (conditional[dof] - per_live).abs() < 1e-12,
                "DOF {} conditional {} != {per_live}",
                BAR_DOF_NAMES[dof],
                conditional[dof]
            );
            assert!(conditional[dof] > nll_dof[dof]);
        }
        // The excluded amount is exactly the flat-bar share of the two shape factors:
        // each of u and v was diluted by `1 - live/bars` of free mass, and nothing else
        // moves.
        let excluded: f64 = conditional.iter().sum::<f64>() - nll_dof.iter().sum::<f64>();
        let expected = 2.0 * per_live * (1.0 - live_bars / bars);
        assert!((excluded - expected).abs() < 1e-12, "{excluded} != {expected}");

        // Per-window: a fully live window is unchanged by the exclusion.
        let all_live = [1.0, 2.0, 3.0, 4.0, 5.0];
        let live_row = all_live.map(|v| v * 8.0);
        assert!(
            (conditional_window_nll(&all_live, &live_row, 8.0) - all_live.iter().sum::<f64>())
                .abs()
                < 1e-12
        );
        // A window with no live bar contributes nothing for u and v rather than the free
        // flat-bar factors, which would smuggle the tautology back in.
        assert!(
            (conditional_window_nll(&all_live, &[0.0; BAR_DOF], 0.0) - (1.0 + 2.0 + 5.0)).abs()
                < 1e-12
        );
    }

    fn test_args(seed: u64, dir: &Path) -> PretrainArgs {
        PretrainArgs {
            weights: None,
            run: None,
            epochs: 1,
            steps: Some(9),
            batch_size: 2,
            seed,
            data_dir: dir.display().to_string(),
            resolution_secs: TEST_RES,
            min_bars: 100,
            support_samples: 1024,
            scoring: BarScoring::Density,
            dyn_horizon: 1,
            lambda_dyn: 1e-2,
            lambda_kl: 1.0,
            validation_windows: 3,
            diagnostic_context: BAR_CONTEXT_RAMP_START,
            snapshot_windows: 1,
            validate_every: 0,
            checkpoint_every: 0,
            log_every: 0,
            split_bounds: None,
            // Derived, not pinned: the fixture is a synthetic three-symbol corpus whose
            // timeline has nothing to do with the campaign's calendar.
            derive_split_bounds: true,
            supports: None,
            freeze_supports: false,
            min_dollar_volume: 0.0,
        }
    }

    /// SETUP-SEL-002. The bench must not move when `--seed` does.
    ///
    /// Before this split, every pinned set was drawn with `args.seed`, so the only way to
    /// obtain a training replicate also resampled all 4096 promotion windows — which makes
    /// the run-to-run noise floor unmeasurable and every ablation delta uninterpretable,
    /// because two runs are then not even scored on the same data.
    #[test]
    fn evaluation_windows_do_not_move_with_the_training_seed() {
        let (_fx, corpus) = corpus_fixture("evalseed");
        let dir = PathBuf::from(corpus.dir());

        let (train_a, eval_a) =
            build_samplers(&corpus, &test_args(0x5EED, &dir)).expect("samplers a");
        let (train_b, eval_b) =
            build_samplers(&corpus, &test_args(0x5EED + 1, &dir)).expect("samplers b");

        // The training sampler DOES follow the seed: that is the replicate.
        assert_eq!(train_a[0].seed(), 0x5EED);
        assert_eq!(train_b[0].seed(), 0x5EED + 1);
        assert_ne!(train_a[0].seed(), train_b[0].seed());

        for (name, a, b) in [
            ("promotion", &eval_a.promotion, &eval_b.promotion),
            ("diagnostic", &eval_a.diagnostic, &eval_b.diagnostic),
            ("snapshot", &eval_a.snapshot, &eval_b.snapshot),
            ("test", &eval_a.test, &eval_b.test),
            ("test_snapshot", &eval_a.test_snapshot, &eval_b.test_snapshot),
        ] {
            assert_eq!(
                a.sampler.seed(),
                EVAL_WINDOW_SEED,
                "{name} must be pinned by the campaign constant, not by --seed"
            );
            assert_eq!(b.sampler.seed(), EVAL_WINDOW_SEED, "{name}");
            assert!(!a.windows.is_empty(), "{name} produced no windows");
            assert_eq!(
                a.windows, b.windows,
                "{name} windows moved when the training seed changed"
            );
        }
    }

    /// SETUP-PROV-010. Supports may only be reused when they provably belong to this
    /// corpus, or when the operator says so out loud.
    #[test]
    fn cached_supports_are_refused_unless_their_provenance_matches() {
        let path = Path::new("long_data/bars/bar_supports.300.json");
        let mine = "a".repeat(64);
        let theirs = "b".repeat(64);
        let bounds = (1_700_000_000_000i64, 1_710_000_000_000i64);
        let stamped = |fingerprint: &str| BarSupportsProvenance {
            corpus_fingerprint: fingerprint.to_owned(),
            split_bounds: bounds,
            sample_count: 4_000_000,
            fitted_utc: "2026-08-15T00:00:00Z".to_owned(),
        };

        // Matching provenance: reused, and not flagged as a deliberate freeze.
        let matched = stamped(&mine);
        assert!(
            !require_supports_provenance(Some(&matched), path, &mine, bounds, false)
                .expect("a matching fingerprint is reusable")
        );

        // A different corpus is a hard error by default — this is the case that used to
        // pass on a bin-count check alone.
        let foreign = stamped(&theirs);
        let refused = require_supports_provenance(Some(&foreign), path, &mine, bounds, false)
            .expect_err("a foreign fingerprint must be refused");
        let message = format!("{refused:#}");
        assert!(
            message.contains("--freeze-supports"),
            "the error must name the way to proceed deliberately: {message}"
        );

        // A provenance-free legacy artifact is equally unverifiable, so equally refused.
        assert!(require_supports_provenance(None, path, &mine, bounds, false).is_err());

        // With the flag, both are accepted AND reported as frozen, so the checkpoint can
        // record that comparability was bought deliberately.
        assert!(
            require_supports_provenance(Some(&foreign), path, &mine, bounds, true)
                .expect("--freeze-supports accepts a mismatch")
        );
        assert!(require_supports_provenance(None, path, &mine, bounds, true)
            .expect("--freeze-supports accepts a legacy artifact"));
    }

    /// Pinned instants must survive to the corpus unchanged, or a campaign that thinks it
    /// froze the held-out region has not. Exercised through the loader the `--split-bounds`
    /// path routes to, so the synthetic fixture does not have to agree with the real
    /// universe ranking's `train_end`.
    #[test]
    fn pinned_split_bounds_override_the_live_percentiles() {
        let (_fx, corpus) = corpus_fixture("bounds");
        let derived = corpus.split_bounds();
        let pinned = (derived.0 - 3_600_000, derived.1 - 3_600_000);
        let pinned_corpus =
            BarCorpus::load_with_bounds(corpus.dir(), TEST_RES, 100, pinned).expect("pinned load");
        assert_eq!(pinned_corpus.split_bounds(), pinned);
        assert_ne!(pinned_corpus.split_bounds(), derived);
        // Different windows means a different corpus identity, which is the point.
        assert_ne!(
            pinned_corpus.identity_fingerprint(),
            corpus.identity_fingerprint()
        );
    }

    /// Deriving the boundary is the OPT-OUT, not the default: on the expanded corpus a
    /// derived boundary lands 26 days earlier than the pin and drops universe-ranking
    /// sessions into validation, which is the selection leak reopening.
    #[test]
    fn split_bounds_default_to_the_campaign_pin() {
        let dir = std::env::temp_dir();
        let mut args = test_args(0x5EED, &dir);

        args.derive_split_bounds = true;
        assert_eq!(
            effective_split_bounds(&args).expect("derivation is always allowed"),
            None,
            "--derive-split-bounds must hand the corpus its own percentiles"
        );

        // Contradicting the pin with an explicit pin is a configuration error, not a
        // precedence question.
        args.split_bounds = Some((1, 2));
        assert!(effective_split_bounds(&args).is_err());

        // Without the opt-out the default is the campaign constant, and it agrees with the
        // instant the shipped universe was ranked as of.
        args.split_bounds = None;
        args.derive_split_bounds = false;
        assert_eq!(
            effective_split_bounds(&args).expect("the shipped pin agrees with the ranking"),
            Some(crate::data::ingest::PINNED_SPLIT_BOUNDS)
        );
    }

    /// `lambda_dyn` is only a meaningful knob if `dyn` is a per-TOKEN loss like
    /// `nll`. Pin the reduction against the closed form: a constant per-component
    /// residual `c` must score `BAR_MODEL_DIM * smooth_l1(c)`, not `smooth_l1(c)`.
    #[test]
    fn next_lat_loss_sums_the_feature_axis() {
        let target = Tensor::zeros([3, 4, BAR_MODEL_DIM], (Kind::Float, Device::Cpu));
        for (residual, per_component) in [(0.5f64, 0.125f64), (2.0, 1.5)] {
            let predicted = &target + residual;
            let measured = next_lat_loss(&predicted, &target).double_value(&[]);
            let expected = BAR_MODEL_DIM as f64 * per_component;
            assert!(
                (measured - expected).abs() < 1e-3,
                "residual {residual}: dyn {measured} != {expected}; a mean reduction \
                 would give {per_component}"
            );
        }
    }

    fn schedule(total: usize) -> Schedule {
        Schedule::new(total, 8)
    }

    #[test]
    fn ramp_contexts_span_the_configured_range_and_stay_aligned() {
        assert_eq!(stage_context(0), BAR_CONTEXT_RAMP_START);
        assert_eq!(stage_context(RAMP_STAGES - 1), BAR_MAX_CONTEXT);
        for stage in 0..RAMP_STAGES {
            assert_eq!(stage_context(stage) % 64, 0);
            if stage > 0 {
                assert!(stage_context(stage) > stage_context(stage - 1));
            }
        }
    }

    #[test]
    fn batch_and_context_ramp_at_thirds() {
        let s = schedule(300);
        assert_eq!(s.stage(0), 0);
        assert_eq!(s.stage(99), 0);
        assert_eq!(s.stage(100), 1);
        assert_eq!(s.stage(199), 1);
        assert_eq!(s.stage(200), 2);
        assert_eq!(s.stage(299), 2);
        assert_eq!(s.batch(0), 8);
        assert_eq!(s.batch(100), 16);
        assert_eq!(s.batch(200), 24);
        assert_eq!(s.context(0), 896);
        assert_eq!(s.context(200), BAR_MAX_CONTEXT);
    }

    #[test]
    fn learning_rate_is_flat_then_linear_with_a_batch_bump() {
        let s = schedule(1000);
        // Plateau, stage 0: exactly the base rate, with no warmup.
        assert!((s.lr_multiplier(0) - 1.0).abs() < 1e-12);
        assert!((s.lr_multiplier(300) - 1.0).abs() < 1e-12);
        // Plateau, stage 1: bumped by sqrt(2).
        assert!((s.lr_multiplier(340) - 2.0_f64.sqrt()).abs() < 1e-12);
        // Terminal value: linear from the stage-2 plateau to the ABSOLUTE floor, so
        // the batch bump is annealed away rather than preserved.
        let last = s.lr_multiplier(999);
        let plateau = 3.0_f64.sqrt();
        let expected = plateau + (LR_FLOOR_MULTIPLIER - plateau) * (0.999 - 0.4) / 0.6;
        assert!((last - expected).abs() < 1e-9, "{last} != {expected}");
        assert!((s.lr_multiplier(1000) - LR_FLOOR_MULTIPLIER).abs() < 1e-12);
        // Monotone within a stage.
        for step in 667..999 {
            assert!(s.lr_multiplier(step) >= s.lr_multiplier(step + 1) - 1e-15);
        }
    }

    #[test]
    fn momentum_warms_up_holds_then_cools_down() {
        let s = schedule(5000);
        assert!((s.momentum(0) - MOMENTUM_START).abs() < 1e-12);
        assert!((s.momentum(MOMENTUM_WARMUP_STEPS) - MOMENTUM_PEAK).abs() < 1e-12);
        assert!((s.momentum(2500) - MOMENTUM_PEAK).abs() < 1e-12);
        assert!((s.momentum(5000 - MOMENTUM_COOLDOWN_STEPS) - MOMENTUM_PEAK).abs() < 1e-12);
        let last = s.momentum(4999);
        assert!(last < MOMENTUM_PEAK && last > MOMENTUM_START, "{last}");
    }

    /// A smoke-length run must still traverse the whole shape rather than sitting at
    /// the warmup start forever.
    #[test]
    fn short_runs_compress_the_momentum_schedule() {
        let s = schedule(60);
        assert_eq!(s.momentum_warmup, 30);
        assert_eq!(s.momentum_cooldown, 30);
        assert!((s.momentum(0) - MOMENTUM_START).abs() < 1e-12);
        assert!((s.momentum(30) - MOMENTUM_PEAK).abs() < 1e-12);
        assert!(s.momentum(59) < MOMENTUM_PEAK);
        assert!(s.in_final_stage(59));
        assert!(!s.in_final_stage(0));
    }

    #[test]
    fn step_count_covers_the_requested_number_of_epochs() {
        let base_batch = 24;
        let bars = 280_000_000u64;
        let steps = Schedule::steps_for_bars(bars * 3, base_batch);
        let s = Schedule::new(steps, base_batch);
        let planned: u64 = (0..s.total_steps).map(|step| s.bars_per_step(step)).sum();
        let reuse = planned as f64 / bars as f64;
        assert!(
            (2.99..3.02).contains(&reuse),
            "3 epochs resolved to {reuse} passes over the corpus"
        );
    }

    /// The DOF conversion must preserve order, because every per-DOF report and the
    /// emission chain both key off `BAR_DOF_NAMES`.
    #[test]
    fn per_dof_conversion_preserves_order() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(dof_array(&t), [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    /// The snapshot window must be long enough for every horizon the report plots,
    /// otherwise the longest series would silently be all-NaN and vanish.
    #[test]
    fn snapshot_horizon_covers_every_reported_rollout_horizon() {
        assert_eq!(BAR_DOF_NAMES.len(), BAR_DOF);
        let longest = *ROLLOUT_HORIZONS.iter().max().expect("horizons") as i64;
        assert!(
            SNAPSHOT_HORIZON >= longest,
            "snapshot horizon {SNAPSHOT_HORIZON} cannot reach rollout horizon {longest}"
        );
        assert!(BAR_CONTEXT_RAMP_START > SNAPSHOT_HORIZON);
    }

    /// A SLOWLY VARYING unit-RMS belief trajectory: one anchor plus `drift` of per-step
    /// noise, renormalized onto the shell the trunk emits on.
    ///
    /// Slowly varying on purpose. That is the regime `dyn_vs_identity` exists to expose —
    /// where doing nothing is already a strong NextLat predictor — and the regime a
    /// zero-init identity MLP is indistinguishable from a trained one in the raw `dyn`
    /// number alone.
    fn drifting_beliefs(batch: i64, context: i64, drift: f64, seed: i64) -> Tensor {
        tch::manual_seed(seed);
        let anchor = Tensor::randn([batch, 1, BAR_MODEL_DIM], (Kind::Float, Device::Cpu));
        let noise = Tensor::randn(
            [batch, context, BAR_MODEL_DIM],
            (Kind::Float, Device::Cpu),
        ) * drift;
        let raw = anchor + noise;
        let scale = raw
            .pow_tensor_scalar(2.0)
            .mean_dim([-1i64].as_slice(), true, Kind::Float)
            .sqrt();
        raw / scale
    }

    /// A zero-init dynamics MLP IS the identity: `fc3` is zero, so `step` returns
    /// `rms_norm(h)` and `z_k == h_t` at every horizon. `dyn_vs_identity` must therefore
    /// read exactly 1.0, which is the reading that says the MLP contributes nothing and
    /// `dyn` is measuring belief smoothness alone.
    ///
    /// This is the collapse direction `rms_norm` does NOT cover. It stops beliefs
    /// shrinking; nothing stops the trunk making `h_{t+1} ~ h_t`, which lowers `dyn` by
    /// destroying the trajectory's temporal resolution rather than by learning dynamics.
    #[test]
    fn a_zero_init_dynamics_mlp_scores_exactly_the_identity_baseline() {
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        let (batch, context, horizon) = (2i64, 12i64, 3i64);
        let beliefs = drifting_beliefs(batch, context, 0.25, 0xD1D1);
        let dof = Tensor::randn(
            [batch, context + 1, BAR_DOF as i64],
            (Kind::Float, Device::Cpu),
        ) * 0.01;
        let bins = Tensor::zeros(
            [batch, context + 1, BAR_DOF as i64],
            (Kind::Int64, Device::Cpu),
        );
        let time_ids = Tensor::zeros(
            [batch, context + 1, BAR_TIME_FEATURES as i64],
            (Kind::Int64, Device::Cpu),
        );

        let measure = || {
            let (dyn_loss, _, identity) = dynamics_losses(
                &modules,
                &dof,
                &bins,
                &time_ids,
                &beliefs,
                context,
                horizon,
                Device::Cpu,
            );
            (dyn_loss.double_value(&[]), identity.double_value(&[]))
        };

        let (dyn_loss, identity) = measure();
        assert!(
            identity > 0.0,
            "the identity baseline is degenerate ({identity}), so the ratio would be \
             meaningless"
        );
        let ratio = dyn_loss / identity;
        assert!(
            (ratio - 1.0).abs() < 1e-3,
            "a zero-init dynamics MLP must score exactly the identity baseline: dyn \
             {dyn_loss} / identity {identity} = {ratio}"
        );

        // And the diagnostic must MOVE once the MLP carries weight, or it is measuring
        // nothing. The identity baseline depends only on the beliefs, so it is unchanged.
        tch::no_grad(|| {
            for variable in vs.trainable_variables() {
                let mut variable = variable;
                let _ = variable.normal_(0.0, 0.5);
            }
        });
        let (woken_dyn, woken_identity) = measure();
        assert!(
            (woken_identity - identity).abs() < 1e-6,
            "the identity baseline must not depend on the dynamics weights"
        );
        let woken_ratio = woken_dyn / woken_identity;
        assert!(
            (woken_ratio - 1.0).abs() > 1e-2,
            "waking the dynamics MLP left dyn_vs_identity at {woken_ratio}"
        );
    }

    /// The shares are of the objective's MAGNITUDE, so they stay meaningful when the
    /// likelihood term is a negative log density — which under the default `density`
    /// scoring it routinely is.
    #[test]
    fn loss_shares_are_magnitudes_and_sum_to_one() {
        let (nll, dyn_share, kl) = loss_shares(17.0, 28.0, 0.0);
        assert!((nll + dyn_share + kl - 1.0).abs() < 1e-12);
        // The regression that motivated the chart: lambda_dyn = 1.0 put dyn at 62%.
        assert!(
            (dyn_share - 28.0 / 45.0).abs() < 1e-12,
            "dyn share {dyn_share}"
        );
        assert!(dyn_share > AUX_SHARE_WARN, "62% must trip the warning");

        // A negative log density must not invert or blow up the denominator.
        let (nll, dyn_share, kl) = loss_shares(-30.0, 10.0, 10.0);
        assert!((nll + dyn_share + kl - 1.0).abs() < 1e-12);
        assert!((nll - 0.6).abs() < 1e-12, "nll share {nll}");
        // A zero objective has no shares to report, and reporting zeros would draw a
        // three-way tie that never happened.
        assert!(loss_shares(0.0, 0.0, 0.0).0.is_nan());

        // The default weight keeps the auxiliary term well inside the threshold at the
        // production init figures the ramp fix measured (dyn ~274, nll ~24.26).
        let (_, dyn_share, _) = loss_shares(24.26, 1e-2 * 274.0, 0.0);
        assert!(
            dyn_share < AUX_SHARE_WARN,
            "the 1e-2 default still leaves dyn at {dyn_share} of the objective"
        );
    }

    /// Holding a ramp stage's batch must move the learning-rate plateau bump with it. A
    /// schedule that kept the planned `sqrt(3)` bump while running the previous stage's
    /// batch would be training at 1.73x the rate the batch justifies.
    #[test]
    fn holding_the_batch_moves_the_lr_plateau_bump() {
        let mut schedule = Schedule::new(3000, 16);
        let stage_1 = 1200; // inside stage 1 of three equal stages.
        assert_eq!(schedule.stage(stage_1), 1);
        assert_eq!(schedule.batch(stage_1), 32);
        assert!((schedule.lr_multiplier(stage_1) - 2.0f64.sqrt()).abs() < 1e-12);
        let planned_tokens = schedule.bars_per_step(stage_1);

        schedule.batch_ramp[1] = BATCH_RAMP[0];
        assert_eq!(schedule.batch(stage_1), 16);
        assert!((schedule.lr_multiplier(stage_1) - 1.0).abs() < 1e-12);
        // The CONTEXT ramp is untouched: promotion happens at the deployed context, so the
        // batch is the only thing that may yield.
        assert_eq!(schedule.context(stage_1), stage_context(1));
        assert_eq!(schedule.bars_per_step(stage_1), planned_tokens / 2);
        // Every other stage keeps its plan.
        assert_eq!(schedule.batch(2500), 16 * BATCH_RAMP[2]);
    }

    /// The ramp's headroom check is only real if NVML actually answers. A silent `None`
    /// makes [`Trainer::hold_batch_if_short_of_vram`] a no-op and hands the next OOM back
    /// to whichever process allocates first, which is exactly the failure two runs already
    /// died of. Read-only: this queries the driver and allocates nothing on the device.
    #[test]
    fn the_vram_probe_answers_on_a_cuda_host() {
        assert_eq!(
            device_memory(Device::Cpu),
            None,
            "a CPU run has no VRAM to gate on"
        );
        let Some(nvml) = NVML.as_ref() else {
            eprintln!(
                "NVML unavailable on this host; the ramp guard degrades to never holding"
            );
            return;
        };
        let count = nvml.device_count().expect("NVML device count");
        if count == 0 {
            eprintln!("no NVML devices; the ramp guard degrades to never holding");
            return;
        }
        let (free, used) =
            device_memory(Device::Cuda(0)).expect("NVML reported a device but no memory");
        assert!(used > 0, "a live driver always holds context memory");
        assert!(free > 0, "the card reports no free memory at all");
        // A card this side of 512 GiB, i.e. the numbers are bytes and not something else.
        assert!(
            free + used < 512 * (1u64 << 30),
            "implausible VRAM total: free {free} used {used}"
        );
    }
}
