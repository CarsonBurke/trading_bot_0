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

use anyhow::{anyhow, bail, ensure, Context, Result};
use nvml_wrapper::Nvml;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::Instant;
use tch::{autocast, nn, Device, Kind, Reduction, Tensor};

use crate::torch::bar_dist::{
    bar_categorical_kl, bar_crps_from_logits, bar_nll_decomposition, bar_nll_from_logits,
    bar_nll_terms, bar_pit_from_logits, bar_supports_format_version, BarScoring, BarSupports,
    BarSupportsProvenance, BAR_CHAIN, BAR_DOF, BAR_DOF_NAMES, BAR_EMISSION_ADAMW_NAME_SUBSTRINGS,
    BAR_LABEL_SIGMA_RATIO, BAR_SUPPORTS_FORMAT_VERSION, BAR_SUPPORTS_MOMENTS_VERSION, DOF_R, DOF_S,
    DOF_U, DOF_V, DOF_W, NUM_BAR_BINS,
};
use crate::torch::cuda::cfg::configure_cuda;
use crate::torch::dataset::{
    iso_ms, mix64, time_ids_without_market, BarBatch, BarCorpus, BarSampler, CorpusAnomalies,
    CoverageAudit, PassCensus, PassLayout, PassLedger, PassPlan, Split, WindowRef,
    BAR_TIME_CARDINALITY, BAR_TIME_CONDITIONING,
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
use shared::{
    paths::RUNS_PATH,
    run_dir::{RunDir, RunProvenance},
};

use super::growth::{self, GrowthSupport, LAMBDA_GROWTH};
use super::optimizer_glue::named_trainable_variables;
use super::pretrain_aux::{
    AuxiliaryConfig, AuxiliaryReport, AuxiliaryStream, AUXILIARY_HELDOUT_CONTEXT,
};
use super::pretrain_reports::{
    belief_effective_rank, EpochBoundary, EpochMetrics, HeldOutBaselines, PitHistogram,
    PretrainReporter, RivalSelection, SnapshotInput, StepMetrics, TestBattery, UnmeasuredMetric,
    AUX_SHARE_WARN, AUX_SHARE_WARN_STREAK, DEPLOYED_CONTEXT_METRICS, MIN_FAN_SAMPLES,
    ROLLOUT_HORIZONS,
};
use super::pretrain_stats::{
    block_bootstrap, calendar_month, window_scores_path, Dispersion, TradeSummary, WindowScore,
    WindowScores, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, WINDOW_SCORES_FORMAT_VERSION,
};
use super::trade_bench::{
    self, BenchConfig, ChunkPaths, MeanShrink, TradeBench, TradeSetup,
};

/// Context length at the start of the ramp. Also the fixed context of the
/// across-run diagnostic evaluation, which must never vary between runs.
pub const BAR_CONTEXT_RAMP_START: i64 = 896;

/// Number of ramp stages. Batch size and context both step at each stage boundary,
/// which sits at an equal fraction of total steps.
pub(super) const RAMP_STAGES: usize = 3;
/// DECLARED batch-size multipliers per stage — the ceiling the ramp aims at, never the ramp
/// it runs.
///
/// The ramp that runs is derived from MEASURED capacity before the first step by
/// [`CapacityModel::derive_batch_ramp`], which caps every stage at what the card can hold.
/// This array only says how far the recipe would like to go: `x3` at the deployed context
/// asks for 147,456 bar-tokens a step, which at the measured 495 KB/bar-token is ~70 GiB on
/// a 32 GiB card. Hardcoding it as the plan is what let job 2856 print a schedule of
/// `batch 24->72` and then run at 24 for all 13,832 steps under a reactive memory hold,
/// with the learning-rate plateau bumps of a batch it never used.
const BATCH_RAMP: [usize; RAMP_STAGES] = [1, 2, 3];
/// Exponent relating a stage's batch multiplier to its learning-rate plateau bump.
///
/// Not a uniform square root. modded-nanogpt uses `(16/8)**0.6` for the 2x step and
/// `(24/8)**0.5` for the 3x step (`train_gpt.py:1980-1985`), i.e. the first step-up gets a
/// deliberately larger bump than the square-root rule would give. Linear scaling (exponent
/// 1) overshoots and square-root scaling undershoots at small batch; the reference splits
/// the difference where it measured a difference. Copying `0.5` everywhere left stage 1
/// running 7% under the reference rate.
const BATCH_RAMP_LR_EXPONENT: [f64; RAMP_STAGES] = [0.5, 0.6, 0.5];
// There is deliberately NO auxiliary-weight anneal. A `AUX_ANNEAL_END_FRACTION = 2/3`
// constant used to live here and drove `Schedule::aux_weight`; both are deleted. The
// evidence is on `Args::lambda_dyn`. Do not reintroduce them.
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
/// Batch sizes the STARTUP capacity probe measures, at the deployed context.
///
/// Two points and not one, because a single point cannot separate the batch-proportional
/// cost from the batch-independent one and the whole ramp is derived from the former. At
/// batch 1 and context 2048 the fixed part — cuBLAS workspaces, allocator rounding, and the
/// optimizer state a forward-and-backward probe has not paid for — is amortized over 2048
/// bar-tokens and inflates a single-point per-token figure by more than half, which would
/// derive a ramp far below what the card holds. The slope between two shapes cancels it
/// exactly.
///
/// Small on purpose: the card is shared, and the largest point costs `3 * 2048` bar-tokens,
/// about 3 GiB at the measured rate. A probe that has to OOM to find the ceiling is not a
/// probe.
const CAPACITY_PROBE_BATCHES: [usize; 2] = [1, 3];
/// Forward-and-backward passes at each probe shape before its footprint is read. The first
/// pass grows the allocator pool; the second one finds it warm, which is the steady state
/// training runs in.
const CAPACITY_PROBE_STEPS: usize = 2;
/// Deployed contexts the banner prices out, so the batch/context tradeoff is visible in real
/// numbers rather than asserted.
///
/// At a fixed bytes-per-bar-token it is the PRODUCT `batch * context` that the card caps, so
/// these are not independent knobs. The middle value is the current ramp's stage-1 context
/// and the last is the deployed context; the first is what halving it would buy. This is
/// reported and never acted on: the deployed context is part of the world-model contract the
/// planner depends on.
const CONTEXT_FRONTIER: [i64; 3] = [1024, 1472, BAR_MAX_CONTEXT];
/// Default fraction of training spent at the flat learning-rate plateau, i.e. the default of
/// `--lr-plateau-fraction`. The value a run actually uses is carried on [`Schedule`]; nothing
/// reads this constant per step.
///
/// At `F = 0.40` a ONE-EPOCH run's plateau ends at 0.4 passes, so such a run always finishes
/// fully annealed to [`LR_FLOOR_MULTIPLIER`], and the operating point where the re-decoded
/// Mincer-Zarnowitz mean slope last measured 1.0058 +/- 0.0355 — one full pass still at peak
/// rate, `bardist_v2` step 10364 — is structurally unreachable in one pass. The same geometry
/// fully annealed (`bardist_v3_rfirst_1ep`, step 10817) measures 0.6653 +/- 0.0286. Raising
/// the fraction toward 1.0 is what makes the peak-rate point reachable in one epoch; see
/// [`Schedule::passes_per_lr_unit`] for why no measurement past the plateau can separate
/// "more passes" from "lower rate".
pub const LR_PLATEAU_FRACTION: f64 = 0.40;
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
/// Extra learning-rate multiplier on the MLP and dynamics down-projections, which is the
/// WHOLE multiplier those matrices get: NorMuon's native per-matrix factor is
/// `max(1, rows/cols).sqrt()`, and every down-projection is stored `[out, in]` with
/// `out < in` (`ff_out_w [512, 2048]`, `bar_dyn_fc3_w [512, 1664]`), so that factor is
/// exactly 1.0 for all eleven of them.
///
/// `4.0`, not `2.0`, for reference parity on the SAME mathematical object. modded-nanogpt
/// stores every MLP matrix as `[mlp_hdim, dim]` in one bank (`train_gpt.py:1300-1301`), so
/// its `c_proj` — the `[in, out]` transpose of our `ff_out_w` — has `rows > cols` and earns
/// `sqrt(4) = 2.0` from the same shape rule (`:510`), and then takes a deliberate `2.0` on
/// top for being a down-projection (`:513-522`): `4.0` total against our `2.0`. The
/// orthogonalized update has unit singular values, so that factor is the parameter delta's
/// spectral norm one-for-one — the reference was moving our largest parameter block twice as
/// far per step, purely because of storage orientation. Our up-projection `ff_in_w
/// [2048, 512]` already collects the native `2.0` and matches the reference's `c_fc`
/// exactly, so the deliberate 2x ratio between down- and up-projection was the part we had
/// lost.
///
/// This also doubles the decoupled decay on those eleven matrices, because the quadratic
/// form is `wd * base_lr * eff_lr` and `eff_lr` carries this multiplier linearly. That is
/// the reference's behaviour too (`:899-901` with `:930`), not a side effect of the change.
const NORMUON_DOWN_PROJECTION_LR_MULT: f64 = 4.0;

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
/// snapshot writer. Must cover the longest reported rollout horizon, which the
/// static assertion below enforces.
///
/// 100 bars, not 64: the NextLat reference measures its recursive d-step rollout over
/// teacher-forced tokens by re-applying the dynamics to its own previous prediction,
/// and what it is interested in is where that recursion DEGRADES, which needs a horizon
/// well past the one the model was shaped at (`--dyn-horizon 4`). Peak memory does not
/// move with this constant: the history is `context - SNAPSHOT_HORIZON` bars, so the KV
/// cache still holds exactly `context` tokens once the rollout finishes. Wall-clock
/// does: both the two-mode `rollout_nll` and the ancestral snapshot are linear in the
/// depth, so this is 1.56x on those two passes, and both run on a handful of pinned
/// windows rather than on the validation set.
const SNAPSHOT_HORIZON: i64 = 100;
const _: () = assert!(
    ROLLOUT_HORIZONS[ROLLOUT_HORIZONS.len() - 1] as i64 == SNAPSHOT_HORIZON,
    "the realized continuation must reach the deepest reported horizon exactly; a horizon \
     past it is silently skipped and a continuation past it is measured by nothing"
);
/// Maximum belief rows fed to the effective-rank diagnostic, which is `O(D^2 * N)`.
const EFFECTIVE_RANK_ROWS: i64 = 8192;
/// Tolerance, in nats per bar, for the reloaded-checkpoint verification.
const PROMOTION_ROUNDTRIP_TOLERANCE: f64 = 1e-4;

/// Share of an epoch's wall clock the boundary work may take before the run says so.
///
/// The Kelly bench and the ancestral snapshots are what make an epoch judgeable at all, so
/// the question is never whether to pay for them but whether the price has drifted. On the
/// production shape — a ~38-minute epoch on a shared RTX 5090, a 256-window bench and a
/// handful of snapshot windows at [`super::pretrain_reports::SNAPSHOT_SAMPLES`] draws — the
/// boundary is a small fraction of a minute against 38, i.e. far under this. 3% is roughly a
/// minute of that epoch: past it, the budgets are worth revisiting for the NEXT run, and the
/// warning says which knobs. It is a warning and never an automatic clamp, because a bench
/// whose window budget changes between two points of the same series has stopped being a
/// series.
const EPOCH_BOUNDARY_OVERHEAD_WARN: f64 = 0.03;

/// Projected delivery, as a share of the requested bar-token budget, below which every
/// epoch boundary states the shortfall.
///
/// A run that delivers 97% of its budget lost a partial batch somewhere and nobody needs
/// telling. A run that delivers 44% was sized from a ramp it is not executing, and every
/// number it produces is a number for a shorter run than the one that was asked for.
pub(crate) const BAR_TOKEN_SHORTFALL_WARN: f64 = 0.95;

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
pub const EVAL_WINDOW_SEED: u64 = 0xE7A1_5E7D_0001;

/// What promotion compares, recorded into every checkpoint's metadata and folded into its
/// lineage hash.
///
/// # The primary criterion is ECONOMIC, guarded by a density non-regression test
///
/// This is the exact MIRROR of the rule that shipped until now — that one selected on
/// `nll_bar_conditional` and guarded `nll_dof[r]`; this one selects on realized trading edge
/// and guards the density — and the inversion is a measurement, not a preference.
///
/// Run `bardist_v2` improved the traded factor's held-out NLL monotonically for its whole
/// 30000-step life: `r` went -4.8510 -> -4.9186 nats, a 0.068-nat gain, and the conditional
/// aggregate went -9.2060 -> -9.3817. Over the same span the trading it exists to do got
/// WORSE: the 0.25x cap edge fell 0.3796 -> 0.3717 -> 0.3382 bps/bar across the three
/// passes, the headline 4x edge 5.0144 -> 4.7158 -> 4.3101, quarter-Kelly Sharpe from a peak
/// of 6.28 at step 5000 to 4.96, and the realized hit rate 0.489 -> 0.485. The old rule
/// promoted step 30000: the BEST conditional NLL of the run and one of its WORST economic
/// reads. Selecting on NLL did not merely fail to help, it picked the bottom of the curve.
///
/// The arithmetic that explains it. Total achievable Kelly growth is `s^2/2` for a per-bar
/// Sharpe `s`, which this bench measures at 5.25e-4 nats/bar — confirmed twice over, by the
/// cap curve peaking at +5.44 bps at 8x and by fractional-Kelly theory putting quarter-Kelly
/// at `(2c - c^2) * g_max` = 2.30 bps against +2.45 measured. So the ENTIRE tradeable content
/// of the `r` prediction is 5.25e-4 nats/bar: 0.011% of `r`'s NLL level and 0.8% of the gain
/// the optimizer banked. Destroying half the economic value costs ~2e-4 nats, 0.3% of that
/// gain. A density objective is ~10^4 times larger than the quantity we trade and only
/// incidentally aligned with it; directional structure is the cheapest thing in the density,
/// is learned by step ~3000, and thereafter the mean drifts under no meaningful constraint
/// (uncapped `|f*|` median 9.22x -> 10.69x while the predicted tails stay WIDE at 0.67-0.89x
/// of promised, so the inflation is in the MEAN, not in a shrinking sigma).
///
/// # Why the criterion is the 0.25x cap column and not the 4x headline
///
/// At the headline 4x cap, 85% of bars sit AT the cap: the position is `4 * sign(f*)` on most
/// bars and `f*` on the rest, so the metric moves when the model's own `|f*|` inflates even
/// if its directional content is unchanged — which is precisely the confound that produced
/// the finding. At 0.25x, 99.0% of bars sit at the cap, so the position is
/// `0.25 * sign(f*)` for essentially every bar and the SIZE is a constant of the rule rather
/// than an output of the model. What survives is exactly the realized-return-weighted sign
/// accuracy of the conditional mean at fixed unit size — the one thing in the density that
/// trading consumes — and it cannot be moved by mean inflation. The cap therefore binds
/// hard on size; what it does NOT do is let the model choose the size, and that is the
/// property selection needs.
///
/// The LEVEL of this number is not a profitability claim. It is charged a flat 2 bps against
/// a measured all-in one-way cost of 11-21 bps at 1% of ADV, so it is a paired MODEL-QUALITY
/// ruler — the same unconditional-marginal null, the same windows, the same fixed context —
/// and nothing more. Its per-bar turnover is logged beside it at every decision so a reader
/// can see what the edge would cost to collect.
///
/// # Both criteria on one ruler
///
/// Edge and NLL are both taken from the DIAGNOSTIC pass: the pinned val windows at the fixed
/// [`PretrainArgs::diagnostic_context`], which every ramp stage has trained at and which does
/// not move for the life of the run. That makes consecutive decisions comparisons across
/// MODELS rather than across rulers, and it is the only context the bench is ever measured
/// at, so the noise scale the thresholds below are calibrated against is the noise scale the
/// rule actually runs at. Scoring the two criteria on two different passes would difference
/// two different quantities. Eligibility is unchanged and still gated on the DEPLOYED
/// context — a checkpoint the planner loads must have been trained at the positional range it
/// runs at — and on this run the two rulers rank models identically for NLL (the 896 and 2048
/// conditional figures agree to 0.0006-0.0022 nats at all twelve eligible reads and never
/// disagree in sign), which is the evidence that the fixed ruler is not misleading.
const SELECTION_METRIC: &str =
    "net Kelly edge over the unconditional-marginal null at the 0.25x leverage cap, measured \
     on the pinned val windows at the fixed diagnostic context, as a PAIRED per-window \
     difference against the incumbent that must clear 2.0 paired standard errors; gated by \
     TWO non-regression guards on the same pass and the same windows, nll_bar_conditional at \
     2.0 paired SE and nll_dof[r] at 1.0 paired SE. The NLL-selected artifact the previous \
     rule would have shipped is kept beside it as pretrain_best_nll.ot and scored on the test \
     split, so the rules are compared rather than asserted";
const SELECTION_WEIGHTS: [f64; BAR_DOF] = [1.0; BAR_DOF];
/// Leverage cap the economic criterion is measured at. See [`SELECTION_METRIC`] for why the
/// 0.25x column rather than the 4x headline: at 0.25x the position size is a constant of the
/// rule, so the metric cannot be moved by the model inflating its own conditional mean.
pub const SELECTION_CAP_SLOT: usize = 0;
pub const SELECTION_CAP: f64 = trade_bench::CAP_GRID[SELECTION_CAP_SLOT];
const _: () = assert!(
    SELECTION_CAP == 0.25,
    "the selection cap must be the lowest charted cap column, so the promotion criterion and \
     the charted cap curve are the same measurement"
);
/// Standard errors of the PAIRED edge difference a candidate must clear to displace the
/// incumbent.
///
/// A single read's edge interval is ~+/-1.6 bps wide, so a naive argmax over the ~31 reads of
/// a run promotes noise essentially every time. That interval is the LEVEL's, though, and the
/// comparison here is paired on identical windows, where the relevant scale is far smaller:
/// a two-way (stage x epoch) fit to the 0.25x edge over `bardist_v2`'s 32 reads leaves a
/// residual sd of 0.0200 bps against a 0.3796 bps base, i.e. 5.3%. The rule does not assume
/// that number — it block-bootstraps the paired difference at every decision — but 0.0200 bps
/// is the scale it will see, so at 2.0 SE the band is ~0.040 bps.
///
/// 2.0 and not 1.0: one-sided, 1.0 SE is a 0.159 false-promotion rate per read, which over a
/// dozen eligible reads makes a noise promotion near-certain, and because promotion is a
/// RATCHET every noise promotion permanently raises the bar against genuine ones. 2.0 puts
/// that rate at 0.023, bounds the expected number of noise promotions over a run's eligible
/// reads at ~0.3, and bounds the noise inflation of the FINAL artifact at ~0.040 bps, 11% of
/// the base. 2.0 and not 3.0: the damage the old rule actually shipped was a 0.0414 bps
/// epoch-over-epoch decline, 2.07 SE, so a 3.0-SE band would no longer resolve the very
/// regression this rule exists to stop.
const SELECTION_EDGE_SE_MULTIPLE: f64 = 2.0;
/// The factor the per-DOF guard protects.
const SELECTION_GUARD_DOF: usize = DOF_R;
/// Standard errors of the PAIRED `r` difference a candidate may drift before promotion is
/// refused. At 1.0 any regression the bench can actually resolve blocks the promotion, and
/// one it cannot resolve is by definition not evidence. UNCHANGED by the inversion of the
/// primary criterion: `r` is the factor the trade is taken on, so a resolvable regression in
/// it is the most sensitive detector available of an economic read that got lucky, and the
/// economic criterion — resolvable only to ~5% — cannot substitute for it.
const SELECTION_GUARD_SE_MULTIPLE: f64 = 1.0;
/// Standard errors of the PAIRED `nll_bar_conditional` difference a candidate may regress by
/// before promotion is refused, whatever its edge.
///
/// Looser than [`SELECTION_GUARD_SE_MULTIPLE`] on purpose, and the asymmetry is the finding
/// again. The aggregate is dominated by the intra-bar shape factors `s`, `u` and `v`, which
/// carry over a nat of headroom each, cannot affect P&L, and move far more than `r` does; a
/// 1-SE trip wire there would veto measured economic gains on movements in factors that the
/// trade never touches. At 2.0 SE, with the paired conditional SE this bench resolves to
/// ~2e-4 nats, the tolerance is ~4e-4 nats — about 76% of the ENTIRE 5.25e-4 nats/bar of
/// tradeable content in the `r` prediction. So the guard fires only when the density has
/// regressed by an amount comparable to everything trading could ever extract from it, which
/// is the only scale at which a density regression is evidence about the economics.
const SELECTION_NLL_TOLERANCE_SE_MULTIPLE: f64 = 2.0;
/// Name of the artifact holding what the PREVIOUS, NLL-primary rule would have promoted.
///
/// Kept as a first-class sidecar and scored on the test split beside the economically
/// selected `pretrain_best.ot`, because a selection rule justified only by the run that
/// motivated it is an assertion. Two artifacts on one held-out split is the comparison that
/// makes it evidence. The planner never loads this file.
const NLL_RULE_CHECKPOINT: &str = "pretrain_best_nll.ot";

/// Monte-Carlo draws behind the marginalized forecast NLL, and the number of independent
/// groups they are split into so the estimate carries a standard error.
///
/// The estimator averages the head's predictive law over `FORECAST_MC_DRAWS` ancestral draws
/// of the same-bar chain prefix, and `-log` of an average is convex, so it is biased UPWARD by
/// order `1 / draws`. 64 draws puts that bias at a few hundredths of a nat against a
/// teacher-forcing inflation measured in whole nats, and the reported group standard error
/// says how much resolution the number actually has instead of implying it is exact. Biasing
/// the honest number pessimistically is the safe direction.
const FORECAST_MC_DRAWS: usize = 64;
const FORECAST_MC_GROUPS: usize = 4;
const _: () = assert!(
    FORECAST_MC_DRAWS % FORECAST_MC_GROUPS == 0,
    "the pooled mixture is the mean of the group mixtures, which requires equal groups"
);
/// Bar positions between successive rows of the forecast estimate.
///
/// The estimator costs `FORECAST_MC_DRAWS * BAR_DOF` small projections per row, so it runs on
/// a strided subset. The teacher-forced figure it is compared against is taken on EXACTLY the
/// same rows, which makes the inflation a paired difference rather than two numbers measured
/// on different data; and at 4096 windows the strided subset is still hundreds of thousands of
/// bars, far more than the mean needs.
const FORECAST_POSITION_STRIDE: i64 = 8;

/// Default optimizer steps between step-tagged crash-recovery checkpoints.
///
/// Job 2856 ran 13831 steps in 3 epochs — 4610 steps per epoch — and produced its first
/// promoted checkpoint at step 9221, because promotion is gated on the deployed context and
/// the memory-aware ramp held the batch. An OOM at step 9220 would have destroyed two epochs
/// of compute. 512 steps is ~11% of an epoch there and ~4% of the shortest ramp stage, so the
/// worst case a crash can cost is a ninth of an epoch, while the write itself — a 128 MiB
/// `vs.save` plus two small sidecars — lands about once per two minutes at that run's 4.5
/// step/s and is invisible beside the optimizer. Zero still disables the path entirely, for
/// the smoke runs whose whole point is to leave nothing behind.
pub const DEFAULT_CHECKPOINT_EVERY: usize = 512;
/// Step-tagged checkpoints kept on disk. Three bounds the rolling window at ~400 MiB of
/// weights while still surviving a crash that corrupts the newest file mid-write. Epoch
/// artifacts and `pretrain_best*.ot` are named differently and are never pruned.
const RETAINED_STEP_CHECKPOINTS: usize = 3;

/// Step-tagged checkpoints exempted from pruning because they BRACKET THE FIRST PASS
/// BOUNDARY while the learning rate is still clipped flat. This is the only design in the
/// recipe that can attribute anything to REPETITION.
///
/// `Schedule::lr_multiplier` is `plateau` for `progress <= F`, the run's plateau fraction, and then
/// EXACTLY AFFINE in step down to `LR_FLOOR_MULTIPLIER`. Affine means every contrast taken
/// in the decay region is rank 1 in (passes, lr): with the realized batch ramp flat the
/// plateau term is 1, so `d(passes)/d(lr_mult) = -epochs * (1 - F) / (1 - L)`, a pure
/// function of the recipe — independent of corpus size, steps per epoch and batch. Two
/// checkpoints past the plateau cannot tell "saw the data again" from "trained at a lower
/// rate" at ANY precision, because the two effects are the same variable there. `bardist_v2`
/// proved this the expensive way: it retained four scoreable artifacts and every one of them
/// except the first sat in the decay region, so its repetition question is unanswerable from
/// the artifacts that survived.
///
/// The escape is a NONLINEARITY, not a longer lever arm. Inside the plateau `lr_mult` is
/// clipped to a constant, and `Schedule::stage_at` is `step % steps_per_epoch`, so the ramp
/// stage, the realized batch, the conditioning depth and — via `PassPlan::counts`, which is
/// computed once and takes no epoch — the symbol composition all REPEAT identically per
/// epoch. At `step == steps_per_epoch` exactly one thing in the entire recipe changes: bars
/// start being visited a second time. A kink there is repetition and nothing else, which is
/// why anchors are placed to BRACKET it rather than to span the plateau: a discontinuity at a
/// pre-registered step needs no extrapolation, while a long-baseline slope difference mixes
/// repetition with "the model kept learning" and, at the low end, with "the model was a
/// random init".
///
/// Three anchors: two tight around the boundary to carry the discontinuity, and one earlier
/// in-plateau anchor to establish the all-fresh trend the kink is measured against. Cost is a
/// fixed 3 x 128 MiB, so the pruner stays crash insurance with a bounded footprint rather
/// than an archive.
///
/// REQUIRES `epochs * F > 1` for the run's own `F` = `--lr-plateau-fraction`, i.e. the first
/// pass boundary must fall strictly inside the plateau. The condition is evaluated against the
/// fraction the run is using, never against a fixed number: at the default `F = 0.40` it takes
/// `--epochs 3` or more, at `--epochs 2` it takes `F > 0.5`, and a single-pass run cannot
/// satisfy it at any `F < 1` because one pass has no repetition boundary at all.
/// [`plateau_anchor_tags`] returns nothing when the condition fails rather than retaining
/// artifacts that cannot answer the question.
const PLATEAU_ANCHOR_CHECKPOINTS: usize = 3;

/// Step tags [`Trainer::prune_step_checkpoints`] must not delete, bracketing the first pass
/// boundary at a constant learning rate. `tags` must be sorted ascending.
///
/// Returned in PRIORITY order — the straddling pair first, then the all-fresh baseline —
/// because the pair carries the discontinuity and the baseline only calibrates the trend it is
/// measured against. At `--epochs 3` and the default plateau fraction,
/// `steps_per_epoch = 10365` and [`DEFAULT_CHECKPOINT_EVERY`] `= 512` this returns 10240,
/// 10752, 5120: the last cadence step before the boundary, the first one after it, and the
/// midpoint of the first pass.
///
/// Empty — deliberately, rather than retaining artifacts that cannot answer the question —
/// when the boundary is not strictly inside the plateau (a single-pass run has no repetition,
/// and `epochs * lr_plateau_fraction <= 1` leaves the boundary in the LR decay where the
/// repetition and rate coefficients are collinear), or when the cadence wrote nothing on one
/// side of it.
fn plateau_anchor_tags(
    tags: &[usize],
    steps_per_epoch: usize,
    total_steps: usize,
    lr_plateau_fraction: f64,
) -> Vec<usize> {
    let plateau_end = (lr_plateau_fraction * total_steps as f64).floor() as usize;
    let boundary = steps_per_epoch;
    if boundary == 0 || boundary >= plateau_end {
        return Vec::new();
    }
    // Strictly after the boundary and still at the clipped rate: the repeated-data side.
    let Some(after) = tags
        .iter()
        .copied()
        .find(|&tag| tag > boundary && tag <= plateau_end)
    else {
        return Vec::new();
    };
    // Strictly before it: the last all-fresh artifact. Tag 0 is a random init, never an anchor.
    let Some(before) = tags
        .iter()
        .copied()
        .filter(|&tag| tag > 0 && tag <= boundary)
        .next_back()
    else {
        return Vec::new();
    };
    let mut anchors = vec![before, after];
    // The all-fresh trend needs a real lever arm without reaching back into the untrained
    // regime, so the baseline is the artifact nearest the MIDPOINT of the first pass.
    if let Some(baseline) = tags
        .iter()
        .copied()
        .filter(|&tag| tag > 0 && tag < before)
        .min_by_key(|&tag| tag.abs_diff(boundary / 2))
    {
        anchors.push(baseline);
    }
    anchors.truncate(PLATEAU_ANCHOR_CHECKPOINTS);
    anchors
}

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
    /// Extra bar resolutions, in seconds, trained on ALONGSIDE `resolution_secs`.
    ///
    /// EMPTY BY DEFAULT. `long_data/bars` holds `*.86400.bars` beside `*.300.bars`, so the daily
    /// corpus enters a run only when it is asked for by name — the first run with it has to be a
    /// deliberate A/B against a baseline, not a silent change to the default corpus.
    ///
    /// Each auxiliary resolution gets its own fitted supports, its own ramp contexts
    /// ([`super::pretrain_aux::AUXILIARY_CONTEXTS`]) and its own pass partition, and contributes
    /// steps ADDITIVE to the deployment pass. Selection and promotion remain on the deployment
    /// resolution's held-out `nll_bar_conditional`: a model that got better at daily bars and
    /// worse at five-minute bars must lose.
    pub auxiliary_resolutions: Vec<u32>,
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
    /// Weight on the NextLat latent term, applied UNCHANGED at every step of the run.
    ///
    /// `dyn` is meaned over every element of `[B, T, BAR_MODEL_DIM]` (see
    /// [`next_lat_loss`]), so the weight is width-independent and `1.0` is the reference
    /// setting; the CLI default is `1e-2` because at `1.0` the term measured 62% of this
    /// objective's magnitude.
    ///
    /// # There is no anneal, and there must not be one
    ///
    /// An earlier revision multiplied both this and [`Self::lambda_kl`] by a `Schedule::
    /// aux_weight` decaying linearly to EXACTLY zero at 2/3 of the run, justified as
    /// copying modded-nanogpt, which "anneals its MTP heads to zero and drops them stage by
    /// stage". That citation was a MISREADING, and it shipped a broken dynamics head. Both
    /// halves of the record, so nobody re-derives it:
    ///
    /// - modded-nanogpt's `mtp_weights` is a weight vector over PREDICTION OFFSETS inside
    ///   one fused cross-entropy against one shared `lm_head`
    ///   (`modded-nanogpt/train_gpt.py:1687`). The kernel settles it: it loops
    ///   `for k in 0..n_predict`, reads the target at `blockIdx.x + k` and scales that
    ///   token's cross-entropy by `mtp_weights[k]`
    ///   (`modded-nanogpt/triton_kernels.py:1073-1077`) — one row of logits, one `lm_head`,
    ///   `k` offsets. There is no MTP module and no MTP parameter.
    ///   The whole path is `if self.training:` (`train_gpt.py:1685`); evaluation takes the
    ///   plain single-target `F.cross_entropy` branch (`train_gpt.py:1688-1692`). So what it
    ///   anneals away is training-only scaffolding that is already absent at inference, and
    ///   annealing it costs exactly nothing. Note also that offset 0 — the real next-token
    ///   objective — is held at `1.0` in EVERY stage
    ///   (`train_gpt.py:1980-1988`); the reference never anneals the weight of the thing it
    ///   ships.
    /// - NextLat, which is where these two lambdas actually come from, does NOT anneal
    ///   them. `lambda_kl` and `lambda_mse` are plain config floats defaulting to `1.0`
    ///   (`NextLat/models/model_nextlat.py:40-41`), threaded once into the model
    ///   (`NextLat/core_train.py:90-94`) and read directly in the loss
    ///   (`NextLat/models/model_nextlat.py:490-493`). No schedule touches them anywhere;
    ///   `core_train.py:317-318` schedules the learning rate and nothing else. And NextLat
    ///   SHIPS its dynamics MLP: `NextLatDynamicsModel`
    ///   (`NextLat/models/model_nextlat.py:47-95`) is instantiated as a real submodule at
    ///   `:122` and driven at inference under `@torch.inference_mode()` by
    ///   `speculative_propose` (`:678-703`), which advances the draft state through it at
    ///   `:701`.
    ///
    /// [`BarDynamics`] is on NextLat's side of that line, not modded-nanogpt's. It is a
    /// shipped component of the frozen inference bundle (`world_model.rs:15-18`), its
    /// weights are in the checkpoint, and [`RolloutMode::Dynamics`] advances beliefs
    /// through it. Its only training signal is these two terms. Annealing them to zero
    /// leaves the trunk training for another third of the run while the dynamics head is
    /// frozen against a moving target, and the head rots: in job 2865 `dyn/identity` was
    /// 0.63-0.75 while the terms were live — the MLP beating the trivial identity map by
    /// ~1.4x — and 154x at step 9580, the first evaluation after the weight hit zero. Every
    /// promoted checkpoint came from that dead zone. [`check_dynamics_beats_identity`] is
    /// the end-of-run guard that would have caught it on the first run.
    pub lambda_dyn: f64,
    /// Weight on the NextLat KL term, applied UNCHANGED at every step. See
    /// [`Self::lambda_dyn`] for why nothing anneals it.
    pub lambda_kl: f64,
    /// Weight on the EXPECTED-LOG-GROWTH term, applied UNCHANGED at every step.
    ///
    /// The default is [`LAMBDA_GROWTH`], which was derived from a gradient-norm
    /// measurement rather than swept; that constant's doc comment carries the measurement.
    /// `0.0` is the ablation's control arm: the term is still computed and still charted,
    /// it simply does not enter the objective, so the two arms differ in exactly one
    /// number and their `pretrain_growth_term` panels are directly comparable.
    ///
    /// Like [`Self::lambda_dyn`] and [`Self::lambda_kl`], nothing anneals it. The reasons
    /// are on `lambda_dyn`, and one more applies here: the whole finding this term answers
    /// is that the economics decay LATE in a run, so a weight that decayed with the
    /// schedule would switch the term off exactly when it is needed.
    pub lambda_growth: f64,
    /// Held-out windows in each pinned evaluation set. Pinned by [`EVAL_WINDOW_SEED`], so
    /// they are identical across runs, seeds and ablations.
    pub validation_windows: usize,
    /// Fixed context of the across-run diagnostic evaluation.
    pub diagnostic_context: i64,
    /// Pinned windows carried into the candle-rollout snapshot reports.
    pub snapshot_windows: usize,
    /// Ancestral draws behind each snapshot window's quantile fan.
    ///
    /// This is what the fan is ESTIMATED from, and the estimate is not free: the standard
    /// error of a sample quantile at probability `q` is `sqrt(q(1-q)/n) / f(x_q)`, so
    /// halving the count widens every band's own error bar by `sqrt(2)`. The rollout is
    /// linear in it and now runs at EVERY epoch boundary, so it is a flag rather than a
    /// constant: [`super::pretrain_reports::SNAPSHOT_SAMPLES`] is the production value and a
    /// smoke run has no use for 256 draws it will never read. Floored at
    /// [`super::pretrain_reports::MIN_FAN_SAMPLES`], below which the 25th and 75th
    /// percentiles of the draws coincide and the picture reports a band of exactly zero.
    pub snapshot_samples: usize,
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
    /// Refuse to start if measured capacity would REDUCE `batch_size`, instead of clamping.
    ///
    /// # The confound this exists to make unlaunchable
    ///
    /// The capacity probe reads free VRAM at startup and clamps the base batch to what the
    /// card can actually hold. That is right for a long production run — it is why a run
    /// survives a shared card at all — and it is categorically wrong for a controlled
    /// experiment, because the clamp depends on what ELSE was resident at launch.
    ///
    /// Measured, on the two arms of the expected-log-growth ablation, both launched
    /// `--batch-size 24` with identical seed and identical config:
    ///
    /// | run | free VRAM at the probe | base batch | steps |
    /// |-----|------------------------|------------|-------|
    /// | `growth_ablation_lambda0`  | 16.37 GiB | 23 | 10818 |
    /// | `growth_ablation_lambda77` | 14.94 GiB | 21 | 11847 |
    ///
    /// The bar budget is fixed by `--epochs`, so the step count moves inversely with the
    /// batch and the pair silently differed in gradient-noise level and in the LENGTH of the
    /// lr and momentum schedules. Both banners were individually honest; the comparison
    /// between them was not. Survival-by-degradation and controlled comparison are opposite
    /// requirements and only the caller knows which one it is running, so this is a flag and
    /// not a policy.
    ///
    /// With it set, a short-fall is an error naming the deficit and the deficit's cause,
    /// before the first step. [`super::pretrain_stats::compare_runs`] is the backstop for
    /// anything this cannot see: it compares REALIZED batch and step count and refuses the
    /// pair, because both of those runs requested 24.
    pub exact_batch: bool,
    /// Fraction of the run held at the flat learning-rate plateau before the linear decay to
    /// [`LR_FLOOR_MULTIPLIER`]. Defaults to [`LR_PLATEAU_FRACTION`]; must lie strictly inside
    /// `(0, 1)`.
    ///
    /// # Why this is a flag
    ///
    /// Past the plateau `lr_multiplier` is EXACTLY AFFINE in the step index, so every contrast
    /// taken there is rank 1 in `(passes, lr)` — `d(passes)/d(lr_mult) = -epochs * (1 - F) /
    /// (P - L)`, a pure function of the recipe ([`Schedule::passes_per_lr_unit`]). "Saw the
    /// data again" and "trained at a lower rate" are the same variable there, at any
    /// precision. The only stretch of a run where they are separable is the plateau, where the
    /// rate is clipped flat.
    ///
    /// Under a one-epoch budget the default puts the end of that stretch at 0.4 passes, so a
    /// run cannot reach one full pass at peak rate — the operating point where the re-decoded
    /// mean slope measured 1.0058 +/- 0.0355 — while a fully annealed single pass measures
    /// 0.6653 +/- 0.0286. Raising `F` is what turns that confound into a measurement instead
    /// of an algebraic identity, and the value is recorded in the checkpoint metadata and in
    /// the run's report so a future reader can tell which schedule produced a number.
    pub lr_plateau_fraction: f64,
}

impl PretrainArgs {
    fn corpus_flags(&self) -> CorpusFlags {
        CorpusFlags {
            data_dir: self.data_dir.clone(),
            resolution_secs: self.resolution_secs,
            min_bars: self.min_bars,
            split_bounds: self.split_bounds,
            derive_split_bounds: self.derive_split_bounds,
            min_dollar_volume: self.min_dollar_volume,
        }
    }
}

/// The flags that decide WHICH bars exist and where the splits fall.
///
/// Factored out of [`PretrainArgs`] because anything that wants to score, chart or
/// picture a checkpoint on the run's own held-out windows has to reproduce this
/// corpus exactly — a different `--min-bars` or a derived boundary silently changes
/// which symbols survive, which moves every pinned window.
#[derive(Clone, Debug)]
pub struct CorpusFlags {
    pub data_dir: String,
    pub resolution_secs: u32,
    pub min_bars: usize,
    pub split_bounds: Option<(i64, i64)>,
    pub derive_split_bounds: bool,
    pub min_dollar_volume: f64,
}

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

/// Context length at a ramp stage, 64-aligned for the attention kernels.
pub(super) fn stage_context(stage: usize) -> i64 {
    stage_context_for(stage, BAR_MAX_CONTEXT)
}

/// [`stage_context`] for a counterfactual deployed context.
///
/// Exists so the banner can price the batch/context frontier — the achievable batch if the
/// deployed context were something else — with the SAME ramp geometry the run would use,
/// rather than a hand-rolled approximation of it. [`CONTEXT_FRONTIER`] is the set of
/// counterfactuals it is called with; nothing changes the deployed context.
fn stage_context_for(stage: usize, deployed: i64) -> i64 {
    debug_assert!(stage < RAMP_STAGES);
    let span = deployed - BAR_CONTEXT_RAMP_START;
    let raw = BAR_CONTEXT_RAMP_START + span * stage as i64 / (RAMP_STAGES as i64 - 1);
    raw - raw % 64
}

/// Every ramp context, ascending. The vector a [`PassPlan`] tiles the corpus with.
pub(super) fn stage_contexts() -> Vec<i64> {
    (0..RAMP_STAGES).map(stage_context).collect()
}

/// State the pass partition before a single step is taken: how much of the split one epoch
/// reaches, what it cannot reach and why, and how much history each stage's bars are predicted
/// from.
///
/// Printed rather than left to the reports because it is the answer to "what does `--epochs 1`
/// mean on this corpus", and that has to be legible before four hours of compute, not after.
fn print_pass_plan(pass: &PassPlan, base_batch: usize, batch_ramp: &[usize; RAMP_STAGES]) {
    let remainder = pass.remainder();
    let shares = pass.stage_bar_shares();
    let weights = ramp_token_weights(batch_ramp);
    let weight_sum: f64 = weights.iter().sum();
    let stages: Vec<String> = (0..RAMP_STAGES)
        .map(|stage| {
            format!(
                "stage {stage}: {} windows x {} bars = {} bars ({:.2}% of the pass against a \
                 {:.2}% token budget), {} steps at batch {}, mean history {:.1} bars",
                pass.windows_per_stage()[stage],
                stage_context(stage),
                pass.windows_per_stage()[stage] as u64 * stage_context(stage) as u64,
                100.0 * shares[stage],
                100.0 * weights[stage] / weight_sum,
                pass.windows_per_stage()[stage]
                    .div_ceil((base_batch * batch_ramp[stage]).max(1)),
                base_batch * batch_ramp[stage],
                pass.mean_conditioning_bars(stage),
            )
        })
        .collect();
    println!(
        "pass plan      one epoch = {} of {} training bars as a prediction target EXACTLY ONCE \
         ({:.4}%), bar-weighted mean history {:.1} bars",
        pass.covered_bars(),
        pass.split_bars(),
        100.0 * pass.covered_bars() as f64 / pass.split_bars().max(1) as f64,
        pass.pass_mean_conditioning_bars(),
    );
    println!(
        "pass remainder {} bars ({:.4}%) cannot be a target in ANY epoch: {} head bars (bar 0 \
         carries no DOF and the first window's anchor is an input, 2 per symbol), {} bars in {} \
         symbols shorter than the {}-bar shortest context, {} bars of sub-context hole (one \
         per symbol, always below {} bars, placed at a per-epoch RANDOM block boundary so it is \
         not always the bars just before the split instant)",
        remainder.total(),
        100.0 * remainder.total() as f64 / pass.split_bars().max(1) as f64,
        remainder.head_bars,
        remainder.short_symbol_bars,
        remainder.short_symbols,
        stage_context(0),
        remainder.hole_bars,
        stage_context(0),
    );
    println!("pass stages    {}", stages.join(" | "));
}

/// Per-stage bar-token budget for one unit of base batch size: `batch_ramp[s] * context[s]`.
///
/// This is the weight vector a [`PassPlan`] partitions the corpus by, so a stage's share of
/// the bars is exactly its share of the run's bar-tokens. It takes the ramp rather than
/// reading [`BATCH_RAMP`] because the run executes a capacity-derived ramp: weighting the
/// partition by the DECLARED ramp while the run executes another is the same class of defect
/// that made job 2856's `--epochs 3` deliver 1.33 epochs — it planned 79,872 bar-tokens a step
/// and delivered 35,328.
pub(super) fn ramp_token_weights(batch_ramp: &[usize; RAMP_STAGES]) -> [f64; RAMP_STAGES] {
    std::array::from_fn(|stage| (batch_ramp[stage] as i64 * stage_context(stage)) as f64)
}

/// Resolved training schedule. Every per-step quantity is a pure function of the step index
/// AND of `batch_ramp`.
///
/// `batch_ramp` is the ramp DERIVED from measured capacity by
/// [`CapacityModel::derive_batch_ramp`], not [`BATCH_RAMP`], and it is also the one place a
/// runtime memory hold is applied: [`Trainer::hold_batch_if_short_of_vram`] lowers a stage's
/// multiplier in place, and [`Self::batch`], [`Self::bars_per_step`] and
/// [`Self::lr_multiplier`] all read it, so the learning-rate plateau bump can never describe
/// a batch the run is not using.
///
/// That hold is the one thing in the recipe that is NOT reproducible from the flags: it
/// depends on what the shared card's other tenants held at one instant. It is therefore
/// announced loudly, states the learning-rate change it implies, and — since it makes a stage
/// run out of steps before it has issued its share of the pass — is caught at the end of the
/// run as an INCOMPLETE EPOCH rather than left to be inferred from a reuse ratio.
///
/// **The ramp's step boundaries come from the corpus partition, not from a step fraction.**
/// `stage(step)` used to be `step * RAMP_STAGES / total_steps`, an equal-thirds split of a
/// step count derived from a bar-token target. That could not make an epoch a full pass: each
/// stage got a third of the steps regardless of how many windows its share of the corpus
/// actually contained, so the stages under- or over-shot their lists and the union of what
/// they issued covered 71% of the corpus. Now `stage_steps[s] = ceil(windows[s] / batch[s])`
/// straight off [`PassPlan::windows_per_stage`], the last step of each stage runs a SHORT
/// batch, and one epoch is exactly `sum(stage_steps)` steps — the ramp, once, over a
/// partition of the corpus. With `--epochs E > 1` the whole ramp REPEATS per epoch: an epoch
/// is a pass, and a pass is tiled by all three contexts, so the curriculum has to restart for
/// each one. (The old geometry made `--epochs 3` mean "epoch 0 is entirely context 896",
/// which was never the intent.)
#[derive(Clone, Copy, Debug)]
pub(super) struct Schedule {
    total_steps: usize,
    /// Optimizer steps each ramp stage runs per epoch: `ceil(windows[s] / batch[s])`.
    stage_steps: [usize; RAMP_STAGES],
    /// `sum(stage_steps)`, at least 1. One epoch is one pass is this many steps.
    steps_per_epoch: usize,
    base_batch: usize,
    /// Batch-size multiplier ACTUALLY used at each ramp stage.
    batch_ramp: [usize; RAMP_STAGES],
    /// Fraction of `total_steps` held at the flat learning-rate plateau: the run's
    /// `--lr-plateau-fraction`, strictly inside `(0, 1)` by [`validate_args`].
    ///
    /// Carried here rather than read off [`LR_PLATEAU_FRACTION`] per step because everything
    /// derived from it — [`Self::lr_multiplier`], [`Self::plateau_last_step`],
    /// [`Self::in_lr_plateau`], [`Self::passes_per_lr_unit`] and [`plateau_anchor_tags`] — has
    /// to describe the run that is EXECUTING and not the default recipe.
    lr_plateau_fraction: f64,
    momentum_warmup: usize,
    momentum_cooldown: usize,
}

impl Schedule {
    pub(super) fn new(
        stage_steps: [usize; RAMP_STAGES],
        total_steps: usize,
        base_batch: usize,
        batch_ramp: [usize; RAMP_STAGES],
        lr_plateau_fraction: f64,
    ) -> Self {
        let total_steps = total_steps.max(1);
        let steps_per_epoch = stage_steps.iter().sum::<usize>().max(1);
        let momentum_warmup = MOMENTUM_WARMUP_STEPS.min(total_steps / 2);
        let momentum_cooldown =
            MOMENTUM_COOLDOWN_STEPS.min(total_steps.saturating_sub(momentum_warmup));
        Self {
            total_steps,
            stage_steps,
            steps_per_epoch,
            base_batch,
            batch_ramp,
            lr_plateau_fraction,
            momentum_warmup,
            momentum_cooldown,
        }
    }

    /// Steps each stage needs to issue every window the pass assigned it.
    ///
    /// `ceil`, not `floor`: flooring would drop the partial final batch of every stage, which
    /// is up to `batch - 1` windows — up to one batch per stage per epoch left untargeted, and
    /// a coverage invariant that no schedule could satisfy. The short step is the price, and it
    /// is three noisier gradients out of ~10,000.
    fn steps_for_pass(
        windows_per_stage: &[usize],
        base_batch: usize,
        batch_ramp: &[usize; RAMP_STAGES],
    ) -> [usize; RAMP_STAGES] {
        std::array::from_fn(|stage| {
            let batch = (base_batch * batch_ramp[stage]).max(1);
            windows_per_stage[stage].div_ceil(batch)
        })
    }

    /// The ramp stage `step` runs at, and how many steps it has already taken inside that
    /// stage THIS epoch.
    fn stage_at(&self, step: usize) -> (usize, usize) {
        let mut within = step % self.steps_per_epoch;
        for stage in 0..RAMP_STAGES {
            if within < self.stage_steps[stage] {
                return (stage, within);
            }
            within -= self.stage_steps[stage];
        }
        // Only reachable when every stage has zero steps, which `steps_per_epoch.max(1)`
        // already forced into a one-step epoch.
        (RAMP_STAGES - 1, 0)
    }

    fn stage(&self, step: usize) -> usize {
        self.stage_at(step).0
    }

    /// The pass `step` belongs to. Exact rather than derived from bar-tokens consumed: an
    /// epoch is a fixed number of steps because it is a fixed partition of the corpus.
    fn epoch_of(&self, step: usize) -> usize {
        step / self.steps_per_epoch
    }

    /// True on the last step of a pass, i.e. exactly where every stage has issued its whole
    /// share and the coverage invariant is due.
    fn completes_epoch(&self, step: usize) -> bool {
        (step + 1) % self.steps_per_epoch == 0
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
    pub(super) fn lr_multiplier(&self, step: usize) -> f64 {
        self.lr_multiplier_for(step, self.batch_ramp[self.stage(step)])
    }

    /// [`Self::lr_multiplier`] with the batch multiplier supplied, so a memory hold can
    /// state the rate the run WOULD have used beside the one it will.
    fn lr_multiplier_for(&self, step: usize, batch_multiple: usize) -> f64 {
        let plateau = (batch_multiple as f64).powf(BATCH_RAMP_LR_EXPONENT[self.stage(step)]);
        let progress = step as f64 / self.total_steps as f64;
        if progress <= self.lr_plateau_fraction {
            return plateau;
        }
        let decayed =
            ((progress - self.lr_plateau_fraction) / (1.0 - self.lr_plateau_fraction)).min(1.0);
        plateau + (LR_FLOOR_MULTIPLIER - plateau) * decayed
    }

    pub(super) fn total_steps(&self) -> usize {
        self.total_steps
    }

    pub(super) fn steps_per_epoch(&self) -> usize {
        self.steps_per_epoch
    }

    /// Passes over the training corpus completed by `step`. One epoch is one pass, so this is
    /// the step index in units of [`Self::steps_per_epoch`], fractional mid-pass.
    ///
    /// Exposed because analysis code outside the trainer had NO way to read either axis a
    /// checkpoint sits on, which is why it took algebra rather than a chart to notice that past
    /// the plateau the two axes are one axis. See [`super::lr_disentangle`].
    pub(super) fn passes_at(&self, step: usize) -> f64 {
        step as f64 / self.steps_per_epoch as f64
    }

    /// Batch-bump term of the learning-rate multiplier at `stage`: the flat plateau value.
    fn lr_plateau_term(&self, stage: usize) -> f64 {
        (self.batch_ramp[stage] as f64).powf(BATCH_RAMP_LR_EXPONENT[stage])
    }

    /// True when every ramp stage carries the SAME batch bump, so [`Self::lr_multiplier`] is
    /// affine in the step index across the whole decay instead of piecewise-affine with a jump
    /// at each stage boundary.
    ///
    /// It holds whenever the realized `batch_ramp` is flat, because then every bump is
    /// `k^exponent` for one `k`. `CapacityModel::derive_batch_ramp(24)` returns `[1, 1, 1]` on
    /// the card this was measured on, so it holds there with every bump exactly `1`.
    pub(super) fn lr_affine_in_step(&self) -> bool {
        let first = self.lr_plateau_term(0);
        (1..RAMP_STAGES).all(|stage| self.lr_plateau_term(stage) == first)
    }

    /// Last step whose learning-rate multiplier is still the flat plateau value.
    ///
    /// [`Self::lr_multiplier_for`] holds the plateau while `step / total_steps <=
    /// lr_plateau_fraction`, so this is the largest step satisfying that. It is the end of the
    /// ONLY stretch of the run across which passes accumulate at zero learning-rate contrast.
    pub(super) fn plateau_last_step(&self) -> usize {
        ((self.lr_plateau_fraction * self.total_steps as f64).floor() as usize)
            .min(self.total_steps)
    }

    pub(super) fn in_lr_plateau(&self, step: usize) -> bool {
        step as f64 / self.total_steps as f64 <= self.lr_plateau_fraction
    }

    /// Passes bought per unit of learning-rate multiplier GIVEN UP, anywhere past the plateau.
    /// `NaN` when [`Self::lr_affine_in_step`] is false, because then there is no one such number
    /// and a returned value would be an average masquerading as an identity.
    ///
    /// Past the plateau both axes are affine in the step index — `passes = step /
    /// steps_per_epoch` and `lr_mult = P - (P - LR_FLOOR_MULTIPLIER) * (step/total_steps -
    /// F)/(1 - F)` — so their ratio is a CONSTANT, and `total_steps = epochs * steps_per_epoch`
    /// cancels the corpus out of it:
    ///
    /// ```text
    /// d(passes)/d(lr_mult) = -epochs * (1 - F) / (P - LR_FLOOR_MULTIPLIER)
    /// ```
    ///
    /// A pure function of the recipe, `F` being the run's own plateau fraction:
    /// `-3 * 0.6 / 0.85 = -36/17` for a three-epoch run at the default `F` and a flat
    /// ramp. Every pair of steps past the plateau therefore moves along ONE direction in
    /// `(passes, lr_mult)` space, and no such pair can tell "another pass over the corpus" apart
    /// from "a lower learning rate" — there, they are the same variable.
    pub(super) fn passes_per_lr_unit(&self) -> f64 {
        if !self.lr_affine_in_step() {
            return f64::NAN;
        }
        let epochs = self.total_steps as f64 / self.steps_per_epoch as f64;
        -epochs * (1.0 - self.lr_plateau_fraction)
            / (self.lr_plateau_term(0) - LR_FLOOR_MULTIPLIER)
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
// Measured device capacity
// ---------------------------------------------------------------------------

/// What the card can actually hold, measured before the first optimizer step.
///
/// A training step's device footprint is modelled as affine in bar-tokens,
/// `step_bytes(batch, context) = fixed_bytes + per_token_bytes * batch * context`, fitted
/// from the two [`CAPACITY_PROBE_BATCHES`] shapes at the deployed context. Linearity is not
/// an assumption: three production probes across a 2.3x token range measured 471,040,
/// 507,401 and 494,883 B/bar-token, because FA4 flash attention never materializes the
/// `O(T^2)` score matrix, so there is no quadratic term to miss.
///
/// Everything the ramp plans is derived from this, which is the entire point. The declared
/// [`BATCH_RAMP`] asks for `x3` of a base 24 at 2048 bars — 147,456 bar-tokens, ~70 GiB at
/// the measured rate, on a 32 GiB card. That target is unreachable by more than 2x on an
/// IDLE card, forever, and no amount of waiting or luck changes it. Discovering that at a
/// stage transition four hours in, as a reactive hold, produced a run whose announced
/// schedule was fiction and whose learning-rate plateau bumps were tuned for a batch it
/// never ran. A schedule derived from this model is one the banner can honestly print.
#[derive(Clone, Copy, Debug)]
struct CapacityModel {
    /// Marginal device bytes per bar-token: the slope between the two probe shapes.
    per_token_bytes: f64,
    /// Batch-independent bytes a step costs on top of the baseline — cuBLAS workspaces,
    /// allocator rounding, and the optimizer state that is not resident yet. Only the LAZY
    /// part is added: NorMuon's buffers are allocated eagerly in `Muon::new_named` and are
    /// therefore already inside [`Self::baseline_bytes`], so the probe charges
    /// `steady_state_bytes - state_bytes`, which is exactly the AdamW moments no forward and
    /// backward pass ever brings into existence.
    fixed_bytes: f64,
    /// Free device bytes with the caching allocator's pool RELEASED, i.e. what a training
    /// step has to fit into once the weights, the gradients, the CUDA context and the card's
    /// other tenants are accounted for.
    free_bytes: u64,
    /// Device bytes in use at that same instant. Reported for the banner's arithmetic;
    /// nothing plans against it.
    baseline_bytes: u64,
}

impl CapacityModel {
    /// Device bytes one step at `batch` x `context` adds above the baseline.
    fn step_bytes(&self, batch: usize, context: i64) -> f64 {
        self.fixed_bytes + self.per_token_bytes * batch as f64 * context as f64
    }

    /// Bytes that must be FREE for a step at `batch` x `context` when the allocator's pool
    /// already holds `previous_bytes` worth of the preceding stage's shape.
    ///
    /// The [`RAMP_MEMORY_MARGIN`] is charged on the INCREMENT, not on the whole footprint,
    /// because that is what it measures: the caching allocator cannot reuse a block of the
    /// wrong shape, so growing fragments the pool by roughly what is newly allocated before
    /// it settles. Charging it on the absolute footprint instead would refuse the batch 24 at
    /// 2048 bars that job 2856 demonstrably ran with 4.73 GiB still free, i.e. it would throw
    /// away a quarter of the card to a margin the measurement does not support.
    fn required_bytes(&self, batch: usize, context: i64, previous_bytes: f64) -> f64 {
        let step = self.step_bytes(batch, context);
        step + RAMP_MEMORY_MARGIN * (step - previous_bytes).max(0.0)
            + RAMP_MEMORY_RESERVE_BYTES as f64
    }

    /// Largest batch whose step fits at `context` given a pool holding `previous_bytes`.
    ///
    /// Closed-form inverse of [`Self::required_bytes`] on its GROWING branch, which is the
    /// only one a ramp can take: contexts increase at every stage and the derived
    /// multipliers never decrease, so a stage's footprint always exceeds its predecessor's.
    /// On the shrinking branch the formula under-reports, never over-reports.
    fn max_batch(&self, context: i64, previous_bytes: f64) -> usize {
        if self.per_token_bytes <= 0.0 || context <= 0 {
            return 0;
        }
        let budget = (self.free_bytes as f64 - RAMP_MEMORY_RESERVE_BYTES as f64
            + RAMP_MEMORY_MARGIN * previous_bytes)
            / (1.0 + RAMP_MEMORY_MARGIN)
            - self.fixed_bytes;
        if budget <= 0.0 {
            return 0;
        }
        (budget / (self.per_token_bytes * context as f64)).floor() as usize
    }

    /// Largest batch the ramp's FINAL stage could hold if the deployed context were
    /// `deployed`, with the batch flat across the ramp.
    ///
    /// This is the frontier the banner prints. It is the honest form of the tradeoff: at a
    /// fixed cost per bar-token the card caps the PRODUCT `batch * context`, so halving the
    /// context does not merely help, it roughly doubles the batch — and larger batches are
    /// what the ported modded-nanogpt recipe is tuned for. The final stage is the binding
    /// one because `context + margin * (context - previous context)` grows with the stage
    /// index, so a flat ramp that fits at the last stage fits at every earlier one.
    fn frontier_batch(&self, deployed: i64) -> usize {
        let last = RAMP_STAGES - 1;
        let context = stage_context_for(last, deployed);
        let previous = if last == 0 {
            0
        } else {
            stage_context_for(last - 1, deployed)
        };
        let room =
            self.free_bytes as f64 - RAMP_MEMORY_RESERVE_BYTES as f64 - self.fixed_bytes;
        if room <= 0.0 || self.per_token_bytes <= 0.0 {
            return 0;
        }
        let weighted = context as f64 + RAMP_MEMORY_MARGIN * (context - previous) as f64;
        (room / (self.per_token_bytes * weighted)).floor() as usize
    }

    /// The batch ramp this card can run, capped stage by stage at measured capacity.
    ///
    /// `base_batch` must already fit flat at the deployed context — [`resolve_ramp`] clamps
    /// it to [`Self::frontier_batch`] before calling — so `x1` is feasible at every stage and
    /// the repair loop below always terminates on a feasible ramp.
    fn derive_batch_ramp(&self, base_batch: usize) -> [usize; RAMP_STAGES] {
        let mut ramp = [1usize; RAMP_STAGES];
        let mut previous = 0.0;
        for stage in 0..RAMP_STAGES {
            let context = stage_context(stage);
            let affordable = self.max_batch(context, previous) / base_batch.max(1);
            ramp[stage] = affordable.clamp(1, BATCH_RAMP[stage]);
            previous = self.step_bytes(base_batch * ramp[stage], context);
        }
        // A batch that SHRINKS mid-run is not a ramp: the plateau bump would step DOWN, which
        // the reference recipe never does, and the sampler's per-stage anchor lists are sized
        // for a batch that only grows. Take the suffix minimum.
        for stage in (0..RAMP_STAGES - 1).rev() {
            ramp[stage] = ramp[stage].min(ramp[stage + 1]);
        }
        // Then re-verify, because that suffix pass can invalidate what it did not touch:
        // lowering an early stage leaves the allocator holding a SMALLER pool at the next
        // step-up, so the transient margin charged on that step-up grows. Each repair
        // strictly lowers the multiplier sum and `x1` everywhere is feasible, so this runs at
        // most `sum(BATCH_RAMP) - RAMP_STAGES` times.
        loop {
            let mut previous = 0.0;
            let mut repaired = false;
            for stage in 0..RAMP_STAGES {
                let context = stage_context(stage);
                let batch = base_batch * ramp[stage];
                if ramp[stage] > 1
                    && self.required_bytes(batch, context, previous) > self.free_bytes as f64
                {
                    let lowered = ramp[stage] - 1;
                    for slot in ramp.iter_mut().take(stage + 1) {
                        *slot = (*slot).min(lowered);
                    }
                    repaired = true;
                    break;
                }
                previous = self.step_bytes(batch, context);
            }
            if !repaired {
                return ramp;
            }
        }
    }

    fn gib(bytes: f64) -> f64 {
        bytes / (1u64 << 30) as f64
    }

    /// Headroom left over at the ramp's final stage: free VRAM minus everything the step
    /// needs, margin and shared-card reserve included.
    fn headroom_bytes(&self, batch_ramp: &[usize; RAMP_STAGES], base_batch: usize) -> f64 {
        let last = RAMP_STAGES - 1;
        let previous = if last == 0 {
            0.0
        } else {
            self.step_bytes(base_batch * batch_ramp[last - 1], stage_context(last - 1))
        };
        self.free_bytes as f64
            - self.required_bytes(base_batch * batch_ramp[last], stage_context(last), previous)
    }
}

/// [`CAPACITY_PROBE_STEPS`] forward-and-backward passes at ONE shape, then the device's
/// `used` bytes with the pool warm and the last graph already dropped — the steady state a
/// training step sits in, not the first pass's transient.
///
/// The pool is released first so the reading prices THIS shape rather than this shape plus a
/// cached predecessor. `None` off CUDA or without NVML; the passes still run, so a CPU test
/// can assert the invariants that make the probe non-destructive.
///
/// Nothing persistent changes here. There is no optimizer step, so the weights reach step 0
/// exactly as initialized and the moment buffers exactly as zeroed; the gradients the
/// backwards allocate are deliberately LEFT resident, because training needs them,
/// [`Trainer::optimizer_step`] zeroes them before its own first backward, and they belong in
/// the baseline rather than in the per-step model.
fn probe_shape_used_bytes(
    modules: &BarModules,
    supports: &BarSupports,
    growth_support: &GrowthSupport,
    sample: &BarBatch,
    args: &PretrainArgs,
    context: i64,
    device: Device,
) -> Option<u64> {
    crate::torch::cuda::empty_cache();
    for _ in 0..CAPACITY_PROBE_STEPS {
        let graph = autocast(device.is_cuda(), || {
            forward_losses(
                modules,
                supports,
                growth_support,
                &sample.dof,
                &sample.time_ids,
                context,
                args.dyn_horizon as i64,
                args.lambda_dyn,
                args.lambda_kl,
                args.lambda_growth,
                args.scoring,
                device,
            )
        });
        graph.loss.backward();
    }
    device_used_bytes(device)
}

/// Measure the affine footprint model on the real training graph, at the deployed context.
///
/// Runs [`CAPACITY_PROBE_STEPS`] forward-and-backward passes at each of
/// [`CAPACITY_PROBE_BATCHES`], reading device-wide `used` with the allocator pool released
/// between shapes so each reading prices ONE shape rather than that shape plus a cached
/// predecessor. No optimizer step: the weights must reach step 0 exactly as initialized and
/// the moment buffers must reach it exactly as zeroed, so the probe cannot pay for the
/// optimizer state and [`Muon::steady_state_bytes`] is added to the fixed term instead. The
/// gradients the backward passes allocate are deliberately LEFT resident — training needs
/// them, `optimizer_step` zeroes them before its first backward, and they belong in the
/// baseline rather than in the per-step model.
///
/// Returns `None` off CUDA, without NVML, or when the readings do not admit a usable slope,
/// in which case the ramp keeps the declared [`BATCH_RAMP`] and the runtime hold is the only
/// protection — exactly the behaviour that preceded this function, under a banner that says
/// so.
fn probe_capacity(
    modules: &BarModules,
    supports: &BarSupports,
    growth_support: &GrowthSupport,
    sampler: &BarSampler,
    optimizer: &Muon,
    args: &PretrainArgs,
    device: Device,
) -> Option<CapacityModel> {
    if !device.is_cuda() {
        return None;
    }
    let context = stage_context(RAMP_STAGES - 1);
    let horizon = args.dyn_horizon as i64;
    if horizon >= context {
        return None;
    }

    let mut points = [(0usize, 0u64); CAPACITY_PROBE_BATCHES.len()];
    for (slot, &batch) in points.iter_mut().zip(CAPACITY_PROBE_BATCHES.iter()) {
        if sampler.batches_per_epoch(batch) == 0 {
            return None;
        }
        let refs = sampler.batch_refs(0, 0, batch);
        let sample = sampler.batch_of(&refs, device);
        *slot = (
            batch,
            probe_shape_used_bytes(
                modules,
                supports,
                growth_support,
                &sample,
                args,
                context,
                device,
            )?,
        );
    }

    // Released pool, so `free` counts what the driver would actually hand out and `baseline`
    // excludes the probe's own activations.
    crate::torch::cuda::empty_cache();
    let (free_bytes, baseline_bytes) = device_memory(device)?;

    let (small_batch, small_used) = points[0];
    let (large_batch, large_used) = points[points.len() - 1];
    let token_span = (large_batch as f64 - small_batch as f64) * context as f64;
    let byte_span = large_used as f64 - small_used as f64;
    if token_span <= 0.0 || byte_span <= 0.0 {
        println!(
            "WARNING: the capacity probe read {} B at batch {small_batch} and {} B at batch \
             {large_batch}; a non-increasing pair admits no per-token slope — most likely \
             another tenant released memory between the two readings. Falling back to the \
             DECLARED ramp {BATCH_RAMP:?} under the runtime memory hold.",
            small_used, large_used
        );
        return None;
    }
    let per_token_bytes = byte_span / token_span;
    // The batch-independent remainder of the smaller reading, plus the optimizer state that is
    // not resident YET. Clamped at zero: a negative intercept means co-tenant noise dominated
    // the fit, and pretending the fixed cost is negative would inflate the ramp.
    //
    // `steady_state_bytes - state_bytes` and not the steady state alone. NorMuon allocates its
    // momentum and second-moment buffers eagerly in `Muon::new_named`, so they are ALREADY
    // inside `baseline_bytes`; only the AdamW moments are lazy, and adding the whole steady
    // state here would charge the 2D branch twice and shrink every derived stage for nothing.
    let pending_optimizer_bytes = optimizer
        .steady_state_bytes()
        .saturating_sub(optimizer.state_bytes()) as f64;
    let fixed_bytes = (small_used as f64
        - baseline_bytes as f64
        - per_token_bytes * small_batch as f64 * context as f64)
        .max(0.0)
        + pending_optimizer_bytes;

    println!(
        "capacity probe: {} B/bar-token at context {context}, from {:.2} GiB used at batch \
         {small_batch} and {:.2} GiB at batch {large_batch} ({} bar-tokens apart); fixed \
         per-step cost {:.2} GiB (incl. {:.2} GiB of AdamW moments the probe did not step into \
         existence, on top of {:.2} GiB of NorMuon state already resident); {:.2} GiB free and \
         {:.2} GiB in use with the allocator pool released",
        per_token_bytes.round(),
        CapacityModel::gib(small_used as f64),
        CapacityModel::gib(large_used as f64),
        token_span,
        CapacityModel::gib(fixed_bytes),
        CapacityModel::gib(pending_optimizer_bytes),
        CapacityModel::gib(optimizer.state_bytes() as f64),
        CapacityModel::gib(free_bytes as f64),
        CapacityModel::gib(baseline_bytes as f64),
    );

    Some(CapacityModel {
        per_token_bytes,
        fixed_bytes,
        free_bytes,
        baseline_bytes,
    })
}

/// The base batch, the ramp the run will actually execute, and the notice explaining any
/// clamp.
///
/// Without a capacity model — off CUDA, or without NVML — the declared [`BATCH_RAMP`] stands
/// and nothing is clamped; the banner says the capacity is unmeasured and the runtime hold is
/// the only protection.
///
/// With one, `requested` is clamped to what the deployed context can hold, and the returned
/// notice states the ARITHMETIC. It is returned rather than printed so a test can assert the
/// numbers are in it: a clamp message that merely says "does not fit" is the kind of thing
/// that gets believed once and then argued with, and this one has to survive being read four
/// hours before an OOM would otherwise have happened. `Err` is reserved for the case where not
/// even a single window fits, because there is nothing to clamp to — and for `exact`, where
/// the caller has declared that a clamp is not an acceptable outcome. See
/// [`PretrainArgs::exact_batch`] for the measured confound that motivates that mode.
#[derive(Clone, Debug)]
struct RampPlan {
    base_batch: usize,
    batch_ramp: [usize; RAMP_STAGES],
    notice: Option<String>,
}

fn resolve_ramp(
    capacity: Option<&CapacityModel>,
    requested: usize,
    exact: bool,
) -> Result<RampPlan> {
    let Some(capacity) = capacity else {
        return Ok(RampPlan {
            base_batch: requested,
            batch_ramp: BATCH_RAMP,
            notice: None,
        });
    };
    let deployed = stage_context(RAMP_STAGES - 1);
    let previous = stage_context(RAMP_STAGES - 2);
    let ceiling = capacity.frontier_batch(deployed);
    ensure!(
        ceiling > 0,
        "not one {deployed}-bar window fits on this device: a single window costs {:.2} GiB of \
         activations at the measured {:.0} B/bar-token plus {:.2} GiB of fixed per-step cost, \
         which with the {:.0}% transient margin and the {:.2} GiB shared-card reserve needs \
         {:.2} GiB against {:.2} GiB free. Free the card or reduce the deployed context — \
         nothing --batch-size can choose will fit.",
        CapacityModel::gib(capacity.per_token_bytes * deployed as f64),
        capacity.per_token_bytes,
        CapacityModel::gib(capacity.fixed_bytes),
        RAMP_MEMORY_MARGIN * 100.0,
        CapacityModel::gib(RAMP_MEMORY_RESERVE_BYTES as f64),
        CapacityModel::gib(capacity.required_bytes(1, deployed, 0.0)),
        CapacityModel::gib(capacity.free_bytes as f64),
    );

    let base_batch = requested.min(ceiling);
    ensure!(
        base_batch == requested || !exact,
        "--exact-batch was set and the measured capacity affords only {base_batch} of the \
         {requested} windows requested at the deployed {deployed}-bar context, a shortfall of \
         {}. Refusing to start rather than clamping, because a clamp makes this run \
         incomparable to its own control: the bar budget is fixed by --epochs, so a smaller \
         base batch means MORE steps, a different gradient-noise level and a different-length \
         lr and momentum schedule. Arithmetic: {:.0} B/bar-token measured x {requested} \
         windows x {deployed} bars = {:.2} GiB of activations, plus {:.2} GiB fixed, plus the \
         {:.0}% transient margin on the step up from {previous} bars and the {:.2} GiB \
         shared-card reserve = {:.2} GiB required, against {:.2} GiB free. Free the card, or \
         request {base_batch} on BOTH arms, or drop --exact-batch if this is a production run \
         rather than an experiment.",
        requested - base_batch,
        capacity.per_token_bytes,
        CapacityModel::gib(capacity.per_token_bytes * requested as f64 * deployed as f64),
        CapacityModel::gib(capacity.fixed_bytes),
        RAMP_MEMORY_MARGIN * 100.0,
        CapacityModel::gib(RAMP_MEMORY_RESERVE_BYTES as f64),
        CapacityModel::gib(capacity.required_bytes(
            requested,
            deployed,
            capacity.step_bytes(requested, previous),
        )),
        CapacityModel::gib(capacity.free_bytes as f64),
    );
    let notice = (base_batch < requested).then(|| {
        format!(
            "WARNING: --batch-size {requested} does not fit at the deployed {deployed}-bar \
             context and is CLAMPED to {base_batch} before the first step. Arithmetic: {:.0} \
             B/bar-token measured x {requested} windows x {deployed} bars = {:.2} GiB of \
             activations, plus {:.2} GiB fixed, plus the {:.0}% transient margin on the step up \
             from {previous} bars and the {:.2} GiB shared-card reserve = {:.2} GiB required, \
             against {:.2} GiB free. {base_batch} windows need {:.2} GiB and fit. Said now \
             rather than four hours in at a stage transition, which is where the two runs that \
             died of CUDA OOM found out.",
            capacity.per_token_bytes,
            CapacityModel::gib(capacity.per_token_bytes * requested as f64 * deployed as f64),
            CapacityModel::gib(capacity.fixed_bytes),
            RAMP_MEMORY_MARGIN * 100.0,
            CapacityModel::gib(RAMP_MEMORY_RESERVE_BYTES as f64),
            CapacityModel::gib(capacity.required_bytes(
                requested,
                deployed,
                capacity.step_bytes(requested, previous),
            )),
            CapacityModel::gib(capacity.free_bytes as f64),
            CapacityModel::gib(capacity.required_bytes(
                base_batch,
                deployed,
                capacity.step_bytes(base_batch, previous),
            )),
        )
    });
    Ok(RampPlan {
        base_batch,
        batch_ramp: capacity.derive_batch_ramp(base_batch),
        notice,
    })
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn pretrain(args: PretrainArgs) -> Result<()> {
    // Before any tensor work in the process, which is the only time torch accepts it.
    configure_threads();
    build_trainer(args, RUNS_PATH, Device::cuda_if_available())?.run_training()
}

/// Everything `pretrain` does before the first optimizer step, split out so a test can drive
/// one validation of a real trainer against a synthetic corpus instead of only unit-testing
/// the pieces around it. `runs_root` is a parameter for exactly that reason: a test must not
/// write into the campaign's run directory.
///
/// `device` is a parameter for the same reason, and it is NOT `Device::cuda_if_available()`
/// resolved inside. A unit test that silently picks up whatever card happens to be visible is
/// wrong twice over. It allocates on a GPU another tenant owns — ours is normally holding a
/// live RL job — and it takes a code path the harness cannot support: [`configure_cuda`] pins
/// the autocast dtype THREAD-LOCALLY in this toolchain, so under `--test-threads=N` only the
/// thread that won the `Once` is configured and every other one aborts in
/// `assert_bf16_autocast`. Passing the device makes both facts a decision at the call site
/// instead of an accident of the machine the suite runs on.
fn build_trainer(mut args: PretrainArgs, runs_root: &str, device: Device) -> Result<Trainer> {
    validate_args(&args)?;
    if device.is_cuda() {
        configure_cuda();
    }

    tch::manual_seed(args.seed as i64);
    if device.is_cuda() {
        tch::Cuda::manual_seed_all(args.seed);
    }

    let run = RunDir::create_fresh(runs_root, args.run.as_deref())
        .context("failed to create pretrain run dir")?;

    let corpus = load_corpus(&args.corpus_flags())?;
    // Taken AFTER any symbol restriction, because the symbol set decides which bars a split
    // contains. The corpus also grows under running jobs and the split instants are
    // percentiles of it, so the identity of the data is a first-class output of the run.
    let corpus_fingerprint = corpus.identity_fingerprint();

    // RECORDED, not inferred, and recorded HERE — before the first optimizer step and before
    // any checkpoint the record has to explain can exist.
    //
    // `bardist_v3_rfirst_1ep/meta.json` held `{"commit": ...}` and nothing else, which made
    // "this run split train|val at 2025-10-07T12:10:00Z" recoverable only by checking out that
    // commit and reading this function's DEFAULT. Every economic number taken off that run
    // rests on the instants, so the instants belong in the run's own record. `corpus`, not
    // `args`, is asked for them: under `--derive-split-bounds` the args carry no instants at
    // all and the RESOLVED pair is the only thing a later reader can validate against.
    let (b0, b1) = corpus.split_bounds();
    run.record_provenance(RunProvenance {
        split_bounds_ms: [b0, b1],
        split_bounds_pinned: !args.derive_split_bounds,
        resolution_secs: args.resolution_secs,
        corpus_fingerprint: corpus_fingerprint.clone(),
        min_bars: args.min_bars,
        min_dollar_volume: args.min_dollar_volume,
        data_dir: args.data_dir.clone(),
        diagnostic_context_bars: args.diagnostic_context,
        deployed_context_bars: stage_context(RAMP_STAGES - 1),
        eval_window_seed: EVAL_WINDOW_SEED,
        train_seed: args.seed,
    })
    .context("failed recording the run's provenance")?;

    let train_bars = corpus.split_bars(Split::Train) as u64;
    ensure!(
        train_bars > 0,
        "training split is empty; check --data-dir, --resolution-secs and --min-bars"
    );

    let (supports, supports_frozen) = fit_supports(&corpus, &args, &corpus_fingerprint)?;
    let supports_dev = supports.to_device(device);
    // Before the capacity probe, because the probe measures the REAL training graph and the
    // growth term is part of it. `new` is also where the log-argument bound is asserted
    // against the ACTUAL fitted support, so a corpus whose `r` support is too wide for the
    // leverage cap fails here rather than producing a NaN objective at step 400.
    let growth_deployment = GrowthSupport::new(&supports_dev, device)
        .context("the deployment r support cannot carry the expected-log-growth term")?;

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

    // The ramp is derived from what the card MEASURABLY holds, before anything is announced
    // or any step is taken. Everything downstream — the step count, the learning-rate
    // plateau bumps, the banner, the eval batch — reads the derived plan, so the runtime
    // memory hold is left as a safety net for contention instead of being the thing that
    // silently rewrites the schedule on every run.
    let capacity = probe_capacity(
        &modules,
        &supports_dev,
        &growth_deployment,
        &train_samplers[RAMP_STAGES - 1],
        &optimizer,
        &args,
        device,
    );
    // The probe consumes RNG through the trunk's forward passes. Re-seed so the training
    // stream is byte-identical to what it would have been without a probe: a capacity
    // measurement must not move the run it is measuring.
    tch::manual_seed(args.seed as i64);
    if device.is_cuda() {
        tch::Cuda::manual_seed_all(args.seed);
    }
    let requested_batch = args.batch_size;
    let RampPlan {
        base_batch,
        batch_ramp,
        notice,
    } = resolve_ramp(capacity.as_ref(), requested_batch, args.exact_batch)?;
    if let Some(notice) = notice {
        println!("{notice}");
    }
    // Every consumer of `args.batch_size` — the eval passes, the test battery, the recorded
    // provenance — must see the batch the run will actually use, not the one it asked for.
    args.batch_size = base_batch;

    // The auxiliary resolutions, AFTER the ramp is resolved because their pass partitions
    // apportion by `batch[s] * context[s]` and so need the batch the run will actually run. Each
    // gets its own supports through the same provenance gate the deployment fit uses.
    let aux = AuxiliaryStream::open(
        &corpus,
        &AuxiliaryConfig {
            resolutions: &args.auxiliary_resolutions,
            base_batch,
            batch_ramp: &batch_ramp,
            seed: args.seed,
            scoring: args.scoring,
            device,
        },
        |aux_corpus, fingerprint| {
            let path = aux_corpus.supports_path();
            let (supports, _frozen) =
                fit_supports_at(aux_corpus, &path, SupportsFit::of(&args), fingerprint)?;
            Ok(supports)
        },
    )?;
    // Every resolution the trunk will see, keyed so the row router cannot score a daily bar
    // against 5-minute bins. Built here rather than at the deployment fit because it is only
    // complete once the auxiliaries are open.
    let support_set_dev = BarSupportSet::new(
        std::iter::once((args.resolution_secs, supports.to_device(device)))
            .chain(
                aux.iter()
                    .map(|stream| (stream.res_secs(), stream.supports().to_device(device))),
            )
            .collect(),
    )
    .context("failed building the resolution-keyed support set")?;
    // One registered base carries every resolution's audit, so the corpus report states what was
    // actually loaded instead of the deployment resolution alone.
    let corpus_audits: Vec<CorpusAnomalies> = std::iter::once(corpus.scan_anomalies())
        .chain(aux.iter().map(AuxiliaryStream::scan_anomalies))
        .collect();
    for audit in &corpus_audits {
        println!("[corpus] {}", audit.summary());
    }
    let aux_heldout: Vec<PinnedSet> = aux
        .iter()
        .map(|stream| {
            PinnedSet::pinned(
                stream.corpus(),
                Split::Val,
                AUXILIARY_HELDOUT_CONTEXT,
                args.validation_windows,
            )
            .with_context(|| {
                format!(
                    "the {}s auxiliary corpus has no {AUXILIARY_HELDOUT_CONTEXT}-bar val window",
                    stream.res_secs()
                )
            })
        })
        .collect::<Result<_>>()?;
    let aux_report = AuxiliaryReport::new(&aux);
    // The corpus partition, and then the schedule FROM the partition. This is the ordering
    // the whole coverage invariant rests on: the stages' shares are their bar-token shares,
    // so the plan needs the ramp that will execute, and the step count is then whatever it
    // takes to issue every window the plan assigned — not a bar-token target divided by an
    // average step size, which is what let a run label 71% of a pass "one epoch".
    let pass = PassPlan::new(
        &corpus,
        Split::Train,
        &stage_contexts(),
        &ramp_token_weights(&batch_ramp),
        args.seed,
    )
    .context("failed partitioning the training split across the ramp contexts")?;
    print_pass_plan(&pass, base_batch, &batch_ramp);
    let stage_steps = Schedule::steps_for_pass(pass.windows_per_stage(), base_batch, &batch_ramp);
    let total_steps = match args.steps {
        Some(steps) => {
            ensure!(steps > 0, "--steps must be positive");
            println!(
                "[pretrain] WARNING --steps {steps} overrides the {} steps one pass needs, so \
                 this run's epochs are DECOUPLED from the corpus: a short override leaves the \
                 final pass incomplete and a long one repeats passes. The coverage invariant is \
                 still checked at every completed pass.",
                stage_steps.iter().sum::<usize>() * args.epochs.max(1)
            );
            steps
        }
        None => stage_steps.iter().sum::<usize>() * args.epochs,
    };
    let schedule = Schedule::new(
        stage_steps,
        total_steps,
        base_batch,
        batch_ramp,
        args.lr_plateau_fraction,
    );
    // Epoch 0's geometry, so the first step draws from a partition rather than from a sampler
    // that would happily hand out the same anchors three times.
    let pass_layout = pass.layout(0);

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
    // The density rule adds `E[ln width]` to every observation's score. It is a constant of
    // the supports, carries no gradient, and would otherwise set the denominator of the
    // loss-term shares — making the 25% auxiliary-domination threshold mean a different
    // thing under each `--scoring`.
    let share_scale_offset = if scoring.is_density() {
        supports.log_measure_bar()
    } else {
        0.0
    };
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

    // One per bin geometry, deployment first — the deployment one was already built and
    // handed to the capacity probe, so it is moved in rather than rebuilt.
    let mut growth_supports = Vec::with_capacity(1 + aux.len());
    growth_supports.push(growth_deployment);
    for stream in &aux {
        growth_supports.push(
            GrowthSupport::new(stream.supports_dev(), device).with_context(|| {
                format!(
                    "the {}s r support cannot carry the expected-log-growth term",
                    stream.res_secs()
                )
            })?,
        );
    }
    // The prefix-free read is architectural, but it is checked here, on the real head and the
    // real device, before a single step has trained on a mean the term could not measure: the
    // failure mode is silent, and the check also catches a TF32 matmul.
    growth::verify_traded_law(&modules.head, &supports_dev, device)
        .context("the expected-log-growth term's traded law does not check out")?;
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
        capacity.as_ref(),
        requested_batch,
    );

    let mut reporter = PretrainReporter::new(&run.gens, marginal_nll_dof);
    reporter.set_held_out_baselines(baselines);
    // Every loaded resolution's audit on the one registered base. Written here, where the
    // generation directory first exists, so a run that loads a daily corpus PROVES it in an
    // artifact instead of in a log line.
    CorpusAnomalies::write_report_of(&corpus_audits, &run.gens)
        .context("failed writing the corpus anomaly report")?;
    // Taken before the trainer owns the set, and re-checked at every boundary.
    let snapshot_window_fingerprint = pinned_fingerprint(&eval.snapshot);
    Ok(Trainer {
        args,
        device,
        schedule,
        run,
        supports,
        supports_dev,
        support_set_dev,
        growth_supports,
        vs,
        modules,
        optimizer,
        train_samplers,
        eval,
        reporter,
        marginal_nll_bar,
        marginal_nll_dof,
        marginal_nll_dof_val,
        baselines,
        corpus_fingerprint,
        supports_frozen,
        symbol_count: corpus.symbols().len(),
        pass_ledger: PassLedger::new(&pass_layout),
        pass_layout,
        pass,
        stage_cursor: [0; RAMP_STAGES],
        completed_passes: 0,
        audit: None,
        census: PassCensus::default(),
        bars_seen: 0,
        epoch: 0,
        best_val_nll_bar: f64::INFINITY,
        best_val_nll_bar_conditional: f64::INFINITY,
        best_scores: None,
        best_selection_edge_bps: f64::NEG_INFINITY,
        best_selection_edge_windows: None,
        best_selection_nll: f64::NAN,
        promoted_step: 0,
        nll_rule_scores: None,
        nll_rule_step: 0,
        nll_rule_promotions: 0,
        nll_rule_edge_bps: f64::NAN,
        promotions: 0,
        train_nll_sum: 0.0,
        train_nll_dof_sum: [0.0; BAR_DOF],
        train_steps: 0,
        aux_share_streak: 0,
        aux,
        aux_heldout,
        aux_report,
        aux_steps: 0,
        aux_bars_seen: 0,
        share_scale_offset,
        vram_baseline_bytes: None,
        // Seeded from the startup probe so the FIRST stage transition is gated on a measured
        // figure. Before this the runtime probe only ran four steps into stage 0, so the
        // transition into stage 1 — the one two runs died at — was taken on `None`, i.e. with
        // the guard disabled.
        activation_bytes_per_token: capacity.as_ref().map(|c| c.per_token_bytes),
        capacity,
        derived_batch_ramp: batch_ramp,
        requested_batch,
        stage_step: 0,
        reached_context: 0,
        selection_context: 0,
        best_by_context: BTreeMap::new(),
        diagnostic_best: None,
        epoch_started: Instant::now(),
        epoch_start_bars: 0,
        epoch_dyn_identity_sum: 0.0,
        epoch_dyn_identity_steps: 0,
        snapshot_window_fingerprint,
    })
}

/// Arguments of the standalone candle-rollout entry point.
#[derive(Clone, Debug)]
pub struct CandleArgs {
    /// Checkpoint to picture. Its `.metadata.json` and `.supports.<res>.json` siblings
    /// are resolved from this path, so a copy must keep the same file stem.
    pub weights: String,
    /// Directory the `.report.bin` pictures are written into.
    pub output: String,
    /// Pinned validation windows to picture. The count is part of the pin — a symbol's
    /// quota and the spacing of its picks both scale with it — so this must equal the
    /// run's `--snapshot-windows` to depict the run's own windows, not merely be smaller.
    pub windows: usize,
    /// Ancestral samples per window. The rollout KV cache is linear in this and is the
    /// whole memory cost of the command.
    pub samples: usize,
    /// Conditioning context, of which the last [`SNAPSHOT_HORIZON`] bars are held out as
    /// the realized continuation. Must match the run's `--diagnostic-context` for the
    /// pictures to depict the run's own snapshot windows.
    pub context: i64,
    /// Optimizer step the checkpoint reached. It only names the output files: the
    /// checkpoint metadata records lineage and provenance but not a step count, so the
    /// operator supplies it when a directory has to hold several checkpoints.
    pub step: usize,
    pub corpus: CorpusFlags,
}

/// Candle pictures of one EXISTING checkpoint against the realized bars, on the pinned
/// validation windows.
///
/// This is [`Trainer::write_snapshot`] without the training loop, and deliberately shares
/// every piece of it that decides WHAT is depicted: [`PinnedSet::pinned`] draws the windows
/// under [`EVAL_WINDOW_SEED`], [`pinned_snapshot_window`] splits off the realized
/// continuation, [`rollout_pinned_windows`] samples one window at a time, and
/// `pretrain_reports::write_candle_windows` chains and writes them. Only the trigger
/// differs: a run cannot picture anything before its first promotion, and a checkpoint
/// mid-ramp is exactly when one wants to look.
pub fn pretrain_candles(args: CandleArgs) -> Result<()> {
    ensure!(args.windows > 0, "--windows must be positive");
    ensure!(args.samples > 0, "--samples must be positive");
    ensure!(
        args.context > SNAPSHOT_HORIZON,
        "--context must exceed the {SNAPSHOT_HORIZON}-bar snapshot horizon"
    );
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
    // The real load path: it hash-checks the checkpoint and every supports sidecar
    // against the metadata, so a torn copy or a mismatched support set fails here
    // rather than producing a plausible-looking picture of the wrong output space.
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
            "checkpoint pinned its bench with eval_window_seed {:#x} but this build uses \
             {EVAL_WINDOW_SEED:#x}; the pictures would depict different data than the run's",
            trained.eval_window_seed
        );
    }

    let set = PinnedSet::pinned(&corpus, Split::Val, args.context, args.windows)?;
    let window = pinned_snapshot_window(&set, device);
    let rollout = tch::no_grad(|| rollout_pinned_windows(&world, &window, args.samples));

    let output = Path::new(&args.output);
    let drawn = super::pretrain_reports::write_candle_windows(
        output,
        args.step,
        // Not an epoch artifact: this is whatever checkpoint the operator named.
        None,
        &rollout,
        &window.future_dof,
    )?;

    println!(
        "candle rollout: {} pinned val windows x {} ancestral samples over {SNAPSHOT_HORIZON} \
         bars at context {}, from {} (lineage {}, step {})",
        drawn.len(),
        args.samples,
        args.context,
        weights.display(),
        world.lineage_sha256(),
        args.step,
    );
    println!("reports written to {}", output.display());
    print_candle_windows(&corpus, &set, &drawn);
    Ok(())
}

/// Arguments of the standalone trading-bench entry point.
#[derive(Clone, Debug)]
pub struct TradeArgs {
    /// Checkpoint to bench. Its `.metadata.json` and `.supports.<res>.json` siblings are
    /// resolved from this path, so a copy must keep the same file stem.
    pub weights: String,
    /// Directory the `pretrain_trade_*.report.bin` charts are written into.
    pub output: String,
    /// Held-out split to trade. `val` is the pinned diagnostic set a run reports on every
    /// validation; `test` is the split that is scored once.
    pub split: Split,
    /// Pinned windows to draw. The bench trades the first
    /// [`trade_bench::TRADE_WINDOWS`] of them, and the count is part of the pin, so this
    /// must equal the run's `--validation-windows` to trade the run's own windows.
    pub windows: usize,
    /// Conditioning context. Must match the context the checkpoint was selected at, or the
    /// bench measures positional extrapolation rather than forecasting.
    pub context: i64,
    /// Evaluation batch, in windows.
    pub batch_size: usize,
    pub corpus: CorpusFlags,
}

/// The trading bench of one EXISTING checkpoint, on pinned held-out windows.
///
/// This is the validation-path bench without the training loop, and it shares every piece
/// that decides WHAT is measured: [`PinnedSet::pinned`] draws the windows under
/// [`EVAL_WINDOW_SEED`], [`evaluate`] produces the positions through [`TradeSetup`], and
/// [`pinned_blocks`] blocks the interval. The numbers are therefore the same numbers the
/// run would report for that artifact, not a second implementation of them.
pub fn pretrain_trade(args: TradeArgs) -> Result<()> {
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
            "checkpoint pinned its bench with eval_window_seed {:#x} but this build uses \
             {EVAL_WINDOW_SEED:#x}; the bench would trade different data than the run's",
            trained.eval_window_seed
        );
    }

    // The scoring rule is part of what `nll_bar` MEANS, and the bench's own numbers do not
    // depend on it at all - the positions come from the head's probabilities - but the NLL
    // printed beside them does, so it is read off the artifact rather than re-declared.
    let scoring: BarScoring = world
        .metadata()
        .training
        .as_ref()
        .map(|trained| trained.scoring.parse())
        .transpose()
        .map_err(|reason| anyhow!("the checkpoint records a scoring rule this build cannot parse: {reason}"))?
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
        trade_bench::TRADE_WINDOWS,
    )?;
    let mut blocks = pinned_blocks(&set);
    blocks.truncate(stats.trade_paths.len());
    let trade = trade_bench::bench(
        &stats.trade_paths.windows,
        &blocks,
        &stats.trade_paths.tail,
        BenchConfig::new(
            trade_bench::DEFAULT_COST_BPS,
            trade_bench::LEVERAGE_CAP,
            trade_bench::marginal_position(
                world.deployment_supports(),
                trade_bench::FREE_LEVERAGE,
            ),
        ),
    );

    println!(
        "trade bench of {} (lineage {}) on the pinned {:?} split at context {}: nll {:.4} \
         nats/bar over {} windows",
        weights.display(),
        world.lineage_sha256(),
        args.split,
        set.context,
        stats.nll_bar,
        set.windows.len(),
    );
    for line in trade.report_lines() {
        println!("{line}");
    }
    let output = Path::new(&args.output);
    super::pretrain_reports::write_trade_bench(
        output,
        &format!("{:?} split, {}", args.split, weights.display()),
        &trade,
    )?;
    println!("reports written to {}", output.display());
    Ok(())
}

/// Arguments of the mean-calibration experiment.
#[derive(Clone, Debug)]
pub struct CalibrationArgs {
    /// Checkpoints to measure, as `path@step`. The step is the x-axis of the reported trend
    /// and is NOT recoverable from the artifact: the metadata sidecar records the context and
    /// the lineage but not the optimizer step, so it is stated rather than guessed.
    pub checkpoints: Vec<String>,
    /// Directory the `pretrain_mean_calibration` and `pretrain_shrunk_policy` charts are
    /// written into.
    pub output: String,
    pub split: Split,
    /// Pinned windows to DRAW. The bench trades the first `trade_windows` of them and the
    /// calibration is fitted on windows drawn from the rest, so this must be large enough to
    /// leave a block-disjoint remainder — and it must equal the run's `--validation-windows` for
    /// the traded prefix to be the run's own windows.
    ///
    /// NOT a free dial. [`BarSampler::pinned_windows`] allocates a per-symbol quota from `count`
    /// and skips symbols whose quota rounds to zero, so raising this MOVES the traded prefix:
    /// measured on the live corpus, going from 4096 to 8192 changes 3,794 of Val's 4,729
    /// symbols' quotas and lets 675 previously-skipped symbols insert themselves ahead of names
    /// that were in the old prefix. Widen `fit_windows` out of the existing remainder instead.
    pub windows: usize,
    /// Windows of the block-disjoint remainder to fit the recalibration on. Truncated to what
    /// the remainder actually holds, so asking for more than exists is safe.
    pub fit_windows: usize,
    /// Windows of the drawn prefix to TRADE.
    ///
    /// Defaults to [`trade_bench::TRADE_WINDOWS`] and every published number was measured at
    /// that value, which is why it is a parameter rather than the constant read in place: a
    /// fresh panel with nothing to stay comparable to should spend the windows it has, and a
    /// panel that IS being compared must not move. `Split::Val` keeps 256 so this batch stays
    /// comparable to itself; a one-shot `Split::Test` read can ask for thousands, because the
    /// interval is set by the `(symbol, calendar month)` block count of the traded slice and
    /// Test holds 43,466 near-disjoint windows at context 896 against the 256 a default draw
    /// would trade.
    pub trade_windows: usize,
    /// Conditioning context. Must match the context the reported bench reads were taken at.
    pub context: i64,
    pub batch_size: usize,
    pub corpus: CorpusFlags,
    /// Symbols the TRADED prefix is narrowed to, empty for the whole prefix.
    ///
    /// Exists so an edge can be measured on exactly the names a cost was priced on. The fit
    /// slice is deliberately NOT restricted: narrowing it too would shrink the slope's own
    /// sample without making the comparison any more matched, since the slope is a property of
    /// the forecaster rather than of the population it is spent on. Whether that is the right
    /// call is itself measurable — see [`trade_bench::MzFit::block_dispersion_measured`], which
    /// says whether the slope varies across blocks by more than its own noise.
    pub restrict_symbols: Vec<String>,
    /// Draw the windows, block them, write the window manifest and the held-out power census,
    /// then STOP — before any checkpoint is opened and before any economic number exists.
    ///
    /// This is not a convenience. `Split::Test` is scored ONCE for the whole campaign, and the
    /// only way to establish that the command addresses the intended data, that the block
    /// partition is disjoint, and that the population has the power to resolve the effect being
    /// looked for is to perform every step that decides WHAT is measured and none of the steps
    /// that measure it. A rehearsal that scored anything would consume the draw it was
    /// rehearsing.
    pub dry_run: bool,
}

/// Measured 95% block-bootstrap half-width on net growth per traded bar, over the 256-block
/// traded slice of `Split::Val` at context [`BAR_CONTEXT_RAMP_START`].
///
/// A CAMPAIGN REFERENCE, not a prediction: it is the width this project actually observed on
/// `bardist_v3_rfirst_1ep`'s primary checkpoint, and it exists so a population can be judged
/// BEFORE it is scored rather than after. Any other split's expected width is this one scaled
/// by `sqrt(reference blocks / blocks)`, which rests on ONE assumption stated where it is used:
/// the bootstrap resamples `(symbol, calendar month)` BLOCKS, so `s_block / sqrt(B)` is the
/// whole interval, `B` is counted directly, and only `s_block` — the cross-block dispersion of
/// per-block net, a property of the regime rather than of the sample size — is imported. A
/// split whose regime is materially more dispersed than Oct 2025..Mar 2026 will be wider than
/// this predicts, which is why the pass that scores it must report its own measured `s_block`
/// against this constant instead of trusting the extrapolation.
const REFERENCE_NET_CI_HALF_WIDTH_BPS: f64 = 1.09;

/// Blocks the reference half-width was measured over.
const REFERENCE_NET_CI_BLOCKS: usize = 256;

/// Expected 95% half-width on net growth per traded bar for a slice of `blocks` blocks.
fn expected_net_ci_half_width_bps(blocks: usize) -> f64 {
    if blocks == 0 {
        return f64::INFINITY;
    }
    REFERENCE_NET_CI_HALF_WIDTH_BPS
        * (REFERENCE_NET_CI_BLOCKS as f64 / blocks as f64).sqrt()
}

/// What one split holds, at one context, before any model sees it.
#[derive(Clone, Copy, Debug)]
pub(super) struct SplitCensus {
    pub(super) split: Split,
    pub(super) bars: usize,
    /// Near-disjoint windows the split can supply at the census context, which is the ceiling on
    /// `--windows` and therefore the ceiling on the interval any pass over it can reach.
    pub(super) anchors: usize,
    /// Symbols holding at least one such window. The pinned draw is quota-allocated per symbol,
    /// so this bounds how many DISTINCT symbols a draw can spread over.
    pub(super) symbols: usize,
}

/// The population a held-out pass will measure on, and the interval that population can
/// support — established by counting, with nothing scored.
///
/// Exists because `Split::Test` is scored ONCE for the whole campaign. "Does this split have
/// the power to resolve the effect we are looking for" has to be answerable before the draw is
/// spent, and it is answerable: the interval is set by the BLOCK count, the block count is a
/// property of the draw rather than of the model, and the draw is reproducible from
/// [`EVAL_WINDOW_SEED`] alone.
pub(super) struct HeldOutPower {
    pub(super) split: Split,
    pub(super) context: i64,
    /// Every split, so the addressed one is readable against the two it is not.
    pub(super) census: Vec<SplitCensus>,
    pub(super) windows_drawn: usize,
    pub(super) traded_windows: usize,
    pub(super) traded_blocks: usize,
    pub(super) fit_windows: usize,
    pub(super) fit_blocks: usize,
    /// `(traded windows, distinct blocks in that prefix, expected 95% half-width in bps/bar)`.
    ///
    /// The block count of each rung is COUNTED over the real draw, never assumed equal to the
    /// window count. That distinction is the point of the rung: at these draw sizes the pinned
    /// allocation gives most symbols a quota of one, so blocks track windows almost exactly —
    /// but "almost" is a measurement and the naive identity is not.
    pub(super) ladder: Vec<(usize, usize, f64)>,
}

impl HeldOutPower {
    /// Rungs of the ladder, as traded-window counts. Powers of two up to the whole draw, so the
    /// pinned prefix a published number was taken on always appears as a rung.
    fn rungs(windows_drawn: usize) -> Vec<usize> {
        let mut rungs: Vec<usize> = std::iter::successors(Some(trade_bench::TRADE_WINDOWS), |n| {
            Some(n * 2)
        })
        .take_while(|n| *n < windows_drawn)
        .collect();
        rungs.push(windows_drawn);
        rungs.retain(|n| *n > 0);
        rungs.dedup();
        rungs
    }

    /// Count, never score. `blocks_all` is the block id of every drawn window in draw order, so
    /// a prefix of it is exactly the block partition of that prefix of the draw.
    pub(super) fn measure(
        corpus: &BarCorpus,
        split: Split,
        context: i64,
        blocks_all: &[u64],
        traded: &[u64],
        fit: &[u64],
    ) -> Self {
        let distinct = |ids: &[u64]| ids.iter().collect::<BTreeSet<_>>().len();
        let census = [Split::Train, Split::Val, Split::Test]
            .into_iter()
            .map(|which| {
                let sampler = BarSampler::new(corpus, which, context, EVAL_WINDOW_SEED);
                let anchors = sampler.anchors();
                // Anchors are emitted symbol by symbol, so a change of symbol is a new symbol.
                let symbols = anchors
                    .windows(2)
                    .filter(|pair| pair[0].symbol != pair[1].symbol)
                    .count()
                    + usize::from(!anchors.is_empty());
                SplitCensus {
                    split: which,
                    bars: corpus.split_bars(which),
                    anchors: anchors.len(),
                    symbols,
                }
            })
            .collect();
        let ladder = Self::rungs(blocks_all.len())
            .into_iter()
            .map(|n| {
                let blocks = distinct(&blocks_all[..n.min(blocks_all.len())]);
                (n, blocks, expected_net_ci_half_width_bps(blocks))
            })
            .collect();
        Self {
            split,
            context,
            census,
            windows_drawn: blocks_all.len(),
            traded_windows: traded.len(),
            traded_blocks: distinct(traded),
            fit_windows: fit.len(),
            fit_blocks: distinct(fit),
            ladder,
        }
    }

    /// The census as exact integers, for the window manifest.
    ///
    /// The chart carries the same numbers as `f32`, where a 41-million bar count rounds to a
    /// multiple of four. That is immaterial to a symlog panel and fatal to a later reader trying
    /// to reproduce a block count, so the record keeps both.
    pub(super) fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "reference_half_width_bps": REFERENCE_NET_CI_HALF_WIDTH_BPS,
            "reference_blocks": REFERENCE_NET_CI_BLOCKS,
            "expected_half_width_bps": expected_net_ci_half_width_bps(self.traded_blocks),
            "context": self.context,
            "windows_drawn": self.windows_drawn,
            "traded_windows": self.traded_windows,
            "traded_blocks": self.traded_blocks,
            "fit_windows": self.fit_windows,
            "fit_blocks": self.fit_blocks,
            "census": self
                .census
                .iter()
                .map(|row| {
                    serde_json::json!({
                        "split": row.split.as_str(),
                        "bars": row.bars,
                        "windows": row.anchors,
                        "symbols": row.symbols,
                    })
                })
                .collect::<Vec<_>>(),
            "ladder": self
                .ladder
                .iter()
                .map(|(windows, blocks, half_width)| {
                    serde_json::json!({
                        "traded_windows": windows,
                        "blocks": blocks,
                        "expected_half_width_bps": half_width,
                    })
                })
                .collect::<Vec<_>>(),
        })
    }

    /// Stdout summary. The chart is the record; this is what a reader watching the rehearsal
    /// needs in order to abort it.
    pub(super) fn report_lines(&self) -> Vec<String> {
        let mut lines: Vec<String> = self
            .census
            .iter()
            .map(|row| {
                format!(
                    "[power] {:<5} {:>13} bars | {:>9} windows at context {} | {:>5} symbols",
                    row.split.as_str(),
                    row.bars,
                    row.anchors,
                    self.context,
                    row.symbols
                )
            })
            .collect();
        lines.push(format!(
            "[power] {} draw: {} windows, traded prefix {} windows over {} blocks, fit slice {} \
             windows over {} blocks",
            self.split.as_str(),
            self.windows_drawn,
            self.traded_windows,
            self.traded_blocks,
            self.fit_windows,
            self.fit_blocks
        ));
        lines.push(format!(
            "[power] expected 95% half-width on net growth at {} traded blocks: {:.3} bps/bar, \
             scaled from the campaign reference {:.2} bps/bar over {} blocks by sqrt(B_ref/B). \
             ASSUMES equal cross-block dispersion; the scoring pass must report its measured \
             s_block against the reference implied {:.3} bps/bar.",
            self.traded_blocks,
            expected_net_ci_half_width_bps(self.traded_blocks),
            REFERENCE_NET_CI_HALF_WIDTH_BPS,
            REFERENCE_NET_CI_BLOCKS,
            REFERENCE_NET_CI_HALF_WIDTH_BPS / 1.959_963_985
                * (REFERENCE_NET_CI_BLOCKS as f64).sqrt(),
        ));
        for (windows, blocks, half_width) in &self.ladder {
            lines.push(format!(
                "[power]   {windows:>6} traded windows -> {blocks:>6} blocks -> {half_width:.3} \
                 bps/bar"
            ));
        }
        lines
    }
}

/// Does the traded conditional MEAN stay calibrated as training proceeds, and does correcting
/// it recover the economics that were lost?
///
/// # The question
///
/// A likelihood is an average over a whole predictive law; a position is a function of that
/// law's conditional MEAN and almost nothing else. The two can move in opposite directions,
/// and on the run this was built for they did: the traded degree of freedom improved its NLL
/// monotonically while the realized Sharpe of the derived Kelly policy fell. Either the model
/// lost directional INFORMATION — in which case nothing after the fact can recover it — or it
/// kept the information and inflated its SCALE, in which case the loss is a sizing error and
/// an affine recalibration fitted out of sample recovers it. This measures which.
///
/// # The protocol, and where it could have been faked
///
/// Each checkpoint gets two passes over pinned held-out windows at the same fixed context:
///
/// 1. A FIT pass over windows the bench does not trade and whose `(symbol, calendar month)`
///    blocks are absent from the traded prefix ([`trade_bench::disjoint_fit_windows`]). It
///    produces the Mincer-Zarnowitz slope, and nothing else.
/// 2. An EVALUATION pass over the traded prefix, which solves the ordinary Kelly position AND
///    a second one under the mean recalibrated by the slope from pass 1.
///
/// Fitting the slope on the bars it is then evaluated on is the one way this experiment can
/// manufacture its own result: OLS minimizes squared error on exactly those bars, so the
/// recalibrated policy would be reading its own answer key. The disjointness is therefore
/// asserted at run time as well as in a test, on BLOCKS rather than on windows, because two
/// windows of one symbol-month share a regime and a slope can read a shared regime.
///
/// # Running it without saturating the machine
///
/// ```text
/// OMP_NUM_THREADS=1 TORCH_NUM_THREADS=1 RAYON_NUM_THREADS=1 ./torch-env.sh \
///     ./target/release/trading_bot_0 pretrain-calibration \
///     --checkpoint <weights>/pretrain_epoch_0_ctx2048.ot@10364 \
///     --checkpoint <weights>/pretrain_best.ot@30000 \
///     --output <run>/gens/<n> --min-dollar-volume 0
/// ```
///
/// Three separate thread budgets reach this pass and one variable each, which is why the
/// usual two are not enough. libtorch's intra-op pool answers to `OMP_NUM_THREADS` only:
/// `TORCH_NUM_THREADS` is a no-op in this binary because a pre-main constructor pins the
/// pool before anything would read it. The corpus load and the batch build run on rayon's
/// GLOBAL pool ([`crate::torch::dataset`] uses bare `par_iter`/`par_chunks_mut` with no pool
/// of its own), which rayon sizes from `RAYON_NUM_THREADS` and otherwise from physical
/// cores. A pool built locally with an explicit `num_threads` answers to neither and can
/// only be changed in code; this pass builds none.
///
/// MEASURED on one checkpoint, same box, card busy with another tenant: `RAYON_NUM_THREADS=1`
/// gives 77.5 s wall against 104.2 s user + 5.1 s sys, i.e. 1.41 cores; unbounded gives
/// 82.7 s wall against 144.6 s user + 8.7 s sys, i.e. 1.85 cores. So bounding the global
/// pool cuts CPU by a quarter and is very slightly FASTER in wall time, but this pass is
/// dominated by serial GPU submission and does not saturate the box either way — it is not
/// the shape of job the serialization rule exists for. Two checkpoints take about two and a
/// half minutes and about 1.1 GiB of device memory when a device is allowed.
pub fn pretrain_calibration(args: CalibrationArgs) -> Result<()> {
    ensure!(
        !args.checkpoints.is_empty(),
        "--checkpoint must be given at least once, as path@step"
    );
    ensure!(args.windows > 0, "--windows must be positive");
    ensure!(args.fit_windows > 0, "--fit-windows must be positive");
    ensure!(args.trade_windows > 0, "--trade-windows must be positive");
    ensure!(
        args.trade_windows < args.windows,
        "--trade-windows {} leaves no block-disjoint remainder out of a {}-window draw; the \
         calibration slope would be fitted on the bars it is evaluated on",
        args.trade_windows,
        args.windows
    );
    ensure!(args.context > 0, "--context must be positive");
    ensure!(args.batch_size > 0, "--batch-size must be positive");
    configure_threads();
    configure_cuda();

    let device = Device::cuda_if_available();
    let requested = args
        .checkpoints
        .iter()
        .map(|entry| parse_checkpoint_at_step(entry))
        .collect::<Result<Vec<_>>>()?;

    let corpus = load_corpus(&args.corpus)?;
    let mut set = PinnedSet::pinned(&corpus, args.split, args.context, args.windows)?;
    // Block ids are assigned over the WHOLE drawn set, so a `(symbol, month)` has one id in
    // both slices and disjointness is a statement about the same partition.
    let blocks_all = pinned_blocks(&set);
    let all_windows = std::mem::take(&mut set.windows);
    let traded_count = args.trade_windows.min(all_windows.len());
    let fit_indices =
        trade_bench::disjoint_fit_windows(&blocks_all, traded_count, args.fit_windows);
    ensure!(
        fit_indices.len() >= 2,
        "only {} of the {} drawn windows are block-disjoint from the traded prefix, which is \
         not enough to fit a calibration slope with an interval; raise --windows",
        fit_indices.len(),
        all_windows.len()
    );
    // Filled after the restriction below, so the interval's resampling units are exactly the
    // windows that were traded and never the unrestricted prefix.

    // POPULATION RESTRICTION. The traded prefix is narrowed to a named symbol set so an edge
    // can be measured on exactly the names a cost was priced on. Two properties are kept
    // deliberately: the selection RULE does not change — the position is a per-bar function of
    // the model's own law with no cross-sectional ranking, so restricting the population
    // cannot re-derive a threshold the way a quantile-based selector would — and the FIT slice
    // is left alone, so the recalibration slope is still fitted on windows disjoint in both
    // symbol and block from every window it is evaluated on.
    let mut traded_indices: Vec<usize> = (0..traded_count).collect();
    if !args.restrict_symbols.is_empty() {
        let wanted: BTreeSet<&str> = args.restrict_symbols.iter().map(String::as_str).collect();
        traded_indices.retain(|index| wanted.contains(set.sampler.symbol(all_windows[*index].symbol)));
        ensure!(
            traded_indices.len() >= 2,
            "--restrict-symbols matched only {} of the {} traded windows; a blocked interval \
             needs at least two resampling units",
            traded_indices.len(),
            traded_count
        );
        let missing: Vec<&str> = wanted
            .iter()
            .copied()
            .filter(|symbol| {
                !traded_indices
                    .iter()
                    .any(|index| set.sampler.symbol(all_windows[*index].symbol) == *symbol)
            })
            .collect();
        println!(
            "population restricted to {} of {} traded windows over {} named symbols{}",
            traded_indices.len(),
            traded_count,
            wanted.len(),
            if missing.is_empty() {
                String::new()
            } else {
                format!(", {} of which the traded prefix does not carry: {}", missing.len(), missing.join(","))
            }
        );
    }
    let eval_blocks: Vec<u64> = traded_indices.iter().map(|index| blocks_all[*index]).collect();
    let fit_blocks: Vec<u64> = fit_indices.iter().map(|index| blocks_all[*index]).collect();
    ensure!(
        trade_bench::blocks_disjoint(&fit_blocks, &eval_blocks),
        "the calibration fit slice shares a (symbol, calendar month) block with the traded \
         slice, so the recalibration would be fitted on the regime it is evaluated in"
    );
    let fit_windows: Vec<WindowRef> = fit_indices
        .iter()
        .map(|index| all_windows[*index])
        .collect();
    let traded_windows: Vec<WindowRef> =
        traded_indices.iter().map(|index| all_windows[*index]).collect();
    println!(
        "mean calibration on the pinned {:?} split at context {}: fitting on {} windows over \
         {} blocks, evaluating on the {} traded windows over {} blocks, block-DISJOINT",
        args.split,
        args.context,
        fit_windows.len(),
        fit_blocks.iter().collect::<BTreeSet<_>>().len(),
        traded_windows.len(),
        eval_blocks.iter().collect::<BTreeSet<_>>().len(),
    );

    // The two slices, named. An economic number measured on 256 of 4096 pinned windows is
    // quotable only beside the symbols and months it was measured on: a cost, a liquidity
    // decile or a capacity figure taken over a different universe is not a cost for these
    // bars, and without this file the mismatch is invisible to whoever builds the table.
    let slice_rows = |windows: &[WindowRef], blocks: &[u64]| -> Vec<serde_json::Value> {
        windows
            .iter()
            .zip(blocks)
            .map(|(window, block)| {
                serde_json::json!({
                    "symbol": set.sampler.symbol(window.symbol),
                    "bar_index": window.bar_index,
                    "ts_ms": set.sampler.anchor_ts_ms(window),
                    "block": block,
                })
            })
            .collect()
    };
    // THE POPULATION, MEASURED, WITH NOTHING SCORED. Every step above decides WHAT is measured
    // — the draw, the traded prefix, the block partition, the disjointness — and no step above
    // has opened a checkpoint, so this is the whole answer to "can this split resolve the effect
    // we are looking for" and it is available before the draw is spent.
    let power = HeldOutPower::measure(
        &corpus,
        args.split,
        args.context,
        &blocks_all,
        &eval_blocks,
        &fit_blocks,
    );
    let provenance = serde_json::json!({
        "split": format!("{:?}", args.split),
        "context": args.context,
        "windows_drawn": all_windows.len(),
        "eval_window_seed": EVAL_WINDOW_SEED,
        "traded": slice_rows(&traded_windows, &eval_blocks),
        "fit": slice_rows(&fit_windows, &fit_blocks),
        // Beside the window list because "this interval is 1.09 bps wide" is only checkable
        // against the block count it was taken over, and the chart rounds counts to f32.
        "power": power.to_json(),
    });
    let provenance_path = Path::new(&args.output).join("pretrain_calibration_windows.json");
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("failed to create {}", args.output))?;
    std::fs::write(&provenance_path, serde_json::to_vec_pretty(&provenance)?)
        .with_context(|| format!("failed to write {}", provenance_path.display()))?;
    println!("traded and fit window slices written to {}", provenance_path.display());

    // The census is part of the pass's provenance as much as the window list is: "this interval
    // is 1.09 bps wide" is only checkable against the block count it was taken over.
    for line in power.report_lines() {
        println!("{line}");
    }
    super::pretrain_reports::write_heldout_power(Path::new(&args.output), &power)?;
    if args.dry_run {
        println!(
            "--dry-run: the {} split loaded, {} windows drawn and blocked, nothing scored. No \
             checkpoint was opened.",
            args.split.as_str(),
            all_windows.len()
        );
        return Ok(());
    }

    let mut points: Vec<super::pretrain_reports::CalibrationPoint> = Vec::new();
    for (path, step) in &requested {
        let weights = Path::new(path);
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
        if let Some(trained) = world.metadata().training.as_ref() {
            ensure!(
                trained.eval_window_seed == EVAL_WINDOW_SEED,
                "checkpoint pinned its bench with eval_window_seed {:#x} but this build uses \
                 {EVAL_WINDOW_SEED:#x}; the windows would not be the run's own",
                trained.eval_window_seed
            );
        }
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

        // Pass 1: the slope, on windows this checkpoint's bench will never trade.
        set.windows = fit_windows.clone();
        let fit_stats = evaluate(
            world.modules(),
            world.deployment_supports(),
            &set,
            args.batch_size,
            device,
            true,
            scoring,
            None,
            fit_windows.len(),
        )?;
        let fit_calibration =
            trade_bench::mean_calibration(&fit_stats.trade_paths.windows, &fit_blocks);
        let shrink = fit_calibration.shrink().ok_or_else(|| {
            anyhow!(
                "the calibration regression degenerated on the fit slice of {}; there is no \
                 slope to recalibrate with",
                weights.display()
            )
        })?;
        for line in fit_calibration.report_lines() {
            println!("{} fit slice: {line}", weights.display());
        }

        // Pass 2: the bench itself, plus the same solve under the recalibrated mean.
        set.windows = traded_windows.clone();
        let stats = evaluate(
            world.modules(),
            world.deployment_supports(),
            &set,
            args.batch_size,
            device,
            true,
            scoring,
            Some(shrink),
            traded_windows.len(),
        )?;
        let config = BenchConfig::new(
            trade_bench::DEFAULT_COST_BPS,
            trade_bench::LEVERAGE_CAP,
            trade_bench::marginal_position(world.deployment_supports(), trade_bench::FREE_LEVERAGE),
        );
        let trade = trade_bench::bench(
            &stats.trade_paths.windows,
            &eval_blocks,
            &stats.trade_paths.tail,
            config,
        );
        let shrunk = trade_bench::shrunk_bench(
            &stats.trade_paths.windows,
            &eval_blocks,
            config,
            shrink,
        )
        .ok_or_else(|| {
            anyhow!("the evaluation pass produced no recalibrated fraction to score")
        })?;
        println!(
            "\n=== {} (step {step}, lineage {}) nll {:.4} nats/bar ===",
            weights.display(),
            world.lineage_sha256(),
            stats.nll_bar,
        );
        for line in trade.report_lines() {
            println!("  {line}");
        }
        for line in shrunk.report_lines(&trade) {
            println!("  {line}");
        }

        // The COST-AWARE sizing axis, on the same windows, the same blocks and the same
        // ledger. The Kelly solve above is cost-blind by construction, so the incumbent
        // policy rebalances to the frictionless optimum every bar; under proportional costs
        // the optimum has a no-trade region instead, and this is what that is worth.
        let mut bands: Vec<trade_bench::BandSweep> = Vec::new();
        for source in [
            trade_bench::BandSource::Frictionless,
            trade_bench::BandSource::Recalibrated,
        ] {
            for shape in trade_bench::SIZING_SHAPES {
                if let Some(sweep) = trade_bench::band_sweep(
                    &stats.trade_paths.windows,
                    &eval_blocks,
                    config,
                    source,
                    shape,
                ) {
                    for line in sweep.report_lines() {
                        println!("  {line}");
                    }
                    bands.push(sweep);
                }
            }
        }
        // Both levers cut turnover, so their gains cannot be added. The interaction is the
        // second difference, paired per window over the same blocks.
        let mut band_overlap: Vec<trade_bench::BandShrinkOverlap> = Vec::new();
        for shape in trade_bench::SIZING_SHAPES {
            if let Some(rows) = trade_bench::band_shrink_overlap(
                &stats.trade_paths.windows,
                &eval_blocks,
                config,
                shape,
            ) {
                println!("  band vs shrink, are they substitutes:");
                for row in &rows {
                    println!("    {}", row.report_line());
                }
                band_overlap.extend(rows);
            }
        }
        let attribution =
            trade_bench::edge_attribution(&stats.trade_paths.windows, &eval_blocks, config);
        // The same arm table on the FIT slice. Its only consumer is the cost join, which is
        // slice-matched by design: the two slices share NO name, so a fit arm scaled against the
        // traded book's participation would mix two disjoint populations. Not printed - the fit
        // slice's economics are not a result, they are the selection's own denominator.
        let fit_attribution = Some(trade_bench::edge_attribution(
            &fit_stats.trade_paths.windows,
            &fit_blocks,
            config,
        ));
        for line in attribution.report_lines() {
            println!("  {line}");
        }
        let hysteresis = trade_bench::hysteresis_sweep(
            &stats.trade_paths.windows,
            &eval_blocks,
            config,
            trade_bench::ConvictionAxis::Raw,
        );
        if let Some(sweep) = &hysteresis {
            for line in sweep.report_lines() {
                println!("  {line}");
            }
        }
        let decay = trade_bench::signal_decay(&stats.trade_paths.windows, &eval_blocks);
        for line in decay.report_lines() {
            println!("  {line}");
        }
        // Both conviction axes, gated identically. The standardized axis exists because a
        // threshold on raw |mu| is a covert LIQUIDITY filter - it retains volatile names, which
        // are thin and dear - so the two are compared on the cost of what they retain, not only
        // on net growth.
        let gates: Vec<trade_bench::HysteresisOos> = trade_bench::CONVICTION_AXES
            .iter()
            .filter_map(|axis| {
                trade_bench::hysteresis_out_of_sample(
                    &fit_stats.trade_paths.windows,
                    &fit_blocks,
                    &stats.trade_paths.windows,
                    &eval_blocks,
                    config,
                    *axis,
                )
            })
            .collect();
        for gate in &gates {
            for line in gate.report_lines() {
                println!("{line}");
            }
        }
        let hysteresis_oos = gates
            .iter()
            .find(|gate| gate.axis == trade_bench::ConvictionAxis::Raw)
            .cloned();
        // The 2x2 is crossed at the margin the GATE chose, never at an argmax re-picked on
        // these windows: a cell chosen here would be compared against three fixed policies.
        let composition = hysteresis_oos.as_ref().and_then(|gate| {
            trade_bench::hysteresis_composition(
                &stats.trade_paths.windows,
                &eval_blocks,
                config,
                gate.fitted_margin_bps,
                gate.axis,
            )
        });
        if let Some(rows) = &composition {
            for line in rows.report_lines() {
                println!("{line}");
            }
        }
        points.push(super::pretrain_reports::CalibrationPoint {
            label: weights
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.clone()),
            step: *step,
            nll_bar: stats.nll_bar,
            nll_bar_conditional: stats.nll_bar_conditional,
            eval: trade.calibration,
            fit: trade.calibration,
            attribution,
            fit_attribution,
            trade,
            shrunk,
            bands,
            band_overlap,
            hysteresis,
            decay,
            composition,
            gates,
        });
    }

    // The turnover-weighted flat-equivalent cost needs the BOOK's own weights, and a window is
    // a symbol, so the join key is the traded window's symbol and the payload is what each arm
    // actually rotated in it. Written back into the provenance file rather than a new artifact
    // because that file already carries the symbol, block and timestamp this joins against;
    // splitting the key from the payload across two files is how a join silently misaligns.
    let mut turnover_rows: Vec<serde_json::Value> = Vec::new();
    let mut fit_turnover_rows: Vec<serde_json::Value> = Vec::new();
    let mut fit_frontier_rows: Vec<serde_json::Value> = Vec::new();
    for point in &points {
        let arms = trade_bench::ATTRIBUTION_NAMES
            .iter()
            .enumerate()
            .map(|(arm, policy)| ((*policy).to_owned(), &point.attribution.turnover[arm]))
            // The recalibrated book is the better-performing arm and its turnover weighting is
            // NOT recoverable by rescaling the unshrunk book's, so it ships as its own row set
            // rather than as a bracket someone has to interpolate inside.
            .chain(std::iter::once((
                "recalibrated (shrunk mean)".to_owned(),
                &point.shrunk.turnover,
            )))
            // Then the WHOLE frontier, every margin on every conviction axis, on BOTH window
            // sets. Per-margin rather than the fitted margin alone because the cost is NOT
            // margin-invariant: a conviction threshold retains volatile names, which are thin and
            // dear, so each margin's book carries its own weighting and the cost has to be
            // measured per row. On both slices because the corrected argmax must be taken on the
            // FIT slice's own cost - pricing a fit-slice argmax with traded-slice weights is
            // in-sample selection one level up.
            .chain(point.gates.iter().flat_map(|gate| {
                gate.traded_frontier
                    .points
                    .iter()
                    .map(move |row| (gate.axis.policy_label(row.margin_bps), &row.turnover))
            }));
        for (policy, rows) in arms {
            if rows.is_empty() {
                continue;
            }
            ensure!(
                rows.len() == traded_windows.len(),
                "{policy} turnover carries {} windows against {} traded: the join key and the \
                 payload have come from different window sets",
                rows.len(),
                traded_windows.len()
            );
            for (window, row) in traded_windows.iter().zip(rows) {
                turnover_rows.push(serde_json::json!({
                    "symbol": set.sampler.symbol(window.symbol),
                    "checkpoint": point.label,
                    "step": point.step,
                    "policy": policy,
                    "turnover": row.total,
                    "turnover_interior": row.interior,
                    "bars": row.bars,
                }));
            }
        }
        // The standing arms on the FIT slice, `actual` among them. Without a fit-side `actual`
        // there is no slice-matched participation baseline, so no fit-side all-in figure can be
        // formed at all - and forming one from the traded incumbent would scale a fit arm against
        // a disjoint population.
        for (arm, policy) in trade_bench::ATTRIBUTION_NAMES.iter().enumerate() {
            let Some(fit_attribution) = &point.fit_attribution else {
                continue;
            };
            let rows = &fit_attribution.turnover[arm];
            if rows.is_empty() {
                continue;
            }
            ensure!(
                rows.len() == fit_windows.len(),
                "{policy} fit turnover carries {} windows against {} fit",
                rows.len(),
                fit_windows.len()
            );
            for (window, row) in fit_windows.iter().zip(rows) {
                fit_turnover_rows.push(serde_json::json!({
                    "symbol": set.sampler.symbol(window.symbol),
                    "checkpoint": point.label,
                    "step": point.step,
                    "policy": policy,
                    "turnover": row.total,
                    "turnover_interior": row.interior,
                    "bars": row.bars,
                }));
            }
        }
        // THE BLOCKING COLUMNS: tau_fit and break_even_fit per grid arm, so the honest argmax
        // net_fit = tau_fit * (break_even_fit - c_fit) can be taken entirely on fit-slice
        // quantities. Every field here is the FIT frontier's own; taking tau or break-even from
        // the traded frontier would be in-sample selection one level up. Scalars rather than
        // per-window rows because the argmax is over arms, not names.
        for gate in &point.gates {
            for row in &gate.fit_frontier.points {
                fit_frontier_rows.push(serde_json::json!({
                    "checkpoint": point.label,
                    "step": point.step,
                    "axis": gate.axis.name(),
                    "units": gate.axis.units(),
                    "margin": if row.margin_bps.is_finite() { Some(row.margin_bps) } else { None },
                    "policy": gate.axis.policy_label(row.margin_bps),
                    "turnover": row.policy.turnover,
                    "break_even_bps": if row.break_even_bps.is_finite() {
                        Some(row.break_even_bps)
                    } else {
                        None
                    },
                    "mean_hold_bars": row.mean_hold_bars,
                    "hit_rate": row.policy.hit_rate,
                    "sharpe": row.policy.sharpe,
                    "edge_bps_at_assumed_cost": row.edge.mean * 1e4,
                    "net_at_selection_cost_bps": row.net_at_measured.mean * 1e4,
                    "net_at_selection_ci_low": row.net_at_measured.ci_low * 1e4,
                    "net_at_selection_ci_high": row.net_at_measured.ci_high * 1e4,
                    "fitted": row.margin_bps == gate.fitted_margin_bps,
                }));
            }
        }
        // The fit slice's own frontier, keyed to the FIT windows. A separate array because it
        // joins against a different window set; sharing one array would invite a join on the
        // wrong keys.
        for gate in &point.gates {
            for row in &gate.fit_frontier.points {
                if row.turnover.is_empty() {
                    continue;
                }
                let policy = gate.axis.policy_label(row.margin_bps);
                ensure!(
                    row.turnover.len() == fit_windows.len(),
                    "{policy} fit turnover carries {} windows against {} fit",
                    row.turnover.len(),
                    fit_windows.len()
                );
                for (window, entry) in fit_windows.iter().zip(&row.turnover) {
                    fit_turnover_rows.push(serde_json::json!({
                        "symbol": set.sampler.symbol(window.symbol),
                        "checkpoint": point.label,
                        "step": point.step,
                        "policy": policy,
                        "turnover": entry.total,
                        "turnover_interior": entry.interior,
                        "bars": entry.bars,
                    }));
                }
            }
        }
    }
    let mut provenance = provenance;
    provenance["turnover"] = serde_json::Value::Array(turnover_rows);
    provenance["turnover_fit"] = serde_json::Value::Array(fit_turnover_rows);
    provenance["fit_frontier"] = serde_json::Value::Array(fit_frontier_rows);
    std::fs::write(&provenance_path, serde_json::to_vec_pretty(&provenance)?)
        .with_context(|| format!("failed to write {}", provenance_path.display()))?;
    let arms_emitted: std::collections::BTreeSet<&str> = provenance["turnover"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .filter_map(|row| row["policy"].as_str())
                .collect()
        })
        .unwrap_or_default();
    println!(
        "per-window turnover for {} arms written to {}: {}",
        arms_emitted.len(),
        provenance_path.display(),
        arms_emitted.into_iter().collect::<Vec<_>>().join(" | ")
    );
    println!(
        "per-window turnover on the FIT slice: {} rows over {} arms, joined to the {} fit windows",
        provenance["turnover_fit"]
            .as_array()
            .map_or(0, |rows| rows.len()),
        provenance["turnover_fit"]
            .as_array()
            .map(|rows| rows
                .iter()
                .filter_map(|row| row["policy"].as_str())
                .collect::<std::collections::BTreeSet<_>>()
                .len())
            .unwrap_or_default(),
        fit_windows.len()
    );

    println!();
    for line in super::pretrain_reports::calibration_verdict_lines(&points) {
        println!("{line}");
    }
    let output = Path::new(&args.output);
    super::pretrain_reports::write_mean_calibration(
        output,
        &format!(
            "{:?} split at context {}, fit on {} block-disjoint windows",
            args.split,
            args.context,
            fit_windows.len()
        ),
        &points,
    )?;
    println!("reports written to {}", output.display());
    Ok(())
}

/// `path@step`, the form the calibration trend's x-axis is stated in.
pub(super) fn parse_checkpoint_at_step(entry: &str) -> Result<(String, usize)> {
    let (path, step) = entry.rsplit_once('@').ok_or_else(|| {
        anyhow!(
            "--checkpoint takes `path@step`, got {entry}; the optimizer step is not recorded \
             in the metadata sidecar and a calibration TREND without an x-axis is not a trend"
        )
    })?;
    let step = step
        .parse::<usize>()
        .with_context(|| format!("the step in {entry} is not an integer"))?;
    ensure!(!path.is_empty(), "--checkpoint {entry} has an empty path");
    Ok((path.to_owned(), step))
}

/// The pictures as numbers: the realized close path against the ancestral median and the
/// 10/90 band, so the run is legible without opening an image.
///
/// Prices are chained from a previous close of `1.0`, so every column is a cumulative
/// return from the forecast origin and windows at different price levels are comparable.
fn print_candle_windows(
    corpus: &BarCorpus,
    set: &PinnedSet,
    drawn: &[super::pretrain_reports::CandleWindow],
) {
    // DOF row `j` of a window carries bar `bar_index + j`, and the history keeps the
    // first `context - SNAPSHOT_HORIZON` of them, so the forecast origin is the bar
    // immediately before the first predicted one.
    let origin_offset = (set.context - SNAPSHOT_HORIZON - 1) as usize;
    for (index, (reference, paths)) in set.windows.iter().zip(drawn).enumerate() {
        let series = reference.symbol as usize;
        let steps = paths.actual_close.len();
        println!(
            "\nwindow {:02}  {}  anchor {}  forecast origin {}  ({steps} bars)",
            index + 1,
            corpus.symbol(series),
            iso_ms(set.sampler.anchor_ts_ms(reference)),
            iso_ms(corpus.ts_ms(series, reference.bar_index as usize + origin_offset)),
        );
        let (p10, centre, p90) = (paths.p10(), paths.fan_centre(), paths.p90());
        println!("     bar   realized       p10    centre       p90     rank   in band");
        // Every bar of a 100-step horizon is a wall of numbers nobody reads; the
        // powers-of-two ladder plus the final bar shows the shape and the endpoint.
        for t in (0..steps).filter(|t| t + 1 == steps || (t + 1).is_power_of_two()) {
            let realized = paths.actual_close[t];
            let (low, mid, high) = (p10[t], centre[t], p90[t]);
            println!(
                "  {:6}  {realized:8.5}  {low:8.5}  {mid:8.5}  {high:8.5}  {:7.3}  {:>7}",
                t + 1,
                paths.rank[t],
                if realized >= low && realized <= high {
                    "yes"
                } else {
                    "NO"
                },
            );
        }
        let covered = (0..steps)
            .filter(|&t| paths.actual_close[t] >= p10[t] && paths.actual_close[t] <= p90[t])
            .count();
        // The centre is the per-horizon median LOCUS, not a forecast path, so its distance
        // from the realization is reported as a rank inside the fan rather than as an error,
        // and it is printed beside the standard error of estimating that centre from
        // `samples` draws. A centre wiggle smaller than that error is noise, and reading it
        // as drift is the specific mistake this table used to invite.
        println!(
            "  coverage {covered}/{steps} ({:.0}%), terminal realized {:+.2}% vs fan centre \
             {:+.2}% +/- {:.2}% (se), terminal band [{:+.2}%, {:+.2}%] over {} ancestral draws",
            100.0 * covered as f64 / steps as f64,
            100.0 * (paths.actual_close[steps - 1] - 1.0) as f64,
            100.0 * (centre[steps - 1] - 1.0) as f64,
            100.0 * paths.centre_log_se(steps - 1),
            100.0 * (p10[steps - 1] - 1.0) as f64,
            100.0 * (p90[steps - 1] - 1.0) as f64,
            paths.samples,
        );
    }
}

fn validate_args(args: &PretrainArgs) -> Result<()> {
    ensure!(args.epochs > 0, "--epochs must be at least 1");
    // The old guard here warned only above 4, on the grounds that "a ~350M bar corpus
    // saturates near 4". That number was ASSERTED: it appeared exactly once in the tree, in
    // the println that printed it, with no measurement, no test and no report base behind it,
    // and it was computed off the NOMINAL bar count. The nominal count is the wrong
    // denominator by two and a half orders of magnitude. 366,163,264 bars per pass are
    // 5,297 symbols sharing ONE wall-clock grid of 197,916 five-minute instants, and
    // same-instant returns across symbols correlate at rho = 0.176 (95% CI 0.158..0.201,
    // measured pairwise-complete on this corpus at min_dollar_volume 0, i.e. the universe a
    // run actually trains on). Cross-sectional design effect 1 + (1850 - 1) * rho = 327
    // against a within-symbol serial inflation of 1.10, so ESS = 1.0M, interval
    // [0.57M, 1.13M]. Deff >> 1 makes ESS -> instants / (rho * inflation), so the figure does
    // not depend on the symbols-per-instant estimate. Against 31.8M parameters that is 31
    // parameters per EFFECTIVE observation while the nominal count suggests 0.087 — the
    // correction flips the regime from data-rich to data-poor.
    //
    // So the useful range starts at 1, not 4, and the warning fires above 1. Passes beyond
    // the first add ZERO new market-factor realizations, because they re-present the same
    // 1,031 sessions; they buy optimization, never information. This warns rather than
    // refusing because a deliberate multi-pass diagnostic arm is legitimate — but it must be
    // deliberate, and `bardist_v2` took 3 from a DEFAULT.
    if args.epochs > 1 {
        println!(
            "warning: --epochs {} repeats the corpus. Passes beyond the first add no new \
             market-factor realizations: the 5,297 symbols share ONE wall-clock grid of \
             ~197,916 five-minute instants and same-instant cross-symbol correlation is \
             rho ~ 0.176, so the ~366M nominal bars per pass carry an effective sample size \
             of ~1.0M ([0.57M, 1.13M]) against 31.8M parameters — ~31 parameters per \
             EFFECTIVE observation at ONE pass. Repetition buys optimization, not information.",
            args.epochs
        );
    }
    ensure!(
        args.lr_plateau_fraction > 0.0 && args.lr_plateau_fraction < 1.0,
        "--lr-plateau-fraction must lie strictly inside (0, 1), got {}. At 1.0 the rate never \
         decays to the {LR_FLOOR_MULTIPLIER}x floor at all, and at 0.0 there is no plateau — \
         no stretch of the run across which passes accumulate at zero learning-rate contrast, \
         which is the only place the two are separable. Both are degenerate schedules rather \
         than extreme ones, so they are refused at the boundary instead of clamped.",
        args.lr_plateau_fraction
    );
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
        args.lambda_growth >= 0.0 && args.lambda_growth.is_finite(),
        "--lambda-growth must be a non-negative finite weight, got {}. A NEGATIVE weight \
         would maximize the expected log LOSS, i.e. it would train the model to bet the \
         wrong way, and it would do it while every likelihood metric kept improving.",
        args.lambda_growth
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
    ensure!(
        args.snapshot_samples >= MIN_FAN_SAMPLES,
        "--snapshot-samples must be at least {MIN_FAN_SAMPLES}: below that p25 and p75 of the \
         ancestral draws are the same order statistic, every band the picture prints has an \
         error bar of exactly zero, and the fan writer refuses it at the first epoch boundary \
         rather than 40 hours in"
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

/// One intra-op and one inter-op thread by default.
///
/// Called from [`pretrain`] and nowhere else: the interop pool can only be sized before torch
/// does any parallel work, and torch raises rather than ignoring a late call. Sizing the
/// process is the entry point's business, not [`build_trainer`]'s, which is also what lets a
/// test drive the trainer inside a harness that has already run tensor work.
pub(crate) fn configure_threads() {
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
pub(super) fn load_corpus(args: &CorpusFlags) -> Result<BarCorpus> {
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
fn effective_split_bounds(args: &CorpusFlags) -> Result<Option<(i64, i64)>> {
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
///
/// This is also where a support that cannot be PERSISTED is refused, at step 0. Every
/// promotion writes the run's own supports back out as a checkpoint sidecar, and
/// [`BarSupports::save`] can only write the current schema, which requires fitted per-bin
/// moments a pre-v5 artifact does not carry. Discovering that at the first promotion costs the
/// whole warmup — the defect this check exists for burned 1000 steps — and every step after it
/// would be unpromotable too, so the run has nothing to gain by starting.
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
    let (supports, frozen) =
        fit_supports_at(corpus, &path, SupportsFit::of(args), corpus_fingerprint)?;
    if !supports.bin_means_measured() {
        let version = bar_supports_format_version(&path)?;
        bail!(
            "the bar supports this run loaded, {}, are format version {version} and carry no \
             fitted per-bin moments, so this run can never write a checkpoint: every promotion \
             persists these supports beside the weights as a version \
             {BAR_SUPPORTS_FORMAT_VERSION} `.supports.<res>.json` sidecar, and that schema \
             requires the moments. Point --supports at a version \
             {BAR_SUPPORTS_MOMENTS_VERSION} artifact carrying fitted moments, or measure them \
             onto this exact geometry with `bar-supports-moments --supports {} \
             --output-supports <new path>`, which never refits the bins and so leaves the \
             `nll_bar` scale untouched",
            path.display(),
            path.display()
        );
    }
    Ok((supports, frozen))
}

/// The three run scalars a support fit depends on, so [`fit_supports_at`] does not need a whole
/// [`PretrainArgs`] and a caller cannot silently pass the wrong `usize` for the `u64`.
#[derive(Clone, Copy, Debug)]
pub(super) struct SupportsFit {
    pub samples: usize,
    pub seed: u64,
    pub freeze: bool,
}

impl SupportsFit {
    fn of(args: &PretrainArgs) -> Self {
        Self {
            samples: args.support_samples,
            seed: args.seed,
            freeze: args.freeze_supports,
        }
    }
}

/// [`fit_supports`] against an explicit path and explicit scalars.
///
/// Exists so an auxiliary resolution's supports go through THIS provenance check rather than a
/// second implementation: `--supports` names one file and can only ever mean the deployment
/// resolution, so the auxiliary always passes its own `bar_supports.<res>.json`. Takes the three
/// scalars it needs rather than [`PretrainArgs`] so fitting a resolution's supports is reachable
/// without standing up a whole run.
pub(super) fn fit_supports_at(
    corpus: &BarCorpus,
    path: &Path,
    fit: SupportsFit,
    corpus_fingerprint: &str,
) -> Result<(BarSupports, bool)> {
    if path.exists() {
        let supports = BarSupports::load(path)
            .with_context(|| format!("cached supports {} are unreadable", path.display()))?;
        ensure!(
            supports.num_bins() == NUM_BAR_BINS,
            "cached supports {} have {} bins, this build uses {NUM_BAR_BINS}",
            path.display(),
            supports.num_bins()
        );
        let frozen = require_supports_provenance(
            supports.provenance(),
            path,
            corpus_fingerprint,
            corpus.split_bounds(),
            fit.freeze,
        )?;
        return Ok((supports, frozen));
    }
    ensure!(
        !fit.freeze,
        "--freeze-supports was given but {} does not exist; point --supports at the frozen \
         artifact, or drop the flag to fit a new one",
        path.display()
    );
    println!(
        "fitting {}s bin supports from {} training bars (seed 0x{:X})",
        corpus.res_secs(),
        fit.samples,
        fit.seed
    );
    let supports = corpus
        .fit_supports(fit.samples, fit.seed)
        .with_provenance(BarSupportsProvenance {
            corpus_fingerprint: corpus_fingerprint.to_owned(),
            split_bounds: corpus.split_bounds(),
            sample_count: fit.samples,
            fitted_utc: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        });
    // `BarCorpus::fit_supports` already persisted the provenance-free object, so rewrite it
    // with the stamp attached rather than leaving an unverifiable artifact on disk.
    supports
        .save(path)
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
///   and never inform any decision during the run. `test_diagnostic` is the same split at
///   the fixed diagnostic context, for the run that never reached the deployed one: the
///   terminal number has to be measured at the context the checkpoint was selected at, or it
///   measures positional extrapolation instead of generalization.
struct EvaluationSets {
    diagnostic: PinnedSet,
    promotion: PinnedSet,
    snapshot: PinnedSet,
    test: PinnedSet,
    test_diagnostic: PinnedSet,
    test_snapshot: PinnedSet,
}

/// A pinned held-out window set. `pub(super)` so the sibling audit modules — the directional
/// skill audit, the horizon sweep — draw the SAME windows under the SAME seed through the same
/// constructor, rather than each reimplementing the draw and quietly measuring different data.
pub(super) struct PinnedSet {
    pub(super) sampler: BarSampler,
    pub(super) windows: Vec<WindowRef>,
    pub(super) context: i64,
}

impl PinnedSet {
    /// Draw `count` near-disjoint windows of `context` bars from `split`.
    ///
    /// EVAL_WINDOW_SEED, never `args.seed`: the bench must not move when the training
    /// seed does, or a seed replicate measures two things at once and neither. This is
    /// the ONLY place a pinned set is built, so the standalone candle entry point
    /// depicts byte-identical windows to the ones a run charts itself.
    pub(super) fn pinned(
        corpus: &BarCorpus,
        split: Split,
        context: i64,
        count: usize,
    ) -> Result<Self> {
        let sampler = BarSampler::new(corpus, split, context, EVAL_WINDOW_SEED);
        let windows = sampler.pinned_windows(count);
        ensure!(
            !windows.is_empty(),
            "the {} split has no window of {context} bars; the corpus is too small",
            split.as_str()
        );
        Ok(Self {
            sampler,
            windows,
            context,
        })
    }
}

impl EvaluationSets {
    fn new(corpus: &BarCorpus, args: &PretrainArgs) -> Result<Self> {
        let build = |split: Split, context: i64, count: usize| -> Result<PinnedSet> {
            PinnedSet::pinned(corpus, split, context, count)
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
            test_diagnostic: build(
                Split::Test,
                args.diagnostic_context,
                args.validation_windows,
            )?,
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
        // The reference cadence: AdamW updates on odd steps over the gradient accumulated
        // across the pair, so the embedding tables and the five emission heads see an
        // effective 2x batch. Set here rather than in `MuonConfig::default` because the
        // PPO and planner trainers share this optimizer under a different recipe.
        adamw_every: 2,
        quadratic_lr_weight_decay: true,
        cautious_weight_decay: true,
        adamw_beta_overrides: beta_overrides,
        adamw_weight_decay_multipliers: wd_multipliers,
        ..MuonConfig::default()
    };

    let mut optimizer = Muon::new_named(named, cfg);
    assert_routing_partitions(named, &optimizer, &muon, &adamw_tables, &adamw_scalars)?;

    // The per-matrix `max(1, rows/cols).sqrt()` multiplier is applied natively by the NorMuon
    // step, and is 1.0 for every down-projection because all of them are wider than tall;
    // only the extra bumps are configured here.
    let down = bar_muon_down_projection_substrings();
    let matched = optimizer.set_named_lr_scale(down, NORMUON_DOWN_PROJECTION_LR_MULT);
    ensure!(
        matched > 0,
        "no MLP down-projection matched {down:?}; the {NORMUON_DOWN_PROJECTION_LR_MULT}x \
         learning-rate bump would be a no-op"
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
    /// Device-resident constants of the expected-log-growth term, one per bin geometry:
    /// index 0 is the deployment resolution and `1 + i` is `aux[i]`. Built once because the
    /// alternative is a `[1, NUM_BAR_BINS]` host-to-device copy on every step, and because
    /// the support-bound assertion inside [`GrowthSupport::new`] belongs where it can fail
    /// before the run starts.
    growth_supports: Vec<GrowthSupport>,
    vs: nn::VarStore,
    modules: BarModules,
    optimizer: Muon,
    train_samplers: Vec<BarSampler>,
    eval: EvaluationSets,
    reporter: PretrainReporter,
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
    /// The corpus partition this run's epochs are passes over. Owns the per-stage window
    /// assignment; `windows_per_stage` is what the step schedule was derived from.
    pass: PassPlan,
    /// The current epoch's anchors, per stage, in issue order. Replaced at every pass
    /// boundary; `Arc` because the plan caches it and the trainer holds it for the epoch.
    pass_layout: Arc<PassLayout>,
    /// Per-window issue counts for the current epoch. The coverage invariant is checked
    /// against this and nothing else, so it is marked where the batch is BUILT.
    pass_ledger: PassLedger,
    /// Windows of each stage already handed out this epoch. Sequential and never wrapped: a
    /// stage that wraps is re-training bars it has already seen while others go untouched,
    /// which is exactly the defect the partition exists to remove.
    stage_cursor: [usize; RAMP_STAGES],
    /// Passes completed, i.e. epochs whose partition was issued in full and audited.
    completed_passes: usize,
    /// The most recently completed pass's reconciliation, so the epoch report and the final
    /// banner state measured coverage rather than a fraction recomputed from a different
    /// source. `None` until the first pass completes.
    audit: Option<CoverageAudit>,
    /// Cross-pass exposure history. `audit` above answers "did THIS pass cover the corpus";
    /// this answers "how many times has the RUN shown the model each bar", which no per-pass
    /// audit can express — `require_full_pass` pins within-pass multiplicity to one, so the
    /// per-pass histogram reads a single spike at one on every epoch of a three-epoch run.
    /// Absorbs one hole start per symbol per completed pass and nothing else, so it is exact
    /// and bounded at ~21 KB per pass rather than one byte per training bar.
    census: PassCensus,
    bars_seen: u64,
    epoch: usize,
    best_val_nll_bar: f64,
    /// Best conditional held-out NLL seen so far, which is what the RIVAL, NLL-primary rule
    /// selects on. It decides [`NLL_RULE_CHECKPOINT`] and nothing else: the artifact the
    /// planner loads is chosen economically. Still tracked and charted so the campaign stays
    /// comparable to every run scored before the rule was inverted.
    best_val_nll_bar_conditional: f64,
    /// Per-window vector of the checkpoint the ECONOMIC rule has promoted, which is what both
    /// non-regression guards pair a candidate against.
    best_scores: Option<WindowScores>,
    /// Mean per-window edge at [`SELECTION_CAP`], in bps/bar, of the promoted checkpoint —
    /// the value a candidate has to beat. `NEG_INFINITY` until the first promotion, so the
    /// first eligible read promotes on eligibility alone.
    best_selection_edge_bps: f64,
    /// Its per-window vector, which is what makes the candidate comparison PAIRED. Without it
    /// the rule could only difference two levels, whose interval is ~+/-1.6 bps and would
    /// resolve nothing at the 0.02 bps scale the decision lives at.
    best_selection_edge_windows: Option<Vec<f64>>,
    /// Conditional NLL of the promoted checkpoint on the selection pass, recorded so the
    /// promotion line and the checkpoint metadata can both state the trade-off the economic
    /// choice bought or paid for.
    best_selection_nll: f64,
    /// Global step the promoted artifact's weights are from, so the terminal rule comparison
    /// names two steps rather than two file names.
    promoted_step: usize,
    /// Per-window vector of the RIVAL, NLL-selected artifact, paired against by its own guard
    /// so the two rules run under exactly the discipline each one claims.
    nll_rule_scores: Option<WindowScores>,
    /// Step, promotion count and economic reading of the rival artifact, for the terminal
    /// comparison. Zero and `NAN` until it first writes.
    nll_rule_step: usize,
    nll_rule_promotions: usize,
    nll_rule_edge_bps: f64,
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
    /// Auxiliary-resolution training streams, empty unless `--auxiliary-resolutions` named one.
    ///
    /// Each holds its OWN supports, ramp contexts, pass partition and coverage ledger. An
    /// auxiliary step is a whole step drawn from one resolution and scored against that
    /// resolution's bins, never a batch with two timeframes in it.
    aux: Vec<AuxiliaryStream>,
    /// Index-aligned with [`Self::aux`]: that resolution's pinned val windows at
    /// [`AUXILIARY_HELDOUT_CONTEXT`]. Separate from `eval` because the deployment sets are
    /// drawn from the deployment corpus and address different files entirely.
    aux_heldout: Vec<PinnedSet>,
    aux_report: AuxiliaryReport,
    /// Auxiliary optimizer steps taken, ADDITIVE to the schedule's own count. The learning-rate
    /// and momentum schedules are indexed by the PRIMARY step, so an auxiliary step runs at the
    /// lr of the primary step it follows: progress through the deployment corpus is what the
    /// schedule is a function of, and the auxiliary stream must not stretch or compress it.
    aux_steps: usize,
    aux_bars_seen: u64,
    /// Additive constant `--scoring` puts into `nll_bar` that no prediction can move:
    /// `BarSupports::log_measure_bar` under the density rule, zero otherwise. Subtracted
    /// before the loss-term shares are formed so [`AUX_SHARE_WARN`] means the same thing
    /// under every rule.
    share_scale_offset: f64,
    /// Device memory in use before the first optimizer step: the weights, the CUDA context
    /// and whatever the card's other tenants already held. Subtracted from a later reading
    /// to attribute the remainder to activations.
    ///
    /// The optimizer's momentum buffers are allocated lazily on the first step and are
    /// therefore counted as "activations", which OVERSTATES the per-token footprint. That
    /// biases the ramp toward holding, which is the only safe direction on a shared card.
    vram_baseline_bytes: Option<u64>,
    /// Measured device bytes per bar-token. Seeded from [`probe_capacity`] before step 0 and
    /// refreshed by [`Trainer::probe_activation_footprint`] once per ramp stage, so the
    /// runtime guard is armed at the FIRST stage transition rather than only after it.
    activation_bytes_per_token: Option<f64>,
    /// The startup capacity measurement the ramp was derived from. `None` off CUDA or without
    /// NVML, in which case the run keeps the declared [`BATCH_RAMP`] under the runtime hold.
    capacity: Option<CapacityModel>,
    /// The ramp [`CapacityModel::derive_batch_ramp`] planned, kept beside
    /// `schedule.batch_ramp` — which a runtime hold mutates — so the final banner can
    /// separate "the card could never do this" from "contention took it away mid-run".
    derived_batch_ramp: [usize; RAMP_STAGES],
    /// `--batch-size` as the operator asked for it, before the startup capacity clamp.
    requested_batch: usize,
    /// Optimizer steps taken inside the current ramp stage.
    stage_step: usize,
    /// Longest context an optimizer step actually ran at.
    ///
    /// The batch ramp is memory-gated on a shared card, so what the run DID is not a function
    /// of the flags. Promotion is gated on this rather than on the stage index alone, and it is
    /// recorded in every checkpoint: a model selected below the deployed context is a
    /// legitimate artifact but not an interchangeable one.
    reached_context: i64,
    /// Context of the held-out set the currently promoted checkpoint was selected on. Zero
    /// until the first promotion.
    selection_context: i64,
    /// Best conditional held-out NLL seen at EACH evaluation context, not only at the deployed
    /// one. The deployed-context entry is what `pretrain_best.ot` holds and the only one the
    /// planner may load; the diagnostic-context entry is what makes a run that never reached
    /// the deployed context still have a defensible best rather than a last-step snapshot.
    best_by_context: BTreeMap<i64, f64>,
    /// The diagnostic-context best artifact, once one exists.
    diagnostic_best: Option<PathBuf>,
    /// Wall clock and bar-token count at the START of the epoch now in progress.
    ///
    /// An epoch's cost is measured as a delta between two boundaries rather than divided
    /// out of the run total: the ramp changes the per-step token count and the boundary
    /// work itself is not free, so a division would attribute both to the wrong epoch.
    epoch_started: Instant,
    epoch_start_bars: u64,
    /// `dyn / identity` summed over the optimizer steps of the epoch now in progress. The
    /// step series carries it per tick; the epoch line needs the epoch's own mean, which
    /// the validation-interval accumulators cannot give because they reset far more often.
    epoch_dyn_identity_sum: f64,
    epoch_dyn_identity_steps: usize,
    /// Identity of the pinned snapshot windows, taken once before the first boundary.
    ///
    /// The epoch-over-epoch candle comparison is a comparison of one model against another
    /// ON ONE SCENE. If the scene moves, a tightening fan is indistinguishable from an
    /// easier window, and the whole picture series becomes unreadable without anybody being
    /// told. So the scene is fingerprinted and every boundary re-checks it.
    snapshot_window_fingerprint: u64,
}

/// One optimizer step's losses, already reduced to host scalars.
struct StepLoss {
    nll_bar: f64,
    nll_dof: [f64; BAR_DOF],
    dyn_loss: f64,
    kl_loss: f64,
    /// Mean `-log(1 + f_hat R)` in nats per bar under the deployed leverage cap. Reported
    /// whatever `--lambda-growth` is, so the ablation's two arms are comparable.
    growth_loss: f64,
    /// The growth term's detached diagnostics: mean `|f_hat|`, the fraction of bars where
    /// the cap chose the size, and the smallest log argument seen.
    growth_stats: growth::GrowthStats,
    total: f64,
    /// Share of the objective's total MAGNITUDE carried by each weighted term, in
    /// `(nll, dyn, kl, growth)` order. They sum to one.
    shares: (f64, f64, f64, f64),
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

/// What a promotion decision measured, carried into the artifact it promoted.
///
/// BOTH numbers, always, in both directions. A promotion that bought edge at the cost of
/// density has to say so on the file itself, and one that improved both has to say that too,
/// so a reader holding the checkpoint can see the trade-off without the run's log.
#[derive(Clone, Copy, Debug)]
struct SelectionRecord {
    step: usize,
    /// Context, in bars, the two criteria were measured at.
    bench_context: i64,
    /// Mean per-window edge over the unconditional-marginal null at [`SELECTION_CAP`], in bps
    /// per bar, and the block-bootstrap standard error of that level.
    edge_bps: f64,
    edge_se_bps: f64,
    /// Conditional NLL of the same pass, in nats per bar.
    nll_conditional: f64,
}

/// What an eligible read DECIDED, for the promotion ledger chart.
///
/// A refusal is the interesting half — the rule exists to refuse — so each reason is a
/// separate value with its own cumulative series rather than a single "did not promote".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionOutcome {
    /// The ramp had not reached the deployed context, so no decision was taken at all. This
    /// is not a refusal and must never read as one.
    NotEligible,
    Promoted,
    /// The paired economic gain did not clear the noise band.
    RefusedInsideNoise,
    /// The conditional density regressed beyond tolerance.
    RefusedNllGuard,
    /// The traded factor regressed beyond tolerance.
    RefusedDofGuard,
    /// The bench produced nothing comparable to the incumbent, so the criterion was unmeasured
    /// at this read. Distinct from a refusal on the merits.
    Unmeasurable,
}

/// One eligible read's full promotion ledger: both criteria, both incumbents, the thresholds
/// actually applied and the decision. Charted on `pretrain_promotions` so the trade-off is
/// readable off one panel instead of reconstructed from the log.
#[derive(Clone, Copy, Debug)]
pub struct SelectionLedger {
    pub outcome: SelectionOutcome,
    /// Context both criteria were measured at.
    pub bench_context: i64,
    pub edge_bps: f64,
    pub edge_se_bps: f64,
    /// Paired gain against the incumbent and its standard error, plus the band it had to
    /// clear. All three in bps/bar; the band IS the number the decision compared against.
    pub edge_gain_bps: f64,
    pub edge_gain_se_bps: f64,
    pub edge_band_bps: f64,
    pub incumbent_edge_bps: f64,
    /// Traded notional per bar at [`SELECTION_CAP`], so the cost of collecting the edge is
    /// visible at the point of decision. Reported only; it never enters the decision.
    ///
    /// ABSOLUTE weight units, not a multiple of gross exposure. At a 0.25x cap a reading of
    /// 0.25 is ONE full rotation of the book per bar, so `turnover / SELECTION_CAP` is the
    /// rotations/bar a reader actually wants and is what the chart plots. Stating the unit
    /// here because reading this field as a multiple of gross understates the rotation rate by
    /// exactly the leverage factor, which is how a book rotating every 1.1 bars gets described
    /// as rotating every third one.
    pub turnover: f64,
    /// MEASURED mean gross exposure per bar at [`SELECTION_CAP`], i.e. mean `|position|`.
    ///
    /// Not the nominal cap. At 0.25x roughly 99% of bars are clamped, so the two agree to ~1%,
    /// but they diverge whenever a bar's signal was degenerate and carried no exposure at all.
    /// The ratio below is taken against this rather than against [`SELECTION_CAP`] so it stays
    /// a measurement in both cases instead of silently becoming an assumption in one.
    pub gross_exposure: f64,
    /// [`Self::turnover`] divided by [`Self::gross_exposure`]: book rotations per bar, where
    /// 1.0 is one full rotation. The number a reader wants, since absolute turnover understates
    /// the rotation rate by exactly the leverage factor.
    pub rotations: f64,
    pub nll_conditional: f64,
    /// Paired conditional-NLL difference against the incumbent, POSITIVE when the candidate is
    /// worse, beside the tolerance the guard allowed.
    pub nll_delta: f64,
    pub nll_tolerance: f64,
    pub incumbent_nll: f64,
    /// Paired difference of the guarded factor, same sign convention.
    pub dof_delta: f64,
}

impl SelectionLedger {
    /// A read that took no decision. Every measurement is `NaN` rather than zero: a zero edge
    /// gain is a finding and must not be confused with one that was never measured.
    pub fn unmeasured() -> Self {
        Self {
            outcome: SelectionOutcome::NotEligible,
            bench_context: 0,
            edge_bps: f64::NAN,
            edge_se_bps: f64::NAN,
            edge_gain_bps: f64::NAN,
            edge_gain_se_bps: f64::NAN,
            edge_band_bps: f64::NAN,
            incumbent_edge_bps: f64::NAN,
            turnover: f64::NAN,
            gross_exposure: f64::NAN,
            rotations: f64::NAN,
            nll_conditional: f64::NAN,
            nll_delta: f64::NAN,
            nll_tolerance: f64::NAN,
            incumbent_nll: f64::NAN,
            dof_delta: f64::NAN,
        }
    }
}

/// THE promotion rule, as a pure function of what the bench and the density measured.
///
/// Extracted from the validation loop so the rule can be exercised without a Trainer, a
/// corpus or a card. A selection rule whose refusals can only be observed by completing a
/// 31k-step run is a rule whose refusals are asserted rather than demonstrated, and this one
/// exists primarily to REFUSE: on the run that motivated it, the honest verdict over twelve
/// eligible reads is "nothing after the first was measurably better".
///
/// The order is the rule and is not arbitrary.
///
/// 1. `first` short-circuits: there is no incumbent, so nothing is comparable and a run with
///    no artifact has produced nothing anybody can act on.
/// 2. Measurability, because an unmeasured criterion is not a verdict against a candidate.
///    The incumbent stands and the read is recorded as unmeasured, not as a refusal.
/// 3. The economic band. A candidate inside it is UNRESOLVED, not better, and the guards have
///    nothing to protect against a promotion that is not going to happen anyway.
/// 4. The two density guards, aggregate before per-factor. Both are non-regression vetoes, so
///    neither can ever cause a promotion — only block one — which is what keeps the rule
///    economic while still refusing a model whose density has genuinely broken.
///
/// Every comparison is `>` against a threshold derived from the PAIRED standard error of the
/// same quantity, so a NaN measurement fails the band and passes the guards: an unresolved
/// candidate does not displace an incumbent, and an unmeasured guard does not veto on
/// evidence it does not have.
fn selection_outcome(
    first: bool,
    candidate_edge: &[f64],
    edge_gain: Option<Dispersion>,
    nll_guard: Option<Dispersion>,
    dof_guard: Option<Dispersion>,
) -> SelectionOutcome {
    if first {
        return SelectionOutcome::Promoted;
    }
    let Some(gain) = edge_gain.filter(|_| !candidate_edge.is_empty()) else {
        return SelectionOutcome::Unmeasurable;
    };
    if !(gain.mean > SELECTION_EDGE_SE_MULTIPLE * gain.se.max(0.0)) {
        return SelectionOutcome::RefusedInsideNoise;
    }
    if nll_guard
        .is_some_and(|delta| delta.mean > SELECTION_NLL_TOLERANCE_SE_MULTIPLE * delta.se.max(0.0))
    {
        return SelectionOutcome::RefusedNllGuard;
    }
    if dof_guard.is_some_and(|delta| delta.mean > SELECTION_GUARD_SE_MULTIPLE * delta.se.max(0.0)) {
        return SelectionOutcome::RefusedDofGuard;
    }
    SelectionOutcome::Promoted
}

/// An incumbent economic reading, or a phrase saying there is not one yet. Printed rather
/// than a bare `-inf`, which reads like a measurement.
fn fmt_incumbent_bps(value: f64) -> String {
    if value.is_finite() {
        format!("{value:+.4} bps/bar")
    } else {
        "none (first eligible read)".to_owned()
    }
}

/// The same for an incumbent NLL level.
fn fmt_incumbent_nats(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.4} nats/bar")
    } else {
        "none (first eligible read)".to_owned()
    }
}

impl Trainer {
    /// Consumes the trainer: `PretrainReporter::finish` takes the reporter by value,
    /// which makes reporting a promotion after the terminal battery a compile error.
    fn run_training(mut self) -> Result<()> {
        let started = Instant::now();
        let mut last_stage = usize::MAX;
        // Everything the card already holds before a single activation is allocated: the
        // weights, the gradients, the CUDA context, and whatever the other tenants of a
        // shared GPU are holding. The ramp's headroom test is measured against this.
        //
        // The pool is RELEASED first. Support fitting and the marginal's encode pass both ran
        // at the deployed context before this line, so without the release their cached blocks
        // sit inside the reading and get attributed to the other tenants, which under-reports
        // free memory and makes the first stage probe under-attribute its activations.
        crate::torch::cuda::empty_cache();
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
                // Cursors are per stage and per epoch, and the stage resumes where the
                // partition left it. It is never reset here: with `--epochs 1` a stage is
                // entered once, and with more the boundary handler resets all three together
                // because a new epoch is a new partition.
                last_stage = stage;
                self.stage_step = 0;
                println!(
                    "step {step}: ramp stage {stage} — batch {} (x{} of the base {}, derived \
                     ceiling x{} of the declared x{}), context {}, lr plateau x{:.3}",
                    self.schedule.batch(step),
                    self.schedule.batch_ramp[stage],
                    self.schedule.base_batch,
                    self.derived_batch_ramp[stage],
                    BATCH_RAMP[stage],
                    self.schedule.context(step),
                    // The schedule's own bump, at the stage's reference exponent — `sqrt` here
                    // printed 1.414x where the schedule applied 1.516x at the 2x step-up.
                    self.schedule.lr_multiplier_for(0, self.schedule.batch_ramp[stage]),
                );
            }

            let planned_batch = self.schedule.batch(step);
            let cursor = self.stage_cursor[stage];
            let (refs, sample) = {
                // The last draw of a stage is SHORT, not dropped, so the pass covers the
                // partial tail of the stage's share instead of leaving up to `batch - 1`
                // windows untargeted.
                let refs = self.pass_layout.draw(stage, cursor, planned_batch).to_vec();
                ensure!(
                    !refs.is_empty(),
                    "step {step} has no window left in ramp stage {stage}'s share of epoch {}: \
                     the cursor is at {cursor} of {} assigned windows. The schedule is derived \
                     from the partition, so this can only happen if the two disagree.",
                    self.epoch,
                    self.pass_layout.windows(stage).len()
                );
                let sample = self.train_samplers[stage].batch_of(&refs, self.device);
                (refs, sample)
            };
            // Marked from the DRAW, not from the cursor that produced it, so a skip or a
            // repeat shows up in the audit as a zero or a two rather than being masked by the
            // counter that caused it.
            let batch = refs.len();
            self.pass_ledger.mark(stage, cursor, batch);
            self.stage_cursor[stage] = cursor + batch;

            let lr_mult = self.schedule.lr_multiplier(step);
            self.optimizer.set_lr(NORMUON_LR * lr_mult);
            self.optimizer.set_adamw_lr(ADAMW_LR * lr_mult);
            let momentum = self.schedule.momentum(step);
            self.optimizer.set_momentum(momentum);

            let loss = self.optimizer_step(&sample, step, None)?;
            // The auxiliary steps for this primary step, at the SAME lr and momentum. Fired
            // after the primary update rather than before it so the primary step of a given
            // index is byte-identical in ordering to a run with no auxiliary stream up to the
            // shared parameter state, which is what makes the A/B a single-variable comparison.
            self.auxiliary_steps(step)?;
            // The ACTUAL windows drawn, not the planned batch: the last step of every stage is
            // short, and counting it as full would overstate the pass by up to one batch per
            // stage and make the bar-token ledger disagree with the coverage ledger.
            let bar_tokens = batch as u64 * self.schedule.context(step) as u64;
            self.bars_seen += bar_tokens;
            // AFTER the step, not at the stage banner: this is the longest context the run has
            // actually optimized at, which is what promotion is allowed to be measured at.
            self.reached_context = self.reached_context.max(self.schedule.context(step));

            self.train_nll_sum += loss.nll_bar;
            for (acc, value) in self.train_nll_dof_sum.iter_mut().zip(loss.nll_dof) {
                *acc += value;
            }
            self.train_steps += 1;
            // Epoch-scoped, so the boundary line reports the mean over the pass rather than
            // whatever the last minibatch happened to give. Non-finite steps are skipped:
            // `dyn_vs_identity` is NaN when the identity baseline is degenerate, and one
            // such step would poison the whole epoch's mean.
            if loss.dyn_vs_identity.is_finite() {
                self.epoch_dyn_identity_sum += loss.dyn_vs_identity;
                self.epoch_dyn_identity_steps += 1;
            }
            self.stage_step += 1;
            if self.stage_step == RAMP_PROBE_AFTER_STEPS {
                self.probe_activation_footprint(step);
            }

            let (nll_share, dyn_share, kl_share, growth_share) = loss.shares;
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
            metrics.growth_loss = loss.growth_loss;
            metrics.growth_share = growth_share;
            metrics.growth_abs_f = loss.growth_stats.mean_abs_f;
            metrics.growth_clamp_bind = loss.growth_stats.clamp_bind;
            metrics.belief_autocorr = loss.belief_autocorr;
            metrics.dyn_vs_identity = loss.dyn_vs_identity;
            metrics.lr_mult = lr_mult;
            metrics.muon_momentum = momentum;
            metrics.grad_norm = loss.grad_norm;
            metrics.context = self.schedule.context(step);
            metrics.batch_size = batch;
            metrics.bars_seen = self.bars_seen;
            // The capacity panel: what the card had free, what the step was projected to
            // cost, and the ceiling the plan was derived against. NVML's `memory_info` is a
            // driver ioctl in the tens of microseconds, invisible beside an optimizer step at
            // ~4.5 step/s, and reading it EVERY step is what makes a contention event visible
            // as a dip rather than as an OOM.
            metrics.free_vram_gib = device_free_bytes(self.device)
                .map_or(f64::NAN, |free| CapacityModel::gib(free as f64));
            metrics.bar_tokens = bar_tokens as f64;
            if let Some(capacity) = self.capacity.as_ref() {
                metrics.projected_footprint_gib = CapacityModel::gib(
                    capacity.step_bytes(batch, self.schedule.context(step)),
                );
                metrics.capacity_ceiling_gib = CapacityModel::gib(
                    capacity.free_bytes as f64 - RAMP_MEMORY_RESERVE_BYTES as f64,
                );
            }
            // Counted over the batch's own slots rather than `bar_tokens`: a window carries
            // `context + 1` bars and the ledger above counts the `context` TARGETS, so reusing
            // it here would understate coverage by one bar per window.
            let span = sample.dof.size();
            metrics.market_missing_bars = sample.market_missing as u64;
            metrics.market_total_bars = (span[0] * span[1]) as u64;
            self.reporter.record_step(&metrics)?;
            self.warn_on_auxiliary_domination(step, dyn_share, kl_share, growth_share);

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
                     | kl {:.4} x{:e} ({:.0}%) | growth {:+.3e} x{:e} ({:.0}%) |f| {:.2} \
                     cap {:.0}% | total {:.4} | autocorr {:.3} | dyn/identity \
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
                    // Scientific notation: the whole tradeable content of the `r` prediction
                    // is 5.25e-4 nats/bar, so three fixed decimals would print this term as
                    // `-0.000` for the entire run.
                    loss.growth_loss,
                    self.args.lambda_growth,
                    100.0 * growth_share,
                    loss.growth_stats.mean_abs_f,
                    100.0 * loss.growth_stats.clamp_bind,
                    loss.total,
                    loss.belief_autocorr,
                    loss.dyn_vs_identity,
                    loss.grad_norm,
                    (step + 1) as f64 / elapsed.max(1e-9)
                );
            }

            // An epoch is a PASS, and a pass is a fixed partition of the corpus, so the
            // boundary is a fixed step index — not `bars_seen / train_bars`, which was a
            // throughput ratio being read as a coverage one and which crossed one whole
            // `train_bars` while 28.7% of the corpus had never been a prediction target.
            let epoch_boundary = self.schedule.completes_epoch(step);
            let periodic = self.args.validate_every > 0
                && step > 0
                && step % self.args.validate_every == 0;
            let final_step = step + 1 == self.schedule.total_steps;

            // Crash insurance, independent of promotion and of the ramp. Promotion is gated on
            // the deployed context, so a memory-held run can go tens of thousands of steps
            // without producing a `pretrain_best.ot` — job 2856 went 9221 steps, two epoch
            // boundaries and nine validations, before its first one. A step-tagged checkpoint
            // every `checkpoint_every` steps bounds what an OOM or a power cut can destroy;
            // the window is pruned so disk does not grow with the run.
            if self.args.checkpoint_every > 0
                && step > 0
                && step % self.args.checkpoint_every == 0
            {
                self.write_step_artifacts(step)?;
            }

            if epoch_boundary {
                // BEFORE validate, so a pass that did not cover the split dies before it
                // writes an epoch artifact and a held-out number that claim it did.
                self.audit = Some(self.finish_pass(step)?);
            }
            if epoch_boundary || periodic || final_step {
                self.validate(step, epoch_boundary, final_step)?;
                if epoch_boundary {
                    self.begin_pass(self.epoch + 1);
                }
            }
        }

        // `pretrain_last.ot` means the weights after the FINAL optimizer step, and this is the
        // only place that can honestly say so. It used to be written inside `validate`, i.e.
        // only at validation boundaries, which quietly made it the last VALIDATED step: job
        // 2884 was killed by SIGTERM at step 30780 of 31095 with its newest validation at step
        // 30000, so the file held step-30000 weights — the same weights the step-30000
        // promotion had just written into `pretrain_best.ot`, which from the outside read as
        // the two files being aliases of each other. Written here, and every
        // `checkpoint_every` steps by `write_step_artifacts`, it is never staler than the
        // newest `pretrain_step_*.ot` and a completed run's copy is exactly the last step.
        if self.schedule.total_steps > 0 {
            let final_step = self.schedule.total_steps - 1;
            let last = self.write_last_checkpoint(final_step)?;
            println!(
                "pretrain finished: {} holds the weights after the FINAL optimizer step \
                 {final_step}, recorded as `global_step` in its metadata. It carries NO \
                 held-out scores and its `selection_context` is 0 — no decision chose it, so \
                 any number measured from it belongs to a read nobody reported. \
                 `pretrain_best.ot` is the artifact a decision chose.",
                last.display(),
            );
        }

        let elapsed = started.elapsed().as_secs_f64();
        println!(
            "pretrain finished: {} steps in {elapsed:.1}s ({:.2} step/s), {} promotions on the \
             0.25x-cap edge criterion, promoted edge {:+.4} bps/bar with conditional nll \
             {:.4} nats/bar; the promoted artifact's deployed-context held-out nll is {:.4} \
             nats/bar under {} scoring ({:+.4} vs the calibrated marginal {:.4}, {:+.4} vs \
             uniform {:.4})",
            self.schedule.total_steps,
            self.schedule.total_steps as f64 / elapsed.max(1e-9),
            self.promotions,
            self.best_selection_edge_bps,
            self.best_selection_nll,
            self.best_val_nll_bar,
            self.args.scoring,
            self.marginal_nll_bar - self.best_val_nll_bar,
            self.marginal_nll_bar,
            self.baselines.uniform_nll_bar - self.best_val_nll_bar,
            self.baselines.uniform_nll_bar,
        );
        // The context the artifact was SELECTED at, always, not only when it disagrees: a
        // number quoted without it is not comparable to anything.
        let deployed = self.eval.promotion.context;
        println!(
            "pretrain finished: selection context {} bars, longest context trained {} bars, \
             deployed context {} bars, batch ramp declared x{:?} -> derived from measured \
             capacity x{:?} -> realized x{:?} at base {} of the {} requested{}",
            self.selection_context,
            self.reached_context,
            deployed,
            BATCH_RAMP,
            self.derived_batch_ramp,
            self.schedule.batch_ramp,
            self.schedule.base_batch,
            self.requested_batch,
            if self.selection_context == deployed {
                ""
            } else {
                " — CAVEAT: this checkpoint was NOT selected at the deployed context. It is \
                 evaluable and loadable, but a planner running it at the deployed context is \
                 running it outside the positional range it was selected in."
            },
        );
        // `total_steps` is sized from the DERIVED ramp, so an on-plan run delivers the tokens
        // it asked for. A runtime hold is the one case that still under-delivers, and it must
        // be stated as a number rather than left for a reader to infer from a reuse chart:
        // job 2856's two holds turned `--epochs 3` into 1.33 epochs at the same step count.
        if self.schedule.batch_ramp != self.derived_batch_ramp {
            let planned: u64 = (0..self.schedule.total_steps)
                .map(|step| {
                    let stage = self.schedule.stage(step);
                    (self.schedule.base_batch * self.derived_batch_ramp[stage]) as u64
                        * self.schedule.context(step) as u64
                })
                .sum();
            println!(
                "pretrain finished: WARNING contention held the batch below the derived plan, so \
                 this run delivered {} bar-tokens against the {planned} its {} steps were sized \
                 for — {:.0}% of the requested --epochs {}, i.e. {:.2} effective epochs over \
                 {} bar-tokens in one pass. The step count is deliberately NOT extended: a run \
                 that silently ran longer would not be comparable to its own siblings.",
                self.bars_seen,
                self.schedule.total_steps,
                100.0 * self.bars_seen as f64 / planned.max(1) as f64,
                self.args.epochs,
                self.bars_seen as f64 / self.full_pass_bar_tokens() as f64,
                self.full_pass_bar_tokens(),
            );
        }
        // The pass ledger, not a ratio: how many epochs actually covered the training split.
        // A batch held by contention makes a stage run out of steps before it has issued its
        // share, which is the one way a run can still under-deliver a pass, so it is stated as
        // a count of completed passes and it is FATAL. A run that trained on part of its corpus
        // while its reports are indexed by epoch is not comparable to its siblings, and the
        // exit code is the only thing a campaign script reads.
        let audit = self
            .audit
            .as_ref()
            .map_or_else(String::new, |audit| format!(" Last pass: {}", audit.summary()));
        println!(
            "pretrain finished: {} of the {} requested passes over the training split completed \
             and audited, {} bars per pass targeted exactly once out of {} in the split.{audit}",
            self.completed_passes,
            self.args.epochs,
            self.pass.covered_bars(),
            self.pass.split_bars(),
        );
        if self.args.steps.is_none() {
            ensure!(
                self.completed_passes >= self.args.epochs,
                "this run completed {} of the {} passes --epochs asked for. An epoch is one pass \
                 over every training bar, so a missing pass means the corpus was NOT covered the \
                 number of times the run claims and its numbers are not comparable to a run that \
                 was. The usual cause is a VRAM hold lowering a stage's batch after the step \
                 count was derived, which leaves that stage short of steps for its share; the \
                 hold is announced above. Re-run when the card is free, or pass --steps to \
                 declare the decoupling deliberate.",
                self.completed_passes,
                self.args.epochs
            );
        }
        // Every context that was ever measured, because they are not comparable to each other
        // and a single "best" hides which problem it was best at. This is also the line that
        // proves a memory-held run still left a defensible selection behind.
        let per_context: Vec<String> = self
            .best_by_context
            .iter()
            .map(|(context, best)| format!("{context} bars: {best:.4}"))
            .collect();
        println!(
            "pretrain finished: best conditional held-out nll per evaluation context [{}]. NLL is \
             the GUARD, not the criterion: the planner loads `pretrain_best.ot`, chosen on the \
             0.25x-cap trade edge, and the NLL-best of the eligible reads is {}{}",
            per_context.join(", "),
            if self.nll_rule_promotions > 0 {
                format!(
                    "{NLL_RULE_CHECKPOINT} from step {} ({} rival promotions), kept for the \
                     test-split comparison and never loaded by the planner",
                    self.nll_rule_step, self.nll_rule_promotions,
                )
            } else {
                "not written: no eligible read ever improved on the first one's conditional nll"
                    .to_owned()
            },
            self.diagnostic_best.as_ref().map_or_else(
                || ". No fixed-context best was ever written.".to_owned(),
                |path| format!(
                    ". {} is the NLL best at the fixed {}-bar diagnostic context and is NOT \
                     interchangeable with either.",
                    path.display(),
                    self.eval.diagnostic.context
                )
            ),
        );
        ensure!(
            self.promotions > 0,
            "no checkpoint was ever promoted; there is nothing for the planner to load"
        );

        let (battery, dyn_identity) = self.test_battery()?;
        // Finalize the report BEFORE the verdict. A run whose dynamics head failed the
        // guard is exactly the run someone needs the full report for, and a hard failure
        // that also destroys `.report.bin` would be the second-worst outcome after shipping
        // the head silently.
        self.reporter.finish(&battery)?;
        check_dynamics_beats_identity(dyn_identity, self.args.dyn_horizon as i64)
    }

    /// Score the promoted checkpoint on the TEST split, exactly once, at the very end.
    /// The model is reloaded from disk rather than read out of memory so the reported
    /// numbers provably belong to the artifact the planner will load.
    fn test_battery(&self) -> Result<(TestBattery, f64)> {
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

        // The test split is scored at the context the checkpoint was SELECTED at. Scoring a
        // model that never trained past 896 bars on 2048-bar windows measures positional
        // extrapolation, not generalization, and it would be the one number in the run nobody
        // could interpret.
        let set = self.test_set();
        let stats = evaluate(
            world.modules(),
            world.deployment_supports(),
            set,
            self.args.batch_size,
            self.device,
            true,
            self.args.scoring,
            None,
            trade_bench::TRADE_WINDOWS,
        )?;
        let dispersion = self.dispersion(set, &stats);
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
        let dyn_identity =
            self.measure_dynamics_versus_identity(world.modules(), world.deployment_supports(), set)?;

        println!(
            "test split ({} windows at context {}, {} scoring): nll {} nats/bar, {:+.4} vs the \
             calibrated marginal {:.4}, {:+.4} vs uniform {:.4}; rollout h1 {:.4} exact / \
             {:.4} dynamics",
            set.windows.len(),
            set.context,
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
        let teacher: f64 = stats.forecast_teacher_nll_dof.iter().sum();
        let forecast: f64 = stats.forecast_nll_dof.iter().sum();
        println!(
            "test split FORECAST-ONLY nll {forecast:.4} +/- {:.4} nats/bar vs TEACHER-FORCED \
             {teacher:.4} on identical rows: teacher-forcing is {:.4} nats/bar OPTIMISTIC. The \
             forecast figure is the headline forecasting number — every factor conditions on \
             strictly past bars; the teacher-forced figure is the joint bar likelihood, kept for \
             comparability with every earlier run.",
            stats.forecast_nll_se,
            forecast - teacher,
        );
        println!(
            "test split selection context {} bars (deployed {}, longest trained {})",
            self.selection_context, self.eval.promotion.context, self.reached_context,
        );

        let trade = self.trade(set, &stats);
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
        battery.forecast_nll_dof = stats.forecast_nll_dof;
        battery.forecast_teacher_nll_dof = stats.forecast_teacher_nll_dof;
        battery.forecast_nll_se = stats.forecast_nll_se;
        battery.selection_context = self.selection_context;
        battery.deployed_context = self.eval.promotion.context;
        battery.reached_context = self.reached_context;
        battery.lr_plateau_fraction = self.schedule.lr_plateau_fraction;
        battery.trade = trade;
        for line in battery.trade.report_lines() {
            println!("test split {line}");
        }
        // The comparison that makes the inverted rule evidence rather than an assertion: the
        // artifact the PREVIOUS rule would have shipped, scored on the SAME test set, in the
        // same pass shape, at the same context.
        let rival = self.nll_rule_battery(set, &battery)?;
        battery.nll_rule = rival;
        Ok((battery, dyn_identity))
    }

    /// Score [`NLL_RULE_CHECKPOINT`] on the test split beside the promoted artifact.
    ///
    /// `None` when the rival rule never wrote a file, which happens exactly when the first
    /// eligible read held the best conditional NLL of the run: then both rules chose the same
    /// weights and there is nothing to compare. That is a finding, and the terminal line says
    /// so rather than leaving a silent gap.
    ///
    /// The two artifacts are scored on ONE set at ONE context and their difference is reported
    /// in both currencies, so the trade-off the economic rule accepted is measured on data that
    /// fed neither rule's decision. This is the only place the test split is allowed to see two
    /// models, and it still sees each exactly once.
    fn nll_rule_battery(
        &self,
        set: &PinnedSet,
        promoted: &TestBattery,
    ) -> Result<Option<RivalSelection>> {
        let checkpoint = self.run.weights.join(NLL_RULE_CHECKPOINT);
        if !checkpoint.exists() {
            println!(
                "test split: the NLL rule wrote no rival artifact — no eligible read improved on \
                 the first one's conditional nll, so both rules selected the same weights and \
                 there is nothing to compare. The economic rule cost nothing here."
            );
            return Ok(None);
        }
        let metadata = world_model_metadata_path(&checkpoint);
        let world = BarWorldModel::load(&checkpoint, &metadata, self.device).with_context(|| {
            format!(
                "the NLL-selected rival {} could not be reloaded for the test comparison",
                checkpoint.display()
            )
        })?;
        let stats = evaluate(
            world.modules(),
            world.deployment_supports(),
            set,
            self.args.batch_size,
            self.device,
            true,
            self.args.scoring,
            None,
            trade_bench::TRADE_WINDOWS,
        )?;
        let trade = self.trade(set, &stats);
        let rival = RivalSelection {
            checkpoint: checkpoint.clone(),
            model_lineage: world.lineage_sha256().to_owned(),
            step: self.nll_rule_step,
            nll_bar_conditional: stats.nll_bar_conditional,
            nll_dof: stats.nll_dof,
            // `CapPoint::edge` is net log growth per bar; the criterion is bps, as everywhere.
            selection_edge_bps: trade.cap_curve[SELECTION_CAP_SLOT].edge * 1.0e4,
            edge_at_default: trade.model_edge().mean * 1.0e4,
            sharpe: trade.policies[trade_bench::POLICY_QUARTER].sharpe,
        };
        let promoted_edge = promoted.trade.cap_curve[SELECTION_CAP_SLOT].edge * 1.0e4;
        println!(
            "test split RULE COMPARISON on {} windows at context {}: ECONOMIC pick (step {}) \
             edge@{SELECTION_CAP:.2}x {promoted_edge:+.4} bps/bar, 4x edge {:+.4}, \
             quarter-Kelly sharpe {:+.2}, conditional nll {:.4}; NLL pick (step {}) \
             edge@{SELECTION_CAP:.2}x {:+.4}, 4x edge {:+.4}, quarter-Kelly sharpe {:+.2}, \
             conditional nll {:.4}. The economic rule bought {:+.4} bps/bar of 0.25x edge for \
             {:+.4} nats of conditional nll (negative means it cost nothing and gained on both). \
             Neither number fed either decision: this is the split that was touched once.",
            set.windows.len(),
            set.context,
            self.promoted_step,
            promoted.trade.model_edge().mean * 1.0e4,
            promoted.trade.policies[trade_bench::POLICY_QUARTER].sharpe,
            promoted.nll_bar_conditional,
            rival.step,
            rival.selection_edge_bps,
            rival.edge_at_default,
            rival.sharpe,
            rival.nll_bar_conditional,
            promoted_edge - rival.selection_edge_bps,
            promoted.nll_bar_conditional - rival.nll_bar_conditional,
        );
        Ok(Some(rival))
    }

    /// Measure the shipped dynamics head against the trivial `z_k = h_t` identity map on the
    /// promoted checkpoint, and report it. The VERDICT is
    /// [`check_dynamics_beats_identity`], which `run_training` applies only after the report
    /// has been finalized.
    ///
    /// [`BarDynamics`] is exported in the checkpoint and [`RolloutMode::Dynamics`] advances
    /// beliefs through it, so `dyn / identity > 1` means the artifact carries a component
    /// that actively degrades the belief it is asked to advance — a trained MLP losing to
    /// doing nothing. That is never a legitimate end state, and it is silent in every other
    /// number the battery prints: the head's own loss keeps shrinking along with the beliefs
    /// it is chasing, so only the ratio against the trivial baseline exposes it.
    ///
    /// The run that motivated the check annealed both NextLat weights to zero at 2/3 of the
    /// schedule (see [`Args::lambda_dyn`]) and promoted every one of its checkpoints from
    /// the region where the dynamics head had received no gradient for thousands of steps
    /// while the trunk kept moving; the ratio read 154 and nothing stopped it shipping. With
    /// the anneal gone this should never fail, which is exactly what makes it worth
    /// asserting: it converts "someone changed the objective and the dynamics head died"
    /// from a full-run-later discovery into an immediate one.
    fn measure_dynamics_versus_identity(
        &self,
        modules: &BarModules,
        supports: &BarSupports,
        set: &PinnedSet,
    ) -> Result<f64> {
        let horizon = self.args.dyn_horizon as i64;
        let ratio = dyn_identity_ratio(
            modules,
            supports,
            set,
            self.args.batch_size,
            horizon,
            self.device,
        )?;
        println!(
            "test split dyn/identity {ratio:.3} at horizon {horizon} (1.0 is the trivial \
             z_k = h_t identity map; below 1.0 means the shipped dynamics head beats it)"
        );
        Ok(ratio)
    }

    /// Forward, backward and update for one batch of `[B, T+1, 5]` DOF plus its
    /// `[B, T+1, 4]` calendar ids.
    ///
    /// `stream` names WHICH resolution's supports the batch is scored against: `None` for the
    /// deployment corpus, `Some(i)` for `self.aux[i]`. It is a parameter rather than an
    /// inference from the batch because nothing in a `BarBatch` carries its resolution — the
    /// calendar ids do, but reading them back to pick a bin geometry would make a silent
    /// cross-resolution scoring an off-by-one away instead of impossible.
    fn optimizer_step(
        &mut self,
        sample: &BarBatch,
        step: usize,
        stream: Option<usize>,
    ) -> Result<StepLoss> {
        let dof = &sample.dof;
        let time_ids = &sample.time_ids;
        let context = dof.size()[1] - 1;
        let horizon = self.args.dyn_horizon as i64;
        ensure!(
            horizon < context,
            "--dyn-horizon {horizon} does not fit in a {context}-bar context"
        );
        // Every auxiliary applies at its configured weight for every step of the run, which
        // is the NextLat reference behaviour and, for `growth`, the behaviour the finding
        // demands: the economics decay LATE, so a decaying weight would switch the term off
        // exactly when it matters. Read once here so the objective, the reported shares and
        // the domination warning all quote the SAME weights.
        let lambda_dyn = self.args.lambda_dyn;
        let lambda_kl = self.args.lambda_kl;
        let lambda_growth = self.args.lambda_growth;
        let (supports, share_scale_offset, growth_support) = match stream {
            None => (
                &self.supports_dev,
                self.share_scale_offset,
                &self.growth_supports[0],
            ),
            Some(index) => (
                self.aux[index].supports_dev(),
                self.aux[index].share_scale_offset(),
                &self.growth_supports[1 + index],
            ),
        };

        // Before the step's own forward, so the measurement cannot see gradients the step
        // accumulated, and only on the deployment stream. It runs its own forwards and
        // zeroes what it leaves behind; see `probe_growth_gradient_share`.
        let growth_probe = if stream.is_none() && GROWTH_PROBE_STEPS.contains(&step) {
            Some(probe_growth_gradient_share(
                &self.vs,
                &self.modules,
                supports,
                growth_support,
                dof,
                time_ids,
                context,
                horizon,
                self.args.scoring,
                self.device,
            )?)
        } else {
            None
        };

        self.optimizer.zero_grad();
        let graph = autocast(self.device.is_cuda(), || {
            forward_losses(
                &self.modules,
                supports,
                growth_support,
                dof,
                time_ids,
                context,
                horizon,
                lambda_dyn,
                lambda_kl,
                lambda_growth,
                self.args.scoring,
                self.device,
            )
        });
        let TrainingGraph {
            loss,
            nll,
            nll_dof,
            dyn_loss,
            kl_loss,
            growth: growth_loss,
            growth_stats,
            identity,
            autocorr,
        } = graph;

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
        // `stream` is also the only thing that distinguishes a primary update from an
        // auxiliary resolution's share of one, and AdamW's `adamw_every` cadence is defined
        // over PRIMARY steps: keying it off a count of `step()` calls would silently halve
        // AdamW's effective interval the moment `--auxiliary-resolutions` is non-empty.
        self.optimizer.step(match stream {
            None => StepKind::Primary,
            Some(_) => StepKind::Auxiliary,
        });
        let dyn_value = dyn_loss.double_value(&[]);
        let kl_value = kl_loss.double_value(&[]);
        let nll_value = nll.double_value(&[]);
        let growth_value = growth_loss.double_value(&[]);
        let growth_stats = growth::GrowthStats::read(&growth_stats);
        let identity = identity.double_value(&[]);
        // The structural bound is 0.6876 at this cap and support, so this can only fire if
        // the support, the cap or the clip stopped agreeing with each other. It is an error
        // and not a clamp because a silently absorbed bad argument is a NaN objective a few
        // steps later, attributed to nothing.
        //
        // The comparison's polarity is deliberate and is the reason this is written as a
        // guard on the GOOD state rather than a test for the bad one. `min_log_argument` is
        // NaN if any bar's argument is NaN, and `NaN > FLOOR` is false, so a NaN argument
        // FAILS this `ensure!` and stops the run. Written the other way round — `ensure!(!(x
        // <= FLOOR))` — a NaN would pass and train silently, which is the three-state bug
        // this repository has now hit four times in one session: a bool over floats cannot
        // say "not measured", so the absent value must land on the failing branch.
        ensure!(
            growth_stats.min_log_argument > growth::LOG_ARGUMENT_FLOOR,
            "the growth term's log argument fell to {:.6} at step {step}, at or below the \
             {} floor. |f_hat| is capped at {} and the r support clips the simple return, so \
             this means one of those three stopped holding — do not lower the floor.",
            growth_stats.min_log_argument,
            growth::LOG_ARGUMENT_FLOOR,
            growth_support.cap()
        );
        if let Some(probe) = growth_probe {
            probe.report(step, lambda_growth);
        }
        Ok(StepLoss {
            nll_bar: nll_value,
            nll_dof: dof_array(&nll_dof),
            dyn_loss: dyn_value,
            kl_loss: kl_value,
            growth_loss: growth_value,
            growth_stats,
            total,
            // The likelihood enters the share denominator on the CATEGORICAL scale: the
            // density rule's measure constant is a property of the binning that no
            // prediction moves and no gradient touches, so leaving it in would make the
            // 25% threshold mean a different thing under each `--scoring`. Taken from the
            // resolution this batch was scored against, because the two bin geometries have
            // different measure constants and a mixed-up one would misreport every share.
            shares: loss_shares(
                nll_value - share_scale_offset,
                lambda_dyn * dyn_value,
                lambda_kl * kl_value,
                lambda_growth * growth_value,
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

    /// The fixed-context diagnostic panel, then the deployed-context promotion decision,
    /// then reports and checkpoints.
    ///
    /// The two are DECOUPLED on purpose. The diagnostic pass runs unconditionally at every
    /// interval from step 0, at a context every ramp stage has trained at, and everything
    /// that does not require the deployed context is populated from it: the per-DOF
    /// breakdown, the conditional variant, the calibration panels, the gain-vs-baselines
    /// curve and the marginalized forecast number. The promotion DECISION alone waits for
    /// the deployed context, because a checkpoint has to be selected at the context it will
    /// be deployed at. When it waits it says so, and the metrics that genuinely were not
    /// measured are DECLARED unmeasured, so they leave a gap in their series instead of a
    /// NaN that reads exactly like a measured catastrophe.
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
            None,
            trade_bench::TRADE_WINDOWS,
        )?;
        self.print_diagnostic(step, &diagnostic);
        // Measured once and carried to the metrics: the bootstrap is the only expensive part
        // and doing it twice would buy nothing. Timed because the epoch line has to state
        // what watching the economics costs — see `EPOCH_BOUNDARY_OVERHEAD_WARN`.
        let bench_started = Instant::now();
        let trade = self.trade(&self.eval.diagnostic, &diagnostic);
        let bench_secs = bench_started.elapsed().as_secs_f64();
        for line in trade.report_lines() {
            println!("step {step}: {line}");
        }

        // The deployed context is what the planner runs at, so it is what selection has to be
        // measured at. Gated on `reached_context` and not on the stage index alone: the ramp
        // is memory-gated, and a hold that ever reaches the context ramp must not be able to
        // promote a model on positional extrapolation just because the step counter says the
        // final stage.
        let mut unmeasured: Vec<UnmeasuredMetric> = Vec::new();
        let deployed_ready = self.schedule.in_final_stage(step)
            && self.reached_context >= self.eval.promotion.context;
        let promotion = if deployed_ready {
            let stats = evaluate(
                &self.modules,
                &self.supports_dev,
                &self.eval.promotion,
                eval_batch,
                self.device,
                false,
                self.args.scoring,
                None,
                trade_bench::TRADE_WINDOWS,
            )?;
            Some((PromotionTarget::Deployed, stats, false))
        } else if final_step {
            // A run that never trained at the deployed context would otherwise end with no
            // promotion, no `pretrain_best.ot` and therefore no held-out number at all — the
            // one outcome a run must never have. Promote at the context actually reached,
            // loudly, and record that context in the checkpoint metadata, the banner and the
            // terminal battery so it can never be mistaken for a full-context selection.
            println!(
                "WARNING step {step}: THE RUN ENDED WITHOUT REACHING THE DEPLOYED {}-BAR \
                 CONTEXT — ramp stage {}, longest context trained {} bars. Promoting on the \
                 {}-bar diagnostic set instead, so this run still leaves an evaluable \
                 checkpoint. CAVEAT: this checkpoint was selected at {} bars of context, not \
                 at the {} bars it would be deployed at; every number derived from it carries \
                 that caveat and the checkpoint metadata records it.",
                self.eval.promotion.context,
                self.schedule.stage(step),
                self.reached_context,
                self.eval.diagnostic.context,
                self.eval.diagnostic.context,
                self.eval.promotion.context,
            );
            Some((PromotionTarget::Diagnostic, diagnostic.clone(), true))
        } else {
            println!(
                "step {step}: skipping promotion — ramp stage {} has not reached the deployed \
                 {}-bar context (longest context trained {} bars). The {}-bar diagnostic panel \
                 above IS measured and charted; only the promotion decision waits.",
                self.schedule.stage(step),
                self.eval.promotion.context,
                self.reached_context,
                self.eval.diagnostic.context,
            );
            let reason = format!(
                "the promotion decision is gated on the deployed {}-bar context and ramp stage \
                 {} has only trained at {} bars, so the deployed-context pass did not run. The \
                 {}-bar diagnostic panel IS measured — read `val diag`, not this series.",
                self.eval.promotion.context,
                self.schedule.stage(step),
                self.reached_context,
                self.eval.diagnostic.context,
            );
            unmeasured.extend(DEPLOYED_CONTEXT_METRICS.iter().map(|metric| {
                UnmeasuredMetric {
                    metric: (*metric).to_owned(),
                    reason: reason.clone(),
                }
            }));
            None
        };

        // `pretrain_last.ot` is deliberately NOT written here. A validation boundary is not
        // the end of the run, and writing it here is exactly what made a killed run's `last` a
        // stale snapshot of the newest VALIDATION rather than of the newest step.
        // `run_training` writes it on the step cadence and once more after the final step.
        // The diagnostic pass just measured THIS model on held-out data at a context every
        // stage has trained at, so a best-so-far at that context is available from step 0 —
        // months before the deployed-context selection can say anything. It is a separate
        // artifact under its own name: the planner still loads `pretrain_best.ot`.
        let diagnostic_scores = self.window_scores(&self.eval.diagnostic, &diagnostic, step);
        self.keep_context_best(step, &diagnostic, &diagnostic_scores)?;
        // The epoch artifact and the snapshots that picture it are the two halves of one
        // record, so the artifact path is carried to the snapshot writer rather than
        // rediscovered: the pictures must depict THESE weights and no others.
        let checkpoint_started = Instant::now();
        let epoch_artifact = if epoch_boundary || final_step {
            Some(self.write_epoch_checkpoint(step, &diagnostic, &diagnostic_scores, &trade)?)
        } else {
            None
        };
        let checkpoint_secs = checkpoint_started.elapsed().as_secs_f64();

        let mut promoted_checkpoint = None;
        let mut promotion_nll = f64::NAN;
        let mut promotion_stats: Option<EvalStats> = None;
        let mut dispersion = Dispersion::nan();
        let mut level = Dispersion::nan();
        let mut promotion_context = f64::NAN;
        // The promotion LEDGER: what the two criteria read, what the incumbent held, and what
        // the decision was. Recorded whether or not it promoted, because a refusal is the
        // interesting half — the rule exists to refuse.
        let mut ledger = SelectionLedger::unmeasured();
        if let Some((target, stats, forced)) = promotion {
            let nll = stats.nll_bar;
            promotion_nll = nll;
            let set = self.promotion_set(target);
            // Copied out of the borrow: every branch below mutates `self`, and the context is
            // the one thing from the set they all have to state.
            let selected_context = set.context;
            promotion_context = selected_context as f64;
            dispersion = self.dispersion(set, &stats);
            level = self.level_dispersion(set, &stats);
            let margin = self.marginal_nll_bar - nll;
            println!(
                "step {step}: held-out nll {dispersion} nats/bar at context {}, {margin:+.4} vs \
                 the calibrated marginal {:.4}{} (diagnostic {:.4} at context {})",
                selected_context,
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
            // ---------------------------------------------------------------------------
            // The decision. The PRIMARY criterion is economic; the density is the guard.
            // [`SELECTION_METRIC`] carries the measurement that inverted this and the
            // calibration of every threshold below.
            //
            // Both criteria are read off the DIAGNOSTIC pass: one pinned set at one fixed
            // context that never moves for the life of the run, and the only context the
            // bench is ever measured at. That makes consecutive decisions comparisons across
            // MODELS rather than across rulers. The DEPLOYED pass above is what makes this
            // read ELIGIBLE — the planner must not load weights selected outside the
            // positional range they run at — and is what the artifact's metadata and the
            // terminal battery are measured at. It is not the comparison.
            // ---------------------------------------------------------------------------
            let scores = self.window_scores(set, &stats, step);
            let selection_nll = diagnostic.nll_bar_conditional;
            let candidate_edge = self.selection_edge_windows(&diagnostic.trade_paths);
            let edge_level = self.bootstrap_traded(&self.eval.diagnostic, &candidate_edge);
            let edge_gain = self.selection_edge_gain(&self.eval.diagnostic, &candidate_edge);
            let nll_guard = self.conditional_regression(&self.eval.diagnostic, &diagnostic_scores);
            let dof_guard = self.returns_regression(&self.eval.diagnostic, &diagnostic_scores);
            // Thresholds in the units the decision is taken in, so the log states the number
            // that was actually compared rather than a multiple the reader has to apply.
            let edge_band = edge_gain.map_or(f64::NAN, |gain| {
                SELECTION_EDGE_SE_MULTIPLE * gain.se.max(0.0)
            });
            let nll_tolerance = nll_guard.map_or(f64::NAN, |delta| {
                SELECTION_NLL_TOLERANCE_SE_MULTIPLE * delta.se.max(0.0)
            });
            // The bench's own row at the selection cap. Read once rather than indexed twice:
            // the criterion, the exposure it was collected at and the rotation rate that pays
            // for it all have to come from ONE measurement of ONE policy.
            let cap_point = trade.cap_curve[SELECTION_CAP_SLOT];
            ledger = SelectionLedger {
                outcome: SelectionOutcome::Unmeasurable,
                bench_context: self.eval.diagnostic.context,
                edge_bps: edge_level.mean,
                edge_se_bps: edge_level.se,
                edge_gain_bps: edge_gain.map_or(f64::NAN, |gain| gain.mean),
                edge_gain_se_bps: edge_gain.map_or(f64::NAN, |gain| gain.se),
                edge_band_bps: edge_band,
                incumbent_edge_bps: self.best_selection_edge_bps,
                turnover: cap_point.turnover,
                gross_exposure: cap_point.mean_abs_position,
                // Against the MEASURED exposure, so a bar whose signal was degenerate and
                // carried no position cannot inflate the rotation rate.
                rotations: cap_point.turnover / cap_point.mean_abs_position,
                nll_conditional: selection_nll,
                nll_delta: nll_guard.map_or(f64::NAN, |delta| delta.mean),
                nll_tolerance,
                incumbent_nll: self.best_selection_nll,
                dof_delta: dof_guard.map_or(f64::NAN, |delta| delta.mean),
            };
            // Every decision states BOTH numbers and both incumbents, in the order the rule
            // reads them, so a promotion that bought edge at the cost of density and one that
            // improved both are distinguishable at a glance instead of by reconstruction.
            println!(
                "step {step}: SELECTION on the {}-bar ruler — edge@{SELECTION_CAP:.2}x \
                 {:+.4} bps/bar (level SE {:.4}, turnover {:.3}/bar absolute at gross {:.3}, \
                 i.e. {:.2} book rotations/bar) vs incumbent {}; \
                 conditional nll {selection_nll:.4} vs incumbent {}",
                self.eval.diagnostic.context,
                edge_level.mean,
                edge_level.se,
                ledger.turnover,
                ledger.gross_exposure,
                ledger.rotations,
                fmt_incumbent_bps(self.best_selection_edge_bps),
                fmt_incumbent_nats(self.best_selection_nll),
            );
            let outcome = selection_outcome(
                forced || self.promotions == 0,
                &candidate_edge,
                edge_gain,
                nll_guard,
                dof_guard,
            );
            // Reporting only. [`selection_outcome`] already decided; this says why, in the
            // units the decision was taken in, so the log states the numbers that were
            // compared rather than multiples a reader has to apply.
            match outcome {
                SelectionOutcome::Promoted if forced || self.promotions == 0 => {
                    // Unconditional. There is no incumbent to beat, and a run whose only
                    // artifact is an unevaluated `pretrain_last.ot` has produced nothing
                    // anybody can act on. `forced` additionally means the run never reached
                    // the deployed context, which is a stated caveat on the file; no selection
                    // at all is a wasted run.
                    println!(
                        "step {step}: FIRST promotion at the {}-bar context{} — there is no \
                         incumbent to pair against, so neither the noise band nor either guard \
                         is applicable and the artifact is written on eligibility alone.",
                        selected_context,
                        if forced {
                            format!(
                                " (FORCED: the run never trained at the deployed {}-bar \
                                 context, so this is the only evaluable artifact it can leave \
                                 and every number derived from it carries that caveat)",
                                self.eval.promotion.context
                            )
                        } else {
                            String::new()
                        },
                    );
                }
                SelectionOutcome::Unmeasurable => {
                    // The bench is the criterion, so a read the bench could not measure is a
                    // read that cannot decide. Refusing keeps the incumbent, which is the safe
                    // half.
                    println!(
                        "step {step}: REFUSING promotion — the trade bench produced no vector \
                         comparable to the incumbent's ({} traded windows against the \
                         incumbent's {}), so the economic criterion is unmeasured at this read. \
                         The incumbent stands; this is not evidence against the candidate.",
                        candidate_edge.len(),
                        self.best_selection_edge_windows
                            .as_ref()
                            .map_or(0, Vec::len),
                    );
                }
                SelectionOutcome::RefusedInsideNoise => {
                    let gain = edge_gain.expect("a measurable comparison implies a gain");
                    println!(
                        "step {step}: REFUSING promotion — paired edge gain {:+.4} +/- {:.4} \
                         bps/bar does not clear the {SELECTION_EDGE_SE_MULTIPLE:.1}-SE band \
                         ({edge_band:+.4}). A single read's edge INTERVAL is ~+/-1.6 bps and \
                         the within-cell read-to-read sd is ~0.02 bps, so an argmax over a \
                         run's reads promotes noise; a candidate inside the band is not \
                         better, it is unresolved.",
                        gain.mean,
                        gain.se,
                    );
                }
                SelectionOutcome::RefusedNllGuard => {
                    let delta = nll_guard.expect("a regression implies a measured delta");
                    println!(
                        "step {step}: REFUSING promotion — edge improved by {:+.4} bps/bar, \
                         which clears the band, but conditional nll REGRESSED by {:+.4} nats \
                         against the incumbent, more than the \
                         {SELECTION_NLL_TOLERANCE_SE_MULTIPLE:.1} paired SE \
                         ({nll_tolerance:.6}) the guard allows. At this bench's paired \
                         resolution that tolerance is comparable to the ENTIRE 5.25e-4 \
                         nats/bar of tradeable content in the r prediction, so a regression \
                         past it is a density that has genuinely broken rather than one that \
                         merely moved.",
                        edge_gain.map_or(f64::NAN, |gain| gain.mean),
                        delta.mean,
                    );
                }
                SelectionOutcome::RefusedDofGuard => {
                    let delta = dof_guard.expect("a regression implies a measured delta");
                    println!(
                        "step {step}: REFUSING promotion — edge improved by {:+.4} bps/bar and \
                         the aggregate density held, but {} regressed by {:+.4} nats, more \
                         than the {SELECTION_GUARD_SE_MULTIPLE:.1} paired SE ({:.6}) the guard \
                         allows. {} is the factor the trade is actually taken on and is \
                         resolved ~100x better than the edge is, so a resolvable regression in \
                         it outranks an economic read that could still be luck.",
                        edge_gain.map_or(f64::NAN, |gain| gain.mean),
                        BAR_DOF_NAMES[SELECTION_GUARD_DOF],
                        delta.mean,
                        delta.se,
                        BAR_DOF_NAMES[SELECTION_GUARD_DOF],
                    );
                }
                SelectionOutcome::Promoted => {
                    let gain = edge_gain.expect("a cleared band implies a gain");
                    println!(
                        "step {step}: PROMOTING — paired edge gain {:+.4} +/- {:.4} bps/bar \
                         clears the {SELECTION_EDGE_SE_MULTIPLE:.1}-SE band ({edge_band:+.4}); \
                         conditional nll {} by {:+.4} nats against a tolerance of \
                         {nll_tolerance:.6}; paired {} delta {:+.4} nats. {}",
                        gain.mean,
                        gain.se,
                        if ledger.nll_delta <= 0.0 {
                            "IMPROVED"
                        } else {
                            "COST, and this promotion is paying for its edge with density:"
                        },
                        -ledger.nll_delta,
                        BAR_DOF_NAMES[SELECTION_GUARD_DOF],
                        ledger.dof_delta,
                        if ledger.nll_delta > 0.0 {
                            "The trade-off is recorded in the artifact's metadata; the \
                             NLL-selected alternative is kept beside it under its own name."
                        } else {
                            "Both criteria moved the same way at this read."
                        },
                    );
                }
                SelectionOutcome::NotEligible => unreachable!(
                    "this branch only runs on an eligible read; `NotEligible` is the ledger's \
                     initial state, not a decision"
                ),
            }
            ledger.outcome = outcome;
            if outcome == SelectionOutcome::Promoted {
                let record = SelectionRecord {
                    step,
                    bench_context: ledger.bench_context,
                    edge_bps: ledger.edge_bps,
                    edge_se_bps: ledger.edge_se_bps,
                    nll_conditional: selection_nll,
                };
                promoted_checkpoint =
                    Some(self.promote(nll, target, eval_batch, &scores, record)?);
                self.best_val_nll_bar = nll;
                self.best_selection_edge_bps = ledger.edge_bps;
                self.best_selection_edge_windows = Some(candidate_edge);
                self.best_selection_nll = selection_nll;
                self.promoted_step = step;
                // The guards pair against the DIAGNOSTIC vector because that is the ruler the
                // decision was taken on; the sidecar written beside the weights stays the
                // deployed-context vector, so cross-run pairing is unchanged.
                self.best_scores = Some(diagnostic_scores);
                self.promotions += 1;
                self.selection_context = selected_context;
            }
            // The RIVAL rule, replayed unchanged on every eligible read: primary
            // `nll_bar_conditional` on the deployed pass, guarded by paired `nll_dof[r]`. It
            // is what the previous rule would have shipped, and it is kept so the inversion
            // is evidence on the test split rather than an assertion. See
            // [`Self::promote_nll_rule`].
            self.promote_nll_rule(step, &stats, &scores, target, ledger.edge_bps)?;
            promotion_stats = Some(stats);
        }

        let (exact, dynamics) = self.rollout_diagnostics();
        // Pictures of the artifact written a few lines above, on the SAME pinned scene at
        // every boundary. Previously this depicted `pretrain_best.ot` and was skipped until
        // the first promotion existed, which is why a 13831-step run left exactly one
        // snapshot: promotion is gated on the deployed context and does not happen for the
        // first two thirds of a run. An epoch artifact always exists at a boundary, so the
        // series has a point at every one of them and every point is that epoch's weights.
        let snapshot_started = Instant::now();
        if let Some(artifact) = epoch_artifact.as_ref() {
            self.write_snapshot(step, artifact)?;
        }
        let snapshot_secs = snapshot_started.elapsed().as_secs_f64();

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
        // Cloned rather than moved: the epoch-boundary record below reads the same
        // diagnostic pass, and a partial move here would make that borrow impossible.
        metrics.val_pit = diagnostic.pit.clone();
        metrics.val_dir_acc = diagnostic.dir_acc;
        metrics.effective_rank = diagnostic.effective_rank;
        metrics.rollout_nll_exact = exact;
        metrics.rollout_nll_dynamics = dynamics;
        metrics.best_val_nll_bar = self.best_val_nll_bar;
        metrics.val_nll_bar_se = dispersion.se;
        metrics.val_nll_bar_ci = (dispersion.ci_low, dispersion.ci_high);
        metrics.val_nll_bar_se_level = level.se;
        // The conditional variant, the per-DOF breakdown and every calibration panel come
        // from the DIAGNOSTIC pass, which always ran. Before this they came from the
        // promotion pass and were therefore NaN for the whole ramp, which is the single
        // reason a 62%-complete run had no held-out signal at all.
        metrics.val_nll_bar_conditional = diagnostic.nll_bar_conditional;
        metrics.val_nll_dof_conditional = diagnostic.nll_dof_conditional;
        metrics.val_nll_bar_conditional_deployed = promotion_stats
            .as_ref()
            .map_or(f64::NAN, |stats| stats.nll_bar_conditional);
        metrics.val_nll_dof_class = diagnostic.nll_dof_class;
        metrics.val_nll_dof_shape = diagnostic.nll_dof_shape;
        // The honest forecasting number beside the teacher-forced one, on identical rows.
        metrics.val_forecast_nll_dof = diagnostic.forecast_nll_dof;
        metrics.val_forecast_teacher_nll_dof = diagnostic.forecast_teacher_nll_dof;
        metrics.val_forecast_nll_se = diagnostic.forecast_nll_se;
        // The bench rides the DIAGNOSTIC pass: fixed context, pinned windows, measured from
        // step 0, so the growth curve is comparable across the whole run and across runs.
        metrics.trade = trade;
        metrics.val_promotion_context = promotion_context;
        metrics.reached_context = self.reached_context as f64;
        metrics.unmeasured = unmeasured;
        metrics.stage_coverage = self.stage_coverage_fractions();
        metrics.promoted_checkpoint = promoted_checkpoint;
        // The whole promotion decision, promoted or refused, so `pretrain_promotions` is a
        // LEDGER rather than a step count. A refusal that leaves no trace is a rule nobody
        // can audit.
        metrics.selection = ledger;
        // Bar-tokens consumed per bar ONE PASS can reach. The denominator is the partition's
        // covered-bar count, not the raw split, so an on-plan run reads exactly 1.000 per
        // epoch instead of sitting permanently below 1 because the unreachable head and
        // sub-context hole are in the denominator. It is a throughput ratio; the coverage
        // ratio is `pass_coverage` beside it, and the two are now independent measurements
        // of two different things rather than one number doing both jobs badly.
        metrics.unique_bar_reuse = self.bars_seen as f64 / self.full_pass_bar_tokens() as f64;
        // Measured coverage of the pass that just ended, or of the pass in progress at a
        // periodic validation. The multiplicity histogram is what makes unevenness visible
        // directly: before the partition it read 28.7% / 45.9% / 22.3% / 3.2% of the corpus at
        // 0, 1, 2 and 3 targets, which no aggregate coverage number could have shown.
        let progress = self.pass.audit(&self.pass_layout, &self.pass_ledger);
        metrics.pass_coverage = progress.coverage_fraction();
        metrics.pass_multiplicity_bars = progress.multiplicity_bars;
        metrics.pass_remainder_bars = [
            progress.remainder.head_bars,
            progress.remainder.short_symbol_bars,
            progress.remainder.hole_bars,
            progress.unissued_bars(),
        ];
        metrics.stage_conditioning_bars = progress.mean_conditioning_bars.clone();
        // The RUN-scoped counterpart of `progress`, and the reason it exists: `progress` is a
        // PER-PASS census whose multiplicity histogram `require_full_pass` pins to a single
        // spike at one, so on a three-pass run it reads "twice: 0, three or more: 0" at every
        // tick of the third pass. That is correct within a pass and it was read as a statement
        // about the RUN for an entire analysis session, in preference to `unique_bar_reuse`
        // showing 2.85 on the same screen. Both go into the reports so the per-pass zeros are
        // drawn beside the number that contradicts them.
        let run = self
            .pass
            .cumulative_coverage(&self.census, &self.pass_layout, &self.pass_ledger);
        // Cheap, and it fires at the FIRST validation of any run rather than in hour forty: the
        // only way this trips is a reconstruction that lost or double-counted bars, which would
        // UNDERSTATE reuse — the exact direction of the original error. It adds no constraint to
        // `require_full_pass`, which continues to own whether the pass itself was complete.
        run.require_accounted()?;
        metrics.run_effective_epochs = run.effective_epochs();
        metrics.run_exposure_bars = run.multiplicity_bars;
        // Known from step zero, not only in hindsight: the projection is what makes a
        // multi-epoch recipe visible on the FIRST validation tick instead of becoming apparent
        // once the realized curve has already crossed one.
        metrics.projected_effective_epochs =
            self.projected_bar_tokens(step) as f64 / self.full_pass_bar_tokens().max(1) as f64;
        metrics.planned_effective_epochs = self.args.epochs as f64;
        self.reporter.record_epoch(&metrics)?;

        self.train_nll_sum = 0.0;
        self.train_nll_dof_sum = [0.0; BAR_DOF];
        self.train_steps = 0;

        println!(
            "step {step}: unique_bar_reuse {:.4} ({} bar-tokens consumed / {} bar-tokens in one \
             pass), rollout nll h1 {:.4} exact / {:.4} dynamics",
            metrics.unique_bar_reuse,
            self.bars_seen,
            self.full_pass_bar_tokens(),
            exact[0],
            dynamics[0]
        );
        println!("step {step}: {}", progress.summary());
        // Printed BESIDE the per-pass line, never instead of it, and always — not only when it
        // exceeds one. A conditional banner would teach a reader that silence means one pass,
        // and silence is what they already mistook for it.
        println!("step {step}: {}", run.summary());
        if metrics.projected_effective_epochs > 1.0 {
            println!(
                "step {step}: MULTI-EPOCH RUN — projected {:.4} passes over the training split \
                 by step {} ({} planned epochs). Every held-out number this run reports is from \
                 a model that will have seen its training bars more than once, and NO per-pass \
                 panel can show that: `pretrain_pass_multiplicity` and `pretrain_stage_coverage` \
                 are per-pass censuses and read identically on pass three and pass one. The \
                 run-scoped series are `cover_effective_epochs` and `cover_run_bar_exposure`.",
                metrics.projected_effective_epochs,
                self.schedule.total_steps,
                self.args.epochs,
            );
        }

        // The epoch-indexed record, emitted at exactly the boundaries that leave an
        // artifact, so the series and the checkpoints on disk are one for one.
        if epoch_artifact.is_some() {
            let boundary = self.epoch_boundary_record(
                step,
                &diagnostic,
                trade,
                bench_secs,
                snapshot_secs,
                checkpoint_secs,
            );
            println!("step {step}: {}", boundary.console_line());
            self.warn_on_boundary_cost(step, &boundary);
            self.warn_on_bar_token_shortfall(step, &boundary);
            self.reporter.record_epoch_boundary(&boundary)?;
            self.epoch_started = Instant::now();
            self.epoch_start_bars = self.bars_seen;
            self.epoch_dyn_identity_sum = 0.0;
            self.epoch_dyn_identity_steps = 0;
        }
        Ok(())
    }

    /// Bar-tokens one full pass over the training split costs.
    ///
    /// THE denominator of every budget number this run reports. It exists as one function
    /// so that when the corpus's notion of a pass changes — a stage partition that makes
    /// the reachable bar count smaller than the raw split, say — exactly one line moves and
    /// the epoch panel, the shortfall warning and the projection cannot disagree.
    fn full_pass_bar_tokens(&self) -> u64 {
        self.pass.covered_bars()
    }

    /// Bar-tokens the run will have delivered at its last step, IF the ramp it is executing
    /// right now holds for the rest of it.
    fn projected_bar_tokens(&self, step: usize) -> u64 {
        projected_bar_tokens(&self.schedule, self.bars_seen, step)
    }

    /// Assemble this boundary's progress row.
    fn epoch_boundary_record(
        &self,
        step: usize,
        diagnostic: &EvalStats,
        trade: TradeBench,
        bench_secs: f64,
        snapshot_secs: f64,
        checkpoint_secs: f64,
    ) -> EpochBoundary {
        let forecast: f64 = diagnostic.forecast_nll_dof.iter().sum();
        let teacher: f64 = diagnostic.forecast_teacher_nll_dof.iter().sum();
        let full_pass = self.full_pass_bar_tokens();
        EpochBoundary {
            epoch: self.epoch,
            global_step: step,
            epoch_bar_tokens: self.bars_seen.saturating_sub(self.epoch_start_bars),
            full_pass_bar_tokens: full_pass,
            run_bar_tokens: self.bars_seen,
            run_target_bar_tokens: full_pass.saturating_mul(self.args.epochs as u64),
            projected_run_bar_tokens: self.projected_bar_tokens(step),
            epoch_secs: self.epoch_started.elapsed().as_secs_f64(),
            boundary_secs: bench_secs + snapshot_secs + checkpoint_secs,
            bench_secs,
            snapshot_secs,
            val_nll_bar: diagnostic.nll_bar,
            forecast_nll_bar: forecast,
            teacher_forcing_inflation: forecast - teacher,
            dyn_vs_identity: if self.epoch_dyn_identity_steps > 0 {
                self.epoch_dyn_identity_sum / self.epoch_dyn_identity_steps as f64
            } else {
                f64::NAN
            },
            trade,
        }
    }

    /// Say so when watching the economics starts costing a visible share of the compute.
    ///
    /// The bench and the snapshots are worth real time — an epoch with no economic reading
    /// is an epoch nobody can judge — but they are not worth an appreciable share of it.
    /// The response is stated rather than taken: the budgets are FIXED, because an epoch
    /// series measured on 256 windows at epoch 0 and on 64 at epoch 3 is not a series, and
    /// silently shrinking one mid-run would destroy the only comparison this panel exists
    /// to make.
    fn warn_on_boundary_cost(&self, step: usize, boundary: &EpochBoundary) {
        let share = boundary.boundary_share();
        if !(share > EPOCH_BOUNDARY_OVERHEAD_WARN) {
            return;
        }
        println!(
            "WARNING step {step}: the epoch-{} boundary cost {:.1} min — {:.1}% of the \
             epoch's {:.1} min, above the {:.0}% this panel budgets for. Bench {:.1} min \
             over {} windows, snapshots {:.1} min over {} windows x {} draws. The rollout \
             is linear in the draws, so `--snapshot-samples` is the cheapest thing to turn \
             down; then `--snapshot-windows`, then `trade_bench::TRADE_WINDOWS`. All three \
             are for the NEXT run and are deliberately not adjusted mid-run, because an \
             epoch series whose window budget moves between its own points is not \
             comparable to itself.",
            boundary.epoch,
            boundary.boundary_secs / 60.0,
            100.0 * share,
            boundary.epoch_secs / 60.0,
            100.0 * EPOCH_BOUNDARY_OVERHEAD_WARN,
            boundary.bench_secs / 60.0,
            trade_bench::TRADE_WINDOWS.min(self.eval.diagnostic.windows.len()),
            boundary.snapshot_secs / 60.0,
            self.eval.snapshot.windows.len(),
            self.args.snapshot_samples,
        );
    }

    /// Say, at every boundary, what the run is on course to actually deliver.
    ///
    /// A step count sized from a ramp the card refuses to execute delivers a fraction of
    /// the tokens it was priced for, and the failure is silent: the run still completes
    /// every step it announced, still writes an artifact per boundary, and still calls the
    /// result `--epochs 3`. Job 2856 delivered 1.33 passes under that label and nothing
    /// said so until the closing banner.
    fn warn_on_bar_token_shortfall(&self, step: usize, boundary: &EpochBoundary) {
        let projected = boundary.projected_fraction();
        if !(projected < BAR_TOKEN_SHORTFALL_WARN) {
            return;
        }
        println!(
            "WARNING step {step}: BAR-TOKEN SHORTFALL — at the batch ramp x{:?} this run is \
             actually executing (derived x{:?}, declared x{:?}) its remaining {} steps \
             project {} bar-tokens against the {} that `--epochs {}` asked for: {:.0}%, i.e. \
             {:.2} effective passes over {} unique training bars, not {}. The step count is \
             deliberately NOT extended — a run that silently ran longer would not be \
             comparable to its siblings — so the honest label for this run is its effective \
             epoch count.",
            self.schedule.batch_ramp,
            self.derived_batch_ramp,
            BATCH_RAMP,
            self.schedule.total_steps.saturating_sub(step + 1),
            boundary.projected_run_bar_tokens,
            boundary.run_target_bar_tokens,
            self.args.epochs,
            100.0 * projected,
            boundary.projected_epochs(),
            boundary.full_pass_bar_tokens,
            self.args.epochs,
        );
    }

    /// The fixed-context held-out panel, printed at EVERY validation from step 0.
    ///
    /// This is the early-warning read. It is measured at [`PretrainArgs::diagnostic_context`]
    /// by construction, so it is comparable across the whole run and across runs, and it does
    /// not wait for the ramp. It used to be computed here and printed only inside the
    /// promotion branch, which made it invisible for the two thirds of a run that cannot
    /// promote.
    ///
    /// The second line is the one that matters for trust: `nll_bar` teacher-forces each chain
    /// factor on the realized value of the SAME bar's earlier factors, so only the first chain
    /// factor is a forecast and the rest is within-bar accounting. The forecast figure scores
    /// every factor against the head's own marginalized law instead, and the difference is how
    /// much of the headline was accounting.
    fn print_diagnostic(&self, step: usize, stats: &EvalStats) {
        println!(
            "step {step}: DIAG at the fixed {}-bar context — nll {:.4} nats/bar ({:+.4} vs the \
             calibrated marginal {:.4}, {:+.4} vs uniform {:.4}), conditional {:.4}, dir acc \
             {:.4}, effective rank {:.1}",
            self.eval.diagnostic.context,
            stats.nll_bar,
            self.marginal_nll_bar - stats.nll_bar,
            self.marginal_nll_bar,
            self.baselines.uniform_nll_bar - stats.nll_bar,
            self.baselines.uniform_nll_bar,
            stats.nll_bar_conditional,
            stats.dir_acc,
            stats.effective_rank,
        );
        println!("step {step}: DIAG {}", self.per_dof_line(stats));
        let teacher: f64 = stats.forecast_teacher_nll_dof.iter().sum();
        let forecast: f64 = stats.forecast_nll_dof.iter().sum();
        let parts: Vec<String> = BAR_DOF_NAMES
            .iter()
            .enumerate()
            .map(|(dof, name)| {
                format!(
                    "{name} {:.4} vs {:.4} ({:+.4})",
                    stats.forecast_nll_dof[dof],
                    stats.forecast_teacher_nll_dof[dof],
                    stats.forecast_nll_dof[dof] - stats.forecast_teacher_nll_dof[dof],
                )
            })
            .collect();
        println!(
            "step {step}: FORECAST-ONLY nll {forecast:.4} +/- {:.4} (MC, {} draws in {} groups) \
             vs TEACHER-FORCED {teacher:.4} on the identical rows: teacher-forcing is {:.4} \
             nats/bar OPTIMISTIC. The forecast number conditions every one of the five factors \
             on strictly PAST bars only and is the honest forecasting figure; the teacher-forced \
             number is the joint likelihood of the bar, and every per-factor term after `{}` in \
             the chain is within-bar accounting given realized same-bar values. Per DOF \
             (forecast vs teacher-forced): {}",
            stats.forecast_nll_se,
            FORECAST_MC_DRAWS,
            FORECAST_MC_GROUPS,
            forecast - teacher,
            BAR_DOF_NAMES[BAR_CHAIN[0]],
            parts.join(" | "),
        );
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
    /// require. The metadata carries the corpus fingerprint, the split instants, the
    /// selection rule — all folded into the lineage hash — and `step`, the optimizer step
    /// these exact weights are from, so no artifact a run writes can be mistaken for another
    /// step's by anyone holding the file without the log.
    fn write_checkpoint(
        &self,
        weights: &Path,
        step: usize,
        selection_context: i64,
        selection: Option<SelectionRecord>,
    ) -> Result<PathBuf> {
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
            Some(self.training_provenance(step, selection_context, selection)),
        )
        .with_context(|| format!("failed writing metadata for {}", weights.display()))
    }

    /// The step-cadence pair: the tagged crash-recovery snapshot and `pretrain_last.ot`.
    ///
    /// They are written TOGETHER because the alternative was measured to be a trap. `last`
    /// used to be written only from `validate`, on the validation cadence, while the tagged
    /// snapshot is written every `checkpoint_every` steps — a much finer one. That left a
    /// directory listing in which `pretrain_step_30720.ot` was a genuinely later state than
    /// `pretrain_last.ot`, which is the opposite of what the name promises. Writing both here
    /// makes `last` provably no older than the newest `pretrain_step_*.ot`.
    fn write_step_artifacts(&self, step: usize) -> Result<()> {
        let path = self.run.weights.join(format!("pretrain_step_{step}.ot"));
        self.write_checkpoint(&path, step, 0, None)?;
        self.prune_step_checkpoints()?;
        self.write_last_checkpoint(step)?;
        Ok(())
    }

    /// `pretrain_last.ot`: the weights after optimizer step `step`, and nothing more.
    ///
    /// It carries NO held-out scores — no `.windows.json` sidecar — and its metadata records
    /// `selection_context = 0`, so the file itself states that no decision chose it and that
    /// any number measured from it belongs to a read nobody reported. `global_step` in that
    /// same metadata says which step it holds, which is the question a reader of a weights
    /// directory actually has and which used to be answerable only from the run's log.
    fn write_last_checkpoint(&self, step: usize) -> Result<PathBuf> {
        let path = self.run.weights.join("pretrain_last.ot");
        self.write_checkpoint(&path, step, 0, None)?;
        Ok(path)
    }

    /// Every epoch boundary leaves one artifact, unconditionally, with its held-out numbers
    /// AND with what those numbers were worth.
    ///
    /// Promotion is gated on the deployed context and the batch ramp is memory-gated, so the
    /// two are not the same event and in job 2856 they were 9221 steps apart: epochs 1 and 2
    /// finished with nothing but a rolling `pretrain_last.ot` that the next step-cadence write
    /// overwrote. An epoch of compute must be recoverable and, separately, EVALUABLE — the
    /// window-score sidecar carries the whole diagnostic pass for these exact weights, so an
    /// epoch artifact can be compared window by window against any other artifact of any run
    /// without being re-scored.
    ///
    /// The Kelly bench of that same pass rides in the sidecar for the same reason. A reader
    /// holding an epoch artifact asks what it was WORTH, and a sidecar that answers only in
    /// nats forces a reload and a re-score of the pass that just ran. Returns the artifact
    /// path, so the snapshots taken at this boundary picture THESE weights.
    fn write_epoch_checkpoint(
        &self,
        step: usize,
        diagnostic: &EvalStats,
        scores: &WindowScores,
        trade: &TradeBench,
    ) -> Result<PathBuf> {
        let context = self.reached_context;
        let path = self.run.weights.join(format!(
            "pretrain_epoch_{}_ctx{context}.ot",
            self.epoch
        ));
        self.write_checkpoint(&path, step, 0, None)?;
        let sidecar = window_scores_path(&path);
        let mut scores = scores.clone();
        // Absent, not zeroed, when nothing was measured: an edge of `0.0` is a finding and
        // "no bench ran here" is not.
        scores.trade = trade.measured().then(|| TradeSummary::from(trade));
        scores
            .save(&sidecar)
            .with_context(|| format!("failed writing {}", sidecar.display()))?;
        println!(
            "step {step}: epoch {} artifact {} — longest context trained {context} bars, \
             held-out nll {:.4} nats/bar (conditional {:.4}) at the fixed {}-bar diagnostic \
             context over {} windows, scores{} beside it in {}. Written unconditionally: this \
             is not a promotion and the metadata records no selection context.",
            self.epoch,
            path.display(),
            diagnostic.nll_bar,
            diagnostic.nll_bar_conditional,
            self.eval.diagnostic.context,
            scores.windows.len(),
            if scores.trade.is_some() {
                " and the Kelly bench of this exact pass"
            } else {
                ""
            },
            sidecar
                .file_name()
                .map_or_else(|| sidecar.display().to_string(), |n| n
                    .to_string_lossy()
                    .into_owned()),
        );
        Ok(path)
    }

    /// Best-so-far at the FIXED DIAGNOSTIC context, maintained at every validation.
    ///
    /// Selection at the deployed context is the only thing the planner may load, because a
    /// model chosen at 896 bars and run at 2048 is being run outside the positional range it
    /// was chosen in. But a run held below the deployed context would otherwise have no
    /// defensible "best" at all, and "the weights at the last step" is not one. So the best at
    /// each context is tracked separately, under its own file name, and the metadata states the
    /// context its selection was taken at.
    ///
    /// Verified by reloading, which proves the artifact is readable and lineage-consistent. It
    /// is deliberately NOT re-scored: the round-trip evaluation `promote` does costs a full
    /// pass over the pinned set, and this fires at nearly every early validation.
    fn keep_context_best(
        &mut self,
        step: usize,
        diagnostic: &EvalStats,
        scores: &WindowScores,
    ) -> Result<()> {
        let context = self.eval.diagnostic.context;
        let selection = diagnostic.nll_bar_conditional;
        if !selection.is_finite() {
            return Ok(());
        }
        let previous = self.best_by_context.get(&context).copied();
        if previous.is_some_and(|best| selection >= best) {
            return Ok(());
        }
        self.record_context_best(context, selection);
        // The deployed-context NLL best is [`NLL_RULE_CHECKPOINT`], written by
        // `promote_nll_rule`; the artifact the planner loads is chosen economically. This path
        // only ever owns the diagnostic-context name.
        if context == self.eval.promotion.context {
            return Ok(());
        }
        let path = self
            .run
            .weights
            .join(format!("pretrain_best_diag{context}.ot"));
        self.write_checkpoint(&path, step, context, None)?;
        let sidecar = window_scores_path(&path);
        scores
            .save(&sidecar)
            .with_context(|| format!("failed writing {}", sidecar.display()))?;
        // Loaded back rather than trusted: it proves the file is readable and that its lineage
        // matches its own metadata, which is the failure this cheap check can actually catch.
        drop(
            BarWorldModel::load(&path, &world_model_metadata_path(&path), self.device)
                .with_context(|| format!("{} is not reloadable", path.display()))?,
        );
        println!(
            "step {step}: new best at the fixed {context}-bar context — conditional nll \
             {selection:.4} nats/bar, improving on {}. Written to {} and reloaded. This is NOT \
             the deployed-context selection the planner loads; its metadata records \
             selection_context={context} against deployed_context={}.",
            previous.map_or_else(
                || "no earlier measurement".to_owned(),
                |best| format!("{best:.4}"),
            ),
            path.display(),
            self.eval.promotion.context,
        );
        Ok(())
    }

    /// Note the best conditional held-out NLL at one evaluation context.
    ///
    /// Kept per context because they are not comparable: a longer context is a strictly
    /// easier prediction problem, so one number spanning both would let a ramp step-up look
    /// like learning. Every entry is reported in the final banner.
    fn record_context_best(&mut self, context: i64, selection: f64) {
        let slot = self
            .best_by_context
            .entry(context)
            .or_insert(f64::INFINITY);
        if selection < *slot {
            *slot = selection;
        }
    }

    /// The RIVAL selection rule, replayed on every eligible read: primary
    /// `nll_bar_conditional` on the deployed-context pass, gated by the paired `nll_dof[r]`
    /// guard at [`SELECTION_GUARD_SE_MULTIPLE`]. Writes [`NLL_RULE_CHECKPOINT`], which the
    /// planner never loads.
    ///
    /// This is the rule that shipped before the inversion, run UNCHANGED and in full — same
    /// primary, same guard, same multiple, same pass, same windows — so the terminal
    /// test-split comparison is between the new rule and the REAL previous rule rather than a
    /// strawman reconstruction of it. Without this artifact the inversion is an assertion;
    /// with it there are two checkpoints on one held-out split and the claim is checkable.
    ///
    /// Deliberately NOT round-trip re-scored the way [`Self::promote`] is: that costs a full
    /// pass over the pinned set, this artifact is never deployed, and the reload below already
    /// proves the file is readable and lineage-consistent, which is the failure a cheap check
    /// can actually catch.
    fn promote_nll_rule(
        &mut self,
        step: usize,
        stats: &EvalStats,
        scores: &WindowScores,
        target: PromotionTarget,
        edge_bps: f64,
    ) -> Result<()> {
        let selection = stats.nll_bar_conditional;
        if !selection.is_finite() {
            return Ok(());
        }
        let set = self.promotion_set(target);
        let context = set.context;
        let guard = self.nll_rule_regression(set, scores);
        let regressed = guard
            .is_some_and(|delta| delta.mean > SELECTION_GUARD_SE_MULTIPLE * delta.se.max(0.0));
        let improved = selection < self.best_val_nll_bar_conditional;
        if !improved || regressed {
            return Ok(());
        }
        let previous = self.best_val_nll_bar_conditional;
        let path = self.run.weights.join(NLL_RULE_CHECKPOINT);
        // The rival's artifact records BOTH numbers too, so the two files can be compared
        // against each other without either run's log.
        let record = SelectionRecord {
            step,
            bench_context: context,
            edge_bps,
            edge_se_bps: f64::NAN,
            nll_conditional: selection,
        };
        self.write_checkpoint(&path, step, context, Some(record))?;
        let sidecar = window_scores_path(&path);
        scores
            .save(&sidecar)
            .with_context(|| format!("failed writing {}", sidecar.display()))?;
        drop(
            BarWorldModel::load(&path, &world_model_metadata_path(&path), self.device)
                .with_context(|| format!("{} is not reloadable", path.display()))?,
        );
        self.best_val_nll_bar_conditional = selection;
        self.nll_rule_scores = Some(scores.clone());
        self.nll_rule_step = step;
        self.nll_rule_promotions += 1;
        self.nll_rule_edge_bps = edge_bps;
        self.record_context_best(context, selection);
        println!(
            "step {step}: the NLL RULE would have promoted here — conditional nll \
             {selection:.4} nats/bar at {context} bars, improving on {}, with the 0.25x edge at \
             {edge_bps:+.4} bps/bar. Written to {} as the rival artifact. The planner loads the \
             ECONOMICALLY selected pretrain_best.ot; this file exists so the two rules are \
             compared on the test split instead of argued about.",
            if previous.is_finite() {
                format!("{previous:.4}")
            } else {
                "no earlier eligible read".to_owned()
            },
            path.display(),
        );
        Ok(())
    }

    /// Keep the newest [`RETAINED_STEP_CHECKPOINTS`] step-tagged checkpoints plus the
    /// [`PLATEAU_ANCHOR_CHECKPOINTS`] pass-boundary anchors, and delete the rest, sidecars
    /// included.
    ///
    /// A 128 MiB artifact every [`DEFAULT_CHECKPOINT_EVERY`] steps would be 3.5 GiB over a
    /// 13831-step run. The epoch artifacts and `pretrain_best*.ot` are named differently and
    /// are never touched here: this window is crash insurance with a bounded footprint, not an
    /// archive.
    ///
    /// The anchors are the one exception and they are exempt PRECISELY BECAUSE they are the
    /// oldest artifacts on disk — a newest-N window deletes them first, which is how
    /// `bardist_v2` ended up with a repetition question and no artifact able to answer it. See
    /// [`PLATEAU_ANCHOR_CHECKPOINTS`].
    fn prune_step_checkpoints(&self) -> Result<()> {
        let mut tagged: Vec<(usize, PathBuf)> = Vec::new();
        let entries = std::fs::read_dir(&self.run.weights).with_context(|| {
            format!("failed listing {}", self.run.weights.display())
        })?;
        for entry in entries {
            let path = entry
                .with_context(|| format!("failed reading {}", self.run.weights.display()))?
                .path();
            // `pretrain_step_<n>.ot` exactly: the sidecars carry further extensions and are
            // deleted with their weights, never matched on their own.
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(rest) = name.strip_prefix("pretrain_step_") else {
                continue;
            };
            let Some(digits) = rest.strip_suffix(".ot") else {
                continue;
            };
            if let Ok(tag) = digits.parse::<usize>() {
                tagged.push((tag, path));
            }
        }
        if tagged.len() <= RETAINED_STEP_CHECKPOINTS {
            return Ok(());
        }
        tagged.sort_unstable_by_key(|(tag, _)| *tag);
        let tags: Vec<usize> = tagged.iter().map(|(tag, _)| *tag).collect();
        let anchors = plateau_anchor_tags(
            &tags,
            self.schedule.steps_per_epoch,
            self.schedule.total_steps,
            self.schedule.lr_plateau_fraction,
        );
        let newest_from = tagged.len() - RETAINED_STEP_CHECKPOINTS;
        for (index, (tag, path)) in tagged.into_iter().enumerate() {
            if index >= newest_from || anchors.contains(&tag) {
                continue;
            }
            for sidecar in [
                world_model_metadata_path(&path),
                world_model_supports_path(&path, self.args.resolution_secs),
                window_scores_path(&path),
            ] {
                if sidecar.exists() {
                    std::fs::remove_file(&sidecar)
                        .with_context(|| format!("failed deleting {}", sidecar.display()))?;
                }
            }
            std::fs::remove_file(&path)
                .with_context(|| format!("failed deleting {}", path.display()))?;
        }
        Ok(())
    }

    /// What this run was trained and selected on, for the checkpoint sidecar.
    ///
    /// `selection_context` is the context of the held-out set the artifact was CHOSEN on, and
    /// zero for an artifact that was never chosen at all: the rolling `pretrain_last.ot`, the
    /// step-tagged crash-recovery checkpoints and the epoch-boundary artifacts. A reader must
    /// be able to tell "best on 4096 held-out windows at 896 bars" from "whatever the weights
    /// happened to be at step 4610", and the file is the only place that can say it.
    /// `selection` is the reading the promotion decision was taken on, present only on the
    /// artifacts a decision actually chose. Both criteria are recorded, whichever way they
    /// pointed: an artifact promoted for its edge while regressing on density must carry that
    /// fact, and so must one that improved both. `step` is unconditional and separate from
    /// both: it is the optimizer step the weights are from, which every artifact has and which
    /// nothing but the run's log used to record for the unselected ones.
    fn training_provenance(
        &self,
        step: usize,
        selection_context: i64,
        selection: Option<SelectionRecord>,
    ) -> BarTrainingProvenance {
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
            // The context this artifact's selection was actually taken at, beside the one it
            // is meant to be deployed at. They differ only when the ramp never got there, and
            // that difference is the difference between a deployable artifact and one that is
            // being run outside the positional range it was selected in. Zero means the
            // artifact was never selected on held-out data.
            selection_context,
            deployed_context: self.eval.promotion.context,
            reached_context: self.reached_context,
            global_step: Some(step),
            selection_bench_context: selection.map(|record| record.bench_context),
            selection_edge_bps: selection.map(|record| record.edge_bps),
            selection_edge_se_bps: selection.map(|record| record.edge_se_bps),
            selection_nll_conditional: selection.map(|record| record.nll_conditional),
            // Absolute windows per stage, not multipliers: the base batch is itself clamped at
            // startup by the capacity probe, so a multiplier does not identify what ran.
            batch_ramp: self
                .schedule
                .batch_ramp
                .iter()
                .map(|multiple| self.schedule.base_batch * multiple)
                .collect(),
            // The learning-rate schedule's one free parameter. Without it a reader cannot tell
            // whether a checkpoint at one full pass sat at peak rate or at the annealed floor,
            // which is the difference between two entirely different experiments.
            lr_plateau_fraction: self.schedule.lr_plateau_fraction,
        }
    }

    /// Which pinned set a promotion decision was taken on.
    fn promotion_set(&self, target: PromotionTarget) -> &PinnedSet {
        match target {
            PromotionTarget::Deployed => &self.eval.promotion,
            PromotionTarget::Diagnostic => &self.eval.diagnostic,
        }
    }

    /// The TEST split at the context the promoted checkpoint was selected at.
    ///
    /// Falls back to the deployed-context set when nothing was promoted yet, which cannot
    /// happen on the battery path — `run_training` refuses to reach it with zero promotions —
    /// but keeps this a total function rather than a panic waiting for a refactor.
    fn test_set(&self) -> &PinnedSet {
        if self.selection_context == self.eval.test_diagnostic.context {
            &self.eval.test_diagnostic
        } else {
            &self.eval.test
        }
    }

    /// Paired per-window difference of one scalar against an incumbent vector, block
    /// bootstrapped, or `None` when there is nothing comparable to pair against.
    ///
    /// The ONE implementation of "how this run compares two checkpoints". Every selection
    /// criterion and every guard goes through it, so they cannot drift apart in blocking, in
    /// bootstrap draws or in what counts as comparable.
    ///
    /// Paired and not two levels: the unpaired SE of an NLL level is ~0.10 nats and its
    /// minimum detectable difference ~0.41, which would let any realistic regression through,
    /// and the unpaired interval of the edge is ~+/-1.6 bps against differences of ~0.02. On
    /// identical windows the per-window correlation between two checkpoints of the same run is
    /// very high, so the difference is resolvable at a few hundredths of a nat and a few
    /// hundredths of a basis point.
    ///
    /// `None` on the first decision (nothing to pair against) and whenever the incumbent was
    /// scored on a different set, which only happens on a run too short to reach the deployed
    /// context.
    fn paired_difference(
        &self,
        set: &PinnedSet,
        incumbent: Option<&WindowScores>,
        candidate: &WindowScores,
        of: impl Fn(&WindowScore) -> f64,
    ) -> Option<Dispersion> {
        let incumbent = incumbent?;
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
            .map(|(new, old)| of(new) - of(old))
            .collect();
        Some(block_bootstrap(
            &deltas,
            &self.blocks(set),
            BOOTSTRAP_DRAWS,
            BOOTSTRAP_SEED,
        ))
    }

    /// Paired regression of the guarded factor against the ECONOMICALLY promoted checkpoint.
    /// Positive means the candidate is worse at the factor the trade is taken on.
    fn returns_regression(&self, set: &PinnedSet, candidate: &WindowScores) -> Option<Dispersion> {
        self.paired_difference(set, self.best_scores.as_ref(), candidate, |window| {
            window.nll_dof[SELECTION_GUARD_DOF]
        })
    }

    /// Paired regression of the conditional aggregate against the same incumbent. Positive
    /// means the candidate's density is worse.
    fn conditional_regression(
        &self,
        set: &PinnedSet,
        candidate: &WindowScores,
    ) -> Option<Dispersion> {
        self.paired_difference(set, self.best_scores.as_ref(), candidate, |window| {
            window.nll_bar_conditional
        })
    }

    /// The rival NLL rule's own guard, pairing against the rival's own incumbent rather than
    /// against the economically promoted artifact. Without this the two rules would not be
    /// running the disciplines they claim, and the terminal comparison would be between the
    /// new rule and a strawman.
    fn nll_rule_regression(
        &self,
        set: &PinnedSet,
        candidate: &WindowScores,
    ) -> Option<Dispersion> {
        self.paired_difference(set, self.nll_rule_scores.as_ref(), candidate, |window| {
            window.nll_dof[SELECTION_GUARD_DOF]
        })
    }

    /// Per-window net Kelly edge over the unconditional-marginal null at [`SELECTION_CAP`], in
    /// BPS per bar, one entry per traded window.
    ///
    /// The fractions are re-clamped from the already-solved uncapped optimum exactly as the
    /// charted cap curve does — same helper, so the promotion criterion and the chart a reader
    /// checks it against are one measurement rather than two that look alike. The null is
    /// model-independent, so subtracting it changes no paired difference between two
    /// checkpoints; it is subtracted anyway so the LEVEL this rule records in the metadata is
    /// the same "edge" the cap curve prints.
    fn selection_edge_windows(&self, paths: &ChunkPaths) -> Vec<f64> {
        if paths.is_empty() {
            return Vec::new();
        }
        let free_marginal =
            trade_bench::marginal_position(&self.supports_dev, trade_bench::FREE_LEVERAGE);
        let recapped = trade_bench::recap(&paths.windows, SELECTION_CAP, free_marginal);
        let cost = trade_bench::DEFAULT_COST_BPS;
        let model =
            trade_bench::window_growth_at(&recapped, trade_bench::POLICY_MODEL, SELECTION_CAP, cost);
        let null = trade_bench::window_growth_at(
            &recapped,
            trade_bench::POLICY_MARGINAL,
            SELECTION_CAP,
            cost,
        );
        model
            .iter()
            .zip(null.iter())
            .map(|(model, null)| (model - null) * 1.0e4)
            .collect()
    }

    /// Block-bootstrapped mean of a per-traded-window quantity.
    ///
    /// The bench trades a PREFIX of the pinned set, so the block ids are the same prefix and
    /// are truncated rather than recomputed: two windows of one ticker inside one calendar
    /// month are one draw here for exactly the reason they are one draw everywhere else.
    fn bootstrap_traded(&self, set: &PinnedSet, values: &[f64]) -> Dispersion {
        if values.is_empty() {
            return Dispersion::nan();
        }
        let mut blocks = self.blocks(set);
        blocks.truncate(values.len());
        block_bootstrap(values, &blocks, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    }

    /// Paired per-window difference of the economic criterion against the promoted
    /// checkpoint's, in bps/bar. Positive means the candidate is economically better.
    ///
    /// `None` until there is an incumbent vector of the same length to pair against.
    fn selection_edge_gain(&self, set: &PinnedSet, candidate: &[f64]) -> Option<Dispersion> {
        let incumbent = self.best_selection_edge_windows.as_ref()?;
        if incumbent.len() != candidate.len() || candidate.is_empty() {
            return None;
        }
        let deltas: Vec<f64> = candidate
            .iter()
            .zip(incumbent.iter())
            .map(|(new, old)| new - old)
            .collect();
        Some(self.bootstrap_traded(set, &deltas))
    }

    /// The split instants every number in this run was measured against.
    fn split_bounds(&self) -> (i64, i64) {
        self.eval.promotion.sampler.split_bounds()
    }

    /// `(symbol, calendar month)` block id of every window in a pinned set, so windows of
    /// one ticker inside one month count as a single draw.
    fn blocks(&self, set: &PinnedSet) -> Vec<u64> {
        pinned_blocks(set)
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

    /// The trading bench over the traded prefix of a pinned set.
    ///
    /// `evaluate` fills its budget in `set.windows` order, chunk by chunk, so the traded
    /// windows are exactly the first `trade_paths.len()` of the set and the block ids line
    /// up by truncation. The bootstrap blocks by `(symbol, month)`, the same blocking the
    /// NLL interval uses: bars inside a window are autocorrelated and windows inside one
    /// ticker-month overlap, so neither is an independent draw.
    fn trade(&self, set: &PinnedSet, stats: &EvalStats) -> TradeBench {
        let mut blocks = self.blocks(set);
        blocks.truncate(stats.trade_paths.len());
        trade_bench::bench(
            &stats.trade_paths.windows,
            &blocks,
            &stats.trade_paths.tail,
            BenchConfig::new(
                trade_bench::DEFAULT_COST_BPS,
                trade_bench::LEVERAGE_CAP,
                // The null is the fitted marginal's UNCAPPED optimum, so the cap curve
                // re-clamps it at every cap instead of comparing against a frozen null.
                trade_bench::marginal_position(&self.supports_dev, trade_bench::FREE_LEVERAGE),
            ),
        )
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

    /// Fraction of each ramp stage's ASSIGNED share of the current pass that has been issued.
    ///
    /// In-progress rather than final: a periodic validation lands mid-pass, and a chart that
    /// only moved at boundaries could not show a stage stalling. The denominator is the stage's
    /// partition, so a completed pass reads exactly 1.0 at every stage — the old series
    /// divided by the stage's whole stride-`C` anchor list and topped out at 0.20 / 0.34 / 0.47.
    fn stage_coverage_fractions(&self) -> Vec<f64> {
        (0..RAMP_STAGES)
            .map(|stage| {
                let assigned = self.pass_layout.windows(stage).len().max(1);
                self.stage_cursor[stage] as f64 / assigned as f64
            })
            .collect()
    }

    /// Reconcile the pass that just ended and REFUSE to continue if it was not a full pass.
    ///
    /// The one exception is `--steps`, which explicitly decouples the step count from the
    /// corpus and is documented as diagnostic. Even there the shortfall is printed in full;
    /// what changes is only whether it is fatal.
    fn finish_pass(&mut self, step: usize) -> Result<CoverageAudit> {
        let audit = self.pass.audit(&self.pass_layout, &self.pass_ledger);
        println!("step {step}: {}", audit.summary());
        // The auxiliary passes are audited on the SAME terms. An auxiliary shortfall means the
        // Bresenham firing rule and the auxiliary pass geometry disagree, which would silently
        // train on a fraction of the daily corpus while the run claimed a pass over it.
        self.finish_auxiliary_passes(step)?;
        match audit.require_full_pass() {
            Ok(()) => {
                self.completed_passes += 1;
                // Absorbed only on a pass the audit CERTIFIED complete, because the
                // cross-pass reconstruction's exactness rests on exactly that: "every block
                // issued once, one contiguous hole skipped" is what makes a hole start
                // sufficient to rebuild per-bar exposure. A truncated pass folded in here
                // would be recorded as a full one and would OVERSTATE reuse.
                self.census.absorb(&self.pass_layout);
                Ok(audit)
            }
            Err(err) if self.args.steps.is_some() => {
                println!(
                    "WARNING step {step}: {err:#}. NOT fatal only because --steps was given, \
                     which decouples the schedule from the corpus by design. Every number this \
                     run reports is measured on the fraction of the corpus above, not on the \
                     whole training split."
                );
                // Deliberately NOT absorbed: this pass was incomplete, so the run's exposure
                // history stays honest about it and `cover_run_bar_exposure` keeps the
                // unissued bars at their true lower exposure instead of crediting a pass that
                // did not happen.
                Ok(audit)
            }
            Err(err) => Err(err),
        }
    }

    /// Reconcile every auxiliary pass, measure each one's held-out NLL, and write the row.
    ///
    /// The held-out measurement is what makes the auxiliary curve a measurement rather than an
    /// assertion, and it is taken HERE, at the pass boundary, so it is paired with the training
    /// mean over exactly the same pass. It is never compared to the deployment number and never
    /// consulted by promotion.
    fn finish_auxiliary_passes(&mut self, step: usize) -> Result<()> {
        if self.aux.is_empty() {
            return Ok(());
        }
        let eval_batch = self.args.batch_size;
        let scoring = self.args.scoring;
        for index in 0..self.aux.len() {
            let res = self.aux[index].res_secs();
            let audit = self.aux[index].audit();
            println!("step {step}: [aux {res}s] {}", audit.summary());
            let train_nll = self.aux[index].take_epoch_nll();
            let held_out = evaluate(
                &self.modules,
                self.aux[index].supports_dev(),
                &self.aux_heldout[index],
                eval_batch,
                self.device,
                false,
                scoring,
                None,
                trade_bench::TRADE_WINDOWS,
            )
            .with_context(|| format!("failed evaluating the {res}s auxiliary held-out set"))?;
            println!(
                "step {step}: [aux {res}s] train nll_bar {train_nll:.4}, held-out nll_bar \
                 {:.4} (conditional {:.4}) over {} windows at {AUXILIARY_HELDOUT_CONTEXT} bars \
                 — training signal only, not a promotion criterion",
                held_out.nll_bar,
                held_out.nll_bar_conditional,
                self.aux_heldout[index].windows.len()
            );
            self.aux_report
                .record(index, train_nll, held_out.nll_bar_conditional);
            if let Err(err) = audit.require_full_pass() {
                ensure!(
                    self.args.steps.is_some(),
                    "{err:#} (the {res}s auxiliary corpus)"
                );
                println!(
                    "WARNING step {step}: [aux {res}s] {err:#}. NOT fatal only because --steps \
                     was given."
                );
            }
        }
        self.aux_report
            .write_report(&self.run.gens)
            .context("failed writing the auxiliary resolution report")?;
        Ok(())
    }

    /// Run each auxiliary resolution's share of step `step`.
    ///
    /// Fires on the Bresenham cadence so a full auxiliary pass completes in exactly the same
    /// number of primary steps one primary pass takes, spread evenly. Every draw returning
    /// `None` while the cadence says fire is a real error, not a quiet skip: it means the
    /// auxiliary partition and the firing rule disagree and the pass would come up short.
    fn auxiliary_steps(&mut self, step: usize) -> Result<()> {
        if self.aux.is_empty() {
            return Ok(());
        }
        let primary_steps = self.schedule.steps_per_epoch;
        let device = self.device;
        for index in 0..self.aux.len() {
            if !self.aux[index].fires_after(step, primary_steps) {
                continue;
            }
            let Some((stage, sample, drawn)) = self.aux[index].draw(device) else {
                // Only reachable under `--steps`, which can end a pass mid-partition.
                continue;
            };
            let loss = self.optimizer_step(&sample, step, Some(index))?;
            let context = self.aux[index].context(stage);
            self.aux[index].record_step(loss.nll_bar);
            self.aux_steps += 1;
            self.aux_bars_seen += drawn as u64 * context as u64;
        }
        Ok(())
    }

    /// Advance to the next pass: a fresh geometry, a fresh ledger, and all three cursors back
    /// to zero because a new epoch is a NEW partition of the same bars.
    fn begin_pass(&mut self, epoch: usize) {
        self.epoch = epoch;
        self.pass_layout = self.pass.layout(epoch);
        self.pass_ledger = PassLedger::new(&self.pass_layout);
        self.stage_cursor = [0; RAMP_STAGES];
        for stream in &mut self.aux {
            stream.begin_pass(epoch);
        }
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
            // Attached by the epoch-boundary writer, which is the only caller that has a
            // bench for these exact windows.
            trade: None,
            // The batch the run EXECUTED, not the one it asked for: the call site overwrites
            // `args.batch_size` with the capacity probe's verdict before the trainer is built,
            // so this is the clamped figure. `pretrain-compare` refuses a pair whose two arms
            // disagree on it — see `WindowScores::realized_batch`.
            realized_batch: Some(self.args.batch_size),
            realized_steps: Some(self.schedule.total_steps),
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
        record: SelectionRecord,
    ) -> Result<PathBuf> {
        let candidate = self.run.weights.join("pretrain_promotion_candidate.ot");
        // The context the decision is being taken at, recorded in the artifact itself: a
        // checkpoint selected on the diagnostic set must not claim it was selected at the
        // deployed context. `record` carries what the two criteria actually read, so the file
        // states the trade-off the choice made and not only the rule that made it.
        let metadata = self.write_checkpoint(
            &candidate,
            record.step,
            self.promotion_set(target).context,
            Some(record),
        )?;
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
            None,
            trade_bench::TRADE_WINDOWS,
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
            "promoted {} — step {}, selection edge@{SELECTION_CAP:.2}x {:+.4} +/- {:.4} bps/bar \
             and conditional nll {:.4} nats/bar, both measured at {} bars; deployed-context \
             held-out {:.4} nats/bar under {} scoring, {:+.4} vs the marginal baseline {:.4} \
             and {:+.4} vs uniform {:.4} (lineage {})",
            best.display(),
            record.step,
            record.edge_bps,
            record.edge_se_bps,
            record.nll_conditional,
            record.bench_context,
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

    /// Ancestral candle-rollout pictures of `checkpoint`, on the pinned snapshot windows.
    ///
    /// Called once per epoch boundary with that boundary's OWN artifact, so the picture
    /// series is one model per epoch on one fixed scene, and flipping between two of them
    /// shows the fan tighten or the calibration move and nothing else. Two properties make
    /// that true and both are enforced rather than assumed:
    ///
    /// * The depicted weights are the epoch's, not the incumbent's. This used to picture
    ///   `pretrain_best.ot` and return early when no promotion had happened yet, which is
    ///   why a 13831-step run left exactly ONE snapshot: promotion is gated on the deployed
    ///   context and cannot fire for the first two thirds of a run. Worse, once it could,
    ///   consecutive boundaries with no promotion between them depicted byte-identical
    ///   weights and read as a model that had stopped moving.
    /// * The scene does not move. `eval.snapshot` is drawn once under [`EVAL_WINDOW_SEED`]
    ///   and never redrawn, and the fingerprint is re-checked here rather than trusted: a
    ///   moved window set makes a tightening fan indistinguishable from an easier window,
    ///   and nothing in the picture would say which had happened.
    fn write_snapshot(&mut self, step: usize, checkpoint: &Path) -> Result<()> {
        let fingerprint = pinned_fingerprint(&self.eval.snapshot);
        ensure!(
            fingerprint == self.snapshot_window_fingerprint,
            "the pinned snapshot windows MOVED between epoch boundaries (fingerprint \
             {fingerprint:#018x} at step {step}, {:#018x} at the start of the run). Every \
             epoch's pictures must depict the identical scene, or the epoch-over-epoch \
             comparison they exist for is measuring two things at once.",
            self.snapshot_window_fingerprint,
        );
        let metadata = world_model_metadata_path(checkpoint);
        let world = BarWorldModel::load(checkpoint, &metadata, self.device).with_context(|| {
            format!(
                "the epoch artifact {} could not be reloaded to picture it",
                checkpoint.display()
            )
        })?;
        let window = self.snapshot_windows();
        let rollout = rollout_pinned_windows(&world, &window, self.args.snapshot_samples);
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
    /// A SAFETY NET FOR CONTENTION, not the schedule. The ramp is derived from measured
    /// capacity before step 0 by [`CapacityModel::derive_batch_ramp`], so on a card whose
    /// occupancy has not changed since the probe this returns without firing: every step-up
    /// left in the plan was already shown to fit. It fires when another tenant took memory
    /// after the probe — which is the only thing it can honestly protect against, and is a
    /// real event on this card. Before the derivation existed it fired on EVERY run, silently
    /// rewriting a declared `batch 24->72` into a flat 24, which is what made the announced
    /// schedule fiction.
    ///
    /// The CONTEXT ramp is never held: the deployed model is selected and promoted at the
    /// full context, so a run that never trains there is not the run we asked for. The
    /// batch is the part that only buys gradient-noise reduction, so it is the part that
    /// yields. Holding it also moves the learning-rate plateau bump, which
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
        // The realized rate, not the plateau ratio: stage 2 starts at progress 2/3, past the
        // default plateau fraction, so by then the schedule is already on the linear decay and
        // the sqrt bump is partly annealed away. Quoting `sqrt(held)/sqrt(planned)` there
        // would name a change the run never experiences.
        let realized = self.schedule.lr_multiplier(step);
        let planned_rate = self.schedule.lr_multiplier_for(step, planned);
        let exponent = BATCH_RAMP_LR_EXPONENT[stage];
        println!(
            "WARNING step {step}: the learning-rate multiplier follows the batch actually \
             used, so from this step it is {realized:.3}x rather than the planned \
             {planned_rate:.3}x — a {:+.1}% change to every parameter group's rate. The \
             plateau bump is {held}**{exponent} = {:.3}x instead of {planned}**{exponent} = \
             {:.3}x; this step is {} the flat plateau.",
            100.0 * (realized / planned_rate - 1.0),
            (held as f64).powf(exponent),
            (planned as f64).powf(exponent),
            if self.schedule.in_lr_plateau(step) {
                "inside"
            } else {
                "past"
            },
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
    ///
    /// `growth` is watched by the same machinery and for the same reason, though it is the
    /// term least likely to trip it: its magnitude is ~5e-4 nats against `nll`'s ~4.93, so
    /// at [`LAMBDA_GROWTH`] its objective share is ~1e-4 and a reading above 25% would mean
    /// something had gone badly wrong with either the weight or the likelihood. Its WEIGHT
    /// was sized on gradient norm, not on objective share — see
    /// [`probe_growth_gradient_share`] — so this is a tripwire on the objective, not the
    /// sizing rule.
    fn warn_on_auxiliary_domination(
        &mut self,
        step: usize,
        dyn_share: f64,
        kl_share: f64,
        growth_share: f64,
    ) {
        let worst = dyn_share.max(kl_share).max(growth_share);
        if !worst.is_finite() || worst <= AUX_SHARE_WARN {
            self.aux_share_streak = 0;
            return;
        }
        self.aux_share_streak += 1;
        if self.aux_share_streak % AUX_SHARE_WARN_STREAK != 0 {
            return;
        }
        let (name, share, lambda) = if worst == growth_share {
            ("growth", growth_share, self.args.lambda_growth)
        } else if dyn_share >= kl_share {
            ("dyn", dyn_share, self.args.lambda_dyn)
        } else {
            ("kl", kl_share, self.args.lambda_kl)
        };
        println!(
            "WARNING step {step}: the {name} term has held {:.0}% of the objective's \
             magnitude — above the {:.0}% threshold — for {} consecutive steps. It is an \
             AUXILIARY term shaping the latent, not the learning signal; at its configured \
             lambda {:e} it is competing with the likelihood. Lower --lambda-{name} or \
             accept that this run is not a maximum-likelihood run. Nothing anneals it away \
             later: the configured weight applies for the whole run, by design.",
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
pub(super) struct EvalStats {
    pub(super) nll_bar: f64,
    nll_dof: [f64; BAR_DOF],
    /// `nll_bar` with the encoding tautology excluded. `encode_dof` sets `u = v = 0.5`
    /// whenever `s == 0`, and the chain predicts `s` first, so those two factors are free
    /// on a flat bar — worth ~0.69 nats/bar, roughly a fifth of the reported gain over the
    /// calibrated marginal. Here `u` and `v` are averaged only over bars with `s != 0`.
    pub(super) nll_bar_conditional: f64,
    nll_dof_conditional: [f64; BAR_DOF],
    /// Per-DOF split of the NLL into the degeneracy class and the continuous shape.
    nll_dof_class: [f64; BAR_DOF],
    nll_dof_shape: [f64; BAR_DOF],
    /// One entry per window of the set, in `set.windows` order. The whole point: a mean
    /// with no dispersion is not a measurement, and pairing two runs window by window is
    /// what makes an ablation detectable at 0.04-0.09 nats instead of 0.41.
    pub(super) window_nll: Vec<f64>,
    pub(super) window_nll_conditional: Vec<f64>,
    window_nll_dof: Vec<[f64; BAR_DOF]>,
    crps_dof: [f64; BAR_DOF],
    pit: PitHistogram,
    dir_acc: f64,
    effective_rank: f64,
    /// Per-DOF MARGINALIZED forecast NLL: every factor scored conditioning ONLY on strictly
    /// past bars, with the same-bar chain prefix marginalized over the head's own predictive
    /// law instead of teacher-forced on its realized value. See [`chunk_forecast`]. NaN
    /// unless `full`.
    forecast_nll_dof: [f64; BAR_DOF],
    /// Teacher-forced per-DOF NLL over EXACTLY the rows [`Self::forecast_nll_dof`] used, so
    /// the teacher-forcing inflation is a paired difference on identical data rather than
    /// two numbers measured on different row sets. NaN unless `full`.
    forecast_teacher_nll_dof: [f64; BAR_DOF],
    /// Monte-Carlo standard error of `forecast_nll_dof.iter().sum()`, from
    /// [`FORECAST_MC_GROUPS`] independent draw groups. NaN unless `full`.
    forecast_nll_se: f64,
    /// Per-window Kelly positions for the trading bench, over the first
    /// [`trade_bench::TRADE_WINDOWS`] windows of the set. Empty unless `full`.
    pub(super) trade_paths: ChunkPaths,
}

/// `(symbol, calendar month)` block id of every window in a pinned set, so windows of one
/// ticker inside one month count as a single bootstrap draw. Free-standing because the
/// standalone bench command has no [`Trainer`] and must block its interval identically.
pub(super) fn pinned_blocks(set: &PinnedSet) -> Vec<u64> {
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

/// Order-sensitive identity of a pinned window set: its context, its size, and every
/// `(symbol, bar_index)` it holds, in the order the evaluation walks them.
///
/// Order matters as much as membership. The trading bench trades the first
/// [`trade_bench::TRADE_WINDOWS`] windows of the set and the candle snapshots picture the
/// first few, so a permuted set is a different scene and a different bench even though it
/// holds the same windows.
fn pinned_fingerprint(set: &PinnedSet) -> u64 {
    let mut acc = mix64(set.context as u64, set.windows.len() as u64);
    for window in &set.windows {
        acc = mix64(acc, (u64::from(window.symbol) << 32) | u64::from(window.bar_index));
    }
    acc
}

/// Bar-tokens a run will have delivered at its last step, given what it has consumed
/// through `step` and the ramp the schedule is CURRENTLY carrying.
///
/// `bars_seen` already includes `step`, so only the steps after it are projected, and the
/// projection reads `schedule.batch_ramp` — which [`Trainer::hold_batch_if_short_of_vram`]
/// MUTATES. A run whose batch was held therefore reports the smaller number from its FIRST
/// epoch boundary instead of at the finish line, which is the whole point: `--epochs 3`
/// executed at a held batch is 1.33 passes over the corpus, and that has to be findable in
/// the first forty minutes rather than in hour forty.
///
/// Free-standing so the accounting can be checked against a hand-summed schedule without
/// building a trainer, a corpus and a 39M-parameter model.
fn projected_bar_tokens(schedule: &Schedule, bars_seen: u64, step: usize) -> u64 {
    let remaining: u64 = ((step + 1)..schedule.total_steps)
        .map(|future| schedule.bars_per_step(future))
        .sum();
    bars_seen + remaining
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

/// Ancestral rollout of every pinned window, `[W, samples, SNAPSHOT_HORIZON, BAR_DOF]`.
///
/// The rollout KV cache is `windows * samples` sequences deep, so a batched call over
/// every window at once would need tens of gigabytes. One window at a time keeps the
/// peak at a few, and the result is identical because each window's ancestral samples
/// are independent.
fn rollout_pinned_windows(
    world: &BarWorldModel,
    window: &SnapshotWindow,
    samples: usize,
) -> Tensor {
    let parts: Vec<Tensor> = (0..window.history_dof.size()[0])
        .map(|index| {
            world.rollout(
                &window.history_dof.narrow(0, index, 1),
                &window.history_time_ids.narrow(0, index, 1),
                &window.future_time_ids.narrow(0, index, 1),
                samples,
                1.0,
            )
        })
        .collect();
    Tensor::cat(&parts, 0)
}

/// The `full`-only diagnostics of one evaluation chunk.
struct ChunkExtras {
    crps: [f64; BAR_DOF],
    /// Per-element PIT values, still on the device.
    pit: Tensor,
    /// `(directional hits, comparisons)`.
    direction: (f64, f64),
    /// Belief participation ratio, measured on the first chunk only.
    rank: Option<f64>,
    class: [f64; BAR_DOF],
    shape: [f64; BAR_DOF],
    forecast: ChunkForecast,
    /// Traded windows contributed by this chunk, up to the pass's budget.
    trade: ChunkPaths,
}

/// The marginalized-forecast estimate of one chunk, on its strided row subset.
struct ChunkForecast {
    /// Rows the estimate covers, i.e. the weight of these means in the pooled figure.
    rows: f64,
    /// Per-DOF forecast NLL, marginalized over the same-bar prefix.
    forecast_dof: [f64; BAR_DOF],
    /// Per-DOF teacher-forced NLL on exactly those rows.
    teacher_dof: [f64; BAR_DOF],
    /// Per-group forecast totals, for the Monte-Carlo standard error.
    group_totals: [f64; FORECAST_MC_GROUPS],
}

/// The HONEST forecasting number beside the teacher-forced one, on the same rows.
///
/// `nll_bar` factorizes the bar as `p(r|h) p(s|h,r) p(u|h,r,s) p(v|..) p(w|..)` and evaluates
/// every factor at the realized prefix. That sum is the proper JOINT one-step-ahead
/// log-likelihood of the bar and stays the comparability anchor — but only its first chain
/// factor, `r`, is a forecast. `s` is scored already knowing the realized return, and `u`,
/// `v`, `w` are scored knowing everything ahead of them: those four terms are within-bar
/// accounting, not prediction.
///
/// Here each factor is scored against the marginalized predictive law
/// [`BarEmissionHead::forecast_log_probs`] — the head's own distribution over the same-bar
/// prefix, integrated out — so every one of the five terms conditions only on strictly past
/// bars. The sum is the code length of a forecaster that must emit the bar without being
/// told any part of it, and by subadditivity it is >= the joint, with equality exactly when
/// the chain factors are conditionally independent given `h`. The gap IS the teacher-forcing
/// inflation.
///
/// Both numbers are taken on every [`FORECAST_POSITION_STRIDE`]-th bar position. The stride
/// is what makes the estimator affordable at [`FORECAST_MC_DRAWS`] draws, and taking the
/// teacher-forced figure on the identical rows makes the difference a paired measurement
/// rather than a comparison of two subsets.
fn chunk_forecast(
    modules: &BarModules,
    supports: &BarSupports,
    beliefs: &Tensor,
    target: &Tensor,
    scoring: BarScoring,
    seed: u64,
) -> ChunkForecast {
    let positions = beliefs.size()[1];
    let stride = FORECAST_POSITION_STRIDE.min(positions.max(1));
    let beliefs = beliefs.slice(1, 0, positions, stride).contiguous();
    let target = target.slice(1, 0, positions, stride).contiguous();
    let rows = (beliefs.size()[0] * beliefs.size()[1]) as f64;
    let targets = supports.targets(&target, scoring);
    let teacher_dof = dof_mean(&bar_nll_terms(
        &modules.head.logits(&beliefs, &supports.bin_ids(&target)),
        &targets,
    ));

    let per_group = FORECAST_MC_DRAWS / FORECAST_MC_GROUPS;
    let mut group_totals = [0.0f64; FORECAST_MC_GROUPS];
    let mut pooled: Option<Tensor> = None;
    for (group, total) in group_totals.iter_mut().enumerate() {
        // Disjoint streams per group, so the spread across groups is an honest Monte-Carlo
        // standard error rather than the same draws counted several times.
        let log_mixture =
            modules
                .head
                .forecast_log_probs(&beliefs, per_group, mix64(seed, group as u64));
        *total = dof_mean(&bar_nll_terms(&log_mixture, &targets)).iter().sum();
        let probs = log_mixture.exp();
        pooled = Some(match pooled {
            Some(acc) => acc + probs,
            None => probs,
        });
    }
    // Equal-size groups, so the mean of the group mixtures IS the mixture over all
    // FORECAST_MC_DRAWS draws — the pooled figure is not the mean of the group figures,
    // which would carry the bias of a `per_group`-draw estimate.
    let pooled = pooled.expect("at least one forecast group") / FORECAST_MC_GROUPS as f64;
    let forecast_dof = dof_mean(&bar_nll_terms(
        &pooled.clamp_min(f32::MIN_POSITIVE as f64).log(),
        &targets,
    ));
    ChunkForecast {
        rows,
        forecast_dof,
        teacher_dof,
        group_totals,
    }
}

/// `[..., BAR_DOF]` per-factor nats reduced to one host mean per factor.
fn dof_mean(terms: &Tensor) -> [f64; BAR_DOF] {
    dof_array(
        &terms
            .reshape([-1, BAR_DOF as i64])
            .mean_dim([0i64].as_slice(), false, Kind::Float),
    )
}

/// Standard error of a mean estimated from `G` independent groups: `sd(groups) / sqrt(G)`.
///
/// Slightly CONSERVATIVE for the pooled figure, because each group estimate is built from
/// `1/G` of the draws and is therefore noisier than the pooled one. Overstating a
/// Monte-Carlo error bar is the safe direction: it can only make a difference look less
/// resolvable than it is.
fn group_standard_error(groups: &[f64]) -> f64 {
    if groups.len() < 2 {
        return f64::NAN;
    }
    let n = groups.len() as f64;
    let mean = groups.iter().sum::<f64>() / n;
    let variance = groups.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (variance / n).sqrt()
}

/// Teacher-forced evaluation over a pinned window set, in full precision so the
/// number is reproducible independently of the training autocast policy. `full` adds
/// the calibration diagnostics; promotion only needs the NLL, and the diagnostics
/// it does not compute are returned as NaN rather than zero.
///
/// `shrink` asks the trading bench to solve a SECOND log-optimal fraction per bar, under the
/// conditional mean recalibrated by an affine map fitted elsewhere — on windows disjoint from
/// this set, which is why it is an argument rather than something this function could derive.
/// `None` on every path but the calibration experiment, and the existing policies never read
/// it, so a run's headline numbers do not move when it is set.
///
/// `trade_budget` is how many of the set's windows retain per-bar trading paths, counted from
/// the front. It is an ARGUMENT rather than [`trade_bench::TRADE_WINDOWS`] read in place because
/// the constant is load-bearing for every economic number already published: 256 windows is the
/// prefix each of those was measured on, and a pass that wants more information out of a
/// held-out draw must be able to ask for it without moving anybody else's number. Every caller
/// but the calibration experiment passes the constant, so the default is bit-identical and
/// visible at each site.
///
/// The per-window vector is always retained, for both paths: it costs one `[B]` host
/// transfer per chunk and it is the only thing that makes the held-out mean a measurement
/// rather than a number.
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate(
    modules: &BarModules,
    supports: &BarSupports,
    set: &PinnedSet,
    batch: usize,
    device: Device,
    full: bool,
    scoring: BarScoring,
    shrink: Option<MeanShrink>,
    trade_budget: usize,
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
    // Forecast accumulators. Weighted by the STRIDED row count, which is the population the
    // marginalized estimate covers, not the chunk's window count.
    let mut forecast_rows = 0.0f64;
    let mut forecast_dof_sum = [0.0f64; BAR_DOF];
    let mut forecast_teacher_sum = [0.0f64; BAR_DOF];
    let mut forecast_group_sums = [0.0f64; FORECAST_MC_GROUPS];
    // The trading bench. Built once: the null's position is a property of the supports, so
    // re-deriving it per chunk would be 170 identical derivations and would leave open the
    // question of whether the null moved.
    let trade_setup = if full {
        Some(TradeSetup::new(supports, device, trade_bench::LEVERAGE_CAP).with_shrink(shrink))
    } else {
        None
    };
    let mut trade_paths = ChunkPaths::default();

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
                ChunkExtras {
                    crps,
                    pit: pit_values,
                    direction,
                    rank,
                    class: dof_array(&parts.class),
                    shape: dof_array(&parts.shape),
                    forecast: chunk_forecast(
                        modules,
                        supports,
                        &beliefs,
                        &target,
                        scoring,
                        mix64(EVAL_WINDOW_SEED, chunk_index as u64),
                    ),
                    // Only `r`, only from strictly past bars: `TradeSetup::paths` takes the
                    // beliefs and selects the realized `r` itself, so no part of the bar
                    // being bet on can reach the position. The budget makes the bench cost
                    // a fixed prefix of the pinned set rather than the whole pass.
                    trade: trade_setup
                        .as_ref()
                        .map(|setup| {
                            setup
                                .paths(
                                    &modules.head,
                                    &beliefs,
                                    &target,
                                    trade_budget.saturating_sub(trade_paths.len()),
                                )
                                .expect("the evaluation loop shapes its own beliefs and targets")
                        })
                        .unwrap_or_default(),
                }
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

        if let Some(extras) = extras {
            for (acc, value) in crps_dof_sum.iter_mut().zip(extras.crps) {
                *acc += value * rows;
            }
            for (acc, value) in class_dof_sum.iter_mut().zip(extras.class) {
                *acc += value * rows;
            }
            for (acc, value) in shape_dof_sum.iter_mut().zip(extras.shape) {
                *acc += value * rows;
            }
            pit.accumulate(&extras.pit);
            direction_correct += extras.direction.0;
            direction_total += extras.direction.1;
            if let Some(rank) = extras.rank {
                effective_rank = rank;
            }
            let forecast = extras.forecast;
            forecast_rows += forecast.rows;
            for (acc, value) in forecast_dof_sum.iter_mut().zip(forecast.forecast_dof) {
                *acc += value * forecast.rows;
            }
            for (acc, value) in forecast_teacher_sum.iter_mut().zip(forecast.teacher_dof) {
                *acc += value * forecast.rows;
            }
            for (acc, value) in forecast_group_sums.iter_mut().zip(forecast.group_totals) {
                *acc += value * forecast.rows;
            }
            trade_paths.absorb(extras.trade);
        }
        rows_total += rows;
    }

    ensure!(rows_total > 0.0, "evaluation set produced no windows");
    let scale = 1.0 / rows_total;
    let nll_dof = nll_dof_sum.map(|v| v * scale);
    let nll_dof_conditional = conditional_nll_dof(&nll_dof, &live_dof_sum, live_bars);
    let forecast_measured = full && forecast_rows > 0.0;
    let forecast_scale = if forecast_measured {
        1.0 / forecast_rows
    } else {
        f64::NAN
    };
    Ok(EvalStats {
        nll_bar: nll_dof.iter().sum(),
        nll_dof,
        nll_bar_conditional: nll_dof_conditional.iter().sum(),
        nll_dof_conditional,
        trade_paths,
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
        forecast_nll_dof: forecast_dof_sum.map(|v| v * forecast_scale),
        forecast_teacher_nll_dof: forecast_teacher_sum.map(|v| v * forecast_scale),
        forecast_nll_se: if forecast_measured {
            group_standard_error(&forecast_group_sums.map(|v| v * forecast_scale))
        } else {
            f64::NAN
        },
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
    // Held-out mass the training fit assigned literally zero probability, per DOF. Capped
    // below rather than charged `-ln(f64::MIN_POSITIVE)`, and REPORTED: a cap that fires on
    // a material share is a statement about the supports, not a rounding detail.
    let mut capped = [0.0f64; BAR_DOF];
    let out = std::array::from_fn(|dof| {
        let row = &q_val[dof * bins as usize..(dof + 1) * bins as usize];
        let train = supports.reference_row(dof, scoring);
        let widths = supports.widths(dof);
        row.iter()
            .enumerate()
            .filter(|(_, observed)| **observed > 0.0)
            .map(|(bin, observed)| {
                let share = observed / rows_total;
                // The reference row is normalized, so a zero entry means the training fit
                // assigned a held-out outcome literally zero mass. Charge it the
                // UNIFORM-FLOOR surprise, `ln(NUM_BAR_BINS)`. `f64::MIN_POSITIVE` would
                // charge 708 nats — the infinity this guard exists to avoid, wearing a
                // finite number's clothes — and would silently inflate the one baseline the
                // banner calls "distribution shift".
                let surprise = if train[bin] > 0.0 {
                    -train[bin].ln()
                } else {
                    capped[dof] += share;
                    (NUM_BAR_BINS as f64).ln()
                };
                // An atom bin has zero width and carries a MASS, so it takes no correction.
                let measure = if density && widths[bin] > 0.0 {
                    widths[bin].ln()
                } else {
                    0.0
                };
                share * (measure + surprise)
            })
            .sum()
    });
    for (dof, share) in capped.iter().enumerate() {
        if *share > 0.0 {
            println!(
                "warning: {:.4}% of held-out {} observations landed in a bin the train fit \
                 gave zero mass; each is charged the uniform-floor surprise {:.4} nats \
                 rather than infinity, so the marginal-on-val line for {} is a LOWER bound",
                100.0 * share,
                BAR_DOF_NAMES[dof],
                (NUM_BAR_BINS as f64).ln(),
                BAR_DOF_NAMES[dof],
            );
        }
    }
    Ok(out)
}

/// Directional accuracy of the model's return sign at the final position of each window —
/// the one position that is a genuine next-bar forecast rather than a mid-sequence
/// conditional. `r` heads the chain, so `BarEmissionHead::sample` draws it from `p(r|h)`
/// before any same-bar factor exists, and the majority sign over `DIRECTION_SAMPLES` draws
/// is a statistic of that law alone.
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

/// NextLat residual between a predicted belief and its stop-gradient target, reduced
/// exactly as the reference implementation does: `smooth_l1` MEANED over every element of
/// `[B, T, BAR_MODEL_DIM]`.
///
/// This is `models/model_nextlat.py:303-308` of the NextLat codebase (arXiv 2511.05963),
/// whose own comment reads "Same as reduction='mean' over masked (B, T, n_embd) elements,
/// i.e., divide over B*T*n_embd elements", paired with `defaults.yaml: lambda_mse: 1.0`.
///
/// It was briefly changed to SUM the feature axis on the theory that a per-token `dyn` is
/// commensurate with a per-token `nll` and that the mean form made `lambda_dyn` inert. That
/// reasoning is wrong twice. It multiplies the term by `BAR_MODEL_DIM`, so `lambda_dyn =
/// 1.0` became 512x the reference and the NextLat term took 62% of the objective while
/// `nll` rose from 16.34 to 17.19 over 4000 steps. And it makes the knob width-dependent:
/// the same `lambda_dyn` would mean something different the moment `BAR_MODEL_DIM` moves,
/// which is exactly the property a swept hyperparameter must not have. Commensurability is
/// `lambda_dyn`'s job, not the reduction's.
fn next_lat_loss(predicted: &Tensor, target: &Tensor) -> Tensor {
    predicted
        .smooth_l1_loss(target, Reduction::Mean, 1.0)
}

/// One optimizer step's graph, still attached.
struct TrainingGraph {
    loss: Tensor,
    nll: Tensor,
    nll_dof: Tensor,
    dyn_loss: Tensor,
    kl_loss: Tensor,
    /// Mean `-log(1 + f_hat R)` under the deployed cap, from [`growth::growth_loss`].
    growth: Tensor,
    /// `[GROWTH_STAT_COUNT]` detached growth diagnostics; see [`growth::GrowthStats`].
    growth_stats: Tensor,
    identity: Tensor,
    autocorr: Tensor,
}

/// The training objective's forward graph: one trunk pass, the teacher-forced likelihood, the
/// two NextLat auxiliaries and the expected-log-growth term, each at its configured weight.
///
/// Call inside [`autocast`]. Shared verbatim by [`Trainer::optimizer_step`] and
/// [`probe_capacity`], which is the whole point of it being a function: a capacity probe that
/// measured a DIFFERENT graph would derive a ramp for a model this run does not train, and
/// the ramp is now the schedule rather than an aspiration the memory gate quietly rewrites.
/// The growth term is therefore computed at `lambda_growth = 0` too — it costs the ablation's
/// control arm ~2% of a step and it is what makes the two arms' `pretrain_growth_term` charts
/// a comparison rather than one curve and one blank panel.
#[allow(clippy::too_many_arguments)]
fn forward_losses(
    modules: &BarModules,
    supports: &BarSupports,
    growth_support: &GrowthSupport,
    dof: &Tensor,
    time_ids: &Tensor,
    context: i64,
    horizon: i64,
    lambda_dyn: f64,
    lambda_kl: f64,
    lambda_growth: f64,
    scoring: BarScoring,
    device: Device,
) -> TrainingGraph {
    let input = dof.narrow(1, 0, context);
    let target = dof.narrow(1, 1, context);
    // `prepare`/`locate` are elementwise, so binning commutes with narrowing: one pass over
    // `[B, T + 1, BAR_DOF]` serves the trunk's input, the head's teacher-forced target and
    // every dynamics horizon. Each pass materializes an `[N, BAR_DOF, NUM_BAR_BINS]`
    // comparison tensor, so this is worth hoisting even though it is small beside the
    // transformer.
    let bins = supports.bin_ids(dof);
    // One transformer pass. Every dynamics horizon reuses this belief sequence, shifted, so
    // recursion costs only MLP evaluations.
    let beliefs = modules.trunk.forward(
        &input,
        &bins.narrow(1, 0, context),
        &time_ids.narrow(1, 0, context),
        0,
        true,
    );

    let logits = modules.head.logits(&beliefs, &bins.narrow(1, 1, context));
    // The objective and every reported baseline read the same `--scoring`.
    let (nll, nll_dof) = bar_nll_from_logits(&logits, &supports.targets(&target, scoring));

    let (dyn_loss, kl_loss, identity) = dynamics_losses(
        modules, dof, &bins, time_ids, &beliefs, context, horizon, device,
    );
    // The SAME beliefs the likelihood is scored from, and the realized log return of the bar
    // each of them predicts. Deliberately NOT `logits`: this hands the head only the causal
    // belief, so no part of the realized bar can reach the position. `r` heads the chain, so
    // its row is a forecast to begin with, and `growth::verify_traded_law` proves that on
    // this head before the first step.
    let growth::Growth {
        loss: growth,
        stats: growth_stats,
    } = growth::growth_loss(
        &modules.head,
        &beliefs,
        &target.select(-1, DOF_R as i64),
        growth_support,
    );
    let autocorr = belief_autocorrelation(&beliefs);
    let loss = &nll + lambda_dyn * &dyn_loss + lambda_kl * &kl_loss + lambda_growth * &growth;
    TrainingGraph {
        loss,
        nll,
        nll_dof,
        dyn_loss,
        kl_loss,
        growth,
        growth_stats,
        identity,
        autocorr,
    }
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
        // Teacher-forced bar t+k, with its calendar, advances the latent one step. The market
        // channels are dropped: `BarDynamics::step` is only ever CALLED on an imagined bar, whose
        // market row is unknowable, so training it on the realized one would fit a channel that
        // is `MARKET_MISSING` at every deployment call site.
        let advance = dof.narrow(1, k, anchors);
        let advance_time = time_ids_without_market(&time_ids.narrow(1, k, anchors));
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

/// The NextLat residual of the dynamics head over the trivial-identity baseline, measured
/// on a pinned set in full precision under `no_grad`.
///
/// The step-time `dyn_vs_identity` reads the same ratio off the training graph, but that one
/// is an autocast bf16 number on the current training batch and it stops existing the moment
/// the run ends. This is the end-of-run measurement on the RELOADED promoted checkpoint and
/// the held-out split, which is the artifact and the data a guard has to speak about.
///
/// Weighted by window count so the last short chunk cannot outvote the full ones, and summed
/// as two separate totals rather than as a mean of per-chunk ratios: the ratio of sums is the
/// pooled quantity, a mean of ratios is not.
fn dyn_identity_ratio(
    modules: &BarModules,
    supports: &BarSupports,
    set: &PinnedSet,
    batch: usize,
    horizon: i64,
    device: Device,
) -> Result<f64> {
    ensure!(
        horizon < set.context,
        "--dyn-horizon {horizon} does not fit in the {}-bar evaluation context",
        set.context
    );
    let mut dyn_sum = 0.0f64;
    let mut identity_sum = 0.0f64;
    for chunk in set.windows.chunks(batch.max(1)) {
        let sample = set.sampler.batch_of(chunk, device);
        let context = sample.dof.size()[1] - 1;
        ensure!(
            horizon < context,
            "--dyn-horizon {horizon} does not fit in a {context}-bar evaluation window"
        );
        let (chunk_dyn, chunk_identity) = tch::no_grad(|| {
            let bins = supports.bin_ids(&sample.dof);
            // `train = false`, so the trunk runs detached and the dynamics terms below carry
            // no graph. Full precision, unlike the training step: the guard's threshold is
            // 1.0 and a bf16 rounding of a ratio near it would decide the run.
            let beliefs = modules.trunk.forward(
                &sample.dof.narrow(1, 0, context),
                &bins.narrow(1, 0, context),
                &sample.time_ids.narrow(1, 0, context),
                0,
                false,
            );
            let (dyn_loss, _, identity) = dynamics_losses(
                modules,
                &sample.dof,
                &bins,
                &sample.time_ids,
                &beliefs,
                context,
                horizon,
                device,
            );
            (dyn_loss.double_value(&[]), identity.double_value(&[]))
        });
        let weight = chunk.len() as f64;
        dyn_sum += chunk_dyn * weight;
        identity_sum += chunk_identity * weight;
    }
    // A zero baseline means the beliefs never move, so there is nothing for the head to
    // predict and no ratio to report. NaN propagates to the caller's finiteness check rather
    // than being papered over with a passing number.
    Ok(if identity_sum > 0.0 {
        dyn_sum / identity_sum
    } else {
        f64::NAN
    })
}

/// The end-of-run verdict on a measured `dyn / identity`: `Ok` iff the shipped dynamics
/// head is at least as good as the trivial `z_k = h_t` identity map.
///
/// Separated from [`dyn_identity_ratio`] because the measurement needs a corpus-backed
/// pinned set and the verdict does not, and because the verdict is the part that decides
/// whether a run's artifact is publishable.
fn check_dynamics_beats_identity(ratio: f64, horizon: i64) -> Result<()> {
    ensure!(
        ratio.is_finite(),
        "the promoted checkpoint's dyn/identity ratio is {ratio} on the test split, so the \
         shipped dynamics head cannot be certified against the trivial-identity baseline at \
         all. A non-finite ratio means the baseline is degenerate — the trunk's beliefs do \
         not move across the {horizon}-bar horizon — which is a collapsed trunk, not a \
         healthy one."
    );
    ensure!(
        ratio <= 1.0,
        "the promoted checkpoint's dynamics head is WORSE THAN DOING NOTHING: dyn/identity \
         is {ratio:.3} on the test split at horizon {horizon}, where 1.0 is the trivial \
         `z_k = h_t` identity map. BarDynamics ships inside the checkpoint and \
         RolloutMode::Dynamics advances beliefs through it, so this artifact would hand a \
         planner a latent predictor that degrades the belief it is asked to advance. The \
         cause is almost always that the dynamics head stopped receiving gradient while the \
         trunk kept training — check that --lambda-dyn and --lambda-kl are non-zero and \
         that nothing scales them down over the run."
    );
    Ok(())
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
        // Cast the INPUTS, not the result. Under bf16 autocast a similarity rounded before
        // the mean has 0.0039 resolution at `cos ~ 1`, which is exactly the resolution this
        // diagnostic exists to provide.
        let current = beliefs.narrow(1, 0, steps - 1).detach().to_kind(Kind::Float);
        let next = beliefs.narrow(1, 1, steps - 1).detach().to_kind(Kind::Float);
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
///
/// `nll` here is the SCORING-INVARIANT likelihood scale, i.e. the density rule's measure
/// constant already removed by the caller. That constant is a property of the binning that
/// no prediction can move and no gradient touches, so leaving it in the denominator would
/// make [`AUX_SHARE_WARN`] mean something different under each rule — and worst under the
/// default one, where a zero-init head starts near `-0.54` nats and any auxiliary term
/// would read as 80%+ of an objective it is not remotely dominating.
///
/// Four terms now, and `growth` is one of them for exactly the same reason `dyn` is: it
/// carries a weight that had to be sized against `nll`, so a chart of absolute curves
/// cannot show it taking over. The shares are returned in objective order —
/// `(nll, dyn, kl, growth)`.
fn loss_shares(
    nll: f64,
    weighted_dyn: f64,
    weighted_kl: f64,
    weighted_growth: f64,
) -> (f64, f64, f64, f64) {
    let total =
        nll.abs() + weighted_dyn.abs() + weighted_kl.abs() + weighted_growth.abs();
    if !(total > 0.0) || !total.is_finite() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    (
        nll.abs() / total,
        weighted_dyn.abs() / total,
        weighted_kl.abs() / total,
        weighted_growth.abs() / total,
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

/// Steps at which a run MEASURES the growth term's share of the total gradient norm.
///
/// Two, because the ratio is not constant: at initialization the head is zero-init and every
/// categorical is uniform, while by a couple of hundred steps the directional structure is
/// partly learned and `var_hat` — the `1/var_hat` in `df_raw/dmu_hat` — has shrunk by an
/// order of magnitude. [`LAMBDA_GROWTH`] was chosen to sit inside the 10-20% band at BOTH,
/// which is a stronger property than hitting a target at one of them, and the run reprints
/// the measurement so a corpus or architecture change that invalidates the constant is
/// visible in the log rather than in an ablation six hours later.
///
/// 200 is after [`RAMP_PROBE_AFTER_STEPS`], so the second probe cannot bias the activation
/// footprint the batch ramp is derived from.
const GROWTH_PROBE_STEPS: [usize; 2] = [0, 200];

/// Target share of the total gradient norm the growth term should carry.
///
/// The middle of the briefed 10-20% band. Below 10% the term is noise beside `nll`'s
/// minibatch variation; above 20% it starts buying economics with density, and NLL STAYS
/// PRIMARY — the density is what makes the model useful for anything beyond the sign.
const GROWTH_GRADIENT_TARGET_SHARE: f64 = 0.15;

/// A measured gradient-norm split between the likelihood and the growth term.
struct GrowthGradientShare {
    /// `||d(nll)/dtheta||` over every trainable parameter.
    nll_norm: f64,
    /// `||d(growth)/dtheta||` at `lambda_growth = 1`, so the number is a property of the
    /// term and not of the current weight.
    unit_growth_norm: f64,
}

impl GrowthGradientShare {
    /// Share of `||g_nll|| + lambda ||g_growth||` carried by the growth term.
    ///
    /// A sum of norms rather than the norm of the sum, deliberately: the two gradients are
    /// not orthogonal and the norm of their sum can be SMALLER than either, which would
    /// make "share" read above one. What has to be sized is how much signal the term
    /// injects, and that is its own norm against the primary's.
    fn share(&self, lambda: f64) -> f64 {
        let weighted = lambda * self.unit_growth_norm;
        let total = self.nll_norm + weighted;
        if total > 0.0 {
            weighted / total
        } else {
            f64::NAN
        }
    }

    /// The weight that would put the term at `target` of the total.
    fn lambda_for(&self, target: f64) -> f64 {
        if self.unit_growth_norm > 0.0 && target < 1.0 {
            target * self.nll_norm / ((1.0 - target) * self.unit_growth_norm)
        } else {
            f64::NAN
        }
    }

    fn report(&self, step: usize, lambda: f64) {
        println!(
            "growth gradient probe at step {step}: ||g_nll|| {:.4e}, ||g_growth|| {:.4e} at \
             lambda 1, so lambda_growth {lambda:e} holds {:.1}% of the total gradient norm. \
             {:.0}% would need lambda {:.3}. The shipped default is {LAMBDA_GROWTH:e}; see \
             growth::LAMBDA_GROWTH for the measurement it was derived from.",
            self.nll_norm,
            self.unit_growth_norm,
            100.0 * self.share(lambda),
            100.0 * GROWTH_GRADIENT_TARGET_SHARE,
            self.lambda_for(GROWTH_GRADIENT_TARGET_SHARE),
        );
    }
}

/// Measure `||d(nll)/dtheta||` and `||d(growth)/dtheta||` separately on THIS batch.
///
/// # Why a separate forward per term rather than one graph
///
/// The objective's own graph would need two retained backward passes through it, and the
/// second would then have to be told not to free what the real step still needs. Two clean
/// forwards cost two extra trunk passes at two steps of a 30,000-step run — well under a
/// thousandth of the wall clock — and they cannot interact with the step that follows.
///
/// # Why this cannot perturb the run
///
/// The training forward consumes no RNG (there is no dropout anywhere in the bar trunk or
/// the emission head), no optimizer step is taken, no weight is written, and the gradients
/// the two backwards accumulate are zeroed before returning — after which `optimizer_step`
/// zeroes them again before its own backward. The allocator pool is released so a probe
/// cannot inflate the `used` reading [`Trainer::probe_activation_footprint`] takes at
/// [`RAMP_PROBE_AFTER_STEPS`]. Both ablation arms run the identical code path at the
/// identical steps, so even a residual effect is common to both.
#[allow(clippy::too_many_arguments)]
fn probe_growth_gradient_share(
    vs: &nn::VarStore,
    modules: &BarModules,
    supports: &BarSupports,
    growth_support: &GrowthSupport,
    dof: &Tensor,
    time_ids: &Tensor,
    context: i64,
    horizon: i64,
    scoring: BarScoring,
    device: Device,
) -> Result<GrowthGradientShare> {
    let zero_grads = || {
        for mut variable in vs.trainable_variables() {
            variable.zero_grad();
        }
    };
    // `lambda_* = 0`: the probe reads the UNWEIGHTED terms off the graph, so the numbers it
    // prints are properties of the objective's pieces and not of the weights in force.
    let measure = |name: &str, want_growth: bool| -> Result<f64> {
        zero_grads();
        let graph = autocast(device.is_cuda(), || {
            forward_losses(
                modules,
                supports,
                growth_support,
                dof,
                time_ids,
                context,
                horizon,
                0.0,
                0.0,
                0.0,
                scoring,
                device,
            )
        });
        let term = if want_growth { &graph.growth } else { &graph.nll };
        let value = term.double_value(&[]);
        ensure!(
            value.is_finite(),
            "the growth gradient probe measured a non-finite {name} term: {value}"
        );
        term.backward();
        let norm = global_grad_norm(vs, device);
        ensure!(
            norm.is_finite(),
            "the growth gradient probe measured a non-finite {name} gradient norm: {norm}"
        );
        Ok(norm)
    };
    let nll_norm = measure("nll", false)?;
    let unit_growth_norm = measure("growth", true)?;
    zero_grads();
    if device.is_cuda() {
        crate::torch::cuda::empty_cache();
    }
    ensure!(
        unit_growth_norm > 0.0,
        "the growth term reached no trainable parameter: its gradient norm is zero, so it \
         is decoration rather than an objective. The likely cause is a detach on the path \
         from the emission head's r factor back to the trunk."
    );
    Ok(GrowthGradientShare {
        nll_norm,
        unit_growth_norm,
    })
}

/// The measured ceiling, the ramp it produced, and the batch/context frontier.
///
/// Three facts, all first-class rather than inferable. The measured cost per bar-token, so
/// nobody has to reverse-engineer it from an OOM. The batch each ramp stage will run and the
/// headroom left at the deployed context, so a run whose plan is at the ceiling says so. And
/// the achievable batch at each of [`CONTEXT_FRONTIER`], because at a fixed cost per
/// bar-token the card caps the PRODUCT of batch and context: they trade off directly, larger
/// batches are what the ported modded-nanogpt recipe is tuned for, and that tradeoff was
/// previously invisible. REPORTED ONLY — the deployed context is part of the world-model
/// contract the planner depends on and nothing here changes it.
fn print_capacity_banner(
    schedule: &Schedule,
    capacity: Option<&CapacityModel>,
    requested_batch: usize,
) {
    let Some(capacity) = capacity else {
        println!(
            "capacity       UNMEASURED — no CUDA device or no NVML, so the ramp keeps the \
             DECLARED ceiling x{BATCH_RAMP:?} and the only protection is the runtime hold at \
             each stage transition. Every number below that depends on device memory is \
             absent, not zero."
        );
        return;
    };
    let deployed = stage_context(RAMP_STAGES - 1);
    let stages: Vec<String> = (0..RAMP_STAGES)
        .map(|stage| {
            let batch = schedule.base_batch * schedule.batch_ramp[stage];
            format!(
                "stage {stage}: {batch}x{} = {} bar-tokens, {:.2} GiB",
                stage_context(stage),
                batch as u64 * stage_context(stage) as u64,
                CapacityModel::gib(capacity.step_bytes(batch, stage_context(stage))),
            )
        })
        .collect();
    println!(
        "capacity       {:.0} B/bar-token measured on the real training graph at context \
         {deployed}, plus {:.2} GiB of fixed per-step cost, against {:.2} GiB free ({:.2} GiB \
         already in use). Largest batch the deployed context can hold: {} windows. Requested \
         {requested_batch}, running {}. Headroom at the final stage {:+.2} GiB after the \
         {:.0}% transient margin and the {:.2} GiB shared-card reserve.",
        capacity.per_token_bytes,
        CapacityModel::gib(capacity.fixed_bytes),
        CapacityModel::gib(capacity.free_bytes as f64),
        CapacityModel::gib(capacity.baseline_bytes as f64),
        capacity.frontier_batch(deployed),
        schedule.base_batch,
        CapacityModel::gib(capacity.headroom_bytes(&schedule.batch_ramp, schedule.base_batch)),
        RAMP_MEMORY_MARGIN * 100.0,
        CapacityModel::gib(RAMP_MEMORY_RESERVE_BYTES as f64),
    );
    println!("ramp           {}", stages.join(" | "));
    let frontier: Vec<String> = CONTEXT_FRONTIER
        .iter()
        .map(|&context| format!("{context} bars -> {} windows", capacity.frontier_batch(context)))
        .collect();
    println!(
        "frontier       achievable FLAT batch if the deployed context were [{}]. The card caps \
         batch x context, so these are the same memory spent differently; the recipe's \
         learning-rate scaling is tuned for the larger batch. REPORTED, NOT CHOSEN: the \
         deployed context is {deployed} bars and is part of the world-model contract the \
         planner loads against.",
        frontier.join(", "),
    );
}

#[allow(clippy::too_many_arguments)]
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
    capacity: Option<&CapacityModel>,
    requested_batch: usize,
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
        "schedule       {} steps, batch {}->{}, context {}->{}, lr flat {:.1}% then linear to \
         {:.2}x, momentum {MOMENTUM_START}->{MOMENTUM_PEAK} over {} steps and back over {}. \
         This ramp is DERIVED from the measured capacity below, so it is the one that will \
         execute; the declared ceiling is x{:?} of the base. Each batch step-up is still gated \
         on free VRAM and is HELD (context ramp kept, lr plateau bump lowered to match) when \
         the projected activation increment does not fit with a {:.0}% transient margin and a \
         {:.2} GiB reserve for the card's other tenants — that guard is now a safety net for \
         CONTENTION, not the thing that decides the plan.",
        schedule.total_steps,
        schedule.batch(0),
        schedule.base_batch * schedule.batch_ramp[RAMP_STAGES - 1],
        stage_context(0),
        stage_context(RAMP_STAGES - 1),
        schedule.lr_plateau_fraction * 100.0,
        LR_FLOOR_MULTIPLIER,
        schedule.momentum_warmup,
        schedule.momentum_cooldown,
        BATCH_RAMP,
        RAMP_MEMORY_MARGIN * 100.0,
        RAMP_MEMORY_RESERVE_BYTES as f64 / (1u64 << 30) as f64,
    );
    print_capacity_banner(schedule, capacity, requested_batch);
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
        "objective      nll + {:e}*dyn + {:e}*kl + {:e}*growth, dynamics horizon {}, scored \
         under {}. `dyn` and `kl` are NextLat (arXiv 2511.05963): `dyn` is smooth_l1 to the \
         stop-gradient belief, MEANED over every element of [B, T, {BAR_MODEL_DIM}] exactly \
         as the reference reduces it, so the weight is width-independent and 1.0 is the \
         reference setting. `growth` is -log(1 + f_hat R) at the log-optimal fraction of \
         p(r|PAST) — the head's prefix-free r row — clamped at the bench's {:.1}x leverage cap: \
         the only term that is a function of the quantity the strategy trades. Its weight was \
         sized on GRADIENT norm, not objective share — its magnitude is ~5e-4 nats against \
         nll's ~4.93 — and the run reprints that measurement at steps {:?}. Every step prints \
         each term's share of the objective's magnitude and the run warns when an auxiliary \
         term holds more than {:.0}% of it for {} consecutive steps.",
        args.lambda_dyn,
        args.lambda_kl,
        args.lambda_growth,
        args.dyn_horizon,
        args.scoring,
        trade_bench::LEVERAGE_CAP,
        GROWTH_PROBE_STEPS,
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
    use crate::torch::bar_dist::BarDof;
    use crate::torch::dataset::BAR_TIME_FEATURES;
    use crate::torch::test_rng;
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

    /// Bound libtorch's INTRA-OP pool for every test that drives a real model.
    ///
    /// The libtest harness's `--test-threads` bounds nothing here: ONE single-threaded test
    /// doing ONE forward pass through a [`BAR_LAYERS`]-deep, [`BAR_MODEL_DIM`]-wide trunk fans
    /// its GEMMs across one thread per core on its own, which is how a single test binary in
    /// this module reached 2000% CPU and a load average of 159.
    ///
    /// Two DIFFERENT levers exist and neither one covers the other, which is why this
    /// function has to exist rather than being replaced by an environment variable:
    ///
    /// * `OMP_NUM_THREADS` sets libtorch's DEFAULT pool size, and therefore only binds code
    ///   that never calls `at::set_num_threads`. An explicit call overrides it.
    /// * `TORCH_NUM_THREADS` is this repo's own convention, honoured by [`configure_threads`]
    ///   and by the sibling cap in `pretrain_reports`. Nothing in libtorch reads it, so it
    ///   binds only where repo code asks for it — which, before this, did not include any
    ///   test in this module.
    ///
    /// Measured on the dynamics-MLP test in this module: uncapped `user/real` 1.42/0.46 =
    /// 3.1x, unchanged at 3.0x under `TORCH_NUM_THREADS=1` alone, and 0.32/0.46 = 0.7x under
    /// `OMP_NUM_THREADS=1`. After this call the repo lever binds here too.
    ///
    /// Defaults to ONE thread when `TORCH_NUM_THREADS` is unset, matching the sibling cap in
    /// `pretrain_reports`: an unset variable is the case that took the machine down, so it has
    /// to be the safe one, and asking for more has to be explicit. The ceiling is clamped to 4
    /// so that an operator's `TORCH_NUM_THREADS=24` cannot re-arm the failure through a test.
    ///
    /// Never RAISED, enforced against the pool's ACTUAL current size via
    /// `tch::get_num_threads` rather than merely asserted: a pre-main constructor in this
    /// crate's test build already lowers the default to 1, and a cap that wrote its own
    /// ceiling unconditionally would quietly undo that. Taking the minimum composes with any
    /// earlier cap instead of fighting it.
    ///
    /// Sets the INTRA-OP pool only. The interop pool is deliberately not touched here:
    /// `at::set_num_interop_threads` RAISES once torch has done any parallel work, so calling
    /// it from a fixture — after the harness has already run other tests — would abort the
    /// run. Interop belongs to [`configure_threads`], which is invoked before `main` in the
    /// test build and from [`pretrain`] in production. That split is why capping intra-op
    /// here is a ceiling rather than the whole answer: a heavy test still pays interop and
    /// rayon on top.
    ///
    /// `Once`, because the trainer fixtures call it repeatedly and the pool is process-wide.
    fn cap_torch_threads() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let ceiling = std::env::var("TORCH_NUM_THREADS")
                .ok()
                .and_then(|value| value.parse::<i32>().ok())
                .unwrap_or(1)
                .clamp(1, 4);
            let threads = ceiling.min(tch::get_num_threads()).max(1);
            tch::set_num_threads(threads);
        });
    }

    /// A corpus just large enough that the 10% validation and test regions each hold a
    /// full deployed-context window, which is what `EvaluationSets::new` requires.
    fn corpus_fixture(label: &str) -> (Fixture, BarCorpus) {
        // Every test that builds a real trainer comes through here, so this is the one place
        // the module's thread ceiling has to be set.
        cap_torch_threads();
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

    /// Auxiliary resolution used by the wiring test. HOURLY, not daily, and deliberately: the
    /// auxiliary held-out set needs [`AUXILIARY_HELDOUT_CONTEXT`] bars inside the val decile, and
    /// a val decile wide enough to hold 64 DAILY bars would need a 650-day fixture. The real
    /// daily numbers are proven against the real corpus by
    /// `pretrain_aux::tests::the_real_daily_corpus_enters_training_and_no_held_out_bar_does`;
    /// this fixture proves the WIRING, which is resolution-agnostic by construction.
    const AUX_TEST_RES: u32 = 3_600;

    /// Add an auxiliary resolution to a corpus fixture directory, spanning from well before the
    /// deployment corpus's first bar to its last, so the auxiliary axis straddles BOTH split
    /// instants — which is the real shape and the only shape that can test the exclusion.
    fn add_auxiliary_fixture(dir: &Path, deployment: &BarCorpus) {
        let last = deployment
            .symbols()
            .iter()
            .enumerate()
            .map(|(series, _)| deployment.ts_ms(series, 0))
            .max()
            .expect("the fixture has symbols");
        let step_ms = AUX_TEST_RES as i64 * 1000;
        let span = 26_000i64 * TEST_RES as i64 * 1000;
        let end = last + span;
        for (index, symbol) in ["AAA", "BBB", "CCC"].iter().enumerate() {
            let mut rng = ChaCha12Rng::seed_from_u64(0x5A17 + index as u64);
            let mut close = 50.0f32;
            let count = 6_000i64;
            let bars: Vec<PackedBar> = (0..count)
                .map(|i| {
                    let open = close;
                    close = (close * (1.0 + rng.random_range(-0.03f32..0.03))).max(1.0);
                    let spread = rng.random_range(0.0f32..0.05) * open;
                    PackedBar {
                        ts_ms: end - (count - i) * step_ms,
                        open,
                        high: open.max(close) + spread,
                        low: (open.min(close) - spread).max(0.5),
                        close,
                        volume: rng.random_range(1_000.0f32..90_000.0),
                        vwap: 0.5 * (open + close),
                        trades: rng.random_range(1u32..900),
                    }
                })
                .collect();
            write_bar_file(
                &dir.join(format!("{symbol}.{AUX_TEST_RES}.{FILE_EXTENSION}")),
                symbol,
                AUX_TEST_RES,
                &bars,
            )
            .expect("write auxiliary bars");
        }
    }

    /// The auxiliary resolution is WIRED, end to end, through the real `build_trainer`.
    ///
    /// This is the test that would have caught the state this session started in: 4,748 daily
    /// files on disk, reachable by no training code path, with no fitted supports. It asserts the
    /// four things that have to hold simultaneously for the corpus to be genuinely in training —
    /// it loads, it gets its OWN supports in the routing set, it takes optimizer steps, and it
    /// leaves a readable artifact — plus the one thing that must NOT hold, that a held-out bar
    /// enters the auxiliary stream.
    #[test]
    fn an_auxiliary_resolution_is_loaded_stepped_and_reported_end_to_end() {
        let _torch_rng_guard = test_rng::exclusive();
        let (fx, corpus) = corpus_fixture("aux_wiring");
        add_auxiliary_fixture(&fx.dir, &corpus);
        let dir = PathBuf::from(corpus.dir());
        let runs = std::env::temp_dir().join(format!(
            "trading_bot_0_pretrain_aux_runs_{}",
            uuid::Uuid::new_v4()
        ));
        let mut args = test_args(0x5EED, &dir);
        args.auxiliary_resolutions = vec![AUX_TEST_RES];
        args.validation_windows = 1;
        args.snapshot_windows = 1;
        args.support_samples = 2_048;
        let mut trainer = build_trainer(args, &runs.display().to_string(), Device::Cpu)
            .expect("a trainer with an auxiliary resolution builds");

        // 1. It loaded, with its own supports in the routing set.
        assert_eq!(trainer.aux.len(), 1, "the auxiliary corpus must be open");
        let aux = &trainer.aux[0];
        assert_eq!(aux.res_secs(), AUX_TEST_RES);
        assert_eq!(aux.corpus().symbols().len(), 3);
        assert!(aux.covered_bars() > 0, "the auxiliary pass covers no bar");
        assert!(aux.steps_per_epoch() > 0);
        assert_eq!(
            trainer.support_set_dev.resolutions(),
            vec![TEST_RES, AUX_TEST_RES],
            "both resolutions must be routable, or one is scored against the other's bins"
        );
        assert_ne!(
            trainer.support_set_dev.get(TEST_RES).unwrap().marginal_nll_bar(trainer.args.scoring),
            trainer
                .support_set_dev
                .get(AUX_TEST_RES)
                .unwrap()
                .marginal_nll_bar(trainer.args.scoring),
            "the two resolutions were fitted to the same geometry, so they are not really separate"
        );
        assert!(
            aux.corpus().supports_path().exists(),
            "the auxiliary fit must be persisted so a later run can reuse and verify it"
        );

        // The auxiliary held-out set is on the auxiliary axis at the auxiliary context.
        assert_eq!(trainer.aux_heldout.len(), 1);
        assert_eq!(
            trainer.aux_heldout[0].context,
            AUXILIARY_HELDOUT_CONTEXT,
            "the auxiliary held-out read is pinned to its own context, not the deployed one"
        );

        // The multi-resolution corpus audit landed on the ONE registered base.
        let anomalies = trainer
            .run
            .gens
            .join("pretrain_corpus_anomalies.report.bin");
        assert!(
            anomalies.exists(),
            "the corpus loader must write its audit for every loaded resolution"
        );
        let report = shared::report::read_report(&anomalies).expect("the audit reads back");
        let shared::report::ReportKind::MultiLine { series } = report.kind else {
            panic!("the audit must be a MultiLine report");
        };
        assert!(
            series.iter().any(|s| s.label.ends_with("@300"))
                && series.iter().any(|s| s.label.ends_with("@3600")),
            "both resolutions must appear in the audit: {:?}",
            series.iter().map(|s| &s.label).collect::<Vec<_>>()
        );

        // 2. It takes real optimizer steps, on the Bresenham cadence, over one primary pass.
        let primary_steps = trainer.schedule.steps_per_epoch;
        let mut fired = 0usize;
        for step in 0..primary_steps {
            trainer.auxiliary_steps(step).expect("an auxiliary step runs");
            if trainer.aux_steps > fired {
                fired = trainer.aux_steps;
            }
        }
        assert!(
            trainer.aux_steps > 0,
            "the auxiliary stream never fired across {primary_steps} primary steps, which is \
             exactly what a corpus that trains nothing looks like"
        );
        assert!(trainer.aux_bars_seen > 0);

        // 3. The pass boundary measures a held-out number and writes the artifact.
        trainer
            .finish_auxiliary_passes(primary_steps)
            .expect("the auxiliary pass boundary reconciles");
        let path = trainer
            .run
            .gens
            .join("pretrain_auxiliary_nll.report.bin");
        assert!(path.exists(), "the auxiliary report must be written");
        let report = shared::report::read_report(&path).expect("the auxiliary report reads back");
        let shared::report::ReportKind::MultiLine { series } = report.kind else {
            panic!("the auxiliary report must be a MultiLine report");
        };
        assert_eq!(
            series.iter().map(|s| s.label.as_str()).collect::<Vec<_>>(),
            vec!["train@3600s", "held-out@3600s"]
        );
        for s in &series {
            assert!(
                s.values.iter().all(|v| v.is_finite()) && !s.values.is_empty(),
                "{} carries no finite measurement",
                s.label
            );
        }

        // 4. And the exclusion. The auxiliary axis straddles both instants, so val and test
        // bars EXIST; the pass must account for the train bars and nothing else.
        let aux_corpus = trainer.aux[0].corpus();
        let (train, val, test) = (
            aux_corpus.split_bars(Split::Train) as u64,
            aux_corpus.split_bars(Split::Val) as u64,
            aux_corpus.split_bars(Split::Test) as u64,
        );
        assert!(
            val > 0 && test > 0,
            "the fixture must straddle both instants or the exclusion is untested"
        );
        let covered = trainer.aux[0].covered_bars();
        assert_eq!(
            covered + trainer.aux[0].pass_remainder_total(),
            train,
            "the auxiliary pass must cover exactly the train bars: {covered} covered against \
             {train} train, {val} val, {test} test"
        );
        drop(trainer);
        let _ = std::fs::remove_dir_all(&runs);
    }

    /// A run whose supports cannot be PERSISTED must refuse at step 0.
    ///
    /// This is the defect in full: the live corpus artifact was `format_version: 4` with no
    /// fitted moments, the run started, trained 1000 steps, promoted — and only then discovered
    /// that the checkpoint sidecar it had to write is a schema its supports cannot satisfy. The
    /// whole warmup was spent on a run that could never produce a checkpoint, and every later
    /// step would have hit the same wall. `build_trainer` must never get past the fit.
    #[test]
    fn a_run_whose_supports_cannot_be_persisted_refuses_before_step_zero() {
        let _torch_rng_guard = test_rng::exclusive();
        let (fx, corpus) = corpus_fixture("unpersistable_supports");
        let dir = PathBuf::from(corpus.dir());
        let mut args = test_args(0x5EED, &dir);
        args.support_samples = 2_048;

        // The artifact exactly as a real run would have left it, provenance and all, then
        // knocked back to the pre-moments schema the live corpus file is actually in. Written
        // through `fit_supports_at` so the provenance gate PASSES and the refusal under test is
        // the only thing that can fire.
        let path = corpus.supports_path();
        fit_supports_at(
            &corpus,
            &path,
            SupportsFit::of(&args),
            &corpus.identity_fingerprint(),
        )
        .expect("the fixture's own supports fit");
        let mut raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("parse");
        let object = raw.as_object_mut().expect("object");
        object.insert("format_version".to_owned(), serde_json::json!(4));
        object.remove("bin_means");
        object.remove("bin_second_moments");
        std::fs::write(&path, serde_json::to_vec(&raw).expect("serialize")).expect("write");

        let runs = std::env::temp_dir()
            .join(format!("trading_bot_0_pretrain_unpersistable_{}", uuid::Uuid::new_v4()));
        let err = build_trainer(args, &runs.display().to_string(), Device::Cpu)
            .err()
            .expect("a run that can never write a checkpoint must not start");
        let message = format!("{err:#}");
        for expected in [
            path.display().to_string().as_str(),
            "format version 4",
            "no fitted per-bin moments",
            "bar-supports-moments",
        ] {
            assert!(
                message.contains(expected),
                "the refusal must name the artifact, its version and the remedy; {expected:?} \
                 missing from: {message}"
            );
        }
        let _ = std::fs::remove_dir_all(&runs);
        drop(fx);
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
            auxiliary_resolutions: Vec::new(),
            support_samples: 1024,
            scoring: BarScoring::Density,
            dyn_horizon: 1,
            lambda_dyn: 1.0,
            lambda_kl: 1.0,
            // The derived weight, not zero: a trainer test that ran the control arm would
            // leave the term's gradient path untested by every test in this file.
            lambda_growth: LAMBDA_GROWTH,
            validation_windows: 3,
            diagnostic_context: BAR_CONTEXT_RAMP_START,
            snapshot_windows: 1,
            // The floor plus a margin. Every boundary now takes an ancestral rollout, the
            // rollout is linear in this, and the production 256 would put a 100-bar
            // autoregressive decode of a 10-layer 512-wide transformer on the CPU in every
            // test that validates. Nothing here asserts a quantile's precision, so the
            // draws are the one part of the fixture that is large for no reason.
            snapshot_samples: 8,
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
            // The fixture runs on CPU, where capacity is unmeasured and nothing is ever
            // clamped, so the refusal mode has nothing to refuse. `false` is also the
            // production default: absent here means "behave exactly as before", which is the
            // correct absent value for a flag that only ever REJECTS a reduction.
            exact_batch: false,
            // The recipe default, so every trainer test in this file exercises the schedule
            // every persisted run was produced under.
            lr_plateau_fraction: LR_PLATEAU_FRACTION,
        }
    }

    /// EVAL-GATE-001 and EVAL-GATE-002. A ramp stage below the deployed context must still
    /// produce a full fixed-context held-out read, an epoch artifact and a defensible best;
    /// and a run that NEVER reaches the deployed context must still end with a promoted
    /// checkpoint and a final held-out number that records the context it was taken at.
    ///
    /// This is the defect job 2856 exhibited: nine validations in a row printed `val diag=NaN`
    /// and wrote no checkpoint, because one `else` branch skipped the whole validation instead
    /// of only the promotion decision. Both halves are asserted on the SAME trainer, in the
    /// order a run performs them.
    #[test]
    fn a_ramp_below_the_deployed_context_still_measures_and_still_leaves_an_artifact() {
        let _torch_rng_guard = test_rng::exclusive();
        let (_fx, corpus) = corpus_fixture("gating");
        let dir = PathBuf::from(corpus.dir());
        let runs = std::env::temp_dir().join(format!(
            "trading_bot_0_pretrain_gating_runs_{}",
            uuid::Uuid::new_v4()
        ));
        let mut args = test_args(0x5EED, &dir);
        // One window per pinned set and no ancestral samples beyond the minimum: this test is
        // about the control flow, and the model is a 512-wide 10-layer transformer on CPU.
        args.validation_windows = 1;
        args.snapshot_windows = 1;
        // CPU, explicitly: this asserts validation and promotion bookkeeping, none of which
        // is device-dependent, and it must never allocate on a card another tenant owns.
        let mut trainer = build_trainer(args, &runs.display().to_string(), Device::Cpu)
            .expect("trainer builds");
        let diag = trainer.eval.diagnostic.context;
        let deployed = trainer.eval.promotion.context;
        assert!(
            diag < deployed,
            "the fixture must exercise the gap between the diagnostic and deployed contexts"
        );
        // The ramp got to the diagnostic context and no further, which is exactly the state a
        // memory-held run sits in for most of its length.
        trainer.reached_context = diag;

        trainer.validate(0, true, false).expect("validation runs");
        assert_eq!(
            trainer.promotions, 0,
            "promotion must stay gated on the deployed context"
        );
        assert!(
            !trainer.run.weights.join("pretrain_best.ot").exists(),
            "the planner's artifact must not be written from a below-deployed selection"
        );
        let epoch_artifact = trainer
            .run
            .weights
            .join(format!("pretrain_epoch_0_ctx{diag}.ot"));
        assert!(
            epoch_artifact.exists(),
            "every epoch boundary must leave an artifact, promotion or not"
        );
        assert!(
            window_scores_path(&epoch_artifact).exists(),
            "the epoch artifact must carry its own held-out scores, or it is not evaluable"
        );
        let diag_best = trainer
            .run
            .weights
            .join(format!("pretrain_best_diag{diag}.ot"));
        assert!(
            diag_best.exists(),
            "a run held below the deployed context must still have a defensible best"
        );
        let best_at_diag = trainer
            .best_by_context
            .get(&diag)
            .copied()
            .expect("the diagnostic context must have a best");
        assert!(
            best_at_diag.is_finite(),
            "the diagnostic best must be a measurement, not NaN: {best_at_diag}"
        );

        // The charts: the fixed-context panel is populated, and the deployed-context series is
        // ABSENT and says so, rather than carrying a NaN that reads as a measured catastrophe.
        let gens = trainer.run.gens.join("0");
        let finite = |base: &str, label: &str| -> (bool, String) {
            let report = shared::report::read_report(&gens.join(format!("{base}.report.bin")))
                .expect("report reads back");
            let shared::report::ReportKind::MultiLine { series } = report.kind else {
                panic!("{base} is not a MultiLine chart");
            };
            let found = series
                .iter()
                .find(|s| s.label.starts_with(label))
                .unwrap_or_else(|| {
                    panic!(
                        "{base} has no series starting with {label}: {:?}",
                        series.iter().map(|s| &s.label).collect::<Vec<_>>()
                    )
                });
            (
                found.values.iter().any(|v| v.is_finite()),
                found.label.clone(),
            )
        };
        assert!(
            finite("pretrain_nll_bar_diag896", "val diag").0,
            "the fixed-context held-out read must be measured from step 0"
        );
        assert!(
            finite("pretrain_nll_dof", "r val diag").0,
            "the per-DOF held-out breakdown must come from the diagnostic pass"
        );
        assert!(
            finite("pretrain_nll_vs_baselines", "val diag").0,
            "the baseline-comparison curve must come from the diagnostic pass"
        );
        assert!(
            finite("pretrain_forecast_nll", "forecast (marginalized)").0,
            "the marginalized forecast number must be measured at every validation"
        );
        let (measured, label) = finite("pretrain_nll_bar", "val deployed");
        assert!(
            !measured,
            "the deployed-context series must be absent, not a NaN pretending to be a point"
        );
        assert!(
            label.contains("NOT MEASURED"),
            "an absent series must say so in its own legend, got {label}"
        );

        // Second half: the final step of a run that never got to the deployed context.
        let last = trainer.schedule.total_steps - 1;
        trainer.validate(last, false, true).expect("final validation");
        assert_eq!(
            trainer.promotions, 1,
            "the final step must promote at the reached context rather than leave nothing"
        );
        assert!(trainer.run.weights.join("pretrain_best.ot").exists());
        assert_eq!(
            trainer.selection_context, diag,
            "the selection context must be the one actually trained at"
        );

        let (battery, dyn_identity) = trainer.test_battery().expect("terminal battery");
        // The shipped dynamics head must beat the trivial identity map, which is the
        // invariant the deleted auxiliary anneal violated on every promotion it made.
        assert!(
            dyn_identity.is_finite(),
            "the terminal battery could not measure dyn/identity at all"
        );
        check_dynamics_beats_identity(dyn_identity, trainer.args.dyn_horizon as i64)
            .expect("a run must not end with a dynamics head worse than doing nothing");
        assert!(
            battery.nll_bar.is_finite(),
            "a run must never end without a held-out number: {}",
            battery.nll_bar
        );
        assert_eq!(battery.selection_context, diag);
        assert_eq!(battery.reached_context, diag);
        assert_eq!(battery.deployed_context, deployed);
        assert!(
            battery.forecast_nll_dof.iter().all(|v| v.is_finite()),
            "the forecast breakdown must be measured on the test split too: {:?}",
            battery.forecast_nll_dof
        );
        // The artifact must state the context it was selected at, on disk, not just in a log.
        let metadata = BarWorldModelMetadata::load(&world_model_metadata_path(
            &trainer.run.weights.join("pretrain_best.ot"),
        ))
        .expect("promoted metadata reads back");
        let provenance = metadata
            .training
            .as_ref()
            .expect("a promoted checkpoint states its provenance");
        assert_eq!(provenance.selection_context, diag);
        assert_eq!(provenance.reached_context, diag);
        assert_eq!(provenance.deployed_context, deployed);

        std::fs::remove_dir_all(&runs).ok();
    }

    /// `meta.json` must state the split instants the run was HANDED, not the instants a reader
    /// would guess by checking out the commit and reading this file's default.
    ///
    /// The distinction is the whole point and one assertion carries it: the fixture DERIVES its
    /// instants from a synthetic 2020-ish corpus, so the recorded pair is necessarily different
    /// from [`crate::data::ingest::PINNED_SPLIT_BOUNDS`]. A record that echoed the constant
    /// would satisfy "the field is present" and convert nothing, which is exactly the state
    /// `training/runs/bardist_v3_rfirst_1ep/meta.json` was in with `{"commit": ...}` alone.
    ///
    /// Driven through `build_trainer`, so what is asserted is the write the RUN performs at run
    /// start, not a unit test of the setter.
    #[test]
    fn a_run_records_the_split_instants_it_was_handed_rather_than_the_code_default() {
        let _torch_rng_guard = test_rng::exclusive();
        let (_fx, corpus) = corpus_fixture("provenance");
        let dir = PathBuf::from(corpus.dir());
        let runs = std::env::temp_dir().join(format!(
            "trading_bot_0_pretrain_provenance_runs_{}",
            uuid::Uuid::new_v4()
        ));
        let mut args = test_args(0x5EED, &dir);
        args.validation_windows = 1;
        args.snapshot_windows = 1;
        let trainer = build_trainer(args, &runs.display().to_string(), Device::Cpu)
            .expect("trainer builds");

        let meta = trainer.run.meta().expect("meta.json reads back");
        let recorded = meta
            .provenance
            .expect("a run must record what it was handed before it takes a step");

        // The instants, against the trainer's own resolved pair rather than against a literal.
        assert_eq!(
            recorded.split_bounds_ms,
            [trainer.split_bounds().0, trainer.split_bounds().1],
            "the record must carry the instants this run actually split on"
        );
        assert_ne!(
            recorded.split_bounds_ms,
            [
                crate::data::ingest::PINNED_SPLIT_BOUNDS.0,
                crate::data::ingest::PINNED_SPLIT_BOUNDS.1
            ],
            "a record that reproduces the campaign default proves nothing about the run"
        );
        assert!(
            !recorded.split_bounds_pinned,
            "the fixture derives its instants, and a reader must be told so: derived instants \
             are percentiles of whatever was on disk that day"
        );

        // Everything else the run ASSUMED. Each is checked against the value handed in, so a
        // field wired to the wrong source fails rather than merely being populated.
        assert_eq!(recorded.resolution_secs, TEST_RES);
        assert_eq!(recorded.min_bars, 100);
        assert_eq!(recorded.min_dollar_volume, 0.0);
        assert_eq!(recorded.data_dir, dir.display().to_string());
        assert_eq!(recorded.diagnostic_context_bars, BAR_CONTEXT_RAMP_START);
        assert_eq!(recorded.deployed_context_bars, trainer.eval.promotion.context);
        assert_eq!(recorded.eval_window_seed, EVAL_WINDOW_SEED);
        assert_eq!(recorded.train_seed, 0x5EED);
        assert_eq!(recorded.corpus_fingerprint, trainer.corpus_fingerprint);

        std::fs::remove_dir_all(&runs).ok();
    }

    /// The held-out power census must COUNT, must count the same thing the sampler and the
    /// bootstrap count, and must land on both registered bases.
    ///
    /// This is the instrument that decides whether `Split::Test` is worth spending, so the
    /// property under test is not "a chart appeared". Three things are asserted:
    ///
    /// * The census agrees bar-for-bar and window-for-window with [`BarCorpus::split_bars`] and
    ///   [`BarSampler`], which are what the scoring pass itself will use. A census computed by a
    ///   second, agreeing-by-luck route would be worse than none.
    /// * Blocks are `<=` windows and the ladder's half-width is NON-INCREASING as the prefix
    ///   grows. The naive identity "one window, one block" is FALSE in general — two windows of
    ///   one symbol inside one calendar month are ONE bootstrap draw — and the interval scales
    ///   with the block count, so a ladder that read blocks off the window count would overstate
    ///   the power of every rung.
    /// * The scaling is exactly `sqrt(B_ref / B)`, checked against the reference itself.
    ///
    /// No checkpoint is opened and nothing is scored, which is the whole point of the pass this
    /// covers.
    #[test]
    fn the_heldout_power_census_writes_both_registered_bases() {
        let (_fx, corpus) = corpus_fixture("power");
        let set = PinnedSet::pinned(&corpus, Split::Test, BAR_CONTEXT_RAMP_START, 64)
            .expect("the fixture's test region holds a ramp-start window");
        let blocks_all = pinned_blocks(&set);
        let cut = blocks_all.len() / 2;
        let power = HeldOutPower::measure(
            &corpus,
            Split::Test,
            BAR_CONTEXT_RAMP_START,
            &blocks_all,
            &blocks_all[..cut],
            &blocks_all[cut..],
        );

        // 1. The census counts what the scoring pass will count.
        assert_eq!(power.census.len(), 3);
        for row in &power.census {
            assert_eq!(
                row.bars,
                corpus.split_bars(row.split),
                "{} bars disagree with the corpus",
                row.split.as_str()
            );
            let sampler =
                BarSampler::new(&corpus, row.split, BAR_CONTEXT_RAMP_START, EVAL_WINDOW_SEED);
            assert_eq!(
                row.anchors,
                sampler.windows(),
                "{} window supply disagrees with the sampler",
                row.split.as_str()
            );
            assert!(
                row.symbols <= corpus.series_count(),
                "{} claims more symbols than the corpus holds",
                row.split.as_str()
            );
        }
        let test_row = power
            .census
            .iter()
            .find(|row| row.split == Split::Test)
            .expect("the test split is censused");
        assert!(
            power.windows_drawn <= test_row.anchors,
            "a draw of {} cannot exceed the {} windows the split supplies",
            power.windows_drawn,
            test_row.anchors
        );

        // 2. Blocks are a coarsening of windows, and power is monotone in the prefix.
        assert!(power.traded_blocks <= power.traded_windows);
        assert!(power.fit_blocks <= power.fit_windows);
        assert!(power.traded_blocks >= 1);
        assert_eq!(
            power.ladder.last().map(|(n, _, _)| *n),
            Some(power.windows_drawn),
            "the whole draw must be the last rung"
        );
        for pair in power.ladder.windows(2) {
            let ((windows_lo, blocks_lo, half_lo), (windows_hi, blocks_hi, half_hi)) =
                (pair[0], pair[1]);
            assert!(windows_lo < windows_hi);
            assert!(
                blocks_lo <= blocks_hi,
                "a longer prefix cannot hold fewer blocks: {blocks_lo} then {blocks_hi}"
            );
            assert!(
                half_hi <= half_lo,
                "more blocks must not widen the interval: {half_lo} then {half_hi}"
            );
            assert!(blocks_hi <= windows_hi, "blocks must coarsen windows, never split them");
        }

        // 3. The scaling law, exactly, against the reference it extrapolates from.
        assert_eq!(
            expected_net_ci_half_width_bps(REFERENCE_NET_CI_BLOCKS),
            REFERENCE_NET_CI_HALF_WIDTH_BPS
        );
        assert!(
            (expected_net_ci_half_width_bps(REFERENCE_NET_CI_BLOCKS)
                / expected_net_ci_half_width_bps(REFERENCE_NET_CI_BLOCKS * 4)
                - 2.0)
                .abs()
                < 1e-12,
            "four times the blocks must halve the half-width"
        );
        assert!(expected_net_ci_half_width_bps(0).is_infinite());

        // 4. Both registered bases land, are readable, and carry finite rows.
        let dir = std::env::temp_dir().join(format!(
            "trading_bot_0_pretrain_power_charts_{}",
            uuid::Uuid::new_v4()
        ));
        crate::torch::train::pretrain_reports::write_heldout_power(&dir, &power)
            .expect("charts write");
        for base in ["pretrain_heldout_census", "pretrain_heldout_power"] {
            assert!(
                shared::report::PRETRAIN_REPORT_BASES.contains(&base),
                "{base} must be registered or the TUI cannot see it"
            );
            let path = dir.join(format!("{base}.report.bin"));
            assert!(path.exists(), "{base} was not written");
            let report = shared::report::read_report(&path).expect("the report reads back");
            let shared::report::ReportKind::MultiLine { series } = report.kind else {
                panic!("{base} must be a MultiLine chart");
            };
            assert!(series.len() >= 3, "{base} carries only {} series", series.len());
            for row in &series {
                assert!(
                    row.values.iter().any(|value| value.is_finite()),
                    "{base} series `{}` is entirely non-finite",
                    row.label
                );
            }
        }
        // The census panel's own x-axis is the split index, in calendar order.
        let census =
            shared::report::read_report(&dir.join("pretrain_heldout_census.report.bin")).unwrap();
        let shared::report::ReportKind::MultiLine { series } = census.kind else {
            panic!("the census must be a MultiLine chart");
        };
        assert_eq!(series[0].values, vec![0.0f32, 1.0, 2.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The capacity probe must leave NOTHING behind: it runs the real training graph forward
    /// and backward, so it has to be provably non-destructive or the ramp is bought by
    /// corrupting the run it plans.
    ///
    /// Three things could leak and all three are checked. Weights: there is no optimizer step,
    /// so every parameter must be bit-identical afterwards. Optimizer state: the AdamW moments
    /// are allocated lazily on the first step, so `initialized_adamw_names` being empty is
    /// direct proof no step was taken, and `state_bytes` must not have moved. Device memory:
    /// the probe releases the allocator pool, and on CPU there is no pool to release and no
    /// NVML to read, which is why the reading is `None` while the passes still run.
    ///
    /// Driven at a short context on a full-size model: the invariant is context-independent
    /// and a 2048-bar CPU pass is not a unit test.
    #[test]
    fn the_capacity_probe_leaves_no_trace_in_the_run_it_measures() {
        let _torch_rng_guard = test_rng::exclusive();
        let (_fx, corpus) = corpus_fixture("probe");
        let dir = PathBuf::from(corpus.dir());
        let runs = std::env::temp_dir().join(format!(
            "trading_bot_0_pretrain_probe_runs_{}",
            uuid::Uuid::new_v4()
        ));
        let mut args = test_args(0x5EED, &dir);
        args.validation_windows = 1;
        args.snapshot_windows = 1;
        let trainer = build_trainer(args, &runs.display().to_string(), Device::Cpu)
            .expect("trainer builds");

        // The device is pinned above, so this is a GUARANTEE and not a property of the host:
        // a CPU build cannot measure capacity, so it must keep the DECLARED ramp rather than
        // invent a ceiling out of a missing reading. Before the device became a parameter this
        // assertion silently inverted on any machine with a visible card.
        assert!(trainer.capacity.is_none(), "there is no NVML behind a CPU device");
        assert_eq!(trainer.derived_batch_ramp, BATCH_RAMP);
        assert_eq!(trainer.schedule.batch_ramp, BATCH_RAMP);

        let before: Vec<(String, Tensor)> = named_trainable_variables(&trainer.vs)
            .into_iter()
            .map(|(name, tensor)| (name, tch::no_grad(|| tensor.detach().copy())))
            .collect();
        let state_before = trainer.optimizer.state_bytes();
        assert!(
            trainer.optimizer.initialized_adamw_names().is_empty(),
            "the AdamW moments must still be unallocated before any step"
        );

        let sampler = &trainer.train_samplers[0];
        let refs = sampler.batch_refs(0, 0, 1);
        let sample = sampler.batch_of(&refs, trainer.device);
        let growth_support = GrowthSupport::new(&trainer.supports_dev, trainer.device)
            .expect("the test support carries the growth term");
        let reading = probe_shape_used_bytes(
            &trainer.modules,
            &trainer.supports_dev,
            &growth_support,
            &sample,
            &trainer.args,
            128,
            trainer.device,
        );
        assert_eq!(reading, None, "a CPU device has no VRAM reading to give");

        // The passes really ran: a backward left gradients behind. They are the ONE thing the
        // probe leaves resident, and `optimizer_step` zeroes them before its own backward.
        let after = named_trainable_variables(&trainer.vs);
        assert!(
            after.iter().any(|(_, t)| t.grad().defined()),
            "the probe must actually run a backward, or it measures nothing"
        );
        for (name, saved) in &before {
            let (_, current) = after
                .iter()
                .find(|(other, _)| other == name)
                .expect("the probe must not add or rename parameters");
            let drift = tch::no_grad(|| {
                (current.detach() - saved)
                    .abs()
                    .max()
                    .double_value(&[])
            });
            assert_eq!(drift, 0.0, "{name} moved by {drift} during the capacity probe");
        }
        assert_eq!(
            trainer.optimizer.state_bytes(),
            state_before,
            "the probe must not allocate optimizer state"
        );
        assert!(
            trainer.optimizer.initialized_adamw_names().is_empty(),
            "an allocated AdamW moment proves the probe took an optimizer step"
        );

        std::fs::remove_dir_all(&runs).ok();
    }

    /// EVAL-CKPT-003. The step-tagged window is bounded and touches nothing else.
    ///
    /// Checkpointing every [`DEFAULT_CHECKPOINT_EVERY`] steps is what stops a crash from
    /// destroying an epoch, and it is only affordable if the window is pruned. Pruning the
    /// wrong file would destroy the artifact the planner loads, so the selection is asserted
    /// against every other name a run writes.
    #[test]
    fn pruning_keeps_a_bounded_step_window_and_spares_every_other_artifact() {
        let _torch_rng_guard = test_rng::exclusive();
        let (_fx, corpus) = corpus_fixture("prune");
        let dir = PathBuf::from(corpus.dir());
        let runs = std::env::temp_dir().join(format!(
            "trading_bot_0_pretrain_prune_runs_{}",
            uuid::Uuid::new_v4()
        ));
        let mut args = test_args(0x5EED, &dir);
        args.validation_windows = 1;
        args.snapshot_windows = 1;
        // Pruning is a decision about file names; the device is irrelevant and a visible card
        // would only mean allocating on someone else's GPU to rename files.
        let trainer = build_trainer(args, &runs.display().to_string(), Device::Cpu)
            .expect("trainer builds");
        let weights = trainer.run.weights.clone();
        // Stand-in artifacts: pruning is a decision about file NAMES, and a real
        // `write_checkpoint` here would move a gigabyte of weights to say nothing more.
        let tags = [100usize, 200, 300, 400, 500];
        for tag in tags {
            let path = weights.join(format!("pretrain_step_{tag}.ot"));
            std::fs::write(&path, b"weights").expect("step artifact");
            std::fs::write(world_model_metadata_path(&path), b"{}").expect("metadata");
            std::fs::write(
                world_model_supports_path(&path, trainer.args.resolution_secs),
                b"{}",
            )
            .expect("supports");
            std::fs::write(window_scores_path(&path), b"{}").expect("scores");
        }
        let spared = [
            "pretrain_best.ot",
            "pretrain_best_diag896.ot",
            "pretrain_last.ot",
            "pretrain_epoch_0_ctx896.ot",
            "pretrain_step_600.other",
        ];
        for name in spared {
            std::fs::write(weights.join(name), b"keep").expect("spared artifact");
        }

        trainer.prune_step_checkpoints().expect("pruning runs");

        let kept: Vec<usize> = tags
            .into_iter()
            .filter(|tag| weights.join(format!("pretrain_step_{tag}.ot")).exists())
            .collect();
        assert_eq!(
            kept,
            vec![300, 400, 500],
            "the newest {RETAINED_STEP_CHECKPOINTS} step checkpoints and no others must \
             survive. `test_args` is a single-pass run, so `plateau_anchor_tags` returns \
             nothing and exempts nothing: a one-pass run has no repetition boundary to \
             bracket. Anchors widening this window is asserted separately."
        );
        for tag in [100usize, 200] {
            let path = weights.join(format!("pretrain_step_{tag}.ot"));
            assert!(
                !world_model_metadata_path(&path).exists()
                    && !window_scores_path(&path).exists()
                    && !world_model_supports_path(&path, trainer.args.resolution_secs).exists(),
                "pruning must take the sidecars with the weights, or the directory fills with \
                 orphans that name a file that no longer exists"
            );
        }
        for tag in [300usize, 400, 500] {
            let path = weights.join(format!("pretrain_step_{tag}.ot"));
            assert!(
                world_model_metadata_path(&path).exists() && window_scores_path(&path).exists(),
                "a retained checkpoint must keep the sidecars that make it loadable"
            );
        }
        for name in spared {
            assert!(
                weights.join(name).exists(),
                "{name} is not a step checkpoint and must never be pruned"
            );
        }

        std::fs::remove_dir_all(&runs).ok();
    }

    /// EVAL-CKPT-003b. The pass-boundary anchors bracket the ONE step where repetition
    /// switches on, and exist only when that step is at a constant learning rate.
    ///
    /// This is the contract that makes the next run's repetition question answerable, so the
    /// geometry it is asserted against is `bardist_v2`'s own: 31095 total steps, 10365 steps
    /// per epoch, artifacts every [`DEFAULT_CHECKPOINT_EVERY`] steps. That run retained the
    /// newest three artifacts, all of them past the LR plateau where the repetition and rate
    /// coefficients are exactly collinear, and so it cannot answer its own central question at
    /// any precision. The negative cases matter as much as the positive one: retaining an
    /// anchor that CANNOT identify repetition would cost disk and buy a misleading comparison.
    #[test]
    fn plateau_anchors_bracket_the_first_pass_boundary() {
        let cadence: Vec<usize> = (1..=60).map(|k| k * DEFAULT_CHECKPOINT_EVERY).collect();

        // Three epochs: plateau ends at 0.4 * 31095 = 12438, the boundary is 10365, so the
        // boundary is inside the plateau with 2073 steps of margin.
        assert_eq!(
            plateau_anchor_tags(&cadence, 10365, 31095, LR_PLATEAU_FRACTION),
            vec![10240, 10752, 5120],
            "the straddling pair comes first because it carries the discontinuity, then the \
             all-fresh baseline at the midpoint of the first pass"
        );
        for tag in plateau_anchor_tags(&cadence, 10365, 31095, LR_PLATEAU_FRACTION) {
            let progress = tag as f64 / 31095.0;
            assert!(
                progress <= LR_PLATEAU_FRACTION,
                "anchor {tag} sits at progress {progress} past LR_PLATEAU_FRACTION, so its \
                 learning rate differs from its partner's and the contrast is collinear again"
            );
        }

        // One epoch: the boundary IS the end of the run, so no bar is ever seen twice and
        // there is nothing to bracket.
        assert!(
            plateau_anchor_tags(&cadence, 10365, 10365, LR_PLATEAU_FRACTION).is_empty(),
            "a single-pass run has no repetition boundary"
        );

        // Two epochs at F = 0.40: the plateau ends at 0.8 passes = step 8292, so the boundary
        // at 10365 has ALREADY entered the LR decay. Anchors there would differ in rate as
        // well as in repetition, which is precisely the confound they exist to avoid.
        assert!(
            plateau_anchor_tags(&cadence, 10365, 20730, LR_PLATEAU_FRACTION).is_empty(),
            "at two epochs the boundary is past the plateau and identifies nothing"
        );

        // The condition is `epochs * F > 1` against the RUN's fraction, not against 0.40: the
        // same two-epoch geometry at F = 0.60 ends its plateau at step 12438, which leaves the
        // boundary 2073 steps inside it, and the anchors come back.
        assert_eq!(
            plateau_anchor_tags(&cadence, 10365, 20730, 0.60),
            vec![10240, 10752, 5120],
            "at F = 0.60 a two-epoch run identifies repetition, and --lr-plateau-fraction is \
             what makes that reachable"
        );

        // Cadence wrote nothing after the boundary: no post-repetition side, no kink.
        let short: Vec<usize> = (1..=20).map(|k| k * DEFAULT_CHECKPOINT_EVERY).collect();
        assert!(
            plateau_anchor_tags(&short, 10365, 31095, LR_PLATEAU_FRACTION).is_empty(),
            "without an artifact past the boundary there is no discontinuity to measure"
        );

        // A random init is never an anchor: it would measure "learned to predict at all".
        let with_zero = vec![0usize, 10240, 10752];
        assert!(
            !plateau_anchor_tags(&with_zero, 10365, 31095, LR_PLATEAU_FRACTION).contains(&0),
            "step 0 is an untrained model and must never anchor a trend"
        );
    }

    /// EVAL-CKPT-004. `pretrain_last.ot` is the LAST OPTIMIZER STEP, not the last validation.
    ///
    /// The defect, measured on job 2884: `last` was written only from `validate`, so a run that
    /// stopped between validations left it holding the newest VALIDATED step. That run was
    /// killed by SIGTERM at step 30780 of 31095 with its newest validation at step 30000, and
    /// that validation had also promoted, so `pretrain_last.ot` and `pretrain_best.ot` held
    /// bit-identical weights: all 140 zip records of the two files share their CRC-32. The
    /// distinct `checkpoint_sha256` that made them look independent is only libtorch writing
    /// the FILE STEM as the zip archive name — `pretrain_best.ot` still contains records named
    /// `pretrain_promotion_candidate/data/*` from the rename in `promote`, which costs 140
    /// bytes per character of stem and says NOTHING about the weights. Meanwhile
    /// `pretrain_step_30720.ot` in the same directory was a genuinely later state.
    ///
    /// So the scenario is asserted in the order a run performs it: promote, take further
    /// optimizer steps, hit the step cadence. `last` must then be the live weights and must NOT
    /// be the promoted ones. Asserted on parameter values read back off the disk, never on
    /// mtimes or paths, because the defect was invisible to both.
    #[test]
    fn last_checkpoint_holds_the_final_step_and_not_the_promoted_weights() {
        let _torch_rng_guard = test_rng::exclusive();
        let (_fx, corpus) = corpus_fixture("laststep");
        let dir = PathBuf::from(corpus.dir());
        let runs = std::env::temp_dir().join(format!(
            "trading_bot_0_pretrain_last_runs_{}",
            uuid::Uuid::new_v4()
        ));
        let mut args = test_args(0x5EED, &dir);
        args.validation_windows = 1;
        args.snapshot_windows = 1;
        // CPU, explicitly: this is bookkeeping about which weights land in which file, none of
        // it device-dependent, and it must never allocate on a card another tenant owns.
        let mut trainer = build_trainer(args, &runs.display().to_string(), Device::Cpu)
            .expect("trainer builds");
        let weights = trainer.run.weights.clone();
        // The fixture's ramp never reaches the deployed context, so the promotion this needs is
        // the forced one `validate` takes on its final step. That is also the promotion that
        // produced the defect, because it writes `best` from the LIVE weights.
        trainer.reached_context = trainer.eval.diagnostic.context;
        trainer
            .validate(0, false, true)
            .expect("the promoting validation runs");
        assert_eq!(
            trainer.promotions, 1,
            "the scenario is meaningless without a promotion for `last` to be confused with"
        );
        let best = weights.join("pretrain_best.ot");
        let promoted = checkpoint_parameters(&best);

        // Optimizer steps AFTER the promotion. `optimizer_step` takes its context from the
        // batch, so the stage-0 sampler is valid at any step index; 4..7 keeps clear of
        // `GROWTH_PROBE_STEPS` and the extra forward passes it would add.
        let sample = {
            let sampler = &trainer.train_samplers[0];
            sampler.batch_of(&sampler.batch_refs(0, 0, 1), trainer.device)
        };
        for step in 4..7 {
            trainer
                .optimizer_step(&sample, step, None)
                .expect("optimizer step");
        }
        let live = live_parameters(&trainer.vs);
        // `is_finite` as well as `> 0.0`: infinity means NOT COMPARABLE, and a bare `> 0.0`
        // would let a missing tensor name, a shape mismatch or a NaN parameter satisfy "the
        // weights moved" without proving anything moved. Provably different, or nothing.
        let moved = max_abs_parameter_diff(&promoted, &live);
        assert!(
            moved > 0.0 && moved.is_finite(),
            "the optimizer steps must actually move the weights, or nothing below is a test; \
             got {moved} (infinity means the two sets are not comparable, which is not evidence \
             of movement)"
        );

        // The step cadence, which is the only thing that refreshes `last` mid-run.
        trainer.write_step_artifacts(6).expect("step artifacts");

        let last = weights.join("pretrain_last.ot");
        let written = checkpoint_parameters(&last);
        assert_eq!(
            max_abs_parameter_diff(&written, &live),
            0.0,
            "pretrain_last.ot must hold the weights after the LAST optimizer step"
        );
        let differs = max_abs_parameter_diff(&written, &promoted);
        assert!(
            differs > 0.0 && differs.is_finite(),
            "pretrain_last.ot still holds the PROMOTED weights, which is the job-2884 defect: \
             `last` written only at validation boundaries leaves a run that took further steps \
             with a file that names the end of the run and holds the last promotion. Got \
             {differs}; infinity would mean the comparison is impossible, not that it passed"
        );
        assert_eq!(
            max_abs_parameter_diff(
                &written,
                &checkpoint_parameters(&weights.join("pretrain_step_6.ot"))
            ),
            0.0,
            "`last` and the step-tagged snapshot of the same step must be the same weights, or a \
             directory listing can again show a step file NEWER than `last`"
        );
        assert_eq!(
            max_abs_parameter_diff(&checkpoint_parameters(&best), &promoted),
            0.0,
            "writing `last` must not disturb the promoted artifact"
        );

        // The file states which step it holds and that nothing selected it, so the next reader
        // cannot mistake a mid-run snapshot for the end of a run without contradicting the
        // artifact itself.
        let metadata = BarWorldModelMetadata::load(&world_model_metadata_path(&last))
            .expect("the last checkpoint's metadata reads back");
        let provenance = metadata
            .training
            .as_ref()
            .expect("every artifact states its provenance");
        assert_eq!(
            provenance.global_step,
            Some(6),
            "pretrain_last.ot must record the optimizer step its weights are from"
        );
        assert_eq!(
            provenance.selection_context, 0,
            "no decision chose `last`, and the file has to say so"
        );
        assert!(
            !window_scores_path(&last).exists(),
            "`last` carries no held-out scores; a sidecar here would attach numbers to a read \
             nobody ever reported"
        );

        std::fs::remove_dir_all(&runs).ok();
    }

    /// Parameters as libtorch wrote them, keyed by name. `Tensor::load_multi` reads the
    /// checkpoint directly instead of going through the world-model loader, so the comparison
    /// is of what is on disk and not of anything a loader might normalize.
    fn checkpoint_parameters(path: &Path) -> BTreeMap<String, Tensor> {
        Tensor::load_multi(path)
            .unwrap_or_else(|error| panic!("{} does not read back: {error}", path.display()))
            .into_iter()
            .collect()
    }

    fn live_parameters(vs: &tch::nn::VarStore) -> BTreeMap<String, Tensor> {
        vs.variables().into_iter().collect()
    }

    /// Largest absolute disagreement over the UNION of both parameter sets, with exactly three
    /// reachable meanings: `0.0` is PROVABLY identical, a positive finite value is PROVABLY
    /// different, and infinity is NOT COMPARABLE.
    ///
    /// Both of the collapses that would make "not comparable" read as `0.0` are closed
    /// deliberately, because every assertion in the test above is `== 0.0` and would pass
    /// VACUOUSLY on either. An EMPTY set compares equal to anything under a fold that starts at
    /// zero, so a checkpoint that read back with no tensors at all would certify as identical to
    /// the live model; that is refused outright. And `f64::max` IGNORES NaN, so one non-finite
    /// parameter would leave `worst` at `0.0` and report two checkpoints full of NaN as
    /// bit-identical — the same shape as `NaN.min(1000.0) == 1000.0` charting an unmeasured
    /// break-even as a measurement. A non-finite difference is promoted to infinity rather than
    /// swallowed.
    fn max_abs_parameter_diff(
        left: &BTreeMap<String, Tensor>,
        right: &BTreeMap<String, Tensor>,
    ) -> f64 {
        assert!(
            !left.is_empty() && !right.is_empty(),
            "a parameter set is EMPTY ({} vs {} tensors), so this comparison certifies nothing \
             and must not be read as agreement",
            left.len(),
            right.len(),
        );
        let mut names: Vec<&String> = left.keys().chain(right.keys()).collect();
        names.sort_unstable();
        names.dedup();
        let mut worst = 0.0f64;
        for name in names {
            let (Some(left), Some(right)) = (left.get(name), right.get(name)) else {
                return f64::INFINITY;
            };
            if left.size() != right.size() {
                return f64::INFINITY;
            }
            let delta = tch::no_grad(|| (left - right).abs().max().double_value(&[]));
            if !delta.is_finite() {
                return f64::INFINITY;
            }
            worst = worst.max(delta);
        }
        worst
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

    /// `pretrain-candles` must picture the run's OWN snapshot windows.
    ///
    /// The whole value of the standalone command is that a mid-ramp checkpoint can be
    /// looked at on the same held-out data the run charts itself. If it drew its own
    /// windows the pictures would depict different symbols at different instants and no
    /// comparison against the run's later snapshots would mean anything.
    ///
    /// The identity holds on the WHOLE `(split, context, count)` triple, and the count is
    /// part of it: `BarSampler::pinned_windows` gives each symbol a quota proportional to
    /// its window count and spaces that symbol's picks evenly across its timeline, so a
    /// smaller count re-spaces every pick rather than truncating the list. `--windows`
    /// must therefore equal the run's `--snapshot-windows`, which is why both default to
    /// the same 8, and this test pins the difference so nobody documents it as a prefix.
    #[test]
    fn standalone_candle_windows_are_the_runs_snapshot_windows() {
        let (_fx, corpus) = corpus_fixture("candlewindows");
        let dir = PathBuf::from(corpus.dir());
        let mut args = test_args(0x5EED, &dir);
        args.snapshot_windows = 4;

        let eval = EvaluationSets::new(&corpus, &args).expect("evaluation sets");
        let standalone = PinnedSet::pinned(
            &corpus,
            Split::Val,
            args.diagnostic_context,
            args.snapshot_windows,
        )
        .expect("standalone set");
        assert_eq!(standalone.sampler.seed(), EVAL_WINDOW_SEED);
        assert_eq!(standalone.context, eval.snapshot.context);
        assert!(!standalone.windows.is_empty());
        assert_eq!(standalone.windows, eval.snapshot.windows);

        // A different count is a different draw, not a prefix of this one.
        let fewer = PinnedSet::pinned(&corpus, Split::Val, args.diagnostic_context, 1)
            .expect("single-window set");
        assert_eq!(fewer.windows.len(), 1);
        if standalone.windows.len() > 1 {
            assert_ne!(fewer.windows[0], standalone.windows[0]);
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
            effective_split_bounds(&args.corpus_flags()).expect("derivation is always allowed"),
            None,
            "--derive-split-bounds must hand the corpus its own percentiles"
        );

        // Contradicting the pin with an explicit pin is a configuration error, not a
        // precedence question.
        args.split_bounds = Some((1, 2));
        assert!(effective_split_bounds(&args.corpus_flags()).is_err());

        // Without the opt-out the default is the campaign constant, and it agrees with the
        // instant the shipped universe was ranked as of.
        args.split_bounds = None;
        args.derive_split_bounds = false;
        assert_eq!(
            effective_split_bounds(&args.corpus_flags())
                .expect("the shipped pin agrees with the ranking"),
            Some(crate::data::ingest::PINNED_SPLIT_BOUNDS)
        );
    }

    /// `dyn` must be reduced exactly as the NextLat reference reduces it: a MEAN over every
    /// element of `[B, T, BAR_MODEL_DIM]`, so a constant per-component residual `c` scores
    /// `smooth_l1(c)` and NOT `BAR_MODEL_DIM * smooth_l1(c)`.
    ///
    /// The width-independence is the point. A sum over the feature axis multiplies the term
    /// by 512 and makes `lambda_dyn` mean a different thing at a different model width; the
    /// reference pairs this reduction with `lambda_mse = 1.0`.
    #[test]
    fn next_lat_loss_means_over_every_element() {
        for width in [BAR_MODEL_DIM, BAR_MODEL_DIM / 2] {
            let target = Tensor::zeros([3, 4, width], (Kind::Float, Device::Cpu));
            for (residual, per_component) in [(0.5f64, 0.125f64), (2.0, 1.5)] {
                let predicted = &target + residual;
                let measured = next_lat_loss(&predicted, &target).double_value(&[]);
                assert!(
                    (measured - per_component).abs() < 1e-4,
                    "width {width}, residual {residual}: dyn {measured} != {per_component}; \
                     a feature-axis SUM would give {}",
                    width as f64 * per_component
                );
            }
        }
    }

    /// A schedule whose three stages are equal in STEPS, which is what the lr, batch and
    /// momentum curves are about.
    ///
    /// Production derives `stage_steps` from the pass partition, so equal stages are no longer
    /// the shape a run has; these tests only need a stage boundary at a known step. `total` is
    /// preserved EXACTLY — the remainder of the division goes to the last stage — because the
    /// lr and momentum curves are parameterized by `total_steps` and a schedule one step short
    /// moves their terminal values.
    fn equal_stages(total: usize, base_batch: usize, batch_ramp: [usize; RAMP_STAGES]) -> Schedule {
        equal_stages_at(total, base_batch, batch_ramp, LR_PLATEAU_FRACTION)
    }

    /// [`equal_stages`] at a stated plateau fraction, for the tests that are about the fraction
    /// itself rather than about the default recipe.
    fn equal_stages_at(
        total: usize,
        base_batch: usize,
        batch_ramp: [usize; RAMP_STAGES],
        lr_plateau_fraction: f64,
    ) -> Schedule {
        let per = total / RAMP_STAGES;
        let mut stage_steps = [per; RAMP_STAGES];
        stage_steps[RAMP_STAGES - 1] = total - per * (RAMP_STAGES - 1);
        Schedule::new(stage_steps, total, base_batch, batch_ramp, lr_plateau_fraction)
    }

    fn schedule(total: usize) -> Schedule {
        equal_stages(total, 8, BATCH_RAMP)
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
        // Plateau, stage 1: bumped by 2**0.6, the reference's exponent for the 2x batch
        // step-up, NOT the square root.
        assert!((s.lr_multiplier(340) - 2.0_f64.powf(0.6)).abs() < 1e-12);
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

    /// `--epochs N` must be N passes over the partition, not N times an average step size.
    ///
    /// The pass plan hands the schedule a per-stage WINDOW count and the step count is
    /// `ceil(windows / batch)` per stage, so every assigned window is issued exactly once per
    /// epoch with at most one short batch per stage. The old derivation priced the run from an
    /// average bar-token rate, which is how job 2856's `--epochs 3` became 1.33 delivered
    /// passes; a step count derived from window counts cannot make that error at all.
    #[test]
    fn the_step_count_issues_every_assigned_window_once_per_epoch() {
        let base_batch = 24;
        // The real corpus's partition under the flat ramp the card actually runs.
        let windows = [82_919usize, 82_917, 82_917];
        for ramp in [[1usize, 1, 1], [1, 2, 3], [1, 1, 2]] {
            let stage_steps = Schedule::steps_for_pass(&windows, base_batch, &ramp);
            for stage in 0..RAMP_STAGES {
                let batch = base_batch * ramp[stage];
                let issued = stage_steps[stage] * batch;
                assert!(
                    issued >= windows[stage],
                    "ramp {ramp:?} stage {stage} issues {issued} of {} assigned windows",
                    windows[stage]
                );
                assert!(
                    issued - windows[stage] < batch,
                    "ramp {ramp:?} stage {stage} over-issues by {} windows, more than the one \
                     short final batch the ceiling costs",
                    issued - windows[stage]
                );
            }
            let epochs = 3usize;
            let per_epoch: usize = stage_steps.iter().sum();
            let schedule = Schedule::new(
                stage_steps,
                per_epoch * epochs,
                base_batch,
                ramp,
                LR_PLATEAU_FRACTION,
            );
            assert_eq!(schedule.steps_per_epoch, per_epoch);
            let boundaries = (0..schedule.total_steps)
                .filter(|&step| schedule.completes_epoch(step))
                .count();
            assert_eq!(boundaries, epochs, "ramp {ramp:?} epoch boundaries");
            // Every epoch is the same partition: the same stage step counts, in the same
            // order. A run whose last epoch is short would under-cover it silently.
            for epoch in 0..epochs {
                let mut per_stage = [0usize; RAMP_STAGES];
                for step in epoch * per_epoch..(epoch + 1) * per_epoch {
                    per_stage[schedule.stage(step)] += 1;
                    assert_eq!(schedule.epoch_of(step), epoch);
                }
                assert_eq!(per_stage, stage_steps, "ramp {ramp:?} epoch {epoch}");
            }
        }
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
        let _torch_rng_guard = test_rng::exclusive();
        // Builds the full trunk, so it fans out exactly like a trainer test does.
        cap_torch_threads();
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
    ///
    /// # What this test guards, and what it deliberately does NOT
    ///
    /// It guards two things about VALUE: that the four shares are magnitudes summing to one,
    /// and that each auxiliary term is a small fraction of the objective's value at its
    /// shipped weight. [`AUX_SHARE_WARN`] and [`Trainer::warn_on_auxiliary_domination`] watch
    /// the same quantity at runtime, since they read `StepLoss::shares`.
    ///
    /// It does NOT govern [`LAMBDA_GROWTH`], and a reader must not conclude from the growth
    /// bound below that the term is negligible. That constant was sized on GRADIENT-NORM
    /// share, measured at 10.3% at step 0 and 19.6% at step 200 on the real corpus at the
    /// deployed batch 24 — see the constant's own doc comment. Value share and gradient share
    /// are different quantities and there is no tension between them: `growth` is ~0.16% of
    /// the loss VALUE while carrying a sixth of the gradient NORM, because `df_raw/dmu_hat =
    /// 1/var_hat` with `var_hat ~ 1e-5` multiplies a tiny per-bar derivative by ~1e5 before it
    /// reaches a parameter. Gradient is what trains; value is incidental.
    ///
    /// Nothing in the tree ASSERTS on gradient share. It is measured and reprinted by
    /// [`probe_growth_gradient_share`] at [`GROWTH_PROBE_STEPS`], which hard-fails only on a
    /// zero or non-finite growth gradient, i.e. on the term being decoration. That is an
    /// observation, not an enforced bound, and it is stated here so the gap is explicit
    /// rather than silently implied by a value-share test sitting next to it.
    #[test]
    fn loss_shares_are_magnitudes_and_sum_to_one() {
        let (nll, dyn_share, kl, growth) = loss_shares(17.0, 28.0, 0.0, 0.0);
        assert!((nll + dyn_share + kl + growth - 1.0).abs() < 1e-12);
        // The regression that motivated the chart: lambda_dyn = 1.0 put dyn at 62%.
        assert!(
            (dyn_share - 28.0 / 45.0).abs() < 1e-12,
            "dyn share {dyn_share}"
        );
        assert!(dyn_share > AUX_SHARE_WARN, "62% must trip the warning");

        // A negative log density must not invert or blow up the denominator.
        let (nll, dyn_share, kl, growth) = loss_shares(-30.0, 10.0, 10.0, 0.0);
        assert!((nll + dyn_share + kl + growth - 1.0).abs() < 1e-12);
        assert!((nll - 0.6).abs() < 1e-12, "nll share {nll}");
        // A zero objective has no shares to report, and reporting zeros would draw a
        // four-way tie that never happened.
        assert!(loss_shares(0.0, 0.0, 0.0, 0.0).0.is_nan());

        // The reference weight keeps the auxiliary term well inside the threshold at the
        // production init figures: `dyn` measured 245 under the SUMMED reduction at step 0,
        // i.e. 245/512 = 0.479 under the reference mean, against a categorical-scale
        // `|nll|` of 24.26.
        let (_, dyn_share, _, _) =
            loss_shares(24.26, 1.0 * (245.0 / BAR_MODEL_DIM as f64), 0.0, 0.0);
        assert!(
            dyn_share < AUX_SHARE_WARN,
            "the reference lambda_dyn = 1.0 leaves dyn at {dyn_share} of the objective"
        );

        // The growth term's VALUE share at the shipped weight, on the production init
        // figures: `|nll|` 24.26 on the categorical scale, `dyn` 0.479 under the reference
        // mean reduction, `kl` 1.0, and the growth term at its whole tradeable content of
        // 5.25e-4 nats/bar. Measures 1.57e-3 at `LAMBDA_GROWTH` = 77.
        //
        // The bound is 5e-3, which is a real constraint and not a rubber stamp: it is 3.2x
        // the current reading and it TRIPS at lambda ~247, well inside the order of magnitude
        // Main asked it to catch. A violation would mean one of two things, both of which
        // want a human. Either `LAMBDA_GROWTH` was raised to make the objective-share chart
        // look respectable — which is sizing the term on the wrong quantity, since the
        // constant is derived from gradient norm — or the growth term's magnitude itself moved
        // by more than 3x, which would mean the 5.25e-4 measurement no longer describes the
        // corpus and the whole premise of the term needs re-deriving.
        //
        // It is NOT a claim that the term is small in any sense that matters. See this test's
        // doc comment: the gradient share at this same weight is 10.3% and 19.6%.
        let (_, _, _, growth_share) = loss_shares(24.26, 0.479, 1.0, LAMBDA_GROWTH * 5.25e-4);
        assert!(
            growth_share < 5e-3,
            "the growth term holds {growth_share} of the objective's MAGNITUDE at \
             lambda_growth = {LAMBDA_GROWTH}; either the weight was raised against the \
             objective-share chart instead of the gradient-norm probe, or the term's \
             magnitude has moved off the measured 5.25e-4 nats/bar"
        );
    }

    /// Holding a ramp stage's batch must move the learning-rate plateau bump with it. A
    /// schedule that kept the planned `sqrt(3)` bump while running the previous stage's
    /// batch would be training at 1.73x the rate the batch justifies.
    #[test]
    fn holding_the_batch_moves_the_lr_plateau_bump() {
        let mut schedule = equal_stages(3000, 16, BATCH_RAMP);
        let stage_1 = 1200; // inside stage 1 of three equal stages.
        assert_eq!(schedule.stage(stage_1), 1);
        assert_eq!(schedule.batch(stage_1), 32);
        assert!((schedule.lr_multiplier(stage_1) - 2.0f64.powf(0.6)).abs() < 1e-12);
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

    /// The plateau bump follows modded-nanogpt's per-stage exponents, NOT a uniform square
    /// root: `(16/8)**0.6` for the 2x step and `(24/8)**0.5` for the 3x step
    /// (`train_gpt.py:1980-1985`). Copying `0.5` everywhere ran stage 1 at 1.414x where the
    /// reference runs 1.516x, i.e. 7% under.
    #[test]
    fn the_plateau_bump_matches_the_reference_exponents() {
        let schedule = equal_stages(3000, 8, BATCH_RAMP);
        // Sampled inside the flat plateau so the decay does not confound the bump; stage 2
        // starts at 2/3, past LR_PLATEAU_FRACTION, so its exponent is checked directly.
        assert!((schedule.lr_multiplier(0) - 1.0).abs() < 1e-12);
        assert!((schedule.lr_multiplier_for(1100, 2) - 1.515_716_5).abs() < 1e-6);
        assert!((schedule.lr_multiplier_for(0, 3) - 3.0f64.sqrt()).abs() < 1e-12);
        // A uniform square root would have given 1.4142 at the 2x step.
        assert!((schedule.lr_multiplier_for(1100, 2) - 2.0f64.sqrt()).abs() > 0.09);
    }

    /// The DEFAULT schedule is BIT-IDENTICAL to the one every persisted artifact in the tree
    /// was produced under.
    ///
    /// `--lr-plateau-fraction` turned the fraction from a module constant into a value, and the
    /// entire point of its default is that it changes nothing: `bardist_v2` at step 10364 and
    /// `bardist_v3_rfirst_1ep` at 10817 are comparable to a future run only if the curve under
    /// them did not move. The reference here is the pre-change formula transcribed literally —
    /// `batch_multiple**exponent` while `progress <= 0.40`, then EXACTLY AFFINE to an absolute
    /// 0.15 — with both numbers spelled out rather than read from the constants, so an edit to
    /// either one fails here instead of silently rebasing every persisted comparison. Probed at
    /// both ends, the midpoint and BOTH SIDES of the plateau boundary, which is the one step
    /// where an off-by-one in the comparison could hide.
    #[test]
    fn the_default_plateau_fraction_reproduces_the_pre_change_schedule() {
        const F: f64 = 0.40;
        const FLOOR: f64 = 0.15;
        assert!((LR_PLATEAU_FRACTION - F).abs() < 1e-12);
        assert!((LR_FLOOR_MULTIPLIER - FLOOR).abs() < 1e-12);
        // 10817 steps is `bardist_v3_rfirst_1ep`'s geometry: the one-epoch run whose re-decoded
        // slope of 0.6653 +/- 0.0286 this flag exists to explain.
        let total = 10_817usize;
        for ramp in [[1, 1, 1], BATCH_RAMP] {
            let schedule = equal_stages(total, 24, ramp);
            let boundary = schedule.plateau_last_step();
            assert_eq!(boundary, 4326, "floor(0.40 * {total})");
            for step in [
                0,
                1,
                boundary - 1,
                boundary,
                boundary + 1,
                total / 2,
                total - 1,
                total,
            ] {
                let stage = schedule.stage(step);
                let plateau = (ramp[stage] as f64).powf(BATCH_RAMP_LR_EXPONENT[stage]);
                let progress = step as f64 / total as f64;
                let expected = if progress <= F {
                    plateau
                } else {
                    plateau + (FLOOR - plateau) * ((progress - F) / (1.0 - F)).min(1.0)
                };
                let actual = schedule.lr_multiplier(step);
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "ramp {ramp:?} step {step}: {actual} is not the pre-change {expected}"
                );
            }
            // The boundary is a boundary: flat up to and including it, strictly lower one step
            // later, and the floor reached EXACTLY rather than approached.
            assert!(schedule.in_lr_plateau(boundary));
            assert!(!schedule.in_lr_plateau(boundary + 1));
            assert!(schedule.lr_multiplier(boundary + 1) < schedule.lr_multiplier(boundary));
            assert!((schedule.lr_multiplier(total) - FLOOR).abs() < 1e-12);
        }
    }

    /// A widened plateau MOVES the schedule, which is the only reason the flag exists.
    ///
    /// Under a one-epoch budget the default ends the plateau at 0.4 passes, so the run always
    /// finishes fully annealed and the operating point where the re-decoded mean slope last
    /// measured 1.0058 +/- 0.0355 — one full pass still at peak rate — is unreachable. At
    /// `F = 0.90` the same single pass is still at the peak multiplier at 0.8 of the run, where
    /// the default is already down at 0.43x, so the rank-1 confound
    /// [`Schedule::passes_per_lr_unit`] describes becomes measurable instead of algebraic.
    #[test]
    fn a_widened_plateau_holds_peak_lr_where_the_default_has_already_decayed() {
        let total = 10_817usize;
        let widened = equal_stages_at(total, 24, [1, 1, 1], 0.90);
        let default = equal_stages(total, 24, [1, 1, 1]);
        // One epoch, i.e. exactly the geometry the flag is for.
        assert_eq!(widened.total_steps(), widened.steps_per_epoch());
        let step = (0.8 * total as f64) as usize;
        assert!(widened.in_lr_plateau(step));
        assert!((widened.lr_multiplier(step) - 1.0).abs() < 1e-12);
        assert!(!default.in_lr_plateau(step));
        let decayed = default.lr_multiplier(step);
        let affine = 1.0 + (LR_FLOOR_MULTIPLIER - 1.0) * (0.8 - 0.4) / 0.6;
        assert!((decayed - affine).abs() < 1e-3, "{decayed} vs {affine}");
        assert!(
            decayed < 0.5,
            "the default is well into its decay at 0.8 passes: {decayed}"
        );
        // Every quantity derived from the fraction moves with it, or a banner, a metadata
        // sidecar or a report would describe the default recipe while the run ran another one.
        assert_eq!(widened.plateau_last_step(), 9735);
        assert_eq!(default.plateau_last_step(), 4326);
        let identity = -0.10 / (1.0 - LR_FLOOR_MULTIPLIER);
        assert!(
            (widened.passes_per_lr_unit() - identity).abs() < 1e-12,
            "{} vs {identity}",
            widened.passes_per_lr_unit()
        );
    }

    /// Supports fitted from bars shaped like real ones — mass piled on flat ranges and a
    /// spread of mid-range closes — so the equal-mass bins and the atoms both exist.
    fn synthetic_supports() -> BarSupports {
        let mut rng = ChaCha12Rng::seed_from_u64(0x5EED_1234);
        let samples: Vec<BarDof> = (0..8192)
            .map(|index| {
                if index % 32 == 0 {
                    return BarDof::default();
                }
                BarDof {
                    r: rng.random_range(-0.03f32..0.03),
                    s: rng.random_range(0.0f32..0.05),
                    u: rng.random_range(0.0f32..1.0),
                    v: rng.random_range(0.0f32..1.0),
                    w: rng.random_range(-2.0f32..2.0),
                }
            })
            .collect();
        BarSupports::fit(&samples)
    }

    fn synthetic_window(batch: i64, len: i64, seed: u64) -> (Tensor, Tensor) {
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        let values: Vec<f32> = (0..batch * len)
            .flat_map(|_| {
                [
                    rng.random_range(-0.03f32..0.03),
                    rng.random_range(0.0f32..0.05),
                    rng.random_range(0.0f32..1.0),
                    rng.random_range(0.0f32..1.0),
                    rng.random_range(-2.0f32..2.0),
                ]
            })
            .collect();
        let dof = Tensor::from_slice(&values)
            .reshape([batch, len, BAR_DOF as i64])
            .to_kind(Kind::Float);
        let time_ids = Tensor::zeros(
            [batch, len, BAR_TIME_FEATURES as i64],
            (Kind::Int64, Device::Cpu),
        );
        (dof, time_ids)
    }

    /// The auxiliary weights are the CONFIGURED ones at the first step, the middle step and
    /// the last step alike. There is no schedule on them, by design — see
    /// [`Args::lambda_dyn`].
    ///
    /// The assertion that matters is the gradient one. `bar_dyn_fc3_w` is the dynamics
    /// head's output projection, and the only paths that reach it are the two auxiliary
    /// terms; if either weight is scaled to zero its gradient is EXACTLY zero and the head
    /// stops tracking the trunk while the trunk keeps training. That is precisely what the
    /// deleted `2/3` anneal did, and it is why job 2865 shipped a dynamics head 154x worse
    /// than the trivial identity map. Under the anneal this test fails at the final step.
    #[test]
    fn the_auxiliary_weights_are_the_configured_ones_at_every_step_of_the_run() {
        let _torch_rng_guard = test_rng::exclusive();
        // Builds the full trunk and backpropagates through it, so it fans out exactly like
        // a trainer test does.
        cap_torch_threads();
        let total = 3000usize;
        let schedule = equal_stages(total, 8, BATCH_RAMP);
        // The three probes span the ramp: the first stage, the middle, and the final stage
        // the anneal used to zero out.
        let steps = [0usize, total / 2, total - 1];
        assert_eq!(schedule.stage(steps[0]), 0);
        assert!(!schedule.in_final_stage(steps[1]));
        assert!(schedule.in_final_stage(steps[2]));

        // Deliberately unequal and not 1.0, so a term picking up the wrong lambda cannot
        // hide behind another one or behind a multiply-by-one.
        let (lambda_dyn, lambda_kl, lambda_growth) = (3e-2f64, 7e-2f64, 11e-2f64);
        let (batch, context, horizon) = (2i64, 12i64, 2i64);

        tch::manual_seed(0x0B3E);
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        let supports = synthetic_supports();
        let growth_support = GrowthSupport::new(&supports, Device::Cpu)
            .expect("the synthetic support carries the growth term");
        let (dof, time_ids) = synthetic_window(batch, context + 1, 0xB0B0);

        let mut totals: Vec<f64> = Vec::with_capacity(steps.len());
        for step in steps {
            for mut variable in vs.trainable_variables() {
                variable.zero_grad();
            }
            let graph = forward_losses(
                &modules,
                &supports,
                &growth_support,
                &dof,
                &time_ids,
                context,
                horizon,
                lambda_dyn,
                lambda_kl,
                lambda_growth,
                BarScoring::Density,
                Device::Cpu,
            );
            // The configured lambdas, and nothing else, assemble the objective. The
            // tolerance is f32-relative: the graph accumulates in f32, so recomposing the
            // sum in f64 from f32-rounded terms cannot reproduce it to more than ~1e-7.
            let expected = graph.nll.double_value(&[])
                + lambda_dyn * graph.dyn_loss.double_value(&[])
                + lambda_kl * graph.kl_loss.double_value(&[])
                + lambda_growth * graph.growth.double_value(&[]);
            let total_loss = graph.loss.double_value(&[]);
            assert!(
                (total_loss - expected).abs() <= 1e-6 * (1.0 + total_loss.abs()),
                "step {step}: objective {total_loss} is not nll + {lambda_dyn}*dyn + \
                 {lambda_kl}*kl + {lambda_growth}*growth = {expected}"
            );

            graph.loss.backward();
            let fc3 = vs.variables();
            let grad = fc3
                .get("bar_dyn_fc3_w")
                .expect("the dynamics output projection is a trainable variable")
                .grad();
            assert!(
                grad.defined(),
                "step {step}: the dynamics head received no gradient at all"
            );
            let magnitude = grad.abs().sum(Kind::Float).double_value(&[]);
            assert!(
                magnitude > 0.0,
                "step {step}: the dynamics head's gradient is exactly zero, so it is no \
                 longer being trained while the trunk still is"
            );
            totals.push(total_loss);
        }

        // Same batch, same objective, at the first and last step of the run.
        for (step, total_loss) in steps.iter().zip(&totals) {
            assert!(
                (total_loss - totals[0]).abs() <= 1e-6 * (1.0 + totals[0].abs()),
                "step {step}: the objective moved to {total_loss} from {} at step 0, so \
                 something still weights the auxiliaries by the step",
                totals[0]
            );
        }

        // The negative control, which is the deleted anneal's terminal state: scale every
        // auxiliary to zero and the dynamics head's gradient vanishes EXACTLY. This is what
        // the run did for its final third, and it is what the assertions above would have
        // caught. Keeping it here means the gradient check cannot silently become vacuous —
        // if it ever stops discriminating, this half fails too.
        //
        // `lambda_growth` is zeroed with the others even though the growth term does not
        // reach the dynamics MLP: leaving it live would make the control's claim "the
        // dynamics head gets nothing from a zero-weighted auxiliary" rest on the growth
        // term's graph shape rather than on the weights, and that is a weaker statement.
        for mut variable in vs.trainable_variables() {
            variable.zero_grad();
        }
        let annealed = forward_losses(
            &modules,
            &supports,
            &growth_support,
            &dof,
            &time_ids,
            context,
            horizon,
            0.0,
            0.0,
            0.0,
            BarScoring::Density,
            Device::Cpu,
        );
        annealed.loss.backward();
        let dead = vs.variables()["bar_dyn_fc3_w"]
            .grad()
            .abs()
            .sum(Kind::Float)
            .double_value(&[]);
        assert_eq!(
            dead, 0.0,
            "with both auxiliary weights at zero the dynamics head must receive no gradient \
             at all; it received {dead}, so this test's gradient check proves nothing"
        );
    }

    /// `--lambda-growth 0` must be EXACTLY inert, not nearly inert.
    ///
    /// Two things depend on that being bit-exact rather than approximate. The ablation's
    /// control arm is only a single-variable comparison if the zero-weight objective is the
    /// pre-change objective to the last bit; and `PromotionGate`'s selection rule reads the
    /// control arm's numbers as the baseline the economic rule is judged against, so a
    /// last-digit difference there would be attributed to the selection rule.
    ///
    /// It is a real assertion and not a tautology about `0.0 * x`, because `0.0 * x` is NOT
    /// zero for every `x`: at `x = inf` or `x = NaN` it is NaN, and `total + NaN` is NaN.
    /// So this is exactly the test that the growth term's clamps and log-argument guard hold
    /// on real model output — a term that quietly produced an inf on some bar would poison
    /// the CONTROL arm of its own ablation, which is the most confusing failure available
    /// here.
    #[test]
    fn a_zero_growth_weight_leaves_the_objective_and_its_gradients_bit_identical() {
        let _torch_rng_guard = test_rng::exclusive();
        cap_torch_threads();
        let (batch, context, horizon) = (2i64, 12i64, 2i64);
        let (lambda_dyn, lambda_kl) = (3e-2f64, 7e-2f64);

        tch::manual_seed(0x0C0F);
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        let supports = synthetic_supports();
        let growth_support = GrowthSupport::new(&supports, Device::Cpu)
            .expect("the synthetic support carries the growth term");
        let (dof, time_ids) = synthetic_window(batch, context + 1, 0xC0FE);

        // Sorted, so the two readings are comparable element by element: `vs.variables()`
        // hands back a `HashMap` whose order is seeded per process.
        let grads = |vs: &nn::VarStore| -> Vec<(String, f64)> {
            let mut out: Vec<(String, f64)> = named_trainable_variables(vs)
                .into_iter()
                .filter(|(_, tensor)| tensor.grad().defined())
                .map(|(name, tensor)| {
                    (
                        name,
                        tensor
                            .grad()
                            .to_kind(Kind::Double)
                            .abs()
                            .sum(Kind::Double)
                            .double_value(&[]),
                    )
                })
                .collect();
            out.sort_by(|a, b| a.0.cmp(&b.0));
            out
        };
        let zero = || {
            for mut variable in vs.trainable_variables() {
                variable.zero_grad();
            }
        };

        // The shipped objective at zero growth weight.
        zero();
        let graph = forward_losses(
            &modules,
            &supports,
            &growth_support,
            &dof,
            &time_ids,
            context,
            horizon,
            lambda_dyn,
            lambda_kl,
            0.0,
            BarScoring::Density,
            Device::Cpu,
        );
        // The growth term still RAN — the control arm charts it, so a broken term would be
        // caught here rather than silently skipped. And it is finite, which is what makes the
        // `0.0 *` above a no-op.
        let growth_value = graph.growth.double_value(&[]);
        assert!(
            growth_value.is_finite(),
            "the growth term is {growth_value} on the control arm, so multiplying it by a \
             zero weight cannot leave the objective unchanged"
        );
        let with_zero_weight = graph.loss.double_value(&[]);
        graph.loss.backward();
        let grads_with_zero_weight = grads(&vs);

        // The pre-change objective, reconstructed from the SAME graph's terms: three terms,
        // no growth summand at all.
        zero();
        let again = forward_losses(
            &modules,
            &supports,
            &growth_support,
            &dof,
            &time_ids,
            context,
            horizon,
            lambda_dyn,
            lambda_kl,
            0.0,
            BarScoring::Density,
            Device::Cpu,
        );
        let three_terms =
            &again.nll + lambda_dyn * &again.dyn_loss + lambda_kl * &again.kl_loss;
        let without_the_term = three_terms.double_value(&[]);
        three_terms.backward();
        let grads_without_the_term = grads(&vs);

        assert_eq!(
            with_zero_weight, without_the_term,
            "a zero growth weight moved the objective from {without_the_term} to \
             {with_zero_weight}, so the ablation's control arm is not the pre-change run"
        );
        assert_eq!(
            grads_with_zero_weight.len(),
            grads_without_the_term.len(),
            "a zero growth weight changed WHICH parameters receive a gradient"
        );
        for ((name, with), (other, without)) in grads_with_zero_weight
            .iter()
            .zip(grads_without_the_term.iter())
        {
            assert_eq!(name, other, "the two gradient readings are not aligned");
            assert_eq!(
                with, without,
                "a zero growth weight moved {name}'s gradient from {without} to {with}"
            );
        }
    }

    /// The growth term ALONE must reach the trunk, with the likelihood contributing nothing.
    ///
    /// This is the assertion that separates an objective from a diagnostic. Every plausible
    /// way of getting the marginalization wrong — a `no_grad` around the mixture, a `detach`
    /// on the belief, reading the moments off a frozen logit table — leaves a term that still
    /// prints a sensible number and still charts, while training nothing. `nll` is never
    /// backwarded here, so the only path from the loss to a trunk weight runs through
    /// `mu_hat` and `var_hat`, which is exactly the path the finding says the objective was
    /// missing.
    ///
    /// # The zero-init head, measured
    ///
    /// At EXACTLY step 0 the trunk correctly receives nothing, and this test asserts that
    /// too rather than papering over it. `BarEmissionHead` is zero-init, so the `r` row is
    /// `logits = 0 * h + 0`; `d logits / d h` is the weight matrix, which is exactly zero, so
    /// the whole growth gradient lands on the head's own weights (`d logits / d W = h`) and
    /// none of it on the representation. That is a property of the initialization, not of the
    /// term: the path opens as soon as the head is non-zero, i.e. after the first optimizer
    /// step. Both halves are pinned here because the first is the reason the second cannot be
    /// tested at init, and a future reader who deletes the perturbation would get a failure
    /// they would be tempted to blame on the marginalization.
    ///
    /// The second half also pins WHERE the gradient lands. Reaching only `bar_dof_head` and
    /// the prefix embedding would mean the term can rescale the emission head's `r` row but
    /// cannot ask the representation for a better conditional mean — and the representation
    /// is what the run's 0.068-nat drift was free to move.
    #[test]
    fn the_growth_term_alone_reaches_the_trunk() {
        let _torch_rng_guard = test_rng::exclusive();
        cap_torch_threads();
        let (batch, context, horizon) = (2i64, 12i64, 2i64);

        tch::manual_seed(0x0D0D);
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        let supports = synthetic_supports();
        let growth_support = GrowthSupport::new(&supports, Device::Cpu)
            .expect("the synthetic support carries the growth term");
        let (dof, time_ids) = synthetic_window(batch, context + 1, 0xD0D0);

        let is_head = |name: &str| {
            BAR_EMISSION_ADAMW_NAME_SUBSTRINGS
                .iter()
                .any(|part| name.contains(part))
        };
        // The growth term trains the trunk and the emission head. `bar_dyn` is the dynamics
        // MLP, which only the NextLat auxiliaries reach, so it is neither.
        let magnitudes = |vs: &nn::VarStore| -> (Vec<(String, f64)>, Vec<(String, f64)>) {
            let mut trunk = Vec::new();
            let mut head = Vec::new();
            for (name, tensor) in named_trainable_variables(vs) {
                if name.contains("bar_dyn") {
                    continue;
                }
                let grad = tensor.grad();
                let magnitude = if grad.defined() {
                    grad.to_kind(Kind::Double)
                        .abs()
                        .sum(Kind::Double)
                        .double_value(&[])
                } else {
                    0.0
                };
                if is_head(&name) {
                    head.push((name, magnitude));
                } else {
                    trunk.push((name, magnitude));
                }
            }
            (trunk, head)
        };
        let backward_only = |want_growth: bool| {
            for mut variable in vs.trainable_variables() {
                variable.zero_grad();
            }
            let graph = forward_losses(
                &modules,
                &supports,
                &growth_support,
                &dof,
                &time_ids,
                context,
                horizon,
                0.0,
                0.0,
                0.0,
                BarScoring::Density,
                Device::Cpu,
            );
            // One term only. `graph.loss` is never touched, so the other three terms
            // contribute exactly zero to what follows.
            if want_growth {
                graph.growth.backward();
            } else {
                graph.nll.backward();
            }
        };
        let backward_growth_only = || backward_only(true);

        // Half one: at zero init the head learns and the trunk cannot, exactly as the weight
        // matrix being zero requires.
        backward_growth_only();
        let (trunk, head) = magnitudes(&vs);
        assert!(
            !trunk.is_empty() && !head.is_empty(),
            "the parameter split found {} trunk and {} head tensors, so this test asserts \
             nothing",
            trunk.len(),
            head.len()
        );
        assert!(
            head.iter().any(|(_, g)| *g > 0.0),
            "the growth term reached none of the {} emission-head tensors even at zero init, \
             so it is disconnected from the model entirely",
            head.len()
        );
        for (name, magnitude) in &trunk {
            assert_eq!(
                *magnitude, 0.0,
                "{name} received {magnitude} from the growth term at zero init, but the head's \
                 weight matrix is exactly zero there, so no gradient can reach the trunk \
                 through it; something else is feeding the trunk"
            );
        }

        // Half two: give the head weights, and the representation becomes trainable toward
        // the conditional mean. Small, so the perturbation cannot be what produces the
        // gradient — it only opens the path.
        tch::no_grad(|| {
            for (name, mut tensor) in named_trainable_variables(&vs) {
                if is_head(&name) {
                    let noise = Tensor::randn_like(&tensor) * 0.02;
                    let _ = tensor.g_add_(&noise);
                }
            }
        });
        backward_growth_only();
        let (trunk, _) = magnitudes(&vs);
        let reached = trunk.iter().filter(|(_, g)| *g > 0.0).count();
        assert!(
            reached > 0,
            "with a non-zero emission head the growth term still reached none of the {} trunk \
             parameters, so it is a diagnostic rather than an objective: it can only be read, \
             never trained toward",
            trunk.len()
        );
        // Two invariants instead of a raw fraction, because a fraction would be measuring the
        // INITIALIZATION. This trunk zero-inits its residual output projections, so at init a
        // gradient stops at the first zero it meets and only 45 of the 117 trunk tensors are
        // reachable at all — by `nll` just as much as by `growth`.
        backward_only(false);
        let (nll_trunk, _) = magnitudes(&vs);
        let live = |rows: &[(String, f64)]| -> BTreeSet<String> {
            rows.iter()
                .filter(|(_, g)| *g > 0.0)
                .map(|(name, _)| name.clone())
                .collect()
        };
        let (growth_live, nll_live) = (live(&trunk), live(&nll_trunk));

        // One: growth cannot reach anything the likelihood cannot. A parameter that only the
        // growth term touches would mean the two terms read different representations.
        let extra: Vec<&String> = growth_live.difference(&nll_live).collect();
        assert!(
            extra.is_empty(),
            "the growth term reached trunk parameters the likelihood cannot reach on the same \
             batch: {extra:?}"
        );

        // Two: it traverses the FULL DEPTH. This is the assertion that "only the last
        // projection is learning" would fail, and it is stated per layer rather than as a
        // count so it cannot be satisfied by a wide gradient in one block. Set equality is
        // deliberately NOT asserted: the growth gradient is ~4 orders smaller than the
        // likelihood's, so a handful of the tiniest scalars (measured: 2 of 47, both
        // `attn_resid_lambda` in the last layers) underflow f32 to exactly zero. That is
        // arithmetic, not detachment, and pinning it would make this test a float-noise
        // detector.
        let layers: BTreeSet<String> = nll_live
            .iter()
            .filter(|name| name.starts_with("bar_layer_"))
            .filter_map(|name| name.split('.').next().map(str::to_owned))
            .collect();
        assert!(
            layers.len() > 1,
            "only {} transformer layers are reachable at all, so a depth assertion proves \
             nothing here",
            layers.len()
        );
        for layer in &layers {
            assert!(
                growth_live.iter().any(|name| name.starts_with(layer)),
                "the growth term's gradient never reaches {layer}, so it trains the layers \
                 above it and leaves the representation beneath untouched; the likelihood \
                 reaches {} trunk tensors and growth {}",
                nll_live.len(),
                growth_live.len()
            );
        }
    }

    /// The end-of-run guard: a shipped dynamics head that loses to `z_k = h_t` is a hard
    /// failure, and one that beats it is not.
    ///
    /// Both ratios here are MEASURED through [`dynamics_losses`] rather than asserted as
    /// literals, so the test exercises the same quantity `dyn_identity_ratio` pools over the
    /// test split and cannot pass against a guard reading a number nothing produces.
    #[test]
    fn the_end_of_run_guard_fires_on_a_stale_dynamics_head_and_passes_on_a_healthy_one() {
        let _torch_rng_guard = test_rng::exclusive();
        // Two full module bundles plus 120 descent steps through the dynamics MLP.
        cap_torch_threads();
        let (batch, context, horizon) = (2i64, 12i64, 1i64);
        let beliefs = drifting_beliefs(batch, context, 0.25, 0xD1D1);
        let (dof, time_ids) = synthetic_window(batch, context + 1, 0xFEED);
        let bins = Tensor::zeros(
            [batch, context + 1, BAR_DOF as i64],
            (Kind::Int64, Device::Cpu),
        );
        let measure = |modules: &BarModules| {
            let (dyn_loss, _, identity) = dynamics_losses(
                modules,
                &dof,
                &bins,
                &time_ids,
                &beliefs,
                context,
                horizon,
                Device::Cpu,
            );
            (dyn_loss, identity.double_value(&[]))
        };

        // STALE: a dynamics head whose weights bear no relation to the beliefs it is asked
        // to advance. That is where a head ends up when its only training signal is switched
        // off and the trunk keeps moving underneath it.
        let stale_vs = nn::VarStore::new(Device::Cpu);
        let stale_modules = BarModules::new(&stale_vs.root());
        tch::no_grad(|| {
            for variable in stale_vs.trainable_variables() {
                let mut variable = variable;
                let _ = variable.normal_(0.0, 0.5);
            }
        });
        let (stale_dyn, stale_identity) = measure(&stale_modules);
        let stale = stale_dyn.double_value(&[]) / stale_identity;
        assert!(
            stale > 1.0,
            "the stale fixture is not actually worse than the identity map ({stale})"
        );
        let error = check_dynamics_beats_identity(stale, horizon)
            .expect_err("a dynamics head worse than doing nothing must fail the run");
        assert!(
            error.to_string().contains("WORSE THAN DOING NOTHING"),
            "unhelpful guard message: {error}"
        );

        // HEALTHY: a head that has actually been fitted to the beliefs it advances, which is
        // what a live NextLat term produces over a run. A zero-init head is EXACTLY the
        // identity map, so descending `dyn` from there is what carries the ratio below 1 and
        // the starting point is the boundary the guard sits on. Only the `bar_dyn_*`
        // variables move; nothing else reaches `dyn`.
        let healthy_vs = nn::VarStore::new(Device::Cpu);
        let healthy_modules = BarModules::new(&healthy_vs.root());
        let (start_dyn, start_identity) = measure(&healthy_modules);
        let start = start_dyn.double_value(&[]) / start_identity;
        assert!(
            (start - 1.0).abs() < 1e-3,
            "a zero-init dynamics head must sit exactly on the identity map, not at {start}"
        );
        for _ in 0..120 {
            for mut variable in healthy_vs.trainable_variables() {
                variable.zero_grad();
            }
            let (dyn_loss, _) = measure(&healthy_modules);
            dyn_loss.backward();
            tch::no_grad(|| {
                for (name, variable) in healthy_vs.variables() {
                    if !name.starts_with("bar_dyn_") {
                        continue;
                    }
                    let grad = variable.grad();
                    if grad.defined() {
                        let mut variable = variable;
                        let _ = variable.f_sub_(&(grad * 0.5)).expect("descent step");
                    }
                }
            });
        }
        let (healthy_dyn, healthy_identity) = measure(&healthy_modules);
        let healthy = healthy_dyn.double_value(&[]) / healthy_identity;
        assert!(
            healthy < 1.0,
            "the fitted dynamics head still loses to the identity map ({healthy}), so the \
             healthy branch is not being exercised"
        );
        check_dynamics_beats_identity(healthy, horizon)
            .expect("a dynamics head that beats the identity map must pass");

        // A degenerate baseline certifies nothing and must not pass either.
        check_dynamics_beats_identity(f64::NAN, horizon)
            .expect_err("a non-finite ratio must not be treated as a passing run");
    }

    /// The card of record: an RTX 5090, 32,607 MiB total, at the per-bar-token cost job 2856
    /// measured at the deployed context.
    ///
    /// `free_bytes` is the total less the 4.46 GiB job 2856 held at its stage-2 probe with the
    /// allocator pool warm (31.84 total - 22.65 activations - 4.73 free), so it stands for an
    /// IDLE card carrying this process's own weights, gradients, optimizer state and CUDA
    /// context. `fixed_bytes` is zero because the two-point fit puts the whole
    /// batch-independent cost in the intercept and the production probe never reported it
    /// separately; the frontier is therefore quoted at the same footing as the measurements it
    /// is validated against.
    fn measured_5090() -> CapacityModel {
        let total = 32_607u64 * (1u64 << 20);
        let baseline_bytes = (4.46 * (1u64 << 30) as f64) as u64;
        CapacityModel {
            per_token_bytes: 494_883.0,
            fixed_bytes: 0.0,
            free_bytes: total - baseline_bytes,
            baseline_bytes,
        }
    }

    /// The model must reproduce what the card DID. Job 2856 ran 24 windows at 2048 bars and
    /// measured 22.65 GiB of activations with 4.73 GiB still free — so a derivation that
    /// refuses batch 24 at the deployed context is wrong, and one that allows much more than
    /// 24 would have OOMed. This pins the calibration the whole ramp rests on.
    #[test]
    fn the_capacity_model_reproduces_the_measured_stage_two_footprint() {
        let capacity = measured_5090();
        let deployed = stage_context(RAMP_STAGES - 1);
        assert_eq!(deployed, 2048);
        let measured_gib = CapacityModel::gib(capacity.step_bytes(24, deployed));
        assert!(
            (measured_gib - 22.65).abs() < 0.05,
            "24x2048 should price at the measured 22.65 GiB, got {measured_gib}"
        );
        // Right at the ceiling, which is what "4.73 GiB free" means: one more window would
        // not have cleared the margin plus the reserve.
        assert_eq!(capacity.frontier_batch(deployed), 24);
    }

    /// THE central invariant. Every stage of the derived ramp must fit in measured capacity,
    /// for every base batch and every card size, because the whole point of deriving it is
    /// that the banner's schedule is the one that executes. A stage that does not fit is a
    /// stage the runtime hold will rewrite, which is the defect this replaces.
    #[test]
    fn the_derived_ramp_never_plans_a_stage_over_measured_capacity() {
        let base = measured_5090();
        for free_gib in [8.0, 14.0, 20.0, 27.4, 40.0, 80.0, 160.0] {
            for &fixed_gib in &[0.0, 0.75] {
                let capacity = CapacityModel {
                    fixed_bytes: fixed_gib * (1u64 << 30) as f64,
                    free_bytes: (free_gib * (1u64 << 30) as f64) as u64,
                    ..base
                };
                let deployed = stage_context(RAMP_STAGES - 1);
                let ceiling = capacity.frontier_batch(deployed);
                if ceiling == 0 {
                    continue;
                }
                for requested in [1, 4, 12, 24, 48, 72, 256] {
                    let plan = resolve_ramp(Some(&capacity), requested, false)
                        .expect("a positive ceiling always yields a plan");
                    assert!(
                        plan.base_batch <= requested && plan.base_batch > 0,
                        "{free_gib} GiB free, requested {requested}: base {}",
                        plan.base_batch
                    );
                    let mut previous = 0.0;
                    for stage in 0..RAMP_STAGES {
                        let context = stage_context(stage);
                        let batch = plan.base_batch * plan.batch_ramp[stage];
                        let required = capacity.required_bytes(batch, context, previous);
                        assert!(
                            required <= capacity.free_bytes as f64,
                            "{free_gib} GiB free, fixed {fixed_gib} GiB, requested {requested}: \
                             derived ramp {:?} plans {batch}x{context} needing {:.2} GiB against \
                             {:.2} GiB free",
                            plan.batch_ramp,
                            CapacityModel::gib(required),
                            CapacityModel::gib(capacity.free_bytes as f64),
                        );
                        assert!(
                            plan.batch_ramp[stage] <= BATCH_RAMP[stage],
                            "the derived ramp may never exceed the declared ceiling: {:?}",
                            plan.batch_ramp
                        );
                        if stage > 0 {
                            assert!(
                                plan.batch_ramp[stage] >= plan.batch_ramp[stage - 1],
                                "the batch must never shrink mid-run: {:?}",
                                plan.batch_ramp
                            );
                        }
                        previous = capacity.step_bytes(batch, context);
                    }
                }
            }
        }
    }

    /// On the card of record the declared `x3` is unreachable by more than 2x, so the derived
    /// ramp at the default `--batch-size 24` must be flat — and the step count must be sized
    /// from THAT, not from the declared ramp. Sizing it from the declared ramp is what turned
    /// job 2856's `--epochs 3` into 1.33 delivered epochs.
    #[test]
    fn the_default_invocation_derives_a_flat_ramp_and_an_honest_step_count() {
        let capacity = measured_5090();
        let plan =
            resolve_ramp(Some(&capacity), 24, false).expect("24 fits at the deployed context");
        assert_eq!(plan.base_batch, 24, "24 is exactly the deployed ceiling");
        assert_eq!(plan.batch_ramp, [1, 1, 1]);
        assert!(plan.notice.is_none(), "24 fits, so nothing is clamped");

        // The production partition, and the step count priced under the derived ramp. Sizing
        // from windows rather than from an average bar-token rate is what makes the pass exact.
        let windows = [82_919usize, 82_917, 82_917];
        let contexts = stage_contexts();
        let pass_bars: u64 = (0..RAMP_STAGES)
            .map(|stage| windows[stage] as u64 * contexts[stage] as u64)
            .sum();
        let derived_steps = Schedule::steps_for_pass(&windows, plan.base_batch, &plan.batch_ramp);
        let issued: u64 = (0..RAMP_STAGES)
            .map(|stage| {
                let batch = plan.base_batch * plan.batch_ramp[stage];
                (derived_steps[stage] * batch).min(windows[stage]) as u64 * contexts[stage] as u64
            })
            .sum();
        assert_eq!(
            issued, pass_bars,
            "the derived step count must issue every assigned window, i.e. exactly one pass"
        );

        // Pricing the steps from the DECLARED x1/x2/x3 while the card runs the flat ramp leaves
        // stages 1 and 2 with a third and a half of their share unissued: the run reaches 52% of
        // the corpus and calls it an epoch. This is job 2856's defect stated in coverage rather
        // than in bar-tokens, and it is now also an error at the epoch boundary.
        let declared_steps = Schedule::steps_for_pass(&windows, plan.base_batch, &BATCH_RAMP);
        let under_declared: u64 = (0..RAMP_STAGES)
            .map(|stage| {
                let batch = plan.base_batch * plan.batch_ramp[stage];
                (declared_steps[stage] * batch).min(windows[stage]) as u64 * contexts[stage] as u64
            })
            .sum();
        let covered = under_declared as f64 / pass_bars as f64;
        assert!(
            (0.51..0.54).contains(&covered),
            "the declared ramp prices a run that covers ~52% of the corpus; got {covered}"
        );
    }

    /// A batch that cannot fit at the deployed context must be dealt with AT STARTUP, and the
    /// message must carry the arithmetic — the measured rate, the windows, the context, the
    /// resulting GiB and the free GiB. "Does not fit" without numbers is what gets argued
    /// with; the declared `x3` of 72 windows is the case that matters.
    #[test]
    fn a_batch_that_cannot_fit_is_clamped_at_startup_with_the_arithmetic() {
        let capacity = measured_5090();
        let plan =
            resolve_ramp(Some(&capacity), 72, false).expect("72 clamps rather than failing");
        assert_eq!(plan.base_batch, 24);
        let notice = plan.notice.expect("a clamp must announce itself");
        for fragment in [
            "--batch-size 72",
            "CLAMPED to 24",
            "494883 B/bar-token",
            "72 windows x 2048 bars",
            // 72 x 2048 x 494,883 B = 67.96 GiB of activations, and 78.02 GiB required once
            // the margin and the reserve are added, on a card with 27.38 GiB free.
            "67.96 GiB",
            "78.02 GiB required",
            "27.38 GiB free",
            "24 windows need 26.34 GiB",
        ] {
            assert!(
                notice.contains(fragment),
                "the clamp notice must state {fragment}; it said:\n{notice}"
            );
        }

        // Nothing fits at all: rejected, not clamped, and the message says why.
        let starved = CapacityModel {
            free_bytes: 1u64 << 30,
            ..capacity
        };
        let error = resolve_ramp(Some(&starved), 24, false)
            .expect_err("a card that cannot hold one window must fail loudly");
        let message = format!("{error}");
        assert!(
            message.contains("not one 2048-bar window fits")
                && message.contains("0.94 GiB of activations"),
            "the rejection must state the arithmetic; it said:\n{message}"
        );

        // No capacity measurement means no clamp and no fabricated ceiling.
        let unmeasured = resolve_ramp(None, 72, false).expect("an unmeasured card cannot reject");
        assert_eq!(unmeasured.base_batch, 72);
        assert_eq!(unmeasured.batch_ramp, BATCH_RAMP);
        assert!(unmeasured.notice.is_none());
    }

    /// `--exact-batch` turns the clamp into a refusal, because survival-by-degradation and
    /// controlled comparison are opposite requirements.
    ///
    /// The two arms of the expected-log-growth ablation were launched identically at
    /// `--batch-size 24` and ran at base 23 / 10818 steps and base 21 / 11847 steps, because
    /// the probe read 16.37 and 14.94 GiB of free VRAM at their two launches. Neither run was
    /// wrong and neither banner lied; the COMPARISON between them absorbed a gradient-noise
    /// and schedule-length difference into a `lambda_growth` effect. This asserts the mode
    /// that makes that pair impossible to launch, and asserts the clamping mode still clamps,
    /// because a flag that silently changed the default would be a worse bug than the one it
    /// fixes.
    #[test]
    fn exact_batch_refuses_the_clamp_that_silently_confounds_an_ablation() {
        let capacity = measured_5090();
        // 72 is the declared `x3` and clamps to 24 on this card. Same card, same request, and
        // the only difference is the flag.
        let clamped = resolve_ramp(Some(&capacity), 72, false)
            .expect("without the flag a short-fall clamps, exactly as before");
        assert_eq!(clamped.base_batch, 24);

        let error = resolve_ramp(Some(&capacity), 72, true)
            .expect_err("with the flag a short-fall must refuse rather than clamp");
        let message = format!("{error}");
        for fragment in [
            "--exact-batch was set",
            "only 24 of the 72 windows requested",
            "a shortfall of 48",
            // The refusal carries the same arithmetic the clamp notice does, so a reader does
            // not have to re-run without the flag to find out by how much and why.
            "494883 B/bar-token",
            "67.96 GiB",
            "78.02 GiB required",
            "27.38 GiB free",
            // And it names the two ways out, one of which preserves the experiment.
            "request 24 on BOTH arms",
        ] {
            assert!(
                message.contains(fragment),
                "the refusal must state {fragment}; it said:\n{message}"
            );
        }

        // A request that FITS is unaffected by the flag: the mode only ever rejects a
        // reduction, so an experiment that asks for what the card can hold still runs.
        let exact = resolve_ramp(Some(&capacity), 24, true)
            .expect("24 fits, so the exact mode has nothing to refuse");
        assert_eq!(exact.base_batch, 24);
        assert!(exact.notice.is_none());

        // An UNMEASURED card cannot refuse, because it never clamps: there is no measurement
        // to fall short of. Off CUDA this flag is inert rather than a hard failure.
        let unmeasured =
            resolve_ramp(None, 72, true).expect("an unmeasured card has no ceiling to enforce");
        assert_eq!(unmeasured.base_batch, 72);
    }

    /// The learning-rate plateau bump must be a function of the REALIZED batch multiplier at
    /// every stage, whether the ramp was derived low at startup or held low mid-run. A run
    /// whose batch was held while its schedule kept the planned bump trains every parameter
    /// group at a rate the batch does not justify — `3**0.5 = 1.73x` too high in job 2856's
    /// final stage.
    #[test]
    fn the_plateau_bump_follows_the_realized_batch_not_the_planned_one() {
        // Stage 2 uses exponent 0.5, so the bump is literally `realized_bs_ratio ** 0.5`.
        assert_eq!(BATCH_RAMP_LR_EXPONENT[RAMP_STAGES - 1], 0.5);
        let derived = measured_5090().derive_batch_ramp(24);
        assert_eq!(derived, [1, 1, 1]);

        // A schedule built from the DERIVED ramp quotes the derived bump, not the declared one.
        let honest = equal_stages(3000, 24, derived);
        let stage_2 = 2400;
        assert_eq!(honest.stage(stage_2), RAMP_STAGES - 1);
        assert_eq!(honest.batch(stage_2), 24);
        let realized = honest.lr_multiplier_for(0, honest.batch_ramp[RAMP_STAGES - 1]);
        assert!((realized - 1.0f64.powf(0.5)).abs() < 1e-12, "{realized}");
        let declared = equal_stages(3000, 24, BATCH_RAMP);
        let planned = declared.lr_multiplier_for(0, BATCH_RAMP[RAMP_STAGES - 1]);
        assert!((planned - 3.0f64.powf(0.5)).abs() < 1e-12, "{planned}");
        assert!(
            (planned / realized - 3.0f64.sqrt()).abs() < 1e-12,
            "the declared plan would have run {:.3}x the rate the realized batch justifies",
            planned / realized
        );

        // And a mid-run hold on top of a derived ramp that DID step up moves it again, at the
        // stage's own reference exponent rather than a uniform square root.
        let mut held = equal_stages(3000, 8, [1, 2, 3]);
        assert!((held.lr_multiplier(stage_2) - held.lr_multiplier_for(stage_2, 3)).abs() < 1e-12);
        held.batch_ramp[RAMP_STAGES - 1] = 2;
        assert_eq!(held.batch(stage_2), 16);
        let after = held.lr_multiplier(stage_2);
        assert!(
            (after - held.lr_multiplier_for(stage_2, 2)).abs() < 1e-12,
            "the rate must follow the realized x2, not the planned x3: {after}"
        );
        assert!(after < held.lr_multiplier_for(stage_2, 3));
    }

    /// The frontier the banner prints. At a measured cost per bar-token the card caps the
    /// PRODUCT of batch and context, so these three numbers are the tradeoff in real terms —
    /// and they must be strictly decreasing in context, or the line says nothing.
    #[test]
    fn the_context_frontier_prices_the_batch_tradeoff() {
        let capacity = measured_5090();
        let batches: Vec<usize> = CONTEXT_FRONTIER
            .iter()
            .map(|&context| capacity.frontier_batch(context))
            .collect();
        assert_eq!(CONTEXT_FRONTIER, [1024, 1472, BAR_MAX_CONTEXT]);
        assert_eq!(batches, vec![55, 35, 24]);
        // Halving the deployed context buys more than double the batch, because the transient
        // margin is charged on a smaller context step too.
        assert!(batches[0] > 2 * batches[2]);
        for pair in batches.windows(2) {
            assert!(pair[0] > pair[1], "the frontier must fall with context: {batches:?}");
        }
    }

    /// The frontier must agree with the ramp derivation at the deployed context: the frontier
    /// is the flat-batch feasibility bound, so a base batch at the frontier must produce a
    /// feasible ramp and one window more must not.
    #[test]
    fn the_frontier_is_exactly_the_feasibility_bound() {
        let capacity = measured_5090();
        let deployed = stage_context(RAMP_STAGES - 1);
        let previous = stage_context(RAMP_STAGES - 2);
        let ceiling = capacity.frontier_batch(deployed);
        let required = |batch: usize| {
            capacity.required_bytes(batch, deployed, capacity.step_bytes(batch, previous))
        };
        assert!(required(ceiling) <= capacity.free_bytes as f64);
        assert!(
            required(ceiling + 1) > capacity.free_bytes as f64,
            "the frontier must be the LARGEST feasible flat batch, not merely a feasible one"
        );
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

    /// A paired measurement with a chosen mean and standard error, for driving
    /// [`selection_outcome`] over a synthetic sequence. Only `mean` and `se` are read.
    fn paired(mean: f64, se: f64) -> Dispersion {
        Dispersion {
            mean,
            se,
            ci_low: mean - 1.96 * se,
            ci_high: mean + 1.96 * se,
            blocks: 5,
            samples: trade_bench::TRADE_WINDOWS,
        }
    }

    /// One synthetic read: the criterion, the guarded density, and the noise each was
    /// measured at. Scales are the ones `bardist_v2` actually produced — 0.02 bps of paired
    /// edge SE, 0.05 nats of paired conditional-NLL SE, 0.0003 nats on `r`.
    #[derive(Clone, Copy)]
    struct Read {
        step: usize,
        edge_bps: f64,
        nll: f64,
    }

    const EDGE_SE: f64 = 0.0204;
    const NLL_SE: f64 = 0.0526;
    const DOF_SE: f64 = 0.0003;

    /// Replay [`selection_outcome`] over a sequence of reads exactly as `validate` does,
    /// pairing each candidate against the standing incumbent. Returns the promoted step and
    /// every decision, so a test can assert on the refusals and not only the winner.
    fn replay(reads: &[Read]) -> (usize, Vec<(usize, SelectionOutcome)>) {
        let mut incumbent: Option<Read> = None;
        let mut trail = Vec::new();
        let edge = vec![0.0; trade_bench::TRADE_WINDOWS];
        for read in reads {
            let outcome = match incumbent {
                None => selection_outcome(true, &edge, None, None, None),
                Some(inc) => selection_outcome(
                    false,
                    &edge,
                    Some(paired(read.edge_bps - inc.edge_bps, EDGE_SE)),
                    // POSITIVE means the candidate is worse, which is the guard's convention.
                    Some(paired(read.nll - inc.nll, NLL_SE)),
                    Some(paired((read.nll - inc.nll) * 0.05, DOF_SE)),
                ),
            };
            if outcome == SelectionOutcome::Promoted {
                incumbent = Some(*read);
            }
            trail.push((read.step, outcome));
        }
        (
            incumbent.expect("the first eligible read always promotes").step,
            trail,
        )
    }

    /// SELECT-ECON-001. When the economically best checkpoint and the NLL-best checkpoint are
    /// DIFFERENT models, the rule promotes the economic one and the decision says what the
    /// density cost.
    ///
    /// This is the whole point of the inversion and it is not hypothetical: on `bardist_v2`
    /// the two rules disagree over the entire run. The NLL-primary rule promoted the LAST
    /// eligible read, whose conditional NLL was the best of the run (-9.3817 nats at the fixed
    /// 896-bar ruler) and whose 0.25x-cap edge was 0.34 bps/bar, near the worst; the economic
    /// peak sat at 0.38-0.40 bps/bar around steps 7000-10364, where the NLL was 0.176 nats
    /// WORSE. The arithmetic says that is the right trade: total achievable Kelly growth is
    /// s^2/2 = 5.25e-4 nats/bar, so 0.176 nats of density is ~340x the entire tradeable
    /// content of the prediction and cannot be paid for in anything we trade.
    ///
    /// The sequence below is that shape, compressed: edge falls monotonically while NLL
    /// improves monotonically. A rule that selects on NLL lands on the last read. This one
    /// must land on the first and must refuse every later read explicitly.
    #[test]
    fn selection_prefers_the_economically_best_read_over_the_nll_best_one() {
        // Each later read is worth 0.15 bps LESS - far outside the 2-SE band of 0.0408 - while
        // its density improves by 0.5 nats, far more than the guard's 0.105-nat tolerance ever
        // has to allow. Nothing here is marginal; the two criteria simply disagree.
        let reads: Vec<Read> = (0..5)
            .map(|i| Read {
                step: 1000 * (i + 1),
                edge_bps: 0.40 - 0.15 * i as f64,
                nll: -9.0 - 0.5 * i as f64,
            })
            .collect();
        let (promoted, trail) = replay(&reads);
        assert_eq!(
            promoted, 1000,
            "the economically best read is the first one; a rule that promoted a later read \
             selected on density, which is the behaviour being removed"
        );
        let nll_best = reads
            .iter()
            .min_by(|a, b| a.nll.total_cmp(&b.nll))
            .expect("a non-empty sequence has an NLL minimum");
        assert_ne!(
            nll_best.step, promoted,
            "this fixture is only a test of the inversion if the two rules disagree"
        );
        assert!(
            trail[1..]
                .iter()
                .all(|(_, outcome)| *outcome == SelectionOutcome::RefusedInsideNoise),
            "every later read is economically worse and must be refused on the CRITERION, not \
             by a guard: {trail:?}"
        );
        // The cost is a real number the promotion has to be able to state, in both directions.
        // The ledger records the paired difference against the incumbent, positive when the
        // candidate is worse, so the promoted artifact's density gap against the NLL-best read
        // is recoverable from the ledger rather than only from the log.
        let cost = reads[0].nll - nll_best.nll;
        assert!(
            cost > 0.0,
            "the economic pick is supposed to be paying for its edge with density here"
        );
        assert!(
            cost > 5.25e-4,
            "a density cost below the {:.3e} nats/bar of total achievable Kelly growth would \
             make the trade-off untestable rather than favourable",
            5.25e-4
        );
    }

    /// SELECT-NOISE-001. A candidate whose economic gain is inside the paired noise band is
    /// REFUSED, and the refusal is attributed to the band rather than to a guard.
    ///
    /// A single read's edge interval is ~+/-1.6 bps, so a naive argmax over the ~31 reads of a
    /// run promotes noise essentially every time. That interval is the LEVEL's and is
    /// common-mode across checkpoints scored on identical windows; the relevant scale is the
    /// PAIRED one, ~0.02 bps within a (stage, epoch) cell. This test pins the band at that
    /// scale from both sides: just inside is refused, just outside promotes. Without the second
    /// half the rule could pass by refusing everything.
    #[test]
    fn a_candidate_inside_the_paired_noise_band_is_refused() {
        let band = SELECTION_EDGE_SE_MULTIPLE * EDGE_SE;
        let edge = vec![0.0; trade_bench::TRADE_WINDOWS];
        // Identical density on both sides, so only the criterion can decide.
        let inside = selection_outcome(
            false,
            &edge,
            Some(paired(band * 0.99, EDGE_SE)),
            Some(paired(0.0, NLL_SE)),
            Some(paired(0.0, DOF_SE)),
        );
        assert_eq!(
            inside,
            SelectionOutcome::RefusedInsideNoise,
            "a gain of {:.4} bps against a {:.4} bps band is unresolved, not better",
            band * 0.99,
            band
        );
        let outside = selection_outcome(
            false,
            &edge,
            Some(paired(band * 1.01, EDGE_SE)),
            Some(paired(0.0, NLL_SE)),
            Some(paired(0.0, DOF_SE)),
        );
        assert_eq!(
            outside,
            SelectionOutcome::Promoted,
            "a gain that clears the band with both guards flat must promote, or the rule is \
             not a selection rule"
        );
        // Exactly at the band is refused: the comparison is strict, so a candidate that only
        // equals its own noise scale does not displace an incumbent.
        assert_eq!(
            selection_outcome(
                false,
                &edge,
                Some(paired(band, EDGE_SE)),
                Some(paired(0.0, NLL_SE)),
                Some(paired(0.0, DOF_SE)),
            ),
            SelectionOutcome::RefusedInsideNoise,
            "the band is a strict threshold"
        );
        // An unresolvable measurement must not promote either. NaN fails `>` in both
        // directions, and the branch order has to make that a refusal rather than a promotion.
        assert_eq!(
            selection_outcome(
                false,
                &edge,
                Some(paired(f64::NAN, EDGE_SE)),
                Some(paired(0.0, NLL_SE)),
                Some(paired(0.0, DOF_SE)),
            ),
            SelectionOutcome::RefusedInsideNoise,
            "an unmeasurable gain must never displace an incumbent"
        );
        // No comparable bench vector at all is a THIRD state, distinct from a refusal on the
        // merits: the incumbent stands but nothing was measured against it.
        assert_eq!(
            selection_outcome(false, &[], Some(paired(1.0, EDGE_SE)), None, None),
            SelectionOutcome::Unmeasurable,
            "an unmeasured criterion is not evidence against the candidate"
        );
        assert_eq!(
            selection_outcome(false, &edge, None, None, None),
            SelectionOutcome::Unmeasurable,
            "no paired vector means no comparison, whatever the level says"
        );
    }

    /// SELECT-GUARD-001. A density regression beyond tolerance BLOCKS promotion even when the
    /// economic criterion improves by more than its band.
    ///
    /// The mirror of the rule that shipped until now, and the half that keeps the inversion
    /// honest: selection is economic, but a model whose predictive law has genuinely broken is
    /// not promotable on a trading read that could still be luck. Both guards are tested
    /// because they protect different things - the aggregate density and the ONE factor the
    /// trade is actually taken on - and because a guard that never fires is indistinguishable
    /// from one that is not wired up.
    #[test]
    fn a_density_regression_blocks_promotion_even_when_the_edge_improves() {
        let edge = vec![0.0; trade_bench::TRADE_WINDOWS];
        let band = SELECTION_EDGE_SE_MULTIPLE * EDGE_SE;
        // A large, unambiguous economic gain: 100x the band. Nothing here is marginal on the
        // criterion, so only a guard can produce a refusal.
        let big_gain = || Some(paired(100.0 * band, EDGE_SE));
        let nll_tolerance = SELECTION_NLL_TOLERANCE_SE_MULTIPLE * NLL_SE;
        assert_eq!(
            selection_outcome(
                false,
                &edge,
                big_gain(),
                Some(paired(nll_tolerance * 1.01, NLL_SE)),
                Some(paired(0.0, DOF_SE)),
            ),
            SelectionOutcome::RefusedNllGuard,
            "a conditional-NLL regression past {nll_tolerance:.6} nats must veto, however good \
             the edge looks"
        );
        // Just inside the tolerance the same candidate promotes, which is what makes the guard
        // a tolerance rather than a prohibition on any density movement at all. Selection is
        // economic, and a density that merely MOVED is not one that broke.
        assert_eq!(
            selection_outcome(
                false,
                &edge,
                big_gain(),
                Some(paired(nll_tolerance * 0.99, NLL_SE)),
                Some(paired(0.0, DOF_SE)),
            ),
            SelectionOutcome::Promoted,
            "a regression inside tolerance must not block an edge that cleared its band"
        );
        // The per-factor guard on `r`, the factor the trade is taken on. It is resolved ~100x
        // better than the edge is, so it is held to a TIGHTER multiple, and it has to fire on
        // a regression the aggregate guard would have waved through.
        let dof_tolerance = SELECTION_GUARD_SE_MULTIPLE * DOF_SE;
        assert!(
            dof_tolerance < nll_tolerance,
            "the guarded factor is resolved far better than the aggregate, so its tolerance \
             must be the tighter of the two"
        );
        assert_eq!(
            selection_outcome(
                false,
                &edge,
                big_gain(),
                // Aggregate density IMPROVES, so only the per-factor guard can refuse.
                Some(paired(-nll_tolerance, NLL_SE)),
                Some(paired(dof_tolerance * 1.01, DOF_SE)),
            ),
            SelectionOutcome::RefusedDofGuard,
            "a resolvable regression in {} must veto even when the aggregate density improved",
            BAR_DOF_NAMES[SELECTION_GUARD_DOF]
        );
        // Neither guard can CAUSE a promotion. A density improvement with no economic gain is
        // refused on the criterion, which is the property that makes selection economic rather
        // than a two-criterion compromise.
        assert_eq!(
            selection_outcome(
                false,
                &edge,
                Some(paired(0.0, EDGE_SE)),
                Some(paired(-10.0 * nll_tolerance, NLL_SE)),
                Some(paired(-10.0 * dof_tolerance, DOF_SE)),
            ),
            SelectionOutcome::RefusedInsideNoise,
            "an enormous density improvement with no edge gain must NOT promote; the guards \
             are vetoes, not criteria"
        );
        // And a guard with nothing measured cannot veto on evidence it does not have.
        assert_eq!(
            selection_outcome(false, &edge, big_gain(), None, None),
            SelectionOutcome::Promoted,
            "an unmeasured guard must not block a resolved economic gain"
        );
    }
}
