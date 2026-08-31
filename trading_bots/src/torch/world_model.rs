//! Discrete distributional bar world model.
//!
//! [`BarTrunk`] is a causal transformer over bars. A forecast-safe token sums
//! five bin embeddings, a raw-DOF projection and exogenous clock embeddings;
//! same-instant observed market ids are masked before every token path. The
//! resulting belief and a separate shared target-clock/current-market embedding
//! condition [`BarEmissionHead`]'s five categoricals.
//!
//! [`BarDynamics`] is the NextLat one-step latent predictor: given the belief
//! after bar `t` and the exact shared trunk token of bar `t+1`, it predicts the
//! next belief without running the transformer. It is trained to approximate
//! exactly the state the cached trunk computes, which makes it a strictly cheaper
//! but drifting substitute at rollout time — see [`RolloutMode`].
//!
//! [`BarWorldModel`] is the frozen inference bundle: trunk + emission head +
//! dynamics + supports + [`BarWorldModelMetadata`], loaded with
//! `require_complete()` and `VarStore::freeze()` so a planner can never silently
//! run against a partially-matched or still-trainable checkpoint.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
};

use anyhow::{bail, ensure, Context, Result};
use ring::digest::{digest, SHA256};
use serde::{Deserialize, Serialize};
use tch::{autocast, nn, nn::Init, Device, Kind, Tensor};

use crate::torch::{
    backward_probe,
    bar_dist::{
        BarEmissionHead, BarSupports, DofScaling, BAR_CHAIN, BAR_DOF, BAR_DOF_NAMES,
        BAR_LABEL_SIGMA_RATIO, BAR_PREFIX_EMBED_DIM, BAR_VOLUME_EMA_SPAN, NUM_BAR_BINS,
    },
    dataset::{
        resolution_class, time_ids_without_market, BAR_TIME_CARDINALITY, BAR_TIME_CONDITIONING,
        BAR_TIME_FEATURES, BAR_TIME_MARKET, TIME_RESOLUTION,
    },
    fa4::{pope_flash_attention_decode_q1, pope_flash_attention_prefill},
    hashing::file_sha256,
    load::load_var_store_partial,
    pope::{
        init_pope_theta_bias, pope_expand_qk, PolarQk, PopePhases, PopeThetaInit,
        POPE_ATTENTION_SCALE, POPE_DIM, POPE_FREQUENCY_BASE, POPE_QK_DIM,
    },
    train::pretrain_profile as profile,
};

/// The exact unequal-width PoPE reference. Only the CPU test path uses it;
/// production prefill and decode go through the strict FA4 bridge.
#[cfg(test)]
use crate::torch::pope::pope_attention_reference;

// ---------------------------------------------------------------------------
// Architecture constants
// ---------------------------------------------------------------------------

pub const BAR_MODEL_DIM: i64 = 512;
pub const BAR_LAYERS: usize = 10;
pub const BAR_HEADS: i64 = 8;
pub const BAR_HEAD_DIM: i64 = 64;
pub const BAR_FF_DIM: i64 = 2048;
pub const BAR_MAX_CONTEXT: i64 = 2048;
pub const BAR_DIRECT_STAGE_COUNT: usize = 3;
pub const BAR_DIRECT_T2_STAGE_MULTIPLIER: [f64; BAR_DIRECT_STAGE_COUNT] =
    [0.5; BAR_DIRECT_STAGE_COUNT];
pub const BAR_DIRECT_T3_STAGE_MULTIPLIER: [f64; BAR_DIRECT_STAGE_COUNT] =
    [0.25; BAR_DIRECT_STAGE_COUNT];

fn direct_schedule_canonical() -> String {
    let encode = |values: &[f64]| {
        values
            .iter()
            .map(|value| format!("{:016x}", value.to_bits()))
            .collect::<Vec<_>>()
            .join(",")
    };
    format!(
        "h2:{}|h3:{}",
        encode(&BAR_DIRECT_T2_STAGE_MULTIPLIER),
        encode(&BAR_DIRECT_T3_STAGE_MULTIPLIER)
    )
}
/// Lineage: `v2` -> `v3` replaces the rank-1 affine chain-prefix conditioning in
/// [`BarEmissionHead`] with per-slot bin embeddings (`binprefix`) and RMS-normalizes
/// the [`BarDynamics`] output onto the trunk belief shell (`dynrms`). `v3` -> `v4`
/// widens the conditioning bank from four calendar channels to nine. `v4` -> `v5`
/// adds the per-layer embedding shortcut (`x0`). `v5` -> `v6` makes every trunk
/// token forecast-safe by masking its same-instant market ids, gives the emission
/// readout an explicit target-clock/current-market conditioning vector, and makes
/// dynamics consume the trunk's exact shared token embedding rather than private
/// DOF and clock encoders. `v6` -> `v7` adds direct-horizon complete-bar supervision.
/// `v7` -> `v8` gives h1, h2 and h3 private emission parameter banks while retaining the
/// shared live trunk. Each transition changes VarStore shapes or semantics, so checkpoints
/// cannot be cross-loaded.
pub const BAR_ARCHITECTURE: &str =
    "bardist-causal-ar-pope64-fa4-time9market-binprefix-dynrms-x0shortcut-forecastcond-sharedtoken-directh23-privateheads-v8";

/// KV-cache layout contract. Any change to the cache geometry, the position
/// bookkeeping or the eviction rule must bump this, because it feeds the lineage
/// hash and therefore invalidates every checkpoint that claims the old one.
pub const BAR_CACHE_CONTRACT: &str = "bar-circular-pope-absolute-bshd-k128-v64-fa4-v1";
pub const BAR_DIRECT_FLOW_CONTRACT: &str =
    "h1-head-private|h2-head-private|h3-head-private|trunk-shared";

/// RMSNorm epsilon. The norm carries no learnable gain anywhere in this model.
pub const BAR_NORM_EPS: f64 = 1e-6;

/// Per-sublayer residual gain at init, `sqrt(1.1)`.
pub const BAR_RESID_LAMBDA_INIT: f64 = 1.048_808_848_170_151_6;
/// Per-sublayer sublayer-output gain at init.
pub const BAR_POST_LAMBDA_INIT: f64 = 1.0;
/// [`BarDynamics`] hidden width, `round_to_128(1.6 * 1024)`.
pub const BAR_DYNAMICS_HIDDEN: i64 = 1664;

/// Metadata schema version. v5 and below are LeJEPA-era and unreadable here. v7 adds
/// [`BarTrainingProvenance`], v8 adds the selection contexts, v9 authenticates direct
/// objective weights, v10 authenticates their stage schedule, v11 authenticates the
/// private-head/shared-trunk gradient-flow contract, and v12 authenticates economic sizing.
pub const BAR_METADATA_VERSION: u32 = 12;

/// ET minutes at which the session channel changes value, re-exported from the
/// producer. Folded into the lineage because the cardinality alone does not pin
/// the semantics — a redefined boundary would silently re-mean every embedding
/// row while leaving the hash equal. Taking it from `dataset` rather than
/// restating it means the lineage cannot drift from the definition it describes.
pub use crate::torch::dataset::SESSION_BOUNDARY_MINUTES as BAR_SESSION_BOUNDARY_MINUTES;

/// Rows of the fused conditioning embedding bank, one block per id channel.
const BAR_TIME_EMBED_ROWS: i64 = {
    let mut rows = 0;
    let mut slot = 0;
    while slot < BAR_TIME_FEATURES {
        rows += BAR_TIME_CARDINALITY[slot];
        slot += 1;
    }
    rows
};

const BAR_QKV_DIM: i64 = 3 * BAR_MODEL_DIM;

const _: () = assert!(BAR_HEADS * BAR_HEAD_DIM == BAR_MODEL_DIM);
const _: () = assert!(BAR_HEAD_DIM == POPE_DIM);
const _: () = assert!(BAR_FF_DIM == 4 * BAR_MODEL_DIM);

/// 2D projection banks, routed to NorMuon. Each name is a unique suffix of the
/// per-layer variable names, so a substring match cannot collect anything else.
const BAR_TRUNK_MUON_SUBSTRINGS: [&str; 4] = ["qkv_w", "attn_out_w", "ff_in_w", "ff_out_w"];
const BAR_DYNAMICS_MUON_SUBSTRINGS: [&str; 3] = ["bar_dyn_fc1_w", "bar_dyn_fc2_w", "bar_dyn_fc3_w"];
/// Down-projections, which take the extra `4.0x` NorMuon learning-rate multiplier.
const BAR_MUON_DOWN_PROJECTION_SUBSTRINGS: [&str; 2] = ["ff_out_w", "bar_dyn_fc3_w"];
/// Lookup tables and the raw-DOF input map, routed to AdamW with the embedding betas.
const BAR_ADAMW_EMBEDDING_SUBSTRINGS: [&str; 3] =
    ["bar_bin_embed", "bar_dof_embed", "bar_time_embed"];
/// Learned scalars and the PoPE phase bias, routed to AdamW with `wd_mul = 0`.
const BAR_ADAMW_SCALAR_SUBSTRINGS: [&str; 2] = ["_lambda", "pope_theta_bias"];

/// Every 2D projection bank in the model, the canonical NorMuon partition.
const BAR_MUON_SUBSTRINGS: [&str; 7] = [
    "qkv_w",
    "attn_out_w",
    "ff_in_w",
    "ff_out_w",
    "bar_dyn_fc1_w",
    "bar_dyn_fc2_w",
    "bar_dyn_fc3_w",
];

/// Canonical union of [`BarTrunk::muon_name_substrings`] and
/// [`BarDynamics::muon_name_substrings`].
pub fn bar_muon_name_substrings() -> &'static [&'static str] {
    &BAR_MUON_SUBSTRINGS
}

pub fn bar_muon_down_projection_substrings() -> &'static [&'static str] {
    &BAR_MUON_DOWN_PROJECTION_SUBSTRINGS
}

pub fn bar_adamw_embedding_substrings() -> &'static [&'static str] {
    &BAR_ADAMW_EMBEDDING_SUBSTRINGS
}

pub fn bar_adamw_scalar_substrings() -> &'static [&'static str] {
    &BAR_ADAMW_SCALAR_SUBSTRINGS
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Sidecar describing the architecture a checkpoint was trained with, plus the
/// hashes that pin the weights and every support it must be paired with.
///
/// `supports_sha256` is keyed by bar resolution because a 128-bin equal-mass
/// support is a discretization of one resolution's distribution: a daily bar's
/// return and range live on a different scale from a five-minute bar's, so a
/// merged run fits and persists one support per resolution. `res_secs` is the
/// DEPLOYMENT resolution — the one held-out selection runs against — and must be
/// one of the keys.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BarWorldModelMetadata {
    pub format_version: u32,
    pub architecture: String,
    pub model_dim: i64,
    pub layers: i64,
    pub heads: i64,
    pub head_dim: i64,
    pub ff_dim: i64,
    pub max_context_bars: i64,
    pub num_bins: i64,
    pub res_secs: u32,
    pub supports_sha256: BTreeMap<u32, String>,
    /// SHA-256 of the weight FILE, which pins this sidecar to the exact bytes it was written
    /// beside. It is not a fingerprint of the weights: a `.ot` is a torch zip whose internal
    /// record names derive from the file stem, so re-saving identical tensors under a different
    /// name yields different bytes and a different hash. Two checkpoints differing here may
    /// hold bit-identical parameters, and equality here is the only direction that means
    /// anything - it says the sidecar and the file belong together. To compare WEIGHTS, load
    /// both and compare tensors, or compare the zip records' CRC-32s.
    pub checkpoint_sha256: String,
    pub lineage_sha256: String,
    /// Absent on a checkpoint written before v7, and on anything but a pretraining run.
    #[serde(default)]
    pub training: Option<BarTrainingProvenance>,
    /// SHA-256 of the optimizer-state sidecar committed with a recovery checkpoint.
    /// Absent for deployment artifacts and checkpoints written before recovery was supported.
    #[serde(default)]
    pub optimizer_checkpoint_sha256: Option<String>,
    /// Lazy production AdamW buffers expected in the strict optimizer schema.
    #[serde(default)]
    pub optimizer_initialized_adamw: Vec<String>,
    /// Exact schedule and objective contract required to continue this training run.
    #[serde(default)]
    pub pretrain_recovery: Option<BarPretrainRecovery>,
}

/// Which DATA and which SELECTION RULE produced a checkpoint.
///
/// The bar corpus is live: `Ingest` appends continuously and the split instants are
/// percentiles of the current trading-time axis, so they drift roughly 0.8 days per
/// ingestion day. Two ablations run a week apart are therefore scored on different held-out
/// windows, and nothing in the weights, the supports or the architecture reveals it. This is
/// the record that does, and it is folded into the lineage hash so it cannot be edited away
/// from the artifact it describes.
///
/// `selection_metric` and `selection_weights` exist because `nll_bar` is an UNWEIGHTED sum
/// over five factors with wildly unequal headroom (`r` 0.110, `s` 0.192, `u` 1.092, `v`
/// 1.174, `w` 0.000 nats below uniform on the live supports). Whatever we promote on, the
/// artifact should say so.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BarTrainingProvenance {
    /// `BarCorpus::identity_fingerprint()` of the corpus this run trained and scored on.
    pub corpus_fingerprint: String,
    /// `(train|val, val|test)` instants in epoch millis, whether derived or pinned.
    pub split_bounds: (i64, i64),
    /// True when the instants came from `--split-bounds` rather than from the live
    /// percentiles. A pinned run is comparable to another pinned run at the same instants.
    pub split_bounds_pinned: bool,
    /// Seed of the pinned evaluation windows. Campaign-constant by design; deliberately NOT
    /// the training seed, or a seed replicate would resample the whole bench.
    pub eval_window_seed: u64,
    /// Seed of the training sampler, the weight init and the CUDA RNG.
    pub train_seed: u64,
    /// Human-readable statement of what promotion actually compared.
    pub selection_metric: String,
    /// Weight applied to each per-DOF term of the selection objective, in `[r, s, u, v, w]`
    /// order. All ones: the objective is the unweighted sum of the five CONDITIONAL terms.
    /// A weight vector is a free parameter nobody can defend, so the asymmetry the campaign
    /// actually needs is expressed as the guard below, not as weights.
    pub selection_weights: [f64; BAR_DOF],
    /// DOF the non-regression guard protects, by name. Promotion is refused when this
    /// factor regresses against the incumbent even if the aggregate improves, because it is
    /// the one that determines trading P&L and it has an order of magnitude less headroom
    /// than the intra-bar shape factors it would be traded against.
    pub selection_guard_dof: String,
    /// How many standard errors of the PAIRED difference the guarded factor is allowed to
    /// drift before promotion is refused. `1.0` means "any regression the bench can
    /// actually resolve blocks the promotion".
    pub selection_guard_se_multiple: f64,
    /// Liquidity floor the symbol universe was gated on, in dollars of median daily volume.
    /// `0.0` means every file on disk was used. Recorded because a universe ablation that
    /// is not recorded is not an ablation.
    pub min_dollar_volume: f64,
    /// Symbols the corpus actually held after every filter.
    pub symbols: usize,
    /// True when the supports were reused under `--freeze-supports` despite provenance that
    /// does not match this corpus. Comparability bought deliberately, and recorded.
    pub supports_frozen: bool,
    /// Corpus fingerprint recorded inside the supports artifact, when it has one.
    pub supports_corpus_fingerprint: Option<String>,
    /// SHA-256 of `long_data/universe.json`, i.e. WHICH symbols the liquidity floor
    /// admitted. The floor and the boundary are only half the record: the ranking is
    /// re-measured periodically and admits a different set at the same floor.
    pub universe_fingerprint: Option<String>,
    /// Instant the universe ranking was measured against, as recorded in `universe.json`.
    /// A corpus SELECTED under a different notion of "train" than it is SCORED under is
    /// the leak this whole record exists to make visible.
    pub universe_train_end_ms: Option<i64>,
    /// Scoring rule the objective and every reported `nll_bar` used, by name.
    ///
    /// The three [`crate::torch::bar_dist::BarScoring`] modes differ by additive constants
    /// that depend on the binning, so two runs scored under different modes are tens of nats
    /// apart on the identical model. It is folded into the lineage hash for exactly that
    /// reason: a checkpoint cannot be relabelled with a mode it was not trained under, and
    /// `pretrain-compare` refuses to pair two runs that disagree.
    pub scoring: String,
    /// Reduction of the predictive return law that sized every economic selection metric.
    ///
    /// Required on newly written artifacts and folded into lineage. Empty only on metadata
    /// predating this field.
    #[serde(default)]
    pub sizing_rule: String,
    /// Base weights of the direct complete-bar t+2 and t+3 Hard-NLL objectives. Their
    /// persistent stage multipliers are training contract `[.5,.5,.5]` and `[.25,.25,.25]`.
    #[serde(default)]
    pub direct_t2_weight: f64,
    #[serde(default)]
    pub direct_t3_weight: f64,
    /// Context length, in bars, of the held-out set the promotion decision was actually
    /// taken on.
    ///
    /// Normally equal to `deployed_context`. It is SHORTER when the run ended without ever
    /// training at the deployed context — a memory-held ramp, or a run cut short — in which
    /// case selection fell back to the fixed diagnostic context so the run would still leave
    /// an evaluable artifact behind. A planner loading such a checkpoint is running a model
    /// outside the positional range it was selected at, and that is a fact about the file,
    /// not about the log.
    #[serde(default)]
    pub selection_context: i64,
    /// Context the model is meant to be deployed at, i.e. the final ramp stage's.
    #[serde(default)]
    pub deployed_context: i64,
    /// Longest context the run actually took an optimizer step at. Below `deployed_context`
    /// only when the ramp never got there.
    #[serde(default)]
    pub reached_context: i64,
    /// Global optimizer step these exact weights are from. Present on every artifact a
    /// pretraining run writes - promoted, epoch-boundary, step-tagged and `pretrain_last.ot`
    /// alike - because the one question a reader of a checkpoint directory has is which step a
    /// file holds, and before this was recorded the only answer was the run's log.
    /// `selection_context` is what says whether a decision chose the file; this says when it
    /// was taken. `None` only on a checkpoint written before this field existed.
    #[serde(default)]
    pub global_step: Option<usize>,
    /// Context, in bars, the selection criteria were MEASURED at, as distinct from
    /// `selection_context`, which is the range the artifact is certified deployable over.
    ///
    /// They differ by design: both criteria are read on one fixed diagnostic ruler that never
    /// moves for the life of a run, so consecutive decisions compare models rather than
    /// rulers, while eligibility to be promoted at all remains gated on the deployed context.
    #[serde(default)]
    pub selection_bench_context: Option<i64>,
    /// The ECONOMIC criterion at the moment of promotion: mean per-window net Kelly edge over
    /// the unconditional-marginal null at the selection cap, in basis points per bar, with the
    /// block-bootstrap standard error of that level.
    ///
    /// Recorded beside `selection_nll_conditional` so BOTH numbers the decision weighed are on
    /// the file. A promotion that bought edge while paying density, and one that improved
    /// both, are then distinguishable from the artifact alone.
    #[serde(default)]
    pub selection_edge_bps: Option<f64>,
    #[serde(default)]
    pub selection_edge_se_bps: Option<f64>,
    /// Conditional held-out NLL of the same pass, in nats per bar: the GUARDED quantity, not
    /// the selected one.
    #[serde(default)]
    pub selection_nll_conditional: Option<f64>,
    /// Batch size, in WINDOWS, the run actually used at each ramp stage.
    ///
    /// Absolute and not a multiplier of `--batch-size`, because `--batch-size` is itself
    /// clamped at startup to what a measured device-capacity probe says the deployed context
    /// can hold — so a multiplier alone does not identify the batch the run trained at, and
    /// the batch is what the learning-rate plateau bump is derived from.
    ///
    /// The ramp is planned from that capacity probe, and the pretrainer additionally lowers a
    /// stage in place when another tenant of a shared card takes memory after the probe. Two
    /// runs whose only difference is such a hold are not the same experiment: with both
    /// step-ups held a run delivers as little as 44% of the bar-token budget its step count
    /// was sized for, and nothing else in the artifact says it happened.
    #[serde(default)]
    pub batch_ramp: Vec<usize>,
    /// Fraction of the run held at the flat learning-rate plateau before the linear decay to
    /// the terminal multiplier, i.e. the run's `--lr-plateau-fraction`.
    ///
    /// Two runs of the same length at different fractions are NOT the same experiment. Past the
    /// plateau the learning-rate multiplier is affine in the step index, so a one-epoch run at
    /// 0.40 finishes fully annealed while at 0.90 finishes at the peak rate, and their
    /// re-decoded calibration slopes differ resolvably. Nothing else in the artifact says which
    /// of the two a checkpoint came from.
    ///
    /// `0.0` means a checkpoint written before this was recorded, never a run that used it: the
    /// flag refuses both degenerate boundaries. Folded into the lineage hash only when
    /// recorded, so an unrecorded sidecar keeps validating against its own stored hash.
    #[serde(default)]
    pub lr_plateau_fraction: f64,
    /// Optimizer recipe that produced this checkpoint. `None` only for sidecars
    /// written before optimizer provenance was recorded.
    #[serde(default)]
    pub optimizer: Option<BarOptimizerProvenance>,
    /// Target parametrization the head was trained against.
    ///
    /// Load-bearing for the lineage exactly as `scoring` is, and for a stronger reason: two
    /// runs scored under different [`crate::torch::bar_dist::BarScoring`] modes differ by an
    /// additive constant, whereas two runs under different [`DofScaling`] modes are modelling
    /// DIFFERENT RANDOM VARIABLES. `dof_scaling` on the supports artifact says what the bins
    /// tile; this says what the weights were fitted to predict, and the two must agree.
    ///
    /// [`DofScaling::Raw`] on every checkpoint written before this existed, which is what those
    /// runs did, so the default is a fact rather than a placeholder.
    #[serde(default)]
    pub dof_scaling: DofScaling,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BarOptimizerProvenance {
    pub name: String,
    /// `None` records that the row controller was disabled for this newly written
    /// checkpoint; the enclosing `optimizer: None` is the legacy/unknown state.
    #[serde(default)]
    pub row_learned_lr: Option<BarRowLearnedLrProvenance>,
    /// Strictly opt-in pretraining optimizer experiment. `None` is the production recipe
    /// and deliberately contributes no canonical text, preserving existing lineage hashes.
    #[serde(default)]
    pub ablation: Option<BarOptimizerAblationProvenance>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BarOptimizerAblationProvenance {
    /// Canonical CLI variant: `fixed-sgd` or `smd-idbd`.
    pub variant: String,
    /// Absolute initial learning rate before the shared schedule multiplier.
    pub initial_lr: f64,
    /// SMD meta learning rate. Absent for the matched fixed-SGD control.
    #[serde(default)]
    pub meta_lr: Option<f64>,
    /// True only for the SMD arm, whose HVP is computed by autograd from a gradient graph.
    pub exact_hvp: bool,
    /// Per-coordinate log-gain multiplier bounds. Absent for fixed SGD.
    #[serde(default)]
    pub beta_bounds: Option<(f64, f64)>,
    /// Per-step per-coordinate beta update clamp. Absent for fixed SGD.
    #[serde(default)]
    pub beta_update_clip: Option<f64>,
}

/// Exact-continuation sidecar contract. Version 5 additionally authenticates the sizing rule
/// that drives checkpoint promotion, so a resumed run cannot change its economic objective.
pub const BAR_PRETRAIN_RECOVERY_VERSION: u32 = 5;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BarPretrainRecovery {
    pub format_version: u32,
    pub total_steps: usize,
    pub stage_steps: Vec<usize>,
    /// Startup-planned batch multipliers used to partition the pass and size each stage.
    /// Empty only on recovery sidecars written before runtime-held ramps were resumable.
    #[serde(default)]
    pub derived_batch_ramp: Vec<usize>,
    pub base_batch: usize,
    pub requested_batch: usize,
    pub exact_batch: bool,
    pub quadratic_kelly: bool,
    pub mean_shrink: bool,
    pub epochs: usize,
    #[serde(default)]
    pub steps_override: Option<usize>,
    pub dyn_horizon: usize,
    pub lambda_dyn: f64,
    pub lambda_kl: f64,
    pub direct_t2_weight: f64,
    pub direct_t3_weight: f64,
    #[serde(default)]
    pub auxiliary_resolutions: Vec<u32>,
    /// Whether the optimizer sidecar carries AdamW gradients retained between cadence updates.
    /// Required rather than defaulted so artifacts written before exact accumulation recovery
    /// are rejected instead of being mistaken for post-update checkpoints.
    pub adamw_accumulation_pending: bool,
    /// Last measured activation footprint in device bytes per bar-token. Zero means unmeasured.
    pub activation_bytes_per_token: f64,
    /// Stable device identity (NVML UUID and name) under which the footprint was measured.
    /// Empty only when the footprint is unmeasured.
    pub activation_device: String,
    pub checkpoint_every: usize,
    pub validate_every: usize,
    pub ablation_lr: f64,
    pub smd_meta_lr: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BarRowLearnedLrProvenance {
    pub c: f64,
    pub controller_lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub warmup_primary_steps: i64,
    pub evidence_clip: f64,
    pub evidence: String,
}

impl BarWorldModelMetadata {
    /// `resolutions` are every resolution whose support was fitted and written
    /// beside the checkpoint; `res_secs` is the deployment resolution and must be
    /// among them. Carries no [`BarTrainingProvenance`]; use
    /// [`Self::for_checkpoint_with`] from a pretraining run, which has one.
    pub fn for_checkpoint(
        checkpoint: impl AsRef<Path>,
        resolutions: &[u32],
        res_secs: u32,
    ) -> Result<Self> {
        Self::for_checkpoint_with(checkpoint, resolutions, res_secs, None)
    }

    /// As [`Self::for_checkpoint`], recording which corpus and which selection rule
    /// produced the artifact. `training` is folded into the lineage hash, so a checkpoint
    /// cannot be re-labelled with a different data provenance and still validate.
    pub fn for_checkpoint_with(
        checkpoint: impl AsRef<Path>,
        resolutions: &[u32],
        res_secs: u32,
        training: Option<BarTrainingProvenance>,
    ) -> Result<Self> {
        let checkpoint = checkpoint.as_ref();
        if res_secs == 0 {
            bail!("world-model bar resolution must be positive");
        }
        if !resolutions.contains(&res_secs) {
            bail!("deployment resolution {res_secs}s has no fitted support; got {resolutions:?}");
        }
        let mut supports_sha256 = BTreeMap::new();
        for &resolution in resolutions {
            if resolution == 0 {
                bail!("world-model bar resolution must be positive");
            }
            let path = world_model_supports_path(checkpoint, resolution);
            if supports_sha256
                .insert(resolution, file_sha256(&path)?)
                .is_some()
            {
                bail!("resolution {resolution}s listed twice");
            }
        }
        let mut metadata = Self {
            format_version: BAR_METADATA_VERSION,
            architecture: BAR_ARCHITECTURE.to_owned(),
            model_dim: BAR_MODEL_DIM,
            layers: BAR_LAYERS as i64,
            heads: BAR_HEADS,
            head_dim: BAR_HEAD_DIM,
            ff_dim: BAR_FF_DIM,
            max_context_bars: BAR_MAX_CONTEXT,
            num_bins: NUM_BAR_BINS,
            res_secs,
            supports_sha256,
            checkpoint_sha256: file_sha256(checkpoint)?,
            lineage_sha256: String::new(),
            training,
            optimizer_checkpoint_sha256: None,
            optimizer_initialized_adamw: Vec::new(),
            pretrain_recovery: None,
        };
        metadata.lineage_sha256 = metadata.compute_lineage_sha256();
        Ok(metadata)
    }

    /// Hash the weights and every support, then write the metadata sidecar next
    /// to the checkpoint. All of those files must already be on disk.
    pub fn save_for_checkpoint(
        checkpoint: impl AsRef<Path>,
        resolutions: &[u32],
        res_secs: u32,
    ) -> Result<PathBuf> {
        Self::save_for_checkpoint_with(checkpoint, resolutions, res_secs, None)
    }

    /// As [`Self::save_for_checkpoint`], with the run's data and selection provenance.
    pub fn save_for_checkpoint_with(
        checkpoint: impl AsRef<Path>,
        resolutions: &[u32],
        res_secs: u32,
        training: Option<BarTrainingProvenance>,
    ) -> Result<PathBuf> {
        let checkpoint = checkpoint.as_ref();
        let path = world_model_metadata_path(checkpoint);
        Self::for_checkpoint_with(checkpoint, resolutions, res_secs, training)?.save(&path)?;
        Ok(path)
    }

    /// Attach the optimizer and exact-continuation contract to recovery metadata, then
    /// recompute lineage so neither the sidecar hash nor the schedule can be edited in place.
    pub fn attach_pretrain_recovery(
        &mut self,
        optimizer_checkpoint_sha256: String,
        optimizer_initialized_adamw: Vec<String>,
        recovery: BarPretrainRecovery,
    ) {
        self.optimizer_checkpoint_sha256 = Some(optimizer_checkpoint_sha256);
        self.optimizer_initialized_adamw = optimizer_initialized_adamw;
        self.pretrain_recovery = Some(recovery);
        self.lineage_sha256 = self.compute_lineage_sha256();
    }

    /// Which corpus and selection rule produced this checkpoint, when it says.
    pub fn training(&self) -> Option<&BarTrainingProvenance> {
        self.training.as_ref()
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("failed to open world-model metadata {}", path.display()))?;
        serde_json::from_reader(BufReader::new(file))
            .with_context(|| format!("failed to parse world-model metadata {}", path.display()))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate_schema()?;
        let path = path.as_ref();
        let file = File::create(path)
            .with_context(|| format!("failed to create world-model metadata {}", path.display()))?;
        serde_json::to_writer_pretty(BufWriter::new(file), self)
            .with_context(|| format!("failed to write world-model metadata {}", path.display()))
    }

    pub fn validate_checkpoint(&self, checkpoint: impl AsRef<Path>) -> Result<()> {
        self.validate_schema()?;
        let actual = file_sha256(checkpoint.as_ref())?;
        if actual != self.checkpoint_sha256 {
            bail!(
                "world-model checkpoint hash mismatch: metadata={}, actual={actual}",
                self.checkpoint_sha256
            );
        }
        Ok(())
    }

    /// Resolutions this checkpoint carries a support for, ascending.
    pub fn resolutions(&self) -> Vec<u32> {
        self.supports_sha256.keys().copied().collect()
    }

    /// The supports define the meaning of every predicted bin, so a mismatch
    /// here is as fatal as a weight mismatch. Checks every resolution.
    pub fn validate_supports(&self, checkpoint: impl AsRef<Path>) -> Result<()> {
        let checkpoint = checkpoint.as_ref();
        for (&resolution, expected) in &self.supports_sha256 {
            let path = world_model_supports_path(checkpoint, resolution);
            let actual = file_sha256(&path)?;
            if &actual != expected {
                bail!(
                    "world-model {resolution}s supports hash mismatch at {}: metadata={expected}, actual={actual}",
                    path.display()
                );
            }
        }
        Ok(())
    }

    /// Require the supports a caller will actually use to carry exactly the bin geometry the
    /// authenticated checkpoint was trained against. The checkpoint-adjacent files are hash
    /// checked before they are loaded, so copied or tampered support sidecars cannot certify a
    /// different output space.
    pub fn validate_support_geometry(
        &self,
        checkpoint: impl AsRef<Path>,
        actual: &BarSupportSet,
    ) -> Result<()> {
        let checkpoint = checkpoint.as_ref();
        self.validate_supports(checkpoint)?;
        ensure!(
            actual.resolutions() == self.resolutions(),
            "world-model support resolutions mismatch: checkpoint={:?}, current={:?}",
            self.resolutions(),
            actual.resolutions()
        );
        for resolution in self.resolutions() {
            let expected = BarSupports::load(&world_model_supports_path(checkpoint, resolution))
                .with_context(|| {
                    format!("failed loading authenticated {resolution}s checkpoint supports")
                })?;
            let current = actual
                .get(resolution)
                .with_context(|| format!("current run has no {resolution}s supports"))?;
            let same = (0..BAR_DOF).all(|dof| {
                expected.lower_bounds(dof) == current.lower_bounds(dof)
                    && expected.widths(dof) == current.widths(dof)
            });
            ensure!(
                same,
                "world-model {resolution}s support geometry differs from the authenticated \
                 checkpoint; refusing to reinterpret its output bins"
            );
        }
        Ok(())
    }

    pub fn validate_schema(&self) -> Result<()> {
        if self.format_version != BAR_METADATA_VERSION {
            bail!(
                "unsupported world-model metadata version {}, expected {BAR_METADATA_VERSION}",
                self.format_version
            );
        }
        if self.architecture != BAR_ARCHITECTURE {
            bail!(
                "incompatible world-model architecture {}, expected {BAR_ARCHITECTURE}",
                self.architecture
            );
        }
        for (name, actual, expected) in [
            ("model_dim", self.model_dim, BAR_MODEL_DIM),
            ("layers", self.layers, BAR_LAYERS as i64),
            ("heads", self.heads, BAR_HEADS),
            ("head_dim", self.head_dim, BAR_HEAD_DIM),
            ("ff_dim", self.ff_dim, BAR_FF_DIM),
            ("max_context_bars", self.max_context_bars, BAR_MAX_CONTEXT),
            ("num_bins", self.num_bins, NUM_BAR_BINS),
        ] {
            if actual != expected {
                bail!("incompatible world-model {name} {actual}, expected {expected}");
            }
        }
        if self.res_secs == 0 {
            bail!("world-model bar resolution must be positive");
        }
        if let Some(training) = &self.training {
            ensure!(
                training.direct_t2_weight.is_finite()
                    && training.direct_t2_weight >= 0.0
                    && training.direct_t3_weight.is_finite()
                    && training.direct_t3_weight >= 0.0,
                "direct horizon weights must be finite and non-negative"
            );
        }
        if let Some(recovery) = &self.pretrain_recovery {
            ensure!(
                recovery.format_version == BAR_PRETRAIN_RECOVERY_VERSION,
                "unsupported pretrain recovery version {}, expected {BAR_PRETRAIN_RECOVERY_VERSION}",
                recovery.format_version
            );
            ensure!(
                recovery.direct_t2_weight.is_finite()
                    && recovery.direct_t2_weight >= 0.0
                    && recovery.direct_t3_weight.is_finite()
                    && recovery.direct_t3_weight >= 0.0,
                "pretrain recovery direct horizon weights must be finite and non-negative"
            );
            ensure!(
                self.optimizer_checkpoint_sha256
                    .as_ref()
                    .is_some_and(|hash| !hash.is_empty()),
                "pretrain recovery metadata has no optimizer-state hash"
            );
            ensure!(
                self.training
                    .as_ref()
                    .and_then(|training| training.global_step)
                    .is_some(),
                "pretrain recovery metadata has no global optimizer step"
            );
            ensure!(
                self.optimizer_initialized_adamw
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    == self.optimizer_initialized_adamw.len(),
                "pretrain recovery metadata repeats an initialized AdamW name"
            );
            ensure!(
                recovery.derived_batch_ramp.is_empty()
                    || (recovery.derived_batch_ramp.len() == recovery.stage_steps.len()
                        && recovery
                            .derived_batch_ramp
                            .iter()
                            .all(|multiple| *multiple > 0)),
                "pretrain recovery metadata carries an invalid planned batch ramp {:?}",
                recovery.derived_batch_ramp
            );
            ensure!(
                recovery.activation_bytes_per_token == 0.0
                    || recovery.activation_bytes_per_token.is_finite()
                        && recovery.activation_bytes_per_token > 0.0,
                "pretrain recovery activation footprint must be zero or finite and positive"
            );
            ensure!(
                (recovery.activation_bytes_per_token == 0.0)
                    == recovery.activation_device.is_empty(),
                "pretrain recovery activation footprint and device provenance must be present together"
            );
        } else {
            ensure!(
                self.optimizer_checkpoint_sha256.is_none()
                    && self.optimizer_initialized_adamw.is_empty(),
                "optimizer recovery fields are present without a recovery contract"
            );
        }
        if self.supports_sha256.is_empty() {
            bail!("world-model metadata has no supports hash");
        }
        // `deployment_supports()` indexes the set by this key and cannot recover
        // if it is absent, so the invariant is established here rather than only
        // in the constructor, which is not on the load path.
        if !self.supports_sha256.contains_key(&self.res_secs) {
            bail!(
                "deployment resolution {}s has no support; metadata carries {:?}",
                self.res_secs,
                self.resolutions()
            );
        }
        let expected = self.compute_lineage_sha256();
        if self.lineage_sha256 != expected {
            bail!(
                "world-model lineage mismatch: metadata={}, expected={expected}",
                self.lineage_sha256
            );
        }
        Ok(())
    }

    /// Canonical rendering of [`Self::training`] for the lineage hash. `none` for a
    /// checkpoint that does not state its data provenance, which is itself a fact worth
    /// hashing: it distinguishes "trained on an unrecorded corpus" from any recorded one.
    fn training_canonical(&self) -> String {
        let Some(training) = &self.training else {
            return "none".to_owned();
        };
        let optimizer_suffix = training.optimizer.as_ref().map_or_else(String::new, |optimizer| {
            let row_lr = optimizer.row_learned_lr.as_ref().map_or_else(
                || "off".to_owned(),
                |controller| {
                    format!(
                        "on,c={:016x},lr={:016x},betas={:016x}:{:016x},eps={:016x},warmup={},clip={:016x},evidence={}",
                        controller.c.to_bits(),
                        controller.controller_lr.to_bits(),
                        controller.beta1.to_bits(),
                        controller.beta2.to_bits(),
                        controller.eps.to_bits(),
                        controller.warmup_primary_steps,
                        controller.evidence_clip.to_bits(),
                        controller.evidence,
                    )
                },
            );
            let ablation = optimizer.ablation.as_ref().map_or_else(String::new, |ablation| {
                let meta_lr = ablation.meta_lr.map_or_else(
                    || "none".to_owned(),
                    |value| format!("{:016x}", value.to_bits()),
                );
                let beta_bounds = ablation.beta_bounds.map_or_else(
                    || "none".to_owned(),
                    |(low, high)| format!("{:016x}:{:016x}", low.to_bits(), high.to_bits()),
                );
                let beta_update_clip = ablation.beta_update_clip.map_or_else(
                    || "none".to_owned(),
                    |value| format!("{:016x}", value.to_bits()),
                );
                format!(
                    ";ablation={},initial_lr={:016x},meta_lr={meta_lr},exact_hvp={},beta_bounds={beta_bounds},beta_update_clip={beta_update_clip}",
                    ablation.variant,
                    ablation.initial_lr.to_bits(),
                    ablation.exact_hvp,
                )
            });
            format!(
                ";optimizer={};row_learned_lr={row_lr}{ablation}",
                optimizer.name
            )
        });
        format!(
            "corpus={};bounds={}:{};pinned={};eval_seed={:016x};train_seed={:016x};\
             metric={};weights={};guard={}@{:016x};min_dollar_volume_bits={:016x};symbols={};\
             supports_frozen={};supports_corpus={};universe={};universe_train_end={};\
             scoring={};sizing_rule={};direct_weights={:016x}:{:016x};direct_schedule={};direct_flow={};context={}@{}/{};batch_ramp={}{}{}{}",
            training.corpus_fingerprint,
            training.split_bounds.0,
            training.split_bounds.1,
            training.split_bounds_pinned,
            training.eval_window_seed,
            training.train_seed,
            training.selection_metric,
            training
                .selection_weights
                .iter()
                .map(|w| format!("{:016x}", w.to_bits()))
                .collect::<Vec<_>>()
                .join(","),
            training.selection_guard_dof,
            training.selection_guard_se_multiple.to_bits(),
            training.min_dollar_volume.to_bits(),
            training.symbols,
            training.supports_frozen,
            training
                .supports_corpus_fingerprint
                .as_deref()
                .unwrap_or("none"),
            training.universe_fingerprint.as_deref().unwrap_or("none"),
            training
                .universe_train_end_ms
                .map(|ms| ms.to_string())
                .unwrap_or_else(|| "none".to_owned()),
            training.scoring,
            training.sizing_rule,
            training.direct_t2_weight.to_bits(),
            training.direct_t3_weight.to_bits(),
            direct_schedule_canonical(),
            BAR_DIRECT_FLOW_CONTRACT,
            training.selection_context,
            training.reached_context,
            training.deployed_context,
            if training.batch_ramp.is_empty() {
                "none".to_owned()
            } else {
                training
                    .batch_ramp
                    .iter()
                    .map(|m| m.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            },
            // Absent — `0.0`, which the flag refuses — renders NOTHING, so every sidecar
            // written before the fraction was recorded still validates against its own hash.
            if training.lr_plateau_fraction == 0.0 {
                String::new()
            } else {
                format!(
                    ";lr_plateau_fraction={:016x}",
                    training.lr_plateau_fraction.to_bits()
                )
            },
            optimizer_suffix,
            // Same discipline: [`DofScaling::Raw`] renders NOTHING, so every checkpoint
            // written before the parametrization was recorded — all of which trained on raw
            // targets — keeps validating against its own stored hash, while a standardized arm
            // can never collide with a raw run's lineage.
            match training.dof_scaling {
                DofScaling::Raw => String::new(),
                scaling => format!(";dof_scaling={scaling}"),
            },
        )
    }

    fn recovery_canonical(&self) -> String {
        let Some(recovery) = &self.pretrain_recovery else {
            return String::new();
        };
        let derived_batch_ramp_suffix = if recovery.derived_batch_ramp.is_empty() {
            String::new()
        } else {
            format!(
                ",derived_batch_ramp={}",
                recovery
                    .derived_batch_ramp
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        format!(
            ";optimizer_checkpoint_sha256={};optimizer_initialized_adamw={};recovery=format_version={},total_steps={},stage_steps={}{},base_batch={},requested_batch={},exact_batch={},quadratic_kelly={},mean_shrink={},epochs={},steps_override={},dyn_horizon={},lambda_dyn={:016x},lambda_kl={:016x},direct_t2_weight={:016x},direct_t3_weight={:016x},direct_schedule={},direct_flow={},auxiliary_resolutions={},adamw_accumulation_pending={},activation_bytes_per_token={:016x},activation_device={},checkpoint_every={},validate_every={},ablation_lr={:016x},smd_meta_lr={:016x}",
            self.optimizer_checkpoint_sha256.as_deref().unwrap_or("none"),
            self.optimizer_initialized_adamw.join(","),
            recovery.format_version,
            recovery.total_steps,
            recovery
                .stage_steps
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
            derived_batch_ramp_suffix,
            recovery.base_batch,
            recovery.requested_batch,
            recovery.exact_batch,
            recovery.quadratic_kelly,
            recovery.mean_shrink,
            recovery.epochs,
            recovery
                .steps_override
                .map_or_else(|| "none".to_owned(), |steps| steps.to_string()),
            recovery.dyn_horizon,
            recovery.lambda_dyn.to_bits(),
            recovery.lambda_kl.to_bits(),
            recovery.direct_t2_weight.to_bits(),
            recovery.direct_t3_weight.to_bits(),
            direct_schedule_canonical(),
            BAR_DIRECT_FLOW_CONTRACT,
            recovery
                .auxiliary_resolutions
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(","),
            recovery.adamw_accumulation_pending,
            recovery.activation_bytes_per_token.to_bits(),
            recovery.activation_device,
            recovery.checkpoint_every,
            recovery.validate_every,
            recovery.ablation_lr.to_bits(),
            recovery.smd_meta_lr.to_bits(),
        )
    }

    /// SHA-256 over every architectural constant that changes what a weight means, including
    /// the supports hash and the corpus the run was trained and scored on.
    fn compute_lineage_sha256(&self) -> String {
        let cardinality = BAR_TIME_CARDINALITY
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let boundaries = BAR_SESSION_BOUNDARY_MINUTES
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let canonical = format!(
            "format_version={};architecture={};model_dim={};layers={};heads={};head_dim={};\
             ff_dim={};max_context_bars={};num_bins={};res_secs={};dof={};dof_names={};chain={};supports_sha256={};\
             weights_sha256={};norm=rmsnorm-no-gain;norm_eps_bits={:016x};qk_norm=head-dim-rmsnorm;\
             mlp=relu-squared;resid_lambda_init_bits={:016x};post_lambda_init_bits={:016x};\
             weight_scalars=qkv-and-out;zero_init=attn-out,ff-out,dyn-fc3,x0-lambda;\
             direct_head_init=copy-h1-without-rng;shortcut=embedding-x0-per-layer-attn-residual;dynamics_hidden={};\
             dynamics_act=gelu-none;dynamics_layers=3;pope_dim={};pope_qk_dim={};\
             pope_frequency_base_bits={:016x};pope_attention_scale_bits={:016x};\
             pope_theta_init=two-pi-block-aware;pope_layout=real-then-imag;\
             fa4_contract=strict-bshd-qk128-v64;cache_contract={};label_sigma_ratio_bits={:016x};\
             prefix_embed_dim={};volume_ema_span_bits={:016x};time_features={};\
             time_cardinality={cardinality};time_conditioning={};\
             session_boundary_minutes={boundaries};training={}{}",
            self.format_version,
            self.architecture,
            self.model_dim,
            self.layers,
            self.heads,
            self.head_dim,
            self.ff_dim,
            self.max_context_bars,
            self.num_bins,
            self.res_secs,
            BAR_DOF,
            BAR_DOF_NAMES.join(","),
            BAR_CHAIN
                .iter()
                .map(|slot| BAR_DOF_NAMES[*slot])
                .collect::<Vec<_>>()
                .join(","),
            self.supports_sha256
                .iter()
                .map(|(res, sha)| format!("{res}:{sha}"))
                .collect::<Vec<_>>()
                .join(","),
            self.checkpoint_sha256,
            BAR_NORM_EPS.to_bits(),
            BAR_RESID_LAMBDA_INIT.to_bits(),
            BAR_POST_LAMBDA_INIT.to_bits(),
            BAR_DYNAMICS_HIDDEN,
            POPE_DIM,
            POPE_QK_DIM,
            POPE_FREQUENCY_BASE.to_bits(),
            POPE_ATTENTION_SCALE.to_bits(),
            BAR_CACHE_CONTRACT,
            BAR_LABEL_SIGMA_RATIO.to_bits(),
            BAR_PREFIX_EMBED_DIM,
            BAR_VOLUME_EMA_SPAN.to_bits(),
            BAR_TIME_FEATURES,
            BAR_TIME_CONDITIONING,
            self.training_canonical(),
            self.recovery_canonical(),
        );
        digest(&SHA256, canonical.as_bytes())
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

/// `foo.ot` -> `foo.<suffix>`. A checkpoint whose name is not `*.ot` gets the
/// suffix appended rather than substituted, so two checkpoints that differ only
/// in a dotted stem (`model.v2` and `model.v3`) cannot share a sidecar.
fn sidecar_path(checkpoint: &Path, suffix: &str) -> PathBuf {
    if checkpoint
        .extension()
        .is_some_and(|extension| extension == "ot")
    {
        return checkpoint.with_extension(suffix);
    }
    let mut name = checkpoint.file_name().unwrap_or_default().to_os_string();
    name.push(".");
    name.push(suffix);
    checkpoint.with_file_name(name)
}

/// Metadata sidecar of a checkpoint: `foo.ot` -> `foo.metadata.json`.
pub fn world_model_metadata_path(checkpoint: impl AsRef<Path>) -> PathBuf {
    sidecar_path(checkpoint.as_ref(), "metadata.json")
}

/// Optimizer recovery sidecar: `foo.ot` -> `foo.optimizer.ot`.
pub fn world_model_optimizer_path(checkpoint: impl AsRef<Path>) -> PathBuf {
    sidecar_path(checkpoint.as_ref(), "optimizer.ot")
}

/// Supports sidecar of a checkpoint at one resolution:
/// `foo.ot` -> `foo.supports.300.json`. One file per resolution, because a
/// 128-bin equal-mass support discretizes one resolution's distribution.
pub fn world_model_supports_path(checkpoint: impl AsRef<Path>, res_secs: u32) -> PathBuf {
    sidecar_path(checkpoint.as_ref(), &format!("supports.{res_secs}.json"))
}

// ---------------------------------------------------------------------------
// Per-resolution supports
// ---------------------------------------------------------------------------

/// Supports keyed by resolution class.
///
/// A daily bar's return and range live on a different scale from a five-minute
/// bar's, so one shared equal-mass support would spend most of its 128 bins
/// separating timeframes instead of separating outcomes within a timeframe.
/// `TIME_RESOLUTION` lets the trunk know which resolution a row is; this is what
/// gives that row the bins its distribution was fitted on.
pub struct BarSupportSet {
    /// Ascending by resolution class: `(class, res_secs, supports)`.
    entries: Vec<(i64, u32, BarSupports)>,
}

impl BarSupportSet {
    pub fn new(entries: Vec<(u32, BarSupports)>) -> Result<Self> {
        if entries.is_empty() {
            bail!("a world model needs at least one fitted support");
        }
        let mut keyed: Vec<(i64, u32, BarSupports)> = entries
            .into_iter()
            .map(|(res_secs, supports)| (resolution_class(res_secs), res_secs, supports))
            .collect();
        keyed.sort_by_key(|(class, _, _)| *class);
        if let Some(window) = keyed.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            // Two resolutions sharing a class would silently share one
            // discretization, which is the failure per-resolution supports exist
            // to prevent.
            bail!(
                "resolutions {}s and {}s share resolution class {}",
                window[0].1,
                window[1].1,
                window[0].0
            );
        }
        Ok(Self { entries: keyed })
    }

    pub fn to_device(&self, device: Device) -> Self {
        Self {
            entries: self
                .entries
                .iter()
                .map(|(class, res, supports)| (*class, *res, supports.to_device(device)))
                .collect(),
        }
    }

    /// Resolutions carried, ascending by class.
    pub fn resolutions(&self) -> Vec<u32> {
        self.entries.iter().map(|(_, res, _)| *res).collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn get(&self, res_secs: u32) -> Option<&BarSupports> {
        let class = resolution_class(res_secs);
        self.entries
            .iter()
            .find(|(entry, _, _)| *entry == class)
            .map(|(_, _, supports)| supports)
    }

    /// The only support, for single-resolution call sites.
    pub fn only(&self) -> &BarSupports {
        assert_eq!(
            self.entries.len(),
            1,
            "expected a single-resolution support set, got {:?}",
            self.resolutions()
        );
        &self.entries[0].2
    }

    /// Every row's resolution class must be one this set carries. A row that
    /// matched nothing would otherwise keep its zero seed — bin 0 across all five
    /// DOF, or an all-zero DOF vector that decodes to a plausible flat bar — and
    /// nothing downstream could tell it from real data. That is exactly the
    /// failure per-resolution supports exist to prevent, and it is distinct from
    /// the duplicate-class collision `new` rejects.
    fn assert_rows_routed(&self, matched: &Tensor, rows: i64) {
        let covered =
            i64::try_from(matched.to_kind(Kind::Int64).sum(Kind::Int64)).expect("routed row count");
        assert_eq!(
            covered,
            rows,
            "{} of {rows} rows carry a resolution class this model has no support for; carried: {:?}",
            rows - covered,
            self.resolutions()
        );
    }

    /// `[..., BAR_DOF]` DOF and `[..., BAR_TIME_FEATURES]` time ids ->
    /// `[..., BAR_DOF]` bin ids, each row read against its own resolution's bins.
    pub fn bin_ids(&self, dof: &Tensor, time_ids: &Tensor) -> Tensor {
        if self.entries.len() == 1 {
            return self.entries[0].2.bin_ids(dof);
        }
        let class = time_ids.select(-1, TIME_RESOLUTION as i64).unsqueeze(-1);
        let mut out = Tensor::zeros(dof.size().as_slice(), (Kind::Int64, dof.device()));
        let mut matched = Tensor::zeros(class.size().as_slice(), (Kind::Bool, class.device()));
        for (entry, _, supports) in &self.entries {
            let selected = class.eq(*entry);
            out = out.where_self(&selected.logical_not(), &supports.bin_ids(dof));
            matched = matched.logical_or(&selected);
        }
        self.assert_rows_routed(&matched, matched.numel() as i64);
        out
    }

    /// Row-routed ancestral sample, `[..., BAR_DOF]` from beliefs and their
    /// explicit forecast conditioning `[..., dim]`.
    pub fn sample(
        &self,
        head: &BarEmissionHead,
        h: &Tensor,
        conditioning: &Tensor,
        time_ids: &Tensor,
        temperature: f64,
    ) -> Tensor {
        self.sample_binned(head, h, conditioning, time_ids, temperature)
            .0
    }

    /// Row-routed ancestral sample returning both decoded values and the exact drawn bins,
    /// each shaped `[..., BAR_DOF]`.
    ///
    /// Rows are PARTITIONED by resolution and each partition is sampled in its
    /// own [`BarEmissionHead::sample_binned`] call. The chain is sequential — each
    /// sampled DOF conditions the next — and every step reads its support's bin
    /// bounds and atoms, so masking two supports together inside one pass would
    /// interleave two bin geometries and silently draw plausible-looking bars
    /// from the wrong bins.
    pub fn sample_binned(
        &self,
        head: &BarEmissionHead,
        h: &Tensor,
        conditioning: &Tensor,
        time_ids: &Tensor,
        temperature: f64,
    ) -> (Tensor, Tensor) {
        if self.entries.len() == 1 {
            return head.sample_binned(h, conditioning, &self.entries[0].2, temperature);
        }
        let shape = h.size();
        assert_eq!(
            conditioning.size(),
            shape,
            "beliefs and forecast conditioning must have identical shapes"
        );
        let dim = *shape.last().expect("latent must be ranked");
        let rows = h.numel() as i64 / dim;
        let flat = h.reshape([rows, dim]);
        let flat_conditioning = conditioning.reshape([rows, dim]);
        let class = time_ids
            .reshape([rows, BAR_TIME_FEATURES as i64])
            .select(1, TIME_RESOLUTION as i64);
        let mut values = Tensor::zeros([rows, BAR_DOF as i64], (Kind::Float, h.device()));
        let mut bins = Tensor::zeros([rows, BAR_DOF as i64], (Kind::Int64, h.device()));
        let mut matched = Tensor::zeros([rows], (Kind::Bool, h.device()));
        for (entry, _, supports) in &self.entries {
            let selected = class.eq(*entry);
            matched = matched.logical_or(&selected);
            let index = selected.nonzero().reshape([-1]);
            if index.numel() == 0 {
                continue;
            }
            let (drawn_values, drawn_bins) = head.sample_binned(
                &flat.index_select(0, &index),
                &flat_conditioning.index_select(0, &index),
                supports,
                temperature,
            );
            values = values.index_copy(0, &index, &drawn_values);
            bins = bins.index_copy(0, &index, &drawn_bins);
        }
        self.assert_rows_routed(&matched, rows);
        let mut target = shape[..shape.len() - 1].to_vec();
        target.push(BAR_DOF as i64);
        (
            values.reshape(target.as_slice()),
            bins.reshape(target.as_slice()),
        )
    }
}

/// `[..., BAR_DOF]` DOF values -> `[..., BAR_DOF]` `i64` bin ids.
///
/// A thin alias for [`BarSupports::bin_ids`], which owns the clamping and the
/// zero-width atom overrides. Never re-derive this from the bin bounds: an exact
/// atom value would land in the neighbouring continuous bin and silently corrupt
/// the trunk's input embedding.
pub fn bar_bin_ids(supports: &BarSupports, dof: &Tensor) -> Tensor {
    supports.bin_ids(dof)
}

// ---------------------------------------------------------------------------
// Trunk
// ---------------------------------------------------------------------------

struct BarLayer {
    /// `[3 * D, D]`
    qkv_w: Tensor,
    qkv_lambda: Tensor,
    /// `[D, D]`, zero-init
    attn_out_w: Tensor,
    attn_out_lambda: Tensor,
    /// `[BAR_HEADS, POPE_DIM]`
    pope_theta_bias: Tensor,
    attn_resid_lambda: Tensor,
    attn_post_lambda: Tensor,
    /// Gain on the embedding shortcut, zero at init. Fused into the attention residual rather
    /// than added as a separate pre-layer term, which is where the reference recipe puts it
    /// (`../modded-nanogpt/train_gpt.py:1638`).
    x0_lambda: Tensor,
    /// `[F, D]`
    ff_in_w: Tensor,
    /// `[D, F]`, zero-init
    ff_out_w: Tensor,
    ff_resid_lambda: Tensor,
    ff_post_lambda: Tensor,
}

impl BarLayer {
    fn new(p: &nn::Path) -> Self {
        Self {
            qkv_w: p.var(
                "qkv_w",
                &[BAR_QKV_DIM, BAR_MODEL_DIM],
                uniform_init(BAR_MODEL_DIM),
            ),
            qkv_lambda: p.var("qkv_lambda", &[1], Init::Const(1.0)),
            attn_out_w: p.var(
                "attn_out_w",
                &[BAR_MODEL_DIM, BAR_MODEL_DIM],
                Init::Const(0.0),
            ),
            attn_out_lambda: p.var("attn_out_lambda", &[1], Init::Const(1.0)),
            pope_theta_bias: init_pope_theta_bias(
                p,
                "pope_theta_bias",
                BAR_HEADS,
                BAR_HEAD_DIM,
                BAR_MAX_CONTEXT,
                PopeThetaInit::TwoPi,
            ),
            attn_resid_lambda: p.var(
                "attn_resid_lambda",
                &[1],
                Init::Const(BAR_RESID_LAMBDA_INIT),
            ),
            attn_post_lambda: p.var("attn_post_lambda", &[1], Init::Const(BAR_POST_LAMBDA_INIT)),
            x0_lambda: p.var("x0_lambda", &[1], Init::Const(0.0)),
            ff_in_w: p.var(
                "ff_in_w",
                &[BAR_FF_DIM, BAR_MODEL_DIM],
                uniform_init(BAR_MODEL_DIM),
            ),
            ff_out_w: p.var("ff_out_w", &[BAR_MODEL_DIM, BAR_FF_DIM], Init::Const(0.0)),
            ff_resid_lambda: p.var("ff_resid_lambda", &[1], Init::Const(BAR_RESID_LAMBDA_INIT)),
            ff_post_lambda: p.var("ff_post_lambda", &[1], Init::Const(BAR_POST_LAMBDA_INIT)),
        }
    }

    /// Fused QKV with the learned scalar folded into the weight matrix, then
    /// per-head QK-norm. Returns `[rows, len, BAR_HEADS, BAR_HEAD_DIM]` triples.
    fn qkv(&self, normed: &Tensor) -> (Tensor, Tensor, Tensor) {
        let rows = normed.size()[0];
        let len = normed.size()[1];
        let fused = normed.linear(&(&self.qkv_lambda * &self.qkv_w), None::<Tensor>);
        let parts = fused.split(BAR_MODEL_DIM, -1);
        let heads = |t: &Tensor| t.view([rows, len, BAR_HEADS, BAR_HEAD_DIM]);
        (
            qk_norm(&heads(&parts[0])),
            qk_norm(&heads(&parts[1])),
            heads(&parts[2]),
        )
    }

    /// `x0` is the trunk's embedding output for the same bars. At init its gain is exactly zero,
    /// so this is a bit-identical no-op and a run cannot be destabilized by the shortcut existing;
    /// what it buys is a path from the current bar's own bins to every depth, which for bars is
    /// the single strongest predictor of the next bar and otherwise has to survive ten layers of
    /// residual mixing to be read.
    fn attention_residual(&self, x: &Tensor, attention: &Tensor, x0: &Tensor) -> Tensor {
        let rows = x.size()[0];
        let len = x.size()[1];
        let flat = attention
            .to_kind(x.kind())
            .contiguous()
            .view([rows, len, BAR_MODEL_DIM]);
        let out = flat.linear(&(&self.attn_out_lambda * &self.attn_out_w), None::<Tensor>);
        &self.attn_resid_lambda * x + &self.attn_post_lambda * out + &self.x0_lambda * x0
    }

    fn feed_forward(&self, x: &Tensor) -> Tensor {
        let hidden = rms_norm(x).linear(&self.ff_in_w, None::<Tensor>).relu();
        let out = (&hidden * &hidden).linear(&self.ff_out_w, None::<Tensor>);
        &self.ff_resid_lambda * x + &self.ff_post_lambda * out
    }
}

/// Fixed, training-free residual recirculation between two trunk depths.
///
/// Layer indices are zero-based. For each token, the first pass captures the
/// post-destination and post-source residuals. Their norm-matched mixture is
/// rerun through layers above the destination, overwriting that token's upper
/// KV entries for future tokens while the first-pass final residual is read out.
#[derive(Clone, Debug, PartialEq)]
pub struct RecirculationConfig {
    source_layer: usize,
    destination_layer: usize,
    alpha: f64,
    beta_one: bool,
    ramp_positions: usize,
}

impl RecirculationConfig {
    pub fn new(
        source_layer: usize,
        destination_layer: usize,
        alpha: f64,
        beta_one: bool,
        ramp_positions: usize,
    ) -> Result<Self> {
        let config = Self {
            source_layer,
            destination_layer,
            alpha,
            beta_one,
            ramp_positions,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.source_layer >= BAR_LAYERS {
            bail!(
                "recirculation source layer {} is outside the zero-based 0..{} trunk",
                self.source_layer,
                BAR_LAYERS
            );
        }
        if self.destination_layer >= BAR_LAYERS {
            bail!(
                "recirculation destination layer {} is outside the zero-based 0..{} trunk",
                self.destination_layer,
                BAR_LAYERS
            );
        }
        if self.destination_layer >= self.source_layer {
            bail!(
                "recirculation requires destination < source, got destination {} and source {}",
                self.destination_layer,
                self.source_layer
            );
        }
        if !self.alpha.is_finite() || !(0.0..=1.0).contains(&self.alpha) {
            bail!(
                "recirculation alpha must be finite and in [0, 1], got {}",
                self.alpha
            );
        }
        if self.ramp_positions == 0 {
            bail!("recirculation ramp positions must be positive");
        }
        Ok(())
    }

    pub fn source_layer(&self) -> usize {
        self.source_layer
    }

    pub fn destination_layer(&self) -> usize {
        self.destination_layer
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn beta_one(&self) -> bool {
        self.beta_one
    }

    pub fn ramp_positions(&self) -> usize {
        self.ramp_positions
    }

    fn alpha_at(&self, position: usize) -> f64 {
        self.alpha * (position.min(self.ramp_positions) as f64 / self.ramp_positions as f64)
    }
}

/// Causal PoPE/FA4 transformer over bars.
pub struct BarTrunk {
    /// Five `[NUM_BAR_BINS, D]` tables, one per degree of freedom.
    bin_embed: Vec<Tensor>,
    /// `[D, BAR_DOF]`, the raw continuous DOF input map.
    dof_embed_w: Tensor,
    /// `[BAR_TIME_EMBED_ROWS, D]`, all [`BAR_TIME_FEATURES`] conditioning channels in one bank.
    time_embed: Tensor,
    /// `[1, 1, BAR_DOF]` constant, `dof * NUM_BAR_BINS`, for the fused lookup.
    bin_offsets: Tensor,
    /// `[1, 1, BAR_TIME_FEATURES]` constant, the block base of each channel.
    time_offsets: Tensor,
    layers: Vec<BarLayer>,
}

impl BarTrunk {
    pub fn new(vs: &nn::Path) -> Self {
        let bin_embed = BAR_DOF_NAMES
            .iter()
            .map(|name| {
                vs.var(
                    &format!("bar_bin_embed_{name}"),
                    &[NUM_BAR_BINS, BAR_MODEL_DIM],
                    uniform_init(BAR_MODEL_DIM),
                )
            })
            .collect();
        let dof_embed_w = vs.var(
            "bar_dof_embed_w",
            &[BAR_MODEL_DIM, BAR_DOF as i64],
            uniform_init(BAR_MODEL_DIM),
        );
        let time_embed = vs.var(
            "bar_time_embed",
            &[BAR_TIME_EMBED_ROWS, BAR_MODEL_DIM],
            uniform_init(BAR_MODEL_DIM),
        );
        let bin_offsets = (Tensor::arange(BAR_DOF as i64, (Kind::Int64, vs.device()))
            * NUM_BAR_BINS)
            .view([1, 1, BAR_DOF as i64]);
        let time_offsets = time_block_offsets(vs.device());
        let layers = (0..BAR_LAYERS)
            .map(|index| BarLayer::new(&(vs / format!("bar_layer_{index}"))))
            .collect();
        Self {
            bin_embed,
            dof_embed_w,
            time_embed,
            bin_offsets,
            time_offsets,
            layers,
        }
    }

    /// 2D projection banks, for the NorMuon/AdamW partition.
    pub fn muon_name_substrings() -> &'static [&'static str] {
        &BAR_TRUNK_MUON_SUBSTRINGS
    }

    /// `dof [B,T,BAR_DOF]`, `bin_ids [B,T,BAR_DOF]` and observed
    /// `time_ids [B,T,BAR_TIME_FEATURES]` -> forecast-safe beliefs `[B,T,D]`.
    ///
    /// `window` is a sliding causal span in bars; `window <= 0`, or a window at
    /// least as long as the sequence, is full causal attention and takes the FA4
    /// kernel. `train == false` runs the pass under `no_grad` and detaches.
    pub fn forward(
        &self,
        dof: &Tensor,
        bin_ids: &Tensor,
        time_ids: &Tensor,
        window: i64,
        train: bool,
    ) -> Tensor {
        if train {
            return self.run(dof, bin_ids, time_ids, window);
        }
        tch::no_grad(|| self.run(dof, bin_ids, time_ids, window).detach())
    }
    /// Serialized teacher-forced trunk pass using one evolving KV cache.
    ///
    /// This is the control arm for fixed recirculation. It is deliberately
    /// separate from [`Self::forward`]: the experiment must not mistake
    /// parallel-prefill versus decode-kernel drift for a recirculation effect.
    pub fn forward_serialized(&self, dof: &Tensor, bin_ids: &Tensor, time_ids: &Tensor) -> Tensor {
        tch::no_grad(|| {
            self.run_serialized(dof, bin_ids, time_ids, None)
                .expect("the no-recirculation serialized path has no invalid config")
                .detach()
        })
    }

    /// Evaluation-only collapsed sequential recirculation.
    ///
    /// One bar is one token. A token's post-source residual is norm-matched
    /// into its own post-destination residual, then layers above the destination
    /// are rerun to overwrite that token's upper KV state for future bars. The
    /// first-pass final residual remains the readout. Normal forward, prefill,
    /// cached serving and training never enter this path.
    pub fn forward_recirculated(
        &self,
        dof: &Tensor,
        bin_ids: &Tensor,
        time_ids: &Tensor,
        config: &RecirculationConfig,
    ) -> Result<Tensor> {
        config.validate()?;
        tch::no_grad(|| {
            self.run_serialized(dof, bin_ids, time_ids, Some(config))
                .map(|beliefs| beliefs.detach())
        })
    }

    fn run_serialized(
        &self,
        dof: &Tensor,
        bin_ids: &Tensor,
        time_ids: &Tensor,
        config: Option<&RecirculationConfig>,
    ) -> Result<Tensor> {
        let tokens = self.token_embedding(dof, bin_ids, time_ids);
        let length = tokens.size()[1];
        if length <= 0 {
            bail!("serialized bar trunk requires at least one token");
        }
        if length > BAR_MAX_CONTEXT {
            bail!("serialized bar trunk length {length} exceeds BAR_MAX_CONTEXT {BAR_MAX_CONTEXT}");
        }
        let mut cache = BarKvCache::new(BAR_MAX_CONTEXT);
        let mut beliefs = Vec::with_capacity(length as usize);
        for position in 0..length {
            let token = tokens.narrow(1, position, 1);
            let belief = if position == 0 {
                self.prefill_serialized_token(&token, &mut cache, config)?
            } else {
                self.decode_serialized_token(&token, &mut cache, config)?
            };
            beliefs.push(belief);
        }
        Ok(Tensor::cat(&beliefs, 1))
    }

    fn recirculate_residual(
        destination: &Tensor,
        source: &Tensor,
        config: &RecirculationConfig,
        position: usize,
    ) -> Tensor {
        let alpha = config.alpha_at(position);
        if alpha == 0.0 {
            return destination.shallow_clone();
        }
        let destination_norm = destination
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist([-1i64].as_slice(), true, Kind::Float)
            .sqrt();
        let source_norm = source
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist([-1i64].as_slice(), true, Kind::Float)
            .sqrt()
            .clamp_min(BAR_NORM_EPS);
        let matched = source * (destination_norm / source_norm).to_kind(source.kind());
        let beta = if config.beta_one { 1.0 } else { 1.0 - alpha };
        beta * destination + alpha * matched
    }

    fn run(&self, dof: &Tensor, bin_ids: &Tensor, time_ids: &Tensor, window: i64) -> Tensor {
        backward_probe::trace_step_boundary();
        let embed_region = profile::region_fine(profile::TRUNK_EMBED);
        let bin_table = self.bin_table();
        let x0 = self.token_embedding_with_table(&bin_table, dof, bin_ids, time_ids);
        drop(embed_region);
        // Boundaries for the backward probe, named for the interval each one closes. `x0` also
        // feeds every layer's attention residual, so its gradient is the last the trunk
        // completes and the table's is the first below the trunk.
        backward_probe::mark(&bin_table, profile::BACKWARD_EMBED);
        backward_probe::mark(&x0, profile::BACKWARD_NORM);
        let mut x = x0.shallow_clone();
        let len = x.size()[1];
        let phases_region = profile::region_fine(profile::TRUNK_POPE);
        let positions = Tensor::arange(len, (Kind::Int64, x.device()));
        let phases = PopePhases::new(&positions, x.device(), POPE_FREQUENCY_BASE);
        drop(phases_region);
        for layer in &self.layers {
            let norm_region = profile::region_fine(profile::TRUNK_NORM);
            let normed = rms_norm(&x);
            drop(norm_region);
            backward_probe::mark(&normed, profile::BACKWARD_POPE);
            let pope_region = profile::region_fine(profile::TRUNK_POPE);
            let (query, key, value) = layer.qkv(&normed);
            let kind = attention_kind(&x);
            let polar = pope_expand_qk(&query, &key, &phases, &layer.pope_theta_bias, kind);
            let value = value.to_kind(kind).contiguous();
            drop(pope_region);
            backward_probe::mark(&polar.query, profile::BACKWARD_ATTN);
            let attn_region = profile::region_fine(profile::TRUNK_ATTN);
            let attention = if window > 0 && window < len {
                windowed_attention(&polar.query, &polar.key, &value, window)
            } else {
                strict_pope_prefill(&polar, &value)
            };
            backward_probe::mark(&attention, profile::BACKWARD_FFN);
            x = layer.attention_residual(&x, &attention, &x0);
            drop(attn_region);
            let ffn_region = profile::region_fine(profile::TRUNK_FFN);
            x = layer.feed_forward(&x);
            drop(ffn_region);
            backward_probe::mark(&x, profile::BACKWARD_NORM);
        }
        let norm_region = profile::region_fine(profile::TRUNK_NORM);
        let belief = rms_norm(&x);
        drop(norm_region);
        backward_probe::begin(&belief);
        belief
    }

    /// Cached forward. An empty cache is prefilled with the whole sequence; a
    /// warm cache advances by exactly one bar per call. Always `no_grad`.
    pub fn forward_cached(
        &self,
        dof: &Tensor,
        bin_ids: &Tensor,
        time_ids: &Tensor,
        cache: &mut BarKvCache,
    ) -> Tensor {
        tch::no_grad(|| {
            let x = self.token_embedding(dof, bin_ids, time_ids);
            if cache.length == 0 {
                self.prefill(&x, cache)
            } else {
                assert_eq!(
                    x.size()[1],
                    1,
                    "a warm bar KV cache advances one bar at a time"
                );
                self.decode(&x, cache)
            }
        })
    }

    /// The five per-DOF tables as one `[BAR_DOF * NUM_BAR_BINS, D]` gather bank. Still
    /// an autograd node over the same parameters, so hoisting it out of a multi-token
    /// loop changes nothing but the number of concatenations.
    pub fn bin_table(&self) -> Tensor {
        Tensor::cat(&self.bin_embed, 0)
    }

    /// Exact shared token embedding used by the trunk and dynamics: discrete bins,
    /// raw continuous DOF and exogenous clock, summed and normalized. The observed
    /// same-instant market channels are always replaced by `dataset::MARKET_MISSING` before
    /// lookup, so neither training nor cached serving can leak a bar's own market row.
    pub fn token_embedding(&self, dof: &Tensor, bin_ids: &Tensor, time_ids: &Tensor) -> Tensor {
        self.token_embedding_with_table(&self.bin_table(), dof, bin_ids, time_ids)
    }

    /// [`Self::token_embedding`] against a [`Self::bin_table`] the caller already owns.
    /// Rollouts that embed one token per step hoist the bank instead of rebuilding it.
    pub fn token_embedding_with_table(
        &self,
        bin_table: &Tensor,
        dof: &Tensor,
        bin_ids: &Tensor,
        time_ids: &Tensor,
    ) -> Tensor {
        let shape = dof.size();
        assert_eq!(shape.len(), 3, "bar DOF must be [batch, len, BAR_DOF]");
        assert_eq!(shape[2], BAR_DOF as i64, "bar DOF must have BAR_DOF slots");
        assert_eq!(bin_ids.size(), shape, "bin ids must match the DOF shape");
        let (batch, len) = (shape[0], shape[1]);
        assert_eq!(
            time_ids.size(),
            [batch, len, BAR_TIME_FEATURES as i64],
            "time ids must be [batch, len, BAR_TIME_FEATURES]"
        );
        let device = self.dof_embed_w.device();
        let bins = Tensor::embedding(
            bin_table,
            &(bin_ids.to_device(device) + self.bin_offsets.to_device(device)).reshape([-1]),
            -1,
            false,
            false,
        )
        .view([batch, len, BAR_DOF as i64, BAR_MODEL_DIM])
        .sum_dim_intlist([2i64].as_slice(), false, Kind::Float);
        let clock = self.time_embedding(&time_ids_without_market(time_ids));
        let raw = dof
            .to_device(device)
            .to_kind(Kind::Float)
            .linear(&self.dof_embed_w, None::<Tensor>);
        rms_norm(&(bins + clock + raw))
    }

    /// Shared fused embedding of all nine conditioning ids. Callers choose which
    /// market row is legal before entering this primitive.
    fn time_embedding(&self, time_ids: &Tensor) -> Tensor {
        let shape = time_ids.size();
        assert_eq!(
            shape.last().copied(),
            Some(BAR_TIME_FEATURES as i64),
            "time ids must end in BAR_TIME_FEATURES"
        );
        let device = self.time_embed.device();
        let rows = time_ids.numel() as i64 / BAR_TIME_FEATURES as i64;
        let embedded = Tensor::embedding(
            &self.time_embed,
            &(time_ids
                .to_device(device)
                .reshape([rows, BAR_TIME_FEATURES as i64])
                + self
                    .time_offsets
                    .to_device(device)
                    .view([1, BAR_TIME_FEATURES as i64]))
            .reshape([-1]),
            -1,
            false,
            false,
        )
        .view([rows, BAR_TIME_FEATURES as i64, BAR_MODEL_DIM])
        .sum_dim_intlist([1i64].as_slice(), false, Kind::Float);
        let mut output_shape = shape[..shape.len() - 1].to_vec();
        output_shape.push(BAR_MODEL_DIM);
        rms_norm(&embedded).reshape(output_shape.as_slice())
    }

    /// Conditioning for predicting a target bar. Exogenous slots through the
    /// day-edge flag come from the target's knowable clock; the three [`BAR_TIME_MARKET`] slots
    /// come from the current bar. No target market id is ever read.
    pub fn forecast_conditioning(
        &self,
        target_time_ids: &Tensor,
        current_time_ids: &Tensor,
    ) -> Tensor {
        assert_eq!(
            target_time_ids.size(),
            current_time_ids.size(),
            "target and current conditioning ids must have identical shapes"
        );
        let device = self.time_embed.device();
        let target = time_ids_without_market(&target_time_ids.to_device(device));
        let current = current_time_ids.to_device(device);
        let channels: [i64; 3] = BAR_TIME_MARKET.map(|slot| slot as i64);
        let index = Tensor::from_slice(&channels).to_device(device);
        let composed = target.index_copy(-1, &index, &current.index_select(-1, &index));
        self.time_embedding(&composed)
    }

    fn prefill(&self, tokens: &Tensor, cache: &mut BarKvCache) -> Tensor {
        let len = tokens.size()[1];
        assert!(
            len <= cache.max_tokens,
            "prefill of {len} bars exceeds the {} bar cache window",
            cache.max_tokens
        );
        let positions = Tensor::arange(len, (Kind::Int64, tokens.device()));
        let phases = PopePhases::new(&positions, tokens.device(), POPE_FREQUENCY_BASE);
        let capacity = ((len as u64).next_power_of_two() as i64).min(cache.max_tokens);
        let mut x = tokens.shallow_clone();
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (query, key, value) = layer.qkv(&rms_norm(&x));
            let kind = attention_kind(&x);
            let polar = pope_expand_qk(&query, &key, &phases, &layer.pope_theta_bias, kind);
            let value = value.to_kind(kind).contiguous();
            let attention = strict_pope_prefill(&polar, &value);
            x = layer.attention_residual(&x, &attention, tokens);
            x = layer.feed_forward(&x);
            layers.push(BarLayerKv::prefilled(&polar.key, &value, capacity));
        }
        cache.layers = layers;
        cache.length = len;
        cache.next_position = len;
        cache.write_index = len % capacity;
        rms_norm(&x)
    }
    fn prefill_serialized_token(
        &self,
        token: &Tensor,
        cache: &mut BarKvCache,
        config: Option<&RecirculationConfig>,
    ) -> Result<Tensor> {
        debug_assert_eq!(token.size()[1], 1);
        let position = Tensor::from_slice(&[0i64]).to_device(token.device());
        let phases = PopePhases::new(&position, token.device(), POPE_FREQUENCY_BASE);
        let mut x = token.shallow_clone();
        let mut destination = None;
        let mut source = None;
        let mut layers = Vec::with_capacity(self.layers.len());
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let (query, key, value) = layer.qkv(&rms_norm(&x));
            let kind = attention_kind(&x);
            let polar = pope_expand_qk(&query, &key, &phases, &layer.pope_theta_bias, kind);
            let value = value.to_kind(kind).contiguous();
            let attention = strict_pope_prefill(&polar, &value);
            x = layer.attention_residual(&x, &attention, token);
            x = layer.feed_forward(&x);
            if config.is_some_and(|config| layer_index == config.destination_layer) {
                destination = Some(x.shallow_clone());
            }
            if config.is_some_and(|config| layer_index == config.source_layer) {
                source = Some(x.shallow_clone());
            }
            layers.push(BarLayerKv::prefilled(&polar.key, &value, 1));
        }
        let belief = rms_norm(&x);
        cache.layers = layers;
        if let Some(config) = config.filter(|config| config.alpha_at(0) > 0.0) {
            self.overwrite_recirculated_cache(
                token,
                cache,
                config,
                0,
                0,
                0,
                destination.expect("validated destination was traversed"),
                source.expect("validated source was traversed"),
            );
        }
        cache.length = 1;
        cache.next_position = 1;
        cache.write_index = 0;
        Ok(belief)
    }

    #[allow(clippy::too_many_arguments)]
    fn overwrite_recirculated_cache(
        &self,
        token: &Tensor,
        cache: &mut BarKvCache,
        config: &RecirculationConfig,
        absolute_position: i64,
        write_index: i64,
        previous_length: i64,
        destination: Tensor,
        source: Tensor,
    ) {
        let position = Tensor::from_slice(&[absolute_position]).to_device(token.device());
        let phases = PopePhases::new(&position, token.device(), POPE_FREQUENCY_BASE);
        let mut x =
            Self::recirculate_residual(&destination, &source, config, absolute_position as usize);
        for (layer, layer_cache) in self
            .layers
            .iter()
            .zip(cache.layers.iter_mut())
            .skip(config.destination_layer + 1)
        {
            let (query, key, value) = layer.qkv(&rms_norm(&x));
            let kind = attention_kind(&x);
            let polar = pope_expand_qk(&query, &key, &phases, &layer.pope_theta_bias, kind);
            let value = value.to_kind(kind).contiguous();
            layer_cache.key.narrow(1, write_index, 1).copy_(&polar.key);
            layer_cache.value.narrow(1, write_index, 1).copy_(&value);
            let (active_key, active_value) = layer_cache.active_after_write(previous_length);
            let attention = strict_pope_decode(&polar.query, &active_key, &active_value);
            x = layer.attention_residual(&x, &attention, token);
            x = layer.feed_forward(&x);
        }
    }

    fn decode_serialized_token(
        &self,
        token: &Tensor,
        cache: &mut BarKvCache,
        config: Option<&RecirculationConfig>,
    ) -> Result<Tensor> {
        assert_eq!(
            cache.layers.len(),
            self.layers.len(),
            "bar KV cache has an incompatible layer count"
        );
        cache.ensure_append_capacity();
        let absolute_position = cache.next_position;
        let position = Tensor::from_slice(&[absolute_position]).to_device(token.device());
        let phases = PopePhases::new(&position, token.device(), POPE_FREQUENCY_BASE);
        let write_index = cache.write_index;
        let previous_length = cache.length;
        let mut x = token.shallow_clone();
        let mut destination = None;
        let mut source = None;
        for (layer_index, (layer, layer_cache)) in
            self.layers.iter().zip(cache.layers.iter_mut()).enumerate()
        {
            let (query, key, value) = layer.qkv(&rms_norm(&x));
            let kind = attention_kind(&x);
            let polar = pope_expand_qk(&query, &key, &phases, &layer.pope_theta_bias, kind);
            let value = value.to_kind(kind).contiguous();
            layer_cache.key.narrow(1, write_index, 1).copy_(&polar.key);
            layer_cache.value.narrow(1, write_index, 1).copy_(&value);
            let (active_key, active_value) = layer_cache.active_after_write(previous_length);
            let attention = strict_pope_decode(&polar.query, &active_key, &active_value);
            x = layer.attention_residual(&x, &attention, token);
            x = layer.feed_forward(&x);
            if config.is_some_and(|config| layer_index == config.destination_layer) {
                destination = Some(x.shallow_clone());
            }
            if config.is_some_and(|config| layer_index == config.source_layer) {
                source = Some(x.shallow_clone());
            }
        }
        let belief = rms_norm(&x);
        if let Some(config) =
            config.filter(|config| config.alpha_at(absolute_position as usize) > 0.0)
        {
            self.overwrite_recirculated_cache(
                token,
                cache,
                config,
                absolute_position,
                write_index,
                previous_length,
                destination.expect("validated destination was traversed"),
                source.expect("validated source was traversed"),
            );
        }
        cache.finish_append();
        Ok(belief)
    }

    fn decode(&self, token: &Tensor, cache: &mut BarKvCache) -> Tensor {
        assert_eq!(
            cache.layers.len(),
            self.layers.len(),
            "bar KV cache has an incompatible layer count"
        );
        cache.ensure_append_capacity();
        let position = Tensor::from_slice(&[cache.next_position]).to_device(token.device());
        let phases = PopePhases::new(&position, token.device(), POPE_FREQUENCY_BASE);
        let write_index = cache.write_index;
        let previous_length = cache.length;
        let mut x = token.shallow_clone();
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            let (query, key, value) = layer.qkv(&rms_norm(&x));
            let kind = attention_kind(&x);
            let polar = pope_expand_qk(&query, &key, &phases, &layer.pope_theta_bias, kind);
            let value = value.to_kind(kind).contiguous();
            layer_cache.key.narrow(1, write_index, 1).copy_(&polar.key);
            layer_cache.value.narrow(1, write_index, 1).copy_(&value);
            let (active_key, active_value) = layer_cache.active_after_write(previous_length);
            let attention = strict_pope_decode(&polar.query, &active_key, &active_value);
            x = layer.attention_residual(&x, &attention, token);
            x = layer.feed_forward(&x);
        }
        cache.finish_append();
        rms_norm(&x)
    }
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct BarLayerKv {
    /// `[rows, capacity, BAR_HEADS, POPE_QK_DIM]`
    key: Tensor,
    /// `[rows, capacity, BAR_HEADS, POPE_DIM]`
    value: Tensor,
}

impl BarLayerKv {
    fn prefilled(key: &Tensor, value: &Tensor, capacity: i64) -> Self {
        let len = key.size()[1];
        let storage = |source: &Tensor| {
            let size = source.size();
            let buffer = Tensor::zeros(
                [size[0], capacity, size[2], size[3]],
                (source.kind(), source.device()),
            );
            buffer.narrow(1, 0, len).copy_(source);
            buffer
        };
        Self {
            key: storage(key),
            value: storage(value),
        }
    }

    fn fork(&self) -> Self {
        Self {
            key: self.key.copy(),
            value: self.value.copy(),
        }
    }

    fn repeat_batch(&self, factor: i64) -> Self {
        Self {
            key: self.key.repeat_interleave_self_int(factor, 0, None),
            value: self.value.repeat_interleave_self_int(factor, 0, None),
        }
    }

    fn grow(&mut self, capacity: i64, length: i64) {
        let grown = |source: &Tensor| {
            let size = source.size();
            let buffer = Tensor::zeros(
                [size[0], capacity, size[2], size[3]],
                (source.kind(), source.device()),
            );
            buffer
                .narrow(1, 0, length)
                .copy_(&source.narrow(1, 0, length));
            buffer
        };
        self.key = grown(&self.key);
        self.value = grown(&self.value);
    }

    fn active_after_write(&self, previous_length: i64) -> (Tensor, Tensor) {
        let capacity = self.key.size()[1];
        if previous_length < capacity {
            let length = previous_length + 1;
            return (
                self.key.narrow(1, 0, length),
                self.value.narrow(1, 0, length),
            );
        }
        (self.key.shallow_clone(), self.value.shallow_clone())
    }
}

/// Circular KV cache holding PoPE-expanded keys and raw values.
///
/// Positions are absolute and already baked into the key phases, so the physical
/// ring order is irrelevant to the Q=1 decode kernel and eviction is a plain
/// modular overwrite. Storage starts at the next power of two above the prefill
/// length and doubles up to the cache window, which is also the sliding-window
/// span the cached path attends over.
#[derive(Debug)]
pub struct BarKvCache {
    layers: Vec<BarLayerKv>,
    next_position: i64,
    max_tokens: i64,
    length: i64,
    write_index: i64,
}

impl BarKvCache {
    /// An empty cache retaining at most `max_tokens` bars. The first
    /// [`BarTrunk::forward_cached`] allocates its storage.
    pub fn new(max_tokens: i64) -> Self {
        assert!(max_tokens > 0, "bar KV-cache window must be positive");
        Self {
            layers: Vec::new(),
            next_position: 0,
            max_tokens,
            length: 0,
            write_index: 0,
        }
    }

    pub fn contract() -> &'static str {
        BAR_CACHE_CONTRACT
    }

    pub fn cached_bars(&self) -> i64 {
        self.length
    }

    pub fn next_position(&self) -> i64 {
        self.next_position
    }

    pub fn max_tokens(&self) -> i64 {
        self.max_tokens
    }

    /// Deep copy, so a speculative continuation never contaminates real history.
    pub fn fork(&self) -> Self {
        Self {
            layers: self.layers.iter().map(BarLayerKv::fork).collect(),
            next_position: self.next_position,
            max_tokens: self.max_tokens,
            length: self.length,
            write_index: self.write_index,
        }
    }

    /// Interleave every row `factor` times, so one prefill over `B` histories
    /// serves `B * factor` independent continuations. Row order matches
    /// `Tensor::repeat_interleave` on dim 0.
    pub fn repeat_batch(&self, factor: i64) -> Self {
        assert!(factor > 0, "cache batch factor must be positive");
        Self {
            layers: self
                .layers
                .iter()
                .map(|layer| layer.repeat_batch(factor))
                .collect(),
            next_position: self.next_position,
            max_tokens: self.max_tokens,
            length: self.length,
            write_index: self.write_index,
        }
    }

    fn capacity(&self) -> i64 {
        self.layers
            .first()
            .map(|layer| layer.key.size()[1])
            .unwrap_or(0)
    }

    fn ensure_append_capacity(&mut self) {
        let capacity = self.capacity();
        if self.length < capacity || capacity == self.max_tokens {
            return;
        }
        let grown = (capacity.max(1) * 2).min(self.max_tokens);
        for layer in &mut self.layers {
            layer.grow(grown, self.length);
        }
        self.write_index = self.length;
    }

    fn finish_append(&mut self) {
        let capacity = self.capacity();
        if self.length < self.max_tokens {
            self.length += 1;
        }
        self.write_index = (self.write_index + 1) % capacity;
        self.next_position += 1;
    }
}

// ---------------------------------------------------------------------------
// Dynamics
// ---------------------------------------------------------------------------

/// NextLat one-step latent predictor:
/// `RMSNorm(h + MLP(RMSNorm([h ; next_token])))`.
///
/// `next_token` is the exact output of [`BarTrunk::token_embedding`]: bins, raw
/// DOF and exogenous time with same-instant market masked. Dynamics deliberately
/// owns no parallel input encoder, so exact and cheap rollouts cannot drift into
/// two incompatible definitions of a bar token.
///
/// The closing RMSNorm is not cosmetic. Every belief this predictor is trained
/// against, and every belief [`BarEmissionHead`] has ever been fitted on, is the
/// output of the gain-free `rms_norm` that closes `BarTrunk::run`, so the unit
/// shell is the head's input domain. `fc3_w` is zero-init and `h` already has unit
/// RMS, so an untrained predictor is exactly the identity.
pub struct BarDynamics {
    /// `[hidden, 2 * dim]`
    fc1_w: Tensor,
    /// `[hidden, hidden]`
    fc2_w: Tensor,
    /// `[dim, hidden]`, zero-init, so an untrained predictor is the identity.
    fc3_w: Tensor,
    dim: i64,
}

impl BarDynamics {
    pub fn new(vs: &nn::Path, dim: i64) -> Self {
        assert!(dim > 0, "bar dynamics needs a positive latent dim");
        let joined = 2 * dim;
        Self {
            fc1_w: vs.var(
                "bar_dyn_fc1_w",
                &[BAR_DYNAMICS_HIDDEN, joined],
                uniform_init(joined),
            ),
            fc2_w: vs.var(
                "bar_dyn_fc2_w",
                &[BAR_DYNAMICS_HIDDEN, BAR_DYNAMICS_HIDDEN],
                uniform_init(BAR_DYNAMICS_HIDDEN),
            ),
            fc3_w: vs.var(
                "bar_dyn_fc3_w",
                &[dim, BAR_DYNAMICS_HIDDEN],
                Init::Const(0.0),
            ),
            dim,
        }
    }

    pub fn muon_name_substrings() -> &'static [&'static str] {
        &BAR_DYNAMICS_MUON_SUBSTRINGS
    }

    pub fn dim(&self) -> i64 {
        self.dim
    }

    /// `h [..., dim]` and the matching shared trunk `next_token [..., dim]` ->
    /// predicted `h' [..., dim]` on the unit-RMS shell, shaped exactly like `h`.
    pub fn step(&self, h: &Tensor, next_token: &Tensor) -> Tensor {
        let shape = h.size();
        assert_eq!(
            shape.last().copied(),
            Some(self.dim),
            "dynamics latent width mismatch"
        );
        assert_eq!(
            next_token.size(),
            shape,
            "dynamics shared token must match the latent shape"
        );
        let rows = h.numel() as i64 / self.dim;
        let device = self.fc1_w.device();
        let flat = h
            .to_device(device)
            .to_kind(Kind::Float)
            .reshape([rows, self.dim]);
        let token = next_token
            .to_device(device)
            .to_kind(Kind::Float)
            .reshape([rows, self.dim]);
        let joined = rms_norm(&Tensor::cat(&[&flat, &token], -1));
        let residual = joined
            .linear(&self.fc1_w, None::<Tensor>)
            .gelu("none")
            .linear(&self.fc2_w, None::<Tensor>)
            .gelu("none")
            .linear(&self.fc3_w, None::<Tensor>);
        rms_norm(&(flat + residual)).reshape(shape.as_slice())
    }
}

// ---------------------------------------------------------------------------
// Module bundle
// ---------------------------------------------------------------------------

/// How an imagined rollout advances its belief.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum RolloutMode {
    /// Append the bar to the KV cache and run the trunk. Exact, `BAR_LAYERS`
    /// blocks per step, no drift.
    #[default]
    Exact,
    /// Advance with [`BarDynamics`]: three matmuls per step and no cache
    /// traffic, but only as accurate as the NextLat objective made it. Its gap
    /// against `Exact` on the same bar sequence is the readout for `lambda_dyn`
    /// and `lambda_kl`.
    Dynamics,
}

impl RolloutMode {
    pub fn as_str(self) -> &'static str {
        match self {
            RolloutMode::Exact => "exact",
            RolloutMode::Dynamics => "dynamics",
        }
    }
}

/// One imagined rollout. `dof[.., i]` is the bar sampled at step `i` and
/// `beliefs[.., i]` is the belief it was drawn from, so the two are aligned and
/// a consumer can summarize the sample axis of either.
#[derive(Debug)]
pub struct BarRollout {
    /// `[B, samples, steps, BAR_DOF]`
    pub dof: Tensor,
    /// `[B, samples, steps, BAR_MODEL_DIM]`
    pub beliefs: Tensor,
}

/// The decision-time belief plus the rollout imagined from it.
#[derive(Debug)]
pub struct BarForecast {
    /// `[B, BAR_MODEL_DIM]`, the belief after the real history.
    pub belief: Tensor,
    pub rollout: BarRollout,
}

/// Trunk, emission head and dynamics under one `VarStore` path.
///
/// Training and inference both go through this constructor, so checkpoint tensor
/// names cannot drift between the two.
pub struct BarModules {
    pub trunk: BarTrunk,
    pub head: BarEmissionHead,
    pub dynamics: BarDynamics,
}

impl BarModules {
    pub fn new(vs: &nn::Path) -> Self {
        Self {
            trunk: BarTrunk::new(vs),
            head: BarEmissionHead::new(vs, BAR_MODEL_DIM),
            dynamics: BarDynamics::new(vs, BAR_MODEL_DIM),
        }
    }

    /// Teacher-forced belief trajectory over `future_dof`.
    ///
    /// Returns `[B, S, D]` where entry `i` is the belief that predicts
    /// `future_dof[:, i]`: entry 0 is the belief after the whole history, and
    /// each later entry is the previous one advanced by the corresponding
    /// observed bar. Running both [`RolloutMode`]s over the same `future_dof`
    /// leaves the belief-advance mechanism as the only difference, which is what
    /// makes the NLL gap between them a measurement of NextLat drift rather than
    /// of sampling noise.
    pub fn rollout_beliefs(
        &self,
        supports: &BarSupportSet,
        history_dof: &Tensor,
        history_time_ids: &Tensor,
        future_dof: &Tensor,
        future_time_ids: &Tensor,
        mode: RolloutMode,
    ) -> Tensor {
        assert_eq!(
            history_dof.dim(),
            3,
            "history must be [batch, len, BAR_DOF]"
        );
        assert_eq!(
            future_dof.dim(),
            3,
            "future must be [batch, steps, BAR_DOF]"
        );
        let history_len = history_dof.size()[1];
        let steps = future_dof.size()[1];
        assert!(history_len > 0, "belief rollout needs a history");
        assert!(steps > 0, "belief rollout needs at least one future bar");
        assert_eq!(
            future_time_ids.size()[1],
            steps,
            "future time ids must cover every future bar"
        );
        tch::no_grad(|| {
            let mut cache = BarKvCache::new(BAR_MAX_CONTEXT);
            let beliefs = self.trunk.forward_cached(
                history_dof,
                &supports.bin_ids(history_dof, history_time_ids),
                history_time_ids,
                &mut cache,
            );
            let mut h = beliefs.narrow(1, history_len - 1, 1);
            let mut out = Vec::with_capacity(steps as usize);
            let bin_table = self.trunk.bin_table();
            for step in 0..steps {
                out.push(h.shallow_clone());
                let dof = future_dof.narrow(1, step, 1);
                let time_ids = future_time_ids.narrow(1, step, 1);
                h = match mode {
                    RolloutMode::Exact => self.trunk.forward_cached(
                        &dof,
                        &supports.bin_ids(&dof, &time_ids),
                        &time_ids,
                        &mut cache,
                    ),
                    RolloutMode::Dynamics => {
                        let bins = supports.bin_ids(&dof, &time_ids);
                        let token =
                            self.trunk
                                .token_embedding_with_table(&bin_table, &dof, &bins, &time_ids);
                        self.dynamics.step(&h, &token)
                    }
                };
            }
            Tensor::cat(&out, 1)
        })
    }
}

// ---------------------------------------------------------------------------
// Frozen inference bundle
// ---------------------------------------------------------------------------

pub struct BarWorldModel {
    var_store: nn::VarStore,
    modules: BarModules,
    supports: BarSupportSet,
    metadata: BarWorldModelMetadata,
}

impl BarWorldModel {
    /// Load a frozen world model. Every resolution's supports sidecar is resolved
    /// from the weights path and hash-checked against the metadata, because the
    /// supports define what every predicted bin means and a checkpoint trained
    /// across resolutions is only meaningful beside all of them.
    pub fn load(weights: &Path, metadata: &Path, device: Device) -> Result<Self> {
        let metadata = BarWorldModelMetadata::load(metadata)?;
        metadata.validate_checkpoint(weights)?;
        metadata.validate_supports(weights)?;
        let mut loaded = Vec::with_capacity(metadata.supports_sha256.len());
        for resolution in metadata.resolutions() {
            let path = world_model_supports_path(weights, resolution);
            loaded.push((
                resolution,
                BarSupports::load(&path)
                    .with_context(|| format!("loading {}", path.display()))?
                    .to_device(device),
            ));
        }
        let supports = BarSupportSet::new(loaded)?;
        if let Some(training) = metadata.training.as_ref() {
            for resolution in metadata.resolutions() {
                let support = supports
                    .get(resolution)
                    .expect("the support set was built from every metadata resolution");
                ensure!(
                    support.dof_scaling() == training.dof_scaling,
                    "checkpoint training provenance records {} targets but its authenticated \
                     {resolution}s supports tile {} targets",
                    training.dof_scaling,
                    support.dof_scaling(),
                );
            }
        }

        let mut var_store = nn::VarStore::new(device);
        let modules = BarModules::new(&var_store.root());
        load_var_store_partial(&mut var_store, weights)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .require_complete()
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .with_context(|| {
                format!(
                    "world-model checkpoint {} is missing required bar-model tensors",
                    weights.display()
                )
            })?;
        var_store.freeze();

        Ok(Self {
            var_store,
            modules,
            supports,
            metadata,
        })
    }

    pub fn metadata(&self) -> &BarWorldModelMetadata {
        &self.metadata
    }

    pub fn lineage_sha256(&self) -> &str {
        &self.metadata.lineage_sha256
    }

    pub fn device(&self) -> Device {
        self.var_store.device()
    }

    /// Every resolution's supports, device-resident.
    pub fn supports(&self) -> &BarSupportSet {
        &self.supports
    }

    /// The support of the deployment resolution: the one held-out selection and
    /// the planner run against.
    pub fn deployment_supports(&self) -> &BarSupports {
        self.supports
            .get(self.metadata.res_secs)
            .expect("metadata validation guarantees the deployment support")
    }

    pub fn supports_for(&self, res_secs: u32) -> Option<&BarSupports> {
        self.supports.get(res_secs)
    }

    pub fn modules(&self) -> &BarModules {
        &self.modules
    }

    pub fn trunk(&self) -> &BarTrunk {
        &self.modules.trunk
    }

    pub fn head(&self) -> &BarEmissionHead {
        &self.modules.head
    }

    pub fn dynamics(&self) -> &BarDynamics {
        &self.modules.dynamics
    }

    pub fn all_parameters_frozen(&self) -> bool {
        self.var_store
            .variables()
            .values()
            .all(|tensor| !tensor.requires_grad())
    }

    /// Beliefs over a bar history, `[B, T, D]`.
    pub fn beliefs(&self, history_dof: &Tensor, history_time_ids: &Tensor) -> Tensor {
        let history = history_dof.to_device(self.device()).to_kind(Kind::Float);
        let history_time_ids = history_time_ids.to_device(self.device());
        let ids = self.supports.bin_ids(&history, &history_time_ids);
        self.modules
            .trunk
            .forward(&history, &ids, &history_time_ids, 0, false)
    }

    /// Exact ancestral rollout of the sampled bars, `[B, samples, steps, BAR_DOF]`,
    /// where `steps == future_time_ids.size(1)`.
    pub fn rollout(
        &self,
        history_dof: &Tensor,
        history_time_ids: &Tensor,
        future_time_ids: &Tensor,
        samples: usize,
        temperature: f64,
    ) -> Tensor {
        self.rollout_with(
            history_dof,
            history_time_ids,
            future_time_ids,
            samples,
            temperature,
            RolloutMode::Exact,
        )
        .dof
    }

    pub fn rollout_with(
        &self,
        history_dof: &Tensor,
        history_time_ids: &Tensor,
        future_time_ids: &Tensor,
        samples: usize,
        temperature: f64,
        mode: RolloutMode,
    ) -> BarRollout {
        self.forecast(
            history_dof,
            history_time_ids,
            future_time_ids,
            samples,
            temperature,
            mode,
        )
        .rollout
    }

    /// The decision-time belief over the real history together with the imagined
    /// rollout that starts from it, sharing a single prefill.
    pub fn forecast(
        &self,
        history_dof: &Tensor,
        history_time_ids: &Tensor,
        future_time_ids: &Tensor,
        samples: usize,
        temperature: f64,
        mode: RolloutMode,
    ) -> BarForecast {
        let session = self.start_session(history_dof, history_time_ids);
        let rollout = self.imagine(&session, future_time_ids, samples, temperature, mode);
        BarForecast {
            belief: session.belief,
            rollout,
        }
    }

    /// Prefill the KV cache over a real bar history, `[B, T, BAR_DOF]` with its
    /// `[B, T, BAR_TIME_FEATURES]` calendar.
    ///
    /// A session is the decision-time state: real bars are appended to it with
    /// [`Self::advance`], and [`Self::imagine`] forks it so a sampled
    /// continuation can never contaminate the real history.
    pub fn start_session(
        &self,
        history_dof: &Tensor,
        history_time_ids: &Tensor,
    ) -> BarWorldModelSession {
        assert_eq!(
            history_dof.dim(),
            3,
            "session history must be [batch, len, BAR_DOF]"
        );
        let history = history_dof.to_device(self.device()).to_kind(Kind::Float);
        let history_len = history.size()[1];
        assert!(history_len > 0, "a session needs a non-empty history");
        let history_time_ids = history_time_ids.to_device(self.device());
        let mut cache = BarKvCache::new(BAR_MAX_CONTEXT);
        let prefill = self.modules.trunk.forward_cached(
            &history,
            &self.supports.bin_ids(&history, &history_time_ids),
            &history_time_ids,
            &mut cache,
        );
        let current_time_ids = history_time_ids.narrow(1, history_len - 1, 1);
        BarWorldModelSession {
            belief: prefill.narrow(1, history_len - 1, 1).squeeze_dim(1),
            batch: history.size()[0],
            cache,
            current_time_ids,
            lineage_sha256: self.metadata.lineage_sha256.clone(),
        }
    }

    /// Append one realized bar, `[B, 1, ..]` or `[B, ..]`.
    pub fn advance(
        &self,
        session: &mut BarWorldModelSession,
        next_dof: &Tensor,
        next_time_ids: &Tensor,
    ) -> Result<()> {
        if session.lineage_sha256 != self.metadata.lineage_sha256 {
            bail!("world-model session has an incompatible inference lineage");
        }
        let next = as_single_bar(
            next_dof,
            self.device(),
            session.batch,
            BAR_DOF as i64,
            "bar",
        )?
        .to_kind(Kind::Float);
        let time_ids = as_single_bar(
            next_time_ids,
            self.device(),
            session.batch,
            BAR_TIME_FEATURES as i64,
            "bar calendar",
        )?;
        session.belief = self
            .modules
            .trunk
            .forward_cached(
                &next,
                &self.supports.bin_ids(&next, &time_ids),
                &time_ids,
                &mut session.cache,
            )
            .squeeze_dim(1);
        session.current_time_ids = time_ids;
        Ok(())
    }

    /// Ancestral rollout from a session: sample a bar from the emission head,
    /// advance the belief per `mode`, repeat. `steps` is `future_time_ids.size(1)`.
    ///
    /// The session's cache is forked and its rows interleaved, so row
    /// `b * samples + s` is sample `s` of batch element `b` and the returned
    /// tensors are `[B, samples, steps, ...]`. The fork replicates the cached
    /// history once per sample, so a caller with a long context should draw its
    /// samples in chunks rather than in one call.
    pub fn imagine(
        &self,
        session: &BarWorldModelSession,
        future_time_ids: &Tensor,
        samples: usize,
        temperature: f64,
        mode: RolloutMode,
    ) -> BarRollout {
        assert!(samples > 0, "rollout needs at least one sample");
        let batch = session.batch;
        let samples = samples as i64;
        assert_eq!(
            future_time_ids.dim(),
            3,
            "future time ids must be [batch, steps, BAR_TIME_FEATURES]"
        );
        let steps = future_time_ids.size()[1];
        assert!(steps > 0, "rollout needs at least one step");
        assert_eq!(
            future_time_ids.size(),
            [batch, steps, BAR_TIME_FEATURES as i64],
            "future time ids must match the session batch"
        );
        assert_eq!(
            session.lineage_sha256, self.metadata.lineage_sha256,
            "world-model session has an incompatible inference lineage"
        );
        tch::no_grad(|| {
            // One row per (batch element, sample), interleaved so the sample axis
            // is the fastest-varying one and the final view needs no permute.
            let future_time_ids = future_time_ids.to_device(self.device());
            let mut h = session.belief.unsqueeze(1);
            let mut clock = future_time_ids.shallow_clone();
            let mut current_time_ids = session.current_time_ids.shallow_clone();
            // `Dynamics` never touches the cache, and `repeat_batch` already
            // materializes fresh storage, so neither the fork nor the replication
            // is paid unless the mode actually reads a cache.
            let mut cache = match (mode, samples > 1) {
                (RolloutMode::Dynamics, _) => BarKvCache::new(session.cache.max_tokens()),
                (RolloutMode::Exact, true) => session.cache.repeat_batch(samples),
                (RolloutMode::Exact, false) => session.cache.fork(),
            };
            if samples > 1 {
                h = h.repeat_interleave_self_int(samples, 0, None);
                clock = clock.repeat_interleave_self_int(samples, 0, None);
                current_time_ids = current_time_ids.repeat_interleave_self_int(samples, 0, None);
            }

            let mut sampled = Vec::with_capacity(steps as usize);
            let mut beliefs = Vec::with_capacity(steps as usize);
            let bin_table = self.modules.trunk.bin_table();
            for step in 0..steps {
                // The belief recorded for a step is the one the step's bar was
                // drawn from, so `beliefs[.., i]` conditions `dof[.., i]`.
                beliefs.push(h.squeeze_dim(1));
                let time_ids = clock.narrow(1, step, 1);
                // The first imagined bar follows a real observation, so its market channels
                // come from the session's final row. Once the rollout enters imagined history,
                // no contemporaneous market proxy is observable.
                let current = if step == 0 {
                    current_time_ids.shallow_clone()
                } else {
                    time_ids_without_market(&time_ids)
                };
                let conditioning = self
                    .modules
                    .trunk
                    .forecast_conditioning(&time_ids, &current);
                let dof = self.supports.sample(
                    &self.modules.head,
                    &h,
                    &conditioning,
                    &time_ids,
                    temperature,
                );
                sampled.push(dof.squeeze_dim(1));
                let bins = self.supports.bin_ids(&dof, &time_ids);
                h = match mode {
                    RolloutMode::Exact => self
                        .modules
                        .trunk
                        .forward_cached(&dof, &bins, &time_ids, &mut cache),
                    RolloutMode::Dynamics => {
                        let token = self.modules.trunk.token_embedding_with_table(
                            &bin_table,
                            &dof,
                            &bins,
                            &time_ids,
                        );
                        self.modules.dynamics.step(&h, &token)
                    }
                };
            }
            BarRollout {
                dof: Tensor::stack(&sampled, 1).view([batch, samples, steps, BAR_DOF as i64]),
                beliefs: Tensor::stack(&beliefs, 1).view([batch, samples, steps, BAR_MODEL_DIM]),
            }
        })
    }
}

/// Stateful decision-time inference context over a real bar history.
pub struct BarWorldModelSession {
    cache: BarKvCache,
    belief: Tensor,
    /// Final observed clock/market row, `[B, 1, BAR_TIME_FEATURES]`.
    current_time_ids: Tensor,
    batch: i64,
    lineage_sha256: String,
}

impl BarWorldModelSession {
    /// `[B, BAR_MODEL_DIM]`, the belief after every bar seen so far.
    pub fn belief(&self) -> &Tensor {
        &self.belief
    }

    pub fn batch_size(&self) -> i64 {
        self.batch
    }

    pub fn cached_bars(&self) -> i64 {
        self.cache.cached_bars()
    }

    pub fn lineage_sha256(&self) -> &str {
        &self.lineage_sha256
    }

    pub fn fork(&self) -> Self {
        Self {
            cache: self.cache.fork(),
            belief: self.belief.shallow_clone(),
            current_time_ids: self.current_time_ids.shallow_clone(),
            batch: self.batch,
            lineage_sha256: self.lineage_sha256.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/// `uniform(+-sqrt(3) * 0.5 * fan_in^-0.5)`, the modded-nanogpt 2D init.
fn uniform_init(fan_in: i64) -> Init {
    let bound = 3f64.sqrt() * 0.5 / (fan_in as f64).sqrt();
    Init::Uniform {
        lo: -bound,
        up: bound,
    }
}

/// Row at which each calendar channel's block starts in the fused embedding bank.
fn time_block_offsets(device: Device) -> Tensor {
    let mut offsets = [0i64; BAR_TIME_FEATURES];
    let mut base = 0;
    for (slot, cardinality) in BAR_TIME_CARDINALITY.iter().enumerate() {
        offsets[slot] = base;
        base += cardinality;
    }
    Tensor::from_slice(&offsets)
        .to_device(device)
        .view([1, 1, BAR_TIME_FEATURES as i64])
}

/// Normalize a `[B, width]` or `[B, 1, width]` single-bar tensor to `[B, 1, width]`.
fn as_single_bar(
    value: &Tensor,
    device: Device,
    batch: i64,
    width: i64,
    what: &str,
) -> Result<Tensor> {
    let value = value.to_device(device);
    let value = match value.dim() {
        2 => value.unsqueeze(1),
        3 => value,
        other => bail!("realized {what} must be rank 2 or 3, got rank {other}"),
    };
    if value.size() != [batch, 1, width] {
        bail!(
            "realized {what} must have shape [{batch}, 1, {width}], got {:?}",
            value.size()
        );
    }
    Ok(value)
}

/// RMSNorm over the last axis, with no learnable gain.
fn rms_norm(value: &Tensor) -> Tensor {
    let width = *value.size().last().expect("rms_norm needs a ranked tensor");
    value.rms_norm([width], None::<&Tensor>, Some(BAR_NORM_EPS))
}

/// QK-norm over `head_dim`, applied before PoPE expands the phases.
fn qk_norm(value: &Tensor) -> Tensor {
    value.rms_norm([BAR_HEAD_DIM], None::<&Tensor>, Some(BAR_NORM_EPS))
}

fn attention_kind(value: &Tensor) -> Kind {
    if value.device().is_cuda() {
        Kind::BFloat16
    } else {
        value.kind()
    }
}

/// Longest query block a windowed attention pass materializes scores for. Bounds
/// the transient to `block * (block + window)` per head instead of `len^2`.
const WINDOW_ATTENTION_BLOCK: i64 = 256;

/// Exact sliding-window causal attention.
///
/// NOT because FA4 cannot window. `flash_attn.cute.flash_attn_func` DOES accept a
/// `window_size` argument — verified by introspection against `.venv-fa4`'s own interpreter,
/// alongside `learnable_sink` and `softcap`, neither of which this model uses — and
/// [`crate::torch::fa4`] already forwards keyword arguments through a `PyDict`, so routing a
/// band to the kernel is one extra kwarg. An earlier version of this comment claimed the
/// opposite and cost an audit a wrong cost estimate for a per-layer receptive-field schedule.
///
/// What is true is that this fp32 fallback is what the CURRENT code reaches for when
/// `0 < window < len`, that the fused SDPA path is unavailable — [`crate::torch::cuda::cfg::configure_cuda`]
/// leaves only the flash backend enabled, which rejects an arbitrary `attn_mask` — and that
/// training never takes this path, because every trunk call passes `window <= 0` and lands on
/// the full-causal FA4 prefill. Every query row admits at least its own key
/// (`delta == 0 < window`), so no row is fully masked and the softmax cannot produce NaN.
fn windowed_attention(query: &Tensor, key: &Tensor, value: &Tensor, window: i64) -> Tensor {
    let len = query.size()[1];
    let device = query.device();
    let block = WINDOW_ATTENTION_BLOCK.min(len);
    let mut blocks = Vec::with_capacity(((len + block - 1) / block) as usize);
    let mut start = 0;
    while start < len {
        let rows = block.min(len - start);
        // Widest key span any query in this block can reach.
        let first_key = (start - window + 1).max(0);
        let keys = start + rows - first_key;
        let q = query
            .narrow(1, start, rows)
            .transpose(1, 2)
            .to_kind(Kind::Float);
        let k = key
            .narrow(1, first_key, keys)
            .transpose(1, 2)
            .to_kind(Kind::Float);
        let v = value
            .narrow(1, first_key, keys)
            .transpose(1, 2)
            .to_kind(Kind::Float);
        let query_pos = Tensor::arange(rows, (Kind::Int64, device)) + start;
        let key_pos = Tensor::arange(keys, (Kind::Int64, device)) + first_key;
        let delta = query_pos.view([rows, 1]) - key_pos.view([1, keys]);
        let blocked = delta
            .ge(0)
            .logical_and(&delta.lt(window))
            .logical_not()
            .view([1, 1, rows, keys]);
        let scores = (q.matmul(&k.transpose(-2, -1)) * POPE_ATTENTION_SCALE)
            .masked_fill(&blocked, f64::NEG_INFINITY);
        blocks.push(scores.softmax(-1, Kind::Float).matmul(&v).transpose(1, 2));
        start += rows;
    }
    Tensor::cat(&blocks, 1).to_kind(value.kind())
}

/// ATen's autocast enable flag says nothing about its dtype, and the CUDA default
/// on this toolchain is fp16. Running the PoPE kernels in fp16 is silently wrong
/// rather than loud: there is no gradient scaler in this crate, so small gradients
/// flush to zero, and `half + bfloat16` promotes to fp32, upcasting the residual
/// stream every layer. [`crate::torch::cuda::cfg::configure_cuda`] pins bf16; this
/// refuses to run if something ever unpins it.
fn assert_bfloat16_autocast() {
    assert!(
        unsafe { torch_sys::at_autocast_is_bfloat16() } != 0,
        "CUDA autocast is not bf16; call configure_cuda() before any world-model attention"
    );
}

fn strict_pope_prefill(qk: &PolarQk, value_bshd: &Tensor) -> Tensor {
    if value_bshd.device().is_cuda() {
        assert_bfloat16_autocast();
        return autocast(true, || pope_flash_attention_prefill(qk, value_bshd))
            .unwrap_or_else(|error| panic!("strict FA4 PoPE prefill failed: {error:#}"));
    }
    #[cfg(test)]
    return pope_attention_reference(qk, value_bshd, true);
    #[cfg(not(test))]
    panic!("bar world model PoPE prefill requires CUDA with the strict FA4 bridge");
}

fn strict_pope_decode(query: &Tensor, key: &Tensor, value: &Tensor) -> Tensor {
    if value.device().is_cuda() {
        assert_bfloat16_autocast();
        return autocast(true, || pope_flash_attention_decode_q1(query, key, value))
            .unwrap_or_else(|error| panic!("strict FA4 PoPE decode failed: {error:#}"));
    }
    #[cfg(test)]
    return pope_attention_reference(
        &PolarQk {
            query: query.shallow_clone(),
            key: key.shallow_clone(),
        },
        value,
        false,
    );
    #[cfg(not(test))]
    panic!("bar world model PoPE decode requires CUDA with the strict FA4 bridge");
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, fs, path::PathBuf};

    use super::*;
    use crate::torch::{
        bar_dist::{decode_dof, BarDof, BarScoring, BAR_EMISSION_ADAMW_NAME_SUBSTRINGS},
        dataset::{
            BAR_TIME_MARKET, TIME_DAY_EDGE, TIME_ELAPSED, TIME_MINUTE, TIME_SESSION, TIME_WEEKDAY,
        },
        test_rng,
    };

    fn temp_dir(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "trading-bot-bar-world-model-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&path).expect("temp dir");
        path
    }

    /// Deterministic pseudo-random stream, so fixtures never depend on global
    /// RNG state or on the corpus that is still downloading.
    struct Lcg(u64);

    impl Lcg {
        fn next_unit(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
        }
    }

    fn synthetic_supports() -> BarSupports {
        synthetic_supports_with_seed(0x5eed_1234)
    }

    fn synthetic_supports_with_seed(seed: u64) -> BarSupports {
        let mut rng = Lcg(seed);
        let samples: Vec<BarDof> = (0..8192)
            .map(|index| {
                // Real bars pile mass on flat ranges and mid-range closes, which
                // is what makes the atom bins exist; keep some of that here.
                if index % 32 == 0 {
                    return BarDof::default();
                }
                BarDof {
                    r: (rng.next_unit() - 0.5) * 0.06,
                    s: rng.next_unit() * 0.05,
                    u: rng.next_unit(),
                    v: rng.next_unit(),
                    w: (rng.next_unit() - 0.5) * 4.0,
                }
            })
            .collect();
        BarSupports::fit(&samples)
    }

    fn synthetic_dof(batch: i64, len: i64, seed: u64) -> Tensor {
        let mut rng = Lcg(seed);
        let values: Vec<f32> = (0..batch * len)
            .flat_map(|_| {
                [
                    (rng.next_unit() - 0.5) * 0.06,
                    rng.next_unit() * 0.05,
                    rng.next_unit(),
                    rng.next_unit(),
                    (rng.next_unit() - 0.5) * 4.0,
                ]
            })
            .collect();
        Tensor::from_slice(&values).view([batch, len, BAR_DOF as i64])
    }

    /// Valid ids for every conditioning channel, walking a plausible ET session so
    /// consecutive bars differ. Built here rather than through `dataset` so the
    /// trunk tests never depend on the corpus or on tz data.
    ///
    /// The elapsed, day-edge and market channels cycle through their whole ranges rather than
    /// sitting at one row, so a shape or offset error in the fused bank surfaces here instead of
    /// only under a real corpus.
    fn synthetic_time_ids(batch: i64, len: i64, start_minute: i64) -> Tensor {
        let values: Vec<i64> = (0..batch)
            .flat_map(|b| {
                (0..len).flat_map(move |t| {
                    let minute = (start_minute + 5 * t + 137 * b) % BAR_TIME_CARDINALITY[0];
                    let session = match minute {
                        m if !(240..1200).contains(&m) => 0,
                        m if m < 570 => 1,
                        m if m < 960 => 2,
                        _ => 3,
                    };
                    let mut ids = [0i64; BAR_TIME_FEATURES];
                    ids[TIME_MINUTE] = minute;
                    ids[TIME_WEEKDAY] = (b + t / 78) % BAR_TIME_CARDINALITY[TIME_WEEKDAY];
                    ids[TIME_SESSION] = session;
                    ids[TIME_RESOLUTION] = 0;
                    ids[TIME_ELAPSED] = (t + b) % BAR_TIME_CARDINALITY[TIME_ELAPSED];
                    ids[TIME_DAY_EDGE] = (t / 78) % BAR_TIME_CARDINALITY[TIME_DAY_EDGE];
                    for (channel, salt) in BAR_TIME_MARKET.into_iter().zip([11i64, 29, 47]) {
                        ids[channel] = (salt * t + b) % BAR_TIME_CARDINALITY[channel];
                    }
                    ids
                })
            })
            .collect();
        Tensor::from_slice(&values).view([batch, len, BAR_TIME_FEATURES as i64])
    }

    /// A trunk fixture: matching DOF, bin ids and calendar ids.
    fn synthetic_inputs(
        supports: &BarSupports,
        batch: i64,
        len: i64,
        seed: u64,
    ) -> (Tensor, Tensor, Tensor) {
        let dof = synthetic_dof(batch, len, seed);
        let bin_ids = bar_bin_ids(supports, &dof);
        let time_ids = synthetic_time_ids(batch, len, 570);
        (dof, bin_ids, time_ids)
    }

    /// Every output projection is zero-init, so an untrained trunk is the
    /// identity on its embedding and every cached/uncached path agrees
    /// trivially. Give the projections real mass before testing either.
    ///
    /// `pope_theta_bias` is 2D but is a phase bias, not a projection, and its
    /// `PopeThetaInit::TwoPi` values are what production runs on; overwriting it
    /// with small noise would clamp roughly half its entries to exactly zero and
    /// leave every attention test running against an essentially unbiased PoPE.
    /// `bin_embed` and `bar_prefix_embed` are 2D lookup tables rather than
    /// projections — `fan_in` is meaningless for a gather, and their production
    /// init is unit scale, so rescaling them would test a head nobody trains.
    fn wake_projections(vs: &nn::VarStore, seed: i64) {
        tch::no_grad(|| {
            let _ = tch::manual_seed(seed);
            for (name, mut tensor) in vs.variables() {
                if tensor.dim() != 2
                    || name.contains("bin_embed")
                    || name.contains("bar_prefix_embed")
                    || name.contains("pope_theta_bias")
                {
                    continue;
                }
                let fan_in = *tensor.size().last().expect("2D tensor");
                let noise =
                    Tensor::randn(tensor.size().as_slice(), (tensor.kind(), tensor.device()))
                        * (0.5 / (fan_in as f64).sqrt());
                tensor.copy_(&noise);
            }
        });
    }

    fn write_fixture(dir: &Path, supports: &BarSupports) -> (PathBuf, PathBuf, nn::VarStore) {
        let weights = dir.join("bar_world_model.ot");
        let supports_path = world_model_supports_path(&weights, 300);
        supports.save(&supports_path).expect("save supports");
        let vs = nn::VarStore::new(Device::Cpu);
        let _ = BarModules::new(&vs.root());
        wake_projections(&vs, 7);
        vs.save(&weights).expect("save weights");
        let metadata = BarWorldModelMetadata::save_for_checkpoint(&weights, &[300], 300)
            .expect("save metadata");
        (weights, metadata, vs)
    }

    #[test]
    fn architecture_constants_are_self_consistent() {
        assert!((BAR_RESID_LAMBDA_INIT - 1.1f64.sqrt()).abs() < 1e-15);
        assert_eq!(BAR_POST_LAMBDA_INIT, 1.0);
        assert_eq!(
            BAR_DYNAMICS_HIDDEN,
            (1.6 * 1024.0f64 / 128.0).round() as i64 * 128
        );
        assert_eq!(BAR_HEADS * BAR_HEAD_DIM, BAR_MODEL_DIM);
    }

    #[test]
    fn parameter_groups_partition_the_var_store() {
        let _torch_rng_guard = test_rng::shared();
        let vs = nn::VarStore::new(Device::Cpu);
        let _ = BarModules::new(&vs.root());
        let muon = bar_muon_name_substrings();
        // The union is hand-maintained and the down-projection list only ever
        // grants a learning-rate multiplier, so neither drifting apart from its
        // sources would fail any other assertion here.
        let halves: Vec<&str> = BarTrunk::muon_name_substrings()
            .iter()
            .chain(BarDynamics::muon_name_substrings())
            .copied()
            .collect();
        assert_eq!(
            muon.to_vec(),
            halves,
            "the NorMuon union drifted from the trunk and dynamics halves"
        );
        for down in bar_muon_down_projection_substrings() {
            assert!(
                muon.contains(down),
                "down-projection substring {down} is not NorMuon-routed, so its 4.0x \
                 learning-rate multiplier would land on an AdamW parameter"
            );
        }
        let mut total = 0i64;
        let mut trunk_and_dynamics = 0i64;
        let mut seen = HashSet::new();
        for (name, tensor) in vs.variables() {
            let emission = BAR_EMISSION_ADAMW_NAME_SUBSTRINGS
                .iter()
                .any(|s| name.contains(s));
            let groups = [
                muon.iter().any(|s| name.contains(s)),
                bar_adamw_embedding_substrings()
                    .iter()
                    .any(|s| name.contains(s)),
                bar_adamw_scalar_substrings()
                    .iter()
                    .any(|s| name.contains(s)),
                emission,
            ];
            assert_eq!(
                groups.iter().filter(|hit| **hit).count(),
                1,
                "{name} must land in exactly one optimizer group, got {groups:?}"
            );
            if muon.iter().any(|s| name.contains(s)) {
                assert_eq!(tensor.dim(), 2, "{name} routed to NorMuon but is not 2D");
            }
            total += tensor.numel() as i64;
            if !emission {
                trunk_and_dynamics += tensor.numel() as i64;
            }
            assert!(seen.insert(name), "duplicate variable name");
        }
        println!("bar world model parameters: total={total} trunk+dynamics={trunk_and_dynamics}");
        assert!(
            (30_000_000..45_000_000).contains(&trunk_and_dynamics),
            "trunk+dynamics parameter count {trunk_and_dynamics} left the ~32M design point"
        );
    }

    #[test]
    fn metadata_round_trips_and_pins_lineage() {
        let _torch_rng_guard = test_rng::exclusive();
        let dir = temp_dir("metadata");
        let supports = synthetic_supports();
        let (weights, metadata_path, _vs) = write_fixture(&dir, &supports);

        let metadata = BarWorldModelMetadata::load(&metadata_path).expect("load metadata");
        assert_eq!(metadata.format_version, BAR_METADATA_VERSION);
        assert_eq!(metadata.architecture, BAR_ARCHITECTURE);
        assert_eq!(metadata.model_dim, BAR_MODEL_DIM);
        assert_eq!(metadata.num_bins, NUM_BAR_BINS);
        assert_eq!(metadata.res_secs, 300);
        assert_eq!(metadata.lineage_sha256.len(), 64);
        assert!(!metadata.supports_sha256.is_empty());
        metadata.validate_checkpoint(&weights).expect("checkpoint");
        metadata.validate_supports(&weights).expect("supports");

        // Recomputing from the same files reproduces an identical sidecar.
        let again =
            BarWorldModelMetadata::for_checkpoint(&weights, &[300], 300).expect("recompute");
        assert_eq!(again, metadata);

        // The lineage is a function of the supports, not just the weights.
        let mut swapped = metadata.clone();
        swapped.supports_sha256.insert(300, "0".repeat(64));
        assert!(swapped.validate_schema().is_err());

        // A deployment resolution with no fitted support is rejected outright.
        assert!(BarWorldModelMetadata::for_checkpoint(&weights, &[300], 86_400).is_err());

        // And of every architectural constant, and of the bar resolution.
        for mutate in [
            (|m: &mut BarWorldModelMetadata| m.model_dim *= 2) as fn(&mut BarWorldModelMetadata),
            |m: &mut BarWorldModelMetadata| m.layers += 1,
            |m: &mut BarWorldModelMetadata| m.res_secs = 60,
            |m: &mut BarWorldModelMetadata| m.format_version = 5,
        ] {
            let mut tampered = metadata.clone();
            mutate(&mut tampered);
            assert!(
                tampered.validate_schema().is_err(),
                "a tampered sidecar must not validate"
            );
        }

        // Recovery fields were added to metadata v8. Their serde defaults and empty canonical
        // rendering keep every older v8 deployment sidecar byte-for-byte lineage compatible.
        let mut legacy_value = serde_json::to_value(&metadata).expect("serialize metadata");
        let legacy_object = legacy_value.as_object_mut().expect("metadata object");
        legacy_object.remove("optimizer_checkpoint_sha256");
        legacy_object.remove("optimizer_initialized_adamw");
        legacy_object.remove("pretrain_recovery");
        let legacy: BarWorldModelMetadata =
            serde_json::from_value(legacy_value).expect("deserialize legacy metadata");
        assert!(legacy.optimizer_checkpoint_sha256.is_none());
        assert!(legacy.optimizer_initialized_adamw.is_empty());
        assert!(legacy.pretrain_recovery.is_none());
        legacy
            .validate_schema()
            .expect("legacy metadata remains lineage-compatible");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn metadata_rejects_tampered_weights_and_mismatched_support_geometry() {
        let _torch_rng_guard = test_rng::exclusive();
        let dir = temp_dir("warm-start-authentication");
        let checkpoint_supports = synthetic_supports();
        let (weights, metadata_path, _vs) = write_fixture(&dir, &checkpoint_supports);
        let metadata = BarWorldModelMetadata::load(&metadata_path).expect("metadata");

        let matching =
            BarSupportSet::new(vec![(300, synthetic_supports())]).expect("matching supports");
        metadata
            .validate_support_geometry(&weights, &matching)
            .expect("identical geometry is valid");

        let mismatched = BarSupportSet::new(vec![(300, synthetic_supports_with_seed(0xdead_beef))])
            .expect("mismatched supports");
        let mismatch = metadata
            .validate_support_geometry(&weights, &mismatched)
            .expect_err("different bin geometry must be rejected")
            .to_string();
        assert!(mismatch.contains("support geometry differs"));

        fs::write(&weights, b"tampered weights").expect("tamper weights");
        let tamper = metadata
            .validate_checkpoint(&weights)
            .expect_err("metadata must authenticate weights before loading")
            .to_string();
        assert!(tamper.contains("checkpoint hash mismatch"));
        fs::remove_dir_all(&dir).ok();
    }

    fn training_fixture(scoring: BarScoring) -> BarTrainingProvenance {
        BarTrainingProvenance {
            corpus_fingerprint: "c0ffee".to_owned(),
            split_bounds: (1_759_839_000_000, 1_773_427_500_000),
            split_bounds_pinned: true,
            eval_window_seed: 0xE7A1_5E7D_0001,
            train_seed: 0x5EED,
            selection_metric: "nll_bar_conditional".to_owned(),
            selection_weights: [1.0; BAR_DOF],
            selection_guard_dof: "r".to_owned(),
            selection_guard_se_multiple: 1.0,
            min_dollar_volume: 0.0,
            symbols: 3000,
            supports_frozen: false,
            supports_corpus_fingerprint: None,
            universe_fingerprint: None,
            universe_train_end_ms: None,
            direct_t2_weight: 0.0,
            direct_t3_weight: 0.0,
            scoring: scoring.to_string(),
            sizing_rule: "global_quadratic_raw_mean".to_owned(),
            selection_context: BAR_MAX_CONTEXT,
            deployed_context: BAR_MAX_CONTEXT,
            reached_context: BAR_MAX_CONTEXT,
            global_step: Some(30_000),
            selection_bench_context: Some(896),
            selection_edge_bps: Some(0.3382),
            selection_edge_se_bps: Some(0.0204),
            selection_nll_conditional: Some(-9.3817),
            batch_ramp: vec![1, 2, 3],
            lr_plateau_fraction: 0.40,
            optimizer: None,
            dof_scaling: DofScaling::Raw,
        }
    }

    #[test]
    fn recovery_planned_batch_ramp_is_lineage_bound_and_legacy_compatible() {
        let _torch_rng_guard = test_rng::exclusive();
        let dir = temp_dir("recovery_ramp_lineage");
        let supports = synthetic_supports();
        let (weights, _metadata_path, _vs) = write_fixture(&dir, &supports);
        let mut metadata = BarWorldModelMetadata::for_checkpoint_with(
            &weights,
            &[300],
            300,
            Some(training_fixture(BarScoring::Hard)),
        )
        .expect("metadata");
        metadata.attach_pretrain_recovery(
            "1".repeat(64),
            Vec::new(),
            BarPretrainRecovery {
                format_version: BAR_PRETRAIN_RECOVERY_VERSION,
                direct_t2_weight: 0.0,
                direct_t3_weight: 0.0,
                total_steps: 90,
                stage_steps: vec![10, 20, 60],
                derived_batch_ramp: vec![1, 2, 3],
                base_batch: 8,
                requested_batch: 8,
                exact_batch: true,
                quadratic_kelly: true,
                mean_shrink: false,
                epochs: 1,
                steps_override: None,
                dyn_horizon: 1,
                lambda_dyn: 1.0,
                lambda_kl: 1.0,
                auxiliary_resolutions: Vec::new(),
                adamw_accumulation_pending: false,
                activation_bytes_per_token: 480_000.0,
                activation_device: "GPU-test:test-device".to_owned(),
                checkpoint_every: 32,
                validate_every: 64,
                ablation_lr: 1e-3,
                smd_meta_lr: 1e5,
            },
        );
        metadata.validate_schema().expect("recovery metadata");

        let mut tampered = metadata.clone();
        tampered
            .pretrain_recovery
            .as_mut()
            .expect("recovery")
            .derived_batch_ramp[1] = 1;
        assert!(
            tampered.validate_schema().is_err(),
            "the planned pass partition must be lineage-bound"
        );

        let mut direct_tampered = metadata.clone();
        direct_tampered
            .pretrain_recovery
            .as_mut()
            .expect("recovery")
            .direct_t2_weight = 0.5;
        assert!(
            direct_tampered.validate_schema().is_err(),
            "direct objective weights must be lineage-authenticated"
        );

        let mut obsolete_recipe = metadata.clone();
        obsolete_recipe
            .pretrain_recovery
            .as_mut()
            .expect("recovery")
            .format_version = BAR_PRETRAIN_RECOVERY_VERSION - 1;
        obsolete_recipe.lineage_sha256 = obsolete_recipe.compute_lineage_sha256();
        assert!(
            obsolete_recipe.validate_schema().is_err(),
            "an obsolete direct-objective schedule must fail even with recomputed lineage"
        );

        let mut legacy = metadata;
        legacy
            .pretrain_recovery
            .as_mut()
            .expect("recovery")
            .derived_batch_ramp
            .clear();
        legacy.lineage_sha256 = legacy.compute_lineage_sha256();
        let mut legacy_value = serde_json::to_value(&legacy).expect("serialize legacy");
        legacy_value
            .get_mut("pretrain_recovery")
            .and_then(serde_json::Value::as_object_mut)
            .expect("recovery object")
            .remove("derived_batch_ramp");
        let decoded: BarWorldModelMetadata =
            serde_json::from_value(legacy_value).expect("deserialize legacy");
        assert!(decoded
            .pretrain_recovery
            .as_ref()
            .expect("recovery")
            .derived_batch_ramp
            .is_empty());
        decoded
            .validate_schema()
            .expect("legacy recovery lineage remains compatible");
        fs::remove_dir_all(&dir).ok();
    }

    /// The scoring rule decides what every `nll_bar` in a run MEANS — the three modes are
    /// tens of nats apart on the identical model — so it has to be part of the artifact's
    /// identity. Two runs that differ only in the rule must not share a lineage hash, and a
    /// sidecar relabelled with a rule its weights were not trained under must not validate.
    #[test]
    fn the_scoring_rule_is_part_of_the_lineage() {
        let _torch_rng_guard = test_rng::exclusive();
        let dir = temp_dir("lineage_scoring");
        let supports = synthetic_supports();
        let (weights, _metadata_path, _vs) = write_fixture(&dir, &supports);

        let of = |scoring| {
            BarWorldModelMetadata::for_checkpoint_with(
                &weights,
                &[300],
                300,
                Some(training_fixture(scoring)),
            )
            .expect("metadata")
        };
        let mut seen = HashSet::new();
        for scoring in BarScoring::ALL {
            let metadata = of(scoring);
            metadata
                .validate_schema()
                .expect("its own lineage validates");
            assert!(
                seen.insert(metadata.lineage_sha256.clone()),
                "{scoring} collided with another mode's lineage"
            );
        }
        assert_eq!(seen.len(), BarScoring::ALL.len());

        // Relabelling the mode after the fact must break the hash rather than silently
        // rewrite what the numbers mean.
        let mut relabelled = of(BarScoring::Density);
        relabelled
            .training
            .as_mut()
            .expect("training provenance")
            .scoring = BarScoring::Smoothed.to_string();
        assert!(relabelled.validate_schema().is_err());

        // The plateau fraction is the learning-rate schedule's one free parameter: at one full
        // pass a checkpoint sits at the peak rate or at the annealed floor depending on it, so
        // relabelling it must break the hash exactly as relabelling the scoring rule does.
        let mut rescheduled = of(BarScoring::Density);
        rescheduled
            .training
            .as_mut()
            .expect("training provenance")
            .lr_plateau_fraction = 0.90;
        assert!(rescheduled.validate_schema().is_err());

        // A sidecar written before the fraction was recorded renders NOTHING for it, which is
        // what keeps the artifacts already on disk valid; it is still distinguishable from one
        // that records a fraction.
        let mut unrecorded = of(BarScoring::Density);
        unrecorded
            .training
            .as_mut()
            .expect("training provenance")
            .lr_plateau_fraction = 0.0;
        unrecorded.lineage_sha256 = unrecorded.compute_lineage_sha256();
        unrecorded
            .validate_schema()
            .expect("an unrecorded fraction still validates");
        assert!(!seen.contains(&unrecorded.lineage_sha256));
        // The byte-level property the 716 sidecars already on disk depend on: an unrecorded
        // fraction contributes NO text to the canonical rendering, so their stored hashes are
        // still the hashes this build computes. A recorded one contributes exactly one field.
        assert!(!unrecorded
            .training_canonical()
            .contains("lr_plateau_fraction"));
        assert!(
            rescheduled
                .training_canonical()
                .ends_with(";lr_plateau_fraction=3feccccccccccccd"),
            "{}",
            rescheduled.training_canonical()
        );

        // Where a reader actually finds it: the sidecar's own `training.lr_plateau_fraction`,
        // beside the batch ramp and the scoring rule the same comparison needs.
        let sidecar = serde_json::to_value(of(BarScoring::Density)).expect("serialize");
        assert_eq!(sidecar["training"]["lr_plateau_fraction"], 0.40);

        let mut off_recipe = training_fixture(BarScoring::Density);
        off_recipe.optimizer = Some(BarOptimizerProvenance {
            name: "NorMuon + AdamW".to_owned(),
            row_learned_lr: None,
            ablation: None,
        });
        let mut on_recipe = off_recipe.clone();
        on_recipe
            .optimizer
            .as_mut()
            .expect("optimizer provenance")
            .row_learned_lr = Some(BarRowLearnedLrProvenance {
            c: 1.0,
            controller_lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            warmup_primary_steps: 100,
            evidence_clip: 3.0,
            evidence: "-mean(g_raw * delta_previous)".to_owned(),
        });
        let off_metadata =
            BarWorldModelMetadata::for_checkpoint_with(&weights, &[300], 300, Some(off_recipe))
                .expect("off optimizer provenance");
        let on_metadata =
            BarWorldModelMetadata::for_checkpoint_with(&weights, &[300], 300, Some(on_recipe))
                .expect("enabled optimizer provenance");
        assert_ne!(off_metadata.lineage_sha256, on_metadata.lineage_sha256);
        let sidecar = serde_json::to_value(on_metadata).expect("serialize controller provenance");
        assert_eq!(sidecar["training"]["optimizer"]["row_learned_lr"]["c"], 1.0);
        assert_eq!(
            sidecar["training"]["optimizer"]["row_learned_lr"]["warmup_primary_steps"],
            100
        );

        // Adding an absent ablation field must not move the production lineage: legacy JSON
        // omits it, serde supplies None, and canonical rendering contributes no bytes.
        let mut legacy_json =
            serde_json::to_value(&off_metadata).expect("serialize production optimizer");
        legacy_json["training"]["optimizer"]
            .as_object_mut()
            .expect("optimizer object")
            .remove("ablation");
        let legacy: BarWorldModelMetadata =
            serde_json::from_value(legacy_json).expect("deserialize legacy optimizer");
        assert_eq!(legacy.lineage_sha256, off_metadata.lineage_sha256);
        assert!(!legacy.training_canonical().contains(";ablation="));

        let mut smd_recipe = training_fixture(BarScoring::Density);
        smd_recipe.optimizer = Some(BarOptimizerProvenance {
            name: "batch-step Schraudolph SMD-IDBD".to_owned(),
            row_learned_lr: None,
            ablation: Some(BarOptimizerAblationProvenance {
                variant: "smd-idbd".to_owned(),
                initial_lr: 1e-3,
                meta_lr: Some(0.05),
                exact_hvp: true,
                beta_bounds: Some((-10.0, 2.0)),
                beta_update_clip: Some(2.0),
            }),
        });
        let smd_metadata =
            BarWorldModelMetadata::for_checkpoint_with(&weights, &[300], 300, Some(smd_recipe))
                .expect("SMD optimizer provenance");
        assert_ne!(smd_metadata.lineage_sha256, off_metadata.lineage_sha256);
        let canonical = smd_metadata.training_canonical();
        assert!(canonical.contains(";ablation=smd-idbd"));
        assert!(canonical.contains("exact_hvp=true"));

        // An artifact that records no provenance at all is still distinguishable from every
        // recorded one, which is the point of hashing "none".
        let bare = BarWorldModelMetadata::for_checkpoint(&weights, &[300], 300).expect("bare");
        assert!(!seen.contains(&bare.lineage_sha256));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn untrained_trunk_emits_finite_beliefs() {
        let _torch_rng_guard = test_rng::exclusive();
        let _ = tch::manual_seed(0xabc);
        let supports = synthetic_supports();
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        let (dof, ids, time_ids) = synthetic_inputs(&supports, 2, 32, 0xabc);
        let beliefs = modules.trunk.forward(&dof, &ids, &time_ids, 0, false);
        assert_eq!(beliefs.size(), vec![2, 32, BAR_MODEL_DIM]);
        assert!(bool::try_from(beliefs.isfinite().all()).expect("finite check"));

        // Every trunk sublayer output is annihilated before it reaches the residual at
        // initialization, so this covers the EMBEDDING path and the starting point of training,
        // not attention. The attention coverage lives in `cached_forward_matches_full_forward`
        // and `sliding_window_bounds_the_receptive_field`, which wake the projections first.
        // The trunk is therefore the identity on its normalized embedding. The emission head,
        // however, deliberately has small fan-in-scaled random weights so categorical NLL has
        // a gradient into the belief on step zero; uniform logits are not its contract.
        let rms = beliefs
            .pow_tensor_scalar(2.0)
            .mean_dim([-1i64].as_slice(), false, Kind::Float)
            .sqrt();
        let deviation = f64::try_from((rms - 1.0).abs().max()).expect("rms");
        assert!(deviation < 1e-3, "belief RMS deviated by {deviation}");

        let conditioning = modules.trunk.forecast_conditioning(&time_ids, &time_ids);
        let bins = supports.bin_ids(&dof);
        let logits = modules.head.logits(&beliefs, &conditioning, &bins);
        assert!(bool::try_from(logits.isfinite().all()).expect("finite logits"));
        let log_probs = logits.log_softmax(-1, Kind::Float);
        let probabilities = log_probs.exp();
        let max_probability =
            f64::try_from(probabilities.max()).expect("maximum initial probability");
        assert!(
            max_probability < 0.10,
            "an untrained categorical collapsed onto one bin with probability {max_probability}"
        );
        let entropy =
            -(&probabilities * &log_probs).sum_dim_intlist([-1i64].as_slice(), false, Kind::Float);
        let min_entropy = f64::try_from(entropy.min()).expect("minimum initial entropy");
        let uniform_entropy = (NUM_BAR_BINS as f64).ln();
        assert!(
            min_entropy > uniform_entropy - 0.5,
            "an untrained categorical lost too much entropy: minimum {min_entropy}, uniform \
             {uniform_entropy}"
        );

        let (nll, _) = modules
            .head
            .nll(&beliefs, &conditioning, &dof, &supports, BarScoring::Hard);
        let nll = f64::try_from(nll).expect("nll");
        let uniform_nll = BAR_DOF as f64 * uniform_entropy;
        let sane_margin = 0.25 * BAR_DOF as f64;
        assert!(
            nll.is_finite() && (nll - uniform_nll).abs() < sane_margin,
            "untrained nll {nll} is not sane relative to uniform {uniform_nll}"
        );

        // Once the projections carry mass the beliefs must stay finite and must
        // stop being a pure function of the embedding.
        wake_projections(&vs, 31);
        let woken = modules.trunk.forward(&dof, &ids, &time_ids, 0, false);
        assert!(bool::try_from(woken.isfinite().all()).expect("finite check"));
        assert!(
            f64::try_from((woken - beliefs).abs().max()).expect("delta") > 1e-4,
            "waking the projections left the trunk unchanged"
        );
    }

    #[test]
    fn calendar_ids_change_the_belief() {
        let _torch_rng_guard = test_rng::exclusive();
        let supports = synthetic_supports();
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        wake_projections(&vs, 23);
        let (dof, ids, regular) = synthetic_inputs(&supports, 1, 16, 0x7070);
        // Same bars, overnight instead of regular-hours clock.
        let overnight = synthetic_time_ids(1, 16, 1260);
        let a = modules.trunk.forward(&dof, &ids, &regular, 0, false);
        let b = modules.trunk.forward(&dof, &ids, &overnight, 0, false);
        let difference = f64::try_from((a - b).abs().max()).expect("difference");
        assert!(
            difference > 1e-4,
            "the trunk ignored the calendar channel ({difference})"
        );
    }

    #[test]
    fn forecast_conditioning_uses_target_clock_and_current_market_only() {
        let _torch_rng_guard = test_rng::exclusive();
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        wake_projections(&vs, 43);

        let h = Tensor::zeros([2, 1, BAR_MODEL_DIM], (Kind::Float, Device::Cpu));
        let bins = Tensor::zeros([2, 1, BAR_DOF as i64], (Kind::Int64, Device::Cpu));
        let target = synthetic_time_ids(2, 1, 570);
        let current = synthetic_time_ids(2, 1, 565);
        let market_channels = Tensor::from_slice(&BAR_TIME_MARKET.map(|slot| slot as i64));

        let target_market_changed = target.index_fill(-1, &market_channels, 127);
        let base_conditioning = modules.trunk.forecast_conditioning(&target, &current);
        let no_target_market = modules
            .trunk
            .forecast_conditioning(&target_market_changed, &current);
        assert_eq!(
            f64::try_from((&base_conditioning - &no_target_market).abs().max()).expect("gap"),
            0.0,
            "the target bar's observed market ids leaked into forecast conditioning"
        );

        let minute = Tensor::from_slice(&[crate::torch::dataset::TIME_MINUTE as i64]);
        let shifted_clock = target.index_fill(-1, &minute, 17);
        let clock_conditioning = modules
            .trunk
            .forecast_conditioning(&shifted_clock, &current);
        let base_logits = modules.head.logits(&h, &base_conditioning, &bins);
        let no_target_logits = modules.head.logits(&h, &no_target_market, &bins);

        assert_eq!(
            f64::try_from((&base_logits - no_target_logits).abs().max()).expect("logit gap"),
            0.0,
            "the target bar's observed market ids leaked into emission logits"
        );
        let clock_logits = modules.head.logits(&h, &clock_conditioning, &bins);
        assert!(
            f64::try_from((clock_logits - &base_logits).abs().max()).expect("clock gap") > 1e-5,
            "the target exogenous clock did not reach the emission logits"
        );

        let changed_current = current.index_fill(-1, &market_channels, 127);
        let market_conditioning = modules
            .trunk
            .forecast_conditioning(&target, &changed_current);
        let market_logits = modules.head.logits(&h, &market_conditioning, &bins);
        assert!(
            f64::try_from((market_logits - base_logits).abs().max()).expect("market gap") > 1e-5,
            "the current observed market did not reach the emission logits after wake-up"
        );
    }
    #[test]
    fn private_direct_heads_copy_h1_initialization_without_aliasing() {
        let _torch_rng_guard = test_rng::exclusive();
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        let h = Tensor::randn([2, 3, BAR_MODEL_DIM], (Kind::Float, Device::Cpu));
        let conditioning = Tensor::randn([2, 3, BAR_MODEL_DIM], (Kind::Float, Device::Cpu));
        let bins = Tensor::zeros([2, 3, BAR_DOF as i64], (Kind::Int64, Device::Cpu));
        let h1 = modules.head.logits(&h, &conditioning, &bins);
        for horizon in [2usize, 3] {
            let direct = modules
                .head
                .direct_logits(&h, &conditioning, &bins, horizon);
            assert_eq!(
                (&h1 - direct).abs().max().double_value(&[]),
                0.0,
                "private h{horizon} bank did not copy the h1 initialization"
            );
        }

        let mut h2_r = vs
            .variables()
            .remove("bar_dof_head_direct_h2_r.weight")
            .expect("private h2 r weight");
        tch::no_grad(|| {
            let _ = h2_r.zero_();
        });
        let h1_after = modules.head.logits(&h, &conditioning, &bins);
        let h2_after = modules.head.direct_logits(&h, &conditioning, &bins, 2);
        assert_eq!((&h1 - h1_after).abs().max().double_value(&[]), 0.0);
        assert!(
            (&h1 - h2_after).abs().max().double_value(&[]) > 0.0,
            "mutating the private h2 bank changed no direct logits"
        );
    }

    #[test]
    fn direct_and_h1_emission_gradients_use_disjoint_banks_with_live_beliefs() {
        let _torch_rng_guard = test_rng::exclusive();
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        let h = Tensor::randn([2, 3, BAR_MODEL_DIM], (Kind::Float, Device::Cpu))
            .set_requires_grad(true);
        let conditioning = Tensor::randn([2, 3, BAR_MODEL_DIM], (Kind::Float, Device::Cpu));
        let bins = Tensor::zeros([2, 3, BAR_DOF as i64], (Kind::Int64, Device::Cpu));

        modules
            .head
            .direct_logits(&h, &conditioning, &bins, 2)
            .square()
            .mean(Kind::Float)
            .backward();
        let variables = vs.variables();
        let main = &variables["bar_dof_head_r.weight"];
        let h2 = &variables["bar_dof_head_direct_h2_r.weight"];
        let h3 = &variables["bar_dof_head_direct_h3_r.weight"];
        assert!(!main.grad().defined(), "direct loss reached the h1 head");
        assert!(h2.grad().defined() && h2.grad().abs().max().double_value(&[]) > 0.0);
        assert!(!h3.grad().defined(), "h2 loss reached the h3 head");
        assert!(
            h.grad().defined() && h.grad().abs().max().double_value(&[]) > 0.0,
            "private direct loss stopped at its readout instead of reaching the shared trunk"
        );

        for tensor in variables.values() {
            let mut grad = tensor.grad();
            if grad.defined() {
                let _ = grad.zero_();
            }
        }
        let mut h_grad = h.grad();
        let _ = h_grad.zero_();
        modules
            .head
            .logits(&h, &conditioning, &bins)
            .square()
            .mean(Kind::Float)
            .backward();
        assert!(main.grad().defined() && main.grad().abs().max().double_value(&[]) > 0.0);
        assert!(
            !h2.grad().defined() || h2.grad().abs().max().double_value(&[]) == 0.0,
            "h1 loss reached the private h2 head"
        );
        assert!(
            !h3.grad().defined() || h3.grad().abs().max().double_value(&[]) == 0.0,
            "h1 loss reached the private h3 head"
        );
    }

    #[test]
    fn fixed_recirculation_is_sequential_validated_and_finite() {
        let _torch_rng_guard = test_rng::exclusive();
        let supports = synthetic_supports();
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        wake_projections(&vs, 0x51A1);
        let (dof, ids, time_ids) = synthetic_inputs(&supports, 2, 16, 0x51A2);

        let serial = modules.trunk.forward_serialized(&dof, &ids, &time_ids);
        let disabled =
            RecirculationConfig::new(4, 1, 0.0, false, 10).expect("valid disabled config");
        let disabled_out = modules
            .trunk
            .forward_recirculated(&dof, &ids, &time_ids, &disabled)
            .expect("disabled recirculation");
        assert_eq!(
            f64::try_from((&serial - &disabled_out).abs().max()).expect("disabled delta"),
            0.0,
            "alpha zero must be an exact no-op on the serialized path"
        );

        let enabled =
            RecirculationConfig::new(4, 1, 0.15, false, 10).expect("valid enabled config");
        let enabled_out = modules
            .trunk
            .forward_recirculated(&dof, &ids, &time_ids, &enabled)
            .expect("enabled recirculation");
        assert_eq!(
            f64::try_from(
                (&serial.narrow(1, 0, 1) - &enabled_out.narrow(1, 0, 1))
                    .abs()
                    .max()
            )
            .expect("token zero delta"),
            0.0,
            "the position-zero ramp must make token zero an exact no-op"
        );
        assert_eq!(
            f64::try_from(
                (&serial.narrow(1, 1, 1) - &enabled_out.narrow(1, 1, 1))
                    .abs()
                    .max()
            )
            .expect("token one delta"),
            0.0,
            "token one must use its first-pass readout; its recirculation can affect only future tokens"
        );
        assert!(
            f64::try_from(
                (serial.narrow(1, 2, 14) - enabled_out.narrow(1, 2, 14))
                    .abs()
                    .max()
            )
            .expect("later delta")
                > 1e-6,
            "positive alpha must change at least one later belief"
        );
        assert!(bool::try_from(enabled_out.isfinite().all()).expect("finite check"));

        assert!(RecirculationConfig::new(2, 2, 0.1, false, 10).is_err());
        assert!(RecirculationConfig::new(1, 4, 0.1, false, 10).is_err());
        assert!(RecirculationConfig::new(BAR_LAYERS, 1, 0.1, false, 10).is_err());
        assert!(RecirculationConfig::new(4, 1, f64::NAN, false, 10).is_err());
    }

    #[test]
    fn train_and_cached_trunk_paths_mask_same_instant_market_ids() {
        let _torch_rng_guard = test_rng::exclusive();
        let supports = synthetic_supports();
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        wake_projections(&vs, 47);
        let (dof, bins, time_ids) = synthetic_inputs(&supports, 2, 8, 0xCA5E);
        let market_channels = Tensor::from_slice(&BAR_TIME_MARKET.map(|slot| slot as i64));
        let changed = time_ids.index_fill(-1, &market_channels, 127);

        let train_a = modules.trunk.forward(&dof, &bins, &time_ids, 0, true);
        let train_b = modules.trunk.forward(&dof, &bins, &changed, 0, true);
        assert_eq!(
            f64::try_from((train_a - train_b).abs().max()).expect("train gap"),
            0.0,
            "the training trunk path ingested same-instant market ids"
        );

        let mut cache_a = BarKvCache::new(BAR_MAX_CONTEXT);
        let mut cache_b = BarKvCache::new(BAR_MAX_CONTEXT);
        let cached_a = modules
            .trunk
            .forward_cached(&dof, &bins, &time_ids, &mut cache_a);
        let cached_b = modules
            .trunk
            .forward_cached(&dof, &bins, &changed, &mut cache_b);
        assert_eq!(
            f64::try_from((cached_a - cached_b).abs().max()).expect("cached gap"),
            0.0,
            "the cached trunk path ingested same-instant market ids"
        );
    }

    #[test]
    fn cached_forward_matches_full_forward() {
        let _torch_rng_guard = test_rng::exclusive();
        let supports = synthetic_supports();
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        wake_projections(&vs, 11);

        let (dof, ids, time_ids) = synthetic_inputs(&supports, 2, 64, 0xfeed);
        let full = modules.trunk.forward(&dof, &ids, &time_ids, 0, false);

        // Prefill half, then decode the rest one bar at a time, so both cache
        // paths are exercised rather than just the prefill.
        let mut cache = BarKvCache::new(BAR_MAX_CONTEXT);
        let mut pieces = vec![modules.trunk.forward_cached(
            &dof.narrow(1, 0, 32),
            &ids.narrow(1, 0, 32),
            &time_ids.narrow(1, 0, 32),
            &mut cache,
        )];
        for step in 32..64 {
            pieces.push(modules.trunk.forward_cached(
                &dof.narrow(1, step, 1),
                &ids.narrow(1, step, 1),
                &time_ids.narrow(1, step, 1),
                &mut cache,
            ));
        }
        let cached = Tensor::cat(&pieces, 1);
        assert_eq!(cached.size(), full.size());
        assert_eq!(cache.cached_bars(), 64);
        assert_eq!(cache.next_position(), 64);

        let error = f64::try_from((full - cached).abs().max()).expect("max error");
        assert!(error < 1e-4, "cached/full belief mismatch {error}");
    }

    /// The ring-wrap arm of the cache: `length` saturating at `max_tokens`,
    /// `write_index` wrapping, the oldest slot being overwritten, and
    /// `active_after_write` falling through to the whole ring. A saturated cache
    /// attends over exactly the trailing `max_tokens` bars, which is the same
    /// span a sliding-window forward of that width sees, so the two must agree.
    /// Nothing else in the suite reaches this arm — the other cache test tops out
    /// at length == capacity — and in production it is first hit at bar 2049 of a
    /// live session, where a wrong answer looks exactly like a bad forecast.
    #[test]
    fn saturated_cache_evicts_the_oldest_bar() {
        let _torch_rng_guard = test_rng::exclusive();
        let supports = synthetic_supports();
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        wake_projections(&vs, 29);

        let window = 8i64;
        let len = 20i64;
        let (dof, ids, time_ids) = synthetic_inputs(&supports, 1, len, 0xc0de);

        let mut cache = BarKvCache::new(window);
        let prefill = 5i64;
        let mut cached = vec![modules.trunk.forward_cached(
            &dof.narrow(1, 0, prefill),
            &ids.narrow(1, 0, prefill),
            &time_ids.narrow(1, 0, prefill),
            &mut cache,
        )];
        for step in prefill..len {
            cached.push(modules.trunk.forward_cached(
                &dof.narrow(1, step, 1),
                &ids.narrow(1, step, 1),
                &time_ids.narrow(1, step, 1),
                &mut cache,
            ));
        }
        let cached = Tensor::cat(&cached, 1);
        assert_eq!(
            cache.cached_bars(),
            window,
            "cache must saturate at its window"
        );
        assert_eq!(
            cache.next_position(),
            len,
            "positions stay absolute past a wrap"
        );

        // A one-layer-deep comparison: recompute the final belief from exactly the
        // trailing `window` bars with a matching sliding window. The trunk is ten
        // layers deep, so only the last position's reach coincides for a
        // single-layer window; compare the whole trajectory against a full forward
        // instead, which the cache must match wherever nothing has been evicted.
        let unsaturated = modules.trunk.forward(
            &dof.narrow(1, 0, window),
            &ids.narrow(1, 0, window),
            &time_ids.narrow(1, 0, window),
            0,
            false,
        );
        let error = f64::try_from((cached.narrow(1, 0, window) - unsaturated).abs().max())
            .expect("max error");
        assert!(
            error < 1e-4,
            "pre-eviction beliefs diverged from a full forward: {error}"
        );

        // Past saturation the beliefs must still be finite and must keep moving:
        // a wrap that silently read stale or zeroed slots would show up as a
        // frozen or non-finite tail.
        let tail = cached.narrow(1, window, len - window);
        assert!(bool::try_from(tail.isfinite().all()).expect("finite"));
        let step_change = f64::try_from(
            (tail.narrow(1, 1, len - window - 1) - tail.narrow(1, 0, len - window - 1))
                .abs()
                .max(),
        )
        .expect("step change");
        assert!(step_change > 1e-6, "beliefs froze after the ring wrapped");
    }

    /// The embedding shortcut has to be two things at once: exactly shut at init, so adding it
    /// cannot destabilize a run that was tuned without it, and — once opened — the SAME path in
    /// all three trunk loops. `run`, `prefill` and `decode` inject `x0` independently, and a
    /// shortcut wired into two of the three is a train/serve skew that no shape check and no
    /// finiteness check can see, because at init all three agree trivially.
    #[test]
    fn the_embedding_shortcut_is_shut_at_init_and_identical_in_every_trunk_loop() {
        let _torch_rng_guard = test_rng::exclusive();
        let supports = synthetic_supports();
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        wake_projections(&vs, 31);

        // Sorted, because `variables()` is a map: an unordered assignment below would make the
        // per-layer gains differ between runs of this test.
        let mut named: Vec<(String, Tensor)> = vs
            .variables()
            .into_iter()
            .filter(|(name, _)| name.ends_with("x0_lambda"))
            .collect();
        named.sort_by(|a, b| a.0.cmp(&b.0));
        let mut gains: Vec<Tensor> = named.into_iter().map(|(_, tensor)| tensor).collect();
        assert_eq!(gains.len(), BAR_LAYERS, "one shortcut gain per trunk layer");
        for gain in &gains {
            assert_eq!(
                f64::try_from(gain).expect("scalar gain"),
                0.0,
                "the shortcut must be EXACTLY zero at init, not merely small: a small gain \
                 perturbs every belief of the first step"
            );
        }

        let len = 16i64;
        let (dof, ids, time_ids) = synthetic_inputs(&supports, 2, len, 0x0bad_f00d);
        let shut = modules.trunk.forward(&dof, &ids, &time_ids, 0, false);

        // Distinct per layer, so a loop that injected one layer's gain everywhere - or summed
        // them - would not survive the comparisons below.
        tch::no_grad(|| {
            for (index, gain) in gains.iter_mut().enumerate() {
                let _ = gain.fill_(0.1 + 0.05 * index as f64);
            }
        });
        let open = modules.trunk.forward(&dof, &ids, &time_ids, 0, false);
        let opened = f64::try_from((&open - &shut).abs().max()).expect("shortcut effect");
        assert!(
            opened > 1e-3,
            "opening the shortcut changed nothing, so the path is not wired: {opened}"
        );

        // Whole-sequence prefill, then bar-at-a-time decode, both against the same weights.
        let mut cache = BarKvCache::new(BAR_MAX_CONTEXT);
        let split = 6i64;
        let mut pieces = vec![modules.trunk.forward_cached(
            &dof.narrow(1, 0, split),
            &ids.narrow(1, 0, split),
            &time_ids.narrow(1, 0, split),
            &mut cache,
        )];
        for step in split..len {
            pieces.push(modules.trunk.forward_cached(
                &dof.narrow(1, step, 1),
                &ids.narrow(1, step, 1),
                &time_ids.narrow(1, step, 1),
                &mut cache,
            ));
        }
        let cached = Tensor::cat(&pieces, 1);
        let skew = f64::try_from((&cached - &open).abs().max()).expect("path skew");
        assert!(
            skew < 1e-4,
            "the cached trunk injects the embedding shortcut differently from the training \
             pass: {skew}"
        );
    }

    /// A window-`w` layer sees `w` bars, but stacking `BAR_LAYERS` of them widens
    /// the receptive field of the last position to `BAR_LAYERS * (w - 1) + 1`
    /// bars: each layer reaches back `w - 1` further through the previous layer's
    /// outputs. The test pins exactly that boundary — a bar just inside it must
    /// matter and a bar just outside it must not — rather than the naive
    /// "last position only sees the last `w` bars", which is false for depth > 1.
    #[test]
    fn sliding_window_bounds_the_receptive_field() {
        let _torch_rng_guard = test_rng::exclusive();
        let supports = synthetic_supports();
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        wake_projections(&vs, 13);

        let window = 2;
        let reach = BAR_LAYERS as i64 * (window - 1);
        let len = reach + 6;
        let last = len - 1;
        let (dof, ids, time_ids) = synthetic_inputs(&supports, 1, len, 0x1010);
        let baseline = modules
            .trunk
            .forward(&dof, &ids, &time_ids, window, false)
            .narrow(1, last, 1);

        let perturbed_at = |bar: i64| {
            let rows = dof.copy();
            rows.narrow(1, bar, 1)
                .copy_(&(dof.narrow(1, bar, 1) + 0.25));
            let rows_ids = bar_bin_ids(&supports, &rows);
            modules
                .trunk
                .forward(&rows, &rows_ids, &time_ids, window, false)
                .narrow(1, last, 1)
        };

        // The boundary is exact in the only sense that matters: a bar one step
        // beyond the reach is bit-identical, a bar at the reach is not. The
        // in-reach magnitude is tiny because that bar's influence survives ten
        // successive two-position softmaxes, so the assertion is on it being
        // nonzero, with an adjacent bar as the scale reference.
        let outside = f64::try_from((&baseline - perturbed_at(last - reach - 1)).abs().max())
            .expect("outside");
        assert_eq!(
            outside, 0.0,
            "a bar outside the {reach}-bar reach changed the last position"
        );
        let oldest = f64::try_from((&baseline - perturbed_at(last - reach)).abs().max())
            .expect("oldest in reach");
        assert!(
            oldest > 0.0,
            "the oldest in-reach bar did not reach the last position"
        );
        let adjacent =
            f64::try_from((&baseline - perturbed_at(last - 1)).abs().max()).expect("adjacent");
        assert!(
            adjacent > 1e-3 && adjacent > oldest,
            "influence should attenuate with distance, got adjacent={adjacent} oldest={oldest}"
        );

        // A window at least as long as the sequence is plain causal attention.
        let full = modules.trunk.forward(&dof, &ids, &time_ids, 0, false);
        let wide = modules.trunk.forward(&dof, &ids, &time_ids, len, false);
        let error = f64::try_from((full - wide).abs().max()).expect("max error");
        assert!(
            error < 1e-5,
            "wide window diverged from full causal {error}"
        );
    }

    #[test]
    fn dynamics_starts_as_the_identity_and_stays_on_the_unit_shell() {
        let _torch_rng_guard = test_rng::exclusive();
        let supports = synthetic_supports();
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        assert!(
            vs.variables()
                .keys()
                .all(|name| !name.contains("bar_dyn_dof_embed")
                    && !name.contains("bar_dyn_time_embed")),
            "dynamics must not own a private token encoder"
        );
        // `step` is only ever handed a belief, and every belief leaves the trunk
        // through the gain-free `rms_norm`, so the identity claim is about the unit
        // shell — feeding an unnormalized latent tests a domain that cannot occur.
        let h = rms_norm(&Tensor::randn(
            [3, 4, BAR_MODEL_DIM],
            (Kind::Float, Device::Cpu),
        ));
        let dof = synthetic_dof(3, 4, 0x2020);
        let time_ids = synthetic_time_ids(3, 4, 570);
        let bins = bar_bin_ids(&supports, &dof);
        let token = modules.trunk.token_embedding(&dof, &bins, &time_ids);
        let predicted = modules.dynamics.step(&h, &token);
        assert_eq!(predicted.size(), h.size());
        let error = f64::try_from((&predicted - &h).abs().max()).expect("max error");
        assert!(
            error < 1e-5,
            "zero-init fc3 must make the step the identity on the unit shell, got {error}"
        );

        wake_projections(&vs, 17);
        let token = modules.trunk.token_embedding(&dof, &bins, &time_ids);
        let predicted = modules.dynamics.step(&h, &token);
        assert!(bool::try_from(predicted.isfinite().all()).expect("finite check"));
        assert!(f64::try_from((&predicted - &h).abs().max()).expect("max") > 0.0);
        // The output has to live where the emission head was fitted, whatever the
        // residual did.
        let shell = predicted
            .pow_tensor_scalar(2.0)
            .mean_dim([-1].as_slice(), false, Kind::Double)
            .sqrt();
        let drift = f64::try_from((shell - 1.0).abs().max()).expect("shell drift");
        assert!(
            drift < 1e-5,
            "a woken dynamics step left the unit-RMS shell by {drift}"
        );

        // The shared token's clock reaches the prediction: same latent and bar, different hour.
        let overnight_token =
            modules
                .trunk
                .token_embedding(&dof, &bins, &synthetic_time_ids(3, 4, 1260));
        let overnight = modules.dynamics.step(&h, &overnight_token);
        assert!(
            f64::try_from((predicted - overnight).abs().max()).expect("max") > 0.0,
            "dynamics ignored the calendar channel"
        );
    }

    #[test]
    fn rollout_shapes_and_decodes_to_valid_bars() {
        let _torch_rng_guard = test_rng::exclusive();
        let dir = temp_dir("rollout");
        let supports = synthetic_supports();
        let (weights, metadata_path, _vs) = write_fixture(&dir, &supports);
        let model = BarWorldModel::load(&weights, &metadata_path, Device::Cpu).expect("load");
        assert!(model.all_parameters_frozen());
        assert_eq!(model.lineage_sha256().len(), 64);
        assert_eq!(model.metadata().res_secs, 300);

        let history = synthetic_dof(2, 16, 0x3030);
        let history_time_ids = synthetic_time_ids(2, 16, 570);
        let (steps, samples) = (6usize, 4usize);
        let future_time_ids = synthetic_time_ids(2, steps as i64, 650);
        let rollout = model.rollout(&history, &history_time_ids, &future_time_ids, samples, 1.0);
        assert_eq!(
            rollout.size(),
            vec![2, samples as i64, steps as i64, BAR_DOF as i64]
        );
        assert!(bool::try_from(rollout.isfinite().all()).expect("finite check"));

        // Pin the sample axis. A transposed interleave would keep the shape, keep
        // every bar valid and pass every other assertion here while building batch
        // element 0's forecast from batch element 1's history. At temperature 0
        // the emission head is the argmax bin center, so every sample of one batch
        // element must be bit-identical and the two elements must differ.
        let greedy = model.rollout(&history, &history_time_ids, &future_time_ids, samples, 0.0);
        for element in 0..2 {
            let rows = greedy.get(element);
            let spread = f64::try_from(
                (rows.narrow(0, 1, samples as i64 - 1) - rows.narrow(0, 0, samples as i64 - 1))
                    .abs()
                    .max(),
            )
            .expect("within-element spread");
            assert_eq!(
                spread, 0.0,
                "greedy samples of one history differ; the sample axis is transposed"
            );
        }
        let across = f64::try_from((greedy.get(0) - greedy.get(1)).abs().max()).expect("across");
        assert!(
            across > 0.0,
            "two different histories produced identical greedy rollouts"
        );

        let flat = Vec::<f32>::try_from(rollout.reshape([-1]).contiguous()).expect("host copy");
        assert_eq!(flat.len(), 2 * samples * steps * BAR_DOF);
        for chunk in flat.chunks_exact(BAR_DOF) {
            let dof = BarDof::from_array([chunk[0], chunk[1], chunk[2], chunk[3], chunk[4]]);
            assert!(dof.is_finite(), "sampled DOF {dof:?} is not finite");
            let bar = decode_dof(100.0, &dof, 1000.0);
            let (open, high, low, close) = (bar.open, bar.high, bar.low, bar.close);
            assert!(
                low > 0.0 && low <= open.min(close) && open.max(close) <= high,
                "sampled bar violates OHLC ordering: o={open} h={high} l={low} c={close}"
            );
            assert!(bar.volume.is_finite() && bar.volume >= 0.0);
        }

        // The dynamics rollout is the same contract at a third of the cost, and
        // both modes hand back beliefs aligned with the bars they generated.
        let dynamic = model.rollout_with(
            &history,
            &history_time_ids,
            &future_time_ids,
            samples,
            1.0,
            RolloutMode::Dynamics,
        );
        assert_eq!(dynamic.dof.size(), rollout.size());
        assert_eq!(
            dynamic.beliefs.size(),
            vec![2, samples as i64, steps as i64, BAR_MODEL_DIM]
        );
        assert!(bool::try_from(dynamic.dof.isfinite().all()).expect("finite check"));
        assert!(bool::try_from(dynamic.beliefs.isfinite().all()).expect("finite check"));

        // A session advanced by a realized bar tracks the same belief the
        // prefill would have produced for the extended history.
        let mut session = model.start_session(&history, &history_time_ids);
        let next_dof = synthetic_dof(2, 1, 0x9090);
        let next_time_ids = synthetic_time_ids(2, 1, 655);
        model
            .advance(&mut session, &next_dof, &next_time_ids)
            .expect("advance");
        let extended = model.beliefs(
            &Tensor::cat(&[&history, &next_dof], 1),
            &Tensor::cat(&[&history_time_ids, &next_time_ids], 1),
        );
        let gap = f64::try_from(
            (session.belief() - extended.narrow(1, 16, 1).squeeze_dim(1))
                .abs()
                .max(),
        )
        .expect("gap");
        assert!(
            gap < 1e-4,
            "session advance diverged from a full prefill: {gap}"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rollout_beliefs_share_their_first_step_across_modes() {
        let _torch_rng_guard = test_rng::exclusive();
        let supports = synthetic_supports();
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        wake_projections(&vs, 19);
        let history = synthetic_dof(2, 12, 0x4040);
        let history_time_ids = synthetic_time_ids(2, 12, 570);
        let future = synthetic_dof(2, 5, 0x5050);
        let future_time_ids = synthetic_time_ids(2, 5, 630);

        let set = BarSupportSet::new(vec![(300, supports)]).expect("support set");
        assert_eq!(set.resolutions(), vec![300]);
        let exact = modules.rollout_beliefs(
            &set,
            &history,
            &history_time_ids,
            &future,
            &future_time_ids,
            RolloutMode::Exact,
        );
        let approx = modules.rollout_beliefs(
            &set,
            &history,
            &history_time_ids,
            &future,
            &future_time_ids,
            RolloutMode::Dynamics,
        );
        assert_eq!(exact.size(), vec![2, 5, BAR_MODEL_DIM]);
        assert_eq!(approx.size(), exact.size());
        // Step 0 is the shared history belief, so its equality is structural.
        // What actually distinguishes the two modes is the tail: if a match arm
        // fell through to the wrong branch, `Dynamics` would silently run the
        // trunk (or vice versa) and every trajectory would coincide.
        let head_gap = f64::try_from((exact.narrow(1, 0, 1) - approx.narrow(1, 0, 1)).abs().max())
            .expect("gap");
        assert_eq!(head_gap, 0.0);
        let tail_gap = f64::try_from((exact.narrow(1, 4, 1) - approx.narrow(1, 4, 1)).abs().max())
            .expect("tail gap");
        assert!(
            tail_gap > 0.0,
            "the cached trunk and the dynamics head produced identical trajectories"
        );
        assert!(bool::try_from(exact.isfinite().all()).expect("finite"));
        assert!(bool::try_from(approx.isfinite().all()).expect("finite"));
    }

    /// A 64-step dynamics rollout must stay on the shell and stay scoreable.
    ///
    /// Without the closing `rms_norm` in [`BarDynamics::step`] the residual
    /// compounds every step — the live run charted `h64 dynamics` at 5,325 nats on
    /// its first validation and 79,321 by its fourth, against 20.93 for the exact
    /// trunk — because the recursion feeds an off-shell latent into a head that was
    /// only ever fitted on unit-RMS beliefs. The belief-norm assertion is the direct
    /// regression guard: it fails the instant the normalization is dropped, long
    /// before the NLL becomes visibly absurd.
    #[test]
    fn dynamics_rollout_stays_finite_and_sane_at_h64() {
        let _torch_rng_guard = test_rng::exclusive();
        const HORIZON: i64 = 64;
        let supports = synthetic_supports();
        let vs = nn::VarStore::new(Device::Cpu);
        let modules = BarModules::new(&vs.root());
        wake_projections(&vs, 23);
        let history = synthetic_dof(2, 96, 0x7070);
        let history_time_ids = synthetic_time_ids(2, 96, 570);
        let future = synthetic_dof(2, HORIZON, 0x8080);
        let future_time_ids = synthetic_time_ids(2, HORIZON, 400);
        let set = BarSupportSet::new(vec![(300, supports)]).expect("support set");

        let measure = |mode: RolloutMode| {
            let beliefs = modules.rollout_beliefs(
                &set,
                &history,
                &history_time_ids,
                &future,
                &future_time_ids,
                mode,
            );
            assert!(
                bool::try_from(beliefs.isfinite().all()).expect("finite"),
                "{} beliefs went non-finite over {HORIZON} steps",
                mode.as_str()
            );
            let last = HORIZON - 1;
            let target_time = future_time_ids.narrow(1, last, 1);
            let current_time = future_time_ids.narrow(1, last - 1, 1);
            let conditioning = modules
                .trunk
                .forecast_conditioning(&target_time, &current_time);
            let (nll, _) = modules.head.nll(
                &beliefs.narrow(1, last, 1),
                &conditioning,
                &future.narrow(1, last, 1),
                set.only(),
                BarScoring::Hard,
            );
            let shell = beliefs
                .pow_tensor_scalar(2.0)
                .mean_dim([-1].as_slice(), false, Kind::Double)
                .sqrt();
            let drift = f64::try_from((shell - 1.0).abs().max()).expect("shell drift");
            (nll.double_value(&[]), drift)
        };

        let (exact_nats, exact_drift) = measure(RolloutMode::Exact);
        let (dynamics_nats, dynamics_drift) = measure(RolloutMode::Dynamics);
        println!(
            "h{HORIZON} nats: exact={exact_nats:.3} dynamics={dynamics_nats:.3} \
             (shell drift exact={exact_drift:.2e} dynamics={dynamics_drift:.2e})"
        );
        assert!(
            exact_drift < 1e-5 && dynamics_drift < 1e-5,
            "beliefs left the unit-RMS shell: exact {exact_drift}, dynamics {dynamics_drift}"
        );
        assert!(
            exact_nats.is_finite() && dynamics_nats.is_finite(),
            "h{HORIZON} nats not finite: exact {exact_nats}, dynamics {dynamics_nats}"
        );
        assert!(
            dynamics_nats < 2.0 * exact_nats,
            "h{HORIZON} dynamics nats {dynamics_nats} ran away from exact {exact_nats}"
        );
    }

    #[test]
    fn bin_ids_match_the_host_side_support_lookup() {
        let supports = synthetic_supports();
        let dof = synthetic_dof(3, 7, 0x6060);
        let ids = bar_bin_ids(&supports, &dof);
        assert_eq!(ids.size(), dof.size());
        let values = Vec::<f32>::try_from(dof.reshape([-1]).contiguous()).expect("dof host");
        let ids = Vec::<i64>::try_from(ids.reshape([-1]).contiguous()).expect("ids host");
        for (index, (&value, &id)) in values.iter().zip(ids.iter()).enumerate() {
            assert_eq!(
                id as usize,
                supports.bin_of(index % BAR_DOF, value as f64),
                "tensor bin lookup disagreed with BarSupports::bin_of"
            );
        }
    }
}
