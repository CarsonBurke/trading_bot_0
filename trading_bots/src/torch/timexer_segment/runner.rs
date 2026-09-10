use super::{
    benchmark::{self, HardwareSampler},
    calibration::{Blocks, FrozenGain, MeanCalibration, Moments, Pairing, CLOSE_CHANNEL},
    compute::{
        Engine, LrSchedule, LrTrajectory, MlpDownLr, OptimizerKind, RecipeKnobs,
        CAPTURE_AFTER_STEPS, NANOGPT_COOLDOWN_FLOOR, NANOGPT_COOLDOWN_FRAC,
        NANOGPT_SCALAR_LR_MULTIPLIER,
    },
    corpus::{self, Batch, Corpus, CorpusContract, WindowRef},
    model::{
        decode_prices, nll_elements, CausalPatchModel, HorizonMean, ModelConfig, X0Lambdas,
        CHANNELS,
    },
    probe,
    reports::{
        self, CandleWindow, EvalTiming, HorizonCurve, HorizonSplit, Metrics, PortfolioCurve,
        StepPhases, TradingCurve, TradingSplit,
    },
    teacher,
    supervision::{self, PatchPhase, RowSelection, SupervisionGeometry},
    target_basis::{self, TargetBasis},
    utility,
};
use crate::torch::{hashing::file_sha256, single_ticker_timexer::runner::cuda_device};
use anyhow::{ensure, Context, Result};
use clap::Args;
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use shared::{paths::RUNS_PATH, report::CandleBar, run_dir::RunDir};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    thread,
    time::Instant,
};
use tch::{nn, Device, Kind, Tensor};
/// Rows in the per-(horizon, channel) amplitude accumulator: the valid-bar count, the target
/// and its square, the three second moments of the (anchor, offset) design, and the two
/// cross moments against the target. Every amplitude quantity this crate reports - the raw
/// per-channel gain, the fitted calibration, the un-gained ratio and the gained ratio - is a
/// function of these eight and nothing else.
const AMPLITUDE_ROWS: i64 = 8;
/// Origins in the in-sample TRAINING amplitude draw. A diagnostic, not a metric: it answers
/// only whether the emitted amplitude is already wrong on the data the weights were fitted to,
/// which is a question about the objective and needs a gain to two digits, not a publishable
/// standard error. 512 strided training origins carry ~512 independent long windows per
/// horizon at a tenth of the sample pass's cost.
const AMPLITUDE_TRAINING_ORIGINS: usize = 512;
/// A cross-section is only usable if it holds enough tickers for a within-timestamp
/// correlation and a decile split to mean anything. Visible to [`super::reports`] because the
/// census panel draws this threshold as a reference line, and a second copy of the number
/// there could drift away from the one the scorer applies.
pub(super) const CROSS_SECTION_MIN: i64 = 20;
/// The `held-out cross-section` draw: the widest uniform block of tickers per evaluation
/// timestamp it can find, and the cap on how many timestamps are drawn.
///
/// The held-out SAMPLE cannot carry a cross-sectional statistic and never could. It is a
/// strided pick over a ticker-major origin list, so its 2,048 origins land on 2,048 different
/// timestamps with one ticker each; every within-timestamp correlation and every decile split
/// is then dropped by [`CROSS_SECTION_MIN`] and the whole trading family reads NaN. That
/// sample is not fixable in place - its persistence NLL is bit-identical across runs, which is
/// what makes step-matched comparison possible, and checkpoint selection is defined on it - so
/// this is a SECOND, separate draw used by nothing but the trading family.
///
/// Timestamp alignment is a property of the corpus and cannot be sampled into existence.
/// Validation origins sit at each ticker's own valid-bar ordinal, `start - 1 + i·pred_len`, so
/// two tickers share a timestamp only when they hold the same number of valid bars between the
/// split boundary and that origin. The measured full split is 433,303 origins over 40,837
/// distinct timestamps - 10.6 per timestamp on average - which is a fully-aligned liquid core
/// plus a long tail of one- and two-ticker timestamps. So the draw ASKS for width and takes
/// what the corpus has, rather than demanding a fixed width and aborting the run:
/// [`cross_section_blocks`] maximizes `timestamps · (width - 1)` over uniform draws, capped at
/// `CROSS_SECTION_TIMESTAMPS · CROSS_SECTION_TICKERS` = 32,768 origins, and fails only if no
/// timestamp at all clears [`CROSS_SECTION_FLOOR`].
///
/// Uniform width is the point, not a convenience. `timestamps · (width - 1)` is the draw's
/// Fisher information for the cross-sectional IC, whose per-timestamp sampling variance is
/// `1/(width - 1)`; a draw mixing 400-name and 45-name timestamps has to be precision-weighted
/// or the thin ones dominate its variance nine to one. Drawing one common width instead makes
/// the scorer's equal-per-timestamp average identically the inverse-variance-weighted one, so
/// no weighting machinery exists to get wrong, and it keeps every timestamp's decile the same
/// size so the pooled spread is a spread of one thing.
///
/// 256 tickers is a 12.8x margin over [`CROSS_SECTION_MIN`] and a 25-name decile per side.
/// `CROSS_SECTION_FLOOR` is twice [`CROSS_SECTION_MIN`]: at the floor a decile is four names,
/// which is already mostly noise, and below it the draw is not worth its wall clock.
/// `pub(super)` because `corpus::in_period_plan` places the in-period hole to SATURATE this
/// cap: a hole whose timestamp blocks are narrower silently costs the in-period draw the
/// precision the in-period-versus-out-of-period comparison is read on.
pub(super) const CROSS_SECTION_TICKERS: usize = 256;
const CROSS_SECTION_TIMESTAMPS: usize = 128;
const CROSS_SECTION_FLOOR: usize = 2 * CROSS_SECTION_MIN as usize;

#[derive(Clone, Debug, Args)]
pub struct TrainArgs {
    /// Optional explicit ticker subset; the default is the eligible universe.
    #[arg(long, value_delimiter = ',')]
    pub ticker: Vec<String>,
    #[arg(long, default_value_os_t = crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub run: Option<String>,
    #[command(flatten)]
    pub model: ModelConfig,
    /// Minimum history used to align origins across context-length comparisons.
    #[arg(long, default_value_t = 6000)]
    pub common_context: usize,
    #[arg(long, default_value_t = 1)]
    pub epochs: usize,
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,
    /// AdamW rate; Polar Express defaults to the modded-nanogpt 0.008 / 0.023 pair. This is the
    /// run's ONE global learning-rate knob: the NorMuon rate is
    /// `OptimizerKind::muon_learning_rate`, exactly proportional to it, so halving this halves
    /// both optimizer groups and no separate multiplier exists or is wanted.
    #[arg(long)]
    pub learning_rate: Option<f64>,
    /// AdamW learning-rate multiplier for the scalar banks - the residual lambdas, the x0
    /// lambdas where they exist, and the U-net skip logits. The 5.0 default is
    /// modded-nanogpt's `"scalars"` group (`train_gpt.py:2030`), tuned against a step budget
    /// two orders of magnitude shorter than one epoch of this corpus.
    #[arg(long, default_value_t = NANOGPT_SCALAR_LR_MULTIPLIER)]
    pub scalar_lr_mult: f64,
    #[arg(long, value_enum, default_value_t=OptimizerKind::PolarExpress)]
    pub optimizer: OptimizerKind,
    #[arg(long, default_value_t = 20260905)]
    pub seed: u64,
    #[arg(long, default_value_t = 1000)]
    pub eval_every: usize,
    #[arg(long, default_value_t = 2048)]
    pub eval_origins: usize,
    /// Rows per batch in every evaluation pass, independent of the TRAINING batch.
    ///
    /// Evaluation runs while the captured training step still owns its private mempool - 17.3
    /// GiB of an 18.1 GiB reservation on the reference recipe - so an evaluation batch is
    /// allocated from what is left of the card, not from a fresh one. A 256-row scoring pass
    /// peaks at 2,690 MiB measured (job 5556), which does not fit beside the graph pool when
    /// foreign tenants hold ~11 GiB, and that single allocation has now killed seven arms
    /// across two days - always inside the evaluation, never inside training. 64 rows costs
    /// four times as many batches at the same 52-53% of the card's measured bf16 peak
    /// (flat from 256 to 1024, job 5556) and reduces the peak roughly linearly.
    ///
    /// This is NOT the training batch and changing it cannot move the training trajectory:
    /// the loss, the optimizer step and the captured graph never see it.
    #[arg(long, default_value_t = 64)]
    pub eval_batch_size: usize,
    /// Complete `pred_len` target runs to reserve as an IN-PERIOD held-out draw, cut out of the
    /// middle of the TRAINING chronological span and purged so that no surviving training row
    /// can supervise any of their target bars.
    ///
    /// This exists to split the one question a chronological held-out split cannot answer.
    /// `held-out full`, `held-out sample` and `held-out cross-section` are all drawn from the
    /// `[80%, 90%)` partition, so every one of them is out-of-sample in origin identity AND
    /// out-of-period in market regime at the same time, and the two have OPPOSITE fixes:
    /// overfitting is answered with capacity, regularization or effective sample size, and
    /// non-stationarity is answered with data recency, target definition or online adaptation.
    /// This draw holds the period fixed and varies only origin identity.
    ///
    /// 0 is off and leaves every existing population, and the manifest digest, byte-identical.
    /// A nonzero value REMOVES training rows - see the load-time census - so a holed arm's
    /// TRAINING losses are not step-comparable to a control's.
    #[arg(long, default_value_t = 0)]
    pub in_period_sections: usize,
    /// Completed full-corpus epochs without improved full-validation NLL.
    #[arg(long, default_value_t = 3)]
    pub patience: usize,
    /// Consecutive held-out sample evaluations (after the first two) without improved NLL.
    #[arg(long, default_value_t = 3)]
    pub preview_patience: usize,
    #[arg(long, default_value_t=true, action=clap::ArgAction::Set)]
    pub fused: bool,
    /// Tickers that must hold a valid bar at a grid timestamp for it to define a market step.
    #[arg(long, default_value_t = 2000)]
    pub market_min_cross_section: usize,
    /// Optimizer steps the learning-rate warmdown is shaped against, INDEPENDENT of how many
    /// steps the run takes; 0 shapes it against the whole run (`steps_per_epoch * epochs`),
    /// which is what every arm did before this knob existed. The cooldown occupies the last
    /// 60% of the budget and ends at 0.15 of the base rate, so a 5,000-step budget starts
    /// cooling at step 2,000 and holds 0.15 afterwards, while the 9,590-step epoch that has
    /// been running holds the full rate until step 3,836 - long after the measured held-out
    /// NLL optimum at step 2,000.
    #[arg(long, default_value_t = 0)]
    pub schedule_budget: usize,
    /// Hard stop, in optimizer steps: the run shuts down CLEANLY at exactly this step - the
    /// interval's held-out sample evaluation, one final full-split validation pass, every
    /// report base and the manifest all flushed - instead of running to `--epochs` or
    /// `--preview-patience`. 0 is no cap.
    ///
    /// A run length and nothing else. The warmdown is shaped by `--schedule-budget` alone, so
    /// `--max-steps 4000 --schedule-budget 9590` runs the first 4,000 steps of exactly the
    /// trajectory a 9,590-step arm ran and its curves are step-matched against them; a cap
    /// that also reshaped the schedule would make every short arm a different experiment.
    ///
    /// A cap at or below the CUDA-graph capture warmup is refused: such an arm never runs a
    /// single captured step, so it does not measure the execution path every baseline ran.
    #[arg(long, default_value_t = 0)]
    pub max_steps: usize,
    /// Effective NorMuon rate for the block MLP-down matrices. `upstream-4x` is the faithful
    /// port of modded-nanogpt's tall `[4D, D]` storage plus `c_proj` `lr_mul = 2`;
    /// `aspect-only` is the quarter-rate our transposed `[D, 4D]` storage implies with no
    /// override. LR-only either way: no storage and no forward-pass change.
    #[arg(long, value_enum, default_value_t = MlpDownLr::AspectOnly)]
    pub mlp_down_lr: MlpDownLr,
    /// Keep every `K`-th training row of each ticker's chronological grid, with a per-ticker
    /// residue drawn from `--seed`. 1 keeps every row and is bit-for-bit today's pool.
    ///
    /// This is the knob that moves PER-OUTCOME MULTIPLICITY, and it is the one intervention in
    /// this family that moves the occupancy curve at all. Rows are enumerated at stride
    /// `pred_len` while each row supervises a dense origin lattice reaching `seq_len -
    /// patch_len·ceil(min_history/patch_len)` bars back, so every interior outcome is
    /// supervised by about `window/pred_len + 1` distinct rows per epoch - 30 at the production
    /// geometry. `K` divides that: `K = 30` leaves one exposure per outcome and a fully fresh
    /// sweep, `K = 4` leaves about 7.5. The retained rows still span every ticker and every
    /// regime, so this thins the OVERLAP, not the support.
    #[arg(long, default_value_t = 1)]
    pub row_stride_multiple: usize,
    /// Keep this fraction of the training pool, drawn uniformly at random over the whole pool.
    /// 1 keeps every row.
    ///
    /// A NEGATIVE CONTROL, and it is here because it is provably inert on occupancy. `n` rows
    /// drawn from a uniformly random `F·R`-subset are still a uniformly random `n`-subset of
    /// the original `R`, so the unseen probability `C(R-M, n)/C(R, n)` at a MATCHED STEP is
    /// identical to the full pool's: the `F`-fold cut in multiplicity cancels the `F`-fold cut
    /// in the pool exactly. An arm that moves under this knob moved for a reason that is not
    /// occupancy - epoch boundaries, block structure, regime mix - which is what makes it worth
    /// running beside `--row-stride-multiple` rather than instead of it.
    #[arg(long, default_value_t = 1.)]
    pub row_fraction: f64,
    /// Where each row's patch grid starts. `fixed` is today's corpus: every row's final origin
    /// is congruent to the same residue modulo `patch_len`, so all ~30 exposures of an outcome
    /// carry an IDENTICAL tokenization - same patch boundaries, same RoPE positions, same token
    /// count before the origin. `random` draws one offset in `0..patch_len` per row.
    ///
    /// It adds ORIGINS, not shocks. The phased lattice covers up to `patch_len` times as many
    /// absolute origins, but the corpus's information ceiling is the number of distinct
    /// realized return shocks and a re-tokenization cannot raise it: a phased origin's label
    /// `cum(A -> A+h)` is an exact difference of two lattice-supervised cumulative returns. So
    /// the support changes and the ceiling does not, which is exactly why this is a regularizer
    /// against tokenization-specific memorization and not a data fix.
    #[arg(long, value_enum, default_value_t = PatchPhase::Fixed)]
    pub patch_phase: PatchPhase,
}
impl TrainArgs {
    /// Every argument check the run can make before it touches the device or the corpus, and
    /// the resolved base learning rate. [`train`] calls this as its FIRST statement, so an
    /// arm configured wrong fails in the second it was submitted rather than after the
    /// minutes a `Corpus::load` of the 4,873-ticker universe costs.
    fn validate(&self) -> Result<f64> {
        self.model.validate()?;
        ensure!(
            self.epochs > 0
                && self.batch_size > 0
                && self.eval_every > 0
                && self.eval_origins > 0
                && self.patience > 0
                && self.preview_patience > 0,
            "training counts must be positive"
        );
        ensure!(
            self.max_steps == 0 || self.max_steps > CAPTURE_AFTER_STEPS,
            "--max-steps {} stops the run at or before the {CAPTURE_AFTER_STEPS}-step \
             graph-capture warmup, so the arm would never run a single captured step and \
             would not measure the execution path every baseline ran; pass 0 for no cap",
            self.max_steps
        );
        let base_learning_rate = self
            .learning_rate
            .unwrap_or_else(|| self.optimizer.default_learning_rate());
        ensure!(
            base_learning_rate.is_finite() && base_learning_rate > 0.,
            "learning rate must be positive"
        );
        ensure!(
            self.scalar_lr_mult.is_finite() && self.scalar_lr_mult > 0.,
            "the scalar learning-rate multiplier must be positive"
        );
        ensure!(
            self.common_context >= self.model.seq_len as usize,
            "common context must cover model history"
        );
        self.row_selection().validate()?;
        Ok(base_learning_rate)
    }

    /// The row-pool selection this arm trains on. The seed is `--seed`, so the retained pool
    /// is reproducible from the manifest and two arms claiming the same knobs at the same seed
    /// retain the identical rows.
    pub(super) fn row_selection(&self) -> RowSelection {
        RowSelection {
            fraction: self.row_fraction,
            stride_multiple: self.row_stride_multiple,
            patch_phase: self.patch_phase,
            seed: self.seed,
        }
    }
}
impl Default for TrainArgs {
    fn default() -> Self {
        Self {
            ticker: vec![],
            data_dir: crate::data::ingest::bars_dir(),
            run: None,
            model: ModelConfig::default(),
            common_context: 6000,
            epochs: 1,
            batch_size: 256,
            learning_rate: None,
            scalar_lr_mult: NANOGPT_SCALAR_LR_MULTIPLIER,
            optimizer: OptimizerKind::PolarExpress,
            seed: 20260905,
            eval_every: 1000,
            eval_origins: 2048,
            eval_batch_size: 64,
            in_period_sections: 0,
            patience: 3,
            preview_patience: 3,
            fused: true,
            market_min_cross_section: 2000,
            schedule_budget: 0,
            max_steps: 0,
            mlp_down_lr: MlpDownLr::AspectOnly,
            row_stride_multiple: 1,
            row_fraction: 1.,
            patch_phase: PatchPhase::Fixed,
        }
    }
}
#[derive(Clone, Debug, Args)]
pub struct EvaluateArgs {
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long,default_value_os_t=crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,
    /// How the validation and cross-section draws are PLACED. `strided` is the historical rule
    /// this project's every published trading number was measured under; `both` scores the
    /// checkpoint twice in one process and writes the anchored set to a subdirectory, which is
    /// the only form in which a placement change can be read as a placement change.
    #[arg(long, value_enum, default_value_t = Placement::Strided)]
    pub placement: Placement,
}

/// Which cross-section placement `evaluate` scores. Deliberately NOT a training flag: a
/// placement change moves every published cross-sectional number, so it earns its way into
/// training runs through the before/after table this enum exists to produce, not before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Placement {
    Strided,
    Anchored,
    Both,
}

#[derive(Clone, Debug, Args)]
pub struct PortfolioEvaluateArgs {
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long, default_value_os_t = crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,
    /// No calibration flag: the amplitude applied here is the CHECKPOINT's own, fitted in the
    /// run that wrote it on the reserved `[70%, 80%)` partition. A gain that could be supplied
    /// separately from the weights it was fitted on is a gain that can be paired with the wrong
    /// weights.
    #[command(flatten)]
    pub schedule: super::portfolio_data::ScheduleConfig,
    #[command(flatten)]
    pub(super) account: super::portfolio::PortfolioConfig,
}

/// Continuous synchronized account evaluation, not independent endpoint trade budgets.
pub fn evaluate_portfolio(args: PortfolioEvaluateArgs) -> Result<()> {
    let started = Instant::now();
    ensure!(args.batch_size > 0, "batch size must be positive");
    let (manifest, corpus, _store, model, device) =
        load_checkpoint(&args.checkpoint, &args.data_dir)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    let pairing = manifest.pairing(&corpus.contract)?;
    let mut result = super::portfolio_data::evaluate(
        &corpus,
        &model,
        device,
        pairing,
        &manifest.mean_gain,
        &args,
    )?;
    result.summary.insert("load_ms".into(), load_ms);
    result.summary.insert("total_ms".into(), started.elapsed().as_secs_f64() * 1000.);
    reports::write_account(&args.output, manifest.epoch, manifest.step, &result)?;
    Ok(())
}
/// The head layout and parameterization are part of this string, not a comment: the head weight
/// keeps its name and its `[pred_len·2·CHANNELS, HEAD_HIDDEN]` shape across the channel-major
/// rewrite, so a `v5` checkpoint loads into the current model without a single shape error and
/// silently means something else - row `c·pred_len + h` of the current layout held channel
/// `h % (2·CHANNELS)` of horizon `h / (2·CHANNELS)`. Measured consequence on
/// `timexer-market-neutral-20260906/weights/best`: the reported mean close coordinate is a
/// log-scale coordinate times `√h` (-0.69, -0.75, -0.74 at h = 16, 64, 192) and the close MSE
/// ratio reads 2.4-2.8 instead of ~0.99. Old checkpoints are not convertible: retrain.
/// `v7` was the modded-nanogpt recipe landing: gainless biasless RMSNorm everywhere (34 norm
/// tensors gone), the four block projections bias-free (32 bias tensors gone), three 1-D lambda
/// banks, one `skip_weights` logit vector and one `value_lambda` per non-source layer. A `v6`
/// state dict carries 66 tensors this model does not want and lacks 12 it needs, so it is not
/// loadable and not convertible: retrain.
/// `v8` splits that one stamp in two, because the x0 injection is now a mode
/// ([`X0Lambdas`]): under `disabled` the `lambdas.x0` bank does not exist, so the two modes
/// have DIFFERENT parameter sets and a state dict of one cannot load into the other. The stamp
/// names which set the weights are, and [`Manifest::read`] additionally requires the stamp to
/// agree with the mode the manifest declares - a hand-edited pair is refused rather than
/// half-loaded.
/// `v9` is the horizon-weighted objective ([`HorizonLoss`](super::model::HorizonLoss)). The parameter set is IDENTICAL to
/// `v8` - same tensors, same shapes, same names, and `--horizon-loss uniform` is bit-for-bit
/// the `v8` loss - so a `v8` state dict would load without a single shape error. The stamp
/// still moves, because the manifest is the only record of which horizons an arm's weights
/// were fitted to, and a `v8` manifest carries no `horizon_loss` field at all: without the
/// bump a `cutoff:32` checkpoint and a `uniform` one would be indistinguishable and their
/// curves would be silently averaged together.
/// `v10` is the structured mean ([`HorizonMean`]), and unlike `v9` it really does change the
/// parameter set: `basis:S:B` deletes `pred_len - S` free mean rows per channel from the head
/// output weight and adds `B` coefficient rows, so `head.output.weight` is
/// `[CHANNELS·(S + B + pred_len), 1024]` instead of `[2·CHANNELS·pred_len, 1024]` and no `v9`
/// state dict has that shape. `--horizon-mean free` is bit-for-bit the `v9` head, so its
/// tensors would load - but its manifest carries no `horizon_mean` field, so a `v9` checkpoint
/// cannot be attributed to a mean parameterization and is refused by name. The spec is part of
/// the stamp rather than a note beside it, because two `basis` arms with different `(S, B)`
/// have different parameter sets too.
/// `v11` is the in-run AMPLITUDE CALIBRATION. The parameter set is identical to `v10` again -
/// same tensors, same shapes, same names, and the gain is applied to the DECODED mean rather
/// than to any weight - but the manifest now carries a mandatory `mean_gain`
/// ([`FrozenGain`]), and what a checkpoint MEANS changes with it: a `v11` checkpoint emits a
/// calibrated mean the moment it is loaded, so its MSE, its trading diagnostics and its
/// portfolio sizing are not the same numbers a `v10` checkpoint of the same weights produced.
/// Both directions of the incompatibility are named rather than tolerated. Old into new: a
/// `v10` manifest fails the stamp check below, and even a hand-restamped one fails to
/// deserialize because `mean_gain` has no default - there is no such thing as a `v11`
/// checkpoint whose amplitude is unstated, and silently loading a `v10` checkpoint would score
/// an uncalibrated head under a calibrated build's labels. New into old: a build that predates
/// this field reads `v11` in `format` and refuses it by name, which is what stops it from
/// loading the weights, ignoring the gain, and reporting an over-amplitudinal forecast as
/// though it were the calibrated one. A checkpoint whose block identified no amplitude carries
/// the identity explicitly ([`FrozenGain::is_identity`]) - that is a measurement, and it is not
/// the same statement as a missing field.
const FORMAT_X0_LEARNED: &str =
    "causal-patch-ohlc-universe-v11-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres-mean-gain-x0-learned";
const FORMAT_X0_NONE: &str =
    "causal-patch-ohlc-universe-v11-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres-mean-gain-x0-none";
/// The stamp a parameter set carries: the x0 mode picks the base, the mean parameterization
/// suffixes it, and those two are the whole mapping because they are the only knobs that
/// change which tensors exist or what shape they are.
fn format_stamp(x0_lambdas: X0Lambdas, horizon_mean: HorizonMean) -> String {
    let base = match x0_lambdas {
        X0Lambdas::Enabled => FORMAT_X0_LEARNED,
        X0Lambdas::Disabled => FORMAT_X0_NONE,
    };
    format!("{base}-mean-{}", horizon_mean.stamp())
}
/// `_v5`: the objective no longer weights the 192 horizons equally - it weights them by
/// [`HorizonLoss`](super::model::HorizonLoss), and CHECKPOINT SELECTION now minimizes that same weighted NLL on the
/// held-out sample instead of the equal-weighted aggregate. Both halves change what a `_v4`
/// curve is comparable to: a `_v4` point is an equal-weighted objective selected on an
/// equal-weighted scalar, which is exactly the configuration jobs 5190-5193 refuted. The
/// targets, the mask and the per-element NLL are unchanged.
const OBJECTIVE: &str = "causal_patch_market_neutral_nll_v5";
const NUMERICS: &str =
    "fp32-masters-fp32-causal-origin-statistics-bf16-causal-SDPA-rope-fp32-decoder-fp64-prices-nanogpt-lr-cooldown-frac=.60-floor=.15";
/// Why a run stopped. Four exits, four different statements about the arm, and after the fact
/// nothing but this field distinguishes them: a `step-cap` arm stopped at an operator's stated
/// budget with its held-out curve possibly still descending, a `preview-patience` stop is the
/// model's own verdict that it was not, and an epoch exit is the corpus running out. A capped
/// arm read as an early-stopped one would be quoted as evidence about convergence it carries
/// none of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Termination {
    /// `--max-steps` reached. Implies nothing about the fit.
    StepCap,
    /// `--preview-patience` consecutive held-out sample evaluations without improvement.
    PreviewPatience,
    /// `--patience` complete epochs without improved held-out full NLL.
    EpochPatience,
    /// `--epochs` epochs completed: the run length the arm asked for.
    EpochLimit,
}
/// Every run-length rule in one place, so the step loop carries no stopping policy of its own
/// and the policy is testable without a device: which steps report, and which of the four
/// exits an evaluation has reached.
///
/// The learning-rate schedule is deliberately NOT in here - see [`schedule_budget`].
#[derive(Clone, Copy, Debug)]
struct StopRules {
    /// `--max-steps`, `None` when the run is uncapped.
    max_steps: Option<usize>,
    eval_every: usize,
    preview_patience: usize,
    patience: usize,
    planned_epochs: usize,
}
/// What the step that just ran observed.
#[derive(Clone, Copy, Debug)]
struct Progress {
    step: usize,
    epoch: usize,
    epoch_complete: bool,
    stale_previews: usize,
    stale_epochs: usize,
}
impl StopRules {
    fn new(args: &TrainArgs) -> Self {
        Self {
            max_steps: (args.max_steps > 0).then_some(args.max_steps),
            eval_every: args.eval_every,
            preview_patience: args.preview_patience,
            patience: args.patience,
            planned_epochs: args.epochs,
        }
    }
    /// Whether the step that just ran must evaluate, report and checkpoint. The cap forces an
    /// interval of its own: a capped run whose last steps went unreported would be missing
    /// exactly the interval the cap was chosen to measure.
    fn evaluates(self, step: usize, epoch_complete: bool) -> bool {
        step % self.eval_every == 0 || epoch_complete || self.capped(step)
    }
    /// A floor, not an equality, so a cap that no evaluation lands exactly on still stops the
    /// run instead of leaking into the epoch limit.
    fn capped(self, step: usize) -> bool {
        self.max_steps.is_some_and(|cap| step >= cap)
    }
    /// The exit this evaluation has reached, `None` to keep training. Precedence is the order
    /// the reasons are checked: a stalled arm is an early stop even when the cap lands on the
    /// same evaluation, because "the held-out NLL stopped improving" is the stronger statement
    /// about the arm, and the cap outranks both epoch exits for the same reason.
    fn termination(self, at: Progress) -> Option<Termination> {
        if at.stale_previews >= self.preview_patience {
            return Some(Termination::PreviewPatience);
        }
        if self.capped(at.step) {
            return Some(Termination::StepCap);
        }
        if !at.epoch_complete {
            return None;
        }
        if at.stale_epochs >= self.patience {
            return Some(Termination::EpochPatience);
        }
        (at.epoch >= self.planned_epochs).then_some(Termination::EpochLimit)
    }
}
/// The learning-rate schedule's endpoint, in optimizer steps: `--schedule-budget` where it is
/// stated, and the whole planned run where it is 0, which is what every arm before that knob
/// existed did by accident. `--max-steps` is absent from the inputs on purpose, and a test
/// pins the absence: the cap is a run length, and a run length that reshaped the warmdown
/// would make a short arm's first N steps a different trajectory from the long arms' first N.
fn schedule_budget(args: &TrainArgs, steps_per_epoch: usize) -> usize {
    match args.schedule_budget {
        0 => steps_per_epoch * args.epochs,
        budget => budget,
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Manifest {
    format: String,
    pub(super) model: ModelConfig,
    pub(super) data: CorpusContract,
    objective: String,
    numerics: String,
    requested_tickers: Vec<String>,
    seed: u64,
    pub(super) epoch: usize,
    pub(super) step: usize,
    completed_origins: usize,
    completed_target_bars: usize,
    epoch_complete: bool,
    planned_epochs: usize,
    /// The `--max-steps` cap in force, `None` when the run was uncapped. Skipped where absent
    /// so an uncapped run's manifest, and the digest authenticating it, is byte-identical to
    /// one written before the cap existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_steps: Option<usize>,
    /// The row-pool diversity selection in force, `None` on an arm that trained on the whole
    /// pool at one shared patch phase. Skipped where absent, so a control run's manifest and
    /// the digest authenticating it are byte-identical to one written before these knobs
    /// existed and every checkpoint on disk still loads.
    ///
    /// It is stamped HERE and not in [`CorpusContract`] on purpose. None of these knobs touches
    /// ticker eligibility, the market grid, the partition boundaries or any held-out draw -
    /// they subset and re-phase `Corpus::train_refs`, which no artifact is cached against - so
    /// putting them in the contract would invalidate the bar-audit and market-grid caches, and
    /// force a cold rescan, for a change no cached artifact is a function of. What they DO
    /// change is which rows this checkpoint's weights were fitted to, which is a property of
    /// the run, and that is what a manifest is for. The seed rides along, so a pairing that
    /// claims the same knobs but drew a different pool is detectable rather than assumed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    row_selection: Option<RowSelection>,
    /// Why the run stopped, `None` in a checkpoint written while it was still running - every
    /// `preview-latest` but the last. This is what keeps a capped arm from being read as one
    /// that early-stopped on patience or finished its epoch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    termination: Option<Termination>,
    batch_size: usize,
    /// The rate at the step this checkpoint was written - `base_learning_rate` times the
    /// nanogpt cooldown scale - so it is not the run's configuration.
    learning_rate: f64,
    /// The run's configured base AdamW rate, the ONE global learning-rate knob (the NorMuon
    /// rate is proportional to it), so a checkpoint from an lr sweep is attributable to its arm.
    base_learning_rate: f64,
    /// The AdamW learning-rate multiplier the scalar banks were trained at.
    scalar_lr_mult: f64,
    fused: bool,
    optimizer: OptimizerKind,
    optimizer_recipe: String,
    eval_every: usize,
    eval_origins: usize,
    validation_nll: f64,
    validation_mse: f64,
    validation_is_full: bool,
    /// Step and held-out sample OBJECTIVE-weighted NLL of the checkpoint `weights/best` points
    /// at; `None` before the first held-out sample evaluation. Renamed from `best_preview_nll`
    /// with `_v5`: the quantity minimized here is the training objective's weighted NLL, not
    /// the equal-weighted aggregate the old name meant, and a silently redefined field is
    /// worse than a renamed one.
    best_step: Option<usize>,
    best_objective_nll: Option<f64>,
    /// The selection criterion, spelled out. `horizon_loss` lives in `model`; this states what
    /// was DONE with it, so a manifest answers "what was this checkpoint chosen to be good at"
    /// without the reader having to know the build.
    selection: String,
    /// The amplitude calibration of the emitted MEAN this checkpoint carries, fitted on the
    /// reserved `[70%, 80%)` partition at the evaluation this checkpoint was written at, and
    /// applied by [`load_checkpoint`] to every consumer. Mandatory and never defaulted: an
    /// unstated amplitude is exactly the defect this field exists to end, and the manifest
    /// digest below authenticates the curve along with everything else, so a gain cannot be
    /// re-aimed at another checkpoint by editing a file.
    ///
    /// NOT part of the selection criterion. `best_objective_nll` above is the UN-GAINED
    /// model's held-out sample NLL, because a mean gain with no matching σ refit makes NLL
    /// worse even where it makes MSE better, and moving the selection scalar mid-experiment
    /// would break comparability against every checkpoint already on disk.
    mean_gain: FrozenGain,
    weights_sha256: String,
    manifest_sha256: String,
}
impl Manifest {
    fn digest(&self) -> Result<String> {
        let mut copy = self.clone();
        copy.manifest_sha256.clear();
        Ok(
            ring::digest::digest(&ring::digest::SHA256, &serde_json::to_vec(&copy)?)
                .as_ref()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
        )
    }
    fn read(directory: &Path) -> Result<Self> {
        let raw: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("manifest.json"))?)?;
        let field = |name: &str| {
            raw.get(name)
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
        };
        let stamp = field("format").to_owned();
        // The family first, so a stale stamp gets named rather than reported as a mean-mode
        // mismatch: every accepted stamp is one of the two bases plus a `-mean-<spec>` suffix,
        // and the exact suffix is cross-checked against the manifest's own config below.
        ensure!(
            stamp.starts_with(FORMAT_X0_LEARNED) || stamp.starts_with(FORMAT_X0_NONE),
            "unsupported universe checkpoint format {stamp:?}; this build reads {FORMAT_X0_LEARNED}-mean-<spec> and {FORMAT_X0_NONE}-mean-<spec>"
        );
        ensure!(
            field("objective") == OBJECTIVE,
            "checkpoint objective {:?} is not {OBJECTIVE}; checkpoints trained on another objective cannot be loaded",
            field("objective")
        );
        ensure!(
            field("numerics") == NUMERICS,
            "checkpoint numerics {:?} differ from {NUMERICS}",
            field("numerics")
        );
        let manifest: Self = serde_json::from_value(raw)?;
        manifest.model.validate()?;
        ensure!(
            stamp == format_stamp(manifest.model.x0_lambdas, manifest.model.horizon_mean),
            "checkpoint format {stamp:?} does not match the {:?} x0 mode and {} mean its own \
             model config declares, whose parameter set is stamped {}",
            manifest.model.x0_lambdas,
            manifest.model.horizon_mean,
            format_stamp(manifest.model.x0_lambdas, manifest.model.horizon_mean)
        );
        ensure!(
            !manifest.optimizer_recipe.is_empty(),
            "missing authenticated training optimizer recipe"
        );
        ensure!(
            manifest.selection == selection_criterion(&manifest.model),
            "checkpoint selection {:?} does not match the {} objective its own model config \
             declares, whose criterion is {:?}",
            manifest.selection,
            manifest.model.horizon_loss,
            selection_criterion(&manifest.model)
        );
        ensure!(
            manifest.manifest_sha256 == manifest.digest()?,
            "checkpoint manifest authentication failed"
        );
        ensure!(
            manifest.weights_sha256 == file_sha256(directory.join("model.safetensors"))?,
            "checkpoint weight authentication failed"
        );
        ensure!(
            manifest.model.seq_len as usize == manifest.data.context
                && manifest.model.pred_len as usize == manifest.data.pred_len
                && manifest.model.features == manifest.data.features,
            "model/data contract mismatch"
        );
        // The structural half of the `v11` guarantee, for the manifest a version string cannot
        // catch: one hand-edited into shape, or one written by a build that changed what the
        // curve means without moving the stamp. A wrong-length or nonpositive curve is named
        // here rather than at the first decode.
        manifest
            .mean_gain
            .validate(manifest.data.pred_len)
            .context("the checkpoint's authenticated mean gain is not applicable to its own \
                      forecast horizon")?;
        Ok(manifest)
    }
    /// Everything a frozen post-hoc calibration has to be pinned to. The digests are the
    /// authenticated ones this manifest already carries, so a calibration inherits the
    /// checkpoint's own integrity guard rather than adding a second, weaker one; the corpus
    /// half is a digest over the whole contract because a calibration fitted against one
    /// universe, split boundary or market construction describes a different `β̂` even at
    /// identical weights.
    fn pairing(&self, corpus: &CorpusContract) -> Result<Pairing> {
        Ok(Pairing {
            checkpoint_format: self.format.clone(),
            objective: self.objective.clone(),
            weights_sha256: self.weights_sha256.clone(),
            manifest_sha256: self.manifest_sha256.clone(),
            step: self.step,
            pred_len: self.data.pred_len,
            corpus_schema: corpus.schema.clone(),
            corpus_sha256: ring::digest::digest(
                &ring::digest::SHA256,
                &serde_json::to_vec(corpus)?,
            )
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
        })
    }
}
/// What `weights/best` is chosen to minimize, stamped into every manifest. Selection and the
/// gradient read the SAME horizon weighting by construction: both come from
/// `model.horizon_weights()`, so no arm can be selected on an objective it was not trained on.
///
/// The amplitude prior is part of the minimized quantity, so it is named here too - and only
/// when it is on, so a control manifest written before the knob existed still authenticates
/// against the identical string. `Manifest::read` cross-checks this against the manifest's own
/// `model`, which is what makes a penalized arm impossible to read as an unpenalized one.
fn selection_criterion(model: &ModelConfig) -> String {
    let base = format!(
        "min held-out sample objective-weighted NLL (horizon-loss={})",
        model.horizon_loss
    );
    if model.amplitude_prior > 0. {
        format!(
            "{base} plus mean-amplitude prior (amplitude-prior lambda {})",
            model.amplitude_prior
        )
    } else {
        base
    }
}
/// Structural guard on the head weight's row layout, for the checkpoints a version string
/// cannot catch: one stamped by hand, or one written by a build that changed the layout without
/// bumping [`format_stamp`].
///
/// The head weight varies smoothly along the forecast horizon and discontinuously across
/// channels - the four candle coordinates and the four log scales live on different scales
/// entirely. So in the correct channel-major layout consecutive rows are adjacent horizons of
/// one channel, while in the stale horizon-major layout consecutive rows are eight different
/// channels of one horizon and it is stride `2·CHANNELS` that walks the horizon. The mean
/// row-to-row cosine therefore peaks at stride 1 exactly when the layout matches the reshape in
/// `CausalPatchModel::forward`. On `timexer-market-neutral-20260906/weights/best` it is 0.285 at
/// stride 1 against 0.956 at stride 8, and the per-channel weight norms separate (2.2, 4.1, 7.5,
/// 3.1 for the coordinates) only under the horizon-major reading.
///
/// The probe is silent when neither stride shows smoothness (`max < 0.5`), which is the case for
/// the zero-initialised head before it has trained: an untrained head carries no layout to
/// check, and inventing a verdict from noise would be worse than the silence.
fn check_head_layout(weight: &Tensor) -> Result<()> {
    let rows = weight.size()[0];
    let stride = 2 * CHANNELS;
    ensure!(
        rows > 2 * stride,
        "head output weight has {rows} rows, too few to carry a horizon"
    );
    let unit = weight
        / weight
            .square()
            .sum_dim_intlist([1i64].as_slice(), true, Kind::Float)
            .sqrt()
            .clamp_min(f64::from(f32::MIN_POSITIVE));
    let cosine = |lag: i64| {
        (unit.narrow(0, 0, rows - lag) * unit.narrow(0, lag, rows - lag))
            .sum(Kind::Double)
            .double_value(&[])
            / (rows - lag) as f64
    };
    let (adjacent, strided) = (cosine(1), cosine(stride));
    ensure!(
        adjacent.max(strided) < 0.5 || adjacent > strided,
        "head weight rows are horizon-major, not channel-major: mean row cosine is {strided:.3} at stride {stride} against {adjacent:.3} at stride 1, so this checkpoint predates the channel-major head. There is no conversion - retrain under {FORMAT_X0_LEARNED} or {FORMAT_X0_NONE}"
    );
    Ok(())
}
fn save(directory: &Path, store: &nn::VarStore, mut manifest: Manifest) -> Result<()> {
    fs::create_dir_all(directory)?;
    let temp = directory.join("pending.safetensors");
    store.save(&temp)?;
    manifest.weights_sha256 = file_sha256(&temp)?;
    manifest.manifest_sha256 = manifest.digest()?;
    fs::write(
        directory.join("manifest.pending.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    fs::rename(temp, directory.join("model.safetensors"))?;
    fs::rename(
        directory.join("manifest.pending.json"),
        directory.join("manifest.json"),
    )?;
    Ok(())
}
fn fixed_origins<T: Copy>(origins: &[T], count: usize) -> Result<Vec<T>> {
    ensure!(
        count > 0 && !origins.is_empty(),
        "evaluation requires nonempty origins"
    );
    let count = count.min(origins.len());
    Ok((0..count)
        .map(|i| {
            origins[if count == 1 {
                origins.len() / 2
            } else {
                i * (origins.len() - 1) / (count - 1)
            }]
        })
        .collect())
}
/// The `held-out cross-section` draw: whole timestamp blocks out of `origins`, for the trading
/// family only. See [`CROSS_SECTION_TICKERS`] for why this exists beside the held-out sample
/// rather than replacing it, and why the width is negotiated rather than demanded.
///
/// The host cost is one memory-mapped bar header per candidate window, the same lookup
/// [`score`] already does for its own grouping.
pub(super) fn cross_section_origins(corpus: &Corpus, origins: &[WindowRef]) -> Result<Vec<WindowRef>> {
    let stamped: Vec<(i64, WindowRef)> = origins
        .iter()
        .map(|reference| {
            (
                corpus.ticker(*reference).timestamp(reference.origin),
                *reference,
            )
        })
        .collect();
    cross_section_blocks(&stamped)
}
/// The draw itself, over `(timestamp, window)` pairs, so the block selection is testable
/// without a corpus behind it.
///
/// Widest-first: sorting the candidate timestamps by member count descending makes the `k`-th
/// entry the widest UNIFORM draw any `k`-timestamp choice can support, so one pass over `k`
/// evaluates every uniform operating point the corpus offers and keeps the one carrying the
/// most information, `k · (width - 1)`. Ties in width keep timestamp order because
/// [`slice::sort_by`] is stable and the [`BTreeMap`] hands them over ascending, so the draw is
/// a deterministic function of the universe - fixed across steps and across runs, which is
/// what makes the series step-matchable.
///
/// Within a chosen timestamp the tickers are strided by [`fixed_origins`] over the corpus's
/// own stable index order, and the chosen timestamps are restored to ascending order so the
/// draw reads over the held-out period in time.
fn cross_section_blocks(stamped: &[(i64, WindowRef)]) -> Result<Vec<WindowRef>> {
    let mut blocks: BTreeMap<i64, Vec<WindowRef>> = BTreeMap::new();
    for (stamp, reference) in stamped {
        blocks.entry(*stamp).or_default().push(*reference);
    }
    let mut candidates: Vec<(i64, Vec<WindowRef>)> = blocks
        .into_iter()
        .filter(|(_, members)| members.len() >= CROSS_SECTION_FLOOR)
        .collect();
    ensure!(
        !candidates.is_empty(),
        "no held-out evaluation timestamp holds the {CROSS_SECTION_FLOOR} tickers a \
         cross-sectional trading measurement needs at all; the trading family cannot be \
         scored on this corpus"
    );
    candidates.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    let (mut count, mut width, mut information) = (0usize, 0usize, 0usize);
    for k in 1..=candidates.len().min(CROSS_SECTION_TIMESTAMPS) {
        let uniform = candidates[k - 1].1.len().min(CROSS_SECTION_TICKERS);
        if k * (uniform - 1) > information {
            (count, width, information) = (k, uniform, k * (uniform - 1));
        }
    }
    candidates.truncate(count);
    candidates.sort_by_key(|(stamp, _)| *stamp);
    let mut drawn = Vec::with_capacity(count * width);
    for (_, members) in &candidates {
        drawn.extend(fixed_origins(members, width)?);
    }
    Ok(drawn)
}

/// The training loop's host loader: one background thread building the next packed batch
/// while the device executes the current step, with one request and one ready batch in
/// flight. `benchmark::loader_arm` drives the same structure so that its timed arm differs
/// from the pinned-host arm in nothing but where the bytes came from.
pub(super) struct Prefetcher {
    requests: Option<mpsc::SyncSender<Vec<WindowRef>>>,
    ready: Option<mpsc::Receiver<Result<(Batch, f64)>>>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Prefetcher {
    pub(super) fn new(corpus: Arc<Corpus>) -> Self {
        Self::with_mode(corpus, None)
    }
    pub(super) fn with_forecasts(corpus: Arc<Corpus>, include_calibration_targets: bool) -> Self {
        Self::with_mode(corpus, Some(include_calibration_targets))
    }
    fn with_mode(corpus: Arc<Corpus>, projected: Option<bool>) -> Self {
        let (requests, incoming) = mpsc::sync_channel::<Vec<WindowRef>>(1);
        let (outgoing, ready) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("timexer-prefetch".into())
            .spawn(move || {
                while let Ok(refs) = incoming.recv() {
                    // Timed HERE, on the thread that does the work. The main thread's wait
                    // measures how much of this failed to overlap the device, which is a
                    // different question and is usually all of it without costing anything.
                    let started = Instant::now();
                    let batch = if let Some(include_targets) = projected {
                        corpus.host_forecast_batch(&refs, include_targets)
                    } else {
                        corpus.host_batch(&refs)
                    };
                    let build_ms = started.elapsed().as_secs_f64() * 1000.;
                    if outgoing.send(batch.map(|batch| (batch, build_ms))).is_err() {
                        break;
                    }
                }
            })
            .expect("starting TimeXer prefetch worker");
        Self {
            requests: Some(requests),
            ready: Some(ready),
            worker: Some(worker),
        }
    }
    pub(super) fn request(&self, refs: &[WindowRef]) -> Result<()> {
        self.requests
            .as_ref()
            .unwrap()
            .send(refs.to_vec())
            .context("prefetch worker stopped")
    }
    /// The next batch and the milliseconds the loader thread spent assembling it.
    pub(super) fn receive(&self) -> Result<(Batch, f64)> {
        self.ready
            .as_ref()
            .unwrap()
            .recv()
            .context("prefetch worker stopped")?
    }
}
impl Drop for Prefetcher {
    fn drop(&mut self) {
        self.requests.take();
        self.ready.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Final-origin held-out scores. `relative` space is the training objective: market-neutral
/// targets, persistence `ŷ = 0`. `absolute` space adds the realized market drift back to the
/// targets while the model's forecast stays `ŷ` (market forecast zero), so its MSE ratio is
/// comparable with runs trained on raw returns. NLL, calibration, and the per-horizon
/// robustness diagnostics are relative only.
struct Evaluation {
    nll: f64,
    /// The held-out NLL under the TRAINING objective's horizon weighting, `Σ w·mask·nll /
    /// Σ w·mask·CHANNELS`. This is what selects checkpoints and drives early stopping; `nll`
    /// above stays the equal-weighted aggregate every report charts, so the two are the same
    /// number under `--horizon-loss uniform` and deliberately different otherwise.
    objective_nll: f64,
    persistence_nll: f64,
    mse: f64,
    persistence_mse: f64,
    absolute_mse: f64,
    absolute_persistence_mse: f64,
    within_1_sigma: f64,
    within_2_sigma: f64,
    rmse_price: f64,
    mae_price: f64,
    invalid_fraction: f64,
    tail_loss_share: f64,
    median_window_ratio: f64,
    horizon: HorizonCurve,
    trading: TradingCurve,
    portfolio: PortfolioCurve,
    /// Per-(horizon, channel) amplitude moments of this split's own emitted mean. Everything
    /// the calibration fit, the un-gained MSE ratio and the GAINED MSE ratio are functions of,
    /// which is why no split needs a second scoring pass to report what a gain would do to it.
    amplitude: Moments,
    timing: EvalTiming,
}
fn invalid_candles(prices: &Tensor) -> Tensor {
    let open = prices.select(-1, 0);
    let high = prices.select(-1, 1);
    let low = prices.select(-1, 2);
    let close = prices.select(-1, 3);
    high.lt_tensor(&open.maximum(&close))
        .logical_or(&low.gt_tensor(&open.minimum(&close)))
        .logical_or(&low.gt_tensor(&high))
        .logical_or(&prices.le(0.).any_dim(-1, false))
        .logical_or(&prices.isfinite().logical_not().any_dim(-1, false))
}
/// Final-origin forecast of every row, `[batch, pred_len, 4|1]` unless noted: σ-scaled relative
/// point forecast, log predictive scale, relative targets with their mask, the market drift in σ
/// units that separates relative from absolute targets, decoded absolute prices (`ŷ`, market
/// forecast zero), the same forecast re-based onto the realized market path for charts, and
/// observed prices. `mid` and `sigma` are `[batch]`: the origin bar's log high/low midpoint in σ
/// units (the microstructure anchor the trade close is bounced away from) and the row's causal
/// return σ, which converts a σ-scaled coordinate back into a log return.
pub(super) struct FinalOrigin {
    pub(super) scaled: Tensor,
    pub(super) log_scale: Tensor,
    pub(super) targets: Tensor,
    pub(super) mask: Tensor,
    pub(super) drift: Tensor,
    pub(super) prices: Tensor,
    pub(super) rebased_prices: Tensor,
    pub(super) target_prices: Tensor,
    pub(super) mid: Tensor,
    pub(super) sigma: Tensor,
}
/// The model is channel-major (`[rows, origins', channel, bar]`) because every training-loss
/// consumer slices a channel; the scoring accumulators below are channel-last, and at the final
/// origin the whole forecast is `rows·pred_len·4` elements, so one transpose here is free.
pub(super) fn final_origin(model: &CausalPatchModel, batch: &Batch) -> FinalOrigin {
    let stats = model.statistics(batch);
    let head = model.forward(batch, &stats, false, true);
    let output = model.output(&head);
    let last = stats.last();
    let bars = model.config().pred_len;
    let channel_last = |tensor: &Tensor| tensor.squeeze_dim(1).transpose(1, 2).contiguous();
    let scaled = channel_last(&model.decode(&output, &last));
    let (targets, mask) = model.targets(batch, &last, true);
    let drift = model
        .market_drift(batch, &last, true)
        .reshape([-1, bars, 1]);
    let anchor = batch.anchor.reshape([-1, 1, 1]);
    let sigma = last.sigma.reshape([-1, 1, 1]);
    let targets = channel_last(&targets);
    let origin_bar = batch.log_prices.select(1, model.config().seq_len - 1);
    let row_sigma = last.sigma.reshape([-1]);
    FinalOrigin {
        prices: decode_prices(&scaled, &anchor, &sigma),
        rebased_prices: decode_prices(&(&scaled + &drift), &anchor, &sigma),
        target_prices: &anchor * (&sigma * (&targets + &drift)).exp(),
        mid: (origin_bar.select(-1, 1) + origin_bar.select(-1, 2)) * 0.5 / &row_sigma,
        sigma: row_sigma,
        scaled,
        log_scale: channel_last(&output.log_scale),
        targets,
        mask: mask.reshape([-1, bars]),
        drift,
    }
}
/// Device-resident final-origin accumulators. Every batch adds masked tensor reductions: the
/// twelve aggregate `sums`, eleven per-horizon rows, and per-(window, bar) close-channel state
/// for the tail-trimmed ratio, the window medians and the whole trading-diagnostic family.
/// Nothing crosses to the host until `finish`, so batches queue behind each other without a
/// synchronization point.
///
/// `bar_forecast`/`bar_target` hold masked neutral close coordinates for forecast-quality
/// diagnostics, grouped by the dense origin-timestamp rank in `groups[window]`.
/// Separate unmasked policy forecasts and predictive scales preserve outcome-independent
/// decisions; raw close targets and next-open entry returns are used only to score payoffs.
struct Scorer {
    pred_len: i64,
    half_log_horizon: Tensor,
    /// The training objective's per-horizon weight, `[1, pred_len, 1]` fp32 to broadcast over
    /// the channel-last scoring layout. Selection reads the SAME vector the gradient does, so
    /// an arm is never selected on an objective it was not trained on.
    horizon_weight: Tensor,
    sums: Tensor,
    horizon_sums: Tensor,
    /// `[AMPLITUDE_ROWS, pred_len, CHANNELS]` f64: the eight masked column reductions the
    /// amplitude calibration and every gained-versus-un-gained ratio are computed from. The
    /// existing close-channel accumulator below is a per-(window, bar) ARRAY because the rank,
    /// decile and cross-section statistics need the individual rows; these are pooled sums, so
    /// generalizing them to all four channels costs eight column reductions of tensors the
    /// pass already has resident rather than four copies of that array.
    amplitude: Tensor,
    bar_squared: Tensor,
    bar_persistence: Tensor,
    bar_forecast: Tensor,
    bar_target: Tensor,
    bar_valid: Tensor,
    bar_policy_forecast: Tensor,
    bar_log_scale: Tensor,
    bar_raw_target: Tensor,
    window_mid: Tensor,
    window_sigma: Tensor,
    window_entry_log: Tensor,
    groups: Tensor,
    group_count: i64,
    bars: usize,
    filled: i64,
}
impl Scorer {
    fn new(
        half_log_horizon: &Tensor,
        horizon_weight: &[f64],
        groups: &Tensor,
        group_count: usize,
        pred_len: i64,
        device: Device,
    ) -> Self {
        assert_eq!(
            horizon_weight.len() as i64,
            pred_len,
            "the objective weight vector must cover the horizon exactly"
        );
        let origins = groups.size()[0];
        let shape = [origins, pred_len];
        let bars = |kind| Tensor::zeros(shape, (kind, device));
        Self {
            pred_len,
            half_log_horizon: half_log_horizon.reshape([1, pred_len, 1]).to_device(device),
            horizon_weight: Tensor::from_slice(
                &horizon_weight
                    .iter()
                    .map(|weight| *weight as f32)
                    .collect::<Vec<f32>>(),
            )
            .reshape([1, pred_len, 1])
            .to_device(device),
            sums: Tensor::zeros([14], (Kind::Double, device)),
            horizon_sums: Tensor::zeros([13, pred_len], (Kind::Double, device)),
            amplitude: Tensor::zeros(
                [AMPLITUDE_ROWS, pred_len, CHANNELS],
                (Kind::Double, device),
            ),
            bar_squared: bars(Kind::Float),
            bar_persistence: bars(Kind::Float),
            bar_forecast: bars(Kind::Float),
            bar_target: bars(Kind::Float),
            bar_valid: bars(Kind::Float),
            bar_policy_forecast: bars(Kind::Float),
            bar_log_scale: bars(Kind::Float),
            bar_raw_target: bars(Kind::Float),
            window_mid: Tensor::zeros([origins], (Kind::Float, device)),
            window_sigma: Tensor::zeros([origins], (Kind::Float, device)),
            window_entry_log: Tensor::zeros([origins], (Kind::Float, device)),
            groups: groups.to_device(device).to_kind(Kind::Int64),
            group_count: group_count as i64,
            bars: 0,
            filled: 0,
        }
    }
    fn accumulate(&mut self, forecast: &FinalOrigin, valid_target_bars: usize) {
        let close = CHANNELS - 1;
        let tail_count = (valid_target_bars * CHANNELS as usize).div_ceil(100) as i64;
        let mask = forecast.mask.unsqueeze(-1);
        let residual = &forecast.targets - &forecast.scaled;
        let squared = residual.square() * &mask;
        let persistence = forecast.targets.square() * &mask;
        let absolute_targets = &forecast.targets + &forecast.drift;
        let absolute_squared = (&absolute_targets - &forecast.scaled).square() * &mask;
        let absolute_persistence = absolute_targets.square() * &mask;
        let nll = nll_elements(&forecast.scaled, &forecast.log_scale, &forecast.targets) * &mask;
        let baseline = nll_elements(
            &forecast.targets.zeros_like(),
            &self.half_log_horizon,
            &forecast.targets,
        ) * &mask;
        let scale = forecast.log_scale.exp();
        let deviation = residual.abs();
        let within_1 = deviation.le_tensor(&scale).to_kind(Kind::Float) * &mask;
        let within_2 = deviation.le_tensor(&(scale * 1.96)).to_kind(Kind::Float) * &mask;
        let price_errors = (&forecast.prices - &forecast.target_prices) * &mask;
        let invalid = invalid_candles(&forecast.prices);
        let absolute = &deviation * &mask;
        let target_abs = forecast.targets.abs() * &mask;
        let magnitude = target_abs.flatten(0, -1);
        let (top, _) = magnitude.topk(tail_count.min(magnitude.size()[0]), 0, true, false);
        let tail = magnitude
            .ge_tensor(&top.min())
            .reshape_as(&squared)
            .to_kind(Kind::Float);
        let close_target = forecast.targets.select(-1, close);
        let close_forecast = forecast.scaled.select(-1, close);
        let win = (&close_target - &close_forecast)
            .abs()
            .lt_tensor(&close_target.abs())
            .to_kind(Kind::Float)
            * &forecast.mask;
        let predicted = close_forecast.ne(0.).to_kind(Kind::Float) * &forecast.mask;
        let hit = close_forecast
            .sign()
            .eq_tensor(&close_target.sign())
            .to_kind(Kind::Float)
            * &predicted;
        let up = close_forecast.gt(0.).to_kind(Kind::Float) * &forecast.mask;
        let rows = squared.size()[0];
        let channel_sum = |t: &Tensor| t.sum_dim_intlist([-1i64].as_slice(), false, Kind::Float);
        self.bar_squared
            .narrow(0, self.filled, rows)
            .copy_(&channel_sum(&squared));
        self.bar_persistence
            .narrow(0, self.filled, rows)
            .copy_(&channel_sum(&persistence));
        self.bar_forecast
            .narrow(0, self.filled, rows)
            .copy_(&(&close_forecast * &forecast.mask));
        self.bar_target
            .narrow(0, self.filled, rows)
            .copy_(&(&close_target * &forecast.mask));
        self.bar_valid
            .narrow(0, self.filled, rows)
            .copy_(&forecast.mask);
        self.bar_policy_forecast
            .narrow(0, self.filled, rows)
            .copy_(&close_forecast);
        self.bar_log_scale
            .narrow(0, self.filled, rows)
            .copy_(&forecast.log_scale.select(-1, close));
        self.bar_raw_target
            .narrow(0, self.filled, rows)
            .copy_(&absolute_targets.select(-1, close));
        // First future bar is the next observed bar, not necessarily the next UTC interval.
        // Its open is an outcome used for payoff accounting, never a decision input.
        let entry_log = (&forecast.sigma * absolute_targets.select(1, 0).select(-1, 0))
            .masked_fill(&forecast.mask.select(1, 0).eq(0.), f64::NAN);
        self.window_entry_log
            .narrow(0, self.filled, rows)
            .copy_(&entry_log);
        self.window_mid
            .narrow(0, self.filled, rows)
            .copy_(&forecast.mid);
        self.window_sigma
            .narrow(0, self.filled, rows)
            .copy_(&forecast.sigma);
        self.filled += rows;
        self.bars += valid_target_bars;
        self.sums += Tensor::stack(
            &[
                squared.sum(Kind::Double),
                persistence.sum(Kind::Double),
                price_errors.square().sum(Kind::Double),
                price_errors.abs().sum(Kind::Double),
                (invalid.to_kind(Kind::Float) * &forecast.mask).sum(Kind::Double),
                (&squared * tail).sum(Kind::Double),
                nll.sum(Kind::Double),
                baseline.sum(Kind::Double),
                within_1.sum(Kind::Double),
                within_2.sum(Kind::Double),
                absolute_squared.sum(Kind::Double),
                absolute_persistence.sum(Kind::Double),
                // The selection scalar's numerator and denominator. `nll` is already
                // `nll_elements · mask`, so this is `Σ w·mask·nll` and `Σ w·mask·CHANNELS` -
                // the same weighted mean `CausalPatchModel::losses` minimizes, evaluated on
                // the held-out split. At `uniform` the weight is exactly 1, so these are
                // `sums[6]` and `bars·CHANNELS` and the scalar is the aggregate NLL unchanged.
                (&nll * &self.horizon_weight).sum(Kind::Double),
                (&mask * &self.horizon_weight).sum(Kind::Double) * CHANNELS as f64,
            ],
            0,
        );
        self.horizon_sums += Tensor::stack(
            &[
                squared.sum_dim_intlist([0i64, 2].as_slice(), false, Kind::Double),
                persistence.sum_dim_intlist([0i64, 2].as_slice(), false, Kind::Double),
                forecast
                    .mask
                    .sum_dim_intlist([0i64].as_slice(), false, Kind::Double)
                    * CHANNELS,
                absolute.sum_dim_intlist([0i64, 2].as_slice(), false, Kind::Double),
                target_abs.sum_dim_intlist([0i64, 2].as_slice(), false, Kind::Double),
                win.sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
                predicted.sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
                hit.sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
                up.sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
                absolute_squared.sum_dim_intlist([0i64, 2].as_slice(), false, Kind::Double),
                absolute_persistence.sum_dim_intlist([0i64, 2].as_slice(), false, Kind::Double),
                // The two coverage rows the aggregate `sums[8]`/`sums[9]` average away. The
                // aggregate is dominated by the short horizons - every horizon contributes the
                // same bar count, and the σ band a one-bar forecast has to cover is the
                // easiest one - so it cannot say whether the predictive scale is
                // over-dispersed at the long end specifically. Same masked tensors the
                // aggregate already reduced, two more column reductions per batch.
                within_1.sum_dim_intlist([0i64, 2].as_slice(), false, Kind::Double),
                within_2.sum_dim_intlist([0i64, 2].as_slice(), false, Kind::Double),
            ],
            0,
        );
        // The amplitude moments, in the two coordinates a decoded candle actually has: the
        // close ANCHOR every channel is built on, and each channel's OFFSET from it. Eight
        // column reductions over the origin axis of tensors already resident, in f64, keyed
        // `[horizon, channel]`. The forecast here is the mean as EMITTED by this pass - on a
        // calibrated model that is the gained mean, which is what makes an `evaluate` panel a
        // statement about what was applied rather than about what could have been.
        let anchor = forecast.scaled.select(-1, close).unsqueeze(-1);
        let offsets = &forecast.scaled - &anchor;
        let column = |values: Tensor| {
            values
                .expand_as(&forecast.scaled)
                .sum_dim_intlist([0i64].as_slice(), false, Kind::Double)
        };
        self.amplitude += Tensor::stack(
            &[
                column(mask.shallow_clone()),
                column(&forecast.targets * &mask),
                column(persistence.shallow_clone()),
                column(anchor.square() * &mask),
                column(&anchor * &offsets * &mask),
                column(offsets.square() * &mask),
                column(&anchor * &forecast.targets * &mask),
                column(&offsets * &forecast.targets * &mask),
            ],
            0,
        );
    }
    fn finish(self, timing: EvalTiming) -> Result<Evaluation> {
        let pred_len = self.pred_len;
        let bars = self.bars;
        ensure!(
            self.filled == self.bar_valid.size()[0],
            "evaluation accumulated {} of {} windows",
            self.filled,
            self.bar_valid.size()[0]
        );
        let (trading, portfolio) = self.trading()?;
        // Window ratios and the per-horizon trimmed sums finish on device; the top-1% |close|
        // thresholds use k_h = ⌈n_h / 100⌉ ≤ n_h and invalid bars sit at -1 below every valid
        // magnitude, so the k_h-th largest is always a valid bar.
        let window_squared = self
            .bar_squared
            .sum_dim_intlist([1i64].as_slice(), false, Kind::Double);
        let window_persistence = self
            .bar_persistence
            .sum_dim_intlist([1i64].as_slice(), false, Kind::Double);
        let scored = window_persistence.gt(0.);
        let median_window_ratio = if scored.sum(Kind::Int64).int64_value(&[]) > 0 {
            (window_squared.masked_select(&scored) / window_persistence.masked_select(&scored))
                .median()
                .double_value(&[])
        } else {
            f64::NAN
        };
        let counts = self.horizon_sums.get(2);
        let tail_counts = ((counts / CHANNELS as f64).round() + 99.)
            .floor_divide_scalar(100)
            .to_kind(Kind::Int64);
        let widest = tail_counts.max().int64_value(&[]);
        // |close target| with -1 on invalid bars, so an invalid bar never reaches a top-1%
        // threshold. Reconstructed rather than stored: the signed close target and its mask
        // are already resident for the trading diagnostics.
        let bar_close = self.bar_target.abs() * &self.bar_valid + &self.bar_valid - 1.;
        let (top, _) = bar_close.topk(widest, 0, true, true);
        let thresholds = top.gather(0, &(tail_counts - 1).reshape([1, pred_len]), false);
        let keep = bar_close.lt_tensor(&thresholds).to_kind(Kind::Float);
        let trimmed = Tensor::stack(
            &[
                (&self.bar_squared * &keep).sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
                (&self.bar_persistence * keep).sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
            ],
            0,
        );
        let horizon_sums = Tensor::cat(&[&self.horizon_sums, &trimmed], 0);
        let sums = Vec::<f64>::try_from(self.sums.to_device(Device::Cpu))?;
        let horizon_sums = Vec::<f64>::try_from(horizon_sums.to_device(Device::Cpu).flatten(0, -1))?;
        ensure!(
            sums.iter().chain(&horizon_sums).all(|x| x.is_finite()) && bars > 0,
            "nonfinite universe validation outputs"
        );
        let mut rows = horizon_sums.chunks_exact(pred_len as usize);
        let mut row = || rows.next().expect("fifteen per-horizon accumulator rows");
        let (model_sums, persistence_sums, counts) = (row(), row(), row());
        let (absolute_sums, target_sums) = (row(), row());
        let (wins, predicted, hits, ups) = (row(), row(), row(), row());
        let (absolute_model_sums, absolute_persistence_sums) = (row(), row());
        let (within_1_sums, within_2_sums) = (row(), row());
        let (trimmed_squared, trimmed_persistence) = (row(), row());
        ensure!(
            counts.iter().all(|n| *n > 0.),
            "every horizon needs a valid validation target bar"
        );
        let ratio = |numerator: &[f64], denominator: &[f64]| -> Vec<f64> {
            numerator
                .iter()
                .zip(denominator)
                .map(|(n, d)| n / d)
                .collect()
        };
        let per_bar = |values: &[f64]| -> Vec<f64> {
            values
                .iter()
                .zip(counts)
                .map(|(v, n)| v * CHANNELS as f64 / n)
                .collect()
        };
        // The `[AMPLITUDE_ROWS, pred_len, CHANNELS]` accumulator, unpacked row by row. Row 0
        // is the mask replicated over channels, so any channel's column is the horizon's valid
        // BAR count - the close channel's is taken, because that is the population the anchor
        // amplitude is measured on.
        let channels = CHANNELS as usize;
        let cells = pred_len as usize * channels;
        let amplitude = Vec::<f64>::try_from(
            self.amplitude
                .to_device(Device::Cpu)
                .to_kind(Kind::Double)
                .flatten(0, -1),
        )?;
        ensure!(
            amplitude.len() == AMPLITUDE_ROWS as usize * cells,
            "the amplitude accumulator holds {} cells for {} horizons and {channels} channels",
            amplitude.len(),
            pred_len
        );
        let mut amplitude_rows = amplitude.chunks_exact(cells);
        let mut amplitude_row = || {
            amplitude_rows
                .next()
                .expect("eight per-(horizon, channel) amplitude rows")
                .to_vec()
        };
        let counted = amplitude_row();
        let amplitude = Moments {
            pred_len: pred_len as usize,
            channels,
            bars: (0..pred_len as usize)
                .map(|horizon| counted[horizon * channels + CLOSE_CHANNEL])
                .collect(),
            target: amplitude_row(),
            target_square: amplitude_row(),
            anchor_square: amplitude_row(),
            anchor_offset: amplitude_row(),
            offset_square: amplitude_row(),
            anchor_target: amplitude_row(),
            offset_target: amplitude_row(),
        };
        amplitude.validate()?;
        // The two reductions must agree where they overlap. `close_mse_ratio` comes from the
        // per-(window, bar) close arrays and `channel_ratio` from the pooled channel moments,
        // by different summation orders over different tensors, and they are the SAME
        // quantity: `Σ(y-f)²/Σy²` on the close channel. A transposed or mis-keyed amplitude
        // accumulator would produce a plausible-looking gain curve and nothing else would
        // catch it, so the agreement is asserted rather than assumed. The tolerance is
        // reduction rounding: the channel form reaches the same number through
        // `Σf² - 2Σfy + Σy²`, which cancels.
        for horizon in 0..pred_len as usize {
            let (pooled, close) = (
                amplitude.channel_ratio(horizon, CLOSE_CHANNEL),
                trading.close_mse_ratio[horizon],
            );
            ensure!(
                (pooled - close).abs() <= 1e-6 * close.abs().max(1.),
                "at h={} the amplitude accumulator's close MSE ratio {pooled} disagrees with \
                 the trading reduction's {close}; the two are the same sum and one of them is \
                 keyed wrong",
                horizon + 1
            );
        }
        let elements = bars as f64 * 4.;
        Ok(Evaluation {
            mse: sums[0] / elements,
            persistence_mse: sums[1] / elements,
            absolute_mse: sums[10] / elements,
            absolute_persistence_mse: sums[11] / elements,
            rmse_price: (sums[2] / elements).sqrt(),
            mae_price: sums[3] / elements,
            invalid_fraction: sums[4] / bars as f64,
            tail_loss_share: if sums[0] > 0. { sums[5] / sums[0] } else { 0. },
            nll: sums[6] / elements,
            objective_nll: sums[12] / sums[13].max(f64::MIN_POSITIVE),
            persistence_nll: sums[7] / elements,
            within_1_sigma: sums[8] / elements,
            within_2_sigma: sums[9] / elements,
            median_window_ratio,
            horizon: HorizonCurve {
                mse: ratio(model_sums, counts),
                persistence_mse: ratio(persistence_sums, counts),
                absolute_mse: ratio(absolute_model_sums, counts),
                absolute_persistence_mse: ratio(absolute_persistence_sums, counts),
                mae_ratio: ratio(absolute_sums, target_sums),
                win_rate: per_bar(wins),
                hit_rate: ratio(hits, predicted),
                up_fraction: per_bar(ups),
                trimmed_mse_ratio: ratio(trimmed_squared, trimmed_persistence),
                within_1_sigma: ratio(within_1_sums, counts),
                within_2_sigma: ratio(within_2_sums, counts),
                valid_elements: counts.to_vec(),
            },
            trading,
            portfolio,
            amplitude,
            timing,
        })
    }
    /// The trading diagnostics, all of them from the resident per-(window, bar) close
    /// coordinate arrays and one host transfer. Every per-horizon reduction below is a
    /// column reduction over dimension 0; the per-timestamp cross-sections are `index_add`
    /// scatters keyed by `groups`; the decile thresholds and rank correlations are `topk`
    /// and `argsort` over dimension 0. The scalar algebra that turns the 35 accumulated
    /// sums into ratios, correlations and gain shares runs on the host over `pred_len`
    /// elements, which is where readable arithmetic belongs.
    fn trading(&self) -> Result<(TradingCurve, PortfolioCurve)> {
        let h = self.pred_len;
        let device = self.bar_valid.device();
        let m = &self.bar_valid;
        let f = &self.bar_forecast;
        let y = &self.bar_target;
        let col = |t: &Tensor| t.sum_dim_intlist([0i64].as_slice(), false, Kind::Double);
        let n = col(m);
        let mu = (&n).clamp_min(1.).reciprocal() * col(f);
        let mu_row = mu.to_kind(Kind::Float).reshape([1, h]);
        // A bet is a bar where forecast and target both carry a nonzero sign; a flat forecast
        // is not a directional claim and a flat target has no side to be right about.
        let rate = |predicted: &Tensor, realized: &Tensor, valid: &Tensor| {
            let (ps, rs) = (predicted.sign(), realized.sign());
            let bet = ps.abs() * rs.abs() * valid;
            let hit = (ps * rs).gt(0.).to_kind(Kind::Float) * valid;
            (col(&hit), col(&bet))
        };
        let (close_hit, close_bet) = rate(f, y, m);
        // Mid anchor: the model's error is unchanged, only persistence's denominator moves
        // from the origin bar's trade close to its log high/low midpoint. Bid-ask bounce
        // inflates the close-anchored denominator and nothing else.
        let mid = self.window_mid.reshape([-1, 1]);
        let mid_target = (y - &mid) * m;
        let mid_forecast = (f - &mid) * m;
        let (mid_hit, mid_bet) = rate(&mid_forecast, &mid_target, m);
        // Delayed-coordinate forecast quality: the t+1 close to t+h close move in neutral
        // sigma units, against zero persistence. This is not execution P&L.
        // Identically zero at h = 1, so that column is undefined.
        let delay_mask = m * m.narrow(1, 0, 1);
        let delayed_target = (y - y.narrow(1, 0, 1)) * &delay_mask;
        let delayed_forecast = (f - f.narrow(1, 0, 1)) * &delay_mask;
        let (delayed_hit, delayed_bet) = rate(&delayed_forecast, &delayed_target, &delay_mask);
        // Ordinal ranks for Spearman. Invalid bars are pushed past every valid one by the
        // sentinel, so a column's valid entries hold exactly the ranks 0..n_h-1 among
        // themselves and the masked Pearson over the ranks is the rank correlation.
        const SENTINEL: f64 = 1e30;
        let ranks = |t: &Tensor| {
            ((t + (1.0_f64 - m) * SENTINEL).argsort(0, false).argsort(0, false)).to_kind(Kind::Float) * m
        };
        let (rf, ry) = (ranks(f), ranks(y));
        // Conviction deciles on the demeaned signal |g|: invalid bars sit at -1 for the top
        // threshold and at the sentinel for the bottom one, so neither can be selected.
        let g = (f - &mu_row) * m;
        let magnitude = g.abs();
        let k = (&n / 10.).floor().to_kind(Kind::Int64).clamp_min(1);
        let widest = k.max().int64_value(&[]);
        let index = (&k - 1).reshape([1, h]);
        let high = &magnitude * m + (m - 1.);
        let (top, _) = high.topk(widest, 0, true, true);
        let top_mask = high.ge_tensor(&top.gather(0, &index, false)).to_kind(Kind::Float) * m;
        let low = &magnitude * m + (1.0_f64 - m) * SENTINEL;
        let (bottom, _) = low.topk(widest, 0, false, true);
        let bottom_mask = low.le_tensor(&bottom.gather(0, &index, false)).to_kind(Kind::Float) * m;
        let side = g.sign() * y;
        let decile = |selected: &Tensor| {
            let (hit, bet) = rate(&g, y, selected);
            [col(selected), col(&(&side * selected)), hit, bet]
        };
        // Per-timestamp cross-sections. One scatter per moment per horizon, no loop.
        let group = |source: &Tensor| {
            Tensor::zeros([self.group_count, h], (Kind::Float, device))
                .index_add(0, &self.groups, source)
        };
        let (gn, gf, gy) = (group(m), group(f), group(y));
        let (gff, gyy, gfy) = (group(&f.square()), group(&y.square()), group(&(f * y)));
        let inverse = gn.clamp_min(1.).reciprocal();
        let (mean_f, mean_y) = (&gf * &inverse, &gy * &inverse);
        let covariance = &gfy * &inverse - &mean_f * &mean_y;
        let spread = ((&gff * &inverse - mean_f.square()).clamp_min(0.)
            * (&gyy * &inverse - mean_y.square()).clamp_min(0.))
        .sqrt();
        let usable = gn
            .ge(CROSS_SECTION_MIN as f64)
            .logical_and(&spread.gt(1e-12))
            .to_kind(Kind::Double);
        let ic = (covariance / spread.clamp_min(1e-30)).to_kind(Kind::Double) * &usable;
        let stacked = Tensor::stack(
            &[
                n,
                col(f),
                col(y),
                col(&f.square()),
                col(&y.square()),
                col(&(f * y)),
                col(&((y - f) * m).square()),
                col(&self.bar_squared),
                col(&self.bar_persistence),
                col(&mid_target.square()),
                mid_hit,
                mid_bet,
                close_hit,
                close_bet,
                col(&(&delayed_target - &delayed_forecast).square()),
                col(&delayed_target.square()),
                delayed_hit,
                delayed_bet,
                col(&rf),
                col(&ry),
                col(&rf.square()),
                col(&ry.square()),
                col(&(&rf * &ry)),
                col(&usable),
                col(&ic),
                col(&ic.square()),
            ]
            .into_iter()
            .chain(decile(&top_mask))
            .chain(decile(&bottom_mask))
            .collect::<Vec<_>>(),
            0,
        );
        let flat = Vec::<f64>::try_from(stacked.to_device(Device::Cpu).flatten(0, -1))?;
        let width = h as usize;
        let at = |row: usize, j: usize| flat[row * width + j];
        let portfolio = self.portfolio()?;
        let mut curve = TradingCurve {
            total_gain: Vec::with_capacity(width),
            all_channel_gain: Vec::with_capacity(width),
            offset_gain: Vec::with_capacity(width),
            demeaned_gain: Vec::with_capacity(width),
            scaling_gain: Vec::with_capacity(width),
            mean_forecast: Vec::with_capacity(width),
            mean_target: Vec::with_capacity(width),
            pearson: Vec::with_capacity(width),
            spearman: Vec::with_capacity(width),
            cross_sectional_ic: Vec::with_capacity(width),
            cross_sectional_ic_se: Vec::with_capacity(width),
            close_mse_ratio: Vec::with_capacity(width),
            mid_anchor_mse_ratio: Vec::with_capacity(width),
            delayed_mse_ratio: Vec::with_capacity(width),
            close_hit_rate: Vec::with_capacity(width),
            mid_anchor_hit_rate: Vec::with_capacity(width),
            delayed_hit_rate: Vec::with_capacity(width),
            top_decile_hit_rate: Vec::with_capacity(width),
            bottom_decile_hit_rate: Vec::with_capacity(width),
            top_decile_return: Vec::with_capacity(width),
            bottom_decile_return: Vec::with_capacity(width),
            conviction_spread_return: Vec::with_capacity(width),
            optimal_gain: Vec::with_capacity(width),
            forecast_variance: Vec::with_capacity(width),
            covariance: Vec::with_capacity(width),
            persistence: Vec::with_capacity(width),
            best_scale_mse_ratio: Vec::with_capacity(width),
            cross_sectional_ic_moments: Vec::with_capacity(width),
            cross_sections: self.group_count as usize,
        };
        for j in 0..width {
            let count = at(0, j);
            ensure!(count > 0., "horizon {j} scored no valid bar");
            let (mean_forecast, mean_target) = (at(1, j) / count, at(2, j) / count);
            let persistence = at(4, j) / count;
            let error = at(6, j) / count;
            let signal_square = at(3, j) / count - mean_forecast * mean_forecast;
            let signal_target = at(5, j) / count - mean_forecast * mean_target;
            let target_variance = persistence - mean_target * mean_target;
            let offset = (2. * mean_forecast * mean_target - mean_forecast * mean_forecast)
                / persistence;
            let demeaned = if signal_square > 0. {
                signal_target * signal_target / signal_square / persistence
            } else {
                0.
            };
            let total = 1. - error / persistence;
            curve.total_gain.push(total);
            curve
                .all_channel_gain
                .push(1. - at(7, j) / at(8, j).max(f64::MIN_POSITIVE));
            curve.offset_gain.push(offset);
            curve.demeaned_gain.push(demeaned);
            curve.scaling_gain.push(total - offset - demeaned);
            // The MSE-optimal gain on the demeaned forecast, `β̂ = mean(y·g)/mean(g²)`, which
            // is identically `ρ·σ_y/σ_f` for the Pearson below. A constant forecast has no
            // amplitude to calibrate, so its gain is undefined rather than 0: 0 would read as
            // "infinitely over-amplified", the opposite of what a flat forecast is.
            curve.optimal_gain.push(if signal_square > 0. {
                signal_target / signal_square
            } else {
                f64::NAN
            });
            // The two halves of that ratio, unreduced. `β̂ > 1` is either a real
            // under-amplitude or a `Var(f)` collapsing toward zero, and no function of `β̂`
            // alone can tell those apart - only the numerator and the denominator beside
            // `mean(y²)`, which is the scale they are both small against.
            curve.forecast_variance.push(signal_square);
            curve.covariance.push(signal_target);
            curve.persistence.push(persistence);
            // What the SAME forecast would score against persistence with its amplitude
            // fixed and nothing else changed: `1 - offset - demeaned`, equivalently the
            // achieved ratio plus the (non-positive) cross term. Read beside
            // `close_mse_ratio` it separates "no signal" - both near 1 - from "signal, wrong
            // amplitude" - achieved above 1 while this one stays below it.
            curve.best_scale_mse_ratio.push(1. - offset - demeaned);
            curve.mean_forecast.push(mean_forecast);
            curve.mean_target.push(mean_target);
            curve
                .pearson
                .push(signal_target / (signal_square * target_variance).sqrt());
            let rank_correlation = |mean_a: f64, mean_b: f64, aa: f64, bb: f64, ab: f64| {
                (ab / count - mean_a * mean_b)
                    / ((aa / count - mean_a * mean_a) * (bb / count - mean_b * mean_b)).sqrt()
            };
            curve.spearman.push(rank_correlation(
                at(18, j) / count,
                at(19, j) / count,
                at(20, j),
                at(21, j),
                at(22, j),
            ));
            let moments = at(23, j);
            let ic_mean = at(24, j) / moments.max(1.);
            let ic_variance = (at(25, j) / moments.max(1.) - ic_mean * ic_mean).max(0.);
            curve.cross_sectional_ic_moments.push(moments);
            curve.cross_sectional_ic.push(if moments > 0. {
                ic_mean
            } else {
                f64::NAN
            });
            // A single contributing cross-section has no dispersion to estimate and none at
            // all has no mean: both must read NaN. A 0 here would render as a zero-width
            // band around an IC that was never measured, which is the strongest possible
            // claim about the weakest possible evidence.
            curve.cross_sectional_ic_se.push(if moments >= 2. {
                (ic_variance / moments).sqrt()
            } else {
                f64::NAN
            });
            curve.close_mse_ratio.push(error / persistence);
            curve.mid_anchor_mse_ratio.push(at(6, j) / at(9, j));
            curve.delayed_mse_ratio.push(at(14, j) / at(15, j));
            curve.close_hit_rate.push(at(12, j) / at(13, j));
            curve.mid_anchor_hit_rate.push(at(10, j) / at(11, j));
            curve.delayed_hit_rate.push(at(16, j) / at(17, j));
            let top_return = at(27, j) / at(26, j);
            let bottom_return = at(31, j) / at(30, j);
            curve.top_decile_return.push(top_return);
            curve.bottom_decile_return.push(bottom_return);
            curve
                .conviction_spread_return
                .push(top_return - bottom_return);
            curve.top_decile_hit_rate.push(at(28, j) / at(29, j));
            curve.bottom_decile_hit_rate.push(at(32, j) / at(33, j));
        }
        Ok((curve, portfolio))
    }
    /// Fixed-policy endpoint utility, separate from neutral-coordinate forecast quality.
    fn portfolio(&self) -> Result<PortfolioCurve> {
        utility::evaluate(
            &self.bar_policy_forecast,
            &self.bar_log_scale,
            &self.bar_raw_target,
            &self.bar_valid,
            &self.window_sigma,
            &self.window_entry_log,
            &self.groups,
            self.group_count,
        )
    }
}
fn synchronized(device: Device, started: Instant) -> f64 {
    if device.is_cuda() {
        tch::Cuda::synchronize(0);
    }
    started.elapsed().as_secs_f64() * 1000.
}
/// One evaluated split's retained per-horizon curves, so a report interval that only scored
/// the held-out sample still redraws the last held-out full curves beside it.
struct Curves {
    horizon: HorizonCurve,
    trading: TradingCurve,
    portfolio: PortfolioCurve,
}
/// Final-origin scores over `origins`, batched like training with the host loader one batch
/// ahead; the forward and metric phases are synchronized so the timing attributes wall clock.
fn score(
    corpus: &Arc<Corpus>,
    model: &CausalPatchModel,
    origins: &[WindowRef],
    batch_size: usize,
    device: Device,
) -> Result<Evaluation> {
    ensure!(
        batch_size > 0 && !origins.is_empty(),
        "evaluation batch/origins must be nonempty"
    );
    let _guard = tch::no_grad_guard();
    let started = Instant::now();
    synchronized(device, started);
    let mut timing = EvalTiming::default();
    // Cross-sectional statistics need the evaluation windows grouped by the moment they were
    // forecast from. The origin clock is already in the row: `origins` names the ticker and the
    // valid-bar ordinal, so the corpus resolves the origin bar's UTC timestamp directly. Dense
    // ranks over the distinct timestamps make a per-timestamp reduction one `index_add` on
    // device, and the host cost is one memory-mapped bar header per scored window - gathered
    // ticker-major by [`Corpus::origin_timestamps`], because a cross-section draw is ordered
    // by timestamp and the mapping is ticker-major, so the naive order takes a cold page per
    // element.
    let stamps = corpus.origin_timestamps(origins);
    let mut distinct = stamps.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let ranked: Vec<i64> = stamps
        .iter()
        .map(|stamp| {
            distinct
                .binary_search(stamp)
                .expect("every origin timestamp is one of the distinct timestamps") as i64
        })
        .collect();
    let mut scorer = Scorer::new(
        model.half_log_horizon(),
        &model.horizon_weights(),
        &Tensor::from_slice(&ranked),
        distinct.len(),
        corpus.contract.pred_len as i64,
        device,
    );
    let loader = Prefetcher::new(Arc::clone(corpus));
    loader.request(&origins[..origins.len().min(batch_size)])?;
    // Reallocated only when the batch shape changes, which is once for the whole run and
    // once more for a partial last batch.
    let mut resident: Option<Batch> = None;
    for index in 0..origins.len().div_ceil(batch_size) {
        // Only the FIRST batch is synchronized between phases. Two `Cuda::synchronize` per
        // batch drained the launch pipeline twice per batch, so no batch's forward could
        // overlap the previous batch's metric reduction and the instrumentation was a large
        // part of what it was measuring. One sampled batch costs one serialized batch, which
        // is the same convention the training loop's phase sample uses.
        let sampled = index == 0;
        let phase = |started: Instant| {
            if sampled {
                synchronized(device, started)
            } else {
                0.
            }
        };
        let waiting = Instant::now();
        let (host, _) = loader.receive()?;
        timing.loader_ms += waiting.elapsed().as_secs_f64() * 1000.;
        let next = (index + 1) * batch_size;
        if next < origins.len() {
            loader.request(&origins[next..(next + batch_size).min(origins.len())])?;
        }
        let valid_target_bars = host.valid_target_bars;
        // The same resident buffer the training step uses, for the same reason: `to_device`
        // allocates and frees the whole packed row block on every batch. Evaluation's last
        // batch is a partial one, so the buffer is reallocated when the shape changes -
        // twice per evaluation, not once per batch.
        if resident
            .as_ref()
            .is_none_or(|batch: &Batch| batch.rows() != host.rows())
        {
            resident = Some(host.resident(device));
        }
        let device_batch = resident.as_mut().expect("just allocated");
        host.upload(device_batch)?;
        let forwarding = Instant::now();
        let forecast = final_origin(model, device_batch);
        timing.forward_ms += phase(forwarding);
        let scoring = Instant::now();
        scorer.accumulate(&forecast, valid_target_bars);
        timing.metrics_ms += phase(scoring);
    }
    timing.total_ms = synchronized(device, started);
    scorer.finish(timing)
}
fn candle_windows(
    corpus: &Corpus,
    model: &CausalPatchModel,
    device: Device,
) -> Result<Vec<CandleWindow>> {
    let _guard = tch::no_grad_guard();
    let refs = fixed_origins(&corpus.validation_refs, 4)?;
    let batch = corpus.batch(&refs, device)?;
    let prices = final_origin(model, &batch).rebased_prices;
    let values = Vec::<f32>::try_from(
        prices
            .to_kind(Kind::Float)
            .to_device(Device::Cpu)
            .flatten(0, -1),
    )?;
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "nonfinite candle prediction"
    );
    let history = (corpus.contract.context - 1).min(32);
    refs.iter()
        .zip(values.chunks_exact(corpus.contract.pred_len * 4))
        .map(|(&reference, values)| {
            let ticker = corpus.ticker(reference);
            Ok(CandleWindow {
                ticker: ticker.contract.ticker.clone(),
                actual: ticker.candle_window(
                    reference.origin,
                    history,
                    corpus.contract.pred_len,
                )?,
                origin: history,
                predicted: values
                    .chunks_exact(4)
                    .map(|p| CandleBar {
                        open: p[0],
                        high: p[1],
                        low: p[2],
                        close: p[3],
                    })
                    .collect(),
                timestamp: ticker
                    .timestamp(reference.origin)
                    .checked_add(300_000)
                    .context("candle timestamp overflow")?,
            })
        })
        .collect()
}

pub fn train(args: TrainArgs) -> Result<()> {
    let base_learning_rate = args.validate()?;
    if let Some(name) = &args.run {
        RunDir::ensure_creatable(RUNS_PATH, name)?;
    }
    // Startup is measured from `execve`, not from here: the dynamic linker maps libtorch and
    // the embedded interpreter initialises before `main` runs, and an operator waiting for the
    // first step waits through all of it.
    let phase = Instant::now();
    let device = cuda_device()?;
    let _backward = crate::torch::cuda::cfg::disable_autograd_multithreading();
    let cuda_context_ms = phase.elapsed().as_secs_f64() * 1000.;
    let run = RunDir::create_fresh(RUNS_PATH, args.run.as_deref())?;
    println!("CausalPatch loading eligible ticker universe and authenticating shared chronological partitions");
    let mut corpus = Corpus::load(
        &args.data_dir,
        &args.ticker,
        args.model.seq_len as usize,
        args.model.pred_len as usize,
        args.common_context,
        &args.model.features,
        args.market_min_cross_section,
        args.in_period_sections,
    )?;
    corpus.prepare(device);
    corpus.timing.cuda_context_ms = cuda_context_ms;
    let phase = Instant::now();
    fs::write(
        run.root.join("timexer-segment-data-contract.json"),
        serde_json::to_vec_pretty(&corpus.contract)?,
    )?;
    reports::write_corpus(
        &run.gens.join("1"),
        &corpus.contract,
        corpus.calibration_refs.len(),
        &corpus.market,
    )?;
    corpus.timing.corpus_report_ms = phase.elapsed().as_secs_f64() * 1000.;
    let phase = Instant::now();
    tch::manual_seed(args.seed as i64);
    tch::Cuda::manual_seed_all(args.seed);
    let store = nn::VarStore::new(device);
    let model = CausalPatchModel::new(&store.root(), &args.model);
    corpus.timing.model_build_ms = phase.elapsed().as_secs_f64() * 1000.;
    let phase = Instant::now();
    let preview = fixed_origins(&corpus.validation_refs, args.eval_origins)?;
    // The trading family's own draw, built once: whole timestamp blocks out of the same
    // held-out origins, because the strided `preview` above puts one ticker on each of its
    // timestamps and no cross-sectional statistic survives that. Separate from `preview` so
    // the held-out sample's NLL - the selection scalar, and the one quantity that is
    // bit-identical across runs - is untouched.
    let cross_section = cross_section_origins(&corpus, &corpus.validation_refs)?;
    // The IN-PERIOD draw, built by the SAME `cross_section_origins` rule off the in-period
    // population instead of the validation population. Structural identity is the point: the
    // two draws are then whole timestamp blocks, strided over the corpus's own stable ticker
    // order, capped at the same width - so the ONLY thing that differs between the two ICs is
    // the market period the origins live in. Any difference in draw construction would be a
    // second explanation for a difference in IC and would destroy the experiment.
    //
    // Empty vector, not an error, when no in-period population was reserved: the whole family
    // then reads NaN and every existing draw is untouched.
    let in_period = if corpus.in_period_refs.is_empty() {
        Vec::new()
    } else {
        cross_section_origins(&corpus, &corpus.in_period_refs)?
    };
    // The amplitude calibration's two draws.
    //
    // The FIT block is a strided pick over the corpus's reserved `[70%, 80%)` partition - the
    // only population that took part in neither training nor checkpoint selection - drawn to
    // the same size as the held-out sample, so fitting at every report interval costs one
    // sample-sized pass rather than a full-split one.
    let calibration_draw = fixed_origins(&corpus.calibration_refs, args.eval_origins)?;
    // The in-sample comparand, and the only thing that answers WHY the amplitude is wrong: a
    // held-out gain that sweeps with the horizon is over-fitting if the in-sample gain is flat
    // and an objective defect if it sweeps too. Small on purpose - it is a diagnostic, not a
    // metric - and it never reaches the fit.
    let training_draw = fixed_origins(&corpus.train_refs, AMPLITUDE_TRAINING_ORIGINS)?;
    // One memory-mapped bar header per end of each window: the origin's own timestamp, and the
    // timestamp of the last bar its `pred_len` cumulative targets read. The second is what the
    // separation between fit and evaluation is measured against - MEASURED, on the realized
    // timestamps, rather than inferred from the boundary arithmetic that produced them.
    let dated = |refs: &[WindowRef]| -> Vec<(i64, i64)> {
        let pred_len = corpus.contract.pred_len;
        refs.iter()
            .map(|reference| {
                let ticker = corpus.ticker(*reference);
                (
                    ticker.timestamp(reference.origin),
                    ticker.timestamp(reference.origin + pred_len),
                )
            })
            .collect()
    };
    let fit_dated = dated(&calibration_draw);
    let amplitude_blocks = Blocks::spanning(&fit_dated, &dated(&corpus.validation_refs))?;
    // A gap in milliseconds is not legible on its own - the number that says whether the purge
    // is comfortable or marginal is how it compares to one origin's own `pred_len`-bar reach,
    // measured here from the same timestamps rather than assumed from a bar duration.
    let reach = fit_dated
        .iter()
        .map(|(origin, target)| target - origin)
        .max()
        .unwrap_or(1)
        .max(1);
    println!(
        "CausalPatch amplitude calibration partition: {} fit origins over [{}, {}] whose targets end at {}, then {} held-out-full origins over [{}, {}]; the two are separated by {} ms and share no bar, which is {:.2}x the {} ms a single origin's {}-bar target reach spans; {} in-sample training origins are scored beside them as the mechanism comparand",
        amplitude_blocks.calibration_origins,
        amplitude_blocks.calibration_first_origin_ms,
        amplitude_blocks.calibration_last_origin_ms,
        amplitude_blocks.calibration_last_target_ms,
        amplitude_blocks.evaluation_origins,
        amplitude_blocks.evaluation_first_origin_ms,
        amplitude_blocks.evaluation_last_origin_ms,
        amplitude_blocks.purge_gap_ms,
        amplitude_blocks.purge_gap_ms as f64 / reach as f64,
        reach,
        corpus.contract.pred_len,
        training_draw.len()
    );
    corpus.timing.held_out_draw_ms = phase.elapsed().as_secs_f64() * 1000.;
    // The row-pool diversity knobs, applied AFTER every held-out draw is built and BEFORE the
    // pool is cloned into the step loop, which is the only correct place for them: the three
    // `held-out *` draws and the calibration partition are functions of `validation_refs`,
    // `in_period_refs` and `calibration_refs` alone and are provably untouched, while
    // `steps_per_epoch`, the schedule budget and the trajectory buffer below all read the
    // RETAINED count and would otherwise describe a pool the run does not train on.
    let phase = Instant::now();
    let selection = args.row_selection();
    let geometry = SupervisionGeometry::new(
        args.model.seq_len as usize,
        args.model.patch_len as usize,
        args.model.min_history as usize,
        args.model.pred_len as usize,
    )?;
    let census = supervision::select(&mut corpus, selection, geometry)?;
    corpus.timing.row_selection_ms = phase.elapsed().as_secs_f64() * 1000.;
    println!(
        "CausalPatch supervision census: {}; {} rows retained of {} enumerated ({} dropped by the patch phase), {} distinct supervised origins at {:.5} mean exposures per sweep, {:.5} of them interior at {} exposures, {} rows with a short horizon; 0.999 of outcomes covered by step {}, 0.99 by step {}, and the retained-versus-full timestamp decile drift is {:.5} against deciles {:?}",
        selection.summary(),
        census.rows,
        census.rows + census.phase_dropped_rows,
        census.phase_dropped_rows,
        census.support(),
        census.mean_multiplicity(),
        census.interior_share(),
        geometry.interior_multiplicity(),
        census.partial_target_rows,
        census.saturation_step(0.999, args.batch_size),
        census.saturation_step(0.99, args.batch_size),
        census.decile_drift(),
        census.retained_deciles.map(|share| (share * 1e4).round() / 1e4),
    );
    // Share of the unholed row population this arm purged, carried into every chart title.
    // The denominator is reconstructed rather than stored: `train_refs` plus the purged rows IS
    // the unholed population, proved by the census assertion in
    // `in_period_origins_share_no_origin_and_no_target_bar_with_any_training_row`.
    let purged_row_share = {
        let purged = corpus.contract.in_period_purged_rows;
        let total = corpus.train_refs.len() + purged;
        if total == 0 {
            0.
        } else {
            purged as f64 / total as f64
        }
    };
    let mut origins = corpus.train_refs.clone();
    // Whole batches only: the forward and backward are captured as one CUDA graph, and a
    // capture records SHAPES, so every step of the run has to present the same row count.
    let steps_per_epoch = origins.len() / args.batch_size;
    ensure!(
        steps_per_epoch > 0,
        "batch {} exceeds the {} training rows",
        args.batch_size,
        origins.len()
    );
    // The schedule's endpoint is stated, not inherited from how many steps an epoch happens
    // to contain, and NOT from `--max-steps`: see [`schedule_budget`].
    let schedule = LrSchedule::new(
        schedule_budget(&args, steps_per_epoch),
        NANOGPT_COOLDOWN_FRAC,
        NANOGPT_COOLDOWN_FLOOR,
    )?;
    let knobs = RecipeKnobs {
        scalar_lr_mult: args.scalar_lr_mult,
        x0_lambdas: args.model.x0_lambdas,
        schedule,
        mlp_down_lr: args.mlp_down_lr,
    };
    let phase = Instant::now();
    let mut engine = Engine::new(
        &store,
        base_learning_rate,
        knobs,
        args.fused,
        args.optimizer,
    )?;
    corpus.timing.model_build_ms += phase.elapsed().as_secs_f64() * 1000.;
    corpus.timing.startup_total_ms =
        super::cache::process_elapsed_ms().unwrap_or(corpus.timing.total_ms);
    println!(
        "CausalPatch startup: {:.1} s from process start to the first optimizer step",
        corpus.timing.startup_total_ms / 1000.
    );
    let corpus = Arc::new(corpus);
    let rules = StopRules::new(&args);
    // Host capacity only - one `f64` per family per step - so the cap keeps a short arm's
    // buffer short. It is not a limit on anything the run does.
    let planned_steps = steps_per_epoch * args.epochs;
    let mut lr_trajectory = LrTrajectory::new(
        &engine,
        rules.max_steps.unwrap_or(planned_steps).min(planned_steps),
    );
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let mut step = 0;
    // Both stopping criteria minimize the OBJECTIVE-weighted held-out NLL, never the
    // equal-weighted aggregate: jobs 5190-5193 showed the aggregate is dominated by horizons
    // with no out-of-period predictability, so it selects on the wrong quantity. Under
    // `--horizon-loss uniform` the two are the same number and nothing about selection moves.
    let mut best_full = f64::INFINITY;
    let mut stale_epochs = 0;
    let mut best_objective = f64::INFINITY;
    let mut best_step: Option<usize> = None;
    let mut previews = 0usize;
    let mut stale_previews = 0usize;
    // The exit the run took, `None` while it is still running. Set by the evaluation block,
    // which is the only place that knows every criterion's state.
    let mut ended: Option<Termination> = None;
    let mut preview_curve: Option<Curves> = None;
    let mut full_curve: Option<Curves> = None;
    let loader = Prefetcher::new(Arc::clone(&corpus));
    println!("CausalPatch: {} tickers, {} training target bars, {} rows, {steps_per_epoch} optimizer steps per full epoch, {} rows left to the next epoch's reshuffle; context {}, horizon {}, {} dense origins per row, batch {}",corpus.contract.tickers.len(),corpus.contract.train_target_bars,origins.len(),origins.len() - steps_per_epoch * args.batch_size,args.model.seq_len,args.model.pred_len,args.model.origins(),args.batch_size);
    if let Some(cap) = rules.max_steps {
        println!("CausalPatch step cap: stopping at exactly {cap} optimizer steps of the {planned_steps} planned, after a final held-out full validation pass; the learning-rate schedule is untouched and still shaped against {} steps, so these are the first {cap} steps of that trajectory", schedule.budget_steps());
    }
    // The interval's per-phase breakdown, refreshed by the sampled step of each interval.
    let mut step_phases = None;
    for epoch in 1..=args.epochs {
        origins.shuffle(&mut rng);
        let mut completed = 0;
        let mut target_bars = 0;
        // `[nll, mse, nonfinite]`, all on device. The third slot replaces a per-step
        // `torch._assert_async` (a GIL acquisition and a tensor bridge on every step) with one
        // indicator kernel per step and one host read per report interval.
        let mut total_loss = Tensor::zeros([3], (Kind::Double, device));
        let mut epoch_steps = 0usize;
        let mut points = Vec::new();
        let mut hardware_history = Vec::new();
        // One host transfer per report interval, never per step: `recipe_scalars` concatenates
        // every learned mixing coefficient on device and copies the whole set once.
        let mut recipe_history: Vec<(usize, Vec<(String, f64)>)> = Vec::new();
        let mut hardware_offset = 0.;
        let mut hardware = Some(HardwareSampler::start()?);
        benchmark::cuda_memory(true)?;
        tch::Cuda::synchronize(0);
        let mut interval_started = Instant::now();
        let mut interval_steps = 0;
        let mut loader_wait_ms = 0.;
        let mut loader_build_ms = 0.;
        // The epoch runs a whole number of batches. The ragged tail of at most
        // `batch_size - 1` rows is left out of THIS epoch; the order is reshuffled every
        // epoch, so it is a different tail each time rather than a systematically excluded
        // set, and no step ever presents a shape the capture did not record.
        let epoch_origins = &origins[..steps_per_epoch * args.batch_size];
        loader.request(&epoch_origins[..args.batch_size])?;
        for (index, refs) in epoch_origins.chunks(args.batch_size).enumerate() {
            // The interval's last step is the one sampled for the per-phase breakdown: it pays
            // four synchronizations, every other step keeps its asynchronous launch pipeline.
            let sampled = (step + 1) % args.eval_every == 0;
            let waiting = Instant::now();
            let (host, build_ms) = loader.receive()?;
            loader_wait_ms += waiting.elapsed().as_secs_f64() * 1000.;
            loader_build_ms += build_ms;
            let valid_bars = host.valid_target_bars;
            if sampled && device.is_cuda() {
                tch::Cuda::synchronize(0);
            }
            // The forward and backward are captured exactly once, on the step after the
            // optimizer's own two bodies have finished capturing into the shared mempool.
            // The host loader is deliberately left quiescent across it - the next request is
            // issued below, after the step - because `pin_memory` can call `cudaHostAlloc`,
            // which synchronizes the device and would invalidate an in-flight capture.
            let losses = if step == CAPTURE_AFTER_STEPS && engine.capture_ready() {
                engine.arm_step_graph(&model, &host)?
            } else if sampled {
                let (losses, phases) = engine.timed_step(&model, &host)?;
                step_phases = Some(StepPhases {
                    h2d_ms: phases[0],
                    forward_backbone_ms: phases[1],
                    forward_head_ms: phases[2],
                    backward_ms: phases[3],
                    captured_replay_ms: phases[4],
                    optimizer_ms: phases[5],
                });
                losses
            } else {
                engine.step(&model, &host)?
            };
            // Requested only now, so the capture step above runs with the loader idle. The
            // loader still has the whole step's device execution to build the next batch,
            // and it needs a fraction of it.
            let next = (index + 1) * args.batch_size;
            if next < epoch_origins.len() {
                loader.request(&epoch_origins[next..next + args.batch_size])?;
            }
            total_loss += Tensor::stack(
                &[
                    losses.nll.shallow_clone(),
                    losses.mse,
                    losses.nll.isfinite().logical_not().to_kind(Kind::Float),
                ],
                0,
            )
            .to_kind(Kind::Double);
            // Recorded here, with `step` still holding the 0-based index of the step that just
            // ran, because that is the index the schedule's own cooldown start names. Host
            // reads of the optimizer's rate state: no device traffic, no synchronization.
            lr_trajectory.record(step, &engine)?;
            epoch_steps += 1;
            step += 1;
            interval_steps += 1;
            completed += refs.len();
            target_bars += valid_bars;
            let epoch_complete = completed == epoch_origins.len();
            if !rules.evaluates(step, epoch_complete) {
                continue;
            }
            tch::Cuda::synchronize(0);
            let step_ms = interval_started.elapsed().as_secs_f64() * 1000. / interval_steps as f64;
            let mut samples = hardware.take().unwrap().finish()?;
            for value in &mut samples {
                value[0] += hardware_offset;
            }
            hardware_offset = samples.last().map_or(hardware_offset, |v| v[0] + 100.);
            hardware_history.extend(samples);
            let (peak_bytes, _) = benchmark::cuda_memory(false)?;
            let selected = if epoch_complete {
                &corpus.validation_refs
            } else {
                &preview
            };
            // Reset here and read after: the evaluation's own peak is what has to fit
            // alongside the captured step's private mempool, which the training peak above
            // does not separate out.
            //
            // `empty_cache` first, because what has to fit is the evaluation's REQUIREMENT and
            // the allocator is otherwise holding the training interval's history on top of the
            // graph pool. The graph's private mempool is owned by the capture and is not
            // released here, so the captured step is untouched.
            crate::torch::cuda::empty_cache();
            benchmark::cuda_memory(true)?;
            let evaluation = score(&corpus, &model, selected, args.eval_batch_size, device)?;
            let (evaluation_peak_bytes, _) = benchmark::cuda_memory(false)?;
            // Leaves the next interval's training peak measuring the next interval.
            benchmark::cuda_memory(true)?;
            // The trading family's pass. Its own `Evaluation` and its own local: nothing here
            // reaches `best_*`, `preview_curve`, `full_curve` or `points`, so selection and
            // every step-matched curve are exactly what they were before this pass existed.
            let cross_started = Instant::now();
            let cross_evaluation =
                score(&corpus, &model, &cross_section, args.eval_batch_size, device)?;
            let cross_section_ms = cross_started.elapsed().as_secs_f64() * 1000.;
            benchmark::cuda_memory(true)?;
            // The in-period pass, scored at every interval beside the out-of-period one so the
            // two are read at the SAME optimizer step. Reading them at different steps would
            // confound the very trajectory the comparison is about.
            let in_period_started = Instant::now();
            let in_period_evaluation = if in_period.is_empty() {
                None
            } else {
                Some(score(&corpus, &model, &in_period, args.eval_batch_size, device)?)
            };
            let in_period_ms = in_period_started.elapsed().as_secs_f64() * 1000.;
            benchmark::cuda_memory(true)?;
            // The amplitude passes, and the FIT. Both are scored with the model exactly as it
            // is - un-gained - because the design matrix the gain is fitted to is the un-gained
            // emission, and because the selection scalar above is the un-gained model's NLL.
            // Nothing here touches `best_*`, `points`, or either curve slot.
            let amplitude_started = Instant::now();
            let calibration_pass =
                score(&corpus, &model, &calibration_draw, args.eval_batch_size, device)?;
            let training_pass =
                score(&corpus, &model, &training_draw, args.eval_batch_size, device)?;
            let amplitude_ms = amplitude_started.elapsed().as_secs_f64() * 1000.;
            benchmark::cuda_memory(true)?;
            let calibration =
                MeanCalibration::fit(amplitude_blocks.clone(), &calibration_pass.amplitude)?;
            let frozen = calibration.frozen();
            frozen.validate(corpus.contract.pred_len)?;
            log_calibration(step, &calibration, &training_pass.amplitude, amplitude_ms);
            if !epoch_complete {
                previews += 1;
                if evaluation.objective_nll < best_objective {
                    best_objective = evaluation.objective_nll;
                    best_step = Some(step);
                    stale_previews = 0;
                } else if previews > 2 {
                    stale_previews += 1;
                }
            }
            if epoch_complete {
                ensure!(
                    target_bars == corpus.contract.train_target_bars,
                    "epoch target coverage mismatch"
                );
                if evaluation.objective_nll < best_full {
                    best_full = evaluation.objective_nll;
                    stale_epochs = 0;
                } else {
                    stale_epochs += 1;
                }
            }
            // Decided here, before anything is written, because the manifest has to carry it:
            // a checkpoint that does not say why its run stopped cannot be told apart from
            // one written mid-run, and a capped arm read as an early-stopped one would be
            // quoted as evidence about convergence it carries none of.
            ended = rules.termination(Progress {
                step,
                epoch,
                epoch_complete,
                stale_previews,
                stale_epochs,
            });
            // A run that ends MID-EPOCH - a step cap, or preview patience - has never scored
            // the full split: only an epoch-complete interval does, so `held-out full` would
            // be absent from every base and the arm would carry nothing but 2,048 sample
            // windows. One extra pass over the 433,303 held-out origins buys the split back:
            // 103 s measured, 15% of a 4,000-step arm's stepping time, paid once, and only by
            // a run that would otherwise have no full-split measurement at all. Both
            // mid-epoch exits pay it, so every arm in a batch carries the same two splits
            // whichever way it ended.
            //
            // Deliberately NOT the interval's own evaluation: selection, `best_step` and
            // every step-matched sample curve stay exactly what an uncapped run wrote at this
            // step, which is the whole point of a capped arm.
            let final_full = if ended.is_some() && !epoch_complete {
                crate::torch::cuda::empty_cache();
                benchmark::cuda_memory(true)?;
                let started = Instant::now();
                let full = score(
                    &corpus,
                    &model,
                    &corpus.validation_refs,
                    args.eval_batch_size,
                    device,
                )?;
                let (full_peak_bytes, _) = benchmark::cuda_memory(false)?;
                benchmark::cuda_memory(true)?;
                println!("CausalPatch final held-out full validation at step {step}: objective-weighted NLL {:.4}, aggregate NLL {:.4} (persistence {:.4}), market-neutral MSE ratio {:.4} over {} origins in {:.1} s",full.objective_nll,full.nll,full.persistence_nll,full.mse/full.persistence_mse,corpus.validation_refs.len(),started.elapsed().as_secs_f64());
                Some((full, full_peak_bytes))
            } else {
                None
            };
            let finishing = Instant::now();
            let windows = candle_windows(&corpus, &model, device)?;
            let output = run.gens.join(epoch.to_string());
            // Sliced before the curves move into the split-keyed slots, because the step
            // axis needs this evaluation's decision horizons attached to THIS step's report
            // point; the curves themselves are overwritten at the next evaluation.
            let horizons = reports::horizon_track(&evaluation.horizon, &evaluation.trading);
            // The same slice through the cross-section pass. This is the one that carries the
            // decision metric at a report interval: the interval's own draw strides one origin
            // per timestamp, so its within-timestamp IC is undefined, and the full draw is
            // scored once per epoch. Its curves are also overwritten at the next evaluation.
            let cross_horizons =
                reports::horizon_track(&cross_evaluation.horizon, &cross_evaluation.trading);
            let in_period_horizons = in_period_evaluation
                .as_ref()
                .map(|e| reports::horizon_track(&e.horizon, &e.trading))
                .unwrap_or_default();
            let curves = Curves {
                horizon: evaluation.horizon,
                trading: evaluation.trading,
                portfolio: evaluation.portfolio,
            };
            if epoch_complete {
                full_curve = Some(curves);
            } else {
                preview_curve = Some(curves);
            }
            let manifest = Manifest {
                format: format_stamp(args.model.x0_lambdas, args.model.horizon_mean),
                model: args.model.clone(),
                data: corpus.contract.clone(),
                objective: OBJECTIVE.into(),
                numerics: NUMERICS.into(),
                requested_tickers: args.ticker.clone(),
                seed: args.seed,
                epoch,
                step,
                completed_origins: completed,
                completed_target_bars: target_bars,
                epoch_complete,
                planned_epochs: args.epochs,
                max_steps: rules.max_steps,
                row_selection: (!selection.is_identity()).then_some(selection),
                termination: ended,
                batch_size: args.batch_size,
                learning_rate: engine.learning_rate(),
                base_learning_rate,
                scalar_lr_mult: args.scalar_lr_mult,
                fused: args.fused,
                optimizer: args.optimizer,
                optimizer_recipe: args.optimizer.recipe(knobs),
                eval_every: args.eval_every,
                eval_origins: selected.len(),
                validation_nll: evaluation.nll,
                validation_mse: evaluation.mse,
                validation_is_full: epoch_complete,
                best_step,
                best_objective_nll: best_step.map(|_| best_objective),
                selection: selection_criterion(&args.model),
                // Fitted at THIS evaluation on the reserved partition, so a checkpoint carries
                // the calibration of the weights inside it and cannot be paired with another's.
                mean_gain: frozen.clone(),
                weights_sha256: String::new(),
                manifest_sha256: String::new(),
            };
            let checkpoint = if epoch_complete {
                format!("epoch-{epoch:04}")
            } else {
                "preview-latest".into()
            };
            save(&run.weights.join(&checkpoint), &store, manifest.clone())?;
            if best_step == Some(step) {
                save(&run.weights.join("preview-best"), &store, manifest)?;
                crate::torch::single_ticker_timexer::checkpoint::update_best(
                    &run.weights,
                    "preview-best",
                )?;
            }
            if epoch_complete && best_step.is_none() {
                crate::torch::single_ticker_timexer::checkpoint::update_best(
                    &run.weights,
                    &checkpoint,
                )?;
            }
            let reports_ms = finishing.elapsed().as_secs_f64() * 1000.;
            let train_loss = Vec::<f64>::try_from(&total_loss / epoch_steps as f64)?;
            ensure!(
                train_loss[2] == 0. && train_loss[0].is_finite(),
                "nonfinite CausalPatch training objective in the interval ending at step {step}"
            );
            points.push(Metrics {
                step,
                epoch,
                completed_origins: completed,
                total_origins: origins.len(),
                completed_target_bars: target_bars,
                total_target_bars: corpus.contract.train_target_bars,
                train_nll: Some(train_loss[0]),
                train_mse: Some(train_loss[1]),
                validation_nll: evaluation.nll,
                validation_objective_nll: evaluation.objective_nll,
                persistence_nll: evaluation.persistence_nll,
                validation_mse: evaluation.mse,
                persistence_mse: evaluation.persistence_mse,
                absolute_mse: evaluation.absolute_mse,
                absolute_persistence_mse: evaluation.absolute_persistence_mse,
                within_1_sigma: evaluation.within_1_sigma,
                within_2_sigma: evaluation.within_2_sigma,
                rmse_price: evaluation.rmse_price,
                mae_price: evaluation.mae_price,
                invalid_ohlc_fraction: evaluation.invalid_fraction,
                tail_loss_share: evaluation.tail_loss_share,
                median_window_ratio: evaluation.median_window_ratio,
                eval: EvalTiming {
                    reports_ms: Some(reports_ms),
                    cross_section_ms: Some(cross_section_ms),
                    in_period_ms: (!in_period.is_empty()).then_some(in_period_ms),
                    ..evaluation.timing
                },
                step_ms: Some(step_ms),
                validation_is_full: epoch_complete,
                validation_origins: selected.len(),
                tickers: corpus.contract.tickers.len(),
                loader_wait_ms: Some(loader_wait_ms / interval_steps as f64),
                loader_build_ms: Some(loader_build_ms / interval_steps as f64),
                peak_allocator_mib: Some(peak_bytes as f64 / 1048576.),
                evaluation_peak_mib: Some(evaluation_peak_bytes as f64 / 1048576.),
                capture_budget: engine.capture_budget(),
                step_phases,
                horizons,
                cross_horizons,
                cross_origins: cross_section.len(),
                in_period_horizons,
                in_period_origins: in_period.len(),
                in_period_purged_row_share: purged_row_share,
                startup: corpus.timing,
            });
            // The capped run's final pass, at the SAME step as the sample point above and in
            // the same `held-out full` slot an epoch-end evaluation fills. Every quantity the
            // step produced rather than the evaluation - the training losses, the step and
            // loader timings, the capture budget - is the sample point's by construction:
            // both points describe one optimizer step.
            let mut full_amplitude = None;
            if let Some((full, full_peak_bytes)) = final_full {
                let point = Metrics {
                    validation_nll: full.nll,
                    validation_objective_nll: full.objective_nll,
                    persistence_nll: full.persistence_nll,
                    validation_mse: full.mse,
                    persistence_mse: full.persistence_mse,
                    absolute_mse: full.absolute_mse,
                    absolute_persistence_mse: full.absolute_persistence_mse,
                    within_1_sigma: full.within_1_sigma,
                    within_2_sigma: full.within_2_sigma,
                    rmse_price: full.rmse_price,
                    mae_price: full.mae_price,
                    invalid_ohlc_fraction: full.invalid_fraction,
                    tail_loss_share: full.tail_loss_share,
                    median_window_ratio: full.median_window_ratio,
                    eval: full.timing,
                    validation_is_full: true,
                    validation_origins: corpus.validation_refs.len(),
                    evaluation_peak_mib: Some(full_peak_bytes as f64 / 1048576.),
                    horizons: reports::horizon_track(&full.horizon, &full.trading),
                    ..points
                        .last()
                        .expect("the interval pushed its own held-out sample point")
                        .clone()
                };
                points.push(point);
                full_amplitude = Some(full.amplitude);
                full_curve = Some(Curves {
                    horizon: full.horizon,
                    trading: full.trading,
                    portfolio: full.portfolio,
                });
            }
            reports::write_metrics(&output, &points)?;
            recipe_history.push((step, model.recipe_scalars()));
            reports::write_recipe_scalars(&output, epoch, step, &recipe_history)?;
            reports::write_horizon_loss_weight(
                &output,
                epoch,
                step,
                &args.model.horizon_loss.to_string(),
                &model.horizon_weights(),
            )?;
            // Written on EVERY evaluation, the identity included: the calibration state has to
            // sit on the same axis as the curves it moves, or a calibrated arm gets
            // step-matched against an uncalibrated one months later with nothing on either
            // chart to say so. The scored splits are reported with the gain and without it from
            // ONE pass, in closed form, so this costs no extra scoring.
            let mut scored = vec![reports::AmplitudeSplit {
                split: if epoch_complete {
                    reports::FULL
                } else {
                    reports::SAMPLE
                },
                origins: selected.len(),
                emission: reports::Emission::Uncalibrated,
                moments: &evaluation.amplitude,
            }];
            if let Some(amplitude) = &full_amplitude {
                scored.push(reports::AmplitudeSplit {
                    split: reports::FULL,
                    origins: corpus.validation_refs.len(),
                    emission: reports::Emission::Uncalibrated,
                    moments: amplitude,
                });
            }
            reports::write_amplitude(
                &output,
                &reports::AmplitudePanels {
                    epoch,
                    step,
                    tickers: corpus.contract.tickers.len(),
                    applied: &frozen,
                    fit: Some(reports::AmplitudeFit {
                        calibration: &calibration,
                        moments: &calibration_pass.amplitude,
                        training: Some(&training_pass.amplitude),
                        training_origins: training_draw.len(),
                    }),
                    scored,
                },
            )?;
            reports::write_horizon(
                &output,
                epoch,
                step,
                preview_curve.as_ref().map(|curves| HorizonSplit {
                    curve: &curves.horizon,
                    origins: preview.len(),
                }),
                full_curve.as_ref().map(|curves| HorizonSplit {
                    curve: &curves.horizon,
                    origins: corpus.validation_refs.len(),
                }),
                corpus.contract.tickers.len(),
            )?;
            reports::write_trading(
                &output,
                epoch,
                step,
                preview_curve.as_ref().map(|curves| TradingSplit {
                    curve: &curves.trading,
                    portfolio: &curves.portfolio,
                    origins: preview.len(),
                }),
                full_curve.as_ref().map(|curves| TradingSplit {
                    curve: &curves.trading,
                    portfolio: &curves.portfolio,
                    origins: corpus.validation_refs.len(),
                }),
                Some(TradingSplit {
                    curve: &cross_evaluation.trading,
                    portfolio: &cross_evaluation.portfolio,
                    origins: cross_section.len(),
                }),
                corpus.contract.tickers.len(),
            )?;
            reports::write_candles(&output, epoch, step, &windows)?;
            benchmark::write_hardware(
                &output,
                "CausalPatch production training (evaluation excluded)",
                &hardware_history,
            )?;
            reports::write_lr_trajectory(
                &output,
                epoch,
                step,
                engine.schedule(),
                &lr_trajectory,
            )?;
            reports::write_supervision_occupancy(&output, epoch, step, args.batch_size, &census)?;
            reports::write_corpus(
                &output,
                &corpus.contract,
                corpus.calibration_refs.len(),
                &corpus.market,
            )?;
            println!("CausalPatch epoch {epoch}/{} step {step}: {target_bars}/{} unique target bars; {} objective-weighted NLL {:.4}, aggregate NLL {:.4} (persistence {:.4}), market-neutral MSE ratio {:.4}, raw MSE ratio {:.4}; reports {}",args.epochs,corpus.contract.train_target_bars,if epoch_complete {"held-out full"} else {"held-out sample"},evaluation.objective_nll,evaluation.nll,evaluation.persistence_nll,evaluation.mse/evaluation.persistence_mse,evaluation.absolute_mse/evaluation.absolute_persistence_mse,output.display());
            // One exit, named. Everything above has already been flushed at this step - the
            // evaluation, the checkpoints, every report base - so a run that stops here is as
            // readable as one that runs out of epochs.
            if let Some(reason) = ended {
                match reason {
                    Termination::StepCap => println!("CausalPatch stopped at step {step}: the --max-steps cap, with the learning-rate schedule still shaped against {} steps; nothing about the fit is implied and the held-out curve may well still have been descending; weights/best retained", engine.schedule().budget_steps()),
                    Termination::PreviewPatience => println!("CausalPatch stopped at step {step}: held-out sample objective-weighted NLL {:.4} has not improved for {stale_previews} consecutive evaluations (best {best_objective:.4} at step {}, {}); weights/best retained", evaluation.objective_nll, best_step.unwrap_or(0), selection_criterion(&args.model)),
                    Termination::EpochPatience => println!("CausalPatch stopped at step {step} after {stale_epochs} complete epochs without improved held-out full NLL; weights/best retained"),
                    Termination::EpochLimit => println!("CausalPatch finished at step {step}: all {} planned epochs complete; weights/best retained", args.epochs),
                }
                break;
            }
            if !epoch_complete {
                hardware = Some(HardwareSampler::start()?);
            }
            tch::Cuda::synchronize(0);
            interval_started = Instant::now();
            interval_steps = 0;
            loader_wait_ms = 0.;
            loader_build_ms = 0.;
        }
        if ended.is_some() {
            break;
        }
    }
    Ok(())
}

/// An authenticated, CALIBRATED checkpoint resident on the device beside the corpus its own
/// manifest names. Every consumer - `evaluate`, the latent probe, the ceiling sweep, the
/// portfolio tape - comes through here, so none of them can end up authenticating less than
/// another or scoring a different amplitude than another.
///
/// The amplitude calibration is applied HERE and nowhere else. It is the checkpoint's own,
/// fitted out of sample by the run that wrote it, so a loaded model emits the calibrated mean
/// and the reports, the trading diagnostics and the portfolio sizing cannot disagree about
/// what was applied. `mean_gain().is_identity()` is what a caller reads to say "this
/// checkpoint's block identified no amplitude", and it is a measurement, not an absence.
///
/// The [`nn::VarStore`] comes back with the model because it is what `store.load` wrote the
/// checkpoint's bytes into; the caller has to hold it for as long as it scores.
pub(super) fn load_checkpoint(
    checkpoint: &Path,
    data_dir: &Path,
) -> Result<(Manifest, Arc<Corpus>, nn::VarStore, CausalPatchModel, Device)> {
    let manifest = Manifest::read(checkpoint)?;
    let mut corpus = Corpus::load(
        data_dir,
        &manifest.requested_tickers,
        manifest.data.context,
        manifest.data.pred_len,
        manifest.data.common_context,
        &manifest.data.features,
        manifest.data.market_min_cross_section,
        manifest.data.in_period_sections,
    )?;
    ensure!(
        corpus.contract == manifest.data,
        "dataset differs from authenticated universe/splits"
    );
    let device = cuda_device()?;
    corpus.prepare(device);
    let mut store = nn::VarStore::new(device);
    let mut model = CausalPatchModel::new(&store.root(), &manifest.model);
    store
        .load(checkpoint.join("model.safetensors"))
        .context("loading universe checkpoint")?;
    check_head_layout(
        store
            .variables()
            .get("head.output.weight")
            .context("checkpoint has no head output weight")?,
    )?;
    model.set_mean_gain(&manifest.mean_gain)?;
    println!(
        "CausalPatch applying the checkpoint's own mean calibration, fitted on {} reserved calibration-partition origins ending {}: anchor gain {:.4} at h=1 and {:.4} at h={}, intrabar offset gain {:.4} and {:.4}{}",
        manifest.mean_gain.blocks.calibration_origins,
        manifest.mean_gain.blocks.calibration_last_origin_ms,
        manifest.mean_gain.anchor[0],
        manifest.mean_gain.anchor[manifest.data.pred_len - 1],
        manifest.data.pred_len,
        manifest.mean_gain.offset[0],
        manifest.mean_gain.offset[manifest.data.pred_len - 1],
        if manifest.mean_gain.is_identity() {
            " (the identity: this checkpoint's calibration block identified no amplitude)"
        } else {
            ""
        }
    );
    store.freeze();
    Ok((manifest, Arc::new(corpus), store, model, device))
}

/// The fit, once per evaluation, in one line: the two curves' endpoints, the shape the run can
/// be compared against a power law by, and the mechanism verdict the in-sample draw decides.
///
/// The log-log slope is printed because it is the one number that makes this curve comparable
/// with the power law the profile was first described by: a gain sweeping 3.662 -> 0.195 over
/// 192 horizons is a slope of -0.56, and the penalized fit is free to differ from it.
fn log_calibration(
    step: usize,
    calibration: &MeanCalibration,
    training: &Moments,
    elapsed_ms: f64,
) {
    let horizons = calibration.anchor.gain.len();
    let last = horizons - 1;
    let slope = |curve: &[f64]| (curve[last] / curve[0]).ln() / (horizons as f64).ln();
    let training_close = &training.channel_gains()[CLOSE_CHANNEL];
    let fitted = &calibration.anchor.measured_gain;
    // The two mechanisms, decided by measurement rather than by argument. `SWEEP_TOLERANCE` is
    // the factor an in-sample amplitude may sweep across the whole horizon axis and still be
    // called flat; the held-out profile that motivated this module sweeps 18.8x.
    const SWEEP_TOLERANCE: f64 = 1.5;
    let sweep = |curve: &[f64]| curve[last] / curve[0];
    let mechanism = match (sweep(training_close), sweep(fitted)) {
        (in_sample, held) if !in_sample.is_finite() || !held.is_finite() => {
            "undetermined (a draw identified no amplitude at an endpoint)"
        }
        (in_sample, _) if in_sample.max(in_sample.recip()) <= SWEEP_TOLERANCE => {
            "OUT-OF-SAMPLE ONLY: the in-sample amplitude is flat across the axis, so the \
             long-horizon over-amplitude is over-fitting and this calibration is the fix"
        }
        _ => {
            "IN SAMPLE TOO: the amplitude is already wrong on the data the weights were fitted \
             to, which is a statement about the objective and not about shrinkage - this \
             calibration is then a patch over a live defect"
        }
    };
    let refusals = [&calibration.anchor, &calibration.offset]
        .into_iter()
        .filter_map(|curve| {
            curve
                .unidentifiable
                .as_deref()
                .map(|reason| format!("; {} identity: {reason}", curve.coordinate.label()))
        })
        .collect::<String>();
    println!(
        "CausalPatch amplitude calibration at step {step} ({elapsed_ms:.0} ms of scoring): close-anchor gain {:.4} at h=1 to {:.4} at h={horizons} (implied log-log slope {:.3}, measured β̂ {:.4} to {:.4}, {:.1} effective dof over {} identified horizons, penalty {:.3e}), intrabar-offset gain {:.4} to {:.4} ({:.1} effective dof over {} identified); in-sample training close gain {:.4} to {:.4}, so the amplitude error is {mechanism}; worst constant-forecast ceiling {:.3e} against worst amplitude cost {:.3e}{refusals}",
        calibration.anchor.gain[0],
        calibration.anchor.gain[last],
        slope(&calibration.anchor.gain),
        fitted[0],
        fitted[last],
        calibration.anchor.effective_dof,
        calibration.anchor.identified,
        calibration.anchor.penalty,
        calibration.offset.gain[0],
        calibration.offset.gain[last],
        calibration.offset.effective_dof,
        calibration.offset.identified,
        training_close[0],
        training_close[last],
        calibration.intercept_ceiling.iter().cloned().fold(0., f64::max),
        calibration.amplitude_cost.iter().cloned().fold(0., f64::max)
    );
}

pub fn evaluate(args: EvaluateArgs) -> Result<()> {
    ensure!(args.batch_size > 0, "batch size must be positive");
    // The mean gain is the checkpoint's own and `load_checkpoint` has already applied it.
    // There is no flag and no artifact: a gain that could be supplied separately from the
    // weights it was fitted on is a gain that can be paired with the wrong weights, and a
    // second application point is a second convention about what the model emits.
    let (manifest, corpus, _store, model, device) =
        load_checkpoint(&args.checkpoint, &args.data_dir)?;
    // `load_checkpoint` hands over the sole reference, so the placement can be replaced in
    // place: the alternative would be a second 28 GB corpus resident at once purely to hold a
    // different origin list.
    let mut corpus = corpus;
    let anchor = |corpus: &mut Arc<Corpus>| -> Result<()> {
        Arc::get_mut(corpus)
            .context("the corpus is shared, so its placement cannot be replaced in place")?
            .anchor_cross_section_draws(CROSS_SECTION_FLOOR)
    };
    if args.placement == Placement::Anchored {
        anchor(&mut corpus)?;
    }
    evaluate_placement(&args, &manifest, &corpus, &model, device, &args.output)?;
    if args.placement == Placement::Both {
        // The comparison this exists for: same process, same weights, same corpus bytes, two
        // placements. Anything that differs between these two report sets is the placement and
        // nothing else.
        anchor(&mut corpus)?;
        let output = args.output.join(corpus::ANCHORED_PLACEMENT);
        evaluate_placement(&args, &manifest, &corpus, &model, device, &output)?;
        println!(
            "CausalPatch placement comparison written: strided at {}, anchored at {}",
            args.output.display(),
            output.display()
        );
    }
    Ok(())
}

/// One evaluation pass over whatever placement `corpus` currently carries.
fn evaluate_placement(
    args: &EvaluateArgs,
    manifest: &Manifest,
    corpus: &Arc<Corpus>,
    model: &CausalPatchModel,
    device: Device,
    output: &Path,
) -> Result<()> {
    let result = score(
        corpus,
        model,
        &corpus.validation_refs,
        args.batch_size,
        device,
    )?;
    // The same draw the training loop charts, scored here too: the full split's own
    // cross-sections are the population and are not comparable with a fixed 256-ticker block,
    // so an epoch-end verdict that only reported the full split could not be laid over the
    // training-time breakeven trajectory it is supposed to conclude.
    let cross_section = cross_section_origins(corpus, &corpus.validation_refs)?;
    let cross_result = score(corpus, model, &cross_section, args.batch_size, device)?;
    let metrics = Metrics {
        step: manifest.step,
        epoch: manifest.epoch,
        completed_origins: manifest.completed_origins,
        total_origins: corpus.train_refs.len(),
        completed_target_bars: manifest.completed_target_bars,
        total_target_bars: corpus.contract.train_target_bars,
        train_nll: None,
        train_mse: None,
        validation_nll: result.nll,
        validation_objective_nll: result.objective_nll,
        persistence_nll: result.persistence_nll,
        validation_mse: result.mse,
        persistence_mse: result.persistence_mse,
        absolute_mse: result.absolute_mse,
        absolute_persistence_mse: result.absolute_persistence_mse,
        within_1_sigma: result.within_1_sigma,
        within_2_sigma: result.within_2_sigma,
        rmse_price: result.rmse_price,
        mae_price: result.mae_price,
        invalid_ohlc_fraction: result.invalid_fraction,
        tail_loss_share: result.tail_loss_share,
        median_window_ratio: result.median_window_ratio,
        eval: result.timing,
        step_ms: None,
        validation_is_full: true,
        validation_origins: corpus.validation_refs.len(),
        tickers: corpus.contract.tickers.len(),
        loader_wait_ms: None,
        loader_build_ms: None,
        peak_allocator_mib: None,
        evaluation_peak_mib: None,
        capture_budget: None,
        step_phases: None,
        horizons: reports::horizon_track(&result.horizon, &result.trading),
        cross_horizons: reports::horizon_track(&cross_result.horizon, &cross_result.trading),
        cross_origins: cross_section.len(),
        in_period_horizons: Vec::new(),
        in_period_origins: 0,
        in_period_purged_row_share: 0.,
        startup: corpus.timing.clone(),
    };
    reports::write_metrics(output, &[metrics])?;
    // Written on EVERY evaluation: the calibration state has to be visible on the same axis as
    // the curves it moves, or a calibrated checkpoint gets step-matched against an
    // uncalibrated one months later with nothing on either chart to say so. The emission here
    // already carries the checkpoint's gain, so the panel reports the un-gained comparand by
    // inverting it in closed form - no second pass, and no fit: this entry point has no
    // calibration-partition population and makes no out-of-sample claim about the fit, only
    // about what the calibrated model scored.
    reports::write_amplitude(
        output,
        &reports::AmplitudePanels {
            epoch: manifest.epoch,
            step: manifest.step,
            tickers: corpus.contract.tickers.len(),
            applied: &manifest.mean_gain,
            fit: None,
            scored: vec![reports::AmplitudeSplit {
                split: reports::FULL,
                origins: corpus.validation_refs.len(),
                emission: reports::Emission::Calibrated,
                moments: &result.amplitude,
            }],
        },
    )?;
    reports::write_horizon(
        output,
        manifest.epoch,
        manifest.step,
        None,
        Some(HorizonSplit {
            curve: &result.horizon,
            origins: corpus.validation_refs.len(),
        }),
        corpus.contract.tickers.len(),
    )?;
    reports::write_trading(
        output,
        manifest.epoch,
        manifest.step,
        None,
        Some(TradingSplit {
            curve: &result.trading,
            portfolio: &result.portfolio,
            origins: corpus.validation_refs.len(),
        }),
        Some(TradingSplit {
            curve: &cross_result.trading,
            portfolio: &cross_result.portfolio,
            origins: cross_section.len(),
        }),
        corpus.contract.tickers.len(),
    )?;
    reports::write_corpus(
        output,
        &corpus.contract,
        corpus.calibration_refs.len(),
        &corpus.market,
    )?;
    reports::write_candles(
        output,
        manifest.epoch,
        manifest.step,
        &candle_windows(corpus, model, device)?,
    )?;
    Ok(())
}
/// How far a within-timestamp statistic may move under a positive per-horizon gain before the
/// run is declared broken rather than reported.
///
/// A positive scalar cannot reorder the rows of one timestamp at one horizon, so the TRUE
/// drift is exactly zero and everything left is fp32 reduction rounding on a rescaled
/// summand. `1e-4` is a thirtieth of the iid IC standard error on the full split (`2.8e-3`),
/// so an implementation that really moved the ranks is caught long before its drift could be
/// mistaken for noise. This is the strongest available self-check on the whole mechanism, so
/// it aborts the run instead of printing a warning.
const IC_INVARIANCE_TOLERANCE: f64 = 1e-4;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::timexer_segment::model::HorizonLoss;
    #[test]
    fn preview_origins_are_fixed_unique_and_cover_partition() {
        let origins: Vec<_> = (800..1300).collect();
        let selected = fixed_origins(&origins, 256).unwrap();
        assert_eq!(selected.len(), 256);
        assert_eq!(selected.first(), Some(&800));
        assert_eq!(selected.last(), Some(&1299));
        assert!(selected.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(fixed_origins(&origins, 1000).unwrap(), origins);
    }
    /// Synthetic final-origin forecasts: 13 windows over three uneven batches, one window with
    /// no valid bars, one with a flat close forecast, one with a zero close target.
    fn synthetic_forecast(windows: i64, pred_len: i64) -> FinalOrigin {
        tch::manual_seed(7);
        let shape = [windows, pred_len, CHANNELS];
        let cpu = (Kind::Float, Device::Cpu);
        let scaled = Tensor::randn(shape, cpu);
        let _ = scaled.select(0, 1).select(-1, CHANNELS - 1).zero_();
        let targets = Tensor::randn(shape, cpu) * 1.5;
        let _ = targets.select(0, 2).select(-1, CHANNELS - 1).zero_();
        let mask = Tensor::rand([windows, pred_len], cpu).lt(0.8).to_kind(Kind::Float);
        let _ = mask.select(0, 3).zero_();
        let drift = Tensor::randn([windows, pred_len, 1], cpu) * 0.5;
        let prices = (&scaled * 0.01).exp() * 100.;
        let target_prices = ((&targets + &drift) * 0.01).exp() * 100.;
        FinalOrigin {
            rebased_prices: ((&scaled + &drift) * 0.01).exp() * 100.,
            log_scale: Tensor::randn(shape, cpu) * 0.3,
            mid: Tensor::randn([windows], cpu) * 0.4,
            sigma: Tensor::rand([windows], cpu) * 0.01 + 0.005,
            scaled,
            targets,
            mask,
            drift,
            prices,
            target_prices,
        }
    }
    fn slice(forecast: &FinalOrigin, start: i64, rows: i64) -> FinalOrigin {
        let cut = |t: &Tensor| t.narrow(0, start, rows);
        FinalOrigin {
            rebased_prices: cut(&forecast.rebased_prices),
            scaled: cut(&forecast.scaled),
            log_scale: cut(&forecast.log_scale),
            targets: cut(&forecast.targets),
            mask: cut(&forecast.mask),
            drift: cut(&forecast.drift),
            prices: cut(&forecast.prices),
            target_prices: cut(&forecast.target_prices),
            mid: cut(&forecast.mid),
            sigma: cut(&forecast.sigma),
        }
    }
    fn host(t: &Tensor) -> Vec<f64> {
        Vec::<f64>::try_from(t.flatten(0, -1).to_kind(Kind::Double)).unwrap()
    }
    fn close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= 1e-5 * (1. + e.abs()), "{a} != {e}");
        }
    }
    #[test]
    fn tensor_scorer_matches_scalar_reference() {
        let _rng = crate::torch::test_rng::exclusive();
        let (w, h, c) = (13usize, 7usize, CHANNELS as usize);
        let forecast = synthetic_forecast(w as i64, h as i64);
        let half_log_horizon = ((Tensor::arange(h as i64, (Kind::Float, Device::Cpu)) + 1.).log() * 0.5)
            .reshape([1, 1, h as i64, 1]);
        let mut scorer = Scorer::new(
            &half_log_horizon,
            &vec![1.0; h],
            &Tensor::zeros([w as i64], (Kind::Int64, Device::Cpu)),
            1,
            h as i64,
            Device::Cpu,
        );
        let mask = host(&forecast.mask);
        let bars_in = |start: usize, rows: usize| -> usize {
            mask[start * h..(start + rows) * h].iter().sum::<f64>() as usize
        };
        for (start, rows) in [(0usize, 5usize), (5, 5), (10, 3)] {
            scorer.accumulate(&slice(&forecast, start as i64, rows as i64), bars_in(start, rows));
        }
        let evaluation = scorer.finish(EvalTiming::default()).unwrap();

        let s = host(&forecast.scaled);
        let t = host(&forecast.targets);
        let ls = host(&forecast.log_scale);
        let at = |i: usize, j: usize, k: usize| (i * h + j) * c + k;
        let m = |i: usize, j: usize| mask[i * h + j];
        let bars: f64 = mask.iter().sum();
        let elements = bars * c as f64;
        let mut sq = 0.;
        let mut pers = 0.;
        let mut within_1 = 0.;
        let mut nll = 0.;
        let mut counts = vec![0.; h];
        let mut mse = vec![0.; h];
        let mut pmse = vec![0.; h];
        let mut abs_err = vec![0.; h];
        let mut abs_target = vec![0.; h];
        let mut wins = vec![0.; h];
        let mut predicted = vec![0.; h];
        let mut hits = vec![0.; h];
        let mut ups = vec![0.; h];
        let mut window_sq = vec![0.; w];
        let mut window_pers = vec![0.; w];
        let mut bar_sq = vec![0.; w * h];
        let mut bar_pers = vec![0.; w * h];
        for i in 0..w {
            for j in 0..h {
                if m(i, j) == 0. {
                    continue;
                }
                counts[j] += 1.;
                for k in 0..c {
                    let e = at(i, j, k);
                    let r = t[e] - s[e];
                    sq += r * r;
                    pers += t[e] * t[e];
                    mse[j] += r * r;
                    pmse[j] += t[e] * t[e];
                    abs_err[j] += r.abs();
                    abs_target[j] += t[e].abs();
                    window_sq[i] += r * r;
                    window_pers[i] += t[e] * t[e];
                    bar_sq[i * h + j] += r * r;
                    bar_pers[i * h + j] += t[e] * t[e];
                    within_1 += f64::from(r.abs() <= ls[e].exp());
                    nll += 0.5 * (r / ls[e].exp()).powi(2) + ls[e];
                }
                let (tc, sc) = (t[at(i, j, c - 1)], s[at(i, j, c - 1)]);
                wins[j] += f64::from((tc - sc).abs() < tc.abs());
                predicted[j] += f64::from(sc != 0.);
                hits[j] += f64::from(sc != 0. && sc.signum() == tc.signum() && tc != 0.);
                ups[j] += f64::from(sc > 0.);
            }
        }
        assert!((evaluation.mse - sq / elements).abs() < 1e-5);
        assert!((evaluation.persistence_mse - pers / elements).abs() < 1e-5);
        assert!((evaluation.nll - nll / elements).abs() < 1e-6);
        assert!((evaluation.within_1_sigma - within_1 / elements).abs() < 1e-5);
        let horizon = &evaluation.horizon;
        let ratio = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a / b).collect::<Vec<_>>();
        let per_bar = |a: &[f64]| a.iter().zip(&counts).map(|(a, n)| a / n).collect::<Vec<_>>();
        close(&horizon.mse, &ratio(&mse, &counts.iter().map(|n| n * c as f64).collect::<Vec<_>>()));
        close(&horizon.persistence_mse, &ratio(&pmse, &counts.iter().map(|n| n * c as f64).collect::<Vec<_>>()));
        close(&horizon.mae_ratio, &ratio(&abs_err, &abs_target));
        close(&horizon.win_rate, &per_bar(&wins));
        close(&horizon.hit_rate, &ratio(&hits, &predicted));
        close(&horizon.up_fraction, &per_bar(&ups));
        assert!(predicted.iter().zip(&counts).any(|(p, n)| p < n), "flat forecasts must be excluded from hit rate");
        // Trimmed ratio: drop the top ⌈n_h/100⌉ = 1 valid |close target| per horizon.
        let mut trimmed = vec![0.; h];
        for j in 0..h {
            let magnitude = |i: usize| t[at(i, j, c - 1)].abs();
            let threshold = (0..w).filter(|&i| m(i, j) > 0.).map(magnitude).fold(f64::MIN, f64::max);
            let (mut num, mut den) = (0., 0.);
            for i in (0..w).filter(|&i| m(i, j) > 0. && magnitude(i) < threshold) {
                num += bar_sq[i * h + j];
                den += bar_pers[i * h + j];
            }
            trimmed[j] = num / den;
        }
        close(&horizon.trimmed_mse_ratio, &trimmed);
        let mut ratios: Vec<f64> = (0..w).filter(|&i| window_pers[i] > 0.).map(|i| window_sq[i] / window_pers[i]).collect();
        assert_eq!(ratios.len(), w - 1, "the empty window is not scored");
        ratios.sort_by(f64::total_cmp);
        assert!((evaluation.median_window_ratio - ratios[(ratios.len() - 1) / 2]).abs() < 1e-5);
        // Tail share: the top ⌈valid elements / 100⌉ |target| elements of each batch.
        let mut tail = 0.;
        let mut widest = 0;
        for (start, rows) in [(0usize, 5usize), (5, 5), (10, 3)] {
            let magnitude = |e: usize| t[e].abs() * m(e / (h * c), e / c % h);
            let elements = start * h * c..(start + rows) * h * c;
            let mut magnitudes: Vec<f64> = elements.clone().map(magnitude).collect();
            magnitudes.sort_by(f64::total_cmp);
            let k = (bars_in(start, rows) * c).div_ceil(100);
            widest = widest.max(k);
            let threshold = magnitudes[magnitudes.len() - k];
            for e in elements.filter(|&e| magnitude(e) >= threshold) {
                tail += (t[e] - s[e]).powi(2);
            }
        }
        assert!(widest > 1, "the reference must exercise a multi-element tail");
        assert!((evaluation.tail_loss_share - tail / sq).abs() < 1e-5);
    }
    /// Checkpoint selection and early stopping read the TRAINING objective, so the held-out
    /// scalar they minimize has to be the training-weighted NLL and nothing else. Pinned
    /// against `Σ w·mask·nll / Σ w·mask·CHANNELS` computed scalar by scalar on the host - the
    /// same functional `CausalPatchModel::losses` minimizes, evaluated on the held-out split -
    /// for every mode, and pinned to the plain aggregate EXACTLY under `uniform`, which is what
    /// makes a `uniform` arm comparable to every curve recorded before the knob existed.
    #[test]
    fn the_selection_scalar_is_the_training_weighted_held_out_nll() {
        let _rng = crate::torch::test_rng::exclusive();
        let (w, h, c) = (13usize, 7usize, CHANNELS as usize);
        let forecast = synthetic_forecast(w as i64, h as i64);
        let half_log_horizon = ((Tensor::arange(h as i64, (Kind::Float, Device::Cpu)) + 1.).log()
            * 0.5)
            .reshape([1, 1, h as i64, 1]);
        let mask = host(&forecast.mask);
        let (s, t, ls) = (
            host(&forecast.scaled),
            host(&forecast.targets),
            host(&forecast.log_scale),
        );
        let bars_in = |start: usize, rows: usize| -> usize {
            mask[start * h..(start + rows) * h].iter().sum::<f64>() as usize
        };
        for mode in [
            HorizonLoss::Uniform,
            HorizonLoss::InverseSqrt,
            HorizonLoss::Inverse,
            HorizonLoss::Cutoff(3),
        ] {
            let weights = mode.weights(h as i64);
            let mut scorer = Scorer::new(
                &half_log_horizon,
                &weights,
                &Tensor::zeros([w as i64], (Kind::Int64, Device::Cpu)),
                1,
                h as i64,
                Device::Cpu,
            );
            for (start, rows) in [(0usize, 5usize), (5, 5), (10, 3)] {
                scorer.accumulate(
                    &slice(&forecast, start as i64, rows as i64),
                    bars_in(start, rows),
                );
            }
            let evaluation = scorer.finish(EvalTiming::default()).unwrap();
            let (mut numerator, mut denominator) = (0., 0.);
            for row in 0..w {
                for step in 0..h {
                    if mask[row * h + step] == 0. {
                        continue;
                    }
                    denominator += weights[step] * c as f64;
                    for channel in 0..c {
                        let e = (row * h + step) * c + channel;
                        numerator += weights[step]
                            * (0.5 * ((t[e] - s[e]) / ls[e].exp()).powi(2) + ls[e]);
                    }
                }
            }
            let expected = numerator / denominator;
            assert!(
                (evaluation.objective_nll - expected).abs() <= 1e-6 * (1. + expected.abs()),
                "{mode}: selection scalar {} is not the weighted NLL {expected}",
                evaluation.objective_nll
            );
            if mode == HorizonLoss::Uniform {
                // Both the numerator and the denominator are the unweighted ones bit for bit:
                // the fp32 fold by 1.0 is the identity and the mask reduction of a 0/1 tensor
                // is an exact integer in fp64.
                assert_eq!(
                    evaluation.objective_nll, evaluation.nll,
                    "uniform must not redefine what the selection scalar means"
                );
            } else {
                assert!(
                    (evaluation.objective_nll - evaluation.nll).abs() > 1e-3,
                    "{mode} left the selection scalar equal to the aggregate {}",
                    evaluation.nll
                );
            }
        }
    }
    /// The trading diagnostics against a straightforward scalar reference. Ninety windows over
    /// three thirty-window evaluation timestamps and sixteen horizons, accumulated in three
    /// uneven batches. Relative tolerances reflect fp32 elementwise arithmetic followed by
    /// fp64 reduction: `1e-6` for gains, correlations, anchored ratios and deciles; `1e-4`
    /// for the cross-sectional IC, whose per-timestamp scatter accumulates in fp32.
    #[test]
    fn trading_statistics_match_scalar_reference() {
        let _rng = crate::torch::test_rng::exclusive();
        let (w, h, c) = (90usize, 16usize, CHANNELS as usize);
        let per_group = 30usize;
        let groups = w / per_group;
        let forecast = synthetic_forecast(w as i64, h as i64);
        let ids: Vec<i64> = (0..w).map(|i| (i / per_group) as i64).collect();
        let half_log_horizon = ((Tensor::arange(h as i64, (Kind::Float, Device::Cpu)) + 1.).log()
            * 0.5)
            .reshape([1, 1, h as i64, 1]);
        let mut scorer = Scorer::new(
            &half_log_horizon,
            &vec![1.0; h],
            &Tensor::from_slice(&ids),
            groups,
            h as i64,
            Device::Cpu,
        );
        let mask = host(&forecast.mask);
        for (start, rows) in [(0usize, 40usize), (40, 25), (65, 25)] {
            let bars = mask[start * h..(start + rows) * h].iter().sum::<f64>() as usize;
            scorer.accumulate(&slice(&forecast, start as i64, rows as i64), bars);
        }
        let evaluation = scorer.finish(EvalTiming::default()).unwrap();
        let curve = &evaluation.trading;

        let s = host(&forecast.scaled);
        let t = host(&forecast.targets);
        let mid = host(&forecast.mid);
        let at = |i: usize, j: usize, k: usize| (i * h + j) * c + k;
        let m = |i: usize, j: usize| mask[i * h + j];
        let fc = |i: usize, j: usize| s[at(i, j, c - 1)] * m(i, j);
        let yc = |i: usize, j: usize| t[at(i, j, c - 1)] * m(i, j);
        // sign(a)·sign(b) > 0 over the bars where both signs are nonzero.
        let hit_rate = |members: &[usize], j: usize, signal: &dyn Fn(usize) -> f64| {
            let (mut hit, mut bet) = (0., 0.);
            for &i in members {
                let (p, r) = (signal(i).signum(), yc(i, j).signum());
                if signal(i) != 0. && yc(i, j) != 0. {
                    bet += 1.;
                    hit += f64::from(p * r > 0.);
                }
            }
            hit / bet
        };
        let pearson = |members: &[usize], a: &dyn Fn(usize) -> f64, b: &dyn Fn(usize) -> f64| {
            let n = members.len() as f64;
            let (ma, mb) = (
                members.iter().map(|&i| a(i)).sum::<f64>() / n,
                members.iter().map(|&i| b(i)).sum::<f64>() / n,
            );
            let cov = members.iter().map(|&i| a(i) * b(i)).sum::<f64>() / n - ma * mb;
            let va = members.iter().map(|&i| a(i) * a(i)).sum::<f64>() / n - ma * ma;
            let vb = members.iter().map(|&i| b(i) * b(i)).sum::<f64>() / n - mb * mb;
            cov / (va * vb).sqrt()
        };
        let mut usable_cross_sections = 0;
        for j in 0..h {
            let valid: Vec<usize> = (0..w).filter(|&i| m(i, j) > 0.).collect();
            let n = valid.len() as f64;
            assert!(n > 0.);
            let sum = |g: &dyn Fn(usize) -> f64| valid.iter().map(|&i| g(i)).sum::<f64>();
            let mu = sum(&|i| fc(i, j)) / n;
            let ybar = sum(&|i| yc(i, j)) / n;
            let persistence = sum(&|i| yc(i, j).powi(2)) / n;
            let error = sum(&|i| (yc(i, j) - fc(i, j)).powi(2)) / n;
            let signal_square = sum(&|i| fc(i, j).powi(2)) / n - mu * mu;
            let signal_target = sum(&|i| fc(i, j) * yc(i, j)) / n - mu * ybar;
            let total = 1. - error / persistence;
            let offset = (2. * mu * ybar - mu * mu) / persistence;
            let demeaned = signal_target * signal_target / signal_square / persistence;
            let tight = |actual: f64, expected: f64, what: &str| {
                assert!(
                    (actual - expected).abs() <= 1e-6 * (1. + expected.abs()),
                    "{what} at h={}: {actual} != {expected}",
                    j + 1
                );
            };
            tight(curve.total_gain[j], total, "total gain");
            tight(curve.offset_gain[j], offset, "offset gain");
            tight(curve.demeaned_gain[j], demeaned, "demeaned gain");
            tight(
                curve.scaling_gain[j],
                total - offset - demeaned,
                "cross term",
            );
            // The cross term is exactly the cost of mis-scaling the demeaned forecast, which
            // is what makes the three components a decomposition rather than three numbers.
            let slope = signal_target / signal_square;
            tight(
                curve.scaling_gain[j],
                -(slope - 1.).powi(2) * signal_square / persistence,
                "cross term identity",
            );
            tight(curve.mean_forecast[j], mu, "mean forecast");
            tight(curve.mean_target[j], ybar, "mean target");
            tight(curve.close_mse_ratio[j], error / persistence, "MSE ratio");
            let all_error = sum(&|i| (0..c).map(|k| (t[at(i, j, k)] - s[at(i, j, k)]).powi(2)).sum());
            let all_persistence = sum(&|i| (0..c).map(|k| t[at(i, j, k)].powi(2)).sum());
            tight(
                curve.all_channel_gain[j],
                1. - all_error / all_persistence,
                "all-channel gain",
            );
            tight(
                curve.pearson[j],
                pearson(&valid, &|i| fc(i, j), &|i| yc(i, j)),
                "Pearson IC",
            );
            // Spearman: ascending ordinal ranks with invalid bars pushed past every valid one.
            let key = |value: f64, valid: bool| if valid { value } else { 1e30 };
            let ranks = |g: &dyn Fn(usize) -> f64| -> Vec<f64> {
                let keys: Vec<f64> = (0..w).map(|i| key(g(i), m(i, j) > 0.)).collect();
                (0..w)
                    .map(|i| {
                        (0..w)
                            .filter(|&o| keys[o] < keys[i] || (keys[o] == keys[i] && o < i))
                            .count() as f64
                    })
                    .collect()
            };
            let (rf, ry) = (ranks(&|i| fc(i, j)), ranks(&|i| yc(i, j)));
            tight(
                curve.spearman[j],
                pearson(&valid, &|i| rf[i], &|i| ry[i]),
                "Spearman IC",
            );
            // Mid anchor: the numerator is unchanged, the denominator re-anchors on the log
            // high/low midpoint of the origin bar.
            tight(
                curve.mid_anchor_mse_ratio[j],
                sum(&|i| (yc(i, j) - fc(i, j)).powi(2)) / sum(&|i| (yc(i, j) - mid[i]).powi(2)),
                "mid-anchored MSE ratio",
            );
            let mid_hit = {
                let (mut hit, mut bet) = (0., 0.);
                for &i in &valid {
                    let (p, r) = (fc(i, j) - mid[i], yc(i, j) - mid[i]);
                    if p != 0. && r != 0. {
                        bet += 1.;
                        hit += f64::from(p.signum() * r.signum() > 0.);
                    }
                }
                hit / bet
            };
            tight(curve.mid_anchor_hit_rate[j], mid_hit, "mid-anchored hit rate");
            tight(
                curve.close_hit_rate[j],
                hit_rate(&valid, j, &|i| fc(i, j)),
                "close hit rate",
            );
            // Delayed neutral-coordinate quality, not execution P&L.
            let delayed: Vec<usize> = (0..w)
                .filter(|&i| m(i, j) > 0. && m(i, 0) > 0.)
                .collect();
            let (mut delay_error, mut delay_persistence) = (0., 0.);
            let (mut delay_hit, mut delay_bet) = (0., 0.);
            for &i in &delayed {
                let (d, p) = (yc(i, j) - yc(i, 0), fc(i, j) - fc(i, 0));
                delay_error += (d - p).powi(2);
                delay_persistence += d * d;
                if d != 0. && p != 0. {
                    delay_bet += 1.;
                    delay_hit += f64::from(d.signum() * p.signum() > 0.);
                }
            }
            if j == 0 {
                assert_eq!(delay_persistence, 0.);
                assert!(
                    curve.delayed_mse_ratio[0].is_nan(),
                    "the delayed move is identically zero at h = 1"
                );
            } else {
                tight(
                    curve.delayed_mse_ratio[j],
                    delay_error / delay_persistence,
                    "delayed MSE ratio",
                );
                tight(
                    curve.delayed_hit_rate[j],
                    delay_hit / delay_bet,
                    "delayed hit rate",
                );
            }
            // Conviction deciles on the demeaned signal.
            let magnitude = |i: usize| (fc(i, j) - mu).abs();
            let k = ((n / 10.).floor() as usize).max(1);
            assert!(k >= 2, "the reference must exercise a multi-member decile");
            let mut sorted: Vec<f64> = valid.iter().map(|&i| magnitude(i)).collect();
            sorted.sort_by(f64::total_cmp);
            let (low, high) = (sorted[k - 1], sorted[sorted.len() - k]);
            let top: Vec<usize> = valid.iter().copied().filter(|&i| magnitude(i) >= high).collect();
            let bottom: Vec<usize> = valid.iter().copied().filter(|&i| magnitude(i) <= low).collect();
            let side = |members: &[usize]| {
                members
                    .iter()
                    .map(|&i| (fc(i, j) - mu).signum() * yc(i, j))
                    .sum::<f64>()
                    / members.len() as f64
            };
            tight(curve.top_decile_return[j], side(&top), "top-decile return");
            tight(
                curve.bottom_decile_return[j],
                side(&bottom),
                "bottom-decile return",
            );
            tight(
                curve.conviction_spread_return[j],
                side(&top) - side(&bottom),
                "conviction spread",
            );
            tight(
                curve.top_decile_hit_rate[j],
                hit_rate(&top, j, &|i| fc(i, j) - mu),
                "top-decile hit rate",
            );
            tight(
                curve.bottom_decile_hit_rate[j],
                hit_rate(&bottom, j, &|i| fc(i, j) - mu),
                "bottom-decile hit rate",
            );
            // Cross-sectional IC: within-timestamp correlation, then averaged.
            let mut moments = Vec::new();
            for g in 0..groups {
                let members: Vec<usize> = (g * per_group..(g + 1) * per_group)
                    .filter(|&i| m(i, j) > 0.)
                    .collect();
                if members.len() < CROSS_SECTION_MIN as usize {
                    continue;
                }
                moments.push(pearson(&members, &|i| fc(i, j), &|i| yc(i, j)));
            }
            usable_cross_sections += moments.len();
            let count = moments.len() as f64;
            let ic = moments.iter().sum::<f64>() / count;
            let variance = moments.iter().map(|v| (v - ic).powi(2)).sum::<f64>() / count;
            assert!(
                (curve.cross_sectional_ic[j] - ic).abs() <= 1e-4,
                "cross-sectional IC at h={}: {} != {ic}",
                j + 1,
                curve.cross_sectional_ic[j]
            );
            assert!(
                (curve.cross_sectional_ic_se[j] - (variance / count).sqrt()).abs() <= 1e-4,
                "cross-sectional IC standard error at h={}",
                j + 1
            );
        }
        assert!(
            usable_cross_sections >= h,
            "the reference must exercise the cross-sectional path at every horizon"
        );
        assert_eq!(curve.cross_sections, groups);
    }
    /// The MSE-optimal gain against a fixture where `β`, `σ_y` and `σ_f` are set BY
    /// CONSTRUCTION, together with the two statistics that must stay NaN when they are
    /// undefined.
    ///
    /// `f` is mean-zero with unit variance, `e` is mean-zero and orthogonal to `f`, and the
    /// target at horizon `j` is `y_j = β_j·f + e`. Then `Cov(f, y_j) = β_j·Var(f)` exactly and
    /// `β̂_j = Cov/Var = β_j` for every residual, so this pins an identity rather than a fit.
    /// The point of the fixture is the case it reproduces: at `β = 0.25` the close MSE ratio is
    /// 1.123 - WORSE than persistence - while the same forecast rescaled would score 0.985.
    /// A reader with only the ratio concludes "no signal"; the two series together say
    /// "signal, amplitude 4x too large", which is the failure this family exists to expose.
    ///
    /// `β̂` is emitted from the moments and never inverted out of the gain decomposition:
    /// `C = -(β̂-1)²·Var(f)/P` and `D = Cov²/(Var(f)·P)` admit TWO positive `β̂` whenever
    /// `|C| < D`, which holds at three of the four horizons here.
    #[test]
    fn the_optimal_gain_reproduces_its_analytic_value_and_undefined_statistics_stay_nan() {
        let (w, h, c) = (8usize, 4usize, CHANNELS as usize);
        let cpu = (Kind::Float, Device::Cpu);
        let sign = |i: usize| if i % 2 == 0 { 1.0_f64 } else { -1.0 };
        let noise = |i: usize| if (i / 2) % 2 == 0 { 2.0_f64 } else { -2.0 };
        let beta = |j: usize| 0.25 * (j + 1) as f64;
        // Non-close channels predict their target exactly, so the per-horizon coverage row is
        // 3/4 plus the close channel's own share: a row that reduced the wrong axis, or the
        // aggregate, could not produce that number.
        let mut scaled = vec![0.5_f32; w * h * c];
        let mut targets = vec![0.5_f32; w * h * c];
        for i in 0..w {
            for j in 0..h {
                let index = (i * h + j) * c + c - 1;
                scaled[index] = sign(i) as f32;
                targets[index] = (beta(j) * sign(i) + noise(i)) as f32;
            }
        }
        let shape = [w as i64, h as i64, CHANNELS];
        let forecast = FinalOrigin {
            scaled: Tensor::from_slice(&scaled).reshape(shape),
            targets: Tensor::from_slice(&targets).reshape(shape),
            log_scale: Tensor::zeros(shape, cpu),
            mask: Tensor::ones([w as i64, h as i64], cpu),
            drift: Tensor::zeros([w as i64, h as i64, 1], cpu),
            prices: Tensor::full(shape, 100., cpu),
            rebased_prices: Tensor::full(shape, 100., cpu),
            target_prices: Tensor::full(shape, 100., cpu),
            mid: Tensor::zeros([w as i64], cpu),
            sigma: Tensor::full([w as i64], 0.01, cpu),
        };
        let mut scorer = Scorer::new(
            &((Tensor::arange(h as i64, cpu) + 1.).log() * 0.5).reshape([1, 1, h as i64, 1]),
            &vec![1.0; h],
            &Tensor::zeros([w as i64], (Kind::Int64, Device::Cpu)),
            1,
            h as i64,
            Device::Cpu,
        );
        // Two batches, because the moments are accumulated across batches and the identity has
        // to survive that rather than hold only within one.
        for (start, rows) in [(0usize, 3usize), (3, 5)] {
            scorer.accumulate(&slice(&forecast, start as i64, rows as i64), rows * h);
        }
        let evaluation = scorer.finish(EvalTiming::default()).unwrap();
        let curve = &evaluation.trading;
        for j in 0..h {
            let b = beta(j);
            let persistence = b * b + 4.;
            let tight = |actual: f64, expected: f64, what: &str| {
                assert!(
                    (actual - expected).abs() <= 1e-6 * (1. + expected.abs()),
                    "{what} at h = {}: {actual} != {expected}",
                    j + 1
                );
            };
            tight(curve.optimal_gain[j], b, "MSE-optimal gain");
            tight(
                curve.pearson[j],
                b / persistence.sqrt(),
                "pooled Pearson (ρ = β·σ_f/σ_y)",
            );
            tight(
                curve.best_scale_mse_ratio[j],
                4. / persistence,
                "best-scale close MSE ratio",
            );
            tight(
                curve.close_mse_ratio[j],
                ((b - 1.) * (b - 1.) + 4.) / persistence,
                "achieved close MSE ratio",
            );
            tight(
                curve.scaling_gain[j],
                -(b - 1.) * (b - 1.) / persistence,
                "mis-scaling cross term",
            );
            tight(curve.offset_gain[j], 0., "offset gain of a mean-zero forecast");
            tight(
                curve.total_gain[j],
                curve.offset_gain[j] + curve.demeaned_gain[j] + curve.scaling_gain[j],
                "the three-part identity",
            );
            // The demonstrated failure mode, at the horizon that reproduces it: the ratio says
            // the forecast is worse than persistence while the amplitude fix says it is better.
            if j == 0 {
                assert!(curve.close_mse_ratio[j] > 1. && curve.best_scale_mse_ratio[j] < 1.);
            }
            // Why β̂ is emitted from the moments and never reconstructed from the gain
            // decomposition: the cross term only fixes |β̂ - 1|, so inverting it yields TWO
            // admissible positive amplitudes, `1 ± d`, and nothing in the decomposition says
            // which. Here `1 - d` is the true one and `1 + d` is spurious, and the two are far
            // apart - 0.25 against 1.75 at the first horizon.
            let d = (curve.scaling_gain[j].abs() * persistence).sqrt();
            if b < 1. {
                assert!(
                    (1. - d - b).abs() < 1e-6 && 1. - d > 0. && (1. + d - b).abs() > 1e-3,
                    "the inverted cross term must admit two positive branches at h = {}: \
                     1 ± {d} against β̂ = {b}",
                    j + 1
                );
            }
            // Eight names is below the cross-section minimum, so no timestamp qualifies: the
            // IC and its standard error are undefined and the population count says why. A 0
            // in any of the three would be a measurement claim this draw cannot support.
            assert!(curve.cross_sectional_ic[j].is_nan());
            assert!(curve.cross_sectional_ic_se[j].is_nan());
            assert_eq!(curve.cross_sectional_ic_moments[j], 0.);
            // Per-horizon σ coverage. The close residual is |(β-1)·f + e|, which never fits
            // inside 1σ here and fits inside 1.96σ for half the rows until β reaches 1, so the
            // 1.96σ row separates h = 4 from the rest - a slice of the aggregate could not.
            let curves = &evaluation.horizon;
            tight(curves.within_1_sigma[j], 0.75, "coverage within 1σ");
            tight(
                curves.within_2_sigma[j],
                if b < 1. { 0.875 } else { 0.75 },
                "coverage within 1.96σ",
            );
            assert_eq!(curves.valid_elements[j], (w * c as usize) as f64);
        }
    }
    #[test]
    fn universe_checkpoint_authenticates_objective_schema_and_weight_bytes() {
        use super::super::{corpus::ExcludedTicker, data::DataContract, features::FeatureSet};
        struct Scratch(PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let directory = Scratch(PathBuf::from(format!(
            "/var/tmp/timexer-manifest-test-{}",
            uuid::Uuid::new_v4()
        )));
        fs::create_dir(&directory.0).unwrap();
        let payload = b"test-only weight payload; no model execution";
        fs::write(directory.0.join("model.safetensors"), payload).unwrap();
        let model = ModelConfig {
            seq_len: 16,
            pred_len: 7,
            min_history: 16,
            features: FeatureSet::NONE,
            // Deliberately NOT the default: the selection stamp has to be checked against a
            // mode that is distinguishable from `uniform`.
            horizon_loss: HorizonLoss::Cutoff(4),
            // Same reason, on the axis that moves the parameter SET: a `free` stamp would be
            // the suffix every default config produces, so it could not catch a stamp that
            // drops the mean spec entirely.
            horizon_mean: HorizonMean::Basis {
                free: 2,
                functions: 2,
            },
            ..ModelConfig::default()
        };
        let ticker = DataContract {
            schema: "test-sigma-scaled".into(),
            ticker: "EXAMPLE".into(),
            fingerprint: "0123456789abcdef".into(),
            source_bars: 10_000,
            valid_bars: 10_000,
            invalid_ohlc_indices: vec![],
            boundaries: [7000, 8000, 9000],
            boundary_timestamps: [700_000, 800_000, 900_000],
            context: 16,
            pred_len: 7,
            purge: 100,
            train_end: 6900,
            common_context: 32,
        };
        let data = CorpusContract {
            schema: "authenticated-pooled-training".into(),
            tickers: vec![ticker],
            boundary_timestamps: [700_000, 800_000, 900_000],
            context: 16,
            pred_len: 7,
            common_context: 32,
            purge: 100,
            features: FeatureSet::NONE,
            auxiliary_schema: String::new(),
            spy_fingerprint: None,
            market_fingerprint: "fedcba9876543210".into(),
            market_min_cross_section: 2000,
            minimum_source_bars: 133,
            minimum_training_bars: 33,
            train_target_bars: 6868,
            validation_target_bars: 896,
            validation_remainder_bars: 4,
            in_period_sections: 0,
            in_period_anchors: vec![],
            in_period_census: vec![],
            in_period_origins: 0,
            in_period_target_bars: 0,
            in_period_purged_rows: 0,
            in_period_purged_target_bars: 0,
            cross_section_placement: String::new(),
            calibration_anchors: vec![],
            validation_anchors: vec![],
            excluded_tickers: vec![ExcludedTicker {
                ticker: "NEW".into(),
                reason: "no historical training targets".into(),
            }],
        };
        let mut manifest = Manifest {
            format: format_stamp(model.x0_lambdas, model.horizon_mean),
            selection: selection_criterion(&model),
            model,
            data,
            objective: OBJECTIVE.into(),
            numerics: NUMERICS.into(),
            requested_tickers: vec![],
            seed: 20260905,
            epoch: 1,
            step: 1000,
            completed_origins: 256000,
            completed_target_bars: 49152000,
            epoch_complete: false,
            planned_epochs: 10,
            max_steps: None,
            row_selection: None,
            termination: None,
            batch_size: 256,
            learning_rate: 0.0048,
            base_learning_rate: 0.008,
            scalar_lr_mult: NANOGPT_SCALAR_LR_MULTIPLIER,
            fused: true,
            optimizer: OptimizerKind::PolarExpress,
            optimizer_recipe: OptimizerKind::PolarExpress
                .recipe(RecipeKnobs::reference(X0Lambdas::Enabled)),
            eval_every: 1000,
            eval_origins: 2048,
            validation_nll: 1.3,
            validation_mse: 0.004194157171231031,
            validation_is_full: false,
            best_step: Some(1000),
            best_objective_nll: Some(1.3),
            weights_sha256: file_sha256(directory.0.join("model.safetensors")).unwrap(),
            // A seven-horizon frozen gain, non-identity so the digest covers a real curve, with
            // one gated horizon carried signed: the manifest is the only place the applied
            // amplitude is recorded, so it has to survive the round trip and the digest.
            mean_gain: FrozenGain {
                estimator: "test".into(),
                blocks: Blocks {
                    calibration_first_origin_ms: 700_000,
                    calibration_last_origin_ms: 780_000,
                    calibration_last_target_ms: 790_000,
                    calibration_origins: 512,
                    evaluation_first_origin_ms: 800_000,
                    evaluation_last_origin_ms: 890_000,
                    evaluation_origins: 128,
                    purge_gap_ms: 10_000,
                },
                anchor: vec![3.662, 2.4, 1.9, 1.2, 0.9, 0.5, 0.2],
                offset: vec![1.1, 1.05, 1.0, 0.98, 0.95, 0.9, 0.85],
                measured_anchor: vec![
                    Some(3.7),
                    Some(2.3),
                    Some(1.95),
                    Some(1.18),
                    Some(0.88),
                    Some(-0.04),
                    None,
                ],
            },
            manifest_sha256: String::new(),
        };
        manifest.manifest_sha256 = manifest.digest().unwrap();
        let write = |value: &Manifest| {
            fs::write(
                directory.0.join("manifest.json"),
                serde_json::to_vec_pretty(value).unwrap(),
            )
            .unwrap()
        };
        write(&manifest);
        let restored = Manifest::read(&directory.0).unwrap();
        assert_eq!(restored.digest().unwrap(), manifest.digest().unwrap());
        assert_eq!(restored.data, manifest.data);
        // An uncapped run's manifest carries NEITHER new field, so its JSON - and the digest
        // authenticating it - is byte-identical to a manifest written before the cap existed
        // and every checkpoint already on disk still authenticates.
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(
            !json.contains("max_steps") && !json.contains("termination"),
            "an uncapped run must write neither field: {json}"
        );
        // A capped arm's manifest states the cap and why it stopped, in that spelling, which
        // is what keeps it from being read as an arm that early-stopped on patience or ran
        // its epoch out.
        let mut capped = manifest.clone();
        capped.max_steps = Some(4000);
        capped.termination = Some(Termination::StepCap);
        capped.manifest_sha256 = capped.digest().unwrap();
        write(&capped);
        let restored = Manifest::read(&directory.0).unwrap();
        assert_eq!(restored.max_steps, Some(4000));
        assert_eq!(restored.termination, Some(Termination::StepCap));
        let json = serde_json::to_string(&capped).unwrap();
        assert!(
            json.contains(r#""max_steps":4000"#) && json.contains(r#""termination":"step-cap""#),
            "{json}"
        );
        for reason in [
            Termination::PreviewPatience,
            Termination::EpochPatience,
            Termination::EpochLimit,
        ] {
            let mut other = manifest.clone();
            other.termination = Some(reason);
            other.manifest_sha256 = other.digest().unwrap();
            write(&other);
            assert_eq!(
                Manifest::read(&directory.0).unwrap().termination,
                Some(reason)
            );
        }
        write(&manifest);
        let mut previous_objective = manifest.clone();
        previous_objective.objective = "causal_patch_nll_v1".into();
        previous_objective.manifest_sha256 = previous_objective.digest().unwrap();
        write(&previous_objective);
        let error = Manifest::read(&directory.0).unwrap_err().to_string();
        assert!(error.contains("objective") && error.contains(OBJECTIVE), "{error}");
        // The pre-channel-major stamp specifically: `causal-patch-ohlc-universe-v5` names a
        // head whose weight rows are horizon-major, and every tensor name and shape in it still
        // matches the current model, so nothing but this string stands between a stale
        // checkpoint and a silently permuted head. `…-v6-head-channel-major-folded-mup` is the
        // pre-recipe backbone: its state dict carries 17 norm gains and biases and four block
        // biases per layer that this model does not have, and none of the recipe scalars that
        // it does. `…-v7-…-unet-vres` is this parameter set exactly, but its manifest records
        // neither the scalar learning-rate multiplier nor the x0 mode its weights were trained
        // under, so a `v7` checkpoint cannot be attributed to a configuration and is refused by
        // name rather than silently assumed to be the 5x learned-x0 arm. `…-v8-…` is this
        // parameter set too and its loss is bit-identical to `uniform`, but its manifest has no
        // `horizon_loss` at all, so a `v8` checkpoint cannot be attributed to a horizon
        // weighting and its curves would be averaged in with arms that weight the objective
        // differently.
        // `…-v9-…-x0-*` is the horizon-weighted objective's stamp. Its `free`-equivalent head
        // weight has the same shape as `--horizon-mean free` here and its manifest carries a
        // `horizon_loss`, so both halves of a `v9` checkpoint look loadable - but it records no
        // `horizon_mean`, so nothing distinguishes a `v9` dense-mean checkpoint from a `v10`
        // one, and a `basis` arm's curves would be averaged in with arms whose long-horizon
        // means were unrestricted. That is the same failure the `v8` bump names, on the axis
        // that this time really does change the tensor shapes.
        for stale in [
            "timexer-ohlc-universe-v4",
            "causal-patch-ohlc-universe-v5",
            "causal-patch-ohlc-universe-v6-head-channel-major-folded-mup",
            "causal-patch-ohlc-universe-v7-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres",
            "causal-patch-ohlc-universe-v8-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres-x0-learned",
            "causal-patch-ohlc-universe-v8-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres-x0-none",
            "causal-patch-ohlc-universe-v9-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres-x0-learned",
            "causal-patch-ohlc-universe-v9-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres-x0-none",
        ] {
            let mut previous_format = manifest.clone();
            previous_format.format = stale.into();
            previous_format.manifest_sha256 = previous_format.digest().unwrap();
            write(&previous_format);
            let error = Manifest::read(&directory.0).unwrap_err().to_string();
            assert!(
                error.contains("format") && error.contains(stale) && error.contains(FORMAT_X0_LEARNED),
                "{error}"
            );
        }
        // A stamp that is accepted on its own but describes the OTHER parameter set: the
        // no-x0 stamp against a manifest declaring the learned injection. Both halves are
        // individually valid, so only the cross-check catches it - and it must, because the
        // stamp is what a state dict's tensor list is authenticated against.
        let mut mismatched = manifest.clone();
        mismatched.format = FORMAT_X0_NONE.into();
        mismatched.manifest_sha256 = mismatched.digest().unwrap();
        write(&mismatched);
        let error = Manifest::read(&directory.0).unwrap_err().to_string();
        assert!(
            error.contains("does not match") && error.contains("Enabled"),
            "{error}"
        );
        // The same cross-check on the selection criterion: a checkpoint whose manifest claims
        // it was chosen on the uniform objective while its own model config says `cutoff:4`
        // would attribute a selection decision to the wrong arm. Both halves are individually
        // well-formed, so only the cross-check catches it.
        let mut mismatched = manifest.clone();
        mismatched.selection = selection_criterion(&ModelConfig {
            horizon_loss: HorizonLoss::Uniform,
            ..manifest.model.clone()
        });
        mismatched.manifest_sha256 = mismatched.digest().unwrap();
        write(&mismatched);
        let error = Manifest::read(&directory.0).unwrap_err().to_string();
        assert!(
            error.contains("selection") && error.contains("cutoff:4"),
            "{error}"
        );
        write(&manifest);
        let mut corrupted = manifest.clone();
        corrupted.data.tickers[0].train_end += 1;
        write(&corrupted);
        assert!(Manifest::read(&directory.0)
            .unwrap_err()
            .to_string()
            .contains("manifest authentication"));
        let mut corrupted = manifest.clone();
        corrupted.data.excluded_tickers[0].ticker = "REPLACED".into();
        write(&corrupted);
        assert!(Manifest::read(&directory.0)
            .unwrap_err()
            .to_string()
            .contains("manifest authentication"));
        let mut mismatched = manifest.clone();
        mismatched.data.context = 32;
        mismatched.manifest_sha256 = mismatched.digest().unwrap();
        write(&mismatched);
        assert!(Manifest::read(&directory.0)
            .unwrap_err()
            .to_string()
            .contains("model/data contract mismatch"));
        write(&manifest);
        fs::write(directory.0.join("model.safetensors"), b"changed payload").unwrap();
        assert!(Manifest::read(&directory.0)
            .unwrap_err()
            .to_string()
            .contains("weight authentication"));
    }
    /// A head weight whose rows are horizon-major is refused, and the channel-major one it was
    /// permuted from is accepted. The synthetic weight carries the structure the guard relies
    /// on and nothing else: a per-channel signature three times the size of a step, plus a
    /// random walk along the horizon, so within a channel the rows drift smoothly and across
    /// channels they are unrelated.
    #[test]
    fn horizon_major_head_weights_are_refused() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(11);
        let (channels, pred_len, hidden) = (2 * CHANNELS, 24i64, 16i64);
        let cpu = (Kind::Float, Device::Cpu);
        let walk = Tensor::randn([channels, pred_len, hidden], cpu).cumsum(1, Kind::Float);
        let signature = Tensor::randn([channels, 1, hidden], cpu) * 3.;
        let major = (walk + signature).reshape([channels * pred_len, hidden]);
        check_head_layout(&major).expect("the channel-major layout the model reshapes to");
        let stale = major
            .reshape([channels, pred_len, hidden])
            .transpose(0, 1)
            .contiguous()
            .reshape([channels * pred_len, hidden]);
        let error = check_head_layout(&stale).unwrap_err().to_string();
        assert!(
            error.contains("horizon-major") && error.contains("retrain"),
            "{error}"
        );
        // An untrained head carries no layout, so the guard must not invent a verdict.
        check_head_layout(&Tensor::zeros([channels * pred_len, hidden], cpu)).unwrap();
    }
    /// The invariant that would have caught the permuted head with no checkpoint at all: an
    /// exact-zero forecast IS persistence, so every ratio the [`Scorer`] reports against
    /// persistence must be exactly 1 at every horizon - in the market-neutral space and in the
    /// absolute space that adds the realized market drift back - and every directional and
    /// correlation statistic must sit at its null. Nonzero per-window σ and a nonzero
    /// `½·ln h` prior are supplied so the σ and `√h` scalings are exercised rather than
    /// bypassed; the two anchored ratios that are *not* 1 (the mid anchor moves the
    /// denominator, and mid ≠ 0) are pinned against their closed form.
    #[test]
    fn zero_forecast_scores_exactly_persistence_at_every_horizon() {
        let _rng = crate::torch::test_rng::exclusive();
        let (w, h) = (90usize, 6usize);
        let per_group = 30usize;
        let mut forecast = synthetic_forecast(w as i64, h as i64);
        forecast.scaled = forecast.scaled.zeros_like();
        forecast.prices = (&forecast.scaled * 0.01).exp() * 100.;
        forecast.rebased_prices = ((&forecast.scaled + &forecast.drift) * 0.01).exp() * 100.;
        let ids: Vec<i64> = (0..w).map(|i| (i / per_group) as i64).collect();
        let half_log_horizon = ((Tensor::arange(h as i64, (Kind::Float, Device::Cpu)) + 1.).log()
            * 0.5)
            .reshape([1, 1, h as i64, 1]);
        let mut scorer = Scorer::new(
            &half_log_horizon,
            &vec![1.0; h],
            &Tensor::from_slice(&ids),
            w / per_group,
            h as i64,
            Device::Cpu,
        );
        let mask = host(&forecast.mask);
        for (start, rows) in [(0usize, 40usize), (40, 25), (65, 25)] {
            let bars = mask[start * h..(start + rows) * h].iter().sum::<f64>() as usize;
            scorer.accumulate(&slice(&forecast, start as i64, rows as i64), bars);
        }
        let evaluation = scorer.finish(EvalTiming::default()).unwrap();
        let curve = &evaluation.trading;
        let horizon = &evaluation.horizon;
        assert_eq!(evaluation.mse, evaluation.persistence_mse);
        assert_eq!(evaluation.absolute_mse, evaluation.absolute_persistence_mse);
        assert_eq!(evaluation.median_window_ratio, 1.);
        for j in 0..h {
            // Both spaces, per horizon, exactly - these are the same sums divided by the same
            // counts, so anything but bit equality means the forecast reached the denominator.
            assert_eq!(horizon.mse[j], horizon.persistence_mse[j], "horizon {j}");
            assert_eq!(
                horizon.absolute_mse[j], horizon.absolute_persistence_mse[j],
                "horizon {j}"
            );
            assert_eq!(horizon.trimmed_mse_ratio[j], 1., "horizon {j}");
            assert_eq!(horizon.mae_ratio[j], 1., "horizon {j}");
            assert_eq!(horizon.win_rate[j], 0., "horizon {j}");
            assert_eq!(horizon.up_fraction[j], 0., "horizon {j}");
            assert!(horizon.hit_rate[j].is_nan(), "horizon {j}");
            assert_eq!(curve.close_mse_ratio[j], 1., "horizon {j}");
            assert_eq!(curve.total_gain[j], 0., "horizon {j}");
            assert_eq!(curve.all_channel_gain[j], 0., "horizon {j}");
            assert_eq!(curve.offset_gain[j], 0., "horizon {j}");
            assert_eq!(curve.demeaned_gain[j], 0., "horizon {j}");
            assert_eq!(curve.scaling_gain[j], 0., "horizon {j}");
            assert_eq!(curve.mean_forecast[j], 0., "horizon {j}");
            // A flat forecast is no directional claim in the space it is flat in, and no
            // ranking, so those rates and correlations are undefined rather than 0.5 or 0. The
            // mid anchor is the exception and is checked against its reference below: a zero
            // close forecast is still a claim about the move away from the mid.
            assert!(curve.close_hit_rate[j].is_nan(), "horizon {j}");
            assert!(curve.pearson[j].is_nan(), "horizon {j}");
            assert!(curve.cross_sectional_ic[j].is_nan(), "horizon {j}");
            // The standard error of a mean over ZERO contributing cross-sections is not 0: a
            // zero-width band around an unmeasured IC is the strongest possible claim on the
            // weakest possible evidence. The population count is what says why both are gaps.
            assert!(curve.cross_sectional_ic_se[j].is_nan(), "horizon {j}");
            assert_eq!(curve.cross_sectional_ic_moments[j], 0., "horizon {j}");
            // A flat forecast has no amplitude to calibrate, so its MSE-optimal gain is
            // undefined - 0 would read as "infinitely over-amplified" - while the ratio an
            // amplitude fix would reach is exactly persistence, because there is nothing to
            // rescale.
            assert!(curve.optimal_gain[j].is_nan(), "horizon {j}");
            assert_eq!(curve.best_scale_mse_ratio[j], 1., "horizon {j}");
            assert_eq!(curve.top_decile_return[j], 0., "horizon {j}");
            assert_eq!(curve.bottom_decile_return[j], 0., "horizon {j}");
            assert_eq!(curve.conviction_spread_return[j], 0., "horizon {j}");
            // The delayed move of a flat forecast is flat, so delayed persistence is matched
            // exactly too; h = 1 has no delayed move at all.
            if j == 0 {
                assert!(curve.delayed_mse_ratio[j].is_nan(), "horizon {j}");
            } else {
                assert_eq!(curve.delayed_mse_ratio[j], 1., "horizon {j}");
            }
        }
        // The mid anchor is the one place a zero close forecast is still a claim: the ratio must
        // NOT be 1 (only the denominator moves) and the hit rate is the rate at which leaning
        // from the trade close toward the mid is the right side.
        let t = host(&forecast.targets);
        let mid = host(&forecast.mid);
        let c = CHANNELS as usize;
        for j in 0..h {
            let (mut model, mut anchored) = (0., 0.);
            let (mut hit, mut bet) = (0., 0.);
            for i in 0..w {
                let m = mask[i * h + j];
                let y = t[(i * h + j) * c + c - 1] * m;
                model += (y * m).powi(2);
                let (anchor_target, anchor_forecast) = ((y - mid[i]) * m, -mid[i] * m);
                anchored += anchor_target.powi(2);
                if anchor_forecast != 0. && anchor_target != 0. {
                    bet += 1.;
                    hit += f64::from(anchor_forecast.signum() * anchor_target.signum() > 0.);
                }
            }
            close(&[curve.mid_anchor_mse_ratio[j]], &[model / anchored]);
            assert!(curve.mid_anchor_mse_ratio[j] != 1.);
            assert!(bet > 0., "the mid anchor reference must take sides");
            close(&[curve.mid_anchor_hit_rate[j]], &[hit / bet]);
        }
    }
    /// Full-validity forecasts with zero drift and one basis point per sigma unit.
    fn portfolio_fixture(
        windows: i64,
        pred_len: i64,
        forecast_of: impl Fn(&Tensor) -> Tensor,
    ) -> FinalOrigin {
        let cpu = (Kind::Float, Device::Cpu);
        let shape = [windows, pred_len, CHANNELS];
        let targets = Tensor::randn(shape, cpu);
        let scaled = forecast_of(&targets);
        FinalOrigin {
            rebased_prices: (&scaled * 1e-4).exp() * 100.,
            prices: (&scaled * 1e-4).exp() * 100.,
            target_prices: (&targets * 1e-4).exp() * 100.,
            log_scale: Tensor::zeros(shape, cpu),
            mid: Tensor::zeros([windows], cpu),
            sigma: Tensor::full([windows], 1e-4, cpu),
            mask: Tensor::ones([windows, pred_len], cpu),
            drift: Tensor::zeros([windows, pred_len, 1], cpu),
            scaled,
            targets,
        }
    }
    /// One accumulate over the whole fixture, with `windows / per_group` evaluation timestamps
    /// of `per_group` tickers each.
    fn score_fixture(forecast: &FinalOrigin, windows: i64, pred_len: i64, per_group: i64) -> Evaluation {
        let ids: Vec<i64> = (0..windows).map(|i| i / per_group).collect();
        let half_log_horizon = ((Tensor::arange(pred_len, (Kind::Float, Device::Cpu)) + 1.).log()
            * 0.5)
            .reshape([1, 1, pred_len, 1]);
        let mut scorer = Scorer::new(
            &half_log_horizon,
            &vec![1.0; pred_len as usize],
            &Tensor::from_slice(&ids),
            (windows / per_group) as usize,
            pred_len,
            Device::Cpu,
        );
        scorer.accumulate(forecast, (windows * pred_len) as usize);
        scorer.finish(EvalTiming::default()).unwrap()
    }
    /// Payoffs restore raw market drift and enter at the next observed OPEN, while decisions
    /// consume the close predictive scale. Missing entry/endpoint data drops whole cohorts.
    #[test]
    fn scorer_utility_uses_raw_next_open_payoffs_and_close_uncertainty() {
        let cpu = (Kind::Float, Device::Cpu);
        let (windows, pred_len, per_group) = (60i64, 2i64, 20i64);
        let shape = [windows, pred_len, CHANNELS];
        let targets = Tensor::full(shape, 0.25, cpu);
        let _ = targets.select(-1, CHANNELS - 1).fill_(0.75);
        let scaled = Tensor::full(shape, 2., cpu);
        let _ = scaled.select(-1, CHANNELS - 1).fill_(0.5);
        let log_scale = Tensor::full(shape, 2., cpu);
        let _ = log_scale.select(-1, CHANNELS - 1).fill_(0.25_f64.ln());
        let drift = Tensor::from_slice(&[0.5_f32, 1.])
            .reshape([1, pred_len, 1])
            .expand([windows, pred_len, 1], true);
        let sigma = Tensor::from_slice(
            &(0..windows)
                .map(|row| 0.01_f32 * (row / per_group + 1) as f32)
                .collect::<Vec<_>>(),
        );
        let mask = Tensor::ones([windows, pred_len], cpu);
        let _ = mask.get(per_group).get(1).zero_();
        let _ = mask.get(2 * per_group).get(0).zero_();
        let fixture = FinalOrigin {
            prices: (&scaled * sigma.reshape([-1, 1, 1])).exp() * 100.,
            rebased_prices: ((&scaled + &drift) * sigma.reshape([-1, 1, 1])).exp() * 100.,
            target_prices: ((&targets + &drift) * sigma.reshape([-1, 1, 1])).exp() * 100.,
            mid: Tensor::zeros([windows], cpu),
            scaled,
            log_scale,
            targets,
            drift,
            sigma,
            mask,
        };
        let ids: Vec<i64> = (0..windows).map(|row| row / per_group).collect();
        let mut scorer = Scorer::new(
            &Tensor::zeros([pred_len], cpu),
            &[1., 1.],
            &Tensor::from_slice(&ids),
            3,
            pred_len,
            Device::Cpu,
        );
        for (start, rows) in [(0, 13), (13, 28), (41, 19)] {
            let batch = slice(&fixture, start, rows);
            let bars = batch.mask.sum(Kind::Int64).int64_value(&[]) as usize;
            scorer.accumulate(&batch, bars);
        }
        let portfolio = scorer.portfolio().unwrap();
        assert_eq!(portfolio.horizons, vec![1, 2]);
        assert_eq!(portfolio.cross_sections, vec![2, 1]);
        let expected = [
            (0.005_f64.exp_m1() + 0.01_f64.exp_m1()) * 0.5 * 1e4,
            0.01_f64.exp_m1() * 1e4,
        ];
        // The positive means clear only the CLOSE predictive scale.
        for label in ["equal long", "one-sigma gated sign"] {
            let policy = portfolio.policies.iter().find(|policy| policy.label == label).unwrap();
            close(&policy.gross_bps, &expected);
            close(&policy.gross_exposure, &[1., 1.]);
        }
    }
    /// The `held-out cross-section` draw takes WHOLE timestamps at ONE uniform width, and
    /// negotiates that width against the corpus instead of demanding [`CROSS_SECTION_TICKERS`]
    /// and aborting. The fixture is the shape of the real corpus: a dense aligned core, a
    /// block one ticker short of the ideal width, and a one-ticker timestamp - which is what
    /// every timestamp of the strided held-out sample looks like, and why that sample can
    /// never carry this family.
    #[test]
    fn the_cross_section_draw_takes_whole_timestamps_at_one_negotiated_width() {
        let block = |stamp: i64, tickers: usize, offset: usize| -> Vec<(i64, WindowRef)> {
            (0..tickers)
                .map(|t| {
                    (
                        stamp,
                        WindowRef {
                            ticker: offset + t,
                            origin: stamp as usize,
                        },
                    )
                })
                .collect()
        };
        let mut stamped = block(300, CROSS_SECTION_TICKERS + 40, 0);
        stamped.extend(block(100, CROSS_SECTION_TICKERS - 1, 1000));
        stamped.extend(block(200, 1, 2000));
        stamped.extend(block(400, CROSS_SECTION_TICKERS, 3000));
        let drawn = cross_section_blocks(&stamped).unwrap();
        let mut per_stamp: BTreeMap<usize, usize> = BTreeMap::new();
        for reference in &drawn {
            *per_stamp.entry(reference.origin).or_default() += 1;
        }
        // Three timestamps at 255 carries 3·254 = 762, against 2·255 = 510 for the two that
        // could have run at the full 256: the draw gives up one ticker of width to buy a whole
        // extra cross-section. The one-ticker timestamp is below the floor and is gone.
        let width = CROSS_SECTION_TICKERS - 1;
        assert_eq!(
            per_stamp.into_iter().collect::<Vec<_>>(),
            vec![(100, width), (300, width), (400, width)],
            "the draw must be uniform across the timestamps it keeps, and must keep the \
             one-short block by narrowing rather than dropping it"
        );
        assert_eq!(drawn.len(), 3 * width);
        // Fixed across calls, which is what makes the series step-matchable.
        assert_eq!(drawn, cross_section_blocks(&stamped).unwrap());
        // The real corpus: a fully aligned liquid core plus a long tail of one- and two-ticker
        // timestamps, 10.6 tickers per timestamp on average. This is the case a fixed-width
        // requirement would have aborted three queued runs on, so it is the case that must
        // yield a draw.
        let mut scattered = Vec::new();
        for stamp in 0..90i64 {
            scattered.extend(block(1_000_000 + stamp, 300, 0));
        }
        for stamp in 0..20_000i64 {
            scattered.extend(block(stamp, 1 + (stamp % 2) as usize, 10_000));
        }
        let core = cross_section_blocks(&scattered).unwrap();
        assert_eq!(
            core.len(),
            CROSS_SECTION_TIMESTAMPS.min(90) * CROSS_SECTION_TICKERS,
            "the aligned core must be drawn at full width and the scattered tail ignored"
        );
        assert!(
            core.iter().all(|reference| reference.ticker < 300),
            "no member of a below-floor timestamp may enter the draw"
        );
        // A corpus whose timestamps are ALL below the floor is refused loudly rather than
        // scored into a family of NaNs nobody reads.
        let thin = block(100, CROSS_SECTION_FLOOR - 1, 0);
        assert!(cross_section_blocks(&thin)
            .unwrap_err()
            .to_string()
            .contains("cross-sectional trading measurement needs"));
    }
    /// The bug that hid this family's blankness for four runs: a holding period no timestamp
    /// cleared divided by `clamp_min(1.)` and reported a clean 0.0 basis points, which reads
    /// as "earned nothing" and is indistinguishable from a real flat spread. It must read as
    /// NaN in every return quantity, and the census must say how many timestamps there were.
    #[test]
    fn an_unpopulated_cross_section_reads_as_nan_and_never_as_zero_bps() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(5);
        // Ten tickers per timestamp, half of CROSS_SECTION_MIN, so no timestamp is scored.
        let (windows, pred_len, per_group) = (200i64, 8i64, 10i64);
        let fixture = portfolio_fixture(windows, pred_len, Tensor::shallow_clone);
        let portfolio = score_fixture(&fixture, windows, pred_len, per_group).portfolio;
        for index in 0..portfolio.horizons.len() {
            let held = portfolio.horizons[index];
            assert_eq!(
                portfolio.cross_sections[index], 0,
                "h={held} must contribute no timestamp"
            );
            for policy in &portfolio.policies {
                for value in [
                    policy.gross_bps[index],
                    policy.gross_rate_bps[index],
                    policy.breakeven_bps[index],
                    policy.net_bps[0][index],
                    policy.net_rate_bps[0][index],
                    policy.payoff_std_bps[index],
                    policy.worst_bps[index],
                ] {
                    assert!(value.is_nan(), "{} at h={held}: {value}", policy.label);
                }
            }
            assert_eq!(portfolio.tickers_per_timestamp[index], per_group as f64);
            assert_eq!(portfolio.narrowest_timestamp[index], per_group as f64);
        }
        // And the same fixture with a scorable cross-section does report numbers, so the NaN
        // above is the population and not a broken code path.
        let populated = score_fixture(&fixture, windows, pred_len, 50).portfolio;
        for index in 0..populated.horizons.len() {
            assert!(populated.cross_sections[index] > 0);
            assert!(populated.policies.iter().all(|policy| policy.gross_bps[index].is_finite()));
            assert_eq!(populated.tickers_per_timestamp[index], 50.);
            assert_eq!(populated.narrowest_timestamp[index], 50.);
        }
    }
    /// The step loop's stopping policy, driven exactly as [`train`] drives it: every step
    /// increments, [`StopRules::evaluates`] decides whether the interval evaluates and
    /// reports, [`StopRules::termination`] decides whether the run is over. Nothing here
    /// re-implements the policy - both decisions are the functions the loop itself calls - so
    /// a cap that stops a step early or late fails here.
    fn drive(rules: StopRules, steps_per_epoch: usize) -> (Vec<usize>, Option<Termination>) {
        let mut reported = Vec::new();
        for epoch in 1..=rules.planned_epochs {
            for index in 0..steps_per_epoch {
                let step = (epoch - 1) * steps_per_epoch + index + 1;
                let epoch_complete = index + 1 == steps_per_epoch;
                if !rules.evaluates(step, epoch_complete) {
                    continue;
                }
                reported.push(step);
                if let Some(reason) = rules.termination(Progress {
                    step,
                    epoch,
                    epoch_complete,
                    stale_previews: 0,
                    stale_epochs: 0,
                }) {
                    return (reported, Some(reason));
                }
            }
        }
        (reported, None)
    }
    #[test]
    fn a_step_cap_stops_at_exactly_its_step_and_reports_that_interval() {
        let rules = StopRules {
            max_steps: Some(4000),
            eval_every: 1000,
            preview_patience: 3,
            patience: 3,
            planned_epochs: 1,
        };
        let (reported, reason) = drive(rules, 9590);
        assert_eq!(reported, vec![1000, 2000, 3000, 4000]);
        assert_eq!(reason, Some(Termination::StepCap));
        // A cap no report interval lands on still evaluates AT the cap and stops there: the
        // interval a capped arm was submitted to measure is never the one that goes missing.
        let (reported, reason) = drive(
            StopRules {
                max_steps: Some(2500),
                ..rules
            },
            9590,
        );
        assert_eq!(reported, vec![1000, 2000, 2500]);
        assert_eq!(reason, Some(Termination::StepCap));
        // Uncapped, the same driver runs the whole epoch and exits on the epoch limit, so the
        // cap is the only thing that shortened the run above.
        let uncapped = StopRules {
            max_steps: None,
            ..rules
        };
        let (reported, reason) = drive(uncapped, 9590);
        assert_eq!(reported.last(), Some(&9590));
        assert_eq!(reported.len(), 10);
        assert_eq!(reason, Some(Termination::EpochLimit));
        // A cap past the run's end never binds.
        let (reported, reason) = drive(
            StopRules {
                max_steps: Some(20000),
                ..rules
            },
            9590,
        );
        assert_eq!(reported.last(), Some(&9590));
        assert_eq!(reason, Some(Termination::EpochLimit));
    }
    /// The property that makes a capped arm comparable to the 9,590-step baselines at all: at
    /// one `--schedule-budget`, the first N steps of a capped run are the first N steps of the
    /// long run's OWN trajectory, bit for bit. If the cap ever leaked into the schedule, every
    /// short arm would be a different experiment from the arms it is being read against and
    /// the comparison would be silently wrong rather than loudly broken.
    #[test]
    fn a_step_cap_leaves_the_learning_rate_schedule_untouched() {
        let steps_per_epoch = 9590;
        let long = TrainArgs {
            schedule_budget: 9590,
            ..TrainArgs::default()
        };
        let capped = TrainArgs {
            max_steps: 4000,
            ..long.clone()
        };
        assert_eq!(
            schedule_budget(&capped, steps_per_epoch),
            schedule_budget(&long, steps_per_epoch)
        );
        let build = |args: &TrainArgs| {
            LrSchedule::new(
                schedule_budget(args, steps_per_epoch),
                NANOGPT_COOLDOWN_FRAC,
                NANOGPT_COOLDOWN_FLOOR,
            )
            .unwrap()
        };
        let (long_schedule, capped_schedule) = (build(&long), build(&capped));
        assert_eq!(long_schedule, capped_schedule);
        for step in 0..4000 {
            assert_eq!(
                capped_schedule.scale(step).to_bits(),
                long_schedule.scale(step).to_bits(),
                "the capped arm's rate multiplier at step {step} is not the long arm's"
            );
            assert_eq!(
                capped_schedule.muon_momentum(step).to_bits(),
                long_schedule.muon_momentum(step).to_bits(),
                "the capped arm's NorMuon momentum at step {step} is not the long arm's"
            );
        }
        // The cooldown is still shaped against the 9,590-step budget: the base rate holds
        // through step 3,835 and the capped arm spends only its last 164 steps in the very
        // start of the warmdown, ending at 0.976 of the base rate. A cap that leaked into the
        // schedule would have cooled from step 1,600 and ended at the 0.15 floor, i.e. a
        // different optimizer trajectory from the baselines at every step past 1,600.
        assert_eq!(capped_schedule.cooldown_start(), Some(3836));
        assert_eq!(capped_schedule.scale(3835).to_bits(), 1.0f64.to_bits());
        let last = capped_schedule.scale(3999);
        assert!(last < 1. && last > 0.97, "{last}");
        let leaked = LrSchedule::new(4000, NANOGPT_COOLDOWN_FRAC, NANOGPT_COOLDOWN_FLOOR).unwrap();
        assert_eq!(leaked.cooldown_start(), Some(1600));
        assert!(leaked.scale(3999) < 0.16);
        // `--schedule-budget 0` still means "the whole planned run", and a cap does not shrink
        // that either: this is the second arm shape, 4,000 steps of a schedule shaped against
        // the full epoch without having to restate the epoch's length.
        assert_eq!(
            schedule_budget(
                &TrainArgs {
                    max_steps: 4000,
                    ..TrainArgs::default()
                },
                steps_per_epoch
            ),
            9590
        );
    }
    #[test]
    fn every_exit_path_names_its_own_termination_reason() {
        let rules = StopRules {
            max_steps: Some(4000),
            eval_every: 1000,
            preview_patience: 3,
            patience: 3,
            planned_epochs: 2,
        };
        let uncapped = StopRules {
            max_steps: None,
            ..rules
        };
        let mid = |step, stale_previews| Progress {
            step,
            epoch: 1,
            epoch_complete: false,
            stale_previews,
            stale_epochs: 0,
        };
        let complete = |epoch, step, stale_epochs| Progress {
            step,
            epoch,
            epoch_complete: true,
            stale_previews: 0,
            stale_epochs,
        };
        assert_eq!(rules.termination(mid(3000, 0)), None);
        assert_eq!(rules.termination(mid(4000, 0)), Some(Termination::StepCap));
        assert_eq!(
            rules.termination(mid(3000, 3)),
            Some(Termination::PreviewPatience)
        );
        // Both criteria on one evaluation: the model's own verdict outranks the operator's
        // budget, because "the held-out NLL stopped improving" is the stronger statement about
        // the arm and a capped label would hide it.
        assert_eq!(
            rules.termination(mid(4000, 3)),
            Some(Termination::PreviewPatience)
        );
        // An uncapped run can never report a cap, whatever step it is at.
        assert_eq!(uncapped.termination(mid(20000, 0)), None);
        // A completed epoch with epochs left and no patience exhausted keeps training.
        assert_eq!(uncapped.termination(complete(1, 9590, 0)), None);
        assert_eq!(
            uncapped.termination(complete(1, 9590, 3)),
            Some(Termination::EpochPatience)
        );
        assert_eq!(
            uncapped.termination(complete(2, 19180, 0)),
            Some(Termination::EpochLimit)
        );
        // The cap outranks both epoch exits, so an arm whose cap happens to land on the last
        // step of its last epoch is still reported as capped rather than as a finished run.
        assert_eq!(
            StopRules {
                max_steps: Some(19180),
                ..rules
            }
            .termination(complete(2, 19180, 0)),
            Some(Termination::StepCap)
        );
    }
    /// The cap is refused by `train` BEFORE `Corpus::load`, which is the only reason this test
    /// can call `train` at all: the data directory does not exist, so any check that ran after
    /// the corpus load would report a missing-data error instead of the cap's.
    #[test]
    fn an_invalid_step_cap_is_refused_before_the_corpus_loads() {
        let args = |max_steps| TrainArgs {
            max_steps,
            data_dir: PathBuf::from("/nonexistent-universe-so-a-corpus-load-fails-loudly"),
            ..TrainArgs::default()
        };
        for cap in 1..=CAPTURE_AFTER_STEPS {
            let error = train(args(cap)).unwrap_err().to_string();
            assert!(
                error.contains("--max-steps") && error.contains("capture"),
                "a cap of {cap} must be refused by name, got {error}"
            );
        }
        args(0).validate().unwrap();
        args(CAPTURE_AFTER_STEPS + 1).validate().unwrap();
    }
    /// The self-check the whole amplitude mechanism rests on, asserted on the real reduction
    /// rather than on the algebra: applying a positive per-horizon gain to the anchored close
    /// leaves every rank and sign statistic where it was, and moves only the quadratic ones.
    ///
    /// The calibrated forecast is built exactly as [`CausalPatchModel::fold_mean_gain`] builds
    /// it - every channel shifted by `(gain - 1)·close`, so the close is scaled and the candle
    /// offsets are untouched - which is what makes this a test of the deployed transform and
    /// not of a convenient paraphrase of it.
    #[test]
    fn a_positive_per_horizon_gain_moves_mse_and_cannot_move_a_rank_statistic() {
        let _rng = crate::torch::test_rng::exclusive();
        let (w, h) = (120i64, 12i64);
        let per_group = 30i64;
        let forecast = synthetic_forecast(w, h);
        // A curve with the measured shape, including one horizon shrunk to a hundredth: the
        // invariance may not depend on the gain being mild.
        let curve: Vec<f32> = (1..=h)
            .map(|bar| match bar {
                1 => 0.6,
                4 => 0.99,
                12 => 0.01,
                bar => 1.0 / (1.0 + 0.1 * bar as f32),
            })
            .collect();
        let gain = Tensor::from_slice(&curve).reshape([1, h, 1]);
        let shift = forecast.scaled.narrow(-1, CHANNELS - 1, 1) * (&gain - 1.);
        let scaled = &forecast.scaled + &shift;
        let calibrated = FinalOrigin {
            prices: (&scaled * 0.01).exp() * 100.,
            rebased_prices: ((&scaled + &forecast.drift) * 0.01).exp() * 100.,
            scaled,
            log_scale: forecast.log_scale.shallow_clone(),
            targets: forecast.targets.shallow_clone(),
            mask: forecast.mask.shallow_clone(),
            drift: forecast.drift.shallow_clone(),
            target_prices: forecast.target_prices.shallow_clone(),
            mid: forecast.mid.shallow_clone(),
            sigma: forecast.sigma.shallow_clone(),
        };
        let curves = |forecast: &FinalOrigin| -> TradingCurve {
            let half_log_horizon = ((Tensor::arange(h, (Kind::Float, Device::Cpu)) + 1.).log()
                * 0.5)
                .reshape([1, 1, h, 1]);
            let ids: Vec<i64> = (0..w).map(|row| row / per_group).collect();
            let mut scorer = Scorer::new(
                &half_log_horizon,
                &vec![1.0; h as usize],
                &Tensor::from_slice(&ids),
                (w / per_group) as usize,
                h,
                Device::Cpu,
            );
            let bars = forecast.mask.sum(Kind::Float).double_value(&[]) as usize;
            scorer.accumulate(forecast, bars);
            scorer.finish(EvalTiming::default()).unwrap().trading
        };
        let (before, after) = (curves(&forecast), curves(&calibrated));
        for bar in 0..h as usize {
            let invariant = [
                ("within-timestamp IC", before.cross_sectional_ic[bar], after.cross_sectional_ic[bar]),
                ("pooled Pearson", before.pearson[bar], after.pearson[bar]),
                ("pooled Spearman", before.spearman[bar], after.spearman[bar]),
                ("close hit rate", before.close_hit_rate[bar], after.close_hit_rate[bar]),
                ("top-decile hit rate", before.top_decile_hit_rate[bar], after.top_decile_hit_rate[bar]),
                ("top-decile return", before.top_decile_return[bar], after.top_decile_return[bar]),
                ("conviction spread", before.conviction_spread_return[bar], after.conviction_spread_return[bar]),
                ("optimal gain", before.optimal_gain[bar] / f64::from(curve[bar]), after.optimal_gain[bar]),
                // `D = Cov²/(Var·mean(y²))` is scale-free; the best-scale ratio it feeds is
                // NOT, because the offset term carries `μ` and a gain moves `μ` to `g·μ`.
                ("demeaned gain", before.demeaned_gain[bar], after.demeaned_gain[bar]),
            ];
            for (quantity, was, now) in invariant {
                assert!(
                    (was.is_nan() && now.is_nan()) || (was - now).abs() <= IC_INVARIANCE_TOLERANCE,
                    "the {quantity} at h={} moved from {was} to {now} under a gain of {}",
                    bar + 1,
                    curve[bar]
                );
            }
            // And the quadratic statistics DO move, or the gain was not applied at all.
            assert!(
                (before.close_mse_ratio[bar] - after.close_mse_ratio[bar]).abs() > 1e-6,
                "the close MSE ratio at h={} did not move",
                bar + 1
            );
            // The mid-anchored and delayed diagnostics are deliberately NOT on the invariant
            // list: they compare the forecast against a nonzero reference (`f - mid`) or
            // against another horizon's forecast (`f_h - f_1`), so they are affine in the
            // forecast rather than rank statistics of it, and a per-horizon gain moves them
            // whenever a bar's forecast crosses that reference. Measured here: the mid-anchor
            // hit rate at h = 1 moves from 0.53125 to 0.55208 under a gain of 0.6. A
            // calibrated arm's mid-anchor curves are therefore not step-matchable against an
            // uncalibrated arm's, which is why nothing asserts them either way.
        }
    }
}

#[derive(Clone, Debug, Args)]
pub struct ProbeArgs {
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long,default_value_os_t=crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    /// Where the three latent-probe report bases are written.
    #[arg(long)]
    pub output: PathBuf,
    /// No split knob, for the same reason [`CalibrateArgs`] has none: the fit block is the
    /// corpus's reserved `[70%, 80%)` partition and the scored block is the held-out full
    /// split, and any knob here would be a knob that can produce leakage.
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,
    /// Cap on the number of FIT origins, deterministically strided over the whole reserved
    /// partition; `0` uses all of it.
    ///
    /// Only the fit side is capped, and the asymmetry is the point. Subsampling the fit block
    /// costs coefficient precision, which the penalty-selection holdout then measures and the
    /// ridge absorbs. Subsampling the SCORED block costs the standard error of the very
    /// difference this whole subcommand exists to measure, so there is no knob for it.
    #[arg(long, default_value_t = 0)]
    pub fit_origins: usize,
    /// Also measure the EXTRACTION SCALING LAW: seven nested fit sets spanning a 64x range of
    /// fit rows, drawn by whole timestamp so every cross-section survives entire, each scored
    /// on the identical held-out block, then `IC(N) = IC_inf - a·N^-b`.
    ///
    /// It answers the question the paired gap cannot: whether a probe that merely matches the
    /// head is at an INFORMATION ceiling or at a SAMPLE-SIZE one. A curve still climbing at the
    /// largest rung says the probe is fit-limited and the World B reading is premature; a flat
    /// one says the latent has been read out as well as this fit population allows. Costs one
    /// extra device-resident moment block per rung and no extra forward pass.
    #[arg(long, default_value_t = false)]
    pub scaling: bool,
}

/// The trunk latent at the origin's token position, the head's own close forecast from the SAME
/// forward, and the close target with its validity mask - all narrowed to the probed horizons
/// and all fp64.
///
/// `backbone` and `head` are called separately rather than through [`final_origin`]'s composed
/// `forward` because the tensor between them is exactly what is being probed: `backbone` ends
/// with the final `rms_norm` and, under `last_only`, narrows to token position `origins - 1`,
/// and `head` consumes that tensor as its `state` argument. Probing it post-norm is not a
/// detail - RMSNorm's gain is per-token, so a pre-norm probe would be fitted on an object no
/// single linear map relates to what the head reads, and a win could be attributed to the row
/// scaling instead of to the representation.
///
/// The head also concatenates a 256-wide projection of the known-future covariates onto that
/// 512-wide state. The probe is NOT given that block, so its input is a strict subset of the
/// head's and a probe win cannot be explained by extra inputs.
fn probe_origin(
    model: &CausalPatchModel,
    batch: &Batch,
    horizon_index: &Tensor,
) -> (Tensor, Tensor, Tensor, Tensor) {
    let stats = model.statistics(batch);
    let state = model.backbone(batch, &stats, false, true);
    let head = model.head(batch, &state, true);
    let output = model.output(&head);
    let last = stats.last();
    let close = CHANNELS - 1;
    let channel_last = |tensor: &Tensor| tensor.squeeze_dim(1).transpose(1, 2).contiguous();
    let forecast = channel_last(&model.decode(&output, &last)).select(-1, close);
    let (targets, mask) = model.targets(batch, &last, true);
    let target = channel_last(&targets).select(-1, close);
    let probed = |tensor: Tensor| {
        tensor
            .index_select(1, horizon_index)
            .to_kind(Kind::Double)
            .contiguous()
    };
    (
        state.squeeze_dim(1).to_kind(Kind::Double).contiguous(),
        probed(forecast),
        probed(target),
        probed(mask.reshape([-1, model.config().pred_len])),
    )
}

/// One grad-free final-origin pass, handing each batch's latent, head forecast, target and mask
/// to `sink` along with the running row offset so a caller can index a per-origin side table.
///
/// The same prefetch-one-batch-ahead structure [`score`] uses, because the probe's two passes
/// are the same shape of work and a second loader idiom here would drift from that one.
fn probe_pass(
    corpus: &Arc<Corpus>,
    model: &CausalPatchModel,
    origins: &[WindowRef],
    batch_size: usize,
    device: Device,
    horizon_index: &Tensor,
    sink: &mut dyn FnMut(i64, &Tensor, &Tensor, &Tensor, &Tensor),
) -> Result<()> {
    ensure!(
        batch_size > 0 && !origins.is_empty(),
        "a probe pass needs a positive batch size and a nonempty origin list"
    );
    let _guard = tch::no_grad_guard();
    let loader = Prefetcher::new(Arc::clone(corpus));
    loader.request(&origins[..origins.len().min(batch_size)])?;
    let mut resident: Option<Batch> = None;
    let mut filled = 0i64;
    for index in 0..origins.len().div_ceil(batch_size) {
        let (host, _) = loader.receive()?;
        let next = (index + 1) * batch_size;
        if next < origins.len() {
            loader.request(&origins[next..(next + batch_size).min(origins.len())])?;
        }
        if resident
            .as_ref()
            .is_none_or(|batch: &Batch| batch.rows() != host.rows())
        {
            resident = Some(host.resident(device));
        }
        let device_batch = resident.as_mut().expect("just allocated");
        host.upload(device_batch)?;
        let (latent, forecast, target, mask) = probe_origin(model, device_batch, horizon_index);
        let rows = latent.size()[0];
        sink(filled, &latent, &forecast, &target, &mask);
        filled += rows;
    }
    ensure!(
        filled == origins.len() as i64,
        "the probe pass covered {filled} of {} origins",
        origins.len()
    );
    Ok(())
}

/// Frozen-trunk latent probe: measure how much long-horizon information the trunk latent
/// already carries that the trained head fails to extract.
///
/// Three passes, and the order is the whole guarantee. The coefficient-fit block and the
/// penalty-selection block are both inside the corpus's reserved `[70%, 80%)` partition, which
/// checkpoint selection never read; the scored block is the held-out full split. Nothing
/// measured on the scored block reaches any fit, and [`probe::Partitions::split`] proves that
/// on realized timestamps rather than on the boundary arithmetic that produced them.
///
/// See [`super::probe`] for why a linear probe on the post-norm origin latent is the decisive
/// instrument, what the two worlds are, and where the pre-registered thresholds come from.
pub fn probe(args: ProbeArgs) -> Result<()> {
    ensure!(args.batch_size > 0, "batch size must be positive");
    let (manifest, corpus, _store, model, device) =
        load_checkpoint(&args.checkpoint, &args.data_dir)?;
    let pred_len = corpus.contract.pred_len;
    let horizons: Vec<usize> = reports::DECISION_HORIZONS
        .iter()
        .copied()
        .filter(|horizon| *horizon <= pred_len)
        .collect();
    ensure!(
        !horizons.is_empty(),
        "a {pred_len}-bar forecast reaches none of the decision horizons"
    );
    let dated = |refs: &[WindowRef]| -> Vec<probe::DatedRef> {
        refs.iter()
            .map(|reference| {
                let ticker = corpus.ticker(*reference);
                (
                    *reference,
                    ticker.timestamp(reference.origin),
                    ticker.timestamp(reference.origin + pred_len),
                )
            })
            .collect()
    };
    // The cap is applied BEFORE the split, so the purge is measured on the origins actually
    // used rather than on a superset of them.
    let fit_refs = match args.fit_origins {
        0 => corpus.calibration_refs.clone(),
        count => fixed_origins(&corpus.calibration_refs, count.min(corpus.calibration_refs.len()))?,
    };
    let partitions = probe::Partitions::split(&dated(&fit_refs), &dated(&corpus.validation_refs))?;
    println!(
        "CausalPatch latent probe: {} coefficient-fit origins and {} penalty-selection origins from the reserved calibration partition ({} purged between them, gap {} ms), scored on {} {}-origin held-out full rows over [{}, {}]; {} scored origins dropped to place the whole scored block behind the fit block's last target bar, after which every fit target completes {} ms before the first scored origin",
        partitions.inner.len(),
        partitions.holdout.len(),
        partitions.purged,
        partitions.inner_blocks.purge_gap_ms,
        partitions.scored.len(),
        pred_len,
        partitions.outer.evaluation_first_origin_ms,
        partitions.outer.evaluation_last_origin_ms,
        partitions.outer_purged,
        partitions.outer.purge_gap_ms
    );
    let width = manifest.model.d_model;
    let horizon_index = Tensor::from_slice(
        &horizons.iter().map(|h| *h as i64 - 1).collect::<Vec<i64>>(),
    )
    .to_device(device);
    let count = horizons.len() as i64;
    let accumulate = |refs: &[WindowRef]| -> Result<probe::HostMoments> {
        let mut moments = probe::Moments::new(width, count, device);
        probe_pass(
            &corpus,
            &model,
            refs,
            args.batch_size,
            device,
            &horizon_index,
            &mut |_, latent, _, target, mask| moments.push(latent, target, mask),
        )?;
        moments.to_host()
    };
    let started = Instant::now();
    let holdout = accumulate(&partitions.holdout)?;
    // ONE pass over the inner fit block fills three decompositions at once, because a second
    // forward over 200k origins is ~109 s of a 600 s lease and buys nothing: the whole block for
    // the headline probes, `RECENCY_TRANCHES` equal-population chronological tranches for the
    // recency slope, and - under `--scaling` - the nested ladder's disjoint shells. Every row is
    // routed to exactly one accumulator per decomposition by row-index selection, so each
    // decomposition costs one pass worth of fp64 Gram work rather than one pass per block.
    let inner_stamps = corpus.origin_timestamps(&partitions.inner);
    let tranche_of = probe::chronological_tranches(&inner_stamps, probe::RECENCY_TRANCHES)?;
    let shell_of = if args.scaling {
        let rungs = teacher::whole_timestamp_ladder(&inner_stamps, probe::SCALING_LEVELS);
        Some((probe::scaling_shells(&rungs)?, rungs))
    } else {
        None
    };
    let mut whole = probe::Moments::new(width, count, device);
    let mut tranches: Vec<probe::Moments> = (0..probe::RECENCY_TRANCHES)
        .map(|_| probe::Moments::new(width, count, device))
        .collect();
    let mut shells: Vec<probe::Moments> = shell_of
        .as_ref()
        .map(|(_, rungs)| {
            (0..rungs.len()).map(|_| probe::Moments::new(width, count, device)).collect()
        })
        .unwrap_or_default();
    probe_pass(
        &corpus,
        &model,
        &partitions.inner,
        args.batch_size,
        device,
        &horizon_index,
        &mut |filled, latent, _, target, mask| {
            whole.push(latent, target, mask);
            let rows = latent.size()[0] as usize;
            let offset = filled as usize;
            let mut route = |owners: &[usize], blocks: &mut Vec<probe::Moments>| {
                for (slot, moments) in blocks.iter_mut().enumerate() {
                    let picked: Vec<i64> = (0..rows)
                        .filter(|row| owners[offset + row] == slot)
                        .map(|row| row as i64)
                        .collect();
                    if picked.is_empty() {
                        continue;
                    }
                    let index = Tensor::from_slice(&picked).to_device(device);
                    moments.push(
                        &latent.index_select(0, &index),
                        &target.index_select(0, &index),
                        &mask.index_select(0, &index),
                    );
                }
            };
            route(&tranche_of, &mut tranches);
            if let Some((owners, _)) = shell_of.as_ref() {
                route(owners, &mut shells);
            }
        },
    )?;
    let inner = whole.to_host()?;
    // The shells were filled by the pass above; all that is left is the prefix merge that turns
    // seven DISJOINT blocks into seven NESTED fit sets, which sufficient-statistic additivity
    // makes free.
    let ladder = match shell_of.as_ref() {
        None => None,
        Some((owners, rungs)) => {
            let mut cumulative: Option<probe::HostMoments> = None;
            let mut ladder_moments = Vec::with_capacity(shells.len());
            let mut counts = Vec::with_capacity(shells.len());
            for (shell, moments) in shells.iter().enumerate() {
                let host = moments.to_host()?;
                cumulative = Some(match cumulative {
                    None => host,
                    Some(seen) => seen.merge(&host)?,
                });
                let mut distinct: Vec<i64> = inner_stamps
                    .iter()
                    .zip(&rungs[shell])
                    .filter(|(_, kept)| **kept)
                    .map(|(stamp, _)| *stamp)
                    .collect();
                distinct.sort_unstable();
                distinct.dedup();
                counts.push((owners.iter().filter(|owner| **owner <= shell).count(), distinct.len()));
                ladder_moments.push(cumulative.as_ref().expect("just assigned").clone());
            }
            Some((ladder_moments, counts))
        }
    };
    let fits = probe::fit_all(&inner, &holdout, &horizons)?;
    println!(
        "CausalPatch latent probe fitted {} closed-form probes over {} horizons in {:.1} s",
        fits.len(),
        horizons.len(),
        started.elapsed().as_secs_f64()
    );
    let bank = probe::ProbeBank::new(&fits, &horizons, width, device)?;
    // One ridge per rung at h = 1 only, each selecting its penalty on the SAME fixed holdout
    // block, so `IC(N)` varies with the number of coefficient-fit rows and with nothing else.
    let rung_fits = ladder
        .as_ref()
        .map(|(moments, counts)| -> Result<_> {
            let mut weights = Vec::with_capacity(moments.len());
            let mut intercepts = Vec::with_capacity(moments.len());
            let mut ridges = Vec::with_capacity(moments.len());
            for block in moments {
                let fit = probe::fit_all(block, &holdout, &[1])?
                    .into_iter()
                    .find(|fit| fit.class == probe::ProbeClass::Ridge)
                    .context("the ridge class is always fitted")?;
                weights.push(fit.weight.shallow_clone());
                intercepts.push(fit.intercept);
                ridges.push(fit.ridge);
            }
            let weight = Tensor::stack(&weights, 1).to_kind(Kind::Double).to_device(device);
            let intercept = Tensor::from_slice(&intercepts).to_device(device).reshape([1, -1]);
            Ok((weight, intercept, ridges, counts.clone()))
        })
        .transpose()?;
    // One ridge per chronological tranche, at every probed horizon, each selecting its penalty
    // on the SAME fixed holdout block. Equal population per tranche, so what varies between them
    // is the fit block's AGE and not its size - which is the control the extraction ladder does
    // not provide and the recency discount is read from.
    let mut tranche_weights = Vec::with_capacity(tranches.len());
    let mut tranche_intercepts = Vec::with_capacity(tranches.len());
    let mut tranche_spans = Vec::with_capacity(tranches.len());
    for (slot, moments) in tranches.iter().enumerate() {
        let host = moments.to_host()?;
        for fit in probe::fit_all(&host, &holdout, &horizons)? {
            if fit.class == probe::ProbeClass::Ridge {
                tranche_weights.push(fit.weight.shallow_clone());
                tranche_intercepts.push(fit.intercept);
            }
        }
        let picked: Vec<i64> = tranche_of
            .iter()
            .zip(&inner_stamps)
            .filter(|(owner, _)| **owner == slot)
            .map(|(_, stamp)| *stamp)
            .collect();
        ensure!(!picked.is_empty(), "chronological tranche {slot} holds no fit origins");
        let mean = picked.iter().map(|stamp| *stamp as f64).sum::<f64>() / picked.len() as f64;
        tranche_spans.push((picked.len(), mean as i64));
    }
    // Column order is tranche-major then horizon, matching what `PairedScorer` assumes.
    let tranche_weight =
        Tensor::stack(&tranche_weights, 1).to_kind(Kind::Double).to_device(device);
    let tranche_intercept =
        Tensor::from_slice(&tranche_intercepts).to_device(device).reshape([1, -1]);
    // Dense timestamp ranks over the scored origins, identical to [`score`]'s: the probe's IC
    // has to be the same statistic on the same grouping as the head's, or the paired difference
    // is a difference of two conventions.
    let stamps = corpus.origin_timestamps(&partitions.scored);
    let mut distinct = stamps.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let ranked: Vec<i64> = stamps
        .iter()
        .map(|stamp| {
            distinct
                .binary_search(stamp)
                .expect("every scored origin timestamp is one of the distinct timestamps") as i64
        })
        .collect();
    let groups = Tensor::from_slice(&ranked).to_device(device);
    let classes = probe::ProbeClass::all();
    let mut scorer = probe::PairedScorer::new(
        count,
        1 + classes.len() as i64,
        distinct.len() as i64,
        device,
    );
    let mut rung_scorer = rung_fits.as_ref().map(|(_, _, ridges, _)| {
        probe::PairedScorer::new(1, ridges.len() as i64, distinct.len() as i64, device)
    });
    let mut tranche_scorer = probe::PairedScorer::new(
        count,
        tranche_spans.len() as i64,
        distinct.len() as i64,
        device,
    );
    probe_pass(
        &corpus,
        &model,
        &partitions.scored,
        args.batch_size,
        device,
        &horizon_index,
        &mut |filled, latent, forecast, target, mask| {
            let rows = latent.size()[0];
            let block = groups.narrow(0, filled, rows);
            // The head's own forecast first, then the probes, all from ONE forward over these
            // exact rows. That is what makes every reported difference paired.
            let all = Tensor::cat(&[forecast.shallow_clone(), bank.forecast(latent)], 1);
            scorer.push(&block, &all, target, mask);
            // The ladder scores on the same rows in the same pass, at h = 1 only, so its curve
            // and the headline gap are statistics of one draw rather than of two.
            if let (Some(rung_scorer), Some((weight, intercept, _, _))) =
                (rung_scorer.as_mut(), rung_fits.as_ref())
            {
                rung_scorer.push(
                    &block,
                    &(latent.matmul(weight) + intercept),
                    &target.narrow(1, 0, 1),
                    &mask.narrow(1, 0, 1),
                );
            }
            // Same rows again for the tranches, so the recency slope and the headline gap are
            // statistics of one draw. Every probe in this subcommand is scored in this one pass.
            tranche_scorer.push(
                &block,
                &(latent.matmul(&tranche_weight) + &tranche_intercept),
                target,
                mask,
            );
        },
    )?;
    let labels: Vec<String> = std::iter::once("trained head".to_owned())
        .chain(classes.iter().map(|class| class.label(width as usize)))
        .collect();
    let mut report = scorer.finish(&labels, &horizons)?;
    report.attach(&fits, width as usize);
    // The head's own age deficit, measured off the corpus rather than assumed from the nominal
    // boundaries: the last bar any TRAINING target reads. Per-ticker ordinal boundaries make the
    // nominal 70% a different instant on every ticker, so only the realized maximum is a
    // universe-wide statement.
    let head_last_target_ms = corpus
        .train_refs
        .iter()
        .map(|reference| corpus.ticker(*reference).timestamp(reference.origin + pred_len))
        .max()
        .context("the training population is empty")?;
    let tranche_labels: Vec<String> = tranche_spans
        .iter()
        .map(|(origins, mid)| format!("ridge fitted on {origins} origins centred at {mid} ms"))
        .collect();
    let tranche_report = tranche_scorer.finish(&tranche_labels, &horizons)?;
    let tranche_ic: Vec<Vec<f64>> = horizons
        .iter()
        .enumerate()
        .map(|(index, _)| {
            tranche_report
                .forecasters
                .iter()
                .map(|forecaster| forecaster.per_horizon[index].ic)
                .collect()
        })
        .collect();
    let recency =
        probe::Recency::fit(tranche_spans.clone(), &horizons, tranche_ic, head_last_target_ms)?;
    let world = probe::verdict(&report, Some(&recency));
    let context = reports::LatentProbeContext {
        fit_origins: partitions.inner.len(),
        penalty_origins: partitions.holdout.len(),
        purged_origins: partitions.purged,
        scored_origins: partitions.scored.len(),
        outer_purged: partitions.outer_purged,
        purge_gap_ms: partitions.outer.purge_gap_ms,
        first_scored_ms: partitions.outer.evaluation_first_origin_ms,
        last_scored_ms: partitions.outer.evaluation_last_origin_ms,
        width,
        verdict: world.label().to_owned(),
    };
    reports::write_latent_probe(&args.output, manifest.epoch, manifest.step, &report, &context)?;
    reports::write_latent_probe_ratio(
        &args.output,
        manifest.epoch,
        manifest.step,
        &report,
        &context,
    )?;
    reports::write_latent_probe_conditioning(
        &args.output,
        manifest.epoch,
        manifest.step,
        &report,
        &context,
    )?;
    reports::write_latent_probe_recency(
        &args.output,
        manifest.epoch,
        manifest.step,
        &recency,
        &context,
    )?;
    for (index, horizon) in recency.horizons.iter().enumerate() {
        println!(
            "CausalPatch recency discount | h={horizon:3}: tranche ICs {:?} over {:.3} years of fit-block age, slope {:+.5} IC/year, head deficit {:.3} years, World A bar {:.5} -> {:.5}",
            recency.tranche_ic[index].iter().map(|ic| format!("{ic:+.5}")).collect::<Vec<_>>(),
            (recency.tranches.last().map_or(0, |(_, mid)| *mid)
                - recency.tranches.first().map_or(0, |(_, mid)| *mid)) as f64
                / (365.25 * 86_400_000.),
            recency.slope_per_year[index],
            recency.age_gap_years,
            probe::WORLD_A_GAIN,
            probe::WORLD_A_GAIN + recency.discount[index].max(0.)
        );
    }
    if let (Some(rung_scorer), Some((_, _, ridges, counts))) = (rung_scorer, rung_fits.as_ref()) {
        let labels: Vec<String> =
            counts.iter().map(|(origins, _)| format!("ridge on {origins} fit origins")).collect();
        let ladder_report = rung_scorer.finish(&labels, &[1])?;
        let rungs: Vec<probe::ScalingRung> = ladder_report
            .forecasters
            .iter()
            .zip(counts)
            .zip(ridges)
            .map(|((forecaster, (origins, timestamps)), ridge)| probe::ScalingRung {
                origins: *origins,
                timestamps: *timestamps,
                ridge: *ridge,
                ic: forecaster.per_horizon[0].ic,
                ic_se: forecaster.per_horizon[0].ic_se,
            })
            .collect();
        let curve = probe::fit_scaling(&rungs)?;
        for rung in &curve.rungs {
            println!(
                "CausalPatch extraction ladder | {} fit origins over {} timestamps: h=1 IC {:+.5} +- {:.5} at ridge {:.3e}",
                rung.origins, rung.timestamps, rung.ic, rung.ic_se, rung.ridge
            );
        }
        println!(
            "CausalPatch extraction ladder | IC(N) = {:+.5} - {:.5}·N^-{:.4}, residual {:.5} IC; largest rung reaches {:.1}% of the asymptote",
            curve.asymptote,
            curve.amplitude,
            curve.exponent,
            curve.residual,
            100. * curve.rungs.last().map_or(f64::NAN, |rung| rung.ic) / curve.asymptote
        );
        reports::write_latent_probe_scaling(
            &args.output,
            manifest.epoch,
            manifest.step,
            &curve,
            &context,
        )?;
    }
    for forecaster in &report.forecasters {
        println!("CausalPatch latent probe | {}", forecaster.label);
        for score in &forecaster.per_horizon {
            println!(
                "  h={:3}: IC {:+.5} +- {:.5} over {} cross-sections | paired gap vs head {:+.5} +- {:.5} over {} | close ratio {:.6}, oracle {:.6}, gain {:.4} | ridge {:.3e}, condition {:.3e} (unpenalized {:.3e}), fit-block holdout correlation {:+.5}",
                score.horizon,
                score.ic,
                score.ic_se,
                score.cross_sections,
                score.gap,
                score.gap_se,
                score.paired_cross_sections,
                score.mse_ratio,
                score.oracle_mse_ratio,
                score.optimal_gain,
                score.ridge,
                score.condition,
                score.raw_condition,
                score.holdout_correlation
            );
        }
    }
    println!(
        "CausalPatch latent probe VERDICT: {} | pre-registered rule: a paired IC gain of at least {:+.3} at any horizon from h={} is World A and justifies a latent objective; every long horizon inside +-{:.3} is World B and retires the line",
        world.label(),
        probe::WORLD_A_GAIN,
        probe::VERDICT_HORIZON_FLOOR,
        probe::WORLD_B_BAND
    );
    Ok(())
}

#[derive(Clone, Debug, Args)]
pub struct CeilingArgs {
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long,default_value_os_t=crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    /// Where `timexer_segment_information_ceiling` is written.
    #[arg(long)]
    pub output: PathBuf,
    /// No split knob and no origin cap. The ceiling is a functional of the held-out full
    /// split's own targets - there is nothing to fit and therefore no fit block to keep
    /// separate - and the paired gap's standard error is the entire deliverable, so
    /// subsampling the scored population would cost precisely the number this measures.
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,
    /// Which cross-section placement the ceiling AND the student IC are read on. Job 5483
    /// measured both moving by 2.07-2.72x between placements at a fixed checkpoint, so a gap
    /// assembled from two placements measures the draw rather than the model; `both` scores
    /// one checkpoint twice in one process so the comparison is a placement comparison.
    #[arg(long, value_enum, default_value_t = Placement::Strided)]
    pub placement: Placement,
    /// `c₁`, the assumed UPPER bound on one-bar predictability, as an IC. See
    /// [`teacher::DEFAULT_ONE_BAR_CEILING_IC`] for why it is an input, why the default is
    /// twice the measured ANCHORED `h = 1` IC, and why the `h = 1` gap is therefore
    /// tautological while the h ≥ 8 gaps are not.
    #[arg(long, default_value_t = teacher::DEFAULT_ONE_BAR_CEILING_IC)]
    pub one_bar_ceiling_ic: f64,
    /// Batch sizes to time, IN ORDER, over one population inside ONE process; empty measures
    /// the ceiling instead and is the default.
    ///
    /// A curve assembled from separately leased runs is not admissible on this machine. The
    /// identical command over the identical corpus measured 337.5 s (job 5529) and 292.6 s
    /// (job 5542), a 15.4% swing, and the CPU-side anchored placement inside those same two
    /// runs measured 20,524.9 ms and 15,654.9 ms; a knee smaller than that cannot be
    /// distinguished from a foreign tenant. So the points share one CUDA context, one corpus
    /// residency and one machine state, and the order is the caller's so it can be written
    /// A-B-A: the two A's differ by the drift that occurred DURING the sweep, which is the
    /// curve's own error bar and is printed as one.
    ///
    /// Every point also re-measures the same ceiling curve, and the sweep asserts the printed
    /// statistics are BIT-IDENTICAL across batch sizes. Batch size is a reduction order: the
    /// nine fp64 per-timestamp banks receive the same rows in different `index_add_`
    /// groupings. That equality is therefore not a formality, it is the proof that the
    /// throughput knob does not move the measurement.
    #[arg(long, value_delimiter = ',')]
    pub throughput_sweep: Vec<usize>,
    /// Origins per sweep point, subsampled by WHOLE TIMESTAMP out of `held-out full`; `0`
    /// takes the whole split. Only the wall clock depends on it - every point sees the same
    /// draw - and it exists so an A-B-A sweep fits inside one ten-minute lease.
    #[arg(long, default_value_t = 0)]
    pub throughput_origins: usize,
}

/// Measure the per-horizon predictable-information ceiling on the held-out full split, beside
/// the checkpoint's own within-timestamp IC on the identical retained origins.
///
/// One forward pass. The ceiling itself needs no model at all - it is a second moment of the
/// targets and [`teacher`] documents exactly what it does and does not bound - and the
/// checkpoint is loaded so the GAP is a paired difference over ONE population rather than two
/// numbers read off two passes over two draws.
pub fn ceiling(args: CeilingArgs) -> Result<()> {
    ensure!(args.batch_size > 0, "batch size must be positive");
    let (manifest, corpus, _store, model, device) =
        load_checkpoint(&args.checkpoint, &args.data_dir)?;
    let mut corpus = corpus;
    let anchor = |corpus: &mut Arc<Corpus>| -> Result<()> {
        Arc::get_mut(corpus)
            .context("the corpus is shared, so its placement cannot be replaced in place")?
            .anchor_cross_section_draws(CROSS_SECTION_FLOOR)
    };
    if !args.throughput_sweep.is_empty() {
        ensure!(
            args.placement != Placement::Both,
            "--throughput-sweep times ONE population inside one process; `both` would re-place \
             the corpus between points and make the two A's differ by a placement instead of \
             by drift"
        );
        if args.placement == Placement::Anchored {
            anchor(&mut corpus)?;
        }
        return throughput_sweep(&args, &corpus, &model, device);
    }
    let passes: Vec<(bool, PathBuf)> = match args.placement {
        Placement::Strided => vec![(false, args.output.clone())],
        Placement::Anchored => vec![(true, args.output.clone())],
        Placement::Both => vec![
            (false, args.output.clone()),
            (true, args.output.join(corpus::ANCHORED_PLACEMENT)),
        ],
    };
    for (anchored, output) in passes {
        if anchored {
            anchor(&mut corpus)?;
        }
        println!(
            "CausalPatch ceiling placement {} -> {}",
            if anchored { "ANCHORED" } else { "STRIDED" },
            output.display()
        );
        ceiling_placement(&args, &manifest, &corpus, &model, device, &output)?;
    }
    Ok(())
}

/// One ceiling measurement over whatever placement `corpus` currently carries. The ceiling and
/// the student IC are accumulated in the same pass over one population precisely so the gap is
/// a paired difference; that pairing is void across a placement boundary, which is why the
/// placement is a parameter of this function rather than of the reader's interpretation.
fn ceiling_placement(
    args: &CeilingArgs,
    manifest: &Manifest,
    corpus: &Arc<Corpus>,
    model: &CausalPatchModel,
    device: Device,
    output: &Path,
) -> Result<()> {
    let started = Instant::now();
    let _guard = tch::no_grad_guard();
    // Three populations, and the order is the argument. The variance-ratio curve exists to be
    // FROZEN into training as a data-derived replacement for the decoder's √h shape gain, and
    // a constant fitted on held-out data and baked into a training run leaks however mildly -
    // so the only clean source is `training`. The other two are not the fit, they are the
    // evidence that licenses the fit: a scale constant may be frozen when the three agree, and
    // may not when they do not, whatever any single one of them measures.
    //
    // The training draw is subsampled by WHOLE TIMESTAMP and never by a strided pick over the
    // ticker-major reference list. A strided pick lands its origins on that many different
    // timestamps with one ticker each, which gives every cross-sectional statistic a
    // one-name cross-section and reads as a blank panel rather than as an error.
    ensure!(
        !corpus.validation_refs.is_empty(),
        "the held-out full split is empty"
    );
    let scored = corpus.validation_refs.len();
    let drawing = Instant::now();
    let draws: Vec<(&str, Vec<WindowRef>)> = vec![
        (
            "training",
            whole_timestamp_draw(corpus, &corpus.train_refs, scored),
        ),
        ("calibration block", corpus.calibration_refs.clone()),
        ("held-out full", corpus.validation_refs.clone()),
    ];
    // Printed because it was invisible and large: job 5544 spent 364.3 s in this function of
    // which only 272.8 s was the scored loop, and every millisecond of the difference is host
    // work with the card idle. A number nobody prints is a number nobody fixes.
    println!(
        "CausalPatch ceiling draws built in {:.1} s (GPU idle): {} training of {} candidates, {} calibration, {} held-out full",
        drawing.elapsed().as_secs_f64(),
        draws[0].1.len(),
        corpus.train_refs.len(),
        draws[1].1.len(),
        draws[2].1.len()
    );
    benchmark::cuda_memory(true)?;
    let sampler = HardwareSampler::start()?;
    let mut timing = PassTiming::default();
    let mut curves = Vec::with_capacity(draws.len());
    for (split, origins) in &draws {
        ensure!(!origins.is_empty(), "the {split} population is empty");
        curves.push(ceiling_pass(
            corpus,
            model,
            device,
            origins,
            args.batch_size,
            args.one_bar_ceiling_ic,
            &mut timing,
        )?);
    }
    let populations: Vec<(&str, &teacher::CeilingCurve)> = draws
        .iter()
        .map(|(split, _)| *split)
        .zip(curves.iter())
        .collect();
    let curve = curves
        .last()
        .expect("three populations were just accumulated");
    let (allocated, _) = benchmark::cuda_memory(false)?;
    // Read BEFORE the peak probe below, which allocates 2.5 GiB of its own and would otherwise
    // become the "peak allocator" the batch-size curve is read on.
    let hardware = sampler.finish()?;
    println!(
        "CausalPatch predictable-information ceiling at step {} over {} of {} {} origins in {:.1} s of which {:.1} s scored three populations (peak allocator {:.0} MiB), one-bar ceiling {:.4} IC",
        manifest.step,
        curve.retained_origins,
        curve.scored_origins,
        "held-out full",
        started.elapsed().as_secs_f64(),
        timing.loop_ms / 1000.,
        allocated as f64 / 1048576.,
        curve.one_bar_ceiling_ic
    );
    let _ = report_pass_throughput(output, model, device, args.batch_size, &timing, &hardware)?;
    let mse = curve.best_mse_ratio();
    for &horizon in reports::DECISION_HORIZONS {
        let Some(index) = horizon
            .checked_sub(1)
            .filter(|index| *index < curve.ceiling.len())
        else {
            continue;
        };
        println!(
            "  h {horizon:3}: ceiling {:.5} (wide {:.5} over {:.0} cross-sections, paired over {:.0}) | student {:.5} ± {:.5} | gap {:.5} ± {:.5} | variance ratio {:.5} | best reachable market-neutral MSE ratio {:.5} | {}",
            curve.ceiling[index],
            curve.wide_ceiling[index],
            curve.wide_cross_sections[index],
            curve.paired_cross_sections[index],
            curve.student_ic[index],
            curve.student_ic_error[index],
            curve.gap[index],
            curve.gap_error[index],
            1. / curve.martingale_ratio[index],
            mse[index],
            curve.verdict(index)
        );
        // The (A2) sensitivity, stated as a verdict rather than left to the reader. The
        // MEASURED h=1 within-timestamp IC is a LOWER bound on the true one-bar ceiling, so a
        // break-even below it means the gap survives every admissible `c₁` and the conclusion
        // stops depending on the assumption at all.
        let breakeven = curve.breakeven_one_bar_ic[index];
        println!(
            "          break-even one-bar ceiling {breakeven:.5} IC: the gap at this horizon {} the assumed {:.4} - it {} for every one-bar ceiling at or above the measured h 1 IC of {:.4}",
            if breakeven < curve.one_bar_ceiling_ic {
                "does not depend on"
            } else {
                "is created by"
            },
            curve.one_bar_ceiling_ic,
            if breakeven < teacher::MEASURED_ONE_BAR_IC {
                "HOLDS"
            } else {
                "FAILS"
            },
            teacher::MEASURED_ONE_BAR_IC
        );
    }
    // The whole curve is one monotone function of `c₁`, so the sensitivity that would
    // otherwise cost four more passes is arithmetic on the variance ratio already measured.
    // Printed rather than charted: it is the same unit as the curve but a different question,
    // and drawing four hypothetical ceilings beside the one this run states would make the
    // panel unreadable about which of them is the result.
    // Bracketing the ANCHORED measurement: zero, the measured h=1 IC itself (the tightest
    // admissible c₁, since a realized forecast cannot beat its own bound), the shipped default
    // at twice it, and a fourth point above. The old sweep bracketed 0.0833, a strided-draw
    // number the same run now measures at 0.16208.
    for candidate in [0., teacher::MEASURED_ONE_BAR_IC, teacher::DEFAULT_ONE_BAR_CEILING_IC, 0.5] {
        let leak = 1. - candidate * candidate;
        let at = |horizon: usize| {
            curve
                .martingale_ratio
                .get(horizon - 1)
                .map_or(f64::NAN, |ratio| (1. - leak * ratio).max(0.).sqrt())
        };
        println!(
            "  one-bar ceiling {candidate:.4} IC implies h 1 {:.5}, h 64 {:.5}, h 192 {:.5}",
            at(1),
            at(64),
            at(192)
        );
    }
    // Two different NaNs, counted separately, because they are two different facts and a
    // single "unidentified" tally would let a broken assumption hide behind a thin draw.
    let census = |kind: &str| -> Vec<usize> {
        (0..curve.ceiling.len())
            .filter(|index| curve.ceiling[*index].is_nan() && curve.verdict(*index) == kind)
            .map(|index| index + 1)
            .collect()
    };
    for kind in [
        "in-window reversal, assumption (A1) broken, nothing stated",
        "never measured, no qualifying cross-section",
    ] {
        let horizons = census(kind);
        if !horizons.is_empty() {
            println!(
                "CausalPatch ceiling: {} of {} horizons read NaN because of \"{kind}\" (first {:?}); largest assumption violation {:.5} IC",
                horizons.len(),
                curve.ceiling.len(),
                &horizons[..horizons.len().min(8)],
                curve
                    .reversal_violation
                    .iter()
                    .copied()
                    .filter(|value| value.is_finite())
                    .fold(0.0f64, f64::max)
            );
        }
    }
    // The variance-ratio readings, per population, at the decision horizons - because whether
    // the three AGREE is the licence to freeze the curve into training, and a chart nobody
    // opens is a worse place for that verdict than the log the run already prints.
    for (split, population) in &populations {
        let (level, onset) = match (population.plateau_ratio, population.plateau_horizon) {
            (Some(level), Some(onset)) => (level, onset),
            _ => (f64::NAN, 0),
        };
        println!(
            "  {split}: fitted variance-ratio plateau {level:.5} from h {onset} over {} origins, implying the √h shape gain over-states the target scale by {:.3}x and a lag-1 autocorrelation of {:.5} under MA(1) against a MEASURED {:.5}",
            population.retained_origins,
            level.sqrt().recip(),
            (level - 1.) / 2.,
            population.lag_autocorrelation[population.lag_autocorrelation.len() - 1]
        );
        for &horizon in reports::DECISION_HORIZONS {
            let Some(index) = horizon
                .checked_sub(1)
                .filter(|index| *index < population.variance_ratio.len())
            else {
                continue;
            };
            println!(
                "    h {horizon:3}: variance over summed per-bar variance {:.5} | variance over the √h assumption {:.5} | measured lag-1 autocorrelation {:.5}",
                population.variance_ratio[index],
                population.walk_ratio[index],
                population.lag_autocorrelation[index]
            );
        }
    }
    fs::create_dir_all(output)?;
    teacher::write_information_ceiling(
        output,
        manifest.epoch,
        manifest.step,
        &curve,
        corpus.contract.tickers.len(),
    )?;
    teacher::write_variance_ratio(output, manifest.epoch, manifest.step, &populations)?;
    teacher::write_return_autocorrelation(output, manifest.epoch, manifest.step, &populations)?;
    Ok(())
}

/// Where a non-training accumulation pass's wall clock went, phase by phase.
///
/// The four phase spans are measured on SAMPLED batches only, and on a sampled batch each one
/// is closed by a `Cuda::synchronize`, so the four partition that batch's wall clock exactly
/// and none of them is an estimate. Sampling is what makes that affordable: draining the
/// launch pipeline four times per batch on all 4,483 batches would be a large part of what it
/// measures, and [`score`] already refuses that for the same reason with a single sampled
/// batch. A pass of thousands of batches can afford better than one sample, so this takes
/// [`PASS_TIMING_SAMPLES`] of them and the reader gets a mean with a visible spread rather
/// than one draw.
///
/// `loop_ms` is the unsynchronized truth: `loop_ms / batches` is the real period, and the
/// sampled phases are what it is made of. `build_ms` is the loader THREAD's own assembly time
/// and is deliberately NOT part of the partition - it overlaps the device, and the only
/// question it answers is whether one batch of prefetch depth is enough, which is
/// `build_ms / batches` against that period.
#[derive(Debug, Default)]
pub(super) struct PassTiming {
    batches: u64,
    rows: u64,
    /// Per sampled batch: the batch ordinal, the four phases that partition it, and the device
    /// backlog drained before the sample, which belongs to no phase.
    trace: Vec<[f64; 6]>,
    wait_ms: f64,
    build_ms: f64,
    loop_ms: f64,
    finish_ms: f64,
    /// Host-side preparation before a pass's first batch: the per-origin timestamp lookups,
    /// the sort and dedup that make dense ranks, and the nine device banks. Zero GPU work, and
    /// measured because it was 91.5 s of job 5544's 364.3 s ceiling placement while being
    /// invisible to every number that run printed.
    setup_ms: f64,
}

/// Synchronized phase samples per population. 64 of 1,367 batches is 4.7% of the batches
/// carrying an instrument that costs each of them five pipeline drains.
const PASS_TIMING_SAMPLES: usize = 64;

impl PassTiming {
    /// Mean milliseconds per SAMPLED batch of each phase, then of the drained backlog.
    fn phase_means(&self) -> [f64; 5] {
        let mut means = [0.; 5];
        if self.trace.is_empty() {
            return means;
        }
        for row in &self.trace {
            for (mean, value) in means.iter_mut().zip(&row[1..]) {
                *mean += value / self.trace.len() as f64;
            }
        }
        means
    }
    fn period_ms(&self) -> f64 {
        self.loop_ms / self.batches.max(1) as f64
    }
}

/// The throughput half of a non-training pass: where its batch went, and how hard the card
/// was working while it went there.
///
/// Printed AND charted, because the two readers are different: the panel is how a successor
/// sees drift across a pass, and the line is what a lease-budget decision is made on before
/// anyone opens a panel.
///
/// The arithmetic figure is an UPPER bound and is labelled as one on the line itself.
/// `step_cost` charges a TRAINING step - a dense head over all 375 origins, and a backward
/// charged at twice the forward - so its forward third over-states a forward-only `last_only`
/// pass by the head's share of forward matrix work, which `timexer_flop_inventory.md` puts at
/// 9.0% (0.5096 of 5.6615 TFLOP at 256 rows) less the 1/375 of it that `last_only` keeps. An
/// over-stated numerator makes the achieved fraction of the card's own measured GEMM peak an
/// over-estimate, which is the safe direction for a claim of the form "this pass is already
/// near the hardware and the only win left is batch size".
fn report_pass_throughput(
    output: &Path,
    model: &CausalPatchModel,
    device: Device,
    batch_size: usize,
    timing: &PassTiming,
    hardware: &[[f64; 5]],
) -> Result<PassThroughput> {
    let [wait, upload, forward, accumulate, backlog] = timing.phase_means();
    let seconds = timing.loop_ms / 1000.;
    // Linear in rows, so charging the measured row count rather than `batches * batch_size`
    // keeps the three partial last batches from inflating the numerator.
    let per_row = model.config().step_cost(batch_size).matmul_flops / 3. / batch_size as f64;
    let (peak_tflops, _) = benchmark::device_peaks(device)?;
    let achieved = per_row * timing.rows as f64 / seconds / 1e12;
    let power = || hardware.iter().map(|row| row[4]);
    let mean_power = power().sum::<f64>() / hardware.len().max(1) as f64;
    println!(
        "CausalPatch pass throughput at {batch_size} rows: {:.1} ms per batch over {} batches and {} rows, {:.0} origins/s | sampled phases (mean of {}, each synchronized, the device drained first): loader wait {wait:.2} + upload {upload:.2} + forward {forward:.2} + accumulate {accumulate:.2} = {:.2} ms | host run-ahead drained before each sample {backlog:.1} ms = {:.1} batches, so the host is {} | host assembly {:.1} ms per batch on the loader thread, {:.2}x the period | setup before the first batch {:.1} s, finish {:.1} s, both GPU-idle | <= {achieved:.1} achieved forward matmul TFLOPS (dense-head ledger third, an upper bound) of {peak_tflops:.1} measured bf16 GEMM peak = <= {:.0}% | {mean_power:.0} W mean, {:.0} W peak, {:.0} MiB peak board VRAM including foreign tenants",
        timing.period_ms(),
        timing.batches,
        timing.rows,
        timing.rows as f64 / seconds,
        timing.trace.len(),
        wait + upload + forward + accumulate,
        backlog / timing.period_ms().max(f64::MIN_POSITIVE),
        if backlog > timing.period_ms() { "AHEAD of the device and cannot be the bottleneck" } else { "keeping pace at best" },
        timing.build_ms / timing.batches.max(1) as f64,
        timing.build_ms / timing.loop_ms.max(f64::MIN_POSITIVE),
        timing.setup_ms / 1000.,
        timing.finish_ms / 1000.,
        100. * achieved / peak_tflops,
        power().fold(0., f64::max),
        hardware.iter().map(|row| row[3]).fold(0., f64::max),
    );
    let point = PassThroughput {
        batch_size,
        period_ms: timing.period_ms(),
        origins_per_second: timing.rows as f64 / seconds,
        achieved_tflops: achieved,
        peak_tflops,
        mean_power,
        peak_power: power().fold(0., f64::max),
        peak_board_mib: hardware.iter().map(|row| row[3]).fold(0., f64::max),
        phases: [wait, upload, forward, accumulate],
        build_ms_per_batch: timing.build_ms / timing.batches.max(1) as f64,
    };
    fs::create_dir_all(output)?;
    let title = format!(
        "CausalPatch non-training pass at {batch_size} rows | where does one batch of a diagnostic go? | {:.1} ms period over {} batches, of which forward {forward:.1}; host assembly {:.1} ms per batch and {backlog:.0} ms of device backlog drained before each sample, so the host runs ahead; {mean_power:.0} W mean of {:.0} W peak",
        timing.period_ms(),
        timing.batches,
        timing.build_ms / timing.batches.max(1) as f64,
        power().fold(0., f64::max),
    );
    benchmark::write_eval_phases(output, &title, &timing.trace)?;
    benchmark::write_hardware(output, &title, hardware)?;
    Ok(point)
}

/// One sweep point, kept so the sweep can print its table after every point has been measured
/// rather than leaving the reader to diff console lines.
#[derive(Clone, Copy, Debug)]
struct PassThroughput {
    batch_size: usize,
    period_ms: f64,
    origins_per_second: f64,
    achieved_tflops: f64,
    peak_tflops: f64,
    mean_power: f64,
    peak_power: f64,
    peak_board_mib: f64,
    phases: [f64; 4],
    build_ms_per_batch: f64,
}

/// Half a unit in the last place any ceiling statistic is ever PRINTED - every console line
/// and every chart title in [`teacher`] carries five decimals - and therefore the bar for
/// "this throughput knob did not move the measurement".
///
/// Not zero, and the difference is a fact about the pipeline rather than a concession. Job
/// 5554 measured the target-only series moving by **2.730e-8** between 256 and 512 rows. That
/// is far above what regrouping ~10^5 fp64 `index_add_` terms can produce (~10^-13), so the
/// nine banks are not the source: the TARGETS themselves are fp32 by design
/// (`model.rs` price normalization and market-neutral construction) and reach the banks
/// through per-row reductions whose CUDA block decomposition is chosen from the tensor's
/// total size, exactly as cuBLAS chooses a GEMM kernel from its operand shape. An fp32
/// quantity near 1 carries ~10^-7 of its own representation error, so 2.7e-8 is one ulp of
/// arithmetic that was always there, surfaced by changing the shape.
///
/// Demanding exact zero across shapes would therefore be demanding shape-invariant reduction
/// kernels from CUDA, which it does not offer. At a REPEATED shape the bar is
/// [`FP64_SCATTER_TOLERANCE`], and for the model series it is exact zero.
const PRINTED_STATISTIC_TOLERANCE: f64 = 5e-6;

/// The reordering noise floor of the nine fp64 banks at a FIXED shape.
///
/// `index_add_` scatters with atomics on CUDA, so the summation order of the ~10^5 rows
/// landing in one timestamp bank is whatever the hardware happened to schedule and is NOT
/// reproducible between two runs of the identical shape. Job 5556 measured the two 256-row
/// points of one sweep differing by **4.441e-16** in the target-only series - two ulps of an
/// fp64 quantity near 1 - while the model series differed by **exactly zero**, which is what
/// separates "the scatter reordered" from "the forward is nondeterministic".
///
/// That measurement is also the strongest available defence of the fp64 choice documented on
/// [`teacher::CeilingAccumulator`]: it is the run-time size of the reordering error those
/// banks exist to keep negligible, and it sits ten orders of magnitude inside the resolution
/// anything is printed at. The bar here is four orders looser than the measurement so that a
/// real regression - a bank that started accumulating in fp32, a lost term - is caught, and
/// atomic scheduling is not.
const FP64_SCATTER_TOLERANCE: f64 = 1e-12;

/// Time one population at several batch sizes inside ONE process, and prove the measurement
/// did not move.
///
/// See [`CeilingArgs::throughput_sweep`] for why the points may not be separate leases. Two
/// things come out of this that separate runs cannot produce: a REPEATED batch size measures
/// the drift that happened during the sweep, which is the only honest error bar on the
/// difference between the other points; and the ceiling curve is re-measured at every point,
/// so a batch size that silently changed the statistic - it changes the `index_add_` grouping
/// of all nine fp64 banks - fails here rather than in a trading verdict six weeks later.
fn throughput_sweep(
    args: &CeilingArgs,
    corpus: &Arc<Corpus>,
    model: &CausalPatchModel,
    device: Device,
) -> Result<()> {
    let _guard = tch::no_grad_guard();
    let origins = match args.throughput_origins {
        0 => corpus.validation_refs.clone(),
        target => whole_timestamp_draw(corpus, &corpus.validation_refs, target),
    };
    ensure!(
        !origins.is_empty(),
        "the held-out full population is empty, so there is nothing to time"
    );
    println!(
        "CausalPatch throughput sweep | {} held-out full origins of {}, {} placement, batch order {:?}",
        origins.len(),
        corpus.validation_refs.len(),
        if args.placement == Placement::Anchored { "ANCHORED" } else { "STRIDED" },
        args.throughput_sweep
    );
    let mut points = Vec::with_capacity(args.throughput_sweep.len());
    let mut reference: Option<(usize, teacher::CeilingCurve)> = None;
    for (index, &rows) in args.throughput_sweep.iter().enumerate() {
        ensure!(rows > 0, "a sweep batch size must be positive");
        benchmark::cuda_memory(true)?;
        let sampler = HardwareSampler::start()?;
        let mut timing = PassTiming::default();
        let curve = ceiling_pass(
            corpus,
            model,
            device,
            &origins,
            rows,
            args.one_bar_ceiling_ic,
            &mut timing,
        )?;
        let (allocated, _) = benchmark::cuda_memory(false)?;
        let hardware = sampler.finish()?;
        let point = report_pass_throughput(
            &args.output.join(format!("sweep-{index}-b{rows}")),
            model,
            device,
            rows,
            &timing,
            &hardware,
        )?;
        println!(
            "  point {index} at {rows} rows: peak allocator {:.0} MiB",
            allocated as f64 / 1048576.
        );
        // Full precision, horizon by horizon, at the horizons every decision is read on. The
        // scalar difference below says whether anything moved; this says whether what moved is
        // visible where it is read, which is the question the acceptance bar actually asks.
        for &horizon in reports::DECISION_HORIZONS {
            let Some(at) = horizon
                .checked_sub(1)
                .filter(|at| *at < curve.ceiling.len())
            else {
                continue;
            };
            println!(
                "    point {index} at {rows} rows, h {horizon:3}: ceiling {:.9} | variance ratio {:.9} | student IC {:.9} | gap {:.9}",
                curve.ceiling[at],
                curve.martingale_ratio[at].recip(),
                curve.student_ic[at],
                curve.gap[at]
            );
        }
        match &reference {
            None => reference = Some((rows, curve)),
            Some((first, expected)) => {
                let (targets, model) = curve.max_statistic_difference(expected);
                // The student error bar this is judged against: the standard error the curve
                // itself reports at h = 1, which is what any reader of the gap already uses.
                let error = expected.student_ic_error.first().copied().unwrap_or(f64::NAN);
                println!(
                    "  point {index} at {rows} rows against the {first}-row curve over all {} horizons: TARGET-ONLY series (ceiling, variance ratio) differ by {targets:.3e} = {:.4} of the {PRINTED_STATISTIC_TOLERANCE:.0e} print resolution - {}; MODEL series (student IC, gap) differ by {model:.3e} = {:.5} of the curve's own h=1 standard error {error:.5}",
                    expected.ceiling.len(),
                    targets / PRINTED_STATISTIC_TOLERANCE,
                    if targets <= PRINTED_STATISTIC_TOLERANCE {
                        "invisible at every printed digit"
                    } else {
                        "VISIBLE in a printed digit, so batch size is a measurement knob"
                    },
                    model / error
                );
                ensure!(
                    targets <= PRINTED_STATISTIC_TOLERANCE,
                    "batch size {rows} moved a TARGET-ONLY statistic against {first} rows by \
                     {targets:.3e}, which is visible at the {PRINTED_STATISTIC_TOLERANCE:.0e} \
                     resolution every report and console line prints. A throughput knob that \
                     changes a printed measurement is a regression, not a speed-up"
                );
                // The repeated batch size is the one comparison where the shape is IDENTICAL,
                // so nothing may SELECT differently and what is left is only the hardware's
                // own scheduling. The forward has none of that and must be exactly
                // reproducible; the nine banks scatter with atomics and cannot be, so they
                // are held to the fp64 reordering floor instead. Two claims, because one
                // band covering both would silently absorb a non-reproducible forward.
                if rows == *first {
                    ensure!(
                        model == 0.,
                        "the repeated {rows}-row point differs from the first by {model:.3e} in \
                         the MODEL series at the SAME shape. Nothing selects differently here, \
                         so this is a nondeterministic forward, and no paired measurement on \
                         this binary can be trusted until it is explained"
                    );
                    ensure!(
                        targets <= FP64_SCATTER_TOLERANCE,
                        "the repeated {rows}-row point differs from the first by {targets:.3e} \
                         in the TARGET-ONLY series at the SAME shape, above the \
                         {FP64_SCATTER_TOLERANCE:.0e} fp64 scatter-reordering floor. Atomic \
                         `index_add_` ordering costs ~1e-16 here; four orders more than that \
                         is a bank that lost precision or lost a term"
                    );
                    println!(
                        "  point {index} REPEATS {rows} rows: forward bit-exact (model {model:.3e}), \
                         fp64 scatter reordering {targets:.3e} = {:.0e} of the \
                         {PRINTED_STATISTIC_TOLERANCE:.0e} print resolution",
                        targets / PRINTED_STATISTIC_TOLERANCE
                    );
                }
            }
        }
        points.push((allocated as f64 / 1048576., point));
    }
    println!(
        "{:>6} {:>10} {:>12} {:>10} {:>9} {:>9} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9}",
        "rows", "ms/batch", "origins/s", "s/1e6 orig", "alloc MiB", "board MiB", "mean W", "peak W",
        "<=TFLOPS", "peak TF", "fwd ms", "build ms"
    );
    for (allocated, point) in &points {
        println!(
            "{:>6} {:>10.1} {:>12.0} {:>10.1} {:>9.0} {:>9.0} {:>8.0} {:>8.0} {:>8.1} {:>8.1} {:>9.2} {:>9.1}",
            point.batch_size,
            point.period_ms,
            point.origins_per_second,
            1e6 / point.origins_per_second,
            allocated,
            point.peak_board_mib,
            point.mean_power,
            point.peak_power,
            point.achieved_tflops,
            point.peak_tflops,
            point.phases[2],
            point.build_ms_per_batch,
        );
    }
    // The error bar, stated as a number rather than left to the reader. Every point after the
    // first repetition is only believable to the extent that it exceeds this.
    let first = points[0].1;
    let repeated = points[1..]
        .iter()
        .rev()
        .map(|(_, point)| *point)
        .find(|point| point.batch_size == first.batch_size);
    match repeated {
        Some(last) => {
            let delta = 100. * (last.origins_per_second - first.origins_per_second) / first.origins_per_second;
            println!(
                "CausalPatch throughput sweep DRIFT at the repeated {} rows: {:.0} -> {:.0} origins/s, {delta:+.2}% | measured bf16 GEMM peak {:.1} -> {:.1} TFLOPS | mean power {:.0} -> {:.0} W. Any difference between the other points smaller than {:.2}% is unresolved at this contention level.",
                first.batch_size,
                first.origins_per_second,
                last.origins_per_second,
                first.peak_tflops,
                last.peak_tflops,
                first.mean_power,
                last.mean_power,
                delta.abs()
            );
        }
        None => println!(
            "CausalPatch throughput sweep has NO repeated batch size, so it carries no error bar and no difference between its points is defensible"
        ),
    }
    Ok(())
}

/// One accumulation pass over one population. Extracted so the three populations are three
/// calls to one body and cannot drift into three conventions.
fn ceiling_pass(
    corpus: &Arc<Corpus>,
    model: &CausalPatchModel,
    device: Device,
    origins: &[WindowRef],
    batch_size: usize,
    one_bar_ceiling_ic: f64,
    timing: &mut PassTiming,
) -> Result<teacher::CeilingCurve> {
    let setting_up = Instant::now();
    // Dense timestamp ranks over the scored origins, identical to [`score`]'s: the student IC
    // this is paired against has to be the same statistic on the same grouping as the one the
    // run's own reports carry, or the difference is a difference of two conventions.
    let stamps = corpus.origin_timestamps(origins);
    let mut distinct = stamps.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let ranked: Vec<i64> = {
        use rayon::prelude::*;
        stamps
            .par_iter()
            .map(|stamp| {
                distinct
                    .binary_search(stamp)
                    .expect("every origin timestamp is one of the distinct timestamps") as i64
            })
            .collect()
    };
    let mut accumulator = teacher::CeilingAccumulator::new(
        &Tensor::from_slice(&ranked),
        distinct.len(),
        corpus.contract.pred_len as i64,
        device,
    );
    let loader = Prefetcher::new(Arc::clone(corpus));
    loader.request(&origins[..origins.len().min(batch_size)])?;
    timing.setup_ms += setting_up.elapsed().as_secs_f64() * 1000.;
    let mut resident: Option<Batch> = None;
    let batches = origins.len().div_ceil(batch_size);
    let stride = batches.div_ceil(PASS_TIMING_SAMPLES);
    let started = Instant::now();
    for index in 0..batches {
        let sampled = index % stride == 0;
        // Drain the device BEFORE the sample, and charge that drain to nothing. Without the
        // per-batch host sync the accumulator used to contain, the host runs arbitrarily far
        // ahead of the device - measured at 183.6 ms, three batches, in job 5544 - so a
        // synchronization anywhere inside the batch charges that whole backlog to whichever
        // phase happened to close first, which is how job 5544 reported a 183.6 ms "upload"
        // inside a 60.8 ms batch. Draining first makes the four spans below span this batch
        // and nothing else; the drained backlog is reported separately, because how far ahead
        // the host runs is the direct evidence that the host is not the bottleneck.
        let backlog_ms = if sampled {
            synchronized(device, Instant::now())
        } else {
            0.
        };
        let phase = |from: Instant| {
            if sampled {
                synchronized(device, from)
            } else {
                0.
            }
        };
        let waiting = Instant::now();
        let (host, build_ms) = loader.receive()?;
        let wait_ms = waiting.elapsed().as_secs_f64() * 1000.;
        timing.wait_ms += wait_ms;
        timing.build_ms += build_ms;
        timing.rows += host.rows() as u64;
        let next = (index + 1) * batch_size;
        if next < origins.len() {
            loader.request(&origins[next..(next + batch_size).min(origins.len())])?;
        }
        if resident
            .as_ref()
            .is_none_or(|batch: &Batch| batch.rows() != host.rows())
        {
            resident = Some(host.resident(device));
        }
        let device_batch = resident.as_mut().expect("just allocated");
        let uploading = Instant::now();
        host.upload(device_batch)?;
        let upload_ms = phase(uploading);
        let forwarding = Instant::now();
        let forecast = final_origin(model, device_batch);
        let forward_ms = phase(forwarding);
        let scoring = Instant::now();
        accumulator.accumulate(&forecast);
        let accumulate_ms = phase(scoring);
        if sampled {
            timing.trace.push([
                (timing.batches + index as u64) as f64,
                wait_ms,
                upload_ms,
                forward_ms,
                accumulate_ms,
                backlog_ms,
            ]);
        }
    }
    timing.loop_ms += synchronized(device, started);
    timing.batches += batches as u64;
    let finishing = Instant::now();
    let curve = accumulator.finish(one_bar_ceiling_ic)?;
    timing.finish_ms += synchronized(device, finishing);
    Ok(curve)
}

/// Subsample a reference list to about `target` origins by dropping whole TIMESTAMPS.
///
/// Corpus plumbing only: which timestamps survive is [`teacher::whole_timestamp_keep`]'s
/// decision, so the property this draw exists for is testable without a corpus and is pinned
/// there rather than described here.
fn whole_timestamp_draw(corpus: &Corpus, refs: &[WindowRef], target: usize) -> Vec<WindowRef> {
    let stamps = corpus.origin_timestamps(refs);
    teacher::whole_timestamp_keep(&stamps, target)
        .into_iter()
        .zip(refs.iter().copied())
        .filter_map(|(keep, reference)| keep.then_some(reference))
        .collect()
}

#[derive(Clone, Debug, Args)]
pub struct BasisStatsArgs {
    /// Any checkpoint of the architecture the arm will train. The whitening scales do not
    /// depend on it at all - they are a second moment of the TARGETS - but `ρ̂` and `β̂` do, and
    /// loading one checkpoint keeps both walks on one corpus and one contract.
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long,default_value_os_t=crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    /// Where the authenticated artifact is written; `--basis-stats` then names this file.
    #[arg(long)]
    pub artifact: PathBuf,
    /// Where `timexer_segment_target_basis` is written. Skipped when no `ρ̂` was measured,
    /// because a panel of 192 NaNs is not a chart.
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Which orthonormal map to measure in. `cumulative` is refused: its coefficients ARE its
    /// horizons, the existing per-horizon family already carries them, and a whitening artifact
    /// for the identity basis would only reproduce `½·ln h`.
    #[arg(long)]
    pub basis: TargetBasis,
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,
    /// Batches of TRAINING origins behind the whitening scales. 16 x 256 rows x 375 origins is
    /// 1.5 M windows, which pins a 192-element second moment far tighter than anything
    /// downstream needs.
    #[arg(long, default_value_t = 16)]
    pub training_batches: usize,
    /// Batches of [70%,80%) calibration origins behind `ρ̂` and `β̂`; `0` measures neither and
    /// produces a whitening-only artifact, which is all `--basis-weight uniform` needs.
    #[arg(long, default_value_t = 8)]
    pub calibration_batches: usize,
}

/// Fit the per-coefficient statistics one arm needs before it can run, and chart what they say.
///
/// Two vectors, two partitions, and the partition is enforced rather than documented:
/// [`target_basis::fit_statistics`] tags the training walk `Training` and the calibration walk
/// `Calibration` beside the reference lists they read, and
/// [`target_basis::BasisStatistics::fit`] refuses any other pairing - fitting on
/// `validation_refs`, the population checkpoint selection reads, is a hard error and not a
/// warning.
///
/// The artifact is authenticated by its own `sha256` over its content and by the basis and
/// `pred_len` it was fitted at, so a DCT artifact cannot be handed to a Haar arm and a tampered
/// vector cannot be handed to anything.
pub fn basis_stats(args: BasisStatsArgs) -> Result<()> {
    ensure!(args.batch_size > 0, "batch size must be positive");
    ensure!(
        !args.basis.is_identity(),
        "--basis cumulative has nothing to measure: its coefficients are its horizons, its \
         whitening prior is the ½·ln h buffer the model already carries, and its per-horizon \
         ρ̂ and β̂ are what timexer_segment_horizon_steps_gain already charts"
    );
    let started = Instant::now();
    let (manifest, corpus, _store, model, device) =
        load_checkpoint(&args.checkpoint, &args.data_dir)?;
    let _guard = tch::no_grad_guard();
    let statistics = target_basis::fit_statistics(
        &corpus,
        &model,
        args.basis,
        device,
        args.batch_size,
        args.training_batches,
        args.calibration_batches,
    )?;
    statistics.store(&args.artifact)?;
    statistics.authenticate(args.basis, corpus.contract.pred_len as i64)?;
    println!(
        "CausalPatch target basis {} | whitening on {} training origin-channel-bars, {:.6} of windows complete | rho measured on {} | artifact {} sha256 {} | {:.1} s",
        args.basis,
        statistics.whitening_bars,
        statistics.whitening_complete_share,
        match statistics.snr_partition {
            Some(partition) => partition.to_string(),
            None => "not measured".to_owned(),
        },
        args.artifact.display(),
        statistics.sha256,
        started.elapsed().as_secs_f64()
    );
    if let (Some(output), Some(_)) = (&args.output, statistics.snr_partition) {
        let transform = target_basis::BasisTransform::new(
            args.basis,
            target_basis::BasisWeight::Snr,
            corpus.contract.pred_len as i64,
            Some(&statistics),
            Device::Cpu,
        )?;
        fs::create_dir_all(output)?;
        reports::write_target_basis(
            output,
            manifest.epoch,
            manifest.step,
            &args.basis.to_string(),
            &target_basis::BasisWeight::Snr.to_string(),
            &statistics.correlation,
            &statistics.amplitude_gain,
            transform.weights(),
            statistics.whitening_complete_share,
            transform.orthonormality_defect(),
            corpus.calibration_refs.len(),
        )?;
    }
    Ok(())
}
