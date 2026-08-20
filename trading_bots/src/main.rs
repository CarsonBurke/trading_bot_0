use clap::{Parser, Subcommand, ValueEnum};
use colored::{self, Colorize};
use shared::{paths::RUNS_PATH, run_dir::RunDir};
use trading_bot_0::torch::model::ModelVariant;
use trading_bot_0::torch::planner::PlannerDataSplit;
use trading_bot_0::torch::train::PretrainArgs;
use trading_bot_0::{genetic, torch};

/// Symbols to paper/live trade when the operator names none.
///
/// Resolved when the `paper` subcommand actually runs, never while clap builds the command
/// tree: this reads the whole packed-bar corpus, which `--help` and every other subcommand
/// have no business doing. Picks the deepest corpus histories rather than whatever happens
/// to sort first alphabetically.
fn default_paper_symbols() -> Vec<String> {
    trading_bot_0::data::universe::deepest_symbols(torch::constants::TICKERS_COUNT as usize)
}

/// `--split-bounds <b0>,<b1>` in epoch millis.
///
/// Two instants rather than a duration or a date: the split is defined by exact wall-clock
/// millis in the corpus, and a run that pins them must reproduce the derived pair to the
/// millisecond or it is scoring different windows.
fn parse_split_bounds(raw: &str) -> Result<(i64, i64), String> {
    let (first, second) = raw
        .split_once(',')
        .ok_or_else(|| format!("expected `<train_val_ms>,<val_test_ms>`, got `{raw}`"))?;
    let parse = |text: &str| -> Result<i64, String> {
        text.trim()
            .parse::<i64>()
            .map_err(|err| format!("`{text}` is not an epoch-millis instant: {err}"))
    };
    let bounds = (parse(first)?, parse(second)?);
    if bounds.0 >= bounds.1 {
        return Err(format!(
            "split bounds must ascend: {} is not before {}",
            bounds.0, bounds.1
        ));
    }
    Ok(bounds)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum StreamingModelVariant {
    #[value(
        name = "uniform-stream",
        alias = "uniform-256-stream",
        alias = "uniform256-stream"
    )]
    UniformStream,
}

impl From<StreamingModelVariant> for ModelVariant {
    fn from(value: StreamingModelVariant) -> Self {
        match value {
            StreamingModelVariant::UniformStream => Self::UniformStream,
        }
    }
}

#[derive(Parser)]
#[command(name = "trading_bot")]
#[command(about = "Trading bot with PPO training and inference", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Genetic {
        #[arg(long, value_enum, default_value_t = genetic::GeneticFamily::TrendBreakout)]
        family: genetic::GeneticFamily,

        #[arg(long)]
        run: Option<String>,

        #[arg(long, default_value_t = 600)]
        generations: usize,

        #[arg(long, default_value_t = 192)]
        population: usize,

        #[arg(long, default_value_t = 0.25)]
        survivor_ratio: f64,

        #[arg(long, value_enum, default_value_t = genetic::TickerSet::Train)]
        train_tickers: genetic::TickerSet,

        #[arg(long, value_enum, default_value_t = genetic::TickerSet::Validation)]
        validation_tickers: genetic::TickerSet,

        #[arg(long, value_enum, default_value_t = genetic::TickerSet::Test)]
        test_tickers: genetic::TickerSet,

        #[arg(long, default_value_t = 5)]
        heavy_report_every: usize,

        #[arg(long, default_value_t = 7)]
        seed: u64,

        #[arg(long, default_value_t = false)]
        skip_additional_downloads: bool,

        #[arg(long, default_value_t = 1.0)]
        mutation_entropy: f64,
    },
    Train {
        #[arg(short, long)]
        weights: Option<String>,

        #[arg(long, value_enum, default_value_t = StreamingModelVariant::UniformStream)]
        model_size: StreamingModelVariant,

        #[arg(long)]
        run: Option<String>,

        /// Reproducible environment, action-sampling, and minibatch seed.
        #[arg(long, default_value_t = 20260811)]
        seed: u64,
    },
    /// Pretrain the discrete distributional bar world model on the local bar corpus.
    Pretrain {
        /// Initialize from an existing pretrain checkpoint. Weights only: training
        /// restarts at step zero with a fresh optimizer and schedule.
        #[arg(short, long)]
        weights: Option<String>,

        #[arg(long)]
        run: Option<String>,

        /// Passes over the training split. One epoch is one pass: the ramp stages own DISJOINT
        /// shares of the corpus and `CoverageAudit::require_full_pass` fails the run when any
        /// assigned window went unissued, so every reachable bar is a prediction target exactly
        /// once per epoch. (The "stage 0 covers only ~27%" shortfall this comment used to
        /// describe belonged to the pre-partition sampler and is gone; all three bardist_v2
        /// gens read stage coverage 1/1/1 at every completed pass.)
        ///
        /// ABOVE ONE MEANS MULTI-EPOCH REUSE, AND NO PER-PASS PANEL CAN SHOW IT.
        /// `pretrain_stage_coverage` and `pretrain_pass_multiplicity` are PER-PASS censuses, so
        /// on the third pass they read "every stage 1.0, every bar exactly once, twice: 0" just
        /// as they do on the first. That is correct within a pass and it was read as a claim
        /// about the run for an entire analysis session, against `pretrain_unique_bar_reuse`
        /// showing 2.85 on the same screen. The RUN-scoped series are `cover_effective_epochs`
        /// and `cover_run_bar_exposure`; those are the ones that answer how many times the model
        /// has seen a bar.
        ///
        /// DEFAULT IS ONE, AND IT IS A MEASURED CHOICE RATHER THAN A CONSERVATIVE ONE.
        /// `bardist_v2` ran this at 3 and its 366,163,264 bars per pass are NOT 366M
        /// independent observations: the corpus is 5,297 symbols over ONE shared wall-clock
        /// grid of 197,916 five-minute instants, and same-instant returns across symbols
        /// correlate at rho = 0.176 (95% CI 0.158..0.201, measured pairwise-complete on the
        /// run's own unfiltered universe). The cross-sectional design effect is
        /// 1 + (1850 - 1) * rho = 327, against a within-symbol serial inflation of only 1.10,
        /// so the effective sample size is 366,163,264 / (327 * 1.10) = 1.0M, interval
        /// [0.57M, 1.13M] — 320x to 640x below the nominal bar count. Against 31.8M
        /// parameters that is 31 parameters per effective observation, where the nominal count
        /// suggests a comfortable 0.087. A second and third pass add ZERO new market-factor
        /// realizations — they re-present the same 1,031 sessions — while adding fitting
        /// pressure, so passes above one buy optimization, never information.
        #[arg(long, default_value_t = 1)]
        epochs: usize,

        /// Override the corpus-derived step count. Diagnostic use only: it decouples
        /// the learning-rate and ramp schedules from the corpus size.
        #[arg(long)]
        steps: Option<usize>,

        /// Fraction of the run held at the FLAT learning-rate plateau, before the linear decay
        /// to the 0.15x floor. Must lie strictly inside `(0, 1)`.
        ///
        /// THIS IS THE ONE KNOB THAT DECIDES WHETHER A RUN'S PASSES AND ITS LEARNING RATE ARE
        /// SEPARABLE. Past the plateau `lr_multiplier` is exactly affine in the step index, so
        /// `d(passes)/d(lr_mult) = -epochs * (1 - F) / (P - L)` is a constant of the recipe:
        /// "saw the data again" and "trained at a lower rate" are the SAME variable there, at
        /// any precision. Only inside the plateau, where the rate is clipped flat, can anything
        /// be attributed to a pass.
        ///
        /// At the default `F = 0.40` a one-epoch run's plateau ends at 0.4 passes, so it always
        /// finishes fully annealed. That matters because the two measurements bracketing this
        /// campaign's open question are one full pass at PEAK rate (`bardist_v2` step 10364,
        /// re-decoded Mincer-Zarnowitz mean slope 1.0058 +/- 0.0355) against one full pass fully
        /// ANNEALED (`bardist_v3_rfirst_1ep` step 10817, 0.6653 +/- 0.0286, which excludes 1.0),
        /// and under a one-epoch budget the first of those operating points is unreachable at
        /// the default. Raising this toward 1.0 makes it reachable. The value is printed in the
        /// startup banner, written into the checkpoint metadata sidecar and charted in the run's
        /// test-split report, because a schedule nobody recorded explains no number later.
        #[arg(long, default_value_t = trading_bot_0::torch::train::pretrain::LR_PLATEAU_FRACTION)]
        lr_plateau_fraction: f64,

        /// Batch size at the first ramp stage. The declared ceiling for the later stages is
        /// 2x and 3x, but the ramp that RUNS is derived from a device capacity probe taken
        /// before the first step, and this value is CLAMPED at startup if it does not fit at
        /// the deployed context. The banner prints the measured bytes per bar-token, the
        /// resulting ceiling and the achievable batch at each of several contexts.
        #[arg(long, default_value_t = 24)]
        batch_size: usize,

        /// Seeds the TRAINING sampler, support fitting and the torch/CUDA RNGs. It does NOT
        /// move the pinned evaluation windows or the PIT draws — those are pinned by the
        /// campaign constant `EVAL_WINDOW_SEED` — so seed replicates measure training noise
        /// on an unchanged bench and every run stays paired on identical windows.
        #[arg(long, default_value_t = 0x5EED)]
        seed: u64,

        #[arg(long, default_value_t = trading_bot_0::data::ingest::bars_dir().to_string_lossy().into_owned())]
        data_dir: String,

        /// Bar resolution to train on, in seconds.
        #[arg(long, default_value_t = 300)]
        resolution_secs: u32,

        /// Drop symbols with fewer bars than this. The default guarantees every
        /// symbol contributes at least one full-context window to each split, and is
        /// shared with `ingest --min-bars` so the universe and the split agree on
        /// which files exist.
        #[arg(long, default_value_t = trading_bot_0::torch::dataset::DEFAULT_MIN_BARS)]
        min_bars: usize,

        /// Extra bar resolutions, in seconds, to train on ALONGSIDE `--resolution-secs`.
        ///
        /// EMPTY BY DEFAULT, so the daily corpus enters a run only when named. `--auxiliary-resolutions 86400`
        /// adds `long_data/bars/*.86400.bars`: 4,748 symbols and 21.5M bars back to 1970-01-02, of which
        /// 74.9% predate the 5-minute corpus and therefore carry the 2000, 2008 and 2020 crash regimes the
        /// deployment corpus does not contain. Each auxiliary resolution gets its OWN fitted supports and
        /// its own ramp; the resolution is a trunk conditioning id, and selection and promotion stay on the
        /// deployment resolution's held-out NLL.
        #[arg(long, value_delimiter = ',', num_args = 0..)]
        auxiliary_resolutions: Vec<u32>,

        /// Training-split bars drawn to fit the equal-mass bin supports.
        #[arg(long, default_value_t = 4_000_000)]
        support_samples: usize,

        /// Scoring rule for the next-bar log-likelihood: `smoothed`, `hard` or `density`.
        ///
        /// THE THREE MODES ARE NOT COMPARABLE IN ABSOLUTE NATS. They differ by additive
        /// constants that depend on the binning, so a `density` figure sits tens of nats
        /// below a `smoothed` one on the identical model. The mode is written into the
        /// checkpoint metadata, folded into the lineage hash, and `pretrain-compare`
        /// refuses to pair two runs that disagree.
        ///
        /// * `density` (default) — the proper log-likelihood of the MIXED law we observe:
        ///   a probability MASS on an atom, a DENSITY inside a continuous bin. No
        ///   unreachable floor, and up to discretization error no dependence on the bin
        ///   count, which is what makes `NUM_BAR_BINS` ablatable.
        /// * `hard` — one-hot cross entropy on the containing bin. Proper for the
        ///   discretized law and floor-free, but its scale moves with the bin count.
        /// * `smoothed` — the old Gaussian label smoothing at 0.75x the local bin width.
        ///   Proper for the SMOOTHED law rather than the observed one, and it imposes an
        ///   unreachable 4.6482 nats/bar floor. Kept only so the campaign's earlier runs
        ///   stay comparable.
        #[arg(long, default_value_t = trading_bot_0::torch::bar_dist::BarScoring::default())]
        scoring: trading_bot_0::torch::bar_dist::BarScoring,

        /// Recursive latent-dynamics rollout depth. The NextLat reference defaults to 1;
        /// the losses are averaged over the horizon either way.
        #[arg(long, default_value_t = 4)]
        dyn_horizon: usize,

        /// Weight on the NextLat hidden-state term, i.e. the reference's `lambda_mse`.
        ///
        /// `1.0` paired with a mean reduction over every element of `[B, T, BAR_MODEL_DIM]`
        /// is the NextLat reference configuration (`models/model_nextlat.py:303-308` and
        /// `defaults.yaml: lambda_mse: 1.0`, arXiv 2511.05963), and it is what this repo
        /// originally ran.
        ///
        /// It briefly became a landmine: the reduction was changed to SUM the 512-wide
        /// feature axis on the theory that the term was inert, which multiplied it by 512
        /// while the default stayed at `1.0`. At that strength `dyn` measured 28 against
        /// `nll` 17 — 62% of the objective — and `nll` ROSE from 16.34 to 17.19 over 4000
        /// steps. Lowering the default to `1e-2` was then proposed, but `1e-2` under a
        /// summed reduction is `5.12` in reference units, i.e. five times the paper rather
        /// than a hundredth of it. The reduction is fixed instead, so this weight is once
        /// again the reference's and is width-independent.
        ///
        /// Every training line prints each term's share of the objective's magnitude and
        /// the run warns when an auxiliary term holds more than 25% of it for 100
        /// consecutive steps, so a repeat cannot go unnoticed for 4000 steps again.
        #[arg(long, default_value_t = 1.0)]
        lambda_dyn: f64,

        /// Weight on the NextLat categorical-KL term, i.e. the reference's `lambda_kl`.
        #[arg(long, default_value_t = 1.0)]
        lambda_kl: f64,

        /// Weight on the EXPECTED-LOG-GROWTH term: `-log(1 + f_hat R)` at the log-optimal
        /// fraction of the model's own `p(r|past)`, with the same-bar `s` marginalized out
        /// and the fraction clamped at the trade bench's leverage cap.
        ///
        /// The default is `growth::LAMBDA_GROWTH`, which was DERIVED from a gradient-norm
        /// measurement rather than swept: the term's magnitude is ~5e-4 nats against the
        /// likelihood's ~4.93, so a weight chosen to make it look substantial in the
        /// objective would be sizing it on the wrong quantity. `0.0` is the ablation's
        /// control arm — the term is still computed and charted, it just does not train.
        #[arg(long, default_value_t = trading_bot_0::torch::train::growth::LAMBDA_GROWTH)]
        lambda_growth: f64,

        /// Held-out windows in each pinned evaluation set. Pinned by the campaign constant
        /// `EVAL_WINDOW_SEED`, so they are identical across runs, seeds and ablations.
        #[arg(long, default_value_t = 4096)]
        validation_windows: usize,

        /// Context of the across-run diagnostic evaluation. Promotion always
        /// evaluates at the full deployed context instead.
        #[arg(long, default_value_t = trading_bot_0::torch::train::pretrain::BAR_CONTEXT_RAMP_START)]
        diagnostic_context: i64,

        /// Pinned windows carried into the candle-rollout snapshot reports.
        #[arg(long, default_value_t = 8)]
        snapshot_windows: usize,

        /// Ancestral draws behind each snapshot window's quantile fan.
        ///
        /// The rollout is linear in this and now runs at EVERY epoch boundary, so it is
        /// the knob to turn down on a shared card. Lowering it widens every band's own
        /// error bar as `1/sqrt(n)`, which the chart states rather than hides.
        #[arg(long, default_value_t = trading_bot_0::torch::train::pretrain_reports::SNAPSHOT_SAMPLES)]
        snapshot_samples: usize,

        /// Validate every N optimizer steps. Validation also always runs at every
        /// epoch boundary and at the end of the run.
        #[arg(long, default_value_t = 1000)]
        validate_every: usize,

        /// Write a step-tagged crash-recovery checkpoint every N optimizer steps (0 disables).
        ///
        /// Promotion is gated on the deployed context and the batch ramp is memory-gated, so a
        /// run can go two epochs without producing a promoted checkpoint. This bounds what a
        /// crash destroys; the newest few are kept and older ones are pruned.
        #[arg(long, default_value_t = trading_bot_0::torch::train::pretrain::DEFAULT_CHECKPOINT_EVERY)]
        checkpoint_every: usize,

        /// Print a training line every N optimizer steps (0 disables).
        #[arg(long, default_value_t = 20)]
        log_every: usize,

        /// Pin the two split instants as `<b0>,<b1>` epoch millis. Defaults to the campaign
        /// pin `ingest::PINNED_SPLIT_BOUNDS`.
        ///
        /// Ingest appends continuously and the bounds are percentiles of the trading-time
        /// axis, so a derived boundary moves with the corpus — after the survivorship
        /// expansion it lands 26 days EARLIER, which drops universe-ranking sessions into
        /// validation and reopens the selection leak. Pinning is therefore the default, and
        /// the pin is checked against the instant the symbol universe was ranked as of.
        #[arg(long, value_parser = parse_split_bounds)]
        split_bounds: Option<(i64, i64)>,

        /// Re-derive the split instants from the current corpus instead of the campaign
        /// pin. Diagnostic use only: such a run is comparable to nothing, and
        /// `pretrain-compare` refuses to pair it with anything else.
        #[arg(long, default_value_t = false)]
        derive_split_bounds: bool,

        /// Explicit path to the bin supports, instead of the corpus default.
        #[arg(long)]
        supports: Option<String>,

        /// Reuse cached supports whose recorded provenance does not match this corpus.
        /// Without it a mismatch is a hard error. Freezing is right mid-campaign — the
        /// supports define the nll_bar scale — but it must be a stated decision, and it is
        /// written into the checkpoint metadata.
        #[arg(long, default_value_t = false)]
        freeze_supports: bool,

        /// Train only on symbols clearing this median dollar volume; 0 uses every file on
        /// disk. The split instants are derived before the filter, so both arms of the
        /// ablation are scored over the same wall-clock held-out window.
        #[arg(long, default_value_t = 0.0)]
        min_dollar_volume: f64,

        /// Refuse to start if measured capacity would REDUCE --batch-size, instead of
        /// clamping to what fits.
        ///
        /// Set this on every arm of an ablation. The capacity probe reads free VRAM at
        /// startup, so two runs launched identically on a shared card can land on different
        /// base batches and therefore different step counts: measured, the two arms of the
        /// expected-log-growth ablation ran at base 23 / 10818 steps and base 21 / 11847
        /// steps from the same `--batch-size 24`, because 16.37 and 14.94 GiB were free.
        /// Clamping is right for a production run and wrong for a controlled comparison, and
        /// only the caller knows which this is.
        #[arg(long, default_value_t = false)]
        exact_batch: bool,
    },
    /// Candle pictures of an EXISTING pretrain checkpoint against the realized bars.
    ///
    /// A run only writes candle snapshots after its first promotion, which cannot happen
    /// before the context ramp reaches the deployed length. This produces the same
    /// pictures from any checkpoint on disk, on the same pinned validation windows the
    /// run charts itself, so a mid-ramp model can be looked at rather than described.
    ///
    /// The corpus flags MUST match the run's, or the pinned windows are drawn from a
    /// different symbol set and the pictures depict different data.
    PretrainCandles {
        /// Checkpoint to picture, e.g. `weights/pretrain_last.ot`. The
        /// `.metadata.json` and `.supports.<res>.json` siblings are resolved from this
        /// path, so a copy taken out of a live run's weights directory must keep the
        /// same file stem.
        #[arg(long)]
        weights: String,

        /// Directory the `.report.bin` pictures are written into.
        #[arg(long)]
        output: String,

        /// Pinned validation windows to picture. Each is rolled out on its own, so this
        /// costs wall-clock rather than VRAM. The count is part of the pin, so it must
        /// EQUAL the run's `--snapshot-windows` (both default to 8) for the windows to be
        /// the run's own; a smaller count is a different draw, not a prefix.
        #[arg(long, default_value_t = 8)]
        windows: usize,

        /// Ancestral samples per window. The rollout KV cache is linear in this and is
        /// the whole VRAM cost of the command; lower it on a shared card.
        #[arg(long, default_value_t = trading_bot_0::torch::train::pretrain_reports::SNAPSHOT_SAMPLES)]
        samples: usize,

        /// Conditioning context. The last 100 bars of it are held out as the realized
        /// continuation, exactly as the in-run snapshot does. Must match the run's
        /// `--diagnostic-context` for the windows to be the run's own.
        #[arg(long, default_value_t = trading_bot_0::torch::train::pretrain::BAR_CONTEXT_RAMP_START)]
        context: i64,

        /// Optimizer step the checkpoint reached. Names the output files only.
        #[arg(long, default_value_t = 0)]
        step: usize,

        #[arg(long, default_value_t = trading_bot_0::data::ingest::bars_dir().to_string_lossy().into_owned())]
        data_dir: String,

        #[arg(long, default_value_t = 300)]
        resolution_secs: u32,

        #[arg(long, default_value_t = trading_bot_0::torch::dataset::DEFAULT_MIN_BARS)]
        min_bars: usize,

        /// Pin the two split instants as `<b0>,<b1>` epoch millis. Defaults to the
        /// campaign pin `ingest::PINNED_SPLIT_BOUNDS`.
        #[arg(long, value_parser = parse_split_bounds)]
        split_bounds: Option<(i64, i64)>,

        /// Re-derive the split instants from the current corpus. Diagnostic use only.
        #[arg(long, default_value_t = false)]
        derive_split_bounds: bool,

        /// Liquidity floor the run used; 0 uses every file on disk.
        #[arg(long, default_value_t = 0.0)]
        min_dollar_volume: f64,
    },
    /// The log-optimal (Kelly) trading bench of an EXISTING pretrain checkpoint.
    ///
    /// Answers what the predictive distribution is WORTH: it runs the identical
    /// growth-optimal policy on the model's conditional law, on the fitted unconditional
    /// marginal, on buy-and-hold and on a perfect-foresight oracle, over the same pinned
    /// held-out windows, and reports the model's edge over the unconditional null with a
    /// block-bootstrap interval and the transaction cost at which that edge vanishes.
    ///
    /// The corpus flags MUST match the run's, or the pinned windows are drawn from a
    /// different symbol set and the numbers describe different data.
    PretrainTrade {
        /// Checkpoint to bench, e.g. `weights/pretrain_best.ot`.
        #[arg(long)]
        weights: String,

        /// Directory the `pretrain_trade_*.report.bin` charts are written into.
        #[arg(long)]
        output: String,

        /// Held-out split to trade. `test` is scored once and is the number that counts.
        #[arg(long, value_enum, default_value_t = PlannerDataSplit::Validation)]
        split: PlannerDataSplit,

        /// Pinned windows to draw. The bench trades the first
        /// `trade_bench::TRADE_WINDOWS` of them. The count is part of the pin, so it must
        /// EQUAL the run's `--validation-windows` for the windows to be the run's own.
        #[arg(long, default_value_t = 4096)]
        windows: usize,

        /// Conditioning context. Must match the context the checkpoint was SELECTED at.
        #[arg(long, default_value_t = trading_bot_0::torch::train::pretrain::BAR_CONTEXT_RAMP_START)]
        context: i64,

        /// Evaluation batch, in windows.
        #[arg(long, default_value_t = 8)]
        batch_size: usize,

        #[arg(long, default_value_t = trading_bot_0::data::ingest::bars_dir().to_string_lossy().into_owned())]
        data_dir: String,

        #[arg(long, default_value_t = 300)]
        resolution_secs: u32,

        #[arg(long, default_value_t = trading_bot_0::torch::dataset::DEFAULT_MIN_BARS)]
        min_bars: usize,

        /// Pin the two split instants as `<b0>,<b1>` epoch millis. Defaults to the
        /// campaign pin `ingest::PINNED_SPLIT_BOUNDS`.
        #[arg(long, value_parser = parse_split_bounds)]
        split_bounds: Option<(i64, i64)>,

        /// Re-derive the split instants from the current corpus. Diagnostic use only.
        #[arg(long, default_value_t = false)]
        derive_split_bounds: bool,

        /// Liquidity floor the run used; 0 uses every file on disk.
        #[arg(long, default_value_t = 0.0)]
        min_dollar_volume: f64,
    },
    /// Does the predictor have a forecast HORIZON it can afford to trade?
    ///
    /// Break-even cost is gross edge over turnover, so the one lever left to a signal that
    /// cannot pay the spread at a 5-minute rebalance is trading less often. This sweeps the
    /// holding period and measures two distinct policies at each one: the CONTROL, which keeps
    /// the one-bar forecast and merely holds it (which is all a no-trade band does), and the
    /// EXPERIMENT, which forecasts the k-bar aggregate log return from a sampled multi-bar
    /// rollout and sizes on that law. Equal-weight, the unconditional marginal null and a
    /// perfect-foresight oracle are measured at every horizon beside the model, because a
    /// corner where a baseline wins is not a model result.
    ///
    /// Inference only, over the calendar-aligned PINNED held-out panel.
    PretrainHorizon {
        /// Checkpoint to sweep, e.g. `weights/pretrain_best.ot`. Its metadata and supports
        /// sidecars are resolved beside it.
        #[arg(long)]
        weights: String,

        /// Directory the `pretrain_horizon_frontier.report.bin` chart is written into.
        #[arg(long)]
        output: String,

        #[arg(long, default_value_t = trading_bot_0::data::ingest::bars_dir().to_string_lossy().into_owned())]
        data_dir: String,

        #[arg(long, default_value_t = 300)]
        resolution_secs: u32,

        /// Pin the two split instants as `<b0>,<b1>` epoch millis. Defaults to the campaign
        /// pin `ingest::PINNED_SPLIT_BOUNDS`, which is what makes the panel held out.
        #[arg(long, value_parser = parse_split_bounds)]
        split_bounds: Option<(i64, i64)>,

        /// Panel breadth: at most this many symbols, ranked by dollar volume measured
        /// STRICTLY BEFORE the traded span.
        #[arg(long, default_value_t = 48)]
        max_symbols: usize,

        /// Panel length, in calendar instants of the deployment resolution.
        #[arg(long, default_value_t = 7_800)]
        max_instants: usize,

        /// Flat one-way cost, in bps, the net-growth column is charged at. The headline
        /// break-even column is a flat-cost equivalent and does not depend on it.
        #[arg(long, default_value_t = trading_bot_0::torch::train::portfolio::DEFAULT_COST_BPS)]
        cost_bps: f32,

        /// Book capital, which is what makes a size a fraction of ADV and therefore a cost.
        #[arg(long, default_value_t = 1.0e7)]
        capital_usd: f64,

        /// Gross exposure cap imposed at every rebalance.
        #[arg(long, default_value_t = trading_bot_0::torch::train::portfolio::DEFAULT_GROSS_CAP)]
        gross_cap: f64,

        /// Monte-Carlo paths per (name, rebalance) of the k-bar rollout.
        #[arg(long, default_value_t = trading_bot_0::torch::train::horizon::DEFAULT_SAMPLES)]
        samples: usize,

        /// Independent replicate sample sets, which is where the reported standard errors of
        /// the sampled rows come from.
        #[arg(long, default_value_t = trading_bot_0::torch::train::horizon::DEFAULT_REPLICATES)]
        replicates: usize,

        #[arg(long, default_value_t = 0x5EED)]
        seed: i64,

        /// Run on the CPU even where CUDA is available.
        #[arg(long, default_value_t = false)]
        cpu: bool,

        /// Label carried into the chart title.
        #[arg(long, default_value_t = String::from("horizon"))]
        label: String,
    },
    /// Does the traded conditional MEAN stay calibrated, and does correcting it recover the
    /// economics that were lost?
    ///
    /// Regresses the realized `r` on the model's predicted conditional mean (Mincer-Zarnowitz:
    /// perfect calibration is intercept 0, slope 1) at each of several checkpoints, and scores
    /// a policy sized on the RECALIBRATED mean beside the untouched one at every leverage cap.
    /// The slope is fitted on pinned held-out windows the bench does not trade AND whose
    /// `(symbol, calendar month)` blocks are absent from the traded prefix, because a slope
    /// fitted on the bars it is evaluated on manufactures its own improvement.
    ///
    /// The corpus flags MUST match the run's, or the pinned windows are drawn from a different
    /// symbol set and the numbers describe different data.
    PretrainCalibration {
        /// Checkpoint to measure, as `path@step`, repeatable. The step is the x-axis of the
        /// reported trend and is not recorded in the metadata sidecar, so it is stated here.
        #[arg(long = "checkpoint", required = true)]
        checkpoints: Vec<String>,

        /// Directory the `pretrain_mean_calibration` and `pretrain_shrunk_policy` charts are
        /// written into.
        #[arg(long)]
        output: String,

        /// Held-out split to measure on.
        #[arg(long, value_enum, default_value_t = PlannerDataSplit::Validation)]
        split: PlannerDataSplit,

        /// Pinned windows to DRAW. Must equal the run's `--validation-windows` for the traded
        /// prefix to be the run's own windows, and must leave a block-disjoint remainder for
        /// the fit slice.
        ///
        /// RAISING THIS MOVES THE TRADED PREFIX. `BarSampler::pinned_windows` allocates a
        /// per-symbol quota from the draw size and skips symbols whose quota rounds to zero, so
        /// a larger draw un-skips symbols that then insert themselves ahead of names the old
        /// prefix held: measured on the live corpus, 4096 -> 8192 changes 3,794 of Val's 4,729
        /// symbols' quotas. To widen the FIT slice, raise `--fit-windows` out of the remainder
        /// this draw already has; to widen the TRADED slice, say so with `--trade-windows`.
        #[arg(long, default_value_t = 4096)]
        windows: usize,

        /// Windows of the block-disjoint remainder to fit the calibration slope on. Truncated to
        /// what the remainder holds, so asking for more than exists is safe.
        #[arg(long, default_value_t = 256)]
        fit_windows: usize,

        /// Windows of the drawn prefix to TRADE.
        ///
        /// Every published number was measured at the default, so leaving it alone reproduces
        /// them exactly. Raise it only on a panel that is not being compared to anything — a
        /// one-shot `test` read, where the interval is set by the traded slice's
        /// `(symbol, calendar month)` block count and the split holds 43,466 near-disjoint
        /// windows at context 896 against the 256 a default draw trades.
        #[arg(long, default_value_t = trading_bot_0::torch::train::trade_bench::TRADE_WINDOWS)]
        trade_windows: usize,

        /// Draw the windows, block them, write the manifest and the held-out power census, then
        /// STOP — before any checkpoint is opened and before any economic number exists.
        ///
        /// The rehearsal for a split that is scored ONCE. It establishes that the command
        /// addresses the intended data, that the fit and traded slices are block-disjoint, and
        /// that the population has the power to resolve the effect being looked for, without
        /// spending the draw.
        #[arg(long, default_value_t = false)]
        dry_run: bool,

        /// Restrict the TRADED windows to these symbols, so an edge is measured on exactly the
        /// names a cost was priced on. Repeatable or comma-separated. Empty trades the whole
        /// prefix. The fit slice is never restricted.
        #[arg(long, value_delimiter = ',')]
        restrict_symbols: Vec<String>,

        /// Conditioning context. Must match the context the compared bench reads were taken
        /// at, which for a run's own diagnostic series is the fixed ramp-start context.
        #[arg(long, default_value_t = trading_bot_0::torch::train::pretrain::BAR_CONTEXT_RAMP_START)]
        context: i64,

        /// Evaluation batch, in windows.
        #[arg(long, default_value_t = 8)]
        batch_size: usize,

        #[arg(long, default_value_t = trading_bot_0::data::ingest::bars_dir().to_string_lossy().into_owned())]
        data_dir: String,

        #[arg(long, default_value_t = 300)]
        resolution_secs: u32,

        #[arg(long, default_value_t = trading_bot_0::torch::dataset::DEFAULT_MIN_BARS)]
        min_bars: usize,

        /// Pin the two split instants as `<b0>,<b1>` epoch millis. Defaults to the
        /// campaign pin `ingest::PINNED_SPLIT_BOUNDS`.
        #[arg(long, value_parser = parse_split_bounds)]
        split_bounds: Option<(i64, i64)>,

        /// Re-derive the split instants from the current corpus. Diagnostic use only.
        #[arg(long, default_value_t = false)]
        derive_split_bounds: bool,

        /// Liquidity floor the run used; 0 uses every file on disk.
        #[arg(long, default_value_t = 0.0)]
        min_dollar_volume: f64,
    },
    /// Measure the FITTED per-bin conditional means of an EXISTING bar support and write the
    /// v5 artifact carrying them, on UNCHANGED bin geometry.
    ///
    /// The live `bar_supports.<res>.json` is `format_version: 4` and carries no `bin_means`, so
    /// `BarSupports::bin_means()` returns `None` and every first-moment decode in the tree falls
    /// back to `centers`, which prices the two open-ended catch-all bins at the support BOUNDS.
    /// On the 300s `r` support those bounds are -883.32 and +880.38 bps: 1.4474% of the mass
    /// controlling 41.00% of the absolute first moment and 92.38% of the decoded mean's
    /// estimation variance. This measures what those bins actually realized.
    ///
    /// GEOMETRY IS NEVER REFITTED. The bin edges, atoms, histogram and smoothed marginal are
    /// carried across unchanged and the result is checked member-by-member against the source,
    /// because the edges define the `nll_bar` scale and every persisted report is expressed on
    /// it. `--samples` and `--seed` MUST be the run's own `--support-samples` and `--seed`: the
    /// pass re-draws that exact sample and REFUSES unless the redraw reproduces the persisted
    /// bin masses, which identifies both the sample and the binning rule against the artifact.
    ///
    /// The result is written to a NEW path. Every checkpoint's `.supports.<res>.json` sidecar is
    /// covered by its own `supports_sha256` and by `lineage_sha256`, so an in-place upgrade would
    /// make existing checkpoints unloadable against their own training geometry.
    ///
    /// Nothing is switched: `MeanDecode::Edge` remains the default and the fitted decode is
    /// available by name only. This pass measures and reports; it changes no predicted mean.
    BarSupportsMoments {
        /// Support to measure moments for. Read, never written.
        #[arg(long, default_value_t = trading_bot_0::data::ingest::bars_dir().join("bar_supports.300.json").to_string_lossy().into_owned())]
        supports: String,

        /// Where the upgraded v5 artifact lands. MUST differ from `--supports`.
        #[arg(long, default_value_t = trading_bot_0::data::ingest::bars_dir().join("bar_supports.300.v5.json").to_string_lossy().into_owned())]
        output_supports: String,

        /// Directory the `support_decode_moments` and `support_decode_bins` charts are written
        /// into, i.e. a run's `gens/<n>`.
        #[arg(long)]
        output: String,

        /// Rows to draw from the train region. MUST equal the `sample_count` the support's
        /// provenance records, which is checked.
        #[arg(long, default_value_t = 4_000_000)]
        samples: usize,

        /// Draw seed. MUST be the `train_seed` of the run that fitted the support.
        #[arg(long, default_value_t = 0x5EED)]
        seed: u64,

        /// Largest per-bin absolute mass deviation accepted when identifying the redrawn sample
        /// against the persisted histogram. An identical sample under an identical binning rule
        /// agrees exactly, so any nonzero value here is slack for decimal JSON alone.
        #[arg(long, default_value_t = trading_bot_0::torch::train::support_moments::DEFAULT_MASS_TOLERANCE)]
        mass_tolerance: f64,

        #[arg(long, default_value_t = trading_bot_0::data::ingest::bars_dir().to_string_lossy().into_owned())]
        data_dir: String,

        #[arg(long, default_value_t = 300)]
        resolution_secs: u32,

        #[arg(long, default_value_t = trading_bot_0::torch::dataset::DEFAULT_MIN_BARS)]
        min_bars: usize,

        /// Pin the two split instants as `<b0>,<b1>` epoch millis. Defaults to the campaign pin;
        /// the support's own provenance is checked against whatever this resolves to.
        #[arg(long, value_parser = parse_split_bounds)]
        split_bounds: Option<(i64, i64)>,

        /// Re-derive the split instants from the current corpus. Diagnostic use only.
        #[arg(long, default_value_t = false)]
        derive_split_bounds: bool,

        /// Liquidity floor the run used; 0 uses every file on disk, which is what bardist_v2 did.
        #[arg(long, default_value_t = 0.0)]
        min_dollar_volume: f64,
    },
    /// Can a CONTINUOUS per-DOF mixed likelihood replace the 128-way equal-mass discrete support?
    ///
    /// The offline GATE, run before any retrain is spent. Fits candidate families to the
    /// UNCONDITIONAL bar law on the SAME train-region draw the live supports were fitted from —
    /// same accessor, same budget, same seed — and either licenses the replacement or names the
    /// measured fact that kills each family. It touches no trainer, no head, no loss and no
    /// `BAR_CHAIN`, and it starts no training run.
    ///
    /// FOUR DELIVERABLES. (a) Whether the mixed likelihood reproduces the measured atom masses,
    /// per DOF — exact BY CONSTRUCTION as a family parameter, so what is measured is that the
    /// redraw reproduces the shares the artifact recorded, and the pass REFUSES if it does not.
    /// (b) A fitted Hill tail index on `r`, with its standard error, against the measured
    /// 1.66-1.84 figure, which is a SPREAD OF SIX PAIRWISE SLOPES and not an estimate — the two
    /// are never compared as though both were. (c) The marginal NLL of each family against the
    /// discrete marginal under `scoring: density`, which is already a log DENSITY on the same
    /// mixed measure because the density rule adds `E[ln width]`, so no offset is applied to
    /// either column. (d) The truncation bound `R_max` a declared max leverage licenses, from
    /// `1 + F(exp(r) - 1) > 0`, tabulated through both live constants.
    ///
    /// CPU ONLY and bounded: one drawn `Vec<BarDof>` buffer, streaming rayon folds with `O(K)`
    /// accumulators, and two fixed-size probes. Nothing is written but reports.
    BarFamily {
        /// Discrete support the families are scored against, and the source of the atom set, the
        /// chart grid and the per-DOF resolution floors. Read, never written.
        #[arg(long, default_value_t = trading_bot_0::data::ingest::bars_dir().join("bar_supports.300.json").to_string_lossy().into_owned())]
        supports: String,

        /// Directory the ten `bar_family_*` charts are written into, i.e. a run's `gens/<n>`.
        #[arg(long)]
        output: String,

        /// Rows to draw from the train region. MUST equal the `sample_count` the support's
        /// provenance records, which is checked.
        #[arg(long, default_value_t = 4_000_000)]
        samples: usize,

        /// Draw seed. MUST be the `train_seed` of the run that fitted the support.
        #[arg(long, default_value_t = 0x5EED)]
        seed: u64,

        /// Smallest component count in the sweep.
        #[arg(long, default_value_t = 4)]
        k_min: usize,

        /// Largest component count in the sweep.
        #[arg(long, default_value_t = 8)]
        k_max: usize,

        /// Largest absolute atom-share deviation accepted between the redraw and the artifact. The
        /// redraw is the SAME draw by construction, so this is decimal-serialization slack alone.
        #[arg(long, default_value_t = trading_bot_0::torch::train::bar_family::DEFAULT_ATOM_TOLERANCE)]
        atom_tolerance: f64,

        #[arg(long, default_value_t = trading_bot_0::data::ingest::bars_dir().to_string_lossy().into_owned())]
        data_dir: String,

        #[arg(long, default_value_t = 300)]
        resolution_secs: u32,

        #[arg(long, default_value_t = trading_bot_0::torch::dataset::DEFAULT_MIN_BARS)]
        min_bars: usize,

        /// Pin the two split instants as `<b0>,<b1>` epoch millis. Defaults to the campaign pin;
        /// the support's own provenance is checked against whatever this resolves to.
        #[arg(long, value_parser = parse_split_bounds)]
        split_bounds: Option<(i64, i64)>,

        /// Re-derive the split instants from the current corpus. Diagnostic use only.
        #[arg(long, default_value_t = false)]
        derive_split_bounds: bool,

        /// Liquidity floor the run used; 0 uses every file on disk.
        #[arg(long, default_value_t = 0.0)]
        min_dollar_volume: f64,
    },
    /// Are the extreme `r` bars in the corpus MARKET MOVES or UNADJUSTED CORPORATE-ACTION SEAMS?
    ///
    /// A read-only audit of the STORED BARS. Three live numbers rest on the outermost `r` bars and
    /// on nothing else: the two catch-all bins of the `r` support are placed at the
    /// `BAR_SUPPORT_CLIP_QUANTILE` quantiles and hold 1.4474% of the mass between them; the
    /// measured tail index on `|r|` is a spread of six pairwise log-log slopes, which a handful of
    /// artificial jumps inflates; and the leverage/ruin licence is `1 + F(exp(r) - 1) > 0` read off
    /// `lo[r][0]` and `hi[r][127]`. If those quantiles landed on a stock split, a corporate action
    /// is setting the leverage cap.
    ///
    /// FOUR INDEPENDENT CRITERIA decide it, and the verdict rests on their CONJUNCTION. A split
    /// seam sits on a SIMPLE RATIONAL ratio, happens at a SESSION BOUNDARY, leaves the bar's own
    /// log range `s` and volume `w` UNREMARKABLE because a level shift does not trade, and is an
    /// ISOLATED one-bar discontinuity. The last one is what separates a split from the other
    /// extreme-`r` population: a bad print at a fifth of the price reverts on the next bar and
    /// therefore prints the OPPOSITE extreme beside it, which is the only way one symbol can put
    /// both `-ln 5` and `+ln 5` into the same draw.
    ///
    /// Writes nothing but reports. No corpus file, no ingest path, no support artifact and no live
    /// constant is touched; the cleaned support edges and the cleaned ruin licence it reports are
    /// COUNTERFACTUALS.
    ///
    /// CPU ONLY and bounded: a rayon fold over SERIES whose accumulator is fixed-size histograms
    /// plus two explicitly capped buffers, and one drawn `Vec<BarDof>` for the tail control, which
    /// is the same allocation `BarCorpus::fit_supports` already makes.
    BarSplitSeams {
        /// Support whose `bin_of` places every bar and whose outer `r` bounds set the live ruin
        /// licence. Read, never written.
        #[arg(long, default_value_t = trading_bot_0::data::ingest::bars_dir().join("bar_supports.300.json").to_string_lossy().into_owned())]
        supports: String,

        /// A second support file whose DOF `r` bounds are compared against `--supports`, so the
        /// claim "which geometry this used does not matter" is a measurement. Empty to skip.
        #[arg(long, default_value_t = trading_bot_0::data::ingest::bars_dir().join("bar_supports.300.v5.json").to_string_lossy().into_owned())]
        cross_check_supports: String,

        /// Directory the six `bar_seam_*` charts are written into, i.e. a run's `gens/<n>`.
        #[arg(long)]
        output: String,

        /// Rows to draw for the tail CONTROL. MUST equal the `sample_count` the support's
        /// provenance records, which is checked: the control only means something if it is the same
        /// draw the live tail figure was measured on.
        #[arg(long, default_value_t = 4_000_000)]
        samples: usize,

        /// Draw seed. MUST be the `train_seed` of the run that fitted the support.
        #[arg(long, default_value_t = 0x5EED)]
        seed: u64,

        #[arg(long, default_value_t = trading_bot_0::data::ingest::bars_dir().to_string_lossy().into_owned())]
        data_dir: String,

        #[arg(long, default_value_t = 300)]
        resolution_secs: u32,

        #[arg(long, default_value_t = trading_bot_0::torch::dataset::DEFAULT_MIN_BARS)]
        min_bars: usize,

        /// Pin the two split instants as `<b0>,<b1>` epoch millis. Defaults to the campaign pin;
        /// the support's own provenance is checked against whatever this resolves to.
        #[arg(long, value_parser = parse_split_bounds)]
        split_bounds: Option<(i64, i64)>,

        /// Re-derive the split instants from the current corpus. Diagnostic use only.
        #[arg(long, default_value_t = false)]
        derive_split_bounds: bool,

        /// Liquidity floor the run used; 0 uses every file on disk.
        #[arg(long, default_value_t = 0.0)]
        min_dollar_volume: f64,
    },
    /// Does the run's THIRD pass over the corpus MEMORIZE, and does that memorization move the
    /// held-out mean slope?
    ///
    /// Two measurements, only one of which carries a verdict. The epoch spine is train-split
    /// against held-out NLL at each pass count; it is CONTAMINATED — train and val are
    /// calendar-disjoint so its LEVEL mixes regime, and `lr_multiplier` is affine in step past
    /// the plateau so its TRAJECTORY mixes passes with a learning rate that weakens implicit
    /// regularization on its own — and it is reported with that attached rather than cleaned.
    ///
    /// The discriminator is the ONE-REPETITION contrast. At one checkpoint, bars in an
    /// already-issued window of the deployed ramp stage have been trained on exactly one more
    /// time than bars in a not-yet-issued one, at the same weights, the same learning rate, the
    /// same momentum, the same context and the same mean conditioning depth. The split is
    /// randomized rather than merely matched, because `PassPlan::build_layout` ends with a
    /// global per-stage shuffle, so the issued prefix is a uniformly random subset.
    ///
    /// `--train-seed`, `--batch-ramp` and the corpus flags MUST be the run's own: the partition
    /// being reconstructed is the TRAINING sampler's, keyed by `(train_seed, epoch)`, so a
    /// different seed reconstructs a different partition and every exposure count would be wrong
    /// while looking right. The seed is checked against each checkpoint's metadata sidecar.
    PretrainMemProbe {
        /// Epoch-spine checkpoint as `path@step`, repeatable. The step is stated because the
        /// metadata sidecar records `reached_context` and the seeds but no `global_step`.
        #[arg(long = "checkpoint", required = true)]
        checkpoints: Vec<String>,

        /// Checkpoint whose weights carry the one-repetition contrast, as `path@step`. Its step
        /// decides the partition, so it is parsed rather than assumed.
        #[arg(long)]
        partition_checkpoint: String,

        /// Directory the four `memprobe_*` charts are written into, i.e. a run's `gens/<n>`.
        #[arg(long)]
        output: String,

        /// Pinned windows drawn per split for the gap. Both splits get the same count through
        /// the same constructor at the same seed, so the two draws differ only in split range.
        #[arg(long, default_value_t = 4096)]
        gap_windows: usize,

        /// Windows sampled from EACH arm of the one-repetition contrast. 1024 windows at the
        /// deployed context pools 2.1M bars an arm, inside the module's 3M-bar cap.
        #[arg(long, default_value_t = 1024)]
        arm_windows: usize,

        /// Conditioning context. MUST be the context the DEPLOYED ramp stage tiles at, which is
        /// checked against the rebuilt partition: the arms would otherwise be scored at a
        /// context the run never trained them at.
        #[arg(long, default_value_t = trading_bot_0::torch::world_model::BAR_MAX_CONTEXT)]
        context: i64,

        /// Evaluation batch, in windows.
        #[arg(long, default_value_t = 8)]
        batch_size: usize,

        /// The run's REALIZED per-stage batch, one comma-separated entry per ramp stage.
        /// `PassPlan` normalizes its token weights, so `24,24,24` and `1,1,1` give a
        /// byte-identical partition; the realized figure is the default because it is what the
        /// metadata records.
        #[arg(
            long,
            value_parser = trading_bot_0::torch::train::mem_probe::parse_batch_ramp,
            default_value = "24,24,24"
        )]
        batch_ramp: [usize; 3],

        /// The run's `train_seed`. NOT `EVAL_WINDOW_SEED`: the partition being reconstructed is
        /// the TRAINING sampler's.
        #[arg(long, default_value_t = 0x5EED)]
        train_seed: u64,

        #[arg(long, default_value_t = trading_bot_0::data::ingest::bars_dir().to_string_lossy().into_owned())]
        data_dir: String,

        #[arg(long, default_value_t = 300)]
        resolution_secs: u32,

        #[arg(long, default_value_t = trading_bot_0::torch::dataset::DEFAULT_MIN_BARS)]
        min_bars: usize,

        /// Pin the two split instants as `<b0>,<b1>` epoch millis. Defaults to the campaign pin;
        /// it MUST be the run's own, or the arms are drawn from a different train region.
        #[arg(long, value_parser = parse_split_bounds)]
        split_bounds: Option<(i64, i64)>,

        /// Re-derive the split instants from the current corpus. Diagnostic use only.
        #[arg(long, default_value_t = false)]
        derive_split_bounds: bool,

        /// Liquidity floor the run used; 0 uses every file on disk.
        #[arg(long, default_value_t = 0.0)]
        min_dollar_volume: f64,
    },
    /// Does the predictor have exploitable DIRECTIONAL skill, with NO trading policy anywhere
    /// in the measurement?
    ///
    /// Scores `sign(E[r | strictly past bars])` against `sign(r)` on pinned held-out windows —
    /// the same-bar range marginalized out, so no lookahead — and reports the statistics that
    /// survive the up/down class imbalance and the panel's cross-sectional heteroskedasticity:
    /// the full 2x2 against all three constant baselines, the information coefficient split
    /// pooled / within-name / standardized-within-symbol-month, the AUC, and the decisive curve
    /// of accuracy and edge against the model's own confidence. Every interval is a block
    /// bootstrap over `(symbol, calendar month)`.
    ///
    /// Reports a break-even cost per traded bar as SCREENING ARITHMETIC. It builds no policy
    /// and runs no backtest; `pretrain-trade` and `pretrain-calibration` own those.
    ///
    /// The corpus flags MUST match the run's, or the pinned windows are drawn from a different
    /// symbol set and the numbers describe different data.
    PretrainSkill {
        /// Checkpoint to audit, e.g. `weights/pretrain_best.ot`.
        #[arg(long)]
        weights: String,

        /// Directory the `pretrain_skill_profile.report.bin` chart is written into.
        #[arg(long)]
        output: String,

        /// Held-out split to score. `test` is scored once and is the number that counts.
        #[arg(long, value_enum, default_value_t = PlannerDataSplit::Validation)]
        split: PlannerDataSplit,

        /// Pinned windows to draw. The audit scores the first `trade_bench::TRADE_WINDOWS` of
        /// them, which is exactly the prefix `pretrain-trade` and `pretrain-calibration`
        /// measure. The count is part of the pin, so it must EQUAL the run's
        /// `--validation-windows` for the windows to be the run's own.
        #[arg(long, default_value_t = 4096)]
        windows: usize,

        /// Conditioning context. Must match the context the compared reads were taken at,
        /// which for a run's own diagnostic series is the fixed ramp-start context.
        #[arg(long, default_value_t = trading_bot_0::torch::train::pretrain::BAR_CONTEXT_RAMP_START)]
        context: i64,

        /// Evaluation batch, in windows.
        #[arg(long, default_value_t = 8)]
        batch_size: usize,

        #[arg(long, default_value_t = trading_bot_0::data::ingest::bars_dir().to_string_lossy().into_owned())]
        data_dir: String,

        #[arg(long, default_value_t = 300)]
        resolution_secs: u32,

        #[arg(long, default_value_t = trading_bot_0::torch::dataset::DEFAULT_MIN_BARS)]
        min_bars: usize,

        /// Pin the two split instants as `<b0>,<b1>` epoch millis. Defaults to the campaign pin
        /// `ingest::PINNED_SPLIT_BOUNDS`.
        #[arg(long, value_parser = parse_split_bounds)]
        split_bounds: Option<(i64, i64)>,

        /// Re-derive the split instants from the current corpus. Diagnostic use only.
        #[arg(long, default_value_t = false)]
        derive_split_bounds: bool,

        /// Liquidity floor the run used; 0 uses every file on disk.
        #[arg(long, default_value_t = 0.0)]
        min_dollar_volume: f64,
    },
    /// Compare two pretraining runs PAIRED on their identical pinned evaluation windows.
    ///
    /// Comparing two absolute levels has a minimum detectable effect of ~0.41 nats, because
    /// the validation split holds only ~4 non-overlapping time slots and the market-common
    /// regime term does not average down. Differencing the same windows takes that to
    /// 0.04-0.09. Each argument is the `pretrain_best.windows.json` written beside a
    /// promoted checkpoint.
    PretrainCompare {
        /// Baseline run's per-window vector.
        #[arg(long)]
        baseline: String,
        /// Candidate run's per-window vector.
        #[arg(long)]
        candidate: String,
    },
    TrainPlanner {
        #[arg(long, default_value = "weights/pretrain_best.ot")]
        world_model_weights: String,

        #[arg(long)]
        world_model_metadata: Option<String>,

        #[arg(long)]
        planner_weights: Option<String>,

        #[arg(long)]
        output: Option<String>,

        #[arg(long)]
        run: Option<String>,

        #[arg(long, default_value_t = 1_000)]
        updates: usize,

        #[arg(long, default_value_t = 100)]
        horizon: usize,

        #[arg(long, default_value_t = 100)]
        rollout_length: usize,

        #[arg(long, default_value_t = 128)]
        environments: usize,

        #[arg(long, default_value_t = 1280)]
        minibatch_size: usize,

        #[arg(long)]
        context_bars: Option<usize>,

        #[arg(long, value_delimiter = ',')]
        tickers: Option<Vec<String>>,

        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
    InferPlanner {
        #[arg(long, default_value = "weights/pretrain_best.ot")]
        world_model_weights: String,

        #[arg(long)]
        world_model_metadata: Option<String>,

        #[arg(long, default_value = "weights/planner.ot")]
        planner_weights: String,

        #[arg(long, default_value_t = 10)]
        episodes: usize,

        #[arg(long)]
        horizon: Option<usize>,

        #[arg(long, default_value_t = 100)]
        rollout_length: usize,

        #[arg(long)]
        context_bars: Option<usize>,

        #[arg(long, value_delimiter = ',')]
        tickers: Option<Vec<String>>,

        #[arg(long, value_enum, default_value_t = PlannerDataSplit::Test)]
        split: PlannerDataSplit,

        #[arg(long)]
        run: Option<String>,
    },
    Infer {
        #[arg(short, long, default_value = "weights/ppo_ep1000.ot")]
        weights: String,

        #[arg(short, long, default_value_t = 10)]
        episodes: usize,

        #[arg(short, long, default_value_t = false)]
        deterministic: bool,

        #[arg(short, long, default_value_t = 1.0)]
        temperature: f64,

        #[arg(long, value_delimiter = ',')]
        tickers: Option<Vec<String>>,

        #[arg(short, long, default_value_t = true)]
        random_start: bool,

        #[arg(long, value_enum, default_value_t = ModelVariant::UniformStream)]
        model_size: ModelVariant,

        #[arg(long)]
        run: Option<String>,
    },
    Paper {
        #[arg(short, long, default_value = "weights/ppo_ep1000.ot")]
        weights: String,

        /// Dedicated IBKR paper account to trade. Existing positions are rejected.
        #[arg(long)]
        account: String,

        /// Defaults to the deepest corpus histories, resolved when the command runs.
        #[arg(short, long, value_delimiter = ',')]
        symbols: Vec<String>,

        #[arg(short, long, default_value_t = 5)]
        interval: u64,

        #[arg(short, long, default_value_t = 500)]
        max_steps: usize,

        #[arg(long, value_enum, default_value_t = ModelVariant::UniformStream)]
        model_size: ModelVariant,
    },
    /// Download the Polygon corpus: measure the universe, then pull aggregates into
    /// `long_data/bars`.
    Ingest {
        /// Liquidity floor for corpus membership, in median dollars traded per session.
        ///
        /// A floor rather than a rank cutoff: "the top 3000 names" is an arbitrary number, whereas
        /// a dollar-volume floor states the quality threshold below which five-minute bars stop
        /// carrying intra-bar structure.
        #[arg(long, default_value_t = trading_bot_0::data::ingest::MIN_DOLLAR_VOLUME)]
        min_dollar_volume: f64,

        #[arg(long, default_value = "5min")]
        resolution: String,

        /// Years of history to request. Raising it is itself the instruction to deepen the corpus;
        /// there is no separate flag and none is needed.
        ///
        /// Why that is safe to say now and was not before. The currency test
        /// (`ingest::covered`, formerly `current_file`) checks the RIGHT edge of an existing file
        /// and deliberately not its left, because a file's left edge is set by the plan's rolling
        /// window, the symbol's listing date and the splice repair applied to reused tickers — so a
        /// left-edge test calls every legitimately-late series incomplete and rewrites it on every
        /// pass, forever, restoring the very splices the repair cut out. The consequence was that
        /// raising this number downloaded NOTHING: every five-year file still passed the right-edge
        /// test.
        ///
        /// `long_data/bars/.ingest_manifest.jsonl` closes that without reintroducing the trap. It
        /// records, per symbol, the absolute window-start INSTANT a completed download requested,
        /// and the skip rule compares requested intent against recorded intent. Recorded intent has
        /// no fixed point — it is written only once the download that satisfied it finished — so
        /// asking for a deeper window refetches exactly once and then skips. The instant is the key
        /// rather than this number because the vendor window ROLLS: `--years 10` resolves to a
        /// different start every day, so a recorded `10` would compare equal to itself forever and
        /// silently skip work while reading as if it had checked something.
        ///
        /// Resumability falls out of the same rule: stop the process and rerun the identical
        /// command, and it continues rather than restarting. Files are installed by rename, so a
        /// hard kill leaves the old complete file or the new one and never a partial one.
        #[arg(long, default_value_t = 5)]
        years: u32,

        #[arg(long, default_value_t = trading_bot_0::data::polygon::DEFAULT_CONCURRENCY)]
        concurrency: usize,

        /// Re-measure liquidity from the vendor instead of reusing the cached ranking.
        #[arg(long)]
        refresh_universe: bool,

        /// RFC 3339 instant the ranking sessions must precede, so universe membership carries no
        /// held-out information. Defaults to the corpus's own `train | val` boundary.
        #[arg(long, value_parser = trading_bot_0::data::ingest::parse_train_end)]
        train_end: Option<chrono::DateTime<chrono::Utc>>,

        /// Bars a symbol needs to count toward that boundary. Must match `pretrain --min-bars`.
        #[arg(long, default_value_t = trading_bot_0::torch::dataset::DEFAULT_MIN_BARS)]
        min_bars: usize,

        /// Measure and report the universe, then stop before downloading anything.
        #[arg(long)]
        universe_only: bool,

        #[arg(long)]
        force: bool,

        #[arg(long, default_value_t = false)]
        daily: bool,
    },
    /// Fill the pre-2021 regime gap: pull free deep daily history into `long_data/bars`.
    ///
    /// The Polygon plan is capped at a rolling five-year window, so the intraday corpus contains
    /// no 2000 and no 2008. This pulls decades of daily bars from a free source instead.
    DeepDaily {
        /// Liquidity floor the cached ranking is filtered by, matching `ingest`.
        #[arg(long, default_value_t = trading_bot_0::data::ingest::MIN_DOLLAR_VOLUME)]
        min_dollar_volume: f64,

        /// Take only the first N symbols of that ranking; 0 takes all of them.
        #[arg(long, default_value_t = 0)]
        limit: usize,

        #[arg(long, default_value_t = trading_bot_0::data::deep_daily::DEFAULT_CONCURRENCY)]
        concurrency: usize,

        /// Rewrite symbols whose daily file already reaches past the Polygon floor.
        #[arg(long)]
        force: bool,

        /// Fetch and audit, but write nothing.
        #[arg(long)]
        dry_run: bool,
    },
}

fn main() {
    // Must precede the runtime: `set_var` is only sound while the process is single-threaded.
    trading_bot_0::data::load_dotenv();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed building the tokio runtime")
        .block_on(run());
}

async fn run() {
    println!("{}", "Start".green());

    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Genetic {
            family,
            run,
            generations,
            population,
            survivor_ratio,
            train_tickers,
            validation_tickers,
            test_tickers,
            heavy_report_every,
            seed,
            skip_additional_downloads,
            mutation_entropy,
        }) => {
            genetic::run(genetic::GeneticArgs {
                family: *family,
                run: run.clone(),
                generations: *generations,
                population: *population,
                survivor_ratio: *survivor_ratio,
                train_tickers: *train_tickers,
                validation_tickers: *validation_tickers,
                test_tickers: *test_tickers,
                heavy_report_every: *heavy_report_every,
                seed: *seed,
                skip_additional_downloads: *skip_additional_downloads,
                mutation_entropy: *mutation_entropy,
            })
            .expect("genetic training failed");
        }
        Some(Commands::Train {
            weights,
            model_size,
            run,
            seed,
        }) => {
            torch::train::train(weights.as_deref(), (*model_size).into(), run.clone(), *seed)
                .await
                .expect("PPO training failed");
        }
        Some(Commands::Pretrain {
            weights,
            run,
            epochs,
            steps,
            batch_size,
            seed,
            data_dir,
            resolution_secs,
            min_bars,
            auxiliary_resolutions,
            support_samples,
            scoring,
            dyn_horizon,
            lambda_dyn,
            lambda_kl,
            validation_windows,
            diagnostic_context,
            snapshot_windows,
            snapshot_samples,
            validate_every,
            checkpoint_every,
            log_every,
            split_bounds,
            derive_split_bounds,
            supports,
            freeze_supports,
            min_dollar_volume,
            lambda_growth,
            exact_batch,
            lr_plateau_fraction,
        }) => {
            let args = PretrainArgs {
                weights: weights.clone(),
                run: run.clone(),
                epochs: *epochs,
                steps: *steps,
                batch_size: *batch_size,
                seed: *seed,
                data_dir: data_dir.clone(),
                resolution_secs: *resolution_secs,
                min_bars: *min_bars,
                auxiliary_resolutions: auxiliary_resolutions.clone(),
                support_samples: *support_samples,
                scoring: *scoring,
                dyn_horizon: *dyn_horizon,
                lambda_dyn: *lambda_dyn,
                lambda_kl: *lambda_kl,
                validation_windows: *validation_windows,
                diagnostic_context: *diagnostic_context,
                snapshot_windows: *snapshot_windows,
                snapshot_samples: *snapshot_samples,
                validate_every: *validate_every,
                checkpoint_every: *checkpoint_every,
                log_every: *log_every,
                split_bounds: *split_bounds,
                derive_split_bounds: *derive_split_bounds,
                supports: supports.clone(),
                freeze_supports: *freeze_supports,
                min_dollar_volume: *min_dollar_volume,
                lambda_growth: *lambda_growth,
                exact_batch: *exact_batch,
                lr_plateau_fraction: *lr_plateau_fraction,
            };
            tokio::task::spawn_blocking(move || torch::train::pretrain(args))
                .await
                .expect("pretraining task panicked")
                .expect("pretraining failed");
        }
        Some(Commands::PretrainCandles {
            weights,
            output,
            windows,
            samples,
            context,
            step,
            data_dir,
            resolution_secs,
            min_bars,
            split_bounds,
            derive_split_bounds,
            min_dollar_volume,
        }) => {
            let args = torch::train::CandleArgs {
                weights: weights.clone(),
                output: output.clone(),
                windows: *windows,
                samples: *samples,
                context: *context,
                step: *step,
                corpus: torch::train::CorpusFlags {
                    data_dir: data_dir.clone(),
                    resolution_secs: *resolution_secs,
                    min_bars: *min_bars,
                    split_bounds: *split_bounds,
                    derive_split_bounds: *derive_split_bounds,
                    min_dollar_volume: *min_dollar_volume,
                },
            };
            tokio::task::spawn_blocking(move || torch::train::pretrain_candles(args))
                .await
                .expect("candle rollout task panicked")
                .expect("candle rollout failed");
        }
        Some(Commands::PretrainTrade {
            weights,
            output,
            split,
            windows,
            context,
            batch_size,
            data_dir,
            resolution_secs,
            min_bars,
            split_bounds,
            derive_split_bounds,
            min_dollar_volume,
        }) => {
            let args = torch::train::TradeArgs {
                weights: weights.clone(),
                output: output.clone(),
                split: split.split(),
                windows: *windows,
                context: *context,
                batch_size: *batch_size,
                corpus: torch::train::CorpusFlags {
                    data_dir: data_dir.clone(),
                    resolution_secs: *resolution_secs,
                    min_bars: *min_bars,
                    split_bounds: *split_bounds,
                    derive_split_bounds: *derive_split_bounds,
                    min_dollar_volume: *min_dollar_volume,
                },
            };
            tokio::task::spawn_blocking(move || torch::train::pretrain_trade(args))
                .await
                .expect("trade bench task panicked")
                .expect("trade bench failed");
        }
        Some(Commands::PretrainSkill {
            weights,
            output,
            split,
            windows,
            context,
            batch_size,
            data_dir,
            resolution_secs,
            min_bars,
            split_bounds,
            derive_split_bounds,
            min_dollar_volume,
        }) => {
            let args = torch::train::skill::SkillArgs {
                weights: weights.clone(),
                output: output.clone(),
                split: split.split(),
                windows: *windows,
                context: *context,
                batch_size: *batch_size,
                corpus: torch::train::CorpusFlags {
                    data_dir: data_dir.clone(),
                    resolution_secs: *resolution_secs,
                    min_bars: *min_bars,
                    split_bounds: *split_bounds,
                    derive_split_bounds: *derive_split_bounds,
                    min_dollar_volume: *min_dollar_volume,
                },
            };
            tokio::task::spawn_blocking(move || torch::train::skill::pretrain_skill(args))
                .await
                .expect("skill audit task panicked")
                .expect("skill audit failed");
        }
        Some(Commands::PretrainHorizon {
            weights,
            output,
            data_dir,
            resolution_secs,
            split_bounds,
            max_symbols,
            max_instants,
            cost_bps,
            capital_usd,
            gross_cap,
            samples,
            replicates,
            seed,
            cpu,
            label,
        }) => {
            let args = torch::train::horizon::HorizonArgs {
                bars_dir: std::path::PathBuf::from(data_dir),
                checkpoint: std::path::PathBuf::from(weights),
                gens_dir: std::path::PathBuf::from(output),
                res_secs: *resolution_secs,
                device: if *cpu {
                    tch::Device::Cpu
                } else {
                    tch::Device::cuda_if_available()
                },
                split_bounds: split_bounds.unwrap_or(trading_bot_0::data::ingest::PINNED_SPLIT_BOUNDS),
                max_symbols: *max_symbols,
                max_instants: *max_instants,
                cost_bps: *cost_bps,
                capital_usd: *capital_usd,
                gross_cap: *gross_cap,
                samples: *samples,
                replicates: *replicates,
                seed: *seed,
                label: label.clone(),
            };
            let frontier = tokio::task::spawn_blocking(move || {
                torch::train::horizon::run_horizon_sweep(&args)
            })
            .await
            .expect("horizon sweep task panicked")
            .expect("horizon sweep failed");
            print!("{}", frontier.table());
        }
        Some(Commands::PretrainCalibration {
            checkpoints,
            output,
            split,
            windows,
            fit_windows,
            trade_windows,
            dry_run,
            context,
            batch_size,
            data_dir,
            resolution_secs,
            min_bars,
            split_bounds,
            derive_split_bounds,
            min_dollar_volume,
            restrict_symbols,
        }) => {
            let args = torch::train::CalibrationArgs {
                checkpoints: checkpoints.clone(),
                output: output.clone(),
                split: split.split(),
                windows: *windows,
                fit_windows: *fit_windows,
                trade_windows: *trade_windows,
                dry_run: *dry_run,
                context: *context,
                batch_size: *batch_size,
                corpus: torch::train::CorpusFlags {
                    data_dir: data_dir.clone(),
                    resolution_secs: *resolution_secs,
                    min_bars: *min_bars,
                    split_bounds: *split_bounds,
                    derive_split_bounds: *derive_split_bounds,
                    min_dollar_volume: *min_dollar_volume,
                },
                restrict_symbols: restrict_symbols.clone(),
            };
            tokio::task::spawn_blocking(move || torch::train::pretrain_calibration(args))
                .await
                .expect("calibration task panicked")
                .expect("mean calibration failed");
        }
        Some(Commands::BarSupportsMoments {
            supports,
            output_supports,
            output,
            samples,
            seed,
            mass_tolerance,
            data_dir,
            resolution_secs,
            min_bars,
            split_bounds,
            derive_split_bounds,
            min_dollar_volume,
        }) => {
            let args = torch::train::support_moments::SupportMomentsArgs {
                supports: supports.clone(),
                output_supports: output_supports.clone(),
                output: output.clone(),
                samples: *samples,
                seed: *seed,
                mass_tolerance: *mass_tolerance,
                corpus: torch::train::CorpusFlags {
                    data_dir: data_dir.clone(),
                    resolution_secs: *resolution_secs,
                    min_bars: *min_bars,
                    split_bounds: *split_bounds,
                    derive_split_bounds: *derive_split_bounds,
                    min_dollar_volume: *min_dollar_volume,
                },
            };
            tokio::task::spawn_blocking(move || {
                torch::train::support_moments::fit_support_moments(args)
            })
            .await
            .expect("support moments task panicked")
            .expect("support moments pass failed");
        }
        Some(Commands::BarFamily {
            supports,
            output,
            samples,
            seed,
            k_min,
            k_max,
            atom_tolerance,
            data_dir,
            resolution_secs,
            min_bars,
            split_bounds,
            derive_split_bounds,
            min_dollar_volume,
        }) => {
            let args = torch::train::bar_family::BarFamilyArgs {
                supports: supports.clone(),
                output: output.clone(),
                samples: *samples,
                seed: *seed,
                k_min: *k_min,
                k_max: *k_max,
                atom_tolerance: *atom_tolerance,
                corpus: torch::train::CorpusFlags {
                    data_dir: data_dir.clone(),
                    resolution_secs: *resolution_secs,
                    min_bars: *min_bars,
                    split_bounds: *split_bounds,
                    derive_split_bounds: *derive_split_bounds,
                    min_dollar_volume: *min_dollar_volume,
                },
            };
            tokio::task::spawn_blocking(move || torch::train::bar_family::fit_bar_families(args))
                .await
                .expect("bar family task panicked")
                .expect("bar family fit failed");
        }
        Some(Commands::BarSplitSeams {
            supports,
            cross_check_supports,
            output,
            samples,
            seed,
            data_dir,
            resolution_secs,
            min_bars,
            split_bounds,
            derive_split_bounds,
            min_dollar_volume,
        }) => {
            let args = torch::train::split_seams::SplitSeamArgs {
                supports: supports.clone(),
                cross_check_supports: cross_check_supports.clone(),
                output: output.clone(),
                samples: *samples,
                seed: *seed,
                corpus: torch::train::CorpusFlags {
                    data_dir: data_dir.clone(),
                    resolution_secs: *resolution_secs,
                    min_bars: *min_bars,
                    split_bounds: *split_bounds,
                    derive_split_bounds: *derive_split_bounds,
                    min_dollar_volume: *min_dollar_volume,
                },
            };
            tokio::task::spawn_blocking(move || {
                torch::train::split_seams::audit_split_seams(args)
            })
            .await
            .expect("split seam audit task panicked")
            .expect("split seam audit failed");
        }
        Some(Commands::PretrainMemProbe {
            checkpoints,
            partition_checkpoint,
            output,
            gap_windows,
            arm_windows,
            context,
            batch_size,
            batch_ramp,
            train_seed,
            data_dir,
            resolution_secs,
            min_bars,
            split_bounds,
            derive_split_bounds,
            min_dollar_volume,
        }) => {
            let args = torch::train::mem_probe::MemProbeArgs {
                checkpoints: checkpoints.clone(),
                partition_checkpoint: partition_checkpoint.clone(),
                output: output.clone(),
                gap_windows: *gap_windows,
                arm_windows: *arm_windows,
                context: *context,
                batch_size: *batch_size,
                batch_ramp: *batch_ramp,
                train_seed: *train_seed,
                corpus: torch::train::CorpusFlags {
                    data_dir: data_dir.clone(),
                    resolution_secs: *resolution_secs,
                    min_bars: *min_bars,
                    split_bounds: *split_bounds,
                    derive_split_bounds: *derive_split_bounds,
                    min_dollar_volume: *min_dollar_volume,
                },
            };
            tokio::task::spawn_blocking(move || torch::train::mem_probe::mem_probe(args))
                .await
                .expect("memorization probe task panicked")
                .expect("memorization probe failed");
        }
        Some(Commands::PretrainCompare {
            baseline,
            candidate,
        }) => {
            let comparison = torch::train::pretrain_stats::compare_runs(
                std::path::Path::new(baseline),
                std::path::Path::new(candidate),
            )
            .expect("paired comparison failed");
            print!("{comparison}");
        }
        Some(Commands::TrainPlanner {
            world_model_weights,
            world_model_metadata,
            planner_weights,
            output,
            run,
            updates,
            horizon,
            rollout_length,
            environments,
            minibatch_size,
            context_bars,
            tickers,
            seed,
        }) => {
            let args = torch::planner::TrainPlannerArgs {
                world_model_weights: world_model_weights.clone(),
                world_model_metadata: world_model_metadata.clone(),
                planner_weights: planner_weights.clone(),
                output: output.clone().unwrap_or_default(),
                run: run.clone(),
                updates: *updates,
                horizon: *horizon,
                rollout_length: *rollout_length,
                environments: *environments,
                minibatch_size: *minibatch_size,
                context_bars: *context_bars,
                tickers: tickers.clone(),
                seed: *seed,
            };
            tokio::task::spawn_blocking(move || torch::planner::train_planner(args))
                .await
                .expect("planner training task panicked")
                .expect("planner training failed");
        }
        Some(Commands::InferPlanner {
            world_model_weights,
            world_model_metadata,
            planner_weights,
            episodes,
            horizon,
            rollout_length,
            context_bars,
            tickers,
            split,
            run,
        }) => {
            let destination = RunDir::create_fresh(RUNS_PATH, run.as_deref())
                .expect("failed to create planner inference run dir");
            let args = torch::planner::InferPlannerArgs {
                world_model_weights: world_model_weights.clone(),
                world_model_metadata: world_model_metadata.clone(),
                planner_weights: planner_weights.clone(),
                episodes: *episodes,
                horizon: *horizon,
                rollout_length: *rollout_length,
                context_bars: *context_bars,
                tickers: tickers.clone(),
                split: *split,
                report_root: Some(destination.root),
            };
            tokio::task::spawn_blocking(move || torch::planner::infer_planner(args))
                .await
                .expect("planner inference task panicked")
                .expect("planner inference failed");
        }
        Some(Commands::Infer {
            weights,
            episodes,
            deterministic,
            temperature,
            tickers,
            random_start,
            model_size,
            run,
        }) => {
            let run_dir = RunDir::create_fresh(RUNS_PATH, run.as_deref())
                .expect("failed to create inference run dir");
            let output_dir = run_dir.root.join("inference");
            torch::infer::run_inference(
                weights,
                *episodes,
                *deterministic,
                *temperature,
                tickers.clone(),
                *random_start,
                *model_size,
                output_dir,
            )
            .expect("inference failed");
        }
        Some(Commands::Paper {
            weights,
            account,
            symbols,
            interval,
            max_steps,
            model_size,
        }) => {
            let symbols = if symbols.is_empty() {
                default_paper_symbols()
            } else {
                symbols.clone()
            };
            torch::infer::run_ibkr_paper_trading(
                weights,
                account.clone(),
                symbols,
                *interval,
                *max_steps,
                *model_size,
            )
            .expect("paper trading failed");
        }
        Some(Commands::Ingest {
            min_dollar_volume,
            resolution,
            years,
            concurrency,
            refresh_universe,
            train_end,
            min_bars,
            universe_only,
            force,
            daily,
        }) => {
            trading_bot_0::data::ingest::run(trading_bot_0::data::ingest::IngestArgs {
                min_dollar_volume: *min_dollar_volume,
                resolution: resolution.clone(),
                years: *years,
                concurrency: *concurrency,
                refresh_universe: *refresh_universe,
                train_end: *train_end,
                min_bars: *min_bars,
                universe_only: *universe_only,
                force: *force,
                daily: *daily,
            })
            .await
            .expect("ingest failed");
        }
        Some(Commands::DeepDaily {
            min_dollar_volume,
            limit,
            concurrency,
            force,
            dry_run,
        }) => {
            trading_bot_0::data::deep_daily::run(trading_bot_0::data::deep_daily::DeepDailyArgs {
                min_dollar_volume: *min_dollar_volume,
                limit: *limit,
                concurrency: *concurrency,
                force: *force,
                dry_run: *dry_run,
            })
            .await
            .expect("deep daily ingest failed");
        }
        None => {
            torch::train::train(None, ModelVariant::UniformStream, None, 20260811)
                .await
                .expect("PPO training failed");
        }
    }

    println!("{}", "End".green())
}

#[cfg(test)]
mod tests {
    use super::{default_paper_symbols, Cli, Commands, PlannerDataSplit, StreamingModelVariant};
    use clap::Parser;
    use trading_bot_0::torch::model::ModelVariant;

    #[test]
    fn train_defaults_are_executable_contracts() {
        let train = Cli::try_parse_from(["trading_bot", "train"]).expect("train should parse");
        assert!(matches!(
            train.command,
            Some(Commands::Train {
                model_size: StreamingModelVariant::UniformStream,
                ..
            })
        ));
    }

    /// Pretrain's defaults are load-bearing: the seed makes runs reproducible, the
    /// batch size sizes every ramp stage, and the step count must stay corpus-derived.
    #[test]
    fn pretrain_defaults_are_executable_contracts() {
        let pretrain =
            Cli::try_parse_from(["trading_bot", "pretrain"]).expect("pretrain should parse");
        let Some(Commands::Pretrain {
            epochs,
            steps,
            batch_size,
            seed,
            resolution_secs,
            dyn_horizon,
            lambda_dyn,
            lambda_kl,
            lambda_growth,
            validation_windows,
            diagnostic_context,
            data_dir,
            lr_plateau_fraction,
            ..
        }) = pretrain.command
        else {
            panic!("pretrain subcommand should parse as Pretrain");
        };
        // One, not three. `bardist_v2` ran at 3 because THIS default was 3 at that run's
        // commit (a0ff3b29 changed it from 1 to 3), so three passes were deliberate rather
        // than a bug — and deliberate is not correct. The corpus carries ~1.0M effective
        // observations against 366M nominal bars and 31.8M parameters; passes above one
        // re-present the same 1,031 sessions and add no market-factor realizations. The
        // default is the recipe, so it is asserted here.
        assert_eq!(epochs, 1);
        assert_eq!(steps, None);
        assert_eq!(batch_size, 24);
        assert_eq!(seed, 0x5EED);
        // The default must stay 0.40: every persisted comparison in `training/runs` was
        // produced under it, so a run that does not pass the flag has to reproduce that
        // schedule exactly.
        assert_eq!(
            lr_plateau_fraction,
            trading_bot_0::torch::train::pretrain::LR_PLATEAU_FRACTION
        );
        assert_eq!(lr_plateau_fraction, 0.40);
        assert_eq!(resolution_secs, 300);
        assert_eq!(dyn_horizon, 4);
        // 1.0 is the NextLat reference's `lambda_mse` (arXiv 2511.05963,
        // `defaults.yaml`), and under `next_lat_loss`'s `Reduction::Mean` over
        // `[B, T, BAR_MODEL_DIM]` it is width-independent, so it stays 1.0 if
        // `BAR_MODEL_DIM` ever moves.
        //
        // This assertion previously read `1.0 / BAR_MODEL_DIM`. That was the
        // compensation for a summed feature axis, and the sum was REVERTED at
        // `pretrain.rs:7859-7862` precisely because it made the knob width-dependent
        // and took 62% of the objective while `nll` rose 16.34 -> 17.19 over 4000
        // steps. The compensation is therefore obsolete: keeping it would have pinned
        // the default at 1/512 of the reference under a mean reduction.
        assert_eq!(lambda_dyn, 1.0);
        assert_eq!(lambda_kl, 1.0);
        assert_eq!(validation_windows, 4096);
        assert_eq!(
            diagnostic_context,
            trading_bot_0::torch::train::pretrain::BAR_CONTEXT_RAMP_START
        );
        // Sized on gradient norm, not on objective share, and hard-coded rather than swept.
        assert_eq!(
            lambda_growth,
            trading_bot_0::torch::train::growth::LAMBDA_GROWTH
        );
        assert!(
            lambda_growth > 0.0,
            "the default arm must TRAIN the growth term; 0.0 is the ablation control"
        );
        assert!(data_dir.ends_with("bars"), "{data_dir}");
    }

    /// `pretrain-calibration`'s defaults are what every published economic number in
    /// `training/runs` was measured under, so they are asserted rather than trusted.
    ///
    /// `--trade-windows` in particular: the traded prefix used to be
    /// `trade_bench::TRADE_WINDOWS` read in place, and it became a flag so a one-shot `test`
    /// read can spend the windows it has. The whole justification for that change is that
    /// omitting the flag reproduces the constant EXACTLY, which is this assertion and not a
    /// comment. `--dry-run` defaults off for the same reason: a flag that skipped the scoring
    /// by default would silently turn every existing invocation into a no-op.
    #[test]
    fn pretrain_calibration_defaults_reproduce_the_published_prefix() {
        let cli = Cli::try_parse_from([
            "trading_bot",
            "pretrain-calibration",
            "--checkpoint",
            "weights/pretrain_best.ot@1",
            "--output",
            "gens/0",
        ])
        .expect("pretrain-calibration should parse");
        let Some(Commands::PretrainCalibration {
            split,
            windows,
            fit_windows,
            trade_windows,
            dry_run,
            context,
            ..
        }) = cli.command
        else {
            panic!("unexpected subcommand");
        };
        assert_eq!(
            trade_windows,
            trading_bot_0::torch::train::trade_bench::TRADE_WINDOWS
        );
        assert_eq!(trade_windows, 256);
        assert!(!dry_run, "the default invocation must still score");
        assert_eq!(windows, 4096);
        assert_eq!(fit_windows, 256);
        assert_eq!(split, PlannerDataSplit::Validation);
        assert_eq!(
            context,
            trading_bot_0::torch::train::pretrain::BAR_CONTEXT_RAMP_START
        );
    }

    /// The `test` split must be ADDRESSABLE from the command line and must reach
    /// `dataset::Split::Test`, because the campaign's one uncontaminated measurement is taken
    /// through exactly this path. A value-enum that parsed but mapped to the wrong split would
    /// score `Val` twice and look like it worked.
    #[test]
    fn the_test_split_is_reachable_from_the_calibration_command_line() {
        let cli = Cli::try_parse_from([
            "trading_bot",
            "pretrain-calibration",
            "--checkpoint",
            "weights/pretrain_best.ot@1",
            "--output",
            "gens/0",
            "--split",
            "test",
            "--trade-windows",
            "4096",
            "--windows",
            "16384",
            "--dry-run",
        ])
        .expect("--split test must parse");
        let Some(Commands::PretrainCalibration {
            split,
            trade_windows,
            windows,
            dry_run,
            ..
        }) = cli.command
        else {
            panic!("unexpected subcommand");
        };
        assert_eq!(split, PlannerDataSplit::Test);
        assert_eq!(
            split.split(),
            trading_bot_0::torch::dataset::Split::Test,
            "the value-enum must reach the dataset split the sampler ranges on"
        );
        assert_eq!(trade_windows, 4096);
        assert_eq!(windows, 16384);
        assert!(dry_run);
    }

    #[test]
    fn streaming_only_commands_reject_unimplemented_model_families() {
        for model in ["base", "ablation-small"] {
            let Err(error) = Cli::try_parse_from(["trading_bot", "train", "--model-size", model])
            else {
                panic!("unsupported streaming model must fail during parsing");
            };
            let message = error.to_string();
            assert!(message.contains("invalid value"));
            assert!(message.contains("uniform-stream"));
        }
    }

    /// The planner loads whatever pretraining promoted; the two defaults must agree.
    #[test]
    fn planner_defaults_point_at_the_promoted_pretrain_checkpoint() {
        for command in ["train-planner", "infer-planner"] {
            let cli = Cli::try_parse_from(["trading_bot", command])
                .expect("planner subcommand should parse");
            let weights = match cli.command {
                Some(Commands::TrainPlanner {
                    world_model_weights,
                    ..
                })
                | Some(Commands::InferPlanner {
                    world_model_weights,
                    ..
                }) => world_model_weights,
                _ => panic!("unexpected subcommand"),
            };
            assert_eq!(weights, "weights/pretrain_best.ot");
        }
    }

    #[test]
    fn offline_inference_exposes_every_implemented_model_family() {
        for (name, expected) in [
            ("base", ModelVariant::Base),
            ("uniform-stream", ModelVariant::UniformStream),
            ("ablation-small", ModelVariant::AblationSmall),
        ] {
            let cli = Cli::try_parse_from(["trading_bot", "infer", "--model-size", name])
                .expect("implemented inference model should parse");
            let Some(Commands::Infer { model_size, .. }) = cli.command else {
                panic!("infer subcommand should parse as Infer");
            };
            assert_eq!(model_size, expected);
        }
    }

    #[test]
    fn paper_symbols_are_not_resolved_while_parsing_the_cli() {
        let cli = Cli::try_parse_from(["trading_bot", "paper", "--account", "DU123"])
            .expect("paper CLI should parse");
        let Some(Commands::Paper {
            symbols, interval, ..
        }) = cli.command
        else {
            panic!("paper subcommand should parse as Paper");
        };
        assert!(
            symbols.is_empty(),
            "the corpus must not be scanned to build the command tree"
        );
        assert_eq!(interval, 5);
    }

    #[test]
    fn paper_defaults_match_model_ticker_count() {
        assert_eq!(
            default_paper_symbols().len(),
            trading_bot_0::torch::constants::TICKERS_COUNT as usize
        );
    }

    #[test]
    fn paper_rejects_unsupported_temperature_sampling() {
        assert!(Cli::try_parse_from([
            "trading_bot",
            "paper",
            "--account",
            "DU123",
            "--temperature",
            "0.8",
        ])
        .is_err());
    }
}
