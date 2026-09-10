use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// The CausalPatch segment forecaster's chart bases, headline first. Each base answers one
/// question in one unit; the split into `_skill` / `_loss` / `_error` and the four-way
/// per-horizon family exists because a shared axis carrying a dimensionless ratio near 1
/// beside a σ-scaled level beside a rate near 0.5 renders all but one of them as flat lines.
pub const TIMEXER_SEGMENT_REPORT_BASES: &[&str] = &[
    "timexer_segment_skill",
    "timexer_segment_loss",
    "timexer_segment_error",
    "timexer_segment_calibration",
    // The train-versus-held-out NLL gap on its own axis, and the step-indexed transposes of
    // the per-horizon family. The horizon-indexed bases below are rewritten in place at every
    // evaluation, so they state a moment and cannot state a trajectory; these carry the
    // quantities a run-versus-run comparison turns on along the step axis, which is what makes
    // two runs comparable at MATCHED steps rather than at whichever step each happened to
    // write last.
    //
    // `_signal` is the decision panel: a cross-sectional model is adopted or rejected on
    // whether the per-horizon within-timestamp information coefficient grows or decays over
    // training, and it was previously the one quantity thrown away at every evaluation. The
    // rest exist to explain a move in it: `_gain` is the MSE-optimal amplitude `β̂`, whose
    // distance from 1 is exactly how much a rank-preserving forecast can lose on MSE;
    // `_best_scale` pairs the achieved close ratio with the one an amplitude fix would reach;
    // `_decomposition` splits the close gain into tilt, conditional signal and mis-scaling;
    // `_calibration` is per-horizon σ coverage, which the aggregate `timexer_segment_
    // calibration` above cannot show because every horizon contributes the same bar count to
    // it; `_pooled` is the pooled correlation, on its own base because it is a different
    // population from the cross-sectional IC; and `_population` is the count behind all of
    // them, so a thin draw reads as thin rather than as a statistic that moved.
    //
    // The retired `timexer_segment_horizon_steps_scaling` carried the mis-scaling cross term
    // alone; `_decomposition` carries it beside the two components it only means anything
    // against, in the same unit.
    "timexer_segment_generalization_gap",
    "timexer_segment_horizon",
    "timexer_segment_horizon_error",
    "timexer_segment_horizon_robust",
    "timexer_segment_horizon_rates",
    "timexer_segment_horizon_steps",
    "timexer_segment_horizon_steps_signal",
    "timexer_segment_horizon_steps_signal_error",
    "timexer_segment_horizon_steps_pooled",
    "timexer_segment_horizon_steps_population",
    "timexer_segment_horizon_steps_decomposition",
    "timexer_segment_horizon_steps_gain",
    "timexer_segment_horizon_steps_best_scale",
    "timexer_segment_horizon_steps_calibration",
    // The per-horizon weight the training objective actually applies. Constant within a run,
    // so it is horizon-indexed and not step-indexed; it belongs beside the horizon family
    // because it is the reason two horizons' curves are not comparable to each other.
    "timexer_segment_horizon_loss_weight",
    "timexer_segment_decomposition",
    "timexer_segment_signal",
    "timexer_segment_offset",
    "timexer_segment_tradable",
    "timexer_segment_tradable_rates",
    // In-run out-of-sample amplitude calibration of the conditional MEAN. `_calibration_gain`
    // carries the dimensionless gain: what was applied, what each population's own MSE-optimal
    // amplitude is, and the in-sample training comparand that says whether the amplitude error
    // is over-fitting or an objective defect. `_amplitude_calibration` carries what applying it
    // does to the MSE ratio, which is a different unit and therefore a different base - the two
    // must never share an axis. Horizon-indexed, beside the diagnostics that motivate them:
    // `timexer_segment_horizon_steps_gain` says how far the amplitude is from 1 and
    // `_best_scale` says what fixing it is worth, so a fitted gain that does not move the ratio
    // toward that bound is legible as a calibration that did not work.
    "timexer_segment_calibration_gain",
    "timexer_segment_amplitude_calibration",
    // The two moments `β̂` is a quotient of, as shares of the same persistence MSE. Its own
    // base and not a pair of series on `_calibration_gain`, because a share of an MSE and a
    // dimensionless gain are different units: a gain above 1 is a real under-amplitude only if
    // the `Var(f)` under it is large enough for correcting it to be worth anything, and that
    // comparison is unreadable on an axis scaled to a gain curve.
    "timexer_segment_calibration_moments",
    // Raw endpoint-cohort diagnostics use distinct bases from retired residual-log spreads.
    "timexer_segment_utility_payoff",
    "timexer_segment_utility_payoff_cost_0p5",
    "timexer_segment_utility_payoff_cost_1",
    "timexer_segment_utility_payoff_cost_2",
    "timexer_segment_utility_payoff_cost_5",
    "timexer_segment_utility_payoff_cost_10",
    "timexer_segment_utility_rate",
    "timexer_segment_utility_rate_cost_0p5",
    "timexer_segment_utility_rate_cost_1",
    "timexer_segment_utility_rate_cost_2",
    "timexer_segment_utility_rate_cost_5",
    "timexer_segment_utility_rate_cost_10",
    "timexer_segment_utility_breakeven",
    "timexer_segment_utility_gross_exposure",
    "timexer_segment_utility_net_exposure",
    "timexer_segment_utility_active_fraction",
    "timexer_segment_utility_turnover",
    "timexer_segment_utility_payoff_std",
    "timexer_segment_utility_worst_payoff",
    "timexer_segment_utility_census",
    "timexer_segment_account_value",
    "timexer_segment_account_costs",
    "timexer_segment_account_risk",
    "timexer_segment_account_activity",
    "timexer_segment_account_daily_pnl",
    "timexer_segment_account_daily_return",
    "timexer_segment_account_monthly_pnl",
    "timexer_segment_account_monthly_return",
    "timexer_segment_account_summary_money",
    "timexer_segment_account_summary_risk",
    "timexer_segment_account_summary_census",
    "timexer_segment_account_timing",
    "timexer_segment_progress",
    "timexer_segment_recipe_scalars",
    // The rate each optimizer family actually stepped at, per step. Beside the recipe scalars
    // because both answer "what did the optimizer do", and on its own axis because a NorMuon
    // matrix rate, an AdamW dense rate and a 5x scalar-bank rate span an order of magnitude.
    "timexer_segment_lr_trajectory",
    "timexer_segment_timing",
    // One-time cost of everything a run pays before its first step. Its own base rather than
    // rows in `timexer_segment_timing`: a 10^5 ms constant on the 2-180 ms per-step axis
    // flattens every series that panel exists to show.
    "timexer_segment_startup",
    "timexer_segment_capture",
    "timexer_segment_candles",
    "timexer_segment_hardware",
    "timexer_segment_benchmark",
    "timexer_segment_benchmark_phases",
    "timexer_segment_benchmark_kernels",
    "timexer_segment_benchmark_kernel_roofline",
    "timexer_segment_benchmark_kernel_activations",
    "timexer_segment_fused_kernels",
    // The frozen-trunk latent probe. Three bases and not one, for the reason stated at the top
    // of this list: the decision quantity is a correlation near 0.05, the amplitude-neutral
    // comparison is an MSE ratio near 1.0, and the fit's conditioning spans decades. On one
    // axis the first two render as a flat line under the third.
    //
    // `_latent_probe` is the verdict panel: the trained head's within-timestamp IC, each
    // probe's, and the PAIRED difference beside the two pre-registered decision thresholds, so
    // the World A / World B call is read off the chart rather than asserted in prose.
    // `_latent_probe_ratio` carries the same comparison in MSE, achieved and oracle-rescaled on
    // both sides, because a probe compared to the head on raw MSE would be re-measuring the
    // known amplitude defect instead of information content. `_latent_probe_conditioning` is
    // the discount panel: a probe that only wins through an ill-conditioned inverse is not
    // evidence, and the fitted parameter count rides in each series label.
    "timexer_segment_latent_probe",
    "timexer_segment_latent_probe_ratio",
    "timexer_segment_latent_probe_conditioning",
    // Same family, different QUESTION, and therefore a fourth base rather than a fourth series:
    // the three above ask whether the latent beats the head at a fixed fit sample, this one asks
    // whether that answer is limited by information or by sample size. Its x axis is
    // coefficient-fit origins, not bars ahead, so it cannot share a panel with them.
    "timexer_segment_latent_probe_scaling",
    // And a fifth, for the third explanation the other four are blind to: how much of the
    // probe's edge is RECENCY rather than latent information. Its series are fit-block tranches
    // and its decisive line is a threshold ADJUSTMENT, not an IC, so it belongs beside the
    // verdict panel rather than inside it.
    "timexer_segment_latent_probe_recency",
    // The per-horizon predictable-information CEILING, beside the checkpoint's own IC and the
    // paired gap between them. ONE base and not three: a ceiling, a realized IC, their
    // difference and the standard error of that difference are all correlation coefficients
    // and all answer one question - how much within-timestamp IC is left at this horizon -
    // so putting them on separate axes would hide the only comparison the panel exists for.
    //
    // It is a property of the TARGETS, not of any model: an upper bound on what any causal
    // forecaster can reach, computed from the within-timestamp second moments of the 192
    // cumulative market-neutral targets under a martingale-residual assumption and a stated
    // one-bar ceiling. Nothing a run does can move it, which is what makes it the reference
    // every architecture decision is judged against rather than another run-dependent curve.
    "timexer_segment_information_ceiling",
    // How the market-neutral target's cross-sectional variance ACCUMULATES with the horizon,
    // in variance-ratio units, on every population the curve is measured over. It sits beside
    // the ceiling rather than inside it because a ratio near 0.35 on the ceiling's correlation
    // axis would be read as an IC of 0.35, which is two orders of magnitude wrong.
    //
    // The decoder's persistence anchor scales its predicted mean and scale by √h, which is the
    // correct growth for a random walk and the wrong one for anything else. This curve is the
    // measured replacement: it is a functional of the targets alone, costs no model, and once
    // it agrees across `training`, the calibration block and `held-out full` it may be frozen
    // into training as a data-derived scale.
    "timexer_segment_variance_ratio",
    // The MECHANISM behind that ratio: the measured lag-1 cross-sectional autocorrelation of
    // the per-bar returns beside the lag-1 autocorrelation an MA(1) residual would need to
    // produce the measured ratio. Correlation units, so its own axis. Two series that must
    // agree for the plateau to be readable as microstructure, and whose disagreement is the
    // finding that the reversal is spread over many lags instead.
    "timexer_segment_return_autocorrelation",
    // Per-coefficient predictive SNR and per-coefficient fitted amplitude gain in whichever
    // orthonormal horizon basis the objective is measured in - see
    // `timexer_segment::target_basis`. Both series are dimensionless ratios of the SAME
    // question, "where along the coefficient axis does the forecast carry signal and where is
    // its amplitude wrong", so they share one axis: `ρ̂²` is the share of a coefficient's
    // variance the forecast explains and `β̂` is the multiple its amplitude is off by, with 1
    // correct and below 1 over-amplified. In an orthonormal basis the amplitude defect is
    // DIAGONAL, so this is the one panel where the 3.8x over-amplification at h = 192 can be
    // attributed to a subspace rather than smeared across 192 near-duplicate rows.
    "timexer_segment_target_basis",
    // How much of its own training signal the run has actually consumed, per step: the share
    // of this arm's distinct supervised (absolute origin, horizon) outcomes that have received
    // at least one, two, four, eight and sixteen gradient passes, plus its epoch progress.
    //
    // Its own base because it is the only panel that is not a property of the model: it is a
    // property of the row pool, computed in closed form from the MEASURED per-outcome
    // multiplicity histogram of the rows the arm retained. It exists so the step at which a
    // run runs out of fresh outcomes is READ off a chart instead of recomputed by hand
    // afterwards - and so an arm whose held-out skill peaks somewhere other than its own
    // saturation step refutes the occupancy explanation on sight.
    //
    // Every series is a share of the same denominator, so one axis carries all of them; the
    // mean exposure COUNT is the epoch-progress series times the mean multiplicity the title
    // states, which keeps a count off an axis of fractions.
    "timexer_segment_supervision_occupancy",
    // The one panel that separates OVERFITTING from NON-STATIONARITY, which every other
    // explanation of a held-out peak hangs on. Chronological splits make every other
    // `held-out *` draw out-of-sample in ORIGIN IDENTITY and out-of-period in MARKET REGIME
    // simultaneously, and those two have opposite fixes - capacity/regularization/effective
    // sample size versus data recency/target definition/online adaptation. This base carries
    // an IN-PERIOD draw, cut from a purged hole inside the training span so it shares no
    // origin and no target bar with any surviving training row, against the out-of-period
    // draw at the same optimizer step.
    //
    // ONE base and not three: two ICs, their difference and all three standard errors are
    // correlation coefficients answering one question, so separate axes would hide the only
    // comparison the panel exists for. Reading rule, pre-registered in `reports.rs` before any
    // run existed: in-period improving while out-of-period collapses is non-stationarity; both
    // collapsing together is overfitting on a correlated-sample budget; in-period peaking less
    // severely is reported as a ratio and decides nothing on its own.
    "timexer_segment_temporal_generalization",
    // Where a NON-TRAINING pass's batch goes: the four synchronized spans - exposed host
    // loader wait, H2D upload, forward, and the nine fp64 per-timestamp scatter sums - that
    // partition one sampled batch of `ceiling-timexer-segment`'s accumulation loop.
    //
    // Deliberately NOT rows in `timexer_segment_timing`. That base is the TRAINING step's
    // panel: its x axis is the optimizer step and its values are an interval's means. This
    // one's x axis is the batch ordinal inside a single diagnostic pass, and the two clocks
    // cannot share an axis without one of them becoming unreadable. Same unit, different
    // question, different axis - so a different base.
    //
    // The diagnostics are the project's measurement instruments and they run repeatedly under
    // a lease budget; before this base existed, the only throughput number any of them
    // produced was one wall clock covering three populations, a corpus load and a `finish`,
    // which is why a 75 ms batch was read as a 189 ms one.
    "timexer_segment_eval_phases",
];

pub const RL_META_REPORT_BASES: &[&str] = &[
    "final_assets",
    "cumulative_reward",
    "outperformance",
    "policy_loss",
    "value_loss",
    "explained_var",
    "actor_grad_norm",
    "critic_grad_norm",
    "total_commissions",
    "beta_policy",
    "advantage_stats_log",
    "logit_scale",
    "clip_fraction",
    "clip_gap",
    "approx_kl",
    "kl_lr",
    "policy_entropy",
    "temporal_embed_debug",
    "gate_stats",
    "hl_gauss_return_range",
];

/// Every chart base written by either first-class pretrainer, and their shared source of
/// truth. Each writer's own test proves its portion, and the TUI builds its meta-chart list
/// by extending from this slice.
///
/// It lives here rather than in either consumer because the two failure modes are silent
/// in opposite directions and both shipped: a base registered with no writer renders as a
/// permanently blank panel, and a base written but not registered is a chart nobody can
/// see. One list makes both unrepresentable.
pub const PRETRAIN_REPORT_BASES: &[&str] = &[
    "pretrain_nll_bar",
    "pretrain_nll_bar_diag896",
    "pretrain_independent_marginal_nll",
    "pretrain_nll_dof",
    "pretrain_nll_vs_baselines",
    "pretrain_crps_dof",
    "pretrain_pit_hist",
    "pretrain_dyn_loss",
    "pretrain_kl_loss",
    "pretrain_total_loss",
    "pretrain_loss_shares",
    "pretrain_direct_horizon_nll",
    "pretrain_direct_objective",
    "pretrain_direct_gradients",
    "pretrain_direct_timing",
    "pretrain_growth_term",
    "pretrain_belief_autocorr",
    "pretrain_dyn_vs_identity",
    "pretrain_teacher_forced_rollout_score",
    "pretrain_ancestral_calibration",
    "pretrain_ancestral_tails",
    "pretrain_ancestral_bar_validity",
    "pretrain_ancestral_distribution_drift",
    "pretrain_dir_acc",
    "pretrain_lr",
    "pretrain_muon_momentum",
    // Centered per-output-row outcome-LR controller diagnostics. These remain distinct
    // panels because alpha allocation and the signed one-step evidence that trains it have
    // different scales and failure signatures.
    "pretrain_sdlr_alpha",
    "pretrain_sdlr_evidence",
    // Strictly opt-in batch-step Schraudolph SMD-IDBD diagnostics.
    "pretrain_smd_idbd_gain",
    "pretrain_smd_idbd_credit",
    "pretrain_grad_norm",
    "pretrain_unique_bar_reuse",
    "pretrain_stage_coverage",
    "pretrain_pass_coverage",
    "pretrain_pass_multiplicity",
    "pretrain_pass_remainder",
    "pretrain_stage_conditioning",
    "pretrain_effective_rank",
    "pretrain_promotions",
    "pretrain_schedule",
    "pretrain_capacity",
    "pretrain_market_coverage",
    "pretrain_candle_rollout_pit",
    "pretrain_candle_rollout_dclose",
    "pretrain_candle_rollout_band",
    "pretrain_candle_rollout_coverage",
    "pretrain_trade_growth",
    "pretrain_trade_vs_baselines",
    "pretrain_trade_cost_curve",
    "pretrain_trade_sharpe",
    "pretrain_trade_exposure",
    "pretrain_trade_cap_curve",
    "pretrain_trade_free_kelly",
    "pretrain_trade_tail",
    // The VOLATILITY half of the same validation, on the same record-tick axis. Every base
    // above scores the conditional MEAN, which on 5-minute bars carries an `R^2` of order
    // `1e-3`; the bar's own variance is the quantity this corpus can actually predict, and
    // until these two existed nothing anywhere said whether the model beats a three-parameter
    // HAR-RV regression at it. Two bases because they are different objects: the first is the
    // LEVEL of each forecaster's QLIKE plus its measured level bias, and the second is the
    // PAIRED difference against the HAR baseline with its block-bootstrap interval, which is
    // not recoverable from four independently-intervalled levels.
    "pretrain_vol_qlike",
    "pretrain_vol_vs_har",
    // The TARGET side of the same run: not what the model predicts, but what it was asked to
    // predict. All three are measured once on the evaluation split by
    // `BarCorpus::audit_target_geometry` before step zero, and all three are written on EVERY
    // run rather than only on the standardized arm — a per-symbol occupancy spread means
    // nothing without the control's spread beside it, and a chart the control does not write
    // cannot supply one.
    //
    // Three bases because they are three different objects on three different axes and none is
    // recoverable from another: the DIVISOR's own distribution and its two correctness
    // tripwires, the QUOTIENT's scale and shape as the supports fitter receives it, and what
    // the quotient does to the 128-bin grid the emission head actually predicts on, decomposed
    // BY SYMBOL because a pooled equal-mass histogram is balanced by construction on either
    // parametrization and therefore cannot see the defect at all.
    "pretrain_sigma_dist",
    "pretrain_target_scale",
    "pretrain_bin_occupancy",
    // The EPOCH-INDEXED panel. Distinct bases from the trade series above on purpose:
    // those are the dense record-tick curves measured at every validation, these are one
    // point per pass over the corpus. Neither is derivable from the other and neither
    // overwrites the other.
    "pretrain_epoch_trade_edge",
    "pretrain_epoch_trade",
    "pretrain_epoch_progress",
    // Written once, at the end of a run, by `PretrainReporter::finish`.
    "pretrain_test",
    // Written by the corpus loader at startup rather than by the reporter, but it lands
    // in the same directory and is read the same way, so it is registered the same way.
    "pretrain_corpus_anomalies",
    // Written by the AUXILIARY-resolution stream at every epoch boundary, for the same
    // reason and by the same convention: the reporter's row schema is the deployment
    // resolution's, and an auxiliary resolution has its own supports, its own ramp and its
    // own held-out geometry, so its curve is a separate object rather than another column.
    // One point per pass per auxiliary resolution. Absent from a run that named none.
    "pretrain_auxiliary_nll",
    // Written by `trading_bots::torch::train::portfolio::write_portfolio_bench`, which runs
    // ONE book over a calendar-aligned panel of the held-out split rather than averaging
    // per-window bets. Distinct from the `pretrain_trade_*` bases above and not derivable
    // from them: those measure a single name's log-optimal bet, these measure a portfolio
    // under one shared capital constraint.
    "pretrain_portfolio_equity",
    "pretrain_portfolio_metrics",
    "pretrain_portfolio_gross_curve",
    "pretrain_portfolio_frontier",
    // The same writer's edge-versus-cost table: what one name-bar of forecast is worth in bps
    // beside what one one-way trade in it costs, per liquidity decile of the traded panel. The
    // two halves have to be measured on the same panel to be a comparison at all, which is why
    // this base is the portfolio writer's and not `pretrain_cost_deciles`'.
    "pretrain_portfolio_edge_vs_cost",
    // Written by `trading_bots::torch::train::portfolio_cost::write_cost_capacity_reports`.
    // Properties of the CORPUS rather than of a training step: a spread, an ADV and a
    // realized cross-sectional covariance are measured from stored bars and do not move
    // when a step does, so no in-run reporter cycle can produce them.
    "pretrain_cost_deciles",
    "pretrain_capacity_curve",
    "pretrain_cross_correlation",
    // Written by `trading_bots::torch::train::pretrain_reports::write_mean_calibration`, from
    // the multi-checkpoint mean-calibration experiment. One point per CHECKPOINT rather than
    // per step of one run: a Mincer-Zarnowitz slope needs a whole held-out pass, and the
    // recalibrated policy beside it needs a second pass on a block-disjoint fit slice, so
    // neither is producible from inside a training cycle.
    "pretrain_mean_calibration",
    "pretrain_shrunk_policy",
    // Written by `pretrain_reports::write_sizing_rule_chart`, from the SAME pass that sized
    // the book, and present only when that pass sized on something other than the control
    // rule. Indexed by LEVERAGE CAP and not derivable from any base above: every other
    // `pretrain_trade_*` base conditions on whatever rule was in force and cannot say what
    // the rule it replaced would have done on the same bars, while `pretrain_shrunk_policy`
    // is the offline two-pass comparison of a fraction the incumbent did NOT trade. This one
    // is the in-run, promotion-deciding difference: exact expected-log versus its
    // second-order surrogate and/or recalibrated versus raw mean, paired window by window at
    // every cap, with both Mincer-Zarnowitz slopes — the APPLIED one and the MEASURED one —
    // as reference lines and the count of control bars whose optimum left the ruin domain.
    "pretrain_sizing_rule",
    // Written by the same writer, from the same two passes: the COST-AWARE sizing axis.
    // `trade_bench`'s Kelly solve maximizes `E[ln(1 + f R)]`, which carries no cost term, so
    // the position is chosen frictionlessly and the charge is levied afterwards on whatever
    // turnover that produced. Under proportional costs the optimal policy instead has a
    // no-trade region, so this base is the band swept as an axis, under both fill rules, with
    // the gain over the unbanded incumbent taken PAIRED window by window. Indexed by BAND
    // WIDTH rather than by step or by cap, and not derivable from `pretrain_shrunk_policy`:
    // that one varies the MEAN the solve is handed and this one varies how often the solve is
    // acted on, and whether the two overlap is the third panel of this base.
    "pretrain_no_trade_band",
    // Written by the same writer, from the same two passes: WHERE the measured edge lives.
    // A hit rate below a coin flip beside an edge whose interval excludes zero cannot be read
    // as directional skill, so the arm table re-scores the identical windows with the model's
    // MAGNITUDE destroyed at matched gross exposure, and again with its SIGN destroyed, each
    // paired against the null and against the actual policy over the same blocks. Indexed by
    // ARM, so it is not derivable from any base above: `pretrain_skill_profile` scores the
    // predictor with no policy at all and every `pretrain_trade_*` base conditions on the
    // undamaged Kelly policy.
    "pretrain_edge_attribution",
    // The panel underneath those arms, indexed by CHECKPOINT: `corr(f, R)`, `corr(|f|, |R|)`
    // and the mean size of a winning bar against a losing one, which is the arithmetic a
    // sub-coin-flip hit rate with positive growth has to satisfy.
    "pretrain_edge_panel",
    // The same panel cut by DECILE of the model's own uncapped `|f*|`. The confidence axis is
    // the discriminator the arm table cannot supply on its own: a hit rate flat across every
    // decile while growth concentrates in the top ones is a size-carried result, and a hit
    // rate that rises with `|f*|` is a direction predictor that knows where its sign is good.
    "pretrain_edge_confidence",
    // The sign-hysteresis frontier, indexed by FLIP MARGIN in bps of predicted mean. On a book
    // whose turnover is almost entirely sign flips, holding the sign longer is the only lever
    // left that can move the cost, and margin zero is the sign-only arm exactly - so this base
    // extends the arm table along an axis the arm table does not have. Not derivable from
    // `pretrain_no_trade_band`: that dead-zones the MAGNITUDE of the target, which on a
    // two-valued book suppresses re-sizings, while this one suppresses REVERSALS.
    "pretrain_edge_hysteresis",
    // The recalibration shrink crossed with sign hysteresis, indexed by CELL of a 2x2. Both
    // levers cut the cost of the same book by trading less, so neither their gains nor their
    // break-evens can be added, and the second difference that decides it is not recoverable
    // from `pretrain_shrunk_policy` and `pretrain_edge_hysteresis` side by side - those score
    // each lever against the incumbent, never against each other on the same windows. Distinct
    // from the band-versus-shrink overlap for the same reason the frontier is distinct from the
    // band: this crosses a REVERSAL rule with the shrink, not a magnitude dead-zone.
    "pretrain_edge_composition",
    // How fast the CURRENT one-bar signal's directional content decays with holding horizon,
    // indexed by HORIZON in bars, with no policy and no cost anywhere in the measurement. It
    // bounds what a one-bar signal HELD longer can be worth, and deliberately says nothing
    // about a model TRAINED on a k-bar target, whose predictable component and noise floor are
    // different quantities. Distinct from `pretrain_horizon_frontier`, which scores POLICIES
    // under two constructions rather than the bare signal.
    "pretrain_signal_decay",
    // The INFORMATION COEFFICIENT of the same signal at the same horizons, with intervals.
    // Separate from `pretrain_signal_decay` because that base's `corr` column is a single
    // pooled Pearson number with no interval, and neither of the two statements this base
    // makes is recoverable from it: a rank correlation bounds every bar's influence, which a
    // moment statistic on a kurtosis-of-tens regressand does not, and it is invariant to any
    // monotone reparameterization of either axis, which makes it the only directional metric
    // here that survives a change of the `r` bin geometry. Both the rank and the Pearson
    // version are charted with block-bootstrap bands so the gap between them is readable.
    "pretrain_rank_ic",
    // Written by `trading_bots::torch::train::skill::write_skill_profile`: the DIRECTIONAL
    // skill of the predictor, scored with no trading policy anywhere in the measurement.
    // Decile-indexed rather than step-indexed - the x axis is the model's own confidence, not
    // training progress - so it is not producible from inside a training cycle and is not
    // derivable from any `pretrain_trade_*` base, which all condition on a Kelly policy.
    "pretrain_skill_profile",
    // Written by `trading_bots::torch::train::horizon::write_horizon_frontier`. Break-even
    // cost against the HOLDING HORIZON, for the model and its three baselines under both the
    // stale-one-bar and the k-bar-aggregate construction. Not derivable from
    // `pretrain_portfolio_frontier`: that curve varies a no-trade band on a one-bar forecast,
    // which freezes stale positions, while this one varies the horizon the forecast is OF.
    // A whole held-out panel and a sampled multi-bar rollout per point, so no in-run cycle
    // can produce it.
    "pretrain_horizon_frontier",
    // Canonical production strategy evaluation. Forecast horizon varies while the decision
    // clock remains one bar; costs, actual holdings and constraints are inside the action
    // solve. The companion base records the strictly trailing shrunk one-factor covariance
    // provenance and its realized constraint audit.
    "pretrain_receding_kelly",
    "pretrain_receding_covariance",
    // Selected production-H attribution ladder. One cached forecast panel is held fixed while
    // the shared receding book adds factor risk, measured non-impact costs and finally impact;
    // stage zero is explicitly a non-self-financing scalar moment diagnostic.
    "pretrain_receding_attribution",
    // Validation-only selected production-H solver-safe dead-zone frontier. It reuses the
    // cached all-in Model incumbent at width zero and reruns only the same constrained action
    // solver at the fixed trade_bench::BAND_FRACTIONS widths; locked test never writes this
    // grid, and no marginal arm or additional inference is involved.
    "pretrain_receding_policy_frontier",
    // Validation-only selected-H causal forecast-moment EMA frontier. It reuses the cached
    // forecast panel and unchanged cost-aware solver, pairing fixed half-lives against the raw
    // Model and Marginal runs on identical decision rows. Locked test never writes this grid.
    "pretrain_receding_persistence",
    // Optional selected-H two-row evidence for one predeclared causal mean-sign hysteresis
    // margin against the cached raw incumbent. It is absent unless explicitly requested and
    // never expands into a locked-test policy-selection grid.
    "pretrain_receding_hysteresis",
    // Written by `trading_bots::torch::train::support_moments::fit_support_moments` via
    // `pretrain_reports::write_support_decode`. Properties of the SUPPORT ARTIFACT alone: the
    // fitted per-bin conditional means measured against the persisted bin geometry, beside the
    // EDGE decode that every production first-moment consumer actually reads, beside the
    // hardcoded two-bin stand-in that preceded the measurement. No model, no checkpoint and no
    // step is involved, so an in-run reporter cycle cannot produce them and they do not move
    // when a step does. Registered here rather than only in the TUI because `meta_chart_bases`
    // extends from THIS slice, which is what makes a written-but-unregistered base
    // unrepresentable.
    "support_decode_moments",
    "support_decode_bins",
    // Written by `PretrainReporter::record_epoch`, and the ONLY run-scoped coverage bases in
    // this list. Every `pretrain_pass_*` and `pretrain_stage_*` base above is a PER-PASS census:
    // `CoverageAudit::require_full_pass` pins within-pass multiplicity to exactly one, so those
    // panels read "every bar once, twice: 0" on the third pass of a three-pass run exactly as on
    // the first. That is correct within a pass and it was read as a claim about the RUN for an
    // entire analysis session, in preference to `pretrain_unique_bar_reuse` showing 2.85 on the
    // same screen. These two carry the cross-pass fact — passes delivered, projected and asked
    // for, and bars by how many times the RUN has targeted them — and are the only bases that
    // can answer "how many times has the model seen this bar".
    "cover_effective_epochs",
    "cover_run_bar_exposure",
    // Written by `trading_bots::torch::train::mem_probe::mem_probe` via
    // `pretrain_reports::write_mem_probe`. The multi-epoch MEMORIZATION test: held-out NLL
    // against TRAIN-split NLL along the epoch spine, and the within-checkpoint contrast between
    // bars the run had trained on three times and bars it had trained on twice at the same
    // step. Neither is producible from inside a training cycle - the spine needs several
    // checkpoints and the contrast needs the training pass partition reconstructed at a
    // checkpoint's own step - and the two are deliberately separate bases because one is
    // contaminated by calendar and by learning rate while the other is randomized by
    // construction, and a reader must never mistake the first for a discriminator.
    "memprobe_epoch_spine",
    "memprobe_one_repetition",
    "memprobe_recency",
    "memprobe_bootstrap_stability",
    // Written by `trading_bots::torch::train::bar_family::fit_bar_families` via
    // `pretrain_reports::write_bar_family`. The offline GATE on replacing the 128-way discrete bar
    // support with a continuous per-DOF mixed likelihood: fitted density against the empirical
    // histogram per DOF, the `r` tail on log-log axes with the measured pairwise-slope band, the
    // component sweep, the marginal NLL against the discrete competitor on one stated footing, the
    // atom census with the u/v lattice probe, and the truncation bound a ruin licence implies.
    // Every panel is a property of a DRAW and a fitted family rather than of an optimizer step, so
    // no in-run reporter cycle can produce any of them and none moves when a step does. Registered
    // here rather than only in the TUI because `meta_chart_bases` extends from THIS slice, which
    // is what makes a written-but-unregistered base unrepresentable.
    "bar_family_density_r",
    "bar_family_density_s",
    "bar_family_density_u",
    "bar_family_density_v",
    "bar_family_density_w",
    "bar_family_tail_r",
    "bar_family_k_sweep",
    "bar_family_nll",
    "bar_family_atoms",
    "bar_family_ruin_bound",
    // Written by `trading_bots::torch::train::split_seams::audit_split_seams` via
    // `pretrain_reports::write_bar_seams`. The corporate-action SEAM audit: whether the extreme `r`
    // bars in the corpus are market moves or unadjusted split seams, and what the seams contaminate.
    // The exceedance census over all 451,507,140 bars, the nearest-simple-rational
    // cross-tabulation of `exp(r)` that decides the split hypothesis, the `s`/`w` comparison
    // against matched ordinary bars, the six pairwise tail slopes with and without the seams, the
    // catch-all bin contamination, and the ruin licence on both sides of the book. Every panel is a
    // property of the STORED BARS and of a support artifact read from disk, so no in-run reporter
    // cycle can produce any of them and none moves when a step does. Registered here rather than
    // only in the TUI because `meta_chart_bases` extends from THIS slice, which is what makes a
    // written-but-unregistered base unrepresentable.
    "bar_seam_census",
    "bar_seam_ratios",
    "bar_seam_context",
    "bar_seam_tail_r",
    "bar_seam_bin_mass",
    "bar_seam_ruin_licence",
    // Written by `trading_bots::torch::train::pretrain_reports::write_heldout_power`, from the
    // window draw that `pretrain-calibration` performs BEFORE it opens a checkpoint. The census
    // of every split at one context — bars, near-disjoint windows, symbols — and the interval a
    // traded prefix of the addressed split can support, as a function of the `(symbol, calendar
    // month)` BLOCK count counted over the real draw. Properties of the CORPUS and of a
    // seed-pinned draw, never of a model: no checkpoint is loaded and nothing is scored, which
    // is the point. `Split::Test` is scored once for the whole campaign, so whether it has the
    // power to resolve the effect being looked for has to be a chart that exists before the draw
    // is spent. Registered here rather than only in the TUI because `meta_chart_bases` extends
    // from THIS slice, which is what makes a written-but-unregistered base unrepresentable.
    "pretrain_heldout_census",
    "pretrain_heldout_power",
    // Written by the frozen-checkpoint fixed recirculation experiment. It contains the
    // serialized-alpha0 control, Stage-A screen, disjoint Stage-B confirmation, and the
    // gated test result in one discoverable artifact.
    "pretrain_recirculation_sweep",
    // Isolated c277 MSE-JEPA/LeJEPA pretrainer. These names deliberately do not share the
    // categorical pretrainer's `pretrain_*` family, so cross-family runs cannot render as one.
    "mse_jepa_loss",
    "mse_jepa_objective",
    "mse_jepa_representation",
    "mse_jepa_optimization",
    "mse_jepa_tail_ema",
    "mse_jepa_emission",
    "mse_jepa_flow",
    "mse_jepa_flow_samples",
    "mse_jepa_posthoc_readout",
    "mse_jepa_rollout_nll",
    "mse_jepa_rollout_calibration",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub title: String,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    pub scale: ScaleKind,
    pub kind: ReportKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSeries {
    pub label: String,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradePoint {
    pub index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleBar {
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
}

impl CandleBar {
    pub fn is_valid_ohlc(&self) -> bool {
        [self.open, self.high, self.low, self.close]
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
            && self.low <= self.open.min(self.close)
            && self.high >= self.open.max(self.close)
    }
}

/// Independent close-price marginal at a fixed horizon, with a central 90% interval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizonForecast {
    pub horizon: usize,
    pub lower: f32,
    pub median: f32,
    pub upper: f32,
}

/// One marginal-quantile locus of a sampled path, e.g. the p10 of the sampled
/// close at each horizon.
///
/// A quantile locus is NOT a path the process can take: `closes[t]` is a
/// property of the marginal distribution at horizon `t`, computed independently
/// per horizon, so consecutive entries need not belong to any single draw. That
/// is exactly why [`ReportKind::CandleFan`] carries genuine draws beside these.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantileBand {
    /// Probability in `(0, 1)`.
    pub probability: f64,
    pub closes: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportKind {
    Simple {
        values: Vec<f32>,
        ema_alpha: Option<f64>,
    },
    MultiLine {
        series: Vec<ReportSeries>,
    },
    Assets {
        total: Vec<f32>,
        cash: Vec<f32>,
        positioned: Option<Vec<f32>>,
        benchmark: Option<Vec<f32>>,
    },
    BuySell {
        prices: Vec<f32>,
        buys: Vec<TradePoint>,
        sells: Vec<TradePoint>,
    },
    /// A realized path against the predictive law it was drawn from: the bars
    /// that happened, the quantile fan of the sampled continuations, and a few
    /// of the sampled continuations themselves.
    ///
    /// There is deliberately no "predicted" field. A single line cannot stand
    /// in for a distribution over paths, and a fan centre rendered against one
    /// realization invites the reader to score a pointwise error that no
    /// forecast ever claimed.
    CandleFan {
        /// Realized bars, the only thing here that actually happened.
        actual: Vec<CandleBar>,
        /// Quantile loci of the sampled close, ASCENDING in probability.
        bands: Vec<QuantileBand>,
        /// Genuine draws of the close path from the predictive law.
        samples: Vec<ReportSeries>,
    },
    Observations {
        observation_tickers: Vec<String>,
        action_tickers: Vec<String>,
        static_observations: Vec<Vec<f32>>,
        attention_weights: Vec<Vec<f32>>,
        action_step0: Option<Vec<f32>>,
        action_final: Option<Vec<f32>>,
    },
    /// Plot-compatible diagnostics with an exact, versioned evidence payload.
    Evidence {
        series: Vec<ReportSeries>,
        metadata: Vec<u8>,
    },
    CandleForecast {
        actual: Vec<CandleBar>,
        /// Index of the final observed candle at the forecast origin.
        origin: usize,
        forecasts: Vec<HorizonForecast>,
    },
    /// Direct deterministic OHLC predictions, beginning one bar after `origin`.
    CandleSegment {
        actual: Vec<CandleBar>,
        origin: usize,
        predicted: Vec<CandleBar>,
    },
    IndexedLines {
        steps: Vec<u64>,
        series: Vec<ReportSeries>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ScaleKind {
    Linear,
    Symlog,
}

impl ReportKind {
    pub fn to_lines(&self) -> Vec<String> {
        match self {
            ReportKind::Simple { values, .. } => values
                .iter()
                .enumerate()
                .map(|(i, v)| format!("{i}\t{v}"))
                .collect(),
            ReportKind::MultiLine { series } | ReportKind::Evidence { series, .. } => {
                let max_len = series.iter().map(|s| s.values.len()).max().unwrap_or(0);
                let mut lines = Vec::with_capacity(max_len);
                for i in 0..max_len {
                    let mut line = format!("{i}");
                    for s in series {
                        if let Some(v) = s.values.get(i) {
                            line.push('\t');
                            line.push_str(&s.label);
                            line.push('=');
                            line.push_str(&v.to_string());
                        }
                    }
                    lines.push(line);
                }
                lines
            }
            ReportKind::Assets {
                total,
                cash,
                positioned,
                benchmark,
            } => {
                let max_len = total.len().max(cash.len());
                let mut lines = Vec::with_capacity(max_len);
                for i in 0..max_len {
                    let mut line = format!("{i}");
                    if let Some(v) = total.get(i) {
                        line.push_str(&format!("\ttotal={v}"));
                    }
                    if let Some(v) = cash.get(i) {
                        line.push_str(&format!("\tcash={v}"));
                    }
                    if let Some(pos) = positioned.as_ref().and_then(|p| p.get(i)) {
                        line.push_str(&format!("\tpositioned={pos}"));
                    }
                    if let Some(bench) = benchmark.as_ref().and_then(|b| b.get(i)) {
                        line.push_str(&format!("\tbenchmark={bench}"));
                    }
                    lines.push(line);
                }
                lines
            }
            ReportKind::BuySell {
                prices,
                buys,
                sells,
            } => {
                let mut buy_map: std::collections::HashSet<usize> =
                    std::collections::HashSet::new();
                let mut sell_map: std::collections::HashSet<usize> =
                    std::collections::HashSet::new();
                for b in buys {
                    buy_map.insert(b.index as usize);
                }
                for s in sells {
                    sell_map.insert(s.index as usize);
                }
                let mut lines = Vec::with_capacity(prices.len());
                for (i, price) in prices.iter().enumerate() {
                    let mut line = format!("{i}\tprice={price}");
                    if buy_map.contains(&i) {
                        line.push_str("\tbuy=1");
                    }
                    if sell_map.contains(&i) {
                        line.push_str("\tsell=1");
                    }
                    lines.push(line);
                }
                lines
            }
            ReportKind::CandleFan {
                actual,
                bands,
                samples,
            } => {
                let max_len = actual
                    .len()
                    .max(bands.iter().map(|b| b.closes.len()).max().unwrap_or(0))
                    .max(samples.iter().map(|s| s.values.len()).max().unwrap_or(0));
                let mut lines = Vec::with_capacity(max_len);
                for i in 0..max_len {
                    let mut line = format!("{i}");
                    if let Some(c) = actual.get(i) {
                        line.push_str(&format!(
                            "\tactual=o:{:.6},h:{:.6},l:{:.6},c:{:.6}",
                            c.open, c.high, c.low, c.close
                        ));
                    }
                    for band in bands {
                        if let Some(close) = band.closes.get(i) {
                            line.push_str(&format!(
                                "\tp{:02}={close:.6}",
                                (band.probability * 100.0).round() as i64
                            ));
                        }
                    }
                    for series in samples {
                        if let Some(close) = series.values.get(i) {
                            line.push_str(&format!("\t{}={close:.6}", series.label));
                        }
                    }
                    lines.push(line);
                }
                lines
            }
            ReportKind::CandleForecast {
                actual,
                origin,
                forecasts,
            } => {
                let mut lines = Vec::with_capacity(actual.len());
                let max_len = forecasts
                    .iter()
                    .filter_map(|f| origin.checked_add(f.horizon))
                    .filter_map(|index| index.checked_add(1))
                    .max()
                    .unwrap_or(0)
                    .max(actual.len());
                for i in 0..max_len {
                    let mut line = format!("{i}\tbar_from_origin={}", i as i64 - *origin as i64);
                    if let Some(c) = actual.get(i) {
                        line.push_str(&format!(
                            "\tactual=o:{:.6},h:{:.6},l:{:.6},c:{:.6}",
                            c.open, c.high, c.low, c.close
                        ));
                    }
                    if i == *origin {
                        line.push_str("\tforecast_origin=1");
                    }
                    for forecast in forecasts
                        .iter()
                        .filter(|f| origin.checked_add(f.horizon) == Some(i))
                    {
                        line.push_str(&format!(
                            "\thorizon={}\tp05={:.6}\tp50={:.6}\tp95={:.6}",
                            forecast.horizon, forecast.lower, forecast.median, forecast.upper
                        ));
                    }
                    lines.push(line);
                }
                lines
            }
            ReportKind::CandleSegment {
                actual,
                origin,
                predicted,
            } => {
                let length = actual
                    .len()
                    .max(origin.saturating_add(1).saturating_add(predicted.len()));
                (0..length)
                    .map(|index| {
                        let mut line =
                            format!("{index}\tbar_from_origin={}", index as i64 - *origin as i64);
                        if let Some(c) = actual.get(index) {
                            line.push_str(&format!(
                                "\tactual=o:{:.6},h:{:.6},l:{:.6},c:{:.6}",
                                c.open, c.high, c.low, c.close
                            ));
                        }
                        if let Some(c) = index
                            .checked_sub(origin.saturating_add(1))
                            .and_then(|i| predicted.get(i))
                        {
                            line.push_str(&format!(
                                "\tpredicted=o:{:.6},h:{:.6},l:{:.6},c:{:.6}\tinvalid_ohlc={}",
                                c.open,
                                c.high,
                                c.low,
                                c.close,
                                u8::from(!c.is_valid_ohlc())
                            ));
                        }
                        if index == *origin {
                            line.push_str("\tforecast_origin=1");
                        }
                        line
                    })
                    .collect()
            }
            ReportKind::IndexedLines { steps, series } => steps
                .iter()
                .enumerate()
                .map(|(index, step)| {
                    let mut line = format!("{step}");
                    for s in series {
                        if let Some(value) = s.values.get(index) {
                            line.push_str(&format!("\t{}={value}", s.label));
                        }
                    }
                    line
                })
                .collect(),
            ReportKind::Observations {
                observation_tickers,
                action_tickers,
                static_observations,
                attention_weights,
                action_step0,
                action_final,
            } => {
                let mut lines = Vec::new();
                if !observation_tickers.is_empty() {
                    lines.push(format!(
                        "observation_tickers\t{}",
                        observation_tickers.join(",")
                    ));
                }
                if !action_tickers.is_empty() {
                    lines.push(format!("action_tickers\t{}", action_tickers.join(",")));
                }
                if let Some(action) = action_step0 {
                    lines.push(format!("action_step0\t{}", format_vec_f32(action)));
                }
                if let Some(action) = action_final {
                    lines.push(format!("action_final\t{}", format_vec_f32(action)));
                }
                for (i, obs) in static_observations.iter().enumerate() {
                    lines.push(format!("static\t{i}\t{}", format_vec_f32(obs)));
                }
                for (i, attn) in attention_weights.iter().enumerate() {
                    lines.push(format!("attn\t{i}\t{}", format_vec_f32(attn)));
                }
                lines
            }
        }
    }
}

fn format_vec_f32(values: &[f32]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

pub fn read_report(path: impl AsRef<Path>) -> io::Result<Report> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|error| report_io_error("read", path, error))?;
    postcard::from_bytes(&bytes).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("decode report {}: {error}", path.display()),
        )
    })
}

pub fn write_report(path: impl AsRef<Path>, report: &Report) -> io::Result<()> {
    let path = path.as_ref();
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        fs::create_dir_all(parent)
            .map_err(|error| report_io_error("create parent directory for", path, error))?;
    }

    let bytes = postcard::to_stdvec(report).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("encode report {}: {error}", path.display()),
        )
    })?;
    let (temporary, mut file) = create_temporary_sibling(path)?;
    let result = (|| {
        file.write_all(&bytes)
            .map_err(|error| report_io_error("write temporary report for", path, error))?;
        file.sync_all()
            .map_err(|error| report_io_error("sync temporary report for", path, error))?;
        drop(file);
        fs::rename(&temporary, path).map_err(|error| report_io_error("publish", path, error))?;
        if let Some(parent) = parent {
            File::open(parent)
                .and_then(|directory| directory.sync_all())
                .map_err(|error| report_io_error("sync parent directory for", path, error))?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn create_temporary_sibling(path: &Path) -> io::Result<(PathBuf, File)> {
    let file_name = path.file_name().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "report path has no file name")
    })?;
    for _ in 0..100 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let mut temporary_name = file_name.to_os_string();
        temporary_name.push(format!(".tmp-{}-{sequence}", std::process::id()));
        let temporary = path.with_file_name(temporary_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => return Ok((temporary, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(report_io_error("create temporary file for", path, error)),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique report temporary file",
    ))
}

fn report_io_error(operation: &str, path: &Path, error: io::Error) -> io::Error {
    io::Error::new(
        error.kind(),
        format!("{operation} report {}: {error}", path.display()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn candle_forecast_roundtrip_preserves_sparse_horizons_and_old_variant_tags() {
        let kind = ReportKind::CandleForecast {
            actual: vec![
                CandleBar {
                    open: 10.0,
                    high: 11.0,
                    low: 9.0,
                    close: 10.5
                };
                5
            ],
            origin: 1,
            forecasts: vec![HorizonForecast {
                horizon: 3,
                lower: 8.0,
                median: 10.0,
                upper: 12.0,
            }],
        };
        let bytes = postcard::to_allocvec(&kind).unwrap();
        assert_eq!(bytes[0], 7);
        let decoded: ReportKind = postcard::from_bytes(&bytes).unwrap();
        let lines = decoded.to_lines();
        assert!(lines[1].contains("forecast_origin=1"));
        assert!(!lines[2].contains("p50="));
        assert!(!lines[3].contains("p50="));
        assert!(lines[4].contains("horizon=3\tp05=8.000000\tp50=10.000000\tp95=12.000000"));
        assert_eq!(
            postcard::to_allocvec(&ReportKind::CandleFan {
                actual: vec![],
                bands: vec![],
                samples: vec![]
            })
            .unwrap()[0],
            4
        );
        assert_eq!(
            postcard::to_allocvec(&ReportKind::Evidence {
                series: vec![],
                metadata: vec![]
            })
            .unwrap()[0],
            6
        );
    }

    #[test]
    fn segment_reports_preserve_invalid_predictions_and_origin_alignment() {
        let actual = CandleBar {
            open: 10.0,
            high: 12.0,
            low: 9.0,
            close: 11.0,
        };
        let invalid = CandleBar {
            open: 10.0,
            high: 8.0,
            low: 13.0,
            close: 11.0,
        };
        assert!(actual.is_valid_ohlc());
        assert!(!invalid.is_valid_ohlc());
        let kind = ReportKind::CandleSegment {
            actual: vec![actual; 3],
            origin: 1,
            predicted: vec![invalid],
        };
        let bytes = postcard::to_allocvec(&kind).unwrap();
        assert_eq!(bytes[0], 8);
        let decoded: ReportKind = postcard::from_bytes(&bytes).unwrap();
        let lines = decoded.to_lines();
        assert!(!lines[1].contains("predicted="));
        assert!(lines[1].contains("forecast_origin=1"));
        assert!(lines[2].contains("predicted=o:10.000000,h:8.000000,l:13.000000,c:11.000000"));
        assert!(lines[2].contains("invalid_ohlc=1"));
    }

    #[test]
    fn indexed_lines_keep_real_steps() {
        let kind = ReportKind::IndexedLines {
            steps: vec![1000, 1907],
            series: vec![ReportSeries {
                label: "MSE".to_owned(),
                values: vec![1.0, 2.0],
            }],
        };
        let decoded: ReportKind =
            postcard::from_bytes(&postcard::to_allocvec(&kind).unwrap()).unwrap();
        assert_eq!(decoded.to_lines(), ["1000\tMSE=1", "1907\tMSE=2"]);
    }

    fn test_report(value: f32) -> Report {
        Report {
            title: "atomic".to_owned(),
            x_label: None,
            y_label: None,
            scale: ScaleKind::Linear,
            kind: ReportKind::Simple {
                values: vec![value; 512],
                ema_alpha: None,
            },
        }
    }

    fn temp_path(test: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "shared-report-{test}-{}-{}.report.bin",
            std::process::id(),
            TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn atomic_writer_reports_directory_failures() {
        let parent = temp_path("not-a-directory");
        fs::write(&parent, b"file").unwrap();
        let error = write_report(parent.join("report.bin"), &test_report(1.0)).unwrap_err();
        assert!(matches!(
            error.kind(),
            io::ErrorKind::AlreadyExists | io::ErrorKind::NotADirectory
        ));
        fs::remove_file(parent).unwrap();
    }

    #[test]
    fn truncated_report_is_an_explicit_decode_error() {
        let path = temp_path("truncated");
        write_report(&path, &test_report(1.0)).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes.truncate(bytes.len() / 2);
        fs::write(&path, bytes).unwrap();
        assert_eq!(
            read_report(&path).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn concurrent_readers_never_observe_partial_reports() {
        let path = Arc::new(temp_path("concurrent"));
        write_report(path.as_ref(), &test_report(0.0)).unwrap();
        let reader_path = Arc::clone(&path);
        let reader = thread::spawn(move || {
            for _ in 0..2_000 {
                let report = read_report(reader_path.as_ref()).unwrap();
                let ReportKind::Simple { values, .. } = report.kind else {
                    panic!("unexpected report kind");
                };
                assert_eq!(values.len(), 512);
                assert!(values.iter().all(|value| *value == values[0]));
            }
        });
        for value in 1..=100 {
            write_report(path.as_ref(), &test_report(value as f32)).unwrap();
        }
        reader.join().unwrap();
        fs::remove_file(path.as_ref()).unwrap();
    }
}
