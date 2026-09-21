//! End-to-end book backtest: raw un-gained 192-horizon head emission -> overlapping-tranche
//! market-neutral target weights -> realistic IBKR Pro account, swept over AUM, commission tier
//! and the two aggregation/weighting design choices the book rests on.
//!
//! This path deliberately does NOT reuse [`super::portfolio_data::evaluate`]. Two differences
//! are load-bearing rather than incidental:
//!
//! * It reads [`CausalPatchModel::output`], the raw head emission, and never
//!   `CausalPatchModel::decode`. `decode` applies the checkpoint's frozen `mean_gain`, whose
//!   signed measurement at h = 64 came out NEGATIVE and is therefore clamped to zero - so the
//!   existing account path sizes every name to zero and trades nothing. A rank-based book is
//!   amplitude-invariant by construction: multiplying every name's mean by one positive scalar
//!   cannot change a cross-sectional ranking, so there is nothing for a gain to calibrate and
//!   nothing for a gain of zero to destroy except the whole book.
//! * It never consults the amplitude sizing gate. The gate exists because LEVEL sizing is affine
//!   in the mean and a mismeasured amplitude would scale positions wrongly. Rank selection reads
//!   only the order, so the gate's premise does not hold here.
use super::{
    book::{self, Aggregation, BookConfig, BookFrame, BookQuote, HorizonPath, Weighting},
    book_diagnostics,
    corpus::{Batch, Corpus, WindowRef, RESOLUTION_MS},
    model::CausalPatchModel,
    portfolio::{
        self, AccountEvaluation, CommissionTier, CostSource, Forecast, PortfolioConfig, Tape,
    },
    portfolio_data, reports,
    runner::{load_checkpoint, Prefetcher},
};
use crate::torch::train::portfolio_cost::{BarCostModel, CostCalibration};
use anyhow::{ensure, Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tch::{Device, Kind, Tensor};

/// The horizon subset the tape stores. A full 192-horizon path for a 256-name universe over the
/// whole validation window is ~7.7 GB of `f32`; these eleven horizons are ~220 MB and span the
/// entire measured IC curve (0.018 at h1 through 0.055 at h64, still rising at h128), so every
/// aggregation and holding-period choice in [`BookConfig`] can be swept without another forward
/// pass. Powers of two plus 48 and 96, so the interior points a precision-weighted aggregate
/// leans on are present, and 192 so the terminal horizon's dispersion is observable.
const BOOK_HORIZONS: [u16; 11] = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192];
const HORIZONS: usize = BOOK_HORIZONS.len();

/// `v2`: the tape stores the RAW un-gained σ-unit emission at [`BOOK_HORIZONS`] plus the realized
/// σ-unit residual label at the same horizons. A `portfolio_data` `v9` tape stores one gained,
/// gate-zeroed scalar mean per name and is not convertible into this; the two formats are refused
/// against each other by name rather than silently reinterpreted. `v1` is refused as well: it was
/// inferred through [`Corpus::host_forecast_batch`], whose projected future-covariate clock is a
/// DIFFERENT conditioning from the one every reported metric of the checkpoint is measured under,
/// and its stored means are not comparable with these.
const CACHE_FORMAT: &str =
    "timexer-book-tape-v2-ungained-output-horizon-subset-sigma-units-realized-labels-label-clock";

/// Trailing dollar volume is measured over the last [`ADV_BARS`] observed REGULAR-session
/// five-minute bars at or before the decision origin - twenty sessions of 78 bars - and reported
/// as USD per session. Twenty sessions is long enough that one halted or holiday-thin day cannot
/// halve a name's apparent capacity, and short enough to follow a real liquidity regime change
/// inside a sixty-session backtest. Extended-hours prints are excluded: they carry a small
/// fraction of the volume, and including them would understate the session capacity a
/// participation limit is quoted against.
const ADV_BARS: usize = 1560;
const SESSION_BARS: f64 = 78.0;
/// Calendar milliseconds of bar history scanned before the first decision origin to fill the
/// [`ADV_BARS`] window. Sixty calendar days is at least twenty-eight regular sessions under every
/// US-equity holiday arrangement.
const ADV_LOOKBACK_MS: i64 = 60 * 86_400_000;

/// Start of the validation partition. Every measured cost estimate is fitted strictly before this
/// instant, so the cost the backtest charges is causal with respect to the window it charges on.
const VALIDATION_START_MS: i64 = 1_723_816_200_000;

#[derive(Args, Clone, Debug, Serialize, Deserialize)]
pub struct BookEvaluateArgs {
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long, default_value_os_t = crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,
    #[command(flatten)]
    pub account: super::portfolio::PortfolioConfig,
    #[command(flatten)]
    pub book: super::book::BookConfig,
    #[command(flatten)]
    pub schedule: super::portfolio_data::ScheduleConfig,
    /// Decision cadence in completed five-minute bars for the BOOK path, overriding the shared
    /// `--rebalance-bars`. The probe that produced the measured decile spreads used disjoint
    /// endpoint cohorts; a book is continuous, so the default is every bar and a name that stays
    /// inside its selected decile requires no trade at all.
    #[arg(long, default_value_t = 1)]
    pub book_rebalance_bars: usize,
    /// Account sizes to run the identical book at, USD. Commission and fees are per SHARE, so
    /// cost in bps falls with account size at fixed participation: the same signal is a different
    /// strategy at $25k and at $100M, and one number for "how much do we make" is meaningless.
    #[arg(long, value_delimiter = ',', default_values_t = [25_000.0, 100_000.0, 1_000_000.0, 10_000_000.0, 100_000_000.0])]
    pub aum_sweep: Vec<f64>,
    /// Charge the Roll-measured per-symbol spread and square-root impact instead of the flat
    /// `--spread-bps`/`--slippage-bps` scenario.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub measured_costs: bool,
    /// Authenticated un-gained forecast-path tape. Mismatched tapes are refused, never reused.
    #[arg(long)]
    pub book_tape_cache: Option<PathBuf>,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub diagnostics: bool,
    /// Population the reproduction diagnostic is measured on. `book` measures only the traded
    /// 256-name cross-sections; `full` additionally measures the in-run scorer's own population -
    /// every `Corpus::validation_refs` origin, grouped into the coincidence cohorts the scorer
    /// groups them into - so the universe restriction and the forecast reconstruction are two
    /// separately reported numbers rather than one confounded disagreement.
    #[arg(long, value_enum, default_value_t = DiagnosticsUniverse::Book)]
    pub diagnostics_universe: DiagnosticsUniverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
pub enum DiagnosticsUniverse {
    Book,
    Full,
}

/// One decision origin's stored emission. `mean`/`std` are RAW un-gained σ-unit values at
/// [`BOOK_HORIZONS`] - `mean` already carries its `√h` horizon scale and `std` the `½·ln h`
/// random-walk prior - and `realized` is the observed cumulative σ-unit residual label at the
/// same horizons, NaN only where the target bar's own validity mask is zero.
#[derive(Clone, Serialize, Deserialize)]
struct CachedRow {
    quote: u32,
    asset: u32,
    sigma: f32,
    adv_usd: f32,
    mean: [f32; HORIZONS],
    std: [f32; HORIZONS],
    realized: [f32; HORIZONS],
}

#[derive(Clone, Serialize, Deserialize)]
struct CachedFrame {
    frame: u32,
    timestamp_ms: i64,
    rows: Vec<CachedRow>,
}

#[derive(Serialize, Deserialize)]
struct CachedBookTape {
    binding: String,
    payload_sha256: String,
    tape: Tape,
    frames: Vec<CachedFrame>,
    summary: BTreeMap<String, f64>,
}

/// Rows aligned with the origins they were requested for.
struct Emission {
    mean: Vec<[f32; HORIZONS]>,
    std: Vec<[f32; HORIZONS]>,
    realized: Vec<[f32; HORIZONS]>,
    sigma: Vec<f32>,
    summary: BTreeMap<String, f64>,
}

fn record(summary: &mut BTreeMap<String, f64>, key: &str, value: f64) {
    if value.is_finite() {
        summary.insert(key.to_owned(), value);
    }
}

/// The book's forward-pass loader.
///
/// [`Prefetcher::new`] is load-bearing and MUST NOT be swapped for
/// [`Prefetcher::with_forecasts`]. The head reads auxiliary covariate tokens over the FORECAST
/// horizon as well as the context, and the two loaders disagree about what those tokens say:
/// `Corpus::host_batch` writes the calendar of the ticker's actual next-observed bars - the same
/// bars `CausalPatchModel::targets` reads the label off, and the conditioning every reported
/// metric of the checkpoint was measured under - while `Corpus::host_forecast_batch` overwrites
/// them with a deterministic 04:00-20:00 five-minute schedule. Those schedules agree exactly out
/// to h64 and then diverge by hours, and the emission is not local in the horizon axis: feeding
/// the projected schedule moves the h32 cross-sectional IC from +0.046 to +0.001 and the h64 from
/// +0.063 to +0.004 on the traded universe, and turns the scorer's own +0.047/+0.062 into
/// -0.002/-0.013. Scoring a forecast conditioned on one horizon clock against a label measured on
/// another is the defect that produced a near-zero book IC; it is a pairing error in the calendar
/// axis, not a scale error, which is why no positive rescaling could have recovered it.
fn book_loader(corpus: &Arc<Corpus>) -> Prefetcher {
    Prefetcher::new(Arc::clone(corpus))
}

/// One BF16 CUDA forward per batch, one host transfer for the whole population.
///
/// Every origin must own a COMPLETE `pred_len` of validation targets. That is not a convenience:
/// the future covariate tokens the head conditions on are written bar by bar out of the same
/// window the label is read from, so an origin whose window is short would be forecast under a
/// truncated, zero-filled clock and scored against whatever survived. `Corpus::host_batch`
/// refuses such an origin by itself; this refuses it earlier and by name, and refuses the locked
/// terminal test at the same time.
fn forecast(
    corpus: &Arc<Corpus>,
    model: &CausalPatchModel,
    origins: &[WindowRef],
    batch_size: usize,
    device: Device,
) -> Result<Emission> {
    ensure!(
        device.is_cuda(),
        "book inference requires the BF16 CUDA model"
    );
    ensure!(
        !origins.is_empty() && batch_size > 0,
        "forecast population and batch size must be nonempty"
    );
    let terminal = corpus.contract.boundary_timestamps[2];
    ensure!(
        origins.iter().all(|reference| {
            let source = corpus.ticker(*reference);
            source.owns_partition_targets(reference.origin, 1)
                && source.timestamp(reference.origin) < terminal
        }),
        "book inference accepts only validation origins owning a complete forecast window; the \
         locked terminal test is refused here as well as in the loader"
    );
    let _guard = tch::no_grad_guard();
    let started = Instant::now();
    let bars: Vec<i64> = BOOK_HORIZONS.iter().map(|h| i64::from(*h) - 1).collect();
    let roots: Vec<f32> = BOOK_HORIZONS
        .iter()
        .map(|h| f64::from(*h).sqrt() as f32)
        .collect();
    let bars = Tensor::from_slice(&bars).to_device(device);
    let roots = Tensor::from_slice(&roots).to_device(device);
    let loader = book_loader(corpus);
    loader.request(&origins[..origins.len().min(batch_size)])?;
    let mut resident: Option<Batch> = None;
    let mut outputs = Vec::new();
    let mut loader_ms = 0.;
    let mut loader_work_ms = 0.;
    for (index, _) in origins.chunks(batch_size).enumerate() {
        let waiting = Instant::now();
        let (host, work_ms) = loader.receive()?;
        loader_ms += waiting.elapsed().as_secs_f64() * 1000.;
        loader_work_ms += work_ms;
        let next = (index + 1) * batch_size;
        if next < origins.len() {
            loader.request(&origins[next..(next + batch_size).min(origins.len())])?;
        }
        if resident
            .as_ref()
            .is_none_or(|batch| batch.rows() != host.rows())
        {
            resident = Some(host.resident(device));
        }
        let batch = resident.as_mut().unwrap();
        host.upload(batch)?;
        let stats = model.statistics(batch);
        let head = model.forward(batch, &stats, false, true);
        // `output`, never `decode`: the raw emission, before the frozen mean calibration.
        let output = model.output(&head);
        let last = stats.last();
        let mean = output
            .coordinates
            .select(1, 0)
            .select(1, 0)
            .index_select(1, &bars)
            * &roots;
        let std = output
            .log_scale
            .select(1, 0)
            .select(1, 3)
            .index_select(1, &bars)
            .exp();
        let (target, mask) = model.targets(batch, &last, true);
        let rows = target.select(1, 0).select(1, 3).index_select(1, &bars);
        let valid = mask
            .select(1, 0)
            .select(1, 0)
            .index_select(1, &bars)
            .gt(0.0);
        let missing = Tensor::full_like(&rows, f64::NAN);
        let realized = rows.where_self(&valid, &missing);
        outputs.push(Tensor::cat(
            &[mean, std, realized, last.sigma.reshape([-1, 1])],
            1,
        ));
    }
    let host = Tensor::cat(&outputs, 0)
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .flatten(0, -1);
    drop(outputs);
    let flat = Vec::<f32>::try_from(host)?;
    let stride = 3 * HORIZONS + 1;
    ensure!(
        flat.len() == origins.len() * stride,
        "book forecast output cardinality changed"
    );
    let mut emission = Emission {
        mean: Vec::with_capacity(origins.len()),
        std: Vec::with_capacity(origins.len()),
        realized: Vec::with_capacity(origins.len()),
        sigma: Vec::with_capacity(origins.len()),
        summary: BTreeMap::new(),
    };
    for row in flat.chunks_exact(stride) {
        let mut block = [[0f32; HORIZONS]; 3];
        for (index, slot) in block.iter_mut().enumerate() {
            slot.copy_from_slice(&row[index * HORIZONS..(index + 1) * HORIZONS]);
        }
        emission.mean.push(block[0]);
        emission.std.push(block[1]);
        emission.realized.push(block[2]);
        emission.sigma.push(row[stride - 1]);
    }
    let prefix = "book";
    for (name, value) in [
        ("inference_ms", started.elapsed().as_secs_f64() * 1000.),
        ("loader_wait_ms", loader_ms),
        ("loader_work_ms", loader_work_ms),
        (
            "inference_batches_count",
            origins.len().div_ceil(batch_size) as f64,
        ),
        ("origins_count", origins.len() as f64),
    ] {
        emission.summary.insert(format!("{prefix}_{name}"), value);
    }
    Ok(emission)
}

/// Per-horizon cross-sectional IC on the population the run's own `timexer_segment_signal`
/// series is measured on: every `Corpus::validation_refs` origin, grouped into the coincidence
/// cohorts their shared origin timestamps form, gated at the scorer's twenty-name floor.
///
/// Measured through the SAME [`forecast`] chain the book trades on, so the only thing that
/// differs between this number and [`book_diagnostics::BookDiagnostics::ic_by_horizon`] is the
/// population. Without it a disagreement with the reported curve has two explanations at once -
/// the reconstruction and the 256-name liquidity screen - and neither is falsifiable.
fn full_universe_ic(
    corpus: &Arc<Corpus>,
    model: &CausalPatchModel,
    batch_size: usize,
    device: Device,
) -> Result<book_diagnostics::CohortIc> {
    let mut cohorts: BTreeMap<i64, Vec<WindowRef>> = BTreeMap::new();
    for reference in &corpus.validation_refs {
        cohorts
            .entry(corpus.ticker(*reference).timestamp(reference.origin))
            .or_default()
            .push(*reference);
    }
    let mut origins = Vec::new();
    let mut bounds = Vec::new();
    for members in cohorts.into_values() {
        if members.len() < 20 {
            continue;
        }
        bounds.push((origins.len(), members.len()));
        origins.extend(members);
    }
    ensure!(
        !bounds.is_empty(),
        "no held-out cohort carries the twenty names a cross-sectional IC needs"
    );
    println!(
        "reproduction pass: {} held-out origins in {} cohorts",
        origins.len(),
        bounds.len()
    );
    let emission = forecast(corpus, model, &origins, batch_size, device)?;
    let mut mean = Vec::with_capacity(origins.len() * HORIZONS);
    let mut realized = Vec::with_capacity(origins.len() * HORIZONS);
    for row in 0..origins.len() {
        mean.extend_from_slice(&emission.mean[row]);
        realized.extend_from_slice(&emission.realized[row]);
    }
    book_diagnostics::cohort_horizon_ic(HORIZONS, &bounds, &mean, &realized)
}

/// Trailing regular-session dollar volume per origin, in USD per session. See [`ADV_BARS`].
fn trailing_adv(corpus: &Corpus, origins: &[WindowRef]) -> Vec<f64> {
    let mut adv = vec![0.0; origins.len()];
    let mut by_ticker: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (row, reference) in origins.iter().enumerate() {
        by_ticker.entry(reference.ticker).or_default().push(row);
    }
    for (index, rows) in by_ticker {
        let source = portfolio_data::ticker(corpus, index);
        let bounds = rows.iter().map(|row| origins[*row].origin);
        let (first, last) = (bounds.clone().min().unwrap(), bounds.max().unwrap());
        let begin = portfolio_data::lower_bound(source, source.timestamp(first) - ADV_LOOKBACK_MS);
        // Inclusive prefix sums over regular-session bars only, so a window is one subtraction.
        let mut positions = Vec::with_capacity(last + 1 - begin);
        let mut prefix = vec![0.0f64];
        for position in begin..=last {
            let bar = source.bar(position);
            if !(570..960).contains(&portfolio_data::clock(bar.ts()).1) {
                continue;
            }
            let volume = f64::from(bar.volume);
            let dollars = if volume.is_finite() && volume > 0.0 {
                f64::from(bar.close) * volume
            } else {
                0.0
            };
            positions.push(position);
            prefix.push(prefix.last().unwrap() + dollars);
        }
        for row in rows {
            let count = positions.partition_point(|position| *position <= origins[row].origin);
            if count == 0 {
                continue;
            }
            let start = count.saturating_sub(ADV_BARS);
            let sessions = ((count - start) as f64 / SESSION_BARS).max(1.0 / SESSION_BARS);
            adv[row] = (prefix[count] - prefix[start]) / sessions;
        }
    }
    adv
}

/// The measured cost panel, fitted on bars strictly before the validation partition.
///
/// The panel is built in universe order, so an asset's index IS its `CostCalibration::symbols`
/// row and `Asset::symbol_index` needs no separate lookup table. Only the Roll (1984) estimator
/// contributes: Corwin-Schultz measures NEGATIVE in all ten liquidity deciles at five-minute
/// sampling, so it identifies nothing at this resolution.
fn measure_costs(tape: &Tape, data_dir: &Path) -> Result<CostCalibration> {
    let files: Vec<shared::bars::BarFile> = tape
        .assets
        .iter()
        .map(|asset| {
            let path = shared::bars::bar_file_path(data_dir, &asset.symbol, 300);
            shared::bars::BarFile::open(&path)
                .with_context(|| format!("opening {} bars for cost measurement", asset.symbol))
        })
        .collect::<Result<_>>()?;
    let series: Vec<(String, &[shared::bars::PackedBar])> = tape
        .assets
        .iter()
        .zip(&files)
        .map(|(asset, file)| (asset.symbol.clone(), file.bars()))
        .collect();
    CostCalibration::from_series(&series, 300, VALIDATION_START_MS)
}

/// A decision frame, the tape frame it marks against, and the tape quote each book quote occupies.
struct Decision {
    frame: u32,
    book: BookFrame,
    slots: Vec<u32>,
}

/// The frames the book decides on, reusable across every [`BookConfig`] in the design sweep: a
/// forecast path is a property of the checkpoint and the origin, not of the policy reading it.
fn decisions(tape: &Tape, cached: &[CachedFrame]) -> Vec<Decision> {
    let horizons = BOOK_HORIZONS.to_vec();
    cached
        .iter()
        .map(|frame| {
            let marks = &tape.frames[frame.frame as usize].quotes;
            Decision {
                frame: frame.frame,
                book: BookFrame {
                    timestamp_ms: frame.timestamp_ms,
                    quotes: frame
                        .rows
                        .iter()
                        .map(|row| BookQuote {
                            asset: row.asset as usize,
                            sigma: f64::from(row.sigma),
                            price: marks[row.quote as usize].close,
                            adv_usd: f64::from(row.adv_usd),
                            path: Some(HorizonPath {
                                mean: row.mean.to_vec(),
                                std: row.std.to_vec(),
                                horizons: horizons.clone(),
                            }),
                        })
                        .collect(),
                },
                slots: frame.rows.iter().map(|row| row.quote).collect(),
            }
        })
        .collect()
}

/// The share of a full-size per-name position the account's minimum-trade floor may occupy in the
/// book path. Above it the entry that builds the position cannot clear the floor, so the floor
/// stops refusing churn and starts refusing the book.
const MIN_TRADE_ENTRY_FRACTION: f64 = 0.25;

/// The tape the account replays under one [`BookConfig`], plus what the book ASKED for before the
/// account's turnover, participation, margin and integer-share machinery got a vote. The gap
/// between the two is the only way to tell a book that made no money from a book that was never
/// allowed to trade.
struct Book {
    tape: Tape,
    summary: BTreeMap<String, f64>,
    /// `(frame index, selected, banded)` per decision: the first two of the four breadth stages,
    /// carried per frame so the stage that turns a 256-name decile book into a three-name book is
    /// readable off one chart rather than inferred from a mean.
    breadth: Vec<(u32, f64, f64)>,
}

/// One-way execution cost as a fraction of notional, per name and per decision.
///
/// The measured panel's `fixed_bps` is half-spread plus commission plus regulatory fees; impact
/// is deliberately excluded because the band is a per-name threshold that must not depend on the
/// order size it is deciding whether to place. The flat arm adds the per-share commission divided
/// by price, which is the whole cross-sectional cost effect a flat bps assumption erases.
fn one_way_cost_fraction(
    model: Option<&BarCostModel>,
    config: &PortfolioConfig,
    symbol: u32,
    timestamp_ms: i64,
    price: f64,
) -> f64 {
    match model {
        Some(model) => model.resolve(symbol, timestamp_ms).fixed_bps() / 10_000.0,
        None => {
            (config.spread_bps * 0.5 + config.slippage_bps) / 10_000.0
                + if price > 0.0 {
                    config.commission_per_share / price
                } else {
                    0.0
                }
        }
    }
}

fn build_book(
    tape: &Tape,
    frames: &[Decision],
    cfg: &BookConfig,
    config: &PortfolioConfig,
    costs: &CostSource,
) -> Book {
    let model = match costs {
        CostSource::Measured(calibration) => Some(BarCostModel::new(Arc::clone(calibration))),
        CostSource::Flat => None,
    };
    let horizon_ms = cfg.trade_horizon as i64 * RESOLUTION_MS;
    let mut stamped = tape.clone();
    let mut previous = vec![0.0f64; tape.assets.len()];
    let mut current = vec![0.0f64; tape.assets.len()];
    let (mut gross, mut net, mut active, mut vol) = (0.0, 0.0, 0.0, 0.0);
    let (mut requested, mut bound, mut empty) = (0.0, 0usize, 0usize);
    let (mut held, mut considered) = (0usize, 0usize);
    let (mut selected_total, mut banded_total) = (0.0, 0.0);
    let mut breadth = Vec::with_capacity(frames.len());
    for decision in frames {
        let target = book::target_weights(&decision.book, cfg);
        current.fill(0.0);
        let expires_ms = decision.book.timestamp_ms + horizon_ms;
        let quotes = &mut stamped.frames[decision.frame as usize].quotes;
        for (slot, (quote, weight)) in decision.slots.iter().zip(&target.weights).enumerate() {
            let source = &decision.book.quotes[slot];
            let edge = source
                .path
                .as_ref()
                .and_then(|path| book::name_edge(path, source.sigma, cfg));
            let mut chosen = *weight;
            if let Some(edge) = edge {
                // The no-trade region itself lives in `book::banded_weight`; it is a property of
                // the book's weights, not of the replay, and it is tested there against the
                // all-zero starting book that the old `per_name_cap` clamp made unenterable.
                if config.risk_aversion > 0.0 && edge.risk_var > 0.0 {
                    let one_way = one_way_cost_fraction(
                        model.as_ref(),
                        config,
                        tape.assets[source.asset].symbol_index,
                        decision.book.timestamp_ms,
                        source.price,
                    );
                    let banded = book::banded_weight(
                        chosen,
                        previous[source.asset],
                        &edge,
                        one_way,
                        config.risk_aversion,
                        cfg,
                    );
                    considered += 1;
                    held += usize::from(banded != chosen);
                    chosen = banded;
                }
                // Sizing authority stays with `target`; this is the alpha and risk estimate the
                // account's own utility needs to decide whether moving to that target is worth
                // its cost. RAW units, because that utility adds `value_usd * mean` to a cost in
                // dollars and penalises `(value_usd * std)^2`: a sigma-unit pair leaves the two
                // terms a full factor of `1/sigma` apart and the risk penalty swamps the alpha.
                quotes[*quote as usize].forecast = Some(Forecast {
                    mean: edge.mu,
                    std: edge.risk_var.sqrt().max(f64::MIN_POSITIVE),
                    expires_ms,
                });
            }
            quotes[*quote as usize].target = Some(chosen);
            current[source.asset] = chosen;
        }
        let selected = target.active as f64;
        let banded = current.iter().filter(|weight| **weight != 0.0).count() as f64;
        selected_total += selected;
        banded_total += banded;
        breadth.push((decision.frame, selected, banded));
        let turnover: f64 = current
            .iter()
            .zip(&previous)
            .map(|(now, before)| (now - before).abs())
            .sum();
        requested += turnover;
        bound += usize::from(turnover > config.max_turnover);
        empty += usize::from(target.active == 0);
        gross += target.gross;
        net += target.net;
        active += target.active as f64;
        vol += target.ex_ante_vol_annual;
        previous.copy_from_slice(&current);
    }
    let count = frames.len().max(1) as f64;
    let mut summary = BTreeMap::new();
    for (key, value) in [
        ("book_decision_count", frames.len() as f64),
        ("book_mean_gross_fraction", gross / count),
        ("book_mean_net_fraction", net / count),
        ("book_mean_active_count", active / count),
        ("book_mean_ex_ante_vol_annual", vol / count),
        ("book_requested_turnover_fraction", requested / count),
        (
            "book_requested_turnover_cap_bound_fraction",
            bound as f64 / count,
        ),
        ("book_empty_target_fraction", empty as f64 / count),
        (
            "book_no_trade_band_held_fraction",
            held as f64 / considered.max(1) as f64,
        ),
        ("book_no_trade_band_held_count", held as f64),
        ("book_breadth_selected", selected_total / count),
        ("book_breadth_banded", banded_total / count),
    ] {
        record(&mut summary, key, value);
    }
    Book {
        tape: stamped,
        summary,
        breadth,
    }
}

fn tier_label(tier: CommissionTier) -> &'static str {
    match tier {
        CommissionTier::TieredRemove => "tiered-remove",
        CommissionTier::TieredAdd => "tiered-add",
        CommissionTier::Fixed => "fixed",
    }
}

fn design_label(cfg: &BookConfig) -> String {
    let aggregation = match cfg.aggregation {
        Aggregation::Single => "single",
        Aggregation::Uniform => "uniform",
        Aggregation::Precision => "precision",
        Aggregation::Exponential => "exponential",
    };
    let weighting = match cfg.weighting {
        Weighting::Equal => "equal",
        Weighting::Mean => "mean",
        Weighting::MeanOverVar => "mean-over-var",
        Weighting::MeanOverVol => "mean-over-vol",
    };
    format!("{aggregation}/{weighting}")
}

/// One (variant, AUM) cell of the answer to "how much does this make on an IBKR Pro account".
///
/// `cap_bound` and the four breadth stages are here because a flat P&L has two completely
/// different causes that the P&L alone cannot separate: a book that traded and made nothing, and
/// a book that was never allowed to carry names. The four counts localise the second case to one
/// stage - selection, the book's no-trade band, the account's feasibility search, or its ascent.
struct Row {
    label: String,
    aum: f64,
    pnl: f64,
    net_return: f64,
    annual_return: f64,
    sharpe: f64,
    drawdown: f64,
    annual_turnover: f64,
    cost_bps: f64,
    cap_bound: f64,
    selected: f64,
    banded: f64,
    requested: f64,
    ramped: f64,
    held: f64,
    trimmed: f64,
    suppressed: f64,
    requested_turnover: f64,
    band_held: f64,
    ex_ante_vol: f64,
    min_trade: f64,
}

fn row(name: &str, aum: f64, evaluation: &AccountEvaluation) -> Row {
    let get = |key: &str| evaluation.summary.get(key).copied().unwrap_or(f64::NAN);
    let either = |primary: &str, fallback: f64| {
        let value = get(primary);
        if value.is_finite() {
            value
        } else {
            fallback
        }
    };
    // The same measurement the report panels annualize with, so the table's `ann_ret` and the
    // charted `annualized net return` cannot disagree about the window's length.
    let span = evaluation.span_years();
    let net_return = get("total_return_fraction");
    let turnover = either("traded_notional_usd", get("turnover_usd"));
    Row {
        label: name.to_owned(),
        aum,
        pnl: either("net_pnl_usd", get("equity_usd") - aum),
        net_return,
        annual_return: (1.0 + net_return).powf(1.0 / span) - 1.0,
        sharpe: get("daily_sharpe_ratio"),
        drawdown: get("max_drawdown_fraction"),
        annual_turnover: either("turnover_annualized", turnover / aum / span),
        cost_bps: either(
            "cost_bps_of_traded_notional",
            get("costs_usd") / turnover * 10_000.0,
        ),
        cap_bound: either(
            "turnover_cap_bound_fraction",
            get("book_requested_turnover_cap_bound_fraction"),
        ),
        selected: get("book_breadth_selected"),
        banded: get("book_breadth_banded"),
        requested: get("book_breadth_requested"),
        ramped: get("book_breadth_ramped"),
        held: either("book_breadth_held", get("mean_active_names")),
        trimmed: get("book_ascent_trim_fraction"),
        suppressed: get("book_min_trade_suppressed_fraction"),
        requested_turnover: get("book_requested_turnover_fraction"),
        band_held: get("book_no_trade_band_held_fraction"),
        ex_ante_vol: get("book_mean_ex_ante_vol_annual"),
        min_trade: get("book_min_trade_usd"),
    }
}

fn print_summary(rows: &[Row]) {
    println!(
        "\n{:<24} {:>13} {:>14} {:>9} {:>9} {:>8} {:>8} {:>9} {:>9} {:>9} {:>9} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>10}",
        "variant",
        "aum_usd",
        "net_pnl_usd",
        "net_ret",
        "ann_ret",
        "sharpe",
        "max_dd",
        "ann_turn",
        "cost_bps",
        "cap_bound",
        "req_turn",
        "bandhold",
        "sel",
        "band",
        "req",
        "ramp",
        "held",
        "trim",
        "supp",
        "exante",
        "min_trade"
    );
    for row in rows {
        println!(
            "{:<24} {:>13.0} {:>14.2} {:>8.2}% {:>8.2}% {:>8.3} {:>7.2}% {:>9.2} {:>9.2} {:>8.1}% {:>9.3} {:>6.1}% {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>7.1} {:>6.1}% {:>6.1}% {:>7.1}% {:>10.2}",
            row.label,
            row.aum,
            row.pnl,
            row.net_return * 100.0,
            row.annual_return * 100.0,
            row.sharpe,
            row.drawdown * 100.0,
            row.annual_turnover,
            row.cost_bps,
            row.cap_bound * 100.0,
            row.requested_turnover,
            row.band_held * 100.0,
            row.selected,
            row.banded,
            row.requested,
            row.ramped,
            row.held,
            row.trimmed * 100.0,
            row.suppressed * 100.0,
            row.ex_ante_vol * 100.0,
            row.min_trade
        );
    }
    println!(
        "\ncost_bps is charged cost over one-way traded notional; ann_turn is one-way traded notional per unit of initial equity per year; cap_bound is the share of decision frames whose requested turnover exceeded the cap and req_turn the one-way turnover the book asked for per decision, directly comparable to max_turnover. bandhold is the share of name-decisions the book's no-trade band froze. The breadth stages are mean names per decision frame: sel = nonzero target weights out of the book's selection, band = still nonzero after the no-trade band, req = nonzero desired share counts entering the account's feasibility search, ramp = still nonzero after the largest uniformly feasible scaling, held = nonzero positions after the ascent. trim is the share of requested names the ascent zeroes and supp the share the minimum-trade floor would suppress at full requested scale. exante is the book's diagonal ex-ante annual vol before the account sees it; min_trade is the dollar floor the account was given."
    );
}

/// Continuous overlapping-tranche market-neutral book on the validation partition.
pub fn evaluate_book(args: BookEvaluateArgs) -> Result<()> {
    let started = Instant::now();
    ensure!(args.batch_size > 0, "batch size must be positive");
    ensure!(
        args.book_rebalance_bars > 0,
        "book decision cadence must be positive"
    );
    ensure!(
        !args.aum_sweep.is_empty()
            && args
                .aum_sweep
                .iter()
                .all(|aum| aum.is_finite() && *aum > 0.0),
        "the AUM sweep must be a nonempty list of positive account sizes"
    );
    // A repeated account size would produce two runs with the identical (label, AUM) key, which
    // the report writer refuses rather than silently averaging into one series.
    let mut sweep: Vec<f64> = Vec::with_capacity(args.aum_sweep.len());
    for aum in &args.aum_sweep {
        if !sweep.iter().any(|seen| seen.to_bits() == aum.to_bits()) {
            sweep.push(*aum);
        }
    }
    // Rejected here rather than inside `simulate`: the account's gross ceiling is raised to
    // admit the book's own cap, so an inadmissible book cap would surface as a confusing
    // complaint about a flag the operator never set.
    ensure!(
        args.book.gross_cap.is_finite()
            && args.book.gross_cap > args.account.max_net
            && args.book.gross_cap <= 4.0,
        "book gross cap {} must exceed the net cap {} and stay at or below the 4.0 Reg-T intraday limit",
        args.book.gross_cap,
        args.account.max_net
    );
    let (manifest, corpus, _store, model, device) =
        load_checkpoint(&args.checkpoint, &args.data_dir)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    let pairing = manifest.pairing(&corpus.contract)?;
    ensure!(
        args.book.trade_horizon > 0 && args.book.trade_horizon <= corpus.contract.pred_len,
        "trade horizon {} is outside the checkpoint's {} emitted horizons",
        args.book.trade_horizon,
        corpus.contract.pred_len
    );
    ensure!(
        BOOK_HORIZONS
            .iter()
            .all(|h| usize::from(*h) <= corpus.contract.pred_len),
        "the stored horizon subset exceeds the checkpoint's emitted horizons"
    );

    let preparing = Instant::now();
    let metadata: Option<portfolio_data::Metadata> = args
        .schedule
        .risk_metadata
        .as_ref()
        .map(|path| -> Result<_> { Ok(serde_json::from_slice(&fs::read(path)?)?) })
        .transpose()?;
    let mut prepared = portfolio_data::plan(
        &corpus,
        &args.schedule,
        args.book.trade_horizon,
        args.book_rebalance_bars,
        1,
        metadata.as_ref(),
    )?;
    let mut summary = std::mem::take(&mut prepared.summary);
    let mut assumptions = std::mem::take(&mut prepared.assumptions);
    record(
        &mut summary,
        "preparation_ms",
        preparing.elapsed().as_secs_f64() * 1000.,
    );
    if let Some(metadata) = &metadata {
        let mut symbols = std::collections::BTreeSet::new();
        ensure!(
            metadata
                .assets
                .iter()
                .all(|row| !row.symbol.is_empty() && symbols.insert(&row.symbol)),
            "risk metadata contains an empty or duplicate symbol"
        );
        ensure!(
            metadata.as_of_ms < prepared.tape.frames[0].timestamp_ms,
            "risk metadata must have an explicit as-of timestamp before evaluation"
        );
        assumptions.push(format!("Sector and short availability metadata is a frozen external snapshot as of {}, assumed unchanged for this window. It is not a historical locate feed.", metadata.as_of_ms));
    } else {
        assumptions.push("No sector or short-availability metadata supplied, so no sector facts are invented and every asset is short-unavailable unless the explicit assumed-short override is enabled. A market-neutral book without borrow is a long-only book: check the shortable count before reading any number here as a long/short result.".into());
    }
    if args.account.allow_assumed_short {
        assumptions.push("ASSUMED SHORT AVAILABILITY OVERRIDE: hypothetical borrow access for every selected asset, not verified broker locates. Borrow cost follows the stated flat annual scenario.".into());
    }

    // The book, not the legacy account defaults, is the policy: the account's horizon-dependent
    // terms must describe the holding period the weights were built for, its name limit must not
    // silently truncate a decile, and its gross ceiling must admit the gross the book targets.
    // Every override is stated in the assumptions rather than applied quietly.
    let mut account = args.account.clone();
    account.rebalance_bars = args.book_rebalance_bars;
    account.horizon = args.book.trade_horizon;
    account.max_names = account.max_names.max(prepared.tape.assets.len());
    account.max_gross = account.max_gross.max(args.book.gross_cap);
    account.validate()?;
    assumptions.push(format!("Book path account overrides, applied over the shared portfolio defaults and reported rather than silently applied: decision cadence {} bars, horizon {} bars, name limit {} (the book's own selection is the name limit; the legacy limit of {} would truncate a decile), gross ceiling {} (raised to admit the book's {} gross cap). Effective gross is min(book gross cap, account max gross) and effective per-name exposure is min({}, {}).",
        account.rebalance_bars, account.horizon, account.max_names, args.account.max_names, account.max_gross, args.book.gross_cap, args.book.per_name_cap, account.max_weight));
    // A dollar-neutral book funds its short leg out of its own cash: short proceeds are credited,
    // so cash stays at equity while the account requires cash >= (1 + initial_margin) * short.
    // With long = short = gross/2 that caps gross at 2/(1 + initial_margin) no matter what the
    // book's own gross cap says. A book targeting more than the ceiling is not levered, it is
    // permanently in scale-down, and the symptom is a feasibility ramp that deletes names every
    // single decision. Stated here rather than clamped: the ceiling is a margin policy, not a
    // sizing choice this path is entitled to make.
    let funding_ceiling = 2.0 / (1.0 + account.initial_margin);
    assumptions.push(format!("Dollar-neutral funding ceiling: with initial margin {} the account can carry at most {:.3} gross on a long/short book funded by its own short proceeds, against the book's {} gross cap. {}", account.initial_margin, funding_ceiling, args.book.gross_cap,
        if args.book.gross_cap > funding_ceiling {
            "The requested gross cap is ABOVE that ceiling and is therefore unreachable: every decision enters the account's feasibility ramp already infeasible and is scaled down, which costs breadth through integer-share rounding rather than costing leverage. Read the reported gross against the ceiling, not against the cap."
        } else {
            "The requested gross cap is within the ceiling, so financing does not bind the book."
        }));

    let binding = portfolio_data::digest(&postcard::to_stdvec(&(
        CACHE_FORMAT,
        &pairing,
        &args.schedule,
        args.book.trade_horizon,
        args.book_rebalance_bars,
        BOOK_HORIZONS,
        &metadata,
        &prepared.tape,
    ))?);
    let caching = Instant::now();
    let cache = args.book_tape_cache.as_ref();
    let hit = cache.is_some_and(|path| path.exists());
    let (tape, frames) = if let Some(path) = cache.filter(|path| path.exists()) {
        let cached: CachedBookTape = postcard::from_bytes(&fs::read(path)?)?;
        ensure!(
            cached.binding == binding,
            "book tape does not match checkpoint/corpus/schedule/horizon-subset inputs; use a new cache path"
        );
        ensure!(
            cached.payload_sha256
                == portfolio_data::digest(&postcard::to_stdvec(&(
                    &cached.tape,
                    &cached.frames,
                    &cached.summary
                ))?),
            "book tape payload authentication failed"
        );
        summary.extend(cached.summary);
        for key in ["book_inference_ms", "book_loader_wait_ms"] {
            summary.insert(key.into(), 0.);
        }
        (cached.tape, cached.frames)
    } else {
        // ONE forward pass over the origins the corpus can materialize a complete forecast
        // window for. An origin without one is refused rather than forecast under a projected
        // clock: the head conditions on the horizon's covariate tokens, and substituting a
        // schedule for the ticker's own next-observed bars measures a different quantity from
        // the one the label - and every reported metric of this checkpoint - is defined on.
        // In a sixty-session window at the head of the validation partition the refusals are
        // names whose corpus history ENDS inside the horizon, which carry no forward marks to
        // book a P&L against either.
        let admissible: Vec<usize> = (0..prepared.origins.len())
            .filter(|row| {
                let reference = prepared.origins[*row];
                corpus
                    .ticker(reference)
                    .owns_partition_targets(reference.origin, 1)
            })
            .collect();
        ensure!(
            !admissible.is_empty(),
            "no decision origin owns a complete validation forecast window"
        );
        let mut mean = vec![[f32::NAN; HORIZONS]; prepared.origins.len()];
        let mut std = vec![[f32::NAN; HORIZONS]; prepared.origins.len()];
        let mut realized = vec![[f32::NAN; HORIZONS]; prepared.origins.len()];
        let mut sigma = vec![0f32; prepared.origins.len()];
        let subset: Vec<WindowRef> = admissible
            .iter()
            .map(|row| prepared.origins[*row])
            .collect();
        let emission = forecast(&corpus, &model, &subset, args.batch_size, device)?;
        summary.extend(emission.summary);
        for (slot, row) in admissible.iter().enumerate() {
            mean[*row] = emission.mean[slot];
            std[*row] = emission.std[slot];
            realized[*row] = emission.realized[slot];
            sigma[*row] = emission.sigma[slot];
        }
        record(
            &mut summary,
            "forecast_window_refused_origins_count",
            (prepared.origins.len() - admissible.len()) as f64,
        );
        let adv = trailing_adv(&corpus, &prepared.origins);
        let mut grouped: BTreeMap<usize, Vec<CachedRow>> = BTreeMap::new();
        let mut rejected = 0usize;
        for (row, (frame, quote)) in prepared.destinations.iter().copied().enumerate() {
            if !(sigma[row].is_finite() && sigma[row] > 0.0)
                || mean[row].iter().any(|value| !value.is_finite())
                || std[row].iter().any(|value| !(*value > 0.0))
            {
                rejected += 1;
                continue;
            }
            grouped.entry(frame).or_default().push(CachedRow {
                quote: quote as u32,
                asset: prepared.tape.frames[frame].quotes[quote].asset as u32,
                sigma: sigma[row],
                adv_usd: adv[row] as f32,
                mean: mean[row],
                std: std[row],
                realized: realized[row],
            });
        }
        record(&mut summary, "nonfinite_emission_count", rejected as f64);
        let frames: Vec<CachedFrame> = grouped
            .into_iter()
            .map(|(frame, rows)| CachedFrame {
                frame: frame as u32,
                timestamp_ms: prepared.tape.frames[frame].timestamp_ms,
                rows,
            })
            .collect();
        ensure!(
            !frames.is_empty(),
            "no decision frame survived the finite-emission screen"
        );
        let tape = std::mem::replace(
            &mut prepared.tape,
            Tape {
                assets: Vec::new(),
                frames: Vec::new(),
            },
        );
        if let Some(path) = cache {
            let payload_sha256 =
                portfolio_data::digest(&postcard::to_stdvec(&(&tape, &frames, &summary))?);
            let cached = CachedBookTape {
                binding,
                payload_sha256,
                tape,
                frames,
                summary: summary.clone(),
            };
            portfolio_data::atomic_write(path, &postcard::to_stdvec(&cached)?)?;
            (cached.tape, cached.frames)
        } else {
            (tape, frames)
        }
    };
    record(
        &mut summary,
        "book_tape_cache_hit_count",
        f64::from(u8::from(hit)),
    );
    record(
        &mut summary,
        "book_tape_load_build_ms",
        caching.elapsed().as_secs_f64() * 1000.,
    );
    record(&mut summary, "book_horizon_count", HORIZONS as f64);
    summary.retain(|_, value| value.is_finite());

    let reproduction = if args.diagnostics_universe == DiagnosticsUniverse::Full {
        let measuring = Instant::now();
        let measured = full_universe_ic(&corpus, &model, args.batch_size, device)?;
        record(
            &mut summary,
            "diagnostics_full_universe_ms",
            measuring.elapsed().as_secs_f64() * 1000.,
        );
        Some(measured)
    } else {
        None
    };

    // Diagnostics run BEFORE the book frames are materialized: both hold a copy of every stored
    // path, and holding two at once doubles the tape's footprint for no reason.
    let diagnostics = if args.diagnostics {
        let measuring = Instant::now();
        let horizons = BOOK_HORIZONS.to_vec();
        let mut panel = Vec::with_capacity(frames.len());
        let mut labels = Vec::with_capacity(frames.len());
        for frame in &frames {
            let mut paths = Vec::with_capacity(frame.rows.len());
            let mut realized = Vec::with_capacity(frame.rows.len());
            for row in &frame.rows {
                paths.push((
                    row.asset as usize,
                    f64::from(row.sigma),
                    HorizonPath {
                        mean: row.mean.to_vec(),
                        std: row.std.to_vec(),
                        horizons: horizons.clone(),
                    },
                ));
                if row.realized.iter().any(|value| value.is_finite()) {
                    realized.push((row.asset as usize, row.realized.to_vec()));
                }
            }
            panel.push((frame.timestamp_ms, paths));
            labels.push((frame.timestamp_ms, realized));
        }
        let measured = book_diagnostics::measure(
            &panel,
            &labels,
            &args.book,
            reproduction
                .as_ref()
                .map(|ic| (BOOK_HORIZONS.as_slice(), ic)),
        )?;
        record(
            &mut summary,
            "diagnostics_ms",
            measuring.elapsed().as_secs_f64() * 1000.,
        );
        for (key, value) in [
            (
                "diagnostics_cross_sections_count",
                measured.cross_sections as f64,
            ),
            ("diagnostics_aggregated_ic", measured.aggregated_ic),
            ("diagnostics_aggregated_ic_se", measured.aggregated_ic_se),
            (
                "diagnostics_aggregated_hit_rate",
                measured.aggregated_hit_rate,
            ),
        ] {
            record(&mut summary, key, value);
        }
        Some(measured)
    } else {
        None
    };

    if let Some(measured) = &reproduction {
        println!(
            "\nreproduction: per-horizon cross-sectional IC on the in-run scorer's OWN population\n\
             (every Corpus::validation_refs origin, grouped by shared origin timestamp, gated at\n\
             20 names) against the same measurement on the traded book universe."
        );
        for (index, horizon) in BOOK_HORIZONS.iter().enumerate() {
            record(
                &mut summary,
                &format!("diagnostics_full_universe_ic_h{horizon}"),
                measured.ic[index],
            );
            record(
                &mut summary,
                &format!("diagnostics_full_universe_ic_se_h{horizon}"),
                measured.se[index],
            );
            record(
                &mut summary,
                &format!("diagnostics_full_universe_cross_sections_h{horizon}"),
                measured.cross_sections[index] as f64,
            );
            let book = diagnostics
                .as_ref()
                .map_or(f64::NAN, |d| d.ic_by_horizon[index]);
            println!(
                "  h{horizon:<4} full-universe IC {:+.4} (se {:.4}, {} cross-sections)   book-universe IC {book:+.4}",
                measured.ic[index], measured.se[index], measured.cross_sections[index]
            );
        }
    }

    let costs = if args.measured_costs {
        let measuring = Instant::now();
        let calibration = measure_costs(&tape, &args.data_dir)?;
        for (key, value) in [
            (
                "cost_calibration_ms",
                measuring.elapsed().as_secs_f64() * 1000.,
            ),
            (
                "cost_unmeasured_spread_count",
                calibration.unmeasured.len() as f64,
            ),
            ("cost_fallback_spread_bps", calibration.fallback_spread_bps),
            ("cost_fallback_adv_usd", calibration.fallback_adv_usd),
        ] {
            record(&mut summary, key, value);
        }
        assumptions.push(format!("Execution cost is MEASURED, not assumed: per-symbol Roll (1984) half-spread and square-root impact from a {}-name panel fitted strictly on bars before {}, the start of the validation partition, so no cost estimate reads a bar the backtest trades on. Corwin-Schultz is not used - it measures negative in all ten liquidity deciles at five-minute sampling, which is a statement about the estimator at this resolution and not about the spreads. {} symbols had no measurable pooled spread and are priced at the {:.3} bps cross-sectional median.", tape.assets.len(), VALIDATION_START_MS, calibration.unmeasured.len(), calibration.fallback_spread_bps));
        CostSource::Measured(Arc::new(calibration))
    } else {
        assumptions.push(format!("Execution cost is the FLAT scenario: {} bps full spread and {} bps additional slippage per side, identical for every name and size. This is the control arm for the measured panel, not a claim about realized fills.", account.spread_bps, account.slippage_bps));
        CostSource::Flat
    };

    let built = decisions(&tape, &frames);
    drop(frames);
    assumptions.push(format!("The book is CONTINUOUS and overlapping: a decision every {} completed five-minute bars over {} decision frames, with target weights recomputed from the CURRENT forecast path each time. A name that stays inside its selected set therefore requires no trade, which is the whole difference from the disjoint endpoint cohorts the 39.5 bps decile spread was measured on. Realized holding time is a distribution, not the nominal {} bars.", args.book_rebalance_bars, built.len(), args.book.trade_horizon));
    assumptions.push("Sizing reads the RAW un-gained head emission (`CausalPatchModel::output`), never the calibrated `decode`. The checkpoint's frozen mean gain measured non-positive at this holding horizon and is clamped to zero, which sizes an amplitude-affine allocator to nothing; a cross-sectional rank is invariant to any positive scalar, so there is no amplitude for a gain to supply and the sizing gate's premise does not hold. The edge being traded is cross-sectional RANK, not level: the close MSE ratio against persistence is 0.991 at h64 and 1.007 at h128, so level accuracy is thin while rank IC is 0.062 at h64 against a 0.0030 standard error.".into());
    assumptions.push("The forward pass is conditioned on the LABEL CLOCK: the auxiliary covariate tokens over the forecast horizon carry the calendar of the ticker's actual next-observed bars (`Corpus::host_batch`), which is the conditioning the checkpoint was trained under and the conditioning every reported metric of it - including the 0.0180/0.0473/0.0616 per-horizon IC at h1/h32/h64 this pass is checked against - is measured under. Substituting the deterministic 04:00-20:00 five-minute projection (`Corpus::host_forecast_batch`) is NOT a neutral change: the two schedules agree exactly out to h64 and then diverge by hours, and feeding the projection measures h32 IC +0.001 and h64 +0.004 on this universe against +0.046 and +0.063 under the label clock, and turns the scorer's own full-population +0.047/+0.062 into -0.002/-0.013. That is a live-trading question, not a backtest one - a deployment cannot know a name's future print times and would have to project them per ticker - and it is stated here rather than buried because the projected-clock arm has NO measurable edge at all.".into());
    assumptions.push(format!("Predicted dispersion is used for cross-horizon precision weighting, ex-ante vol targeting ({:.1}% annual, 0 disables), inverse-variance weights inside the selected set, and a cost-aware no-trade band at {}x the round-trip cost. It is NOT treated as alpha: at a fixed horizon the mean and the standard deviation carry the same sqrt(h)*sigma scaling, so ranking on mean/std measured the same Sharpe as ranking on mean (0.181 against 0.182), and a one-sigma gate fired on 0.12% of names.", args.book.target_vol_annual * 100.0, args.book.no_trade_band_cost_multiple));
    assumptions.push(format!("Each decision quote carries BOTH the target weight and the aggregated edge as a RAW-unit forecast (mean = mu, dispersion = sqrt(risk_var), expiring {} bars later). Raw and not sigma units because the account's utility adds `position_value_usd * mean` to a cost in dollars and penalises `(position_value_usd * dispersion)^2`: a sigma-unit pair leaves alpha and risk a full factor of 1/sigma_k apart and the risk term swamps the cross-section. The dispersion is the predicted RETURN variance at the trade horizon, read off the head nearest that horizon, NOT the precision-aggregated estimator variance, which is the dispersion of an average over {} overlapping heads and is two orders of magnitude smaller. The target weight is the sizing authority - the amplitude mean-variance branch never runs - and the forecast exists so the account's own utility, its risk-aversion variance penalty and its cost term can decide whether moving to that target is worth paying for. With no forecast those three terms are identically zero and the allocator would either liquidate the book or accept every move for free.", args.book.trade_horizon, args.book.agg_max_horizon - args.book.agg_min_horizon + 1));
    assumptions.push(format!("The cost-aware no-trade band is applied at the BOOK, before the tape reaches the account: a name holds its previous weight unless the requested change exceeds 4*{}*one_way_cost/(risk_aversion*risk_var), which is where the quadratic risk term's utility gain overtakes the round trip. It is derived from the quadratic term because a linear alpha scales gain and cost identically in trade size and therefore has no interior no-trade region at all. The one-way cost is the measured panel's half-spread plus commission plus regulatory fees when measured costs are on, and the flat spread/slippage scenario plus per-share commission over price otherwise; order-size impact is excluded so a per-name threshold cannot depend on the order it is gating. The band is clamped to HALF THE NAME'S OWN TARGET WEIGHT, not to the per-name cap {}: no weight may exceed the per-name cap, so a band clamped there is unsatisfiable and freezes every name at the weight it holds, which starts at zero. Against half the target, an entry from flat, a full exit and any sign flip clear the band by construction while a rebalance smaller than half the name's own size is refused.", args.book.no_trade_band_cost_multiple, args.book.per_name_cap));
    assumptions.push("Validation carries checkpoint-selection exposure: this is strategy development evidence, not a terminal holdout. The terminal test partition is locked and no part of this path reads it.".into());
    assumptions.push(format!("The account's minimum-trade floor is lowered in the book path to at most {}x the measured per-name notional (AUM x mean gross / mean selected names) at each account size, and the floor actually used is reported per run as book_min_trade_usd next to book_name_notional_usd. A floor above a fraction of a full-size position is not a churn filter, it is a refusal to build the book: the entry that establishes the position cannot clear it, so the account would report zero held names with no other symptom. Where the unlowered floor still exceeds the notional the book can put on, that is a real capacity limit of the account size and is visible as the gap between the two reported numbers.", MIN_TRADE_ENTRY_FRACTION));

    let middle = sweep[sweep.len() / 2];
    let mut variants: Vec<(String, f64, CommissionTier, BookConfig)> = Vec::new();
    for aum in &sweep {
        for tier in [
            CommissionTier::TieredRemove,
            CommissionTier::TieredAdd,
            CommissionTier::Fixed,
        ] {
            variants.push((tier_label(tier).to_owned(), *aum, tier, args.book.clone()));
        }
    }
    // The two design choices the book rests on, falsified rather than assumed: a single-horizon
    // read against the precision-weighted aggregate, and equal weights against inverse-variance
    // weights. Run at the middle account size on the tiered remove-liquidity schedule so the
    // comparison is not confounded by a different cost regime.
    for aggregation in [Aggregation::Single, Aggregation::Precision] {
        for weighting in [Weighting::Equal, Weighting::MeanOverVar] {
            let cfg = BookConfig {
                aggregation,
                weighting,
                ..args.book.clone()
            };
            variants.push((
                design_label(&cfg),
                middle,
                CommissionTier::TieredRemove,
                cfg,
            ));
        }
    }

    // Grouped by design, because a stamped tape is a full copy of every mark in the window and
    // only one needs to be resident at a time.
    let mut designs: Vec<(String, BookConfig, Vec<(String, f64, CommissionTier)>)> = Vec::new();
    for (name, aum, tier, cfg) in variants {
        let design = design_label(&cfg);
        match designs
            .iter()
            .position(|(existing, _, _)| *existing == design)
        {
            Some(index) => designs[index].2.push((name, aum, tier)),
            None => designs.push((design, cfg, vec![(name, aum, tier)])),
        }
    }
    let simulating = Instant::now();
    let mut runs: Vec<(String, f64, AccountEvaluation)> = Vec::new();
    let mut rows = Vec::new();
    for (_, cfg, cells) in &designs {
        let book = build_book(&tape, &built, cfg, &account, &costs);
        let read = |key: &str| book.summary.get(key).copied().unwrap_or(0.0);
        let (mean_gross, selected) = (
            read("book_mean_gross_fraction"),
            read("book_breadth_selected"),
        );
        // The same (design, size, fee schedule) cell can appear twice by construction - once in
        // the AUM sweep and once in the design sweep - and `simulate` is the expensive half.
        let mut seen: BTreeMap<(u64, &'static str), usize> = BTreeMap::new();
        for (name, aum, tier) in cells {
            let key = (aum.to_bits(), tier_label(*tier));
            if let Some(previous) = seen.get(&key) {
                let evaluation = runs[*previous].2.clone();
                rows.push(row(name, *aum, &evaluation));
                runs.push((name.clone(), *aum, evaluation));
                continue;
            }
            let mut config = account.clone();
            config.initial_cash = *aum;
            config.commission_tier = *tier;
            // A minimum-trade floor is there to refuse churn, not to refuse the book: a floor
            // above a fraction of a full-size position cannot be cleared by the entry that
            // builds that position, so the account would silently hold nothing at the small end
            // of the sweep. Lower it to that fraction of the measured per-name notional when the
            // AUM is small enough for the two to collide, and report both numbers so the
            // remaining capacity limit is visible rather than inferred.
            let name_notional = aum * mean_gross / selected.max(1.0);
            config.min_trade = account
                .min_trade
                .min(MIN_TRADE_ENTRY_FRACTION * name_notional);
            let mut evaluation = portfolio::simulate(&book.tape, &config, &costs)?;
            for (frame, selected, banded) in &book.breadth {
                let Some(point) = evaluation.points.get_mut(*frame as usize) else {
                    continue;
                };
                point
                    .values
                    .insert("breadth_selected_count".into(), *selected);
                point.values.insert("breadth_banded_count".into(), *banded);
            }
            record(
                &mut evaluation.summary,
                "book_min_trade_usd",
                config.min_trade,
            );
            record(
                &mut evaluation.summary,
                "book_name_notional_usd",
                name_notional,
            );
            evaluation.summary.extend(book.summary.clone());
            record(&mut evaluation.summary, "book_aum_usd", *aum);
            if let (Some(capped), Some(count)) = (
                evaluation.summary.get("turnover_capped_count").copied(),
                evaluation
                    .summary
                    .get("decision_count")
                    .copied()
                    .filter(|count| *count > 0.0),
            ) {
                record(
                    &mut evaluation.summary,
                    "turnover_cap_bound_fraction",
                    capped / count,
                );
            }
            evaluation.summary.extend(summary.clone());
            evaluation.assumptions.extend(assumptions.clone());
            rows.push(row(name, *aum, &evaluation));
            seen.insert(key, runs.len());
            runs.push((name.clone(), *aum, evaluation));
        }
    }
    let totals = BTreeMap::from([
        (
            "simulation_ms".to_owned(),
            simulating.elapsed().as_secs_f64() * 1000.,
        ),
        ("load_ms".to_owned(), load_ms),
        ("run_count".to_owned(), runs.len() as f64),
        ("design_count".to_owned(), designs.len() as f64),
        (
            "total_ms".to_owned(),
            started.elapsed().as_secs_f64() * 1000.,
        ),
    ]);
    for (_, _, evaluation) in runs.iter_mut() {
        evaluation.summary.extend(totals.clone());
    }

    // `report_cli` and the TUI both scan `<root>/gens/<n>/<base>.report.bin` (shared/src/run_dir.rs:78,
    // tui/src/main.rs:582), so a base written flat into `--output` is invisible to every reader.
    let gen = args.output.join("gens").join("0");
    fs::create_dir_all(&gen)
        .with_context(|| format!("create report generation directory {}", gen.display()))?;
    reports::write_book(
        &gen,
        manifest.epoch,
        manifest.step,
        &runs,
        diagnostics.as_ref(),
    )?;
    print_summary(&rows);
    if let Some(measured) = &diagnostics {
        for line in measured.verdict() {
            println!("diagnostic: {line}");
        }
        for warning in &measured.warnings {
            println!("diagnostic warning: {warning}");
        }
    }
    println!(
        "\nbook: {} runs over {} designs and {} account sizes, {} decision frames, {} names, {} costs, reports written to {}",
        runs.len(),
        designs.len(),
        sweep.len(),
        built.len(),
        tape.assets.len(),
        if args.measured_costs {
            "measured"
        } else {
            "flat"
        },
        args.output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::timexer_segment::features::FeatureSet;
    use shared::bars::{bar_file_path, write_bar_file, PackedBar};

    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn scratch(tag: &str) -> Scratch {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = PathBuf::from(format!(
            "/var/tmp/timexer-book-{tag}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }

    /// The defect this file was fixed for, in the one place it is observable without a
    /// checkpoint: the loader the book's forward pass runs on.
    ///
    /// `CausalPatchModel` conditions on auxiliary covariate tokens over the FORECAST horizon,
    /// and the label `CausalPatchModel::targets` reads is measured on the ticker's actual
    /// next-observed bars. The two must describe the same clock. `Corpus::host_forecast_batch`
    /// overwrites those tokens with a deterministic 04:00-20:00 five-minute projection, which
    /// is a different clock; running the book on it scored a forecast for one horizon axis
    /// against a label on another and measured a near-zero IC at every horizon.
    ///
    /// The assertion is on the SESSION-GAP channel because it is the channel that carries the
    /// clock: the fixture's bars are contiguous five-minute prints around the clock, so the
    /// label clock has no gaps at all, while any projection that skips 20:00 to 04:00 must
    /// manufacture one.
    #[test]
    fn the_book_forward_pass_is_conditioned_on_the_label_clock() {
        let directory = scratch("clock");
        let series = |offset: f32| -> Vec<PackedBar> {
            (0..8_000)
                .map(|index| {
                    let price = 100.0 + offset + index as f32 * 0.01;
                    PackedBar {
                        ts_ms: 1_500_000_000_000 + index * RESOLUTION_MS,
                        open: price,
                        high: price + 1.0,
                        low: price - 1.0,
                        close: price + 0.5,
                        volume: 1_000.0,
                        vwap: price,
                        trades: 10,
                    }
                })
                .collect()
        };
        for (symbol, offset) in [("AAA", 0.0), ("BBB", 5.0), ("CCC", 9.0)] {
            write_bar_file(
                &bar_file_path(&directory.0, symbol, 300),
                symbol,
                300,
                &series(offset),
            )
            .unwrap();
        }
        let features = FeatureSet {
            spy: false,
            ..FeatureSet::ALL
        };
        let (context, pred_len) = (16usize, 192usize);
        let corpus = Corpus::load(&directory.0, &[], context, pred_len, 32, &features, 1, 0)
            .expect("fixture corpus");
        let channels = features.channels();
        let reference = *corpus
            .validation_refs
            .first()
            .expect("fixture has a held-out origin");
        let gaps = |batch: &Batch| -> Vec<[f32; 2]> {
            let aux = Vec::<f32>::try_from(batch.aux.reshape([-1])).unwrap();
            (context..context + pred_len)
                .map(|bar| {
                    let base = bar * channels + 4;
                    [aux[base], aux[base + 1]]
                })
                .collect()
        };

        let source = corpus.ticker(reference);
        let expected: Vec<[f32; 2]> = (0..pred_len)
            .map(|step| {
                let bar = reference.origin + step;
                let delta = source.timestamp(bar + 1) - source.timestamp(bar);
                if delta > RESOLUTION_MS {
                    [1.0, (delta as f64 / RESOLUTION_MS as f64).ln() as f32]
                } else {
                    [0.0, 0.0]
                }
            })
            .collect();

        let projected = corpus.host_forecast_batch(&[reference], true).unwrap();
        let corpus = Arc::new(corpus);
        let loader = book_loader(&corpus);
        loader.request(&[reference]).unwrap();
        let (labelled, _) = loader.receive().unwrap();

        assert_eq!(
            gaps(&labelled),
            expected,
            "the book's forward pass must see the clock its label is measured on"
        );
        assert_ne!(
            gaps(&projected),
            expected,
            "the projected-schedule loader must remain observably a DIFFERENT conditioning; if \
             this ever passes the fixture stopped exercising the defect"
        );
    }
}
