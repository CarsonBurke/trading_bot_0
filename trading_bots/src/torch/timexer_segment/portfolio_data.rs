use super::{
    calibration::{FrozenGain, Pairing},
    corpus::{Batch, Corpus, CorpusTicker, RESOLUTION_MS, WindowRef},
    model::CausalPatchModel,
    portfolio::{self, AccountEvaluation, Asset, Forecast, Frame, Quote, Tape},
    portfolio_calibration,
    runner::{PortfolioEvaluateArgs, Prefetcher},
};
use anyhow::{Context, Result, ensure};
use chrono::{NaiveDate, TimeZone, Timelike};
use chrono_tz::America::New_York;
use clap::Args;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tch::{Device, Kind, Tensor};

// `v9`: the applied amplitude is the checkpoint's per-horizon two-coordinate calibration and
// the sizing gate reads its signed measurement, so a `v8` tape's forecasts were built from a
// different mean at the same checkpoint and the two are not interchangeable.
const CACHE_FORMAT: &str =
    "timexer-account-tape-v9-checkpoint-mean-gain-sizing-gate-bound-validation-origins";

#[derive(Clone, Debug, Args, Serialize, Deserialize)]
pub struct ScheduleConfig {
    /// Consecutive observed exchange sessions, not sparsely sampled endpoint windows.
    #[arg(long, default_value_t = 60)]
    pub sessions: usize,
    /// First exchange date (YYYY-MM-DD), inside validation. Default: first complete validation day.
    /// Validation has checkpoint-selection exposure; the terminal test remains locked.
    #[arg(long)]
    pub start_date: Option<NaiveDate>,
    /// Fixed universe selected by trailing regular-session dollar volume before the first decision.
    #[arg(long, default_value_t = 256)]
    pub universe_size: usize,
    /// Calendar days of strictly pre-evaluation history for liquidity, beta and correlation groups.
    #[arg(long, default_value_t = 40)]
    pub history_days: usize,
    /// Minimum observed regular-session history bars; no future survival requirement is imposed.
    #[arg(long, default_value_t = 1000)]
    pub min_history_bars: usize,
    #[arg(long, default_value_t = 5.0)]
    pub min_price: f64,
    /// Minimum mean historical regular-session dollar volume per five-minute bar.
    #[arg(long, default_value_t = 100_000.0)]
    pub min_bar_dollar_volume: f64,
    /// Absolute historical return correlation required to join an empirical risk group.
    #[arg(long, default_value_t = 0.65)]
    pub risk_correlation: f64,
    /// Optional JSON {as_of_ms, assets:[{symbol, sector?, shortable}]}; must predate evaluation.
    /// Missing symbols are not shortable. Availability is an explicit frozen metadata assumption.
    #[arg(long)]
    pub risk_metadata: Option<PathBuf>,
    /// Latest eligible synchronized sessions in the reserved calibration partition.
    #[arg(long, default_value_t = 5)]
    pub calibration_sessions: usize,
    /// Authenticated forecast/mark tape for policy replay. Existing mismatched tapes are refused.
    #[arg(long)]
    pub tape_cache: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Metadata {
    pub as_of_ms: i64,
    pub assets: Vec<MetadataAsset>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct MetadataAsset {
    pub symbol: String,
    #[serde(default)]
    pub sector: Option<String>,
    #[serde(default)]
    pub shortable: bool,
}

pub(super) struct Plan {
    pub tape: Tape,
    pub origins: Vec<WindowRef>,
    pub destinations: Vec<(usize, usize)>,
    pub summary: BTreeMap<String, f64>,
    pub assumptions: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct CachedTape {
    binding: String,
    payload_sha256: String,
    tape: Tape,
    diagnostics: BTreeMap<String, f64>,
}

#[derive(Clone, Serialize, Deserialize)]
struct ScalarCalibration {
    format: String,
    pairing: Pairing,
    fit_config_sha256: String,
    horizon: usize,
    gain: f64,
    first_origin_ms: i64,
    last_target_ms: i64,
    diagnostics: BTreeMap<String, f64>,
    sha256: String,
}

pub(super) fn digest(bytes: &[u8]) -> String {
    ring::digest::digest(&ring::digest::SHA256, bytes)
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(super) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let pending = path.with_extension(format!("pending-{}", std::process::id()));
    fs::write(&pending, bytes)?;
    fs::rename(pending, path)?;
    Ok(())
}

pub(super) fn clock(timestamp: i64) -> (NaiveDate, u32) {
    let time = New_York
        .timestamp_millis_opt(timestamp)
        .single()
        .expect("valid corpus timestamp");
    (time.date_naive(), time.hour() * 60 + time.minute())
}

fn at(date: NaiveDate, hour: u32, minute: u32) -> i64 {
    New_York
        .from_local_datetime(&date.and_hms_opt(hour, minute, 0).unwrap())
        .single()
        .expect("regular session is outside DST ambiguity")
        .timestamp_millis()
}

pub(super) fn lower_bound(ticker: &CorpusTicker, timestamp: i64) -> usize {
    let (mut low, mut high) = (0, ticker.contract.valid_bars);
    while low < high {
        let middle = low + (high - low) / 2;
        if ticker.timestamp(middle) < timestamp {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

pub(super) fn ticker(corpus: &Corpus, index: usize) -> &CorpusTicker {
    corpus.ticker(WindowRef {
        ticker: index,
        origin: 0,
    })
}

/// Calendar discovery does not rank names or select dates by returns, forecast quality or width.
/// The union admits a session even when only one corpus member traded; the universe stays fixed.
fn session_dates(
    corpus: &Corpus,
    first: usize,
    count: usize,
    requested: Option<NaiveDate>,
) -> Result<Vec<NaiveDate>> {
    let start = corpus.contract.boundary_timestamps[first];
    let end = corpus.contract.boundary_timestamps[first + 1];
    let first_date = requested.unwrap_or_else(|| clock(start).0.succ_opt().unwrap());
    ensure!(
        at(first_date, 9, 30) >= start && at(first_date, 9, 30) < end,
        "start date must be wholly inside the selected unlocked partition"
    );
    let mut dates = BTreeSet::new();
    if first == 0 {
        // The purge and complete-target ownership are fixed corpus rules, not a
        // forecast-quality screen. Use recent fit data rather than the partition's
        // oldest week, which can be a year behind the evaluation window.
        let last_origin = corpus
            .calibration_refs
            .iter()
            .map(|reference| corpus.ticker(*reference).timestamp(reference.origin))
            .max()
            .context("no eligible calibration origin")?;
        for index in 0..corpus.contract.tickers.len() {
            let source = ticker(corpus, index);
            let mut position = lower_bound(source, last_origin + 1);
            while position > 0 {
                let stamp = source.timestamp(position - 1);
                if stamp < at(first_date, 9, 30) {
                    break;
                }
                let date = clock(stamp).0;
                let open = lower_bound(source, at(date, 9, 30));
                let close = lower_bound(source, at(date, 16, 0));
                if open < close && source.owns_partition_targets(open, 0) {
                    dates.insert(date);
                    if dates.len() > count {
                        dates.pop_first();
                    }
                }
                if dates.len() == count && date <= *dates.first().unwrap() {
                    break;
                }
                position = lower_bound(source, at(date, 0, 0));
            }
        }
        ensure!(
            dates.len() == count,
            "requested {count} recent calibration sessions do not fit"
        );
        return Ok(dates.into_iter().collect());
    }
    // Only headers in the requested interval are visited; no OHLC outcome is consulted.
    for index in 0..corpus.contract.tickers.len() {
        let source = ticker(corpus, index);
        let mut position = lower_bound(source, at(first_date, 9, 30));
        let stop = lower_bound(source, end);
        while position < stop {
            let (date, minute) = clock(source.timestamp(position));
            if (570..960).contains(&minute) {
                dates.insert(date);
            }
            // One date per ticker is enough; binary search skips that day's remaining bars.
            position = lower_bound(source, at(date.succ_opt().unwrap(), 9, 30));
            if dates.len() >= count {
                let last = *dates.iter().nth(count - 1).unwrap();
                if date >= last {
                    break;
                }
            }
        }
    }
    let dates: Vec<_> = dates.into_iter().take(count).collect();
    ensure!(
        dates.len() == count,
        "requested {count} sessions do not fit in unlocked partition"
    );
    Ok(dates)
}

struct Candidate {
    ticker: usize,
    liquidity: f64,
    history: Vec<(i64, f64)>,
    beta: f64,
    risk_group: usize,
}

fn correlation(a: &[(i64, f64)], b: &[(i64, f64)]) -> Option<(f64, f64)> {
    let (mut n, mut x, mut y, mut xx, mut yy, mut xy) = (0., 0., 0., 0., 0., 0.);
    let (mut left, mut right) = (0, 0);
    while left < a.len() && right < b.len() {
        match a[left].0.cmp(&b[right].0) {
            std::cmp::Ordering::Less => left += 1,
            std::cmp::Ordering::Greater => right += 1,
            std::cmp::Ordering::Equal => {
                let (value, other) = (a[left].1, b[right].1);
                n += 1.;
                x += value;
                y += other;
                xx += value * value;
                yy += other * other;
                xy += value * other;
                left += 1;
                right += 1;
            }
        }
    }
    if n < 100. {
        return None;
    }
    let vx = xx - x * x / n;
    let vy = yy - y * y / n;
    let cov = xy - x * y / n;
    (vx > 0. && vy > 0.).then(|| (cov / (vx * vy).sqrt(), cov / vy))
}

pub(super) fn plan(
    corpus: &Corpus,
    config: &ScheduleConfig,
    horizon: usize,
    rebalance: usize,
    partition: usize,
    metadata: Option<&Metadata>,
) -> Result<Plan> {
    let days = if partition == 0 {
        config.calibration_sessions
    } else {
        config.sessions
    };
    let dates = session_dates(
        corpus,
        partition,
        days,
        if partition == 1 {
            config.start_date
        } else {
            None
        },
    )?;
    let start = at(dates[0], 9, 30);
    let stop = at(*dates.last().unwrap(), 16, 0);
    let history_start = start - config.history_days as i64 * 86_400_000;
    let mut summary = BTreeMap::new();
    let mut candidates = Vec::new();
    let mut eligible = 0;
    let mut missing_context = 0;
    let mut history_rejected = 0;
    let mut price_rejected = 0;
    let mut liquidity_rejected = 0;
    let mut stale_rejected = 0;
    let mut market: BTreeMap<i64, (f64, usize)> = BTreeMap::new();
    for index in 0..corpus.contract.tickers.len() {
        let source = ticker(corpus, index);
        let end = lower_bound(source, start);
        if end < corpus.contract.context.max(corpus.contract.common_context) {
            missing_context += 1;
            continue;
        }
        // Last completed information only. A future listing/delisting never enters this decision.
        if source.timestamp(end - 1) < start - 4 * 86_400_000 {
            stale_rejected += 1;
            continue;
        }
        let price = f64::from(source.bar(end - 1).close);
        if price < config.min_price {
            price_rejected += 1;
            continue;
        }
        let begin = lower_bound(source, history_start);
        let mut count = 0;
        let mut dollars = 0.;
        let mut history = Vec::with_capacity(end - begin);
        for position in begin..end {
            let bar = source.bar(position);
            let (_, minute) = clock(bar.ts());
            if !(570..960).contains(&minute) {
                continue;
            }
            let volume = f64::from(bar.volume);
            if volume.is_finite() && volume > 0. {
                dollars += f64::from(bar.close) * volume;
            }
            count += 1;
            if position > 0 && source.timestamp(position - 1) + RESOLUTION_MS == bar.ts() {
                let value = (f64::from(bar.close) / f64::from(source.bar(position - 1).close)).ln();
                history.push((bar.ts(), value));
                let entry = market.entry(bar.ts()).or_default();
                entry.0 += value;
                entry.1 += 1;
            }
        }
        if count < config.min_history_bars {
            history_rejected += 1;
            continue;
        }
        let liquidity = dollars / count as f64;
        if liquidity < config.min_bar_dollar_volume {
            liquidity_rejected += 1;
            continue;
        }
        candidates.push(Candidate {
            ticker: index,
            liquidity,
            history,
            beta: 0.,
            risk_group: 0,
        });
        eligible += 1;
        // Keep only a bounded liquid shortlist while scanning the entire population.
        if candidates.len() >= config.universe_size.saturating_mul(2) {
            candidates.sort_by(|a, b| {
                b.liquidity
                    .total_cmp(&a.liquidity)
                    .then(a.ticker.cmp(&b.ticker))
            });
            candidates.truncate(config.universe_size);
        }
    }
    candidates.sort_by(|a, b| {
        b.liquidity
            .total_cmp(&a.liquidity)
            .then(a.ticker.cmp(&b.ticker))
    });
    candidates.truncate(config.universe_size);
    ensure!(
        !candidates.is_empty(),
        "causal eligibility scan produced no liquid universe"
    );
    let market: Vec<_> = market
        .into_iter()
        .filter_map(|(stamp, (sum, count))| {
            (count >= corpus.contract.market_min_cross_section)
                .then_some((stamp, sum / count as f64))
        })
        .collect();
    let mut weak_beta = 0;
    for candidate in &mut candidates {
        if let Some((_, beta)) = correlation(&candidate.history, &market) {
            candidate.beta = beta;
        } else {
            weak_beta += 1;
            candidate.beta = 1.;
        }
    }
    // Connected components of the pre-window correlation graph: no sector labels are invented.
    let mut parent: Vec<usize> = (0..candidates.len()).collect();
    fn root(parent: &[usize], mut index: usize) -> usize {
        while parent[index] != index {
            index = parent[index];
        }
        index
    }
    let mut correlated_pairs = 0;
    for a in 0..candidates.len() {
        for b in 0..a {
            if correlation(&candidates[a].history, &candidates[b].history)
                .is_some_and(|(rho, _)| rho.abs() >= config.risk_correlation)
            {
                let (ra, rb) = (root(&parent, a), root(&parent, b));
                parent[ra] = rb;
                correlated_pairs += 1;
            }
        }
    }
    for (index, candidate) in candidates.iter_mut().enumerate() {
        candidate.risk_group = root(&parent, index);
    }
    let assets: Vec<Asset> = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let symbol = ticker(corpus, candidate.ticker).contract.ticker.clone();
            let metadata = metadata.and_then(|m| m.assets.iter().find(|row| row.symbol == symbol));
            Asset {
                symbol,
                sector: metadata.and_then(|m| m.sector.clone()),
                shortable: metadata.is_some_and(|m| m.shortable),
                beta: candidate.beta,
                risk_group: candidate.risk_group,
                // The measured cost panel is built in this same asset order, so an asset's
                // position in the universe IS its `CostCalibration::symbols` row.
                symbol_index: index as u32,
            }
        })
        .collect();
    let dates_set: BTreeSet<_> = dates.iter().copied().collect();
    let mut frames: BTreeMap<i64, Vec<(Quote, WindowRef)>> = BTreeMap::new();
    let terminal = corpus.contract.boundary_timestamps[partition + 1];
    // Retain every observed mark, including extended hours and gaps between decisions. Include
    // each member's first post-window bar; truncate to the earliest common-clock liquidation event.
    let mut liquidation = i64::MAX;
    for (asset, candidate) in candidates.iter().enumerate() {
        let source = ticker(corpus, candidate.ticker);
        let end = lower_bound(source, stop);
        if end < source.contract.valid_bars && source.timestamp(end) < terminal {
            liquidation = liquidation.min(source.timestamp(end));
        }
        for origin in lower_bound(source, start)..(end + 1).min(source.contract.valid_bars) {
            let bar = source.bar(origin);
            if bar.ts() >= terminal {
                break;
            }
            let volume = f64::from(bar.volume);
            frames.entry(bar.ts()).or_default().push((
                Quote {
                    asset,
                    open: f64::from(bar.open),
                    close: f64::from(bar.close),
                    volume: if volume.is_finite() && volume >= 0. {
                        volume
                    } else {
                        0.
                    },
                    forecast: None,
                    target: None,
                },
                WindowRef {
                    ticker: candidate.ticker,
                    origin,
                },
            ));
        }
    }
    ensure!(
        liquidation != i64::MAX,
        "window has no unlocked observed liquidation bar; choose an earlier start"
    );
    let mut origins = Vec::new();
    let mut destinations = Vec::new();
    let mut tape_frames = Vec::new();
    let mut widths = Vec::new();
    for (stamp, rows) in frames
        .into_iter()
        .take_while(|(stamp, _)| *stamp <= liquidation)
    {
        let (date, minute) = clock(stamp);
        let regular = (570..960).contains(&minute);
        let elapsed = minute.saturating_sub(570) as usize / 5;
        let last_day_room = date != *dates.last().unwrap() || minute as usize + horizon * 5 < 960;
        let decision = stamp < stop
            && dates_set.contains(&date)
            && regular
            && elapsed % rebalance == 0
            && last_day_room;
        let frame = tape_frames.len();
        let mut width = 0;
        let mut quotes = Vec::with_capacity(rows.len());
        for (quote, reference) in rows {
            // Evaluation never looks at future validity. Calibration alone requires owned labels.
            if decision
                && (partition == 1
                    || ticker(corpus, reference.ticker).owns_partition_targets(reference.origin, 0))
            {
                origins.push(reference);
                destinations.push((frame, quotes.len()));
                width += 1;
            }
            quotes.push(quote);
        }
        if decision {
            widths.push(width);
        }
        tape_frames.push(Frame {
            timestamp_ms: stamp,
            quotes,
            decision,
        });
    }
    ensure!(
        !origins.is_empty(),
        "synchronized schedule produced no forecast origins"
    );
    for (name, count) in [
        ("scanned_tickers_count", corpus.contract.tickers.len()),
        (
            "corpus_excluded_tickers_count",
            corpus.excluded_tickers.len(),
        ),
        ("eligible_tickers_count", eligible),
        ("selected_tickers_count", candidates.len()),
        ("context_rejected_count", missing_context),
        ("history_rejected_count", history_rejected),
        ("price_rejected_count", price_rejected),
        ("liquidity_rejected_count", liquidity_rejected),
        ("stale_rejected_count", stale_rejected),
        ("weak_beta_count", weak_beta),
        (
            "risk_groups_count",
            candidates
                .iter()
                .map(|c| c.risk_group)
                .collect::<BTreeSet<_>>()
                .len(),
        ),
        ("correlated_pairs_count", correlated_pairs),
        ("sessions_count", dates.len()),
        ("decision_timestamps_count", widths.len()),
        ("forecast_origins_count", origins.len()),
        ("mark_timestamps_count", tape_frames.len()),
    ] {
        summary.insert(name.into(), count as f64);
    }
    summary.insert(
        "cross_section_min_count".into(),
        *widths.iter().min().unwrap_or(&0) as f64,
    );
    summary.insert(
        "cross_section_max_count".into(),
        *widths.iter().max().unwrap_or(&0) as f64,
    );
    summary.insert(
        "cross_section_mean_count".into(),
        origins.len() as f64 / widths.len().max(1) as f64,
    );
    summary.insert(
        "selection_coverage_fraction".into(),
        candidates.len() as f64 / eligible.max(1) as f64,
    );
    summary.insert(
        "decision_coverage_fraction".into(),
        origins.len() as f64 / (widths.len() * candidates.len()).max(1) as f64,
    );
    let assumptions = vec![
        "All head future calendar/gap inputs use the existing deterministic 04:00–20:00 US-equity exchange calendar, never future ticker print availability. The training label remains next-observed bars, so missing prints can make realized holding time differ from the nominal horizon; marks/fills always use actual observations.".into(),
        format!("Continuous {} observed New York regular sessions from {} through {}; decisions every {} completed five-minute bars, horizon {}; all actual extended-hours marks retained through liquidation at {}. Session dates use the corpus union, never outcome/width selection.", dates.len(), dates[0], dates.last().unwrap(), rebalance, horizon, liquidation),
        format!("Fixed {}-name universe from {} causally eligible names in {} authenticated corpus tickers. Rank uses only trailing {} calendar days' regular-session dollar volume before {}. No evaluation survival or future-target eligibility screen; corpus eligibility itself is inherited from the authenticated checkpoint and cannot recover names absent from that corpus.", candidates.len(), eligible, corpus.contract.tickers.len(), config.history_days, start),
        format!("Beta uses pre-window synchronized five-minute returns against the contemporaneous broad historical cross-section. Empirical groups are connected components at absolute correlation >= {}; {} weak-history betas use the explicitly conservative unit-beta assumption.", config.risk_correlation, weak_beta),
        "Validation has model/checkpoint-selection exposure: this is strategy development evidence, not a terminal holdout. Terminal test is locked. Calibration and risk fitting never use scored outcomes.".into(),
        "Forecasts are cumulative residual log returns, not market forecasts. Residual predictive uncertainty does not insure systematic market risk; beta/group constraints control that separate exposure. Missing/invalid volume is zero execution capacity, never repaired.".into(),
    ];
    Ok(Plan {
        tape: Tape {
            assets,
            frames: tape_frames,
        },
        origins,
        destinations,
        summary,
        assumptions,
    })
}

struct Prediction {
    values: Vec<[f64; 2]>,      // account mean and predictive standard deviation
    diagnostics: Vec<[f64; 4]>, // raw close forecast, target h, target h1, centered step energy
    summary: BTreeMap<String, f64>,
}

/// One BF16 GPU forward per batch and one host transfer for the entire population. Diagnostic
/// targets use the model's causal residual coordinates; cross-section centering spans batches.
fn predict(
    corpus: &Arc<Corpus>,
    model: &CausalPatchModel,
    origins: &[WindowRef],
    horizon: usize,
    batch_size: usize,
    device: Device,
    targets: bool,
    // `[anchor(h), anchor(1), offset(1)]` - the checkpoint's own calibration at the two
    // horizons this account reads, and the intrabar-offset gain that goes with the entry
    // anchor. Unit everywhere on a diagnostic pass, which measures the un-gained emission.
    gains: [f64; 3],
) -> Result<Prediction> {
    ensure!(
        device.is_cuda(),
        "portfolio inference requires the BF16 CUDA model"
    );
    ensure!(
        !origins.is_empty() && batch_size > 0,
        "forecast population and batch size must be nonempty"
    );
    let _guard = tch::no_grad_guard();
    let started = Instant::now();
    let loader = Prefetcher::with_forecasts(Arc::clone(corpus), targets);
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
        let output = model.output(&head);
        let last = stats.last();
        let sigma = last.sigma.reshape([-1]);
        let coordinate = |channel, bar| {
            output
                .coordinates
                .select(1, 0)
                .select(1, channel)
                .select(1, bar)
        };
        let mean_sigma = coordinate(0, horizon as i64 - 1) * (horizon as f64).sqrt();
        if targets {
            let (target, _) = model.targets(batch, &last, true);
            let path = target
                .select(1, 0)
                .select(1, 3)
                .narrow(1, 0, horizon as i64);
            let previous = Tensor::cat(
                &[
                    Tensor::zeros_like(&path.narrow(1, 0, 1)),
                    path.narrow(1, 0, horizon as i64 - 1),
                ],
                1,
            );
            let increments = &path - previous;
            outputs.push(Tensor::cat(
                &[
                    mean_sigma.unsqueeze(1),
                    path.narrow(1, horizon as i64 - 1, 1),
                    path.narrow(1, 0, 1),
                    increments,
                ],
                1,
            ));
        } else {
            let std = output
                .log_scale
                .select(1, 0)
                .select(1, 3)
                .select(1, horizon as i64 - 1)
                .exp()
                * &sigma;
            // The two amplitude coordinates the decoded candle has, both from the checkpoint's
            // frozen calibration: the close ANCHOR scales the σ-units mean, the intrabar OFFSET
            // scales the candle geometry the next-open entry is built from. Applying one and
            // not the other would move the entry away from the anchor it is quoted against.
            let range = last
                .range
                .reshape([-1])
                .clamp_min(f64::from(f32::MIN_POSITIVE))
                * coordinate(1, 0).softplus()
                / std::f64::consts::LN_2;
            let open_offset = (coordinate(3, 0).sigmoid() * &range).log1p();
            let close_offset = (coordinate(2, 0).sigmoid() * range).log1p();
            let next_open =
                coordinate(0, 0) * gains[1] * &sigma + (open_offset - close_offset) * gains[2];
            let mean = &mean_sigma * gains[0] * &sigma - next_open;
            outputs.push(Tensor::stack(&[mean, std], 1));
        }
    }
    let output = Tensor::cat(&outputs, 0);
    drop(outputs);
    let output = if targets {
        // A validation timestamp can occur in different ticker-major batches. Reduce the complete
        // population on-device, never per-batch cohorts or uncentered squared increments.
        let mut groups = BTreeMap::new();
        let mut counts = Vec::<f32>::new();
        let indices: Vec<i64> = origins
            .iter()
            .map(|reference| {
                let timestamp = corpus.ticker(*reference).timestamp(reference.origin);
                let next = groups.len();
                let group = *groups.entry(timestamp).or_insert_with(|| {
                    counts.push(0.);
                    next
                });
                counts[group] += 1.;
                group as i64
            })
            .collect();
        let indices = Tensor::from_slice(&indices).to_device(device);
        let counts = Tensor::from_slice(&counts).to_device(device).unsqueeze(1);
        let increments = output.narrow(1, 3, horizon as i64);
        let sums = Tensor::zeros([groups.len() as i64, horizon as i64], (Kind::Float, device))
            .index_add(0, &indices, &increments);
        let centered = increments - (sums / counts).index_select(0, &indices);
        let energy = centered
            .square()
            .sum_dim_intlist([1i64].as_slice(), true, Kind::Float);
        Tensor::cat(&[output.narrow(1, 0, 3), energy], 1)
    } else {
        output
    };
    let host = output
        .to_device(Device::Cpu)
        .to_kind(Kind::Double)
        .flatten(0, -1);
    let flat = Vec::<f64>::try_from(host)?;
    let (values, diagnostics) = if targets {
        (
            Vec::new(),
            flat.chunks_exact(4)
                .map(|r| [r[0], r[1], r[2], r[3]])
                .collect(),
        )
    } else {
        (
            flat.chunks_exact(2).map(|r| [r[0], r[1]]).collect(),
            Vec::new(),
        )
    };
    let summary = BTreeMap::from([
        (
            "inference_ms".into(),
            started.elapsed().as_secs_f64() * 1000.,
        ),
        ("loader_wait_ms".into(), loader_ms),
        ("loader_work_ms".into(), loader_work_ms),
        (
            "inference_batches_count".into(),
            origins.len().div_ceil(batch_size) as f64,
        ),
    ]);
    Ok(Prediction {
        values,
        diagnostics,
        summary,
    })
}

/// The reserved block's own cross-sectional NNLS gain, in the PORTFOLIO's causal
/// projected-calendar input contract, measured and reported and NEVER applied.
///
/// It used to be the applied amplitude. It is not any more: the checkpoint carries a
/// per-horizon two-coordinate calibration fitted on the same reserved `[70%, 80%)` block, and
/// fitting a second scalar on the same population is fitting one amplitude twice. What this
/// keeps is the part the checkpoint's fit cannot supply - the same quantity measured under the
/// projected-calendar covariates inference actually runs on - so the transfer of the applied
/// amplitude across the two input contracts is a measured comparison instead of an assumption
/// nobody checked. The nonnegativity that made it a safe SIZING gain is now
/// [`FrozenGain::tradable`]'s job, on the checkpoint's own signed measurement.
fn measured_portfolio_gain(
    corpus: &Arc<Corpus>,
    model: &CausalPatchModel,
    device: Device,
    pairing: &Pairing,
    args: &PortfolioEvaluateArgs,
    evaluation_start: i64,
) -> Result<(f64, String, BTreeMap<String, f64>)> {
    let started = Instant::now();
    let artifact_path = args.output.join("portfolio-mean-calibration.json");
    let c = &args.schedule;
    let fit_config_sha256 = digest(&postcard::to_stdvec(&(
        c.calibration_sessions,
        c.universe_size,
        c.history_days,
        c.min_history_bars,
        c.min_price,
        c.min_bar_dollar_volume,
        args.account.rebalance_bars,
    ))?);
    if artifact_path.exists() {
        let mut artifact: ScalarCalibration = serde_json::from_slice(&fs::read(&artifact_path)?)?;
        let sha = std::mem::take(&mut artifact.sha256);
        ensure!(
            sha == digest(&serde_json::to_vec(&artifact)?),
            "scalar calibration authentication failed"
        );
        ensure!(
            artifact.format == CACHE_FORMAT
                && artifact.pairing == *pairing
                && artifact.fit_config_sha256 == fit_config_sha256
                && artifact.horizon == args.account.horizon
                && artifact.last_target_ms < evaluation_start
                && artifact.first_origin_ms >= corpus.contract.boundary_timestamps[0]
                && artifact.last_target_ms < corpus.contract.boundary_timestamps[1]
                && artifact.gain.is_finite()
                && artifact.gain >= 0.
                && artifact.diagnostics.get("recent_nnls_gain_ratio") == Some(&artifact.gain),
            "stored scalar calibration pairing/horizon/block mismatch"
        );
        let mut summary = artifact.diagnostics;
        summary.insert("calibration_cache_hit_count".into(), 1.);
        summary.insert("calibration_ms".into(), 0.);
        for key in [
            "inference_ms",
            "loader_wait_ms",
            "loader_work_ms",
            "inference_batches_count",
        ] {
            summary.insert(format!("recent_{key}"), 0.);
        }
        summary.insert(
            "calibration_purge_gap_days_count".into(),
            (evaluation_start - artifact.last_target_ms) as f64 / 86_400_000.,
        );
        return Ok((artifact.gain, sha, summary));
    }
    let fit = plan(
        corpus,
        &args.schedule,
        args.account.horizon,
        args.account.rebalance_bars,
        0,
        None,
    )?;
    let first_origin_ms = corpus
        .ticker(fit.origins[0])
        .timestamp(fit.origins[0].origin);
    let last_target_ms = fit
        .origins
        .iter()
        .map(|reference| {
            corpus
                .ticker(*reference)
                .timestamp(reference.origin + args.account.horizon)
                + RESOLUTION_MS
        })
        .max()
        .unwrap();
    ensure!(
        last_target_ms < evaluation_start,
        "calibration targets overlap the account period"
    );
    let prediction = predict(
        corpus,
        model,
        &fit.origins,
        args.account.horizon,
        args.batch_size,
        device,
        true,
        [1., 1., 1.],
    )?;
    let timestamps: Vec<_> = fit
        .origins
        .iter()
        .map(|reference| corpus.ticker(*reference).timestamp(reference.origin))
        .collect();
    let mut summary = portfolio_calibration::diagnostics(
        &prediction.diagnostics,
        &timestamps,
        args.account.horizon,
        "recent",
    )?;
    ensure!(
        summary["recent_applicable_origins_count"] > 1.,
        "reserved calibration block contains no usable cross-section"
    );
    let gain = *summary
        .get("recent_nnls_gain_ratio")
        .context("reserved calibration has no identifiable finite nonnegative gain")?;
    summary.extend(
        prediction
            .summary
            .into_iter()
            .map(|(key, value)| (format!("recent_{key}"), value)),
    );
    summary.insert(
        "recent_selected_tickers_count".into(),
        fit.summary["selected_tickers_count"],
    );
    summary.insert(
        "recent_tickers_count".into(),
        fit.origins
            .iter()
            .map(|reference| reference.ticker)
            .collect::<BTreeSet<_>>()
            .len() as f64,
    );
    let mut artifact = ScalarCalibration {
        format: CACHE_FORMAT.into(),
        pairing: pairing.clone(),
        fit_config_sha256,
        horizon: args.account.horizon,
        gain,
        first_origin_ms,
        last_target_ms,
        diagnostics: summary,
        sha256: String::new(),
    };
    artifact.sha256 = digest(&serde_json::to_vec(&artifact)?);
    atomic_write(&artifact_path, &serde_json::to_vec_pretty(&artifact)?)?;
    let mut summary = artifact.diagnostics;
    summary.insert("calibration_cache_hit_count".into(), 0.);
    summary.insert(
        "calibration_ms".into(),
        started.elapsed().as_secs_f64() * 1000.,
    );
    summary.insert(
        "calibration_purge_gap_days_count".into(),
        (evaluation_start - last_target_ms) as f64 / 86_400_000.,
    );
    Ok((gain, artifact.sha256, summary))
}

pub(super) fn evaluate(
    corpus: &Arc<Corpus>,
    model: &CausalPatchModel,
    device: Device,
    pairing: Pairing,
    // The checkpoint's own frozen calibration: the applied amplitude, and the signed
    // measurement the sizing gate reads.
    applied: &FrozenGain,
    args: &PortfolioEvaluateArgs,
) -> Result<AccountEvaluation> {
    args.account.validate()?;
    applied.validate(corpus.contract.pred_len)?;
    ensure!(
        args.account.horizon <= corpus.contract.pred_len,
        "account horizon {} exceeds the checkpoint's calibrated {} horizons",
        args.account.horizon,
        corpus.contract.pred_len
    );
    let config = &args.schedule;
    ensure!(
        config.sessions > 0
            && config.calibration_sessions > 0
            && config.universe_size > 0
            && config.history_days > 0
            && config.min_history_bars >= 100,
        "sessions, universe and history must be positive (at least 100 history bars)"
    );
    ensure!(
        config.min_price.is_finite()
            && config.min_price > 0.
            && config.min_bar_dollar_volume.is_finite()
            && config.min_bar_dollar_volume >= 0.
            && config.risk_correlation.is_finite()
            && config.risk_correlation > 0.
            && config.risk_correlation <= 1.,
        "invalid causal universe/risk thresholds"
    );
    ensure!(
        args.account.horizon <= corpus.contract.pred_len,
        "account horizon exceeds checkpoint horizon"
    );
    let preparing = Instant::now();
    let metadata: Option<Metadata> = config
        .risk_metadata
        .as_ref()
        .map(|path| -> Result<Metadata> { Ok(serde_json::from_slice(&fs::read(path)?)?) })
        .transpose()?;
    if let Some(metadata) = &metadata {
        let mut symbols = BTreeSet::new();
        ensure!(
            metadata
                .assets
                .iter()
                .all(|row| !row.symbol.is_empty() && symbols.insert(&row.symbol)),
            "risk metadata contains an empty or duplicate symbol"
        );
    }
    let mut prepared = plan(
        corpus,
        config,
        args.account.horizon,
        args.account.rebalance_bars,
        1,
        metadata.as_ref(),
    )?;
    let start = prepared.tape.frames[0].timestamp_ms;
    if let Some(metadata) = &metadata {
        ensure!(
            metadata.as_of_ms < start,
            "risk metadata must have an explicit as-of timestamp before evaluation"
        );
        prepared.assumptions.push(format!("Sector and short availability metadata is a frozen external snapshot as of {}. Availability is assumed unchanged for this window; it is not a historical locate feed.", metadata.as_of_ms));
    } else {
        prepared.assumptions.push("No sector or short-availability metadata supplied. No sector facts are invented; empirical correlation groups still apply. Every asset is short-unavailable unless the explicit assumed-short override is enabled.".into());
    }
    if args.account.allow_assumed_short {
        prepared.assumptions.push("ASSUMED SHORT AVAILABILITY OVERRIDE: hypothetical borrow access for all selected assets, not verified broker locates. Borrow costs follow the stated scenario.".into());
    }
    prepared.summary.insert(
        "preparation_ms".into(),
        preparing.elapsed().as_secs_f64() * 1000.,
    );
    let calibration_started = Instant::now();
    // The applied amplitude, from the checkpoint and from nowhere else. One mechanism: the same
    // curves every report, evaluation and trading consumer in this crate reads, at the two
    // horizons this account is built from.
    let (horizon, entry) = (args.account.horizon, 1usize);
    let (gain, entry_gain, offset_gain) = (
        applied.anchor[horizon - 1],
        applied.anchor[entry - 1],
        applied.offset[entry - 1],
    );
    // THE GATE. The applied curve is positive at every horizon by construction - it is fitted
    // on `ln g` - so it can never tell a consumer that a horizon has no usable amplitude. The
    // signed MEASUREMENT can, and this is where it is read: a horizon whose own
    // calibration-block gain is non-positive or absent is sized to zero rather than traded on
    // its neighbours' evidence, and the value that gated it is reported rather than clipped.
    // Sizing is affine in the mean, so a zero mean is a zero position at every name.
    let refusal = applied.sizing_refusal(&[horizon, entry]);
    let gated = refusal.is_some();
    if let Some(refusal) = &refusal {
        prepared.assumptions.push(format!(
            "SIZED TO ZERO BY THE AMPLITUDE GATE: the checkpoint's calibration block measured no usable close-anchor amplitude at {refusal} (h{horizon} is the exit anchor, h1 the entry anchor), so this account takes no position. The fitted curve is positive everywhere by construction and cannot express this; the signed measurement is what gates. Reported, never clipped: an all-zero book with no stated reason is indistinguishable from a broken pipeline."
        ));
    }
    // Measured under the projected-calendar covariates inference runs on, reported, never
    // applied: the transfer of the checkpoint's amplitude across the two input contracts is
    // then a number in the summary instead of an assumption nobody checked.
    let (measured_portfolio_contract_gain, calibration_sha, calibration_summary) =
        measured_portfolio_gain(corpus, model, device, &pairing, args, start)?;
    prepared.assumptions.push(format!(
        "The applied h{horizon} mean gain {gain} and h1 entry gain {entry_gain} (intrabar offset {offset_gain}) are the authenticated checkpoint's own, fitted on {} reserved calibration-partition origins ending {} with the corpus's observed-calendar covariates. Portfolio inference uses causal projected-calendar covariates, so the amplitude's transfer across the two input contracts is an assumption; the reserved block's own cross-sectional NNLS gain under the portfolio's contract is {measured_portfolio_contract_gain}, reported for exactly that comparison and never applied. Predictive standard deviation is unchanged.",
        applied.blocks.calibration_origins,
        applied.blocks.calibration_last_origin_ms
    ));
    prepared.summary.insert(
        "measured_portfolio_contract_gain_ratio".into(),
        measured_portfolio_contract_gain,
    );
    prepared.summary.insert(
        "calibration_measured_gain_ratio".into(),
        applied.measured_anchor[horizon - 1].unwrap_or(f64::NAN),
    );
    prepared.summary.insert(
        "entry_calibration_measured_gain_ratio".into(),
        applied.measured_anchor[entry - 1].unwrap_or(f64::NAN),
    );
    prepared
        .summary
        .insert("calibration_gated_count".into(), u8::from(gated) as f64);
    prepared.summary.insert(
        "calibration_gated_horizons_count".into(),
        applied.gated().len() as f64,
    );
    prepared.summary.extend(calibration_summary);
    prepared
        .summary
        .entry("calibration_ms".into())
        .or_insert_with(|| calibration_started.elapsed().as_secs_f64() * 1000.);
    prepared
        .summary
        .insert("calibration_gain_ratio".into(), gain);
    prepared
        .summary
        .insert("entry_calibration_gain_ratio".into(), entry_gain);
    prepared
        .summary
        .insert("offset_calibration_gain_ratio".into(), offset_gain);
    prepared.assumptions.push("Allocator mean is calibrated predicted terminal close minus calibrated predicted NEXT OPEN, both causal residual log units. Never subtract a realized future open to create a signal. Dispersion is the model's terminal-close residual predictive dispersion, not epistemic confidence and not a fitted joint entry/exit uncertainty; missing raw market risk is addressed separately by beta/group limits.".into());
    prepared.assumptions.push("The corpus uses split/dividend-adjusted prices. Integer shares are adjusted-price share units; per-share commissions and rounding are scenario approximations, not broker-exact historical fills. No point-in-time raw-price/corporate-action cashflow metadata is available, and dividends are not added again.".into());
    prepared.assumptions.push(format!("The reserved-block NNLS diagnostic is authenticated as {calibration_sha}; no scored-day fitting anywhere in this path. Its zero is the constrained least-squares boundary when the block identifies no positive cross-sectional signal, which is a measurement of that block and not a signal inversion."));
    prepared.assumptions.push("Recent and full raw gains are pooled within-origin-timestamp Cov(forecast,target)/Var(forecast), using only cohorts with at least two names. Counts report raw origins, applicable origins, timestamps and New York sessions; singletons do not identify cross-sectional gain. Session-level dependence means origin count is not effective independent n.".into());
    prepared.assumptions.push("Gain standard errors use delete-one-New-York-session jackknife, not independent origin errors. Any 95% normal interval is approximate and especially unreliable for the recent fit's few session clusters; undefined uncertainty is omitted rather than invented. No shrinkage is applied (shrinkage weight zero), and a nonpositive fit keeps the NNLS zero boundary.".into());
    prepared.assumptions.push("Full-window diagnostics use every corpus.validation_refs origin with the same causal projected-calendar inputs as recent calibration, including a width census. Validation has checkpoint/model-selection exposure: its full-window gain is descriptive, is never a policy fit, and is not an untouched-test estimate; terminal test labels remain locked.".into());
    prepared.assumptions.push("Recent calibration uses its frozen liquidity-selected universe; full validation uses the corpus-wide validation population, including narrow and singleton timestamps. Reported recent/full ticker and width counts make this cohort mismatch explicit: model, causal calendar and normalization match, but these are not the same cohort or a controlled temporal gain comparison.".into());
    prepared.assumptions.push("Vh/Dh uses the sum of individually cross-section-centered residual step variances over the same h-window. Vh/(h*V1) instead uses that population's first-step variance times h: these denominators are not interchangeable. Variance scale a=sqrt(Vh/(h*V1)) and normalized forecast gain g/a separate target variance normalization from raw signal gain g; their product leaves calibrated means unchanged.".into());
    prepared.assumptions.push(format!(
        "Authenticated checkpoint/corpus identity: {}",
        serde_json::to_string(&pairing)?
    ));
    prepared.summary.insert(
        "calibration_zero_gain_count".into(),
        u8::from(gated || gain == 0.0) as f64,
    );
    let caching = Instant::now();
    // The hash includes exact unpredicted marks, schedule, historical risk metadata, all corpus and
    // checkpoint identities, and the frozen gain. Financial policy knobs intentionally do not bind
    // the tape: changing those is precisely policy replay, not pretending forecasts were refitted.
    let mut validation_refs_hash = ring::digest::Context::new(&ring::digest::SHA256);
    for reference in &corpus.validation_refs {
        validation_refs_hash.update(&(reference.ticker as u64).to_le_bytes());
        validation_refs_hash.update(&(reference.origin as u64).to_le_bytes());
    }
    let validation_refs_hash = validation_refs_hash.finish();
    let binding = digest(&postcard::to_stdvec(&(
        CACHE_FORMAT,
        &pairing,
        config,
        args.account.horizon,
        args.account.rebalance_bars,
        &calibration_sha,
        gain,
        entry_gain,
        offset_gain,
        gated,
        &metadata,
        &prepared.tape,
        validation_refs_hash.as_ref(),
    ))?);
    let cache = config.tape_cache.as_ref();
    let hit = cache.is_some_and(|path| path.exists());
    if let Some(path) = cache.filter(|path| path.exists()) {
        let cached: CachedTape = postcard::from_bytes(&fs::read(path)?)?;
        ensure!(
            cached.binding == binding,
            "forecast tape does not match checkpoint/corpus/schedule/calibration/risk inputs; use a new cache path"
        );
        ensure!(
            cached.payload_sha256
                == digest(&postcard::to_stdvec(&(&cached.tape, &cached.diagnostics))?),
            "forecast tape or full-window diagnostic payload authentication failed"
        );
        ensure!(
            cached.diagnostics.get("full_origins_count")
                == Some(&(corpus.validation_refs.len() as f64)),
            "forecast tape lacks complete matching validation diagnostics"
        );
        prepared.summary.extend(cached.diagnostics);
        for key in [
            "inference_ms",
            "loader_wait_ms",
            "loader_work_ms",
            "inference_batches_count",
        ] {
            prepared.summary.insert(key.into(), 0.);
            prepared.summary.insert(format!("full_{key}"), 0.);
        }
        prepared.tape = cached.tape;
    } else {
        let prediction = predict(
            corpus,
            model,
            &prepared.origins,
            args.account.horizon,
            args.batch_size,
            device,
            false,
            [gain, entry_gain, offset_gain],
        )?;
        prepared.summary.extend(prediction.summary);
        ensure!(
            prediction.values.len() == prepared.destinations.len(),
            "forecast output cardinality changed"
        );
        let expiries: Vec<_> = prepared
            .tape
            .frames
            .iter()
            .map(|frame| {
                if frame.decision {
                    crate::torch::dataset::forecast_schedule_after(
                        frame.timestamp_ms,
                        args.account.horizon,
                        300,
                    )
                    .last()
                    .copied()
                    .unwrap()
                        + RESOLUTION_MS
                } else {
                    0
                }
            })
            .collect();
        for ((frame, quote), row) in prepared.destinations.iter().zip(prediction.values) {
            ensure!(
                row[0].is_finite() && row[1].is_finite() && row[1] > 0.,
                "nonfinite or invalid model forecast"
            );
            prepared.tape.frames[*frame].quotes[*quote].forecast = Some(Forecast {
                // The gate lands HERE and not on the gain, because zeroing the anchor gain
                // alone would leave the entry offset as the whole signal - a position taken on
                // candle geometry. Sizing is affine in the mean, so a zero mean is no position.
                mean: if gated { 0. } else { row[0] },
                std: row[1],
                expires_ms: expiries[*frame],
            });
        }
        // Diagnostic only: the full validation gain must never enter either calibrated anchor.
        let full = predict(
            corpus,
            model,
            &corpus.validation_refs,
            args.account.horizon,
            args.batch_size,
            device,
            true,
            [1., 1., 1.],
        )?;
        let timestamps: Vec<_> = corpus
            .validation_refs
            .iter()
            .map(|reference| corpus.ticker(*reference).timestamp(reference.origin))
            .collect();
        let mut diagnostics = portfolio_calibration::diagnostics(
            &full.diagnostics,
            &timestamps,
            args.account.horizon,
            "full",
        )?;
        diagnostics.insert(
            "full_tickers_count".into(),
            corpus
                .validation_refs
                .iter()
                .map(|reference| reference.ticker)
                .collect::<BTreeSet<_>>()
                .len() as f64,
        );
        diagnostics.extend(
            full.summary
                .into_iter()
                .map(|(key, value)| (format!("full_{key}"), value)),
        );
        if let Some(path) = cache {
            let payload_sha256 = digest(&postcard::to_stdvec(&(&prepared.tape, &diagnostics))?);
            let cached = CachedTape {
                binding,
                payload_sha256,
                tape: prepared.tape,
                diagnostics,
            };
            atomic_write(path, &postcard::to_stdvec(&cached)?)?;
            prepared.tape = cached.tape;
            prepared.summary.extend(cached.diagnostics);
        } else {
            prepared.summary.extend(diagnostics);
        }
    }
    prepared
        .summary
        .insert("tape_cache_hit_count".into(), u8::from(hit) as f64);
    prepared.summary.insert(
        "full_diagnostics_cache_hit_count".into(),
        u8::from(hit) as f64,
    );
    prepared.summary.insert(
        "tape_load_build_ms".into(),
        caching.elapsed().as_secs_f64() * 1000.,
    );
    let simulation = Instant::now();
    let mut result =
        portfolio::simulate(&prepared.tape, &args.account, &portfolio::CostSource::Flat)?;
    result.summary.insert(
        "simulation_ms".into(),
        simulation.elapsed().as_secs_f64() * 1000.,
    );
    result.summary.extend(prepared.summary);
    result.assumptions.extend(prepared.assumptions);
    // Census follows the actual account clock and is preserved in the same binary report path.
    for point in &mut result.points {
        if let Ok(frame) = prepared
            .tape
            .frames
            .binary_search_by_key(&point.timestamp_ms, |frame| {
                frame.timestamp_ms + RESOLUTION_MS
            })
        {
            let frame = &prepared.tape.frames[frame];
            point
                .values
                .insert("observed_quotes_count".into(), frame.quotes.len() as f64);
            point.values.insert(
                "available_forecasts_count".into(),
                frame
                    .quotes
                    .iter()
                    .filter(|quote| quote.forecast.is_some())
                    .count() as f64,
            );
        }
    }
    Ok(result)
}
