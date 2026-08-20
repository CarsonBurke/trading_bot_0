use anyhow::{ensure, Context, Result};
use ring::digest::{Context as DigestContext, SHA256};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use super::single::{load_market_data, EnvMarketData};
use super::{Env, VecEnv};
use crate::data::macro_econ;
use crate::data::universe::{cached_bar_universe, LIVE_RES_SECS};
use crate::history::{
    episode_tickers_combined::EpisodeHistory, meta_tickers_combined::MetaHistory,
};
use crate::torch::constants::{ACTION_COUNT, ACTION_HISTORY_LEN, STEPS_PER_EPISODE, TICKERS_COUNT};
use crate::types::Account;
use shared::bars::bar_file_path;
use shared::paths::DATA_PATH;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct EnvSnapshot {
    env_id: usize,
    tickers: Vec<String>,
    market_sha256: String,
    total_data_length: usize,
    step: usize,
    max_step: usize,
    account: Account,
    episode_history: EpisodeHistory,
    meta_history: MetaHistory,
    explained_var: Vec<Option<f64>>,
    episode: usize,
    action_history: VecDeque<Vec<f64>>,
    episode_start_offset: usize,
    random_start: bool,
    resample_tickers_on_reset: bool,
    peak_assets: f64,
    last_fill_ratio: f64,
    trade_activity_ema: Vec<f64>,
    steps_since_trade: Vec<usize>,
    position_open_step: Vec<Option<usize>>,
    ticker_perm: Vec<usize>,
    target_weights: Vec<f64>,
    realized_weights: Vec<f64>,
    rng_seed: u64,
    rng_counter: u64,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::history::{
        episode_tickers_combined::EpisodeHistory, meta_tickers_combined::MetaHistory,
    };
    use crate::torch::constants::{
        ACTION_COUNT, PRICE_DELTAS_PER_TICKER, STEPS_PER_EPISODE, TICKERS_COUNT,
    };
    use crate::torch::env::{
        earnings::EarningsIndicators, macro_ind::MacroIndicators, momentum::MomentumIndicators,
    };
    use std::sync::Arc;

    pub(crate) fn synthetic_env() -> Env {
        let n = PRICE_DELTAS_PER_TICKER + STEPS_PER_EPISODE + 16;
        let ticker_count = TICKERS_COUNT as usize;
        let prices = (0..n)
            .map(|index| 100.0 + index as f64 * 0.01)
            .collect::<Vec<_>>();
        let mut deltas = vec![0.0; n];
        for index in 1..n {
            deltas[index] = prices[index] / prices[index - 1] - 1.0;
        }
        Env {
            env_id: 0,
            step: 0,
            max_step: STEPS_PER_EPISODE - 2,
            tickers: (0..ticker_count)
                .map(|index| format!("TEST{index}"))
                .collect(),
            prices: vec![prices.clone(); ticker_count],
            price_deltas: vec![deltas; ticker_count],
            account: Account::new(Env::STARTING_CASH, ticker_count),
            episode_history: EpisodeHistory::new(ticker_count),
            meta_history: MetaHistory::default(),
            episode_start: Instant::now(),
            episode: 7,
            action_history: VecDeque::new(),
            episode_start_offset: PRICE_DELTAS_PER_TICKER,
            total_data_length: n,
            random_start: true,
            resample_tickers_on_reset: false,
            peak_assets: Env::STARTING_CASH,
            last_fill_ratio: 1.0,
            trade_activity_ema: vec![0.0; ticker_count],
            steps_since_trade: vec![0; ticker_count],
            position_open_step: vec![None; ticker_count],
            ticker_perm: (0..ticker_count).collect(),
            target_weights: {
                let mut weights = vec![0.0; ticker_count + 1];
                weights[ticker_count] = 1.0;
                weights
            },
            realized_weights: {
                let mut weights = vec![0.0; ticker_count + 1];
                weights[ticker_count] = 1.0;
                weights
            },
            momentum: vec![Arc::new(MomentumIndicators::compute(&prices)); ticker_count],
            earnings: vec![Arc::new(EarningsIndicators::empty(n)); ticker_count],
            macro_ind: Arc::new(MacroIndicators::empty(n)),
            record_history_io: false,
            gens_path: None,
            rng_seed: 91,
            rng_counter: 3,
        }
    }

    fn market_from(env: &Env) -> EnvMarketData {
        EnvMarketData {
            prices: env.prices.clone(),
            price_deltas: env.price_deltas.clone(),
            momentum: env.momentum.clone(),
            earnings: env.earnings.clone(),
            macro_ind: env.macro_ind.clone(),
            total_data_length: env.total_data_length,
        }
    }

    #[test]
    fn mid_episode_snapshot_resumes_through_a_terminal_reset_exactly() {
        let mut uninterrupted = synthetic_env();
        let hold = vec![0.0; ACTION_COUNT as usize];
        for _ in 0..(STEPS_PER_EPISODE - 4) {
            let transition = uninterrupted.step_step_single(&hold);
            assert_eq!(transition.is_done, 0.0);
        }
        uninterrupted.snapshot().validate().unwrap();
        let encoded = postcard::to_stdvec(&uninterrupted.snapshot()).unwrap();
        let snapshot: EnvSnapshot = postcard::from_bytes(&encoded).unwrap();

        let mut resumed = synthetic_env();
        resumed.apply_snapshot(snapshot, market_from(&uninterrupted));

        for update in 0..4 {
            let action = vec![0.02 + update as f64 * 0.01; ACTION_COUNT as usize];
            let left = uninterrupted.step_step_single(&action);
            let right = resumed.step_step_single(&action);
            assert_eq!(left.is_done, right.is_done);
            assert_eq!(left.reward.to_bits(), right.reward.to_bits());
            assert_eq!(left.step_deltas, right.step_deltas);
            assert_eq!(left.static_obs, right.static_obs);
            assert_eq!(
                uninterrupted.account.total_assets.to_bits(),
                resumed.account.total_assets.to_bits()
            );
            if left.is_done == 1.0 {
                uninterrupted.reset_existing_episode_state();
                resumed.reset_existing_episode_state();
                assert_eq!(
                    uninterrupted.episode_start_offset,
                    resumed.episode_start_offset
                );
                assert_eq!(uninterrupted.ticker_perm, resumed.ticker_perm);
                assert_eq!(uninterrupted.rng_counter, resumed.rng_counter);
            }
        }
    }

    #[test]
    fn snapshot_validation_rejects_non_finite_and_invalid_permutation() {
        let env = synthetic_env();
        let mut snapshot = env.snapshot();
        snapshot.account.cash = f64::NAN;
        assert!(snapshot.validate().is_err());
        snapshot.account.cash = 10_000.0;
        snapshot.ticker_perm[0] = TICKERS_COUNT as usize;
        assert!(snapshot.validate().is_err());

        let mut snapshot = env.snapshot();
        snapshot.episode_history.raw_actions.clear();
        assert!(snapshot.validate().is_err());

        let mut env = synthetic_env();
        env.meta_history.record_explained_var(f64::INFINITY);
        assert!(env.snapshot().validate().is_err());

        let mut env = synthetic_env();
        env.meta_history.record_policy_loss(f64::NAN);
        assert!(env.snapshot().validate().is_err());
    }

    #[test]
    fn snapshot_round_trip_preserves_undefined_explained_variance() {
        let mut env = synthetic_env();
        env.meta_history.record_explained_var(0.25);
        env.meta_history.record_explained_var(f64::NAN);

        let snapshot = env.snapshot();
        assert!(snapshot.meta_history.explained_var.is_empty());
        assert_eq!(snapshot.explained_var, vec![Some(0.25), None]);
        snapshot.validate().unwrap();

        let encoded = postcard::to_stdvec(&snapshot).unwrap();
        let decoded: EnvSnapshot = postcard::from_bytes(&encoded).unwrap();
        decoded.validate().unwrap();

        let mut resumed = synthetic_env();
        resumed.apply_snapshot(decoded, market_from(&env));
        assert_eq!(resumed.meta_history.explained_var.len(), 2);
        assert_eq!(
            resumed.meta_history.explained_var[0].to_bits(),
            0.25f64.to_bits()
        );
        assert!(resumed.meta_history.explained_var[1].is_nan());
    }

    #[test]
    fn ppo_input_fingerprint_covers_inactive_ticker_inputs() {
        let root = std::env::temp_dir().join(format!(
            "trading-bot-snapshot-fingerprint-{}",
            uuid::Uuid::new_v4()
        ));
        let data_path = root.join("data");
        let bars_path = root.join("bars");
        fs::create_dir_all(&data_path).unwrap();
        fs::create_dir_all(&bars_path).unwrap();
        let eligible = vec!["ACTIVE".to_owned(), "FUTURE".to_owned()];
        let bar_file = |ticker: &str| bar_file_path(&bars_path, ticker, LIVE_RES_SECS);
        fs::write(bar_file("ACTIVE"), b"active").unwrap();
        fs::write(bar_file("FUTURE"), b"future-v1").unwrap();
        let before = fingerprint_training_inputs(&data_path, &bars_path, &eligible).unwrap();

        fs::write(bar_file("FUTURE"), b"future-v2").unwrap();
        let after = fingerprint_training_inputs(&data_path, &bars_path, &eligible).unwrap();
        assert_ne!(before, after, "a changed bar corpus must change the fingerprint");

        fs::write(data_path.join("FUTURE_earnings_fmp.bin"), b"reports").unwrap();
        let with_earnings = fingerprint_training_inputs(&data_path, &bars_path, &eligible).unwrap();
        assert_ne!(
            after, with_earnings,
            "a changed earnings cache must change the fingerprint"
        );

        fs::remove_dir_all(&root).unwrap();
    }

    /// The fingerprint must move when a single bar moves. This is why it hashes the corpus
    /// records rather than any derived block: prices, price deltas and the indicator grids
    /// are all lossy functions of the bars, so a change confined to (say) a bar's high or
    /// volume could otherwise slip through unnoticed on resume.
    #[test]
    fn market_fingerprint_moves_when_one_bar_changes() {
        let bars_path = std::env::temp_dir().join(format!(
            "trading-bot-market-fingerprint-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&bars_path).unwrap();
        let tickers = vec!["SMOKE".to_owned()];
        let path = bar_file_path(&bars_path, "SMOKE", LIVE_RES_SECS);

        let mut bars: Vec<shared::bars::PackedBar> = (0..8)
            .map(|index| shared::bars::PackedBar {
                ts_ms: index * 300_000,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.0,
                volume: 1_000.0,
                vwap: 100.0,
                trades: 1,
            })
            .collect();
        shared::bars::write_bar_file(&path, "SMOKE", LIVE_RES_SECS, &bars).unwrap();

        let env = synthetic_env();
        let market = market_from(&env);
        let before = market_fingerprint(&tickers, &bars_path, &market);

        // Touch only the high: no close, so no price, price-delta or indicator value moves.
        bars[4].high = 102.0;
        shared::bars::write_bar_file(&path, "SMOKE", LIVE_RES_SECS, &bars).unwrap();
        let after = market_fingerprint(&tickers, &bars_path, &market);

        fs::remove_dir_all(&bars_path).unwrap();
        assert_ne!(
            before, after,
            "a single changed bar record must change the market fingerprint"
        );
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct VecEnvSnapshot {
    format_version: u32,
    universe_sha256: String,
    envs: Vec<EnvSnapshot>,
}

pub(crate) struct ValidatedVecEnvSnapshot {
    snapshot: VecEnvSnapshot,
    markets: Vec<EnvMarketData>,
}

const ENV_SNAPSHOT_FORMAT_VERSION: u32 = 3;
static UNIVERSE_FINGERPRINT: OnceLock<String> = OnceLock::new();

fn update_str(context: &mut DigestContext, value: &str) {
    context.update(&(value.len() as u64).to_le_bytes());
    context.update(value.as_bytes());
}

fn update_f64s(context: &mut DigestContext, values: &[f64]) {
    context.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        context.update(&value.to_bits().to_le_bytes());
    }
}

fn update_f32s(context: &mut DigestContext, values: impl IntoIterator<Item = f32>) {
    for value in values {
        context.update(&value.to_bits().to_le_bytes());
    }
}

fn finish_hex(context: DigestContext) -> String {
    context
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn update_input_file(context: &mut DigestContext, path: &Path) -> Result<()> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("training input has an invalid filename: {}", path.display()))?;
    update_str(context, name);
    match fs::read(path) {
        Ok(bytes) => {
            context.update(&[1]);
            context.update(&(bytes.len() as u64).to_le_bytes());
            context.update(&bytes);
        }
        Err(error) if error.kind() == ErrorKind::NotFound => context.update(&[0]),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed reading training input {}", path.display()))
        }
    }
    Ok(())
}

/// SHA-256 over every file a PPO observation is built from: each eligible symbol's packed bars
/// and earnings caches, then the macro caches.
///
/// The macro half enumerates [`macro_econ::ALL_SERIES`] rather than globbing the data directory.
/// It used to filter `read_dir` on `starts_with("macro_") && ends_with(".bin")`, which made this a
/// fingerprint of the DIRECTORY rather than of the inputs — and those differ exactly when someone
/// tidies up. Measured when this was changed, that glob hashed 163,840,726 bytes across 15
/// `macro_indicators*.bin` files left by a schema nothing constructs any more, and ZERO files
/// matching the live `macro_{series}_{units}_{freq}_v{N}.bin` name that
/// [`macro_econ::series_cache_path`] builds — so it recorded nothing at all about the series an
/// observation reads. `update_input_file` records absence explicitly, so an uncached series is
/// still represented.
fn fingerprint_training_inputs(
    data_path: &Path,
    bars_path: &Path,
    eligible: &[String],
) -> Result<String> {
    let mut context = DigestContext::new(&SHA256);
    for ticker in eligible {
        update_str(&mut context, ticker);
        update_input_file(&mut context, &bar_file_path(bars_path, ticker, LIVE_RES_SECS))?;
        for provider in ["alphavantage", "finnhub", "fmp"] {
            update_input_file(
                &mut context,
                &data_path.join(format!("{ticker}_earnings_{provider}.bin")),
            )?;
        }
    }

    context.update(&(macro_econ::ALL_SERIES.len() as u64).to_le_bytes());
    for series in macro_econ::ALL_SERIES {
        update_str(&mut context, series.series_id());
        update_input_file(&mut context, &macro_econ::series_cache_path(series))?;
    }
    Ok(finish_hex(context))
}

/// Identity of the PPO training universe and its inputs, memoized for the process.
///
/// Named for the trainer it gates, because [`crate::data::ingest::universe_fingerprint`] used to
/// share the name and is an entirely different thing: that one is a SHA-256 of
/// `long_data/universe.json` alone, it is folded into `BarTrainingProvenance` and thence into the
/// bar world-model LINEAGE HASH, and `BarWorldModelMetadata::validate_schema` re-verifies that
/// hash on every checkpoint load — so changing ITS definition makes every bar checkpoint on disk
/// unloadable rather than merely stale. This one hashes the per-symbol input FILES and gates
/// exactly one thing, the `universe_sha256` resume check below. The two are independent; the
/// shared name was the hazard.
fn ppo_input_fingerprint() -> Result<String> {
    if let Some(fingerprint) = UNIVERSE_FINGERPRINT.get() {
        return Ok(fingerprint.clone());
    }
    let computed = fingerprint_training_inputs(
        Path::new(DATA_PATH),
        &crate::data::ingest::bars_dir(),
        cached_bar_universe(),
    )?;
    let _ = UNIVERSE_FINGERPRINT.set(computed.clone());
    Ok(UNIVERSE_FINGERPRINT.get().cloned().unwrap_or(computed))
}

/// Identity of the market data an environment is running on.
///
/// Hashes the corpus bar file behind every ticker, not a derived block: the degrees of
/// freedom, the close series and the indicator grids are all lossy functions of the bars, so
/// the raw records are the strictly most sensitive thing available. `bars_path` is explicit
/// so a test can point at a corpus it controls.
fn market_fingerprint(tickers: &[String], bars_path: &Path, market: &EnvMarketData) -> String {
    let mut context = DigestContext::new(&SHA256);
    context.update(&(market.total_data_length as u64).to_le_bytes());
    for ticker in tickers {
        update_str(&mut context, ticker);
        // Absent files hash as an absence marker rather than failing: a fingerprint that
        // cannot be computed is worse than one that records "no file here".
        let path = bar_file_path(bars_path, ticker, LIVE_RES_SECS);
        if update_input_file(&mut context, &path).is_err() {
            context.update(&[2]);
        }
    }
    for values in &market.prices {
        update_f64s(&mut context, values);
    }
    for values in &market.price_deltas {
        update_f64s(&mut context, values);
    }
    for indicators in &market.momentum {
        for values in [
            &indicators.rsi,
            &indicators.mom_5,
            &indicators.mom_60,
            &indicators.mom_120,
            &indicators.mom_accel,
            &indicators.vol_adj_mom,
            &indicators.range_pos,
            &indicators.zscore,
            &indicators.efficiency,
            &indicators.macd,
            &indicators.stoch_k,
            &indicators.trend_strength,
        ] {
            update_f64s(&mut context, values);
        }
    }
    for indicators in &market.earnings {
        for values in [
            &indicators.steps_since_available,
            &indicators.revenue_growth,
            &indicators.opex_growth,
            &indicators.net_profit_growth,
            &indicators.eps,
            &indicators.eps_surprise,
        ] {
            update_f64s(&mut context, values);
        }
    }
    let macro_ind = &market.macro_ind;
    for values in [
        &macro_ind.gdp_growth,
        &macro_ind.unemployment,
        &macro_ind.jobs_growth,
        &macro_ind.cpi_yoy,
        &macro_ind.core_cpi_yoy,
        &macro_ind.fed_funds,
        &macro_ind.treasury_10y,
        &macro_ind.yield_spread,
        &macro_ind.consumer_sentiment,
        &macro_ind.initial_claims,
        &macro_ind.steps_to_jobs,
        &macro_ind.steps_to_cpi,
        &macro_ind.steps_to_fomc,
        &macro_ind.steps_to_gdp,
    ] {
        update_f64s(&mut context, values);
    }
    finish_hex(context)
}

fn finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn meta_history_is_finite(history: &MetaHistory) -> bool {
    [
        &history.final_assets,
        &history.cumulative_reward,
        &history.outperformance,
        &history.policy_loss,
        &history.value_loss,
        &history.actor_grad_norm,
        &history.critic_grad_norm,
        &history.total_commissions,
        &history.beta_alpha_mean,
        &history.beta_action_mean,
        &history.beta_beta_mean,
        &history.beta_concentration_mean,
        &history.mean_advantage,
        &history.min_advantage,
        &history.max_advantage,
        &history.logit_scale,
        &history.clip_fraction,
        &history.clip_gap,
        &history.temporal_tau,
        &history.temporal_attn_entropy,
        &history.temporal_attn_max,
        &history.temporal_attn_eff_len,
        &history.temporal_attn_center,
        &history.temporal_attn_last_weight,
        &history.policy_entropy_mean,
        &history.policy_entropy_min,
        &history.policy_entropy_max,
        &history.approx_kl,
        &history.kl_lr_scale,
        &history.kl_lr_scale_next,
        &history.kl_lr_ema,
        &history.kl_lr_signal,
        &history.gate_mean,
        &history.gate_std,
        &history.return_min,
        &history.return_max,
        &history.support_min,
        &history.support_max,
        &history.return_below_support_frac,
        &history.return_above_support_frac,
    ]
    .into_iter()
    .all(|values| finite(values))
}

impl EnvSnapshot {
    fn validate(&self) -> Result<()> {
        let n = self.tickers.len();
        ensure!(
            n == TICKERS_COUNT as usize,
            "snapshot ticker count mismatch"
        );
        ensure!(self.step <= self.max_step, "snapshot step exceeds max_step");
        let remaining = self
            .total_data_length
            .checked_sub(self.episode_start_offset)
            .context("snapshot episode offset exceeds market data")?;
        ensure!(remaining >= 2, "snapshot episode has no usable frontier");
        ensure!(
            self.max_step == remaining.min(STEPS_PER_EPISODE) - 2,
            "snapshot max_step is inconsistent with its market frontier"
        );
        ensure!(
            self.episode_start_offset
                .checked_add(self.step)
                .is_some_and(|frontier| frontier < self.total_data_length),
            "snapshot observation frontier exceeds market data"
        );
        ensure!(self.account.positions.len() == n, "position count mismatch");
        ensure!(
            self.trade_activity_ema.len() == n,
            "activity count mismatch"
        );
        ensure!(
            self.steps_since_trade.len() == n,
            "trade-age count mismatch"
        );
        ensure!(
            self.position_open_step.len() == n,
            "position-age count mismatch"
        );
        ensure!(
            self.ticker_perm.len() == n,
            "ticker permutation length mismatch"
        );
        let mut sorted_perm = self.ticker_perm.clone();
        sorted_perm.sort_unstable();
        ensure!(
            sorted_perm == (0..n).collect::<Vec<_>>(),
            "invalid ticker permutation"
        );
        ensure!(
            self.target_weights.len() == n + 1,
            "target weight length mismatch"
        );
        ensure!(
            self.realized_weights.len() == n + 1,
            "realized weight length mismatch"
        );
        let absolute_frontier = self.episode_start_offset + self.step;
        ensure!(
            self.position_open_step
                .iter()
                .flatten()
                .all(|open_step| *open_step <= absolute_frontier),
            "position-open step lies beyond the observation frontier"
        );
        ensure!(
            self.account.cash.is_finite()
                && self.account.cash >= 0.0
                && self.account.total_assets.is_finite()
                && self.account.total_assets > 0.0
                && self.account.positions.iter().all(|position| {
                    position.quantity.is_finite()
                        && position.quantity >= 0.0
                        && position.avg_price.is_finite()
                        && position.avg_price >= 0.0
                })
                && self.peak_assets.is_finite()
                && self.peak_assets > 0.0
                && self.last_fill_ratio.is_finite()
                && (0.0..=1.0).contains(&self.last_fill_ratio)
                && finite(&self.trade_activity_ema)
                && self
                    .trade_activity_ema
                    .iter()
                    .all(|value| (0.0..=1.0).contains(value))
                && finite(&self.target_weights)
                && self
                    .target_weights
                    .iter()
                    .all(|value| (0.0..=1.0).contains(value))
                && finite(&self.realized_weights)
                && self
                    .realized_weights
                    .iter()
                    .all(|value| (0.0..=1.0).contains(value))
                && self
                    .action_history
                    .iter()
                    .all(|values| values.len() == ACTION_COUNT as usize && finite(values))
                && self.meta_history.explained_var.is_empty()
                && self
                    .explained_var
                    .iter()
                    .flatten()
                    .all(|value| value.is_finite())
                && meta_history_is_finite(&self.meta_history),
            "snapshot contains non-finite causal state"
        );
        ensure!(
            self.action_history.len() == self.step.min(ACTION_HISTORY_LEN),
            "action history frontier mismatch"
        );
        ensure!(
            self.episode_history.buys.len() == n,
            "buy history count mismatch"
        );
        ensure!(
            self.episode_history.sells.len() == n,
            "sell history count mismatch"
        );
        ensure!(
            self.episode_history.positioned.len() == n,
            "position history count mismatch"
        );
        ensure!(
            self.episode_history.raw_actions.len() == n,
            "raw-action history count mismatch"
        );
        ensure!(
            self.episode_history.target_weights.len() == n,
            "target-weight history count mismatch"
        );
        ensure!(
            self.episode_history
                .action_step0
                .as_ref()
                .is_none_or(|values| values.len() == ACTION_COUNT as usize)
                && self
                    .episode_history
                    .action_final
                    .as_ref()
                    .is_none_or(|values| values.len() == ACTION_COUNT as usize),
            "episode action snapshot length mismatch"
        );
        ensure!(
            self.episode_history.cash.len() == self.step
                && self.episode_history.rewards.len() == self.step
                && self.episode_history.cash_weight.len() == self.step
                && self
                    .episode_history
                    .positioned
                    .iter()
                    .all(|values| values.len() == self.step)
                && self
                    .episode_history
                    .raw_actions
                    .iter()
                    .all(|values| values.len() == self.step)
                && self
                    .episode_history
                    .target_weights
                    .iter()
                    .all(|values| values.len() == self.step),
            "episode history frontier mismatch"
        );
        ensure!(
            finite(&self.episode_history.cash)
                && finite(&self.episode_history.rewards)
                && finite(&self.episode_history.cash_weight)
                && self.episode_history.total_commissions.is_finite()
                && self
                    .episode_history
                    .positioned
                    .iter()
                    .all(|values| finite(values))
                && self
                    .episode_history
                    .raw_actions
                    .iter()
                    .all(|values| finite(values))
                && self
                    .episode_history
                    .target_weights
                    .iter()
                    .all(|values| finite(values))
                && self
                    .episode_history
                    .static_observations
                    .iter()
                    .flatten()
                    .all(|value| value.is_finite())
                && self
                    .episode_history
                    .attention_weights
                    .iter()
                    .flatten()
                    .all(|value| value.is_finite())
                && self
                    .episode_history
                    .action_step0
                    .as_deref()
                    .is_none_or(finite)
                && self
                    .episode_history
                    .action_final
                    .as_deref()
                    .is_none_or(finite)
                && self
                    .episode_history
                    .buys
                    .iter()
                    .chain(&self.episode_history.sells)
                    .all(|trades| trades
                        .values()
                        .all(|(price, quantity)| { price.is_finite() && quantity.is_finite() })),
            "episode history contains non-finite state"
        );
        Ok(())
    }
}

impl Env {
    fn snapshot(&self) -> EnvSnapshot {
        let market = EnvMarketData {
            prices: self.prices.clone(),
            price_deltas: self.price_deltas.clone(),
            momentum: self.momentum.clone(),
            earnings: self.earnings.clone(),
            macro_ind: self.macro_ind.clone(),
            total_data_length: self.total_data_length,
        };
        let mut meta_history = self.meta_history.clone();
        let explained_var = std::mem::take(&mut meta_history.explained_var)
            .into_iter()
            .map(|value| if value.is_nan() { None } else { Some(value) })
            .collect();
        EnvSnapshot {
            env_id: self.env_id,
            tickers: self.tickers.clone(),
            market_sha256: market_fingerprint(
                &self.tickers,
                &crate::data::ingest::bars_dir(),
                &market,
            ),
            total_data_length: self.total_data_length,
            step: self.step,
            max_step: self.max_step,
            account: self.account.clone(),
            episode_history: self.episode_history.clone(),
            meta_history,
            explained_var,
            episode: self.episode,
            action_history: self.action_history.clone(),
            episode_start_offset: self.episode_start_offset,
            random_start: self.random_start,
            resample_tickers_on_reset: self.resample_tickers_on_reset,
            peak_assets: self.peak_assets,
            last_fill_ratio: self.last_fill_ratio,
            trade_activity_ema: self.trade_activity_ema.clone(),
            steps_since_trade: self.steps_since_trade.clone(),
            position_open_step: self.position_open_step.clone(),
            ticker_perm: self.ticker_perm.clone(),
            target_weights: self.target_weights.clone(),
            realized_weights: self.realized_weights.clone(),
            rng_seed: self.rng_seed,
            rng_counter: self.rng_counter,
        }
    }

    fn apply_snapshot(&mut self, snapshot: EnvSnapshot, market: EnvMarketData) {
        self.env_id = snapshot.env_id;
        self.step = snapshot.step;
        self.max_step = snapshot.max_step;
        self.tickers = snapshot.tickers;
        self.prices = market.prices;
        self.price_deltas = market.price_deltas;
        self.momentum = market.momentum;
        self.earnings = market.earnings;
        self.macro_ind = market.macro_ind;
        self.total_data_length = market.total_data_length;
        self.account = snapshot.account;
        self.episode_history = snapshot.episode_history;
        self.meta_history = snapshot.meta_history;
        self.meta_history.explained_var = snapshot
            .explained_var
            .into_iter()
            .map(|value| value.unwrap_or(f64::NAN))
            .collect();
        self.episode = snapshot.episode;
        self.action_history = snapshot.action_history;
        self.episode_start_offset = snapshot.episode_start_offset;
        self.random_start = snapshot.random_start;
        self.resample_tickers_on_reset = snapshot.resample_tickers_on_reset;
        self.peak_assets = snapshot.peak_assets;
        self.last_fill_ratio = snapshot.last_fill_ratio;
        self.trade_activity_ema = snapshot.trade_activity_ema;
        self.steps_since_trade = snapshot.steps_since_trade;
        self.position_open_step = snapshot.position_open_step;
        self.ticker_perm = snapshot.ticker_perm;
        self.target_weights = snapshot.target_weights;
        self.realized_weights = snapshot.realized_weights;
        self.rng_seed = snapshot.rng_seed;
        self.rng_counter = snapshot.rng_counter;
        self.episode_start = Instant::now();
    }
}

impl VecEnvSnapshot {
    pub(crate) fn to_bytes(&self) -> Result<Vec<u8>> {
        postcard::to_stdvec(self).context("failed serializing PPO trajectory state")
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let snapshot: Self =
            postcard::from_bytes(bytes).context("failed parsing PPO trajectory state")?;
        ensure!(
            snapshot.format_version == ENV_SNAPSHOT_FORMAT_VERSION,
            "unsupported environment snapshot format {}",
            snapshot.format_version
        );
        ensure!(
            snapshot.universe_sha256 == ppo_input_fingerprint()?,
            "eligible training universe changed since checkpoint"
        );
        ensure!(!snapshot.envs.is_empty(), "environment snapshot is empty");
        for env in &snapshot.envs {
            env.validate()?;
        }
        Ok(snapshot)
    }

    /// Validate trajectory identity and load every immutable input before
    /// checkpoint weights or optimizer state are applied.
    pub(crate) fn preflight(
        self,
        expected_nprocs: usize,
        expected_seed: u64,
    ) -> Result<ValidatedVecEnvSnapshot> {
        ensure!(
            self.envs.len() == expected_nprocs,
            "PPO_NPROCS mismatch: saved={}, requested={expected_nprocs}",
            self.envs.len()
        );
        for (index, env) in self.envs.iter().enumerate() {
            ensure!(env.env_id == index, "environment IDs are not canonical");
            ensure!(
                env.random_start && env.resample_tickers_on_reset,
                "saved environment flags are incompatible with PPO training"
            );
            ensure!(
                env.rng_seed == expected_seed.wrapping_add(index as u64),
                "environment RNG stream identity mismatch for env {index}"
            );
        }
        let mut markets = Vec::with_capacity(self.envs.len());
        for env in &self.envs {
            let market = load_market_data(&env.tickers, false);
            ensure!(
                market.total_data_length == env.total_data_length,
                "market length changed for {:?}",
                env.tickers
            );
            ensure!(
                market_fingerprint(&env.tickers, &crate::data::ingest::bars_dir(), &market)
                    == env.market_sha256,
                "market data changed for {:?}",
                env.tickers
            );
            markets.push(market);
        }
        Ok(ValidatedVecEnvSnapshot {
            snapshot: self,
            markets,
        })
    }
}

impl VecEnv {
    pub(crate) fn snapshot(&self) -> Result<VecEnvSnapshot> {
        Ok(VecEnvSnapshot {
            format_version: ENV_SNAPSHOT_FORMAT_VERSION,
            universe_sha256: ppo_input_fingerprint()?,
            envs: self.envs.iter().map(Env::snapshot).collect(),
        })
    }

    /// Apply a plan whose complete causal and immutable state was preflighted.
    pub(crate) fn restore_snapshot(&mut self, plan: ValidatedVecEnvSnapshot) {
        debug_assert_eq!(plan.snapshot.envs.len(), self.envs.len());
        for ((env, saved), market) in self
            .envs
            .iter_mut()
            .zip(plan.snapshot.envs)
            .zip(plan.markets)
        {
            env.apply_snapshot(saved, market);
        }
    }

    #[cfg(test)]
    pub(crate) fn restore_snapshot_from_current_markets(
        &mut self,
        snapshot: VecEnvSnapshot,
    ) -> Result<()> {
        ensure!(
            snapshot.envs.len() == self.envs.len(),
            "test env count mismatch"
        );
        let mut markets = Vec::with_capacity(self.envs.len());
        for (current, saved) in self.envs.iter().zip(&snapshot.envs) {
            let market = EnvMarketData {
                prices: current.prices.clone(),
                price_deltas: current.price_deltas.clone(),
                momentum: current.momentum.clone(),
                earnings: current.earnings.clone(),
                macro_ind: current.macro_ind.clone(),
                total_data_length: current.total_data_length,
            };
            ensure!(
                market_fingerprint(&saved.tickers, &crate::data::ingest::bars_dir(), &market)
                    == saved.market_sha256,
                "test market mismatch"
            );
            markets.push(market);
        }
        for ((env, saved), market) in self.envs.iter_mut().zip(snapshot.envs).zip(markets) {
            env.apply_snapshot(saved, market);
        }
        Ok(())
    }
}
