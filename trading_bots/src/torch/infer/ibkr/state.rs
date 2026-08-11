use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tch::Tensor;
use time::OffsetDateTime;

use crate::data::get_cached_earnings_data_any;
use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT};
use crate::torch::env::earnings::EarningsIndicators;
use crate::torch::env::macro_ind::MacroIndicators;
use crate::torch::env::momentum::MomentumIndicators;
use crate::torch::env::obs::{build_static_obs, realized_weight, GlobalObsInputs, TickerObsInputs};
use crate::torch::env::TRADE_EMA_ALPHA;
use crate::types::Account;

pub(super) const MAX_ACCOUNT_VALUE: Option<f64> = Some(10_000.0);

#[derive(Debug)]
enum FeedHealth {
    Starting {
        since: Instant,
    },
    Live {
        last_receipt: Instant,
        last_source: OffsetDateTime,
    },
    Failed(String),
}

#[derive(Debug, Clone)]
pub(super) struct TradeBatchOutcome {
    pub(super) commission: f64,
    pub(super) fill_ratio: f64,
}

impl Default for TradeBatchOutcome {
    fn default() -> Self {
        Self {
            commission: 0.0,
            fill_ratio: 1.0,
        }
    }
}

pub(super) struct LiveMarketState {
    pub(super) symbols: Vec<String>,
    pub(super) prices: Vec<VecDeque<f64>>,
    pub(super) price_deltas: Vec<VecDeque<f64>>,
    /// Exact timestamp of each retained completed bar, shared across tickers.
    pub(super) bar_times: VecDeque<OffsetDateTime>,
    pub(super) account: Account,
    pub(super) starting_cash: f64,
    pub(super) peak_assets: f64,
    pub(super) total_commissions: f64,
    pub(super) step_count: usize,
    pub(super) last_fill_ratio: f64,
    pub(super) steps_since_trade: Vec<usize>,
    pub(super) position_open_step: Vec<Option<usize>>,
    pub(super) trade_activity_ema: Vec<f64>,
    feed_health: Vec<FeedHealth>,
    latest_execution_prices: Vec<Option<f64>>,
    building_buckets: Vec<Option<(i64, f64)>>,
    staged_buckets: BTreeMap<i64, Vec<Option<f64>>>,
    committed_bucket: Option<i64>,
    committed_sequence: u64,
}

#[cfg(test)]
mod tests {
    use super::{LiveMarketState, TradeBatchOutcome};
    use std::time::{Duration, Instant};
    use time::OffsetDateTime;

    fn bar_time(bucket: i64, offset_seconds: i64) -> OffsetDateTime {
        OffsetDateTime::from_unix_timestamp(bucket * 300 + offset_seconds).unwrap()
    }

    #[test]
    fn historical_seed_is_not_actionable_until_every_live_feed_completes_one_bucket() {
        let now = Instant::now();
        let alignment_window = Duration::from_secs(600);
        let mut state = LiveMarketState::new(vec!["A".into(), "B".into()], 10_000.0);
        assert!(state
            .actionable_prices(now, bar_time(100, 5), alignment_window, None,)
            .unwrap()
            .is_none());

        state
            .record_realtime_bar(0, bar_time(100, 5), 10.0, now)
            .unwrap();
        state
            .record_realtime_bar(0, bar_time(101, 5), 11.0, now)
            .unwrap();
        state
            .record_realtime_bar(1, bar_time(100, 5), 20.0, now)
            .unwrap();
        assert!(state
            .actionable_prices(now, bar_time(101, 5), alignment_window, None,)
            .unwrap()
            .is_none());

        state
            .record_realtime_bar(1, bar_time(101, 5), 21.0, now)
            .unwrap();
        let (bucket, sequence, prices) = state
            .actionable_prices(now, bar_time(101, 5), alignment_window, None)
            .unwrap()
            .unwrap();
        assert_eq!(bucket, 100);
        assert_eq!(sequence, 1);
        assert_eq!(prices, vec![11.0, 21.0]);
        assert!(state
            .actionable_prices(now, bar_time(101, 5), alignment_window, Some(sequence),)
            .unwrap()
            .is_none());
    }

    #[test]
    fn terminal_and_stale_feeds_fail_closed() {
        let now = Instant::now();
        let starting = LiveMarketState::new(vec!["A".into()], 10_000.0);
        assert!(starting
            .actionable_prices(
                now + Duration::from_secs(31),
                OffsetDateTime::now_utc(),
                Duration::from_secs(30),
                None,
            )
            .unwrap_err()
            .contains("startup deadline"));

        let mut failed = LiveMarketState::new(vec!["A".into()], 10_000.0);
        failed.mark_feed_failed(0, "ended");
        assert!(failed
            .actionable_prices(
                now,
                OffsetDateTime::now_utc(),
                Duration::from_secs(30),
                None,
            )
            .unwrap_err()
            .contains("ended"));

        let mut stale = LiveMarketState::new(vec!["A".into()], 10_000.0);
        stale
            .record_realtime_bar(0, bar_time(100, 5), 10.0, now - Duration::from_secs(31))
            .unwrap();
        assert!(stale
            .actionable_prices(now, bar_time(100, 5), Duration::from_secs(30), None,)
            .unwrap_err()
            .contains("stale"));

        let mut delayed = LiveMarketState::new(vec!["A".into()], 10_000.0);
        delayed
            .record_realtime_bar(0, bar_time(100, 5), 10.0, now)
            .unwrap();
        assert!(delayed
            .actionable_prices(now, bar_time(100, 40), Duration::from_secs(30), None,)
            .unwrap_err()
            .contains("source timestamp"));
    }

    #[test]
    fn consumed_frame_can_age_out_while_healthy_feed_waits_for_next_bar() {
        let now = Instant::now();
        let mut state = LiveMarketState::new(vec!["A".into()], 10_000.0);
        state
            .record_realtime_bar(0, bar_time(100, 5), 10.0, now)
            .unwrap();
        state
            .record_realtime_bar(0, bar_time(101, 5), 11.0, now)
            .unwrap();

        assert!(state
            .actionable_prices(now, bar_time(101, 31), Duration::from_secs(30), Some(1),)
            .unwrap()
            .is_none());
        assert!(state
            .actionable_prices(now, bar_time(101, 31), Duration::from_secs(30), None)
            .unwrap_err()
            .contains("model-observation frame is stale"));
    }

    #[test]
    fn post_fill_feedback_updates_the_next_observation_state() {
        let mut state = LiveMarketState::new(vec!["TEST".into()], 10_000.0);
        state.prices[0].extend([100.0, 101.0, 102.0]);
        state.account.positions[0].quantity = 1.0;
        state.account.positions[0].avg_price = 101.0;
        state.account.total_assets = 10_000.0;
        state
            .apply_execution_feedback(
                &[0.0],
                &TradeBatchOutcome {
                    commission: 1.25,
                    fill_ratio: 0.75,
                },
            )
            .unwrap();

        assert_eq!(state.step_count, 1);
        assert_eq!(state.total_commissions, 1.25);
        assert_eq!(state.last_fill_ratio, 0.75);
        assert_eq!(state.steps_since_trade[0], 0);
        assert_eq!(state.position_open_step[0], Some(0));
        let prices = state.prices[0].iter().copied().collect::<Vec<_>>();
        let dates = vec!["2024-01-02".to_string(); prices.len()];
        let inputs = state.ticker_inputs(0, &prices, &dates, prices.len() - 1);
        assert_eq!(inputs.steps_since_trade, 0);
        assert!(inputs.position_age > 0.0);
        assert!(inputs.trade_activity_ema > 0.0);
    }

    #[test]
    fn skipped_completed_frames_advance_ages_without_intrabar_peak_leakage() {
        let mut state = LiveMarketState::new(vec!["TEST".into()], 10_000.0);
        state.trade_activity_ema[0] = 1.0;
        state.steps_since_trade[0] = 2;
        state.prepare_observation(4).unwrap();
        assert_eq!(state.step_count, 3);
        assert_eq!(state.steps_since_trade[0], 5);
        let expected_decay = (1.0 - crate::torch::env::TRADE_EMA_ALPHA).powi(3);
        assert!((state.trade_activity_ema[0] - expected_decay).abs() < 1e-12);

        state.account.positions[0].quantity = 1.0;
        state.revalue_account_with_prices(&[20_000.0]);
        assert_eq!(state.peak_assets, 10_000.0);
        state.update_observation_value_with_prices(&[20_000.0]);
        assert_eq!(state.peak_assets, state.account.total_assets);
    }

    #[test]
    fn overnight_rollover_is_discarded_and_skipped_frames_preserve_peak() {
        let now = Instant::now();
        let mut state = LiveMarketState::new(vec!["TEST".into()], 10_000.0);
        state
            .record_realtime_bar(0, bar_time(100, 5), 100.0, now)
            .unwrap();
        state
            .record_realtime_bar(0, bar_time(300, 5), 100.0, now)
            .unwrap();
        assert!(state
            .actionable_prices(now, bar_time(300, 5), Duration::from_secs(30), None)
            .unwrap()
            .is_none());

        state.account.cash = 0.0;
        state.account.positions[0].quantity = 1.0;
        state
            .record_realtime_bar(0, bar_time(301, 5), 20_000.0, now)
            .unwrap();
        state
            .record_realtime_bar(0, bar_time(302, 5), 5_000.0, now)
            .unwrap();
        state
            .record_realtime_bar(0, bar_time(303, 5), 5_000.0, now)
            .unwrap();
        assert_eq!(state.peak_assets, 20_000.0);
        let (bucket, _, _) = state
            .actionable_prices(now, bar_time(303, 5), Duration::from_secs(30), None)
            .unwrap()
            .unwrap();
        assert_eq!(bucket, 302);
    }
}

impl LiveMarketState {
    pub(super) fn new(symbols: Vec<String>, starting_cash: f64) -> Self {
        let ticker_count = symbols.len();
        Self {
            symbols,
            prices: vec![VecDeque::with_capacity(PRICE_DELTAS_PER_TICKER + 1); ticker_count],
            price_deltas: vec![VecDeque::with_capacity(PRICE_DELTAS_PER_TICKER); ticker_count],
            bar_times: VecDeque::with_capacity(PRICE_DELTAS_PER_TICKER + 1),
            account: Account::new(starting_cash, ticker_count),
            starting_cash,
            peak_assets: starting_cash,
            total_commissions: 0.0,
            step_count: 0,
            last_fill_ratio: 1.0,
            steps_since_trade: vec![0; ticker_count],
            position_open_step: vec![None; ticker_count],
            trade_activity_ema: vec![0.0; ticker_count],
            feed_health: (0..ticker_count)
                .map(|_| FeedHealth::Starting {
                    since: Instant::now(),
                })
                .collect(),
            latest_execution_prices: vec![None; ticker_count],
            building_buckets: vec![None; ticker_count],
            staged_buckets: BTreeMap::new(),
            committed_bucket: None,
            committed_sequence: 0,
        }
    }

    /// Seed price/delta history and bar dates from historical 5-minute bars so
    /// the model can be fed a full observation window immediately, matching the
    /// resolution and warm-up the model was trained on.
    pub(super) fn seed_history(
        &mut self,
        ticker_idx: usize,
        closes: &[f64],
        times: &[OffsetDateTime],
    ) {
        for &close in closes {
            self.update_price(ticker_idx, close);
        }
        if ticker_idx == 0 {
            for &time in times {
                self.push_bar_time(time);
            }
        }
    }

    fn push_bar_time(&mut self, time: OffsetDateTime) {
        self.bar_times.push_back(time);
        if self.bar_times.len() > PRICE_DELTAS_PER_TICKER + 1 {
            self.bar_times.pop_front();
        }
    }

    pub(super) fn update_price(&mut self, ticker_idx: usize, price: f64) {
        self.prices[ticker_idx].push_back(price);
        if self.prices[ticker_idx].len() > PRICE_DELTAS_PER_TICKER + 1 {
            self.prices[ticker_idx].pop_front();
        }

        if self.prices[ticker_idx].len() >= 2 {
            let len = self.prices[ticker_idx].len();
            let prev_price = self.prices[ticker_idx][len - 2];
            let delta = (price / prev_price).ln();

            self.price_deltas[ticker_idx].push_back(delta);
            if self.price_deltas[ticker_idx].len() > PRICE_DELTAS_PER_TICKER {
                self.price_deltas[ticker_idx].pop_front();
            }
        }
    }

    pub(super) fn record_realtime_bar(
        &mut self,
        ticker_idx: usize,
        timestamp: OffsetDateTime,
        close: f64,
        received_at: Instant,
    ) -> Result<(), String> {
        if ticker_idx >= self.symbols.len() || !close.is_finite() || close <= 0.0 {
            return Err(format!("invalid realtime bar for feed {ticker_idx}"));
        }
        self.feed_health[ticker_idx] = FeedHealth::Live {
            last_receipt: received_at,
            last_source: timestamp,
        };
        self.latest_execution_prices[ticker_idx] = Some(close);
        let bucket = timestamp.unix_timestamp().div_euclid(300);
        match self.building_buckets[ticker_idx] {
            Some((current, _)) if bucket < current => {
                return Err(format!(
                    "out-of-order realtime bucket for {}: {bucket} < {current}",
                    self.symbols[ticker_idx]
                ));
            }
            Some((current, _)) if bucket == current => {
                self.building_buckets[ticker_idx] = Some((current, close));
            }
            Some((current, completed_close)) => {
                // A regular five-minute rollover advances exactly one bucket.
                // Across an overnight/session gap the prior partial bucket is
                // stale, so discard it and warm up the new session instead.
                if bucket == current + 1 {
                    self.stage_completed_bar(ticker_idx, current, completed_close);
                }
                self.building_buckets[ticker_idx] = Some((bucket, close));
                self.commit_aligned_buckets()?;
            }
            None => self.building_buckets[ticker_idx] = Some((bucket, close)),
        }
        Ok(())
    }

    fn stage_completed_bar(&mut self, ticker_idx: usize, bucket: i64, close: f64) {
        self.staged_buckets
            .entry(bucket)
            .or_insert_with(|| vec![None; self.symbols.len()])[ticker_idx] = Some(close);
    }

    fn commit_aligned_buckets(&mut self) -> Result<(), String> {
        let ready = self
            .staged_buckets
            .iter()
            .filter(|(bucket, closes)| {
                Some(**bucket) > self.committed_bucket && closes.iter().all(Option::is_some)
            })
            .map(|(&bucket, _)| bucket)
            .collect::<Vec<_>>();
        for bucket in ready {
            let closes = self
                .staged_buckets
                .remove(&bucket)
                .expect("ready bucket disappeared");
            for (ticker_idx, close) in closes.into_iter().enumerate() {
                self.update_price(ticker_idx, close.expect("ready bucket has every close"));
            }
            let frame_assets = self.account.cash
                + self
                    .account
                    .positions
                    .iter()
                    .zip(&self.prices)
                    .map(|(position, prices)| {
                        position.value_with_price(*prices.back().expect("committed close missing"))
                    })
                    .sum::<f64>();
            if frame_assets.is_finite() {
                self.peak_assets = self.peak_assets.max(frame_assets);
            }
            // IBKR historical five-minute bars are timestamped at bucket start.
            // Preserve that convention so live macro alignment is identical to
            // training and never exposes a within-bucket release early.
            let bar_timestamp = bucket
                .checked_mul(300)
                .ok_or_else(|| "realtime bucket timestamp overflow".to_string())?;
            self.push_bar_time(
                OffsetDateTime::from_unix_timestamp(bar_timestamp)
                    .map_err(|error| format!("invalid completed bucket timestamp: {error}"))?,
            );
            self.committed_bucket = Some(bucket);
            self.committed_sequence = self
                .committed_sequence
                .checked_add(1)
                .ok_or_else(|| "completed live-bar sequence overflowed".to_string())?;
        }
        self.staged_buckets
            .retain(|bucket, _| Some(*bucket) > self.committed_bucket);
        Ok(())
    }

    pub(super) fn mark_feed_failed(&mut self, ticker_idx: usize, error: impl Into<String>) {
        if ticker_idx < self.feed_health.len() {
            self.feed_health[ticker_idx] = FeedHealth::Failed(error.into());
        }
    }

    pub(super) fn actionable_prices(
        &self,
        now: Instant,
        wall_now: OffsetDateTime,
        max_feed_age: Duration,
        last_acted_sequence: Option<u64>,
    ) -> Result<Option<(i64, u64, Vec<f64>)>, String> {
        for (ticker_idx, health) in self.feed_health.iter().enumerate() {
            match health {
                FeedHealth::Starting { since }
                    if now.saturating_duration_since(*since) > max_feed_age =>
                {
                    return Err(format!(
                        "{} feed did not become live before the startup deadline",
                        self.symbols[ticker_idx]
                    ));
                }
                FeedHealth::Starting { .. } => return Ok(None),
                FeedHealth::Failed(error) => {
                    return Err(format!("{} feed failed: {error}", self.symbols[ticker_idx]));
                }
                FeedHealth::Live { last_receipt, .. }
                    if now.saturating_duration_since(*last_receipt) > max_feed_age =>
                {
                    return Err(format!("{} feed is stale", self.symbols[ticker_idx]));
                }
                FeedHealth::Live { last_source, .. } => {
                    let source_age = wall_now.unix_timestamp() - last_source.unix_timestamp();
                    if source_age > max_feed_age.as_secs() as i64 {
                        return Err(format!(
                            "{} feed source timestamp is stale",
                            self.symbols[ticker_idx]
                        ));
                    }
                    if source_age < -15 {
                        return Err(format!(
                            "{} feed source timestamp is implausibly in the future",
                            self.symbols[ticker_idx]
                        ));
                    }
                }
            }
        }
        let Some(bucket) = self.committed_bucket else {
            return Ok(None);
        };
        if last_acted_sequence.is_some_and(|sequence| self.committed_sequence <= sequence) {
            return Ok(None);
        }
        let completed_at = bucket
            .checked_add(1)
            .and_then(|next| next.checked_mul(300))
            .ok_or_else(|| "completed live-bar timestamp overflowed".to_string())?;
        let completed_age = wall_now.unix_timestamp() - completed_at;
        if completed_age > max_feed_age.as_secs() as i64 {
            return Err("completed model-observation frame is stale".to_string());
        }
        let prices = self
            .latest_execution_prices
            .iter()
            .enumerate()
            .map(|(ticker_idx, price)| {
                price.ok_or_else(|| format!("{} has no executable price", self.symbols[ticker_idx]))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some((bucket, self.committed_sequence, prices)))
    }

    pub(super) fn prepare_observation(&mut self, sequence: u64) -> Result<(), String> {
        let desired_step = sequence
            .checked_sub(1)
            .ok_or_else(|| "live observation sequence starts at zero".to_string())?;
        let current_step = u64::try_from(self.step_count)
            .map_err(|_| "paper step count exceeds u64".to_string())?;
        if desired_step < current_step {
            return Err("live observation sequence moved behind model state".to_string());
        }
        let elapsed = desired_step - current_step;
        let elapsed_usize = usize::try_from(elapsed)
            .map_err(|_| "skipped live-bar count exceeds usize".to_string())?;
        if elapsed_usize > 0 {
            let decay = (1.0 - TRADE_EMA_ALPHA).powi(
                i32::try_from(elapsed_usize)
                    .map_err(|_| "skipped live-bar count exceeds i32".to_string())?,
            );
            for activity in &mut self.trade_activity_ema {
                *activity *= decay;
            }
            for steps in &mut self.steps_since_trade {
                *steps = steps
                    .checked_add(elapsed_usize)
                    .ok_or_else(|| "paper trade-age counter overflowed".to_string())?;
            }
            self.step_count = self
                .step_count
                .checked_add(elapsed_usize)
                .ok_or_else(|| "paper step counter overflowed".to_string())?;
        }
        Ok(())
    }

    pub(super) fn get_current_prices(&self) -> Vec<f64> {
        self.prices
            .iter()
            .map(|q| *q.back().unwrap_or(&0.0))
            .collect()
    }

    pub(super) fn apply_execution_feedback(
        &mut self,
        previous_quantities: &[f64],
        outcome: &TradeBatchOutcome,
    ) -> Result<(), String> {
        if previous_quantities.len() != self.account.positions.len()
            || !outcome.commission.is_finite()
            || !outcome.fill_ratio.is_finite()
            || !(0.0..=1.0).contains(&outcome.fill_ratio)
        {
            return Err("invalid paper execution feedback".to_string());
        }
        self.total_commissions += outcome.commission;
        self.last_fill_ratio = outcome.fill_ratio;
        for ticker_idx in 0..self.account.positions.len() {
            self.trade_activity_ema[ticker_idx] *= 1.0 - TRADE_EMA_ALPHA;
            self.steps_since_trade[ticker_idx] += 1;
            let before = previous_quantities[ticker_idx];
            let after = self.account.positions[ticker_idx].quantity;
            if (after - before).abs() <= 1e-8 {
                continue;
            }
            self.trade_activity_ema[ticker_idx] += TRADE_EMA_ALPHA;
            self.steps_since_trade[ticker_idx] = 0;
            if before <= 1e-8 && after > 1e-8 {
                self.position_open_step[ticker_idx] = Some(self.step_count);
            } else if after <= 1e-8 {
                self.position_open_step[ticker_idx] = None;
            }
        }
        // Match Env::execute_step_core: trades are stamped at the current
        // absolute step, then the observation frontier advances one step.
        self.step_count += 1;
        Ok(())
    }

    pub(super) fn revalue_account_with_prices(&mut self, current_prices: &[f64]) {
        let position_values: f64 = self
            .account
            .positions
            .iter()
            .enumerate()
            .map(|(i, p)| p.value_with_price(current_prices[i]))
            .sum();
        self.account.total_assets = position_values + self.account.cash;
    }

    pub(super) fn update_observation_value_with_prices(&mut self, current_prices: &[f64]) {
        self.revalue_account_with_prices(current_prices);
        self.peak_assets = self.peak_assets.max(self.account.total_assets);
    }

    fn ticker_inputs(
        &self,
        ticker_idx: usize,
        prices: &[f64],
        bar_dates: &[String],
        last: usize,
    ) -> TickerObsInputs {
        let momentum = MomentumIndicators::compute(prices);

        let reports = get_cached_earnings_data_any(&self.symbols[ticker_idx]);
        let earnings = if reports.is_empty() {
            Arc::new(EarningsIndicators::empty(prices.len()))
        } else {
            Arc::new(EarningsIndicators::compute(&reports, bar_dates, prices))
        };

        let current_price = prices[last];
        let position = &self.account.positions[ticker_idx];
        let position_percent = if self.account.total_assets > 0.0 {
            position.value_with_price(current_price) / self.account.total_assets
        } else {
            0.0
        };

        let mom_20 = current_price / prices[last.saturating_sub(20)] - 1.0;

        TickerObsInputs {
            position_percent,
            appreciation: position.appreciation(current_price),
            trade_activity_ema: self.trade_activity_ema[ticker_idx],
            steps_since_trade: self.steps_since_trade[ticker_idx],
            position_age: self.position_open_step[ticker_idx]
                .map(|s| (self.step_count.saturating_sub(s) as f64 / 500.0).min(1.0))
                .unwrap_or(0.0),
            realized_weight: realized_weight(
                position.value_with_price(current_price),
                self.account.total_assets,
            ),
            mom_5: momentum.mom_5[last],
            mom_20,
            mom_60: momentum.mom_60[last],
            mom_120: momentum.mom_120[last],
            mom_accel: momentum.mom_accel[last],
            vol_adj_mom: momentum.vol_adj_mom[last],
            efficiency: momentum.efficiency[last],
            trend_strength: momentum.trend_strength[last],
            rsi: momentum.rsi[last],
            range_pos: momentum.range_pos[last],
            stoch_k: momentum.stoch_k[last],
            zscore: momentum.zscore[last],
            macd: momentum.macd[last],
            earnings_steps_since_available: earnings.steps_since_available[last],
            revenue_growth: earnings.revenue_growth[last],
            opex_growth: earnings.opex_growth[last],
            net_profit_growth: earnings.net_profit_growth[last],
            eps: earnings.eps[last],
            eps_surprise: earnings.eps_surprise[last],
        }
    }

    pub(super) fn build_observation(&self) -> Option<(Tensor, Tensor)> {
        if self
            .price_deltas
            .iter()
            .any(|d| d.len() < PRICE_DELTAS_PER_TICKER)
        {
            return None;
        }
        if self.bar_times.len() != self.prices[0].len() {
            return None;
        }

        let mut price_deltas_flat =
            Vec::with_capacity(TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER);
        for ticker_deltas in &self.price_deltas {
            for &delta in ticker_deltas.iter().take(PRICE_DELTAS_PER_TICKER) {
                price_deltas_flat.push(delta as f32);
            }
        }

        let bar_times: Vec<OffsetDateTime> = self.bar_times.iter().copied().collect();
        let bar_dates = bar_times
            .iter()
            .map(|bar| {
                format!(
                    "{:04}-{:02}-{:02}",
                    bar.year(),
                    bar.month() as u8,
                    bar.day()
                )
            })
            .collect::<Vec<_>>();
        let macro_ind = MacroIndicators::get_or_compute(&bar_times);
        let mlast = macro_ind.gdp_growth.len() - 1;

        let global = GlobalObsInputs {
            cash_percent: self.account.cash / self.account.total_assets,
            pnl: (self.account.total_assets / self.starting_cash) - 1.0,
            drawdown: if self.peak_assets > 0.0 {
                (self.account.total_assets / self.peak_assets) - 1.0
            } else {
                0.0
            },
            commissions: self.total_commissions / self.starting_cash,
            last_fill_ratio: self.last_fill_ratio,
            gdp_growth: macro_ind.gdp_growth[mlast],
            unemployment: macro_ind.unemployment[mlast],
            jobs_growth: macro_ind.jobs_growth[mlast],
            cpi_yoy: macro_ind.cpi_yoy[mlast],
            core_cpi_yoy: macro_ind.core_cpi_yoy[mlast],
            fed_funds: macro_ind.fed_funds[mlast],
            treasury_10y: macro_ind.treasury_10y[mlast],
            yield_spread: macro_ind.yield_spread[mlast],
            consumer_sentiment: macro_ind.consumer_sentiment[mlast],
            initial_claims: macro_ind.initial_claims[mlast],
            steps_to_jobs: macro_ind.steps_to_jobs[mlast],
            steps_to_cpi: macro_ind.steps_to_cpi[mlast],
            steps_to_fomc: macro_ind.steps_to_fomc[mlast],
            steps_to_gdp: macro_ind.steps_to_gdp[mlast],
        };

        let tickers: Vec<TickerObsInputs> = (0..TICKERS_COUNT as usize)
            .map(|ticker_idx| {
                let prices: Vec<f64> = self.prices[ticker_idx].iter().copied().collect();
                let last = prices.len() - 1;
                self.ticker_inputs(ticker_idx, &prices, &bar_dates, last)
            })
            .collect();

        let static_obs = build_static_obs(&global, &tickers);

        let price_deltas_tensor = Tensor::from_slice(&price_deltas_flat)
            .view([1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        let static_obs_tensor =
            Tensor::from_slice(&static_obs).view([1, STATIC_OBSERVATIONS as i64]);

        Some((price_deltas_tensor, static_obs_tensor))
    }
}
