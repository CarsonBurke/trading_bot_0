use rand::seq::{IndexedRandom, SliceRandom};
use rand::Rng;
use shared::constants::{
    ACTION_COUNT as ACTION_COUNT_USIZE, ACTION_HISTORY_LEN as ACTION_HISTORY_LEN_USIZE,
    STATIC_OBSERVATIONS as STATIC_OBSERVATIONS_USIZE, TICKERS_COUNT as TICKERS_COUNT_USIZE,
};
use std::collections::VecDeque;
use std::time::Instant;

use colored::Colorize;
use tch::Tensor;

use super::earnings::EarningsIndicators;
use super::macro_ind::MacroIndicators;
use super::momentum::MomentumIndicators;
use crate::{
    data::historical::{get_historical_data, get_historical_series},
    data::universe::cached_eligible_training_universe,
    history::{episode_tickers_combined::EpisodeHistory, meta_tickers_combined::MetaHistory},
    torch::constants::{
        ACTION_HISTORY_LEN, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, STEPS_PER_EPISODE,
        TICKERS_COUNT,
    },
    types::Account,
    utils::{create_folder_if_not_exists, get_price_deltas},
};
use std::sync::Arc;

pub struct Env {
    pub env_id: usize,
    pub step: usize,
    pub max_step: usize,
    pub tickers: Vec<String>,
    pub prices: Vec<Vec<f64>>,
    pub price_deltas: Vec<Vec<f64>>,
    pub account: Account,
    pub episode_history: EpisodeHistory,
    pub meta_history: MetaHistory,
    episode_start: Instant,
    pub episode: usize,
    pub action_history: VecDeque<Vec<f64>>,
    pub episode_start_offset: usize,
    total_data_length: usize,
    random_start: bool,
    resample_tickers_on_reset: bool,
    pub peak_assets: f64,
    pub last_reward: f64,
    pub last_fill_ratio: f64,
    pub trade_activity_ema: Vec<f64>,
    pub steps_since_trade: Vec<usize>,
    pub position_open_step: Vec<Option<usize>>,
    pub ticker_perm: Vec<usize>,
    pub target_weights: Vec<f64>,
    pub realized_weights: Vec<f64>,
    pub momentum: Vec<Arc<MomentumIndicators>>,
    pub earnings: Vec<Arc<EarningsIndicators>>,
    pub macro_ind: Arc<MacroIndicators>,
    record_history_io: bool,
    gens_path: Option<String>,
}

pub const TRADE_EMA_ALPHA: f64 = 0.05; // ~40-step equivalent window

#[derive(Clone)]
pub(crate) struct EnvMarketSnapshot {
    pub tickers: Vec<String>,
    prices: Vec<Vec<f64>>,
    price_deltas: Vec<Vec<f64>>,
    momentum: Vec<Arc<MomentumIndicators>>,
    earnings: Vec<Arc<EarningsIndicators>>,
    macro_ind: Arc<MacroIndicators>,
    total_data_length: usize,
    pub ticker_perm: Vec<usize>,
}

struct EnvMarketData {
    prices: Vec<Vec<f64>>,
    price_deltas: Vec<Vec<f64>>,
    momentum: Vec<Arc<MomentumIndicators>>,
    earnings: Vec<Arc<EarningsIndicators>>,
    macro_ind: Arc<MacroIndicators>,
    total_data_length: usize,
}

pub(crate) fn sample_training_tickers(rng: &mut impl Rng) -> Vec<String> {
    let universe = cached_eligible_training_universe();
    assert!(
        universe.len() >= TICKERS_COUNT as usize,
        "need at least {} cached eligible tickers, found {}",
        TICKERS_COUNT,
        universe.len()
    );
    universe
        .choose_multiple(rng, TICKERS_COUNT as usize)
        .cloned()
        .collect()
}

fn load_market_data(tickers: &[String], log_progress: bool) -> EnvMarketData {
    if log_progress {
        eprint!("  hist..");
    }
    let ticker_refs = tickers
        .iter()
        .map(|ticker| ticker.as_str())
        .collect::<Vec<&str>>();
    let mapped_bars = get_historical_data(Some(&ticker_refs));
    let mut prices = Vec::with_capacity(tickers.len());
    let mut price_deltas = Vec::with_capacity(tickers.len());
    for (i, ticker) in tickers.iter().enumerate() {
        if let Some((cached_prices, cached_deltas)) = get_historical_series(ticker) {
            prices.push(cached_prices);
            price_deltas.push(cached_deltas);
        } else {
            prices.push(mapped_bars[i].iter().map(|bar| bar.close).collect());
            price_deltas.push(get_price_deltas(&mapped_bars[i]));
        }
    }

    if log_progress {
        eprint!("mom..");
    }
    let momentum: Vec<Arc<MomentumIndicators>> = tickers
        .iter()
        .zip(prices.iter())
        .map(|(ticker, p)| MomentumIndicators::get_or_compute(ticker, p))
        .collect();

    let total_data_length = prices[0].len();

    if log_progress {
        eprint!("dates..");
    }
    let bar_dates: Vec<Vec<String>> = mapped_bars
        .iter()
        .map(|bars| {
            bars.iter()
                .map(|b| {
                    format!(
                        "{:04}-{:02}-{:02}",
                        b.date.year(),
                        b.date.month() as u8,
                        b.date.day()
                    )
                })
                .collect()
        })
        .collect();

    if log_progress {
        eprint!("earn..");
    }
    let mut earnings: Vec<Arc<EarningsIndicators>> = Vec::with_capacity(tickers.len());
    for (i, ticker) in tickers.iter().enumerate() {
        if log_progress {
            eprint!("{}..", ticker);
        }
        let reports = crate::data::get_earnings_data_any(ticker);
        if log_progress {
            eprint!("r");
        }
        let ind = EarningsIndicators::get_or_compute(ticker, &reports, &bar_dates[i], &prices[i]);
        if log_progress {
            eprint!("i");
        }
        earnings.push(ind);
    }

    if log_progress {
        eprint!("macro..");
    }
    let macro_ind = MacroIndicators::get_or_compute(&bar_dates[0]);
    if log_progress {
        eprintln!("done");
    }

    EnvMarketData {
        prices,
        price_deltas,
        momentum,
        earnings,
        macro_ind,
        total_data_length,
    }
}

impl Env {
    pub const STARTING_CASH: f64 = 10_000.0;

    pub fn new(random_start: bool) -> Self {
        Self::new_with_recording(random_start, true, None)
    }

    pub fn new_with_tickers(tickers: Vec<String>, random_start: bool) -> Self {
        Self::new_with_tickers_recording_and_resampling(tickers, random_start, true, None, false)
    }

    pub fn new_with_recording(
        random_start: bool,
        record_history_io: bool,
        gens_path: Option<String>,
    ) -> Self {
        let rng = &mut rand::rng();
        let tickers = sample_training_tickers(rng);

        Self::new_with_tickers_recording_and_resampling(
            tickers,
            random_start,
            record_history_io,
            gens_path,
            random_start,
        )
    }

    pub fn new_with_tickers_and_recording(
        tickers: Vec<String>,
        random_start: bool,
        record_history_io: bool,
        gens_path: Option<String>,
    ) -> Self {
        Self::new_with_tickers_recording_and_resampling(
            tickers,
            random_start,
            record_history_io,
            gens_path,
            false,
        )
    }

    fn new_with_tickers_recording_and_resampling(
        tickers: Vec<String>,
        random_start: bool,
        record_history_io: bool,
        gens_path: Option<String>,
        resample_tickers_on_reset: bool,
    ) -> Self {
        let market_data = load_market_data(&tickers, true);

        let num_tickers = tickers.len();
        let mut target_weights = vec![0.0; num_tickers + 1];
        target_weights[num_tickers] = 1.0;
        let mut realized_weights = vec![0.0; num_tickers + 1];
        realized_weights[num_tickers] = 1.0;

        Self {
            env_id: 0,
            step: 0,
            max_step: market_data.total_data_length - 2,
            prices: market_data.prices,
            price_deltas: market_data.price_deltas,
            account: Account::default(),
            episode_history: EpisodeHistory::new(num_tickers),
            meta_history: MetaHistory::default(),
            tickers,
            episode: 0,
            episode_start: Instant::now(),
            action_history: VecDeque::with_capacity(ACTION_HISTORY_LEN),
            episode_start_offset: 0,
            total_data_length: market_data.total_data_length,
            random_start,
            resample_tickers_on_reset,
            peak_assets: Self::STARTING_CASH,
            last_reward: 0.0,
            last_fill_ratio: 1.0,
            trade_activity_ema: vec![0.0; num_tickers],
            steps_since_trade: vec![0; num_tickers],
            position_open_step: vec![None; num_tickers],
            ticker_perm: (0..num_tickers).collect(),
            target_weights,
            realized_weights,
            momentum: market_data.momentum,
            earnings: market_data.earnings,
            macro_ind: market_data.macro_ind,
            record_history_io,
            gens_path,
        }
    }

    fn handle_episode_end(&mut self, absolute_step: usize) {
        let mut index_return = 0.0;
        for ticker_idx in 0..self.tickers.len() {
            let start_price = self.prices[ticker_idx][self.episode_start_offset];
            let end_price = self.prices[ticker_idx][absolute_step];
            index_return += (end_price / start_price - 1.0) * 100.0;
        }
        index_return /= self.tickers.len() as f64;

        let strategy_return = (self.account.total_assets / Self::STARTING_CASH - 1.0) * 100.0;
        let outperformance = strategy_return - index_return;

        let strategy_str = if strategy_return >= 0.0 {
            format!("{:.2}%", strategy_return).green()
        } else {
            format!("{:.2}%", strategy_return).red()
        };

        let index_str = if index_return >= 0.0 {
            format!("{:.2}%", index_return).cyan()
        } else {
            format!("{:.2}%", index_return).red()
        };

        let outperf_str = if outperformance > 0.0 {
            format!("+{:.2}%", outperformance).bright_green().bold()
        } else if outperformance < 0.0 {
            format!("{:.2}%", outperformance).bright_red().bold()
        } else {
            format!("{:.2}%", outperformance).yellow()
        };

        println!(
            "{} {} [{}] - Total Assets: {} ({}) cumulative reward {:.2} | Index: {} | Outperformance: {} | Commissions: {} | tickers {:?} time {:.2}s {}",
            "Episode".bright_blue(),
            self.episode.to_string().bright_blue().bold(),
            format!("Env {}", self.env_id).bright_blue(),
            format!("${:.2}", self.account.total_assets).bright_white().bold(),
            strategy_str,
            self.episode_history.rewards.iter().sum::<f64>(),
            index_str,
            outperf_str,
            format!("${:.2}", self.episode_history.total_commissions).yellow(),
            self.tickers,
            Instant::now().duration_since(self.episode_start).as_secs_f32(),
            match self.record_history_io {
                true => "| recorded history",
                false => "",
            }
        );

        if self.record_history_io {
            if let Some(ref gp) = self.gens_path {
                self.episode_history.record_to_path(
                    gp,
                    self.episode,
                    &self.tickers,
                    &self.prices,
                    self.episode_start_offset,
                );
            } else {
                self.episode_history.record(
                    self.episode,
                    &self.tickers,
                    &self.prices,
                    self.episode_start_offset,
                );
            }
            self.meta_history
                .record(&self.episode_history, outperformance);

            if self.episode % 5 == 0 {
                if let Some(ref gp) = self.gens_path {
                    self.meta_history.write_reports(self.episode, gp);
                } else {
                    self.meta_history.write_reports_default(self.episode);
                }
            }
        }

        self.episode_start = Instant::now();
        self.episode += 1;
    }

    fn set_training_tickers(&mut self, tickers: Vec<String>) {
        if tickers == self.tickers {
            return;
        }

        let market_data = load_market_data(&tickers, false);
        let num_tickers = tickers.len();

        self.tickers = tickers;
        self.prices = market_data.prices;
        self.price_deltas = market_data.price_deltas;
        self.momentum = market_data.momentum;
        self.earnings = market_data.earnings;
        self.macro_ind = market_data.macro_ind;
        self.total_data_length = market_data.total_data_length;
        self.max_step = self.total_data_length - 2;
        self.ticker_perm = (0..num_tickers).collect();
        self.trade_activity_ema = vec![0.0; num_tickers];
        self.steps_since_trade = vec![0; num_tickers];
        self.position_open_step = vec![None; num_tickers];
    }

    fn resample_training_tickers(&mut self) {
        let mut rng = rand::rng();
        let tickers = sample_training_tickers(&mut rng);
        self.set_training_tickers(tickers);
    }

    pub(crate) fn market_snapshot(&self) -> EnvMarketSnapshot {
        EnvMarketSnapshot {
            tickers: self.tickers.clone(),
            prices: self.prices.clone(),
            price_deltas: self.price_deltas.clone(),
            momentum: self.momentum.clone(),
            earnings: self.earnings.clone(),
            macro_ind: self.macro_ind.clone(),
            total_data_length: self.total_data_length,
            ticker_perm: self.ticker_perm.clone(),
        }
    }

    fn apply_market_snapshot(&mut self, snapshot: &EnvMarketSnapshot) {
        let num_tickers = snapshot.tickers.len();

        self.tickers.clone_from(&snapshot.tickers);
        self.prices.clone_from(&snapshot.prices);
        self.price_deltas.clone_from(&snapshot.price_deltas);
        self.momentum.clone_from(&snapshot.momentum);
        self.earnings.clone_from(&snapshot.earnings);
        self.macro_ind = snapshot.macro_ind.clone();
        self.total_data_length = snapshot.total_data_length;
        self.max_step = self.total_data_length - 2;
        self.ticker_perm.clone_from(&snapshot.ticker_perm);
        self.trade_activity_ema.resize(num_tickers, 0.0);
        self.steps_since_trade.resize(num_tickers, 0);
        self.position_open_step.resize(num_tickers, None);
    }

    pub(crate) fn sample_episode_start_offset(&self) -> usize {
        let max_start = self
            .total_data_length
            .saturating_sub(STEPS_PER_EPISODE + PRICE_DELTAS_PER_TICKER);

        if self.random_start && max_start > PRICE_DELTAS_PER_TICKER {
            let mut rng = rand::rng();
            rng.random_range(PRICE_DELTAS_PER_TICKER..max_start)
        } else {
            PRICE_DELTAS_PER_TICKER
        }
    }

    pub(super) fn has_next_transition(&self) -> bool {
        self.step <= self.max_step
    }

    fn get_is_done(&self) -> f32 {
        if self.step >= self.max_step {
            1.0
        } else {
            0.0
        }
    }

    fn reset_existing_episode_state(&mut self) {
        if self.resample_tickers_on_reset && self.episode > 0 {
            self.resample_training_tickers();
        }

        self.reset_existing_episode_state_at(self.sample_episode_start_offset());
    }

    fn reset_existing_episode_state_at(&mut self, episode_start_offset: usize) {
        self.episode_start_offset = episode_start_offset;

        self.step = 0;
        self.max_step =
            (self.total_data_length - self.episode_start_offset).min(STEPS_PER_EPISODE) - 2;

        self.account = Account::new(Self::STARTING_CASH, self.tickers.len());
        self.episode_start = Instant::now();
        self.episode_history = EpisodeHistory::new(self.tickers.len());
        self.action_history.clear();

        self.peak_assets = Self::STARTING_CASH;
        self.last_reward = 0.0;
        self.last_fill_ratio = 1.0;
        self.trade_activity_ema.fill(0.0);
        self.steps_since_trade.fill(0);
        self.position_open_step.fill(None);
        let n = self.tickers.len();
        // self.target_weights = vec![1.0 / n as f64; n + 1];
        // self.target_weights[n] = 0.0; // cash starts at 0

        // Initialize to 100% cash, 0% tickers.
        self.target_weights = vec![0.0; n + 1];
        self.target_weights[n] = 1.0;
        self.realized_weights = vec![0.0; n + 1];
        self.realized_weights[n] = 1.0;

        // Shuffle ticker permutation for this episode
        let mut rng = rand::rng();
        self.ticker_perm.shuffle(&mut rng);
    }

    pub(crate) fn reset_single_to_episode(
        &mut self,
        market: &EnvMarketSnapshot,
        episode_start_offset: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        self.apply_market_snapshot(market);
        self.reset_existing_episode_state_at(episode_start_offset);
        self.ticker_perm.clone_from(&market.ticker_perm);
        self.get_next_obs()
    }

    pub(crate) fn reset_single_resampled_training_episode(&mut self) -> (Vec<f32>, Vec<f32>) {
        self.resample_training_tickers();
        self.reset_existing_episode_state_at(self.sample_episode_start_offset());
        self.get_next_obs()
    }

    pub(crate) fn reset_step_single_to_episode(
        &mut self,
        market: &EnvMarketSnapshot,
        episode_start_offset: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        self.apply_market_snapshot(market);
        self.reset_existing_episode_state_at(episode_start_offset);
        self.ticker_perm.clone_from(&market.ticker_perm);

        let (step_deltas, static_obs) = self.get_next_step_obs();
        (step_deltas.to_vec(), static_obs.to_vec())
    }

    pub(crate) fn reset_step_single_resampled_training_episode(&mut self) -> (Vec<f32>, Vec<f32>) {
        self.resample_training_tickers();
        self.reset_existing_episode_state_at(self.sample_episode_start_offset());

        let (step_deltas, static_obs) = self.get_next_step_obs();
        (step_deltas.to_vec(), static_obs.to_vec())
    }

    pub fn reset(&mut self) -> (Tensor, Tensor) {
        self.reset_existing_episode_state();

        let (price_deltas, static_obs) = self.get_next_obs();
        let price_deltas_tensor = Tensor::from_slice(&price_deltas)
            .view([1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        let static_obs_tensor =
            Tensor::from_slice(&static_obs).view([1, STATIC_OBSERVATIONS as i64]);

        (price_deltas_tensor, static_obs_tensor)
    }

    /// Reset for VecEnv - returns raw vectors instead of tensors
    pub fn reset_single(&mut self) -> (Vec<f32>, Vec<f32>) {
        self.reset_existing_episode_state();
        self.get_next_obs()
    }

    pub fn reset_step_single(&mut self) -> (Vec<f32>, Vec<f32>) {
        self.reset_existing_episode_state();

        let (step_deltas, static_obs) = self.get_next_step_obs();
        (step_deltas.to_vec(), static_obs.to_vec())
    }

    /// Single-environment step for VecEnv
    pub fn step_single(&mut self, actions: &[f64]) -> SingleStep {
        let absolute_step = self.episode_start_offset + self.step;
        self.account.update_total(&self.prices, absolute_step);

        for ema in &mut self.trade_activity_ema {
            *ema *= 1.0 - TRADE_EMA_ALPHA;
        }
        for steps in &mut self.steps_since_trade {
            *steps += 1;
        }

        let pre_total_assets = self.account.total_assets;

        let mut real_actions = [0.0; ACTION_COUNT_USIZE];
        for (perm_idx, &real_idx) in self.ticker_perm.iter().enumerate() {
            real_actions[real_idx] = actions[perm_idx];
        }

        if self.step == 0 {
            self.episode_history.action_step0 = Some(real_actions.to_vec());
        }

        if ACTION_HISTORY_LEN_USIZE > 0 {
            self.action_history.push_back(real_actions.to_vec());
            if self.action_history.len() > ACTION_HISTORY_LEN {
                self.action_history.pop_front();
            }
        }

        let _commissions = self.trade_by_target_weights(&real_actions, absolute_step);
        self.account.update_total(&self.prices, absolute_step);
        self.sync_realized_weights(absolute_step);
        let (reward, reward_per_ticker) =
            self.get_unrealized_pnl_reward_breakdown(absolute_step, pre_total_assets);

        self.last_reward = reward;
        if self.account.total_assets > self.peak_assets {
            self.peak_assets = self.account.total_assets;
        }

        let is_done = self.get_is_done();

        for (index, _) in self.tickers.iter().enumerate() {
            self.episode_history.positioned[index].push(
                self.account.positions[index].value_with_price(self.prices[index][absolute_step]),
            );
            self.episode_history.raw_actions[index].push(real_actions[index]);
            self.episode_history.target_weights[index].push(self.target_weights[index]);
        }
        self.episode_history.cash.push(self.account.cash);
        self.episode_history.rewards.push(reward);
        self.episode_history
            .cash_weight
            .push(self.target_weights[self.tickers.len()]);

        if is_done == 1.0 {
            self.episode_history.action_final = Some(real_actions.to_vec());
            self.handle_episode_end(absolute_step);
        }

        self.step += 1;
        let (price_deltas, static_obs) = self.get_next_obs();
        SingleStep {
            reward,
            reward_per_ticker,
            price_deltas,
            static_obs: static_obs.try_into().unwrap(),
            is_done,
        }
    }

    pub fn step_step_single(&mut self, actions: &[f64]) -> SingleStepStep {
        let absolute_step = self.episode_start_offset + self.step;
        self.account.update_total(&self.prices, absolute_step);

        for ema in &mut self.trade_activity_ema {
            *ema *= 1.0 - TRADE_EMA_ALPHA;
        }
        for steps in &mut self.steps_since_trade {
            *steps += 1;
        }

        let pre_total_assets = self.account.total_assets;

        let mut real_actions = [0.0; ACTION_COUNT_USIZE];
        for (perm_idx, &real_idx) in self.ticker_perm.iter().enumerate() {
            real_actions[real_idx] = actions[perm_idx];
        }

        if self.step == 0 {
            self.episode_history.action_step0 = Some(real_actions.to_vec());
        }

        if ACTION_HISTORY_LEN_USIZE > 0 {
            self.action_history.push_back(real_actions.to_vec());
            if self.action_history.len() > ACTION_HISTORY_LEN {
                self.action_history.pop_front();
            }
        }

        let _commissions = self.trade_by_target_weights(&real_actions, absolute_step);
        self.account.update_total(&self.prices, absolute_step);
        self.sync_realized_weights(absolute_step);
        let (reward, reward_per_ticker) =
            self.get_unrealized_pnl_reward_breakdown(absolute_step, pre_total_assets);

        self.last_reward = reward;
        if self.account.total_assets > self.peak_assets {
            self.peak_assets = self.account.total_assets;
        }

        let is_done = self.get_is_done();

        for (index, _) in self.tickers.iter().enumerate() {
            self.episode_history.positioned[index].push(
                self.account.positions[index].value_with_price(self.prices[index][absolute_step]),
            );
            self.episode_history.raw_actions[index].push(real_actions[index]);
            self.episode_history.target_weights[index].push(self.target_weights[index]);
        }
        self.episode_history.cash.push(self.account.cash);
        self.episode_history.rewards.push(reward);
        self.episode_history
            .cash_weight
            .push(self.target_weights[self.tickers.len()]);

        if is_done == 1.0 {
            self.episode_history.action_final = Some(real_actions.to_vec());
            self.handle_episode_end(absolute_step);
        }

        self.step += 1;
        let (step_deltas, static_obs) = self.get_next_step_obs();
        SingleStepStep {
            reward,
            reward_per_ticker,
            step_deltas,
            static_obs,
            is_done,
        }
    }

    pub fn record_inference(&self, episode: usize) {
        let infer_dir = "../infer";
        create_folder_if_not_exists(&infer_dir.to_string());

        self.episode_history.record_to_path(
            infer_dir,
            episode,
            &self.tickers,
            &self.prices,
            self.episode_start_offset,
        );
    }
}

/// Single-environment step result with raw values (for VecEnv)
pub struct SingleStep {
    pub reward: f64,
    pub reward_per_ticker: [f32; TICKERS_COUNT_USIZE],
    pub price_deltas: Vec<f32>,
    pub static_obs: [f32; STATIC_OBSERVATIONS_USIZE],
    pub is_done: f32,
}

pub struct SingleStepStep {
    pub reward: f64,
    pub reward_per_ticker: [f32; TICKERS_COUNT_USIZE],
    pub step_deltas: [f32; TICKERS_COUNT_USIZE],
    pub static_obs: [f32; STATIC_OBSERVATIONS_USIZE],
    pub is_done: f32,
}
