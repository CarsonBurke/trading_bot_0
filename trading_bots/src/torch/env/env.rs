use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use colored::Colorize;
use tch::Tensor;

use crate::{
    data::historical::get_historical_data,
    history::{episode_tickers_combined::EpisodeHistory, meta_tickers_combined::MetaHistory},
    torch::{
        constants::{
            ACTION_COUNT, ACTION_HISTORY_LEN, PRICE_DELTAS_PER_TICKER, RETROACTIVE_BUY_REWARD,
            STATIC_OBSERVATIONS, STEPS_PER_EPISODE, TICKERS_COUNT,
        },
        ppo::NPROCS,
    },
    types::Account,
    utils::{create_folder_if_not_exists, get_mapped_price_deltas},
};

/// Precomputed earnings indicators per step (from cached quarterly reports)
#[derive(Debug)]
pub(super) struct EarningsIndicators {
    pub steps_to_next: Vec<f64>, // Steps until next earnings [-1,1] normalized
    pub revenue_growth: Vec<f64>, // QoQ revenue growth [-1,1]
    pub opex_growth: Vec<f64>,   // QoQ operating expenses growth [-1,1]
    pub net_profit_growth: Vec<f64>, // QoQ net profit growth [-1,1]
    pub eps: Vec<f64>,           // EPS normalized by price
    pub eps_surprise: Vec<f64>,  // Last earnings surprise % [-1,1]
}

impl EarningsIndicators {
    pub fn empty(n: usize) -> Self {
        Self {
            steps_to_next: vec![0.0; n],
            revenue_growth: vec![0.0; n],
            opex_growth: vec![0.0; n],
            net_profit_growth: vec![0.0; n],
            eps: vec![0.0; n],
            eps_surprise: vec![0.0; n],
        }
    }

    /// Compute earnings indicators aligned to bar timestamps
    /// reports: quarterly earnings sorted oldest-first
    /// bar_dates: date strings "YYYY-MM-DD" for each bar
    /// prices: closing prices for EPS normalization
    pub fn compute(
        reports: &[crate::data::EarningsReport],
        bar_dates: &[String],
        prices: &[f64],
    ) -> Self {
        let n = bar_dates.len();
        if reports.is_empty() {
            return Self::empty(n);
        }

        let mut steps_to_next = vec![0.0; n];
        let mut revenue_growth = vec![0.0; n];
        let mut opex_growth = vec![0.0; n];
        let mut net_profit_growth = vec![0.0; n];
        let mut eps = vec![0.0; n];
        let mut eps_surprise = vec![0.0; n];

        // Find report index for each bar (most recent report before bar date)
        let mut report_idx = 0;
        for (i, bar_date) in bar_dates.iter().enumerate() {
            // Advance to most recent report before this bar
            while report_idx + 1 < reports.len() && reports[report_idx + 1].date <= *bar_date {
                report_idx += 1;
            }

            let report = &reports[report_idx];

            // Steps to next earnings (normalized: 0 = just happened, 1 = ~90 days away)
            // ~78 trading 5-min bars per day * 63 trading days per quarter ≈ 4914 steps
            if report_idx + 1 < reports.len() {
                let next_date = &reports[report_idx + 1].date;
                let days_to_next = date_diff_days(bar_date, next_date).max(0) as f64;
                steps_to_next[i] = (days_to_next / 90.0).clamp(0.0, 1.0);
            }

            revenue_growth[i] = report.revenue_growth.unwrap_or(0.0).clamp(-1.0, 1.0);
            opex_growth[i] = report.opex_growth.unwrap_or(0.0).clamp(-1.0, 1.0);
            net_profit_growth[i] = report.net_income_growth.unwrap_or(0.0).clamp(-1.0, 1.0);

            // EPS normalized by current price (yield-like)
            if let Some(e) = report.eps {
                let price = prices[i].max(1.0);
                eps[i] = (e / price * 4.0).clamp(-0.5, 0.5); // annualized, clamped
            }

            eps_surprise[i] = report.eps_surprise.unwrap_or(0.0).clamp(-1.0, 1.0);
        }

        Self {
            steps_to_next,
            revenue_growth,
            opex_growth,
            net_profit_growth,
            eps,
            eps_surprise,
        }
    }
}

fn date_diff_days(from: &str, to: &str) -> i32 {
    // Simple date diff: "YYYY-MM-DD" format
    let parse = |s: &str| -> Option<i32> {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 3 {
            return None;
        }
        let y: i32 = parts[0].parse().ok()?;
        let m: i32 = parts[1].parse().ok()?;
        let d: i32 = parts[2].parse().ok()?;
        Some(y * 365 + m * 30 + d) // approximate
    };
    match (parse(from), parse(to)) {
        (Some(f), Some(t)) => t - f,
        _ => 0,
    }
}

/// Precomputed momentum indicators (SOTA for trend prediction)
pub(super) struct MomentumIndicators {
    pub rsi: Vec<f64>,            // RSI 14-period [0,1]
    pub mom_5: Vec<f64>,          // 5-step momentum
    pub mom_60: Vec<f64>,         // 60-step momentum
    pub mom_120: Vec<f64>,        // 120-step momentum
    pub mom_accel: Vec<f64>,      // Momentum acceleration
    pub vol_adj_mom: Vec<f64>,    // Volatility-adjusted momentum
    pub range_pos: Vec<f64>,      // Position in high-low range [-1,1]
    pub zscore: Vec<f64>,         // Z-score from mean [-3,3]
    pub efficiency: Vec<f64>,     // Efficiency ratio [0,1]
    pub macd: Vec<f64>,           // MACD normalized
    pub stoch_k: Vec<f64>,        // Stochastic %K [0,1]
    pub trend_strength: Vec<f64>, // Trend consistency [0,1]
}

impl MomentumIndicators {
    pub fn compute(prices: &[f64]) -> Self {
        let n = prices.len();
        let mut rsi = vec![0.5; n];
        let mut mom_5 = vec![0.0; n];
        let mut mom_60 = vec![0.0; n];
        let mut mom_120 = vec![0.0; n];
        let mut mom_accel = vec![0.0; n];
        let mut vol_adj_mom = vec![0.0; n];
        let mut range_pos = vec![0.0; n];
        let mut zscore = vec![0.0; n];
        let mut efficiency = vec![0.0; n];
        let mut macd = vec![0.0; n];
        let mut stoch_k = vec![0.5; n];
        let mut trend_strength = vec![0.0; n];

        let mut ema_12 = prices.first().copied().unwrap_or(1.0);
        let mut ema_26 = ema_12;

        for i in 1..n {
            let p = prices[i];

            if i >= 5 {
                mom_5[i] = (p / prices[i - 5] - 1.0).clamp(-0.5, 0.5);
            }
            if i >= 60 {
                mom_60[i] = (p / prices[i - 60] - 1.0).clamp(-1.0, 1.0);
            }
            if i >= 120 {
                mom_120[i] = (p / prices[i - 120] - 1.0).clamp(-2.0, 2.0);
            }
            if i >= 10 {
                mom_accel[i] = (mom_5[i] - mom_5[i - 5]).clamp(-0.2, 0.2);
            }

            if i >= 14 {
                let (mut gains, mut losses) = (0.0, 0.0);
                for j in (i - 13)..=i {
                    let chg = prices[j] - prices[j - 1];
                    if chg > 0.0 {
                        gains += chg;
                    } else {
                        losses -= chg;
                    }
                }
                rsi[i] = (100.0 - 100.0 / (1.0 + gains / losses.max(1e-10))) / 100.0;

                let (mut up, mut down) = (0, 0);
                for j in (i - 13)..=i {
                    if prices[j] > prices[j - 1] {
                        up += 1;
                    } else if prices[j] < prices[j - 1] {
                        down += 1;
                    }
                }
                trend_strength[i] = up.max(down) as f64 / 14.0;
            }

            if i >= 20 {
                let w = &prices[i - 19..=i];
                let high = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let low = w.iter().cloned().fold(f64::INFINITY, f64::min);
                let mean: f64 = w.iter().sum::<f64>() / 20.0;
                let std = (w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 20.0)
                    .sqrt()
                    .max(1e-10);
                let range = (high - low).max(1e-10);

                range_pos[i] = ((p - low) / range * 2.0 - 1.0).clamp(-1.0, 1.0);
                stoch_k[i] = ((p - low) / range).clamp(0.0, 1.0);
                zscore[i] = ((p - mean) / std).clamp(-3.0, 3.0);
                vol_adj_mom[i] = ((p / prices[i - 20] - 1.0) / (std / mean)).clamp(-5.0, 5.0);

                let net = (p - prices[i - 20]).abs();
                let total: f64 = (i - 19..=i)
                    .map(|j| (prices[j] - prices[j - 1]).abs())
                    .sum();
                efficiency[i] = (net / total.max(1e-10)).clamp(0.0, 1.0);
            }

            ema_12 = 2.0 / 13.0 * p + (1.0 - 2.0 / 13.0) * ema_12;
            ema_26 = 2.0 / 27.0 * p + (1.0 - 2.0 / 27.0) * ema_26;
            macd[i] = ((ema_12 - ema_26) / p.max(1e-10) * 100.0).clamp(-5.0, 5.0);
        }

        Self {
            rsi,
            mom_5,
            mom_60,
            mom_120,
            mom_accel,
            vol_adj_mom,
            range_pos,
            zscore,
            efficiency,
            macd,
            stoch_k,
            trend_strength,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct BuyLot {
    pub step: usize,
    pub price: f64,
    pub quantity: f64,
}

pub struct Env {
    pub(super) env_id: usize,
    pub step: usize,
    pub max_step: usize,
    pub tickers: Vec<String>,
    pub prices: Vec<Vec<f64>>,
    pub(super) price_deltas: Vec<Vec<f64>>,
    pub(super) account: Account,
    pub episode_history: EpisodeHistory,
    pub meta_history: MetaHistory,
    episode_start: Instant,
    pub episode: usize,
    pub(super) action_history: VecDeque<Vec<f64>>,
    pub(super) episode_start_offset: usize,
    total_data_length: usize,
    random_start: bool,
    pub(super) buy_lots: Vec<VecDeque<BuyLot>>,
    pub(super) retroactive_rewards: HashMap<usize, f64>,
    pub(super) peak_assets: f64,
    pub(super) last_reward: f64,
    pub(super) last_fill_ratio: f64,
    pub(super) trade_activity_ema: Vec<f64>,
    pub(super) steps_since_trade: Vec<usize>,
    pub(super) position_open_step: Vec<Option<usize>>,
    pub(super) ticker_perm: Vec<usize>,
    pub(super) target_weights: Vec<f64>,
    /// Precomputed momentum indicators per ticker
    pub(super) momentum: Vec<MomentumIndicators>,
    /// Precomputed earnings indicators per ticker
    pub(super) earnings: Vec<EarningsIndicators>,
    record_history_io: bool,
}

pub(super) const TRADE_EMA_ALPHA: f64 = 0.05; // ~40-step equivalent window

impl Env {
    pub(super) const STARTING_CASH: f64 = 10_000.0;

    pub fn new(random_start: bool) -> Self {
        Self::new_with_recording(random_start, true)
    }

    pub fn new_with_recording(random_start: bool, record_history_io: bool) -> Self {
        let tickers = vec![
            // "TSLA".to_string(),
            // "AAPL".to_string(),
            // "AMD".to_string(),
            // "INTC".to_string(),
            // "MSFT".to_string(),
            "NVDA".to_string(),
        ];

        let mapped_bars = get_historical_data(Some(
            &tickers
                .iter()
                .map(|ticker| ticker.as_str())
                .collect::<Vec<&str>>(),
        ));
        let prices: Vec<Vec<f64>> = mapped_bars
            .iter()
            .map(|bar| bar.iter().map(|bar| bar.close).collect())
            .collect();
        let price_deltas = get_mapped_price_deltas(&mapped_bars);

        // Precompute momentum indicators for all tickers
        let momentum: Vec<MomentumIndicators> = prices
            .iter()
            .map(|p| MomentumIndicators::compute(p))
            .collect();

        let total_data_length = prices[0].len();

        // Extract bar dates for earnings alignment
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

        // Precompute earnings indicators from cached/fetched data
        let earnings: Vec<EarningsIndicators> = tickers
            .iter()
            .enumerate()
            .map(|(i, ticker)| {
                let reports = crate::data::get_earnings_data_any(ticker);
                if reports.is_empty() {
                    EarningsIndicators::empty(prices[i].len())
                } else {
                    EarningsIndicators::compute(&reports, &bar_dates[i], &prices[i])
                }
            })
            .collect();

        let num_tickers = tickers.len();
        Self {
            env_id: 0,
            step: 0,
            max_step: total_data_length - 2,
            prices,
            price_deltas,
            account: Account::default(),
            episode_history: EpisodeHistory::new(num_tickers),
            meta_history: MetaHistory::default(),
            tickers,
            episode: 0,
            episode_start: Instant::now(),
            action_history: VecDeque::with_capacity(ACTION_HISTORY_LEN),
            episode_start_offset: 0,
            total_data_length,
            random_start,
            buy_lots: vec![VecDeque::new(); num_tickers],
            retroactive_rewards: HashMap::new(),
            peak_assets: Self::STARTING_CASH,
            last_reward: 0.0,
            last_fill_ratio: 1.0,
            trade_activity_ema: vec![0.0; num_tickers],
            steps_since_trade: vec![0; num_tickers],
            position_open_step: vec![None; num_tickers],
            ticker_perm: (0..num_tickers).collect(),
            target_weights: vec![0.0; num_tickers + 1],
            momentum,
            earnings,
            record_history_io,
        }
    }

    pub fn step(&mut self, all_actions: Vec<Vec<f64>>) -> Step {
        let mut rewards = Vec::with_capacity(NPROCS as usize);
        let mut is_dones = Vec::with_capacity(NPROCS as usize);
        let mut all_price_deltas = Vec::new();
        let mut all_static_obs = Vec::new();

        for actions in all_actions.iter() {
            let absolute_step = self.episode_start_offset + self.step;
            self.account.update_total(&self.prices, absolute_step);

            // Decay trade activity EMAs (trades will add TRADE_EMA_ALPHA back)
            for ema in &mut self.trade_activity_ema {
                *ema *= 1.0 - TRADE_EMA_ALPHA;
            }
            for steps in &mut self.steps_since_trade {
                *steps += 1;
            }

            // Actions come in permuted order - map back to real ticker order
            // Last action (cash) is not permuted
            let mut real_actions = vec![0.0; ACTION_COUNT as usize];
            for (perm_idx, &real_idx) in self.ticker_perm.iter().enumerate() {
                real_actions[real_idx] = actions[perm_idx];
            }
            real_actions[TICKERS_COUNT as usize] = actions[TICKERS_COUNT as usize]; // cash

            if self.step == 0 {
                self.episode_history.action_step0 = Some(real_actions.clone());
            }

            self.action_history.push_back(real_actions.clone());
            if self.action_history.len() > ACTION_HISTORY_LEN {
                self.action_history.pop_front();
            }

            let (total_commission, trade_sell_reward) =
                self.trade_by_target_weights(&real_actions, absolute_step);
            let reward = self.get_unrealized_pnl_reward(absolute_step, total_commission)
                + if RETROACTIVE_BUY_REWARD {
                    trade_sell_reward
                } else {
                    0.0
                };

            self.last_reward = reward;
            if self.account.total_assets > self.peak_assets {
                self.peak_assets = self.account.total_assets;
            }

            let is_done = self.get_is_done();

            for (index, _) in self.tickers.iter().enumerate() {
                self.episode_history.positioned[index].push(
                    self.account.positions[index]
                        .value_with_price(self.prices[index][absolute_step]),
                );
                self.episode_history.raw_actions[index].push(real_actions[index]);
                self.episode_history.target_weights[index].push(self.target_weights[index]);
            }
            self.episode_history.cash.push(self.account.cash);
            self.episode_history.rewards.push(reward);
            // Use target cash weight (last element) for consistent charting with ticker target weights
            self.episode_history
                .cash_weight
                .push(self.target_weights[self.tickers.len()]);

            if is_done == 1.0 {
                self.handle_episode_end(absolute_step);
                self.episode_history.action_final = Some(real_actions.clone());
            }

            rewards.push(reward);
            is_dones.push(is_done);

            let (price_deltas, static_obs) = self.get_next_obs();
            all_price_deltas.push(price_deltas);
            all_static_obs.push(static_obs);
        }

        let price_deltas_flat: Vec<f32> = all_price_deltas.into_iter().flatten().collect();
        let static_obs_flat: Vec<f32> = all_static_obs.into_iter().flatten().collect();

        let price_deltas_tensor = Tensor::from_slice(&price_deltas_flat)
            .view([NPROCS, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        let static_obs_tensor =
            Tensor::from_slice(&static_obs_flat).view([NPROCS, STATIC_OBSERVATIONS as i64]);

        Step {
            reward: Tensor::from_slice(&rewards),
            is_done: Tensor::from_slice(&is_dones),
            price_deltas: price_deltas_tensor,
            static_obs: static_obs_tensor,
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
            "{} {} [{}] - Total Assets: {} ({}) cumulative reward {:.2} | Index: {} | Outperformance: {} | Commissions: {} | tickers {:?} time {:.2}s",
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
            Instant::now().duration_since(self.episode_start).as_secs_f32()
        );

        if self.record_history_io {
            self.episode_history.record(
                self.episode,
                &self.tickers,
                &self.prices,
                self.episode_start_offset,
            );
            self.meta_history
                .record(&self.episode_history, outperformance);

            if self.episode % 5 == 0 {
                self.meta_history.chart(self.episode);
            }
        }

        self.episode_start = Instant::now();
        self.episode += 1;
    }

    fn get_is_done(&self) -> f32 {
        if self.step + 2 > self.max_step {
            1.0
        } else {
            0.0
        }
    }

    pub fn reset(&mut self) -> (Tensor, Tensor) {
        let max_start = self
            .total_data_length
            .saturating_sub(STEPS_PER_EPISODE + PRICE_DELTAS_PER_TICKER);

        self.episode_start_offset = if self.random_start && max_start > PRICE_DELTAS_PER_TICKER {
            let mut rng = rand::rng();
            rng.random_range(PRICE_DELTAS_PER_TICKER..max_start)
        } else {
            PRICE_DELTAS_PER_TICKER
        };

        self.step = 0;
        self.max_step =
            (self.total_data_length - self.episode_start_offset).min(STEPS_PER_EPISODE) - 2;

        self.account = Account::new(Self::STARTING_CASH, self.tickers.len());
        self.episode_start = Instant::now();
        self.episode_history = EpisodeHistory::new(self.tickers.len());
        self.action_history.clear();

        for lot_queue in &mut self.buy_lots {
            lot_queue.clear();
        }
        self.retroactive_rewards.clear();

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

        // Shuffle ticker permutation for this episode
        let mut rng = rand::rng();
        self.ticker_perm.shuffle(&mut rng);

        let (price_deltas, static_obs) = self.get_next_obs();
        let price_deltas_tensor = Tensor::from_slice(&price_deltas)
            .view([1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        let static_obs_tensor =
            Tensor::from_slice(&static_obs).view([1, STATIC_OBSERVATIONS as i64]);

        (price_deltas_tensor, static_obs_tensor)
    }

    /// Reset for VecEnv - returns raw vectors instead of tensors
    pub fn reset_single(&mut self) -> (Vec<f32>, Vec<f32>) {
        let max_start = self
            .total_data_length
            .saturating_sub(STEPS_PER_EPISODE + PRICE_DELTAS_PER_TICKER);

        self.episode_start_offset = if self.random_start && max_start > PRICE_DELTAS_PER_TICKER {
            let mut rng = rand::rng();
            rng.random_range(PRICE_DELTAS_PER_TICKER..max_start)
        } else {
            PRICE_DELTAS_PER_TICKER
        };

        self.step = 0;
        self.max_step =
            (self.total_data_length - self.episode_start_offset).min(STEPS_PER_EPISODE) - 2;

        self.account = Account::new(Self::STARTING_CASH, self.tickers.len());
        self.episode_start = Instant::now();
        self.episode_history = EpisodeHistory::new(self.tickers.len());
        self.action_history.clear();

        for lot_queue in &mut self.buy_lots {
            lot_queue.clear();
        }
        self.retroactive_rewards.clear();

        self.peak_assets = Self::STARTING_CASH;
        self.last_reward = 0.0;
        self.last_fill_ratio = 1.0;
        self.trade_activity_ema.fill(0.0);
        self.steps_since_trade.fill(0);
        self.position_open_step.fill(None);
        let n = self.tickers.len();
        self.target_weights = vec![0.0; n + 1];
        self.target_weights[n] = 1.0;

        let mut rng = rand::rng();
        self.ticker_perm.shuffle(&mut rng);

        self.get_next_obs()
    }

    /// Single-environment step for VecEnv
    pub fn step_single(&mut self, actions: Vec<f64>) -> SingleStep {
        let absolute_step = self.episode_start_offset + self.step;
        self.account.update_total(&self.prices, absolute_step);

        for ema in &mut self.trade_activity_ema {
            *ema *= 1.0 - TRADE_EMA_ALPHA;
        }
        for steps in &mut self.steps_since_trade {
            *steps += 1;
        }

        let mut real_actions = vec![0.0; ACTION_COUNT as usize];
        for (perm_idx, &real_idx) in self.ticker_perm.iter().enumerate() {
            real_actions[real_idx] = actions[perm_idx];
        }
        real_actions[TICKERS_COUNT as usize] = actions[TICKERS_COUNT as usize];

        if self.step == 0 {
            self.episode_history.action_step0 = Some(real_actions.clone());
        }

        self.action_history.push_back(real_actions.clone());
        if self.action_history.len() > ACTION_HISTORY_LEN {
            self.action_history.pop_front();
        }

        let (commissions, trade_sell_reward) =
            self.trade_by_target_weights(&real_actions, absolute_step);
        let reward = self.get_unrealized_pnl_reward(absolute_step, commissions)
            + if RETROACTIVE_BUY_REWARD {
                trade_sell_reward
            } else {
                0.0
            };

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
            self.handle_episode_end(absolute_step);
            self.episode_history.action_final = Some(real_actions.clone());
        }

        let (price_deltas, static_obs) = self.get_next_obs();
        SingleStep {
            reward,
            price_deltas,
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

pub struct Step {
    pub reward: Tensor,
    pub price_deltas: Tensor,
    pub static_obs: Tensor,
    pub is_done: Tensor,
}

/// Single-environment step result with raw values (for VecEnv)
pub struct SingleStep {
    pub reward: f64,
    pub price_deltas: Vec<f32>,
    pub static_obs: Vec<f32>,
    pub is_done: f32,
}
