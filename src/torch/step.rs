use rand::Rng;
use std::collections::VecDeque;
use std::time::Instant;

use colored::Colorize;
use tch::Tensor;

use crate::{
    data::historical::get_historical_data,
    history::{episode_tickers_combined::EpisodeHistory, meta_tickers_combined::MetaHistory},
    torch::{
        constants::{
            ACTION_HISTORY_LEN, ACTION_THRESHOLD, COMMISSION_RATE, PRICE_DELTAS_PER_TICKER,
            STATIC_OBSERVATIONS, STEPS_PER_EPISODE, TICKERS_COUNT,
        },
        ppo::NPROCS,
    },
    types::Account,
    utils::{create_folder_if_not_exists, get_mapped_price_deltas},
};

// Reward scaling factors
const REWARD_SCALE: f64 = 1.0; // Scale rewards for better gradient signal
const SHARPE_LAMBDA: f64 = 100.0;
const SORTINO_LAMBDA: f64 = 100.0;
const RISK_ADJUSTED_REWARD_LAMBDA: f64 = 2.0;

pub struct Env {
    pub step: usize,
    pub max_step: usize,
    pub tickers: Vec<String>,
    pub prices: Vec<Vec<f64>>,
    max_prices: Vec<f64>,
    price_deltas: Vec<Vec<f64>>,
    account: Account,
    episode_history: EpisodeHistory,
    pub meta_history: MetaHistory,
    episode_start: Instant,
    pub episode: usize,
    action_history: VecDeque<Vec<f64>>,
    episode_start_offset: usize,
    total_data_length: usize,
    random_start: bool,
}

impl Env {
    const STARTING_CASH: f64 = 10_000.0;

    pub fn new(random_start: bool) -> Self {
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
        let max_prices = prices
            .iter()
            .map(|price| {
                *price
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
            })
            .collect();
        let price_deltas = get_mapped_price_deltas(&mapped_bars);

        let total_data_length = prices[0].len();

        Self {
            step: 0,
            max_step: total_data_length - 2,
            max_prices,
            prices,
            price_deltas,
            account: Account::default(),
            episode_history: EpisodeHistory::new(tickers.len()),
            meta_history: MetaHistory::default(),
            tickers,
            episode: 0,
            episode_start: Instant::now(),
            action_history: VecDeque::with_capacity(5),
            episode_start_offset: 0,
            total_data_length,
            random_start,
        }
    }

    pub fn step(&mut self, all_actions: Vec<Vec<f64>>) -> Step {
        let mut rewards = Vec::with_capacity(NPROCS as usize);
        let mut is_dones = Vec::with_capacity(NPROCS as usize);
        let mut all_price_deltas = Vec::new();
        let mut all_static_obs = Vec::new();

        // Loop through processes
        for actions in all_actions.iter() {
            // Update portfolio value with current step's prices (offset by episode start)
            let absolute_step = self.episode_start_offset + self.step;
            self.account.update_total(&self.prices, absolute_step);

            // Store actions in history
            self.action_history.push_back(actions.clone());
            if self.action_history.len() > ACTION_HISTORY_LEN {
                self.action_history.pop_front();
            }

            let (buy_sell_actions, hold_actions) = actions.split_at(TICKERS_COUNT as usize);

            // Execute trade based on current observations (only use buy/sell actions)
            self.trade(buy_sell_actions, absolute_step);

            let reward = self.get_risk_adjusted_benchmarked_reward(absolute_step);

            let is_done = self.get_is_done();

            for (index, _) in self.tickers.iter().enumerate() {
                self.episode_history.positioned[index].push(
                    self.account.positions[index]
                        .value_with_price(self.prices[index][absolute_step]),
                );
                self.episode_history.hold_actions[index].push(hold_actions[index]);
                self.episode_history.raw_actions[index].push(buy_sell_actions[index]);
            }
            self.episode_history.cash.push(self.account.cash);

            self.episode_history.rewards.push(reward);

            if is_done == 1.0 {
                let start_price = self.prices[0][self.episode_start_offset];
                let end_price = self.prices[0][absolute_step];
                let buy_hold_return = (end_price / start_price - 1.0) * 100.0;
                let strategy_return =
                    (self.account.total_assets / Self::STARTING_CASH - 1.0) * 100.0;
                let outperformance = strategy_return - buy_hold_return;

                let strategy_str = if strategy_return >= 0.0 {
                    format!("{:.2}%", strategy_return).green()
                } else {
                    format!("{:.2}%", strategy_return).red()
                };

                let buy_hold_str = if buy_hold_return >= 0.0 {
                    format!("{:.2}%", buy_hold_return).cyan()
                } else {
                    format!("{:.2}%", buy_hold_return).red()
                };

                let outperf_str = if outperformance > 0.0 {
                    format!("+{:.2}%", outperformance).bright_green().bold()
                } else if outperformance < 0.0 {
                    format!("{:.2}%", outperformance).bright_red().bold()
                } else {
                    format!("{:.2}%", outperformance).yellow()
                };

                println!(
                    "{} {} - Total Assets: {} ({}) cumulative reward {:.2} | Buy&Hold: {} | Outperformance: {} | tickers {:?} time {:.2}s",
                    "Episode".bright_blue(),
                    self.episode.to_string().bright_blue().bold(),
                    format!("${:.2}", self.account.total_assets).bright_white().bold(),
                    strategy_str,
                    self.episode_history.rewards.iter().sum::<f64>(),
                    buy_hold_str,
                    outperf_str,
                    self.tickers,
                    Instant::now().duration_since(self.episode_start).as_secs_f32()
                );

                self.episode_history
                    .record(self.episode, &self.tickers, &self.prices);
                self.meta_history
                    .record(&self.episode_history, outperformance);

                if self.episode % 5 == 0 {
                    self.meta_history.chart(self.episode);
                }

                self.episode_start = Instant::now();

                self.episode += 1;
            }

            rewards.push(reward);
            is_dones.push(is_done);

            let (price_deltas, static_obs) = self.get_next_obs();
            all_price_deltas.push(price_deltas);
            all_static_obs.push(static_obs);
        }

        // Create separate tensors for price deltas and static observations
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

    fn get_unrealized_pnl_reward(&self, absolute_step: usize) -> f64 {
        if self.step + 1 < self.max_step && self.account.total_assets > 0.0 {
            // Current portfolio value with new positions at next step's prices
            let next_absolute_step = absolute_step + 1;
            let total_assets_after_trade = self
                .account
                .position_values(&self.prices, next_absolute_step)
                .iter()
                .sum::<f64>()
                + self.account.cash;

            (total_assets_after_trade / self.account.total_assets).ln()
        } else {
            0.0
        }
    }

    /// Is really just a shifted version of get_unrealized_pnl_reward, need to rethink this
    #[deprecated]
    fn get_excess_returns_reward(&self, absolute_step: usize) -> f64 {
        if self.step + 1 < self.max_step && self.account.total_assets > 0.0 {
            let next_absolute_step = absolute_step + 1;

            // Calculate strategy log return
            let total_assets_after_trade = self
                .account
                .position_values(&self.prices, next_absolute_step)
                .iter()
                .sum::<f64>()
                + self.account.cash;
            let strategy_log_return = (total_assets_after_trade / self.account.total_assets).ln();

            // Calculate buy-and-hold log return (using first ticker as benchmark)
            let current_price = self.prices[0][absolute_step];
            let next_price = self.prices[0][next_absolute_step];
            let buy_hold_log_return = (next_price / current_price).ln();

            // Reward excess return over benchmark
            (strategy_log_return - buy_hold_log_return) * REWARD_SCALE
        } else {
            0.0
        }
    }

    /// Sharpe ratio-like risk-adjusted reward
    fn get_sharpe_ratio_adjusted_reward(&self, absolute_step: usize) -> f64 {
        if self.step + 1 < self.max_step && self.account.total_assets > 0.0 {
            let next_absolute_step = absolute_step + 1;

            // Calculate current step's return
            let total_assets_after_trade = self
                .account
                .position_values(&self.prices, next_absolute_step)
                .iter()
                .sum::<f64>()
                + self.account.cash;
            let step_return = (total_assets_after_trade / self.account.total_assets).ln();

            let sharpe_ratio = step_return - SHARPE_LAMBDA * step_return * step_return;
            sharpe_ratio * REWARD_SCALE
        } else {
            0.0
        }
    }

    fn get_sortino_ratio_adjusted_reward(&self, absolute_step: usize) -> f64 {
        if self.step + 1 < self.max_step && self.account.total_assets > 0.0 {
            let next_absolute_step = absolute_step + 1;

            // Calculate current step's return
            let total_assets_after_trade = self
                .account
                .position_values(&self.prices, next_absolute_step)
                .iter()
                .sum::<f64>()
                + self.account.cash;
            let step_return = (total_assets_after_trade / self.account.total_assets).ln();

            let downside = if step_return < 0.0 {
                step_return * step_return
            } else {
                0.0
            };
            let sortino_ratio = step_return - SORTINO_LAMBDA * downside;
            sortino_ratio * REWARD_SCALE
        } else {
            0.0
        }
    }

    fn get_risk_adjusted_benchmarked_reward(&self, absolute_step: usize) -> f64 {
        if self.step + 1 < self.max_step && self.account.total_assets > 0.0 {
            let next_absolute_step = absolute_step + 1;

            // Calculate current step's return
            let total_assets_after_trade = self
                .account
                .position_values(&self.prices, next_absolute_step)
                .iter()
                .sum::<f64>()
                + self.account.cash;
            let step_return = (total_assets_after_trade / self.account.total_assets).ln();

            // First ticker as benchmark
            let price_curr = self.prices[0][absolute_step];
            let price_next = self.prices[0][next_absolute_step];
            let benchmark_return = (price_next / price_curr).ln();

            let excess_return = step_return - benchmark_return;

            let downside = if excess_return < 0.0 {
                -excess_return
            } else {
                0.0
            };
            let rar = excess_return - RISK_ADJUSTED_REWARD_LAMBDA * downside;
            rar * REWARD_SCALE
        } else {
            0.0
        }
    }

    fn trade(&mut self, actions: &[f64], absolute_step: usize) {

        for (ticker_index, action) in actions.iter().enumerate() {
            // Filter out weak signals below threshold
            if action.abs() < ACTION_THRESHOLD {
                continue;
            }

            let price = self.prices[ticker_index][absolute_step];

            let max_ownership = self.account.total_assets / self.tickers.len() as f64;
            let target = max_ownership * ((action + 1.0) / 2.0); // Convert [-1, 1] to [0, 1]
            let current_value = self.account.positions[ticker_index].value_with_price(price);

            let desired_delta = target - current_value;

            // BUY
            if desired_delta > 0.0 {
                let quantity = desired_delta / price;
                let cost = quantity * COMMISSION_RATE;
                let total_cost = desired_delta + cost;

                if total_cost > self.account.cash {
                    continue;
                }

                self.account.cash -= total_cost;
                self.account.positions[ticker_index].add(price, quantity);

                self.episode_history.buys[ticker_index].insert(absolute_step, (price, quantity));
            }
            // SELL
            else if desired_delta < 0.0 {
                let position_value = current_value;
                let desired_sell = -desired_delta;
                let trade_value = desired_sell.min(position_value);

                if trade_value <= 0.0 {
                    continue;
                }

                let quantity = trade_value / price;
                let cost = quantity * COMMISSION_RATE;

                self.account.cash += trade_value - cost;
                self.account.positions[ticker_index].quantity -= quantity;

                self.episode_history.sells[ticker_index].insert(absolute_step, (price, quantity));
            }
        }
    }

    fn get_is_done(&self) -> f32 {
        if self.step + 2 > self.max_step {
            println!("is done");
            1.0
        } else {
            0.0
        }
    }

    fn get_next_obs(&self) -> (Vec<f32>, Vec<f32>) {
        // Return two separate vectors: (price_deltas, static_obs)

        // Part 1: Price deltas for all tickers (will go through convolutions)
        let mut price_deltas = Vec::with_capacity(TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER);

        let absolute_step = self.episode_start_offset + self.step;

        for ticker_price_deltas in self.price_deltas.iter() {
            // Get historical price deltas (using absolute position in data)
            let start_idx = absolute_step.saturating_sub(PRICE_DELTAS_PER_TICKER - 1);
            let end_idx = (absolute_step + 1).min(ticker_price_deltas.len());
            let slice = &ticker_price_deltas[start_idx..end_idx];
            let to_take = slice.len().min(PRICE_DELTAS_PER_TICKER);

            // Pad with zeros if not enough historical data
            let padding_needed = PRICE_DELTAS_PER_TICKER - to_take;
            if padding_needed > 0 {
                price_deltas.extend(std::iter::repeat(0.0f32).take(padding_needed));
            }

            // Add price deltas (reversed to get most recent first)
            price_deltas.extend(slice.iter().rev().take(to_take).map(|&x| x as f32));
        }

        // Part 2: Static observations (will bypass convolutions)
        let mut static_obs = Vec::with_capacity(STATIC_OBSERVATIONS);

        let cash_percent = (self.account.cash / self.account.total_assets) as f32;
        let position_percents = self.account.position_percents(&self.prices, absolute_step);

        // Current step (normalized)
        static_obs.push((self.step + 1) as f32 / self.max_step as f32);

        // Cash percent
        static_obs.push(cash_percent);

        // Position percents for each ticker
        for position_percent in position_percents {
            static_obs.push(position_percent as f32);
        }

        // Action history, padded with zeros if needed
        for i in 0..ACTION_HISTORY_LEN {
            if i < self.action_history.len() {
                let action_idx = self.action_history.len() - 1 - i; // Most recent first
                for &action in &self.action_history[action_idx] {
                    static_obs.push(action as f32);
                }
            } else {
                // Pad with zeros if we don't have ACTION_HISTORY_LEN actions yet
                for _ in 0..(TICKERS_COUNT * 2) {
                    static_obs.push(0.0f32);
                }
            }
        }

        (price_deltas, static_obs)
    }

    pub fn reset(&mut self) -> (Tensor, Tensor) {
        // Leave at least STEPS_PER_EPISODE steps for a full episode + observations buffer

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
        // Episode runs from offset to end of data (or max length)
        self.max_step =
            (self.total_data_length - self.episode_start_offset).min(STEPS_PER_EPISODE) - 2;

        self.account = Account::new(Self::STARTING_CASH, self.tickers.len());

        self.episode_start = Instant::now();

        self.episode_history = EpisodeHistory::new(self.tickers.len());

        self.action_history.clear();

        // Return initial observation as two separate tensors
        let (price_deltas, static_obs) = self.get_next_obs();
        let price_deltas_tensor = Tensor::from_slice(&price_deltas)
            .view([1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        let static_obs_tensor =
            Tensor::from_slice(&static_obs).view([1, STATIC_OBSERVATIONS as i64]);

        (price_deltas_tensor, static_obs_tensor)
    }

    pub fn record_inference(&self, episode: usize) {
        let infer_dir = "infer";
        create_folder_if_not_exists(&infer_dir.to_string());

        self.episode_history
            .record_to_path(infer_dir, episode, &self.tickers, &self.prices);
    }
}

pub struct Step {
    pub reward: Tensor,
    pub price_deltas: Tensor,
    pub static_obs: Tensor,
    pub is_done: Tensor,
}
