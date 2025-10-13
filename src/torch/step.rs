use rand::Rng;
use std::collections::VecDeque;
use std::time::Instant;

use tch::Tensor;

use crate::{
    charts::general::simple_chart,
    data::historical::get_historical_data,
    history::{episode_tickers_combined::EpisodeHistory, meta_tickers_combined::MetaHistory},
    torch::{
        constants::{
            ACTION_HISTORY_LEN, OBSERVATION_SPACE,
            PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT,
        },
        ppo::NPROCS,
    },
    types::Account,
    utils::{create_folder_if_not_exists, get_mapped_price_deltas},
};

pub struct Env {
    pub step: usize,
    pub max_step: usize,
    tickers: Vec<String>,
    pub prices: Vec<Vec<f64>>,
    max_prices: Vec<f64>,
    price_deltas: Vec<Vec<f64>>,
    obs: Tensor,
    account: Account,
    episode_history: EpisodeHistory,
    meta_history: MetaHistory,
    episode_start: Instant,
    pub episode: usize,
    action_history: VecDeque<Vec<f64>>, // Store last 5 actions (each action is a vec of TICKERS_COUNT values)
    episode_start_offset: usize,        // Random starting point in historical data for this episode
    total_data_length: usize,           // Total length of historical data
}

impl Env {
    const STARTING_CASH: f64 = 10_000.0;

    pub fn new() -> Self {
        let tickers = vec![
            "TSLA".to_string(),
            "AAPL".to_string(),
            "AMD".to_string(),
            "INTC".to_string(),
            "MSFT".to_string(),
        ]; // vec![
           // "NVDA".to_string(),
           // vec!["TSLA", "AAPL", "AMD", "INTC", "MSFT"]
           // .choose(&mut self.rng)
           // .unwrap()
           // .to_string()
           // ];

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
            obs: Tensor::zeros(&[1], (tch::Kind::Float, tch::Device::Cpu)),
            account: Account::default(),
            episode_history: EpisodeHistory::new(tickers.len()),
            meta_history: MetaHistory::default(),
            tickers,
            episode: 0,
            episode_start: Instant::now(),
            action_history: VecDeque::with_capacity(5),
            episode_start_offset: 0,
            total_data_length,
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
            if self.action_history.len() > 5 {
                self.action_history.pop_front();
            }
            
            let (buy_sell_actions, hold_actions) = actions.split_at(TICKERS_COUNT);

            // Execute trade based on current observations
            let _ = self.trade(actions, absolute_step);

            let reward = self.get_unrealized_pnl_reward(absolute_step);

            let is_done = self.get_is_done();

            for (index, _) in self.tickers.iter().enumerate() {
                self.episode_history.positioned[index].push(
                    self.account.positions[index]
                        .value_with_price(self.prices[index][absolute_step]),
                );
            }
            self.episode_history.cash.push(self.account.cash);

            self.episode_history.rewards.push(reward);

            if is_done == 1.0 {
                println!(
                    "Episode {} - Total Assets: {:.2} cumulative reward {:.2} tickers {:?} time secs {:.2}",
                    self.episode,
                    self.account.total_assets,
                    self.episode_history.rewards.iter().sum::<f64>(),
                    self.tickers,
                    Instant::now().duration_since(self.episode_start).as_secs_f32()
                );

                self.episode_history
                    .record(self.episode, &self.tickers, &self.prices);
                self.meta_history.record(&self.episode_history);

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
            // Calculate portfolio value with new positions at next step's prices
            let next_absolute_step = absolute_step + 1;
            let total_assets_after_trade = self
                .account
                .position_values(&self.prices, next_absolute_step)
                .iter()
                .sum::<f64>()
                + self.account.cash;

            // Log return: ln(after/before) - naturally bounded and symmetric
            // 10% gain → 0.095, 10% loss → -0.105
            (total_assets_after_trade / self.account.total_assets).ln()
        } else {
            0.0
        }
    }

    fn trade(&mut self, actions: &[f64], absolute_step: usize) -> f64 {
        let mut total_reward = 0.0;

        for (ticker_index, action) in actions.iter().enumerate() {
            let current_price = self.prices[ticker_index][absolute_step];

            if *action > 0.0 {
                // buy - no reward for buying
                let max_ownership = self.account.total_assets / self.tickers.len() as f64;
                let buy_total = (max_ownership
                    - self.account.positions[ticker_index].value_with_price(current_price))
                    * (*action as f64);

                if buy_total > 0.0 {
                    self.account.cash -= buy_total;

                    let quantity = buy_total / current_price;
                    self.account.positions[ticker_index].add(current_price, quantity);

                    self.episode_history.buys[ticker_index]
                        .insert(absolute_step, (current_price, quantity));

                    let reward = (self.max_prices[ticker_index] / current_price).ln() * quantity;
                    total_reward += reward;
                }

                continue;
            }

            // sell - reward based on realized P&L
            let position = &self.account.positions[ticker_index];
            let sell_total = position.value_with_price(current_price) * (-*action as f64);

            if sell_total > 0.0 {
                let quantity = sell_total / current_price;
                let avg_cost = position.avg_price;

                // Realized return: (sell_price - cost_basis) / cost_basis
                // Using log return for time-weighted calculation
                let return_ratio = current_price / avg_cost;
                let log_return = return_ratio.ln();

                // Weight by the proportion of total assets this trade represents
                let trade_weight = sell_total / self.account.total_assets;

                // Time-weighted realized return, scaled by trade size
                // Multiply by 100 for better gradient signal
                let trade_reward = log_return * trade_weight * 100.0;
                total_reward += trade_reward;

                self.account.cash += sell_total;
                self.account.positions[ticker_index].quantity -= quantity;

                self.episode_history.sells[ticker_index]
                    .insert(absolute_step, (current_price, quantity));
            }

            continue;
        }

        total_reward
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
                // Pad with zeros if we don't have 5 actions yet
                for _ in 0..(TICKERS_COUNT * 2) {
                    static_obs.push(0.0f32);
                }
            }
        }

        (price_deltas, static_obs)
    }

    pub fn reset(&mut self) -> (Tensor, Tensor) {
        // Randomly select starting point in historical data
        // Leave at least 1500 steps for a full episode + observations buffer
        let min_episode_length = 1500;
        let max_start = self
            .total_data_length
            .saturating_sub(min_episode_length + PRICE_DELTAS_PER_TICKER);

        let mut rng = rand::thread_rng();
        self.episode_start_offset = if max_start > PRICE_DELTAS_PER_TICKER {
            rng.gen_range(PRICE_DELTAS_PER_TICKER..max_start)
        } else {
            PRICE_DELTAS_PER_TICKER
        };

        self.step = 0;
        // Episode runs from offset to end of data (or max length)
        self.max_step =
            (self.total_data_length - self.episode_start_offset).min(min_episode_length) - 2;

        self.account = Account::new(Self::STARTING_CASH, self.tickers.len());

        self.episode_start = Instant::now();

        self.episode_history = EpisodeHistory::new(self.tickers.len());

        self.action_history.clear();

        if self.episode == 0 {
            create_folder_if_not_exists(&"training/data".to_string());
            for (ticker_index, _) in self.tickers.iter().enumerate() {
                let _ = simple_chart(
                    &"training/data".to_string(),
                    format!("price_observations_{}", ticker_index).as_str(),
                    &self.price_deltas[ticker_index],
                );
            }
        }

        // Return initial observation as two separate tensors
        let (price_deltas, static_obs) = self.get_next_obs();
        let price_deltas_tensor = Tensor::from_slice(&price_deltas)
            .view([1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        let static_obs_tensor =
            Tensor::from_slice(&static_obs).view([1, STATIC_OBSERVATIONS as i64]);

        (price_deltas_tensor, static_obs_tensor)
    }
}

pub struct Step {
    pub reward: Tensor,
    pub price_deltas: Tensor,
    pub static_obs: Tensor,
    pub is_done: Tensor,
}

pub struct ProcValues {
    pub reward: f64,
    pub obs: Vec<f32>,
    pub is_done: bool,
}
