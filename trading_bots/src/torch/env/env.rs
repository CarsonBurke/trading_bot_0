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
            ACTION_HISTORY_LEN, PRICE_DELTAS_PER_TICKER, RETROACTIVE_BUY_REWARD,
            STATIC_OBSERVATIONS, STEPS_PER_EPISODE, TICKERS_COUNT,
        },
        ppo::NPROCS,
    },
    types::Account,
    utils::{create_folder_if_not_exists, get_mapped_price_deltas},
};

#[derive(Debug, Clone)]
pub(super) struct BuyLot {
    pub step: usize,
    pub price: f64,
    pub quantity: f64,
}

pub struct Env {
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
    pub(super) last_traded_step: Vec<usize>,
}

impl Env {
    pub(super) const STARTING_CASH: f64 = 10_000.0;

    pub fn new(random_start: bool) -> Self {
        let tickers = vec![
            "TSLA".to_string(),
            "AAPL".to_string(),
            "AMD".to_string(),
            "INTC".to_string(),
            "MSFT".to_string(),
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

        let total_data_length = prices[0].len();

        let num_tickers = tickers.len();
        Self {
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
            action_history: VecDeque::with_capacity(5),
            episode_start_offset: 0,
            total_data_length,
            random_start,
            buy_lots: vec![VecDeque::new(); num_tickers],
            retroactive_rewards: HashMap::new(),
            peak_assets: Self::STARTING_CASH,
            last_reward: 0.0,
            last_fill_ratio: 1.0,
            last_traded_step: vec![0; num_tickers],
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

            self.action_history.push_back(actions.clone());
            if self.action_history.len() > ACTION_HISTORY_LEN {
                self.action_history.pop_front();
            }

            let (buy_sell_actions, hold_actions) = actions.split_at(TICKERS_COUNT as usize);

            let (total_commission, trade_sell_reward) =
                self.trade_by_delta_percent_with_hold(buy_sell_actions, hold_actions, absolute_step);
            let reward = self.get_index_benchmark_pnl_reward(absolute_step, total_commission)
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
                self.episode_history.hold_actions[index].push(hold_actions[index]);
                self.episode_history.raw_actions[index].push(buy_sell_actions[index]);
            }
            self.episode_history.cash.push(self.account.cash);
            self.episode_history.rewards.push(reward);

            if is_done == 1.0 {
                self.handle_episode_end(absolute_step);
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
            "{} {} - Total Assets: {} ({}) cumulative reward {:.2} | Index: {} | Outperformance: {} | Commissions: {} | tickers {:?} time {:.2}s",
            "Episode".bright_blue(),
            self.episode.to_string().bright_blue().bold(),
            format!("${:.2}", self.account.total_assets).bright_white().bold(),
            strategy_str,
            self.episode_history.rewards.iter().sum::<f64>(),
            index_str,
            outperf_str,
            format!("${:.2}", self.episode_history.total_commissions).yellow(),
            self.tickers,
            Instant::now().duration_since(self.episode_start).as_secs_f32()
        );

        self.episode_history
            .record(self.episode, &self.tickers, &self.prices, self.episode_start_offset);
        self.meta_history
            .record(&self.episode_history, outperformance);

        if self.episode % 5 == 0 {
            self.meta_history.chart(self.episode);
        }

        self.episode_start = Instant::now();
        self.episode += 1;
    }

    fn get_is_done(&self) -> f32 {
        if self.step + 2 > self.max_step {
            println!("is done");
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
        self.last_traded_step.fill(self.episode_start_offset);

        let (price_deltas, static_obs) = self.get_next_obs();
        let price_deltas_tensor = Tensor::from_slice(&price_deltas)
            .view([1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        let static_obs_tensor =
            Tensor::from_slice(&static_obs).view([1, STATIC_OBSERVATIONS as i64]);

        (price_deltas_tensor, static_obs_tensor)
    }

    pub fn record_inference(&self, episode: usize) {
        let infer_dir = "../infer";
        create_folder_if_not_exists(&infer_dir.to_string());

        self.episode_history
            .record_to_path(infer_dir, episode, &self.tickers, &self.prices, self.episode_start_offset);
    }
}

pub struct Step {
    pub reward: Tensor,
    pub price_deltas: Tensor,
    pub static_obs: Tensor,
    pub is_done: Tensor,
}
