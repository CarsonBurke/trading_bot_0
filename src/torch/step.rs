use std::time::Instant;

use tch::Tensor;

use crate::{
    charts::general::simple_chart,
    data::historical::get_historical_data,
    history::{episode_tickers_combined::EpisodeHistory, meta_tickers_combined::MetaHistory},
    torch::{
        constants::{OBSERVATIONS_PER_TICKER, OBSERVATION_SPACE, TICKERS_COUNT},
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
    price_deltas: Vec<Vec<f64>>,
    obs: Tensor,
    account: Account,
    episode_history: EpisodeHistory,
    meta_history: MetaHistory,
    episode_start: Instant,
    pub episode: usize,
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
           let price_deltas = get_mapped_price_deltas(&mapped_bars);

        Self {
            step: 0,
            max_step: prices[0].len() - 2,
            prices,
            price_deltas,
            obs: Tensor::zeros(&[1], (tch::Kind::Float, tch::Device::Cpu)),
            account: Account::default(),
            episode_history: EpisodeHistory::new(tickers.len()),
            meta_history: MetaHistory::default(),
            tickers,
            episode: 0,
            episode_start: Instant::now(),
        }
    }

    pub fn step(&mut self, all_actions: Vec<Vec<f64>>) -> Step {
        let mut rewards = Vec::with_capacity((NPROCS * TICKERS_COUNT) as usize);
        let mut is_done = Vec::with_capacity((NPROCS * TICKERS_COUNT) as usize);
        let mut obs = Vec::with_capacity((NPROCS * TICKERS_COUNT) as usize * OBSERVATION_SPACE);

        // Loop through processes
        for actions in all_actions.iter() {
            self.account.update_total(&self.prices, self.step);

            // actions

            self.trade(actions);
            
            let reward = self.get_reward();

            rewards.push(reward);
            is_done.push(self.get_is_done());
            obs.push(self.get_next_obs());
            
            for (index, _) in self.tickers.iter().enumerate() {
                self.episode_history.positioned[index].push(
                    self.account.positions[index].value_with_price(self.prices[index][self.step]),
                );
            }
            self.episode_history.cash.push(self.account.cash);
            
            self.episode_history.rewards.push(reward);

            if self.get_is_done() == 1.0 {
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
        }

        let obs_flat: Vec<f32> = obs.into_iter().flatten().collect();
        // Reshape to [NPROCS, TICKERS_COUNT, OBSERVATIONS_PER_TICKER]
        let obs_tensor = Tensor::from_slice(&obs_flat).view([
            NPROCS,
            TICKERS_COUNT,
            OBSERVATIONS_PER_TICKER as i64,
        ]);

        Step {
            reward: Tensor::from_slice(&rewards),
            is_done: Tensor::from_slice(&is_done),
            obs: obs_tensor,
        }
    }

    fn trade(&mut self, actions: &[f64]) {
        for (ticker_index, action) in actions.iter().enumerate() {
            // Assume it is already hyperbolized
            // let action = (*action).min(-1).max(1);
            // println!("action {}", action);
            let current_price = self.prices[ticker_index][self.step];

            if *action > 0.0 {
                // buy
                let max_ownership = self.account.total_assets / self.tickers.len() as f64;
                let buy_total = (max_ownership
                    - self.account.positions[ticker_index].value_with_price(current_price))
                    * (*action as f64);

                if buy_total > 0.0 {
                    // println!("Buying ticker {} amount {}", ticker_index, current_price);
                    self.account.cash -= buy_total;

                    let quantity = buy_total / current_price;
                    self.account.positions[ticker_index].add(current_price, quantity);

                    self.episode_history.buys[ticker_index].insert(self.step, (current_price, quantity));
                }

                continue;
            }

            // sell
            let sell_total = self.account.positions[ticker_index].value_with_price(current_price)
                * (-*action as f64);

            if sell_total > 0.0 {
                // println!("Selling ticker {} amount {}", ticker_index, current_price);
                self.account.cash += sell_total;

                let quantity = sell_total / current_price;
                self.account.positions[ticker_index].quantity -= quantity;

                self.episode_history.sells[ticker_index].insert(self.step, (current_price, quantity));
            }

            continue;
        }
    }

    fn get_reward(&self) -> f64 {
        let mut reward = 0.0;

        // Reward based on account total assets change
        // Simple approach: return percentage change in total assets

        for (ticker_idx, ticker_prices) in self.prices.iter().enumerate() {
            if self.step < ticker_prices.len() {
                let current_price = ticker_prices[self.step];
                let previous_price = ticker_prices[self.step.saturating_sub(1)];

                if previous_price != 0.0 {
                    let price_change = (current_price - previous_price) / previous_price;
                    // Weight by position size
                    let position_value =
                        self.account.positions[ticker_idx].value_with_price(current_price);
                    reward += price_change * (position_value / self.account.total_assets);
                }
            }
        }

        reward
    }

    fn get_is_done(&self) -> f32 {
        if self.step + 2 > self.max_step {
            println!("is done");
            1.0
        } else {
            0.0
        }
    }

    fn get_next_obs(&self) -> Vec<f32> {
        // Structure: Each ticker gets OBSERVATIONS_PER_TICKER values
        // First 3 values per ticker: step info, cash percent, position percent for that ticker
        // Remaining values: historical price deltas for that ticker

        let mut data = Vec::with_capacity(OBSERVATION_SPACE);
        let cash_percent = (self.account.cash / self.account.total_assets) as f32;
        let position_percents = self.account.position_percents(&self.prices, self.step + 1);

        for (ticker_idx, ticker_price_deltas) in self.price_deltas.iter().enumerate() {
            let mut ticker_data = Vec::with_capacity(OBSERVATIONS_PER_TICKER);

            // Add metadata for this ticker (3 values)
            ticker_data.push((self.step + 1 - self.max_step) as f32);
            ticker_data.push(cash_percent);
            ticker_data.push(position_percents[ticker_idx] as f32);

            // Calculate remaining space for price deltas
            let price_delta_space = OBSERVATIONS_PER_TICKER - 3;

            // Get historical price deltas
            let start_idx = self.step.saturating_sub(price_delta_space - 1);
            let end_idx = (self.step + 1).min(ticker_price_deltas.len());
            let slice = &ticker_price_deltas[start_idx..end_idx];
            let to_take = slice.len().min(price_delta_space);

            // Pad with zeros if not enough historical data
            let padding_needed = price_delta_space - to_take;
            if padding_needed > 0 {
                ticker_data.extend(std::iter::repeat(0.0f32).take(padding_needed));
            }

            // Add price deltas (reversed to get most recent first)
            ticker_data.extend(slice.iter().rev().take(to_take).map(|&x| x as f32));

            data.extend(ticker_data);
        }

        data
    }

    pub fn reset(&mut self) -> Tensor {

        self.account = Account::new(Self::STARTING_CASH, self.tickers.len());

        self.episode_start = Instant::now();

        self.episode_history = EpisodeHistory::new(self.tickers.len());

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

        // Return initial observation properly shaped
        let obs_vec = self.get_next_obs();
        Tensor::from_slice(&obs_vec).view([1, TICKERS_COUNT, OBSERVATIONS_PER_TICKER as i64])
    }
}

pub struct Step {
    pub reward: Tensor,
    pub obs: Tensor,
    pub is_done: Tensor,
}

pub struct ProcValues {
    pub reward: f64,
    pub obs: Vec<f32>,
    pub is_done: bool,
}
