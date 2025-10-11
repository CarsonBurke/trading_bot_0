use tch::Tensor;

use crate::{
    data::historical::get_historical_data,
    history::{episode_tickers_combined::EpisodeHistory, meta_tickers_combined::MetaHistory},
    torch::{constants::{OBSERVATION_SPACE, TICKERS_COUNT}, ppo::NPROCS},
    types::Account,
    utils::get_mapped_price_deltas,
};

pub struct Env {
    step: usize,
    max_step: usize,
    tickers: Vec<String>,
    prices: Vec<Vec<f64>>,
    price_deltas: Vec<Vec<f64>>,
    obs: Tensor,
    account: Account,
    episode_history: EpisodeHistory,
    meta_history: MetaHistory,
}

impl Env {
    const STARTING_STEP: usize = OBSERVATION_SPACE;
    const STARTING_CASH: f64 = 10_000.0;
    const BUY_PERCENT: f64 = 0.05;
    const SELL_PERCENT: f64 = 0.05;

    pub fn new() -> Self {
        let tickers = vec![
            "NVDA".to_string(),
            // vec!["TSLA", "AAPL", "AMD", "INTC", "MSFT"]
            // .choose(&mut self.rng)
            // .unwrap()
            // .to_string()
        ];

        Self {
            step: 0,
            max_step: 0,
            prices: Vec::new(),
            price_deltas: Vec::new(),
            obs: Tensor::zeros(&[1], (tch::Kind::Float, tch::Device::Cpu)),
            account: Account::default(),
            episode_history: EpisodeHistory::new(tickers.len()),
            meta_history: MetaHistory::default(),
            tickers,
        }
    }

    pub fn step(&mut self, all_actions: Vec<Vec<f64>>) -> Step {
        let mut rewards = Vec::with_capacity((NPROCS * TICKERS_COUNT) as usize);
        let mut is_done = Vec::with_capacity((NPROCS * TICKERS_COUNT) as usize);
        let mut obs: Vec<Vec<f32>> = Vec::with_capacity((NPROCS * TICKERS_COUNT) as usize * OBSERVATION_SPACE);

        // Loop through processes
        for actions in all_actions.iter() {
            self.account.update_total(&self.prices, self.step);

            // actions

            self.trade(actions);

            rewards.push(self.get_reward());
            is_done.push(self.get_is_done());
            obs.push(self.get_next_obs());
        };

        let obs_flat: Vec<f32> = obs.into_iter().flatten().collect();
        let obs_tensor = Tensor::from_slice(&obs_flat);

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

            let current_price = self.prices[ticker_index][self.step];

            if *action > 0.0 {
                // buy
                let max_ownership = self.account.total_assets / self.tickers.len() as f64;
                let buy_total = (max_ownership
                    - self.account.positions[ticker_index].value_with_price(current_price))
                    * (*action as f64);

                if buy_total > 0.0 {
                    self.account.cash -= buy_total;

                    let quantity = buy_total / current_price;
                    self.account.positions[0].add(current_price, quantity);

                    self.episode_history.buys[0].insert(self.step, (current_price, quantity));
                }

                continue;
            }

            // sell
            let sell_total = self.account.positions[ticker_index].value_with_price(current_price)
                * (*action as f64);

            if sell_total > 0.0 {
                self.account.cash -= sell_total;

                let quantity = sell_total / current_price;
                self.account.positions[0].quantity -= quantity;

                self.episode_history.sells[0].insert(self.step, (current_price, quantity));
            }

            continue;
        }
    }

    fn get_reward(&self) -> f64 {
        let mut reward = 0.0;

        for ticker_prices in self.prices.iter() {
            let current_price = ticker_prices[0];
            let previous_price = ticker_prices[1];
            reward += (current_price - previous_price) / previous_price;
        }

        reward
    }

    fn get_is_done(&self) -> bool {
        self.step + 1 > self.prices[0].len()
    }

    fn get_next_obs(&self) -> Vec<f32> {
        let mut data = Vec::with_capacity(OBSERVATION_SPACE);
        
        data.push((self.step + 1 - self.max_step) as f32);
        data.push((self.account.cash / self.account.total_assets) as f32);
        data.extend(
            self.account
                .position_percents(&self.prices, self.step + 1)
                .iter()
                .map(|percent| *percent as f32)
                .collect::<Vec<f32>>(),
        );
        
        // Calculate exact amount per ticker to evenly distribute remaining space
        let remaining_space = OBSERVATION_SPACE - data.len();
        let per_ticker_space = remaining_space / self.tickers.len();

        for ticker_price_deltas in self.price_deltas.iter() {
            let start_idx = self.step.saturating_sub(per_ticker_space - 1);
            let end_idx = (self.step + 1).min(ticker_price_deltas.len());

            let slice = &ticker_price_deltas[start_idx..end_idx];
            let to_take = slice.len().min(per_ticker_space);

            // Pad with zeros if we don't have enough historical data
            let padding_needed = per_ticker_space - to_take;
            if padding_needed > 0 {
                data.extend(std::iter::repeat(0.0f32).take(padding_needed));
            }

            // Add exactly `to_take` price deltas (reversed to get most recent first)
            data.extend(slice.iter().rev().take(to_take).map(|&x| x as f32));
        }

        data
    }

    pub fn reset(&mut self) -> &Tensor {
        let mapped_bars = get_historical_data(Some(
            &self
                .tickers
                .iter()
                .map(|ticker| ticker.as_str())
                .collect::<Vec<&str>>(),
        ));
        self.prices = mapped_bars
            .iter()
            .map(|bar| bar.iter().map(|bar| bar.close).collect())
            .collect();
        self.price_deltas = get_mapped_price_deltas(&mapped_bars);

        self.account = Account::new(Self::STARTING_CASH, self.tickers.len());
        self.step = Self::STARTING_STEP;
        self.max_step = self.prices[0].len();

        &self.obs
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
