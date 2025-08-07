use hashbrown::HashMap;
use tch::{nn, nn::Module, Tensor};

use crate::charts::general::{assets_chart, buy_sell_chart, reward_chart};
use crate::constants::files::TRAINING_PATH;
use crate::constants::TICKERS;
use crate::types::MappedHistorical;
use crate::utils::create_folder_if_not_exists;

pub mod train;

pub const OBSERVATION_DIM: i64 = 10; // price, volume, moving averages, etc.
pub const ACTION_DIM: i64 = 3; // buy, sell, hold
pub const HIDDEN_DIM: i64 = 64;
pub const LEARNING_RATE: f64 = 1e-4;
pub const GAMMA: f64 = 0.99;
pub const EPSILON: f64 = 0.2;
pub const VALUE_COEF: f64 = 0.5;
pub const ENTROPY_COEF: f64 = 0.01;

pub struct ActorCritic {
    shared: nn::Sequential,
    actor_head: nn::Linear,
    critic_head: nn::Linear,
}

impl ActorCritic {
    fn new(vs: &nn::Path) -> Self {
        let shared = nn::seq()
            .add(nn::linear(
                vs / "fc1",
                OBSERVATION_DIM,
                HIDDEN_DIM,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                vs / "fc2",
                HIDDEN_DIM,
                HIDDEN_DIM,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu());

        let actor_head = nn::linear(vs / "actor", HIDDEN_DIM, ACTION_DIM, Default::default());
        let critic_head = nn::linear(vs / "critic", HIDDEN_DIM, 1, Default::default());

        ActorCritic {
            shared,
            actor_head,
            critic_head,
        }
    }

    fn forward(&self, obs: &Tensor) -> (Tensor, Tensor) {
        let shared_features = self.shared.forward(obs);
        let action_logits = self.actor_head.forward(&shared_features);
        let value = self.critic_head.forward(&shared_features);
        (action_logits, value)
    }
}

pub struct TradingEnvironment {
    prices: Vec<f64>,
    current_step: usize,
    position: f64,
    cash: f64,
    initial_cash: f64,
}

impl TradingEnvironment {
    fn new(ticker_count: usize) -> Self {
        TradingEnvironment {
            prices: vec![],
            current_step: 10,
            position: 0.0,
            cash: 10000.0,
            initial_cash: 10000.0,
        }
    }

    fn reset(&mut self, ticker_count: usize, prices: Vec<f64>) -> Tensor {
        self.prices = prices;
        self.current_step = 10;
        self.position = 0.0;
        self.cash = self.initial_cash;
        self.get_observation()
    }

    fn get_observation(&self) -> Tensor {
        let mut obs = vec![];

        // Current price - normalize
        obs.push((self.prices[self.current_step] / 100.0) as f32);

        // Price changes - normalize
        for i in 1..5 {
            if self.current_step >= i {
                obs.push(
                    ((self.prices[self.current_step] - self.prices[self.current_step - i]) / 100.0)
                        as f32,
                );
            } else {
                obs.push(0.0);
            }
        }

        // Simple moving averages - normalize
        let ma5 = self.prices[self.current_step.saturating_sub(5)..=self.current_step]
            .iter()
            .sum::<f64>()
            / 5.0;
        let ma10 = self.prices[self.current_step.saturating_sub(10)..=self.current_step]
            .iter()
            .sum::<f64>()
            / 10.0;

        obs.push((ma5 / 100.0) as f32);
        obs.push((ma10 / 100.0) as f32);
        obs.push((self.position / 100.0) as f32); // normalize position
        obs.push((self.cash / self.initial_cash) as f32);

        // Ensure we have exactly OBSERVATION_DIM elements
        while obs.len() < OBSERVATION_DIM as usize {
            obs.push(0.0);
        }
        obs.truncate(OBSERVATION_DIM as usize);

        Tensor::from_slice(&obs).view([1, OBSERVATION_DIM])
    }

    fn learn_step(
        &mut self,
        action: i64,
        history: &mut History,
        ticker_index: usize,
    ) -> (Tensor, f64) {
        let current_price = self.prices[self.current_step];
        let portfolio_value_before = self.cash + self.position * current_price;

        match action {
            0 => {
                // Buy
                let shares_to_buy = (self.cash * 0.1 / current_price).floor();
                self.position += shares_to_buy;
                self.cash -= shares_to_buy * current_price;

                history.buys[ticker_index]
                    .insert(self.current_step, (current_price, shares_to_buy));
            }
            1 => {
                // Sell
                let shares_to_sell = (self.position * 0.1).floor();
                self.position -= shares_to_sell;
                self.cash += shares_to_sell * current_price;

                history.sells[ticker_index]
                    .insert(self.current_step, (current_price, shares_to_sell));
            }
            _ => {} // Hold
        }

        self.current_step += 1;

        let new_price = self.prices[self.current_step];
        let portfolio_value_after = self.cash + self.position * new_price;
        let reward =
            ((portfolio_value_after - portfolio_value_before) / self.initial_cash).clamp(-1.0, 1.0);

        let next_obs = self.get_observation();

        history.positioned[ticker_index].push(self.position);
        history.cash[ticker_index].push(self.cash);
        history.rewards[ticker_index].push(reward);

        // if self.current_step % 100 == 0 {
        //     println!("total assets {}", self.cash + self.position * new_price);
        // }

        (next_obs, reward)
    }
}

pub struct History {
    pub buys: Vec<HashMap<usize, (f64, f64)>>,
    pub sells: Vec<HashMap<usize, (f64, f64)>>,
    pub positioned: Vec<Vec<f64>>,
    pub cash: Vec<Vec<f64>>,
    pub rewards: Vec<Vec<f64>>,
}

impl History {
    pub fn new(ticker_count: usize) -> Self {
        History {
            buys: vec![HashMap::new(); ticker_count],
            sells: vec![HashMap::new(); ticker_count],
            positioned: vec![vec![]; ticker_count],
            cash: vec![vec![]; ticker_count],
            rewards: vec![vec![]; ticker_count],
        }
    }

    pub fn record(&self, generation: u32, mapped_historical: &MappedHistorical) {

        let base_dir = format!("{TRAINING_PATH}/gens/{}", generation);
        create_folder_if_not_exists(&base_dir);

        for (ticker_index, bars) in mapped_historical.iter().enumerate() {
            let prices = bars.iter().map(|bar| bar.close).collect::<Vec<f64>>();

            let ticker = TICKERS[ticker_index].to_string();
            let ticker_dir = format!("{TRAINING_PATH}/gens/{}/{ticker}", generation);
            create_folder_if_not_exists(&ticker_dir);

            let ticker_buy_indexes = &self.buys[ticker_index];
            let ticker_sell_indexes = &self.sells[ticker_index];
            let _ = buy_sell_chart(
                &ticker_dir,
                &prices,
                ticker_buy_indexes,
                ticker_sell_indexes,
            );

            let ticker_reward_indexes = &self.rewards[ticker_index];
            let _ = reward_chart(&ticker_dir, ticker_reward_indexes);

            let positioned_assets = &self.positioned[ticker_index];
            let cash_indexes = &self.cash[ticker_index];
            let total_assets = positioned_assets
                .iter()
                .zip(cash_indexes.iter())
                .map(|(a, b)| a + b)
                .collect::<Vec<f64>>();

            let _ = assets_chart(
                &ticker_dir,
                &total_assets,
                &cash_indexes,
                Some(positioned_assets),
            );
        }
    }

    /// The ticker with the lowest final assets with the assets amount
    pub fn ticker_lowest_final_assets(&self) -> (usize, f64) {
        self.cash
            .iter()
            .enumerate()
            .zip(self.positioned.iter())
            .map(|((cash_index, cash), positioned)| {
                (cash_index, cash.last().unwrap() + positioned.last().unwrap())
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
    }
    
    /// Avg final assets across participating tickers
    pub fn avg_final_assets(&self) -> f64 {
        self.cash
            .iter()
            .zip(self.positioned.iter())
            .map(|(cash, positioned)| cash.last().unwrap() + positioned.last().unwrap())
            .sum::<f64>()
            / self.cash.len() as f64
    }
}

pub struct PPOBuffer {
    observations: Vec<Tensor>,
    actions: Vec<i64>,
    rewards: Vec<f64>,
    values: Vec<f64>,
    log_probs: Vec<f64>,
    advantages: Vec<f64>,
    returns: Vec<f64>,
}

impl PPOBuffer {
    fn new() -> Self {
        PPOBuffer {
            observations: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
            values: Vec::new(),
            log_probs: Vec::new(),
            advantages: Vec::new(),
            returns: Vec::new(),
        }
    }

    fn add(&mut self, obs: Tensor, action: i64, reward: f64, value: f64, log_prob: f64) {
        self.observations.push(obs);
        self.actions.push(action);
        self.rewards.push(reward);
        self.values.push(value);
        self.log_probs.push(log_prob);
    }

    fn compute_advantages(&mut self) {
        let mut returns = vec![0.0; self.rewards.len()];
        let mut advantages = vec![0.0; self.rewards.len()];

        let mut gae = 0.0;
        for i in (0..self.rewards.len()).rev() {
            if i == self.rewards.len() - 1 {
                returns[i] = self.rewards[i];
                advantages[i] = self.rewards[i] - self.values[i];
            } else {
                returns[i] = self.rewards[i] + GAMMA * returns[i + 1];
                let delta = self.rewards[i] + GAMMA * self.values[i + 1] - self.values[i];
                gae = delta + GAMMA * 0.95 * gae;
                advantages[i] = gae;
            }
        }

        // Normalize advantages
        let mean: f64 = advantages.iter().sum::<f64>() / advantages.len() as f64;
        let var: f64 = advantages
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>()
            / advantages.len() as f64;
        let std = (var + 1e-8).sqrt();

        for adv in &mut advantages {
            *adv = (*adv - mean) / std;
        }

        self.returns = returns;
        self.advantages = advantages;
    }

    fn clear(&mut self) {
        self.observations.clear();
        self.actions.clear();
        self.rewards.clear();
        self.values.clear();
        self.log_probs.clear();
        self.advantages.clear();
        self.returns.clear();
    }
}
