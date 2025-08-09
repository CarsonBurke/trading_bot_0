use hashbrown::HashMap;
use tch::{nn, nn::Module, Tensor};

use crate::charts::general::{assets_chart, buy_sell_chart, reward_chart, simple_chart};
use crate::constants::files::TRAINING_PATH;
use crate::constants::TICKERS;
use crate::types::{MappedHistorical, Position};
use crate::utils::create_folder_if_not_exists;

pub mod train;
pub mod check;

pub const OBSERVATION_DIM: i64 = 1000; // price, volume, moving averages, etc.
pub const ACTION_DIM: i64 = 2; // want amount, hold
pub const HIDDEN_DIM: i64 = 64;
pub const LEARNING_RATE: f64 = 1e-4;
pub const GAMMA: f64 = 0.99;
pub const EPSILON: f64 = 0.2;
pub const VALUE_COEF: f64 = 0.5;
pub const ENTROPY_COEF: f64 = 0.01;

pub struct ActionIndex;

impl ActionIndex {
    pub const WANT: usize = 0;
    pub const HOLD: usize = 1;
}

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
    position: Position,
    cash: f64,
    initial_cash: f64,
}

impl TradingEnvironment {
    fn new() -> Self {
        TradingEnvironment {
            prices: vec![],
            current_step: 500,
            position: Position::default(),
            cash: 10000.0,
            initial_cash: 10000.0,
        }
    }

    fn reset(&mut self, prices: Vec<f64>) -> Tensor {
        self.prices = prices;
        self.current_step = 500;
        self.position = Position::default();
        self.cash = self.initial_cash;
        self.get_observation()
    }

    fn get_observation(&self) -> Tensor {
        let mut obs = vec![];

        let total_assets = self
            .position
            .value_with_price(self.prices[self.current_step])
            + self.cash;
        let price = self.prices[self.current_step];

        // Simple moving averages - normalize
        let moving_average_times = vec![5, 10, 20, 50, 100, 200, 400, 800];
        for &time in &moving_average_times {
            let ma = self.prices[self.current_step.saturating_sub(time)..=self.current_step]
                .iter()
                .sum::<f64>()
                / time as f64;

            obs.push((ma / 100.0) as f32);
        }

        obs.push(self.position.appreciation(price) as f32);
        obs.push((self.cash / total_assets) as f32);

        // Price changes - normalize
        for i in 0..(OBSERVATION_DIM as usize - obs.len()) {
            if self.current_step >= i {
                if let Some(previous_price) = self.prices.get(self.current_step - i) {
                    obs.push(((self.prices[self.current_step] - previous_price) / previous_price) as f32);
                    continue;
                }
            }

            obs.push(0.0);
        }

        // Ensure we have exactly OBSERVATION_DIM elements
        while obs.len() < OBSERVATION_DIM as usize {
            obs.push(0.0);
        }
        obs.truncate(OBSERVATION_DIM as usize);

        Tensor::from_slice(&obs).view([1, OBSERVATION_DIM])
    }

    fn learn_step(
        &mut self,
        actions: &[f64],
        history: &mut GenHistory,
        ticker_index: usize,
    ) -> (Tensor, f64) {
        let current_price = self.prices[self.current_step];
        let portfolio_value_before = self.cash + self.position.value_with_price(current_price);

        let mut reward = 0.0;

        // if actions[ActionIndex::HOLD] >= 1.0 {
        //     // Do nothing
        // } else {
            // See if we should buy or sell
            let net_action = (actions[ActionIndex::WANT] - actions[ActionIndex::HOLD])/ 10.0;

            if net_action > 0.0 {
                // Buy percent of portfolio value
                let buy_total = (portfolio_value_before * net_action).floor().min(self.cash);
                if buy_total > 0.0 {
                    self.cash -= buy_total;

                    let quantity = buy_total / current_price;
                    self.position.add(current_price, quantity);

                    history.buys[ticker_index].insert(self.current_step, (current_price, quantity));
                }
            } else if net_action < 0.0 {
                // Sell percent of asset
                let position_value = self.position.value_with_price(current_price);
                let sell_total = ((position_value) * -net_action)
                    .floor()
                    .min(position_value);

                if sell_total > 0.0 {
                    let quantity = sell_total / current_price;
                    // reward += self.position.appreciation(current_price) * quantity;
                    // println!("reward {} appreciation {}", reward, self.position.appreciation(current_price) * 100.0);
                    self.cash += sell_total;
                    
                    self.position.quantity -= quantity;

                    history.sells[ticker_index]
                        .insert(self.current_step, (current_price, quantity));
                }
            }
        // }

        self.current_step += 1;

        let new_price = self.prices[self.current_step];
        let position_value = self.position.value_with_price(new_price);
        let portfolio_value_after = self.cash + position_value;
        // Reward based on asset appreciation
        // let reward = ((portfolio_value_after - portfolio_value_before) / self.initial_cash).clamp(-1.0, 1.0);
        // reward based on portfolio value
        let reward = (portfolio_value_after - self.initial_cash) / self.initial_cash;
        
        // asset appreciation but based on position (avoids positive when position is negative)
        // let reward = if self.position.quantity > 0.0 {
        //     self.position.appreciation(new_price) * self.position.quantity
        // } else {
        //     0.0
        // };

        let next_obs = self.get_observation();

        history.positioned[ticker_index].push(position_value);
        history.cash[ticker_index].push(self.cash);
        history.rewards[ticker_index].push(reward);

        (next_obs, reward)
    }
}

pub struct GenHistory {
    pub buys: Vec<HashMap<usize, (f64, f64)>>,
    pub sells: Vec<HashMap<usize, (f64, f64)>>,
    pub positioned: Vec<Vec<f64>>,
    pub cash: Vec<Vec<f64>>,
    pub rewards: Vec<Vec<f64>>,
}

impl GenHistory {
    pub fn new(ticker_count: usize) -> Self {
        GenHistory {
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
                (
                    cash_index,
                    cash.last().unwrap() + positioned.last().unwrap(),
                )
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

#[derive(Default)]
pub struct MetaHistory {
    pub min_assets: Vec<f64>,
    pub avg_assets: Vec<f64>,
}

impl MetaHistory {
    pub fn record(&mut self, history: &GenHistory) {
        self.min_assets.push(history.ticker_lowest_final_assets().1);
        self.avg_assets.push(history.avg_final_assets());
    }
    
    pub fn chart(&self, generation: u32) {
        let base_dir = format!("{TRAINING_PATH}/gens/{}", generation);
        create_folder_if_not_exists(&base_dir);
        let _ = simple_chart(&base_dir, "min_assets", &self.min_assets);
        let _ = simple_chart(&base_dir, "avg_assets", &self.avg_assets);
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
