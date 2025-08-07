use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};
use rand::Rng;

const OBSERVATION_DIM: i64 = 10; // price, volume, moving averages, etc.
const ACTION_DIM: i64 = 3; // buy, sell, hold
const HIDDEN_DIM: i64 = 64;
const LEARNING_RATE: f64 = 1e-4;
const GAMMA: f32 = 0.99;
const EPSILON: f32 = 0.2;
const VALUE_COEF: f32 = 0.5;
const ENTROPY_COEF: f32 = 0.01;

struct ActorCritic {
    shared: nn::Sequential,
    actor_head: nn::Linear,
    critic_head: nn::Linear,
}

impl ActorCritic {
    fn new(vs: &nn::Path) -> Self {
        let shared = nn::seq()
            .add(nn::linear(vs / "fc1", OBSERVATION_DIM, HIDDEN_DIM, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "fc2", HIDDEN_DIM, HIDDEN_DIM, Default::default()))
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

struct TradingEnvironment {
    prices: Vec<f32>,
    current_step: usize,
    position: f32,
    cash: f32,
    initial_cash: f32,
}

impl TradingEnvironment {
    fn new() -> Self {
        let mut rng = rand::rng();
        let mut prices = vec![100.0];

        // Generate fake price data
        for _ in 1..1000 {
            // Tend to increase more than decrease
            let change = rng.random_range(-2.0..4.0);
            let new_price = (*prices.last().unwrap() as f32 + change).max(1.0_f32);
            prices.push(new_price);
        }

        TradingEnvironment {
            prices,
            current_step: 10,
            position: 0.0,
            cash: 10000.0,
            initial_cash: 10000.0,
        }
    }

    fn reset(&mut self) -> Tensor {
        self.current_step = 10;
        self.position = 0.0;
        self.cash = self.initial_cash;
        self.get_observation()
    }

    fn get_observation(&self) -> Tensor {
        let mut obs = vec![];

        // Current price - normalize
        obs.push(self.prices[self.current_step] / 100.0);

        // Price changes - normalize
        for i in 1..5 {
            if self.current_step >= i {
                obs.push((self.prices[self.current_step] - self.prices[self.current_step - i]) / 100.0);
            } else {
                obs.push(0.0);
            }
        }

        // Simple moving averages - normalize
        let ma5 = self.prices[self.current_step.saturating_sub(5)..=self.current_step]
            .iter()
            .sum::<f32>() / 5.0;
        let ma10 = self.prices[self.current_step.saturating_sub(10)..=self.current_step]
            .iter()
            .sum::<f32>() / 10.0;

        obs.push(ma5 / 100.0);
        obs.push(ma10 / 100.0);
        obs.push(self.position / 100.0); // normalize position
        obs.push(self.cash / self.initial_cash);

        // Ensure we have exactly OBSERVATION_DIM elements
        while obs.len() < OBSERVATION_DIM as usize {
            obs.push(0.0);
        }
        obs.truncate(OBSERVATION_DIM as usize);

        Tensor::from_slice(&obs).view([1, OBSERVATION_DIM])
    }

    fn learn_step(&mut self, action: i64) -> (Tensor, f32, bool) {
        let current_price = self.prices[self.current_step];
        let portfolio_value_before = self.cash + self.position * current_price;

        match action {
            0 => { // Buy
                let shares_to_buy = (self.cash * 0.1 / current_price).floor();
                self.position += shares_to_buy;
                self.cash -= shares_to_buy * current_price;
            }
            1 => { // Sell
                let shares_to_sell = (self.position * 0.1).floor();
                self.position -= shares_to_sell;
                self.cash += shares_to_sell * current_price;
            }
            _ => {} // Hold
        }

        self.current_step += 1;

        let new_price = self.prices[self.current_step];
        let portfolio_value_after = self.cash + self.position * new_price;
        let reward = ((portfolio_value_after - portfolio_value_before) / self.initial_cash).clamp(-1.0, 1.0);

        let done = self.current_step >= self.prices.len() - 1;
        let next_obs = self.get_observation();
        if self.current_step % 100 == 0 {
            println!("total assets {}", self.cash + self.position * new_price);
        }
        (next_obs, reward, done)
    }
}

struct PPOBuffer {
    observations: Vec<Tensor>,
    actions: Vec<i64>,
    rewards: Vec<f32>,
    values: Vec<f32>,
    log_probs: Vec<f32>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
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

    fn add(&mut self, obs: Tensor, action: i64, reward: f32, value: f32, log_prob: f32) {
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
        let mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let var: f32 = advantages.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / advantages.len() as f32;
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

fn main() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let model = ActorCritic::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE).unwrap();

    let mut env = TradingEnvironment::new();
    let mut buffer = PPOBuffer::new();

    for episode in 0..1000 {
        let mut obs = env.reset();
        let mut episode_reward = 0.0;

        // Collect trajectory
        loop {
            let (action_logits, value) = tch::no_grad(|| {
                model.forward(&obs)
            });

            let probs = action_logits.softmax(-1, tch::Kind::Float);
            let action = probs.multinomial(1, true).int64_value(&[0]);
            let log_prob = probs.log().gather(1, &Tensor::from_slice(&[action]).view([1, 1]), false);

            buffer.add(
                obs.shallow_clone(),
                action,
                0.0, // Will be updated with actual reward
                value.double_value(&[]) as f32,
                log_prob.double_value(&[]) as f32,
            );

            let (next_obs, reward, done) = env.learn_step(action);
            buffer.rewards.last_mut().unwrap().clone_from(&reward);
            episode_reward += reward;

            if done {
                break;
            }

            obs = next_obs;
        }

        buffer.compute_advantages();

        // PPO update
        for _ in 0..3 {
            let obs_batch = Tensor::cat(&buffer.observations, 0);
            let actions_batch = Tensor::from_slice(&buffer.actions).view([-1, 1]);
            let old_log_probs = Tensor::from_slice(&buffer.log_probs);
            let advantages = Tensor::from_slice(&buffer.advantages);
            let returns = Tensor::from_slice(&buffer.returns);

            let (action_logits, values) = model.forward(&obs_batch);
            let values = values.squeeze();

            // Check for NaN values and skip update if found
            if values.isnan().any().int64_value(&[]) != 0 {
                println!("Warning: NaN detected in values, skipping update");
                break;
            }

            let probs = action_logits.softmax(-1, tch::Kind::Float);
            let log_probs = probs.log().gather(1, &actions_batch, false).squeeze();

            let ratio = (log_probs - old_log_probs).exp();
            let clipped_ratio = ratio.clamp((1.0 - EPSILON) as f64, (1.0 + EPSILON) as f64);

            let policy_loss = -Tensor::min_other(&(ratio * &advantages), &(clipped_ratio * &advantages)).mean(tch::Kind::Float);

            let value_loss = (values - returns).pow_tensor_scalar(2).mean(tch::Kind::Float);

            let entropy = -(probs.copy() * probs.log()).sum_dim_intlist(&[1i64][..], false, tch::Kind::Float).mean(tch::Kind::Float);

            let total_loss = &policy_loss + VALUE_COEF * &value_loss - ENTROPY_COEF * &entropy;

            // Check for NaN in total loss
            if total_loss.isnan().any().int64_value(&[]) != 0 {
                println!("Warning: NaN detected in loss, skipping update");
                break;
            }

            opt.zero_grad();
            total_loss.backward();

            opt.step();
        }

        buffer.clear();

        if episode % 10 == 0 {
            println!("Episode {}: Reward = {:.4}", episode, episode_reward);
        }
    }

    println!("Training completed!");
}