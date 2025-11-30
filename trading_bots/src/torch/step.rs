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
            ACTION_HISTORY_LEN, ACTION_THRESHOLD, COMMISSION_RATE, GLOBAL_STATIC_OBS,
            PER_TICKER_STATIC_OBS, PRICE_DELTAS_PER_TICKER, RETROACTIVE_BUY_REWARD,
            STATIC_OBSERVATIONS, STEPS_PER_EPISODE, TICKERS_COUNT,
        },
        ppo::NPROCS,
    },
    types::Account,
    utils::{create_folder_if_not_exists, get_mapped_price_deltas},
};

// Reward scaling factors
const REWARD_SCALE: f64 = 20.0;
const SHARPE_LAMBDA: f64 = 100.0;
const SORTINO_LAMBDA: f64 = 100.0;
const RISK_ADJUSTED_REWARD_LAMBDA: f64 = 0.01;
const COMMISSIONS_PENALTY_LAMBDA: f64 = 0.01;

#[derive(Debug, Clone)]
struct BuyLot {
    step: usize,
    price: f64,
    quantity: f64,
}

pub struct Env {
    pub step: usize,
    pub max_step: usize,
    pub tickers: Vec<String>,
    pub prices: Vec<Vec<f64>>,
    price_deltas: Vec<Vec<f64>>,
    account: Account,
    pub episode_history: EpisodeHistory,
    pub meta_history: MetaHistory,
    episode_start: Instant,
    pub episode: usize,
    action_history: VecDeque<Vec<f64>>,
    episode_start_offset: usize,
    total_data_length: usize,
    random_start: bool,
    buy_lots: Vec<VecDeque<BuyLot>>,
    retroactive_rewards: HashMap<usize, f64>,
    peak_assets: f64,
    last_reward: f64,
}

impl Env {
    const STARTING_CASH: f64 = 10_000.0;

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

            let (total_commission, trade_sell_reward) = self.trade_by_delta_percent_with_hold(buy_sell_actions, hold_actions, absolute_step);
            let reward = self.get_index_benchmark_pnl_reward(absolute_step, total_commission)
                + if RETROACTIVE_BUY_REWARD { trade_sell_reward } else { 0.0 };

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
                let mut index_return = 0.0;
                for ticker_idx in 0..self.tickers.len() {
                    let start_price = self.prices[ticker_idx][self.episode_start_offset];
                    let end_price = self.prices[ticker_idx][absolute_step];
                    index_return += (end_price / start_price - 1.0) * 100.0;
                }
                index_return /= self.tickers.len() as f64;

                let strategy_return =
                    (self.account.total_assets / Self::STARTING_CASH - 1.0) * 100.0;
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

    fn get_unrealized_pnl_reward(&self, absolute_step: usize, commissions: f64) -> f64 {
        if self.step + 1 < self.max_step && self.account.total_assets > 0.0 {
            let next_absolute_step = absolute_step + 1;
            let total_assets_after_trade = self
                .account
                .position_values(&self.prices, next_absolute_step)
                .iter()
                .sum::<f64>()
                + self.account.cash;

            let pnl_reward = (total_assets_after_trade / self.account.total_assets).ln();

            pnl_reward * REWARD_SCALE
        } else {
            0.0
        }
    }

    fn get_index_benchmark_pnl_reward(&self, absolute_step: usize, commissions: f64) -> f64 {
        if self.step + 1 < self.max_step && self.account.total_assets > 0.0 {
            let next_absolute_step = absolute_step + 1;

            let total_assets_after_trade = self
                .account
                .position_values(&self.prices, next_absolute_step)
                .iter()
                .sum::<f64>()
                + self.account.cash;
            let strategy_log_return = (total_assets_after_trade / self.account.total_assets).ln();

            let mut index_log_return = 0.0;
            for ticker_idx in 0..self.tickers.len() {
                let current_price = self.prices[ticker_idx][absolute_step];
                let next_price = self.prices[ticker_idx][next_absolute_step];
                index_log_return += (next_price / current_price).ln();
            }
            index_log_return /= self.tickers.len() as f64;

            let excess_return = strategy_log_return - index_log_return;

            let commissions_relative = commissions / self.account.total_assets;
            let commissions_penalty = -commissions_relative * COMMISSIONS_PENALTY_LAMBDA;

            (excess_return + commissions_penalty) * REWARD_SCALE
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

    fn get_risk_adjusted_reward(&self, absolute_step: usize, commissions: f64) -> f64 {
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
                -step_return
            } else {
                0.0
            };
            let rar = step_return - RISK_ADJUSTED_REWARD_LAMBDA * downside;
            
            let commissions_relative = commissions / self.account.total_assets;
            let commisions_penalty = -commissions_relative * RISK_ADJUSTED_REWARD_LAMBDA;
            let reward = commisions_penalty + rar;
            
            reward * REWARD_SCALE
        } else {
            0.0
        }
    }

    fn trade_buy_sell_to(&mut self, actions: &[f64], absolute_step: usize) -> f64 {
        let mut total_commissions = 0.0;

        for (ticker_index, action) in actions.iter().enumerate() {

            let price = self.prices[ticker_index][absolute_step];

            let max_ownership = self.account.total_assets / self.tickers.len() as f64;
            let target = max_ownership * ((action + 1.0) / 2.0); // Convert [-1, 1] to [0, 1]
            let current_value = self.account.positions[ticker_index].value_with_price(price);

            let desired_delta = target - current_value;
            
            let threshold_normal = ACTION_THRESHOLD * self.account.total_assets;
            if desired_delta.abs() < threshold_normal {
                continue;
            }

            // BUY
            if desired_delta > 0.0 {
                let quantity = desired_delta / price;
                let commission = quantity * COMMISSION_RATE;
                let total_cost = desired_delta + commission;

                if total_cost > self.account.cash {
                    continue;
                }
                
                total_commissions += commission;

                self.account.cash -= total_cost;
                self.account.positions[ticker_index].add(price, quantity);
                self.episode_history.total_commissions += commission;

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
                let commission = quantity * COMMISSION_RATE;
                
                total_commissions += commission;

                self.account.cash += trade_value - commission;
                self.account.positions[ticker_index].quantity -= quantity;
                self.episode_history.total_commissions += commission;

                self.episode_history.sells[ticker_index].insert(absolute_step, (price, quantity));
            }
        }
        
        total_commissions
    }
    
    /// Trade using percentage deltas with hold action scaling and retroactive reward tracking.
    /// - buy_sell near 0 = hold
    /// - buy_sell = 0.1 = buy 10% of total_assets worth of stock
    /// - buy_sell = -0.1 = sell 10% of current position value
    /// - hold > 0 amplifies trades (up to 2x), hold < 0 suppresses (down to 0.1x)
    /// Returns (total_commission, sell_reward)
    fn trade_by_delta_percent_with_hold(&mut self, buy_sell_actions: &[f64], hold_actions: &[f64], absolute_step: usize) -> (f64, f64) {
        let mut total_commission = 0.0;
        let mut sell_reward = 0.0;

        for (ticker_index, &buy_sell) in buy_sell_actions.iter().enumerate() {
            if buy_sell.abs() < ACTION_THRESHOLD {
                continue;
            }

            let hold = hold_actions[ticker_index];
            let action = buy_sell * (1.0 + hold * 0.5).clamp(0.5, 1.5);

            let price = self.prices[ticker_index][absolute_step];
            let current_value = self.account.positions[ticker_index].value_with_price(price);

            if action > 0.0 {
                let buy_amount = action * self.account.total_assets;

                if buy_amount <= 0.0 {
                    continue;
                }

                let quantity = buy_amount / price;
                let commission = quantity * COMMISSION_RATE;
                let total_cost = buy_amount + commission;

                if total_cost > self.account.cash {
                    continue;
                }

                total_commission += commission;

                self.account.cash -= total_cost;
                self.account.positions[ticker_index].add(price, quantity);
                self.episode_history.total_commissions += commission;
                self.episode_history.buys[ticker_index].insert(absolute_step, (price, quantity));

                if RETROACTIVE_BUY_REWARD {
                    self.buy_lots[ticker_index].push_back(BuyLot {
                        step: absolute_step,
                        price,
                        quantity,
                    });
                }
            } else {
                let sell_pct = -action;
                let sell_amount = sell_pct * current_value;

                if sell_amount <= 0.0 || current_value <= 0.0 {
                    continue;
                }

                let quantity = sell_amount / price;
                let commission = quantity * COMMISSION_RATE;

                total_commission += commission;

                self.account.cash += sell_amount - commission;
                self.account.positions[ticker_index].quantity -= quantity;
                self.episode_history.total_commissions += commission;
                self.episode_history.sells[ticker_index].insert(absolute_step, (price, quantity));

                if RETROACTIVE_BUY_REWARD {
                    sell_reward += self.calculate_retroactive_rewards(ticker_index, absolute_step, price, quantity);
                }
            }
        }

        (total_commission, sell_reward)
    }

    /// Calculate retroactive rewards using FIFO lot matching.
    /// Rewards sells based on time-weighted return and retroactively rewards
    /// the buys that contributed to profitable sells.
    /// Returns the immediate sell reward (50%), stores buy rewards (50%) retroactively.
    fn calculate_retroactive_rewards(&mut self, ticker_index: usize, sell_step: usize, sell_price: f64, mut sell_quantity: f64) -> f64 {
        let lots = &mut self.buy_lots[ticker_index];
        let mut total_sell_reward = 0.0;

        while sell_quantity > 1e-8 && !lots.is_empty() {
            let lot = lots.front_mut().unwrap();
            let take_qty = sell_quantity.min(lot.quantity);

            let return_pct = (sell_price - lot.price) / lot.price;
            let hold_time = (sell_step - lot.step).max(1) as f64;
            let time_weighted_return = return_pct / hold_time.sqrt();

            let sell_reward = time_weighted_return * take_qty * lot.price;
            let immediate_sell_reward = sell_reward * 0.5;
            total_sell_reward += immediate_sell_reward;

            let buy_contribution = sell_reward * 0.5;
            *self.retroactive_rewards.entry(lot.step).or_insert(0.0) += buy_contribution;

            sell_quantity -= take_qty;
            lot.quantity -= take_qty;

            if lot.quantity < 1e-8 {
                lots.pop_front();
            }
        }

        total_sell_reward
    }

    pub fn apply_retroactive_rewards(&self, rewards_tensor: &Tensor) {
        for (&step, &reward) in &self.retroactive_rewards {
            let relative_step = step.saturating_sub(self.episode_start_offset);
            if (relative_step as i64) < rewards_tensor.size()[0] {
                let step_tensor = rewards_tensor.get(relative_step as i64);
                let current = f64::try_from(step_tensor.get(0)).unwrap_or(0.0);
                step_tensor.get(0).copy_(&Tensor::from(current + reward));
            }
        }
    }

    fn calculate_position_time_weighted_return(&self, ticker_index: usize, current_step: usize) -> f64 {
        if self.buy_lots[ticker_index].is_empty() {
            return 0.0;
        }

        let current_price = self.prices[ticker_index][current_step];
        let mut total_weighted_return = 0.0;
        let mut total_quantity = 0.0;

        for lot in &self.buy_lots[ticker_index] {
            let return_pct = (current_price - lot.price) / lot.price;
            let hold_time = (current_step - lot.step).max(1) as f64;
            let time_weighted_return = return_pct / hold_time.sqrt();

            total_weighted_return += time_weighted_return * lot.quantity;
            total_quantity += lot.quantity;
        }

        if total_quantity > 0.0 {
            total_weighted_return / total_quantity
        } else {
            0.0
        }
    }

    fn trade_by_delta(&mut self, actions: &[f64], absolute_step: usize) -> f64 {
        let n_tickers = self.tickers.len();
        let mut total_commissions = 0.0;
    
        let total_assets = self.account.total_assets;
        if total_assets <= 0.0 {
            return 0.0;
        }
    
        // ---- 1. Build logits for N assets + 1 cash slot ----
        // actions are unconstrained-ish; we just treat them as logits
        let mut logits = Vec::with_capacity(n_tickers + 1);
        logits.extend_from_slice(actions);              // N tickers
        logits.push(0.0);                               // cash logit baseline
    
        // Softmax for weights in [0,1] that sum to 1
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exps: f64 = exps.iter().sum();
        let weights: Vec<f64> = exps.iter().map(|e| e / sum_exps).collect();
    
        // weights[0..N] = asset weights, weights[N] = cash weight
    
        // ---- 2. Compute current position values ----
        let mut current_values = Vec::with_capacity(n_tickers);
        for ticker_index in 0..n_tickers {
            let price = self.prices[ticker_index][absolute_step];
            current_values.push(self.account.positions[ticker_index].value_with_price(price));
        }
    
        // ---- 3. Compute target dollar values for each ticker ----
        let rebalance_rate: f64 = 1.0;
        let min_trade_frac: f64 = 0.005; // skip trades < 0.05% of total assets
        let min_trade_notional = min_trade_frac * total_assets;
    
        for ticker_index in 0..n_tickers {
            let price = self.prices[ticker_index][absolute_step];
    
            let target_value = weights[ticker_index] * total_assets;
            let current_value = current_values[ticker_index];
    
            // desired full delta
            let full_delta = target_value - current_value;
    
            // partial move toward target
            let desired_delta = full_delta * rebalance_rate;
    
            // Hysteresis: ignore very small trades
            if desired_delta.abs() < min_trade_notional {
                continue;
            }
    
            if desired_delta > 0.0 {
                // BUY
                let trade_value = desired_delta.min(self.account.cash); // no leverage
                if trade_value <= 0.0 {
                    continue;
                }
    
                let quantity = trade_value / price;
                let commission = quantity * COMMISSION_RATE;
                let total_cost = trade_value + commission;
    
                if total_cost > self.account.cash {
                    continue;
                }
    
                total_commissions += commission;
    
                self.account.cash -= total_cost;
                self.account.positions[ticker_index].add(price, quantity);
                self.episode_history.total_commissions += commission;
                self.episode_history
                    .buys[ticker_index]
                    .insert(absolute_step, (price, quantity));
            } else {
                // SELL
                let desired_sell_value = -desired_delta;
                let position_value = current_value;
                let trade_value = desired_sell_value.min(position_value);
    
                if trade_value <= 0.0 {
                    continue;
                }
    
                let quantity = trade_value / price;
                let commission = quantity * COMMISSION_RATE;
    
                total_commissions += commission;
    
                self.account.cash += trade_value - commission;
                self.account.positions[ticker_index].quantity -= quantity;
                self.episode_history.total_commissions += commission;
                self.episode_history
                    .sells[ticker_index]
                    .insert(absolute_step, (price, quantity));
            }
        }
    
        total_commissions
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
        // Static obs format: [global_obs (6), per_ticker_obs for each ticker]
        // This ticker-major format enables ticker-agnostic processing in the model

        let mut price_deltas = Vec::with_capacity(TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER);
        let absolute_step = self.episode_start_offset + self.step;

        for ticker_price_deltas in self.price_deltas.iter() {
            let start_idx = absolute_step.saturating_sub(PRICE_DELTAS_PER_TICKER - 1);
            let end_idx = (absolute_step + 1).min(ticker_price_deltas.len());
            let slice = &ticker_price_deltas[start_idx..end_idx];
            let to_take = slice.len().min(PRICE_DELTAS_PER_TICKER);

            let padding_needed = PRICE_DELTAS_PER_TICKER - to_take;
            if padding_needed > 0 {
                price_deltas.extend(std::iter::repeat(0.0f32).take(padding_needed));
            }
            price_deltas.extend(slice.iter().rev().take(to_take).map(|&x| x as f32));
        }

        let mut static_obs = Vec::with_capacity(STATIC_OBSERVATIONS);

        // === Global observations (GLOBAL_STATIC_OBS = 6) ===
        static_obs.push(1.0 - (self.step as f32 / (self.max_step - 1).max(1) as f32)); // step progress
        static_obs.push((self.account.cash / self.account.total_assets) as f32); // cash_percent
        static_obs.push(((self.account.total_assets / Self::STARTING_CASH) - 1.0) as f32); // pnl
        static_obs.push(if self.peak_assets > 0.0 {
            ((self.account.total_assets / self.peak_assets) - 1.0) as f32
        } else {
            0.0
        }); // drawdown
        static_obs.push((self.episode_history.total_commissions / Self::STARTING_CASH) as f32); // commissions
        static_obs.push(self.last_reward as f32); // last_reward
        debug_assert_eq!(static_obs.len(), GLOBAL_STATIC_OBS);

        // === Per-ticker observations (ticker-major format) ===
        let position_percents = self.account.position_percents(&self.prices, absolute_step);

        for ticker_index in 0..TICKERS_COUNT as usize {
            let current_price = self.prices[ticker_index][absolute_step];

            // Position percent
            static_obs.push(position_percents[ticker_index] as f32);

            // Unrealized P&L %
            static_obs.push(
                self.account.positions[ticker_index]
                    .appreciation(current_price) as f32,
            );

            // Momentum (20-step lookback)
            let past_step = absolute_step.saturating_sub(20);
            let past_price = self.prices[ticker_index][past_step];
            static_obs.push(((current_price / past_price) - 1.0) as f32);

            // Action history for this ticker (most recent first)
            for i in 0..ACTION_HISTORY_LEN {
                if i < self.action_history.len() {
                    let action_idx = self.action_history.len() - 1 - i;
                    // buy_sell action for this ticker
                    static_obs.push(self.action_history[action_idx][ticker_index] as f32);
                    // hold action for this ticker
                    static_obs.push(
                        self.action_history[action_idx][TICKERS_COUNT as usize + ticker_index]
                            as f32,
                    );
                } else {
                    static_obs.push(0.0f32); // buy_sell padding
                    static_obs.push(0.0f32); // hold padding
                }
            }
            debug_assert_eq!(
                static_obs.len(),
                GLOBAL_STATIC_OBS + (ticker_index + 1) * PER_TICKER_STATIC_OBS
            );
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

        for lot_queue in &mut self.buy_lots {
            lot_queue.clear();
        }
        self.retroactive_rewards.clear();

        self.peak_assets = Self::STARTING_CASH;
        self.last_reward = 0.0;

        // Return initial observation as two separate tensors
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
