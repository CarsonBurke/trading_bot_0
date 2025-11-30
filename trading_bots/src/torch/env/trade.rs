use tch::Tensor;

use crate::torch::constants::{ACTION_THRESHOLD, COMMISSION_RATE, RETROACTIVE_BUY_REWARD};

use super::env::{BuyLot, Env, TRADE_EMA_ALPHA};

impl Env {
    pub(super) fn trade_by_delta_percent_with_hold(
        &mut self,
        buy_sell_actions: &[f64],
        hold_actions: &[f64],
        absolute_step: usize,
    ) -> (f64, f64) {
        let mut total_commission = 0.0;
        let mut sell_reward = 0.0;

        let actions: Vec<f64> = buy_sell_actions
            .iter()
            .zip(hold_actions.iter())
            .map(|(&buy_sell, &hold)| {
                if buy_sell.abs() < ACTION_THRESHOLD {
                    0.0
                } else {
                    buy_sell * (1.0 + hold * 0.5).clamp(0.5, 1.5)
                }
            })
            .collect();

        // === PASS 1: Execute all SELLS first (frees up cash) ===
        for (ticker_index, &action) in actions.iter().enumerate() {
            if action >= 0.0 {
                continue;
            }

            let price = self.prices[ticker_index][absolute_step];
            let current_value = self.account.positions[ticker_index].value_with_price(price);
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
            self.trade_activity_ema[ticker_index] += TRADE_EMA_ALPHA;

            if RETROACTIVE_BUY_REWARD {
                sell_reward +=
                    self.calculate_retroactive_rewards(ticker_index, absolute_step, price, quantity);
            }
        }

        // === PASS 2: Collect buy intents and compute proportional fill ===
        struct BuyIntent {
            ticker_index: usize,
            price: f64,
            desired_amount: f64,
        }

        let mut buy_intents: Vec<BuyIntent> = Vec::new();
        let mut total_buy_demand = 0.0;

        for (ticker_index, &action) in actions.iter().enumerate() {
            if action <= 0.0 {
                continue;
            }

            let price = self.prices[ticker_index][absolute_step];
            let desired_amount = action * self.account.total_assets;

            if desired_amount <= 0.0 {
                continue;
            }

            total_buy_demand += desired_amount;
            buy_intents.push(BuyIntent {
                ticker_index,
                price,
                desired_amount,
            });
        }

        let available_cash = self.account.cash;
        let fill_ratio = if total_buy_demand > 0.0 {
            (available_cash / total_buy_demand).min(1.0)
        } else {
            1.0
        };
        self.last_fill_ratio = fill_ratio;

        // === PASS 3: Execute buys with proportional scaling ===
        for intent in buy_intents {
            let scaled_amount = intent.desired_amount * fill_ratio;
            let quantity = scaled_amount / intent.price;
            let commission = quantity * COMMISSION_RATE;
            let total_cost = scaled_amount + commission;

            if total_cost > self.account.cash || total_cost <= 0.0 {
                continue;
            }

            total_commission += commission;
            self.account.cash -= total_cost;
            self.account.positions[intent.ticker_index].add(intent.price, quantity);
            self.episode_history.total_commissions += commission;
            self.episode_history.buys[intent.ticker_index]
                .insert(absolute_step, (intent.price, quantity));
            self.trade_activity_ema[intent.ticker_index] += TRADE_EMA_ALPHA;

            if RETROACTIVE_BUY_REWARD {
                self.buy_lots[intent.ticker_index].push_back(BuyLot {
                    step: absolute_step,
                    price: intent.price,
                    quantity,
                });
            }
        }

        (total_commission, sell_reward)
    }

    fn calculate_retroactive_rewards(
        &mut self,
        ticker_index: usize,
        sell_step: usize,
        sell_price: f64,
        mut sell_quantity: f64,
    ) -> f64 {
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    fn trade_buy_sell_to(&mut self, actions: &[f64], absolute_step: usize) -> f64 {
        let mut total_commissions = 0.0;

        for (ticker_index, action) in actions.iter().enumerate() {
            let price = self.prices[ticker_index][absolute_step];

            let max_ownership = self.account.total_assets / self.tickers.len() as f64;
            let target = max_ownership * ((action + 1.0) / 2.0);
            let current_value = self.account.positions[ticker_index].value_with_price(price);

            let desired_delta = target - current_value;

            let threshold_normal = ACTION_THRESHOLD * self.account.total_assets;
            if desired_delta.abs() < threshold_normal {
                continue;
            }

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
            } else if desired_delta < 0.0 {
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

    #[allow(dead_code)]
    fn trade_by_delta(&mut self, actions: &[f64], absolute_step: usize) -> f64 {
        let n_tickers = self.tickers.len();
        let mut total_commissions = 0.0;

        let total_assets = self.account.total_assets;
        if total_assets <= 0.0 {
            return 0.0;
        }

        let mut logits = Vec::with_capacity(n_tickers + 1);
        logits.extend_from_slice(actions);
        logits.push(0.0);

        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exps: f64 = exps.iter().sum();
        let weights: Vec<f64> = exps.iter().map(|e| e / sum_exps).collect();

        let mut current_values = Vec::with_capacity(n_tickers);
        for ticker_index in 0..n_tickers {
            let price = self.prices[ticker_index][absolute_step];
            current_values.push(self.account.positions[ticker_index].value_with_price(price));
        }

        let rebalance_rate: f64 = 1.0;
        let min_trade_frac: f64 = 0.005;
        let min_trade_notional = min_trade_frac * total_assets;

        for ticker_index in 0..n_tickers {
            let price = self.prices[ticker_index][absolute_step];

            let target_value = weights[ticker_index] * total_assets;
            let current_value = current_values[ticker_index];

            let full_delta = target_value - current_value;
            let desired_delta = full_delta * rebalance_rate;

            if desired_delta.abs() < min_trade_notional {
                continue;
            }

            if desired_delta > 0.0 {
                let trade_value = desired_delta.min(self.account.cash);
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
                self.episode_history.buys[ticker_index].insert(absolute_step, (price, quantity));
            } else {
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
                self.episode_history.sells[ticker_index].insert(absolute_step, (price, quantity));
            }
        }

        total_commissions
    }
}
