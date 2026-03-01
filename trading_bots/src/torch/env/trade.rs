use crate::torch::constants::{ACTION_THRESHOLD, COMMISSION_RATE};

use super::env::{Env, TRADE_EMA_ALPHA};

impl Env {
    #[allow(dead_code)]
    pub fn trade_by_delta_percent(&mut self, actions: &[f64], absolute_step: usize) -> f64 {
        let mut total_commission = 0.0;

        // === PASS 1: Execute all SELLS first (frees up cash) ===
        for (ticker_index, &action) in actions.iter().enumerate() {
            if action >= -ACTION_THRESHOLD {
                continue;
            }

            let price = self.prices[ticker_index][absolute_step];
            let current_value = self.account.positions[ticker_index].value_with_price(price);
            if current_value <= 0.0 {
                continue;
            }

            let sell_pct = -action;
            let sell_amount = sell_pct * current_value;

            if sell_amount <= 0.0 {
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
            self.steps_since_trade[ticker_index] = 0;
            if self.account.positions[ticker_index].quantity < 1e-8 {
                self.position_open_step[ticker_index] = None;
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
            if action <= ACTION_THRESHOLD {
                continue;
            }

            let price = self.prices[ticker_index][absolute_step];
            let desired_amount = action * self.account.total_assets;

            if desired_amount <= 0.0 {
                continue;
            }

            total_buy_demand += desired_amount * (1.0 + COMMISSION_RATE);
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
            let was_empty = self.account.positions[intent.ticker_index].quantity < 1e-8;
            self.account.positions[intent.ticker_index].add(intent.price, quantity);
            self.episode_history.total_commissions += commission;
            self.episode_history.buys[intent.ticker_index]
                .insert(absolute_step, (intent.price, quantity));
            self.trade_activity_ema[intent.ticker_index] += TRADE_EMA_ALPHA;
            self.steps_since_trade[intent.ticker_index] = 0;
            if was_empty {
                self.position_open_step[intent.ticker_index] = Some(absolute_step);
            }
        }

        total_commission
    }

    #[allow(dead_code)]
    const WEIGHT_DELTA_MIN_TRADE_FRAC: f64 = 0.005;

    #[allow(dead_code)]
    pub fn trade_by_weight_delta(&mut self, actions: &[f64], absolute_step: usize) -> f64 {
        let n_tickers = self.tickers.len();
        let min_trade_frac = Self::WEIGHT_DELTA_MIN_TRADE_FRAC;

        let mut total_commission = 0.0;

        // Apply delta with dead zone to ticker weights only.
        // Cash is computed as residual (1 - sum(ticker_weights)).
        for (i, &z) in actions.iter().take(n_tickers).enumerate() {
            if z.is_finite() && z.abs() > ACTION_THRESHOLD {
                self.target_weights[i] = (self.target_weights[i] + z).clamp(0.0, 1.0);
            }
        }

        // Enforce invariants and normalize tickers if needed.
        for w in self.target_weights.iter_mut().take(n_tickers) {
            if !w.is_finite() {
                *w = 0.0;
            } else {
                *w = w.clamp(0.0, 1.0);
            }
        }

        let mut ticker_sum: f64 = self.target_weights.iter().take(n_tickers).sum();
        if ticker_sum > 1.0 {
            for w in self.target_weights.iter_mut().take(n_tickers) {
                *w /= ticker_sum;
            }
            ticker_sum = 1.0;
        }

        // Residual cash weight (always in [0,1]).
        if self.target_weights.len() > n_tickers {
            self.target_weights[n_tickers] = (1.0 - ticker_sum).clamp(0.0, 1.0);
        }

        // 3. Calculate current portfolio weights and target deltas
        let total_assets = self.account.total_assets;
        if total_assets <= 0.0 {
            return 0.0;
        }

        // Collect sell and buy intents
        struct TradeIntent {
            ticker_index: usize,
            price: f64,
            delta_value: f64, // positive = buy, negative = sell
        }
        let mut sell_intents: Vec<TradeIntent> = Vec::new();
        let mut buy_intents: Vec<TradeIntent> = Vec::new();

        for ticker_index in 0..n_tickers {
            let price = self.prices[ticker_index][absolute_step];
            let current_value = self.account.positions[ticker_index].value_with_price(price);
            let target_value = self.target_weights[ticker_index] * total_assets;
            let delta_value = target_value - current_value;

            let min_trade_notional = min_trade_frac * target_value.max(current_value);
            if delta_value.abs() < min_trade_notional {
                continue;
            }

            if delta_value < 0.0 {
                sell_intents.push(TradeIntent {
                    ticker_index,
                    price,
                    delta_value,
                });
            } else {
                buy_intents.push(TradeIntent {
                    ticker_index,
                    price,
                    delta_value,
                });
            }
        }

        // 4. Execute sells first (frees up cash)
        for intent in sell_intents {
            let sell_value = (-intent.delta_value)
                .min(self.account.positions[intent.ticker_index].value_with_price(intent.price));
            if sell_value <= 0.0 {
                continue;
            }

            let quantity = sell_value / intent.price;
            let commission = quantity * COMMISSION_RATE;

            total_commission += commission;
            self.account.cash += sell_value - commission;
            self.account.positions[intent.ticker_index].quantity -= quantity;
            self.episode_history.total_commissions += commission;
            self.episode_history.sells[intent.ticker_index]
                .insert(absolute_step, (intent.price, quantity));
            self.trade_activity_ema[intent.ticker_index] += TRADE_EMA_ALPHA;
            self.steps_since_trade[intent.ticker_index] = 0;

            if self.account.positions[intent.ticker_index].quantity < 1e-8 {
                self.position_open_step[intent.ticker_index] = None;
            }
        }

        // 5. Execute buys with proportional scaling if insufficient cash
        let total_buy_demand: f64 = buy_intents
            .iter()
            .map(|i| i.delta_value * (1.0 + COMMISSION_RATE))
            .sum();
        let available_cash = self.account.cash;
        let fill_ratio = if total_buy_demand > 0.0 {
            (available_cash / total_buy_demand).min(1.0)
        } else {
            1.0
        };
        self.last_fill_ratio = fill_ratio;

        for intent in buy_intents {
            let scaled_amount = intent.delta_value * fill_ratio;
            let quantity = scaled_amount / intent.price;
            let commission = quantity * COMMISSION_RATE;
            let total_cost = scaled_amount + commission;

            if total_cost > self.account.cash || total_cost <= 0.0 {
                continue;
            }

            total_commission += commission;
            self.account.cash -= total_cost;
            let was_empty = self.account.positions[intent.ticker_index].quantity < 1e-8;
            self.account.positions[intent.ticker_index].add(intent.price, quantity);
            self.episode_history.total_commissions += commission;
            self.episode_history.buys[intent.ticker_index]
                .insert(absolute_step, (intent.price, quantity));
            self.trade_activity_ema[intent.ticker_index] += TRADE_EMA_ALPHA;
            self.steps_since_trade[intent.ticker_index] = 0;

            if was_empty {
                self.position_open_step[intent.ticker_index] = Some(absolute_step);
            }
        }

        total_commission
    }

    pub fn trade_by_target_weights(&mut self, actions: &[f64], absolute_step: usize) -> f64 {
        let n_tickers = self.tickers.len();

        let mut total_commission = 0.0;

        for ticker_index in 0..n_tickers {
            let weight = actions
                .get(ticker_index)
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            self.target_weights[ticker_index] = weight;
        }
        let cash_weight = actions
            .get(n_tickers)
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        self.target_weights[n_tickers] = cash_weight;

        // 2. Calculate target deltas and execute trades
        let total_assets = self.account.total_assets;
        if total_assets <= 0.0 {
            return 0.0;
        }

        struct TradeIntent {
            ticker_index: usize,
            price: f64,
            delta_value: f64,
        }
        let mut sell_intents: Vec<TradeIntent> = Vec::new();
        let mut buy_intents: Vec<TradeIntent> = Vec::new();

        for ticker_index in 0..n_tickers {
            let price = self.prices[ticker_index][absolute_step];
            let current_value = self.account.positions[ticker_index].value_with_price(price);
            let target_value = self.target_weights[ticker_index] * total_assets;
            let delta_value = target_value - current_value;

            let min_trade_notional = ACTION_THRESHOLD * target_value.max(current_value);
            if delta_value.abs() < min_trade_notional {
                continue;
            }

            if delta_value < 0.0 {
                sell_intents.push(TradeIntent {
                    ticker_index,
                    price,
                    delta_value,
                });
            } else {
                buy_intents.push(TradeIntent {
                    ticker_index,
                    price,
                    delta_value,
                });
            }
        }

        // 3. Execute sells first
        for intent in sell_intents {
            let sell_value = (-intent.delta_value)
                .min(self.account.positions[intent.ticker_index].value_with_price(intent.price));
            if sell_value <= 0.0 {
                continue;
            }

            let quantity = sell_value / intent.price;
            let commission = quantity * COMMISSION_RATE;

            total_commission += commission;
            self.account.cash += sell_value - commission;
            self.account.positions[intent.ticker_index].quantity -= quantity;
            self.episode_history.total_commissions += commission;
            self.episode_history.sells[intent.ticker_index]
                .insert(absolute_step, (intent.price, quantity));
            self.trade_activity_ema[intent.ticker_index] += TRADE_EMA_ALPHA;
            self.steps_since_trade[intent.ticker_index] = 0;

            if self.account.positions[intent.ticker_index].quantity < 1e-8 {
                self.position_open_step[intent.ticker_index] = None;
            }
        }

        // 4. Execute buys with proportional scaling if insufficient cash
        let total_buy_demand: f64 = buy_intents
            .iter()
            .map(|i| i.delta_value * (1.0 + COMMISSION_RATE))
            .sum();
        let available_cash = self.account.cash;
        let fill_ratio = if total_buy_demand > 0.0 {
            (available_cash / total_buy_demand).min(1.0)
        } else {
            1.0
        };
        self.last_fill_ratio = fill_ratio;

        for intent in buy_intents {
            let scaled_amount = intent.delta_value * fill_ratio;
            let quantity = scaled_amount / intent.price;
            let commission = quantity * COMMISSION_RATE;
            let total_cost = scaled_amount + commission;

            if total_cost > self.account.cash || total_cost <= 0.0 {
                continue;
            }

            total_commission += commission;
            self.account.cash -= total_cost;
            let was_empty = self.account.positions[intent.ticker_index].quantity < 1e-8;
            self.account.positions[intent.ticker_index].add(intent.price, quantity);
            self.episode_history.total_commissions += commission;
            self.episode_history.buys[intent.ticker_index]
                .insert(absolute_step, (intent.price, quantity));
            self.trade_activity_ema[intent.ticker_index] += TRADE_EMA_ALPHA;
            self.steps_since_trade[intent.ticker_index] = 0;

            if was_empty {
                self.position_open_step[intent.ticker_index] = Some(absolute_step);
            }
        }

        total_commission
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
    const DELTA_REBALANCE_RATE: f64 = 1.0;
    #[allow(dead_code)]
    const DELTA_MIN_TRADE_FRAC: f64 = 0.005;

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

        let rebalance_rate = Self::DELTA_REBALANCE_RATE;
        let min_trade_notional = Self::DELTA_MIN_TRADE_FRAC * total_assets;

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
