use crate::torch::constants::{ACTION_THRESHOLD, COMMISSION_RATE};

use super::single::{Env, TRADE_EMA_ALPHA};

impl Env {
    pub fn sync_realized_weights(&mut self, absolute_step: usize) {
        let n_tickers = self.tickers.len();
        let total_assets = self.account.total_assets;
        if total_assets <= 0.0 {
            self.realized_weights.fill(0.0);
            return;
        }

        let inv_total_assets = 1.0 / total_assets;
        for ticker_index in 0..n_tickers {
            let price = self.prices[ticker_index][absolute_step];
            let value = self.account.positions[ticker_index].value_with_price(price);
            self.realized_weights[ticker_index] = (value * inv_total_assets).clamp(0.0, 1.0);
        }
        self.realized_weights[n_tickers] = (self.account.cash * inv_total_assets).clamp(0.0, 1.0);
    }

    pub fn trade_by_target_weights(&mut self, actions: &[f64], absolute_step: usize) -> f64 {
        let n_tickers = self.tickers.len();
        assert_eq!(n_tickers, 1, "single-ticker action space expected");

        let mut total_commission = 0.0;

        let cash_idx = n_tickers;
        let ticker_weight = actions
            .first()
            .copied()
            .filter(|weight| weight.is_finite())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        self.target_weights[0] = ticker_weight;
        self.target_weights[cash_idx] = 1.0 - ticker_weight;

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
            .map(|i| i.delta_value + (i.delta_value / i.price) * COMMISSION_RATE)
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
}
