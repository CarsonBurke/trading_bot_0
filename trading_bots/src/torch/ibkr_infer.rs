use tch::{Device, Tensor};
use std::path::Path;
use std::time::Instant;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;

use ibapi::{
    accounts::{types::AccountGroup, AccountSummaryResult, AccountSummaryTags, PositionUpdate},
    contracts::Contract,
    market_data::{realtime::{BarSize, WhatToShow}, TradingHours},
    Client,
};

use crate::constants::api;
use crate::torch::constants::{ACTION_HISTORY_LEN, ACTION_THRESHOLD, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT};
use crate::torch::infer::{load_model, sample_actions};
use crate::types::{Account, Position};

const MAX_ACCOUNT_VALUE: Option<f64> = Some(10_000.0);

struct LiveMarketState {
    prices: Vec<VecDeque<f64>>,
    price_deltas: Vec<VecDeque<f64>>,
    account: Account,
    action_history: VecDeque<Vec<f64>>,
    step_count: usize,
}

impl LiveMarketState {
    fn new(ticker_count: usize, starting_cash: f64) -> Self {
        Self {
            prices: vec![VecDeque::with_capacity(PRICE_DELTAS_PER_TICKER + 1); ticker_count],
            price_deltas: vec![VecDeque::with_capacity(PRICE_DELTAS_PER_TICKER); ticker_count],
            account: Account::new(starting_cash, ticker_count),
            action_history: VecDeque::with_capacity(ACTION_HISTORY_LEN),
            step_count: 0,
        }
    }

    fn update_price(&mut self, ticker_idx: usize, price: f64) {
        self.prices[ticker_idx].push_back(price);
        if self.prices[ticker_idx].len() > PRICE_DELTAS_PER_TICKER + 1 {
            self.prices[ticker_idx].pop_front();
        }

        if self.prices[ticker_idx].len() >= 2 {
            let len = self.prices[ticker_idx].len();
            let prev_price = self.prices[ticker_idx][len - 2];
            let delta = (price - prev_price) / prev_price;

            self.price_deltas[ticker_idx].push_back(delta);
            if self.price_deltas[ticker_idx].len() > PRICE_DELTAS_PER_TICKER {
                self.price_deltas[ticker_idx].pop_front();
            }
        }
    }

    fn get_current_prices(&self) -> Vec<f64> {
        self.prices
            .iter()
            .map(|q| *q.back().unwrap_or(&0.0))
            .collect()
    }

    fn update_account_total(&mut self) {
        let current_prices = self.get_current_prices();
        let position_values: f64 = self.account.positions
            .iter()
            .enumerate()
            .map(|(i, p)| p.value_with_price(current_prices[i]))
            .sum();
        self.account.total_assets = position_values + self.account.cash;
    }

    fn build_observation(&self) -> Option<(Tensor, Tensor)> {
        if self.price_deltas.iter().any(|d| d.len() < PRICE_DELTAS_PER_TICKER) {
            return None;
        }

        let mut price_deltas_flat = Vec::with_capacity(TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER);
        for ticker_deltas in &self.price_deltas {
            for &delta in ticker_deltas.iter().rev().take(PRICE_DELTAS_PER_TICKER) {
                price_deltas_flat.push(delta as f32);
            }
        }

        let current_prices = self.get_current_prices();
        let position_percents: Vec<f64> = self.account.positions
            .iter()
            .enumerate()
            .map(|(i, p)| p.value_with_price(current_prices[i]) / self.account.total_assets)
            .collect();

        let mut static_obs = Vec::with_capacity(STATIC_OBSERVATIONS);
        static_obs.push(1.0f32);
        static_obs.push((self.account.cash / self.account.total_assets) as f32);

        for &pos_pct in &position_percents {
            static_obs.push(pos_pct as f32);
        }

        // Time-weighted returns (simplified for live trading without lot tracking)
        for (i, position) in self.account.positions.iter().enumerate() {
            let time_weighted_return = if position.quantity > 0.0 && position.avg_price > 0.0 {
                let current_price = current_prices[i];
                let return_pct = (current_price - position.avg_price) / position.avg_price;
                let hold_time = (self.step_count.max(1)) as f64;
                return_pct / hold_time.sqrt()
            } else {
                0.0
            };
            static_obs.push(time_weighted_return as f32);
        }

        for i in 0..ACTION_HISTORY_LEN {
            if i < self.action_history.len() {
                let action_idx = self.action_history.len() - 1 - i;
                for &action in &self.action_history[action_idx] {
                    static_obs.push(action as f32);
                }
            } else {
                for _ in 0..(TICKERS_COUNT * 2) {
                    static_obs.push(0.0f32);
                }
            }
        }

        let price_deltas_tensor = Tensor::from_slice(&price_deltas_flat)
            .view([1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        let static_obs_tensor = Tensor::from_slice(&static_obs)
            .view([1, STATIC_OBSERVATIONS as i64]);

        Some((price_deltas_tensor, static_obs_tensor))
    }
}

fn sync_account_from_ibkr(
    client: &Client,
    symbols: &[String],
    account: &mut Account,
) -> Result<(), Box<dyn std::error::Error>> {
    let account_subscription = client.account_summary(&AccountGroup::from("All"), AccountSummaryTags::ALL)?;

    for update in &account_subscription {
        match update {
            AccountSummaryResult::Summary(summary) => {
                if summary.tag == "TotalCashValue" {
                    account.cash = summary.value.parse::<f64>()
                        .unwrap_or_else(|_| {
                            println!("Warning: Could not parse cash value");
                            account.cash
                        });

                    if let Some(max_value) = MAX_ACCOUNT_VALUE {
                        account.cash = account.cash.min(max_value);
                    }

                    account_subscription.cancel();
                    break;
                }
            }
            AccountSummaryResult::End => {
                account_subscription.cancel();
                break;
            }
        }
    }

    drop(account_subscription);

    let positions_subscription = client.positions()?;

    for ticker_idx in 0..symbols.len() {
        account.positions[ticker_idx].quantity = 0.0;
        account.positions[ticker_idx].avg_price = 0.0;
    }

    for position in &positions_subscription {
        match position {
            PositionUpdate::Position(pos) => {
                if let Some(ticker_idx) = symbols.iter().position(|s| s == pos.contract.symbol.as_str()) {
                    account.positions[ticker_idx].quantity = pos.position;
                    account.positions[ticker_idx].avg_price = pos.average_cost;
                }
            }
            PositionUpdate::PositionEnd => {
                positions_subscription.cancel();
                break;
            }
        }
    }

    Ok(())
}

pub fn run_ibkr_paper_trading<P: AsRef<Path>>(
    weight_path: P,
    symbols: Vec<String>,
    update_interval_secs: u64,
    max_steps: usize,
    temperature: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IBKR Paper Trading ===");
    println!("Weight path: {:?}", weight_path.as_ref());
    println!("Symbols: {:?}", symbols);
    println!("Update interval: {}s", update_interval_secs);
    println!("Max steps: {}", max_steps);

    let client = Client::connect(api::CONNECTION_URL, 100)?;
    println!("Connected to IBKR");

    let account_subscription = client.account_summary(&AccountGroup::from("All"), AccountSummaryTags::ALL)?;

    let mut starting_cash = 0.0;
    for update in &account_subscription {
        match update {
            AccountSummaryResult::Summary(summary) => {
                if summary.tag == "TotalCashValue" {
                    starting_cash = summary.value.parse::<f64>()
                        .unwrap_or_else(|_| {
                            println!("Warning: Could not parse cash value '{}', using 0.0", summary.value);
                            0.0
                        });

                    if let Some(max_value) = MAX_ACCOUNT_VALUE {
                        starting_cash = starting_cash.min(max_value);
                        println!("Account cash: ${:.2} (limited to ${:.2})", starting_cash, max_value);
                    } else {
                        println!("Account cash: ${:.2}", starting_cash);
                    }

                    account_subscription.cancel();
                    break;
                }
            }
            AccountSummaryResult::End => {
                account_subscription.cancel();
                break;
            }
        }
    }

    if starting_cash == 0.0 {
        return Err("Could not retrieve account cash from IBKR".into());
    }

    drop(account_subscription);

    let device = Device::cuda_if_available();
    let (_vs, model) = load_model(weight_path, device)?;

    let ticker_count = symbols.len();
    let state = Arc::new(Mutex::new(LiveMarketState::new(ticker_count, starting_cash)));

    let client_arc = Arc::new(client);

    for (ticker_idx, symbol) in symbols.iter().enumerate() {
        let contract = Contract::stock(symbol).build();
        let client_clone = Arc::clone(&client_arc);
        let state_clone = Arc::clone(&state);
        let symbol_clone = symbol.clone();

        thread::spawn(move || {
            let subscription = match client_clone.realtime_bars(
                &contract,
                BarSize::Sec5,
                WhatToShow::Trades,
                TradingHours::Regular,
            ) {
                Ok(sub) => sub,
                Err(e) => {
                    println!("Failed to subscribe to {}: {}", symbol_clone, e);
                    return;
                }
            };

            for bar in &subscription {
                let mut state = state_clone.lock().unwrap();
                state.update_price(ticker_idx, bar.close);
                state.update_account_total();
            }
        });
    }

    thread::sleep(std::time::Duration::from_secs(PRICE_DELTAS_PER_TICKER as u64 / 12 + 10));

    let mut step = 0;
    let start_time = Instant::now();

    loop {
        if step >= max_steps {
            break;
        }

        thread::sleep(std::time::Duration::from_secs(update_interval_secs));

        let mut state_guard = state.lock().unwrap();

        sync_account_from_ibkr(&client_arc, &symbols, &mut state_guard.account)?;
        state_guard.update_account_total();

        if let Some((price_deltas_tensor, static_obs_tensor)) = state_guard.build_observation() {
            let price_deltas_gpu = price_deltas_tensor.to_device(device);
            let static_obs_gpu = static_obs_tensor.to_device(device);

            let actions = sample_actions(
                model.as_ref(),
                &price_deltas_gpu,
                &static_obs_gpu,
                false,
                temperature,
            );

            let actions_vec = Vec::<f64>::try_from(actions.flatten(0, -1)).unwrap();

            state_guard.action_history.push_back(actions_vec.clone());
            if state_guard.action_history.len() > ACTION_HISTORY_LEN {
                state_guard.action_history.pop_front();
            }

            let (buy_sell_actions, _hold_actions) = actions_vec.split_at(TICKERS_COUNT as usize);
            let current_prices = state_guard.get_current_prices();

            execute_trades(
                &client_arc,
                &symbols,
                buy_sell_actions,
                &current_prices,
                &state_guard.account,
            )?;

            step += 1;
            state_guard.step_count = step;

            if step % 10 == 0 {
                print_status(step, &state_guard, &start_time);
            }
        } else {
            println!("Waiting for sufficient price data... ({}/{})",
                state_guard.price_deltas.iter().map(|d| d.len()).min().unwrap_or(0),
                PRICE_DELTAS_PER_TICKER);
        }
    }

    let final_state = state.lock().unwrap();
    println!("\n=== Final Summary ===");
    println!("Total steps: {}", final_state.step_count);
    println!("Starting cash: ${:.2}", starting_cash);
    println!("Final total assets: ${:.2}", final_state.account.total_assets);
    println!("Total P&L: ${:.2} ({:.2}%)",
        final_state.account.total_assets - starting_cash,
        (final_state.account.total_assets / starting_cash - 1.0) * 100.0);

    Ok(())
}

fn execute_trades(
    client: &Arc<Client>,
    symbols: &[String],
    actions: &[f64],
    current_prices: &[f64],
    account: &Account,
) -> Result<(), Box<dyn std::error::Error>> {
    for (ticker_idx, &action) in actions.iter().enumerate() {
        if action.abs() < ACTION_THRESHOLD {
            continue;
        }

        let current_price = current_prices[ticker_idx];
        let position = &account.positions[ticker_idx];
        
        // Needs to implement new trade fn logic from step.rs

        // if action > 0.0 {
        //     let max_ownership = account.total_assets / symbols.len() as f64;
        //     let buy_total = (max_ownership - position.value_with_price(current_price)) * action;

        //     if buy_total > MIN_ORDER_VALUE {
        //         let buy_total_clamped = buy_total.min(account.cash);

        //         if buy_total_clamped < MIN_ORDER_VALUE {
        //             continue;
        //         }

        //         let quantity = (buy_total_clamped / current_price).floor();

        //         println!("BUY {} shares of {} @ ${:.2} (total: ${:.2})",
        //             quantity, symbols[ticker_idx], current_price, buy_total_clamped);

        //         let contract = Contract::stock(&symbols[ticker_idx]);
        //         let order_id = client.next_order_id();
        //         let order = order_builder::market_order(Action::Buy, quantity);

        //         match client.place_order(order_id, &contract, &order) {
        //             Ok(subscription) => {
        //                 for event in &subscription {
        //                     if let PlaceOrder::ExecutionData(_) = event {
        //                         break;
        //                     }
        //                 }
        //             }
        //             Err(e) => println!("Order failed: {}", e),
        //         }
        //     }
        // } else {
        //     let sell_total = position.value_with_price(current_price) * (-action);
        //     if sell_total > MIN_ORDER_VALUE {
        //         let quantity = (sell_total / current_price).floor();

        //         let quantity_clamped = quantity.min(position.quantity);

        //         let sell_total_actual = quantity_clamped * current_price;

        //         if sell_total_actual < MIN_ORDER_VALUE {
        //             continue;
        //         }

        //         println!("SELL {} shares of {} @ ${:.2} (total: ${:.2})",
        //             quantity_clamped, symbols[ticker_idx], current_price, sell_total_actual);

        //         let contract = Contract::stock(&symbols[ticker_idx]);
        //         let order_id = client.next_order_id();
        //         let order = order_builder::market_order(Action::Sell, quantity_clamped);

        //         match client.place_order(order_id, &contract, &order) {
        //             Ok(subscription) => {
        //                 for event in &subscription {
        //                     if let PlaceOrder::ExecutionData(_) = event {
        //                         break;
        //                     }
        //                 }
        //             }
        //             Err(e) => println!("Order failed: {}", e),
        //         }
        //     }
        // }
    }

    Ok(())
}

fn print_status(step: usize, state: &LiveMarketState, start_time: &Instant) {
    let elapsed = Instant::now().duration_since(*start_time).as_secs_f32();
    let current_prices = state.get_current_prices();

    println!("\n--- Step {} (elapsed: {:.1}s) ---", step, elapsed);
    println!("Total Assets: ${:.2}", state.account.total_assets);
    println!("Cash: ${:.2}", state.account.cash);

    for (i, position) in state.account.positions.iter().enumerate() {
        if position.quantity > 0.0 {
            let value = position.value_with_price(current_prices[i]);
            let pnl_pct = position.appreciation(current_prices[i]) * 100.0;
            println!("Position {}: {:.2} shares @ ${:.2} (value: ${:.2}, P&L: {:.2}%)",
                i, position.quantity, current_prices[i], value, pnl_pct);
        }
    }
}
