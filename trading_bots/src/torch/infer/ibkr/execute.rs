use std::sync::Arc;

use ibapi::Client;

use crate::torch::constants::{ACTION_THRESHOLD, TICKERS_COUNT};
use crate::types::Account;

pub(super) fn execute_trades(
    _client: &Arc<Client>,
    _symbols: &[String],
    actions: &[f64],
    current_prices: &[f64],
    account: &Account,
) -> Result<(), Box<dyn std::error::Error>> {
    for (ticker_idx, &action) in actions.iter().take(TICKERS_COUNT as usize).enumerate() {
        if action.abs() < ACTION_THRESHOLD {
            continue;
        }

        let _current_price = current_prices[ticker_idx];
        let _position = &account.positions[ticker_idx];

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
