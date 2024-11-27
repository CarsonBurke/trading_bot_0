use std::collections::HashSet;

use ibapi::{client, market_data::historical::{self, HistoricalData}, Client};

use crate::{constants::rsi, types::{Account, Data, Position}, utils::{buy_sell_chart, get_rsi_values, round_to_stock}};

pub fn basic(client: &Client, data: &Data, account: &mut Account) {

    let ticker = "APPL";

    let rsi_values = get_rsi_values(data);
    let mut buy_indexes = Vec::new();
    let mut sell_indexes = Vec::new();

    account.cash = 1000.;
    account.positions.insert(ticker.to_string(), Position::default());
    
    for ((index, price), rsi) in data.iter().enumerate().zip(rsi_values) {
        if rsi >= rsi::MIN_SELL {
            let Some(position) = account.positions.get_mut(ticker) else {
                continue;
            };
            if position.quantity == 0 {
                continue;
            }
            let sell_want = max_sell_for_rsi(rsi, position.quantity as f64 * *price);
            let (sell_price, sell_quantity) = round_to_stock(*price, sell_want);

            if sell_quantity == 0 {
                continue;
            }

            sell_indexes.push((index, sell_price));
            position.quantity -= sell_quantity;
            account.cash += sell_price;
            continue;
        }

        if *price > account.cash {
            println!("insufficient funds");
            continue;
        }

        if rsi <= rsi::MAX_BUY {
            println!("buy pos check");
            let Some(position) = account.positions.get_mut(ticker) else {
                continue;
            };
            println!("trying to buy");
            let want_stake = max_stake_for_rsi(rsi, account.cash);
            let (buy_price, buy_quantity) = round_to_stock(*price, want_stake);

            buy_indexes.push((index, buy_price));
            position.quantity += buy_quantity;
            account.cash -= buy_price;
            continue;
        }
    }

    println!("buy indexes: {buy_indexes:?}");
    println!("sell indexes: {sell_indexes:?}");

    let cash = account.cash;
    println!("ended up with cash: {cash:?}");

    let value = account.positions.get(ticker).unwrap().quantity as f64 * *data.last().unwrap();
    println!("ended up with value: {value:?}");

    buy_sell_chart(buy_indexes, sell_indexes).unwrap();    
}

pub fn max_stake_for_rsi(rsi: f64, available: f64) -> f64 {
    available * (rsi / 100.)
}

pub fn max_sell_for_rsi(rsi: f64, holding: f64) -> f64 {
    holding * (rsi / 100.)
}