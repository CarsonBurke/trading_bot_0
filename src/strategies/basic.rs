use core::f64;
use std::collections::{HashMap, HashSet};

use ibapi::{
    client,
    market_data::historical::{self, HistoricalData},
    Client,
};

use crate::{
    constants::{self, rsi, BUY_WEIGHT, MAX_CHANGE, SELL_WEIGHT},
    types::{Account, Data, Position},
    utils::{assets_chart, buy_sell_chart, get_rsi_values, round_to_stock},
};

pub fn basic(client: &Client, data: &Data, account: &mut Account) {
    let ticker = "NVDA";

    let mut total_assets = Vec::new();
    let mut positioned_assets = Vec::new();

    let rsi_values = get_rsi_values(data);
    let mut buy_indexes = HashMap::new();
    let mut sell_indexes = HashMap::new();

    // reset when selling
    // must be below this amount by a certain percent before can huy again
    let mut last_buy_price: Option<f64> = None;
    // reset when buying
    // must be above this amount by a certain percent before can sell again
    let mut last_sell_price: Option<f64> = None;

    account.cash = 10_000.;
    account
        .positions
        .insert(ticker.to_string(), Position::default());

    for ((index, price), rsi) in data.iter().enumerate().zip(rsi_values) {
        let position = account.positions.get_mut(ticker).unwrap();

        let positioned = position.value_with_price(*price);
        positioned_assets.push(positioned);

        let assets = positioned + account.cash;
        total_assets.push(assets);

        if rsi >= rsi::MIN_SELL {
            if position.avg_price >= *price {
                println!("insufficient avg price {}", position.avg_price);
                continue;
            }
            if position.quantity == 0 {
                println!("no position to sell");
                continue;
            }
            if let Some(last_sell_price) = last_sell_price {
                let percent_of_last_sell = *price / last_sell_price;

                // Iterate if we are below X% of last sell
                if percent_of_last_sell < 1.02 {
                    println!("insufficient diff from last sell price {}", last_sell_price);
                    continue;
                }
            }

            let Some((sell_price, sell_quantity)) = get_sell_price_quantity(position, *price, rsi, assets) else {
                continue;
            };

            if sell_quantity == 0 {
                panic!("insufficient sell quantity");
            }

            last_sell_price = Some(*price);
            last_buy_price = None;

            sell_indexes.insert(index, (sell_price, sell_quantity));
            position.quantity -= sell_quantity;
            account.cash += sell_price;
            continue;
        }

        if *price * 1.1 > account.cash {
            continue;
        }

        if rsi <= rsi::MAX_BUY {
            if let Some(last_buy_price) = last_buy_price {
                let percent_of_last_buy = *price / last_buy_price;

                // Iterate if we are below X% of last buy
                if percent_of_last_buy > 0.98 {
                    continue;
                }
            };

            let Some((total_price, buy_quantity)) = get_buy_price_quantity(&position, *price, rsi, assets, account.cash) else {
                continue;
            };


            if buy_quantity == 0 {
                let cash = account.cash;
                println!("buy failed 0 quantity {price} {total_price} {cash}");
                continue;
            }

            last_buy_price = Some(*price);

            buy_indexes.insert(index, (total_price, buy_quantity));
            position.add(*price, buy_quantity);
            account.cash -= total_price;
            continue;
        }
    }

    println!("buy indexes: {buy_indexes:?}");
    println!("sell indexes: {sell_indexes:?}");

    let cash = account.cash;
    println!("ended up with cash: {cash:?}");

    let value = account.positions.get(ticker).unwrap().quantity as f64 * *data.last().unwrap();
    println!("ended up with value: {value:?}");

    let total = cash + value;
    println!("ended up with total value of: {total:?}");

    println!("assest positions over time {:?}", positioned_assets);

    buy_sell_chart(&data, &buy_indexes, &sell_indexes).unwrap();
    assets_chart(&total_assets, &positioned_assets).unwrap();
}

pub fn get_sell_price_quantity(
    position: &Position,
    price: f64,
    rsi: f64,
    assets: f64,
) -> Option<(f64, u32)> {
    let available_sell = ((position.quantity as f64) * price).min(assets * MAX_CHANGE);
    let sell_want = max_sell_for_rsi(rsi, available_sell);
    if sell_want < price {
        println!("insufficient sell want {}", sell_want);
        return Some((price, 1));
    }

    Some(round_to_stock(price, sell_want))
}

pub fn get_buy_price_quantity(
    position: &Position,
    price: f64,
    rsi: f64,
    assets: f64,
    cash: f64,
) -> Option<(f64, u32)> {

    let available_cash = cash.min(assets * MAX_CHANGE).min(cash);
    let want_stake = max_stake_for_rsi(rsi, available_cash);
    if want_stake < price {
        return None
    }
    Some(round_to_stock(price, want_stake))
}

pub fn max_stake_for_rsi(rsi: f64, available: f64) -> f64 {
    available * (rsi / 100.) * BUY_WEIGHT
}

pub fn max_sell_for_rsi(rsi: f64, holding: f64) -> f64 {
    holding * (rsi / 100.) * SELL_WEIGHT
}
