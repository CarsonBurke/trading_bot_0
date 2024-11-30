use core::f64;
use std::{
    collections::{HashMap, HashSet},
    fs,
};

use ibapi::{
    client,
    market_data::historical::{self, HistoricalData},
    Client,
};

use crate::{
    agent::{self, Weight},
    charts::general::{assets_chart, buy_sell_chart, candle_chart, rsi_chart},
    constants::{self, rsi, BUY_WEIGHT, MAX_CHANGE, SELL_WEIGHT, TICKER, TICKERS},
    types::{Account, MakeCharts, MappedHistorical, Position},
    utils::{convert_historical, get_rsi_values, round_to_stock},
};

/// Returns: total assets
pub fn basic(
    mapped_data: &MappedHistorical,
    agent: &agent::Agent,
    account: &mut Account,
    make_charts: Option<MakeCharts>,
) -> f64 {
    let indexes = mapped_data.get(TICKERS[0]).unwrap().len();
    let mut positions_by_ticker: HashMap<String, Vec<f64>> = HashMap::new();

    for ticker in mapped_data.keys() {
        positions_by_ticker.insert(ticker.to_string(), Vec::new());
    }

    let mut total_assets = Vec::new();
    let mut cash_graph = Vec::new();

    let mut rsi_values_by_ticker = HashMap::new();

    for (ticker, bars) in mapped_data.iter() {
        let data = convert_historical(bars);
        let rsi = get_rsi_values(&data, agent.weights.map[Weight::RsiEmaAlpha]);
        rsi_values_by_ticker.insert(ticker, rsi);
    }

    let mut buy_indexes = HashMap::new();
    let mut sell_indexes = HashMap::new();

    // reset when selling
    // must be below this amount by a certain percent before can huy again
    let mut last_buy_price: Option<f64> = None;
    // reset when buying
    // must be above this amount by a certain percent before can sell again
    let mut last_sell_price: Option<f64> = None;

    account.cash = 10_000.;

    for (ticker, data) in mapped_data.iter() {
        account
            .positions
            .insert(ticker.to_string(), Position::default());
    }

    for index in 0..indexes {
        // Get and record some important data

        let mut total_positioned = 0.0;

        for (ticker, bars) in mapped_data.iter() {
            let price = bars[index].close;

            let position = account.positions.get_mut(ticker).unwrap();
            let positioned = position.value_with_price(price);

            positions_by_ticker
                .get_mut(ticker)
                .unwrap()
                .push(positioned);
            total_positioned += positioned;
        }

        let assets = account.cash + total_positioned;

        cash_graph.push(account.cash);
        total_assets.push(assets);

        // Conduct buys and sells

        for (ticker, bars) in mapped_data.iter() {
            let price = bars[index].close;

            let rsi_values = rsi_values_by_ticker.get(ticker).unwrap();
            let rsi = rsi_values[index];

            let position = account.positions.get_mut(ticker).unwrap();
            let ticker_positioned = position.value_with_price(price);

            if rsi >= agent.weights.map[Weight::MinRsiSell] * 100.0 {
                if position.avg_price >= price {
                    println!("insufficient avg price {}", position.avg_price);
                    continue;
                }
                if position.quantity == 0 {
                    println!("no position to sell");
                    continue;
                }
                if let Some(last_sell_price) = last_sell_price {
                    let percent_of_last_sell = price / last_sell_price;

                    // Iterate if we are below X% of last sell
                    if percent_of_last_sell < 1. + agent.weights.map[Weight::DiffToSell] {
                        println!("insufficient diff from last sell price {}", last_sell_price);
                        continue;
                    }
                }

                let Some((sell_price, sell_quantity)) =
                    get_sell_price_quantity(position, price, rsi, assets / TICKERS.len() as f64)
                else {
                    continue;
                };

                if sell_quantity == 0 {
                    panic!("insufficient sell quantity");
                }

                last_sell_price = Some(price);
                last_buy_price = None;

                sell_indexes.insert(index, (sell_price, sell_quantity));
                position.quantity -= sell_quantity;
                account.cash += sell_price;
                continue;
            }

            if price * 1.1 > account.cash {
                continue;
            }

            if rsi <= agent.weights.map[Weight::MaxRsiBuy] * 100.0 {
                if let Some(last_buy_price) = last_buy_price {
                    let percent_of_last_buy = price / last_buy_price;

                    // Iterate if we are below X% of last buy
                    if percent_of_last_buy > 1. - agent.weights.map[Weight::DiffToBuy] {
                        continue;
                    }
                };

                let Some((total_price, buy_quantity)) = get_buy_price_quantity(
                    &position,
                    price,
                    rsi,
                    assets / TICKERS.len() as f64,
                    account.cash,
                ) else {
                    continue;
                };

                if buy_quantity == 0 {
                    let cash = account.cash;
                    println!("buy failed 0 quantity {price} {total_price} {cash}");
                    continue;
                }

                last_buy_price = Some(price);

                buy_indexes.insert(index, (total_price, buy_quantity));
                position.add(price, buy_quantity);
                account.cash -= total_price;
                continue;
            }
        }
    }

    // for (index, (ticker, bars)) in mapped_data.iter().enumerate() {

    //     let rsi_values = rsi_values_by_ticker.get(ticker).unwrap();

    //     let mut total_positioned = 0.0;

    //     for bar in bars.iter() {
    //         let price = bar.close;

    //         let position = account.positions.get_mut(ticker).unwrap();
    //         let positioned = position.value_with_price(price);

    //         positions_by_ticker.get_mut(ticker).unwrap().push(positioned);
    //         total_positioned += positioned;
    //     }

    //     let assets = total_positioned + account.cash;

    //     cash_graph.push(account.cash);
    //     total_assets.push(assets);

    //     for (bar, rsi) in bars.iter().zip(rsi_values) {
    //         let rsi = *rsi;
    //         let price = bar.close;

    //         let position = account.positions.get_mut(ticker).unwrap();
    //         let ticker_positioned = position.value_with_price(price);

    //         if rsi >= agent.weights.map[Weight::MinRsiSell] * 100.0 {
    //             if position.avg_price >= price {
    //                 println!("insufficient avg price {}", position.avg_price);
    //                 continue;
    //             }
    //             if position.quantity == 0 {
    //                 println!("no position to sell");
    //                 continue;
    //             }
    //             if let Some(last_sell_price) = last_sell_price {
    //                 let percent_of_last_sell = price / last_sell_price;

    //                 // Iterate if we are below X% of last sell
    //                 if percent_of_last_sell < 1. + agent.weights.map[Weight::DiffToSell] {
    //                     println!("insufficient diff from last sell price {}", last_sell_price);
    //                     continue;
    //                 }
    //             }

    //             let Some((sell_price, sell_quantity)) = get_sell_price_quantity(position, price, rsi, assets / TICKERS.len() as f64) else {
    //                 continue;
    //             };

    //             if sell_quantity == 0 {
    //                 panic!("insufficient sell quantity");
    //             }

    //             last_sell_price = Some(price);
    //             last_buy_price = None;

    //             sell_indexes.insert(index, (sell_price, sell_quantity));
    //             position.quantity -= sell_quantity;
    //             account.cash += sell_price;
    //             continue;
    //         }

    //         if price * 1.1 > account.cash {
    //             continue;
    //         }

    //         if rsi <= agent.weights.map[Weight::MaxRsiBuy] * 100.0 {
    //             if let Some(last_buy_price) = last_buy_price {
    //                 let percent_of_last_buy = price / last_buy_price;

    //                 // Iterate if we are below X% of last buy
    //                 if percent_of_last_buy > 1. - agent.weights.map[Weight::DiffToBuy] {
    //                     continue;
    //                 }
    //             };

    //             let Some((total_price, buy_quantity)) = get_buy_price_quantity(&position, price, rsi, assets / TICKERS.len() as f64, account.cash) else {
    //                 continue;
    //             };

    //             if buy_quantity == 0 {
    //                 let cash = account.cash;
    //                 println!("buy failed 0 quantity {price} {total_price} {cash}");
    //                 continue;
    //             }

    //             last_buy_price = Some(price);

    //             buy_indexes.insert(index, (total_price, buy_quantity));
    //             position.add(price, buy_quantity);
    //             account.cash -= total_price;
    //             continue;
    //         }
    //     }
    // }

    println!("buy indexes: {buy_indexes:?}");
    println!("sell indexes: {sell_indexes:?}");

    let cash = account.cash;
    println!("ended up with cash: {cash:?}");

    /* let value = account.positions.get(TICKER).unwrap().quantity as f64 * *data.last().unwrap();
    println!("ended up with value: {value:?}"); */

    /* let total = cash + value;
    println!("ended up with total value of: {total:?}"); */

    /* buy_sell_chart(data, &buy_indexes, &sell_indexes).unwrap();
    assets_chart(&total_assets, &positioned_assets).unwrap(); */

    if let Some(charts_config) = make_charts {
        let base_dir = format!("charts/gens/{}", charts_config.generation);
        create_folder_if_not_exists(&base_dir);

        assets_chart(&base_dir, &total_assets, &cash_graph);

        for (ticker, bars) in mapped_data.iter() {
            let ticker_dir = format!("charts/gens/{}/{ticker}", charts_config.generation);
            create_folder_if_not_exists(&ticker_dir);

            let data = convert_historical(bars);

            candle_chart(&ticker_dir, bars);
            buy_sell_chart(&ticker_dir, &data, &buy_indexes, &sell_indexes);

            let rsi_values = rsi_values_by_ticker.get(ticker).unwrap();
            rsi_chart(&ticker_dir, &rsi_values);

            assets_chart(&ticker_dir, &total_assets, &cash_graph);
        }
    }

    *total_assets.last().unwrap()
}

pub fn create_folder_if_not_exists(dir: &String) {
    let _ = fs::create_dir_all(dir);
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
        return None;
    }
    Some(round_to_stock(price, want_stake))
}

pub fn max_stake_for_rsi(rsi: f64, available: f64) -> f64 {
    available * (rsi / 100.) * BUY_WEIGHT
}

pub fn max_sell_for_rsi(rsi: f64, holding: f64) -> f64 {
    holding * (rsi / 100.) * SELL_WEIGHT
}
