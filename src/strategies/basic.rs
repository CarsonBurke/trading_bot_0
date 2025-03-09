use core::f64;

use hashbrown::HashMap;
use ibapi::{
    client,
    market_data::historical::{self, HistoricalData},
    Client,
};
use time::convert::Week;

use crate::{
    agent::{self, Agent, Weight},
    charts::general::{assets_chart, buy_sell_chart, candle_chart, simple_chart},
    constants::{self, files::TRAINING_PATH, rsi, MAX_VALUE_PER_TICKER, TICKERS},
    types::{Account, MakeCharts, MappedHistorical, Position},
    utils::{convert_historical, create_folder_if_not_exists, ema, get_rsi_values, round_to_stock},
};

/// Returns: total assets
pub fn basic(
    mapped_data: &MappedHistorical,
    agent: &agent::Agent,
    account: &mut Account,
    make_charts: Option<MakeCharts>,
) -> f64 {
    let indexes = mapped_data[0].len();
    let mut positions_by_ticker: Vec<Vec<f64>> = Vec::new();

    for (ticker, _) in mapped_data.iter().enumerate() {
        positions_by_ticker.insert(ticker, Vec::new());
    }

    let mut total_assets = Vec::new();
    let mut cash_graph = Vec::new();

    let mut decider_rsi_values_by_ticker = Vec::new();

    for (ticker, bars) in mapped_data.iter().enumerate() {
        let data = convert_historical(bars);
        let rsi = get_rsi_values(&data, agent.weights.map[Weight::DeciderRsiEmaAlpha]);
        decider_rsi_values_by_ticker.insert(ticker, rsi);
    }

    let mut amount_rsi_values_by_ticker = Vec::new();

    for (ticker, bars) in mapped_data.iter().enumerate() {
        let data = convert_historical(bars);
        let rsi = get_rsi_values(&data, agent.weights.map[Weight::AmountRsiEmaAlpha]);
        amount_rsi_values_by_ticker.insert(ticker, rsi);
    }

    let mut price_emas_by_ticker = Vec::new();

    for (ticker, bars) in mapped_data.iter().enumerate() {
        let data = convert_historical(bars);
        let ema = ema(&data, agent.weights.map[Weight::PriceEmaAlpha]);
        price_emas_by_ticker.insert(ticker, ema);
    }

    let mut buy_indexes = Vec::new();
    let mut sell_indexes = Vec::new();

    for _ in 0..indexes {
        buy_indexes.push(HashMap::new());
        sell_indexes.push(HashMap::new());
    }

    // The lowest RSI since we last bought
    let mut lowest_rsis: HashMap<usize, f64> = HashMap::new();
    // The highest RSI since we last sold
    let mut highest_rsis: HashMap<usize, f64> = HashMap::new();

    // reset when selling
    // must be below this amount by a certain percent before can buy again
    let mut last_buy_price: HashMap<usize, f64> = HashMap::new();
    // reset when buying
    // must be above this amount by a certain percent before can sell again
    let mut last_sell_price: HashMap<usize, f64> = HashMap::new();

    account.cash = 10_000.;

    for (ticker, _) in mapped_data.iter().enumerate() {
        account
            .positions
            .insert(ticker, Position::default());
    }

    for index in 0..indexes {
        // Get and record some important data

        let mut total_positioned = 0.0;

        for (ticker, bars) in mapped_data.iter().enumerate() {
            let price = bars[ticker].close;

            let position = account.positions.get_mut(ticker).unwrap();
            let positioned = position.value_with_price(price);

            positions_by_ticker[ticker]
                .push(positioned);
            total_positioned += positioned;
        }

        let assets = account.cash + total_positioned;

        cash_graph.push(account.cash);
        total_assets.push(assets);

        // Conduct buys and sells

        for (ticker, bars) in mapped_data.iter().enumerate() {
            let price = bars[ticker].close;

            let decider_rsi_values = decider_rsi_values_by_ticker.get(ticker).unwrap();
            let decider_rsi = decider_rsi_values[index];

            let amount_rsi_values = amount_rsi_values_by_ticker.get(ticker).unwrap();
            let amount_rsi = amount_rsi_values[index];

            let price_ema = price_emas_by_ticker.get(ticker).unwrap()[index];

            if can_try_sell(ticker, decider_rsi, &mut highest_rsis, agent) {
                try_sell(
                    ticker,
                    index,
                    price,
                    amount_rsi,
                    decider_rsi,
                    assets,
                    agent,
                    account,
                    &mut highest_rsis,
                    &mut last_sell_price,
                    &mut last_buy_price,
                    &mut sell_indexes,
                    price_ema,
                );
            }

            if price * 1.1 > account.cash {
                continue;
            }

            if can_try_buy(ticker, decider_rsi, &mut lowest_rsis, agent) {
                if let Some(last_buy_price) = last_buy_price.get(&ticker) {
                    let percent_of_last_buy = price / last_buy_price;

                    // Iterate if we are below X% of last buy
                    if percent_of_last_buy > 1. - agent.weights.map[Weight::DiffToBuy] {
                        continue;
                    }
                };

                let position = account.positions.get_mut(ticker).unwrap();

                // println!("decider rsi: {}", decider_rsi);
                // println!("amount rsi: {}", amount_rsi);
                // println!("diff {}", decider_rsi - amount_rsi);
                let rsi_diff = amount_rsi;
                let Some((total_price, buy_quantity)) = get_buy_price_quantity(
                    position,
                    price,
                    rsi_diff,
                    assets, /*  / (TICKERS.len() / 2) as f64 */
                    account.cash,
                    agent,
                    price_ema,
                ) else {
                    continue;
                };

                if buy_quantity == 0 {
                    continue;
                }

                /* // Require the rsi to have rebounded slightly up before we can purchase
                if let Some(ticker_lowest_rsi) = lowest_rsis.get(ticker) {
                    // lowest rsi = 30
                    // min = 30 + 30 * 0.05 = 31.5
                    // rsi = 31
                    // rsi must be above min to be accepted
                    let min = ticker_lowest_rsi
                        + ticker_lowest_rsi * agent.weights.map[Weight::ReboundSellThreshold];
                    if decider_rsi < min {
                        continue;
                    }
                } */

                // Buy

                // println!("bought at rsi_diff {}", rsi_diff);

                lowest_rsis.remove(&ticker);

                last_buy_price.insert(ticker, price);
                last_sell_price.remove(&ticker);

                let ticker_buy_indexes = buy_indexes.get_mut(ticker).unwrap();
                ticker_buy_indexes.insert(index, (total_price, buy_quantity as f64));

                position.add(price, buy_quantity as f64);
                account.cash -= total_price;
                continue;
            }
        }
    }

    /*     println!("buy indexes: {buy_indexes:?}");
    println!("sell indexes: {sell_indexes:?}"); */

    /*     let cash = account.cash;
    println!("ended up with cash: {cash:?}");

    let assets = total_assets.last().unwrap();
    println!("ended up with assets: {assets:?}"); */

    /* let value = account.positions.get(TICKER).unwrap().quantity as f64 * *data.last().unwrap();
    println!("ended up with value: {value:?}"); */

    /* let total = cash + value;
    println!("ended up with total value of: {total:?}"); */

    /* buy_sell_chart(data, &buy_indexes, &sell_indexes).unwrap();
    assets_chart(&total_assets, &positioned_assets).unwrap(); */

    if let Some(charts_config) = make_charts {
        println!("Generating charts for gen: {}", charts_config.generation);

        let base_dir = format!("training/gens/{}", charts_config.generation);
        create_folder_if_not_exists(&base_dir);

        assets_chart(&base_dir, &total_assets, &cash_graph, None);

        for (ticker, bars) in mapped_data.iter().enumerate() {
            let price = bars[ticker].close;

            let ticker_dir = format!("{TRAINING_PATH}/gens/{}/{ticker}", charts_config.generation);
            create_folder_if_not_exists(&ticker_dir);

            let data = convert_historical(bars);

            /* candle_chart(&ticker_dir, bars); */

            let ticker_buy_indexes = buy_indexes.get(ticker).unwrap();
            let ticker_sell_indexes = sell_indexes.get(ticker).unwrap();
            buy_sell_chart(
                &ticker_dir,
                &data,
                &ticker_buy_indexes,
                &ticker_sell_indexes,
            );

            let rsi_values = decider_rsi_values_by_ticker.get(ticker).unwrap();
            simple_chart(&ticker_dir, "decider_rsi", &rsi_values);

            let amount_rsi_values = amount_rsi_values_by_ticker.get(ticker).unwrap();
            simple_chart(&ticker_dir, "amount_rsi", &amount_rsi_values);

            let price_ema_values = price_emas_by_ticker.get(ticker).unwrap();
            simple_chart(&ticker_dir, "price_ema", &price_ema_values);

            /* let rsi_diff_values = rsi_values
                .iter()
                .zip(amount_rsi_values.iter())
                .map(|(decider, amount)| amount - decider)
                .collect();
            simple_chart(&ticker_dir, "rsi_diff", &rsi_diff_values); */

            let positioned_assets = positions_by_ticker.get(ticker).unwrap();
            assets_chart(
                &ticker_dir,
                &total_assets,
                &cash_graph,
                Some(&positioned_assets),
            );
        }
    }

    *total_assets.last().unwrap()
}

pub fn can_try_sell(
    ticker: usize,
    decider_rsi: f64,
    highest_rsis: &mut HashMap<usize, f64>,
    agent: &Agent,
) -> bool {
    // Require the rsi to have rebounded slightly down before we can purchase
    /* if let Some(ticker_highest_rsi) = highest_rsis.get(ticker) {
        let max = ticker_highest_rsi
            + ticker_highest_rsi * agent.weights.map[Weight::ReboundSellThreshold];
        if decider_rsi <= max {
            // highest_rsis.remove(ticker);
            return true;
        }
    } */

    if decider_rsi < agent.weights.map[Weight::MinRsiSell] * 100.0 {
        return false;
    }

    // Record the RSI if it's a new local maximum
    if let Some(ticker_highest_rsi) = highest_rsis.get(&ticker) {
        if decider_rsi > *ticker_highest_rsi {
            highest_rsis.insert(ticker, decider_rsi);

            // However, we know we aren't allowed to sell if this is a new local max
            return false;
        }
    } else {
        highest_rsis.insert(ticker, decider_rsi);
    }

    true
}

pub fn can_try_buy(
    ticker: usize,
    decider_rsi: f64,
    lowest_rsis: &mut HashMap<usize, f64>,
    agent: &Agent,
) -> bool {
    // Require the rsi to have rebounded slightly down before we can purchase
    /* if let Some(ticker_lowest_rsi) = lowest_rsis.get(ticker) {
        let min =
            ticker_lowest_rsi + ticker_lowest_rsi * agent.weights.map[Weight::ReboundSellThreshold];
        if decider_rsi >= min {
            // lowest_rsis.remove(ticker);
            return true;
        }
    } */

    if decider_rsi > agent.weights.map[Weight::MaxRsiBuy] * 100.0 {
        return false;
    }

    // Record the RSI if it's a new local minimum
    if let Some(ticker_lowest_rsi) = lowest_rsis.get(&ticker) {
        if decider_rsi < *ticker_lowest_rsi {
            lowest_rsis.insert(ticker, decider_rsi);
            return false;
        }
    } else {
        lowest_rsis.insert(ticker, decider_rsi);
    }
    true
}

pub fn try_sell(
    ticker: usize,
    index: usize,
    price: f64,
    amount_rsi: f64,
    decider_rsi: f64,
    assets: f64,
    agent: &Agent,
    account: &mut Account,
    highest_rsis: &mut HashMap<usize, f64>,
    last_sell_price: &mut HashMap<usize, f64>,
    last_buy_price: &mut HashMap<usize, f64>,
    sell_indexes: &mut Vec<HashMap<usize, (f64, f64)>>,
    price_ema: f64,
) {
    let position = &mut account.positions[ticker];

    if position.avg_price >= price {
        return;
    }
    if position.quantity == 0. {
        /* println!("no position to sell"); */
        return;
    }
    /* if let Some(last_sell_price) = last_sell_price.get(ticker) {
        let percent_of_last_sell = price / last_sell_price;

        // Iterate if we are below X% of last sell
        if percent_of_last_sell < 1. + agent.weights.map[Weight::DiffToSell] {
            continue;
        }
    } */

    // println!("decider rsi: {}", decider_rsi);
    // println!("amount rsi: {}", amount_rsi);
    // println!("diff {}", decider_rsi - amount_rsi);
    let rsi_diff = amount_rsi;
    let Some((sell_price, sell_quantity)) = get_sell_price_quantity(
        &position,
        price,
        rsi_diff,
        assets, /*  / (TICKERS.len() / 2) as f64 */
        account.cash,
        agent,
        price_ema,
    ) else {
        return;
    };

    if sell_quantity == 0 {
        return;
    }

    // Sell

    // println!("sold at rsi_diff {}", rsi_diff);

    highest_rsis.remove(&ticker);

    last_sell_price.insert(ticker, price);
    last_buy_price.remove(&ticker);

    let ticker_sell_indexes = sell_indexes.get_mut(ticker).unwrap();
    ticker_sell_indexes.insert(index, (sell_price, sell_quantity as f64));

    position.quantity -= sell_quantity as f64;
    account.cash += sell_price;
}

pub fn get_sell_price_quantity(
    position: &Position,
    price: f64,
    rsi: f64,
    assets: f64,
    cash: f64,
    agent: &Agent,
    price_ema: f64,
) -> Option<(f64, u32)> {
    // let available_sell = (position.quantity as f64) * price;
    // let weight = agent.weights.map[Weight::RsiSellAmountWeight];

    // let sell_want = max_sell_for_rsi(rsi, assets, available_sell, weight).min(available_sell);

    let available =
        cash.min((assets/* - position.value_with_price(price) */) / TICKERS.len() as f64);
    let has = position.value_with_price(price);
    let weight = agent.weights.map[Weight::RsiSellAmountWeight];

    let diff_from_avg = price_ema - price;
    let percent_diff_from_avg = diff_from_avg / price_ema;
    if percent_diff_from_avg < 0. {
        return None;
    }

    let percent = percent_diff_from_avg.powf(weight); /* ((100. - rsi) / 100.) * weight/* .powf(weight * 10.) */; */
    // How much of this stock we want to have
    let want = available * percent;
    // if want <= has {
    //     return None
    // }
    // println!("want: {}", want);
    // println!("has: {}", position.value_with_price(price));
    let sell_want = (has - want).min(has);
    if sell_want < price {
        return None; /* Some((price, 1)); */
    }

    Some(round_to_stock(price, sell_want))
}

pub fn get_buy_price_quantity(
    position: &Position,
    price: f64,
    rsi: f64,
    assets: f64,
    cash: f64,
    agent: &Agent,
    price_ema: f64,
) -> Option<(f64, u32)> {
    // let available_cash = cash.min(assets /* - position.value_with_price(price) */);
    // let weight = agent.weights.map[Weight::RsiBuyAmountWeight];

    // let buy_want = max_buy_for_rsi(rsi, assets, available_cash, weight).min(available_cash);

    let available =
        cash.min((assets/* - position.value_with_price(price) */) / TICKERS.len() as f64);
    let weight = agent.weights.map[Weight::RsiBuyAmountWeight];

    let diff_from_avg = price_ema - price;
    let percent_diff_from_avg = diff_from_avg / price_ema;
    if percent_diff_from_avg < 0. {
        return None;
    }

    let percent = percent_diff_from_avg.powf(weight); /* ((100. - rsi) / 100.) * weight; */
    /* ((100. - rsi) / 100.).powf(weight * 10.); */
    // How much of this stock we want to have
    let want = available * percent;

    let buy_want = want - position.value_with_price(price);
    // println!("buy want: {}", buy_want);
    if buy_want < price {
        return None;
    }
    Some(round_to_stock(price, buy_want))
}

pub fn max_buy_for_rsi(rsi: f64, assets: f64, available: f64, weight: f64) -> f64 {
    if weight <= 0. {
        return available * (rsi / 100.) * (1. - weight);
    }
    /* available *  */
    assets * ((rsi::MIDDLE - rsi) / 100. / 100.).powf(1. - weight)
}

pub fn max_sell_for_rsi(rsi: f64, assets: f64, holding: f64, weight: f64) -> f64 {
    if weight <= 0. {
        return holding * (rsi / 100.) * (1. - weight);
    }
    /* holding *  */
    assets * ((rsi - rsi::MIDDLE) / 100. / 10.).powf(1. - weight)
}
