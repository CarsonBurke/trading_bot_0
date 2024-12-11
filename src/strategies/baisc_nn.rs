use colored::Colorize;
use hashbrown::HashMap;
use rust_neural_network::neural_network::{Input, NeuralNetwork};

use crate::{
    charts::general::{assets_chart, buy_sell_chart, simple_chart, want_chart},
    constants::{
        self,
        files::TRAINING_PATH,
        neural_net::{BUY_INDEX, INDEX_STEP, MAX_STEPS, SELL_INDEX},
        TICKERS,
    },
    neural_net::create::Indicators,
    types::{Account, Data, MakeCharts, MappedHistorical, Position},
    utils::{convert_historical, create_folder_if_not_exists, ema, find_highest, get_rsi_values},
};

pub fn baisc_nn(
    mapped_data: &MappedHistorical,
    account: &mut Account,
    mut neural_network: NeuralNetwork,
    // mapped_indicators: &Vec<Indicators>,
    mut inputs: Vec<Input>,
    mut make_charts: Option<MakeCharts>,
) -> f64 {
    let indexes = mapped_data[0].len();

    for ticker_index in TICKERS {
        account.positions.push(Position::default());
    }

    let mut positions_by_ticker: Vec<Vec<f64>> = Vec::new();

    for ticker_index in 0..mapped_data.len() {
        positions_by_ticker.push(Vec::new());
    }

    let mut cash_graph = Vec::new();
    let mut total_assets = Vec::new();

    let mut buy_indexes = Vec::new();
    let mut sell_indexes = Vec::new();

    for ticker_index in 0..mapped_data.len() {
        buy_indexes.push(HashMap::new());
        sell_indexes.push(HashMap::new());
    }

    let mut want_indexes = Vec::new();

    for ticker_index in 0..mapped_data.len() {
        want_indexes.push(HashMap::new());
    }

    account.cash = 10_000.;

    for index in 100..indexes {
        // Get and record some important data

        let mut total_positioned = 0.0;

        for (ticker_index, bars) in mapped_data.iter().enumerate() {
            let price = bars[index].close;

            let position = &account.positions[ticker_index];
            let positioned = position.value_with_price(price);

            positions_by_ticker[ticker_index].push(positioned);
            total_positioned += positioned;
        }

        let assets = account.cash + total_positioned;

        cash_graph.push(account.cash);
        total_assets.push(assets);

        for (ticker_index, bars) in mapped_data.iter().enumerate() {
            let price = bars[index].close;

            let position = &mut account.positions[ticker_index];

            // Assign inputs

            inputs[0].values[0] = account.cash / assets;
            inputs[1].values[0] = position.value_with_price(price) / assets;
            inputs[2].values[0] = match position.quantity {
                0. => 0.,
                _ => (price - position.avg_price) / position.avg_price,
            };

            for i in ((index.saturating_sub(INDEX_STEP))..index).rev() {
                let diff_percent = (mapped_data[ticker_index][i].close - price) / price;
                // println!("i: {}", index - i);
                inputs[index - i + 3].values[0] = diff_percent;
            }

            let min = index.saturating_sub(MAX_STEPS * INDEX_STEP);
            for (i, stocki) in (min..(index - INDEX_STEP - 1)).step_by(INDEX_STEP).rev().enumerate() {
                let diff_percent = (mapped_data[ticker_index][stocki].close - price) / price;
                // println!("i: {} index: {} i2: {}", i, index, i / INDEX_STEP + 3);
                inputs[i + 3].values[0] = diff_percent;
            }

            // let input_count = MAX_STEPS + 3;

            // let indicators = &mapped_indicators[0];
            // for (key, val) in indicators.iter() {
            //     inputs[key as usize + input_count].values[0] = val[index];
            // }

            // let input_values = inputs.iter().map(|input| input.values[0]).collect::<Vec<f64>>();
            // println!("inputs: {input_values:?}");

            // println!(
            //     "inputs 0: {} inputs 1: {} inputs 2: {} avg_price: {} quantity {}",
            //     inputs[0].values[0],
            //     inputs[1].values[0],
            //     inputs[2].values[0],
            //     position.avg_price,
            //     position.quantity
            // );
            // Forward propagate

            neural_network.forward_propagate(&inputs);

            let last_layer = neural_network.activation_layers.last().unwrap();

            let (output_index, percent) = find_highest(last_layer);
            if *percent <= 0. {
                continue;
            }

            /* if output_index == constants::neural_net::HOLD_INDEX as usize {
                continue;
            } */

            // println!("index: {}, value: {}", index, percent);
            // let values = inputs.iter().map(|input| input.values[0]).collect::<Vec<f64>>();
            // println!("inputs: {values:?}");

            /* let change = last_layer[constants::neural_net::BUY_INDEX]
                - last_layer[constants::neural_net::SELL_INDEX];
            let current = match position.quantity {
                0. => 0.,
                _ => position.value_with_price(price),
            };

            want_indexes[ticker_index].insert(index, change);

            // println!("change: {change}, current: {current}");
            if change > 0. {
                let buy = (change).min(account.cash).min(assets / TICKERS.len() as f64 - current);
                if buy <= 0. {
                    continue;
                }

                let quantity = buy / price;

                position.add(price, quantity);
                account.cash -= buy;

                buy_indexes[ticker_index].insert(index, (price, quantity));
                continue;
            }

            let sell = (change.abs()).min(current);
            if sell == 0. {
                continue;
            }

            let quantity = sell / price;

            position.quantity -= quantity;
            account.cash += sell;

            sell_indexes[ticker_index].insert(index, (price, quantity));

            continue; */

            // Get the want from the determined percent, at a maximum of 10%
            let gross_want = assets * last_layer[BUY_INDEX] / 100. /* - assets * last_layer[SELL_INDEX] / 100. */ /* *percent *//* assets * percent / 10000. *//* assets * (percent / 1000.).min(0.2) */;
            if gross_want <= 1. {
                continue;
            }

            let current = match position.quantity {
                0. => 0.,
                _ => position.value_with_price(price),
            };

            want_indexes[ticker_index].insert(index, gross_want);

            // If we want more than we have, try to buy
            if gross_want > current {
                let buy = (gross_want - current).min(account.cash).min(assets / 5.);
                if buy == 0. {
                    continue;
                }

                let quantity = buy / price;

                position.add(price, quantity);
                account.cash -= buy;

                buy_indexes[ticker_index].insert(index, (price, quantity));
                continue;
            }

            // Otherwise we want less than we have, try to sell

            let sell = ((gross_want - current).abs()).min(current).min(assets / 5.);
            if sell == 0. {
                continue;
            }

            let quantity = sell / price;

            position.quantity -= quantity;
            account.cash += sell;

            sell_indexes[ticker_index].insert(index, (price, quantity));
        }
    }

    if *total_assets.last().unwrap() > 1_000_000. {
        println!("{}", "total assets exceeds 1m".red());

        make_charts = Some(MakeCharts {
            generation: 1_000_000,
        });
    }

    if let Some(charts_config) = make_charts {
        println!("Generating charts for gen: {}", charts_config.generation);

        let base_dir = format!("training/gens/{}", charts_config.generation);
        create_folder_if_not_exists(&base_dir);

        let _ = assets_chart(&base_dir, &total_assets, &cash_graph, None);

        for (ticker_index, bars) in mapped_data.iter().enumerate() {
            let ticker = TICKERS[ticker_index].to_string();

            let ticker_dir = format!("{TRAINING_PATH}/gens/{}/{ticker}", charts_config.generation);
            create_folder_if_not_exists(&ticker_dir);

            let data = convert_historical(bars);

            let ticker_buy_indexes = &buy_indexes[ticker_index];
            let ticker_sell_indexes = &sell_indexes[ticker_index];
            let _ = buy_sell_chart(&ticker_dir, &data, ticker_buy_indexes, ticker_sell_indexes);

            let ticker_want_indexes = &want_indexes[ticker_index];
            let _ = want_chart(&ticker_dir, &data, ticker_want_indexes);

            /* let rsi_diff_values = rsi_values
                .iter()
                .zip(amount_rsi_values.iter())
                .map(|(decider, amount)| amount - decider)
                .collect();
            simple_chart(&ticker_dir, "rsi_diff", &rsi_diff_values); */

            let positioned_assets = &positions_by_ticker[ticker_index];
            let _ = assets_chart(
                &ticker_dir,
                &total_assets,
                &cash_graph,
                Some(positioned_assets),
            );
        }
    }

    *total_assets.last().unwrap()
}
