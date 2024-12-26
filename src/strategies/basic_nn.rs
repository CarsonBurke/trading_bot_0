use core::f64;

use colored::Colorize;
use hashbrown::HashMap;
use rand::seq::index::sample;
use rust_neural_network::neural_network::{NeuralNetwork};

use crate::{
    charts::general::{assets_chart, buy_sell_chart, simple_chart, want_chart},
    constants::{
        self, agent::STARTING_CASH, files::TRAINING_PATH, neural_net::{BUY_INDEX, INDEX_STEP, MAX_STEPS, SAMPLE_INDEXES, SELL_INDEX, TICKER_SETS}, TICKERS
    },
    neural_net::create::Indicators,
    types::{Account, Data, MakeCharts, MappedHistorical, Position},
    utils::{convert_historical, create_folder_if_not_exists, ema, find_highest, get_rsi_values},
};

pub fn baisc_nn(
    ticker_sets: &[Vec<usize>],
    mapped_data: &MappedHistorical,
    neural_network: NeuralNetwork,
    // mapped_indicators: &Vec<Indicators>,
    inputs_count: usize,
    make_charts: Option<MakeCharts>,
) -> f64 {

    let mut all_assets = 0.;
    let mut all_min: f64 = f64::MAX;

    for (ticker_set_index, ticker_set) in ticker_sets.iter().enumerate() {
        let tickers_slice = ticker_set.as_slice();

        let indexes = mapped_data[0].len();
        let mut account = Account::default();

        for ticker_index in 0..mapped_data.len() {
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

        account.cash = STARTING_CASH;

        for index in 100..indexes {
            // Get and record some important data

            let mut total_positioned = 0.0;

            for ticker_index_bor in tickers_slice {
                let ticker_index = *ticker_index_bor;

                let bars = &mapped_data[ticker_index];
                let price = bars[index].close;

                let position = &account.positions[ticker_index];
                let positioned = position.value_with_price(price);

                positions_by_ticker[ticker_index].push(positioned);
                total_positioned += positioned;
            }

            let assets = account.cash + total_positioned;

            cash_graph.push(account.cash);
            total_assets.push(assets);

            for ticker_index_bor in tickers_slice {
                let ticker_index = *ticker_index_bor;

                let bars = &mapped_data[ticker_index];
                let price = bars[index].close;

                let position = &mut account.positions[ticker_index];

                // Assign inputs

                let mut inputs = vec![0.; inputs_count];

                inputs[0] = (account.cash / assets) as f32;
                inputs[1] = (position.value_with_price(price) / assets) as f32;
                inputs[2] = match position.quantity {
                    0. => 0.,
                    _ => ((price - position.avg_price) / position.avg_price) as f32,
                };

                for i in ((index.saturating_sub(INDEX_STEP))..index).rev() {
                    let diff_percent = (price - mapped_data[ticker_index][i].close) / price;
                    // println!("i: {}", index - i);
                    inputs[index - i + 3] = diff_percent as f32;
                }

                let min = index.saturating_sub(MAX_STEPS * INDEX_STEP);
                for (i, bari) in (min..(index - INDEX_STEP - 1))
                    .step_by(INDEX_STEP)
                    .rev()
                    .enumerate()
                {
                    let diff_percent = (price - mapped_data[ticker_index][bari].close) / price;
                    // println!("i: {} index: {} i2: {}", i, index, i / INDEX_STEP + 3);
                    inputs[i + 3] = diff_percent as f32;
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

                let activation_layers = neural_network.forward_propagate(inputs);
                let last_layer: Vec<f32> = activation_layers.last().unwrap().rows().into_iter().map(|x| x[0]).collect();

                // println!("activation layers: {activation_layers:?}");

                /* let (output_index, percent) = find_highest(last_layer);
                if *percent <= 0. {
                    continue;z
                } */

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

                let output = last_layer[BUY_INDEX] as f64;
                // println!("output: {}", output);
                let gross_want = (assets * output).min(assets / SAMPLE_INDEXES as f64) /* - assets * last_layer[SELL_INDEX] / 100. */ /* *percent *//* assets * percent / 10000. *//* assets * (percent / 1000.).min(0.2) */;
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

                // If we make a really dumb sell (selling lower than we avg bought) then criple the bot
                if price <= position.avg_price {
                    continue;
                }

                let quantity = sell / price;

                position.quantity -= quantity;
                account.cash += sell;

                sell_indexes[ticker_index].insert(index, (price, quantity));
            }
        }

        let final_assets = *total_assets.last().unwrap();
        all_assets += final_assets;

        all_min = all_min.min(final_assets);

        // If we're on our last set
        if ticker_set_index == TICKER_SETS - 1 {
            if final_assets > 1_000_000. {
                println!("{}", "total assets exceeds 1m".red());

                // make_charts = Some(MakeCharts {
                //     generation: 1_000_000,
                // });
            }

            if let Some(charts_config) = &make_charts {
                println!("Generating charts for gen: {}", charts_config.generation);

                let base_dir = format!("training/gens/{}", charts_config.generation);
                create_folder_if_not_exists(&base_dir);

                let _ = assets_chart(&base_dir, &total_assets, &cash_graph, None);

                for ticker_index_bor in tickers_slice {
                    let ticker_index = *ticker_index_bor;
                    let ticker = TICKERS[ticker_index].to_string();

                    let ticker_dir =
                        format!("{TRAINING_PATH}/gens/{}/{ticker}", charts_config.generation);
                    create_folder_if_not_exists(&ticker_dir);

                    let bars = &mapped_data[ticker_index];
                    let data = convert_historical(bars);

                    let ticker_buy_indexes = &buy_indexes[ticker_index];
                    let ticker_sell_indexes = &sell_indexes[ticker_index];
                    let _ =
                        buy_sell_chart(&ticker_dir, &data, ticker_buy_indexes, ticker_sell_indexes);

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
        }
    }

    // all_assets / TICKER_SETS as f64
    all_min
}
