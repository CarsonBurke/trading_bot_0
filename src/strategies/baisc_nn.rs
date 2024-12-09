use hashbrown::HashMap;
use rust_neural_network::neural_network::{Input, NeuralNetwork};

use crate::{
    charts::general::{assets_chart, buy_sell_chart, simple_chart, want_chart},
    constants::{self, files::TRAINING_PATH, TICKERS},
    neural_net::create::Indicators,
    types::{Account, Data, MakeCharts, MappedHistorical, Position},
    utils::{convert_historical, create_folder_if_not_exists, ema, find_highest, get_rsi_values},
};

pub fn baisc_nn(
    mapped_data: &MappedHistorical,
    account: &mut Account,
    neural_network: &mut NeuralNetwork,
    mapped_indicators: &HashMap<String, Indicators>,
    inputs: &mut [Input],
    make_charts: Option<MakeCharts>,
) -> f64 {
    let indexes = mapped_data.get(TICKERS[0]).unwrap().len();

    for ticker in TICKERS {
        account
            .positions
            .insert(ticker.to_string(), Position::default());
    }

    let mut positions_by_ticker: HashMap<String, Vec<f64>> = HashMap::new();

    for ticker in mapped_data.keys() {
        positions_by_ticker.insert(ticker.to_string(), Vec::new());
    }

    let mut cash_graph = Vec::new();
    let mut total_assets = Vec::new();

    let mut buy_indexes = HashMap::new();
    let mut sell_indexes = HashMap::new();

    for (ticker, _) in mapped_data.iter() {
        buy_indexes.insert(ticker.to_string(), HashMap::new());
        sell_indexes.insert(ticker.to_string(), HashMap::new());
    }

    let mut want_indexes = HashMap::new();

    for (ticker, _) in mapped_data.iter() {
        want_indexes.insert(ticker.to_string(), HashMap::new());
    }

    account.cash = 10_000.;

    for index in 100..indexes {
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

        for (ticker, bars) in mapped_data.iter() {
            let price = bars[index].close;

            let position = account.positions.get_mut(ticker).unwrap();

            // Assign inputs

            inputs[0].values[0] = account.cash / assets;
            inputs[1].values[0] = position.value_with_price(price) / assets;
            inputs[2].values[0] = match position.quantity {
                0. => 0.,
                _ => (price - position.avg_price) / position.avg_price,
            };

            let indicators = mapped_indicators.get(ticker).unwrap();
            for (key, val) in indicators.iter() {
                inputs[key as usize + 3].values[0] = val[index];
            }

            // Forward propagate

            neural_network.forward_propagate(inputs);

            let last_layer = neural_network.activation_layers.last().unwrap();

            let (output_index, percent) = find_highest(last_layer);
            if *percent <= 0. {
                continue;
            }
            // println!("index: {}, value: {}", index, percent);
            // let values = inputs.iter().map(|input| input.values[0]).collect::<Vec<f64>>();
            // println!("inputs: {values:?}");

            if output_index as u32 == constants::neural_net::CHANGE_INDEX {
                // Get the want from the determined percent, at a maximum of 10%
                let gross_want = assets * (percent / 1000.).min(0.2);
                let current = match position.quantity {
                    0. => 0.,
                    _ => position.value_with_price(price),
                };

                want_indexes
                    .get_mut(ticker)
                    .unwrap()
                    .insert(index, gross_want);

                // If we want more than we have, try to buy
                if gross_want > current {
                    let buy = (gross_want - current).min(account.cash);
                    if buy == 0. {
                        continue;
                    }
                    
                    let quantity = buy / price;

                    position.add(price, quantity);
                    account.cash -= buy;

                    buy_indexes
                        .get_mut(ticker)
                        .unwrap()
                        .insert(index, (price, quantity));
                    continue;
                }

                // Otherwise we want less than we have, try to sell

                let sell = ((gross_want - current).abs()).min(current);
                if sell == 0. {
                    continue;
                }

                let quantity = sell / price;

                position.quantity -= quantity;
                account.cash += sell;

                sell_indexes
                    .get_mut(ticker)
                    .unwrap()
                    .insert(index, (price, quantity));
            }
        }
    }

    if let Some(charts_config) = make_charts {
        println!("Generating charts for gen: {}", charts_config.generation);

        let base_dir = format!("training/gens/{}", charts_config.generation);
        create_folder_if_not_exists(&base_dir);

        let _ = assets_chart(&base_dir, &total_assets, &cash_graph, None);

        for (ticker, bars) in mapped_data.iter() {
            let ticker_dir = format!("{TRAINING_PATH}/gens/{}/{ticker}", charts_config.generation);
            create_folder_if_not_exists(&ticker_dir);

            let data = convert_historical(bars);

            /* candle_chart(&ticker_dir, bars); */

            let ticker_buy_indexes = buy_indexes.get(ticker).unwrap();
            let ticker_sell_indexes = sell_indexes.get(ticker).unwrap();
            let _ = buy_sell_chart(&ticker_dir, &data, ticker_buy_indexes, ticker_sell_indexes);

            let ticker_want_indexes = want_indexes.get(ticker).unwrap();
            let _ = want_chart(&ticker_dir, &data, ticker_want_indexes);

            /* let rsi_diff_values = rsi_values
                .iter()
                .zip(amount_rsi_values.iter())
                .map(|(decider, amount)| amount - decider)
                .collect();
            simple_chart(&ticker_dir, "rsi_diff", &rsi_diff_values); */

            let positioned_assets = positions_by_ticker.get(ticker).unwrap();
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
