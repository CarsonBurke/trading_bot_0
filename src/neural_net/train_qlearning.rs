use std::{collections::VecDeque, sync::Arc};

use colored::Colorize;
use hashbrown::{HashMap, HashSet};
use ibapi::{market_data::historical, Client};

use rand::{
    seq::{index::sample, SliceRandom},
    Rng,
};
use rust_neural_network::neural_network::NeuralNetwork;

use crate::{
    charts::general::simple_chart,
    constants::{
        agent::{KEEP_AGENTS_PER_GENERATION, TARGET_AGENT_COUNT, TARGET_GENERATIONS},
        files::TRAINING_PATH,
        neural_net::{self, INDEX_STEP, MAX_STEPS, SAMPLE_INDEXES, TICKER_SETS},
        TICKERS,
    },
    data::historical::get_historical_data,
    neural_net::{
        create::{create_mapped_diffs, create_mapped_indicators},
        train_genetic::{chart_indicators, generate_tickers_set},
        Replay,
    },
    strategies::basic_nn::basic_nn,
    types::{Account, MakeCharts},
    utils::create_folder_if_not_exists,
};

use super::create::{create_networks, Indicator, Indicators};

pub fn train_networks_qlearning() {
    let time = std::time::Instant::now();

    let mapped_historical = Arc::new(get_historical_data());
    let mapped_indicators = create_mapped_indicators(&mapped_historical);

    let mapped_diffs = Arc::new(create_mapped_diffs(&mapped_historical));

    let mut most_final_assets = 0.0;
    let mut best_of_gens = Vec::<NeuralNetwork>::new();

    let mut inputs = vec![
        // Percent of assets that are in cash
        0., // Percent of total assets in the position
        0.,
        // Percent difference between current price and average purchase price (or 0 if we have no money in position)
        0.,
    ];

    let indicators = &mapped_indicators[0];
    for _ in 0..indicators.len() {
        inputs.push(0.);
    }

    for _ in 0..(MAX_STEPS + INDEX_STEP) {
        inputs.push(0.);
    }

    let input_count = inputs.len();

    println!("Inputs {}", input_count);
    // let inputs_arc = Arc::new(inputs.to_vec());

    let output_count = 4;

    let bias = 0.0;
    let learning_rate = 0.1;
    let layers = [input_count, 10, 10, output_count];
    println!(
        "Creating neural networks with bias {bias} learning rate {learning_rate} layers {layers:?}"
    );

    let neural_net = NeuralNetwork::new(bias, learning_rate, layers.to_vec());

    let mut rng = rand::thread_rng();

    // run
    // run this multiple times with the same network, over and over

    let mut replays: Vec<Replay> = Vec::new();

    for gen in 0..TARGET_GENERATIONS {
        let tickers_set = generate_tickers_set(&mut rng);

        let assets = basic_nn(
            &tickers_set,
            &mapped_historical,
            &neural_net,
            // &mapped_indicators,
            // inputs.to_vec(),
            &mapped_diffs,
            input_count,
            Some(MakeCharts { generation: 0 }),
        );
        
        // Periodically run a replay
        // Unclear exactly how to integrate this

        

        //

        println!("Completed generation: {gen} with assets: {}", assets);
    }

    // Once more for charts and stuff

    let tickers_set = generate_tickers_set(&mut rng);

    let final_assets = basic_nn(
        &tickers_set,
        &mapped_historical,
        &neural_net,
        &mapped_diffs,
        // &mapped_indicators,
        // inputs.to_vec(),
        input_count,
        Some(MakeCharts {
            generation: TARGET_GENERATIONS - 1,
        }),
    );

    println!("Final assets: {final_assets:.2}");

    let start = 10_000.0;
    println!(
        "Profit: ${:.2} ({:.2}%)",
        final_assets - start,
        ((final_assets - start) / start) * 100.0
    );

    /* let diff = final_assets - first_assets;
    println!(
        "Improvement from training of : ${diff:.2} ({:.2}%)",
        (final_assets - first_assets) / start * 100.0
    ); */

    chart_indicators(&mapped_indicators);

    neural_net.write_to_file();

    println!("Completed training in {} seconds", time.elapsed().as_secs());
}
