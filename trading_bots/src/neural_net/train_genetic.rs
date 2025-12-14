use std::{collections::VecDeque, sync::Arc};

use colored::Colorize;
use hashbrown::{HashMap, HashSet};
use ibapi::{market_data::historical, Client};

use rand::{seq::{index::sample, SliceRandom}, Rng};
use rust_neural_network::neural_network::NeuralNetwork;

use crate::{
    charts::simple_chart, constants::{
        agent::{KEEP_AGENTS_PER_GENERATION, TARGET_AGENT_COUNT, TARGET_GENERATIONS}, files::TRAINING_PATH, neural_net::{self, INDEX_STEP, MAX_STEPS, SAMPLE_INDEXES, TICKER_SETS}, TICKERS
    }, data::historical::get_historical_data, neural_net::{create::create_mapped_indicators, Replay}, strategies::basic_nn::basic_nn, types::{Account, MakeCharts}, utils::{create_folder_if_not_exists, get_mapped_price_deltas}
};

use super::create::{create_networks, Indicator, Indicators};

pub async fn train_networks_genetic() {
    let time = std::time::Instant::now();

    let mapped_historical = Arc::new(get_historical_data(None));
    let mapped_indicators = create_mapped_indicators(&mapped_historical);

    let mapped_diffs = Arc::new(get_mapped_price_deltas(&mapped_historical));

    let mut most_final_assets = 0.0;
    let mut best_of_gens = Vec::<NeuralNetwork>::new();

    let mut inputs = vec![
        // Percent of assets that are in cash
        0.,
        // Percent of total assets in the position
        0.,
        // Percent difference between current price and average purchase price (or 0 if we have no money in position)
        0.,
    ];

    let indicators = &mapped_indicators[0];
    for _ in 0..indicators.len() {
        inputs.push(0.);
    }

    for _ in 0..(MAX_STEPS+INDEX_STEP) {
        inputs.push(0.);
    }

    let input_count = inputs.len();

    println!("Inputs {}", input_count);
    // let inputs_arc = Arc::new(inputs.to_vec());

    let output_count = 4;

    // for (index, data) in mapped_historical.iter().enumerate() {
    //     println!("check {index} {}", data.len());
    // }

    let mut neural_nets = create_networks(input_count, output_count);
    let mut rng = rand::thread_rng();

    for gen in 0..TARGET_GENERATIONS {
        let mut neural_net_ids = Vec::new();
        let mut handles = Vec::new();

        let mut replays: Vec<Replay> = Vec::new();

        let tickers_set = generate_tickers_set(&mut rng);

        // let ticker_data = mapped_historical.choose_multiple(&mut rng, 10);

        for (_, neural_net) in neural_nets.iter_mut() {
            let id = neural_net.id;
            let neural_net = neural_net.clone();
            // let cloned_inputs = inputs.to_vec();// Arc::clone(&inputs_arc);

            let cloned_historical = Arc::clone(&mapped_historical);
            let cloned_diffs = Arc::clone(&mapped_diffs);
            let cloned_tickers_set = tickers_set.to_vec();

            // let indexes = ticker_indexes.clone();
            
            // let cloned_historical = ticker_indexes.iter().map(|index| mapped_historical[*index].clone()).collect::<Vec<Vec<historical::Bar>>>();
            // let cloned_indicators = mapped_indicators.clone();
            // println!("cloned historical len: {}", cloned_historical.len());
            let handle = tokio::task::spawn(async move {
                let assets = basic_nn(
                    &cloned_tickers_set,
                    &cloned_historical,
                    &neural_net,
                    &cloned_diffs,
                    // &cloned_indicators,
                    input_count,
                    None,
                );
                println!("assets: {:.2}", assets);
                // neural_net_ids.push((neural_net.id, assets));

                (id, assets)
            });

            handles.push(handle)
        }

        for handle in handles {
            let (net_id, assets) = handle.await.unwrap();
            neural_net_ids.push((net_id, assets));
        }

        neural_net_ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        record_finances(&neural_net_ids, gen);

        neural_net_ids.truncate(KEEP_AGENTS_PER_GENERATION as usize);

        let id_set: HashSet<u32> = HashSet::from_iter(neural_net_ids.iter().map(|a| a.0));
        neural_nets.retain(|id, _| id_set.contains(id));

        //

        let (best_net_id, gen_best_assets) = neural_net_ids[0];
        if gen_best_assets > most_final_assets {
            most_final_assets = gen_best_assets;
        }

        let best_gen_net = neural_nets.get(&best_net_id).unwrap().clone();
        best_of_gens.push(best_gen_net.clone());

        // duplicate neural nets

        let mut new_nets = VecDeque::new();

        while new_nets.len() + neural_nets.len() < TARGET_AGENT_COUNT as usize {
            for (_, neural_net) in neural_nets.iter() {
                new_nets.push_front(neural_net.clone());
            }
        }

        while let Some(net) = new_nets.pop_back() {
            neural_nets.insert(net.id, net);
        }

        // mutate

        for net in neural_nets.values_mut() {
            net.mutate();
        }

        neural_nets.insert(best_gen_net.id, best_gen_net);

        println!(
            "Completed generation: {gen} with networks: {}",
            neural_nets.len()
        );
        println!("{} {gen_best_assets:.2}", "Highest this gen:".bright_green());
    }

    println!("Completed training");

    let cloned_historical = Arc::clone(&mapped_historical);

    let first_net = best_of_gens.first().unwrap();

    let tickers_set = generate_tickers_set(&mut rng);

    let first_assets = basic_nn(
        &tickers_set,
        &cloned_historical,
        first_net,
        // &mapped_indicators,
        // inputs.to_vec(),
        &mapped_diffs,
        input_count,
        Some(MakeCharts { generation: 0 }),
    );
    println!("Gen 1 final assets: {first_assets:.2}");

    let last_net = best_of_gens.last().unwrap();
    let final_assets = basic_nn(
        &tickers_set,
        &cloned_historical,
        last_net,
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

    let diff = final_assets - first_assets;
    println!(
        "Improvement from training of : ${diff:.2} ({:.2}%)",
        (final_assets - first_assets) / start * 100.0
    );
    
    chart_indicators(&mapped_indicators);

    last_net.write_to_file();

    println!("Completed training in {} seconds", time.elapsed().as_secs());
}

#[cfg(feature = "debug_training")]
fn record_finances(neural_net_ids: &[(u32, f64)], gen: u32) {
    use std::fs;

    use crate::{constants::files::TRAINING_PATH, utils::create_folder_if_not_exists};

    let dir = format!("{TRAINING_PATH}/gens/{gen}");
    create_folder_if_not_exists(&dir);

    let agents_only_finances = neural_net_ids.iter().map(|a| a.1).collect::<Vec<f64>>();

    fs::write(
        format!("{dir}/agents.txt"),
        format!("{agents_only_finances:.2?}"),
    )
    .unwrap();
}

pub fn chart_indicators(mapped_indicators: &Vec<Indicators>) {
    for (ticker_index, indicators) in mapped_indicators.iter().enumerate() {
        let ticker = TICKERS[ticker_index];
        let dir = format!("{TRAINING_PATH}/indicators/{ticker}");
        create_folder_if_not_exists(&dir);

        simple_chart(&dir, "EmaDiff100", &indicators[Indicator::EMADiff100]).unwrap();
        simple_chart(&dir, "EmaDiff1000", &indicators[Indicator::EMADiff1000]).unwrap();
        simple_chart(&dir, "Rsi100", &indicators[Indicator::RSI100]).unwrap();

        simple_chart(&dir, "StochasticOscillator", &indicators[Indicator::StochasticOscillator]).unwrap();
        simple_chart(&dir, "MacdDiff", &indicators[Indicator::MACDDiff]).unwrap();
    }
}

pub fn generate_tickers_set(rng: &mut impl rand::Rng) -> Vec<Vec<usize>> {
    let mut tickers_set = Vec::new();
    
    for _ in 0..TICKER_SETS {
        let indexes: Vec<usize> = match sample(rng, TICKERS.len() - 1, SAMPLE_INDEXES) {
            rand::seq::index::IndexVec::U64(v) => v.into_iter().map(|i| i as usize).collect(),
            rand::seq::index::IndexVec::U32(v) => v.into_iter().map(|i| i as usize).collect(),
        };

        tickers_set.push(indexes.clone());
    }


    tickers_set
}