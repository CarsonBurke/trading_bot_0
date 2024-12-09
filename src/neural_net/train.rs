use std::collections::VecDeque;

use hashbrown::HashSet;
use ibapi::Client;

use rust_neural_network::neural_network::{Input, InputName, NeuralNetwork, Output, OutputName};

use crate::{
    constants::{
        agent::{KEEP_AGENTS_PER_GENERATION, TARGET_AGENT_COUNT, TARGET_GENERATIONS},
        neural_net, TICKERS,
    },
    data::historical::get_historical_data,
    neural_net::create::create_mapped_indicators,
    strategies::baisc_nn::baisc_nn,
    types::{Account, MakeCharts},
};

use super::create::create_networks;

pub async fn train_networks(client: &Client) {
    let mapped_historical = get_historical_data(client);
    let mapped_indicators = create_mapped_indicators(&mapped_historical);

    let mut most_final_assets = 0.0;
    let mut best_of_gens = Vec::<NeuralNetwork>::new();

    let mut inputs = vec![
        // Percent of assets that are in cash
        Input {
            name: InputName::X,
            values: vec![1.],
            weight_ids: vec![0],
        },
        // Percent of total assets in the position
        Input {
            name: InputName::X,
            values: vec![1.],
            weight_ids: vec![1],
        },
        // Percent difference between current price and average purchase price (or 0 if we have no money in position)
        Input {
            name: InputName::X,
            values: vec![1.],
            weight_ids: vec![2],
        },
    ];

    let indicators = mapped_indicators.get(TICKERS[0]).unwrap();
    for index in 0..indicators.len() {
        inputs.push(Input {
            name: InputName::X,
            values: vec![1.],
            weight_ids: vec![(index + inputs.len()) as u32],
        });
    }

    let outputs = vec![
        Output {
            name: OutputName::Result,
        },
        Output {
            name: OutputName::Result,
        },
        Output {
            name: OutputName::Result,
        },
    ];

    let mut neural_nets = create_networks(&inputs, outputs.len());

    for gen in 0..TARGET_GENERATIONS {
        let mut neural_net_ids = Vec::new();
        let mut handles = Vec::new();

        for (_, neural_net) in neural_nets.iter_mut() {
            // let mut neural_net = neural_net.clone();

            let cloned_inputs = inputs.clone();

            let cloned_historical = mapped_historical.clone();
            let cloned_indicators = mapped_indicators.clone();

            let handle = tokio::task::spawn(async move {
                let assets = baisc_nn(
                    &cloned_historical,
                    &mut Account::default(),
                    *neural_net,
                    &cloned_indicators,
                    cloned_inputs,
                    None,
                );
                println!("assets: {:.2}", assets);
                // neural_net_ids.push((neural_net.id, assets));

                (neural_net.id, assets)
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
        println!("Highest this gen: {gen_best_assets:.2}");
    }

    println!("Completed training");

    let first_net = best_of_gens.first_mut().unwrap();

    let first_assets = baisc_nn(
        &mapped_historical,
        &mut Account::default(),
        *first_net,
        &mapped_indicators,
        inputs.clone(),
        Some(MakeCharts { generation: 0 }),
    );
    println!("Gen 1 final assets: {first_assets:.2}");

    let last_net = best_of_gens.last_mut().unwrap();

    let final_assets = baisc_nn(
        &mapped_historical,
        &mut Account::default(),
        *last_net,
        &mapped_indicators,
        inputs,
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

    last_net.write_to_file();
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
