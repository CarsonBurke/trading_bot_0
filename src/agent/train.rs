use std::{collections::BTreeMap, fs};

use hashbrown::{HashMap, HashSet};
use ibapi::Client;
use uuid::Uuid;

use crate::{
    agent::Agent, charts::general::assets_chart, constants::{agent::{ KEEP_AGENTS_PER_GENERATION, TARGET_AGENT_COUNT, TARGET_GENERATIONS}, files::{TRAINING_PATH, WEIGHTS_PATH}}, data::historical::get_historical_data, strategies, types::{Account, MakeCharts}, utils::{convert_historical, get_rsi_values}
};

use super::{create::create_agents};

pub fn train_agents(client: &Client) {
    let mapped_historical = get_historical_data(client);

    let mut most_final_assets = 0.0;
    let mut best_of_gens = Vec::<Agent>::new();

    let mut agents = create_agents();

    for gen in 0..TARGET_GENERATIONS {
        let mut agents_vec = Vec::new();

        for (_, agent) in agents.iter() {
            let assets = strategies::basic::basic(
                &mapped_historical,
                agent,
                &mut Account::default(),
                None,
            );

            agents_vec.push((agent.id, assets));
        }

        agents_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        record_finances(&agents_vec, gen);

        agents_vec.truncate(KEEP_AGENTS_PER_GENERATION as usize);

        let agents_set: HashSet<Uuid> = HashSet::from_iter(agents_vec.iter().map(|a| a.0));

        agents.retain(|id, _| agents_set.contains(id));

        //

        let (gen_best_agent_id, gen_best_assets) = agents_vec[0];
        if gen_best_assets > most_final_assets {
                most_final_assets = gen_best_assets;
        }

        let best_gen_agent = agents.get(&gen_best_agent_id).unwrap();
        best_of_gens.push(best_gen_agent.clone());

        // duplicate agents

        let mut new_agents = Vec::new();

        // Has the potantial to create a few more agents than the target count, which seems fine
        while new_agents.len() + agents.len() < TARGET_AGENT_COUNT as usize {
            for (_, agent) in agents.iter() {
                let cloned_agent = agent.clone();

                new_agents.push(cloned_agent);
            }
        }

        while let Some(agent) = new_agents.pop() {
            agents.insert(agent.id, agent);
        }

        // Mutate agents

        for agent in agents.values_mut() {
            agent.weights.mutate();
        }

        //

        println!("completed generation {gen}");
        println!("Highest this gen: {gen_best_assets}");
    }

    println!("Completed training");

    let first_agent = best_of_gens.first().unwrap();
    println!("First agent =====");

    let first_assets = strategies::basic::basic(
        &mapped_historical,
        first_agent,
        &mut Account::default(),
        Some(MakeCharts { generation: 0}),
    );
    println!("Final assets: ${first_assets}");

    let last_agent = best_of_gens.last().unwrap();
    println!("Last agent =====");

    let final_assets = strategies::basic::basic(
        &mapped_historical,
        last_agent,
        &mut Account::default(),
        Some(MakeCharts { generation: TARGET_GENERATIONS - 1}),
    );
    println!("Final assets: ${final_assets}");

    let start = 10_000.0;
    println!("Profit: ${:.2} ({:.2}%)", final_assets - start, ((final_assets - start) / start) * 100.0);

    let diff = final_assets - first_assets;
    println!("Improvement from training of : ${diff:.2} ({:.2}%)", (final_assets - first_assets) / start * 100.0);

    record_weights(last_agent);
}

#[cfg(feature = "debug_training")]
fn record_finances(agents: &[(Uuid, f64)], gen: u32) {
    use crate::utils::create_folder_if_not_exists;

    let dir = format!("{TRAINING_PATH}/gens/{gen}");
    create_folder_if_not_exists(&dir);

    let agents_only_finances = agents.iter().map(|a| a.1).collect::<Vec<f64>>();

    fs::write(format!("{dir}/agents.txt"), format!("{agents_only_finances:?}")).unwrap();
}

pub fn record_weights(agent: &Agent) {
    let dir = WEIGHTS_PATH;
    fs::write(format!("{dir}/weights.txt"), agent.weights.to_string()).unwrap();

    let encoded = postcard::to_stdvec(&agent.weights).unwrap();
    fs::write(format!("{dir}/weights.bin"), encoded).unwrap();
}