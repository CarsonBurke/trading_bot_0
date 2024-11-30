use std::{collections::BTreeMap, fs};

use hashbrown::{HashMap, HashSet};
use ibapi::Client;
use uuid::Uuid;

use crate::{
    agent::Agent, charts::general::assets_chart, constants::agent::{TARGET_AGENT_COUNT, TARGET_GENERATIONS}, data::historical::get_historical_data, strategies, types::{Account, MakeCharts}, utils::{convert_historical, get_rsi_values}
};

use super::{create::create_agents};

pub fn train_agents(client: &Client) {
    let mapped_historical = get_historical_data(client);

    let mut best_of_gens = Vec::<Agent>::new();

    let mut agents = create_agents();

    for gen in 0..TARGET_GENERATIONS {
        let mut assets_by_agent = BTreeMap::new();

        for (_, agent) in agents.iter() {
            let assets = strategies::basic::basic(
                &mapped_historical,
                agent,
                &mut Account::default(),
                None,
            );

            assets_by_agent.insert(agent.id, assets);
        }

        // for i in (TARGET_AGENT_COUNT / 2)..TARGET_AGENT_COUNT {
        //     assets_by_agent.iter().rev()
        // }

        // for (index, (id, agent)) in assets_by_agent.iter().rev().enumerate() {

        // }

        let mut agents_vec = assets_by_agent
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect::<Vec<(Uuid, f64)>>();
        // agents_vec.sort_by(|a, b| b.1.partial_cmp(&a .1).unwrap());

        // Kill all but the top 10 agents
        agents_vec.truncate((TARGET_AGENT_COUNT / 10) as usize);
        let agents_set: HashSet<Uuid> = HashSet::from_iter(agents_vec.iter().map(|a| a.0));

        agents.retain(|id, _| agents_set.contains(id));

        // duplicate agents

        let mut new_agents = Vec::new();

        // Has the potantial to create a few more agents than the target count, which seems fine
        while new_agents.len() + agents.len() < TARGET_AGENT_COUNT as usize {
            for (_, agent) in agents.iter() {
                let mut cloned_agent = agent.clone();
                cloned_agent.id = Uuid::default();

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

        println!("total agents count: {}", agents.len());

        // assign agents to agents vec

        let (best_agent_id, best_agent_total_assets) = agents_vec.first().unwrap();
        let best_agent = agents.get(best_agent_id).unwrap();
        let best_agent_assets = assets_by_agent.get(best_agent_id).unwrap();

        best_of_gens.push(best_agent.clone());

        println!("completed generation {gen}");
    }

    let first_agent = best_of_gens.first().unwrap();
    println!("First agent =====");

    let assets = strategies::basic::basic(
        &mapped_historical,
        first_agent,
        &mut Account::default(),
        Some(MakeCharts { generation: 0}),
    );
    println!("Final assets {assets}");

    let last_agent = best_of_gens.last().unwrap();
    println!("Last agent =====");

    let assets = strategies::basic::basic(
        &mapped_historical,
        last_agent,
        &mut Account::default(),
        Some(MakeCharts { generation: TARGET_GENERATIONS - 1}),
    );
    println!("Final assets {assets}");
}

pub fn record_weights() {}
