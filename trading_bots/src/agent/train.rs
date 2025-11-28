use std::{
    collections::{BTreeMap, VecDeque},
    fs,
    sync::Arc,
};

use hashbrown::{HashMap, HashSet};
use uuid::Uuid;

use crate::{
    agent::Agent,
    charts::general::assets_chart,
    constants::{
        agent::{KEEP_AGENTS_PER_GENERATION, TARGET_AGENT_COUNT, TARGET_GENERATIONS},
        files::{TRAINING_PATH, WEIGHTS_PATH}, TICKERS,
    },
    data::historical::get_historical_data,
    strategies,
    types::{Account, MakeCharts, MappedHistorical},
    utils::{convert_historical, get_rsi_values},
};

use super::create::create_agents;

#[derive(Copy, Clone)]
pub enum AgentStrategy {
    Combined,
    RsiRebound,
    PriceRebound,
}

pub async fn train_agents(strategy: AgentStrategy) {
    let mapped_historical = Arc::new(get_historical_data(None));

    let mut most_final_assets = 0.0;
    let mut best_of_gens = Vec::<Agent>::new();

    let mut agents = create_agents();

    for gen in 0..TARGET_GENERATIONS {
        let mut handles = Vec::new();

        for (id, agent) in agents.iter_mut() {
            let id = *id;
            let cloned_agent = agent.clone();
            let cloned_historical = Arc::clone(&mapped_historical);

            let handle = tokio::task::spawn(async move {
                /* let assets = strategies::basic::basic(
                    &cloned_historical,
                    &cloned_agent,
                    &mut Account::default(),
                    None,
                ); */
                let assets = run_strategy(strategy, &cloned_agent, &cloned_historical, None);
                println!("assets: {:.2}", assets);

                (id, assets)
            });

            handles.push(handle)
        }

        let mut agents_vec = Vec::new();

        for handle in handles {
            let (agent_id, assets) = handle.await.unwrap();
            agents_vec.push((agent_id, assets));
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

        let best_gen_agent = agents.get(&gen_best_agent_id).unwrap().clone();
        best_of_gens.push(best_gen_agent.clone());

        // duplicate agents

        let mut new_agents = VecDeque::new();

        // Has the potantial to create a few more agents than the target count, which seems fine
        while new_agents.len() + agents.len() < TARGET_AGENT_COUNT as usize {
            for (_, agent) in agents.iter() {
                new_agents.push_front(agent.clone());
            }
        }

        while let Some(agent) = new_agents.pop_back() {
            agents.insert(agent.id, agent);
        }

        // Mutate agents

        for agent in agents.values_mut() {
            agent.weights.mutate();
        }

        // Add the best agent without mutations
        agents.insert(best_gen_agent.id, best_gen_agent);

        //

        println!("completed generation: {gen}");
        println!("Highest this gen: {gen_best_assets}");
    }

    println!("Completed training");

    let first_agent = best_of_gens.first().unwrap();
    println!("First agent =====");

    let first_assets = run_strategy(
        strategy,
        &first_agent,
        &mapped_historical,
        Some(MakeCharts { generation: 0 }),
    );
    println!("Final assets: ${first_assets}");

    let last_agent = best_of_gens.last().unwrap();
    println!("Last agent =====");

    let final_assets = run_strategy(
        strategy,
        &last_agent,
        &mapped_historical,
        Some(MakeCharts {
            generation: TARGET_GENERATIONS - 1,
        }),
    );
    println!("Final assets: ${final_assets}");

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

    record_weights(last_agent);
}

#[cfg(feature = "debug_training")]
fn record_finances(agents: &[(Uuid, f64)], gen: u32) {
    use crate::utils::create_folder_if_not_exists;

    let dir = format!("{TRAINING_PATH}/gens/{gen}");
    create_folder_if_not_exists(&dir);

    let agents_only_finances = agents.iter().map(|a| a.1).collect::<Vec<f64>>();

    fs::write(
        format!("{dir}/agents.txt"),
        format!("{agents_only_finances:?}"),
    )
    .unwrap();
}

pub fn record_weights(agent: &Agent) {
    let dir = WEIGHTS_PATH;
    fs::write(format!("{dir}/weights.txt"), agent.weights.to_string()).unwrap();

    let encoded = postcard::to_stdvec(&agent.weights).unwrap();
    fs::write(format!("{dir}/weights.bin"), encoded).unwrap();
}

/// Returns: the total final value of assets
pub fn run_strategy(
    strategy: AgentStrategy,
    agent: &Agent,
    mapped_data: &MappedHistorical,
    make_charts: Option<MakeCharts>,
) -> f64 {
    match strategy {
        AgentStrategy::PriceRebound => {
            return strategies::price_rebound::basic(
                &mapped_data,
                &agent,
                &mut Account::default(),
                make_charts,
            )
        }
        AgentStrategy::Combined => {
            return 1.;
            /* return strategies::combined::basic(
                &mapped_data,
                &agent,
                &mut Account::default(),
                make_charts,
            ) */
        }
        AgentStrategy::RsiRebound => {
            return strategies::rsi_rebound::basic(
                &mapped_data,
                &agent,
                &mut Account::default(),
                make_charts,
            )
        }
    }
}
