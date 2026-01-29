use hashbrown::HashMap;
use uuid::Uuid;

use crate::constants;

use super::Agent;

pub fn create_agents() -> HashMap<Uuid, Agent> {
    let mut agents = HashMap::new();

    for _i in 0..constants::agent::TARGET_AGENT_COUNT {
        let agent = Agent::default();
        agents.insert(agent.id, agent);
    }

    agents
}