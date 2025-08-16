use crate::burn::{action::TradeAction, agent::base::{Action, Environment, Snapshot}, obs_state::{ObservationState}};

#[derive(Debug)]
pub struct Env {
    step: usize,
    state: ObservationState,
    visualized: bool,
}

impl Environment for Env {
    type ActionType = TradeAction;
    type StateType = ObservationState;
    type RewardType = f64;
    // const MAX_STEPS: usize = usize::MAX;
    
    fn new(visualized: bool) -> Self {
        Env { 
            step: 0,
            state: ObservationState::new(),
            visualized,
        }
    }
    
    fn step(&mut self, action: Self::ActionType) -> Snapshot<Self> {
        // Implement step logic here
        
        let mut reward = 0.0;
        
        return Snapshot::new(self.state, 0.0, false)
    }
    
    fn reset(&mut self) -> Snapshot<Self> {
        // Implement reset logic here
        // 
        Snapshot::new(self.state, 0.0, true)
    }
    
    fn state(&self) -> ObservationState {
        ObservationState::new()
    }
}