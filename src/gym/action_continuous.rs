use rand::Rng;

use crate::gym::base::Action;

#[derive(Debug, Clone, Copy)]
pub struct ActionContinuous(pub f32);

impl ActionContinuous {
    pub fn new(id: u32) -> Self {
        match id {
            0 => ActionContinuous::Buy,
            1 => ActionContinuous::Sell,
            2 => ActionContinuous::Hold,
            _ => panic!("Invalid action"),
        }
    }
}

impl Action for ActionContinuous {
    fn random() -> Self {
        (rand::rng().random_range(0..Self::size()) as f32).into()
    }

    fn enumerate() -> Vec<Self> {
        vec![ActionContinuous::Buy, ActionContinuous::Sell, ActionContinuous::Hold]
    }
}

impl From<f32> for ActionContinuous {
    fn from(value: f32) -> Self {
        ActionContinuous(value)
    }
}

impl From<ActionContinuous> for f32 {
    fn from(action: ActionContinuous) -> Self {
        action.0
    }
}

impl From<u32> for ActionContinuous {
    fn from(value: u32) -> Self {
        ActionContinuous(value as f32)
    }
}

impl From<ActionContinuous> for u32 {
    fn from(action: ActionContinuous) -> Self {
        action.0 as u32
    }
}
