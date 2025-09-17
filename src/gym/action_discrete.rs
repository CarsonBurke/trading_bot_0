use rand::Rng;

use crate::gym::base::Action;

#[derive(Debug, Clone, Copy)]
pub enum ActionDiscrete {
    Buy,
    Sell,
    Hold,
}

impl ActionDiscrete {
    pub fn new(id: u32) -> Self {
        match id {
            0 => ActionDiscrete::Buy,
            1 => ActionDiscrete::Sell,
            2 => ActionDiscrete::Hold,
            _ => panic!("Invalid action"),
        }
    }
}

impl Action for ActionDiscrete {

    fn size() -> usize {
        vec![ActionDiscrete::Buy, ActionDiscrete::Sell, ActionDiscrete::Hold].len()
    }
}

impl From<u32> for ActionDiscrete {
    fn from(value: u32) -> Self {
        match value {
            0 => ActionDiscrete::Buy,
            1 => ActionDiscrete::Sell,
            2 => ActionDiscrete::Hold,
            _ => panic!("Invalid action"),
        }
    }
}

impl From<ActionDiscrete> for u32 {
    fn from(action: ActionDiscrete) -> Self {
        match action {
            ActionDiscrete::Buy => 0,
            ActionDiscrete::Sell => 1,
            ActionDiscrete::Hold => 2,
        }
    }
}

impl From<f32> for ActionDiscrete {
    fn from(value: f32) -> Self {
        match value {
            0.0 => ActionDiscrete::Buy,
            1.0 => ActionDiscrete::Sell,
            2.0 => ActionDiscrete::Hold,
            _ => panic!("Invalid action"),
        }
    }
}

impl From<ActionDiscrete> for f32 {
    fn from(action: ActionDiscrete) -> Self {
        match action {
            ActionDiscrete::Buy => 0.0,
            ActionDiscrete::Sell => 1.0,
            ActionDiscrete::Hold => 2.0,
        }
    }
}
