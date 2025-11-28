use rand::Rng;

use crate::burn::agent::base::Action;

#[derive(Debug, Clone, Copy)]
pub enum TradeAction {
    Buy,
    Sell,
    Hold,
}

impl TradeAction {
    pub fn new(id: u32) -> Self {
        match id {
            0 => TradeAction::Buy,
            1 => TradeAction::Sell,
            2 => TradeAction::Hold,
            _ => panic!("Invalid action"),
        }
    }
}

impl Action for TradeAction {
    fn random() -> Self {
        (rand::rng().random_range(0..Self::size()) as u32).into()
    }

    fn enumerate() -> Vec<Self> {
        vec![TradeAction::Buy, TradeAction::Sell, TradeAction::Hold]
    }
}

impl From<u32> for TradeAction {
    fn from(value: u32) -> Self {
        match value {
            0 => TradeAction::Buy,
            1 => TradeAction::Sell,
            2 => TradeAction::Hold,
            _ => panic!("Invalid action"),
        }
    }
}

impl From<TradeAction> for u32 {
    fn from(action: TradeAction) -> Self {
        match action {
            TradeAction::Buy => 0,
            TradeAction::Sell => 1,
            TradeAction::Hold => 2,
        }
    }
}
