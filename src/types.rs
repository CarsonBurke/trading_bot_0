use std::collections::HashMap;

/// A list of prices, where the last index is the most recent
pub type Data = Vec<f64>;

pub type SparseData = Vec<Option<f64>>;

/// A list of tuples of (time, price), where the last index is the most recent
pub type TimedData = Vec<(f64, f64)>;

#[derive(Default, Debug)]
pub struct Account {
    pub cash: f64,
    pub positions: Positions,
}

/// A list of positions 
pub type Positions = HashMap<String, Position>;

#[derive(Default, Debug)]
pub struct Position {
    pub quantity: u32,
}

impl Position {
    pub fn new(quantity: u32) -> Self {
        Self {
            quantity,
        }
    }

    /// The total value of the position based on a provided Price Per Unit
    pub fn value(&self, ppu: f64) -> f64 {
        ppu * self.quantity as f64
    }
}