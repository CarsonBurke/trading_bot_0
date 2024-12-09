use hashbrown::HashMap;
use ibapi::market_data::historical;

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

/// Key: Ticker, Value: Historical Bars
pub type MappedHistorical = HashMap<String, Vec<historical::Bar>>;

#[derive(Default, Debug)]
pub struct Position {
    pub quantity: f64,
    /// The average price that these positions were purchased at
    pub avg_price: f64,
}

impl Position {
    pub fn new(quantity: f64) -> Self {
        Self {
            quantity,
            avg_price: 0.,
        }
    }

    pub fn add(&mut self, price: f64, quantity: f64) {
        let sum = self.avg_price * self.quantity as f64;
        self.quantity += quantity;
        self.avg_price = (sum + price * quantity as f64) / (self.quantity as f64);
    }

    /// The total value of the position based on a provided Price Per Unit
    pub fn value(&self) -> f64 {
        self.avg_price * self.quantity as f64
    }

    pub fn value_with_price(&self, price: f64) -> f64 {
        price * self.quantity as f64
    }
}

pub struct MakeCharts {
    pub generation: u32,
}