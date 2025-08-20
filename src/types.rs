use hashbrown::HashMap;
use ibapi::market_data::historical;

/// A list of prices, where the last index is the most recent
pub type Data = Vec<f64>;

#[derive(Default, Debug)]
pub struct Account {
    pub cash: f64,
    pub total_assets: f64,
    pub positions: Positions,
}

impl Account {
    pub fn new(cash: f64, ticker_count: usize) -> Self {
        
        Self {
            cash,
            total_assets: cash,
            positions: vec![Position::default(); ticker_count],
        }
    }
    
    pub fn update_total(&mut self, prices: &[Vec<f64>], step: usize) {
        self.total_assets = self.position_values(prices, step).iter().sum::<f64>() + self.cash;
    }
    
    pub fn cash_cost_basis_ratio(&self) -> f64 {
        let cost_basis = self.cost_basis();
        if cost_basis == 0.0 {
            return 0.0;
        }
        
        self.total_assets / cost_basis
    }
    
    pub fn cost_basis(&self) -> f64 {
        self.positions.iter().map(|p| p.value()).sum::<f64>()
    }
    
    pub fn position_percents(&self, prices: &[Vec<f64>], step: usize) -> Vec<f64> {
        self.positions.iter().enumerate().map(|(index, p)| p.value_with_price(prices[index][step]) / self.total_assets).collect()
    }
    
    pub fn position_values(&self, prices: &[Vec<f64>], step: usize) -> Vec<f64> {
        self.positions.iter().enumerate().map(|(index, p)| p.value_with_price(prices[index][step])).collect()
    }
}

/// A list of positions 
pub type Positions = Vec<Position>;

/// Key: Ticker, Value: Historical Bars
pub type MappedHistorical = Vec<Vec<historical::Bar>>;

#[derive(Default, Debug, Clone, Copy)]
pub struct Position {
    pub quantity: f64,
    /// The average price that these positions were purchased at
    pub avg_price: f64,
}

impl Position {
    pub fn add(&mut self, price: f64, quantity: f64) {
        let sum = self.avg_price * self.quantity;
        self.quantity += quantity;
        self.avg_price = (sum + price * quantity) / (self.quantity);
    }

    /// The total value of the position based on the purchased avg price
    pub fn value(&self) -> f64 {
        self.avg_price * self.quantity
    }

    /// The total value of the position based on a provided Price Per Unit
    pub fn value_with_price(&self, price: f64) -> f64 {
        price * self.quantity
    }
    
    /// The percentage appreciation of the position based on a provided Price Per Unit
    pub fn appreciation(&self, price: f64) -> f64 {
        if self.quantity == 0.0 {
            return 0.0;
        }
        (price - self.avg_price) / self.avg_price
    }
}

pub struct MakeCharts {
    pub generation: u32,
}