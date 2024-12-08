pub const TICKERS: [&str; 5] = ["TSLA", "AAPL", "MSFT", "AMD", "INTC"/* , "NVDA" */];

pub mod rsi {
    pub const MIDDLE: f64 = 50.;
}

pub mod files {
    pub const DATA_PATH: &str = "data/";
    pub const WEIGHTS_PATH: &str = "weights/";
    pub const TRAINING_PATH: &str = "training/";
}

pub mod agent {
    pub const LEARNING_RATE: f64 = 0.003/* 0.003 */;
    /// How many agents we want in training at each training step
    pub const TARGET_AGENT_COUNT: u32 = 100;
    pub const KEEP_AGENTS_PER_GENERATION: u32 = 20;
    /// How many generations to run to train the agents
    pub const TARGET_GENERATIONS: u32 = 100;
    pub const MAX_WEIGHT: f64 = 1.0;
    pub const MIN_WEIGHT: f64 = 0.0;
}

pub mod neural_net {
    pub const BUY_INDEX: usize = 0;
    pub const SELL_INDEX: usize = 1;
    /* pub const CHANGE_INDEX: u32 = 0; */
    pub const HOLD_INDEX: u32 = 2;
    pub const INDEX_STEP: usize = 10;
    pub const MAX_STEPS: usize = 100;
}

/// A ticker may have no more than this percept in total assets
pub const MAX_VALUE_PER_TICKER: f64 = 0.1;