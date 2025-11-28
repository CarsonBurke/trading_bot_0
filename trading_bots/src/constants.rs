/* pub const TICKERS: [&str; 32] = [
    // iShares S&P500 ETF
    "SPY", "TSLA", "AAPL", "MSFT", "AMD", "INTC", "NVDA", "IBM", "GOOG", "META", "AMZN", "NFLX",
    "TSM", "QCOM", "ORCL", "PFE", "SOUN", "SMCI", "LLY", "CE", "GCT", "UNH", "LODE", "KULR",
    "LUNR", "NABL", "DVN", "BASE", "NET", "CRWD", "JD", "NEXT",
]; */
/* pub const TICKERS: [&str; 1] = [
    // iShares S&P500 ETF
    "NVDA",
]; */
pub const TICKERS: [&str; 7] = [
    // iShares S&P500 ETF
    "SPY", "TSLA", "AAPL", "MSFT", "AMD", "INTC", "NVDA",
];
// pub const TICKERS: [&str; 1] = [
//     // iShares S&P500 ETF
//     "NVDA",
// ];

pub mod rsi {
    pub const MIDDLE: f64 = 50.;
}

pub mod api {
    pub const CONNECTION_URL: &str = "127.0.0.1:4002";
}

pub mod files {
    pub use shared::paths::*;
}

pub mod agent {
    pub const LEARNING_RATE: f64 = 0.01/* 0.003 */;
    /// How many agents we want in training at each training step
    pub const TARGET_AGENT_COUNT: u32 = 150;
    pub const KEEP_AGENTS_PER_GENERATION: u32 = 50;
    /// How many generations to run to train the agents
    pub const TARGET_GENERATIONS: u32 = 1000;
    pub const MAX_WEIGHT: f64 = 1.0;
    pub const MIN_WEIGHT: f64 = 0.0;
    pub const STARTING_CASH: f64 = 10_000.;
}

pub mod neural_net {
    pub const CHANGE_INDEX: usize = 0;
    pub const BUY_INDEX: usize = 0;
    pub const SELL_INDEX: usize = 1;
    /* pub const CHANGE_INDEX: u32 = 0; */
    pub const HOLD_INDEX: u32 = 2;
    pub const INDEX_STEP: usize = 3;
    pub const MAX_STEPS: usize = 200;
    pub const SAMPLE_INDEXES: usize = 10;
    pub const TICKER_SETS: usize = 5;
}

/// A ticker may have no more than this percept in total assets
pub const MAX_VALUE_PER_TICKER: f64 = 0.1;
