pub const TICKERS: [&str; 1] = ["TSLA"/* , "AAPL", "MSFT" */];

pub mod rsi {
    pub const MIN_SELL: f64 = 70.;
    pub const MAX_BUY: f64 = 40.;
    pub const MOVING_AVG_DAYS: u32 = 14;
}