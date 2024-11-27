pub const TICKERS: [&str; 1] = ["TSLA"/* , "AAPL", "MSFT" */];

pub mod rsi {
    pub const MIN_SELL: u32 = 70;
    pub const MAX_BUY: u32 = 30;
    pub const MOVING_AVG_DAYS: u32 = 14;
}