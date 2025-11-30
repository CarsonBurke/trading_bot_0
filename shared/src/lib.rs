pub mod paths {
    pub const DATA_PATH: &str = "../long_data";
    pub const WEIGHTS_PATH: &str = "../weights";
    pub const TRAINING_PATH: &str = "../training";
    pub const INFER_PATH: &str = "../infer";
}

pub mod constants {
    pub const TICKERS_COUNT: usize = 6;
    pub const ACTION_HISTORY_LEN: usize = 20;
    pub const PRICE_DELTAS_PER_TICKER: usize = 3400;

    pub const GLOBAL_STATIC_OBS: usize = 6;
    pub const PER_TICKER_STATIC_OBS: usize = 3 + (ACTION_HISTORY_LEN * 2);
    pub const STATIC_OBSERVATIONS: usize = GLOBAL_STATIC_OBS + (TICKERS_COUNT * PER_TICKER_STATIC_OBS);
    pub const OBSERVATION_SPACE: usize = (TICKERS_COUNT * PRICE_DELTAS_PER_TICKER) + STATIC_OBSERVATIONS;

    pub const STEPS_PER_EPISODE: usize = 10_000;
    pub const ACTION_THRESHOLD: f64 = 0.01;
    pub const COMMISSION_RATE: f64 = 0.005;
    pub const RETROACTIVE_BUY_REWARD: bool = false;
}

pub mod theme;
