pub mod paths {
    pub const DATA_PATH: &str = "../long_data";
    pub const WEIGHTS_PATH: &str = "../weights";
    pub const TRAINING_PATH: &str = "../training";
    pub const INFER_PATH: &str = "../infer";
}

pub mod constants {
    pub const TICKERS_COUNT: usize = 6;
    pub const ACTION_COUNT: usize = TICKERS_COUNT + 1; // +1 for cash weight
    pub const ACTION_HISTORY_LEN: usize = 20;
    pub const PRICE_DELTAS_PER_TICKER: usize = 3400;

    // Global: step_progress, cash_percent, pnl, drawdown, commissions, last_reward, last_fill_ratio
    pub const GLOBAL_STATIC_OBS: usize = 7;
    // Per-ticker: position_pct, unrealized_pnl, momentum, trade_activity_ema, steps_since_trade, position_age, target_weight, action_history
    pub const PER_TICKER_STATIC_OBS: usize = 7 + ACTION_HISTORY_LEN;
    pub const STATIC_OBSERVATIONS: usize = GLOBAL_STATIC_OBS + (TICKERS_COUNT * PER_TICKER_STATIC_OBS);
    pub const OBSERVATION_SPACE: usize = (TICKERS_COUNT * PRICE_DELTAS_PER_TICKER) + STATIC_OBSERVATIONS;

    pub const STEPS_PER_EPISODE: usize = 10_000;
    pub const ACTION_THRESHOLD: f64 = 0.001;
    pub const COMMISSION_RATE: f64 = 0.005;
    pub const RETROACTIVE_BUY_REWARD: bool = false;
}

pub mod theme;
