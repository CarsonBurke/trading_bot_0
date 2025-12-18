pub mod paths {
    pub const DATA_PATH: &str = "../long_data";
    pub const WEIGHTS_PATH: &str = "../weights";
    pub const TRAINING_PATH: &str = "../training";
    pub const INFER_PATH: &str = "../infer";
}

pub mod constants {
    pub const TICKERS_COUNT: usize = 1 ;
    pub const ACTION_COUNT: usize = TICKERS_COUNT + 1; // +1 for cash weight
    pub const ACTION_HISTORY_LEN: usize = 0;
    pub const PRICE_DELTAS_PER_TICKER: usize = 3400;
    pub const REWARD_RANGE: f64 = 100.0;

    // Global (7): step_progress, cash_percent, pnl, drawdown, commissions, last_reward, last_fill_ratio
    // Macro (14): gdp_growth, unemployment, jobs_growth, cpi_yoy, core_cpi_yoy, fed_funds, treasury_10y, yield_spread, consumer_sentiment, initial_claims
    //             + steps_to_jobs, steps_to_cpi, steps_to_fomc, steps_to_gdp
    pub const GLOBAL_MACRO_OBS: usize = 14;
    pub const GLOBAL_STATIC_OBS: usize = 7 + GLOBAL_MACRO_OBS;
    // Per-ticker (25 total):
    // Portfolio (6): position_pct, unrealized_pnl, trade_ema, steps_since, position_age, target_weight
    // Multi-scale momentum (4): mom_5, mom_20, mom_60, mom_120
    // Momentum quality (4): acceleration, vol_adjusted, efficiency, trend_strength
    // Oscillators (4): rsi, range_pos, stoch_k, zscore
    // Trend (1): macd
    // Earnings (6): steps_to_next, revenue_growth, opex_growth, net_profit_growth, eps, eps_surprise
    pub const PER_TICKER_EARNINGS_OBS: usize = 6;
    pub const PER_TICKER_STATIC_OBS: usize = 19 + PER_TICKER_EARNINGS_OBS;
    pub const STATIC_OBSERVATIONS: usize = GLOBAL_STATIC_OBS + (TICKERS_COUNT * PER_TICKER_STATIC_OBS);
    pub const OBSERVATION_SPACE: usize = (TICKERS_COUNT * PRICE_DELTAS_PER_TICKER) + STATIC_OBSERVATIONS;

    pub const STEPS_PER_EPISODE: usize = 4_000;
    pub const ACTION_THRESHOLD: f64 = 0.001;
    pub const COMMISSION_RATE: f64 = 0.005;
    pub const RETROACTIVE_BUY_REWARD: bool = false;
}

pub fn symlog_target_clip() -> f64 {
    (constants::REWARD_RANGE + 1.0).ln()
}

pub mod theme;
