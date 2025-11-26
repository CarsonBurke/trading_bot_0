pub const TICKERS_COUNT: i64 = 1;
pub const ACTION_HISTORY_LEN: usize = 20;
pub const PRICE_DELTAS_PER_TICKER: usize = 2400;
// Static observations: step (1) + cash_percent (1) + position_percents (TICKER_COUNT) + action_history (TICKER_COUNT * ACTION_HISTORY_LEN)
pub const STATIC_OBSERVATIONS: usize = 1 + 1 + TICKERS_COUNT as usize + (ACTION_HISTORY_LEN * TICKERS_COUNT as usize * 2);
pub const OBSERVATION_SPACE: usize = (TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER) + STATIC_OBSERVATIONS;
pub const STEPS_PER_EPISODE: usize = 12_000;