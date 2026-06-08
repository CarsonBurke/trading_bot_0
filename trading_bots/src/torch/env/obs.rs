use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, TICKERS_COUNT};
use shared::constants::{
    STATIC_OBSERVATIONS as STATIC_OBSERVATIONS_USIZE, TICKERS_COUNT as TICKERS_COUNT_USIZE,
};

use super::single::Env;

/// Global/macro scalars for the static observation, in canonical order.
/// Shared by training (`Env::build_static_obs_array`) and live inference so the
/// two paths cannot drift. Values are passed raw; scaling that applies to all
/// callers lives in `build_static_obs`.
pub struct GlobalObsInputs {
    pub cash_percent: f64,
    pub pnl: f64,
    pub drawdown: f64,
    pub commissions: f64,
    pub last_fill_ratio: f64,
    pub gdp_growth: f64,
    pub unemployment: f64,
    pub jobs_growth: f64,
    pub cpi_yoy: f64,
    pub core_cpi_yoy: f64,
    pub fed_funds: f64,
    pub treasury_10y: f64,
    pub yield_spread: f64,
    pub consumer_sentiment: f64,
    pub initial_claims: f64,
    pub steps_to_jobs: f64,
    pub steps_to_cpi: f64,
    pub steps_to_fomc: f64,
    pub steps_to_gdp: f64,
}

/// Per-ticker raw inputs for the static observation, in ticker-permutation order.
/// Raw momentum/earnings values are scaled inside `build_static_obs` so the
/// transforms live in exactly one place.
pub struct TickerObsInputs {
    pub position_percent: f64,
    pub appreciation: f64,
    pub trade_activity_ema: f64,
    pub steps_since_trade: usize,
    pub position_age: f64,
    pub realized_weight: f64,
    pub mom_5: f64,
    pub mom_20: f64,
    pub mom_60: f64,
    pub mom_120: f64,
    pub mom_accel: f64,
    pub vol_adj_mom: f64,
    pub efficiency: f64,
    pub trend_strength: f64,
    pub rsi: f64,
    pub range_pos: f64,
    pub stoch_k: f64,
    pub zscore: f64,
    pub macd: f64,
    pub earnings_steps_to_next: f64,
    pub revenue_growth: f64,
    pub opex_growth: f64,
    pub net_profit_growth: f64,
    pub eps: f64,
    pub eps_surprise: f64,
}

/// Canonical static-observation builder shared by training and live inference.
/// `tickers` must be supplied in ticker-permutation order.
pub fn build_static_obs(
    global: &GlobalObsInputs,
    tickers: &[TickerObsInputs],
) -> [f32; STATIC_OBSERVATIONS_USIZE] {
    let mut static_obs = [0.0; STATIC_OBSERVATIONS_USIZE];
    let mut idx = 0usize;
    let mut push = |value: f32| {
        static_obs[idx] = value;
        idx += 1;
    };

    push(global.cash_percent as f32);
    push(global.pnl as f32);
    push(global.drawdown as f32);
    push(global.commissions as f32);
    push(global.last_fill_ratio as f32);

    push(global.gdp_growth as f32);
    push(global.unemployment as f32);
    push(global.jobs_growth as f32);
    push(global.cpi_yoy as f32);
    push(global.core_cpi_yoy as f32);
    push(global.fed_funds as f32);
    push(global.treasury_10y as f32);
    push(global.yield_spread as f32);
    push(global.consumer_sentiment as f32);
    push(global.initial_claims as f32);
    push(global.steps_to_jobs as f32);
    push(global.steps_to_cpi as f32);
    push(global.steps_to_fomc as f32);
    push(global.steps_to_gdp as f32);

    for t in tickers {
        push(t.position_percent.clamp(-1.0, 1.0) as f32);
        push(t.appreciation.clamp(-1.0, 1.0) as f32);
        push(t.trade_activity_ema as f32);
        push((1.0 / (1.0 + t.steps_since_trade as f64 / 50.0)) as f32);
        push(t.position_age as f32);
        push(t.realized_weight.clamp(0.0, 1.0) as f32);
        push(t.mom_5 as f32);
        push(t.mom_20.clamp(-0.5, 0.5) as f32);
        push(t.mom_60 as f32);
        push(t.mom_120 as f32);
        push(t.mom_accel as f32);
        push(t.vol_adj_mom as f32);
        push(t.efficiency as f32);
        push(t.trend_strength as f32);
        push((t.rsi * 2.0 - 1.0) as f32);
        push(t.range_pos as f32);
        push((t.stoch_k * 2.0 - 1.0) as f32);
        push((t.zscore / 3.0) as f32);
        push(t.macd as f32);
        push(t.earnings_steps_to_next as f32);
        push(t.revenue_growth as f32);
        push(t.opex_growth as f32);
        push(t.net_profit_growth as f32);
        push(t.eps as f32);
        push(t.eps_surprise as f32);
    }

    static_obs
}

impl Env {
    fn build_static_obs_array(&self, absolute_step: usize) -> [f32; STATIC_OBSERVATIONS_USIZE] {
        let macro_ind = &self.macro_ind;
        let global = GlobalObsInputs {
            cash_percent: self.account.cash / self.account.total_assets,
            pnl: (self.account.total_assets / Self::STARTING_CASH) - 1.0,
            drawdown: if self.peak_assets > 0.0 {
                (self.account.total_assets / self.peak_assets) - 1.0
            } else {
                0.0
            },
            commissions: self.episode_history.total_commissions / Self::STARTING_CASH,
            last_fill_ratio: self.last_fill_ratio,
            gdp_growth: macro_ind.gdp_growth[absolute_step],
            unemployment: macro_ind.unemployment[absolute_step],
            jobs_growth: macro_ind.jobs_growth[absolute_step],
            cpi_yoy: macro_ind.cpi_yoy[absolute_step],
            core_cpi_yoy: macro_ind.core_cpi_yoy[absolute_step],
            fed_funds: macro_ind.fed_funds[absolute_step],
            treasury_10y: macro_ind.treasury_10y[absolute_step],
            yield_spread: macro_ind.yield_spread[absolute_step],
            consumer_sentiment: macro_ind.consumer_sentiment[absolute_step],
            initial_claims: macro_ind.initial_claims[absolute_step],
            steps_to_jobs: macro_ind.steps_to_jobs[absolute_step],
            steps_to_cpi: macro_ind.steps_to_cpi[absolute_step],
            steps_to_fomc: macro_ind.steps_to_fomc[absolute_step],
            steps_to_gdp: macro_ind.steps_to_gdp[absolute_step],
        };

        let position_percents = self.account.position_percents(&self.prices, absolute_step);

        let tickers: Vec<TickerObsInputs> = self
            .ticker_perm
            .iter()
            .map(|&real_idx| {
                let m = &self.momentum[real_idx];
                let e = &self.earnings[real_idx];
                TickerObsInputs {
                    position_percent: position_percents[real_idx],
                    appreciation: self.account.positions[real_idx]
                        .appreciation(self.prices[real_idx][absolute_step]),
                    trade_activity_ema: self.trade_activity_ema[real_idx],
                    steps_since_trade: self.steps_since_trade[real_idx],
                    position_age: self.position_open_step[real_idx]
                        .map(|s| (absolute_step.saturating_sub(s) as f64 / 500.0).min(1.0))
                        .unwrap_or(0.0),
                    realized_weight: self.realized_weights[real_idx],
                    mom_5: m.mom_5[absolute_step],
                    mom_20: self.prices[real_idx][absolute_step]
                        / self.prices[real_idx][absolute_step.saturating_sub(20)]
                        - 1.0,
                    mom_60: m.mom_60[absolute_step],
                    mom_120: m.mom_120[absolute_step],
                    mom_accel: m.mom_accel[absolute_step],
                    vol_adj_mom: m.vol_adj_mom[absolute_step],
                    efficiency: m.efficiency[absolute_step],
                    trend_strength: m.trend_strength[absolute_step],
                    rsi: m.rsi[absolute_step],
                    range_pos: m.range_pos[absolute_step],
                    stoch_k: m.stoch_k[absolute_step],
                    zscore: m.zscore[absolute_step],
                    macd: m.macd[absolute_step],
                    earnings_steps_to_next: e.steps_to_next[absolute_step],
                    revenue_growth: e.revenue_growth[absolute_step],
                    opex_growth: e.opex_growth[absolute_step],
                    net_profit_growth: e.net_profit_growth[absolute_step],
                    eps: e.eps[absolute_step],
                    eps_surprise: e.eps_surprise[absolute_step],
                }
            })
            .collect();

        build_static_obs(&global, &tickers)
    }

    pub fn get_next_obs(&self) -> (Vec<f32>, Vec<f32>) {
        let mut price_deltas = Vec::with_capacity(TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER);
        let absolute_step = self.episode_start_offset + self.step;
        debug_assert!(
            absolute_step + 1 >= PRICE_DELTAS_PER_TICKER,
            "absolute_step {} too small for full window {}",
            absolute_step,
            PRICE_DELTAS_PER_TICKER
        );

        for &real_idx in &self.ticker_perm {
            let ticker_price_deltas = &self.price_deltas[real_idx];
            let end_idx = absolute_step + 1;
            let start_idx = end_idx - PRICE_DELTAS_PER_TICKER;
            let slice = &ticker_price_deltas[start_idx..end_idx];
            price_deltas.extend(slice.iter().map(|&x| x as f32));
        }

        let static_obs = self.build_static_obs_array(absolute_step).to_vec();

        (price_deltas, static_obs)
    }

    pub fn get_next_step_obs(
        &self,
    ) -> ([f32; TICKERS_COUNT_USIZE], [f32; STATIC_OBSERVATIONS_USIZE]) {
        let absolute_step = self.episode_start_offset + self.step;
        let mut step_deltas = [0.0; TICKERS_COUNT_USIZE];

        for (i, &real_idx) in self.ticker_perm.iter().enumerate() {
            let ticker_price_deltas = &self.price_deltas[real_idx];
            let v = ticker_price_deltas
                .get(absolute_step)
                .copied()
                .unwrap_or(0.0);
            step_deltas[i] = v as f32;
        }

        (step_deltas, self.build_static_obs_array(absolute_step))
    }
}
