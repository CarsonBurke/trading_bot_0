use std::collections::VecDeque;
use std::sync::Arc;
use tch::Tensor;

use crate::data::get_cached_earnings_data_any;
use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT};
use crate::torch::env::earnings::EarningsIndicators;
use crate::torch::env::macro_ind::MacroIndicators;
use crate::torch::env::momentum::MomentumIndicators;
use crate::torch::env::obs::{build_static_obs, GlobalObsInputs, TickerObsInputs};
use crate::types::Account;

pub(super) const MAX_ACCOUNT_VALUE: Option<f64> = Some(10_000.0);

pub(super) struct LiveMarketState {
    pub(super) symbols: Vec<String>,
    pub(super) prices: Vec<VecDeque<f64>>,
    pub(super) price_deltas: Vec<VecDeque<f64>>,
    /// Calendar date (YYYY-MM-DD) of each retained price bar, shared across tickers.
    pub(super) bar_dates: VecDeque<String>,
    pub(super) account: Account,
    pub(super) starting_cash: f64,
    pub(super) peak_assets: f64,
    pub(super) total_commissions: f64,
    pub(super) step_count: usize,
    pub(super) last_fill_ratio: f64,
    pub(super) steps_since_trade: Vec<usize>,
    pub(super) position_open_step: Vec<Option<usize>>,
    pub(super) trade_activity_ema: Vec<f64>,
    /// Pending deltas per ticker for streaming inference.
    pub(super) pending_deltas: Vec<VecDeque<f64>>,
    /// Whether model has been initialized with full observation
    pub(super) model_initialized: bool,
}

impl LiveMarketState {
    pub(super) fn new(symbols: Vec<String>, starting_cash: f64) -> Self {
        let ticker_count = symbols.len();
        Self {
            symbols,
            prices: vec![VecDeque::with_capacity(PRICE_DELTAS_PER_TICKER + 1); ticker_count],
            price_deltas: vec![VecDeque::with_capacity(PRICE_DELTAS_PER_TICKER); ticker_count],
            bar_dates: VecDeque::with_capacity(PRICE_DELTAS_PER_TICKER + 1),
            account: Account::new(starting_cash, ticker_count),
            starting_cash,
            peak_assets: starting_cash,
            total_commissions: 0.0,
            step_count: 0,
            last_fill_ratio: 1.0,
            steps_since_trade: vec![0; ticker_count],
            position_open_step: vec![None; ticker_count],
            trade_activity_ema: vec![0.0; ticker_count],
            pending_deltas: vec![VecDeque::new(); ticker_count],
            model_initialized: false,
        }
    }

    /// Seed price/delta history and bar dates from historical 5-minute bars so
    /// the model can be fed a full observation window immediately, matching the
    /// resolution and warm-up the model was trained on.
    pub(super) fn seed_history(&mut self, ticker_idx: usize, closes: &[f64], dates: &[String]) {
        for &close in closes {
            self.update_price(ticker_idx, close);
        }
        if ticker_idx == 0 {
            for date in dates {
                self.push_bar_date(date.clone());
            }
        }
        self.pending_deltas[ticker_idx].clear();
    }

    pub(super) fn push_bar_date(&mut self, date: String) {
        self.bar_dates.push_back(date);
        if self.bar_dates.len() > PRICE_DELTAS_PER_TICKER + 1 {
            self.bar_dates.pop_front();
        }
    }

    pub(super) fn update_price(&mut self, ticker_idx: usize, price: f64) {
        self.prices[ticker_idx].push_back(price);
        if self.prices[ticker_idx].len() > PRICE_DELTAS_PER_TICKER + 1 {
            self.prices[ticker_idx].pop_front();
        }

        if self.prices[ticker_idx].len() >= 2 {
            let len = self.prices[ticker_idx].len();
            let prev_price = self.prices[ticker_idx][len - 2];
            let delta = (price / prev_price).ln();

            self.price_deltas[ticker_idx].push_back(delta);
            if self.price_deltas[ticker_idx].len() > PRICE_DELTAS_PER_TICKER {
                self.price_deltas[ticker_idx].pop_front();
            }

            self.pending_deltas[ticker_idx].push_back(delta);
        }
    }

    pub(super) fn get_current_prices(&self) -> Vec<f64> {
        self.prices
            .iter()
            .map(|q| *q.back().unwrap_or(&0.0))
            .collect()
    }

    pub(super) fn update_account_total(&mut self) {
        let current_prices = self.get_current_prices();
        let position_values: f64 = self
            .account
            .positions
            .iter()
            .enumerate()
            .map(|(i, p)| p.value_with_price(current_prices[i]))
            .sum();
        self.account.total_assets = position_values + self.account.cash;
        self.peak_assets = self.peak_assets.max(self.account.total_assets);
    }

    fn ticker_inputs(
        &self,
        ticker_idx: usize,
        prices: &[f64],
        bar_dates: &[String],
        last: usize,
    ) -> TickerObsInputs {
        let momentum = MomentumIndicators::compute(prices);

        let reports = get_cached_earnings_data_any(&self.symbols[ticker_idx]);
        let earnings = if reports.is_empty() {
            Arc::new(EarningsIndicators::empty(prices.len()))
        } else {
            Arc::new(EarningsIndicators::compute(&reports, bar_dates, prices))
        };

        let current_price = prices[last];
        let position = &self.account.positions[ticker_idx];
        let position_percent = if self.account.total_assets > 0.0 {
            position.value_with_price(current_price) / self.account.total_assets
        } else {
            0.0
        };

        let mom_20 = current_price / prices[last.saturating_sub(20)] - 1.0;

        TickerObsInputs {
            position_percent,
            appreciation: position.appreciation(current_price),
            trade_activity_ema: self.trade_activity_ema[ticker_idx],
            steps_since_trade: self.steps_since_trade[ticker_idx],
            position_age: self.position_open_step[ticker_idx]
                .map(|s| (self.step_count.saturating_sub(s) as f64 / 500.0).min(1.0))
                .unwrap_or(0.0),
            realized_weight: position_percent,
            mom_5: momentum.mom_5[last],
            mom_20,
            mom_60: momentum.mom_60[last],
            mom_120: momentum.mom_120[last],
            mom_accel: momentum.mom_accel[last],
            vol_adj_mom: momentum.vol_adj_mom[last],
            efficiency: momentum.efficiency[last],
            trend_strength: momentum.trend_strength[last],
            rsi: momentum.rsi[last],
            range_pos: momentum.range_pos[last],
            stoch_k: momentum.stoch_k[last],
            zscore: momentum.zscore[last],
            macd: momentum.macd[last],
            earnings_steps_to_next: earnings.steps_to_next[last],
            revenue_growth: earnings.revenue_growth[last],
            opex_growth: earnings.opex_growth[last],
            net_profit_growth: earnings.net_profit_growth[last],
            eps: earnings.eps[last],
            eps_surprise: earnings.eps_surprise[last],
        }
    }

    pub(super) fn build_observation(&self) -> Option<(Tensor, Tensor)> {
        if self
            .price_deltas
            .iter()
            .any(|d| d.len() < PRICE_DELTAS_PER_TICKER)
        {
            return None;
        }
        if self.bar_dates.len() != self.prices[0].len() {
            return None;
        }

        let mut price_deltas_flat =
            Vec::with_capacity(TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER);
        for ticker_deltas in &self.price_deltas {
            for &delta in ticker_deltas.iter().take(PRICE_DELTAS_PER_TICKER) {
                price_deltas_flat.push(delta as f32);
            }
        }

        let bar_dates: Vec<String> = self.bar_dates.iter().cloned().collect();
        let macro_ind = MacroIndicators::get_or_compute(&bar_dates);
        let mlast = macro_ind.gdp_growth.len() - 1;

        let global = GlobalObsInputs {
            cash_percent: self.account.cash / self.account.total_assets,
            pnl: (self.account.total_assets / self.starting_cash) - 1.0,
            drawdown: if self.peak_assets > 0.0 {
                (self.account.total_assets / self.peak_assets) - 1.0
            } else {
                0.0
            },
            commissions: self.total_commissions / self.starting_cash,
            last_fill_ratio: self.last_fill_ratio,
            gdp_growth: macro_ind.gdp_growth[mlast],
            unemployment: macro_ind.unemployment[mlast],
            jobs_growth: macro_ind.jobs_growth[mlast],
            cpi_yoy: macro_ind.cpi_yoy[mlast],
            core_cpi_yoy: macro_ind.core_cpi_yoy[mlast],
            fed_funds: macro_ind.fed_funds[mlast],
            treasury_10y: macro_ind.treasury_10y[mlast],
            yield_spread: macro_ind.yield_spread[mlast],
            consumer_sentiment: macro_ind.consumer_sentiment[mlast],
            initial_claims: macro_ind.initial_claims[mlast],
            steps_to_jobs: macro_ind.steps_to_jobs[mlast],
            steps_to_cpi: macro_ind.steps_to_cpi[mlast],
            steps_to_fomc: macro_ind.steps_to_fomc[mlast],
            steps_to_gdp: macro_ind.steps_to_gdp[mlast],
        };

        let tickers: Vec<TickerObsInputs> = (0..TICKERS_COUNT as usize)
            .map(|ticker_idx| {
                let prices: Vec<f64> = self.prices[ticker_idx].iter().copied().collect();
                let last = prices.len() - 1;
                self.ticker_inputs(ticker_idx, &prices, &bar_dates, last)
            })
            .collect();

        let static_obs = build_static_obs(&global, &tickers);

        let price_deltas_tensor = Tensor::from_slice(&price_deltas_flat)
            .view([1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        let static_obs_tensor =
            Tensor::from_slice(&static_obs).view([1, STATIC_OBSERVATIONS as i64]);

        Some((price_deltas_tensor, static_obs_tensor))
    }

    pub(super) fn has_pending_step(&self) -> bool {
        self.pending_deltas.iter().all(|deltas| !deltas.is_empty())
    }

    /// Take the next pending step delta per ticker for streaming inference.
    pub(super) fn take_step_deltas(&mut self) -> Option<Tensor> {
        if !self.has_pending_step() {
            return None;
        }
        let mut step_deltas = Vec::with_capacity(self.pending_deltas.len());
        for deltas in &mut self.pending_deltas {
            step_deltas.push(deltas.pop_front().unwrap_or(0.0) as f32);
        }
        Some(Tensor::from_slice(&step_deltas))
    }
}
