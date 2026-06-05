use rand::seq::IndexedRandom;
use rand::Rng;
use shared::constants::{
    STATIC_OBSERVATIONS as STATIC_OBSERVATIONS_USIZE, TICKERS_COUNT as TICKERS_COUNT_USIZE,
};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use super::earnings::EarningsIndicators;
use super::macro_ind::MacroIndicators;
use super::momentum::MomentumIndicators;
use crate::{
    data::historical::{get_historical_data, get_historical_series},
    data::universe::cached_eligible_training_universe,
    history::{episode_tickers_combined::EpisodeHistory, meta_tickers_combined::MetaHistory},
    torch::constants::TICKERS_COUNT,
    types::Account,
    utils::get_price_deltas,
};

pub struct Env {
    pub env_id: usize,
    pub step: usize,
    pub max_step: usize,
    pub tickers: Vec<String>,
    pub prices: Vec<Vec<f64>>,
    pub price_deltas: Vec<Vec<f64>>,
    pub account: Account,
    pub episode_history: EpisodeHistory,
    pub meta_history: MetaHistory,
    pub(super) episode_start: Instant,
    pub episode: usize,
    pub action_history: VecDeque<Vec<f64>>,
    pub episode_start_offset: usize,
    pub(super) total_data_length: usize,
    pub(super) random_start: bool,
    pub(super) resample_tickers_on_reset: bool,
    pub peak_assets: f64,
    pub last_fill_ratio: f64,
    pub trade_activity_ema: Vec<f64>,
    pub steps_since_trade: Vec<usize>,
    pub position_open_step: Vec<Option<usize>>,
    pub ticker_perm: Vec<usize>,
    pub target_weights: Vec<f64>,
    pub realized_weights: Vec<f64>,
    pub momentum: Vec<Arc<MomentumIndicators>>,
    pub earnings: Vec<Arc<EarningsIndicators>>,
    pub macro_ind: Arc<MacroIndicators>,
    pub(super) record_history_io: bool,
    pub(super) gens_path: Option<String>,
}

pub const TRADE_EMA_ALPHA: f64 = 0.05; // ~40-step equivalent window

#[derive(Clone)]
pub(super) struct EnvMarketSnapshot {
    pub tickers: Vec<String>,
    pub(super) prices: Vec<Vec<f64>>,
    pub(super) price_deltas: Vec<Vec<f64>>,
    pub(super) momentum: Vec<Arc<MomentumIndicators>>,
    pub(super) earnings: Vec<Arc<EarningsIndicators>>,
    pub(super) macro_ind: Arc<MacroIndicators>,
    pub(super) total_data_length: usize,
    pub ticker_perm: Vec<usize>,
}

pub(super) struct EnvMarketData {
    pub(super) prices: Vec<Vec<f64>>,
    pub(super) price_deltas: Vec<Vec<f64>>,
    pub(super) momentum: Vec<Arc<MomentumIndicators>>,
    pub(super) earnings: Vec<Arc<EarningsIndicators>>,
    pub(super) macro_ind: Arc<MacroIndicators>,
    pub(super) total_data_length: usize,
}

pub(crate) fn sample_training_tickers(rng: &mut impl Rng) -> Vec<String> {
    let universe = cached_eligible_training_universe();
    assert!(
        universe.len() >= TICKERS_COUNT as usize,
        "need at least {} cached eligible tickers, found {}",
        TICKERS_COUNT,
        universe.len()
    );
    universe
        .choose_multiple(rng, TICKERS_COUNT as usize)
        .cloned()
        .collect()
}

pub(super) fn load_market_data(tickers: &[String], log_progress: bool) -> EnvMarketData {
    if log_progress {
        eprint!("  hist..");
    }
    let ticker_refs = tickers
        .iter()
        .map(|ticker| ticker.as_str())
        .collect::<Vec<&str>>();
    let mapped_bars = get_historical_data(Some(&ticker_refs));
    let mut prices = Vec::with_capacity(tickers.len());
    let mut price_deltas = Vec::with_capacity(tickers.len());
    for (i, ticker) in tickers.iter().enumerate() {
        if let Some((cached_prices, cached_deltas)) = get_historical_series(ticker) {
            prices.push(cached_prices);
            price_deltas.push(cached_deltas);
        } else {
            prices.push(mapped_bars[i].iter().map(|bar| bar.close).collect());
            price_deltas.push(get_price_deltas(&mapped_bars[i]));
        }
    }

    if log_progress {
        eprint!("mom..");
    }
    let momentum: Vec<Arc<MomentumIndicators>> = tickers
        .iter()
        .zip(prices.iter())
        .map(|(ticker, p)| MomentumIndicators::get_or_compute(ticker, p))
        .collect();

    let total_data_length = prices[0].len();

    if log_progress {
        eprint!("dates..");
    }
    let bar_dates: Vec<Vec<String>> = mapped_bars
        .iter()
        .map(|bars| {
            bars.iter()
                .map(|b| {
                    format!(
                        "{:04}-{:02}-{:02}",
                        b.date.year(),
                        b.date.month() as u8,
                        b.date.day()
                    )
                })
                .collect()
        })
        .collect();

    if log_progress {
        eprint!("earn..");
    }
    let mut earnings: Vec<Arc<EarningsIndicators>> = Vec::with_capacity(tickers.len());
    for (i, ticker) in tickers.iter().enumerate() {
        if log_progress {
            eprint!("{}..", ticker);
        }
        let reports = crate::data::get_cached_earnings_data_any(ticker);
        if log_progress {
            eprint!("r");
        }
        let ind = EarningsIndicators::get_or_compute(ticker, &reports, &bar_dates[i], &prices[i]);
        if log_progress {
            eprint!("i");
        }
        earnings.push(ind);
    }

    if log_progress {
        eprint!("macro..");
    }
    let macro_ind = MacroIndicators::get_or_compute(&bar_dates[0]);
    if log_progress {
        eprintln!("done");
    }

    EnvMarketData {
        prices,
        price_deltas,
        momentum,
        earnings,
        macro_ind,
        total_data_length,
    }
}

/// Single-environment step result with raw values (for VecEnv)
pub struct SingleStep {
    pub reward: f64,
    pub reward_per_ticker: [f32; TICKERS_COUNT_USIZE],
    pub price_deltas: Vec<f32>,
    pub static_obs: [f32; STATIC_OBSERVATIONS_USIZE],
    pub is_done: f32,
}

pub struct SingleStepStep {
    pub reward: f64,
    pub reward_per_ticker: [f32; TICKERS_COUNT_USIZE],
    pub step_deltas: [f32; TICKERS_COUNT_USIZE],
    pub static_obs: [f32; STATIC_OBSERVATIONS_USIZE],
    pub is_done: f32,
}
