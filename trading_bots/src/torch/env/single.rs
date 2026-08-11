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
    data::historical::get_historical_data,
    data::universe::cached_eligible_training_universe,
    history::{episode_tickers_combined::EpisodeHistory, meta_tickers_combined::MetaHistory},
    torch::constants::TICKERS_COUNT,
    types::Account,
    utils::get_price_deltas,
};

pub const OHLC_BAR_FEATURES: usize = 16;

pub struct Env {
    pub env_id: usize,
    pub step: usize,
    pub max_step: usize,
    pub tickers: Vec<String>,
    pub prices: Vec<Vec<f64>>,
    pub price_deltas: Vec<Vec<f64>>,
    pub ohlc_features: Vec<Vec<[f32; OHLC_BAR_FEATURES]>>,
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
    pub(super) ohlc_features: Vec<Vec<[f32; OHLC_BAR_FEATURES]>>,
    pub(super) momentum: Vec<Arc<MomentumIndicators>>,
    pub(super) earnings: Vec<Arc<EarningsIndicators>>,
    pub(super) macro_ind: Arc<MacroIndicators>,
    pub(super) total_data_length: usize,
    pub ticker_perm: Vec<usize>,
}

pub(super) struct EnvMarketData {
    pub(super) prices: Vec<Vec<f64>>,
    pub(super) price_deltas: Vec<Vec<f64>>,
    pub(super) ohlc_features: Vec<Vec<[f32; OHLC_BAR_FEATURES]>>,
    pub(super) momentum: Vec<Arc<MomentumIndicators>>,
    pub(super) earnings: Vec<Arc<EarningsIndicators>>,
    pub(super) macro_ind: Arc<MacroIndicators>,
    pub(super) total_data_length: usize,
}

enum MacroLoadMode {
    Required,
    #[cfg(test)]
    Empty,
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
    load_market_data_with_macro(tickers, log_progress, MacroLoadMode::Required)
}

#[cfg(test)]
pub(super) fn load_market_data_without_macro(
    tickers: &[String],
    log_progress: bool,
) -> EnvMarketData {
    load_market_data_with_macro(tickers, log_progress, MacroLoadMode::Empty)
}

fn load_market_data_with_macro(
    tickers: &[String],
    log_progress: bool,
    macro_mode: MacroLoadMode,
) -> EnvMarketData {
    if log_progress {
        eprint!("  hist..");
    }
    let ticker_refs = tickers
        .iter()
        .map(|ticker| ticker.as_str())
        .collect::<Vec<&str>>();
    let mapped_bars = get_historical_data(Some(&ticker_refs));
    let mut prices: Vec<Vec<f64>> = Vec::with_capacity(tickers.len());
    let mut price_deltas = Vec::with_capacity(tickers.len());
    let mut ohlc_features = Vec::with_capacity(tickers.len());
    for bars in &mapped_bars {
        ohlc_features.push(build_ohlc_features(bars));
        prices.push(bars.iter().map(|bar| bar.close).collect::<Vec<_>>());
        price_deltas.push(get_price_deltas(bars));
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
    let macro_ind = match macro_mode {
        MacroLoadMode::Required => MacroIndicators::get_or_compute(&bar_dates[0]),
        #[cfg(test)]
        MacroLoadMode::Empty => Arc::new(MacroIndicators::empty(total_data_length)),
    };
    if log_progress {
        eprintln!("done");
    }

    EnvMarketData {
        prices,
        price_deltas,
        ohlc_features,
        momentum,
        earnings,
        macro_ind,
        total_data_length,
    }
}

pub(crate) fn build_ohlc_features(
    bars: &[ibapi::market_data::historical::Bar],
) -> Vec<[f32; OHLC_BAR_FEATURES]> {
    bars.iter()
        .enumerate()
        .map(|(i, bar)| {
            let open = bar.open;
            let high = bar.high.max(open).max(bar.close);
            let low = bar.low.min(open).min(bar.close);
            let close = bar.close;
            let prev = if i == 0 { bar } else { &bars[i - 1] };
            let prev_open = prev.open;
            let prev_close = prev.close;
            let prev_high = prev.high.max(prev_open).max(prev_close);
            let prev_low = prev.low.min(prev_open).min(prev_close);
            [
                rel_delta(open, prev_open),
                rel_delta(high, prev_high),
                rel_delta(low, prev_low),
                rel_delta(close, prev_close),
                rel_delta(open, high),
                rel_delta(open, low),
                rel_delta(open, close),
                rel_delta(high, open),
                rel_delta(high, low),
                rel_delta(high, close),
                rel_delta(low, open),
                rel_delta(low, high),
                rel_delta(low, close),
                rel_delta(close, open),
                rel_delta(close, high),
                rel_delta(close, low),
            ]
        })
        .collect()
}

fn rel_delta(a: f64, b: f64) -> f32 {
    if a.is_finite() && b.is_finite() && b > 0.0 {
        (a / b - 1.0) as f32
    } else {
        0.0
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

#[cfg(test)]
mod tests {
    use super::{build_ohlc_features, OHLC_BAR_FEATURES};
    use ibapi::market_data::historical::Bar;
    use time::{Duration, OffsetDateTime};

    fn bar(open: f64, high: f64, low: f64, close: f64) -> Bar {
        Bar {
            date: OffsetDateTime::UNIX_EPOCH + Duration::minutes(5),
            open,
            high,
            low,
            close,
            volume: 1_000.0,
            wap: close,
            count: 1,
        }
    }

    #[test]
    fn ohlc_features_have_sixteen_dimensions_and_expected_layout() {
        assert_eq!(OHLC_BAR_FEATURES, 16);
        let prev = bar(100.0, 105.0, 98.0, 102.0);
        let cur = bar(102.0, 108.0, 101.0, 106.0);
        let feats = build_ohlc_features(&[prev, cur]);
        assert_eq!(feats.len(), 2);
        let row = feats[1];
        assert_eq!(row.len(), 16);

        let rd = |a: f64, b: f64| (a / b - 1.0) as f32;
        let (o, h, l, c) = (102.0f64, 108.0f64, 101.0f64, 106.0f64);
        let (po, ph, pl, pc) = (100.0f64, 105.0f64, 98.0f64, 102.0f64);

        assert!((row[0] - rd(o, po)).abs() < 1e-6);
        assert!((row[1] - rd(h, ph)).abs() < 1e-6);
        assert!((row[2] - rd(l, pl)).abs() < 1e-6);
        assert!((row[3] - rd(c, pc)).abs() < 1e-6);

        assert!((row[4] - rd(o, h)).abs() < 1e-6);
        assert!((row[6] - rd(o, c)).abs() < 1e-6);
        assert!((row[8] - rd(h, l)).abs() < 1e-6);
        assert!((row[12] - rd(l, c)).abs() < 1e-6);
        assert!((row[13] - rd(c, o)).abs() < 1e-6);
        assert!((row[15] - rd(c, l)).abs() < 1e-6);
    }

    #[test]
    fn first_bar_inter_features_are_zero() {
        let feats = build_ohlc_features(&[bar(100.0, 110.0, 95.0, 104.0)]);
        let row = feats[0];
        for i in 0..4 {
            assert_eq!(row[i], 0.0);
        }
    }
}
