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
    data::historical::{exchange_time, get_packed_historical_data},
    data::universe::cached_bar_universe,
    history::{episode_tickers_combined::EpisodeHistory, meta_tickers_combined::MetaHistory},
    torch::constants::TICKERS_COUNT,
    types::Account,
    utils::log_returns,
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
    /// Counter-based RNG state. Each stochastic environment operation uses a
    /// fresh, domain-separated stream so a checkpoint only needs two integers.
    pub(super) rng_seed: u64,
    pub(super) rng_counter: u64,
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

enum MacroLoadMode {
    Required,
    #[cfg(test)]
    Empty,
}

pub(crate) fn sample_training_tickers(rng: &mut impl Rng) -> Vec<String> {
    let universe = cached_bar_universe();
    assert!(
        universe.len() >= TICKERS_COUNT as usize,
        "need at least {} eligible corpus tickers, found {}",
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
    let mapped_bars = get_packed_historical_data(tickers);
    let mut prices: Vec<Vec<f64>> = Vec::with_capacity(tickers.len());
    let mut price_deltas = Vec::with_capacity(tickers.len());
    for bars in &mapped_bars {
        let closes: Vec<f64> = bars.iter().map(|bar| f64::from(bar.close)).collect();
        price_deltas.push(log_returns(closes.iter().copied()));
        prices.push(closes);
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

    let bar_times: Vec<Vec<time::OffsetDateTime>> = mapped_bars
        .iter()
        .map(|bars| bars.iter().map(|bar| exchange_time(bar.ts())).collect())
        .collect();

    if log_progress {
        eprint!("dates..");
    }
    let bar_dates: Vec<Vec<String>> = bar_times
        .iter()
        .map(|times| {
            times
                .iter()
                .map(|t| format!("{:04}-{:02}-{:02}", t.year(), t.month() as u8, t.day()))
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
        MacroLoadMode::Required => MacroIndicators::get_or_compute(&bar_times[0]),
        #[cfg(test)]
        MacroLoadMode::Empty => Arc::new(MacroIndicators::empty(total_data_length)),
    };
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

#[cfg(test)]
mod tests {
    use super::Env;
    use crate::torch::constants::{ACTION_COUNT, PRICE_DELTAS_PER_TICKER, STEPS_PER_EPISODE};
    use shared::constants::TICKERS_COUNT as TICKERS_COUNT_USIZE;

    /// End-to-end proof that PPO runs off the packed corpus: a real symbol is loaded from
    /// `long_data/bars/<SYM>.300.bars` and stepped. Macro indicators are stubbed because
    /// they need FRED over the network; every other channel is the production path.
    #[test]
    fn ppo_env_loads_the_packed_corpus_and_steps() {
        let mut env = Env::new_without_macro_for_test(true, 0x5A11_0000_D0F0);
        eprintln!(
            "smoke: {:?} with {} bars from the packed corpus",
            env.tickers, env.total_data_length
        );
        assert_eq!(env.prices.len(), env.tickers.len());
        // The floor `full_episode_start_offsets` actually enforces. `MIN_TRADING_BARS` is a
        // universe filter measured on the raw file, before `usable()` drops any bar, so it
        // is not the invariant to assert here.
        assert!(
            env.total_data_length >= PRICE_DELTAS_PER_TICKER + STEPS_PER_EPISODE,
            "a tradable symbol must serve a full observation window plus an episode"
        );

        let (price_deltas, static_obs) = env.reset_single();
        assert_eq!(
            price_deltas.len(),
            TICKERS_COUNT_USIZE * PRICE_DELTAS_PER_TICKER
        );
        assert!(price_deltas.iter().all(|value| value.is_finite()));
        assert!(static_obs.iter().all(|value| value.is_finite()));

        let hold = vec![0.0; ACTION_COUNT as usize];
        for step in 0..8 {
            let transition = env.step_step_single(&hold);
            assert_eq!(transition.is_done, 0.0, "step {step} ended the episode early");
            assert!(transition.reward.is_finite(), "step {step} reward");
            assert!(transition.static_obs.iter().all(|value| value.is_finite()));
        }
    }
}
