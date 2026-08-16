use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use std::collections::VecDeque;
use std::ops::RangeInclusive;
use std::time::Instant;

use colored::Colorize;

#[cfg(test)]
use super::single::load_market_data_without_macro;
use super::single::{
    load_market_data, sample_training_tickers, Env, EnvMarketData, EnvMarketSnapshot,
};
use crate::{
    history::{episode_tickers_combined::EpisodeHistory, meta_tickers_combined::MetaHistory},
    torch::constants::{ACTION_HISTORY_LEN, PRICE_DELTAS_PER_TICKER, STEPS_PER_EPISODE},
    types::Account,
};

impl Env {
    pub const STARTING_CASH: f64 = 10_000.0;

    pub fn new(random_start: bool) -> Self {
        Self::new_with_recording(random_start, true, None)
    }

    pub fn new_with_tickers(tickers: Vec<String>, random_start: bool) -> Self {
        Self::new_with_tickers_recording_and_resampling(
            tickers,
            random_start,
            true,
            None,
            false,
            rand::rng().random(),
        )
    }

    pub fn new_with_recording(
        random_start: bool,
        record_history_io: bool,
        gens_path: Option<String>,
    ) -> Self {
        let rng = &mut rand::rng();
        let seed = rng.random::<u64>();
        let tickers = sample_training_tickers(rng);

        Self::new_with_tickers_recording_and_resampling(
            tickers,
            random_start,
            record_history_io,
            gens_path,
            random_start,
            seed,
        )
    }

    pub(super) fn new_with_recording_seed(
        random_start: bool,
        record_history_io: bool,
        gens_path: Option<String>,
        seed: u64,
    ) -> Self {
        let mut rng = ChaCha12Rng::seed_from_u64(derive_rng_seed(seed, 0));
        let tickers = sample_training_tickers(&mut rng);
        Self::new_with_tickers_recording_and_resampling(
            tickers,
            random_start,
            record_history_io,
            gens_path,
            random_start,
            seed,
        )
    }

    pub fn new_with_tickers_and_recording(
        tickers: Vec<String>,
        random_start: bool,
        record_history_io: bool,
        gens_path: Option<String>,
    ) -> Self {
        Self::new_with_tickers_recording_and_resampling(
            tickers,
            random_start,
            record_history_io,
            gens_path,
            false,
            rand::rng().random(),
        )
    }

    fn new_with_tickers_recording_and_resampling(
        tickers: Vec<String>,
        random_start: bool,
        record_history_io: bool,
        gens_path: Option<String>,
        resample_tickers_on_reset: bool,
        rng_seed: u64,
    ) -> Self {
        let market_data = load_market_data(&tickers, true);
        Self::new_from_market_data(
            tickers,
            random_start,
            record_history_io,
            gens_path,
            resample_tickers_on_reset,
            market_data,
            rng_seed,
        )
    }

    /// Deterministic environment over the real packed corpus, with macro indicators stubbed
    /// because they need FRED over the network.
    #[cfg(test)]
    pub(crate) fn new_without_macro_for_test(random_start: bool, seed: u64) -> Self {
        let mut rng = ChaCha12Rng::seed_from_u64(derive_rng_seed(seed, 0));
        let tickers = sample_training_tickers(&mut rng);
        let market_data = load_market_data_without_macro(&tickers, false);
        Self::new_from_market_data(
            tickers,
            random_start,
            true,
            None,
            false,
            market_data,
            seed,
        )
    }

    fn new_from_market_data(
        tickers: Vec<String>,
        random_start: bool,
        record_history_io: bool,
        gens_path: Option<String>,
        resample_tickers_on_reset: bool,
        market_data: EnvMarketData,
        rng_seed: u64,
    ) -> Self {
        let num_tickers = tickers.len();
        let mut target_weights = vec![0.0; num_tickers + 1];
        target_weights[num_tickers] = 1.0;
        let mut realized_weights = vec![0.0; num_tickers + 1];
        realized_weights[num_tickers] = 1.0;

        Self {
            env_id: 0,
            step: 0,
            max_step: market_data.total_data_length - 2,
            prices: market_data.prices,
            price_deltas: market_data.price_deltas,
            account: Account::default(),
            episode_history: EpisodeHistory::new(num_tickers),
            meta_history: MetaHistory::default(),
            tickers,
            episode: 0,
            episode_start: Instant::now(),
            action_history: VecDeque::with_capacity(ACTION_HISTORY_LEN),
            episode_start_offset: 0,
            total_data_length: market_data.total_data_length,
            random_start,
            resample_tickers_on_reset,
            peak_assets: Self::STARTING_CASH,
            last_fill_ratio: 1.0,
            trade_activity_ema: vec![0.0; num_tickers],
            steps_since_trade: vec![0; num_tickers],
            position_open_step: vec![None; num_tickers],
            ticker_perm: (0..num_tickers).collect(),
            target_weights,
            realized_weights,
            momentum: market_data.momentum,
            earnings: market_data.earnings,
            macro_ind: market_data.macro_ind,
            record_history_io,
            gens_path,
            rng_seed,
            rng_counter: 1,
        }
    }

    pub(super) fn handle_episode_end(&mut self, absolute_step: usize) {
        let history_start_offset = self.episode_start_offset + 1;
        let mut index_return = 0.0;
        for ticker_idx in 0..self.tickers.len() {
            let start_price = self.prices[ticker_idx][self.episode_start_offset];
            let end_price = self.prices[ticker_idx][absolute_step];
            index_return += (end_price / start_price - 1.0) * 100.0;
        }
        index_return /= self.tickers.len() as f64;

        let strategy_return = (self.account.total_assets / Self::STARTING_CASH - 1.0) * 100.0;
        let outperformance = strategy_return - index_return;

        let strategy_str = if strategy_return >= 0.0 {
            format!("{:.2}%", strategy_return).green()
        } else {
            format!("{:.2}%", strategy_return).red()
        };

        let index_str = if index_return >= 0.0 {
            format!("{:.2}%", index_return).cyan()
        } else {
            format!("{:.2}%", index_return).red()
        };

        let outperf_str = if outperformance > 0.0 {
            format!("+{:.2}%", outperformance).bright_green().bold()
        } else if outperformance < 0.0 {
            format!("{:.2}%", outperformance).bright_red().bold()
        } else {
            format!("{:.2}%", outperformance).yellow()
        };

        let recorded_history = if self.record_history_io {
            self.meta_history
                .record(&self.episode_history, outperformance);
            let report_result = if let Some(ref gens_path) = self.gens_path {
                self.episode_history.record_to_path(
                    gens_path,
                    self.episode,
                    &self.tickers,
                    &self.prices,
                    history_start_offset,
                )
            } else {
                self.episode_history.record(
                    self.episode,
                    &self.tickers,
                    &self.prices,
                    history_start_offset,
                )
            }
            .and_then(|()| {
                if self.episode % 5 != 0 {
                    return Ok(());
                }
                if let Some(ref gens_path) = self.gens_path {
                    self.meta_history.write_reports(self.episode, gens_path)
                } else {
                    self.meta_history.write_reports_default(self.episode)
                }
            });
            match report_result {
                Ok(()) => true,
                Err(error) => {
                    eprintln!(
                        "failed persisting reports for episode {} in env {}: {error:#}",
                        self.episode, self.env_id
                    );
                    false
                }
            }
        } else {
            false
        };

        println!(
            "{} {} [{}] - Total Assets: {} ({}) cumulative reward {:.2} | Index: {} | Outperformance: {} | Commissions: {} | tickers {:?} time {:.2}s {}",
            "Episode".bright_blue(),
            self.episode.to_string().bright_blue().bold(),
            format!("Env {}", self.env_id).bright_blue(),
            format!("${:.2}", self.account.total_assets).bright_white().bold(),
            strategy_str,
            self.episode_history.rewards.iter().sum::<f64>(),
            index_str,
            outperf_str,
            format!("${:.2}", self.episode_history.total_commissions).yellow(),
            self.tickers,
            Instant::now().duration_since(self.episode_start).as_secs_f32(),
            match recorded_history {
                true => "| recorded history",
                false => "",
            }
        );

        self.episode_start = Instant::now();
        self.episode += 1;
    }

    fn set_training_tickers(&mut self, tickers: Vec<String>) {
        if tickers == self.tickers {
            return;
        }

        let market_data = load_market_data(&tickers, false);
        let num_tickers = tickers.len();

        self.tickers = tickers;
        self.prices = market_data.prices;
        self.price_deltas = market_data.price_deltas;
        self.momentum = market_data.momentum;
        self.earnings = market_data.earnings;
        self.macro_ind = market_data.macro_ind;
        self.total_data_length = market_data.total_data_length;
        self.max_step = self.total_data_length - 2;
        self.ticker_perm = (0..num_tickers).collect();
        self.trade_activity_ema = vec![0.0; num_tickers];
        self.steps_since_trade = vec![0; num_tickers];
        self.position_open_step = vec![None; num_tickers];
    }

    pub(super) fn resample_training_tickers(&mut self) {
        let mut rng = self.next_rng();
        let tickers = sample_training_tickers(&mut rng);
        self.set_training_tickers(tickers);
    }

    pub(super) fn market_snapshot(&self) -> EnvMarketSnapshot {
        EnvMarketSnapshot {
            tickers: self.tickers.clone(),
            prices: self.prices.clone(),
            price_deltas: self.price_deltas.clone(),
            momentum: self.momentum.clone(),
            earnings: self.earnings.clone(),
            macro_ind: self.macro_ind.clone(),
            total_data_length: self.total_data_length,
            ticker_perm: self.ticker_perm.clone(),
        }
    }

    pub(super) fn apply_market_snapshot(&mut self, snapshot: &EnvMarketSnapshot) {
        let num_tickers = snapshot.tickers.len();

        self.tickers.clone_from(&snapshot.tickers);
        self.prices.clone_from(&snapshot.prices);
        self.price_deltas.clone_from(&snapshot.price_deltas);
        self.momentum.clone_from(&snapshot.momentum);
        self.earnings.clone_from(&snapshot.earnings);
        self.macro_ind = snapshot.macro_ind.clone();
        self.total_data_length = snapshot.total_data_length;
        self.max_step = self.total_data_length - 2;
        self.ticker_perm.clone_from(&snapshot.ticker_perm);
        self.trade_activity_ema.resize(num_tickers, 0.0);
        self.steps_since_trade.resize(num_tickers, 0);
        self.position_open_step.resize(num_tickers, None);
    }

    pub(crate) fn sample_episode_start_offset(&mut self) -> usize {
        let valid_starts = full_episode_start_offsets(self.total_data_length);
        if self.random_start {
            let mut rng = self.next_rng();
            rng.random_range(valid_starts)
        } else {
            *valid_starts.start()
        }
    }

    pub(super) fn has_next_transition(&self) -> bool {
        self.step <= self.max_step
    }

    pub(super) fn get_is_done(&self) -> f32 {
        if self.step >= self.max_step {
            1.0
        } else {
            0.0
        }
    }

    pub(crate) fn reset_existing_episode_state(&mut self) {
        if self.resample_tickers_on_reset && self.episode > 0 {
            self.resample_training_tickers();
        }

        let start_offset = self.sample_episode_start_offset();
        self.reset_existing_episode_state_at(start_offset);
    }

    pub(super) fn reset_existing_episode_state_at(&mut self, episode_start_offset: usize) {
        self.episode_start_offset = episode_start_offset;

        self.step = 0;
        self.max_step =
            (self.total_data_length - self.episode_start_offset).min(STEPS_PER_EPISODE) - 2;

        self.account = Account::new(Self::STARTING_CASH, self.tickers.len());
        self.episode_start = Instant::now();
        self.episode_history = EpisodeHistory::new(self.tickers.len());
        self.action_history.clear();

        self.peak_assets = Self::STARTING_CASH;
        self.last_fill_ratio = 1.0;
        self.trade_activity_ema.fill(0.0);
        self.steps_since_trade.fill(0);
        self.position_open_step.fill(None);
        let n = self.tickers.len();
        // self.target_weights = vec![1.0 / n as f64; n + 1];
        // self.target_weights[n] = 0.0; // cash starts at 0

        // Initialize to 100% cash, 0% tickers.
        self.target_weights = vec![0.0; n + 1];
        self.target_weights[n] = 1.0;
        self.realized_weights = vec![0.0; n + 1];
        self.realized_weights[n] = 1.0;

        // Shuffle ticker permutation for this episode
        let mut rng = self.next_rng();
        self.ticker_perm.shuffle(&mut rng);
    }

    pub fn record_inference_to(
        &self,
        infer_dir: &std::path::Path,
        episode: usize,
    ) -> anyhow::Result<()> {
        self.episode_history.record_to_path(
            infer_dir.to_string_lossy().as_ref(),
            episode,
            &self.tickers,
            &self.prices,
            self.episode_start_offset + 1,
        )
    }
}

fn derive_rng_seed(seed: u64, counter: u64) -> u64 {
    let mut z = seed
        .wrapping_add(counter.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl Env {
    #[cfg(test)]
    pub(crate) fn test_rng_counter(&self) -> u64 {
        self.rng_counter
    }

    fn next_rng(&mut self) -> ChaCha12Rng {
        let stream = self.rng_counter;
        self.rng_counter = self
            .rng_counter
            .checked_add(1)
            .expect("environment RNG counter overflowed");
        ChaCha12Rng::seed_from_u64(derive_rng_seed(self.rng_seed, stream))
    }
}

fn full_episode_start_offsets(total_data_length: usize) -> RangeInclusive<usize> {
    let minimum_data_length = PRICE_DELTAS_PER_TICKER
        .checked_add(STEPS_PER_EPISODE)
        .expect("episode data-length requirement overflowed");
    assert!(
        total_data_length >= minimum_data_length,
        "full episode requires at least {minimum_data_length} bars ({PRICE_DELTAS_PER_TICKER} context + {STEPS_PER_EPISODE} episode), found {total_data_length}"
    );

    PRICE_DELTAS_PER_TICKER..=total_data_length - STEPS_PER_EPISODE
}

#[cfg(test)]
mod tests {
    use super::full_episode_start_offsets;
    use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STEPS_PER_EPISODE};

    #[test]
    fn exact_minimum_history_has_one_full_episode_start() {
        let data_len = PRICE_DELTAS_PER_TICKER + STEPS_PER_EPISODE;
        let starts = full_episode_start_offsets(data_len);

        assert_eq!(*starts.start(), PRICE_DELTAS_PER_TICKER);
        assert_eq!(*starts.end(), PRICE_DELTAS_PER_TICKER);
        assert_eq!(starts.count(), 1);
    }

    #[test]
    fn all_full_episode_starts_include_newest_valid_offset() {
        let extra_starts = 37;
        let data_len = PRICE_DELTAS_PER_TICKER + STEPS_PER_EPISODE + extra_starts;
        let starts = full_episode_start_offsets(data_len);
        let newest_start = *starts.end();
        let terminal_step = STEPS_PER_EPISODE - 2;

        assert_eq!(starts.count(), extra_starts + 1);
        assert_eq!(newest_start, data_len - STEPS_PER_EPISODE);
        assert_eq!(newest_start + terminal_step + 1, data_len - 1);
    }

    #[test]
    #[should_panic(expected = "full episode requires at least")]
    fn history_shorter_than_one_full_episode_is_rejected() {
        full_episode_start_offsets(PRICE_DELTAS_PER_TICKER + STEPS_PER_EPISODE - 1);
    }
}
