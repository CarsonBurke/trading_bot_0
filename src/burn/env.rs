use std::time::Instant;

use rand::{rngs::ThreadRng, seq::IndexedRandom};
use time::Duration;

use crate::{
    burn::{
        action::TradeAction,
        agent::base::{Action, ElemType, Environment, Snapshot},
        constants::OBSERVATION_SIZE,
        obs_state::ObservationState,
    },
    constants::TICKERS,
    data::historical::get_historical_data,
    history::{episode_tickers_combined::EpisodeHistory, meta_tickers_combined::MetaHistory},
    types::Account,
    utils::{get_mapped_price_deltas, get_price_deltas},
};

#[derive(Debug)]
pub struct Env {
    step: usize,
    episode: usize,
    account: Account,
    episode_history: EpisodeHistory,
    meta_history: MetaHistory,
    tickers: Vec<String>,
    prices: Vec<Vec<f64>>,
    price_deltas: Vec<Vec<f64>>,
    state: ObservationState,
    visualized: bool,
    episode_start: Instant,
    rng: ThreadRng,
}

impl Env {
    const STARTING_STEP: usize = OBSERVATION_SIZE;
    const STARTING_CASH: f64 = 10_000.0;
    const BUY_PERCENT: f64 = 0.05;
    const SELL_PERCENT: f64 = 0.05;

    fn on_done() {}
}

impl Environment for Env {
    type ActionType = TradeAction;
    type StateType = ObservationState;
    type RewardType = ElemType;

    fn new(visualized: bool) -> Self {
        let tickers = vec![
            // "SPY".to_string(),
            "TSLA".to_string(),
            // "AAPL".to_string(),
            // "TSLA".to_string(),
            // "AMD".to_string(),
            // "INTC".to_string(),
            // "NVDA".to_string(),
        ];

        Self {
            step: Self::STARTING_STEP,
            episode_start: Instant::now(),
            rng: rand::rng(),
            episode: 0,
            account: Account::new(Self::STARTING_CASH, tickers.len()),
            episode_history: EpisodeHistory::new(tickers.len()),
            meta_history: MetaHistory::default(),
            tickers,
            prices: vec![],
            price_deltas: vec![],
            state: ObservationState::new_random(),
            visualized,
        }
    }

    fn step(&mut self, action: Self::ActionType) -> Snapshot<Self> {
        let current_prices = &self.prices[0];

        self.account.update_total(&self.prices, self.step);
        let total_assets = self.account.total_assets;

        let mut reward = 0.0;

        match action {
            TradeAction::Buy => {
                let buy_total = (total_assets * Self::BUY_PERCENT).min(self.account.cash);

                if buy_total > 0.0 {
                    self.account.cash -= buy_total;

                    let quantity = buy_total / current_prices[self.step];
                    self.account.positions[0].add(current_prices[self.step], quantity);

                    self.episode_history.buys[0]
                        .insert(self.step, (current_prices[self.step], quantity));
                }
            }
            TradeAction::Sell => {
                let sell_total = (total_assets * Self::SELL_PERCENT)
                    .min(self.account.positions[0].value_with_price(current_prices[self.step]));

                if sell_total > 0.0 {
                    self.account.cash += sell_total;

                    let quantity = sell_total / current_prices[self.step];
                    self.account.positions[0].quantity -= quantity;

                    self.episode_history.sells[0]
                        .insert(self.step, (current_prices[self.step], quantity));

                    reward += self
                        .account
                        .positions
                        .iter()
                        .enumerate()
                        .map(|(index, position)| {
                            position.appreciation(self.prices[index][self.step]) * sell_total
                        })
                        .sum::<f64>();
                }
            }
            TradeAction::Hold => {}
        }

        for (index, _) in self.tickers.iter().enumerate() {
            self.episode_history.positioned[index].push(
                self.account.positions[index].value_with_price(self.prices[index][self.step]),
            );
        }
        self.episode_history.cash.push(self.account.cash);

        // Next step

        self.step += 1;
        let next_prices = &self.prices[0];

        self.state =
            ObservationState::new(self.step, &self.account, &self.prices, &self.price_deltas);

        // Reward

        self.account.update_total(&self.prices, self.step);
        // let reward = (self.account.total_assets - total_assets) / total_assets;
        // let reward = self.account.total_assets - Self::STARTING_CASH;
        // Increase/decrease in value of positions
        // let reward = self.account.positions.iter().enumerate().map(|(index, position)| position.appreciation(self.prices[index][self.step])).sum();

        self.episode_history.rewards.push(reward);

        // Done

        let is_done = self.step + 2 > self.prices[0].len();

        if is_done {
            println!(
                "Episode {} - Total Assets: {:.2} cumulative reward {:.2} tickers {:?} time secs {}",
                self.episode,
                self.account.total_assets,
                self.episode_history.rewards.iter().sum::<f64>(),
                self.tickers,
                Instant::now().duration_since(self.episode_start).as_secs()
            );

            self.episode_history
                .record(self.episode, &self.tickers, &self.prices);
            self.meta_history.record(&self.episode_history);

            if self.episode % 5 == 0 {
                self.meta_history.chart(self.episode);
            }

            self.episode_start = Instant::now();

            self.episode += 1;
        }
        return Snapshot::new(self.state, reward as ElemType, is_done);
    }

    fn reset(&mut self) -> Snapshot<Self> {
        self.tickers = vec![vec!["TSLA", "AAPL", "AMD", "INTC", "MSFT"]
            .choose(&mut self.rng)
            .unwrap()
            .to_string()];
        let mapped_bars = get_historical_data(Some(
            &self
                .tickers
                .iter()
                .map(|ticker| ticker.as_str())
                .collect::<Vec<&str>>(),
        ));
        self.prices = mapped_bars
            .iter()
            .map(|bar| bar.iter().map(|bar| bar.close).collect())
            .collect();
        self.price_deltas = get_mapped_price_deltas(&mapped_bars);

        self.account = Account::new(Self::STARTING_CASH, self.tickers.len());
        self.step = Self::STARTING_STEP;

        self.episode_history = EpisodeHistory::new(self.tickers.len());
        self.state = ObservationState::new_random();

        Snapshot::new(self.state, 0.0, false)
    }

    fn state(&self) -> Self::StateType {
        self.state
    }
}
