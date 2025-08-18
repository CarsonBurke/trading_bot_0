use crate::{
    burn::{
        action::TradeAction,
        agent::base::{Action, Environment, Snapshot},
        obs_state::ObservationState,
    }, constants::TICKERS, data::historical::get_historical_data, history::{episode_tickers_combined::EpisodeHistory, meta_tickers_combined::MetaHistory}, types::Account, utils::{get_mapped_price_deltas, get_price_deltas}
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
}

impl Env {
    const STARTING_STEP: usize = 500;
    const BUY_PERCENT: f64 = 0.05;
    const SELL_PERCENT: f64 = 0.05;
}

impl Environment for Env {
    type ActionType = TradeAction;
    type StateType = ObservationState;
    type RewardType = f64;

    fn new(visualized: bool) -> Self {
        let mapped_bars = get_historical_data();
        let price_deltas = get_mapped_price_deltas(&mapped_bars);
        
        let tickers = vec![
            // "SPY".to_string(),
            // "TSLA".to_string(),
            // "AAPL".to_string(),
            // "MSFT".to_string(),
            // "AMD".to_string(),
            // "INTC".to_string(),
            "NVDA".to_string(),
        ];

        Self {
            step: Self::STARTING_STEP,
            episode: 0,
            account: Account::new(10_000., tickers.len()),
            episode_history: EpisodeHistory::new(tickers.len()),
            meta_history: MetaHistory::default(),
            tickers,
            prices: mapped_bars
                .iter()
                .map(|bar| bar.iter().map(|bar| bar.close).collect())
                .collect(),
            price_deltas,
            state: ObservationState::new_empty(),
            visualized,
        }
    }

    fn step(&mut self, action: Self::ActionType) -> Snapshot<Self> {
        let current_prices = &self.prices[self.step];
        self.account.update_total(current_prices);
        let total_assets = self.account.total_assets;
        
        match action {
            TradeAction::Buy => {
                let buy_total = (total_assets * Self::BUY_PERCENT).min(self.account.cash);
                
                if buy_total > 0.0 {
                    self.account.cash -= buy_total;
                    
                    let quantity = buy_total / current_prices[0];
                    self.account.positions[0].add(current_prices[0], quantity);
                    
                    self.episode_history.buys[0].push(current_prices[0]);
                }
            }
            TradeAction::Sell => {
                let sell_total = (total_assets * Self::SELL_PERCENT).min(self.account.positions[0].value_with_price(current_prices[0]));
                
                if sell_total > 0.0 {
                    self.account.cash += sell_total;
                    
                    let quantity = sell_total / current_prices[0];
                    self.account.positions[0].quantity -= quantity;
                    
                    self.episode_history.sells[0].push(current_prices[0]);
                }
            }
            TradeAction::Hold => {
                
            }
        }
        
        self.episode_history.positioned.push(self.account.position_values(current_prices));
        
        // Next step

        self.step += 1;
        let next_prices = &self.prices[self.step];
        
        self.state = ObservationState::new(self.step, &self.account, next_prices, &self.price_deltas);
        
        // Reward
        
        self.account.update_total(next_prices);
        let portfolio_delta_percent = (self.account.total_assets - total_assets) / total_assets;
        let reward = portfolio_delta_percent;
        
        self.episode_history.rewards.push(reward);
        
        // Done
        
        let is_done = self.step > self.prices.len();
        if is_done {
            println!("Episode {} - Total Assets: {}", self.episode, self.account.total_assets);
            
            self.episode_history.record(self.episode, &self.tickers, &self.prices);
            self.meta_history.record(&self.episode_history);
            self.episode += 1;
        }
        return Snapshot::new(self.state, 0.0, is_done);
    }

    fn reset(&mut self) -> Snapshot<Self> {
        self.episode_history = EpisodeHistory::new(self.tickers.len());
        self.state = ObservationState::new_empty();
        
        Snapshot::new(self.state, 0.0, true)
    }

    fn state(&self) -> Self::StateType {
        self.state
    }
}
