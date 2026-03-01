use hashbrown::HashMap;

use crate::{
    charts::{assets_chart, buy_sell_chart, reward_chart},
    constants::{files::TRAINING_PATH, TICKERS},
    types::MappedHistorical,
    utils::create_folder_if_not_exists,
};

#[derive(Debug)]
pub struct EpisodeHistory {
    pub buys: Vec<HashMap<usize, (f64, f64)>>,
    pub sells: Vec<HashMap<usize, (f64, f64)>>,
    pub positioned: Vec<Vec<f64>>,
    pub cash: Vec<Vec<f64>>,
    pub rewards: Vec<Vec<f64>>,
}

impl EpisodeHistory {
    pub fn new(ticker_count: usize) -> Self {
        EpisodeHistory {
            buys: vec![HashMap::new(); ticker_count],
            sells: vec![HashMap::new(); ticker_count],
            positioned: vec![vec![]; ticker_count],
            cash: vec![vec![]; ticker_count],
            rewards: vec![vec![]; ticker_count],
        }
    }

    pub fn record(&self, generation: u32, mapped_historical: &MappedHistorical) {
        let base_dir = format!("{TRAINING_PATH}/gens/{}", generation);
        create_folder_if_not_exists(&base_dir);

        for (ticker_index, bars) in mapped_historical.iter().enumerate() {
            let prices = bars.iter().map(|bar| bar.close).collect::<Vec<f64>>();

            let ticker = TICKERS[ticker_index].to_string();
            let ticker_dir = format!("{TRAINING_PATH}/gens/{}/{ticker}", generation);
            create_folder_if_not_exists(&ticker_dir);

            let ticker_buy_indexes = &self.buys[ticker_index];
            let ticker_sell_indexes = &self.sells[ticker_index];
            let _ = buy_sell_chart(
                &ticker_dir,
                &prices,
                ticker_buy_indexes,
                ticker_sell_indexes,
            );

            let ticker_reward_indexes = &self.rewards[ticker_index];
            let _ = reward_chart(&ticker_dir, ticker_reward_indexes);

            let positioned_assets = &self.positioned[ticker_index];
            let cash_indexes = &self.cash[ticker_index];
            let total_assets = positioned_assets
                .iter()
                .zip(cash_indexes.iter())
                .map(|(a, b)| a + b)
                .collect::<Vec<f64>>();

            let _ = assets_chart(
                &ticker_dir,
                &total_assets,
                &cash_indexes,
                Some(positioned_assets),
                None,
            );
        }
    }

    /// The ticker with the lowest final assets with the assets amount
    pub fn ticker_lowest_final_assets(&self) -> (usize, f64) {
        self.cash
            .iter()
            .enumerate()
            .zip(self.positioned.iter())
            .map(|((cash_index, cash), positioned)| {
                (
                    cash_index,
                    cash.last().unwrap() + positioned.last().unwrap(),
                )
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Avg final assets across participating tickers
    pub fn avg_final_assets(&self) -> f64 {
        self.cash
            .iter()
            .zip(self.positioned.iter())
            .map(|(cash, positioned)| cash.last().unwrap() + positioned.last().unwrap())
            .sum::<f64>()
            / self.cash.len() as f64
    }
}
