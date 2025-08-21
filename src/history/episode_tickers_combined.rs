use hashbrown::HashMap;

use crate::{
    charts::general::{assets_chart, buy_sell_chart, buy_sell_chart_vec, reward_chart},
    constants::{files::TRAINING_PATH, TICKERS},
    types::MappedHistorical,
    utils::create_folder_if_not_exists,
};

#[derive(Debug)]
pub struct EpisodeHistory {
    pub buys: Vec<HashMap<usize, (f64, f64)>>,
    pub sells: Vec<HashMap<usize, (f64, f64)>>,
    pub positioned: Vec<Vec<f64>>,
    pub cash: Vec<f64>,
    pub rewards: Vec<f64>,
}

impl EpisodeHistory {
    pub fn new(ticker_count: usize) -> Self {
        EpisodeHistory {
            buys: vec![HashMap::new(); ticker_count],
            sells: vec![HashMap::new(); ticker_count],
            positioned: vec![vec![]; ticker_count],
            cash: Vec::new(),
            rewards: Vec::new(),
        }
    }

    pub fn record(&self, episode: usize, tickers: &[String], prices: &[Vec<f64>]) {
        let episode_dir = format!("{TRAINING_PATH}/gens/{}", episode);
        create_folder_if_not_exists(&episode_dir);

        for (ticker_index, prices) in prices.iter().enumerate() {
            let ticker = &tickers[ticker_index];
            let ticker_dir = format!("{TRAINING_PATH}/gens/{}/{ticker}", episode);
            create_folder_if_not_exists(&ticker_dir);

            let ticker_buy_indexes = &self.buys[ticker_index];
            let ticker_sell_indexes = &self.sells[ticker_index];
            let _ = buy_sell_chart(
                &ticker_dir,
                &prices,
                ticker_buy_indexes,
                ticker_sell_indexes,
            );

            let positioned_assets = &self.positioned[ticker_index];
            let total_assets = positioned_assets
                .iter()
                .zip(self.cash.iter())
                .map(|(positioned, cash)| positioned + cash)
                .collect::<Vec<f64>>();

            let _ = assets_chart(
                &ticker_dir,
                &total_assets,
                &self.cash,
                Some(positioned_assets),
            );
        }

        let positioned_assets = &self
            .positioned
            .iter()
            .map(|positioned| positioned.iter().sum())
            .collect::<Vec<f64>>();
        let total_assets = positioned_assets
            .iter()
            .zip(self.cash.iter())
            .map(|(positioned, cash)| positioned + cash)
            .collect::<Vec<f64>>();

        let _ = assets_chart(
            &episode_dir,
            &total_assets,
            &self.cash,
            Some(positioned_assets),
        );

        let _ = reward_chart(&episode_dir, &self.rewards);
    }

    pub fn final_assets(&self) -> f64 {
        let positioned = self.positioned.iter().map(|p| p.last().unwrap()).sum::<f64>();
        positioned + self.cash.last().unwrap()
    }
}
