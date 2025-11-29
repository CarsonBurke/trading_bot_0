use hashbrown::HashMap;
use std::fs::File;
use std::io::Write;

use crate::{
    charts::general::{assets_chart, buy_sell_chart, buy_sell_chart_vec, hold_action_chart, raw_action_chart, reward_chart},
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
    pub hold_actions: Vec<Vec<f64>>,
    pub raw_actions: Vec<Vec<f64>>,
    pub total_commissions: f64,
    pub static_observations: Vec<Vec<f32>>,
    pub attention_weights: Vec<Vec<f32>>,
}

impl EpisodeHistory {
    pub fn new(ticker_count: usize) -> Self {
        EpisodeHistory {
            buys: vec![HashMap::new(); ticker_count],
            sells: vec![HashMap::new(); ticker_count],
            positioned: vec![vec![]; ticker_count],
            cash: Vec::new(),
            rewards: Vec::new(),
            hold_actions: vec![vec![]; ticker_count],
            raw_actions: vec![vec![]; ticker_count],
            total_commissions: 0.0,
            static_observations: Vec::new(),
            attention_weights: Vec::new(),
        }
    }

    pub fn record(&self, episode: usize, tickers: &[String], prices: &[Vec<f64>]) {
        self.record_to_path(&format!("{TRAINING_PATH}/gens"), episode, tickers, prices);
    }

    pub fn record_to_path(&self, base_path: &str, episode: usize, tickers: &[String], prices: &[Vec<f64>]) {
        let episode_dir = format!("{}/{}", base_path, episode);
        create_folder_if_not_exists(&episode_dir);

        for (ticker_index, prices) in prices.iter().enumerate() {
            let ticker = &tickers[ticker_index];
            let ticker_dir = format!("{}/{}/{ticker}", base_path, episode);
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

            let benchmark = if !prices.is_empty() && !total_assets.is_empty() {
                let initial_account_value = total_assets[0];
                let initial_price = prices[0];
                Some(
                    prices[..total_assets.len()]
                        .iter()
                        .map(|&current_price| initial_account_value * current_price / initial_price)
                        .collect()
                )
            } else {
                None
            };

            let _ = assets_chart(
                &ticker_dir,
                &total_assets,
                &self.cash,
                Some(positioned_assets),
                benchmark.as_ref(),
            );

            let _ = hold_action_chart(
                &ticker_dir,
                &self.hold_actions[ticker_index],
            );

            let _ = raw_action_chart(
                &ticker_dir,
                &self.raw_actions[ticker_index],
            );
        }

        let num_steps = self.cash.len();
        let mut positioned_assets_per_step = vec![0.0; num_steps];
        for ticker_positioned in &self.positioned {
            for (step, &value) in ticker_positioned.iter().enumerate() {
                positioned_assets_per_step[step] += value;
            }
        }

        let total_assets = positioned_assets_per_step
            .iter()
            .zip(self.cash.iter())
            .map(|(positioned, cash)| positioned + cash)
            .collect::<Vec<f64>>();

        let _ = assets_chart(
            &episode_dir,
            &total_assets,
            &self.cash,
            Some(&positioned_assets_per_step),
            None,
        );

        let _ = reward_chart(&episode_dir, &self.rewards);

        // Write static observations and attention weights
        if !self.static_observations.is_empty() && !self.attention_weights.is_empty() {
            let observations_path = format!("{}/observations.json", episode_dir);
            if let Ok(mut file) = File::create(&observations_path) {
                let json_data = serde_json::json!({
                    "static_observations": self.static_observations,
                    "attention_weights": self.attention_weights,
                });
                let _ = file.write_all(json_data.to_string().as_bytes());
            }
        }
    }

    pub fn final_assets(&self) -> f64 {
        let positioned = self.positioned.iter().map(|p| p.last().unwrap()).sum::<f64>();
        positioned + self.cash.last().unwrap()
    }
}
