use crate::charts::general::simple_chart;
use crate::history::episode_tickers_combined::EpisodeHistory;
use crate::constants::files::TRAINING_PATH;
use crate::utils::create_folder_if_not_exists;

#[derive(Default, Debug)]
pub struct MetaHistory {
    pub final_assets: Vec<f64>,
    pub cumulative_reward: Vec<f64>,
}

impl MetaHistory {
    pub fn record(&mut self, history: &EpisodeHistory) {
        self.final_assets.push(history.final_assets());
        self.cumulative_reward.push(history.rewards.iter().sum::<f64>());
    }

    pub fn chart(&self, episode: usize) {
        let base_dir = format!("{TRAINING_PATH}/gens/{}", episode);
        create_folder_if_not_exists(&base_dir);
        let _ = simple_chart(&base_dir, "final_assets", &self.final_assets);
        let _ = simple_chart(&base_dir, "cum_reward", &self.cumulative_reward);
    }
}