use crate::charts::general::simple_chart;
use crate::history::episode_tickers_combined::EpisodeHistory;
use crate::constants::files::TRAINING_PATH;
use crate::utils::create_folder_if_not_exists;

#[derive(Default, Debug)]
pub struct MetaHistory {
    pub final_assets: Vec<f64>,
    pub cumulative_reward: Vec<f64>,
    pub outperformance: Vec<f64>,
    pub loss: Vec<f64>,
}

impl MetaHistory {
    pub fn record(&mut self, history: &EpisodeHistory, outperformance: f64) {
        self.final_assets.push(history.final_assets());
        self.cumulative_reward.push(history.rewards.iter().sum::<f64>());
        self.outperformance.push(outperformance);
    }

    pub fn record_loss(&mut self, loss: f64) {
        self.loss.push(loss);
    }

    pub fn chart(&self, episode: usize) {
        let base_dir = format!("{TRAINING_PATH}/gens/{}", episode);
        create_folder_if_not_exists(&base_dir);
        if !self.final_assets.is_empty() {
            let _ = simple_chart(&base_dir, "final_assets", &self.final_assets);
        }
        if !self.cumulative_reward.is_empty() {
            let _ = simple_chart(&base_dir, "cum_reward", &self.cumulative_reward);
        }
        if !self.outperformance.is_empty() {
            let _ = simple_chart(&base_dir, "outperformance", &self.outperformance);
        }
        if !self.loss.is_empty() {
            let _ = simple_chart(&base_dir, "loss", &self.loss);
        }
    }
}