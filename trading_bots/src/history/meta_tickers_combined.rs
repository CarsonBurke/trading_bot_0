use crate::charts::general::{simple_chart, simple_chart_log};
use crate::history::episode_tickers_combined::EpisodeHistory;
use crate::constants::files::TRAINING_PATH;
use crate::utils::create_folder_if_not_exists;

#[derive(Default, Debug)]
pub struct MetaHistory {
    pub final_assets: Vec<f64>,
    pub cumulative_reward: Vec<f64>,
    pub outperformance: Vec<f64>,
    pub loss: Vec<f64>,
    pub total_commissions: Vec<f64>,
    pub mean_std: Vec<f64>,
}

impl MetaHistory {
    pub fn record(&mut self, history: &EpisodeHistory, outperformance: f64) {
        self.final_assets.push(history.final_assets());
        self.cumulative_reward.push(history.rewards.iter().sum::<f64>());
        self.outperformance.push(outperformance);
        self.total_commissions.push(history.total_commissions);
    }

    pub fn record_loss(&mut self, loss: f64) {
        self.loss.push(loss);
    }

    pub fn record_mean_std(&mut self, mean_std: f64) {
        self.mean_std.push(mean_std);
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
            let _ = simple_chart_log(&base_dir, "loss (log scale)", &self.loss);
        }
        if !self.total_commissions.is_empty() {
            let _ = simple_chart(&base_dir, "total_commissions", &self.total_commissions);
        }
        if !self.mean_std.is_empty() {
            let _ = simple_chart(&base_dir, "mean_std", &self.mean_std);
        }
    }
}