use crate::charts::general::simple_chart;
use crate::history::episode::EpisodeHistory;
use crate::constants::files::TRAINING_PATH;
use crate::utils::create_folder_if_not_exists;

#[derive(Default)]
pub struct MetaHistory {
    pub min_assets: Vec<f64>,
    pub avg_assets: Vec<f64>,
}

impl MetaHistory {
    pub fn record(&mut self, history: &EpisodeHistory) {
        self.min_assets.push(history.ticker_lowest_final_assets().1);
        self.avg_assets.push(history.avg_final_assets());
    }

    pub fn chart(&self, generation: u32) {
        let base_dir = format!("{TRAINING_PATH}/gens/{}", generation);
        create_folder_if_not_exists(&base_dir);
        let _ = simple_chart(&base_dir, "min_assets", &self.min_assets);
        let _ = simple_chart(&base_dir, "avg_assets", &self.avg_assets);
    }
}