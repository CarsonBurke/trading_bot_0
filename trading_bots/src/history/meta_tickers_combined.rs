use crate::charts::general::{simple_chart, simple_chart_log, multi_line_chart};
use crate::history::episode_tickers_combined::EpisodeHistory;
use crate::constants::files::TRAINING_PATH;
use crate::utils::create_folder_if_not_exists;

#[derive(Default, Debug)]
pub struct MetaHistory {
    pub final_assets: Vec<f64>,
    pub cumulative_reward: Vec<f64>,
    pub outperformance: Vec<f64>,
    pub loss: Vec<f64>,
    pub grad_norm: Vec<f64>,
    pub total_commissions: Vec<f64>,
    pub mean_std: Vec<f64>,
    pub min_std: Vec<f64>,
    pub max_std: Vec<f64>,
    pub mean_advantage: Vec<f64>,
    pub min_advantage: Vec<f64>,
    pub max_advantage: Vec<f64>,
    pub mean_divisor: Vec<f64>,
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

    pub fn record_grad_norm(&mut self, grad_norm: f64) {
        self.grad_norm.push(grad_norm);
    }

    pub fn record_std_stats(&mut self, mean_std: f64, min_std: f64, max_std: f64) {
        self.mean_std.push(mean_std);
        self.min_std.push(min_std);
        self.max_std.push(max_std);
    }

    pub fn record_advantage_stats(&mut self, mean: f64, min: f64, max: f64) {
        self.mean_advantage.push(mean);
        self.min_advantage.push(min);
        self.max_advantage.push(max);
    }

    pub fn record_divisor(&mut self, mean_divisor: f64) {
        self.mean_divisor.push(mean_divisor);
    }

    pub fn chart(&self, episode: usize) {
        let base_dir = format!("{TRAINING_PATH}/gens/{}", episode);
        create_folder_if_not_exists(&base_dir);
        if !self.final_assets.is_empty() {
            let _ = simple_chart(&base_dir, "final_assets", &self.final_assets);
        }
        if !self.cumulative_reward.is_empty() {
            let _ = simple_chart(&base_dir, "cumulative_reward", &self.cumulative_reward);
        }
        if !self.outperformance.is_empty() {
            let _ = simple_chart(&base_dir, "outperformance", &self.outperformance);
        }
        if !self.loss.is_empty() {
            let _ = simple_chart_log(&base_dir, "loss (log scale)", &self.loss, "Episode");
        }
        if !self.grad_norm.is_empty() {
            let _ = simple_chart_log(&base_dir, "grad_norm (log scale)", &self.grad_norm, "Episode");
        }
        if !self.total_commissions.is_empty() {
            let _ = simple_chart(&base_dir, "total_commissions", &self.total_commissions);
        }
        if !self.mean_std.is_empty() {
            let _ = multi_line_chart(
                &base_dir,
                "std_stats",
                &[
                    ("mean", &self.mean_std),
                    ("min", &self.min_std),
                    ("max", &self.max_std),
                ],
                1,
                "Episode",
            );
        }
        if !self.mean_advantage.is_empty() {
            let _ = multi_line_chart(
                &base_dir,
                "advantage_stats",
                &[
                    ("mean", &self.mean_advantage),
                    ("min", &self.min_advantage),
                    ("max", &self.max_advantage),
                ],
                1,
                "Episode",
            );
        }
        if !self.mean_divisor.is_empty() {
            let _ = simple_chart(&base_dir, "divisor", &self.mean_divisor);
        }
    }
}
