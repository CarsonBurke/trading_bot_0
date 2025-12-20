use hashbrown::HashMap;

use crate::constants::files::TRAINING_PATH;
use crate::history::report::{write_report, Report, ReportKind, ReportSeries, ScaleKind, TradePoint};
use crate::utils::create_folder_if_not_exists;

#[derive(Debug)]
pub struct EpisodeHistory {
    pub buys: Vec<HashMap<usize, (f64, f64)>>,
    pub sells: Vec<HashMap<usize, (f64, f64)>>,
    pub positioned: Vec<Vec<f64>>,
    pub cash: Vec<f64>,
    pub rewards: Vec<f64>,
    pub raw_actions: Vec<Vec<f64>>,
    pub total_commissions: f64,
    pub static_observations: Vec<Vec<f32>>,
    pub attention_weights: Vec<Vec<f32>>,
    pub target_weights: Vec<Vec<f64>>,
    pub cash_weight: Vec<f64>,
    pub action_step0: Option<Vec<f64>>,
    pub action_final: Option<Vec<f64>>,
}

impl EpisodeHistory {
    pub fn new(ticker_count: usize) -> Self {
        EpisodeHistory {
            buys: vec![HashMap::new(); ticker_count],
            sells: vec![HashMap::new(); ticker_count],
            positioned: vec![vec![]; ticker_count],
            cash: Vec::new(),
            rewards: Vec::new(),
            raw_actions: vec![vec![]; ticker_count],
            total_commissions: 0.0,
            static_observations: Vec::new(),
            attention_weights: Vec::new(),
            target_weights: vec![vec![]; ticker_count],
            cash_weight: Vec::new(),
            action_step0: None,
            action_final: None,
        }
    }

    pub fn record(&self, episode: usize, tickers: &[String], prices: &[Vec<f64>], start_offset: usize) {
        self.record_to_path(&format!("{TRAINING_PATH}/gens"), episode, tickers, prices, start_offset);
    }

    pub fn record_to_path(&self, base_path: &str, episode: usize, tickers: &[String], prices: &[Vec<f64>], start_offset: usize) {
        let episode_dir = format!("{}/{}", base_path, episode);
        create_folder_if_not_exists(&episode_dir);

        let num_steps = self.cash.len();
        let mut total_assets_per_step = vec![0.0; num_steps];
        for ticker_positioned in &self.positioned {
            for (step, &value) in ticker_positioned.iter().enumerate() {
                total_assets_per_step[step] += value;
            }
        }
        for (step, &cash) in self.cash.iter().enumerate() {
            total_assets_per_step[step] += cash;
        }

        let index_benchmark = if !prices.is_empty() && num_steps > 0 {
            let initial_value = total_assets_per_step[0];
            let mut benchmark = vec![initial_value];

            for step in 1..num_steps {
                let abs_step = start_offset + step;
                let prev_abs_step = start_offset + step - 1;
                let mut step_return = 0.0;
                for ticker_prices in prices {
                    if abs_step < ticker_prices.len() {
                        step_return += ticker_prices[abs_step] / ticker_prices[prev_abs_step];
                    }
                }
                step_return /= prices.len() as f64;

                let new_value = benchmark.last().unwrap() * step_return;
                benchmark.push(new_value);
            }
            Some(benchmark)
        } else {
            None
        };

        for (ticker_index, ticker_prices) in prices.iter().enumerate() {
            let ticker = &tickers[ticker_index];
            let ticker_dir = format!("{}/{}/{ticker}", base_path, episode);
            create_folder_if_not_exists(&ticker_dir);

            let ticker_buy_indexes = &self.buys[ticker_index];
            let ticker_sell_indexes = &self.sells[ticker_index];
            let buys: Vec<TradePoint> = ticker_buy_indexes
                .iter()
                .map(|(index, (price, quantity))| TradePoint {
                    index: *index,
                    price: *price,
                    quantity: *quantity,
                })
                .collect();
            let sells: Vec<TradePoint> = ticker_sell_indexes
                .iter()
                .map(|(index, (price, quantity))| TradePoint {
                    index: *index,
                    price: *price,
                    quantity: *quantity,
                })
                .collect();
            let report = Report {
                title: "Buy Sell".to_string(),
                x_label: Some("Step".to_string()),
                y_label: Some("Price".to_string()),
                scale: ScaleKind::Linear,
                kind: ReportKind::BuySell {
                    prices: ticker_prices.clone(),
                    buys,
                    sells,
                },
            };
            let _ = write_report(&format!("{ticker_dir}/buy_sell.report.bin"), &report);

            let positioned_assets = &self.positioned[ticker_index];

            let ticker_benchmark = if !ticker_prices.is_empty() && num_steps > 0 && start_offset < ticker_prices.len() {
                let initial_value = total_assets_per_step[0];
                let initial_price = ticker_prices[start_offset];
                let end_idx = (start_offset + num_steps).min(ticker_prices.len());
                Some(
                    ticker_prices[start_offset..end_idx]
                        .iter()
                        .map(|&current_price| initial_value * current_price / initial_price)
                        .collect()
                )
            } else {
                None
            };

            let report = Report {
                title: "Assets".to_string(),
                x_label: Some("Step".to_string()),
                y_label: Some("Assets".to_string()),
                scale: ScaleKind::Linear,
                kind: ReportKind::Assets {
                    total: total_assets_per_step.clone(),
                    cash: self.cash.clone(),
                    positioned: Some(positioned_assets.clone()),
                    benchmark: ticker_benchmark,
                },
            };
            let _ = write_report(&format!("{ticker_dir}/assets.report.bin"), &report);

            let report = Report {
                title: "Raw Action".to_string(),
                x_label: Some("Step".to_string()),
                y_label: None,
                scale: ScaleKind::Linear,
                kind: ReportKind::Simple {
                    values: self.raw_actions[ticker_index].clone(),
                    ema_alpha: None,
                },
            };
            let _ = write_report(&format!("{ticker_dir}/raw_action.report.bin"), &report);
        }

        let mut positioned_assets_per_step = vec![0.0; num_steps];
        for ticker_positioned in &self.positioned {
            for (step, &value) in ticker_positioned.iter().enumerate() {
                positioned_assets_per_step[step] += value;
            }
        }

        let report = Report {
            title: "Assets".to_string(),
            x_label: Some("Step".to_string()),
            y_label: Some("Assets".to_string()),
            scale: ScaleKind::Linear,
            kind: ReportKind::Assets {
                total: total_assets_per_step.clone(),
                cash: self.cash.clone(),
                positioned: Some(positioned_assets_per_step),
                benchmark: index_benchmark,
            },
        };
        let _ = write_report(&format!("{episode_dir}/assets.report.bin"), &report);

        let report = Report {
            title: "Rewards".to_string(),
            x_label: Some("Step".to_string()),
            y_label: Some("Reward".to_string()),
            scale: ScaleKind::Linear,
            kind: ReportKind::Simple {
                values: self.rewards.clone(),
                ema_alpha: None,
            },
        };
        let _ = write_report(&format!("{episode_dir}/reward.report.bin"), &report);

        // Combined target weights chart (all tickers + cash) - every 5 episodes like meta charts
        if episode % 5 == 0 && !self.cash_weight.is_empty() && self.target_weights.iter().any(|w| !w.is_empty()) {
            let mut series: Vec<ReportSeries> = Vec::new();
            for (ticker_index, ticker) in tickers.iter().enumerate() {
                if !self.target_weights[ticker_index].is_empty() {
                    series.push(ReportSeries {
                        label: ticker.to_string(),
                        values: self.target_weights[ticker_index].clone(),
                    });
                }
            }
            series.push(ReportSeries {
                label: "cash".to_string(),
                values: self.cash_weight.clone(),
            });
            let report = Report {
                title: "Target Weights".to_string(),
                x_label: Some("Step".to_string()),
                y_label: None,
                scale: ScaleKind::Linear,
                kind: ReportKind::MultiLine { series },
            };
            let _ = write_report(&format!("{episode_dir}/target_weights.report.bin"), &report);
        }

        // Write static observations and attention weights
        if episode & 5 == 0 && !self.static_observations.is_empty() && !self.attention_weights.is_empty() {
            let report = Report {
                title: "Observations".to_string(),
                x_label: None,
                y_label: None,
                scale: ScaleKind::Linear,
                kind: ReportKind::Observations {
                    static_observations: self.static_observations.clone(),
                    attention_weights: self.attention_weights.clone(),
                    action_step0: self.action_step0.clone(),
                    action_final: self.action_final.clone(),
                },
            };
            let _ = write_report(&format!("{episode_dir}/observations.report.bin"), &report);
        }
    }

    pub fn final_assets(&self) -> f64 {
        let positioned = self.positioned.iter().map(|p| p.last().unwrap()).sum::<f64>();
        positioned + self.cash.last().unwrap()
    }
}
