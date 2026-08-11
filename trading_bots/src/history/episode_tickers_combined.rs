use anyhow::{Context, Result};
use hashbrown::HashMap;
use std::fs;

use crate::constants::files::TRAINING_PATH;
use crate::history::report::{
    write_report, Report, ReportKind, ReportSeries, ScaleKind, TradePoint,
};
use shared::constants::{GLOBAL_STATIC_OBS, PER_TICKER_STATIC_OBS};

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
    pub observation_tickers: Vec<String>,
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
            observation_tickers: Vec::new(),
            attention_weights: Vec::new(),
            target_weights: vec![vec![]; ticker_count],
            cash_weight: Vec::new(),
            action_step0: None,
            action_final: None,
        }
    }

    pub fn record(
        &self,
        episode: usize,
        tickers: &[String],
        prices: &[Vec<f64>],
        start_offset: usize,
    ) -> Result<()> {
        self.record_to_path(
            &format!("{TRAINING_PATH}/gens"),
            episode,
            tickers,
            prices,
            start_offset,
        )
    }

    pub fn record_to_path(
        &self,
        base_path: &str,
        episode: usize,
        tickers: &[String],
        prices: &[Vec<f64>],
        start_offset: usize,
    ) -> Result<()> {
        let episode_dir = format!("{}/{}", base_path, episode);
        fs::create_dir_all(&episode_dir)
            .with_context(|| format!("failed creating episode report directory {episode_dir}"))?;
        let sleeve_cash = equal_cash_sleeve_curve(&self.cash, tickers.len());

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
            fs::create_dir_all(&ticker_dir)
                .with_context(|| format!("failed creating ticker report directory {ticker_dir}"))?;

            let ticker_buy_indexes = &self.buys[ticker_index];
            let ticker_sell_indexes = &self.sells[ticker_index];
            let buys: Vec<TradePoint> = ticker_buy_indexes
                .keys()
                .map(|index| TradePoint {
                    index: *index as u32,
                })
                .collect();
            let sells: Vec<TradePoint> = ticker_sell_indexes
                .keys()
                .map(|index| TradePoint {
                    index: *index as u32,
                })
                .collect();
            let report = Report {
                title: "Buy Sell".to_string(),
                x_label: Some("Step".to_string()),
                y_label: Some("Price".to_string()),
                scale: ScaleKind::Linear,
                kind: ReportKind::BuySell {
                    prices: f64_to_f32(ticker_prices),
                    buys,
                    sells,
                },
            };
            write_report(format!("{ticker_dir}/buy_sell.report.bin"), &report)?;

            let positioned_assets = &self.positioned[ticker_index];
            let sleeve_total = sleeve_cash
                .iter()
                .zip(positioned_assets.iter())
                .map(|(cash, positioned)| cash + positioned)
                .collect::<Vec<_>>();

            let ticker_benchmark =
                if !ticker_prices.is_empty() && num_steps > 0 && start_offset < ticker_prices.len()
                {
                    let initial_value = sleeve_total.first().copied().unwrap_or_else(|| {
                        self.cash.first().copied().unwrap_or(0.0) / tickers.len().max(1) as f64
                    });
                    let initial_price = ticker_prices[start_offset];
                    let end_idx = (start_offset + num_steps).min(ticker_prices.len());
                    Some(
                        ticker_prices[start_offset..end_idx]
                            .iter()
                            .map(|&current_price| initial_value * current_price / initial_price)
                            .collect::<Vec<f64>>(),
                    )
                } else {
                    None
                };

            let report = Report {
                title: "Sleeve Assets".to_string(),
                x_label: Some("Step".to_string()),
                y_label: Some("Assets".to_string()),
                scale: ScaleKind::Linear,
                kind: ReportKind::Assets {
                    total: f64_to_f32(&sleeve_total),
                    cash: f64_to_f32(&sleeve_cash),
                    positioned: Some(f64_to_f32(positioned_assets)),
                    benchmark: ticker_benchmark.as_ref().map(|b| f64_to_f32(b)),
                },
            };
            write_report(format!("{ticker_dir}/assets.report.bin"), &report)?;

            let report = Report {
                title: "Raw Action".to_string(),
                x_label: Some("Step".to_string()),
                y_label: None,
                scale: ScaleKind::Linear,
                kind: ReportKind::Simple {
                    values: f64_to_f32(&self.raw_actions[ticker_index]),
                    ema_alpha: None,
                },
            };
            write_report(format!("{ticker_dir}/raw_action.report.bin"), &report)?;
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
                total: f64_to_f32(&total_assets_per_step),
                cash: f64_to_f32(&self.cash),
                positioned: Some(f64_to_f32(&positioned_assets_per_step)),
                benchmark: index_benchmark.as_ref().map(|b| f64_to_f32(b)),
            },
        };
        write_report(format!("{episode_dir}/assets.report.bin"), &report)?;

        let report = Report {
            title: "Rewards".to_string(),
            x_label: Some("Step".to_string()),
            y_label: Some("Reward".to_string()),
            scale: ScaleKind::Linear,
            kind: ReportKind::Simple {
                values: f64_to_f32(&self.rewards),
                ema_alpha: None,
            },
        };
        write_report(format!("{episode_dir}/reward.report.bin"), &report)?;

        // Combined target weights chart (all tickers + cash) - every 5 episodes like meta charts
        if episode % 5 == 0
            && !self.cash_weight.is_empty()
            && self.target_weights.iter().any(|w| !w.is_empty())
        {
            let mut series: Vec<ReportSeries> = Vec::new();
            for (ticker_index, ticker) in tickers.iter().enumerate() {
                if !self.target_weights[ticker_index].is_empty() {
                    series.push(ReportSeries {
                        label: ticker.to_string(),
                        values: f64_to_f32(&self.target_weights[ticker_index]),
                    });
                }
            }
            series.push(ReportSeries {
                label: "cash".to_string(),
                values: f64_to_f32(&self.cash_weight),
            });
            let report = Report {
                title: "Target Weights".to_string(),
                x_label: Some("Step".to_string()),
                y_label: None,
                scale: ScaleKind::Linear,
                kind: ReportKind::MultiLine { series },
            };
            write_report(format!("{episode_dir}/target_weights.report.bin"), &report)?;
        }

        // Write static observations and attention weights
        if episode % 5 == 0 && !self.static_observations.is_empty() {
            let expected_width =
                GLOBAL_STATIC_OBS + self.observation_tickers.len() * PER_TICKER_STATIC_OBS;
            anyhow::ensure!(
                self.static_observations
                    .iter()
                    .all(|observation| observation.len() == expected_width),
                "observation history row width does not match its ticker labels"
            );
            let report = Report {
                title: "Observations".to_string(),
                x_label: None,
                y_label: None,
                scale: ScaleKind::Linear,
                kind: ReportKind::Observations {
                    observation_tickers: self.observation_tickers.clone(),
                    action_tickers: tickers.to_vec(),
                    static_observations: self.static_observations.clone(),
                    attention_weights: self.attention_weights.clone(),
                    action_step0: self.action_step0.as_ref().map(|v| f64_to_f32(v)),
                    action_final: self.action_final.as_ref().map(|v| f64_to_f32(v)),
                },
            };
            write_report(format!("{episode_dir}/observations.report.bin"), &report)?;
        }
        Ok(())
    }

    pub fn final_assets(&self) -> f64 {
        let positioned = self
            .positioned
            .iter()
            .map(|p| p.last().unwrap())
            .sum::<f64>();
        positioned + self.cash.last().unwrap()
    }
}

fn f64_to_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|v| *v as f32).collect()
}

fn equal_cash_sleeve_curve(cash: &[f64], ticker_count: usize) -> Vec<f64> {
    let divisor = ticker_count.max(1) as f64;
    cash.iter().map(|cash_value| cash_value / divisor).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::report::read_report;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn observation_report_is_persisted_without_attention_data() {
        let root = std::env::temp_dir().join(format!(
            "trading-bot-observations-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut history = EpisodeHistory::new(1);
        history.observation_tickers.push("BRK.B".to_owned());
        history
            .static_observations
            .push(vec![0.0; GLOBAL_STATIC_OBS + PER_TICKER_STATIC_OBS]);
        history
            .record_to_path(root.to_str().unwrap(), 5, &["BRK.B".to_owned()], &[], 0)
            .unwrap();

        let report = read_report(root.join("5/observations.report.bin")).unwrap();
        let ReportKind::Observations {
            observation_tickers,
            action_tickers,
            static_observations,
            attention_weights,
            ..
        } = report.kind
        else {
            panic!("expected observations report");
        };
        assert_eq!(observation_tickers, ["BRK.B"]);
        assert_eq!(action_tickers, ["BRK.B"]);
        assert_eq!(static_observations.len(), 1);
        assert!(attention_weights.is_empty());
        fs::remove_dir_all(root).unwrap();
    }
}
