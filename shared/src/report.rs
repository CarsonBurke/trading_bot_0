use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub title: String,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    pub scale: ScaleKind,
    pub kind: ReportKind,
    #[serde(default)]
    pub x_offset: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSeries {
    pub label: String,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradePoint {
    pub index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportKind {
    Simple {
        values: Vec<f32>,
        ema_alpha: Option<f64>,
    },
    MultiLine {
        series: Vec<ReportSeries>,
    },
    Assets {
        total: Vec<f32>,
        cash: Vec<f32>,
        positioned: Option<Vec<f32>>,
        benchmark: Option<Vec<f32>>,
    },
    BuySell {
        prices: Vec<f32>,
        buys: Vec<TradePoint>,
        sells: Vec<TradePoint>,
    },
    Observations {
        static_observations: Vec<Vec<f32>>,
        attention_weights: Vec<Vec<f32>>,
        action_step0: Option<Vec<f32>>,
        action_final: Option<Vec<f32>>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ScaleKind {
    Linear,
    Symlog,
}

impl ReportKind {
    pub fn to_lines(&self) -> Vec<String> {
        match self {
            ReportKind::Simple { values, .. } => values
                .iter()
                .enumerate()
                .map(|(i, v)| format!("{i}\t{v}"))
                .collect(),
            ReportKind::MultiLine { series } => {
                let max_len = series.iter().map(|s| s.values.len()).max().unwrap_or(0);
                let mut lines = Vec::with_capacity(max_len);
                for i in 0..max_len {
                    let mut line = format!("{i}");
                    for s in series {
                        if let Some(v) = s.values.get(i) {
                            line.push('\t');
                            line.push_str(&s.label);
                            line.push('=');
                            line.push_str(&v.to_string());
                        }
                    }
                    lines.push(line);
                }
                lines
            }
            ReportKind::Assets {
                total,
                cash,
                positioned,
                benchmark,
            } => {
                let max_len = total.len().max(cash.len());
                let mut lines = Vec::with_capacity(max_len);
                for i in 0..max_len {
                    let mut line = format!("{i}");
                    if let Some(v) = total.get(i) {
                        line.push_str(&format!("\ttotal={v}"));
                    }
                    if let Some(v) = cash.get(i) {
                        line.push_str(&format!("\tcash={v}"));
                    }
                    if let Some(pos) = positioned.as_ref().and_then(|p| p.get(i)) {
                        line.push_str(&format!("\tpositioned={pos}"));
                    }
                    if let Some(bench) = benchmark.as_ref().and_then(|b| b.get(i)) {
                        line.push_str(&format!("\tbenchmark={bench}"));
                    }
                    lines.push(line);
                }
                lines
            }
            ReportKind::BuySell {
                prices,
                buys,
                sells,
            } => {
                let mut buy_map: std::collections::HashSet<usize> =
                    std::collections::HashSet::new();
                let mut sell_map: std::collections::HashSet<usize> =
                    std::collections::HashSet::new();
                for b in buys {
                    buy_map.insert(b.index as usize);
                }
                for s in sells {
                    sell_map.insert(s.index as usize);
                }
                let mut lines = Vec::with_capacity(prices.len());
                for (i, price) in prices.iter().enumerate() {
                    let mut line = format!("{i}\tprice={price}");
                    if buy_map.contains(&i) {
                        line.push_str("\tbuy=1");
                    }
                    if sell_map.contains(&i) {
                        line.push_str("\tsell=1");
                    }
                    lines.push(line);
                }
                lines
            }
            ReportKind::Observations {
                static_observations,
                attention_weights,
                action_step0,
                action_final,
            } => {
                let mut lines = Vec::new();
                if let Some(action) = action_step0 {
                    lines.push(format!("action_step0\t{}", format_vec_f32(action)));
                }
                if let Some(action) = action_final {
                    lines.push(format!("action_final\t{}", format_vec_f32(action)));
                }
                for (i, obs) in static_observations.iter().enumerate() {
                    lines.push(format!("static\t{i}\t{}", format_vec_f32(obs)));
                }
                for (i, attn) in attention_weights.iter().enumerate() {
                    lines.push(format!("attn\t{i}\t{}", format_vec_f32(attn)));
                }
                lines
            }
        }
    }
}

fn format_vec_f32(values: &[f32]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}
