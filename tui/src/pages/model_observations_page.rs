use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};
use serde_json::Value;
use shared::constants::{GLOBAL_STATIC_OBS, PER_TICKER_STATIC_OBS, TICKERS_COUNT};
use std::fs;

use crate::{components::episode_status, theme, App};

pub fn render(f: &mut Frame, app: &mut App) {
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(4),
        ])
        .split(f.area());

    let content_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(30),
            Constraint::Percentage(45),
            Constraint::Percentage(25),
        ])
        .split(main_chunks[1]);

    let chunks = main_chunks;

    let is_training = app.is_training_running();
    let current_episode = app.get_current_episode();

    let mut title_spans = vec![Span::styled(
        " Model Observations ",
        Style::default()
            .fg(theme::MAUVE)
            .add_modifier(Modifier::BOLD),
    )];
    title_spans.extend(episode_status::episode_status_spans(
        is_training,
        current_episode,
    ));

    let title = Paragraph::new(Line::from(title_spans)).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::LAVENDER)),
    );
    f.render_widget(title, chunks[0]);

    let latest_episode = app.generation_browser.generations.first();

    let left_block = Block::default()
        .borders(Borders::ALL)
        .title("Global Observations")
        .border_style(Style::default().fg(theme::SURFACE2));

    let right_block = Block::default()
        .borders(Borders::ALL)
        .title("Per-Ticker Observations")
        .border_style(Style::default().fg(theme::SURFACE2));

    if let Some(gen) = latest_episode {
        let obs_path = format!("../training/gens/{}/observations.json", gen.number);

        if let Ok(contents) = fs::read_to_string(&obs_path) {
            if let Ok(json) = serde_json::from_str::<Value>(&contents) {
                let static_obs = json["static_observations"].as_array();
                let action_step0 = json["action_step0"].as_array();
                let action_final = json["action_final"].as_array();

                if let Some(obs_arr) = static_obs {
                    let last_step_idx = obs_arr.len().saturating_sub(1);

                    if last_step_idx < obs_arr.len() {
                        let last_obs = &obs_arr[last_step_idx];

                        if let Some(obs_vec) = last_obs.as_array() {
                            let mut left_lines = vec![];
                            left_lines.push(Line::from(vec![
                                Span::styled(
                                    "Step ",
                                    Style::default()
                                        .fg(theme::TEXT)
                                        .add_modifier(Modifier::BOLD),
                                ),
                                Span::styled(
                                    format!("{}/{}", last_step_idx + 1, obs_arr.len()),
                                    Style::default().fg(theme::SUBTEXT0),
                                ),
                            ]));
                            left_lines.push(Line::from(""));

                            let global_labels = [
                                "Step Progress",
                                "Cash %",
                                "PnL",
                                "Drawdown",
                                "Commissions",
                                "Last Reward",
                                "Fill Ratio",
                            ];

                            for (i, label) in global_labels.iter().enumerate() {
                                if i < obs_vec.len() {
                                    let val = obs_vec[i].as_f64().unwrap_or(0.0);
                                    let color = match i {
                                        0 => theme::BLUE,
                                        1 => theme::GREEN,
                                        2 => {
                                            if val >= 0.0 {
                                                theme::GREEN
                                            } else {
                                                theme::RED
                                            }
                                        }
                                        3 => {
                                            if val >= 0.0 {
                                                theme::GREEN
                                            } else {
                                                theme::RED
                                            }
                                        }
                                        4 => theme::YELLOW,
                                        5 => {
                                            if val >= 0.0 {
                                                theme::GREEN
                                            } else {
                                                theme::RED
                                            }
                                        }
                                        6 => {
                                            if val >= 0.9 {
                                                theme::GREEN
                                            } else {
                                                theme::YELLOW
                                            }
                                        }
                                        _ => theme::TEXT,
                                    };

                                    let display_val = match i {
                                        0 | 1 | 6 => format!("{:6.2}%", val * 100.0),
                                        2 | 3 => format!("{:+7.2}%", val * 100.0),
                                        4 | 5 => format!("{:8.4}", val),
                                        _ => format!("{:8.4}", val),
                                    };

                                    left_lines.push(Line::from(vec![
                                        Span::styled(
                                            format!("{:14}: ", label),
                                            Style::default().fg(theme::SUBTEXT1),
                                        ),
                                        Span::styled(display_val, Style::default().fg(color)),
                                    ]));
                                }
                            }

                            let left_paragraph = Paragraph::new(left_lines).block(left_block);
                            f.render_widget(left_paragraph, content_chunks[0]);

                            let mut right_lines = vec![];
                            right_lines.push(Line::from(Span::styled(
                                "Ticker Data",
                                Style::default()
                                    .fg(theme::TEXT)
                                    .add_modifier(Modifier::BOLD),
                            )));
                            right_lines.push(Line::from(""));

                            let ticker_names = ["TSLA", "AAPL", "AMD", "INTC", "MSFT", "NVDA"];

                            for ticker_idx in 0..TICKERS_COUNT {
                                let base_idx =
                                    GLOBAL_STATIC_OBS + (ticker_idx * PER_TICKER_STATIC_OBS);

                                if base_idx + 2 < obs_vec.len() {
                                    let position_pct = obs_vec[base_idx].as_f64().unwrap_or(0.0);
                                    let unrealized_pnl =
                                        obs_vec[base_idx + 1].as_f64().unwrap_or(0.0);
                                    let momentum = obs_vec[base_idx + 2].as_f64().unwrap_or(0.0);

                                    right_lines.push(Line::from(vec![
                                        Span::styled(
                                            format!("{:5} ", ticker_names[ticker_idx]),
                                            Style::default().fg(theme::MAUVE),
                                        ),
                                        Span::styled(
                                            format!("Pos:{:5.1}% ", position_pct * 100.0),
                                            Style::default().fg(if position_pct > 0.01 {
                                                theme::SKY
                                            } else {
                                                theme::SURFACE2
                                            }),
                                        ),
                                        Span::styled(
                                            format!("PnL:{:+6.2}% ", unrealized_pnl * 100.0),
                                            Style::default().fg(if unrealized_pnl >= 0.0 {
                                                theme::GREEN
                                            } else {
                                                theme::RED
                                            }),
                                        ),
                                        Span::styled(
                                            format!("Mom:{:+6.2}%", momentum * 100.0),
                                            Style::default().fg(if momentum >= 0.0 {
                                                theme::GREEN
                                            } else {
                                                theme::RED
                                            }),
                                        ),
                                    ]));

                                    let action_start = base_idx + 3;
                                    let recent_actions = 3;
                                    let mut action_line_parts =
                                        vec![Span::styled("  ", Style::default())];

                                    for hist_idx in 0..recent_actions {
                                        let buy_sell_idx = action_start + (hist_idx * 2);
                                        let hold_idx = buy_sell_idx + 1;

                                        if hold_idx < obs_vec.len() {
                                            let buy_sell =
                                                obs_vec[buy_sell_idx].as_f64().unwrap_or(0.0);
                                            let hold = obs_vec[hold_idx].as_f64().unwrap_or(0.0);

                                            let bs_color = if buy_sell.abs() < 0.01 {
                                                theme::SURFACE2
                                            } else if buy_sell > 0.0 {
                                                theme::GREEN
                                            } else {
                                                theme::RED
                                            };

                                            action_line_parts.push(Span::styled(
                                                format!("T-{}: ", hist_idx),
                                                Style::default().fg(theme::OVERLAY0),
                                            ));
                                            action_line_parts.push(Span::styled(
                                                format!("{:+.2}", buy_sell),
                                                Style::default().fg(bs_color),
                                            ));
                                            action_line_parts.push(Span::styled(
                                                format!("/{:+.2} ", hold),
                                                Style::default().fg(theme::OVERLAY1),
                                            ));
                                        }
                                    }

                                    right_lines.push(Line::from(action_line_parts));
                                    right_lines.push(Line::from(""));
                                }
                            }

                            let right_paragraph = Paragraph::new(right_lines).block(right_block);
                            f.render_widget(right_paragraph, content_chunks[1]);

                            // Side panel split: actions + temporal attention
                            let side_chunks = Layout::default()
                                .direction(Direction::Vertical)
                                .constraints([Constraint::Length(10), Constraint::Min(0)])
                                .split(content_chunks[2]);

                            // Action snapshots
                            let action_block = Block::default()
                                .borders(Borders::ALL)
                                .title("Actions (Step 0 / Final)")
                                .border_style(Style::default().fg(theme::SURFACE2));

                            let ticker_names =
                                ["TSLA", "AAPL", "AMD", "INTC", "MSFT", "NVDA", "CASH"];
                            let step0_vals: Vec<f64> = action_step0
                                .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
                                .unwrap_or_default();
                            let final_vals: Vec<f64> = action_final
                                .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
                                .unwrap_or_default();

                            let mut action_lines = vec![Line::from(Span::styled(
                                "Target weights at episode boundaries",
                                Style::default()
                                    .fg(theme::TEXT)
                                    .add_modifier(Modifier::BOLD),
                            ))];
                            action_lines.push(Line::from(""));

                            if step0_vals.is_empty() && final_vals.is_empty() {
                                action_lines.push(Line::from(Span::styled(
                                    "No action snapshots recorded",
                                    Style::default().fg(theme::SUBTEXT0),
                                )));
                            } else {
                                for (idx, name) in ticker_names.iter().enumerate() {
                                    let s0 = step0_vals.get(idx).cloned();
                                    let sf = final_vals.get(idx).cloned();

                                    let s0_str = s0
                                        .map(|v| format!("{:+.3}", v))
                                        .unwrap_or_else(|| "  n/a".to_string());
                                    let sf_str = sf
                                        .map(|v| format!("{:+.3}", v))
                                        .unwrap_or_else(|| "  n/a".to_string());
                                    let sf_color = sf
                                        .map(|v| if v >= 0.0 { theme::GREEN } else { theme::RED })
                                        .unwrap_or(theme::SUBTEXT0);

                                    action_lines.push(Line::from(vec![
                                        Span::styled(
                                            format!("{:>4} ", name),
                                            Style::default().fg(theme::MAUVE),
                                        ),
                                        Span::styled(
                                            format!("0:{:>7} ", s0_str),
                                            Style::default().fg(theme::SUBTEXT1),
                                        ),
                                        Span::styled(
                                            format!("F:{:>7}", sf_str),
                                            Style::default().fg(sf_color),
                                        ),
                                    ]));
                                }
                            }

                            let action_paragraph = Paragraph::new(action_lines).block(action_block);
                            f.render_widget(action_paragraph, side_chunks[0]);

                            // Temporal attention panel
                            let temporal_block = Block::default()
                                .borders(Borders::ALL)
                                .title("Temporal Attention")
                                .border_style(Style::default().fg(theme::SURFACE2));

                            let attn_weights = json["attention_weights"].as_array();
                            let mut temporal_lines = vec![];

                            if let Some(attn_arr) = attn_weights {
                                if let Some(last_attn) = attn_arr.last() {
                                    if let Some(weights) = last_attn.as_array() {
                                        let weights_f: Vec<f64> =
                                            weights.iter().filter_map(|v| v.as_f64()).collect();

                                        if !weights_f.is_empty() {
                                            let len = weights_f.len();
                                            let max_val =
                                                weights_f.iter().cloned().fold(0.0_f64, f64::max);
                                            let max_idx = weights_f
                                                .iter()
                                                .position(|&x| x == max_val)
                                                .unwrap_or(0);
                                            let peak_pct = (max_idx as f64 / len as f64) * 100.0;

                                            // Entropy: -sum(p * ln(p))
                                            let entropy: f64 = weights_f
                                                .iter()
                                                .filter(|&&p| p > 1e-10)
                                                .map(|&p| -p * p.ln())
                                                .sum();
                                            let max_entropy = (len as f64).ln();
                                            let norm_entropy = entropy / max_entropy;

                                            temporal_lines.push(Line::from(vec![
                                                Span::styled(
                                                    "Peak: ",
                                                    Style::default().fg(theme::SUBTEXT1),
                                                ),
                                                Span::styled(
                                                    format!("{:.1}%", peak_pct),
                                                    Style::default().fg(theme::PEACH),
                                                ),
                                                Span::styled(
                                                    " of seq",
                                                    Style::default().fg(theme::SUBTEXT0),
                                                ),
                                            ]));
                                            temporal_lines.push(Line::from(vec![
                                                Span::styled(
                                                    "Focus: ",
                                                    Style::default().fg(theme::SUBTEXT1),
                                                ),
                                                Span::styled(
                                                    format!("{:.1}%", (1.0 - norm_entropy) * 100.0),
                                                    Style::default().fg(if norm_entropy < 0.5 {
                                                        theme::GREEN
                                                    } else {
                                                        theme::YELLOW
                                                    }),
                                                ),
                                            ]));
                                            temporal_lines.push(Line::from(""));

                                            // Sparkline: divide into 10 buckets
                                            let buckets = 10;
                                            let bucket_size = len / buckets;
                                            let mut bucket_sums: Vec<f64> =
                                                Vec::with_capacity(buckets);
                                            for i in 0..buckets {
                                                let start = i * bucket_size;
                                                let end = if i == buckets - 1 {
                                                    len
                                                } else {
                                                    (i + 1) * bucket_size
                                                };
                                                let sum: f64 = weights_f[start..end].iter().sum();
                                                bucket_sums.push(sum);
                                            }
                                            let bucket_max =
                                                bucket_sums.iter().cloned().fold(0.0_f64, f64::max);

                                            temporal_lines.push(Line::from(Span::styled(
                                                "Old -> Recent",
                                                Style::default().fg(theme::SUBTEXT0),
                                            )));
                                            let bars = ["_", ".", "-", "=", "#"];
                                            let bar_line: String = bucket_sums
                                                .iter()
                                                .map(|&v| {
                                                    let level =
                                                        ((v / bucket_max) * 4.0).round() as usize;
                                                    bars[level.min(4)]
                                                })
                                                .collect();
                                            temporal_lines.push(Line::from(Span::styled(
                                                format!("[{}]", bar_line),
                                                Style::default().fg(theme::SKY),
                                            )));
                                        }
                                    }
                                }
                            }

                            if temporal_lines.is_empty() {
                                temporal_lines.push(Line::from(Span::styled(
                                    "No attention data",
                                    Style::default().fg(theme::SUBTEXT0),
                                )));
                            }

                            let temporal_paragraph =
                                Paragraph::new(temporal_lines).block(temporal_block);
                            f.render_widget(temporal_paragraph, side_chunks[1]);
                        } else {
                            let msg = Paragraph::new("No observation data in step")
                                .block(left_block)
                                .style(Style::default().fg(theme::SUBTEXT0));
                            f.render_widget(msg, content_chunks[0]);
                            f.render_widget(
                                Block::default()
                                    .borders(Borders::ALL)
                                    .border_style(Style::default().fg(theme::SURFACE2)),
                                content_chunks[1],
                            );
                            f.render_widget(
                                Block::default()
                                    .borders(Borders::ALL)
                                    .border_style(Style::default().fg(theme::SURFACE2)),
                                content_chunks[2],
                            );
                        }
                    } else {
                        let msg = Paragraph::new("No observations data available")
                            .block(left_block)
                            .style(Style::default().fg(theme::SUBTEXT0));
                        f.render_widget(msg, content_chunks[0]);
                        f.render_widget(
                            Block::default()
                                .borders(Borders::ALL)
                                .border_style(Style::default().fg(theme::SURFACE2)),
                            content_chunks[1],
                        );
                        f.render_widget(
                            Block::default()
                                .borders(Borders::ALL)
                                .border_style(Style::default().fg(theme::SURFACE2)),
                            content_chunks[2],
                        );
                    }
                } else {
                    let msg = Paragraph::new("Invalid observations format")
                        .block(left_block)
                        .style(Style::default().fg(theme::RED));
                    f.render_widget(msg, content_chunks[0]);
                    f.render_widget(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(theme::SURFACE2)),
                        content_chunks[1],
                    );
                    f.render_widget(
                        Block::default()
                            .borders(Borders::ALL)
                            .border_style(Style::default().fg(theme::SURFACE2)),
                        content_chunks[2],
                    );
                }
            } else {
                let msg = Paragraph::new("Failed to parse observations JSON")
                    .block(left_block)
                    .style(Style::default().fg(theme::RED));
                f.render_widget(msg, content_chunks[0]);
                f.render_widget(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(theme::SURFACE2)),
                    content_chunks[1],
                );
                f.render_widget(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(theme::SURFACE2)),
                    content_chunks[2],
                );
            }
        } else {
            let msg = Paragraph::new(format!("No observations file found at {}", obs_path))
                .block(left_block)
                .style(Style::default().fg(theme::SUBTEXT0));
            f.render_widget(msg, content_chunks[0]);
            f.render_widget(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(theme::SURFACE2)),
                content_chunks[1],
            );
            f.render_widget(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(theme::SURFACE2)),
                content_chunks[2],
            );
        }
    } else {
        let msg = Paragraph::new("No episodes found. Train a model first.")
            .block(left_block)
            .style(Style::default().fg(theme::SUBTEXT0));
        f.render_widget(msg, content_chunks[0]);
        f.render_widget(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::SURFACE2)),
            content_chunks[1],
        );
        f.render_widget(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::SURFACE2)),
            content_chunks[2],
        );
    }

    let help_text = " ESC: Back to Main | R: Refresh ";
    let help =
        Paragraph::new(help_text).block(Block::default().borders(Borders::ALL).title(" Controls "));
    f.render_widget(help, chunks[2]);
}
