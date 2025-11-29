use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Sparkline},
    Frame,
};
use std::fs;
use serde_json::Value;

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
            Constraint::Percentage(50),
            Constraint::Percentage(50),
        ])
        .split(main_chunks[1]);

    let chunks = main_chunks;

    let is_training = app.is_training_running();
    let current_episode = app.get_current_episode();

    let mut title_spans = vec![
        Span::styled(" Model Observations ", Style::default().fg(theme::MAUVE).add_modifier(Modifier::BOLD)),
    ];
    title_spans.extend(episode_status::episode_status_spans(is_training, current_episode));

    let title = Paragraph::new(Line::from(title_spans))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::LAVENDER)),
        );
    f.render_widget(title, chunks[0]);

    let latest_episode = app.generation_browser.generations.first();

    let left_block = Block::default()
        .borders(Borders::ALL)
        .title("Current Step")
        .border_style(Style::default().fg(theme::SURFACE2));

    let right_block = Block::default()
        .borders(Borders::ALL)
        .title("Temporal Evolution (Last 100 Steps)")
        .border_style(Style::default().fg(theme::SURFACE2));

    if let Some(gen) = latest_episode {
        let obs_path = format!("../training/gens/{}/observations.json", gen.number);

        if let Ok(contents) = fs::read_to_string(&obs_path) {
            if let Ok(json) = serde_json::from_str::<Value>(&contents) {
                let static_obs = json["static_observations"].as_array();
                let attn_weights = json["attention_weights"].as_array();

                if let (Some(obs_arr), Some(attn_arr)) = (static_obs, attn_weights) {
                    let last_step_idx = obs_arr.len().saturating_sub(1);

                    if last_step_idx < obs_arr.len() {
                        let last_obs = &obs_arr[last_step_idx];
                        let last_attn = &attn_arr[last_step_idx];

                        let mut lines = vec![];
                        lines.push(Line::from(vec![
                            Span::styled("Latest Step Observations ", Style::default().fg(theme::SKY).add_modifier(Modifier::BOLD)),
                            Span::styled(format!("(Step {}/{})", last_step_idx + 1, obs_arr.len()), Style::default().fg(theme::SUBTEXT0)),
                        ]));
                        lines.push(Line::from(""));

                        if let Some(obs_vec) = last_obs.as_array() {
                            if let Some(attn_vec) = last_attn.as_array() {
                                lines.push(Line::from(Span::styled("Static Observations:", Style::default().fg(theme::BLUE).add_modifier(Modifier::BOLD))));

                                let obs_labels = [
                                    "Step Progress",
                                    "Cash %",
                                    "Position %",
                                    "Time-Wtd Return",
                                ];

                                for (i, label) in obs_labels.iter().enumerate() {
                                    if i < obs_vec.len() {
                                        let val = obs_vec[i].as_f64().unwrap_or(0.0);
                                        let attn = if i < attn_vec.len() {
                                            attn_vec[i].as_f64().unwrap_or(0.0)
                                        } else {
                                            0.0
                                        };

                                        let bar_width = (attn * 30.0) as usize;
                                        let bar = "█".repeat(bar_width);

                                        lines.push(Line::from(vec![
                                            Span::styled(format!("{:18}: ", label), Style::default().fg(theme::TEXT)),
                                            Span::styled(format!("{:8.4} ", val), Style::default().fg(theme::GREEN)),
                                            Span::styled(format!("Attn: {:5.3} ", attn), Style::default().fg(theme::PEACH)),
                                            Span::styled(bar, Style::default().fg(theme::MAUVE)),
                                        ]));
                                    }
                                }

                                lines.push(Line::from(""));
                                lines.push(Line::from(Span::styled("Action History (most recent 5):", Style::default().fg(theme::BLUE).add_modifier(Modifier::BOLD))));

                                let action_start = 4;
                                let actions_per_step = 2;
                                let history_len = 5;

                                for hist_idx in 0..history_len {
                                    let base_idx = action_start + (hist_idx * actions_per_step);
                                    if base_idx + 1 < obs_vec.len() {
                                        let buy_sell = obs_vec[base_idx].as_f64().unwrap_or(0.0);
                                        let hold = obs_vec[base_idx + 1].as_f64().unwrap_or(0.0);

                                        let buy_sell_attn = if base_idx < attn_vec.len() {
                                            attn_vec[base_idx].as_f64().unwrap_or(0.0)
                                        } else {
                                            0.0
                                        };
                                        let hold_attn = if base_idx + 1 < attn_vec.len() {
                                            attn_vec[base_idx + 1].as_f64().unwrap_or(0.0)
                                        } else {
                                            0.0
                                        };

                                        let buy_sell_bar_width = (buy_sell_attn * 20.0) as usize;
                                        let hold_bar_width = (hold_attn * 20.0) as usize;
                                        let buy_sell_bar = "█".repeat(buy_sell_bar_width);
                                        let hold_bar = "█".repeat(hold_bar_width);

                                        lines.push(Line::from(vec![
                                            Span::styled(format!("  T-{}: ", hist_idx), Style::default().fg(theme::SUBTEXT0)),
                                            Span::styled(format!("B/S={:6.3} ", buy_sell), Style::default().fg(theme::SKY)),
                                            Span::styled(format!("[{:4.2}]", buy_sell_attn), Style::default().fg(theme::PEACH)),
                                            Span::styled(format!("{:5} ", buy_sell_bar), Style::default().fg(theme::BLUE)),
                                            Span::styled(format!("H={:6.3} ", hold), Style::default().fg(theme::LAVENDER)),
                                            Span::styled(format!("[{:4.2}]", hold_attn), Style::default().fg(theme::PEACH)),
                                            Span::styled(hold_bar, Style::default().fg(theme::MAUVE)),
                                        ]));
                                    }
                                }
                            }
                        }

                        let paragraph = Paragraph::new(lines).block(left_block);
                        f.render_widget(paragraph, content_chunks[0]);

                        // Render temporal charts in right panel
                        let mut temporal_lines = vec![];
                        temporal_lines.push(Line::from(Span::styled("Key Metrics Over Time", Style::default().fg(theme::SKY).add_modifier(Modifier::BOLD))));
                        temporal_lines.push(Line::from(""));

                        // Extract time series for cash%
                        let cash_series: Vec<u64> = obs_arr.iter()
                            .filter_map(|step| step.as_array())
                            .filter_map(|obs_vec| obs_vec.get(1))
                            .filter_map(|val| val.as_f64())
                            .map(|v| (v * 100.0) as u64)
                            .collect();

                        if !cash_series.is_empty() {
                            let cash_sparkline = Sparkline::default()
                                .block(Block::default().title("Cash %").borders(Borders::ALL).border_style(Style::default().fg(theme::SURFACE1)))
                                .data(&cash_series)
                                .style(Style::default().fg(theme::GREEN));

                            temporal_lines.push(Line::from(""));
                            temporal_lines.push(Line::from(format!("Cash: {:.2}% (min: {}, max: {})",
                                cash_series.last().unwrap_or(&0),
                                cash_series.iter().min().unwrap_or(&0),
                                cash_series.iter().max().unwrap_or(&0))));
                        }

                        // Extract time series for position%
                        let pos_series: Vec<u64> = obs_arr.iter()
                            .filter_map(|step| step.as_array())
                            .filter_map(|obs_vec| obs_vec.get(2))
                            .filter_map(|val| val.as_f64())
                            .map(|v| (v * 100.0) as u64)
                            .collect();

                        if !pos_series.is_empty() {
                            temporal_lines.push(Line::from(""));
                            temporal_lines.push(Line::from(format!("Position: {:.2}% (min: {}, max: {})",
                                pos_series.last().unwrap_or(&0),
                                pos_series.iter().min().unwrap_or(&0),
                                pos_series.iter().max().unwrap_or(&0))));
                        }

                        // Extract attention weights for static obs (first 4)
                        temporal_lines.push(Line::from(""));
                        temporal_lines.push(Line::from(Span::styled("Attention Evolution:", Style::default().fg(theme::BLUE).add_modifier(Modifier::BOLD))));

                        for obs_idx in 0..4 {
                            let attn_series: Vec<u64> = attn_arr.iter()
                                .filter_map(|step| step.as_array())
                                .filter_map(|attn_vec| attn_vec.get(obs_idx))
                                .filter_map(|val| val.as_f64())
                                .map(|v| (v * 1000.0) as u64)
                                .collect();

                            if !attn_series.is_empty() {
                                let labels = ["Step Prog", "Cash %", "Position %", "TWR"];
                                let avg = attn_series.iter().sum::<u64>() as f64 / attn_series.len() as f64 / 1000.0;
                                temporal_lines.push(Line::from(format!("  {}: avg={:.3}", labels[obs_idx], avg)));
                            }
                        }

                        let temporal_paragraph = Paragraph::new(temporal_lines).block(right_block);
                        f.render_widget(temporal_paragraph, content_chunks[1]);
                    } else {
                        let msg = Paragraph::new("No observations data available")
                            .block(left_block)
                            .style(Style::default().fg(theme::SUBTEXT0));
                        f.render_widget(msg, content_chunks[0]);
                        f.render_widget(Block::default().borders(Borders::ALL).border_style(Style::default().fg(theme::SURFACE2)), content_chunks[1]);
                    }
                } else {
                    let msg = Paragraph::new("Invalid observations format")
                        .block(left_block)
                        .style(Style::default().fg(theme::RED));
                    f.render_widget(msg, content_chunks[0]);
                    f.render_widget(Block::default().borders(Borders::ALL).border_style(Style::default().fg(theme::SURFACE2)), content_chunks[1]);
                }
            } else {
                let msg = Paragraph::new("Failed to parse observations JSON")
                    .block(left_block)
                    .style(Style::default().fg(theme::RED));
                f.render_widget(msg, content_chunks[0]);
                f.render_widget(Block::default().borders(Borders::ALL).border_style(Style::default().fg(theme::SURFACE2)), content_chunks[1]);
            }
        } else {
            let msg = Paragraph::new(format!("No observations file found at {}", obs_path))
                .block(left_block)
                .style(Style::default().fg(theme::SUBTEXT0));
            f.render_widget(msg, content_chunks[0]);
            f.render_widget(Block::default().borders(Borders::ALL).border_style(Style::default().fg(theme::SURFACE2)), content_chunks[1]);
        }
    } else {
        let msg = Paragraph::new("No episodes found. Train a model first.")
            .block(left_block)
            .style(Style::default().fg(theme::SUBTEXT0));
        f.render_widget(msg, content_chunks[0]);
        f.render_widget(Block::default().borders(Borders::ALL).border_style(Style::default().fg(theme::SURFACE2)), content_chunks[1]);
    }

    let help_text = " ESC: Back to Main ";
    let help = Paragraph::new(help_text)
        .block(Block::default().borders(Borders::ALL).title(" Controls "));
    f.render_widget(help, chunks[2]);
}
