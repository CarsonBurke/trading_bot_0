use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};
use shared::paths::WEIGHTS_PATH;

use crate::{theme, App};

pub fn render(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(6),
        ])
        .split(f.area());

    let title = Paragraph::new(" Trading Bot TUI ")
        .style(
            Style::default()
                .fg(theme::MAUVE)
                .add_modifier(Modifier::BOLD),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::LAVENDER)),
        );
    f.render_widget(title, chunks[0]);

    let is_training = app.is_training_running();
    let status = if is_training {
        "RUNNING ●"
    } else {
        "STOPPED ○"
    };

    let status_color = if is_training {
        theme::GREEN
    } else {
        theme::RED
    };

    let weights_info = if let Some(weights) = &app.weights_path {
        format!("{}", weights)
    } else {
        format!("{} (default)", WEIGHTS_PATH)
    };

    let (current_episode, _) = parse_current_episode(&app.training_output);
    let episode_info = if is_training {
        if let Some(ep) = current_episode {
            format!("Episode {}", ep)
        } else {
            "Starting...".to_string()
        }
    } else {
        "N/A".to_string()
    };

    let info_text = vec![
        Line::from(vec![
            Span::styled("Status: ", Style::default().fg(theme::SUBTEXT1)),
            Span::styled(
                status,
                Style::default()
                    .fg(status_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  Episode: ", Style::default().fg(theme::SUBTEXT1)),
            Span::styled(episode_info, Style::default().fg(theme::SKY)),
            Span::styled("  Weights: ", Style::default().fg(theme::SUBTEXT1)),
            Span::styled(weights_info, Style::default().fg(theme::BLUE)),
        ]),
    ];

    let info = Paragraph::new(info_text)
        .style(Style::default().fg(theme::TEXT))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::SURFACE2))
                .title(" Info ")
                .title_style(Style::default().fg(theme::SKY)),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(info, chunks[1]);

    let meta_chart_items: Vec<Line> = if app.latest_meta_charts.is_empty() {
        vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No meta-history charts found yet",
                Style::default().fg(theme::OVERLAY0),
            )),
        ]
    } else {
        app.latest_meta_charts
            .iter()
            .map(|path| {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");
                Line::from(vec![
                    Span::styled(" ", Style::default()),
                    Span::styled(name, Style::default().fg(theme::TEXT)),
                ])
            })
            .collect()
    };

    let meta_charts = Paragraph::new(meta_chart_items)
        .style(Style::default().fg(theme::TEXT))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::SURFACE2))
                .title(" Latest Meta-History Charts ")
                .title_style(Style::default().fg(theme::PEACH)),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(meta_charts, chunks[2]);

    let help_text = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("s", Style::default().fg(Color::Green)),
            Span::raw(": Start Training  "),
            Span::styled("f", Style::default().fg(Color::Blue)),
            Span::raw(": Run Inference  "),
            Span::styled("x", Style::default().fg(Color::Red)),
            Span::raw(": Stop Training  "),
            Span::styled("l", Style::default().fg(Color::Yellow)),
            Span::raw(": Logs  "),
            Span::styled("q", Style::default().fg(Color::Red)),
            Span::raw(": Quit"),
        ]),
        Line::from(vec![
            Span::styled("e", Style::default().fg(Color::Cyan)),
            Span::raw(": View Episodes  "),
            Span::styled("i", Style::default().fg(Color::Cyan)),
            Span::raw(": View Inferences  "),
            Span::styled("m", Style::default().fg(Color::Magenta)),
            Span::raw(": View Meta Charts"),
        ]),
    ];

    let help = Paragraph::new(help_text).block(Block::default().borders(Borders::ALL).title("Controls"));
    f.render_widget(help, chunks[3]);
}

fn parse_current_episode(output: &[String]) -> (Option<usize>, Option<usize>) {
    for line in output.iter().rev() {
        // Look for actual episode completion logs: "Episode N - Total Assets..."
        // Skip PPO progress logs: "[Ep N] Episodes: ..."
        if line.contains("Episode") && line.contains("Total Assets") && !line.starts_with("[Ep") {
            if let Some(ep_str) = line.split("Episode").nth(1) {
                if let Some(num_str) = ep_str.trim().split_whitespace().next() {
                    // Strip ANSI codes if present
                    let clean_num = num_str.chars().filter(|c| c.is_ascii_digit()).collect::<String>();
                    if let Ok(ep) = clean_num.parse::<usize>() {
                        return (Some(ep), None);
                    }
                }
            }
        }
    }
    (None, None)
}
