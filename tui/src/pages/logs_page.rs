use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Frame,
};

use crate::{theme, App};

pub fn render(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(5),
            Constraint::Min(0),
            Constraint::Length(4),
        ])
        .split(f.area());

    let title = Paragraph::new(" Training Logs ")
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
    let (current_episode, total_episodes) = parse_current_episode(&app.training_output);

    let status_text = if is_training {
        if let Some(ep) = current_episode {
            if let Some(total) = total_episodes {
                format!("Training: Episode {}/{}", ep, total)
            } else {
                format!("Training: Episode {}", ep)
            }
        } else {
            "Training: Starting...".to_string()
        }
    } else {
        "Not training".to_string()
    };

    let status_color = if is_training { theme::GREEN } else { theme::RED };

    let status_info = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Status: ", Style::default().fg(theme::SUBTEXT1)),
            Span::styled(status_text, Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::SURFACE2))
            .title(" Status "),
    );
    f.render_widget(status_info, chunks[1]);

    let display_lines: Vec<&String> = app.training_output.iter().rev().take(100).rev().collect();

    let items: Vec<ListItem> = display_lines
        .iter()
        .map(|line| {
            ListItem::new(parse_ansi_line(line))
        })
        .collect();

    let mut logs_state = ListState::default();
    if !items.is_empty() {
        logs_state.select(Some(app.logs_scroll.min(items.len().saturating_sub(1))));
    }

    let logs_list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::BLUE))
                .title(format!(" Logs ({} lines) ", app.training_output.len())),
        )
        .highlight_style(Style::default().fg(theme::YELLOW));

    f.render_stateful_widget(logs_list, chunks[2], &mut logs_state);

    let help = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("↑/↓/PgUp/PgDn/Home/End", Style::default().fg(theme::BLUE)),
            Span::raw(": Scroll  "),
            Span::styled("c", Style::default().fg(theme::PEACH)),
            Span::raw(": Clear  "),
            Span::styled("q/Esc", Style::default().fg(theme::RED)),
            Span::raw(": Back"),
        ]),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::SURFACE2))
            .title(" Controls "),
    );
    f.render_widget(help, chunks[3]);
}

fn parse_current_episode(output: &[String]) -> (Option<usize>, Option<usize>) {
    for line in output.iter().rev() {
        if line.contains("Episode") {
            if let Some(ep_str) = line.split("Episode").nth(1) {
                if let Some(num_str) = ep_str.trim().split_whitespace().next() {
                    if let Ok(ep) = num_str.parse::<usize>() {
                        return (Some(ep), None);
                    }
                }
            }
        }
    }
    (None, None)
}

fn parse_ansi_line(line: &str) -> Line<'_> {
    let bytes = line.as_bytes();
    let mut spans = Vec::new();
    let mut start = 0;
    let mut current_style = Style::default();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'\x1b' && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            if start < i {
                spans.push(Span::styled(&line[start..i], current_style));
            }

            i += 2;
            let code_start = i;

            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b';') {
                i += 1;
            }

            if i < bytes.len() && bytes[i] == b'm' {
                current_style = parse_ansi_code(&line[code_start..i]);
                i += 1;
                start = i;
            }
        } else {
            i += 1;
        }
    }

    if start < bytes.len() {
        spans.push(Span::styled(&line[start..], current_style));
    }

    Line::from(spans)
}

fn parse_ansi_code(code: &str) -> Style {
    let mut style = Style::default();

    for part in code.split(';') {
        match part {
            "0" => style = Style::default(),
            "1" => style = style.add_modifier(Modifier::BOLD),
            "30" => style = style.fg(Color::Black),
            "31" => style = style.fg(Color::Red),
            "32" => style = style.fg(Color::Green),
            "33" => style = style.fg(Color::Yellow),
            "34" => style = style.fg(Color::Blue),
            "35" => style = style.fg(Color::Magenta),
            "36" => style = style.fg(Color::Cyan),
            "37" => style = style.fg(Color::White),
            "90" => style = style.fg(Color::DarkGray),
            "91" => style = style.fg(Color::LightRed),
            "92" => style = style.fg(Color::LightGreen),
            "93" => style = style.fg(Color::LightYellow),
            "94" => style = style.fg(Color::LightBlue),
            "95" => style = style.fg(Color::LightMagenta),
            "96" => style = style.fg(Color::LightCyan),
            "97" => style = style.fg(Color::Gray),
            _ => {}
        }
    }

    style
}
