use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

use crate::{theme, App};

pub fn render(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
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

    // Create list items from training output
    let items: Vec<ListItem> = app.training_output
        .iter()
        .map(|line| ListItem::new(parse_ansi_line(line)))
        .collect();

    let total_lines = items.len();

    let logs_list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::BLUE))
                .title(format!(" Logs ({} lines) ", total_lines)),
        );

    app.logs_list_area = chunks[2];
    let mut logs_state = app.logs_list_state.clone();
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
    let parts: Vec<&str> = code.split(';').collect();
    let mut i = 0;

    while i < parts.len() {
        match parts[i] {
            // Reset
            "0" => style = Style::default(),
            // Modifiers
            "1" => style = style.add_modifier(Modifier::BOLD),
            "2" => style = style.add_modifier(Modifier::DIM),
            "3" => style = style.add_modifier(Modifier::ITALIC),
            "4" => style = style.add_modifier(Modifier::UNDERLINED),
            "5" => style = style.add_modifier(Modifier::SLOW_BLINK),
            "6" => style = style.add_modifier(Modifier::RAPID_BLINK),
            "7" => style = style.add_modifier(Modifier::REVERSED),
            "8" => style = style.add_modifier(Modifier::HIDDEN),
            "9" => style = style.add_modifier(Modifier::CROSSED_OUT),
            // Remove modifiers
            "22" => style = style.remove_modifier(Modifier::BOLD | Modifier::DIM),
            "23" => style = style.remove_modifier(Modifier::ITALIC),
            "24" => style = style.remove_modifier(Modifier::UNDERLINED),
            "25" => style = style.remove_modifier(Modifier::SLOW_BLINK | Modifier::RAPID_BLINK),
            "27" => style = style.remove_modifier(Modifier::REVERSED),
            "28" => style = style.remove_modifier(Modifier::HIDDEN),
            "29" => style = style.remove_modifier(Modifier::CROSSED_OUT),
            // Foreground colors (standard 16)
            "30" => style = style.fg(Color::Black),
            "31" => style = style.fg(Color::Red),
            "32" => style = style.fg(Color::Green),
            "33" => style = style.fg(Color::Yellow),
            "34" => style = style.fg(Color::Blue),
            "35" => style = style.fg(Color::Magenta),
            "36" => style = style.fg(Color::Cyan),
            "37" => style = style.fg(Color::White),
            "39" => style = style.fg(Color::Reset),
            // Background colors (standard 16)
            "40" => style = style.bg(Color::Black),
            "41" => style = style.bg(Color::Red),
            "42" => style = style.bg(Color::Green),
            "43" => style = style.bg(Color::Yellow),
            "44" => style = style.bg(Color::Blue),
            "45" => style = style.bg(Color::Magenta),
            "46" => style = style.bg(Color::Cyan),
            "47" => style = style.bg(Color::White),
            "49" => style = style.bg(Color::Reset),
            // Bright foreground colors
            "90" => style = style.fg(Color::DarkGray),
            "91" => style = style.fg(Color::LightRed),
            "92" => style = style.fg(Color::LightGreen),
            "93" => style = style.fg(Color::LightYellow),
            "94" => style = style.fg(Color::LightBlue),
            "95" => style = style.fg(Color::LightMagenta),
            "96" => style = style.fg(Color::LightCyan),
            "97" => style = style.fg(Color::Gray),
            // Bright background colors
            "100" => style = style.bg(Color::DarkGray),
            "101" => style = style.bg(Color::LightRed),
            "102" => style = style.bg(Color::LightGreen),
            "103" => style = style.bg(Color::LightYellow),
            "104" => style = style.bg(Color::LightBlue),
            "105" => style = style.bg(Color::LightMagenta),
            "106" => style = style.bg(Color::LightCyan),
            "107" => style = style.bg(Color::Gray),
            // 256-color and RGB support
            "38" => {
                // Foreground extended color
                if i + 1 < parts.len() {
                    match parts[i + 1] {
                        "5" => {
                            // 256-color mode: ESC[38;5;Nm
                            if i + 2 < parts.len() {
                                if let Ok(color_idx) = parts[i + 2].parse::<u8>() {
                                    style = style.fg(Color::Indexed(color_idx));
                                }
                                i += 2;
                            }
                        }
                        "2" => {
                            // RGB mode: ESC[38;2;R;G;Bm
                            if i + 4 < parts.len() {
                                if let (Ok(r), Ok(g), Ok(b)) = (
                                    parts[i + 2].parse::<u8>(),
                                    parts[i + 3].parse::<u8>(),
                                    parts[i + 4].parse::<u8>(),
                                ) {
                                    style = style.fg(Color::Rgb(r, g, b));
                                }
                                i += 4;
                            }
                        }
                        _ => {}
                    }
                }
            }
            "48" => {
                // Background extended color
                if i + 1 < parts.len() {
                    match parts[i + 1] {
                        "5" => {
                            // 256-color mode: ESC[48;5;Nm
                            if i + 2 < parts.len() {
                                if let Ok(color_idx) = parts[i + 2].parse::<u8>() {
                                    style = style.bg(Color::Indexed(color_idx));
                                }
                                i += 2;
                            }
                        }
                        "2" => {
                            // RGB mode: ESC[48;2;R;G;Bm
                            if i + 4 < parts.len() {
                                if let (Ok(r), Ok(g), Ok(b)) = (
                                    parts[i + 2].parse::<u8>(),
                                    parts[i + 3].parse::<u8>(),
                                    parts[i + 4].parse::<u8>(),
                                ) {
                                    style = style.bg(Color::Rgb(r, g, b));
                                }
                                i += 4;
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    style
}
