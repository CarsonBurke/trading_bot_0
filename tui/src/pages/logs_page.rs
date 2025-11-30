use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

use crate::{components::episode_status, theme, App};

pub fn render(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(f.area());

    let is_training = app.is_training_running();
    let current_episode = app.get_current_episode();

    let mut title_spans = vec![
        Span::styled(" Training Logs ", Style::default().fg(theme::MAUVE).add_modifier(Modifier::BOLD)),
    ];
    title_spans.extend(episode_status::episode_status_spans(is_training, current_episode));

    let title = Paragraph::new(Line::from(title_spans))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::LAVENDER)),
        );
    f.render_widget(title, chunks[0]);

    // Create list items from training output
    let items: Vec<ListItem> = app.logs_page.training_output
        .iter()
        .map(|line| {
            let parsed_line = parse_ansi_line(line);
            ListItem::new(parsed_line)
        })
        .collect();

    let total_lines = items.len();

    let logs_list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::BLUE))
                .title(format!(" Logs ({} lines) ", total_lines)),
        );

    app.logs_page.logs_list_area = chunks[1];
    f.render_stateful_widget(logs_list, chunks[1], &mut app.logs_page.logs_list_state);

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
    f.render_widget(help, chunks[2]);
}


fn parse_ansi_line(line: &str) -> Line<'static> {
    let mut spans = Vec::new();
    let mut current_text = String::new();
    let mut current_style = Style::default().fg(theme::TEXT);
    let bytes = line.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'\x1b' && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            // Save current text if any
            if !current_text.is_empty() {
                spans.push(Span::styled(current_text.clone(), current_style));
                current_text.clear();
            }

            // Parse ANSI escape sequence
            i += 2;
            let mut code_str = String::new();
            while i < bytes.len() && bytes[i] != b'm' {
                code_str.push(bytes[i] as char);
                i += 1;
            }
            if i < bytes.len() {
                i += 1; // Skip 'm'
            }

            // Apply ANSI code to current style
            current_style = apply_ansi_code(&code_str, current_style);
        } else {
            current_text.push(bytes[i] as char);
            i += 1;
        }
    }

    // Add remaining text
    if !current_text.is_empty() {
        spans.push(Span::styled(current_text, current_style));
    }

    // If no spans were created, add a default empty span
    if spans.is_empty() {
        spans.push(Span::raw(""));
    }

    Line::from(spans)
}

fn apply_ansi_code(code: &str, mut style: Style) -> Style {
    for part in code.split(';') {
        match part.trim() {
            "0" => style = Style::default().fg(theme::TEXT), // Reset
            "1" => style = style.add_modifier(Modifier::BOLD),
            "4" => style = style.add_modifier(Modifier::UNDERLINED),
            "22" => style = style.remove_modifier(Modifier::BOLD),
            "24" => style = style.remove_modifier(Modifier::UNDERLINED),
            // Foreground colors (30-37, 90-97)
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
            "39" => style = style.fg(theme::TEXT), // Default foreground
            _ => {}
        }
    }
    style
}
