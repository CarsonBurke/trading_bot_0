use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
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
            let stripped = strip_ansi(line);
            let style = style_from_content(&stripped);
            ListItem::new(stripped).style(style)
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


fn strip_ansi(line: &str) -> String {
    let mut result = String::new();
    let bytes = line.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'\x1b' && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            // Skip ANSI escape sequence
            i += 2;
            while i < bytes.len() && bytes[i] != b'm' {
                i += 1;
            }
            if i < bytes.len() {
                i += 1; // Skip 'm'
            }
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }

    result
}

fn style_from_content(line: &str) -> Style {
    let lower = line.to_lowercase();

    // Error patterns
    if lower.contains("error") || lower.contains("failed") || lower.contains("panic") {
        return Style::default().fg(theme::RED);
    }

    // Warning patterns
    if lower.contains("warn") || lower.contains("warning") {
        return Style::default().fg(theme::PEACH);
    }

    // Episode/training progress
    if lower.contains("episode") {
        return Style::default().fg(theme::BLUE);
    }

    // Success patterns
    if lower.contains("finished") || lower.contains("completed") || lower.contains("success") {
        return Style::default().fg(theme::GREEN);
    }

    // Compiling messages
    if lower.contains("compiling") || lower.contains("checking") {
        return Style::default().fg(theme::LAVENDER);
    }

    // Default
    Style::default().fg(theme::TEXT)
}
