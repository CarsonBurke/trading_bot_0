use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use super::base::BaseDialog;
use crate::{theme, AppMode};

pub fn render(f: &mut Frame, selected: usize, current_mode: AppMode) {
    let inner = BaseDialog::new(" Go to Page ")
        .width(40)
        .min_height(12)
        .max_height(16)
        .render(f);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),    // Page list
            Constraint::Length(3), // Help bar
        ])
        .split(inner);

    // Page options (None for ChartViewer means meta charts, not episode charts)
    let pages: [(Option<AppMode>, &str, &str); 5] = [
        (Some(AppMode::Main), "1", "Dashboard"),
        (Some(AppMode::GenerationBrowser), "2", "Episodes"),
        (Some(AppMode::InferenceBrowser), "3", "Inferences"),
        (None, "4", "Meta Charts"), // Special case: ChartViewer with meta charts
        (Some(AppMode::Logs), "5", "Logs"),
    ];

    let page_lines: Vec<Line> = pages
        .iter()
        .enumerate()
        .map(|(i, (mode_opt, key, name))| {
            let is_selected = i == selected;
            let is_current = match mode_opt {
                Some(mode) => *mode == current_mode,
                None => current_mode == AppMode::ChartViewer, // Meta charts
            };

            let prefix = if is_selected { ">" } else { " " };
            let suffix = if is_current { " (current)" } else { "" };

            let style = if is_selected {
                Style::default()
                    .fg(theme::BASE)
                    .bg(theme::LAVENDER)
                    .add_modifier(Modifier::BOLD)
            } else if is_current {
                Style::default().fg(theme::SUBTEXT0)
            } else {
                Style::default().fg(theme::TEXT)
            };

            Line::from(vec![
                Span::styled(format!("{} ", prefix), style),
                Span::styled(
                    format!("[{}]", key),
                    if is_selected {
                        Style::default()
                            .fg(theme::BASE)
                            .bg(theme::LAVENDER)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                            .fg(theme::PEACH)
                            .add_modifier(Modifier::BOLD)
                    },
                ),
                Span::styled(format!(" {}{}", name, suffix), style),
            ])
        })
        .collect();

    let pages_widget = Paragraph::new(page_lines).style(Style::default().bg(theme::BASE));
    f.render_widget(pages_widget, chunks[0]);

    // Help bar
    let help = Paragraph::new(vec![Line::from(vec![
        Span::styled(
            "1-5",
            Style::default()
                .fg(theme::PEACH)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Jump  ", Style::default().fg(theme::SUBTEXT1)),
        Span::styled(
            "j/k",
            Style::default()
                .fg(theme::BLUE)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Navigate  ", Style::default().fg(theme::SUBTEXT1)),
        Span::styled(
            "Enter",
            Style::default()
                .fg(theme::GREEN)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Go  ", Style::default().fg(theme::SUBTEXT1)),
        Span::styled(
            "Esc",
            Style::default().fg(theme::RED).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" Cancel", Style::default().fg(theme::SUBTEXT1)),
    ])])
    .style(Style::default().bg(theme::BASE))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::SURFACE2)),
    );
    f.render_widget(help, chunks[1]);
}
