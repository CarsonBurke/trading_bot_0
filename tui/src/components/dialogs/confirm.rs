use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use super::base::{BaseDialog, DialogStyle};
use crate::theme;

pub fn render(f: &mut Frame, title: &str, message: &str) {
    let dialog_title = if title.contains("Quit") {
        " Confirm Quit "
    } else if title.contains("Stop") {
        " Stop Training "
    } else {
        title
    };

    let inner = BaseDialog::new(dialog_title)
        .width(55)
        .min_height(10)
        .max_height(14)
        .style(DialogStyle::Warning)
        .render(f);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Message
            Constraint::Min(1),    // Spacer
            Constraint::Length(3), // Help bar
        ])
        .split(inner);

    // Message
    let message_widget = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled(message, Style::default().fg(theme::TEXT)),
        ]),
    ])
    .style(Style::default().bg(theme::BASE))
    .wrap(Wrap { trim: false });
    f.render_widget(message_widget, chunks[0]);

    // Help bar
    let help = Paragraph::new(vec![Line::from(vec![
        Span::styled(
            " Enter ",
            Style::default()
                .fg(theme::GREEN)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("Confirm", Style::default().fg(theme::SUBTEXT1)),
        Span::raw("   "),
        Span::styled(
            " Esc ",
            Style::default().fg(theme::RED).add_modifier(Modifier::BOLD),
        ),
        Span::styled("Cancel", Style::default().fg(theme::SUBTEXT1)),
    ])])
    .style(Style::default().fg(theme::TEXT).bg(theme::BASE))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::SURFACE2)),
    );
    f.render_widget(help, chunks[2]);
}
