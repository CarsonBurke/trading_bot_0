use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::{theme, App};
use super::base::BaseDialog;

pub fn render(f: &mut Frame, app: &App, for_training: bool, for_inference: bool) {
    let title = if for_training {
        " Start Training "
    } else if for_inference {
        " Run Inference "
    } else {
        " Weights Input "
    };

    let default_text = if for_inference {
        "infer.pt"
    } else {
        "none (train from scratch)"
    };

    let inner = BaseDialog::new(title)
        .width(70)
        .min_height(12)
        .max_height(16)
        .render(f);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Hint text
            Constraint::Length(3), // Input field
            Constraint::Min(1),    // Spacer
            Constraint::Length(3), // Help bar
        ])
        .split(inner);

    // Hint text
    let prompt_text = vec![Line::from(vec![
        Span::styled("Shorthand: ", Style::default().fg(theme::SUBTEXT0)),
        Span::styled("'400' -> 'ppo_ep400.pt'", Style::default().fg(theme::TEAL)),
        Span::raw("  |  "),
        Span::styled("Default: ", Style::default().fg(theme::SUBTEXT0)),
        Span::styled(default_text, Style::default().fg(theme::GREEN)),
    ])];
    let prompt = Paragraph::new(prompt_text)
        .style(Style::default().fg(theme::TEXT).bg(theme::BASE));
    f.render_widget(prompt, chunks[0]);

    // Input field
    let input_line = if app.input.is_empty() {
        Line::from(vec![Span::styled(
            format!(" {}", default_text),
            Style::default().fg(theme::OVERLAY0),
        )])
    } else {
        Line::from(vec![Span::styled(
            format!(" {}", app.input),
            Style::default().fg(theme::TEXT),
        )])
    };

    let input_widget = Paragraph::new(vec![input_line])
        .style(Style::default().bg(theme::MANTLE))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::BLUE))
                .title(" Filename ")
                .title_style(Style::default().fg(theme::SKY)),
        );
    f.render_widget(input_widget, chunks[1]);

    // Help bar
    let help = Paragraph::new(vec![Line::from(vec![
        Span::styled(" Enter ", Style::default().fg(theme::GREEN).add_modifier(Modifier::BOLD)),
        Span::styled("Start Training", Style::default().fg(theme::SUBTEXT1)),
        Span::raw("   "),
        Span::styled(" Esc ", Style::default().fg(theme::RED).add_modifier(Modifier::BOLD)),
        Span::styled("Cancel", Style::default().fg(theme::SUBTEXT1)),
    ])])
    .style(Style::default().fg(theme::TEXT).bg(theme::BASE))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::SURFACE2)),
    );
    f.render_widget(help, chunks[3]);
}
