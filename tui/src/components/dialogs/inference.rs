use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::{theme, App, InferenceField};
use super::base::BaseDialog;

pub fn render(f: &mut Frame, app: &App, focused: InferenceField) {
    let inner = BaseDialog::new(" Run Inference ")
        .width(70)
        .min_height(18)
        .max_height(22)
        .render(f);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Help/hint text
            Constraint::Length(3), // Weights field
            Constraint::Length(1), // Spacer
            Constraint::Length(3), // Ticker field
            Constraint::Length(1), // Spacer
            Constraint::Length(3), // Episodes field
            Constraint::Min(1),    // Spacer
            Constraint::Length(3), // Bottom help bar
        ])
        .split(inner);

    // Top hint
    let help = Paragraph::new(vec![Line::from(vec![
        Span::styled("Tab", Style::default().fg(theme::PEACH)),
        Span::styled(": Navigate  ", Style::default().fg(theme::SUBTEXT1)),
        Span::styled("Enter", Style::default().fg(theme::GREEN)),
        Span::styled(": Start  ", Style::default().fg(theme::SUBTEXT1)),
        Span::styled("Esc", Style::default().fg(theme::RED)),
        Span::styled(": Cancel  ", Style::default().fg(theme::SUBTEXT1)),
        Span::styled("Tip: ", Style::default().fg(theme::SUBTEXT0)),
        Span::styled("'400' -> 'ppo_ep400.ot'", Style::default().fg(theme::TEAL)),
    ])])
    .style(Style::default().bg(theme::BASE));
    f.render_widget(help, chunks[0]);

    // Weights field
    let weights_focused = focused == InferenceField::Weights;
    let weights_border = if weights_focused { theme::BLUE } else { theme::SURFACE2 };
    let weights_line = if app.input.is_empty() {
        Line::from(vec![Span::styled(" infer.ot (default)", Style::default().fg(theme::OVERLAY0))])
    } else {
        Line::from(vec![Span::styled(
            format!(" {}{}", app.input, if weights_focused { "" } else { "" }),
            Style::default().fg(theme::TEXT),
        )])
    };
    let weights_widget = Paragraph::new(vec![weights_line])
        .style(Style::default().bg(theme::MANTLE))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(weights_border))
                .title(" Weights File ")
                .title_style(Style::default().fg(theme::SKY)),
        );
    f.render_widget(weights_widget, chunks[1]);

    // Ticker field
    let ticker_focused = focused == InferenceField::Ticker;
    let ticker_border = if ticker_focused { theme::BLUE } else { theme::SURFACE2 };
    let ticker_line = if app.ticker_input.is_empty() {
        Line::from(vec![Span::styled(" (use default tickers)", Style::default().fg(theme::OVERLAY0))])
    } else {
        Line::from(vec![Span::styled(
            format!(" {}{}", app.ticker_input, if ticker_focused { "" } else { "" }),
            Style::default().fg(theme::TEXT),
        )])
    };
    let ticker_widget = Paragraph::new(vec![ticker_line])
        .style(Style::default().bg(theme::MANTLE))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(ticker_border))
                .title(" Ticker Override (optional) ")
                .title_style(Style::default().fg(theme::SKY)),
        );
    f.render_widget(ticker_widget, chunks[3]);

    // Episodes field
    let episodes_focused = focused == InferenceField::Episodes;
    let episodes_border = if episodes_focused { theme::BLUE } else { theme::SURFACE2 };
    let episodes_line = if app.episodes_input.is_empty() {
        Line::from(vec![Span::styled(" 10 (default)", Style::default().fg(theme::OVERLAY0))])
    } else {
        Line::from(vec![Span::styled(
            format!(" {}{}", app.episodes_input, if episodes_focused { "" } else { "" }),
            Style::default().fg(theme::TEXT),
        )])
    };
    let episodes_widget = Paragraph::new(vec![episodes_line])
        .style(Style::default().bg(theme::MANTLE))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(episodes_border))
                .title(" Number of Episodes ")
                .title_style(Style::default().fg(theme::SKY)),
        );
    f.render_widget(episodes_widget, chunks[5]);

    // Bottom help bar
    let help = Paragraph::new(vec![Line::from(vec![
        Span::styled(" Enter ", Style::default().fg(theme::GREEN).add_modifier(Modifier::BOLD)),
        Span::styled("Start Inference", Style::default().fg(theme::SUBTEXT1)),
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
    f.render_widget(help, chunks[7]);
}
