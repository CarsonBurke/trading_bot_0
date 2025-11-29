use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::{components::episode_status, theme, App};

pub fn render(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(4),
        ])
        .split(f.area());

    let is_training_for_title = app.is_training_running();
    let current_episode_for_title = app.get_current_episode();

    let mut title_spans = vec![
        Span::styled(" Trading Bot TUI ", Style::default().fg(theme::MAUVE).add_modifier(Modifier::BOLD)),
    ];
    title_spans.extend(episode_status::episode_status_spans(is_training_for_title, current_episode_for_title));

    let title = Paragraph::new(Line::from(title_spans))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::LAVENDER)),
        );
    f.render_widget(title, chunks[0]);

    // Empty block for middle section
    let empty_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::SURFACE2));
    f.render_widget(empty_block, chunks[1]);

    let help_text = vec![
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
            Span::raw(": View Meta Charts  "),
            Span::styled("v", Style::default().fg(Color::Magenta)),
            Span::raw(": Model Observations"),
        ]),
    ];

    let help = Paragraph::new(help_text).block(Block::default().borders(Borders::ALL).title(" Controls "));
    f.render_widget(help, chunks[2]);
}

