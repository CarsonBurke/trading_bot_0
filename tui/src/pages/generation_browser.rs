use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

use crate::{components::episode_status, App};

pub fn render(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(4),
        ])
        .split(f.area());

    let is_training = app.is_training_running();
    let current_episode = app.get_current_episode();

    let mut title_spans = vec![
        Span::styled(" Training Episodes ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
    ];
    title_spans.extend(episode_status::episode_status_spans(is_training, current_episode));

    let title = Paragraph::new(Line::from(title_spans))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    let search_style = if app.generation_browser.searching {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let search_text = if app.generation_browser.searching {
        format!("Search: {}_", app.generation_browser.search_input)
    } else {
        format!("Search: {} (press / to search)", app.generation_browser.search_input)
    };

    let search = Paragraph::new(search_text)
        .style(search_style)
        .block(Block::default().borders(Borders::ALL).title("Filter"));
    f.render_widget(search, chunks[1]);

    let items: Vec<ListItem> = app
        .generation_browser.filtered_generations
        .iter()
        .map(|gen| {
            ListItem::new(format!("Episode {}", gen.number))
                .style(Style::default().fg(Color::White))
        })
        .collect();

    let list_title = format!(
        "Episodes ({}/{})",
        app.generation_browser.filtered_generations.len(),
        app.generation_browser.generations.len()
    );

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(list_title))
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");

    app.generation_browser.list_area = chunks[2];
    f.render_stateful_widget(list, chunks[2], &mut app.generation_browser.list_state);

    let help = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("↑/k", Style::default().fg(Color::Cyan)),
            Span::raw(": Up  "),
            Span::styled("↓/j", Style::default().fg(Color::Cyan)),
            Span::raw(": Down  "),
            Span::styled("Wheel", Style::default().fg(Color::Cyan)),
            Span::raw(": Scroll  "),
            Span::styled("Click", Style::default().fg(Color::Cyan)),
            Span::raw(": Select"),
        ]),
        Line::from(vec![
            Span::styled("/", Style::default().fg(Color::Yellow)),
            Span::raw(": Search  "),
            Span::styled("Enter", Style::default().fg(Color::Green)),
            Span::raw(": View  "),
            Span::styled("r", Style::default().fg(Color::Yellow)),
            Span::raw(": Refresh  "),
            Span::styled("q/Esc", Style::default().fg(Color::Red)),
            Span::raw(": Back"),
        ]),
    ])
    .block(Block::default().borders(Borders::ALL).title("Controls"));
    f.render_widget(help, chunks[3]);
}
