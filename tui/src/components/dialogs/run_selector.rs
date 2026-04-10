use chrono::NaiveDateTime;
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use super::base::BaseDialog;
use crate::{theme, RunInfo, RunSelectorPurpose};

fn friendly_date(name: &str) -> String {
    NaiveDateTime::parse_from_str(name, "%Y-%m-%d_%H-%M-%S")
        .map(|dt| dt.format("%b %-d, %-I:%M %p").to_string())
        .unwrap_or_else(|_| name.to_string())
}

pub fn render(f: &mut Frame, selected: usize, runs: &[RunInfo], purpose: &RunSelectorPurpose) {
    let title = match purpose {
        RunSelectorPurpose::View => " Switch Run ",
        RunSelectorPurpose::Train => " Select Run for Training ",
    };

    let inner = BaseDialog::new(title)
        .width(60)
        .min_height(8)
        .max_height(30)
        .render(f);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)])
        .split(inner);

    let visible_height = chunks[0].height as usize;
    let scroll = selected.saturating_sub(visible_height.saturating_sub(1));

    let lines: Vec<Line> = runs
        .iter()
        .enumerate()
        .skip(scroll)
        .take(visible_height)
        .map(|(i, run)| {
            let is_sel = i == selected;
            let marker = if run.is_active { "*" } else { " " };

            let style = if is_sel {
                Style::default()
                    .fg(theme::BASE)
                    .bg(theme::LAVENDER)
                    .add_modifier(Modifier::BOLD)
            } else if run.is_active {
                Style::default().fg(theme::GREEN)
            } else {
                Style::default().fg(theme::TEXT)
            };

            let detail = match purpose {
                RunSelectorPurpose::View => format!("  {} eps", run.gen_count),
                RunSelectorPurpose::Train => {
                    let w = run.weights.len();
                    if w > 0 {
                        format!("  {} eps, {} weights", run.gen_count, w)
                    } else {
                        format!("  {} eps", run.gen_count)
                    }
                }
            };

            Line::from(vec![
                Span::styled(
                    format!(" {}{} ", if is_sel { ">" } else { " " }, marker),
                    style,
                ),
                Span::styled(friendly_date(&run.name), style),
                Span::styled(
                    detail,
                    if is_sel {
                        style
                    } else {
                        Style::default().fg(theme::SUBTEXT0)
                    },
                ),
            ])
        })
        .collect();

    let list = Paragraph::new(lines).style(Style::default().bg(theme::BASE));
    f.render_widget(list, chunks[0]);

    let extra_hint = matches!(purpose, RunSelectorPurpose::Train);
    render_help_bar(
        f,
        chunks[1],
        match purpose {
            RunSelectorPurpose::View => "Switch",
            RunSelectorPurpose::Train => "Select",
        },
        extra_hint,
    );
}

pub fn render_weights(f: &mut Frame, run_name: &str, selected: usize, weights: &[String]) {
    let title = format!(" Weights: {} ", friendly_date(run_name));

    let inner = BaseDialog::new(&title)
        .width(50)
        .min_height(6)
        .max_height(20)
        .render(f);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)])
        .split(inner);

    let visible_height = chunks[0].height as usize;
    let scroll = selected.saturating_sub(visible_height.saturating_sub(1));

    let lines: Vec<Line> = weights
        .iter()
        .enumerate()
        .skip(scroll)
        .take(visible_height)
        .map(|(i, name)| {
            let is_sel = i == selected;
            let style = if is_sel {
                Style::default()
                    .fg(theme::BASE)
                    .bg(theme::LAVENDER)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme::TEXT)
            };

            Line::from(Span::styled(
                format!(" {} {}", if is_sel { ">" } else { " " }, name),
                style,
            ))
        })
        .collect();

    let list = Paragraph::new(lines).style(Style::default().bg(theme::BASE));
    f.render_widget(list, chunks[0]);

    render_help_bar(f, chunks[1], "Train", false);
}

fn render_help_bar(f: &mut Frame, area: ratatui::layout::Rect, action: &str, show_new: bool) {
    let mut spans = vec![
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
        Span::styled(
            format!(" {}  ", action),
            Style::default().fg(theme::SUBTEXT1),
        ),
    ];
    if show_new {
        spans.push(Span::styled(
            "n",
            Style::default()
                .fg(theme::PEACH)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(" New  ", Style::default().fg(theme::SUBTEXT1)));
    }
    spans.push(Span::styled(
        "Esc",
        Style::default().fg(theme::RED).add_modifier(Modifier::BOLD),
    ));
    spans.push(Span::styled(
        " Cancel",
        Style::default().fg(theme::SUBTEXT1),
    ));

    let help = Paragraph::new(vec![Line::from(spans)])
        .style(Style::default().bg(theme::BASE))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::SURFACE2)),
        );
    f.render_widget(help, area);
}
