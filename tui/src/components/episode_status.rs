use ratatui::{
    style::{Modifier, Style},
    text::Span,
};

use crate::theme;

/// Creates spans for displaying training status and episode info
pub fn episode_status_spans(is_training: bool, current_episode: Option<usize>) -> Vec<Span<'static>> {
    let (status_text, status_icon) = if is_training {
        ("Training", "●")
    } else {
        ("Inactive", "○")
    };
    let status_color = if is_training { theme::GREEN } else { theme::RED };

    let episode_display = if is_training {
        if let Some(ep) = current_episode {
            format!("{} ", ep)
        } else {
            "Starting... ".to_string()
        }
    } else {
        "N/A ".to_string()
    };

    vec![
        Span::raw("   "),
        Span::styled(format!("{} ", status_text), Style::default().fg(theme::SUBTEXT1)),
        Span::styled(format!("{}", status_icon), Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
        Span::raw("   "),
        Span::styled("Episode: ", Style::default().fg(theme::SUBTEXT1)),
        Span::styled(episode_display, Style::default().fg(theme::SKY)),
        Span::raw("   "),
    ]
}
