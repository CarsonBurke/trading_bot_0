use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
};

#[allow(dead_code)]
pub fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

/// Creates a centered dialog with adaptive sizing based on terminal height
/// Uses row-based min/max constraints instead of percentages for better small screen support
pub fn centered_dialog(width_percent: u16, min_height: u16, max_height: u16, r: Rect) -> Rect {
    // Calculate height based on available space with min/max bounds
    let available_height = r.height;
    let target_height = if available_height <= min_height + 4 {
        // Very small screens: use most of available space
        available_height.saturating_sub(2)
    } else if available_height >= max_height + 10 {
        // Large screens: use max_height
        max_height
    } else {
        // Medium screens: scale between min and max
        ((available_height as f32 * 0.6).round() as u16).clamp(min_height, max_height)
    };

    let vertical_margin = (available_height.saturating_sub(target_height)) / 2;

    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(vertical_margin),
            Constraint::Length(target_height),
            Constraint::Length(vertical_margin),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - width_percent) / 2),
            Constraint::Percentage(width_percent),
            Constraint::Percentage((100 - width_percent) / 2),
        ])
        .split(popup_layout[1])[1]
}
