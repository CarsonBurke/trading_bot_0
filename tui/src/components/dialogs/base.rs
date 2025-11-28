use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    widgets::{Block, Borders},
    Frame,
};

use crate::{theme, utils::layout::centered_rect};

/// Dialog style variants for different contexts
#[derive(Clone, Copy)]
pub enum DialogStyle {
    /// Info dialogs (mauve title, lavender border)
    Info,
    /// Warning/confirm dialogs (peach title, maroon border)
    Warning,
}

/// Base dialog component that handles common rendering:
/// - Backdrop overlay
/// - Solid background
/// - Bordered dialog with title
pub struct BaseDialog<'a> {
    title: &'a str,
    width_percent: u16,
    height_percent: u16,
    style: DialogStyle,
}

impl<'a> BaseDialog<'a> {
    pub fn new(title: &'a str) -> Self {
        Self {
            title,
            width_percent: 60,
            height_percent: 30,
            style: DialogStyle::Info,
        }
    }

    pub fn width(mut self, percent: u16) -> Self {
        self.width_percent = percent;
        self
    }

    pub fn height(mut self, percent: u16) -> Self {
        self.height_percent = percent;
        self
    }

    pub fn style(mut self, style: DialogStyle) -> Self {
        self.style = style;
        self
    }

    /// Renders the dialog frame and returns the inner area for content
    pub fn render(&self, f: &mut Frame) -> Rect {
        let area = centered_rect(self.width_percent, self.height_percent, f.area());

        // Soft backdrop overlay
        let backdrop = Block::default().style(Style::default().bg(theme::CRUST));
        f.render_widget(backdrop, f.area());

        // Dialog border and title
        let (title_color, border_color) = match self.style {
            DialogStyle::Info => (theme::MAUVE, theme::LAVENDER),
            DialogStyle::Warning => (theme::PEACH, theme::MAROON),
        };

        let dialog = Block::default()
            .title(self.title)
            .title_style(
                Style::default()
                    .fg(title_color)
                    .add_modifier(Modifier::BOLD),
            )
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))
            .style(Style::default().bg(theme::BASE));

        let inner = dialog.inner(area);
        f.render_widget(dialog, area);

        inner
    }
}
