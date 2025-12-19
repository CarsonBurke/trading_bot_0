use plotters::{prelude::SeriesLabelPosition, style::Color};
use shared::theme::plotters_colors as theme;

pub const CHART_DIMS: (u32, u32) = (2560, 780);

pub struct LegendConfig;

impl LegendConfig {
    pub fn position() -> SeriesLabelPosition {
        SeriesLabelPosition::UpperRight
    }

    pub fn background() -> plotters::style::RGBAColor {
        theme::SURFACE0.mix(0.5)
    }

    pub fn border() -> &'static plotters::style::RGBColor {
        &theme::SURFACE1
    }

    pub fn font() -> (&'static str, i32, &'static plotters::style::RGBColor) {
        ("sans-serif", 14, &theme::TEXT)
    }
}

pub fn legend_rect(
    color: &impl Color,
) -> impl Fn((i32, i32)) -> plotters::element::Rectangle<(i32, i32)> + '_ {
    move |(x, y)| {
        plotters::element::Rectangle::new([(x, y - 5), (x + 20, y + 5)], color.mix(0.8).filled())
    }
}
