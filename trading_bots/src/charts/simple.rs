use plotters::{
    prelude::{BitMapBackend, IntoDrawingArea},
    series::{AreaSeries, LineSeries},
    style::{Color, ShapeStyle},
};
use shared::theme::plotters_colors as theme;

use super::utils::{legend_rect, LegendConfig, CHART_DIMS};
use crate::constants::CHART_IMAGE_FORMAT;
use crate::types::Data;

pub(crate) fn compute_ema(data: &[f64], alpha: f64) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }
    let mut result = Vec::with_capacity(data.len());
    let mut ema = data[0];
    for &v in data {
        ema = alpha * v + (1.0 - alpha) * ema;
        result.push(ema);
    }
    result
}

pub fn simple_chart(dir: &String, name: &str, data: &Data) -> Result<(), Box<dyn std::error::Error>> {
    simple_chart_with_labels(dir, name, data, None, None)
}

pub fn simple_chart_with_labels(
    dir: &String,
    name: &str,
    data: &Data,
    x_label: Option<&str>,
    y_label: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/{name}.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), CHART_DIMS).into_drawing_area();
    root.fill(&theme::BASE)?;

    let y_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_range = (y_max - y_min).max(0.01);
    let y_min = y_min - y_range * 0.05;
    let y_max = y_max + y_range * 0.05;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption(name, ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..data.len() as u32, y_min..y_max)?;

    let mut mesh = chart.configure_mesh();
    mesh.label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .light_line_style(&theme::SURFACE0);
    if let Some(x) = x_label {
        mesh.x_desc(x);
    }
    if let Some(y) = y_label {
        mesh.y_desc(y);
    }
    mesh.draw()?;

    chart
        .draw_series(
            AreaSeries::new(
                data.iter()
                    .enumerate()
                    .map(|(index, value)| (index as u32, *value)),
                0.0,
                theme::BLUE.mix(0.2),
            )
            .border_style(ShapeStyle::from(&theme::BLUE).stroke_width(1)),
        )?
        .label("value")
        .legend(legend_rect(&theme::BLUE));

    let alpha = 0.05;
    let ema = compute_ema(data, alpha);
    chart
        .draw_series(LineSeries::new(
            ema.iter()
                .enumerate()
                .map(|(i, v)| (i as u32, *v)),
            ShapeStyle::from(&theme::YELLOW).stroke_width(1),
        ))?
        .label("EMA")
        .legend(legend_rect(&theme::YELLOW));

    chart
        .configure_series_labels()
        .position(LegendConfig::position())
        .background_style(LegendConfig::background())
        .border_style(LegendConfig::border())
        .label_font(LegendConfig::font())
        .draw()?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}

fn symlog(x: f64) -> f64 {
    x.signum() * (1.0 + x.abs()).ln()
}

fn symlog_inv(y: f64) -> f64 {
    y.signum() * (y.abs().exp() - 1.0)
}

pub fn simple_chart_log(
    dir: &String,
    name: &str,
    data: &Data,
    x_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    simple_chart_log_with_labels(dir, name, data, Some(x_label), None)
}

pub fn simple_chart_log_with_labels(
    dir: &String,
    name: &str,
    data: &Data,
    x_label: Option<&str>,
    y_label: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/{name}.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), CHART_DIMS).into_drawing_area();
    root.fill(&theme::BASE)?;

    let finite_data: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
    if finite_data.is_empty() {
        return Ok(());
    }

    let y_min = finite_data
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
        .unwrap_or(0.0);
    let y_max = finite_data
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
        .unwrap_or(1.0);
    let y_range = y_max - y_min;
    let y_min_t = symlog(y_min - y_range * 0.05);
    let y_max_t = symlog(y_max + y_range * 0.05);

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption(name, ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(70)
        .build_cartesian_2d(0..data.len() as u32, y_min_t..y_max_t)?;

    let mut mesh = chart.configure_mesh();
    mesh.label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .light_line_style(&theme::SURFACE0)
        .y_label_formatter(&|v| format!("{:.2e}", symlog_inv(*v)));
    if let Some(x) = x_label {
        mesh.x_desc(x);
    }
    if let Some(y) = y_label {
        mesh.y_desc(y);
    }
    mesh.draw()?;

    chart
        .draw_series(LineSeries::new(
            data.iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .map(|(index, value)| (index as u32, symlog(*value))),
            ShapeStyle::from(&theme::BLUE).stroke_width(1),
        ))?
        .label("value")
        .legend(legend_rect(&theme::BLUE));

    let alpha = 0.05;
    let ema = compute_ema(data, alpha);
    chart
        .draw_series(LineSeries::new(
            ema.iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .map(|(i, v)| (i as u32, symlog(*v))),
            ShapeStyle::from(&theme::YELLOW).stroke_width(1),
        ))?
        .label("EMA")
        .legend(legend_rect(&theme::YELLOW));

    chart
        .configure_series_labels()
        .position(LegendConfig::position())
        .background_style(LegendConfig::background())
        .border_style(LegendConfig::border())
        .label_font(LegendConfig::font())
        .draw()?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}
