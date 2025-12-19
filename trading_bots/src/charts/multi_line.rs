use plotters::{
    prelude::{BitMapBackend, IntoDrawingArea, SeriesLabelPosition},
    series::LineSeries,
    style::{Color, ShapeStyle},
};
use shared::theme::plotters_colors as theme;

use crate::constants::CHART_IMAGE_FORMAT;
use crate::types::Data;

pub fn multi_line_chart(
    dir: &String,
    name: &str,
    series: &[(&str, &Data)],
    x_scale: u32,
    x_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    multi_line_chart_with_labels(dir, name, series, x_scale, Some(x_label), None)
}

pub fn multi_line_chart_with_labels(
    dir: &String,
    name: &str,
    series: &[(&str, &Data)],
    x_scale: u32,
    x_label: Option<&str>,
    y_label: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/{name}.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), (2560, 780)).into_drawing_area();
    root.fill(&theme::BASE)?;

    let colors = [
        &theme::BLUE,
        &theme::GREEN,
        &theme::RED,
        &theme::YELLOW,
        &theme::MAUVE,
    ];

    let all_values: Vec<f64> = series
        .iter()
        .flat_map(|(_, data)| data.iter().copied())
        .collect();
    if all_values.is_empty() {
        return Ok(());
    }

    let y_min = all_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = all_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_range = (y_max - y_min).max(0.01);
    let y_min = y_min - y_range * 0.05;
    let y_max = y_max + y_range * 0.05;
    let x_max = series
        .iter()
        .map(|(_, data)| data.len())
        .max()
        .unwrap_or(1) as u32
        * x_scale;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption(name, ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..x_max, y_min..y_max)?;

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

    for (i, (label, data)) in series.iter().enumerate() {
        let color = colors[i % colors.len()].mix(0.8);
        chart
            .draw_series(LineSeries::new(
                data.iter()
                    .enumerate()
                    .map(|(idx, v)| (idx as u32 * x_scale, *v)),
                ShapeStyle::from(&color).stroke_width(1),
            ))?
            .label(*label)
            .legend(move |(x, y)| {
                plotters::element::Rectangle::new([(x, y - 5), (x + 20, y + 5)], color.filled())
            });
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(theme::SURFACE0.mix(0.7))
        .border_style(&theme::SURFACE1)
        .label_font(("sans-serif", 14, &theme::TEXT))
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

pub fn multi_line_chart_log(
    dir: &String,
    name: &str,
    series: &[(&str, &Data)],
    x_scale: u32,
    x_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    multi_line_chart_log_with_labels(dir, name, series, x_scale, Some(x_label), None)
}

pub fn multi_line_chart_log_with_labels(
    dir: &String,
    name: &str,
    series: &[(&str, &Data)],
    x_scale: u32,
    x_label: Option<&str>,
    y_label: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/{name}.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), (2560, 780)).into_drawing_area();
    root.fill(&theme::BASE)?;

    let colors = [
        &theme::BLUE,
        &theme::GREEN,
        &theme::RED,
        &theme::YELLOW,
        &theme::MAUVE,
    ];

    let all_values: Vec<f64> = series
        .iter()
        .flat_map(|(_, data)| data.iter().copied())
        .filter(|v| v.is_finite())
        .collect();
    if all_values.is_empty() {
        return Ok(());
    }

    let y_min = all_values
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
        .unwrap_or(0.0);
    let y_max = all_values
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
        .unwrap_or(1.0);
    let y_range = y_max - y_min;
    let y_min_t = symlog(y_min - y_range * 0.05);
    let y_max_t = symlog(y_max + y_range * 0.05);
    let x_max = series
        .iter()
        .map(|(_, data)| data.len())
        .max()
        .unwrap_or(1) as u32
        * x_scale;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption(name, ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(70)
        .build_cartesian_2d(0..x_max, y_min_t..y_max_t)?;

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

    for (i, (label, data)) in series.iter().enumerate() {
        let color = colors[i % colors.len()].mix(0.8);
        chart
            .draw_series(LineSeries::new(
                data.iter()
                    .enumerate()
                    .filter(|(_, v)| v.is_finite())
                    .map(|(idx, v)| (idx as u32 * x_scale, symlog(*v))),
                ShapeStyle::from(&color).stroke_width(1),
            ))?
            .label(*label)
            .legend(move |(x, y)| {
                plotters::element::Rectangle::new([(x, y - 5), (x + 20, y + 5)], color.filled())
            });
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(theme::SURFACE0.mix(0.7))
        .border_style(&theme::SURFACE1)
        .label_font(("sans-serif", 14, &theme::TEXT))
        .draw()?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}
