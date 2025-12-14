use plotters::{
    prelude::{BitMapBackend, IntoDrawingArea},
    series::{AreaSeries, LineSeries},
    style::{Color, ShapeStyle},
};
use shared::theme::plotters_colors as theme;

use crate::constants::CHART_IMAGE_FORMAT;
use crate::types::Data;

pub(crate) fn compute_moving_avg(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window {
        return data.to_vec();
    }
    let mut result = Vec::with_capacity(data.len());
    let mut sum: f64 = data[..window].iter().sum();
    for _ in 0..window - 1 {
        result.push(f64::NAN);
    }
    result.push(sum / window as f64);
    for i in window..data.len() {
        sum += data[i] - data[i - window];
        result.push(sum / window as f64);
    }
    result
}

pub fn simple_chart(dir: &String, name: &str, data: &Data) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/{name}.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), (2560, 800)).into_drawing_area();
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

    chart
        .configure_mesh()
        .label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .light_line_style(&theme::SURFACE0)
        .draw()?;

    chart.draw_series(
        AreaSeries::new(
            data.iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value)),
            0.0,
            theme::BLUE.mix(0.2),
        )
        .border_style(ShapeStyle::from(&theme::BLUE).stroke_width(1)),
    )?;

    let window = (data.len() / 10).max(5).min(50);
    let ma = compute_moving_avg(data, window);
    chart.draw_series(LineSeries::new(
        ma.iter()
            .enumerate()
            .filter(|(_, v)| !v.is_nan())
            .map(|(i, v)| (i as u32, *v)),
        ShapeStyle::from(&theme::YELLOW).stroke_width(1),
    ))?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}

pub fn simple_chart_log(
    dir: &String,
    name: &str,
    data: &Data,
    x_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::coord::combinators::IntoLogRange;

    let path = format!("{dir}/{name}.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), (2560, 800)).into_drawing_area();
    root.fill(&theme::BASE)?;

    let y_min = data
        .iter()
        .filter(|v| **v > 0.0)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
        .unwrap_or(1e-10)
        .max(1e-10);

    let y_max = data
        .iter()
        .filter(|v| v.is_finite())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
        .unwrap_or(1.0)
        * 1.1;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption(name, ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(70)
        .build_cartesian_2d(0..data.len() as u32, (y_min..y_max).log_scale())?;

    chart
        .configure_mesh()
        .label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .light_line_style(&theme::SURFACE0)
        .x_desc(x_label)
        .y_label_formatter(&|v| format!("{:.2}", v))
        .draw()?;

    chart.draw_series(LineSeries::new(
        data.iter()
            .enumerate()
            .filter(|(_, v)| **v > 0.0)
            .map(|(index, value)| (index as u32, *value)),
        ShapeStyle::from(&theme::BLUE).stroke_width(1),
    ))?;

    let window = (data.len() / 10).max(5).min(50);
    let ma = compute_moving_avg(data, window);
    chart.draw_series(LineSeries::new(
        ma.iter()
            .enumerate()
            .filter(|(_, v)| !v.is_nan() && **v > 0.0)
            .map(|(i, v)| (i as u32, *v)),
        ShapeStyle::from(&theme::YELLOW).stroke_width(1),
    ))?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}
