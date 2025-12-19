use plotters::{
    prelude::{BitMapBackend, IntoDrawingArea, SeriesLabelPosition},
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
    )?
    .label("value")
    .legend(|(x, y)| plotters::element::Rectangle::new([(x, y - 5), (x + 20, y + 5)], theme::BLUE.mix(0.8).filled()));

    let window = (data.len() / 10).max(5).min(50);
    let ma = compute_moving_avg(data, window);
    chart.draw_series(LineSeries::new(
        ma.iter()
            .enumerate()
            .filter(|(_, v)| !v.is_nan())
            .map(|(i, v)| (i as u32, *v)),
        ShapeStyle::from(&theme::YELLOW).stroke_width(1),
    ))?
    .label("MA")
    .legend(|(x, y)| plotters::element::Rectangle::new([(x, y - 5), (x + 20, y + 5)], theme::YELLOW.mix(0.8).filled()));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(&theme::SURFACE0)
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

pub fn simple_chart_log(
    dir: &String,
    name: &str,
    data: &Data,
    x_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/{name}.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), (2560, 800)).into_drawing_area();
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

    chart
        .configure_mesh()
        .label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .light_line_style(&theme::SURFACE0)
        .x_desc(x_label)
        .y_label_formatter(&|v| format!("{:.2e}", symlog_inv(*v)))
        .draw()?;

    chart.draw_series(LineSeries::new(
        data.iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(index, value)| (index as u32, symlog(*value))),
        ShapeStyle::from(&theme::BLUE).stroke_width(1),
    ))?
    .label("value")
    .legend(|(x, y)| plotters::element::Rectangle::new([(x, y - 5), (x + 20, y + 5)], theme::BLUE.mix(0.8).filled()));

    let window = (data.len() / 10).max(5).min(50);
    let ma = compute_moving_avg(data, window);
    chart.draw_series(LineSeries::new(
        ma.iter()
            .enumerate()
            .filter(|(_, v)| !v.is_nan())
            .map(|(i, v)| (i as u32, symlog(*v))),
        ShapeStyle::from(&theme::YELLOW).stroke_width(1),
    ))?
    .label("MA")
    .legend(|(x, y)| plotters::element::Rectangle::new([(x, y - 5), (x + 20, y + 5)], theme::YELLOW.mix(0.8).filled()));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(&theme::SURFACE0)
        .border_style(&theme::SURFACE1)
        .label_font(("sans-serif", 14, &theme::TEXT))
        .draw()?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}
