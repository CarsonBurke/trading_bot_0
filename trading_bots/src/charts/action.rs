use plotters::{
    prelude::{BitMapBackend, IntoDrawingArea},
    series::{AreaSeries, LineSeries},
    style::{Color, ShapeStyle},
};
use shared::theme::plotters_colors as theme;

use super::utils::CHART_DIMS;
use crate::constants::CHART_IMAGE_FORMAT;

pub fn reward_chart(dir: &String, rewards: &Vec<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/reward.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), CHART_DIMS).into_drawing_area();
    root.fill(&theme::BASE)?;

    let y_min = rewards.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = rewards.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_range = (y_max - y_min).max(0.01);
    let y_min = y_min - y_range * 0.05;
    let y_max = y_max + y_range * 0.05;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Rewards", ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..rewards.len() as u32, y_min..y_max)?;

    chart
        .configure_mesh()
        .label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .light_line_style(&theme::SURFACE0)
        .draw()?;

    chart.draw_series(
        AreaSeries::new(
            rewards
                .iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value)),
            0.0,
            theme::YELLOW.mix(0.2),
        )
        .border_style(ShapeStyle::from(&theme::YELLOW).stroke_width(1)),
    )?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}

pub fn hold_action_chart(
    dir: &String,
    hold_actions: &Vec<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/hold_action.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), CHART_DIMS).into_drawing_area();
    root.fill(&theme::BASE)?;

    let y_min = hold_actions.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = hold_actions
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let y_range = (y_max - y_min).max(0.01);
    let y_min = y_min - y_range * 0.05;
    let y_max = y_max + y_range * 0.05;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Hold Action", ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..hold_actions.len() as u32, y_min..y_max)?;

    chart
        .configure_mesh()
        .label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .light_line_style(&theme::SURFACE0)
        .draw()?;

    chart.draw_series(LineSeries::new(
        hold_actions
            .iter()
            .enumerate()
            .map(|(index, value)| (index as u32, *value)),
        ShapeStyle::from(&theme::MAUVE).stroke_width(1),
    ))?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}

pub fn raw_action_chart(
    dir: &String,
    raw_actions: &Vec<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/raw_action.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), CHART_DIMS).into_drawing_area();
    root.fill(&theme::BASE)?;

    if raw_actions.is_empty() {
        return Ok(());
    }

    let data_min = raw_actions
        .iter()
        .filter(|x| x.is_finite())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let data_max = raw_actions
        .iter()
        .filter(|x| x.is_finite())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = data_min.min(0.0);
    let y_max = data_max.max(1.0);
    let y_range = (y_max - y_min).max(0.01);
    let y_min = y_min - y_range * 0.05;
    let y_max = y_max + y_range * 0.05;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Raw Action", ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..raw_actions.len() as u32, y_min..y_max)?;

    chart
        .configure_mesh()
        .label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .light_line_style(&theme::SURFACE0)
        .draw()?;

    chart.draw_series(LineSeries::new(
        raw_actions
            .iter()
            .enumerate()
            .map(|(index, value)| (index as u32, *value)),
        ShapeStyle::from(&theme::RED).stroke_width(1),
    ))?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}
