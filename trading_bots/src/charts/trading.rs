use hashbrown::HashMap;
use plotters::{
    prelude::{BitMapBackend, Circle, EmptyElement, IntoDrawingArea},
    series::{AreaSeries, LineSeries, PointSeries},
    style::{Color, ShapeStyle},
};
use shared::theme::plotters_colors as theme;

use super::utils::{legend_rect, LegendConfig, CHART_DIMS};
use crate::constants::CHART_IMAGE_FORMAT;
use crate::types::Data;

pub fn buy_sell_chart(
    dir: &String,
    data: &Data,
    buy_indexes: &HashMap<usize, (f64, f64)>,
    sell_indexes: &HashMap<usize, (f64, f64)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/buy_sell.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), CHART_DIMS).into_drawing_area();
    root.fill(&theme::BASE)?;

    let y_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_range = (y_max - y_min).max(0.01);
    let y_min = y_min - y_range * 0.05;
    let y_max = y_max + y_range * 0.05;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Buy Sell", ("sans-serif", 20, &theme::TEXT))
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

    let point_size = 3;

    chart.draw_series(PointSeries::of_element(
        sell_indexes
            .iter()
            .map(|(index, value)| (*index as u32, value.0)),
        point_size,
        theme::YELLOW.mix(0.9).filled(),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    chart.draw_series(PointSeries::of_element(
        buy_indexes
            .iter()
            .map(|(index, value)| (*index as u32, value.0)),
        point_size,
        theme::RED.mix(0.9).filled(),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}

pub fn buy_sell_chart_vec(
    dir: &String,
    data: &Data,
    buy_indexes: &[f64],
    sell_indexes: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/buy_sell_vec.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), CHART_DIMS).into_drawing_area();
    root.fill(&theme::BASE)?;

    let y_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_range = (y_max - y_min).max(0.01);
    let y_min = y_min - y_range * 0.05;
    let y_max = y_max + y_range * 0.05;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Buy Sell Vec", ("sans-serif", 20, &theme::TEXT))
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

    let point_size = 3;

    chart.draw_series(PointSeries::of_element(
        sell_indexes
            .iter()
            .enumerate()
            .filter(|(_, &value)| value > 0.0)
            .map(|(index, _)| (index as u32, data[index])),
        point_size,
        theme::YELLOW.mix(0.9).filled(),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    chart.draw_series(PointSeries::of_element(
        buy_indexes
            .iter()
            .enumerate()
            .filter(|(_, &value)| value > 0.0)
            .map(|(index, _)| (index as u32, data[index])),
        point_size,
        theme::RED.mix(0.9).filled(),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}

pub fn assets_chart(
    dir: &String,
    assets: &Data,
    cash: &Data,
    positioned: Option<&Data>,
    benchmark: Option<&Data>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/assets.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), CHART_DIMS).into_drawing_area();
    root.fill(&theme::BASE)?;

    let mut max_val = assets.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if let Some(bench) = benchmark {
        let bench_max = bench.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        max_val = max_val.max(bench_max);
    }

    let y_max = max_val as f32 * 1.1;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Assets", ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..assets.len() as u32, 0.0..y_max)?;

    chart
        .configure_mesh()
        .label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .light_line_style(&theme::SURFACE0)
        .draw()?;

    chart
        .draw_series(
            AreaSeries::new(
                assets
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (index as u32, *value as f32)),
                0.0,
                theme::BLUE.mix(0.2),
            )
            .border_style(ShapeStyle::from(&theme::BLUE).stroke_width(1)),
        )?
        .label("total")
        .legend(legend_rect(&theme::BLUE));

    let positioned = {
        if let Some(positioned) = positioned {
            positioned
        } else {
            &assets
                .iter()
                .zip(cash)
                .map(|(a, b)| a - b)
                .collect::<Vec<f64>>()
        }
    };

    chart
        .draw_series(
            AreaSeries::new(
                positioned
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (index as u32, *value as f32)),
                0.0,
                theme::RED.mix(0.2),
            )
            .border_style(ShapeStyle::from(&theme::RED).stroke_width(1)),
        )?
        .label("positioned")
        .legend(legend_rect(&theme::RED));

    chart
        .draw_series(
            AreaSeries::new(
                cash.iter()
                    .enumerate()
                    .map(|(index, value)| (index as u32, *value as f32)),
                0.0,
                theme::GREEN.mix(0.2),
            )
            .border_style(ShapeStyle::from(&theme::GREEN).stroke_width(1)),
        )?
        .label("cash")
        .legend(legend_rect(&theme::GREEN));

    if let Some(bench) = benchmark {
        chart
            .draw_series(LineSeries::new(
                bench
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (index as u32, *value as f32)),
                ShapeStyle::from(&theme::YELLOW).stroke_width(1),
            ))?
            .label("benchmark")
            .legend(legend_rect(&theme::YELLOW));
    }

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

pub fn want_chart(
    dir: &String,
    data: &Data,
    wants: &HashMap<usize, f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/want.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), CHART_DIMS).into_drawing_area();
    root.fill(&theme::BASE)?;

    let smoothed_wants = (0..data.len())
        .map(|index| (index as u32, *wants.get(&index).unwrap_or(&0.)))
        .collect::<Vec<(u32, f64)>>();

    let y_min = wants.values().cloned().fold(f64::INFINITY, f64::min);
    let y_max = wants.values().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_range = (y_max - y_min).max(0.01);
    let y_min = y_min - y_range * 0.05;
    let y_max = y_max + y_range * 0.05;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Want", ("sans-serif", 20, &theme::TEXT))
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
        AreaSeries::new(smoothed_wants, 0.0, theme::MAUVE.mix(0.2))
            .border_style(ShapeStyle::from(&theme::MAUVE).stroke_width(1)),
    )?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}
