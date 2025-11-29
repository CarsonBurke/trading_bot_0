use chrono::{DateTime, Local, NaiveDateTime, TimeZone};
use hashbrown::HashMap;
use ibapi::market_data::{historical, realtime};
use plotters::{
    coord::types::RangedCoordf32,
    data,
    prelude::{BitMapBackend, CandleStick, Circle, EmptyElement, IntoDrawingArea, Text},
    series::{AreaSeries, LineSeries, PointSeries},
    style::{Color, ShapeStyle},
};
use shared::theme::plotters_colors as theme;
use time::OffsetDateTime;

use crate::{constants::{CHART_IMAGE_FORMAT}, types::Data};

pub fn chart(data: &Data) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("charts/chart.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), (2560, 800)).into_drawing_area();
    root.fill(&theme::BASE)?;

    let y_min = *data
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap() as f32;
    let y_max = *data
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap() as f32;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Chart", ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..data.len() as u32, y_min..y_max)?;

    chart.configure_mesh()
        .label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .light_line_style(&theme::SURFACE0)
        .draw()?;

    chart.draw_series(data.iter().enumerate().map(|(index, x)| {
        CandleStick::new(index as u32, 0., 0., 0., *x as f32, theme::GREEN.filled(), &theme::RED, 15)
    }))?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}

pub fn candle_chart(
    dir: &String,
    bars: &[historical::Bar],
) -> Result<(), Box<dyn std::error::Error>> {
    let dimensions = (2560, 800);
    let path = format!("{dir}/candle.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), dimensions).into_drawing_area();
    root.fill(&theme::BASE)?;

    let y_min = bars
        .iter()
        .map(|bar| bar.close)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap() as f32
        * 0.9;
    let y_max = bars
        .iter()
        .map(|bar| bar.close)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap() as f32
        * 1.1;

    let dates = bars
        .iter()
        .map(|bar| bar.date)
        .collect::<Vec<OffsetDateTime>>();

    let mut chrono_dates = Vec::new();

    for date in dates {
        chrono_dates.push({
            let x = DateTime::from_timestamp_nanos(date.unix_timestamp_nanos() as i64);
            Local.from_utc_datetime(&x.naive_local()).time()
        });
    }
    let chrono_range = chrono_dates[0]..chrono_dates[chrono_dates.len() - 1];

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Candle Chart", ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..bars.len() as u32, y_min..y_max)?;

    chart.configure_mesh().label_style(("sans-serif", 15, &theme::TEXT)).axis_style(&theme::SURFACE1).light_line_style(&theme::SURFACE0).draw()?;

    let mut draw = chart.draw_series(bars.iter().enumerate().map(|(index, bar)| {
        /* let local = Local.from_utc_datetime(&DateTime::from_timestamp_nanos(bar.date.unix_timestamp_nanos() as i64).naive_local());
        let chrono_date = local.timestamp();
        let z = local.time();
        println!("time: {z:?}"); */
        CandleStick::new(
            index as u32,
            bar.open as f32,
            bar.high as f32,
            bar.low as f32,
            bar.close as f32,
            theme::GREEN.filled(),
            theme::RED.filled(),
            dimensions.0 / bars.len() as u32,
        )
    }))?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}

pub fn simple_chart(
    dir: &String,
    name: &str,
    data: &Data,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/{name}.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), (2560, 800)).into_drawing_area();
    root.fill(&theme::BASE)?;

    let y_min = data
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        * 0.9;

    let y_max = data
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        * 1.1;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption(name, ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..data.len() as u32, y_min..y_max)?;

    chart.configure_mesh().label_style(("sans-serif", 15, &theme::TEXT)).axis_style(&theme::SURFACE1).light_line_style(&theme::SURFACE0).draw()?;

    chart.draw_series(
        AreaSeries::new(
            data.iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value)),
            0.0,
            theme::BLUE.mix(0.2),
        )
        .border_style(&theme::BLUE),
    )?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}

pub fn buy_sell_chart(
    dir: &String,
    data: &Data,
    buy_indexes: &HashMap<usize, (f64, f64)>,
    sell_indexes: &HashMap<usize, (f64, f64)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/buy_sell.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), (2560, 800)).into_drawing_area();
    root.fill(&theme::BASE)?;

    let y_min = data
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        * 0.9;
    let y_max = data
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        * 1.1;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Buy Sell Chart", ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..data.len() as u32, y_min..y_max)?;

    chart.configure_mesh().label_style(("sans-serif", 15, &theme::TEXT)).axis_style(&theme::SURFACE1).light_line_style(&theme::SURFACE0).draw()?;

    // Data
    chart.draw_series(
        AreaSeries::new(
            data.iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value)),
            0.0,
            theme::BLUE.mix(0.2),
        )
        .border_style(&theme::BLUE),
    )?;

    let point_size = 4;

    // Sells
    chart.draw_series(PointSeries::of_element(
        sell_indexes
            .iter()
            .map(|(index, value)| (*index as u32, value.0)),
        point_size,
        theme::YELLOW.mix(0.9).filled(),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    // Buys
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
    let root = BitMapBackend::new(path.as_str(), (2560, 800)).into_drawing_area();
    root.fill(&theme::BASE)?;

    let y_min = data
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        * 0.9;
    let y_max = data
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        * 1.1;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Buy Sell Chart Vec", ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..data.len() as u32, y_min..y_max)?;

    chart.configure_mesh().label_style(("sans-serif", 15, &theme::TEXT)).axis_style(&theme::SURFACE1).light_line_style(&theme::SURFACE0).draw()?;

    // Data
    chart.draw_series(
        AreaSeries::new(
            data.iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value)),
            0.0,
            theme::BLUE.mix(0.2),
        )
        .border_style(&theme::BLUE),
    )?;

    let point_size = 4;

    // Sells
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

    // Buys
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
    let root = BitMapBackend::new(path.as_str(), (2560, 800)).into_drawing_area();
    root.fill(&theme::BASE)?;

    let mut max_val = *assets
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    if let Some(bench) = benchmark {
        let bench_max = *bench
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        max_val = max_val.max(bench_max);
    }

    let y_max = max_val as f32 * 1.1;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Assets: Total; Positioned; Cash", ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..assets.len() as u32, 0.0..y_max)?;

    chart.configure_mesh().label_style(("sans-serif", 15, &theme::TEXT)).axis_style(&theme::SURFACE1).light_line_style(&theme::SURFACE0).draw()?;

    chart.draw_series(
        AreaSeries::new(
            assets
                .iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value as f32)),
            0.0,
            theme::BLUE.mix(0.2),
        )
        .border_style(&theme::BLUE),
    )?;

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

    chart.draw_series(
        AreaSeries::new(
            positioned
                .iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value as f32)),
            0.0,
            theme::RED.mix(0.2),
        )
        .border_style(&theme::RED),
    )?;

    chart.draw_series(
        AreaSeries::new(
            cash.iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value as f32)),
            0.0,
            theme::GREEN.mix(0.2),
        )
        .border_style(&theme::GREEN),
    )?;

    if let Some(bench) = benchmark {
        chart.draw_series(LineSeries::new(
            bench.iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value as f32)),
            ShapeStyle::from(&theme::YELLOW).stroke_width(2),
        ))?;
    }

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
    let root = BitMapBackend::new(path.as_str(), (2560, 800)).into_drawing_area();
    root.fill(&theme::BASE)?;

    let smoothed_wants = (0..data.len())
        .map(|index| (index as u32, *wants.get(&index).unwrap_or(&0.)))
        .collect::<Vec<(u32, f64)>>();

    let y_min = wants
        .iter()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap_or((&0, &0.))
        .1
        * 0.9;
    let y_max = wants
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap_or((&0, &0.))
        .1
        * 1.1;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Buy Sell Chart", ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..data.len() as u32, y_min..y_max)?;

    chart.configure_mesh().label_style(("sans-serif", 15, &theme::TEXT)).axis_style(&theme::SURFACE1).light_line_style(&theme::SURFACE0).draw()?;

    // Wants

    chart.draw_series(
        AreaSeries::new(
            smoothed_wants,
            /* wants.iter()
            .map(|(index, value)| (*index as u32, *value)), */
            0.0,
            theme::MAUVE.mix(0.2),
        )
        .border_style(&theme::MAUVE),
    )?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}

pub fn reward_chart(dir: &String, rewards: &Vec<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/reward.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), (2560, 800)).into_drawing_area();
    root.fill(&theme::BASE)?;

    let y_min = rewards
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&0.)
        * 0.9;
    let y_max = rewards
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&0.)
        * 1.1;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Rewards", ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..rewards.len() as u32, y_min..y_max)?;

    chart.configure_mesh().label_style(("sans-serif", 15, &theme::TEXT)).axis_style(&theme::SURFACE1).light_line_style(&theme::SURFACE0).draw()?;

    // Rewards

    chart.draw_series(
        AreaSeries::new(
            rewards
                .iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value)),
            0.0,
            theme::YELLOW.mix(0.2),
        )
        .border_style(&theme::YELLOW),
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
    let root = BitMapBackend::new(path.as_str(), (2560, 800)).into_drawing_area();
    root.fill(&theme::BASE)?;

    let y_min = hold_actions
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&-1.0)
        * 1.1;
    let y_max = hold_actions
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&1.0)
        * 1.1;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Hold Action (range: -1 to 1)", ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..hold_actions.len() as u32, y_min..y_max)?;

    chart.configure_mesh().label_style(("sans-serif", 15, &theme::TEXT)).axis_style(&theme::SURFACE1).light_line_style(&theme::SURFACE0).draw()?;

    // Hold actions as line series
    chart.draw_series(LineSeries::new(
        hold_actions
            .iter()
            .enumerate()
            .map(|(index, value)| (index as u32, *value)),
        &theme::MAUVE,
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
    let root = BitMapBackend::new(path.as_str(), (2560, 800)).into_drawing_area();
    root.fill(&theme::BASE)?;

    let y_min = raw_actions
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&-1.0)
        * 1.1;
    let y_max = raw_actions
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&1.0)
        * 1.1;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Raw Buy/Sell Action Output", ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..raw_actions.len() as u32, y_min..y_max)?;

    chart.configure_mesh().label_style(("sans-serif", 15, &theme::TEXT)).axis_style(&theme::SURFACE1).light_line_style(&theme::SURFACE0).draw()?;

    chart.draw_series(LineSeries::new(
        raw_actions
            .iter()
            .enumerate()
            .map(|(index, value)| (index as u32, *value)),
        &theme::RED,
    ))?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}
