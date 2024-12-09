use chrono::{DateTime, Local, NaiveDateTime, TimeZone};
use hashbrown::HashMap;
use ibapi::market_data::{historical, realtime};
use plotters::{
    coord::types::RangedCoordf32,
    data,
    prelude::{BitMapBackend, CandleStick, Circle, EmptyElement, IntoDrawingArea, Text},
    series::{AreaSeries, LineSeries, PointSeries},
    style::{full_palette::PURPLE, Color, BLUE, GREEN, RED, WHITE, YELLOW},
};
use time::OffsetDateTime;

use crate::{types::Data};

pub fn chart(data: &Data) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("charts/chart.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let y_min = *data
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap() as f32;
    let y_max = *data
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap() as f32;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Chart", ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..data.len() as u32, y_min..y_max)?;

    chart.configure_mesh().light_line_style(WHITE).draw()?;

    chart.draw_series(data.iter().enumerate().map(|(index, x)| {
        CandleStick::new(index as u32, 0., 0., 0., *x as f32, GREEN.filled(), RED, 15)
    }))?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}

pub fn candle_chart(
    dir: &String,
    bars: &[historical::Bar],
) -> Result<(), Box<dyn std::error::Error>> {
    let dimensions = (1024, 768);
    let path = format!("{dir}/candle.png");
    let root = BitMapBackend::new(path.as_str(), dimensions).into_drawing_area();
    root.fill(&WHITE)?;

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
        .caption("Candle Chart", ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..bars.len() as u32, y_min..y_max)?;

    chart.configure_mesh().light_line_style(WHITE).draw()?;

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
            GREEN.filled(),
            RED.filled(),
            dimensions.0 / bars.len() as u32,
        )
    }))?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}

pub fn simple_chart(dir: &String, name: &str, data: &Data) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/{name}.png");
    let root = BitMapBackend::new(path.as_str(), (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let y_min = data
    .iter()
    .min_by(|a, b| a.partial_cmp(b).unwrap())
    .unwrap().min(0f64) as f32
    * 0.9;

let y_max = data
    .iter()
    .max_by(|a, b| a.partial_cmp(b).unwrap())
    .unwrap().max(100f64) as f32
    * 1.1;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption(name, ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..data.len() as u32, y_min..y_max)?;

    chart.configure_mesh().light_line_style(WHITE).draw()?;

    chart.draw_series(
        AreaSeries::new(
            data.iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value as f32)),
            0.0,
            BLUE.mix(0.2),
        )
        .border_style(BLUE),
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
    let path = format!("{dir}/buy_sell.png");
    let root = BitMapBackend::new(path.as_str(), (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut filled_buy = Vec::new();
    let mut filled_sell = Vec::new();

    for i in 0..data.len() {
        if let Some(buy) = buy_indexes.get(&i) {
            filled_buy.push(buy.0);
            filled_sell.push(0.0);
            continue;
        }

        if let Some(sell) = sell_indexes.get(&i) {
            filled_buy.push(0.0);
            filled_sell.push(sell.0);
            continue;
        }
    }

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
        .caption("Buy Sell Chart", ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..data.len() as u32, y_min..y_max)?;

    chart.configure_mesh().light_line_style(WHITE).draw()?;

    // Data
    chart.draw_series(
        AreaSeries::new(
            data.iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value)),
            0.0,
            BLUE.mix(0.2),
        )
        .border_style(BLUE),
    )?;

    let point_size = 4;
    
    // Sells
    chart.draw_series(PointSeries::of_element(
        sell_indexes
            .iter()
            .map(|(index, value)| (*index as u32, value.0)),
            point_size,
        YELLOW.mix(0.9).filled(),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    // Buys
    chart.draw_series(PointSeries::of_element(
        buy_indexes
            .iter()
            .map(|(index, value)| (*index as u32, value.0)),
            point_size,
        RED.mix(0.9).filled(),
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
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/assets.png");
    let root = BitMapBackend::new(path.as_str(), (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let y_max = *assets
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap() as f32
        * 1.1;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Assets: Total; Positioned; Cash", ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..assets.len() as u32, 0.0..y_max)?;

    chart.configure_mesh().light_line_style(WHITE).draw()?;

    chart.draw_series(
        AreaSeries::new(
            assets
                .iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value as f32)),
            0.0,
            BLUE.mix(0.2),
        )
        .border_style(BLUE),
    )?;

    let positioned = {
        if let Some(positioned) = positioned {
            positioned
        } else {
            &assets.iter().zip(cash).map(|(a, b)| a - b).collect::<Vec<f64>>()
        }
    };

    chart.draw_series(
        AreaSeries::new(
            positioned.iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value as f32)),
            0.0,
            RED.mix(0.2),
        )
        .border_style(RED),
    )?;

    chart.draw_series(
        AreaSeries::new(
            cash.iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value as f32)),
            0.0,
            GREEN.mix(0.2),
        )
        .border_style(GREEN),
    )?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}

pub fn want_chart(
    dir: &String,
    data: &Data,
    want_indexes: &HashMap<usize, f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{dir}/want.png");
    let root = BitMapBackend::new(path.as_str(), (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut wants = Vec::new();

    for i in 0..data.len() {
        if let Some(want) = want_indexes.get(&i) {
            wants.push(*want);
            continue;
        }

        wants.push(0.0);
    }

    let y_min = wants
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        * 0.9;
    let y_max = wants
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        * 1.1;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Buy Sell Chart", ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..data.len() as u32, y_min..y_max)?;

    chart.configure_mesh().light_line_style(WHITE).draw()?;

    // Wants

    chart.draw_series(
        AreaSeries::new(
            wants.iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value)),
            0.0,
            PURPLE.mix(0.2),
        )
        .border_style(PURPLE),
    )?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}