use std::collections::HashMap;

use chrono::{DateTime, Local, NaiveDateTime, TimeZone};
use ibapi::market_data::{historical, realtime};
use plotters::{
    coord::types::RangedCoordf32,
    data,
    prelude::{BitMapBackend, CandleStick, Circle, EmptyElement, IntoDrawingArea, Text},
    series::{AreaSeries, LineSeries, PointSeries},
    style::{Color, BLUE, GREEN, RED, WHITE, YELLOW},
};
use time::OffsetDateTime;

use crate::{constants::rsi::MOVING_AVG_DAYS, types::Data};

pub fn convert_historical(data: &Vec<historical::Bar>) -> Data {
    data.iter().map(|bar| bar.close).collect()
}

/// Get the relative strength index value for each data point
pub fn get_rsi_values(data: &Data) -> Data {
    let diffs = get_differences(data);

    let mut upwards = Vec::new();
    let mut downwards = Vec::new();

    for diff in diffs.iter() {
        if *diff >= 0. {
            upwards.push(*diff);
            downwards.push(0.);
            continue;
        }

        downwards.push(diff.abs());
        upwards.push(0.);
    }

    let alpha = 1. / (MOVING_AVG_DAYS as f64 + 1.);

    let upward_avg = ema(&upwards, alpha);
    let downward_avg = ema(&downwards, alpha);

    let rsi_values = upward_avg
        .iter()
        .zip(downward_avg.iter())
        .map(|(up, down)| {
            let rs = up / down;
            100. - (100. / (1. + rs))
        })
        .collect();
    rsi_values
}

/// Calcualtes the exponential moving average
///
/// # Arguments
///
/// * `alpha` - The exponential weight to apply to the previous average. For example, 0.18 adds 18% of the previous average
///
pub fn ema(data: &Data, alpha: f64) -> Data {
    let mut averages = Vec::new();

    for (index, value) in data.iter().enumerate() {
        // let mut sum = 0.;
        // let steps_back = (index as u32 - MOVING_AVG_DAYS).min(0);

        // for i in steps_back..(index as u32) {
        //     sum += *value * alpha + averages.get()/* (alpha.powf((i - steps_back) as f64)) */;
        // }

        // averages.push(sum / steps_back as f64);

        let previous = {
            let (previous_index, overflowed) = index.overflowing_sub(1);
            if overflowed {
                averages.push(*value * alpha);
                continue;
            } else {
                averages[previous_index]
            }
        };
        let avg = *value * alpha + previous * (1. - alpha);
        averages.push(avg);
    }
    averages
}

pub fn get_differences(data: &Data) -> Data {
    let mut diff = vec![];

    for (index, value) in data.iter().enumerate() {
        let previous = {
            let (previous_index, overflowed) = index.overflowing_sub(1);

            if overflowed {
                100.
            } else {
                data[previous_index]
            }
        };
        diff.push(value - previous)
    }

    diff
}

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

pub fn candle_chart(bars: &[historical::Bar]) -> Result<(), Box<dyn std::error::Error>> {
    let dimensions = (1024, 768);
    let root = BitMapBackend::new("charts/candle.png", dimensions).into_drawing_area();
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

pub fn rsi_chart(data: &Data) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("charts/rsi.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("RSI", ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..data.len() as u32, 0f32..100f32)?;

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
    data: &Data,
    buy_indexes: &HashMap<usize, (f64, u32)>,
    sell_indexes: &HashMap<usize, (f64, u32)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("charts/buy_sell.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut filled_buy = Vec::new();
    let mut filled_sell = Vec::new();

    for i in 0..data.len() {
        if let Some(buy) = buy_indexes.get(&i) {
            filled_buy.push(buy.0 / buy.1 as f64);
            filled_sell.push(0.0);
            continue;
        }

        if let Some(sell) = sell_indexes.get(&i) {
            filled_buy.push(0.0 / sell.1 as f64);
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
    // Sell
    /* chart.draw_series(
        AreaSeries::new(
            filled_sell
                .iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value)),
            0.0,
            YELLOW.mix(0.0),
        )
        .border_style(YELLOW),
    )?; */

    // Sell
    /* chart.draw_series(
        PointSeries::<_, _, Circle<>>::new(
            filled_sell
                .iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value)),
            5,
            YELLOW.mix(0.5),
        ),
    )?; */
    let sells = sell_indexes
        .iter()
        .map(|(index, value)| (*index as u32, value.0 / value.1 as f64));
    println!("{:?}", sells.collect::<Vec<_>>());

    chart.draw_series(PointSeries::of_element(
        sell_indexes
            .iter()
            .map(|(index, value)| (*index as u32, value.0 / value.1 as f64)),
        5,
        YELLOW.filled(),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;
    // buys

    chart.draw_series(PointSeries::of_element(
        buy_indexes
            .iter()
            .map(|(index, value)| (*index as u32, value.0 / value.1 as f64)),
        5,
        BLUE.filled(),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}

pub fn assets_chart(
    assets: &Data,
    positioned_assets: &Data,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("charts/assets.png", (1024, 768)).into_drawing_area();
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
            assets.iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value as f32)),
            0.0,
            BLUE.mix(0.2),
        )
        .border_style(BLUE),
    )?;

    chart.draw_series(
        AreaSeries::new(
            positioned_assets.iter()
                .enumerate()
                .map(|(index, value)| (index as u32, *value as f32)),
            0.0,
            RED.mix(0.2),
        )
        .border_style(RED),
    )?;

    let cash = assets.iter().zip(positioned_assets).map(|(a, b)| a - b);

    chart.draw_series(
        AreaSeries::new(
            cash
                .enumerate()
                .map(|(index, value)| (index as u32, value as f32)),
            0.0,
            GREEN.mix(0.2),
        )
        .border_style(GREEN),
    )?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}

/// Returns how many stocks can be bought for a given a total price and a max amount
pub fn round_to_stock(price: f64, max: f64) -> (f64, u32) {
    let quantity = (max / price).floor() as u32;

    (price * quantity as f64, quantity)
}

pub fn estimate_stock_value(financials: String) {
    // based on growth, with a weighted preference (ema) for more recent data
    // quarterly for the last 3 years?
    // total revenue is all that matters? Or also care about cost of revenue and marginal?
}
