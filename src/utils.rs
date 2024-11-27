use chrono::{DateTime, Local, NaiveDateTime, TimeZone};
use ibapi::market_data::{historical, realtime};
use plotters::{
    coord::types::RangedCoordf32,
    prelude::{BitMapBackend, CandleStick, IntoDrawingArea},
    series::{AreaSeries, LineSeries},
    style::{Color, BLUE, GREEN, RED, WHITE},
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
    println!("EMA: {averages:?}");
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
    let root = BitMapBackend::new("charts/rsi_chart.png", (1024, 768)).into_drawing_area();
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
