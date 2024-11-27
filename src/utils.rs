use ibapi::market_data::historical;
use plotters::{
    coord::types::RangedCoordf32,
    prelude::{BitMapBackend, CandleStick, IntoDrawingArea},
    style::{Color, GREEN, RED, WHITE},
};

use crate::{constants::rsi::MOVING_AVG_DAYS, types::Data};

pub fn convert_historical(data: &Vec<historical::Bar>) -> Data {
    data.iter().map(|bar| bar.close).collect()
}

/// Get the relative strength index value for each data point
pub fn get_rsi_values(data: &Data) -> Data {
    let diffs = get_differences(&data);

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
        .map(|(up, down)| 100. * up / (up + down))
        .collect();
    println!("RSI values {rsi_values:?}");
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
                0.
            } else {
                averages[previous_index]
            }
        };
        let avg = *value * alpha + previous * (1. - alpha);
        averages.push(avg);
    }
    println!("EWMs {averages:?}");
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
        diff.push(previous - value)
    }

    diff
}

pub fn chart(data: &Data) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("charts/chart.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let y_min = *data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() as f32;
    let y_max = *data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() as f32;

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

    chart.draw_series(data.iter().enumerate().map(|(index, x)| {
        CandleStick::new(index as u32, *x as f32 - 10., 0., 0., *x as f32, GREEN.filled(), RED, 15)
    }))?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}