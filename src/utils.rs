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
pub fn get_rsi_values(data: &Data, ema_alpha: f64) -> Data {
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
