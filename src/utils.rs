use std::{collections::HashMap, fs};

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

use crate::types::Data;

pub fn convert_historical(data: &Vec<historical::Bar>) -> Data {
    data.iter().map(|bar| bar.close).collect()
}

/// Get the relative strength index value for each data point
pub fn get_rsi_values(data: &[f64], ema_alpha: f64) -> Data {
    let mut diffs = get_differences(data);
    diffs[0] = 1.;

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

    let upward_avg = ema(&upwards, ema_alpha);
    let downward_avg = ema(&downwards, ema_alpha);

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

/// Get the relative strength index value for each data point
pub fn get_rsi_percents(data: &[f64], ema_alpha: f64) -> Data {
    let mut diffs = get_differences(data);
    diffs[0] = 1.;

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

    let upward_avg = ema(&upwards, ema_alpha);
    let downward_avg = ema(&downwards, ema_alpha);

    let rsi_values = upward_avg
        .iter()
        .zip(downward_avg.iter())
        .map(|(up, down)| {
            let rs = up / down;
            1. - (1. / (1. + rs))
            /* (100. - (100. / (1. + rs))) / 100. */
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
pub fn ema(data: &[f64], alpha: f64) -> Data {
    let mut averages = Vec::new();
    let mut previous = data[0];

    for (index, value) in data.iter().enumerate() {
        let avg = *value * alpha + previous * (1. - alpha);
        averages.push(avg);

        previous = avg;
    }
    averages
}

pub fn ema_diff_percent(data: &[f64], alpha: f64) -> Data {
    let mut averages = Vec::new();
    let mut previous = data[0];

    for (index, value) in data.iter().enumerate() {
        let avg = *value * alpha + previous * (1. - alpha);
        let diff_percent = (value - avg) / avg;
        averages.push(diff_percent);

        previous = avg;
    }
    averages
}

pub fn get_macd(data: &[f64]) -> Data {
    let ema_12 = ema(data, 1. / 12.);
    let ema_26 = ema(data, 1. / 26.);

    ema_12.iter().zip(ema_26.iter()).map(|(a, b)| a - b).collect()
}

pub fn get_stochastic_oscillator(bars: &[historical::Bar]) -> Data {
    let mut values = Vec::new();

    for (index, bar) in bars.iter().enumerate() {
        let high = bars
            .iter()
            .take(index + 1)
            .map(|b| b.high)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let low = bars
            .iter()
            .take(index + 1)
            .map(|b| b.low)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        values.push((high - low) / bar.close);
    }

    values
}

pub fn get_w_percent_range(bars: &[historical::Bar]) -> Data {
    let mut values = Vec::new();

    // William's percent range 0-1

    for (index, bar) in bars.iter().enumerate() {
        let high = bars
            .iter()
            .take(index + 1)
            .map(|b| b.high)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let low = bars
            .iter()
            .take(index + 1)
            .map(|b| b.low)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        values.push((high - low) / bar.close);
    }

    values
}

pub fn get_differences(data: &[f64]) -> Data {
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

pub fn get_diff_percents(data: &[f64]) -> Data {
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
        diff.push((value - previous) / previous)
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

pub fn create_folder_if_not_exists(dir: &String) {
    let _ = fs::create_dir_all(dir);
}

pub fn find_highest<T: std::cmp::PartialOrd>(iter: &[T]) -> (usize, &T) {
    let mut best_index = 0;
    let mut highest_value = &iter[0];

    for i in 1..iter.len() {
        let value = &iter[1];

        if value <= highest_value {
            continue;
        }
        best_index = i;
        highest_value = value;
    }

    (best_index, highest_value)
}
