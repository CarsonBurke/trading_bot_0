use chrono::{DateTime, Local, TimeZone};
use ibapi::market_data::historical;
use plotters::prelude::{BitMapBackend, CandleStick, IntoDrawingArea};
use plotters::style::Color;
use shared::theme::plotters_colors as theme;
use time::OffsetDateTime;

use crate::constants::CHART_IMAGE_FORMAT;
use crate::types::Data;

pub fn chart(data: &Data) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("charts/chart.{}", CHART_IMAGE_FORMAT);
    let root = BitMapBackend::new(path.as_str(), (2560, 780)).into_drawing_area();
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
        .caption("Price", ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..data.len() as u32, y_min..y_max)?;

    chart
        .configure_mesh()
        .label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .light_line_style(&theme::SURFACE0)
        .draw()?;

    chart.draw_series(data.iter().enumerate().map(|(index, x)| {
        CandleStick::new(
            index as u32,
            0.,
            0.,
            0.,
            *x as f32,
            theme::GREEN.filled(),
            &theme::RED,
            15,
        )
    }))?;

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}

pub fn candle_chart(
    dir: &String,
    bars: &[historical::Bar],
) -> Result<(), Box<dyn std::error::Error>> {
    let dimensions = (2560, 780);
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

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Candles", ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..bars.len() as u32, y_min..y_max)?;

    chart
        .configure_mesh()
        .label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .light_line_style(&theme::SURFACE0)
        .draw()?;

    chart.draw_series(bars.iter().enumerate().map(|(index, bar)| {
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
