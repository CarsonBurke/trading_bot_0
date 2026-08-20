use anyhow::{anyhow, Result};
use image::{DynamicImage, RgbImage};
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};
use shared::report::{
    CandleBar, QuantileBand, Report, ReportKind, ReportSeries, ScaleKind, TradePoint,
};
use shared::theme::plotters_colors as theme;

const CHART_DIMS: (u32, u32) = (2560, 780);

/// The single rendering path: both the interactive chart viewer and the
/// `render` CLI subcommand come through here.
pub fn render_report_with_options(
    report: &Report,
    skip: usize,
    show_legend: bool,
    solo_series: Option<usize>,
) -> Result<DynamicImage> {
    let mut buffer = vec![0u8; (CHART_DIMS.0 * CHART_DIMS.1 * 3) as usize];
    {
        let root = BitMapBackend::with_buffer(&mut buffer, CHART_DIMS).into_drawing_area();
        root.fill(&theme::BASE)?;

        let x_offset = skip as u32;

        match &report.kind {
            ReportKind::Simple { values, ema_alpha } => {
                let values = skip_slice(values, skip);
                render_simple(
                    &root,
                    report,
                    values,
                    *ema_alpha,
                    x_offset,
                    show_legend,
                    solo_series,
                )?;
            }
            ReportKind::MultiLine { series } => {
                let series: Vec<ReportSeries> = series
                    .iter()
                    .map(|s| ReportSeries {
                        label: s.label.clone(),
                        values: skip_slice(&s.values, skip).to_vec(),
                    })
                    .collect();
                render_multi_line(&root, report, &series, x_offset, show_legend, solo_series)?;
            }
            ReportKind::Assets {
                total,
                cash,
                positioned,
                benchmark,
            } => {
                let total = skip_slice(total, skip);
                let cash = skip_slice(cash, skip);
                let positioned: Option<Vec<f32>> =
                    positioned.as_ref().map(|p| skip_slice(p, skip).to_vec());
                let benchmark: Option<Vec<f32>> =
                    benchmark.as_ref().map(|b| skip_slice(b, skip).to_vec());
                render_assets(
                    &root,
                    report,
                    total,
                    cash,
                    positioned.as_ref(),
                    benchmark.as_ref(),
                    x_offset,
                    show_legend,
                    solo_series,
                )?;
            }
            ReportKind::BuySell {
                prices,
                buys,
                sells,
            } => {
                let prices = skip_slice(prices, skip);
                // Filter buys/sells to only those within skipped range, adjust indices
                let buys: Vec<TradePoint> = buys
                    .iter()
                    .filter(|p| (p.index as usize) >= skip)
                    .map(|p| TradePoint {
                        index: p.index - skip as u32,
                    })
                    .collect();
                let sells: Vec<TradePoint> = sells
                    .iter()
                    .filter(|p| (p.index as usize) >= skip)
                    .map(|p| TradePoint {
                        index: p.index - skip as u32,
                    })
                    .collect();
                render_buy_sell(&root, report, prices, &buys, &sells, x_offset)?;
            }
            ReportKind::CandleFan {
                actual,
                bands,
                samples,
            } => {
                let actual = skip_slice(actual, skip);
                let bands: Vec<QuantileBand> = bands
                    .iter()
                    .map(|band| QuantileBand {
                        probability: band.probability,
                        closes: skip_slice(&band.closes, skip).to_vec(),
                    })
                    .collect();
                let samples: Vec<ReportSeries> = samples
                    .iter()
                    .map(|series| ReportSeries {
                        label: series.label.clone(),
                        values: skip_slice(&series.values, skip).to_vec(),
                    })
                    .collect();
                render_candle_fan(
                    &root,
                    report,
                    actual,
                    &bands,
                    &samples,
                    x_offset,
                    show_legend,
                    solo_series,
                )?;
            }
            ReportKind::Observations { .. } => {
                return Err(anyhow!("report type not renderable"));
            }
        }

        root.present()?;
    }

    let image = RgbImage::from_raw(CHART_DIMS.0, CHART_DIMS.1, buffer)
        .ok_or_else(|| anyhow!("failed to build image"))?;
    Ok(DynamicImage::ImageRgb8(image))
}

fn skip_slice<T>(slice: &[T], skip: usize) -> &[T] {
    if skip >= slice.len() {
        &[]
    } else {
        &slice[skip..]
    }
}

fn render_simple(
    root: &DrawingArea<BitMapBackend, Shift>,
    report: &Report,
    values: &[f32],
    ema_alpha: Option<f64>,
    x_offset: u32,
    show_legend: bool,
    solo_series: Option<usize>,
) -> Result<()> {
    if values.is_empty() {
        return Ok(());
    }
    if !values.iter().any(|value| value.is_finite()) {
        let message = format!("{} — no finite values", normalize_title(&report.title));
        root.draw(&Text::new(
            message,
            (CHART_DIMS.0 as i32 / 2, CHART_DIMS.1 as i32 / 2),
            ("sans-serif", 24)
                .into_font()
                .color(&theme::SUBTEXT0)
                .pos(Pos::new(HPos::Center, VPos::Center)),
        ))?;
        return Ok(());
    }

    // Series: 0=value, 1=EMA. Out-of-range solo → no solo.
    let series_count = if ema_alpha.is_some() { 2 } else { 1 };
    let solo = match solo_series {
        Some(idx) if idx < series_count => Some(idx),
        _ => None,
    };
    let value_active = solo.is_none() || solo == Some(0);
    let ema_active = ema_alpha.is_some() && (solo.is_none() || solo == Some(1));

    let scale = report.scale;
    // Y-range from active series only
    let range_values: Vec<f32> = if value_active && ema_active {
        let ema = compute_ema(values, ema_alpha.unwrap());
        values.iter().chain(ema.iter()).copied().collect()
    } else if ema_active {
        compute_ema(values, ema_alpha.unwrap())
    } else {
        values.to_vec()
    };
    let (y_min, y_max) = range_for(&range_values, scale == ScaleKind::Symlog)?;
    let title = normalize_title(&report.title);
    let x_end = x_offset + values.len() as u32;
    let mut chart = plotters::chart::ChartBuilder::on(root)
        .caption(title.as_str(), ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(x_offset..x_end, y_min..y_max)?;

    let mut mesh = chart.configure_mesh();
    mesh.label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .light_line_style(&theme::SURFACE0);
    if let Some(label) = report.x_label.as_deref() {
        mesh.x_desc(label);
    }
    if let Some(label) = report.y_label.as_deref() {
        mesh.y_desc(label);
    }
    if scale == ScaleKind::Symlog {
        mesh.y_label_formatter(&|v| format!("{:.2e}", symlog_inv(*v)));
    }
    mesh.draw()?;

    // Value series
    if value_active {
        let mapped = values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(idx, v)| (x_offset + idx as u32, map_value(*v as f64, scale)));

        if scale == ScaleKind::Symlog {
            chart
                .draw_series(LineSeries::new(
                    mapped,
                    ShapeStyle::from(&theme::BLUE).stroke_width(1),
                ))?
                .label("value")
                .legend(legend_rect(&theme::BLUE));
        } else {
            chart
                .draw_series(
                    AreaSeries::new(mapped, 0.0, theme::BLUE.mix(0.2))
                        .border_style(ShapeStyle::from(&theme::BLUE).stroke_width(1)),
                )?
                .label("value")
                .legend(legend_rect(&theme::BLUE));
        }
    } else {
        chart
            .draw_series(LineSeries::new(
                std::iter::empty::<(u32, f64)>(),
                ShapeStyle::from(&theme::SURFACE2).stroke_width(1),
            ))?
            .label("value")
            .legend(legend_rect(&theme::BLUE));
    }

    // EMA series
    if let Some(alpha) = ema_alpha {
        if ema_active {
            let ema = compute_ema(values, alpha);
            let ema_series = ema
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .map(|(idx, v)| (x_offset + idx as u32, map_value(*v as f64, scale)));
            chart
                .draw_series(LineSeries::new(
                    ema_series,
                    ShapeStyle::from(&theme::YELLOW).stroke_width(1),
                ))?
                .label("EMA")
                .legend(legend_rect(&theme::YELLOW));
        } else {
            chart
                .draw_series(LineSeries::new(
                    std::iter::empty::<(u32, f64)>(),
                    ShapeStyle::from(&theme::SURFACE2).stroke_width(1),
                ))?
                .label("EMA")
                .legend(legend_rect(&theme::YELLOW));
        }
    }

    if show_legend {
        chart
            .configure_series_labels()
            .position(LegendConfig::position())
            .background_style(LegendConfig::background())
            .border_style(LegendConfig::border())
            .label_font(LegendConfig::font())
            .draw()?;
    }

    Ok(())
}

fn render_multi_line(
    root: &DrawingArea<BitMapBackend, Shift>,
    report: &Report,
    series: &[ReportSeries],
    x_offset: u32,
    show_legend: bool,
    solo_series: Option<usize>,
) -> Result<()> {
    let solo = match solo_series {
        Some(idx) if idx < series.len() => Some(idx),
        _ => None,
    };

    // Y-range from solo series only (or all if no solo)
    let range_series: Vec<&ReportSeries> = match solo {
        Some(idx) => vec![&series[idx]],
        None => series.iter().collect(),
    };
    let all_values: Vec<f32> = range_series
        .iter()
        .flat_map(|s| s.values.iter())
        .copied()
        .collect();
    if all_values.is_empty() {
        return Ok(());
    }

    let scale = report.scale;
    let (y_min, y_max) = range_for(&all_values, scale == ScaleKind::Symlog)?;
    let x_len = series.iter().map(|s| s.values.len()).max().unwrap_or(1) as u32;
    let x_end = x_offset + x_len;

    let title = normalize_title(&report.title);
    let mut chart = plotters::chart::ChartBuilder::on(root)
        .caption(title.as_str(), ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(x_offset..x_end, y_min..y_max)?;

    let mut mesh = chart.configure_mesh();
    mesh.label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .light_line_style(&theme::SURFACE0);
    if let Some(label) = report.x_label.as_deref() {
        mesh.x_desc(label);
    }
    if let Some(label) = report.y_label.as_deref() {
        mesh.y_desc(label);
    }
    if scale == ScaleKind::Symlog {
        mesh.y_label_formatter(&|v| format!("{:.2e}", symlog_inv(*v)));
    }
    mesh.draw()?;

    let colors = [
        &theme::BLUE,
        &theme::GREEN,
        &theme::RED,
        &theme::YELLOW,
        &theme::MAUVE,
    ];

    for (i, s) in series.iter().enumerate() {
        let active = solo.is_none() || solo == Some(i);
        let color = colors[i % colors.len()];
        if active {
            let mapped: Vec<_> = s
                .values
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .map(|(idx, v)| (x_offset + idx as u32, map_value(*v as f64, scale)))
                .collect();
            chart
                .draw_series(LineSeries::new(
                    mapped.iter().copied(),
                    ShapeStyle::from(&color.mix(0.8)).stroke_width(1),
                ))?
                .label(s.label.as_str())
                .legend(legend_rect(color));
            if mapped.len() == 1 {
                chart.draw_series(std::iter::once(Circle::new(
                    mapped[0],
                    3,
                    color.mix(0.8).filled(),
                )))?;
            }
        } else {
            // Empty series to reserve legend entry, keep original color
            chart
                .draw_series(LineSeries::new(
                    std::iter::empty::<(u32, f64)>(),
                    ShapeStyle::from(&theme::SURFACE2).stroke_width(1),
                ))?
                .label(s.label.as_str())
                .legend(legend_rect(color));
        }
    }

    if show_legend {
        chart
            .configure_series_labels()
            .position(LegendConfig::position())
            .background_style(LegendConfig::background())
            .border_style(LegendConfig::border())
            .label_font(LegendConfig::font())
            .draw()?;
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AssetSeriesKind {
    Total,
    Positioned,
    Cash,
    Benchmark,
}

fn asset_series_kinds(
    positioned: Option<&[f32]>,
    benchmark: Option<&[f32]>,
) -> Vec<AssetSeriesKind> {
    let mut kinds = vec![AssetSeriesKind::Total];
    if positioned.is_some_and(|values| !values.is_empty()) {
        kinds.push(AssetSeriesKind::Positioned);
    }
    kinds.push(AssetSeriesKind::Cash);
    if benchmark.is_some_and(|values| !values.is_empty()) {
        kinds.push(AssetSeriesKind::Benchmark);
    }
    kinds
}

pub(crate) fn asset_series_count(positioned: Option<&[f32]>, benchmark: Option<&[f32]>) -> usize {
    asset_series_kinds(positioned, benchmark).len()
}

fn asset_series_values<'a>(
    kind: AssetSeriesKind,
    total: &'a [f32],
    cash: &'a [f32],
    positioned: Option<&'a [f32]>,
    benchmark: Option<&'a [f32]>,
) -> &'a [f32] {
    match kind {
        AssetSeriesKind::Total => total,
        AssetSeriesKind::Positioned => positioned.unwrap_or_default(),
        AssetSeriesKind::Cash => cash,
        AssetSeriesKind::Benchmark => benchmark.unwrap_or_default(),
    }
}

fn asset_series_label(kind: AssetSeriesKind) -> &'static str {
    match kind {
        AssetSeriesKind::Total => "total",
        AssetSeriesKind::Positioned => "positioned",
        AssetSeriesKind::Cash => "cash",
        AssetSeriesKind::Benchmark => "benchmark",
    }
}

fn asset_series_color(kind: AssetSeriesKind) -> &'static RGBColor {
    match kind {
        AssetSeriesKind::Total => &theme::BLUE,
        AssetSeriesKind::Positioned => &theme::RED,
        AssetSeriesKind::Cash => &theme::GREEN,
        AssetSeriesKind::Benchmark => &theme::MAUVE,
    }
}

fn render_assets(
    root: &DrawingArea<BitMapBackend, Shift>,
    report: &Report,
    total: &[f32],
    cash: &[f32],
    positioned: Option<&Vec<f32>>,
    benchmark: Option<&Vec<f32>>,
    x_offset: u32,
    show_legend: bool,
    solo_series: Option<usize>,
) -> Result<()> {
    if total.is_empty() {
        return Ok(());
    }

    let positioned = positioned.map(Vec::as_slice);
    let benchmark = benchmark.map(Vec::as_slice);
    let kinds = asset_series_kinds(positioned, benchmark);
    let selected = solo_series.and_then(|index| kinds.get(index).copied());
    let is_active = |kind| selected.is_none() || selected == Some(kind);

    let max_val = kinds
        .iter()
        .copied()
        .filter(|kind| is_active(*kind))
        .flat_map(|kind| asset_series_values(kind, total, cash, positioned, benchmark))
        .copied()
        .filter(|value| value.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let max_val = if max_val.is_finite() { max_val } else { 1.0 };

    let y_max = if max_val > 0.0 { max_val * 1.1 } else { 1.0 };
    let x_end = x_offset + total.len() as u32;

    let title = normalize_title(&report.title);
    let mut chart = plotters::chart::ChartBuilder::on(root)
        .caption(title.as_str(), ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(x_offset..x_end, 0.0..y_max)?;

    chart
        .configure_mesh()
        .label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .light_line_style(&theme::SURFACE0)
        .draw()?;

    for kind in kinds {
        let values = asset_series_values(kind, total, cash, positioned, benchmark);
        let color = asset_series_color(kind);
        let annotation = if is_active(kind) {
            if kind == AssetSeriesKind::Benchmark {
                chart.draw_series(LineSeries::new(
                    values
                        .iter()
                        .enumerate()
                        .filter(|(_, value)| value.is_finite())
                        .map(|(index, value)| (x_offset + index as u32, *value)),
                    ShapeStyle::from(color).stroke_width(1),
                ))?
            } else {
                chart.draw_series(
                    AreaSeries::new(
                        values
                            .iter()
                            .enumerate()
                            .filter(|(_, value)| value.is_finite())
                            .map(|(index, value)| (x_offset + index as u32, *value)),
                        0.0,
                        color.mix(0.2),
                    )
                    .border_style(ShapeStyle::from(color).stroke_width(1)),
                )?
            }
        } else {
            chart.draw_series(LineSeries::new(
                std::iter::empty::<(u32, f32)>(),
                ShapeStyle::from(&theme::SURFACE2).stroke_width(1),
            ))?
        };
        annotation
            .label(asset_series_label(kind))
            .legend(legend_rect(color));
    }

    if show_legend {
        chart
            .configure_series_labels()
            .position(LegendConfig::position())
            .background_style(LegendConfig::background())
            .border_style(LegendConfig::border())
            .label_font(LegendConfig::font())
            .draw()?;
    }

    Ok(())
}

fn render_buy_sell(
    root: &DrawingArea<BitMapBackend, Shift>,
    report: &Report,
    prices: &[f32],
    buys: &[TradePoint],
    sells: &[TradePoint],
    x_offset: u32,
) -> Result<()> {
    if prices.is_empty() {
        return Ok(());
    }

    let y_min = prices
        .iter()
        .map(|v| *v as f64)
        .fold(f64::INFINITY, f64::min);
    let y_max = prices
        .iter()
        .map(|v| *v as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let y_range = (y_max - y_min).max(0.01);
    let y_min = y_min - y_range * 0.05;
    let y_max = y_max + y_range * 0.05;
    let x_end = x_offset + prices.len() as u32;

    let title = normalize_title(&report.title);
    let mut chart = plotters::chart::ChartBuilder::on(root)
        .caption(title.as_str(), ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(x_offset..x_end, y_min..y_max)?;

    chart
        .configure_mesh()
        .label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .light_line_style(&theme::SURFACE0)
        .draw()?;

    chart.draw_series(
        AreaSeries::new(
            prices
                .iter()
                .enumerate()
                .map(|(index, value)| (x_offset + index as u32, *value as f64)),
            0.0f64,
            theme::BLUE.mix(0.2),
        )
        .border_style(ShapeStyle::from(&theme::BLUE).stroke_width(1)),
    )?;

    // Filter and offset buy/sell points
    let point_size = 3;
    chart.draw_series(PointSeries::of_element(
        sells
            .iter()
            .filter(|p| (p.index as usize) < prices.len())
            .map(|p| {
                (
                    x_offset + p.index,
                    prices.get(p.index as usize).copied().unwrap_or(0.0) as f64,
                )
            }),
        point_size,
        theme::YELLOW.mix(0.9).filled(),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    chart.draw_series(PointSeries::of_element(
        buys.iter()
            .filter(|p| (p.index as usize) < prices.len())
            .map(|p| {
                (
                    x_offset + p.index,
                    prices.get(p.index as usize).copied().unwrap_or(0.0) as f64,
                )
            }),
        point_size,
        theme::RED.mix(0.9).filled(),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    Ok(())
}

/// A realized path against the quantile fan of the sampled continuations, with a
/// few genuine draws overlaid.
///
/// The realized bars are candles because every field of them happened. The
/// predictive law is a fan and a handful of draws, never a single line: the fan
/// centre is a locus of per-horizon medians and no draw follows it, so drawing it
/// as "the forecast" is what makes a reader score a pointwise error against a
/// distribution over paths.
///
/// Series indices for solo: `0` is the realized path, `1..=bands.len()` the
/// quantile loci in the order given, and the remainder the sampled draws.
fn render_candle_fan(
    root: &DrawingArea<BitMapBackend, Shift>,
    report: &Report,
    actual: &[CandleBar],
    bands: &[QuantileBand],
    samples: &[ReportSeries],
    x_offset: u32,
    show_legend: bool,
    solo_series: Option<usize>,
) -> Result<()> {
    if actual.is_empty() && bands.is_empty() && samples.is_empty() {
        return Ok(());
    }
    let series_count = 1 + bands.len() + samples.len();
    let solo = match solo_series {
        Some(idx) if idx < series_count => Some(idx),
        _ => None,
    };
    let active = |index: usize| solo.is_none() || solo == Some(index);
    let actual_active = active(0);

    let mut values = Vec::with_capacity(actual.len() * 4 + series_count * actual.len());
    if actual_active {
        for candle in actual {
            values.extend([candle.open, candle.high, candle.low, candle.close]);
        }
    }
    for (index, band) in bands.iter().enumerate() {
        if active(1 + index) {
            values.extend(band.closes.iter().copied());
        }
    }
    for (index, series) in samples.iter().enumerate() {
        if active(1 + bands.len() + index) {
            values.extend(series.values.iter().copied());
        }
    }
    let (y_min, y_max) = range_for(&values, false)?;
    let x_len = actual
        .len()
        .max(bands.iter().map(|b| b.closes.len()).max().unwrap_or(0))
        .max(samples.iter().map(|s| s.values.len()).max().unwrap_or(0))
        .max(1) as f64;
    let x_start = x_offset as f64;
    let x_end = x_start + x_len;

    let title = normalize_title(&report.title);
    let mut chart = plotters::chart::ChartBuilder::on(root)
        .caption(title.as_str(), ("sans-serif", 20, &theme::TEXT))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(x_start..x_end, y_min..y_max)?;

    let mut mesh = chart.configure_mesh();
    mesh.label_style(("sans-serif", 15, &theme::TEXT))
        .axis_style(&theme::SURFACE1)
        .x_labels(8)
        .y_labels(6)
        .bold_line_style(&theme::OVERLAY0)
        .light_line_style(&TRANSPARENT);
    if let Some(label) = report.x_label.as_deref() {
        mesh.x_desc(label);
    }
    if let Some(label) = report.y_label.as_deref() {
        mesh.y_desc(label);
    }
    mesh.draw()?;

    // Sampled draws first, thin and dim, so they read as texture behind the fan rather
    // than competing with it. Measured against a rendered 100-bar fan: at `mix(0.45)`
    // and the fan's original 1px extremes, MAUVE on this background out-contrasts
    // SAPPHIRE and TEAL and the eye lands on a random draw instead of the band, which
    // is the same misreading in colour that the old median line was in geometry.
    for (index, series) in samples.iter().enumerate() {
        let slot = 1 + bands.len() + index;
        let faded = theme::MAUVE.mix(0.28);
        let style = ShapeStyle::from(&faded).stroke_width(1);
        let points: Vec<(f64, f64)> = if active(slot) {
            series
                .values
                .iter()
                .enumerate()
                .filter(|(_, value)| value.is_finite())
                .map(|(i, value)| (x_start + i as f64 + 0.5, *value as f64))
                .collect()
        } else {
            Vec::new()
        };
        let handle = chart.draw_series(LineSeries::new(points, style))?;
        if index == 0 {
            handle
                .label(format!("{} ancestral draws", samples.len()))
                .legend(legend_rect(&theme::MAUVE));
        }
    }

    // Quantile loci. The extremes are dimmest and the centre brightest, which is
    // the opposite of a forecast line's emphasis on purpose: the reader's eye
    // should land on the WIDTH of the fan first.
    for (index, band) in bands.iter().enumerate() {
        let distance = (band.probability - 0.5).abs() * 2.0;
        let color: &'static RGBColor = if band.probability > 0.5 {
            &theme::SAPPHIRE
        } else if band.probability < 0.5 {
            &theme::TEAL
        } else {
            &theme::YELLOW
        };
        let width = if (band.probability - 0.5).abs() < 1.0e-9 {
            3
        } else {
            2
        };
        let faded = color.mix(1.0 - 0.5 * distance);
        let style = ShapeStyle::from(&faded).stroke_width(width);
        let points: Vec<(f64, f64)> = if active(1 + index) {
            band.closes
                .iter()
                .enumerate()
                .filter(|(_, value)| value.is_finite())
                .map(|(i, value)| (x_start + i as f64 + 0.5, *value as f64))
                .collect()
        } else {
            Vec::new()
        };
        let label = if (band.probability - 0.5).abs() < 1.0e-9 {
            "p50 (fan centre, NOT a draw)".to_owned()
        } else {
            format!("p{:02}", (band.probability * 100.0).round() as i64)
        };
        chart
            .draw_series(LineSeries::new(points, style))?
            .label(label)
            .legend(legend_rect(color));
    }

    // Realized bars last, on top, in the up=green / down=red language.
    if actual_active {
        chart
            .draw_series(actual.iter().enumerate().map(|(idx, candle)| {
                let x = x_start + idx as f64;
                Rectangle::new(
                    [
                        (x + 0.2, candle_body_low(candle)),
                        (x + 0.8, candle_body_high(candle)),
                    ],
                    direction_color(candle).filled(),
                )
            }))?
            .label("realized")
            .legend(legend_rect(&theme::GREEN));
        chart.draw_series(actual.iter().enumerate().map(|(idx, candle)| {
            let mid = x_start + idx as f64 + 0.5;
            PathElement::new(
                vec![(mid, candle.low as f64), (mid, candle.high as f64)],
                ShapeStyle::from(&direction_color(candle)).stroke_width(2),
            )
        }))?;
    } else {
        chart
            .draw_series(LineSeries::new(
                std::iter::empty::<(f64, f64)>(),
                ShapeStyle::from(&theme::SURFACE2).stroke_width(1),
            ))?
            .label("realized")
            .legend(legend_rect(&theme::GREEN));
    }

    if show_legend {
        chart
            .configure_series_labels()
            .position(LegendConfig::position())
            .background_style(LegendConfig::background())
            .border_style(LegendConfig::border())
            .label_font(LegendConfig::font())
            .draw()?;
    }

    Ok(())
}

fn direction_color(candle: &CandleBar) -> RGBColor {
    if candle.close >= candle.open {
        theme::GREEN
    } else {
        theme::RED
    }
}

fn candle_body_low(candle: &CandleBar) -> f64 {
    let open = candle.open as f64;
    let close = candle.close as f64;
    open.min(close)
}

fn candle_body_high(candle: &CandleBar) -> f64 {
    let open = candle.open as f64;
    let close = candle.close as f64;
    open.max(close)
}

fn compute_ema(data: &[f32], alpha: f64) -> Vec<f32> {
    let mut result = Vec::with_capacity(data.len());
    let mut ema = None;
    for &v in data {
        if !v.is_finite() {
            result.push(v);
            continue;
        }
        let next = ema.map_or(v as f64, |previous| {
            alpha * v as f64 + (1.0 - alpha) * previous
        });
        ema = Some(next);
        result.push(next as f32);
    }
    result
}

fn symlog(x: f64) -> f64 {
    x.signum() * (1.0 + x.abs()).ln()
}

fn symlog_inv(y: f64) -> f64 {
    y.signum() * (y.abs().exp() - 1.0)
}

fn map_value(value: f64, scale: ScaleKind) -> f64 {
    match scale {
        ScaleKind::Linear => value,
        ScaleKind::Symlog => symlog(value),
    }
}

fn range_for(values: &[f32], is_symlog: bool) -> Result<(f64, f64)> {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| v as f64)
        .collect();
    if finite.is_empty() {
        return Err(anyhow!("no finite values"));
    }

    let y_min = finite
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
        .unwrap_or(0.0);
    let y_max = finite
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .copied()
        .unwrap_or(1.0);
    let y_range = (y_max - y_min).max(0.01);

    if is_symlog {
        Ok((
            symlog(y_min - y_range * 0.05),
            symlog(y_max + y_range * 0.05),
        ))
    } else {
        Ok((y_min - y_range * 0.05, y_max + y_range * 0.05))
    }
}

fn normalize_title(name: &str) -> String {
    let mut parts = Vec::new();
    for word in name.split_whitespace() {
        if word.eq_ignore_ascii_case("log") {
            parts.push("(Log)".to_string());
            continue;
        }
        let mut chars = word.chars();
        if let Some(first) = chars.next() {
            let rest = chars.as_str().to_ascii_lowercase();
            let mut word_out = String::new();
            word_out.push(first.to_ascii_uppercase());
            word_out.push_str(&rest);
            parts.push(word_out);
        }
    }
    parts.join(" ")
}

struct LegendConfig;

impl LegendConfig {
    fn position() -> SeriesLabelPosition {
        SeriesLabelPosition::UpperLeft
    }

    fn background() -> RGBAColor {
        theme::SURFACE0.mix(0.5)
    }

    fn border() -> &'static RGBColor {
        &theme::SURFACE1
    }

    fn font() -> (&'static str, i32, &'static RGBColor) {
        ("sans-serif", 14, &theme::TEXT)
    }
}

fn legend_rect(
    color: &impl Color,
) -> impl Fn((i32, i32)) -> plotters::element::Rectangle<(i32, i32)> + '_ {
    move |(x, y)| {
        plotters::element::Rectangle::new([(x, y - 5), (x + 20, y + 5)], color.mix(0.8).filled())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiline_single_point_is_visible() {
        let report = Report {
            title: "single point".to_string(),
            x_label: None,
            y_label: None,
            scale: ScaleKind::Linear,
            kind: ReportKind::MultiLine {
                series: vec![ReportSeries {
                    label: "sparse".to_string(),
                    values: vec![f32::NAN, 0.5, f32::NAN],
                }],
            },
        };

        let image = render_report_with_options(&report, 0, false, None)
            .unwrap()
            .to_rgb8();
        let blue_pixels = image
            .pixels()
            .filter(|pixel| {
                pixel[2] as i16 > pixel[0] as i16 + 50 && pixel[2] as i16 > pixel[1] as i16 + 30
            })
            .count();

        assert!(blue_pixels >= 10);
    }

    #[test]
    fn all_nan_simple_report_renders_an_explicit_placeholder() {
        let report = Report {
            title: "Undefined EV".to_owned(),
            x_label: Some("update".to_owned()),
            y_label: Some("EV".to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::Simple {
                values: vec![f32::NAN],
                ema_alpha: None,
            },
        };
        assert!(render_report_with_options(&report, 0, true, None).is_ok());
    }

    #[test]
    fn ema_preserves_finite_gaps_and_resumes_from_last_state() {
        let leading_gap = compute_ema(&[f32::NAN, 2.0, 4.0], 0.5);
        assert!(leading_gap[0].is_nan());
        assert_eq!(&leading_gap[1..], &[2.0, 3.0]);

        let middle_gap = compute_ema(&[1.0, f32::NAN, f32::INFINITY, 3.0], 0.5);
        assert_eq!(middle_gap[0], 1.0);
        assert!(middle_gap[1].is_nan());
        assert!(middle_gap[2].is_infinite());
        assert_eq!(middle_gap[3], 2.0);
    }

    #[test]
    fn assets_series_mapping_is_compact_when_optional_series_are_absent() {
        let values = [1.0];
        assert_eq!(
            asset_series_kinds(None, Some(&values)),
            vec![
                AssetSeriesKind::Total,
                AssetSeriesKind::Cash,
                AssetSeriesKind::Benchmark,
            ]
        );
        assert_eq!(
            asset_series_kinds(Some(&values), None),
            vec![
                AssetSeriesKind::Total,
                AssetSeriesKind::Positioned,
                AssetSeriesKind::Cash,
            ]
        );
        assert_eq!(asset_series_count(None, None), 2);
    }
}
