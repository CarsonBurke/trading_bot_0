use anyhow::{anyhow, Result};
use image::{DynamicImage, RgbImage};
use plotters::coord::Shift;
use plotters::prelude::*;
use shared::report::{Report, ReportKind, ReportSeries, ScaleKind, TradePoint};
use shared::theme::plotters_colors as theme;

const CHART_DIMS: (u32, u32) = (2560, 780);

pub fn render_report(report: &Report) -> Result<DynamicImage> {
    render_report_with_options(report, 0, true)
}

pub fn render_report_with_skip(report: &Report, skip: usize) -> Result<DynamicImage> {
    render_report_with_options(report, skip, true)
}

pub fn render_report_with_options(
    report: &Report,
    skip: usize,
    show_legend: bool,
) -> Result<DynamicImage> {
    let mut buffer = vec![0u8; (CHART_DIMS.0 * CHART_DIMS.1 * 3) as usize];
    {
        let root = BitMapBackend::with_buffer(&mut buffer, CHART_DIMS).into_drawing_area();
        root.fill(&theme::BASE)?;

        let x_offset = skip as u32 + report.x_offset;

        match &report.kind {
            ReportKind::Simple { values, ema_alpha } => {
                let values = skip_slice(values, skip);
                render_simple(&root, report, values, *ema_alpha, x_offset, show_legend)?;
            }
            ReportKind::MultiLine { series } => {
                let series: Vec<ReportSeries> = series
                    .iter()
                    .map(|s| ReportSeries {
                        label: s.label.clone(),
                        values: skip_slice(&s.values, skip).to_vec(),
                    })
                    .collect();
                render_multi_line(&root, report, &series, x_offset, show_legend)?;
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
) -> Result<()> {
    if values.is_empty() {
        return Ok(());
    }

    let scale = report.scale;
    let (y_min, y_max) = range_for(values, scale == ScaleKind::Symlog)?;
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

    if let Some(alpha) = ema_alpha {
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
) -> Result<()> {
    let all_values: Vec<f32> = series.iter().flat_map(|s| s.values.iter()).copied().collect();
    if all_values.is_empty() {
        return Ok(());
    }

    let scale = report.scale;
    let (y_min, y_max) = range_for(&all_values, scale == ScaleKind::Symlog)?;
    let x_len = series
        .iter()
        .map(|s| s.values.len())
        .max()
        .unwrap_or(1) as u32;
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

    for (i, series) in series.iter().enumerate() {
        let color = colors[i % colors.len()];
        let mapped = series
            .values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(idx, v)| (x_offset + idx as u32, map_value(*v as f64, scale)));
        chart
            .draw_series(LineSeries::new(
                mapped,
                ShapeStyle::from(&color.mix(0.8)).stroke_width(1),
            ))?
            .label(series.label.as_str())
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

fn render_assets(
    root: &DrawingArea<BitMapBackend, Shift>,
    report: &Report,
    total: &[f32],
    cash: &[f32],
    positioned: Option<&Vec<f32>>,
    benchmark: Option<&Vec<f32>>,
    x_offset: u32,
    show_legend: bool,
) -> Result<()> {
    if total.is_empty() {
        return Ok(());
    }

    let mut max_val = total
        .iter()
        .map(|v| *v as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    if let Some(bench) = benchmark {
        let bench_max = bench
            .iter()
            .map(|v| *v as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        max_val = max_val.max(bench_max);
    }

    let y_max = max_val as f32 * 1.1;
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

    chart
        .draw_series(
        AreaSeries::new(
            total
                .iter()
                .enumerate()
                .map(|(index, value)| (x_offset + index as u32, *value as f32)),
            0.0,
            theme::BLUE.mix(0.2),
        )
        .border_style(ShapeStyle::from(&theme::BLUE).stroke_width(1)),
    )?
        .label("total")
        .legend(legend_rect(&theme::BLUE));

    let positioned = positioned.map(|p| p.as_slice()).unwrap_or(&[]);
    if !positioned.is_empty() {
        chart
            .draw_series(
            AreaSeries::new(
                positioned
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (x_offset + index as u32, *value as f32)),
                0.0,
                theme::RED.mix(0.2),
            )
            .border_style(ShapeStyle::from(&theme::RED).stroke_width(1)),
        )?
            .label("positioned")
            .legend(legend_rect(&theme::RED));
    }

    chart
        .draw_series(
        AreaSeries::new(
            cash.iter()
                .enumerate()
                .map(|(index, value)| (x_offset + index as u32, *value as f32)),
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
                .map(|(index, value)| (x_offset + index as u32, *value as f32)),
            ShapeStyle::from(&theme::MAUVE).stroke_width(1),
        ))?
            .label("benchmark")
            .legend(legend_rect(&theme::MAUVE));
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
            .map(|p| (x_offset + p.index, prices.get(p.index as usize).copied().unwrap_or(0.0) as f64)),
        point_size,
        theme::YELLOW.mix(0.9).filled(),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    chart.draw_series(PointSeries::of_element(
        buys
            .iter()
            .filter(|p| (p.index as usize) < prices.len())
            .map(|p| (x_offset + p.index, prices.get(p.index as usize).copied().unwrap_or(0.0) as f64)),
        point_size,
        theme::RED.mix(0.9).filled(),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    Ok(())
}

fn compute_ema(data: &[f32], alpha: f64) -> Vec<f32> {
    if data.is_empty() {
        return vec![];
    }
    let mut result = Vec::with_capacity(data.len());
    let mut ema = data[0] as f64;
    for &v in data {
        ema = alpha * v as f64 + (1.0 - alpha) * ema;
        result.push(ema as f32);
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
