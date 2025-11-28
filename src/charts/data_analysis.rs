use plotters::prelude::*;
use std::error::Error;

use crate::utils::create_folder_if_not_exists;

pub fn create_data_analysis_charts(
    ticker: &str,
    prices: &[f64],
    base_dir: &str,
) -> Result<(), Box<dyn Error>> {
    let ticker_dir = format!("{}/{}", base_dir, ticker);
    create_folder_if_not_exists(&ticker_dir);

    volatility_chart(&ticker_dir, ticker, prices)?;
    returns_distribution_chart(&ticker_dir, ticker, prices)?;
    rolling_metrics_chart(&ticker_dir, ticker, prices)?;
    price_statistics_chart(&ticker_dir, ticker, prices)?;

    Ok(())
}

fn volatility_chart(
    dir: &str,
    ticker: &str,
    prices: &[f64],
) -> Result<(), Box<dyn Error>> {
    const WINDOW: usize = 20;

    let mut rolling_volatility = Vec::new();

    for i in WINDOW..prices.len() {
        let window_prices = &prices[i - WINDOW..i];
        let returns: Vec<f64> = window_prices
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let std_dev = variance.sqrt();

        // Annualized volatility (assuming 252 trading days, 78 5-min bars per day)
        let annualized_vol = std_dev * (252.0 * 78.0_f64).sqrt();
        rolling_volatility.push(annualized_vol * 100.0); // As percentage
    }

    if rolling_volatility.is_empty() {
        return Ok(());
    }

    let path = format!("{}/volatility.png", dir);
    let root = BitMapBackend::new(&path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_vol = rolling_volatility
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_vol = rolling_volatility
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} - Rolling 20-Period Volatility (Annualized %)", ticker),
            ("sans-serif", 30),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0..rolling_volatility.len(),
            min_vol * 0.9..max_vol * 1.1,
        )?;

    chart
        .configure_mesh()
        .x_desc("Time Step")
        .y_desc("Volatility (%)")
        .draw()?;

    chart.draw_series(LineSeries::new(
        rolling_volatility.iter().enumerate().map(|(i, v)| (i, *v)),
        &BLUE,
    ))?;

    root.present()?;
    Ok(())
}

fn returns_distribution_chart(
    dir: &str,
    ticker: &str,
    prices: &[f64],
) -> Result<(), Box<dyn Error>> {
    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| ((w[1] / w[0]) - 1.0) * 100.0) // Percentage returns
        .collect();

    if returns.is_empty() {
        return Ok(());
    }

    let path = format!("{}/returns_distribution.png", dir);
    let root = BitMapBackend::new(&path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Create histogram bins
    const NUM_BINS: usize = 50;
    let max_return = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_return = returns.iter().cloned().fold(f64::INFINITY, f64::min);
    let bin_width = (max_return - min_return) / NUM_BINS as f64;

    let mut histogram = vec![0u32; NUM_BINS];
    for &ret in &returns {
        let bin = ((ret - min_return) / bin_width).floor() as usize;
        let bin = bin.min(NUM_BINS - 1);
        histogram[bin] += 1;
    }

    let max_count = *histogram.iter().max().unwrap_or(&1);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} - Returns Distribution", ticker),
            ("sans-serif", 30),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            min_return..max_return,
            0u32..max_count + max_count / 10,
        )?;

    chart
        .configure_mesh()
        .x_desc("Return (%)")
        .y_desc("Frequency")
        .draw()?;

    // Draw histogram as bars
    chart.draw_series(histogram.iter().enumerate().map(|(i, &count)| {
        let x_left = min_return + i as f64 * bin_width;
        let x_right = min_return + (i + 1) as f64 * bin_width;
        Rectangle::new(
            [(x_left, 0), (x_right, count)],
            BLUE.filled(),
        )
    }))?;

    root.present()?;
    Ok(())
}

fn rolling_metrics_chart(
    dir: &str,
    ticker: &str,
    prices: &[f64],
) -> Result<(), Box<dyn Error>> {
    const WINDOW: usize = 100;

    let mut rolling_sharpe = Vec::new();
    let mut rolling_max_drawdown = Vec::new();

    for i in WINDOW..prices.len() {
        let window_prices = &prices[i - WINDOW..i];

        // Calculate Sharpe ratio
        let returns: Vec<f64> = window_prices
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let std_dev = variance.sqrt();

        let sharpe = if std_dev > 1e-10 {
            mean_return / std_dev * (252.0 * 78.0_f64).sqrt() // Annualized
        } else {
            0.0
        };
        rolling_sharpe.push(sharpe);

        // Calculate max drawdown
        let mut peak = window_prices[0];
        let mut max_dd = 0.0;

        for &price in window_prices {
            if price > peak {
                peak = price;
            }
            let drawdown = (peak - price) / peak;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }
        rolling_max_drawdown.push(max_dd * 100.0); // As percentage
    }

    if rolling_sharpe.is_empty() {
        return Ok(());
    }

    let path = format!("{}/rolling_metrics.png", dir);
    let root = BitMapBackend::new(&path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(300);

    // Sharpe ratio chart
    {
        let max_sharpe = rolling_sharpe
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_sharpe = rolling_sharpe
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        let mut chart = ChartBuilder::on(&upper)
            .caption(
                format!("{} - Rolling 100-Period Sharpe Ratio", ticker),
                ("sans-serif", 25),
            )
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(60)
            .build_cartesian_2d(
                0..rolling_sharpe.len(),
                min_sharpe * 1.1..max_sharpe * 1.1,
            )?;

        chart
            .configure_mesh()
            .y_desc("Sharpe Ratio")
            .draw()?;

        chart.draw_series(LineSeries::new(
            rolling_sharpe.iter().enumerate().map(|(i, v)| (i, *v)),
            &GREEN,
        ))?;
    }

    // Max drawdown chart
    {
        let max_dd = rolling_max_drawdown
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(&lower)
            .caption(
                format!("{} - Rolling 100-Period Max Drawdown", ticker),
                ("sans-serif", 25),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0..rolling_max_drawdown.len(), 0.0..max_dd * 1.1)?;

        chart
            .configure_mesh()
            .x_desc("Time Step")
            .y_desc("Max Drawdown (%)")
            .draw()?;

        chart.draw_series(LineSeries::new(
            rolling_max_drawdown.iter().enumerate().map(|(i, v)| (i, *v)),
            &RED,
        ))?;
    }

    root.present()?;
    Ok(())
}

fn price_statistics_chart(
    dir: &str,
    ticker: &str,
    prices: &[f64],
) -> Result<(), Box<dyn Error>> {
    if prices.is_empty() {
        return Ok(());
    }

    let path = format!("{}/price_statistics.png", dir);
    let root = BitMapBackend::new(&path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    // Calculate statistics
    let min_price = prices.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_price = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_price = prices.iter().sum::<f64>() / prices.len() as f64;

    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| ((w[1] / w[0]) - 1.0) * 100.0)
        .collect();

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance_return = returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;
    let std_return = variance_return.sqrt();

    // Draw price with statistics overlay
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} - Price with Statistics", ticker),
            ("sans-serif", 30),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..prices.len(), min_price * 0.95..max_price * 1.05)?;

    chart
        .configure_mesh()
        .x_desc("Time Step")
        .y_desc("Price")
        .draw()?;

    // Price line
    chart.draw_series(LineSeries::new(
        prices.iter().enumerate().map(|(i, p)| (i, *p)),
        &BLUE,
    ))?
    .label("Price")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Mean line
    chart.draw_series(LineSeries::new(
        (0..prices.len()).map(|i| (i, mean_price)),
        GREEN.stroke_width(2),
    ))?
    .label(format!("Mean: ${:.2}", mean_price))
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

    // Add text with statistics
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    // Draw statistics text box - use draw instead for better control
    let text_style = TextStyle::from(("sans-serif", 20).into_font()).color(&BLACK);
    let stats_text = format!(
        "Statistics:\nMin: ${:.2}\nMax: ${:.2}\nMean Return: {:.4}%\nStd Return: {:.4}%\nTotal Points: {}",
        min_price, max_price, mean_return, std_return, prices.len()
    );

    let text_elem = Text::new(stats_text, (50, 100), text_style);
    root.draw(&text_elem)?;

    root.present()?;
    Ok(())
}
