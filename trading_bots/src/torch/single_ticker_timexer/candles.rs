use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use clap::Args;
use shared::report::{write_report, HorizonForecast, Report, ReportKind, ScaleKind};
use tch::{nn, Device, Kind};

use super::{
    checkpoint,
    data::{Dataset, Split, HORIZONS},
    model::ForecastModel,
    support::{Supports, BINS},
};

const HISTORY: usize = 32;
const WINDOWS: usize = 4;

#[derive(Clone, Debug, Args)]
pub struct CandleArgs {
    /// Checkpoint whose historical validation forecasts should be visualized.
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long, default_value_os_t = crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
}

pub fn run(args: CandleArgs) -> Result<()> {
    let manifest = checkpoint::read(&args.checkpoint)?;
    ensure!(
        manifest.model_kind.is_probabilistic(),
        "candle forecasts require a probabilistic model"
    );
    let dataset = Dataset::load(&args.data_dir, &manifest.data.ticker)?;
    manifest.authenticate_dataset(&dataset)?;
    let device = super::runner::cuda_device()?;
    let mut store = nn::VarStore::new(device);
    let model = ForecastModel::new(&store.root(), manifest.model_kind);
    checkpoint::load_weights(&args.checkpoint, &mut store)?;
    store.freeze();
    write_windows(
        &dataset,
        &model,
        device,
        manifest.epoch,
        manifest.step,
        &args.output,
    )
}

pub(super) fn write_windows(
    dataset: &Dataset,
    model: &ForecastModel,
    device: Device,
    epoch: usize,
    step: usize,
    output: &Path,
) -> Result<()> {
    ensure!(
        model.kind().is_probabilistic(),
        "candle forecasts require a probabilistic model"
    );
    let origins = fixed_origins(dataset.origins(Split::Validation))?;
    let batch = dataset.inference_batch(&origins, device)?;
    let probabilities = tch::no_grad(|| {
        model
            .forward(
                &batch.endogenous,
                &batch.exogenous,
                &batch.validity,
                &batch.future_clock,
                false,
            )
            .to_kind(Kind::Float)
            .softmax(-1, Kind::Float)
            .to_device(Device::Cpu)
            .to_kind(Kind::Double)
            .view([-1])
    });
    let probabilities = Vec::<f64>::try_from(probabilities)?;
    ensure!(
        probabilities.len() == WINDOWS * HORIZONS.len() * BINS,
        "unexpected candle forecast shape"
    );
    for (window, (&origin, probabilities)) in origins
        .iter()
        .zip(probabilities.chunks_exact(HORIZONS.len() * BINS))
        .enumerate()
    {
        let actual = dataset
            .candle_window(origin, HISTORY, HORIZONS[5])?
            .to_vec();
        let forecasts = price_forecasts(
            &dataset.supports,
            probabilities,
            actual[HISTORY].close as f64,
            dataset.sigma(origin),
        )?;
        let completed_at = dataset
            .timestamp(origin)
            .checked_add(300_000)
            .context("forecast origin timestamp overflow")?;
        let clock = chrono::DateTime::from_timestamp_millis(completed_at)
            .context("invalid forecast origin timestamp")?
            .with_timezone(&chrono_tz::America::New_York);
        let report = Report {
            title: format!("{} epoch {epoch} | validation window {} | origin {} | 90% marginal close intervals", dataset.ticker, window + 1, clock.format("%Y-%m-%d %H:%M %Z")),
            x_label: Some("observed five-minute bars relative to forecast origin".to_owned()),
            y_label: Some("price".to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::CandleForecast { actual, origin: HISTORY, forecasts },
        };
        if window == 0 {
            write_report(output.join("timexer_candles.report.bin"), &report)?;
        }
        write_report(
            output.join("candle_snapshots").join(format!(
                "step{step}_epoch{epoch:03}_window{:02}_fan.report.bin",
                window + 1
            )),
            &report,
        )?;
    }
    Ok(())
}

fn fixed_origins(origins: &[usize]) -> Result<[usize; WINDOWS]> {
    ensure!(
        origins.len() >= WINDOWS,
        "candle charts need four validation origins"
    );
    Ok(std::array::from_fn(|i| {
        origins[(origins.len() - 1) * i / (WINDOWS - 1)]
    }))
}

fn price_forecasts(
    supports: &Supports,
    probabilities: &[f64],
    close: f64,
    sigma: f64,
) -> Result<Vec<HorizonForecast>> {
    ensure!(
        close.is_finite() && close > 0.0 && sigma.is_finite() && sigma > 0.0,
        "invalid forecast origin price or volatility"
    );
    ensure!(
        probabilities.len() == HORIZONS.len() * BINS,
        "unexpected horizon probability count"
    );
    HORIZONS
        .iter()
        .zip(&supports.horizons)
        .zip(probabilities.chunks_exact(BINS))
        .map(|((&horizon, support), probabilities)| {
            ensure!(
                probabilities.iter().all(|p| p.is_finite() && *p >= 0.0),
                "invalid forecast probability"
            );
            let mass: f64 = probabilities.iter().sum();
            ensure!(
                (mass - 1.0).abs() < 1e-5,
                "forecast probabilities do not sum to one"
            );
            let probabilities: Vec<_> = probabilities.iter().map(|p| p / mass).collect();
            let [lower, median, upper] = support.quantiles(&probabilities, [0.05, 0.5, 0.95]);
            ensure!(
                lower <= median && median <= upper,
                "unordered forecast quantiles"
            );
            let scale = sigma * (horizon as f64).sqrt();
            Ok(HorizonForecast {
                horizon,
                lower: quantile_price(close, scale, lower)?,
                median: quantile_price(close, scale, median)?,
                upper: quantile_price(close, scale, upper)?,
            })
        })
        .collect()
}

fn quantile_price(close: f64, scale: f64, quantile: f64) -> Result<f32> {
    let price = (close.ln() + scale * quantile).exp();
    let plotted = price as f32;
    ensure!(
        plotted.is_finite() && plotted > 0.0,
        "forecast price cannot be represented in a candle report"
    );
    Ok(plotted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_origins_are_evenly_spaced_without_outcome_selection() {
        assert_eq!(
            fixed_origins(&(2000..2100).collect::<Vec<_>>()).unwrap(),
            [2000, 2033, 2066, 2099]
        );
        assert_eq!(fixed_origins(&[10, 20, 30, 40]).unwrap(), [10, 20, 30, 40]);
        assert!(fixed_origins(&[10, 20, 30]).is_err());
    }

    #[test]
    fn marginal_returns_map_to_exact_horizon_close_prices() {
        let targets: Vec<_> = (0..1024)
            .map(|i| [((i as f64 + 0.5) / 1024.0 - 0.5) * 4.0; 6])
            .collect();
        let supports = Supports::fit(&targets).unwrap();
        let probabilities = vec![1.0 / BINS as f64; HORIZONS.len() * BINS];
        let forecasts = price_forecasts(&supports, &probabilities, 100.0, 0.01).unwrap();
        for (i, forecast) in forecasts.iter().enumerate() {
            assert_eq!(forecast.horizon, HORIZONS[i]);
            assert_eq!(HISTORY + forecast.horizon, [33, 36, 48, 71, 110, 132][i]);
            let q = supports.horizons[i].quantile(&probabilities[..BINS], 0.95);
            let expected = (100.0 * (0.01 * (HORIZONS[i] as f64).sqrt() * q).exp()) as f32;
            assert!((forecast.upper - expected).abs() < 0.0001);
            assert!(forecast.lower <= forecast.median && forecast.median <= forecast.upper);
        }
        let mut invalid = probabilities;
        invalid[0] = f64::NAN;
        assert!(price_forecasts(&supports, &invalid, 100.0, 0.01).is_err());
    }

    #[test]
    fn unrepresentable_price_quantiles_fail_explicitly() {
        assert_eq!(quantile_price(100.0, 0.01, 0.0).unwrap(), 100.0);
        for quantile in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e8, -1e8] {
            assert!(quantile_price(100.0, 0.01, quantile).is_err());
        }
    }
}
