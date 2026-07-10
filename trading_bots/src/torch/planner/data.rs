use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use rand::{rngs::StdRng, Rng};
use tch::{Device, Tensor};

use crate::{
    data::{historical::get_cached_historical_bars, universe::cached_eligible_training_universe},
    torch::{
        env::{build_ohlc_features, OHLC_BAR_FEATURES},
        world_model::WorldModelMetadata,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum PlannerDataSplit {
    Train,
    Validation,
    Test,
}

#[derive(Clone)]
pub struct PlannerSeries {
    pub ticker: String,
    pub features: Vec<[f32; OHLC_BAR_FEATURES]>,
    pub closes: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannerEndpoint {
    pub series: usize,
    /// Last realized bar in the initial context.
    pub bar: usize,
}

pub struct PlannerDataset {
    series: Vec<PlannerSeries>,
}

impl PlannerDataset {
    pub fn load_cached(tickers: Option<&[String]>) -> Result<Self> {
        let tickers = tickers
            .map(|values| values.to_vec())
            .unwrap_or_else(|| cached_eligible_training_universe().to_vec());
        if tickers.is_empty() {
            bail!("planner dataset has no tickers");
        }

        let mut series = Vec::with_capacity(tickers.len());
        for ticker in tickers {
            let bars = get_cached_historical_bars(&ticker)
                .with_context(|| format!("no cached historical bars for {ticker}"))?;
            if bars.len() < 3 {
                continue;
            }
            let closes = bars.iter().map(|bar| bar.close).collect();
            series.push(PlannerSeries {
                ticker,
                features: build_ohlc_features(&bars),
                closes,
            });
        }
        if series.is_empty() {
            bail!("none of the requested tickers has usable cached history");
        }
        Ok(Self { series })
    }

    pub fn series(&self, index: usize) -> &PlannerSeries {
        &self.series[index]
    }

    pub fn sample_endpoints(
        &self,
        split: PlannerDataSplit,
        count: usize,
        context_bars: usize,
        actual_future_bars: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<PlannerEndpoint>> {
        let eligible = self.eligible_ranges(split, context_bars, actual_future_bars);
        if eligible.is_empty() {
            bail!(
                "no {split:?} planner endpoints have context={context_bars} and future={actual_future_bars}"
            );
        }
        let total: usize = eligible.iter().map(|(_, start, end)| end - start).sum();
        let mut endpoints = Vec::with_capacity(count);
        for _ in 0..count {
            let mut draw = rng.random_range(0..total);
            for &(series, start, end) in &eligible {
                let width = end - start;
                if draw < width {
                    endpoints.push(PlannerEndpoint {
                        series,
                        bar: start + draw,
                    });
                    break;
                }
                draw -= width;
            }
        }
        Ok(endpoints)
    }

    pub fn deterministic_endpoints(
        &self,
        split: PlannerDataSplit,
        count: usize,
        context_bars: usize,
        actual_future_bars: usize,
    ) -> Result<Vec<PlannerEndpoint>> {
        let eligible = self.eligible_ranges(split, context_bars, actual_future_bars);
        let all = eligible
            .iter()
            .flat_map(|&(series, start, end)| {
                (start..end).map(move |bar| PlannerEndpoint { series, bar })
            })
            .collect::<Vec<_>>();
        if all.is_empty() {
            bail!(
                "no {split:?} planner endpoints have context={context_bars} and future={actual_future_bars}"
            );
        }
        let count = count.min(all.len());
        Ok((0..count)
            .map(|index| all[index * all.len() / count])
            .collect())
    }

    pub fn contexts(
        &self,
        endpoints: &[PlannerEndpoint],
        advances: &[usize],
        context_bars: usize,
        device: Device,
    ) -> Result<Tensor> {
        if endpoints.len() != advances.len() || endpoints.is_empty() {
            bail!("planner endpoints/advances must be non-empty and equally sized");
        }
        let mut data = Vec::with_capacity(endpoints.len() * context_bars * OHLC_BAR_FEATURES);
        for (endpoint, advance) in endpoints.iter().zip(advances) {
            let series = self.series(endpoint.series);
            let end = endpoint.bar + advance + 1;
            let start = end
                .checked_sub(context_bars)
                .context("planner context precedes available history")?;
            let slice = series
                .features
                .get(start..end)
                .context("planner context exceeds available history")?;
            data.extend(slice.iter().flatten().copied());
        }
        Ok(Tensor::from_slice(&data)
            .view([
                endpoints.len() as i64,
                1,
                context_bars as i64,
                OHLC_BAR_FEATURES as i64,
            ])
            .to_device(device))
    }

    fn eligible_ranges(
        &self,
        split: PlannerDataSplit,
        context_bars: usize,
        actual_future_bars: usize,
    ) -> Vec<(usize, usize, usize)> {
        self.series
            .iter()
            .enumerate()
            .filter_map(|(index, series)| {
                let (split_start, split_end) = chronological_split(series.closes.len(), split);
                let start = split_start.max(context_bars.saturating_sub(1));
                let end = split_end.saturating_sub(actual_future_bars);
                (start < end).then_some((index, start, end))
            })
            .collect()
    }
}

pub fn planner_context_bars(
    metadata: &WorldModelMetadata,
    requested: Option<usize>,
) -> Result<usize> {
    let maximum = metadata.max_context_bars as usize;
    let context = requested.unwrap_or(maximum);
    if context == 0 || context > maximum {
        bail!("planner context bars must be in 1..={maximum}, got {context}");
    }
    Ok(context)
}

fn chronological_split(length: usize, split: PlannerDataSplit) -> (usize, usize) {
    let train = length * 8 / 10;
    let validation = length * 9 / 10;
    match split {
        PlannerDataSplit::Train => (0, train),
        PlannerDataSplit::Validation => (train, validation),
        PlannerDataSplit::Test => (validation, length),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(length: usize) -> PlannerDataset {
        PlannerDataset {
            series: vec![PlannerSeries {
                ticker: "TEST".to_owned(),
                features: (0..length)
                    .map(|index| [index as f32; OHLC_BAR_FEATURES])
                    .collect(),
                closes: (0..length).map(|index| index as f64 + 1.0).collect(),
            }],
        }
    }

    #[test]
    fn chronological_ranges_keep_targets_inside_split() {
        let data = synthetic(1_000);
        let train = data.eligible_ranges(PlannerDataSplit::Train, 100, 100);
        let validation = data.eligible_ranges(PlannerDataSplit::Validation, 100, 20);
        let test = data.eligible_ranges(PlannerDataSplit::Test, 100, 20);
        assert_eq!(train, vec![(0, 99, 700)]);
        assert_eq!(validation, vec![(0, 800, 880)]);
        assert_eq!(test, vec![(0, 900, 980)]);
    }

    #[test]
    fn context_ends_at_endpoint_plus_advance() {
        let data = synthetic(200);
        let tensor = data
            .contexts(
                &[PlannerEndpoint {
                    series: 0,
                    bar: 100,
                }],
                &[3],
                4,
                Device::Cpu,
            )
            .unwrap();
        assert_eq!(tensor.size(), [1, 1, 4, OHLC_BAR_FEATURES as i64]);
        assert_eq!(tensor.double_value(&[0, 0, 0, 0]), 100.0);
        assert_eq!(tensor.double_value(&[0, 0, 3, 0]), 103.0);
    }

    #[test]
    fn deterministic_endpoints_are_stable_and_spread() {
        let data = synthetic(1_000);
        let endpoints = data
            .deterministic_endpoints(PlannerDataSplit::Test, 3, 100, 20)
            .unwrap();
        assert_eq!(
            endpoints,
            vec![
                PlannerEndpoint {
                    series: 0,
                    bar: 900
                },
                PlannerEndpoint {
                    series: 0,
                    bar: 926
                },
                PlannerEndpoint {
                    series: 0,
                    bar: 953
                },
            ]
        );
    }
}
