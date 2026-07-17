use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use rand::{rngs::StdRng, Rng};
use ring::digest::{Context as DigestContext, SHA256};
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

fn no_endpoints_err(
    split: PlannerDataSplit,
    context_bars: usize,
    actual_future_bars: usize,
) -> anyhow::Error {
    anyhow::anyhow!(
        "no {split:?} planner endpoints have context={context_bars} and future={actual_future_bars}"
    )
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

    pub fn evaluation_fingerprint(
        &self,
        split: PlannerDataSplit,
        endpoints: &[PlannerEndpoint],
        horizon: usize,
        context_bars: usize,
        rollout_length: usize,
    ) -> Result<String> {
        if endpoints.is_empty() || context_bars == 0 || rollout_length == 0 {
            bail!("planner evaluation fingerprint requires non-empty endpoints and lengths");
        }
        let mut digest = DigestContext::new(&SHA256);
        digest.update(b"planner-evaluation-v1");
        digest.update(&[match split {
            PlannerDataSplit::Train => 0,
            PlannerDataSplit::Validation => 1,
            PlannerDataSplit::Test => 2,
        }]);
        for value in [horizon, context_bars, rollout_length, endpoints.len()] {
            digest.update(&(value as u64).to_le_bytes());
        }
        for endpoint in endpoints {
            let series = self.series(endpoint.series);
            let start = (endpoint.bar + 1)
                .checked_sub(context_bars)
                .context("planner evaluation context precedes available history")?;
            let end = endpoint.bar + rollout_length + 1;
            let features = series
                .features
                .get(start..end)
                .context("planner evaluation features exceed available history")?;
            let closes = series
                .closes
                .get(start..end)
                .context("planner evaluation prices exceed available history")?;
            digest.update(&(series.ticker.len() as u64).to_le_bytes());
            digest.update(series.ticker.as_bytes());
            digest.update(&(endpoint.bar as u64).to_le_bytes());
            for feature in features.iter().flatten() {
                digest.update(&feature.to_bits().to_le_bytes());
            }
            for close in closes {
                digest.update(&close.to_bits().to_le_bytes());
            }
        }
        Ok(digest
            .finish()
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect())
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
            return Err(no_endpoints_err(split, context_bars, actual_future_bars));
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
            return Err(no_endpoints_err(split, context_bars, actual_future_bars));
        }
        let count = count.min(all.len());
        if count == 0 {
            bail!("planner deterministic endpoint count must be positive");
        }
        Ok((0..count)
            .map(|index| all[index * all.len() / count])
            .collect())
    }

    pub fn deterministic_ticker_stratified_endpoints(
        &self,
        split: PlannerDataSplit,
        count: usize,
        context_bars: usize,
        actual_future_bars: usize,
    ) -> Result<Vec<PlannerEndpoint>> {
        if count == 0 {
            bail!("planner stratified endpoint count must be positive");
        }
        let eligible = self.eligible_ranges(split, context_bars, actual_future_bars);
        if eligible.is_empty() {
            return Err(no_endpoints_err(split, context_bars, actual_future_bars));
        }
        let count = count.min(eligible.len());
        Ok((0..count)
            .map(|index| {
                let &(series, start, end) = &eligible[index * eligible.len() / count];
                PlannerEndpoint {
                    series,
                    bar: start + (end - start) / 2,
                }
            })
            .collect())
    }

    pub fn deterministic_ticker_time_stratified_endpoints(
        &self,
        split: PlannerDataSplit,
        count: usize,
        context_bars: usize,
        actual_future_bars: usize,
    ) -> Result<Vec<PlannerEndpoint>> {
        if count == 0 {
            bail!("planner ticker-time stratified endpoint count must be positive");
        }
        let eligible = self.eligible_ranges(split, context_bars, actual_future_bars);
        if eligible.is_empty() {
            return Err(no_endpoints_err(split, context_bars, actual_future_bars));
        }

        const MIN_TIME_STRATA: usize = 4;
        let initial_series = eligible.len().min((count / MIN_TIME_STRATA).max(1));
        let mut selected_indices = (0..initial_series)
            .map(|index| index * eligible.len() / initial_series)
            .collect::<Vec<_>>();
        let mut selected = vec![false; eligible.len()];
        for &index in &selected_indices {
            selected[index] = true;
        }
        let mut capacity = selected_indices
            .iter()
            .map(|&index| eligible[index].2 - eligible[index].1)
            .sum::<usize>();
        if capacity < count {
            for (index, &(_, start, end)) in eligible.iter().enumerate() {
                if !selected[index] {
                    selected_indices.push(index);
                    capacity += end - start;
                    if capacity >= count {
                        break;
                    }
                }
            }
        }
        if capacity < count {
            bail!(
                "planner ticker-time stratification requires {count} distinct endpoints, but only {capacity} fit inside the requested split"
            );
        }

        let capacities = selected_indices
            .iter()
            .map(|&index| eligible[index].2 - eligible[index].1)
            .collect::<Vec<_>>();
        let mut allocations = vec![0usize; selected_indices.len()];
        let mut remaining = count;
        while remaining > 0 {
            for (allocation, &series_capacity) in allocations.iter_mut().zip(&capacities) {
                if *allocation < series_capacity && remaining > 0 {
                    *allocation += 1;
                    remaining -= 1;
                }
            }
        }
        let mut endpoints = Vec::with_capacity(count);
        for (&eligible_index, &slots) in selected_indices.iter().zip(&allocations) {
            let (series, start, end) = eligible[eligible_index];
            for slot in 0..slots {
                let offset = (2 * slot + 1) * (end - start) / (2 * slots);
                endpoints.push(PlannerEndpoint {
                    series,
                    bar: start + offset,
                });
            }
        }
        debug_assert_eq!(endpoints.len(), count);
        Ok(endpoints)
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
    fn evaluation_fingerprint_covers_contract_endpoints_and_data() {
        let mut dataset = synthetic(1_000);
        let endpoints = vec![PlannerEndpoint {
            series: 0,
            bar: 900,
        }];
        let base = dataset
            .evaluation_fingerprint(PlannerDataSplit::Test, &endpoints, 100, 128, 50)
            .unwrap();
        assert_ne!(
            base,
            dataset
                .evaluation_fingerprint(PlannerDataSplit::Test, &endpoints, 100, 128, 51)
                .unwrap()
        );
        dataset.series[0].closes[900] += 1.0;
        assert_ne!(
            base,
            dataset
                .evaluation_fingerprint(PlannerDataSplit::Test, &endpoints, 100, 128, 50)
                .unwrap()
        );
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

    #[test]
    fn ticker_stratified_endpoints_select_distinct_series_midpoints() {
        let data = PlannerDataset {
            series: (0..4)
                .map(|index| PlannerSeries {
                    ticker: format!("T{index}"),
                    features: vec![[0.0; OHLC_BAR_FEATURES]; 1_000],
                    closes: vec![1.0; 1_000],
                })
                .collect(),
        };
        let endpoints = data
            .deterministic_ticker_stratified_endpoints(PlannerDataSplit::Validation, 3, 100, 20)
            .unwrap();
        assert_eq!(
            endpoints.iter().map(|e| e.series).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(endpoints.iter().all(|endpoint| endpoint.bar == 840));
    }

    #[test]
    fn ticker_time_stratification_covers_each_ticker_and_multiple_periods() {
        let data = PlannerDataset {
            series: (0..4)
                .map(|index| PlannerSeries {
                    ticker: format!("T{index}"),
                    features: vec![[0.0; OHLC_BAR_FEATURES]; 1_000],
                    closes: vec![1.0; 1_000],
                })
                .collect(),
        };
        let endpoints = data
            .deterministic_ticker_time_stratified_endpoints(
                PlannerDataSplit::Validation,
                16,
                100,
                20,
            )
            .unwrap();
        assert_eq!(endpoints.len(), 16);
        for series in 0..4 {
            let bars = endpoints
                .iter()
                .filter(|endpoint| endpoint.series == series)
                .map(|endpoint| endpoint.bar)
                .collect::<Vec<_>>();
            assert_eq!(bars.len(), 4);
            assert!(bars.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(bars[0] < 840 && bars[2] > 840);
        }
    }

    #[test]
    fn ticker_time_stratification_redistributes_short_series_capacity() {
        let data = PlannerDataset {
            series: [210, 230, 500, 1_000]
                .into_iter()
                .enumerate()
                .map(|(index, length)| PlannerSeries {
                    ticker: format!("T{index}"),
                    features: vec![[0.0; OHLC_BAR_FEATURES]; length],
                    closes: vec![1.0; length],
                })
                .collect(),
        };
        let endpoints = data
            .deterministic_ticker_time_stratified_endpoints(
                PlannerDataSplit::Validation,
                64,
                100,
                20,
            )
            .unwrap();
        assert_eq!(endpoints.len(), 64);
        assert_eq!(
            endpoints
                .iter()
                .filter(|endpoint| endpoint.series == 0)
                .count(),
            1
        );
        assert_eq!(
            endpoints
                .iter()
                .filter(|endpoint| endpoint.series == 1)
                .count(),
            3
        );
    }

    #[test]
    fn ticker_time_stratification_fails_when_global_capacity_is_too_small() {
        let data = PlannerDataset {
            series: [210, 230]
                .into_iter()
                .enumerate()
                .map(|(index, length)| PlannerSeries {
                    ticker: format!("T{index}"),
                    features: vec![[0.0; OHLC_BAR_FEATURES]; length],
                    closes: vec![1.0; length],
                })
                .collect(),
        };
        assert!(data
            .deterministic_ticker_time_stratified_endpoints(
                PlannerDataSplit::Validation,
                5,
                100,
                20,
            )
            .is_err());
    }
}
