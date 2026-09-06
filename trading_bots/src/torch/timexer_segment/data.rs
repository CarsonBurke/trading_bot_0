use anyhow::{ensure, Result};
use ring::digest::{Context as Digest, SHA256};
use serde::{Deserialize, Serialize};
use shared::bars::PackedBar;

const RESOLUTION: u32 = 300;
const SCHEMA: &str = "timexer-segment-ohlc-v2;persistence-anchored-sigma-scaled-log-returns;next-valid-observed-bars;shared-universe-utc-split-boundaries;left-label-purge;completed-five-minute-bars;purge-only-at-observed-partition-boundaries;strict-raw-timestamps;skip-only-nonfinite-nonpositive-or-inconsistent-source-OHLC-v1;logical-valid-bar-ordinals";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DataContract {
    pub schema: String,
    pub ticker: String,
    pub fingerprint: String,
    pub source_bars: usize,
    pub valid_bars: usize,
    pub invalid_ohlc_indices: Vec<usize>,
    pub boundaries: [usize; 3],
    pub boundary_timestamps: [i64; 3],
    pub context: usize,
    pub pred_len: usize,
    pub purge: usize,
    /// Exclusive end of purged training targets, in valid-bar ordinals.
    pub train_end: usize,
    pub common_context: usize,
}

pub(super) fn valid_ohlc(bar: &PackedBar) -> bool {
    [bar.open, bar.high, bar.low, bar.close]
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
        && bar.high >= bar.open.max(bar.close)
        && bar.low <= bar.open.min(bar.close)
}

pub(super) fn filtered_contract(
    ticker: &str,
    bars: &[PackedBar],
    context: usize,
    pred_len: usize,
    common_context: usize,
    bounds: [i64; 3],
) -> Result<DataContract> {
    ensure!(
        context > 0 && pred_len > 0 && common_context >= context,
        "invalid history or prediction length"
    );
    ensure!(
        bounds.windows(2).all(|pair| pair[0] < pair[1]),
        "split timestamps must be strictly increasing"
    );
    let resolution_ms = i64::from(RESOLUTION) * 1000;
    let now = chrono::Utc::now().timestamp_millis();
    let mut invalid_ohlc_indices = Vec::new();
    for (index, bar) in bars.iter().enumerate() {
        ensure!(
            index == 0 || bars[index - 1].ts() < bar.ts(),
            "{ticker}: timestamps must be strictly increasing at raw bar {index}"
        );
        ensure!(
            bar.ts().rem_euclid(resolution_ms) == 0,
            "{ticker}: off-grid five-minute timestamp at raw bar {index}"
        );
        if !valid_ohlc(bar) {
            invalid_ohlc_indices.push(index);
        }
    }
    ensure!(
        bars.last().is_none_or(|bar| bar
            .ts()
            .checked_add(resolution_ms)
            .is_some_and(|completion| completion <= now)),
        "{ticker}: corpus contains an uncompleted bar"
    );
    let source_bars = bars.len();
    let valid_bars = source_bars - invalid_ohlc_indices.len();
    let raw_boundaries = bounds.map(|timestamp| bars.partition_point(|bar| bar.ts() < timestamp));
    let boundaries = raw_boundaries
        .map(|boundary| boundary - invalid_ohlc_indices.partition_point(|&index| index < boundary));
    let purge = pred_len.max(100);
    let train_end = retained_partition_end(boundaries[0], valid_bars, purge);
    let mut digest = Digest::new(&SHA256);
    digest.update(SCHEMA.as_bytes());
    digest.update(&(ticker.len() as u64).to_le_bytes());
    digest.update(ticker.as_bytes());
    digest.update(&(source_bars as u64).to_le_bytes());
    hash_bars(&mut digest, bars);
    let fingerprint = digest
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok(DataContract {
        schema: SCHEMA.into(),
        ticker: ticker.into(),
        fingerprint,
        source_bars,
        valid_bars,
        invalid_ohlc_indices,
        boundaries,
        boundary_timestamps: bounds,
        context,
        pred_len,
        purge,
        train_end,
        common_context,
    })
}

pub(super) fn retained_partition_end(boundary: usize, source_bars: usize, purge: usize) -> usize {
    if boundary == source_bars {
        source_bars
    } else {
        boundary.saturating_sub(purge)
    }
}

fn hash_bars(digest: &mut Digest, bars: &[PackedBar]) {
    #[cfg(target_endian = "little")]
    digest.update(bytemuck::cast_slice(bars));
    #[cfg(target_endian = "big")]
    for bar in bars {
        digest.update(&bar.ts().to_le_bytes());
        for value in [bar.open, bar.high, bar.low, bar.close, bar.volume, bar.vwap] {
            digest.update(&value.to_bits().to_le_bytes());
        }
        digest.update(&bar.trades.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bars() -> Vec<PackedBar> {
        (0..5000)
            .map(|i| {
                let close = 100.0 + i as f32 * 0.01;
                PackedBar {
                    ts_ms: 1_500_000_000_000 + i * 300_000,
                    open: close - 0.25,
                    high: close + 0.5,
                    low: close - 0.5,
                    close,
                    volume: 100.0,
                    vwap: close,
                    trades: 1,
                }
            })
            .collect()
    }

    #[test]
    fn bulk_fingerprint_matches_canonical_field_encoding() {
        let source = bars();
        let mut expected = Digest::new(&SHA256);
        for bar in &source {
            expected.update(&bar.ts().to_le_bytes());
            for value in [bar.open, bar.high, bar.low, bar.close, bar.volume, bar.vwap] {
                expected.update(&value.to_bits().to_le_bytes());
            }
            expected.update(&bar.trades.to_le_bytes());
        }
        let mut actual = Digest::new(&SHA256);
        hash_bars(&mut actual, &source);
        assert_eq!(actual.finish().as_ref(), expected.finish().as_ref());
    }

    #[test]
    fn filtered_metadata_quarantines_invalid_rows_and_preserves_raw_authentication() {
        let mut source = bars();
        source[42].high = source[42].low;
        source[3000].open = f32::NAN;
        let bounds = [source[3500].ts(), source[4000].ts(), source[4500].ts()];
        let contract = filtered_contract("ONE", &source, 96, 192, 128, bounds).unwrap();
        assert_eq!(contract.source_bars, 5000);
        assert_eq!(contract.valid_bars, 4998);
        assert_eq!(contract.invalid_ohlc_indices, [42, 3000]);
        assert_eq!(contract.boundaries, [3498, 3998, 4498]);
        assert_eq!(contract.boundary_timestamps, bounds);
        assert_eq!(contract.train_end, 3306);
        source[42].volume *= 2.0;
        let changed = filtered_contract("ONE", &source, 96, 192, 128, bounds).unwrap();
        assert_eq!(contract.invalid_ohlc_indices, changed.invalid_ohlc_indices);
        assert_ne!(contract.fingerprint, changed.fingerprint);
        source[42].ts_ms = source[41].ts();
        assert!(filtered_contract("ONE", &source, 96, 192, 128, bounds).is_err());
    }

    #[test]
    fn raw_source_fingerprint_is_identical_across_context_comparisons() {
        let source = bars();
        let bounds = [source[3500].ts(), source[4000].ts(), source[4500].ts()];
        let short = filtered_contract("ONE", &source, 96, 192, 2048, bounds).unwrap();
        let long = filtered_contract("ONE", &source, 2048, 192, 2048, bounds).unwrap();
        assert_eq!(short.fingerprint, long.fingerprint);
        assert_eq!(short.boundaries, long.boundaries);
    }

    #[test]
    fn all_invalid_training_metadata_is_finite_but_ineligible() {
        let mut source = bars();
        let bounds = [source[3500].ts(), source[4000].ts(), source[4500].ts()];
        for bar in &mut source[..3500] {
            bar.close = 0.0;
        }
        let contract = filtered_contract("ONE", &source, 96, 192, 128, bounds).unwrap();
        assert_eq!(contract.train_end, 0);
        assert!(contract.train_end <= contract.common_context);
    }

    #[test]
    fn delisted_ticker_keeps_its_complete_training_history() {
        let source = bars();
        let bounds = [
            source[4999].ts() + 300_000,
            source[4999].ts() + 600_000,
            source[4999].ts() + 900_000,
        ];
        let contract = filtered_contract("GONE", &source, 96, 192, 128, bounds).unwrap();
        assert_eq!(contract.boundaries, [5000; 3]);
        assert_eq!(contract.train_end, 5000);
    }
}
