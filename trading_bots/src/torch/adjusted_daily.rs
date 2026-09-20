//! Point-in-time adjusted-daily context shared by screening and direct-return modelling.
//!
//! Deep-daily records are stamped at the regular-session open. A row becomes eligible only
//! after a full regular session has elapsed from that stamp. This deliberately treats half days
//! conservatively: context can arrive late, but an in-progress daily bar can never arrive early.

use std::path::Path;

use shared::bars::{bar_file_path, BarFile, PackedBar};

pub const ADJUSTED_DAILY_RES_SECS: u32 = 86_400;
pub const ADJUSTED_DAILY_CONTEXT_FEATURES: usize = 6;
pub const ADJUSTED_DAILY_CONTEXT_CONTRACT: &str =
    "last_fully_completed_adjusted_daily:log_close_over_prev,log_close_over_up_to_5_sessions,log_high_low,log_dollar_volume,age_days,availability_mask;eligible_at_open_plus_6h30-v1";

const REGULAR_SESSION_MILLIS: i64 = 6 * 3_600_000 + 30 * 60_000;
const MILLIS_PER_DAY: i64 = 86_400_000;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdjustedDailyFeatures {
    pub values: [f64; ADJUSTED_DAILY_CONTEXT_FEATURES],
}

impl AdjustedDailyFeatures {
    pub const MISSING: Self = Self {
        values: [0.0; ADJUSTED_DAILY_CONTEXT_FEATURES],
    };

    pub fn is_available(self) -> bool {
        self.values[ADJUSTED_DAILY_CONTEXT_FEATURES - 1] == 1.0
    }
}

/// Daily histories mmap'd once and indexed in the same symbol order as their owning corpus.
/// Missing files remain explicit zero-plus-mask rows; they never trigger a fallback to another
/// symbol, another timestamp, or reconstructed intraday data.
pub struct AdjustedDailyContextStore {
    series: Vec<Option<BarFile>>,
    initialization_error: Option<String>,
}
pub struct AdjustedDailyCursor<'a> {
    bars: &'a [PackedBar],
    end: usize,
    last_decision_ms: Option<i64>,
}

impl AdjustedDailyCursor<'_> {
    /// Amortized `O(1)` lookup for a nondecreasing timestamp stream. A backwards timestamp
    /// deterministically falls back to binary search rather than reusing later state.
    pub fn features(&mut self, decision_ms: i64) -> AdjustedDailyFeatures {
        let completed_before = decision_ms.saturating_sub(REGULAR_SESSION_MILLIS);
        if self
            .last_decision_ms
            .is_some_and(|previous| decision_ms >= previous)
        {
            while self
                .bars
                .get(self.end)
                .is_some_and(|bar| bar.ts() <= completed_before)
            {
                self.end += 1;
            }
        } else {
            self.end = self
                .bars
                .partition_point(|bar| bar.ts() <= completed_before);
        }
        self.last_decision_ms = Some(decision_ms);
        features_at_end(self.bars, self.end, decision_ms)
    }
}

impl AdjustedDailyContextStore {
    pub fn open(dir: &Path, symbols: &[String]) -> Self {
        let mut series = Vec::with_capacity(symbols.len());
        let mut failures = Vec::new();
        for symbol in symbols {
            let path = bar_file_path(dir, symbol, ADJUSTED_DAILY_RES_SECS);
            match BarFile::open(&path) {
                Ok(file) => series.push(Some(file)),
                Err(error) => {
                    series.push(None);
                    failures.push(format!("{}: {error}", path.display()));
                }
            }
        }
        let initialization_error = (failures.len() == symbols.len()).then(|| {
            format!(
                "adjusted daily files unavailable for all {} selected names (first error: {})",
                symbols.len(),
                failures.first().map(String::as_str).unwrap_or("none")
            )
        });
        Self {
            series,
            initialization_error,
        }
    }

    pub fn initialization_error(&self) -> Option<&str> {
        self.initialization_error.as_deref()
    }

    pub fn file_coverage(&self) -> (usize, usize) {
        let available = self.series.iter().filter(|series| series.is_some()).count();
        (available, self.series.len() - available)
    }

    pub fn file_identity(&self, symbol: usize) -> Option<(usize, i64, i64)> {
        let file = self.series.get(symbol)?.as_ref()?;
        Some((
            file.len(),
            file.first_ts_ms().unwrap_or(0),
            file.last_ts_ms().unwrap_or(0),
        ))
    }

    /// Features at `decision_ms`, using only daily bars whose regular session is fully complete.
    /// The result is allocation-free and each query is `O(log daily_history)`.
    pub fn features(&self, symbol: usize, decision_ms: i64) -> AdjustedDailyFeatures {
        let Some(Some(file)) = self.series.get(symbol) else {
            return AdjustedDailyFeatures::MISSING;
        };
        features_from_bars(file.bars(), decision_ms)
    }

    pub fn cursor(&self, symbol: usize) -> AdjustedDailyCursor<'_> {
        let bars = match self.series.get(symbol).and_then(Option::as_ref) {
            Some(file) => file.bars(),
            None => &[],
        };
        AdjustedDailyCursor {
            bars,
            end: 0,
            last_decision_ms: None,
        }
    }
}

fn features_from_bars(bars: &[PackedBar], decision_ms: i64) -> AdjustedDailyFeatures {
    let completed_before = decision_ms.saturating_sub(REGULAR_SESSION_MILLIS);
    let end = bars.partition_point(|bar| bar.ts() <= completed_before);
    features_at_end(bars, end, decision_ms)
}

fn features_at_end(bars: &[PackedBar], end: usize, decision_ms: i64) -> AdjustedDailyFeatures {
    if end < 2 {
        return AdjustedDailyFeatures::MISSING;
    }
    let current = bars[end - 1];
    let previous = bars[end - 2];
    let anchor = bars[end.saturating_sub(6)];
    let (Some(close), Some(previous_close), Some(anchor_close)) = (
        positive(current.close),
        positive(previous.close),
        positive(anchor.close),
    ) else {
        return AdjustedDailyFeatures::MISSING;
    };
    let range = if current.high.is_finite() && current.low.is_finite() && current.low > 0.0 {
        (f64::from(current.high) / f64::from(current.low)).ln()
    } else {
        0.0
    };
    let dollars = dollar_volume(current);
    AdjustedDailyFeatures {
        values: [
            (close / previous_close).ln(),
            (close / anchor_close).ln(),
            range,
            dollars.max(1.0).ln(),
            (decision_ms - current.ts()).max(0) as f64 / MILLIS_PER_DAY as f64,
            1.0,
        ],
    }
}

fn positive(value: f32) -> Option<f64> {
    (value.is_finite() && value > 0.0).then_some(f64::from(value))
}

fn dollar_volume(bar: PackedBar) -> f64 {
    let price = if bar.vwap.is_finite() && bar.vwap > 0.0 {
        f64::from(bar.vwap)
    } else {
        positive(bar.close).unwrap_or(0.0)
    };
    let volume = if bar.volume.is_finite() && bar.volume > 0.0 {
        f64::from(bar.volume)
    } else {
        0.0
    };
    price * volume
}

#[cfg(test)]
mod tests {
    use super::*;

    fn daily(ts_ms: i64, close: f32) -> PackedBar {
        PackedBar {
            ts_ms,
            open: close - 0.5,
            high: close + 1.0,
            low: close - 1.0,
            close,
            volume: 1_000.0,
            vwap: close,
            trades: 100,
        }
    }

    #[test]
    fn intraday_decision_cannot_see_its_in_progress_daily_bar() {
        let day = MILLIS_PER_DAY;
        let bars = vec![daily(0, 100.0), daily(day, 110.0), daily(2 * day, 121.0)];
        let intraday = features_from_bars(&bars, 2 * day + 3 * 3_600_000);
        assert!(intraday.is_available());
        assert!((intraday.values[0] - (110.0f64 / 100.0).ln()).abs() < 1.0e-12);

        let completed = features_from_bars(&bars, 2 * day + REGULAR_SESSION_MILLIS);
        assert!(completed.is_available());
        let expected = [
            (121.0f64 / 110.0).ln(),
            (121.0f64 / 100.0).ln(),
            (122.0f64 / 120.0).ln(),
            (121.0f64 * 1_000.0).ln(),
            REGULAR_SESSION_MILLIS as f64 / MILLIS_PER_DAY as f64,
            1.0,
        ];
        let mut cursor = AdjustedDailyCursor {
            bars: &bars,
            end: 0,
            last_decision_ms: None,
        };
        for decision in [
            day + REGULAR_SESSION_MILLIS,
            2 * day + 3 * 3_600_000,
            2 * day + REGULAR_SESSION_MILLIS,
            day + REGULAR_SESSION_MILLIS,
        ] {
            assert_eq!(
                cursor.features(decision),
                features_from_bars(&bars, decision),
                "cursor lookup drifted at {decision}"
            );
        }
        for (actual, expected) in completed.values.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn unavailable_symbol_and_insufficient_history_are_neutral_and_masked() {
        let store = AdjustedDailyContextStore {
            series: vec![None],
            initialization_error: None,
        };
        assert_eq!(store.features(0, i64::MAX), AdjustedDailyFeatures::MISSING);
        assert_eq!(
            features_from_bars(&[daily(0, 100.0)], i64::MAX),
            AdjustedDailyFeatures::MISSING
        );
        assert!(!AdjustedDailyFeatures::MISSING.is_available());
    }
}
