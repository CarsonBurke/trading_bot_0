use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

struct MomentumCacheEntry {
    price_bits: Vec<u64>,
    indicators: Arc<MomentumIndicators>,
}

/// Global cache for momentum indicators (ticker -> latest aligned grid)
static MOMENTUM_CACHE: OnceLock<Mutex<HashMap<String, MomentumCacheEntry>>> = OnceLock::new();

fn get_cache() -> &'static Mutex<HashMap<String, MomentumCacheEntry>> {
    MOMENTUM_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn same_price_grid(price_bits: &[u64], prices: &[f64]) -> bool {
    price_bits.len() == prices.len()
        && price_bits
            .iter()
            .zip(prices)
            .all(|(cached, price)| *cached == price.to_bits())
}

/// Precomputed momentum indicators
pub struct MomentumIndicators {
    pub rsi: Vec<f64>,
    pub mom_5: Vec<f64>,
    pub mom_60: Vec<f64>,
    pub mom_120: Vec<f64>,
    pub mom_accel: Vec<f64>,
    pub vol_adj_mom: Vec<f64>,
    pub range_pos: Vec<f64>,
    pub zscore: Vec<f64>,
    pub efficiency: Vec<f64>,
    pub macd: Vec<f64>,
    pub stoch_k: Vec<f64>,
    pub trend_strength: Vec<f64>,
}

impl MomentumIndicators {
    /// Get cached momentum indicators or compute if not present
    pub fn get_or_compute(ticker: &str, prices: &[f64]) -> Arc<MomentumIndicators> {
        let cache = get_cache();
        {
            let locked = cache.lock().unwrap();
            if let Some(cached) = locked.get(ticker) {
                if same_price_grid(&cached.price_bits, prices) {
                    return cached.indicators.clone();
                }
            }
        }
        let computed = Arc::new(Self::compute(prices));
        cache.lock().unwrap().insert(
            ticker.to_string(),
            MomentumCacheEntry {
                price_bits: prices.iter().map(|price| price.to_bits()).collect(),
                indicators: computed.clone(),
            },
        );
        computed
    }

    pub fn compute(prices: &[f64]) -> Self {
        let n = prices.len();
        let mut rsi = vec![0.5; n];
        let mut mom_5 = vec![0.0; n];
        let mut mom_60 = vec![0.0; n];
        let mut mom_120 = vec![0.0; n];
        let mut mom_accel = vec![0.0; n];
        let mut vol_adj_mom = vec![0.0; n];
        let mut range_pos = vec![0.0; n];
        let mut zscore = vec![0.0; n];
        let mut efficiency = vec![0.0; n];
        let mut macd = vec![0.0; n];
        let mut stoch_k = vec![0.5; n];
        let mut trend_strength = vec![0.0; n];

        let mut ema_12 = prices.first().copied().unwrap_or(1.0);
        let mut ema_26 = ema_12;

        for i in 1..n {
            let p = prices[i];

            if i >= 5 {
                mom_5[i] = (p / prices[i - 5] - 1.0).clamp(-0.5, 0.5);
            }
            if i >= 60 {
                mom_60[i] = (p / prices[i - 60] - 1.0).clamp(-1.0, 1.0);
            }
            if i >= 120 {
                mom_120[i] = (p / prices[i - 120] - 1.0).clamp(-2.0, 2.0);
            }
            if i >= 10 {
                mom_accel[i] = (mom_5[i] - mom_5[i - 5]).clamp(-0.2, 0.2);
            }

            if i >= 14 {
                let (mut gains, mut losses) = (0.0, 0.0);
                for j in (i - 13)..=i {
                    let chg = prices[j] - prices[j - 1];
                    if chg > 0.0 {
                        gains += chg;
                    } else {
                        losses -= chg;
                    }
                }
                rsi[i] = (100.0 - 100.0 / (1.0 + gains / losses.max(1e-10))) / 100.0;

                let (mut up, mut down) = (0, 0);
                for j in (i - 13)..=i {
                    if prices[j] > prices[j - 1] {
                        up += 1;
                    } else if prices[j] < prices[j - 1] {
                        down += 1;
                    }
                }
                trend_strength[i] = up.max(down) as f64 / 14.0;
            }

            if i >= 20 {
                let w = &prices[i - 19..=i];
                let high = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let low = w.iter().cloned().fold(f64::INFINITY, f64::min);
                let mean: f64 = w.iter().sum::<f64>() / 20.0;
                let std = (w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 20.0)
                    .sqrt()
                    .max(1e-10);
                let range = (high - low).max(1e-10);

                range_pos[i] = ((p - low) / range * 2.0 - 1.0).clamp(-1.0, 1.0);
                stoch_k[i] = ((p - low) / range).clamp(0.0, 1.0);
                zscore[i] = ((p - mean) / std).clamp(-3.0, 3.0);
                vol_adj_mom[i] = ((p / prices[i - 20] - 1.0) / (std / mean)).clamp(-5.0, 5.0);

                let net = (p - prices[i - 20]).abs();
                let total: f64 = (i - 19..=i)
                    .map(|j| (prices[j] - prices[j - 1]).abs())
                    .sum();
                efficiency[i] = (net / total.max(1e-10)).clamp(0.0, 1.0);
            }

            ema_12 = 2.0 / 13.0 * p + (1.0 - 2.0 / 13.0) * ema_12;
            ema_26 = 2.0 / 27.0 * p + (1.0 - 2.0 / 27.0) * ema_26;
            macd[i] = ((ema_12 - ema_26) / p.max(1e-10) * 100.0).clamp(-5.0, 5.0);
        }

        Self {
            rsi,
            mom_5,
            mom_60,
            mom_120,
            mom_accel,
            vol_adj_mom,
            range_pos,
            zscore,
            efficiency,
            macd,
            stoch_k,
            trend_strength,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::MomentumIndicators;

    #[test]
    fn cache_recomputes_same_length_ticker_on_a_different_price_grid() {
        let rising = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let falling = [15.0, 14.0, 13.0, 12.0, 11.0, 10.0];
        let first = MomentumIndicators::get_or_compute("MOMENTUM_GRID_TEST", &rising);
        let second = MomentumIndicators::get_or_compute("MOMENTUM_GRID_TEST", &falling);

        assert!(!Arc::ptr_eq(&first, &second));
        assert!(first.mom_5[5] > 0.0);
        assert!(second.mom_5[5] < 0.0);
    }
}
