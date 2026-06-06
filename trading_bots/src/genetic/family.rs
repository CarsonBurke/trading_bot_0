use rand::rngs::StdRng;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum GeneticFamily {
    PriceRebound,
    PriceReboundCashBreadth,
    PriceReboundCashLeaderGap,
    PriceReboundCashWeakRegime,
    RsiRebound,
    TrendBreakout,
}

#[derive(Clone, Copy, Debug)]
pub struct IndicatorConfig {
    pub decider_rsi_alpha: f64,
    pub amount_rsi_alpha: f64,
    pub price_ema_alpha: f64,
    pub fast_ema_alpha: Option<f64>,
    pub slow_ema_alpha: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct DecisionContext {
    pub index: usize,
    pub ticker_count: usize,
    pub ticker_idx: usize,
    pub price: f64,
    pub decider_rsi: f64,
    pub amount_rsi: f64,
    pub price_ema: f64,
    pub fast_ema: Option<f64>,
    pub slow_ema: Option<f64>,
    pub local_minimum: f64,
    pub local_maximum: f64,
    pub last_buy_price: Option<f64>,
    pub last_sell_price: Option<f64>,
    pub lowest_rsi_since_flat: Option<f64>,
    pub highest_rsi_since_long: Option<f64>,
    pub position_value: f64,
    pub position_avg_price: f64,
    pub position_quantity: f64,
    pub bars_since_entry: Option<usize>,
    pub benchmark_return_since_entry_pct: Option<f64>,
    pub excess_return_since_entry_pct: Option<f64>,
    pub excess_return_peak_pct: Option<f64>,
    pub excess_return_delta_pct: Option<f64>,
    pub underperformance_streak: Option<usize>,
    pub assets: f64,
    pub cash: f64,
}

impl DecisionContext {
    pub fn unrealized_pnl_pct(self) -> f64 {
        if self.position_quantity <= 0.0 || self.position_avg_price <= 0.0 {
            return 0.0;
        }
        ((self.price / self.position_avg_price) - 1.0) * 100.0
    }

    pub fn equal_weight_position_value(self) -> f64 {
        self.assets / self.ticker_count.max(1) as f64
    }

    pub fn benchmark_return_since_entry_pct(self) -> f64 {
        self.benchmark_return_since_entry_pct.unwrap_or(0.0)
    }

    pub fn excess_return_since_entry_pct(self) -> f64 {
        self.excess_return_since_entry_pct
            .unwrap_or_else(|| self.unrealized_pnl_pct())
    }

    pub fn excess_return_peak_pct(self) -> f64 {
        self.excess_return_peak_pct
            .unwrap_or_else(|| self.excess_return_since_entry_pct())
    }

    pub fn excess_return_delta_pct(self) -> f64 {
        self.excess_return_delta_pct.unwrap_or(0.0)
    }

    pub fn underperformance_streak(self) -> usize {
        self.underperformance_streak.unwrap_or(0)
    }

    pub fn bars_since_entry(self) -> usize {
        self.bars_since_entry.unwrap_or(0)
    }

    pub fn current_weight(self) -> f64 {
        if self.assets <= f64::EPSILON {
            0.0
        } else {
            self.position_value / self.assets
        }
    }

    pub fn cash_weight(self) -> f64 {
        if self.assets <= f64::EPSILON {
            0.0
        } else {
            self.cash / self.assets
        }
    }
}

pub trait StrategyFamilySpec: Sync {
    type Genome: Clone + Send + Sync + Serialize + DeserializeOwned + std::fmt::Debug;

    fn kind(&self) -> GeneticFamily;
    fn seed_genome(&self, rng: &mut StdRng) -> Self::Genome;
    fn mutate(&self, genome: &mut Self::Genome, rng: &mut StdRng, entropy: f64);
    fn crossover(
        &self,
        left: &Self::Genome,
        right: &Self::Genome,
        rng: &mut StdRng,
    ) -> Self::Genome;
    fn indicator_config(&self, genome: &Self::Genome) -> IndicatorConfig;
    fn asset_desirability(&self, genome: &Self::Genome, ctx: &DecisionContext) -> f64;
    fn cash_desirability(&self, _genome: &Self::Genome, _contexts: &[DecisionContext]) -> f64 {
        1.0
    }
    fn min_target_weight(&self, _genome: &Self::Genome, _ctx: &DecisionContext) -> f64 {
        0.0
    }
    fn max_target_weight(&self, _genome: &Self::Genome, _ctx: &DecisionContext) -> f64 {
        1.0
    }

    fn describe(&self, genome: &Self::Genome) -> String {
        serde_json::to_string_pretty(genome).unwrap_or_else(|_| format!("{genome:?}"))
    }
}
