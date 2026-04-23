use rand::rngs::StdRng;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum GeneticFamily {
    PriceRebound,
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
}

pub trait StrategyFamilySpec: Sync {
    type Genome: Clone + Send + Sync + Serialize + DeserializeOwned + std::fmt::Debug;

    fn kind(&self) -> GeneticFamily;
    fn seed_genome(&self, rng: &mut StdRng) -> Self::Genome;
    fn mutate(&self, genome: &mut Self::Genome, rng: &mut StdRng);
    fn crossover(
        &self,
        left: &Self::Genome,
        right: &Self::Genome,
        rng: &mut StdRng,
    ) -> Self::Genome;
    fn indicator_config(&self, genome: &Self::Genome) -> IndicatorConfig;
    fn allow_buy(&self, genome: &Self::Genome, ctx: &DecisionContext) -> bool;
    fn allow_sell(&self, genome: &Self::Genome, ctx: &DecisionContext) -> bool;
    fn buy_fraction(&self, genome: &Self::Genome, ctx: &DecisionContext) -> f64;
    fn sell_fraction(&self, genome: &Self::Genome, ctx: &DecisionContext) -> f64;

    fn describe(&self, genome: &Self::Genome) -> String {
        serde_json::to_string_pretty(genome).unwrap_or_else(|_| format!("{genome:?}"))
    }
}
