use enum_map::{Enum, EnumMap};
use rand::{rngs::StdRng, Rng};
use serde::{Deserialize, Serialize};

use super::super::family::{DecisionContext, GeneticFamily, IndicatorConfig, StrategyFamilySpec};

#[derive(Clone, Copy, Debug, Enum, Serialize, Deserialize)]
enum Gene {
    MinRsiSell,
    MaxRsiBuy,
    ReboundSellThreshold,
    ReboundBuyThreshold,
    DeciderRsiAlpha,
    AmountRsiAlpha,
    BuyPercent,
    SellPercent,
}

impl Gene {
    const ALL: [Self; 8] = [
        Self::MinRsiSell,
        Self::MaxRsiBuy,
        Self::ReboundSellThreshold,
        Self::ReboundBuyThreshold,
        Self::DeciderRsiAlpha,
        Self::AmountRsiAlpha,
        Self::BuyPercent,
        Self::SellPercent,
    ];
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Genome {
    genes: EnumMap<Gene, f64>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Family;

impl StrategyFamilySpec for Family {
    type Genome = Genome;

    fn kind(&self) -> GeneticFamily {
        GeneticFamily::RsiRebound
    }

    fn seed_genome(&self, rng: &mut StdRng) -> Self::Genome {
        let mut genes = EnumMap::default();
        for gene in Gene::ALL {
            genes[gene] = jitter(spec(gene).init, spec(gene).mutation, spec(gene), rng);
        }
        Genome { genes }
    }

    fn mutate(&self, genome: &mut Self::Genome, rng: &mut StdRng, entropy: f64) {
        for gene in Gene::ALL {
            genome.genes[gene] = jitter(
                genome.genes[gene],
                spec(gene).mutation * entropy.max(0.0),
                spec(gene),
                rng,
            );
        }
    }

    fn crossover(
        &self,
        left: &Self::Genome,
        right: &Self::Genome,
        rng: &mut StdRng,
    ) -> Self::Genome {
        let mut genes = EnumMap::default();
        for gene in Gene::ALL {
            let weight = rng.random_range(0.25..0.75);
            genes[gene] = clamp(
                left.genes[gene] * weight + right.genes[gene] * (1.0 - weight),
                spec(gene),
            );
        }
        Genome { genes }
    }

    fn indicator_config(&self, genome: &Self::Genome) -> IndicatorConfig {
        IndicatorConfig {
            decider_rsi_alpha: genome.genes[Gene::DeciderRsiAlpha],
            amount_rsi_alpha: genome.genes[Gene::AmountRsiAlpha],
            price_ema_alpha: 0.05,
            fast_ema_alpha: None,
            slow_ema_alpha: None,
        }
    }

    fn allow_buy(&self, genome: &Self::Genome, ctx: &DecisionContext) -> bool {
        if ctx.decider_rsi > genome.genes[Gene::MaxRsiBuy] * 100.0 {
            return false;
        }
        let Some(lowest) = ctx.lowest_rsi_since_flat else {
            return false;
        };
        ctx.decider_rsi >= lowest * (1.0 + genome.genes[Gene::ReboundBuyThreshold].max(0.0))
    }

    fn allow_sell(&self, genome: &Self::Genome, ctx: &DecisionContext) -> bool {
        if ctx.position_quantity <= 0.0 || ctx.position_avg_price >= ctx.price {
            return false;
        }
        if ctx.decider_rsi < genome.genes[Gene::MinRsiSell] * 100.0 {
            return false;
        }
        let Some(highest) = ctx.highest_rsi_since_long else {
            return false;
        };
        ctx.decider_rsi <= highest * (1.0 - genome.genes[Gene::ReboundSellThreshold].max(0.0))
    }

    fn buy_budget(&self, genome: &Self::Genome, ctx: &DecisionContext) -> f64 {
        let oversold = ((50.0 - ctx.amount_rsi).max(0.0) / 50.0).clamp(0.0, 1.0);
        let fraction = (genome.genes[Gene::BuyPercent] * (0.5 + oversold)).clamp(0.0, 1.0);
        let equal_weight_headroom =
            (ctx.equal_weight_position_value() - ctx.position_value).max(0.0);
        ctx.cash.min(equal_weight_headroom * fraction)
    }

    fn sell_budget(&self, genome: &Self::Genome, ctx: &DecisionContext) -> f64 {
        let overbought = ((ctx.amount_rsi - 50.0).max(0.0) / 50.0).clamp(0.0, 1.0);
        let fraction = (genome.genes[Gene::SellPercent] * (0.5 + overbought)).clamp(0.0, 1.0);
        ctx.position_value * fraction
    }
}

#[derive(Clone, Copy)]
struct GeneSpec {
    min: f64,
    max: f64,
    init: f64,
    mutation: f64,
}

fn spec(gene: Gene) -> GeneSpec {
    match gene {
        Gene::MinRsiSell => GeneSpec {
            min: 0.55,
            max: 0.9,
            init: 0.68,
            mutation: 0.03,
        },
        Gene::MaxRsiBuy => GeneSpec {
            min: 0.1,
            max: 0.45,
            init: 0.32,
            mutation: 0.03,
        },
        Gene::ReboundSellThreshold => GeneSpec {
            min: 0.01,
            max: 0.15,
            init: 0.04,
            mutation: 0.01,
        },
        Gene::ReboundBuyThreshold => GeneSpec {
            min: 0.01,
            max: 0.15,
            init: 0.05,
            mutation: 0.01,
        },
        Gene::DeciderRsiAlpha => GeneSpec {
            min: 0.01,
            max: 0.2,
            init: 0.02,
            mutation: 0.01,
        },
        Gene::AmountRsiAlpha => GeneSpec {
            min: 0.01,
            max: 0.2,
            init: 0.02,
            mutation: 0.01,
        },
        Gene::BuyPercent => GeneSpec {
            min: 0.03,
            max: 0.8,
            init: 0.18,
            mutation: 0.06,
        },
        Gene::SellPercent => GeneSpec {
            min: 0.1,
            max: 1.0,
            init: 0.7,
            mutation: 0.08,
        },
    }
}

fn jitter(value: f64, amount: f64, spec: GeneSpec, rng: &mut StdRng) -> f64 {
    clamp(value + rng.random_range(-amount..amount), spec)
}

fn clamp(value: f64, spec: GeneSpec) -> f64 {
    value.clamp(spec.min, spec.max)
}
