use enum_map::{Enum, EnumMap};
use rand::{rngs::StdRng, Rng};
use serde::{Deserialize, Serialize};

use crate::utils::{percent_diff, percent_diff_abs};

use super::super::family::{DecisionContext, GeneticFamily, IndicatorConfig, StrategyFamilySpec};

#[derive(Clone, Copy, Debug, Enum, Serialize, Deserialize)]
enum Gene {
    PriceEmaAlpha,
    ReboundSellPriceThreshold,
    ReboundBuyPriceThreshold,
    MaxReboundBuyPriceThreshold,
    DropBuyThreshold,
    SellDropBuyThreshold,
    BuyPercent,
    SellPercent,
    BuyDistanceWeightAmount,
    SellDistanceWeightAmount,
}

impl Gene {
    const ALL: [Self; 10] = [
        Self::PriceEmaAlpha,
        Self::ReboundSellPriceThreshold,
        Self::ReboundBuyPriceThreshold,
        Self::MaxReboundBuyPriceThreshold,
        Self::DropBuyThreshold,
        Self::SellDropBuyThreshold,
        Self::BuyPercent,
        Self::SellPercent,
        Self::BuyDistanceWeightAmount,
        Self::SellDistanceWeightAmount,
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
        GeneticFamily::PriceRebound
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
            decider_rsi_alpha: 0.02,
            amount_rsi_alpha: 0.02,
            price_ema_alpha: genome.genes[Gene::PriceEmaAlpha],
            fast_ema_alpha: None,
            slow_ema_alpha: None,
        }
    }

    fn allow_buy(&self, genome: &Self::Genome, ctx: &DecisionContext) -> bool {
        if percent_diff(ctx.price, ctx.local_minimum)
            <= genome.genes[Gene::ReboundBuyPriceThreshold]
        {
            return false;
        }
        if percent_diff(ctx.local_maximum, ctx.price)
            <= genome.genes[Gene::MaxReboundBuyPriceThreshold]
        {
            return false;
        }
        if let Some(last_buy) = ctx.last_buy_price {
            if ctx.price > last_buy {
                return false;
            }
            if percent_diff(last_buy, ctx.price) <= genome.genes[Gene::DropBuyThreshold] {
                return false;
            }
        }
        if let Some(last_sell) = ctx.last_sell_price {
            if percent_diff_abs(ctx.price, last_sell) <= genome.genes[Gene::SellDropBuyThreshold] {
                return false;
            }
        }
        true
    }

    fn allow_sell(&self, genome: &Self::Genome, ctx: &DecisionContext) -> bool {
        if ctx.position_quantity <= 0.0 || ctx.position_avg_price >= ctx.price {
            return false;
        }
        ctx.price * (1.0 + genome.genes[Gene::ReboundSellPriceThreshold]) < ctx.local_maximum
    }

    fn buy_budget(&self, genome: &Self::Genome, ctx: &DecisionContext) -> f64 {
        let distance = percent_diff(ctx.local_maximum, ctx.price);
        let scale = distance / genome.genes[Gene::BuyDistanceWeightAmount].max(1e-6);
        let fraction = (genome.genes[Gene::BuyPercent] * scale).clamp(0.0, 1.0);
        let equal_weight_headroom =
            (ctx.equal_weight_position_value() - ctx.position_value).max(0.0);
        ctx.cash.min(equal_weight_headroom * fraction)
    }

    fn sell_budget(&self, genome: &Self::Genome, ctx: &DecisionContext) -> f64 {
        let distance = percent_diff(ctx.price, ctx.local_minimum);
        let scale = distance / genome.genes[Gene::SellDistanceWeightAmount].max(1e-6);
        let fraction = (genome.genes[Gene::SellPercent] * scale).clamp(0.0, 1.0);
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
        Gene::PriceEmaAlpha => GeneSpec {
            min: 0.01,
            max: 0.2,
            init: 0.05,
            mutation: 0.01,
        },
        Gene::ReboundSellPriceThreshold => GeneSpec {
            min: 0.005,
            max: 0.05,
            init: 0.015,
            mutation: 0.004,
        },
        Gene::ReboundBuyPriceThreshold => GeneSpec {
            min: 0.001,
            max: 0.03,
            init: 0.005,
            mutation: 0.003,
        },
        Gene::MaxReboundBuyPriceThreshold => GeneSpec {
            min: 0.01,
            max: 0.12,
            init: 0.04,
            mutation: 0.01,
        },
        Gene::DropBuyThreshold => GeneSpec {
            min: 0.0005,
            max: 0.03,
            init: 0.002,
            mutation: 0.002,
        },
        Gene::SellDropBuyThreshold => GeneSpec {
            min: 0.005,
            max: 0.08,
            init: 0.03,
            mutation: 0.006,
        },
        Gene::BuyPercent => GeneSpec {
            min: 0.02,
            max: 0.75,
            init: 0.12,
            mutation: 0.05,
        },
        Gene::SellPercent => GeneSpec {
            min: 0.1,
            max: 1.0,
            init: 0.8,
            mutation: 0.08,
        },
        Gene::BuyDistanceWeightAmount => GeneSpec {
            min: 0.01,
            max: 0.2,
            init: 0.05,
            mutation: 0.01,
        },
        Gene::SellDistanceWeightAmount => GeneSpec {
            min: 0.01,
            max: 0.2,
            init: 0.05,
            mutation: 0.01,
        },
    }
}

fn jitter(value: f64, amount: f64, spec: GeneSpec, rng: &mut StdRng) -> f64 {
    clamp(value + rng.random_range(-amount..amount), spec)
}

fn clamp(value: f64, spec: GeneSpec) -> f64 {
    value.clamp(spec.min, spec.max)
}
