use enum_map::{Enum, EnumMap};
use rand::{rngs::StdRng, Rng};
use serde::{Deserialize, Serialize};

use crate::utils::percent_diff;

use super::super::family::{DecisionContext, GeneticFamily, IndicatorConfig, StrategyFamilySpec};

#[derive(Clone, Copy, Debug, Enum, Serialize, Deserialize)]
enum Gene {
    FastEmaAlpha,
    SlowEmaAlpha,
    PullbackMinPct,
    PullbackMaxPct,
    TrendSpreadMinPct,
    TrendExitPct,
    StopLossPct,
    TakeProfitPct,
    BuyPercent,
    SellPercent,
    MinEntryRsi,
    MaxExitRsi,
}

impl Gene {
    const ALL: [Self; 12] = [
        Self::FastEmaAlpha,
        Self::SlowEmaAlpha,
        Self::PullbackMinPct,
        Self::PullbackMaxPct,
        Self::TrendSpreadMinPct,
        Self::TrendExitPct,
        Self::StopLossPct,
        Self::TakeProfitPct,
        Self::BuyPercent,
        Self::SellPercent,
        Self::MinEntryRsi,
        Self::MaxExitRsi,
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
        GeneticFamily::TrendBreakout
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
        if genome.genes[Gene::FastEmaAlpha] <= genome.genes[Gene::SlowEmaAlpha] {
            genome.genes[Gene::FastEmaAlpha] =
                (genome.genes[Gene::SlowEmaAlpha] + 0.01).min(spec(Gene::FastEmaAlpha).max);
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
            let weight = rng.random_range(0.20..0.80);
            genes[gene] = clamp(
                left.genes[gene] * weight + right.genes[gene] * (1.0 - weight),
                spec(gene),
            );
        }
        if genes[Gene::FastEmaAlpha] <= genes[Gene::SlowEmaAlpha] {
            genes[Gene::FastEmaAlpha] =
                (genes[Gene::SlowEmaAlpha] + 0.01).min(spec(Gene::FastEmaAlpha).max);
        }
        Genome { genes }
    }

    fn indicator_config(&self, genome: &Self::Genome) -> IndicatorConfig {
        IndicatorConfig {
            decider_rsi_alpha: 0.03,
            amount_rsi_alpha: 0.05,
            price_ema_alpha: genome.genes[Gene::FastEmaAlpha],
            fast_ema_alpha: Some(genome.genes[Gene::FastEmaAlpha]),
            slow_ema_alpha: Some(genome.genes[Gene::SlowEmaAlpha]),
        }
    }

    fn allow_buy(&self, genome: &Self::Genome, ctx: &DecisionContext) -> bool {
        let Some(fast_ema) = ctx.fast_ema else {
            return false;
        };
        let Some(slow_ema) = ctx.slow_ema else {
            return false;
        };
        if fast_ema <= slow_ema {
            return false;
        }
        let trend_spread = (fast_ema - slow_ema) / slow_ema.max(1e-6);
        if trend_spread < genome.genes[Gene::TrendSpreadMinPct] {
            return false;
        }
        if ctx.price <= slow_ema || ctx.price < fast_ema * 0.995 {
            return false;
        }
        let pullback = percent_diff(ctx.local_maximum, ctx.price);
        if pullback < genome.genes[Gene::PullbackMinPct]
            || pullback > genome.genes[Gene::PullbackMaxPct]
        {
            return false;
        }
        ctx.decider_rsi >= genome.genes[Gene::MinEntryRsi] * 100.0
    }

    fn allow_sell(&self, genome: &Self::Genome, ctx: &DecisionContext) -> bool {
        if ctx.position_quantity <= 0.0 {
            return false;
        }
        let pnl_pct = ctx.unrealized_pnl_pct() / 100.0;
        let Some(fast_ema) = ctx.fast_ema else {
            return false;
        };
        if pnl_pct <= -genome.genes[Gene::StopLossPct] {
            return true;
        }
        if pnl_pct >= genome.genes[Gene::TakeProfitPct] {
            return true;
        }
        if ctx.price < fast_ema * (1.0 - genome.genes[Gene::TrendExitPct]) {
            return true;
        }
        ctx.amount_rsi >= genome.genes[Gene::MaxExitRsi] * 100.0
    }

    fn buy_budget(&self, genome: &Self::Genome, ctx: &DecisionContext) -> f64 {
        let Some(fast_ema) = ctx.fast_ema else {
            return 0.0;
        };
        let Some(slow_ema) = ctx.slow_ema else {
            return 0.0;
        };
        let trend_spread = ((fast_ema - slow_ema) / slow_ema.max(1e-6)).max(0.0);
        let pullback = percent_diff(ctx.local_maximum, ctx.price);
        let trend_term = (trend_spread / genome.genes[Gene::TrendSpreadMinPct].max(1e-6)).min(2.0);
        let pullback_term =
            (pullback / genome.genes[Gene::PullbackMaxPct].max(1e-6)).clamp(0.2, 1.0);
        let fraction =
            (genome.genes[Gene::BuyPercent] * trend_term * pullback_term).clamp(0.0, 1.0);
        let equal_weight_headroom =
            (ctx.equal_weight_position_value() - ctx.position_value).max(0.0);
        ctx.cash.min(equal_weight_headroom * fraction)
    }

    fn sell_budget(&self, genome: &Self::Genome, ctx: &DecisionContext) -> f64 {
        let pnl_pct = ctx.unrealized_pnl_pct().max(0.0) / 100.0;
        let pnl_term = (pnl_pct / genome.genes[Gene::TakeProfitPct].max(1e-6)).clamp(0.5, 2.0);
        let fraction = (genome.genes[Gene::SellPercent] * pnl_term).clamp(0.0, 1.0);
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
        Gene::FastEmaAlpha => GeneSpec {
            min: 0.05,
            max: 0.3,
            init: 0.12,
            mutation: 0.02,
        },
        Gene::SlowEmaAlpha => GeneSpec {
            min: 0.01,
            max: 0.12,
            init: 0.04,
            mutation: 0.01,
        },
        Gene::PullbackMinPct => GeneSpec {
            min: 0.002,
            max: 0.04,
            init: 0.01,
            mutation: 0.004,
        },
        Gene::PullbackMaxPct => GeneSpec {
            min: 0.01,
            max: 0.15,
            init: 0.05,
            mutation: 0.01,
        },
        Gene::TrendSpreadMinPct => GeneSpec {
            min: 0.001,
            max: 0.08,
            init: 0.01,
            mutation: 0.003,
        },
        Gene::TrendExitPct => GeneSpec {
            min: 0.003,
            max: 0.08,
            init: 0.02,
            mutation: 0.004,
        },
        Gene::StopLossPct => GeneSpec {
            min: 0.005,
            max: 0.12,
            init: 0.03,
            mutation: 0.005,
        },
        Gene::TakeProfitPct => GeneSpec {
            min: 0.01,
            max: 0.25,
            init: 0.08,
            mutation: 0.01,
        },
        Gene::BuyPercent => GeneSpec {
            min: 0.03,
            max: 0.9,
            init: 0.2,
            mutation: 0.06,
        },
        Gene::SellPercent => GeneSpec {
            min: 0.15,
            max: 1.0,
            init: 0.7,
            mutation: 0.06,
        },
        Gene::MinEntryRsi => GeneSpec {
            min: 0.35,
            max: 0.75,
            init: 0.52,
            mutation: 0.03,
        },
        Gene::MaxExitRsi => GeneSpec {
            min: 0.6,
            max: 0.95,
            init: 0.78,
            mutation: 0.03,
        },
    }
}

fn jitter(value: f64, amount: f64, spec: GeneSpec, rng: &mut StdRng) -> f64 {
    clamp(value + rng.random_range(-amount..amount), spec)
}

fn clamp(value: f64, spec: GeneSpec) -> f64 {
    value.clamp(spec.min, spec.max)
}
