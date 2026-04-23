use enum_map::{Enum, EnumMap};
use rand::{rngs::StdRng, Rng};
use serde::{Deserialize, Serialize};

use crate::utils::{percent_diff, percent_diff_abs};

use super::super::family::{DecisionContext, GeneticFamily, IndicatorConfig, StrategyFamilySpec};

#[derive(Clone, Copy, Debug, Enum, Serialize, Deserialize)]
enum Gene {
    PriceEmaAlpha,
    FastEmaAlpha,
    SlowEmaAlpha,
    ReboundSellPriceThreshold,
    ReboundBuyPriceThreshold,
    MaxReboundBuyPriceThreshold,
    DropBuyThreshold,
    SellDropBuyThreshold,
    StopLossPct,
    TakeProfitPct,
    MaxRsiBuy,
    MinRsiSell,
    BuyPercent,
    SellPercent,
    BuyDistanceWeightAmount,
    SellDistanceWeightAmount,
    BuyScoreThreshold,
    SellScoreThreshold,
    BuyReboundWeight,
    BuyEmaWeight,
    BuyRsiWeight,
    BuyDropWeight,
    BuyTrendWeight,
    BuyRsiRecoveryWeight,
    SellTrailWeight,
    SellEmaWeight,
    SellRsiWeight,
    SellPnlWeight,
    SellTrendWeight,
    SellRsiFadeWeight,
    SellAdverseWeight,
    NeutralBand,
    MaxTargetWeight,
    FreshEntryScale,
    HoldWeight,
    HoldFloorFraction,
    CorePositionFraction,
}

impl Gene {
    const ALL: [Self; 37] = [
        Self::PriceEmaAlpha,
        Self::FastEmaAlpha,
        Self::SlowEmaAlpha,
        Self::ReboundSellPriceThreshold,
        Self::ReboundBuyPriceThreshold,
        Self::MaxReboundBuyPriceThreshold,
        Self::DropBuyThreshold,
        Self::SellDropBuyThreshold,
        Self::StopLossPct,
        Self::TakeProfitPct,
        Self::MaxRsiBuy,
        Self::MinRsiSell,
        Self::BuyPercent,
        Self::SellPercent,
        Self::BuyDistanceWeightAmount,
        Self::SellDistanceWeightAmount,
        Self::BuyScoreThreshold,
        Self::SellScoreThreshold,
        Self::BuyReboundWeight,
        Self::BuyEmaWeight,
        Self::BuyRsiWeight,
        Self::BuyDropWeight,
        Self::BuyTrendWeight,
        Self::BuyRsiRecoveryWeight,
        Self::SellTrailWeight,
        Self::SellEmaWeight,
        Self::SellRsiWeight,
        Self::SellPnlWeight,
        Self::SellTrendWeight,
        Self::SellRsiFadeWeight,
        Self::SellAdverseWeight,
        Self::NeutralBand,
        Self::MaxTargetWeight,
        Self::FreshEntryScale,
        Self::HoldWeight,
        Self::HoldFloorFraction,
        Self::CorePositionFraction,
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
            fast_ema_alpha: Some(genome.genes[Gene::FastEmaAlpha]),
            slow_ema_alpha: Some(genome.genes[Gene::SlowEmaAlpha]),
        }
    }

    fn asset_desirability(&self, genome: &Self::Genome, ctx: &DecisionContext) -> f64 {
        let pnl_pct = ctx.unrealized_pnl_pct() / 100.0;
        if pnl_pct <= -genome.genes[Gene::StopLossPct] {
            return 0.0;
        }

        let active_buy = active_buy_score(genome, ctx);
        let active_sell = active_sell_score(genome, ctx);
        let buy_edge = directional_edge(active_buy, active_sell, genome.genes[Gene::NeutralBand]);
        let sell_edge = directional_edge(active_sell, active_buy, genome.genes[Gene::NeutralBand]);
        let hold_signal = hold_signal(genome, ctx, sell_edge, buy_edge);
        let hold_term = hold_signal * genome.genes[Gene::HoldWeight];
        let reentry_penalty = reentry_penalty(genome, ctx);

        let over_target_penalty =
            (ctx.current_weight() / self.max_target_weight(genome, ctx).max(1e-6)).clamp(0.0, 1.0);
        let take_profit_pressure = if pnl_pct > 0.0 {
            normalized_signal(
                pnl_pct,
                genome.genes[Gene::TakeProfitPct] * 0.5,
                genome.genes[Gene::TakeProfitPct],
            ) * active_sell.min(1.0)
                * 0.4
        } else {
            0.0
        };
        let desirability = hold_term
            + buy_edge * (1.0 - reentry_penalty) * genome.genes[Gene::BuyPercent] * 1.15
            - sell_edge * genome.genes[Gene::SellPercent] * 0.65
            - over_target_penalty * 0.25
            - take_profit_pressure;
        desirability.max(0.0)
    }

    fn cash_desirability(&self, genome: &Self::Genome) -> f64 {
        0.05 + genome.genes[Gene::CorePositionFraction] * 0.2
    }

    fn min_target_weight(&self, genome: &Self::Genome, ctx: &DecisionContext) -> f64 {
        if ctx.position_quantity <= 0.0 {
            return 0.0;
        }
        if (ctx.unrealized_pnl_pct() / 100.0) <= -genome.genes[Gene::StopLossPct] {
            return 0.0;
        }
        let active_buy = active_buy_score(genome, ctx);
        let active_sell = active_sell_score(genome, ctx);
        let sell_edge = directional_edge(active_sell, active_buy, genome.genes[Gene::NeutralBand]);
        let floor_fraction = genome.genes[Gene::HoldFloorFraction]
            * (1.0 - (sell_edge / 2.0).clamp(0.0, 1.0))
            * (hold_signal(genome, ctx, sell_edge, active_buy) / 2.0).clamp(0.0, 1.0);
        ctx.current_weight() * floor_fraction
    }

    fn max_target_weight(&self, genome: &Self::Genome, ctx: &DecisionContext) -> f64 {
        let cap = genome.genes[Gene::MaxTargetWeight];
        if ctx.position_quantity > 0.0 {
            return cap;
        }
        let conviction = active_buy_score(genome, ctx).clamp(0.0, 1.0);
        let entry_scale = genome.genes[Gene::FreshEntryScale];
        cap * (entry_scale + (1.0 - entry_scale) * conviction)
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
        Gene::FastEmaAlpha => GeneSpec {
            min: 0.09,
            max: 0.35,
            init: 0.16,
            mutation: 0.02,
        },
        Gene::SlowEmaAlpha => GeneSpec {
            min: 0.005,
            max: 0.06,
            init: 0.028,
            mutation: 0.006,
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
        Gene::StopLossPct => GeneSpec {
            min: 0.01,
            max: 0.25,
            init: 0.08,
            mutation: 0.01,
        },
        Gene::TakeProfitPct => GeneSpec {
            min: 0.02,
            max: 0.4,
            init: 0.14,
            mutation: 0.015,
        },
        Gene::MaxRsiBuy => GeneSpec {
            min: 0.15,
            max: 0.5,
            init: 0.34,
            mutation: 0.03,
        },
        Gene::MinRsiSell => GeneSpec {
            min: 0.5,
            max: 0.9,
            init: 0.68,
            mutation: 0.03,
        },
        Gene::BuyPercent => GeneSpec {
            min: 0.05,
            max: 1.0,
            init: 0.22,
            mutation: 0.07,
        },
        Gene::SellPercent => GeneSpec {
            min: 0.05,
            max: 1.0,
            init: 0.45,
            mutation: 0.07,
        },
        Gene::BuyDistanceWeightAmount => GeneSpec {
            min: 0.01,
            max: 0.18,
            init: 0.045,
            mutation: 0.008,
        },
        Gene::SellDistanceWeightAmount => GeneSpec {
            min: 0.01,
            max: 0.18,
            init: 0.045,
            mutation: 0.008,
        },
        Gene::BuyScoreThreshold => GeneSpec {
            min: 0.35,
            max: 0.9,
            init: 0.6,
            mutation: 0.035,
        },
        Gene::SellScoreThreshold => GeneSpec {
            min: 0.3,
            max: 0.85,
            init: 0.52,
            mutation: 0.035,
        },
        Gene::BuyReboundWeight => weight_spec(1.2),
        Gene::BuyEmaWeight => weight_spec(0.9),
        Gene::BuyRsiWeight => weight_spec(1.0),
        Gene::BuyDropWeight => weight_spec(0.8),
        Gene::BuyTrendWeight => weight_spec(1.1),
        Gene::BuyRsiRecoveryWeight => weight_spec(0.8),
        Gene::SellTrailWeight => weight_spec(1.2),
        Gene::SellEmaWeight => weight_spec(0.9),
        Gene::SellRsiWeight => weight_spec(1.0),
        Gene::SellPnlWeight => weight_spec(1.1),
        Gene::SellTrendWeight => weight_spec(1.1),
        Gene::SellRsiFadeWeight => weight_spec(0.9),
        Gene::SellAdverseWeight => weight_spec(1.0),
        Gene::NeutralBand => GeneSpec {
            min: 0.02,
            max: 0.8,
            init: 0.16,
            mutation: 0.05,
        },
        Gene::MaxTargetWeight => GeneSpec {
            min: 0.05,
            max: 0.5,
            init: 0.22,
            mutation: 0.04,
        },
        Gene::FreshEntryScale => GeneSpec {
            min: 0.05,
            max: 0.8,
            init: 0.3,
            mutation: 0.05,
        },
        Gene::HoldWeight => GeneSpec {
            min: 0.0,
            max: 3.0,
            init: 1.0,
            mutation: 0.15,
        },
        Gene::HoldFloorFraction => GeneSpec {
            min: 0.0,
            max: 1.0,
            init: 0.55,
            mutation: 0.06,
        },
        Gene::CorePositionFraction => GeneSpec {
            min: 0.0,
            max: 0.8,
            init: 0.22,
            mutation: 0.05,
        },
    }
}

fn buy_score(genome: &Genome, ctx: &DecisionContext) -> f64 {
    let rebound = normalized_signal(
        percent_diff(ctx.price, ctx.local_minimum),
        genome.genes[Gene::ReboundBuyPriceThreshold],
        genome.genes[Gene::BuyDistanceWeightAmount],
    );
    let ema_discount = normalized_signal(
        ((ctx.price_ema - ctx.price) / ctx.price_ema.max(1e-6)).max(0.0),
        0.0,
        genome.genes[Gene::BuyDistanceWeightAmount] * 0.8,
    );
    let rsi_oversold = normalized_signal(
        ((genome.genes[Gene::MaxRsiBuy] * 100.0) - ctx.decider_rsi).max(0.0) / 100.0,
        0.0,
        0.12,
    );
    let drop_since_last_buy = ctx.last_buy_price.map_or(0.45, |last_buy| {
        normalized_signal(
            percent_diff(last_buy, ctx.price),
            genome.genes[Gene::DropBuyThreshold],
            genome.genes[Gene::BuyDistanceWeightAmount],
        )
    });
    let trend_alignment = trend_alignment_signal(ctx, genome.genes[Gene::BuyDistanceWeightAmount]);
    let rsi_recovery = ctx.lowest_rsi_since_flat.map_or(0.0, |lowest_rsi| {
        normalized_signal(((ctx.decider_rsi - lowest_rsi).max(0.0)) / 100.0, 0.0, 0.10)
    });

    weighted_score(&[
        (rebound, genome.genes[Gene::BuyReboundWeight]),
        (ema_discount, genome.genes[Gene::BuyEmaWeight]),
        (rsi_oversold, genome.genes[Gene::BuyRsiWeight]),
        (drop_since_last_buy, genome.genes[Gene::BuyDropWeight]),
        (trend_alignment, genome.genes[Gene::BuyTrendWeight]),
        (rsi_recovery, genome.genes[Gene::BuyRsiRecoveryWeight]),
    ])
}

fn active_buy_score(genome: &Genome, ctx: &DecisionContext) -> f64 {
    let setup = buy_setup_score(genome, ctx);
    let buy = buy_score(genome, ctx);
    let threshold = genome.genes[Gene::BuyScoreThreshold];
    let conviction = normalized_signal(buy, threshold * 0.75, (1.0 - threshold).max(1e-6));
    weighted_score(&[(setup, 0.45), (conviction, 0.55)])
}

fn sell_score(genome: &Genome, ctx: &DecisionContext) -> f64 {
    let trailing_fade = normalized_signal(
        percent_diff(ctx.local_maximum, ctx.price),
        genome.genes[Gene::ReboundSellPriceThreshold],
        genome.genes[Gene::SellDistanceWeightAmount],
    );
    let ema_premium = normalized_signal(
        ((ctx.price - ctx.price_ema) / ctx.price_ema.max(1e-6)).max(0.0),
        0.0,
        genome.genes[Gene::SellDistanceWeightAmount] * 0.8,
    );
    let rsi_overbought = normalized_signal(
        (ctx.amount_rsi - (genome.genes[Gene::MinRsiSell] * 100.0)).max(0.0) / 100.0,
        0.0,
        0.12,
    );
    let pnl_signal = normalized_signal(
        (ctx.unrealized_pnl_pct() / 100.0).max(0.0),
        0.0,
        genome.genes[Gene::TakeProfitPct],
    );
    let trend_breakdown = trend_breakdown_signal(ctx, genome.genes[Gene::SellDistanceWeightAmount]);
    let rsi_fade = ctx.highest_rsi_since_long.map_or(0.0, |highest_rsi| {
        normalized_signal(((highest_rsi - ctx.amount_rsi).max(0.0)) / 100.0, 0.0, 0.12)
    });
    let adverse_excursion = if ctx.position_quantity > 0.0 {
        normalized_signal(
            ((ctx.position_avg_price - ctx.price) / ctx.position_avg_price.max(1e-6)).max(0.0),
            0.0,
            genome.genes[Gene::SellDistanceWeightAmount] * 1.2,
        )
    } else {
        0.0
    };

    weighted_score(&[
        (trailing_fade, genome.genes[Gene::SellTrailWeight]),
        (ema_premium, genome.genes[Gene::SellEmaWeight]),
        (rsi_overbought, genome.genes[Gene::SellRsiWeight]),
        (pnl_signal, genome.genes[Gene::SellPnlWeight]),
        (trend_breakdown, genome.genes[Gene::SellTrendWeight]),
        (rsi_fade, genome.genes[Gene::SellRsiFadeWeight]),
        (adverse_excursion, genome.genes[Gene::SellAdverseWeight]),
    ])
}

fn active_sell_score(genome: &Genome, ctx: &DecisionContext) -> f64 {
    let sell = sell_score(genome, ctx);
    let threshold = genome.genes[Gene::SellScoreThreshold];
    normalized_signal(sell, threshold * 0.8, (1.0 - threshold).max(1e-6))
}

fn hold_signal(genome: &Genome, ctx: &DecisionContext, sell_edge: f64, buy_edge: f64) -> f64 {
    if ctx.position_quantity <= 0.0 {
        return 0.0;
    }
    let pnl_pct = ctx.unrealized_pnl_pct() / 100.0;
    let sell_relief = 1.0 - (sell_edge / 2.0).clamp(0.0, 1.0);
    let neutral_hold = 1.0 - ((buy_edge - sell_edge).abs() / 1.5).clamp(0.0, 1.0);
    let trend_support = trend_alignment_signal(ctx, genome.genes[Gene::BuyDistanceWeightAmount]);
    let follow_through = normalized_signal(
        pnl_pct.max(0.0),
        0.0,
        genome.genes[Gene::BuyDistanceWeightAmount] * 1.5,
    );
    let profit_support = if pnl_pct > 0.0 {
        normalized_signal(pnl_pct, 0.0, genome.genes[Gene::TakeProfitPct]) * 0.35
    } else {
        0.0
    };
    (0.25
        + sell_relief * 0.45
        + neutral_hold * 0.35
        + trend_support * 0.30
        + follow_through * 0.35
        + profit_support)
        .clamp(0.0, 2.0)
}

fn buy_setup_score(genome: &Genome, ctx: &DecisionContext) -> f64 {
    let rebound = normalized_signal(
        percent_diff(ctx.price, ctx.local_minimum),
        genome.genes[Gene::ReboundBuyPriceThreshold] * 0.35,
        genome.genes[Gene::BuyDistanceWeightAmount],
    );
    let pullback = normalized_signal(
        percent_diff(ctx.local_maximum, ctx.price),
        genome.genes[Gene::MaxReboundBuyPriceThreshold] * 0.35,
        genome.genes[Gene::BuyDistanceWeightAmount] * 1.2,
    );
    weighted_score(&[(rebound, 0.55), (pullback, 0.45)])
}

fn reentry_penalty(genome: &Genome, ctx: &DecisionContext) -> f64 {
    let Some(last_sell) = ctx.last_sell_price else {
        return 0.0;
    };
    let cooldown = genome.genes[Gene::SellDropBuyThreshold].max(1e-6);
    let distance = percent_diff_abs(ctx.price, last_sell);
    (1.0 - (distance / cooldown).clamp(0.0, 1.0)) * 0.85
}

fn directional_edge(primary: f64, opposing: f64, neutral_band: f64) -> f64 {
    normalized_signal(primary - opposing, neutral_band, 0.9)
}

fn trend_alignment_signal(ctx: &DecisionContext, scale: f64) -> f64 {
    let (Some(fast), Some(slow)) = (ctx.fast_ema, ctx.slow_ema) else {
        return 0.0;
    };
    let fast_over_slow = ((fast - slow) / slow.max(1e-6)).max(0.0);
    let price_over_fast = ((ctx.price - fast) / fast.max(1e-6)).max(0.0);
    weighted_score(&[
        (normalized_signal(fast_over_slow, 0.0, scale * 0.8), 0.65),
        (normalized_signal(price_over_fast, 0.0, scale * 0.9), 0.35),
    ])
}

fn trend_breakdown_signal(ctx: &DecisionContext, scale: f64) -> f64 {
    let (Some(fast), Some(slow)) = (ctx.fast_ema, ctx.slow_ema) else {
        return 0.0;
    };
    let slow_over_fast = ((slow - fast) / slow.max(1e-6)).max(0.0);
    let fast_over_price = ((fast - ctx.price) / fast.max(1e-6)).max(0.0);
    weighted_score(&[
        (normalized_signal(slow_over_fast, 0.0, scale * 0.8), 0.6),
        (normalized_signal(fast_over_price, 0.0, scale * 0.9), 0.4),
    ])
}

fn normalized_signal(value: f64, threshold: f64, scale: f64) -> f64 {
    ((value - threshold) / scale.max(1e-6)).clamp(0.0, 2.0)
}

fn weighted_score(parts: &[(f64, f64)]) -> f64 {
    let total_weight = parts
        .iter()
        .map(|(_, weight)| *weight)
        .sum::<f64>()
        .max(1e-6);
    parts
        .iter()
        .map(|(signal, weight)| signal * weight)
        .sum::<f64>()
        / total_weight
}

fn weight_spec(init: f64) -> GeneSpec {
    GeneSpec {
        min: 0.1,
        max: 3.0,
        init,
        mutation: 0.2,
    }
}

fn jitter(value: f64, amount: f64, spec: GeneSpec, rng: &mut StdRng) -> f64 {
    clamp(value + rng.random_range(-amount..amount), spec)
}

fn clamp(value: f64, spec: GeneSpec) -> f64 {
    value.clamp(spec.min, spec.max)
}
