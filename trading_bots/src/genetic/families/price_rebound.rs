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
    MaxTargetWeight,
    FreshEntryScale,
    HoldFloorFraction,
    CorePositionFraction,
    CashBias,
    OverTargetPenaltyWeight,
    SetupReboundWeight,
    SetupPullbackWeight,
    ActiveBuySetupWeight,
    ActiveBuyConvictionWeight,
    TargetSetupWeight,
    TargetBuyWeight,
    TargetSellWeight,
    TargetReentryWeight,
    TargetTakeProfitWeight,
    MarketFitSetupWeight,
    MarketFitBuyWeight,
    MarketFitRelativeWeight,
    TargetRelativeWeight,
    RelativeForgivenessPct,
    RelativeGraceBars,
    RelativeLagStreakThreshold,
    RelativeLagStreakScale,
    RelativeMomentumWeight,
    RelativeRecoveryWeight,
    RelativeDrawdownWeight,
    RelativeStrengthThreshold,
    RelativeStrengthScale,
    RelativeDeltaThreshold,
    RelativeDeltaScale,
    RelativeDrawdownThreshold,
    RelativeDrawdownScale,
    SignalConvictionThreshold,
    SignalConvictionScale,
    HoldConvictionThreshold,
    HoldConvictionScale,
    MaxWeightConvictionThreshold,
    MaxWeightConvictionScale,
}

impl Gene {
    const ALL: [Self; 69] = [
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
        Self::MaxTargetWeight,
        Self::FreshEntryScale,
        Self::HoldFloorFraction,
        Self::CorePositionFraction,
        Self::CashBias,
        Self::OverTargetPenaltyWeight,
        Self::SetupReboundWeight,
        Self::SetupPullbackWeight,
        Self::ActiveBuySetupWeight,
        Self::ActiveBuyConvictionWeight,
        Self::TargetSetupWeight,
        Self::TargetBuyWeight,
        Self::TargetSellWeight,
        Self::TargetReentryWeight,
        Self::TargetTakeProfitWeight,
        Self::MarketFitSetupWeight,
        Self::MarketFitBuyWeight,
        Self::MarketFitRelativeWeight,
        Self::TargetRelativeWeight,
        Self::RelativeForgivenessPct,
        Self::RelativeGraceBars,
        Self::RelativeLagStreakThreshold,
        Self::RelativeLagStreakScale,
        Self::RelativeMomentumWeight,
        Self::RelativeRecoveryWeight,
        Self::RelativeDrawdownWeight,
        Self::RelativeStrengthThreshold,
        Self::RelativeStrengthScale,
        Self::RelativeDeltaThreshold,
        Self::RelativeDeltaScale,
        Self::RelativeDrawdownThreshold,
        Self::RelativeDrawdownScale,
        Self::SignalConvictionThreshold,
        Self::SignalConvictionScale,
        Self::HoldConvictionThreshold,
        Self::HoldConvictionScale,
        Self::MaxWeightConvictionThreshold,
        Self::MaxWeightConvictionScale,
    ];
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Genome {
    genes: EnumMap<Gene, f64>,
}

#[derive(Clone, Copy, Debug)]
enum CashVariant {
    Static,
    Breadth,
    LeaderGap,
    WeakRegime,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Family;

#[derive(Clone, Copy, Debug, Default)]
pub struct CashBreadthFamily;

#[derive(Clone, Copy, Debug, Default)]
pub struct CashLeaderGapFamily;

#[derive(Clone, Copy, Debug, Default)]
pub struct CashWeakRegimeFamily;

macro_rules! impl_price_rebound_family {
    ($name:ident, $kind:expr, $cash_variant:expr) => {
        impl StrategyFamilySpec for $name {
            type Genome = Genome;

            fn kind(&self) -> GeneticFamily {
                $kind
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

                let over_target_penalty = (ctx.current_weight()
                    / self.max_target_weight(genome, ctx).max(1e-6))
                .clamp(0.0, 1.0);
                let desirability = target_signal(genome, ctx)
                    - over_target_penalty * genome.genes[Gene::OverTargetPenaltyWeight];
                desirability.max(0.0)
            }

            fn cash_desirability(
                &self,
                genome: &Self::Genome,
                contexts: &[DecisionContext],
            ) -> f64 {
                cash_desirability($cash_variant, genome, contexts)
            }

            fn min_target_weight(&self, genome: &Self::Genome, ctx: &DecisionContext) -> f64 {
                if ctx.position_quantity <= 0.0 {
                    return 0.0;
                }
                if (ctx.unrealized_pnl_pct() / 100.0) <= -genome.genes[Gene::StopLossPct] {
                    return 0.0;
                }
                let signal = target_signal(genome, ctx);
                let hold_conviction = normalized_signal(
                    signal,
                    genome.genes[Gene::HoldConvictionThreshold],
                    genome.genes[Gene::HoldConvictionScale],
                )
                .clamp(0.0, 1.0);
                let floor_fraction =
                    genome.genes[Gene::HoldFloorFraction] * hold_conviction * hold_conviction;
                ctx.current_weight() * floor_fraction
            }

            fn max_target_weight(&self, genome: &Self::Genome, ctx: &DecisionContext) -> f64 {
                let cap = genome.genes[Gene::MaxTargetWeight];
                let conviction = normalized_signal(
                    target_signal(genome, ctx),
                    genome.genes[Gene::MaxWeightConvictionThreshold],
                    genome.genes[Gene::MaxWeightConvictionScale],
                )
                .clamp(0.0, 1.0);
                let curve = 1.0 + genome.genes[Gene::FreshEntryScale] * 6.0;
                cap * conviction.powf(curve)
            }
        }
    };
}

impl_price_rebound_family!(Family, GeneticFamily::PriceRebound, CashVariant::Static);
impl_price_rebound_family!(
    CashBreadthFamily,
    GeneticFamily::PriceReboundCashBreadth,
    CashVariant::Breadth
);
impl_price_rebound_family!(
    CashLeaderGapFamily,
    GeneticFamily::PriceReboundCashLeaderGap,
    CashVariant::LeaderGap
);
impl_price_rebound_family!(
    CashWeakRegimeFamily,
    GeneticFamily::PriceReboundCashWeakRegime,
    CashVariant::WeakRegime
);

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
        Gene::MaxTargetWeight => GeneSpec {
            min: 0.05,
            max: 0.5,
            init: 0.22,
            mutation: 0.04,
        },
        Gene::FreshEntryScale => GeneSpec {
            min: 0.02,
            max: 0.24,
            init: 0.08,
            mutation: 0.02,
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
        Gene::CashBias => GeneSpec {
            min: 0.0,
            max: 0.3,
            init: 0.05,
            mutation: 0.02,
        },
        Gene::OverTargetPenaltyWeight => GeneSpec {
            min: 0.0,
            max: 0.6,
            init: 0.18,
            mutation: 0.04,
        },
        Gene::SetupReboundWeight => weight_spec(0.55),
        Gene::SetupPullbackWeight => weight_spec(0.45),
        Gene::ActiveBuySetupWeight => weight_spec(0.45),
        Gene::ActiveBuyConvictionWeight => weight_spec(0.55),
        Gene::TargetSetupWeight => weight_spec(0.28),
        Gene::TargetBuyWeight => weight_spec(0.98),
        Gene::TargetSellWeight => weight_spec(0.72),
        Gene::TargetReentryWeight => weight_spec(0.28),
        Gene::TargetTakeProfitWeight => weight_spec(0.28),
        Gene::MarketFitSetupWeight => weight_spec(0.34),
        Gene::MarketFitBuyWeight => weight_spec(0.46),
        Gene::MarketFitRelativeWeight => weight_spec(0.35),
        Gene::TargetRelativeWeight => weight_spec(0.7),
        Gene::RelativeForgivenessPct => GeneSpec {
            min: 0.0,
            max: 0.08,
            init: 0.02,
            mutation: 0.008,
        },
        Gene::RelativeGraceBars => GeneSpec {
            min: 0.0,
            max: 40.0,
            init: 8.0,
            mutation: 3.0,
        },
        Gene::RelativeLagStreakThreshold => GeneSpec {
            min: 0.0,
            max: 20.0,
            init: 2.0,
            mutation: 1.5,
        },
        Gene::RelativeLagStreakScale => GeneSpec {
            min: 0.5,
            max: 20.0,
            init: 5.0,
            mutation: 1.5,
        },
        Gene::RelativeMomentumWeight => weight_spec(0.55),
        Gene::RelativeRecoveryWeight => weight_spec(0.45),
        Gene::RelativeDrawdownWeight => weight_spec(0.75),
        Gene::RelativeStrengthThreshold => GeneSpec {
            min: 0.0,
            max: 0.12,
            init: 0.01,
            mutation: 0.01,
        },
        Gene::RelativeStrengthScale => GeneSpec {
            min: 0.01,
            max: 0.25,
            init: 0.04,
            mutation: 0.015,
        },
        Gene::RelativeDeltaThreshold => GeneSpec {
            min: 0.0,
            max: 0.05,
            init: 0.002,
            mutation: 0.004,
        },
        Gene::RelativeDeltaScale => GeneSpec {
            min: 0.002,
            max: 0.12,
            init: 0.02,
            mutation: 0.008,
        },
        Gene::RelativeDrawdownThreshold => GeneSpec {
            min: 0.0,
            max: 0.12,
            init: 0.01,
            mutation: 0.01,
        },
        Gene::RelativeDrawdownScale => GeneSpec {
            min: 0.01,
            max: 0.3,
            init: 0.05,
            mutation: 0.015,
        },
        Gene::SignalConvictionThreshold => GeneSpec {
            min: 0.0,
            max: 0.6,
            init: 0.04,
            mutation: 0.03,
        },
        Gene::SignalConvictionScale => GeneSpec {
            min: 0.05,
            max: 1.5,
            init: 0.42,
            mutation: 0.06,
        },
        Gene::HoldConvictionThreshold => GeneSpec {
            min: 0.0,
            max: 1.2,
            init: 0.32,
            mutation: 0.05,
        },
        Gene::HoldConvictionScale => GeneSpec {
            min: 0.1,
            max: 1.5,
            init: 0.72,
            mutation: 0.06,
        },
        Gene::MaxWeightConvictionThreshold => GeneSpec {
            min: 0.0,
            max: 1.2,
            init: 0.34,
            mutation: 0.05,
        },
        Gene::MaxWeightConvictionScale => GeneSpec {
            min: 0.1,
            max: 1.5,
            init: 0.76,
            mutation: 0.06,
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
    weighted_score(&[
        (setup, genome.genes[Gene::ActiveBuySetupWeight]),
        (conviction, genome.genes[Gene::ActiveBuyConvictionWeight]),
    ])
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
    let benchmark_lag = relative_underperformance_signal(genome, ctx);
    let relative_drawdown = relative_drawdown_signal(genome, ctx);
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
        (
            weighted_score(&[
                (benchmark_lag, genome.genes[Gene::SellPnlWeight]),
                (relative_drawdown, genome.genes[Gene::RelativeDrawdownWeight]),
            ]),
            1.0,
        ),
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
    weighted_score(&[
        (rebound, genome.genes[Gene::SetupReboundWeight]),
        (pullback, genome.genes[Gene::SetupPullbackWeight]),
    ])
}

fn reentry_penalty(genome: &Genome, ctx: &DecisionContext) -> f64 {
    let Some(last_sell) = ctx.last_sell_price else {
        return 0.0;
    };
    let cooldown = genome.genes[Gene::SellDropBuyThreshold].max(1e-6);
    let distance = percent_diff_abs(ctx.price, last_sell);
    (1.0 - (distance / cooldown).clamp(0.0, 1.0)) * 0.85
}

fn target_signal(genome: &Genome, ctx: &DecisionContext) -> f64 {
    let setup_signal = buy_setup_score(genome, ctx);
    let buy_signal = active_buy_score(genome, ctx);
    let sell_signal = active_sell_score(genome, ctx);
    let reentry_penalty = reentry_penalty(genome, ctx);
    let relative_outperformance = relative_outperformance_signal(genome, ctx);
    let relative_momentum = relative_momentum_signal(genome, ctx);
    let relative_recovery = relative_recovery_signal(genome, ctx);
    let relative_drawdown = relative_drawdown_signal(genome, ctx);
    let supportive = setup_signal * genome.genes[Gene::TargetSetupWeight]
        + buy_signal
            * (1.0 - reentry_penalty)
            * genome.genes[Gene::BuyPercent]
            * genome.genes[Gene::TargetBuyWeight]
        + relative_outperformance * genome.genes[Gene::TargetRelativeWeight]
        + relative_momentum * genome.genes[Gene::RelativeMomentumWeight]
        + relative_recovery * genome.genes[Gene::RelativeRecoveryWeight];
    let opposing = sell_signal * genome.genes[Gene::SellPercent] * genome.genes[Gene::TargetSellWeight]
        + reentry_penalty * genome.genes[Gene::TargetReentryWeight]
        + relative_drawdown * genome.genes[Gene::RelativeDrawdownWeight] * 0.35;
    let balance = supportive - opposing;
    let market_fit = weighted_score(&[
        (setup_signal.min(1.6), genome.genes[Gene::MarketFitSetupWeight]),
        (buy_signal.min(1.6), genome.genes[Gene::MarketFitBuyWeight]),
        (
            (relative_outperformance + relative_momentum + relative_recovery).min(1.6),
            genome.genes[Gene::MarketFitRelativeWeight],
        ),
    ]);
    let conviction = normalized_signal(
        balance,
        genome.genes[Gene::SignalConvictionThreshold],
        genome.genes[Gene::SignalConvictionScale],
    );

    (conviction * market_fit).clamp(0.0, 2.5)
}

fn relative_outperformance_signal(genome: &Genome, ctx: &DecisionContext) -> f64 {
    let forgiveness = relative_forgiveness(genome, ctx);
    normalized_signal(
        ((ctx.excess_return_since_entry_pct() / 100.0) + forgiveness).max(0.0),
        genome.genes[Gene::RelativeStrengthThreshold],
        genome.genes[Gene::RelativeStrengthScale],
    )
}

fn relative_underperformance_signal(genome: &Genome, ctx: &DecisionContext) -> f64 {
    let forgiveness = relative_forgiveness(genome, ctx);
    let streak_signal = normalized_signal(
        ctx.underperformance_streak() as f64,
        genome.genes[Gene::RelativeLagStreakThreshold],
        genome.genes[Gene::RelativeLagStreakScale],
    );
    normalized_signal(
        ((-ctx.excess_return_since_entry_pct() / 100.0) - forgiveness).max(0.0),
        genome.genes[Gene::RelativeStrengthThreshold],
        genome.genes[Gene::RelativeStrengthScale],
    ) * streak_signal
}

fn relative_momentum_signal(genome: &Genome, ctx: &DecisionContext) -> f64 {
    normalized_signal(
        (ctx.excess_return_delta_pct() / 100.0).max(0.0),
        genome.genes[Gene::RelativeDeltaThreshold],
        genome.genes[Gene::RelativeDeltaScale],
    )
}

fn relative_recovery_signal(genome: &Genome, ctx: &DecisionContext) -> f64 {
    let lagged = normalized_signal(
        (-ctx.excess_return_since_entry_pct() / 100.0).max(0.0),
        0.0,
        genome.genes[Gene::RelativeStrengthScale] * 1.5,
    );
    relative_momentum_signal(genome, ctx) * lagged
}

fn relative_drawdown_signal(genome: &Genome, ctx: &DecisionContext) -> f64 {
    normalized_signal(
        ((ctx.excess_return_peak_pct() - ctx.excess_return_since_entry_pct()) / 100.0).max(0.0),
        genome.genes[Gene::RelativeDrawdownThreshold],
        genome.genes[Gene::RelativeDrawdownScale],
    )
}

fn relative_forgiveness(genome: &Genome, ctx: &DecisionContext) -> f64 {
    let grace_bars = genome.genes[Gene::RelativeGraceBars].max(1.0);
    let early_life_scale = (1.0 - (ctx.bars_since_entry() as f64 / grace_bars)).clamp(0.0, 1.0);
    genome.genes[Gene::RelativeForgivenessPct] * (0.35 + early_life_scale * 0.65)
}

fn cash_desirability(cash_variant: CashVariant, genome: &Genome, contexts: &[DecisionContext]) -> f64 {
    let base = genome.genes[Gene::CashBias] + genome.genes[Gene::CorePositionFraction] * 0.2;
    if contexts.is_empty() {
        return base;
    }

    let mut signals = contexts
        .iter()
        .map(|ctx| target_signal(genome, ctx))
        .collect::<Vec<_>>();
    signals.sort_by(|left, right| right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal));

    let count = signals.len() as f64;
    let top = signals[0];
    let second = signals.get(1).copied().unwrap_or(0.0);
    let mean = signals.iter().sum::<f64>() / count.max(1.0);
    let strong_fraction = signals.iter().filter(|signal| **signal >= 0.65).count() as f64 / count;
    let live_fraction = signals.iter().filter(|signal| **signal >= 0.35).count() as f64 / count;
    let variance = signals
        .iter()
        .map(|signal| {
            let centered = signal - mean;
            centered * centered
        })
        .sum::<f64>()
        / count.max(1.0);
    let dispersion = variance.sqrt();
    let held_relative_drag = contexts
        .iter()
        .filter(|ctx| ctx.position_quantity > 0.0)
        .map(|ctx| (-ctx.excess_return_since_entry_pct() / 100.0).max(0.0))
        .sum::<f64>()
        / contexts
            .iter()
            .filter(|ctx| ctx.position_quantity > 0.0)
            .count()
            .max(1) as f64;

    let cash_pressure = match cash_variant {
        CashVariant::Static => 0.0,
        CashVariant::Breadth => {
            (1.0 - strong_fraction) * 0.85 + (0.45 - mean).max(0.0) * 0.75
        }
        CashVariant::LeaderGap => {
            let leader_gap = (top - second).max(0.0);
            (0.85 - top).max(0.0) * 0.9 + (0.35 - leader_gap).max(0.0) * 1.1
        }
        CashVariant::WeakRegime => {
            (1.0 - live_fraction) * 0.7
                + (0.35 - dispersion).max(0.0) * 0.9
                + (0.5 - mean).max(0.0) * 0.8
                + held_relative_drag * 0.8
        }
    };

    (base + cash_pressure).clamp(0.0, 3.0)
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
