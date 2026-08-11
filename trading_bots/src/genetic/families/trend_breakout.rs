use enum_map::{Enum, EnumMap};
use rand::{rngs::StdRng, Rng};
use serde::{de::Error as _, Deserialize, Deserializer, Serialize};

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
    MinEntryRsi,
    MaxExitRsi,
}

impl Gene {
    const ALL: [Self; 11] = [
        Self::FastEmaAlpha,
        Self::SlowEmaAlpha,
        Self::PullbackMinPct,
        Self::PullbackMaxPct,
        Self::TrendSpreadMinPct,
        Self::TrendExitPct,
        Self::StopLossPct,
        Self::TakeProfitPct,
        Self::BuyPercent,
        Self::MinEntryRsi,
        Self::MaxExitRsi,
    ];
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Genome {
    genes: EnumMap<Gene, f64>,
}

#[derive(Serialize, Deserialize)]
struct RawGenome {
    genes: EnumMap<Gene, f64>,
}

impl<'de> Deserialize<'de> for Genome {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawGenome::deserialize(deserializer)?;
        if Gene::ALL.iter().any(|gene| !raw.genes[*gene].is_finite()) {
            return Err(D::Error::custom("trend-breakout genes must be finite"));
        }
        Ok(Self { genes: raw.genes }.normalized())
    }
}

impl Genome {
    fn normalized(mut self) -> Self {
        for gene in Gene::ALL {
            let gene_spec = spec(gene);
            let value = self.genes[gene];
            self.genes[gene] = if value.is_finite() {
                clamp(value, gene_spec)
            } else {
                gene_spec.init
            };
        }

        if self.genes[Gene::FastEmaAlpha] <= self.genes[Gene::SlowEmaAlpha] {
            self.genes[Gene::FastEmaAlpha] =
                (self.genes[Gene::SlowEmaAlpha] + 0.01).min(spec(Gene::FastEmaAlpha).max);
        }
        if self.genes[Gene::PullbackMinPct] > self.genes[Gene::PullbackMaxPct] {
            let minimum = self.genes[Gene::PullbackMaxPct];
            self.genes[Gene::PullbackMaxPct] = self.genes[Gene::PullbackMinPct];
            self.genes[Gene::PullbackMinPct] = minimum;
        }
        self
    }
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
        Genome { genes }.normalized()
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
        *genome = (*genome).normalized();
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
        Genome { genes }.normalized()
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

    fn asset_desirability(&self, genome: &Self::Genome, ctx: &DecisionContext) -> f64 {
        let Some(fast_ema) = ctx.fast_ema else {
            return 0.0;
        };
        let Some(slow_ema) = ctx.slow_ema else {
            return 0.0;
        };
        if fast_ema <= slow_ema {
            return 0.0;
        }
        let trend_spread = (fast_ema - slow_ema) / slow_ema.max(1e-6);
        if trend_spread < genome.genes[Gene::TrendSpreadMinPct] {
            return 0.0;
        }
        if ctx.price <= slow_ema || ctx.price < fast_ema * 0.995 {
            return 0.0;
        }
        let pullback = percent_diff(ctx.local_maximum, ctx.price);
        if pullback < genome.genes[Gene::PullbackMinPct]
            || pullback > genome.genes[Gene::PullbackMaxPct]
        {
            return 0.0;
        }
        let rsi_gate = ctx.decider_rsi - genome.genes[Gene::MinEntryRsi] * 100.0;
        let pnl_pct = ctx.unrealized_pnl_pct() / 100.0;
        if pnl_pct <= -genome.genes[Gene::StopLossPct] {
            return 0.0;
        }
        let trend_term =
            (trend_spread / genome.genes[Gene::TrendSpreadMinPct].max(1e-6)).clamp(0.0, 2.5);
        let pullback_term =
            (pullback / genome.genes[Gene::PullbackMaxPct].max(1e-6)).clamp(0.0, 1.0);
        let rsi_term = (rsi_gate / 25.0).clamp(0.0, 1.5);

        let mut desirability =
            genome.genes[Gene::BuyPercent] * (0.4 + trend_term + pullback_term + rsi_term);
        if pnl_pct >= genome.genes[Gene::TakeProfitPct] {
            desirability *= 0.35;
        }
        if ctx.price < fast_ema * (1.0 - genome.genes[Gene::TrendExitPct]) {
            desirability *= 0.15;
        }
        if ctx.amount_rsi >= genome.genes[Gene::MaxExitRsi] * 100.0 {
            desirability *= 0.25;
        }
        desirability.max(0.0)
    }

    fn cash_desirability(&self, genome: &Self::Genome, _contexts: &[DecisionContext]) -> f64 {
        0.35 + (1.0 - genome.genes[Gene::BuyPercent]).max(0.0) * 0.75
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

#[cfg(test)]
mod tests {
    use enum_map::EnumMap;
    use rand::{rngs::StdRng, SeedableRng};

    use super::{spec, Family, Gene, Genome, RawGenome};
    use crate::genetic::family::StrategyFamilySpec;

    fn assert_invariants(genome: &Genome) {
        for gene in Gene::ALL {
            let value = genome.genes[gene];
            let range = spec(gene);
            assert!(value.is_finite());
            assert!(value >= range.min && value <= range.max);
        }
        assert!(genome.genes[Gene::FastEmaAlpha] > genome.genes[Gene::SlowEmaAlpha]);
        assert!(genome.genes[Gene::PullbackMinPct] <= genome.genes[Gene::PullbackMaxPct]);
    }

    #[test]
    fn seed_mutation_and_crossover_preserve_relational_invariants() {
        let family = Family;
        let mut rng = StdRng::seed_from_u64(0xBAD5EED);
        let mut left = family.seed_genome(&mut rng);
        let mut right = family.seed_genome(&mut rng);

        for _ in 0..1_000 {
            family.mutate(&mut left, &mut rng, 4.0);
            family.mutate(&mut right, &mut rng, 4.0);
            let child = family.crossover(&left, &right, &mut rng);
            assert_invariants(&left);
            assert_invariants(&right);
            assert_invariants(&child);
            left = right;
            right = child;
        }
    }

    #[test]
    fn deserialization_normalizes_inverted_pullback_bounds() {
        let family = Family;
        let mut genome = family.seed_genome(&mut StdRng::seed_from_u64(7));
        genome.genes[Gene::PullbackMinPct] = 0.04;
        genome.genes[Gene::PullbackMaxPct] = 0.01;

        let encoded = serde_json::to_vec(&genome).expect("genome should serialize");
        let decoded: Genome = serde_json::from_slice(&encoded).expect("genome should deserialize");

        assert_eq!(decoded.genes[Gene::PullbackMinPct], 0.01);
        assert_eq!(decoded.genes[Gene::PullbackMaxPct], 0.04);
        assert_invariants(&decoded);
    }

    #[test]
    fn deserialization_rejects_non_finite_genes() {
        let family = Family;
        let genome = family.seed_genome(&mut StdRng::seed_from_u64(11));
        let mut genes: EnumMap<Gene, f64> = genome.genes;
        genes[Gene::PullbackMinPct] = f64::NAN;
        let encoded = postcard::to_allocvec(&RawGenome { genes })
            .expect("raw genome should serialize for regression setup");

        postcard::from_bytes::<Genome>(&encoded).expect_err("non-finite genes must be rejected");
    }
}
