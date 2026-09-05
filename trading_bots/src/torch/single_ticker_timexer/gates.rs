use std::collections::BTreeMap;

use anyhow::{ensure, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use super::data::HORIZONS;
use super::evaluate::{Evaluation, OriginScores, EVIDENCE_SCHEMA, FROZEN_SEEDS, SCORING_CONTRACT};

const BOOTSTRAP_REPLICATES: usize = 4_000;
const MINIMUM_CALENDAR_BLOCKS: usize = 8;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateCheck {
    pub name: String,
    pub passed: bool,
    pub value: f64,
    pub threshold: f64,
    pub confidence: Option<[f64; 2]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateDecision {
    pub promoted: bool,
    pub identity: String,
    pub candidate: String,
    pub baseline: String,
    pub seeds: [u64; 3],
    pub calendar_blocks: usize,
    pub bootstrap_replicates: usize,
    pub checks: Vec<GateCheck>,
}

#[derive(Clone, Copy)]
enum Metric {
    Nll,
    Crps,
    SquaredError,
    Brier,
}

#[derive(Clone, Copy)]
struct PairedBlock {
    candidate: f64,
    baseline: f64,
    count: usize,
}

/// Weekly calendar blocks preserve overlapping 100-bar outcomes; all seeds share each draw.
pub fn assess(candidate: &[Evaluation], baseline: &[Evaluation]) -> Result<GateDecision> {
    let (candidate, baseline) = validate_pairing(candidate, baseline, "BarTrunk")?;
    let blocks = paired_blocks(&candidate, &baseline, Metric::Nll, None);
    ensure!(
        blocks.len() >= MINIMUM_CALENDAR_BLOCKS,
        "promotion needs at least {MINIMUM_CALENDAR_BLOCKS} calendar weeks; got {}",
        blocks.len()
    );
    let mut decision = GateDecision {
        promoted: false,
        identity: candidate[0].identity.clone(),
        candidate: candidate[0].model_kind.clone(),
        baseline: baseline[0].model_kind.clone(),
        seeds: FROZEN_SEEDS,
        calendar_blocks: blocks.len(),
        bootstrap_replicates: BOOTSTRAP_REPLICATES,
        checks: Vec::new(),
    };
    let (delta, confidence) = bootstrap(&blocks, false, false);
    decision.add(
        "mean NLL improvement >= 0.02 nat",
        delta <= -0.02,
        delta,
        -0.02,
        Some(confidence),
    );
    decision.add(
        "mean NLL paired 95% upper bound < 0",
        confidence[1] < 0.0,
        confidence[1],
        0.0,
        Some(confidence),
    );
    let mut significant = 0;
    for (h, &horizon) in HORIZONS.iter().enumerate() {
        let (delta, interval) = bootstrap(
            &paired_blocks(&candidate, &baseline, Metric::Nll, Some(h)),
            false,
            false,
        );
        significant += usize::from(interval[1] < 0.0);
        decision.add(
            &format!("H{horizon} NLL regression <= 0.01 nat"),
            delta <= 0.01,
            delta,
            0.01,
            Some(interval),
        );
    }
    decision.add(
        "at least four individually significant NLL horizons",
        significant >= 4,
        significant as f64,
        4.0,
        None,
    );
    let (ratio, interval) = bootstrap(
        &paired_blocks(&candidate, &baseline, Metric::Crps, None),
        true,
        false,
    );
    decision.add(
        "standardized CRPS improvement >= 2%",
        ratio <= 0.98,
        ratio,
        0.98,
        Some(interval),
    );
    decision.add(
        "standardized CRPS 95% upper ratio < 1",
        interval[1] < 1.0,
        interval[1],
        1.0,
        Some(interval),
    );
    for (h, &horizon) in HORIZONS.iter().enumerate() {
        let (ratio, interval) = bootstrap(
            &paired_blocks(&candidate, &baseline, Metric::SquaredError, Some(h)),
            true,
            true,
        );
        if horizon >= 4 {
            decision.add(
                &format!("H{horizon} predictive-mean RMSE improvement >= 2%"),
                ratio <= 0.98,
                ratio,
                0.98,
                Some(interval),
            );
        }
        decision.add(
            &format!("H{horizon} RMSE regression <= 1%"),
            ratio <= 1.01,
            ratio,
            1.01,
            Some(interval),
        );
        let candidate_tv = candidate
            .iter()
            .map(|e| e.horizons[h].pit_tv.unwrap())
            .sum::<f64>()
            / 3.0;
        let baseline_tv = baseline
            .iter()
            .map(|e| e.horizons[h].pit_tv.unwrap())
            .sum::<f64>()
            / 3.0;
        decision.add(
            &format!("H{horizon} randomized PIT TV does not regress"),
            candidate_tv <= baseline_tv,
            candidate_tv - baseline_tv,
            0.0,
            None,
        );
        for (i, nominal) in [0.5, 0.8, 0.9].into_iter().enumerate() {
            let coverage = candidate
                .iter()
                .map(|e| e.horizons[h].coverage.unwrap()[i])
                .sum::<f64>()
                / 3.0;
            decision.add(
                &format!(
                    "H{horizon} {:.0}% coverage within 2 percentage points",
                    nominal * 100.0
                ),
                (coverage - nominal).abs() <= 0.02000000000001,
                coverage,
                nominal,
                None,
            );
        }
        for (i, tail) in ["lower", "upper"].into_iter().enumerate() {
            let rate = candidate
                .iter()
                .map(|e| e.horizons[h].tails.unwrap()[i])
                .sum::<f64>()
                / 3.0;
            decision.add(
                &format!("H{horizon} {tail} 5% tail between 4% and 6%"),
                (0.04..=0.06).contains(&rate),
                rate,
                0.05,
                None,
            );
        }
    }
    let (delta, interval) = bootstrap(
        &paired_blocks(&candidate, &baseline, Metric::Brier, None),
        false,
        false,
    );
    decision.add(
        "direction Brier improves",
        delta < 0.0,
        delta,
        0.0,
        Some(interval),
    );
    for (i, &seed) in FROZEN_SEEDS.iter().enumerate() {
        let (delta, interval) = bootstrap(
            &paired_blocks(&[candidate[i]], &[baseline[i]], Metric::Nll, None),
            false,
            false,
        );
        decision.add(
            &format!("seed {seed} has no resolved NLL regression"),
            interval[0] <= 0.0,
            delta,
            0.0,
            Some(interval),
        );
        let (candidate_latency, baseline_latency) =
            (candidate[i].latency_p95_ms, baseline[i].latency_p95_ms);
        let ratio = candidate_latency
            .zip(baseline_latency)
            .map(|(a, b)| a / b)
            .filter(|r| r.is_finite())
            .unwrap_or(f64::INFINITY);
        let measured = candidate[i].latency_samples >= 100 && baseline[i].latency_samples >= 100;
        decision.add(
            &format!("seed {seed} measured p95 latency <= 1.25x"),
            measured && ratio <= 1.25,
            ratio,
            1.25,
            None,
        );
        let memory_ratio = candidate[i]
            .peak_device_memory_bytes
            .zip(baseline[i].peak_device_memory_bytes)
            .filter(|(_, b)| *b > 0)
            .map(|(a, b)| a as f64 / b as f64)
            .unwrap_or(f64::INFINITY);
        decision.add(
            &format!("seed {seed} measured peak memory <= baseline"),
            memory_ratio <= 1.0,
            memory_ratio,
            1.0,
            None,
        );
    }
    decision.promoted = decision.checks.iter().all(|check| check.passed);
    Ok(decision)
}

/// The asymmetric bridge must also earn its complexity against the matched PatchTST arm.
pub fn assess_with_patchtst(
    candidate: &[Evaluation],
    baseline: &[Evaluation],
    patchtst: &[Evaluation],
) -> Result<GateDecision> {
    let mut decision = assess(candidate, baseline)?;
    let (candidate, patchtst) = validate_pairing(candidate, patchtst, "PatchTst")?;
    for (metric, name) in [
        (Metric::Nll, "bridge NLL does not lose to PatchTST"),
        (Metric::Crps, "bridge CRPS does not lose to PatchTST"),
    ] {
        let (delta, interval) = bootstrap(
            &paired_blocks(&candidate, &patchtst, metric, None),
            false,
            false,
        );
        decision.add(name, delta <= 0.0, delta, 0.0, Some(interval));
    }
    decision.promoted = decision.checks.iter().all(|check| check.passed);
    Ok(decision)
}

impl GateDecision {
    fn add(
        &mut self,
        name: &str,
        passed: bool,
        value: f64,
        threshold: f64,
        confidence: Option<[f64; 2]>,
    ) {
        self.checks.push(GateCheck {
            name: name.to_owned(),
            passed,
            value,
            threshold,
            confidence,
        });
    }
}

fn validate_pairing<'a>(
    candidate: &'a [Evaluation],
    baseline: &'a [Evaluation],
    expected_baseline: &str,
) -> Result<(Vec<&'a Evaluation>, Vec<&'a Evaluation>)> {
    ensure!(
        candidate.len() == 3 && baseline.len() == 3,
        "promotion requires exactly three frozen initialization seeds per model"
    );
    let ordered = |evaluations: &'a [Evaluation]| -> Result<Vec<&'a Evaluation>> {
        FROZEN_SEEDS
            .iter()
            .map(|seed| {
                let matching: Vec<_> = evaluations
                    .iter()
                    .filter(|e| e.training_seed == *seed)
                    .collect();
                ensure!(
                    matching.len() == 1,
                    "expected exactly one evaluation for initialization seed {seed}"
                );
                Ok(matching[0])
            })
            .collect()
    };
    let candidate = ordered(candidate)?;
    let baseline = ordered(baseline)?;
    let reference = candidate[0];
    ensure!(
        reference.model_kind == "SingleTickerTimeXer",
        "replacement candidate must be SingleTickerTimeXer"
    );
    ensure!(
        baseline[0].model_kind == expected_baseline,
        "replacement gates require the compliant {expected_baseline} baseline"
    );
    for (index, (a, b)) in candidate.iter().zip(&baseline).enumerate() {
        ensure!(
            a.model_kind == reference.model_kind && b.model_kind == baseline[0].model_kind,
            "mixed model families across seeds"
        );
        for evaluation in [*a, *b] {
            ensure!(
                evaluation.scoring_contract == SCORING_CONTRACT,
                "scoring contracts differ"
            );
            ensure!(
                evaluation.training_protocol_sha256.is_some()
                    && evaluation.training_protocol_sha256 == reference.training_protocol_sha256,
                "training exposure/selection protocols differ"
            );
            ensure!(
                evaluation.schema == EVIDENCE_SCHEMA,
                "unsupported evaluation schema"
            );
            ensure!(
                evaluation.frozen,
                "checkpoint for seed {} is not frozen",
                FROZEN_SEEDS[index]
            );
            ensure!(
                evaluation.checkpoint_sha256.as_ref().is_some_and(
                    |hash| hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit())
                ),
                "missing checkpoint authentication"
            );
            ensure!(
                !evaluation.identity.is_empty() && evaluation.identity == reference.identity,
                "different ticker/data/support/split identities"
            );
            ensure!(
                evaluation.split == reference.split,
                "comparison splits differ"
            );
            ensure!(
                evaluation.split == "Validation" || evaluation.split == "Test",
                "promotion evidence must use validation or unlocked terminal test"
            );
            ensure!(
                evaluation.horizons.len() == 6
                    && evaluation
                        .horizons
                        .iter()
                        .zip(HORIZONS)
                        .all(|(metric, h)| metric.horizon == h),
                "horizon contract differs"
            );
            ensure!(
                evaluation.origins.len() == reference.origins.len()
                    && !evaluation.origins.is_empty(),
                "origin sets differ"
            );
            ensure!(evaluation.device == reference.device && evaluation.precision == reference.precision && evaluation.evaluation_batch_size == reference.evaluation_batch_size, "resource measurements require identical device, precision and evaluation batch size");
            ensure!(
                evaluation
                    .origins
                    .windows(2)
                    .all(|pair| pair[0].origin < pair[1].origin
                        && pair[0].timestamp < pair[1].timestamp),
                "origins must be strictly chronological and unique"
            );
            let mut recomputed = evaluation.clone();
            recomputed.summarize()?;
            ensure!(
                postcard::to_stdvec(&recomputed.horizons)?
                    == postcard::to_stdvec(&evaluation.horizons)?,
                "cached evaluation summaries differ from paired evidence"
            );
            for (row, original) in evaluation.origins.iter().zip(&reference.origins) {
                ensure!(
                    row.origin == original.origin
                        && row.timestamp == original.timestamp
                        && row.target == original.target,
                    "evaluation origins or targets differ"
                );
                ensure!(
                    row.nll.is_some()
                        && row.crps.is_some()
                        && row.brier.is_some()
                        && row.pit.is_some()
                        && row.coverage.is_some()
                        && row.tails.is_some(),
                    "point forecasts cannot supply probabilistic promotion evidence"
                );
                ensure!(
                    row.nll
                        .unwrap()
                        .iter()
                        .chain(row.crps.unwrap().iter())
                        .chain(row.squared_error.iter())
                        .chain(row.brier.unwrap().iter())
                        .all(|v| v.is_finite()),
                    "nonfinite paired scores"
                );
            }
            for metric in &evaluation.horizons {
                ensure!(
                    metric.pit_tv.is_some_and(f64::is_finite)
                        && metric.coverage.is_some()
                        && metric.tails.is_some(),
                    "missing calibration diagnostics"
                );
            }
        }
    }
    Ok((candidate, baseline))
}

fn score(row: &OriginScores, metric: Metric, horizon: Option<usize>) -> f64 {
    let values = match metric {
        Metric::Nll => row.nll.unwrap(),
        Metric::Crps => row.crps.unwrap(),
        Metric::SquaredError => row.squared_error,
        Metric::Brier => row.brier.unwrap(),
    };
    match horizon {
        Some(h) => values[h],
        None => values.iter().sum::<f64>() / 6.0,
    }
}

fn calendar_week(timestamp_millis: i64) -> i64 {
    (timestamp_millis.div_euclid(1_000) + 3 * 86_400).div_euclid(7 * 86_400)
}

fn paired_blocks(
    candidate: &[&Evaluation],
    baseline: &[&Evaluation],
    metric: Metric,
    horizon: Option<usize>,
) -> Vec<PairedBlock> {
    let mut blocks = BTreeMap::<i64, PairedBlock>::new();
    for (a, b) in candidate.iter().zip(baseline) {
        for (left, right) in a.origins.iter().zip(&b.origins) {
            // Unix epoch was Thursday; this offset aligns weeks to Monday UTC.
            let week = calendar_week(left.timestamp);
            let block = blocks.entry(week).or_insert(PairedBlock {
                candidate: 0.0,
                baseline: 0.0,
                count: 0,
            });
            block.candidate += score(left, metric, horizon);
            block.baseline += score(right, metric, horizon);
            block.count += 1;
        }
    }
    blocks.into_values().collect()
}

fn bootstrap(blocks: &[PairedBlock], ratio: bool, square_root: bool) -> (f64, [f64; 2]) {
    let statistic = |candidate: f64, baseline: f64, count: usize| {
        let value = if ratio {
            if baseline > 0.0 {
                candidate / baseline
            } else {
                f64::INFINITY
            }
        } else {
            (candidate - baseline) / count as f64
        };
        if square_root {
            value.sqrt()
        } else {
            value
        }
    };
    let sum = blocks.iter().fold((0.0, 0.0, 0), |(a, b, n), block| {
        (a + block.candidate, b + block.baseline, n + block.count)
    });
    let point = statistic(sum.0, sum.1, sum.2);
    let mut rng = ChaCha8Rng::seed_from_u64(0x74696d657865725f);
    let mut draws = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    for _ in 0..BOOTSTRAP_REPLICATES {
        let mut sample = (0.0, 0.0, 0);
        for _ in blocks {
            let block = blocks[rng.random_range(0..blocks.len())];
            sample.0 += block.candidate;
            sample.1 += block.baseline;
            sample.2 += block.count;
        }
        draws.push(statistic(sample.0, sample.1, sample.2));
    }
    draws.sort_by(f64::total_cmp);
    (
        point,
        [
            draws[BOOTSTRAP_REPLICATES * 25 / 1000],
            draws[BOOTSTRAP_REPLICATES * 975 / 1000],
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(kind: &str, seed: u64, improvement: bool) -> Evaluation {
        let mut evaluation = Evaluation {
            schema: EVIDENCE_SCHEMA,
            scoring_contract: SCORING_CONTRACT.to_owned(),
            training_protocol_sha256: Some("protocol".to_owned()),
            selected_epoch: 1,
            selected_step: 1,
            identity: "same-dataset-supports-and-splits".to_owned(),
            split: "Validation".to_owned(),
            model_kind: kind.to_owned(),
            training_seed: seed,
            checkpoint_sha256: Some("a".repeat(64)),
            frozen: true,
            origins: Vec::new(),
            horizons: Vec::new(),
            latency_p95_ms: Some(1.0),
            latency_samples: 128,
            peak_device_memory_bytes: Some(100),
            device: "test".to_owned(),
            precision: "bfloat16".to_owned(),
            evaluation_batch_size: 8,
        };
        for index in 0..200 {
            let rank = index % 20;
            let coverage = [
                f64::from((5..15).contains(&rank)),
                f64::from((2..18).contains(&rank)),
                f64::from((1..19).contains(&rank)),
            ];
            let tails = [f64::from(rank == 0), f64::from(rank == 19)];
            evaluation.origins.push(OriginScores {
                origin: index,
                timestamp: 1_704_067_200_000 + index as i64 * 86_400_000,
                target: [0.0; 6],
                nll: Some([if improvement { 3.9 } else { 4.0 }; 6]),
                crps: Some([if improvement { 0.9 } else { 1.0 }; 6]),
                squared_error: [if improvement { 0.000081 } else { 0.0001 }; 6],
                standardized_squared_error: [if improvement { 0.81 } else { 1.0 }; 6],
                absolute_error: [0.01; 6],
                brier: Some([if improvement { 0.2 } else { 0.25 }; 6]),
                pit: Some([(rank as f64 + 0.5) / 20.0; 6]),
                coverage: Some([coverage; 6]),
                tails: Some([tails; 6]),
            });
        }
        evaluation.summarize().unwrap();
        evaluation
    }

    #[test]
    fn complete_paired_evidence_passes_but_missing_telemetry_and_exposure_fail() {
        let candidate: Vec<_> = FROZEN_SEEDS
            .iter()
            .map(|&seed| evidence("SingleTickerTimeXer", seed, true))
            .collect();
        let baseline: Vec<_> = FROZEN_SEEDS
            .iter()
            .map(|&seed| evidence("BarTrunk", seed, false))
            .collect();
        let patchtst: Vec<_> = FROZEN_SEEDS
            .iter()
            .map(|&seed| evidence("PatchTst", seed, false))
            .collect();
        let result = assess_with_patchtst(&candidate, &baseline, &patchtst).unwrap();
        assert!(
            result.promoted,
            "{:?}",
            result
                .checks
                .iter()
                .filter(|check| !check.passed)
                .collect::<Vec<_>>()
        );
        let mut unmeasured = candidate.clone();
        unmeasured[0].peak_device_memory_bytes = None;
        assert!(!assess(&unmeasured, &baseline).unwrap().promoted);
        let mut mismatched = candidate.clone();
        mismatched[0].training_protocol_sha256 = Some("different-budget".to_owned());
        assert!(assess(&mismatched, &baseline)
            .unwrap_err()
            .to_string()
            .contains("protocol"));
    }

    #[test]
    fn duplicated_origins_and_stale_summaries_cannot_promote() {
        let mut candidate: Vec<_> = FROZEN_SEEDS
            .iter()
            .map(|&seed| evidence("SingleTickerTimeXer", seed, true))
            .collect();
        let baseline: Vec<_> = FROZEN_SEEDS
            .iter()
            .map(|&seed| evidence("BarTrunk", seed, false))
            .collect();
        candidate[0].horizons[0].pit_tv = Some(0.5);
        assert!(assess(&candidate, &baseline)
            .unwrap_err()
            .to_string()
            .contains("summaries"));
        candidate[0].summarize().unwrap();
        candidate[0].origins[1] = candidate[0].origins[0].clone();
        candidate[0].summarize().unwrap();
        assert!(assess(&candidate, &baseline)
            .unwrap_err()
            .to_string()
            .contains("unique"));
    }

    #[test]
    fn epoch_milliseconds_keep_overlapping_labels_in_calendar_weeks() {
        let monday = 1_704_067_200_000_i64; // 2024-01-01 UTC.
        assert_eq!(calendar_week(monday), calendar_week(monday + 100 * 300_000));
        assert_eq!(
            calendar_week(monday),
            calendar_week(monday + 7 * 86_400_000 - 1)
        );
        assert_eq!(
            calendar_week(monday) + 1,
            calendar_week(monday + 7 * 86_400_000)
        );
        assert_eq!(calendar_week(monday) - 1, calendar_week(monday - 1));
    }

    #[test]
    fn paired_bootstrap_preserves_constant_improvement() {
        let blocks: Vec<_> = (1..=12)
            .map(|n| PairedBlock {
                candidate: n as f64 * 4.0,
                baseline: n as f64 * 4.03,
                count: n,
            })
            .collect();
        let (point, interval) = bootstrap(&blocks, false, false);
        assert!((point + 0.03).abs() < 1e-12);
        assert!(interval.iter().all(|bound| (*bound + 0.03).abs() < 1e-12));
    }

    #[test]
    fn rmse_gate_uses_ratio_of_root_mean_squared_errors() {
        let blocks = vec![
            PairedBlock {
                candidate: 81.0,
                baseline: 100.0,
                count: 10
            };
            8
        ];
        let (point, interval) = bootstrap(&blocks, true, true);
        assert!((point - 0.9).abs() < 1e-12);
        assert!(interval.iter().all(|bound| (*bound - 0.9).abs() < 1e-12));
    }

    #[test]
    fn incomplete_seed_evidence_cannot_promote() {
        assert!(assess(&[], &[])
            .unwrap_err()
            .to_string()
            .contains("three frozen"));
    }
}
