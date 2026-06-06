use std::fs;

use ibapi::market_data::historical::Bar;
use time::{Duration, OffsetDateTime};
use trading_bot_0::{
    data::universe::TARGET_UNIVERSE_TICKERS,
    genetic::{
        run_family_with_markets, DatasetBundle, GeneticFamily, MarketDataset, PriceReboundFamily,
        RsiReboundFamily, SessionPaths, TickerSet, TrainingConfig, TrendBreakoutFamily,
    },
    history::report::{read_report, ReportKind},
};

fn synthetic_bars(base: f64, trend: f64, amplitude: f64, phase: f64, len: usize) -> Vec<Bar> {
    let mut previous = base;
    (0..len)
        .map(|index| {
            let t = index as f64;
            let trend_component = trend * t;
            let wave = amplitude * ((t / 6.0) + phase).sin();
            let pullback = if index % 29 < 6 {
                -amplitude * 0.35 * (index % 6) as f64
            } else {
                0.0
            };
            let close = (base + trend_component + wave + pullback).max(5.0);
            let open = previous;
            let high = open.max(close) * 1.01;
            let low = open.min(close) * 0.99;
            previous = close;

            Bar {
                date: OffsetDateTime::UNIX_EPOCH + Duration::minutes(index as i64 * 5),
                open,
                high,
                low,
                close,
                volume: 1_000.0 + index as f64,
                wap: close,
                count: 1,
            }
        })
        .collect()
}

fn market(split: &str, tickers: &[&str], seed_offset: f64) -> MarketDataset {
    let bars = tickers
        .iter()
        .enumerate()
        .map(|(index, _)| {
            synthetic_bars(
                80.0 + seed_offset * 3.0 + index as f64 * 7.5,
                0.22 + index as f64 * 0.015,
                2.5 + seed_offset + index as f64 * 0.4,
                seed_offset + index as f64 * 0.7,
                120,
            )
        })
        .collect();

    MarketDataset::new(
        split.to_string(),
        tickers.iter().map(|ticker| ticker.to_string()).collect(),
        bars,
    )
}

fn fixture_bundle() -> DatasetBundle {
    DatasetBundle {
        train: market("train", &["TRN_A", "TRN_B", "TRN_C"], 0.0),
        validation: market("validation", &["VAL_A", "VAL_B", "VAL_C"], 0.8),
        test: market("test", &["TST_A", "TST_B", "TST_C"], 1.6),
    }
}

fn session_paths(name: &str) -> SessionPaths {
    let root = std::env::temp_dir().join(format!("trading_bot_0_{name}_{}", uuid::Uuid::new_v4()));
    let gens = root.join("gens");
    let weights = root.join("weights");
    fs::create_dir_all(&gens).unwrap();
    fs::create_dir_all(&weights).unwrap();
    SessionPaths {
        root: root.clone(),
        gens,
        weights,
        log_file: root.join("training.log"),
    }
}

fn base_config(family: GeneticFamily, seed: u64) -> TrainingConfig {
    TrainingConfig {
        family,
        generations: 4,
        population: 12,
        survivor_ratio: 4.0 / 12.0,
        heavy_report_every: 2,
        seed,
        mutation_entropy: 1.0,
    }
}

#[test]
fn trend_breakout_run_writes_checkpoint_and_tui_reports() {
    let paths = session_paths("trend_breakout");
    let summary = run_family_with_markets(
        &TrendBreakoutFamily,
        base_config(GeneticFamily::TrendBreakout, 11),
        fixture_bundle(),
        paths.clone(),
    )
    .expect("trend breakout run should succeed");

    assert!(summary.best_validation_generation < 4);
    assert!(summary.best_test_metrics.final_assets.is_finite());
    assert!(summary.best_test_metrics.final_assets > 0.0);

    let checkpoint = paths.weights.join("ga_best_validation.json");
    let summary_json = paths.root.join("ga_summary.json");
    let fitness_report = paths.gens.join("0/ga_fitness.report.bin");
    let mutation_entropy_report = paths.gens.join("0/ga_mutation_entropy.report.bin");
    let commissions_report = paths.gens.join("0/ga_total_commissions.report.bin");
    let test_assets_report = paths.gens.join("3/ga_test_assets.report.bin");
    let ticker_assets_report = paths.gens.join("3/TST_A/assets.report.bin");
    assert!(
        checkpoint.exists(),
        "missing checkpoint at {}",
        checkpoint.display()
    );
    assert!(
        summary_json.exists(),
        "missing summary at {}",
        summary_json.display()
    );
    assert!(
        fitness_report.exists(),
        "missing report at {}",
        fitness_report.display()
    );
    assert!(
        mutation_entropy_report.exists(),
        "missing report at {}",
        mutation_entropy_report.display()
    );
    assert!(
        commissions_report.exists(),
        "missing report at {}",
        commissions_report.display()
    );
    assert!(
        test_assets_report.exists(),
        "missing test assets report at {}",
        test_assets_report.display()
    );
    assert!(
        ticker_assets_report.exists(),
        "missing ticker assets report at {}",
        ticker_assets_report.display()
    );

    let report = read_report(fitness_report.to_string_lossy().as_ref()).unwrap();
    match report.kind {
        ReportKind::MultiLine { series } => {
            let labels = series
                .into_iter()
                .map(|series| series.label)
                .collect::<Vec<_>>();
            assert!(labels.contains(&"train".to_string()));
            assert!(labels.contains(&"validation".to_string()));
            assert!(labels.contains(&"best_validation".to_string()));
        }
        other => panic!("expected multiline fitness report, got {other:?}"),
    }

    let checkpoint_body = fs::read_to_string(&checkpoint).unwrap();
    assert!(checkpoint_body.contains("TrendBreakout"));

    let commissions = read_report(commissions_report.to_string_lossy().as_ref()).unwrap();
    match commissions.kind {
        ReportKind::MultiLine { series } => {
            let labels = series
                .iter()
                .map(|series| series.label.as_str())
                .collect::<Vec<_>>();
            assert!(labels.contains(&"train"));
            assert!(labels.contains(&"validation"));
            assert!(series
                .iter()
                .flat_map(|series| series.values.iter())
                .any(|value| *value > 0.0));
        }
        other => panic!("expected multiline commissions report, got {other:?}"),
    }

    let ticker_assets = read_report(ticker_assets_report.to_string_lossy().as_ref()).unwrap();
    match ticker_assets.kind {
        ReportKind::Assets {
            total,
            cash,
            positioned,
            benchmark,
        } => {
            let positioned = positioned.expect("expected positioned series");
            assert!(benchmark.is_none());
            assert!(total[0].abs() < 1e-6);
            assert!(cash[0].abs() < 1e-6);
            assert!(positioned[0].abs() < 1e-6);
        }
        other => panic!("expected assets report, got {other:?}"),
    }
}

#[test]
fn price_and_rsi_families_run_through_same_engine() {
    let families: Vec<(GeneticFamily, Box<dyn Fn() -> Result<(), String>>)> = vec![
        (
            GeneticFamily::PriceRebound,
            Box::new(|| {
                let paths = session_paths("price_rebound");
                let summary = run_family_with_markets(
                    &PriceReboundFamily,
                    base_config(GeneticFamily::PriceRebound, 21),
                    fixture_bundle(),
                    paths.clone(),
                )
                .map_err(|err| err.to_string())?;
                assert!(summary.best_test_metrics.final_assets > 0.0);
                assert!(paths.weights.join("ga_best_validation.json").exists());
                Ok(())
            }),
        ),
        (
            GeneticFamily::RsiRebound,
            Box::new(|| {
                let paths = session_paths("rsi_rebound");
                let summary = run_family_with_markets(
                    &RsiReboundFamily,
                    base_config(GeneticFamily::RsiRebound, 31),
                    fixture_bundle(),
                    paths.clone(),
                )
                .map_err(|err| err.to_string())?;
                assert!(summary.best_test_metrics.final_assets > 0.0);
                assert!(paths.weights.join("ga_best_validation.json").exists());
                Ok(())
            }),
        ),
    ];

    for (family, run) in families {
        run().unwrap_or_else(|err| panic!("{family:?} run failed: {err}"));
    }
}

#[test]
fn default_ticker_sets_are_disjoint_and_non_empty() {
    let train = TickerSet::Train.tickers();
    let validation = TickerSet::Validation.tickers();
    let test = TickerSet::Test.tickers();

    assert!(!train.is_empty());
    assert!(!validation.is_empty());
    assert!(!test.is_empty());

    for ticker in &train {
        assert!(!validation.contains(ticker));
        assert!(!test.contains(ticker));
    }
    for ticker in &validation {
        assert!(!test.contains(ticker));
    }

    assert_eq!(
        train.len() + validation.len() + test.len(),
        TARGET_UNIVERSE_TICKERS.len()
    );
    assert_eq!(train.len(), 68);
    assert_eq!(validation.len(), 16);
    assert_eq!(test.len(), 16);
}
