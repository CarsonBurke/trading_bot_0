//! Authenticated final-decision residual witnesses. Both feature families use one frozen
//! BF16 backbone pass. Ridge selection and refitting see training labels only.
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{ensure, Context, Result};
use clap::Args;
use shared::report::ReportSeries;
use tch::{Device, Kind, Tensor};

use super::{
    corpus::{Corpus, ResearchSamplePlan, WindowRef},
    jepa_eval::{
        chronological_split, dated, lines, ridge_with_refit, series, verify_membership, Dated, Fit,
        FIT_PANEL_LIMIT, SCORE_PANEL_LIMIT,
    },
    jepa_runner::load_panel,
    model::{CausalPatchModel, HorizonMean, ModelConfig},
    temporal_moments::{TemporalMoments, HORIZONS},
};
use crate::torch::{hashing::file_sha256, single_ticker_timexer::runner::cuda_device};

#[derive(Clone, Debug, Args)]
pub struct DiagnoseArgs {
    #[arg(long)]
    pub run_root: PathBuf,
    #[arg(long, default_value_os_t = crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    /// Native BF16 frozen inference batch size; ridge normal equations remain CUDA FP64.
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,
}

struct Panel {
    fit: Vec<Dated>,
    inner: Vec<Dated>,
    holdout: Vec<Dated>,
    validation: Vec<Dated>,
    incomplete: Vec<WindowRef>,
    outer_purged: Vec<Dated>,
    inner_purged: Vec<Dated>,
    validation_boundary: i64,
}

fn purge_before(rows: Vec<Dated>, first_decision: i64) -> (Vec<Dated>, Vec<Dated>) {
    rows.into_iter().partition(|row| row.reach < first_decision)
}

/// Timestamp ties stay together: the halves cannot share a decision clock.
fn chronological_boundary(rows: &[Dated]) -> Result<i64> {
    ensure!(
        rows.len() >= 2,
        "chronological scoring requires two dated rows"
    );
    let mut dates: Vec<_> = rows.iter().map(|r| r.origin).collect();
    dates.sort_unstable();
    let boundary = dates[dates.len() / 2];
    ensure!(
        dates[0] < boundary,
        "validation has no nonempty chronological halves"
    );
    Ok(boundary)
}

fn panel(config: &ModelConfig, corpus: &Corpus, plan: &ResearchSamplePlan) -> Result<Panel> {
    ensure!(
        config.seq_len as usize == corpus.contract.context
            && config.pred_len as usize == corpus.contract.pred_len
            && config.features == corpus.contract.features,
        "witness model/corpus geometry mismatch"
    );
    ensure!(
        !config.jepa_mode.enabled()
            && config.target_basis.is_identity()
            && !matches!(config.horizon_mean, HorizonMean::Increment { .. })
            && !config.future_calendar
            && config.pred_len >= HORIZONS[6],
        "temporal moment witnesses require causal forecast-only cumulative close geometry"
    );
    ensure!(
        !plan.probe_fit_refs.is_empty()
            && plan.probe_fit_refs.len() <= FIT_PANEL_LIMIT
            && plan.validation_refs.len() == SCORE_PANEL_LIMIT,
        "witness requires authenticated bounded training and complete 2048-row validation panels"
    );
    verify_membership(&plan.probe_fit_refs, &corpus.train_refs, "witness training")?;
    verify_membership(
        &plan.validation_refs,
        &corpus.validation_refs,
        "witness validation",
    )?;
    ensure!(
        plan.probe_fit_refs.len() == plan.manifest.probe_fit.origins.len()
            && plan.validation_refs.len() == plan.manifest.validation.origins.len(),
        "witness panel certificates do not cover every source"
    );
    let mut complete = Vec::new();
    let mut incomplete = Vec::new();
    // This is the corpus's authenticated FINAL-origin target prefix, not a new label rule
    // or the old JEPA probe's earlier within-context source.
    for (&reference, certificate) in plan
        .probe_fit_refs
        .iter()
        .zip(&plan.manifest.probe_fit.origins)
    {
        ensure!(
            reference.origin == certificate.origin,
            "training source certificate mismatch"
        );
        if certificate.valid_target_prefix >= HORIZONS[6] as usize {
            complete.push(reference);
        } else {
            incomplete.push(reference);
        }
    }
    ensure!(
        plan.manifest
            .validation
            .origins
            .iter()
            .all(|r| r.valid_target_prefix >= HORIZONS[6] as usize),
        "fixed validation lacks complete final-origin decision labels; do not silently change it"
    );
    let source = config.origins() - 1;
    let mut validation = dated(corpus, &plan.validation_refs, source, config, HORIZONS[6])?;
    validation.sort_by_key(|r| (r.origin, r.reference.ticker, r.reference.origin));
    let first_score = validation[0].origin;
    ensure!(
        plan.manifest.training_last_target_ms < first_score,
        "a scored final decision precedes or coincides with a dense model-training target"
    );
    let fit = dated(corpus, &complete, source, config, HORIZONS[6])?;
    ensure!(
        fit.iter()
            .all(|r| r.reach <= plan.manifest.training_last_target_ms),
        "a witness training target exceeds the authenticated training-label envelope"
    );
    let (mut fit, outer_purged) = purge_before(fit, first_score);
    fit.sort_by_key(|r| (r.origin, r.reference.ticker, r.reference.origin));
    let (inner, holdout, _) = chronological_split(&fit)?;
    let first_holdout = holdout.iter().map(|r| r.origin).min().unwrap();
    let inner_purged = fit
        .iter()
        .copied()
        .filter(|r| r.origin < first_holdout && r.reach >= first_holdout)
        .collect();
    let validation_boundary = chronological_boundary(&validation)?;
    Ok(Panel {
        fit,
        inner,
        holdout,
        validation,
        incomplete,
        outer_purged,
        inner_purged,
        validation_boundary,
    })
}

struct Cache {
    fixed: Tensor,
    state: Tensor,
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    sigma: Tensor,
}

fn cache(
    model: &CausalPatchModel,
    geometry: &TemporalMoments,
    corpus: &Corpus,
    rows: &[Dated],
    batch_size: usize,
    device: Device,
) -> Result<Cache> {
    let mut fixed = Vec::new();
    let mut state = Vec::new();
    let mut prediction = Vec::new();
    let mut target = Vec::new();
    let mut mask = Vec::new();
    let mut sigma = Vec::new();
    for chunk in rows.chunks(batch_size) {
        let refs: Vec<_> = chunk.iter().map(|r| r.reference).collect();
        let batch = corpus.batch(&refs, device)?;
        let stats = model.statistics(&batch);
        let contextual = model.backbone(&batch, &stats, false, true);
        ensure!(
            contextual.kind() == Kind::BFloat16,
            "frozen witness backbone must stay native BF16"
        );
        let head = model.head(&batch, &contextual, true);
        let final_stats = stats.last();
        let (labels, valid) = model.targets(&batch, &final_stats, true);
        let decisions = geometry.select(model, &head, &labels, &valid);
        ensure!(
            decisions.mask.eq(1.).all().int64_value(&[]) == 1
                && decisions.prediction.isfinite().all().int64_value(&[]) == 1
                && decisions.target.isfinite().all().int64_value(&[]) == 1,
            "authenticated complete decision source has invalid or nonfinite close labels/forecasts"
        );
        // Copy only final-source features, so cache parts do not retain full-context storage.
        fixed.push(
            geometry
                .instruments(&batch, &stats)
                .select(1, model.config().origins() - 1)
                .contiguous(),
        );
        state.push(contextual.squeeze_dim(1).to_kind(Kind::Float));
        prediction.push(decisions.prediction.squeeze_dim(1));
        target.push(decisions.target.squeeze_dim(1));
        mask.push(decisions.mask.squeeze_dim(1));
        sigma.push(final_stats.sigma.to_kind(Kind::Double));
    }
    Ok(Cache {
        fixed: Tensor::cat(&fixed, 0),
        state: Tensor::cat(&state, 0),
        prediction: Tensor::cat(&prediction, 0),
        target: Tensor::cat(&target, 0),
        mask: Tensor::cat(&mask, 0),
        sigma: Tensor::cat(&sigma, 0),
    })
}

fn indices(subset: &[Dated], population: &[Dated], device: Device) -> Tensor {
    let lookup: HashMap<_, _> = population
        .iter()
        .enumerate()
        .map(|(i, row)| ((row.reference.ticker, row.reference.origin), i as i64))
        .collect();
    Tensor::from_slice(
        &subset
            .iter()
            .map(|r| lookup[&(r.reference.ticker, r.reference.origin)])
            .collect::<Vec<_>>(),
    )
    .to_device(device)
}

fn counts(mask: &Tensor) -> Tensor {
    mask.to_kind(Kind::Double)
        .sum_dim_intlist([0i64].as_slice(), false, Kind::Double)
}

/// All arrays are actual close forecasts/labels in source sigma*sqrt(h) units. A ridge
/// predicts (target - original), so correction ADDS to the original decision forecast.
fn corrected(prediction: &Tensor, correction: &Tensor) -> Tensor {
    prediction.to_kind(Kind::Double) + correction
}

struct Metrics {
    suffix: &'static str,
    unit: &'static str,
    values: Vec<(&'static str, Tensor)>,
}

fn measurements(
    prediction: &Tensor,
    target: &Tensor,
    correction: &Tensor,
    mask: &Tensor,
    sigma: &Tensor,
    horizon_scale: &Tensor,
) -> Vec<Metrics> {
    let prediction = prediction.to_kind(Kind::Double);
    let target = target.to_kind(Kind::Double);
    let amended = corrected(&prediction, correction);
    let mask = mask.to_kind(Kind::Double);
    let count = counts(&mask);
    let sum =
        |value: Tensor| (value * &mask).sum_dim_intlist([0i64].as_slice(), false, Kind::Double);
    let mean = |value: Tensor| sum(value) / &count;
    let original_error = (&target - &prediction) * horizon_scale;
    let amended_error = (&target - &amended) * horizon_scale;
    let delta = correction * horizon_scale;
    let actual = &target * horizon_scale;
    let persistence = mean(actual.square());
    let original_mse = mean(original_error.square());
    let amended_mse = mean(amended_error.square());
    let delta_mean = mean(delta.shallow_clone());
    let residual_mean = mean(original_error.shallow_clone());
    let variance = mean(delta.square()) - delta_mean.square();
    let covariance = mean(&delta * &original_error) - &delta_mean * residual_mean;
    let direction = |p: &Tensor| {
        let bets = p.ne(0.).logical_and(&target.ne(0.)).to_kind(Kind::Double);
        let n = sum(bets);
        let hits = sum((p * &target).gt(0.).to_kind(Kind::Double));
        (hits / &n, n)
    };
    let (original_hit, original_bets) = direction(&prediction);
    let (amended_hit, amended_bets) = direction(&amended);
    let log_persistence = mean((&actual * sigma).square());
    let log_original = mean((&original_error * sigma).square());
    let log_amended = mean((&amended_error * sigma).square());
    vec![
        Metrics {
            suffix: "",
            unit: "MSE / matched persistence MSE",
            values: vec![
                (
                    "original neutral close MSE / persistence",
                    &original_mse / &persistence,
                ),
                (
                    "corrected neutral close MSE / persistence",
                    &amended_mse / &persistence,
                ),
                (
                    "secondary original log-return MSE / log-return persistence",
                    &log_original / &log_persistence,
                ),
                (
                    "secondary corrected log-return MSE / log-return persistence",
                    &log_amended / &log_persistence,
                ),
            ],
        },
        Metrics {
            suffix: "_direction",
            unit: "signed directional hit fraction",
            values: vec![
                (
                    "original directional hit (nonzero forecast and target)",
                    original_hit,
                ),
                (
                    "corrected directional hit (nonzero forecast and target)",
                    amended_hit,
                ),
            ],
        },
        Metrics {
            suffix: "_correction",
            unit: "moment / same close persistence MSE",
            values: vec![
                (
                    "close MSE reduction / persistence (positive improves)",
                    (&original_mse - &amended_mse) / &persistence,
                ),
                (
                    "correction squared mean / persistence",
                    delta_mean.square() / &persistence,
                ),
                ("correction variance / persistence", variance / &persistence),
                (
                    "correction covariance with original residual / persistence",
                    covariance / &persistence,
                ),
                (
                    "correction second moment / persistence",
                    mean(delta.square()) / &persistence,
                ),
                (
                    "correction residual cross moment / persistence",
                    mean(delta * original_error) / &persistence,
                ),
            ],
        },
        Metrics {
            suffix: "_error",
            unit: "source-sigma-scaled neutral close MSE",
            values: vec![
                ("original close MSE (sigma squared)", original_mse),
                ("corrected close MSE (sigma squared)", amended_mse),
                (
                    "close-anchored persistence MSE (sigma squared)",
                    persistence,
                ),
            ],
        },
        Metrics {
            suffix: "_log_error",
            unit: "neutral log-return squared (secondary)",
            values: vec![
                ("original neutral log-return MSE", log_original),
                ("corrected neutral log-return MSE", log_amended),
                ("neutral log-return persistence MSE", log_persistence),
            ],
        },
        Metrics {
            suffix: "_population",
            unit: "decision rows",
            values: vec![
                ("valid decision rows", count),
                ("original nonzero directional rows", original_bets),
                ("corrected nonzero directional rows", amended_bets),
            ],
        },
    ]
}

fn report(
    output: &Path,
    base: &str,
    title: String,
    unit: &str,
    metrics: Vec<(&str, Tensor)>,
) -> Result<()> {
    let values = Tensor::stack(
        &metrics
            .iter()
            .map(|(_, t)| t.shallow_clone())
            .collect::<Vec<_>>(),
        0,
    )
    .to_device(Device::Cpu);
    let tracks: Vec<ReportSeries> = metrics
        .iter()
        .enumerate()
        .map(|(i, (label, _))| {
            series(
                *label,
                (0..HORIZONS.len()).map(|h| values.double_value(&[i as i64, h as i64])),
            )
        })
        .collect();
    lines(
        output,
        base,
        title,
        unit,
        "observed bars ahead",
        &HORIZONS.map(|h| h as u64),
        tracks,
    )
}

fn dated_provenance(rows: &[Dated]) -> serde_json::Value {
    serde_json::json!(rows
        .iter()
        .map(|r| serde_json::json!({
            "ticker_index": r.reference.ticker, "origin_ordinal": r.reference.origin,
            "decision_ms": r.origin, "last_target_ms": r.reach,
        }))
        .collect::<Vec<_>>())
}

pub fn diagnose(args: DiagnoseArgs) -> Result<()> {
    ensure!(
        args.batch_size > 0 && !args.output.exists(),
        "witness requires a positive batch size and fresh output path"
    );
    let device = cuda_device()?;
    let (checkpoint, corpus, plan, _) = load_panel(&args.run_root, &args.data_dir, device)?;
    let panel = panel(&checkpoint.model, &corpus, &plan)?;
    let weights_sha256 = file_sha256(args.run_root.join("weights/jepa.safetensors"))?;
    let (_store, model) = checkpoint.load_model(&args.run_root, device)?;
    let _guard = tch::no_grad_guard();
    let geometry = TemporalMoments::new(model.config(), device);
    let training = cache(
        &model,
        &geometry,
        &corpus,
        &panel.fit,
        args.batch_size,
        device,
    )?;
    let inner_index = indices(&panel.inner, &panel.fit, device);
    let holdout_index = indices(&panel.holdout, &panel.fit, device);
    let residual =
        training.target.to_kind(Kind::Double) - training.prediction.to_kind(Kind::Double);
    let inner_y = residual.index_select(0, &inner_index);
    let holdout_y = residual.index_select(0, &holdout_index);
    let inner_mask = training.mask.index_select(0, &inner_index);
    let holdout_mask = training.mask.index_select(0, &holdout_index);
    // Both fits are finalized before a validation label or feature is even cached.
    let mut fits: Vec<(&str, Fit)> = Vec::new();
    for (family, features) in [("fixed", &training.fixed), ("state", &training.state)] {
        let fit = ridge_with_refit(
            &features.index_select(0, &inner_index),
            &inner_y,
            &inner_mask,
            &features.index_select(0, &holdout_index),
            &holdout_y,
            &holdout_mask,
            Some((features, &residual, &training.mask)),
        )
        .with_context(|| format!("fitting {family} training-only residual witness"))?;
        fits.push((family, fit));
    }
    let inner_count = counts(&inner_mask);
    let holdout_count = counts(&holdout_mask);
    let refit_count = counts(&training.mask);
    drop(training);
    let validation = cache(
        &model,
        &geometry,
        &corpus,
        &panel.validation,
        args.batch_size,
        device,
    )?;
    let horizon_scale = Tensor::from_slice(&HORIZONS.map(|h| (h as f64).sqrt())).to_device(device);
    let early_rows = panel
        .validation
        .partition_point(|r| r.origin < panel.validation_boundary) as i64;
    let total_rows = panel.validation.len() as i64;
    fs::create_dir(&args.output)?;
    for (family, fit) in &fits {
        let features = if *family == "fixed" {
            &validation.fixed
        } else {
            &validation.state
        };
        let correction = fit.predict(features);
        ensure!(
            correction.isfinite().all().int64_value(&[]) == 1,
            "nonfinite frozen correction"
        );
        let constant =
            |n: usize| Tensor::full([HORIZONS.len() as i64], n as f64, (Kind::Double, device));
        let diagnostics = fit
            .diagnostics
            .as_ref()
            .context("missing train-only ridge diagnostics")?;
        let title = format!("Frozen {family} residual witness | training-only selection and all-eligible-training refit");
        report(
            &args.output,
            &format!("timexer_segment_temporal_moment_witness_fit_{family}"),
            title.clone(),
            "ridge penalty / mean centered feature eigenvalue",
            vec![("chosen relative ridge penalty", fit.penalty.shallow_clone())],
        )?;
        report(
            &args.output,
            &format!("timexer_segment_temporal_moment_witness_fit_gain_{family}"),
            title.clone(),
            "fraction original residual MSE removed (positive improves)",
            vec![
                (
                    "inner training achieved correction gain (selection coefficients)",
                    diagnostics.gains.get(0),
                ),
                (
                    "later training holdout achieved correction gain (selection coefficients)",
                    diagnostics.gains.get(1),
                ),
                (
                    "all eligible training achieved correction gain (refitted coefficients)",
                    diagnostics.gains.get(2),
                ),
            ],
        )?;
        report(
            &args.output,
            &format!("timexer_segment_temporal_moment_witness_spectrum_{family}"),
            title.clone(),
            "uncentered feature second-moment energy",
            vec![
                (
                    "eligible-train second-moment trace",
                    diagnostics.spectrum.get(0),
                ),
                (
                    "eligible-train second-moment top eigenvalue",
                    diagnostics.spectrum.get(1),
                ),
                (
                    "eligible-train squared mean-feature norm",
                    diagnostics.spectrum.get(3),
                ),
            ],
        )?;
        report(
            &args.output,
            &format!("timexer_segment_temporal_moment_witness_spectrum_rank_{family}"),
            title.clone(),
            "feature dimensions",
            vec![
                (
                    "eligible-train second-moment participation rank",
                    diagnostics.spectrum.get(2),
                ),
                (
                    "frozen feature dimension",
                    constant(features.size()[1] as usize),
                ),
            ],
        )?;
        report(
            &args.output,
            &format!("timexer_segment_temporal_moment_witness_population_{family}_fit"),
            title,
            "decision rows",
            vec![
                ("inner training valid rows", inner_count.shallow_clone()),
                (
                    "later training selection valid rows",
                    holdout_count.shallow_clone(),
                ),
                (
                    "final training refit valid rows",
                    refit_count.shallow_clone(),
                ),
                ("unchanged validation valid rows", counts(&validation.mask)),
                (
                    "authenticated fit candidate rows",
                    constant(plan.probe_fit_refs.len()),
                ),
                (
                    "fit rows excluded for incomplete final-source targets",
                    constant(panel.incomplete.len()),
                ),
                (
                    "fit rows excluded at validation chronology boundary",
                    constant(panel.outer_purged.len()),
                ),
                (
                    "inner target-purged rows excluded during selection; INCLUDED in final refit",
                    constant(panel.inner_purged.len()),
                ),
            ],
        )?;
        for (scope, start, count) in [
            ("pooled", 0, total_rows),
            ("early", 0, early_rows),
            ("late", early_rows, total_rows - early_rows),
        ] {
            let take = |t: &Tensor| t.narrow(0, start, count);
            let charts = measurements(
                &take(&validation.prediction),
                &take(&validation.target),
                &take(&correction),
                &take(&validation.mask),
                &take(&validation.sigma),
                &horizon_scale,
            );
            for chart in charts {
                report(&args.output, &format!("timexer_segment_temporal_moment_witness{}_{family}_{scope}", chart.suffix),
                    format!("Frozen {family} residual witness | {scope} fixed validation | actual final decision; no validation fitting"),
                    chart.unit, chart.values)?;
            }
        }
    }
    // Identity and chronology only: every measured scalar lives in registered report.bin files.
    let provenance = serde_json::json!({
        "schema": "temporal-moment-final-decision-residual-witness-v1",
        "run_root": args.run_root,
        "checkpoint": checkpoint.identity()?,
        "checkpoint_weights_sha256": weights_sha256,
        "evaluation_executable_sha256": file_sha256(std::env::current_exe()?)?,
        "corpus_sha256": plan.manifest.corpus_sha256,
        "sample_plan_sha256": checkpoint.sample_plan_sha256,
        "authenticated_fit_origins_sha256": plan.manifest.probe_fit.origins_sha256,
        "authenticated_fit_target_mask_sha256": plan.manifest.probe_fit.target_mask_sha256,
        "validation_origins_sha256": plan.manifest.validation.origins_sha256,
        "validation_target_mask_sha256": plan.manifest.validation.target_mask_sha256,
        "model_training_last_target_ms": plan.manifest.training_last_target_ms,
        "source_patch": checkpoint.model.origins() - 1,
        "source_bar": checkpoint.model.seq_len - 1,
        "source_lookback_bars": 0,
        "horizons_bars": HORIZONS,
        "inference_batch_size": args.batch_size,
        "feature_families": ["shared TemporalMoments fixed instruments, seed20260920", "frozen final causal backbone state, before actual forecast head"],
        "inference": "one existing backbone and head forward per row; native BF16; source statistics and shared decision geometry FP32; CUDA ridge FP64",
        "fit_target": "shared decision target minus unchanged forecast, source sigma*sqrt(h) units; add fitted residual to original forecast",
        "selection": "same CUDA ridge penalty grid as jepa_eval; per-horizon later training-only last fifth; strict target-reach purge and timestamp ties kept together; intercept fitted from training only",
        "refit": "all eligible authenticated training fit rows, including the inner purge gap, after freezing each penalty; all labels strictly precede every scored decision",
        "validation": "entire original authenticated validation panel, final source; chronological reorder only; no validation tuning, fitting, coefficient selection, or test access",
        "validation_half_boundary_ms": panel.validation_boundary,
        "half_rule": "early decision_ms < boundary, late decision_ms >= boundary; boundary is sorted median decision timestamp; same fitted coefficients in all scopes",
        "primary_mse": "source-sigma-scaled neutral close MSE / zero-neutral-return persistence; identical per-horizon ratio to sigma*sqrt(h) residual geometry; actual unscaled neutral log-return errors also reported separately",
        "direction": "signed hit conditional on both forecast and target nonzero, as matched accuracy panel; denominator reported separately; no bets yields undefined NaN",
        "correction_moments": "population-centered variance and covariance with original residual, same valid rows, all divided by the SAME sigma-scaled close persistence MSE; uncentered second/cross moments also reported so normalized MSE reduction = 2 cross moment - second moment",
        "feature_spectrum": "CUDA FP64 eigenvalues of UNcentered eligible-training E[xx^T], reusing ridge normal equations; trace, top eigenvalue, trace squared / sum eigenvalue squared, and squared norm of E[x]; includes constant mode, no whitening or validation fitting",
        "fit_gains": "one minus residual-correction error / original residual second moment; inner and later-training holdout use selected inner-fit coefficients; final eligible training uses refitted coefficients; not held-out generalization evidence",
        "eligible_refit_sources": dated_provenance(&panel.fit),
        "inner_training_sources": dated_provenance(&panel.inner),
        "selection_training_sources": dated_provenance(&panel.holdout),
        "selection_purged_but_refitted_sources": dated_provenance(&panel.inner_purged),
        "outer_chronology_excluded_sources": dated_provenance(&panel.outer_purged),
        "incomplete_final_target_excluded_sources": panel.incomplete.iter().map(|r| serde_json::json!({
            "ticker_index": r.ticker, "origin_ordinal": r.origin,
            "decision_ms": corpus.ticker(*r).timestamp(r.origin),
            "reason": "authenticated final-source valid_target_prefix shorter than longest decision horizon"
        })).collect::<Vec<_>>(),
        "validation_scoring_order": dated_provenance(&panel.validation),
        "updates": "none; model store frozen; diagnostic coefficients not attached to model or saved as a deployable checkpoint",
        "interpretation": "decodable missed conditional-mean signal only; neither deployment gain nor evidence of information absence; correlated market rows are not independent significance units"
    });
    fs::write(
        args.output.join("witness-provenance.json"),
        serde_json::to_vec_pretty(&provenance)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(origin: i64, reach: i64) -> Dated {
        Dated {
            reference: WindowRef {
                ticker: 0,
                origin: origin as usize,
            },
            origin,
            reach,
        }
    }

    #[test]
    fn target_reaching_scored_decision_is_excluded_even_when_source_is_earlier() {
        let (eligible, excluded) = purge_before(vec![row(10, 19), row(11, 20), row(12, 21)], 20);
        assert_eq!(eligible.iter().map(|r| r.origin).collect::<Vec<_>>(), [10]);
        assert_eq!(
            excluded.iter().map(|r| r.origin).collect::<Vec<_>>(),
            [11, 12]
        );
    }

    #[test]
    fn chronological_halves_keep_tied_decisions_together_and_refuse_one_clock() {
        let rows = [row(10, 12), row(20, 22), row(20, 23), row(30, 32)];
        let boundary = chronological_boundary(&rows).unwrap();
        assert_eq!(rows.iter().filter(|r| r.origin < boundary).count(), 1);
        assert_eq!(rows.iter().filter(|r| r.origin >= boundary).count(), 3);
        assert!(chronological_boundary(&[row(10, 12), row(10, 13)]).is_err());
    }

    #[test]
    fn residual_correction_adds_to_forecast_and_reports_actual_error_decomposition() {
        let prediction = Tensor::from_slice(&[1f64, -1., 0., 2.]).reshape([4, 1]);
        let target = Tensor::from_slice(&[2f64, -2., 1., 0.]).reshape([4, 1]);
        let correction = &target - &prediction;
        let values = measurements(
            &prediction,
            &target,
            &correction,
            &Tensor::ones([4, 1], (Kind::Double, Device::Cpu)),
            &Tensor::full([4, 1], 0.5, (Kind::Double, Device::Cpu)),
            &Tensor::from_slice(&[2f64]),
        );
        let metric = |name| {
            values
                .iter()
                .flat_map(|chart| chart.values.iter())
                .find(|(label, _)| *label == name)
                .unwrap()
                .1
                .double_value(&[0])
        };
        assert_eq!(metric("corrected neutral close MSE / persistence"), 0.);
        assert_eq!(metric("corrected close MSE (sigma squared)"), 0.);
        assert_eq!(metric("original close MSE (sigma squared)"), 7.);
        assert_eq!(metric("original neutral log-return MSE"), 1.75);
        assert_eq!(metric("original nonzero directional rows"), 2.);
        assert_eq!(metric("corrected nonzero directional rows"), 3.);
        assert_eq!(
            metric("corrected directional hit (nonzero forecast and target)"),
            1.
        );
        assert!(
            (metric("original neutral close MSE / persistence")
                - 2. * metric("correction residual cross moment / persistence")
                + metric("correction second moment / persistence"))
            .abs()
                < 1e-12
        );
    }
}
