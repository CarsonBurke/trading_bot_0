//! Paired final-origin forecast accuracy. No fitting, latent loss, or terminal-test access.
use super::{
    jepa_eval::{lines, series},
    jepa_runner::{self, Checkpoint},
    model::{ModelConfig, ScaleCoupling},
    reports::{horizon_track, HorizonPoint, DECISION_HORIZONS},
    runner::{self, Evaluation, Manifest},
    supervision::HorizonDecimation,
};
use crate::torch::{hashing::file_sha256, single_ticker_timexer::runner::cuda_device};
use anyhow::{ensure, Context, Result};
use clap::Args;
use serde_json::{json, Value};
use std::{collections::HashSet, fs, path::PathBuf, str::FromStr};

#[derive(Clone, Debug)]
pub struct NamedPath {
    name: String,
    path: PathBuf,
}
impl FromStr for NamedPath {
    type Err = String;
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let (name, path) = value.split_once('=').ok_or("expected NAME=PATH")?;
        if name.trim().is_empty() || name.chars().any(char::is_control) || path.is_empty() {
            return Err("expected a nonempty printable NAME and PATH".into());
        }
        Ok(Self {
            name: name.to_owned(),
            path: path.into(),
        })
    }
}

#[derive(Clone, Debug, Args)]
pub struct CompareArgs {
    /// Authenticated forecast-only, causal decoupled+lattice run defining the fixed panels.
    #[arg(long)]
    pub reference_run: PathBuf,
    /// Named causal fixed-endpoint research run; repeat for each candidate.
    #[arg(long)]
    pub research_run: Vec<NamedPath>,
    /// Named historical calibrated checkpoint directory; always a NONMATCHED reference.
    #[arg(long)]
    pub legacy_checkpoint: Vec<NamedPath>,
    #[arg(long, default_value_os_t = crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,
}

fn compatible_geometry(reference: &ModelConfig, candidate: &ModelConfig) -> Result<()> {
    ensure!(
        reference.seq_len == candidate.seq_len
            && reference.pred_len == candidate.pred_len
            && reference.patch_len == candidate.patch_len
            && reference.min_history == candidate.min_history
            && reference.features == candidate.features,
        "checkpoint source/target geometry differs from the reference"
    );
    Ok(())
}

fn matched_research(reference: &Checkpoint, candidate: &Checkpoint) -> Result<()> {
    ensure!(
        reference.data == candidate.data,
        "research corpus differs from reference"
    );
    compatible_geometry(&reference.model, &candidate.model)?;
    ensure!(
        reference.sample_plan_sha256 == candidate.sample_plan_sha256
            && reference.cross_section_sha256 == candidate.cross_section_sha256
            && reference.seed == candidate.seed
            && reference.validation_rows == candidate.validation_rows
            && reference.probe_fit_rows == candidate.probe_fit_rows
            && reference.source_lookback_bars == candidate.source_lookback_bars,
        "research fixed panels differ from reference"
    );
    // These are the intentionally varied treatments, not grounds for changing eligibility.
    let mut model = candidate.model.clone();
    model.jepa_mode = reference.model.jepa_mode;
    model.jepa = reference.model.jepa.clone();
    model.scale_coupling = reference.model.scale_coupling;
    model.horizon_decimation = reference.model.horizon_decimation;
    ensure!(
        model == reference.model,
        "research model differs beyond the declared objective/recipe treatments"
    );
    let (reference_identity, candidate_identity) = (reference.identity()?, candidate.identity()?);
    for field in [
        "seed",
        "batch_size",
        "completed_steps",
        "schedule_budget",
        "base_learning_rate",
        "optimizer",
        "scalar_lr_mult",
        "mlp_down_lr",
    ] {
        ensure!(
            reference_identity[field] == candidate_identity[field],
            "research training protocol differs at {field}"
        );
    }
    ensure!(
        candidate.completed_steps == candidate.schedule_budget,
        "research checkpoint is not its fixed endpoint"
    );
    Ok(())
}

fn calibration_precedes_panels(
    last_target_ms: i64,
    validation_first_ms: i64,
    cross_first_ms: i64,
) -> Result<()> {
    ensure!(
        last_target_ms < validation_first_ms && last_target_ms < cross_first_ms,
        "legacy calibration target reach {last_target_ms} is not strictly before both scored final-origin panels ({validation_first_ms}, {cross_first_ms})"
    );
    Ok(())
}

struct Metric {
    base: &'static str,
    title: &'static str,
    unit: &'static str,
    value: fn(&HorizonPoint, &Evaluation, &Evaluation, usize) -> f64,
    anchor: Option<(&'static str, f64)>,
}
const METRICS: &[Metric] = &[
    Metric { base: "close_ratio", title: "Neutral close MSE / close persistence MSE; lower is better", unit: "MSE ratio", value: |p, _, _, _| p.close_ratio, anchor: Some(("zero-return persistence", 1.)) },
    Metric { base: "delayed_ratio", title: "Neutral delayed close MSE / delayed persistence MSE; t+1 to t+h, h1 undefined", unit: "MSE ratio", value: |_, e, _, i| e.trading.delayed_mse_ratio[i], anchor: Some(("delayed zero-return persistence (h>1)", 1.)) },
    Metric { base: "close_hit", title: "Neutral close directional hit; conditional on NONZERO forecast AND target", unit: "hit fraction", value: |_, e, _, i| e.trading.close_hit_rate[i], anchor: None },
    Metric { base: "delayed_hit", title: "Neutral delayed directional hit; conditional on NONZERO forecast AND target; h1 undefined", unit: "hit fraction", value: |_, e, _, i| e.trading.delayed_hit_rate[i], anchor: None },
    Metric { base: "pearson", title: "Signed pooled neutral-close Pearson; no absolute value or squaring", unit: "signed correlation", value: |p, _, _, _| p.pooled_pearson, anchor: Some(("zero correlation", 0.)) },
    Metric { base: "cross_ic", title: "Signed cross-sectional neutral-close IC; equal mean over eligible timestamps", unit: "signed correlation", value: |_, _, c, i| c.trading.cross_sectional_ic[i], anchor: Some(("zero correlation", 0.)) },
    Metric { base: "neutral_ohlc", title: "Neutral four-channel OHLC MSE / persistence MSE", unit: "MSE ratio", value: |p, _, _, _| p.neutral_ratio, anchor: Some(("zero-return persistence", 1.)) },
    Metric { base: "raw_ohlc", title: "Raw four-channel OHLC MSE / persistence MSE; model market forecast stays zero", unit: "MSE ratio", value: |p, _, _, _| p.raw_ratio, anchor: Some(("zero-return persistence", 1.)) },
    Metric { base: "nll", title: "Neutral four-channel Gaussian NLL, constant 0.5*ln(2*pi) omitted; lower is better", unit: "nats per valid channel", value: |_, e, _, i| e.horizon_nll[i], anchor: None },
    Metric { base: "coverage_1", title: "Neutral four-channel predictive plus/minus one-sigma coverage", unit: "coverage fraction", value: |p, _, _, _| p.within_1_sigma, anchor: Some(("nominal Gaussian one-sigma coverage", 0.6826894921370859)) },
    Metric { base: "coverage_95", title: "Neutral four-channel predictive plus/minus 1.96-sigma coverage", unit: "coverage fraction", value: |p, _, _, _| p.within_2_sigma, anchor: Some(("nominal Gaussian 95-percent coverage", 0.95)) },
];

struct Scored {
    label: String,
    values: Vec<Vec<f64>>,
    persistence_nll: Vec<f64>,
    valid_elements: Vec<f64>,
    cross_sections: Vec<f64>,
    cross_ic_se: Vec<f64>,
}
impl Scored {
    fn new(label: String, sample: Evaluation, cross: Evaluation) -> Result<Self> {
        let points = horizon_track(&sample.horizon, &sample.trading, None);
        ensure!(!points.is_empty(), "no scored decision horizons");
        let values = METRICS
            .iter()
            .map(|metric| {
                points
                    .iter()
                    .map(|p| (metric.value)(p, &sample, &cross, p.horizon - 1))
                    .collect()
            })
            .collect();
        Ok(Self {
            label,
            values,
            persistence_nll: points
                .iter()
                .map(|p| sample.horizon_persistence_nll[p.horizon - 1])
                .collect(),
            valid_elements: points.iter().map(|p| p.valid_elements).collect(),
            cross_sections: points
                .iter()
                .map(|p| cross.trading.cross_sectional_ic_moments[p.horizon - 1])
                .collect(),
            cross_ic_se: points
                .iter()
                .map(|p| cross.trading.cross_sectional_ic_se[p.horizon - 1])
                .collect(),
        })
    }
}

fn difference(candidate: &[f64], reference: &[f64]) -> Vec<f64> {
    candidate
        .iter()
        .zip(reference)
        .map(|(a, b)| a - b)
        .collect()
}

fn write_comparison(output: &std::path::Path, horizons: &[u64], scored: &[Scored]) -> Result<()> {
    let reference = &scored[0];
    let context = "forecast-only decoupled+lattice reference; identical held-out observations; NONMATCHED historical series are not causal matched-budget evidence";
    for (index, metric) in METRICS.iter().enumerate() {
        let mut curves = scored
            .iter()
            .map(|model| series(&model.label, model.values[index].iter().copied()))
            .collect::<Vec<_>>();
        if let Some((label, value)) = metric.anchor {
            curves.push(series(
                label,
                horizons.iter().map(|h| {
                    if metric.base == "delayed_ratio" && *h == 1 {
                        f64::NAN
                    } else {
                        value
                    }
                }),
            ));
        }
        if metric.base == "nll" {
            curves.push(series(
                "zero mean, sigma=sqrt(h) Gaussian persistence",
                reference.persistence_nll.iter().copied(),
            ));
        }
        lines(
            output,
            &format!("timexer_accuracy_{}", metric.base),
            format!("{} | {context}", metric.title),
            metric.unit,
            "observed bars ahead",
            horizons,
            curves,
        )?;
        let mut deltas = scored
            .iter()
            .skip(1)
            .map(|model| {
                series(
                    format!("{} minus {}", model.label, reference.label),
                    difference(&model.values[index], &reference.values[index]),
                )
            })
            .collect::<Vec<_>>();
        deltas.push(series(
            "reference minus itself",
            horizons.iter().map(|h| {
                if metric.base.starts_with("delayed_") && *h == 1 {
                    f64::NAN
                } else {
                    0.
                }
            }),
        ));
        lines(
            output,
            &format!("timexer_accuracy_{}_delta", metric.base),
            format!(
                "Candidate minus forecast-only decoupled+lattice: {} | {context}",
                metric.title
            ),
            metric.unit,
            "observed bars ahead",
            horizons,
            deltas,
        )?;
    }
    for (base, title, unit, field) in [
        (
            "valid_elements",
            "Valid OHLC channel elements in the fixed validation panel",
            "count",
            (|s: &Scored| &s.valid_elements) as fn(&Scored) -> &Vec<f64>,
        ),
        (
            "cross_sections",
            "Eligible IC timestamps; zero forecast/target variance excluded",
            "count",
            (|s: &Scored| &s.cross_sections) as fn(&Scored) -> &Vec<f64>,
        ),
        (
            "cross_ic_se",
            "Cross-sectional IC standard error across contributing timestamps",
            "correlation standard error",
            (|s: &Scored| &s.cross_ic_se) as fn(&Scored) -> &Vec<f64>,
        ),
    ] {
        lines(
            output,
            &format!("timexer_accuracy_{base}"),
            format!("{title} | {context}"),
            unit,
            "observed bars ahead",
            horizons,
            scored
                .iter()
                .map(|s| series(&s.label, field(s).iter().copied()))
                .collect(),
        )?;
    }
    Ok(())
}

pub fn compare(args: CompareArgs) -> Result<()> {
    ensure!(
        args.batch_size > 0 && !args.output.exists(),
        "comparison requires positive batch size and a fresh output path"
    );
    let reference_path = fs::canonicalize(&args.reference_run).context("locating reference run")?;
    let mut names = HashSet::new();
    let mut paths = HashSet::new();
    let mut reference_name = "forecast-decoupled-lattice".to_owned();
    let mut research = Vec::new();
    for mut named in args.research_run {
        ensure!(
            names.insert(named.name.clone()),
            "duplicate model name {}",
            named.name
        );
        named.path = fs::canonicalize(&named.path)?;
        ensure!(
            paths.insert(named.path.clone()),
            "duplicate research path {}",
            named.path.display()
        );
        if named.path == reference_path {
            reference_name = named.name;
        } else {
            research.push(named);
        }
    }
    if !paths.contains(&reference_path) {
        ensure!(
            names.insert(reference_name.clone()),
            "the automatic reference name is already used"
        );
    }
    for named in &args.legacy_checkpoint {
        ensure!(
            names.insert(named.name.clone()),
            "duplicate model name {}",
            named.name
        );
    }
    let device = cuda_device()?;
    let (reference, corpus, plan, cross) =
        jepa_runner::load_panel(&reference_path, &args.data_dir, device)?;
    ensure!(
        !reference.model.jepa_mode.enabled()
            && reference.model.scale_coupling == ScaleCoupling::Decoupled
            && reference.model.horizon_decimation == HorizonDecimation::Lattice,
        "reference must be a forecast-only causal decoupled+lattice research run"
    );
    matched_research(&reference, &reference)?;
    let validation_first_ms = corpus
        .origin_timestamps(&plan.validation_refs)
        .into_iter()
        .min()
        .context("empty validation panel")?;
    let cross_first_ms = corpus
        .origin_timestamps(&cross)
        .into_iter()
        .min()
        .context("empty cross-section panel")?;
    let mut authenticated_research = Vec::new();
    for named in research {
        let checkpoint = Checkpoint::read(&named.path)
            .with_context(|| format!("authenticating {}", named.name))?;
        matched_research(&reference, &checkpoint)
            .with_context(|| format!("comparing {}", named.name))?;
        authenticated_research.push((named, checkpoint));
    }
    let mut authenticated_legacy = Vec::new();
    for mut named in args.legacy_checkpoint {
        named.path = fs::canonicalize(&named.path)?;
        let manifest = Manifest::read(&named.path)
            .with_context(|| format!("authenticating {}", named.name))?;
        ensure!(
            manifest.data == corpus.contract,
            "legacy corpus differs from reference for {}",
            named.name
        );
        compatible_geometry(&reference.model, &manifest.model)?;
        calibration_precedes_panels(
            manifest.calibration_last_target_ms(),
            validation_first_ms,
            cross_first_ms,
        )?;
        authenticated_legacy.push((named, manifest));
    }
    // No report is emitted until every checkpoint, corpus, geometry, and fixed panel passed.
    let mut identities: Vec<Value> = Vec::new();
    let mut scored = Vec::new();
    for (named, checkpoint) in std::iter::once((
        NamedPath {
            name: reference_name,
            path: reference_path,
        },
        reference,
    ))
    .chain(authenticated_research)
    {
        let label = format!("{} [matched causal; uncalibrated]", named.name);
        let (_store, model) = checkpoint.load_model(&named.path, device)?;
        let sample = runner::score(
            &corpus,
            &model,
            &plan.validation_refs,
            args.batch_size,
            device,
        )?;
        let cross_score = runner::score(&corpus, &model, &cross, args.batch_size, device)?;
        scored.push(Scored::new(label.clone(), sample, cross_score)?);
        identities.push(json!({
            "name": named.name, "path": named.path, "series_label": label,
            "eligibility_group": "matched-causal-research",
            "future_calendar": false, "gain_status": "uncalibrated; no post-hoc fitting",
            "checkpoint": checkpoint.identity()?,
        }));
    }
    for (named, manifest) in authenticated_legacy {
        let label = format!(
            "{} [NONMATCHED historical; calibrated; future-calendar]",
            named.name
        );
        let (_store, model) = manifest.load_model(&named.path, &corpus, device)?;
        let sample = runner::score(
            &corpus,
            &model,
            &plan.validation_refs,
            args.batch_size,
            device,
        )?;
        let cross_score = runner::score(&corpus, &model, &cross, args.batch_size, device)?;
        scored.push(Scored::new(label.clone(), sample, cross_score)?);
        identities.push(json!({
            "name": named.name, "path": named.path, "series_label": label,
            "eligibility_group": "nonmatched-historical-calibrated-future-calendar-reference",
            "future_calendar": manifest.model.future_calendar,
            "gain_status": "checkpoint's authenticated frozen gain applied without refitting",
            "calibration_last_target_ms": manifest.calibration_last_target_ms(),
            "checkpoint": manifest.accuracy_identity(),
        }));
    }
    let horizons = DECISION_HORIZONS
        .iter()
        .copied()
        .filter(|h| *h <= corpus.contract.pred_len)
        .map(|h| h as u64)
        .collect::<Vec<_>>();
    let mut protocol = json!({
        "schema": "timexer-paired-final-origin-accuracy-v1",
        "scorer": "runner::score; final_origin; no training, probes, calibration fits, or terminal-test scoring",
        "scoring_executable_sha256": file_sha256(std::env::current_exe()?)?,
        "batch_size": args.batch_size,
        "reference_series": scored[0].label,
        "decision_horizons_observed_bars": horizons,
        "corpus_sha256": plan.manifest.corpus_sha256,
        "sample_plan_manifest": plan.manifest,
        "validation_origins_sha256": corpus.origins_sha256(&plan.validation_refs),
        "cross_section_origins_sha256": corpus.origins_sha256(&cross),
        "cross_section_origins": cross.iter().map(|r| json!({"ticker": corpus.ticker(*r).contract.ticker, "origin": r.origin})).collect::<Vec<_>>(),
        "validation_first_final_origin_ms": validation_first_ms,
        "cross_section_first_final_origin_ms": cross_first_ms,
        "cross_section_min_valid_names": runner::CROSS_SECTION_MIN,
        "models": identities,
        "limitations": [
            "Only research arms share causal inputs, fixed training budget, seed, and absence of post-hoc gain fitting.",
            "Historical calibrated future-calendar models are NONMATCHED references despite identical scored observations; future observed print times are not causal decision inputs.",
            "Every scored final origin is strictly after every included historical calibration target reach.",
            "Historical schedule budgets absent from authenticated manifests remain unknown, not inferred from directory names.",
            "Forecast-only decoupled+lattice is the delta reference; candidate minus reference is not a confidence interval or a causal effect estimate.",
            "Directional rates condition on both forecast and target being nonzero; they do not count flat predictions as correct.",
            "Delayed scores measure neutral t+1-close to t+h-close coordinates, not execution P&L; h1 is undefined.",
            "Gaussian NLL and coverage concern market-neutral OHLC; raw-space MSE uses realized market drift only in targets, never the forecast.",
            "Cross-sectional IC uses its separate fixed synchronized panel; pooled metrics use the fixed validation panel.",
            "Fixed validation observations may already have informed experiment choices; terminal test remains locked."
        ]
    });
    // The identity binds all protocol/configuration fields, but is not a metric sidechannel.
    let digest = ring::digest::digest(&ring::digest::SHA256, &serde_json::to_vec(&protocol)?);
    protocol["protocol_sha256"] =
        Value::String(digest.as_ref().iter().map(|b| format!("{b:02x}")).collect());
    if let Some(parent) = args
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&args.output).context("reserving fresh comparison output")?;
    write_comparison(&args.output, &horizons, &scored)?;
    fs::write(
        args.output.join("accuracy-protocol.json"),
        serde_json::to_vec_pretty(&protocol)?,
    )?;
    println!(
        "Authenticated forecast accuracy reports: {}",
        args.output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_paths_keep_equals_in_the_path_and_reject_missing_identity() {
        let named: NamedPath = "forecast=/tmp/checkpoint=42".parse().unwrap();
        assert_eq!(named.name, "forecast");
        assert_eq!(named.path, PathBuf::from("/tmp/checkpoint=42"));
        for invalid in ["path-only", "=path", "name=", "bad\nlabel=path"] {
            assert!(invalid.parse::<NamedPath>().is_err());
        }
    }

    #[test]
    fn geometry_allows_recipe_controls_but_rejects_changed_source_or_target() {
        let reference = ModelConfig::default();
        let mut candidate = reference.clone();
        candidate.scale_coupling = ScaleCoupling::Decoupled;
        candidate.horizon_decimation = HorizonDecimation::Lattice;
        assert!(compatible_geometry(&reference, &candidate).is_ok());
        candidate.patch_len *= 2;
        assert!(compatible_geometry(&reference, &candidate).is_err());
        candidate = reference.clone();
        candidate.pred_len /= 2;
        assert!(compatible_geometry(&reference, &candidate).is_err());
    }

    #[test]
    fn calibration_target_reach_must_precede_both_final_origin_panels_strictly() {
        assert!(calibration_precedes_panels(99, 100, 101).is_ok());
        assert!(calibration_precedes_panels(100, 100, 101).is_err());
        assert!(calibration_precedes_panels(100, 101, 100).is_err());
        assert!(calibration_precedes_panels(101, 102, 100).is_err());
    }

    #[test]
    fn comparison_deltas_preserve_signed_information_and_undefined_horizons() {
        let delta = difference(&[-0.2, 0.1, f64::NAN], &[0.1, -0.2, 0.]);
        assert!((delta[0] + 0.3).abs() < 1e-12);
        assert!((delta[1] - 0.3).abs() < 1e-12);
        assert!(delta[2].is_nan());
    }
}
