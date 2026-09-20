//! Frozen, authenticated common-source temporal diagnostics. Independent rows, not time,
//! are the population axis. Only patch weight/bias derivatives are requested; no updates.
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{ensure, Context, Result};
use clap::Args;
use shared::report::ReportSeries;
use tch::{Device, Kind, Tensor};

use super::{
    corpus::{Batch, Corpus, WindowRef},
    jepa::{masked_mse, population_sigreg, JepaRandom},
    jepa_eval::{lines, probe_geometry, series, SCORE_PANEL_LIMIT},
    jepa_runner::load_panel,
    model::{CausalPatchModel, ModelConfig, Statistics},
    reports::DECISION_HORIZONS,
};
use crate::torch::{hashing::file_sha256, single_ticker_timexer::runner::cuda_device};

const COHORT: usize = 256;
const DIRECTION_SEED: u64 = 0x5349_4752_4449_4147;
const PREFIX: &str = "timexer_segment_sigreg_";

#[derive(Clone, Debug, Args)]
pub struct DiagnoseArgs {
    #[arg(long)]
    pub run_root: PathBuf,
    #[arg(long, default_value_os_t = crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    /// Frozen inference batch size. The gradient population is always one full 256-row batch.
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,
}

/// Collect scalars on CUDA and transfer only the completed chart, never covariance matrices.
#[derive(Default)]
struct Chart(BTreeMap<String, Vec<Tensor>>);
impl Chart {
    fn push(&mut self, name: impl Into<String>, value: Tensor) {
        self.0.entry(name.into()).or_default().push(value.detach());
    }
    fn write(
        self,
        output: &Path,
        base: &str,
        title: &str,
        unit: &str,
        axis: &str,
        steps: &[u64],
    ) -> Result<()> {
        let mut curves: Vec<ReportSeries> = Vec::new();
        for (name, values) in self.0 {
            let values = Tensor::stack(&values, 0)
                .to_kind(Kind::Double)
                .to_device(Device::Cpu);
            let values = Vec::<f64>::try_from(values)?;
            ensure!(
                values.iter().all(|x| x.is_finite()),
                "nonfinite diagnostic series {base}/{name}"
            );
            curves.push(series(name, values));
        }
        if !curves.is_empty() {
            lines(
                output,
                &format!("{PREFIX}{base}"),
                title.into(),
                unit,
                axis,
                steps,
                curves,
            )?;
        }
        Ok(())
    }
}

fn masks(model: &CausalPatchModel, batch: &Batch, stats: &Statistics) -> (Tensor, Tensor) {
    let c = model.config();
    let valid = batch
        .valid
        .narrow(1, 0, c.seq_len)
        .reshape([-1, c.origins(), c.patch_len])
        .amin([-1i64].as_slice(), false);
    let source = &valid * &stats.mask;
    (valid, source)
}

fn at_source(stats: &Statistics, source: i64) -> Statistics {
    Statistics {
        sigma: stats.sigma.narrow(1, source, 1),
        range: stats.range.narrow(1, source, 1),
        log_close: stats.log_close.narrow(1, source, 1),
        market: stats.market.narrow(1, source, 1),
        beta: stats.beta.narrow(1, source, 1),
        mask: stats.mask.narrow(1, source, 1),
    }
}

struct Panel {
    observation: Tensor,
    post_norm: Tensor,
    state: Tensor,
    target: Tensor,
    prediction: Option<Tensor>,
    valid: Tensor,
    source_valid: Tensor,
    horizons: Vec<i64>,
}

fn cache(
    model: &CausalPatchModel,
    corpus: &Corpus,
    refs: &[WindowRef],
    batch_size: usize,
    device: Device,
    positions: &Tensor,
    source: i64,
) -> Result<Panel> {
    let _guard = tch::no_grad_guard();
    let mut observation = Vec::new();
    let mut post_norm = Vec::new();
    let mut state = Vec::new();
    let mut target = Vec::new();
    let mut prediction = Vec::new();
    let mut valid = Vec::new();
    let mut source_valid = Vec::new();
    let mut horizons = Vec::new();
    for rows in refs.chunks(batch_size) {
        let batch = corpus.batch(rows, device)?;
        let stats = model.statistics(&batch);
        let views = model.representation_views(&batch, false);
        let (v, s) = masks(model, &batch, &stats);
        let obs = views.observation.index_select(1, positions);
        // Exact gainless fused RMSNorm used at the trunk entrance, in the native BF16 dtype.
        post_norm.push(
            obs.internal_fused_rms_norm([model.config().d_model], None::<&Tensor>, Some(1e-6))
                .0,
        );
        observation.push(obs);
        state.push(views.state.index_select(1, positions));
        target.push(views.target.index_select(1, positions));
        if let Some(p) = views.prediction {
            prediction.push(p.select(1, source));
        }
        horizons = views.horizons;
        valid.push(v.index_select(1, positions));
        source_valid.push(s.index_select(1, positions));
    }
    Ok(Panel {
        observation: Tensor::cat(&observation, 0),
        post_norm: Tensor::cat(&post_norm, 0),
        state: Tensor::cat(&state, 0),
        target: Tensor::cat(&target, 0),
        prediction: (!prediction.is_empty()).then(|| Tensor::cat(&prediction, 0)),
        valid: Tensor::cat(&valid, 0),
        source_valid: Tensor::cat(&source_valid, 0),
        horizons,
    })
}

/// Population moments with one identical row mask for all terms; feature variances are averaged.
fn centered(
    prediction: &Tensor,
    target: &Tensor,
    persistence: &Tensor,
    mask: &Tensor,
) -> Vec<(&'static str, Tensor)> {
    let p = prediction.to_kind(Kind::Double);
    let y = target.to_kind(Kind::Double);
    let old = persistence.to_kind(Kind::Double);
    let w = mask.to_kind(Kind::Double).unsqueeze(-1);
    let n = w.sum(Kind::Double);
    let denom = n.clamp_min(1.);
    let mean =
        |x: &Tensor| (x * &w).sum_dim_intlist([0i64].as_slice(), false, Kind::Double) / &denom;
    let pm = mean(&p);
    let ym = mean(&y);
    let pc = &p - &pm;
    let yc = &y - &ym;
    let vp = mean(&pc.square()).mean(Kind::Double);
    let vy = mean(&yc.square()).mean(Kind::Double);
    let cov = mean(&(&pc * &yc)).mean(Kind::Double);
    let bias = (&pm - &ym).square().mean(Kind::Double);
    let mse = mean(&(&p - &y).square()).mean(Kind::Double);
    let persist = mean(&(old - &y).square()).mean(Kind::Double);
    let residual = &mse - (&vp + &vy - &cov * 2. + &bias);
    vec![
        ("Vtarget", vy),
        ("Vpred", vp),
        ("covariance", cov),
        ("mean_bias_squared", bias),
        ("MSE", mse),
        ("persistence_MSE", persist),
        ("identity_residual", residual),
        ("valid_rows", n),
    ]
}

fn covariance(values: &Tensor, mask: &Tensor, affine_ceiling: i64) -> Vec<(&'static str, Tensor)> {
    // BF16 forward values are promoted, not recomputed: fp64 eigensolver roundoff and BF16
    // embedding quantization are different phenomena. Numerical rank is deliberately omitted.
    let x = values.to_kind(Kind::Double);
    let w = mask.to_kind(Kind::Double).unsqueeze(-1);
    let n = w.sum(Kind::Double);
    let denominator = n.clamp_min(1.);
    let mean = (&x * &w).sum_dim_intlist([0i64].as_slice(), true, Kind::Double) / &denominator;
    let center = (x - mean) * w.sqrt();
    let cov = center.transpose(0, 1).matmul(&center) / denominator;
    let eigen = cov.linalg_eigvalsh("L");
    let positive = eigen.clamp_min(0.);
    let trace = positive.sum(Kind::Double);
    let energy = positive.square().sum(Kind::Double);
    let rank = trace.square() / energy.clamp_min(1e-300);
    let top_share = positive.max() / trace.clamp_min(1e-300);
    let width = values.size()[1];
    let ceiling = (&n - 1.).clamp_min(0.).clamp_max(width as f64);
    let tail = if width > affine_ceiling {
        positive
            .narrow(0, 0, width - affine_ceiling)
            .sum(Kind::Double)
            / trace.clamp_min(1e-300)
    } else {
        trace.zeros_like()
    };
    vec![
        ("trace", trace),
        ("participation_rank", rank),
        ("top_eigenvalue_share", top_share),
        ("sample_rank_ceiling", ceiling),
        ("valid_rows", n),
        ("minimum_eigenvalue_fp64", eigen.min()),
        ("tail_share_beyond_affine_input_bound", tail),
    ]
}

fn gaussian_null(device: Device) -> Tensor {
    // Exact finite-N IID N(0,I) expectation for the shared kernel's trapezoidal quadrature:
    // E[N |phi_hat(t)-phi_0(t)|²] = 1 - exp(-t²). Nonzero is not evidence of failure.
    let t = Tensor::linspace(0., 3., 17, (Kind::Double, device));
    let weights = Tensor::full([17], 0.375, (Kind::Double, device));
    let _ = weights.narrow(0, 0, 1).fill_(0.1875);
    let _ = weights.narrow(0, 16, 1).fill_(0.1875);
    let normal_square = (-t.square()).exp();
    ((normal_square.ones_like() - normal_square) * (-t.square() * 0.5).exp() * weights)
        .sum(Kind::Double)
}

fn population_reports(
    output: &Path,
    panel: &Panel,
    offsets: &[i64],
    horizons: &[i64],
    random: &JepaRandom,
    observation_random: &JepaRandom,
    affine_ceiling: i64,
) -> Result<()> {
    let steps: Vec<_> = offsets.iter().map(|&x| x as u64).collect();
    let mut geometry = Chart::default();
    let mut rank = Chart::default();
    let mut shares = Chart::default();
    let mut counts = Chart::default();
    let mut numerical = Chart::default();
    let mut sigreg = Chart::default();
    for (name, tensor) in [
        ("observation", &panel.observation),
        ("post_RMSNorm", &panel.post_norm),
        ("state", &panel.state),
        ("target", &panel.target),
    ] {
        for index in 0..offsets.len() as i64 {
            let values = tensor.select(1, index);
            let valid = panel.source_valid.select(1, index);
            for (metric, value) in covariance(&values, &valid, affine_ceiling) {
                if metric == "tail_share_beyond_affine_input_bound" && name != "observation" {
                    continue;
                }
                let chart = match metric {
                    "trace" => &mut geometry,
                    "participation_rank" | "sample_rank_ceiling" => &mut rank,
                    "top_eigenvalue_share" | "tail_share_beyond_affine_input_bound" => &mut shares,
                    "valid_rows" => &mut counts,
                    _ => &mut numerical,
                };
                chart.push(format!("{name}/{metric}"), value);
            }
            if name == "observation" || name == "target" {
                let (score, n) = population_sigreg(
                    &values.to_kind(Kind::Float).unsqueeze(0),
                    &valid.unsqueeze(0),
                    if name == "observation" {
                        observation_random
                    } else {
                        random
                    },
                );
                sigreg.push(format!("{name}/standalone_SIGReg"), score);
                counts.push(format!("{name}/SIGReg_N"), n);
            }
        }
    }
    for _ in offsets {
        sigreg.push(
            "IID_Gaussian_null_expectation_not_failure_threshold",
            gaussian_null(panel.target.device()),
        );
    }
    geometry.write(
        output,
        "covariance_trace",
        "Full 2048-row panel; separate positions, centered population covariance",
        "trace",
        "patch offset from common source",
        &steps,
    )?;
    rank.write(
        output,
        "covariance_rank",
        "Participation rank versus min(valid N - 1, D); no inferred numerical rank",
        "rank",
        "patch offset from common source",
        &steps,
    )?;
    shares.write(output, "covariance_share", "BF16 forward activations; affine input bound applies only to raw observation before rounding", "share", "patch offset from common source", &steps)?;
    numerical.write(
        output,
        "covariance_roundoff",
        "CUDA fp64 eigensolver minimum eigenvalue; not BF16 structural rank",
        "eigenvalue",
        "patch offset from common source",
        &steps,
    )?;
    counts.write(
        output,
        "population",
        "Authenticated validation panel; independent row axis, not flattened time",
        "valid rows",
        "patch offset from common source",
        &steps,
    )?;
    sigreg.write(
        output,
        "population_score",
        "Standalone same-direction SIGReg on complete heldout populations; IID null is nonzero",
        "statistic",
        "patch offset from common source",
        &steps,
    )?;
    // Off and latent-one modes still have real targets at every ladder position, but no
    // fabricated missing predictor. Report their target/persistence geometry independently.
    let mut target_moments = Chart::default();
    let mut paired_sigreg = Chart::default();
    let mut paired_counts = Chart::default();
    for position in 1..offsets.len() as i64 {
        let mask = panel.source_valid.select(1, 0) * panel.valid.select(1, position);
        let source_target = panel.target.select(1, 0);
        for (name, value) in centered(
            &source_target,
            &panel.target.select(1, position),
            &source_target,
            &mask,
        ) {
            if name == "Vtarget" || name == "persistence_MSE" {
                target_moments.push(name, value);
            }
        }
        for (name, values) in [
            ("source", source_target),
            ("future_target", panel.target.select(1, position)),
        ] {
            let (score, n) = population_sigreg(
                &values.to_kind(Kind::Float).unsqueeze(0),
                &mask.unsqueeze(0),
                random,
            );
            paired_sigreg.push(name, score);
            paired_counts.push(name, n);
        }
        paired_sigreg.push(
            "IID_Gaussian_null_expectation",
            gaussian_null(panel.target.device()),
        );
    }
    let all_hs: Vec<_> = horizons.iter().map(|&h| h as u64).collect();
    target_moments.write(
        output,
        "target_moments",
        "Actual attached-target variance and persistence on all ladder positions, including off",
        "feature mean square",
        "forecast bars",
        &all_hs,
    )?;
    paired_sigreg.write(
        output,
        "paired_population_score",
        "Standalone source and future-target SIGReg on identical paired rows and directions",
        "statistic",
        "forecast bars",
        &all_hs,
    )?;
    paired_counts.write(
        output,
        "paired_population",
        "Exactly matched source and future-target population counts",
        "paired rows",
        "forecast bars",
        &all_hs,
    )?;
    if let Some(prediction) = &panel.prediction {
        let mut moments = Chart::default();
        let mut residual = Chart::default();
        let mut population = Chart::default();
        let mut predicted_rank = Chart::default();
        let mut predicted_trace = Chart::default();
        let mut predicted_share = Chart::default();
        for (index, &h) in panel.horizons.iter().enumerate() {
            let position = horizons
                .iter()
                .position(|&x| x == h)
                .context("predictor horizon outside diagnostic ladder")?
                + 1;
            let mask = panel.source_valid.select(1, 0) * panel.valid.select(1, position as i64);
            let predicted = prediction.select(1, index as i64);
            for (name, value) in centered(
                &predicted,
                &panel.target.select(1, position as i64),
                &panel.target.select(1, 0),
                &mask,
            ) {
                match name {
                    "identity_residual" => residual.push(name, value),
                    "valid_rows" => population.push(name, value),
                    _ => moments.push(name, value),
                }
            }
            for (name, value) in covariance(&predicted, &mask, predicted.size()[1]) {
                match name {
                    "trace" => predicted_trace.push(name, value),
                    "participation_rank" | "sample_rank_ceiling" => {
                        predicted_rank.push(name, value)
                    }
                    "top_eigenvalue_share" => predicted_share.push(name, value),
                    _ => {}
                }
            }
        }
        let hs: Vec<_> = panel.horizons.iter().map(|&h| h as u64).collect();
        moments.write(output, "prediction_moments", "Same-mask population moments: MSE = Vtarget + Vpred - 2 covariance + mean bias squared", "feature mean square", "forecast bars", &hs)?;
        residual.write(
            output,
            "prediction_identity",
            "Centered MSE identity residual; CUDA fp64 moments of native model outputs",
            "MSE identity residual",
            "forecast bars",
            &hs,
        )?;
        population.write(
            output,
            "prediction_population",
            "Valid paired rows at the common source; no synthetic predictor for off mode",
            "paired rows",
            "forecast bars",
            &hs,
        )?;
        predicted_trace.write(
            output,
            "prediction_covariance_trace",
            "Prediction covariance on the identical target-pair mask",
            "trace",
            "forecast bars",
            &hs,
        )?;
        predicted_rank.write(
            output,
            "prediction_covariance_rank",
            "Prediction covariance participation rank and sample ceiling",
            "rank",
            "forecast bars",
            &hs,
        )?;
        predicted_share.write(
            output,
            "prediction_covariance_share",
            "Prediction covariance top eigenvalue share",
            "share",
            "forecast bars",
            &hs,
        )?;
    }
    Ok(())
}

fn gradient(loss: &Tensor, parameters: &[Tensor]) -> Vec<Tensor> {
    Tensor::run_backward(&[loss], parameters, true, false)
        .into_iter()
        .map(|g| g.to_kind(Kind::Float))
        .collect()
}
fn dot(a: &[Tensor], b: &[Tensor]) -> Tensor {
    Tensor::stack(
        &a.iter()
            .zip(b)
            .map(|(x, y)| (x * y).sum(Kind::Float))
            .collect::<Vec<_>>(),
        0,
    )
    .sum(Kind::Float)
}
fn cosine(a: &[Tensor], b: &[Tensor]) -> Tensor {
    dot(a, b) / (dot(a, a) * dot(b, b)).sqrt().clamp_min(1e-30)
}
fn gradient_norms(chart: &mut Chart, name: &str, gradient: &[Tensor]) {
    chart.push(
        format!("{name}/patch.weight"),
        gradient[0].square().sum(Kind::Float).sqrt(),
    );
    chart.push(
        format!("{name}/patch.bias"),
        gradient[1].square().sum(Kind::Float).sqrt(),
    );
    chart.push(
        format!("{name}/patch.combined"),
        dot(gradient, gradient).sqrt(),
    );
}

fn gradient_reports(
    output: &Path,
    model: &CausalPatchModel,
    corpus: &Corpus,
    refs: &[WindowRef],
    source: i64,
    positions: &Tensor,
    horizons: &[i64],
    random: &JepaRandom,
    parameters: &[Tensor],
) -> Result<()> {
    let device = parameters[0].device();
    let config = model.config();
    let batch = corpus.batch(refs, device)?;
    let stats = model.statistics(&batch);
    let views = model.representation_views(&batch, false);
    let (valid, source_valid) = masks(model, &batch, &stats);
    let source_stats = at_source(&stats, source);
    // Authentication forbids future calendar, so a one-source head cannot gather future covariates.
    let head = model.head(&batch, &views.state.narrow(1, source, 1), false);
    let (targets, target_mask) = {
        let (targets, target_mask) = model.targets(&batch, &stats, false);
        (
            targets.narrow(1, source, 1).contiguous(),
            target_mask.narrow(1, source, 1).contiguous(),
        )
    };
    let losses = model.losses(&head, &source_stats, &targets, &target_mask);
    let forecast = if config.jepa_mode.detached_forecast() {
        parameters.iter().map(Tensor::zeros_like).collect()
    } else {
        gradient(&losses.objective, parameters)
    };
    let decoded_close = model
        .decode(&model.output(&head), &source_stats)
        .select(2, 3)
        .select(1, 0);
    let close_target = targets.select(2, 3).select(1, 0);
    let close_mask = target_mask.select(2, 0).select(1, 0);
    let target_views = views
        .target
        .index_select(1, positions)
        .transpose(0, 1)
        .to_kind(Kind::Float);
    let population_valid = source_valid.index_select(1, positions).transpose(0, 1);
    let (reg, reg_n) = population_sigreg(&target_views, &population_valid, random);
    let weighted_reg = &reg * config.jepa.sigreg_weight;
    let reg_gradient = gradient(&weighted_reg, parameters);
    let radial_gradient = Tensor::run_backward(&[&reg], &[&target_views], true, false).remove(0);
    let radial_energy = (&radial_gradient * &target_views)
        .sum_dim_intlist([-1i64].as_slice(), false, Kind::Float)
        .square()
        / target_views
            .square()
            .sum_dim_intlist([-1i64].as_slice(), false, Kind::Float)
            .clamp_min(1e-30);
    let radial_share =
        radial_energy.sum(Kind::Float) / radial_gradient.square().sum(Kind::Float).clamp_min(1e-30);
    let mut packet = Chart::default();
    let mut norms = Chart::default();
    let mut cosines = Chart::default();
    let mut horizon_norms = Chart::default();
    let mut horizon_cosines = Chart::default();
    let mut identity = Chart::default();
    packet.push("standalone_SIGReg", reg);
    packet.push("SIGReg_mean_valid_N", reg_n);
    packet.push("SIGReg_radial_gradient_energy_share", radial_share);
    packet.push("IID_Gaussian_null_expectation", gaussian_null(device));
    packet.push(
        "forecast_training_surrogate_value",
        losses.objective.detach(),
    );
    gradient_norms(
        &mut norms,
        "forecast_training_gradient_undecimated_common_source",
        &forecast,
    );
    gradient_norms(&mut norms, "weighted_standalone_SIGReg", &reg_gradient);
    cosines.push(
        "SIGReg_vs_forecast_training_gradient",
        cosine(&reg_gradient, &forecast),
    );
    let mut accuracy_losses = Vec::new();
    let mut accuracy_gradients = Vec::new();
    for &h in horizons {
        let y = close_target.narrow(1, h - 1, 1);
        let mask = close_mask.select(1, h - 1);
        let baseline = masked_mse(&y.zeros_like(), &y, &mask);
        ensure!(
            baseline.double_value(&[]) > 0.,
            "zero persistence error at diagnostic horizon {h}"
        );
        let accuracy = masked_mse(&decoded_close.narrow(1, h - 1, 1), &y, &mask) / baseline;
        let g = gradient(&accuracy, parameters);
        gradient_norms(&mut horizon_norms, "decoded_close_accuracy_gradient", &g);
        horizon_cosines.push(
            "SIGReg_vs_decoded_close_accuracy",
            cosine(&reg_gradient, &g),
        );
        horizon_cosines.push(
            "forecast_training_vs_decoded_close_accuracy",
            cosine(&forecast, &g),
        );
        accuracy_losses.push(accuracy);
        accuracy_gradients.push(g);
    }
    let mean_accuracy = Tensor::stack(&accuracy_losses, 0).mean(Kind::Float);
    let accuracy = gradient(&mean_accuracy, parameters);
    gradient_norms(&mut norms, "mean_decoded_close_accuracy", &accuracy);
    cosines.push(
        "SIGReg_vs_mean_decoded_close_accuracy",
        cosine(&reg_gradient, &accuracy),
    );
    if let Some(prediction) = &views.prediction {
        let prediction = prediction.select(1, source).to_kind(Kind::Float);
        let paired = Tensor::stack(
            &views
                .horizons
                .iter()
                .map(|&h| views.target.select(1, source + h / config.patch_len))
                .collect::<Vec<_>>(),
            1,
        )
        .to_kind(Kind::Float);
        let mask = Tensor::stack(
            &views
                .horizons
                .iter()
                .map(|&h| valid.select(1, source + h / config.patch_len))
                .collect::<Vec<_>>(),
            1,
        ) * source_valid.select(1, source).unsqueeze(1);
        let attached = masked_mse(&prediction, &paired, &mask) * config.jepa.prediction_weight;
        let predictor_only =
            masked_mse(&prediction, &paired.detach(), &mask) * config.jepa.prediction_weight;
        let target_only =
            masked_mse(&prediction.detach(), &paired, &mask) * config.jepa.prediction_weight;
        let attached_g = gradient(&attached, parameters);
        let predictor_g = gradient(&predictor_only, parameters);
        let target_g = gradient(&target_only, parameters);
        for (name, g) in [
            ("weighted_attached_latent", &attached_g),
            ("weighted_predictor_only_latent", &predictor_g),
            ("weighted_target_only_latent", &target_g),
        ] {
            gradient_norms(&mut norms, name, g);
            cosines.push(
                format!("{name}_vs_forecast_training_gradient"),
                cosine(g, &forecast),
            );
            cosines.push(
                format!("{name}_vs_mean_decoded_close_accuracy"),
                cosine(g, &accuracy),
            );
            cosines.push(format!("{name}_vs_SIGReg"), cosine(g, &reg_gradient));
            for g_accuracy in &accuracy_gradients {
                horizon_cosines.push(
                    format!("{name}_vs_decoded_close_accuracy"),
                    cosine(g, g_accuracy),
                );
            }
        }
        let residual: Vec<_> = attached_g
            .iter()
            .zip(&predictor_g)
            .zip(&target_g)
            .map(|((a, p), t)| a - p - t)
            .collect();
        identity.push(
            "attached_minus_predictor_minus_target_norm",
            dot(&residual, &residual).sqrt(),
        );
        identity.push(
            "relative_gradient_sum_residual_BF16_backward_roundoff",
            dot(&residual, &residual).sqrt()
                / dot(&attached_g, &attached_g).sqrt().clamp_min(1e-30),
        );
        packet.push("weighted_attached_latent_MSE", attached);
    }
    let hs: Vec<_> = horizons.iter().map(|&h| h as u64).collect();
    packet.write(
        output,
        "gradient_packet",
        "First predetermined 256 heldout rows; scalar diagnostics, no optimizer or updates",
        "named statistic",
        "gradient cohort",
        &[COHORT as u64],
    )?;
    norms.write(output, "gradient_norm", "Weighted patch parameter derivatives; frozen others; common-source, undecimated, eval dropout", "L2 norm", "gradient cohort", &[COHORT as u64])?;
    cosines.write(output, "gradient_cosine", "Patch parameter cosines; forecast derivative is the configured training surrogate, not accuracy", "cosine (zero if either norm zero)", "gradient cohort", &[COHORT as u64])?;
    horizon_norms.write(
        output,
        "accuracy_gradient_norm",
        "Decoded market-neutral close MSE/persistence derivatives, predefined horizons",
        "L2 norm",
        "forecast bars",
        &hs,
    )?;
    horizon_cosines.write(output, "accuracy_gradient_cosine", "SIGReg and latent patch derivatives versus decoded close accuracy at heldout common source", "cosine (zero if either norm zero)", "forecast bars", &hs)?;
    identity.write(
        output,
        "gradient_identity",
        "Attached latent derivative = predictor-only + target-only, up to BF16 backward rounding",
        "gradient residual",
        "gradient cohort",
        &[COHORT as u64],
    )
}

pub fn diagnose(args: DiagnoseArgs) -> Result<()> {
    ensure!(
        args.batch_size > 0 && !args.output.exists(),
        "diagnostic requires positive batch size and fresh output path"
    );
    let device = cuda_device()?;
    let (checkpoint, corpus, plan, _) = load_panel(&args.run_root, &args.data_dir, device)?;
    ensure!(
        !checkpoint.model.jepa_mode.conditional(),
        "learned-latent diagnostics do not apply to anchored-conditional characteristic predictions; use the frozen conditional CF reports from the JEPA evaluation endpoint"
    );
    ensure!(
        plan.validation_refs.len() == SCORE_PANEL_LIMIT,
        "diagnostic requires the complete authenticated 2048-row validation panel"
    );
    let (source, horizons) = probe_geometry(&checkpoint.model)?;
    let lookback = (checkpoint.model.seq_len - (source + 1) * checkpoint.model.patch_len) as usize;
    for reference in &plan.validation_refs {
        let ordinal = reference
            .origin
            .checked_sub(lookback)
            .context("common source precedes available history")?;
        ensure!(
            corpus.ticker(*reference).timestamp(ordinal) > plan.manifest.training_last_target_ms,
            "common-source diagnostic origin overlaps dense training targets"
        );
    }
    let (store, model) = checkpoint.load_model(&args.run_root, device)?;
    let variables = store.variables();
    let parameters: Vec<_> = ["patch.weight", "patch.bias"]
        .into_iter()
        .map(|name| {
            variables
                .get(name)
                .map(Tensor::shallow_clone)
                .with_context(|| format!("checkpoint lacks {name}"))
        })
        .collect::<Result<_>>()?;
    let affine_ceiling = parameters[0].size()[1] - 1; // final patch close is identically zero after origin centering.
    let offsets: Vec<_> = std::iter::once(0)
        .chain(horizons.iter().map(|h| h / checkpoint.model.patch_len))
        .collect();
    let positions = Tensor::from_slice(&offsets.iter().map(|k| source + k).collect::<Vec<_>>())
        .to_device(device);
    let mut random_config: ModelConfig = checkpoint.model.clone();
    random_config.jepa.seed = DIRECTION_SEED;
    random_config.jepa.views = offsets.len() as i64;
    let mut random = JepaRandom::new(&random_config, device);
    random.refresh()?;
    random.positions.copy_(&positions);
    let observation_random = if checkpoint
        .model
        .jepa_mode
        .target_width(checkpoint.model.d_model)
        != checkpoint.model.d_model
    {
        let mut observation_config = random_config.clone();
        observation_config.jepa_mode = super::jepa::JepaMode::Off;
        let mut draw = JepaRandom::new(&observation_config, device);
        draw.refresh()?;
        Some(draw)
    } else {
        None
    };
    let panel = cache(
        &model,
        &corpus,
        &plan.validation_refs,
        args.batch_size,
        device,
        &positions,
        source,
    )?;
    fs::create_dir(&args.output)?;
    tch::no_grad(|| {
        population_reports(
            &args.output,
            &panel,
            &offsets,
            &horizons,
            &random,
            observation_random.as_ref().unwrap_or(&random),
            affine_ceiling,
        )
    })?;
    drop(panel);
    for parameter in &parameters {
        let _ = parameter.set_requires_grad(true);
    }
    let accuracy_horizons: Vec<i64> = DECISION_HORIZONS
        .iter()
        .copied()
        .filter(|&h| h <= checkpoint.model.pred_len as usize)
        .map(|h| h as i64)
        .collect();
    gradient_reports(
        &args.output,
        &model,
        &corpus,
        &plan.validation_refs[..COHORT],
        source,
        &positions,
        &accuracy_horizons,
        &random,
        &parameters,
    )?;
    for parameter in &parameters {
        let _ = parameter.set_requires_grad(false);
    }
    // Provenance only. Every measured scalar is exclusively in a registered .report.bin.
    let provenance = serde_json::json!({
        "schema": "temporal-sigreg-common-source-diagnostic-v1",
        "run_root": args.run_root,
        "checkpoint": checkpoint.identity()?,
        "checkpoint_weights_sha256": file_sha256(args.run_root.join("weights/jepa.safetensors"))?,
        "corpus_sha256": plan.manifest.corpus_sha256,
        "sample_plan_sha256": checkpoint.sample_plan_sha256,
        "validation_origins_sha256": plan.manifest.validation.origins_sha256,
        "validation_target_mask_sha256": plan.manifest.validation.target_mask_sha256,
        "gradient_cohort_origins_sha256": corpus.origins_sha256(&plan.validation_refs[..COHORT]),
        "gradient_cohort_selection": "first 256 authenticated validation refs in manifest order; no selection by labels",
        "direction_seed": DIRECTION_SEED,
        "direction_rng_stream_xor": 0x4a45_5041_5349_4752_u64,
        "direction_draw": "first refresh of fresh shared JepaRandom; target draw reused for every target population and gradient; same-seed independent observation-width draw only when target width differs",
        "direction_count": random_config.jepa.directions,
        "target_direction_dimension": checkpoint.model.jepa_mode.target_width(checkpoint.model.d_model),
        "observation_direction_dimension": checkpoint.model.d_model,
        "common_source_patch": source,
        "common_source_lookback_bars": lookback,
        "position_offsets_patches": offsets,
        "horizons_bars": horizons,
        "accuracy_horizons_bars": accuracy_horizons,
        "panel_rows": SCORE_PANEL_LIMIT,
        "gradient_rows": COHORT,
        "inference_batch_size": args.batch_size,
        "gradient_parameters": ["patch.weight", "patch.bias"],
        "weights": {"latent": checkpoint.model.jepa.prediction_weight, "standalone_sigreg": checkpoint.model.jepa.sigreg_weight,
            "training_sigreg": if checkpoint.model.jepa_mode.regularized() { checkpoint.model.jepa.sigreg_weight } else { 0. }},
        "population": "independent validation rows, positions separate; no pooled time or training-context origins",
        "covariance": "CUDA fp64 centered population covariance of native BF16 outputs; min(N-1,D) sample ceiling; BF16 quantization may populate algebraic affine-null directions; participation rank is not numerical rank",
        "centering": "feature-averaged population moments on identical source-valid/target-patch-valid mask; Vtarget-Vpred is not innovation variance",
        "forecast_gradient": "shared model.losses objective derivative, configured scale coupling and horizon weights; decoupled means surrogate gradient, not derivative of reported true NLL; undecimated one common source, evaluation dropout; zero for modes detaching the forecast encoder; not a sampled full training step",
        "accuracy_gradient": "decoded market-neutral sigma-scaled close MSE/persistence, separately at predefined decision horizons and their equal mean; same cohort denominator, no selected horizons",
        "latent_gradient": "shared masked_mse at common source, all real predictor horizons; attached, detached target, detached prediction; weighted as configured",
        "sigreg_gradient": "standalone shared population_sigreg on actual target at common source plus future positions; configured weight even for off/no-SIGReg; does not assert this is a training term",
        "null": "IID N(0,I) finite-sample expected statistic is nonzero; correlated market rows need not follow IID null",
        "off_mode": "no fake predictor or latent gradient; observation, target, SIGReg and forecast diagnostics remain available",
        "updates": "none; one resident CUDA model; all parameters frozen except patch weight/bias during autograd queries"
    });
    fs::write(
        args.output.join("diagnostic-provenance.json"),
        serde_json::to_vec_pretty(&provenance)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_identity_uses_one_mask_for_every_moment() {
        let p = Tensor::from_slice(&[1., 3., 3., 5., 900., -900.]).reshape([3, 2]);
        let y = Tensor::from_slice(&[2., 1., 2., 5., -900., 900.]).reshape([3, 2]);
        let old = Tensor::zeros([3, 2], (Kind::Double, Device::Cpu));
        let mask = Tensor::from_slice(&[1., 1., 0.]);
        let values: BTreeMap<_, _> = centered(&p, &y, &old, &mask)
            .into_iter()
            .map(|(name, value)| (name, value.double_value(&[])))
            .collect();
        // Included rows: Vp=1, Vy=2, cov=1, bias²=.5, MSE=1.5.
        for (name, expected) in [
            ("Vpred", 1.),
            ("Vtarget", 2.),
            ("covariance", 1.),
            ("mean_bias_squared", 0.5),
            ("MSE", 1.5),
            ("persistence_MSE", 8.5),
            ("identity_residual", 0.),
            ("valid_rows", 2.),
        ] {
            assert!((values[name] - expected).abs() < 1e-12, "{name}");
        }
    }
}
