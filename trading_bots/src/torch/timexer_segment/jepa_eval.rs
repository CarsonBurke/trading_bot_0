//! Bounded, frozen CUDA ridge probes. These measure linear decodability, not absence of
//! information. Forecast labels are actual later close returns; reconstruction is same-time.
use std::{collections::HashSet, path::Path};

use anyhow::{ensure, Context, Result};
use shared::report::{write_report, Report, ReportKind, ReportSeries, ScaleKind};
use tch::{Device, Kind, Tensor};

use super::{
    corpus::{Batch, Corpus, WindowRef},
    model::{CausalPatchModel, ModelConfig},
};

pub const FIT_PANEL_LIMIT: usize = 4096;
pub const SCORE_PANEL_LIMIT: usize = 2048;
const RECENT_PATCHES: i64 = 4;
const PENALTIES: [f64; 13] = [
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10., 100., 1e3, 1e4, 1e5, 1e6,
];

/// All arms, including off and latent-one, use the same source position and bar horizons.
pub(super) fn probe_geometry(config: &ModelConfig) -> Result<(i64, Vec<i64>)> {
    let horizons = [1, 2, 4, 8, 12].map(|k| k * config.patch_len).to_vec();
    let max = *horizons.last().unwrap();
    ensure!(
        max <= config.pred_len,
        "probe latent ladder exceeds the declared forecast horizon"
    );
    let source = config.origins() - (config.pred_len + config.patch_len - 1) / config.patch_len - 1;
    ensure!(
        source >= RECENT_PATCHES - 1,
        "probe context cannot hold the common origin and recent-history control"
    );
    ensure!(
        (source + 1) * config.patch_len >= config.min_history,
        "probe common origin precedes min_history"
    );
    Ok((source, horizons))
}

pub(super) fn source_lookback_bars(config: &ModelConfig) -> Result<usize> {
    let (source, _) = probe_geometry(config)?;
    Ok((config.seq_len - (source + 1) * config.patch_len) as usize)
}

struct ProbePanel {
    source: i64,
    horizons: Vec<i64>,
    inner: Vec<Dated>,
    holdout: Vec<Dated>,
    validation: Vec<Dated>,
    inner_purged: usize,
    outer_purged: usize,
}

fn probe_panel(
    config: &ModelConfig,
    corpus: &Corpus,
    fit_refs: &[WindowRef],
    score_refs: &[WindowRef],
) -> Result<ProbePanel> {
    ensure!(
        !fit_refs.is_empty() && !score_refs.is_empty(),
        "empty frozen probe panel"
    );
    ensure!(
        config.seq_len as usize == corpus.contract.context
            && config.pred_len as usize == corpus.contract.pred_len
            && config.features == corpus.contract.features,
        "probe model and corpus disagree on source/target or feature alignment"
    );
    verify_membership(fit_refs, &corpus.train_refs, "training")?;
    verify_membership(score_refs, &corpus.validation_refs, "validation")?;
    let (source, horizons) = probe_geometry(config)?;
    ensure!(fit_refs.len() <= FIT_PANEL_LIMIT && score_refs.len() <= SCORE_PANEL_LIMIT,
        "explicit probe panels exceed the declared fit={FIT_PANEL_LIMIT}/score={SCORE_PANEL_LIMIT} cache limits; select smaller fixed panels in the caller");
    let max_h = *horizons.last().unwrap();
    let fit = dated(corpus, fit_refs, source, config, max_h)?;
    let validation = dated(corpus, score_refs, source, config, max_h)?;
    let first_score = validation.iter().map(|r| r.origin).min().unwrap();
    let before = fit.len();
    let fit: Vec<_> = fit.into_iter().filter(|r| r.reach < first_score).collect();
    let outer_purged = before - fit.len();
    let (inner, holdout, inner_purged) = chronological_split(&fit)?;
    let width = config.d_model.max(config.patch_len * 4) as usize;
    ensure!(inner.len() > width,
        "only {} target-purged inner probe rows for {width}-wide features; increase --probe-fit-origins before training",
        inner.len());
    Ok(ProbePanel {
        source,
        horizons,
        inner,
        holdout,
        validation,
        inner_purged,
        outer_purged,
    })
}

pub(super) fn validate_panel(
    config: &ModelConfig,
    corpus: &Corpus,
    fit_refs: &[WindowRef],
    score_refs: &[WindowRef],
) -> Result<()> {
    probe_panel(config, corpus, fit_refs, score_refs).map(|_| ())
}

#[derive(Clone, Copy, Debug)]
struct Dated {
    reference: WindowRef,
    origin: i64,
    reach: i64,
}

fn chronological_split(rows: &[Dated]) -> Result<(Vec<Dated>, Vec<Dated>, usize)> {
    ensure!(
        rows.len() >= 10,
        "probe fit panel needs at least ten dated rows"
    );
    let mut rows = rows.to_vec();
    rows.sort_by_key(|r| (r.origin, r.reference.ticker, r.reference.origin));
    let boundary = rows[rows.len() * 4 / 5].origin;
    let inner: Vec<_> = rows
        .iter()
        .copied()
        .filter(|r| r.reach < boundary)
        .collect();
    let holdout: Vec<_> = rows
        .iter()
        .copied()
        .filter(|r| r.origin >= boundary)
        .collect();
    let purged = rows.len() - inner.len() - holdout.len();
    ensure!(
        !inner.is_empty() && !holdout.is_empty(),
        "no target-purged chronological probe split"
    );
    Ok((inner, holdout, purged))
}

fn verify_membership(refs: &[WindowRef], population: &[WindowRef], label: &str) -> Result<()> {
    let mut owed: HashSet<_> = refs.iter().map(|r| (r.ticker, r.origin)).collect();
    ensure!(
        owed.len() == refs.len(),
        "duplicate origins in {label} probe panel"
    );
    for row in population {
        owed.remove(&(row.ticker, row.origin));
    }
    ensure!(
        owed.is_empty(),
        "{label} panel contains origins outside its declared corpus partition"
    );
    Ok(())
}

fn dated(
    corpus: &Corpus,
    refs: &[WindowRef],
    source: i64,
    config: &ModelConfig,
    horizon: i64,
) -> Result<Vec<Dated>> {
    let back = (config.seq_len - (source + 1) * config.patch_len) as usize;
    refs.iter()
        .map(|&reference| {
            let origin = reference
                .origin
                .checked_sub(back)
                .context("probe source precedes ticker history")?;
            let reach = origin + horizon as usize;
            let ticker = corpus.ticker(reference);
            ensure!(
                reach < ticker.contract.valid_bars,
                "probe target exceeds observed ticker data"
            );
            Ok(Dated {
                reference,
                origin: ticker.timestamp(origin),
                reach: ticker.timestamp(reach),
            })
        })
        .collect()
}

/// Raw fixed-future close log returns, not normalized latent targets. An invalid origin or
/// any missing bar in its return interval invalidates that output; zero is never a label repair.
pub(super) fn future_targets(batch: &Batch, source_bar: i64, horizons: &[i64]) -> (Tensor, Tensor) {
    let close = batch.log_prices.select(2, 3);
    let mut target = Vec::new();
    let mut masks = Vec::new();
    for &h in horizons {
        let y = close.select(1, source_bar + h) - close.select(1, source_bar);
        let valid = batch
            .valid
            .narrow(1, source_bar, h + 1)
            .gt(0.5)
            .all_dim(1, false)
            .logical_and(&y.isfinite());
        target.push(y.where_self(&valid, &y.zeros_like()));
        masks.push(valid.to_kind(Kind::Float));
    }
    (Tensor::stack(&target, 1), Tensor::stack(&masks, 1))
}

struct Cache {
    features: Vec<(String, Tensor)>,
    future: Tensor,
    future_mask: Tensor,
    reconstruction: Tensor,
    reconstruction_mask: Tensor,
    // (row temporal means, row temporal second moments, valid token counts).
    token_moments: Vec<(String, Tensor, Tensor, Tensor)>,
    conditional: Option<ConditionalCache>,
}

struct ConditionalCache {
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
}

/// Per-horizon empirical characteristic mean, fitted exclusively on training caches.
fn conditional_mean(inner: &ConditionalCache, holdout: &ConditionalCache) -> (Tensor, Tensor) {
    let moments = |cache: &ConditionalCache| {
        let target = cache.target.to_kind(Kind::Double);
        let mask = cache.mask.to_kind(Kind::Double);
        let selected = target.where_self(&mask.unsqueeze(-1).gt(0.5), &target.zeros_like());
        (
            selected.sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
            mask.sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
        )
    };
    let (inner_sum, inner_count) = moments(inner);
    let (holdout_sum, holdout_count) = moments(holdout);
    let count = inner_count + holdout_count;
    ((inner_sum + holdout_sum) / count.unsqueeze(-1), count)
}

/// Columns are feature MSE, frozen-mean MSE, psi(0) MSE, mean-relative MSE, and rows.
fn conditional_score(cache: &ConditionalCache, mean: &Tensor) -> Tensor {
    let target = cache.target.to_kind(Kind::Double);
    let prediction = cache.prediction.to_kind(Kind::Double);
    let mask = cache.mask.to_kind(Kind::Double);
    let count = mask.sum_dim_intlist([0i64].as_slice(), false, Kind::Double);
    let mse = |prediction: &Tensor| {
        let error = (prediction - &target)
            .square()
            .mean_dim([-1i64].as_slice(), false, Kind::Double);
        error
            .where_self(&mask.gt(0.5), &error.zeros_like())
            .sum_dim_intlist([0i64].as_slice(), false, Kind::Double)
            / &count
    };
    let zero_return = Tensor::from_slice(&[1f64, 0., 1., 0., 1., 0., 1., 0., 1., 0.])
        .to_device(target.device());
    let actual = mse(&prediction);
    let baseline = mse(mean);
    Tensor::stack(
        &[
            actual.shallow_clone(),
            baseline.shallow_clone(),
            mse(&zero_return),
            actual / baseline,
            count,
        ],
        1,
    )
}

fn conditional_reports(
    output: &Path,
    title: &str,
    horizons: &[i64],
    inner: &ConditionalCache,
    holdout: &ConditionalCache,
    validation: &ConditionalCache,
) -> Result<()> {
    let (mean, fit_count) = conditional_mean(inner, holdout);
    ensure!(
        fit_count.min().double_value(&[]) > 0.,
        "conditional CF empirical mean requires valid training targets at every horizon"
    );
    let scores = conditional_score(validation, &mean).to_device(Device::Cpu);
    let fit_count = fit_count.to_device(Device::Cpu);
    let title = format!("{title}; CONDITIONAL CF, fixed source-anchored market-neutral close return / (source sigma * sqrt(h)); frequencies [.25,.5,1,2,4], interleaved cos/sin; MSE averaged over 10 coordinates; empirical mean fitted once per horizon/coordinate on valid inner+holdout training rows only; no validation fitting or tuning");
    let steps: Vec<_> = horizons.iter().map(|&h| h as u64).collect();
    let column = |name: &str, index: i64| {
        series(
            name,
            (0..horizons.len() as i64).map(|h| scores.double_value(&[h, index])),
        )
    };
    lines(
        output,
        "timexer_segment_jepa_conditional_cf_error",
        title.clone(),
        "validation characteristic-feature MSE",
        "future bars",
        &steps,
        vec![
            column("conditional prediction", 0),
            column("frozen train-only empirical mean", 1),
            column("zero-return psi(0)", 2),
        ],
    )?;
    lines(
        output,
        "timexer_segment_jepa_conditional_cf_ratio",
        title.clone(),
        "validation MSE / frozen train-mean MSE",
        "future bars",
        &steps,
        vec![column("conditional prediction / train-only mean", 3)],
    )?;
    lines(
        output,
        "timexer_segment_jepa_conditional_cf_count",
        title,
        "valid origin-target pairs",
        "future bars",
        &steps,
        vec![
            column("validation", 4),
            series(
                "empirical-mean fit: inner + holdout training",
                (0..horizons.len() as i64).map(|h| fit_count.double_value(&[h])),
            ),
        ],
    )
}

fn token_moments(tokens: &Tensor, mask: &Tensor) -> (Tensor, Tensor, Tensor) {
    let tokens = tokens.to_kind(Kind::Float);
    let count = mask.sum_dim_intlist([1i64].as_slice(), false, Kind::Float);
    let mean =
        (&tokens * mask.unsqueeze(-1)).sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
            / count.clamp_min(1.).unsqueeze(-1);
    let second = (tokens.square() * mask.unsqueeze(-1)).sum_dim_intlist(
        [1i64].as_slice(),
        false,
        Kind::Float,
    ) / count.clamp_min(1.).unsqueeze(-1);
    (mean, second, count)
}

fn cache(
    model: &CausalPatchModel,
    corpus: &Corpus,
    refs: &[Dated],
    batch_size: usize,
    device: Device,
    source: i64,
    horizons: &[i64],
) -> Result<Cache> {
    let _guard = tch::no_grad_guard();
    let config = model.config();
    let mut feature_parts: Vec<(String, Vec<Tensor>)> = Vec::new();
    let mut future = Vec::new();
    let mut future_mask = Vec::new();
    let mut reconstruction = Vec::new();
    let mut reconstruction_mask = Vec::new();
    let mut moments: Vec<(String, Vec<Tensor>, Vec<Tensor>, Vec<Tensor>)> = Vec::new();
    let mut conditional_predictions = Vec::new();
    let mut conditional_targets = Vec::new();
    let mut conditional_masks = Vec::new();
    let predicted_name = if config.jepa_mode.conditional() {
        "predicted-characteristic"
    } else {
        "predicted-latent"
    };
    for rows in refs.chunks(batch_size) {
        let refs: Vec<_> = rows.iter().map(|r| r.reference).collect();
        let batch = corpus.batch(&refs, device)?;
        let views = model.representation_views(&batch, false);
        ensure!(
            views.conditional.is_some() == config.jepa_mode.conditional(),
            "conditional CF target availability disagrees with the configured objective"
        );
        let recent = model.representation_state_at_recent(&batch, source, RECENT_PATCHES);
        let target = views
            .reconstruction_target
            .select(1, source)
            .to_kind(Kind::Float)
            .contiguous();
        let mut features = vec![
            ("direct-input".to_owned(), target.shallow_clone()),
            (
                "observation".to_owned(),
                views.observation.select(1, source).to_kind(Kind::Float),
            ),
            ("recent-state".to_owned(), recent.to_kind(Kind::Float)),
            (
                "full-state".to_owned(),
                views.state.select(1, source).to_kind(Kind::Float),
            ),
        ];
        if let Some(prediction) = &views.prediction {
            for (k, horizon) in views.horizons.iter().enumerate() {
                features.push((
                    format!("{predicted_name}:{horizon}"),
                    prediction
                        .select(1, source)
                        .select(1, k as i64)
                        .to_kind(Kind::Float),
                ));
            }
        }
        if let Some(conditional) = &views.conditional {
            ensure!(
                views.horizons == horizons,
                "conditional CF and common-source probe horizons disagree"
            );
            conditional_predictions.push(
                views
                    .prediction
                    .as_ref()
                    .context("conditional CF targets require a predictor")?
                    .select(1, source)
                    .to_kind(Kind::Float),
            );
            conditional_targets.push(conditional.values.select(1, source));
            conditional_masks.push(conditional.mask.select(1, source));
        }
        if feature_parts.is_empty() {
            feature_parts = features
                .iter()
                .map(|(name, _)| (name.clone(), Vec::new()))
                .collect();
        }
        for ((_, storage), (_, tensor)) in feature_parts.iter_mut().zip(features) {
            storage.push(tensor);
        }
        let bar = (source + 1) * config.patch_len - 1;
        let (y, mask) = future_targets(&batch, bar, horizons);
        let source_valid = batch
            .valid
            .narrow(1, source * config.patch_len, config.patch_len)
            .gt(0.5)
            .all_dim(1, false)
            .logical_and(
                &batch
                    .valid
                    .narrow(1, 0, bar + 1)
                    .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
                    .ge(config.min_history as f64),
            );
        future.push(y);
        future_mask.push(mask * source_valid.to_kind(Kind::Float).unsqueeze(1));
        let reconstruction_valid = source_valid.unsqueeze(1).logical_and(&target.isfinite());
        reconstruction.push(target.where_self(&reconstruction_valid, &target.zeros_like()));
        reconstruction_mask.push(reconstruction_valid.to_kind(Kind::Float));
        let token_mask = batch
            .valid
            .narrow(1, 0, config.seq_len)
            .reshape([batch.rows(), config.origins(), config.patch_len])
            .gt(0.5)
            .all_dim(2, false)
            .to_kind(Kind::Float);
        if moments.is_empty() {
            moments = ["observation", "state"]
                .into_iter()
                .map(|name| (name.to_owned(), Vec::new(), Vec::new(), Vec::new()))
                .collect();
        }
        for ((_, means, seconds, counts), tokens) in
            moments.iter_mut().zip([&views.observation, &views.state])
        {
            let (mean, second, count) = token_moments(tokens, &token_mask);
            means.push(mean);
            seconds.push(second);
            counts.push(count);
        }
    }
    Ok(Cache {
        features: feature_parts
            .into_iter()
            .map(|(name, values)| (name, Tensor::cat(&values, 0)))
            .collect(),
        future: Tensor::cat(&future, 0),
        future_mask: Tensor::cat(&future_mask, 0),
        reconstruction: Tensor::cat(&reconstruction, 0),
        reconstruction_mask: Tensor::cat(&reconstruction_mask, 0),
        token_moments: moments
            .into_iter()
            .map(|(name, mean, second, count)| {
                (
                    name,
                    Tensor::cat(&mean, 0),
                    Tensor::cat(&second, 0),
                    Tensor::cat(&count, 0),
                )
            })
            .collect(),
        conditional: (!conditional_targets.is_empty()).then(|| ConditionalCache {
            prediction: Tensor::cat(&conditional_predictions, 0),
            target: Tensor::cat(&conditional_targets, 0),
            mask: Tensor::cat(&conditional_masks, 0),
        }),
    })
}

/// Six masked sufficient statistics, kept on CUDA, as in the existing probe module. Unlike
/// HostMoments, neither its normal equations nor eigensolves ever cross to CPU.
struct Moments {
    count: Tensor,
    x: Tensor,
    y: Tensor,
    xx: Tensor,
    xy: Tensor,
    yy: Tensor,
}
impl Moments {
    fn new(x: &Tensor, y: &Tensor, mask: &Tensor) -> Self {
        let (x, y, mask) = (
            x.to_kind(Kind::Double),
            y.to_kind(Kind::Double),
            mask.to_kind(Kind::Double),
        );
        // Most representative panels have one shared validity mask. One Gram/eigensolve
        // then serves every output, including all price reconstruction channels.
        let shared_mask = mask.eq_tensor(&mask.narrow(1, 0, 1)).all().int64_value(&[]) == 1;
        let x_mask = if shared_mask {
            mask.narrow(1, 0, 1)
        } else {
            mask.shallow_clone()
        };
        let mx = x_mask.transpose(0, 1).unsqueeze(-1) * x.unsqueeze(0);
        Self {
            count: mask.sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
            x: mx.sum_dim_intlist([1i64].as_slice(), false, Kind::Double),
            y: (&y * &mask).sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
            xx: mx.transpose(1, 2).matmul(&x.unsqueeze(0)),
            xy: (&y * &mask).transpose(0, 1).matmul(&x),
            yy: (y.square() * mask).sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
        }
    }
    fn merge(&self, rhs: &Self) -> Self {
        Self {
            count: &self.count + &rhs.count,
            x: &self.x + &rhs.x,
            y: &self.y + &rhs.y,
            xx: &self.xx + &rhs.xx,
            xy: &self.xy + &rhs.xy,
            yy: &self.yy + &rhs.yy,
        }
    }
    fn system(&self) -> Result<System> {
        let width = self.x.size()[1];
        ensure!(
            self.count.min().double_value(&[]) > width as f64,
            "too few valid rows for {width}-wide probe"
        );
        let x_count = if self.xx.size()[0] == 1 {
            self.count.narrow(0, 0, 1)
        } else {
            self.count.shallow_clone()
        };
        let x = &self.x / x_count.unsqueeze(-1);
        let y = &self.y / &self.count;
        let covariance =
            &self.xx / x_count.reshape([-1, 1, 1]) - x.unsqueeze(-1).matmul(&x.unsqueeze(1));
        let cross = &self.xy / self.count.unsqueeze(-1) - &x * y.unsqueeze(-1);
        let (values, vectors) = covariance.linalg_eigh("L");
        ensure!(
            values.isfinite().all().int64_value(&[]) == 1,
            "CUDA ridge covariance is nonfinite"
        );
        let scale = values
            .mean_dim([-1i64].as_slice(), true, Kind::Double)
            .clamp_min(1e-20);
        Ok(System {
            x,
            y,
            values: values.clamp_min(0.),
            vectors,
            cross,
            scale,
        })
    }
    fn error(&self, weight: &Tensor, bias: &Tensor) -> Tensor {
        let wx = (weight * &self.x).sum_dim_intlist([-1i64].as_slice(), false, Kind::Double);
        let quadratic = if self.xx.size()[0] == 1 {
            (weight.matmul(&self.xx.squeeze_dim(0)) * weight).sum_dim_intlist(
                [-1i64].as_slice(),
                false,
                Kind::Double,
            )
        } else {
            weight
                .unsqueeze(1)
                .matmul(&self.xx)
                .matmul(&weight.unsqueeze(-1))
                .reshape([-1])
        };
        (&self.yy
            - (weight * &self.xy).sum_dim_intlist([-1i64].as_slice(), false, Kind::Double) * 2.
            - bias * &self.y * 2.
            + quadratic
            + bias * wx * 2.
            + bias.square() * &self.count)
            / &self.count
    }
}
struct System {
    x: Tensor,
    y: Tensor,
    values: Tensor,
    vectors: Tensor,
    cross: Tensor,
    scale: Tensor,
}
impl System {
    fn fit(&self, penalty: &Tensor) -> (Tensor, Tensor) {
        let denominator = &self.values + &self.scale * penalty.reshape([-1, 1]);
        let weights = if self.vectors.size()[0] == 1 {
            // Ordinary GEMMs avoid materializing a copy of the eigensystem per output.
            let vectors = self.vectors.squeeze_dim(0);
            (self.cross.matmul(&vectors) / denominator).matmul(&vectors.transpose(0, 1))
        } else {
            let rotated = self
                .vectors
                .transpose(1, 2)
                .matmul(&self.cross.unsqueeze(-1));
            self.vectors
                .matmul(&(rotated / denominator.unsqueeze(-1)))
                .squeeze_dim(-1)
        };
        let bias =
            &self.y - (&weights * &self.x).sum_dim_intlist([-1i64].as_slice(), false, Kind::Double);
        (weights, bias)
    }
}

pub(super) struct Fit {
    pub weight: Tensor,
    pub bias: Tensor,
    pub penalty: Tensor,
}
impl Fit {
    pub fn predict(&self, x: &Tensor) -> Tensor {
        x.to_kind(Kind::Double).matmul(&self.weight.transpose(0, 1)) + &self.bias
    }
}

fn ridge(
    inner_x: &Tensor,
    inner_y: &Tensor,
    inner_mask: &Tensor,
    holdout_x: &Tensor,
    holdout_y: &Tensor,
    holdout_mask: &Tensor,
) -> Result<Fit> {
    ensure!(
        inner_x.device().is_cuda() && holdout_x.device().is_cuda(),
        "frozen probe fitting requires CUDA"
    );
    ensure!(
        inner_x.isfinite().all().int64_value(&[]) == 1
            && holdout_x.isfinite().all().int64_value(&[]) == 1,
        "nonfinite probe features"
    );
    let inner = Moments::new(inner_x, inner_y, inner_mask);
    let holdout = Moments::new(holdout_x, holdout_y, holdout_mask);
    ensure!(
        holdout.count.min().double_value(&[]) > 0.,
        "no valid chronological penalty-selection rows"
    );
    let system = inner.system()?;
    let grid = Tensor::from_slice(&PENALTIES).to_device(inner_x.device());
    let errors: Vec<_> = (0..PENALTIES.len() as i64)
        .map(|i| {
            let (w, b) = system.fit(&grid.get(i));
            holdout.error(&w, &b)
        })
        .collect();
    let errors = Tensor::stack(&errors, 0);
    ensure!(
        errors.isfinite().all().int64_value(&[]) == 1,
        "nonfinite ridge penalty score"
    );
    let penalty = grid.index_select(0, &errors.argmin(0, false));
    let (weight, bias) = inner.merge(&holdout).system()?.fit(&penalty);
    Ok(Fit {
        weight,
        bias,
        penalty,
    })
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Score {
    pub mse: f64,
    pub ratio: f64,
    pub correlation: f64,
    pub count: f64,
}
pub(super) fn score(prediction: &Tensor, target: &Tensor, mask: &Tensor) -> Vec<Score> {
    let (prediction, target, mask) = (
        prediction.to_kind(Kind::Double),
        target.to_kind(Kind::Double),
        mask.to_kind(Kind::Double),
    );
    let sum = |x: Tensor| (x * &mask).sum_dim_intlist([0i64].as_slice(), false, Kind::Double);
    let count = mask.sum_dim_intlist([0i64].as_slice(), false, Kind::Double);
    let n = count.clamp_min(1.);
    let error = sum((&prediction - &target).square()) / &n;
    let baseline = sum(target.square()) / &n;
    let p = sum(prediction.shallow_clone()) / &n;
    let y = sum(target.shallow_clone()) / &n;
    let cov = sum(&prediction * &target) / &n - &p * &y;
    let variance = ((sum(prediction.square()) / &n - p.square())
        * (sum(target.square()) / &n - y.square()))
    .clamp_min(0.)
    .sqrt();
    let correlation = &cov / &variance;
    let all = Tensor::stack(
        &[error.shallow_clone(), error / baseline, correlation, count],
        1,
    )
    .to_device(Device::Cpu);
    (0..all.size()[0])
        .map(|i| Score {
            mse: all.double_value(&[i, 0]),
            ratio: all.double_value(&[i, 1]),
            correlation: all.double_value(&[i, 2]),
            count: all.double_value(&[i, 3]),
        })
        .collect()
}

pub(super) fn lines(
    output: &Path,
    base: &str,
    title: String,
    unit: &str,
    axis: &str,
    steps: &[u64],
    series: Vec<ReportSeries>,
) -> Result<()> {
    ensure!(
        series.iter().all(|s| s.values.len() == steps.len()),
        "report series and axis disagree for {base}"
    );
    write_report(
        output.join(format!("{base}.report.bin")),
        &Report {
            title,
            x_label: Some(axis.to_owned()),
            y_label: Some(unit.to_owned()),
            scale: ScaleKind::Linear,
            kind: ReportKind::IndexedLines {
                steps: steps.to_vec(),
                series,
            },
        },
    )?;
    Ok(())
}
pub(super) fn series(
    label: impl Into<String>,
    values: impl IntoIterator<Item = f64>,
) -> ReportSeries {
    ReportSeries {
        label: label.into(),
        values: values.into_iter().map(|v| v as f32).collect(),
    }
}

/// Fit once on bounded train-only refs, choose penalties in the chronologically last fifth
/// with actual target-reach purging, refit, then score one frozen validation panel. The fit
/// and validation refs are authenticated against the corpus partitions. No test/calibration
/// rows or score labels select a coefficient, penalty, projection or origin.
pub fn evaluate(
    model: &CausalPatchModel,
    corpus: &Corpus,
    fit_refs: &[WindowRef],
    score_refs: &[WindowRef],
    batch_size: usize,
    device: Device,
    output: &Path,
    step: usize,
) -> Result<()> {
    ensure!(
        device.is_cuda() && batch_size > 0,
        "JEPA probes require CUDA and a positive batch size"
    );
    let ProbePanel {
        source,
        horizons,
        inner,
        holdout,
        validation,
        inner_purged,
        outer_purged,
    } = probe_panel(model.config(), corpus, fit_refs, score_refs)?;
    let _guard = tch::no_grad_guard();
    let inner_cache = cache(model, corpus, &inner, batch_size, device, source, &horizons)?;
    let holdout_cache = cache(
        model, corpus, &holdout, batch_size, device, source, &horizons,
    )?;
    let scored = cache(
        model,
        corpus,
        &validation,
        batch_size,
        device,
        source,
        &horizons,
    )?;
    let refs_of = |rows: &[Dated]| rows.iter().map(|r| r.reference).collect::<Vec<_>>();
    let width = model.config().d_model;
    let input_width = model.config().patch_len * 4;
    let predicted_name = if model.config().jepa_mode.conditional() {
        "predicted-characteristic"
    } else {
        "predicted-latent"
    };
    let (coefficient_description, prediction_description) = if model.config().jepa_mode.conditional() {
        (
            format!("full {width}+1 coefficients/output in observation/state; 10+1 in predicted-characteristic"),
            "predicted characteristic",
        )
    } else if model.config().jepa_mode.target_width(width) != width {
        (
            format!("full {width}+1 coefficients/output in observation/state; {}+1 in predicted-latent", model.config().jepa_mode.target_width(width)),
            "predicted latent",
        )
    } else {
        (
            format!("full {width}+1 coefficients/output in every learned representation"),
            "predicted latent",
        )
    };
    let title = format!("Frozen CUDA probes s{step}; {} train fit + {} chronological penalty rows; {} validation rows; purge inner={inner_purged} outer={outer_purged}; common patch source={source}; recent={RECENT_PATCHES} patches; {coefficient_description}, no compression; direct normalized-price input baseline {input_width}+1 coefficients; {prediction_description} uses matching horizon only; known-future covariates excluded from every readout; observation includes causal normalization; fit={} holdout={} validation={}; linear failure is not absence of information", inner.len(), holdout.len(), validation.len(), corpus.origins_sha256(&refs_of(&inner)), corpus.origins_sha256(&refs_of(&holdout)), corpus.origins_sha256(&refs_of(&validation)));
    if let Some(validation) = &scored.conditional {
        conditional_reports(
            output,
            &title,
            &horizons,
            inner_cache.conditional.as_ref().context("missing inner conditional cache")?,
            holdout_cache.conditional.as_ref().context("missing holdout conditional cache")?,
            validation,
        )?;
    }
    let mut future_results = Vec::new();
    let mut penalties = Vec::new();
    let mut reconstruction_results = Vec::new();
    let missing = Score {
        mse: f64::NAN,
        ratio: f64::NAN,
        correlation: f64::NAN,
        count: 0.,
    };
    let mut predicted_scores = vec![missing; horizons.len()];
    let mut predicted_penalties = vec![f64::NAN; horizons.len()];
    let mut have_predicted = false;
    let predicted_prefix = format!("{predicted_name}:");
    for (index, (name, x)) in scored.features.iter().enumerate() {
        let a = &inner_cache.features[index].1;
        let b = &holdout_cache.features[index].1;
        if let Some(horizon) = name.strip_prefix(predicted_prefix.as_str()) {
            let horizon: i64 = horizon.parse()?;
            let Some(h) = horizons.iter().position(|&value| value == horizon) else {
                continue;
            };
            let column = |x: &Tensor| x.narrow(1, h as i64, 1);
            let fitted = ridge(
                a,
                &column(&inner_cache.future),
                &column(&inner_cache.future_mask),
                b,
                &column(&holdout_cache.future),
                &column(&holdout_cache.future_mask),
            )?;
            predicted_scores[h] = score(
                &fitted.predict(x),
                &column(&scored.future),
                &column(&scored.future_mask),
            )[0];
            predicted_penalties[h] = fitted.penalty.double_value(&[0]);
            have_predicted = true;
            continue;
        }
        let future_fit = ridge(
            a,
            &inner_cache.future,
            &inner_cache.future_mask,
            b,
            &holdout_cache.future,
            &holdout_cache.future_mask,
        )?;
        future_results.push((
            name.clone(),
            score(&future_fit.predict(x), &scored.future, &scored.future_mask),
        ));
        let penalty = future_fit.penalty.to_device(Device::Cpu);
        penalties.push(series(
            name.clone(),
            (0..horizons.len() as i64).map(|i| penalty.double_value(&[i])),
        ));
        let reconstruction_fit = ridge(
            a,
            &inner_cache.reconstruction,
            &inner_cache.reconstruction_mask,
            b,
            &holdout_cache.reconstruction,
            &holdout_cache.reconstruction_mask,
        )?;
        reconstruction_results.push((
            name.clone(),
            score(
                &reconstruction_fit.predict(x),
                &scored.reconstruction,
                &scored.reconstruction_mask,
            ),
        ));
    }
    if have_predicted {
        future_results.push((
            format!("{predicted_name} (matching horizon)"),
            predicted_scores,
        ));
        penalties.push(series(
            format!("{predicted_name} (matching horizon)"),
            predicted_penalties,
        ));
    }
    let steps: Vec<_> = horizons.iter().map(|&h| h as u64).collect();
    let full = &future_results
        .iter()
        .find(|(n, _)| n == "full-state")
        .unwrap()
        .1;
    for (base, unit, field) in [
        (
            "timexer_segment_jepa_probe_ratio",
            "MSE / paired zero-return persistence MSE",
            0,
        ),
        (
            "timexer_segment_jepa_probe_correlation",
            "signed pooled Pearson correlation",
            1,
        ),
        (
            "timexer_segment_jepa_probe_error",
            "close log-return MSE",
            2,
        ),
        (
            "timexer_segment_jepa_probe_count",
            "valid origin-target pairs",
            3,
        ),
        (
            "timexer_segment_jepa_probe_gap",
            "(view MSE - full-state MSE) / persistence MSE",
            4,
        ),
    ] {
        lines(
            output,
            base,
            title.clone(),
            unit,
            "future bars",
            &steps,
            future_results
                .iter()
                .map(|(name, scores)| {
                    series(
                        name,
                        scores.iter().enumerate().map(|(i, s)| match field {
                            0 => s.ratio,
                            1 => s.correlation,
                            2 => s.mse,
                            3 => s.count,
                            _ => s.ratio - full[i].ratio,
                        }),
                    )
                })
                .collect(),
        )?;
    }
    lines(
        output,
        "timexer_segment_jepa_probe_penalty",
        title.clone(),
        "ridge / mean feature eigenvalue",
        "future bars",
        &steps,
        penalties,
    )?;
    let reconstruction_steps: Vec<_> = (0..scored.reconstruction.size()[1] as u64).collect();
    for (base, unit, field) in [
        (
            "timexer_segment_jepa_reconstruction",
            "same-time normalized-price MSE / zero-input MSE",
            0,
        ),
        (
            "timexer_segment_jepa_reconstruction_correlation",
            "same-time signed Pearson correlation",
            1,
        ),
        (
            "timexer_segment_jepa_reconstruction_count",
            "valid same-time pairs",
            2,
        ),
    ] {
        lines(
            output,
            base,
            format!("{title}; SAME-TIME reconstruction, not future forecasting"),
            unit,
            "patch price feature (bar-major OHLC)",
            &reconstruction_steps,
            reconstruction_results
                .iter()
                .map(|(name, scores)| {
                    series(
                        name,
                        scores.iter().map(|s| match field {
                            0 => s.ratio,
                            1 => s.correlation,
                            _ => s.count,
                        }),
                    )
                })
                .collect(),
        )?;
    }
    let mut variance_series = Vec::new();
    for (name, means, seconds, counts) in &scored.token_moments {
        let within = (seconds - means.square())
            .mean(Kind::Float)
            .double_value(&[]);
        let between = (means
            .square()
            .mean_dim([0i64].as_slice(), false, Kind::Float)
            - means
                .mean_dim([0i64].as_slice(), false, Kind::Float)
                .square())
        .mean(Kind::Float)
        .double_value(&[]);
        variance_series.push(series(
            format!("{name}: within-window temporal variance"),
            [within],
        ));
        variance_series.push(series(
            format!("{name}: between-window temporal-mean variance"),
            [between],
        ));
        ensure!(
            counts.min().double_value(&[]) > 0.,
            "empty token window in variance panel"
        );
    }
    lines(
        output,
        "timexer_segment_jepa_token_variance",
        title.clone(),
        "mean squared representation units",
        "optimizer step",
        &[step as u64],
        variance_series,
    )?;
    lines(
        output,
        "timexer_segment_jepa_probe_population",
        title,
        "rows",
        "optimizer step",
        &[step as u64],
        vec![
            series("inner fit", [inner.len() as f64]),
            series("penalty selection", [holdout.len() as f64]),
            series("validation", [validation.len() as f64]),
            series("inner purge", [inner_purged as f64]),
            series("outer purge", [outer_purged as f64]),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditional_baseline_uses_only_valid_training_pairs_per_horizon() {
        let target = Tensor::from_slice(&[0.2f64, f64::NAN, 0.8, 0.4, f64::NAN, 0.2])
            .reshape([3, 2, 1])
            .repeat([1, 1, 10]);
        let mask = Tensor::from_slice(&[1f64, 0., 1., 1., 0., 1.]).reshape([3, 2]);
        let training = |start, rows| ConditionalCache {
            prediction: target.narrow(0, start, rows).zeros_like(),
            target: target.narrow(0, start, rows),
            mask: mask.narrow(0, start, rows),
        };
        let (mean, count) = conditional_mean(&training(0, 2), &training(2, 1));
        assert!((mean.double_value(&[0, 0]) - 0.5).abs() < 1e-12);
        assert!((mean.double_value(&[1, 0]) - 0.3).abs() < 1e-12);
        assert_eq!(count.double_value(&[0]), 2.);
        assert_eq!(count.double_value(&[1]), 2.);
        let validation = ConditionalCache {
            prediction: Tensor::from_slice(&[0.6f64, 0.5, f64::NAN, f64::NAN])
                .reshape([2, 2, 1])
                .repeat([1, 1, 10]),
            target: Tensor::from_slice(&[0.7f64, 0.9, f64::NAN, f64::NAN])
                .reshape([2, 2, 1])
                .repeat([1, 1, 10]),
            mask: Tensor::from_slice(&[1f64, 1., 0., 0.]).reshape([2, 2]),
        };
        let scored = conditional_score(&validation, &mean);
        for (h, expected) in [
            [0.01, 0.04, 0.29, 0.25, 1.],
            [0.16, 0.36, 0.41, 0.16 / 0.36, 1.],
        ]
        .iter()
        .enumerate()
        {
            for (column, value) in expected.iter().enumerate() {
                assert!((scored.double_value(&[h as i64, column as i64]) - value).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn chronological_selection_purges_actual_target_reach_and_timestamp_ties() {
        let rows: Vec<_> = (0..20)
            .map(|i| Dated {
                reference: WindowRef {
                    ticker: i % 2,
                    origin: i,
                },
                origin: (i / 2 * 10) as i64,
                reach: (i / 2 * 10 + 20) as i64,
            })
            .collect();
        let (inner, holdout, purged) = chronological_split(&rows).unwrap();
        let first = holdout.iter().map(|r| r.origin).min().unwrap();
        assert!(inner.iter().all(|r| r.reach < first));
        assert!(holdout.iter().all(|r| r.origin >= first));
        assert_eq!(purged, 4);
        assert_eq!(inner.len() + holdout.len() + purged, rows.len());
    }
    #[test]
    fn missing_intermediate_bar_masks_return_without_turning_it_into_zero_label() {
        let packed = Tensor::zeros(
            [1, Batch::row_width(4, 4, 0) as i64],
            (Kind::Float, Device::Cpu),
        );
        let mut batch = Batch::from_packed(packed, 4, 4, 0, 4);
        batch.log_prices.copy_(
            &Tensor::arange(8, (Kind::Float, Device::Cpu))
                .reshape([1, 8, 1])
                .repeat([1, 1, 4]),
        );
        let _ = batch.valid.fill_(1.);
        let _ = batch.valid.narrow(1, 3, 1).fill_(0.);
        let (target, mask) = future_targets(&batch, 1, &[1, 2, 4]);
        assert_eq!(target.double_value(&[0, 0]), 1.);
        assert_eq!(mask.double_value(&[0, 0]), 1.);
        assert_eq!(mask.double_value(&[0, 1]), 0.);
        assert_eq!(mask.double_value(&[0, 2]), 0.);
    }
    #[test]
    fn partition_authentication_rejects_validation_as_training_and_duplicates() {
        let train = [WindowRef {
            ticker: 0,
            origin: 100,
        }];
        assert!(verify_membership(
            &[WindowRef {
                ticker: 0,
                origin: 200
            }],
            &train,
            "training"
        )
        .is_err());
        assert!(verify_membership(&[train[0], train[0]], &train, "training").is_err());
    }
    #[test]
    fn common_probe_origin_cannot_move_with_objective_horizon_count() {
        use super::super::jepa::JepaMode;
        let expected = probe_geometry(&ModelConfig::default()).unwrap();
        for mode in [
            JepaMode::Off,
            JepaMode::LatentOne,
            JepaMode::LatentMulti,
            JepaMode::Anchored,
            JepaMode::AnchoredNoSigreg,
            JepaMode::AnchoredReconstruct,
            JepaMode::AnchoredProjected,
            JepaMode::AnchoredProjectedNoSigreg,
            JepaMode::AnchoredConditional,
            JepaMode::AnchoredProjectedSmall,
        ] {
            let config = ModelConfig {
                jepa_mode: mode,
                ..ModelConfig::default()
            };
            assert_eq!(probe_geometry(&config).unwrap(), expected);
            assert_eq!(
                (expected.0 + 1) * config.patch_len + config.pred_len,
                config.seq_len
            );
        }
        let unaligned = ModelConfig {
            pred_len: 200,
            ..ModelConfig::default()
        };
        assert_eq!(
            source_lookback_bars(&unaligned).unwrap(),
            208,
            "chronology guard must reach the actual earlier patch-boundary decision"
        );
    }
    #[test]
    fn masked_moment_error_matches_paired_residuals_after_merging_mask_patterns() {
        let x = Tensor::from_slice(&[-2f64, -1., 0., 1., 2.]).reshape([5, 1]);
        let y =
            Tensor::from_slice(&[-3f64, 5., -1., 4., 1.5, 3., 3., 2., 5., 1000.]).reshape([5, 2]);
        let mask = Tensor::from_slice(&[1f64, 1., 1., 1., 1., 1., 1., 1., 1., 0.]).reshape([5, 2]);
        let inner = Moments::new(
            &x.narrow(0, 0, 3),
            &y.narrow(0, 0, 3),
            &mask.narrow(0, 0, 3),
        );
        let holdout = Moments::new(
            &x.narrow(0, 3, 2),
            &y.narrow(0, 3, 2),
            &mask.narrow(0, 3, 2),
        );
        let weights = Tensor::from_slice(&[2f64, -1.]).reshape([2, 1]);
        let bias = Tensor::from_slice(&[1f64, 3.]);
        let error = inner.merge(&holdout).error(&weights, &bias);
        let residual = (x.matmul(&weights.transpose(0, 1)) + bias - y).square() * &mask;
        let expected = residual.sum_dim_intlist([0i64].as_slice(), false, Kind::Double)
            / mask.sum_dim_intlist([0i64].as_slice(), false, Kind::Double);
        assert!((error - expected).abs().max().double_value(&[]) < 1e-12);
    }
}
