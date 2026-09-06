use super::{
    benchmark::{self, HardwareSampler},
    compute::{Engine, OptimizerKind, CAPTURE_AFTER_STEPS},
    corpus::{Batch, Corpus, CorpusContract, WindowRef},
    model::{decode_prices, nll_elements, CausalPatchModel, ModelConfig, CHANNELS},
    reports::{
        self, CandleWindow, EvalTiming, HorizonCurve, HorizonSplit, Metrics, PortfolioCurve,
        StepPhases, TradingCurve, TradingSplit,
    },
};
use crate::torch::{hashing::file_sha256, single_ticker_timexer::runner::cuda_device};
use anyhow::{ensure, Context, Result};
use clap::Args;
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use shared::{paths::RUNS_PATH, report::CandleBar, run_dir::RunDir};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{mpsc, Arc},
    thread,
    time::Instant,
};
use tch::{nn, Device, Kind, Tensor};

/// A cross-section is only usable if it holds enough tickers for a within-timestamp
/// correlation and a decile split to mean anything.
const CROSS_SECTION_MIN: i64 = 20;
/// Holding periods the decile long/short backtest reports, in bars.
const PORTFOLIO_HORIZONS: [u64; 5] = [1, 4, 16, 64, 192];
/// Per-side transaction costs swept by the backtest, in basis points.
const COSTS_BPS: [f64; 5] = [0., 1., 2., 5., 10.];
/// Regular-session five-minute bars in a trading year: 252 days of 78 bars.
const BARS_PER_YEAR: f64 = 252. * 78.;

#[derive(Clone, Debug, Args)]
pub struct TrainArgs {
    /// Optional explicit ticker subset; the default is the eligible universe.
    #[arg(long, value_delimiter = ',')]
    pub ticker: Vec<String>,
    #[arg(long, default_value_os_t = crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub run: Option<String>,
    #[command(flatten)]
    pub model: ModelConfig,
    /// Minimum history used to align origins across context-length comparisons.
    #[arg(long, default_value_t = 6000)]
    pub common_context: usize,
    #[arg(long, default_value_t = 1)]
    pub epochs: usize,
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,
    /// AdamW rate; Polar Express defaults to the modded-nanogpt 0.008 / 0.023 pair.
    #[arg(long)]
    pub learning_rate: Option<f64>,
    #[arg(long, value_enum, default_value_t=OptimizerKind::PolarExpress)]
    pub optimizer: OptimizerKind,
    #[arg(long, default_value_t = 20260905)]
    pub seed: u64,
    #[arg(long, default_value_t = 1000)]
    pub eval_every: usize,
    #[arg(long, default_value_t = 2048)]
    pub eval_origins: usize,
    /// Completed full-corpus epochs without improved full-validation NLL.
    #[arg(long, default_value_t = 3)]
    pub patience: usize,
    /// Consecutive held-out sample evaluations (after the first two) without improved NLL.
    #[arg(long, default_value_t = 3)]
    pub preview_patience: usize,
    #[arg(long, default_value_t=true, action=clap::ArgAction::Set)]
    pub fused: bool,
    /// Tickers that must hold a valid bar at a grid timestamp for it to define a market step.
    #[arg(long, default_value_t = 2000)]
    pub market_min_cross_section: usize,
}
impl Default for TrainArgs {
    fn default() -> Self {
        Self {
            ticker: vec![],
            data_dir: crate::data::ingest::bars_dir(),
            run: None,
            model: ModelConfig::default(),
            common_context: 6000,
            epochs: 1,
            batch_size: 256,
            learning_rate: None,
            optimizer: OptimizerKind::PolarExpress,
            seed: 20260905,
            eval_every: 1000,
            eval_origins: 2048,
            patience: 3,
            preview_patience: 3,
            fused: true,
            market_min_cross_section: 2000,
        }
    }
}
#[derive(Clone, Debug, Args)]
pub struct EvaluateArgs {
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long,default_value_os_t=crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,
}
/// The head layout and parameterization are part of this string, not a comment: the head weight
/// keeps its name and its `[pred_len·2·CHANNELS, HEAD_HIDDEN]` shape across the channel-major
/// rewrite, so a `v5` checkpoint loads into the current model without a single shape error and
/// silently means something else - row `c·pred_len + h` of the current layout held channel
/// `h % (2·CHANNELS)` of horizon `h / (2·CHANNELS)`. Measured consequence on
/// `timexer-market-neutral-20260906/weights/best`: the reported mean close coordinate is a
/// log-scale coordinate times `√h` (-0.69, -0.75, -0.74 at h = 16, 64, 192) and the close MSE
/// ratio reads 2.4-2.8 instead of ~0.99. Old checkpoints are not convertible: retrain.
/// `v7` is the modded-nanogpt recipe landing: gainless biasless RMSNorm everywhere (34 norm
/// tensors gone), the four block projections bias-free (32 bias tensors gone), three 1-D lambda
/// banks, one `skip_weights` logit vector and one `value_lambda` per non-source layer. A `v6`
/// state dict carries 66 tensors this model does not want and lacks 12 it needs, so it is not
/// loadable and not convertible: retrain.
const FORMAT: &str =
    "causal-patch-ohlc-universe-v7-head-channel-major-folded-mup-rmsnorm-relu2-lambdas-unet-vres";
/// `_v3`: the backbone numerics changed (QK-norm before the rotation, `relu(x)²` for GELU,
/// learned residual/post/x0 lambdas, U-net skips, the value residual), so a `_v2` loss curve is
/// not comparable to this one even where the parameter set happens to match.
const OBJECTIVE: &str = "causal_patch_market_neutral_nll_v3";
const NUMERICS: &str =
    "fp32-masters-fp32-causal-origin-statistics-bf16-causal-SDPA-rope-fp32-decoder-fp64-prices-nanogpt-lr-cooldown-frac=.60-floor=.15";
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Manifest {
    format: String,
    model: ModelConfig,
    data: CorpusContract,
    objective: String,
    numerics: String,
    requested_tickers: Vec<String>,
    seed: u64,
    epoch: usize,
    step: usize,
    completed_origins: usize,
    completed_target_bars: usize,
    epoch_complete: bool,
    planned_epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    fused: bool,
    optimizer: OptimizerKind,
    optimizer_recipe: String,
    eval_every: usize,
    eval_origins: usize,
    validation_nll: f64,
    validation_mse: f64,
    validation_is_full: bool,
    /// Step and held-out sample NLL of the checkpoint `weights/best` points at; `None` before
    /// the first held-out sample evaluation.
    best_step: Option<usize>,
    best_preview_nll: Option<f64>,
    weights_sha256: String,
    manifest_sha256: String,
}
impl Manifest {
    fn digest(&self) -> Result<String> {
        let mut copy = self.clone();
        copy.manifest_sha256.clear();
        Ok(
            ring::digest::digest(&ring::digest::SHA256, &serde_json::to_vec(&copy)?)
                .as_ref()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
        )
    }
    fn read(directory: &Path) -> Result<Self> {
        let raw: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("manifest.json"))?)?;
        let field = |name: &str| {
            raw.get(name)
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
        };
        ensure!(
            field("format") == FORMAT,
            "unsupported universe checkpoint format {:?}; this build reads {FORMAT}",
            field("format")
        );
        ensure!(
            field("objective") == OBJECTIVE,
            "checkpoint objective {:?} is not {OBJECTIVE}; checkpoints trained on another objective cannot be loaded",
            field("objective")
        );
        ensure!(
            field("numerics") == NUMERICS,
            "checkpoint numerics {:?} differ from {NUMERICS}",
            field("numerics")
        );
        let manifest: Self = serde_json::from_value(raw)?;
        manifest.model.validate()?;
        ensure!(
            !manifest.optimizer_recipe.is_empty(),
            "missing authenticated training optimizer recipe"
        );
        ensure!(
            manifest.manifest_sha256 == manifest.digest()?,
            "checkpoint manifest authentication failed"
        );
        ensure!(
            manifest.weights_sha256 == file_sha256(directory.join("model.safetensors"))?,
            "checkpoint weight authentication failed"
        );
        ensure!(
            manifest.model.seq_len as usize == manifest.data.context
                && manifest.model.pred_len as usize == manifest.data.pred_len
                && manifest.model.features == manifest.data.features,
            "model/data contract mismatch"
        );
        Ok(manifest)
    }
}
/// Structural guard on the head weight's row layout, for the checkpoints a version string
/// cannot catch: one stamped by hand, or one written by a build that changed the layout without
/// bumping [`FORMAT`].
///
/// The head weight varies smoothly along the forecast horizon and discontinuously across
/// channels - the four candle coordinates and the four log scales live on different scales
/// entirely. So in the correct channel-major layout consecutive rows are adjacent horizons of
/// one channel, while in the stale horizon-major layout consecutive rows are eight different
/// channels of one horizon and it is stride `2·CHANNELS` that walks the horizon. The mean
/// row-to-row cosine therefore peaks at stride 1 exactly when the layout matches the reshape in
/// `CausalPatchModel::forward`. On `timexer-market-neutral-20260906/weights/best` it is 0.285 at
/// stride 1 against 0.956 at stride 8, and the per-channel weight norms separate (2.2, 4.1, 7.5,
/// 3.1 for the coordinates) only under the horizon-major reading.
///
/// The probe is silent when neither stride shows smoothness (`max < 0.5`), which is the case for
/// the zero-initialised head before it has trained: an untrained head carries no layout to
/// check, and inventing a verdict from noise would be worse than the silence.
fn check_head_layout(weight: &Tensor) -> Result<()> {
    let rows = weight.size()[0];
    let stride = 2 * CHANNELS;
    ensure!(
        rows > 2 * stride,
        "head output weight has {rows} rows, too few to carry a horizon"
    );
    let unit = weight
        / weight
            .square()
            .sum_dim_intlist([1i64].as_slice(), true, Kind::Float)
            .sqrt()
            .clamp_min(f64::from(f32::MIN_POSITIVE));
    let cosine = |lag: i64| {
        (unit.narrow(0, 0, rows - lag) * unit.narrow(0, lag, rows - lag))
            .sum(Kind::Double)
            .double_value(&[])
            / (rows - lag) as f64
    };
    let (adjacent, strided) = (cosine(1), cosine(stride));
    ensure!(
        adjacent.max(strided) < 0.5 || adjacent > strided,
        "head weight rows are horizon-major, not channel-major: mean row cosine is {strided:.3} at stride {stride} against {adjacent:.3} at stride 1, so this checkpoint predates the channel-major head. There is no conversion - retrain under {FORMAT}"
    );
    Ok(())
}
fn save(directory: &Path, store: &nn::VarStore, mut manifest: Manifest) -> Result<()> {
    fs::create_dir_all(directory)?;
    let temp = directory.join("pending.safetensors");
    store.save(&temp)?;
    manifest.weights_sha256 = file_sha256(&temp)?;
    manifest.manifest_sha256 = manifest.digest()?;
    fs::write(
        directory.join("manifest.pending.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    fs::rename(temp, directory.join("model.safetensors"))?;
    fs::rename(
        directory.join("manifest.pending.json"),
        directory.join("manifest.json"),
    )?;
    Ok(())
}
fn fixed_origins<T: Copy>(origins: &[T], count: usize) -> Result<Vec<T>> {
    ensure!(
        count > 0 && !origins.is_empty(),
        "evaluation requires nonempty origins"
    );
    let count = count.min(origins.len());
    Ok((0..count)
        .map(|i| {
            origins[if count == 1 {
                origins.len() / 2
            } else {
                i * (origins.len() - 1) / (count - 1)
            }]
        })
        .collect())
}

struct Prefetcher {
    requests: Option<mpsc::SyncSender<Vec<WindowRef>>>,
    ready: Option<mpsc::Receiver<Result<Batch>>>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Prefetcher {
    fn new(corpus: Arc<Corpus>) -> Self {
        let (requests, incoming) = mpsc::sync_channel::<Vec<WindowRef>>(1);
        let (outgoing, ready) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("timexer-prefetch".into())
            .spawn(move || {
                while let Ok(refs) = incoming.recv() {
                    if outgoing.send(corpus.host_batch(&refs)).is_err() {
                        break;
                    }
                }
            })
            .expect("starting TimeXer prefetch worker");
        Self {
            requests: Some(requests),
            ready: Some(ready),
            worker: Some(worker),
        }
    }
    fn request(&self, refs: &[WindowRef]) -> Result<()> {
        self.requests
            .as_ref()
            .unwrap()
            .send(refs.to_vec())
            .context("prefetch worker stopped")
    }
    fn receive(&self) -> Result<Batch> {
        self.ready
            .as_ref()
            .unwrap()
            .recv()
            .context("prefetch worker stopped")?
    }
}
impl Drop for Prefetcher {
    fn drop(&mut self) {
        self.requests.take();
        self.ready.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Final-origin held-out scores. `relative` space is the training objective: market-neutral
/// targets, persistence `ŷ = 0`. `absolute` space adds the realized market drift back to the
/// targets while the model's forecast stays `ŷ` (market forecast zero), so its MSE ratio is
/// comparable with runs trained on raw returns. NLL, calibration, and the per-horizon
/// robustness diagnostics are relative only.
struct Evaluation {
    nll: f64,
    persistence_nll: f64,
    mse: f64,
    persistence_mse: f64,
    absolute_mse: f64,
    absolute_persistence_mse: f64,
    within_1_sigma: f64,
    within_2_sigma: f64,
    rmse_price: f64,
    mae_price: f64,
    invalid_fraction: f64,
    tail_loss_share: f64,
    median_window_ratio: f64,
    horizon: HorizonCurve,
    trading: TradingCurve,
    portfolio: PortfolioCurve,
    timing: EvalTiming,
}
fn invalid_candles(prices: &Tensor) -> Tensor {
    let open = prices.select(-1, 0);
    let high = prices.select(-1, 1);
    let low = prices.select(-1, 2);
    let close = prices.select(-1, 3);
    high.lt_tensor(&open.maximum(&close))
        .logical_or(&low.gt_tensor(&open.minimum(&close)))
        .logical_or(&low.gt_tensor(&high))
        .logical_or(&prices.le(0.).any_dim(-1, false))
        .logical_or(&prices.isfinite().logical_not().any_dim(-1, false))
}
/// Final-origin forecast of every row, `[batch, pred_len, 4|1]` unless noted: σ-scaled relative
/// point forecast, log predictive scale, relative targets with their mask, the market drift in σ
/// units that separates relative from absolute targets, decoded absolute prices (`ŷ`, market
/// forecast zero), the same forecast re-based onto the realized market path for charts, and
/// observed prices. `mid` and `sigma` are `[batch]`: the origin bar's log high/low midpoint in σ
/// units (the microstructure anchor the trade close is bounced away from) and the row's causal
/// return σ, which converts a σ-scaled coordinate back into a log return.
struct FinalOrigin {
    scaled: Tensor,
    log_scale: Tensor,
    targets: Tensor,
    mask: Tensor,
    drift: Tensor,
    prices: Tensor,
    rebased_prices: Tensor,
    target_prices: Tensor,
    mid: Tensor,
    sigma: Tensor,
}
/// The model is channel-major (`[rows, origins', channel, bar]`) because every training-loss
/// consumer slices a channel; the scoring accumulators below are channel-last, and at the final
/// origin the whole forecast is `rows·pred_len·4` elements, so one transpose here is free.
fn final_origin(model: &CausalPatchModel, batch: &Batch) -> FinalOrigin {
    let stats = model.statistics(batch);
    let head = model.forward(batch, &stats, false, true);
    let output = model.output(&head);
    let last = stats.last();
    let bars = model.config().pred_len;
    let channel_last = |tensor: &Tensor| tensor.squeeze_dim(1).transpose(1, 2).contiguous();
    let scaled = channel_last(&model.decode(&output, &last));
    let (targets, mask) = model.targets(batch, &last, true);
    let drift = model
        .market_drift(batch, &last, true)
        .reshape([-1, bars, 1]);
    let anchor = batch.anchor.reshape([-1, 1, 1]);
    let sigma = last.sigma.reshape([-1, 1, 1]);
    let targets = channel_last(&targets);
    let origin_bar = batch.log_prices.select(1, model.config().seq_len - 1);
    let row_sigma = last.sigma.reshape([-1]);
    FinalOrigin {
        prices: decode_prices(&scaled, &anchor, &sigma),
        rebased_prices: decode_prices(&(&scaled + &drift), &anchor, &sigma),
        target_prices: &anchor * (&sigma * (&targets + &drift)).exp(),
        mid: (origin_bar.select(-1, 1) + origin_bar.select(-1, 2)) * 0.5 / &row_sigma,
        sigma: row_sigma,
        scaled,
        log_scale: channel_last(&output.log_scale),
        targets,
        mask: mask.reshape([-1, bars]),
        drift,
    }
}
/// Device-resident final-origin accumulators. Every batch adds masked tensor reductions: the
/// twelve aggregate `sums`, eleven per-horizon rows, and per-(window, bar) close-channel state
/// for the tail-trimmed ratio, the window medians and the whole trading-diagnostic family.
/// Nothing crosses to the host until `finish`, so batches queue behind each other without a
/// synchronization point.
///
/// `bar_forecast`/`bar_target` hold the signed close coordinate (zeroed where invalid) and
/// `bar_valid` its mask, which together are everything the trading statistics need: the
/// tail-trim threshold's `|close target|`-with-`-1`-on-invalid array is reconstructed from
/// them in `finish` rather than stored, and the cross-sectional statistics group these rows by
/// `groups[window]`, the dense rank of the window's origin timestamp. That index is built once
/// on the host from the origin list `score` already has, so a per-timestamp cross-section is a
/// single `index_add_` over dimension 0 and never a per-element loop.
struct Scorer {
    pred_len: i64,
    half_log_horizon: Tensor,
    sums: Tensor,
    horizon_sums: Tensor,
    bar_squared: Tensor,
    bar_persistence: Tensor,
    bar_forecast: Tensor,
    bar_target: Tensor,
    bar_valid: Tensor,
    window_mid: Tensor,
    window_sigma: Tensor,
    groups: Tensor,
    group_count: i64,
    bars: usize,
    filled: i64,
}
impl Scorer {
    fn new(
        half_log_horizon: &Tensor,
        groups: &Tensor,
        group_count: usize,
        pred_len: i64,
        device: Device,
    ) -> Self {
        let origins = groups.size()[0];
        let shape = [origins, pred_len];
        let bars = |kind| Tensor::zeros(shape, (kind, device));
        Self {
            pred_len,
            half_log_horizon: half_log_horizon.reshape([1, pred_len, 1]).to_device(device),
            sums: Tensor::zeros([12], (Kind::Double, device)),
            horizon_sums: Tensor::zeros([11, pred_len], (Kind::Double, device)),
            bar_squared: bars(Kind::Float),
            bar_persistence: bars(Kind::Float),
            bar_forecast: bars(Kind::Float),
            bar_target: bars(Kind::Float),
            bar_valid: bars(Kind::Float),
            window_mid: Tensor::zeros([origins], (Kind::Float, device)),
            window_sigma: Tensor::zeros([origins], (Kind::Float, device)),
            groups: groups.to_device(device).to_kind(Kind::Int64),
            group_count: group_count as i64,
            bars: 0,
            filled: 0,
        }
    }
    fn accumulate(&mut self, forecast: &FinalOrigin, valid_target_bars: usize) {
        let close = CHANNELS - 1;
        let tail_count = (valid_target_bars * CHANNELS as usize).div_ceil(100) as i64;
        let mask = forecast.mask.unsqueeze(-1);
        let residual = &forecast.targets - &forecast.scaled;
        let squared = residual.square() * &mask;
        let persistence = forecast.targets.square() * &mask;
        let absolute_targets = &forecast.targets + &forecast.drift;
        let absolute_squared = (&absolute_targets - &forecast.scaled).square() * &mask;
        let absolute_persistence = absolute_targets.square() * &mask;
        let nll = nll_elements(&forecast.scaled, &forecast.log_scale, &forecast.targets) * &mask;
        let baseline = nll_elements(
            &forecast.targets.zeros_like(),
            &self.half_log_horizon,
            &forecast.targets,
        ) * &mask;
        let scale = forecast.log_scale.exp();
        let deviation = residual.abs();
        let within_1 = deviation.le_tensor(&scale).to_kind(Kind::Float) * &mask;
        let within_2 = deviation.le_tensor(&(scale * 1.96)).to_kind(Kind::Float) * &mask;
        let price_errors = (&forecast.prices - &forecast.target_prices) * &mask;
        let invalid = invalid_candles(&forecast.prices);
        let absolute = &deviation * &mask;
        let target_abs = forecast.targets.abs() * &mask;
        let magnitude = target_abs.flatten(0, -1);
        let (top, _) = magnitude.topk(tail_count.min(magnitude.size()[0]), 0, true, false);
        let tail = magnitude
            .ge_tensor(&top.min())
            .reshape_as(&squared)
            .to_kind(Kind::Float);
        let close_target = forecast.targets.select(-1, close);
        let close_forecast = forecast.scaled.select(-1, close);
        let win = (&close_target - &close_forecast)
            .abs()
            .lt_tensor(&close_target.abs())
            .to_kind(Kind::Float)
            * &forecast.mask;
        let predicted = close_forecast.ne(0.).to_kind(Kind::Float) * &forecast.mask;
        let hit = close_forecast
            .sign()
            .eq_tensor(&close_target.sign())
            .to_kind(Kind::Float)
            * &predicted;
        let up = close_forecast.gt(0.).to_kind(Kind::Float) * &forecast.mask;
        let rows = squared.size()[0];
        let channel_sum = |t: &Tensor| t.sum_dim_intlist([-1i64].as_slice(), false, Kind::Float);
        self.bar_squared
            .narrow(0, self.filled, rows)
            .copy_(&channel_sum(&squared));
        self.bar_persistence
            .narrow(0, self.filled, rows)
            .copy_(&channel_sum(&persistence));
        self.bar_forecast
            .narrow(0, self.filled, rows)
            .copy_(&(&close_forecast * &forecast.mask));
        self.bar_target
            .narrow(0, self.filled, rows)
            .copy_(&(&close_target * &forecast.mask));
        self.bar_valid
            .narrow(0, self.filled, rows)
            .copy_(&forecast.mask);
        self.window_mid
            .narrow(0, self.filled, rows)
            .copy_(&forecast.mid);
        self.window_sigma
            .narrow(0, self.filled, rows)
            .copy_(&forecast.sigma);
        self.filled += rows;
        self.bars += valid_target_bars;
        self.sums += Tensor::stack(
            &[
                squared.sum(Kind::Double),
                persistence.sum(Kind::Double),
                price_errors.square().sum(Kind::Double),
                price_errors.abs().sum(Kind::Double),
                (invalid.to_kind(Kind::Float) * &forecast.mask).sum(Kind::Double),
                (&squared * tail).sum(Kind::Double),
                nll.sum(Kind::Double),
                baseline.sum(Kind::Double),
                within_1.sum(Kind::Double),
                within_2.sum(Kind::Double),
                absolute_squared.sum(Kind::Double),
                absolute_persistence.sum(Kind::Double),
            ],
            0,
        );
        self.horizon_sums += Tensor::stack(
            &[
                squared.sum_dim_intlist([0i64, 2].as_slice(), false, Kind::Double),
                persistence.sum_dim_intlist([0i64, 2].as_slice(), false, Kind::Double),
                forecast
                    .mask
                    .sum_dim_intlist([0i64].as_slice(), false, Kind::Double)
                    * CHANNELS,
                absolute.sum_dim_intlist([0i64, 2].as_slice(), false, Kind::Double),
                target_abs.sum_dim_intlist([0i64, 2].as_slice(), false, Kind::Double),
                win.sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
                predicted.sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
                hit.sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
                up.sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
                absolute_squared.sum_dim_intlist([0i64, 2].as_slice(), false, Kind::Double),
                absolute_persistence.sum_dim_intlist([0i64, 2].as_slice(), false, Kind::Double),
            ],
            0,
        );
    }
    fn finish(self, timing: EvalTiming) -> Result<Evaluation> {
        let pred_len = self.pred_len;
        let bars = self.bars;
        ensure!(
            self.filled == self.bar_valid.size()[0],
            "evaluation accumulated {} of {} windows",
            self.filled,
            self.bar_valid.size()[0]
        );
        let (trading, portfolio) = self.trading()?;
        // Window ratios and the per-horizon trimmed sums finish on device; the top-1% |close|
        // thresholds use k_h = ⌈n_h / 100⌉ ≤ n_h and invalid bars sit at -1 below every valid
        // magnitude, so the k_h-th largest is always a valid bar.
        let window_squared = self
            .bar_squared
            .sum_dim_intlist([1i64].as_slice(), false, Kind::Double);
        let window_persistence = self
            .bar_persistence
            .sum_dim_intlist([1i64].as_slice(), false, Kind::Double);
        let scored = window_persistence.gt(0.);
        let median_window_ratio = if scored.sum(Kind::Int64).int64_value(&[]) > 0 {
            (window_squared.masked_select(&scored) / window_persistence.masked_select(&scored))
                .median()
                .double_value(&[])
        } else {
            f64::NAN
        };
        let counts = self.horizon_sums.get(2);
        let tail_counts = ((counts / CHANNELS as f64).round() + 99.)
            .floor_divide_scalar(100)
            .to_kind(Kind::Int64);
        let widest = tail_counts.max().int64_value(&[]);
        // |close target| with -1 on invalid bars, so an invalid bar never reaches a top-1%
        // threshold. Reconstructed rather than stored: the signed close target and its mask
        // are already resident for the trading diagnostics.
        let bar_close = self.bar_target.abs() * &self.bar_valid + &self.bar_valid - 1.;
        let (top, _) = bar_close.topk(widest, 0, true, true);
        let thresholds = top.gather(0, &(tail_counts - 1).reshape([1, pred_len]), false);
        let keep = bar_close.lt_tensor(&thresholds).to_kind(Kind::Float);
        let trimmed = Tensor::stack(
            &[
                (&self.bar_squared * &keep).sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
                (&self.bar_persistence * keep).sum_dim_intlist([0i64].as_slice(), false, Kind::Double),
            ],
            0,
        );
        let horizon_sums = Tensor::cat(&[&self.horizon_sums, &trimmed], 0);
        let sums = Vec::<f64>::try_from(self.sums.to_device(Device::Cpu))?;
        let horizon_sums = Vec::<f64>::try_from(horizon_sums.to_device(Device::Cpu).flatten(0, -1))?;
        ensure!(
            sums.iter().chain(&horizon_sums).all(|x| x.is_finite()) && bars > 0,
            "nonfinite universe validation outputs"
        );
        let mut rows = horizon_sums.chunks_exact(pred_len as usize);
        let mut row = || rows.next().expect("thirteen per-horizon accumulator rows");
        let (model_sums, persistence_sums, counts) = (row(), row(), row());
        let (absolute_sums, target_sums) = (row(), row());
        let (wins, predicted, hits, ups) = (row(), row(), row(), row());
        let (absolute_model_sums, absolute_persistence_sums) = (row(), row());
        let (trimmed_squared, trimmed_persistence) = (row(), row());
        ensure!(
            counts.iter().all(|n| *n > 0.),
            "every horizon needs a valid validation target bar"
        );
        let ratio = |numerator: &[f64], denominator: &[f64]| -> Vec<f64> {
            numerator
                .iter()
                .zip(denominator)
                .map(|(n, d)| n / d)
                .collect()
        };
        let per_bar = |values: &[f64]| -> Vec<f64> {
            values
                .iter()
                .zip(counts)
                .map(|(v, n)| v * CHANNELS as f64 / n)
                .collect()
        };
        let elements = bars as f64 * 4.;
        Ok(Evaluation {
            mse: sums[0] / elements,
            persistence_mse: sums[1] / elements,
            absolute_mse: sums[10] / elements,
            absolute_persistence_mse: sums[11] / elements,
            rmse_price: (sums[2] / elements).sqrt(),
            mae_price: sums[3] / elements,
            invalid_fraction: sums[4] / bars as f64,
            tail_loss_share: if sums[0] > 0. { sums[5] / sums[0] } else { 0. },
            nll: sums[6] / elements,
            persistence_nll: sums[7] / elements,
            within_1_sigma: sums[8] / elements,
            within_2_sigma: sums[9] / elements,
            median_window_ratio,
            horizon: HorizonCurve {
                mse: ratio(model_sums, counts),
                persistence_mse: ratio(persistence_sums, counts),
                absolute_mse: ratio(absolute_model_sums, counts),
                absolute_persistence_mse: ratio(absolute_persistence_sums, counts),
                mae_ratio: ratio(absolute_sums, target_sums),
                win_rate: per_bar(wins),
                hit_rate: ratio(hits, predicted),
                up_fraction: per_bar(ups),
                trimmed_mse_ratio: ratio(trimmed_squared, trimmed_persistence),
            },
            trading,
            portfolio,
            timing,
        })
    }
    /// The trading diagnostics, all of them from the resident per-(window, bar) close
    /// coordinate arrays and one host transfer. Every per-horizon reduction below is a
    /// column reduction over dimension 0; the per-timestamp cross-sections are `index_add`
    /// scatters keyed by `groups`; the decile thresholds and rank correlations are `topk`
    /// and `argsort` over dimension 0. The scalar algebra that turns the 35 accumulated
    /// sums into ratios, correlations and gain shares runs on the host over `pred_len`
    /// elements, which is where readable arithmetic belongs.
    fn trading(&self) -> Result<(TradingCurve, PortfolioCurve)> {
        let h = self.pred_len;
        let device = self.bar_valid.device();
        let m = &self.bar_valid;
        let f = &self.bar_forecast;
        let y = &self.bar_target;
        let col = |t: &Tensor| t.sum_dim_intlist([0i64].as_slice(), false, Kind::Double);
        let n = col(m);
        let mu = (&n).clamp_min(1.).reciprocal() * col(f);
        let mu_row = mu.to_kind(Kind::Float).reshape([1, h]);
        // A bet is a bar where forecast and target both carry a nonzero sign; a flat forecast
        // is not a directional claim and a flat target has no side to be right about.
        let rate = |predicted: &Tensor, realized: &Tensor, valid: &Tensor| {
            let (ps, rs) = (predicted.sign(), realized.sign());
            let bet = ps.abs() * rs.abs() * valid;
            let hit = (ps * rs).gt(0.).to_kind(Kind::Float) * valid;
            (col(&hit), col(&bet))
        };
        let (close_hit, close_bet) = rate(f, y, m);
        // Mid anchor: the model's error is unchanged, only persistence's denominator moves
        // from the origin bar's trade close to its log high/low midpoint. Bid-ask bounce
        // inflates the close-anchored denominator and nothing else.
        let mid = self.window_mid.reshape([-1, 1]);
        let mid_target = (y - &mid) * m;
        let mid_forecast = (f - &mid) * m;
        let (mid_hit, mid_bet) = rate(&mid_forecast, &mid_target, m);
        // One-bar execution delay: the bar t+1 close to bar t+h close move, forecast by the
        // difference of the origin's own two coordinates, against delayed persistence (zero).
        // Identically zero at h = 1, which is why that column is reported as undefined.
        let delay_mask = m * m.narrow(1, 0, 1);
        let delayed_target = (y - y.narrow(1, 0, 1)) * &delay_mask;
        let delayed_forecast = (f - f.narrow(1, 0, 1)) * &delay_mask;
        let (delayed_hit, delayed_bet) = rate(&delayed_forecast, &delayed_target, &delay_mask);
        // Ordinal ranks for Spearman. Invalid bars are pushed past every valid one by the
        // sentinel, so a column's valid entries hold exactly the ranks 0..n_h-1 among
        // themselves and the masked Pearson over the ranks is the rank correlation.
        const SENTINEL: f64 = 1e30;
        let ranks = |t: &Tensor| {
            ((t + (1.0_f64 - m) * SENTINEL).argsort(0, false).argsort(0, false)).to_kind(Kind::Float) * m
        };
        let (rf, ry) = (ranks(f), ranks(y));
        // Conviction deciles on the demeaned signal |g|: invalid bars sit at -1 for the top
        // threshold and at the sentinel for the bottom one, so neither can be selected.
        let g = (f - &mu_row) * m;
        let magnitude = g.abs();
        let k = (&n / 10.).floor().to_kind(Kind::Int64).clamp_min(1);
        let widest = k.max().int64_value(&[]);
        let index = (&k - 1).reshape([1, h]);
        let high = &magnitude * m + (m - 1.);
        let (top, _) = high.topk(widest, 0, true, true);
        let top_mask = high.ge_tensor(&top.gather(0, &index, false)).to_kind(Kind::Float) * m;
        let low = &magnitude * m + (1.0_f64 - m) * SENTINEL;
        let (bottom, _) = low.topk(widest, 0, false, true);
        let bottom_mask = low.le_tensor(&bottom.gather(0, &index, false)).to_kind(Kind::Float) * m;
        let side = g.sign() * y;
        let decile = |selected: &Tensor| {
            let (hit, bet) = rate(&g, y, selected);
            [col(selected), col(&(&side * selected)), hit, bet]
        };
        // Per-timestamp cross-sections. One scatter per moment per horizon, no loop.
        let group = |source: &Tensor| {
            Tensor::zeros([self.group_count, h], (Kind::Float, device))
                .index_add(0, &self.groups, source)
        };
        let (gn, gf, gy) = (group(m), group(f), group(y));
        let (gff, gyy, gfy) = (group(&f.square()), group(&y.square()), group(&(f * y)));
        let inverse = gn.clamp_min(1.).reciprocal();
        let (mean_f, mean_y) = (&gf * &inverse, &gy * &inverse);
        let covariance = &gfy * &inverse - &mean_f * &mean_y;
        let spread = ((&gff * &inverse - mean_f.square()).clamp_min(0.)
            * (&gyy * &inverse - mean_y.square()).clamp_min(0.))
        .sqrt();
        let usable = gn
            .ge(CROSS_SECTION_MIN as f64)
            .logical_and(&spread.gt(1e-12))
            .to_kind(Kind::Double);
        let ic = (covariance / spread.clamp_min(1e-30)).to_kind(Kind::Double) * &usable;
        let stacked = Tensor::stack(
            &[
                n,
                col(f),
                col(y),
                col(&f.square()),
                col(&y.square()),
                col(&(f * y)),
                col(&((y - f) * m).square()),
                col(&self.bar_squared),
                col(&self.bar_persistence),
                col(&mid_target.square()),
                mid_hit,
                mid_bet,
                close_hit,
                close_bet,
                col(&(&delayed_target - &delayed_forecast).square()),
                col(&delayed_target.square()),
                delayed_hit,
                delayed_bet,
                col(&rf),
                col(&ry),
                col(&rf.square()),
                col(&ry.square()),
                col(&(&rf * &ry)),
                col(&usable),
                col(&ic),
                col(&ic.square()),
            ]
            .into_iter()
            .chain(decile(&top_mask))
            .chain(decile(&bottom_mask))
            .collect::<Vec<_>>(),
            0,
        );
        let flat = Vec::<f64>::try_from(stacked.to_device(Device::Cpu).flatten(0, -1))?;
        let width = h as usize;
        let at = |row: usize, j: usize| flat[row * width + j];
        let portfolio = self.portfolio()?;
        let mut curve = TradingCurve {
            total_gain: Vec::with_capacity(width),
            all_channel_gain: Vec::with_capacity(width),
            offset_gain: Vec::with_capacity(width),
            demeaned_gain: Vec::with_capacity(width),
            scaling_gain: Vec::with_capacity(width),
            mean_forecast: Vec::with_capacity(width),
            mean_target: Vec::with_capacity(width),
            pearson: Vec::with_capacity(width),
            spearman: Vec::with_capacity(width),
            cross_sectional_ic: Vec::with_capacity(width),
            cross_sectional_ic_se: Vec::with_capacity(width),
            close_mse_ratio: Vec::with_capacity(width),
            mid_anchor_mse_ratio: Vec::with_capacity(width),
            delayed_mse_ratio: Vec::with_capacity(width),
            close_hit_rate: Vec::with_capacity(width),
            mid_anchor_hit_rate: Vec::with_capacity(width),
            delayed_hit_rate: Vec::with_capacity(width),
            top_decile_hit_rate: Vec::with_capacity(width),
            bottom_decile_hit_rate: Vec::with_capacity(width),
            top_decile_return: Vec::with_capacity(width),
            bottom_decile_return: Vec::with_capacity(width),
            conviction_spread_return: Vec::with_capacity(width),
            cross_sections: self.group_count as usize,
        };
        for j in 0..width {
            let count = at(0, j);
            ensure!(count > 0., "horizon {j} scored no valid bar");
            let (mean_forecast, mean_target) = (at(1, j) / count, at(2, j) / count);
            let persistence = at(4, j) / count;
            let error = at(6, j) / count;
            let signal_square = at(3, j) / count - mean_forecast * mean_forecast;
            let signal_target = at(5, j) / count - mean_forecast * mean_target;
            let target_variance = persistence - mean_target * mean_target;
            let offset = (2. * mean_forecast * mean_target - mean_forecast * mean_forecast)
                / persistence;
            let demeaned = if signal_square > 0. {
                signal_target * signal_target / signal_square / persistence
            } else {
                0.
            };
            let total = 1. - error / persistence;
            curve.total_gain.push(total);
            curve
                .all_channel_gain
                .push(1. - at(7, j) / at(8, j).max(f64::MIN_POSITIVE));
            curve.offset_gain.push(offset);
            curve.demeaned_gain.push(demeaned);
            curve.scaling_gain.push(total - offset - demeaned);
            curve.mean_forecast.push(mean_forecast);
            curve.mean_target.push(mean_target);
            curve
                .pearson
                .push(signal_target / (signal_square * target_variance).sqrt());
            let rank_correlation = |mean_a: f64, mean_b: f64, aa: f64, bb: f64, ab: f64| {
                (ab / count - mean_a * mean_b)
                    / ((aa / count - mean_a * mean_a) * (bb / count - mean_b * mean_b)).sqrt()
            };
            curve.spearman.push(rank_correlation(
                at(18, j) / count,
                at(19, j) / count,
                at(20, j),
                at(21, j),
                at(22, j),
            ));
            let moments = at(23, j);
            let ic_mean = at(24, j) / moments.max(1.);
            let ic_variance = (at(25, j) / moments.max(1.) - ic_mean * ic_mean).max(0.);
            curve.cross_sectional_ic.push(if moments > 0. {
                ic_mean
            } else {
                f64::NAN
            });
            curve
                .cross_sectional_ic_se
                .push((ic_variance / moments.max(1.)).sqrt());
            curve.close_mse_ratio.push(error / persistence);
            curve.mid_anchor_mse_ratio.push(at(6, j) / at(9, j));
            curve.delayed_mse_ratio.push(at(14, j) / at(15, j));
            curve.close_hit_rate.push(at(12, j) / at(13, j));
            curve.mid_anchor_hit_rate.push(at(10, j) / at(11, j));
            curve.delayed_hit_rate.push(at(16, j) / at(17, j));
            let top_return = at(27, j) / at(26, j);
            let bottom_return = at(31, j) / at(30, j);
            curve.top_decile_return.push(top_return);
            curve.bottom_decile_return.push(bottom_return);
            curve
                .conviction_spread_return
                .push(top_return - bottom_return);
            curve.top_decile_hit_rate.push(at(28, j) / at(29, j));
            curve.bottom_decile_hit_rate.push(at(32, j) / at(33, j));
        }
        Ok((curve, portfolio))
    }
    /// Cost-aware decile long/short at a few holding periods. Per-timestamp decile membership
    /// without a loop over timestamps: rank the predicted returns globally (a double
    /// `argsort`), then sort the composite key `group·windows + global rank`, which orders the
    /// windows by timestamp and, within a timestamp, by predicted return. Subtracting each
    /// group's start offset from the sorted position gives the within-timestamp rank, and
    /// invalid windows carry a sentinel prediction so they land after every valid member of
    /// their own timestamp and are excluded by the `rank < valid count` test.
    fn portfolio(&self) -> Result<PortfolioCurve> {
        const SENTINEL: f64 = 1e30;
        let device = self.bar_valid.device();
        let windows = self.bar_valid.size()[0];
        let double = (Kind::Double, device);
        let horizons: Vec<u64> = PORTFOLIO_HORIZONS
            .iter()
            .copied()
            .filter(|h| *h <= self.pred_len as u64)
            .collect();
        ensure!(!horizons.is_empty(), "no reportable portfolio holding period");
        let scatter = |index: &Tensor, source: &Tensor| {
            Tensor::zeros([self.group_count], double).index_add(0, index, source)
        };
        let totals = scatter(
            &self.groups,
            &Tensor::ones([windows], double),
        );
        let starts = totals.cumsum(0, Kind::Double) - &totals;
        let position = Tensor::arange(windows, double);
        let mut moments = Vec::with_capacity(horizons.len());
        for horizon in &horizons {
            let j = *horizon as i64 - 1;
            let valid = self.bar_valid.select(1, j).to_kind(Kind::Double);
            let sigma = self.window_sigma.to_kind(Kind::Double);
            let predicted = self.bar_forecast.select(1, j).to_kind(Kind::Double) * &sigma;
            let realized = self.bar_target.select(1, j).to_kind(Kind::Double) * &sigma;
            let keyed = &predicted * &valid + (1.0_f64 - &valid) * SENTINEL;
            let rank = keyed.argsort(0, false).argsort(0, false);
            let order = (&self.groups * windows + rank).argsort(0, false);
            let sorted_group = self.groups.index_select(0, &order);
            let within = &position - starts.index_select(0, &sorted_group);
            let available = scatter(&self.groups, &valid);
            let per_side = (&available / 10.).floor();
            let usable = available.ge(CROSS_SECTION_MIN as f64).to_kind(Kind::Double);
            let (count, side, eligible) = (
                available.index_select(0, &sorted_group),
                per_side.index_select(0, &sorted_group),
                usable.index_select(0, &sorted_group),
            );
            let ranked = within.lt_tensor(&count).to_kind(Kind::Double) * &eligible;
            let long = within.ge_tensor(&(&count - &side)).to_kind(Kind::Double) * &ranked;
            let short = within.lt_tensor(&side).to_kind(Kind::Double) * &ranked;
            let returns = realized.index_select(0, &order);
            let leg = |selected: Tensor| scatter(&sorted_group, &(&returns * &selected));
            let spread = (leg(long) - leg(short)) / per_side.clamp_min(1.);
            let periods = usable.sum(Kind::Double);
            let mean = (&spread * &usable).sum(Kind::Double) / periods.clamp_min(1.);
            let variance =
                ((&spread - &mean).square() * &usable).sum(Kind::Double) / periods.clamp_min(1.);
            moments.push(Tensor::stack(&[mean, variance.sqrt(), periods], 0));
        }
        let flat = Vec::<f64>::try_from(
            Tensor::stack(&moments, 0)
                .to_device(Device::Cpu)
                .flatten(0, -1),
        )?;
        let mut net_bps = Vec::with_capacity(COSTS_BPS.len());
        let mut net_sharpe = Vec::with_capacity(COSTS_BPS.len());
        for cost in COSTS_BPS {
            // One holding period opens and closes both legs, so a per-side cost of `cost` bps
            // is charged four times against the spread return.
            let charge = 4. * cost;
            let mut returns = Vec::with_capacity(horizons.len());
            let mut sharpe = Vec::with_capacity(horizons.len());
            for (index, horizon) in horizons.iter().enumerate() {
                let (mean, deviation) = (flat[index * 3] * 1e4, flat[index * 3 + 1] * 1e4);
                let net = mean - charge;
                returns.push(net);
                sharpe.push(net / deviation * (BARS_PER_YEAR / *horizon as f64).sqrt());
            }
            net_bps.push(returns);
            net_sharpe.push(sharpe);
        }
        Ok(PortfolioCurve {
            horizons,
            costs_bps: COSTS_BPS.to_vec(),
            net_bps,
            net_sharpe,
            cross_sections: (0..flat.len() / 3).map(|i| flat[i * 3 + 2] as usize).collect(),
        })
    }
}
fn synchronized(device: Device, started: Instant) -> f64 {
    if device.is_cuda() {
        tch::Cuda::synchronize(0);
    }
    started.elapsed().as_secs_f64() * 1000.
}
/// One evaluated split's retained per-horizon curves, so a report interval that only scored
/// the held-out sample still redraws the last held-out full curves beside it.
struct Curves {
    horizon: HorizonCurve,
    trading: TradingCurve,
    portfolio: PortfolioCurve,
}
/// Final-origin scores over `origins`, batched like training with the host loader one batch
/// ahead; the forward and metric phases are synchronized so the timing attributes wall clock.
fn score(
    corpus: &Arc<Corpus>,
    model: &CausalPatchModel,
    origins: &[WindowRef],
    batch_size: usize,
    device: Device,
) -> Result<Evaluation> {
    ensure!(
        batch_size > 0 && !origins.is_empty(),
        "evaluation batch/origins must be nonempty"
    );
    let _guard = tch::no_grad_guard();
    let started = Instant::now();
    synchronized(device, started);
    let mut timing = EvalTiming::default();
    // Cross-sectional statistics need the evaluation windows grouped by the moment they were
    // forecast from. The origin clock is already in the row: `origins` names the ticker and the
    // valid-bar ordinal, so the corpus resolves the origin bar's UTC timestamp directly. Dense
    // ranks over the distinct timestamps make a per-timestamp reduction one `index_add` on
    // device, and the host cost is one memory-mapped bar header per scored window.
    let stamps: Vec<i64> = origins
        .iter()
        .map(|reference| corpus.ticker(*reference).timestamp(reference.origin))
        .collect();
    let mut distinct = stamps.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let ranked: Vec<i64> = stamps
        .iter()
        .map(|stamp| {
            distinct
                .binary_search(stamp)
                .expect("every origin timestamp is one of the distinct timestamps") as i64
        })
        .collect();
    let mut scorer = Scorer::new(
        model.half_log_horizon(),
        &Tensor::from_slice(&ranked),
        distinct.len(),
        corpus.contract.pred_len as i64,
        device,
    );
    let loader = Prefetcher::new(Arc::clone(corpus));
    loader.request(&origins[..origins.len().min(batch_size)])?;
    // Reallocated only when the batch shape changes, which is once for the whole run and
    // once more for a partial last batch.
    let mut resident: Option<Batch> = None;
    for index in 0..origins.len().div_ceil(batch_size) {
        // Only the FIRST batch is synchronized between phases. Two `Cuda::synchronize` per
        // batch drained the launch pipeline twice per batch, so no batch's forward could
        // overlap the previous batch's metric reduction and the instrumentation was a large
        // part of what it was measuring. One sampled batch costs one serialized batch, which
        // is the same convention the training loop's phase sample uses.
        let sampled = index == 0;
        let phase = |started: Instant| {
            if sampled {
                synchronized(device, started)
            } else {
                0.
            }
        };
        let waiting = Instant::now();
        let host = loader.receive()?;
        timing.loader_ms += waiting.elapsed().as_secs_f64() * 1000.;
        let next = (index + 1) * batch_size;
        if next < origins.len() {
            loader.request(&origins[next..(next + batch_size).min(origins.len())])?;
        }
        let valid_target_bars = host.valid_target_bars;
        // The same resident buffer the training step uses, for the same reason: `to_device`
        // allocates and frees the whole packed row block on every batch. Evaluation's last
        // batch is a partial one, so the buffer is reallocated when the shape changes -
        // twice per evaluation, not once per batch.
        if resident
            .as_ref()
            .is_none_or(|batch: &Batch| batch.rows() != host.rows())
        {
            resident = Some(host.resident(device));
        }
        let device_batch = resident.as_mut().expect("just allocated");
        host.upload(device_batch)?;
        let forwarding = Instant::now();
        let forecast = final_origin(model, device_batch);
        timing.forward_ms += phase(forwarding);
        let scoring = Instant::now();
        scorer.accumulate(&forecast, valid_target_bars);
        timing.metrics_ms += phase(scoring);
    }
    timing.total_ms = synchronized(device, started);
    scorer.finish(timing)
}
fn candle_windows(
    corpus: &Corpus,
    model: &CausalPatchModel,
    device: Device,
) -> Result<Vec<CandleWindow>> {
    let _guard = tch::no_grad_guard();
    let refs = fixed_origins(&corpus.validation_refs, 4)?;
    let batch = corpus.batch(&refs, device)?;
    let prices = final_origin(model, &batch).rebased_prices;
    let values = Vec::<f32>::try_from(
        prices
            .to_kind(Kind::Float)
            .to_device(Device::Cpu)
            .flatten(0, -1),
    )?;
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "nonfinite candle prediction"
    );
    let history = (corpus.contract.context - 1).min(32);
    refs.iter()
        .zip(values.chunks_exact(corpus.contract.pred_len * 4))
        .map(|(&reference, values)| {
            let ticker = corpus.ticker(reference);
            Ok(CandleWindow {
                ticker: ticker.contract.ticker.clone(),
                actual: ticker.candle_window(
                    reference.origin,
                    history,
                    corpus.contract.pred_len,
                )?,
                origin: history,
                predicted: values
                    .chunks_exact(4)
                    .map(|p| CandleBar {
                        open: p[0],
                        high: p[1],
                        low: p[2],
                        close: p[3],
                    })
                    .collect(),
                timestamp: ticker
                    .timestamp(reference.origin)
                    .checked_add(300_000)
                    .context("candle timestamp overflow")?,
            })
        })
        .collect()
}

pub fn train(args: TrainArgs) -> Result<()> {
    args.model.validate()?;
    ensure!(
        args.epochs > 0
            && args.batch_size > 0
            && args.eval_every > 0
            && args.eval_origins > 0
            && args.patience > 0
            && args.preview_patience > 0,
        "training counts must be positive"
    );
    let base_learning_rate = args
        .learning_rate
        .unwrap_or_else(|| args.optimizer.default_learning_rate());
    ensure!(
        base_learning_rate.is_finite() && base_learning_rate > 0.,
        "learning rate must be positive"
    );
    ensure!(
        args.common_context >= args.model.seq_len as usize,
        "common context must cover model history"
    );
    if let Some(name) = &args.run {
        RunDir::ensure_creatable(RUNS_PATH, name)?;
    }
    let device = cuda_device()?;
    let _backward = crate::torch::cuda::cfg::disable_autograd_multithreading();
    let run = RunDir::create_fresh(RUNS_PATH, args.run.as_deref())?;
    println!("CausalPatch loading eligible ticker universe and authenticating shared chronological partitions");
    let mut corpus = Corpus::load(
        &args.data_dir,
        &args.ticker,
        args.model.seq_len as usize,
        args.model.pred_len as usize,
        args.common_context,
        &args.model.features,
        args.market_min_cross_section,
    )?;
    corpus.prepare(device);
    let corpus = Arc::new(corpus);
    fs::write(
        run.root.join("timexer-segment-data-contract.json"),
        serde_json::to_vec_pretty(&corpus.contract)?,
    )?;
    reports::write_corpus(&run.gens.join("1"), &corpus.contract, &corpus.market)?;
    tch::manual_seed(args.seed as i64);
    tch::Cuda::manual_seed_all(args.seed);
    let store = nn::VarStore::new(device);
    let model = CausalPatchModel::new(&store.root(), &args.model);
    let preview = fixed_origins(&corpus.validation_refs, args.eval_origins)?;
    let mut origins = corpus.train_refs.clone();
    // Whole batches only: the forward and backward are captured as one CUDA graph, and a
    // capture records SHAPES, so every step of the run has to present the same row count.
    let steps_per_epoch = origins.len() / args.batch_size;
    ensure!(
        steps_per_epoch > 0,
        "batch {} exceeds the {} training rows",
        args.batch_size,
        origins.len()
    );
    let scheduled_steps = steps_per_epoch * args.epochs;
    let mut engine = Engine::new(
        &store,
        base_learning_rate,
        args.fused,
        args.optimizer,
        scheduled_steps,
    )?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let mut step = 0;
    let mut best_full = f64::INFINITY;
    let mut stale_epochs = 0;
    let mut best_preview = f64::INFINITY;
    let mut best_step: Option<usize> = None;
    let mut previews = 0usize;
    let mut stale_previews = 0usize;
    let mut stopped = false;
    let mut preview_curve: Option<Curves> = None;
    let mut full_curve: Option<Curves> = None;
    let loader = Prefetcher::new(Arc::clone(&corpus));
    println!("CausalPatch: {} tickers, {} training target bars, {} rows, {steps_per_epoch} optimizer steps per full epoch, {} rows left to the next epoch's reshuffle; context {}, horizon {}, {} dense origins per row, batch {}",corpus.contract.tickers.len(),corpus.contract.train_target_bars,origins.len(),origins.len() - steps_per_epoch * args.batch_size,args.model.seq_len,args.model.pred_len,args.model.origins(),args.batch_size);
    // The interval's per-phase breakdown, refreshed by the sampled step of each interval.
    let mut step_phases = None;
    for epoch in 1..=args.epochs {
        origins.shuffle(&mut rng);
        let mut completed = 0;
        let mut target_bars = 0;
        // `[nll, mse, nonfinite]`, all on device. The third slot replaces a per-step
        // `torch._assert_async` (a GIL acquisition and a tensor bridge on every step) with one
        // indicator kernel per step and one host read per report interval.
        let mut total_loss = Tensor::zeros([3], (Kind::Double, device));
        let mut epoch_steps = 0usize;
        let mut points = Vec::new();
        let mut hardware_history = Vec::new();
        // One host transfer per report interval, never per step: `recipe_scalars` concatenates
        // every learned mixing coefficient on device and copies the whole set once.
        let mut recipe_history: Vec<(usize, Vec<(String, f64)>)> = Vec::new();
        let mut hardware_offset = 0.;
        let mut hardware = Some(HardwareSampler::start()?);
        benchmark::cuda_memory(true)?;
        tch::Cuda::synchronize(0);
        let mut interval_started = Instant::now();
        let mut interval_steps = 0;
        let mut loader_wait_ms = 0.;
        // The epoch runs a whole number of batches. The ragged tail of at most
        // `batch_size - 1` rows is left out of THIS epoch; the order is reshuffled every
        // epoch, so it is a different tail each time rather than a systematically excluded
        // set, and no step ever presents a shape the capture did not record.
        let epoch_origins = &origins[..steps_per_epoch * args.batch_size];
        loader.request(&epoch_origins[..args.batch_size])?;
        for (index, refs) in epoch_origins.chunks(args.batch_size).enumerate() {
            // The interval's last step is the one sampled for the per-phase breakdown: it pays
            // four synchronizations, every other step keeps its asynchronous launch pipeline.
            let sampled = (step + 1) % args.eval_every == 0;
            let waiting = Instant::now();
            let host = loader.receive()?;
            let host_batch_ms = waiting.elapsed().as_secs_f64() * 1000.;
            loader_wait_ms += host_batch_ms;
            let valid_bars = host.valid_target_bars;
            if sampled && device.is_cuda() {
                tch::Cuda::synchronize(0);
            }
            // The forward and backward are captured exactly once, on the step after the
            // optimizer's own two bodies have finished capturing into the shared mempool.
            // The host loader is deliberately left quiescent across it - the next request is
            // issued below, after the step - because `pin_memory` can call `cudaHostAlloc`,
            // which synchronizes the device and would invalidate an in-flight capture.
            let losses = if step == CAPTURE_AFTER_STEPS && engine.capture_ready() {
                engine.arm_step_graph(&model, &host)?
            } else if sampled {
                let (losses, phases) = engine.timed_step(&model, &host)?;
                step_phases = Some(StepPhases {
                    host_batch_ms,
                    h2d_ms: phases[0],
                    forward_backbone_ms: phases[1],
                    forward_head_ms: phases[2],
                    backward_ms: phases[3],
                    captured_replay_ms: phases[4],
                    optimizer_ms: phases[5],
                });
                losses
            } else {
                engine.step(&model, &host)?
            };
            // Requested only now, so the capture step above runs with the loader idle. The
            // loader still has the whole step's device execution to build the next batch,
            // and it needs a fraction of it.
            let next = (index + 1) * args.batch_size;
            if next < epoch_origins.len() {
                loader.request(&epoch_origins[next..next + args.batch_size])?;
            }
            total_loss += Tensor::stack(
                &[
                    losses.nll.shallow_clone(),
                    losses.mse,
                    losses.nll.isfinite().logical_not().to_kind(Kind::Float),
                ],
                0,
            )
            .to_kind(Kind::Double);
            epoch_steps += 1;
            step += 1;
            interval_steps += 1;
            completed += refs.len();
            target_bars += valid_bars;
            let epoch_complete = completed == epoch_origins.len();
            if step % args.eval_every != 0 && !epoch_complete {
                continue;
            }
            tch::Cuda::synchronize(0);
            let step_ms = interval_started.elapsed().as_secs_f64() * 1000. / interval_steps as f64;
            let mut samples = hardware.take().unwrap().finish()?;
            for value in &mut samples {
                value[0] += hardware_offset;
            }
            hardware_offset = samples.last().map_or(hardware_offset, |v| v[0] + 100.);
            hardware_history.extend(samples);
            let (peak_bytes, _) = benchmark::cuda_memory(false)?;
            let selected = if epoch_complete {
                &corpus.validation_refs
            } else {
                &preview
            };
            // Reset here and read after: the evaluation's own peak is what has to fit
            // alongside the captured step's private mempool, which the training peak above
            // does not separate out.
            benchmark::cuda_memory(true)?;
            let evaluation = score(&corpus, &model, selected, args.batch_size, device)?;
            let (evaluation_peak_bytes, _) = benchmark::cuda_memory(false)?;
            // Leaves the next interval's training peak measuring the next interval.
            benchmark::cuda_memory(true)?;
            let finishing = Instant::now();
            let windows = candle_windows(&corpus, &model, device)?;
            let output = run.gens.join(epoch.to_string());
            let curves = Curves {
                horizon: evaluation.horizon,
                trading: evaluation.trading,
                portfolio: evaluation.portfolio,
            };
            if epoch_complete {
                full_curve = Some(curves);
            } else {
                preview_curve = Some(curves);
            }
            if !epoch_complete {
                previews += 1;
                if evaluation.nll < best_preview {
                    best_preview = evaluation.nll;
                    best_step = Some(step);
                    stale_previews = 0;
                } else if previews > 2 {
                    stale_previews += 1;
                }
            }
            let manifest = Manifest {
                format: FORMAT.into(),
                model: args.model.clone(),
                data: corpus.contract.clone(),
                objective: OBJECTIVE.into(),
                numerics: NUMERICS.into(),
                requested_tickers: args.ticker.clone(),
                seed: args.seed,
                epoch,
                step,
                completed_origins: completed,
                completed_target_bars: target_bars,
                epoch_complete,
                planned_epochs: args.epochs,
                batch_size: args.batch_size,
                learning_rate: engine.learning_rate(),
                fused: args.fused,
                optimizer: args.optimizer,
                optimizer_recipe: args.optimizer.recipe().into(),
                eval_every: args.eval_every,
                eval_origins: selected.len(),
                validation_nll: evaluation.nll,
                validation_mse: evaluation.mse,
                validation_is_full: epoch_complete,
                best_step,
                best_preview_nll: best_step.map(|_| best_preview),
                weights_sha256: String::new(),
                manifest_sha256: String::new(),
            };
            let checkpoint = if epoch_complete {
                format!("epoch-{epoch:04}")
            } else {
                "preview-latest".into()
            };
            save(&run.weights.join(&checkpoint), &store, manifest.clone())?;
            if best_step == Some(step) {
                save(&run.weights.join("preview-best"), &store, manifest)?;
                crate::torch::single_ticker_timexer::checkpoint::update_best(
                    &run.weights,
                    "preview-best",
                )?;
            }
            if epoch_complete && best_step.is_none() {
                crate::torch::single_ticker_timexer::checkpoint::update_best(
                    &run.weights,
                    &checkpoint,
                )?;
            }
            let reports_ms = finishing.elapsed().as_secs_f64() * 1000.;
            let train_loss = Vec::<f64>::try_from(&total_loss / epoch_steps as f64)?;
            ensure!(
                train_loss[2] == 0. && train_loss[0].is_finite(),
                "nonfinite CausalPatch training objective in the interval ending at step {step}"
            );
            points.push(Metrics {
                step,
                epoch,
                completed_origins: completed,
                total_origins: origins.len(),
                completed_target_bars: target_bars,
                total_target_bars: corpus.contract.train_target_bars,
                train_nll: Some(train_loss[0]),
                train_mse: Some(train_loss[1]),
                validation_nll: evaluation.nll,
                persistence_nll: evaluation.persistence_nll,
                validation_mse: evaluation.mse,
                persistence_mse: evaluation.persistence_mse,
                absolute_mse: evaluation.absolute_mse,
                absolute_persistence_mse: evaluation.absolute_persistence_mse,
                within_1_sigma: evaluation.within_1_sigma,
                within_2_sigma: evaluation.within_2_sigma,
                rmse_price: evaluation.rmse_price,
                mae_price: evaluation.mae_price,
                invalid_ohlc_fraction: evaluation.invalid_fraction,
                tail_loss_share: evaluation.tail_loss_share,
                median_window_ratio: evaluation.median_window_ratio,
                eval: EvalTiming {
                    reports_ms: Some(reports_ms),
                    ..evaluation.timing
                },
                step_ms: Some(step_ms),
                validation_is_full: epoch_complete,
                validation_origins: selected.len(),
                tickers: corpus.contract.tickers.len(),
                loader_wait_ms: Some(loader_wait_ms / interval_steps as f64),
                peak_allocator_mib: Some(peak_bytes as f64 / 1048576.),
                evaluation_peak_mib: Some(evaluation_peak_bytes as f64 / 1048576.),
                capture_budget: engine.capture_budget(),
                step_phases,
            });
            reports::write_metrics(&output, &points)?;
            recipe_history.push((step, model.recipe_scalars()));
            reports::write_recipe_scalars(&output, epoch, step, &recipe_history)?;
            reports::write_horizon(
                &output,
                epoch,
                step,
                preview_curve.as_ref().map(|curves| HorizonSplit {
                    curve: &curves.horizon,
                    origins: preview.len(),
                }),
                full_curve.as_ref().map(|curves| HorizonSplit {
                    curve: &curves.horizon,
                    origins: corpus.validation_refs.len(),
                }),
                corpus.contract.tickers.len(),
            )?;
            reports::write_trading(
                &output,
                epoch,
                step,
                preview_curve.as_ref().map(|curves| TradingSplit {
                    curve: &curves.trading,
                    portfolio: &curves.portfolio,
                    origins: preview.len(),
                }),
                full_curve.as_ref().map(|curves| TradingSplit {
                    curve: &curves.trading,
                    portfolio: &curves.portfolio,
                    origins: corpus.validation_refs.len(),
                }),
                corpus.contract.tickers.len(),
            )?;
            reports::write_candles(&output, epoch, step, &windows)?;
            benchmark::write_hardware(
                &output,
                "CausalPatch production training (evaluation excluded)",
                &hardware_history,
            )?;
            reports::write_corpus(&output, &corpus.contract, &corpus.market)?;
            if epoch_complete {
                ensure!(
                    target_bars == corpus.contract.train_target_bars,
                    "epoch target coverage mismatch"
                );
                if evaluation.nll < best_full {
                    best_full = evaluation.nll;
                    stale_epochs = 0;
                } else {
                    stale_epochs += 1;
                }
            }
            println!("CausalPatch epoch {epoch}/{} step {step}: {target_bars}/{} unique target bars; {} NLL {:.4} (persistence {:.4}), market-neutral MSE ratio {:.4}, raw MSE ratio {:.4}; reports {}",args.epochs,corpus.contract.train_target_bars,if epoch_complete {"held-out full"} else {"held-out sample"},evaluation.nll,evaluation.persistence_nll,evaluation.mse/evaluation.persistence_mse,evaluation.absolute_mse/evaluation.absolute_persistence_mse,output.display());
            if stale_previews >= args.preview_patience {
                println!("CausalPatch stopped at step {step}: held-out sample NLL {:.4} has not improved for {stale_previews} consecutive evaluations (best {best_preview:.4} at step {}); weights/best retained", evaluation.nll, best_step.unwrap_or(0));
                stopped = true;
                break;
            }
            if !epoch_complete {
                hardware = Some(HardwareSampler::start()?);
            }
            tch::Cuda::synchronize(0);
            interval_started = Instant::now();
            interval_steps = 0;
            loader_wait_ms = 0.;
        }
        if stopped {
            break;
        }
        if stale_epochs >= args.patience {
            println!("CausalPatch stopped after {stale_epochs} complete epochs without improved held-out full NLL; weights/best retained");
            break;
        }
    }
    Ok(())
}

pub fn evaluate(args: EvaluateArgs) -> Result<()> {
    ensure!(args.batch_size > 0, "batch size must be positive");
    let manifest = Manifest::read(&args.checkpoint)?;
    let mut corpus = Corpus::load(
        &args.data_dir,
        &manifest.requested_tickers,
        manifest.data.context,
        manifest.data.pred_len,
        manifest.data.common_context,
        &manifest.data.features,
        manifest.data.market_min_cross_section,
    )?;
    ensure!(
        corpus.contract == manifest.data,
        "dataset differs from authenticated universe/splits"
    );
    let device = cuda_device()?;
    corpus.prepare(device);
    let corpus = Arc::new(corpus);
    let mut store = nn::VarStore::new(device);
    let model = CausalPatchModel::new(&store.root(), &manifest.model);
    store
        .load(args.checkpoint.join("model.safetensors"))
        .context("loading universe checkpoint")?;
    check_head_layout(
        store
            .variables()
            .get("head.output.weight")
            .context("checkpoint has no head output weight")?,
    )?;
    store.freeze();
    let result = score(
        &corpus,
        &model,
        &corpus.validation_refs,
        args.batch_size,
        device,
    )?;
    let metrics = Metrics {
        step: manifest.step,
        epoch: manifest.epoch,
        completed_origins: manifest.completed_origins,
        total_origins: corpus.train_refs.len(),
        completed_target_bars: manifest.completed_target_bars,
        total_target_bars: corpus.contract.train_target_bars,
        train_nll: None,
        train_mse: None,
        validation_nll: result.nll,
        persistence_nll: result.persistence_nll,
        validation_mse: result.mse,
        persistence_mse: result.persistence_mse,
        absolute_mse: result.absolute_mse,
        absolute_persistence_mse: result.absolute_persistence_mse,
        within_1_sigma: result.within_1_sigma,
        within_2_sigma: result.within_2_sigma,
        rmse_price: result.rmse_price,
        mae_price: result.mae_price,
        invalid_ohlc_fraction: result.invalid_fraction,
        tail_loss_share: result.tail_loss_share,
        median_window_ratio: result.median_window_ratio,
        eval: result.timing,
        step_ms: None,
        validation_is_full: true,
        validation_origins: corpus.validation_refs.len(),
        tickers: corpus.contract.tickers.len(),
        loader_wait_ms: None,
        peak_allocator_mib: None,
        evaluation_peak_mib: None,
        capture_budget: None,
        step_phases: None,
    };
    reports::write_metrics(&args.output, &[metrics])?;
    reports::write_horizon(
        &args.output,
        manifest.epoch,
        manifest.step,
        None,
        Some(HorizonSplit {
            curve: &result.horizon,
            origins: corpus.validation_refs.len(),
        }),
        corpus.contract.tickers.len(),
    )?;
    reports::write_trading(
        &args.output,
        manifest.epoch,
        manifest.step,
        None,
        Some(TradingSplit {
            curve: &result.trading,
            portfolio: &result.portfolio,
            origins: corpus.validation_refs.len(),
        }),
        corpus.contract.tickers.len(),
    )?;
    reports::write_corpus(&args.output, &corpus.contract, &corpus.market)?;
    reports::write_candles(
        &args.output,
        manifest.epoch,
        manifest.step,
        &candle_windows(&corpus, &model, device)?,
    )?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preview_origins_are_fixed_unique_and_cover_partition() {
        let origins: Vec<_> = (800..1300).collect();
        let selected = fixed_origins(&origins, 256).unwrap();
        assert_eq!(selected.len(), 256);
        assert_eq!(selected.first(), Some(&800));
        assert_eq!(selected.last(), Some(&1299));
        assert!(selected.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(fixed_origins(&origins, 1000).unwrap(), origins);
    }
    /// Synthetic final-origin forecasts: 13 windows over three uneven batches, one window with
    /// no valid bars, one with a flat close forecast, one with a zero close target.
    fn synthetic_forecast(windows: i64, pred_len: i64) -> FinalOrigin {
        tch::manual_seed(7);
        let shape = [windows, pred_len, CHANNELS];
        let cpu = (Kind::Float, Device::Cpu);
        let scaled = Tensor::randn(shape, cpu);
        let _ = scaled.select(0, 1).select(-1, CHANNELS - 1).zero_();
        let targets = Tensor::randn(shape, cpu) * 1.5;
        let _ = targets.select(0, 2).select(-1, CHANNELS - 1).zero_();
        let mask = Tensor::rand([windows, pred_len], cpu).lt(0.8).to_kind(Kind::Float);
        let _ = mask.select(0, 3).zero_();
        let drift = Tensor::randn([windows, pred_len, 1], cpu) * 0.5;
        let prices = (&scaled * 0.01).exp() * 100.;
        let target_prices = ((&targets + &drift) * 0.01).exp() * 100.;
        FinalOrigin {
            rebased_prices: ((&scaled + &drift) * 0.01).exp() * 100.,
            log_scale: Tensor::randn(shape, cpu) * 0.3,
            mid: Tensor::randn([windows], cpu) * 0.4,
            sigma: Tensor::rand([windows], cpu) * 0.01 + 0.005,
            scaled,
            targets,
            mask,
            drift,
            prices,
            target_prices,
        }
    }
    fn slice(forecast: &FinalOrigin, start: i64, rows: i64) -> FinalOrigin {
        let cut = |t: &Tensor| t.narrow(0, start, rows);
        FinalOrigin {
            rebased_prices: cut(&forecast.rebased_prices),
            scaled: cut(&forecast.scaled),
            log_scale: cut(&forecast.log_scale),
            targets: cut(&forecast.targets),
            mask: cut(&forecast.mask),
            drift: cut(&forecast.drift),
            prices: cut(&forecast.prices),
            target_prices: cut(&forecast.target_prices),
            mid: cut(&forecast.mid),
            sigma: cut(&forecast.sigma),
        }
    }
    fn host(t: &Tensor) -> Vec<f64> {
        Vec::<f64>::try_from(t.flatten(0, -1).to_kind(Kind::Double)).unwrap()
    }
    fn close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= 1e-5 * (1. + e.abs()), "{a} != {e}");
        }
    }
    #[test]
    fn tensor_scorer_matches_scalar_reference() {
        let _rng = crate::torch::test_rng::exclusive();
        let (w, h, c) = (13usize, 7usize, CHANNELS as usize);
        let forecast = synthetic_forecast(w as i64, h as i64);
        let half_log_horizon = ((Tensor::arange(h as i64, (Kind::Float, Device::Cpu)) + 1.).log() * 0.5)
            .reshape([1, 1, h as i64, 1]);
        let mut scorer = Scorer::new(
            &half_log_horizon,
            &Tensor::zeros([w as i64], (Kind::Int64, Device::Cpu)),
            1,
            h as i64,
            Device::Cpu,
        );
        let mask = host(&forecast.mask);
        let bars_in = |start: usize, rows: usize| -> usize {
            mask[start * h..(start + rows) * h].iter().sum::<f64>() as usize
        };
        for (start, rows) in [(0usize, 5usize), (5, 5), (10, 3)] {
            scorer.accumulate(&slice(&forecast, start as i64, rows as i64), bars_in(start, rows));
        }
        let evaluation = scorer.finish(EvalTiming::default()).unwrap();

        let s = host(&forecast.scaled);
        let t = host(&forecast.targets);
        let ls = host(&forecast.log_scale);
        let at = |i: usize, j: usize, k: usize| (i * h + j) * c + k;
        let m = |i: usize, j: usize| mask[i * h + j];
        let bars: f64 = mask.iter().sum();
        let elements = bars * c as f64;
        let mut sq = 0.;
        let mut pers = 0.;
        let mut within_1 = 0.;
        let mut nll = 0.;
        let mut counts = vec![0.; h];
        let mut mse = vec![0.; h];
        let mut pmse = vec![0.; h];
        let mut abs_err = vec![0.; h];
        let mut abs_target = vec![0.; h];
        let mut wins = vec![0.; h];
        let mut predicted = vec![0.; h];
        let mut hits = vec![0.; h];
        let mut ups = vec![0.; h];
        let mut window_sq = vec![0.; w];
        let mut window_pers = vec![0.; w];
        let mut bar_sq = vec![0.; w * h];
        let mut bar_pers = vec![0.; w * h];
        for i in 0..w {
            for j in 0..h {
                if m(i, j) == 0. {
                    continue;
                }
                counts[j] += 1.;
                for k in 0..c {
                    let e = at(i, j, k);
                    let r = t[e] - s[e];
                    sq += r * r;
                    pers += t[e] * t[e];
                    mse[j] += r * r;
                    pmse[j] += t[e] * t[e];
                    abs_err[j] += r.abs();
                    abs_target[j] += t[e].abs();
                    window_sq[i] += r * r;
                    window_pers[i] += t[e] * t[e];
                    bar_sq[i * h + j] += r * r;
                    bar_pers[i * h + j] += t[e] * t[e];
                    within_1 += f64::from(r.abs() <= ls[e].exp());
                    nll += 0.5 * (r / ls[e].exp()).powi(2) + ls[e];
                }
                let (tc, sc) = (t[at(i, j, c - 1)], s[at(i, j, c - 1)]);
                wins[j] += f64::from((tc - sc).abs() < tc.abs());
                predicted[j] += f64::from(sc != 0.);
                hits[j] += f64::from(sc != 0. && sc.signum() == tc.signum() && tc != 0.);
                ups[j] += f64::from(sc > 0.);
            }
        }
        assert!((evaluation.mse - sq / elements).abs() < 1e-5);
        assert!((evaluation.persistence_mse - pers / elements).abs() < 1e-5);
        assert!((evaluation.nll - nll / elements).abs() < 1e-6);
        assert!((evaluation.within_1_sigma - within_1 / elements).abs() < 1e-5);
        let horizon = &evaluation.horizon;
        let ratio = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a / b).collect::<Vec<_>>();
        let per_bar = |a: &[f64]| a.iter().zip(&counts).map(|(a, n)| a / n).collect::<Vec<_>>();
        close(&horizon.mse, &ratio(&mse, &counts.iter().map(|n| n * c as f64).collect::<Vec<_>>()));
        close(&horizon.persistence_mse, &ratio(&pmse, &counts.iter().map(|n| n * c as f64).collect::<Vec<_>>()));
        close(&horizon.mae_ratio, &ratio(&abs_err, &abs_target));
        close(&horizon.win_rate, &per_bar(&wins));
        close(&horizon.hit_rate, &ratio(&hits, &predicted));
        close(&horizon.up_fraction, &per_bar(&ups));
        assert!(predicted.iter().zip(&counts).any(|(p, n)| p < n), "flat forecasts must be excluded from hit rate");
        // Trimmed ratio: drop the top ⌈n_h/100⌉ = 1 valid |close target| per horizon.
        let mut trimmed = vec![0.; h];
        for j in 0..h {
            let magnitude = |i: usize| t[at(i, j, c - 1)].abs();
            let threshold = (0..w).filter(|&i| m(i, j) > 0.).map(magnitude).fold(f64::MIN, f64::max);
            let (mut num, mut den) = (0., 0.);
            for i in (0..w).filter(|&i| m(i, j) > 0. && magnitude(i) < threshold) {
                num += bar_sq[i * h + j];
                den += bar_pers[i * h + j];
            }
            trimmed[j] = num / den;
        }
        close(&horizon.trimmed_mse_ratio, &trimmed);
        let mut ratios: Vec<f64> = (0..w).filter(|&i| window_pers[i] > 0.).map(|i| window_sq[i] / window_pers[i]).collect();
        assert_eq!(ratios.len(), w - 1, "the empty window is not scored");
        ratios.sort_by(f64::total_cmp);
        assert!((evaluation.median_window_ratio - ratios[(ratios.len() - 1) / 2]).abs() < 1e-5);
        // Tail share: the top ⌈valid elements / 100⌉ |target| elements of each batch.
        let mut tail = 0.;
        let mut widest = 0;
        for (start, rows) in [(0usize, 5usize), (5, 5), (10, 3)] {
            let magnitude = |e: usize| t[e].abs() * m(e / (h * c), e / c % h);
            let elements = start * h * c..(start + rows) * h * c;
            let mut magnitudes: Vec<f64> = elements.clone().map(magnitude).collect();
            magnitudes.sort_by(f64::total_cmp);
            let k = (bars_in(start, rows) * c).div_ceil(100);
            widest = widest.max(k);
            let threshold = magnitudes[magnitudes.len() - k];
            for e in elements.filter(|&e| magnitude(e) >= threshold) {
                tail += (t[e] - s[e]).powi(2);
            }
        }
        assert!(widest > 1, "the reference must exercise a multi-element tail");
        assert!((evaluation.tail_loss_share - tail / sq).abs() < 1e-5);
    }
    /// The trading diagnostics against a straightforward scalar reference. Ninety windows over
    /// three thirty-window evaluation timestamps and six horizons, accumulated in three uneven
    /// batches, so both the per-timestamp cross-sections and the decile long/short clear
    /// [`CROSS_SECTION_MIN`]. Relative tolerances, all set by where the implementation's
    /// arithmetic actually happens: `1e-6` for the gain decomposition, correlations, anchored
    /// ratios and decile statistics, whose residuals and squares are formed elementwise in the
    /// resident fp32 arrays before an fp64 reduction while the reference works in fp64
    /// throughout; `1e-4` for the cross-sectional IC, whose per-timestamp scatter also
    /// accumulates in fp32; `1e-9` for the decile-spread return and Sharpe, which the
    /// backtest computes in fp64 end to end.
    #[test]
    fn trading_statistics_match_scalar_reference() {
        let _rng = crate::torch::test_rng::exclusive();
        let (w, h, c) = (90usize, 6usize, CHANNELS as usize);
        let per_group = 30usize;
        let groups = w / per_group;
        let forecast = synthetic_forecast(w as i64, h as i64);
        let ids: Vec<i64> = (0..w).map(|i| (i / per_group) as i64).collect();
        let half_log_horizon = ((Tensor::arange(h as i64, (Kind::Float, Device::Cpu)) + 1.).log()
            * 0.5)
            .reshape([1, 1, h as i64, 1]);
        let mut scorer = Scorer::new(
            &half_log_horizon,
            &Tensor::from_slice(&ids),
            groups,
            h as i64,
            Device::Cpu,
        );
        let mask = host(&forecast.mask);
        for (start, rows) in [(0usize, 40usize), (40, 25), (65, 25)] {
            let bars = mask[start * h..(start + rows) * h].iter().sum::<f64>() as usize;
            scorer.accumulate(&slice(&forecast, start as i64, rows as i64), bars);
        }
        let evaluation = scorer.finish(EvalTiming::default()).unwrap();
        let curve = &evaluation.trading;

        let s = host(&forecast.scaled);
        let t = host(&forecast.targets);
        let mid = host(&forecast.mid);
        let sigma = host(&forecast.sigma);
        let at = |i: usize, j: usize, k: usize| (i * h + j) * c + k;
        let m = |i: usize, j: usize| mask[i * h + j];
        let fc = |i: usize, j: usize| s[at(i, j, c - 1)] * m(i, j);
        let yc = |i: usize, j: usize| t[at(i, j, c - 1)] * m(i, j);
        // sign(a)·sign(b) > 0 over the bars where both signs are nonzero.
        let hit_rate = |members: &[usize], j: usize, signal: &dyn Fn(usize) -> f64| {
            let (mut hit, mut bet) = (0., 0.);
            for &i in members {
                let (p, r) = (signal(i).signum(), yc(i, j).signum());
                if signal(i) != 0. && yc(i, j) != 0. {
                    bet += 1.;
                    hit += f64::from(p * r > 0.);
                }
            }
            hit / bet
        };
        let pearson = |members: &[usize], a: &dyn Fn(usize) -> f64, b: &dyn Fn(usize) -> f64| {
            let n = members.len() as f64;
            let (ma, mb) = (
                members.iter().map(|&i| a(i)).sum::<f64>() / n,
                members.iter().map(|&i| b(i)).sum::<f64>() / n,
            );
            let cov = members.iter().map(|&i| a(i) * b(i)).sum::<f64>() / n - ma * mb;
            let va = members.iter().map(|&i| a(i) * a(i)).sum::<f64>() / n - ma * ma;
            let vb = members.iter().map(|&i| b(i) * b(i)).sum::<f64>() / n - mb * mb;
            cov / (va * vb).sqrt()
        };
        let mut usable_cross_sections = 0;
        for j in 0..h {
            let valid: Vec<usize> = (0..w).filter(|&i| m(i, j) > 0.).collect();
            let n = valid.len() as f64;
            assert!(n > 0.);
            let sum = |g: &dyn Fn(usize) -> f64| valid.iter().map(|&i| g(i)).sum::<f64>();
            let mu = sum(&|i| fc(i, j)) / n;
            let ybar = sum(&|i| yc(i, j)) / n;
            let persistence = sum(&|i| yc(i, j).powi(2)) / n;
            let error = sum(&|i| (yc(i, j) - fc(i, j)).powi(2)) / n;
            let signal_square = sum(&|i| fc(i, j).powi(2)) / n - mu * mu;
            let signal_target = sum(&|i| fc(i, j) * yc(i, j)) / n - mu * ybar;
            let total = 1. - error / persistence;
            let offset = (2. * mu * ybar - mu * mu) / persistence;
            let demeaned = signal_target * signal_target / signal_square / persistence;
            let tight = |actual: f64, expected: f64, what: &str| {
                assert!(
                    (actual - expected).abs() <= 1e-6 * (1. + expected.abs()),
                    "{what} at h={}: {actual} != {expected}",
                    j + 1
                );
            };
            tight(curve.total_gain[j], total, "total gain");
            tight(curve.offset_gain[j], offset, "offset gain");
            tight(curve.demeaned_gain[j], demeaned, "demeaned gain");
            tight(
                curve.scaling_gain[j],
                total - offset - demeaned,
                "cross term",
            );
            // The cross term is exactly the cost of mis-scaling the demeaned forecast, which
            // is what makes the three components a decomposition rather than three numbers.
            let slope = signal_target / signal_square;
            tight(
                curve.scaling_gain[j],
                -(slope - 1.).powi(2) * signal_square / persistence,
                "cross term identity",
            );
            tight(curve.mean_forecast[j], mu, "mean forecast");
            tight(curve.mean_target[j], ybar, "mean target");
            tight(curve.close_mse_ratio[j], error / persistence, "MSE ratio");
            let all_error = sum(&|i| (0..c).map(|k| (t[at(i, j, k)] - s[at(i, j, k)]).powi(2)).sum());
            let all_persistence = sum(&|i| (0..c).map(|k| t[at(i, j, k)].powi(2)).sum());
            tight(
                curve.all_channel_gain[j],
                1. - all_error / all_persistence,
                "all-channel gain",
            );
            tight(
                curve.pearson[j],
                pearson(&valid, &|i| fc(i, j), &|i| yc(i, j)),
                "Pearson IC",
            );
            // Spearman: ascending ordinal ranks with invalid bars pushed past every valid one.
            let key = |value: f64, valid: bool| if valid { value } else { 1e30 };
            let ranks = |g: &dyn Fn(usize) -> f64| -> Vec<f64> {
                let keys: Vec<f64> = (0..w).map(|i| key(g(i), m(i, j) > 0.)).collect();
                (0..w)
                    .map(|i| {
                        (0..w)
                            .filter(|&o| keys[o] < keys[i] || (keys[o] == keys[i] && o < i))
                            .count() as f64
                    })
                    .collect()
            };
            let (rf, ry) = (ranks(&|i| fc(i, j)), ranks(&|i| yc(i, j)));
            tight(
                curve.spearman[j],
                pearson(&valid, &|i| rf[i], &|i| ry[i]),
                "Spearman IC",
            );
            // Mid anchor: the numerator is unchanged, the denominator re-anchors on the log
            // high/low midpoint of the origin bar.
            tight(
                curve.mid_anchor_mse_ratio[j],
                sum(&|i| (yc(i, j) - fc(i, j)).powi(2)) / sum(&|i| (yc(i, j) - mid[i]).powi(2)),
                "mid-anchored MSE ratio",
            );
            let mid_hit = {
                let (mut hit, mut bet) = (0., 0.);
                for &i in &valid {
                    let (p, r) = (fc(i, j) - mid[i], yc(i, j) - mid[i]);
                    if p != 0. && r != 0. {
                        bet += 1.;
                        hit += f64::from(p.signum() * r.signum() > 0.);
                    }
                }
                hit / bet
            };
            tight(curve.mid_anchor_hit_rate[j], mid_hit, "mid-anchored hit rate");
            tight(
                curve.close_hit_rate[j],
                hit_rate(&valid, j, &|i| fc(i, j)),
                "close hit rate",
            );
            // One-bar execution delay.
            let delayed: Vec<usize> = (0..w)
                .filter(|&i| m(i, j) > 0. && m(i, 0) > 0.)
                .collect();
            let (mut delay_error, mut delay_persistence) = (0., 0.);
            let (mut delay_hit, mut delay_bet) = (0., 0.);
            for &i in &delayed {
                let (d, p) = (yc(i, j) - yc(i, 0), fc(i, j) - fc(i, 0));
                delay_error += (d - p).powi(2);
                delay_persistence += d * d;
                if d != 0. && p != 0. {
                    delay_bet += 1.;
                    delay_hit += f64::from(d.signum() * p.signum() > 0.);
                }
            }
            if j == 0 {
                assert_eq!(delay_persistence, 0.);
                assert!(
                    curve.delayed_mse_ratio[0].is_nan(),
                    "the delayed move is identically zero at h = 1"
                );
            } else {
                tight(
                    curve.delayed_mse_ratio[j],
                    delay_error / delay_persistence,
                    "delayed MSE ratio",
                );
                tight(
                    curve.delayed_hit_rate[j],
                    delay_hit / delay_bet,
                    "delayed hit rate",
                );
            }
            // Conviction deciles on the demeaned signal.
            let magnitude = |i: usize| (fc(i, j) - mu).abs();
            let k = ((n / 10.).floor() as usize).max(1);
            assert!(k >= 2, "the reference must exercise a multi-member decile");
            let mut sorted: Vec<f64> = valid.iter().map(|&i| magnitude(i)).collect();
            sorted.sort_by(f64::total_cmp);
            let (low, high) = (sorted[k - 1], sorted[sorted.len() - k]);
            let top: Vec<usize> = valid.iter().copied().filter(|&i| magnitude(i) >= high).collect();
            let bottom: Vec<usize> = valid.iter().copied().filter(|&i| magnitude(i) <= low).collect();
            let side = |members: &[usize]| {
                members
                    .iter()
                    .map(|&i| (fc(i, j) - mu).signum() * yc(i, j))
                    .sum::<f64>()
                    / members.len() as f64
            };
            tight(curve.top_decile_return[j], side(&top), "top-decile return");
            tight(
                curve.bottom_decile_return[j],
                side(&bottom),
                "bottom-decile return",
            );
            tight(
                curve.conviction_spread_return[j],
                side(&top) - side(&bottom),
                "conviction spread",
            );
            tight(
                curve.top_decile_hit_rate[j],
                hit_rate(&top, j, &|i| fc(i, j) - mu),
                "top-decile hit rate",
            );
            tight(
                curve.bottom_decile_hit_rate[j],
                hit_rate(&bottom, j, &|i| fc(i, j) - mu),
                "bottom-decile hit rate",
            );
            // Cross-sectional IC: within-timestamp correlation, then averaged.
            let mut moments = Vec::new();
            for g in 0..groups {
                let members: Vec<usize> = (g * per_group..(g + 1) * per_group)
                    .filter(|&i| m(i, j) > 0.)
                    .collect();
                if members.len() < CROSS_SECTION_MIN as usize {
                    continue;
                }
                moments.push(pearson(&members, &|i| fc(i, j), &|i| yc(i, j)));
            }
            usable_cross_sections += moments.len();
            let count = moments.len() as f64;
            let ic = moments.iter().sum::<f64>() / count;
            let variance = moments.iter().map(|v| (v - ic).powi(2)).sum::<f64>() / count;
            assert!(
                (curve.cross_sectional_ic[j] - ic).abs() <= 1e-4,
                "cross-sectional IC at h={}: {} != {ic}",
                j + 1,
                curve.cross_sectional_ic[j]
            );
            assert!(
                (curve.cross_sectional_ic_se[j] - (variance / count).sqrt()).abs() <= 1e-4,
                "cross-sectional IC standard error at h={}",
                j + 1
            );
        }
        assert!(
            usable_cross_sections >= h,
            "the reference must exercise the cross-sectional path at every horizon"
        );
        assert_eq!(curve.cross_sections, groups);

        // Decile long/short: rank each timestamp's valid tickers by predicted return.
        let portfolio = &evaluation.portfolio;
        assert_eq!(portfolio.horizons, vec![1, 4]);
        for (index, horizon) in portfolio.horizons.iter().enumerate() {
            let j = *horizon as usize - 1;
            let mut spreads = Vec::new();
            for g in 0..groups {
                let mut members: Vec<usize> = (g * per_group..(g + 1) * per_group)
                    .filter(|&i| m(i, j) > 0.)
                    .collect();
                if members.len() < CROSS_SECTION_MIN as usize {
                    continue;
                }
                let side = members.len() / 10;
                members.sort_by(|&a, &b| {
                    (fc(a, j) * sigma[a])
                        .total_cmp(&(fc(b, j) * sigma[b]))
                        .then(a.cmp(&b))
                });
                let leg = |slice: &[usize]| {
                    slice.iter().map(|&i| yc(i, j) * sigma[i]).sum::<f64>() / side as f64
                };
                spreads.push(leg(&members[members.len() - side..]) - leg(&members[..side]));
            }
            let count = spreads.len() as f64;
            assert!(count > 1., "the Sharpe reference needs several timestamps");
            assert_eq!(portfolio.cross_sections[index], spreads.len());
            let mean = spreads.iter().sum::<f64>() / count;
            let deviation = (spreads.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count).sqrt();
            for (level, cost) in portfolio.costs_bps.iter().enumerate() {
                let net = mean * 1e4 - 4. * cost;
                let annualized = net / (deviation * 1e4) * (BARS_PER_YEAR / *horizon as f64).sqrt();
                assert!(
                    (portfolio.net_bps[level][index] - net).abs() <= 1e-9 * (1. + net.abs()),
                    "net bps at h={horizon} cost={cost}: {} != {net}",
                    portfolio.net_bps[level][index]
                );
                assert!(
                    (portfolio.net_sharpe[level][index] - annualized).abs()
                        <= 1e-9 * (1. + annualized.abs()),
                    "net Sharpe at h={horizon} cost={cost}: {} != {annualized}",
                    portfolio.net_sharpe[level][index]
                );
            }
        }
        assert!(
            portfolio.net_bps[0][0] > portfolio.net_bps[4][0],
            "a higher per-side cost must lower the net return"
        );
    }
    #[test]
    fn universe_checkpoint_authenticates_objective_schema_and_weight_bytes() {
        use super::super::{corpus::ExcludedTicker, data::DataContract, features::FeatureSet};
        struct Scratch(PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
        let directory = Scratch(PathBuf::from(format!(
            "/var/tmp/timexer-manifest-test-{}",
            uuid::Uuid::new_v4()
        )));
        fs::create_dir(&directory.0).unwrap();
        let payload = b"test-only weight payload; no model execution";
        fs::write(directory.0.join("model.safetensors"), payload).unwrap();
        let model = ModelConfig {
            seq_len: 16,
            pred_len: 7,
            min_history: 16,
            features: FeatureSet::NONE,
            ..ModelConfig::default()
        };
        let ticker = DataContract {
            schema: "test-sigma-scaled".into(),
            ticker: "EXAMPLE".into(),
            fingerprint: "0123456789abcdef".into(),
            source_bars: 10_000,
            valid_bars: 10_000,
            invalid_ohlc_indices: vec![],
            boundaries: [7000, 8000, 9000],
            boundary_timestamps: [700_000, 800_000, 900_000],
            context: 16,
            pred_len: 7,
            purge: 100,
            train_end: 6900,
            common_context: 32,
        };
        let data = CorpusContract {
            schema: "authenticated-pooled-training".into(),
            tickers: vec![ticker],
            boundary_timestamps: [700_000, 800_000, 900_000],
            context: 16,
            pred_len: 7,
            common_context: 32,
            purge: 100,
            features: FeatureSet::NONE,
            auxiliary_schema: String::new(),
            spy_fingerprint: None,
            market_fingerprint: "fedcba9876543210".into(),
            market_min_cross_section: 2000,
            minimum_source_bars: 133,
            minimum_training_bars: 33,
            train_target_bars: 6868,
            validation_target_bars: 896,
            validation_remainder_bars: 4,
            excluded_tickers: vec![ExcludedTicker {
                ticker: "NEW".into(),
                reason: "no historical training targets".into(),
            }],
        };
        let mut manifest = Manifest {
            format: FORMAT.into(),
            model,
            data,
            objective: OBJECTIVE.into(),
            numerics: NUMERICS.into(),
            requested_tickers: vec![],
            seed: 20260905,
            epoch: 1,
            step: 1000,
            completed_origins: 256000,
            completed_target_bars: 49152000,
            epoch_complete: false,
            planned_epochs: 10,
            batch_size: 256,
            learning_rate: 0.008,
            fused: true,
            optimizer: OptimizerKind::PolarExpress,
            optimizer_recipe: OptimizerKind::PolarExpress.recipe().into(),
            eval_every: 1000,
            eval_origins: 2048,
            validation_nll: 1.3,
            validation_mse: 0.004194157171231031,
            validation_is_full: false,
            best_step: Some(1000),
            best_preview_nll: Some(1.3),
            weights_sha256: file_sha256(directory.0.join("model.safetensors")).unwrap(),
            manifest_sha256: String::new(),
        };
        manifest.manifest_sha256 = manifest.digest().unwrap();
        let write = |value: &Manifest| {
            fs::write(
                directory.0.join("manifest.json"),
                serde_json::to_vec_pretty(value).unwrap(),
            )
            .unwrap()
        };
        write(&manifest);
        let restored = Manifest::read(&directory.0).unwrap();
        assert_eq!(restored.digest().unwrap(), manifest.digest().unwrap());
        assert_eq!(restored.data, manifest.data);
        let mut previous_objective = manifest.clone();
        previous_objective.objective = "causal_patch_nll_v1".into();
        previous_objective.manifest_sha256 = previous_objective.digest().unwrap();
        write(&previous_objective);
        let error = Manifest::read(&directory.0).unwrap_err().to_string();
        assert!(error.contains("objective") && error.contains(OBJECTIVE), "{error}");
        // The pre-channel-major stamp specifically: `causal-patch-ohlc-universe-v5` names a
        // head whose weight rows are horizon-major, and every tensor name and shape in it still
        // matches the current model, so nothing but this string stands between a stale
        // checkpoint and a silently permuted head. `…-v6-head-channel-major-folded-mup` is the
        // pre-recipe backbone: its state dict carries 17 norm gains and biases and four block
        // biases per layer that this model does not have, and none of the recipe scalars that
        // it does, so it must be refused by name rather than half-loaded.
        for stale in [
            "timexer-ohlc-universe-v4",
            "causal-patch-ohlc-universe-v5",
            "causal-patch-ohlc-universe-v6-head-channel-major-folded-mup",
        ] {
            let mut previous_format = manifest.clone();
            previous_format.format = stale.into();
            previous_format.manifest_sha256 = previous_format.digest().unwrap();
            write(&previous_format);
            let error = Manifest::read(&directory.0).unwrap_err().to_string();
            assert!(
                error.contains("format") && error.contains(stale) && error.contains(FORMAT),
                "{error}"
            );
        }
        write(&manifest);
        let mut corrupted = manifest.clone();
        corrupted.data.tickers[0].train_end += 1;
        write(&corrupted);
        assert!(Manifest::read(&directory.0)
            .unwrap_err()
            .to_string()
            .contains("manifest authentication"));
        let mut corrupted = manifest.clone();
        corrupted.data.excluded_tickers[0].ticker = "REPLACED".into();
        write(&corrupted);
        assert!(Manifest::read(&directory.0)
            .unwrap_err()
            .to_string()
            .contains("manifest authentication"));
        let mut mismatched = manifest.clone();
        mismatched.data.context = 32;
        mismatched.manifest_sha256 = mismatched.digest().unwrap();
        write(&mismatched);
        assert!(Manifest::read(&directory.0)
            .unwrap_err()
            .to_string()
            .contains("model/data contract mismatch"));
        write(&manifest);
        fs::write(directory.0.join("model.safetensors"), b"changed payload").unwrap();
        assert!(Manifest::read(&directory.0)
            .unwrap_err()
            .to_string()
            .contains("weight authentication"));
    }
    /// A head weight whose rows are horizon-major is refused, and the channel-major one it was
    /// permuted from is accepted. The synthetic weight carries the structure the guard relies
    /// on and nothing else: a per-channel signature three times the size of a step, plus a
    /// random walk along the horizon, so within a channel the rows drift smoothly and across
    /// channels they are unrelated.
    #[test]
    fn horizon_major_head_weights_are_refused() {
        let _rng = crate::torch::test_rng::exclusive();
        tch::manual_seed(11);
        let (channels, pred_len, hidden) = (2 * CHANNELS, 24i64, 16i64);
        let cpu = (Kind::Float, Device::Cpu);
        let walk = Tensor::randn([channels, pred_len, hidden], cpu).cumsum(1, Kind::Float);
        let signature = Tensor::randn([channels, 1, hidden], cpu) * 3.;
        let major = (walk + signature).reshape([channels * pred_len, hidden]);
        check_head_layout(&major).expect("the channel-major layout the model reshapes to");
        let stale = major
            .reshape([channels, pred_len, hidden])
            .transpose(0, 1)
            .contiguous()
            .reshape([channels * pred_len, hidden]);
        let error = check_head_layout(&stale).unwrap_err().to_string();
        assert!(
            error.contains("horizon-major") && error.contains("retrain"),
            "{error}"
        );
        // An untrained head carries no layout, so the guard must not invent a verdict.
        check_head_layout(&Tensor::zeros([channels * pred_len, hidden], cpu)).unwrap();
    }
    /// The invariant that would have caught the permuted head with no checkpoint at all: an
    /// exact-zero forecast IS persistence, so every ratio the [`Scorer`] reports against
    /// persistence must be exactly 1 at every horizon - in the market-neutral space and in the
    /// absolute space that adds the realized market drift back - and every directional and
    /// correlation statistic must sit at its null. Nonzero per-window σ and a nonzero
    /// `½·ln h` prior are supplied so the σ and `√h` scalings are exercised rather than
    /// bypassed; the two anchored ratios that are *not* 1 (the mid anchor moves the
    /// denominator, and mid ≠ 0) are pinned against their closed form.
    #[test]
    fn zero_forecast_scores_exactly_persistence_at_every_horizon() {
        let _rng = crate::torch::test_rng::exclusive();
        let (w, h) = (90usize, 6usize);
        let per_group = 30usize;
        let mut forecast = synthetic_forecast(w as i64, h as i64);
        forecast.scaled = forecast.scaled.zeros_like();
        forecast.prices = (&forecast.scaled * 0.01).exp() * 100.;
        forecast.rebased_prices = ((&forecast.scaled + &forecast.drift) * 0.01).exp() * 100.;
        let ids: Vec<i64> = (0..w).map(|i| (i / per_group) as i64).collect();
        let half_log_horizon = ((Tensor::arange(h as i64, (Kind::Float, Device::Cpu)) + 1.).log()
            * 0.5)
            .reshape([1, 1, h as i64, 1]);
        let mut scorer = Scorer::new(
            &half_log_horizon,
            &Tensor::from_slice(&ids),
            w / per_group,
            h as i64,
            Device::Cpu,
        );
        let mask = host(&forecast.mask);
        for (start, rows) in [(0usize, 40usize), (40, 25), (65, 25)] {
            let bars = mask[start * h..(start + rows) * h].iter().sum::<f64>() as usize;
            scorer.accumulate(&slice(&forecast, start as i64, rows as i64), bars);
        }
        let evaluation = scorer.finish(EvalTiming::default()).unwrap();
        let curve = &evaluation.trading;
        let horizon = &evaluation.horizon;
        assert_eq!(evaluation.mse, evaluation.persistence_mse);
        assert_eq!(evaluation.absolute_mse, evaluation.absolute_persistence_mse);
        assert_eq!(evaluation.median_window_ratio, 1.);
        for j in 0..h {
            // Both spaces, per horizon, exactly - these are the same sums divided by the same
            // counts, so anything but bit equality means the forecast reached the denominator.
            assert_eq!(horizon.mse[j], horizon.persistence_mse[j], "horizon {j}");
            assert_eq!(
                horizon.absolute_mse[j], horizon.absolute_persistence_mse[j],
                "horizon {j}"
            );
            assert_eq!(horizon.trimmed_mse_ratio[j], 1., "horizon {j}");
            assert_eq!(horizon.mae_ratio[j], 1., "horizon {j}");
            assert_eq!(horizon.win_rate[j], 0., "horizon {j}");
            assert_eq!(horizon.up_fraction[j], 0., "horizon {j}");
            assert!(horizon.hit_rate[j].is_nan(), "horizon {j}");
            assert_eq!(curve.close_mse_ratio[j], 1., "horizon {j}");
            assert_eq!(curve.total_gain[j], 0., "horizon {j}");
            assert_eq!(curve.all_channel_gain[j], 0., "horizon {j}");
            assert_eq!(curve.offset_gain[j], 0., "horizon {j}");
            assert_eq!(curve.demeaned_gain[j], 0., "horizon {j}");
            assert_eq!(curve.scaling_gain[j], 0., "horizon {j}");
            assert_eq!(curve.mean_forecast[j], 0., "horizon {j}");
            // A flat forecast is no directional claim in the space it is flat in, and no
            // ranking, so those rates and correlations are undefined rather than 0.5 or 0. The
            // mid anchor is the exception and is checked against its reference below: a zero
            // close forecast is still a claim about the move away from the mid.
            assert!(curve.close_hit_rate[j].is_nan(), "horizon {j}");
            assert!(curve.pearson[j].is_nan(), "horizon {j}");
            assert!(curve.cross_sectional_ic[j].is_nan(), "horizon {j}");
            assert_eq!(curve.cross_sectional_ic_se[j], 0., "horizon {j}");
            assert_eq!(curve.top_decile_return[j], 0., "horizon {j}");
            assert_eq!(curve.bottom_decile_return[j], 0., "horizon {j}");
            assert_eq!(curve.conviction_spread_return[j], 0., "horizon {j}");
            // The delayed move of a flat forecast is flat, so delayed persistence is matched
            // exactly too; h = 1 has no delayed move at all.
            if j == 0 {
                assert!(curve.delayed_mse_ratio[j].is_nan(), "horizon {j}");
            } else {
                assert_eq!(curve.delayed_mse_ratio[j], 1., "horizon {j}");
            }
        }
        // The mid anchor is the one place a zero close forecast is still a claim: the ratio must
        // NOT be 1 (only the denominator moves) and the hit rate is the rate at which leaning
        // from the trade close toward the mid is the right side.
        let t = host(&forecast.targets);
        let mid = host(&forecast.mid);
        let c = CHANNELS as usize;
        for j in 0..h {
            let (mut model, mut anchored) = (0., 0.);
            let (mut hit, mut bet) = (0., 0.);
            for i in 0..w {
                let m = mask[i * h + j];
                let y = t[(i * h + j) * c + c - 1] * m;
                model += (y * m).powi(2);
                let (anchor_target, anchor_forecast) = ((y - mid[i]) * m, -mid[i] * m);
                anchored += anchor_target.powi(2);
                if anchor_forecast != 0. && anchor_target != 0. {
                    bet += 1.;
                    hit += f64::from(anchor_forecast.signum() * anchor_target.signum() > 0.);
                }
            }
            close(&[curve.mid_anchor_mse_ratio[j]], &[model / anchored]);
            assert!(curve.mid_anchor_mse_ratio[j] != 1.);
            assert!(bet > 0., "the mid anchor reference must take sides");
            close(&[curve.mid_anchor_hit_rate[j]], &[hit / bet]);
        }
        for level in evaluation
            .portfolio
            .net_bps
            .iter()
            .chain(&evaluation.portfolio.net_sharpe)
        {
            assert!(level.iter().all(|v| v.is_finite()), "{level:?}");
        }
    }
}
