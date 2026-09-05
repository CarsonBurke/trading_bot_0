use super::{
    benchmark::{self, HardwareSampler},
    compute::{Engine, OptimizerKind},
    corpus::{Batch, Corpus, CorpusContract, WindowRef},
    geometry,
    model::{ModelConfig, SegmentModel},
    reports::{self, CandleWindow, Metrics},
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
    /// Completed full-corpus epochs without improved full-validation MSE.
    #[arg(long, default_value_t = 3)]
    pub patience: usize,
    #[arg(long, default_value_t=true, action=clap::ArgAction::Set)]
    pub fused: bool,
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
            fused: true,
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
    /// Compare the checkpoint with its exact weighted valid-candle projection.
    #[arg(long)]
    pub project_candles: bool,
}
const FORMAT: &str = "timexer-ohlc-universe-v3";
const OBJECTIVE: &str = "MSE-OHLC-train-standardized-levels-disjoint-target-pass-masked-tails";
const NUMERICS: &str =
    "fp32-masters-normalization-bf16-SDPA-nanogpt-lr-cooldown-frac=.60-floor=.15";
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
    validation_mse: f64,
    validation_is_full: bool,
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
        let manifest: Self = serde_json::from_slice(&fs::read(directory.join("manifest.json"))?)?;
        ensure!(
            manifest.format == FORMAT
                && manifest.objective == OBJECTIVE
                && (manifest.numerics == NUMERICS
                    || manifest.numerics == "fp32-masters-normalization-bf16-SDPA-type1-epoch-LR"),
            "unsupported universe checkpoint contract"
        );
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
                && manifest.model.volume_features == manifest.data.volume_features,
            "model/data contract mismatch"
        );
        Ok(manifest)
    }
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

struct Evaluation {
    mse: f64,
    projected_mse: Option<f64>,
    persistence_mse: f64,
    rmse_price: f64,
    mae_price: f64,
    invalid_fraction: f64,
    projected_invalid_fraction: Option<f64>,
    elapsed_ms: f64,
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
fn score(
    corpus: &Arc<Corpus>,
    model: &SegmentModel,
    origins: &[WindowRef],
    batch_size: usize,
    device: Device,
    project_candles: bool,
) -> Result<Evaluation> {
    ensure!(
        batch_size > 0 && !origins.is_empty(),
        "evaluation batch/origins must be nonempty"
    );
    let _guard = tch::no_grad_guard();
    tch::Cuda::synchronize(0);
    let started = Instant::now();
    let mut sums = Tensor::zeros([7], (Kind::Double, device));
    let mut bars = 0usize;
    let loader = Prefetcher::new(Arc::clone(corpus));
    loader.request(&origins[..origins.len().min(batch_size)])?;
    for (index, refs) in origins.chunks(batch_size).enumerate() {
        let host = loader.receive()?;
        let next = (index + 1) * batch_size;
        if next < origins.len() {
            loader.request(&origins[next..(next + batch_size).min(origins.len())])?;
        }
        bars += host.valid_target_bars;
        let batch = host.to_device(device);
        let forecast = model.forward_with_aux(
            &batch.inputs,
            batch.auxiliary.as_ref(),
            &batch.price_scaling,
            &batch.geometry_context,
            false,
        );
        let prediction = &forecast.standardized;
        let errors = prediction - &batch.targets;
        let persistence = batch
            .inputs
            .narrow(1, corpus.contract.context as i64 - 1, 1)
            .expand_as(&batch.targets);
        let prices = &forecast.prices;
        let price_errors = &errors * batch.price_scaling.select(1, 1).unsqueeze(1);
        let invalid = invalid_candles(prices);
        let mask = batch.target_mask.unsqueeze(-1);
        let projected =
            project_candles.then(|| geometry::project(prediction, &batch.price_scaling));
        let projected_error = projected.as_ref().map_or_else(
            || Tensor::zeros([], (Kind::Double, device)),
            |p| ((&p.standardized - &batch.targets).square() * &mask).sum(Kind::Double),
        );
        let projected_invalid = projected.as_ref().map_or_else(
            || Tensor::zeros([], (Kind::Double, device)),
            |p| {
                (invalid_candles(&p.prices).to_kind(Kind::Float) * &batch.target_mask)
                    .sum(Kind::Double)
            },
        );
        sums += Tensor::stack(
            &[
                (errors.square() * &mask).sum(Kind::Double),
                ((persistence - &batch.targets).square() * &mask).sum(Kind::Double),
                (price_errors.square() * &mask).sum(Kind::Double),
                (price_errors.abs() * &mask).sum(Kind::Double),
                (invalid.to_kind(Kind::Float) * &batch.target_mask).sum(Kind::Double),
                projected_error,
                projected_invalid,
            ],
            0,
        );
    }
    let sums = Vec::<f64>::try_from(sums.to_device(Device::Cpu))?;
    ensure!(
        sums.iter().all(|x| x.is_finite()) && bars > 0,
        "nonfinite universe validation outputs"
    );
    Ok(Evaluation {
        mse: sums[0] / (bars as f64 * 4.),
        projected_mse: project_candles.then_some(sums[5] / (bars as f64 * 4.)),
        persistence_mse: sums[1] / (bars as f64 * 4.),
        rmse_price: (sums[2] / (bars as f64 * 4.)).sqrt(),
        mae_price: sums[3] / (bars as f64 * 4.),
        invalid_fraction: sums[4] / bars as f64,
        projected_invalid_fraction: project_candles.then_some(sums[6] / bars as f64),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.,
    })
}
fn candle_windows(
    corpus: &Corpus,
    model: &SegmentModel,
    device: Device,
    project_candles: bool,
) -> Result<Vec<CandleWindow>> {
    let _guard = tch::no_grad_guard();
    let refs = fixed_origins(&corpus.validation_refs, 4)?;
    let batch = corpus.batch(&refs, device)?;
    let prediction = model.forward_with_aux(
        &batch.inputs,
        batch.auxiliary.as_ref(),
        &batch.price_scaling,
        &batch.geometry_context,
        false,
    );
    let prices = if project_candles {
        geometry::project(&prediction.standardized, &batch.price_scaling).prices
    } else {
        prediction.prices
    };
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
            && args.patience > 0,
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
    println!("TimeXer loading eligible ticker universe and authenticating shared chronological partitions");
    let mut corpus = Corpus::load(
        &args.data_dir,
        &args.ticker,
        args.model.seq_len as usize,
        args.model.pred_len as usize,
        args.common_context,
        args.model.volume_features,
    )?;
    corpus.prepare(device);
    let corpus = Arc::new(corpus);
    fs::write(
        run.root.join("timexer-segment-data-contract.json"),
        serde_json::to_vec_pretty(&corpus.contract)?,
    )?;
    reports::write_corpus(&run.gens.join("1"), &corpus.contract)?;
    tch::manual_seed(args.seed as i64);
    tch::Cuda::manual_seed_all(args.seed);
    let store = nn::VarStore::new(device);
    let model = SegmentModel::new(&store.root(), &args.model);
    let preview = fixed_origins(&corpus.validation_refs, args.eval_origins)?;
    let mut origins = corpus.train_refs.clone();
    let scheduled_steps = origins.len().div_ceil(args.batch_size) * args.epochs;
    let mut engine = Engine::new(
        &store,
        base_learning_rate,
        args.fused,
        args.optimizer,
        scheduled_steps,
    )?;
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let mut step = 0;
    let mut best = f64::INFINITY;
    let mut stale = 0;
    let loader = Prefetcher::new(Arc::clone(&corpus));
    println!("TimeXer: {} tickers, {} training target bars, {} segments, {} optimizer steps per full epoch; context {}, horizon {}, batch {}",corpus.contract.tickers.len(),corpus.contract.train_target_bars,origins.len(),origins.len().div_ceil(args.batch_size),args.model.seq_len,args.model.pred_len,args.batch_size);
    for epoch in 1..=args.epochs {
        origins.shuffle(&mut rng);
        let mut completed = 0;
        let mut target_bars = 0;
        let mut total_loss = Tensor::zeros([], (Kind::Double, device));
        let mut points = Vec::new();
        let mut hardware_history = Vec::new();
        let mut hardware_offset = 0.;
        let mut hardware = Some(HardwareSampler::start()?);
        benchmark::cuda_memory(true)?;
        tch::Cuda::synchronize(0);
        let mut interval_started = Instant::now();
        let mut interval_steps = 0;
        let mut loader_wait_ms = 0.;
        loader.request(&origins[..origins.len().min(args.batch_size)])?;
        for (index, refs) in origins.chunks(args.batch_size).enumerate() {
            let waiting = Instant::now();
            let host = loader.receive()?;
            loader_wait_ms += waiting.elapsed().as_secs_f64() * 1000.;
            let next = (index + 1) * args.batch_size;
            if next < origins.len() {
                loader.request(&origins[next..(next + args.batch_size).min(origins.len())])?;
            }
            let valid_bars = host.valid_target_bars;
            let batch = host.to_device(device);
            let loss = engine.step(
                &model,
                &batch.inputs,
                &batch.targets,
                &batch.price_scaling,
                &batch.geometry_context,
                batch.auxiliary.as_ref(),
                Some(&batch.target_mask),
            )?;
            total_loss += loss.to_kind(Kind::Double) * valid_bars as f64;
            step += 1;
            interval_steps += 1;
            completed += refs.len();
            target_bars += valid_bars;
            let epoch_complete = completed == origins.len();
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
            let evaluation = score(&corpus, &model, selected, args.batch_size, device, false)?;
            let windows = candle_windows(&corpus, &model, device, false)?;
            let output = run.gens.join(epoch.to_string());
            points.push(Metrics {
                step,
                epoch,
                completed_origins: completed,
                total_origins: origins.len(),
                completed_target_bars: target_bars,
                total_target_bars: corpus.contract.train_target_bars,
                train_mse: Some(total_loss.double_value(&[]) / target_bars as f64),
                validation_mse: evaluation.mse,
                projected_mse: None,
                persistence_mse: evaluation.persistence_mse,
                rmse_price: evaluation.rmse_price,
                mae_price: evaluation.mae_price,
                invalid_ohlc_fraction: evaluation.invalid_fraction,
                projected_invalid_fraction: None,
                eval_ms: evaluation.elapsed_ms,
                step_ms: Some(step_ms),
                validation_is_full: epoch_complete,
                validation_origins: selected.len(),
                loader_wait_ms: Some(loader_wait_ms / interval_steps as f64),
                peak_allocator_mib: Some(peak_bytes as f64 / 1048576.),
            });
            reports::write_metrics(&output, &points)?;
            reports::write_candles(&output, false, epoch, step, &windows)?;
            benchmark::write_hardware(
                &output,
                "TimeXer production training (evaluation excluded)",
                &hardware_history,
            )?;
            reports::write_corpus(&output, &corpus.contract)?;
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
                validation_mse: evaluation.mse,
                validation_is_full: epoch_complete,
                weights_sha256: String::new(),
                manifest_sha256: String::new(),
            };
            let checkpoint = if epoch_complete {
                format!("epoch-{epoch:04}")
            } else {
                "preview-latest".into()
            };
            save(&run.weights.join(&checkpoint), &store, manifest)?;
            if epoch_complete {
                ensure!(
                    target_bars == corpus.contract.train_target_bars,
                    "epoch target coverage mismatch"
                );
                if evaluation.mse < best {
                    best = evaluation.mse;
                    stale = 0;
                    crate::torch::single_ticker_timexer::checkpoint::update_best(
                        &run.weights,
                        &checkpoint,
                    )?;
                } else {
                    stale += 1;
                }
            }
            println!("TimeXer epoch {epoch}/{} step {step}: {target_bars}/{} unique target bars; {} reports {}",args.epochs,corpus.contract.train_target_bars,if epoch_complete {"full validation"} else {"fixed preview"},output.display());
            if !epoch_complete {
                hardware = Some(HardwareSampler::start()?);
            }
            tch::Cuda::synchronize(0);
            interval_started = Instant::now();
            interval_steps = 0;
            loader_wait_ms = 0.;
        }
        if stale >= args.patience {
            println!("TimeXer stopped after {stale} complete epochs without improved full-validation MSE; best checkpoint retained");
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
        manifest.data.volume_features,
    )?;
    ensure!(
        corpus.contract == manifest.data,
        "dataset differs from authenticated universe/splits/scalers"
    );
    let device = cuda_device()?;
    corpus.prepare(device);
    let corpus = Arc::new(corpus);
    let mut store = nn::VarStore::new(device);
    let model = SegmentModel::new(&store.root(), &manifest.model);
    store
        .load(args.checkpoint.join("model.safetensors"))
        .context("loading universe checkpoint")?;
    store.freeze();
    let result = score(
        &corpus,
        &model,
        &corpus.validation_refs,
        args.batch_size,
        device,
        args.project_candles,
    )?;
    let metrics = Metrics {
        step: manifest.step,
        epoch: manifest.epoch,
        completed_origins: manifest.completed_origins,
        total_origins: corpus.train_refs.len(),
        completed_target_bars: manifest.completed_target_bars,
        total_target_bars: corpus.contract.train_target_bars,
        train_mse: None,
        validation_mse: result.mse,
        projected_mse: result.projected_mse,
        persistence_mse: result.persistence_mse,
        rmse_price: result.rmse_price,
        mae_price: result.mae_price,
        invalid_ohlc_fraction: result.invalid_fraction,
        projected_invalid_fraction: result.projected_invalid_fraction,
        eval_ms: result.elapsed_ms,
        step_ms: None,
        validation_is_full: true,
        validation_origins: corpus.validation_refs.len(),
        loader_wait_ms: None,
        peak_allocator_mib: None,
    };
    reports::write_metrics(&args.output, &[metrics])?;
    reports::write_corpus(&args.output, &corpus.contract)?;
    reports::write_candles(
        &args.output,
        args.project_candles,
        manifest.epoch,
        manifest.step,
        &candle_windows(&corpus, &model, device, args.project_candles)?,
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
    #[test]
    fn universe_checkpoint_authenticates_roundtrip_scalers_schema_and_weight_bytes() {
        use super::super::{corpus::ExcludedTicker, data::DataContract};
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
            ..ModelConfig::default()
        };
        let ticker = DataContract {
            schema: "test-training-only-scaler".into(),
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
            scaler_fit_bars: 6900,
            means: [
                0.05898186155550259,
                f64::from_bits(1.0f64.to_bits() + 1),
                -0.0,
                138.73812039307464,
            ],
            stds: [
                0.00013897122900211475,
                2.3797001161863034,
                14.132441831700357,
                0.9280099882217092,
            ],
            common_context: 32,
            volume_features: false,
            auxiliary_schema: String::new(),
        };
        let data = CorpusContract {
            schema: "authenticated-pooled-training".into(),
            tickers: vec![ticker],
            boundary_timestamps: [700_000, 800_000, 900_000],
            context: 16,
            pred_len: 7,
            common_context: 32,
            purge: 100,
            volume_features: false,
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
            validation_mse: 0.004194157171231031,
            validation_is_full: false,
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
        let mut legacy = manifest.clone();
        legacy.model.decoder = super::super::model::DecoderKind::Legacy;
        legacy.numerics = "fp32-masters-normalization-bf16-SDPA-type1-epoch-LR".into();
        legacy.optimizer_recipe =
            "authenticated older optimizer; inference does not resume it".into();
        legacy.manifest_sha256 = legacy.digest().unwrap();
        write(&legacy);
        assert!(serde_json::to_value(&legacy).unwrap()["model"]
            .get("decoder")
            .is_none());
        assert_eq!(
            Manifest::read(&directory.0).unwrap().digest().unwrap(),
            legacy.digest().unwrap()
        );
        for (before, after) in manifest.data.tickers[0]
            .means
            .iter()
            .chain(manifest.data.tickers[0].stds.iter())
            .zip(
                restored.data.tickers[0]
                    .means
                    .iter()
                    .chain(restored.data.tickers[0].stds.iter()),
            )
        {
            assert_eq!(before.to_bits(), after.to_bits());
        }
        let mut corrupted = manifest.clone();
        corrupted.data.tickers[0].stds[0] =
            f64::from_bits(corrupted.data.tickers[0].stds[0].to_bits() + 1);
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
}
