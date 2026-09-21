use std::path::PathBuf;

use anyhow::{ensure, Context, Result};
use clap::Args;
use pyo3::{prelude::*, sync::PyOnceLock};
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;
use shared::{paths::RUNS_PATH, run_dir::RunDir};
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use super::{
    checkpoint::{self, DataContract, Manifest},
    data::{Dataset, Split},
    evaluate,
    model::{ForecastModel, ModelKind},
    reports,
};

#[derive(Clone, Debug, Args)]
pub struct TrainArgs {
    /// Exactly one ticker; alternatively configure TIMEXER_TICKER.
    #[arg(long)]
    pub ticker: Option<String>,
    #[arg(long, default_value_os_t = crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub run: Option<String>,
    #[arg(long, value_enum, default_value_t = ModelKind::SingleTickerTimeXer)]
    pub model: ModelKind,
    #[arg(long, default_value_t = 20)]
    pub epochs: usize,
    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 0.0003)]
    pub learning_rate: f64,
    #[arg(long, default_value_t = super::FROZEN_SEEDS[0])]
    pub seed: u64,
}

impl Default for TrainArgs {
    fn default() -> Self {
        Self {
            ticker: None,
            data_dir: crate::data::ingest::bars_dir(),
            run: None,
            model: ModelKind::SingleTickerTimeXer,
            epochs: 20,
            batch_size: 32,
            learning_rate: 0.0003,
            seed: super::FROZEN_SEEDS[0],
        }
    }
}

#[derive(Clone, Debug, Args)]
pub struct EvaluateArgs {
    /// Authenticated checkpoint directory containing weights.ot and manifest.json.
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long, default_value_os_t = crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    #[arg(long, value_enum, default_value_t = Split::Validation)]
    pub split: Split,
    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,
    #[arg(long)]
    pub output: PathBuf,
    /// Also score the train-fitted unconditional and HAR Student-t baselines.
    #[arg(long)]
    pub statistical_baselines: bool,
}

#[derive(Clone, Debug, Args)]
pub struct FreezeArgs {
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Clone, Debug, Args)]
pub struct CompareArgs {
    /// Three timexer_evidence.report.bin files, one per frozen initialization seed.
    #[arg(long, num_args = 3, required = true)]
    pub candidate: Vec<PathBuf>,
    #[arg(long, num_args = 3, required = true)]
    pub baseline: Vec<PathBuf>,
    /// Matched PatchTST evidence is mandatory before accepting the global bridge.
    #[arg(long, num_args = 3, required = true)]
    pub patch_tst: Vec<PathBuf>,
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Clone, Debug, Args)]
pub struct ForecastArgs {
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long, default_value_os_t = crate::data::ingest::bars_dir())]
    pub data_dir: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
}

pub fn configured_ticker(explicit: Option<&str>) -> Result<String> {
    let value = explicit
        .map(str::to_owned)
        .or_else(|| std::env::var("TIMEXER_TICKER").ok())
        .context("configure exactly one ticker with --ticker or TIMEXER_TICKER")?;
    let ticker = value.trim().to_ascii_uppercase();
    ensure!(
        !ticker.is_empty()
            && ticker.len() <= 24
            && ticker
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'-')),
        "invalid single ticker {ticker:?}"
    );
    Ok(ticker)
}

pub(crate) fn cuda_device() -> Result<Device> {
    ensure!(
        tch::Cuda::is_available(),
        "SingleTickerTimeXer requires CUDA; CPU execution is only for correctness tests"
    );
    crate::torch::train::pretrain::configure_threads();
    crate::torch::cuda::cfg::configure_cuda();
    crate::torch::cuda::cfg::enable_tf32_matmul()?;
    Ok(Device::Cuda(0))
}

const OPTIMIZER_CONTRACT: &str =
    "AdamW-fp32-master(lr=config,beta1=.9,beta2=.999,epsilon=1e-8,wd=.01-matrices-only)-v1";

pub(super) fn optimizer(store: &nn::VarStore, learning_rate: f64) -> Result<nn::Optimizer> {
    {
        let mut variables = store.variables_.lock().unwrap();
        for parameter in &mut variables.trainable_variables {
            parameter.group = usize::from(parameter.tensor.dim() < 2);
        }
    }
    let mut optimizer = nn::AdamW {
        wd: 0.01,
        ..Default::default()
    }
    .build(store, learning_rate)?;
    optimizer.set_weight_decay_group(1, 0.0);
    Ok(optimizer)
}

pub fn hard_loss(logits: &Tensor, bins: &Tensor, weights: &Tensor) -> Tensor {
    let row_nll = -logits
        .to_kind(Kind::Float)
        .log_softmax(-1, Kind::Float)
        .gather(-1, &bins.unsqueeze(-1), false)
        .squeeze_dim(-1)
        .mean_dim(&[-1i64][..], false, Kind::Float);
    (row_nll * weights).mean(Kind::Float)
}

pub(crate) fn check_objective(loss: &Tensor) -> Result<()> {
    static ASSERT_ASYNC: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    Python::attach(|py| -> Result<()> {
        let assert = ASSERT_ASYNC.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
            Ok(py.import("torch")?.getattr("_assert_async")?.unbind())
        })?;
        let finite = crate::torch::fa4::tensor_object(py, &loss.isfinite())?;
        assert
            .bind(py)
            .call1((finite, "nonfinite TimeXer training objective"))?;
        Ok(())
    })
}

pub fn train(args: TrainArgs) -> Result<()> {
    ensure!(
        args.epochs > 0 && args.batch_size > 0,
        "epochs and batch size must be positive"
    );
    ensure!(
        args.learning_rate.is_finite() && args.learning_rate > 0.,
        "learning rate must be positive and finite"
    );
    let ticker = configured_ticker(args.ticker.as_deref())?;
    if let Some(name) = &args.run {
        RunDir::ensure_creatable(RUNS_PATH, name)?;
    }
    let device = cuda_device()?;
    let _backward = crate::torch::cuda::cfg::disable_autograd_multithreading();
    let mut dataset = Dataset::load(&args.data_dir, &ticker)?;
    dataset.prepare(device);
    tch::manual_seed(args.seed as i64);
    tch::Cuda::manual_seed_all(args.seed);
    let mut store = nn::VarStore::new(device);
    let model = ForecastModel::new(&store.root(), args.model);
    let mut optimizer = optimizer(&store, args.learning_rate)?;
    let run = RunDir::create_fresh(RUNS_PATH, args.run.as_deref())?;
    let contract = DataContract::from_dataset(&dataset);
    std::fs::write(
        run.root.join("timexer-data-contract.json"),
        serde_json::to_vec_pretty(&contract)?,
    )?;
    let mut origins = dataset.origins(Split::Train).to_vec();
    ensure!(!origins.is_empty(), "no training origins");
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let mut best_loss = f64::INFINITY;
    let mut step = 0;
    for epoch in 1..=args.epochs {
        origins.shuffle(&mut rng);
        let mut total_loss = Tensor::zeros([], (Kind::Double, device));
        for batch_origins in origins.chunks(args.batch_size) {
            let batch = dataset.batch(batch_origins, device)?;
            optimizer.zero_grad();
            let output = model.forward(
                &batch.endogenous,
                &batch.exogenous,
                &batch.validity,
                &batch.future_clock,
                true,
            );
            let loss = if args.model == ModelKind::RawTimeXer {
                ((output.to_kind(Kind::Float) - &batch.targets)
                    .square()
                    .mean_dim(&[-1i64][..], false, Kind::Float)
                    * &batch.weights)
                    .mean(Kind::Float)
            } else {
                hard_loss(&output, &batch.bins, &batch.weights)
            };
            check_objective(&loss)?;
            loss.backward();
            optimizer.step();
            step += 1;
            total_loss += loss.detach().to_kind(Kind::Double) * batch_origins.len() as f64;
        }
        let mut evaluation = evaluate::evaluate_model(
            &dataset,
            &model,
            device,
            Split::Validation,
            args.batch_size,
            args.seed,
        )?;
        let score = evaluation.selection_loss();
        ensure!(score.is_finite(), "nonfinite validation objective");
        let manifest = checkpoint::save(
            &run.weights.join(format!("epoch-{epoch:04}")),
            &store,
            Manifest {
                format: checkpoint::FORMAT.to_owned(),
                architecture: super::model::architecture_contract(args.model),
                model_kind: args.model,
                data: contract.clone(),
                objective: checkpoint::objective(args.model).to_owned(),
                seed: args.seed,
                epoch,
                step,
                planned_epochs: args.epochs,
                selection_epochs: 0,
                batch_size: args.batch_size,
                learning_rate: args.learning_rate,
                optimizer: OPTIMIZER_CONTRACT.to_owned(),
                weights_sha256: String::new(),
                frozen: false,
                manifest_sha256: String::new(),
            },
        )?;
        evaluation.authenticate_checkpoint(&manifest)?;
        reports::write_training(
            &run.gens,
            epoch,
            total_loss.double_value(&[]) / origins.len() as f64,
            args.learning_rate,
        )?;
        reports::write_evaluation(&run.gens, epoch, &evaluation)?;
        if args.model != ModelKind::RawTimeXer {
            super::candles::write_windows(
                &dataset,
                &model,
                device,
                epoch,
                step,
                &run.gens.join(epoch.to_string()),
            )?;
        }
        if score < best_loss {
            best_loss = score;
            checkpoint::update_best(&run.weights, &format!("epoch-{epoch:04}"))?;
        }
        println!(
            "TimeXer completed epoch {epoch}/{}; reports: {}",
            args.epochs,
            run.gens.join(epoch.to_string()).display()
        );
    }
    optimizer.zero_grad();
    drop(optimizer);
    let best = run.weights.join("best");
    let mut selected = checkpoint::read(&best)?;
    checkpoint::load_weights(&best, &mut store)?;
    selected.selection_epochs = args.epochs;
    checkpoint::save(&run.weights.join("selected"), &store, selected)?;
    checkpoint::update_best(&run.weights, "selected")?;
    println!(
        "Checkpoint selected on validation: {}. Terminal test remains locked until freeze-timexer.",
        run.weights.join("best").display()
    );
    Ok(())
}

pub fn evaluate(args: EvaluateArgs) -> Result<()> {
    ensure!(args.batch_size > 0, "batch size must be positive");
    let manifest = checkpoint::read(&args.checkpoint)?;
    ensure!(
        args.split != Split::Test || manifest.frozen,
        "terminal test is locked: freeze architecture and checkpoint with freeze-timexer first"
    );
    let mut dataset = Dataset::load(&args.data_dir, &manifest.data.ticker)?;
    manifest.authenticate_dataset(&dataset)?;
    let device = cuda_device()?;
    dataset.prepare(device);
    let mut store = nn::VarStore::new(device);
    let model = ForecastModel::new(&store.root(), manifest.model_kind);
    checkpoint::load_weights(&args.checkpoint, &mut store)?;
    store.freeze();
    let mut evaluation = evaluate::evaluate_model(
        &dataset,
        &model,
        device,
        args.split,
        args.batch_size,
        manifest.seed,
    )?;
    evaluation.authenticate_checkpoint(&manifest)?;
    reports::write_evaluation(&args.output, 0, &evaluation)?;
    if args.statistical_baselines {
        reports::write_evaluation(
            &args.output.join("unconditional"),
            0,
            &evaluate::evaluate_unconditional(&dataset, args.split)?,
        )?;
        reports::write_evaluation(
            &args.output.join("har-student-t"),
            0,
            &evaluate::HarStudentT::fit(&dataset)?.evaluate(&dataset, args.split)?,
        )?;
    }
    Ok(())
}

pub fn compare(args: CompareArgs) -> Result<()> {
    let candidates = args
        .candidate
        .iter()
        .map(|p| reports::read_evaluation(p))
        .collect::<Result<Vec<_>>>()?;
    let baselines = args
        .baseline
        .iter()
        .map(|p| reports::read_evaluation(p))
        .collect::<Result<Vec<_>>>()?;
    let patch_tst = args
        .patch_tst
        .iter()
        .map(|p| reports::read_evaluation(p))
        .collect::<Result<Vec<_>>>()?;
    let decision = super::gates::assess_with_patchtst(&candidates, &baselines, &patch_tst)?;
    reports::write_gate_decision(&args.output, 0, &decision)?;
    println!(
        "Replacement-gate decision written to {}",
        args.output.display()
    );
    Ok(())
}

pub fn forecast(args: ForecastArgs) -> Result<()> {
    let manifest = checkpoint::read(&args.checkpoint)?;
    ensure!(manifest.frozen, "forecast requires a frozen checkpoint");
    ensure!(
        manifest.model_kind != ModelKind::RawTimeXer,
        "raw TimeXer is an RMSE/MAE comparison only"
    );
    let dataset = Dataset::load_for_inference(
        &args.data_dir,
        &manifest.data.ticker,
        manifest.data.source_bars,
        &manifest.data.corpus_sha256,
        &manifest.data.supports,
    )?;
    let device = cuda_device()?;
    let mut store = nn::VarStore::new(device);
    let model = ForecastModel::new(&store.root(), manifest.model_kind);
    checkpoint::load_weights(&args.checkpoint, &mut store)?;
    store.freeze();
    let origin = dataset.latest_origin();
    let batch = dataset.inference_batch(&[origin], device)?;
    let probabilities = tch::no_grad(|| {
        model
            .forward(
                &batch.endogenous,
                &batch.exogenous,
                &batch.validity,
                &batch.future_clock,
                false,
            )
            .to_kind(Kind::Float)
            .softmax(-1, Kind::Float)
            .to_device(Device::Cpu)
    });
    let probabilities: Vec<Vec<f64>> = (0..6)
        .map(|h| Vec::<f64>::try_from(probabilities.get(0).get(h).to_kind(Kind::Double)))
        .collect::<Result<_, _>>()?;
    reports::write_forecast(
        &args.output,
        0,
        &dataset.supports,
        &probabilities,
        dataset.sigma(origin),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn asynchronous_objective_check_rejects_nonfinite_values() {
        assert!(check_objective(&Tensor::from(2.0f32)).is_ok());
        assert!(check_objective(&Tensor::from(f32::NAN)).is_err());
        assert!(check_objective(&Tensor::from(f32::INFINITY)).is_err());
    }

    #[test]
    fn adamw_preserves_normalization_scales_and_keeps_float_masters() {
        let store = nn::VarStore::new(Device::Cpu);
        let matrix = store.root().var("matrix", &[2, 2], nn::Init::Const(1.0));
        let scale = store.root().var("scale", &[2], nn::Init::Const(1.0));
        let mut optimizer = optimizer(&store, 0.1).unwrap();
        ((matrix.sum(Kind::Float) + scale.sum(Kind::Float)) * 0.0).backward();
        optimizer.step();
        assert!((matrix.double_value(&[0, 0]) - 0.999).abs() < 1e-6);
        assert_eq!(scale.double_value(&[0]), 1.0);
        assert_eq!(matrix.kind(), Kind::Float);
    }

    #[test]
    fn hard_likelihood_uses_global_uniqueness_without_batch_renormalization() {
        let logits = Tensor::zeros([2, 6, 128], (Kind::Float, Device::Cpu));
        let bins = Tensor::zeros([2, 6], (Kind::Int64, Device::Cpu));
        let loss = hard_loss(&logits, &bins, &Tensor::from_slice(&[0.25f32, 0.75]));
        assert!((loss.double_value(&[]) - 0.5 * 128f64.ln()).abs() < 1e-6);
    }
    #[test]
    fn explicit_ticker_cannot_name_multiple_symbols_or_paths() {
        for invalid in ["AAPL,MSFT", "AAPL MSFT", "../AAPL", "", "*"] {
            assert!(configured_ticker(Some(invalid)).is_err());
        }
        assert_eq!(configured_ticker(Some("aapl")).unwrap(), "AAPL");
    }
}
