mod reports;

use std::{path::Path, time::Instant};

use anyhow::{ensure, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use shared::{paths::RUNS_PATH, run_dir::RunDir};
use tch::{nn, Device, Kind, Tensor};

use crate::torch::cuda::cfg::{configure_cuda, disable_autograd_multithreading};
use crate::torch::lejepa::checkpoint::{load_authenticated, save_bundle};
use crate::torch::lejepa::dataset::MseJepaDataset;
use crate::torch::lejepa::model::{LATENT_DIM, MAX_CONTEXT_BARS, PREDICTOR_BLOCKS};
use crate::torch::lejepa::sigreg::{
    sample_temporal_views, sigreg_loss, sigreg_loss_with_directions, SIGREG_PROJECTIONS,
};
use crate::torch::lejepa::MseJepaModel;
use crate::torch::optim::muon::{Muon, MuonConfig, StepKind};

use super::config::{LEARNING_RATE, MAX_GRAD_NORM, MUON_LR, MUON_MOMENTUM_WARMUP_START, USE_MUON};
use super::optimizer_glue::{muon_momentum_for_step, named_trainable_variables};
use reports::{MseJepaReporter, StepMetrics};

pub const DEFAULT_BATCH_SIZE: usize = 8;
pub const DEFAULT_VALIDATION_WINDOWS: usize = DEFAULT_BATCH_SIZE;
pub const DEFAULT_CHECKPOINT_EVERY: usize = 2_048;

#[derive(Clone, Debug)]
pub struct MseJepaArgs {
    pub weights: Option<String>,
    pub run: Option<String>,
    pub epochs: usize,
    pub steps: Option<usize>,
    pub batch_size: usize,
    pub seed: u64,
    pub data_dir: String,
    pub resolution_secs: u32,
    pub min_bars: usize,
    pub validation_windows: usize,
    pub validate_every: usize,
    pub checkpoint_every: usize,
    pub lambda_sigreg: f64,
    pub split_bounds: Option<(i64, i64)>,
    pub derive_split_bounds: bool,
}

pub fn pretrain_mse_jepa(args: MseJepaArgs) -> Result<()> {
    validate_args(&args)?;
    super::pretrain::configure_threads();
    configure_cuda();
    let device = Device::cuda_if_available();
    ensure!(
        device.is_cuda(),
        "pretrain-mse-jepa requires CUDA and strict FA4; CPU is supported only by focused tests"
    );
    let _autograd_guard = disable_autograd_multithreading();
    tch::manual_seed(args.seed as i64);

    let split_bounds = if args.derive_split_bounds {
        None
    } else {
        Some(
            args.split_bounds
                .unwrap_or(crate::data::ingest::PINNED_SPLIT_BOUNDS),
        )
    };
    let dataset = MseJepaDataset::load(
        Path::new(&args.data_dir),
        args.resolution_secs,
        args.min_bars,
        split_bounds,
        args.seed,
    )?;
    let batches_per_epoch = dataset.batches_per_epoch(args.batch_size);
    ensure!(
        batches_per_epoch > 0,
        "--batch-size {} exceeds the MSE-JEPA training window count",
        args.batch_size
    );
    let available_batches = args.epochs.saturating_mul(batches_per_epoch);
    let planned_batches = args
        .steps
        .unwrap_or(available_batches)
        .min(available_batches);
    ensure!(
        planned_batches <= batches_per_epoch,
        "MSE-JEPA permits one near-disjoint mmap sampler pass; reduce --steps or use --epochs 1"
    );

    let run = RunDir::create_fresh(RUNS_PATH, args.run.as_deref())?;
    let mut var_store = nn::VarStore::new(device);
    let model = MseJepaModel::new(&var_store.root());
    if let Some(weights) = &args.weights {
        let authenticated = load_authenticated(&mut var_store, weights)?;
        println!(
            "Loaded authenticated {:?} MSE-JEPA weights from {}",
            authenticated.kind,
            authenticated.path.display()
        );
    }

    let named = named_trainable_variables(&var_store);
    let mut optimizer = build_optimizer(&named);
    let mut reporter = MseJepaReporter::new(&run.gens);
    let validation_refs = dataset.validation_refs(args.validation_windows);
    ensure!(
        !validation_refs.is_empty(),
        "MSE-JEPA validation panel is empty"
    );
    let validation_sigreg_directions = fixed_validation_sigreg_directions(device, args.seed);
    let mut best_validation = f64::INFINITY;
    let mut global_step = 0usize;
    let requested_steps = args.steps.unwrap_or(usize::MAX);

    for epoch in 0..args.epochs {
        for batch_index in 0..batches_per_epoch {
            if global_step >= requested_steps {
                break;
            }
            let step_started = Instant::now();
            optimizer.zero_grad();
            let bars = dataset.train_batch(epoch, batch_index, args.batch_size, device);
            let losses = objective(
                &model,
                &bars,
                args.lambda_sigreg,
                true,
                (global_step + 1) % REPRESENTATION_STATS_EVERY == 0,
                None,
            );
            losses.total.backward();
            let grad_norm = clip_gradients(&named, MAX_GRAD_NORM, device);
            let muon_momentum = muon_momentum_for_step(global_step as i64);
            optimizer.set_momentum(muon_momentum);
            let packed_metrics = pack_step_metrics(&losses, &grad_norm);
            optimizer.step(StepKind::Primary);
            let metric_values = read_packed_step_metrics(&packed_metrics);
            let completed_step_seconds = step_started.elapsed().as_secs_f64();
            global_step += 1;

            reporter.record_step(step_metrics(
                metric_values,
                muon_momentum,
                completed_step_seconds,
                args.batch_size,
            ));
            if args.validate_every > 0 && global_step % args.validate_every == 0 {
                let validation = validation_metrics(
                    &model,
                    &dataset,
                    &validation_refs,
                    args.lambda_sigreg,
                    &validation_sigreg_directions,
                    device,
                );
                reporter.record_validation(
                    validation.total_loss,
                    validation.prediction_mse,
                    validation.persistence_mse,
                    validation.skill_vs_persistence,
                );
                if validation.total_loss < best_validation {
                    best_validation = validation.total_loss;
                    save_bundle(
                        &var_store,
                        run.weights.join("mse_jepa_best.ot"),
                        args.lambda_sigreg,
                    )?;
                }
                reporter.write(epoch)?;
            }
            if args.checkpoint_every > 0 && global_step % args.checkpoint_every == 0 {
                save_bundle(
                    &var_store,
                    run.weights.join(format!("mse_jepa_step{global_step}.ot")),
                    args.lambda_sigreg,
                )?;
            }
        }

        if global_step > 0 {
            let validation = validation_metrics(
                &model,
                &dataset,
                &validation_refs,
                args.lambda_sigreg,
                &validation_sigreg_directions,
                device,
            );
            reporter.record_validation(
                validation.total_loss,
                validation.prediction_mse,
                validation.persistence_mse,
                validation.skill_vs_persistence,
            );
            if validation.total_loss < best_validation {
                best_validation = validation.total_loss;
                save_bundle(
                    &var_store,
                    run.weights.join("mse_jepa_best.ot"),
                    args.lambda_sigreg,
                )?;
            }
            reporter.write(epoch)?;
        }
        if global_step >= requested_steps {
            break;
        }
    }
    save_bundle(
        &var_store,
        run.weights.join("mse_jepa.ot"),
        args.lambda_sigreg,
    )?;
    println!(
        "Saved self-contained MSE-JEPA bundle under {}",
        run.weights.display()
    );
    Ok(())
}

fn validate_args(args: &MseJepaArgs) -> Result<()> {
    ensure!(args.epochs > 0, "--epochs must be positive");
    ensure!(
        args.steps != Some(0),
        "--steps must be positive when specified"
    );
    ensure!(args.batch_size > 0, "--batch-size must be positive");
    ensure!(
        args.resolution_secs > 0,
        "--resolution-secs must be positive"
    );
    ensure!(
        args.min_bars >= (MAX_CONTEXT_BARS + 2) as usize,
        "--min-bars must admit a full context plus target"
    );
    ensure!(
        args.validation_windows > 0,
        "--validation-windows must be positive"
    );
    ensure!(
        args.lambda_sigreg.is_finite() && args.lambda_sigreg >= 0.0,
        "--lambda-sigreg must be finite and non-negative"
    );
    ensure!(
        !(args.derive_split_bounds && args.split_bounds.is_some()),
        "--derive-split-bounds conflicts with --split-bounds"
    );
    Ok(())
}

fn build_optimizer(named: &[(String, Tensor)]) -> Muon {
    let mut allowlist = vec![
        "bar_proj".to_owned(),
        "bar_enrich".to_owned(),
        "lejepa_projector_fc".to_owned(),
        "lejepa_layer_".to_owned(),
        "lejepa_predictor_in_proj".to_owned(),
        "lejepa_predictor_out_proj".to_owned(),
    ];
    for index in 0..PREDICTOR_BLOCKS {
        allowlist.push(format!("lejepa_predictor_block_{index}.gate"));
        allowlist.push(format!("lejepa_predictor_block_{index}.value"));
        allowlist.push(format!("lejepa_predictor_block_{index}.out"));
    }
    Muon::new_named(
        named,
        MuonConfig {
            lr: MUON_LR,
            use_muon_for_2d: USE_MUON,
            momentum: MUON_MOMENTUM_WARMUP_START,
            adamw_lr: LEARNING_RATE,
            adamw_betas: (0.9, 0.95),
            adamw_eps: 1e-8,
            adamw_wd: 1e-3,
            adamw_no_weight_decay_name_substrings: vec!["pope_theta_bias".to_owned()],
            force_adamw_name_substrings: vec!["pope_theta_bias".to_owned()],
            muon_name_allowlist: allowlist,
            capture_step_graphs: true,
            ..MuonConfig::default()
        },
    )
}

struct Objective {
    total: Tensor,
    prediction_mse: Tensor,
    persistence_mse: Tensor,
    skill_vs_persistence: Tensor,
    sigreg: Tensor,
    representation_std: Tensor,
    prediction_std: Tensor,
    target_std: Tensor,
}

const REPRESENTATION_STATS_EVERY: usize = 32;
const STEP_METRIC_COUNT: usize = 9;
const VALIDATION_SIGREG_SEED_DOMAIN: u64 = 0x4C45_574D_5641_4C31;

fn objective(
    model: &MseJepaModel,
    bars: &Tensor,
    lambda_sigreg: f64,
    train: bool,
    sample_representation_stats: bool,
    sigreg_directions: Option<&Tensor>,
) -> Objective {
    let forward = model.forward(bars, train);
    let prediction_mse = attached_prediction_mse(&forward.predictions, &forward.targets);
    let persistence_mse = attached_prediction_mse(
        &forward.all_tokens.narrow(2, 0, MAX_CONTEXT_BARS),
        &forward.targets,
    );
    let skill_vs_persistence = latent_skill_vs_persistence(&prediction_mse, &persistence_mse);
    let views = sample_temporal_views(&forward.all_tokens, train);
    let sigreg = match sigreg_directions {
        Some(directions) => sigreg_loss_with_directions(&views, directions),
        None => sigreg_loss(&views),
    };
    let total = &prediction_mse + &sigreg * lambda_sigreg;
    let (representation_std, prediction_std, target_std) = if sample_representation_stats {
        (
            feature_std(&forward.all_tokens.detach()),
            forward.predictions.detach().std(false),
            forward.targets.detach().std(false),
        )
    } else {
        let nan = Tensor::full([], f64::NAN, (Kind::Float, bars.device()));
        (nan.shallow_clone(), nan.shallow_clone(), nan)
    };
    Objective {
        total,
        prediction_mse,
        persistence_mse,
        skill_vs_persistence,
        sigreg,
        representation_std,
        prediction_std,
        target_std,
    }
}

fn attached_prediction_mse(predictions: &Tensor, targets: &Tensor) -> Tensor {
    (predictions - targets).square().mean(Kind::Float)
}

fn latent_skill_vs_persistence(prediction_mse: &Tensor, persistence_mse: &Tensor) -> Tensor {
    Tensor::full_like(persistence_mse, f64::NAN).where_self(
        &persistence_mse.eq(0.0),
        &(Tensor::ones_like(persistence_mse) - prediction_mse / persistence_mse),
    )
}

fn feature_std(tokens: &Tensor) -> Tensor {
    let flat = tokens.reshape([-1, LATENT_DIM]).to_kind(Kind::Float);
    let mean = flat.mean_dim([0i64].as_slice(), true, Kind::Float);
    (&flat - mean)
        .square()
        .mean_dim([0i64].as_slice(), false, Kind::Float)
        .sqrt()
        .mean(Kind::Float)
}

fn clip_gradients(named: &[(String, Tensor)], maximum: f64, device: Device) -> Tensor {
    tch::no_grad(|| {
        let gradients = named
            .iter()
            .map(|(_, parameter)| parameter.grad())
            .filter(|gradient| gradient.defined())
            .collect::<Vec<_>>();
        let norms = gradients
            .iter()
            .map(|gradient| gradient.to_kind(Kind::Float).norm())
            .collect::<Vec<_>>();
        let norm = if norms.is_empty() {
            Tensor::zeros([], (Kind::Float, device))
        } else {
            Tensor::stack(&norms, 0).norm()
        };
        let coefficient =
            (Tensor::from(maximum as f32).to_device(device) / (&norm + 1e-6)).clamp_max(1.0);
        for mut gradient in gradients {
            let _ = gradient.g_mul_(&coefficient.to_kind(gradient.kind()));
        }
        norm
    })
}

fn pack_step_metrics(losses: &Objective, grad_norm: &Tensor) -> Tensor {
    let flat_f32 = |tensor: &Tensor| tensor.detach().to_kind(Kind::Float).reshape([-1]);
    Tensor::cat(
        &[
            flat_f32(&losses.total),
            flat_f32(&losses.prediction_mse),
            flat_f32(&losses.persistence_mse),
            flat_f32(&losses.skill_vs_persistence),
            flat_f32(&losses.sigreg),
            flat_f32(&losses.representation_std),
            flat_f32(&losses.prediction_std),
            flat_f32(&losses.target_std),
            flat_f32(grad_norm),
        ],
        0,
    )
}

fn read_packed_metrics<const N: usize>(packed: &Tensor) -> [f64; N] {
    let values = Vec::<f32>::try_from(packed.reshape([-1]))
        .expect("packed MSE-JEPA metrics are convertible");
    let values: [f32; N] = values.try_into().unwrap_or_else(|values: Vec<f32>| {
        panic!(
            "packed MSE-JEPA metrics carry {} entries, expected {N}",
            values.len()
        )
    });
    values.map(f64::from)
}

fn read_packed_step_metrics(packed: &Tensor) -> [f64; STEP_METRIC_COUNT] {
    read_packed_metrics(packed)
}

fn step_metrics(
    values: [f64; STEP_METRIC_COUNT],
    muon_momentum: f64,
    completed_step_seconds: f64,
    batch_size: usize,
) -> StepMetrics {
    StepMetrics {
        total_loss: values[0],
        prediction_mse: values[1],
        persistence_mse: values[2],
        skill_vs_persistence: values[3],
        sigreg: values[4],
        representation_std: values[5],
        prediction_std: values[6],
        target_std: values[7],
        grad_norm: values[8],
        muon_learning_rate: MUON_LR,
        adamw_learning_rate: LEARNING_RATE,
        muon_momentum,
        completed_step_seconds,
        windows_per_second: batch_size as f64 / completed_step_seconds,
        predicted_latent_positions_per_second: batch_size as f64 * MAX_CONTEXT_BARS as f64
            / completed_step_seconds,
    }
}

#[derive(Clone, Copy)]
struct ValidationMetrics {
    total_loss: f64,
    prediction_mse: f64,
    persistence_mse: f64,
    skill_vs_persistence: f64,
}

fn fixed_validation_sigreg_directions(device: Device, seed: u64) -> Tensor {
    let mut rng = ChaCha12Rng::seed_from_u64(seed ^ VALIDATION_SIGREG_SEED_DOMAIN);
    let count = (LATENT_DIM * SIGREG_PROJECTIONS) as usize;
    let mut values = Vec::with_capacity(count);
    while values.len() < count {
        let uniform = (1.0 - rng.random::<f64>()).max(f64::MIN_POSITIVE);
        let radius = (-2.0 * uniform.ln()).sqrt();
        let angle = std::f64::consts::TAU * rng.random::<f64>();
        values.push((radius * angle.cos()) as f32);
        if values.len() < count {
            values.push((radius * angle.sin()) as f32);
        }
    }
    let directions = Tensor::from_slice(&values)
        .view([LATENT_DIM, SIGREG_PROJECTIONS])
        .to_device(device);
    &directions / directions.norm_scalaropt_dim(2, [0i64].as_slice(), true)
}

fn validation_metrics(
    model: &MseJepaModel,
    dataset: &MseJepaDataset,
    refs: &[crate::torch::dataset::WindowRef],
    lambda_sigreg: f64,
    sigreg_directions: &Tensor,
    device: Device,
) -> ValidationMetrics {
    tch::no_grad(|| {
        let bars = dataset.features(refs, device);
        let losses = objective(
            model,
            &bars,
            lambda_sigreg,
            false,
            false,
            Some(sigreg_directions),
        );
        let packed = Tensor::stack(
            &[
                losses.total.detach(),
                losses.prediction_mse.detach(),
                losses.persistence_mse.detach(),
                losses.skill_vs_persistence.detach(),
            ],
            0,
        )
        .to_kind(Kind::Float);
        let values = read_packed_metrics::<4>(&packed);
        ValidationMetrics {
            total_loss: values[0],
            prediction_mse: values[1],
            persistence_mse: values[2],
            skill_vs_persistence: values[3],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{
        attached_prediction_mse, fixed_validation_sigreg_directions, latent_skill_vs_persistence,
        validate_args, MseJepaArgs,
    };
    use crate::torch::lejepa::sigreg::DEFAULT_SIGREG_LAMBDA;
    use tch::{Kind, Tensor};

    #[test]
    fn production_prediction_loss_backpropagates_through_both_attached_branches() {
        let prediction = Tensor::from_slice(&[0.2f32, -0.1, 0.8]).set_requires_grad(true);
        let target = Tensor::from_slice(&[0.4f32, 0.3, -0.2]).set_requires_grad(true);
        attached_prediction_mse(&prediction, &target).backward();
        assert!(prediction.grad().abs().sum(Kind::Float).double_value(&[]) > 0.0);
        assert!(target.grad().abs().sum(Kind::Float).double_value(&[]) > 0.0);
    }
    #[test]
    fn validation_sigreg_directions_are_seeded_and_unit_norm() {
        let first = fixed_validation_sigreg_directions(tch::Device::Cpu, 17);
        let second = fixed_validation_sigreg_directions(tch::Device::Cpu, 17);
        assert!(first.equal(&second));
        let max_norm_error = (first.norm_scalaropt_dim(2, [0i64].as_slice(), false) - 1.0)
            .abs()
            .max()
            .double_value(&[]);
        assert!(max_norm_error < 1e-5);
    }

    #[test]
    fn latent_skill_is_undefined_when_persistence_is_perfect() {
        let prediction_mse = Tensor::from(0.0f32);
        let persistence_mse = Tensor::from(0.0f32);
        let undefined =
            latent_skill_vs_persistence(&prediction_mse, &persistence_mse).double_value(&[]);
        assert!(undefined.is_nan());

        let prediction_mse = Tensor::from(0.5f32);
        let persistence_mse = Tensor::from(1.0f32);
        let skill =
            latent_skill_vs_persistence(&prediction_mse, &persistence_mse).double_value(&[]);
        assert!((skill - 0.5).abs() < 1e-7);
    }

    #[test]
    fn zero_steps_cannot_publish_an_untrained_bundle() {
        let args = MseJepaArgs {
            weights: None,
            run: None,
            epochs: 1,
            steps: Some(0),
            batch_size: 8,
            seed: 0x5EED,
            data_dir: "unused".to_owned(),
            resolution_secs: 300,
            min_bars: 60_000,
            validation_windows: 8,
            validate_every: 1_000,
            checkpoint_every: 2_048,
            lambda_sigreg: DEFAULT_SIGREG_LAMBDA,
            split_bounds: None,
            derive_split_bounds: false,
        };
        let error = validate_args(&args).unwrap_err().to_string();
        assert!(error.contains("--steps must be positive"));
    }
}
