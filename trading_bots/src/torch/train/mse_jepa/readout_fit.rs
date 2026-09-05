use std::path::Path;

use anyhow::{ensure, Context, Result};
use rand::{seq::index::sample, SeedableRng};
use rand_chacha::ChaCha12Rng;
use shared::{paths::RUNS_PATH, run_dir::RunDir};
use tch::{autocast, nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::torch::bar_dist::{bar_nll_from_logits, BarScoring, BarSupports, BAR_DOF};
use crate::torch::cuda::cfg::{
    configure_cuda, disable_autograd_multithreading, enable_tf32_matmul, pin_bfloat16_autocast,
};
use crate::torch::lejepa::checkpoint::{
    load_authenticated, save_bundle, CheckpointKind, CheckpointProvenance, HeadFitProvenance,
    WeightReadout, CANONICAL_READOUT_FIT_BATCH_SIZE, CANONICAL_READOUT_FIT_SEED,
    CANONICAL_READOUT_FIT_STEPS, CANONICAL_READOUT_OPTIMIZER_RECIPE_ID,
    CANONICAL_READOUT_TOKEN_ROWS, CANONICAL_READOUT_VALIDATION_WINDOWS,
};
use crate::torch::lejepa::dataset::MseJepaDataset;
use crate::torch::lejepa::model::{LATENT_DIM, MAX_CONTEXT_BARS};
use crate::torch::lejepa::{MseJepaModel, MseJepaTokenProbe};

use super::reports::{
    inherit_source_reports, preflight_source_reports, write_posthoc_readout_report,
    PosthocReadoutReport,
};

pub const DEFAULT_READOUT_FIT_STEPS: usize = CANONICAL_READOUT_FIT_STEPS as usize;
pub const DEFAULT_READOUT_FIT_BATCH_SIZE: usize = CANONICAL_READOUT_FIT_BATCH_SIZE as usize;
pub const DEFAULT_READOUT_TOKEN_ROWS: usize = CANONICAL_READOUT_TOKEN_ROWS as usize;
pub const DEFAULT_READOUT_VALIDATION_WINDOWS: usize = CANONICAL_READOUT_VALIDATION_WINDOWS as usize;
pub const DEFAULT_READOUT_FIT_SEED: u64 = CANONICAL_READOUT_FIT_SEED;
pub const READOUT_OPTIMIZER_RECIPE_ID: &str = CANONICAL_READOUT_OPTIMIZER_RECIPE_ID;
const READOUT_LR: f64 = 3e-4;
const READOUT_ADAMW: nn::AdamW = nn::AdamW {
    beta1: 0.9,
    beta2: 0.999,
    wd: 0.01,
    eps: 1e-8,
    amsgrad: false,
};
const TOKEN_ROW_SEED_DOMAIN: u64 = 0x544f_4b45_4e52_4f57;
const TOKEN_PROBE_PREFIX: &str = "posthoc_token_probe.";
const EMISSION_PREFIX: &str = "lejepa_emission.";

#[derive(Clone, Debug)]
pub struct FitMseJepaReadoutsArgs {
    pub weights: String,
    pub run: String,
    pub steps: usize,
    pub batch_size: usize,
    pub token_rows_per_step: usize,
    pub validation_windows: usize,
    pub seed: u64,
    pub data_dir: String,
    pub resolution_secs: u32,
    pub min_bars: usize,
    pub split_bounds: Option<(i64, i64)>,
    pub derive_split_bounds: bool,
}

/// Fit fresh belief-emission and same-time token heads against one frozen, fully trained core
/// backbone. Training split batches are the only optimizer input; the pinned validation panel is
/// materialized exactly once, after the declared update schedule is complete.
pub fn fit_mse_jepa_readouts(args: FitMseJepaReadoutsArgs) -> Result<()> {
    validate_args(&args)?;
    RunDir::ensure_creatable(RUNS_PATH, &args.run)
        .context("validating the posthoc readout output run")?;
    super::super::pretrain::configure_threads();
    configure_cuda();
    enable_tf32_matmul()?;
    let device = Device::cuda_if_available();
    ensure!(
        device.is_cuda(),
        "fit-mse-jepa-readouts requires CUDA and strict FA4; there is no CPU production path"
    );
    let _autograd_guard = disable_autograd_multithreading();

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
    dataset.advise_random_access()?;
    let supports = super::load_authenticated_supports(&dataset)?;

    let mut var_store = nn::VarStore::new(device);
    let mut model = MseJepaModel::new(&var_store.root());
    let authenticated = load_authenticated(&mut var_store, &args.weights)?;
    ensure!(
        authenticated.kind == CheckpointKind::Core,
        "posthoc readout fitting accepts only a core checkpoint, not an already-fitted endpoint"
    );
    ensure!(
        authenticated.provenance.completed_steps == authenticated.provenance.planned_steps,
        "posthoc readout fitting requires completed_steps == planned_steps"
    );
    super::authenticate_core_origin(
        &authenticated.provenance,
        &dataset,
        &supports,
        args.resolution_secs,
        args.min_bars,
        !args.derive_split_bounds,
    )
    .context("authenticating the core training data/support origin")?;
    let source_run = RunDir::from_weights_path_in(&authenticated.path, RUNS_PATH)
        .context("resolving the authenticated MSE-JEPA source run")?;
    let source_reports = preflight_source_reports(
        &source_run.gens,
        usize::try_from(authenticated.provenance.completed_steps)
            .context("source completed-step count exceeds this platform")?,
    )
    .context("preflighting core reports before readout fitting")?;

    let mut token_probe = MseJepaTokenProbe::new(&(var_store.root() / "posthoc_token_probe"));
    var_store.freeze();
    model.reset_emission(args.seed);
    token_probe.reset(args.seed);
    set_head_trainability(&var_store)?;
    assert_only_heads_trainable(&var_store)?;
    let backbone_before = backbone_fingerprint(&var_store);

    let mut optimizer = READOUT_ADAMW
        .build(&var_store, READOUT_LR)
        .context("building posthoc readout AdamW")?;
    let mut scratch = dataset.train_scratch();
    let mut emission_curve = Vec::with_capacity(args.steps);
    let mut emission_hard_curve = Vec::with_capacity(args.steps);
    let mut token_curve = Vec::with_capacity(args.steps);
    let mut token_hard_curve = Vec::with_capacity(args.steps);
    for step in 0..args.steps {
        optimizer.zero_grad();
        let (bars, next_dof) = dataset
            .readout_train_host_batch(step, args.batch_size, &mut scratch)
            .to_device(device);
        let forward = tch::no_grad(|| model.forward(&bars, false));
        let beliefs = forward.beliefs.detach().reshape([-1, LATENT_DIM]);
        let tokens = forward
            .all_tokens
            .detach()
            .narrow(2, 1, MAX_CONTEXT_BARS)
            .reshape([-1, LATENT_DIM]);
        let dof = next_dof.reshape([-1, BAR_DOF as i64]);
        let picked = fixed_row_indices(
            args.seed,
            step,
            beliefs.size()[0],
            args.token_rows_per_step,
            device,
        );
        let beliefs = beliefs.index_select(0, &picked);
        let tokens = tokens.index_select(0, &picked);
        let dof = dof.index_select(0, &picked);
        let bins = supports.bin_ids(&dof);
        pin_bfloat16_autocast();
        let (emission_logits, token_logits) = autocast(true, || {
            (
                model.emission_logits(&beliefs, &bins),
                token_probe.logits(&tokens, &bins),
            )
        });
        let smoothed_targets = supports.targets(&dof, BarScoring::Smoothed);
        let hard_targets = supports.targets(&dof, BarScoring::Hard);
        let emission_loss = bar_nll_from_logits(&emission_logits, &smoothed_targets).0;
        let token_loss = bar_nll_from_logits(&token_logits, &smoothed_targets).0;
        let emission_hard = bar_nll_from_logits(&emission_logits.detach(), &hard_targets).0;
        let token_hard = bar_nll_from_logits(&token_logits.detach(), &hard_targets).0;
        let packed = Tensor::stack(
            &[
                emission_loss.detach(),
                emission_hard,
                token_loss.detach(),
                token_hard,
            ],
            0,
        )
        .to_kind(Kind::Float)
        .to_device(Device::Cpu);
        let values = Vec::<f32>::try_from(packed).expect("four fitted readout losses");
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "posthoc readout loss became non-finite at optimizer step {step}"
        );
        emission_curve.push(values[0] as f64);
        emission_hard_curve.push(values[1] as f64);
        token_curve.push(values[2] as f64);
        token_hard_curve.push(values[3] as f64);
        (emission_loss + token_loss).backward();
        optimizer.step();
    }
    assert_only_heads_trainable(&var_store)?;
    ensure!(
        backbone_before == backbone_fingerprint(&var_store),
        "frozen MSE-JEPA backbone changed during posthoc readout fitting"
    );

    let validation_refs = dataset.validation_refs(args.validation_windows);
    ensure!(
        validation_refs.len() == args.validation_windows,
        "pinned validation panel is smaller than --validation-windows"
    );
    let validation_dof = dataset.validation_next_dof(&validation_refs, device);
    let validation_bars = dataset.features(&validation_refs, device);
    let validation = tch::no_grad(|| {
        evaluate_readouts(
            &model,
            &token_probe,
            &supports,
            &validation_bars,
            &validation_dof,
        )
    });
    ensure!(
        [validation.0, validation.1, validation.2, validation.3]
            .iter()
            .all(|value| value.is_finite()),
        "posthoc readout validation produced a non-finite NLL"
    );
    ensure!(
        backbone_before == backbone_fingerprint(&var_store),
        "frozen MSE-JEPA backbone changed during final validation"
    );

    let requested_root = Path::new(RUNS_PATH).join(&args.run);
    let previous_latest = RunDir::latest(RUNS_PATH)
        .ok()
        .filter(|previous| previous.root != requested_root);
    let run = RunDir::create_fresh(RUNS_PATH, Some(&args.run))?;
    let publish = (|| -> Result<()> {
        inherit_source_reports(&run.gens, &source_reports)
            .context("inheriting core reports into fitted readout run")?;
        let source_kind = match &authenticated.provenance.weight_readout {
            WeightReadout::Raw => "raw",
            WeightReadout::TailEma { .. } => "tail_ema",
        };
        let gradient_mode = authenticated.provenance.emission_gradient_mode.to_string();
        write_posthoc_readout_report(
            &run.gens,
            &PosthocReadoutReport {
                emission_train_smoothed: &emission_curve,
                token_train_smoothed: &token_curve,
                emission_train_hard: &emission_hard_curve,
                emission_validation_smoothed: validation.0,
                token_train_hard: &token_hard_curve,
                emission_validation_hard: validation.1,
                token_validation_smoothed: validation.2,
                token_validation_hard: validation.3,
                seed: args.seed,
                steps: args.steps,
                batch_size: args.batch_size,
                token_rows_per_step: args.token_rows_per_step,
                validation_windows: args.validation_windows,
                source_checkpoint_sha256: &authenticated.checkpoint_sha256,
                source_lineage_sha256: &authenticated.lineage_sha256,
                source_kind,
                emission_gradient_mode: &gradient_mode,
                optimizer_recipe_id: READOUT_OPTIMIZER_RECIPE_ID,
            },
        )?;
        let checkpoint_name = match &authenticated.provenance.weight_readout {
            WeightReadout::Raw => "mse_jepa_fitted.ot",
            WeightReadout::TailEma { .. } => "mse_jepa_tail_ema_fitted.ot",
        };
        save_bundle(
            &var_store,
            run.weights.join(checkpoint_name),
            authenticated.sigreg_lambda,
            authenticated.flow_lambda,
            CheckpointProvenance {
                completed_steps: authenticated.provenance.completed_steps,
                planned_steps: authenticated.provenance.planned_steps,
                steps_per_pass: authenticated.provenance.steps_per_pass,
                optimizer_recipe_id: authenticated.provenance.optimizer_recipe_id.clone(),
                emission_gradient_mode: authenticated.provenance.emission_gradient_mode,
                weight_readout: authenticated.provenance.weight_readout.clone(),
                core_origin: authenticated.provenance.core_origin.clone(),
                head_fit: Some(HeadFitProvenance {
                    source_checkpoint_sha256: authenticated.checkpoint_sha256.clone(),
                    source_lineage_sha256: authenticated.lineage_sha256.clone(),
                    source_weight_readout: authenticated.provenance.weight_readout.clone(),
                    source_core_origin: authenticated.provenance.core_origin.clone(),
                    seed: args.seed,
                    steps: args.steps as u64,
                    batch_size: args.batch_size as u64,
                    token_rows_per_step: args.token_rows_per_step as u64,
                    validation_windows: args.validation_windows as u64,
                    optimizer_recipe_id: READOUT_OPTIMIZER_RECIPE_ID.to_owned(),
                }),
            },
        )?;
        Ok(())
    })();
    if let Err(error) = publish {
        return super::rollout::cleanup_failed_publication(
            Path::new(RUNS_PATH),
            &run,
            previous_latest,
            error,
        );
    }
    println!(
        "Saved fitted MSE-JEPA readout endpoint and report under {}",
        run.root.display()
    );
    Ok(())
}

fn validate_args(args: &FitMseJepaReadoutsArgs) -> Result<()> {
    ensure!(!args.weights.is_empty(), "--weights is required");
    ensure!(!args.run.is_empty(), "--run is required");
    ensure!(
        args.steps == DEFAULT_READOUT_FIT_STEPS,
        "--steps must equal the canonical posthoc fit schedule ({DEFAULT_READOUT_FIT_STEPS})"
    );
    ensure!(
        args.batch_size == DEFAULT_READOUT_FIT_BATCH_SIZE,
        "--batch-size must equal the canonical posthoc fit batch ({DEFAULT_READOUT_FIT_BATCH_SIZE})"
    );
    ensure!(
        args.token_rows_per_step == DEFAULT_READOUT_TOKEN_ROWS,
        "--token-rows-per-step must equal the canonical aligned-row count ({DEFAULT_READOUT_TOKEN_ROWS})"
    );
    ensure!(
        args.validation_windows == DEFAULT_READOUT_VALIDATION_WINDOWS,
        "--validation-windows must equal the canonical posthoc panel ({DEFAULT_READOUT_VALIDATION_WINDOWS})"
    );
    ensure!(
        args.seed == DEFAULT_READOUT_FIT_SEED,
        "--seed must equal the dedicated canonical posthoc seed ({DEFAULT_READOUT_FIT_SEED:#x})"
    );
    ensure!(
        args.resolution_secs == 300,
        "fit-mse-jepa-readouts requires the train-fitted raw 300s BarSupports"
    );
    ensure!(
        args.min_bars >= (MAX_CONTEXT_BARS + 2) as usize,
        "--min-bars must admit a full context and held-out target"
    );
    ensure!(
        !(args.derive_split_bounds && args.split_bounds.is_some()),
        "--derive-split-bounds conflicts with --split-bounds"
    );
    Ok(())
}

fn set_head_trainability(var_store: &nn::VarStore) -> Result<()> {
    let mut found_emission = false;
    let mut found_probe = false;
    for (name, variable) in var_store.variables() {
        if name.starts_with(EMISSION_PREFIX) {
            found_emission = true;
            let _ = variable.set_requires_grad(true);
        } else if name.starts_with(TOKEN_PROBE_PREFIX) {
            found_probe = true;
            let _ = variable.set_requires_grad(true);
        }
    }
    ensure!(
        found_emission && found_probe,
        "posthoc readout parameters are incomplete"
    );
    Ok(())
}

fn assert_only_heads_trainable(var_store: &nn::VarStore) -> Result<()> {
    for (name, variable) in var_store.variables() {
        let is_head = name.starts_with(EMISSION_PREFIX) || name.starts_with(TOKEN_PROBE_PREFIX);
        ensure!(
            variable.requires_grad() == is_head,
            "posthoc parameter trainability mismatch for {name}"
        );
        if !is_head {
            ensure!(
                !variable.grad().defined(),
                "frozen backbone parameter {name} acquired a gradient"
            );
        }
    }
    Ok(())
}

fn backbone_fingerprint(var_store: &nn::VarStore) -> Vec<(String, [u64; 3])> {
    tch::no_grad(|| {
        let mut fingerprints = var_store
            .variables()
            .into_iter()
            .filter(|(name, _)| {
                !name.starts_with(EMISSION_PREFIX) && !name.starts_with(TOKEN_PROBE_PREFIX)
            })
            .map(|(name, tensor)| {
                let flat = tensor.to_kind(Kind::Double).reshape([-1]);
                let sum = flat.sum(Kind::Double).double_value(&[]).to_bits();
                let square_sum = flat.square().sum(Kind::Double).double_value(&[]).to_bits();
                let max = flat.abs().max().double_value(&[]).to_bits();
                (name, [sum, square_sum, max])
            })
            .collect::<Vec<_>>();
        fingerprints.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        fingerprints
    })
}

fn fixed_row_indices(
    seed: u64,
    step: usize,
    rows: i64,
    requested: usize,
    device: Device,
) -> Tensor {
    assert!(
        requested <= rows as usize,
        "declared readout row subsample exceeds this batch"
    );
    let count = requested;
    let mut rng = ChaCha12Rng::seed_from_u64(
        seed ^ TOKEN_ROW_SEED_DOMAIN ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    );
    let indices = sample(&mut rng, rows as usize, count)
        .into_vec()
        .into_iter()
        .map(|index| index as i64)
        .collect::<Vec<_>>();
    Tensor::from_slice(&indices).to_device(device)
}

fn evaluate_readouts(
    model: &MseJepaModel,
    token_probe: &MseJepaTokenProbe,
    supports: &BarSupports,
    bars: &Tensor,
    next_dof: &Tensor,
) -> (f64, f64, f64, f64) {
    let forward = model.forward(bars, false);
    let beliefs = forward.beliefs.reshape([-1, LATENT_DIM]);
    let tokens = forward
        .all_tokens
        .narrow(2, 1, MAX_CONTEXT_BARS)
        .reshape([-1, LATENT_DIM]);
    let dof = next_dof.reshape([-1, BAR_DOF as i64]);
    let bins = supports.bin_ids(&dof);
    pin_bfloat16_autocast();
    let (emission_logits, token_logits) = autocast(true, || {
        (
            model.emission_logits(&beliefs, &bins),
            token_probe.logits(&tokens, &bins),
        )
    });
    let smoothed = supports.targets(&dof, BarScoring::Smoothed);
    let hard = supports.targets(&dof, BarScoring::Hard);
    (
        bar_nll_from_logits(&emission_logits, &smoothed)
            .0
            .double_value(&[]),
        bar_nll_from_logits(&emission_logits, &hard)
            .0
            .double_value(&[]),
        bar_nll_from_logits(&token_logits, &smoothed)
            .0
            .double_value(&[]),
        bar_nll_from_logits(&token_logits, &hard)
            .0
            .double_value(&[]),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        assert_only_heads_trainable, fixed_row_indices, set_head_trainability, validate_args,
        FitMseJepaReadoutsArgs, DEFAULT_READOUT_FIT_BATCH_SIZE, DEFAULT_READOUT_FIT_SEED,
        DEFAULT_READOUT_FIT_STEPS, DEFAULT_READOUT_TOKEN_ROWS, DEFAULT_READOUT_VALIDATION_WINDOWS,
    };
    use tch::{nn, nn::OptimizerConfig, Device, Kind};

    #[test]
    fn aligned_row_subsample_is_seeded_and_arm_independent() {
        let first = fixed_row_indices(17, 3, 48_000, 4_096, Device::Cpu);
        let replay = fixed_row_indices(17, 3, 48_000, 4_096, Device::Cpu);
        let next = fixed_row_indices(17, 4, 48_000, 4_096, Device::Cpu);
        assert!(first.equal(&replay));
        assert!(!first.equal(&next));
        assert_eq!(first.size(), [4_096]);
    }

    #[test]
    fn frozen_backbone_stays_exact_while_both_readout_groups_train() {
        let mut var_store = nn::VarStore::new(Device::Cpu);
        let backbone = var_store
            .root()
            .var("backbone_weight", &[2], nn::Init::Const(1.0));
        let emission =
            (var_store.root() / "lejepa_emission").var("weight", &[2], nn::Init::Const(1.0));
        let probe =
            (var_store.root() / "posthoc_token_probe").var("weight", &[2], nn::Init::Const(1.0));
        var_store.freeze();
        set_head_trainability(&var_store).unwrap();
        assert_only_heads_trainable(&var_store).unwrap();
        let frozen = backbone.copy();
        let emission_before = emission.copy();
        let probe_before = probe.copy();
        let mut optimizer = nn::AdamW::default().build(&var_store, 1e-2).unwrap();
        (emission.square().sum(Kind::Float) + probe.square().sum(Kind::Float)).backward();
        optimizer.step();
        assert!(backbone.equal(&frozen));
        assert!(!emission.equal(&emission_before));
        assert!(!probe.equal(&probe_before));
        assert_only_heads_trainable(&var_store).unwrap();
    }

    #[test]
    fn fit_cli_contract_rejects_noncanonical_schedule() {
        let mut args = FitMseJepaReadoutsArgs {
            weights: "mse_jepa.ot".to_owned(),
            run: "fit".to_owned(),
            steps: 0,
            batch_size: DEFAULT_READOUT_FIT_BATCH_SIZE,
            token_rows_per_step: DEFAULT_READOUT_TOKEN_ROWS,
            validation_windows: DEFAULT_READOUT_VALIDATION_WINDOWS,
            seed: DEFAULT_READOUT_FIT_SEED,
            data_dir: "unused".to_owned(),
            resolution_secs: 300,
            min_bars: 60_002,
            split_bounds: None,
            derive_split_bounds: false,
        };
        assert!(validate_args(&args)
            .unwrap_err()
            .to_string()
            .contains("--steps"));
        args.steps = DEFAULT_READOUT_FIT_STEPS;
        args.token_rows_per_step = 0;
        assert!(validate_args(&args)
            .unwrap_err()
            .to_string()
            .contains("--token-rows-per-step"));
    }
}
