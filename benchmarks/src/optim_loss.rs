//! Loss-convergence sweep: AdamW vs NorMuon on a small GPT-style decoder
//! transformer trained on a deterministic, learnable in-context recall task,
//! with an LR sweep (and a beta2 spot-check at the best LR).
//!
//! This is the quick sibling of `optim_grid.rs` (the full grid). It shares the
//! transformer + dataset in `optim_transformer.rs`. Every optimizer/HP combo
//! trains from the SAME parameter init (seed 42) and sees the SAME minibatch
//! order, so differences are attributable to the optimizer alone.
//!
//! Routing matches the NorMuon paper: token/pos embeddings + LM head -> AdamW
//! (forced), RMSNorm gains (1D) -> AdamW, attention Q/K/V/O + MLP up/down 2D
//! weights -> NorMuon.

use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};
use trading_bot_0::torch::optim::{Muon, MuonConfig};

use crate::optim_transformer::{
    self as tf, force_adamw_substrings, named_trainable, GptModel, COMPUTE_KIND,
};

const TRAIN_STEPS: usize = 400;
const DATASET_SIZE: i64 = 1024;
const BATCH_SIZE: i64 = 32;
const SEED: i64 = 42;
const EVAL_EVERY: usize = 20;
/// CE threshold for "converged" on this recall task. Uniform-over-values CE is
/// ln(64) ≈ 4.16; a model that has learned the induction mechanism drops well
/// below 0.5, so 0.5 is a sensible "converged" mark.
const TARGET_LOSS: f64 = 0.5;

struct RunStats {
    final_loss: f64,
    auc: f64,
    steps_to_target: Option<usize>,
    diverged: bool,
}

impl RunStats {
    fn from_curve(curve: &[(usize, f64)]) -> Self {
        let final_loss = curve.last().map(|&(_, l)| l).unwrap_or(f64::NAN);
        let auc = curve.iter().map(|&(_, l)| l).sum::<f64>() / curve.len().max(1) as f64;
        let steps_to_target = curve
            .iter()
            .find(|&&(_, l)| l <= TARGET_LOSS)
            .map(|&(s, _)| s);
        let diverged = !final_loss.is_finite() || final_loss > 1e3;
        Self {
            final_loss,
            auc,
            steps_to_target,
            diverged,
        }
    }
}

fn eval_loss(model: &GptModel, inputs: &Tensor, targets: &Tensor) -> f64 {
    tch::no_grad(|| {
        let logits = model.forward(inputs);
        tf::lm_loss(&logits, targets).double_value(&[])
    })
}

/// Build the model with a fixed init seed and cast params to the compute dtype.
fn build_model(device: Device) -> (nn::VarStore, GptModel) {
    tch::manual_seed(SEED);
    let mut vs = nn::VarStore::new(device);
    let model = GptModel::new(&vs.root());
    if COMPUTE_KIND == Kind::BFloat16 {
        vs.bfloat16();
    }
    (vs, model)
}

fn train<F>(device: Device, model: &GptModel, mut step_fn: F) -> Vec<(usize, f64)>
where
    F: FnMut(&Tensor),
{
    tch::manual_seed(SEED + 1000);
    let data = tf::make_dataset(device, DATASET_SIZE);
    let mut curve = Vec::new();

    for step in 0..TRAIN_STEPS {
        let idx = Tensor::randint(DATASET_SIZE, [BATCH_SIZE], (Kind::Int64, device));
        let xb = data.inputs.index_select(0, &idx);
        let yb = data.targets.index_select(0, &idx);
        let logits = model.forward(&xb);
        let loss = tf::lm_loss(&logits, &yb);
        step_fn(&loss);

        if step % EVAL_EVERY == 0 || step == TRAIN_STEPS - 1 {
            curve.push((step + 1, eval_loss(model, &data.inputs, &data.targets)));
        }
    }
    curve
}

fn run_adamw(device: Device, lr: f64) -> RunStats {
    let (vs, model) = build_model(device);
    let mut opt = nn::AdamW::default().build(&vs, lr).expect("adamw build");
    let curve = train(device, &model, |loss| opt.backward_step(loss));
    RunStats::from_curve(&curve)
}

fn run_normuon(device: Device, lr: f64, beta2: f64, print_split: bool) -> RunStats {
    let (vs, model) = build_model(device);
    let named = named_trainable(&vs);
    let mut opt = Muon::new_named(
        &named,
        MuonConfig {
            lr,
            beta2,
            adamw_lr: 1e-3,
            force_adamw_name_substrings: force_adamw_substrings(),
            quiet: !print_split,
            ..MuonConfig::default()
        },
    );
    let curve = train(device, &model, |loss| {
        loss.backward();
        opt.step();
        opt.zero_grad();
    });
    RunStats::from_curve(&curve)
}

fn fmt_target(s: Option<usize>) -> String {
    s.map(|s| s.to_string()).unwrap_or_else(|| "—".into())
}

fn print_row(label: &str, stats: &RunStats) {
    let flag = if stats.diverged { "  DIVERGED" } else { "" };
    println!(
        "  {:<24} {:>12.6} {:>12.6} {:>14}{}",
        label,
        stats.final_loss,
        stats.auc,
        fmt_target(stats.steps_to_target),
        flag
    );
}

pub fn run(device: Device) {
    println!("--- Optimizer Loss-Convergence Sweep (AdamW vs NorMuon) ---");
    println!(
        "  task: GPT decoder (d={} L={} H={} seq={} vocab={}), in-context recall",
        tf::D_MODEL,
        tf::N_LAYERS,
        tf::N_HEADS,
        tf::SEQ_LEN,
        tf::VOCAB
    );
    println!(
        "  {} steps, batch {}, seed {} (deterministic), SDPA causal (FlashAttention on CUDA)",
        TRAIN_STEPS, BATCH_SIZE, SEED
    );
    println!(
        "  metrics: final CE | AUC (mean eval CE) | steps-to-CE<{}\n",
        TARGET_LOSS
    );

    // Print the NorMuon/AdamW routing split once.
    let _ = run_normuon(device, 1e-3, 0.95, true);

    println!(
        "  {:<24} {:>12} {:>12} {:>14}",
        "config", "final", "AUC", "steps->target"
    );
    println!("  {}", "-".repeat(66));

    let adamw_baseline = run_adamw(device, 3e-4);
    print_row("AdamW-all lr=3e-4", &adamw_baseline);
    let adamw_strong = run_adamw(device, 1e-3);
    print_row("AdamW-all lr=1e-3", &adamw_strong);
    let adamw_best = adamw_strong.final_loss.min(adamw_baseline.final_loss);

    println!();
    let lr_grid = [1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2];
    let mut best_lr = lr_grid[0];
    let mut best_final = f64::INFINITY;
    for &lr in &lr_grid {
        let stats = run_normuon(device, lr, 0.95, false);
        print_row(&format!("NorMuon lr={:.0e} b2=0.95", lr), &stats);
        if !stats.diverged && stats.final_loss < best_final {
            best_final = stats.final_loss;
            best_lr = lr;
        }
    }

    println!();
    println!("  beta2 sweep at best NorMuon lr={:.0e}:", best_lr);
    let mut best_beta2 = 0.95;
    let mut best_b2_final = f64::INFINITY;
    for &beta2 in &[0.9, 0.95, 0.99] {
        let stats = run_normuon(device, best_lr, beta2, false);
        print_row(
            &format!("NorMuon lr={:.0e} b2={:.2}", best_lr, beta2),
            &stats,
        );
        if !stats.diverged && stats.final_loss < best_b2_final {
            best_b2_final = stats.final_loss;
            best_beta2 = beta2;
        }
    }

    println!();
    let winner = if best_b2_final < adamw_best {
        "NorMuon"
    } else {
        "AdamW"
    };
    println!(
        "  BEST NorMuon: lr={:.0e} beta2={:.2} -> final {:.6}",
        best_lr, best_beta2, best_b2_final
    );
    println!("  BEST AdamW-all: final {:.6}", adamw_best);
    println!(
        "  WINNER: {} ({:.3}x final-CE ratio NorMuon/AdamW)\n",
        winner,
        best_b2_final / adamw_best
    );
}
