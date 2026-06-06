//! Full NorMuon hyperparameter grid search on a small GPT-style decoder
//! transformer trained on a deterministic, learnable in-context recall task
//! (shared with `optim_loss.rs` via `optim_transformer.rs`). Every config sees
//! identical init (seed 42) + identical minibatch stream, so all results are
//! directly comparable.
//!
//! Routing matches the NorMuon paper exactly:
//!   - token embedding, position embedding, LM head -> AdamW (forced, even 2D)
//!   - RMSNorm gains (1D) -> AdamW (automatic, dim==1)
//!   - attention Q/K/V/O + MLP up/down (2D hidden matrices) -> NorMuon
//!
//! Stages:
//!   1. AdamW-FOR-EVERYTHING baseline LR sweep (use_muon=false), so we can see
//!      NorMuon-split vs all-AdamW on the SAME task.
//!   2. Primary cartesian grid over LR x momentum(beta1) x nesterov (beta2 fixed
//!      0.95), at the best AdamW head LR.
//!   3. Secondary sweeps at the primary best: AdamW head LR, beta2 spot-check,
//!      Newton-Schulz steps, decoupled weight_decay.
//!
//! Nothing here mutates the real training config or the optimizer's default NS
//! step count (5); NS-steps is varied only through the benchmark-only
//! `MuonConfig.ns_steps` field.

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
/// CE threshold for "converged". Uniform-over-values CE is ln(64) ≈ 4.16.
const TARGET_LOSS: f64 = 0.5;

#[derive(Clone)]
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

/// AdamW-for-everything baseline (the real all-AdamW alternative).
fn run_adamw_all(device: Device, lr: f64) -> RunStats {
    let (vs, model) = build_model(device);
    let mut opt = nn::AdamW::default().build(&vs, lr).expect("adamw build");
    let curve = train(device, &model, |loss| opt.backward_step(loss));
    RunStats::from_curve(&curve)
}

/// One NorMuon-split config. `adamw_lr` is the LR for the AdamW arm
/// (embeddings + LM head + 1D norm gains); it matters on a transformer.
#[derive(Clone, Copy)]
struct NorMuonCfg {
    lr: f64,
    momentum: f64,
    beta2: f64,
    nesterov: bool,
    ns_steps: usize,
    weight_decay: f64,
    adamw_lr: f64,
}

impl Default for NorMuonCfg {
    fn default() -> Self {
        Self {
            lr: 2e-2,
            momentum: 0.95,
            beta2: 0.95,
            nesterov: true,
            ns_steps: 5,
            weight_decay: 0.0,
            adamw_lr: 1e-3,
        }
    }
}

fn run_normuon(device: Device, c: NorMuonCfg, print_split: bool) -> RunStats {
    let (vs, model) = build_model(device);
    let named = named_trainable(&vs);
    let mut opt = Muon::new_named(
        &named,
        MuonConfig {
            lr: c.lr,
            momentum: c.momentum,
            beta2: c.beta2,
            nesterov: c.nesterov,
            ns_steps: c.ns_steps,
            weight_decay: c.weight_decay,
            adamw_lr: c.adamw_lr,
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

fn fmt_bool(b: bool) -> &'static str {
    if b {
        "T"
    } else {
        "F"
    }
}

struct GridRow {
    cfg: NorMuonCfg,
    stats: RunStats,
}

fn label(c: &NorMuonCfg) -> String {
    format!(
        "lr={:<5.0e} b1={:.2} b2={:.2} nest={} hlr={:.0e}",
        c.lr,
        c.momentum,
        c.beta2,
        fmt_bool(c.nesterov),
        c.adamw_lr
    )
}

fn print_header() {
    println!(
        "  {:<46} {:>12} {:>12} {:>14}",
        "config", "final", "AUC", "steps->target"
    );
    println!("  {}", "-".repeat(86));
}

fn print_row(lbl: &str, s: &RunStats) {
    let flag = if s.diverged { "  DIVERGED" } else { "" };
    println!(
        "  {:<46} {:>12.6} {:>12.6} {:>14}{}",
        lbl,
        s.final_loss,
        s.auc,
        fmt_target(s.steps_to_target),
        flag
    );
}

pub fn run(device: Device) {
    println!("--- NorMuon FULL Hyperparameter Grid Search (TRANSFORMER) ---");
    println!(
        "  task: GPT decoder (d={} L={} H={} head_dim={} seq={} vocab={}), in-context recall",
        tf::D_MODEL,
        tf::N_LAYERS,
        tf::N_HEADS,
        tf::HEAD_DIM,
        tf::SEQ_LEN,
        tf::VOCAB
    );
    println!(
        "  {} steps, batch {}, seed {} (deterministic), SDPA causal (FlashAttention on CUDA)",
        TRAIN_STEPS, BATCH_SIZE, SEED
    );
    println!(
        "  metrics: final CE | AUC (mean eval CE) | steps-to-CE<{}",
        TARGET_LOSS
    );
    println!("  routing: tok/pos embed + lm_head -> AdamW (forced); norm gains (1D) -> AdamW;");
    println!("           attn Q/K/V/O + MLP up/down (2D) -> NorMuon\n");

    // Print the routing split once (NorMuon vs AdamW param counts).
    println!("[routing] NorMuon-vs-AdamW param split:");
    let _ = run_normuon(device, NorMuonCfg::default(), true);
    println!();

    // ---- Stage 1: AdamW-for-everything baseline LR sweep. ----
    println!("[1/3] AdamW-for-EVERYTHING baseline LR sweep (use_muon=false for all)");
    print_header();
    let mut adamw_best = f64::INFINITY;
    let mut adamw_best_lr = 0.0;
    for &lr in &[3e-4, 1e-3, 3e-3] {
        let s = run_adamw_all(device, lr);
        print_row(&format!("AdamW-all lr={:.0e}", lr), &s);
        if !s.diverged && s.final_loss < adamw_best {
            adamw_best = s.final_loss;
            adamw_best_lr = lr;
        }
    }
    println!(
        "  -> best AdamW-all: lr={:.0e} final {:.6}\n",
        adamw_best_lr, adamw_best
    );

    // ---- Stage 2: primary cartesian grid (beta2 fixed 0.95). ----
    // First, pick the AdamW head LR for the NorMuon-split arm at a reasonable
    // central NorMuon LR, so the primary grid runs at a good head LR.
    println!("[head-lr] AdamW head/embed LR pick for the NorMuon split (at lr=1e-2, b1=0.95):");
    print_header();
    let mut head_best = f64::INFINITY;
    let mut head_best_lr = 1e-3;
    for &hlr in &[3e-4, 1e-3] {
        let cfg = NorMuonCfg {
            lr: 1e-2,
            momentum: 0.95,
            adamw_lr: hlr,
            ..Default::default()
        };
        let s = run_normuon(device, cfg, false);
        print_row(&label(&cfg), &s);
        if !s.diverged && s.final_loss < head_best {
            head_best = s.final_loss;
            head_best_lr = hlr;
        }
    }
    println!(
        "  -> best head LR: {:.0e} (final {:.6})\n",
        head_best_lr, head_best
    );

    let lr_grid = [1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2];
    let mom_grid = [0.9, 0.95, 0.99];
    let nesterov_grid = [true, false];
    let total = lr_grid.len() * mom_grid.len() * nesterov_grid.len();
    println!(
        "[2/3] Primary grid: LR({}) x b1({}) x nesterov({}) = {} configs (b2=0.95, hlr={:.0e})",
        lr_grid.len(),
        mom_grid.len(),
        nesterov_grid.len(),
        total,
        head_best_lr
    );

    let mut rows: Vec<GridRow> = Vec::with_capacity(total);
    let mut done = 0usize;
    for &lr in &lr_grid {
        for &momentum in &mom_grid {
            for &nesterov in &nesterov_grid {
                let cfg = NorMuonCfg {
                    lr,
                    momentum,
                    nesterov,
                    adamw_lr: head_best_lr,
                    ..NorMuonCfg::default()
                };
                let stats = run_normuon(device, cfg, false);
                done += 1;
                println!(
                    "  [{:>3}/{}] {:<46} final {:.6} auc {:.6} ->{}{}",
                    done,
                    total,
                    label(&cfg),
                    stats.final_loss,
                    stats.auc,
                    fmt_target(stats.steps_to_target),
                    if stats.diverged { " DIVERGED" } else { "" }
                );
                rows.push(GridRow { cfg, stats });
            }
        }
    }

    let key = |r: &GridRow| {
        if r.stats.diverged {
            f64::INFINITY
        } else {
            r.stats.final_loss
        }
    };
    rows.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap());

    println!("\n  === TOP 10 configs (by final CE) ===");
    print_header();
    for r in rows.iter().take(10) {
        print_row(&label(&r.cfg), &r.stats);
    }

    println!("\n  === Best config PER LR (LR x momentum interaction) ===");
    print_header();
    for &lr in &lr_grid {
        if let Some(best) = rows
            .iter()
            .filter(|r| r.cfg.lr == lr && !r.stats.diverged)
            .min_by(|a, b| a.stats.final_loss.partial_cmp(&b.stats.final_loss).unwrap())
        {
            print_row(&label(&best.cfg), &best.stats);
        } else {
            println!("  lr={:<5.0e}  (all diverged)", lr);
        }
    }

    let primary_best = rows
        .iter()
        .find(|r| !r.stats.diverged)
        .expect("at least one primary config converged")
        .cfg;
    println!(
        "\n  -> primary best: {}  (final {:.6})\n",
        label(&primary_best),
        run_normuon(device, primary_best, false).final_loss
    );

    // ---- Stage 3: secondary 1-D sweeps at the primary best. ----
    println!("[3/3] Secondary 1-D sweeps (holding primary best fixed)");

    println!("\n  beta2 spot-check:");
    print_header();
    let mut best_b2 = primary_best.beta2;
    let mut best_b2_final = f64::INFINITY;
    for &b2 in &[0.9, 0.95, 0.99] {
        let cfg = NorMuonCfg {
            beta2: b2,
            ..primary_best
        };
        let s = run_normuon(device, cfg, false);
        print_row(&label(&cfg), &s);
        if !s.diverged && s.final_loss < best_b2_final {
            best_b2_final = s.final_loss;
            best_b2 = b2;
        }
    }

    println!("\n  AdamW head/embed LR sweep (at primary best):");
    print_header();
    let mut best_hlr = primary_best.adamw_lr;
    let mut best_hlr_final = f64::INFINITY;
    for &hlr in &[3e-4, 1e-3] {
        let cfg = NorMuonCfg {
            adamw_lr: hlr,
            beta2: best_b2,
            ..primary_best
        };
        let s = run_normuon(device, cfg, false);
        print_row(&label(&cfg), &s);
        if !s.diverged && s.final_loss < best_hlr_final {
            best_hlr_final = s.final_loss;
            best_hlr = hlr;
        }
    }

    println!("\n  Newton-Schulz steps sweep:");
    print_header();
    let mut best_ns = primary_best.ns_steps;
    let mut best_ns_final = f64::INFINITY;
    for &ns in &[3usize, 5, 6] {
        let cfg = NorMuonCfg {
            ns_steps: ns,
            beta2: best_b2,
            adamw_lr: best_hlr,
            ..primary_best
        };
        let s = run_normuon(device, cfg, false);
        let mark = if ns == 5 {
            " (canonical/real default)"
        } else {
            ""
        };
        print_row(&format!("{}  ns={}{}", label(&cfg), ns, mark), &s);
        if !s.diverged && s.final_loss < best_ns_final {
            best_ns_final = s.final_loss;
            best_ns = ns;
        }
    }

    println!("\n  weight_decay sweep (decoupled):");
    print_header();
    let mut best_wd = primary_best.weight_decay;
    let mut best_wd_final = f64::INFINITY;
    for &wd in &[0.0, 1e-4, 1e-3] {
        let cfg = NorMuonCfg {
            weight_decay: wd,
            beta2: best_b2,
            adamw_lr: best_hlr,
            ns_steps: best_ns,
            ..primary_best
        };
        let s = run_normuon(device, cfg, false);
        print_row(&format!("{}  wd={:.0e}", label(&cfg), wd), &s);
        if !s.diverged && s.final_loss < best_wd_final {
            best_wd_final = s.final_loss;
            best_wd = wd;
        }
    }

    // ---- Verdict + recommended config. ----
    let recommended = NorMuonCfg {
        beta2: best_b2,
        adamw_lr: best_hlr,
        ns_steps: best_ns,
        weight_decay: best_wd,
        ..primary_best
    };
    let rec_stats = run_normuon(device, recommended, false);

    println!("\n=== RECOMMENDED NorMuon (split) CONFIG ===");
    println!(
        "  lr={:.0e}  momentum(b1)={:.2}  beta2={:.2}  nesterov={}  ns_steps={}  wd={:.0e}  head_lr(AdamW)={:.0e}",
        recommended.lr,
        recommended.momentum,
        recommended.beta2,
        recommended.nesterov,
        recommended.ns_steps,
        recommended.weight_decay,
        recommended.adamw_lr
    );
    println!(
        "  -> final {:.6} | AUC {:.6} | steps->target {}",
        rec_stats.final_loss,
        rec_stats.auc,
        fmt_target(rec_stats.steps_to_target)
    );
    println!(
        "  best AdamW-FOR-EVERYTHING baseline: lr={:.0e} final {:.6}",
        adamw_best_lr, adamw_best
    );
    let winner = if rec_stats.final_loss < adamw_best {
        "NorMuon-split"
    } else {
        "AdamW-all"
    };
    println!(
        "  WINNER: {} ({:.3}x final-CE ratio NorMuon/AdamW-all)",
        winner,
        rec_stats.final_loss / adamw_best
    );
}
