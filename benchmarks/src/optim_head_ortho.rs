//! NorMuon whole-matrix vs per-attention-head orthogonalization on the shared
//! transformer recall benchmark.

use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};
use trading_bot_0::torch::optim::{Muon, MuonConfig};

use crate::optim_transformer::{
    self as tf, force_adamw_substrings, named_trainable, FusedQkvGptModel, GptModel, COMPUTE_KIND,
};

const TRAIN_STEPS: usize = 400;
const DATASET_SIZE: i64 = 1024;
const BATCH_SIZE: i64 = 32;
const SEED: i64 = 42;
const EVAL_EVERY: usize = 20;
const TARGET_LOSS: f64 = 0.5;
const LR_GRID: &[f64] = &[3.6e-4, 7.5e-4, 1.15e-3, 1.5e-3, 2e-3, 3e-3, 5e-3];

#[derive(Clone, Copy)]
enum ModelKind {
    SplitQkv,
    FusedQkv,
}

impl ModelKind {
    fn label(self) -> &'static str {
        match self {
            Self::SplitQkv => "split-QKV",
            Self::FusedQkv => "fused-QKV",
        }
    }
}

enum BenchModel {
    Split(GptModel),
    Fused(FusedQkvGptModel),
}

impl BenchModel {
    fn forward(&self, tokens: &Tensor) -> Tensor {
        match self {
            Self::Split(model) => model.forward(tokens),
            Self::Fused(model) => model.forward(tokens),
        }
    }
}

#[derive(Clone, Copy)]
enum OrthoMode {
    WholeMatrix,
    AttentionQkvHeads,
    AttentionQkvoHeads,
}

impl OrthoMode {
    fn label(self) -> &'static str {
        match self {
            Self::WholeMatrix => "NorMuon whole-matrix",
            Self::AttentionQkvHeads => "NorMuon per-QKV-head",
            Self::AttentionQkvoHeads => "NorMuon per-QKVO-head",
        }
    }
}

struct RunStats {
    final_loss: f64,
    auc: f64,
    steps_to_target: Option<usize>,
    step_ms: f64,
    diverged: bool,
}

impl RunStats {
    fn from_curve(curve: &[(usize, f64)], step_ms: f64) -> Self {
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
            step_ms,
            diverged,
        }
    }
}

fn sync_device(device: Device) {
    if let Device::Cuda(id) = device {
        tch::Cuda::synchronize(id as i64);
    }
}

fn eval_loss(model: &BenchModel, inputs: &Tensor, targets: &Tensor) -> f64 {
    tch::no_grad(|| {
        let logits = model.forward(inputs);
        tf::lm_loss(&logits, targets).double_value(&[])
    })
}

fn build_model(device: Device, kind: ModelKind) -> (nn::VarStore, BenchModel) {
    tch::manual_seed(SEED);
    let mut vs = nn::VarStore::new(device);
    let model = match kind {
        ModelKind::SplitQkv => BenchModel::Split(GptModel::new(&vs.root())),
        ModelKind::FusedQkv => BenchModel::Fused(FusedQkvGptModel::new(&vs.root())),
    };
    if COMPUTE_KIND == Kind::BFloat16 {
        vs.bfloat16();
    }
    (vs, model)
}

fn train<F>(device: Device, model: &BenchModel, mut step_fn: F) -> (Vec<(usize, f64)>, f64)
where
    F: FnMut(&Tensor),
{
    tch::manual_seed(SEED + 1000);
    let data = tf::make_dataset(device, DATASET_SIZE);
    let mut curve = Vec::new();
    sync_device(device);
    let start = std::time::Instant::now();

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

    sync_device(device);
    let step_ms = start.elapsed().as_secs_f64() * 1000.0 / TRAIN_STEPS as f64;
    (curve, step_ms)
}

fn run_adamw(device: Device, kind: ModelKind, lr: f64) -> RunStats {
    let (vs, model) = build_model(device, kind);
    let mut opt = nn::AdamW::default().build(&vs, lr).expect("adamw build");
    let (curve, step_ms) = train(device, &model, |loss| opt.backward_step(loss));
    RunStats::from_curve(&curve, step_ms)
}

fn run_normuon(
    device: Device,
    kind: ModelKind,
    lr: f64,
    mode: OrthoMode,
    print_split: bool,
) -> RunStats {
    let (vs, model) = build_model(device, kind);
    let named = named_trainable(&vs);
    let mut opt = Muon::new_named(
        &named,
        MuonConfig {
            lr,
            beta2: 0.95,
            adamw_lr: 1e-3,
            force_adamw_name_substrings: force_adamw_substrings(),
            per_attention_head_ortho: !matches!(mode, OrthoMode::WholeMatrix),
            per_attention_output_head_ortho: matches!(mode, OrthoMode::AttentionQkvoHeads),
            attention_head_dim: tf::HEAD_DIM,
            quiet: !print_split,
            ..MuonConfig::default()
        },
    );
    let (curve, step_ms) = train(device, &model, |loss| {
        loss.backward();
        opt.step();
        opt.zero_grad();
    });
    RunStats::from_curve(&curve, step_ms)
}

fn fmt_target(s: Option<usize>) -> String {
    s.map(|s| s.to_string()).unwrap_or_else(|| "-".into())
}

fn print_header() {
    println!(
        "  {:<32} {:>12} {:>12} {:>14} {:>12}",
        "config", "final", "AUC", "steps->target", "ms/step"
    );
    println!("  {}", "-".repeat(88));
}

fn print_row(label: &str, stats: &RunStats) {
    let flag = if stats.diverged { "  DIVERGED" } else { "" };
    println!(
        "  {:<32} {:>12.6} {:>12.6} {:>14} {:>12.3}{}",
        label,
        stats.final_loss,
        stats.auc,
        fmt_target(stats.steps_to_target),
        stats.step_ms,
        flag
    );
}

pub fn run(device: Device) {
    println!("--- NorMuon Attention-Head Orthogonalization Benchmark ---");
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
        "  {} steps, batch {}, seed {}; fused QKV uses head-major QKV row blocks",
        TRAIN_STEPS, BATCH_SIZE, SEED
    );
    println!(
        "  metrics: final CE | AUC (mean eval CE) | steps-to-CE<{} | end-to-end train ms/step\n",
        TARGET_LOSS
    );

    run_suite(device, ModelKind::SplitQkv);
    println!();
    run_suite(device, ModelKind::FusedQkv);
}

fn run_suite(device: Device, kind: ModelKind) {
    println!("=== {} model ===", kind.label());
    println!("[routing] Existing whole-matrix route:");
    let _ = run_normuon(device, kind, 5e-3, OrthoMode::WholeMatrix, true);
    println!("\n[routing] Per-QKV-head route:");
    let _ = run_normuon(device, kind, 5e-3, OrthoMode::AttentionQkvHeads, true);
    println!("\n[routing] Per-QKVO-head route:");
    let _ = run_normuon(device, kind, 5e-3, OrthoMode::AttentionQkvoHeads, true);
    println!("\n[comparison] LR sweep");
    print_header();
    let adamw = run_adamw(device, kind, 1e-3);
    print_row("AdamW-all lr=1e-3", &adamw);

    let mut best_whole = (f64::INFINITY, 0.0);
    let mut best_qkv = (f64::INFINITY, 0.0);
    let mut best_qkvo = (f64::INFINITY, 0.0);
    for &lr in LR_GRID {
        let whole = run_normuon(device, kind, lr, OrthoMode::WholeMatrix, false);
        print_row(
            &format!("{} lr={:.2e}", OrthoMode::WholeMatrix.label(), lr),
            &whole,
        );
        if !whole.diverged && whole.final_loss < best_whole.0 {
            best_whole = (whole.final_loss, lr);
        }

        let heads = run_normuon(device, kind, lr, OrthoMode::AttentionQkvHeads, false);
        print_row(
            &format!("{} lr={:.2e}", OrthoMode::AttentionQkvHeads.label(), lr),
            &heads,
        );
        if !heads.diverged && heads.final_loss < best_qkv.0 {
            best_qkv = (heads.final_loss, lr);
        }

        let heads = run_normuon(device, kind, lr, OrthoMode::AttentionQkvoHeads, false);
        print_row(
            &format!("{} lr={:.2e}", OrthoMode::AttentionQkvoHeads.label(), lr),
            &heads,
        );
        if !heads.diverged && heads.final_loss < best_qkvo.0 {
            best_qkvo = (heads.final_loss, lr);
        }
    }

    println!("\n=== Summary ===");
    println!(
        "  best whole-matrix NorMuon: lr={:.2e} final {:.6}",
        best_whole.1, best_whole.0
    );
    println!(
        "  best per-QKV-head NorMuon: lr={:.2e} final {:.6}",
        best_qkv.1, best_qkv.0
    );
    println!(
        "  best per-QKVO-head NorMuon: lr={:.2e} final {:.6}",
        best_qkvo.1, best_qkvo.0
    );
    println!("  AdamW-all lr=1e-3 final {:.6}", adamw.final_loss);
    println!(
        "  QKV-head/whole final-CE ratio: {:.3}x",
        best_qkv.0 / best_whole.0
    );
    println!(
        "  QKVO-head/whole final-CE ratio: {:.3}x",
        best_qkvo.0 / best_whole.0
    );
}
