//! Reference-aligned NorMuon optimizer: EMA momentum/Nesterov, Newton-Schulz 5
//! orthogonalization, per-row second-moment (NorMuon) rescaling with Frobenius-
//! norm preservation, and AdamW for non-matrix params.

use std::collections::HashMap;
use tch::{Kind, Tensor};

const NS_A: f64 = 3.4445;
const NS_B: f64 = -4.7750;
const NS_C: f64 = 2.0315;
/// Canonical Newton-Schulz iteration count; the reference default and the only
/// value real training should ever use.
pub const DEFAULT_NS_STEPS: usize = 5;

pub struct MuonConfig {
    pub lr: f64,
    pub use_muon_for_2d: bool,
    pub momentum: f64,
    pub nesterov: bool,
    /// NorMuon second-moment EMA decay (beta2). Reference default 0.95.
    pub beta2: f64,
    pub weight_decay: f64,
    /// AdamW LR for scalar/1D params (biases, norms, embeddings).
    pub adamw_lr: f64,
    pub adamw_betas: (f64, f64),
    pub adamw_eps: f64,
    pub adamw_wd: f64,
    /// Newton-Schulz iteration count for orthogonalization. Reference default 5.
    /// Exposed only so offline sweeps can map the NS-steps landscape; real
    /// training must leave this at `DEFAULT_NS_STEPS`.
    pub ns_steps: usize,
    /// Parameter name fragments that should use AdamW even if they are 2D.
    pub force_adamw_name_substrings: Vec<String>,
    /// Suppress the one-line routing-split print at construction. Benchmarks
    /// that build many optimizers set this; real training leaves it false.
    pub quiet: bool,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: 5e-3,
            use_muon_for_2d: true,
            momentum: 0.99,
            nesterov: true,
            beta2: 0.95,
            weight_decay: 0.0,
            adamw_lr: 3e-4,
            adamw_betas: (0.9, 0.95),
            adamw_eps: 1e-8,
            adamw_wd: 0.0,
            ns_steps: DEFAULT_NS_STEPS,
            force_adamw_name_substrings: Vec::new(),
            quiet: false,
        }
    }
}

/// Per-2D-param state. Each 2D param holds its own momentum.
/// We deliberately do *not* stack same-shape params into a batched state tensor:
/// the N× VRAM multiplier from batching NS5's intermediates dwarfs any step-time
/// savings on realistic models. Keeping state per-param caps NS5's peak working
/// set at a single [m, n] matrix's worth of transients.
struct Entry2D {
    idx: usize,
    /// First-moment EMA buffer, shape [m, n].
    momentum: Tensor,
    /// NorMuon second-moment EMA buffer (mean-of-squares per row), shape [m, 1].
    /// Kept in fp32 regardless of param dtype: an EMA at gain (1-beta2)=0.05 in
    /// bf16 silently stalls because small increments round to zero.
    second_momentum: Tensor,
}

struct AdamWParamState {
    m: Tensor,
    v: Tensor,
}

pub struct Muon {
    cfg: MuonConfig,
    entries_2d: Vec<Entry2D>,
    adamw_indices: Vec<usize>,
    adamw_state: HashMap<usize, AdamWParamState>,
    step_count: i64,
    params: Vec<Tensor>,
}

/// Single-matrix Newton-Schulz iteration.
/// Runs NS_STEPS iterations in bf16 for speed; returns in the input's kind.
///
/// Each iteration is:  x ← NS_A·x + NS_B·(a·x) + NS_C·((a·a)·x), where a = x·xᵀ.
/// We fuse this into two `baddbmm` calls (on [1,p,q] views) to cut the number
/// of transient allocations per iter roughly in half vs naïve arithmetic.
/// tch's 2D `addmm` doesn't accept scalar beta/alpha, so we lift to 3D.
///
/// Peak live tensors during iter: `x` ([p,q]) + `a` ([p,p]) + `b` ([p,p]) +
/// `tmp` ([p,q]) = ~2·[p,q] + 2·[p,p]. For p == q this is ~4·[p,q]. This is
/// the inherent working-set cost of NS5 and cannot be eliminated without
/// changing the algorithm.
fn newtonschulz5(g: &Tensor, ns_steps: usize) -> Tensor {
    let orig_kind = g.kind();
    let transposed = g.size()[0] > g.size()[1];
    let x2d = if orig_kind == Kind::BFloat16 {
        g.shallow_clone()
    } else {
        g.to_kind(Kind::BFloat16)
    };
    let nrm = x2d
        .frobenius_norm([0i64, 1].as_slice(), false)
        .clamp_min(1e-7);
    let x2d = &x2d / &nrm;
    let x2d = if transposed { x2d.transpose(0, 1) } else { x2d };
    let mut x = x2d.unsqueeze(0); // [1, p, q] view — baddbmm needs 3D

    for _ in 0..ns_steps {
        let a = x.matmul(&x.transpose(-2, -1));
        let b = a.matmul(&a);
        // tmp = NS_A·x + NS_B·(a·x)
        let tmp = x.baddbmm(&a, &x, NS_A, NS_B);
        // x = tmp + NS_C·(b·x)
        x = tmp.baddbmm(&b, &x, 1.0, NS_C);
    }

    let x = x.squeeze_dim(0);
    let x = if transposed {
        x.transpose(0, 1).contiguous()
    } else {
        x
    };
    if orig_kind == Kind::BFloat16 {
        x
    } else {
        x.to_kind(orig_kind)
    }
}

/// NorMuon per-row second-moment rescale, all math in fp32. Scaling each row by
/// `step_size_i * ratio` keeps the total Frobenius norm of the update
/// (approximately) equal to its pre-divide value, because `ratio` is exactly the
/// global correction `||U||_F / ||diag(step_size) U||_F`. The `lerp_` writes the
/// raw second-moment EMA in place.
fn normuon_rescale(update: &Tensor, second_momentum: &mut Tensor, beta2: f64) -> Tensor {
    let uf = update.to_kind(Kind::Float);
    let cols = update.size()[1] as f64;
    // Per-row sum of squares over fan-in: [rows, 1].
    let row_sq_sum = uf
        .square()
        .sum_dim_intlist([-1i64].as_slice(), true, Kind::Float);
    // Per-row MEAN of squares (note the /cols): [rows, 1].
    let v_mean = &row_sq_sum / cols;
    // Frobenius^2 of the post-NS update, BEFORE the per-row divide: [1, 1].
    let vnorm_sq = row_sq_sum.sum_dim_intlist([-2i64].as_slice(), true, Kind::Float);
    // Raw EMA from 0, no bias correction: v = v*beta2 + v_mean*(1-beta2).
    let _ = second_momentum.lerp_(&v_mean, 1.0 - beta2);
    // Per-row step size = 1/(sqrt(v)+1e-10): [rows, 1].
    let step_size = (second_momentum.sqrt() + 1e-10).reciprocal();
    // Analytic post-divide Frobenius^2 = sum_i step_size_i^2 * row_sq_sum_i.
    let vnorm_new_sq = (&step_size * &step_size * &row_sq_sum).sum_dim_intlist(
        [-2i64].as_slice(),
        true,
        Kind::Float,
    );
    // Frobenius-preservation ratio: [1, 1].
    let ratio = vnorm_sq.sqrt() / (vnorm_new_sq.sqrt() + 1e-10);
    // Fused per-row scale, cast back to update kind: [rows, 1].
    let scale = (&step_size * &ratio).to_kind(update.kind());
    update * &scale
}

impl Muon {
    pub fn new(trainable_vars: &[Tensor], cfg: MuonConfig) -> Self {
        let named: Vec<(String, Tensor)> = trainable_vars
            .iter()
            .map(|t| (String::new(), t.shallow_clone()))
            .collect();
        Self::new_named(&named, cfg)
    }

    pub fn new_named(trainable_vars: &[(String, Tensor)], cfg: MuonConfig) -> Self {
        let params: Vec<Tensor> = trainable_vars
            .iter()
            .map(|(_, t)| t.shallow_clone())
            .collect();
        let mut entries_2d = Vec::new();
        let mut adamw_indices = Vec::new();

        for (i, (name, p)) in trainable_vars.iter().enumerate() {
            let force_adamw = cfg
                .force_adamw_name_substrings
                .iter()
                .any(|needle| name.contains(needle));
            if cfg.use_muon_for_2d && p.dim() == 2 && !force_adamw {
                let size = p.size();
                let (m, n) = (size[0], size[1]);
                let kind = p.kind();
                let device = p.device();
                entries_2d.push(Entry2D {
                    idx: i,
                    momentum: Tensor::zeros([m, n], (kind, device)),
                    second_momentum: Tensor::zeros([m, 1], (Kind::Float, device)),
                });
            } else {
                adamw_indices.push(i);
            }
        }

        if !cfg.quiet {
            if cfg.use_muon_for_2d {
                println!(
                    "NorMuon optimizer: {} 2D params (NS5 + per-row second moment), {} other params (AdamW)",
                    entries_2d.len(),
                    adamw_indices.len()
                );
            } else {
                println!(
                    "AdamW optimizer: {} params (Muon disabled for root-cause logging)",
                    adamw_indices.len()
                );
            }
        }

        Self {
            cfg,
            entries_2d,
            adamw_indices,
            adamw_state: HashMap::new(),
            step_count: 0,
            params,
        }
    }

    pub fn step(&mut self) {
        tch::no_grad(|| {
            self.step_count += 1;
            self.step_all_normuon();
            self.step_all_adamw();
        });
    }

    fn step_all_normuon(&mut self) {
        let beta1 = self.cfg.momentum;
        let beta2 = self.cfg.beta2;
        let nesterov = self.cfg.nesterov;
        let lr = self.cfg.lr;
        let wd = self.cfg.weight_decay;
        let wd_factor = if wd > 0.0 { Some(1.0 - lr * wd) } else { None };

        for entry in &mut self.entries_2d {
            let grad = self.params[entry.idx].grad();
            if !grad.defined() {
                continue;
            }

            // First-moment EMA: buf = buf*beta1 + grad*(1-beta1).
            let _ = entry.momentum.lerp_(&grad, 1.0 - beta1);

            // Nesterov combine: update = grad*(1-beta1) + momentum*beta1.
            let update = if nesterov {
                grad.lerp(&entry.momentum, beta1)
            } else {
                entry.momentum.shallow_clone()
            };

            // Newton-Schulz orthogonalization; returns [rows, cols] orientation.
            let update = newtonschulz5(&update, self.cfg.ns_steps);

            // NorMuon: per-row second-moment rescale, all math in fp32.
            let update = normuon_rescale(&update, &mut entry.second_momentum, beta2);

            // Aspect-ratio scale max(1, rows/cols)^0.5 (after NorMuon rescale).
            let size = update.size();
            let aspect_scale = (1.0_f64).max(size[0] as f64 / size[1] as f64).sqrt();
            let update = update.g_mul_scalar(aspect_scale);

            // Apply to param: decoupled weight decay, then the update.
            let mut p = self.params[entry.idx].shallow_clone();
            if let Some(k) = wd_factor {
                let _ = p.g_mul_scalar_(k);
            }
            let update = update.to_kind(p.kind()) * (-lr);
            let _ = p.g_add_(&update);
        }
    }

    fn step_all_adamw(&mut self) {
        let (beta1, beta2) = self.cfg.adamw_betas;
        let lr = self.cfg.adamw_lr;
        let eps = self.cfg.adamw_eps;
        let wd = self.cfg.adamw_wd;

        let bc1 = 1.0 - beta1.powi(self.step_count as i32);
        let bc2 = 1.0 - beta2.powi(self.step_count as i32);
        let step_size = -lr / bc1;
        let inv_bc2_sqrt = 1.0 / bc2.sqrt();

        for &idx in &self.adamw_indices {
            let mut p = self.params[idx].shallow_clone();
            let grad = p.grad();
            if !grad.defined() {
                continue;
            }

            let state = self
                .adamw_state
                .entry(idx)
                .or_insert_with(|| AdamWParamState {
                    m: Tensor::zeros_like(&grad),
                    v: Tensor::zeros_like(&grad),
                });

            if wd > 0.0 {
                let _ = p.g_mul_scalar_(1.0 - lr * wd);
            }

            let _ = state.m.lerp_(&grad, 1.0 - beta1);
            let _ = state.v.lerp_(&grad.square(), 1.0 - beta2);

            let denom = state.v.sqrt() * inv_bc2_sqrt + eps;
            let _ = p.g_add_(&(&state.m / &denom * step_size));
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            let mut g = p.grad();
            if g.defined() {
                let _ = g.zero_();
            }
        }
    }

    pub fn lr(&self) -> f64 {
        self.cfg.lr
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.cfg.lr = lr;
    }

    pub fn set_momentum(&mut self, momentum: f64) {
        self.cfg.momentum = momentum;
    }

    pub fn set_adamw_lr(&mut self, lr: f64) {
        self.cfg.adamw_lr = lr;
    }

    /// Total bytes of optimizer state currently allocated.
    /// 2D params: `momentum` + `second_momentum` per param.
    /// 1D params: AdamW `m` + `v` per param (lazy — zero until first step).
    pub fn state_bytes(&self) -> usize {
        let tensor_bytes = |t: &Tensor| t.numel() * t.kind().elt_size_in_bytes();
        let muon: usize = self
            .entries_2d
            .iter()
            .map(|e| tensor_bytes(&e.momentum) + tensor_bytes(&e.second_momentum))
            .sum();
        let adamw: usize = self
            .adamw_state
            .values()
            .map(|s| tensor_bytes(&s.m) + tensor_bytes(&s.v))
            .sum();
        muon + adamw
    }

    /// Test-only: shallow clone of the NorMuon second-moment buffer for the
    /// `n`-th 2D entry, so tests can assert it updates away from zero.
    #[cfg(test)]
    fn second_momentum_at(&self, n: usize) -> Tensor {
        self.entries_2d[n].second_momentum.shallow_clone()
    }
}

#[cfg(test)]
mod tests {
    use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

    use super::{normuon_rescale, Muon, MuonConfig};

    const HIDDEN: i64 = 128;
    const TRAIN_STEPS: usize = 500;
    const DATASET_SIZE: i64 = 2048;
    const BATCH_SIZE: i64 = 64;
    const INPUT_DIM: i64 = 16;

    fn build_mlp(vs: &nn::Path) -> impl Module {
        nn::seq()
            .add(nn::linear(
                vs / "fc1",
                INPUT_DIM,
                HIDDEN,
                Default::default(),
            ))
            .add_fn(|x| x.gelu("none"))
            .add(nn::linear(vs / "fc2", HIDDEN, HIDDEN, Default::default()))
            .add_fn(|x| x.gelu("none"))
            .add(nn::linear(vs / "fc3", HIDDEN, HIDDEN, Default::default()))
            .add_fn(|x| x.gelu("none"))
            .add(nn::linear(vs / "fc4", HIDDEN, 1, Default::default()))
    }

    /// Fixed dataset; training draws fresh minibatches via random indexing.
    fn make_dataset(device: Device) -> (Tensor, Tensor) {
        let _guard = tch::no_grad_guard();
        let w = Tensor::randn([INPUT_DIM, 4], (Kind::Float, device));
        let x = Tensor::randn([DATASET_SIZE, INPUT_DIM], (Kind::Float, device));
        let h = x.matmul(&w);
        let y = (h.slice(1, 0, 1, 1).sin() * h.slice(1, 1, 2, 1).cos()
            + 0.3 * h.slice(1, 2, 3, 1) * h.slice(1, 3, 4, 1).tanh())
            + 0.05 * Tensor::randn([DATASET_SIZE, 1], (Kind::Float, device));
        (x, y)
    }

    /// Eval loss over full dataset (no grad).
    fn eval_loss(net: &dyn Module, x: &Tensor, y: &Tensor) -> f64 {
        tch::no_grad(|| {
            let pred = net.forward(x);
            (&pred - y).square().mean(Kind::Float).double_value(&[])
        })
    }

    fn train_adamw(device: Device, seed: i64) -> Vec<f64> {
        tch::manual_seed(seed);
        let vs = nn::VarStore::new(device);
        let net = build_mlp(&vs.root());
        let mut opt = nn::AdamW::default().build(&vs, 1e-3).expect("adamw");

        tch::manual_seed(seed + 1000);
        let (x_all, y_all) = make_dataset(device);
        let mut losses = Vec::with_capacity(TRAIN_STEPS);

        for step in 0..TRAIN_STEPS {
            let idx = Tensor::randint(DATASET_SIZE, [BATCH_SIZE], (Kind::Int64, device));
            let xb = x_all.index_select(0, &idx);
            let yb = y_all.index_select(0, &idx);

            let pred = net.forward(&xb);
            let loss = (&pred - &yb).square().mean(Kind::Float);
            opt.backward_step(&loss);

            if step % 50 == 0 || step == TRAIN_STEPS - 1 {
                losses.push(eval_loss(&net, &x_all, &y_all));
            }
        }
        losses
    }

    fn train_muon(device: Device, seed: i64) -> Vec<f64> {
        tch::manual_seed(seed);
        let vs = nn::VarStore::new(device);
        let net = build_mlp(&vs.root());
        let trainable = vs.trainable_variables();
        let mut opt = Muon::new(
            &trainable,
            MuonConfig {
                lr: 5e-3,
                adamw_lr: 1e-3,
                ..MuonConfig::default()
            },
        );

        tch::manual_seed(seed + 1000);
        let (x_all, y_all) = make_dataset(device);
        let mut losses = Vec::with_capacity(TRAIN_STEPS);

        for step in 0..TRAIN_STEPS {
            let idx = Tensor::randint(DATASET_SIZE, [BATCH_SIZE], (Kind::Int64, device));
            let xb = x_all.index_select(0, &idx);
            let yb = y_all.index_select(0, &idx);

            let pred = net.forward(&xb);
            let loss = (&pred - &yb).square().mean(Kind::Float);
            loss.backward();
            opt.step();
            opt.zero_grad();

            if step % 50 == 0 || step == TRAIN_STEPS - 1 {
                losses.push(eval_loss(&net, &x_all, &y_all));
            }
        }
        losses
    }

    /// Exercises the bf16 path of `batched_newtonschulz5`: the production
    /// trading bot runs after `vs.bfloat16()` so every ShapeGroup stores its
    /// momentum in bf16. This path has different code from fp32 (shallow vs
    /// kind-convert, in-place vs out-of-place ops) and can silently diverge
    /// if the NS5 implementation aliases caller storage.
    fn train_muon_bf16(device: Device, seed: i64) -> Vec<f64> {
        tch::manual_seed(seed);
        let mut vs = nn::VarStore::new(device);
        let net = build_mlp(&vs.root());
        vs.bfloat16();
        let trainable = vs.trainable_variables();
        let mut opt = Muon::new(
            &trainable,
            MuonConfig {
                lr: 5e-3,
                adamw_lr: 1e-3,
                ..MuonConfig::default()
            },
        );

        tch::manual_seed(seed + 1000);
        let (x_all_f32, y_all_f32) = make_dataset(device);
        let x_all = x_all_f32.to_kind(Kind::BFloat16);
        let y_all = y_all_f32.to_kind(Kind::BFloat16);
        let mut losses = Vec::with_capacity(TRAIN_STEPS);

        for step in 0..TRAIN_STEPS {
            let idx = Tensor::randint(DATASET_SIZE, [BATCH_SIZE], (Kind::Int64, device));
            let xb = x_all.index_select(0, &idx);
            let yb = y_all.index_select(0, &idx);

            let pred = net.forward(&xb);
            let loss = (&pred - &yb).square().mean(Kind::Float);
            loss.backward();
            opt.step();
            opt.zero_grad();

            if step % 50 == 0 || step == TRAIN_STEPS - 1 {
                losses.push(eval_loss(&net, &x_all, &y_all));
            }
        }
        losses
    }

    #[test]
    fn muon_converges_bf16() {
        let device = Device::Cpu;
        let losses = train_muon_bf16(device, 42);
        let first = losses[0];
        let last = *losses.last().unwrap();
        println!(
            "Muon bf16: loss {:.6} -> {:.6} ({:.1}x reduction)",
            first,
            last,
            first / last
        );
        assert!(
            last < first * 0.2,
            "Muon bf16 failed to converge: {:.6} -> {:.6}",
            first,
            last
        );
    }

    #[test]
    fn muon_converges_on_synthetic_regression() {
        let device = Device::Cpu;
        let losses = train_muon(device, 42);
        let first = losses[0];
        let last = *losses.last().unwrap();
        println!(
            "Muon: loss {:.6} -> {:.6} ({:.1}x reduction)",
            first,
            last,
            first / last
        );
        assert!(
            last < first * 0.1,
            "Muon failed to converge: {:.6} -> {:.6}",
            first,
            last
        );
    }

    #[test]
    fn adamw_converges_on_synthetic_regression() {
        let device = Device::Cpu;
        let losses = train_adamw(device, 42);
        let first = losses[0];
        let last = *losses.last().unwrap();
        println!(
            "AdamW: loss {:.6} -> {:.6} ({:.1}x reduction)",
            first,
            last,
            first / last
        );
        assert!(
            last < first * 0.1,
            "AdamW failed to converge: {:.6} -> {:.6}",
            first,
            last
        );
    }

    #[test]
    fn muon_vs_adamw_comparison() {
        let device = Device::Cpu;
        let seed = 42;

        let adamw_losses = train_adamw(device, seed);
        let muon_losses = train_muon(device, seed);

        println!(
            "\n{:<8} {:>12} {:>12} {:>10}",
            "Step", "AdamW", "Muon", "Winner"
        );
        println!("{}", "-".repeat(46));
        let steps: Vec<usize> = (0..TRAIN_STEPS)
            .filter(|&s| s % 50 == 0 || s == TRAIN_STEPS - 1)
            .collect();
        for (i, &s) in steps.iter().enumerate() {
            let a = adamw_losses[i];
            let m = muon_losses[i];
            let winner = if m < a { "Muon" } else { "AdamW" };
            println!("{:<8} {:>12.6} {:>12.6} {:>10}", s + 1, a, m, winner);
        }

        let adamw_final = *adamw_losses.last().unwrap();
        let muon_final = *muon_losses.last().unwrap();
        println!(
            "\nFinal ratio (Muon/AdamW): {:.3}x  — {}",
            muon_final / adamw_final,
            if muon_final < adamw_final {
                "Muon wins"
            } else {
                "AdamW wins"
            }
        );

        // Both must converge
        assert!(
            adamw_final < 0.5,
            "AdamW did not converge: {:.6}",
            adamw_final
        );
        assert!(muon_final < 0.5, "Muon did not converge: {:.6}", muon_final);
    }

    #[test]
    fn normuon_rescale_preserves_frobenius_norm() {
        let _g = tch::no_grad_guard();
        tch::manual_seed(7);
        let device = Device::Cpu;
        // Post-NS-shaped update: [rows, cols], rows != cols to exercise broadcast.
        let update = Tensor::randn([37, 53], (Kind::Float, device));
        let pre_norm = update.square().sum(Kind::Float).sqrt().double_value(&[]);

        let mut second_momentum = Tensor::zeros([37, 1], (Kind::Float, device));
        let rescaled = normuon_rescale(&update, &mut second_momentum, 0.95);

        let post_norm = rescaled.square().sum(Kind::Float).sqrt().double_value(&[]);
        let rel = (post_norm - pre_norm).abs() / pre_norm;
        println!(
            "NorMuon rescale: ||U||_F {:.6} -> {:.6} (rel diff {:.2e})",
            pre_norm, post_norm, rel
        );
        assert!(
            rel < 1e-4,
            "Frobenius norm not preserved: {:.6} -> {:.6} (rel {:.2e})",
            pre_norm,
            post_norm,
            rel
        );

        // Second moment must have moved off zero and stay finite.
        let v_min = second_momentum.min().double_value(&[]);
        let v_max = second_momentum.max().double_value(&[]);
        assert!(v_min > 0.0, "second_momentum stayed at zero: min={}", v_min);
        assert!(
            v_max.is_finite() && v_min.is_finite(),
            "second_momentum not finite: [{}, {}]",
            v_min,
            v_max
        );
    }

    #[test]
    fn normuon_step_updates_second_moment_and_stays_finite() {
        let device = Device::Cpu;
        tch::manual_seed(11);
        let vs = nn::VarStore::new(device);
        let net = build_mlp(&vs.root());
        let trainable = vs.trainable_variables();
        let mut opt = Muon::new(
            &trainable,
            MuonConfig {
                lr: 5e-3,
                adamw_lr: 1e-3,
                ..MuonConfig::default()
            },
        );

        tch::manual_seed(99);
        let (x_all, y_all) = make_dataset(device);

        // Before any step, every second-moment buffer is exactly zero.
        let before = opt.second_momentum_at(0);
        assert_eq!(before.max().double_value(&[]), 0.0);

        for _ in 0..5 {
            let idx = Tensor::randint(DATASET_SIZE, [BATCH_SIZE], (Kind::Int64, device));
            let xb = x_all.index_select(0, &idx);
            let yb = y_all.index_select(0, &idx);
            let pred = net.forward(&xb);
            let loss = (&pred - &yb).square().mean(Kind::Float);
            loss.backward();
            opt.step();
            opt.zero_grad();
        }

        let after = opt.second_momentum_at(0);
        let v_min = after.min().double_value(&[]);
        let v_max = after.max().double_value(&[]);
        println!(
            "second_momentum[0] after 5 steps: [{:.3e}, {:.3e}]",
            v_min, v_max
        );
        assert!(v_min > 0.0, "second_momentum did not update: min={}", v_min);
        assert!(
            v_min.is_finite() && v_max.is_finite(),
            "second_momentum not finite: [{}, {}]",
            v_min,
            v_max
        );
        // second_momentum is per-row of the [out, in] weight => [out, 1].
        assert_eq!(after.size(), vec![HIDDEN, 1]);
        assert_eq!(after.kind(), Kind::Float);
    }
}
