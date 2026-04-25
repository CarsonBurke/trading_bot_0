//! NorMuon optimizer (Neuron-wise Normalized Muon).
//! arXiv:2510.05491v1 — Muon + per-row adaptive LR via second moment of NS5 output.
//! Named `muon`/`Muon` in the API for continuity; the algorithm is NorMuon.

use std::collections::HashMap;
use tch::{Kind, Tensor};

const NS_A: f64 = 3.4445;
const NS_B: f64 = -4.7750;
const NS_C: f64 = 2.0315;
const NS_STEPS: usize = 5;
const RMS_MATCH_COEF: f64 = 0.2;

pub struct MuonConfig {
    pub lr: f64,
    pub use_muon_for_2d: bool,
    /// EMA coef for first moment (momentum before NS5).
    pub beta1: f64,
    /// EMA coef for per-row second moment of NS5 output.
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    /// AdamW LR for scalar/1D params (biases, norms, embeddings).
    pub adamw_lr: f64,
    pub adamw_betas: (f64, f64),
    pub adamw_eps: f64,
    pub adamw_wd: f64,
    /// Parameter name fragments that should use AdamW even if they are 2D.
    pub force_adamw_name_substrings: Vec<String>,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            // NorMuon effective per-entry step is ~0.2*lr, shape-invariant.
            // Comparable to AdamW lr=1e-3 at NorMuon lr=5e-3.
            lr: 5e-3,
            use_muon_for_2d: true,
            beta1: 0.95,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.0,
            adamw_lr: 3e-4,
            adamw_betas: (0.9, 0.999),
            adamw_eps: 1e-8,
            adamw_wd: 0.0,
            force_adamw_name_substrings: Vec::new(),
        }
    }
}

/// Per-2D-param state. Each 2D param holds its own momentum and row second-moment.
/// We deliberately do *not* stack same-shape params into a batched state tensor:
/// the N× VRAM multiplier from batching NS5's intermediates dwarfs any step-time
/// savings on realistic models. Keeping state per-param caps NS5's peak working
/// set at a single [m, n] matrix's worth of transients.
struct Entry2D {
    idx: usize,
    transposed: bool,
    /// sqrt(m * n) — used in RMS-match LR scale.
    sqrt_mn: f64,
    /// First-moment EMA, shape [m, n].
    momentum: Tensor,
    /// Per-row second-moment EMA, shape [m] (m = output neurons).
    row_sq_mean: Tensor,
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

/// Single-matrix Newton-Schulz iteration. Input: [p, q] with p <= q.
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
fn newtonschulz5(g: &Tensor) -> Tensor {
    let orig_kind = g.kind();
    // g may alias caller storage (shallow_clone / transpose view); compute the
    // normalized copy out-of-place so in-place ops below don't mutate it.
    let nrm = g
        .frobenius_norm([0i64, 1].as_slice(), false)
        .clamp_min(1e-7);
    let x2d = if orig_kind == Kind::BFloat16 {
        g / &nrm
    } else {
        g.to_kind(Kind::BFloat16) / &nrm
    };
    let mut x = x2d.unsqueeze(0); // [1, p, q] view — baddbmm needs 3D

    for _ in 0..NS_STEPS {
        let a = x.matmul(&x.transpose(-2, -1));
        let b = a.matmul(&a);
        // tmp = NS_A·x + NS_B·(a·x)
        let tmp = x.baddbmm(&a, &x, NS_A, NS_B);
        // x = tmp + NS_C·(b·x)
        x = tmp.baddbmm(&b, &x, 1.0, NS_C);
    }

    let x = x.squeeze_dim(0);
    if orig_kind == Kind::BFloat16 {
        x
    } else {
        x.to_kind(orig_kind)
    }
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
                    transposed: m > n,
                    sqrt_mn: ((m * n) as f64).sqrt(),
                    momentum: Tensor::zeros([m, n], (kind, device)),
                    row_sq_mean: Tensor::zeros([m], (kind, device)),
                });
            } else {
                adamw_indices.push(i);
            }
        }

        if cfg.use_muon_for_2d {
            println!(
                "NorMuon optimizer: {} 2D params (per-param NS5), {} other params (AdamW)",
                entries_2d.len(),
                adamw_indices.len()
            );
        } else {
            println!(
                "AdamW optimizer: {} params (Muon disabled for root-cause logging)",
                adamw_indices.len()
            );
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
        let beta1 = self.cfg.beta1;
        let beta2 = self.cfg.beta2;
        let eps = self.cfg.eps;
        let lr = self.cfg.lr;
        let wd = self.cfg.weight_decay;
        let neg_eta_coef = -RMS_MATCH_COEF * lr;
        let wd_factor = if wd > 0.0 { Some(1.0 - lr * wd) } else { None };

        for entry in &mut self.entries_2d {
            let grad = self.params[entry.idx].grad();
            if !grad.defined() {
                continue;
            }

            // EMA momentum: M = β1*M + (1-β1)*g (in place).
            let _ = entry.momentum.lerp_(&grad, 1.0 - beta1);

            // NS5 on the [p, q] orientation with p <= q, then rotate back.
            let ns_input = if entry.transposed {
                entry.momentum.transpose(0, 1)
            } else {
                entry.momentum.shallow_clone()
            };
            let u = newtonschulz5(&ns_input);
            let mut o = if entry.transposed {
                u.transpose(0, 1).contiguous()
            } else {
                u
            };

            // Row second-moment update.
            let sq_mean = o.square().mean_dim([-1i64].as_slice(), false, o.kind());
            let _ = entry.row_sq_mean.lerp_(&sq_mean, 1.0 - beta2);

            // Row-normalize, then RMS-match the Frobenius norm to produce a
            // shape-invariant effective step of ~0.2*lr per entry. We own
            // `o`, so all subsequent scalings run in place.
            let denom = (entry.row_sq_mean.sqrt() + eps).unsqueeze(-1);
            let _ = o.g_div_(&denom);
            let frob = o
                .frobenius_norm([-2i64, -1].as_slice(), true)
                .clamp_min(1e-12);
            let neg_eta = (neg_eta_coef * entry.sqrt_mn) / frob;
            let _ = o.g_mul_(&neg_eta);

            // Apply to param.
            let mut p = self.params[entry.idx].shallow_clone();
            if let Some(k) = wd_factor {
                let _ = p.g_mul_scalar_(k);
            }
            let _ = p.g_add_(&o);
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

    pub fn set_adamw_lr(&mut self, lr: f64) {
        self.cfg.adamw_lr = lr;
    }

    /// Total bytes of optimizer state currently allocated.
    /// 2D params: `momentum` + `row_sq_mean` per param.
    /// 1D params: AdamW `m` + `v` per param (lazy — zero until first step).
    pub fn state_bytes(&self) -> usize {
        let tensor_bytes = |t: &Tensor| t.numel() * t.kind().elt_size_in_bytes();
        let muon: usize = self
            .entries_2d
            .iter()
            .map(|e| tensor_bytes(&e.momentum) + tensor_bytes(&e.row_sq_mean))
            .sum();
        let adamw: usize = self
            .adamw_state
            .values()
            .map(|s| tensor_bytes(&s.m) + tensor_bytes(&s.v))
            .sum();
        muon + adamw
    }
}
