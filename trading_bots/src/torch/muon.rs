use std::collections::HashMap;
use tch::{Kind, Tensor};

const NS_A: f64 = 3.4445;
const NS_B: f64 = -4.7750;
const NS_C: f64 = 2.0315;
const NS_STEPS: usize = 5;

pub struct MuonConfig {
    pub lr: f64,
    pub momentum: f64,
    pub nesterov: bool,
    pub weight_decay: f64,
    /// AdamW LR for scalar/1D params (biases, norms, embeddings).
    pub adamw_lr: f64,
    pub adamw_betas: (f64, f64),
    pub adamw_eps: f64,
    pub adamw_wd: f64,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: 0.02,
            momentum: 0.95,
            nesterov: true,
            weight_decay: 0.0,
            adamw_lr: 3e-4,
            adamw_betas: (0.9, 0.999),
            adamw_eps: 1e-8,
            adamw_wd: 0.0,
        }
    }
}

struct MuonParamState {
    momentum_buf: Tensor,
}

struct AdamWParamState {
    m: Tensor,
    v: Tensor,
}

/// A group of same-shape 2D params for batched Newton-Schulz.
struct ShapeGroup {
    indices: Vec<usize>,
    transposed: bool,
    /// sqrt(max(1, orig_m / orig_n)) — constant per shape.
    dim_scale: f64,
    /// Precomputed: -lr * dim_scale.
    neg_lr_scale: f64,
}

pub struct Muon {
    cfg: MuonConfig,
    shape_groups: Vec<ShapeGroup>,
    adamw_indices: Vec<usize>,
    muon_state: HashMap<usize, MuonParamState>,
    adamw_state: HashMap<usize, AdamWParamState>,
    step_count: i64,
    params: Vec<Tensor>,
}

/// Batched Newton-Schulz iteration.
/// Input: [batch, m, n] with m <= n. Runs NS_STEPS iterations in bf16.
fn batched_newtonschulz5(g: &Tensor) -> Tensor {
    let orig_kind = g.kind();
    let g_bf16 = g.to_kind(Kind::BFloat16);
    let nrm = g_bf16
        .frobenius_norm([-2i64, -1].as_slice(), true)
        .clamp_min(1e-7);
    let mut x = &g_bf16 / &nrm;

    for _ in 0..NS_STEPS {
        let a = x.matmul(&x.transpose(-2, -1));
        x = (NS_C * a.matmul(&a) + NS_B * &a).matmul(&x) + NS_A * &x;
    }

    x.to_kind(orig_kind)
}

/// Single-matrix Newton-Schulz (for groups of size 1, avoids batch dim overhead).
fn newtonschulz5(g: &Tensor) -> Tensor {
    let orig_kind = g.kind();
    let g_bf16 = g.to_kind(Kind::BFloat16);
    let nrm = g_bf16
        .frobenius_norm([0i64, 1].as_slice(), false)
        .clamp_min(1e-7);
    let mut x = &g_bf16 / &nrm;

    for _ in 0..NS_STEPS {
        let a = x.matmul(&x.transpose(0, 1));
        x = (NS_C * a.matmul(&a) + NS_B * &a).matmul(&x) + NS_A * &x;
    }

    x.to_kind(orig_kind)
}

impl Muon {
    pub fn new(trainable_vars: &[Tensor], cfg: MuonConfig) -> Self {
        let params: Vec<Tensor> = trainable_vars.iter().map(|t| t.shallow_clone()).collect();
        let mut adamw_indices = Vec::new();

        let mut shape_map: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (i, p) in params.iter().enumerate() {
            if p.dim() == 2 {
                let size = p.size();
                shape_map.entry((size[0], size[1])).or_default().push(i);
            } else {
                adamw_indices.push(i);
            }
        }

        let mut muon_count = 0;
        let shape_groups: Vec<ShapeGroup> = shape_map
            .into_iter()
            .map(|((m, n), indices)| {
                muon_count += indices.len();
                let dim_scale = (1.0f64).max(m as f64 / n as f64).sqrt();
                ShapeGroup {
                    indices,
                    transposed: m > n,
                    dim_scale,
                    neg_lr_scale: -cfg.lr * dim_scale,
                }
            })
            .collect();

        println!(
            "Muon optimizer: {} 2D params (Muon, {} shape groups), {} other params (AdamW)",
            muon_count,
            shape_groups.len(),
            adamw_indices.len()
        );

        Self {
            cfg,
            shape_groups,
            adamw_indices,
            muon_state: HashMap::new(),
            adamw_state: HashMap::new(),
            step_count: 0,
            params,
        }
    }

    pub fn step(&mut self) {
        tch::no_grad(|| {
            self.step_count += 1;
            self.step_all_muon();
            self.step_all_adamw();
        });
    }

    fn step_all_muon(&mut self) {
        for gi in 0..self.shape_groups.len() {
            let group = &self.shape_groups[gi];
            let transposed = group.transposed;
            let neg_lr_scale = group.neg_lr_scale;
            let indices: Vec<usize> = group.indices.clone();

            // Collect momentum-processed gradients (only defined ones)
            let mut active: Vec<(usize, Tensor)> = Vec::new();
            for (local_i, &idx) in indices.iter().enumerate() {
                let p = &self.params[idx];
                let grad = p.grad();
                if !grad.defined() {
                    continue;
                }

                if self.cfg.weight_decay > 0.0 {
                    let mut p = p.shallow_clone();
                    let _ = p.g_mul_scalar_(1.0 - self.cfg.lr * self.cfg.weight_decay);
                }

                let g = if self.cfg.momentum > 0.0 {
                    let state = self
                        .muon_state
                        .entry(idx)
                        .or_insert_with(|| MuonParamState {
                            momentum_buf: Tensor::zeros_like(&grad),
                        });
                    let _ = state
                        .momentum_buf
                        .g_mul_scalar_(self.cfg.momentum)
                        .g_add_(&grad);
                    if self.cfg.nesterov {
                        &grad + self.cfg.momentum * &state.momentum_buf
                    } else {
                        state.momentum_buf.shallow_clone()
                    }
                } else {
                    grad.shallow_clone()
                };

                let g = if transposed { g.transpose(0, 1) } else { g };
                active.push((local_i, g));
            }
            if active.is_empty() {
                continue;
            }

            // Batched or single NS, then scale in-place (we own these tensors)
            let mut updates: Vec<Tensor> = if active.len() == 1 {
                let (_, ref g) = active[0];
                let mut u = newtonschulz5(g);
                if transposed {
                    u = u.transpose(0, 1);
                }
                let _ = u.g_mul_scalar_(neg_lr_scale);
                vec![u]
            } else {
                let batch: Vec<&Tensor> = active.iter().map(|(_, g)| g).collect();
                let stacked = Tensor::stack(&batch, 0);
                let u_batch = batched_newtonschulz5(&stacked);
                (0..active.len() as i64)
                    .map(|i| {
                        let mut u = u_batch.select(0, i);
                        if transposed {
                            u = u.transpose(0, 1);
                        }
                        let _ = u.g_mul_scalar_(neg_lr_scale);
                        u
                    })
                    .collect()
            };

            for (j, (local_i, _)) in active.into_iter().enumerate() {
                let idx = indices[local_i];
                let mut p = self.params[idx].shallow_clone();
                let _ = p.g_add_(&updates[j]);
            }
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

        for i in 0..self.adamw_indices.len() {
            let idx = self.adamw_indices[i];
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
            let grad_sq = grad.square();
            let _ = state.v.lerp_(&grad_sq, 1.0 - beta2);

            // Folded bias correction: denom = sqrt(v) / sqrt(bc2) + eps
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
        for group in &mut self.shape_groups {
            group.neg_lr_scale = -lr * group.dim_scale;
        }
    }

    pub fn set_adamw_lr(&mut self, lr: f64) {
        self.cfg.adamw_lr = lr;
    }
}
