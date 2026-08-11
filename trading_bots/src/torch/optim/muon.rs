//! Reference-aligned NorMuon optimizer: EMA momentum/Nesterov, Newton-Schulz 5
//! orthogonalization, per-row second-moment (NorMuon) rescaling with Frobenius-
//! norm preservation, and AdamW for non-matrix params.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{ensure, Context, Result};
use tch::{Device, Kind, Tensor};

const NS_A: f64 = 3.4445;
const NS_B: f64 = -4.7750;
const NS_C: f64 = 2.0315;
/// Canonical Newton-Schulz iteration count; the reference default and the only
/// value real training should ever use.
pub const DEFAULT_NS_STEPS: usize = 5;

pub(crate) fn newton_schulz_polynomial_bits() -> [u64; 3] {
    [NS_A.to_bits(), NS_B.to_bits(), NS_C.to_bits()]
}

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
    /// Parameter name fragments excluded from AdamW's decoupled weight decay.
    pub adamw_no_weight_decay_name_substrings: Vec<String>,
    /// Newton-Schulz iteration count for orthogonalization. Reference default 5.
    /// Exposed only so offline sweeps can map the NS-steps landscape; real
    /// training must leave this at `DEFAULT_NS_STEPS`.
    pub ns_steps: usize,
    /// Parameter name fragments that should use AdamW even if they are 2D.
    pub force_adamw_name_substrings: Vec<String>,
    /// Optional allowlist for Muon-routed 2D parameters. Empty permits every
    /// otherwise-eligible matrix; the AdamW blocklist always takes precedence.
    pub muon_name_allowlist: Vec<String>,
    /// Benchmark/experiment mode: split attention projection matrices into
    /// per-head 2D blocks before Newton-Schulz orthogonalization.
    pub per_attention_head_ortho: bool,
    /// Include attention output projections in `per_attention_head_ortho`.
    pub per_attention_output_head_ortho: bool,
    /// Self-attention head width used by `per_attention_head_ortho`.
    pub attention_head_dim: i64,
    /// Cross-attention head width used by `per_attention_head_ortho`.
    pub cross_attention_head_dim: i64,
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
            adamw_no_weight_decay_name_substrings: Vec::new(),
            ns_steps: DEFAULT_NS_STEPS,
            force_adamw_name_substrings: Vec::new(),
            muon_name_allowlist: Vec::new(),
            per_attention_head_ortho: false,
            per_attention_output_head_ortho: true,
            attention_head_dim: 0,
            cross_attention_head_dim: 0,
            quiet: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrthoLayout {
    Matrix,
    RowHeads { heads: i64, head_dim: i64 },
    ColHeads { heads: i64, head_dim: i64 },
}

/// Per-2D-param state. Each 2D param holds its own momentum.
/// We deliberately do *not* stack same-shape params into a batched state tensor:
/// the N× VRAM multiplier from batching NS5's intermediates dwarfs any step-time
/// savings on realistic models. Keeping state per-param caps NS5's peak working
/// set at a single [m, n] matrix's worth of transients.
struct Entry2D {
    idx: usize,
    layout: OrthoLayout,
    /// First-moment EMA buffer, shape [m, n].
    momentum: Tensor,
    /// NorMuon second-moment EMA buffer (mean-of-squares per row). Matrix layout
    /// stores [m, 1]; per-head layouts store [heads, block_rows, 1].
    /// Kept in fp32 regardless of param dtype: an EMA at gain (1-beta2)=0.05 in
    /// bf16 silently stalls because small increments round to zero.
    second_momentum: Tensor,
}

struct AdamWParamState {
    m: Tensor,
    v: Tensor,
    step_count: i64,
}

pub struct Muon {
    cfg: MuonConfig,
    entries_2d: Vec<Entry2D>,
    adamw_indices: Vec<usize>,
    adamw_state: HashMap<usize, AdamWParamState>,
    step_count: i64,
    params: Vec<Tensor>,
    /// Variable name per param index; keys optimizer-state sidecar tensors so
    /// restoration is robust to param ordering. Empty for the unnamed `new` path.
    names: Vec<String>,
    /// Per-parameter multiplier over the routed optimizer's base learning rate.
    /// These are runtime schedule values, not optimizer moments; callers restore
    /// them from their training-controller state after loading a checkpoint.
    lr_scales: Vec<f64>,
    /// Runtime parameter-step mask. Disabled parameters retain both their value
    /// and optimizer moments, even when an earlier backward left a defined zero
    /// gradient in their slot.
    step_enabled: Vec<bool>,
}

/// Single-matrix Newton-Schulz iteration.
/// Runs NS_STEPS iterations in bf16 for speed; returns in the input's kind.
///
/// Each iteration is:  x ← NS_A·x + NS_B·(a·x) + NS_C·((a·a)·x), where a = x·xᵀ.
/// Factoring the two correction terms over the shared right-multiply by `x`,
///   NS_B·(a·x) + NS_C·((a·a)·x) = (NS_B·a + NS_C·(a·a))·x = b·x,
/// collapses the iteration to **three** matmuls (x·xᵀ, a·a, b·x) instead of the
/// four a naïve expansion needs, expressed as two `baddbmm` calls on [1,p,q]
/// views. tch's 2D `addmm` doesn't accept scalar beta/alpha, so we lift to 3D.
///
/// Peak live tensors during iter: `x` ([p,q]) + `a` ([p,p]) + `b` ([p,p]) =
/// ~[p,q] + 2·[p,p]. For p == q this is ~3·[p,q]. This is the inherent
/// working-set cost of NS5 and cannot be eliminated without changing the
/// algorithm.
fn newtonschulz5(g: &Tensor, ns_steps: usize) -> Tensor {
    let orig_kind = g.kind();
    let transposed = g.size()[0] > g.size()[1];
    let x2d = if orig_kind == Kind::BFloat16 {
        g.shallow_clone()
    } else {
        g.to_kind(Kind::BFloat16)
    };
    let nrm = x2d.norm().clamp_min(1e-7);
    let x2d = &x2d / &nrm;
    let x2d = if transposed { x2d.transpose(0, 1) } else { x2d };
    let mut x = x2d.unsqueeze(0); // [1, p, q] view — baddbmm needs 3D

    for _ in 0..ns_steps {
        let a = x.matmul(&x.transpose(-2, -1));
        // b = NS_B·a + NS_C·(a·a)  — fold both corrections into one matrix.
        let b = a.baddbmm(&a, &a, NS_B, NS_C);
        // x = NS_A·x + b·x
        x = x.baddbmm(&b, &x, NS_A, 1.0);
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

fn batched_newtonschulz5(g: &Tensor, ns_steps: usize) -> Tensor {
    let orig_kind = g.kind();
    let transposed = g.size()[1] > g.size()[2];
    let x3d = if orig_kind == Kind::BFloat16 {
        g.shallow_clone()
    } else {
        g.to_kind(Kind::BFloat16)
    };
    let nrm = x3d
        .square()
        .sum_dim_intlist([-2i64, -1].as_slice(), true, Kind::BFloat16)
        .sqrt()
        .clamp_min(1e-7);
    let x3d = &x3d / &nrm;
    let mut x = if transposed {
        x3d.transpose(-2, -1).contiguous()
    } else {
        x3d
    };

    for _ in 0..ns_steps {
        let a = x.matmul(&x.transpose(-2, -1));
        let b = a.baddbmm(&a, &a, NS_B, NS_C);
        x = x.baddbmm(&b, &x, NS_A, 1.0);
    }

    let x = if transposed {
        x.transpose(-2, -1).contiguous()
    } else {
        x
    };
    if orig_kind == Kind::BFloat16 {
        x
    } else {
        x.to_kind(orig_kind)
    }
}

fn attention_ortho_layout(name: &str, size: &[i64], cfg: &MuonConfig) -> OrthoLayout {
    if !cfg.per_attention_head_ortho || size.len() != 2 {
        return OrthoLayout::Matrix;
    }
    let Some(head_dim) = attention_head_dim_for_name(name, cfg) else {
        return OrthoLayout::Matrix;
    };

    let rows = size[0];
    let cols = size[1];
    let is_output = is_attention_output_projection_name(name);
    if is_output && !cfg.per_attention_output_head_ortho {
        OrthoLayout::Matrix
    } else if is_output && cols % head_dim == 0 {
        OrthoLayout::ColHeads {
            heads: cols / head_dim,
            head_dim,
        }
    } else if name.contains("attn_qkv") && rows == 3 * cols && rows % (3 * head_dim) == 0 {
        OrthoLayout::RowHeads {
            heads: rows / (3 * head_dim),
            head_dim: 3 * head_dim,
        }
    } else if rows % head_dim == 0 {
        OrthoLayout::RowHeads {
            heads: rows / head_dim,
            head_dim,
        }
    } else {
        OrthoLayout::Matrix
    }
}

fn attention_head_dim_for_name(name: &str, cfg: &MuonConfig) -> Option<i64> {
    if is_cross_attention_projection_name(name) {
        (cfg.cross_attention_head_dim > 0).then_some(cfg.cross_attention_head_dim)
    } else if is_self_attention_projection_name(name) {
        (cfg.attention_head_dim > 0).then_some(cfg.attention_head_dim)
    } else {
        None
    }
}

fn is_self_attention_projection_name(name: &str) -> bool {
    ["attn_q", "attn_k", "attn_v", "attn_qkv", "attn_o"]
        .iter()
        .any(|needle| name.contains(needle))
}

fn is_cross_attention_projection_name(name: &str) -> bool {
    ["ca_q", "ca_k", "ca_v", "ca_out"]
        .iter()
        .any(|needle| name.contains(needle))
}

fn is_attention_output_projection_name(name: &str) -> bool {
    ["attn_o", "ca_out"]
        .iter()
        .any(|needle| name.contains(needle))
}

#[cfg(test)]
fn orthogonalize_update(update: &Tensor, layout: OrthoLayout, ns_steps: usize) -> Tensor {
    match layout {
        OrthoLayout::Matrix => newtonschulz5(update, ns_steps),
        OrthoLayout::RowHeads { heads, head_dim } => {
            let cols = update.size()[1];
            batched_newtonschulz5(&update.reshape([heads, head_dim, cols]), ns_steps)
                .reshape(update.size().as_slice())
        }
        OrthoLayout::ColHeads { heads, head_dim } => {
            let rows = update.size()[0];
            batched_newtonschulz5(
                &update
                    .reshape([rows, heads, head_dim])
                    .permute([1, 0, 2])
                    .contiguous(),
                ns_steps,
            )
            .permute([1, 0, 2])
            .contiguous()
            .reshape(update.size().as_slice())
        }
    }
}

fn second_momentum_shape(size: &[i64], layout: OrthoLayout) -> Vec<i64> {
    match layout {
        OrthoLayout::Matrix => vec![size[0], 1],
        OrthoLayout::RowHeads { heads, head_dim } => vec![heads, head_dim, 1],
        OrthoLayout::ColHeads { heads, .. } => vec![heads, size[0], 1],
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
    let vnorm_new_sq =
        (step_size.square() * &row_sq_sum).sum_dim_intlist([-2i64].as_slice(), true, Kind::Float);
    // Frobenius-preservation ratio: [1, 1].
    let ratio = vnorm_sq.sqrt() / (vnorm_new_sq.sqrt() + 1e-10);
    // Fused per-row scale, cast back to update kind: [rows, 1].
    let scale = (&step_size * &ratio).to_kind(update.kind());
    update * &scale
}

fn normuon_rescale_batched(update: &Tensor, second_momentum: &mut Tensor, beta2: f64) -> Tensor {
    let uf = update.to_kind(Kind::Float);
    let cols = update.size()[2] as f64;
    let row_sq_sum = uf
        .square()
        .sum_dim_intlist([-1i64].as_slice(), true, Kind::Float);
    let v_mean = &row_sq_sum / cols;
    let vnorm_sq = row_sq_sum.sum_dim_intlist([-2i64].as_slice(), true, Kind::Float);
    let _ = second_momentum.lerp_(&v_mean, 1.0 - beta2);
    let step_size = (second_momentum.sqrt() + 1e-10).reciprocal();
    let vnorm_new_sq =
        (step_size.square() * &row_sq_sum).sum_dim_intlist([-2i64].as_slice(), true, Kind::Float);
    let ratio = vnorm_sq.sqrt() / (vnorm_new_sq.sqrt() + 1e-10);
    let scale = (&step_size * &ratio).to_kind(update.kind());
    update * &scale
}

fn normuon_transform(
    update: &Tensor,
    layout: OrthoLayout,
    second_momentum: &mut Tensor,
    beta2: f64,
    ns_steps: usize,
) -> (Tensor, f64) {
    match layout {
        OrthoLayout::Matrix => {
            let update = newtonschulz5(update, ns_steps);
            let update = normuon_rescale(&update, second_momentum, beta2);
            let size = update.size();
            let aspect_scale = (1.0_f64).max(size[0] as f64 / size[1] as f64).sqrt();
            (update, aspect_scale)
        }
        OrthoLayout::RowHeads { heads, head_dim } => {
            let cols = update.size()[1];
            let blocks = update.reshape([heads, head_dim, cols]);
            let blocks = batched_newtonschulz5(&blocks, ns_steps);
            let blocks = normuon_rescale_batched(&blocks, second_momentum, beta2);
            let aspect_scale = (1.0_f64).max(head_dim as f64 / cols as f64).sqrt();
            (blocks.reshape(update.size().as_slice()), aspect_scale)
        }
        OrthoLayout::ColHeads { heads, head_dim } => {
            let rows = update.size()[0];
            let blocks = update
                .reshape([rows, heads, head_dim])
                .permute([1, 0, 2])
                .contiguous();
            let blocks = batched_newtonschulz5(&blocks, ns_steps);
            let blocks = normuon_rescale_batched(&blocks, second_momentum, beta2);
            let aspect_scale = (1.0_f64).max(rows as f64 / head_dim as f64).sqrt();
            let update = blocks
                .permute([1, 0, 2])
                .contiguous()
                .reshape(update.size().as_slice());
            (update, aspect_scale)
        }
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
        let names: Vec<String> = trainable_vars
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        let mut entries_2d = Vec::new();
        let mut adamw_indices = Vec::new();

        for (i, (name, p)) in trainable_vars.iter().enumerate() {
            let force_adamw = cfg
                .force_adamw_name_substrings
                .iter()
                .any(|needle| name.contains(needle));
            let allowed = cfg.muon_name_allowlist.is_empty()
                || cfg
                    .muon_name_allowlist
                    .iter()
                    .any(|needle| name.contains(needle));
            if cfg.use_muon_for_2d && p.dim() == 2 && !force_adamw && allowed {
                let size = p.size();
                let (m, n) = (size[0], size[1]);
                let kind = p.kind();
                let device = p.device();
                let layout = attention_ortho_layout(name, &size, &cfg);
                entries_2d.push(Entry2D {
                    idx: i,
                    layout,
                    momentum: Tensor::zeros([m, n], (kind, device)),
                    second_momentum: Tensor::zeros(
                        second_momentum_shape(&size, layout).as_slice(),
                        (Kind::Float, device),
                    ),
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
                let head_ortho = entries_2d
                    .iter()
                    .filter(|entry| entry.layout != OrthoLayout::Matrix)
                    .count();
                if head_ortho > 0 {
                    println!(
                        "  attention-head ortho: {} params split into batched NS blocks (self_head_dim={}, cross_head_dim={})",
                        head_ortho, cfg.attention_head_dim, cfg.cross_attention_head_dim
                    );
                }
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
            lr_scales: vec![1.0; names.len()],
            step_enabled: vec![true; names.len()],
            names,
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
        let wd = self.cfg.weight_decay;

        for entry in &mut self.entries_2d {
            if !self.step_enabled[entry.idx] {
                continue;
            }
            let grad = self.params[entry.idx].grad();
            if !grad.defined() {
                continue;
            }
            let lr = self.cfg.lr * self.lr_scales[entry.idx];
            let wd_factor = if wd > 0.0 { Some(1.0 - lr * wd) } else { None };

            // First-moment EMA: buf = buf*beta1 + grad*(1-beta1).
            let _ = entry.momentum.lerp_(&grad, 1.0 - beta1);

            // Nesterov combine: update = grad*(1-beta1) + momentum*beta1.
            let update = if nesterov {
                grad.lerp(&entry.momentum, beta1)
            } else {
                entry.momentum.shallow_clone()
            };

            let (update, aspect_scale) = normuon_transform(
                &update,
                entry.layout,
                &mut entry.second_momentum,
                beta2,
                self.cfg.ns_steps,
            );

            // Apply to param: decoupled weight decay, then the update.
            let mut p = self.params[entry.idx].shallow_clone();
            if let Some(k) = wd_factor {
                let _ = p.g_mul_scalar_(k);
            }
            let update = update.to_kind(p.kind()) * (-lr * aspect_scale);
            let _ = p.g_add_(&update);
        }
    }

    fn step_all_adamw(&mut self) {
        let (beta1, beta2) = self.cfg.adamw_betas;
        let eps = self.cfg.adamw_eps;
        let wd = self.cfg.adamw_wd;

        for &idx in &self.adamw_indices {
            if !self.step_enabled[idx] {
                continue;
            }
            let mut p = self.params[idx].shallow_clone();
            let grad = p.grad();
            if !grad.defined() {
                continue;
            }
            let lr = self.cfg.adamw_lr * self.lr_scales[idx];

            let state = self
                .adamw_state
                .entry(idx)
                .or_insert_with(|| AdamWParamState {
                    m: Tensor::zeros_like(&grad),
                    v: Tensor::zeros_like(&grad),
                    step_count: 0,
                });
            state.step_count += 1;
            let bc1 = 1.0 - beta1.powi(state.step_count as i32);
            let bc2 = 1.0 - beta2.powi(state.step_count as i32);
            let step_size = -lr / bc1;
            let inv_bc2_sqrt = 1.0 / bc2.sqrt();

            let apply_weight_decay = wd > 0.0
                && !self
                    .cfg
                    .adamw_no_weight_decay_name_substrings
                    .iter()
                    .any(|needle| self.names[idx].contains(needle));
            if apply_weight_decay {
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

    /// Set a learning-rate multiplier for every named parameter matching at
    /// least one substring. Returns the number of matched parameters so callers
    /// can reject stale routing names instead of silently disabling a schedule.
    pub fn set_named_lr_scale(&mut self, name_substrings: &[&str], lr_scale: f64) -> usize {
        assert!(lr_scale.is_finite() && lr_scale > 0.0);
        let mut matched = 0;
        for (index, name) in self.names.iter().enumerate() {
            if name_substrings
                .iter()
                .any(|substring| name.contains(substring))
            {
                self.lr_scales[index] = lr_scale;
                matched += 1;
            }
        }
        matched
    }

    /// Enable or disable optimizer steps for matching named parameters. This is
    /// stronger than zeroing gradients: disabled parameters also skip momentum,
    /// second-moment, and weight-decay updates.
    pub fn set_named_step_enabled(&mut self, name_substrings: &[&str], enabled: bool) -> usize {
        let mut matched = 0;
        for (index, name) in self.names.iter().enumerate() {
            if name_substrings
                .iter()
                .any(|substring| name.contains(substring))
            {
                self.step_enabled[index] = enabled;
                matched += 1;
            }
        }
        matched
    }

    /// Serialize all optimizer state (per-param momentum/second-momentum for the
    /// NorMuon 2D params, AdamW m/v for the rest, and the global step counter) to
    /// a named-tensor sidecar. Keying by variable name makes restoration robust to
    /// param ordering. Requires `new_named` (the unnamed path stores empty names
    /// and its keys would collide).
    pub fn save_state(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let mut named: Vec<(String, Tensor)> = Vec::new();
        // Suffix separator is `.` because tch's save/load round-trips `.`<->`|`
        // internally; any other separator would not survive the round trip.
        for entry in &self.entries_2d {
            let name = &self.names[entry.idx];
            named.push((
                format!("{name}.__momentum"),
                entry.momentum.to_device(Device::Cpu),
            ));
            named.push((
                format!("{name}.__second_momentum"),
                entry.second_momentum.to_device(Device::Cpu),
            ));
        }
        for (&idx, state) in &self.adamw_state {
            let name = &self.names[idx];
            named.push((format!("{name}.__adamw_m"), state.m.to_device(Device::Cpu)));
            named.push((format!("{name}.__adamw_v"), state.v.to_device(Device::Cpu)));
            named.push((
                format!("{name}.__adamw_step_count"),
                Tensor::from(state.step_count),
            ));
        }
        named.push((
            "__muon_step_count__".to_owned(),
            Tensor::from(self.step_count),
        ));
        let refs: Vec<(&str, &Tensor)> = named.iter().map(|(n, t)| (n.as_str(), t)).collect();
        Tensor::save_multi(&refs, path)
            .with_context(|| format!("failed saving optimizer state {}", path.display()))
    }

    /// Restore optimizer state saved by [`Muon::save_state`], copying buffers in
    /// place so device/dtype match the live params. Absent 2D buffers are an
    /// error (the checkpoint is incomplete); absent AdamW buffers leave that param
    /// lazily re-initialized on its next step.
    pub fn load_state(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let device = self
            .params
            .first()
            .map(|p| p.device())
            .unwrap_or(Device::Cpu);
        let loaded: HashMap<String, Tensor> = Tensor::load_multi_with_device(path, device)
            .with_context(|| format!("failed loading optimizer state {}", path.display()))?
            .into_iter()
            .collect();
        let global_step_count = loaded
            .get("__muon_step_count__")
            .map(|t| t.int64_value(&[]))
            .unwrap_or(0);
        tch::no_grad(|| -> Result<()> {
            for entry in &mut self.entries_2d {
                let name = &self.names[entry.idx];
                let momentum = loaded
                    .get(&format!("{name}.__momentum"))
                    .with_context(|| format!("optimizer state missing momentum for {name}"))?;
                let second = loaded
                    .get(&format!("{name}.__second_momentum"))
                    .with_context(|| {
                        format!("optimizer state missing second_momentum for {name}")
                    })?;
                entry.momentum.copy_(momentum);
                entry.second_momentum.copy_(second);
            }
            self.adamw_state.clear();
            for &idx in &self.adamw_indices {
                let name = &self.names[idx];
                if let (Some(m), Some(v)) = (
                    loaded.get(&format!("{name}.__adamw_m")),
                    loaded.get(&format!("{name}.__adamw_v")),
                ) {
                    self.adamw_state.insert(
                        idx,
                        AdamWParamState {
                            m: m.shallow_clone(),
                            v: v.shallow_clone(),
                            step_count: loaded
                                .get(&format!("{name}.__adamw_step_count"))
                                .map(|step| step.int64_value(&[]))
                                .unwrap_or(global_step_count),
                        },
                    );
                }
            }
            Ok(())
        })?;
        self.step_count = global_step_count;
        Ok(())
    }

    /// Names of AdamW parameters whose lazy moments have been initialized.
    /// Persisting this set distinguishes a legitimate never-stepped parameter
    /// from a truncated resume sidecar.
    pub fn initialized_adamw_names(&self) -> Vec<String> {
        let mut names = self
            .adamw_state
            .keys()
            .map(|&idx| self.names[idx].clone())
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    /// Validate the complete optimizer tensor schema against a freshly
    /// constructed optimizer without mutating any live parameter or state.
    pub fn validate_state_strict(
        &self,
        path: impl AsRef<Path>,
        expected_initialized_adamw: &[String],
        expected_step: i64,
    ) -> Result<()> {
        let path = path.as_ref();
        let loaded: HashMap<String, Tensor> = Tensor::load_multi_with_device(path, Device::Cpu)
            .with_context(|| format!("failed loading optimizer state {}", path.display()))?
            .into_iter()
            .collect();
        let mut expected_keys = HashSet::new();
        expected_keys.insert("__muon_step_count__".to_owned());
        let global_step = loaded
            .get("__muon_step_count__")
            .context("optimizer state missing global step")?;
        ensure!(
            global_step.numel() == 1 && global_step.int64_value(&[]) == expected_step,
            "optimizer global step disagrees with checkpoint metadata"
        );

        for entry in &self.entries_2d {
            let name = &self.names[entry.idx];
            let momentum_name = format!("{name}.__momentum");
            let second_name = format!("{name}.__second_momentum");
            let momentum = loaded
                .get(&momentum_name)
                .with_context(|| format!("optimizer state missing momentum for {name}"))?;
            let second = loaded
                .get(&second_name)
                .with_context(|| format!("optimizer state missing second momentum for {name}"))?;
            ensure!(
                momentum.size() == entry.momentum.size()
                    && momentum.kind() == entry.momentum.kind(),
                "optimizer momentum schema mismatch for {name}"
            );
            ensure!(
                second.size() == entry.second_momentum.size()
                    && second.kind() == entry.second_momentum.kind(),
                "optimizer second-momentum schema mismatch for {name}"
            );
            expected_keys.insert(momentum_name);
            expected_keys.insert(second_name);
        }

        let initialized = expected_initialized_adamw
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        ensure!(
            initialized.len() == expected_initialized_adamw.len(),
            "initialized AdamW names are not unique"
        );
        for name in &initialized {
            let idx = self
                .names
                .iter()
                .position(|candidate| candidate == name)
                .with_context(|| format!("optimizer checkpoint names unknown parameter {name}"))?;
            ensure!(
                self.adamw_indices.contains(&idx),
                "optimizer checkpoint routes non-AdamW parameter {name} through AdamW"
            );
            let m_name = format!("{name}.__adamw_m");
            let v_name = format!("{name}.__adamw_v");
            let step_name = format!("{name}.__adamw_step_count");
            let m = loaded
                .get(&m_name)
                .with_context(|| format!("optimizer state missing AdamW m for {name}"))?;
            let v = loaded
                .get(&v_name)
                .with_context(|| format!("optimizer state missing AdamW v for {name}"))?;
            let step = loaded
                .get(&step_name)
                .with_context(|| format!("optimizer state missing AdamW step for {name}"))?;
            ensure!(
                m.size() == self.params[idx].size() && v.size() == self.params[idx].size(),
                "optimizer AdamW moment shape mismatch for {name}"
            );
            ensure!(
                m.kind() == self.params[idx].kind() && v.kind() == self.params[idx].kind(),
                "optimizer AdamW moment dtype mismatch for {name}"
            );
            ensure!(
                step.numel() == 1,
                "optimizer AdamW step is not scalar for {name}"
            );
            expected_keys.extend([m_name, v_name, step_name]);
        }

        let actual_keys = loaded.keys().cloned().collect::<HashSet<_>>();
        ensure!(
            actual_keys == expected_keys,
            "optimizer tensor schema differs from the current model: missing={:?}, unexpected={:?}",
            expected_keys.difference(&actual_keys).collect::<Vec<_>>(),
            actual_keys.difference(&expected_keys).collect::<Vec<_>>()
        );
        ensure!(
            loaded.values().all(|tensor| {
                !tensor.is_floating_point() || tensor.isfinite().all().int64_value(&[]) != 0
            }),
            "optimizer state contains non-finite tensors"
        );
        Ok(())
    }

    pub fn load_state_strict(
        &mut self,
        path: impl AsRef<Path>,
        expected_initialized_adamw: &[String],
    ) -> Result<()> {
        self.load_state(path)?;
        let actual = self.initialized_adamw_names();
        anyhow::ensure!(
            actual == expected_initialized_adamw,
            "optimizer AdamW state is incomplete: expected {:?}, restored {:?}",
            expected_initialized_adamw,
            actual
        );
        Ok(())
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

    use super::{
        attention_ortho_layout, batched_newtonschulz5, newtonschulz5, normuon_rescale,
        orthogonalize_update, Muon, MuonConfig, OrthoLayout,
    };

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
    fn batched_newtonschulz_matches_independent_single_matrix_path() {
        let _g = tch::no_grad_guard();
        tch::manual_seed(17);
        let device = Device::Cpu;
        let heads = 4;
        let update = Tensor::randn([heads, 32, 96], (Kind::Float, device));
        let batched = batched_newtonschulz5(&update, 5);
        let independent: Vec<Tensor> = (0..heads)
            .map(|head| newtonschulz5(&update.get(head), 5))
            .collect();
        let independent = Tensor::stack(&independent, 0);
        let max_diff = (&batched - independent).abs().max().double_value(&[]);
        assert!(
            max_diff < 3e-2,
            "batched NS diverged from independent NS: max diff={max_diff:.3e}"
        );
    }

    #[test]
    fn attention_head_ortho_preserves_original_matrix_shape() {
        let _g = tch::no_grad_guard();
        tch::manual_seed(23);
        let device = Device::Cpu;
        let row_split = Tensor::randn([256, 256], (Kind::Float, device));
        let col_split = Tensor::randn([256, 256], (Kind::Float, device));

        let row_out = orthogonalize_update(
            &row_split,
            OrthoLayout::RowHeads {
                heads: 4,
                head_dim: 64,
            },
            5,
        );
        let col_out = orthogonalize_update(
            &col_split,
            OrthoLayout::ColHeads {
                heads: 4,
                head_dim: 64,
            },
            5,
        );

        assert_eq!(row_out.size(), row_split.size());
        assert_eq!(col_out.size(), col_split.size());
        assert!(row_out.isfinite().all().int64_value(&[]) != 0);
        assert!(col_out.isfinite().all().int64_value(&[]) != 0);
    }

    #[test]
    fn cross_attention_requires_explicit_cross_head_dim() {
        let self_only = MuonConfig {
            per_attention_head_ortho: true,
            attention_head_dim: 64,
            ..MuonConfig::default()
        };
        assert_eq!(
            attention_ortho_layout("cross_attn.ca_q", &[256, 256], &self_only),
            OrthoLayout::Matrix
        );

        let with_cross = MuonConfig {
            cross_attention_head_dim: 128,
            ..self_only
        };
        assert_eq!(
            attention_ortho_layout("cross_attn.ca_q", &[256, 256], &with_cross),
            OrthoLayout::RowHeads {
                heads: 2,
                head_dim: 128
            }
        );
    }

    #[test]
    fn muon_name_allowlist_routes_only_matching_matrices() {
        let vars = vec![
            (
                "flow.fc1.weight".to_owned(),
                Tensor::zeros([8, 8], (Kind::Float, Device::Cpu)).set_requires_grad(true),
            ),
            (
                "flow.mod.weight".to_owned(),
                Tensor::zeros([8, 8], (Kind::Float, Device::Cpu)).set_requires_grad(true),
            ),
        ];
        let opt = Muon::new_named(
            &vars,
            MuonConfig {
                muon_name_allowlist: vec!["flow.fc".to_owned()],
                ..MuonConfig::default()
            },
        );
        assert_eq!(opt.entries_2d.len(), 1);
        assert_eq!(opt.adamw_indices.len(), 1);
        assert_eq!(opt.entries_2d[0].idx, 0);
        assert_eq!(opt.adamw_indices[0], 1);
    }

    #[test]
    fn named_lr_scale_changes_only_matching_parameter_updates() {
        let actor = Tensor::zeros([1], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let critic = Tensor::zeros([1], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let shared = Tensor::zeros([1], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let vars = vec![
            (
                "policy_concentration.bias".to_owned(),
                actor.shallow_clone(),
            ),
            ("value_projection.bias".to_owned(), critic.shallow_clone()),
            ("trunk_norm.weight".to_owned(), shared.shallow_clone()),
        ];
        let mut opt = Muon::new_named(
            &vars,
            MuonConfig {
                use_muon_for_2d: false,
                adamw_lr: 0.1,
                adamw_betas: (0.0, 0.0),
                adamw_eps: 0.0,
                quiet: true,
                ..MuonConfig::default()
            },
        );

        assert_eq!(opt.set_named_lr_scale(&["policy_concentration"], 0.25), 1);
        (&actor + &critic + &shared).sum(Kind::Float).backward();
        opt.step();

        assert!((actor.double_value(&[]) + 0.025).abs() < 1e-7);
        assert!((critic.double_value(&[]) + 0.1).abs() < 1e-7);
        assert!((shared.double_value(&[]) + 0.1).abs() < 1e-7);
    }

    #[test]
    fn disabled_named_parameters_skip_state_and_retain_their_adamw_clock() {
        let actor = Tensor::zeros([1], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let critic = Tensor::zeros([1], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let vars = vec![
            (
                "policy_concentration.bias".to_owned(),
                actor.shallow_clone(),
            ),
            ("value_projection.bias".to_owned(), critic.shallow_clone()),
        ];
        let mut opt = Muon::new_named(
            &vars,
            MuonConfig {
                use_muon_for_2d: false,
                adamw_lr: 0.1,
                quiet: true,
                ..MuonConfig::default()
            },
        );

        (&actor + &critic).sum(Kind::Float).backward();
        opt.step();
        opt.zero_grad();
        let actor_before_critic_only_step = actor.double_value(&[]);

        assert_eq!(
            opt.set_named_step_enabled(&["policy_concentration"], false),
            1
        );
        critic.sum(Kind::Float).backward();
        opt.step();

        assert_eq!(actor.double_value(&[]), actor_before_critic_only_step);
        assert!(critic.double_value(&[]) < actor_before_critic_only_step);
        assert_eq!(opt.adamw_state[&0].step_count, 1);
        assert_eq!(opt.adamw_state[&1].step_count, 2);

        let state_path = std::env::temp_dir().join(format!(
            "muon-per-param-clock-{}-{}.ot",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        opt.save_state(&state_path).unwrap();
        let restored_actor = Tensor::zeros([1], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let restored_critic =
            Tensor::zeros([1], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let mut restored = Muon::new_named(
            &[
                (
                    "policy_concentration.bias".to_owned(),
                    restored_actor.shallow_clone(),
                ),
                (
                    "value_projection.bias".to_owned(),
                    restored_critic.shallow_clone(),
                ),
            ],
            MuonConfig {
                use_muon_for_2d: false,
                quiet: true,
                ..MuonConfig::default()
            },
        );
        restored.load_state(&state_path).unwrap();
        assert_eq!(restored.adamw_state[&0].step_count, 1);
        assert_eq!(restored.adamw_state[&1].step_count, 2);
        std::fs::remove_file(state_path).unwrap();

        opt.zero_grad();
        opt.set_named_step_enabled(&["policy_concentration"], true);
        actor.sum(Kind::Float).backward();
        opt.step();
        assert_eq!(opt.adamw_state[&0].step_count, 2);
        assert!((actor.double_value(&[]) + 0.2).abs() < 1e-6);
    }

    #[test]
    fn adamw_named_no_weight_decay_excludes_only_matching_parameters() {
        let phase = Tensor::ones([4], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let regular = Tensor::ones([4], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let vars = vec![
            ("layer.pope_theta_bias".to_owned(), phase.shallow_clone()),
            ("layer.bias".to_owned(), regular.shallow_clone()),
        ];
        let mut opt = Muon::new_named(
            &vars,
            MuonConfig {
                use_muon_for_2d: false,
                adamw_lr: 0.1,
                adamw_wd: 0.5,
                adamw_no_weight_decay_name_substrings: vec!["pope_theta_bias".to_owned()],
                ..MuonConfig::default()
            },
        );
        (&phase.sum(Kind::Float) * 0.0 + &regular.sum(Kind::Float) * 0.0).backward();
        opt.step();
        assert_eq!(phase.min().double_value(&[]), 1.0);
        assert!((regular.max().double_value(&[]) - 0.95).abs() < 1e-6);
    }

    #[test]
    fn fused_qkv_groups_qkv_rows_by_attention_head() {
        let cfg = MuonConfig {
            per_attention_head_ortho: true,
            attention_head_dim: 64,
            ..MuonConfig::default()
        };
        assert_eq!(
            attention_ortho_layout("block0.attn_qkv", &[768, 256], &cfg),
            OrthoLayout::RowHeads {
                heads: 4,
                head_dim: 192
            }
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
