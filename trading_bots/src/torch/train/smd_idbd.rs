//! Strictly opt-in pretraining optimizer ablations.
//!
//! `SmdIdbd` implements Schraudolph's stochastic meta-descent (SMD) for a nonlinear
//! objective. It is deliberately described as batch-step SMD-IDBD: the training loss is the
//! existing batch loss, not a sum of per-example meta-updates. For `d = -grad(loss)` and a
//! detached sensitivity trace `v`, every coordinate follows this order:
//!
//! `beta += mu * d * v; p = base_lr * exp(beta); w += p * d; v += p * (d - H v)`.
//!
//! `H v` is an exact Pearlmutter Hessian-vector product obtained from a gradient graph with
//! `Tensor::run_backward(..., create_graph = true)`. No empirical-Fisher or gradient-alignment
//! substitute exists here.

use anyhow::{ensure, Context, Result};
use clap::ValueEnum;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use tch::{no_grad, Device, Kind, Tensor};

use crate::torch::optim::muon::{Muon, StepKind};

pub const SMD_BETA_MIN: f64 = -10.0;
pub const SMD_BETA_MAX: f64 = 2.0;
pub const SMD_BETA_UPDATE_CLIP: f64 = 2.0;
pub const SMD_METRIC_COUNT: usize = 10;

/// Explicit pretraining optimizer experiment. Absence selects the production recipe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum PretrainOptimizerAblation {
    FixedSgd,
    SmdIdbd,
}

pub(crate) struct FixedSgd {
    params: Vec<(String, Tensor)>,
    lr: f64,
}

struct SmdParameter {
    name: String,
    parameter: Tensor,
    /// Per-coordinate log gain multiplier, always fp32.
    beta: Tensor,
    /// Schraudolph sensitivity trace, always fp32.
    v: Tensor,
    /// Exact instantaneous Hessian-vector product for the pending update, always fp32.
    pending_hv: Tensor,
}

pub(crate) struct SmdIdbd {
    params: Vec<SmdParameter>,
    base_lr: f64,
    meta_lr: f64,
    pending: bool,
    pending_metrics: Option<Tensor>,
}

/// Keeps the production path and both ablation arms disjoint. Constructing `Production`
/// allocates no SMD state and never builds a gradient graph for an HVP.
pub(crate) enum PretrainOptimizer {
    Production(Muon),
    FixedSgd(FixedSgd),
    SmdIdbd(SmdIdbd),
}

impl PretrainOptimizer {
    pub fn production(optimizer: Muon) -> Self {
        Self::Production(optimizer)
    }

    pub fn fixed_sgd(named: &[(String, Tensor)], lr: f64) -> Self {
        Self::FixedSgd(FixedSgd {
            params: clone_named(named),
            lr,
        })
    }

    pub fn smd_idbd(named: &[(String, Tensor)], lr: f64, meta_lr: f64) -> Self {
        let params = named
            .iter()
            .map(|(name, parameter)| SmdParameter {
                name: name.clone(),
                parameter: parameter.shallow_clone(),
                beta: Tensor::zeros(parameter.size(), (Kind::Float, parameter.device())),
                v: Tensor::zeros(parameter.size(), (Kind::Float, parameter.device())),
                pending_hv: Tensor::zeros(parameter.size(), (Kind::Float, parameter.device())),
            })
            .collect();
        Self::SmdIdbd(SmdIdbd {
            params,
            base_lr: lr,
            meta_lr,
            pending: false,
            pending_metrics: None,
        })
    }

    #[cfg(test)]
    pub fn production_ref(&self) -> Option<&Muon> {
        match self {
            Self::Production(optimizer) => Some(optimizer),
            Self::FixedSgd(_) | Self::SmdIdbd(_) => None,
        }
    }

    pub fn set_learning_rates(&mut self, normuon_lr: f64, adamw_lr: f64, ablation_lr: f64) {
        match self {
            Self::Production(optimizer) => {
                optimizer.set_lr(normuon_lr);
                optimizer.set_adamw_lr(adamw_lr);
            }
            Self::FixedSgd(optimizer) => optimizer.lr = ablation_lr,
            Self::SmdIdbd(optimizer) => optimizer.base_lr = ablation_lr,
        }
    }

    pub fn set_momentum(&mut self, momentum: f64) {
        if let Self::Production(optimizer) = self {
            optimizer.set_momentum(momentum);
        }
    }

    pub fn zero_grad(&mut self) {
        match self {
            Self::Production(optimizer) => optimizer.zero_grad(),
            Self::FixedSgd(optimizer) => zero_named(&mut optimizer.params),
            Self::SmdIdbd(optimizer) => {
                for entry in &mut optimizer.params {
                    entry.parameter.zero_grad();
                }
                optimizer.pending = false;
                optimizer.pending_metrics = None;
            }
        }
    }

    /// Builds gradients for the selected arm. Only SMD asks autograd for a gradient graph.
    pub fn backward(&mut self, loss: &Tensor) -> Result<()> {
        match self {
            Self::Production(_) | Self::FixedSgd(_) => {
                loss.backward();
                Ok(())
            }
            Self::SmdIdbd(optimizer) => optimizer.backward_exact_hvp(loss),
        }
    }

    pub fn step(&mut self, kind: StepKind) -> Result<()> {
        match self {
            Self::Production(optimizer) => {
                optimizer.step(kind);
                Ok(())
            }
            Self::FixedSgd(optimizer) => optimizer.step(),
            Self::SmdIdbd(optimizer) => optimizer.step(),
        }
    }

    pub fn row_learned_lr_metrics_tensor(&self) -> Option<Tensor> {
        match self {
            Self::Production(optimizer) => optimizer.row_learned_lr_metrics_tensor(),
            Self::FixedSgd(_) | Self::SmdIdbd(_) => None,
        }
    }

    /// Device-resident diagnostics for the pending SMD update. The caller concatenates this
    /// into the existing one-transfer step packet before mutating parameters.
    pub fn smd_metrics_tensor(&self) -> Option<Tensor> {
        match self {
            Self::SmdIdbd(optimizer) => optimizer
                .pending_metrics
                .as_ref()
                .map(Tensor::shallow_clone),
            Self::Production(_) | Self::FixedSgd(_) => None,
        }
    }

    /// Resident optimizer bytes. SMD accounts for beta, v and the pending exact HVP.
    pub fn state_bytes(&self) -> usize {
        match self {
            Self::Production(optimizer) => optimizer.state_bytes(),
            Self::FixedSgd(_) => 0,
            Self::SmdIdbd(optimizer) => optimizer.state_bytes(),
        }
    }

    pub fn steady_state_bytes(&self) -> usize {
        match self {
            Self::Production(optimizer) => optimizer.steady_state_bytes(),
            Self::FixedSgd(_) => 0,
            Self::SmdIdbd(optimizer) => optimizer.state_bytes(),
        }
    }
}

impl PretrainOptimizer {
    /// Names of lazily initialized production AdamW buffers. The other variants either carry
    /// their complete per-parameter state eagerly (SMD) or deliberately carry no state (SGD).
    pub fn initialized_adamw_names(&self) -> Vec<String> {
        match self {
            Self::Production(optimizer) => optimizer.initialized_adamw_names(),
            Self::FixedSgd(_) | Self::SmdIdbd(_) => Vec::new(),
        }
    }

    /// Persist the selected optimizer's complete recovery state as named tensors.
    pub fn save_state(&self, path: impl AsRef<Path>) -> Result<()> {
        match self {
            Self::Production(optimizer) => optimizer.save_state(path),
            Self::FixedSgd(_) => {
                let marker = Tensor::from(0i64);
                Tensor::save_multi(&[("__fixed_sgd_no_state__", &marker)], path.as_ref())
                    .with_context(|| {
                        format!(
                            "failed saving fixed-SGD optimizer marker {}",
                            path.as_ref().display()
                        )
                    })
            }
            Self::SmdIdbd(optimizer) => optimizer.save_state(path.as_ref()),
        }
    }

    /// Validate the complete named-tensor schema without mutating live optimizer state.
    pub fn validate_state_strict(
        &self,
        path: impl AsRef<Path>,
        expected_initialized_adamw: &[String],
        expected_primary_steps: i64,
    ) -> Result<()> {
        match self {
            Self::Production(optimizer) => optimizer.validate_state_strict(
                path,
                expected_initialized_adamw,
                expected_primary_steps,
            ),
            Self::FixedSgd(_) => {
                ensure!(
                    expected_initialized_adamw.is_empty(),
                    "fixed SGD cannot have initialized AdamW state"
                );
                validate_fixed_sgd_state(path.as_ref())
            }
            Self::SmdIdbd(optimizer) => {
                ensure!(
                    expected_initialized_adamw.is_empty(),
                    "SMD-IDBD cannot have initialized AdamW state"
                );
                optimizer.validate_state_strict(path.as_ref())
            }
        }
    }

    /// Restore state only after strict schema validation, copying into the live buffers.
    pub fn load_state_strict(
        &mut self,
        path: impl AsRef<Path>,
        expected_initialized_adamw: &[String],
        expected_primary_steps: i64,
    ) -> Result<()> {
        let path = path.as_ref();
        self.validate_state_strict(path, expected_initialized_adamw, expected_primary_steps)?;
        match self {
            Self::Production(optimizer) => {
                optimizer.load_state_strict(path, expected_initialized_adamw)
            }
            Self::FixedSgd(_) => Ok(()),
            Self::SmdIdbd(optimizer) => optimizer.load_state(path),
        }
    }
}

fn validate_fixed_sgd_state(path: &Path) -> Result<()> {
    let loaded: HashMap<String, Tensor> = Tensor::load_multi_with_device(path, Device::Cpu)
        .with_context(|| {
            format!(
                "failed loading fixed-SGD optimizer state {}",
                path.display()
            )
        })?
        .into_iter()
        .collect();
    let expected = HashSet::from(["__fixed_sgd_no_state__".to_owned()]);
    let actual = loaded.keys().cloned().collect::<HashSet<_>>();
    ensure!(
        actual == expected,
        "fixed-SGD optimizer tensor schema differs: missing={:?}, unexpected={:?}",
        expected.difference(&actual).collect::<Vec<_>>(),
        actual.difference(&expected).collect::<Vec<_>>()
    );
    let marker = &loaded["__fixed_sgd_no_state__"];
    ensure!(
        marker.numel() == 1 && marker.kind() == Kind::Int64 && marker.int64_value(&[]) == 0,
        "fixed-SGD optimizer no-state marker is invalid"
    );
    Ok(())
}

fn clone_named(named: &[(String, Tensor)]) -> Vec<(String, Tensor)> {
    named
        .iter()
        .map(|(name, parameter)| (name.clone(), parameter.shallow_clone()))
        .collect()
}

fn zero_named(named: &mut [(String, Tensor)]) {
    for (_, parameter) in named {
        parameter.zero_grad();
    }
}

impl FixedSgd {
    fn step(&mut self) -> Result<()> {
        no_grad(|| {
            for (name, parameter) in &mut self.params {
                let gradient = parameter.grad();
                ensure!(
                    gradient.defined(),
                    "fixed SGD parameter {name} has no gradient"
                );
                // The trainer's one packed host transfer rejects non-finite gradients before
                // this mutation; do not add a per-parameter synchronization here.
                let delta = gradient.to_kind(Kind::Float) * (-self.lr);
                let _ = parameter.g_add_(&delta.to_kind(parameter.kind()));
            }
            Ok(())
        })
    }
}

impl SmdIdbd {
    fn backward_exact_hvp(&mut self, loss: &Tensor) -> Result<()> {
        let parameters: Vec<Tensor> = self
            .params
            .iter()
            .map(|entry| entry.parameter.shallow_clone())
            .collect();
        // Retain the ordinary loss graph because `loss.backward()` below repopulates the
        // parameter grad slots consumed by the existing gradient-norm and metric path.
        let gradients = Tensor::run_backward(&[loss], &parameters, true, true);
        ensure!(
            gradients.len() == self.params.len(),
            "SMD gradient count {} does not match parameter count {}",
            gradients.len(),
            self.params.len()
        );

        // Pearlmutter product: grad_w <grad(loss), stop_grad(v)> = H v. `create_graph=true`
        // above is the load-bearing distinction from an empirical-Fisher approximation.
        let directional_terms: Vec<Tensor> = gradients
            .iter()
            .zip(&self.params)
            .map(|(gradient, entry)| {
                (gradient.to_kind(Kind::Float) * entry.v.detach()).sum(Kind::Float)
            })
            .collect();
        let directional = Tensor::stack(&directional_terms, 0).sum(Kind::Float);
        let hv = Tensor::run_backward(&[directional], &parameters, true, false);
        ensure!(
            hv.len() == self.params.len(),
            "SMD HVP count {} does not match parameter count {}",
            hv.len(),
            self.params.len()
        );

        no_grad(|| {
            for (entry, product) in self.params.iter_mut().zip(&hv) {
                entry.pending_hv.copy_(&product.to_kind(Kind::Float));
            }
        });
        self.pending_metrics = Some(self.metrics_for_pending_update(&gradients, &hv));
        self.pending = true;

        // `run_backward` follows autograd-grad semantics and does not populate `.grad`.
        // Preserve the existing metric contract by performing ordinary backward while the
        // retained graph is still alive. The returned graph gradients above remain the exact
        // values used for H v; this call is only the ordinary-grad-slot bridge.
        loss.backward();
        Ok(())
    }

    fn metrics_for_pending_update(&self, gradients: &[Tensor], hv: &[Tensor]) -> Tensor {
        let mut count = 0f64;
        let mut gain_sum = Vec::with_capacity(self.params.len());
        let mut gain_square_sum = Vec::with_capacity(self.params.len());
        let mut gain_min = Vec::with_capacity(self.params.len());
        let mut gain_max = Vec::with_capacity(self.params.len());
        let mut bound_sum = Vec::with_capacity(self.params.len());
        let mut credit_sum = Vec::with_capacity(self.params.len());
        let mut credit_square_sum = Vec::with_capacity(self.params.len());
        let mut beta_update_abs_sum = Vec::with_capacity(self.params.len());
        let mut trace_square_sum = Vec::with_capacity(self.params.len());
        let mut hv_square_sum = Vec::with_capacity(self.params.len());
        let mut gradient_square_sum = Vec::with_capacity(self.params.len());

        for ((entry, gradient), product) in self.params.iter().zip(gradients).zip(hv) {
            let d = -gradient.detach().to_kind(Kind::Float);
            let credit = &d * entry.v.detach();
            let beta_update =
                (&credit * self.meta_lr).clamp(-SMD_BETA_UPDATE_CLIP, SMD_BETA_UPDATE_CLIP);
            let beta = (&entry.beta + &beta_update).clamp(SMD_BETA_MIN, SMD_BETA_MAX);
            let gain = beta.exp() * self.base_lr;
            count += entry.parameter.numel() as f64;
            gain_sum.push(gain.sum(Kind::Float));
            gain_square_sum.push(gain.square().sum(Kind::Float));
            gain_min.push(gain.min());
            gain_max.push(gain.max());
            bound_sum.push(
                beta.le(SMD_BETA_MIN)
                    .logical_or(&beta.ge(SMD_BETA_MAX))
                    .to_kind(Kind::Float)
                    .sum(Kind::Float),
            );
            credit_sum.push(credit.sum(Kind::Float));
            credit_square_sum.push(credit.square().sum(Kind::Float));
            beta_update_abs_sum.push(beta_update.abs().sum(Kind::Float));
            trace_square_sum.push(
                (&entry.v + &gain * (&d - product.detach().to_kind(Kind::Float)))
                    .square()
                    .sum(Kind::Float),
            );
            hv_square_sum.push(
                product
                    .detach()
                    .to_kind(Kind::Float)
                    .square()
                    .sum(Kind::Float),
            );
            gradient_square_sum.push(d.square().sum(Kind::Float));
        }

        let sum = |values: &[Tensor]| Tensor::stack(values, 0).sum(Kind::Float);
        let gain_mean = sum(&gain_sum) / count;
        let gain_variance = (sum(&gain_square_sum) / count - gain_mean.square()).clamp_min(0.0);
        let credit_mean = sum(&credit_sum) / count;
        let credit_variance =
            (sum(&credit_square_sum) / count - credit_mean.square()).clamp_min(0.0);
        let gradient_squares = sum(&gradient_square_sum);
        Tensor::stack(
            &[
                gain_mean,
                gain_variance.sqrt(),
                Tensor::stack(&gain_min, 0).min(),
                Tensor::stack(&gain_max, 0).max(),
                sum(&bound_sum) / count,
                credit_mean,
                credit_variance.sqrt(),
                sum(&beta_update_abs_sum) / count,
                (sum(&trace_square_sum) / count).sqrt(),
                (sum(&hv_square_sum) / gradient_squares.clamp_min(1e-30)).sqrt(),
            ],
            0,
        )
    }

    fn step(&mut self) -> Result<()> {
        ensure!(self.pending, "SMD step called without a pending exact HVP");
        no_grad(|| {
            for entry in &mut self.params {
                let gradient = entry.parameter.grad();
                ensure!(
                    gradient.defined(),
                    "SMD parameter {} has no gradient",
                    entry.name
                );
                let d = -gradient.to_kind(Kind::Float);
                let credit = &d * &entry.v;
                let beta_update =
                    (&credit * self.meta_lr).clamp(-SMD_BETA_UPDATE_CLIP, SMD_BETA_UPDATE_CLIP);
                let beta = (&entry.beta + beta_update).clamp(SMD_BETA_MIN, SMD_BETA_MAX);
                let gain = beta.exp() * self.base_lr;
                entry.beta.copy_(&beta);
                let _ = entry
                    .parameter
                    .g_add_(&(&gain * &d).to_kind(entry.parameter.kind()));
                let trace_increment = &gain * (&d - &entry.pending_hv);
                let _ = entry.v.g_add_(&trace_increment);
            }
            self.pending = false;
            self.pending_metrics = None;
            Ok(())
        })
    }

    fn save_state(&self, path: &Path) -> Result<()> {
        ensure!(
            !self.pending,
            "refusing to save SMD-IDBD state while an exact HVP update is pending"
        );
        let mut named = Vec::with_capacity(self.params.len() * 2);
        for entry in &self.params {
            named.push((
                format!("{}.__smd_beta", entry.name),
                entry.beta.to_device(Device::Cpu),
            ));
            named.push((
                format!("{}.__smd_v", entry.name),
                entry.v.to_device(Device::Cpu),
            ));
        }
        let refs = named
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        Tensor::save_multi(&refs, path)
            .with_context(|| format!("failed saving SMD-IDBD state {}", path.display()))
    }

    fn validate_state_strict(&self, path: &Path) -> Result<()> {
        let loaded: HashMap<String, Tensor> = Tensor::load_multi_with_device(path, Device::Cpu)
            .with_context(|| format!("failed loading SMD-IDBD state {}", path.display()))?
            .into_iter()
            .collect();
        let mut expected_keys = HashSet::with_capacity(self.params.len() * 2);
        for entry in &self.params {
            for (suffix, expected) in [("__smd_beta", &entry.beta), ("__smd_v", &entry.v)] {
                let key = format!("{}.{suffix}", entry.name);
                let tensor = loaded.get(&key).with_context(|| {
                    format!("SMD-IDBD state missing {suffix} for {}", entry.name)
                })?;
                ensure!(
                    tensor.size() == expected.size() && tensor.kind() == Kind::Float,
                    "SMD-IDBD {suffix} schema mismatch for {}: expected fp32 {:?}, got {:?} {:?}",
                    entry.name,
                    expected.size(),
                    tensor.kind(),
                    tensor.size()
                );
                ensure!(
                    tensor.isfinite().all().int64_value(&[]) != 0,
                    "SMD-IDBD {suffix} contains non-finite values for {}",
                    entry.name
                );
                expected_keys.insert(key);
            }
        }
        let actual_keys = loaded.keys().cloned().collect::<HashSet<_>>();
        ensure!(
            actual_keys == expected_keys,
            "SMD-IDBD tensor schema differs from the current model: missing={:?}, unexpected={:?}",
            expected_keys.difference(&actual_keys).collect::<Vec<_>>(),
            actual_keys.difference(&expected_keys).collect::<Vec<_>>()
        );
        Ok(())
    }

    fn load_state(&mut self, path: &Path) -> Result<()> {
        ensure!(
            !self.pending,
            "refusing to load SMD-IDBD state while an exact HVP update is pending"
        );
        let loaded: HashMap<String, Tensor> = Tensor::load_multi_with_device(path, Device::Cpu)
            .with_context(|| format!("failed loading SMD-IDBD state {}", path.display()))?
            .into_iter()
            .collect();
        no_grad(|| {
            for entry in &mut self.params {
                entry
                    .beta
                    .copy_(&loaded[&format!("{}.__smd_beta", entry.name)]);
                entry.v.copy_(&loaded[&format!("{}.__smd_v", entry.name)]);
                let _ = entry.pending_hv.zero_();
            }
        });
        self.pending = false;
        self.pending_metrics = None;
        Ok(())
    }

    fn state_bytes(&self) -> usize {
        self.params
            .iter()
            .map(|entry| {
                (entry.beta.numel() + entry.v.numel() + entry.pending_hv.numel())
                    * Kind::Float.elt_size_in_bytes()
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, path::PathBuf, process, time::SystemTime};
    use tch::Device;

    fn scalar(value: f32) -> Tensor {
        Tensor::from_slice(&[value]).set_requires_grad(true)
    }

    #[test]
    fn fixed_sgd_is_the_exact_plain_update() {
        let parameter = scalar(2.0);
        let named = vec![("w".to_owned(), parameter.shallow_clone())];
        let mut optimizer = PretrainOptimizer::fixed_sgd(&named, 0.1);
        optimizer.zero_grad();
        let loss = parameter.square().sum(Kind::Float);
        optimizer.backward(&loss).unwrap();
        optimizer.step(StepKind::Primary).unwrap();
        assert_eq!(Vec::<f32>::try_from(&parameter).unwrap(), vec![1.6]);
    }

    #[test]
    fn smd_first_step_matches_fixed_sgd_when_the_trace_is_zero() {
        let fixed_parameter = scalar(2.0);
        let smd_parameter = scalar(2.0);
        let mut fixed =
            PretrainOptimizer::fixed_sgd(&[("w".to_owned(), fixed_parameter.shallow_clone())], 0.1);
        let mut smd = PretrainOptimizer::smd_idbd(
            &[("w".to_owned(), smd_parameter.shallow_clone())],
            0.1,
            0.05,
        );
        for (optimizer, parameter) in [(&mut fixed, &fixed_parameter), (&mut smd, &smd_parameter)] {
            optimizer.zero_grad();
            optimizer
                .backward(&parameter.square().sum(Kind::Float))
                .unwrap();
            optimizer.step(StepKind::Primary).unwrap();
        }
        assert_eq!(
            Vec::<f32>::try_from(&fixed_parameter).unwrap(),
            Vec::<f32>::try_from(&smd_parameter).unwrap()
        );
    }

    #[test]
    fn persistent_same_sign_credit_raises_the_gain_on_a_quadratic() {
        let parameter = scalar(2.0);
        let mut optimizer =
            PretrainOptimizer::smd_idbd(&[("w".to_owned(), parameter.shallow_clone())], 0.01, 0.05);
        let mut gains = Vec::new();
        for _ in 0..4 {
            optimizer.zero_grad();
            optimizer
                .backward(&(parameter.square() * 0.5).sum(Kind::Float))
                .unwrap();
            gains.push(Vec::<f32>::try_from(optimizer.smd_metrics_tensor().unwrap()).unwrap()[0]);
            optimizer.step(StepKind::Primary).unwrap();
        }
        assert!(
            gains[3] > gains[1],
            "persistent credit should raise gain: {gains:?}"
        );
    }

    #[test]
    fn beta_updates_are_clipped_and_beta_stays_bounded() {
        let parameter = scalar(100.0);
        let mut optimizer = SmdIdbd {
            params: vec![SmdParameter {
                name: "w".to_owned(),
                parameter: parameter.shallow_clone(),
                beta: Tensor::full([1], SMD_BETA_MAX, (Kind::Float, Device::Cpu)),
                v: Tensor::full([1], -1e20, (Kind::Float, Device::Cpu)),
                pending_hv: Tensor::zeros([1], (Kind::Float, Device::Cpu)),
            }],
            base_lr: 1e-3,
            meta_lr: 0.05,
            pending: true,
            pending_metrics: None,
        };
        parameter.square().sum(Kind::Float).backward();
        optimizer.step().unwrap();
        assert_eq!(optimizer.params[0].beta.double_value(&[0]), SMD_BETA_MAX);
        assert!(optimizer.params[0].v.double_value(&[0]).is_finite());
    }

    #[test]
    fn nonfinite_smd_state_reaches_the_packed_guard_instead_of_being_replaced() {
        let parameter = scalar(1.0);
        let optimizer = SmdIdbd {
            params: vec![SmdParameter {
                name: "w".to_owned(),
                parameter,
                beta: Tensor::zeros([1], (Kind::Float, Device::Cpu)),
                v: Tensor::ones([1], (Kind::Float, Device::Cpu)),
                pending_hv: Tensor::zeros([1], (Kind::Float, Device::Cpu)),
            }],
            base_lr: 1e-3,
            meta_lr: 0.05,
            pending: false,
            pending_metrics: None,
        };
        let metrics = optimizer.metrics_for_pending_update(
            &[Tensor::full([1], f64::NAN, (Kind::Float, Device::Cpu))],
            &[Tensor::zeros([1], (Kind::Float, Device::Cpu))],
        );
        let values = Vec::<f32>::try_from(metrics).unwrap();
        assert!(
            values.iter().any(|value| !value.is_finite()),
            "non-finite SMD state must remain visible to the trainer's fail-closed packet"
        );
    }

    fn temporary_state_path(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "trading_bot_smd_{label}_{}_{}.ot",
            process::id(),
            nonce
        ))
    }

    #[test]
    fn smd_state_round_trip_restores_beta_and_trace_only() {
        let path = temporary_state_path("round_trip");
        let parameter = scalar(1.0);
        let mut source = PretrainOptimizer::smd_idbd(&[("w".to_owned(), parameter)], 1e-3, 0.1);
        let PretrainOptimizer::SmdIdbd(source_state) = &mut source else {
            unreachable!()
        };
        no_grad(|| {
            let _ = source_state.params[0].beta.fill_(0.7);
            let _ = source_state.params[0].v.fill_(-0.3);
            let _ = source_state.params[0].pending_hv.fill_(9.0);
        });
        source.save_state(&path).expect("save SMD state");

        let restored_parameter = scalar(1.0);
        let mut restored =
            PretrainOptimizer::smd_idbd(&[("w".to_owned(), restored_parameter)], 1e-3, 0.1);
        restored
            .load_state_strict(&path, &[], 0)
            .expect("strict SMD restore");
        let PretrainOptimizer::SmdIdbd(restored_state) = restored else {
            unreachable!()
        };
        assert_eq!(
            Vec::<f32>::try_from(&restored_state.params[0].beta).unwrap(),
            vec![0.7]
        );
        assert_eq!(
            Vec::<f32>::try_from(&restored_state.params[0].v).unwrap(),
            vec![-0.3]
        );
        assert_eq!(
            Vec::<f32>::try_from(&restored_state.params[0].pending_hv).unwrap(),
            vec![0.0]
        );
        assert!(!restored_state.pending);
        assert!(restored_state.pending_metrics.is_none());
        fs::remove_file(path).expect("remove SMD state");
    }

    #[test]
    fn strict_smd_state_rejects_missing_wrong_shape_and_unexpected_tensors() {
        let parameter = scalar(1.0);
        let optimizer = PretrainOptimizer::smd_idbd(&[("w".to_owned(), parameter)], 1e-3, 0.1);
        let beta = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        let trace = Tensor::zeros([1], (Kind::Float, Device::Cpu));

        let missing = temporary_state_path("missing");
        Tensor::save_multi(&[("w.__smd_beta", &beta)], &missing).expect("save missing schema");
        assert!(optimizer.validate_state_strict(&missing, &[], 0).is_err());
        fs::remove_file(missing).expect("remove missing schema");

        let wrong_shape = temporary_state_path("wrong_shape");
        let wrong_trace = Tensor::zeros([2], (Kind::Float, Device::Cpu));
        Tensor::save_multi(
            &[("w.__smd_beta", &beta), ("w.__smd_v", &wrong_trace)],
            &wrong_shape,
        )
        .expect("save wrong-shape schema");
        assert!(optimizer
            .validate_state_strict(&wrong_shape, &[], 0)
            .is_err());
        fs::remove_file(wrong_shape).expect("remove wrong-shape schema");

        let unexpected = temporary_state_path("unexpected");
        let extra = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        Tensor::save_multi(
            &[
                ("w.__smd_beta", &beta),
                ("w.__smd_v", &trace),
                ("w.__unexpected", &extra),
            ],
            &unexpected,
        )
        .expect("save unexpected schema");
        assert!(optimizer
            .validate_state_strict(&unexpected, &[], 0)
            .is_err());
        fs::remove_file(unexpected).expect("remove unexpected schema");
    }

    #[test]
    fn smd_state_cannot_be_saved_with_a_pending_hvp() {
        let path = temporary_state_path("pending");
        let parameter = scalar(1.0);
        let mut optimizer = PretrainOptimizer::smd_idbd(&[("w".to_owned(), parameter)], 1e-3, 0.1);
        let PretrainOptimizer::SmdIdbd(state) = &mut optimizer else {
            unreachable!()
        };
        state.pending = true;
        assert!(optimizer.save_state(&path).is_err());
        assert!(!path.exists());
    }
}
