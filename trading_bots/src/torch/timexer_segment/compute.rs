use crate::torch::optim::muon::{Muon, MuonConfig, Orthogonalizer, StepKind};
use crate::torch::train::optimizer_glue::named_trainable_variables;
use anyhow::{ensure, Result};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tch::{nn, nn::OptimizerConfig, Kind, Tensor};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum OptimizerKind {
    #[default]
    PolarExpress,
    Adam,
}

/// modded-nanogpt `train_gpt.py` Adam / NorMuon bases (`:2065-2076`).
const NANOGPT_ADAMW_LR: f64 = 0.008;
const NANOGPT_MUON_LR: f64 = 0.023;
const NANOGPT_COOLDOWN_FRAC: f64 = 0.60;
const NANOGPT_COOLDOWN_FLOOR: f64 = 0.15;
const NANOGPT_MUON_WARMUP_STEPS: usize = 300;
const NANOGPT_MUON_COOLDOWN_STEPS: usize = 50;
const NANOGPT_MOMENTUM_MIN: f64 = 0.85;
const NANOGPT_MOMENTUM_MAX: f64 = 0.95;

impl OptimizerKind {
    pub fn default_learning_rate(self) -> f64 {
        match self {
            Self::PolarExpress => NANOGPT_ADAMW_LR,
            Self::Adam => 0.0001,
        }
    }

    pub fn muon_learning_rate(self, learning_rate: f64) -> Option<f64> {
        (self == Self::PolarExpress).then_some(learning_rate * (NANOGPT_MUON_LR / NANOGPT_ADAMW_LR))
    }

    pub fn recipe(self) -> &'static str {
        match self {
            Self::PolarExpress => {
                "normuon-pe5-block-hidden2D-only;input-variate-global-norm-bias-head-AdamW;muonLR=.023;adamwLR=.008;muonWD=1.2;adamwWD=.005;head-embed-wd_mul=150;quadraticWD;cautiousWD;b2=.9;nesterov;momentum=.85-to-.95-over300-cd50;AdamWbetas=.9,.95;eps=1e-10;AdamWevery=1;stepgraphs;nanogpt-cooldown-frac=.60-floor=.15-v1"
            }
            Self::Adam => {
                "Adam-fp32-masters;betas=.9,.999;eps=1e-8;WD=0;nanogpt-cooldown-frac=.60-floor=.15-v1"
            }
        }
    }
}

/// Step multiplier from modded-nanogpt `TrainingSchedule.get_lr` without the
/// batch-size stage bumps (`train_gpt.py:1968-1976`, `cooldown_frac=0.60`).
pub fn nanogpt_lr_scale(step: usize, scheduled_steps: usize) -> f64 {
    if scheduled_steps == 0 {
        return 1.0;
    }
    let cooldown_start =
        ((scheduled_steps as f64) * (1.0 - NANOGPT_COOLDOWN_FRAC)).floor() as usize;
    if step < cooldown_start {
        return 1.0;
    }
    let span = (scheduled_steps - cooldown_start).max(1) as f64;
    let t = ((step - cooldown_start) as f64 / span).clamp(0.0, 1.0);
    (1.0 - t) + NANOGPT_COOLDOWN_FLOOR * t
}

/// NorMuon momentum from modded-nanogpt `get_muon_momentum` (`train_gpt.py:1995-2007`).
pub fn nanogpt_muon_momentum(step: usize, scheduled_steps: usize) -> f64 {
    if scheduled_steps == 0 {
        return NANOGPT_MOMENTUM_MAX;
    }
    let cooldown_start = scheduled_steps.saturating_sub(NANOGPT_MUON_COOLDOWN_STEPS);
    if step < NANOGPT_MUON_WARMUP_STEPS {
        let frac = step as f64 / NANOGPT_MUON_WARMUP_STEPS as f64;
        NANOGPT_MOMENTUM_MIN + frac * (NANOGPT_MOMENTUM_MAX - NANOGPT_MOMENTUM_MIN)
    } else if step > cooldown_start {
        let frac = ((step - cooldown_start) as f64 / NANOGPT_MUON_COOLDOWN_STEPS as f64).min(1.0);
        NANOGPT_MOMENTUM_MAX - frac * (NANOGPT_MOMENTUM_MAX - NANOGPT_MOMENTUM_MIN)
    } else {
        NANOGPT_MOMENTUM_MAX
    }
}

fn polar_express(named: &[(String, Tensor)], learning_rate: f64) -> Muon {
    let optimizer = Muon::new_named(
        named,
        MuonConfig {
            lr: OptimizerKind::PolarExpress
                .muon_learning_rate(learning_rate)
                .unwrap(),
            use_muon_for_2d: true,
            momentum: NANOGPT_MOMENTUM_MIN,
            nesterov: true,
            beta2: 0.9,
            weight_decay: 1.2,
            adamw_lr: learning_rate,
            adamw_betas: (0.9, 0.95),
            adamw_eps: 1e-10,
            adamw_wd: 0.005,
            adamw_no_weight_decay_name_substrings: vec!["norm".into(), ".bias".into()],
            adamw_weight_decay_multipliers: vec![
                ("head.".into(), 150.),
                ("patch.".into(), 150.),
                ("variate.".into(), 150.),
                ("global_token".into(), 150.),
            ],
            muon_name_allowlist: vec!["block_".into()],
            force_adamw_name_substrings: vec![
                "patch.".into(),
                "variate.".into(),
                "global_token".into(),
                "norm".into(),
                ".bias".into(),
                "head.".into(),
            ],
            orthogonalizer: Orthogonalizer::PolarExpress5,
            adamw_every: 1,
            quadratic_lr_weight_decay: true,
            cautious_weight_decay: true,
            row_learned_lr: false,
            capture_step_graphs: true,
            ..MuonConfig::default()
        },
    );
    let muon: HashSet<_> = optimizer.muon_param_names().into_iter().collect();
    let adamw: HashSet<_> = optimizer.adamw_param_names().into_iter().collect();
    assert!(muon.is_disjoint(&adamw));
    assert_eq!(
        muon.len() + adamw.len(),
        named.len(),
        "optimizer must cover every parameter exactly once"
    );
    for (name, tensor) in named {
        let hidden = name.starts_with("block_") && name.ends_with(".weight") && tensor.dim() == 2;
        assert_eq!(
            muon.contains(name),
            hidden,
            "unexpected Polar Express routing: {name}"
        );
        assert_eq!(
            adamw.contains(name),
            !hidden,
            "unexpected AdamW routing: {name}"
        );
    }
    optimizer
}

use super::model::{SegmentModel, CHANNELS};

enum Optimizer {
    PolarExpress(Box<Muon>),
    Native(nn::Optimizer),
    Fused(Py<PyAny>),
}

pub struct Engine {
    optimizer: Optimizer,
    completed_steps: usize,
    base_lr: f64,
    scheduled_steps: usize,
}

impl Engine {
    pub fn new(
        store: &nn::VarStore,
        learning_rate: f64,
        fused: bool,
        kind: OptimizerKind,
        scheduled_steps: usize,
    ) -> Result<Self> {
        ensure!(
            learning_rate.is_finite() && learning_rate > 0.,
            "invalid learning rate"
        );
        let optimizer = if kind == OptimizerKind::PolarExpress {
            Optimizer::PolarExpress(Box::new(polar_express(
                &named_trainable_variables(store),
                learning_rate,
            )))
        } else if fused {
            let optimizer = Python::attach(|py| -> Result<Py<PyAny>> {
                let torch = py.import("torch")?;
                let parameters = PyList::empty(py);
                for tensor in store.trainable_variables() {
                    ensure!(
                        tensor.kind() == Kind::Float,
                        "Adam requires FP32 master parameters"
                    );
                    parameters.append(crate::torch::fa4::tensor_object(py, &tensor)?)?;
                }
                let options = PyDict::new(py);
                options.set_item("lr", learning_rate)?;
                options.set_item("fused", true)?;
                options.set_item("weight_decay", 0.)?;
                Ok(torch
                    .getattr("optim")?
                    .getattr("Adam")?
                    .call((parameters,), Some(&options))?
                    .unbind())
            })?;
            Optimizer::Fused(optimizer)
        } else {
            Optimizer::Native(nn::Adam::default().build(store, learning_rate)?)
        };
        let mut engine = Self {
            optimizer,
            completed_steps: 0,
            base_lr: learning_rate,
            scheduled_steps,
        };
        engine.apply_schedule()?;
        Ok(engine)
    }

    pub fn learning_rate(&self) -> f64 {
        let step = self.completed_steps.saturating_sub(1);
        self.base_lr * nanogpt_lr_scale(step, self.scheduled_steps)
    }

    fn apply_schedule(&mut self) -> Result<()> {
        let scale = nanogpt_lr_scale(self.completed_steps, self.scheduled_steps);
        let adamw_lr = self.base_lr * scale;
        match &mut self.optimizer {
            Optimizer::PolarExpress(optimizer) => {
                optimizer.set_lr(
                    OptimizerKind::PolarExpress
                        .muon_learning_rate(adamw_lr)
                        .unwrap(),
                );
                optimizer.set_adamw_lr(adamw_lr);
                optimizer.set_momentum(nanogpt_muon_momentum(
                    self.completed_steps,
                    self.scheduled_steps,
                ));
            }
            Optimizer::Native(optimizer) => optimizer.set_lr(adamw_lr),
            Optimizer::Fused(optimizer) => Python::attach(|py| -> Result<()> {
                for group in optimizer.bind(py).getattr("param_groups")?.try_iter()? {
                    group?.set_item("lr", adamw_lr)?;
                }
                Ok(())
            })?,
        }
        Ok(())
    }

    pub fn set_lr(&mut self, learning_rate: f64) -> Result<()> {
        ensure!(
            learning_rate.is_finite() && learning_rate > 0.,
            "invalid learning rate"
        );
        self.base_lr = learning_rate;
        self.apply_schedule()
    }

    pub fn zero_grad(&mut self) -> Result<()> {
        match &mut self.optimizer {
            Optimizer::PolarExpress(optimizer) => optimizer.zero_grad(),
            Optimizer::Native(optimizer) => optimizer.zero_grad(),
            Optimizer::Fused(optimizer) => Python::attach(|py| -> PyResult<()> {
                optimizer.bind(py).call_method0("zero_grad")?;
                Ok(())
            })?,
        }
        Ok(())
    }

    pub fn optimizer_step(&mut self) -> Result<()> {
        self.apply_schedule()?;
        match &mut self.optimizer {
            Optimizer::PolarExpress(optimizer) => {
                optimizer.step(StepKind::Primary);
                assert!(
                    !optimizer.adamw_accumulation_pending(),
                    "every-step AdamW must consume its gradients"
                );
            }
            Optimizer::Native(optimizer) => optimizer.step(),
            Optimizer::Fused(optimizer) => Python::attach(|py| -> PyResult<()> {
                optimizer.bind(py).call_method0("step")?;
                Ok(())
            })?,
        }
        self.completed_steps += 1;
        self.ensure_optimizer_ready()
    }

    pub fn optimizer_graph_captured(&self) -> Option<bool> {
        match &self.optimizer {
            Optimizer::PolarExpress(optimizer) => Some(optimizer.with_adamw_step_graph_captured()),
            _ => None,
        }
    }

    pub fn ensure_optimizer_ready(&self) -> Result<()> {
        ensure!(
            self.completed_steps < 4 || self.optimizer_graph_captured() != Some(false),
            "Polar Express optimizer graph did not remain captured after warmup"
        );
        Ok(())
    }

    pub fn forward_loss(
        model: &SegmentModel,
        input: &Tensor,
        target: &Tensor,
        price_scaling: &Tensor,
        geometry_context: &Tensor,
        auxiliary: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Tensor {
        masked_mse(
            &model
                .forward_with_aux(input, auxiliary, price_scaling, geometry_context, true)
                .standardized,
            target,
            mask,
        )
    }

    pub fn step(
        &mut self,
        model: &SegmentModel,
        input: &Tensor,
        target: &Tensor,
        price_scaling: &Tensor,
        geometry_context: &Tensor,
        auxiliary: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.zero_grad()?;
        let loss = Self::forward_loss(
            model,
            input,
            target,
            price_scaling,
            geometry_context,
            auxiliary,
            mask,
        );
        crate::torch::single_ticker_timexer::runner::check_objective(&loss)?;
        loss.backward();
        self.optimizer_step()?;
        Ok(loss.detach())
    }
}

fn masked_mse(prediction: &Tensor, target: &Tensor, mask: Option<&Tensor>) -> Tensor {
    let squared = (prediction - target).square();
    match mask {
        Some(mask) => {
            assert_eq!(mask.size(), squared.size()[..2]);
            (squared * mask.unsqueeze(-1)).sum(Kind::Float) / (mask.sum(Kind::Float) * CHANNELS)
        }
        None => squared.mean(Kind::Float),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn polar_express_routes_only_hidden_matrices_and_keeps_readout_on_adamw() {
        let store = nn::VarStore::new(Device::Cpu);
        for (path, name, shape) in [
            ("block_0.self_attention.query", "weight", vec![8, 8]),
            ("block_0.cross_attention.key_value", "weight", vec![16, 8]),
            ("block_0.first", "weight", vec![32, 8]),
            ("block_0.second", "weight", vec![8, 32]),
            ("block_0.first", "bias", vec![32]),
            ("block_0.norm_0", "weight", vec![8]),
            ("patch", "weight", vec![8, 4]),
            ("variate", "weight", vec![8, 64]),
            ("head", "weight", vec![4, 128]),
            ("head", "bias", vec![4]),
        ] {
            let path = path.split('.').fold(store.root(), |path, part| path / part);
            let _ = path.var(name, &shape, nn::Init::Const(0.));
        }
        let _ = store
            .root()
            .var("global_token", &[1, 4, 1, 8], nn::Init::Const(0.));
        let named = named_trainable_variables(&store);
        let optimizer = polar_express(&named, NANOGPT_ADAMW_LR);
        assert_eq!(optimizer.muon_param_names().len(), 4);
        assert_eq!(optimizer.adamw_param_names().len(), 7);
        for name in [
            "head.weight",
            "variate.weight",
            "patch.weight",
            "global_token",
        ] {
            assert!(optimizer
                .adamw_param_names()
                .iter()
                .any(|candidate| candidate == name));
        }
        assert!((optimizer.lr() - NANOGPT_MUON_LR).abs() < 1e-12);
        assert!(!optimizer.with_adamw_step_graph_captured());
        let mut engine = Engine::new(
            &store,
            NANOGPT_ADAMW_LR,
            true,
            OptimizerKind::PolarExpress,
            0,
        )
        .unwrap();
        engine.set_lr(NANOGPT_ADAMW_LR / 2.0).unwrap();
        let Optimizer::PolarExpress(optimizer) = engine.optimizer else {
            unreachable!()
        };
        assert!((optimizer.lr() - NANOGPT_MUON_LR / 2.0).abs() < 1e-12);
        assert!(!optimizer.adamw_accumulation_pending());
    }

    #[test]
    fn nanogpt_lr_scale_holds_then_cools_to_the_floor() {
        let scheduled = 1270;
        let start = ((scheduled as f64) * 0.4).floor() as usize;
        assert_eq!(start, 508);
        assert!((nanogpt_lr_scale(0, scheduled) - 1.0).abs() < 1e-12);
        assert!((nanogpt_lr_scale(start, scheduled) - 1.0).abs() < 1e-12);
        let mid = start + (scheduled - start) / 2;
        assert!((nanogpt_lr_scale(mid, scheduled) - 0.575).abs() < 1e-12);
        assert!((nanogpt_lr_scale(scheduled, scheduled) - 0.15).abs() < 1e-12);
        assert!((nanogpt_lr_scale(scheduled + 15, scheduled) - 0.15).abs() < 1e-12);
        assert!((nanogpt_lr_scale(0, 0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nanogpt_muon_momentum_warms_up_and_cools_down() {
        assert!((nanogpt_muon_momentum(0, 1270) - 0.85).abs() < 1e-12);
        assert!((nanogpt_muon_momentum(150, 1270) - 0.90).abs() < 1e-12);
        assert!((nanogpt_muon_momentum(300, 1270) - 0.95).abs() < 1e-12);
        assert!((nanogpt_muon_momentum(1000, 1270) - 0.95).abs() < 1e-12);
        assert!((nanogpt_muon_momentum(1245, 1270) - 0.90).abs() < 1e-12);
        assert!((nanogpt_muon_momentum(1270, 1270) - 0.85).abs() < 1e-12);
        assert!((nanogpt_muon_momentum(0, 0) - 0.95).abs() < 1e-12);
    }

    #[test]
    fn partial_segments_count_only_owned_target_bars() {
        let prediction = Tensor::from_slice(&[1f32, 2., 100., 3., 100., 100.])
            .reshape([2, 3, 1])
            .repeat([1, 1, CHANNELS])
            .set_requires_grad(true);
        let target = Tensor::zeros_like(&prediction);
        let mask = Tensor::from_slice(&[1f32, 1., 0., 1., 0., 0.]).reshape([2, 3]);
        let loss = masked_mse(&prediction, &target, Some(&mask));
        assert!((loss.double_value(&[]) - 14. / 3.).abs() < 1e-6);
        loss.backward();
        let invalid = Tensor::ones([2, 3], (Kind::Float, Device::Cpu)) - mask;
        assert_eq!(
            (prediction.grad() * invalid.unsqueeze(-1))
                .abs()
                .max()
                .double_value(&[]),
            0.
        );
    }
}
