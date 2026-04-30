use tch::{nn, Device, Kind, Tensor};

use crate::torch::optim::muon::Muon;

use super::config::{MUON_MOMENTUM, MUON_MOMENTUM_WARMUP_START, MUON_MOMENTUM_WARMUP_STEPS};

pub(crate) fn clip_grad_norm_on_device(
    trainable_vars: &[Tensor],
    max_grad_norm: f64,
    device: Device,
) -> Tensor {
    tch::no_grad(|| {
        let mut total_norm_sq = Tensor::zeros([], (Kind::Float, device));
        let mut has_grads = false;
        for v in trainable_vars {
            let g = v.grad();
            if g.defined() {
                total_norm_sq += g.square().sum(Kind::Float);
                has_grads = true;
            }
        }
        if !has_grads {
            return Tensor::zeros([], (Kind::Float, device));
        }

        let total_norm = total_norm_sq.sqrt();
        let clip_coef = Tensor::from(max_grad_norm as f32).to_device(device) / (&total_norm + 1e-6);
        let clip_coef = clip_coef.clamp_max(1.0);

        for v in trainable_vars {
            let mut g = v.grad();
            if g.defined() {
                let coef = clip_coef.to_kind(g.kind());
                let _ = g.g_mul_(&coef);
            }
        }

        total_norm
    })
}

pub(crate) fn named_trainable_variables(vs: &nn::VarStore) -> Vec<(String, Tensor)> {
    let mut vars: Vec<(String, Tensor)> = vs
        .variables()
        .into_iter()
        .filter(|(_, tensor)| tensor.requires_grad())
        .collect();
    vars.sort_by(|a, b| a.0.cmp(&b.0));
    vars
}

pub(crate) fn muon_momentum_for_step(step: i64) -> f64 {
    let frac = if MUON_MOMENTUM_WARMUP_STEPS > 0 {
        (step as f64 / MUON_MOMENTUM_WARMUP_STEPS as f64).clamp(0.0, 1.0)
    } else {
        1.0
    };
    (1.0 - frac) * MUON_MOMENTUM_WARMUP_START + frac * MUON_MOMENTUM
}

pub(crate) fn step_optimizer(opt: &mut Muon, optimizer_step: &mut i64) {
    opt.set_momentum(muon_momentum_for_step(*optimizer_step));
    opt.step();
    *optimizer_step += 1;
}
