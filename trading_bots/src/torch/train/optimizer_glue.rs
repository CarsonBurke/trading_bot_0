use tch::{nn, Device, Kind, Tensor};

use crate::torch::optim::muon::Muon;

use super::config::{
    LEARNING_RATE, MUON_LR, MUON_MOMENTUM, MUON_MOMENTUM_WARMUP_START, MUON_MOMENTUM_WARMUP_STEPS,
};

const ACTOR_GRAD_CLIP_PATTERNS: &[&str] = &["policy_concentration"];
const CRITIC_GRAD_CLIP_PATTERNS: &[&str] = &["value_proj"];
const KL_LR_SCALE_EXPONENT: f64 = 0.5;

pub(crate) struct GradClipGroups {
    pub(crate) actor: Vec<Tensor>,
    pub(crate) critic: Vec<Tensor>,
    pub(crate) shared: Vec<Tensor>,
}

pub(crate) fn grad_clip_groups(named_vars: &[(String, Tensor)]) -> GradClipGroups {
    let mut actor = Vec::new();
    let mut critic = Vec::new();
    let mut shared = Vec::new();

    for (name, tensor) in named_vars {
        if ACTOR_GRAD_CLIP_PATTERNS
            .iter()
            .any(|pattern| name.contains(pattern))
        {
            actor.push(tensor.shallow_clone());
        } else if CRITIC_GRAD_CLIP_PATTERNS
            .iter()
            .any(|pattern| name.contains(pattern))
        {
            critic.push(tensor.shallow_clone());
        } else {
            shared.push(tensor.shallow_clone());
        }
    }

    GradClipGroups {
        actor,
        critic,
        shared,
    }
}

fn clip_grad_tensors_on_device(grads: &mut [Tensor], max_grad_norm: f64, device: Device) -> Tensor {
    tch::no_grad(|| {
        let mut total_norm_sq = Tensor::zeros([], (Kind::Float, device));
        for grad in grads.iter() {
            if !grad.defined() {
                continue;
            }
            total_norm_sq += grad.square().sum(Kind::Float);
        }

        let total_norm = total_norm_sq.sqrt();
        let clip_coef = Tensor::from(max_grad_norm as f32).to_device(device) / (&total_norm + 1e-6);
        let clip_coef = clip_coef.clamp_max(1.0);

        for grad in grads {
            if !grad.defined() {
                continue;
            }
            let coef = clip_coef.to_kind(grad.kind());
            let _ = grad.g_mul_(&coef);
        }

        total_norm
    })
}

fn clear_grads(params: &[Tensor]) {
    for param in params {
        let mut param = param.shallow_clone();
        param.zero_grad();
    }
}

fn accumulate_grad_surrogate(surrogate: Tensor, params: &[Tensor], grads: &[Tensor]) -> Tensor {
    params
        .iter()
        .zip(grads.iter())
        .fold(surrogate, |acc, (param, grad)| {
            if !grad.defined() {
                return acc;
            }
            acc + (param * &grad.detach()).sum(Kind::Float)
        })
}

pub(crate) fn backward_actor_critic_with_separate_clips(
    groups: &GradClipGroups,
    trainable_vars: &[Tensor],
    actor_loss: &Tensor,
    critic_loss: &Tensor,
    max_grad_norm: f64,
    device: Device,
) -> (Tensor, Tensor) {
    if max_grad_norm <= 0.0 {
        (actor_loss + critic_loss).backward();
        let zero = Tensor::zeros([], (Kind::Float, device));
        return (zero.shallow_clone(), zero);
    }

    let mut actor_params: Vec<Tensor> = groups
        .actor
        .iter()
        .chain(groups.shared.iter())
        .map(Tensor::shallow_clone)
        .collect();
    let mut critic_params: Vec<Tensor> = groups
        .critic
        .iter()
        .chain(groups.shared.iter())
        .map(Tensor::shallow_clone)
        .collect();

    let mut actor_grads = Tensor::run_backward(&[actor_loss], &actor_params, true, false);
    let actor_norm = clip_grad_tensors_on_device(&mut actor_grads, max_grad_norm, device);
    let mut critic_grads = Tensor::run_backward(&[critic_loss], &critic_params, false, false);
    let critic_norm = clip_grad_tensors_on_device(&mut critic_grads, max_grad_norm, device);

    clear_grads(trainable_vars);
    let surrogate = Tensor::zeros([], (Kind::Float, device));
    let surrogate = accumulate_grad_surrogate(surrogate, &actor_params, &actor_grads);
    let surrogate = accumulate_grad_surrogate(surrogate, &critic_params, &critic_grads);
    surrogate.backward();

    // Keep the param vectors mutable until after backward so their gradient slots stay live.
    actor_params.clear();
    critic_params.clear();

    (actor_norm, critic_norm)
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

#[derive(Debug, Clone)]
pub(crate) struct KlLrController {
    target: f64,
    ema_alpha: f64,
    min_scale: f64,
    max_scale: f64,
    ema: f64,
    scale: f64,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(crate) struct KlLrControllerState {
    version: u32,
    ema: f64,
    scale: f64,
}

impl KlLrController {
    pub(crate) fn new(target: f64, half_life: f64, min_scale: f64, max_scale: f64) -> Self {
        assert!(target.is_finite() && target > 0.0);
        assert!(half_life.is_finite() && half_life > 0.0);
        assert!(min_scale.is_finite() && min_scale > 0.0);
        assert!(max_scale.is_finite() && max_scale >= min_scale);
        Self {
            target,
            ema_alpha: kl_lr_ema_alpha(half_life),
            min_scale,
            max_scale,
            ema: target,
            scale: 1.0,
        }
    }

    pub(crate) fn scale(&self) -> f64 {
        self.scale
    }

    pub(crate) fn ema(&self) -> f64 {
        self.ema
    }

    pub(crate) fn restore(&mut self, ema: f64, scale: f64) {
        if ema.is_finite() && ema > 0.0 {
            self.ema = ema;
        }
        if scale.is_finite() && scale > 0.0 {
            self.scale = scale.clamp(self.min_scale, self.max_scale);
        }
    }

    pub(crate) fn state(&self) -> KlLrControllerState {
        KlLrControllerState {
            version: 1,
            ema: self.ema,
            scale: self.scale,
        }
    }

    pub(crate) fn restore_state(&mut self, state: KlLrControllerState) -> bool {
        if state.version != 1
            || !state.ema.is_finite()
            || state.ema <= 0.0
            || !state.scale.is_finite()
            || state.scale <= 0.0
        {
            return false;
        }
        self.restore(state.ema, state.scale);
        true
    }

    pub(crate) fn observe(&mut self, observed_kl: f64) {
        if !observed_kl.is_finite() {
            return;
        }
        let signal = observed_kl.max(0.0);
        self.ema = self.ema_alpha * signal + (1.0 - self.ema_alpha) * self.ema;
        self.scale = (self.scale * (self.target / self.ema.max(1e-12)).powf(KL_LR_SCALE_EXPONENT))
            .clamp(self.min_scale, self.max_scale);
    }
}

pub(crate) fn kl_lr_ema_alpha(half_life: f64) -> f64 {
    assert!(half_life.is_finite() && half_life > 0.0);
    1.0 - 0.5f64.powf(1.0 / half_life)
}

pub(crate) fn apply_lr_scale(opt: &mut Muon, lr_scale: f64) {
    assert!(lr_scale.is_finite() && lr_scale > 0.0);
    opt.set_lr(MUON_LR * lr_scale);
    opt.set_adamw_lr(LEARNING_RATE * lr_scale);
}

pub(crate) fn step_optimizer(opt: &mut Muon, optimizer_step: &mut i64) {
    opt.set_momentum(muon_momentum_for_step(*optimizer_step));
    opt.step();
    *optimizer_step += 1;
}

#[cfg(test)]
mod tests {
    use super::{
        backward_actor_critic_with_separate_clips, grad_clip_groups, kl_lr_ema_alpha,
        GradClipGroups, KlLrController,
    };
    use crate::torch::model::{TradingModel, TradingModelConfig};
    use crate::torch::train::optimizer_glue::named_trainable_variables;
    use std::collections::HashMap;
    use tch::nn;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn kl_lr_ema_alpha_matches_half_life() {
        let alpha = kl_lr_ema_alpha(20.0);
        let mut value = 1.0;
        for _ in 0..20 {
            value *= 1.0 - alpha;
        }

        assert!((value - 0.5).abs() < 1e-12);
    }

    #[test]
    fn kl_lr_controller_tracks_target_and_clamps() {
        let mut controller = KlLrController::new(0.02, 20.0, 0.1, 10.0);
        controller.observe(0.02);
        assert!((controller.ema() - 0.02).abs() < 1e-12);
        assert!((controller.scale() - 1.0).abs() < 1e-12);

        let mut high_kl_no_clamp = KlLrController::new(0.02, 0.01, 0.1, 10.0);
        high_kl_no_clamp.observe(0.08);
        assert!((high_kl_no_clamp.scale() - 0.5).abs() < 1e-12);

        let mut low_kl_no_clamp = KlLrController::new(0.02, 0.01, 0.1, 10.0);
        low_kl_no_clamp.observe(0.005);
        assert!((low_kl_no_clamp.scale() - 2.0).abs() < 1e-12);

        let mut low_kl = KlLrController::new(0.02, 0.01, 0.1, 10.0);
        low_kl.observe(0.0);
        assert!((low_kl.scale() - 10.0).abs() < 1e-9);

        let mut high_kl = KlLrController::new(0.02, 0.01, 0.1, 10.0);
        high_kl.observe(100.0);
        assert!((high_kl.scale() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn kl_lr_controller_ignores_non_finite_kl() {
        let mut controller = KlLrController::new(0.02, 20.0, 0.1, 10.0);
        controller.observe(f64::NAN);
        assert!((controller.ema() - 0.02).abs() < 1e-12);
        assert!((controller.scale() - 1.0).abs() < 1e-12);

        controller.observe(f64::INFINITY);
        assert!((controller.ema() - 0.02).abs() < 1e-12);
        assert!((controller.scale() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn kl_lr_controller_restore_clamps_and_ignores_invalid_state() {
        let mut controller = KlLrController::new(0.02, 20.0, 0.1, 10.0);
        controller.restore(0.05, 20.0);
        assert!((controller.ema() - 0.05).abs() < 1e-12);
        assert!((controller.scale() - 10.0).abs() < 1e-12);

        controller.restore(f64::NAN, f64::NEG_INFINITY);
        assert!((controller.ema() - 0.05).abs() < 1e-12);
        assert!((controller.scale() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn kl_lr_controller_state_roundtrips_and_validates() {
        let mut controller = KlLrController::new(0.02, 20.0, 0.1, 10.0);
        controller.restore(0.04, 2.5);
        let json = serde_json::to_string(&controller.state()).unwrap();

        let state = serde_json::from_str(&json).unwrap();
        let mut restored = KlLrController::new(0.02, 20.0, 0.1, 10.0);
        assert!(restored.restore_state(state));
        assert!((restored.ema() - 0.04).abs() < 1e-12);
        assert!((restored.scale() - 2.5).abs() < 1e-12);

        let invalid = serde_json::from_str(r#"{"version":2,"ema":0.04,"scale":2.5}"#).unwrap();
        assert!(!restored.restore_state(invalid));
        assert!((restored.ema() - 0.04).abs() < 1e-12);
        assert!((restored.scale() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn separate_actor_critic_clip_adds_shared_grads_after_independent_clips() {
        let device = Device::Cpu;
        let actor = Tensor::from_slice(&[1.0f32])
            .to_device(device)
            .set_requires_grad(true);
        let critic = Tensor::from_slice(&[1.0f32])
            .to_device(device)
            .set_requires_grad(true);
        let shared = Tensor::from_slice(&[1.0f32])
            .to_device(device)
            .set_requires_grad(true);
        let groups = GradClipGroups {
            actor: vec![actor.shallow_clone()],
            critic: vec![critic.shallow_clone()],
            shared: vec![shared.shallow_clone()],
        };
        let trainable = vec![
            actor.shallow_clone(),
            critic.shallow_clone(),
            shared.shallow_clone(),
        ];
        let actor_loss = &actor * 3.0 + &shared * 4.0;
        let critic_loss = &critic * 30.0 + &shared * 40.0;

        let (actor_norm, critic_norm) = backward_actor_critic_with_separate_clips(
            &groups,
            &trainable,
            &actor_loss.sum(Kind::Float),
            &critic_loss.sum(Kind::Float),
            1.0,
            device,
        );

        assert!((actor_norm.double_value(&[]) - 5.0).abs() < 1e-6);
        assert!((critic_norm.double_value(&[]) - 50.0).abs() < 1e-6);
        assert!((actor.grad().double_value(&[0]) - 0.6).abs() < 1e-6);
        assert!((critic.grad().double_value(&[0]) - 0.6).abs() < 1e-6);
        assert!((shared.grad().double_value(&[0]) - 1.6).abs() < 1e-6);
    }

    #[test]
    fn separate_actor_critic_clip_skips_unused_branch_grads() {
        let device = Device::Cpu;
        let actor = Tensor::from_slice(&[1.0f32])
            .to_device(device)
            .set_requires_grad(true);
        let critic = Tensor::from_slice(&[1.0f32])
            .to_device(device)
            .set_requires_grad(true);
        let actor_shared = Tensor::from_slice(&[1.0f32])
            .to_device(device)
            .set_requires_grad(true);
        let critic_shared = Tensor::from_slice(&[1.0f32])
            .to_device(device)
            .set_requires_grad(true);
        let groups = GradClipGroups {
            actor: vec![actor.shallow_clone()],
            critic: vec![critic.shallow_clone()],
            shared: vec![actor_shared.shallow_clone(), critic_shared.shallow_clone()],
        };
        let trainable = vec![
            actor.shallow_clone(),
            critic.shallow_clone(),
            actor_shared.shallow_clone(),
            critic_shared.shallow_clone(),
        ];
        let actor_loss = &actor * 3.0 + &actor_shared * 4.0;
        let critic_loss = &critic * 30.0 + &critic_shared * 40.0;

        let (actor_norm, critic_norm) = backward_actor_critic_with_separate_clips(
            &groups,
            &trainable,
            &actor_loss.sum(Kind::Float),
            &critic_loss.sum(Kind::Float),
            1.0,
            device,
        );

        assert!((actor_norm.double_value(&[]) - 5.0).abs() < 1e-6);
        assert!((critic_norm.double_value(&[]) - 50.0).abs() < 1e-6);
        assert!((actor.grad().double_value(&[0]) - 0.6).abs() < 1e-6);
        assert!((actor_shared.grad().double_value(&[0]) - 0.8).abs() < 1e-6);
        assert!((critic.grad().double_value(&[0]) - 0.6).abs() < 1e-6);
        assert!((critic_shared.grad().double_value(&[0]) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn real_model_grad_clip_groups_match_actor_critic_topology() {
        let vs = nn::VarStore::new(Device::Cpu);
        let _model = TradingModel::new_with_config(&vs.root(), TradingModelConfig::default());
        let named = named_trainable_variables(&vs);
        let groups = grad_clip_groups(&named);
        let mut by_id = HashMap::new();
        for tensor in &groups.actor {
            by_id.insert(tensor.data_ptr() as usize, "actor");
        }
        for tensor in &groups.critic {
            by_id.insert(tensor.data_ptr() as usize, "critic");
        }
        for tensor in &groups.shared {
            by_id.insert(tensor.data_ptr() as usize, "shared");
        }

        let group_for = |name: &str| {
            let tensor = named
                .iter()
                .find(|(param_name, _)| param_name == name)
                .unwrap_or_else(|| panic!("missing parameter {name}"));
            by_id[&(tensor.1.data_ptr() as usize)]
        };

        assert_eq!(group_for("policy_concentration.weight"), "actor");
        assert_eq!(group_for("value_proj.weight"), "critic");
        assert_eq!(group_for("patch_embed_weight"), "shared");
    }
}
