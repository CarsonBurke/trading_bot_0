use std::collections::{BTreeMap, HashSet};

use anyhow::{ensure, Result};
use tch::{nn, Device, Kind, Tensor};

use crate::torch::lejepa::checkpoint::{eligible_name_sha256, WeightReadout};
use crate::torch::lejepa::MseJepaModel;
use crate::torch::train::optimizer_glue::named_trainable_variables;

const TRACK_STEPS: usize = 2_900;
const START_STEP_NUMERATOR: usize = 2_400;
const HORIZON_NUMERATOR: usize = 150;
const BLEND: f64 = 0.6;
const EXCLUDED_NAME_SUBSTRING: &str = "bar_prefix_embed";

struct LiveParameter {
    name: String,
    tensor: Tensor,
    eligible: bool,
}

/// Late fp32 parameter EMA used only to materialize the predeclared evaluation readout.
///
/// The timing preserves modded-nanogpt Track 3's ratios: the shadow starts at 2400/2900 of
/// the planned run, has a 150/2900-step horizon, and is blended 60% into the live endpoint.
/// It never enters the training forward pass or optimizer state.
pub(super) struct TailEma {
    start_step: usize,
    planned_steps: usize,
    horizon_steps: usize,
    updates: usize,
    eligible_name_sha256: String,
    live: Vec<LiveParameter>,
    shadows: Option<Vec<Tensor>>,
}

impl TailEma {
    pub(super) fn new(named: &[(String, Tensor)], planned_steps: usize) -> Result<Self> {
        ensure!(planned_steps > 0, "tail EMA needs a nonzero planned run");
        let start_step =
            ((planned_steps as u128 * START_STEP_NUMERATOR as u128) / TRACK_STEPS as u128) as usize;
        let start_step = start_step.max(1);
        let horizon_steps = ((planned_steps as u128 * HORIZON_NUMERATOR as u128
            + (TRACK_STEPS / 2) as u128)
            / TRACK_STEPS as u128) as usize;
        let horizon_steps = horizon_steps.max(1);
        let live = named
            .iter()
            .map(|(name, tensor)| LiveParameter {
                name: name.clone(),
                tensor: tensor.shallow_clone(),
                eligible: !name.contains(EXCLUDED_NAME_SUBSTRING),
            })
            .collect::<Vec<_>>();
        ensure!(
            !live.is_empty(),
            "tail EMA received no trainable parameters"
        );
        ensure!(
            live.iter()
                .all(|parameter| parameter.tensor.kind() == Kind::Float),
            "tail EMA requires fp32 master parameters"
        );
        let eligible_names = live
            .iter()
            .filter(|parameter| parameter.eligible)
            .map(|parameter| parameter.name.as_str())
            .collect::<Vec<_>>();
        let eligible_name_sha256 = eligible_name_sha256(eligible_names)?;

        Ok(Self {
            start_step,
            planned_steps,
            horizon_steps,
            updates: 0,
            eligible_name_sha256,
            live,
            shadows: None,
        })
    }

    /// Observe post-update weights through the planned endpoint. Including the live endpoint
    /// gives the final blended readout the same completed-step ledger carried in metadata.
    pub(super) fn update(&mut self, completed_step: usize) {
        if completed_step < self.start_step || completed_step > self.planned_steps {
            return;
        }
        tch::no_grad(|| {
            if let Some(shadows) = self.shadows.as_mut() {
                let alpha = 1.0 / self.horizon_steps as f64;
                for (shadow, parameter) in shadows
                    .iter_mut()
                    .zip(self.live.iter().filter(|parameter| parameter.eligible))
                {
                    let _ = shadow.lerp_(&parameter.tensor, alpha);
                }
            } else {
                self.shadows = Some(
                    self.live
                        .iter()
                        .filter(|parameter| parameter.eligible)
                        .map(|parameter| parameter.tensor.detach().copy())
                        .collect(),
                );
            }
        });
        self.updates += 1;
    }

    pub(super) fn is_initialized(&self) -> bool {
        self.shadows.is_some()
    }

    pub(super) fn readout(&self) -> Result<WeightReadout> {
        ensure!(self.is_initialized(), "tail EMA was never initialized");
        Ok(WeightReadout::TailEma {
            start_step: self.start_step as u64,
            horizon_steps: self.horizon_steps as u64,
            blend: BLEND,
            update_count: self.updates as u64,
            eligible_name_sha256: self.eligible_name_sha256.clone(),
        })
    }

    /// Build an isolated model containing the blended endpoint without mutating live training
    /// weights. This is called only after the final raw validation, so construction cannot alter
    /// any subsequent training randomness.
    pub(super) fn materialize(&self, device: Device) -> Result<(nn::VarStore, MseJepaModel)> {
        let shadows = self
            .shadows
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("tail EMA was never initialized"))?;
        let shadow_by_name = self
            .live
            .iter()
            .filter(|parameter| parameter.eligible)
            .zip(shadows)
            .map(|(parameter, shadow)| (parameter.name.as_str(), shadow))
            .collect::<BTreeMap<_, _>>();
        let live_by_name = self
            .live
            .iter()
            .map(|parameter| (parameter.name.as_str(), &parameter.tensor))
            .collect::<BTreeMap<_, _>>();

        let var_store = nn::VarStore::new(device);
        let model = MseJepaModel::new(&var_store.root());
        let target = named_trainable_variables(&var_store);
        let target_names = target
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<HashSet<_>>();
        ensure!(
            target_names.len() == live_by_name.len()
                && live_by_name.keys().all(|name| target_names.contains(name)),
            "tail EMA model parameter contract drifted"
        );

        tch::no_grad(|| {
            for (name, mut destination) in target {
                let live = live_by_name[&name.as_str()];
                if let Some(shadow) = shadow_by_name.get(name.as_str()) {
                    destination.copy_(&(live * (1.0 - BLEND) + *shadow * BLEND));
                } else {
                    destination.copy_(live);
                }
            }
        });
        Ok((var_store, model))
    }
}

#[cfg(test)]
mod tests {
    use super::{TailEma, BLEND};
    use tch::{Device, Kind, Tensor};

    #[test]
    fn track_three_timing_scales_with_the_planned_run() {
        let named = vec![(
            "weight".to_owned(),
            Tensor::zeros([1], (Kind::Float, Device::Cpu)),
        )];
        let ema = TailEma::new(&named, 2_900).unwrap();
        assert_eq!(ema.start_step, 2_400);
        assert_eq!(ema.planned_steps, 2_900);
        assert_eq!(ema.horizon_steps, 150);
        assert_eq!(BLEND, 0.6);
    }

    #[test]
    fn update_is_post_step_unbiased_and_excludes_prefix_embeddings() {
        let weight = Tensor::from_slice(&[2.0f32]);
        let prefix = Tensor::from_slice(&[7.0f32]);
        let named = vec![
            ("weight".to_owned(), weight.shallow_clone()),
            ("bar_prefix_embed".to_owned(), prefix),
        ];
        let mut ema = TailEma::new(&named, 2_900).unwrap();
        ema.update(2_399);
        assert!(!ema.is_initialized());
        ema.update(2_400);
        assert_eq!(ema.updates, 1);
        weight.shallow_clone().copy_(&Tensor::from_slice(&[5.0f32]));
        ema.update(2_401);
        let expected = 2.0 + (5.0 - 2.0) / 150.0;
        let shadows = ema.shadows.as_ref().unwrap();
        assert_eq!(shadows.len(), 1);
        assert!((shadows[0].double_value(&[]) - expected).abs() < 1e-6);
        ema.update(2_901);
        assert_eq!(ema.updates, 2);
    }
}
