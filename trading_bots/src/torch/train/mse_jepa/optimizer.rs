use std::collections::HashSet;

use tch::Tensor;

use crate::torch::bar_dist::BAR_EMISSION_ADAMW_NAME_SUBSTRINGS;
use crate::torch::lejepa::model::FLOW_BLOCKS;
use crate::torch::optim::muon::{Muon, MuonConfig, Orthogonalizer};

pub(super) const RECIPE_ID: &str = "dbwm-normuon-pe5-wsd-v1";
pub(super) const MUON_BASE_LR: f64 = 5e-3;
pub(super) const ADAMW_BASE_LR: f64 = 3e-4;

const LR_FLOOR_MULTIPLIER: f64 = 0.15;
const MOMENTUM_START: f64 = 0.92;
const MOMENTUM_PEAK: f64 = 0.95;
const MOMENTUM_WARMUP_STEPS: usize = 50;
const ADAMW_BETAS: (f64, f64) = (0.9, 0.95);
const ADAMW_EPS: f64 = 1e-8;
const ADAMW_WEIGHT_DECAY: f64 = 1e-3;

/// Complete optimizer settings for one update in the planned DBWM run.
///
/// `completed_step` is the number of updates completed before this point is applied. All
/// moving values are published through Muon's scalar setters, which refresh the device-side
/// scalar pack before either an eager step or a CUDA-graph replay.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct OptimizerPoint {
    pub(super) training_pass: f64,
    pub(super) learning_rate_multiplier: f64,
    pub(super) muon_learning_rate: f64,
    pub(super) adamw_learning_rate: f64,
    pub(super) muon_momentum: f64,
}

impl OptimizerPoint {
    pub(super) fn apply(self, optimizer: &mut Muon) {
        optimizer.set_lr(self.muon_learning_rate);
        optimizer.set_adamw_lr(self.adamw_learning_rate);
        optimizer.set_momentum(self.muon_momentum);
    }
}

/// Pass-aware DBWM learning-rate and momentum schedule.
///
/// A run no longer than one pass remains at the base rates. Longer runs keep the first full
/// pass flat and reserve at most the final full pass for a linear decay. If the final pass is
/// truncated, only its available suffix decays. The last planned update therefore lands on
/// the floor exactly rather than one update above it.
#[derive(Clone, Copy, Debug)]
pub(super) struct DbwmSchedule {
    steps_per_pass: usize,
    planned_steps: usize,
    decay_start: Option<usize>,
    momentum_warmup_steps: usize,
}

impl DbwmSchedule {
    pub(super) fn new(steps_per_pass: usize, planned_steps: usize) -> Self {
        assert!(steps_per_pass > 0, "DBWM steps per pass must be nonzero");
        assert!(planned_steps > 0, "DBWM planned steps must be nonzero");

        let decay_start = (planned_steps > steps_per_pass)
            .then(|| steps_per_pass.max(planned_steps - steps_per_pass));
        let momentum_warmup_steps = MOMENTUM_WARMUP_STEPS
            .min(steps_per_pass)
            .min(planned_steps / 2);

        Self {
            steps_per_pass,
            planned_steps,
            decay_start,
            momentum_warmup_steps,
        }
    }

    pub(super) fn point(&self, completed_step: usize) -> OptimizerPoint {
        assert!(
            completed_step < self.planned_steps,
            "DBWM schedule step {completed_step} is outside the {} planned updates",
            self.planned_steps
        );

        let learning_rate_multiplier = self
            .decay_start
            .map(|decay_start| {
                if completed_step + 1 == self.planned_steps {
                    return LR_FLOOR_MULTIPLIER;
                }
                let decay_intervals = self.planned_steps - decay_start - 1;
                if completed_step <= decay_start {
                    1.0
                } else {
                    let elapsed = completed_step - decay_start;
                    let progress = elapsed as f64 / decay_intervals as f64;
                    1.0 + (LR_FLOOR_MULTIPLIER - 1.0) * progress
                }
            })
            .unwrap_or(1.0);

        let muon_momentum =
            if self.momentum_warmup_steps == 0 || completed_step >= self.momentum_warmup_steps {
                MOMENTUM_PEAK
            } else {
                let progress = completed_step as f64 / self.momentum_warmup_steps as f64;
                MOMENTUM_START + (MOMENTUM_PEAK - MOMENTUM_START) * progress
            };

        OptimizerPoint {
            training_pass: completed_step as f64 / self.steps_per_pass as f64,
            learning_rate_multiplier,
            muon_learning_rate: MUON_BASE_LR * learning_rate_multiplier,
            adamw_learning_rate: ADAMW_BASE_LR * learning_rate_multiplier,
            muon_momentum,
        }
    }
}

/// Flow-head parameters that must never see Muon: every zero-initialized modulation
/// projection, the zero-initialized output projection, and the time-embedding MLP.
const FLOW_ADAMW_NAME_SUBSTRINGS: [&str; 4] = [
    "lejepa_flow_time_",
    ".modulation",
    "lejepa_flow_final_modulation",
    "lejepa_flow_out_proj",
];

fn muon_allowlist() -> Vec<String> {
    let mut allowlist = vec![
        "bar_proj".to_owned(),
        "bar_enrich".to_owned(),
        "lejepa_projector_fc".to_owned(),
        "lejepa_layer_".to_owned(),
        "lejepa_flow_belief_proj".to_owned(),
        "lejepa_flow_in_proj".to_owned(),
    ];
    for index in 0..FLOW_BLOCKS {
        allowlist.push(format!("lejepa_flow_block_{index}.gate"));
        allowlist.push(format!("lejepa_flow_block_{index}.value"));
        allowlist.push(format!("lejepa_flow_block_{index}.out"));
    }
    allowlist
}

fn force_adamw_list() -> Vec<String> {
    FLOW_ADAMW_NAME_SUBSTRINGS
        .iter()
        .chain(["pope_theta_bias"].iter())
        .chain(BAR_EMISSION_ADAMW_NAME_SUBSTRINGS.iter())
        .map(|substring| (*substring).to_owned())
        .collect()
}

pub(super) fn build_optimizer(named: &[(String, Tensor)]) -> Muon {
    let allowlist = muon_allowlist();
    let force_adamw = force_adamw_list();
    let optimizer = Muon::new_named(
        named,
        MuonConfig {
            lr: MUON_BASE_LR,
            use_muon_for_2d: true,
            momentum: MOMENTUM_START,
            nesterov: true,
            beta2: 0.95,
            weight_decay: 0.0,
            adamw_lr: ADAMW_BASE_LR,
            adamw_betas: ADAMW_BETAS,
            adamw_eps: ADAMW_EPS,
            adamw_wd: ADAMW_WEIGHT_DECAY,
            adamw_no_weight_decay_name_substrings: vec![
                "pope_theta_bias".to_owned(),
                "bar_prefix_embed".to_owned(),
            ],
            force_adamw_name_substrings: force_adamw.clone(),
            muon_name_allowlist: allowlist.clone(),
            orthogonalizer: Orthogonalizer::PolarExpress5,
            adamw_every: 1,
            quadratic_lr_weight_decay: false,
            cautious_weight_decay: false,
            adamw_beta_overrides: Vec::new(),
            adamw_weight_decay_multipliers: Vec::new(),
            row_learned_lr: false,
            capture_step_graphs: true,
            ..MuonConfig::default()
        },
    );
    assert_exact_routing(named, &optimizer, &allowlist, &force_adamw);
    optimizer
}

fn assert_exact_routing(
    named: &[(String, Tensor)],
    optimizer: &Muon,
    muon_allowlist: &[String],
    force_adamw: &[String],
) {
    let muon_names = optimizer.muon_param_names();
    let adamw_names = optimizer.adamw_param_names();
    let muon: HashSet<&str> = muon_names.iter().map(String::as_str).collect();
    let adamw: HashSet<&str> = adamw_names.iter().map(String::as_str).collect();
    let registered: HashSet<&str> = named.iter().map(|(name, _)| name.as_str()).collect();

    assert_eq!(
        registered.len(),
        named.len(),
        "duplicate trainable parameter name"
    );
    assert_eq!(muon.len(), muon_names.len(), "duplicate Muon routing entry");
    assert_eq!(
        adamw.len(),
        adamw_names.len(),
        "duplicate AdamW routing entry"
    );
    assert!(muon.is_disjoint(&adamw), "Muon and AdamW routing overlap");
    assert_eq!(
        muon.len() + adamw.len(),
        registered.len(),
        "optimizer routing does not exactly partition the trainable parameters"
    );

    for (name, tensor) in named {
        let force_adamw = force_adamw.iter().any(|needle| name.contains(needle));
        let allow_muon = muon_allowlist.iter().any(|needle| name.contains(needle));
        let expected_muon = tensor.dim() == 2 && allow_muon && !force_adamw;
        assert_eq!(
            muon.contains(name.as_str()),
            expected_muon,
            "unexpected optimizer routing for {name}"
        );
        assert_eq!(
            adamw.contains(name.as_str()),
            !expected_muon,
            "{name} was not routed exactly once"
        );

        if !expected_muon {
            let (lr_scale, betas, wd_scale) = optimizer
                .adamw_group_settings(name)
                .unwrap_or_else(|| panic!("missing AdamW settings for {name}"));
            assert_eq!(lr_scale, 1.0, "unexpected AdamW LR scale for {name}");
            assert_eq!(betas, ADAMW_BETAS, "unexpected AdamW betas for {name}");
            let no_decay = name.contains("pope_theta_bias") || name.contains("bar_prefix_embed");
            assert_eq!(
                wd_scale,
                if no_decay { 0.0 } else { 1.0 },
                "unexpected AdamW weight-decay scale for {name}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_optimizer, DbwmSchedule, ADAMW_BASE_LR, FLOW_ADAMW_NAME_SUBSTRINGS,
        LR_FLOOR_MULTIPLIER, MOMENTUM_PEAK, MOMENTUM_START, MUON_BASE_LR,
    };
    use crate::torch::lejepa::model::FLOW_BLOCKS;
    use crate::torch::lejepa::MseJepaModel;
    use crate::torch::test_rng;
    use crate::torch::train::optimizer_glue::named_trainable_variables;
    use tch::{nn, Device};

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn one_pass_and_shorter_runs_keep_the_base_learning_rates() {
        for (steps_per_pass, planned_steps) in [(100, 100), (100, 37), (1, 1)] {
            let schedule = DbwmSchedule::new(steps_per_pass, planned_steps);
            for step in 0..planned_steps {
                let point = schedule.point(step);
                assert_eq!(point.learning_rate_multiplier, 1.0);
                assert_eq!(point.muon_learning_rate, MUON_BASE_LR);
                assert_eq!(point.adamw_learning_rate, ADAMW_BASE_LR);
                assert_close(point.training_pass, step as f64 / steps_per_pass as f64);
            }
        }
    }

    #[test]
    fn final_pass_decay_has_exact_boundaries_and_endpoint() {
        let schedule = DbwmSchedule::new(100, 300);
        assert_eq!(schedule.point(199).learning_rate_multiplier, 1.0);
        assert_eq!(schedule.point(200).learning_rate_multiplier, 1.0);
        let first_decay = schedule.point(201).learning_rate_multiplier;
        assert!(first_decay < 1.0 && first_decay > LR_FLOOR_MULTIPLIER);
        assert_eq!(
            schedule.point(299).learning_rate_multiplier,
            LR_FLOOR_MULTIPLIER
        );
        assert_close(
            schedule.point(299).muon_learning_rate,
            MUON_BASE_LR * LR_FLOOR_MULTIPLIER,
        );
        assert_close(
            schedule.point(299).adamw_learning_rate,
            ADAMW_BASE_LR * LR_FLOOR_MULTIPLIER,
        );
    }

    #[test]
    fn truncated_additional_pass_decays_only_after_the_first_complete_pass() {
        let schedule = DbwmSchedule::new(100, 137);
        assert_eq!(schedule.point(99).learning_rate_multiplier, 1.0);
        assert_eq!(schedule.point(100).learning_rate_multiplier, 1.0);
        assert!(schedule.point(101).learning_rate_multiplier < 1.0);
        assert_eq!(
            schedule.point(136).learning_rate_multiplier,
            LR_FLOOR_MULTIPLIER
        );

        let one_update_tail = DbwmSchedule::new(8, 9);
        assert_eq!(one_update_tail.point(7).learning_rate_multiplier, 1.0);
        assert_eq!(
            one_update_tail.point(8).learning_rate_multiplier,
            LR_FLOOR_MULTIPLIER
        );
    }

    #[test]
    fn momentum_warmup_is_bounded_by_pass_and_run() {
        let ordinary = DbwmSchedule::new(100, 300);
        assert_eq!(ordinary.point(0).muon_momentum, MOMENTUM_START);
        assert_close(ordinary.point(25).muon_momentum, 0.935);
        assert_eq!(ordinary.point(50).muon_momentum, MOMENTUM_PEAK);
        assert_eq!(ordinary.point(299).muon_momentum, MOMENTUM_PEAK);

        let run_bounded = DbwmSchedule::new(100, 20);
        assert_eq!(run_bounded.point(0).muon_momentum, MOMENTUM_START);
        assert_eq!(run_bounded.point(10).muon_momentum, MOMENTUM_PEAK);

        let pass_bounded = DbwmSchedule::new(4, 20);
        assert_eq!(pass_bounded.point(0).muon_momentum, MOMENTUM_START);
        assert_eq!(pass_bounded.point(4).muon_momentum, MOMENTUM_PEAK);
    }

    #[test]
    #[should_panic(expected = "steps per pass must be nonzero")]
    fn schedule_rejects_zero_steps_per_pass() {
        let _ = DbwmSchedule::new(0, 1);
    }

    #[test]
    #[should_panic(expected = "planned steps must be nonzero")]
    fn schedule_rejects_zero_planned_steps() {
        let _ = DbwmSchedule::new(1, 0);
    }

    #[test]
    fn real_model_routing_is_exact_and_zero_init_flow_paths_stay_on_adamw() {
        let _torch_rng_guard = test_rng::shared();
        let var_store = nn::VarStore::new(Device::Cpu);
        let _model = MseJepaModel::new(&var_store.root());
        let named = named_trainable_variables(&var_store);
        let optimizer = build_optimizer(&named);
        let muon = optimizer.muon_param_names();
        let adamw = optimizer.adamw_param_names();

        assert_eq!(muon.len() + adamw.len(), named.len());
        for (name, _) in &named {
            assert_ne!(
                muon.contains(name),
                adamw.contains(name),
                "{name} must be routed exactly once"
            );
            if FLOW_ADAMW_NAME_SUBSTRINGS
                .iter()
                .any(|substring| name.contains(substring))
            {
                assert!(adamw.contains(name), "{name} must train under AdamW");
                assert!(!muon.contains(name), "{name} leaked onto Muon");
            }
        }

        for name in [
            "lejepa_flow_belief_proj.weight".to_owned(),
            "lejepa_flow_in_proj.weight".to_owned(),
        ] {
            assert!(muon.contains(&name), "{name} must train under Muon");
        }
        for block in 0..FLOW_BLOCKS {
            for projection in ["gate", "value", "out"] {
                let name = format!("lejepa_flow_block_{block}.{projection}.weight");
                assert!(muon.contains(&name), "{name} must train under Muon");
            }
            let name = format!("lejepa_flow_block_{block}.modulation.weight");
            assert!(adamw.contains(&name), "{name} must train under AdamW");
        }
    }

    #[test]
    fn every_schedule_point_can_be_applied_without_adam_cadence_state() {
        let _torch_rng_guard = test_rng::shared();
        let var_store = nn::VarStore::new(Device::Cpu);
        let _model = MseJepaModel::new(&var_store.root());
        let named = named_trainable_variables(&var_store);
        let mut optimizer = build_optimizer(&named);
        let schedule = DbwmSchedule::new(4, 12);

        for completed_step in 0..12 {
            schedule.point(completed_step).apply(&mut optimizer);
            assert!(!optimizer.adamw_accumulation_pending());
        }
    }
}
